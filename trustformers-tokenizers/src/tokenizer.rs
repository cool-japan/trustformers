use crate::alignment::{AlignedSpan, AlignmentConfig, AlignmentEngine, TokenAlignment};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
// SciRS2 Integration Policy: Use re-exported tokenizers types from trustformers_core.
// `TruncationParams` / the normalizer types are not re-exported there, so they come
// from the same `tokenizers` crate version this workspace already depends on.
use tokenizers::normalizers::{Lowercase, NormalizerWrapper, Sequence as NormalizerSequence};
use tokenizers::TruncationParams;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};
use trustformers_core::{Encoding, Tokenizer as HFTokenizer, TokenizerError};

#[derive(Debug, Clone)]
pub struct TokenizedInputWithOffsets {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub token_type_ids: Option<Vec<u32>>,
    pub offset_mapping: Option<Vec<(usize, usize)>>,
    pub special_tokens_mask: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct TokenizedInputWithAlignment {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub token_type_ids: Option<Vec<u32>>,
    pub offset_mapping: Option<Vec<(usize, usize)>>,
    pub special_tokens_mask: Option<Vec<u8>>,
    pub word_alignments: Vec<TokenAlignment>,
    pub words: Vec<crate::alignment::Word>,
}

impl From<TokenizedInputWithOffsets> for TokenizedInput {
    fn from(input: TokenizedInputWithOffsets) -> Self {
        TokenizedInput {
            input_ids: input.input_ids,
            attention_mask: input.attention_mask,
            token_type_ids: input.token_type_ids,
            special_tokens_mask: input.special_tokens_mask,
            offset_mapping: input.offset_mapping,
            overflowing_tokens: None,
        }
    }
}

impl From<TokenizedInputWithAlignment> for TokenizedInput {
    fn from(input: TokenizedInputWithAlignment) -> Self {
        TokenizedInput {
            input_ids: input.input_ids,
            attention_mask: input.attention_mask,
            token_type_ids: input.token_type_ids,
            special_tokens_mask: input.special_tokens_mask,
            offset_mapping: input.offset_mapping,
            overflowing_tokens: None,
        }
    }
}

impl From<TokenizedInputWithAlignment> for TokenizedInputWithOffsets {
    fn from(input: TokenizedInputWithAlignment) -> Self {
        TokenizedInputWithOffsets {
            input_ids: input.input_ids,
            attention_mask: input.attention_mask,
            token_type_ids: input.token_type_ids,
            offset_mapping: input.offset_mapping,
            special_tokens_mask: input.special_tokens_mask,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenizerImpl {
    tokenizer: Arc<HFTokenizer>,
    do_lower_case: bool,
    max_length: Option<usize>,
    alignment_engine: Option<AlignmentEngine>,
}

impl TokenizerImpl {
    pub fn from_file(path: &Path) -> Result<Self> {
        let tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))?;
        Ok(Self::from_hf_tokenizer(tokenizer))
    }

    /// Wrap a loaded HuggingFace tokenizer.
    ///
    /// `max_length` mirrors the truncation the tokenizer actually carries (from
    /// its `tokenizer.json`), so the getter never advertises a limit that is not
    /// enforced.
    fn from_hf_tokenizer(tokenizer: HFTokenizer) -> Self {
        let max_length = tokenizer.get_truncation().map(|params| params.max_length);
        Self {
            tokenizer: Arc::new(tokenizer),
            do_lower_case: false,
            max_length,
            alignment_engine: None,
        }
    }

    pub fn from_pretrained(name: &str) -> Result<Self> {
        Self::from_pretrained_with_revision(name, None)
    }

    /// Load a tokenizer from a local path or the `huggingface_hub` cache.
    ///
    /// Nothing is downloaded. The following locations are probed, in order:
    ///
    /// * `name` itself, when it is a `tokenizer.json` file
    /// * `{name}/tokenizer.json`
    /// * `{hub}/models--{org}--{model}/refs/{revision}` -> `snapshots/{sha}/tokenizer.json`
    /// * every `{hub}/models--{org}--{model}/snapshots/*/tokenizer.json`
    ///
    /// where `{hub}` is `$HF_HUB_CACHE`, `$HF_HOME/hub`, `$TRANSFORMERS_CACHE`
    /// or `$HOME/.cache/huggingface/hub`, and `{revision}` defaults to `main`.
    pub fn from_pretrained_with_revision(name: &str, revision: Option<&str>) -> Result<Self> {
        let mut probed: Vec<String> = Vec::new();

        let direct = Path::new(name);
        if direct.is_file() {
            return Self::from_file(direct);
        }

        let direct_json = direct.join("tokenizer.json");
        probed.push(direct_json.display().to_string());
        if direct_json.is_file() {
            return Self::from_file(&direct_json);
        }

        let repo_dir_name = format!("models--{}", name.replace('/', "--"));
        let revision = revision.unwrap_or("main");

        for hub_root in Self::hub_cache_roots() {
            let repo_dir = hub_root.join(&repo_dir_name);

            // refs/{revision} holds the commit sha; the files live under snapshots/{sha}.
            let ref_file = repo_dir.join("refs").join(revision);
            probed.push(ref_file.display().to_string());
            if let Ok(sha) = std::fs::read_to_string(&ref_file) {
                let snapshot = repo_dir.join("snapshots").join(sha.trim()).join("tokenizer.json");
                probed.push(snapshot.display().to_string());
                if snapshot.is_file() {
                    return Self::from_file(&snapshot);
                }
            }

            let snapshots = repo_dir.join("snapshots");
            if let Ok(entries) = std::fs::read_dir(&snapshots) {
                for entry in entries.flatten() {
                    let candidate = entry.path().join("tokenizer.json");
                    probed.push(candidate.display().to_string());
                    if candidate.is_file() {
                        return Self::from_file(&candidate);
                    }
                }
            }
        }

        Err(TrustformersError::invalid_input(format!(
            "Tokenizer '{}' (revision '{}') was not found locally. Download it first \
             (this crate does not fetch from the network). Probed paths: {}.",
            name,
            revision,
            probed.join(", ")
        )))
    }

    /// Candidate `huggingface_hub` cache roots, in priority order.
    fn hub_cache_roots() -> Vec<PathBuf> {
        let mut roots = Vec::new();

        if let Ok(dir) = std::env::var("HF_HUB_CACHE") {
            roots.push(PathBuf::from(dir));
        }
        if let Ok(dir) = std::env::var("HF_HOME") {
            roots.push(Path::new(&dir).join("hub"));
        }
        if let Ok(dir) = std::env::var("TRANSFORMERS_CACHE") {
            roots.push(PathBuf::from(dir));
        }
        if let Ok(home) = std::env::var("HOME") {
            roots.push(Path::new(&home).join(".cache").join("huggingface").join("hub"));
        }

        roots
    }

    pub fn from_tokenizer_json(json_str: &str) -> Result<Self> {
        let tokenizer = HFTokenizer::from_str(json_str).map_err(|e: TokenizerError| {
            TrustformersError::other(anyhow::anyhow!(e).to_string())
        })?;
        Ok(Self::from_hf_tokenizer(tokenizer))
    }

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let json = self
            .tokenizer
            .to_string(false)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))?;
        std::fs::write(path, json)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))?;
        Ok(())
    }

    pub fn to_json(&self) -> Result<String> {
        self.tokenizer
            .to_string(false)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))
    }

    /// Apply lowercasing and/or truncation to the underlying tokenizer.
    ///
    /// Both settings take real effect: `max_length` installs
    /// `TruncationParams` on the wrapped tokenizer, and `do_lower_case`
    /// *appends* a `Lowercase` normalizer to whatever normalizer the tokenizer
    /// already has (the existing one is preserved, never replaced).
    pub fn with_config(mut self, do_lower_case: bool, max_length: Option<usize>) -> Result<Self> {
        {
            let tokenizer = Arc::make_mut(&mut self.tokenizer);

            let truncation = max_length.map(|max_length| TruncationParams {
                max_length,
                ..Default::default()
            });
            tokenizer.with_truncation(truncation).map_err(|e| {
                TrustformersError::invalid_config(format!("Failed to configure truncation: {}", e))
            })?;

            if do_lower_case {
                let lowercase = NormalizerWrapper::Lowercase(Lowercase);
                let composed = match tokenizer.get_normalizer().cloned() {
                    Some(NormalizerWrapper::Sequence(existing)) => {
                        let mut parts: Vec<NormalizerWrapper> = existing.into_iter().collect();
                        parts.push(lowercase);
                        NormalizerWrapper::Sequence(NormalizerSequence::new(parts))
                    },
                    Some(existing) => NormalizerWrapper::Sequence(NormalizerSequence::new(vec![
                        existing, lowercase,
                    ])),
                    None => lowercase,
                };
                tokenizer.with_normalizer(Some(composed)).map_err(|e| {
                    TrustformersError::invalid_config(format!(
                        "Failed to install the lowercase normalizer: {}",
                        e
                    ))
                })?;
            }
        }

        self.do_lower_case = do_lower_case;
        self.max_length = max_length;
        Ok(self)
    }

    /// Whether [`Self::with_config`] installed a lowercase normalizer.
    ///
    /// A `Lowercase` normalizer that came from the loaded `tokenizer.json` is
    /// not reported here; this reflects the configuration applied through this
    /// wrapper.
    pub fn does_lower_case(&self) -> bool {
        self.do_lower_case
    }

    /// The truncation length actually installed on the tokenizer, if any.
    pub fn max_length(&self) -> Option<usize> {
        self.max_length
    }

    pub fn encode_with_offsets(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<TokenizedInputWithOffsets> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))?;
        Ok(self.encoding_to_tokenized_input_with_offsets(encoding))
    }

    pub fn encode_pair_with_offsets(
        &self,
        text: &str,
        text2: &str,
        add_special_tokens: bool,
    ) -> Result<TokenizedInputWithOffsets> {
        let encoding = self
            .tokenizer
            .encode((text, text2), add_special_tokens)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))?;
        Ok(self.encoding_to_tokenized_input_with_offsets(encoding))
    }

    pub fn decode_with_special_tokens(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))
    }

    pub fn get_vocab(&self) -> HashMap<String, u32> {
        self.tokenizer.get_vocab(false)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    /// Configure word alignment engine
    pub fn with_alignment_config(mut self, config: AlignmentConfig) -> Self {
        self.alignment_engine = Some(AlignmentEngine::new(config));
        self
    }

    /// Enable word alignment with default configuration
    pub fn with_word_alignment(mut self) -> Self {
        self.alignment_engine = Some(AlignmentEngine::new(AlignmentConfig::default()));
        self
    }

    /// Get mutable reference to alignment engine
    pub fn alignment_engine_mut(&mut self) -> Option<&mut AlignmentEngine> {
        self.alignment_engine.as_mut()
    }

    /// Encode text with word alignment
    pub fn encode_with_alignment(
        &mut self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<TokenizedInputWithAlignment> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))?;

        self.encoding_to_tokenized_input_with_alignment(text, encoding, None)
    }

    /// Encode text pair with word alignment
    pub fn encode_pair_with_alignment(
        &mut self,
        text: &str,
        text2: &str,
        add_special_tokens: bool,
    ) -> Result<TokenizedInputWithAlignment> {
        let encoding = self
            .tokenizer
            .encode((text, text2), add_special_tokens)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))?;

        // The encoding's offsets for the second sequence are relative to `text2`.
        // Joining the two texts with a single space shifts every second-sequence
        // offset by exactly `text.len() + 1`, so the shifted offsets index the
        // combined string exactly (no approximation).
        let combined_text = format!("{} {}", text, text2);
        let shift = text.len() + 1;
        self.encoding_to_tokenized_input_with_alignment(&combined_text, encoding, Some(shift))
    }

    /// Extract spans with word alignment
    pub fn extract_aligned_spans(
        &mut self,
        text: &str,
        spans: &[(usize, usize)],
        add_special_tokens: bool,
    ) -> Result<Vec<AlignedSpan>> {
        let input_with_alignment = self.encode_with_alignment(text, add_special_tokens)?;

        if let Some(engine) = &mut self.alignment_engine {
            engine.extract_spans(text, &input_with_alignment.word_alignments, spans)
        } else {
            Err(TrustformersError::other(
                "Word alignment engine not configured".to_string(),
            ))
        }
    }

    /// Preserve entity boundaries in tokenization
    pub fn preserve_entities(
        &mut self,
        text: &str,
        entities: &[(usize, usize, String)],
        add_special_tokens: bool,
    ) -> Result<Vec<AlignedSpan>> {
        let input_with_alignment = self.encode_with_alignment(text, add_special_tokens)?;

        if let Some(engine) = &mut self.alignment_engine {
            engine.preserve_entities(text, &input_with_alignment.word_alignments, entities)
        } else {
            Err(TrustformersError::other(
                "Word alignment engine not configured".to_string(),
            ))
        }
    }

    /// Get word boundaries for a specific token
    pub fn get_word_boundaries_for_token(
        &self,
        alignments: &[TokenAlignment],
        token_index: usize,
    ) -> Option<(usize, usize)> {
        if let Some(engine) = &self.alignment_engine {
            engine.get_word_boundaries_for_token(alignments, token_index)
        } else {
            None
        }
    }

    /// Check if tokens form a complete word
    pub fn tokens_form_complete_word(
        &self,
        alignments: &[TokenAlignment],
        token_indices: &[usize],
    ) -> bool {
        if let Some(engine) = &self.alignment_engine {
            engine.tokens_form_complete_word(alignments, token_indices)
        } else {
            false
        }
    }

    /// Offsets of an encoding, with second-sequence offsets shifted into a
    /// combined coordinate space when `second_sequence_shift` is given.
    fn offsets_of(
        encoding: &Encoding,
        second_sequence_shift: Option<usize>,
    ) -> Option<Vec<(usize, usize)>> {
        let offsets = encoding.get_offsets();
        if offsets.is_empty() {
            return None;
        }

        let Some(shift) = second_sequence_shift else {
            return Some(offsets.to_vec());
        };

        let sequence_ids = encoding.get_sequence_ids();
        Some(
            offsets
                .iter()
                .enumerate()
                .map(|(index, &(start, end))| {
                    if sequence_ids.get(index).copied().flatten() == Some(1) {
                        (start + shift, end + shift)
                    } else {
                        (start, end)
                    }
                })
                .collect(),
        )
    }

    fn special_tokens_mask_of(encoding: &Encoding) -> Option<Vec<u8>> {
        let mask = encoding.get_special_tokens_mask();
        if mask.is_empty() {
            None
        } else {
            Some(mask.iter().map(|&x| x as u8).collect())
        }
    }

    /// Convert an `Encoding`, keeping the offsets and special-tokens mask it
    /// already carries (they are computed by the tokenizer for free).
    fn encoding_to_tokenized_input(&self, encoding: Encoding) -> TokenizedInput {
        TokenizedInput {
            input_ids: encoding.get_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().iter().map(|&x| x as u8).collect(),
            token_type_ids: if encoding.get_type_ids().is_empty() {
                None
            } else {
                Some(encoding.get_type_ids().to_vec())
            },
            special_tokens_mask: Self::special_tokens_mask_of(&encoding),
            offset_mapping: Self::offsets_of(&encoding, None),
            overflowing_tokens: None,
        }
    }

    fn encoding_to_tokenized_input_with_offsets(
        &self,
        encoding: Encoding,
    ) -> TokenizedInputWithOffsets {
        let offset_mapping = Self::offsets_of(&encoding, None);
        let special_tokens_mask = Self::special_tokens_mask_of(&encoding);

        TokenizedInputWithOffsets {
            input_ids: encoding.get_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().iter().map(|&x| x as u8).collect(),
            token_type_ids: if encoding.get_type_ids().is_empty() {
                None
            } else {
                Some(encoding.get_type_ids().to_vec())
            },
            offset_mapping,
            special_tokens_mask,
        }
    }

    /// Build an aligned encoding against `text`.
    ///
    /// For sequence pairs, `second_sequence_shift` moves the second sequence's
    /// offsets into the coordinate space of the combined `text`, so alignment
    /// never runs against a string the offsets do not describe.
    fn encoding_to_tokenized_input_with_alignment(
        &mut self,
        text: &str,
        encoding: Encoding,
        second_sequence_shift: Option<usize>,
    ) -> Result<TokenizedInputWithAlignment> {
        let offset_mapping = Self::offsets_of(&encoding, second_sequence_shift);
        let special_tokens_mask = Self::special_tokens_mask_of(&encoding);

        // Perform word alignment if engine is available
        let (word_alignments, words) = if let Some(engine) = &mut self.alignment_engine {
            if let Some(ref offsets) = offset_mapping {
                let alignments =
                    engine.align_tokens_to_words(text, offsets, special_tokens_mask.as_deref())?;
                let words = engine.extract_words(text);
                (alignments, words)
            } else {
                // If no offsets available, create empty alignments
                (Vec::new(), Vec::new())
            }
        } else {
            return Err(TrustformersError::other(
                "Word alignment engine not configured".to_string(),
            ));
        };

        Ok(TokenizedInputWithAlignment {
            input_ids: encoding.get_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().iter().map(|&x| x as u8).collect(),
            token_type_ids: if encoding.get_type_ids().is_empty() {
                None
            } else {
                Some(encoding.get_type_ids().to_vec())
            },
            offset_mapping,
            special_tokens_mask,
            word_alignments,
            words,
        })
    }
}

impl Tokenizer for TokenizerImpl {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let encoding = self.tokenizer.encode(text, false).map_err(|e| {
            trustformers_core::errors::TrustformersError::other(anyhow::anyhow!(e).to_string())
        })?;
        Ok(self.encoding_to_tokenized_input(encoding))
    }

    fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        let encoding = self.tokenizer.encode((text, text2), false).map_err(|e| {
            trustformers_core::errors::TrustformersError::other(anyhow::anyhow!(e).to_string())
        })?;
        Ok(self.encoding_to_tokenized_input(encoding))
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, false).map_err(|e| {
            trustformers_core::errors::TrustformersError::other(anyhow::anyhow!(e).to_string())
        })
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.tokenizer.get_vocab(false)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }
}

#[derive(Debug, Clone)]
pub enum TokenizerWrapper {
    WordPiece(crate::wordpiece::WordPieceTokenizer),
    BPE(crate::bpe::BPETokenizer),
    Unigram(crate::unigram::UnigramTokenizer),
    Char(crate::char::CharTokenizer),
    HuggingFace(TokenizerImpl),
}

impl Tokenizer for TokenizerWrapper {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        match self {
            TokenizerWrapper::WordPiece(t) => t.encode(text),
            TokenizerWrapper::BPE(t) => t.encode(text),
            TokenizerWrapper::Unigram(t) => t.encode(text),
            TokenizerWrapper::Char(t) => t.encode(text),
            TokenizerWrapper::HuggingFace(t) => t.encode(text),
        }
    }

    fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        match self {
            TokenizerWrapper::WordPiece(t) => t.encode_pair(text, text2),
            TokenizerWrapper::BPE(t) => t.encode_pair(text, text2),
            TokenizerWrapper::Unigram(t) => t.encode_pair(text, text2),
            TokenizerWrapper::Char(t) => t.encode_pair(text, text2),
            TokenizerWrapper::HuggingFace(t) => t.encode_pair(text, text2),
        }
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        match self {
            TokenizerWrapper::WordPiece(t) => t.decode(ids),
            TokenizerWrapper::BPE(t) => t.decode(ids),
            TokenizerWrapper::Unigram(t) => t.decode(ids),
            TokenizerWrapper::Char(t) => t.decode(ids),
            TokenizerWrapper::HuggingFace(t) => t.decode(ids),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            TokenizerWrapper::WordPiece(t) => t.vocab_size(),
            TokenizerWrapper::BPE(t) => t.vocab_size(),
            TokenizerWrapper::Unigram(t) => t.vocab_size(),
            TokenizerWrapper::Char(t) => t.vocab_size(),
            TokenizerWrapper::HuggingFace(t) => t.vocab_size(),
        }
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        match self {
            TokenizerWrapper::WordPiece(t) => t.get_vocab(),
            TokenizerWrapper::BPE(t) => t.get_vocab(),
            TokenizerWrapper::Unigram(t) => t.get_vocab(),
            TokenizerWrapper::Char(t) => t.get_vocab(),
            TokenizerWrapper::HuggingFace(t) => t.get_vocab(),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            TokenizerWrapper::WordPiece(t) => t.token_to_id(token),
            TokenizerWrapper::BPE(t) => t.token_to_id(token),
            TokenizerWrapper::Unigram(t) => t.token_to_id(token),
            TokenizerWrapper::Char(t) => t.token_to_id(token),
            TokenizerWrapper::HuggingFace(t) => t.token_to_id(token),
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match self {
            TokenizerWrapper::WordPiece(t) => t.id_to_token(id),
            TokenizerWrapper::BPE(t) => t.id_to_token(id),
            TokenizerWrapper::Unigram(t) => t.id_to_token(id),
            TokenizerWrapper::Char(t) => t.id_to_token(id),
            TokenizerWrapper::HuggingFace(t) => t.id_to_token(id),
        }
    }
}

/// Layout version written by [`TokenizerWrapper::save_pretrained`].
///
/// Version `1.0` directories contained a type marker only (no vocabulary) and
/// therefore cannot be restored; loading one is an error rather than a silently
/// empty tokenizer.
const TOKENIZER_STATE_VERSION: &str = "2.0";

const CONFIG_FILE: &str = "tokenizer_config.json";
const VOCAB_FILE: &str = "vocab.json";
const MERGES_FILE: &str = "merges.txt";
const SCORES_FILE: &str = "scores.json";

impl TokenizerWrapper {
    /// Load a tokenizer from a directory previously written by
    /// [`TokenizerWrapper::save_pretrained`], or from a HuggingFace
    /// `tokenizer.json`.
    ///
    /// Returns an error naming every probed path when nothing loads; it never
    /// falls back to an empty tokenizer.
    pub fn from_pretrained<P: AsRef<Path>>(model_name_or_path: P) -> Result<Self> {
        let path = model_name_or_path.as_ref();
        let mut probed: Vec<String> = Vec::new();

        // A HuggingFace tokenizer.json, either directly or inside the directory.
        if path.is_file() && path.extension().is_some_and(|ext| ext == "json") {
            return Ok(TokenizerWrapper::HuggingFace(TokenizerImpl::from_file(
                path,
            )?));
        }

        let tokenizer_json_path = path.join("tokenizer.json");
        probed.push(tokenizer_json_path.display().to_string());
        if tokenizer_json_path.is_file() {
            let tokenizer = TokenizerImpl::from_file(&tokenizer_json_path)?;
            return Ok(TokenizerWrapper::HuggingFace(tokenizer));
        }

        // A directory written by save_pretrained.
        let config_path = path.join(CONFIG_FILE);
        probed.push(config_path.display().to_string());
        if config_path.is_file() {
            return Self::from_state_dir(path, &config_path);
        }

        // Finally, treat the argument as a hub id resolved against the local cache.
        match TokenizerImpl::from_pretrained(path.to_string_lossy().as_ref()) {
            Ok(tokenizer) => Ok(TokenizerWrapper::HuggingFace(tokenizer)),
            Err(hub_error) => Err(TrustformersError::invalid_input(format!(
                "No tokenizer could be loaded from {:?}. Probed: {}. Local hub \
                 lookup also failed: {}",
                path,
                probed.join(", "),
                hub_error
            ))),
        }
    }

    /// Restore a tokenizer from a `tokenizer_config.json` state directory.
    fn from_state_dir(dir: &Path, config_path: &Path) -> Result<Self> {
        let config = Self::read_json(config_path)?;

        let tokenizer_type =
            config.get("tokenizer_type").and_then(|v| v.as_str()).ok_or_else(|| {
                TrustformersError::invalid_input(format!(
                    "{:?} has no \"tokenizer_type\" field",
                    config_path
                ))
            })?;

        match tokenizer_type {
            "WordPiece" => {
                let vocab = Self::read_vocab(dir, &config, config_path)?;
                let do_lower_case =
                    config.get("do_lower_case").and_then(|v| v.as_bool()).unwrap_or(false);
                Ok(TokenizerWrapper::WordPiece(
                    crate::wordpiece::WordPieceTokenizer::new(vocab, do_lower_case),
                ))
            },
            "BPE" => {
                let vocab = Self::read_vocab(dir, &config, config_path)?;
                let merges_file =
                    config.get("merges_file").and_then(|v| v.as_str()).unwrap_or(MERGES_FILE);
                let merges = Self::read_merges(&dir.join(merges_file))?;

                let bool_field = |name: &str, default: bool| {
                    config.get(name).and_then(|v| v.as_bool()).unwrap_or(default)
                };
                let string_field = |name: &str, default: &str| {
                    config.get(name).and_then(|v| v.as_str()).unwrap_or(default).to_string()
                };

                let tokenizer = crate::bpe::BPETokenizer::with_options(
                    vocab,
                    merges,
                    bool_field("normalize_unicode", true),
                    bool_field("preserve_case", true),
                    bool_field("handle_chinese_chars", false),
                    config
                        .get("max_input_chars_per_word")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize)
                        .unwrap_or(100),
                )
                .with_special_tokens(
                    string_field("unk_token", "<|endoftext|>"),
                    string_field("pad_token", "<|endoftext|>"),
                    string_field("bos_token", "<|endoftext|>"),
                    string_field("eos_token", "<|endoftext|>"),
                );

                Ok(TokenizerWrapper::BPE(tokenizer))
            },
            "Unigram" => {
                let vocab = Self::read_vocab(dir, &config, config_path)?;
                let scores_file =
                    config.get("scores_file").and_then(|v| v.as_str()).unwrap_or(SCORES_FILE);
                let scores = Self::read_scores(&dir.join(scores_file))?;
                let escape_whitespace =
                    config.get("escape_whitespace").and_then(|v| v.as_bool()).unwrap_or(false);

                let tokenizer = crate::unigram::UnigramTokenizer::new(vocab, scores)?
                    .with_whitespace_escaping(escape_whitespace);
                Ok(TokenizerWrapper::Unigram(tokenizer))
            },
            "Character" => {
                let vocab = Self::read_vocab(dir, &config, config_path)?;
                Ok(TokenizerWrapper::Char(crate::char::CharTokenizer::new(
                    vocab,
                )))
            },
            other => Err(TrustformersError::invalid_input(format!(
                "Unsupported tokenizer type: {}",
                other
            ))),
        }
    }

    /// Save the tokenizer, including its full vocabulary and configuration.
    pub fn save_pretrained<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();

        std::fs::create_dir_all(path)
            .map_err(|e| TrustformersError::other(format!("I/O error: {}", e)))?;

        match self {
            TokenizerWrapper::HuggingFace(tokenizer) => {
                let tokenizer_path = path.join("tokenizer.json");
                tokenizer.save_to_file(&tokenizer_path)
            },
            TokenizerWrapper::WordPiece(tokenizer) => {
                Self::write_vocab(path, &tokenizer.get_vocab())?;
                Self::write_config(
                    path,
                    serde_json::json!({
                        "tokenizer_type": "WordPiece",
                        "model_type": "WordPiece",
                        "version": TOKENIZER_STATE_VERSION,
                        "vocab_file": VOCAB_FILE,
                        "do_lower_case": tokenizer.do_lower_case(),
                    }),
                )
            },
            TokenizerWrapper::BPE(tokenizer) => {
                Self::write_vocab(path, tokenizer.get_vocab_map())?;
                Self::write_merges(&path.join(MERGES_FILE), tokenizer.get_merge_rules())?;
                Self::write_config(
                    path,
                    serde_json::json!({
                        "tokenizer_type": "BPE",
                        "model_type": "BPE",
                        "version": TOKENIZER_STATE_VERSION,
                        "vocab_file": VOCAB_FILE,
                        "merges_file": MERGES_FILE,
                        "normalize_unicode": tokenizer.normalizes_unicode(),
                        "preserve_case": tokenizer.preserves_case(),
                        "handle_chinese_chars": tokenizer.handles_chinese_chars(),
                        "max_input_chars_per_word": tokenizer.max_input_chars_per_word(),
                        "unk_token": tokenizer.unk_token(),
                        "pad_token": tokenizer.pad_token(),
                        "bos_token": tokenizer.bos_token(),
                        "eos_token": tokenizer.eos_token(),
                    }),
                )
            },
            TokenizerWrapper::Unigram(tokenizer) => {
                Self::write_vocab(path, &tokenizer.get_vocab())?;
                Self::write_json_pretty(&path.join(SCORES_FILE), tokenizer.scores())?;
                Self::write_config(
                    path,
                    serde_json::json!({
                        "tokenizer_type": "Unigram",
                        "model_type": "Unigram",
                        "version": TOKENIZER_STATE_VERSION,
                        "vocab_file": VOCAB_FILE,
                        "scores_file": SCORES_FILE,
                        "escape_whitespace": tokenizer.escapes_whitespace(),
                    }),
                )
            },
            TokenizerWrapper::Char(tokenizer) => {
                Self::write_vocab(path, &tokenizer.get_vocab())?;
                Self::write_config(
                    path,
                    serde_json::json!({
                        "tokenizer_type": "Character",
                        "model_type": "Character",
                        "version": TOKENIZER_STATE_VERSION,
                        "vocab_file": VOCAB_FILE,
                    }),
                )
            },
        }
    }

    fn write_config(dir: &Path, config: serde_json::Value) -> Result<()> {
        Self::write_json_pretty(&dir.join(CONFIG_FILE), &config)
    }

    fn write_json_pretty<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
        let json = serde_json::to_string_pretty(value)
            .map_err(|e| TrustformersError::serialization_error(e.to_string()))?;
        std::fs::write(path, json)
            .map_err(|e| TrustformersError::io_error(format!("Failed to write {:?}: {}", path, e)))
    }

    fn read_json(path: &Path) -> Result<serde_json::Value> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to read {:?}: {}", path, e))
        })?;
        serde_json::from_str(&text)
            .map_err(|e| TrustformersError::serialization_error(format!("{:?}: {}", path, e)))
    }

    /// Write the vocabulary as an explicit `{token: id}` map.
    ///
    /// A line-indexed `vocab.txt` would renumber any non-contiguous id space on
    /// reload, so the ids are stored explicitly.
    fn write_vocab(dir: &Path, vocab: &HashMap<String, u32>) -> Result<()> {
        Self::write_json_pretty(&dir.join(VOCAB_FILE), vocab)
    }

    fn read_vocab(
        dir: &Path,
        config: &serde_json::Value,
        config_path: &Path,
    ) -> Result<HashMap<String, u32>> {
        let vocab_file = config.get("vocab_file").and_then(|v| v.as_str()).unwrap_or(VOCAB_FILE);
        let vocab_path = dir.join(vocab_file);

        if !vocab_path.is_file() {
            return Err(TrustformersError::invalid_input(format!(
                "{:?} does not reference a usable vocabulary ({:?} is missing). \
                 Directories written before state format {} stored only a type \
                 marker; re-save the tokenizer with save_pretrained.",
                config_path, vocab_path, TOKENIZER_STATE_VERSION
            )));
        }

        let text = std::fs::read_to_string(&vocab_path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to read {:?}: {}", vocab_path, e))
        })?;
        let vocab: HashMap<String, u32> = serde_json::from_str(&text).map_err(|e| {
            TrustformersError::serialization_error(format!("{:?}: {}", vocab_path, e))
        })?;

        if vocab.is_empty() {
            return Err(TrustformersError::invalid_input(format!(
                "{:?} contains an empty vocabulary",
                vocab_path
            )));
        }

        Ok(vocab)
    }

    fn read_scores(path: &Path) -> Result<HashMap<String, f32>> {
        if !path.is_file() {
            return Err(TrustformersError::invalid_input(format!(
                "Unigram tokenizer state is missing its scores file {:?}",
                path
            )));
        }

        let text = std::fs::read_to_string(path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to read {:?}: {}", path, e))
        })?;
        serde_json::from_str(&text)
            .map_err(|e| TrustformersError::serialization_error(format!("{:?}: {}", path, e)))
    }

    fn write_merges(path: &Path, merges: &[(String, String)]) -> Result<()> {
        let mut text = format!("#version: {}\n", TOKENIZER_STATE_VERSION);
        for (first, second) in merges {
            text.push_str(first);
            text.push(' ');
            text.push_str(second);
            text.push('\n');
        }

        std::fs::write(path, text)
            .map_err(|e| TrustformersError::io_error(format!("Failed to write {:?}: {}", path, e)))
    }

    fn read_merges(path: &Path) -> Result<Vec<(String, String)>> {
        if !path.is_file() {
            return Err(TrustformersError::invalid_input(format!(
                "BPE tokenizer state is missing its merges file {:?}",
                path
            )));
        }

        let text = std::fs::read_to_string(path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to read {:?}: {}", path, e))
        })?;

        let mut merges = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(TrustformersError::invalid_input(format!(
                    "{:?} line {}: expected '<first> <second>', got {:?}",
                    path,
                    index + 1,
                    line
                )));
            }
            merges.push((parts[0].to_string(), parts[1].to_string()));
        }

        Ok(merges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenized_input_with_offsets_conversion() {
        let input_with_offsets = TokenizedInputWithOffsets {
            input_ids: vec![101, 2023, 2003, 102],
            attention_mask: vec![1, 1, 1, 1],
            token_type_ids: Some(vec![0, 0, 0, 0]),
            offset_mapping: Some(vec![(0, 0), (0, 4), (5, 7), (0, 0)]),
            special_tokens_mask: Some(vec![1, 0, 0, 1]),
        };

        let regular_input: TokenizedInput = input_with_offsets.into();

        assert_eq!(regular_input.input_ids, vec![101, 2023, 2003, 102]);
        assert_eq!(regular_input.attention_mask, vec![1, 1, 1, 1]);
        assert_eq!(regular_input.token_type_ids, Some(vec![0, 0, 0, 0]));
    }

    #[test]
    fn test_tokenizer_wrapper_char() {
        let text = "Hello World!";
        let tokenizer = crate::char::CharTokenizer::from_text(text, 1000);
        let wrapper = TokenizerWrapper::Char(tokenizer);

        let encoded = wrapper.encode(text).expect("Encoding failed");
        let decoded = wrapper.decode(&encoded.input_ids).expect("Decoding failed");

        assert!(!encoded.input_ids.is_empty());
        assert!(decoded.contains("Hello"));
        assert!(wrapper.vocab_size() > 0);
    }

    #[test]
    fn test_tokenizer_from_json_string() {
        // Simple minimal tokenizer JSON for testing
        let json_str = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 0,
                    "content": "[PAD]",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                },
                {
                    "id": 1,
                    "content": "[UNK]",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }
            ],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "[PAD]": 0,
                    "[UNK]": 1,
                    "hello": 2,
                    "world": 3
                },
                "unk_token": "[UNK]"
            }
        }"#;

        let result = TokenizerImpl::from_tokenizer_json(json_str);
        assert!(result.is_ok());

        if let Ok(tokenizer) = result {
            assert_eq!(tokenizer.vocab_size(), 4);
            assert_eq!(tokenizer.token_to_id("hello"), Some(2));
            assert_eq!(tokenizer.id_to_token(3), Some("world".to_string()));
        }
    }

    fn temp_dir_for(test_name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "trustformers_tokenizer_{}_{}",
            test_name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir must be creatable");
        dir
    }

    fn sample_bpe() -> crate::bpe::BPETokenizer {
        // Deliberately non-contiguous ids: a line-indexed vocab file would
        // silently renumber these on reload.
        let mut vocab = HashMap::new();
        vocab.insert("H".to_string(), 0);
        vocab.insert("h".to_string(), 1);
        vocab.insert("llo".to_string(), 5);
        vocab.insert("\u{0120}".to_string(), 9);
        vocab.insert("<|endoftext|>".to_string(), 11);

        let merges = vec![
            ("l".to_string(), "l".to_string()),
            ("ll".to_string(), "o".to_string()),
        ];

        crate::bpe::BPETokenizer::with_options(vocab, merges, true, true, false, 64)
    }

    /// Regression: save/load used to write a type marker only, so every reloaded
    /// token encoded to id 0.
    #[test]
    fn test_save_load_round_trip_preserves_bpe_state() {
        let dir = temp_dir_for("bpe_round_trip");
        let original = TokenizerWrapper::BPE(sample_bpe());

        // Mixed case plus a CJK character: a lost `preserve_case` or
        // `handle_chinese_chars` flag changes the ids.
        let text = "Hllo hllo \u{4e16}";
        let expected = original.encode(text).expect("encoding must succeed");

        original.save_pretrained(&dir).expect("saving must succeed");
        let reloaded = TokenizerWrapper::from_pretrained(&dir).expect("reloading must succeed");

        let TokenizerWrapper::BPE(reloaded_bpe) = &reloaded else {
            panic!("expected a BPE tokenizer, got {:?}", reloaded);
        };

        // Ids survive exactly, including the non-contiguous ones.
        assert_eq!(reloaded_bpe.get_vocab_map(), original_vocab(&original));
        assert_eq!(reloaded_bpe.token_to_id("llo"), Some(5));
        assert_eq!(reloaded_bpe.token_to_id("\u{0120}"), Some(9));
        assert_eq!(reloaded_bpe.vocab_size(), 5);

        // Merges survive in order.
        assert_eq!(
            reloaded_bpe.get_merge_rules().as_slice(),
            [
                ("l".to_string(), "l".to_string()),
                ("ll".to_string(), "o".to_string())
            ]
        );

        // Options survive.
        assert!(reloaded_bpe.preserves_case());
        assert!(!reloaded_bpe.handles_chinese_chars());
        assert!(reloaded_bpe.normalizes_unicode());
        assert_eq!(reloaded_bpe.max_input_chars_per_word(), 64);

        // And the encoding is identical, not all-zeros.
        let actual = reloaded.encode(text).expect("encoding must succeed");
        assert_eq!(actual.input_ids, expected.input_ids);
        assert!(
            actual.input_ids.iter().any(|&id| id != 0),
            "a reloaded tokenizer that maps everything to 0 has lost its vocabulary"
        );
        // "Hllo" merges to ["H", "llo"] = [0, 5].
        assert_eq!(&actual.input_ids[..2], &[0, 5]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn original_vocab(wrapper: &TokenizerWrapper) -> &HashMap<String, u32> {
        match wrapper {
            TokenizerWrapper::BPE(t) => t.get_vocab_map(),
            _ => panic!("expected a BPE tokenizer"),
        }
    }

    #[test]
    fn test_save_load_round_trip_preserves_wordpiece_state() {
        let dir = temp_dir_for("wordpiece_round_trip");

        let mut vocab = HashMap::new();
        for (index, token) in [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world",
        ]
        .iter()
        .enumerate()
        {
            // Non-contiguous ids again.
            vocab.insert((*token).to_string(), (index as u32) * 3);
        }
        let original = TokenizerWrapper::WordPiece(crate::wordpiece::WordPieceTokenizer::new(
            vocab.clone(),
            true,
        ));
        let expected = original.encode("HELLO world").expect("encoding must succeed");

        original.save_pretrained(&dir).expect("saving must succeed");
        let reloaded = TokenizerWrapper::from_pretrained(&dir).expect("reloading must succeed");

        let TokenizerWrapper::WordPiece(reloaded_wp) = &reloaded else {
            panic!("expected a WordPiece tokenizer");
        };
        assert_eq!(&reloaded_wp.get_vocab(), &vocab);
        assert!(
            reloaded_wp.do_lower_case(),
            "do_lower_case must survive the round trip"
        );
        assert_eq!(
            reloaded.encode("HELLO world").expect("encoding must succeed").input_ids,
            expected.input_ids
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_load_round_trip_preserves_unigram_scores() {
        let dir = temp_dir_for("unigram_round_trip");

        let mut vocab = HashMap::new();
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("\u{2581}hello".to_string(), 4);
        vocab.insert("\u{2581}world".to_string(), 8);

        let mut scores = HashMap::new();
        scores.insert("<unk>".to_string(), -20.0f32);
        scores.insert("\u{2581}hello".to_string(), -1.5f32);
        scores.insert("\u{2581}world".to_string(), -2.5f32);

        let tokenizer = crate::unigram::UnigramTokenizer::new(vocab.clone(), scores.clone())
            .expect("construction must succeed")
            .with_whitespace_escaping(true);
        let original = TokenizerWrapper::Unigram(tokenizer);
        let expected = original.encode("hello world").expect("encoding must succeed");
        assert_eq!(expected.input_ids, vec![4, 8]);

        original.save_pretrained(&dir).expect("saving must succeed");
        let reloaded = TokenizerWrapper::from_pretrained(&dir).expect("reloading must succeed");

        let TokenizerWrapper::Unigram(reloaded_unigram) = &reloaded else {
            panic!("expected a Unigram tokenizer");
        };
        assert_eq!(&reloaded_unigram.get_vocab(), &vocab);
        assert_eq!(reloaded_unigram.scores(), &scores);
        assert!(reloaded_unigram.escapes_whitespace());
        assert_eq!(
            reloaded.encode("hello world").expect("encoding must succeed").input_ids,
            expected.input_ids
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_load_round_trip_preserves_char_vocab() {
        let dir = temp_dir_for("char_round_trip");

        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 2);
        vocab.insert("b".to_string(), 7);
        let original = TokenizerWrapper::Char(crate::char::CharTokenizer::new(vocab.clone()));

        original.save_pretrained(&dir).expect("saving must succeed");
        let reloaded = TokenizerWrapper::from_pretrained(&dir).expect("reloading must succeed");

        assert_eq!(reloaded.token_to_id("a"), Some(2));
        assert_eq!(reloaded.token_to_id("b"), Some(7));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression: a missing/unreadable model used to yield an empty BPE tokenizer.
    #[test]
    fn test_from_pretrained_errors_instead_of_returning_an_empty_tokenizer() {
        let dir = temp_dir_for("empty_dir");

        let error = TokenizerWrapper::from_pretrained(&dir)
            .expect_err("an empty directory must not produce a usable tokenizer");
        let message = error.to_string();
        assert!(
            message.contains("tokenizer.json"),
            "error must list probed paths: {}",
            message
        );
        assert!(message.contains("tokenizer_config.json"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression: version 1.0 configs carried a type marker and no vocabulary.
    #[test]
    fn test_from_pretrained_rejects_legacy_type_marker_config() {
        let dir = temp_dir_for("legacy_config");
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"tokenizer_type":"BPE","model_type":"BPE","version":"1.0"}"#,
        )
        .expect("config must be writable");

        let error = TokenizerWrapper::from_pretrained(&dir)
            .expect_err("a vocabulary-less config must be rejected");
        assert!(error.to_string().contains("vocab.json"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn word_level_tokenizer() -> TokenizerImpl {
        let json_str = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3},
                "unk_token": "[UNK]"
            }
        }"#;
        TokenizerImpl::from_tokenizer_json(json_str).expect("tokenizer json must parse")
    }

    /// Regression: `with_config` assigned fields that nothing ever read.
    #[test]
    fn test_with_config_applies_truncation_and_lowercasing() {
        let plain = word_level_tokenizer();
        assert_eq!(
            plain.encode("HELLO").expect("encoding must succeed").input_ids,
            vec![1],
            "without lowercasing, HELLO is unknown"
        );

        let configured = word_level_tokenizer()
            .with_config(true, Some(2))
            .expect("configuration must apply");

        assert_eq!(
            configured.encode("HELLO").expect("encoding must succeed").input_ids,
            vec![2],
            "do_lower_case must actually lowercase the input"
        );

        let truncated =
            configured.encode("hello world hello world").expect("encoding must succeed");
        assert_eq!(
            truncated.input_ids.len(),
            2,
            "max_length must actually truncate"
        );

        assert!(configured.does_lower_case());
        assert_eq!(configured.max_length(), Some(2));
    }

    /// Regression: the primary encode path discarded offsets it already had.
    #[test]
    fn test_encode_keeps_offsets_and_special_tokens_mask() {
        let tokenizer = word_level_tokenizer();
        let encoded = tokenizer.encode("hello world").expect("encoding must succeed");

        assert_eq!(
            encoded.offset_mapping,
            Some(vec![(0, 5), (6, 11)]),
            "offsets computed by the tokenizer must be propagated"
        );
        assert_eq!(encoded.special_tokens_mask, Some(vec![0, 0]));
    }

    /// Regression: pair offsets were aligned against a fabricated concatenation,
    /// so every second-sequence token pointed at the wrong span.
    #[test]
    fn test_encode_pair_alignment_shifts_second_sequence_offsets() {
        let mut tokenizer = word_level_pair_tokenizer().with_word_alignment();

        let text = "hello world";
        let text2 = "foo bar";
        let aligned = tokenizer
            .encode_pair_with_alignment(text, text2, false)
            .expect("pair encoding must succeed");

        let offsets = aligned.offset_mapping.expect("pair encoding must carry offsets");
        assert_eq!(offsets.len(), 4);

        // First sequence keeps its own coordinates.
        assert_eq!(offsets[0], (0, 5));
        assert_eq!(offsets[1], (6, 11));

        // Second sequence is shifted into the combined string "hello world foo bar".
        let shift = text.len() + 1;
        assert_eq!(offsets[2], (shift, shift + 3));
        assert_eq!(offsets[3], (shift + 4, shift + 7));

        // The shifted spans really do index the combined text.
        let combined = format!("{} {}", text, text2);
        assert_eq!(&combined[offsets[2].0..offsets[2].1], "foo");
        assert_eq!(&combined[offsets[3].0..offsets[3].1], "bar");

        // ...and the alignment engine maps them onto the right words.
        let third = aligned
            .word_alignments
            .iter()
            .find(|alignment| alignment.token_index == 2)
            .expect("token 2 must be aligned");
        assert_eq!(third.char_start, shift);
        let word = aligned
            .words
            .get(third.word_index.expect("token 2 must belong to a word"))
            .expect("word index must be valid");
        assert_eq!(word.text, "foo");
    }

    fn word_level_pair_tokenizer() -> TokenizerImpl {
        let json_str = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "foo": 3, "bar": 4},
                "unk_token": "[UNK]"
            }
        }"#;
        TokenizerImpl::from_tokenizer_json(json_str).expect("tokenizer json must parse")
    }

    /// Regression: the old probe used `{cache}/{name}/tokenizer.json`, a layout
    /// huggingface_hub never writes, so a genuinely cached model was never found.
    ///
    /// This test sets `HF_HUB_CACHE`, which is process-global. Under `cargo
    /// nextest` each test runs in its own process; under plain `cargo test` the
    /// only other environment-reading test asserts an error either way.
    #[test]
    fn test_from_pretrained_reads_the_huggingface_hub_cache_layout() {
        let root = temp_dir_for("hub_cache");
        let hub = root.join("hub");
        let repo = hub.join("models--acme--demo-model");
        let sha = "0123456789abcdef0123456789abcdef01234567";

        std::fs::create_dir_all(repo.join("refs")).expect("refs dir must be creatable");
        std::fs::create_dir_all(repo.join("snapshots").join(sha))
            .expect("snapshot dir must be creatable");
        std::fs::write(repo.join("refs").join("main"), sha).expect("ref must be writable");

        let tokenizer_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "hello": 1},
                "unk_token": "[UNK]"
            }
        }"#;
        std::fs::write(
            repo.join("snapshots").join(sha).join("tokenizer.json"),
            tokenizer_json,
        )
        .expect("tokenizer.json must be writable");

        std::env::set_var("HF_HUB_CACHE", &hub);
        let loaded = TokenizerImpl::from_pretrained("acme/demo-model");
        std::env::remove_var("HF_HUB_CACHE");

        let loaded = loaded.expect("the hub cache layout must resolve");
        assert_eq!(loaded.token_to_id("hello"), Some(1));

        let _ = std::fs::remove_dir_all(&root);
    }
}
