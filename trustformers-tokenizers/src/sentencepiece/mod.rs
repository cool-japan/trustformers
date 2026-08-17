//! SentencePiece tokenizer (Unigram / BPE / Word / Char models).
//!
//! Unigram segmentation is a genuine Viterbi search over a lattice of character
//! positions (see [`SentencePieceTokenizer::segmentation_score`]), `.model`
//! files are parsed with the bounds-checked protobuf reader in [`proto`], and
//! loading never falls back to an invented vocabulary: if no model file
//! resolves, [`SentencePieceTokenizer::from_pretrained`] returns an error that
//! names every path it probed.

pub mod proto;

use proto::PieceType;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};
use unicode_normalization::UnicodeNormalization;

/// SentencePiece word-boundary marker (U+2581 LOWER ONE EIGHTH BLOCK).
pub const WHITESPACE_MARKER: char = '▁';

/// Penalty below the least likely known piece used for unknown characters,
/// mirroring SentencePiece's `kUnkPenalty`.
const UNK_PENALTY: f32 = 10.0;

#[derive(Debug, Clone)]
pub struct SentencePieceTokenizer {
    vocab: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    special_tokens: HashMap<String, u32>,
    scores: HashMap<u32, f32>,
    pad_token_id: Option<u32>,
    unk_token_id: Option<u32>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,

    // Tokenizer configuration
    model_type: ModelType,
    normalization: bool,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    treat_whitespace_as_suffix: bool,
    byte_fallback: bool,

    // Character normalization settings
    nfc_normalization: bool,
    nfkc_normalization: bool,
    escape_whitespaces: bool,

    // Derived lattice statistics (refreshed whenever the vocabulary changes)
    max_piece_chars: usize,
    unk_score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    Unigram,
    Bpe,
    Word,
    Char,
}

impl SentencePieceTokenizer {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            special_tokens: HashMap::new(),
            scores: HashMap::new(),
            pad_token_id: None,
            unk_token_id: None,
            bos_token_id: None,
            eos_token_id: None,

            model_type: ModelType::Unigram,
            normalization: true,
            add_dummy_prefix: true,
            remove_extra_whitespaces: true,
            treat_whitespace_as_suffix: false,
            byte_fallback: false,

            nfc_normalization: true,
            nfkc_normalization: false,
            escape_whitespaces: true,

            max_piece_chars: 0,
            unk_score: -UNK_PENALTY,
        }
    }

    /// Load a SentencePiece model from a `.model` file.
    ///
    /// The binary protobuf format is tried first; files that are not protobuf
    /// are loaded as SentencePiece's plain-text `piece<TAB>score` vocabulary.
    pub fn from_model_file<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let path = model_path.as_ref();
        if !path.is_file() {
            return Err(TrustformersError::invalid_config(format!(
                "Model file not found: {:?}",
                path
            )));
        }

        let mut tokenizer = Self::new();
        tokenizer.load_vocab_from_model_file(path)?;
        Ok(tokenizer)
    }

    /// Load vocabulary from either SentencePiece model format.
    pub fn load_vocab_from_model_file<P: AsRef<Path>>(&mut self, model_path: P) -> Result<()> {
        let path = model_path.as_ref();

        // Try to parse as protobuf first (standard SentencePiece format)
        if self.load_protobuf_model(path).is_ok() {
            return Ok(());
        }

        // Fallback to text-based vocabulary loading
        self.load_text_vocab(path)
    }

    /// Load a SentencePiece protobuf `.model` file.
    fn load_protobuf_model<P: AsRef<Path>>(&mut self, model_path: P) -> Result<()> {
        let mut file = File::open(model_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let model = proto::parse_model_proto(&buffer)?;

        self.model_type = match model.trainer_spec.model_type {
            2 => ModelType::Bpe,
            3 => ModelType::Word,
            4 => ModelType::Char,
            _ => ModelType::Unigram,
        };

        self.add_dummy_prefix = model.normalizer_spec.add_dummy_prefix;
        self.remove_extra_whitespaces = model.normalizer_spec.remove_extra_whitespaces;
        self.escape_whitespaces = model.normalizer_spec.escape_whitespaces;
        // SentencePiece always runs its normalizer; the individual flags above
        // (not this switch) decide what it actually does.
        self.normalization = true;
        self.treat_whitespace_as_suffix = model.trainer_spec.treat_whitespace_as_suffix;
        self.byte_fallback = model.trainer_spec.byte_fallback;

        for (id, piece) in model.pieces.iter().enumerate() {
            let token_id = id as u32;
            self.vocab.insert(piece.piece.clone(), token_id);
            self.id_to_token.insert(token_id, piece.piece.clone());
            self.scores.insert(token_id, piece.score);

            match piece.piece_type {
                PieceType::Unknown => {
                    self.unk_token_id = Some(token_id);
                    self.special_tokens.insert(piece.piece.clone(), token_id);
                },
                PieceType::Control => {
                    self.special_tokens.insert(piece.piece.clone(), token_id);

                    if piece.piece == "<pad>" {
                        self.pad_token_id = Some(token_id);
                    } else if piece.piece == "<s>" {
                        self.bos_token_id = Some(token_id);
                    } else if piece.piece == "</s>" {
                        self.eos_token_id = Some(token_id);
                    }
                },
                PieceType::UserDefined => {
                    self.special_tokens.insert(piece.piece.clone(), token_id);
                },
                _ => {},
            }
        }

        self.refresh_stats();
        Ok(())
    }

    /// Load text-based vocabulary file
    fn load_text_vocab<P: AsRef<Path>>(&mut self, vocab_path: P) -> Result<()> {
        let path = vocab_path.as_ref();
        let file = File::open(path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to open vocab file: {}", e))
        })?;

        let reader = BufReader::new(file);
        self.load_text_vocab_from_reader(reader)
    }

    /// Load vocabulary from the `piece<TAB>score` text format.
    ///
    /// A malformed score is an error rather than a silent `0.0`: scores are log
    /// probabilities, so `0.0` is the *most* likely value a piece can have and a
    /// corrupt field would quietly dominate every Viterbi path.
    fn load_text_vocab_from_reader<R: BufRead>(&mut self, reader: R) -> Result<()> {
        // Parse everything before touching `self`, so a malformed file never
        // leaves a half-populated vocabulary behind.
        let mut entries: Vec<(u32, String, f32)> = Vec::new();

        for (id, line) in reader.lines().enumerate() {
            let line = line
                .map_err(|e| TrustformersError::io_error(format!("Failed to read line: {}", e)))?;
            let mut parts = line.split('\t');

            let Some(token) = parts.next() else {
                continue;
            };
            if token.is_empty() {
                continue;
            }

            let score = match parts.next() {
                Some(raw) => raw.trim().parse::<f32>().map_err(|e| {
                    TrustformersError::invalid_input(format!(
                        "SentencePiece text vocabulary line {}: {:?} is not a valid piece \
                         score ({})",
                        id + 1,
                        raw,
                        e
                    ))
                })?,
                None => 0.0,
            };

            entries.push((id as u32, token.to_string(), score));
        }

        if entries.is_empty() {
            return Err(TrustformersError::invalid_input(
                "SentencePiece text vocabulary contains no pieces".to_string(),
            ));
        }

        for (token_id, token, score) in entries {
            self.vocab.insert(token.clone(), token_id);
            self.id_to_token.insert(token_id, token.clone());
            self.scores.insert(token_id, score);

            // Detect special tokens
            if token == "<pad>" || token == "[PAD]" {
                self.pad_token_id = Some(token_id);
                self.special_tokens.insert(token, token_id);
            } else if token == "<unk>" || token == "[UNK]" {
                self.unk_token_id = Some(token_id);
                self.special_tokens.insert(token, token_id);
            } else if token == "<s>" || token == "[CLS]" {
                self.bos_token_id = Some(token_id);
                self.special_tokens.insert(token, token_id);
            } else if token == "</s>" || token == "[SEP]" {
                self.eos_token_id = Some(token_id);
                self.special_tokens.insert(token, token_id);
            }
        }

        self.refresh_stats();
        Ok(())
    }

    /// Configure the tokenizer model type
    pub fn with_model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = model_type;
        self
    }

    /// Configure normalization settings
    pub fn with_normalization(mut self, enable: bool) -> Self {
        self.normalization = enable;
        self
    }

    /// Configure dummy prefix addition
    pub fn with_dummy_prefix(mut self, enable: bool) -> Self {
        self.add_dummy_prefix = enable;
        self
    }

    /// Configure byte fallback
    pub fn with_byte_fallback(mut self, enable: bool) -> Self {
        self.byte_fallback = enable;
        self
    }

    /// Load a SentencePiece tokenizer for `model_name_or_path` from disk.
    ///
    /// The following locations are probed, in order:
    ///
    /// * `{model_name_or_path}/spiece.model`
    /// * `{model_name_or_path}/sentencepiece.bpe.model`
    /// * `{model_name_or_path}/tokenizer.model`
    /// * `{model_name_or_path}.model`
    /// * `{model_name_or_path}` itself, when it points at a file
    ///
    /// If none resolves, an error naming every probed path is returned. This
    /// function never fabricates a vocabulary: a bare hub name such as
    /// `"t5-small"` must be downloaded first.
    pub fn from_pretrained(model_name_or_path: &str) -> Result<Self> {
        let potential_paths = vec![
            format!("{}/spiece.model", model_name_or_path),
            format!("{}/sentencepiece.bpe.model", model_name_or_path),
            format!("{}/tokenizer.model", model_name_or_path),
            format!("{}.model", model_name_or_path),
            model_name_or_path.to_string(),
        ];

        let mut failures = Vec::new();
        for candidate in &potential_paths {
            if !Path::new(candidate).is_file() {
                continue;
            }
            match Self::from_model_file(candidate) {
                Ok(tokenizer) => return Ok(tokenizer),
                Err(e) => failures.push(format!("{} ({})", candidate, e)),
            }
        }

        let detail = if failures.is_empty() {
            format!("probed paths: {}", potential_paths.join(", "))
        } else {
            format!(
                "probed paths: {}; candidates that existed but failed to load: {}",
                potential_paths.join(", "),
                failures.join("; ")
            )
        };

        Err(TrustformersError::invalid_input(format!(
            "No SentencePiece model found for '{}'. A `.model` file from the \
             checkpoint is required; {}.",
            model_name_or_path, detail
        )))
    }

    /// Load a plain one-token-per-line vocabulary file into this tokenizer.
    pub fn load_vocab_from_file(&mut self, vocab_file: &str) -> Result<()> {
        let content = std::fs::read_to_string(vocab_file)
            .map_err(|e| TrustformersError::other(format!("Failed to read vocab file: {}", e)))?;

        for (id, line) in content.lines().enumerate() {
            let token = line.trim().to_string();
            let token_id = id as u32;
            self.vocab.insert(token.clone(), token_id);
            self.id_to_token.insert(token_id, token);
        }

        self.refresh_stats();
        Ok(())
    }

    /// Recompute lattice statistics derived from the vocabulary and scores.
    fn refresh_stats(&mut self) {
        self.max_piece_chars = self.vocab.keys().map(|k| k.chars().count()).max().unwrap_or(0);

        let min_score = self
            .scores
            .values()
            .copied()
            .filter(|s| s.is_finite())
            .fold(f32::INFINITY, f32::min);
        self.unk_score = if min_score.is_finite() { min_score - UNK_PENALTY } else { -UNK_PENALTY };
    }

    fn max_piece_chars(&self) -> usize {
        if self.max_piece_chars > 0 {
            self.max_piece_chars
        } else {
            self.vocab.keys().map(|k| k.chars().count()).max().unwrap_or(1).max(1)
        }
    }

    /// Normalize input text according to SentencePiece standards
    fn normalize_text(&self, text: &str) -> String {
        if !self.normalization {
            return text.to_string();
        }

        let mut normalized = text.to_string();

        // Unicode normalization
        if self.nfc_normalization {
            normalized = normalized.nfc().collect();
        }
        if self.nfkc_normalization {
            normalized = normalized.nfkc().collect();
        }

        // Remove extra whitespaces
        if self.remove_extra_whitespaces {
            normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
        }

        // Escape whitespaces
        if self.escape_whitespaces {
            normalized = normalized.replace(' ', "▁");
        }

        // Add dummy prefix if needed
        if self.add_dummy_prefix && !normalized.starts_with(WHITESPACE_MARKER) {
            normalized = format!("{}{}", WHITESPACE_MARKER, normalized);
        }

        normalized
    }

    /// Tokenize text using the appropriate algorithm based on model type
    fn tokenize_text(&self, text: &str) -> Vec<String> {
        let normalized_text = self.normalize_text(text);

        match self.model_type {
            ModelType::Unigram => self.tokenize_unigram(&normalized_text),
            ModelType::Bpe => self.tokenize_bpe(&normalized_text),
            ModelType::Word => self.tokenize_word(&normalized_text),
            ModelType::Char => self.tokenize_char(&normalized_text),
        }
    }

    /// Tokenize `text` into SentencePiece pieces (public for inspection/tests).
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.tokenize_text(text)
    }

    /// Score a candidate segmentation with the model's unigram log-probabilities.
    ///
    /// Pieces outside the vocabulary are charged [`Self::unk_score`], exactly as
    /// the Viterbi lattice does, so two segmentations can be compared directly.
    pub fn segmentation_score(&self, pieces: &[String]) -> f32 {
        pieces.iter().map(|piece| self.piece_score(piece)).sum()
    }

    /// Score of a single piece (unknown pieces get [`Self::unk_score`]).
    pub fn piece_score(&self, piece: &str) -> f32 {
        match self.vocab.get(piece) {
            Some(id) => self.scores.get(id).copied().unwrap_or(self.unk_score),
            None => self.unk_score,
        }
    }

    /// Score assigned to an unknown single-character lattice edge.
    pub fn unk_score(&self) -> f32 {
        self.unk_score
    }

    /// Unigram tokenization: Viterbi search over the piece lattice.
    ///
    /// `best[i]` holds the highest total log-probability of any segmentation of
    /// the first `i` characters together with the start of the piece that
    /// achieved it. Every position is reachable because an unknown
    /// single-character edge (scored [`Self::unk_score`]) is always available,
    /// so a single out-of-vocabulary character cannot invalidate the path.
    fn tokenize_unigram(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return vec![];
        }

        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let window = self.max_piece_chars();
        let unk_score = self.unk_score;

        let mut best = vec![(f32::NEG_INFINITY, 0usize); len + 1];
        best[0] = (0.0, 0);

        let mut candidate_buffer = String::new();
        for end in 1..=len {
            let lower = end.saturating_sub(window);
            for start in lower..end {
                if best[start].0 == f32::NEG_INFINITY {
                    continue;
                }

                candidate_buffer.clear();
                candidate_buffer.extend(chars[start..end].iter());

                let score = match self.vocab.get(candidate_buffer.as_str()) {
                    Some(id) => self.scores.get(id).copied().unwrap_or(unk_score),
                    // Unknown single character: always reachable, heavily penalized.
                    None if end - start == 1 => unk_score,
                    None => continue,
                };

                if !score.is_finite() {
                    continue;
                }

                let candidate = best[start].0 + score;
                if candidate > best[end].0 {
                    best[end] = (candidate, start);
                }
            }
        }

        let mut pieces = Vec::new();
        let mut pos = len;
        while pos > 0 {
            let start = best[pos].1;
            debug_assert!(start < pos, "Viterbi backtrack must make progress");
            if start >= pos {
                break;
            }
            pieces.push(chars[start..pos].iter().collect::<String>());
            pos = start;
        }

        pieces.reverse();
        pieces
    }

    /// BPE tokenization: repeatedly merge the highest-scoring adjacent pair.
    fn tokenize_bpe(&self, text: &str) -> Vec<String> {
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        loop {
            let mut best_pair = None;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_pos = 0;

            for i in 0..(tokens.len().saturating_sub(1)) {
                let merged = format!("{}{}", tokens[i], tokens[i + 1]);
                if let Some(&token_id) = self.vocab.get(&merged) {
                    let score = self.scores.get(&token_id).copied().unwrap_or(0.0);
                    if score > best_score {
                        best_score = score;
                        best_pair = Some(merged);
                        best_pos = i;
                    }
                }
            }

            if let Some(pair) = best_pair {
                tokens[best_pos] = pair;
                tokens.remove(best_pos + 1);
            } else {
                break;
            }
        }

        tokens
    }

    /// Word-level tokenization
    fn tokenize_word(&self, text: &str) -> Vec<String> {
        text.split(WHITESPACE_MARKER)
            .filter(|s| !s.is_empty())
            .map(|s| format!("{}{}", WHITESPACE_MARKER, s))
            .collect()
    }

    /// Character-level tokenization
    fn tokenize_char(&self, text: &str) -> Vec<String> {
        text.chars().map(|c| c.to_string()).collect()
    }

    /// The id of the unknown piece, or an error when the model has none.
    fn unk_id(&self) -> Result<u32> {
        self.unk_token_id.ok_or_else(|| {
            TrustformersError::invalid_config(
                "SentencePiece model has no <unk> piece, so out-of-vocabulary text \
                 cannot be encoded"
                    .to_string(),
            )
        })
    }

    fn convert_tokens_to_ids(&self, tokens: &[String]) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(tokens.len());

        for token in tokens {
            if let Some(&id) = self.vocab.get(token) {
                ids.push(id);
                continue;
            }

            if self.byte_fallback {
                ids.extend(self.byte_fallback_ids(token)?);
            } else {
                ids.push(self.unk_id()?);
            }
        }

        Ok(ids)
    }

    /// Map an out-of-vocabulary piece onto the model's `<0xNN>` byte pieces.
    fn byte_fallback_ids(&self, token: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(token.len());

        for &byte in token.as_bytes() {
            let piece = Self::byte_piece(byte);
            let id = self.vocab.get(&piece).copied().ok_or_else(|| {
                TrustformersError::invalid_config(format!(
                    "byte fallback is enabled but the vocabulary has no `{}` piece \
                     (needed for byte 0x{:02X} of {:?})",
                    piece, byte, token
                ))
            })?;
            ids.push(id);
        }

        Ok(ids)
    }

    /// The SentencePiece byte piece for a raw byte, e.g. `<0x41>`.
    fn byte_piece(byte: u8) -> String {
        format!("<0x{:02X}>", byte)
    }

    /// Parse a `<0xNN>` byte piece back to its byte value.
    fn byte_piece_value(token: &str) -> Option<u8> {
        let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
        if hex.len() != 2 {
            return None;
        }
        u8::from_str_radix(hex, 16).ok()
    }

    fn convert_ids_to_tokens(&self, ids: &[u32]) -> Vec<String> {
        ids.iter().filter_map(|id| self.id_to_token.get(id).cloned()).collect()
    }

    /// Reconstruct text from pieces, re-assembling `<0xNN>` byte runs.
    fn decode_tokens(&self, tokens: &[String], skip_special_tokens: bool) -> String {
        let mut result = String::new();
        let mut byte_run: Vec<u8> = Vec::new();

        for token in tokens {
            if let Some(byte) = Self::byte_piece_value(token) {
                byte_run.push(byte);
                continue;
            }

            if !byte_run.is_empty() {
                result.push_str(&String::from_utf8_lossy(&byte_run));
                byte_run.clear();
            }

            if skip_special_tokens && self.is_special_token(token) {
                continue;
            }

            result.push_str(token);
        }

        if !byte_run.is_empty() {
            result.push_str(&String::from_utf8_lossy(&byte_run));
        }

        result.replace(WHITESPACE_MARKER, " ").trim().to_string()
    }

    /// Whether `token` is one of the model's special (control/unknown/user
    /// defined) pieces.
    ///
    /// Driven entirely by the pieces recorded while loading the model, so T5
    /// sentinels are only treated as special when the checkpoint declares them.
    fn is_special_token(&self, token: &str) -> bool {
        if self.special_tokens.contains_key(token) {
            return true;
        }

        [
            self.pad_token_id,
            self.unk_token_id,
            self.bos_token_id,
            self.eos_token_id,
        ]
        .iter()
        .flatten()
        .any(|id| self.id_to_token.get(id).map(|t| t == token).unwrap_or(false))
    }

    /// Decode with explicit control over special-token removal.
    ///
    /// `skip_special_tokens = false` keeps control pieces (and T5
    /// `<extra_id_N>` sentinels) in the output, which is required to interpret
    /// span-corruption generations.
    pub fn decode_with_options(&self, ids: &[u32], skip_special_tokens: bool) -> String {
        let tokens = self.convert_ids_to_tokens(ids);
        self.decode_tokens(&tokens, skip_special_tokens)
    }

    /// Get token score for ranking
    pub fn get_token_score(&self, token_id: u32) -> Option<f32> {
        self.scores.get(&token_id).copied()
    }

    /// Get all tokens sorted by score
    pub fn get_tokens_by_score(&self) -> Vec<(String, u32, f32)> {
        let mut tokens: Vec<_> = self
            .vocab
            .iter()
            .map(|(token, &id)| {
                let score = self.scores.get(&id).copied().unwrap_or(0.0);
                (token.clone(), id, score)
            })
            .collect();

        tokens.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        tokens
    }

    /// Get the score for a specific token ID
    pub fn get_score(&self, token_id: u32) -> Option<f32> {
        self.scores.get(&token_id).copied()
    }

    /// Piece log-probabilities keyed by token id.
    pub fn scores(&self) -> &HashMap<u32, f32> {
        &self.scores
    }

    /// Check if a token is a special token (public version)
    pub fn is_special_token_public(&self, token: &str) -> bool {
        self.is_special_token(token)
    }

    /// Get the UNK token string
    pub fn unk_token(&self) -> Option<&str> {
        self.unk_token_id.and_then(|id| self.id_to_token.get(&id).map(|s| s.as_str()))
    }

    /// Get BOS token ID
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Get PAD token ID
    pub fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }

    /// Get UNK token ID
    pub fn unk_token_id(&self) -> Option<u32> {
        self.unk_token_id
    }

    /// Check if normalization is enabled
    pub fn uses_normalization(&self) -> bool {
        self.normalization
    }

    /// Check if extra whitespace removal is enabled
    pub fn removes_extra_whitespaces(&self) -> bool {
        self.remove_extra_whitespaces
    }

    /// Check if whitespace is treated as suffix
    pub fn treats_whitespace_as_suffix(&self) -> bool {
        self.treat_whitespace_as_suffix
    }

    /// Get model type as string
    pub fn model_type_string(&self) -> String {
        match self.model_type {
            ModelType::Unigram => "Unigram".to_string(),
            ModelType::Bpe => "BPE".to_string(),
            ModelType::Word => "Word".to_string(),
            ModelType::Char => "Char".to_string(),
        }
    }

    /// Check if byte fallback is enabled
    pub fn uses_byte_fallback(&self) -> bool {
        self.byte_fallback
    }
}

impl Default for SentencePieceTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let tokens = self.tokenize_text(text);
        let input_ids = self.convert_tokens_to_ids(&tokens)?;

        let attention_mask = vec![1u8; input_ids.len()];
        let special_tokens_mask: Vec<u8> = input_ids
            .iter()
            .map(|id| {
                let is_special = self
                    .id_to_token
                    .get(id)
                    .map(|token| self.is_special_token(token))
                    .unwrap_or(false);
                u8::from(is_special)
            })
            .collect();

        Ok(TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids: None,
            special_tokens_mask: Some(special_tokens_mask),
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        // T5-style pairing: the sequences are joined by the EOS piece.
        let combined_text = format!("{} </s> {}", text, text2);
        self.encode(&combined_text)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(self.decode_with_options(ids, true))
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.id_to_token.get(&id).cloned()
    }
}

#[cfg(test)]
mod tests;
