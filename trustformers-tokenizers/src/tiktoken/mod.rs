//! Tiktoken-compatible byte-level BPE tokenizer (OpenAI encodings).
//!
//! This module implements the real tiktoken algorithm: text is split with the
//! encoding's regex, and each pre-token's raw bytes are merged by BPE **rank**
//! until no ranked pair remains.
//!
//! Rank tables are never invented. `cl100k_base()` / `p50k_base()` /
//! `r50k_base()` locate a `.tiktoken` rank file through the documented search
//! paths in [`ranks::search_paths`] (overridable with
//! `$TRUSTFORMERS_TIKTOKEN_DIR`) and return an instructive error when none is
//! present; [`TiktokenTokenizer::from_file`] and
//! [`TiktokenTokenizer::from_reader`] load one explicitly.

pub mod ranks;

use crate::bpe::{byte_to_unicode, unicode_to_byte};
use fancy_regex::Regex as FancyRegex;
use ranks::RankMap;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::path::Path;
use std::sync::RwLock;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// GPT-2 / GPT-3 (`r50k_base`, `p50k_base`) pre-tokenizer pattern.
pub const R50K_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// GPT-3.5 / GPT-4 (`cl100k_base`) pre-tokenizer pattern.
pub const CL100K_PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
);

/// Upper bound on cached pre-token encodings.
const CACHE_CAPACITY: usize = 1 << 16;

/// Static description of a published OpenAI encoding.
#[derive(Debug, Clone, Copy)]
pub struct EncodingSpec {
    /// Encoding name, also the `{name}.tiktoken` rank-file stem.
    pub name: &'static str,
    /// Pre-tokenizer pattern.
    pub pattern: &'static str,
    /// Special tokens and their ids (part of the published encoding).
    pub special_tokens: &'static [(&'static str, usize)],
}

/// `cl100k_base` (GPT-3.5-turbo, GPT-4).
pub const CL100K_BASE: EncodingSpec = EncodingSpec {
    name: "cl100k_base",
    pattern: CL100K_PATTERN,
    special_tokens: &[
        ("<|endoftext|>", 100257),
        ("<|fim_prefix|>", 100258),
        ("<|fim_middle|>", 100259),
        ("<|fim_suffix|>", 100260),
        ("<|endofprompt|>", 100276),
    ],
};

/// `p50k_base` (Codex, `text-davinci-002/003`).
pub const P50K_BASE: EncodingSpec = EncodingSpec {
    name: "p50k_base",
    pattern: R50K_PATTERN,
    special_tokens: &[("<|endoftext|>", 50256)],
};

/// `r50k_base` (GPT-2, `text-davinci-001`).
pub const R50K_BASE: EncodingSpec = EncodingSpec {
    name: "r50k_base",
    pattern: R50K_PATTERN,
    special_tokens: &[("<|endoftext|>", 50256)],
};

impl EncodingSpec {
    fn compile_pattern(&self) -> Result<FancyRegex> {
        FancyRegex::new(self.pattern).map_err(|e| {
            TrustformersError::invalid_config(format!(
                "Failed to compile the {} pre-tokenizer pattern: {}",
                self.name, e
            ))
        })
    }

    fn special_token_map(&self) -> HashMap<String, usize> {
        self.special_tokens.iter().map(|&(token, id)| (token.to_string(), id)).collect()
    }
}

/// Tiktoken-style byte-level BPE tokenizer.
#[derive(Debug)]
pub struct TiktokenTokenizer {
    encoder: RankMap,
    decoder: HashMap<usize, Vec<u8>>,
    special_tokens: HashMap<String, usize>,
    pattern: FancyRegex,
    /// Cache keyed on **pre-token** bytes (never on whole inputs) and bounded.
    cache: RwLock<HashMap<Vec<u8>, Vec<usize>>>,
}

impl Clone for TiktokenTokenizer {
    fn clone(&self) -> Self {
        Self {
            encoder: self.encoder.clone(),
            decoder: self.decoder.clone(),
            special_tokens: self.special_tokens.clone(),
            pattern: self.pattern.clone(),
            cache: RwLock::new(HashMap::new()), // Create new cache for clone
        }
    }
}

impl TiktokenTokenizer {
    /// Create a tokenizer from an explicit rank table.
    pub fn new(
        encoder: RankMap,
        special_tokens: HashMap<String, usize>,
        pattern: Option<FancyRegex>,
    ) -> Result<Self> {
        let decoder: HashMap<usize, Vec<u8>> =
            encoder.iter().map(|(bytes, &rank)| (rank, bytes.clone())).collect();

        let pattern = match pattern {
            Some(pattern) => pattern,
            None => R50K_BASE.compile_pattern()?,
        };

        Ok(Self {
            encoder,
            decoder,
            special_tokens,
            pattern,
            cache: RwLock::new(HashMap::new()),
        })
    }

    /// Load a `.tiktoken` rank table from a reader (base64 token + rank/line).
    pub fn from_reader<R: BufRead>(reader: R) -> Result<Self> {
        let encoder = ranks::load_ranks_from_reader(reader)?;
        Self::new(encoder, HashMap::new(), None)
    }

    /// Load a `.tiktoken` rank file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let encoder = ranks::load_ranks_from_file(path)?;
        Self::new(encoder, HashMap::new(), None)
    }

    /// Attach special tokens (builder form).
    pub fn with_special_tokens(mut self, special_tokens: HashMap<String, usize>) -> Self {
        self.special_tokens = special_tokens;
        self
    }

    /// Replace the pre-tokenizer pattern (builder form).
    pub fn with_pattern(mut self, pattern: FancyRegex) -> Self {
        self.pattern = pattern;
        self
    }

    /// Replace the pre-tokenizer pattern from its source (builder form).
    pub fn with_pattern_str(self, pattern: &str) -> Result<Self> {
        let compiled = FancyRegex::new(pattern).map_err(|e| {
            TrustformersError::invalid_config(format!("Invalid pre-tokenizer pattern: {}", e))
        })?;
        Ok(self.with_pattern(compiled))
    }

    /// Build a published encoding from an already-loaded rank table.
    pub fn from_encoding_spec(spec: EncodingSpec, encoder: RankMap) -> Result<Self> {
        Self::new(
            encoder,
            spec.special_token_map(),
            Some(spec.compile_pattern()?),
        )
    }

    /// Build a published encoding, locating its rank file on disk.
    ///
    /// See [`ranks::search_paths`] for the probed locations. Returns an
    /// instructive error when no rank file is available.
    pub fn from_encoding_spec_on_disk(spec: EncodingSpec) -> Result<Self> {
        let path = ranks::find_rank_file(spec.name)
            .ok_or_else(|| ranks::missing_rank_file_error(spec.name))?;
        let encoder = ranks::load_ranks_from_file(&path)?;
        Self::from_encoding_spec(spec, encoder)
    }

    /// Look up a published encoding by name (`cl100k_base`, `p50k_base`, `r50k_base`).
    pub fn from_encoding_name(name: &str) -> Result<Self> {
        let spec = match name {
            "cl100k_base" => CL100K_BASE,
            "p50k_base" => P50K_BASE,
            "r50k_base" => R50K_BASE,
            other => {
                return Err(TrustformersError::invalid_input(format!(
                    "Unknown tiktoken encoding '{}'. Known encodings: cl100k_base, \
                     p50k_base, r50k_base. Use `TiktokenTokenizer::from_file` for \
                     any other rank table.",
                    other
                )))
            },
        };
        Self::from_encoding_spec_on_disk(spec)
    }

    /// `cl100k_base` (GPT-3.5-turbo / GPT-4), loaded from a local rank file.
    pub fn cl100k_base() -> Result<Self> {
        Self::from_encoding_spec_on_disk(CL100K_BASE)
    }

    /// `p50k_base` (Codex), loaded from a local rank file.
    pub fn p50k_base() -> Result<Self> {
        Self::from_encoding_spec_on_disk(P50K_BASE)
    }

    /// `r50k_base` (GPT-2), loaded from a local rank file.
    pub fn r50k_base() -> Result<Self> {
        Self::from_encoding_spec_on_disk(R50K_BASE)
    }

    /// Load a rank file plus an optional `token id` special-token file.
    pub fn from_tiktoken_file(
        encoder_path: &str,
        special_tokens_path: Option<&str>,
    ) -> Result<Self> {
        use std::fs::File;
        use std::io::BufReader;

        let encoder = ranks::load_ranks_from_file(encoder_path)?;

        let mut special_tokens = HashMap::new();
        if let Some(special_path) = special_tokens_path {
            let file = File::open(special_path).map_err(|e| {
                TrustformersError::io_error(format!(
                    "Failed to open special tokens file {}: {}",
                    special_path, e
                ))
            })?;

            for (index, line) in BufReader::new(file).lines().enumerate() {
                let line_number = index + 1;
                let line = line.map_err(|e| {
                    TrustformersError::io_error(format!(
                        "Failed to read special tokens line {}: {}",
                        line_number, e
                    ))
                })?;
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }

                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() != 2 {
                    return Err(TrustformersError::invalid_input(format!(
                        "Invalid special token format at line {}: expected '<token> <id>'",
                        line_number
                    )));
                }

                let id: usize = parts[1].parse().map_err(|e| {
                    TrustformersError::invalid_input(format!(
                        "Invalid special token id at line {}: {}",
                        line_number, e
                    ))
                })?;
                special_tokens.insert(parts[0].to_string(), id);
            }
        }

        Self::new(encoder, special_tokens, None)
    }

    /// Merge one pre-token's bytes by BPE rank (the tiktoken algorithm).
    ///
    /// `parts[i] = (byte offset, rank of merging this part with the next)`. The
    /// lowest-ranked merge is applied repeatedly and only the two neighbouring
    /// ranks are repaired, which is the standard incremental formulation.
    fn byte_pair_merge(&self, piece: &[u8]) -> Vec<(usize, usize)> {
        let mut parts: Vec<(usize, usize)> = (0..=piece.len()).map(|i| (i, usize::MAX)).collect();

        let rank_of = |parts: &Vec<(usize, usize)>, start: usize, skip: usize| -> usize {
            if start + skip + 2 < parts.len() {
                let range = parts[start].0..parts[start + skip + 2].0;
                self.encoder.get(&piece[range]).copied().unwrap_or(usize::MAX)
            } else {
                usize::MAX
            }
        };

        for i in 0..parts.len().saturating_sub(2) {
            parts[i].1 = rank_of(&parts, i, 0);
        }

        while parts.len() > 1 {
            let mut best_rank = usize::MAX;
            let mut best_index = 0usize;

            for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
                if rank < best_rank {
                    best_rank = rank;
                    best_index = i;
                }
            }

            if best_rank == usize::MAX {
                break;
            }

            parts[best_index].1 = rank_of(&parts, best_index, 1);
            if best_index > 0 {
                parts[best_index - 1].1 = rank_of(&parts, best_index - 1, 1);
            }
            parts.remove(best_index + 1);
        }

        parts
    }

    /// Encode one pre-token's bytes into token ids.
    fn encode_piece(&self, piece: &[u8]) -> Result<Vec<usize>> {
        if piece.is_empty() {
            return Ok(Vec::new());
        }

        if let Some(&rank) = self.encoder.get(piece) {
            return Ok(vec![rank]);
        }

        if let Ok(cache) = self.cache.read() {
            if let Some(cached) = cache.get(piece) {
                return Ok(cached.clone());
            }
        }

        let parts = self.byte_pair_merge(piece);
        let mut tokens = Vec::with_capacity(parts.len().saturating_sub(1));

        for window in parts.windows(2) {
            let range = window[0].0..window[1].0;
            let bytes = &piece[range];
            let rank = self.encoder.get(bytes).copied().ok_or_else(|| {
                TrustformersError::invalid_config(format!(
                    "BPE rank table is incomplete: no rank for the byte sequence {:?} \
                     (a valid tiktoken table contains every single byte)",
                    bytes
                ))
            })?;
            tokens.push(rank);
        }

        if let Ok(mut cache) = self.cache.write() {
            if cache.len() >= CACHE_CAPACITY {
                cache.clear();
            }
            cache.insert(piece.to_vec(), tokens.clone());
        }

        Ok(tokens)
    }

    /// Encode text, treating special-token text as ordinary text.
    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<usize>> {
        let mut tokens = Vec::new();

        for matched in self.pattern.find_iter(text) {
            let m = matched.map_err(|e| {
                TrustformersError::other(format!("Pre-tokenizer regex failed: {}", e))
            })?;
            tokens.extend(self.encode_piece(m.as_str().as_bytes())?);
        }

        Ok(tokens)
    }

    /// Encode text, recognizing **all** of the tokenizer's special tokens.
    ///
    /// Special tokens are matched leftmost-longest in a single pass, so the
    /// result never depends on hash-map iteration order.
    pub fn encode_text(&self, text: &str) -> Result<Vec<usize>> {
        self.encode_scanning_specials(text, None)
    }

    /// Encode text, recognizing only the special tokens in `allowed_special`.
    ///
    /// Names in `allowed_special` that this tokenizer does not know are ignored
    /// (there is no id to emit for them); the surrounding text is still encoded
    /// normally and the scan always makes progress.
    pub fn encode_with_special_tokens(
        &self,
        text: &str,
        allowed_special: &HashSet<String>,
    ) -> Result<Vec<usize>> {
        self.encode_scanning_specials(text, Some(allowed_special))
    }

    fn encode_scanning_specials(
        &self,
        text: &str,
        allowed_special: Option<&HashSet<String>>,
    ) -> Result<Vec<usize>> {
        if self.special_tokens.is_empty() {
            return self.encode_ordinary(text);
        }

        let mut result = Vec::new();
        let mut segment_start = 0usize;
        let mut cursor = 0usize;

        while cursor < text.len() {
            if !text.is_char_boundary(cursor) {
                cursor += 1;
                continue;
            }

            let mut best: Option<(&str, usize)> = None;
            for (token, &id) in &self.special_tokens {
                if let Some(allowed) = allowed_special {
                    if !allowed.contains(token) {
                        continue;
                    }
                }
                if !text[cursor..].starts_with(token.as_str()) {
                    continue;
                }
                // Leftmost-longest: at a given position the longest match wins.
                if best.map(|(current, _)| token.len() > current.len()).unwrap_or(true) {
                    best = Some((token.as_str(), id));
                }
            }

            match best {
                Some((token, id)) => {
                    if segment_start < cursor {
                        result.extend(self.encode_ordinary(&text[segment_start..cursor])?);
                    }
                    result.push(id);
                    cursor += token.len();
                    segment_start = cursor;
                },
                None => cursor += 1,
            }
        }

        if segment_start < text.len() {
            result.extend(self.encode_ordinary(&text[segment_start..])?);
        }

        Ok(result)
    }

    /// Decode token ids to raw bytes.
    pub fn decode_bytes(&self, tokens: &[usize]) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();

        for &token_id in tokens {
            if let Some(special) = self.special_token_text(token_id) {
                bytes.extend_from_slice(special.as_bytes());
                continue;
            }

            let token_bytes = self.decoder.get(&token_id).ok_or_else(|| {
                TrustformersError::invalid_input(format!(
                    "Token id {} is not present in this encoding",
                    token_id
                ))
            })?;
            bytes.extend_from_slice(token_bytes);
        }

        Ok(bytes)
    }

    /// Decode token IDs to text.
    pub fn decode_tokens(&self, tokens: &[usize]) -> Result<String> {
        let bytes = self.decode_bytes(tokens)?;
        String::from_utf8(bytes)
            .map_err(|e| TrustformersError::other(format!("Failed to decode UTF-8: {}", e)))
    }

    fn special_token_text(&self, token_id: usize) -> Option<&str> {
        self.special_tokens
            .iter()
            .find(|(_, &id)| id == token_id)
            .map(|(token, _)| token.as_str())
    }

    /// Number of ranked byte tokens plus special tokens.
    pub fn vocab_size(&self) -> usize {
        self.encoder.len() + self.special_tokens.len()
    }

    /// Get special tokens
    pub fn special_tokens(&self) -> &HashMap<String, usize> {
        &self.special_tokens
    }

    /// The BPE rank table.
    pub fn encoder(&self) -> &RankMap {
        &self.encoder
    }

    /// Check if a token ID is a special token
    pub fn is_special_token(&self, token_id: usize) -> bool {
        self.special_tokens.values().any(|&id| id == token_id)
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let tokens = self.encode_text(text)?;

        let input_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        let attention_mask = vec![1u8; input_ids.len()];
        let special_tokens_mask: Vec<u8> =
            tokens.iter().map(|&t| u8::from(self.is_special_token(t))).collect();

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
        // Tiktoken encodings have no sequence-pair convention; join with a space.
        let combined = format!("{} {}", text, text2);
        self.encode(&combined)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let tokens: Vec<usize> = ids.iter().map(|&id| id as usize).collect();
        self.decode_tokens(&tokens)
    }

    fn vocab_size(&self) -> usize {
        TiktokenTokenizer::vocab_size(self)
    }

    /// Vocabulary keyed on the GPT-2 byte-level alphabet, so every rank entry —
    /// including byte sequences that are not valid UTF-8 — has a distinct,
    /// reversible string form. Built on demand; this is `O(vocab)`.
    fn get_vocab(&self) -> HashMap<String, u32> {
        let mut vocab: HashMap<String, u32> = self
            .encoder
            .iter()
            .map(|(bytes, &rank)| {
                (
                    bytes.iter().map(|&b| byte_to_unicode(b)).collect(),
                    rank as u32,
                )
            })
            .collect();

        for (token, &id) in &self.special_tokens {
            vocab.insert(token.clone(), id as u32);
        }

        vocab
    }

    /// Resolve a token string to its id.
    ///
    /// Special tokens are checked first, then the token's raw UTF-8 bytes, then
    /// its GPT-2 byte-alphabet reading (which is how non-UTF-8 pieces are
    /// spelled by [`Tokenizer::get_vocab`]).
    fn token_to_id(&self, token: &str) -> Option<u32> {
        if let Some(&id) = self.special_tokens.get(token) {
            return Some(id as u32);
        }

        if let Some(&rank) = self.encoder.get(token.as_bytes()) {
            return Some(rank as u32);
        }

        let decoded: Option<Vec<u8>> = token.chars().map(unicode_to_byte).collect();
        decoded.and_then(|bytes| self.encoder.get(&bytes).map(|&rank| rank as u32))
    }

    /// The byte-alphabet spelling of a token id (inverse of [`Self::token_to_id`]).
    fn id_to_token(&self, id: u32) -> Option<String> {
        let id = id as usize;
        if let Some(special) = self.special_token_text(id) {
            return Some(special.to_string());
        }

        self.decoder
            .get(&id)
            .map(|bytes| bytes.iter().map(|&b| byte_to_unicode(b)).collect())
    }
}

#[cfg(test)]
mod tests;
