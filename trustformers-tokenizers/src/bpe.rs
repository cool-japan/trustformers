//! Byte-level BPE (GPT-2 / RoBERTa style) tokenizer.
//!
//! The implementation follows the reference GPT-2 encoder:
//!
//! 1. text is split with the exact GPT-2 pre-tokenizer pattern (which needs a
//!    lookahead, hence `fancy_regex`),
//! 2. each pre-token is mapped byte-by-byte through the GPT-2
//!    `bytes_to_unicode` alphabet,
//! 3. merges are applied in rank order until no ranked pair remains.

use crate::offsets::{
    aligned_lowercase, aligned_nfc, ceil_char_boundary, floor_char_boundary, AlignmentBuilder,
    OffsetAlignment,
};
use crate::vocab::Vocab;
use fancy_regex::Regex as FancyRegex;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::RwLock;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Separator used to key merge ranks by a single joined string.
///
/// Byte-level BPE symbols only ever contain characters from the
/// `bytes_to_unicode` alphabet (`U+0021..=U+00FF` minus a few holes, plus
/// `U+0100..=U+01FF`), so `U+0001` can never occur inside a symbol and is a
/// safe, allocation-free separator for `(first, second)` lookups.
const MERGE_KEY_SEPARATOR: char = '\u{1}';

/// Upper bound on the number of cached pre-token BPE results.
///
/// The cache is keyed on arbitrary input substrings, so it must be bounded or a
/// long-running service leaks one entry per distinct pre-token. When the bound
/// is hit the cache is cleared wholesale (cheap, and BPE results are trivially
/// recomputable).
const BPE_CACHE_CAPACITY: usize = 1 << 16;

/// Words shorter than this many bytes are cheaper to re-merge than to look up
/// under the shared lock, so they bypass the cache entirely.
const BPE_CACHE_MIN_LEN: usize = 4;

/// Exact GPT-2 byte-level BPE pre-tokenizer pattern.
///
/// `\s+(?!\S)` is load-bearing: it makes a run of whitespace that is followed by
/// a non-space character give its *last* space to the following pre-token, which
/// is what produces the familiar `Ġword` pieces. `regex` cannot express the
/// lookahead, so the pure-Rust backtracking engine `fancy_regex` is used.
static GPT2_PATTERN: Lazy<FancyRegex> = Lazy::new(|| {
    // reason: compile-time-constant pattern; a `static` initializer has no
    // fallible channel to propagate an error through.
    FancyRegex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        .expect("built-in GPT-2 byte-level BPE regex must compile")
});

/// GPT-2 `bytes_to_unicode()` table: byte value -> printable character.
static BYTE_ENCODER: Lazy<[char; 256]> = Lazy::new(build_byte_encoder);

/// Inverse of [`BYTE_ENCODER`], indexed by code point (all entries are < 512).
static BYTE_DECODER: Lazy<[Option<u8>; 512]> = Lazy::new(build_byte_decoder);

fn build_byte_encoder() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next_code_point = 256u32;

    for (byte, slot) in table.iter_mut().enumerate() {
        let byte = byte as u8;
        if (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174 {
            *slot = byte as char;
        } else {
            // reason: `next_code_point` stays inside 256..=511, a range that
            // contains no surrogate code points, so `from_u32` is always `Some`.
            *slot = char::from_u32(next_code_point)
                .expect("code points 256..=511 are valid Unicode scalar values");
            next_code_point += 1;
        }
    }

    table
}

fn build_byte_decoder() -> [Option<u8>; 512] {
    let mut table = [None; 512];
    let encoder = build_byte_encoder();
    for (b, &ch) in encoder.iter().enumerate() {
        let code_point = ch as usize;
        debug_assert!(code_point < 512, "byte-level alphabet stays below U+0200");
        if code_point < 512 {
            table[code_point] = Some(b as u8);
        }
    }
    table
}

/// Map a single byte to its GPT-2 byte-level alphabet character.
#[inline]
pub fn byte_to_unicode(byte: u8) -> char {
    BYTE_ENCODER[byte as usize]
}

/// Map a GPT-2 byte-level alphabet character back to its byte.
#[inline]
pub fn unicode_to_byte(ch: char) -> Option<u8> {
    let code_point = ch as usize;
    if code_point < 512 {
        BYTE_DECODER[code_point]
    } else {
        None
    }
}

/// Encode a raw byte slice into the GPT-2 byte-level alphabet.
pub fn bytes_to_unicode_string(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| byte_to_unicode(b)).collect()
}

#[derive(Debug)]
pub struct BPETokenizer {
    vocab: Vocab,
    merges: Vec<(String, String)>,
    /// `first \u{1} second -> rank`, so a merge lookup needs no allocation.
    merge_ranks: HashMap<String, usize>,
    unk_token: String,
    pad_token: String,
    bos_token: String,
    eos_token: String,
    cache: RwLock<HashMap<String, Vec<String>>>,
    // Enhanced byte-level BPE features
    normalize_unicode: bool,
    preserve_case: bool,
    handle_chinese_chars: bool,
    max_input_chars_per_word: usize,
}

impl Clone for BPETokenizer {
    fn clone(&self) -> Self {
        Self {
            vocab: self.vocab.clone(),
            merges: self.merges.clone(),
            merge_ranks: self.merge_ranks.clone(),
            unk_token: self.unk_token.clone(),
            pad_token: self.pad_token.clone(),
            bos_token: self.bos_token.clone(),
            eos_token: self.eos_token.clone(),
            cache: RwLock::new(HashMap::new()), // Create new cache for clone
            normalize_unicode: self.normalize_unicode,
            preserve_case: self.preserve_case,
            handle_chinese_chars: self.handle_chinese_chars,
            max_input_chars_per_word: self.max_input_chars_per_word,
        }
    }
}

impl BPETokenizer {
    /// Create a byte-level BPE tokenizer.
    ///
    /// Byte-level BPE (GPT-2 / RoBERTa) is case sensitive and must not alter the
    /// byte stream, so case folding and CJK space padding are **off** by
    /// default. Use [`BPETokenizer::with_options`] to opt into them.
    pub fn new(vocab: HashMap<String, u32>, merges: Vec<(String, String)>) -> Self {
        let merge_ranks = Self::build_merge_ranks(&merges);

        Self {
            vocab: Vocab::from_map(vocab),
            merges,
            merge_ranks,
            unk_token: "<|endoftext|>".to_string(),
            pad_token: "<|endoftext|>".to_string(),
            bos_token: "<|endoftext|>".to_string(),
            eos_token: "<|endoftext|>".to_string(),
            cache: RwLock::new(HashMap::new()),
            normalize_unicode: true,
            preserve_case: true,
            handle_chinese_chars: false,
            max_input_chars_per_word: 100,
        }
    }

    fn build_merge_ranks(merges: &[(String, String)]) -> HashMap<String, usize> {
        let mut merge_ranks = HashMap::with_capacity(merges.len());
        for (rank, (first, second)) in merges.iter().enumerate() {
            merge_ranks.insert(Self::merge_key(first, second), rank);
        }
        merge_ranks
    }

    fn merge_key(first: &str, second: &str) -> String {
        let mut key = String::with_capacity(first.len() + second.len() + 1);
        key.push_str(first);
        key.push(MERGE_KEY_SEPARATOR);
        key.push_str(second);
        key
    }

    /// Create a new BPE tokenizer with custom options
    pub fn with_options(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
        normalize_unicode: bool,
        preserve_case: bool,
        handle_chinese_chars: bool,
        max_input_chars_per_word: usize,
    ) -> Self {
        let mut tokenizer = Self::new(vocab, merges);
        tokenizer.normalize_unicode = normalize_unicode;
        tokenizer.preserve_case = preserve_case;
        tokenizer.handle_chinese_chars = handle_chinese_chars;
        tokenizer.max_input_chars_per_word = max_input_chars_per_word;
        tokenizer
    }

    /// Override the special tokens (used when restoring a saved tokenizer).
    pub fn with_special_tokens(
        mut self,
        unk_token: String,
        pad_token: String,
        bos_token: String,
        eos_token: String,
    ) -> Self {
        self.unk_token = unk_token;
        self.pad_token = pad_token;
        self.bos_token = bos_token;
        self.eos_token = eos_token;
        self
    }

    /// Get the vocabulary
    pub fn get_vocab_ref(&self) -> &Vocab {
        &self.vocab
    }

    /// Get the merge rules
    pub fn get_merge_rules(&self) -> &Vec<(String, String)> {
        &self.merges
    }

    /// Get the vocabulary mapping
    pub fn get_vocab_map(&self) -> &HashMap<String, u32> {
        self.vocab.get_token_to_id_map()
    }

    /// Whether Unicode (NFC) normalization is applied before tokenization.
    pub fn normalizes_unicode(&self) -> bool {
        self.normalize_unicode
    }

    /// Whether the original casing is preserved (required for byte-level BPE).
    pub fn preserves_case(&self) -> bool {
        self.preserve_case
    }

    /// Whether CJK characters get space padding before tokenization.
    pub fn handles_chinese_chars(&self) -> bool {
        self.handle_chinese_chars
    }

    /// Maximum number of characters processed as a single word.
    pub fn max_input_chars_per_word(&self) -> usize {
        self.max_input_chars_per_word
    }

    /// The unknown-token string.
    pub fn unk_token(&self) -> &str {
        &self.unk_token
    }

    /// The padding-token string.
    pub fn pad_token(&self) -> &str {
        &self.pad_token
    }

    /// The beginning-of-sequence token string.
    pub fn bos_token(&self) -> &str {
        &self.bos_token
    }

    /// The end-of-sequence token string.
    pub fn eos_token(&self) -> &str {
        &self.eos_token
    }

    pub fn from_files(vocab_path: &str, merges_path: &str) -> Result<Self> {
        let vocab = Self::load_vocab_from_file(vocab_path)?;
        let merges = Self::load_merges_from_file(merges_path)?;
        Ok(Self::new(vocab, merges))
    }

    fn load_vocab_from_file(vocab_path: &str) -> Result<HashMap<String, u32>> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(vocab_path).map_err(|e| {
            TrustformersError::io_error(format!("Failed to open vocab file {}: {}", vocab_path, e))
        })?;
        let reader = BufReader::new(file);

        let mut vocab = HashMap::new();
        for (id, line) in reader.lines().enumerate() {
            let token = line
                .map_err(|e| TrustformersError::io_error(format!("Failed to read line: {}", e)))?;
            let token = token.trim().to_string();
            if !token.is_empty() {
                vocab.insert(token, id as u32);
            }
        }

        if vocab.is_empty() {
            return Err(TrustformersError::other(
                "Empty vocabulary file".to_string(),
            ));
        }

        Ok(vocab)
    }

    fn load_merges_from_file(merges_path: &str) -> Result<Vec<(String, String)>> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(merges_path).map_err(|e| {
            TrustformersError::io_error(format!(
                "Failed to open merges file {}: {}",
                merges_path, e
            ))
        })?;
        let reader = BufReader::new(file);

        let mut merges = Vec::new();
        for line in reader.lines() {
            let line = line
                .map_err(|e| TrustformersError::io_error(format!("Failed to read line: {}", e)))?;
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse merge rule: "token1 token2"
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                merges.push((parts[0].to_string(), parts[1].to_string()));
            }
        }

        Ok(merges)
    }

    pub fn from_roberta_files(vocab_path: &str, merges_path: &str) -> Result<Self> {
        let vocab = Self::load_vocab_from_file(vocab_path)?;
        let merges = Self::load_merges_from_file(merges_path)?;

        // Create RoBERTa-specific BPE tokenizer with RoBERTa-specific settings
        let mut tokenizer = Self::new(vocab, merges);

        // RoBERTa-specific configuration
        tokenizer.unk_token = "<unk>".to_string();
        tokenizer.pad_token = "<pad>".to_string();
        tokenizer.bos_token = "<s>".to_string();
        tokenizer.eos_token = "</s>".to_string();

        Ok(tokenizer)
    }

    /// Normalize text for improved Unicode handling.
    ///
    /// Public for inspection and testing, like [`Self::pre_tokenize`]. It is a
    /// projection of [`Self::aligned_normalize_text`], the single
    /// implementation the encoder runs, so it can never describe a
    /// normalization the encoder does not perform.
    pub fn normalize_text(&self, text: &str) -> String {
        self.aligned_normalize_text(text).0
    }

    /// [`Self::normalize_text`] plus a byte alignment from the normalized
    /// string back into `text`.
    ///
    /// Each configured stage — NFC, case folding, CJK space padding — is
    /// aligned individually and the alignments are composed, so a token found
    /// in the normalized string can always be reported against the caller's
    /// original bytes. When a stage changes nothing (the common case: ASCII, or
    /// text that is already NFC with `preserve_case`) its alignment is the
    /// zero-cost identity.
    fn aligned_normalize_text(&self, text: &str) -> (String, OffsetAlignment) {
        if !self.normalize_unicode {
            return (text.to_string(), OffsetAlignment::identity(text));
        }

        // Apply Unicode normalization (NFC form)
        let (normalized, normalized_alignment) = aligned_nfc(text);

        // Handle case normalization if needed
        let (case_normalized, case_alignment) = if self.preserve_case {
            (normalized, normalized_alignment)
        } else {
            let (lowered, lower_alignment) = aligned_lowercase(&normalized);
            (lowered, lower_alignment.rebase(&normalized_alignment))
        };

        // Handle Chinese characters specially if enabled
        if self.handle_chinese_chars {
            let (padded, pad_alignment) = self.aligned_handle_chinese_text(&case_normalized);
            (padded, pad_alignment.rebase(&case_alignment))
        } else {
            (case_normalized, case_alignment)
        }
    }

    /// Special handling for Chinese characters, with a byte alignment back into
    /// `text`.
    ///
    /// The separator spaces are insertions with no source of their own; the
    /// pre-tokenizer consumes them and they never end up inside a token.
    fn aligned_handle_chinese_text(&self, text: &str) -> (String, OffsetAlignment) {
        // Add spaces around Chinese characters for better tokenization
        let mut result = String::new();
        let mut alignment = AlignmentBuilder::new();
        let mut prev_was_chinese = false;

        for (index, ch) in text.char_indices() {
            let is_chinese = self.is_chinese_char(ch);

            // Add space when transitioning between Chinese and non-Chinese text
            if (is_chinese != prev_was_chinese) && !result.is_empty() && !result.ends_with(' ') {
                result.push(' ');
                alignment.skip_output(1);
            }

            result.push(ch);
            alignment.push(ch.len_utf8(), index, index + ch.len_utf8());
            prev_was_chinese = is_chinese;
        }

        (result, alignment.finish(text.len()))
    }

    /// Check if a character is a Chinese character
    fn is_chinese_char(&self, ch: char) -> bool {
        let cp = ch as u32;
        // CJK Unified Ideographs and related ranges
        (0x4E00..=0x9FFF).contains(&cp) ||   // CJK Unified Ideographs
        (0x3400..=0x4DBF).contains(&cp) ||   // CJK Extension A
        (0x20000..=0x2A6DF).contains(&cp) || // CJK Extension B
        (0x2A700..=0x2B73F).contains(&cp) || // CJK Extension C
        (0x2B740..=0x2B81F).contains(&cp) || // CJK Extension D
        (0x2B820..=0x2CEAF).contains(&cp) || // CJK Extension E
        (0xF900..=0xFAFF).contains(&cp) ||   // CJK Compatibility Ideographs
        (0x2F800..=0x2FA1F).contains(&cp) // CJK Compatibility Supplement
    }

    /// Byte spans of the GPT-2 pre-tokens of `text`.
    ///
    /// If the backtracking engine ever bails out (backtrack limit), the
    /// remaining suffix is returned as one span instead of being dropped, so no
    /// input is ever silently lost.
    fn pre_token_spans(text: &str) -> Vec<(usize, usize)> {
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for matched in GPT2_PATTERN.find_iter(text) {
            match matched {
                Ok(m) => spans.push((m.start(), m.end())),
                Err(_) => {
                    let resume = spans.last().map(|&(_, end)| end).unwrap_or(0);
                    if resume < text.len() {
                        spans.push((resume, text.len()));
                    }
                    break;
                },
            }
        }
        spans
    }

    /// Split `text` into GPT-2 pre-tokens (public for testing and inspection).
    pub fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        Self::pre_token_spans(text).into_iter().map(|(s, e)| &text[s..e]).collect()
    }

    /// Apply the BPE merge sequence to one (already normalized) pre-token.
    ///
    /// The input is interpreted as raw UTF-8 bytes mapped through the GPT-2
    /// byte-level alphabet; no further normalization happens here (the caller
    /// normalizes once for the whole text).
    fn bpe(&self, token: &str) -> Vec<String> {
        if token.is_empty() {
            return vec![];
        }

        let cacheable = token.len() >= BPE_CACHE_MIN_LEN;
        if cacheable {
            if let Ok(cache) = self.cache.read() {
                if let Some(cached) = cache.get(token) {
                    return cached.clone();
                }
            }
        }

        // Limit input length to prevent excessive processing
        if token.chars().count() > self.max_input_chars_per_word {
            // For very long tokens, split into chunks
            let chunks: Vec<String> = token
                .chars()
                .collect::<Vec<_>>()
                .chunks(self.max_input_chars_per_word)
                .map(|chunk| chunk.iter().collect())
                .collect();

            let mut result = vec![];
            for chunk in chunks {
                result.extend(self.bpe(&chunk));
            }
            return result;
        }

        // Every BPE symbol is a contiguous run of the byte-encoded word, so the
        // whole merge loop runs over `(start, end)` spans of one buffer: one
        // allocation for the buffer, none per input byte and none per candidate
        // pair. Only the final pieces become owned `String`s.
        let encoded: String = token.as_bytes().iter().map(|&b| byte_to_unicode(b)).collect();
        let mut symbols: Vec<(usize, usize)> = encoded
            .char_indices()
            .map(|(offset, ch)| (offset, offset + ch.len_utf8()))
            .collect();

        if symbols.len() > 1 {
            self.apply_merges(&encoded, &mut symbols);
        }

        let word: Vec<String> =
            symbols.iter().map(|&(start, end)| encoded[start..end].to_string()).collect();

        if cacheable {
            if let Ok(mut cache) = self.cache.write() {
                if cache.len() >= BPE_CACHE_CAPACITY {
                    cache.clear();
                }
                cache.insert(token.to_string(), word.clone());
            }
        }

        word
    }

    /// Repeatedly merge the lowest-ranked adjacent pair, in place.
    ///
    /// `symbols` are `(start, end)` byte spans of `encoded`, which tile it
    /// completely and in order; merging two adjacent symbols is therefore just
    /// joining two adjacent spans. Nothing is allocated per candidate pair: the
    /// lookup key is built into one reusable buffer and the spans are `Copy`.
    fn apply_merges(&self, encoded: &str, symbols: &mut Vec<(usize, usize)>) {
        if self.merge_ranks.is_empty() {
            return;
        }

        let mut key = String::new();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(symbols.len());

        while symbols.len() > 1 {
            let mut best_rank = usize::MAX;
            let mut best_index: Option<usize> = None;

            for i in 0..symbols.len() - 1 {
                key.clear();
                key.push_str(&encoded[symbols[i].0..symbols[i].1]);
                key.push(MERGE_KEY_SEPARATOR);
                key.push_str(&encoded[symbols[i + 1].0..symbols[i + 1].1]);

                if let Some(&rank) = self.merge_ranks.get(key.as_str()) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_index = Some(i);
                    }
                }
            }

            let Some(first_index) = best_index else {
                break;
            };

            // Merge *every* occurrence of the winning pair in one pass, exactly
            // like the reference GPT-2 implementation.
            let (first_start, first_end) = symbols[first_index];
            let (second_start, second_end) = symbols[first_index + 1];
            let first_text = &encoded[first_start..first_end];
            let second_text = &encoded[second_start..second_end];

            merged.clear();
            let mut i = 0;
            while i < symbols.len() {
                let matches_pair = i + 1 < symbols.len()
                    && &encoded[symbols[i].0..symbols[i].1] == first_text
                    && &encoded[symbols[i + 1].0..symbols[i + 1].1] == second_text;

                if matches_pair {
                    merged.push((symbols[i].0, symbols[i + 1].1));
                    i += 2;
                } else {
                    merged.push(symbols[i]);
                    i += 1;
                }
            }

            std::mem::swap(symbols, &mut merged);
        }
    }

    /// Tokenize `text` into byte-level BPE string tokens.
    ///
    /// A projection of [`Self::tokenize_with_offsets`], so the token sequence
    /// the encoder emits and the token sequence the offsets describe are the
    /// same sequence by construction, not by coincidence.
    ///
    /// Public for parity with [`crate::wordpiece::WordPieceTokenizer::tokenize`].
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.tokenize_with_offsets(text).0
    }

    /// Tokenization with byte offsets into the **original** text.
    ///
    /// Normalization runs once over the whole string (exactly as
    /// [`Self::normalize_text`] describes it) and the pre-tokenizer runs on the
    /// normalized result, which is what fixes the token sequence; every span is
    /// then mapped back through the normalization alignment, so the returned
    /// offsets index the caller's original bytes even when normalization
    /// changed lengths. See [`crate::offsets`] for the byte-offset convention
    /// and for [`crate::offsets::byte_offsets_to_char_offsets`], which converts
    /// these to the character offsets a Python caller needs.
    ///
    /// Within a pre-token, each BPE piece covers exactly as many bytes as it
    /// has symbol characters (one byte-level symbol == one source byte), and
    /// those exact byte positions are what the cursor advances by — the
    /// reported span is only widened outward to the enclosing character
    /// boundaries so that `&text[start..end]` never panics. A piece that splits
    /// a multi-byte character therefore reports the whole character (as
    /// HuggingFace does) without shifting the pieces that follow it.
    ///
    /// The two returned vectors always have the same length.
    pub fn tokenize_with_offsets(&self, text: &str) -> (Vec<String>, Vec<(usize, usize)>) {
        let (normalized, alignment) = self.aligned_normalize_text(text);

        let mut tokens = vec![];
        let mut offsets = vec![];

        for (start, end) in Self::pre_token_spans(&normalized) {
            let pieces = self.bpe(&normalized[start..end]);

            // Exact byte cursor into `normalized`: never adjusted for character
            // boundaries, so the spans of successive pieces stay perfectly
            // tiled before they are widened and mapped back.
            let mut cursor = start;

            for piece in pieces {
                let piece_end = (cursor + piece.chars().count()).min(end);
                let span = alignment.map_span(
                    floor_char_boundary(&normalized, cursor),
                    ceil_char_boundary(&normalized, piece_end),
                );
                cursor = piece_end;

                tokens.push(piece);
                offsets.push(span);
            }
        }

        (tokens, offsets)
    }

    /// The single string [`Tokenizer::encode_pair`] actually encodes.
    ///
    /// Byte-level BPE has no separator token of its own, so this tokenizer's
    /// pair encoding is the two sequences joined by one space and encoded as
    /// one sequence. Exposed because `encode_pair`'s `offset_mapping` indexes
    /// *this* string — the only coordinate space in which those offsets mean
    /// anything — so a caller that wants to slice by them can reconstruct it.
    pub fn pair_input_text(text: &str, text2: &str) -> String {
        format!("{} {}", text, text2)
    }
}

impl Tokenizer for BPETokenizer {
    /// Encode `text`. Byte-level BPE adds no special tokens, so every position
    /// is a content token.
    ///
    /// `offset_mapping` is always populated with byte spans into `text` — see
    /// [`BPETokenizer::tokenize_with_offsets`], which produces the ids and the
    /// spans in one pass.
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let (tokens, offsets) = self.tokenize_with_offsets(text);

        let input_ids: Vec<u32> = tokens
            .iter()
            .map(|token| {
                self.vocab.get_id(token).unwrap_or_else(|| {
                    self.vocab.get_id(&self.unk_token).unwrap_or(0) // Fallback to 0 if unk_token not found
                })
            })
            .collect();

        let attention_mask = vec![1u8; input_ids.len()];

        Ok(TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: Some(offsets),
            overflowing_tokens: None,
        })
    }

    /// Encode a sequence pair by joining the two sequences with a single space
    /// and encoding the result as one sequence — byte-level BPE has no
    /// separator token to place between them.
    ///
    /// Consequently `offset_mapping` indexes that joined string, **not** `text`
    /// or `text2`: build it with [`BPETokenizer::pair_input_text`] to slice by
    /// these offsets. Callers that need per-sequence offsets should encode the
    /// two sequences separately, or use a tokenizer whose pair encoding has a
    /// real sequence boundary (`WordPieceTokenizer::encode_pair`).
    fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        let combined = Self::pair_input_text(text, text2);
        self.encode(&combined)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let tokens: Vec<String> = ids.iter().filter_map(|&id| self.vocab.get_token(id)).collect();

        // Join tokens and decode bytes
        let text = tokens.join("");
        let mut bytes = Vec::with_capacity(text.len());

        for ch in text.chars() {
            if let Some(byte) = unicode_to_byte(ch) {
                bytes.push(byte);
            }
        }

        String::from_utf8(bytes)
            .map_err(|e| TrustformersError::other(format!("Failed to decode bytes: {}", e)))
    }

    fn vocab_size(&self) -> usize {
        self.vocab.size()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.get_token_to_id_map().clone()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get_token(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_tokenizer() -> BPETokenizer {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("the".to_string(), 2);
        vocab.insert("Ġ".to_string(), 3);
        vocab.insert("<|endoftext|>".to_string(), 4);

        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("l".to_string(), "l".to_string()),
            ("o".to_string(), "w".to_string()),
        ];

        BPETokenizer::new(vocab, merges)
    }

    /// Round-trip and pin the GPT-2 `bytes_to_unicode` table.
    #[test]
    fn test_byte_encoder_table_is_gpt2_bytes_to_unicode() {
        let mut seen = std::collections::HashSet::new();
        for b in 0..=255u8 {
            let ch = byte_to_unicode(b);
            assert!(seen.insert(ch), "byte {} produced a duplicate character", b);
            assert_eq!(
                unicode_to_byte(ch),
                Some(b),
                "byte {} did not round-trip through the byte alphabet",
                b
            );
        }
        assert_eq!(seen.len(), 256);

        // Canonical anchors of the GPT-2 table.
        assert_eq!(byte_to_unicode(b' '), '\u{0120}'); // 'Ġ'
        assert_eq!(byte_to_unicode(0), '\u{0100}'); // 'Ā'
        assert_eq!(byte_to_unicode(b'a'), 'a');
        assert_eq!(byte_to_unicode(b'\n'), '\u{010a}');
    }

    /// Byte-level round-trip: every byte string maps to symbols and back.
    #[test]
    fn test_bytes_to_unicode_string_round_trip() {
        let samples: [&[u8]; 4] = [
            b"Hello, world!",
            b"\xf0\x9f\x98\x80",
            b"\x00\x01\x02",
            b" \t\n",
        ];
        for sample in samples {
            let encoded = bytes_to_unicode_string(sample);
            let decoded: Vec<u8> = encoded.chars().filter_map(unicode_to_byte).collect::<Vec<u8>>();
            assert_eq!(decoded.as_slice(), sample);
        }
    }

    /// Regression: the old pattern lacked `\s+(?!\S)`, so a whitespace run was
    /// consumed whole and the following word never got its leading space.
    #[test]
    fn test_gpt2_pattern_gives_trailing_space_to_next_word() {
        let tokenizer = create_test_tokenizer();

        assert_eq!(tokenizer.pre_tokenize("a   b"), vec!["a", "  ", " b"]);
        assert_eq!(
            tokenizer.pre_tokenize("hello world"),
            vec!["hello", " world"]
        );
        // Trailing whitespace at end of input is kept as a single run.
        assert_eq!(tokenizer.pre_tokenize("hi   "), vec!["hi", "   "]);
    }

    /// Regression: `new()` used to lowercase everything, corrupting byte-level BPE.
    #[test]
    fn test_new_preserves_case_and_bytes() {
        let tokenizer = BPETokenizer::new(HashMap::new(), Vec::new());
        assert!(tokenizer.preserves_case());
        assert!(!tokenizer.handles_chinese_chars());

        let text = "Hello WORLD";
        let tokens = tokenizer.tokenize(text);
        let joined: String = tokens.concat();
        let bytes: Vec<u8> = joined.chars().filter_map(unicode_to_byte).collect();
        assert_eq!(
            String::from_utf8(bytes).expect("byte-level BPE must round-trip UTF-8"),
            text
        );
    }

    /// CJK input must not be silently space-padded by the default constructor.
    #[test]
    fn test_new_does_not_pad_cjk() {
        let tokenizer = BPETokenizer::new(HashMap::new(), Vec::new());
        let text = "hello世界world";
        let tokens = tokenizer.tokenize(text);
        let joined: String = tokens.concat();
        let bytes: Vec<u8> = joined.chars().filter_map(unicode_to_byte).collect();
        assert_eq!(String::from_utf8(bytes).expect("valid UTF-8"), text);
    }

    #[test]
    fn test_merges_are_applied_in_rank_order() {
        // "ab" merges before "bc": with word = a b c the result must be ["ab", "c"].
        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
        ];
        let tokenizer = BPETokenizer::new(HashMap::new(), merges);
        assert_eq!(
            tokenizer.bpe("abc"),
            vec!["ab".to_string(), "c".to_string()]
        );

        // Reverse the ranks and the segmentation flips.
        let merges = vec![
            ("b".to_string(), "c".to_string()),
            ("a".to_string(), "b".to_string()),
        ];
        let tokenizer = BPETokenizer::new(HashMap::new(), merges);
        assert_eq!(
            tokenizer.bpe("abc"),
            vec!["a".to_string(), "bc".to_string()]
        );
    }

    #[test]
    fn test_merge_applies_to_all_occurrences_in_one_pass() {
        let merges = vec![("a".to_string(), "a".to_string())];
        let tokenizer = BPETokenizer::new(HashMap::new(), merges);
        assert_eq!(
            tokenizer.bpe("aaaa"),
            vec!["aa".to_string(), "aa".to_string()],
            "a single merge rank must be applied to every occurrence before re-ranking"
        );
    }

    #[test]
    fn test_enhanced_bpe_unicode_normalization() {
        let tokenizer = create_test_tokenizer();

        // Test Unicode normalization
        let text = "héllo"; // With accent
        let tokens = tokenizer.tokenize(text);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_chinese_character_handling() {
        let tokenizer = create_test_tokenizer();

        // Test Chinese character handling
        let text = "hello世界world";
        let tokens = tokenizer.tokenize(text);
        assert!(!tokens.is_empty());

        // Should properly handle Chinese characters
        let chinese_text = "你好世界";
        let chinese_tokens = tokenizer.tokenize(chinese_text);
        assert!(!chinese_tokens.is_empty());
    }

    #[test]
    fn test_is_chinese_char() {
        let tokenizer = create_test_tokenizer();

        // Test Chinese character detection
        assert!(tokenizer.is_chinese_char('你'));
        assert!(tokenizer.is_chinese_char('好'));
        assert!(tokenizer.is_chinese_char('世'));
        assert!(tokenizer.is_chinese_char('界'));

        // Test non-Chinese characters
        assert!(!tokenizer.is_chinese_char('a'));
        assert!(!tokenizer.is_chinese_char('1'));
        assert!(!tokenizer.is_chinese_char(' '));
    }

    /// Offsets must index the ORIGINAL text and follow real piece boundaries.
    #[test]
    fn test_tokenize_with_offsets_indexes_original_text() {
        // "he" is a merge, so "Hello" splits into pieces of unequal length and a
        // uniform division of the word length would be wrong.
        let merges = vec![("l".to_string(), "l".to_string())];
        let tokenizer = BPETokenizer::new(HashMap::new(), merges);

        let text = "Hello World";
        let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

        assert_eq!(tokens.len(), offsets.len());
        assert!(!tokens.is_empty());

        for (token, &(start, end)) in tokens.iter().zip(offsets.iter()) {
            assert!(start <= end, "offset span must be ordered");
            assert!(end <= text.len(), "offset must index the original text");
            // Each symbol character corresponds to exactly one source byte.
            assert_eq!(
                &text.as_bytes()[start..end],
                token.chars().filter_map(unicode_to_byte).collect::<Vec<u8>>().as_slice(),
                "token {:?} must cover exactly the bytes it encodes",
                token
            );
        }

        // Case is preserved: the first token starts at byte 0 of "Hello".
        assert_eq!(offsets[0].0, 0);
        // The offsets tile the pre-tokens contiguously.
        assert_eq!(offsets[offsets.len() - 1].1, text.len());
    }

    /// A BPE piece may split a multi-byte character. The reported span must then
    /// widen to the enclosing character (so `&text[start..end]` is valid) without
    /// shifting the pieces that follow it.
    #[test]
    fn test_tokenize_with_offsets_handles_multibyte_characters() {
        let tokenizer = BPETokenizer::new(HashMap::new(), Vec::new());

        // 'é' is U+00E9 = bytes C3 A9, so it becomes two byte-level symbols.
        let text = "aé";
        assert_eq!(text.len(), 3);

        let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);
        assert_eq!(tokens.len(), 3, "one symbol per source byte");
        assert_eq!(offsets.len(), tokens.len());

        // Every span must be a valid, non-empty slice of the original text.
        for (token, &(start, end)) in tokens.iter().zip(offsets.iter()) {
            assert!(
                start < end,
                "token {:?} must not report an empty span",
                token
            );
            assert!(text.is_char_boundary(start) && text.is_char_boundary(end));
            let _ = &text[start..end];
        }

        assert_eq!(offsets[0], (0, 1), "the ASCII 'a' keeps its exact span");
        // Both halves of 'é' report the whole character rather than drifting.
        assert_eq!(offsets[1], (1, 3));
        assert_eq!(offsets[2], (1, 3));
        assert_eq!(offsets[offsets.len() - 1].1, text.len());

        // A merge that straddles the character boundary must not shift the rest.
        let merges = vec![("a".to_string(), byte_to_unicode(0xC3).to_string())];
        let merged = BPETokenizer::new(HashMap::new(), merges);
        let (tokens, offsets) = merged.tokenize_with_offsets(text);
        assert_eq!(tokens.len(), 2);
        assert_eq!(offsets[0], (0, 3));
        assert_eq!(offsets[1], (1, 3));
    }

    #[test]
    fn test_with_options() {
        let vocab = HashMap::new();
        let merges = vec![];

        let tokenizer = BPETokenizer::with_options(
            vocab, merges, false, // normalize_unicode
            true,  // preserve_case
            false, // handle_chinese_chars
            50,    // max_input_chars_per_word
        );

        assert!(!tokenizer.normalizes_unicode());
        assert!(tokenizer.preserves_case());
        assert!(!tokenizer.handles_chinese_chars());
        assert_eq!(tokenizer.max_input_chars_per_word(), 50);
    }

    #[test]
    fn test_long_input_chunking() {
        let tokenizer = BPETokenizer::with_options(
            HashMap::new(),
            vec![],
            true,
            false,
            true,
            5, // Small max_input_chars_per_word for testing
        );

        let long_text = "this is a very long text that should be chunked";
        let tokens = tokenizer.tokenize(long_text);
        // Should handle long input without panicking
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_case_preservation() {
        let vocab = HashMap::new();
        let merges = vec![];

        let case_preserving = BPETokenizer::with_options(
            vocab.clone(),
            merges.clone(),
            true,
            true, // preserve_case = true
            false,
            100,
        );

        let case_lowering = BPETokenizer::with_options(
            vocab, merges, true, false, // preserve_case = false
            false, 100,
        );

        let text = "Hello World";
        let preserved = case_preserving.normalize_text(text);
        let lowered = case_lowering.normalize_text(text);

        assert_ne!(preserved, lowered);
        assert_eq!(lowered, text.to_lowercase());
    }

    /// The pre-token cache must stay bounded under high-cardinality input.
    #[test]
    fn test_bpe_cache_is_bounded() {
        let tokenizer = BPETokenizer::new(HashMap::new(), Vec::new());
        for i in 0..(BPE_CACHE_CAPACITY + 16) {
            let _ = tokenizer.bpe(&format!("token{}", i));
        }
        let cache_len = tokenizer.cache.read().map(|c| c.len()).unwrap_or(usize::MAX);
        assert!(
            cache_len <= BPE_CACHE_CAPACITY,
            "cache must never exceed its bound"
        );
        assert!(
            cache_len < BPE_CACHE_CAPACITY + 16,
            "cache must have evicted once the bound was reached"
        );

        // Short pre-tokens bypass the cache entirely.
        let _ = tokenizer.bpe("ab");
        let cache = tokenizer.cache.read().expect("cache lock must not be poisoned");
        assert!(!cache.contains_key("ab"));
    }
}
