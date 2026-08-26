//! WebAssembly-compatible tokenizers
//!
//! ## Honesty policy: no fabricated vocabulary, real byte-level BPE
//!
//! This module used to auto-populate every new tokenizer with a hardcoded
//! a-z/A-Z/0-9 "vocabulary" (`init_basic_vocab`), and its `TokenizerType::BPE`
//! implementation was character splitting dressed up as BPE - one vocabulary
//! lookup per Unicode `char`, no byte-level alphabet, no merge rules,
//! `TokenizerType::SentencePiece` simply calling the same thing. Per
//! project-wide policy, constructors never fabricate a vocabulary: a fresh
//! [`WasmTokenizer`] starts empty and [`WasmTokenizer::encode`]/
//! [`WasmTokenizer::decode`] return an instructive [`JsValue`] error until
//! real data has been loaded via [`WasmTokenizer::load_vocab`] (and, for
//! `BPE`/`SentencePiece`, [`WasmTokenizer::load_merges`]) - this is wasm,
//! there is no filesystem to read a real vocab/merges file from, so JS must
//! supply it.
//!
//! `TokenizerType::BPE` now implements real byte-level BPE (GPT-2/RoBERTa
//! style): a rank-ordered merge table applied over the GPT-2 byte-to-unicode
//! alphabet, mirroring the algorithm in
//! `trustformers-tokenizers/src/bpe.rs` (read as the reference for this
//! port; **not** taken as a dependency here - that crate pulls in
//! `fancy_regex` for its lookahead-based pre-tokenizer regex, and per the
//! porting brief for this module, wasm32-cleanliness was to be verified
//! before ever depending on it, and a from-scratch pre-tokenizer avoids the
//! question entirely). [`TokenizerType::SentencePiece`] continues to
//! delegate to the same real BPE implementation (SentencePiece's actual
//! Unigram-LM algorithm is out of scope here) - now honestly, since the BPE
//! it delegates to is real.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::string::{String, ToString};
use std::sync::OnceLock;
use std::vec::Vec;
use std::{format, vec};
use wasm_bindgen::prelude::*;

/// Tokenizer type
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerType {
    WordPiece,
    BPE,
    SentencePiece,
}

/// Special tokens used by tokenizers
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pad_token: String,
    unk_token: String,
    cls_token: String,
    sep_token: String,
    mask_token: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

#[wasm_bindgen]
impl SpecialTokens {
    #[wasm_bindgen(getter)]
    pub fn pad_token(&self) -> String {
        self.pad_token.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn unk_token(&self) -> String {
        self.unk_token.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn cls_token(&self) -> String {
        self.cls_token.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn sep_token(&self) -> String {
        self.sep_token.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn mask_token(&self) -> String {
        self.mask_token.clone()
    }
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            pad_token: "[PAD]".to_string(),
            unk_token: "[UNK]".to_string(),
            cls_token: "[CLS]".to_string(),
            sep_token: "[SEP]".to_string(),
            mask_token: "[MASK]".to_string(),
            bos_token: None,
            eos_token: None,
        }
    }
}

// ---------------------------------------------------------------------
// GPT-2 byte-level alphabet
// ---------------------------------------------------------------------

/// GPT-2 `bytes_to_unicode()` table: byte value -> printable character.
/// Printable Latin-1 bytes map to themselves; the remaining "unprintable"
/// bytes (control characters, etc.) are shifted into the unused
/// `U+0100..=U+01FF` range so every byte gets its own distinct, printable
/// symbol. Built once and cached (no external `once_cell` dependency
/// needed - `std::sync::OnceLock` has covered this since Rust 1.70).
fn byte_encoder_table() -> &'static [char; 256] {
    static TABLE: OnceLock<[char; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = ['\0'; 256];
        let mut next_code_point: u32 = 256;

        for (byte, slot) in table.iter_mut().enumerate() {
            let byte = byte as u8;
            if (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174 {
                *slot = byte as char;
            } else {
                // `next_code_point` stays inside 256..=511, which contains
                // no surrogate code points, so `from_u32` is always `Some`.
                *slot = char::from_u32(next_code_point)
                    .unwrap_or('\u{fffd}' /* unreachable: see comment above */);
                next_code_point += 1;
            }
        }

        table
    })
}

/// Inverse of [`byte_encoder_table`], indexed by code point (all entries
/// are `< 512`).
fn byte_decoder_table() -> &'static [Option<u8>; 512] {
    static TABLE: OnceLock<[Option<u8>; 512]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [None; 512];
        for (b, &ch) in byte_encoder_table().iter().enumerate() {
            let code_point = ch as usize;
            if code_point < 512 {
                table[code_point] = Some(b as u8);
            }
        }
        table
    })
}

/// Map a single byte to its GPT-2 byte-level alphabet character.
fn byte_to_unicode(byte: u8) -> char {
    byte_encoder_table()[byte as usize]
}

/// Map a GPT-2 byte-level alphabet character back to its byte, if it is
/// part of the byte-level alphabet.
fn unicode_to_byte(ch: char) -> Option<u8> {
    let code_point = ch as usize;
    if code_point < 512 {
        byte_decoder_table()[code_point]
    } else {
        None
    }
}

// ---------------------------------------------------------------------
// GPT-2 pre-tokenization (byte-level BPE word splitting)
// ---------------------------------------------------------------------

/// The reference GPT-2 pre-tokenizer is the regex
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`,
/// whose `(?!\S)` lookahead has no `regex`-crate equivalent (hence
/// `trustformers-tokenizers` pulling in `fancy_regex` for it). This
/// implements the same splitting behavior imperatively instead: contractions
/// first, then runs of one Unicode category (letters / digits / "other"),
/// each optionally preceded by exactly one leading space, and finally
/// whitespace runs - where a whitespace run immediately followed by more
/// content gives up its last character to that following pre-token (which
/// is what produces the familiar leading-space-attached-to-word pieces),
/// while a run that reaches the end of the string is kept whole.
///
/// Approximates `\p{L}`/`\p{N}` with [`char::is_alphabetic`]/
/// [`char::is_numeric`], which covers ordinary text well even though the
/// two aren't byte-for-byte identical to the Unicode General Category
/// classes the reference regex uses.
fn gpt2_pre_token_spans(text: &str) -> Vec<(usize, usize)> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

    #[derive(PartialEq, Eq)]
    enum Category {
        Letter,
        Number,
        Other,
    }
    fn category_of(ch: char) -> Category {
        if ch.is_alphabetic() {
            Category::Letter
        } else if ch.is_numeric() {
            Category::Number
        } else {
            Category::Other
        }
    }

    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut i = 0usize;

    while i < chars.len() {
        let (start_byte, ch) = chars[i];

        // 1. Contractions, matched against the literal byte suffix here.
        if ch == '\'' {
            if let Some(matched) = CONTRACTIONS.iter().find(|c| text[start_byte..].starts_with(**c))
            {
                let end_byte = start_byte + matched.len();
                spans.push((start_byte, end_byte));
                i += matched.chars().count();
                continue;
            }
        }

        if ch == ' ' {
            if let Some(&(_, next_ch)) = chars.get(i + 1) {
                if !next_ch.is_whitespace() {
                    // " ?\p{L}+" / " ?\p{N}+" / " ?[^\s\p{L}\p{N}]+": this
                    // one leading space plus a maximal run of `next_ch`'s
                    // category.
                    let category = category_of(next_ch);
                    let mut j = i + 1;
                    while j < chars.len() && category_of(chars[j].1) == category {
                        j += 1;
                    }
                    let end_byte = chars.get(j).map_or(text.len(), |&(b, _)| b);
                    spans.push((start_byte, end_byte));
                    i = j;
                    continue;
                }
            }
            // Fall through to the whitespace-run handling below: nothing
            // (or more whitespace) follows this single space.
        } else if ch.is_whitespace() {
            // Non-space whitespace (tab, newline, ...) always falls to the
            // whitespace-run handling below.
        } else {
            // Non-space, non-contraction: run of `ch`'s own category.
            let category = category_of(ch);
            let mut j = i + 1;
            while j < chars.len() && category_of(chars[j].1) == category {
                j += 1;
            }
            let end_byte = chars.get(j).map_or(text.len(), |&(b, _)| b);
            spans.push((start_byte, end_byte));
            i = j;
            continue;
        }

        // Whitespace-run handling (`\s+(?!\S)` / `\s+`): consume the whole
        // run if it reaches the end of the string, otherwise all but its
        // last character (which becomes the next pre-token's leading
        // space).
        let mut j = i;
        while j < chars.len() && chars[j].1.is_whitespace() {
            j += 1;
        }
        let run_reaches_end = j == chars.len();
        let take_to = if run_reaches_end { j } else { j.saturating_sub(1) }.max(i + 1);
        let end_byte = chars.get(take_to).map_or(text.len(), |&(b, _)| b);
        spans.push((start_byte, end_byte));
        i = take_to;
    }

    spans
}

// ---------------------------------------------------------------------
// WasmTokenizer
// ---------------------------------------------------------------------

/// WebAssembly-compatible tokenizer.
///
/// Starts with an empty vocabulary (and, for BPE, empty merge rules) - real
/// data must be loaded via [`Self::load_vocab`]/[`Self::load_merges`] before
/// [`Self::encode`]/[`Self::decode`] will do anything but return an
/// instructive error. See the module-level doc comment.
#[wasm_bindgen]
pub struct WasmTokenizer {
    tokenizer_type: TokenizerType,
    vocab: BTreeMap<String, u32>,
    reverse_vocab: BTreeMap<u32, String>,
    /// BPE merge rules in rank order (rank == index). Only meaningful for
    /// `TokenizerType::BPE`/`SentencePiece`. Empty is a legitimate, honest
    /// state (byte-level tokenization with no merges applied yet), not a
    /// fabricated default.
    merges: Vec<(String, String)>,
    /// `(first, second) -> rank`, built from `merges` by [`Self::load_merges`].
    merge_ranks: HashMap<(String, String), usize>,
    special_tokens: SpecialTokens,
    max_length: usize,
}

#[wasm_bindgen]
impl WasmTokenizer {
    /// Create a new, empty tokenizer. Call [`Self::load_vocab`] (and, for
    /// `BPE`/`SentencePiece`, [`Self::load_merges`]) before
    /// [`Self::encode`]/[`Self::decode`] - there is no fabricated
    /// placeholder vocabulary to fall back on.
    #[wasm_bindgen(constructor)]
    pub fn new(tokenizer_type: TokenizerType) -> Self {
        Self {
            tokenizer_type,
            vocab: BTreeMap::new(),
            reverse_vocab: BTreeMap::new(),
            merges: Vec::new(),
            merge_ranks: HashMap::new(),
            special_tokens: SpecialTokens::default(),
            max_length: 512,
        }
    }

    /// Load a real vocabulary (token string -> id) supplied by JS. Errors
    /// if the supplied data doesn't parse as a `{token: id}` map, or if it
    /// parses but is empty - either way, this is the only source of real
    /// vocabulary data (wasm has no filesystem to read one from).
    pub fn load_vocab(&mut self, vocab_js: JsValue) -> Result<(), JsValue> {
        let vocab: BTreeMap<String, u32> = serde_wasm_bindgen::from_value(vocab_js)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse vocab: {e:?}")))?;
        self.load_vocab_map(vocab).map_err(|e| JsValue::from_str(&e))
    }

    /// Load real BPE merge rules supplied by JS, as an ordered array of
    /// `[first, second]` pairs (rank == array index) - the same information
    /// GPT-2's `merges.txt` encodes, one pair per line in priority order.
    /// Only meaningful for `TokenizerType::BPE`/`SentencePiece`; an empty
    /// list is valid (byte-level tokenization with no merges).
    pub fn load_merges(&mut self, merges_js: JsValue) -> Result<(), JsValue> {
        let merges: Vec<(String, String)> = serde_wasm_bindgen::from_value(merges_js)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse merges: {e:?}")))?;
        self.load_merges_vec(merges);
        Ok(())
    }

    /// Encode text to token IDs.
    ///
    /// Errors (rather than fabricating output) if no vocabulary has been
    /// loaded via [`Self::load_vocab`], or if a piece of the input has no
    /// vocabulary entry and no `unk_token` is configured to fall back on.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, JsValue> {
        self.encode_core(text, add_special_tokens).map_err(|e| JsValue::from_str(&e))
    }

    /// Decode token IDs to text.
    ///
    /// Errors (rather than fabricating output) if no vocabulary has been
    /// loaded via [`Self::load_vocab`], or if the decoded byte-level BPE
    /// symbols don't reconstitute valid UTF-8.
    pub fn decode(
        &self,
        token_ids: Vec<u32>,
        skip_special_tokens: bool,
    ) -> Result<String, JsValue> {
        self.decode_core(&token_ids, skip_special_tokens)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Batch encode multiple texts.
    pub fn batch_encode(
        &self,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> Result<BatchEncodingOutput, JsValue> {
        let encoded_sequences = texts
            .iter()
            .map(|text| self.encode_core(text, add_special_tokens))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| JsValue::from_str(&e))?;
        Ok(BatchEncodingOutput { encoded_sequences })
    }

    /// Get vocabulary size
    #[wasm_bindgen(getter)]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Whether a real vocabulary has been loaded via [`Self::load_vocab`].
    #[wasm_bindgen(getter)]
    pub fn has_vocab(&self) -> bool {
        !self.vocab.is_empty()
    }

    /// Set maximum sequence length
    pub fn set_max_length(&mut self, max_length: usize) {
        self.max_length = max_length;
    }

    /// Get special token IDs
    pub fn get_special_token_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();

        let special_tokens = vec![
            &self.special_tokens.pad_token,
            &self.special_tokens.unk_token,
            &self.special_tokens.cls_token,
            &self.special_tokens.sep_token,
            &self.special_tokens.mask_token,
        ];

        for token in special_tokens {
            if let Some(&id) = self.vocab.get(token) {
                ids.push(id);
            }
        }

        ids
    }

    // Private helper methods

    fn create_substr(&self, chars: &[char], add_prefix: bool) -> String {
        let substr: String = chars.iter().collect();
        if add_prefix {
            format!("##{substr}")
        } else {
            substr
        }
    }

    fn find_vocab_match(
        &self,
        chars: &[char],
        start: usize,
        mut end: usize,
    ) -> (Option<String>, usize) {
        while start < end {
            let substr = self.create_substr(&chars[start..end], start > 0);
            if self.vocab.contains_key(&substr) {
                return (Some(substr), end);
            }
            end -= 1;
        }
        (None, end)
    }

    fn is_special_token(&self, token: &str) -> bool {
        token == self.special_tokens.pad_token
            || token == self.special_tokens.unk_token
            || token == self.special_tokens.cls_token
            || token == self.special_tokens.sep_token
            || token == self.special_tokens.mask_token
    }

    fn wordpiece_tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        let unk_id = self.vocab.get(&self.special_tokens.unk_token).copied();
        let mut tokens = Vec::new();

        for word in text.split_whitespace() {
            let chars: Vec<char> = word.chars().collect();
            let mut is_bad = false;
            let mut start = 0;
            let mut word_tokens = Vec::new();

            while start < chars.len() {
                let end = chars.len();
                let (found_substr, new_end) = self.find_vocab_match(&chars, start, end);

                match found_substr.and_then(|substr| self.vocab.get(&substr).copied()) {
                    Some(token_id) => {
                        word_tokens.push(token_id);
                        start = new_end;
                    },
                    None => {
                        is_bad = true;
                        break;
                    },
                }
            }

            if is_bad {
                match unk_id {
                    Some(id) => tokens.push(id),
                    None => {
                        return Err(format!(
                            "WasmTokenizer: word {word:?} has no WordPiece match in the loaded \
                             vocabulary and no unk_token is configured"
                        ))
                    },
                }
            } else {
                tokens.extend(word_tokens);
            }
        }

        Ok(tokens)
    }

    /// Apply the loaded merge rules to one byte-level-encoded pre-token,
    /// mirroring `trustformers-tokenizers::bpe::BPETokenizer::apply_merges`:
    /// repeatedly find the lowest-rank adjacent pair among the current
    /// symbols and merge every occurrence of that exact pair in one pass,
    /// until no pair in `self.merge_ranks` remains (or one symbol is left).
    fn apply_bpe_merges(&self, symbols: &mut Vec<String>) {
        if self.merge_ranks.is_empty() {
            return;
        }

        while symbols.len() > 1 {
            let mut best_rank = usize::MAX;
            let mut best_index: Option<usize> = None;

            for i in 0..symbols.len() - 1 {
                if let Some(&rank) =
                    self.merge_ranks.get(&(symbols[i].clone(), symbols[i + 1].clone()))
                {
                    if rank < best_rank {
                        best_rank = rank;
                        best_index = Some(i);
                    }
                }
            }

            let Some(first_index) = best_index else {
                break;
            };

            let first_text = symbols[first_index].clone();
            let second_text = symbols[first_index + 1].clone();

            let mut merged = Vec::with_capacity(symbols.len());
            let mut i = 0;
            while i < symbols.len() {
                let matches_pair = i + 1 < symbols.len()
                    && symbols[i] == first_text
                    && symbols[i + 1] == second_text;
                if matches_pair {
                    merged.push(format!("{first_text}{second_text}"));
                    i += 2;
                } else {
                    merged.push(symbols[i].clone());
                    i += 1;
                }
            }
            *symbols = merged;
        }
    }

    /// Byte-level-encode one pre-token and apply the loaded merges.
    fn bpe_symbols(&self, pretoken: &str) -> Vec<String> {
        let mut symbols: Vec<String> =
            pretoken.bytes().map(|b| byte_to_unicode(b).to_string()).collect();
        self.apply_bpe_merges(&mut symbols);
        symbols
    }

    /// Real byte-level BPE: GPT-2-style pre-tokenization, byte-to-unicode
    /// encoding, then rank-ordered merges - see the module-level doc
    /// comment. Replaces the former "BPE" that just looked up one vocab
    /// entry per `char`.
    fn bpe_tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        let unk_id = self.vocab.get(&self.special_tokens.unk_token).copied();
        let mut ids = Vec::new();

        for (start, end) in gpt2_pre_token_spans(text) {
            for symbol in self.bpe_symbols(&text[start..end]) {
                match self.vocab.get(&symbol) {
                    Some(&id) => ids.push(id),
                    None => match unk_id {
                        Some(id) => ids.push(id),
                        None => {
                            return Err(format!(
                                "WasmTokenizer: BPE symbol {symbol:?} is not in the loaded \
                                 vocabulary and no unk_token is configured"
                            ))
                        },
                    },
                }
            }
        }

        Ok(ids)
    }

    fn decode_wordpiece(&self, tokens: &[String]) -> String {
        let mut result = String::new();

        for (i, token) in tokens.iter().enumerate() {
            if let Some(stripped) = token.strip_prefix("##") {
                result.push_str(stripped);
            } else {
                if i > 0 {
                    result.push(' ');
                }
                result.push_str(token);
            }
        }

        result
    }

    /// Reverse byte-level BPE: join the symbol strings and map every
    /// character back through the byte-level alphabet to reconstitute the
    /// original UTF-8 bytes. Errors (rather than silently dropping bytes or
    /// lossily replacing them) if a character isn't part of the alphabet or
    /// the recovered bytes aren't valid UTF-8.
    fn decode_bpe(&self, tokens: &[String]) -> Result<String, String> {
        let joined: String = tokens.concat();
        let mut bytes = Vec::with_capacity(joined.len());

        for ch in joined.chars() {
            match unicode_to_byte(ch) {
                Some(b) => bytes.push(b),
                None => {
                    return Err(format!(
                        "WasmTokenizer: character {ch:?} is not part of the byte-level BPE alphabet"
                    ))
                },
            }
        }

        String::from_utf8(bytes)
            .map_err(|e| format!("WasmTokenizer: decoded bytes are not valid UTF-8: {e}"))
    }

    // ---------------------------------------------------------------
    // JsValue-free core (see module doc comment / lib.rs's
    // `InferenceSession` for why: constructing a `JsValue` unconditionally
    // panics off wasm32, so the fallible core stays plain-Rust-`String`
    // errors and only the `#[wasm_bindgen]` methods above convert to
    // `JsValue`, making all of this natively testable).
    // ---------------------------------------------------------------

    pub(crate) fn load_vocab_map(&mut self, vocab: BTreeMap<String, u32>) -> Result<(), String> {
        if vocab.is_empty() {
            return Err("WasmTokenizer::load_vocab: supplied vocabulary is empty".to_string());
        }
        self.reverse_vocab = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        self.vocab = vocab;
        Ok(())
    }

    pub(crate) fn load_merges_vec(&mut self, merges: Vec<(String, String)>) {
        self.merge_ranks = merges
            .iter()
            .enumerate()
            .map(|(rank, (first, second))| ((first.clone(), second.clone()), rank))
            .collect();
        self.merges = merges;
    }

    pub(crate) fn encode_core(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, String> {
        if self.vocab.is_empty() {
            return Err("WasmTokenizer: no vocabulary loaded (call load_vocab first)".to_string());
        }

        let mut tokens = match self.tokenizer_type {
            TokenizerType::WordPiece => self.wordpiece_tokenize(text)?,
            // SentencePiece delegates to the same real byte-level BPE (see
            // the module doc comment) - not a fabricated Unigram-LM stand-in.
            TokenizerType::BPE | TokenizerType::SentencePiece => self.bpe_tokenize(text)?,
        };

        if add_special_tokens {
            if let Some(&cls_id) = self.vocab.get(&self.special_tokens.cls_token) {
                tokens.insert(0, cls_id);
            }
            if let Some(&sep_id) = self.vocab.get(&self.special_tokens.sep_token) {
                tokens.push(sep_id);
            }
        }

        if tokens.len() > self.max_length {
            tokens.truncate(self.max_length);
        }

        Ok(tokens)
    }

    pub(crate) fn decode_core(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, String> {
        if self.vocab.is_empty() {
            return Err("WasmTokenizer: no vocabulary loaded (call load_vocab first)".to_string());
        }

        let mut tokens = Vec::new();
        for &id in token_ids {
            if let Some(token) = self.reverse_vocab.get(&id) {
                if skip_special_tokens && self.is_special_token(token) {
                    continue;
                }
                tokens.push(token.clone());
            }
        }

        match self.tokenizer_type {
            TokenizerType::WordPiece => Ok(self.decode_wordpiece(&tokens)),
            TokenizerType::BPE | TokenizerType::SentencePiece => self.decode_bpe(&tokens),
        }
    }
}

/// Tokenizer output with attention mask
#[wasm_bindgen]
pub struct TokenizerOutput {
    input_ids: Vec<u32>,
    attention_mask: Vec<u32>,
    token_type_ids: Option<Vec<u32>>,
}

#[wasm_bindgen]
impl TokenizerOutput {
    #[wasm_bindgen(getter)]
    pub fn input_ids(&self) -> Vec<u32> {
        self.input_ids.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn attention_mask(&self) -> Vec<u32> {
        self.attention_mask.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn token_type_ids(&self) -> Option<Vec<u32>> {
        self.token_type_ids.clone()
    }
}

/// Batch encoding output
#[wasm_bindgen]
pub struct BatchEncodingOutput {
    encoded_sequences: Vec<Vec<u32>>,
}

#[wasm_bindgen]
impl BatchEncodingOutput {
    /// Get the number of sequences
    pub fn len(&self) -> usize {
        self.encoded_sequences.len()
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.encoded_sequences.is_empty()
    }

    /// Get a specific sequence by index
    pub fn get_sequence(&self, index: usize) -> Option<Vec<u32>> {
        self.encoded_sequences.get(index).cloned()
    }
}

#[wasm_bindgen]
impl TokenizerOutput {
    /// Create tokenizer output with padding
    pub fn with_padding(input_ids: Vec<u32>, max_length: usize, pad_token_id: u32) -> Self {
        let mut padded_ids = input_ids.clone();
        let mut attention_mask = vec![1u32; input_ids.len()];

        // Pad to max_length
        while padded_ids.len() < max_length {
            padded_ids.push(pad_token_id);
            attention_mask.push(0);
        }

        Self {
            input_ids: padded_ids,
            attention_mask,
            token_type_ids: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wordpiece_vocab() -> BTreeMap<String, u32> {
        let mut vocab = BTreeMap::new();
        for (i, tok) in [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "##lo", "hel",
        ]
        .iter()
        .enumerate()
        {
            vocab.insert((*tok).to_string(), i as u32);
        }
        vocab
    }

    fn wordpiece_tokenizer() -> WasmTokenizer {
        let mut tok = WasmTokenizer::new(TokenizerType::WordPiece);
        tok.load_vocab_map(wordpiece_vocab()).expect("non-empty vocab");
        tok
    }

    // -----------------------------------------------------------------
    // Constructors never fabricate a vocabulary.
    // -----------------------------------------------------------------

    #[test]
    fn test_new_tokenizer_has_no_vocab() {
        let tok = WasmTokenizer::new(TokenizerType::BPE);
        assert_eq!(tok.vocab_size(), 0);
        assert!(!tok.has_vocab());
    }

    #[test]
    fn test_encode_without_vocab_errors_not_fabricates() {
        let tok = WasmTokenizer::new(TokenizerType::BPE);
        let err = tok.encode_core("hello world", false).expect_err("must error without a vocab");
        assert!(err.contains("no vocabulary loaded"));
    }

    #[test]
    fn test_decode_without_vocab_errors_not_fabricates() {
        let tok = WasmTokenizer::new(TokenizerType::BPE);
        let err = tok.decode_core(&[0, 1, 2], false).expect_err("must error without a vocab");
        assert!(err.contains("no vocabulary loaded"));
    }

    #[test]
    fn test_load_vocab_map_rejects_empty_vocab() {
        let mut tok = WasmTokenizer::new(TokenizerType::BPE);
        let err = tok.load_vocab_map(BTreeMap::new()).expect_err("empty vocab must be rejected");
        assert!(err.contains("empty"));
    }

    // -----------------------------------------------------------------
    // GPT-2 byte-level alphabet: round-trips every byte, matches known
    // canonical anchors.
    // -----------------------------------------------------------------

    #[test]
    fn test_byte_alphabet_round_trips_every_byte() {
        let mut seen = std::collections::HashSet::new();
        for b in 0..=255u8 {
            let ch = byte_to_unicode(b);
            assert!(seen.insert(ch), "byte {b} produced a duplicate character");
            assert_eq!(unicode_to_byte(ch), Some(b), "byte {b} did not round-trip");
        }
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn test_byte_alphabet_matches_gpt2_canonical_anchors() {
        assert_eq!(byte_to_unicode(b' '), '\u{0120}'); // 'Ġ'
        assert_eq!(byte_to_unicode(0), '\u{0100}'); // 'Ā'
        assert_eq!(byte_to_unicode(b'a'), 'a');
        assert_eq!(byte_to_unicode(b'\n'), '\u{010a}');
    }

    // -----------------------------------------------------------------
    // Pre-tokenization: the whitespace-attaches-to-following-word rule.
    // -----------------------------------------------------------------

    fn spans_as_strs(text: &str) -> Vec<&str> {
        gpt2_pre_token_spans(text).into_iter().map(|(s, e)| &text[s..e]).collect()
    }

    #[test]
    fn test_pre_tokenizer_gives_trailing_space_to_next_word() {
        assert_eq!(spans_as_strs("a   b"), vec!["a", "  ", " b"]);
        assert_eq!(spans_as_strs("hello world"), vec!["hello", " world"]);
        assert_eq!(spans_as_strs("hi   "), vec!["hi", "   "]);
    }

    #[test]
    fn test_pre_tokenizer_handles_contractions_and_punctuation() {
        assert_eq!(spans_as_strs("it's"), vec!["it", "'s"]);
        assert_eq!(spans_as_strs("a,b"), vec!["a", ",", "b"]);
    }

    // -----------------------------------------------------------------
    // Real byte-level BPE encode/decode round-trip and known-tokenization
    // tests over a small, hand-computed vocabulary + merge table.
    // -----------------------------------------------------------------

    /// Builds a tiny byte-level BPE vocab/merge table by hand:
    /// bytes 'a'..'d' each map to themselves in the byte alphabet (they're
    /// all printable ASCII), plus a merge "a"+"b" -> "ab" and a vocab entry
    /// for the merged symbol "ab".
    fn tiny_bpe_tokenizer() -> WasmTokenizer {
        let mut vocab = BTreeMap::new();
        for (i, tok) in ["a", "b", "c", "d", "ab", "<unk>"].iter().enumerate() {
            vocab.insert((*tok).to_string(), i as u32);
        }
        let mut tok = WasmTokenizer::new(TokenizerType::BPE);
        tok.load_vocab_map(vocab).expect("non-empty vocab");
        tok.load_merges_vec(vec![("a".to_string(), "b".to_string())]);
        tok
    }

    #[test]
    fn test_bpe_encode_applies_merge_rank() {
        let tok = tiny_bpe_tokenizer();
        // "ab" -> byte symbols ['a','b'] -> merges to ["ab"] -> vocab id 4.
        let ids = tok.encode_core("ab", false).expect("should encode");
        assert_eq!(ids, vec![4]);
    }

    #[test]
    fn test_bpe_encode_no_merge_when_pair_not_adjacent_in_table() {
        let tok = tiny_bpe_tokenizer();
        // "cd" has no merge rule for ('c','d'), so it stays two symbols.
        let ids = tok.encode_core("cd", false).expect("should encode");
        assert_eq!(ids, vec![2, 3]); // "c", "d"
    }

    #[test]
    fn test_bpe_encode_decode_round_trip() {
        let tok = tiny_bpe_tokenizer();
        for text in ["ab", "abc", "abcd", "cdab"] {
            let ids = tok.encode_core(text, false).expect("should encode");
            let decoded = tok.decode_core(&ids, false).expect("should decode");
            assert_eq!(decoded, text, "round trip failed for {text:?}");
        }
    }

    #[test]
    fn test_bpe_encode_unknown_symbol_falls_back_to_unk() {
        let mut vocab = BTreeMap::new();
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("a".to_string(), 1);
        let mut tok = WasmTokenizer::new(TokenizerType::BPE);
        tok.special_tokens.unk_token = "<unk>".to_string();
        tok.load_vocab_map(vocab).expect("non-empty vocab");
        // 'z' has no vocab entry, so it must fall back to <unk> (id 0), not
        // error and not silently vanish.
        let ids = tok.encode_core("z", false).expect("should fall back to unk");
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn test_bpe_encode_unknown_symbol_without_unk_errors() {
        let mut vocab = BTreeMap::new();
        vocab.insert("a".to_string(), 0);
        let mut tok = WasmTokenizer::new(TokenizerType::BPE);
        tok.special_tokens.unk_token = "<not-in-vocab>".to_string();
        tok.load_vocab_map(vocab).expect("non-empty vocab");
        let err = tok.encode_core("z", false).expect_err("must error, not fabricate a token");
        assert!(err.contains("not in the loaded vocabulary"));
    }

    #[test]
    fn test_bpe_full_byte_range_round_trips_through_encode_decode() {
        // Build a vocab covering every single-byte symbol so every byte
        // round-trips even with no merges.
        let mut vocab = BTreeMap::new();
        for b in 0..=255u8 {
            vocab.insert(byte_to_unicode(b).to_string(), b as u32);
        }
        let mut tok = WasmTokenizer::new(TokenizerType::BPE);
        tok.load_vocab_map(vocab).expect("non-empty vocab");

        let text = "Hello, world! \u{1F600} \n\t";
        let ids = tok.encode_core(text, false).expect("should encode arbitrary UTF-8");
        let decoded = tok.decode_core(&ids, false).expect("should decode back");
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_sentencepiece_delegates_to_real_bpe_not_fabricated() {
        let mut vocab = BTreeMap::new();
        for (i, tok) in ["a", "b", "ab"].iter().enumerate() {
            vocab.insert((*tok).to_string(), i as u32);
        }
        let mut tok = WasmTokenizer::new(TokenizerType::SentencePiece);
        tok.load_vocab_map(vocab).expect("non-empty vocab");
        tok.load_merges_vec(vec![("a".to_string(), "b".to_string())]);

        let ids = tok.encode_core("ab", false).expect("should encode");
        assert_eq!(ids, vec![2]); // merged "ab"
        let decoded = tok.decode_core(&ids, false).expect("should decode");
        assert_eq!(decoded, "ab");
    }

    // -----------------------------------------------------------------
    // WordPiece: unchanged algorithm, but no more fabricated vocabulary
    // and a real unk-fallback / error split.
    // -----------------------------------------------------------------

    #[test]
    fn test_wordpiece_encode_decode() {
        let tok = wordpiece_tokenizer();
        let ids = tok.encode_core("hello", false).expect("should encode");
        assert!(!ids.is_empty());
        let decoded = tok.decode_core(&ids, false).expect("should decode");
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_wordpiece_adds_special_tokens() {
        let tok = wordpiece_tokenizer();
        let ids = tok.encode_core("hello", true).expect("should encode");
        // [CLS] hello [SEP] at minimum.
        assert!(ids.len() >= 3);
        assert_eq!(ids[0], tok.vocab["[CLS]"]);
        assert_eq!(
            *ids.last().expect("ids must be non-empty"),
            tok.vocab["[SEP]"]
        );
    }

    #[test]
    fn test_wordpiece_unmatched_word_without_unk_errors() {
        let mut vocab = BTreeMap::new();
        vocab.insert("known".to_string(), 0);
        let mut tok = WasmTokenizer::new(TokenizerType::WordPiece);
        tok.special_tokens.unk_token = "<not-in-vocab>".to_string();
        tok.load_vocab_map(vocab).expect("non-empty vocab");
        let err = tok.encode_core("unmatched", false).expect_err("must error, not fabricate");
        assert!(err.contains("no WordPiece match"));
    }
}
