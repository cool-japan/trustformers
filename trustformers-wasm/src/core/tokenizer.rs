//! WebAssembly-compatible tokenizers

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::string::{String, ToString};
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

/// WebAssembly-compatible tokenizer
#[wasm_bindgen]
pub struct WasmTokenizer {
    tokenizer_type: TokenizerType,
    vocab: BTreeMap<String, u32>,
    reverse_vocab: BTreeMap<u32, String>,
    special_tokens: SpecialTokens,
    max_length: usize,
}

#[wasm_bindgen]
impl WasmTokenizer {
    /// Create a new tokenizer
    #[wasm_bindgen(constructor)]
    pub fn new(tokenizer_type: TokenizerType) -> Self {
        let mut tokenizer = Self {
            tokenizer_type,
            vocab: BTreeMap::new(),
            reverse_vocab: BTreeMap::new(),
            special_tokens: SpecialTokens::default(),
            max_length: 512,
        };

        // Initialize with basic vocabulary
        tokenizer.init_basic_vocab();
        tokenizer
    }

    /// Load vocabulary from JS object
    pub fn load_vocab(&mut self, vocab_js: JsValue) -> Result<(), JsValue> {
        let vocab: BTreeMap<String, u32> = serde_wasm_bindgen::from_value(vocab_js)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse vocab: {e:?}")))?;

        self.vocab = vocab;
        self.reverse_vocab = self.vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        Ok(())
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        let mut tokens = match self.tokenizer_type {
            TokenizerType::WordPiece => self.wordpiece_tokenize(text),
            TokenizerType::BPE => self.bpe_tokenize(text),
            TokenizerType::SentencePiece => self.sentencepiece_tokenize(text),
        };

        if add_special_tokens {
            // Add [CLS] at beginning and [SEP] at end for BERT-style
            if let Some(&cls_id) = self.vocab.get(&self.special_tokens.cls_token) {
                tokens.insert(0, cls_id);
            }
            if let Some(&sep_id) = self.vocab.get(&self.special_tokens.sep_token) {
                tokens.push(sep_id);
            }
        }

        // Truncate if necessary
        if tokens.len() > self.max_length {
            tokens.truncate(self.max_length);
        }

        tokens
    }

    /// Decode token IDs to text
    pub fn decode(&self, token_ids: Vec<u32>, skip_special_tokens: bool) -> String {
        let mut tokens = Vec::new();

        for &id in &token_ids {
            if let Some(token) = self.reverse_vocab.get(&id) {
                if skip_special_tokens && self.is_special_token(token) {
                    continue;
                }
                tokens.push(token.clone());
            }
        }

        // Join tokens and clean up
        match self.tokenizer_type {
            TokenizerType::WordPiece => self.decode_wordpiece(tokens),
            TokenizerType::BPE => self.decode_bpe(tokens),
            TokenizerType::SentencePiece => self.decode_sentencepiece(tokens),
        }
    }

    /// Batch encode multiple texts
    pub fn batch_encode(
        &self,
        texts: Vec<String>,
        add_special_tokens: bool,
    ) -> BatchEncodingOutput {
        let encoded_sequences =
            texts.iter().map(|text| self.encode(text, add_special_tokens)).collect();
        BatchEncodingOutput { encoded_sequences }
    }

    /// Get vocabulary size
    #[wasm_bindgen(getter)]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
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

    fn init_basic_vocab(&mut self) {
        // Initialize with special tokens and basic vocabulary
        let mut id = 0u32;

        // Special tokens
        self.vocab.insert(self.special_tokens.pad_token.clone(), id);
        id += 1;
        self.vocab.insert(self.special_tokens.unk_token.clone(), id);
        id += 1;
        self.vocab.insert(self.special_tokens.cls_token.clone(), id);
        id += 1;
        self.vocab.insert(self.special_tokens.sep_token.clone(), id);
        id += 1;
        self.vocab.insert(self.special_tokens.mask_token.clone(), id);
        id += 1;

        // Basic ASCII vocabulary
        for c in b'a'..=b'z' {
            self.vocab.insert((c as char).to_string(), id);
            id += 1;
        }
        for c in b'A'..=b'Z' {
            self.vocab.insert((c as char).to_string(), id);
            id += 1;
        }
        for c in b'0'..=b'9' {
            self.vocab.insert((c as char).to_string(), id);
            id += 1;
        }

        // Common punctuation
        for &c in &['.', ',', '!', '?', ';', ':', '-', '_', ' '] {
            self.vocab.insert(c.to_string(), id);
            id += 1;
        }

        // Create reverse vocabulary
        self.reverse_vocab = self.vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
    }

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

    fn wordpiece_tokenize(&self, text: &str) -> Vec<u32> {
        // Simplified WordPiece tokenization
        let mut tokens = Vec::new();
        let unk_id = self.vocab.get(&self.special_tokens.unk_token).copied().unwrap_or(1);

        for word in text.split_whitespace() {
            let chars: Vec<char> = word.chars().collect();
            let mut is_bad = false;
            let mut start = 0;

            while start < chars.len() {
                let end = chars.len();

                let (found_substr, new_end) = self.find_vocab_match(&chars, start, end);
                let cur_substr = found_substr;

                if let Some(substr) = cur_substr {
                    tokens.push(*self.vocab.get(&substr).unwrap());
                    start = new_end;
                } else {
                    is_bad = true;
                    break;
                }
            }

            if is_bad {
                tokens.push(unk_id);
            }
        }

        tokens
    }

    fn bpe_tokenize(&self, text: &str) -> Vec<u32> {
        // Simplified BPE tokenization
        let mut tokens = Vec::new();
        let unk_id = self.vocab.get(&self.special_tokens.unk_token).copied().unwrap_or(1);

        // For demo, just split by characters
        for ch in text.chars() {
            let token = ch.to_string();
            tokens.push(*self.vocab.get(&token).unwrap_or(&unk_id));
        }

        tokens
    }

    fn sentencepiece_tokenize(&self, text: &str) -> Vec<u32> {
        // Simplified SentencePiece tokenization
        // In practice, this would implement the full SentencePiece algorithm
        self.bpe_tokenize(text)
    }

    fn decode_wordpiece(&self, tokens: Vec<String>) -> String {
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

    fn decode_bpe(&self, tokens: Vec<String>) -> String {
        // BPE tokens are typically joined without spaces
        tokens.join("")
    }

    fn decode_sentencepiece(&self, tokens: Vec<String>) -> String {
        // SentencePiece uses ▁ for spaces
        tokens.join("").replace("▁", " ")
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

    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = WasmTokenizer::new(TokenizerType::WordPiece);
        assert!(tokenizer.vocab_size() > 0);
    }

    #[test]
    fn test_encode_decode() {
        let tokenizer = WasmTokenizer::new(TokenizerType::WordPiece);
        let text = "hello world";
        let tokens = tokenizer.encode(text, false);
        assert!(!tokens.is_empty());

        let decoded = tokenizer.decode(tokens, false);
        // Decoded might not exactly match due to tokenization
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = WasmTokenizer::new(TokenizerType::WordPiece);
        let tokens = tokenizer.encode("test", true);

        // Should have [CLS] at start and [SEP] at end
        assert!(tokens.len() >= 3); // At least [CLS], one token, [SEP]
    }
}
