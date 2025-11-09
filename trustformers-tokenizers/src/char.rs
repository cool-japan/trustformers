use crate::vocab::Vocab;
use std::collections::HashMap;
use trustformers_core::errors::{ErrorKind, Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Character-level tokenizer that splits text into individual characters
#[derive(Debug, Clone)]
pub struct CharTokenizer {
    vocab: Vocab,
    unk_token: String,
    pad_token: String,
    bos_token: String,
    eos_token: String,
    max_length: Option<usize>,
    lowercase: bool,
    handle_chinese_chars: bool,
}

impl CharTokenizer {
    /// Create a new character tokenizer with vocabulary
    pub fn new(vocab: HashMap<String, u32>) -> Self {
        Self {
            vocab: Vocab::from_map(vocab),
            unk_token: "[UNK]".to_string(),
            pad_token: "[PAD]".to_string(),
            bos_token: "[CLS]".to_string(),
            eos_token: "[SEP]".to_string(),
            max_length: None,
            lowercase: false,
            handle_chinese_chars: true,
        }
    }

    /// Create a new character tokenizer and build vocabulary from text
    pub fn from_text(text: &str, vocab_size: usize) -> Self {
        let mut char_counts: HashMap<String, u32> = HashMap::new();

        // Add special tokens first
        let special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];
        for (i, token) in special_tokens.iter().enumerate() {
            char_counts.insert(token.to_string(), u32::MAX - i as u32);
        }

        // Count character frequencies from both original and lowercased text
        // to ensure both cases are in vocabulary
        for ch in text.chars() {
            if !ch.is_control() || ch == '\n' || ch == '\t' {
                let char_str = ch.to_string();
                *char_counts.entry(char_str).or_insert(0) += 1;

                // Also add lowercase version
                let lower_char = ch.to_lowercase().to_string();
                if lower_char != ch.to_string() {
                    *char_counts.entry(lower_char).or_insert(0) += 1;
                }
            }
        }

        // Add space character if not present
        char_counts.entry(" ".to_string()).or_insert(1);

        // Sort by frequency and take top vocab_size
        let mut sorted_chars: Vec<_> = char_counts.into_iter().collect();
        sorted_chars.sort_by(|a, b| b.1.cmp(&a.1));

        let vocab: HashMap<String, u32> = sorted_chars
            .into_iter()
            .take(vocab_size)
            .enumerate()
            .map(|(i, (ch, _))| (ch, i as u32))
            .collect();

        Self::new(vocab)
    }

    /// Set special tokens
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

    /// Set maximum sequence length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Enable/disable lowercase conversion
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Enable/disable Chinese character handling
    pub fn with_chinese_chars(mut self, handle_chinese_chars: bool) -> Self {
        self.handle_chinese_chars = handle_chinese_chars;
        self
    }

    /// Preprocess text before tokenization
    fn preprocess_text(&self, text: &str) -> String {
        let mut processed = text.to_string();

        if self.lowercase {
            processed = processed.to_lowercase();
        }

        if self.handle_chinese_chars {
            // Add spaces around Chinese characters for better tokenization
            processed = self.add_spaces_around_chinese_chars(&processed);
        }

        // Normalize whitespace
        processed = processed.chars().map(|c| if c.is_whitespace() { ' ' } else { c }).collect();

        processed
    }

    /// Add spaces around Chinese characters
    fn add_spaces_around_chinese_chars(&self, text: &str) -> String {
        let mut result = String::new();
        let chars: Vec<char> = text.chars().collect();

        for (i, &ch) in chars.iter().enumerate() {
            let is_chinese = self.is_chinese_char(ch);
            let prev_is_chinese = i > 0 && self.is_chinese_char(chars[i - 1]);
            let next_is_chinese = i + 1 < chars.len() && self.is_chinese_char(chars[i + 1]);

            // Add space before Chinese char if previous is not Chinese
            if is_chinese && !prev_is_chinese && i > 0 && !chars[i - 1].is_whitespace() {
                result.push(' ');
            }

            result.push(ch);

            // Add space after Chinese char if next is not Chinese
            if is_chinese
                && !next_is_chinese
                && i + 1 < chars.len()
                && !chars[i + 1].is_whitespace()
            {
                result.push(' ');
            }
        }

        result
    }

    /// Check if character is Chinese
    fn is_chinese_char(&self, ch: char) -> bool {
        let code = ch as u32;
        // CJK Unified Ideographs and other CJK ranges
        (0x4E00..=0x9FFF).contains(&code) || // CJK Unified Ideographs
        (0x3400..=0x4DBF).contains(&code) || // CJK Extension A
        (0x20000..=0x2A6DF).contains(&code) || // CJK Extension B
        (0x2A700..=0x2B73F).contains(&code) || // CJK Extension C
        (0x2B740..=0x2B81F).contains(&code) || // CJK Extension D
        (0x2B820..=0x2CEAF).contains(&code) || // CJK Extension E
        (0x2CEB0..=0x2EBEF).contains(&code) || // CJK Extension F
        (0x30000..=0x3134F).contains(&code) // CJK Extension G
    }

    /// Convert characters to token IDs
    fn chars_to_ids(&self, chars: Vec<String>) -> Vec<u32> {
        chars
            .into_iter()
            .map(|ch| {
                self.vocab
                    .get_id(&ch)
                    .unwrap_or_else(|| self.vocab.get_id(&self.unk_token).unwrap_or(1))
            })
            .collect()
    }

    /// Convert token IDs to characters
    fn ids_to_chars(&self, ids: Vec<u32>) -> Result<Vec<String>> {
        ids.into_iter()
            .map(|id| {
                self.vocab.get_token(id).ok_or_else(|| {
                    TrustformersError::new(ErrorKind::TokenizationError {
                        reason: format!("Invalid token ID: {}", id),
                    })
                })
            })
            .collect()
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let processed_text = self.preprocess_text(text);

        // Split into characters
        let chars: Vec<String> = processed_text.chars().map(|c| c.to_string()).collect();

        // Convert to token IDs
        let mut token_ids = self.chars_to_ids(chars.clone());

        // Add special tokens
        if !self.bos_token.is_empty() {
            if let Some(bos_id) = self.vocab.get_id(&self.bos_token) {
                token_ids.insert(0, bos_id);
            }
        }

        if !self.eos_token.is_empty() {
            if let Some(eos_id) = self.vocab.get_id(&self.eos_token) {
                token_ids.push(eos_id);
            }
        }

        // Truncate if max_length is set
        if let Some(max_len) = self.max_length {
            if token_ids.len() > max_len {
                token_ids.truncate(max_len);
                // Ensure EOS token is at the end if it was added
                if !self.eos_token.is_empty() {
                    if let Some(eos_id) = self.vocab.get_id(&self.eos_token) {
                        token_ids[max_len - 1] = eos_id;
                    }
                }
            }
        }

        // Create attention mask (all 1s for non-padding tokens)
        let attention_mask = vec![1; token_ids.len()];

        Ok(TokenizedInput {
            input_ids: token_ids,
            attention_mask,
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let chars = self.ids_to_chars(token_ids.to_vec())?;

        // Filter out special tokens and join
        let text = chars
            .into_iter()
            .filter(|token| {
                token != &self.pad_token && token != &self.bos_token && token != &self.eos_token
            })
            .collect::<Vec<_>>()
            .join("");

        Ok(text)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.size()
    }

    fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        let processed_text1 = self.preprocess_text(text);
        let processed_text2 = self.preprocess_text(text2);

        // Split into characters
        let chars1: Vec<String> = processed_text1.chars().map(|c| c.to_string()).collect();
        let chars2: Vec<String> = processed_text2.chars().map(|c| c.to_string()).collect();

        // Convert to token IDs
        let token_ids1 = self.chars_to_ids(chars1);
        let token_ids2 = self.chars_to_ids(chars2);

        // Store lengths before moving
        let token_ids1_len = token_ids1.len();
        let _token_ids2_len = token_ids2.len();

        // Add special tokens
        let mut token_ids = Vec::new();

        if !self.bos_token.is_empty() {
            if let Some(bos_id) = self.vocab.get_id(&self.bos_token) {
                token_ids.push(bos_id);
            }
        }

        token_ids.extend(token_ids1);

        if !self.eos_token.is_empty() {
            if let Some(eos_id) = self.vocab.get_id(&self.eos_token) {
                token_ids.push(eos_id);
            }
        }

        token_ids.extend(token_ids2);

        if !self.eos_token.is_empty() {
            if let Some(eos_id) = self.vocab.get_id(&self.eos_token) {
                token_ids.push(eos_id);
            }
        }

        // Truncate if max_length is set
        if let Some(max_len) = self.max_length {
            if token_ids.len() > max_len {
                token_ids.truncate(max_len);
            }
        }

        // Create attention mask and token type IDs
        let attention_mask = vec![1; token_ids.len()];
        let token_type_ids = Some({
            let mut types = vec![0; token_ids1_len + 1]; // +1 for bos
            types.extend(vec![1; token_ids.len() - types.len()]); // rest are type 1
            types
        });

        Ok(TokenizedInput {
            input_ids: token_ids,
            attention_mask,
            token_type_ids,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
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

    #[test]
    fn test_char_tokenizer_basic() {
        let text = "Hello World!";
        let tokenizer = CharTokenizer::from_text(text, 1000);

        let encoded = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&encoded.input_ids).unwrap();

        assert_eq!(decoded.trim(), text);
    }

    #[test]
    fn test_char_tokenizer_chinese() {
        let text = "Hello 世界!";
        let tokenizer = CharTokenizer::from_text(text, 1000).with_chinese_chars(true);

        let encoded = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&encoded.input_ids).unwrap();

        // Should handle Chinese characters properly
        assert!(decoded.contains("世"));
        assert!(decoded.contains("界"));
    }

    #[test]
    fn test_char_tokenizer_lowercase() {
        let text = "Hello WORLD!";
        let tokenizer = CharTokenizer::from_text(text, 1000).with_lowercase(true);

        let encoded = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&encoded.input_ids).unwrap();

        assert_eq!(decoded.trim().to_lowercase(), text.to_lowercase());
    }

    #[test]
    fn test_char_tokenizer_max_length() {
        let text = "This is a very long sentence that should be truncated.";
        let max_len = 10;
        let tokenizer = CharTokenizer::from_text(text, 1000).with_max_length(max_len);

        let encoded = tokenizer.encode(text).unwrap();

        assert_eq!(encoded.input_ids.len(), max_len);
        assert_eq!(encoded.attention_mask.len(), max_len);
    }

    #[test]
    fn test_char_tokenizer_special_tokens() {
        let text = "Hello";
        // Create vocabulary that includes both default and custom special tokens
        let mut vocab = HashMap::new();
        vocab.insert("[PAD]".to_string(), 0);
        vocab.insert("[UNK]".to_string(), 1);
        vocab.insert("[BOS]".to_string(), 2);
        vocab.insert("[EOS]".to_string(), 3);

        // Add characters from the text
        for (i, ch) in text.chars().enumerate() {
            vocab.insert(ch.to_string(), 4 + i as u32);
        }

        let tokenizer = CharTokenizer::new(vocab).with_special_tokens(
            "[UNK]".to_string(),
            "[PAD]".to_string(),
            "[BOS]".to_string(),
            "[EOS]".to_string(),
        );

        let encoded = tokenizer.encode(text).unwrap();

        // Should have BOS and EOS tokens (text length + 2 special tokens)
        assert_eq!(encoded.input_ids.len(), text.len() + 2);
    }
}
