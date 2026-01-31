use std::collections::HashMap;
use trustformers_core::errors::Result;
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// CANINE (Character Architecture with No tokenization In Neural Encoders) tokenizer
/// Uses character-level encoding without requiring a fixed vocabulary
#[derive(Debug, Clone)]
pub struct CanineTokenizer {
    /// Maximum sequence length
    max_length: Option<usize>,
    /// Downsampling rate for sequence length reduction
    downsample_rate: usize,
    /// Hash table size for character hashing
    hash_size: usize,
    /// Special token IDs
    cls_token_id: u32,
    sep_token_id: u32,
    pad_token_id: u32,
    mask_token_id: u32,
    /// Whether to add special tokens
    add_special_tokens: bool,
}

impl CanineTokenizer {
    /// Create a new CANINE tokenizer
    pub fn new() -> Self {
        Self {
            max_length: None,
            downsample_rate: 1, // Default to no downsampling for compatibility
            hash_size: 16384,   // 2^14
            cls_token_id: 0,
            sep_token_id: 1,
            pad_token_id: 2,
            mask_token_id: 3,
            add_special_tokens: true,
        }
    }

    /// Set maximum sequence length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Set downsampling rate
    pub fn with_downsample_rate(mut self, downsample_rate: usize) -> Self {
        self.downsample_rate = downsample_rate;
        self
    }

    /// Set hash table size
    pub fn with_hash_size(mut self, hash_size: usize) -> Self {
        self.hash_size = hash_size;
        self
    }

    /// Set special token IDs
    pub fn with_special_tokens(
        mut self,
        cls_token_id: u32,
        sep_token_id: u32,
        pad_token_id: u32,
        mask_token_id: u32,
    ) -> Self {
        self.cls_token_id = cls_token_id;
        self.sep_token_id = sep_token_id;
        self.pad_token_id = pad_token_id;
        self.mask_token_id = mask_token_id;
        self
    }

    /// Enable/disable adding special tokens
    pub fn with_add_special_tokens(mut self, add_special_tokens: bool) -> Self {
        self.add_special_tokens = add_special_tokens;
        self
    }

    /// Hash a character to a token ID using FNV hash
    fn hash_char(&self, ch: char) -> u32 {
        let code_point = ch as u32;

        // Special handling for ASCII characters (0-127)
        if code_point <= 127 {
            // Reserve first 4 slots for special tokens, then ASCII chars
            return 4 + code_point;
        }

        // Use FNV-1a hash for non-ASCII characters
        let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
        let fnv_prime: u64 = 0x100000001b3; // FNV prime

        // Hash the Unicode code point
        let bytes = code_point.to_le_bytes();
        for byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(fnv_prime);
        }

        // Map to hash table size, avoiding special token IDs (0-131)
        let hashed = (hash % (self.hash_size as u64 - 132)) + 132;
        hashed as u32
    }

    /// Convert character sequence to token IDs
    fn chars_to_ids(&self, text: &str) -> Vec<u32> {
        text.chars().map(|ch| self.hash_char(ch)).collect()
    }

    /// Apply downsampling to reduce sequence length
    fn downsample_sequence(&self, token_ids: Vec<u32>) -> Vec<u32> {
        if self.downsample_rate <= 1 {
            return token_ids;
        }

        // Simple strided downsampling - take every nth token
        token_ids
            .into_iter()
            .enumerate()
            .filter_map(
                |(i, id)| {
                    if i % self.downsample_rate == 0 {
                        Some(id)
                    } else {
                        None
                    }
                },
            )
            .collect()
    }

    /// Prepare input with special tokens
    fn add_special_tokens_to_sequence(&self, token_ids: Vec<u32>) -> Vec<u32> {
        if !self.add_special_tokens {
            return token_ids;
        }

        let mut result = Vec::new();
        result.push(self.cls_token_id);
        result.extend(token_ids);
        result.push(self.sep_token_id);
        result
    }

    /// Create attention mask for the sequence
    fn create_attention_mask(&self, length: usize) -> Vec<u8> {
        vec![1; length]
    }

    /// Pad or truncate sequence to max length
    fn pad_or_truncate(
        &self,
        mut token_ids: Vec<u32>,
        mut attention_mask: Vec<u8>,
    ) -> (Vec<u32>, Vec<u8>) {
        if let Some(max_len) = self.max_length {
            if token_ids.len() > max_len {
                // Truncate
                token_ids.truncate(max_len);
                attention_mask.truncate(max_len);

                // Ensure SEP token at the end if special tokens are enabled
                if self.add_special_tokens && max_len > 0 {
                    token_ids[max_len - 1] = self.sep_token_id;
                }
            } else if token_ids.len() < max_len {
                // Pad
                let pad_length = max_len - token_ids.len();
                token_ids.extend(vec![self.pad_token_id; pad_length]);
                attention_mask.extend(vec![0; pad_length]);
            }
        }

        (token_ids, attention_mask)
    }
}

impl Default for CanineTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for CanineTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        // Convert characters to token IDs using hashing
        let char_ids = self.chars_to_ids(text);

        // Apply downsampling to reduce sequence length
        let downsampled_ids = self.downsample_sequence(char_ids);

        // Add special tokens
        let token_ids = self.add_special_tokens_to_sequence(downsampled_ids);

        // Create attention mask
        let attention_mask = self.create_attention_mask(token_ids.len());

        // Apply padding/truncation
        let (final_token_ids, final_attention_mask) =
            self.pad_or_truncate(token_ids, attention_mask);

        Ok(TokenizedInput {
            input_ids: final_token_ids,
            attention_mask: final_attention_mask,
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        // CANINE decoding is not straightforward since it uses hashing
        // This is a simplified version that handles special tokens
        let mut result = String::new();

        for &token_id in token_ids {
            if token_id == self.cls_token_id
                || token_id == self.sep_token_id
                || token_id == self.pad_token_id
            {
                continue; // Skip special tokens
            }

            // For ASCII characters (IDs 4-131), we can reverse the mapping
            if (4..=131).contains(&token_id) {
                let ascii_code = token_id - 4;
                if let Some(ch) = char::from_u32(ascii_code) {
                    result.push(ch);
                }
            } else {
                // For hashed non-ASCII characters, we can't easily reverse
                // In practice, CANINE models learn embeddings that don't require exact decoding
                result.push('�'); // Use replacement character
            }
        }

        Ok(result)
    }

    fn vocab_size(&self) -> usize {
        self.hash_size
    }

    fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        // Encode both texts separately
        let char_ids1 = self.chars_to_ids(text);
        let char_ids2 = self.chars_to_ids(text2);

        // Apply downsampling
        let downsampled_ids1 = self.downsample_sequence(char_ids1);
        let downsampled_ids2 = self.downsample_sequence(char_ids2);

        // Calculate first sequence length before moving downsampled_ids1
        let sep_count = if self.add_special_tokens { 1 } else { 0 };
        let first_seq_len = 1 + downsampled_ids1.len() + sep_count; // CLS + text1 + SEP

        // Combine with special tokens: [CLS] text1 [SEP] text2 [SEP]
        let mut token_ids = Vec::new();
        if self.add_special_tokens {
            token_ids.push(self.cls_token_id);
        }
        token_ids.extend(downsampled_ids1);
        if self.add_special_tokens {
            token_ids.push(self.sep_token_id);
        }
        token_ids.extend(downsampled_ids2);
        if self.add_special_tokens {
            token_ids.push(self.sep_token_id);
        }

        // Create attention mask
        let attention_mask = self.create_attention_mask(token_ids.len());

        // Create token type IDs (0 for first sequence, 1 for second)
        let mut token_type_ids = Vec::new();

        // First sequence (including CLS and first SEP)
        token_type_ids.extend(vec![0; first_seq_len]);
        // Second sequence (text2 + final SEP)
        token_type_ids.extend(vec![1; token_ids.len() - first_seq_len]);

        // Apply padding/truncation
        let (final_token_ids, final_attention_mask) =
            self.pad_or_truncate(token_ids, attention_mask);

        // Truncate token_type_ids to match final length
        token_type_ids.truncate(final_token_ids.len());

        Ok(TokenizedInput {
            input_ids: final_token_ids,
            attention_mask: final_attention_mask,
            token_type_ids: Some(token_type_ids),
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        // CANINE doesn't have a fixed vocabulary, so return empty HashMap
        HashMap::new()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        // CANINE uses hashing, so we can't directly convert tokens to IDs
        // For single characters, we can use the char_to_id method
        if token.len() == 1 {
            token.chars().next().map(|c| self.hash_char(c))
        } else {
            None
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        // CANINE uses hashing, so we can't directly convert IDs to tokens
        // For ASCII characters (IDs 4-131), we can reverse the mapping
        if (4..=131).contains(&id) {
            Some(((id - 4) as u8 as char).to_string())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canine_basic_encoding() {
        let tokenizer = CanineTokenizer::new();
        let text = "Hello";

        let encoded = tokenizer.encode(text).unwrap();

        // Should have CLS + characters + SEP
        assert_eq!(encoded.input_ids.len(), 7); // 1 + 5 + 1
        assert_eq!(encoded.input_ids[0], tokenizer.cls_token_id);
        assert_eq!(encoded.input_ids[6], tokenizer.sep_token_id);
    }

    #[test]
    fn test_canine_ascii_characters() {
        let tokenizer = CanineTokenizer::new();
        let text = "A";

        let encoded = tokenizer.encode(text).unwrap();

        // 'A' is ASCII 65, so token ID should be 4 + 65 = 69
        assert_eq!(encoded.input_ids[1], 69); // CLS(0) + A(69)
    }

    #[test]
    fn test_canine_downsampling() {
        let tokenizer = CanineTokenizer::new().with_downsample_rate(2);
        let text = "Hello World";

        let encoded = tokenizer.encode(text).unwrap();

        // With downsampling rate 2, should take every 2nd character (indices 0, 2, 4, ...)
        // Original: "Hello World" (11 chars: H e l l o   W o r l d)
        // Downsampled: H l o W r d (6 chars)
        let expected_downsampled_chars = text.len().div_ceil(2); // 6 chars
        let expected_total = expected_downsampled_chars + 2; // + CLS + SEP = 8

        assert_eq!(encoded.input_ids.len(), expected_total);
    }

    #[test]
    fn test_canine_max_length() {
        let tokenizer = CanineTokenizer::new().with_max_length(5);
        let text = "Hello World";

        let encoded = tokenizer.encode(text).unwrap();

        assert_eq!(encoded.input_ids.len(), 5);
        assert_eq!(encoded.attention_mask.len(), 5);
        // Last token should be SEP due to truncation
        assert_eq!(encoded.input_ids[4], tokenizer.sep_token_id);
    }

    #[test]
    fn test_canine_encode_pair() {
        let tokenizer = CanineTokenizer::new();
        let text1 = "Hello";
        let text2 = "World";

        let encoded = tokenizer.encode_pair(text1, text2).unwrap();

        // Should have CLS + text1 + SEP + text2 + SEP
        let expected_len = 1 + text1.len() + 1 + text2.len() + 1;
        assert_eq!(encoded.input_ids.len(), expected_len);

        // Check token type IDs
        assert!(encoded.token_type_ids.is_some());
        let token_types = encoded.token_type_ids.unwrap();
        assert_eq!(token_types.len(), expected_len);

        // First sequence should be type 0
        assert_eq!(token_types[0], 0); // CLS
        assert_eq!(token_types[1], 0); // First char of text1

        // Second sequence should be type 1
        let second_seq_start = 1 + text1.len() + 1; // CLS + text1 + SEP
        assert_eq!(token_types[second_seq_start], 1); // First char of text2
    }

    #[test]
    fn test_canine_unicode_handling() {
        let tokenizer = CanineTokenizer::new();
        let text = "Hello 世界"; // Mix of ASCII and Unicode

        let encoded = tokenizer.encode(text).unwrap();

        // Should handle both ASCII and Unicode characters
        assert!(encoded.input_ids.len() > 2); // At least CLS + some chars + SEP

        // ASCII characters should have predictable IDs
        let h_id = encoded.input_ids[1]; // 'H' after CLS
        assert_eq!(h_id, 4 + 72); // 'H' is ASCII 72
    }

    #[test]
    fn test_canine_decode_ascii() {
        let tokenizer = CanineTokenizer::new();
        let text = "Hello";

        let encoded = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&encoded.input_ids).unwrap();

        // Should decode ASCII characters correctly
        assert!(decoded.contains("Hello"));
    }

    #[test]
    fn test_canine_no_special_tokens() {
        let tokenizer = CanineTokenizer::new().with_add_special_tokens(false);
        let text = "Hi";

        let encoded = tokenizer.encode(text).unwrap();

        // Should only have the character tokens, no CLS/SEP
        assert_eq!(encoded.input_ids.len(), text.len());
    }
}
