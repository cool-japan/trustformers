use std::collections::HashMap;
use trustformers_core::errors::Result;
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// The size of the Unicode code point space CANINE's token IDs range over
/// (`0..=0x10FFFF`, i.e. `char::MAX as u32 + 1`), matching HuggingFace's
/// `CanineTokenizer.UNICODE_VOCAB_SIZE`.
pub const UNICODE_VOCAB_SIZE: usize = 0x110000;

/// CANINE (Character Architecture with No tokenization In Neural Encoders) tokenizer.
///
/// Uses character-level encoding without requiring a fixed vocabulary: a
/// token ID is simply the Unicode code point (`char as u32`) of the
/// character it represents, exactly as HuggingFace's `CanineTokenizer`
/// does. This keeps IDs directly compatible with a real CANINE checkpoint's
/// embedding table (indexed by [`UNICODE_VOCAB_SIZE`]) and makes decoding
/// exact for every character, not just ASCII.
///
/// Downsampling is deliberately *not* performed here: real CANINE
/// downsamples inside the model via a strided convolution over the full
/// per-character hidden states, which is lossy-but-informed (every
/// character still contributes to the representation before pooling).
/// Dropping characters out of the raw input sequence before the model ever
/// sees them, as an earlier version of this tokenizer did, discards input
/// data outright rather than downsampling a representation of it.
#[derive(Debug, Clone)]
pub struct CanineTokenizer {
    /// Maximum sequence length
    max_length: Option<usize>,
    /// Special token IDs
    cls_token_id: u32,
    sep_token_id: u32,
    pad_token_id: u32,
    mask_token_id: u32,
    /// Whether to add special tokens
    add_special_tokens: bool,
}

impl CanineTokenizer {
    /// Create a new CANINE tokenizer.
    ///
    /// Special token IDs default to HuggingFace `CanineTokenizer`'s own
    /// values, placed in the Unicode Private Use Area (`U+E000..U+F8FF`)
    /// so they never collide with a real character's code point: `PAD` =
    /// 0, `CLS` = `0xE000`, `SEP` = `0xE001`, `MASK` = `0xE003`.
    pub fn new() -> Self {
        Self {
            max_length: None,
            cls_token_id: 0xE000,
            sep_token_id: 0xE001,
            pad_token_id: 0,
            mask_token_id: 0xE003,
            add_special_tokens: true,
        }
    }

    /// Set maximum sequence length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
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

    /// Convert a character sequence to token IDs: each character's own
    /// Unicode code point, exactly as HuggingFace's `CanineTokenizer` does.
    /// Unlike a hash, this mapping is injective (no two distinct
    /// characters ever collide on the same ID) and trivially reversible by
    /// [`char::from_u32`], which is what makes [`Tokenizer::decode`] exact
    /// for every character rather than only ASCII.
    fn chars_to_ids(&self, text: &str) -> Vec<u32> {
        text.chars().map(|ch| ch as u32).collect()
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
        // Convert characters to token IDs (each character's own code point).
        let char_ids = self.chars_to_ids(text);

        // Add special tokens
        let token_ids = self.add_special_tokens_to_sequence(char_ids);

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
        // Each ID is a Unicode code point, so decoding is the exact
        // inverse of `chars_to_ids`: `char::from_u32` for every ID that
        // isn't one of this tokenizer's special tokens.
        let mut result = String::new();

        for &token_id in token_ids {
            if token_id == self.cls_token_id
                || token_id == self.sep_token_id
                || token_id == self.pad_token_id
                || token_id == self.mask_token_id
            {
                continue; // Skip special tokens
            }

            match char::from_u32(token_id) {
                Some(ch) => result.push(ch),
                // Not every u32 is a valid Unicode scalar value (surrogate
                // code points, or values above U+10FFFF): `encode` never
                // produces such an ID, but `decode` is a public API that
                // can be called on arbitrary/adversarial input, so an
                // invalid ID still needs a defined (if lossy) fallback
                // rather than panicking.
                None => result.push('\u{fffd}'),
            }
        }

        Ok(result)
    }

    fn vocab_size(&self) -> usize {
        UNICODE_VOCAB_SIZE
    }

    fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        // Encode both texts separately
        let char_ids1 = self.chars_to_ids(text);
        let char_ids2 = self.chars_to_ids(text2);

        // Calculate first sequence length before moving char_ids1
        let sep_count = if self.add_special_tokens { 1 } else { 0 };
        let first_seq_len = 1 + char_ids1.len() + sep_count; // CLS + text1 + SEP

        // Combine with special tokens: [CLS] text1 [SEP] text2 [SEP]
        let mut token_ids = Vec::new();
        if self.add_special_tokens {
            token_ids.push(self.cls_token_id);
        }
        token_ids.extend(char_ids1);
        if self.add_special_tokens {
            token_ids.push(self.sep_token_id);
        }
        token_ids.extend(char_ids2);
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
        // CANINE's "vocabulary" is the entire Unicode code point space
        // (see `vocab_size`/`UNICODE_VOCAB_SIZE`), not a finite enumerable
        // piece list the way a BPE/WordPiece vocabulary is, so there is no
        // meaningful finite map to return here.
        HashMap::new()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        // Every token is exactly one character; `chars().count()` (not
        // `len()`, which counts UTF-8 *bytes* and would wrongly reject
        // any single non-ASCII character, e.g. "世".len() == 3) is the
        // correct one-character check.
        let mut chars = token.chars();
        let first = chars.next()?;
        if chars.next().is_some() {
            return None; // more than one character
        }
        Some(first as u32)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        if id == self.cls_token_id
            || id == self.sep_token_id
            || id == self.pad_token_id
            || id == self.mask_token_id
        {
            return None;
        }
        char::from_u32(id).map(|c| c.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canine_basic_encoding() {
        let tokenizer = CanineTokenizer::new();
        let text = "Hello";

        let encoded = tokenizer.encode(text).expect("Encoding failed");

        // Should have CLS + characters + SEP
        assert_eq!(encoded.input_ids.len(), 7); // 1 + 5 + 1
        assert_eq!(encoded.input_ids[0], tokenizer.cls_token_id);
        assert_eq!(encoded.input_ids[6], tokenizer.sep_token_id);
    }

    #[test]
    fn test_canine_ascii_characters() {
        let tokenizer = CanineTokenizer::new();
        let text = "A";

        let encoded = tokenizer.encode(text).expect("Encoding failed");

        // The token ID is simply the character's own code point: 'A' is
        // U+0041 == 65, not a hashed/offset value.
        assert_eq!(encoded.input_ids[1], 65); // CLS + A(65)
        assert_eq!(encoded.input_ids[1], 'A' as u32);
    }

    /// Regression test: an earlier version of this tokenizer applied
    /// strided "downsampling" *inside the tokenizer* by dropping every
    /// other raw character before the model ever saw it -- genuine input
    /// data loss, not the strided-convolution downsampling a real CANINE
    /// model performs internally over full per-character representations.
    /// The tokenizer must now preserve every character.
    #[test]
    fn test_canine_preserves_every_character() {
        let tokenizer = CanineTokenizer::new();
        let text = "Hello World"; // 11 characters
        let encoded = tokenizer.encode(text).expect("Encoding failed");

        assert_eq!(encoded.input_ids.len(), text.chars().count() + 2); // + CLS + SEP
    }

    #[test]
    fn test_canine_max_length() {
        let tokenizer = CanineTokenizer::new().with_max_length(5);
        let text = "Hello World";

        let encoded = tokenizer.encode(text).expect("Encoding failed");

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

        let encoded = tokenizer.encode_pair(text1, text2).expect("Operation failed in test");

        // Should have CLS + text1 + SEP + text2 + SEP
        let expected_len = 1 + text1.len() + 1 + text2.len() + 1;
        assert_eq!(encoded.input_ids.len(), expected_len);

        // Check token type IDs
        assert!(encoded.token_type_ids.is_some());
        let token_types = encoded.token_type_ids.expect("Operation failed in test");
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

        let encoded = tokenizer.encode(text).expect("Encoding failed");

        // Should handle both ASCII and Unicode characters
        assert!(encoded.input_ids.len() > 2); // At least CLS + some chars + SEP

        // Every character's ID is exactly its own code point.
        let h_id = encoded.input_ids[1]; // 'H' after CLS
        assert_eq!(h_id, 'H' as u32);
        let shi_id = encoded.input_ids[7]; // '世' (index: CLS,H,e,l,l,o,' ',世)
        assert_eq!(shi_id, '世' as u32);
    }

    #[test]
    fn test_canine_decode_ascii() {
        let tokenizer = CanineTokenizer::new();
        let text = "Hello";

        let encoded = tokenizer.encode(text).expect("Encoding failed");
        let decoded = tokenizer.decode(&encoded.input_ids).expect("Decoding failed");

        // Should decode ASCII characters correctly
        assert!(decoded.contains("Hello"));
    }

    /// Regression test: the old FNV-hash mapping for non-ASCII characters
    /// was many-to-one and irreversible, so `decode` conceded defeat and
    /// emitted the U+FFFD replacement character for every non-ASCII
    /// input. Token IDs are now Unicode code points, so decoding is exact.
    #[test]
    fn test_canine_decode_round_trips_non_ascii() {
        let tokenizer = CanineTokenizer::new();
        let text = "Hello 世界 café";

        let encoded = tokenizer.encode(text).expect("Encoding failed");
        assert!(
            !encoded.input_ids.contains(&0xFFFD),
            "non-ASCII characters must not collapse onto the replacement-character ID"
        );

        let decoded = tokenizer.decode(&encoded.input_ids).expect("Decoding failed");
        assert_eq!(
            decoded, text,
            "encode -> decode must be the identity for any Unicode text"
        );
        assert!(!decoded.contains('\u{fffd}'));
    }

    /// Regression test: token IDs must be unique per character (an earlier
    /// hash-based mapping folded distinct non-ASCII characters onto the
    /// same ID whenever they collided in the hash table).
    #[test]
    fn test_canine_distinct_characters_get_distinct_ids() {
        let tokenizer = CanineTokenizer::new();
        let text = "アイウエオ日本語한국어";
        let ids = tokenizer.encode(text).expect("Encoding failed").input_ids;
        // Strip CLS/SEP; every remaining ID should equal that character's
        // own code point, so distinct characters are trivially distinct
        // IDs (this also directly checks compatibility with a real CANINE
        // checkpoint's embedding table, which is indexed by code point).
        let content = &ids[1..ids.len() - 1];
        for (id, ch) in content.iter().zip(text.chars()) {
            assert_eq!(*id, ch as u32);
        }
    }

    #[test]
    fn test_canine_vocab_size_is_full_unicode_range() {
        let tokenizer = CanineTokenizer::new();
        assert_eq!(tokenizer.vocab_size(), UNICODE_VOCAB_SIZE);
        assert_eq!(tokenizer.vocab_size(), 0x110000);
    }

    #[test]
    fn test_canine_token_to_id_handles_multibyte_char() {
        let tokenizer = CanineTokenizer::new();
        // "世" is 3 UTF-8 bytes but exactly one character; the old
        // `token.len() == 1` (byte-length) check wrongly rejected it.
        assert_eq!(tokenizer.token_to_id("世"), Some('世' as u32));
        assert_eq!(tokenizer.id_to_token('世' as u32), Some("世".to_string()));
    }

    #[test]
    fn test_canine_no_special_tokens() {
        let tokenizer = CanineTokenizer::new().with_add_special_tokens(false);
        let text = "Hi";

        let encoded = tokenizer.encode(text).expect("Encoding failed");

        // Should only have the character tokens, no CLS/SEP
        assert_eq!(encoded.input_ids.len(), text.len());
    }
}
