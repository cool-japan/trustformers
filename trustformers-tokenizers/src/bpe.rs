use crate::vocab::Vocab;
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};
use unicode_normalization::UnicodeNormalization;

#[derive(Debug)]
pub struct BPETokenizer {
    vocab: Vocab,
    merges: Vec<(String, String)>,
    merge_ranks: HashMap<(String, String), usize>,
    unk_token: String,
    pad_token: String,
    bos_token: String,
    eos_token: String,
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    cache: RwLock<HashMap<String, Vec<String>>>,
    // Enhanced byte-level BPE features
    normalize_unicode: bool,
    preserve_case: bool,
    handle_chinese_chars: bool,
    max_input_chars_per_word: usize,
}

// GPT-2 uses a special byte-level BPE
static GPT2_PATTERN: Lazy<Regex> = Lazy::new(|| {
    // Simplified regex without lookahead - matches the same patterns but less precisely
    Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
        .expect("valid regex")
});

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
            byte_encoder: self.byte_encoder.clone(),
            byte_decoder: self.byte_decoder.clone(),
            cache: RwLock::new(HashMap::new()), // Create new cache for clone
            normalize_unicode: self.normalize_unicode,
            preserve_case: self.preserve_case,
            handle_chinese_chars: self.handle_chinese_chars,
            max_input_chars_per_word: self.max_input_chars_per_word,
        }
    }
}

impl BPETokenizer {
    pub fn new(vocab: HashMap<String, u32>, merges: Vec<(String, String)>) -> Self {
        let mut merge_ranks = HashMap::new();
        for (i, merge) in merges.iter().enumerate() {
            merge_ranks.insert(merge.clone(), i);
        }

        let (byte_encoder, byte_decoder) = Self::create_byte_encoder();

        Self {
            vocab: Vocab::from_map(vocab),
            merges,
            merge_ranks,
            unk_token: "<|endoftext|>".to_string(),
            pad_token: "<|endoftext|>".to_string(),
            bos_token: "<|endoftext|>".to_string(),
            eos_token: "<|endoftext|>".to_string(),
            byte_encoder,
            byte_decoder,
            cache: RwLock::new(HashMap::new()),
            normalize_unicode: true,
            preserve_case: false,
            handle_chinese_chars: true,
            max_input_chars_per_word: 100,
        }
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
        tokenizer.normalize_unicode = true;
        tokenizer.preserve_case = true;
        tokenizer.handle_chinese_chars = false; // RoBERTa typically doesn't handle Chinese specially

        Ok(tokenizer)
    }

    /// Normalize text for improved Unicode handling
    fn normalize_text(&self, text: &str) -> String {
        if !self.normalize_unicode {
            return text.to_string();
        }

        // Apply Unicode normalization (NFC form)
        let normalized: String = text.nfc().collect();

        // Handle case normalization if needed
        let case_normalized =
            if self.preserve_case { normalized } else { normalized.to_lowercase() };

        // Handle Chinese characters specially if enabled
        if self.handle_chinese_chars {
            self.handle_chinese_text(&case_normalized)
        } else {
            case_normalized
        }
    }

    /// Special handling for Chinese characters
    fn handle_chinese_text(&self, text: &str) -> String {
        // Add spaces around Chinese characters for better tokenization
        let mut result = String::new();
        let mut prev_was_chinese = false;

        for ch in text.chars() {
            let is_chinese = self.is_chinese_char(ch);

            // Add space when transitioning between Chinese and non-Chinese text
            if (is_chinese != prev_was_chinese) && !result.is_empty() && !result.ends_with(' ') {
                result.push(' ');
            }

            result.push(ch);
            prev_was_chinese = is_chinese;
        }

        result
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

    /// Improved byte encoding with better Unicode support
    fn encode_as_bytes(&self, text: &str) -> Vec<u8> {
        // First normalize the text
        let normalized = self.normalize_text(text);

        // Convert to UTF-8 bytes
        normalized.as_bytes().to_vec()
    }

    /// Create byte-level encoder/decoder for GPT-2 style tokenization
    fn create_byte_encoder() -> (HashMap<u8, char>, HashMap<char, u8>) {
        let mut byte_encoder = HashMap::new();
        let mut byte_decoder = HashMap::new();

        // Printable ASCII characters (33-126) map to themselves
        let _n = 0;
        for b in 0..=255u8 {
            if (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174 {
                byte_encoder.insert(b, b as char);
                byte_decoder.insert(b as char, b);
            }
        }

        // Other bytes map to unicode characters starting from 256
        let mut char_val = 256u32;
        for b in 0..=255u8 {
            if let std::collections::hash_map::Entry::Vacant(e) = byte_encoder.entry(b) {
                e.insert(char::from_u32(char_val).unwrap());
                byte_decoder.insert(char::from_u32(char_val).unwrap(), b);
                char_val += 1;
            }
        }

        (byte_encoder, byte_decoder)
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        // Check cache first
        if let Ok(cache) = self.cache.read() {
            if let Some(cached) = cache.get(token) {
                return cached.clone();
            }
        }

        if token.is_empty() {
            return vec![];
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

        // Use improved byte encoding
        let word_bytes = self.encode_as_bytes(token);
        let mut word: Vec<String> =
            word_bytes.iter().map(|&b| self.byte_encoder[&b].to_string()).collect();

        if word.len() == 1 {
            return word;
        }

        // Optimized BPE algorithm with early termination
        loop {
            let pairs = Self::get_pairs(&word);
            if pairs.is_empty() {
                break;
            }

            // Find the best pair to merge with improved efficiency
            let mut min_rank = usize::MAX;
            let mut best_pair: Option<(String, String)> = None;

            for pair in &pairs {
                if let Some(&rank) = self.merge_ranks.get(pair) {
                    if rank < min_rank {
                        min_rank = rank;
                        best_pair = Some(pair.clone());
                    }
                }
            }

            if best_pair.is_none() {
                break;
            }

            let (first, second) = best_pair.unwrap();
            let mut new_word = Vec::with_capacity(word.len());
            let mut i = 0;

            while i < word.len() {
                if i < word.len() - 1 && word[i] == first && word[i + 1] == second {
                    new_word.push(format!("{}{}", first, second));
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }

            word = new_word;
            if word.len() == 1 {
                break;
            }
        }

        // Cache the result
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(token.to_string(), word.clone());
        }

        word
    }

    fn get_pairs(word: &[String]) -> HashSet<(String, String)> {
        let mut pairs = HashSet::new();
        for i in 0..word.len().saturating_sub(1) {
            pairs.insert((word[i].clone(), word[i + 1].clone()));
        }
        pairs
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = vec![];

        // Apply normalization first
        let normalized_text = self.normalize_text(text);

        // Use GPT-2 regex pattern to split text
        for mat in GPT2_PATTERN.find_iter(&normalized_text) {
            let word = mat.as_str();
            let bpe_tokens = self.bpe(word);
            tokens.extend(bpe_tokens);
        }

        tokens
    }

    /// Enhanced tokenization with offset tracking
    pub fn tokenize_with_offsets(&self, text: &str) -> (Vec<String>, Vec<(usize, usize)>) {
        let mut tokens = vec![];
        let mut offsets = vec![];

        let normalized_text = self.normalize_text(text);
        let mut current_offset = 0;

        // Use GPT-2 regex pattern to split text
        for mat in GPT2_PATTERN.find_iter(&normalized_text) {
            let word = mat.as_str();
            let start = current_offset;
            let end = start + word.len();

            let bpe_tokens = self.bpe(word);

            // Distribute the word offset across its BPE tokens
            let token_count = bpe_tokens.len();
            if token_count > 0 {
                let chars_per_token = word.chars().count() / token_count;
                let mut token_start = start;

                for (i, token) in bpe_tokens.iter().enumerate() {
                    let token_end =
                        if i == token_count - 1 { end } else { token_start + chars_per_token };

                    tokens.push(token.clone());
                    offsets.push((token_start, token_end));
                    token_start = token_end;
                }
            }

            current_offset = end;
        }

        (tokens, offsets)
    }
}

impl Tokenizer for BPETokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let tokens = self.tokenize(text);

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
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        let combined = format!("{} {}", text, text2);
        self.encode(&combined)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let tokens: Vec<String> = ids.iter().filter_map(|&id| self.vocab.get_token(id)).collect();

        // Join tokens and decode bytes
        let text = tokens.join("");
        let mut bytes = Vec::new();

        for ch in text.chars() {
            if let Some(&byte) = self.byte_decoder.get(&ch) {
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

    #[test]
    fn test_tokenize_with_offsets() {
        let tokenizer = create_test_tokenizer();

        let text = "hello world";
        let (tokens, offsets) = tokenizer.tokenize_with_offsets(text);

        assert_eq!(tokens.len(), offsets.len());
        assert!(!tokens.is_empty());
        assert!(!offsets.is_empty());

        // Check that offsets are reasonable
        for &(start, end) in &offsets {
            assert!(start <= end);
            assert!(end <= text.len());
        }
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

        assert!(!tokenizer.normalize_unicode);
        assert!(tokenizer.preserve_case);
        assert!(!tokenizer.handle_chinese_chars);
        assert_eq!(tokenizer.max_input_chars_per_word, 50);
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
}
