use crate::vocab::{FlexibleVocab, LazyVocab, Vocab};
use anyhow::Result as AnyhowResult;
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Custom vocabulary tokenizer that can work with user-provided vocabularies
#[derive(Debug)]
pub struct CustomVocabTokenizer {
    vocab: FlexibleVocab,
    unk_token: String,
    unk_token_id: u32,
    special_tokens: HashMap<String, u32>,
    max_length: Option<usize>,
    padding_token: Option<String>,
    truncate: bool,
}

impl CustomVocabTokenizer {
    /// Create a new custom vocabulary tokenizer
    pub fn new(vocab: FlexibleVocab) -> AnyhowResult<Self> {
        let unk_token = "[UNK]".to_string();
        let unk_token_id = 0; // Default UNK token ID

        Ok(Self {
            vocab,
            unk_token,
            unk_token_id,
            special_tokens: HashMap::new(),
            max_length: None,
            padding_token: None,
            truncate: false,
        })
    }

    /// Create from an immediate vocabulary
    pub fn from_vocab(vocab: Vocab) -> AnyhowResult<Self> {
        let flex_vocab = FlexibleVocab::immediate(vocab);
        Self::new(flex_vocab)
    }

    /// Create from a lazy vocabulary
    pub fn from_lazy_vocab(lazy_vocab: LazyVocab) -> AnyhowResult<Self> {
        let flex_vocab = FlexibleVocab::lazy(lazy_vocab);
        Self::new(flex_vocab)
    }

    /// Create from a vocabulary file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> AnyhowResult<Self> {
        let flex_vocab = FlexibleVocab::from_file(path);
        Self::new(flex_vocab)
    }

    /// Create from a vocabulary map
    pub fn from_token_map(token_map: HashMap<String, u32>) -> AnyhowResult<Self> {
        let vocab = Vocab::from_map(token_map);
        Self::from_vocab(vocab)
    }

    /// Set the unknown token
    pub fn with_unk_token(mut self, unk_token: String, unk_token_id: u32) -> Self {
        self.unk_token = unk_token;
        self.unk_token_id = unk_token_id;
        self
    }

    /// Add a special token
    pub fn with_special_token(mut self, token: String, id: u32) -> Self {
        self.special_tokens.insert(token, id);
        self
    }

    /// Set maximum sequence length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Set padding token
    pub fn with_padding_token(mut self, padding_token: String) -> Self {
        self.padding_token = Some(padding_token);
        self
    }

    /// Enable truncation
    pub fn with_truncation(mut self, truncate: bool) -> Self {
        self.truncate = truncate;
        self
    }

    /// Get token ID for a given token
    fn get_token_id(&self, token: &str) -> AnyhowResult<u32> {
        // Check special tokens first
        if let Some(&id) = self.special_tokens.get(token) {
            return Ok(id);
        }

        // Check vocabulary
        if let Some(id) = self.vocab.get_id(token)? {
            Ok(id)
        } else {
            Ok(self.unk_token_id)
        }
    }

    /// Simple whitespace tokenization
    fn tokenize_text(&self, text: &str) -> Vec<String> {
        text.split_whitespace().map(|s| s.to_string()).collect()
    }

    /// Convert tokens to IDs
    fn tokens_to_ids(&self, tokens: &[String]) -> AnyhowResult<Vec<u32>> {
        tokens.iter().map(|token| self.get_token_id(token)).collect()
    }

    /// Apply truncation and padding
    fn apply_length_constraints(&self, mut ids: Vec<u32>) -> AnyhowResult<Vec<u32>> {
        // Apply truncation
        if let Some(max_len) = self.max_length {
            if self.truncate && ids.len() > max_len {
                ids.truncate(max_len);
            }
        }

        // Apply padding
        if let (Some(max_len), Some(pad_token)) = (self.max_length, &self.padding_token) {
            if ids.len() < max_len {
                let pad_id = self.get_token_id(pad_token)?;
                ids.resize(max_len, pad_id);
            }
        }

        Ok(ids)
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> AnyhowResult<usize> {
        self.vocab.size()
    }

    /// Check if vocabulary is loaded
    pub fn is_vocab_loaded(&self) -> bool {
        self.vocab.is_loaded()
    }

    /// Get the unknown token
    pub fn unk_token(&self) -> &str {
        &self.unk_token
    }

    /// Get special tokens
    pub fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special_tokens
    }

    /// Get token ID (helper method)
    pub fn token_to_id(&self, token: &str) -> AnyhowResult<Option<u32>> {
        if let Some(&id) = self.special_tokens.get(token) {
            Ok(Some(id))
        } else {
            Ok(self.vocab.get_id(token)?)
        }
    }

    /// Get token by ID (helper method)
    pub fn id_to_token(&self, id: u32) -> AnyhowResult<Option<String>> {
        // Check if it's a special token
        for (token, &special_id) in &self.special_tokens {
            if special_id == id {
                return Ok(Some(token.clone()));
            }
        }

        self.vocab.get_token(id)
    }
}

impl Tokenizer for CustomVocabTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let tokens = self.tokenize_text(text);
        let ids = self
            .tokens_to_ids(&tokens)
            .map_err(|e| TrustformersError::other(e.to_string()))?;
        let ids = self
            .apply_length_constraints(ids)
            .map_err(|e| TrustformersError::other(e.to_string()))?;

        // Create attention mask (all 1s for real tokens)
        let attention_mask = vec![1u8; ids.len()];

        Ok(TokenizedInput {
            input_ids: ids,
            attention_mask,
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let mut tokens = Vec::new();

        for &id in token_ids {
            if let Some(token) =
                self.vocab.get_token(id).map_err(|e| TrustformersError::other(e.to_string()))?
            {
                // Skip padding tokens
                if let Some(pad_token) = &self.padding_token {
                    if token == *pad_token {
                        continue;
                    }
                }
                tokens.push(token);
            } else {
                tokens.push(self.unk_token.clone());
            }
        }

        Ok(tokens.join(" "))
    }

    fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<TokenizedInput> {
        let tokens_a = self.tokenize_text(text_a);
        let tokens_b = self.tokenize_text(text_b);

        let mut all_tokens = tokens_a;
        all_tokens.extend(tokens_b);

        let ids = self
            .tokens_to_ids(&all_tokens)
            .map_err(|e| TrustformersError::other(e.to_string()))?;
        let ids = self
            .apply_length_constraints(ids)
            .map_err(|e| TrustformersError::other(e.to_string()))?;

        // Create token type IDs (0 for first sentence, 1 for second)
        let type_ids = {
            let tokens_a_len = self.tokenize_text(text_a).len();
            let mut type_ids = vec![0; tokens_a_len];
            type_ids.extend(vec![1; ids.len() - tokens_a_len]);
            if type_ids.len() > ids.len() {
                type_ids.truncate(ids.len());
            }
            type_ids
        };

        // Create attention mask (all 1s for real tokens)
        let attention_mask = vec![1u8; ids.len()];

        Ok(TokenizedInput {
            input_ids: ids,
            attention_mask,
            token_type_ids: Some(type_ids),
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    fn vocab_size(&self) -> usize {
        self.vocab.size().unwrap_or(0)
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        // For FlexibleVocab, we need to reconstruct the vocab map
        // This is a simple implementation - in practice, we'd want to optimize this
        match &self.vocab {
            FlexibleVocab::Immediate(vocab) => vocab.get_vocab().clone(),
            FlexibleVocab::Lazy(_) => {
                // For lazy vocab, we return empty for now
                // A proper implementation would load and iterate
                HashMap::new()
            },
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get_id(token).ok().flatten()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get_token(id).ok().flatten()
    }
}

/// Builder for CustomVocabTokenizer
pub struct CustomVocabTokenizerBuilder {
    vocab: Option<FlexibleVocab>,
    unk_token: String,
    unk_token_id: u32,
    special_tokens: HashMap<String, u32>,
    max_length: Option<usize>,
    padding_token: Option<String>,
    truncate: bool,
}

impl CustomVocabTokenizerBuilder {
    pub fn new() -> Self {
        Self {
            vocab: None,
            unk_token: "[UNK]".to_string(),
            unk_token_id: 0,
            special_tokens: HashMap::new(),
            max_length: None,
            padding_token: None,
            truncate: false,
        }
    }

    pub fn vocab(mut self, vocab: FlexibleVocab) -> Self {
        self.vocab = Some(vocab);
        self
    }

    pub fn vocab_from_map(mut self, token_map: HashMap<String, u32>) -> Self {
        let vocab = Vocab::from_map(token_map);
        self.vocab = Some(FlexibleVocab::immediate(vocab));
        self
    }

    pub fn vocab_from_file<P: AsRef<std::path::Path>>(mut self, path: P) -> Self {
        self.vocab = Some(FlexibleVocab::from_file(path));
        self
    }

    pub fn unk_token(mut self, token: String, id: u32) -> Self {
        self.unk_token = token;
        self.unk_token_id = id;
        self
    }

    pub fn special_token(mut self, token: String, id: u32) -> Self {
        self.special_tokens.insert(token, id);
        self
    }

    pub fn max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    pub fn padding_token(mut self, padding_token: String) -> Self {
        self.padding_token = Some(padding_token);
        self
    }

    pub fn truncation(mut self, truncate: bool) -> Self {
        self.truncate = truncate;
        self
    }

    pub fn build(self) -> AnyhowResult<CustomVocabTokenizer> {
        let vocab = self.vocab.ok_or_else(|| anyhow::anyhow!("Vocabulary is required"))?;

        Ok(CustomVocabTokenizer {
            vocab,
            unk_token: self.unk_token,
            unk_token_id: self.unk_token_id,
            special_tokens: self.special_tokens,
            max_length: self.max_length,
            padding_token: self.padding_token,
            truncate: self.truncate,
        })
    }
}

impl Default for CustomVocabTokenizerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_vocab_tokenizer_basic() {
        let mut token_map = HashMap::new();
        token_map.insert("hello".to_string(), 1);
        token_map.insert("world".to_string(), 2);
        token_map.insert("[UNK]".to_string(), 0);

        let tokenizer = CustomVocabTokenizer::from_token_map(token_map).unwrap();

        let result = tokenizer.encode("hello world").unwrap();
        assert_eq!(result.input_ids, vec![1, 2]);

        let decoded = tokenizer.decode(&result.input_ids).unwrap();
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_custom_vocab_tokenizer_unk() {
        let mut token_map = HashMap::new();
        token_map.insert("hello".to_string(), 1);
        token_map.insert("[UNK]".to_string(), 0);

        let tokenizer = CustomVocabTokenizer::from_token_map(token_map).unwrap();

        let result = tokenizer.encode("hello unknown").unwrap();
        assert_eq!(result.input_ids, vec![1, 0]); // unknown -> [UNK]
    }

    #[test]
    fn test_custom_vocab_tokenizer_special_tokens() {
        let mut token_map = HashMap::new();
        token_map.insert("hello".to_string(), 1);
        token_map.insert("world".to_string(), 2);

        let tokenizer = CustomVocabTokenizer::from_token_map(token_map)
            .unwrap()
            .with_special_token("[CLS]".to_string(), 100)
            .with_special_token("[SEP]".to_string(), 101);

        assert_eq!(tokenizer.token_to_id("[CLS]").unwrap(), Some(100));
        assert_eq!(tokenizer.token_to_id("[SEP]").unwrap(), Some(101));
        assert_eq!(
            tokenizer.id_to_token(100).unwrap(),
            Some("[CLS]".to_string())
        );
    }

    #[test]
    fn test_custom_vocab_tokenizer_pair() {
        let mut token_map = HashMap::new();
        token_map.insert("hello".to_string(), 1);
        token_map.insert("world".to_string(), 2);
        token_map.insert("from".to_string(), 3);
        token_map.insert("rust".to_string(), 4);

        let tokenizer = CustomVocabTokenizer::from_token_map(token_map).unwrap();

        let result = tokenizer.encode_pair("hello world", "from rust").unwrap();
        assert_eq!(result.input_ids, vec![1, 2, 3, 4]);
        assert_eq!(result.token_type_ids, Some(vec![0, 0, 1, 1]));
    }

    #[test]
    fn test_custom_vocab_tokenizer_truncation() {
        let mut token_map = HashMap::new();
        token_map.insert("a".to_string(), 1);
        token_map.insert("b".to_string(), 2);
        token_map.insert("c".to_string(), 3);

        let tokenizer = CustomVocabTokenizer::from_token_map(token_map)
            .unwrap()
            .with_max_length(2)
            .with_truncation(true);

        let result = tokenizer.encode("a b c").unwrap();
        assert_eq!(result.input_ids, vec![1, 2]); // truncated to 2 tokens
    }

    #[test]
    fn test_custom_vocab_tokenizer_padding() {
        let mut token_map = HashMap::new();
        token_map.insert("hello".to_string(), 1);
        token_map.insert("[PAD]".to_string(), 99);

        let tokenizer = CustomVocabTokenizer::from_token_map(token_map)
            .unwrap()
            .with_max_length(3)
            .with_padding_token("[PAD]".to_string());

        let result = tokenizer.encode("hello").unwrap();
        assert_eq!(result.input_ids, vec![1, 99, 99]); // padded to length 3
    }

    #[test]
    fn test_custom_vocab_tokenizer_builder() {
        let mut token_map = HashMap::new();
        token_map.insert("test".to_string(), 1);

        let tokenizer = CustomVocabTokenizerBuilder::new()
            .vocab_from_map(token_map)
            .unk_token("[UNK]".to_string(), 0)
            .special_token("[CLS]".to_string(), 100)
            .max_length(10)
            .truncation(true)
            .build()
            .unwrap();

        assert_eq!(tokenizer.unk_token(), "[UNK]");
        assert_eq!(tokenizer.token_to_id("test").unwrap(), Some(1));
        assert_eq!(tokenizer.token_to_id("[CLS]").unwrap(), Some(100));
    }
}
