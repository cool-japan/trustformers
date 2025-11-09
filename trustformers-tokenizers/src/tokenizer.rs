use crate::alignment::{AlignedSpan, AlignmentConfig, AlignmentEngine, TokenAlignment};
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use tokenizers::{Encoding, Tokenizer as HFTokenizer};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

#[derive(Debug, Clone)]
pub struct TokenizedInputWithOffsets {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub token_type_ids: Option<Vec<u32>>,
    pub offset_mapping: Option<Vec<(usize, usize)>>,
    pub special_tokens_mask: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct TokenizedInputWithAlignment {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub token_type_ids: Option<Vec<u32>>,
    pub offset_mapping: Option<Vec<(usize, usize)>>,
    pub special_tokens_mask: Option<Vec<u8>>,
    pub word_alignments: Vec<TokenAlignment>,
    pub words: Vec<crate::alignment::Word>,
}

impl From<TokenizedInputWithOffsets> for TokenizedInput {
    fn from(input: TokenizedInputWithOffsets) -> Self {
        TokenizedInput {
            input_ids: input.input_ids,
            attention_mask: input.attention_mask,
            token_type_ids: input.token_type_ids,
            special_tokens_mask: input.special_tokens_mask,
            offset_mapping: input.offset_mapping,
            overflowing_tokens: None,
        }
    }
}

impl From<TokenizedInputWithAlignment> for TokenizedInput {
    fn from(input: TokenizedInputWithAlignment) -> Self {
        TokenizedInput {
            input_ids: input.input_ids,
            attention_mask: input.attention_mask,
            token_type_ids: input.token_type_ids,
            special_tokens_mask: input.special_tokens_mask,
            offset_mapping: input.offset_mapping,
            overflowing_tokens: None,
        }
    }
}

impl From<TokenizedInputWithAlignment> for TokenizedInputWithOffsets {
    fn from(input: TokenizedInputWithAlignment) -> Self {
        TokenizedInputWithOffsets {
            input_ids: input.input_ids,
            attention_mask: input.attention_mask,
            token_type_ids: input.token_type_ids,
            offset_mapping: input.offset_mapping,
            special_tokens_mask: input.special_tokens_mask,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenizerImpl {
    tokenizer: Arc<HFTokenizer>,
    do_lower_case: bool,
    max_length: Option<usize>,
    alignment_engine: Option<AlignmentEngine>,
}

impl TokenizerImpl {
    pub fn from_file(path: &Path) -> Result<Self> {
        let tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))?;
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            do_lower_case: false,
            max_length: Some(512),
            alignment_engine: None,
        })
    }

    pub fn from_pretrained(name: &str) -> Result<Self> {
        Self::from_pretrained_with_revision(name, None)
    }

    pub fn from_pretrained_with_revision(name: &str, revision: Option<&str>) -> Result<Self> {
        // Simplified version - in practice, this would download from HuggingFace Hub
        // For now, try to load from a local cache path
        let cache_dir = std::env::var("HF_HOME")
            .or_else(|_| std::env::var("TRANSFORMERS_CACHE"))
            .unwrap_or_else(|_| {
                format!(
                    "{}/.cache/huggingface/transformers",
                    std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string())
                )
            });

        // Include revision in path if specified
        let tokenizer_path = match revision {
            Some(rev) => format!("{}/{}/refs/{}/tokenizer.json", cache_dir, name, rev),
            None => format!("{}/{}/tokenizer.json", cache_dir, name),
        };
        let path = Path::new(&tokenizer_path);

        if path.exists() {
            Self::from_file(path)
        } else {
            Err(TrustformersError::other(anyhow::anyhow!(
                "Model '{}' not found locally. Please download it first or implement model downloading.",
                name
            ).to_string()))
        }
    }

    pub fn from_tokenizer_json(json_str: &str) -> Result<Self> {
        let tokenizer = HFTokenizer::from_str(json_str).map_err(|e: tokenizers::Error| {
            TrustformersError::other(anyhow::anyhow!(e).to_string())
        })?;
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            do_lower_case: false,
            max_length: Some(512),
            alignment_engine: None,
        })
    }

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let json = self
            .tokenizer
            .to_string(false)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))?;
        std::fs::write(path, json)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))?;
        Ok(())
    }

    pub fn to_json(&self) -> Result<String> {
        self.tokenizer
            .to_string(false)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))
    }

    pub fn with_config(mut self, do_lower_case: bool, max_length: Option<usize>) -> Self {
        self.do_lower_case = do_lower_case;
        self.max_length = max_length;
        self
    }

    pub fn encode_with_offsets(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<TokenizedInputWithOffsets> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))?;
        Ok(self.encoding_to_tokenized_input_with_offsets(encoding))
    }

    pub fn encode_pair_with_offsets(
        &self,
        text: &str,
        text2: &str,
        add_special_tokens: bool,
    ) -> Result<TokenizedInputWithOffsets> {
        let encoding = self
            .tokenizer
            .encode((text, text2), add_special_tokens)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))?;
        Ok(self.encoding_to_tokenized_input_with_offsets(encoding))
    }

    pub fn decode_with_special_tokens(
        &self,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))
    }

    pub fn get_vocab(&self) -> HashMap<String, u32> {
        self.tokenizer.get_vocab(false)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    /// Configure word alignment engine
    pub fn with_alignment_config(mut self, config: AlignmentConfig) -> Self {
        self.alignment_engine = Some(AlignmentEngine::new(config));
        self
    }

    /// Enable word alignment with default configuration
    pub fn with_word_alignment(mut self) -> Self {
        self.alignment_engine = Some(AlignmentEngine::new(AlignmentConfig::default()));
        self
    }

    /// Get mutable reference to alignment engine
    pub fn alignment_engine_mut(&mut self) -> Option<&mut AlignmentEngine> {
        self.alignment_engine.as_mut()
    }

    /// Encode text with word alignment
    pub fn encode_with_alignment(
        &mut self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<TokenizedInputWithAlignment> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))?;

        self.encoding_to_tokenized_input_with_alignment(text, encoding)
    }

    /// Encode text pair with word alignment
    pub fn encode_pair_with_alignment(
        &mut self,
        text: &str,
        text2: &str,
        add_special_tokens: bool,
    ) -> Result<TokenizedInputWithAlignment> {
        let encoding = self
            .tokenizer
            .encode((text, text2), add_special_tokens)
            .map_err(|e| TrustformersError::other(anyhow::anyhow!(e).to_string()))?;

        // Combine texts for alignment
        let combined_text = format!("{} {}", text, text2);
        self.encoding_to_tokenized_input_with_alignment(&combined_text, encoding)
    }

    /// Extract spans with word alignment
    pub fn extract_aligned_spans(
        &mut self,
        text: &str,
        spans: &[(usize, usize)],
        add_special_tokens: bool,
    ) -> Result<Vec<AlignedSpan>> {
        let input_with_alignment = self.encode_with_alignment(text, add_special_tokens)?;

        if let Some(engine) = &mut self.alignment_engine {
            engine.extract_spans(text, &input_with_alignment.word_alignments, spans)
        } else {
            Err(TrustformersError::other(
                "Word alignment engine not configured".to_string(),
            ))
        }
    }

    /// Preserve entity boundaries in tokenization
    pub fn preserve_entities(
        &mut self,
        text: &str,
        entities: &[(usize, usize, String)],
        add_special_tokens: bool,
    ) -> Result<Vec<AlignedSpan>> {
        let input_with_alignment = self.encode_with_alignment(text, add_special_tokens)?;

        if let Some(engine) = &mut self.alignment_engine {
            engine.preserve_entities(text, &input_with_alignment.word_alignments, entities)
        } else {
            Err(TrustformersError::other(
                "Word alignment engine not configured".to_string(),
            ))
        }
    }

    /// Get word boundaries for a specific token
    pub fn get_word_boundaries_for_token(
        &self,
        alignments: &[TokenAlignment],
        token_index: usize,
    ) -> Option<(usize, usize)> {
        if let Some(engine) = &self.alignment_engine {
            engine.get_word_boundaries_for_token(alignments, token_index)
        } else {
            None
        }
    }

    /// Check if tokens form a complete word
    pub fn tokens_form_complete_word(
        &self,
        alignments: &[TokenAlignment],
        token_indices: &[usize],
    ) -> bool {
        if let Some(engine) = &self.alignment_engine {
            engine.tokens_form_complete_word(alignments, token_indices)
        } else {
            false
        }
    }

    fn encoding_to_tokenized_input(&self, encoding: Encoding) -> TokenizedInput {
        TokenizedInput {
            input_ids: encoding.get_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().iter().map(|&x| x as u8).collect(),
            token_type_ids: if encoding.get_type_ids().is_empty() {
                None
            } else {
                Some(encoding.get_type_ids().to_vec())
            },
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        }
    }

    fn encoding_to_tokenized_input_with_offsets(
        &self,
        encoding: Encoding,
    ) -> TokenizedInputWithOffsets {
        let offset_mapping = if !encoding.get_offsets().is_empty() {
            Some(encoding.get_offsets().to_vec())
        } else {
            None
        };

        let special_tokens_mask = if !encoding.get_special_tokens_mask().is_empty() {
            Some(encoding.get_special_tokens_mask().iter().map(|&x| x as u8).collect())
        } else {
            None
        };

        TokenizedInputWithOffsets {
            input_ids: encoding.get_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().iter().map(|&x| x as u8).collect(),
            token_type_ids: if encoding.get_type_ids().is_empty() {
                None
            } else {
                Some(encoding.get_type_ids().to_vec())
            },
            offset_mapping,
            special_tokens_mask,
        }
    }

    fn encoding_to_tokenized_input_with_alignment(
        &mut self,
        text: &str,
        encoding: Encoding,
    ) -> Result<TokenizedInputWithAlignment> {
        let offset_mapping = if !encoding.get_offsets().is_empty() {
            Some(encoding.get_offsets().to_vec())
        } else {
            None
        };

        let special_tokens_mask = if !encoding.get_special_tokens_mask().is_empty() {
            Some(encoding.get_special_tokens_mask().iter().map(|&x| x as u8).collect())
        } else {
            None
        };

        // Perform word alignment if engine is available
        let (word_alignments, words) = if let Some(engine) = &mut self.alignment_engine {
            if let Some(ref offsets) = offset_mapping {
                let alignments =
                    engine.align_tokens_to_words(text, offsets, special_tokens_mask.as_deref())?;
                let words = engine.extract_words(text);
                (alignments, words)
            } else {
                // If no offsets available, create empty alignments
                (Vec::new(), Vec::new())
            }
        } else {
            return Err(TrustformersError::other(
                "Word alignment engine not configured".to_string(),
            ));
        };

        Ok(TokenizedInputWithAlignment {
            input_ids: encoding.get_ids().to_vec(),
            attention_mask: encoding.get_attention_mask().iter().map(|&x| x as u8).collect(),
            token_type_ids: if encoding.get_type_ids().is_empty() {
                None
            } else {
                Some(encoding.get_type_ids().to_vec())
            },
            offset_mapping,
            special_tokens_mask,
            word_alignments,
            words,
        })
    }
}

impl Tokenizer for TokenizerImpl {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let encoding = self.tokenizer.encode(text, false).map_err(|e| {
            trustformers_core::errors::TrustformersError::other(anyhow::anyhow!(e).to_string())
        })?;
        Ok(self.encoding_to_tokenized_input(encoding))
    }

    fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        let encoding = self.tokenizer.encode((text, text2), false).map_err(|e| {
            trustformers_core::errors::TrustformersError::other(anyhow::anyhow!(e).to_string())
        })?;
        Ok(self.encoding_to_tokenized_input(encoding))
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, false).map_err(|e| {
            trustformers_core::errors::TrustformersError::other(anyhow::anyhow!(e).to_string())
        })
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.tokenizer.get_vocab(false)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }
}

#[derive(Debug, Clone)]
pub enum TokenizerWrapper {
    WordPiece(crate::wordpiece::WordPieceTokenizer),
    BPE(crate::bpe::BPETokenizer),
    Unigram(crate::unigram::UnigramTokenizer),
    Char(crate::char::CharTokenizer),
    HuggingFace(TokenizerImpl),
}

impl Tokenizer for TokenizerWrapper {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        match self {
            TokenizerWrapper::WordPiece(t) => t.encode(text),
            TokenizerWrapper::BPE(t) => t.encode(text),
            TokenizerWrapper::Unigram(t) => t.encode(text),
            TokenizerWrapper::Char(t) => t.encode(text),
            TokenizerWrapper::HuggingFace(t) => t.encode(text),
        }
    }

    fn encode_pair(&self, text: &str, text2: &str) -> Result<TokenizedInput> {
        match self {
            TokenizerWrapper::WordPiece(t) => t.encode_pair(text, text2),
            TokenizerWrapper::BPE(t) => t.encode_pair(text, text2),
            TokenizerWrapper::Unigram(t) => t.encode_pair(text, text2),
            TokenizerWrapper::Char(t) => t.encode_pair(text, text2),
            TokenizerWrapper::HuggingFace(t) => t.encode_pair(text, text2),
        }
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        match self {
            TokenizerWrapper::WordPiece(t) => t.decode(ids),
            TokenizerWrapper::BPE(t) => t.decode(ids),
            TokenizerWrapper::Unigram(t) => t.decode(ids),
            TokenizerWrapper::Char(t) => t.decode(ids),
            TokenizerWrapper::HuggingFace(t) => t.decode(ids),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            TokenizerWrapper::WordPiece(t) => t.vocab_size(),
            TokenizerWrapper::BPE(t) => t.vocab_size(),
            TokenizerWrapper::Unigram(t) => t.vocab_size(),
            TokenizerWrapper::Char(t) => t.vocab_size(),
            TokenizerWrapper::HuggingFace(t) => t.vocab_size(),
        }
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        match self {
            TokenizerWrapper::WordPiece(t) => t.get_vocab(),
            TokenizerWrapper::BPE(t) => t.get_vocab(),
            TokenizerWrapper::Unigram(t) => t.get_vocab(),
            TokenizerWrapper::Char(t) => t.get_vocab(),
            TokenizerWrapper::HuggingFace(t) => t.get_vocab(),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            TokenizerWrapper::WordPiece(t) => t.token_to_id(token),
            TokenizerWrapper::BPE(t) => t.token_to_id(token),
            TokenizerWrapper::Unigram(t) => t.token_to_id(token),
            TokenizerWrapper::Char(t) => t.token_to_id(token),
            TokenizerWrapper::HuggingFace(t) => t.token_to_id(token),
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match self {
            TokenizerWrapper::WordPiece(t) => t.id_to_token(id),
            TokenizerWrapper::BPE(t) => t.id_to_token(id),
            TokenizerWrapper::Unigram(t) => t.id_to_token(id),
            TokenizerWrapper::Char(t) => t.id_to_token(id),
            TokenizerWrapper::HuggingFace(t) => t.id_to_token(id),
        }
    }
}

impl TokenizerWrapper {
    /// Load a tokenizer from a pretrained model or path
    pub fn from_pretrained<P: AsRef<Path>>(model_name_or_path: P) -> Result<Self> {
        let path = model_name_or_path.as_ref();

        // First try to load as a HuggingFace tokenizer from tokenizer.json
        let tokenizer_json_path = path.join("tokenizer.json");
        if tokenizer_json_path.exists() {
            let tokenizer = TokenizerImpl::from_file(&tokenizer_json_path)?;
            return Ok(TokenizerWrapper::HuggingFace(tokenizer));
        }

        // Try to load from tokenizer config
        let config_path = path.join("tokenizer_config.json");
        if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| TrustformersError::other(format!("I/O error: {}", e)))?;
            let config: serde_json::Value = serde_json::from_str(&config_str)
                .map_err(|e| TrustformersError::serialization_error(e.to_string()))?;

            if let Some(tokenizer_type) = config.get("tokenizer_type").and_then(|v| v.as_str()) {
                match tokenizer_type {
                    "WordPiece" => {
                        // Create a basic WordPiece tokenizer with minimal config
                        let vocab = std::collections::HashMap::new();
                        let tokenizer = crate::wordpiece::WordPieceTokenizer::new(vocab, false);
                        return Ok(TokenizerWrapper::WordPiece(tokenizer));
                    },
                    "BPE" => {
                        // Create a basic BPE tokenizer
                        let vocab = std::collections::HashMap::new();
                        let merges = Vec::new();
                        let tokenizer = crate::bpe::BPETokenizer::new(vocab, merges);
                        return Ok(TokenizerWrapper::BPE(tokenizer));
                    },
                    "Unigram" => {
                        // Create a basic Unigram tokenizer
                        let vocab = std::collections::HashMap::new();
                        let scores = std::collections::HashMap::new();
                        let tokenizer = crate::unigram::UnigramTokenizer::new(vocab, scores)?;
                        return Ok(TokenizerWrapper::Unigram(tokenizer));
                    },
                    "Character" => {
                        // Create a basic Character tokenizer
                        let vocab = std::collections::HashMap::new();
                        let tokenizer = crate::char::CharTokenizer::new(vocab);
                        return Ok(TokenizerWrapper::Char(tokenizer));
                    },
                    _ => {
                        return Err(TrustformersError::invalid_input(format!(
                            "Unsupported tokenizer type: {}",
                            tokenizer_type
                        )));
                    },
                }
            }
        }

        // If no config found, try to load as a HuggingFace tokenizer directly
        // (in case the path is a model name for hub download)
        match TokenizerImpl::from_pretrained(path.to_string_lossy().as_ref()) {
            Ok(tokenizer) => Ok(TokenizerWrapper::HuggingFace(tokenizer)),
            Err(_) => {
                // As a last resort, create a basic BPE tokenizer
                let vocab = std::collections::HashMap::new();
                let merges = Vec::new();
                Ok(TokenizerWrapper::BPE(crate::bpe::BPETokenizer::new(
                    vocab, merges,
                )))
            },
        }
    }

    /// Save the tokenizer to a directory path
    pub fn save_pretrained<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();

        // Create directory if it doesn't exist
        std::fs::create_dir_all(path)
            .map_err(|e| TrustformersError::other(format!("I/O error: {}", e)))?;

        match self {
            TokenizerWrapper::HuggingFace(tokenizer) => {
                // For HuggingFace tokenizers, use the existing save_to_file method
                let tokenizer_path = path.join("tokenizer.json");
                tokenizer.save_to_file(&tokenizer_path)
            },
            TokenizerWrapper::WordPiece(_) => {
                // For WordPiece, create a simple config file indicating the type
                let config_path = path.join("tokenizer_config.json");
                let config = serde_json::json!({
                    "tokenizer_type": "WordPiece",
                    "model_type": "WordPiece",
                    "version": "1.0"
                });
                std::fs::write(config_path, serde_json::to_string_pretty(&config).unwrap())
                    .map_err(|e| TrustformersError::other(format!("I/O error: {}", e)))?;

                // Note: Full WordPiece serialization would require implementing
                // vocabulary and config serialization for WordPieceTokenizer
                Ok(())
            },
            TokenizerWrapper::BPE(_) => {
                // For BPE, create a simple config file indicating the type
                let config_path = path.join("tokenizer_config.json");
                let config = serde_json::json!({
                    "tokenizer_type": "BPE",
                    "model_type": "BPE",
                    "version": "1.0"
                });
                std::fs::write(config_path, serde_json::to_string_pretty(&config).unwrap())
                    .map_err(|e| TrustformersError::other(format!("I/O error: {}", e)))?;
                Ok(())
            },
            TokenizerWrapper::Unigram(_) => {
                // For Unigram, create a simple config file indicating the type
                let config_path = path.join("tokenizer_config.json");
                let config = serde_json::json!({
                    "tokenizer_type": "Unigram",
                    "model_type": "Unigram",
                    "version": "1.0"
                });
                std::fs::write(config_path, serde_json::to_string_pretty(&config).unwrap())
                    .map_err(|e| TrustformersError::other(format!("I/O error: {}", e)))?;
                Ok(())
            },
            TokenizerWrapper::Char(_) => {
                // For Char, create a simple config file indicating the type
                let config_path = path.join("tokenizer_config.json");
                let config = serde_json::json!({
                    "tokenizer_type": "Character",
                    "model_type": "Character",
                    "version": "1.0"
                });
                std::fs::write(config_path, serde_json::to_string_pretty(&config).unwrap())
                    .map_err(|e| TrustformersError::other(format!("I/O error: {}", e)))?;
                Ok(())
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenized_input_with_offsets_conversion() {
        let input_with_offsets = TokenizedInputWithOffsets {
            input_ids: vec![101, 2023, 2003, 102],
            attention_mask: vec![1, 1, 1, 1],
            token_type_ids: Some(vec![0, 0, 0, 0]),
            offset_mapping: Some(vec![(0, 0), (0, 4), (5, 7), (0, 0)]),
            special_tokens_mask: Some(vec![1, 0, 0, 1]),
        };

        let regular_input: TokenizedInput = input_with_offsets.into();

        assert_eq!(regular_input.input_ids, vec![101, 2023, 2003, 102]);
        assert_eq!(regular_input.attention_mask, vec![1, 1, 1, 1]);
        assert_eq!(regular_input.token_type_ids, Some(vec![0, 0, 0, 0]));
    }

    #[test]
    fn test_tokenizer_wrapper_char() {
        let text = "Hello World!";
        let tokenizer = crate::char::CharTokenizer::from_text(text, 1000);
        let wrapper = TokenizerWrapper::Char(tokenizer);

        let encoded = wrapper.encode(text).unwrap();
        let decoded = wrapper.decode(&encoded.input_ids).unwrap();

        assert!(!encoded.input_ids.is_empty());
        assert!(decoded.contains("Hello"));
        assert!(wrapper.vocab_size() > 0);
    }

    #[test]
    fn test_tokenizer_from_json_string() {
        // Simple minimal tokenizer JSON for testing
        let json_str = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 0,
                    "content": "[PAD]",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                },
                {
                    "id": 1,
                    "content": "[UNK]",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }
            ],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "[PAD]": 0,
                    "[UNK]": 1,
                    "hello": 2,
                    "world": 3
                },
                "unk_token": "[UNK]"
            }
        }"#;

        let result = TokenizerImpl::from_tokenizer_json(json_str);
        assert!(result.is_ok());

        if let Ok(tokenizer) = result {
            assert_eq!(tokenizer.vocab_size(), 4);
            assert_eq!(tokenizer.token_to_id("hello"), Some(2));
            assert_eq!(tokenizer.id_to_token(3), Some("world".to_string()));
        }
    }
}
