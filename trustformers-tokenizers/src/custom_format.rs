use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Represents a custom tokenizer format specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomTokenizerFormat {
    pub format_name: String,
    pub format_version: String,
    pub vocabulary: CustomVocabulary,
    pub special_tokens: Vec<CustomSpecialToken>,
    pub normalization_rules: Vec<NormalizationRule>,
    pub pre_tokenization_rules: Vec<PreTokenizationRule>,
    pub post_processing_rules: Vec<PostProcessingRule>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Custom vocabulary definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomVocabulary {
    pub vocab_type: VocabularyType,
    pub tokens: Vec<CustomToken>,
    pub size: usize,
    pub unk_token: Option<String>,
    pub special_token_mapping: HashMap<String, u32>,
}

/// Types of vocabularies supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VocabularyType {
    WordLevel,
    SubwordBPE,
    SubwordWordPiece,
    CharacterLevel,
    SentencePiece,
    Custom(String),
}

/// Custom token with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomToken {
    pub text: String,
    pub id: u32,
    pub frequency: Option<f64>,
    pub is_special: bool,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Custom special token definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomSpecialToken {
    pub token: String,
    pub id: u32,
    pub token_type: SpecialTokenType,
    pub context: Option<String>,
}

/// Types of special tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpecialTokenType {
    Pad,
    Unk,
    Cls,
    Sep,
    Mask,
    BOS,
    EOS,
    UserDefined(String),
}

/// Normalization rule for text preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationRule {
    pub rule_type: NormalizationType,
    pub pattern: Option<String>,
    pub replacement: Option<String>,
    pub enabled: bool,
}

/// Types of normalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationType {
    Lowercase,
    RemoveAccents,
    NormalizeWhitespace,
    NormalizeUnicode,
    RemovePunctuation,
    Regex(String),
    Custom(String),
}

/// Pre-tokenization rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreTokenizationRule {
    pub rule_type: PreTokenizationType,
    pub pattern: Option<String>,
    pub enabled: bool,
}

/// Types of pre-tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreTokenizationType {
    WhitespaceSplit,
    PunctuationSplit,
    WordBoundary,
    Regex(String),
    Custom(String),
}

/// Post-processing rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostProcessingRule {
    pub rule_type: PostProcessingType,
    pub parameters: HashMap<String, serde_json::Value>,
    pub enabled: bool,
}

/// Types of post-processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PostProcessingType {
    AddSpecialTokens,
    Truncation,
    Padding,
    AttentionMask,
    TokenTypeIds,
    Custom(String),
}

/// Custom format tokenizer implementation
#[derive(Debug, Clone)]
pub struct CustomFormatTokenizer {
    format: CustomTokenizerFormat,
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    max_length: Option<usize>,
}

impl CustomFormatTokenizer {
    /// Create a new tokenizer from a custom format
    pub fn from_format(format: CustomTokenizerFormat) -> Result<Self> {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        // Build vocabulary maps
        for token in &format.vocabulary.tokens {
            token_to_id.insert(token.text.clone(), token.id);
            id_to_token.insert(token.id, token.text.clone());
        }

        // Add special tokens
        for special_token in &format.special_tokens {
            token_to_id.insert(special_token.token.clone(), special_token.id);
            id_to_token.insert(special_token.id, special_token.token.clone());
        }

        Ok(Self {
            format,
            token_to_id,
            id_to_token,
            max_length: Some(512),
        })
    }

    /// Load tokenizer from custom format file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            TrustformersError::other(anyhow::anyhow!("Failed to read file: {}", e).to_string())
        })?;
        let format: CustomTokenizerFormat = serde_json::from_str(&content).map_err(|e| {
            TrustformersError::other(anyhow::anyhow!("Failed to parse format: {}", e).to_string())
        })?;
        Self::from_format(format)
    }

    /// Save tokenizer to custom format file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.format).map_err(|e| {
            TrustformersError::other(
                anyhow::anyhow!("Failed to serialize format: {}", e).to_string(),
            )
        })?;
        std::fs::write(path, content).map_err(|e| {
            TrustformersError::other(anyhow::anyhow!("Failed to write file: {}", e).to_string())
        })?;
        Ok(())
    }

    /// Set maximum sequence length
    pub fn with_max_length(mut self, max_length: Option<usize>) -> Self {
        self.max_length = max_length;
        self
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.format.vocabulary.size
    }

    /// Get token ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token from ID
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.id_to_token.get(&id).cloned()
    }

    /// Get vocabulary
    pub fn get_vocab(&self) -> &HashMap<String, u32> {
        &self.token_to_id
    }

    /// Apply normalization rules
    fn normalize_text(&self, text: &str) -> String {
        let mut normalized = text.to_string();

        for rule in &self.format.normalization_rules {
            if !rule.enabled {
                continue;
            }

            normalized = match &rule.rule_type {
                NormalizationType::Lowercase => normalized.to_lowercase(),
                NormalizationType::RemoveAccents => self.remove_accents(&normalized),
                NormalizationType::NormalizeWhitespace => {
                    normalized.split_whitespace().collect::<Vec<_>>().join(" ")
                },
                NormalizationType::NormalizeUnicode => {
                    unicode_normalization::UnicodeNormalization::nfc(normalized.as_str()).collect()
                },
                NormalizationType::RemovePunctuation => {
                    normalized.chars().filter(|c| !c.is_ascii_punctuation()).collect()
                },
                NormalizationType::Regex(_pattern) => {
                    if let (Some(pattern), Some(replacement)) = (&rule.pattern, &rule.replacement) {
                        if let Ok(re) = regex::Regex::new(pattern) {
                            re.replace_all(&normalized, replacement).to_string()
                        } else {
                            normalized
                        }
                    } else {
                        normalized
                    }
                },
                NormalizationType::Custom(_) => {
                    // Custom normalization would be implemented based on specific needs
                    normalized
                },
            };
        }

        normalized
    }

    /// Remove accents from text
    fn remove_accents(&self, text: &str) -> String {
        use unicode_normalization::UnicodeNormalization;
        text.nfd()
            .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
            .collect()
    }

    /// Apply pre-tokenization rules
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = vec![text.to_string()];

        for rule in &self.format.pre_tokenization_rules {
            if !rule.enabled {
                continue;
            }

            let mut new_tokens = Vec::new();
            for token in tokens {
                match &rule.rule_type {
                    PreTokenizationType::WhitespaceSplit => {
                        new_tokens.extend(token.split_whitespace().map(|s| s.to_string()));
                    },
                    PreTokenizationType::PunctuationSplit => {
                        let mut current = String::new();
                        for ch in token.chars() {
                            if ch.is_ascii_punctuation() {
                                if !current.is_empty() {
                                    new_tokens.push(current.clone());
                                    current.clear();
                                }
                                new_tokens.push(ch.to_string());
                            } else {
                                current.push(ch);
                            }
                        }
                        if !current.is_empty() {
                            new_tokens.push(current);
                        }
                    },
                    PreTokenizationType::WordBoundary => {
                        // Simple word boundary implementation
                        let words: Vec<String> = token
                            .split(|c: char| !c.is_alphanumeric())
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string())
                            .collect();
                        new_tokens.extend(words);
                    },
                    PreTokenizationType::Regex(pattern) => {
                        if let Ok(re) = regex::Regex::new(pattern) {
                            let splits: Vec<String> = re
                                .split(&token)
                                .filter(|s| !s.is_empty())
                                .map(|s| s.to_string())
                                .collect();
                            new_tokens.extend(splits);
                        } else {
                            new_tokens.push(token);
                        }
                    },
                    PreTokenizationType::Custom(_) => {
                        // Custom pre-tokenization would be implemented based on specific needs
                        new_tokens.push(token);
                    },
                }
            }
            tokens = new_tokens;
        }

        tokens
    }

    /// Tokenize text into subwords
    fn tokenize_subwords(&self, tokens: Vec<String>) -> Vec<String> {
        let mut subwords = Vec::new();

        for token in tokens {
            // Simple greedy tokenization - can be improved with more sophisticated algorithms
            let mut remaining = token.as_str();
            while !remaining.is_empty() {
                let mut found = false;
                // Try to find the longest matching token
                for len in (1..=remaining.len()).rev() {
                    let candidate = &remaining[..len];
                    if self.token_to_id.contains_key(candidate) {
                        subwords.push(candidate.to_string());
                        remaining = &remaining[len..];
                        found = true;
                        break;
                    }
                }
                if !found {
                    // Use UNK token or skip character
                    if let Some(unk_token) = &self.format.vocabulary.unk_token {
                        subwords.push(unk_token.clone());
                    }
                    remaining = &remaining[1..];
                }
            }
        }

        subwords
    }
}

impl Tokenizer for CustomFormatTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let normalized = self.normalize_text(text);
        let pre_tokens = self.pre_tokenize(&normalized);
        let subwords = self.tokenize_subwords(pre_tokens);

        let mut input_ids = Vec::new();
        for token in &subwords {
            if let Some(id) = self.token_to_id(token) {
                input_ids.push(id);
            } else if let Some(unk_token) = &self.format.vocabulary.unk_token {
                if let Some(unk_id) = self.token_to_id(unk_token) {
                    input_ids.push(unk_id);
                }
            }
        }

        // Apply max length constraint
        if let Some(max_len) = self.max_length {
            input_ids.truncate(max_len);
        }

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

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let tokens: Vec<String> = ids.iter().filter_map(|&id| self.id_to_token(id)).collect();
        Ok(tokens.join(" "))
    }

    fn encode_pair(&self, text_a: &str, text_b: &str) -> Result<TokenizedInput> {
        // Simple concatenation with separator
        let combined = format!("{} {} {}", text_a, "[SEP]", text_b);
        self.encode(&combined)
    }

    fn vocab_size(&self) -> usize {
        self.format.vocabulary.size
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.format
            .vocabulary
            .tokens
            .iter()
            .map(|token| (token.text.clone(), token.id))
            .collect()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.format.vocabulary.tokens.iter().find(|t| t.text == token).map(|t| t.id)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.format
            .vocabulary
            .tokens
            .iter()
            .find(|t| t.id == id)
            .map(|t| t.text.clone())
    }
}

/// Custom format converter for converting between different tokenizer formats
pub struct CustomFormatConverter;

impl CustomFormatConverter {
    /// Convert HuggingFace tokenizer.json to custom format
    pub fn from_huggingface_json(json_str: &str) -> Result<CustomTokenizerFormat> {
        let hf_json: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
            TrustformersError::other(anyhow::anyhow!("Failed to parse HF JSON: {}", e).to_string())
        })?;

        let mut tokens = Vec::new();
        let mut special_tokens = Vec::new();

        // Extract vocabulary
        if let Some(vocab) = hf_json["model"]["vocab"].as_object() {
            for (token_text, token_id) in vocab {
                if let Some(id) = token_id.as_u64() {
                    tokens.push(CustomToken {
                        text: token_text.clone(),
                        id: id as u32,
                        frequency: None,
                        is_special: false,
                        metadata: HashMap::new(),
                    });
                }
            }
        }

        // Extract special tokens
        if let Some(added_tokens) = hf_json["added_tokens"].as_array() {
            for token in added_tokens {
                if let (Some(content), Some(id)) = (token["content"].as_str(), token["id"].as_u64())
                {
                    special_tokens.push(CustomSpecialToken {
                        token: content.to_string(),
                        id: id as u32,
                        token_type: SpecialTokenType::UserDefined("unknown".to_string()),
                        context: None,
                    });
                }
            }
        }

        let tokens_len = tokens.len();
        let vocabulary = CustomVocabulary {
            vocab_type: VocabularyType::SubwordBPE, // Default assumption
            tokens,
            size: tokens_len,
            unk_token: Some("[UNK]".to_string()),
            special_token_mapping: HashMap::new(),
        };

        Ok(CustomTokenizerFormat {
            format_name: "TrustformersCustom".to_string(),
            format_version: "1.0".to_string(),
            vocabulary,
            special_tokens,
            normalization_rules: vec![NormalizationRule {
                rule_type: NormalizationType::NormalizeUnicode,
                pattern: None,
                replacement: None,
                enabled: true,
            }],
            pre_tokenization_rules: vec![PreTokenizationRule {
                rule_type: PreTokenizationType::WhitespaceSplit,
                pattern: None,
                enabled: true,
            }],
            post_processing_rules: vec![PostProcessingRule {
                rule_type: PostProcessingType::AddSpecialTokens,
                parameters: HashMap::new(),
                enabled: true,
            }],
            metadata: HashMap::new(),
        })
    }

    /// Convert custom format to HuggingFace tokenizer.json
    pub fn to_huggingface_json(format: &CustomTokenizerFormat) -> Result<String> {
        let mut hf_json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": {
                "type": "Sequence",
                "normalizers": []
            },
            "pre_tokenizer": {
                "type": "Sequence",
                "pre_tokenizers": []
            },
            "post_processor": null,
            "decoder": {
                "type": "BPEDecoder"
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": format.vocabulary.unk_token,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "vocab": {},
                "merges": []
            }
        });

        // Add vocabulary
        let mut vocab_map = serde_json::Map::new();
        for token in &format.vocabulary.tokens {
            vocab_map.insert(
                token.text.clone(),
                serde_json::Value::Number(token.id.into()),
            );
        }
        hf_json["model"]["vocab"] = serde_json::Value::Object(vocab_map);

        // Add special tokens
        let mut added_tokens = Vec::new();
        for special_token in &format.special_tokens {
            added_tokens.push(serde_json::json!({
                "id": special_token.id,
                "content": special_token.token,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }));
        }
        hf_json["added_tokens"] = serde_json::Value::Array(added_tokens);

        serde_json::to_string_pretty(&hf_json).map_err(|e| {
            TrustformersError::other(
                anyhow::anyhow!("Failed to serialize HF JSON: {}", e).to_string(),
            )
        })
    }

    /// Convert SentencePiece model to custom format
    pub fn from_sentencepiece_model(model_path: &Path) -> Result<CustomTokenizerFormat> {
        use crate::sentencepiece::SentencePieceTokenizer;

        // Load the SentencePiece model
        let sp_tokenizer = SentencePieceTokenizer::from_model_file(model_path)?;

        // Get vocabulary from the tokenizer
        let vocab_size = sp_tokenizer.vocab_size();
        let mut tokens = Vec::new();
        let mut special_tokens = Vec::new();
        let mut special_token_mapping = HashMap::new();

        // Extract tokens and their metadata
        for id in 0..vocab_size {
            let id_u32 = id as u32;
            if let Some(token_text) = sp_tokenizer.id_to_token(id_u32) {
                let score = sp_tokenizer.get_score(id_u32).unwrap_or(0.0);
                let is_special = sp_tokenizer.is_special_token_public(&token_text);

                let custom_token = CustomToken {
                    text: token_text.clone(),
                    id: id_u32,
                    frequency: Some(score as f64),
                    is_special,
                    metadata: HashMap::new(),
                };
                tokens.push(custom_token);

                // Handle special tokens
                if is_special {
                    let token_type = if token_text == "<pad>" {
                        SpecialTokenType::Pad
                    } else if token_text == "<unk>" {
                        SpecialTokenType::Unk
                    } else if token_text == "<s>" {
                        SpecialTokenType::BOS
                    } else if token_text == "</s>" {
                        SpecialTokenType::EOS
                    } else if token_text == "[CLS]" {
                        SpecialTokenType::Cls
                    } else if token_text == "[SEP]" {
                        SpecialTokenType::Sep
                    } else if token_text == "[MASK]" {
                        SpecialTokenType::Mask
                    } else {
                        SpecialTokenType::UserDefined(token_text.clone())
                    };

                    special_tokens.push(CustomSpecialToken {
                        token: token_text.clone(),
                        id: id_u32,
                        token_type,
                        context: None,
                    });
                    special_token_mapping.insert(token_text, id_u32);
                }
            }
        }

        // Create custom vocabulary
        let vocabulary = CustomVocabulary {
            vocab_type: VocabularyType::SentencePiece,
            tokens,
            size: vocab_size,
            unk_token: sp_tokenizer.unk_token().map(|s| s.to_string()),
            special_token_mapping,
        };

        // Create normalization rules based on SentencePiece configuration
        let mut normalization_rules = Vec::new();

        if sp_tokenizer.uses_normalization() {
            normalization_rules.push(NormalizationRule {
                rule_type: NormalizationType::NormalizeUnicode,
                pattern: None,
                replacement: None,
                enabled: true,
            });
        }

        if sp_tokenizer.removes_extra_whitespaces() {
            normalization_rules.push(NormalizationRule {
                rule_type: NormalizationType::NormalizeWhitespace,
                pattern: None,
                replacement: None,
                enabled: true,
            });
        }

        // Create pre-tokenization rules
        let mut pre_tokenization_rules = Vec::new();
        if sp_tokenizer.treats_whitespace_as_suffix() {
            pre_tokenization_rules.push(PreTokenizationRule {
                rule_type: PreTokenizationType::WhitespaceSplit,
                pattern: None,
                enabled: true,
            });
        }

        // Create post-processing rules for special tokens
        let mut post_processing_rules = Vec::new();
        if sp_tokenizer.bos_token_id().is_some() || sp_tokenizer.eos_token_id().is_some() {
            let mut parameters = HashMap::new();
            parameters.insert(
                "template".to_string(),
                serde_json::Value::String("$A".to_string()),
            );
            parameters.insert(
                "tokens".to_string(),
                serde_json::Value::Array(
                    special_tokens
                        .iter()
                        .map(|st| serde_json::Value::String(st.token.clone()))
                        .collect(),
                ),
            );

            post_processing_rules.push(PostProcessingRule {
                rule_type: PostProcessingType::AddSpecialTokens,
                parameters,
                enabled: true,
            });
        }

        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert(
            "source".to_string(),
            serde_json::Value::String("SentencePiece".to_string()),
        );
        metadata.insert(
            "model_type".to_string(),
            serde_json::Value::String(sp_tokenizer.model_type_string()),
        );
        metadata.insert(
            "vocab_size".to_string(),
            serde_json::Value::Number(serde_json::Number::from(vocab_size)),
        );
        metadata.insert(
            "uses_byte_fallback".to_string(),
            serde_json::Value::Bool(sp_tokenizer.uses_byte_fallback()),
        );

        Ok(CustomTokenizerFormat {
            format_name: "SentencePiece".to_string(),
            format_version: "1.0".to_string(),
            vocabulary,
            special_tokens,
            normalization_rules,
            pre_tokenization_rules,
            post_processing_rules,
            metadata,
        })
    }

    /// Validate custom format
    pub fn validate_format(format: &CustomTokenizerFormat) -> Result<Vec<String>> {
        let mut warnings = Vec::new();

        // Check vocabulary consistency
        if format.vocabulary.tokens.len() != format.vocabulary.size {
            warnings.push(format!(
                "Vocabulary size mismatch: declared {} but found {} tokens",
                format.vocabulary.size,
                format.vocabulary.tokens.len()
            ));
        }

        // Check for duplicate token IDs
        let mut seen_ids = std::collections::HashSet::new();
        for token in &format.vocabulary.tokens {
            if !seen_ids.insert(token.id) {
                warnings.push(format!("Duplicate token ID: {}", token.id));
            }
        }

        // Check special tokens
        for special_token in &format.special_tokens {
            if !seen_ids.contains(&special_token.id) {
                warnings.push(format!(
                    "Special token '{}' has ID {} not found in vocabulary",
                    special_token.token, special_token.id
                ));
            }
        }

        Ok(warnings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_format_creation() {
        let format = CustomTokenizerFormat {
            format_name: "TestFormat".to_string(),
            format_version: "1.0".to_string(),
            vocabulary: CustomVocabulary {
                vocab_type: VocabularyType::WordLevel,
                tokens: vec![
                    CustomToken {
                        text: "hello".to_string(),
                        id: 0,
                        frequency: Some(0.1),
                        is_special: false,
                        metadata: HashMap::new(),
                    },
                    CustomToken {
                        text: "world".to_string(),
                        id: 1,
                        frequency: Some(0.05),
                        is_special: false,
                        metadata: HashMap::new(),
                    },
                ],
                size: 2,
                unk_token: Some("[UNK]".to_string()),
                special_token_mapping: HashMap::new(),
            },
            special_tokens: vec![CustomSpecialToken {
                token: "[UNK]".to_string(),
                id: 2,
                token_type: SpecialTokenType::Unk,
                context: None,
            }],
            normalization_rules: vec![],
            pre_tokenization_rules: vec![],
            post_processing_rules: vec![],
            metadata: HashMap::new(),
        };

        let tokenizer = CustomFormatTokenizer::from_format(format).unwrap();
        assert_eq!(tokenizer.vocab_size(), 2);
        assert_eq!(tokenizer.token_to_id("hello"), Some(0));
        assert_eq!(tokenizer.id_to_token(1), Some("world".to_string()));
    }

    #[test]
    fn test_custom_tokenizer_encode() {
        let format = CustomTokenizerFormat {
            format_name: "TestFormat".to_string(),
            format_version: "1.0".to_string(),
            vocabulary: CustomVocabulary {
                vocab_type: VocabularyType::WordLevel,
                tokens: vec![
                    CustomToken {
                        text: "hello".to_string(),
                        id: 0,
                        frequency: None,
                        is_special: false,
                        metadata: HashMap::new(),
                    },
                    CustomToken {
                        text: "world".to_string(),
                        id: 1,
                        frequency: None,
                        is_special: false,
                        metadata: HashMap::new(),
                    },
                ],
                size: 2,
                unk_token: Some("[UNK]".to_string()),
                special_token_mapping: HashMap::new(),
            },
            special_tokens: vec![],
            normalization_rules: vec![],
            pre_tokenization_rules: vec![PreTokenizationRule {
                rule_type: PreTokenizationType::WhitespaceSplit,
                pattern: None,
                enabled: true,
            }],
            post_processing_rules: vec![],
            metadata: HashMap::new(),
        };

        let tokenizer = CustomFormatTokenizer::from_format(format).unwrap();
        let result = tokenizer.encode("hello world").unwrap();
        assert_eq!(result.input_ids, vec![0, 1]);
        assert_eq!(result.attention_mask, vec![1, 1]);
    }

    #[test]
    fn test_format_validation() {
        let format = CustomTokenizerFormat {
            format_name: "TestFormat".to_string(),
            format_version: "1.0".to_string(),
            vocabulary: CustomVocabulary {
                vocab_type: VocabularyType::WordLevel,
                tokens: vec![CustomToken {
                    text: "hello".to_string(),
                    id: 0,
                    frequency: None,
                    is_special: false,
                    metadata: HashMap::new(),
                }],
                size: 2, // Mismatch: claims 2 but only has 1 token
                unk_token: None,
                special_token_mapping: HashMap::new(),
            },
            special_tokens: vec![],
            normalization_rules: vec![],
            pre_tokenization_rules: vec![],
            post_processing_rules: vec![],
            metadata: HashMap::new(),
        };

        let warnings = CustomFormatConverter::validate_format(&format).unwrap();
        assert!(!warnings.is_empty());
        assert!(warnings[0].contains("Vocabulary size mismatch"));
    }
}
