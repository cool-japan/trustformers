use chrono;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// MessagePack-compatible tokenizer metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePackTokenizerMetadata {
    pub name: String,
    pub version: String,
    pub tokenizer_type: String,
    pub vocab_size: u32,
    pub special_tokens: HashMap<String, u32>,
    pub max_length: Option<u32>,
    pub truncation_side: String,
    pub padding_side: String,
    pub do_lower_case: bool,
    pub strip_accents: Option<bool>,
    pub add_prefix_space: bool,
    pub trim_offsets: bool,
    pub created_at: String,
    pub model_id: Option<String>,
    pub custom_attributes: HashMap<String, Vec<u8>>, // For extension data
}

/// MessagePack-compatible vocabulary entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePackVocabEntry {
    pub token: String,
    pub id: u32,
    pub frequency: f64,
    pub is_special: bool,
    pub token_type: u32, // Enumerated token type
}

/// MessagePack-compatible normalization rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePackNormalizationRule {
    pub rule_type: u32, // Enumerated rule type
    pub pattern: Option<String>,
    pub replacement: Option<String>,
    pub enabled: bool,
    pub priority: u32,
}

/// MessagePack-compatible merge rule (for BPE)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePackMergeRule {
    pub first_token: String,
    pub second_token: String,
    pub merged_token: String,
    pub priority: u32,
    pub frequency: f64,
}

/// MessagePack-compatible tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePackTokenizerConfig {
    pub metadata: MessagePackTokenizerMetadata,
    pub vocabulary: Vec<MessagePackVocabEntry>,
    pub normalization_rules: Vec<MessagePackNormalizationRule>,
    pub merge_rules: Vec<MessagePackMergeRule>,
    pub preprocessing_config: HashMap<String, Vec<u8>>,
    pub postprocessing_config: HashMap<String, Vec<u8>>,
    pub training_config: Option<HashMap<String, Vec<u8>>>,
}

/// MessagePack-compatible tokenized input representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagePackTokenizedInput {
    pub input_ids: Vec<u32>,
    pub attention_mask: Option<Vec<u32>>,
    pub token_type_ids: Option<Vec<u32>>,
    pub special_tokens_mask: Option<Vec<u32>>,
    pub offsets: Option<Vec<(u32, u32)>>,
    pub tokens: Vec<String>,
    pub overflow: bool,
    pub sequence_length: u32,
    pub metadata: HashMap<String, Vec<u8>>,
}

/// Configuration options for MessagePack serialization
#[derive(Debug, Clone)]
pub struct MessagePackConfig {
    /// Whether to use binary or named format
    pub use_binary_format: bool,

    /// Whether to include metadata in the serialized data
    pub include_metadata: bool,

    /// Whether to include vocabulary in the serialized data
    pub include_vocabulary: bool,

    /// Whether to include training configuration
    pub include_training_config: bool,

    /// Whether to compress the output (using built-in MessagePack compression)
    pub compress: bool,

    /// Custom attributes to include
    pub custom_attributes: HashMap<String, Vec<u8>>,
}

impl Default for MessagePackConfig {
    fn default() -> Self {
        Self {
            use_binary_format: true,
            include_metadata: true,
            include_vocabulary: true,
            include_training_config: false,
            compress: false,
            custom_attributes: HashMap::new(),
        }
    }
}

/// MessagePack serializer for tokenizers and tokenized inputs
pub struct MessagePackSerializer {
    config: MessagePackConfig,
}

impl MessagePackSerializer {
    /// Create a new MessagePack serializer with the given configuration
    pub fn new(config: MessagePackConfig) -> Self {
        Self { config }
    }

    /// Create a new MessagePack serializer with default configuration
    pub fn default() -> Self {
        Self {
            config: MessagePackConfig::default(),
        }
    }

    /// Serialize a tokenizer to MessagePack format
    pub fn serialize_tokenizer<T: Tokenizer>(
        &self,
        tokenizer: &T,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<Vec<u8>> {
        let vocab = tokenizer.get_vocab();
        let special_tokens = self.detect_special_tokens(&vocab);
        let vocab_entries: Vec<MessagePackVocabEntry> = vocab
            .iter()
            .map(|(token, &id)| MessagePackVocabEntry {
                token: token.clone(),
                id,
                frequency: 1.0, // Default frequency
                is_special: special_tokens.contains(token),
                token_type: if special_tokens.contains(token) { 1 } else { 0 },
            })
            .collect();

        let tokenizer_metadata = MessagePackTokenizerMetadata {
            name: metadata
                .as_ref()
                .and_then(|m| m.get("name"))
                .unwrap_or(&"unknown".to_string())
                .clone(),
            version: metadata
                .as_ref()
                .and_then(|m| m.get("version"))
                .unwrap_or(&"1.0.0".to_string())
                .clone(),
            tokenizer_type: self.get_tokenizer_type(&metadata),
            vocab_size: vocab.len() as u32,
            special_tokens: special_tokens
                .iter()
                .enumerate()
                .map(|(i, token)| (token.clone(), i as u32))
                .collect(),
            max_length: metadata
                .as_ref()
                .and_then(|m| m.get("max_length"))
                .and_then(|v| v.parse().ok()),
            truncation_side: "right".to_string(),
            padding_side: "right".to_string(),
            do_lower_case: false,
            strip_accents: None,
            add_prefix_space: false,
            trim_offsets: true,
            created_at: chrono::Utc::now().to_rfc3339(),
            model_id: metadata.as_ref().and_then(|m| m.get("model_id")).cloned(),
            custom_attributes: self.config.custom_attributes.clone(),
        };

        let config = MessagePackTokenizerConfig {
            metadata: tokenizer_metadata,
            vocabulary: if self.config.include_vocabulary { vocab_entries } else { Vec::new() },
            normalization_rules: self.extract_normalization_rules(&metadata),
            merge_rules: self.extract_merge_rules(&metadata),
            preprocessing_config: HashMap::new(),
            postprocessing_config: HashMap::new(),
            training_config: if self.config.include_training_config {
                Some(HashMap::new())
            } else {
                None
            },
        };

        self.serialize_to_messagepack(&config)
    }

    /// Serialize a tokenized input to MessagePack format
    pub fn serialize_tokenized_input(&self, input: &TokenizedInput) -> Result<Vec<u8>> {
        let msgpack_input = MessagePackTokenizedInput {
            input_ids: input.input_ids.clone(),
            attention_mask: Some(input.attention_mask.iter().map(|&x| x as u32).collect()),
            token_type_ids: input.token_type_ids.clone(),
            special_tokens_mask: None,
            offsets: None,
            tokens: Vec::new(),
            overflow: false,
            sequence_length: input.input_ids.len() as u32,
            metadata: HashMap::new(),
        };

        self.serialize_to_messagepack(&msgpack_input)
    }

    /// Serialize a TokenizedInput batch to MessagePack format
    pub fn serialize_tokenized_batch(&self, batch: &[TokenizedInput]) -> Result<Vec<u8>> {
        let msgpack_batch: Vec<MessagePackTokenizedInput> = batch
            .iter()
            .map(|input| MessagePackTokenizedInput {
                input_ids: input.input_ids.clone(),
                attention_mask: Some(input.attention_mask.iter().map(|&x| x as u32).collect()),
                token_type_ids: input.token_type_ids.clone(),
                special_tokens_mask: None,
                offsets: None,
                tokens: Vec::new(),
                overflow: false,
                sequence_length: input.input_ids.len() as u32,
                metadata: HashMap::new(),
            })
            .collect();

        self.serialize_to_messagepack(&msgpack_batch)
    }

    /// Deserialize a tokenizer configuration from MessagePack format
    pub fn deserialize_tokenizer_config(&self, data: &[u8]) -> Result<MessagePackTokenizerConfig> {
        self.deserialize_from_messagepack(data)
    }

    /// Deserialize a tokenized input from MessagePack format
    pub fn deserialize_tokenized_input(&self, data: &[u8]) -> Result<TokenizedInput> {
        let msgpack_input: MessagePackTokenizedInput = self.deserialize_from_messagepack(data)?;

        let input_ids_len = msgpack_input.input_ids.len();
        Ok(TokenizedInput {
            input_ids: msgpack_input.input_ids,
            attention_mask: msgpack_input
                .attention_mask
                .unwrap_or_else(|| vec![1; input_ids_len])
                .into_iter()
                .map(|x| x as u8)
                .collect(),
            token_type_ids: msgpack_input.token_type_ids,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        })
    }

    /// Deserialize a batch of tokenized inputs from MessagePack format
    pub fn deserialize_tokenized_batch(&self, data: &[u8]) -> Result<Vec<TokenizedInput>> {
        let msgpack_batch: Vec<MessagePackTokenizedInput> =
            self.deserialize_from_messagepack(data)?;

        Ok(msgpack_batch
            .into_iter()
            .map(|msgpack_input| {
                let input_ids_len = msgpack_input.input_ids.len();
                TokenizedInput {
                    input_ids: msgpack_input.input_ids,
                    attention_mask: msgpack_input
                        .attention_mask
                        .unwrap_or_else(|| vec![1; input_ids_len])
                        .into_iter()
                        .map(|x| x as u8)
                        .collect(),
                    token_type_ids: msgpack_input.token_type_ids,
                    special_tokens_mask: None,
                    offset_mapping: None,
                    overflowing_tokens: None,
                }
            })
            .collect())
    }

    /// Save a tokenizer to a MessagePack file
    pub fn save_tokenizer_to_file<T: Tokenizer, P: AsRef<Path>>(
        &self,
        tokenizer: &T,
        path: P,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<()> {
        let data = self.serialize_tokenizer(tokenizer, metadata)?;
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(&data)?;
        file.flush()?;
        Ok(())
    }

    /// Save a tokenized input to a MessagePack file
    pub fn save_tokenized_input_to_file<P: AsRef<Path>>(
        &self,
        input: &TokenizedInput,
        path: P,
    ) -> Result<()> {
        let data = self.serialize_tokenized_input(input)?;
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(&data)?;
        file.flush()?;
        Ok(())
    }

    /// Load a tokenizer configuration from a MessagePack file
    pub fn load_tokenizer_config_from_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<MessagePackTokenizerConfig> {
        let mut file = BufReader::new(File::open(path)?);
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        self.deserialize_tokenizer_config(&data)
    }

    /// Load a tokenized input from a MessagePack file
    pub fn load_tokenized_input_from_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<TokenizedInput> {
        let mut file = BufReader::new(File::open(path)?);
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        self.deserialize_tokenized_input(&data)
    }

    /// Validate MessagePack data structure
    pub fn validate_messagepack_data(&self, data: &[u8]) -> Result<bool> {
        // Try to deserialize to validate structure
        match rmp_serde::from_slice::<serde_json::Value>(data) {
            Ok(_) => Ok(true),
            Err(e) => Err(TrustformersError::serialization_error(format!(
                "Invalid MessagePack data: {}",
                e
            ))),
        }
    }

    /// Get information about MessagePack data
    pub fn get_messagepack_info(&self, data: &[u8]) -> Result<HashMap<String, String>> {
        let mut info = HashMap::new();

        info.insert("format".to_string(), "MessagePack".to_string());
        info.insert("size_bytes".to_string(), data.len().to_string());

        // Try to parse as tokenizer config first
        if let Ok(config) = self.deserialize_tokenizer_config(data) {
            info.insert("data_type".to_string(), "tokenizer_config".to_string());
            info.insert("tokenizer_type".to_string(), config.metadata.tokenizer_type);
            info.insert(
                "vocab_size".to_string(),
                config.metadata.vocab_size.to_string(),
            );
            info.insert("version".to_string(), config.metadata.version);
        } else if let Ok(_input) = self.deserialize_tokenized_input(data) {
            info.insert("data_type".to_string(), "tokenized_input".to_string());
        } else if let Ok(batch) = self.deserialize_tokenized_batch(data) {
            info.insert("data_type".to_string(), "tokenized_batch".to_string());
            info.insert("batch_size".to_string(), batch.len().to_string());
        } else {
            info.insert("data_type".to_string(), "unknown".to_string());
        }

        Ok(info)
    }

    /// Compare two MessagePack files
    pub fn compare_messagepack_files<P1: AsRef<Path>, P2: AsRef<Path>>(
        &self,
        path1: P1,
        path2: P2,
    ) -> Result<HashMap<String, String>> {
        let mut file1 = BufReader::new(File::open(path1)?);
        let mut file2 = BufReader::new(File::open(path2)?);

        let mut data1 = Vec::new();
        let mut data2 = Vec::new();

        file1.read_to_end(&mut data1)?;
        file2.read_to_end(&mut data2)?;

        let mut comparison = HashMap::new();

        comparison.insert("size1_bytes".to_string(), data1.len().to_string());
        comparison.insert("size2_bytes".to_string(), data2.len().to_string());
        comparison.insert(
            "sizes_equal".to_string(),
            (data1.len() == data2.len()).to_string(),
        );
        comparison.insert("contents_equal".to_string(), (data1 == data2).to_string());

        let info1 = self.get_messagepack_info(&data1)?;
        let info2 = self.get_messagepack_info(&data2)?;

        comparison.insert(
            "type1".to_string(),
            info1.get("data_type").unwrap_or(&"unknown".to_string()).clone(),
        );
        comparison.insert(
            "type2".to_string(),
            info2.get("data_type").unwrap_or(&"unknown".to_string()).clone(),
        );

        Ok(comparison)
    }

    /// Generic serialization method
    fn serialize_to_messagepack<T: Serialize>(&self, data: &T) -> Result<Vec<u8>> {
        rmp_serde::to_vec(data).map_err(|e| {
            TrustformersError::serialization_error(format!(
                "MessagePack serialization failed: {}",
                e
            ))
        })
    }

    /// Generic deserialization method
    fn deserialize_from_messagepack<T: for<'de> Deserialize<'de>>(&self, data: &[u8]) -> Result<T> {
        rmp_serde::from_slice(data).map_err(|e| {
            TrustformersError::serialization_error(format!(
                "MessagePack deserialization failed: {}",
                e
            ))
        })
    }

    /// Detect common special tokens in vocabulary
    fn detect_special_tokens(&self, vocab: &HashMap<String, u32>) -> HashSet<String> {
        let common_special_tokens = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "<|endoftext|>",
            "<|startoftext|>",
            "<|padding|>",
            "<pad>",
            "<unk>",
            "<cls>",
            "<sep>",
            "<mask>",
            "<s>",
            "</s>",
            "<eos>",
            "<bos>",
        ];

        vocab
            .keys()
            .filter(|token| {
                common_special_tokens.contains(&token.as_str())
                    || token.starts_with('<') && token.ends_with('>')
                    || token.starts_with('[') && token.ends_with(']')
            })
            .cloned()
            .collect()
    }

    /// Get tokenizer type from metadata
    fn get_tokenizer_type(&self, metadata: &Option<HashMap<String, String>>) -> String {
        metadata
            .as_ref()
            .and_then(|m| m.get("tokenizer_type"))
            .cloned()
            .unwrap_or_else(|| "generic".to_string())
    }

    /// Extract normalization rules from metadata
    fn extract_normalization_rules(
        &self,
        metadata: &Option<HashMap<String, String>>,
    ) -> Vec<MessagePackNormalizationRule> {
        let mut rules = Vec::new();

        if let Some(meta) = metadata {
            if meta.get("normalize_case").map(|v| v == "true").unwrap_or(false) {
                rules.push(MessagePackNormalizationRule {
                    rule_type: 1, // Lowercase
                    pattern: None,
                    replacement: None,
                    enabled: true,
                    priority: 1,
                });
            }
            if meta.get("strip_accents").map(|v| v == "true").unwrap_or(false) {
                rules.push(MessagePackNormalizationRule {
                    rule_type: 2, // Strip accents
                    pattern: None,
                    replacement: None,
                    enabled: true,
                    priority: 2,
                });
            }
        }

        rules
    }

    /// Extract merge rules from metadata (for BPE tokenizers)
    fn extract_merge_rules(
        &self,
        metadata: &Option<HashMap<String, String>>,
    ) -> Vec<MessagePackMergeRule> {
        let mut rules = Vec::new();

        if let Some(meta) = metadata {
            if let Some(merge_data) = meta.get("merge_rules") {
                // Parse merge rules from metadata (simplified implementation)
                for (i, line) in merge_data.lines().enumerate() {
                    let parts: Vec<&str> = line.split(' ').collect();
                    if parts.len() >= 2 {
                        rules.push(MessagePackMergeRule {
                            first_token: parts[0].to_string(),
                            second_token: parts[1].to_string(),
                            merged_token: format!("{}{}", parts[0], parts[1]),
                            priority: i as u32,
                            frequency: 1.0,
                        });
                    }
                }
            }
        }

        rules
    }
}

/// Utility functions for MessagePack operations
pub struct MessagePackUtils;

impl MessagePackUtils {
    /// Convert MessagePack to JSON for inspection
    pub fn messagepack_to_json(data: &[u8]) -> Result<String> {
        let value: serde_json::Value = rmp_serde::from_slice(data).map_err(|e| {
            TrustformersError::serialization_error(format!(
                "MessagePack to JSON conversion failed: {}",
                e
            ))
        })?;

        serde_json::to_string_pretty(&value).map_err(|e| {
            TrustformersError::serialization_error(format!("JSON serialization failed: {}", e))
        })
    }

    /// Convert JSON to MessagePack
    pub fn json_to_messagepack(json: &str) -> Result<Vec<u8>> {
        let value: serde_json::Value = serde_json::from_str(json).map_err(|e| {
            TrustformersError::serialization_error(format!("JSON parsing failed: {}", e))
        })?;

        rmp_serde::to_vec(&value).map_err(|e| {
            TrustformersError::serialization_error(format!(
                "JSON to MessagePack conversion failed: {}",
                e
            ))
        })
    }

    /// Get MessagePack data statistics
    pub fn get_statistics(data: &[u8]) -> Result<HashMap<String, String>> {
        let mut stats = HashMap::new();

        stats.insert("format".to_string(), "MessagePack".to_string());
        stats.insert("size_bytes".to_string(), data.len().to_string());

        // Try to parse and count elements
        if let Ok(value) = rmp_serde::from_slice::<serde_json::Value>(data) {
            match &value {
                serde_json::Value::Object(map) => {
                    stats.insert("type".to_string(), "object".to_string());
                    stats.insert("fields_count".to_string(), map.len().to_string());
                },
                serde_json::Value::Array(arr) => {
                    stats.insert("type".to_string(), "array".to_string());
                    stats.insert("elements_count".to_string(), arr.len().to_string());
                },
                _ => {
                    stats.insert("type".to_string(), "primitive".to_string());
                },
            }
        }

        Ok(stats)
    }

    /// Validate MessagePack file integrity
    pub fn validate_file<P: AsRef<Path>>(path: P) -> Result<bool> {
        let mut file = BufReader::new(File::open(path)?);
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        match rmp_serde::from_slice::<serde_json::Value>(&data) {
            Ok(_) => Ok(true),
            Err(e) => Err(TrustformersError::serialization_error(format!(
                "MessagePack file validation failed: {}",
                e
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::tempdir;

    #[test]
    fn test_messagepack_config_default() {
        let config = MessagePackConfig::default();
        assert!(config.use_binary_format);
        assert!(config.include_metadata);
        assert!(config.include_vocabulary);
        assert!(!config.include_training_config);
        assert!(!config.compress);
    }

    #[test]
    fn test_messagepack_serializer_creation() {
        let config = MessagePackConfig::default();
        let _serializer = MessagePackSerializer::new(config);

        // Test that default constructor works
        let default_serializer = MessagePackSerializer::default();
        assert!(default_serializer.config.use_binary_format);
    }

    #[test]
    fn test_serialize_tokenized_input() {
        let serializer = MessagePackSerializer::default();

        let input = TokenizedInput {
            input_ids: vec![1, 2, 3, 4],
            attention_mask: vec![1, 1, 1, 1],
            token_type_ids: Some(vec![0, 0, 1, 1]),
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let serialized = serializer.serialize_tokenized_input(&input).unwrap();
        assert!(!serialized.is_empty());

        let deserialized = serializer.deserialize_tokenized_input(&serialized).unwrap();
        assert_eq!(input.input_ids, deserialized.input_ids);
        assert_eq!(input.attention_mask, deserialized.attention_mask);
        assert_eq!(input.token_type_ids, deserialized.token_type_ids);
    }

    #[test]
    fn test_serialize_tokenized_batch() {
        let serializer = MessagePackSerializer::default();

        let batch = vec![
            TokenizedInput {
                input_ids: vec![1, 2, 3],
                attention_mask: vec![1, 1, 1],
                token_type_ids: None,
                special_tokens_mask: None,
                offset_mapping: None,
                overflowing_tokens: None,
            },
            TokenizedInput {
                input_ids: vec![4, 5, 6, 7],
                attention_mask: vec![1, 1, 1, 1],
                token_type_ids: None,
                special_tokens_mask: None,
                offset_mapping: None,
                overflowing_tokens: None,
            },
        ];

        let serialized = serializer.serialize_tokenized_batch(&batch).unwrap();
        assert!(!serialized.is_empty());

        let deserialized = serializer.deserialize_tokenized_batch(&serialized).unwrap();
        assert_eq!(batch.len(), deserialized.len());
        assert_eq!(batch[0].input_ids, deserialized[0].input_ids);
        assert_eq!(batch[1].input_ids, deserialized[1].input_ids);
    }

    #[test]
    fn test_messagepack_validation() {
        let serializer = MessagePackSerializer::default();

        let input = TokenizedInput {
            input_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let serialized = serializer.serialize_tokenized_input(&input).unwrap();

        // Valid data should validate successfully
        assert!(serializer.validate_messagepack_data(&serialized).unwrap());

        // Invalid data should fail validation
        // Try with truncated MessagePack data (incomplete)
        let invalid_data = vec![0x82]; // Map with 2 elements but no actual data
        assert!(serializer.validate_messagepack_data(&invalid_data).is_err());
    }

    #[test]
    fn test_messagepack_info() {
        let serializer = MessagePackSerializer::default();

        let input = TokenizedInput {
            input_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let serialized = serializer.serialize_tokenized_input(&input).unwrap();
        let info = serializer.get_messagepack_info(&serialized).unwrap();

        assert_eq!(info.get("format").unwrap(), "MessagePack");
        assert_eq!(info.get("data_type").unwrap(), "tokenized_input");
        assert_eq!(
            info.get("size_bytes").unwrap(),
            &serialized.len().to_string()
        );
    }

    #[test]
    fn test_file_operations() {
        let serializer = MessagePackSerializer::default();
        let temp_dir = tempdir().unwrap();

        let input = TokenizedInput {
            input_ids: vec![1, 2, 3, 4],
            attention_mask: vec![1, 1, 1, 1],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let file_path = temp_dir.path().join("test_input.msgpack");

        // Save to file
        serializer.save_tokenized_input_to_file(&input, &file_path).unwrap();
        assert!(file_path.exists());

        // Load from file
        let loaded_input = serializer.load_tokenized_input_from_file(&file_path).unwrap();
        assert_eq!(input.input_ids, loaded_input.input_ids);
        assert_eq!(input.attention_mask, loaded_input.attention_mask);
        assert_eq!(input.token_type_ids, loaded_input.token_type_ids);
    }

    #[test]
    fn test_messagepack_utils() {
        let test_data = r#"{"test": "data", "number": 42}"#;

        // Convert JSON to MessagePack
        let msgpack_data = MessagePackUtils::json_to_messagepack(test_data).unwrap();
        assert!(!msgpack_data.is_empty());

        // Convert MessagePack back to JSON
        let json_data = MessagePackUtils::messagepack_to_json(&msgpack_data).unwrap();
        assert!(json_data.contains("test"));
        assert!(json_data.contains("42"));

        // Get statistics
        let stats = MessagePackUtils::get_statistics(&msgpack_data).unwrap();
        assert_eq!(stats.get("format").unwrap(), "MessagePack");
        assert_eq!(stats.get("type").unwrap(), "object");
    }

    #[test]
    fn test_file_validation() {
        let serializer = MessagePackSerializer::default();
        let temp_dir = tempdir().unwrap();

        let input = TokenizedInput {
            input_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let file_path = temp_dir.path().join("validation_test.msgpack");
        serializer.save_tokenized_input_to_file(&input, &file_path).unwrap();

        // Valid file should validate successfully
        assert!(MessagePackUtils::validate_file(&file_path).unwrap());
    }

    #[test]
    fn test_file_comparison() {
        let serializer = MessagePackSerializer::default();
        let temp_dir = tempdir().unwrap();

        let input1 = TokenizedInput {
            input_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let input2 = TokenizedInput {
            input_ids: vec![4, 5, 6],
            attention_mask: vec![1, 1, 1],
            token_type_ids: None,
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let file1_path = temp_dir.path().join("compare1.msgpack");
        let file2_path = temp_dir.path().join("compare2.msgpack");

        serializer.save_tokenized_input_to_file(&input1, &file1_path).unwrap();
        serializer.save_tokenized_input_to_file(&input2, &file2_path).unwrap();

        let comparison = serializer.compare_messagepack_files(&file1_path, &file2_path).unwrap();

        assert_eq!(comparison.get("contents_equal").unwrap(), "false");
        assert_eq!(comparison.get("type1").unwrap(), "tokenized_input");
        assert_eq!(comparison.get("type2").unwrap(), "tokenized_input");
    }
}
