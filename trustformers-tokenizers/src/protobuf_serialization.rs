use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Protobuf-compatible tokenizer metadata
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct ProtobufTokenizerMetadata {
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

/// Protobuf-compatible vocabulary entry
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct ProtobufVocabEntry {
    pub token: String,
    pub id: u32,
    pub frequency: f64,
    pub is_special: bool,
    pub token_type: u32, // Enumerated token type
}

/// Protobuf-compatible normalization rule
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct ProtobufNormalizationRule {
    pub rule_type: u32, // Enumerated rule type
    pub pattern: Option<String>,
    pub replacement: Option<String>,
    pub enabled: bool,
    pub priority: u32,
}

/// Protobuf-compatible merge rule (for BPE)
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct ProtobufMergeRule {
    pub first_token: String,
    pub second_token: String,
    pub merged_token: String,
    pub priority: u32,
}

/// Complete protobuf tokenizer model
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct ProtobufTokenizerModel {
    pub metadata: ProtobufTokenizerMetadata,
    pub vocabulary: Vec<ProtobufVocabEntry>,
    pub normalization_rules: Vec<ProtobufNormalizationRule>,
    pub merge_rules: Vec<ProtobufMergeRule>,
    pub added_tokens: Vec<ProtobufVocabEntry>,
}

/// Protobuf-compatible tokenized input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtobufTokenizedInput {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub special_tokens_mask: Vec<u32>,
    pub offset_mapping: Vec<ProtobufOffset>,
    pub overflowing_tokens: Vec<ProtobufTokenizedInput>,
    pub num_truncated_tokens: u32,
}

/// Offset information for protobuf
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtobufOffset {
    pub start: u32,
    pub end: u32,
}

/// Batch tokenized input for protobuf
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtobufBatchTokenizedInput {
    pub batch: Vec<ProtobufTokenizedInput>,
    pub batch_size: u32,
    pub max_length: u32,
    pub padding_strategy: u32, // Enumerated padding strategy
}

/// Protobuf serialization utilities
pub struct ProtobufSerializer;

impl ProtobufSerializer {
    /// Convert tokenizer to protobuf model
    pub fn serialize_tokenizer<T: Tokenizer>(
        tokenizer: &T,
        metadata: ProtobufTokenizerMetadata,
    ) -> Result<ProtobufTokenizerModel> {
        // Extract vocabulary from tokenizer
        let vocab_map = tokenizer.get_vocab();
        let mut vocabulary = Vec::new();

        // Convert vocabulary to protobuf format
        for (token, id) in vocab_map.iter() {
            vocabulary.push(ProtobufVocabEntry {
                token: token.clone(),
                id: *id,
                frequency: 0.0, // Default frequency, could be extracted if available
                is_special: Self::is_special_token(token),
                token_type: 0, // Default token type (Normal)
            });
        }

        // Sort vocabulary by ID for consistency
        vocabulary.sort_by_key(|token| token.id);

        // Extract normalization rules (basic implementation)
        let normalization_rules = vec![
            ProtobufNormalizationRule {
                rule_type: 1, // NFC normalize
                enabled: true,
                pattern: None,
                replacement: None,
                priority: 1,
            },
            ProtobufNormalizationRule {
                rule_type: 2,   // Lowercase
                enabled: false, // Default to false, could be configured
                pattern: None,
                replacement: None,
                priority: 2,
            },
        ];

        // For merge rules, we'd need tokenizer-specific logic
        // This is a basic implementation that works for most tokenizers
        let merge_rules = vec![];

        // Identify special tokens from the vocabulary
        let mut added_tokens = Vec::new();
        for (token, id) in vocab_map.iter() {
            if Self::is_special_token(token) {
                added_tokens.push(ProtobufVocabEntry {
                    token: token.clone(),
                    id: *id,
                    frequency: 1.0, // Special tokens usually have high frequency
                    is_special: true,
                    token_type: 1, // Special token type
                });
            }
        }

        Ok(ProtobufTokenizerModel {
            metadata,
            vocabulary,
            normalization_rules,
            merge_rules,
            added_tokens,
        })
    }

    /// Helper function to identify special tokens
    fn is_special_token(token: &str) -> bool {
        // Common special token patterns
        token.starts_with('<') && token.ends_with('>')
            || token.starts_with('[') && token.ends_with(']')
            || matches!(
                token,
                "<pad>"
                    | "<unk>"
                    | "<s>"
                    | "</s>"
                    | "<cls>"
                    | "<sep>"
                    | "<mask>"
                    | "[PAD]"
                    | "[UNK]"
                    | "[CLS]"
                    | "[SEP]"
                    | "[MASK]"
                    | "[BOS]"
                    | "[EOS]"
            )
    }

    /// Convert tokenized input to protobuf format
    pub fn serialize_tokenized_input(input: &TokenizedInput) -> ProtobufTokenizedInput {
        ProtobufTokenizedInput {
            input_ids: input.input_ids.clone(),
            attention_mask: input.attention_mask.iter().map(|&x| x as u32).collect(),
            token_type_ids: input.token_type_ids.clone().unwrap_or_default(),
            special_tokens_mask: vec![], // Would need to be computed
            offset_mapping: vec![],      // Would need offset information
            overflowing_tokens: vec![],
            num_truncated_tokens: 0,
        }
    }

    /// Convert protobuf tokenized input back to standard format
    pub fn deserialize_tokenized_input(protobuf_input: &ProtobufTokenizedInput) -> TokenizedInput {
        TokenizedInput {
            input_ids: protobuf_input.input_ids.clone(),
            attention_mask: protobuf_input.attention_mask.iter().map(|&x| x as u8).collect(),
            token_type_ids: if protobuf_input.token_type_ids.is_empty() {
                None
            } else {
                Some(protobuf_input.token_type_ids.clone())
            },
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        }
    }

    /// Serialize to protobuf binary format
    pub fn to_protobuf_bytes(model: &ProtobufTokenizerModel) -> Result<Vec<u8>> {
        // Using serde with bincode as a simplified protobuf-like format
        // In a real implementation, you'd use actual protobuf libraries like prost
        bincode::encode_to_vec(model, bincode::config::standard()).map_err(|e| {
            TrustformersError::other(
                anyhow::anyhow!("Failed to serialize protobuf: {}", e).to_string(),
            )
        })
    }

    /// Deserialize from protobuf binary format
    pub fn from_protobuf_bytes(bytes: &[u8]) -> Result<ProtobufTokenizerModel> {
        let (result, _): (ProtobufTokenizerModel, usize) =
            bincode::decode_from_slice(bytes, bincode::config::standard()).map_err(|e| {
                TrustformersError::other(
                    anyhow::anyhow!("Failed to deserialize protobuf: {}", e).to_string(),
                )
            })?;
        Ok(result)
    }

    /// Save tokenizer model to protobuf file
    pub fn save_to_file<P: AsRef<Path>>(model: &ProtobufTokenizerModel, path: P) -> Result<()> {
        let bytes = Self::to_protobuf_bytes(model)?;
        std::fs::write(path, bytes).map_err(|e| {
            TrustformersError::other(
                anyhow::anyhow!("Failed to write protobuf file: {}", e).to_string(),
            )
        })
    }

    /// Load tokenizer model from protobuf file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<ProtobufTokenizerModel> {
        let bytes = std::fs::read(path).map_err(|e| {
            TrustformersError::other(
                anyhow::anyhow!("Failed to read protobuf file: {}", e).to_string(),
            )
        })?;
        Self::from_protobuf_bytes(&bytes)
    }

    /// Convert to text-based protobuf format (proto text)
    pub fn to_proto_text(model: &ProtobufTokenizerModel) -> Result<String> {
        // Simplified proto text format
        let mut text = String::new();

        text.push_str("# Tokenizer Model (Proto Text Format)\n");
        text.push_str("metadata {\n");
        text.push_str(&format!("  name: \"{}\"\n", model.metadata.name));
        text.push_str(&format!("  version: \"{}\"\n", model.metadata.version));
        text.push_str(&format!(
            "  tokenizer_type: \"{}\"\n",
            model.metadata.tokenizer_type
        ));
        text.push_str(&format!("  vocab_size: {}\n", model.metadata.vocab_size));
        text.push_str(&format!(
            "  do_lower_case: {}\n",
            model.metadata.do_lower_case
        ));
        text.push_str("}\n\n");

        if !model.vocabulary.is_empty() {
            text.push_str("vocabulary {\n");
            for (i, entry) in model.vocabulary.iter().enumerate() {
                if i >= 10 {
                    // Limit output for readability
                    text.push_str(&format!(
                        "  # ... {} more entries\n",
                        model.vocabulary.len() - 10
                    ));
                    break;
                }
                text.push_str("  entry {\n");
                text.push_str(&format!("    token: \"{}\"\n", entry.token));
                text.push_str(&format!("    id: {}\n", entry.id));
                text.push_str(&format!("    frequency: {}\n", entry.frequency));
                text.push_str(&format!("    is_special: {}\n", entry.is_special));
                text.push_str("  }\n");
            }
            text.push_str("}\n\n");
        }

        if !model.merge_rules.is_empty() {
            text.push_str("merge_rules {\n");
            for (i, rule) in model.merge_rules.iter().enumerate() {
                if i >= 5 {
                    // Limit output for readability
                    text.push_str(&format!(
                        "  # ... {} more rules\n",
                        model.merge_rules.len() - 5
                    ));
                    break;
                }
                text.push_str("  rule {\n");
                text.push_str(&format!("    first_token: \"{}\"\n", rule.first_token));
                text.push_str(&format!("    second_token: \"{}\"\n", rule.second_token));
                text.push_str(&format!("    merged_token: \"{}\"\n", rule.merged_token));
                text.push_str(&format!("    priority: {}\n", rule.priority));
                text.push_str("  }\n");
            }
            text.push_str("}\n");
        }

        Ok(text)
    }

    /// Parse from text-based protobuf format
    pub fn from_proto_text(text: &str) -> Result<ProtobufTokenizerModel> {
        // Simplified parser for proto text format
        // In a real implementation, you'd use a proper protobuf text parser

        let mut metadata = ProtobufTokenizerMetadata {
            name: "unknown".to_string(),
            version: "1.0".to_string(),
            tokenizer_type: "unknown".to_string(),
            vocab_size: 0,
            special_tokens: HashMap::new(),
            max_length: None,
            truncation_side: "right".to_string(),
            padding_side: "right".to_string(),
            do_lower_case: false,
            strip_accents: None,
            add_prefix_space: false,
            trim_offsets: true,
            created_at: chrono::Utc::now().to_rfc3339(),
            model_id: None,
            custom_attributes: HashMap::new(),
        };

        // Simple pattern matching for key fields
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with("name:") {
                if let Some(name) = Self::extract_quoted_value(line) {
                    metadata.name = name;
                }
            } else if line.starts_with("version:") {
                if let Some(version) = Self::extract_quoted_value(line) {
                    metadata.version = version;
                }
            } else if line.starts_with("tokenizer_type:") {
                if let Some(tokenizer_type) = Self::extract_quoted_value(line) {
                    metadata.tokenizer_type = tokenizer_type;
                }
            } else if line.starts_with("vocab_size:") {
                if let Some(size_str) = line.split(':').nth(1) {
                    if let Ok(size) = size_str.trim().parse::<u32>() {
                        metadata.vocab_size = size;
                    }
                }
            } else if line.starts_with("do_lower_case:") {
                if let Some(bool_str) = line.split(':').nth(1) {
                    metadata.do_lower_case = bool_str.trim() == "true";
                }
            }
        }

        Ok(ProtobufTokenizerModel {
            metadata,
            vocabulary: vec![],
            normalization_rules: vec![],
            merge_rules: vec![],
            added_tokens: vec![],
        })
    }

    /// Extract quoted value from proto text line
    fn extract_quoted_value(line: &str) -> Option<String> {
        if let Some(start) = line.find('"') {
            if let Some(end) = line.rfind('"') {
                if start < end {
                    return Some(line[start + 1..end].to_string());
                }
            }
        }
        None
    }

    /// Validate protobuf model
    pub fn validate_model(model: &ProtobufTokenizerModel) -> Result<Vec<String>> {
        let mut warnings = Vec::new();

        // Check vocabulary consistency
        if model.vocabulary.len() != model.metadata.vocab_size as usize {
            warnings.push(format!(
                "Vocabulary size mismatch: metadata claims {} but found {} tokens",
                model.metadata.vocab_size,
                model.vocabulary.len()
            ));
        }

        // Check for duplicate token IDs
        let mut seen_ids = std::collections::HashSet::new();
        for entry in &model.vocabulary {
            if !seen_ids.insert(entry.id) {
                warnings.push(format!("Duplicate token ID: {}", entry.id));
            }
        }

        // Check merge rules validity
        for rule in &model.merge_rules {
            if rule.first_token.is_empty() || rule.second_token.is_empty() {
                warnings.push("Empty tokens in merge rule".to_string());
            }
        }

        Ok(warnings)
    }

    /// Get model statistics
    pub fn get_model_stats(model: &ProtobufTokenizerModel) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();

        stats.insert(
            "vocab_size".to_string(),
            serde_json::Value::Number(model.vocabulary.len().into()),
        );

        stats.insert(
            "special_tokens_count".to_string(),
            serde_json::Value::Number(model.metadata.special_tokens.len().into()),
        );

        stats.insert(
            "merge_rules_count".to_string(),
            serde_json::Value::Number(model.merge_rules.len().into()),
        );

        stats.insert(
            "normalization_rules_count".to_string(),
            serde_json::Value::Number(model.normalization_rules.len().into()),
        );

        let special_token_ratio = if model.metadata.vocab_size > 0 {
            model.metadata.special_tokens.len() as f64 / model.metadata.vocab_size as f64
        } else {
            0.0
        };
        stats.insert(
            "special_token_ratio".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(special_token_ratio).unwrap()),
        );

        stats
    }

    /// Compress protobuf data
    pub fn compress_model(model: &ProtobufTokenizerModel) -> Result<Vec<u8>> {
        let serialized = Self::to_protobuf_bytes(model)?;

        use flate2::{write::GzEncoder, Compression};
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&serialized).map_err(|e| {
            TrustformersError::other(anyhow::anyhow!("Failed to compress: {}", e).to_string())
        })?;

        encoder.finish().map_err(|e| {
            TrustformersError::other(
                anyhow::anyhow!("Failed to finish compression: {}", e).to_string(),
            )
        })
    }

    /// Decompress protobuf data
    pub fn decompress_model(compressed_data: &[u8]) -> Result<ProtobufTokenizerModel> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(compressed_data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).map_err(|e| {
            TrustformersError::other(anyhow::anyhow!("Failed to decompress: {}", e).to_string())
        })?;

        Self::from_protobuf_bytes(&decompressed)
    }
}

/// Helper trait for protobuf conversion
pub trait ProtobufConvertible {
    /// Convert to protobuf model
    fn to_protobuf_model(
        &self,
        metadata: ProtobufTokenizerMetadata,
    ) -> Result<ProtobufTokenizerModel>;

    /// Create from protobuf model
    fn from_protobuf_model(model: &ProtobufTokenizerModel) -> Result<Self>
    where
        Self: Sized;
}

/// Configuration for protobuf export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtobufExportConfig {
    pub include_vocabulary: bool,
    pub include_merge_rules: bool,
    pub include_normalization_rules: bool,
    pub compress_output: bool,
    pub validate_output: bool,
    pub export_format: ProtobufFormat,
}

/// Protobuf export formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProtobufFormat {
    Binary,
    TextFormat,
    Json,
    CompressedBinary,
}

impl Default for ProtobufExportConfig {
    fn default() -> Self {
        Self {
            include_vocabulary: true,
            include_merge_rules: true,
            include_normalization_rules: true,
            compress_output: false,
            validate_output: true,
            export_format: ProtobufFormat::Binary,
        }
    }
}

/// Protobuf export utility
pub struct ProtobufExporter {
    config: ProtobufExportConfig,
}

impl ProtobufExporter {
    /// Create new exporter with configuration
    pub fn new(config: ProtobufExportConfig) -> Self {
        Self { config }
    }

    /// Export tokenizer model
    pub fn export_model<P: AsRef<Path>>(
        &self,
        model: &ProtobufTokenizerModel,
        path: P,
    ) -> Result<()> {
        // Validate if requested
        if self.config.validate_output {
            let warnings = ProtobufSerializer::validate_model(model)?;
            if !warnings.is_empty() {
                eprintln!("Validation warnings:");
                for warning in warnings {
                    eprintln!("  - {}", warning);
                }
            }
        }

        match self.config.export_format {
            ProtobufFormat::Binary => {
                if self.config.compress_output {
                    let compressed = ProtobufSerializer::compress_model(model)?;
                    std::fs::write(path, compressed).map_err(|e| {
                        TrustformersError::other(
                            anyhow::anyhow!("Failed to write file: {}", e).to_string(),
                        )
                    })?;
                } else {
                    ProtobufSerializer::save_to_file(model, path)?;
                }
            },
            ProtobufFormat::TextFormat => {
                let text = ProtobufSerializer::to_proto_text(model)?;
                std::fs::write(path, text).map_err(|e| {
                    TrustformersError::other(
                        anyhow::anyhow!("Failed to write text file: {}", e).to_string(),
                    )
                })?;
            },
            ProtobufFormat::Json => {
                let json = serde_json::to_string_pretty(model).map_err(|e| {
                    TrustformersError::other(
                        anyhow::anyhow!("Failed to serialize JSON: {}", e).to_string(),
                    )
                })?;
                std::fs::write(path, json).map_err(|e| {
                    TrustformersError::other(
                        anyhow::anyhow!("Failed to write JSON file: {}", e).to_string(),
                    )
                })?;
            },
            ProtobufFormat::CompressedBinary => {
                let compressed = ProtobufSerializer::compress_model(model)?;
                std::fs::write(path, compressed).map_err(|e| {
                    TrustformersError::other(
                        anyhow::anyhow!("Failed to write compressed file: {}", e).to_string(),
                    )
                })?;
            },
        }

        Ok(())
    }

    /// Import tokenizer model
    pub fn import_model<P: AsRef<Path>>(&self, path: P) -> Result<ProtobufTokenizerModel> {
        match self.config.export_format {
            ProtobufFormat::Binary => ProtobufSerializer::load_from_file(path),
            ProtobufFormat::TextFormat => {
                let text = std::fs::read_to_string(path).map_err(|e| {
                    TrustformersError::other(
                        anyhow::anyhow!("Failed to read text file: {}", e).to_string(),
                    )
                })?;
                ProtobufSerializer::from_proto_text(&text)
            },
            ProtobufFormat::Json => {
                let json = std::fs::read_to_string(path).map_err(|e| {
                    TrustformersError::other(
                        anyhow::anyhow!("Failed to read JSON file: {}", e).to_string(),
                    )
                })?;
                serde_json::from_str(&json).map_err(|e| {
                    TrustformersError::other(
                        anyhow::anyhow!("Failed to parse JSON: {}", e).to_string(),
                    )
                })
            },
            ProtobufFormat::CompressedBinary => {
                let compressed = std::fs::read(path).map_err(|e| {
                    TrustformersError::other(
                        anyhow::anyhow!("Failed to read compressed file: {}", e).to_string(),
                    )
                })?;
                ProtobufSerializer::decompress_model(&compressed)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protobuf_metadata_creation() {
        let metadata = ProtobufTokenizerMetadata {
            name: "test-tokenizer".to_string(),
            version: "1.0".to_string(),
            tokenizer_type: "bpe".to_string(),
            vocab_size: 1000,
            special_tokens: HashMap::new(),
            max_length: Some(512),
            truncation_side: "right".to_string(),
            padding_side: "right".to_string(),
            do_lower_case: false,
            strip_accents: None,
            add_prefix_space: false,
            trim_offsets: true,
            created_at: chrono::Utc::now().to_rfc3339(),
            model_id: None,
            custom_attributes: HashMap::new(),
        };

        assert_eq!(metadata.name, "test-tokenizer");
        assert_eq!(metadata.vocab_size, 1000);
    }

    #[test]
    fn test_tokenized_input_conversion() {
        let input = TokenizedInput {
            input_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            token_type_ids: Some(vec![0, 0, 0]),
            special_tokens_mask: None,
            offset_mapping: None,
            overflowing_tokens: None,
        };

        let protobuf_input = ProtobufSerializer::serialize_tokenized_input(&input);
        let converted_back = ProtobufSerializer::deserialize_tokenized_input(&protobuf_input);

        assert_eq!(input.input_ids, converted_back.input_ids);
        assert_eq!(input.attention_mask, converted_back.attention_mask);
        assert_eq!(input.token_type_ids, converted_back.token_type_ids);
    }

    #[test]
    fn test_protobuf_serialization() {
        let metadata = ProtobufTokenizerMetadata {
            name: "test".to_string(),
            version: "1.0".to_string(),
            tokenizer_type: "test".to_string(),
            vocab_size: 0,
            special_tokens: HashMap::new(),
            max_length: None,
            truncation_side: "right".to_string(),
            padding_side: "right".to_string(),
            do_lower_case: false,
            strip_accents: None,
            add_prefix_space: false,
            trim_offsets: true,
            created_at: chrono::Utc::now().to_rfc3339(),
            model_id: None,
            custom_attributes: HashMap::new(),
        };

        let model = ProtobufTokenizerModel {
            metadata,
            vocabulary: vec![],
            normalization_rules: vec![],
            merge_rules: vec![],
            added_tokens: vec![],
        };

        let bytes = ProtobufSerializer::to_protobuf_bytes(&model).unwrap();
        let recovered = ProtobufSerializer::from_protobuf_bytes(&bytes).unwrap();

        assert_eq!(model.metadata.name, recovered.metadata.name);
        assert_eq!(model.metadata.version, recovered.metadata.version);
    }

    #[test]
    fn test_proto_text_format() {
        let metadata = ProtobufTokenizerMetadata {
            name: "test-tokenizer".to_string(),
            version: "1.0".to_string(),
            tokenizer_type: "bpe".to_string(),
            vocab_size: 100,
            special_tokens: HashMap::new(),
            max_length: None,
            truncation_side: "right".to_string(),
            padding_side: "right".to_string(),
            do_lower_case: true,
            strip_accents: None,
            add_prefix_space: false,
            trim_offsets: true,
            created_at: chrono::Utc::now().to_rfc3339(),
            model_id: None,
            custom_attributes: HashMap::new(),
        };

        let model = ProtobufTokenizerModel {
            metadata,
            vocabulary: vec![],
            normalization_rules: vec![],
            merge_rules: vec![],
            added_tokens: vec![],
        };

        let text = ProtobufSerializer::to_proto_text(&model).unwrap();
        assert!(text.contains("name: \"test-tokenizer\""));
        assert!(text.contains("version: \"1.0\""));
        assert!(text.contains("vocab_size: 100"));
        assert!(text.contains("do_lower_case: true"));

        let parsed = ProtobufSerializer::from_proto_text(&text).unwrap();
        assert_eq!(parsed.metadata.name, "test-tokenizer");
        assert_eq!(parsed.metadata.version, "1.0");
        assert_eq!(parsed.metadata.vocab_size, 100);
        assert!(parsed.metadata.do_lower_case);
    }

    #[test]
    fn test_model_validation() {
        let metadata = ProtobufTokenizerMetadata {
            name: "test".to_string(),
            version: "1.0".to_string(),
            tokenizer_type: "test".to_string(),
            vocab_size: 2,
            special_tokens: HashMap::new(),
            max_length: None,
            truncation_side: "right".to_string(),
            padding_side: "right".to_string(),
            do_lower_case: false,
            strip_accents: None,
            add_prefix_space: false,
            trim_offsets: true,
            created_at: chrono::Utc::now().to_rfc3339(),
            model_id: None,
            custom_attributes: HashMap::new(),
        };

        let model = ProtobufTokenizerModel {
            metadata,
            vocabulary: vec![ProtobufVocabEntry {
                token: "hello".to_string(),
                id: 0,
                frequency: 0.1,
                is_special: false,
                token_type: 0,
            }], // Only 1 token but metadata claims 2
            normalization_rules: vec![],
            merge_rules: vec![],
            added_tokens: vec![],
        };

        let warnings = ProtobufSerializer::validate_model(&model).unwrap();
        assert!(!warnings.is_empty());
        assert!(warnings[0].contains("Vocabulary size mismatch"));
    }

    #[test]
    fn test_compression() {
        let metadata = ProtobufTokenizerMetadata {
            name: "test".to_string(),
            version: "1.0".to_string(),
            tokenizer_type: "test".to_string(),
            vocab_size: 0,
            special_tokens: HashMap::new(),
            max_length: None,
            truncation_side: "right".to_string(),
            padding_side: "right".to_string(),
            do_lower_case: false,
            strip_accents: None,
            add_prefix_space: false,
            trim_offsets: true,
            created_at: chrono::Utc::now().to_rfc3339(),
            model_id: None,
            custom_attributes: HashMap::new(),
        };

        let model = ProtobufTokenizerModel {
            metadata,
            vocabulary: vec![],
            normalization_rules: vec![],
            merge_rules: vec![],
            added_tokens: vec![],
        };

        let compressed = ProtobufSerializer::compress_model(&model).unwrap();
        let decompressed = ProtobufSerializer::decompress_model(&compressed).unwrap();

        assert_eq!(model.metadata.name, decompressed.metadata.name);
    }
}
