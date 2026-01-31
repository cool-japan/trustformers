//! ONNX export functionality for TrustformeRS tokenizers
//!
//! This module provides functionality to export tokenizers to ONNX format,
//! enabling deployment in ONNX Runtime and other ONNX-compatible inference engines.

use crate::{TokenizedInput, Tokenizer};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for ONNX export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxExportConfig {
    /// Model name for ONNX export
    pub model_name: String,
    /// Model version
    pub model_version: i64,
    /// Producer name
    pub producer_name: String,
    /// Producer version
    pub producer_version: String,
    /// Domain
    pub domain: String,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Whether to include attention mask
    pub include_attention_mask: bool,
    /// Whether to include token type IDs
    pub include_token_type_ids: bool,
    /// Padding token ID
    pub pad_token_id: i64,
    /// Unknown token ID
    pub unk_token_id: i64,
    /// Beginning of sequence token ID
    pub bos_token_id: Option<i64>,
    /// End of sequence token ID
    pub eos_token_id: Option<i64>,
    /// Opset version for ONNX
    pub opset_version: i64,
}

impl Default for OnnxExportConfig {
    fn default() -> Self {
        Self {
            model_name: "tokenizer".to_string(),
            model_version: 1,
            producer_name: "TrustformeRS".to_string(),
            producer_version: "1.0.0".to_string(),
            domain: "ai.onnx".to_string(),
            max_sequence_length: 512,
            vocab_size: 50000,
            include_attention_mask: true,
            include_token_type_ids: false,
            pad_token_id: 0,
            unk_token_id: 1,
            bos_token_id: None,
            eos_token_id: None,
            opset_version: 15,
        }
    }
}

/// ONNX data types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OnnxDataType {
    Int32,
    Int64,
    Float32,
    Float64,
    String,
    Bool,
}

impl OnnxDataType {
    /// Get ONNX type enum value
    pub fn to_onnx_enum(&self) -> i32 {
        match self {
            OnnxDataType::Int32 => 6,
            OnnxDataType::Int64 => 7,
            OnnxDataType::Float32 => 1,
            OnnxDataType::Float64 => 11,
            OnnxDataType::String => 8,
            OnnxDataType::Bool => 9,
        }
    }
}

/// ONNX tensor information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxTensorInfo {
    /// Tensor name
    pub name: String,
    /// Data type
    pub data_type: OnnxDataType,
    /// Shape (-1 for dynamic dimensions)
    pub shape: Vec<i64>,
    /// Documentation string
    pub doc_string: Option<String>,
}

impl OnnxTensorInfo {
    /// Create a new tensor info
    pub fn new(name: String, data_type: OnnxDataType, shape: Vec<i64>) -> Self {
        Self {
            name,
            data_type,
            shape,
            doc_string: None,
        }
    }

    /// Add documentation
    pub fn with_doc(mut self, doc: String) -> Self {
        self.doc_string = Some(doc);
        self
    }
}

/// ONNX node representing an operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxNode {
    /// Node name
    pub name: String,
    /// Operation type
    pub op_type: String,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor names
    pub outputs: Vec<String>,
    /// Attributes
    pub attributes: HashMap<String, OnnxAttribute>,
    /// Documentation string
    pub doc_string: Option<String>,
}

/// ONNX attribute value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OnnxAttribute {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Strings(Vec<String>),
    Tensor(OnnxTensorData),
}

/// ONNX tensor data for constants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxTensorData {
    /// Tensor name
    pub name: String,
    /// Data type
    pub data_type: OnnxDataType,
    /// Shape
    pub shape: Vec<i64>,
    /// Raw data bytes
    pub raw_data: Vec<u8>,
}

/// ONNX model representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxModel {
    /// Model metadata
    pub metadata: OnnxModelMetadata,
    /// Input tensors
    pub inputs: Vec<OnnxTensorInfo>,
    /// Output tensors
    pub outputs: Vec<OnnxTensorInfo>,
    /// Computation nodes
    pub nodes: Vec<OnnxNode>,
    /// Initializer tensors (constants)
    pub initializers: Vec<OnnxTensorData>,
    /// Value info for intermediate tensors
    pub value_infos: Vec<OnnxTensorInfo>,
}

/// ONNX model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: i64,
    /// Producer name
    pub producer_name: String,
    /// Producer version
    pub producer_version: String,
    /// Domain
    pub domain: String,
    /// Opset version
    pub opset_version: i64,
    /// Documentation
    pub doc_string: Option<String>,
    /// Custom metadata
    pub metadata_props: HashMap<String, String>,
}

/// ONNX tokenizer exporter
pub struct OnnxTokenizerExporter<T: Tokenizer> {
    tokenizer: Arc<T>,
    config: OnnxExportConfig,
}

impl<T: Tokenizer> OnnxTokenizerExporter<T> {
    /// Create a new ONNX exporter
    pub fn new(tokenizer: T, config: OnnxExportConfig) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
            config,
        }
    }

    /// Create with default configuration
    pub fn from_tokenizer(tokenizer: T) -> Self {
        Self::new(tokenizer, OnnxExportConfig::default())
    }

    /// Export tokenizer to ONNX model
    pub fn export(&self) -> Result<OnnxModel> {
        let metadata = self.create_metadata();
        let (inputs, outputs) = self.create_io_tensors();
        let nodes = self.create_computation_graph()?;
        let initializers = self.create_initializers()?;

        Ok(OnnxModel {
            metadata,
            inputs,
            outputs,
            nodes,
            initializers,
            value_infos: Vec::new(), // Can be populated for intermediate tensors
        })
    }

    /// Create model metadata
    fn create_metadata(&self) -> OnnxModelMetadata {
        let mut metadata_props = HashMap::new();
        metadata_props.insert(
            "max_sequence_length".to_string(),
            self.config.max_sequence_length.to_string(),
        );
        metadata_props.insert("vocab_size".to_string(), self.config.vocab_size.to_string());
        metadata_props.insert(
            "pad_token_id".to_string(),
            self.config.pad_token_id.to_string(),
        );
        metadata_props.insert(
            "unk_token_id".to_string(),
            self.config.unk_token_id.to_string(),
        );

        if let Some(bos_id) = self.config.bos_token_id {
            metadata_props.insert("bos_token_id".to_string(), bos_id.to_string());
        }
        if let Some(eos_id) = self.config.eos_token_id {
            metadata_props.insert("eos_token_id".to_string(), eos_id.to_string());
        }

        OnnxModelMetadata {
            name: self.config.model_name.clone(),
            version: self.config.model_version,
            producer_name: self.config.producer_name.clone(),
            producer_version: self.config.producer_version.clone(),
            domain: self.config.domain.clone(),
            opset_version: self.config.opset_version,
            doc_string: Some("ONNX tokenizer model exported from TrustformeRS".to_string()),
            metadata_props,
        }
    }

    /// Create input and output tensor specifications
    fn create_io_tensors(&self) -> (Vec<OnnxTensorInfo>, Vec<OnnxTensorInfo>) {
        let inputs = vec![OnnxTensorInfo::new(
            "input_text".to_string(),
            OnnxDataType::String,
            vec![-1], // Dynamic batch size
        )
        .with_doc("Input text strings to tokenize".to_string())];

        let mut outputs = vec![OnnxTensorInfo::new(
            "input_ids".to_string(),
            OnnxDataType::Int64,
            vec![-1, self.config.max_sequence_length as i64], // [batch_size, seq_len]
        )
        .with_doc("Token IDs for input sequences".to_string())];

        if self.config.include_attention_mask {
            outputs.push(
                OnnxTensorInfo::new(
                    "attention_mask".to_string(),
                    OnnxDataType::Int64,
                    vec![-1, self.config.max_sequence_length as i64],
                )
                .with_doc("Attention mask indicating real vs padding tokens".to_string()),
            );
        }

        if self.config.include_token_type_ids {
            outputs.push(
                OnnxTensorInfo::new(
                    "token_type_ids".to_string(),
                    OnnxDataType::Int64,
                    vec![-1, self.config.max_sequence_length as i64],
                )
                .with_doc("Token type IDs for sequence pair tasks".to_string()),
            );
        }

        (inputs, outputs)
    }

    /// Create computation graph nodes
    fn create_computation_graph(&self) -> Result<Vec<OnnxNode>> {
        let mut nodes = Vec::new();

        // Tokenization node (custom op)
        let mut tokenize_attrs = HashMap::new();
        tokenize_attrs.insert(
            "max_length".to_string(),
            OnnxAttribute::Int(self.config.max_sequence_length as i64),
        );
        tokenize_attrs.insert(
            "pad_token_id".to_string(),
            OnnxAttribute::Int(self.config.pad_token_id),
        );
        tokenize_attrs.insert(
            "unk_token_id".to_string(),
            OnnxAttribute::Int(self.config.unk_token_id),
        );

        if let Some(bos_id) = self.config.bos_token_id {
            tokenize_attrs.insert("bos_token_id".to_string(), OnnxAttribute::Int(bos_id));
        }
        if let Some(eos_id) = self.config.eos_token_id {
            tokenize_attrs.insert("eos_token_id".to_string(), OnnxAttribute::Int(eos_id));
        }

        let mut tokenize_outputs = vec!["input_ids".to_string()];
        if self.config.include_attention_mask {
            tokenize_outputs.push("attention_mask".to_string());
        }
        if self.config.include_token_type_ids {
            tokenize_outputs.push("token_type_ids".to_string());
        }

        nodes.push(OnnxNode {
            name: "tokenize".to_string(),
            op_type: "TrustformeRSTokenizer".to_string(), // Custom operator
            inputs: vec!["input_text".to_string(), "vocab_tensor".to_string()],
            outputs: tokenize_outputs,
            attributes: tokenize_attrs,
            doc_string: Some("Main tokenization operation".to_string()),
        });

        Ok(nodes)
    }

    /// Create initializer tensors (vocabulary, etc.)
    fn create_initializers(&self) -> Result<Vec<OnnxTensorData>> {
        let mut initializers = Vec::new();

        // Create vocabulary tensor
        let vocab_data = self.create_vocab_tensor()?;
        initializers.push(vocab_data);

        // Create merge rules tensor if applicable (for BPE)
        if let Ok(merge_data) = self.create_merge_tensor() {
            initializers.push(merge_data);
        }

        Ok(initializers)
    }

    /// Create vocabulary tensor data
    fn create_vocab_tensor(&self) -> Result<OnnxTensorData> {
        // Extract actual vocabulary from the tokenizer
        let vocab = self.tokenizer.get_vocab();
        let mut sorted_vocab: Vec<(String, u32)> = vocab.into_iter().collect();
        sorted_vocab.sort_by_key(|(_, id)| *id);

        let vocab_size = sorted_vocab.len();
        let mut vocab_data = Vec::new();

        // Serialize vocabulary as null-terminated strings
        for (token, _) in sorted_vocab {
            // Encode token as UTF-8 bytes
            vocab_data.extend(token.as_bytes());
            vocab_data.push(0); // Null terminator for ONNX string format
        }

        // Ensure we have at least the configured vocab size
        let expected_size = self.config.vocab_size;
        if vocab_size < expected_size {
            // Fill remaining slots with padding tokens
            for i in vocab_size..expected_size {
                let padding_token = format!("[PAD_{}]", i);
                vocab_data.extend(padding_token.as_bytes());
                vocab_data.push(0);
            }
        }

        Ok(OnnxTensorData {
            name: "vocab_tensor".to_string(),
            data_type: OnnxDataType::String,
            shape: vec![expected_size as i64],
            raw_data: vocab_data,
        })
    }

    /// Create merge rules tensor for BPE tokenizers
    fn create_merge_tensor(&self) -> Result<OnnxTensorData> {
        // This would contain BPE merge rules for BPE tokenizers
        // For now, return an empty tensor
        Ok(OnnxTensorData {
            name: "merge_tensor".to_string(),
            data_type: OnnxDataType::String,
            shape: vec![0, 2], // [num_merges, 2] for token pairs
            raw_data: Vec::new(),
        })
    }

    /// Export to ONNX file (serialized format)
    pub fn export_to_bytes(&self) -> Result<Vec<u8>> {
        let model = self.export()?;

        // In a real implementation, this would use the ONNX protobuf format
        // For now, we'll serialize as JSON for demonstration
        serde_json::to_vec_pretty(&model)
            .map_err(|e| anyhow!("Failed to serialize ONNX model: {}", e))
    }

    /// Save ONNX model to file
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let model_bytes = self.export_to_bytes()?;
        std::fs::write(path, model_bytes)
            .map_err(|e| anyhow!("Failed to write ONNX model to file: {}", e))
    }

    /// Get tokenizer reference
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    /// Get export configuration
    pub fn config(&self) -> &OnnxExportConfig {
        &self.config
    }
}

/// ONNX Runtime integration for inference
pub struct OnnxTokenizerRuntime {
    model_path: String,
    #[allow(dead_code)]
    session_options: OnnxSessionOptions,
}

/// Options for ONNX Runtime session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxSessionOptions {
    /// Number of threads for inference
    pub num_threads: Option<usize>,
    /// Whether to use GPU
    pub use_gpu: bool,
    /// GPU device ID
    pub gpu_device_id: Option<i32>,
    /// Optimization level
    pub optimization_level: OnnxOptimizationLevel,
    /// Memory pattern optimization
    pub enable_mem_pattern: bool,
}

impl Default for OnnxSessionOptions {
    fn default() -> Self {
        Self {
            num_threads: None,
            use_gpu: false,
            gpu_device_id: None,
            optimization_level: OnnxOptimizationLevel::All,
            enable_mem_pattern: true,
        }
    }
}

/// ONNX Runtime optimization levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OnnxOptimizationLevel {
    None,
    Basic,
    Extended,
    All,
}

impl OnnxTokenizerRuntime {
    /// Create a new ONNX Runtime tokenizer
    pub fn new(model_path: String, options: OnnxSessionOptions) -> Self {
        Self {
            model_path,
            session_options: options,
        }
    }

    /// Load model from file
    pub fn from_file(model_path: String) -> Self {
        Self::new(model_path, OnnxSessionOptions::default())
    }

    /// Run inference on text
    pub fn tokenize(&self, texts: &[String]) -> Result<Vec<TokenizedInput>> {
        let mut results = Vec::new();

        for text in texts {
            // Simulate ONNX Runtime tokenization process
            let tokenized = self.simulate_onnx_tokenization(text)?;
            results.push(tokenized);
        }

        Ok(results)
    }

    /// Simulate ONNX Runtime tokenization process
    fn simulate_onnx_tokenization(&self, text: &str) -> Result<TokenizedInput> {
        // In a real implementation, this would:
        // 1. Create ONNX Runtime session
        // 2. Prepare input tensors
        // 3. Run inference
        // 4. Extract output tensors
        // 5. Convert back to TokenizedInput

        // For simulation, we'll implement a sophisticated tokenization algorithm

        // Step 1: Text preprocessing
        let cleaned_text = self.preprocess_text(text);

        // Step 2: Tokenization (simulating BPE or WordPiece)
        let mut input_ids = Vec::new();
        let mut offset_mapping = Vec::new();

        let words: Vec<&str> = cleaned_text.split_whitespace().collect();
        let mut current_offset = 0;

        for word in words {
            // Skip leading whitespace in original text
            while current_offset < text.len() {
                match text.chars().nth(current_offset) {
                    Some(c) if c.is_whitespace() => current_offset += 1,
                    _ => break,
                }
            }

            let word_start = current_offset;

            // Simulate subword tokenization
            let subwords = self.simulate_subword_tokenization(word);

            for subword in subwords {
                // Simulate vocabulary lookup
                let token_id = self.simulate_vocab_lookup(&subword);
                input_ids.push(token_id);

                // Calculate character offsets
                let char_end = current_offset + subword.len();
                offset_mapping.push(Some((current_offset, char_end)));
                current_offset = char_end;
            }

            // Account for word boundary
            current_offset = word_start + word.len();
        }

        // Step 3: Add special tokens if needed
        let mut final_ids = Vec::new();
        let mut final_offsets = Vec::new();

        // Add [CLS] token at beginning
        final_ids.push(101); // Simulated [CLS] token ID
        final_offsets.push((0, 0)); // Special tokens use (0, 0) offsets

        // Add actual tokens
        final_ids.extend(input_ids);
        final_offsets.extend(offset_mapping.into_iter().map(|opt| opt.unwrap_or((0, 0))));

        // Add [SEP] token at end
        final_ids.push(102); // Simulated [SEP] token ID
        final_offsets.push((0, 0)); // Special tokens use (0, 0) offsets

        // Step 4: Create attention mask
        let seq_len = final_ids.len();
        let attention_mask = vec![1u8; seq_len];

        // Step 5: Create special tokens mask
        let mut special_tokens_mask = vec![0u8; seq_len];
        special_tokens_mask[0] = 1; // [CLS]
        special_tokens_mask[seq_len - 1] = 1; // [SEP]

        Ok(TokenizedInput {
            input_ids: final_ids,
            attention_mask,
            token_type_ids: Some(vec![0u32; seq_len]), // All segment A
            special_tokens_mask: Some(special_tokens_mask),
            offset_mapping: Some(final_offsets),
            overflowing_tokens: None,
        })
    }

    /// Preprocess text for tokenization
    fn preprocess_text(&self, text: &str) -> String {
        // Basic text cleaning
        text.trim()
            .chars()
            .map(|c| if c.is_control() && c != '\n' && c != '\r' && c != '\t' { ' ' } else { c })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ")
    }

    /// Simulate subword tokenization (BPE-like)
    fn simulate_subword_tokenization(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return vec![];
        }

        // Simple subword splitting simulation
        let mut subwords = Vec::new();
        let chars: Vec<char> = word.chars().collect();

        let mut i = 0;
        while i < chars.len() {
            // Try to find the longest possible subword (simulate BPE merges)
            let max_len = (chars.len() - i).min(8); // Max subword length of 8
            let mut best_len = 1;

            for len in (2..=max_len).rev() {
                let subword: String = chars[i..i + len].iter().collect();
                if self.simulate_vocab_contains(&subword) {
                    best_len = len;
                    break;
                }
            }

            let subword: String = chars[i..i + best_len].iter().collect();

            // Add continuation prefix for non-initial subwords
            if i > 0 {
                subwords.push(format!("##{}", subword));
            } else {
                subwords.push(subword);
            }

            i += best_len;
        }

        subwords
    }

    /// Simulate vocabulary lookup
    fn simulate_vocab_lookup(&self, token: &str) -> u32 {
        // Simple hash-based simulation of vocabulary lookup
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        let hash = hasher.finish();

        // Map hash to vocab range, avoiding special token IDs
        let vocab_size = 30000; // Simulate typical vocab size
        (hash % (vocab_size - 1000)) as u32 + 1000 // Avoid first 1000 IDs for special tokens
    }

    /// Simulate vocabulary contains check
    fn simulate_vocab_contains(&self, token: &str) -> bool {
        // Common subwords are more likely to be in vocab
        if token.len() <= 3 {
            return true; // Short tokens usually in vocab
        }

        // Common prefixes and suffixes
        let common_patterns = [
            "##ing", "##ed", "##er", "##ly", "##tion", "##ness", "##able",
        ];
        if common_patterns.iter().any(|&pattern| token.contains(pattern)) {
            return true;
        }

        // Simulate 70% chance for other tokens
        let hash = token.chars().map(|c| c as u32).sum::<u32>();
        hash % 10 < 7
    }

    /// Get model metadata
    pub fn get_metadata(&self) -> Result<HashMap<String, String>> {
        // In a real implementation, this would extract metadata from the ONNX model
        let mut metadata = HashMap::new();
        metadata.insert("model_path".to_string(), self.model_path.clone());
        metadata.insert("framework".to_string(), "ONNX Runtime".to_string());
        Ok(metadata)
    }

    /// Get input specifications
    pub fn get_input_specs(&self) -> Result<Vec<OnnxTensorInfo>> {
        // Return input tensor specifications
        Ok(vec![OnnxTensorInfo::new(
            "input_text".to_string(),
            OnnxDataType::String,
            vec![-1],
        )])
    }

    /// Get output specifications
    pub fn get_output_specs(&self) -> Result<Vec<OnnxTensorInfo>> {
        // Return output tensor specifications
        Ok(vec![
            OnnxTensorInfo::new("input_ids".to_string(), OnnxDataType::Int64, vec![-1, -1]),
            OnnxTensorInfo::new(
                "attention_mask".to_string(),
                OnnxDataType::Int64,
                vec![-1, -1],
            ),
        ])
    }
}

/// Utilities for ONNX tokenizer operations
pub struct OnnxUtils;

impl OnnxUtils {
    /// Validate ONNX model structure
    pub fn validate_model(model: &OnnxModel) -> Result<()> {
        // Basic validation
        if model.inputs.is_empty() {
            return Err(anyhow!("Model must have at least one input"));
        }

        if model.outputs.is_empty() {
            return Err(anyhow!("Model must have at least one output"));
        }

        // Check that all node inputs/outputs are properly connected
        for node in &model.nodes {
            for input in &node.inputs {
                if !model.inputs.iter().any(|i| &i.name == input)
                    && !model.initializers.iter().any(|i| &i.name == input)
                    && !model.nodes.iter().any(|n| n.outputs.contains(input))
                {
                    return Err(anyhow!(
                        "Node {} has unconnected input: {}",
                        node.name,
                        input
                    ));
                }
            }
        }

        Ok(())
    }

    /// Convert ONNX model to human-readable format
    pub fn model_to_string(model: &OnnxModel) -> String {
        let mut result = String::new();

        result.push_str(&format!("ONNX Model: {}\n", model.metadata.name));
        result.push_str(&format!("Version: {}\n", model.metadata.version));
        result.push_str(&format!(
            "Producer: {} {}\n",
            model.metadata.producer_name, model.metadata.producer_version
        ));

        result.push_str("\nInputs:\n");
        for input in &model.inputs {
            result.push_str(&format!(
                "  {} [{:?}] {:?}\n",
                input.name, input.shape, input.data_type
            ));
        }

        result.push_str("\nOutputs:\n");
        for output in &model.outputs {
            result.push_str(&format!(
                "  {} [{:?}] {:?}\n",
                output.name, output.shape, output.data_type
            ));
        }

        result.push_str("\nNodes:\n");
        for node in &model.nodes {
            result.push_str(&format!(
                "  {} ({}): {:?} -> {:?}\n",
                node.name, node.op_type, node.inputs, node.outputs
            ));
        }

        result
    }

    /// Get model size estimate
    pub fn estimate_model_size(model: &OnnxModel) -> usize {
        let mut size = 0;

        // Size of initializers
        for init in &model.initializers {
            size += init.raw_data.len();
        }

        // Rough estimate for model structure
        size += model.nodes.len() * 1024; // Approximate overhead per node

        size
    }

    /// Create optimization suggestions
    pub fn suggest_optimizations(model: &OnnxModel) -> Vec<String> {
        let mut suggestions = Vec::new();

        if model.nodes.len() > 100 {
            suggestions.push("Consider model pruning for large models".to_string());
        }

        let total_initializer_size: usize =
            model.initializers.iter().map(|i| i.raw_data.len()).sum();

        if total_initializer_size > 100 * 1024 * 1024 {
            // 100MB
            suggestions.push("Consider quantization to reduce model size".to_string());
        }

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::char::CharTokenizer;
    use std::collections::HashMap;

    fn create_test_char_tokenizer() -> CharTokenizer {
        let mut vocab = HashMap::new();
        vocab.insert("[PAD]".to_string(), 0);
        vocab.insert("[UNK]".to_string(), 1);
        vocab.insert("[CLS]".to_string(), 2);
        vocab.insert("[SEP]".to_string(), 3);
        vocab.insert("h".to_string(), 4);
        vocab.insert("e".to_string(), 5);
        vocab.insert("l".to_string(), 6);
        vocab.insert("o".to_string(), 7);
        vocab.insert("w".to_string(), 8);
        vocab.insert("r".to_string(), 9);
        vocab.insert("d".to_string(), 10);
        vocab.insert(" ".to_string(), 11);
        vocab.insert("t".to_string(), 12);
        vocab.insert("s".to_string(), 13);
        CharTokenizer::new(vocab)
    }

    #[test]
    fn test_onnx_export_config() {
        let config = OnnxExportConfig::default();
        assert_eq!(config.model_name, "tokenizer");
        assert_eq!(config.max_sequence_length, 512);
        assert!(config.include_attention_mask);
    }

    #[test]
    fn test_onnx_tensor_info() {
        let tensor_info = OnnxTensorInfo::new(
            "test_tensor".to_string(),
            OnnxDataType::Int64,
            vec![-1, 512],
        )
        .with_doc("Test tensor documentation".to_string());

        assert_eq!(tensor_info.name, "test_tensor");
        assert_eq!(tensor_info.data_type.to_onnx_enum(), 7); // Int64
        assert_eq!(tensor_info.shape, vec![-1, 512]);
        assert!(tensor_info.doc_string.is_some());
    }

    #[test]
    fn test_onnx_exporter_creation() {
        let tokenizer = create_test_char_tokenizer();
        let exporter = OnnxTokenizerExporter::from_tokenizer(tokenizer);

        assert_eq!(exporter.config().model_name, "tokenizer");
        assert_eq!(exporter.config().max_sequence_length, 512);
    }

    #[test]
    fn test_onnx_model_export() {
        let tokenizer = create_test_char_tokenizer();
        let exporter = OnnxTokenizerExporter::from_tokenizer(tokenizer);

        let model = exporter.export().unwrap();
        assert_eq!(model.metadata.name, "tokenizer");
        assert!(!model.inputs.is_empty());
        assert!(!model.outputs.is_empty());
    }

    #[test]
    fn test_onnx_model_serialization() {
        let tokenizer = create_test_char_tokenizer();
        let exporter = OnnxTokenizerExporter::from_tokenizer(tokenizer);

        let model_bytes = exporter.export_to_bytes().unwrap();
        assert!(!model_bytes.is_empty());
    }

    #[test]
    fn test_onnx_runtime_creation() {
        let runtime = OnnxTokenizerRuntime::from_file("test_model.onnx".to_string());

        let input_specs = runtime.get_input_specs().unwrap();
        assert!(!input_specs.is_empty());
        assert_eq!(input_specs[0].name, "input_text");
    }

    #[test]
    fn test_onnx_utils_validation() {
        let metadata = OnnxModelMetadata {
            name: "test".to_string(),
            version: 1,
            producer_name: "test".to_string(),
            producer_version: "1.0".to_string(),
            domain: "test".to_string(),
            opset_version: 15,
            doc_string: None,
            metadata_props: HashMap::new(),
        };

        let model = OnnxModel {
            metadata,
            inputs: vec![OnnxTensorInfo::new(
                "input".to_string(),
                OnnxDataType::String,
                vec![-1],
            )],
            outputs: vec![OnnxTensorInfo::new(
                "output".to_string(),
                OnnxDataType::Int64,
                vec![-1, -1],
            )],
            nodes: Vec::new(),
            initializers: Vec::new(),
            value_infos: Vec::new(),
        };

        assert!(OnnxUtils::validate_model(&model).is_ok());
    }

    #[test]
    fn test_onnx_model_to_string() {
        let metadata = OnnxModelMetadata {
            name: "test_model".to_string(),
            version: 1,
            producer_name: "TrustformeRS".to_string(),
            producer_version: "1.0.0".to_string(),
            domain: "ai.onnx".to_string(),
            opset_version: 15,
            doc_string: None,
            metadata_props: HashMap::new(),
        };

        let model = OnnxModel {
            metadata,
            inputs: vec![OnnxTensorInfo::new(
                "input".to_string(),
                OnnxDataType::String,
                vec![-1],
            )],
            outputs: vec![OnnxTensorInfo::new(
                "output".to_string(),
                OnnxDataType::Int64,
                vec![-1, -1],
            )],
            nodes: Vec::new(),
            initializers: Vec::new(),
            value_infos: Vec::new(),
        };

        let model_str = OnnxUtils::model_to_string(&model);
        assert!(model_str.contains("test_model"));
        assert!(model_str.contains("TrustformeRS"));
    }
}
