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
    /// Reserved for future use (e.g. a declared upper bound for the
    /// exported vocabulary). `create_vocab_tensor` always emits exactly
    /// the wrapped tokenizer's real vocabulary rather than padding or
    /// truncating to this value.
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
        // The *real* vocabulary size, not `self.config.vocab_size` (which is
        // only a caller-declared upper bound used to validate
        // `create_vocab_tensor`, and previously defaulted to 50000
        // regardless of how many tokens the wrapped tokenizer actually
        // had -- reporting it here as if it were the model's real
        // vocabulary size was misleading for every tokenizer smaller than
        // that default).
        metadata_props.insert(
            "vocab_size".to_string(),
            self.tokenizer.vocab_size().to_string(),
        );
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

    /// Create vocabulary tensor data.
    ///
    /// A real ONNX string tensor of shape `[vocab_size]` encodes token `i`
    /// at position `i` -- there is no separate "id" field alongside each
    /// string. That convention only holds for a dense, 0-based vocabulary,
    /// so a tokenizer whose `get_vocab()` ids are not exactly `0..len()`
    /// (with no gaps or duplicates) is rejected here rather than silently
    /// mis-encoded. Previously this method instead padded every vocabulary
    /// smaller than `self.config.vocab_size` (default `50000`) with
    /// invented `[PAD_N]` entries, claiming a vocabulary far larger than
    /// the tokenizer actually had; the tensor now always contains exactly
    /// (and only) the tokenizer's real vocabulary, in real id order, which
    /// is also what lets [`OnnxTokenizerRuntime::from_file`] recover the
    /// real vocabulary losslessly by position.
    fn create_vocab_tensor(&self) -> Result<OnnxTensorData> {
        let vocab = self.tokenizer.get_vocab();
        let vocab_size = vocab.len();

        let mut ordered: Vec<Option<String>> = vec![None; vocab_size];
        for (token, id) in vocab {
            match ordered.get_mut(id as usize) {
                Some(slot @ None) => *slot = Some(token),
                Some(Some(existing)) => {
                    return Err(anyhow!(
                        "tokenizer vocabulary has two tokens sharing id {}: {:?} and {:?}",
                        id,
                        existing,
                        token
                    ));
                },
                None => {
                    return Err(anyhow!(
                        "tokenizer vocabulary is not densely 0-based: id {} is outside the \
                         [0, {}) range a vocab_tensor represents by position",
                        id,
                        vocab_size
                    ));
                },
            }
        }
        let ordered: Vec<String> = ordered
            .into_iter()
            .enumerate()
            .map(|(id, slot)| {
                slot.ok_or_else(|| {
                    anyhow!(
                        "tokenizer vocabulary is missing id {} (not densely 0-based)",
                        id
                    )
                })
            })
            .collect::<Result<_>>()?;

        // Serialize vocabulary as null-terminated strings, in real id order.
        let mut vocab_data = Vec::new();
        for token in &ordered {
            vocab_data.extend(token.as_bytes());
            vocab_data.push(0); // Null terminator for ONNX string format
        }

        Ok(OnnxTensorData {
            name: "vocab_tensor".to_string(),
            data_type: OnnxDataType::String,
            shape: vec![ordered.len() as i64],
            raw_data: vocab_data,
        })
    }

    /// Merge-rules tensor for BPE tokenizers.
    ///
    /// Always empty: `OnnxTokenizerExporter<T>` is generic over any
    /// [`Tokenizer`] implementation, and that trait exposes no BPE merge
    /// rules -- only a BPE-specific type carries them, with no shared
    /// trait method to reach them generically. This is a real, documented
    /// API limitation of exporting through the generic [`Tokenizer`]
    /// trait, not a placeholder standing in for data this method silently
    /// fails to produce for a tokenizer that could otherwise supply it.
    fn create_merge_tensor(&self) -> Result<OnnxTensorData> {
        Ok(OnnxTensorData {
            name: "merge_tensor".to_string(),
            data_type: OnnxDataType::String,
            shape: vec![0, 2], // [num_merges, 2] for token pairs
            raw_data: Vec::new(),
        })
    }

    /// Serialize the exported model to this crate's own JSON interchange
    /// format.
    ///
    /// This is **not** the binary ONNX protobuf wire format that
    /// `onnxruntime` or other real ONNX runtimes consume -- this crate
    /// does not implement an ONNX protobuf encoder. It is a JSON dump of
    /// [`OnnxModel`] (metadata, tensor specs, the computation-graph
    /// description, and a real `vocab_tensor` initializer built from the
    /// wrapped tokenizer's actual vocabulary), structured to mirror
    /// ONNX's graph model closely enough that
    /// [`OnnxTokenizerRuntime::from_file`] can load it back and tokenize
    /// with the real recovered vocabulary. Do not feed the output of this
    /// method to a real ONNX runtime.
    pub fn export_to_bytes(&self) -> Result<Vec<u8>> {
        let model = self.export()?;

        serde_json::to_vec_pretty(&model)
            .map_err(|e| anyhow!("Failed to serialize tokenizer model: {}", e))
    }

    /// Save the exported model to `path` in this crate's own JSON
    /// interchange format -- see [`Self::export_to_bytes`]. Despite the
    /// "ONNX" naming (this type mirrors ONNX's graph/tensor model closely
    /// enough to interoperate with [`OnnxTokenizerRuntime`]), the file
    /// this writes is not a real ONNX protobuf model.
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

/// ONNX Runtime integration for inference.
///
/// This crate does not implement a real ONNX Runtime session or a binary
/// ONNX protobuf parser. [`Self::from_file`]/[`Self::new`] load this
/// crate's own JSON interchange format (see
/// [`OnnxTokenizerExporter::export_to_bytes`]) and recover the real
/// vocabulary and special-token configuration that was exported into it;
/// [`Self::tokenize`] then performs real, deterministic greedy
/// longest-match tokenization against that recovered vocabulary. A genuine
/// binary ONNX protobuf `.onnx` file, or any other file not in this
/// crate's export format, is rejected with a structured error at load time
/// rather than silently producing hash-derived fake token IDs.
pub struct OnnxTokenizerRuntime {
    model_path: String,
    // reason: stored from the constructor; reserved for forwarding session tuning
    // to a real ONNX Runtime session once one is wired up. Not read by the
    // real-vocabulary greedy-tokenization path below.
    #[allow(dead_code)]
    session_options: OnnxSessionOptions,
    loaded: LoadedTokenizerModel,
}

/// The real, recovered contents of an exported tokenizer model.
struct LoadedTokenizerModel {
    /// `id -> token`, dense and 0-based (see [`OnnxTokenizerExporter::create_vocab_tensor`]).
    id_to_token: Vec<String>,
    /// `token -> id`, the inverse of `id_to_token`.
    token_to_id: HashMap<String, u32>,
    unk_token_id: u32,
    #[allow(dead_code)] // Reserved: padding is not yet implemented for this runtime.
    pad_token_id: u32,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    max_sequence_length: Option<usize>,
    metadata_props: HashMap<String, String>,
    inputs: Vec<OnnxTensorInfo>,
    outputs: Vec<OnnxTensorInfo>,
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
    /// Load a tokenizer model with explicit session options.
    ///
    /// Reads and parses `model_path` immediately -- unlike the earlier
    /// version of this type, construction genuinely opens and validates
    /// the file rather than only remembering its path. Fails if the file
    /// cannot be read, is not this crate's JSON export format (including a
    /// genuine binary ONNX protobuf model, which this crate does not
    /// parse), or is missing the vocabulary/special-token metadata
    /// [`OnnxTokenizerExporter`] always writes.
    pub fn new(model_path: String, session_options: OnnxSessionOptions) -> Result<Self> {
        let bytes = std::fs::read(&model_path).map_err(|e| {
            anyhow!(
                "Failed to read tokenizer model file {:?}: {}",
                model_path,
                e
            )
        })?;
        let loaded = Self::load(&bytes)?;
        Ok(Self {
            model_path,
            session_options,
            loaded,
        })
    }

    /// Load a tokenizer model, using default session options.
    pub fn from_file(model_path: String) -> Result<Self> {
        Self::new(model_path, OnnxSessionOptions::default())
    }

    /// Parse this crate's JSON export format and recover its real
    /// vocabulary and special-token configuration.
    fn load(bytes: &[u8]) -> Result<LoadedTokenizerModel> {
        let model: OnnxModel = serde_json::from_slice(bytes).map_err(|e| {
            anyhow!(
                "not a TrustformeRS tokenizer export: this crate reads back its own JSON \
                 interchange format from `OnnxTokenizerExporter::export_to_bytes`; it does not \
                 implement a binary ONNX protobuf parser, so a genuine `.onnx` model file (or \
                 any other unrecognized file) is rejected here rather than producing fabricated \
                 tokenization ({})",
                e
            )
        })?;

        let vocab_tensor =
            model.initializers.iter().find(|init| init.name == "vocab_tensor").ok_or_else(
                || anyhow!("tokenizer model is missing its \"vocab_tensor\" initializer"),
            )?;

        // `create_vocab_tensor` writes each token as a null-terminated
        // UTF-8 string, in real vocabulary-id order; position in the split
        // sequence is the token's id (see that method's docs).
        let id_to_token: Vec<String> = vocab_tensor
            .raw_data
            .split(|&b| b == 0)
            .filter(|chunk| !chunk.is_empty())
            .map(|chunk| {
                String::from_utf8(chunk.to_vec())
                    .map_err(|e| anyhow!("vocab_tensor contains invalid UTF-8: {}", e))
            })
            .collect::<Result<Vec<_>>>()?;
        if id_to_token.is_empty() {
            return Err(anyhow!("tokenizer model's vocab_tensor is empty"));
        }

        let mut token_to_id = HashMap::with_capacity(id_to_token.len());
        for (id, token) in id_to_token.iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
        }

        let props = model.metadata.metadata_props.clone();
        let parse_id = |key: &str| -> Option<u32> {
            props
                .get(key)
                .and_then(|v| v.parse::<i64>().ok())
                .and_then(|v| u32::try_from(v).ok())
        };

        let unk_token_id = parse_id("unk_token_id").ok_or_else(|| {
            anyhow!("tokenizer model metadata is missing a valid \"unk_token_id\"")
        })?;
        let pad_token_id = parse_id("pad_token_id").ok_or_else(|| {
            anyhow!("tokenizer model metadata is missing a valid \"pad_token_id\"")
        })?;
        let bos_token_id = parse_id("bos_token_id");
        let eos_token_id = parse_id("eos_token_id");
        let max_sequence_length =
            props.get("max_sequence_length").and_then(|v| v.parse::<usize>().ok());

        Ok(LoadedTokenizerModel {
            id_to_token,
            token_to_id,
            unk_token_id,
            pad_token_id,
            bos_token_id,
            eos_token_id,
            max_sequence_length,
            metadata_props: props,
            inputs: model.inputs,
            outputs: model.outputs,
        })
    }

    /// Tokenize `texts` using the real vocabulary recovered from the
    /// loaded model. Every ID comes from a real lookup against that
    /// vocabulary; none is derived from a hash of the input text.
    pub fn tokenize(&self, texts: &[String]) -> Result<Vec<TokenizedInput>> {
        texts.iter().map(|text| self.tokenize_one(text)).collect()
    }

    fn tokenize_one(&self, text: &str) -> Result<TokenizedInput> {
        let cleaned_text = self.preprocess_text(text);
        let (piece_ids, offsets) = self.greedy_longest_match(&cleaned_text);

        let mut final_ids = Vec::with_capacity(piece_ids.len() + 2);
        let mut final_offsets = Vec::with_capacity(piece_ids.len() + 2);
        let mut special_tokens_mask = Vec::with_capacity(piece_ids.len() + 2);

        if let Some(bos_id) = self.loaded.bos_token_id {
            final_ids.push(bos_id);
            final_offsets.push((0, 0));
            special_tokens_mask.push(1);
        }

        for (id, offset) in piece_ids.into_iter().zip(offsets) {
            final_ids.push(id);
            final_offsets.push(offset);
            special_tokens_mask.push(0);
        }

        if let Some(eos_id) = self.loaded.eos_token_id {
            final_ids.push(eos_id);
            final_offsets.push((0, 0));
            special_tokens_mask.push(1);
        }

        if let Some(max_len) = self.loaded.max_sequence_length {
            if max_len > 0 && final_ids.len() > max_len {
                final_ids.truncate(max_len);
                final_offsets.truncate(max_len);
                special_tokens_mask.truncate(max_len);
            }
        }

        let seq_len = final_ids.len();
        let attention_mask = vec![1u8; seq_len];

        Ok(TokenizedInput {
            input_ids: final_ids,
            attention_mask,
            token_type_ids: Some(vec![0u32; seq_len]), // All segment A
            special_tokens_mask: Some(special_tokens_mask),
            offset_mapping: Some(final_offsets),
            overflowing_tokens: None,
        })
    }

    /// Basic text cleanup: collapse control characters and whitespace runs.
    fn preprocess_text(&self, text: &str) -> String {
        text.trim()
            .chars()
            .map(|c| if c.is_control() && c != '\n' && c != '\r' && c != '\t' { ' ' } else { c })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ")
    }

    /// Real greedy longest-match segmentation over the recovered
    /// vocabulary (WordPiece-style): at each position, the longest
    /// vocabulary entry starting there is used, falling back to the
    /// configured unknown-token id for exactly one character when nothing
    /// matches. Deterministic and entirely dependent on the real
    /// vocabulary content recovered from the model file -- never
    /// hash-derived, and never a fixed "N% chance" heuristic.
    fn greedy_longest_match(&self, text: &str) -> (Vec<u32>, Vec<(usize, usize)>) {
        const MAX_PIECE_CHARS: usize = 32;

        let chars: Vec<char> = text.chars().collect();
        let mut ids = Vec::with_capacity(chars.len());
        let mut offsets = Vec::with_capacity(chars.len());
        let mut byte_pos = 0usize;
        let mut i = 0usize;

        while i < chars.len() {
            let max_len = (chars.len() - i).min(MAX_PIECE_CHARS);
            let mut found: Option<(usize, u32)> = None;

            for len in (1..=max_len).rev() {
                let candidate: String = chars[i..i + len].iter().collect();
                if let Some(&id) = self.loaded.token_to_id.get(&candidate) {
                    found = Some((len, id));
                    break;
                }
            }

            let (piece_len, id) = found.unwrap_or((1, self.loaded.unk_token_id));
            let piece_byte_len: usize = chars[i..i + piece_len].iter().map(|c| c.len_utf8()).sum();

            ids.push(id);
            offsets.push((byte_pos, byte_pos + piece_byte_len));
            byte_pos += piece_byte_len;
            i += piece_len;
        }

        (ids, offsets)
    }

    /// Real metadata recovered from the loaded model (not hardcoded
    /// placeholders).
    pub fn get_metadata(&self) -> Result<HashMap<String, String>> {
        let mut metadata = self.loaded.metadata_props.clone();
        metadata.insert("model_path".to_string(), self.model_path.clone());
        metadata.insert(
            "vocab_size".to_string(),
            self.loaded.id_to_token.len().to_string(),
        );
        Ok(metadata)
    }

    /// Real input tensor specs recovered from the loaded model.
    pub fn get_input_specs(&self) -> Result<Vec<OnnxTensorInfo>> {
        Ok(self.loaded.inputs.clone())
    }

    /// Real output tensor specs recovered from the loaded model.
    pub fn get_output_specs(&self) -> Result<Vec<OnnxTensorInfo>> {
        Ok(self.loaded.outputs.clone())
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
    use tempfile::tempdir;

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

        let model = exporter.export().expect("Operation failed in test");
        assert_eq!(model.metadata.name, "tokenizer");
        assert!(!model.inputs.is_empty());
        assert!(!model.outputs.is_empty());
    }

    #[test]
    fn test_onnx_model_serialization() {
        let tokenizer = create_test_char_tokenizer();
        let exporter = OnnxTokenizerExporter::from_tokenizer(tokenizer);

        let model_bytes = exporter.export_to_bytes().expect("Operation failed in test");
        assert!(!model_bytes.is_empty());
    }

    /// Regression test: `from_file` used to return an infallible `Self`
    /// that only remembered the path, never opening it. It must now
    /// actually try to read the file and fail when it does not exist.
    #[test]
    fn test_onnx_runtime_from_file_rejects_missing_file() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let missing_path = temp_dir.path().join("does-not-exist.onnx");

        let result =
            OnnxTokenizerRuntime::from_file(missing_path.to_str().expect("utf8 path").to_string());
        assert!(result.is_err());
    }

    /// A file that is not this crate's JSON export format (standing in for
    /// a genuine binary ONNX protobuf model, which this crate does not
    /// parse) must be rejected outright rather than producing a runtime
    /// that goes on to fabricate tokenization.
    #[test]
    fn test_onnx_runtime_rejects_non_export_bytes() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let model_path = temp_dir.path().join("garbage.onnx");
        std::fs::write(&model_path, [0x08u8, 0x01, 0xFF, 0xFE, 0x00, 0x00])
            .expect("Operation failed in test");

        let result =
            OnnxTokenizerRuntime::from_file(model_path.to_str().expect("utf8 path").to_string());
        assert!(result.is_err());
    }

    /// A vocabulary whose ids are not exactly `0..len()` (no gaps, no
    /// duplicates) cannot be represented by a `vocab_tensor`'s implicit
    /// position-is-id convention, so exporting it must fail instead of
    /// silently mis-encoding it (the old code padded blindly by count and
    /// never checked this at all).
    #[test]
    fn test_create_vocab_tensor_rejects_non_contiguous_vocab() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 0u32);
        vocab.insert("z".to_string(), 5u32);
        let tokenizer = CharTokenizer::new(vocab);
        let exporter = OnnxTokenizerExporter::from_tokenizer(tokenizer);

        assert!(exporter.export().is_err());
    }

    /// Regression test for the fabricated `OnnxTokenizerRuntime::tokenize`:
    /// the old `simulate_vocab_lookup` derived every ID from
    /// `DefaultHasher(token) % 29000 + 1000`, never touching any real
    /// vocabulary. A real export/load/tokenize round trip must instead
    /// reproduce the tokenizer's exact real ids, which this hand-computed
    /// reference checks directly: every character of "hello" is a
    /// single-char entry in `create_test_char_tokenizer`'s vocabulary
    /// (h=4, e=5, l=6, o=7).
    #[test]
    fn test_onnx_export_round_trip_tokenizes_with_real_vocab() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let model_path = temp_dir.path().join("tokenizer.onnx.json");

        let tokenizer = create_test_char_tokenizer();
        OnnxTokenizerExporter::from_tokenizer(tokenizer)
            .save_to_file(model_path.to_str().expect("utf8 path"))
            .expect("Operation failed in test");

        let runtime =
            OnnxTokenizerRuntime::from_file(model_path.to_str().expect("utf8 path").to_string())
                .expect("a real exported model must load");

        let result = runtime.tokenize(&["hello".to_string()]).expect("Operation failed in test");

        assert_eq!(result[0].input_ids, vec![4, 5, 6, 6, 7]);
    }

    /// Different input texts must produce different real token IDs (the
    /// old hash-based fake also varied with input, but not with the real
    /// vocabulary -- this pins the exact real ids for a second text too).
    #[test]
    fn test_onnx_runtime_tokenize_output_varies_with_input() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let model_path = temp_dir.path().join("tokenizer.onnx.json");

        let tokenizer = create_test_char_tokenizer();
        OnnxTokenizerExporter::from_tokenizer(tokenizer)
            .save_to_file(model_path.to_str().expect("utf8 path"))
            .expect("Operation failed in test");

        let runtime =
            OnnxTokenizerRuntime::from_file(model_path.to_str().expect("utf8 path").to_string())
                .expect("Operation failed in test");

        let result = runtime
            .tokenize(&["hello".to_string(), "world".to_string()])
            .expect("Operation failed in test");

        assert_ne!(result[0].input_ids, result[1].input_ids);
        assert_eq!(result[1].input_ids, vec![8, 7, 9, 6, 10]); // w, o, r, l, d
    }

    /// Regression test: `get_metadata` used to hardcode
    /// `{"framework": "ONNX Runtime"}` and never report a real vocabulary
    /// size at all. It must now report the real, recovered vocabulary size
    /// (this fixture has exactly 14 entries), not
    /// `OnnxExportConfig::vocab_size`'s default of 50000.
    #[test]
    fn test_get_metadata_reports_real_vocab_size() {
        let temp_dir = tempdir().expect("Operation failed in test");
        let model_path = temp_dir.path().join("tokenizer.onnx.json");

        let tokenizer = create_test_char_tokenizer();
        OnnxTokenizerExporter::from_tokenizer(tokenizer)
            .save_to_file(model_path.to_str().expect("utf8 path"))
            .expect("Operation failed in test");

        let runtime =
            OnnxTokenizerRuntime::from_file(model_path.to_str().expect("utf8 path").to_string())
                .expect("Operation failed in test");

        let metadata = runtime.get_metadata().expect("Operation failed in test");
        assert_eq!(metadata.get("vocab_size"), Some(&"14".to_string()));
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
