//! TensorFlow integration for TrustformeRS tokenizers
//!
//! This module provides direct integration with TensorFlow tensors and models,
//! enabling seamless tokenization workflows within TensorFlow pipelines.

use crate::{TokenizedInput, Tokenizer};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for TensorFlow tokenizer integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorFlowConfig {
    /// Data type for tensors (int32, int64, float32, float64)
    pub dtype: TfDType,
    /// Maximum sequence length for padding/truncation
    pub max_length: Option<usize>,
    /// Padding strategy
    pub padding: TfPaddingStrategy,
    /// Truncation strategy
    pub truncation: TfTruncationStrategy,
    /// Return attention masks
    pub return_attention_mask: bool,
    /// Return token type IDs
    pub return_token_type_ids: bool,
    /// Return position IDs
    pub return_position_ids: bool,
    /// Batch size for processing
    pub batch_size: usize,
    /// Use ragged tensors for variable length sequences
    pub use_ragged_tensors: bool,
}

impl Default for TensorFlowConfig {
    fn default() -> Self {
        Self {
            dtype: TfDType::Int64,
            max_length: Some(512),
            padding: TfPaddingStrategy::LongestFirst,
            truncation: TfTruncationStrategy::LongestFirst,
            return_attention_mask: true,
            return_token_type_ids: false,
            return_position_ids: false,
            batch_size: 32,
            use_ragged_tensors: false,
        }
    }
}

/// TensorFlow data types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TfDType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    Bool,
    String,
}

impl TfDType {
    /// Get size in bytes for numeric types
    pub fn size_bytes(&self) -> usize {
        match self {
            TfDType::Int8 | TfDType::UInt8 | TfDType::Bool => 1,
            TfDType::Int16 | TfDType::UInt16 | TfDType::Float16 => 2,
            TfDType::Int32 | TfDType::UInt32 | TfDType::Float32 => 4,
            TfDType::Int64 | TfDType::UInt64 | TfDType::Float64 => 8,
            TfDType::String => 0, // Variable size
        }
    }

    /// Check if type is integer
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            TfDType::Int8
                | TfDType::Int16
                | TfDType::Int32
                | TfDType::Int64
                | TfDType::UInt8
                | TfDType::UInt16
                | TfDType::UInt32
                | TfDType::UInt64
        )
    }

    /// Check if type is floating point
    pub fn is_float(&self) -> bool {
        matches!(self, TfDType::Float16 | TfDType::Float32 | TfDType::Float64)
    }
}

/// Padding strategies for TensorFlow
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TfPaddingStrategy {
    /// No padding
    False,
    /// Pad to longest sequence in batch
    LongestFirst,
    /// Pad to maximum length
    MaxLength,
    /// Use ragged tensors (no padding)
    Ragged,
}

/// Truncation strategies for TensorFlow
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TfTruncationStrategy {
    /// No truncation
    False,
    /// Truncate longest sequences first
    LongestFirst,
    /// Truncate to maximum length
    MaxLength,
    /// Only truncate first sequence in pairs
    OnlyFirst,
    /// Only truncate second sequence in pairs
    OnlySecond,
}

/// TensorFlow tensor representation
#[derive(Debug, Clone)]
pub struct TensorFlowTensor {
    /// Tensor data as flattened vector
    pub data: Vec<i64>,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: TfDType,
    /// Tensor name (for TensorFlow graphs)
    pub name: Option<String>,
}

impl TensorFlowTensor {
    /// Create a new tensor
    pub fn new(data: Vec<i64>, shape: Vec<usize>, dtype: TfDType) -> Self {
        Self {
            data,
            shape,
            dtype,
            name: None,
        }
    }

    /// Create a named tensor
    pub fn new_named(data: Vec<i64>, shape: Vec<usize>, dtype: TfDType, name: String) -> Self {
        Self {
            data,
            shape,
            dtype,
            name: Some(name),
        }
    }

    /// Get tensor rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get tensor shape
    pub fn get_shape(&self) -> &[usize] {
        &self.shape
    }

    /// Reshape tensor
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.numel() {
            return Err(anyhow!("Cannot reshape tensor: size mismatch"));
        }

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            dtype: self.dtype,
            name: self.name.clone(),
        })
    }

    /// Transpose tensor (2D only)
    pub fn transpose(&self) -> Result<Self> {
        if self.rank() != 2 {
            return Err(anyhow!("Transpose only supported for 2D tensors"));
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut transposed_data = vec![0i64; self.numel()];

        for i in 0..rows {
            for j in 0..cols {
                transposed_data[j * rows + i] = self.data[i * cols + j];
            }
        }

        Ok(Self {
            data: transposed_data,
            shape: vec![cols, rows],
            dtype: self.dtype,
            name: self.name.clone(),
        })
    }

    /// Convert to different data type
    pub fn cast(&self, new_dtype: TfDType) -> Self {
        Self {
            data: self.data.clone(), // In real implementation, would convert data
            shape: self.shape.clone(),
            dtype: new_dtype,
            name: self.name.clone(),
        }
    }

    /// Set tensor name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
}

/// Ragged tensor for variable-length sequences
#[derive(Debug, Clone)]
pub struct RaggedTensor {
    /// Flat values
    pub values: Vec<i64>,
    /// Row splits indicating where each sequence starts/ends
    pub row_splits: Vec<usize>,
    /// Data type
    pub dtype: TfDType,
    /// Tensor name
    pub name: Option<String>,
}

impl RaggedTensor {
    /// Create a new ragged tensor
    pub fn new(values: Vec<i64>, row_splits: Vec<usize>, dtype: TfDType) -> Self {
        Self {
            values,
            row_splits,
            dtype,
            name: None,
        }
    }

    /// Get number of sequences
    pub fn nrows(&self) -> usize {
        if self.row_splits.len() < 2 {
            0
        } else {
            self.row_splits.len() - 1
        }
    }

    /// Get sequence at index
    pub fn get_sequence(&self, index: usize) -> Option<&[i64]> {
        if index >= self.nrows() {
            return None;
        }

        let start = self.row_splits[index];
        let end = self.row_splits[index + 1];
        Some(&self.values[start..end])
    }

    /// Convert to dense tensor with padding
    pub fn to_dense(&self, max_length: Option<usize>, pad_value: i64) -> TensorFlowTensor {
        let nrows = self.nrows();
        let max_len = max_length.unwrap_or_else(|| {
            (0..nrows)
                .map(|i| self.row_splits[i + 1] - self.row_splits[i])
                .max()
                .unwrap_or(0)
        });

        let mut dense_data = vec![pad_value; nrows * max_len];

        for i in 0..nrows {
            let start = self.row_splits[i];
            let end = self.row_splits[i + 1];
            let seq_len = (end - start).min(max_len);

            for j in 0..seq_len {
                dense_data[i * max_len + j] = self.values[start + j];
            }
        }

        TensorFlowTensor::new(dense_data, vec![nrows, max_len], self.dtype)
    }

    /// Set tensor name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
}

/// Batch of tokenized inputs formatted for TensorFlow
#[derive(Debug, Clone)]
pub struct TensorFlowBatch {
    /// Input token IDs
    pub input_ids: TensorOrRagged,
    /// Attention mask (optional)
    pub attention_mask: Option<TensorFlowTensor>,
    /// Token type IDs (optional)
    pub token_type_ids: Option<TensorFlowTensor>,
    /// Position IDs (optional)
    pub position_ids: Option<TensorFlowTensor>,
    /// Special tokens mask (optional)
    pub special_tokens_mask: Option<TensorOrRagged>,
    /// Original sequence lengths
    pub sequence_lengths: Vec<usize>,
}

/// Either a regular tensor or ragged tensor
#[derive(Debug, Clone)]
pub enum TensorOrRagged {
    Tensor(TensorFlowTensor),
    Ragged(RaggedTensor),
}

impl TensorOrRagged {
    /// Get batch size
    pub fn batch_size(&self) -> usize {
        match self {
            TensorOrRagged::Tensor(t) => t.shape[0],
            TensorOrRagged::Ragged(r) => r.nrows(),
        }
    }

    /// Convert to dense tensor if ragged
    pub fn to_dense(&self, max_length: Option<usize>, pad_value: i64) -> TensorFlowTensor {
        match self {
            TensorOrRagged::Tensor(t) => t.clone(),
            TensorOrRagged::Ragged(r) => r.to_dense(max_length, pad_value),
        }
    }
}

impl TensorFlowBatch {
    /// Create a new batch
    pub fn new(
        input_ids: TensorOrRagged,
        attention_mask: Option<TensorFlowTensor>,
        token_type_ids: Option<TensorFlowTensor>,
        position_ids: Option<TensorFlowTensor>,
        special_tokens_mask: Option<TensorOrRagged>,
        sequence_lengths: Vec<usize>,
    ) -> Self {
        Self {
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            special_tokens_mask,
            sequence_lengths,
        }
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.input_ids.batch_size()
    }

    /// Get sequence length (for dense tensors)
    pub fn sequence_length(&self) -> Option<usize> {
        match &self.input_ids {
            TensorOrRagged::Tensor(t) => Some(t.shape[1]),
            TensorOrRagged::Ragged(_) => None, // Variable length
        }
    }

    /// Convert ragged tensors to dense
    pub fn to_dense(&self, max_length: Option<usize>, pad_value: i64) -> Self {
        Self {
            input_ids: TensorOrRagged::Tensor(self.input_ids.to_dense(max_length, pad_value)),
            attention_mask: self.attention_mask.clone(),
            token_type_ids: self.token_type_ids.clone(),
            position_ids: self.position_ids.clone(),
            special_tokens_mask: self.special_tokens_mask.clone(),
            sequence_lengths: self.sequence_lengths.clone(),
        }
    }
}

/// TensorFlow integration wrapper for tokenizers
pub struct TensorFlowTokenizer<T: Tokenizer> {
    tokenizer: Arc<T>,
    config: TensorFlowConfig,
}

impl<T: Tokenizer> TensorFlowTokenizer<T> {
    /// Create a new TensorFlow tokenizer wrapper
    pub fn new(tokenizer: T, config: TensorFlowConfig) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
            config,
        }
    }

    /// Create with default configuration
    pub fn from_tokenizer(tokenizer: T) -> Self {
        Self::new(tokenizer, TensorFlowConfig::default())
    }

    /// Update configuration
    pub fn with_config(mut self, config: TensorFlowConfig) -> Self {
        self.config = config;
        self
    }

    /// Encode text to TensorFlow tensors
    pub fn encode_to_tensors(&self, text: &str) -> Result<TensorFlowBatch> {
        let tokenized = self.tokenizer.encode(text)?;
        self.convert_to_batch(vec![tokenized])
    }

    /// Encode text pair to TensorFlow tensors
    pub fn encode_pair_to_tensors(&self, text_a: &str, text_b: &str) -> Result<TensorFlowBatch> {
        let tokenized = self.tokenizer.encode_pair(text_a, text_b)?;
        self.convert_to_batch(vec![tokenized])
    }

    /// Encode batch of texts to TensorFlow tensors
    pub fn encode_batch_to_tensors(&self, texts: &[String]) -> Result<TensorFlowBatch> {
        let mut tokenized_batch = Vec::new();

        for text in texts {
            let tokenized = self.tokenizer.encode(text)?;
            tokenized_batch.push(tokenized);
        }

        self.convert_to_batch(tokenized_batch)
    }

    /// Encode batch of text pairs to TensorFlow tensors
    pub fn encode_pair_batch_to_tensors(
        &self,
        text_pairs: &[(String, String)],
    ) -> Result<TensorFlowBatch> {
        let mut tokenized_batch = Vec::new();

        for (text_a, text_b) in text_pairs {
            let tokenized = self.tokenizer.encode_pair(text_a, text_b)?;
            tokenized_batch.push(tokenized);
        }

        self.convert_to_batch(tokenized_batch)
    }

    /// Convert tokenized inputs to TensorFlow batch
    fn convert_to_batch(&self, tokenized_inputs: Vec<TokenizedInput>) -> Result<TensorFlowBatch> {
        if tokenized_inputs.is_empty() {
            return Err(anyhow!("Cannot create batch from empty input"));
        }

        let _batch_size = tokenized_inputs.len();
        let sequence_lengths: Vec<usize> =
            tokenized_inputs.iter().map(|t| t.input_ids.len()).collect();

        // Handle ragged tensors
        if self.config.use_ragged_tensors
            || matches!(self.config.padding, TfPaddingStrategy::Ragged)
        {
            return self.create_ragged_batch(tokenized_inputs, sequence_lengths);
        }

        // Handle dense tensors with padding
        self.create_dense_batch(tokenized_inputs, sequence_lengths)
    }

    /// Create ragged tensor batch
    fn create_ragged_batch(
        &self,
        tokenized_inputs: Vec<TokenizedInput>,
        sequence_lengths: Vec<usize>,
    ) -> Result<TensorFlowBatch> {
        let mut values = Vec::new();
        let mut row_splits = vec![0];
        let mut special_tokens_values = Vec::new();
        let mut special_tokens_row_splits = vec![0];
        let mut has_special_tokens = false;

        for tokenized in &tokenized_inputs {
            values.extend(tokenized.input_ids.iter().map(|&id| id as i64));
            row_splits.push(values.len());

            // Process special tokens mask for ragged tensors
            if let Some(mask) = &tokenized.special_tokens_mask {
                special_tokens_values.extend(mask.iter().map(|&m| m as i64));
                has_special_tokens = has_special_tokens || mask.iter().any(|&m| m != 0);
            } else {
                special_tokens_values.extend(vec![0; tokenized.input_ids.len()]);
            }
            special_tokens_row_splits.push(special_tokens_values.len());
        }

        let input_ids = TensorOrRagged::Ragged(
            RaggedTensor::new(values, row_splits, self.config.dtype)
                .with_name("input_ids".to_string()),
        );

        let special_tokens_mask = if has_special_tokens {
            Some(TensorOrRagged::Ragged(
                RaggedTensor::new(
                    special_tokens_values,
                    special_tokens_row_splits,
                    self.config.dtype,
                )
                .with_name("special_tokens_mask".to_string()),
            ))
        } else {
            None
        };

        Ok(TensorFlowBatch::new(
            input_ids,
            None, // Attention mask not applicable for ragged tensors
            None, // Token type IDs not implemented for ragged
            None, // Position IDs not implemented for ragged
            special_tokens_mask,
            sequence_lengths,
        ))
    }

    /// Create dense tensor batch with padding
    fn create_dense_batch(
        &self,
        tokenized_inputs: Vec<TokenizedInput>,
        sequence_lengths: Vec<usize>,
    ) -> Result<TensorFlowBatch> {
        let batch_size = tokenized_inputs.len();

        // Determine sequence length
        let seq_length = match self.config.padding {
            TfPaddingStrategy::False => {
                let first_len = sequence_lengths[0];
                if !sequence_lengths.iter().all(|&len| len == first_len) {
                    return Err(anyhow!(
                        "All sequences must be same length when padding is disabled"
                    ));
                }
                first_len
            },
            TfPaddingStrategy::LongestFirst => sequence_lengths.iter().copied().max().unwrap_or(0),
            TfPaddingStrategy::MaxLength => self.config.max_length.unwrap_or(512),
            TfPaddingStrategy::Ragged => unreachable!(), // Handled above
        };

        // Apply truncation
        let final_seq_length = if let Some(max_len) = self.config.max_length {
            match self.config.truncation {
                TfTruncationStrategy::False => seq_length,
                _ => seq_length.min(max_len),
            }
        } else {
            seq_length
        };

        // Create tensors
        let mut input_ids_data = Vec::with_capacity(batch_size * final_seq_length);
        let mut attention_mask_data = Vec::with_capacity(batch_size * final_seq_length);
        let mut token_type_ids_data = Vec::with_capacity(batch_size * final_seq_length);
        let mut position_ids_data = Vec::with_capacity(batch_size * final_seq_length);
        let mut special_tokens_mask_data = Vec::with_capacity(batch_size * final_seq_length);

        let pad_token_id = 0i64;

        for tokenized in &tokenized_inputs {
            // Handle input_ids
            let mut seq_input_ids = tokenized.input_ids.clone();

            if seq_input_ids.len() > final_seq_length {
                seq_input_ids.truncate(final_seq_length);
            }

            while seq_input_ids.len() < final_seq_length {
                seq_input_ids.push(pad_token_id as u32);
            }

            input_ids_data.extend(seq_input_ids.into_iter().map(|id| id as i64));

            // Create attention mask
            if self.config.return_attention_mask {
                let actual_length = tokenized.input_ids.len().min(final_seq_length);
                for i in 0..final_seq_length {
                    attention_mask_data.push(if i < actual_length { 1 } else { 0 });
                }
            }

            // Create token type IDs
            if self.config.return_token_type_ids {
                let token_type_ids = tokenized
                    .token_type_ids
                    .clone()
                    .unwrap_or_else(|| vec![0; tokenized.input_ids.len()]);

                let mut seq_token_type_ids = token_type_ids;

                if seq_token_type_ids.len() > final_seq_length {
                    seq_token_type_ids.truncate(final_seq_length);
                }

                while seq_token_type_ids.len() < final_seq_length {
                    seq_token_type_ids.push(0);
                }

                token_type_ids_data.extend(seq_token_type_ids.into_iter().map(|id| id as i64));
            }

            // Create position IDs
            if self.config.return_position_ids {
                for i in 0..final_seq_length {
                    position_ids_data.push(i as i64);
                }
            }

            // Create special tokens mask
            let special_tokens_mask = tokenized
                .special_tokens_mask
                .clone()
                .unwrap_or_else(|| vec![0; tokenized.input_ids.len()]);

            let mut seq_special_tokens_mask = special_tokens_mask;

            if seq_special_tokens_mask.len() > final_seq_length {
                seq_special_tokens_mask.truncate(final_seq_length);
            }

            while seq_special_tokens_mask.len() < final_seq_length {
                seq_special_tokens_mask.push(0);
            }

            special_tokens_mask_data
                .extend(seq_special_tokens_mask.into_iter().map(|mask| mask as i64));
        }

        // Create tensors
        let input_ids = TensorOrRagged::Tensor(
            TensorFlowTensor::new(
                input_ids_data,
                vec![batch_size, final_seq_length],
                self.config.dtype,
            )
            .with_name("input_ids".to_string()),
        );

        let attention_mask = if self.config.return_attention_mask {
            Some(
                TensorFlowTensor::new(
                    attention_mask_data,
                    vec![batch_size, final_seq_length],
                    self.config.dtype,
                )
                .with_name("attention_mask".to_string()),
            )
        } else {
            None
        };

        let token_type_ids = if self.config.return_token_type_ids {
            Some(
                TensorFlowTensor::new(
                    token_type_ids_data,
                    vec![batch_size, final_seq_length],
                    self.config.dtype,
                )
                .with_name("token_type_ids".to_string()),
            )
        } else {
            None
        };

        let position_ids = if self.config.return_position_ids {
            Some(
                TensorFlowTensor::new(
                    position_ids_data,
                    vec![batch_size, final_seq_length],
                    self.config.dtype,
                )
                .with_name("position_ids".to_string()),
            )
        } else {
            None
        };

        // Create special tokens mask tensor (only if any sequence has special tokens)
        let special_tokens_mask = if special_tokens_mask_data.iter().any(|&mask| mask != 0) {
            Some(
                TensorFlowTensor::new(
                    special_tokens_mask_data,
                    vec![batch_size, final_seq_length],
                    self.config.dtype,
                )
                .with_name("special_tokens_mask".to_string()),
            )
        } else {
            None
        };

        Ok(TensorFlowBatch::new(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            special_tokens_mask.map(TensorOrRagged::Tensor),
            sequence_lengths,
        ))
    }

    /// Get underlying tokenizer
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    /// Get configuration
    pub fn config(&self) -> &TensorFlowConfig {
        &self.config
    }
}

/// TensorFlow dataset wrapper
pub struct TensorFlowDataset {
    texts: Vec<String>,
    config: TensorFlowConfig,
}

impl TensorFlowDataset {
    /// Create a new dataset
    pub fn new(texts: Vec<String>, config: TensorFlowConfig) -> Self {
        Self { texts, config }
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.texts.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    /// Get sample at index
    pub fn get_item(&self, index: usize) -> Option<&str> {
        self.texts.get(index).map(|s| s.as_str())
    }

    /// Create tf.data.Dataset equivalent iterator
    pub fn tf_data_iter(&self, batch_size: usize) -> TfDataIterator<'_> {
        TfDataIterator::new(&self.texts, batch_size, self.config.clone())
    }
}

/// Iterator for TensorFlow tf.data.Dataset compatibility
pub struct TfDataIterator<'a> {
    texts: &'a [String],
    batch_size: usize,
    current_index: usize,
    #[allow(dead_code)]
    config: TensorFlowConfig,
}

impl<'a> TfDataIterator<'a> {
    fn new(texts: &'a [String], batch_size: usize, config: TensorFlowConfig) -> Self {
        Self {
            texts,
            batch_size,
            current_index: 0,
            config,
        }
    }

    /// Apply mapping function (similar to tf.data.Dataset.map)
    pub fn map<F>(self, _func: F) -> Self
    where
        F: Fn(&str) -> String,
    {
        // In a real implementation, would apply the function
        self
    }

    /// Repeat dataset
    pub fn repeat(self, _count: Option<usize>) -> Self {
        // In a real implementation, would repeat the dataset
        self
    }

    /// Shuffle dataset
    pub fn shuffle(self, _buffer_size: usize) -> Self {
        // In a real implementation, would shuffle the dataset
        self
    }
}

impl<'a> Iterator for TfDataIterator<'a> {
    type Item = &'a [String];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.texts.len() {
            return None;
        }

        let end_index = (self.current_index + self.batch_size).min(self.texts.len());
        let batch = &self.texts[self.current_index..end_index];
        self.current_index = end_index;

        Some(batch)
    }
}

/// Utilities for TensorFlow integration
pub struct TensorFlowUtils;

impl TensorFlowUtils {
    /// Convert tensor to TensorFlow SavedModel format description
    pub fn tensor_to_signature_def(tensor: &TensorFlowTensor) -> HashMap<String, String> {
        let mut signature = HashMap::new();

        signature.insert("dtype".to_string(), format!("{:?}", tensor.dtype));
        signature.insert("shape".to_string(), format!("{:?}", tensor.shape));

        if let Some(ref name) = tensor.name {
            signature.insert("name".to_string(), name.clone());
        }

        signature
    }

    /// Calculate tensor memory usage
    pub fn tensor_memory_usage(tensor: &TensorFlowTensor) -> usize {
        tensor.numel() * tensor.dtype.size_bytes()
    }

    /// Create TensorFlow serving input signature
    pub fn create_serving_signature(
        batch: &TensorFlowBatch,
    ) -> HashMap<String, HashMap<String, String>> {
        let mut inputs = HashMap::new();

        match &batch.input_ids {
            TensorOrRagged::Tensor(t) => {
                inputs.insert("input_ids".to_string(), Self::tensor_to_signature_def(t));
            },
            TensorOrRagged::Ragged(_) => {
                let mut ragged_sig = HashMap::new();
                ragged_sig.insert("type".to_string(), "RaggedTensor".to_string());
                inputs.insert("input_ids".to_string(), ragged_sig);
            },
        }

        if let Some(ref mask) = batch.attention_mask {
            inputs.insert(
                "attention_mask".to_string(),
                Self::tensor_to_signature_def(mask),
            );
        }

        if let Some(ref type_ids) = batch.token_type_ids {
            inputs.insert(
                "token_type_ids".to_string(),
                Self::tensor_to_signature_def(type_ids),
            );
        }

        inputs
    }

    /// Export batch to TensorFlow SavedModel format (conceptual)
    pub fn export_to_saved_model_format(batch: &TensorFlowBatch) -> Result<String> {
        // In a real implementation, this would create actual TensorFlow SavedModel files
        let signature = Self::create_serving_signature(batch);
        serde_json::to_string_pretty(&signature)
            .map_err(|e| anyhow!("Failed to serialize signature: {}", e))
    }

    /// Validate TensorFlow model inputs
    pub fn validate_model_inputs(batch: &TensorFlowBatch) -> Result<()> {
        let batch_size = batch.batch_size();

        // Validate attention_mask if present
        if let Some(ref mask) = batch.attention_mask {
            if mask.shape[0] != batch_size {
                return Err(anyhow!("Attention mask batch size mismatch"));
            }
        }

        // Validate token_type_ids if present
        if let Some(ref type_ids) = batch.token_type_ids {
            if type_ids.shape[0] != batch_size {
                return Err(anyhow!("Token type IDs batch size mismatch"));
            }
        }

        Ok(())
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
    fn test_tensorflow_config() {
        let config = TensorFlowConfig::default();
        assert_eq!(config.dtype, TfDType::Int64);
        assert_eq!(config.max_length, Some(512));
        assert!(config.return_attention_mask);
        assert!(!config.return_token_type_ids);
    }

    #[test]
    fn test_tensorflow_tensor() {
        let data = vec![1, 2, 3, 4];
        let shape = vec![2, 2];
        let tensor = TensorFlowTensor::new(data.clone(), shape.clone(), TfDType::Int64);

        assert_eq!(tensor.data, data);
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.numel(), 4);
    }

    #[test]
    fn test_tensor_reshape() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let tensor = TensorFlowTensor::new(data, vec![2, 3], TfDType::Int64);

        let reshaped = tensor.reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped.shape, vec![3, 2]);
        assert_eq!(reshaped.numel(), 6);
    }

    #[test]
    fn test_ragged_tensor() {
        let values = vec![1, 2, 3, 4, 5];
        let row_splits = vec![0, 2, 5];
        let ragged = RaggedTensor::new(values, row_splits, TfDType::Int64);

        assert_eq!(ragged.nrows(), 2);
        assert_eq!(ragged.get_sequence(0), Some([1, 2].as_slice()));
        assert_eq!(ragged.get_sequence(1), Some([3, 4, 5].as_slice()));
    }

    #[test]
    fn test_ragged_to_dense() {
        let values = vec![1, 2, 3, 4, 5];
        let row_splits = vec![0, 2, 5];
        let ragged = RaggedTensor::new(values, row_splits, TfDType::Int64);

        let dense = ragged.to_dense(Some(4), 0);
        assert_eq!(dense.shape, vec![2, 4]);
        assert_eq!(dense.data, vec![1, 2, 0, 0, 3, 4, 5, 0]);
    }

    #[test]
    fn test_tensorflow_tokenizer() {
        let tokenizer = create_test_char_tokenizer();
        let tf_tokenizer = TensorFlowTokenizer::from_tokenizer(tokenizer);

        let batch = tf_tokenizer.encode_to_tensors("hello").unwrap();
        assert_eq!(batch.batch_size(), 1);
        assert!(batch.attention_mask.is_some());
    }

    #[test]
    fn test_batch_encoding() {
        let tokenizer = create_test_char_tokenizer();
        let tf_tokenizer = TensorFlowTokenizer::from_tokenizer(tokenizer);

        let texts = vec!["hello".to_string(), "world".to_string()];
        let batch = tf_tokenizer.encode_batch_to_tensors(&texts).unwrap();

        assert_eq!(batch.batch_size(), 2);
        assert!(batch.attention_mask.is_some());
        assert_eq!(batch.sequence_lengths.len(), 2);
    }

    #[test]
    fn test_ragged_tensor_batch() {
        let tokenizer = create_test_char_tokenizer();
        let mut config = TensorFlowConfig::default();
        config.use_ragged_tensors = true;

        let tf_tokenizer = TensorFlowTokenizer::new(tokenizer, config);

        let texts = vec!["hi".to_string(), "hello world".to_string()];
        let batch = tf_tokenizer.encode_batch_to_tensors(&texts).unwrap();

        assert_eq!(batch.batch_size(), 2);
        assert!(matches!(batch.input_ids, TensorOrRagged::Ragged(_)));
    }

    #[test]
    fn test_tensorflow_dataset() {
        let texts = vec!["hello".to_string(), "world".to_string(), "test".to_string()];
        let config = TensorFlowConfig::default();
        let dataset = TensorFlowDataset::new(texts, config);

        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.get_item(0), Some("hello"));

        let batches: Vec<_> = dataset.tf_data_iter(2).collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 1);
    }

    #[test]
    fn test_tensorflow_utils() {
        let tensor = TensorFlowTensor::new(vec![1, 2, 3, 4], vec![2, 2], TfDType::Int64);

        let signature = TensorFlowUtils::tensor_to_signature_def(&tensor);
        assert!(signature.contains_key("dtype"));
        assert!(signature.contains_key("shape"));

        let memory_usage = TensorFlowUtils::tensor_memory_usage(&tensor);
        assert_eq!(memory_usage, 4 * 8); // 4 elements * 8 bytes (Int64)
    }
}
