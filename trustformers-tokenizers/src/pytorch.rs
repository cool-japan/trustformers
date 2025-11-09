//! PyTorch integration for TrustformeRS tokenizers
//!
//! This module provides direct integration with PyTorch tensors and models,
//! enabling seamless tokenization workflows within PyTorch pipelines.

use crate::{TokenizedInput, Tokenizer};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for PyTorch tokenizer integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyTorchConfig {
    /// Device to use for tensor operations (cpu, cuda:0, etc.)
    pub device: String,
    /// Data type for tensors (int32, int64)
    pub dtype: TensorDType,
    /// Maximum sequence length for padding/truncation
    pub max_length: Option<usize>,
    /// Padding strategy
    pub padding: PaddingStrategy,
    /// Truncation strategy
    pub truncation: TruncationStrategy,
    /// Return attention masks
    pub return_attention_mask: bool,
    /// Return token type IDs
    pub return_token_type_ids: bool,
    /// Batch size for processing
    pub batch_size: usize,
}

impl Default for PyTorchConfig {
    fn default() -> Self {
        Self {
            device: "cpu".to_string(),
            dtype: TensorDType::Int64,
            max_length: Some(512),
            padding: PaddingStrategy::LongestFirst,
            truncation: TruncationStrategy::LongestFirst,
            return_attention_mask: true,
            return_token_type_ids: false,
            batch_size: 32,
        }
    }
}

/// Tensor data types supported by PyTorch
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TensorDType {
    Int32,
    Int64,
    Float32,
    Float64,
}

/// Padding strategies for batch processing
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PaddingStrategy {
    /// Don't pad
    False,
    /// Pad to longest sequence in batch
    LongestFirst,
    /// Pad to maximum length
    MaxLength,
}

/// Truncation strategies for sequence processing
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TruncationStrategy {
    /// Don't truncate
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

/// PyTorch tensor representation
#[derive(Debug, Clone)]
pub struct PyTorchTensor {
    /// Tensor data as flattened vector
    pub data: Vec<i64>,
    /// Tensor shape [batch_size, sequence_length]
    pub shape: Vec<usize>,
    /// Device location
    pub device: String,
    /// Data type
    pub dtype: TensorDType,
}

impl PyTorchTensor {
    /// Create a new tensor
    pub fn new(data: Vec<i64>, shape: Vec<usize>, device: String, dtype: TensorDType) -> Self {
        Self {
            data,
            shape,
            device,
            dtype,
        }
    }

    /// Get tensor size
    pub fn size(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
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
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    /// Convert to device
    pub fn to_device(&self, device: &str) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            device: device.to_string(),
            dtype: self.dtype,
        }
    }

    /// Convert data type
    pub fn to_dtype(&self, dtype: TensorDType) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            device: self.device.clone(),
            dtype,
        }
    }
}

/// Batch of tokenized inputs formatted for PyTorch
#[derive(Debug, Clone)]
pub struct PyTorchBatch {
    /// Input token IDs tensor
    pub input_ids: PyTorchTensor,
    /// Attention mask tensor (optional)
    pub attention_mask: Option<PyTorchTensor>,
    /// Token type IDs tensor (optional)
    pub token_type_ids: Option<PyTorchTensor>,
    /// Special tokens mask (optional)
    pub special_tokens_mask: Option<PyTorchTensor>,
    /// Original sequence lengths before padding
    pub sequence_lengths: Vec<usize>,
}

impl PyTorchBatch {
    /// Create a new batch
    pub fn new(
        input_ids: PyTorchTensor,
        attention_mask: Option<PyTorchTensor>,
        token_type_ids: Option<PyTorchTensor>,
        special_tokens_mask: Option<PyTorchTensor>,
        sequence_lengths: Vec<usize>,
    ) -> Self {
        Self {
            input_ids,
            attention_mask,
            token_type_ids,
            special_tokens_mask,
            sequence_lengths,
        }
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.input_ids.shape[0]
    }

    /// Get sequence length
    pub fn sequence_length(&self) -> usize {
        self.input_ids.shape[1]
    }

    /// Move batch to device
    pub fn to_device(&self, device: &str) -> Self {
        Self {
            input_ids: self.input_ids.to_device(device),
            attention_mask: self.attention_mask.as_ref().map(|t| t.to_device(device)),
            token_type_ids: self.token_type_ids.as_ref().map(|t| t.to_device(device)),
            special_tokens_mask: self.special_tokens_mask.as_ref().map(|t| t.to_device(device)),
            sequence_lengths: self.sequence_lengths.clone(),
        }
    }

    /// Convert to specific data type
    pub fn to_dtype(&self, dtype: TensorDType) -> Self {
        Self {
            input_ids: self.input_ids.to_dtype(dtype),
            attention_mask: self.attention_mask.as_ref().map(|t| t.to_dtype(dtype)),
            token_type_ids: self.token_type_ids.as_ref().map(|t| t.to_dtype(dtype)),
            special_tokens_mask: self.special_tokens_mask.as_ref().map(|t| t.to_dtype(dtype)),
            sequence_lengths: self.sequence_lengths.clone(),
        }
    }

    /// Pin memory for faster GPU transfer
    pub fn pin_memory(&self) -> Self {
        // In a real implementation, this would use PyTorch's pin_memory functionality
        self.clone()
    }
}

/// PyTorch integration wrapper for tokenizers
pub struct PyTorchTokenizer<T: Tokenizer> {
    tokenizer: Arc<T>,
    config: PyTorchConfig,
}

impl<T: Tokenizer> PyTorchTokenizer<T> {
    /// Create a new PyTorch tokenizer wrapper
    pub fn new(tokenizer: T, config: PyTorchConfig) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
            config,
        }
    }

    /// Create with default configuration
    pub fn from_tokenizer(tokenizer: T) -> Self {
        Self::new(tokenizer, PyTorchConfig::default())
    }

    /// Update configuration
    pub fn with_config(mut self, config: PyTorchConfig) -> Self {
        self.config = config;
        self
    }

    /// Encode text to PyTorch tensors
    pub fn encode_to_tensors(&self, text: &str) -> Result<PyTorchBatch> {
        let tokenized = self.tokenizer.encode(text)?;
        self.convert_to_batch(vec![tokenized])
    }

    /// Encode text pair to PyTorch tensors
    pub fn encode_pair_to_tensors(&self, text_a: &str, text_b: &str) -> Result<PyTorchBatch> {
        let tokenized = self.tokenizer.encode_pair(text_a, text_b)?;
        self.convert_to_batch(vec![tokenized])
    }

    /// Encode batch of texts to PyTorch tensors
    pub fn encode_batch_to_tensors(&self, texts: &[String]) -> Result<PyTorchBatch> {
        let mut tokenized_batch = Vec::new();

        for text in texts {
            let tokenized = self.tokenizer.encode(text)?;
            tokenized_batch.push(tokenized);
        }

        self.convert_to_batch(tokenized_batch)
    }

    /// Encode batch of text pairs to PyTorch tensors
    pub fn encode_pair_batch_to_tensors(
        &self,
        text_pairs: &[(String, String)],
    ) -> Result<PyTorchBatch> {
        let mut tokenized_batch = Vec::new();

        for (text_a, text_b) in text_pairs {
            let tokenized = self.tokenizer.encode_pair(text_a, text_b)?;
            tokenized_batch.push(tokenized);
        }

        self.convert_to_batch(tokenized_batch)
    }

    /// Convert tokenized inputs to PyTorch batch
    fn convert_to_batch(&self, tokenized_inputs: Vec<TokenizedInput>) -> Result<PyTorchBatch> {
        if tokenized_inputs.is_empty() {
            return Err(anyhow!("Cannot create batch from empty input"));
        }

        let batch_size = tokenized_inputs.len();
        let sequence_lengths: Vec<usize> =
            tokenized_inputs.iter().map(|t| t.input_ids.len()).collect();

        // Determine sequence length based on padding strategy
        let seq_length = match self.config.padding {
            PaddingStrategy::False => {
                // No padding - all sequences must be same length
                let first_len = sequence_lengths[0];
                if !sequence_lengths.iter().all(|&len| len == first_len) {
                    return Err(anyhow!(
                        "All sequences must be same length when padding is disabled"
                    ));
                }
                first_len
            },
            PaddingStrategy::LongestFirst => sequence_lengths.iter().copied().max().unwrap_or(0),
            PaddingStrategy::MaxLength => self.config.max_length.unwrap_or(512),
        };

        // Apply truncation if needed
        let final_seq_length = if let Some(max_len) = self.config.max_length {
            match self.config.truncation {
                TruncationStrategy::False => seq_length,
                _ => seq_length.min(max_len),
            }
        } else {
            seq_length
        };

        // Create input_ids tensor
        let mut input_ids_data = Vec::with_capacity(batch_size * final_seq_length);
        let mut attention_mask_data = Vec::with_capacity(batch_size * final_seq_length);
        let mut token_type_ids_data = Vec::with_capacity(batch_size * final_seq_length);
        let mut special_tokens_mask_data = Vec::with_capacity(batch_size * final_seq_length);

        let pad_token_id = 0i64; // Typically 0 for most tokenizers

        for tokenized in &tokenized_inputs {
            // Handle input_ids
            let mut seq_input_ids = tokenized.input_ids.clone();

            // Truncate if necessary
            if seq_input_ids.len() > final_seq_length {
                seq_input_ids.truncate(final_seq_length);
            }

            // Pad if necessary
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

                // Truncate if necessary
                if seq_token_type_ids.len() > final_seq_length {
                    seq_token_type_ids.truncate(final_seq_length);
                }

                // Pad if necessary
                while seq_token_type_ids.len() < final_seq_length {
                    seq_token_type_ids.push(0);
                }

                token_type_ids_data.extend(seq_token_type_ids.into_iter().map(|id| id as i64));
            }

            // Create special tokens mask
            let special_tokens_mask = tokenized
                .special_tokens_mask
                .clone()
                .unwrap_or_else(|| vec![0; tokenized.input_ids.len()]);

            let mut seq_special_tokens_mask = special_tokens_mask;

            // Truncate if necessary
            if seq_special_tokens_mask.len() > final_seq_length {
                seq_special_tokens_mask.truncate(final_seq_length);
            }

            // Pad if necessary (special tokens in padding are marked as 0)
            while seq_special_tokens_mask.len() < final_seq_length {
                seq_special_tokens_mask.push(0);
            }

            special_tokens_mask_data
                .extend(seq_special_tokens_mask.into_iter().map(|mask| mask as i64));
        }

        // Create tensors
        let input_ids = PyTorchTensor::new(
            input_ids_data,
            vec![batch_size, final_seq_length],
            self.config.device.clone(),
            self.config.dtype,
        );

        let attention_mask = if self.config.return_attention_mask {
            Some(PyTorchTensor::new(
                attention_mask_data,
                vec![batch_size, final_seq_length],
                self.config.device.clone(),
                self.config.dtype,
            ))
        } else {
            None
        };

        let token_type_ids = if self.config.return_token_type_ids {
            Some(PyTorchTensor::new(
                token_type_ids_data,
                vec![batch_size, final_seq_length],
                self.config.device.clone(),
                self.config.dtype,
            ))
        } else {
            None
        };

        // Create special tokens mask tensor (only if any sequence has special tokens)
        let special_tokens_mask = if special_tokens_mask_data.iter().any(|&mask| mask != 0) {
            Some(PyTorchTensor::new(
                special_tokens_mask_data,
                vec![batch_size, final_seq_length],
                self.config.device.clone(),
                self.config.dtype,
            ))
        } else {
            None
        };

        Ok(PyTorchBatch::new(
            input_ids,
            attention_mask,
            token_type_ids,
            special_tokens_mask,
            sequence_lengths,
        ))
    }

    /// Get underlying tokenizer
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    /// Get configuration
    pub fn config(&self) -> &PyTorchConfig {
        &self.config
    }

    /// Set device
    pub fn set_device(&mut self, device: String) {
        self.config.device = device;
    }

    /// Set maximum length
    pub fn set_max_length(&mut self, max_length: Option<usize>) {
        self.config.max_length = max_length;
    }

    /// Set padding strategy
    pub fn set_padding(&mut self, padding: PaddingStrategy) {
        self.config.padding = padding;
    }

    /// Set truncation strategy
    pub fn set_truncation(&mut self, truncation: TruncationStrategy) {
        self.config.truncation = truncation;
    }
}

/// Dataset wrapper for PyTorch DataLoader compatibility
pub struct PyTorchDataset {
    texts: Vec<String>,
    #[allow(dead_code)]
    tokenizer_config: PyTorchConfig,
}

impl PyTorchDataset {
    /// Create a new dataset
    pub fn new(texts: Vec<String>, config: PyTorchConfig) -> Self {
        Self {
            texts,
            tokenizer_config: config,
        }
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

    /// Create batched iterator
    pub fn batch_iter(&self, batch_size: usize) -> BatchIterator<'_> {
        BatchIterator::new(&self.texts, batch_size)
    }
}

/// Iterator for batched dataset processing
pub struct BatchIterator<'a> {
    texts: &'a [String],
    batch_size: usize,
    current_index: usize,
}

impl<'a> BatchIterator<'a> {
    fn new(texts: &'a [String], batch_size: usize) -> Self {
        Self {
            texts,
            batch_size,
            current_index: 0,
        }
    }
}

impl<'a> Iterator for BatchIterator<'a> {
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

/// Utilities for PyTorch integration
pub struct PyTorchUtils;

impl PyTorchUtils {
    /// Convert tensor to numpy-like format for debugging
    pub fn tensor_to_debug_string(tensor: &PyTorchTensor) -> String {
        format!(
            "PyTorchTensor(shape={:?}, device={}, dtype={:?}, data={:?})",
            tensor.shape,
            tensor.device,
            tensor.dtype,
            &tensor.data[..tensor.data.len().min(10)] // Show first 10 elements
        )
    }

    /// Calculate tensor memory usage
    pub fn tensor_memory_usage(tensor: &PyTorchTensor) -> usize {
        let element_size = match tensor.dtype {
            TensorDType::Int32 | TensorDType::Float32 => 4,
            TensorDType::Int64 | TensorDType::Float64 => 8,
        };
        tensor.numel() * element_size
    }

    /// Create collate function for data loading
    pub fn collate_fn<T: Tokenizer>(
        tokenizer: &PyTorchTokenizer<T>,
        texts: Vec<String>,
    ) -> Result<PyTorchBatch> {
        tokenizer.encode_batch_to_tensors(&texts)
    }

    /// Validate tensor shapes for model input
    pub fn validate_model_inputs(batch: &PyTorchBatch) -> Result<()> {
        let batch_size = batch.batch_size();
        let seq_length = batch.sequence_length();

        // Validate input_ids
        if batch.input_ids.shape != vec![batch_size, seq_length] {
            return Err(anyhow!("Invalid input_ids shape"));
        }

        // Validate attention_mask if present
        if let Some(ref mask) = batch.attention_mask {
            if mask.shape != vec![batch_size, seq_length] {
                return Err(anyhow!("Invalid attention_mask shape"));
            }
        }

        // Validate token_type_ids if present
        if let Some(ref type_ids) = batch.token_type_ids {
            if type_ids.shape != vec![batch_size, seq_length] {
                return Err(anyhow!("Invalid token_type_ids shape"));
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
    fn test_pytorch_config() {
        let config = PyTorchConfig::default();
        assert_eq!(config.device, "cpu");
        assert_eq!(config.max_length, Some(512));
        assert!(config.return_attention_mask);
        assert!(!config.return_token_type_ids);
    }

    #[test]
    fn test_pytorch_tensor() {
        let data = vec![1, 2, 3, 4];
        let shape = vec![2, 2];
        let tensor = PyTorchTensor::new(
            data.clone(),
            shape.clone(),
            "cpu".to_string(),
            TensorDType::Int64,
        );

        assert_eq!(tensor.data, data);
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.device, "cpu");
        assert_eq!(tensor.numel(), 4);
    }

    #[test]
    fn test_tensor_reshape() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let tensor = PyTorchTensor::new(data, vec![2, 3], "cpu".to_string(), TensorDType::Int64);

        let reshaped = tensor.reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped.shape, vec![3, 2]);
        assert_eq!(reshaped.numel(), 6);
    }

    #[test]
    fn test_pytorch_tokenizer() {
        let tokenizer = create_test_char_tokenizer();
        let pytorch_tokenizer = PyTorchTokenizer::from_tokenizer(tokenizer);

        let batch = pytorch_tokenizer.encode_to_tensors("hello").unwrap();
        assert_eq!(batch.batch_size(), 1);
        assert!(batch.attention_mask.is_some());
    }

    #[test]
    fn test_batch_encoding() {
        let tokenizer = create_test_char_tokenizer();
        let pytorch_tokenizer = PyTorchTokenizer::from_tokenizer(tokenizer);

        let texts = vec!["hello".to_string(), "world".to_string()];
        let batch = pytorch_tokenizer.encode_batch_to_tensors(&texts).unwrap();

        assert_eq!(batch.batch_size(), 2);
        assert!(batch.attention_mask.is_some());
        assert_eq!(batch.sequence_lengths.len(), 2);
    }

    #[test]
    fn test_pytorch_dataset() {
        let texts = vec!["hello".to_string(), "world".to_string(), "test".to_string()];
        let config = PyTorchConfig::default();
        let dataset = PyTorchDataset::new(texts, config);

        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.get_item(0), Some("hello"));

        let batches: Vec<_> = dataset.batch_iter(2).collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 1);
    }

    #[test]
    fn test_tensor_utilities() {
        let tensor = PyTorchTensor::new(
            vec![1, 2, 3, 4],
            vec![2, 2],
            "cpu".to_string(),
            TensorDType::Int64,
        );

        let debug_str = PyTorchUtils::tensor_to_debug_string(&tensor);
        assert!(debug_str.contains("shape=[2, 2]"));
        assert!(debug_str.contains("device=cpu"));

        let memory_usage = PyTorchUtils::tensor_memory_usage(&tensor);
        assert_eq!(memory_usage, 4 * 8); // 4 elements * 8 bytes (Int64)
    }

    #[test]
    fn test_padding_strategies() {
        let tokenizer = create_test_char_tokenizer();
        let mut config = PyTorchConfig::default();
        config.padding = PaddingStrategy::MaxLength;
        config.max_length = Some(10);

        let pytorch_tokenizer = PyTorchTokenizer::new(tokenizer, config);

        let texts = vec!["hi".to_string(), "hello world".to_string()];
        let batch = pytorch_tokenizer.encode_batch_to_tensors(&texts).unwrap();

        assert_eq!(batch.sequence_length(), 10);
        assert_eq!(batch.batch_size(), 2);
    }
}
