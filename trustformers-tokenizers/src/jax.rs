//! JAX integration for TrustformeRS tokenizers
//!
//! This module provides direct integration with JAX arrays and functions,
//! enabling seamless tokenization workflows within JAX/Flax pipelines.

use crate::{TokenizedInput, Tokenizer};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for JAX tokenizer integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaxConfig {
    /// Data type for arrays
    pub dtype: JaxDType,
    /// Maximum sequence length for padding/truncation
    pub max_length: Option<usize>,
    /// Padding strategy
    pub padding: JaxPaddingStrategy,
    /// Truncation strategy
    pub truncation: JaxTruncationStrategy,
    /// Return attention masks
    pub return_attention_mask: bool,
    /// Return token type IDs
    pub return_token_type_ids: bool,
    /// Return position IDs
    pub return_position_ids: bool,
    /// Batch size for processing
    pub batch_size: usize,
    /// Device placement (cpu, gpu:0, tpu:0, etc.)
    pub device: JaxDevice,
    /// Use XLA compilation
    pub use_xla: bool,
}

impl Default for JaxConfig {
    fn default() -> Self {
        Self {
            dtype: JaxDType::Int32,
            max_length: Some(512),
            padding: JaxPaddingStrategy::LongestFirst,
            truncation: JaxTruncationStrategy::LongestFirst,
            return_attention_mask: true,
            return_token_type_ids: false,
            return_position_ids: false,
            batch_size: 32,
            device: JaxDevice::Cpu,
            use_xla: true,
        }
    }
}

/// JAX data types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum JaxDType {
    Bool,
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
    Complex64,
    Complex128,
}

impl JaxDType {
    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            JaxDType::Bool | JaxDType::Int8 | JaxDType::UInt8 => 1,
            JaxDType::Int16 | JaxDType::UInt16 | JaxDType::Float16 => 2,
            JaxDType::Int32 | JaxDType::UInt32 | JaxDType::Float32 => 4,
            JaxDType::Int64 | JaxDType::UInt64 | JaxDType::Float64 | JaxDType::Complex64 => 8,
            JaxDType::Complex128 => 16,
        }
    }

    /// Check if type is integer
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            JaxDType::Int8
                | JaxDType::Int16
                | JaxDType::Int32
                | JaxDType::Int64
                | JaxDType::UInt8
                | JaxDType::UInt16
                | JaxDType::UInt32
                | JaxDType::UInt64
        )
    }

    /// Check if type is floating point
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            JaxDType::Float16 | JaxDType::Float32 | JaxDType::Float64
        )
    }

    /// Check if type is complex
    pub fn is_complex(&self) -> bool {
        matches!(self, JaxDType::Complex64 | JaxDType::Complex128)
    }
}

/// JAX device specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JaxDevice {
    Cpu,
    Gpu(u32),
    Tpu(u32),
    Custom(String),
}

impl std::fmt::Display for JaxDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JaxDevice::Cpu => write!(f, "cpu"),
            JaxDevice::Gpu(id) => write!(f, "gpu:{}", id),
            JaxDevice::Tpu(id) => write!(f, "tpu:{}", id),
            JaxDevice::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Padding strategies for JAX
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum JaxPaddingStrategy {
    /// No padding
    False,
    /// Pad to longest sequence in batch
    LongestFirst,
    /// Pad to maximum length
    MaxLength,
    /// Dynamic padding with reshaping
    Dynamic,
}

/// Truncation strategies for JAX
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum JaxTruncationStrategy {
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

/// JAX array representation
#[derive(Debug, Clone)]
pub struct JaxArray {
    /// Array data as flattened vector
    pub data: Vec<i32>,
    /// Array shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: JaxDType,
    /// Device placement
    pub device: JaxDevice,
    /// Array name (for debugging)
    pub name: Option<String>,
    /// Whether the array is sharded
    pub is_sharded: bool,
    /// Sharding specification
    pub sharding: Option<JaxSharding>,
}

/// JAX sharding specification for distributed computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaxSharding {
    /// Mesh specification
    pub mesh: JaxMesh,
    /// Partition specification
    pub partition_spec: Vec<Option<String>>,
}

/// JAX device mesh for distributed computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaxMesh {
    /// Device array
    pub devices: Vec<JaxDevice>,
    /// Mesh shape
    pub shape: Vec<usize>,
    /// Axis names
    pub axis_names: Vec<String>,
}

impl JaxArray {
    /// Create a new JAX array
    pub fn new(data: Vec<i32>, shape: Vec<usize>, dtype: JaxDType, device: JaxDevice) -> Self {
        Self {
            data,
            shape,
            dtype,
            device,
            name: None,
            is_sharded: false,
            sharding: None,
        }
    }

    /// Create a named array
    pub fn new_named(
        data: Vec<i32>,
        shape: Vec<usize>,
        dtype: JaxDType,
        device: JaxDevice,
        name: String,
    ) -> Self {
        Self {
            data,
            shape,
            dtype,
            device,
            name: Some(name),
            is_sharded: false,
            sharding: None,
        }
    }

    /// Get array rank (number of dimensions)
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get number of elements
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get array shape
    pub fn get_shape(&self) -> &[usize] {
        &self.shape
    }

    /// Reshape array
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.size() {
            return Err(anyhow!("Cannot reshape array: size mismatch"));
        }

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            dtype: self.dtype,
            device: self.device.clone(),
            name: self.name.clone(),
            is_sharded: self.is_sharded,
            sharding: self.sharding.clone(),
        })
    }

    /// Transpose array (2D only)
    pub fn transpose(&self) -> Result<Self> {
        if self.ndim() != 2 {
            return Err(anyhow!("Transpose only supported for 2D arrays"));
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut transposed_data = vec![0i32; self.size()];

        for i in 0..rows {
            for j in 0..cols {
                transposed_data[j * rows + i] = self.data[i * cols + j];
            }
        }

        Ok(Self {
            data: transposed_data,
            shape: vec![cols, rows],
            dtype: self.dtype,
            device: self.device.clone(),
            name: self.name.clone(),
            is_sharded: self.is_sharded,
            sharding: self.sharding.clone(),
        })
    }

    /// Convert to different data type
    pub fn astype(&self, new_dtype: JaxDType) -> Self {
        Self {
            data: self.data.clone(), // In real implementation, would convert data
            shape: self.shape.clone(),
            dtype: new_dtype,
            device: self.device.clone(),
            name: self.name.clone(),
            is_sharded: self.is_sharded,
            sharding: self.sharding.clone(),
        }
    }

    /// Move array to different device
    pub fn to_device(&self, device: JaxDevice) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device,
            name: self.name.clone(),
            is_sharded: self.is_sharded,
            sharding: self.sharding.clone(),
        }
    }

    /// Shard array across devices
    pub fn shard(&self, sharding: JaxSharding) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
            name: self.name.clone(),
            is_sharded: true,
            sharding: Some(sharding),
        }
    }

    /// Set array name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Block until array computation is complete
    pub fn block_until_ready(&self) -> Self {
        // In real implementation, would synchronize with JAX computation
        self.clone()
    }
}

/// Batch of tokenized inputs formatted for JAX
#[derive(Debug, Clone)]
pub struct JaxBatch {
    /// Input token IDs array
    pub input_ids: JaxArray,
    /// Attention mask array (optional)
    pub attention_mask: Option<JaxArray>,
    /// Token type IDs array (optional)
    pub token_type_ids: Option<JaxArray>,
    /// Position IDs array (optional)
    pub position_ids: Option<JaxArray>,
    /// Special tokens mask (optional)
    pub special_tokens_mask: Option<JaxArray>,
    /// Original sequence lengths
    pub sequence_lengths: Vec<usize>,
}

impl JaxBatch {
    /// Create a new batch
    pub fn new(
        input_ids: JaxArray,
        attention_mask: Option<JaxArray>,
        token_type_ids: Option<JaxArray>,
        position_ids: Option<JaxArray>,
        special_tokens_mask: Option<JaxArray>,
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
        self.input_ids.shape[0]
    }

    /// Get sequence length
    pub fn sequence_length(&self) -> usize {
        self.input_ids.shape[1]
    }

    /// Move batch to device
    pub fn to_device(&self, device: JaxDevice) -> Self {
        Self {
            input_ids: self.input_ids.to_device(device.clone()),
            attention_mask: self.attention_mask.as_ref().map(|a| a.to_device(device.clone())),
            token_type_ids: self.token_type_ids.as_ref().map(|a| a.to_device(device.clone())),
            position_ids: self.position_ids.as_ref().map(|a| a.to_device(device.clone())),
            special_tokens_mask: self.special_tokens_mask.as_ref().map(|a| a.to_device(device)),
            sequence_lengths: self.sequence_lengths.clone(),
        }
    }

    /// Convert to different data type
    pub fn astype(&self, dtype: JaxDType) -> Self {
        Self {
            input_ids: self.input_ids.astype(dtype),
            attention_mask: self.attention_mask.as_ref().map(|a| a.astype(dtype)),
            token_type_ids: self.token_type_ids.as_ref().map(|a| a.astype(dtype)),
            position_ids: self.position_ids.as_ref().map(|a| a.astype(dtype)),
            special_tokens_mask: self.special_tokens_mask.as_ref().map(|a| a.astype(dtype)),
            sequence_lengths: self.sequence_lengths.clone(),
        }
    }

    /// Shard batch across devices
    pub fn shard(&self, sharding: JaxSharding) -> Self {
        Self {
            input_ids: self.input_ids.shard(sharding.clone()),
            attention_mask: self.attention_mask.as_ref().map(|a| a.shard(sharding.clone())),
            token_type_ids: self.token_type_ids.as_ref().map(|a| a.shard(sharding.clone())),
            position_ids: self.position_ids.as_ref().map(|a| a.shard(sharding.clone())),
            special_tokens_mask: self.special_tokens_mask.as_ref().map(|a| a.shard(sharding)),
            sequence_lengths: self.sequence_lengths.clone(),
        }
    }

    /// Block until all arrays are ready
    pub fn block_until_ready(&self) -> Self {
        Self {
            input_ids: self.input_ids.block_until_ready(),
            attention_mask: self.attention_mask.as_ref().map(|a| a.block_until_ready()),
            token_type_ids: self.token_type_ids.as_ref().map(|a| a.block_until_ready()),
            position_ids: self.position_ids.as_ref().map(|a| a.block_until_ready()),
            special_tokens_mask: self.special_tokens_mask.as_ref().map(|a| a.block_until_ready()),
            sequence_lengths: self.sequence_lengths.clone(),
        }
    }
}

/// JAX integration wrapper for tokenizers
pub struct JaxTokenizer<T: Tokenizer> {
    tokenizer: Arc<T>,
    config: JaxConfig,
}

impl<T: Tokenizer + Clone> JaxTokenizer<T> {
    /// Create a new JAX tokenizer wrapper
    pub fn new(tokenizer: T, config: JaxConfig) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
            config,
        }
    }

    /// Create with default configuration
    pub fn from_tokenizer(tokenizer: T) -> Self {
        Self::new(tokenizer, JaxConfig::default())
    }

    /// Update configuration
    pub fn with_config(mut self, config: JaxConfig) -> Self {
        self.config = config;
        self
    }

    /// Encode text to JAX arrays
    pub fn encode_to_arrays(&self, text: &str) -> Result<JaxBatch> {
        let tokenized = self.tokenizer.encode(text)?;
        self.convert_to_batch(vec![tokenized])
    }

    /// Encode text pair to JAX arrays
    pub fn encode_pair_to_arrays(&self, text_a: &str, text_b: &str) -> Result<JaxBatch> {
        let tokenized = self.tokenizer.encode_pair(text_a, text_b)?;
        self.convert_to_batch(vec![tokenized])
    }

    /// Encode batch of texts to JAX arrays
    pub fn encode_batch_to_arrays(&self, texts: &[String]) -> Result<JaxBatch> {
        let mut tokenized_batch = Vec::new();

        for text in texts {
            let tokenized = self.tokenizer.encode(text)?;
            tokenized_batch.push(tokenized);
        }

        self.convert_to_batch(tokenized_batch)
    }

    /// Encode batch of text pairs to JAX arrays
    pub fn encode_pair_batch_to_arrays(&self, text_pairs: &[(String, String)]) -> Result<JaxBatch> {
        let mut tokenized_batch = Vec::new();

        for (text_a, text_b) in text_pairs {
            let tokenized = self.tokenizer.encode_pair(text_a, text_b)?;
            tokenized_batch.push(tokenized);
        }

        self.convert_to_batch(tokenized_batch)
    }

    /// Convert tokenized inputs to JAX batch
    fn convert_to_batch(&self, tokenized_inputs: Vec<TokenizedInput>) -> Result<JaxBatch> {
        if tokenized_inputs.is_empty() {
            return Err(anyhow!("Cannot create batch from empty input"));
        }

        let batch_size = tokenized_inputs.len();
        let sequence_lengths: Vec<usize> =
            tokenized_inputs.iter().map(|t| t.input_ids.len()).collect();

        // Determine sequence length
        let seq_length = match self.config.padding {
            JaxPaddingStrategy::False => {
                let first_len = sequence_lengths[0];
                if !sequence_lengths.iter().all(|&len| len == first_len) {
                    return Err(anyhow!(
                        "All sequences must be same length when padding is disabled"
                    ));
                }
                first_len
            },
            JaxPaddingStrategy::LongestFirst => sequence_lengths.iter().copied().max().unwrap_or(0),
            JaxPaddingStrategy::MaxLength => self.config.max_length.unwrap_or(512),
            JaxPaddingStrategy::Dynamic => {
                // Use actual longest sequence for dynamic padding
                sequence_lengths.iter().copied().max().unwrap_or(0)
            },
        };

        // Apply truncation
        let final_seq_length = if let Some(max_len) = self.config.max_length {
            match self.config.truncation {
                JaxTruncationStrategy::False => seq_length,
                _ => seq_length.min(max_len),
            }
        } else {
            seq_length
        };

        // Create arrays
        let mut input_ids_data = Vec::with_capacity(batch_size * final_seq_length);
        let mut attention_mask_data = Vec::with_capacity(batch_size * final_seq_length);
        let mut token_type_ids_data = Vec::with_capacity(batch_size * final_seq_length);
        let mut position_ids_data = Vec::with_capacity(batch_size * final_seq_length);
        let mut special_tokens_mask_data = Vec::with_capacity(batch_size * final_seq_length);

        let pad_token_id = 0i32;

        for tokenized in &tokenized_inputs {
            // Handle input_ids
            let mut seq_input_ids = tokenized.input_ids.clone();

            if seq_input_ids.len() > final_seq_length {
                seq_input_ids.truncate(final_seq_length);
            }

            while seq_input_ids.len() < final_seq_length {
                seq_input_ids.push(pad_token_id as u32);
            }

            input_ids_data.extend(seq_input_ids.into_iter().map(|id| id as i32));

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

                token_type_ids_data.extend(seq_token_type_ids.into_iter().map(|id| id as i32));
            }

            // Create position IDs
            if self.config.return_position_ids {
                for i in 0..final_seq_length {
                    position_ids_data.push(i as i32);
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
                .extend(seq_special_tokens_mask.into_iter().map(|mask| mask as i32));
        }

        // Create JAX arrays
        let input_ids = JaxArray::new(
            input_ids_data,
            vec![batch_size, final_seq_length],
            self.config.dtype,
            self.config.device.clone(),
        )
        .with_name("input_ids".to_string());

        let attention_mask = if self.config.return_attention_mask {
            Some(
                JaxArray::new(
                    attention_mask_data,
                    vec![batch_size, final_seq_length],
                    self.config.dtype,
                    self.config.device.clone(),
                )
                .with_name("attention_mask".to_string()),
            )
        } else {
            None
        };

        let token_type_ids = if self.config.return_token_type_ids {
            Some(
                JaxArray::new(
                    token_type_ids_data,
                    vec![batch_size, final_seq_length],
                    self.config.dtype,
                    self.config.device.clone(),
                )
                .with_name("token_type_ids".to_string()),
            )
        } else {
            None
        };

        let position_ids = if self.config.return_position_ids {
            Some(
                JaxArray::new(
                    position_ids_data,
                    vec![batch_size, final_seq_length],
                    self.config.dtype,
                    self.config.device.clone(),
                )
                .with_name("position_ids".to_string()),
            )
        } else {
            None
        };

        // Create special tokens mask array (only if any sequence has special tokens)
        let special_tokens_mask = if special_tokens_mask_data.iter().any(|&mask| mask != 0) {
            Some(
                JaxArray::new(
                    special_tokens_mask_data,
                    vec![batch_size, final_seq_length],
                    self.config.dtype,
                    self.config.device.clone(),
                )
                .with_name("special_tokens_mask".to_string()),
            )
        } else {
            None
        };

        Ok(JaxBatch::new(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            special_tokens_mask,
            sequence_lengths,
        ))
    }

    /// Get underlying tokenizer
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    /// Get configuration
    pub fn config(&self) -> &JaxConfig {
        &self.config
    }

    /// Create XLA-compiled version
    pub fn jit_compile(&self) -> Result<JaxCompiledTokenizer<T>> {
        JaxCompiledTokenizer::new(self.tokenizer.clone(), self.config.clone())
    }
}

/// XLA-compiled JAX tokenizer for high performance
pub struct JaxCompiledTokenizer<T: Tokenizer> {
    tokenizer: Arc<T>,
    config: JaxConfig,
    compiled: bool,
}

impl<T: Tokenizer + Clone> JaxCompiledTokenizer<T> {
    /// Create a new compiled tokenizer
    pub fn new(tokenizer: Arc<T>, config: JaxConfig) -> Result<Self> {
        // In real implementation, would compile with XLA
        Ok(Self {
            tokenizer,
            config,
            compiled: true,
        })
    }

    /// Encode batch with compiled function
    pub fn encode_batch_compiled(&self, texts: &[String]) -> Result<JaxBatch> {
        if !self.compiled {
            return Err(anyhow!("Tokenizer not compiled"));
        }

        // Use the same logic as the regular tokenizer for now
        // In real implementation, would use compiled XLA function
        let jax_tokenizer = JaxTokenizer::new((*self.tokenizer).clone(), self.config.clone());
        jax_tokenizer.encode_batch_to_arrays(texts)
    }

    /// Check if compiled
    pub fn is_compiled(&self) -> bool {
        self.compiled
    }
}

/// JAX dataset for data loading
pub struct JaxDataset {
    texts: Vec<String>,
    config: JaxConfig,
}

impl JaxDataset {
    /// Create a new dataset
    pub fn new(texts: Vec<String>, config: JaxConfig) -> Self {
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

    /// Create batch iterator
    pub fn batch_iter(&self, batch_size: usize) -> JaxDataIterator<'_> {
        JaxDataIterator::new(&self.texts, batch_size, self.config.clone())
    }

    /// Shuffle dataset
    pub fn shuffle(&self, seed: Option<u64>) -> Self {
        let mut texts = self.texts.clone();

        if let Some(seed_val) = seed {
            // Implement proper seeded shuffling using Fisher-Yates algorithm

            // Create a simple linear congruential generator with the seed
            let mut rng_state = seed_val;

            // Fisher-Yates shuffle algorithm
            for i in (1..texts.len()).rev() {
                // Generate next pseudo-random number
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);

                // Get random index from 0 to i (inclusive)
                let j = (rng_state as usize) % (i + 1);

                // Swap elements
                texts.swap(i, j);
            }
        }

        Self {
            texts,
            config: self.config.clone(),
        }
    }

    /// Repeat dataset
    pub fn repeat(&self, count: usize) -> Self {
        let mut texts = Vec::new();
        for _ in 0..count {
            texts.extend(self.texts.clone());
        }

        Self {
            texts,
            config: self.config.clone(),
        }
    }
}

/// Iterator for JAX data loading
pub struct JaxDataIterator<'a> {
    texts: &'a [String],
    batch_size: usize,
    current_index: usize,
    #[allow(dead_code)]
    config: JaxConfig,
}

impl<'a> JaxDataIterator<'a> {
    fn new(texts: &'a [String], batch_size: usize, config: JaxConfig) -> Self {
        Self {
            texts,
            batch_size,
            current_index: 0,
            config,
        }
    }

    /// Apply transformation function
    pub fn map<F>(self, _func: F) -> Self
    where
        F: Fn(&str) -> String,
    {
        // In real implementation, would apply the function
        self
    }

    /// Filter samples
    pub fn filter<F>(self, _predicate: F) -> Self
    where
        F: Fn(&str) -> bool,
    {
        // In real implementation, would filter samples
        self
    }
}

impl<'a> Iterator for JaxDataIterator<'a> {
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

/// Utilities for JAX integration
pub struct JaxUtils;

impl JaxUtils {
    /// Convert array to debug string
    pub fn array_to_debug_string(array: &JaxArray) -> String {
        format!(
            "JaxArray(shape={:?}, dtype={:?}, device={}, data={:?})",
            array.shape,
            array.dtype,
            array.device,
            &array.data[..array.data.len().min(10)] // Show first 10 elements
        )
    }

    /// Calculate array memory usage
    pub fn array_memory_usage(array: &JaxArray) -> usize {
        array.size() * array.dtype.size_bytes()
    }

    /// Create device mesh for distributed computation
    pub fn create_device_mesh(
        devices: Vec<JaxDevice>,
        shape: Vec<usize>,
        axis_names: Vec<String>,
    ) -> JaxMesh {
        JaxMesh {
            devices,
            shape,
            axis_names,
        }
    }

    /// Create sharding specification
    pub fn create_sharding(mesh: JaxMesh, partition_spec: Vec<Option<String>>) -> JaxSharding {
        JaxSharding {
            mesh,
            partition_spec,
        }
    }

    /// Validate batch for model input
    pub fn validate_model_inputs(batch: &JaxBatch) -> Result<()> {
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

    /// Create optimized device placement
    pub fn suggest_device_placement(
        array_size: usize,
        available_devices: &[JaxDevice],
    ) -> JaxDevice {
        // Simple heuristic: use GPU for large arrays, CPU for small ones
        if array_size > 1_000_000
            && available_devices.iter().any(|d| matches!(d, JaxDevice::Gpu(_)))
        {
            available_devices
                .iter()
                .find(|d| matches!(d, JaxDevice::Gpu(_)))
                .unwrap_or(&JaxDevice::Cpu)
                .clone()
        } else {
            JaxDevice::Cpu
        }
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
    fn test_jax_config() {
        let config = JaxConfig::default();
        assert_eq!(config.dtype, JaxDType::Int32);
        assert_eq!(config.max_length, Some(512));
        assert!(config.return_attention_mask);
        assert!(config.use_xla);
    }

    #[test]
    fn test_jax_array() {
        let data = vec![1, 2, 3, 4];
        let shape = vec![2, 2];
        let array = JaxArray::new(data.clone(), shape.clone(), JaxDType::Int32, JaxDevice::Cpu);

        assert_eq!(array.data, data);
        assert_eq!(array.shape, shape);
        assert_eq!(array.ndim(), 2);
        assert_eq!(array.size(), 4);
    }

    #[test]
    fn test_array_reshape() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let array = JaxArray::new(data, vec![2, 3], JaxDType::Int32, JaxDevice::Cpu);

        let reshaped = array.reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped.shape, vec![3, 2]);
        assert_eq!(reshaped.size(), 6);
    }

    #[test]
    fn test_jax_tokenizer() {
        let tokenizer = create_test_char_tokenizer();
        let jax_tokenizer = JaxTokenizer::from_tokenizer(tokenizer);

        let batch = jax_tokenizer.encode_to_arrays("hello").unwrap();
        assert_eq!(batch.batch_size(), 1);
        assert!(batch.attention_mask.is_some());
    }

    #[test]
    fn test_batch_encoding() {
        let tokenizer = create_test_char_tokenizer();
        let jax_tokenizer = JaxTokenizer::from_tokenizer(tokenizer);

        let texts = vec!["hello".to_string(), "world".to_string()];
        let batch = jax_tokenizer.encode_batch_to_arrays(&texts).unwrap();

        assert_eq!(batch.batch_size(), 2);
        assert!(batch.attention_mask.is_some());
        assert_eq!(batch.sequence_lengths.len(), 2);
    }

    #[test]
    fn test_device_placement() {
        let array = JaxArray::new(vec![1, 2, 3], vec![3], JaxDType::Int32, JaxDevice::Cpu);
        let gpu_array = array.to_device(JaxDevice::Gpu(0));

        assert!(matches!(gpu_array.device, JaxDevice::Gpu(0)));
    }

    #[test]
    fn test_jax_dataset() {
        let texts = vec!["hello".to_string(), "world".to_string(), "test".to_string()];
        let config = JaxConfig::default();
        let dataset = JaxDataset::new(texts, config);

        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.get_item(0), Some("hello"));

        let batches: Vec<_> = dataset.batch_iter(2).collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 1);
    }

    #[test]
    fn test_compiled_tokenizer() {
        let tokenizer = create_test_char_tokenizer();
        let jax_tokenizer = JaxTokenizer::from_tokenizer(tokenizer);

        let compiled = jax_tokenizer.jit_compile().unwrap();
        assert!(compiled.is_compiled());

        let texts = vec!["hello".to_string()];
        let batch = compiled.encode_batch_compiled(&texts).unwrap();
        assert_eq!(batch.batch_size(), 1);
    }

    #[test]
    fn test_jax_utils() {
        let array = JaxArray::new(
            vec![1, 2, 3, 4],
            vec![2, 2],
            JaxDType::Int32,
            JaxDevice::Cpu,
        );

        let debug_str = JaxUtils::array_to_debug_string(&array);
        assert!(debug_str.contains("shape=[2, 2]"));
        assert!(debug_str.contains("dtype=Int32"));

        let memory_usage = JaxUtils::array_memory_usage(&array);
        assert_eq!(memory_usage, 4 * 4); // 4 elements * 4 bytes (Int32)
    }

    #[test]
    fn test_sharding() {
        let devices = vec![JaxDevice::Gpu(0), JaxDevice::Gpu(1)];
        let mesh = JaxUtils::create_device_mesh(devices, vec![2], vec!["data".to_string()]);
        let sharding = JaxUtils::create_sharding(mesh, vec![Some("data".to_string())]);

        assert_eq!(sharding.mesh.devices.len(), 2);
        assert_eq!(sharding.partition_spec.len(), 1);
    }
}
