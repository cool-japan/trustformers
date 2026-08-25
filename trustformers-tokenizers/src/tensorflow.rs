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

/// How many full passes over the (transformed) dataset [`TfDataIterator`]
/// produces. Mirrors `tf.data.Dataset.repeat`: the *last* `.repeat()` call
/// in a chain governs the whole pipeline (calling it twice replaces the
/// setting rather than compounding it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TfRepeat {
    /// Exactly one pass -- the default, un-repeated behavior.
    Once,
    /// Exactly `n` passes back to back. `Times(0)` yields nothing.
    Times(usize),
    /// Passes forever. Legal (like `std::iter::repeat`), but the caller
    /// must bound consumption itself (e.g. `.take(n)`); see
    /// [`TfDataIterator::size_hint`].
    Infinite,
}

/// A small, fast, deterministic pseudo-random generator (SplitMix64), used
/// only to shuffle batch order reproducibly. Not cryptographic.
///
/// `scirs2_core::random` (this workspace's usual RNG) is not usable here:
/// its `random` module is behind scirs2-core's own `random` cargo feature,
/// which this crate's `scirs2-core` dependency does not enable (only
/// `parallel` is enabled -- see this crate's Cargo.toml; enabling `random`
/// there is a Cargo.toml change, tracked as a follow-up, not something this
/// file can do by itself). A small local generator is an established
/// pattern in this workspace for exactly this situation -- see
/// `trustformers-models/src/bert/config.rs`'s in-tree LCG for test-fixture
/// generation.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniformly-distributed value in `0..bound` (for `bound > 0`), via
    /// Lemire's debiased-modulo method (<https://arxiv.org/abs/1805.10941>;
    /// the same rejection strategy `rand`'s `Uniform` uses) -- a single
    /// 64x64->128 multiply on the common path, no division.
    fn next_below(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        let mut m = u128::from(self.next_u64()) * u128::from(bound);
        let mut low = m as u64;
        if low < bound {
            let threshold = bound.wrapping_neg() % bound;
            while low < threshold {
                m = u128::from(self.next_u64()) * u128::from(bound);
                low = m as u64;
            }
        }
        (m >> 64) as u64
    }
}

/// A best-effort, non-deterministic seed for the *unseeded*
/// [`TfDataIterator::shuffle`] entry point; callers who need
/// reproducibility use [`TfDataIterator::shuffle_seeded`] instead.
/// Not cryptographic: `RandomState`'s per-process random keys are the
/// standard library's own source of ambient randomness (the same
/// mechanism that makes `HashMap` iteration order unpredictable), reused
/// here so an unseeded shuffle looks shuffled without adding a dependency.
fn entropy_seed() -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    RandomState::new().build_hasher().finish()
}

/// Iterator for TensorFlow tf.data.Dataset compatibility, with real
/// `.map()`, `.repeat()` and `.shuffle()`.
///
/// # Design
///
/// `map`/`repeat`/`shuffle` record a small pipeline description
/// (`transforms`, `repeat_spec`, `shuffle_spec`) rather than eagerly
/// running, so they compose in any call order --
/// `.shuffle(n).repeat(k).map(f)` and `.map(f).shuffle(n).repeat(k)` behave
/// the same. Every `.map()` call appends one function to `transforms`,
/// applied to each text in a batch in the order added; `.repeat()` and
/// `.shuffle()`/`.shuffle_seeded()` each *replace* the previous call of the
/// same kind, mirroring `tf.data`, where the last call of a given kind
/// governs the whole upstream pipeline.
///
/// Each pass re-derives its batches directly from the original `texts`
/// (nothing is ever materialized into a buffered `Vec` beyond the shuffle
/// reservoir, which is capped at `buffer_size`), so `.repeat(None)` -- a
/// genuinely infinite iterator, like `std::iter::repeat` -- cannot hang a
/// caller that bounds consumption (e.g. with `.take(n)`); see
/// [`TfDataIterator::size_hint`]. Shuffling does not mix batches across a
/// `repeat()`'s pass boundaries: each pass's reservoir is drained before
/// the next pass starts filling, and (when a fixed seed was given) is
/// reseeded deterministically per pass so a repeated, seeded shuffle is
/// reproducible across runs without being identical on every single pass.
pub struct TfDataIterator<'a> {
    texts: &'a [String],
    batch_size: usize,
    // reason: stored from the constructor; reserved for planned per-batch
    // TensorFlow configuration that the iterator does not yet consume.
    #[allow(dead_code)]
    config: TensorFlowConfig,
    transforms: Vec<Box<dyn Fn(&str) -> String + 'a>>,
    shuffle_spec: Option<(usize, Option<u64>)>,
    repeat_spec: TfRepeat,

    // Runtime state.
    pass: usize,
    cursor: usize,
    shuffle_rng: Option<SplitMix64>,
    shuffle_buf: Vec<Vec<String>>,
    finished: bool,
}

impl<'a> TfDataIterator<'a> {
    fn new(texts: &'a [String], batch_size: usize, config: TensorFlowConfig) -> Self {
        Self {
            texts,
            // A batch size of 0 would make every batch empty forever;
            // clamp to 1 so this iterator always makes real progress
            // instead of hanging a caller that iterates it fully.
            batch_size: batch_size.max(1),
            config,
            transforms: Vec::new(),
            shuffle_spec: None,
            repeat_spec: TfRepeat::Once,
            pass: 0,
            cursor: 0,
            shuffle_rng: None,
            shuffle_buf: Vec::new(),
            finished: false,
        }
    }

    /// Apply `func` to every text in every yielded batch (similar to
    /// `tf.data.Dataset.map`). Composable: a second `.map()` call applies
    /// its function *after* the first, to each already-mapped text.
    pub fn map<F>(mut self, func: F) -> Self
    where
        F: Fn(&str) -> String + 'a,
    {
        self.transforms.push(Box::new(func));
        self
    }

    /// Replay the (transformed) dataset `count` times back to back;
    /// `None` repeats forever. See the type-level doc for the iterator
    /// contract this guarantees (no materialization, so an infinite
    /// repeat cannot hang a bounded consumer).
    pub fn repeat(mut self, count: Option<usize>) -> Self {
        self.repeat_spec = match count {
            None => TfRepeat::Infinite,
            Some(n) => TfRepeat::Times(n),
        };
        self
    }

    /// Maintain a real reservoir of up to `buffer_size` yielded batches,
    /// draining a uniformly-random one on every pull and backfilling from
    /// upstream, seeded from process entropy (unpredictable, not
    /// reproducible). Use [`Self::shuffle_seeded`] for deterministic
    /// shuffling (e.g. in tests).
    pub fn shuffle(self, buffer_size: usize) -> Self {
        self.shuffle_seeded(buffer_size, None)
    }

    /// Like [`Self::shuffle`], but with an explicit seed: the same seed
    /// always produces the same permutation of the same multiset of
    /// batches.
    pub fn shuffle_seeded(mut self, buffer_size: usize, seed: Option<u64>) -> Self {
        self.shuffle_spec = Some((buffer_size.max(1), seed));
        self.shuffle_rng = None; // re-derive lazily from the (possibly new) seed
        self.shuffle_buf.clear();
        self
    }

    fn current_pass_is_valid(&self) -> bool {
        match self.repeat_spec {
            TfRepeat::Once => self.pass == 0,
            TfRepeat::Times(n) => self.pass < n,
            TfRepeat::Infinite => true,
        }
    }

    fn next_raw_batch(&mut self) -> Option<Vec<String>> {
        if self.cursor >= self.texts.len() {
            return None;
        }
        let end = (self.cursor + self.batch_size).min(self.texts.len());
        let batch = self.texts[self.cursor..end].to_vec();
        self.cursor = end;
        Some(batch)
    }

    fn apply_transforms(&self, batch: Vec<String>) -> Vec<String> {
        if self.transforms.is_empty() {
            return batch;
        }
        batch
            .into_iter()
            .map(|text| self.transforms.iter().fold(text, |acc, f| f(acc.as_str())))
            .collect()
    }

    fn next_shuffle_index(&mut self, len: usize) -> usize {
        if self.shuffle_rng.is_none() {
            let seed = self.shuffle_spec.and_then(|(_, seed)| seed).unwrap_or_else(entropy_seed);
            self.shuffle_rng = Some(SplitMix64::new(seed));
        }
        let rng = self.shuffle_rng.as_mut().expect("just initialized above");
        rng.next_below(len as u64) as usize
    }

    fn pop_random_from_shuffle_buf(&mut self) -> Vec<String> {
        let len = self.shuffle_buf.len();
        let idx = self.next_shuffle_index(len);
        self.shuffle_buf.swap_remove(idx)
    }

    fn start_next_pass(&mut self) {
        self.pass += 1;
        self.cursor = 0;
        if let Some((_, Some(seed))) = self.shuffle_spec {
            self.shuffle_rng = Some(SplitMix64::new(seed.wrapping_add(self.pass as u64)));
        }
    }
}

impl<'a> Iterator for TfDataIterator<'a> {
    type Item = Vec<String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.finished {
                return None;
            }
            if !self.current_pass_is_valid() {
                self.finished = true;
                return None;
            }

            if let Some(batch) = self.next_raw_batch() {
                let mapped = self.apply_transforms(batch);
                match self.shuffle_spec {
                    None => return Some(mapped),
                    Some((buffer_size, _)) => {
                        self.shuffle_buf.push(mapped);
                        if self.shuffle_buf.len() >= buffer_size {
                            return Some(self.pop_random_from_shuffle_buf());
                        }
                        // Reservoir not yet full: keep pulling before
                        // yielding anything.
                        continue;
                    },
                }
            } else if self.shuffle_spec.is_some() && !self.shuffle_buf.is_empty() {
                // This pass's raw batches are exhausted; drain the
                // reservoir (also in random order) before moving on.
                return Some(self.pop_random_from_shuffle_buf());
            } else {
                self.start_next_pass();
                // Loop back around: the top-of-loop `current_pass_is_valid`
                // check decides whether the new pass actually runs.
                continue;
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.repeat_spec {
            // Honest per the `Iterator` contract: unbounded, so a caller
            // relying on `size_hint` (e.g. before eagerly collecting) is
            // told not to.
            TfRepeat::Infinite => (usize::MAX, None),
            // The exact remaining count depends on runtime cursor/reservoir
            // state that is not cheap to compute in advance; no lower bound
            // beyond the trivial one is claimed.
            _ => (0, None),
        }
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

        let reshaped = tensor.reshape(vec![3, 2]).expect("Operation failed in test");
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

        let batch = tf_tokenizer.encode_to_tensors("hello").expect("Operation failed in test");
        assert_eq!(batch.batch_size(), 1);
        assert!(batch.attention_mask.is_some());
    }

    #[test]
    fn test_batch_encoding() {
        let tokenizer = create_test_char_tokenizer();
        let tf_tokenizer = TensorFlowTokenizer::from_tokenizer(tokenizer);

        let texts = vec!["hello".to_string(), "world".to_string()];
        let batch = tf_tokenizer.encode_batch_to_tensors(&texts).expect("Operation failed in test");

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
        let batch = tf_tokenizer.encode_batch_to_tensors(&texts).expect("Operation failed in test");

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

    // ---- TfDataIterator: map / repeat / shuffle actually do something ----
    //
    // Regression coverage for the previous implementation, where
    // `.map(f)`/`.repeat(n)`/`.shuffle(n)` all discarded their arguments
    // and returned `self` unchanged, so a chained `.shuffle().repeat()
    // .map(f)` silently yielded the original, untransformed batches.

    fn three_texts() -> Vec<String> {
        vec!["hello".to_string(), "world".to_string(), "test".to_string()]
    }

    #[test]
    fn map_transforms_every_text_in_every_batch() {
        let texts = three_texts();
        let dataset = TensorFlowDataset::new(texts.clone(), TensorFlowConfig::default());

        let batches: Vec<Vec<String>> = dataset.tf_data_iter(2).map(|s| s.to_uppercase()).collect();

        let flattened: Vec<String> = batches.into_iter().flatten().collect();
        let expected: Vec<String> = texts.iter().map(|s| s.to_uppercase()).collect();
        assert_eq!(
            flattened, expected,
            "map must apply the function, not discard it"
        );
    }

    #[test]
    fn chained_map_calls_compose_in_order() {
        let texts = vec!["a".to_string()];
        let dataset = TensorFlowDataset::new(texts, TensorFlowConfig::default());

        let batches: Vec<Vec<String>> = dataset
            .tf_data_iter(1)
            .map(|s| format!("{s}1"))
            .map(|s| format!("{s}2"))
            .collect();

        assert_eq!(batches, vec![vec!["a12".to_string()]]);
    }

    #[test]
    fn repeat_two_doubles_the_batch_and_element_count() {
        let texts = three_texts();
        let dataset = TensorFlowDataset::new(texts.clone(), TensorFlowConfig::default());

        let once: Vec<Vec<String>> = dataset.tf_data_iter(2).collect();
        let dataset2 = TensorFlowDataset::new(texts, TensorFlowConfig::default());
        let repeated: Vec<Vec<String>> = dataset2.tf_data_iter(2).repeat(Some(2)).collect();

        assert_eq!(
            repeated.len(),
            once.len() * 2,
            "repeat(2) must double the batch count"
        );
        let flattened: Vec<&String> = repeated.iter().flatten().collect();
        assert_eq!(
            flattened.len(),
            6,
            "repeat(2) must double the element count (3 -> 6)"
        );
    }

    #[test]
    fn repeat_zero_yields_nothing() {
        let texts = three_texts();
        let dataset = TensorFlowDataset::new(texts, TensorFlowConfig::default());

        let batches: Vec<Vec<String>> = dataset.tf_data_iter(2).repeat(Some(0)).collect();
        assert!(
            batches.is_empty(),
            "repeat(Some(0)) must yield no batches at all"
        );
    }

    #[test]
    fn repeat_none_is_infinite_but_take_terminates() {
        let texts = three_texts();
        let dataset = TensorFlowDataset::new(texts, TensorFlowConfig::default());

        // Regression guard for the iterator contract: this must not hang.
        // `usize::MAX` low bound signals "unbounded" honestly via size_hint.
        let iter = dataset.tf_data_iter(2).repeat(None);
        assert_eq!(iter.size_hint(), (usize::MAX, None));
        let batches: Vec<Vec<String>> = iter.take(100).collect();
        assert_eq!(batches.len(), 100);
    }

    #[test]
    fn shuffle_seeded_is_deterministic_for_a_fixed_seed() {
        let texts: Vec<String> = (0..20).map(|i| i.to_string()).collect();

        let dataset_a = TensorFlowDataset::new(texts.clone(), TensorFlowConfig::default());
        let a: Vec<Vec<String>> = dataset_a.tf_data_iter(1).shuffle_seeded(5, Some(42)).collect();

        let dataset_b = TensorFlowDataset::new(texts, TensorFlowConfig::default());
        let b: Vec<Vec<String>> = dataset_b.tf_data_iter(1).shuffle_seeded(5, Some(42)).collect();

        assert_eq!(a, b, "the same seed must produce the same permutation");
    }

    #[test]
    fn shuffle_seeded_yields_the_same_multiset_reordered() {
        let texts: Vec<String> = (0..20).map(|i| i.to_string()).collect();
        let dataset = TensorFlowDataset::new(texts.clone(), TensorFlowConfig::default());

        let mut shuffled: Vec<String> =
            dataset.tf_data_iter(1).shuffle_seeded(5, Some(7)).flatten().collect();
        let mut original = texts;

        assert_ne!(
            shuffled, original,
            "a real shuffle over 20 items must reorder them"
        );
        shuffled.sort();
        original.sort();
        assert_eq!(
            shuffled, original,
            "shuffling must not lose or invent any element"
        );
    }

    #[test]
    fn chained_shuffle_repeat_map_all_take_effect_together() {
        // The exact pattern the previous no-op implementation silently
        // defeated: `.shuffle().repeat().map(f)`.
        let texts = three_texts();
        let dataset = TensorFlowDataset::new(texts.clone(), TensorFlowConfig::default());

        let batches: Vec<Vec<String>> = dataset
            .tf_data_iter(1)
            .shuffle_seeded(2, Some(1))
            .repeat(Some(2))
            .map(|s| s.to_uppercase())
            .collect();

        let flattened: Vec<String> = batches.into_iter().flatten().collect();
        assert_eq!(
            flattened.len(),
            6,
            "repeat(2) over 3 texts must yield 6 elements"
        );
        assert!(
            flattened
                .iter()
                .all(|s| s.chars().all(|c| c.is_uppercase() || !c.is_alphabetic())),
            "map must have run: every element must be uppercase: {flattened:?}"
        );
        let mut expected_multiset: Vec<String> = texts
            .iter()
            .map(|s| s.to_uppercase())
            .chain(texts.iter().map(|s| s.to_uppercase()))
            .collect();
        let mut got = flattened;
        expected_multiset.sort();
        got.sort();
        assert_eq!(
            got, expected_multiset,
            "repeat must not lose or duplicate beyond 2x"
        );
    }

    #[test]
    fn zero_batch_size_is_clamped_and_does_not_hang() {
        let texts = three_texts();
        let dataset = TensorFlowDataset::new(texts, TensorFlowConfig::default());

        let batches: Vec<Vec<String>> = dataset.tf_data_iter(0).collect();
        // Clamped to 1: three texts, one per batch.
        assert_eq!(batches.len(), 3);
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
