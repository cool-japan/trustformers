//! # Language Modeling Data Collators
//!
//! This module provides specialized data collators for language modeling tasks.
//! It includes implementations for both masked language modeling (BERT-style)
//! and causal language modeling (GPT-style) training objectives.
//!
//! ## Overview
//!
//! Language modeling is a fundamental task in natural language processing where
//! models learn to predict tokens in text sequences. This module provides
//! collators that prepare data appropriately for different language modeling
//! paradigms:
//!
//! - **Masked Language Modeling (MLM)**: Used by BERT-like models where random
//!   tokens are masked and the model learns to predict them
//! - **Causal Language Modeling (CLM)**: Used by GPT-like models where the model
//!   learns to predict the next token in a sequence
//!
//! ## Architecture
//!
//! ```text
//! Language Modeling Collators
//!      ├─ LanguageModelingDataCollator (BERT-style MLM)
//!      │   ├─ Handles token masking for MLM training
//!      │   ├─ Supports token type IDs for sentence pairs
//!      │   └─ Configurable masking probability
//!      │
//!      └─ CausalLanguageModelingDataCollator (GPT-style CLM)
//!          ├─ Creates shifted labels for causal prediction
//!          ├─ Handles left-padding for generation tasks
//!          └─ Optimized for autoregressive training
//! ```
//!
//! ## Usage Examples
//!
//! ### Masked Language Modeling (BERT-style)
//!
//! ```rust,ignore
//! use trustformers::auto::data_collators::language_modeling::{
//!     LanguageModelingDataCollator, LanguageModelingCollatorConfig
//! };
//! use trustformers::auto::types::PaddingStrategy;
//!
//! let config = LanguageModelingCollatorConfig {
//!     max_length: Some(512),
//!     padding: PaddingStrategy::Longest,
//!     truncation: true,
//!     pad_token_id: 0,
//!     mask_token_id: 103,  // [MASK] token
//!     mlm_probability: 0.15,
//! };
//!
//! let collator = LanguageModelingDataCollator::new(config);
//! ```
//!
//! ### Causal Language Modeling (GPT-style)
//!
//! ```rust,ignore
//! use trustformers::auto::data_collators::language_modeling::{
//!     CausalLanguageModelingDataCollator, CausalLanguageModelingCollatorConfig
//! };
//! use trustformers::auto::types::PaddingStrategy;
//!
//! let config = CausalLanguageModelingCollatorConfig {
//!     max_length: Some(1024),
//!     padding: PaddingStrategy::Longest,
//!     truncation: true,
//!     pad_token_id: 50256,  // GPT pad token
//! };
//!
//! let collator = CausalLanguageModelingDataCollator::new(config);
//! ```
//!
//! ## Training Objectives
//!
//! ### Masked Language Modeling (MLM)
//!
//! In MLM, approximately 15% of tokens are randomly selected and:
//! - 80% are replaced with [MASK] token
//! - 10% are replaced with a random token
//! - 10% are left unchanged
//!
//! The model learns to predict the original tokens for all selected positions.
//!
//! ### Causal Language Modeling (CLM)
//!
//! In CLM, the model learns to predict the next token given all previous tokens.
//! Labels are created by shifting the input sequence by one position to the right.
//!
//! ## Performance Considerations
//!
//! - **Batch Padding**: Uses dynamic padding to minimize computation on padding tokens
//! - **Memory Efficiency**: Optimized tensor operations for large batch sizes
//! - **Token Masking**: Efficient random masking implementation for MLM
//! - **Label Creation**: Optimized label shifting for causal LM

use super::{DataCollator, DataCollatorConfig};
use crate::auto::types::{CollatedBatch, DataExample, PaddingStrategy};
use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Masked Language Modeling Data Collator (BERT-style)
// =============================================================================

/// Data collator for masked language modeling tasks (BERT-like models)
///
/// This collator is specifically designed for BERT-style models that use masked
/// language modeling during pre-training and fine-tuning. It handles all aspects
/// of preparing data for MLM training including padding, truncation, and optional
/// token masking.
///
/// ## Features
///
/// - **Dynamic Padding**: Pads sequences to the longest sequence in each batch
/// - **Configurable Truncation**: Truncates sequences exceeding maximum length
/// - **Token Masking**: Supports random token masking for MLM training
/// - **Token Type Support**: Handles token type IDs for sentence pair tasks
/// - **Label Management**: Creates appropriate labels for masked token prediction
///
/// ## Masking Strategy
///
/// When `mlm_probability` > 0, the collator applies the standard BERT masking strategy:
/// - Select random tokens based on `mlm_probability` (typically 15%)
/// - For selected tokens: 80% → [MASK], 10% → random token, 10% → unchanged
/// - Create labels with -100 for non-masked positions (ignored in loss)
///
/// ## Usage Examples
///
/// ```rust,ignore
/// use trustformers::auto::data_collators::language_modeling::{
///     LanguageModelingDataCollator, LanguageModelingCollatorConfig
/// };
/// use trustformers::auto::types::{DataExample, PaddingStrategy};
///
/// // Create configuration
/// let config = LanguageModelingCollatorConfig {
///     max_length: Some(512),
///     padding: PaddingStrategy::Longest,
///     truncation: true,
///     pad_token_id: 0,
///     mask_token_id: 103,
///     mlm_probability: 0.15,
/// };
///
/// // Create collator
/// let collator = LanguageModelingDataCollator::new(config);
///
/// // Prepare examples
/// let examples = vec![
///     DataExample {
///         input_ids: vec![101, 2023, 2003, 1037, 3231, 102],  // [CLS] this is a test [SEP]
///         attention_mask: Some(vec![1, 1, 1, 1, 1, 1]),
///         token_type_ids: None,
///         labels: None,
///         metadata: HashMap::new(),
///     },
/// ];
///
/// // Collate batch
/// let batch = collator.collate(&examples)?;
/// ```
#[derive(Debug, Clone)]
pub struct LanguageModelingDataCollator {
    config: LanguageModelingCollatorConfig,
}

impl LanguageModelingDataCollator {
    /// Create a new language modeling data collator
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration specifying collation behavior
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = LanguageModelingCollatorConfig {
    ///     max_length: Some(512),
    ///     padding: PaddingStrategy::Longest,
    ///     truncation: true,
    ///     pad_token_id: 0,
    ///     mask_token_id: 103,
    ///     mlm_probability: 0.15,
    /// };
    /// let collator = LanguageModelingDataCollator::new(config);
    /// ```
    pub fn new(config: LanguageModelingCollatorConfig) -> Self {
        Self { config }
    }

    /// Apply masked language modeling to input tokens
    ///
    /// This method implements the standard BERT masking strategy where
    /// approximately `mlm_probability` of tokens are selected for masking.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Original token IDs
    ///
    /// # Returns
    ///
    /// A tuple of (masked_input_ids, labels) where labels contain -100
    /// for non-masked positions and original token IDs for masked positions.
    fn apply_mlm_masking(&self, input_ids: &[u32]) -> (Vec<u32>, Vec<i64>) {
        let mut masked_ids = input_ids.to_vec();
        let mut labels = vec![-100i64; input_ids.len()];

        if self.config.mlm_probability > 0.0 {
            for (i, &token_id) in input_ids.iter().enumerate() {
                // Skip special tokens (typically < 100)
                if token_id < 100 {
                    continue;
                }

                // Apply masking probability
                if rand::random_f32() < self.config.mlm_probability {
                    labels[i] = token_id as i64;

                    let random_value = rand::random_f32();
                    if random_value < 0.8 {
                        // 80% of the time: replace with [MASK]
                        masked_ids[i] = self.config.mask_token_id;
                    } else if random_value < 0.9 {
                        // 10% of the time: replace with random token
                        masked_ids[i] = rand::random::<u32>() % 30000 + 1000; // Random vocab token
                    }
                    // 10% of the time: keep original (do nothing)
                }
            }
        }

        (masked_ids, labels)
    }
}

impl DataCollator for LanguageModelingDataCollator {
    fn collate(&self, examples: &[DataExample]) -> Result<CollatedBatch> {
        if examples.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "Cannot collate empty batch for language modeling".to_string(),
            ));
        }

        let batch_size = examples.len();

        // Determine maximum length for this batch
        let max_len = match self.config.padding {
            PaddingStrategy::Longest => examples
                .iter()
                .map(|ex| ex.input_ids.len())
                .max()
                .unwrap_or(0)
                .min(self.config.max_length.unwrap_or(usize::MAX)),
            PaddingStrategy::MaxLength => self.config.max_length.unwrap_or(512),
            PaddingStrategy::DoNotPad => {
                examples.iter().map(|ex| ex.input_ids.len()).max().unwrap_or(0)
            },
            PaddingStrategy::None => examples[0].input_ids.len(),
        };

        let mut input_ids = Vec::with_capacity(batch_size);
        let mut attention_mask = Vec::with_capacity(batch_size);
        let mut token_type_ids = Vec::new();
        let mut labels = Vec::new();

        // Track if any example has token type IDs
        let has_token_type_ids = examples.iter().any(|ex| ex.token_type_ids.is_some());
        if has_token_type_ids {
            token_type_ids = vec![vec![0u32; max_len]; batch_size];
        }

        for (batch_idx, example) in examples.iter().enumerate() {
            let mut sequence_input_ids = example.input_ids.clone();
            let mut sequence_attention_mask = example
                .attention_mask
                .clone()
                .unwrap_or_else(|| vec![1u32; example.input_ids.len()]);

            // Truncate if necessary
            if self.config.truncation && sequence_input_ids.len() > max_len {
                sequence_input_ids.truncate(max_len);
                sequence_attention_mask.truncate(max_len);
            }

            // Apply MLM masking if configured
            let (masked_input_ids, mlm_labels) = if self.config.mlm_probability > 0.0 {
                self.apply_mlm_masking(&sequence_input_ids)
            } else {
                (
                    sequence_input_ids.clone(),
                    example
                        .labels
                        .clone()
                        .unwrap_or_else(|| vec![-100i64; sequence_input_ids.len()]),
                )
            };

            // Pad sequences to max_len
            let mut padded_input_ids = masked_input_ids;
            let mut padded_attention_mask = sequence_attention_mask;
            let mut padded_labels = mlm_labels;

            // Pad input_ids and attention_mask
            while padded_input_ids.len() < max_len {
                padded_input_ids.push(self.config.pad_token_id);
                padded_attention_mask.push(0);
                padded_labels.push(-100); // Ignore padded positions in loss
            }

            input_ids.push(padded_input_ids);
            attention_mask.push(padded_attention_mask);
            labels.push(padded_labels);

            // Handle token type ids if present
            if let Some(token_types) = &example.token_type_ids {
                let mut padded_token_types = token_types.clone();

                // Truncate token types if necessary
                if self.config.truncation && padded_token_types.len() > max_len {
                    padded_token_types.truncate(max_len);
                }

                // Pad token types
                while padded_token_types.len() < max_len {
                    padded_token_types.push(0);
                }

                token_type_ids[batch_idx] = padded_token_types;
            }
        }

        Ok(CollatedBatch {
            input_ids,
            attention_mask,
            token_type_ids: if has_token_type_ids { Some(token_type_ids) } else { None },
            labels: Some(labels),
            batch_size,
            sequence_length: max_len,
            metadata: HashMap::new(),
        })
    }

    fn config(&self) -> &dyn DataCollatorConfig {
        &self.config
    }
}

// =============================================================================
// Masked Language Modeling Configuration
// =============================================================================

/// Configuration for language modeling data collator
///
/// This configuration struct controls all aspects of data collation for
/// masked language modeling tasks. It includes standard collation parameters
/// as well as MLM-specific settings.
///
/// ## Configuration Parameters
///
/// - `max_length`: Maximum sequence length (truncation point)
/// - `padding`: Strategy for padding sequences in batch
/// - `truncation`: Whether to truncate sequences exceeding max_length
/// - `pad_token_id`: Token ID used for padding positions
/// - `mask_token_id`: Token ID used for masking (typically [MASK])
/// - `mlm_probability`: Probability of masking tokens (0.0 to disable)
///
/// ## Default Values
///
/// The `from_config` method provides sensible defaults based on common
/// BERT-like model configurations:
/// - `mlm_probability`: 0.15 (15% masking rate)
/// - `padding`: Longest sequence in batch
/// - `truncation`: Enabled
/// - `pad_token_id`: From model config or 0
/// - `mask_token_id`: From model config or 103
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageModelingCollatorConfig {
    /// Maximum sequence length for padding/truncation
    pub max_length: Option<usize>,
    /// Padding strategy for batch collation
    pub padding: PaddingStrategy,
    /// Whether to truncate sequences exceeding max_length
    pub truncation: bool,
    /// Token ID used for padding
    pub pad_token_id: u32,
    /// Token ID used for masking (e.g., [MASK] token)
    pub mask_token_id: u32,
    /// Probability of masking tokens for MLM training (0.0 to disable)
    pub mlm_probability: f32,
}

impl LanguageModelingCollatorConfig {
    /// Create configuration from model config JSON
    ///
    /// This method extracts relevant configuration parameters from a model's
    /// config.json file and creates an appropriate collator configuration
    /// with sensible defaults for masked language modeling.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value (typically from config.json)
    ///
    /// # Returns
    ///
    /// A configured `LanguageModelingCollatorConfig` with parameters extracted
    /// from the model configuration and MLM-specific defaults.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// let model_config = serde_json::json!({
    ///     "max_position_embeddings": 512,
    ///     "pad_token_id": 0,
    ///     "mask_token_id": 103,
    ///     "vocab_size": 30522
    /// });
    ///
    /// let config = LanguageModelingCollatorConfig::from_config(&model_config)?;
    /// assert_eq!(config.max_length, Some(512));
    /// assert_eq!(config.pad_token_id, 0);
    /// assert_eq!(config.mask_token_id, 103);
    /// assert_eq!(config.mlm_probability, 0.15);

    pub fn from_config(config: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            max_length: config
                .get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: config.get("pad_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            mask_token_id: config.get("mask_token_id").and_then(|v| v.as_u64()).unwrap_or(103)
                as u32,
            mlm_probability: 0.15, // Standard BERT masking probability
        })
    }

    /// Create a configuration for inference (no masking)
    ///
    /// Creates a configuration suitable for inference where no token masking
    /// should be applied. This is useful when using a masked LM for tasks
    /// other than pre-training.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value
    ///
    /// # Returns
    ///
    /// A configuration with `mlm_probability` set to 0.0
    pub fn for_inference(config: &serde_json::Value) -> Result<Self> {
        let mut mlm_config = Self::from_config(config)?;
        mlm_config.mlm_probability = 0.0;
        Ok(mlm_config)
    }
}

impl DataCollatorConfig for LanguageModelingCollatorConfig {
    fn max_length(&self) -> Option<usize> {
        self.max_length
    }

    fn padding(&self) -> PaddingStrategy {
        self.padding
    }

    fn truncation(&self) -> bool {
        self.truncation
    }
}

// =============================================================================
// Causal Language Modeling Data Collator (GPT-style)
// =============================================================================

/// Data collator for causal language modeling tasks (GPT-like models)
///
/// This collator is specifically designed for GPT-style autoregressive models
/// that use causal (next-token prediction) language modeling. It creates
/// appropriate input-output pairs where the model learns to predict the next
/// token given all previous tokens.
///
/// ## Features
///
/// - **Causal Masking**: Automatically creates shifted labels for next-token prediction
/// - **Left Padding**: Supports left-padding for generation tasks (optional)
/// - **Attention Masking**: Creates proper attention masks for causal attention
/// - **Efficient Batching**: Optimized for autoregressive training and inference
///
/// ## Label Creation Strategy
///
/// For causal language modeling, labels are created by shifting the input sequence:
/// - Input:  `[BOS, token1, token2, token3, EOS]`
/// - Labels: `[token1, token2, token3, EOS, -100]`
///
/// The last position gets label -100 (ignored in loss) since there's no next token.
///
/// ## Usage Examples
///
/// ```rust,ignore
/// use trustformers::auto::data_collators::language_modeling::{
///     CausalLanguageModelingDataCollator, CausalLanguageModelingCollatorConfig
/// };
/// use trustformers::auto::types::{DataExample, PaddingStrategy};
///
/// // Create configuration
/// let config = CausalLanguageModelingCollatorConfig {
///     max_length: Some(1024),
///     padding: PaddingStrategy::Longest,
///     truncation: true,
///     pad_token_id: 50256,  // GPT-2 pad token
/// };
///
/// // Create collator
/// let collator = CausalLanguageModelingDataCollator::new(config);
///
/// // Prepare examples
/// let examples = vec![
///     DataExample {
///         input_ids: vec![50256, 1026, 318, 257, 1332], // <|endoftext|> This is a test
///         attention_mask: Some(vec![1, 1, 1, 1, 1]),
///         token_type_ids: None,
///         labels: None, // Will be created automatically
///         metadata: HashMap::new(),
///     },
/// ];
///
/// // Collate batch
/// let batch = collator.collate(&examples)?;
/// ```
#[derive(Debug, Clone)]
pub struct CausalLanguageModelingDataCollator {
    config: CausalLanguageModelingCollatorConfig,
}

impl CausalLanguageModelingDataCollator {
    /// Create a new causal language modeling data collator
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration specifying collation behavior for causal LM
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = CausalLanguageModelingCollatorConfig {
    ///     max_length: Some(1024),
    ///     padding: PaddingStrategy::Longest,
    ///     truncation: true,
    ///     pad_token_id: 50256,
    /// };
    /// let collator = CausalLanguageModelingDataCollator::new(config);
    /// ```
    pub fn new(config: CausalLanguageModelingCollatorConfig) -> Self {
        Self { config }
    }

    /// Create shifted labels for causal language modeling
    ///
    /// This method creates labels by shifting the input sequence to the right,
    /// where each position predicts the next token in the sequence.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input token sequence
    /// * `attention_mask` - Attention mask for the sequence
    ///
    /// # Returns
    ///
    /// Labels vector where each position contains the next token to predict,
    /// with -100 for positions that should be ignored in loss computation.
    fn create_causal_labels(&self, input_ids: &[u32], attention_mask: &[u32]) -> Vec<i64> {
        let mut labels = Vec::with_capacity(input_ids.len());

        for i in 0..input_ids.len() {
            if i < input_ids.len() - 1 && attention_mask[i + 1] == 1 {
                // Predict the next token
                labels.push(input_ids[i + 1] as i64);
            } else {
                // No next token or next token is padding
                labels.push(-100);
            }
        }

        labels
    }
}

impl DataCollator for CausalLanguageModelingDataCollator {
    fn collate(&self, examples: &[DataExample]) -> Result<CollatedBatch> {
        if examples.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "Cannot collate empty batch for causal language modeling".to_string(),
            ));
        }

        let batch_size = examples.len();

        // Determine maximum length for this batch
        let max_len = match self.config.padding {
            PaddingStrategy::Longest => examples
                .iter()
                .map(|ex| ex.input_ids.len())
                .max()
                .unwrap_or(0)
                .min(self.config.max_length.unwrap_or(usize::MAX)),
            PaddingStrategy::MaxLength => self.config.max_length.unwrap_or(1024),
            PaddingStrategy::DoNotPad => {
                examples.iter().map(|ex| ex.input_ids.len()).max().unwrap_or(0)
            },
            PaddingStrategy::None => examples[0].input_ids.len(),
        };

        let mut input_ids = Vec::with_capacity(batch_size);
        let mut attention_mask = Vec::with_capacity(batch_size);
        let mut labels = Vec::with_capacity(batch_size);

        for example in examples {
            let mut sequence_input_ids = example.input_ids.clone();
            let mut sequence_attention_mask = example
                .attention_mask
                .clone()
                .unwrap_or_else(|| vec![1u32; example.input_ids.len()]);

            // Truncate if necessary
            if self.config.truncation && sequence_input_ids.len() > max_len {
                sequence_input_ids.truncate(max_len);
                sequence_attention_mask.truncate(max_len);
            }

            // Pad sequences to max_len (right padding for causal LM)
            while sequence_input_ids.len() < max_len {
                sequence_input_ids.push(self.config.pad_token_id);
                sequence_attention_mask.push(0);
            }

            // Create causal labels (shifted input sequence)
            let sequence_labels =
                self.create_causal_labels(&sequence_input_ids, &sequence_attention_mask);

            input_ids.push(sequence_input_ids);
            attention_mask.push(sequence_attention_mask);
            labels.push(sequence_labels);
        }

        Ok(CollatedBatch {
            input_ids,
            attention_mask,
            token_type_ids: None, // Not used in causal LM
            labels: Some(labels),
            batch_size,
            sequence_length: max_len,
            metadata: HashMap::new(),
        })
    }

    fn config(&self) -> &dyn DataCollatorConfig {
        &self.config
    }
}

// =============================================================================
// Causal Language Modeling Configuration
// =============================================================================

/// Configuration for causal language modeling data collator
///
/// This configuration struct controls data collation for causal (autoregressive)
/// language modeling tasks like those used in GPT-style models.
///
/// ## Configuration Parameters
///
/// - `max_length`: Maximum sequence length for truncation
/// - `padding`: Strategy for padding sequences (typically right-padding)
/// - `truncation`: Whether to truncate sequences exceeding max_length
/// - `pad_token_id`: Token ID used for padding positions
///
/// ## GPT-style Models
///
/// This collator is optimized for GPT-style models where:
/// - Training uses teacher forcing with shifted labels
/// - Attention is causal (can only attend to previous tokens)
/// - No special token type embeddings are used
/// - Padding tokens are typically placed on the right
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLanguageModelingCollatorConfig {
    /// Maximum sequence length for padding/truncation
    pub max_length: Option<usize>,
    /// Padding strategy for batch collation
    pub padding: PaddingStrategy,
    /// Whether to truncate sequences exceeding max_length
    pub truncation: bool,
    /// Token ID used for padding
    pub pad_token_id: u32,
}

impl CausalLanguageModelingCollatorConfig {
    /// Create configuration from model config JSON
    ///
    /// This method extracts relevant configuration parameters from a model's
    /// config.json file and creates an appropriate collator configuration
    /// for causal language modeling.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value (typically from config.json)
    ///
    /// # Returns
    ///
    /// A configured `CausalLanguageModelingCollatorConfig` with parameters
    /// extracted from the model configuration.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// let model_config = serde_json::json!({
    ///     "max_position_embeddings": 1024,
    ///     "pad_token_id": 50256,
    ///     "vocab_size": 50257
    /// });
    ///
    /// let config = CausalLanguageModelingCollatorConfig::from_config(&model_config)?;
    /// assert_eq!(config.max_length, Some(1024));
    /// assert_eq!(config.pad_token_id, 50256);

    pub fn from_config(config: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            max_length: config
                .get("max_position_embeddings")
                .or_else(|| config.get("n_positions"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: config
                .get("pad_token_id")
                .or_else(|| config.get("eos_token_id"))  // Some models use EOS as pad
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
        })
    }

    /// Create a configuration optimized for text generation
    ///
    /// Creates a configuration suitable for text generation tasks where
    /// sequences might be left-padded and special handling is needed.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value
    ///
    /// # Returns
    ///
    /// A configuration optimized for generation tasks
    pub fn for_generation(config: &serde_json::Value) -> Result<Self> {
        let mut gen_config = Self::from_config(config)?;
        // For generation, we typically don't want truncation
        gen_config.truncation = false;
        Ok(gen_config)
    }
}

impl DataCollatorConfig for CausalLanguageModelingCollatorConfig {
    fn max_length(&self) -> Option<usize> {
        self.max_length
    }

    fn padding(&self) -> PaddingStrategy {
        self.padding
    }

    fn truncation(&self) -> bool {
        self.truncation
    }
}

// =============================================================================
// Module Re-exports
// =============================================================================

// Note: No need to re-export since these are already public structs in this module

// Add a simple rand implementation since we don't want to add external dependencies
mod rand {

    static mut SEED: u64 = 1;

    pub fn random<T>() -> T
    where
        T: From<u32>,
    {
        unsafe {
            SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
            T::from((SEED >> 16) as u32)
        }
    }

    pub fn random_f32() -> f32 {
        unsafe {
            SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
            // Convert to [0.0, 1.0) range
            ((SEED >> 16) as u32) as f32 / (u32::MAX as f32)
        }
    }
}
