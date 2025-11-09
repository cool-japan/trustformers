//! # Sequence-to-Sequence Data Collators
//!
//! This module provides specialized data collators for sequence-to-sequence tasks.
//! It includes implementations for T5/BART-style models that perform text-to-text
//! generation, translation, summarization, and other seq2seq tasks.
//!
//! ## Overview
//!
//! Sequence-to-sequence learning involves training models to map input sequences
//! to output sequences. This module provides collators that prepare data
//! appropriately for these tasks, handling both encoder inputs and decoder targets.
//!
//! ## Architecture
//!
//! ```text
//! Seq2Seq Data Collator
//!      ├─ Handles encoder input sequences
//!      ├─ Manages decoder target sequences
//!      ├─ Supports different maximum lengths for input/target
//!      ├─ Creates appropriate attention masks
//!      └─ Handles teacher forcing for training
//! ```
//!
//! ## Features
//!
//! - **Dual Sequence Handling**: Supports separate input and target sequences
//! - **Length Management**: Different max lengths for encoder and decoder
//! - **Teacher Forcing**: Prepares decoder inputs for training
//! - **Attention Masking**: Creates proper masks for encoder-decoder attention
//! - **Label Shifting**: Handles label shifting for decoder training
//!
//! ## Usage Examples
//!
//! ### Translation Task
//!
//! ```rust
//! use trustformers::auto::data_collators::seq2seq::{
//!     Seq2SeqDataCollator, Seq2SeqCollatorConfig
//! };
//! use trustformers::auto::types::PaddingStrategy;
//!
//! let config = Seq2SeqCollatorConfig {
//!     max_length: Some(512),
//!     max_target_length: Some(256),
//!     padding: PaddingStrategy::Longest,
//!     truncation: true,
//!     pad_token_id: 0,
//! };
//!
//! let collator = Seq2SeqDataCollator::new(config);
//! ```
//!
//! ### Summarization Task
//!
//! ```rust
//! use trustformers::auto::data_collators::seq2seq::{
//!     Seq2SeqDataCollator, Seq2SeqCollatorConfig
//! };
//! use trustformers::auto::types::{DataExample, PaddingStrategy};
//!
//! // Create configuration for summarization
//! let config = Seq2SeqCollatorConfig {
//!     max_length: Some(1024),        // Long input documents
//!     max_target_length: Some(128),  // Short summaries
//!     padding: PaddingStrategy::Longest,
//!     truncation: true,
//!     pad_token_id: 0,
//! };
//!
//! let collator = Seq2SeqDataCollator::new(config);
//!
//! // Prepare examples with both input and target
//! let examples = vec![
//!     DataExample {
//!         input_ids: vec![0, 42, 15, 16, 2],  // Source: <s> Hello world </s>
//!         attention_mask: Some(vec![1, 1, 1, 1, 1]),
//!         token_type_ids: None,
//!         labels: Some(vec![42, 372, 2, -100]),  // Target: Hello there </s> <pad>
//!         metadata: HashMap::new(),
//!     },
//! ];
//!
//! let batch = collator.collate(&examples)?;
//! ```
//!
//! ## Supported Tasks
//!
//! - **Translation**: Language-to-language translation
//! - **Summarization**: Document to summary generation
//! - **Text-to-Text Generation**: General seq2seq tasks
//! - **Question Generation**: Generating questions from context
//! - **Paraphrasing**: Text reformulation tasks
//!
//! ## Training vs Inference
//!
//! During training, the collator prepares both encoder inputs and decoder targets
//! with appropriate shifting for teacher forcing. During inference, only encoder
//! inputs are typically needed as the decoder generates outputs autoregressively.

use super::{DataCollator, DataCollatorConfig};
use crate::auto::data_collators::language_modeling::{
    LanguageModelingCollatorConfig, LanguageModelingDataCollator,
};
use crate::auto::types::{CollatedBatch, DataExample, PaddingStrategy};
use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};

// =============================================================================
// Sequence-to-Sequence Data Collator
// =============================================================================

/// Data collator for sequence-to-sequence tasks (T5/BART-like models)
///
/// This collator is specifically designed for encoder-decoder models that perform
/// sequence-to-sequence learning such as T5, BART, and similar architectures.
/// It handles the complexities of preparing both encoder inputs and decoder targets
/// for training and inference.
///
/// ## Features
///
/// - **Encoder-Decoder Support**: Handles both input and target sequences
/// - **Flexible Length Limits**: Different max lengths for input and target
/// - **Teacher Forcing**: Prepares decoder inputs for training
/// - **Label Management**: Creates appropriate labels for generation training
/// - **Attention Masking**: Generates proper attention masks for both sequences
///
/// ## Data Format
///
/// The collator expects examples with:
/// - `input_ids`: Encoder input tokens (source sequence)
/// - `labels`: Decoder target tokens (target sequence)
/// - `attention_mask`: Attention mask for encoder (optional)
///
/// ## Training Strategy
///
/// During training, the collator:
/// 1. Prepares encoder inputs with padding and attention masks
/// 2. Creates decoder inputs by shifting target labels (teacher forcing)
/// 3. Sets up appropriate loss masks to ignore padding tokens
/// 4. Handles different sequence lengths for encoder and decoder
///
/// ## Usage Examples
///
/// ```rust
/// use trustformers::auto::data_collators::seq2seq::{
///     Seq2SeqDataCollator, Seq2SeqCollatorConfig
/// };
/// use trustformers::auto::types::{DataExample, PaddingStrategy};
///
/// // Configuration for T5-style model
/// let config = Seq2SeqCollatorConfig {
///     max_length: Some(512),
///     max_target_length: Some(64),
///     padding: PaddingStrategy::Longest,
///     truncation: true,
///     pad_token_id: 0,
/// };
///
/// let collator = Seq2SeqDataCollator::new(config);
///
/// // Example for translation
/// let examples = vec![
///     DataExample {
///         input_ids: vec![21820, 10, 86, 5, 1], // "translate: This is a test"
///         attention_mask: Some(vec![1, 1, 1, 1, 1]),
///         token_type_ids: None,
///         labels: Some(vec![100, 19, 3, 9, 794, 1]), // "Das ist ein Test"
///         metadata: HashMap::new(),
///     },
/// ];
///
/// let batch = collator.collate(&examples)?;
/// ```
#[derive(Debug, Clone)]
pub struct Seq2SeqDataCollator {
    config: Seq2SeqCollatorConfig,
}

impl Seq2SeqDataCollator {
    /// Create a new sequence-to-sequence data collator
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration specifying collation behavior for seq2seq tasks
    ///
    /// # Examples
    ///
    /// ```rust
    /// let config = Seq2SeqCollatorConfig {
    ///     max_length: Some(512),
    ///     max_target_length: Some(128),
    ///     padding: PaddingStrategy::Longest,
    ///     truncation: true,
    ///     pad_token_id: 0,
    /// };
    /// let collator = Seq2SeqDataCollator::new(config);
    /// ```
    pub fn new(config: Seq2SeqCollatorConfig) -> Self {
        Self { config }
    }

    /// Prepare decoder inputs from target labels for teacher forcing
    ///
    /// In sequence-to-sequence training, decoder inputs are typically the target
    /// sequence shifted by one position. This method creates appropriate decoder
    /// inputs for teacher forcing during training.
    ///
    /// # Arguments
    ///
    /// * `target_labels` - Target sequence labels
    /// * `bos_token_id` - Beginning-of-sequence token ID (if applicable)
    ///
    /// # Returns
    ///
    /// Decoder input sequence prepared for teacher forcing
    fn prepare_decoder_inputs(&self, target_labels: &[i64], bos_token_id: Option<u32>) -> Vec<u32> {
        let mut decoder_inputs = Vec::with_capacity(target_labels.len());

        // Add BOS token if specified
        if let Some(bos_id) = bos_token_id {
            decoder_inputs.push(bos_id);
        }

        // Shift target labels to create decoder inputs (exclude last token)
        for &label in target_labels.iter().take(target_labels.len().saturating_sub(1)) {
            if label != -100 {
                decoder_inputs.push(label as u32);
            }
        }

        decoder_inputs
    }

    /// Process target sequences for decoder training
    ///
    /// This method handles the target sequences, ensuring they are properly
    /// formatted for decoder training with appropriate padding and truncation.
    ///
    /// # Arguments
    ///
    /// * `examples` - Input examples with target labels
    /// * `max_target_len` - Maximum target sequence length
    ///
    /// # Returns
    ///
    /// Processed target sequences and decoder attention masks
    fn process_target_sequences(
        &self,
        examples: &[DataExample],
        max_target_len: usize,
    ) -> Result<(Vec<Vec<i64>>, Vec<Vec<u32>>)> {
        let mut processed_labels = Vec::with_capacity(examples.len());
        let mut decoder_attention_masks = Vec::with_capacity(examples.len());

        for example in examples {
            if let Some(ref labels) = example.labels {
                let mut sequence_labels = labels.clone();

                // Truncate target sequence if necessary
                if self.config.truncation && sequence_labels.len() > max_target_len {
                    sequence_labels.truncate(max_target_len);
                }

                // Create attention mask for target sequence
                let mut attention_mask = vec![1u32; sequence_labels.len()];

                // Pad target sequence
                while sequence_labels.len() < max_target_len {
                    sequence_labels.push(-100); // Ignore padded positions in loss
                    attention_mask.push(0);
                }

                processed_labels.push(sequence_labels);
                decoder_attention_masks.push(attention_mask);
            } else {
                // No target labels provided
                processed_labels.push(vec![-100i64; max_target_len]);
                decoder_attention_masks.push(vec![0u32; max_target_len]);
            }
        }

        Ok((processed_labels, decoder_attention_masks))
    }
}

impl DataCollator for Seq2SeqDataCollator {
    fn collate(&self, examples: &[DataExample]) -> Result<CollatedBatch> {
        if examples.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "Cannot collate empty batch for sequence-to-sequence".to_string(),
            ));
        }

        let batch_size = examples.len();

        // Determine maximum lengths for encoder and decoder
        let max_encoder_len = match self.config.padding {
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

        let max_decoder_len = match self.config.padding {
            PaddingStrategy::Longest => examples
                .iter()
                .filter_map(|ex| ex.labels.as_ref())
                .map(|labels| labels.len())
                .max()
                .unwrap_or(0)
                .min(self.config.max_target_length.unwrap_or(usize::MAX)),
            PaddingStrategy::MaxLength => self.config.max_target_length.unwrap_or(max_encoder_len),
            PaddingStrategy::DoNotPad => examples
                .iter()
                .filter_map(|ex| ex.labels.as_ref())
                .map(|labels| labels.len())
                .max()
                .unwrap_or(0),
            PaddingStrategy::None => examples
                .first()
                .and_then(|ex| ex.labels.as_ref())
                .map(|labels| labels.len())
                .unwrap_or(0),
        };

        // Use the base language modeling collator for encoder sequences
        let encoder_collator = LanguageModelingDataCollator::new(LanguageModelingCollatorConfig {
            max_length: Some(max_encoder_len),
            padding: self.config.padding,
            truncation: self.config.truncation,
            pad_token_id: self.config.pad_token_id,
            mask_token_id: 0, // No masking for seq2seq encoder
            mlm_probability: 0.0,
        });

        // Collate encoder sequences
        let mut batch = encoder_collator.collate(examples)?;

        // Process target sequences for decoder
        let (processed_labels, decoder_attention_masks) =
            self.process_target_sequences(examples, max_decoder_len)?;

        // Store processed labels
        batch.labels = Some(processed_labels);

        // Add decoder attention masks to metadata
        batch.metadata.insert(
            "decoder_attention_mask".to_string(),
            serde_json::to_value(decoder_attention_masks)
                .map_err(|e| TrustformersError::runtime_error(e.to_string()))?,
        );

        // Add target sequence length to metadata
        batch.metadata.insert(
            "target_sequence_length".to_string(),
            serde_json::Value::Number(max_decoder_len.into()),
        );

        Ok(batch)
    }

    fn config(&self) -> &dyn DataCollatorConfig {
        &self.config
    }

    fn preprocess_examples(&self, examples: &[DataExample]) -> Result<Vec<DataExample>> {
        // For seq2seq, we might want to add special preprocessing
        // like adding task prefixes (e.g., "translate English to German: ")
        // For now, just return examples as-is
        Ok(examples.to_vec())
    }
}

// =============================================================================
// Sequence-to-Sequence Configuration
// =============================================================================

/// Configuration for sequence-to-sequence data collator
///
/// This configuration struct controls all aspects of data collation for
/// sequence-to-sequence tasks. It extends the basic collation parameters
/// with seq2seq-specific settings like separate target length limits.
///
/// ## Configuration Parameters
///
/// - `max_length`: Maximum encoder sequence length (source)
/// - `max_target_length`: Maximum decoder sequence length (target)
/// - `padding`: Strategy for padding sequences in batch
/// - `truncation`: Whether to truncate sequences exceeding max lengths
/// - `pad_token_id`: Token ID used for padding positions
///
/// ## Length Management
///
/// The collator supports different maximum lengths for encoder and decoder:
/// - `max_length`: Controls input sequence length (encoder)
/// - `max_target_length`: Controls target sequence length (decoder)
///
/// This is particularly useful for tasks like summarization where inputs
/// are typically much longer than outputs.
///
/// ## Default Values
///
/// The `from_config` method provides sensible defaults:
/// - `max_target_length`: Same as `max_length` if not specified
/// - `padding`: Longest sequence in batch
/// - `truncation`: Enabled
/// - `pad_token_id`: From model config or 0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Seq2SeqCollatorConfig {
    /// Maximum encoder sequence length for padding/truncation
    pub max_length: Option<usize>,
    /// Maximum decoder sequence length for target padding/truncation
    pub max_target_length: Option<usize>,
    /// Padding strategy for batch collation
    pub padding: PaddingStrategy,
    /// Whether to truncate sequences exceeding max lengths
    pub truncation: bool,
    /// Token ID used for padding
    pub pad_token_id: u32,
}

impl Seq2SeqCollatorConfig {
    /// Create configuration from model config JSON
    ///
    /// This method extracts relevant configuration parameters from a model's
    /// config.json file and creates an appropriate collator configuration
    /// for sequence-to-sequence tasks.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value (typically from config.json)
    ///
    /// # Returns
    ///
    /// A configured `Seq2SeqCollatorConfig` with parameters extracted
    /// from the model configuration and seq2seq-specific defaults.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let model_config = serde_json::json!({
    ///     "max_position_embeddings": 512,
    ///     "max_target_length": 128,
    ///     "pad_token_id": 0,
    ///     "vocab_size": 32128
    /// });
    ///
    /// let config = Seq2SeqCollatorConfig::from_config(&model_config)?;
    /// assert_eq!(config.max_length, Some(512));
    /// assert_eq!(config.max_target_length, Some(128));
    /// assert_eq!(config.pad_token_id, 0);
    /// ```
    pub fn from_config(config: &serde_json::Value) -> Result<Self> {
        let max_length = config
            .get("max_position_embeddings")
            .or_else(|| config.get("max_length"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let max_target_length = config
            .get("max_target_length")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or(max_length); // Default to same as max_length

        Ok(Self {
            max_length,
            max_target_length,
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: config.get("pad_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        })
    }

    /// Create a configuration optimized for translation tasks
    ///
    /// Creates a configuration with settings commonly used for translation
    /// where source and target sequences are typically of similar length.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value
    ///
    /// # Returns
    ///
    /// A configuration optimized for translation tasks
    pub fn for_translation(config: &serde_json::Value) -> Result<Self> {
        let mut translation_config = Self::from_config(config)?;
        // For translation, keep similar lengths for source and target
        if translation_config.max_target_length.is_none() {
            translation_config.max_target_length = translation_config.max_length;
        }
        Ok(translation_config)
    }

    /// Create a configuration optimized for summarization tasks
    ///
    /// Creates a configuration with settings commonly used for summarization
    /// where target sequences (summaries) are typically much shorter than
    /// source sequences (documents).
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value
    /// * `summary_ratio` - Ratio of summary length to document length (default: 0.25)
    ///
    /// # Returns
    ///
    /// A configuration optimized for summarization tasks
    pub fn for_summarization(
        config: &serde_json::Value,
        summary_ratio: Option<f32>,
    ) -> Result<Self> {
        let mut summarization_config = Self::from_config(config)?;
        let ratio = summary_ratio.unwrap_or(0.25);

        if let Some(max_len) = summarization_config.max_length {
            summarization_config.max_target_length = Some((max_len as f32 * ratio) as usize);
        }

        Ok(summarization_config)
    }
}

impl DataCollatorConfig for Seq2SeqCollatorConfig {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_seq2seq_collator_creation() {
        let config = Seq2SeqCollatorConfig {
            max_length: Some(512),
            max_target_length: Some(128),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
        };

        let collator = Seq2SeqDataCollator::new(config);
        assert_eq!(collator.config().max_length(), Some(512));
        assert_eq!(collator.config.max_target_length, Some(128));
    }

    #[test]
    fn test_seq2seq_config_from_json() {
        let config_json = serde_json::json!({
            "max_position_embeddings": 512,
            "max_target_length": 64,
            "pad_token_id": 1,
            "vocab_size": 32000
        });

        let config = Seq2SeqCollatorConfig::from_config(&config_json).unwrap();
        assert_eq!(config.max_length, Some(512));
        assert_eq!(config.max_target_length, Some(64));
        assert_eq!(config.pad_token_id, 1);
    }

    #[test]
    fn test_collate_seq2seq_examples() {
        let config = Seq2SeqCollatorConfig {
            max_length: Some(10),
            max_target_length: Some(8),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
        };

        let collator = Seq2SeqDataCollator::new(config);

        let examples = vec![
            DataExample {
                input_ids: vec![1, 2, 3, 4],
                attention_mask: Some(vec![1, 1, 1, 1]),
                token_type_ids: None,
                labels: Some(vec![5, 6, 7]),
                metadata: HashMap::new(),
            },
            DataExample {
                input_ids: vec![1, 2],
                attention_mask: Some(vec![1, 1]),
                token_type_ids: None,
                labels: Some(vec![8, 9, 10, 11]),
                metadata: HashMap::new(),
            },
        ];

        let batch = collator.collate(&examples).unwrap();
        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.input_ids.len(), 2);
        assert!(batch.labels.is_some());

        let labels = batch.labels.as_ref().unwrap();
        assert_eq!(labels.len(), 2);
    }

    #[test]
    fn test_summarization_config() {
        let model_config = serde_json::json!({
            "max_position_embeddings": 1024,
            "pad_token_id": 0
        });

        let config = Seq2SeqCollatorConfig::for_summarization(&model_config, Some(0.2)).unwrap();
        assert_eq!(config.max_length, Some(1024));
        assert_eq!(config.max_target_length, Some(204)); // 1024 * 0.2 = 204.8 -> 204
    }
}
