//! # Data Collators for TrustformeRS
//!
//! This module provides the data collation system for the TrustformeRS library.
//! Data collators are responsible for converting individual examples into batches
//! suitable for model training and inference.
//!
//! ## Overview
//!
//! The data collation system follows an extensible architecture where:
//!
//! - **AutoDataCollator**: Automatically selects the appropriate collator based on model type or task
//! - **Base Traits**: Define the common interface for all data collators
//! - **Specific Collators**: Implement task-specific collation logic
//!
//! ## Architecture
//!
//! ```text
//! AutoDataCollator
//!      ├─ from_pretrained() -> Box<dyn DataCollator>
//!      ├─ from_config() -> Box<dyn DataCollator>
//!      └─ for_task() -> Box<dyn DataCollator>
//!               │
//!               ▼
//!        DataCollator Trait
//!      ┌────────┼────────┐
//!      ▼        ▼        ▼
//!  Language  Causal   Seq2Seq  ...
//! Modeling   LM      Collator
//! Collator  Collator
//! ```
//!
//! ## Usage
//!
//! ### Automatic Selection
//!
//! ```rust,ignore
//! use trustformers::auto::data_collators::AutoDataCollator;
//!
//! // From pretrained model
//! let collator = AutoDataCollator::from_pretrained("bert-base-uncased")?;
//!
//! // From configuration
//! let collator = AutoDataCollator::from_config(&config)?;
//!
//! // For specific task
//! let collator = AutoDataCollator::for_task("text-classification", &config)?;
//! ```
//!
//! ### Manual Creation
//!
//! ```rust,ignore
//! use trustformers::auto::data_collators::{
//!     LanguageModelingDataCollator, LanguageModelingCollatorConfig
//! };
//!
//! let config = LanguageModelingCollatorConfig {
//!     max_length: Some(512),
//!     padding: PaddingStrategy::Longest,
//!     truncation: true,
//!     pad_token_id: 0,
//!     mask_token_id: 103,
//!     mlm_probability: 0.15,
//! };
//!
//! let collator = LanguageModelingDataCollator::new(config);
//! ```
//!
//! ### Collating Data
//!
//! ```rust,ignore
//! use trustformers::auto::types::{DataExample, CollatedBatch};
//!
//! let examples = vec![
//!     DataExample::new(vec![101, 2023, 2003, 102]),
//!     DataExample::new(vec![101, 2023, 102]),
//! ];
//!
//! let batch: CollatedBatch = collator.collate(&examples)?;
//! ```
//!
//! ## Supported Tasks
//!
//! | Task | Collator | Description |
//! |------|----------|-------------|
//! | `masked-lm`, `fill-mask` | `LanguageModelingDataCollator` | For BERT-like masked language modeling |
//! | `causal-lm`, `text-generation` | `CausalLanguageModelingDataCollator` | For GPT-like causal language modeling |
//! | `text2text-generation`, `translation`, `summarization` | `Seq2SeqDataCollator` | For T5/BART-like sequence-to-sequence tasks |
//! | `text-classification`, `sentiment-analysis` | `ClassificationDataCollator` | For classification tasks |
//! | `question-answering` | `QuestionAnsweringDataCollator` | For extractive question answering |
//! | Default | `DefaultDataCollator` | Fallback for unknown tasks |
//!
//! ## Extending the System
//!
//! To add a new data collator:
//!
//! 1. Implement the `DataCollator` trait
//! 2. Create a corresponding config struct implementing `DataCollatorConfig`
//! 3. Add the collator to `AutoDataCollator::from_config()` and `AutoDataCollator::for_task()`
//! 4. Re-export the new collator from this module

use crate::auto::types::{CollatedBatch, DataExample, PaddingStrategy};
use crate::error::Result;

// Import all collator modules
pub mod classification;
pub mod default;
pub mod language_modeling;
pub mod question_answering;
pub mod seq2seq;

// Note: Types are imported via pub use statements below for re-export

// =============================================================================
// Auto Data Collator
// =============================================================================

/// Automatically create data collators based on task and data format
///
/// `AutoDataCollator` provides a high-level interface for automatically selecting
/// and creating the appropriate data collator for a given model or task. This follows
/// the same pattern as HuggingFace Transformers' AutoTokenizer but for data collation.
///
/// ## Examples
///
/// ```rust,ignore
/// use trustformers::auto::data_collators::AutoDataCollator;
///
/// // From a pretrained model
/// let collator = AutoDataCollator::from_pretrained("bert-base-uncased")?;
///
/// // From model configuration
/// let config = serde_json::json!({
///     "model_type": "bert",
///     "pad_token_id": 0,
///     "max_position_embeddings": 512
/// });
/// let collator = AutoDataCollator::from_config(&config)?;
///
/// // For a specific task
/// let collator = AutoDataCollator::for_task("text-classification", &config)?;
/// ```
#[derive(Debug, Clone)]
pub struct AutoDataCollator;

impl AutoDataCollator {
    /// Create a data collator from model configuration loaded from the HuggingFace Hub
    ///
    /// This method loads the model configuration from the HuggingFace Hub and
    /// creates an appropriate data collator based on the model type.
    ///
    /// # Arguments
    ///
    /// * `model_name_or_path` - Model name on HuggingFace Hub or local path
    ///
    /// # Returns
    ///
    /// A boxed trait object implementing `DataCollator`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// let collator = AutoDataCollator::from_pretrained("bert-base-uncased")?;

    pub fn from_pretrained(model_name_or_path: &str) -> Result<Box<dyn DataCollator>> {
        let config = crate::hub::load_config_from_hub(model_name_or_path, None)?;
        Self::from_config(&config)
    }

    /// Create a data collator from model configuration
    ///
    /// Automatically selects the appropriate data collator based on the model type
    /// specified in the configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value
    ///
    /// # Returns
    ///
    /// A boxed trait object implementing `DataCollator`
    ///
    /// # Supported Model Types
    ///
    /// - `bert`, `roberta`, `electra` → `LanguageModelingDataCollator`
    /// - `gpt2`, `gpt_neo`, `gpt_j` → `CausalLanguageModelingDataCollator`
    /// - `t5`, `bart`, `pegasus` → `Seq2SeqDataCollator`
    /// - Default → `DefaultDataCollator`
    pub fn from_config(config: &serde_json::Value) -> Result<Box<dyn DataCollator>> {
        let model_type = config.get("model_type").and_then(|v| v.as_str()).unwrap_or("default");

        match model_type {
            "bert" | "roberta" | "electra" => Ok(Box::new(LanguageModelingDataCollator::new(
                LanguageModelingCollatorConfig::from_config(config)?,
            ))),
            "gpt2" | "gpt_neo" | "gpt_j" => Ok(Box::new(CausalLanguageModelingDataCollator::new(
                CausalLanguageModelingCollatorConfig::from_config(config)?,
            ))),
            "t5" | "bart" | "pegasus" => Ok(Box::new(seq2seq::Seq2SeqDataCollator::new(
                seq2seq::Seq2SeqCollatorConfig::from_config(config)?,
            ))),
            _ => Ok(Box::new(default::DefaultDataCollator::new(
                default::DefaultCollatorConfig::from_config(config)?,
            ))),
        }
    }

    /// Create a data collator for a specific task
    ///
    /// Selects the appropriate data collator based on the task type rather than
    /// the model architecture. This is useful when you want to override the
    /// default collator selection.
    ///
    /// # Arguments
    ///
    /// * `task` - The task identifier
    /// * `config` - Model configuration as JSON value
    ///
    /// # Returns
    ///
    /// A boxed trait object implementing `DataCollator`
    ///
    /// # Supported Tasks
    ///
    /// - `masked-lm`, `fill-mask` → `LanguageModelingDataCollator`
    /// - `causal-lm`, `text-generation` → `CausalLanguageModelingDataCollator`
    /// - `text2text-generation`, `translation`, `summarization` → `Seq2SeqDataCollator`
    /// - `text-classification`, `sentiment-analysis` → `ClassificationDataCollator`
    /// - `question-answering` → `QuestionAnsweringDataCollator`
    /// - Default → `DefaultDataCollator`
    pub fn for_task(task: &str, config: &serde_json::Value) -> Result<Box<dyn DataCollator>> {
        match task {
            "masked-lm" | "fill-mask" => Ok(Box::new(LanguageModelingDataCollator::new(
                LanguageModelingCollatorConfig::from_config(config)?,
            ))),
            "causal-lm" | "text-generation" => {
                Ok(Box::new(CausalLanguageModelingDataCollator::new(
                    CausalLanguageModelingCollatorConfig::from_config(config)?,
                )))
            },
            "text2text-generation" | "translation" | "summarization" => {
                Ok(Box::new(seq2seq::Seq2SeqDataCollator::new(
                    seq2seq::Seq2SeqCollatorConfig::from_config(config)?,
                )))
            },
            "text-classification" | "sentiment-analysis" => {
                Ok(Box::new(classification::ClassificationDataCollator::new(
                    classification::ClassificationCollatorConfig::from_config(config)?,
                )))
            },
            "question-answering" => Ok(Box::new(
                question_answering::QuestionAnsweringDataCollator::new(
                    question_answering::QuestionAnsweringCollatorConfig::from_config(config)?,
                ),
            )),
            _ => Ok(Box::new(default::DefaultDataCollator::new(
                default::DefaultCollatorConfig::from_config(config)?,
            ))),
        }
    }
}

// =============================================================================
// Base Traits
// =============================================================================

/// Core trait for data collation functionality
///
/// This trait defines the interface that all data collators must implement.
/// It provides methods for collating examples into batches and managing
/// collator configuration.
///
/// ## Implementation Guidelines
///
/// When implementing this trait:
///
/// 1. **Collation Logic**: Implement `collate()` to handle padding, truncation, and batching
/// 2. **Configuration**: Return a reference to your config struct from `config()`
/// 3. **Preprocessing**: Override `preprocess_examples()` if you need custom preprocessing
///
/// ## Examples
///
/// ```rust,ignore
/// use trustformers::auto::data_collators::{DataCollator, DataCollatorConfig};
/// use trustformers::auto::types::{DataExample, CollatedBatch, PaddingStrategy};
///
/// struct MyDataCollator {
///     config: MyCollatorConfig,
/// }
///
/// impl DataCollator for MyDataCollator {
///     fn collate(&self, examples: &[DataExample]) -> Result<CollatedBatch> {
///         // Implementation here
///         todo!()
///     }
///
///     fn config(&self) -> &dyn DataCollatorConfig {
///         &self.config
///     }
/// }
/// ```
pub trait DataCollator: Send + Sync {
    /// Collate a batch of examples into tensors ready for model consumption
    ///
    /// This is the core method that transforms a slice of individual examples
    /// into a single batched structure with appropriate padding and alignment.
    ///
    /// # Arguments
    ///
    /// * `examples` - Slice of data examples to collate
    ///
    /// # Returns
    ///
    /// A `CollatedBatch` containing the batched and padded data
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The examples slice is empty
    /// - Examples have incompatible formats
    /// - Memory allocation fails during batching
    fn collate(&self, examples: &[DataExample]) -> Result<CollatedBatch>;

    /// Get the collator configuration
    ///
    /// Returns a reference to the configuration object that controls
    /// the collation behavior (padding strategy, max length, etc.).
    fn config(&self) -> &dyn DataCollatorConfig;

    /// Preprocess examples before collation
    ///
    /// This method allows collators to perform custom preprocessing
    /// on examples before the main collation logic runs. The default
    /// implementation returns the examples unchanged.
    ///
    /// # Arguments
    ///
    /// * `examples` - Slice of data examples to preprocess
    ///
    /// # Returns
    ///
    /// A vector of preprocessed examples
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// fn preprocess_examples(&self, examples: &[DataExample]) -> Result<Vec<DataExample>> {
    ///     examples.iter()
    ///         .map(|example| {
    ///             // Apply custom preprocessing
    ///             let mut processed = example.clone();
    ///             processed.input_ids.truncate(self.config().max_length().unwrap_or(512));
    ///             Ok(processed)
    ///         })
    ///         .collect()
    /// }
    /// ```
    fn preprocess_examples(&self, examples: &[DataExample]) -> Result<Vec<DataExample>> {
        Ok(examples.to_vec())
    }
}

/// Configuration trait for data collators
///
/// This trait defines the common configuration parameters that all data
/// collators should support. It provides a uniform interface for querying
/// collation settings.
///
/// ## Implementation Guidelines
///
/// When implementing this trait:
///
/// 1. **Consistency**: Ensure the returned values match your actual collation behavior
/// 2. **Defaults**: Provide sensible defaults for optional parameters
/// 3. **Validation**: Consider validating configuration parameters in your constructor
///
/// ## Examples
///
/// ```rust,ignore
/// use trustformers::auto::data_collators::DataCollatorConfig;
/// use trustformers::auto::types::PaddingStrategy;
///
/// struct MyCollatorConfig {
///     max_length: Option<usize>,
///     padding: PaddingStrategy,
///     truncation: bool,
/// }
///
/// impl DataCollatorConfig for MyCollatorConfig {
///     fn max_length(&self) -> Option<usize> {
///         self.max_length
///     }
///
///     fn padding(&self) -> PaddingStrategy {
///         self.padding
///     }
///
///     fn truncation(&self) -> bool {
///         self.truncation
///     }
/// }
/// ```
pub trait DataCollatorConfig: Send + Sync {
    /// Get the maximum sequence length for padding/truncation
    ///
    /// Returns `None` if no maximum length is specified, in which case
    /// the collator should use dynamic padding based on the longest
    /// sequence in each batch.
    fn max_length(&self) -> Option<usize>;

    /// Get the padding strategy
    ///
    /// Determines how sequences should be padded when creating batches.
    /// See `PaddingStrategy` for available options.
    fn padding(&self) -> PaddingStrategy;

    /// Check if truncation is enabled
    ///
    /// Returns `true` if sequences should be truncated to fit within
    /// the maximum length, `false` otherwise.
    fn truncation(&self) -> bool;
}

// =============================================================================
// Language Modeling Data Collators (Imported from dedicated module)
// =============================================================================

// Language modeling collators are now implemented in the language_modeling module
// for better organization and maintainability. The full implementations include
// comprehensive masking strategies, optimized padding/truncation, and detailed
// documentation for both BERT-style MLM and GPT-style causal LM.

// =============================================================================
// Additional Collator Implementations
// =============================================================================

// All collator implementations have been moved to dedicated modules:
// - language_modeling.rs: LanguageModelingDataCollator, CausalLanguageModelingDataCollator
// - seq2seq.rs: Seq2SeqDataCollator
// - classification.rs: ClassificationDataCollator
// - question_answering.rs: QuestionAnsweringDataCollator
// - default.rs: DefaultDataCollator
//
// This improves code organization and maintainability by separating
// each collator type into its own focused module with comprehensive
// documentation and testing.

// =============================================================================
// Module Re-exports
// =============================================================================

// Note: AutoDataCollator is already public struct in this module

// Re-export collators from dedicated modules
pub use classification::{ClassificationCollatorConfig, ClassificationDataCollator};
pub use default::{DefaultCollatorConfig, DefaultDataCollator};
pub use language_modeling::{
    CausalLanguageModelingCollatorConfig, CausalLanguageModelingDataCollator,
    LanguageModelingCollatorConfig, LanguageModelingDataCollator,
};
pub use question_answering::{QuestionAnsweringCollatorConfig, QuestionAnsweringDataCollator};
pub use seq2seq::{Seq2SeqCollatorConfig, Seq2SeqDataCollator};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto::types::{DataExample, PaddingStrategy};

    // -------------------------------------------------------------------------
    // PaddingStrategy
    // -------------------------------------------------------------------------

    #[test]
    fn test_padding_strategy_none_no_padding() {
        let strategy = PaddingStrategy::None;
        assert!(!strategy.should_pad(), "None should not require padding");
    }

    #[test]
    fn test_padding_strategy_longest_pads() {
        let strategy = PaddingStrategy::Longest;
        assert!(strategy.should_pad(), "Longest should require padding");
    }

    #[test]
    fn test_padding_strategy_max_length_pads() {
        let strategy = PaddingStrategy::MaxLength;
        assert!(strategy.should_pad(), "MaxLength should require padding");
    }

    #[test]
    fn test_padding_strategy_do_not_pad_no_padding() {
        let strategy = PaddingStrategy::DoNotPad;
        assert!(
            !strategy.should_pad(),
            "DoNotPad should not require padding"
        );
    }

    #[test]
    fn test_padding_strategy_longest_is_dynamic() {
        let strategy = PaddingStrategy::Longest;
        assert!(strategy.is_dynamic(), "Longest should be dynamic");
    }

    #[test]
    fn test_padding_strategy_max_length_not_dynamic() {
        let strategy = PaddingStrategy::MaxLength;
        assert!(!strategy.is_dynamic(), "MaxLength should not be dynamic");
    }

    #[test]
    fn test_padding_strategy_none_not_dynamic() {
        let strategy = PaddingStrategy::None;
        assert!(!strategy.is_dynamic(), "None should not be dynamic");
    }

    // -------------------------------------------------------------------------
    // DataExample
    // -------------------------------------------------------------------------

    #[test]
    fn test_data_example_new() {
        let example = DataExample::new(vec![101, 2023, 102]);
        assert_eq!(example.input_ids, vec![101, 2023, 102]);
        assert!(
            example.attention_mask.is_none(),
            "attention_mask should be None by default"
        );
        assert!(example.labels.is_none(), "labels should be None by default");
    }

    #[test]
    fn test_data_example_sequence_length() {
        let example = DataExample::new(vec![101, 2023, 3000, 102]);
        assert_eq!(example.sequence_length(), 4);
    }

    #[test]
    fn test_data_example_with_attention_mask() {
        let example = DataExample::new(vec![101, 102]).with_attention_mask(vec![1, 1]);
        assert!(example.attention_mask.is_some());
        if let Some(mask) = &example.attention_mask {
            assert_eq!(mask, &vec![1, 1]);
        }
    }

    #[test]
    fn test_data_example_with_labels() {
        let example = DataExample::new(vec![101, 102]).with_labels(vec![0]);
        assert!(example.has_labels(), "Example should have labels");
    }

    #[test]
    fn test_data_example_without_labels() {
        let example = DataExample::new(vec![101, 102]);
        assert!(
            !example.has_labels(),
            "Example without labels should return false"
        );
    }

    #[test]
    fn test_data_example_with_token_type_ids() {
        let example = DataExample::new(vec![101, 200, 102]).with_token_type_ids(vec![0, 0, 0]);
        assert!(example.token_type_ids.is_some());
    }

    // -------------------------------------------------------------------------
    // LanguageModelingDataCollator
    // -------------------------------------------------------------------------

    #[test]
    fn test_language_modeling_collator_config_creation() {
        let config = LanguageModelingCollatorConfig {
            max_length: Some(512),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
            mask_token_id: 103,
            mlm_probability: 0.15,
        };
        assert_eq!(config.max_length, Some(512));
        assert_eq!(config.pad_token_id, 0);
        assert_eq!(config.mask_token_id, 103);
        let diff = (config.mlm_probability - 0.15).abs();
        assert!(diff < 1e-6, "mlm_probability should be 0.15");
    }

    #[test]
    fn test_language_modeling_collator_creation() {
        let config = LanguageModelingCollatorConfig {
            max_length: Some(128),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
            mask_token_id: 103,
            mlm_probability: 0.0, // Disable masking for deterministic test
        };
        let collator = LanguageModelingDataCollator::new(config);
        // Verify config accessor
        assert_eq!(collator.config().max_length(), Some(128));
        assert!(collator.config().truncation());
    }

    #[test]
    fn test_language_modeling_collate_single_example() {
        let config = LanguageModelingCollatorConfig {
            max_length: Some(16),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
            mask_token_id: 103,
            mlm_probability: 0.0,
        };
        let collator = LanguageModelingDataCollator::new(config);
        let examples = vec![DataExample::new(vec![101_u32, 2023, 2003, 102])];
        let result = collator.collate(&examples);
        assert!(result.is_ok(), "Collation should succeed");
        if let Ok(batch) = result {
            assert_eq!(batch.batch_size, 1);
            assert_eq!(batch.input_ids[0].len(), batch.sequence_length);
        }
    }

    #[test]
    fn test_language_modeling_collate_pads_shorter_sequence() {
        let config = LanguageModelingCollatorConfig {
            max_length: None,
            padding: PaddingStrategy::Longest,
            truncation: false,
            pad_token_id: 0,
            mask_token_id: 103,
            mlm_probability: 0.0,
        };
        let collator = LanguageModelingDataCollator::new(config);
        let examples = vec![
            DataExample::new(vec![101_u32, 200, 300, 102]),
            DataExample::new(vec![101_u32, 400, 102]),
        ];
        let result = collator.collate(&examples);
        if let Ok(batch) = result {
            assert_eq!(batch.batch_size, 2);
            // All sequences should be same length (padded to longest)
            let len0 = batch.input_ids[0].len();
            let len1 = batch.input_ids[1].len();
            assert_eq!(len0, len1, "All sequences should be padded to same length");
        }
    }

    // -------------------------------------------------------------------------
    // CausalLanguageModelingDataCollator
    // -------------------------------------------------------------------------

    #[test]
    fn test_causal_lm_collator_config_creation() {
        let config = CausalLanguageModelingCollatorConfig {
            max_length: Some(1024),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 50256,
        };
        assert_eq!(config.max_length, Some(1024));
        assert_eq!(config.pad_token_id, 50256);
        assert!(config.truncation);
    }

    #[test]
    fn test_causal_lm_collator_collate_single() {
        let config = CausalLanguageModelingCollatorConfig {
            max_length: Some(32),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
        };
        let collator = CausalLanguageModelingDataCollator::new(config);
        let examples = vec![DataExample::new(vec![1_u32, 2, 3, 4, 5])];
        let result = collator.collate(&examples);
        assert!(result.is_ok(), "Causal LM collation should succeed");
        if let Ok(batch) = result {
            assert_eq!(batch.batch_size, 1);
        }
    }

    // -------------------------------------------------------------------------
    // AutoDataCollator::from_config
    // -------------------------------------------------------------------------

    #[test]
    fn test_auto_collator_from_config_bert() {
        let config = serde_json::json!({
            "model_type": "bert",
            "pad_token_id": 0,
            "mask_token_id": 103,
            "max_position_embeddings": 512
        });
        let result = AutoDataCollator::from_config(&config);
        assert!(
            result.is_ok(),
            "AutoDataCollator::from_config for bert should succeed"
        );
    }

    #[test]
    fn test_auto_collator_from_config_gpt2() {
        let config = serde_json::json!({
            "model_type": "gpt2",
            "pad_token_id": 50256,
            "n_positions": 1024
        });
        let result = AutoDataCollator::from_config(&config);
        assert!(
            result.is_ok(),
            "AutoDataCollator::from_config for gpt2 should succeed"
        );
    }

    #[test]
    fn test_auto_collator_from_config_t5() {
        let config = serde_json::json!({
            "model_type": "t5",
            "pad_token_id": 0
        });
        let result = AutoDataCollator::from_config(&config);
        assert!(
            result.is_ok(),
            "AutoDataCollator::from_config for t5 should succeed"
        );
    }

    #[test]
    fn test_auto_collator_from_config_unknown_uses_default() {
        let config = serde_json::json!({
            "model_type": "custom-model-xyz"
        });
        let result = AutoDataCollator::from_config(&config);
        assert!(
            result.is_ok(),
            "Unknown model type should fall back to DefaultDataCollator"
        );
    }

    // -------------------------------------------------------------------------
    // AutoDataCollator::for_task
    // -------------------------------------------------------------------------

    #[test]
    fn test_auto_collator_for_task_masked_lm() {
        let config = serde_json::json!({"pad_token_id": 0, "mask_token_id": 103});
        let result = AutoDataCollator::for_task("masked-lm", &config);
        assert!(result.is_ok(), "for_task masked-lm should succeed");
    }

    #[test]
    fn test_auto_collator_for_task_causal_lm() {
        let config = serde_json::json!({"pad_token_id": 0});
        let result = AutoDataCollator::for_task("causal-lm", &config);
        assert!(result.is_ok(), "for_task causal-lm should succeed");
    }

    #[test]
    fn test_auto_collator_for_task_text_generation() {
        let config = serde_json::json!({"pad_token_id": 50256});
        let result = AutoDataCollator::for_task("text-generation", &config);
        assert!(result.is_ok(), "for_task text-generation should succeed");
    }

    #[test]
    fn test_auto_collator_for_task_classification() {
        let config = serde_json::json!({"pad_token_id": 0});
        let result = AutoDataCollator::for_task("text-classification", &config);
        assert!(
            result.is_ok(),
            "for_task text-classification should succeed"
        );
    }

    #[test]
    fn test_auto_collator_for_task_question_answering() {
        let config = serde_json::json!({"pad_token_id": 0});
        let result = AutoDataCollator::for_task("question-answering", &config);
        assert!(result.is_ok(), "for_task question-answering should succeed");
    }

    #[test]
    fn test_auto_collator_for_task_summarization() {
        let config = serde_json::json!({"pad_token_id": 0});
        let result = AutoDataCollator::for_task("summarization", &config);
        assert!(result.is_ok(), "for_task summarization should succeed");
    }

    #[test]
    fn test_auto_collator_for_task_translation() {
        let config = serde_json::json!({"pad_token_id": 0});
        let result = AutoDataCollator::for_task("translation", &config);
        assert!(result.is_ok(), "for_task translation should succeed");
    }

    #[test]
    fn test_auto_collator_for_task_unknown_default() {
        let config = serde_json::json!({"pad_token_id": 0});
        let result = AutoDataCollator::for_task("very-unusual-task-xyz", &config);
        assert!(
            result.is_ok(),
            "Unknown task should fall back to DefaultDataCollator"
        );
    }
}
