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
