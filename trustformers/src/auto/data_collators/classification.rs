//! # Classification Data Collators
//!
//! This module provides specialized data collators for text classification tasks.
//! It includes implementations for BERT-style models performing sentiment analysis,
//! topic classification, and other classification tasks.
//!
//! ## Overview
//!
//! Text classification involves training models to assign labels to input texts.
//! This module provides collators that prepare data appropriately for classification
//! tasks, handling label formatting and ensuring proper input representation.
//!
//! ## Architecture
//!
//! ```text
//! Classification Data Collator
//!      ├─ Handles text input sequences
//!      ├─ Manages single or multi-label classifications
//!      ├─ Creates appropriate attention masks
//!      ├─ Formats labels for loss computation
//!      └─ Supports various classification architectures
//! ```
//!
//! ## Features
//!
//! - **Label Formatting**: Converts labels to appropriate format for loss computation
//! - **Multi-label Support**: Handles both single and multi-label classification
//! - **Class Balancing**: Supports weighted loss through metadata
//! - **Sequence Processing**: Standard text preprocessing and padding
//! - **Flexible Architecture**: Works with BERT, RoBERTa, and similar models
//!
//! ## Usage Examples
//!
//! ### Sentiment Analysis
//!
//! ```rust
//! use trustformers::auto::data_collators::classification::{
//!     ClassificationDataCollator, ClassificationCollatorConfig
//! };
//! use trustformers::auto::types::PaddingStrategy;
//!
//! let config = ClassificationCollatorConfig {
//!     max_length: Some(512),
//!     padding: PaddingStrategy::Longest,
//!     truncation: true,
//!     pad_token_id: 0,
//!     num_labels: 3,  // Negative, Neutral, Positive
//! };
//!
//! let collator = ClassificationDataCollator::new(config);
//! ```
//!
//! ### Multi-label Classification
//!
//! ```rust
//! use trustformers::auto::data_collators::classification::{
//!     ClassificationDataCollator, ClassificationCollatorConfig
//! };
//! use trustformers::auto::types::{DataExample, PaddingStrategy};
//!
//! // Configuration for multi-label topic classification
//! let config = ClassificationCollatorConfig {
//!     max_length: Some(256),
//!     padding: PaddingStrategy::Longest,
//!     truncation: true,
//!     pad_token_id: 0,
//!     num_labels: 10,  // Multiple topics can be assigned
//! };
//!
//! let collator = ClassificationDataCollator::new(config);
//!
//! // Example with multiple labels
//! let examples = vec![
//!     DataExample {
//!         input_ids: vec![101, 2023, 2003, 102],  // [CLS] this is [SEP]
//!         attention_mask: Some(vec![1, 1, 1, 1]),
//!         token_type_ids: None,
//!         labels: Some(vec![1, 4]),  // Multiple class labels
//!         metadata: HashMap::new(),
//!     },
//! ];
//!
//! let batch = collator.collate(&examples)?;
//! ```
//!
//! ## Supported Tasks
//!
//! - **Sentiment Analysis**: Positive/Negative/Neutral classification
//! - **Topic Classification**: Document categorization
//! - **Intent Recognition**: Understanding user intents
//! - **Spam Detection**: Email/message filtering
//! - **Language Identification**: Detecting input language
//! - **Multi-label Classification**: Multiple categories per text
//!
//! ## Label Formats
//!
//! The collator supports various label formats:
//! - **Single Label**: One class per example (sentiment analysis)
//! - **Multi-label**: Multiple classes per example (topic tagging)
//! - **Hierarchical**: Nested classification categories
//! - **Weighted**: Different importance weights per class

use super::{DataCollator, DataCollatorConfig};
use crate::auto::data_collators::language_modeling::{
    LanguageModelingCollatorConfig, LanguageModelingDataCollator,
};
use crate::auto::types::{CollatedBatch, DataExample, PaddingStrategy};
use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};

// =============================================================================
// Classification Data Collator
// =============================================================================

/// Data collator for text classification tasks (BERT-like models)
///
/// This collator is specifically designed for classification models that need
/// to assign labels to input text sequences. It handles the formatting of
/// labels and ensures proper input representation for classification training.
///
/// ## Features
///
/// - **Label Processing**: Converts various label formats to model-compatible form
/// - **Single/Multi-label**: Supports both single and multi-label classification
/// - **Class Handling**: Manages different numbers of output classes
/// - **Loss Masking**: Properly handles missing labels with -100 (ignored in loss)
/// - **Sequence Processing**: Standard text tokenization and padding
///
/// ## Label Format
///
/// The collator expects labels in the `DataExample.labels` field:
/// - **Single-label**: `vec![class_id]` (e.g., `vec![2]` for class 2)
/// - **Multi-label**: `vec![class1, class2, ...]` (e.g., `vec![0, 3, 7]`)
/// - **No label**: `None` (will be replaced with -100 for loss computation)
///
/// ## Training vs Inference
///
/// During training, labels are required and formatted appropriately for loss
/// computation. During inference, labels can be omitted and the collator will
/// handle missing labels gracefully.
///
/// ## Usage Examples
///
/// ```rust
/// use trustformers::auto::data_collators::classification::{
///     ClassificationDataCollator, ClassificationCollatorConfig
/// };
/// use trustformers::auto::types::{DataExample, PaddingStrategy};
///
/// // Configuration for binary sentiment classification
/// let config = ClassificationCollatorConfig {
///     max_length: Some(128),
///     padding: PaddingStrategy::Longest,
///     truncation: true,
///     pad_token_id: 0,
///     num_labels: 2,  // Binary: negative (0), positive (1)
/// };
///
/// let collator = ClassificationDataCollator::new(config);
///
/// // Example for sentiment analysis
/// let examples = vec![
///     DataExample {
///         input_ids: vec![101, 1045, 2293, 2023, 102],  // [CLS] I love this [SEP]
///         attention_mask: Some(vec![1, 1, 1, 1, 1]),
///         token_type_ids: None,
///         labels: Some(vec![1]),  // Positive sentiment
///         metadata: HashMap::new(),
///     },
/// ];
///
/// let batch = collator.collate(&examples)?;
/// ```
#[derive(Debug, Clone)]
pub struct ClassificationDataCollator {
    config: ClassificationCollatorConfig,
}

impl ClassificationDataCollator {
    /// Create a new classification data collator
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration specifying collation behavior for classification
    ///
    /// # Examples
    ///
    /// ```rust
    /// let config = ClassificationCollatorConfig {
    ///     max_length: Some(512),
    ///     padding: PaddingStrategy::Longest,
    ///     truncation: true,
    ///     pad_token_id: 0,
    ///     num_labels: 5,
    /// };
    /// let collator = ClassificationDataCollator::new(config);
    /// ```
    pub fn new(config: ClassificationCollatorConfig) -> Self {
        Self { config }
    }

    /// Process labels for classification training
    ///
    /// This method converts the raw labels from examples into the format
    /// expected by classification models. It handles both single-label and
    /// multi-label scenarios.
    ///
    /// # Arguments
    ///
    /// * `examples` - Input examples with labels
    ///
    /// # Returns
    ///
    /// Processed labels formatted for classification loss computation
    fn process_classification_labels(&self, examples: &[DataExample]) -> Vec<Vec<i64>> {
        let mut processed_labels = Vec::with_capacity(examples.len());

        for example in examples {
            if let Some(ref example_labels) = example.labels {
                if example_labels.is_empty() {
                    // No labels provided
                    processed_labels.push(vec![-100]);
                } else if example_labels.len() == 1 {
                    // Single-label classification
                    processed_labels.push(vec![example_labels[0]]);
                } else {
                    // Multi-label classification - take the first label as primary
                    // In a full implementation, this would create a proper multi-label vector
                    processed_labels.push(vec![example_labels[0]]);
                }
            } else {
                // No labels field - use -100 (ignored in loss)
                processed_labels.push(vec![-100]);
            }
        }

        processed_labels
    }

    /// Create one-hot encoded labels for multi-label classification
    ///
    /// This method creates one-hot encoded label vectors for multi-label
    /// classification tasks where each example can belong to multiple classes.
    ///
    /// # Arguments
    ///
    /// * `examples` - Input examples with multi-label annotations
    ///
    /// # Returns
    ///
    /// One-hot encoded label vectors for multi-label classification
    fn create_multilabel_encoding(&self, examples: &[DataExample]) -> Vec<Vec<f32>> {
        let mut multilabel_vectors = Vec::with_capacity(examples.len());

        for example in examples {
            let mut label_vector = vec![0.0f32; self.config.num_labels];

            if let Some(ref example_labels) = example.labels {
                for &label in example_labels {
                    if label >= 0 && (label as usize) < self.config.num_labels {
                        label_vector[label as usize] = 1.0;
                    }
                }
            }

            multilabel_vectors.push(label_vector);
        }

        multilabel_vectors
    }

    /// Validate label consistency across the batch
    ///
    /// This method checks that all labels in the batch are consistent with
    /// the configured number of classes.
    ///
    /// # Arguments
    ///
    /// * `examples` - Input examples to validate
    ///
    /// # Returns
    ///
    /// Result indicating whether labels are valid
    fn validate_labels(&self, examples: &[DataExample]) -> Result<()> {
        for (i, example) in examples.iter().enumerate() {
            if let Some(ref labels) = example.labels {
                for &label in labels {
                    if label >= 0 && (label as usize) >= self.config.num_labels {
                        return Err(TrustformersError::invalid_input_simple(format!(
                            "Label {} in example {} exceeds num_labels {}",
                            label, i, self.config.num_labels
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

impl DataCollator for ClassificationDataCollator {
    fn collate(&self, examples: &[DataExample]) -> Result<CollatedBatch> {
        if examples.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "Cannot collate empty batch for classification".to_string(),
            ));
        }

        // Validate labels before processing
        self.validate_labels(examples)?;

        // Use the base language modeling collator for sequence processing
        // (without MLM masking)
        let sequence_collator = LanguageModelingDataCollator::new(LanguageModelingCollatorConfig {
            max_length: self.config.max_length,
            padding: self.config.padding,
            truncation: self.config.truncation,
            pad_token_id: self.config.pad_token_id,
            mask_token_id: 0, // No masking for classification
            mlm_probability: 0.0,
        });

        // Collate input sequences
        let mut batch = sequence_collator.collate(examples)?;

        // Process classification labels
        let processed_labels = self.process_classification_labels(examples);
        batch.labels = Some(processed_labels);

        // Add classification-specific metadata
        batch.metadata.insert(
            "num_labels".to_string(),
            serde_json::Value::Number(self.config.num_labels.into()),
        );

        // Check if this is a multi-label task
        let is_multilabel = examples
            .iter()
            .any(|ex| ex.labels.as_ref().is_some_and(|labels| labels.len() > 1));

        if is_multilabel {
            let multilabel_encoding = self.create_multilabel_encoding(examples);
            batch.metadata.insert(
                "multilabel_targets".to_string(),
                serde_json::to_value(multilabel_encoding)
                    .map_err(|e| TrustformersError::runtime_error(e.to_string()))?,
            );
            batch.metadata.insert(
                "task_type".to_string(),
                serde_json::Value::String("multilabel".to_string()),
            );
        } else {
            batch.metadata.insert(
                "task_type".to_string(),
                serde_json::Value::String("single_label".to_string()),
            );
        }

        Ok(batch)
    }

    fn config(&self) -> &dyn DataCollatorConfig {
        &self.config
    }

    fn preprocess_examples(&self, examples: &[DataExample]) -> Result<Vec<DataExample>> {
        // For classification, we might want to add preprocessing like
        // text normalization, but for now just return as-is
        Ok(examples.to_vec())
    }
}

// =============================================================================
// Classification Configuration
// =============================================================================

/// Configuration for classification data collator
///
/// This configuration struct controls all aspects of data collation for
/// text classification tasks. It extends the basic collation parameters
/// with classification-specific settings like the number of output classes.
///
/// ## Configuration Parameters
///
/// - `max_length`: Maximum sequence length for padding/truncation
/// - `padding`: Strategy for padding sequences in batch
/// - `truncation`: Whether to truncate sequences exceeding max_length
/// - `pad_token_id`: Token ID used for padding positions
/// - `num_labels`: Number of classification classes
///
/// ## Classification Types
///
/// The collator supports various classification scenarios:
/// - **Binary Classification**: `num_labels = 2` (e.g., spam detection)
/// - **Multi-class**: `num_labels > 2` (e.g., topic classification)
/// - **Multi-label**: Multiple labels per example (via label formatting)
///
/// ## Default Values
///
/// The `from_config` method provides sensible defaults:
/// - `num_labels`: 2 (binary classification)
/// - `padding`: Longest sequence in batch
/// - `truncation`: Enabled
/// - `pad_token_id`: From model config or 0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationCollatorConfig {
    /// Maximum sequence length for padding/truncation
    pub max_length: Option<usize>,
    /// Padding strategy for batch collation
    pub padding: PaddingStrategy,
    /// Whether to truncate sequences exceeding max_length
    pub truncation: bool,
    /// Token ID used for padding
    pub pad_token_id: u32,
    /// Number of classification labels/classes
    pub num_labels: usize,
}

impl ClassificationCollatorConfig {
    /// Create configuration from model config JSON
    ///
    /// This method extracts relevant configuration parameters from a model's
    /// config.json file and creates an appropriate collator configuration
    /// for classification tasks.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value (typically from config.json)
    ///
    /// # Returns
    ///
    /// A configured `ClassificationCollatorConfig` with parameters extracted
    /// from the model configuration and classification-specific defaults.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let model_config = serde_json::json!({
    ///     "max_position_embeddings": 512,
    ///     "pad_token_id": 0,
    ///     "num_labels": 3,
    ///     "vocab_size": 30522
    /// });
    ///
    /// let config = ClassificationCollatorConfig::from_config(&model_config)?;
    /// assert_eq!(config.max_length, Some(512));
    /// assert_eq!(config.pad_token_id, 0);
    /// assert_eq!(config.num_labels, 3);
    /// ```
    pub fn from_config(config: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            max_length: config
                .get("max_position_embeddings")
                .or_else(|| config.get("max_length"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: config.get("pad_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            num_labels: config.get("num_labels").and_then(|v| v.as_u64()).unwrap_or(2) as usize,
        })
    }

    /// Create a configuration for binary classification
    ///
    /// Creates a configuration specifically for binary classification tasks
    /// like sentiment analysis (positive/negative) or spam detection.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value
    ///
    /// # Returns
    ///
    /// A configuration optimized for binary classification
    pub fn for_binary_classification(config: &serde_json::Value) -> Result<Self> {
        let mut binary_config = Self::from_config(config)?;
        binary_config.num_labels = 2;
        Ok(binary_config)
    }

    /// Create a configuration for multi-label classification
    ///
    /// Creates a configuration for multi-label classification where each
    /// example can belong to multiple classes simultaneously.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value
    /// * `num_labels` - Number of possible labels
    ///
    /// # Returns
    ///
    /// A configuration optimized for multi-label classification
    pub fn for_multilabel_classification(
        config: &serde_json::Value,
        num_labels: usize,
    ) -> Result<Self> {
        let mut multilabel_config = Self::from_config(config)?;
        multilabel_config.num_labels = num_labels;
        Ok(multilabel_config)
    }

    /// Create a configuration for sentiment analysis
    ///
    /// Creates a configuration specifically for sentiment analysis tasks
    /// with common settings for positive/negative/neutral classification.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value
    /// * `include_neutral` - Whether to include neutral class (3-class) or not (2-class)
    ///
    /// # Returns
    ///
    /// A configuration optimized for sentiment analysis
    pub fn for_sentiment_analysis(
        config: &serde_json::Value,
        include_neutral: bool,
    ) -> Result<Self> {
        let mut sentiment_config = Self::from_config(config)?;
        sentiment_config.num_labels = if include_neutral { 3 } else { 2 };
        Ok(sentiment_config)
    }
}

impl DataCollatorConfig for ClassificationCollatorConfig {
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
    fn test_classification_collator_creation() {
        let config = ClassificationCollatorConfig {
            max_length: Some(128),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
            num_labels: 3,
        };

        let collator = ClassificationDataCollator::new(config);
        assert_eq!(collator.config().max_length(), Some(128));
        assert_eq!(collator.config.num_labels, 3);
    }

    #[test]
    fn test_classification_config_from_json() {
        let config_json = serde_json::json!({
            "max_position_embeddings": 512,
            "pad_token_id": 1,
            "num_labels": 5,
            "vocab_size": 30522
        });

        let config = ClassificationCollatorConfig::from_config(&config_json).unwrap();
        assert_eq!(config.max_length, Some(512));
        assert_eq!(config.pad_token_id, 1);
        assert_eq!(config.num_labels, 5);
    }

    #[test]
    fn test_collate_classification_examples() {
        let config = ClassificationCollatorConfig {
            max_length: Some(10),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
            num_labels: 2,
        };

        let collator = ClassificationDataCollator::new(config);

        let examples = vec![
            DataExample {
                input_ids: vec![101, 2023, 2003, 102], // [CLS] this is [SEP]
                attention_mask: Some(vec![1, 1, 1, 1]),
                token_type_ids: None,
                labels: Some(vec![1]), // Positive class
                metadata: HashMap::new(),
            },
            DataExample {
                input_ids: vec![101, 2025, 102], // [CLS] bad [SEP]
                attention_mask: Some(vec![1, 1, 1]),
                token_type_ids: None,
                labels: Some(vec![0]), // Negative class
                metadata: HashMap::new(),
            },
        ];

        let batch = collator.collate(&examples).unwrap();
        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.input_ids.len(), 2);
        assert!(batch.labels.is_some());

        let labels = batch.labels.as_ref().unwrap();
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], vec![1]);
        assert_eq!(labels[1], vec![0]);
    }

    #[test]
    fn test_multilabel_encoding() {
        let config = ClassificationCollatorConfig {
            max_length: Some(10),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
            num_labels: 5,
        };

        let collator = ClassificationDataCollator::new(config);

        let examples = vec![DataExample {
            input_ids: vec![101, 1037, 3231, 102],
            attention_mask: Some(vec![1, 1, 1, 1]),
            token_type_ids: None,
            labels: Some(vec![0, 2, 4]), // Multi-label
            metadata: HashMap::new(),
        }];

        let multilabel_encoding = collator.create_multilabel_encoding(&examples);
        assert_eq!(multilabel_encoding.len(), 1);
        assert_eq!(multilabel_encoding[0].len(), 5);
        assert_eq!(multilabel_encoding[0][0], 1.0);
        assert_eq!(multilabel_encoding[0][1], 0.0);
        assert_eq!(multilabel_encoding[0][2], 1.0);
        assert_eq!(multilabel_encoding[0][3], 0.0);
        assert_eq!(multilabel_encoding[0][4], 1.0);
    }

    #[test]
    fn test_sentiment_analysis_config() {
        let model_config = serde_json::json!({
            "max_position_embeddings": 128,
            "pad_token_id": 0
        });

        let config =
            ClassificationCollatorConfig::for_sentiment_analysis(&model_config, true).unwrap();
        assert_eq!(config.num_labels, 3); // Negative, Neutral, Positive

        let config =
            ClassificationCollatorConfig::for_sentiment_analysis(&model_config, false).unwrap();
        assert_eq!(config.num_labels, 2); // Negative, Positive
    }

    #[test]
    fn test_label_validation() {
        let config = ClassificationCollatorConfig {
            max_length: Some(10),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
            num_labels: 2, // Only 2 classes allowed
        };

        let collator = ClassificationDataCollator::new(config);

        // Valid labels
        let valid_examples = vec![DataExample {
            input_ids: vec![101, 102],
            attention_mask: Some(vec![1, 1]),
            token_type_ids: None,
            labels: Some(vec![1]),
            metadata: HashMap::new(),
        }];

        assert!(collator.validate_labels(&valid_examples).is_ok());

        // Invalid labels (class 2 doesn't exist with num_labels=2)
        let invalid_examples = vec![DataExample {
            input_ids: vec![101, 102],
            attention_mask: Some(vec![1, 1]),
            token_type_ids: None,
            labels: Some(vec![2]), // Invalid: only 0,1 allowed
            metadata: HashMap::new(),
        }];

        assert!(collator.validate_labels(&invalid_examples).is_err());
    }
}
