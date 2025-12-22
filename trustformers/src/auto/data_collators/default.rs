//! # Default Data Collators
//!
//! This module provides a default/fallback data collator for general-purpose use.
//! It serves as a basic collator when no task-specific collator is available
//! or when working with unknown model types or tasks.
//!
//! ## Overview
//!
//! The default data collator provides basic functionality for text sequence
//! collation without task-specific logic. It handles standard operations like
//! padding, truncation, and attention mask creation, making it suitable as
//! a fallback option for any text-based task.
//!
//! ## Architecture
//!
//! ```text
//! Default Data Collator
//!      ├─ Basic sequence padding and truncation
//!      ├─ Standard attention mask creation
//!      ├─ Token type ID handling (if present)
//!      ├─ No task-specific label processing
//!      └─ Simple pass-through for unknown tasks
//! ```
//!
//! ## Features
//!
//! - **Universal Compatibility**: Works with any text-based task
//! - **Basic Collation**: Standard padding, truncation, and masking
//! - **Minimal Processing**: No task-specific label manipulation
//! - **Flexible Configuration**: Configurable padding and length limits
//! - **Safe Fallback**: Reliable operation for unknown scenarios
//! - **Metadata Preservation**: Maintains input metadata where possible
//!
//! ## Usage Examples
//!
//! ### General Purpose Collation
//!
//! ```rust,ignore
//! use trustformers::auto::data_collators::default::{
//!     DefaultDataCollator, DefaultCollatorConfig
//! };
//! use trustformers::auto::types::PaddingStrategy;
//!
//! let config = DefaultCollatorConfig {
//!     max_length: Some(512),
//!     padding: PaddingStrategy::Longest,
//!     truncation: true,
//!     pad_token_id: 0,
//! };
//!
//! let collator = DefaultDataCollator::new(config);
//! ```
//!
//! ### Unknown Task Handling
//!
//! ```rust,ignore
//! use trustformers::auto::data_collators::default::{
//!     DefaultDataCollator, DefaultCollatorConfig
//! };
//! use trustformers::auto::types::{DataExample, PaddingStrategy};
//!
//! // Configuration for unknown task
//! let config = DefaultCollatorConfig {
//!     max_length: Some(256),
//!     padding: PaddingStrategy::Longest,
//!     truncation: true,
//!     pad_token_id: 0,
//! };
//!
//! let collator = DefaultDataCollator::new(config);
//!
//! // Basic text sequences
//! let examples = vec![
//!     DataExample {
//!         input_ids: vec![101, 2023, 2003, 1037, 3231, 102],
//!         attention_mask: Some(vec![1, 1, 1, 1, 1, 1]),
//!         token_type_ids: None,
//!         labels: None,
//!         metadata: HashMap::new(),
//!     },
//! ];
//!
//! let batch = collator.collate(&examples)?;
//! ```
//!
//! ## Use Cases
//!
//! - **Unknown Tasks**: When task type cannot be determined
//! - **Custom Tasks**: For novel tasks not covered by specific collators
//! - **Prototyping**: Quick setup for experimental work
//! - **Model Testing**: Basic testing without task-specific logic
//! - **Multi-task Models**: When using models for multiple tasks
//! - **Inference Only**: When labels are not needed
//!
//! ## Limitations
//!
//! The default collator has intentional limitations:
//! - **No Label Processing**: Labels are passed through unchanged
//! - **No Task Optimization**: No task-specific optimizations
//! - **Basic Features**: Minimal feature set for broad compatibility
//! - **No Special Tokens**: No automatic special token handling
//!
//! These limitations ensure broad compatibility while encouraging use of
//! task-specific collators when available.

use super::{DataCollator, DataCollatorConfig};
use crate::auto::data_collators::language_modeling::{
    LanguageModelingCollatorConfig, LanguageModelingDataCollator,
};
use crate::auto::types::{CollatedBatch, DataExample, PaddingStrategy};
use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Default Data Collator
// =============================================================================

/// Default data collator for general-purpose text sequence collation
///
/// This collator provides basic functionality for text sequence processing
/// without task-specific optimizations. It serves as a reliable fallback
/// when no specialized collator is available or appropriate.
///
/// ## Features
///
/// - **Basic Collation**: Standard padding, truncation, and attention masks
/// - **Universal Compatibility**: Works with any text sequence format
/// - **Label Preservation**: Passes through labels without modification
/// - **Metadata Handling**: Preserves and forwards metadata appropriately
/// - **Token Type Support**: Handles token type IDs when present
/// - **Configurable Behavior**: Flexible padding and truncation options
///
/// ## Design Philosophy
///
/// The default collator follows a "do no harm" approach:
/// - Minimal assumptions about data format
/// - Conservative processing to avoid data corruption
/// - Broad compatibility over task-specific optimization
/// - Predictable behavior across different input types
///
/// ## When to Use
///
/// - **Unknown Tasks**: When automatic task detection fails
/// - **Custom Applications**: For novel or experimental tasks
/// - **Multi-purpose Models**: When models handle multiple task types
/// - **Development**: During prototyping and testing phases
/// - **Inference**: When no training-specific processing is needed
///
/// ## Usage Examples
///
/// ```rust,ignore
/// use trustformers::auto::data_collators::default::{
///     DefaultDataCollator, DefaultCollatorConfig
/// };
/// use trustformers::auto::types::{DataExample, PaddingStrategy};
///
/// // Basic configuration
/// let config = DefaultCollatorConfig {
///     max_length: Some(512),
///     padding: PaddingStrategy::Longest,
///     truncation: true,
///     pad_token_id: 0,
/// };
///
/// let collator = DefaultDataCollator::new(config);
///
/// // General text sequence
/// let examples = vec![
///     DataExample {
///         input_ids: vec![1, 2, 3, 4, 5],
///         attention_mask: Some(vec![1, 1, 1, 1, 1]),
///         token_type_ids: None,
///         labels: Some(vec![1, 0, 1]), // Arbitrary labels
///         metadata: HashMap::new(),
///     },
/// ];
///
/// let batch = collator.collate(&examples)?;
/// ```
#[derive(Debug, Clone)]
pub struct DefaultDataCollator {
    config: DefaultCollatorConfig,
}

impl DefaultDataCollator {
    /// Create a new default data collator
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration specifying basic collation behavior
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = DefaultCollatorConfig {
    ///     max_length: Some(256),
    ///     padding: PaddingStrategy::Longest,
    ///     truncation: true,
    ///     pad_token_id: 0,
    /// };
    /// let collator = DefaultDataCollator::new(config);
    /// ```
    pub fn new(config: DefaultCollatorConfig) -> Self {
        Self { config }
    }

    /// Process labels with minimal modification
    ///
    /// The default collator passes labels through with minimal processing,
    /// ensuring compatibility with various label formats while avoiding
    /// task-specific assumptions.
    ///
    /// # Arguments
    ///
    /// * `examples` - Input examples with potential labels
    ///
    /// # Returns
    ///
    /// Labels formatted for general use, or None if no labels present
    fn process_generic_labels(&self, examples: &[DataExample]) -> Option<Vec<Vec<i64>>> {
        let has_labels = examples.iter().any(|ex| ex.labels.is_some());

        if !has_labels {
            return None;
        }

        let mut processed_labels = Vec::with_capacity(examples.len());

        for example in examples {
            if let Some(ref example_labels) = example.labels {
                // Pass through labels as-is, ensuring they're in the right format
                processed_labels.push(example_labels.clone());
            } else {
                // No labels for this example
                processed_labels.push(vec![-100]); // Standard "ignore" label
            }
        }

        Some(processed_labels)
    }

    /// Create basic metadata for the batch
    ///
    /// This method creates minimal metadata that describes the basic
    /// properties of the collated batch without making task-specific
    /// assumptions.
    ///
    /// # Arguments
    ///
    /// * `examples` - Input examples to analyze
    /// * `batch_size` - Size of the collated batch
    /// * `sequence_length` - Length of sequences in the batch
    ///
    /// # Returns
    ///
    /// Basic metadata about the batch
    fn create_basic_metadata(
        &self,
        examples: &[DataExample],
        batch_size: usize,
        sequence_length: usize,
    ) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();

        // Basic batch information
        metadata.insert(
            "collator_type".to_string(),
            serde_json::Value::String("default".to_string()),
        );

        // Check if token type IDs are present
        let has_token_types = examples.iter().any(|ex| ex.token_type_ids.is_some());
        metadata.insert(
            "has_token_type_ids".to_string(),
            serde_json::Value::Bool(has_token_types),
        );

        // Check if labels are present
        let has_labels = examples.iter().any(|ex| ex.labels.is_some());
        metadata.insert(
            "has_labels".to_string(),
            serde_json::Value::Bool(has_labels),
        );

        // Sequence statistics
        let input_lengths: Vec<usize> = examples.iter().map(|ex| ex.input_ids.len()).collect();

        if !input_lengths.is_empty() {
            let min_length = *input_lengths.iter().min().unwrap();
            let max_length = *input_lengths.iter().max().unwrap();
            let avg_length =
                input_lengths.iter().sum::<usize>() as f64 / input_lengths.len() as f64;

            metadata.insert(
                "original_sequence_stats".to_string(),
                serde_json::json!({
                    "min_length": min_length,
                    "max_length": max_length,
                    "avg_length": avg_length,
                    "total_sequences": input_lengths.len()
                }),
            );
        }

        // Configuration info
        metadata.insert(
            "padding_strategy".to_string(),
            serde_json::Value::String(format!("{:?}", self.config.padding)),
        );
        metadata.insert(
            "truncation_enabled".to_string(),
            serde_json::Value::Bool(self.config.truncation),
        );

        metadata
    }

    /// Validate input examples for basic consistency
    ///
    /// Performs basic validation to ensure examples are in a format
    /// that can be safely processed by the default collator.
    ///
    /// # Arguments
    ///
    /// * `examples` - Input examples to validate
    ///
    /// # Returns
    ///
    /// Result indicating whether examples are valid
    fn validate_examples(&self, examples: &[DataExample]) -> Result<()> {
        if examples.is_empty() {
            return Err(TrustformersError::invalid_input(
                "Cannot collate empty batch".to_string(),
                Some("examples".to_string()),
                Some("non-empty batch".to_string()),
                Some("empty batch".to_string()),
            ));
        }

        for (i, example) in examples.iter().enumerate() {
            if example.input_ids.is_empty() {
                return Err(TrustformersError::invalid_input(
                    format!("Empty input_ids in example {}", i),
                    Some("input_ids"),
                    Some("non-empty input_ids"),
                    Some("empty input_ids"),
                ));
            }

            // Check attention mask consistency
            if let Some(ref attention_mask) = example.attention_mask {
                if attention_mask.len() != example.input_ids.len() {
                    return Err(TrustformersError::invalid_input(                        format!("Attention mask length {} doesn't match input_ids length {} in example {}", attention_mask.len(), example.input_ids.len(), i),
                        Some("attention_mask"),
                        Some(format!("length {}", example.input_ids.len())),
                        Some(format!("length {}", attention_mask.len()))
                    ));
                }
            }

            // Check token type IDs consistency
            if let Some(ref token_type_ids) = example.token_type_ids {
                if token_type_ids.len() != example.input_ids.len() {
                    return Err(TrustformersError::invalid_input(                        format!("Token type IDs length {} doesn't match input_ids length {} in example {}", token_type_ids.len(), example.input_ids.len(), i),
                        Some("token_type_ids"),
                        Some(format!("length {}", example.input_ids.len())),
                        Some(format!("length {}", token_type_ids.len()))
                    ));
                }
            }
        }

        Ok(())
    }
}

impl DataCollator for DefaultDataCollator {
    fn collate(&self, examples: &[DataExample]) -> Result<CollatedBatch> {
        // Validate input examples
        self.validate_examples(examples)?;

        let batch_size = examples.len();

        // Use the base language modeling collator for basic sequence processing
        // This provides standard padding, truncation, and attention mask handling
        let base_collator = LanguageModelingDataCollator::new(LanguageModelingCollatorConfig {
            max_length: self.config.max_length,
            padding: self.config.padding,
            truncation: self.config.truncation,
            pad_token_id: self.config.pad_token_id,
            mask_token_id: 0, // No masking in default collator
            mlm_probability: 0.0,
        });

        // Collate using base functionality
        let mut batch = base_collator.collate(examples)?;

        // Process labels with minimal modification
        let processed_labels = self.process_generic_labels(examples);
        batch.labels = processed_labels;

        // Create and add basic metadata
        let basic_metadata =
            self.create_basic_metadata(examples, batch_size, batch.sequence_length);
        for (key, value) in basic_metadata {
            batch.metadata.insert(key, value);
        }

        // Preserve any original metadata from examples
        let mut example_metadata = HashMap::new();
        for (i, example) in examples.iter().enumerate() {
            if !example.metadata.is_empty() {
                example_metadata.insert(
                    format!("example_{}_metadata", i),
                    serde_json::to_value(&example.metadata).unwrap_or(serde_json::Value::Null),
                );
            }
        }

        if !example_metadata.is_empty() {
            batch.metadata.insert(
                "original_metadata".to_string(),
                serde_json::to_value(example_metadata).unwrap_or(serde_json::Value::Null),
            );
        }

        Ok(batch)
    }

    fn config(&self) -> &dyn DataCollatorConfig {
        &self.config
    }

    fn preprocess_examples(&self, examples: &[DataExample]) -> Result<Vec<DataExample>> {
        // Default collator does minimal preprocessing
        // Just ensure examples are valid and return them as-is
        self.validate_examples(examples)?;
        Ok(examples.to_vec())
    }
}

// =============================================================================
// Default Configuration
// =============================================================================

/// Configuration for default data collator
///
/// This configuration struct controls the basic collation behavior for
/// the default collator. It provides only essential parameters needed
/// for general-purpose text sequence processing.
///
/// ## Configuration Parameters
///
/// - `max_length`: Maximum sequence length for padding/truncation
/// - `padding`: Strategy for padding sequences in batch
/// - `truncation`: Whether to truncate sequences exceeding max_length
/// - `pad_token_id`: Token ID used for padding positions
///
/// ## Design Principles
///
/// The configuration is intentionally minimal to:
/// - Avoid task-specific assumptions
/// - Ensure broad compatibility
/// - Provide predictable behavior
/// - Minimize configuration complexity
///
/// ## Default Values
///
/// The `from_config` method provides conservative defaults:
/// - `padding`: Longest sequence in batch (dynamic padding)
/// - `truncation`: Enabled (prevent memory issues)
/// - `pad_token_id`: From model config or 0
/// - `max_length`: From model config or unlimited
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultCollatorConfig {
    /// Maximum sequence length for padding/truncation
    pub max_length: Option<usize>,
    /// Padding strategy for batch collation
    pub padding: PaddingStrategy,
    /// Whether to truncate sequences exceeding max_length
    pub truncation: bool,
    /// Token ID used for padding
    pub pad_token_id: u32,
}

impl DefaultCollatorConfig {
    /// Create configuration from model config JSON
    ///
    /// This method extracts basic configuration parameters from a model's
    /// config.json file and creates a conservative collator configuration
    /// suitable for general use.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value (typically from config.json)
    ///
    /// # Returns
    ///
    /// A configured `DefaultCollatorConfig` with parameters extracted
    /// from the model configuration and conservative defaults.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// let model_config = serde_json::json!({
    ///     "max_position_embeddings": 512,
    ///     "pad_token_id": 0,
    ///     "vocab_size": 30522
    /// });
    ///
    /// let config = DefaultCollatorConfig::from_config(&model_config)?;
    /// assert_eq!(config.max_length, Some(512));
    /// assert_eq!(config.pad_token_id, 0);
    /// assert_eq!(config.truncation, true);

    pub fn from_config(config: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            max_length: config
                .get("max_position_embeddings")
                .or_else(|| config.get("max_length"))
                .or_else(|| config.get("model_max_length"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: config.get("pad_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        })
    }

    /// Create a minimal configuration for basic text processing
    ///
    /// Creates a configuration with minimal settings suitable for
    /// simple text processing tasks where advanced features are not needed.
    ///
    /// # Arguments
    ///
    /// * `pad_token_id` - Token ID to use for padding
    ///
    /// # Returns
    ///
    /// A minimal configuration for basic use
    pub fn minimal(pad_token_id: u32) -> Self {
        Self {
            max_length: None,
            padding: PaddingStrategy::Longest,
            truncation: false,
            pad_token_id,
        }
    }

    /// Create a configuration for inference-only scenarios
    ///
    /// Creates a configuration optimized for inference where labels
    /// are not needed and processing should be minimal.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value
    ///
    /// # Returns
    ///
    /// A configuration optimized for inference
    pub fn for_inference(config: &serde_json::Value) -> Result<Self> {
        let mut inference_config = Self::from_config(config)?;
        // For inference, we might want to be more lenient with truncation
        inference_config.truncation = false;
        Ok(inference_config)
    }

    /// Create a configuration for experimental/development use
    ///
    /// Creates a configuration suitable for experimentation and development
    /// where flexibility and debugging information are prioritized.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value
    ///
    /// # Returns
    ///
    /// A configuration optimized for development
    pub fn for_development(config: &serde_json::Value) -> Result<Self> {
        let mut dev_config = Self::from_config(config)?;
        // For development, use smaller max length for faster iteration
        if let Some(max_len) = dev_config.max_length {
            dev_config.max_length = Some(max_len.min(128));
        } else {
            dev_config.max_length = Some(128);
        }
        Ok(dev_config)
    }

    /// Create a configuration with custom length limits
    ///
    /// Creates a configuration with user-specified length limits,
    /// useful for specific deployment constraints or memory limitations.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration as JSON value
    /// * `max_length` - Custom maximum sequence length
    ///
    /// # Returns
    ///
    /// A configuration with custom length limits
    pub fn with_max_length(config: &serde_json::Value, max_length: usize) -> Result<Self> {
        let mut length_config = Self::from_config(config)?;
        length_config.max_length = Some(max_length);
        length_config.truncation = true;
        Ok(length_config)
    }
}

impl DataCollatorConfig for DefaultCollatorConfig {
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
    fn test_default_collator_creation() {
        let config = DefaultCollatorConfig {
            max_length: Some(128),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
        };

        let collator = DefaultDataCollator::new(config);
        assert_eq!(collator.config().max_length(), Some(128));
        assert_eq!(collator.config().padding(), PaddingStrategy::Longest);
        assert_eq!(collator.config().truncation(), true);
    }

    #[test]
    fn test_default_config_from_json() {
        let config_json = serde_json::json!({
            "max_position_embeddings": 512,
            "pad_token_id": 1,
            "vocab_size": 30522
        });

        let config = DefaultCollatorConfig::from_config(&config_json).unwrap();
        assert_eq!(config.max_length, Some(512));
        assert_eq!(config.pad_token_id, 1);
        assert_eq!(config.truncation, true);
        assert_eq!(config.padding, PaddingStrategy::Longest);
    }

    #[test]
    fn test_collate_basic_examples() {
        let config = DefaultCollatorConfig {
            max_length: Some(10),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
        };

        let collator = DefaultDataCollator::new(config);

        let examples = vec![
            DataExample {
                input_ids: vec![1, 2, 3, 4],
                attention_mask: Some(vec![1, 1, 1, 1]),
                token_type_ids: None,
                labels: Some(vec![1, 0, 1]),
                metadata: HashMap::new(),
            },
            DataExample {
                input_ids: vec![5, 6],
                attention_mask: Some(vec![1, 1]),
                token_type_ids: None,
                labels: Some(vec![0]),
                metadata: HashMap::new(),
            },
        ];

        let batch = collator.collate(&examples).unwrap();
        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.input_ids.len(), 2);
        assert!(batch.labels.is_some());

        let labels = batch.labels.as_ref().unwrap();
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], vec![1, 0, 1]);
        assert_eq!(labels[1], vec![0]);
    }

    #[test]
    fn test_examples_without_labels() {
        let config = DefaultCollatorConfig {
            max_length: Some(10),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
        };

        let collator = DefaultDataCollator::new(config);

        let examples = vec![DataExample {
            input_ids: vec![1, 2, 3],
            attention_mask: Some(vec![1, 1, 1]),
            token_type_ids: None,
            labels: None,
            metadata: HashMap::new(),
        }];

        let batch = collator.collate(&examples).unwrap();
        assert_eq!(batch.batch_size, 1);
        assert!(batch.labels.is_none());
    }

    #[test]
    fn test_metadata_creation() {
        let config = DefaultCollatorConfig {
            max_length: Some(10),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
        };

        let collator = DefaultDataCollator::new(config);

        let examples = vec![DataExample {
            input_ids: vec![1, 2, 3, 4, 5],
            attention_mask: Some(vec![1, 1, 1, 1, 1]),
            token_type_ids: Some(vec![0, 0, 1, 1, 1]),
            labels: Some(vec![1]),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert(
                    "source".to_string(),
                    serde_json::Value::String("test".to_string()),
                );
                meta
            },
        }];

        let metadata = collator.create_basic_metadata(&examples, 1, 5);
        assert_eq!(metadata.get("collator_type").unwrap(), "default");
        assert_eq!(
            metadata.get("has_token_type_ids").unwrap(),
            &serde_json::Value::Bool(true)
        );
        assert_eq!(
            metadata.get("has_labels").unwrap(),
            &serde_json::Value::Bool(true)
        );
        assert!(metadata.contains_key("original_sequence_stats"));
    }

    #[test]
    fn test_example_validation() {
        let config = DefaultCollatorConfig {
            max_length: Some(10),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
        };

        let collator = DefaultDataCollator::new(config);

        // Valid examples
        let valid_examples = vec![DataExample {
            input_ids: vec![1, 2, 3],
            attention_mask: Some(vec![1, 1, 1]),
            token_type_ids: None,
            labels: None,
            metadata: HashMap::new(),
        }];

        assert!(collator.validate_examples(&valid_examples).is_ok());

        // Empty input_ids
        let empty_examples = vec![DataExample {
            input_ids: vec![],
            attention_mask: Some(vec![]),
            token_type_ids: None,
            labels: None,
            metadata: HashMap::new(),
        }];

        assert!(collator.validate_examples(&empty_examples).is_err());

        // Mismatched attention mask
        let mismatched_examples = vec![DataExample {
            input_ids: vec![1, 2, 3],
            attention_mask: Some(vec![1, 1]), // Wrong length
            token_type_ids: None,
            labels: None,
            metadata: HashMap::new(),
        }];

        assert!(collator.validate_examples(&mismatched_examples).is_err());
    }

    #[test]
    fn test_minimal_config() {
        let config = DefaultCollatorConfig::minimal(42);
        assert_eq!(config.pad_token_id, 42);
        assert_eq!(config.max_length, None);
        assert_eq!(config.truncation, false);
        assert_eq!(config.padding, PaddingStrategy::Longest);
    }

    #[test]
    fn test_inference_config() {
        let model_config = serde_json::json!({
            "max_position_embeddings": 512,
            "pad_token_id": 0
        });

        let config = DefaultCollatorConfig::for_inference(&model_config).unwrap();
        assert_eq!(config.max_length, Some(512));
        assert_eq!(config.truncation, false); // More lenient for inference
    }

    #[test]
    fn test_development_config() {
        let model_config = serde_json::json!({
            "max_position_embeddings": 1024,
            "pad_token_id": 0
        });

        let config = DefaultCollatorConfig::for_development(&model_config).unwrap();
        assert_eq!(config.max_length, Some(128)); // Smaller for development
    }

    #[test]
    fn test_custom_length_config() {
        let model_config = serde_json::json!({
            "max_position_embeddings": 1024,
            "pad_token_id": 0
        });

        let config = DefaultCollatorConfig::with_max_length(&model_config, 256).unwrap();
        assert_eq!(config.max_length, Some(256));
        assert_eq!(config.truncation, true);
    }

    #[test]
    fn test_preserve_original_metadata() {
        let config = DefaultCollatorConfig {
            max_length: Some(10),
            padding: PaddingStrategy::Longest,
            truncation: true,
            pad_token_id: 0,
        };

        let collator = DefaultDataCollator::new(config);

        let mut example_metadata = HashMap::new();
        example_metadata.insert(
            "custom_field".to_string(),
            serde_json::Value::String("test_value".to_string()),
        );

        let examples = vec![DataExample {
            input_ids: vec![1, 2, 3],
            attention_mask: Some(vec![1, 1, 1]),
            token_type_ids: None,
            labels: None,
            metadata: example_metadata,
        }];

        let batch = collator.collate(&examples).unwrap();
        assert!(batch.metadata.contains_key("original_metadata"));
    }
}
