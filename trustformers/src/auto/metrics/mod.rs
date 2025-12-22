//! # Metrics Module for TrustformeRS Auto Framework
//!
//! This module provides comprehensive evaluation metrics for machine learning tasks.
//! It includes automatic metric creation, base trait definitions, and specific metric
//! implementations for various NLP and computer vision tasks.
//!
//! ## Overview
//!
//! The metrics system is designed around the `AutoMetric` entry point, which automatically
//! creates appropriate metrics based on the task type. All metrics implement the common
//! `Metric` trait, providing a unified interface for evaluation.
//!
//! ## Core Components
//!
//! - **AutoMetric**: Main entry point for automatic metric creation
//! - **Metric Trait**: Common interface for all metric implementations
//! - **MetricInput**: Enumeration of supported input types for metrics
//! - **MetricResult**: Standard result format for metric computation
//!
//! ## Supported Tasks
//!
//! The metrics system supports the following tasks:
//!
//! - **Text Classification**: Accuracy, precision, recall, F1-score
//! - **Text Generation**: BLEU-like scores, perplexity
//! - **Language Modeling**: Perplexity, log-likelihood
//! - **Sequence-to-Sequence**: BLEU, ROUGE scores
//! - **Question Answering**: Exact match, token-level F1
//! - **Token Classification**: Entity-level precision, recall, F1
//!
//! ## Usage Examples
//!
//! ### Basic Usage
//!
//! ```rust,ignore
//! use trustformers::auto::metrics::{AutoMetric, MetricInput};
//!
//! // Create a metric for text classification
//! let mut metric = AutoMetric::for_task("text-classification")?;
//!
//! // Add predictions and references
//! let predictions = MetricInput::Classifications(vec![0, 1, 0, 1]);
//! let references = MetricInput::Classifications(vec![0, 0, 1, 1]);
//! metric.add_batch(&predictions, &references)?;
//!
//! // Compute results
//! let result = metric.compute()?;
//! println!("Accuracy: {}", result.details.get("accuracy").unwrap());
//! ```
//!
//! ### Composite Metrics
//!
//! ```rust,ignore
//! use trustformers::auto::metrics::AutoMetric;
//!
//! // Create metrics for multiple tasks
//! let composite = AutoMetric::composite(&[
//!     "text-classification",
//!     "text-generation"
//! ])?;
//!
//! // Compute all metrics at once
//! let results = composite.compute_all()?;
//! ```
//!
//! ## Architecture
//!
//! The metrics module follows a layered architecture:
//!
//! 1. **Auto Layer**: `AutoMetric` provides automatic metric selection
//! 2. **Base Layer**: `Metric` trait defines the common interface
//! 3. **Implementation Layer**: Specific metrics implement the trait
//! 4. **Input/Output Layer**: Standardized data types for consistency
//!
//! ## Adding New Metrics
//!
//! To add a new metric:
//!
//! 1. Implement the `Metric` trait
//! 2. Handle appropriate `MetricInput` variants
//! 3. Return results in `MetricResult` format
//! 4. Add the metric to `AutoMetric::for_task`
//! 5. Export the metric from this module

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Note: Individual metrics modules import required types directly

// =============================================================================
// Auto Metric Creation
// =============================================================================

/// Automatically create metrics based on task and model configuration
///
/// `AutoMetric` is the main entry point for the metrics system. It provides
/// automatic metric selection based on task type, eliminating the need to
/// manually choose and configure metrics for common NLP and vision tasks.
///
/// ## Design Principles
///
/// - **Task-Oriented**: Metrics are selected based on the specific task
/// - **Automatic**: No manual configuration required for standard tasks
/// - **Extensible**: Easy to add support for new tasks and metrics
/// - **Composable**: Support for multi-task scenarios through composite metrics
///
/// ## Performance Characteristics
///
/// - Metrics are created lazily and cached when appropriate
/// - Memory usage is optimized for large-scale evaluation
/// - Computation is vectorized where possible for efficiency
#[derive(Debug, Clone)]
pub struct AutoMetric;

impl AutoMetric {
    /// Create metrics for a specific task
    ///
    /// This method automatically selects and creates the most appropriate metrics
    /// for the given task. The selection is based on established best practices
    /// for each task type.
    ///
    /// # Arguments
    ///
    /// * `task` - The task type (e.g., "text-classification", "text-generation")
    ///
    /// # Returns
    ///
    /// Returns a boxed metric implementation suitable for the task.
    ///
    /// # Supported Tasks
    ///
    /// - `"text-classification"`, `"sentiment-analysis"`: Classification metrics
    /// - `"text-generation"`, `"causal-lm"`: Generation metrics
    /// - `"masked-lm"`, `"fill-mask"`: Language modeling metrics
    /// - `"translation"`, `"summarization"`, `"text2text-generation"`: Seq2Seq metrics
    /// - `"question-answering"`: QA-specific metrics
    /// - `"token-classification"`, `"ner"`: Token-level metrics
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// use trustformers::auto::metrics::AutoMetric;
    ///
    /// // Create classification metric
    /// let metric = AutoMetric::for_task("text-classification")?;
    ///
    /// // Create generation metric
    /// let metric = AutoMetric::for_task("text-generation")?;

    pub fn for_task(task: &str) -> Result<Box<dyn Metric>> {
        match task {
            "text-classification" | "sentiment-analysis" => {
                Ok(Box::new(ClassificationMetric::new()))
            },
            "text-generation" | "causal-lm" => Ok(Box::new(GenerationMetric::new())),
            "masked-lm" | "fill-mask" => Ok(Box::new(LanguageModelingMetric::new())),
            "translation" | "summarization" | "text2text-generation" => {
                Ok(Box::new(Seq2SeqMetric::new()))
            },
            "question-answering" => Ok(Box::new(QuestionAnsweringMetric::new())),
            "token-classification" | "ner" => Ok(Box::new(TokenClassificationMetric::new())),
            _ => Ok(Box::new(DefaultMetric::new())),
        }
    }

    /// Create composite metrics for multiple tasks
    ///
    /// This method creates a composite metric that can handle multiple tasks
    /// simultaneously. This is useful for multi-task training scenarios or
    /// when you need to evaluate a model on multiple criteria.
    ///
    /// # Arguments
    ///
    /// * `tasks` - Array of task types to create metrics for
    ///
    /// # Returns
    ///
    /// Returns a `CompositeMetric` that manages multiple individual metrics.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// use trustformers::auto::metrics::AutoMetric;
    ///
    /// // Create composite metric for multiple tasks
    /// let composite = AutoMetric::composite(&[
    ///     "text-classification",
    ///     "text-generation"
    /// ])?;
    ///
    /// // Compute all metrics
    /// let results = composite.compute_all()?;

    pub fn composite(tasks: &[&str]) -> Result<CompositeMetric> {
        let mut metrics = Vec::new();
        for task in tasks {
            metrics.push(Self::for_task(task)?);
        }
        Ok(CompositeMetric::new(metrics))
    }
}

// =============================================================================
// Base Metric Trait and Types
// =============================================================================

/// Trait for evaluation metrics
///
/// This trait defines the common interface that all metrics must implement.
/// It provides a standardized way to add evaluation data, compute results,
/// and manage metric state.
///
/// ## Design Philosophy
///
/// The trait is designed to be:
/// - **Stateful**: Metrics accumulate data over multiple batches
/// - **Flexible**: Supports various input types through `MetricInput`
/// - **Consistent**: All metrics return results in the same format
/// - **Resettable**: Metrics can be reset for new evaluation runs
///
/// ## Thread Safety
///
/// All metric implementations must be thread-safe (`Send + Sync`) to support
/// parallel evaluation scenarios.
pub trait Metric: Send + Sync + std::fmt::Debug {
    /// Add predictions and references for evaluation
    ///
    /// This method accumulates evaluation data for later computation. It's
    /// designed to handle streaming evaluation where data arrives in batches.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Model predictions in the appropriate format
    /// * `references` - Ground truth references in the same format
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input types are incompatible with the metric
    /// - Predictions and references have mismatched dimensions
    /// - Invalid data is provided (e.g., negative probabilities)
    fn add_batch(&mut self, predictions: &MetricInput, references: &MetricInput) -> Result<()>;

    /// Compute the final metric values
    ///
    /// This method computes the final metric values based on all accumulated
    /// data. It can be called multiple times without modifying the internal
    /// state, allowing for repeated queries.
    ///
    /// # Returns
    ///
    /// Returns a `MetricResult` containing:
    /// - Primary metric value
    /// - Detailed breakdown of sub-metrics
    /// - Metadata about the computation
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No data has been added yet
    /// - Accumulated data is inconsistent
    /// - Computation fails due to numerical issues
    fn compute(&self) -> Result<MetricResult>;

    /// Reset the metric state
    ///
    /// This method clears all accumulated data, preparing the metric for
    /// a new evaluation run. This is more efficient than creating a new
    /// metric instance.
    fn reset(&mut self);

    /// Get the metric name
    ///
    /// Returns a string identifier for the metric type. This is used for
    /// logging, debugging, and result identification.
    fn name(&self) -> &str;
}

/// Input for metric computation
///
/// This enum defines the various input formats that metrics can accept.
/// It provides type safety while allowing flexibility in the types of
/// data that can be evaluated.
///
/// ## Input Types
///
/// Each variant is designed for specific types of model outputs:
///
/// - **Classifications**: For discrete classification outputs
/// - **Probabilities**: For probability distributions over vocabularies
/// - **Tokens**: For token-level predictions (token IDs)
/// - **Text**: For generated text outputs
/// - **Spans**: For structured outputs like NER spans
/// - **Scores**: For regression or scoring tasks
#[derive(Debug, Clone)]
pub enum MetricInput {
    /// Classification predictions as class indices
    ///
    /// Used for tasks where the model outputs discrete class predictions.
    /// Each value represents the predicted class index.
    ///
    /// Example: `[0, 1, 2, 1]` for 4 samples with 3 classes
    Classifications(Vec<usize>),

    /// Probability distributions over vocabulary
    ///
    /// Used for language modeling and generation tasks. The outer vector
    /// represents sequences, the middle vector represents positions, and
    /// the inner vector represents probability distributions over vocabulary.
    ///
    /// Shape: `[batch_size, sequence_length, vocab_size]`
    Probabilities(Vec<Vec<Vec<f32>>>), // Seq of positions of vocab distributions

    /// Token-level predictions as token IDs
    ///
    /// Used for sequence labeling and generation tasks where the output
    /// is a sequence of token IDs.
    ///
    /// Shape: `[batch_size, sequence_length]`
    Tokens(Vec<Vec<u32>>),

    /// Text predictions as strings
    ///
    /// Used for generation tasks where the final output is text.
    /// Each string represents a complete generated sequence.
    ///
    /// Example: `["Hello world", "How are you?"]`
    Text(Vec<String>),

    /// Span-based predictions for structured tasks
    ///
    /// Used for named entity recognition and similar tasks. Each span
    /// is represented as (start, end, label).
    ///
    /// Format: `[(start_pos, end_pos, "ENTITY_TYPE")]`
    Spans(Vec<Vec<(usize, usize, String)>>),

    /// Numeric scores for regression tasks
    ///
    /// Used for tasks that output continuous values or confidence scores.
    ///
    /// Example: `[0.8, 0.3, 0.9, 0.1]` for regression outputs
    Scores(Vec<f32>),
}

/// Metric computation result
///
/// This struct provides a standardized format for metric results, ensuring
/// consistency across all metric implementations.
///
/// ## Structure
///
/// - **name**: Identifies the metric type
/// - **value**: Primary metric value (often the most important score)
/// - **details**: Breakdown of individual metric components
/// - **metadata**: Additional information about the computation
///
/// ## Usage
///
/// The primary value is typically the most important metric for the task
/// (e.g., accuracy for classification, BLEU for translation). The details
/// map provides access to individual components and sub-metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    /// Name of the metric (e.g., "classification", "generation")
    pub name: String,

    /// Primary metric value (the most important score for this metric type)
    pub value: f64,

    /// Detailed breakdown of metric components
    ///
    /// Common keys include:
    /// - "accuracy", "precision", "recall", "f1" for classification
    /// - "bleu", "rouge", "perplexity" for generation
    /// - "exact_match" for question answering
    pub details: HashMap<String, f64>,

    /// Additional metadata about the computation
    ///
    /// May include information like:
    /// - Number of samples evaluated
    /// - Computation parameters
    /// - Confidence intervals
    /// - Processing time
    pub metadata: HashMap<String, serde_json::Value>,
}

// Note: CompositeMetric implementation has been moved to composite.rs module

// =============================================================================
// Metric Implementations (Module Declarations and Re-exports)
// =============================================================================

// Individual metric implementation modules
pub mod classification;
pub mod composite;
pub mod generation;
pub mod language_modeling;
pub mod question_answering;
pub mod seq2seq;
pub mod token_classification;

// Re-export all metric implementations for convenient access
pub use classification::ClassificationMetric;
pub use composite::{CompositeMetric, DefaultMetric};
pub use generation::GenerationMetric;
pub use language_modeling::LanguageModelingMetric;
pub use question_answering::QuestionAnsweringMetric;
pub use seq2seq::Seq2SeqMetric;
pub use token_classification::TokenClassificationMetric;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_metric_for_task() {
        // Test classification metric creation
        let metric = AutoMetric::for_task("text-classification");
        assert!(metric.is_ok());
        assert_eq!(metric.unwrap().name(), "classification");

        // Test generation metric creation
        let metric = AutoMetric::for_task("text-generation");
        assert!(metric.is_ok());
        assert_eq!(metric.unwrap().name(), "generation");

        // Test unknown task defaults to default metric
        let metric = AutoMetric::for_task("unknown-task");
        assert!(metric.is_ok());
        assert_eq!(metric.unwrap().name(), "default");
    }

    #[test]
    fn test_composite_metric() {
        let composite = AutoMetric::composite(&["text-classification", "text-generation"]);
        assert!(composite.is_ok());

        let composite = composite.unwrap();
        assert_eq!(composite.metrics().len(), 2);
    }

    #[test]
    fn test_metric_input_variants() {
        let classifications = MetricInput::Classifications(vec![0, 1, 2]);
        let text = MetricInput::Text(vec!["hello".to_string(), "world".to_string()]);
        let scores = MetricInput::Scores(vec![0.1, 0.8, 0.3]);

        // Test that variants can be created and matched
        match classifications {
            MetricInput::Classifications(ref data) => assert_eq!(data.len(), 3),
            _ => panic!("Expected Classifications variant"),
        }

        match text {
            MetricInput::Text(ref data) => assert_eq!(data.len(), 2),
            _ => panic!("Expected Text variant"),
        }

        match scores {
            MetricInput::Scores(ref data) => assert_eq!(data.len(), 3),
            _ => panic!("Expected Scores variant"),
        }
    }

    #[test]
    fn test_metric_result_creation() {
        let mut details = HashMap::new();
        details.insert("accuracy".to_string(), 0.85);
        details.insert("f1".to_string(), 0.82);

        let mut metadata = HashMap::new();
        metadata.insert("samples".to_string(), serde_json::Value::Number(100.into()));

        let result = MetricResult {
            name: "test_metric".to_string(),
            value: 0.85,
            details,
            metadata,
        };

        assert_eq!(result.name, "test_metric");
        assert_eq!(result.value, 0.85);
        assert_eq!(result.details.get("accuracy"), Some(&0.85));
        assert_eq!(result.details.get("f1"), Some(&0.82));
    }
}
