//! # Sequence-to-Sequence Metrics for TrustformeRS
//!
//! This module provides evaluation metrics for sequence-to-sequence tasks, including
//! translation, summarization, and text-to-text generation.
//!
//! ## Overview
//!
//! The `Seq2SeqMetric` implementation provides specialized evaluation for sequence-to-sequence
//! models by wrapping and extending the GenerationMetric with seq2seq-specific behaviors.
//! It focuses on text generation quality through word overlap and content similarity.
//!
//! ## Features
//!
//! - **BLEU-like scoring**: Based on word overlap between generated and reference sequences
//! - **Seq2Seq specialization**: Tailored for translation and summarization tasks
//! - **Generation metric foundation**: Built on proven text generation evaluation
//! - **Flexible input handling**: Works with various text generation formats
//!
//! ## Usage Examples
//!
//! ### Translation Evaluation
//!
//! ```rust,ignore
//! use trustformers::auto::metrics::{Seq2SeqMetric, MetricInput, Metric};
//!
//! let mut metric = Seq2SeqMetric::new();
//!
//! // Translation pairs
//! let predictions = MetricInput::Text(vec![
//!     "The quick brown fox jumps".to_string(),
//!     "Hello how are you today".to_string(),
//! ]);
//! let references = MetricInput::Text(vec![
//!     "The quick brown fox jumps over the lazy dog".to_string(),
//!     "Hello how are you doing today".to_string(),
//! ]);
//!
//! metric.add_batch(&predictions, &references)?;
//!
//! // Compute results
//! let result = metric.compute()?;
//! println!("Seq2Seq Score: {:.3}", result.value);
//! println!("BLEU-like: {:.3}", result.details.get("bleu_like").unwrap());
//! ```
//!
//! ### Summarization Evaluation
//!
//! ```rust,ignore
//! use trustformers::auto::metrics::{Seq2SeqMetric, MetricInput, Metric};
//!
//! let mut metric = Seq2SeqMetric::new();
//!
//! // Generated summaries vs reference summaries
//! let predictions = MetricInput::Text(vec![
//!     "AI research advances rapidly".to_string(),
//! ]);
//! let references = MetricInput::Text(vec![
//!     "Artificial intelligence research is advancing at a rapid pace".to_string(),
//! ]);
//!
//! metric.add_batch(&predictions, &references)?;
//! let result = metric.compute()?;
//! ```
//!
//! ## Implementation Details
//!
//! ### Architecture
//!
//! The `Seq2SeqMetric` is implemented as a wrapper around `GenerationMetric`,
//! providing seq2seq-specific branding and potential future extensions while
//! maintaining the proven evaluation logic.
//!
//! ### Evaluation Algorithm
//!
//! 1. **Tokenization**: Split sequences into words using whitespace
//! 2. **Word matching**: Count overlapping words between prediction and reference
//! 3. **Precision/Recall**: Calculate based on word overlap
//! 4. **F1 aggregation**: Combine precision and recall for balanced scoring
//! 5. **Averaging**: Average F1 scores across all sequence pairs
//!
//! ### Use Cases
//!
//! - **Machine Translation**: Evaluate translation quality
//! - **Text Summarization**: Assess summary content overlap
//! - **Text-to-Text Generation**: General seq2seq model evaluation
//! - **Paraphrasing**: Measure semantic similarity through word overlap

use super::{GenerationMetric, Metric, MetricInput, MetricResult};
use crate::error::Result;

/// Sequence-to-sequence metric implementation
///
/// Provides specialized evaluation metrics for seq2seq tasks by wrapping
/// the GenerationMetric with seq2seq-specific behavior and naming.
///
/// ## Design Principles
///
/// - **Composition**: Built on proven GenerationMetric foundation
/// - **Specialization**: Tailored for seq2seq use cases
/// - **Compatibility**: Maintains same interface as other metrics
/// - **Extensibility**: Easy to add seq2seq-specific features in the future
///
/// ## Supported Input Types
///
/// - `Text`: Generated sequences and reference sequences as strings
///
/// ## Relationship to GenerationMetric
///
/// The Seq2SeqMetric internally uses GenerationMetric for computation,
/// but provides seq2seq-specific naming and could be extended with
/// additional seq2seq-specific features like:
/// - Multi-reference support
/// - ROUGE scores
/// - Task-specific preprocessing
#[derive(Debug, Clone)]
pub struct Seq2SeqMetric {
    /// Internal generation metric for computation
    generation_metric: GenerationMetric,
}

impl Seq2SeqMetric {
    /// Create a new sequence-to-sequence metric instance
    ///
    /// Initializes a new seq2seq metric with an internal GenerationMetric
    /// for handling the actual evaluation computation.
    ///
    /// # Returns
    ///
    /// New `Seq2SeqMetric` instance ready for evaluation.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use trustformers::auto::metrics::Seq2SeqMetric;
    ///
    /// let metric = Seq2SeqMetric::new();
    /// assert_eq!(metric.name(), "seq2seq");
    /// ```
    pub fn new() -> Self {
        Self {
            generation_metric: GenerationMetric::new(),
        }
    }
}

impl Metric for Seq2SeqMetric {
    /// Add a batch of sequence predictions and references
    ///
    /// Delegates to the internal GenerationMetric for batch processing.
    /// Handles sequence-to-sequence data including translations, summaries,
    /// and other text-to-text generation outputs.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Generated sequences from the seq2seq model (Text)
    /// * `references` - Ground truth reference sequences (Text)
    ///
    /// # Input Format Requirements
    ///
    /// - **Text**: Vector of strings containing generated and reference sequences
    /// - Both predictions and references must have compatible lengths
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error if input formats are incompatible.
    ///
    /// # Errors
    ///
    /// - `InvalidInput`: If input types are not both Text variants
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// use trustformers::auto::metrics::{Seq2SeqMetric, MetricInput, Metric};
    ///
    /// let mut metric = Seq2SeqMetric::new();
    ///
    /// let predictions = MetricInput::Text(vec![
    ///     "Bonjour le monde".to_string(),  // Translation output
    ///     "AI is powerful".to_string(),     // Summary output
    /// ]);
    /// let references = MetricInput::Text(vec![
    ///     "Bonjour tout le monde".to_string(),  // Reference translation
    ///     "Artificial intelligence is very powerful".to_string(), // Reference summary
    /// ]);
    ///
    /// metric.add_batch(&predictions, &references)?;

    fn add_batch(&mut self, predictions: &MetricInput, references: &MetricInput) -> Result<()> {
        self.generation_metric.add_batch(predictions, references)
    }

    /// Compute sequence-to-sequence metrics
    ///
    /// Delegates computation to the internal GenerationMetric and returns
    /// results with seq2seq-specific naming and branding.
    ///
    /// # Returns
    ///
    /// `MetricResult` containing:
    /// - **Primary value**: Average F1 score across all sequence pairs
    /// - **Details**:
    ///   - `bleu_like`: BLEU-like score based on word overlap
    /// - **Name**: "seq2seq" (distinguishes from generic generation)
    ///
    /// # Errors
    ///
    /// - `InvalidInput`: If no data has been added to the metric
    /// - Passes through any errors from the underlying GenerationMetric
    ///
    /// # Algorithm Details
    ///
    /// The computation follows the same algorithm as GenerationMetric:
    /// 1. For each (prediction, reference) pair:
    ///    - Tokenize by whitespace
    ///    - Count word overlaps
    ///    - Calculate precision and recall
    ///    - Compute F1 score
    /// 2. Average F1 scores across all pairs
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// use trustformers::auto::metrics::{Seq2SeqMetric, MetricInput, Metric};
    ///
    /// let mut metric = Seq2SeqMetric::new();
    /// metric.add_batch(
    ///     &MetricInput::Text(vec!["hello world".to_string()]),
    ///     &MetricInput::Text(vec!["hello universe".to_string()])
    /// )?;
    ///
    /// let result = metric.compute()?;
    /// assert_eq!(result.name, "seq2seq");
    /// assert!(result.value >= 0.0 && result.value <= 1.0);
    /// assert!(result.details.contains_key("bleu_like"));

    fn compute(&self) -> Result<MetricResult> {
        let mut result = self.generation_metric.compute()?;
        // Override the name to indicate this is a seq2seq metric
        result.name = "seq2seq".to_string();
        Ok(result)
    }

    /// Reset the metric state
    ///
    /// Delegates to the internal GenerationMetric to clear all accumulated
    /// sequence pairs, preparing for a new evaluation run.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// use trustformers::auto::metrics::{Seq2SeqMetric, MetricInput, Metric};
    ///
    /// let mut metric = Seq2SeqMetric::new();
    /// metric.add_batch(
    ///     &MetricInput::Text(vec!["hello".to_string()]),
    ///     &MetricInput::Text(vec!["world".to_string()])
    /// )?;
    ///
    /// metric.reset();
    /// // Metric is now ready for new sequence pairs

    fn reset(&mut self) {
        self.generation_metric.reset();
    }

    /// Get the metric name
    ///
    /// Returns the identifier for this metric type, distinguishing it
    /// from the generic generation metric.
    ///
    /// # Returns
    ///
    /// String slice "seq2seq"
    fn name(&self) -> &str {
        "seq2seq"
    }
}

impl Default for Seq2SeqMetric {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seq2seq_metric_basic() {
        let mut metric = Seq2SeqMetric::new();

        let predictions = MetricInput::Text(vec![
            "the quick brown fox".to_string(),
            "hello world".to_string(),
        ]);
        let references = MetricInput::Text(vec![
            "the quick brown fox jumps".to_string(),
            "hello world test".to_string(),
        ]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.name, "seq2seq");
        assert!(result.value >= 0.0 && result.value <= 1.0);
        assert!(result.details.contains_key("bleu_like"));
    }

    #[test]
    fn test_seq2seq_metric_perfect_match() {
        let mut metric = Seq2SeqMetric::new();

        let predictions = MetricInput::Text(vec!["hello world".to_string()]);
        let references = MetricInput::Text(vec!["hello world".to_string()]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.name, "seq2seq");
        assert_eq!(result.value, 1.0); // Perfect match should give 1.0
    }

    #[test]
    fn test_seq2seq_metric_no_overlap() {
        let mut metric = Seq2SeqMetric::new();

        let predictions = MetricInput::Text(vec!["foo bar".to_string()]);
        let references = MetricInput::Text(vec!["baz qux".to_string()]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.name, "seq2seq");
        assert_eq!(result.value, 0.0); // No overlap should give 0.0
    }

    #[test]
    fn test_seq2seq_metric_translation_example() {
        let mut metric = Seq2SeqMetric::new();

        // Simulated translation evaluation
        let predictions = MetricInput::Text(vec![
            "The cat sits on the mat".to_string(),
            "I love artificial intelligence".to_string(),
        ]);
        let references = MetricInput::Text(vec![
            "The cat is sitting on the mat".to_string(),
            "I love AI and machine learning".to_string(),
        ]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.name, "seq2seq");
        assert!(result.value > 0.0); // Should have some overlap
    }

    #[test]
    fn test_seq2seq_metric_summarization_example() {
        let mut metric = Seq2SeqMetric::new();

        // Simulated summarization evaluation
        let predictions = MetricInput::Text(vec!["AI research advances rapidly".to_string()]);
        let references = MetricInput::Text(vec![
            "Artificial intelligence research is advancing rapidly".to_string(),
        ]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.name, "seq2seq");
        assert!(result.value > 0.0); // Should capture some overlap
    }

    #[test]
    fn test_seq2seq_metric_reset() {
        let mut metric = Seq2SeqMetric::new();

        let predictions = MetricInput::Text(vec!["hello".to_string()]);
        let references = MetricInput::Text(vec!["world".to_string()]);
        metric.add_batch(&predictions, &references).unwrap();

        metric.reset();

        // Should fail because no data after reset
        assert!(metric.compute().is_err());
    }

    #[test]
    fn test_seq2seq_metric_invalid_input() {
        let mut metric = Seq2SeqMetric::new();

        let predictions = MetricInput::Classifications(vec![0, 1]);
        let references = MetricInput::Text(vec!["hello".to_string()]);

        let result = metric.add_batch(&predictions, &references);
        assert!(result.is_err());
    }

    #[test]
    fn test_seq2seq_metric_multiple_batches() {
        let mut metric = Seq2SeqMetric::new();

        // First batch - perfect match
        metric
            .add_batch(
                &MetricInput::Text(vec!["hello world".to_string()]),
                &MetricInput::Text(vec!["hello world".to_string()]),
            )
            .unwrap();

        // Second batch - no overlap
        metric
            .add_batch(
                &MetricInput::Text(vec!["foo bar".to_string()]),
                &MetricInput::Text(vec!["baz qux".to_string()]),
            )
            .unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.name, "seq2seq");
        // Should average: 1.0 (perfect) + 0.0 (no overlap) = 0.5
        assert_eq!(result.value, 0.5);
    }

    #[test]
    fn test_seq2seq_metric_name() {
        let metric = Seq2SeqMetric::new();
        assert_eq!(metric.name(), "seq2seq");
    }

    #[test]
    fn test_seq2seq_metric_empty_sequences() {
        let mut metric = Seq2SeqMetric::new();

        let predictions = MetricInput::Text(vec!["".to_string()]);
        let references = MetricInput::Text(vec!["hello world".to_string()]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.name, "seq2seq");
        // Empty prediction should result in 0 score
        assert_eq!(result.value, 0.0);
    }
}
