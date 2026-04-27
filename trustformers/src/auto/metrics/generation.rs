//! # Generation Metrics for TrustformeRS
//!
//! This module provides evaluation metrics for text generation tasks, including
//! causal language modeling, text completion, and other generative NLP tasks.
//!
//! ## Overview
//!
//! The `GenerationMetric` implementation provides evaluation metrics specifically
//! designed for generated text, focusing on content overlap and quality measures
//! between generated and reference texts.
//!
//! ## Features
//!
//! - **BLEU-like scoring**: Simplified BLEU score based on word overlap
//! - **Word-level evaluation**: Focuses on lexical similarity
//! - **F1-based aggregation**: Combines precision and recall for balanced scoring
//! - **Flexible text input**: Works with raw generated text strings
//!
//! ## Usage Examples
//!
//! ### Basic Generation Evaluation
//!
//! ```rust,ignore
//! use trustformers::auto::metrics::{GenerationMetric, MetricInput, Metric};
//!
//! let mut metric = GenerationMetric::new();
//!
//! // Add generated text and references
//! let predictions = MetricInput::Text(vec![
//!     "The quick brown fox jumps".to_string(),
//!     "Hello world how are you".to_string(),
//! ]);
//! let references = MetricInput::Text(vec![
//!     "The quick brown fox jumps over".to_string(),
//!     "Hello world how are you doing".to_string(),
//! ]);
//! metric.add_batch(&predictions, &references)?;
//!
//! // Compute results
//! let result = metric.compute()?;
//! println!("Generation Score: {:.3}", result.value);
//! println!("BLEU-like: {:.3}", result.details.get("bleu_like").unwrap());
//! ```
//!
//! ### Multiple Batches
//!
//! ```rust,ignore
//! use trustformers::auto::metrics::{GenerationMetric, MetricInput, Metric};
//!
//! let mut metric = GenerationMetric::new();
//!
//! // First batch
//! metric.add_batch(
//!     &MetricInput::Text(vec!["First generated text".to_string()]),
//!     &MetricInput::Text(vec!["First reference text".to_string()])
//! )?;
//!
//! // Second batch
//! metric.add_batch(
//!     &MetricInput::Text(vec!["Second generated text".to_string()]),
//!     &MetricInput::Text(vec!["Second reference text".to_string()])
//! )?;
//!
//! let result = metric.compute()?;
//! ```
//!
//! ## Implementation Details
//!
//! ### Scoring Algorithm
//!
//! 1. **Tokenization**: Split text into whitespace-separated words
//! 2. **Word Matching**: Count overlapping words between prediction and reference
//! 3. **Precision**: `overlapping_words / prediction_words`
//! 4. **Recall**: `overlapping_words / reference_words`
//! 5. **F1 Score**: `2 * precision * recall / (precision + recall)`
//! 6. **Aggregation**: Average F1 scores across all examples
//!
//! ### Performance Characteristics
//!
//! - **Time complexity**: O(n*m) where n and m are word counts in texts
//! - **Space complexity**: O(n) for storing text pairs
//! - **Memory efficient**: Processes one text pair at a time during computation

use super::{Metric, MetricInput, MetricResult};
use crate::error::{Result, TrustformersError};
use crate::evaluation::bridge::NlpAdapter;
use std::collections::HashMap;

/// Generation metric implementation
///
/// Provides evaluation metrics for text generation tasks using BLEU-like scoring
/// based on word overlap between generated and reference texts.
///
/// ## Design Principles
///
/// - **Accumulative**: Collects text pairs over multiple batches
/// - **Word-based**: Uses whitespace tokenization for simplicity
/// - **Balanced**: Uses F1 score to balance precision and recall
/// - **Robust**: Handles empty texts and edge cases gracefully
///
/// ## Supported Input Types
///
/// - `Text`: Generated text strings and reference text strings
#[derive(Debug, Clone)]
pub struct GenerationMetric {
    /// Accumulated generated text predictions
    predictions: Vec<String>,
    /// Accumulated reference texts
    references: Vec<String>,
}

impl GenerationMetric {
    /// Create a new generation metric instance
    ///
    /// Initializes an empty metric ready to accumulate generated texts and references.
    ///
    /// # Returns
    ///
    /// New `GenerationMetric` instance with empty state.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use trustformers::auto::metrics::GenerationMetric;
    ///
    /// let metric = GenerationMetric::new();
    /// assert_eq!(metric.name(), "generation");
    /// ```
    pub fn new() -> Self {
        Self {
            predictions: Vec::new(),
            references: Vec::new(),
        }
    }

    pub fn predictions(&self) -> &Vec<String> {
        &self.predictions
    }

    pub fn references(&self) -> &Vec<String> {
        &self.references
    }
}

impl Metric for GenerationMetric {
    /// Add a batch of generated texts and references
    ///
    /// Accumulates text generation data for later metric computation. Both
    /// predictions and references must be text strings.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Generated text strings from the model
    /// * `references` - Ground truth reference text strings
    ///
    /// # Input Format Requirements
    ///
    /// - **Text**: Vector of strings containing generated and reference texts
    /// - Both predictions and references must have the same length
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error if input formats are incompatible.
    ///
    /// # Errors
    ///
    /// - `InvalidInput`: If input types are not both Text variants
    /// - The length mismatch is handled during computation, not here
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// use trustformers::auto::metrics::{GenerationMetric, MetricInput, Metric};
    ///
    /// let mut metric = GenerationMetric::new();
    ///
    /// let predictions = MetricInput::Text(vec![
    ///     "Generated text one".to_string(),
    ///     "Generated text two".to_string(),
    /// ]);
    /// let references = MetricInput::Text(vec![
    ///     "Reference text one".to_string(),
    ///     "Reference text two".to_string(),
    /// ]);
    ///
    /// metric.add_batch(&predictions, &references)?;

    fn add_batch(&mut self, predictions: &MetricInput, references: &MetricInput) -> Result<()> {
        match (predictions, references) {
            (MetricInput::Text(pred), MetricInput::Text(ref_)) => {
                self.predictions.extend(pred.clone());
                self.references.extend(ref_.clone());
                Ok(())
            },
            _ => Err(TrustformersError::invalid_input_simple("Invalid input types for generation metric: expected Text for both predictions and references".to_string()
            )),
        }
    }

    /// Compute generation metrics
    ///
    /// Calculates BLEU-like scores based on word overlap between generated
    /// and reference texts. Uses F1 score aggregation for balanced evaluation.
    ///
    /// # Returns
    ///
    /// `MetricResult` containing:
    /// - **Primary value**: Average F1 score across all text pairs
    /// - **Details**:
    ///   - `bleu_like`: The computed BLEU-like score (same as primary value)
    ///
    /// # Errors
    ///
    /// - `InvalidInput`: If no data has been added
    /// - Note: Length mismatches are handled gracefully (shorter list determines pairs)
    ///
    /// # Algorithm Details
    ///
    /// For each (prediction, reference) pair:
    /// 1. Split texts into words using whitespace
    /// 2. Count overlapping words (case-sensitive)
    /// 3. Calculate precision: `matches / prediction_words`
    /// 4. Calculate recall: `matches / reference_words`
    /// 5. Calculate F1: `2 * precision * recall / (precision + recall)`
    /// 6. Average F1 scores across all pairs
    ///
    /// # Edge Cases
    ///
    /// - Empty predictions or references result in 0 score for that pair
    /// - Length mismatches use the shorter of the two lists
    /// - NaN results from division are handled as 0
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// use trustformers::auto::metrics::{GenerationMetric, MetricInput, Metric};
    ///
    /// let mut metric = GenerationMetric::new();
    /// metric.add_batch(
    ///     &MetricInput::Text(vec!["hello world".to_string()]),
    ///     &MetricInput::Text(vec!["hello universe".to_string()])
    /// )?;
    ///
    /// let result = metric.compute()?;
    /// assert_eq!(result.name, "generation");
    /// assert!(result.value >= 0.0 && result.value <= 1.0);

    fn compute(&self) -> Result<MetricResult> {
        if self.predictions.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "No data available for metric computation".to_string(),
            ));
        }

        // Delegate to NlpAdapter::bleu(4, true) for production-quality BLEU-4 scoring.
        let mut bleu_adapter = NlpAdapter::bleu(4, true);
        let preds = MetricInput::Text(self.predictions.clone());
        let refs = MetricInput::Text(self.references.clone());
        bleu_adapter.add_batch(&preds, &refs)?;
        let bleu_result = bleu_adapter.compute()?;
        let bleu_score = bleu_result.value;

        let mut details = HashMap::new();
        details.insert("bleu_like".to_string(), bleu_score);
        details.insert("bleu".to_string(), bleu_score);

        Ok(MetricResult {
            name: "generation".to_string(),
            value: bleu_score,
            details,
            metadata: HashMap::new(),
        })
    }

    /// Reset the metric state
    ///
    /// Clears all accumulated predictions and references, preparing the metric
    /// for a new evaluation run. This is more efficient than creating a new
    /// metric instance.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// use trustformers::auto::metrics::{GenerationMetric, MetricInput, Metric};
    ///
    /// let mut metric = GenerationMetric::new();
    /// metric.add_batch(
    ///     &MetricInput::Text(vec!["hello".to_string()]),
    ///     &MetricInput::Text(vec!["world".to_string()])
    /// )?;
    ///
    /// metric.reset();
    /// // Metric is now ready for new data

    fn reset(&mut self) {
        self.predictions.clear();
        self.references.clear();
    }

    /// Get the metric name
    ///
    /// Returns the identifier for this metric type, used in logging and results.
    ///
    /// # Returns
    ///
    /// String slice "generation"
    fn name(&self) -> &str {
        "generation"
    }
}

impl Default for GenerationMetric {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_metric_basic() {
        let mut metric = GenerationMetric::new();

        let predictions = MetricInput::Text(vec![
            "the quick brown fox".to_string(),
            "hello world".to_string(),
        ]);
        let references = MetricInput::Text(vec![
            "the quick brown fox jumps".to_string(),
            "hello world test".to_string(),
        ]);

        metric.add_batch(&predictions, &references).expect("add operation failed");

        let result = metric.compute().expect("operation failed in test");
        assert_eq!(result.name, "generation");
        assert!(result.value >= 0.0 && result.value <= 1.0);
        assert!(result.details.contains_key("bleu_like"));
    }

    #[test]
    fn test_generation_metric_perfect_match() {
        let mut metric = GenerationMetric::new();

        // Use a longer sentence so all 4-gram orders are present → BLEU-4 = 1.0.
        let predictions = MetricInput::Text(vec![
            "the quick brown fox jumps over the lazy dog".to_string()
        ]);
        let references = MetricInput::Text(vec![
            "the quick brown fox jumps over the lazy dog".to_string()
        ]);

        metric.add_batch(&predictions, &references).expect("add operation failed");

        let result = metric.compute().expect("operation failed in test");
        // Perfect match on a ≥4-token sentence: BLEU-4 with smoothing = 1.0
        assert!(
            (result.value - 1.0).abs() < 1e-6,
            "perfect match should give BLEU=1.0, got {}",
            result.value
        );
    }

    #[test]
    fn test_generation_metric_no_overlap() {
        let mut metric = GenerationMetric::new();

        let predictions = MetricInput::Text(vec!["foo bar".to_string()]);
        let references = MetricInput::Text(vec!["baz qux".to_string()]);

        metric.add_batch(&predictions, &references).expect("add operation failed");

        let result = metric.compute().expect("operation failed in test");
        // No token overlap → BLEU is very low (smoothing may yield a small positive value)
        assert!(
            result.value < 0.5,
            "no-overlap BLEU should be low, got {}",
            result.value
        );
    }

    #[test]
    fn test_generation_metric_empty_text() {
        let mut metric = GenerationMetric::new();

        let predictions = MetricInput::Text(vec!["".to_string()]);
        let references = MetricInput::Text(vec!["hello world".to_string()]);

        metric.add_batch(&predictions, &references).expect("add operation failed");

        let result = metric.compute().expect("operation failed in test");
        // Empty hypothesis → BLEU = 0.0
        assert_eq!(result.value, 0.0, "empty hypothesis should give BLEU=0.0");
    }

    #[test]
    fn test_generation_metric_partial_overlap() {
        let mut metric = GenerationMetric::new();

        let predictions = MetricInput::Text(vec!["hello world test".to_string()]);
        let references = MetricInput::Text(vec!["hello universe test".to_string()]);

        metric.add_batch(&predictions, &references).expect("add operation failed");

        let result = metric.compute().expect("operation failed in test");
        // Should have some overlap but not perfect
        assert!(result.value > 0.0 && result.value < 1.0);
    }

    #[test]
    fn test_generation_metric_reset() {
        let mut metric = GenerationMetric::new();

        let predictions = MetricInput::Text(vec!["hello".to_string()]);
        let references = MetricInput::Text(vec!["world".to_string()]);
        metric.add_batch(&predictions, &references).expect("add operation failed");

        metric.reset();

        // Should fail because no data
        assert!(metric.compute().is_err());
    }

    #[test]
    fn test_generation_metric_invalid_input() {
        let mut metric = GenerationMetric::new();

        let predictions = MetricInput::Classifications(vec![0, 1]);
        let references = MetricInput::Text(vec!["hello".to_string()]);

        let result = metric.add_batch(&predictions, &references);
        assert!(result.is_err());
    }

    #[test]
    fn test_generation_metric_multiple_batches() {
        let mut metric = GenerationMetric::new();

        // First batch
        metric
            .add_batch(
                &MetricInput::Text(vec!["hello world".to_string()]),
                &MetricInput::Text(vec!["hello world".to_string()]),
            )
            .expect("operation failed in test");

        // Second batch
        metric
            .add_batch(
                &MetricInput::Text(vec!["foo bar".to_string()]),
                &MetricInput::Text(vec!["baz qux".to_string()]),
            )
            .expect("operation failed in test");

        let result = metric.compute().expect("operation failed in test");
        // Sentence-level BLEU averaged: one perfect pair, one no-overlap pair.
        // With smoothing the no-overlap pair yields a small positive value,
        // so the average is in (0, 1) and below the perfect pair's score of 1.0.
        assert!(
            result.value > 0.0 && result.value < 1.0,
            "averaged BLEU should be in (0, 1), got {}",
            result.value
        );
    }
}
