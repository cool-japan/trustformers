//! # Question Answering Metrics for TrustformeRS
//!
//! This module provides evaluation metrics for question answering tasks, including
//! extractive QA, reading comprehension, and answer generation systems.
//!
//! ## Overview
//!
//! The `QuestionAnsweringMetric` implementation provides specialized evaluation for
//! question answering models using Exact Match (EM) and token-level F1 scores,
//! which are the standard metrics for QA evaluation.
//!
//! ## Features
//!
//! - **Exact Match (EM)**: Measures percentage of predictions that match exactly
//! - **Token-level F1**: Measures overlap between predicted and reference answer tokens
//! - **Answer normalization**: Standardizes text for fair comparison
//! - **Combined scoring**: Averages EM and F1 for comprehensive evaluation
//!
//! ## Usage Examples
//!
//! ### Basic QA Evaluation
//!
//! ```rust
//! use trustformers::auto::metrics::{QuestionAnsweringMetric, MetricInput, Metric};
//!
//! let mut metric = QuestionAnsweringMetric::new();
//!
//! // QA predictions and references
//! let predictions = MetricInput::Text(vec![
//!     "Barack Obama".to_string(),
//!     "July 20, 1969".to_string(),
//! ]);
//! let references = MetricInput::Text(vec![
//!     "Barack Hussein Obama".to_string(),
//!     "July 20, 1969".to_string(),
//! ]);
//!
//! metric.add_batch(&predictions, &references)?;
//!
//! // Compute results
//! let result = metric.compute()?;
//! println!("QA Score: {:.3}", result.value);
//! println!("Exact Match: {:.3}", result.details.get("exact_match").unwrap());
//! println!("F1: {:.3}", result.details.get("f1").unwrap());
//! ```
//!
//! ### Reading Comprehension Evaluation
//!
//! ```rust
//! use trustformers::auto::metrics::{QuestionAnsweringMetric, MetricInput, Metric};
//!
//! let mut metric = QuestionAnsweringMetric::new();
//!
//! // Model answers vs gold standard answers
//! let predictions = MetricInput::Text(vec![
//!     "The capital of France".to_string(),
//!     "Machine learning algorithm".to_string(),
//! ]);
//! let references = MetricInput::Text(vec![
//!     "Paris is the capital of France".to_string(),
//!     "A machine learning algorithm".to_string(),
//! ]);
//!
//! metric.add_batch(&predictions, &references)?;
//! let result = metric.compute()?;
//! ```
//!
//! ## Implementation Details
//!
//! ### Evaluation Metrics
//!
//! 1. **Exact Match (EM)**:
//!    - Percentage of predictions that match references exactly after normalization
//!    - Binary: either 0 (no match) or 1 (exact match) for each sample
//!    - Final score is the average across all samples
//!
//! 2. **Token-level F1**:
//!    - Precision: `common_tokens / predicted_tokens`
//!    - Recall: `common_tokens / reference_tokens`
//!    - F1: `2 * precision * recall / (precision + recall)`
//!    - Averaged across all samples
//!
//! ### Answer Normalization
//!
//! Before comparison, answers are normalized by:
//! - Converting to lowercase
//! - Removing non-alphanumeric characters (except spaces)
//! - Collapsing multiple spaces to single spaces
//! - Trimming leading/trailing whitespace
//!
//! This ensures fair comparison despite formatting differences.
//!
//! ### Performance Characteristics
//!
//! - **Time complexity**: O(n*m) where n and m are token counts in answers
//! - **Space complexity**: O(n) for storing answer pairs
//! - **Normalization cost**: Minimal preprocessing overhead

use super::{Metric, MetricInput, MetricResult};
use crate::error::{Result, TrustformersError};
use std::collections::HashMap;

/// Question answering metric implementation
///
/// Provides evaluation metrics for question answering tasks using Exact Match
/// and token-level F1 scores, following standard QA evaluation practices.
///
/// ## Design Principles
///
/// - **Standard metrics**: Implements widely-accepted QA evaluation measures
/// - **Fair comparison**: Normalizes answers to handle formatting differences
/// - **Comprehensive**: Combines exact matching with fuzzy token-level scoring
/// - **Accumulative**: Supports batch-wise evaluation for large datasets
///
/// ## Supported Input Types
///
/// - `Text`: Predicted answers and reference answers as strings
///
/// ## Metric Interpretation
///
/// - **Exact Match**: Higher values indicate more precise predictions
/// - **F1 Score**: Higher values indicate better token-level overlap
/// - **Combined Score**: Average of EM and F1 for balanced evaluation
#[derive(Debug, Clone)]
pub struct QuestionAnsweringMetric {
    /// Accumulated predicted answers
    predictions: Vec<String>,
    /// Accumulated reference answers
    references: Vec<String>,
}

impl QuestionAnsweringMetric {
    /// Create a new question answering metric instance
    ///
    /// Initializes an empty metric ready to accumulate answer predictions and references.
    ///
    /// # Returns
    ///
    /// New `QuestionAnsweringMetric` instance with empty state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::QuestionAnsweringMetric;
    ///
    /// let metric = QuestionAnsweringMetric::new();
    /// assert_eq!(metric.name(), "question_answering");
    /// ```
    pub fn new() -> Self {
        Self {
            predictions: Vec::new(),
            references: Vec::new(),
        }
    }

    /// Normalize an answer string for comparison
    ///
    /// Applies standard normalization to ensure fair comparison between
    /// predicted and reference answers, handling common formatting variations.
    ///
    /// # Arguments
    ///
    /// * `s` - The answer string to normalize
    ///
    /// # Returns
    ///
    /// Normalized string ready for comparison
    ///
    /// # Normalization Steps
    ///
    /// 1. Convert to lowercase
    /// 2. Keep only alphanumeric characters and spaces
    /// 3. Collapse multiple spaces to single spaces
    /// 4. Trim leading and trailing whitespace
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::QuestionAnsweringMetric;
    ///
    /// let metric = QuestionAnsweringMetric::new();
    /// let normalized = metric.normalize_answer("  Barack H. Obama!  ");
    /// assert_eq!(normalized, "barack h obama");
    /// ```
    fn normalize_answer(&self, s: &str) -> String {
        s.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ")
    }
}

impl Metric for QuestionAnsweringMetric {
    /// Add a batch of answer predictions and references
    ///
    /// Accumulates question answering data for later metric computation.
    /// Both predictions and references must be text strings representing answers.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Predicted answers from the QA model (Text)
    /// * `references` - Ground truth reference answers (Text)
    ///
    /// # Input Format Requirements
    ///
    /// - **Text**: Vector of strings containing predicted and reference answers
    /// - Both predictions and references should have the same length
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
    /// ```rust
    /// use trustformers::auto::metrics::{QuestionAnsweringMetric, MetricInput, Metric};
    ///
    /// let mut metric = QuestionAnsweringMetric::new();
    ///
    /// let predictions = MetricInput::Text(vec![
    ///     "Paris".to_string(),
    ///     "Albert Einstein".to_string(),
    /// ]);
    /// let references = MetricInput::Text(vec![
    ///     "Paris, France".to_string(),
    ///     "Einstein".to_string(),
    /// ]);
    ///
    /// metric.add_batch(&predictions, &references)?;
    /// ```
    fn add_batch(&mut self, predictions: &MetricInput, references: &MetricInput) -> Result<()> {
        match (predictions, references) {
            (MetricInput::Text(pred), MetricInput::Text(ref_)) => {
                self.predictions.extend(pred.clone());
                self.references.extend(ref_.clone());
                Ok(())
            },
            _ => Err(TrustformersError::invalid_input_simple("Invalid input types for QA metric: expected Text for both predictions and references".to_string()
            )),
        }
    }

    /// Compute question answering metrics
    ///
    /// Calculates Exact Match and token-level F1 scores for question answering
    /// evaluation, following standard QA evaluation practices.
    ///
    /// # Returns
    ///
    /// `MetricResult` containing:
    /// - **Primary value**: Average of Exact Match and F1 scores
    /// - **Details**:
    ///   - `exact_match`: Percentage of predictions matching exactly
    ///   - `f1`: Average token-level F1 score across all samples
    ///
    /// # Errors
    ///
    /// - `InvalidInput`: If no data has been added to the metric
    /// - Note: Length mismatches are handled gracefully (shorter list determines pairs)
    ///
    /// # Algorithm Details
    ///
    /// For each (prediction, reference) pair:
    /// 1. **Normalization**: Apply answer normalization to both strings
    /// 2. **Exact Match**: Check if normalized strings are identical
    /// 3. **Token F1**:
    ///    - Tokenize by whitespace
    ///    - Count overlapping tokens
    ///    - Calculate precision, recall, and F1
    /// 4. **Aggregation**: Average EM and F1 across all pairs
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::{QuestionAnsweringMetric, MetricInput, Metric};
    ///
    /// let mut metric = QuestionAnsweringMetric::new();
    /// metric.add_batch(
    ///     &MetricInput::Text(vec!["Barack Obama".to_string()]),
    ///     &MetricInput::Text(vec!["Barack Hussein Obama".to_string()])
    /// )?;
    ///
    /// let result = metric.compute()?;
    /// assert_eq!(result.name, "question_answering");
    /// assert!(result.value >= 0.0 && result.value <= 1.0);
    /// assert!(result.details.contains_key("exact_match"));
    /// assert!(result.details.contains_key("f1"));
    /// ```
    fn compute(&self) -> Result<MetricResult> {
        if self.predictions.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "No data available for metric computation".to_string(),
            ));
        }

        let mut exact_matches = 0;
        let mut total_f1 = 0.0;
        let num_pairs = self.predictions.len().min(self.references.len());

        for (pred, ref_) in self.predictions.iter().zip(self.references.iter()) {
            // Normalize strings for comparison
            let pred_norm = self.normalize_answer(pred);
            let ref_norm = self.normalize_answer(ref_);

            // Exact match check
            if pred_norm == ref_norm {
                exact_matches += 1;
            }

            // Token-level F1 calculation
            let pred_tokens: Vec<&str> = pred_norm.split_whitespace().collect();
            let ref_tokens: Vec<&str> = ref_norm.split_whitespace().collect();

            // Count common tokens
            let mut common = 0;
            for token in &pred_tokens {
                if ref_tokens.contains(token) {
                    common += 1;
                }
            }

            // Calculate precision, recall, and F1
            let precision = if pred_tokens.is_empty() {
                0.0
            } else {
                common as f64 / pred_tokens.len() as f64
            };

            let recall = if ref_tokens.is_empty() {
                0.0
            } else {
                common as f64 / ref_tokens.len() as f64
            };

            let f1 = if precision + recall == 0.0 {
                0.0
            } else {
                2.0 * precision * recall / (precision + recall)
            };

            total_f1 += f1;
        }

        // Calculate final scores
        let exact_match_score =
            if num_pairs > 0 { exact_matches as f64 / num_pairs as f64 } else { 0.0 };

        let avg_f1 = if num_pairs > 0 { total_f1 / num_pairs as f64 } else { 0.0 };

        // Build result details
        let mut details = HashMap::new();
        details.insert("exact_match".to_string(), exact_match_score);
        details.insert("f1".to_string(), avg_f1);

        Ok(MetricResult {
            name: "question_answering".to_string(),
            value: (exact_match_score + avg_f1) / 2.0, // Average of EM and F1
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
    /// ```rust
    /// use trustformers::auto::metrics::{QuestionAnsweringMetric, MetricInput, Metric};
    ///
    /// let mut metric = QuestionAnsweringMetric::new();
    /// metric.add_batch(
    ///     &MetricInput::Text(vec!["Paris".to_string()]),
    ///     &MetricInput::Text(vec!["Paris, France".to_string()])
    /// )?;
    ///
    /// metric.reset();
    /// // Metric is now ready for new answer pairs
    /// ```
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
    /// String slice "question_answering"
    fn name(&self) -> &str {
        "question_answering"
    }
}

impl Default for QuestionAnsweringMetric {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qa_metric_basic() {
        let mut metric = QuestionAnsweringMetric::new();

        let predictions = MetricInput::Text(vec![
            "Barack Obama".to_string(),
            "July 20, 1969".to_string(),
        ]);
        let references = MetricInput::Text(vec![
            "Barack Hussein Obama".to_string(),
            "July 20, 1969".to_string(),
        ]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.name, "question_answering");
        assert!(result.value >= 0.0 && result.value <= 1.0);
        assert!(result.details.contains_key("exact_match"));
        assert!(result.details.contains_key("f1"));
    }

    #[test]
    fn test_qa_metric_perfect_match() {
        let mut metric = QuestionAnsweringMetric::new();

        let predictions = MetricInput::Text(vec!["Barack Obama".to_string()]);
        let references = MetricInput::Text(vec!["Barack Obama".to_string()]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.details.get("exact_match"), Some(&1.0));
        assert_eq!(result.details.get("f1"), Some(&1.0));
        assert_eq!(result.value, 1.0); // Average of 1.0 and 1.0
    }

    #[test]
    fn test_qa_metric_no_match() {
        let mut metric = QuestionAnsweringMetric::new();

        let predictions = MetricInput::Text(vec!["completely different".to_string()]);
        let references = MetricInput::Text(vec!["Barack Obama".to_string()]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.details.get("exact_match"), Some(&0.0));
        assert_eq!(result.details.get("f1"), Some(&0.0));
        assert_eq!(result.value, 0.0); // Average of 0.0 and 0.0
    }

    #[test]
    fn test_qa_metric_partial_match() {
        let mut metric = QuestionAnsweringMetric::new();

        let predictions = MetricInput::Text(vec!["Barack Obama".to_string()]);
        let references = MetricInput::Text(vec!["Barack Hussein Obama".to_string()]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.details.get("exact_match"), Some(&0.0)); // No exact match

        // F1 should be > 0 due to overlapping tokens
        let f1 = result.details.get("f1").unwrap();
        assert!(*f1 > 0.0 && *f1 < 1.0);
    }

    #[test]
    fn test_qa_metric_normalization() {
        let mut metric = QuestionAnsweringMetric::new();

        let predictions = MetricInput::Text(vec!["  Barack H. Obama!  ".to_string()]);
        let references = MetricInput::Text(vec!["Barack H Obama".to_string()]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.details.get("exact_match"), Some(&1.0)); // Should match after normalization
    }

    #[test]
    fn test_qa_metric_empty_answer() {
        let mut metric = QuestionAnsweringMetric::new();

        let predictions = MetricInput::Text(vec!["".to_string()]);
        let references = MetricInput::Text(vec!["Barack Obama".to_string()]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.details.get("exact_match"), Some(&0.0));
        assert_eq!(result.details.get("f1"), Some(&0.0)); // No tokens to match
    }

    #[test]
    fn test_qa_metric_multiple_batches() {
        let mut metric = QuestionAnsweringMetric::new();

        // First batch - perfect match
        metric
            .add_batch(
                &MetricInput::Text(vec!["Barack Obama".to_string()]),
                &MetricInput::Text(vec!["Barack Obama".to_string()]),
            )
            .unwrap();

        // Second batch - no match
        metric
            .add_batch(
                &MetricInput::Text(vec!["different answer".to_string()]),
                &MetricInput::Text(vec!["Barack Obama".to_string()]),
            )
            .unwrap();

        let result = metric.compute().unwrap();
        // Should average: (1.0 + 0.0) / 2 = 0.5 for EM, (1.0 + 0.0) / 2 = 0.5 for F1
        assert_eq!(result.details.get("exact_match"), Some(&0.5));
        assert_eq!(result.details.get("f1"), Some(&0.5));
        assert_eq!(result.value, 0.5);
    }

    #[test]
    fn test_qa_metric_reset() {
        let mut metric = QuestionAnsweringMetric::new();

        metric
            .add_batch(
                &MetricInput::Text(vec!["Barack Obama".to_string()]),
                &MetricInput::Text(vec!["Barack Obama".to_string()]),
            )
            .unwrap();

        metric.reset();

        // Should fail because no data after reset
        assert!(metric.compute().is_err());
    }

    #[test]
    fn test_qa_metric_invalid_input() {
        let mut metric = QuestionAnsweringMetric::new();

        let predictions = MetricInput::Classifications(vec![0, 1]);
        let references = MetricInput::Text(vec!["Barack Obama".to_string()]);

        let result = metric.add_batch(&predictions, &references);
        assert!(result.is_err());
    }

    #[test]
    fn test_answer_normalization() {
        let metric = QuestionAnsweringMetric::new();

        // Test various normalization cases
        assert_eq!(
            metric.normalize_answer("  Barack H. Obama!  "),
            "barack h obama"
        );
        assert_eq!(metric.normalize_answer("PARIS"), "paris");
        assert_eq!(metric.normalize_answer("New York City"), "new york city");
        // Hyphens are punctuation and get removed (standard SQuAD normalization)
        assert_eq!(metric.normalize_answer("123-456-7890"), "1234567890");
        assert_eq!(metric.normalize_answer(""), "");
        assert_eq!(metric.normalize_answer("   "), "");
    }

    #[test]
    fn test_qa_metric_case_sensitivity() {
        let mut metric = QuestionAnsweringMetric::new();

        // Different cases should match after normalization
        let predictions = MetricInput::Text(vec!["BARACK OBAMA".to_string()]);
        let references = MetricInput::Text(vec!["barack obama".to_string()]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.details.get("exact_match"), Some(&1.0));
        assert_eq!(result.details.get("f1"), Some(&1.0));
    }
}
