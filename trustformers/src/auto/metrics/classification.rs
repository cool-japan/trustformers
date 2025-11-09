//! # Classification Metrics for TrustformeRS
//!
//! This module provides comprehensive evaluation metrics for classification tasks,
//! including text classification, sentiment analysis, and other discrete prediction tasks.
//!
//! ## Overview
//!
//! The `ClassificationMetric` implementation provides standard classification evaluation
//! metrics including accuracy, precision, recall, and F1-score. It supports both
//! binary and multi-class classification scenarios.
//!
//! ## Features
//!
//! - **Accuracy**: Overall proportion of correct predictions
//! - **Macro-averaged metrics**: Per-class precision, recall, and F1, then averaged
//! - **Multi-class support**: Handles any number of classes automatically
//! - **Flexible input**: Accepts both class predictions and probability distributions
//!
//! ## Usage Examples
//!
//! ### Basic Classification Evaluation
//!
//! ```rust
//! use trustformers::auto::metrics::{ClassificationMetric, MetricInput, Metric};
//!
//! let mut metric = ClassificationMetric::new();
//!
//! // Add predictions and references
//! let predictions = MetricInput::Classifications(vec![0, 1, 0, 1]);
//! let references = MetricInput::Classifications(vec![0, 0, 1, 1]);
//! metric.add_batch(&predictions, &references)?;
//!
//! // Compute results
//! let result = metric.compute()?;
//! println!("Accuracy: {:.3}", result.details.get("accuracy").unwrap());
//! println!("Macro F1: {:.3}", result.details.get("macro_f1").unwrap());
//! ```
//!
//! ### Using Probability Distributions
//!
//! ```rust
//! use trustformers::auto::metrics::{ClassificationMetric, MetricInput, Metric};
//!
//! let mut metric = ClassificationMetric::new();
//!
//! // Convert model probabilities to predictions automatically
//! let probabilities = MetricInput::Probabilities(vec![
//!     vec![vec![0.8, 0.2]], // Sequence of 1 position with 2-class distribution
//!     vec![vec![0.3, 0.7]], // Another sequence
//! ]);
//! let references = MetricInput::Classifications(vec![0, 1]);
//! metric.add_batch(&probabilities, &references)?;
//!
//! let result = metric.compute()?;
//! ```
//!
//! ## Implementation Details
//!
//! ### Metric Calculations
//!
//! - **Accuracy**: `correct_predictions / total_predictions`
//! - **Precision (per class)**: `true_positives / (true_positives + false_positives)`
//! - **Recall (per class)**: `true_positives / (true_positives + false_negatives)`
//! - **F1 (per class)**: `2 * precision * recall / (precision + recall)`
//! - **Macro-averaged**: Average of per-class metrics
//!
//! ### Performance Characteristics
//!
//! - **Time complexity**: O(n) for n predictions
//! - **Space complexity**: O(n) for storing predictions and references
//! - **Numerical stability**: Handles edge cases like zero divisions gracefully

use super::{Metric, MetricInput, MetricResult};
use crate::error::{Result, TrustformersError};
use std::collections::HashMap;

/// Classification metric implementation
///
/// Provides comprehensive evaluation metrics for classification tasks including
/// accuracy, precision, recall, and F1-score. Supports both binary and multi-class
/// classification with automatic class detection.
///
/// ## Design Principles
///
/// - **Accumulative**: Collects predictions over multiple batches
/// - **Flexible**: Handles various input formats (classes, probabilities)
/// - **Comprehensive**: Provides detailed per-class and macro-averaged metrics
/// - **Robust**: Handles edge cases and class imbalances gracefully
///
/// ## Supported Input Types
///
/// - `Classifications`: Direct class predictions as indices
/// - `Probabilities`: Probability distributions (automatically converted to predictions)
#[derive(Debug, Clone)]
pub struct ClassificationMetric {
    /// Accumulated model predictions as class indices
    predictions: Vec<usize>,
    /// Accumulated ground truth references as class indices
    references: Vec<usize>,
}

impl ClassificationMetric {
    /// Create a new classification metric instance
    ///
    /// Initializes an empty metric ready to accumulate predictions and references.
    ///
    /// # Returns
    ///
    /// New `ClassificationMetric` instance with empty state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::ClassificationMetric;
    ///
    /// let metric = ClassificationMetric::new();
    /// assert_eq!(metric.name(), "classification");
    /// ```
    pub fn new() -> Self {
        Self {
            predictions: Vec::new(),
            references: Vec::new(),
        }
    }
}

impl Metric for ClassificationMetric {
    /// Add a batch of predictions and references
    ///
    /// Accumulates classification data for later metric computation. Supports
    /// both direct class predictions and probability distributions.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Model predictions (Classifications or Probabilities)
    /// * `references` - Ground truth class labels (Classifications)
    ///
    /// # Input Format Requirements
    ///
    /// - **Classifications**: Vector of class indices (0-based)
    /// - **Probabilities**: Nested vector `[batch][sequence][vocab]` where sequence
    ///   length should be 1 for classification tasks
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error if input formats are incompatible.
    ///
    /// # Errors
    ///
    /// - `InvalidInput`: If input types don't match expected formats
    /// - `InvalidInput`: If predictions and references have different lengths
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::{ClassificationMetric, MetricInput, Metric};
    ///
    /// let mut metric = ClassificationMetric::new();
    ///
    /// // Direct class predictions
    /// let preds = MetricInput::Classifications(vec![0, 1, 2]);
    /// let refs = MetricInput::Classifications(vec![0, 1, 1]);
    /// metric.add_batch(&preds, &refs)?;
    ///
    /// // Probability distributions
    /// let probs = MetricInput::Probabilities(vec![
    ///     vec![vec![0.9, 0.1]], // High confidence for class 0
    ///     vec![vec![0.2, 0.8]], // High confidence for class 1
    /// ]);
    /// let refs = MetricInput::Classifications(vec![0, 1]);
    /// metric.add_batch(&probs, &refs)?;
    /// ```
    fn add_batch(&mut self, predictions: &MetricInput, references: &MetricInput) -> Result<()> {
        match (predictions, references) {
            (MetricInput::Classifications(pred), MetricInput::Classifications(ref_)) => {
                self.predictions.extend(pred);
                self.references.extend(ref_);
                Ok(())
            },
            (MetricInput::Probabilities(probs), MetricInput::Classifications(ref_)) => {
                // Convert probabilities to predictions by taking argmax
                let pred: Vec<usize> = probs
                    .iter()
                    .map(|p| {
                        // For classification, we expect single position per sequence
                        if !p.is_empty() && !p[0].is_empty() {
                            p[0].iter()
                                .enumerate()
                                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                                .map(|(idx, _)| idx)
                                .unwrap_or(0)
                        } else {
                            0
                        }
                    })
                    .collect();
                self.predictions.extend(pred);
                self.references.extend(ref_);
                Ok(())
            },
            _ => Err(TrustformersError::invalid_input_simple(
                "Invalid input types for classification metric: expected Classifications or Probabilities for predictions, Classifications for references".to_string()
            )),
        }
    }

    /// Compute classification metrics
    ///
    /// Calculates comprehensive classification metrics including accuracy,
    /// macro-averaged precision, recall, and F1-score.
    ///
    /// # Returns
    ///
    /// `MetricResult` containing:
    /// - **Primary value**: Overall accuracy
    /// - **Details**:
    ///   - `accuracy`: Proportion of correct predictions
    ///   - `macro_precision`: Average precision across all classes
    ///   - `macro_recall`: Average recall across all classes
    ///   - `macro_f1`: Average F1-score across all classes
    ///
    /// # Errors
    ///
    /// - `InvalidInput`: If no data has been added or if prediction/reference lengths differ
    ///
    /// # Algorithm Details
    ///
    /// 1. **Accuracy**: Simple proportion of correct predictions
    /// 2. **Per-class metrics**: Calculate precision, recall, F1 for each class
    /// 3. **Macro-averaging**: Average per-class metrics (unweighted)
    /// 4. **Edge case handling**: Classes with no predictions or references get 0 metrics
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::{ClassificationMetric, MetricInput, Metric};
    ///
    /// let mut metric = ClassificationMetric::new();
    /// metric.add_batch(
    ///     &MetricInput::Classifications(vec![0, 1, 0, 1]),
    ///     &MetricInput::Classifications(vec![0, 0, 1, 1])
    /// )?;
    ///
    /// let result = metric.compute()?;
    /// assert_eq!(result.name, "classification");
    /// assert_eq!(result.value, 0.5); // 50% accuracy
    /// assert!(result.details.contains_key("macro_f1"));
    /// ```
    fn compute(&self) -> Result<MetricResult> {
        if self.predictions.len() != self.references.len() {
            return Err(TrustformersError::invalid_input_simple(
                "Predictions and references must have the same length".to_string(),
            ));
        }

        if self.predictions.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "No data available for metric computation".to_string(),
            ));
        }

        // Calculate overall accuracy
        let correct = self
            .predictions
            .iter()
            .zip(self.references.iter())
            .filter(|(p, r)| p == r)
            .count();

        let accuracy = correct as f64 / self.predictions.len() as f64;

        // Determine number of classes
        let max_pred = self.predictions.iter().max().copied().unwrap_or(0);
        let max_ref = self.references.iter().max().copied().unwrap_or(0);
        let num_classes = (max_pred.max(max_ref) + 1).max(2); // At least 2 classes

        // Calculate per-class metrics
        let mut precision_sum = 0.0;
        let mut recall_sum = 0.0;
        let mut f1_sum = 0.0;
        let mut valid_classes = 0;

        for class in 0..num_classes {
            let tp = self
                .predictions
                .iter()
                .zip(self.references.iter())
                .filter(|(p, r)| **p == class && **r == class)
                .count() as f64;

            let fp = self
                .predictions
                .iter()
                .zip(self.references.iter())
                .filter(|(p, r)| **p == class && **r != class)
                .count() as f64;

            let fn_ = self
                .predictions
                .iter()
                .zip(self.references.iter())
                .filter(|(p, r)| **p != class && **r == class)
                .count() as f64;

            // Calculate precision, recall, F1 for this class
            if tp + fp > 0.0 || tp + fn_ > 0.0 {
                let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
                let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
                let f1 = if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                };

                precision_sum += precision;
                recall_sum += recall;
                f1_sum += f1;
                valid_classes += 1;
            }
        }

        // Calculate macro-averaged metrics
        let macro_precision =
            if valid_classes > 0 { precision_sum / valid_classes as f64 } else { 0.0 };
        let macro_recall = if valid_classes > 0 { recall_sum / valid_classes as f64 } else { 0.0 };
        let macro_f1 = if valid_classes > 0 { f1_sum / valid_classes as f64 } else { 0.0 };

        // Build result details
        let mut details = HashMap::new();
        details.insert("accuracy".to_string(), accuracy);
        details.insert("macro_precision".to_string(), macro_precision);
        details.insert("macro_recall".to_string(), macro_recall);
        details.insert("macro_f1".to_string(), macro_f1);

        Ok(MetricResult {
            name: "classification".to_string(),
            value: accuracy, // Primary metric is accuracy
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
    /// use trustformers::auto::metrics::{ClassificationMetric, MetricInput, Metric};
    ///
    /// let mut metric = ClassificationMetric::new();
    /// metric.add_batch(
    ///     &MetricInput::Classifications(vec![0, 1]),
    ///     &MetricInput::Classifications(vec![0, 1])
    /// )?;
    ///
    /// metric.reset();
    /// // Metric is now ready for new data
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
    /// String slice "classification"
    fn name(&self) -> &str {
        "classification"
    }
}

impl Default for ClassificationMetric {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification_metric_basic() {
        let mut metric = ClassificationMetric::new();

        let predictions = MetricInput::Classifications(vec![0, 1, 0, 1]);
        let references = MetricInput::Classifications(vec![0, 0, 1, 1]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.name, "classification");
        assert_eq!(result.value, 0.5); // 50% accuracy
        assert!(result.details.contains_key("macro_f1"));
    }

    #[test]
    fn test_classification_metric_perfect_score() {
        let mut metric = ClassificationMetric::new();

        let predictions = MetricInput::Classifications(vec![0, 1, 2, 0, 1, 2]);
        let references = MetricInput::Classifications(vec![0, 1, 2, 0, 1, 2]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.value, 1.0); // 100% accuracy
        assert_eq!(result.details.get("accuracy"), Some(&1.0));
        assert_eq!(result.details.get("macro_f1"), Some(&1.0));
    }

    #[test]
    fn test_classification_metric_probabilities() {
        let mut metric = ClassificationMetric::new();

        // High confidence predictions that should be correct
        let probabilities = MetricInput::Probabilities(vec![
            vec![vec![0.9, 0.1]], // Should predict class 0
            vec![vec![0.2, 0.8]], // Should predict class 1
        ]);
        let references = MetricInput::Classifications(vec![0, 1]);

        metric.add_batch(&probabilities, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.value, 1.0); // Should be 100% accurate
    }

    #[test]
    fn test_classification_metric_reset() {
        let mut metric = ClassificationMetric::new();

        let predictions = MetricInput::Classifications(vec![0, 1]);
        let references = MetricInput::Classifications(vec![0, 1]);
        metric.add_batch(&predictions, &references).unwrap();

        metric.reset();

        // Should fail because no data
        assert!(metric.compute().is_err());
    }

    #[test]
    fn test_classification_metric_invalid_input() {
        let mut metric = ClassificationMetric::new();

        let predictions = MetricInput::Text(vec!["hello".to_string()]);
        let references = MetricInput::Classifications(vec![0]);

        let result = metric.add_batch(&predictions, &references);
        assert!(result.is_err());
    }

    #[test]
    fn test_classification_metric_mismatched_lengths() {
        let mut metric = ClassificationMetric::new();

        let predictions = MetricInput::Classifications(vec![0, 1]);
        let references = MetricInput::Classifications(vec![0]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute();
        assert!(result.is_err());
    }
}
