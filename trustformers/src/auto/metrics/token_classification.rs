//! # Token Classification Metrics for TrustformeRS
//!
//! This module provides evaluation metrics for token classification tasks, including
//! Named Entity Recognition (NER), part-of-speech tagging, and sequence labeling.
//!
//! ## Overview
//!
//! The `TokenClassificationMetric` implementation provides evaluation for token-level
//! classification tasks using precision, recall, and F1 scores. It handles entity-level
//! evaluation for NER and other structured prediction tasks.
//!
//! ## Features
//!
//! - **Entity-level evaluation**: Evaluates complete entities, not just individual tokens
//! - **Precision/Recall/F1**: Standard metrics for classification quality
//! - **Multiple tag schemes**: Supports various tagging formats (IOB, IOBES, etc.)
//! - **Flexible input**: Handles different input formats for predictions and references
//!
//! ## Usage Examples
//!
//! ### Basic NER Evaluation
//!
//! ```rust,ignore
//! use trustformers::auto::metrics::{TokenClassificationMetric, MetricInput, Metric};
//!
//! let mut metric = TokenClassificationMetric::new();
//!
//! // Token-level predictions and references
//! let predictions = MetricInput::Spans(vec![
//!     vec![(0, 2, "PERSON".to_string()), (5, 8, "LOCATION".to_string())],
//!     vec![(1, 3, "ORG".to_string())],
//! ]);
//! let references = MetricInput::Spans(vec![
//!     vec![(0, 2, "PERSON".to_string()), (5, 8, "LOCATION".to_string())],
//!     vec![(1, 4, "ORG".to_string())], // Different span boundary
//! ]);
//!
//! metric.add_batch(&predictions, &references)?;
//!
//! // Compute results
//! let result = metric.compute()?;
//! println!("Token Classification F1: {:.3}", result.value);
//! println!("Precision: {:.3}", result.details.get("precision").unwrap());
//! println!("Recall: {:.3}", result.details.get("recall").unwrap());
//! ```
//!
//! ### Sequence Labeling with Tags
//!
//! ```rust,ignore
//! use trustformers::auto::metrics::{TokenClassificationMetric, MetricInput, Metric};
//!
//! let mut metric = TokenClassificationMetric::new();
//!
//! // Using text-based tag sequences (simplified for demonstration)
//! // In practice, you would convert IOB tags to spans first
//! let predictions = MetricInput::Text(vec![
//!     "B-PER I-PER O B-LOC O".to_string(),
//! ]);
//! let references = MetricInput::Text(vec![
//!     "B-PER I-PER O B-LOC O".to_string(),
//! ]);
//!
//! metric.add_batch(&predictions, &references)?;
//! let result = metric.compute()?;
//! ```
//!
//! ## Implementation Details
//!
//! ### Entity Evaluation
//!
//! Token classification is evaluated at the entity level:
//! - **Exact Match**: An entity is correct only if both boundaries and type match exactly
//! - **Precision**: `correct_entities / predicted_entities`
//! - **Recall**: `correct_entities / reference_entities`
//! - **F1**: `2 * precision * recall / (precision + recall)`
//!
//! ### Supported Input Formats
//!
//! 1. **Spans**: List of (start, end, label) tuples for each sequence
//! 2. **Text**: Tag sequences that can be parsed into entities
//!
//! ### Performance Characteristics
//!
//! - **Time complexity**: O(n*m) where n and m are numbers of predicted and reference entities
//! - **Space complexity**: O(n) for storing entity spans
//! - **Entity matching**: Fast exact matching for span boundaries and labels

use super::{Metric, MetricInput, MetricResult};
use crate::error::{Result, TrustformersError};
use std::collections::HashMap;

/// Token classification metric implementation
///
/// Provides evaluation metrics for token classification tasks with entity-level
/// evaluation using precision, recall, and F1 scores.
///
/// ## Design Principles
///
/// - **Entity-centric**: Evaluates complete entities rather than individual tokens
/// - **Strict matching**: Requires exact boundary and label agreement
/// - **Standard metrics**: Implements widely-used NER evaluation measures
/// - **Flexible input**: Supports multiple input formats for different use cases
///
/// ## Supported Input Types
///
/// - `Spans`: Entity spans as (start, end, label) tuples
/// - `Text`: Tag sequences (for backward compatibility and simple cases)
///
/// ## Evaluation Philosophy
///
/// Entity-level evaluation is stricter but more meaningful than token-level evaluation
/// because it requires models to correctly identify complete entities, not just
/// individual tokens within entities.
#[derive(Debug, Clone)]
pub struct TokenClassificationMetric {
    /// Accumulated predicted entity spans for each sequence
    predictions: Vec<Vec<String>>,
    /// Accumulated reference entity spans for each sequence
    references: Vec<Vec<String>>,
}

impl TokenClassificationMetric {
    /// Create a new token classification metric instance
    ///
    /// Initializes an empty metric ready to accumulate entity predictions and references.
    ///
    /// # Returns
    ///
    /// New `TokenClassificationMetric` instance with empty state.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use trustformers::auto::metrics::TokenClassificationMetric;
    ///
    /// let metric = TokenClassificationMetric::new();
    /// assert_eq!(metric.name(), "token_classification");
    /// ```
    pub fn new() -> Self {
        Self {
            predictions: Vec::new(),
            references: Vec::new(),
        }
    }

    /// Convert spans to comparable string representations
    ///
    /// Converts entity spans to string format for easy comparison and storage.
    /// Each span becomes a string in the format "start:end:label".
    ///
    /// # Arguments
    ///
    /// * `spans` - Vector of (start, end, label) tuples
    ///
    /// # Returns
    ///
    /// Vector of string representations for each span
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let spans = vec![(0, 3, "PERSON".to_string()), (5, 8, "LOC".to_string())];
    /// let strings = convert_spans_to_strings(&spans);
    /// assert_eq!(strings, vec!["0:3:PERSON", "5:8:LOC"]);
    /// ```
    fn convert_spans_to_strings(&self, spans: &[(usize, usize, String)]) -> Vec<String> {
        spans
            .iter()
            .map(|(start, end, label)| format!("{}:{}:{}", start, end, label))
            .collect()
    }

    /// Parse tag sequence into entity spans
    ///
    /// Converts IOB/IOBES tag sequences into entity spans. This is a simplified
    /// implementation that handles basic tag patterns.
    ///
    /// # Arguments
    ///
    /// * `tags` - Space-separated tag sequence (e.g., "B-PER I-PER O B-LOC")
    ///
    /// # Returns
    ///
    /// Vector of entity spans as (start, end, label) tuples
    ///
    /// # Tag Format
    ///
    /// Supports IOB format:
    /// - `B-LABEL`: Beginning of entity with given label
    /// - `I-LABEL`: Inside/continuation of entity with given label
    /// - `O`: Outside any entity
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let tags = "B-PER I-PER O B-LOC O";
    /// let spans = parse_tags_to_spans(tags);
    /// assert_eq!(spans, vec![(0, 2, "PER".to_string()), (3, 4, "LOC".to_string())]);
    /// ```
    fn parse_tags_to_spans(&self, tags: &str) -> Vec<(usize, usize, String)> {
        let tag_list: Vec<&str> = tags.split_whitespace().collect();
        let mut spans = Vec::new();
        let mut current_start = None;
        let mut current_label: Option<String> = None;

        for (i, tag) in tag_list.iter().enumerate() {
            if tag.starts_with("B-") {
                // End previous entity if exists
                if let (Some(start), Some(label)) = (current_start, &current_label) {
                    spans.push((start, i, label.clone()));
                }
                // Start new entity
                current_start = Some(i);
                current_label = Some(tag[2..].to_string());
            } else if tag.starts_with("I-") {
                // Continue current entity (no action needed)
                // Could add validation that label matches current entity
            } else {
                // "O" or other tag - end current entity
                if let (Some(start), Some(label)) = (current_start, &current_label) {
                    spans.push((start, i, label.clone()));
                }
                current_start = None;
                current_label = None;
            }
        }

        // End final entity if exists
        if let (Some(start), Some(label)) = (current_start, current_label) {
            spans.push((start, tag_list.len(), label));
        }

        spans
    }
}

impl Metric for TokenClassificationMetric {
    /// Add a batch of token classification predictions and references
    ///
    /// Accumulates entity data for later metric computation. Supports both
    /// span-based and text-based input formats.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Model predictions (Spans or Text)
    /// * `references` - Ground truth references (Spans or Text)
    ///
    /// # Input Format Requirements
    ///
    /// - **Spans**: Vector of sequences, each containing (start, end, label) tuples
    /// - **Text**: Vector of tag sequences (space-separated IOB tags)
    /// - Both predictions and references must use the same format
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error if input formats are incompatible.
    ///
    /// # Errors
    ///
    /// - `InvalidInput`: If input types don't match expected formats
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// use trustformers::auto::metrics::{TokenClassificationMetric, MetricInput, Metric};
    ///
    /// let mut metric = TokenClassificationMetric::new();
    ///
    /// // Using spans
    /// let predictions = MetricInput::Spans(vec![
    ///     vec![(0, 2, "PERSON".to_string())],
    /// ]);
    /// let references = MetricInput::Spans(vec![
    ///     vec![(0, 2, "PERSON".to_string())],
    /// ]);
    ///
    /// metric.add_batch(&predictions, &references)?;
    ///
    /// // Using text tags
    /// let predictions = MetricInput::Text(vec!["B-PER I-PER O".to_string()]);
    /// let references = MetricInput::Text(vec!["B-PER I-PER O".to_string()]);
    ///
    /// metric.add_batch(&predictions, &references)?;

    fn add_batch(&mut self, predictions: &MetricInput, references: &MetricInput) -> Result<()> {
        match (predictions, references) {
            (MetricInput::Spans(pred_spans), MetricInput::Spans(ref_spans)) => {
                // Convert spans to string representations for easy comparison
                for pred_seq in pred_spans {
                    self.predictions.push(self.convert_spans_to_strings(pred_seq));
                }
                for ref_seq in ref_spans {
                    self.references.push(self.convert_spans_to_strings(ref_seq));
                }
                Ok(())
            },
            (MetricInput::Text(pred_text), MetricInput::Text(ref_text)) => {
                // Parse tag sequences into spans, then convert to strings
                for pred_tags in pred_text {
                    let spans = self.parse_tags_to_spans(pred_tags);
                    self.predictions.push(self.convert_spans_to_strings(&spans));
                }
                for ref_tags in ref_text {
                    let spans = self.parse_tags_to_spans(ref_tags);
                    self.references.push(self.convert_spans_to_strings(&spans));
                }
                Ok(())
            },
            _ => Err(TrustformersError::invalid_input_simple("Invalid input types for token classification metric: expected Spans or Text for both predictions and references".to_string()
            )),
        }
    }

    /// Compute token classification metrics
    ///
    /// Calculates entity-level precision, recall, and F1 scores based on
    /// exact matching of entity boundaries and labels.
    ///
    /// # Returns
    ///
    /// `MetricResult` containing:
    /// - **Primary value**: F1 score (harmonic mean of precision and recall)
    /// - **Details**:
    ///   - `precision`: Proportion of predicted entities that are correct
    ///   - `recall`: Proportion of reference entities that were predicted
    ///   - `f1`: F1 score (same as primary value)
    ///
    /// # Errors
    ///
    /// - Returns valid results even with no data (all metrics = 0.0)
    /// - No errors are returned from this method
    ///
    /// # Algorithm Details
    ///
    /// 1. **Entity Matching**: Count entities that appear in both predictions and references
    /// 2. **Precision**: `correct_entities / total_predicted_entities`
    /// 3. **Recall**: `correct_entities / total_reference_entities`
    /// 4. **F1**: `2 * precision * recall / (precision + recall)`
    ///
    /// # Entity Matching Criteria
    ///
    /// Entities match if:
    /// - Start position is identical
    /// - End position is identical
    /// - Entity label is identical
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// use trustformers::auto::metrics::{TokenClassificationMetric, MetricInput, Metric};
    ///
    /// let mut metric = TokenClassificationMetric::new();
    /// metric.add_batch(
    ///     &MetricInput::Spans(vec![vec![(0, 2, "PERSON".to_string())]]),
    ///     &MetricInput::Spans(vec![vec![(0, 2, "PERSON".to_string())]])
    /// )?;
    ///
    /// let result = metric.compute()?;
    /// assert_eq!(result.name, "token_classification");
    /// assert_eq!(result.value, 1.0); // Perfect match
    /// assert_eq!(result.details.get("precision"), Some(&1.0));
    /// assert_eq!(result.details.get("recall"), Some(&1.0));
    /// assert_eq!(result.details.get("f1"), Some(&1.0));

    fn compute(&self) -> Result<MetricResult> {
        let mut total_predicted = 0;
        let mut total_reference = 0;
        let mut total_correct = 0;

        // Count entities across all sequences
        let num_sequences = self.predictions.len().min(self.references.len());

        for i in 0..num_sequences {
            let pred_entities = &self.predictions[i];
            let ref_entities = &self.references[i];

            total_predicted += pred_entities.len();
            total_reference += ref_entities.len();

            // Count correct predictions (entities that appear in both)
            for pred_entity in pred_entities {
                if ref_entities.contains(pred_entity) {
                    total_correct += 1;
                }
            }
        }

        // Calculate metrics
        let precision = if total_predicted > 0 {
            total_correct as f64 / total_predicted as f64
        } else {
            0.0
        };

        let recall = if total_reference > 0 {
            total_correct as f64 / total_reference as f64
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        // Build result details
        let mut details = HashMap::new();
        details.insert("precision".to_string(), precision);
        details.insert("recall".to_string(), recall);
        details.insert("f1".to_string(), f1);

        Ok(MetricResult {
            name: "token_classification".to_string(),
            value: f1, // Primary metric is F1 score
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
    ///    /// use trustformers::auto::metrics::{TokenClassificationMetric, MetricInput, Metric};
    ///
    /// let mut metric = TokenClassificationMetric::new();
    /// metric.add_batch(
    ///     &MetricInput::Spans(vec![vec![(0, 2, "PERSON".to_string())]]),
    ///     &MetricInput::Spans(vec![vec![(0, 2, "PERSON".to_string())]])
    /// )?;
    ///
    /// metric.reset();
    /// // Metric is now ready for new entity data

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
    /// String slice "token_classification"
    fn name(&self) -> &str {
        "token_classification"
    }
}

impl Default for TokenClassificationMetric {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_classification_metric_spans() {
        let mut metric = TokenClassificationMetric::new();

        let predictions = MetricInput::Spans(vec![vec![
            (0, 2, "PERSON".to_string()),
            (3, 5, "LOCATION".to_string()),
        ]]);
        let references = MetricInput::Spans(vec![vec![
            (0, 2, "PERSON".to_string()),
            (3, 5, "LOCATION".to_string()),
        ]]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.name, "token_classification");
        assert_eq!(result.value, 1.0); // Perfect match
        assert_eq!(result.details.get("precision"), Some(&1.0));
        assert_eq!(result.details.get("recall"), Some(&1.0));
        assert_eq!(result.details.get("f1"), Some(&1.0));
    }

    #[test]
    fn test_token_classification_metric_partial_match() {
        let mut metric = TokenClassificationMetric::new();

        let predictions = MetricInput::Spans(vec![vec![
            (0, 2, "PERSON".to_string()),
            (3, 5, "LOCATION".to_string()),
        ]]);
        let references = MetricInput::Spans(vec![
            vec![(0, 2, "PERSON".to_string()), (3, 6, "LOCATION".to_string())], // Different end
        ]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        // Only PERSON matches exactly, LOCATION has different boundaries
        assert_eq!(result.details.get("precision"), Some(&0.5)); // 1/2 predicted correct
        assert_eq!(result.details.get("recall"), Some(&0.5)); // 1/2 reference found
        assert_eq!(result.details.get("f1"), Some(&0.5));
    }

    #[test]
    fn test_token_classification_metric_no_match() {
        let mut metric = TokenClassificationMetric::new();

        let predictions = MetricInput::Spans(vec![vec![(0, 2, "PERSON".to_string())]]);
        let references = MetricInput::Spans(vec![vec![(3, 5, "LOCATION".to_string())]]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.details.get("precision"), Some(&0.0));
        assert_eq!(result.details.get("recall"), Some(&0.0));
        assert_eq!(result.details.get("f1"), Some(&0.0));
    }

    #[test]
    fn test_token_classification_metric_text_tags() {
        let mut metric = TokenClassificationMetric::new();

        let predictions = MetricInput::Text(vec!["B-PER I-PER O B-LOC O".to_string()]);
        let references = MetricInput::Text(vec!["B-PER I-PER O B-LOC O".to_string()]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.value, 1.0); // Perfect match
    }

    #[test]
    fn test_token_classification_metric_empty() {
        let mut metric = TokenClassificationMetric::new();

        let predictions = MetricInput::Spans(vec![vec![]]);
        let references = MetricInput::Spans(vec![vec![]]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        // No entities to evaluate
        assert_eq!(result.details.get("precision"), Some(&0.0));
        assert_eq!(result.details.get("recall"), Some(&0.0));
        assert_eq!(result.details.get("f1"), Some(&0.0));
    }

    #[test]
    fn test_token_classification_metric_reset() {
        let mut metric = TokenClassificationMetric::new();

        metric
            .add_batch(
                &MetricInput::Spans(vec![vec![(0, 2, "PERSON".to_string())]]),
                &MetricInput::Spans(vec![vec![(0, 2, "PERSON".to_string())]]),
            )
            .unwrap();

        metric.reset();

        let result = metric.compute().unwrap();
        // Should have no data after reset
        assert_eq!(result.value, 0.0);
    }

    #[test]
    fn test_token_classification_metric_invalid_input() {
        let mut metric = TokenClassificationMetric::new();

        let predictions = MetricInput::Classifications(vec![0, 1]);
        let references = MetricInput::Spans(vec![vec![(0, 2, "PERSON".to_string())]]);

        let result = metric.add_batch(&predictions, &references);
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_spans_to_strings() {
        let metric = TokenClassificationMetric::new();
        let spans = vec![(0, 3, "PERSON".to_string()), (5, 8, "LOC".to_string())];
        let strings = metric.convert_spans_to_strings(&spans);

        assert_eq!(strings, vec!["0:3:PERSON", "5:8:LOC"]);
    }

    #[test]
    fn test_parse_tags_to_spans() {
        let metric = TokenClassificationMetric::new();
        let tags = "B-PER I-PER O B-LOC O B-ORG";
        let spans = metric.parse_tags_to_spans(tags);

        assert_eq!(
            spans,
            vec![
                (0, 2, "PER".to_string()),
                (3, 4, "LOC".to_string()),
                (5, 6, "ORG".to_string()),
            ]
        );
    }

    #[test]
    fn test_parse_tags_empty() {
        let metric = TokenClassificationMetric::new();
        let tags = "O O O";
        let spans = metric.parse_tags_to_spans(tags);

        assert_eq!(spans, vec![]);
    }

    #[test]
    fn test_mixed_input_types() {
        let mut metric = TokenClassificationMetric::new();

        let predictions = MetricInput::Spans(vec![vec![(0, 2, "PERSON".to_string())]]);
        let references = MetricInput::Text(vec!["B-PER I-PER".to_string()]);

        let result = metric.add_batch(&predictions, &references);
        assert!(result.is_err()); // Should reject mixed input types
    }
}
