//! # Composite and Default Metrics for TrustformeRS
//!
//! This module provides composite metrics for multi-task scenarios and default
//! fallback metrics for unknown or general-purpose evaluation tasks.
//!
//! ## Overview
//!
//! The module contains two main components:
//! - **DefaultMetric**: A fallback metric that uses classification evaluation
//! - **CompositeMetric**: A container for managing multiple metrics simultaneously
//!
//! ## Features
//!
//! - **Multi-task evaluation**: Evaluate models on multiple tasks simultaneously
//! - **Fallback handling**: Default metrics for unknown task types
//! - **Unified interface**: Consistent API across all metric combinations
//! - **Parallel computation**: Independent metric calculations for efficiency
//!
//! ## Usage Examples
//!
//! ### Using Default Metric
//!
//! ```rust
//! use trustformers::auto::metrics::{DefaultMetric, MetricInput, Metric};
//!
//! let mut metric = DefaultMetric::new();
//!
//! // Default metric handles classification-like evaluation
//! let predictions = MetricInput::Classifications(vec![0, 1, 0, 1]);
//! let references = MetricInput::Classifications(vec![0, 0, 1, 1]);
//! metric.add_batch(&predictions, &references)?;
//!
//! let result = metric.compute()?;
//! println!("Default Score: {:.3}", result.value);
//! ```
//!
//! ### Using Composite Metric
//!
//! ```rust
//! use trustformers::auto::metrics::{CompositeMetric, ClassificationMetric, GenerationMetric};
//!
//! let mut composite = CompositeMetric::new(vec![
//!     Box::new(ClassificationMetric::new()),
//!     Box::new(GenerationMetric::new()),
//! ]);
//!
//! // Add data for both metrics (composite will distribute appropriately)
//! // ... add batch data ...
//!
//! // Compute all metrics at once
//! let results = composite.compute_all()?;
//! for result in results {
//!     println!("{}: {:.3}", result.name, result.value);
//! }
//! ```
//!
//! ### Creating Composite from AutoMetric
//!
//! ```rust
//! use trustformers::auto::metrics::AutoMetric;
//!
//! // Create composite metric for multiple tasks
//! let composite = AutoMetric::composite(&[
//!     "text-classification",
//!     "text-generation",
//!     "question-answering"
//! ])?;
//!
//! let results = composite.compute_all()?;
//! ```
//!
//! ## Implementation Details
//!
//! ### DefaultMetric Architecture
//!
//! The DefaultMetric serves as a fallback when no specific metric is available:
//! - Wraps ClassificationMetric for basic evaluation
//! - Provides consistent interface for unknown tasks
//! - Handles common input types gracefully
//!
//! ### CompositeMetric Architecture
//!
//! The CompositeMetric manages multiple individual metrics:
//! - Independent metric state management
//! - Parallel evaluation capabilities
//! - Unified result aggregation
//! - Dynamic metric addition support

use super::{ClassificationMetric, Metric, MetricInput, MetricResult};
use crate::error::{Result, TrustformersError};

/// Default metric implementation
///
/// Provides a fallback metric that uses classification evaluation for unknown
/// or general-purpose tasks. This metric serves as a safe default when no
/// specific metric is available for a task.
///
/// ## Design Principles
///
/// - **Fallback safety**: Always provides reasonable evaluation results
/// - **Classification-based**: Uses proven classification metrics as foundation
/// - **Compatibility**: Handles common input types gracefully
/// - **Simplicity**: Minimal configuration required
///
/// ## Use Cases
///
/// - Unknown task types in automatic metric selection
/// - General-purpose model evaluation
/// - Baseline comparison metrics
/// - Development and testing scenarios
///
/// ## Supported Input Types
///
/// - `Classifications`: Direct class predictions
/// - `Probabilities`: Probability distributions (converted to classifications)
#[derive(Debug, Clone)]
pub struct DefaultMetric {
    /// Internal classification metric for computation
    classification_metric: ClassificationMetric,
}

impl DefaultMetric {
    /// Create a new default metric instance
    ///
    /// Initializes a default metric with an internal ClassificationMetric
    /// for handling the actual evaluation computation.
    ///
    /// # Returns
    ///
    /// New `DefaultMetric` instance ready for evaluation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::DefaultMetric;
    ///
    /// let metric = DefaultMetric::new();
    /// assert_eq!(metric.name(), "default");
    /// ```
    pub fn new() -> Self {
        Self {
            classification_metric: ClassificationMetric::new(),
        }
    }
}

impl Metric for DefaultMetric {
    /// Add a batch of predictions and references
    ///
    /// Delegates to the internal ClassificationMetric for batch processing.
    /// Handles classification-like evaluation for general-purpose use.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Model predictions (Classifications or Probabilities)
    /// * `references` - Ground truth references (Classifications)
    ///
    /// # Input Format Requirements
    ///
    /// Same as ClassificationMetric:
    /// - **Classifications**: Vector of class indices
    /// - **Probabilities**: Probability distributions (converted to classifications)
    /// - **References**: Must be Classifications
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error if input formats are incompatible.
    ///
    /// # Errors
    ///
    /// - Passes through errors from the underlying ClassificationMetric
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::{DefaultMetric, MetricInput, Metric};
    ///
    /// let mut metric = DefaultMetric::new();
    ///
    /// let predictions = MetricInput::Classifications(vec![0, 1, 0]);
    /// let references = MetricInput::Classifications(vec![0, 0, 1]);
    ///
    /// metric.add_batch(&predictions, &references)?;
    /// ```
    fn add_batch(&mut self, predictions: &MetricInput, references: &MetricInput) -> Result<()> {
        self.classification_metric.add_batch(predictions, references)
    }

    /// Compute default metrics
    ///
    /// Delegates computation to the internal ClassificationMetric and returns
    /// results with default-specific naming.
    ///
    /// # Returns
    ///
    /// `MetricResult` containing:
    /// - **Primary value**: Classification accuracy
    /// - **Details**: Same as ClassificationMetric (accuracy, precision, recall, F1)
    /// - **Name**: "default" (distinguishes from direct classification)
    ///
    /// # Errors
    ///
    /// - Passes through errors from the underlying ClassificationMetric
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::{DefaultMetric, MetricInput, Metric};
    ///
    /// let mut metric = DefaultMetric::new();
    /// metric.add_batch(
    ///     &MetricInput::Classifications(vec![0, 1]),
    ///     &MetricInput::Classifications(vec![0, 1])
    /// )?;
    ///
    /// let result = metric.compute()?;
    /// assert_eq!(result.name, "default");
    /// assert_eq!(result.value, 1.0); // Perfect accuracy
    /// ```
    fn compute(&self) -> Result<MetricResult> {
        let mut result = self.classification_metric.compute()?;
        // Override the name to indicate this is a default metric
        result.name = "default".to_string();
        Ok(result)
    }

    /// Reset the metric state
    ///
    /// Delegates to the internal ClassificationMetric to clear all accumulated
    /// data, preparing for a new evaluation run.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::{DefaultMetric, MetricInput, Metric};
    ///
    /// let mut metric = DefaultMetric::new();
    /// metric.add_batch(
    ///     &MetricInput::Classifications(vec![0, 1]),
    ///     &MetricInput::Classifications(vec![0, 1])
    /// )?;
    ///
    /// metric.reset();
    /// // Metric is now ready for new data
    /// ```
    fn reset(&mut self) {
        self.classification_metric.reset();
    }

    /// Get the metric name
    ///
    /// Returns the identifier for this metric type, distinguishing it
    /// from direct classification metrics.
    ///
    /// # Returns
    ///
    /// String slice "default"
    fn name(&self) -> &str {
        "default"
    }
}

impl Default for DefaultMetric {
    fn default() -> Self {
        Self::new()
    }
}

/// Composite metric for multiple tasks
///
/// Manages multiple individual metrics, allowing for evaluation across
/// multiple tasks or criteria simultaneously. This is particularly useful
/// in multi-task learning scenarios or comprehensive benchmarking.
///
/// ## Design Principles
///
/// - **Independent metrics**: Each metric maintains its own state
/// - **Unified interface**: Consistent API for metric management
/// - **Flexible composition**: Dynamic addition and removal of metrics
/// - **Parallel evaluation**: Independent metric computations
///
/// ## Use Cases
///
/// - Multi-task model evaluation
/// - Comprehensive benchmarking suites
/// - A/B testing with multiple criteria
/// - Research experiments requiring diverse metrics
///
/// ## Thread Safety
///
/// The CompositeMetric is designed to be thread-safe for read operations,
/// but write operations (adding batches, resetting) should be synchronized
/// externally if used across multiple threads.
pub struct CompositeMetric {
    /// Vector of individual metric implementations
    metrics: Vec<Box<dyn Metric>>,
}

impl std::fmt::Debug for CompositeMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeMetric")
            .field("metrics_count", &self.metrics.len())
            .finish()
    }
}

impl CompositeMetric {
    /// Create a new composite metric
    ///
    /// Initializes a composite metric with the provided individual metrics.
    ///
    /// # Arguments
    ///
    /// * `metrics` - Vector of individual metric implementations
    ///
    /// # Returns
    ///
    /// New `CompositeMetric` managing the provided metrics.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::{CompositeMetric, ClassificationMetric, GenerationMetric};
    ///
    /// let composite = CompositeMetric::new(vec![
    ///     Box::new(ClassificationMetric::new()),
    ///     Box::new(GenerationMetric::new()),
    /// ]);
    ///
    /// assert_eq!(composite.metrics.len(), 2);
    /// ```
    pub fn new(metrics: Vec<Box<dyn Metric>>) -> Self {
        Self { metrics }
    }

    /// Get a reference to the metrics vector
    pub fn metrics(&self) -> &Vec<Box<dyn Metric>> {
        &self.metrics
    }

    /// Add a metric to the composite
    ///
    /// Dynamically adds a new metric to the composite after creation.
    /// This allows for flexible composition of evaluation metrics.
    ///
    /// # Arguments
    ///
    /// * `metric` - Metric implementation to add to the composite
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::{CompositeMetric, ClassificationMetric};
    ///
    /// let mut composite = CompositeMetric::new(vec![]);
    /// composite.add_metric(Box::new(ClassificationMetric::new()));
    ///
    /// assert_eq!(composite.metrics.len(), 1);
    /// ```
    pub fn add_metric(&mut self, metric: Box<dyn Metric>) {
        self.metrics.push(metric);
    }

    /// Get the number of metrics in the composite
    ///
    /// Returns the count of individual metrics managed by this composite.
    ///
    /// # Returns
    ///
    /// Number of metrics in the composite
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::{CompositeMetric, ClassificationMetric};
    ///
    /// let composite = CompositeMetric::new(vec![
    ///     Box::new(ClassificationMetric::new()),
    /// ]);
    ///
    /// assert_eq!(composite.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.metrics.len()
    }

    /// Check if the composite is empty
    ///
    /// Returns true if no metrics are managed by this composite.
    ///
    /// # Returns
    ///
    /// `true` if the composite contains no metrics
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::CompositeMetric;
    ///
    /// let composite = CompositeMetric::new(vec![]);
    /// assert!(composite.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }

    /// Compute all metrics and return results
    ///
    /// Computes all individual metrics and returns their results as a vector.
    /// The order corresponds to the order metrics were added to the composite.
    ///
    /// # Returns
    ///
    /// Vector of `MetricResult` objects, one for each individual metric.
    ///
    /// # Errors
    ///
    /// Returns an error if any individual metric computation fails. The error
    /// will be from the first metric that fails; subsequent metrics are not
    /// evaluated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::{CompositeMetric, ClassificationMetric, MetricInput};
    ///
    /// let mut composite = CompositeMetric::new(vec![
    ///     Box::new(ClassificationMetric::new()),
    /// ]);
    ///
    /// // Add data to metrics (would need to add batch data first)
    /// // ...
    ///
    /// let results = composite.compute_all()?;
    /// assert_eq!(results.len(), 1);
    /// ```
    pub fn compute_all(&self) -> Result<Vec<MetricResult>> {
        self.metrics.iter().map(|m| m.compute()).collect()
    }

    /// Reset all metrics
    ///
    /// Resets the state of all individual metrics, preparing them for a new
    /// evaluation run. This is more efficient than recreating the composite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::{CompositeMetric, ClassificationMetric};
    ///
    /// let mut composite = CompositeMetric::new(vec![
    ///     Box::new(ClassificationMetric::new()),
    /// ]);
    ///
    /// // Add some data...
    /// // ...
    ///
    /// composite.reset_all();
    /// // All metrics are now reset and ready for new data
    /// ```
    pub fn reset_all(&mut self) {
        for metric in &mut self.metrics {
            metric.reset();
        }
    }

    /// Get metric names
    ///
    /// Returns a vector of names for all metrics in the composite.
    /// Useful for logging and result identification.
    ///
    /// # Returns
    ///
    /// Vector of metric names in the same order as the metrics
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::{CompositeMetric, ClassificationMetric, GenerationMetric};
    ///
    /// let composite = CompositeMetric::new(vec![
    ///     Box::new(ClassificationMetric::new()),
    ///     Box::new(GenerationMetric::new()),
    /// ]);
    ///
    /// let names = composite.get_metric_names();
    /// assert_eq!(names, vec!["classification", "generation"]);
    /// ```
    pub fn get_metric_names(&self) -> Vec<&str> {
        self.metrics.iter().map(|m| m.name()).collect()
    }

    /// Add batch data to all compatible metrics
    ///
    /// Attempts to add the batch data to all metrics in the composite.
    /// Metrics that cannot handle the input types will be skipped silently.
    /// This allows for flexible multi-task evaluation where different metrics
    /// may require different input formats.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Model predictions in any supported format
    /// * `references` - Ground truth references in any supported format
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if at least one metric successfully processed the data,
    /// or an error if no metrics could handle the input.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use trustformers::auto::metrics::{CompositeMetric, ClassificationMetric, MetricInput};
    ///
    /// let mut composite = CompositeMetric::new(vec![
    ///     Box::new(ClassificationMetric::new()),
    /// ]);
    ///
    /// let predictions = MetricInput::Classifications(vec![0, 1]);
    /// let references = MetricInput::Classifications(vec![0, 1]);
    ///
    /// composite.add_batch_to_compatible(&predictions, &references)?;
    /// ```
    pub fn add_batch_to_compatible(
        &mut self,
        predictions: &MetricInput,
        references: &MetricInput,
    ) -> Result<()> {
        let mut successful = false;

        for metric in &mut self.metrics {
            if metric.add_batch(predictions, references).is_ok() {
                successful = true;
            }
        }

        if successful {
            Ok(())
        } else {
            Err(TrustformersError::invalid_input_simple(
                "No metrics in the composite could handle the provided input types".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto::metrics::{GenerationMetric, MetricInput};

    #[test]
    fn test_default_metric_basic() {
        let mut metric = DefaultMetric::new();

        let predictions = MetricInput::Classifications(vec![0, 1, 0, 1]);
        let references = MetricInput::Classifications(vec![0, 0, 1, 1]);

        metric.add_batch(&predictions, &references).unwrap();

        let result = metric.compute().unwrap();
        assert_eq!(result.name, "default");
        assert_eq!(result.value, 0.5); // 50% accuracy
        assert!(result.details.contains_key("accuracy"));
    }

    #[test]
    fn test_default_metric_reset() {
        let mut metric = DefaultMetric::new();

        metric
            .add_batch(
                &MetricInput::Classifications(vec![0, 1]),
                &MetricInput::Classifications(vec![0, 1]),
            )
            .unwrap();

        metric.reset();

        // Should fail because no data after reset
        assert!(metric.compute().is_err());
    }

    #[test]
    fn test_composite_metric_creation() {
        let composite = CompositeMetric::new(vec![
            Box::new(ClassificationMetric::new()),
            Box::new(GenerationMetric::new()),
        ]);

        assert_eq!(composite.len(), 2);
        assert!(!composite.is_empty());

        let names = composite.get_metric_names();
        assert_eq!(names, vec!["classification", "generation"]);
    }

    #[test]
    fn test_composite_metric_add_metric() {
        let mut composite = CompositeMetric::new(vec![]);
        assert!(composite.is_empty());

        composite.add_metric(Box::new(ClassificationMetric::new()));
        assert_eq!(composite.len(), 1);
        assert!(!composite.is_empty());
    }

    #[test]
    fn test_composite_metric_reset_all() {
        let mut composite = CompositeMetric::new(vec![Box::new(ClassificationMetric::new())]);

        // Add some data
        composite
            .add_batch_to_compatible(
                &MetricInput::Classifications(vec![0, 1]),
                &MetricInput::Classifications(vec![0, 1]),
            )
            .unwrap();

        composite.reset_all();

        // Should have no results after reset (would fail or return empty)
        let results = composite.compute_all();
        // The classification metric should fail because no data
        assert!(results.is_err());
    }

    #[test]
    fn test_composite_metric_compute_all() {
        let mut composite = CompositeMetric::new(vec![Box::new(ClassificationMetric::new())]);

        composite
            .add_batch_to_compatible(
                &MetricInput::Classifications(vec![0, 1]),
                &MetricInput::Classifications(vec![0, 1]),
            )
            .unwrap();

        let results = composite.compute_all().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "classification");
    }

    #[test]
    fn test_composite_metric_incompatible_input() {
        let mut composite = CompositeMetric::new(vec![Box::new(ClassificationMetric::new())]);

        // Try to add incompatible input
        let result = composite.add_batch_to_compatible(
            &MetricInput::Text(vec!["hello".to_string()]),
            &MetricInput::Text(vec!["world".to_string()]),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_composite_metric_mixed_compatibility() {
        let mut composite = CompositeMetric::new(vec![
            Box::new(ClassificationMetric::new()),
            Box::new(GenerationMetric::new()),
        ]);

        // Text input should work for generation metric but not classification
        let result = composite.add_batch_to_compatible(
            &MetricInput::Text(vec!["hello".to_string()]),
            &MetricInput::Text(vec!["world".to_string()]),
        );

        assert!(result.is_ok()); // Should succeed because generation metric can handle it

        // compute_all() will fail because ClassificationMetric has no data
        // This is expected behavior - all metrics must have data to compute
        let results = composite.compute_all();
        assert!(results.is_err()); // Should fail because classification metric has no data
    }

    #[test]
    fn test_default_metric_name() {
        let metric = DefaultMetric::new();
        assert_eq!(metric.name(), "default");
    }

    #[test]
    fn test_composite_metric_empty() {
        let composite = CompositeMetric::new(vec![]);
        assert!(composite.is_empty());
        assert_eq!(composite.len(), 0);

        let results = composite.compute_all().unwrap();
        assert_eq!(results.len(), 0);
    }
}
