//! Core data structures and type definitions for model diagnostics.
//!
//! This module contains all the fundamental types, enums, and data structures
//! used throughout the model diagnostics system, providing a centralized
//! location for type definitions and ensuring consistency across components.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model performance metrics collected during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceMetrics {
    /// Current training step
    pub training_step: usize,
    /// Current loss value
    pub loss: f64,
    /// Optional accuracy metric
    pub accuracy: Option<f64>,
    /// Current learning rate
    pub learning_rate: f64,
    /// Batch size being used
    pub batch_size: usize,
    /// Training throughput in samples per second
    pub throughput_samples_per_sec: f64,
    /// Memory usage in megabytes
    pub memory_usage_mb: f64,
    /// GPU utilization percentage (if available)
    pub gpu_utilization: Option<f64>,
    /// Timestamp of measurement
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Model architecture information and analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitectureInfo {
    /// Total number of parameters
    pub total_parameters: usize,
    /// Number of trainable parameters
    pub trainable_parameters: usize,
    /// Model size in megabytes
    pub model_size_mb: f64,
    /// Total number of layers
    pub layer_count: usize,
    /// Count of each layer type
    pub layer_types: HashMap<String, usize>,
    /// Model depth (longest path)
    pub depth: usize,
    /// Model width (maximum layer size)
    pub width: usize,
    /// Count of each activation function type
    pub activation_functions: HashMap<String, usize>,
}

/// Layer activation statistics for monitoring layer behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerActivationStats {
    /// Name of the layer
    pub layer_name: String,
    /// Mean activation value
    pub mean_activation: f64,
    /// Standard deviation of activations
    pub std_activation: f64,
    /// Minimum activation value
    pub min_activation: f64,
    /// Maximum activation value
    pub max_activation: f64,
    /// Ratio of dead neurons (zero activations)
    pub dead_neurons_ratio: f64,
    /// Ratio of saturated neurons (max activations)
    pub saturated_neurons_ratio: f64,
    /// Sparsity of activations
    pub sparsity: f64,
    /// Output shape of the layer
    pub output_shape: Vec<usize>,
}

/// Training dynamics analysis including convergence and stability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDynamics {
    /// Current convergence status
    pub convergence_status: ConvergenceStatus,
    /// Training stability assessment
    pub training_stability: TrainingStability,
    /// Learning efficiency score
    pub learning_efficiency: f64,
    /// Detected overfitting indicators
    pub overfitting_indicators: Vec<OverfittingIndicator>,
    /// Detected underfitting indicators
    pub underfitting_indicators: Vec<UnderfittingIndicator>,
    /// Plateau detection information
    pub plateau_detection: Option<PlateauInfo>,
}

/// Convergence status enumeration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvergenceStatus {
    /// Training is converging normally
    Converging,
    /// Training has converged
    Converged,
    /// Training is diverging
    Diverging,
    /// Training is oscillating
    Oscillating,
    /// Training has reached a plateau
    Plateau,
    /// Status is unknown
    Unknown,
}

/// Training stability assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingStability {
    /// Training is stable
    Stable,
    /// Training is unstable
    Unstable,
    /// Training has high variance
    HighVariance,
    /// Stability is unknown
    Unknown,
}

/// Overfitting indicators that can be detected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverfittingIndicator {
    /// Gap between training and validation performance
    TrainValidationGap { gap: f64 },
    /// Validation loss is increasing
    ValidationLossIncreasing { duration_steps: usize },
    /// High variance in validation metrics
    HighVarianceInValidation,
    /// Perfect training accuracy achieved
    PerfectTrainingAccuracy,
}

/// Underfitting indicators that can be detected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnderfittingIndicator {
    /// Training loss is too high
    HighTrainingLoss { loss: f64, threshold: f64 },
    /// Convergence is too slow
    SlowConvergence { steps_taken: usize, expected: usize },
    /// Training accuracy is too low
    LowTrainingAccuracy { accuracy: f64, threshold: f64 },
    /// No learning progress detected
    NoLearning { steps_without_improvement: usize },
}

/// Plateau detection information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlateauInfo {
    /// Step where plateau started
    pub start_step: usize,
    /// Duration of plateau in steps
    pub duration_steps: usize,
    /// Value at which plateau occurred
    pub plateau_value: f64,
    /// Variance during plateau
    pub variance: f64,
}

/// Model diagnostic alerts for various issues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelDiagnosticAlert {
    /// Performance degradation detected
    PerformanceDegradation {
        metric: String,
        current: f64,
        previous_avg: f64,
        degradation_percent: f64,
    },
    /// Memory leak detected
    MemoryLeak {
        current_usage_mb: f64,
        growth_rate_mb_per_step: f64,
    },
    /// Training instability detected
    TrainingInstability { variance: f64, threshold: f64 },
    /// Convergence issue detected
    ConvergenceIssue {
        issue_type: ConvergenceStatus,
        duration_steps: usize,
    },
    /// Architectural concern identified
    ArchitecturalConcern {
        concern: String,
        recommendation: String,
    },
}

/// Performance summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Total training steps completed
    pub total_steps: usize,
    /// Current loss value
    pub current_loss: f64,
    /// Best loss achieved
    pub best_loss: f64,
    /// Average loss over all steps
    pub avg_loss: f64,
    /// Current throughput
    pub current_throughput: f64,
    /// Average throughput
    pub avg_throughput: f64,
    /// Peak memory usage
    pub peak_memory_mb: f64,
    /// Average memory usage
    pub avg_memory_mb: f64,
}

impl Default for PerformanceSummary {
    fn default() -> Self {
        Self {
            total_steps: 0,
            current_loss: 0.0,
            best_loss: f64::INFINITY,
            avg_loss: 0.0,
            current_throughput: 0.0,
            avg_throughput: 0.0,
            peak_memory_mb: 0.0,
            avg_memory_mb: 0.0,
        }
    }
}

/// Architectural analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturalAnalysis {
    /// Parameter efficiency score
    pub parameter_efficiency: f64,
    /// Computational complexity assessment
    pub computational_complexity: String,
    /// Memory efficiency score
    pub memory_efficiency: f64,
    /// Architecture recommendations
    pub recommendations: Vec<String>,
    /// Identified bottlenecks
    pub bottlenecks: Vec<String>,
}

/// Layer analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAnalysis {
    /// Layer name
    pub layer_name: String,
    /// Layer type
    pub layer_type: String,
    /// Health score (0.0 to 1.0)
    pub health_score: f64,
    /// Identified issues
    pub issues: Vec<String>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    /// Activation statistics summary
    pub activation_summary: String,
}

/// Weight distribution analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightDistribution {
    /// Mean weight value
    pub mean: f64,
    /// Standard deviation of weights
    pub std_dev: f64,
    /// Minimum weight value
    pub min: f64,
    /// Maximum weight value
    pub max: f64,
    /// Weight sparsity ratio
    pub sparsity: f64,
    /// Distribution shape assessment
    pub distribution_shape: String,
}

/// Activation heatmap data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationHeatmap {
    /// Heatmap data as 2D array
    pub data: Vec<Vec<f64>>,
    /// Data dimensions
    pub dimensions: (usize, usize),
    /// Value range
    pub value_range: (f64, f64),
    /// Interpretation notes
    pub interpretation: String,
}

/// Attention visualization data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionVisualization {
    /// Attention weights matrix
    pub attention_weights: Vec<Vec<f64>>,
    /// Input token information
    pub input_tokens: Vec<String>,
    /// Output token information
    pub output_tokens: Vec<String>,
    /// Attention patterns identified
    pub patterns: Vec<String>,
}

/// Hidden state analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenStateAnalysis {
    /// Dimensionality information
    pub dimensionality: usize,
    /// Information content score
    pub information_content: f64,
    /// Clustering results
    pub clustering_results: ClusteringResults,
    /// Temporal dynamics
    pub temporal_dynamics: TemporalDynamics,
    /// Representation stability
    pub representation_stability: RepresentationStability,
}

/// Clustering analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResults {
    /// Number of clusters found
    pub num_clusters: usize,
    /// Cluster centers
    pub cluster_centers: Vec<Vec<f64>>,
    /// Cluster assignments for data points
    pub cluster_assignments: Vec<usize>,
    /// Silhouette score
    pub silhouette_score: f64,
    /// Cluster inertia
    pub inertia: f64,
}

/// Temporal dynamics analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDynamics {
    /// Temporal consistency score
    pub temporal_consistency: f64,
    /// Rate of change
    pub change_rate: f64,
    /// Stability time windows
    pub stability_windows: Vec<(usize, usize)>,
    /// Drift detection results
    pub drift_detection: DriftInfo,
}

/// Distribution drift information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftInfo {
    /// Whether drift was detected
    pub drift_detected: bool,
    /// Magnitude of drift
    pub drift_magnitude: f64,
    /// Direction of drift
    pub drift_direction: String,
    /// Step when drift started
    pub onset_step: Option<usize>,
}

/// Representation stability analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepresentationStability {
    /// Overall stability score
    pub stability_score: f64,
    /// Variance across different batches
    pub variance_across_batches: f64,
    /// Consistency measure
    pub consistency_measure: f64,
    /// Robustness to input noise
    pub robustness_to_noise: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_summary_default() {
        let summary = PerformanceSummary::default();
        assert_eq!(summary.total_steps, 0);
        assert_eq!(summary.best_loss, f64::INFINITY);
    }

    #[test]
    fn test_convergence_status_serialization() {
        let status = ConvergenceStatus::Converging;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: ConvergenceStatus = serde_json::from_str(&serialized).unwrap();

        matches!(deserialized, ConvergenceStatus::Converging);
    }

    #[test]
    fn test_layer_activation_stats() {
        let stats = LayerActivationStats {
            layer_name: "test_layer".to_string(),
            mean_activation: 0.5,
            std_activation: 0.2,
            min_activation: 0.0,
            max_activation: 1.0,
            dead_neurons_ratio: 0.1,
            saturated_neurons_ratio: 0.05,
            sparsity: 0.3,
            output_shape: vec![128, 256],
        };

        assert_eq!(stats.layer_name, "test_layer");
        assert_eq!(stats.output_shape, vec![128, 256]);
    }
}
