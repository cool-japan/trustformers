//! Core Types and Configuration for Gradient Debugging
//!
//! This module provides the fundamental types, enums, and configuration structures
//! used throughout the gradient debugging system for analyzing gradient flow,
//! detecting anomalies, and monitoring model training dynamics.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Layer health status for gradient debugging
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerHealth {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

/// Gradient flow information for a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFlow {
    pub layer_name: String,
    pub step: usize,
    pub gradient_norm: f64,
    pub gradient_mean: f64,
    pub gradient_std: f64,
    /// Largest gradient element, when the real per-element tensor was
    /// available (see
    /// [`crate::gradient_debugger::debugger::GradientDebugger::record_gradient_values`]).
    ///
    /// `None` for the reduced entry point `record_gradient_flow`, which only
    /// receives norm/mean/std. It used to be filled with `mean + std`, which is
    /// not a maximum of anything -- for Gaussian-ish gradients the true max is
    /// several sigma out, and for any skewed distribution `mean + std` can even
    /// fall below the actual maximum's own sign.
    pub gradient_max: Option<f64>,
    /// Smallest gradient element; `None` for the reduced entry point, for the
    /// same reason as [`Self::gradient_max`] (it used to be `mean - std`).
    pub gradient_min: Option<f64>,
    /// Fraction of gradient elements whose magnitude is at or below
    /// [`GradientDebugConfig::dead_gradient_magnitude`].
    ///
    /// `None` for the reduced entry point. It used to come from a three-step
    /// constant ladder over the gradient NORM (`0.9` / `0.3` / `0.05`), which
    /// asserted that 90% of a layer's neurons were dead purely because the
    /// aggregate norm was small -- and that fabricated ratio drove real
    /// [`GradientAlert::DeadNeurons`] alerts.
    pub dead_neurons_ratio: Option<f64>,
    /// `1 - dead_neurons_ratio`, and `None` whenever that is `None`.
    pub active_neurons_ratio: Option<f64>,
    pub timestamp: DateTime<Utc>,
}

/// Historical gradient statistics for tracking trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientHistory {
    pub layer_name: String,
    pub gradient_norms: VecDeque<f64>,
    pub gradient_means: VecDeque<f64>,
    pub gradient_stds: VecDeque<f64>,
    pub step_numbers: VecDeque<usize>,
    pub max_history_length: usize,
    /// Real element count of this layer's gradient tensor, when a caller
    /// has reported one via
    /// [`crate::gradient_debugger::debugger::GradientDebugger::set_layer_parameter_count`].
    /// `record_gradient_flow` only ever receives reduced scalar statistics
    /// (norm/mean/std) -- never the tensor itself -- so this stays `None`
    /// (an honest absence, not a placeholder) unless a caller with access
    /// to the real tensor shape opts in. `#[serde(default)]` keeps this
    /// backward-compatible with snapshots serialized before this field
    /// existed.
    #[serde(default)]
    pub parameter_count: Option<usize>,
}

impl GradientHistory {
    pub fn new(layer_name: String, max_length: usize) -> Self {
        Self {
            layer_name,
            gradient_norms: VecDeque::with_capacity(max_length),
            gradient_means: VecDeque::with_capacity(max_length),
            gradient_stds: VecDeque::with_capacity(max_length),
            step_numbers: VecDeque::with_capacity(max_length),
            max_history_length: max_length,
            parameter_count: None,
        }
    }

    pub fn add_gradient_flow(&mut self, flow: &GradientFlow) {
        if self.gradient_norms.len() >= self.max_history_length {
            self.gradient_norms.pop_front();
            self.gradient_means.pop_front();
            self.gradient_stds.pop_front();
            self.step_numbers.pop_front();
        }

        self.gradient_norms.push_back(flow.gradient_norm);
        self.gradient_means.push_back(flow.gradient_mean);
        self.gradient_stds.push_back(flow.gradient_std);
        self.step_numbers.push_back(flow.step);
    }

    pub fn get_trend_slope(&self) -> Option<f64> {
        if self.gradient_norms.len() < 3 {
            return None;
        }

        // Simple linear regression for gradient norm trend
        let n = self.gradient_norms.len() as f64;
        let sum_x: f64 = (0..self.gradient_norms.len()).map(|i| i as f64).sum();
        let sum_y: f64 = self.gradient_norms.iter().sum();
        let sum_xy: f64 = self.gradient_norms.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..self.gradient_norms.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
        Some(slope)
    }
}

/// Gradient debugging alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GradientAlert {
    VanishingGradients {
        layer_name: String,
        norm: f64,
        threshold: f64,
    },
    ExplodingGradients {
        layer_name: String,
        norm: f64,
        threshold: f64,
    },
    DeadNeurons {
        layer_name: String,
        ratio: f64,
        threshold: f64,
    },
    GradientOscillation {
        layer_name: String,
        variance: f64,
    },
    NoGradientFlow {
        layer_name: String,
        steps_without_gradient: usize,
    },
}

/// Configuration for gradient debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientDebugConfig {
    pub vanishing_threshold: f64,
    pub exploding_threshold: f64,
    /// Fraction of dead elements in a layer that raises
    /// [`GradientAlert::DeadNeurons`].
    pub dead_neuron_threshold: f64,
    /// Magnitude at or below which a single gradient element counts as dead.
    ///
    /// Only used by
    /// [`crate::gradient_debugger::debugger::GradientDebugger::record_gradient_values`],
    /// which is the only entry point that sees per-element gradients.
    #[serde(default = "default_dead_gradient_magnitude")]
    pub dead_gradient_magnitude: f64,
    pub oscillation_variance_threshold: f64,
    pub no_gradient_steps_threshold: usize,
}

/// Default for [`GradientDebugConfig::dead_gradient_magnitude`].
fn default_dead_gradient_magnitude() -> f64 {
    1e-8
}

impl Default for GradientDebugConfig {
    fn default() -> Self {
        Self {
            vanishing_threshold: 1e-7,
            exploding_threshold: 10.0,
            dead_neuron_threshold: 0.1, // 10% dead neurons trigger alert
            dead_gradient_magnitude: default_dead_gradient_magnitude(),
            oscillation_variance_threshold: 100.0,
            no_gradient_steps_threshold: 10,
        }
    }
}

/// Gradient statistics for detailed analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientStatistics {
    pub mean: f64,
    pub std: f64,
    pub median: f64,
    pub percentile_95: f64,
    pub percentile_5: f64,
    pub samples: usize,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Flow characteristics for gradient analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowCharacteristics {
    pub consistency_score: f64,
    pub smoothness_index: f64,
    pub trend_strength: f64,
    pub oscillation_frequency: f64,
    pub stability_measure: f64,
}

/// Layer health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerHealthMetrics {
    pub overall_health: LayerHealth,
    pub gradient_stability: f64,
    pub information_flow_rate: f64,
    pub neuron_activity_ratio: f64,
    pub convergence_indicator: f64,
    pub risk_factors: Vec<String>,
}

/// Comparative analysis between layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeAnalysis {
    pub relative_performance: f64,
    pub rank_among_layers: usize,
    pub similar_layers: Vec<String>,
    pub performance_gap: f64,
    pub optimization_potential: f64,
}
