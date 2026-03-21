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
    pub gradient_max: f64,
    pub gradient_min: f64,
    pub dead_neurons_ratio: f64,
    pub active_neurons_ratio: f64,
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
    pub dead_neuron_threshold: f64,
    pub oscillation_variance_threshold: f64,
    pub no_gradient_steps_threshold: usize,
}

impl Default for GradientDebugConfig {
    fn default() -> Self {
        Self {
            vanishing_threshold: 1e-7,
            exploding_threshold: 10.0,
            dead_neuron_threshold: 0.1, // 10% dead neurons trigger alert
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
