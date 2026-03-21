//! Real-time Gradient Monitoring and Adaptive Thresholds
//!
//! This module provides real-time gradient monitoring capabilities with adaptive
//! thresholds that dynamically adjust based on gradient behavior patterns.

use super::types::*;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Adaptive thresholds for dynamic gradient monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveThresholds {
    pub layer_name: String,
    pub vanishing_threshold: f64,
    pub exploding_threshold: f64,
    pub adaptation_rate: f64,
    pub recent_gradients: VecDeque<f64>,
    pub last_updated: DateTime<Utc>,
}

impl AdaptiveThresholds {
    pub fn new(layer_name: String, initial_vanishing: f64, initial_exploding: f64) -> Self {
        Self {
            layer_name,
            vanishing_threshold: initial_vanishing,
            exploding_threshold: initial_exploding,
            adaptation_rate: 0.1,
            recent_gradients: VecDeque::with_capacity(100),
            last_updated: Utc::now(),
        }
    }

    pub fn update_thresholds(&mut self, gradient_norm: f64) {
        // Add new gradient to history
        if self.recent_gradients.len() >= 100 {
            self.recent_gradients.pop_front();
        }
        self.recent_gradients.push_back(gradient_norm);

        // Update thresholds based on recent history
        if self.recent_gradients.len() >= 10 {
            let mean =
                self.recent_gradients.iter().sum::<f64>() / self.recent_gradients.len() as f64;
            let variance = self.recent_gradients.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / self.recent_gradients.len() as f64;
            let std_dev = variance.sqrt();

            // Adaptive threshold updates
            let new_vanishing = (mean - 2.0 * std_dev).max(1e-8);
            let new_exploding = mean + 3.0 * std_dev;

            self.vanishing_threshold = self.vanishing_threshold * (1.0 - self.adaptation_rate)
                + new_vanishing * self.adaptation_rate;
            self.exploding_threshold = self.exploding_threshold * (1.0 - self.adaptation_rate)
                + new_exploding * self.adaptation_rate;

            self.last_updated = Utc::now();
        }
    }

    pub fn check_thresholds(&self, gradient_norm: f64) -> Vec<GradientAlert> {
        let mut alerts = Vec::new();

        if gradient_norm < self.vanishing_threshold {
            alerts.push(GradientAlert::VanishingGradients {
                layer_name: self.layer_name.clone(),
                norm: gradient_norm,
                threshold: self.vanishing_threshold,
            });
        }

        if gradient_norm > self.exploding_threshold {
            alerts.push(GradientAlert::ExplodingGradients {
                layer_name: self.layer_name.clone(),
                norm: gradient_norm,
                threshold: self.exploding_threshold,
            });
        }

        alerts
    }

    /// Create adaptive thresholds from gradient history
    pub fn from_history(history: &GradientHistory) -> Self {
        let layer_name = history.layer_name.clone();

        if history.gradient_norms.is_empty() {
            return Self::new(layer_name, 1e-6, 10.0);
        }

        // Calculate initial thresholds from history
        let norms: Vec<f64> = history.gradient_norms.iter().cloned().collect();
        let mean = norms.iter().sum::<f64>() / norms.len() as f64;
        let variance = norms.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / norms.len() as f64;
        let std_dev = variance.sqrt();

        let initial_vanishing = (mean - 2.0 * std_dev).max(1e-8);
        let initial_exploding = mean + 3.0 * std_dev;

        let mut thresholds = Self::new(layer_name, initial_vanishing, initial_exploding);

        // Pre-populate with recent gradients
        for &norm in norms.iter().rev().take(50) {
            if thresholds.recent_gradients.len() >= 100 {
                thresholds.recent_gradients.pop_front();
            }
            thresholds.recent_gradients.push_back(norm);
        }

        thresholds
    }
}

/// Real-time gradient monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeGradientMonitor {
    pub layer_name: String,
    pub current_gradient_norm: f64,
    pub gradient_velocity: f64,
    pub gradient_acceleration: f64,
    pub stability_window: VecDeque<f64>,
    pub anomaly_score: f64,
}

impl RealTimeGradientMonitor {
    pub fn new(layer_name: String) -> Self {
        Self {
            layer_name,
            current_gradient_norm: 0.0,
            gradient_velocity: 0.0,
            gradient_acceleration: 0.0,
            stability_window: VecDeque::with_capacity(10),
            anomaly_score: 0.0,
        }
    }

    pub fn update(&mut self, new_gradient_norm: f64) {
        let previous_norm = self.current_gradient_norm;
        let previous_velocity = self.gradient_velocity;

        self.current_gradient_norm = new_gradient_norm;
        self.gradient_velocity = new_gradient_norm - previous_norm;
        self.gradient_acceleration = self.gradient_velocity - previous_velocity;

        // Update stability window
        if self.stability_window.len() >= 10 {
            self.stability_window.pop_front();
        }
        self.stability_window.push_back(new_gradient_norm);

        // Update anomaly score
        self.anomaly_score = self.compute_anomaly_score();
    }

    fn compute_anomaly_score(&self) -> f64 {
        if self.stability_window.len() < 5 {
            return 0.0;
        }

        let mean = self.stability_window.iter().sum::<f64>() / self.stability_window.len() as f64;
        let variance = self.stability_window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / self.stability_window.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        // Z-score based anomaly detection
        let z_score = (self.current_gradient_norm - mean) / std_dev;
        z_score.abs().min(5.0) / 5.0 // Normalize to 0-1 range
    }

    pub fn get_stability_score(&self) -> f64 {
        if self.stability_window.len() < 3 {
            return 1.0;
        }

        let variance = self
            .stability_window
            .iter()
            .map(|&x| (x - self.current_gradient_norm).powi(2))
            .sum::<f64>()
            / self.stability_window.len() as f64;

        // Higher variance = lower stability
        1.0 / (1.0 + variance)
    }

    pub fn is_stable(&self, threshold: f64) -> bool {
        self.get_stability_score() > threshold
    }

    pub fn is_oscillating(&self) -> bool {
        if self.stability_window.len() < 6 {
            return false;
        }

        // Check for oscillating pattern by looking at sign changes
        let mut sign_changes = 0;
        let values: Vec<f64> = self.stability_window.iter().cloned().collect();

        for i in 1..values.len() - 1 {
            let prev_diff = values[i] - values[i - 1];
            let curr_diff = values[i + 1] - values[i];

            if prev_diff * curr_diff < 0.0 {
                sign_changes += 1;
            }
        }

        // If more than half the intervals change sign, consider it oscillating
        sign_changes > values.len() / 2
    }
}

/// Gradient monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_adaptive_thresholds: bool,
    pub enable_real_time_monitoring: bool,
    pub stability_threshold: f64,
    pub anomaly_threshold: f64,
    pub update_frequency: usize,
    pub history_window_size: usize,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_thresholds: true,
            enable_real_time_monitoring: true,
            stability_threshold: 0.8,
            anomaly_threshold: 0.7,
            update_frequency: 1,
            history_window_size: 100,
        }
    }
}

/// Monitoring results and insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringResults {
    pub layer_name: String,
    pub timestamp: DateTime<Utc>,
    pub current_status: LayerHealth,
    pub stability_score: f64,
    pub anomaly_score: f64,
    pub alerts: Vec<GradientAlert>,
    pub recommendations: Vec<String>,
}

impl MonitoringResults {
    pub fn new(layer_name: String) -> Self {
        Self {
            layer_name,
            timestamp: Utc::now(),
            current_status: LayerHealth::Healthy,
            stability_score: 1.0,
            anomaly_score: 0.0,
            alerts: Vec::new(),
            recommendations: Vec::new(),
        }
    }

    pub fn add_alert(&mut self, alert: GradientAlert) {
        self.alerts.push(alert);
        self.update_status();
    }

    pub fn add_recommendation(&mut self, recommendation: String) {
        self.recommendations.push(recommendation);
    }

    fn update_status(&mut self) {
        if self.alerts.iter().any(|alert| {
            matches!(
                alert,
                GradientAlert::ExplodingGradients { .. } | GradientAlert::NoGradientFlow { .. }
            )
        }) {
            self.current_status = LayerHealth::Critical;
        } else if !self.alerts.is_empty() || self.anomaly_score > 0.7 {
            self.current_status = LayerHealth::Warning;
        } else {
            self.current_status = LayerHealth::Healthy;
        }
    }
}
