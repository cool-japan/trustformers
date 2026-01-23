//! # Optimizer Monitoring and Analysis Tools
//!
//! This module provides tools for monitoring optimizer performance, tracking metrics,
//! and diagnosing optimization issues during training.
//!
//! ## Features
//!
//! - **Optimizer State Tracking**: Monitor learning rates, gradient norms, parameter changes
//! - **Convergence Analysis**: Track loss trends, detect plateaus, measure convergence rates
//! - **Performance Profiling**: Measure optimizer overhead and memory usage
//! - **Debugging Tools**: Detect gradient explosions, vanishing gradients, oscillations

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use trustformers_core::tensor::Tensor;

/// Configuration for optimizer monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Whether to track gradient norms
    pub track_gradient_norms: bool,
    /// Whether to track parameter changes
    pub track_parameter_changes: bool,
    /// Whether to track learning rate changes
    pub track_learning_rates: bool,
    /// Whether to track convergence metrics
    pub track_convergence: bool,
    /// Whether to track performance metrics
    pub track_performance: bool,
    /// History window size for rolling statistics
    pub history_window: usize,
    /// Frequency of detailed logging (every N steps)
    pub log_frequency: usize,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            track_gradient_norms: true,
            track_parameter_changes: true,
            track_learning_rates: true,
            track_convergence: true,
            track_performance: false,
            history_window: 100,
            log_frequency: 10,
        }
    }
}

/// Statistics for a single metric over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStats {
    /// Recent values (up to history_window size)
    pub values: VecDeque<f32>,
    /// Current value
    pub current: f32,
    /// Mean over history window
    pub mean: f32,
    /// Standard deviation over history window
    pub std: f32,
    /// Minimum value in history
    pub min: f32,
    /// Maximum value in history
    pub max: f32,
    /// Trend (positive = increasing, negative = decreasing)
    pub trend: f32,
}

impl MetricStats {
    pub fn new(window_size: usize) -> Self {
        Self {
            values: VecDeque::with_capacity(window_size),
            current: 0.0,
            mean: 0.0,
            std: 0.0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            trend: 0.0,
        }
    }

    /// Update the metric with a new value.
    pub fn update(&mut self, value: f32, window_size: usize) {
        self.current = value;
        self.values.push_back(value);

        if self.values.len() > window_size {
            self.values.pop_front();
        }

        self.compute_statistics();
    }

    fn compute_statistics(&mut self) {
        if self.values.is_empty() {
            return;
        }

        // Basic statistics
        let sum: f32 = self.values.iter().sum();
        self.mean = sum / self.values.len() as f32;

        let variance: f32 = self.values.iter().map(|x| (x - self.mean).powi(2)).sum::<f32>()
            / self.values.len() as f32;
        self.std = variance.sqrt();

        self.min = self.values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        self.max = self.values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute trend (linear regression slope)
        if self.values.len() >= 2 {
            let n = self.values.len() as f32;
            let x_mean = (n - 1.0) / 2.0; // 0, 1, 2, ... n-1 mean

            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for (i, &y) in self.values.iter().enumerate() {
                let x = i as f32;
                numerator += (x - x_mean) * (y - self.mean);
                denominator += (x - x_mean).powi(2);
            }

            self.trend = if denominator > 1e-8 { numerator / denominator } else { 0.0 };
        }
    }

    /// Check if the metric has plateaued (low variance and trend).
    pub fn is_plateaued(&self, variance_threshold: f32, trend_threshold: f32) -> bool {
        self.std < variance_threshold && self.trend.abs() < trend_threshold
    }

    /// Check if the metric is trending upward.
    pub fn is_increasing(&self, threshold: f32) -> bool {
        self.trend > threshold
    }

    /// Check if the metric is trending downward.
    pub fn is_decreasing(&self, threshold: f32) -> bool {
        self.trend < -threshold
    }
}

/// Performance monitoring data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Total time spent in optimization steps
    pub total_step_time: Duration,
    /// Average time per step
    pub avg_step_time: Duration,
    /// Number of optimization steps
    pub step_count: usize,
    /// Memory usage statistics
    pub memory_usage: Option<MemoryStats>,
}

impl PerformanceStats {
    pub fn new() -> Self {
        Self {
            total_step_time: Duration::new(0, 0),
            avg_step_time: Duration::new(0, 0),
            step_count: 0,
            memory_usage: None,
        }
    }

    /// Record a step timing.
    pub fn record_step_time(&mut self, duration: Duration) {
        self.total_step_time += duration;
        self.step_count += 1;
        self.avg_step_time = self.total_step_time / self.step_count as u32;
    }
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// GPU memory usage in bytes
    pub gpu_memory_bytes: usize,
    /// CPU memory usage in bytes
    pub cpu_memory_bytes: usize,
    /// Peak memory usage
    pub peak_memory_bytes: usize,
}

/// Comprehensive optimizer monitoring data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerMetrics {
    /// Current step number
    pub step: usize,
    /// Learning rate statistics
    pub learning_rate: MetricStats,
    /// Gradient norm statistics
    pub gradient_norm: MetricStats,
    /// Parameter change norm statistics
    pub parameter_change_norm: MetricStats,
    /// Loss statistics (if provided)
    pub loss: MetricStats,
    /// Performance statistics
    pub performance: PerformanceStats,
    /// Parameter-specific gradient norms
    pub parameter_gradient_norms: HashMap<String, MetricStats>,
    /// Convergence indicators
    pub convergence_indicators: ConvergenceIndicators,
}

/// Convergence analysis indicators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceIndicators {
    /// Whether loss appears to have plateaued
    pub loss_plateaued: bool,
    /// Whether gradients are vanishing (very small norms)
    pub gradients_vanishing: bool,
    /// Whether gradients are exploding (very large norms)
    pub gradients_exploding: bool,
    /// Whether training appears to be oscillating
    pub oscillating: bool,
    /// Estimated convergence rate
    pub convergence_rate: f32,
}

impl ConvergenceIndicators {
    pub fn new() -> Self {
        Self {
            loss_plateaued: false,
            gradients_vanishing: false,
            gradients_exploding: false,
            oscillating: false,
            convergence_rate: 0.0,
        }
    }
}

impl Default for ConvergenceIndicators {
    fn default() -> Self {
        Self::new()
    }
}

/// Main optimizer monitor that tracks various metrics.
#[derive(Debug)]
pub struct OptimizerMonitor {
    config: MonitoringConfig,
    metrics: OptimizerMetrics,
    step_start_time: Option<Instant>,
    previous_parameters: Option<Vec<Tensor>>,
}

impl OptimizerMonitor {
    /// Create a new optimizer monitor.
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            metrics: OptimizerMetrics {
                step: 0,
                learning_rate: MetricStats::new(config.history_window),
                gradient_norm: MetricStats::new(config.history_window),
                parameter_change_norm: MetricStats::new(config.history_window),
                loss: MetricStats::new(config.history_window),
                performance: PerformanceStats::new(),
                parameter_gradient_norms: HashMap::new(),
                convergence_indicators: ConvergenceIndicators::new(),
            },
            config,
            step_start_time: None,
            previous_parameters: None,
        }
    }

    /// Create a monitor with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(MonitoringConfig::default())
    }

    /// Called before an optimizer step to start timing.
    pub fn before_step(&mut self) {
        if self.config.track_performance {
            self.step_start_time = Some(Instant::now());
        }
    }

    /// Called after an optimizer step to record metrics.
    pub fn after_step(
        &mut self,
        learning_rate: f32,
        parameters: &[Tensor],
        loss: Option<f32>,
    ) -> Result<()> {
        self.metrics.step += 1;

        // Record timing
        if let Some(start_time) = self.step_start_time.take() {
            let duration = start_time.elapsed();
            self.metrics.performance.record_step_time(duration);
        }

        // Track learning rate
        if self.config.track_learning_rates {
            self.metrics.learning_rate.update(learning_rate, self.config.history_window);
        }

        // Track gradient norms
        if self.config.track_gradient_norms {
            let total_grad_norm = self.compute_total_gradient_norm(parameters)?;
            self.metrics.gradient_norm.update(total_grad_norm, self.config.history_window);

            // Track per-parameter gradient norms
            for (i, param) in parameters.iter().enumerate() {
                if let Ok(grad) = param.grad() {
                    let param_name = format!("param_{}", i);
                    let grad_norm = grad.norm()?;

                    let param_stats = self
                        .metrics
                        .parameter_gradient_norms
                        .entry(param_name)
                        .or_insert_with(|| MetricStats::new(self.config.history_window));
                    param_stats.update(grad_norm, self.config.history_window);
                }
            }
        }

        // Track parameter changes
        if self.config.track_parameter_changes {
            if let Some(prev_params) = &self.previous_parameters {
                let change_norm = self.compute_parameter_change_norm(parameters, prev_params)?;
                self.metrics
                    .parameter_change_norm
                    .update(change_norm, self.config.history_window);
            }
            self.previous_parameters = Some(parameters.to_vec());
        }

        // Track loss
        if let Some(loss_value) = loss {
            self.metrics.loss.update(loss_value, self.config.history_window);
        }

        // Update convergence indicators
        if self.config.track_convergence {
            self.update_convergence_indicators();
        }

        Ok(())
    }

    /// Update loss value (can be called independently of step).
    pub fn update_loss(&mut self, loss: f32) {
        self.metrics.loss.update(loss, self.config.history_window);
        if self.config.track_convergence {
            self.update_convergence_indicators();
        }
    }

    /// Get current metrics.
    pub fn get_metrics(&self) -> &OptimizerMetrics {
        &self.metrics
    }

    /// Check if we should log detailed metrics this step.
    pub fn should_log(&self) -> bool {
        self.metrics.step % self.config.log_frequency == 0
    }

    /// Get a summary report of current optimizer status.
    pub fn get_summary_report(&self) -> String {
        format!(
            "Step {}: LR={:.6}, GradNorm={:.6}±{:.6}, ParamChange={:.6}, Loss={:.6} (trend: {:.6})",
            self.metrics.step,
            self.metrics.learning_rate.current,
            self.metrics.gradient_norm.mean,
            self.metrics.gradient_norm.std,
            self.metrics.parameter_change_norm.current,
            self.metrics.loss.current,
            self.metrics.loss.trend
        )
    }

    /// Get convergence status report.
    pub fn get_convergence_report(&self) -> String {
        let indicators = &self.metrics.convergence_indicators;
        format!(
            "Convergence Status: Loss Plateaued: {}, Gradients Vanishing: {}, Gradients Exploding: {}, Oscillating: {}, Rate: {:.6}",
            indicators.loss_plateaued,
            indicators.gradients_vanishing,
            indicators.gradients_exploding,
            indicators.oscillating,
            indicators.convergence_rate
        )
    }

    /// Reset monitoring state.
    pub fn reset(&mut self) {
        self.metrics = OptimizerMetrics {
            step: 0,
            learning_rate: MetricStats::new(self.config.history_window),
            gradient_norm: MetricStats::new(self.config.history_window),
            parameter_change_norm: MetricStats::new(self.config.history_window),
            loss: MetricStats::new(self.config.history_window),
            performance: PerformanceStats::new(),
            parameter_gradient_norms: HashMap::new(),
            convergence_indicators: ConvergenceIndicators::new(),
        };
        self.previous_parameters = None;
        self.step_start_time = None;
    }

    fn compute_total_gradient_norm(&self, parameters: &[Tensor]) -> Result<f32> {
        let mut total_norm_sq = 0.0;
        for param in parameters {
            if let Ok(grad) = param.grad() {
                let norm_sq = grad.norm_squared()?.to_scalar()?;
                total_norm_sq += norm_sq;
            }
        }
        Ok(total_norm_sq.sqrt())
    }

    fn compute_parameter_change_norm(
        &self,
        current: &[Tensor],
        previous: &[Tensor],
    ) -> Result<f32> {
        if current.len() != previous.len() {
            return Err(anyhow!("Parameter count mismatch"));
        }

        let mut total_change_sq = 0.0;
        for (curr, prev) in current.iter().zip(previous.iter()) {
            let diff = curr.sub(prev)?;
            let norm_sq = diff.norm_squared()?.to_scalar()?;
            total_change_sq += norm_sq;
        }
        Ok(total_change_sq.sqrt())
    }

    fn update_convergence_indicators(&mut self) {
        let indicators = &mut self.metrics.convergence_indicators;

        // Check for loss plateau
        indicators.loss_plateaued = self.metrics.loss.is_plateaued(1e-6, 1e-6);

        // Check for vanishing gradients (very small gradient norms)
        indicators.gradients_vanishing = self.metrics.gradient_norm.current < 1e-8;

        // Check for exploding gradients (very large gradient norms)
        indicators.gradients_exploding = self.metrics.gradient_norm.current > 100.0;

        // Check for oscillations (high variance in loss)
        indicators.oscillating = self.metrics.loss.std > self.metrics.loss.mean * 0.1;

        // Estimate convergence rate from loss trend
        indicators.convergence_rate = -self.metrics.loss.trend; // Negative trend = positive convergence
    }
}

/// Configuration for hyperparameter sensitivity analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSensitivityConfig {
    /// Whether to enable sensitivity analysis
    pub enabled: bool,
    /// Perturbation magnitude for finite difference approximation
    pub perturbation_magnitude: f32,
    /// Number of steps to analyze sensitivity over
    pub analysis_window: usize,
    /// Minimum number of samples before computing sensitivity
    pub min_samples: usize,
    /// Which hyperparameters to analyze
    pub analyze_learning_rate: bool,
    /// Whether to analyze momentum parameters
    pub analyze_momentum: bool,
    /// Whether to analyze weight decay
    pub analyze_weight_decay: bool,
    /// Whether to analyze epsilon (for Adam-like optimizers)
    pub analyze_epsilon: bool,
    /// Frequency of sensitivity analysis (every N steps)
    pub analysis_frequency: usize,
}

impl Default for HyperparameterSensitivityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            perturbation_magnitude: 0.01, // 1% perturbation
            analysis_window: 50,
            min_samples: 10,
            analyze_learning_rate: true,
            analyze_momentum: true,
            analyze_weight_decay: true,
            analyze_epsilon: false,
            analysis_frequency: 25,
        }
    }
}

/// Sensitivity metrics for a specific hyperparameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSensitivityMetrics {
    /// Hyperparameter name
    pub name: String,
    /// Current sensitivity estimate (∂loss/∂hyperparameter)
    pub current_sensitivity: f32,
    /// Historical sensitivity values
    pub sensitivity_history: VecDeque<f32>,
    /// Mean sensitivity over history
    pub mean_sensitivity: f32,
    /// Standard deviation of sensitivity
    pub std_sensitivity: f32,
    /// Normalized sensitivity (sensitivity / hyperparameter_value)
    pub normalized_sensitivity: f32,
    /// Relative importance score (0-1)
    pub importance_score: f32,
}

impl HyperparameterSensitivityMetrics {
    pub fn new(name: String, window_size: usize) -> Self {
        Self {
            name,
            current_sensitivity: 0.0,
            sensitivity_history: VecDeque::with_capacity(window_size),
            mean_sensitivity: 0.0,
            std_sensitivity: 0.0,
            normalized_sensitivity: 0.0,
            importance_score: 0.0,
        }
    }

    /// Update sensitivity with a new measurement.
    pub fn update(&mut self, sensitivity: f32, hyperparameter_value: f32, window_size: usize) {
        self.current_sensitivity = sensitivity;
        self.sensitivity_history.push_back(sensitivity);

        if self.sensitivity_history.len() > window_size {
            self.sensitivity_history.pop_front();
        }

        self.compute_statistics(hyperparameter_value);
    }

    fn compute_statistics(&mut self, hyperparameter_value: f32) {
        if self.sensitivity_history.is_empty() {
            return;
        }

        // Basic statistics
        let sum: f32 = self.sensitivity_history.iter().sum();
        self.mean_sensitivity = sum / self.sensitivity_history.len() as f32;

        let variance: f32 = self
            .sensitivity_history
            .iter()
            .map(|x| (x - self.mean_sensitivity).powi(2))
            .sum::<f32>()
            / self.sensitivity_history.len() as f32;
        self.std_sensitivity = variance.sqrt();

        // Normalized sensitivity (relative to hyperparameter value)
        if hyperparameter_value.abs() > 1e-8 {
            self.normalized_sensitivity = self.current_sensitivity / hyperparameter_value;
        } else {
            self.normalized_sensitivity = 0.0;
        }

        // Importance score based on magnitude and stability
        let magnitude_score = self.normalized_sensitivity.abs().tanh(); // Bounded [0,1]
        let stability_score = (-self.std_sensitivity.abs()).exp(); // Higher for more stable
        self.importance_score = magnitude_score * stability_score;
    }

    /// Check if this hyperparameter is highly sensitive.
    pub fn is_highly_sensitive(&self, threshold: f32) -> bool {
        self.importance_score > threshold
    }

    /// Check if sensitivity is stable (low variance).
    pub fn is_stable(&self, variance_threshold: f32) -> bool {
        self.std_sensitivity < variance_threshold
    }
}

/// Main hyperparameter sensitivity analyzer.
#[derive(Debug)]
pub struct HyperparameterSensitivity {
    config: HyperparameterSensitivityConfig,
    sensitivity_metrics: HashMap<String, HyperparameterSensitivityMetrics>,
    baseline_loss: Option<f32>,
    perturbation_losses: HashMap<String, f32>,
    step_count: usize,
}

impl HyperparameterSensitivity {
    /// Create a new sensitivity analyzer.
    pub fn new(config: HyperparameterSensitivityConfig) -> Self {
        Self {
            config,
            sensitivity_metrics: HashMap::new(),
            baseline_loss: None,
            perturbation_losses: HashMap::new(),
            step_count: 0,
        }
    }

    /// Create analyzer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(HyperparameterSensitivityConfig::default())
    }

    /// Record baseline loss for sensitivity analysis.
    pub fn record_baseline_loss(&mut self, loss: f32) {
        self.baseline_loss = Some(loss);
    }

    /// Record loss after hyperparameter perturbation.
    pub fn record_perturbation_loss(&mut self, hyperparameter_name: String, loss: f32) {
        self.perturbation_losses.insert(hyperparameter_name, loss);
    }

    /// Compute sensitivity for a specific hyperparameter.
    pub fn compute_sensitivity(
        &mut self,
        hyperparameter_name: &str,
        hyperparameter_value: f32,
        perturbed_value: f32,
        loss_change: f32,
    ) -> f32 {
        let param_change = perturbed_value - hyperparameter_value;

        // Avoid division by zero
        if param_change.abs() < 1e-12 {
            return 0.0;
        }

        // Finite difference approximation: ∂loss/∂param ≈ Δloss/Δparam
        let sensitivity = loss_change / param_change;

        // Update or create sensitivity metrics
        let metrics = self
            .sensitivity_metrics
            .entry(hyperparameter_name.to_string())
            .or_insert_with(|| {
                HyperparameterSensitivityMetrics::new(
                    hyperparameter_name.to_string(),
                    self.config.analysis_window,
                )
            });

        metrics.update(
            sensitivity,
            hyperparameter_value,
            self.config.analysis_window,
        );

        sensitivity
    }

    /// Analyze sensitivity for learning rate.
    pub fn analyze_learning_rate_sensitivity(
        &mut self,
        current_lr: f32,
        baseline_loss: f32,
        perturbed_loss: f32,
    ) -> f32 {
        let perturbed_lr = current_lr * (1.0 + self.config.perturbation_magnitude);
        let loss_change = perturbed_loss - baseline_loss;

        self.compute_sensitivity("learning_rate", current_lr, perturbed_lr, loss_change)
    }

    /// Analyze sensitivity for momentum parameter.
    pub fn analyze_momentum_sensitivity(
        &mut self,
        current_momentum: f32,
        baseline_loss: f32,
        perturbed_loss: f32,
    ) -> f32 {
        let perturbed_momentum = current_momentum * (1.0 + self.config.perturbation_magnitude);
        let loss_change = perturbed_loss - baseline_loss;

        self.compute_sensitivity(
            "momentum",
            current_momentum,
            perturbed_momentum,
            loss_change,
        )
    }

    /// Analyze sensitivity for weight decay.
    pub fn analyze_weight_decay_sensitivity(
        &mut self,
        current_weight_decay: f32,
        baseline_loss: f32,
        perturbed_loss: f32,
    ) -> f32 {
        let perturbed_weight_decay =
            current_weight_decay * (1.0 + self.config.perturbation_magnitude);
        let loss_change = perturbed_loss - baseline_loss;

        self.compute_sensitivity(
            "weight_decay",
            current_weight_decay,
            perturbed_weight_decay,
            loss_change,
        )
    }

    /// Analyze sensitivity for epsilon parameter.
    pub fn analyze_epsilon_sensitivity(
        &mut self,
        current_epsilon: f32,
        baseline_loss: f32,
        perturbed_loss: f32,
    ) -> f32 {
        let perturbed_epsilon = current_epsilon * (1.0 + self.config.perturbation_magnitude);
        let loss_change = perturbed_loss - baseline_loss;

        self.compute_sensitivity("epsilon", current_epsilon, perturbed_epsilon, loss_change)
    }

    /// Get sensitivity metrics for a specific hyperparameter.
    pub fn get_sensitivity_metrics(
        &self,
        hyperparameter: &str,
    ) -> Option<&HyperparameterSensitivityMetrics> {
        self.sensitivity_metrics.get(hyperparameter)
    }

    /// Get all sensitivity metrics.
    pub fn get_all_sensitivity_metrics(
        &self,
    ) -> &HashMap<String, HyperparameterSensitivityMetrics> {
        &self.sensitivity_metrics
    }

    /// Get most sensitive hyperparameters (sorted by importance score).
    pub fn get_most_sensitive_hyperparameters(
        &self,
    ) -> Vec<(&String, &HyperparameterSensitivityMetrics)> {
        let mut sorted: Vec<_> = self.sensitivity_metrics.iter().collect();
        sorted.sort_by(|a, b| {
            b.1.importance_score
                .partial_cmp(&a.1.importance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Check if sensitivity analysis should be performed this step.
    pub fn should_analyze(&self) -> bool {
        self.config.enabled
            && self.step_count % self.config.analysis_frequency == 0
            && self.step_count >= self.config.min_samples
    }

    /// Increment step count.
    pub fn step(&mut self) {
        self.step_count += 1;
    }

    /// Get a summary report of hyperparameter sensitivities.
    pub fn get_sensitivity_report(&self) -> String {
        let mut report = String::from("Hyperparameter Sensitivity Analysis:\n");

        let sorted_metrics = self.get_most_sensitive_hyperparameters();

        for (name, metrics) in sorted_metrics.iter().take(5) {
            // Top 5 most sensitive
            report.push_str(&format!(
                "  {}: Sensitivity={:.6}, Normalized={:.6}, Importance={:.3} ({})\n",
                name,
                metrics.current_sensitivity,
                metrics.normalized_sensitivity,
                metrics.importance_score,
                if metrics.is_highly_sensitive(0.5) { "HIGH" } else { "LOW" }
            ));
        }

        if sorted_metrics.is_empty() {
            report.push_str("  No sensitivity data available yet.\n");
        }

        report
    }

    /// Get recommendations based on sensitivity analysis.
    pub fn get_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        for (name, metrics) in &self.sensitivity_metrics {
            if metrics.is_highly_sensitive(0.7) {
                recommendations.push(format!(
                    "Consider careful tuning of {}: high sensitivity detected (score: {:.3})",
                    name, metrics.importance_score
                ));
            }

            if !metrics.is_stable(0.1) {
                recommendations.push(format!(
                    "Consider stabilizing {}: sensitivity varies significantly (std: {:.6})",
                    name, metrics.std_sensitivity
                ));
            }
        }

        if recommendations.is_empty() {
            recommendations.push(
                "All hyperparameters appear to have reasonable sensitivity profiles.".to_string(),
            );
        }

        recommendations
    }

    /// Reset sensitivity analysis state.
    pub fn reset(&mut self) {
        self.sensitivity_metrics.clear();
        self.baseline_loss = None;
        self.perturbation_losses.clear();
        self.step_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_stats_creation() {
        let stats = MetricStats::new(10);
        assert_eq!(stats.values.capacity(), 10);
        assert_eq!(stats.current, 0.0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_metric_stats_update() {
        let mut stats = MetricStats::new(3);

        stats.update(1.0, 3);
        assert_eq!(stats.current, 1.0);
        assert_eq!(stats.mean, 1.0);

        stats.update(2.0, 3);
        assert_eq!(stats.current, 2.0);
        assert_eq!(stats.mean, 1.5);

        stats.update(3.0, 3);
        assert_eq!(stats.current, 3.0);
        assert_eq!(stats.mean, 2.0);

        // Should maintain window size
        stats.update(4.0, 3);
        assert_eq!(stats.values.len(), 3);
        assert_eq!(stats.mean, 3.0); // (2+3+4)/3
    }

    #[test]
    fn test_metric_stats_trend() {
        let mut stats = MetricStats::new(10);

        // Add increasing values
        for i in 1..=5 {
            stats.update(i as f32, 10);
        }

        // Should detect positive trend
        assert!(stats.trend > 0.0);
        assert!(stats.is_increasing(0.5));
        assert!(!stats.is_decreasing(0.5));
    }

    #[test]
    fn test_performance_stats() {
        let mut perf = PerformanceStats::new();
        assert_eq!(perf.step_count, 0);

        perf.record_step_time(Duration::from_millis(100));
        assert_eq!(perf.step_count, 1);
        assert_eq!(perf.avg_step_time, Duration::from_millis(100));

        perf.record_step_time(Duration::from_millis(200));
        assert_eq!(perf.step_count, 2);
        assert_eq!(perf.avg_step_time, Duration::from_millis(150));
    }

    #[test]
    fn test_convergence_indicators() {
        let indicators = ConvergenceIndicators::new();
        assert!(!indicators.loss_plateaued);
        assert!(!indicators.gradients_vanishing);
        assert!(!indicators.gradients_exploding);
        assert!(!indicators.oscillating);
        assert_eq!(indicators.convergence_rate, 0.0);
    }

    #[test]
    fn test_optimizer_monitor_creation() {
        let monitor = OptimizerMonitor::with_defaults();
        assert_eq!(monitor.metrics.step, 0);
        assert!(monitor.previous_parameters.is_none());
    }

    #[test]
    fn test_monitor_should_log() {
        let mut monitor = OptimizerMonitor::with_defaults();

        // Should log at step 0
        assert!(monitor.should_log());

        monitor.metrics.step = 5;
        assert!(!monitor.should_log()); // Default frequency is 10

        monitor.metrics.step = 10;
        assert!(monitor.should_log());
    }

    #[test]
    fn test_hyperparameter_sensitivity_config() {
        let config = HyperparameterSensitivityConfig::default();
        assert!(config.enabled);
        assert_eq!(config.perturbation_magnitude, 0.01);
        assert_eq!(config.analysis_window, 50);
        assert_eq!(config.min_samples, 10);
        assert!(config.analyze_learning_rate);
        assert!(config.analyze_momentum);
        assert!(config.analyze_weight_decay);
        assert!(!config.analyze_epsilon);
        assert_eq!(config.analysis_frequency, 25);
    }

    #[test]
    fn test_hyperparameter_sensitivity_metrics() {
        let mut metrics = HyperparameterSensitivityMetrics::new("learning_rate".to_string(), 10);
        assert_eq!(metrics.name, "learning_rate");
        assert_eq!(metrics.current_sensitivity, 0.0);
        assert_eq!(metrics.importance_score, 0.0);

        // Update with some sensitivity values
        metrics.update(0.5, 0.01, 10); // sensitivity=0.5, lr=0.01
        assert_eq!(metrics.current_sensitivity, 0.5);
        assert_eq!(metrics.normalized_sensitivity, 0.5 / 0.01);
        assert!(metrics.importance_score > 0.0);

        metrics.update(0.3, 0.01, 10);
        assert_eq!(metrics.current_sensitivity, 0.3);
        assert_eq!(metrics.sensitivity_history.len(), 2);

        // Check if metrics are computed correctly
        assert_eq!(metrics.mean_sensitivity, 0.4); // (0.5 + 0.3) / 2
    }

    #[test]
    fn test_hyperparameter_sensitivity_analyzer() {
        let mut analyzer = HyperparameterSensitivity::with_defaults();

        // Test baseline loss recording
        analyzer.record_baseline_loss(1.0);
        assert_eq!(analyzer.baseline_loss, Some(1.0));

        // Test perturbation loss recording
        analyzer.record_perturbation_loss("learning_rate".to_string(), 1.1);
        assert_eq!(
            analyzer.perturbation_losses.get("learning_rate"),
            Some(&1.1)
        );

        // Test sensitivity computation
        let sensitivity = analyzer.compute_sensitivity("learning_rate", 0.01, 0.0101, 0.1);
        let expected = 0.1 / 0.0001; // loss_change / param_change = 1000.0
        assert!(
            (sensitivity - expected).abs() < 0.01,
            "Expected {}, got {}",
            expected,
            sensitivity
        );

        // Check that metrics were created
        assert!(analyzer.sensitivity_metrics.contains_key("learning_rate"));
    }

    #[test]
    fn test_sensitivity_analysis_methods() {
        let mut analyzer = HyperparameterSensitivity::with_defaults();

        // Test learning rate sensitivity
        let lr_sensitivity = analyzer.analyze_learning_rate_sensitivity(0.01, 1.0, 1.1);
        assert!(lr_sensitivity > 0.0);
        assert!(analyzer.sensitivity_metrics.contains_key("learning_rate"));

        // Test momentum sensitivity
        let momentum_sensitivity = analyzer.analyze_momentum_sensitivity(0.9, 1.0, 0.95);
        assert!(momentum_sensitivity < 0.0); // Loss decreased
        assert!(analyzer.sensitivity_metrics.contains_key("momentum"));

        // Test weight decay sensitivity
        let wd_sensitivity = analyzer.analyze_weight_decay_sensitivity(0.01, 1.0, 1.05);
        assert!(wd_sensitivity > 0.0);
        assert!(analyzer.sensitivity_metrics.contains_key("weight_decay"));
    }

    #[test]
    fn test_sensitivity_should_analyze() {
        let mut analyzer = HyperparameterSensitivity::with_defaults();

        // Should not analyze initially (below min_samples)
        assert!(!analyzer.should_analyze());

        // Step forward to reach min_samples
        for _ in 0..10 {
            analyzer.step();
        }
        assert!(!analyzer.should_analyze()); // Still not at frequency

        // Step to reach analysis frequency
        for _ in 0..15 {
            analyzer.step();
        }
        assert!(analyzer.should_analyze()); // Now at step 25
    }

    #[test]
    fn test_sensitivity_report_generation() {
        let mut analyzer = HyperparameterSensitivity::with_defaults();

        // Add some sensitivity data
        analyzer.compute_sensitivity("learning_rate", 0.01, 0.0101, 0.1);
        analyzer.compute_sensitivity("momentum", 0.9, 0.909, -0.05);

        let report = analyzer.get_sensitivity_report();
        assert!(report.contains("Hyperparameter Sensitivity Analysis"));
        assert!(report.contains("learning_rate"));
        assert!(report.contains("momentum"));

        let recommendations = analyzer.get_recommendations();
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_sensitivity_most_sensitive_hyperparameters() {
        let mut analyzer = HyperparameterSensitivity::with_defaults();

        // Add different sensitivity levels
        analyzer.compute_sensitivity("learning_rate", 0.01, 0.0101, 0.2); // High sensitivity
        analyzer.compute_sensitivity("momentum", 0.9, 0.909, 0.01); // Low sensitivity
        analyzer.compute_sensitivity("weight_decay", 0.01, 0.0101, 0.15); // Medium sensitivity

        let most_sensitive = analyzer.get_most_sensitive_hyperparameters();
        assert_eq!(most_sensitive.len(), 3);

        // Should be sorted by importance score (descending)
        let first_importance = most_sensitive[0].1.importance_score;
        let second_importance = most_sensitive[1].1.importance_score;
        assert!(first_importance >= second_importance);
    }

    #[test]
    fn test_sensitivity_reset() {
        let mut analyzer = HyperparameterSensitivity::with_defaults();

        // Add some data
        analyzer.record_baseline_loss(1.0);
        analyzer.record_perturbation_loss("learning_rate".to_string(), 1.1);
        analyzer.compute_sensitivity("learning_rate", 0.01, 0.0101, 0.1);
        analyzer.step();

        // Verify data exists
        assert!(analyzer.baseline_loss.is_some());
        assert!(!analyzer.perturbation_losses.is_empty());
        assert!(!analyzer.sensitivity_metrics.is_empty());
        assert_eq!(analyzer.step_count, 1);

        // Reset and verify everything is cleared
        analyzer.reset();
        assert!(analyzer.baseline_loss.is_none());
        assert!(analyzer.perturbation_losses.is_empty());
        assert!(analyzer.sensitivity_metrics.is_empty());
        assert_eq!(analyzer.step_count, 0);
    }
}

/// Optimizer Performance Analysis and Selection Tool
///
/// Helps users choose the right optimizer based on their requirements
/// including performance characteristics, model size, and training objectives.
#[derive(Debug, Clone)]
pub struct OptimizerSelector {
    /// Model parameter count
    pub model_size: usize,
    /// Training duration requirements (training time sensitivity)
    pub time_sensitive: bool,
    /// Memory constraints
    pub memory_constrained: bool,
    /// Convergence speed priority
    pub fast_convergence: bool,
    /// Robustness requirements (handling diverse training conditions)
    pub robustness_priority: bool,
    /// Advanced features requirements (entropy weighting, adaptive norms)
    pub advanced_features: bool,
}

/// Optimizer recommendation with performance characteristics
#[derive(Debug, Clone)]
pub struct OptimizerRecommendation {
    pub name: String,
    pub description: String,
    pub performance_tier: PerformanceTier,
    pub convergence_speed: ConvergenceSpeed,
    pub memory_usage: MemoryUsage,
    pub use_cases: Vec<String>,
    pub estimated_overhead: f32, // multiplier compared to Adam baseline
}

#[derive(Debug, Clone)]
pub enum PerformanceTier {
    Fastest,  // Traditional optimizers (Adam, SGD, AdamW)
    Moderate, // HN-Adam, Lookahead variants
    Advanced, // BGE-Adam, complex entropy-based methods
}

#[derive(Debug, Clone)]
pub enum ConvergenceSpeed {
    Fast,
    Moderate,
    Superior, // Better than standard methods
}

#[derive(Debug, Clone)]
pub enum MemoryUsage {
    Low,      // Similar to SGD
    Standard, // Adam-level
    High,     // Complex state tracking
}

impl OptimizerSelector {
    pub fn new(model_size: usize) -> Self {
        Self {
            model_size,
            time_sensitive: false,
            memory_constrained: false,
            fast_convergence: false,
            robustness_priority: false,
            advanced_features: false,
        }
    }

    pub fn time_sensitive(mut self, sensitive: bool) -> Self {
        self.time_sensitive = sensitive;
        self
    }

    pub fn memory_constrained(mut self, constrained: bool) -> Self {
        self.memory_constrained = constrained;
        self
    }

    pub fn fast_convergence(mut self, fast: bool) -> Self {
        self.fast_convergence = fast;
        self
    }

    pub fn robustness_priority(mut self, robust: bool) -> Self {
        self.robustness_priority = robust;
        self
    }

    pub fn advanced_features(mut self, advanced: bool) -> Self {
        self.advanced_features = advanced;
        self
    }

    /// Get optimizer recommendations ranked by suitability
    pub fn get_recommendations(&self) -> Vec<OptimizerRecommendation> {
        let mut recommendations = self.generate_all_recommendations();
        self.rank_recommendations(&mut recommendations);
        recommendations
    }

    /// Generate recommendations for all available optimizers
    fn generate_all_recommendations(&self) -> Vec<OptimizerRecommendation> {
        vec![
            OptimizerRecommendation {
                name: "AdamW".to_string(),
                description: "Decoupled weight decay Adam - excellent all-around optimizer"
                    .to_string(),
                performance_tier: PerformanceTier::Fastest,
                convergence_speed: ConvergenceSpeed::Fast,
                memory_usage: MemoryUsage::Standard,
                use_cases: vec![
                    "General purpose training".to_string(),
                    "Large language models".to_string(),
                    "Computer vision".to_string(),
                    "Production training".to_string(),
                ],
                estimated_overhead: 1.0, // baseline
            },
            OptimizerRecommendation {
                name: "Adam".to_string(),
                description: "Classic adaptive moment estimation optimizer".to_string(),
                performance_tier: PerformanceTier::Fastest,
                convergence_speed: ConvergenceSpeed::Fast,
                memory_usage: MemoryUsage::Standard,
                use_cases: vec![
                    "General purpose training".to_string(),
                    "Research and experimentation".to_string(),
                    "Quick prototyping".to_string(),
                ],
                estimated_overhead: 1.05, // slightly slower than AdamW
            },
            OptimizerRecommendation {
                name: "SGD".to_string(),
                description: "Stochastic gradient descent with momentum - simple and effective"
                    .to_string(),
                performance_tier: PerformanceTier::Fastest,
                convergence_speed: ConvergenceSpeed::Moderate,
                memory_usage: MemoryUsage::Low,
                use_cases: vec![
                    "Memory-constrained training".to_string(),
                    "Simple models".to_string(),
                    "Fine-tuning".to_string(),
                    "Educational purposes".to_string(),
                ],
                estimated_overhead: 1.1, // simple but effective
            },
            OptimizerRecommendation {
                name: "HN-Adam".to_string(),
                description: "Hybrid Norm Adam with adaptive step size based on parameter norms"
                    .to_string(),
                performance_tier: PerformanceTier::Moderate,
                convergence_speed: ConvergenceSpeed::Superior,
                memory_usage: MemoryUsage::Standard,
                use_cases: vec![
                    "Transformer training".to_string(),
                    "Computer vision tasks".to_string(),
                    "When adaptive learning rates are needed".to_string(),
                    "Research requiring latest optimization techniques".to_string(),
                ],
                estimated_overhead: 2.5, // ~2.5x slower but adaptive
            },
            OptimizerRecommendation {
                name: "BGE-Adam".to_string(),
                description: "Entropy-weighted Adam with adaptive gradient strategies".to_string(),
                performance_tier: PerformanceTier::Advanced,
                convergence_speed: ConvergenceSpeed::Superior,
                memory_usage: MemoryUsage::High,
                use_cases: vec![
                    "Research and experimentation".to_string(),
                    "Complex training scenarios".to_string(),
                    "When robustness is critical".to_string(),
                    "Handling diverse gradient conditions".to_string(),
                ],
                estimated_overhead: 13.0, // ~13x slower due to entropy calculations
            },
        ]
    }

    /// Rank recommendations based on user requirements
    fn rank_recommendations(&self, recommendations: &mut [OptimizerRecommendation]) {
        recommendations.sort_by(|a, b| {
            let score_a = self.calculate_suitability_score(a);
            let score_b = self.calculate_suitability_score(b);
            score_b.partial_cmp(&score_a).unwrap()
        });
    }

    /// Calculate suitability score for a recommendation
    fn calculate_suitability_score(&self, rec: &OptimizerRecommendation) -> f32 {
        let mut score = 0.0;

        // Performance requirements
        if self.time_sensitive {
            score += match rec.performance_tier {
                PerformanceTier::Fastest => 10.0,
                PerformanceTier::Moderate => 5.0,
                PerformanceTier::Advanced => 1.0,
            };
        }

        // Memory constraints
        if self.memory_constrained {
            score += match rec.memory_usage {
                MemoryUsage::Low => 10.0,
                MemoryUsage::Standard => 5.0,
                MemoryUsage::High => 1.0,
            };
        }

        // Convergence speed priority
        if self.fast_convergence {
            score += match rec.convergence_speed {
                ConvergenceSpeed::Superior => 10.0,
                ConvergenceSpeed::Fast => 7.0,
                ConvergenceSpeed::Moderate => 3.0,
            };
        }

        // Robustness priority
        if self.robustness_priority {
            match rec.name.as_str() {
                "BGE-Adam" => score += 10.0, // Highest robustness
                "HN-Adam" => score += 7.0,   // Good robustness
                "AdamW" => score += 5.0,     // Standard robustness
                _ => score += 3.0,
            }
        }

        // Advanced features
        if self.advanced_features {
            match rec.name.as_str() {
                "BGE-Adam" => score += 10.0, // Entropy weighting
                "HN-Adam" => score += 8.0,   // Adaptive norms
                _ => score += 2.0,
            }
        }

        // Model size considerations
        if self.model_size > 1_000_000 {
            // Large models benefit from stable optimizers
            match rec.name.as_str() {
                "AdamW" | "Adam" => score += 5.0,
                "HN-Adam" => score += 3.0,
                _ => score += 1.0,
            }
        }

        // Base score for general usability
        match rec.name.as_str() {
            "AdamW" => score += 8.0,    // Excellent general purpose
            "Adam" => score += 7.0,     // Good general purpose
            "HN-Adam" => score += 6.0,  // Good with advanced features
            "SGD" => score += 5.0,      // Simple and reliable
            "BGE-Adam" => score += 4.0, // Specialized use case
            _ => score += 2.0,
        }

        score
    }

    /// Generate a detailed report with recommendations
    pub fn generate_report(&self) -> String {
        let recommendations = self.get_recommendations();
        let mut report = String::new();

        report.push_str("🚀 TrustformeRS Optimizer Selection Report\n");
        report.push_str("=========================================\n\n");

        report.push_str("📊 Model Configuration:\n");
        report.push_str(&format!(
            "   • Model size: {} parameters\n",
            self.model_size
        ));
        report.push_str(&format!("   • Time sensitive: {}\n", self.time_sensitive));
        report.push_str(&format!(
            "   • Memory constrained: {}\n",
            self.memory_constrained
        ));
        report.push_str(&format!(
            "   • Fast convergence priority: {}\n",
            self.fast_convergence
        ));
        report.push_str(&format!(
            "   • Robustness priority: {}\n",
            self.robustness_priority
        ));
        report.push_str(&format!(
            "   • Advanced features: {}\n\n",
            self.advanced_features
        ));

        report.push_str("🏆 Recommended Optimizers (ranked by suitability):\n\n");

        for (i, rec) in recommendations.iter().enumerate() {
            let rank_emoji = match i {
                0 => "🥇",
                1 => "🥈",
                2 => "🥉",
                _ => "📊",
            };

            report.push_str(&format!(
                "{} {} - {}\n",
                rank_emoji, rec.name, rec.description
            ));
            report.push_str(&format!(
                "   Performance: {:?} | Convergence: {:?} | Memory: {:?}\n",
                rec.performance_tier, rec.convergence_speed, rec.memory_usage
            ));
            report.push_str(&format!(
                "   Overhead: {:.1}x compared to baseline\n",
                rec.estimated_overhead
            ));
            report.push_str(&format!("   Use cases: {}\n\n", rec.use_cases.join(", ")));
        }

        report.push_str("💡 Performance Insights from Latest Benchmarks:\n");
        report.push_str("   • AdamW: 238µs/iter (100K params) - Fast and reliable\n");
        report.push_str("   • Adam: 248µs/iter - Slightly slower than AdamW\n");
        report.push_str("   • SGD: 257µs/iter - Simple and memory efficient\n");
        report.push_str("   • HN-Adam: 633µs/iter - 2.5x slower, adaptive step sizes\n");
        report.push_str("   • BGE-Adam: 3.3ms/iter - 13x slower, entropy-based robustness\n\n");

        report.push_str("🎯 Quick Selection Guide:\n");
        report.push_str("   • Production training: AdamW\n");
        report.push_str("   • Memory constrained: SGD\n");
        report.push_str("   • Research/experimentation: HN-Adam or BGE-Adam\n");
        report.push_str("   • Maximum robustness: BGE-Adam\n");
        report.push_str("   • Adaptive learning rates: HN-Adam\n");

        report
    }
}

#[cfg(test)]
mod optimizer_selection_tests {
    use super::*;

    #[test]
    fn test_optimizer_selector_basic() {
        let selector = OptimizerSelector::new(10000);
        let recommendations = selector.get_recommendations();
        assert!(!recommendations.is_empty());
        assert_eq!(recommendations.len(), 5); // All available optimizers
    }

    #[test]
    fn test_time_sensitive_selection() {
        let selector = OptimizerSelector::new(10000).time_sensitive(true);

        let recommendations = selector.get_recommendations();
        let top_rec = &recommendations[0];

        // Should prioritize fastest optimizers
        assert!(matches!(top_rec.performance_tier, PerformanceTier::Fastest));
        assert!(top_rec.name == "AdamW" || top_rec.name == "Adam" || top_rec.name == "SGD");
    }

    #[test]
    fn test_memory_constrained_selection() {
        let selector = OptimizerSelector::new(10000).memory_constrained(true);

        let recommendations = selector.get_recommendations();
        let top_rec = &recommendations[0];

        // Should prioritize low memory optimizers
        assert!(top_rec.name == "SGD" || matches!(top_rec.memory_usage, MemoryUsage::Low));
    }

    #[test]
    fn test_robustness_priority_selection() {
        let selector = OptimizerSelector::new(10000).robustness_priority(true);

        let recommendations = selector.get_recommendations();
        let top_rec = &recommendations[0];

        // Should prioritize BGE-Adam for robustness
        assert!(top_rec.name == "BGE-Adam" || top_rec.name == "HN-Adam");
    }

    #[test]
    fn test_advanced_features_selection() {
        let selector = OptimizerSelector::new(10000).advanced_features(true);

        let recommendations = selector.get_recommendations();
        let top_rec = &recommendations[0];

        // Should prioritize advanced optimizers
        assert!(top_rec.name == "BGE-Adam" || top_rec.name == "HN-Adam");
    }

    #[test]
    fn test_report_generation() {
        let selector = OptimizerSelector::new(50000).time_sensitive(true).fast_convergence(true);

        let report = selector.generate_report();
        assert!(report.contains("TrustformeRS Optimizer Selection Report"));
        assert!(report.contains("Model size: 50000"));
        assert!(report.contains("Time sensitive: true"));
        assert!(report.contains("🥇")); // Should have rankings
        assert!(report.contains("Performance Insights"));
    }
}
