//! Advanced Training Stability Monitoring System
//!
//! This module provides predictive anomaly detection and proactive recovery mechanisms
//! that go beyond traditional reactive monitoring to prevent training failures before they occur.

use anyhow::Result;
use log;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use trustformers_core::errors::runtime_error;
use trustformers_core::tensor::Tensor;

/// Configuration for advanced stability monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedStabilityConfig {
    /// Enable predictive anomaly detection
    pub predictive_detection: bool,
    /// Enable proactive recovery mechanisms
    pub proactive_recovery: bool,
    /// Enable training dynamics analysis
    pub dynamics_analysis: bool,
    /// Enable loss landscape monitoring
    pub loss_landscape_monitoring: bool,
    /// Prediction horizon (steps ahead)
    pub prediction_horizon: usize,
    /// Confidence threshold for predictions
    pub prediction_confidence_threshold: f32,
    /// Pattern detection window size
    pub pattern_window_size: usize,
    /// Stability score threshold
    pub stability_threshold: f32,
    /// Adaptive recovery enabled
    pub adaptive_recovery: bool,
}

impl Default for AdvancedStabilityConfig {
    fn default() -> Self {
        Self {
            predictive_detection: true,
            proactive_recovery: true,
            dynamics_analysis: true,
            loss_landscape_monitoring: true,
            prediction_horizon: 10,
            prediction_confidence_threshold: 0.7,
            pattern_window_size: 50,
            stability_threshold: 0.8,
            adaptive_recovery: true,
        }
    }
}

/// Training dynamics patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDynamics {
    /// Loss trajectory trend
    pub loss_trend: TrendDirection,
    /// Gradient norm evolution
    pub gradient_trend: TrendDirection,
    /// Learning rate effectiveness
    pub lr_effectiveness: f32,
    /// Convergence velocity
    pub convergence_velocity: f32,
    /// Oscillation frequency
    pub oscillation_frequency: f32,
    /// Phase space trajectory
    pub phase_trajectory: Vec<(f32, f32)>, // (loss, gradient_norm)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Decreasing,
    Increasing,
    Stable,
    Oscillating,
    Diverging,
}

/// Predictive anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveAnomaly {
    /// Predicted step where anomaly will occur
    pub predicted_step: usize,
    /// Type of predicted anomaly
    pub anomaly_type: PredictedAnomalyType,
    /// Confidence of prediction (0-1)
    pub confidence: f32,
    /// Time to occurrence (estimated steps)
    pub time_to_occurrence: usize,
    /// Suggested preventive actions
    pub preventive_actions: Vec<PreventiveAction>,
    /// Risk level
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictedAnomalyType {
    GradientExplosion,
    GradientVanishing,
    TrainingStagnation,
    ConvergenceFailure,
    NumericalInstability,
    OscillatingLoss,
    MemoryExhaustion,
    LearningRateDeterioration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreventiveAction {
    ReduceLearningRate {
        factor: f32,
    },
    IncreaseGradientClipping {
        new_threshold: f32,
    },
    AdjustOptimizer {
        suggested_params: HashMap<String, f32>,
    },
    TriggerEarlyCheckpoint,
    ModifyBatchSize {
        new_size: usize,
    },
    AdjustWarmupSchedule,
    EnableNoise {
        noise_level: f32,
    },
    ResetAccumulatedGradients,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Loss landscape analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossLandscapeAnalysis {
    /// Local curvature estimate
    pub local_curvature: f32,
    /// Gradient consistency score
    pub gradient_consistency: f32,
    /// Escape difficulty from current region
    pub escape_difficulty: f32,
    /// Basin stability
    pub basin_stability: f32,
    /// Saddle point probability
    pub saddle_point_prob: f32,
}

/// Stability score breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityScore {
    /// Overall stability score (0-1)
    pub overall_score: f32,
    /// Gradient stability component
    pub gradient_stability: f32,
    /// Loss stability component
    pub loss_stability: f32,
    /// Convergence stability component
    pub convergence_stability: f32,
    /// Numerical stability component
    pub numerical_stability: f32,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Advanced stability monitor
#[allow(dead_code)]
pub struct AdvancedStabilityMonitor {
    config: AdvancedStabilityConfig,
    loss_history: VecDeque<f32>,
    gradient_history: VecDeque<f32>,
    lr_history: VecDeque<f32>,
    dynamics_history: Vec<TrainingDynamics>,
    predicted_anomalies: Vec<PredictiveAnomaly>,
    landscape_analyses: VecDeque<LossLandscapeAnalysis>,
    stability_scores: VecDeque<StabilityScore>,
    #[allow(dead_code)]
    recovery_effectiveness: HashMap<PreventiveAction, f32>,
    pattern_detector: PatternDetector,
}

impl AdvancedStabilityMonitor {
    pub fn new(config: AdvancedStabilityConfig) -> Self {
        Self {
            config,
            loss_history: VecDeque::new(),
            gradient_history: VecDeque::new(),
            lr_history: VecDeque::new(),
            dynamics_history: Vec::new(),
            predicted_anomalies: Vec::new(),
            landscape_analyses: VecDeque::new(),
            stability_scores: VecDeque::new(),
            recovery_effectiveness: HashMap::new(),
            pattern_detector: PatternDetector::new(),
        }
    }

    /// Analyze current training step and predict future stability
    pub fn analyze_step(
        &mut self,
        step: usize,
        loss: f32,
        gradient_norm: f32,
        learning_rate: f32,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<()> {
        // Update histories
        self.update_histories(loss, gradient_norm, learning_rate);

        // Analyze training dynamics
        if self.config.dynamics_analysis {
            let dynamics = self.analyze_training_dynamics()?;
            self.dynamics_history.push(dynamics);
        }

        // Perform loss landscape analysis
        if self.config.loss_landscape_monitoring {
            let landscape = self.analyze_loss_landscape(gradients)?;
            self.landscape_analyses.push_back(landscape);
            if self.landscape_analyses.len() > self.config.pattern_window_size {
                self.landscape_analyses.pop_front();
            }
        }

        // Compute stability score
        let stability = self.compute_stability_score()?;
        self.stability_scores.push_back(stability);
        if self.stability_scores.len() > self.config.pattern_window_size {
            self.stability_scores.pop_front();
        }

        // Predictive anomaly detection
        if self.config.predictive_detection {
            let predictions = self.predict_anomalies(step)?;
            self.predicted_anomalies.extend(predictions);
        }

        Ok(())
    }

    /// Get stability report with predictions and recommendations
    pub fn get_stability_report(&self) -> StabilityReport {
        let current_stability =
            self.stability_scores.back().map(|s| s.overall_score).unwrap_or(1.0);

        let immediate_risks: Vec<PredictiveAnomaly> = self
            .predicted_anomalies
            .iter()
            .filter(|anomaly| anomaly.time_to_occurrence <= 5)
            .cloned()
            .collect();

        let trend_analysis = self.analyze_stability_trend();

        StabilityReport {
            current_stability_score: current_stability,
            stability_trend: trend_analysis,
            immediate_risks,
            predicted_anomalies: self.predicted_anomalies.clone(),
            landscape_health: self.landscape_analyses.back().cloned(),
            recommendations: self.generate_recommendations(),
            confidence_level: self.compute_prediction_confidence(),
        }
    }

    /// Apply proactive recovery based on predictions
    pub fn apply_proactive_recovery(
        &mut self,
        trainer_params: &mut TrainerParameters,
    ) -> Result<Vec<PreventiveAction>> {
        if !self.config.proactive_recovery {
            return Ok(Vec::new());
        }

        let mut applied_actions = Vec::new();

        // Collect actions to apply first to avoid borrowing conflicts
        let mut actions_to_apply = Vec::new();

        for anomaly in &self.predicted_anomalies {
            if anomaly.confidence >= self.config.prediction_confidence_threshold
                && anomaly.time_to_occurrence <= 3
            {
                for action in &anomaly.preventive_actions {
                    if self.should_apply_action(action, trainer_params) {
                        actions_to_apply.push(action.clone());
                    }
                }
            }
        }

        // Apply the collected actions
        for action in actions_to_apply {
            self.apply_preventive_action(&action, trainer_params)?;
            applied_actions.push(action);
        }

        Ok(applied_actions)
    }

    fn update_histories(&mut self, loss: f32, gradient_norm: f32, learning_rate: f32) {
        self.loss_history.push_back(loss);
        self.gradient_history.push_back(gradient_norm);
        self.lr_history.push_back(learning_rate);

        let max_len = self.config.pattern_window_size;
        if self.loss_history.len() > max_len {
            self.loss_history.pop_front();
        }
        if self.gradient_history.len() > max_len {
            self.gradient_history.pop_front();
        }
        if self.lr_history.len() > max_len {
            self.lr_history.pop_front();
        }
    }

    fn analyze_training_dynamics(&self) -> Result<TrainingDynamics> {
        let loss_trend = self.compute_trend(&self.loss_history);
        let gradient_trend = self.compute_trend(&self.gradient_history);
        let lr_effectiveness = self.compute_lr_effectiveness();
        let convergence_velocity = self.compute_convergence_velocity();
        let oscillation_frequency = self.compute_oscillation_frequency();
        let phase_trajectory = self.compute_phase_trajectory();

        Ok(TrainingDynamics {
            loss_trend,
            gradient_trend,
            lr_effectiveness,
            convergence_velocity,
            oscillation_frequency,
            phase_trajectory,
        })
    }

    fn analyze_loss_landscape(
        &self,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<LossLandscapeAnalysis> {
        let local_curvature = self.estimate_local_curvature(gradients).unwrap_or_else(|e| {
            log::warn!("Failed to estimate local curvature: {}", e);
            0.1
        });

        let gradient_consistency =
            self.compute_gradient_consistency(gradients).unwrap_or_else(|e| {
                log::warn!("Failed to compute gradient consistency: {}", e);
                0.8
            });

        let escape_difficulty = self.estimate_escape_difficulty();
        let basin_stability = self.estimate_basin_stability();

        let saddle_point_prob =
            self.estimate_saddle_point_probability(gradients).unwrap_or_else(|e| {
                log::warn!("Failed to estimate saddle point probability: {}", e);
                0.2
            });

        Ok(LossLandscapeAnalysis {
            local_curvature,
            gradient_consistency,
            escape_difficulty,
            basin_stability,
            saddle_point_prob,
        })
    }

    fn compute_stability_score(&self) -> Result<StabilityScore> {
        let gradient_stability = self.compute_gradient_stability();
        let loss_stability = self.compute_loss_stability();
        let convergence_stability = self.compute_convergence_stability();
        let numerical_stability = self.compute_numerical_stability();

        let overall_score =
            (gradient_stability + loss_stability + convergence_stability + numerical_stability)
                / 4.0;

        let recommendations = self.generate_stability_recommendations(
            gradient_stability,
            loss_stability,
            convergence_stability,
            numerical_stability,
        );

        Ok(StabilityScore {
            overall_score,
            gradient_stability,
            loss_stability,
            convergence_stability,
            numerical_stability,
            recommendations,
        })
    }

    fn predict_anomalies(&self, current_step: usize) -> Result<Vec<PredictiveAnomaly>> {
        let mut predictions = Vec::new();

        // Predict gradient explosion
        if let Some(anomaly) = self.predict_gradient_explosion(current_step)? {
            predictions.push(anomaly);
        }

        // Predict training stagnation
        if let Some(anomaly) = self.predict_training_stagnation(current_step)? {
            predictions.push(anomaly);
        }

        // Predict numerical instability
        if let Some(anomaly) = self.predict_numerical_instability(current_step)? {
            predictions.push(anomaly);
        }

        // Predict oscillating loss
        if let Some(anomaly) = self.predict_oscillating_loss(current_step)? {
            predictions.push(anomaly);
        }

        Ok(predictions)
    }

    // Helper methods for trend analysis
    fn compute_trend(&self, history: &VecDeque<f32>) -> TrendDirection {
        if history.len() < 3 {
            return TrendDirection::Stable;
        }

        let recent: Vec<f32> = history.iter().rev().take(10).cloned().collect();
        let slope = self.compute_slope(&recent);
        let variance = self.compute_variance(&recent);

        if variance > 0.1 {
            TrendDirection::Oscillating
        } else if slope < -0.01 {
            TrendDirection::Decreasing
        } else if slope > 0.01 {
            TrendDirection::Increasing
        } else {
            TrendDirection::Stable
        }
    }

    fn compute_slope(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f32;
        let sum_x: f32 = (0..values.len()).map(|i| i as f32).sum();
        let sum_y: f32 = values.iter().sum();
        let sum_xy: f32 = values.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
        let sum_x2: f32 = (0..values.len()).map(|i| (i as f32).powi(2)).sum();

        (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    }

    fn compute_variance(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        let variance: f32 =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;

        variance
    }

    // Enhanced implementations for complex analysis methods
    fn compute_lr_effectiveness(&self) -> f32 {
        if self.loss_history.len() < 5 || self.lr_history.len() < 5 {
            return 0.5;
        }

        // Compute correlation between LR changes and loss improvements
        let mut lr_effectiveness_scores = Vec::new();

        for window in self
            .loss_history
            .iter()
            .zip(self.lr_history.iter())
            .collect::<Vec<_>>()
            .windows(3)
        {
            if let [(l1, lr1), (l2, lr2), (_l3, _lr3)] = window {
                let loss_improvement = (*l1 - *l2) / l1.max(1e-8f32);
                let lr_change = (*lr2 - *lr1) / lr1.max(1e-8f32);

                // Higher effectiveness if LR increases lead to loss decreases (and vice versa)
                if loss_improvement > 0.0 && lr_change > 0.0 {
                    lr_effectiveness_scores.push(0.8);
                } else if loss_improvement < 0.0 && lr_change < 0.0 {
                    lr_effectiveness_scores.push(0.6);
                } else {
                    lr_effectiveness_scores.push(0.3);
                }
            }
        }

        if lr_effectiveness_scores.is_empty() {
            0.5
        } else {
            lr_effectiveness_scores.iter().sum::<f32>() / lr_effectiveness_scores.len() as f32
        }
    }

    fn compute_convergence_velocity(&self) -> f32 {
        if self.loss_history.len() < 10 {
            return 0.0;
        }

        let recent_losses: Vec<f32> = self.loss_history.iter().rev().take(10).cloned().collect();
        let slope = self.compute_slope(&recent_losses);

        // Normalize slope to get velocity (more negative slope = faster convergence)

        if slope < 0.0 {
            (-slope * 100.0).min(1.0)
        } else {
            0.0
        }
    }

    fn compute_oscillation_frequency(&self) -> f32 {
        if self.loss_history.len() < 10 {
            return 0.0;
        }

        let recent_losses: Vec<f32> = self.loss_history.iter().rev().take(20).cloned().collect();
        let mut direction_changes = 0;

        for window in recent_losses.windows(3) {
            if (window[1] > window[0]) != (window[2] > window[1]) {
                direction_changes += 1;
            }
        }

        // Normalize by the number of possible direction changes
        direction_changes as f32 / (recent_losses.len() - 2).max(1) as f32
    }

    fn compute_phase_trajectory(&self) -> Vec<(f32, f32)> {
        self.loss_history
            .iter()
            .zip(self.gradient_history.iter())
            .map(|(&l, &g)| (l, g))
            .collect()
    }

    fn estimate_local_curvature(&self, gradients: &HashMap<String, Tensor>) -> Result<f32> {
        if gradients.is_empty() || self.gradient_history.len() < 3 {
            return Ok(0.1);
        }

        // Estimate curvature using finite differences of gradient norms
        let _current_norm = self.compute_total_gradient_norm(gradients)?;
        let recent_norms: Vec<f32> = self.gradient_history.iter().rev().take(3).cloned().collect();

        if recent_norms.len() >= 3 {
            // Second derivative approximation using finite differences
            let second_derivative = recent_norms[0] - 2.0 * recent_norms[1] + recent_norms[2];
            let curvature = second_derivative.abs() / (recent_norms[1].max(1e-8));
            Ok(curvature.min(10.0)) // Cap extreme curvature values
        } else {
            Ok(0.1)
        }
    }

    fn compute_gradient_consistency(&self, gradients: &HashMap<String, Tensor>) -> Result<f32> {
        if gradients.len() < 2 {
            return Ok(1.0);
        }

        // Compute consistency by checking gradient norm ratios across layers
        let mut norms = Vec::new();
        for tensor in gradients.values() {
            let data = tensor.data().unwrap_or_default();
            let norm = data.iter().map(|&x| x * x).sum::<f32>().sqrt();
            norms.push(norm);
        }

        if norms.is_empty() {
            return Ok(1.0);
        }

        let mean_norm = norms.iter().sum::<f32>() / norms.len() as f32;
        let variance =
            norms.iter().map(|&x| (x - mean_norm).powi(2)).sum::<f32>() / norms.len() as f32;
        let cv = variance.sqrt() / mean_norm.max(1e-8);

        // Higher consistency (lower CV) gets higher score
        Ok((1.0 / (1.0 + cv * 2.0)).clamp(0.0, 1.0))
    }

    fn estimate_escape_difficulty(&self) -> f32 {
        if self.loss_history.len() < 20 {
            return 0.3;
        }

        // Estimate difficulty based on local minima detection
        let recent_losses: Vec<f32> = self.loss_history.iter().rev().take(20).cloned().collect();
        let mut local_minima_count = 0;

        for window in recent_losses.windows(5) {
            if window[2] < window[0]
                && window[2] < window[1]
                && window[2] < window[3]
                && window[2] < window[4]
            {
                local_minima_count += 1;
            }
        }

        // More local minima suggest higher escape difficulty
        (local_minima_count as f32 / 5.0).min(1.0)
    }

    fn estimate_basin_stability(&self) -> f32 {
        if self.loss_history.len() < 10 {
            return 0.7;
        }

        // Estimate stability based on loss variance and trend
        let recent_losses: Vec<f32> = self.loss_history.iter().rev().take(10).cloned().collect();
        let variance = self.compute_variance(&recent_losses);
        let mean_loss = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
        let cv = variance.sqrt() / mean_loss.max(1e-8);

        // Lower variance indicates more stable basin
        (1.0 / (1.0 + cv * 3.0)).clamp(0.0, 1.0)
    }

    fn estimate_saddle_point_probability(
        &self,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<f32> {
        if gradients.is_empty() || self.gradient_history.len() < 5 {
            return Ok(0.2);
        }

        let current_grad_norm = self.compute_total_gradient_norm(gradients)?;

        // Saddle points typically have small gradients but high curvature
        let small_gradient = current_grad_norm < 0.01;
        let curvature = self.estimate_local_curvature(gradients)?;
        let high_curvature = curvature > 0.1;

        let probability = if small_gradient && high_curvature {
            0.8
        } else if small_gradient {
            0.4
        } else {
            0.1
        };

        Ok(probability)
    }

    fn compute_gradient_stability(&self) -> f32 {
        if self.gradient_history.len() < 5 {
            return 1.0;
        }
        let variance =
            self.compute_variance(&self.gradient_history.iter().cloned().collect::<Vec<_>>());
        (1.0 / (1.0 + variance)).clamp(0.0, 1.0)
    }

    fn compute_loss_stability(&self) -> f32 {
        if self.loss_history.len() < 5 {
            return 1.0;
        }
        let recent_losses: Vec<f32> = self.loss_history.iter().rev().take(10).cloned().collect();
        let slope = self.compute_slope(&recent_losses);
        if slope < 0.0 {
            0.9
        } else if slope < 0.01 {
            0.7
        } else {
            0.3
        }
    }

    fn compute_convergence_stability(&self) -> f32 {
        if self.loss_history.len() < 10 {
            return 0.8;
        }

        let convergence_velocity = self.compute_convergence_velocity();
        let oscillation_freq = self.compute_oscillation_frequency();

        // Balance between good convergence speed and low oscillation
        let velocity_score = convergence_velocity.min(0.5) * 2.0; // Normalize to 0-1
        let stability_score = (1.0 - oscillation_freq).max(0.0);

        (velocity_score * 0.6 + stability_score * 0.4).clamp(0.0, 1.0)
    }

    fn compute_numerical_stability(&self) -> f32 {
        if self.loss_history.is_empty() || self.gradient_history.is_empty() {
            return 0.9;
        }

        // Check for numerical issues in recent history
        let recent_losses: Vec<f32> = self.loss_history.iter().rev().take(10).cloned().collect();
        let recent_grads: Vec<f32> = self.gradient_history.iter().rev().take(10).cloned().collect();

        let loss_issues = recent_losses.iter().any(|&x| !x.is_finite());
        let grad_issues = recent_grads.iter().any(|&x| !x.is_finite());

        let extreme_values = recent_losses.iter().any(|&x| !(-1e6..=1e6).contains(&x))
            || recent_grads.iter().any(|&x| !(-1e6..=1e6).contains(&x));

        if loss_issues || grad_issues {
            0.0 // Critical numerical instability
        } else if extreme_values {
            0.3 // Potential instability
        } else {
            let max_loss = recent_losses.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let max_grad = recent_grads.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Penalize very large values
            let loss_penalty = if max_loss > 1000.0 { 0.3 } else { 0.0 };
            let grad_penalty = if max_grad > 100.0 { 0.2 } else { 0.0 };

            (1.0f32 - loss_penalty - grad_penalty).max(0.0f32)
        }
    }

    fn generate_stability_recommendations(
        &self,
        gs: f32,
        ls: f32,
        cs: f32,
        ns: f32,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if gs < 0.5 {
            recommendations.push(
                "Consider gradient clipping or normalization to improve gradient stability"
                    .to_string(),
            );
        }

        if ls < 0.5 {
            recommendations.push(
                "Loss appears unstable - consider reducing learning rate or adjusting optimizer"
                    .to_string(),
            );
        }

        if cs < 0.5 {
            recommendations.push("Poor convergence stability - consider learning rate scheduling or different optimizer".to_string());
        }

        if ns < 0.5 {
            recommendations.push("Numerical instability detected - check for NaN/Inf values and consider mixed precision".to_string());
        }

        let overall_score = (gs + ls + cs + ns) / 4.0;

        if overall_score < 0.3 {
            recommendations.push(
                "Critical stability issues - consider checkpoint rollback and parameter reset"
                    .to_string(),
            );
        } else if overall_score < 0.6 {
            recommendations.push("Moderate stability issues - monitor closely and consider conservative training settings".to_string());
        } else if recommendations.is_empty() {
            recommendations.push("Training stability is good - continue monitoring".to_string());
        }

        recommendations
    }

    fn analyze_stability_trend(&self) -> TrendDirection {
        let scores: Vec<f32> = self.stability_scores.iter().map(|s| s.overall_score).collect();
        self.compute_trend(&scores.into_iter().collect())
    }

    fn generate_recommendations(&self) -> Vec<String> {
        vec!["Continue monitoring training progress".to_string()]
    }

    fn compute_prediction_confidence(&self) -> f32 {
        // Base confidence on data quality and history length
        let history_quality = if self.loss_history.len() >= 20 { 0.9 } else { 0.5 };
        let data_quality = if self.loss_history.iter().all(|&x| x.is_finite()) { 0.9 } else { 0.3 };
        let trend_consistency = if self.dynamics_history.len() >= 3 { 0.8 } else { 0.4 };

        (history_quality * 0.4f32 + data_quality * 0.4f32 + trend_consistency * 0.2f32).min(1.0f32)
    }

    /// Helper method to compute total gradient norm across all tensors
    fn compute_total_gradient_norm(&self, gradients: &HashMap<String, Tensor>) -> Result<f32> {
        let mut total_norm_sq = 0.0f32;

        for tensor in gradients.values() {
            let data = tensor.data().map_err(|_| runtime_error("Failed to get tensor data"))?;
            let tensor_norm_sq: f32 = data.iter().map(|&x| x * x).sum();
            total_norm_sq += tensor_norm_sq;
        }

        Ok(total_norm_sq.sqrt())
    }

    /// Detect exponential growth in gradient sequence
    fn detect_exponential_growth(&self, values: &[f32]) -> bool {
        if values.len() < 5 {
            return false;
        }

        // Check if each value is consistently larger than the previous by a significant factor
        let mut growth_count = 0;
        for window in values.windows(2) {
            if window[0] > 0.0 && window[1] / window[0] > 1.5 {
                growth_count += 1;
            }
        }

        growth_count >= (values.len() - 1) / 2 // At least half show significant growth
    }

    /// Detect increasing variance in recent values
    fn detect_variance_increase(&self, values: &[f32]) -> bool {
        if values.len() < 8 {
            return false;
        }

        let mid_point = values.len() / 2;
        let early_half = &values[..mid_point];
        let recent_half = &values[mid_point..];

        let early_variance = self.compute_variance(early_half);
        let recent_variance = self.compute_variance(recent_half);

        recent_variance > early_variance * 2.0 // Recent variance is significantly higher
    }

    /// Detect lack of improvement over a threshold
    fn detect_no_improvement(&self, losses: &[f32], threshold: f32) -> bool {
        if losses.len() < 5 {
            return false;
        }

        let best_loss = losses.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let recent_loss = losses[0]; // Most recent (reversed order)

        // No improvement if recent loss is not significantly better than the best
        (best_loss - recent_loss) / best_loss.max(1e-8) < threshold
    }

    /// Compute oscillation amplitude
    fn compute_oscillation_amplitude(&self) -> f32 {
        if self.loss_history.len() < 10 {
            return 0.0;
        }

        let recent_losses: Vec<f32> = self.loss_history.iter().rev().take(10).cloned().collect();
        let max_loss = recent_losses.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_loss = recent_losses.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let mean_loss = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;

        if mean_loss > 0.0 {
            (max_loss - min_loss) / mean_loss
        } else {
            0.0
        }
    }

    fn predict_gradient_explosion(&self, current_step: usize) -> Result<Option<PredictiveAnomaly>> {
        if self.gradient_history.len() < 5 {
            return Ok(None);
        }

        let recent_grads: Vec<f32> = self.gradient_history.iter().rev().take(10).cloned().collect();

        // Multiple indicators for gradient explosion
        let trend = self.compute_trend(&self.gradient_history);
        let exponential_growth = self.detect_exponential_growth(&recent_grads);
        let variance_increase = self.detect_variance_increase(&recent_grads);

        let base_confidence = match trend {
            TrendDirection::Increasing => 0.6,
            TrendDirection::Diverging => 0.9,
            _ => 0.0,
        };

        let growth_factor = if exponential_growth { 0.3f32 } else { 0.0f32 };
        let variance_factor = if variance_increase { 0.2f32 } else { 0.0f32 };

        let confidence = (base_confidence + growth_factor + variance_factor).min(1.0f32);

        if confidence >= self.config.prediction_confidence_threshold {
            let time_to_occurrence = if exponential_growth { 2 } else { 5 };
            let risk_level = if confidence > 0.8 { RiskLevel::Critical } else { RiskLevel::High };

            return Ok(Some(PredictiveAnomaly {
                predicted_step: current_step + time_to_occurrence,
                anomaly_type: PredictedAnomalyType::GradientExplosion,
                confidence,
                time_to_occurrence,
                preventive_actions: vec![
                    PreventiveAction::ReduceLearningRate {
                        factor: if confidence > 0.8 { 0.1 } else { 0.5 },
                    },
                    PreventiveAction::IncreaseGradientClipping { new_threshold: 1.0 },
                    PreventiveAction::TriggerEarlyCheckpoint,
                ],
                risk_level,
            }));
        }

        Ok(None)
    }

    fn predict_training_stagnation(
        &self,
        current_step: usize,
    ) -> Result<Option<PredictiveAnomaly>> {
        if self.loss_history.len() < 20 {
            return Ok(None);
        }

        let recent_losses: Vec<f32> = self.loss_history.iter().rev().take(15).cloned().collect();
        let variance = self.compute_variance(&recent_losses);
        let slope = self.compute_slope(&recent_losses);

        // Multiple stagnation indicators
        let low_variance = variance < 1e-6;
        let flat_slope = slope.abs() < 1e-5;
        let no_improvement = self.detect_no_improvement(&recent_losses, 0.001);

        let stagnation_indicators =
            [low_variance, flat_slope, no_improvement].iter().filter(|&&x| x).count();

        if stagnation_indicators >= 2 {
            let confidence = match stagnation_indicators {
                3 => 0.95,
                2 => 0.7,
                _ => 0.5,
            };

            if confidence >= self.config.prediction_confidence_threshold {
                return Ok(Some(PredictiveAnomaly {
                    predicted_step: current_step + 10,
                    anomaly_type: PredictedAnomalyType::TrainingStagnation,
                    confidence,
                    time_to_occurrence: 10,
                    preventive_actions: vec![
                        PreventiveAction::AdjustOptimizer {
                            suggested_params: [
                                ("momentum".to_string(), 0.9),
                                ("learning_rate_multiplier".to_string(), 1.5),
                            ]
                            .into_iter()
                            .collect(),
                        },
                        PreventiveAction::EnableNoise { noise_level: 0.01 },
                        PreventiveAction::AdjustWarmupSchedule,
                    ],
                    risk_level: if confidence > 0.8 { RiskLevel::High } else { RiskLevel::Medium },
                }));
            }
        }

        Ok(None)
    }

    fn predict_numerical_instability(
        &self,
        current_step: usize,
    ) -> Result<Option<PredictiveAnomaly>> {
        if self.loss_history.len() < 5 {
            return Ok(None);
        }

        let recent_loss = self.loss_history.back().unwrap_or(&1.0);
        if recent_loss.is_nan() || recent_loss.is_infinite() || *recent_loss > 1e6 {
            return Ok(Some(PredictiveAnomaly {
                predicted_step: current_step + 1,
                anomaly_type: PredictedAnomalyType::NumericalInstability,
                confidence: 0.95,
                time_to_occurrence: 1,
                preventive_actions: vec![
                    PreventiveAction::ReduceLearningRate { factor: 0.1 },
                    PreventiveAction::TriggerEarlyCheckpoint,
                ],
                risk_level: RiskLevel::Critical,
            }));
        }

        Ok(None)
    }

    fn predict_oscillating_loss(&self, current_step: usize) -> Result<Option<PredictiveAnomaly>> {
        if self.loss_history.len() < 15 {
            return Ok(None);
        }

        let oscillation_freq = self.compute_oscillation_frequency();
        let amplitude = self.compute_oscillation_amplitude();

        // Oscillation severity based on frequency and amplitude
        let severity_score = oscillation_freq * amplitude;

        if oscillation_freq > 0.3 || severity_score > 0.2 {
            let confidence = (oscillation_freq * 2.0 + severity_score).min(1.0);

            if confidence >= self.config.prediction_confidence_threshold {
                return Ok(Some(PredictiveAnomaly {
                    predicted_step: current_step + 5,
                    anomaly_type: PredictedAnomalyType::OscillatingLoss,
                    confidence,
                    time_to_occurrence: 5,
                    preventive_actions: vec![
                        PreventiveAction::ReduceLearningRate {
                            factor: if severity_score > 0.5 { 0.5 } else { 0.8 },
                        },
                        PreventiveAction::AdjustWarmupSchedule,
                        PreventiveAction::EnableNoise { noise_level: 0.005 }, // Small noise to break oscillations
                        PreventiveAction::ModifyBatchSize { new_size: 64 }, // Larger batch for stability
                    ],
                    risk_level: if severity_score > 0.5 {
                        RiskLevel::High
                    } else {
                        RiskLevel::Medium
                    },
                }));
            }
        }

        Ok(None)
    }

    fn should_apply_action(&self, _action: &PreventiveAction, _params: &TrainerParameters) -> bool {
        true // Simplified logic
    }

    fn apply_preventive_action(
        &mut self,
        action: &PreventiveAction,
        params: &mut TrainerParameters,
    ) -> Result<()> {
        match action {
            PreventiveAction::ReduceLearningRate { factor } => {
                params.learning_rate *= factor;
            },
            PreventiveAction::IncreaseGradientClipping { new_threshold } => {
                params.gradient_clip_threshold = *new_threshold;
            },
            PreventiveAction::ModifyBatchSize { new_size } => {
                params.batch_size = *new_size;
            },
            _ => {
                // Other actions would be implemented based on trainer interface
            },
        }
        Ok(())
    }
}

/// Pattern detector for complex training dynamics
pub struct PatternDetector {
    #[allow(dead_code)]
    pattern_library: HashMap<String, Pattern>,
}

impl Default for PatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternDetector {
    pub fn new() -> Self {
        Self {
            pattern_library: HashMap::new(),
        }
    }

    pub fn detect_patterns(&self, _dynamics: &TrainingDynamics) -> Vec<DetectedPattern> {
        Vec::new() // Placeholder
    }
}

#[derive(Debug, Clone)]
pub struct Pattern {
    pub name: String,
    pub description: String,
    pub indicators: Vec<PatternIndicator>,
}

#[derive(Debug, Clone)]
pub struct PatternIndicator {
    pub metric: String,
    pub condition: String,
    pub threshold: f32,
}

#[derive(Debug, Clone)]
pub struct DetectedPattern {
    pub pattern: Pattern,
    pub confidence: f32,
    pub severity: RiskLevel,
}

/// Trainer parameters that can be modified by recovery actions
#[derive(Debug, Clone)]
pub struct TrainerParameters {
    pub learning_rate: f32,
    pub gradient_clip_threshold: f32,
    pub batch_size: usize,
    pub optimizer_params: HashMap<String, f32>,
}

/// Comprehensive stability report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityReport {
    pub current_stability_score: f32,
    pub stability_trend: TrendDirection,
    pub immediate_risks: Vec<PredictiveAnomaly>,
    pub predicted_anomalies: Vec<PredictiveAnomaly>,
    pub landscape_health: Option<LossLandscapeAnalysis>,
    pub recommendations: Vec<String>,
    pub confidence_level: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_stability_monitor_creation() {
        let config = AdvancedStabilityConfig::default();
        let monitor = AdvancedStabilityMonitor::new(config);
        assert!(monitor.loss_history.is_empty());
        assert!(monitor.predicted_anomalies.is_empty());
    }

    #[test]
    fn test_stability_analysis() {
        let config = AdvancedStabilityConfig::default();
        let mut monitor = AdvancedStabilityMonitor::new(config);
        let gradients = HashMap::new();

        let result = monitor.analyze_step(0, 1.0, 0.5, 0.001, &gradients);
        assert!(result.is_ok());
    }

    #[test]
    fn test_trend_computation() {
        let config = AdvancedStabilityConfig::default();
        let monitor = AdvancedStabilityMonitor::new(config);

        let values: VecDeque<f32> = vec![1.0, 0.9, 0.8, 0.7, 0.6].into();
        let trend = monitor.compute_trend(&values);
        assert!(matches!(trend, TrendDirection::Decreasing));
    }

    #[test]
    fn test_stability_report_generation() {
        let config = AdvancedStabilityConfig::default();
        let monitor = AdvancedStabilityMonitor::new(config);

        let report = monitor.get_stability_report();
        assert!(report.current_stability_score >= 0.0);
        assert!(report.confidence_level >= 0.0);
    }
}
