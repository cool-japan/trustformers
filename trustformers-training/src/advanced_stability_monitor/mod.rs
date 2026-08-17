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
pub struct AdvancedStabilityMonitor {
    config: AdvancedStabilityConfig,
    loss_history: VecDeque<f32>,
    gradient_history: VecDeque<f32>,
    lr_history: VecDeque<f32>,
    dynamics_history: Vec<TrainingDynamics>,
    predicted_anomalies: Vec<PredictiveAnomaly>,
    landscape_analyses: VecDeque<LossLandscapeAnalysis>,
    stability_scores: VecDeque<StabilityScore>,
    recovery_effectiveness: HashMap<PreventiveAction, f32>,
    pattern_detector: PatternDetector,
    /// Actions the monitor decided on but cannot carry out itself, because they are
    /// instructions to the training loop rather than edits to [`TrainerParameters`].
    pending_signals: Vec<PreventiveAction>,
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
            pending_signals: Vec::new(),
        }
    }

    /// Drain the actions that the training loop itself must carry out.
    ///
    /// [`PreventiveAction::TriggerEarlyCheckpoint`],
    /// [`PreventiveAction::AdjustWarmupSchedule`] and
    /// [`PreventiveAction::ResetAccumulatedGradients`] cannot be expressed as a change to
    /// [`TrainerParameters`] — they are instructions to the loop. They are queued here
    /// instead of being silently dropped; a caller that never drains this queue simply does
    /// not act on them, but it is never told they were applied when they were not.
    pub fn take_pending_signals(&mut self) -> Vec<PreventiveAction> {
        std::mem::take(&mut self.pending_signals)
    }

    /// Actions currently queued for the training loop, without draining them.
    pub fn pending_signals(&self) -> &[PreventiveAction] {
        &self.pending_signals
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

        // Take the 10 most recent values and restore chronological order
        let mut recent: Vec<f32> = history.iter().rev().take(10).cloned().collect();
        recent.reverse(); // Restore chronological order for slope computation
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

    /// Would `action` actually change `params` in the intended direction?
    ///
    /// This used to be `true` unconditionally, so the monitor happily "applied" a learning
    /// rate multiplier of `1.0`, a clipping threshold looser than the current one, or a
    /// batch size identical to the one already in use, and reported each of them as an
    /// applied recovery. Each arm below rejects the cases that would be a no-op or would
    /// push a parameter out of its valid range.
    fn should_apply_action(&self, action: &PreventiveAction, params: &TrainerParameters) -> bool {
        match action {
            PreventiveAction::ReduceLearningRate { factor } => {
                factor.is_finite()
                    && *factor > 0.0
                    && *factor < 1.0
                    && params.learning_rate * factor >= MIN_PREVENTIVE_LEARNING_RATE
            },
            // "Increase clipping" means clip harder, i.e. a *lower* threshold. A threshold
            // of zero or below would zero every gradient.
            PreventiveAction::IncreaseGradientClipping { new_threshold } => {
                new_threshold.is_finite()
                    && *new_threshold > 0.0
                    && (params.gradient_clip_threshold <= 0.0
                        || *new_threshold < params.gradient_clip_threshold)
            },
            PreventiveAction::ModifyBatchSize { new_size } => {
                *new_size > 0
                    && *new_size <= MAX_PREVENTIVE_BATCH_SIZE
                    && *new_size != params.batch_size
            },
            PreventiveAction::AdjustOptimizer { suggested_params } => {
                suggested_params.iter().any(|(key, value)| {
                    value.is_finite()
                        && params
                            .optimizer_params
                            .get(key)
                            .is_none_or(|current| (current - value).abs() > OPTIMIZER_PARAM_EPSILON)
                })
            },
            PreventiveAction::EnableNoise { noise_level } => {
                noise_level.is_finite()
                    && *noise_level > 0.0
                    && params.optimizer_params.get(GRADIENT_NOISE_PARAM).is_none_or(|current| {
                        (current - noise_level).abs() > OPTIMIZER_PARAM_EPSILON
                    })
            },
            // Instructions to the training loop; always meaningful to raise.
            PreventiveAction::TriggerEarlyCheckpoint
            | PreventiveAction::AdjustWarmupSchedule
            | PreventiveAction::ResetAccumulatedGradients => true,
        }
    }

    /// Carry out `action`.
    ///
    /// Actions that map onto [`TrainerParameters`] are applied here. The three that do not
    /// are queued for [`AdvancedStabilityMonitor::take_pending_signals`] instead of falling
    /// into a silent catch-all arm, which previously reported them as applied while doing
    /// nothing at all.
    fn apply_preventive_action(
        &mut self,
        action: &PreventiveAction,
        params: &mut TrainerParameters,
    ) -> Result<()> {
        match action {
            PreventiveAction::ReduceLearningRate { factor } => {
                params.learning_rate =
                    (params.learning_rate * factor).max(MIN_PREVENTIVE_LEARNING_RATE);
            },
            PreventiveAction::IncreaseGradientClipping { new_threshold } => {
                params.gradient_clip_threshold = *new_threshold;
            },
            PreventiveAction::ModifyBatchSize { new_size } => {
                params.batch_size = *new_size;
            },
            PreventiveAction::AdjustOptimizer { suggested_params } => {
                for (key, value) in suggested_params {
                    if value.is_finite() {
                        params.optimizer_params.insert(key.clone(), *value);
                    }
                }
            },
            PreventiveAction::EnableNoise { noise_level } => {
                params.optimizer_params.insert(GRADIENT_NOISE_PARAM.to_string(), *noise_level);
            },
            PreventiveAction::TriggerEarlyCheckpoint
            | PreventiveAction::AdjustWarmupSchedule
            | PreventiveAction::ResetAccumulatedGradients => {
                self.pending_signals.push(action.clone());
            },
        }
        Ok(())
    }
}

/// Floor a [`PreventiveAction::ReduceLearningRate`] may not push the learning rate below.
const MIN_PREVENTIVE_LEARNING_RATE: f32 = 1e-8;

/// Ceiling for a [`PreventiveAction::ModifyBatchSize`] suggestion.
const MAX_PREVENTIVE_BATCH_SIZE: usize = 65_536;

/// Two optimizer hyper-parameters closer than this count as unchanged.
const OPTIMIZER_PARAM_EPSILON: f32 = 1e-9;

/// Key under which [`PreventiveAction::EnableNoise`] records its noise standard deviation in
/// [`TrainerParameters::optimizer_params`].
pub const GRADIENT_NOISE_PARAM: &str = "gradient_noise_std";

/// Instability [`Pattern`]s and the detector that evaluates them.
mod pattern;
pub use pattern::{DetectedPattern, Pattern, PatternDetector, PatternIndicator};

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

    // ── Additional tests ──────────────────────────────────────────────────────

    #[test]
    fn test_config_default_prediction_horizon_positive() {
        let cfg = AdvancedStabilityConfig::default();
        assert!(cfg.prediction_horizon > 0);
    }

    #[test]
    fn test_config_default_pattern_window_positive() {
        let cfg = AdvancedStabilityConfig::default();
        assert!(cfg.pattern_window_size > 0);
    }

    #[test]
    fn test_config_default_confidence_threshold_range() {
        let cfg = AdvancedStabilityConfig::default();
        assert!(cfg.prediction_confidence_threshold >= 0.0);
        assert!(cfg.prediction_confidence_threshold <= 1.0);
    }

    #[test]
    fn test_config_default_stability_threshold_range() {
        let cfg = AdvancedStabilityConfig::default();
        assert!(cfg.stability_threshold >= 0.0);
        assert!(cfg.stability_threshold <= 1.0);
    }

    #[test]
    fn test_config_predictive_detection_enabled_default() {
        let cfg = AdvancedStabilityConfig::default();
        assert!(
            cfg.predictive_detection,
            "predictive_detection should be true by default"
        );
    }

    #[test]
    fn test_config_adaptive_recovery_default_true() {
        let cfg = AdvancedStabilityConfig::default();
        assert!(cfg.adaptive_recovery);
    }

    #[test]
    fn test_trend_direction_increasing() {
        let cfg = AdvancedStabilityConfig::default();
        let monitor = AdvancedStabilityMonitor::new(cfg);
        let values: VecDeque<f32> = vec![0.6, 0.7, 0.8, 0.9, 1.0].into();
        let trend = monitor.compute_trend(&values);
        assert!(matches!(trend, TrendDirection::Increasing));
    }

    #[test]
    fn test_trend_direction_stable() {
        let cfg = AdvancedStabilityConfig::default();
        let monitor = AdvancedStabilityMonitor::new(cfg);
        let values: VecDeque<f32> = vec![0.5, 0.5, 0.5, 0.5, 0.5].into();
        let trend = monitor.compute_trend(&values);
        assert!(matches!(trend, TrendDirection::Stable));
    }

    #[test]
    fn test_risk_level_variants_exist() {
        let _ = RiskLevel::Low;
        let _ = RiskLevel::Medium;
        let _ = RiskLevel::High;
        let _ = RiskLevel::Critical;
    }

    #[test]
    fn test_predicted_anomaly_type_variants() {
        let _ = PredictedAnomalyType::GradientExplosion;
        let _ = PredictedAnomalyType::GradientVanishing;
        let _ = PredictedAnomalyType::TrainingStagnation;
        let _ = PredictedAnomalyType::ConvergenceFailure;
    }

    #[test]
    fn test_analyze_step_with_gradient_norms() {
        let cfg = AdvancedStabilityConfig::default();
        let mut monitor = AdvancedStabilityMonitor::new(cfg);
        let gradients: HashMap<String, Tensor> = HashMap::new();
        let result = monitor.analyze_step(1, 0.8, 0.4, 0.001, &gradients);
        assert!(result.is_ok());
    }

    #[test]
    fn test_analyze_multiple_steps_accumulates_history() {
        let cfg = AdvancedStabilityConfig::default();
        let mut monitor = AdvancedStabilityMonitor::new(cfg);
        let gradients: HashMap<String, Tensor> = HashMap::new();
        let mut s = 42u64;
        for step in 0..5usize {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let loss = (s % 1000) as f32 / 1000.0 + 0.1;
            let _ = monitor.analyze_step(step, loss, 0.5, 0.001, &gradients);
        }
        assert!(
            !monitor.loss_history.is_empty(),
            "loss history should be populated"
        );
    }

    #[test]
    fn test_stability_report_score_non_negative() {
        let cfg = AdvancedStabilityConfig::default();
        let mut monitor = AdvancedStabilityMonitor::new(cfg);
        let gradients: HashMap<String, Tensor> = HashMap::new();
        for step in 0..3 {
            let _ = monitor.analyze_step(step, 1.0 - step as f32 * 0.1, 0.5, 0.001, &gradients);
        }
        let report = monitor.get_stability_report();
        assert!(report.current_stability_score >= 0.0);
        assert!(report.current_stability_score <= 1.0 + f32::EPSILON);
    }

    #[test]
    fn test_stability_report_confidence_bounded() {
        let cfg = AdvancedStabilityConfig::default();
        let monitor = AdvancedStabilityMonitor::new(cfg);
        let report = monitor.get_stability_report();
        assert!(report.confidence_level >= 0.0 && report.confidence_level <= 1.0);
    }

    #[test]
    fn test_stability_score_struct_bounds() {
        let score = StabilityScore {
            overall_score: 0.75,
            gradient_stability: 0.8,
            loss_stability: 0.7,
            convergence_stability: 0.75,
            numerical_stability: 0.8,
            recommendations: vec![],
        };
        assert!(score.overall_score >= 0.0 && score.overall_score <= 1.0);
        assert!(score.gradient_stability >= 0.0);
    }

    #[test]
    fn test_training_dynamics_struct_creation() {
        let td = TrainingDynamics {
            loss_trend: TrendDirection::Decreasing,
            gradient_trend: TrendDirection::Stable,
            lr_effectiveness: 0.8,
            convergence_velocity: 0.5,
            oscillation_frequency: 0.1,
            phase_trajectory: vec![(1.0, 0.5), (0.9, 0.4)],
        };
        assert!(!td.phase_trajectory.is_empty());
        assert!(matches!(td.loss_trend, TrendDirection::Decreasing));
    }

    #[test]
    fn test_analyze_step_extreme_loss_nan_handled() {
        let cfg = AdvancedStabilityConfig::default();
        let mut monitor = AdvancedStabilityMonitor::new(cfg);
        let gradients = HashMap::new();
        // NaN loss should return an error or handle gracefully
        let result = monitor.analyze_step(0, f32::NAN, 0.5, 0.001, &gradients);
        // Either error or ok — just should not panic
        let _ = result;
    }

    // ── PatternDetector actually detects ─────────────────────────────────────

    fn dynamics(
        loss_trend: TrendDirection,
        gradient_trend: TrendDirection,
        lr_effectiveness: f32,
        convergence_velocity: f32,
        oscillation_frequency: f32,
        phase_trajectory: Vec<(f32, f32)>,
    ) -> TrainingDynamics {
        TrainingDynamics {
            loss_trend,
            gradient_trend,
            lr_effectiveness,
            convergence_velocity,
            oscillation_frequency,
            phase_trajectory,
        }
    }

    fn healthy_dynamics() -> TrainingDynamics {
        dynamics(
            TrendDirection::Decreasing,
            TrendDirection::Decreasing,
            0.9,
            0.5,
            0.05,
            vec![(1.0, 1.0), (0.9, 0.95), (0.8, 0.9)],
        )
    }

    #[test]
    fn test_pattern_library_is_not_empty() {
        // Regression: `pattern_library` was initialised empty and nothing ever inserted.
        let detector = PatternDetector::new();
        assert!(
            detector.pattern_count() >= 7,
            "the built-in library must be loaded, got {} patterns",
            detector.pattern_count()
        );
    }

    #[test]
    fn test_detect_patterns_reports_loss_divergence() {
        // Regression: `detect_patterns` returned `Vec::new()` for every input.
        let detector = PatternDetector::new();
        let diverging = dynamics(
            TrendDirection::Diverging,
            TrendDirection::Increasing,
            0.05,
            0.0,
            0.1,
            vec![(1.0, 10.0), (50.0, 500.0)],
        );
        let detected = detector.detect_patterns(&diverging);
        assert!(
            !detected.is_empty(),
            "a diverging run must report at least one pattern"
        );
        assert!(
            detected.iter().any(|d| d.pattern.name == "loss_divergence"),
            "expected loss_divergence among {:?}",
            detected.iter().map(|d| &d.pattern.name).collect::<Vec<_>>()
        );
        let divergence = detected
            .iter()
            .find(|d| d.pattern.name == "loss_divergence")
            .expect("loss_divergence must be present");
        assert!(
            (divergence.confidence - 1.0).abs() < 1e-6,
            "both indicators hold, so confidence should be 1.0, got {}",
            divergence.confidence
        );
        assert!(matches!(divergence.severity, RiskLevel::Critical));
    }

    #[test]
    fn test_detect_patterns_is_quiet_on_a_healthy_run() {
        let detector = PatternDetector::new();
        let detected = detector.detect_patterns(&healthy_dynamics());
        assert!(
            detected.is_empty(),
            "a healthy run should trigger nothing, got {:?}",
            detected.iter().map(|d| &d.pattern.name).collect::<Vec<_>>()
        );
    }

    fn trainer_params() -> TrainerParameters {
        TrainerParameters {
            learning_rate: 1e-3,
            gradient_clip_threshold: 5.0,
            batch_size: 32,
            optimizer_params: HashMap::new(),
        }
    }

    #[test]
    fn test_should_apply_action_rejects_no_ops_and_unsafe_values() {
        // Regression: this returned `true` for everything, so a factor of 1.0 or a looser
        // clipping threshold was "applied" and reported as a recovery.
        let monitor = AdvancedStabilityMonitor::new(AdvancedStabilityConfig::default());
        let params = trainer_params();

        assert!(monitor.should_apply_action(
            &PreventiveAction::ReduceLearningRate { factor: 0.5 },
            &params
        ));
        for factor in [1.0f32, 1.5, 0.0, -0.5, f32::NAN] {
            assert!(
                !monitor
                    .should_apply_action(&PreventiveAction::ReduceLearningRate { factor }, &params),
                "factor {factor} is not a reduction"
            );
        }

        assert!(monitor.should_apply_action(
            &PreventiveAction::IncreaseGradientClipping { new_threshold: 1.0 },
            &params
        ));
        for threshold in [5.0f32, 9.0, 0.0, -1.0] {
            assert!(
                !monitor.should_apply_action(
                    &PreventiveAction::IncreaseGradientClipping {
                        new_threshold: threshold
                    },
                    &params
                ),
                "threshold {threshold} does not tighten clipping from 5.0"
            );
        }

        assert!(monitor
            .should_apply_action(&PreventiveAction::ModifyBatchSize { new_size: 64 }, &params));
        assert!(!monitor
            .should_apply_action(&PreventiveAction::ModifyBatchSize { new_size: 32 }, &params));
        assert!(!monitor
            .should_apply_action(&PreventiveAction::ModifyBatchSize { new_size: 0 }, &params));
    }

    #[test]
    fn test_every_preventive_action_is_carried_out_or_queued() {
        // Regression: five of the eight variants fell into a `_ => {}` arm and were reported
        // as applied while changing nothing.
        let mut monitor = AdvancedStabilityMonitor::new(AdvancedStabilityConfig::default());
        let mut params = trainer_params();

        monitor
            .apply_preventive_action(
                &PreventiveAction::AdjustOptimizer {
                    suggested_params: HashMap::from([("beta1".to_string(), 0.85f32)]),
                },
                &mut params,
            )
            .expect("apply");
        assert_eq!(params.optimizer_params.get("beta1"), Some(&0.85));

        monitor
            .apply_preventive_action(
                &PreventiveAction::EnableNoise { noise_level: 0.01 },
                &mut params,
            )
            .expect("apply");
        assert_eq!(
            params.optimizer_params.get(GRADIENT_NOISE_PARAM),
            Some(&0.01)
        );

        monitor
            .apply_preventive_action(
                &PreventiveAction::ReduceLearningRate { factor: 0.5 },
                &mut params,
            )
            .expect("apply");
        assert!((params.learning_rate - 5e-4).abs() < 1e-12);

        for signal in [
            PreventiveAction::TriggerEarlyCheckpoint,
            PreventiveAction::AdjustWarmupSchedule,
            PreventiveAction::ResetAccumulatedGradients,
        ] {
            monitor.apply_preventive_action(&signal, &mut params).expect("apply");
        }
        assert_eq!(monitor.pending_signals().len(), 3);
        let drained = monitor.take_pending_signals();
        assert_eq!(drained.len(), 3);
        assert!(
            monitor.pending_signals().is_empty(),
            "draining must empty the queue"
        );

        // An already-set optimizer parameter is not re-applied.
        assert!(!monitor.should_apply_action(
            &PreventiveAction::AdjustOptimizer {
                suggested_params: HashMap::from([("beta1".to_string(), 0.85f32)]),
            },
            &params
        ));
        assert!(!monitor.should_apply_action(
            &PreventiveAction::EnableNoise { noise_level: 0.01 },
            &params
        ));
    }

    #[test]
    fn test_reduce_learning_rate_respects_the_floor() {
        let mut monitor = AdvancedStabilityMonitor::new(AdvancedStabilityConfig::default());
        let mut params = trainer_params();
        params.learning_rate = 1e-8;
        // Reducing further would fall under the floor, so the action is not applicable.
        assert!(!monitor.should_apply_action(
            &PreventiveAction::ReduceLearningRate { factor: 0.5 },
            &params
        ));
        // Applying it anyway still cannot drive the learning rate to zero.
        monitor
            .apply_preventive_action(
                &PreventiveAction::ReduceLearningRate { factor: 1e-6 },
                &mut params,
            )
            .expect("apply");
        assert!(params.learning_rate >= 1e-8);
    }

    #[test]
    fn test_builtin_patterns_are_conjunctions() {
        // Regression: the detector used a global "half the indicators is enough" rule, so a
        // healthy run whose gradients were merely *shrinking* satisfied one of the two
        // `gradient_norm_collapse` indicators and the pattern fired at confidence 0.5.
        let healthy = healthy_dynamics();
        assert!(matches!(healthy.gradient_trend, TrendDirection::Decreasing));

        let detector = PatternDetector::new();
        for (pattern, _) in PatternDetector::builtin_patterns() {
            assert_eq!(
                pattern.min_indicators,
                pattern.indicators.len(),
                "built-in pattern '{}' must require all of its indicators",
                pattern.name
            );
        }
        assert!(
            !detector
                .detect_patterns(&healthy)
                .iter()
                .any(|d| d.pattern.name == "gradient_norm_collapse"),
            "a decreasing gradient trend alone is not a collapse"
        );

        // The same two indicators registered as a disjunction *do* fire on that run —
        // proving the difference comes from `min_indicators`, not from the indicators.
        let mut permissive = PatternDetector::empty();
        permissive
            .register_pattern(
                Pattern::any_of(
                    "half_collapse",
                    "either half of the collapse rule",
                    vec![
                        PatternIndicator::new("max_gradient_norm", "less_than", 1e-4),
                        PatternIndicator::new("gradient_trend_decreasing", "greater_or_equal", 1.0),
                    ],
                ),
                RiskLevel::High,
            )
            .expect("registration should succeed");
        let hits = permissive.detect_patterns(&healthy);
        assert_eq!(hits.len(), 1);
        assert!(
            (hits[0].confidence - 0.5).abs() < 1e-6,
            "one of two indicators held, so confidence is 0.5, got {}",
            hits[0].confidence
        );
    }

    #[test]
    fn test_detect_patterns_output_varies_with_input() {
        // The core regression: the old detector returned the same empty vector for every
        // possible dynamics value.
        let detector = PatternDetector::new();
        let healthy = detector.detect_patterns(&healthy_dynamics());
        let spiking = detector.detect_patterns(&dynamics(
            TrendDirection::Increasing,
            TrendDirection::Stable,
            0.5,
            0.2,
            0.1,
            // median loss 1.0, max 20.0 → spike ratio 20 > 3
            vec![(1.0, 1.0), (1.0, 1.0), (20.0, 1.0)],
        ));
        assert_ne!(
            healthy.len(),
            spiking.len(),
            "detection must depend on the dynamics"
        );
        assert!(spiking.iter().any(|d| d.pattern.name == "loss_spike"));
    }

    #[test]
    fn test_detect_patterns_finds_gradient_explosion_and_collapse() {
        let detector = PatternDetector::new();

        let exploding = detector.detect_patterns(&dynamics(
            TrendDirection::Increasing,
            TrendDirection::Increasing,
            0.4,
            0.1,
            0.1,
            vec![(1.0, 5.0), (1.2, 5000.0)],
        ));
        assert!(exploding.iter().any(|d| d.pattern.name == "gradient_explosion"));

        let collapsed = detector.detect_patterns(&dynamics(
            TrendDirection::Stable,
            TrendDirection::Decreasing,
            0.4,
            0.5,
            0.1,
            vec![(1.0, 1e-7), (1.0, 1e-8)],
        ));
        assert!(collapsed.iter().any(|d| d.pattern.name == "gradient_norm_collapse"));
    }

    #[test]
    fn test_detect_patterns_finds_stagnation_and_lr_divergence() {
        let detector = PatternDetector::new();

        let stalled = detector.detect_patterns(&dynamics(
            TrendDirection::Stable,
            TrendDirection::Stable,
            0.5,
            1e-6,
            0.01,
            vec![(0.5, 0.1), (0.5, 0.1)],
        ));
        assert!(stalled.iter().any(|d| d.pattern.name == "training_stagnation"));

        let lr_dead = detector.detect_patterns(&dynamics(
            TrendDirection::Increasing,
            TrendDirection::Stable,
            0.01,
            0.0,
            0.1,
            vec![(0.5, 0.1), (0.6, 0.1)],
        ));
        assert!(lr_dead.iter().any(|d| d.pattern.name == "learning_rate_divergence"));
    }

    #[test]
    fn test_detect_patterns_sorted_by_descending_confidence() {
        let detector = PatternDetector::new();
        let detected = detector.detect_patterns(&dynamics(
            TrendDirection::Diverging,
            TrendDirection::Increasing,
            0.01,
            0.0,
            0.9,
            vec![(1.0, 1.0), (1.0, 1.0), (500.0, 5000.0)],
        ));
        assert!(detected.len() >= 2, "expected several concurrent patterns");
        for pair in detected.windows(2) {
            assert!(
                pair[0].confidence >= pair[1].confidence,
                "results must be ordered by descending confidence"
            );
        }
    }

    #[test]
    fn test_register_pattern_rejects_unknown_metric_and_condition() {
        let mut detector = PatternDetector::empty();
        assert_eq!(detector.pattern_count(), 0);

        let bad_metric = Pattern::all_of(
            "typo",
            "references a metric that does not exist",
            vec![PatternIndicator::new(
                "loss_trend_explodingg",
                "greater_than",
                1.0,
            )],
        );
        assert!(
            detector.register_pattern(bad_metric, RiskLevel::High).is_err(),
            "an unknown metric must be rejected at registration, not silently ignored"
        );

        let bad_condition = Pattern::all_of(
            "typo2",
            "references a condition that does not exist",
            vec![PatternIndicator::new(
                "lr_effectiveness",
                "much_bigger_than",
                1.0,
            )],
        );
        assert!(detector.register_pattern(bad_condition, RiskLevel::High).is_err());

        let empty = Pattern::all_of("empty", "no indicators", vec![]);
        assert!(detector.register_pattern(empty, RiskLevel::Low).is_err());

        // min_indicators must be reachable, otherwise the rule is dead on arrival.
        let unreachable = Pattern::at_least(
            "unreachable",
            "asks for more indicators than it has",
            vec![PatternIndicator::new("lr_effectiveness", "less_than", 0.5)],
            2,
        );
        assert!(detector.register_pattern(unreachable, RiskLevel::High).is_err());

        let never = Pattern::at_least(
            "never",
            "min_indicators of zero",
            vec![PatternIndicator::new("lr_effectiveness", "less_than", 0.5)],
            0,
        );
        assert!(detector.register_pattern(never, RiskLevel::High).is_err());
        assert_eq!(detector.pattern_count(), 0);
    }

    #[test]
    fn test_custom_pattern_is_evaluated() {
        let mut detector = PatternDetector::empty();
        detector
            .register_pattern(
                Pattern::all_of(
                    "slow_convergence",
                    "convergence velocity under 0.2",
                    vec![PatternIndicator::new(
                        "convergence_velocity",
                        "less_than",
                        0.2,
                    )],
                ),
                RiskLevel::Medium,
            )
            .expect("registration should succeed for known metric/condition");

        let slow = dynamics(
            TrendDirection::Decreasing,
            TrendDirection::Stable,
            0.8,
            0.05,
            0.0,
            vec![(1.0, 1.0)],
        );
        let hits = detector.detect_patterns(&slow);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].pattern.name, "slow_convergence");

        let fast = dynamics(
            TrendDirection::Decreasing,
            TrendDirection::Stable,
            0.8,
            0.9,
            0.0,
            vec![(1.0, 1.0)],
        );
        assert!(detector.detect_patterns(&fast).is_empty());
    }

    #[test]
    fn test_trajectory_stats_handle_an_empty_trajectory() {
        let detector = PatternDetector::new();
        let empty = dynamics(
            TrendDirection::Stable,
            TrendDirection::Stable,
            0.5,
            0.5,
            0.0,
            vec![],
        );
        // Must not panic or divide by zero; a spike ratio of 1.0 means "no spike".
        let detected = detector.detect_patterns(&empty);
        assert!(detected.iter().all(|d| d.pattern.name != "loss_spike"));
    }
}
