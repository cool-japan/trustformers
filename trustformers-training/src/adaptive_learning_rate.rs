//! Adaptive Learning Rate Schedulers for Dynamic Training Optimization
//!
//! This module implements advanced learning rate scheduling techniques that automatically
//! adjust learning rates based on real-time training dynamics and performance metrics.

use anyhow::Result;
use log;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Configuration for adaptive learning rate scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLearningRateConfig {
    /// Initial learning rate
    pub initial_lr: f32,
    /// Minimum learning rate
    pub min_lr: f32,
    /// Maximum learning rate
    pub max_lr: f32,
    /// Enable loss-based adaptation
    pub loss_based_adaptation: bool,
    /// Enable gradient-based adaptation
    pub gradient_based_adaptation: bool,
    /// Enable plateau detection
    pub plateau_detection: bool,
    /// Patience for plateau detection (steps)
    pub plateau_patience: usize,
    /// Plateau threshold (relative improvement)
    pub plateau_threshold: f32,
    /// Adaptation factor for reductions
    pub reduction_factor: f32,
    /// Adaptation factor for increases
    pub increase_factor: f32,
    /// Window size for trend analysis
    pub trend_window: usize,
    /// Momentum for exponential moving averages
    pub momentum: f32,
    /// Enable cyclical learning rates
    pub cyclical_lr: bool,
    /// Cycle length for cyclical LR
    pub cycle_length: usize,
    /// Enable learning rate range test mode
    pub lr_range_test: bool,
}

impl Default for AdaptiveLearningRateConfig {
    fn default() -> Self {
        Self {
            initial_lr: 1e-3,
            min_lr: 1e-7,
            max_lr: 1e-1,
            loss_based_adaptation: true,
            gradient_based_adaptation: true,
            plateau_detection: true,
            plateau_patience: 50,
            plateau_threshold: 1e-4,
            reduction_factor: 0.5,
            increase_factor: 1.1,
            trend_window: 20,
            momentum: 0.9,
            cyclical_lr: false,
            cycle_length: 1000,
            lr_range_test: false,
        }
    }
}

/// Training dynamics for learning rate adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDynamics {
    pub step: usize,
    pub loss: f32,
    pub gradient_norm: f32,
    pub learning_rate: f32,
    pub accuracy: Option<f32>,
}

/// Learning rate adaptation strategy
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    ReduceOnPlateau,
    CosineAnnealing,
    ExponentialDecay,
    PolynomialDecay,
    CyclicalLR,
    OneCycleLR,
    GradientNormAdaptive,
    LossVarianceAdaptive,
    PerformanceBasedAdaptive,
}

/// Learning rate scheduler state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerState {
    pub current_lr: f32,
    pub best_loss: f32,
    pub plateau_counter: usize,
    pub step_count: usize,
    pub cycle_position: usize,
    pub adaptation_history: VecDeque<f32>,
    pub performance_trend: PerformanceTrend,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Deteriorating,
    Oscillating,
    Unknown,
}

/// Adaptive learning rate scheduler
pub struct AdaptiveLearningRateScheduler {
    config: AdaptiveLearningRateConfig,
    state: SchedulerState,
    dynamics_history: VecDeque<TrainingDynamics>,
    loss_ema: f32,
    gradient_norm_ema: f32,
    strategies: Vec<AdaptationStrategy>,
    strategy_weights: HashMap<AdaptationStrategy, f32>,
    /// Track strategy effectiveness over time
    strategy_effectiveness: HashMap<AdaptationStrategy, f32>,
    /// Emergency fallback when all strategies fail
    emergency_mode: bool,
}

impl AdaptiveLearningRateScheduler {
    pub fn new(config: AdaptiveLearningRateConfig) -> Self {
        let state = SchedulerState {
            current_lr: config.initial_lr,
            best_loss: f32::INFINITY,
            plateau_counter: 0,
            step_count: 0,
            cycle_position: 0,
            adaptation_history: VecDeque::with_capacity(config.trend_window),
            performance_trend: PerformanceTrend::Unknown,
        };

        let strategies = vec![
            AdaptationStrategy::ReduceOnPlateau,
            AdaptationStrategy::GradientNormAdaptive,
            AdaptationStrategy::LossVarianceAdaptive,
        ];

        let strategy_weights =
            strategies.iter().map(|s| (s.clone(), 1.0 / strategies.len() as f32)).collect();

        let strategy_effectiveness = strategies.iter()
            .map(|s| (s.clone(), 0.5)) // Initialize with neutral effectiveness
            .collect();

        Self {
            config,
            state,
            dynamics_history: VecDeque::with_capacity(1000),
            loss_ema: 0.0,
            gradient_norm_ema: 0.0,
            strategies,
            strategy_weights,
            strategy_effectiveness,
            emergency_mode: false,
        }
    }

    /// Update learning rate based on current training dynamics
    pub fn step(&mut self, dynamics: TrainingDynamics) -> Result<LearningRateUpdate> {
        // Validate input dynamics
        if !dynamics.loss.is_finite() || !dynamics.gradient_norm.is_finite() {
            log::warn!(
                "Invalid training dynamics: loss={}, grad_norm={}",
                dynamics.loss,
                dynamics.gradient_norm
            );
            return Err(anyhow::anyhow!("Invalid training dynamics detected"));
        }

        self.state.step_count += 1;

        // Update exponential moving averages with validation
        if self.state.step_count == 1 {
            self.loss_ema = dynamics.loss;
            self.gradient_norm_ema = dynamics.gradient_norm;
        } else {
            // Robust EMA update with bounds checking
            let loss_update = (1.0 - self.config.momentum) * dynamics.loss;
            let grad_update = (1.0 - self.config.momentum) * dynamics.gradient_norm;

            if loss_update.is_finite() {
                self.loss_ema = self.config.momentum * self.loss_ema + loss_update;
            }

            if grad_update.is_finite() {
                self.gradient_norm_ema =
                    self.config.momentum * self.gradient_norm_ema + grad_update;
            }
        }

        // Store dynamics
        self.dynamics_history.push_back(dynamics.clone());
        if self.dynamics_history.len() > self.dynamics_history.capacity() {
            self.dynamics_history.pop_front();
        }

        // Analyze performance trend
        self.state.performance_trend = self.analyze_performance_trend();

        // Compute adaptive learning rate with safety checks
        let new_lr = match self.compute_adaptive_learning_rate(&dynamics) {
            Ok(lr) if lr.is_finite() && lr > 0.0 => lr,
            Ok(lr) => {
                log::warn!("Invalid learning rate computed: {}. Using current LR.", lr);
                self.state.current_lr
            },
            Err(e) => {
                log::error!(
                    "Failed to compute adaptive learning rate: {}. Using current LR.",
                    e
                );
                self.state.current_lr
            },
        };

        let old_lr = self.state.current_lr;
        self.state.current_lr = new_lr.clamp(self.config.min_lr, self.config.max_lr);

        // Additional safety check
        if !self.state.current_lr.is_finite() {
            log::error!("Learning rate became non-finite. Resetting to initial LR.");
            self.state.current_lr = self.config.initial_lr;
        }

        // Update adaptation history
        let adaptation_ratio = self.state.current_lr / old_lr;
        self.state.adaptation_history.push_back(adaptation_ratio);
        if self.state.adaptation_history.len() > self.config.trend_window {
            self.state.adaptation_history.pop_front();
        }

        // Update plateau detection
        if dynamics.loss < self.state.best_loss - self.config.plateau_threshold {
            self.state.best_loss = dynamics.loss;
            self.state.plateau_counter = 0;
        } else {
            self.state.plateau_counter += 1;
        }

        // Update cycle position for cyclical learning rates
        if self.config.cyclical_lr {
            self.state.cycle_position = (self.state.cycle_position + 1) % self.config.cycle_length;
        }

        // Update strategy effectiveness based on performance improvement
        self.update_strategy_effectiveness(&dynamics, old_lr, self.state.current_lr);

        // Check if we need emergency mode
        if self.should_enter_emergency_mode() {
            self.emergency_mode = true;
            self.state.current_lr = self.config.initial_lr * 0.1; // Conservative emergency LR
            log::warn!("Entering emergency mode - using conservative learning rate");
        } else if self.emergency_mode && self.can_exit_emergency_mode() {
            self.emergency_mode = false;
            log::info!("Exiting emergency mode - performance stabilized");
        }

        Ok(LearningRateUpdate {
            old_lr,
            new_lr: self.state.current_lr,
            adaptation_reason: self.get_adaptation_reason(),
            strategy_contributions: self.compute_strategy_contributions(&dynamics)?,
            confidence: self.compute_adaptation_confidence(),
            dynamics: dynamics.clone(),
        })
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f32 {
        self.state.current_lr
    }

    /// Get scheduler state
    pub fn get_state(&self) -> &SchedulerState {
        &self.state
    }

    /// Get comprehensive statistics
    pub fn get_statistics(&self) -> AdaptiveLRStatistics {
        AdaptiveLRStatistics {
            current_lr: self.state.current_lr,
            steps_taken: self.state.step_count,
            adaptations_made: self.count_adaptations(),
            performance_trend: self.state.performance_trend.clone(),
            plateau_detected: self.state.plateau_counter >= self.config.plateau_patience,
            loss_ema: self.loss_ema,
            gradient_norm_ema: self.gradient_norm_ema,
            adaptation_frequency: self.compute_adaptation_frequency(),
            stability_score: self.compute_stability_score(),
        }
    }

    // Private helper methods
    fn compute_adaptive_learning_rate(&mut self, dynamics: &TrainingDynamics) -> Result<f32> {
        let mut contributions = Vec::new();

        for strategy in &self.strategies {
            if let Some(weight) = self.strategy_weights.get(strategy) {
                let contribution = self.compute_strategy_contribution(strategy, dynamics)? * weight;
                contributions.push(contribution);
            }
        }

        // Weighted average of strategy contributions
        let adaptive_factor: f32 = contributions.iter().sum::<f32>() / contributions.len() as f32;

        // Apply cyclical learning rate if enabled
        let base_lr = if self.config.cyclical_lr {
            self.compute_cyclical_lr()
        } else {
            self.state.current_lr
        };

        Ok(base_lr * adaptive_factor)
    }

    fn compute_strategy_contribution(
        &self,
        strategy: &AdaptationStrategy,
        dynamics: &TrainingDynamics,
    ) -> Result<f32> {
        let contribution = match strategy {
            AdaptationStrategy::ReduceOnPlateau => {
                let plateau_severity = (self.state.plateau_counter as f32
                    / self.config.plateau_patience as f32)
                    .min(2.0);
                if self.state.plateau_counter >= self.config.plateau_patience {
                    // Gradual reduction based on plateau severity
                    self.config.reduction_factor.powf(plateau_severity * 0.5)
                } else {
                    1.0
                }
            },
            AdaptationStrategy::GradientNormAdaptive => {
                let grad_ratio = dynamics.gradient_norm / self.gradient_norm_ema.max(1e-8);

                // Smooth adaptation based on gradient ratio
                if grad_ratio > 2.0 {
                    let severity = (grad_ratio / 2.0 - 1.0).min(2.0);
                    self.config.reduction_factor.powf(severity * 0.3)
                } else if grad_ratio < 0.5 {
                    let boost = (1.0 - grad_ratio * 2.0).min(1.0);
                    self.config.increase_factor.powf(boost * 0.2)
                } else {
                    // Smooth transition in the middle range
                    1.0 + (grad_ratio - 1.0) * 0.1
                }
            },
            AdaptationStrategy::LossVarianceAdaptive => {
                if self.dynamics_history.len() < 10 {
                    return Ok(1.0);
                }

                let recent_losses: Vec<f32> =
                    self.dynamics_history.iter().rev().take(10).map(|d| d.loss).collect();

                let variance = self.compute_variance(&recent_losses);
                let cv = variance.sqrt() / self.loss_ema.max(1e-8);

                // Smoother variance-based adaptation
                if cv > 0.1 {
                    let instability = (cv - 0.1) / 0.1;
                    self.config.reduction_factor.powf(instability.min(1.0) * 0.5)
                } else if cv < 0.05 {
                    // Very stable -> can increase slightly
                    let stability = (0.05 - cv) / 0.05;
                    self.config.increase_factor.powf(stability * 0.1)
                } else {
                    1.0
                }
            },
            AdaptationStrategy::PerformanceBasedAdaptive => {
                match self.state.performance_trend {
                    PerformanceTrend::Improving => {
                        // Conservative increase for improving performance
                        self.config.increase_factor.powf(0.3)
                    },
                    PerformanceTrend::Deteriorating => {
                        // More aggressive reduction for deteriorating performance
                        self.config.reduction_factor.powf(0.7)
                    },
                    PerformanceTrend::Oscillating => {
                        // Stabilize oscillations with slight reduction
                        self.config.reduction_factor.powf(0.2)
                    },
                    _ => 1.0,
                }
            },
            _ => 1.0, // Default: no change
        };

        // Ensure contribution is valid and within reasonable bounds
        if contribution.is_finite() && contribution > 0.0 {
            Ok(contribution.clamp(0.1, 10.0))
        } else {
            log::warn!(
                "Invalid strategy contribution computed for {:?}: {}",
                strategy,
                contribution
            );
            Ok(1.0)
        }
    }

    fn compute_cyclical_lr(&self) -> f32 {
        let cycle_progress = self.state.cycle_position as f32 / self.config.cycle_length as f32;
        let lr_range = self.config.max_lr - self.config.min_lr;

        // Triangular cyclical learning rate
        if cycle_progress < 0.5 {
            self.config.min_lr + lr_range * (2.0 * cycle_progress)
        } else {
            self.config.max_lr - lr_range * (2.0 * (cycle_progress - 0.5))
        }
    }

    fn analyze_performance_trend(&self) -> PerformanceTrend {
        if self.dynamics_history.len() < self.config.trend_window {
            return PerformanceTrend::Unknown;
        }

        // Take the most recent entries and restore chronological order (oldest to newest)
        // This ensures slope is negative when loss is decreasing over time
        let mut recent_losses: Vec<f32> = self
            .dynamics_history
            .iter()
            .rev()
            .take(self.config.trend_window)
            .map(|d| d.loss)
            .collect();
        recent_losses.reverse(); // Restore chronological order

        let slope = self.compute_slope(&recent_losses);
        let variance = self.compute_variance(&recent_losses);

        if variance > 0.1 {
            PerformanceTrend::Oscillating
        } else if slope < -0.01 {
            PerformanceTrend::Improving
        } else if slope > 0.01 {
            PerformanceTrend::Deteriorating
        } else {
            PerformanceTrend::Stable
        }
    }

    fn compute_slope(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        // Filter out invalid values first
        let valid_pairs: Vec<(f32, f32)> = values
            .iter()
            .enumerate()
            .filter_map(
                |(i, &y)| {
                    if y.is_finite() {
                        Some((i as f32, y))
                    } else {
                        None
                    }
                },
            )
            .collect();

        if valid_pairs.len() < 2 {
            return 0.0;
        }

        let n = valid_pairs.len() as f32;
        let sum_x: f32 = valid_pairs.iter().map(|(x, _)| x).sum();
        let sum_y: f32 = valid_pairs.iter().map(|(_, y)| y).sum();
        let sum_xy: f32 = valid_pairs.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f32 = valid_pairs.iter().map(|(x, _)| x * x).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;

        if denominator.abs() < 1e-10 {
            return 0.0; // Avoid division by zero
        }

        (n * sum_xy - sum_x * sum_y) / denominator
    }

    fn compute_variance(&self, values: &[f32]) -> f32 {
        if values.len() <= 1 {
            return 0.0;
        }

        // Use Welford's online algorithm for numerical stability
        let mut mean = 0.0;
        let mut m2 = 0.0;

        for (i, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                continue; // Skip invalid values
            }

            let delta = value - mean;
            mean += delta / (i + 1) as f32;
            let delta2 = value - mean;
            m2 += delta * delta2;
        }

        if values.len() > 1 {
            m2 / (values.len() - 1) as f32
        } else {
            0.0
        }
    }

    fn get_adaptation_reason(&self) -> String {
        if self.state.plateau_counter >= self.config.plateau_patience {
            "Plateau detected".to_string()
        } else if matches!(
            self.state.performance_trend,
            PerformanceTrend::Deteriorating
        ) {
            "Performance deteriorating".to_string()
        } else if matches!(self.state.performance_trend, PerformanceTrend::Improving) {
            "Performance improving".to_string()
        } else {
            "Routine adaptation".to_string()
        }
    }

    fn compute_strategy_contributions(
        &self,
        dynamics: &TrainingDynamics,
    ) -> Result<HashMap<AdaptationStrategy, f32>> {
        let mut contributions = HashMap::new();

        for strategy in &self.strategies {
            let contribution = self.compute_strategy_contribution(strategy, dynamics)?;
            contributions.insert(strategy.clone(), contribution);
        }

        Ok(contributions)
    }

    fn compute_adaptation_confidence(&self) -> f32 {
        // Confidence based on trend consistency and data quality
        let trend_consistency = if self.dynamics_history.len() >= self.config.trend_window {
            0.8
        } else {
            self.dynamics_history.len() as f32 / self.config.trend_window as f32
        };

        let data_quality =
            if self.loss_ema > 0.0 && !self.loss_ema.is_infinite() { 0.9 } else { 0.5 };

        (trend_consistency * data_quality).min(1.0)
    }

    fn count_adaptations(&self) -> usize {
        self.state
            .adaptation_history
            .iter()
            .filter(|&&ratio| (ratio - 1.0).abs() > 0.01)
            .count()
    }

    fn compute_adaptation_frequency(&self) -> f32 {
        if self.state.step_count == 0 {
            return 0.0;
        }

        self.count_adaptations() as f32 / self.state.step_count as f32
    }

    fn compute_stability_score(&self) -> f32 {
        if self.state.adaptation_history.is_empty() {
            return 1.0;
        }

        let variance = self
            .compute_variance(&self.state.adaptation_history.iter().cloned().collect::<Vec<_>>());
        (1.0 / (1.0 + variance)).clamp(0.0, 1.0)
    }

    /// Update effectiveness scores for strategies based on performance
    fn update_strategy_effectiveness(
        &mut self,
        dynamics: &TrainingDynamics,
        old_lr: f32,
        new_lr: f32,
    ) {
        // Simple effectiveness metric based on loss improvement and LR change correlation
        if self.dynamics_history.len() >= 2 {
            let prev_loss = self.dynamics_history.back().map(|d| d.loss).unwrap_or(dynamics.loss);
            let loss_improvement =
                if prev_loss > 0.0 { (prev_loss - dynamics.loss) / prev_loss } else { 0.0 };

            let lr_change_magnitude = (new_lr / old_lr - 1.0).abs();

            // Reward strategies that contribute to improvements without excessive LR changes
            let base_effectiveness = if loss_improvement > 0.0 {
                (loss_improvement * 10.0).min(1.0)
            } else {
                0.2 // Small penalty for no improvement
            };

            // Penalize excessive LR changes
            let stability_bonus =
                if lr_change_magnitude < 0.1 { 0.1 } else { -lr_change_magnitude * 0.5 };

            let overall_effectiveness = (base_effectiveness + stability_bonus).clamp(0.0, 1.0);

            // Update strategy effectiveness with exponential moving average
            for (_strategy, effectiveness) in self.strategy_effectiveness.iter_mut() {
                let learning_rate = 0.1;
                *effectiveness =
                    learning_rate * overall_effectiveness + (1.0 - learning_rate) * *effectiveness;
            }
        }
    }

    /// Check if we should enter emergency mode
    fn should_enter_emergency_mode(&self) -> bool {
        if self.emergency_mode {
            return false; // Already in emergency mode
        }

        // Enter emergency mode if:
        // 1. Recent loss has exploded
        // 2. Multiple consecutive bad adaptations
        // 3. All strategies are performing poorly

        let recent_loss_explosion = self.dynamics_history.len() >= 2 && {
            let recent_losses: Vec<f32> =
                self.dynamics_history.iter().rev().take(3).map(|d| d.loss).collect();
            recent_losses.windows(2).any(|w| w[0] > w[1] * 5.0)
        };

        let poor_strategy_performance = self.strategy_effectiveness.values().all(|&eff| eff < 0.3);

        let high_variance = if self.dynamics_history.len() >= 10 {
            let recent_losses: Vec<f32> =
                self.dynamics_history.iter().rev().take(10).map(|d| d.loss).collect();
            let variance = self.compute_variance(&recent_losses);
            let cv = variance.sqrt() / self.loss_ema.max(1e-8);
            cv > 0.5
        } else {
            false
        };

        recent_loss_explosion || (poor_strategy_performance && high_variance)
    }

    /// Check if we can exit emergency mode
    fn can_exit_emergency_mode(&self) -> bool {
        if !self.emergency_mode {
            return false;
        }

        // Exit emergency mode if performance has stabilized
        let stable_loss = if self.dynamics_history.len() >= 5 {
            let recent_losses: Vec<f32> =
                self.dynamics_history.iter().rev().take(5).map(|d| d.loss).collect();
            let variance = self.compute_variance(&recent_losses);
            let cv = variance.sqrt() / self.loss_ema.max(1e-8);
            cv < 0.1
        } else {
            false
        };

        let improving_trend = matches!(
            self.state.performance_trend,
            PerformanceTrend::Improving | PerformanceTrend::Stable
        );

        stable_loss && improving_trend
    }
}

/// Learning rate update result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRateUpdate {
    pub old_lr: f32,
    pub new_lr: f32,
    pub adaptation_reason: String,
    pub strategy_contributions: HashMap<AdaptationStrategy, f32>,
    pub confidence: f32,
    pub dynamics: TrainingDynamics,
}

/// Comprehensive learning rate statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLRStatistics {
    pub current_lr: f32,
    pub steps_taken: usize,
    pub adaptations_made: usize,
    pub performance_trend: PerformanceTrend,
    pub plateau_detected: bool,
    pub loss_ema: f32,
    pub gradient_norm_ema: f32,
    pub adaptation_frequency: f32,
    pub stability_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_lr_scheduler_creation() {
        let config = AdaptiveLearningRateConfig::default();
        let scheduler = AdaptiveLearningRateScheduler::new(config.clone());
        assert_eq!(scheduler.get_lr(), config.initial_lr);
    }

    #[test]
    fn test_learning_rate_adaptation() {
        let config = AdaptiveLearningRateConfig::default();
        let mut scheduler = AdaptiveLearningRateScheduler::new(config);

        let dynamics = TrainingDynamics {
            step: 1,
            loss: 1.0,
            gradient_norm: 0.5,
            learning_rate: 1e-3,
            accuracy: Some(0.8),
        };

        let update = scheduler.step(dynamics).unwrap();
        assert!(update.new_lr > 0.0);
        assert!(!update.adaptation_reason.is_empty());
    }

    #[test]
    fn test_plateau_detection() {
        let mut config = AdaptiveLearningRateConfig::default();
        config.plateau_patience = 3;
        let mut scheduler = AdaptiveLearningRateScheduler::new(config);

        // Simulate plateau by using same loss repeatedly
        for i in 1..=5 {
            let dynamics = TrainingDynamics {
                step: i,
                loss: 1.0, // Same loss
                gradient_norm: 0.5,
                learning_rate: scheduler.get_lr(),
                accuracy: None,
            };
            scheduler.step(dynamics).unwrap();
        }

        let stats = scheduler.get_statistics();
        assert!(stats.plateau_detected);
    }

    #[test]
    fn test_performance_trend_analysis() {
        let config = AdaptiveLearningRateConfig::default();
        let mut scheduler = AdaptiveLearningRateScheduler::new(config);

        // Simulate improving performance
        for i in 1..=25 {
            let dynamics = TrainingDynamics {
                step: i,
                loss: 2.0 - (i as f32) * 0.05, // Decreasing loss
                gradient_norm: 0.5,
                learning_rate: scheduler.get_lr(),
                accuracy: None,
            };
            scheduler.step(dynamics).unwrap();
        }

        let stats = scheduler.get_statistics();
        assert!(matches!(
            stats.performance_trend,
            PerformanceTrend::Improving
        ));
    }

    #[test]
    fn test_cyclical_learning_rate() {
        let mut config = AdaptiveLearningRateConfig::default();
        config.cyclical_lr = true;
        config.cycle_length = 10;
        let scheduler = AdaptiveLearningRateScheduler::new(config);

        let cyclical_lr = scheduler.compute_cyclical_lr();
        assert!(cyclical_lr >= scheduler.config.min_lr);
        assert!(cyclical_lr <= scheduler.config.max_lr);
    }
}
