//! Training Dynamics Analysis
//!
//! Advanced analysis tools for understanding training dynamics including
//! loss curve analysis, convergence detection, and learning rate impact assessment.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Configuration for training dynamics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDynamicsConfig {
    /// Enable loss curve analysis
    pub enable_loss_curve_analysis: bool,
    /// Enable learning rate impact analysis
    pub enable_learning_rate_analysis: bool,
    /// Enable batch size effects analysis
    pub enable_batch_size_analysis: bool,
    /// Enable convergence detection
    pub enable_convergence_detection: bool,
    /// Enable plateau identification
    pub enable_plateau_identification: bool,
    /// Window size for moving averages
    pub moving_average_window: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f32,
    /// Plateau detection threshold
    pub plateau_threshold: f32,
    /// Minimum epochs for convergence detection
    pub min_epochs_for_convergence: usize,
    /// Maximum history length
    pub max_history_length: usize,
}

impl Default for TrainingDynamicsConfig {
    fn default() -> Self {
        Self {
            enable_loss_curve_analysis: true,
            enable_learning_rate_analysis: true,
            enable_batch_size_analysis: true,
            enable_convergence_detection: true,
            enable_plateau_identification: true,
            moving_average_window: 10,
            convergence_tolerance: 1e-6,
            plateau_threshold: 1e-4,
            min_epochs_for_convergence: 20,
            max_history_length: 10000,
        }
    }
}

/// Training metrics at a specific point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub step: usize,
    pub train_loss: f32,
    pub validation_loss: Option<f32>,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub gradient_norm: Option<f32>,
    pub accuracy: Option<f32>,
    pub timestamp: f64,
}

/// Loss curve analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossCurveAnalysis {
    pub trend: LossTrend,
    pub smoothness: f32,
    pub volatility: f32,
    pub improvement_rate: f32,
    pub best_loss: f32,
    pub current_loss: f32,
    pub loss_reduction_percentage: f32,
    pub epochs_since_improvement: usize,
    pub moving_averages: MovingAverages,
    pub loss_statistics: LossStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossTrend {
    Decreasing,
    Increasing,
    Oscillating,
    Plateaued,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovingAverages {
    pub short_term: f32,  // Last 5-10 epochs
    pub medium_term: f32, // Last 20-50 epochs
    pub long_term: f32,   // Last 100+ epochs
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossStatistics {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub median: f32,
    pub percentile_25: f32,
    pub percentile_75: f32,
    pub autocorrelation: f32,
}

/// Learning rate impact analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRateAnalysis {
    pub current_lr: f32,
    pub lr_schedule_type: LRScheduleType,
    pub lr_impact_score: f32,
    pub optimal_lr_estimate: f32,
    pub lr_sensitivity: f32,
    pub lr_history: Vec<LearningRatePoint>,
    pub recommendations: Vec<LRRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LRScheduleType {
    Constant,
    StepDecay,
    ExponentialDecay,
    CosineAnnealing,
    ReduceOnPlateau,
    Warmup,
    Cyclical,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRatePoint {
    pub epoch: usize,
    pub learning_rate: f32,
    pub loss_change: f32,
    pub gradient_norm: Option<f32>,
    pub effectiveness: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LRRecommendation {
    pub action: LRAction,
    pub confidence: f32,
    pub rationale: String,
    pub expected_improvement: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LRAction {
    Increase,
    Decrease,
    KeepCurrent,
    AddScheduler,
    ChangeScheduler,
    AddWarmup,
}

/// Batch size effects analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSizeAnalysis {
    pub current_batch_size: usize,
    pub batch_size_efficiency: f32,
    pub gradient_noise_level: f32,
    pub convergence_speed: f32,
    pub memory_utilization: f32,
    pub optimal_batch_size_estimate: usize,
    pub batch_size_history: Vec<BatchSizePoint>,
    pub recommendations: Vec<BatchSizeRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSizePoint {
    pub epoch: usize,
    pub batch_size: usize,
    pub loss_improvement: f32,
    pub gradient_stability: f32,
    pub throughput: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSizeRecommendation {
    pub suggested_batch_size: usize,
    pub confidence: f32,
    pub rationale: String,
    pub expected_benefits: Vec<String>,
}

/// Convergence detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceAnalysis {
    pub convergence_status: ConvergenceStatus,
    pub convergence_probability: f32,
    pub epochs_to_convergence_estimate: Option<usize>,
    pub convergence_criteria: Vec<ConvergenceCriterion>,
    pub early_stopping_recommendation: Option<EarlyStoppingRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvergenceStatus {
    Converging,
    Converged,
    Diverging,
    Oscillating,
    TooEarly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceCriterion {
    pub criterion_type: ConvergenceCriterionType,
    pub current_value: f32,
    pub threshold: f32,
    pub satisfied: bool,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvergenceCriterionType {
    LossStability,
    GradientMagnitude,
    LossImprovement,
    ValidationGap,
    LearningRateDecay,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingRecommendation {
    pub should_stop: bool,
    pub confidence: f32,
    pub rationale: String,
    pub suggested_epochs_remaining: usize,
}

/// Plateau identification results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlateauAnalysis {
    pub plateau_detected: bool,
    pub plateau_duration: usize,
    pub plateau_level: f32,
    pub plateau_type: PlateauType,
    pub escape_probability: f32,
    pub plateau_characteristics: PlateauCharacteristics,
    pub recommendations: Vec<PlateauRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlateauType {
    LossPlayteau,
    GradientPlateau,
    AccuracyPlateau,
    LearningRatePlateau,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlateauCharacteristics {
    pub stability: f32,
    pub noise_level: f32,
    pub gradient_magnitude: f32,
    pub overfitting_risk: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlateauRecommendation {
    pub action: PlateauAction,
    pub priority: Priority,
    pub description: String,
    pub implementation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlateauAction {
    IncreaseLearningRate,
    DecreaseLearningRate,
    ChangeBatchSize,
    AddRegularization,
    RemoveRegularization,
    ChangeOptimizer,
    AddNoise,
    EarlyStopping,
    ContinueTraining,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

/// Comprehensive training dynamics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDynamicsReport {
    pub loss_curve_analysis: Option<LossCurveAnalysis>,
    pub learning_rate_analysis: Option<LearningRateAnalysis>,
    pub batch_size_analysis: Option<BatchSizeAnalysis>,
    pub convergence_analysis: Option<ConvergenceAnalysis>,
    pub plateau_analysis: Option<PlateauAnalysis>,
    pub training_summary: TrainingSummary,
    pub recommendations: Vec<TrainingRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSummary {
    pub total_epochs: usize,
    pub total_steps: usize,
    pub training_efficiency: f32,
    pub convergence_health: f32,
    pub stability_score: f32,
    pub overall_progress: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRecommendation {
    pub category: TrainingCategory,
    pub priority: Priority,
    pub description: String,
    pub implementation: String,
    pub expected_impact: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingCategory {
    LearningRate,
    BatchSize,
    Optimization,
    Regularization,
    EarlyStopping,
    Architecture,
}

/// Training dynamics analyzer
#[derive(Debug)]
pub struct TrainingDynamicsAnalyzer {
    config: TrainingDynamicsConfig,
    metrics_history: VecDeque<TrainingMetrics>,
    analysis_cache: HashMap<String, TrainingDynamicsReport>,
}

impl TrainingDynamicsAnalyzer {
    /// Create a new training dynamics analyzer
    pub fn new(config: TrainingDynamicsConfig) -> Self {
        Self {
            config,
            metrics_history: VecDeque::new(),
            analysis_cache: HashMap::new(),
        }
    }

    /// Record training metrics
    pub fn record_metrics(&mut self, metrics: TrainingMetrics) {
        self.metrics_history.push_back(metrics);

        // Limit history size
        while self.metrics_history.len() > self.config.max_history_length {
            self.metrics_history.pop_front();
        }
    }

    /// Perform comprehensive training dynamics analysis
    pub async fn analyze(&mut self) -> Result<TrainingDynamicsReport> {
        let mut report = TrainingDynamicsReport {
            loss_curve_analysis: None,
            learning_rate_analysis: None,
            batch_size_analysis: None,
            convergence_analysis: None,
            plateau_analysis: None,
            training_summary: TrainingSummary {
                total_epochs: 0,
                total_steps: 0,
                training_efficiency: 0.0,
                convergence_health: 0.0,
                stability_score: 0.0,
                overall_progress: 0.0,
            },
            recommendations: Vec::new(),
        };

        if self.config.enable_loss_curve_analysis {
            report.loss_curve_analysis = Some(self.analyze_loss_curve().await?);
        }

        if self.config.enable_learning_rate_analysis {
            report.learning_rate_analysis = Some(self.analyze_learning_rate().await?);
        }

        if self.config.enable_batch_size_analysis {
            report.batch_size_analysis = Some(self.analyze_batch_size().await?);
        }

        if self.config.enable_convergence_detection {
            report.convergence_analysis = Some(self.detect_convergence().await?);
        }

        if self.config.enable_plateau_identification {
            report.plateau_analysis = Some(self.identify_plateau().await?);
        }

        self.generate_training_summary(&mut report);
        self.generate_training_recommendations(&mut report);

        Ok(report)
    }

    /// Analyze loss curve patterns
    async fn analyze_loss_curve(&self) -> Result<LossCurveAnalysis> {
        if self.metrics_history.is_empty() {
            return Ok(LossCurveAnalysis {
                trend: LossTrend::Unknown,
                smoothness: 0.0,
                volatility: 0.0,
                improvement_rate: 0.0,
                best_loss: 0.0,
                current_loss: 0.0,
                loss_reduction_percentage: 0.0,
                epochs_since_improvement: 0,
                moving_averages: MovingAverages {
                    short_term: 0.0,
                    medium_term: 0.0,
                    long_term: 0.0,
                },
                loss_statistics: LossStatistics {
                    mean: 0.0,
                    std: 0.0,
                    min: 0.0,
                    max: 0.0,
                    median: 0.0,
                    percentile_25: 0.0,
                    percentile_75: 0.0,
                    autocorrelation: 0.0,
                },
            });
        }

        let losses: Vec<f32> = self.metrics_history.iter().map(|m| m.train_loss).collect();

        let trend = self.detect_loss_trend(&losses);
        let smoothness = self.calculate_smoothness(&losses);
        let volatility = self.calculate_volatility(&losses);
        let improvement_rate = self.calculate_improvement_rate(&losses);

        let best_loss = losses.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let current_loss = *losses.last().unwrap();
        let loss_reduction_percentage = if losses.len() > 1 {
            ((losses[0] - current_loss) / losses[0].abs()) * 100.0
        } else {
            0.0
        };

        let epochs_since_improvement = self.calculate_epochs_since_improvement(&losses, best_loss);
        let moving_averages = self.calculate_moving_averages(&losses);
        let loss_statistics = self.calculate_loss_statistics(&losses);

        Ok(LossCurveAnalysis {
            trend,
            smoothness,
            volatility,
            improvement_rate,
            best_loss,
            current_loss,
            loss_reduction_percentage,
            epochs_since_improvement,
            moving_averages,
            loss_statistics,
        })
    }

    /// Detect overall trend in loss curve
    fn detect_loss_trend(&self, losses: &[f32]) -> LossTrend {
        if losses.len() < 3 {
            return LossTrend::Unknown;
        }

        let window_size = (losses.len() / 4).max(5).min(20);
        let recent_losses = &losses[losses.len().saturating_sub(window_size)..];
        let early_losses = &losses[..window_size.min(losses.len())];

        let recent_mean = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
        let early_mean = early_losses.iter().sum::<f32>() / early_losses.len() as f32;

        let improvement = (early_mean - recent_mean) / early_mean.abs();

        // Check for plateau
        let recent_std = self.calculate_std(recent_losses);
        let recent_mean_abs = recent_mean.abs();

        if recent_std / recent_mean_abs.max(1e-8) < self.config.plateau_threshold {
            return LossTrend::Plateaued;
        }

        // Check for oscillation
        let oscillation_score = self.detect_oscillation(losses);
        if oscillation_score > 0.5 {
            return LossTrend::Oscillating;
        }

        if improvement > 0.01 {
            LossTrend::Decreasing
        } else if improvement < -0.01 {
            LossTrend::Increasing
        } else {
            LossTrend::Plateaued
        }
    }

    /// Calculate smoothness of loss curve
    fn calculate_smoothness(&self, losses: &[f32]) -> f32 {
        if losses.len() < 2 {
            return 1.0;
        }

        let differences: Vec<f32> = losses.windows(2).map(|w| (w[1] - w[0]).abs()).collect();

        let mean_diff = differences.iter().sum::<f32>() / differences.len() as f32;
        let mean_loss = losses.iter().sum::<f32>() / losses.len() as f32;

        // Smoothness is inverse of relative variation
        1.0 / (1.0 + mean_diff / mean_loss.abs().max(1e-8))
    }

    /// Calculate volatility of loss curve
    fn calculate_volatility(&self, losses: &[f32]) -> f32 {
        if losses.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f32> =
            losses.windows(2).map(|w| (w[1] - w[0]) / w[0].abs().max(1e-8)).collect();

        self.calculate_std(&returns)
    }

    /// Calculate improvement rate
    fn calculate_improvement_rate(&self, losses: &[f32]) -> f32 {
        if losses.len() < 2 {
            return 0.0;
        }

        let total_improvement = losses[0] - losses[losses.len() - 1];
        let epochs = losses.len() as f32;

        total_improvement / epochs
    }

    /// Calculate epochs since last improvement
    fn calculate_epochs_since_improvement(&self, losses: &[f32], best_loss: f32) -> usize {
        for (i, &loss) in losses.iter().rev().enumerate() {
            if (loss - best_loss).abs() < 1e-8 {
                return i;
            }
        }
        losses.len()
    }

    /// Calculate moving averages
    fn calculate_moving_averages(&self, losses: &[f32]) -> MovingAverages {
        let short_window = 5.min(losses.len());
        let medium_window = 20.min(losses.len());
        let long_window = 100.min(losses.len());

        let short_term = if short_window > 0 {
            losses[losses.len() - short_window..].iter().sum::<f32>() / short_window as f32
        } else {
            0.0
        };

        let medium_term = if medium_window > 0 {
            losses[losses.len() - medium_window..].iter().sum::<f32>() / medium_window as f32
        } else {
            0.0
        };

        let long_term = if long_window > 0 {
            losses[losses.len() - long_window..].iter().sum::<f32>() / long_window as f32
        } else {
            0.0
        };

        MovingAverages {
            short_term,
            medium_term,
            long_term,
        }
    }

    /// Calculate comprehensive loss statistics
    fn calculate_loss_statistics(&self, losses: &[f32]) -> LossStatistics {
        if losses.is_empty() {
            return LossStatistics {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
                percentile_25: 0.0,
                percentile_75: 0.0,
                autocorrelation: 0.0,
            };
        }

        let mean = losses.iter().sum::<f32>() / losses.len() as f32;
        let std = self.calculate_std(losses);

        let mut sorted_losses = losses.to_vec();
        sorted_losses.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted_losses[0];
        let max = sorted_losses[sorted_losses.len() - 1];
        let median = sorted_losses[sorted_losses.len() / 2];
        let percentile_25 = sorted_losses[sorted_losses.len() / 4];
        let percentile_75 = sorted_losses[3 * sorted_losses.len() / 4];

        let autocorrelation = self.calculate_autocorrelation(losses, 1);

        LossStatistics {
            mean,
            std,
            min,
            max,
            median,
            percentile_25,
            percentile_75,
            autocorrelation,
        }
    }

    /// Calculate standard deviation
    fn calculate_std(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;

        variance.sqrt()
    }

    /// Detect oscillation in loss curve
    fn detect_oscillation(&self, losses: &[f32]) -> f32 {
        if losses.len() < 4 {
            return 0.0;
        }

        let mut direction_changes = 0;
        let mut total_comparisons = 0;

        for i in 1..losses.len() - 1 {
            let prev_direction = losses[i] > losses[i - 1];
            let next_direction = losses[i + 1] > losses[i];

            if prev_direction != next_direction {
                direction_changes += 1;
            }
            total_comparisons += 1;
        }

        direction_changes as f32 / total_comparisons as f32
    }

    /// Calculate autocorrelation
    fn calculate_autocorrelation(&self, values: &[f32], lag: usize) -> f32 {
        if values.len() <= lag {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..values.len() - lag {
            numerator += (values[i] - mean) * (values[i + lag] - mean);
        }

        for &value in values {
            denominator += (value - mean).powi(2);
        }

        if denominator > 1e-8 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Analyze learning rate impact
    async fn analyze_learning_rate(&self) -> Result<LearningRateAnalysis> {
        if self.metrics_history.is_empty() {
            return Ok(LearningRateAnalysis {
                current_lr: 0.0,
                lr_schedule_type: LRScheduleType::Unknown,
                lr_impact_score: 0.0,
                optimal_lr_estimate: 0.0,
                lr_sensitivity: 0.0,
                lr_history: Vec::new(),
                recommendations: Vec::new(),
            });
        }

        let current_lr = self.metrics_history.back().unwrap().learning_rate;
        let lr_schedule_type = self.detect_lr_schedule_type();

        let lr_history = self.build_lr_history();
        let lr_impact_score = self.calculate_lr_impact_score(&lr_history);
        let optimal_lr_estimate = self.estimate_optimal_lr(&lr_history);
        let lr_sensitivity = self.calculate_lr_sensitivity(&lr_history);
        let recommendations = self.generate_lr_recommendations(current_lr, &lr_history);

        Ok(LearningRateAnalysis {
            current_lr,
            lr_schedule_type,
            lr_impact_score,
            optimal_lr_estimate,
            lr_sensitivity,
            lr_history,
            recommendations,
        })
    }

    /// Detect learning rate schedule type
    fn detect_lr_schedule_type(&self) -> LRScheduleType {
        let lrs: Vec<f32> = self.metrics_history.iter().map(|m| m.learning_rate).collect();

        if lrs.len() < 3 {
            return LRScheduleType::Unknown;
        }

        // Check for constant LR
        let lr_std = self.calculate_std(&lrs);
        if lr_std < 1e-8 {
            return LRScheduleType::Constant;
        }

        // Check for step decay (sudden drops)
        let mut step_drops = 0;
        for window in lrs.windows(2) {
            if window[1] < window[0] * 0.9 {
                step_drops += 1;
            }
        }

        if step_drops > lrs.len() / 20 {
            return LRScheduleType::StepDecay;
        }

        // Check for exponential decay
        let log_lrs: Vec<f32> = lrs.iter().map(|&lr| lr.ln()).collect();
        let exponential_trend = self.calculate_linear_trend(&log_lrs);
        if exponential_trend < -0.01 {
            return LRScheduleType::ExponentialDecay;
        }

        // Check for cyclical patterns
        let cyclical_score = self.detect_cyclical_pattern(&lrs);
        if cyclical_score > 0.3 {
            return LRScheduleType::Cyclical;
        }

        LRScheduleType::Unknown
    }

    /// Calculate linear trend
    fn calculate_linear_trend(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f32;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f32>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f32;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator > 1e-8 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Detect cyclical patterns
    fn detect_cyclical_pattern(&self, values: &[f32]) -> f32 {
        // Simplified cyclical detection using autocorrelation
        let mut max_autocorr: f32 = 0.0;
        for lag in 2..=values.len() / 4 {
            let autocorr = self.calculate_autocorrelation(values, lag).abs();
            max_autocorr = max_autocorr.max(autocorr);
        }
        max_autocorr
    }

    /// Build learning rate history with effectiveness scores
    fn build_lr_history(&self) -> Vec<LearningRatePoint> {
        let mut history = Vec::new();

        for (i, metrics) in self.metrics_history.iter().enumerate() {
            let loss_change = if i > 0 {
                self.metrics_history[i - 1].train_loss - metrics.train_loss
            } else {
                0.0
            };

            let effectiveness = if loss_change > 0.0 {
                loss_change / metrics.learning_rate.max(1e-8)
            } else {
                0.0
            };

            history.push(LearningRatePoint {
                epoch: metrics.epoch,
                learning_rate: metrics.learning_rate,
                loss_change,
                gradient_norm: metrics.gradient_norm,
                effectiveness,
            });
        }

        history
    }

    /// Calculate learning rate impact score
    fn calculate_lr_impact_score(&self, lr_history: &[LearningRatePoint]) -> f32 {
        if lr_history.is_empty() {
            return 0.0;
        }

        let avg_effectiveness =
            lr_history.iter().map(|p| p.effectiveness).sum::<f32>() / lr_history.len() as f32;

        avg_effectiveness.max(0.0).min(1.0)
    }

    /// Estimate optimal learning rate
    fn estimate_optimal_lr(&self, lr_history: &[LearningRatePoint]) -> f32 {
        if lr_history.is_empty() {
            return 0.001; // Default
        }

        // Find LR with highest effectiveness
        lr_history
            .iter()
            .max_by(|a, b| a.effectiveness.partial_cmp(&b.effectiveness).unwrap())
            .map(|p| p.learning_rate)
            .unwrap_or(0.001)
    }

    /// Calculate learning rate sensitivity
    fn calculate_lr_sensitivity(&self, lr_history: &[LearningRatePoint]) -> f32 {
        if lr_history.len() < 2 {
            return 0.0;
        }

        let effectiveness_values: Vec<f32> = lr_history.iter().map(|p| p.effectiveness).collect();

        self.calculate_std(&effectiveness_values)
    }

    /// Generate learning rate recommendations
    fn generate_lr_recommendations(
        &self,
        current_lr: f32,
        lr_history: &[LearningRatePoint],
    ) -> Vec<LRRecommendation> {
        let mut recommendations = Vec::new();

        if lr_history.is_empty() {
            return recommendations;
        }

        let recent_effectiveness =
            lr_history.iter().rev().take(5).map(|p| p.effectiveness).sum::<f32>()
                / 5.0f32.min(lr_history.len() as f32);

        if recent_effectiveness < 0.1 {
            recommendations.push(LRRecommendation {
                action: LRAction::Decrease,
                confidence: 0.7,
                rationale: "Low learning effectiveness detected".to_string(),
                expected_improvement: 0.3,
            });
        }

        let optimal_lr = self.estimate_optimal_lr(lr_history);
        if current_lr > optimal_lr * 2.0 {
            recommendations.push(LRRecommendation {
                action: LRAction::Decrease,
                confidence: 0.8,
                rationale: "Current LR significantly higher than estimated optimal".to_string(),
                expected_improvement: 0.4,
            });
        } else if current_lr < optimal_lr * 0.5 {
            recommendations.push(LRRecommendation {
                action: LRAction::Increase,
                confidence: 0.6,
                rationale: "Current LR significantly lower than estimated optimal".to_string(),
                expected_improvement: 0.3,
            });
        }

        recommendations
    }

    /// Analyze batch size effects
    async fn analyze_batch_size(&self) -> Result<BatchSizeAnalysis> {
        if self.metrics_history.is_empty() {
            return Ok(BatchSizeAnalysis {
                current_batch_size: 0,
                batch_size_efficiency: 0.0,
                gradient_noise_level: 0.0,
                convergence_speed: 0.0,
                memory_utilization: 0.0,
                optimal_batch_size_estimate: 32,
                batch_size_history: Vec::new(),
                recommendations: Vec::new(),
            });
        }

        let current_batch_size = self.metrics_history.back().unwrap().batch_size;
        let batch_size_history = self.build_batch_size_history();

        let batch_size_efficiency = self.calculate_batch_size_efficiency(&batch_size_history);
        let gradient_noise_level = self.estimate_gradient_noise_level();
        let convergence_speed = self.estimate_convergence_speed();
        let memory_utilization = self.estimate_memory_utilization(current_batch_size);
        let optimal_batch_size_estimate = self.estimate_optimal_batch_size(&batch_size_history);
        let recommendations =
            self.generate_batch_size_recommendations(current_batch_size, &batch_size_history);

        Ok(BatchSizeAnalysis {
            current_batch_size,
            batch_size_efficiency,
            gradient_noise_level,
            convergence_speed,
            memory_utilization,
            optimal_batch_size_estimate,
            batch_size_history,
            recommendations,
        })
    }

    /// Build batch size history
    fn build_batch_size_history(&self) -> Vec<BatchSizePoint> {
        let mut history = Vec::new();

        for (i, metrics) in self.metrics_history.iter().enumerate() {
            let loss_improvement = if i > 0 {
                self.metrics_history[i - 1].train_loss - metrics.train_loss
            } else {
                0.0
            };

            let gradient_stability =
                metrics.gradient_norm.map(|gn| 1.0 / (1.0 + gn)).unwrap_or(0.5);
            let throughput = 1.0; // Simplified throughput metric

            history.push(BatchSizePoint {
                epoch: metrics.epoch,
                batch_size: metrics.batch_size,
                loss_improvement,
                gradient_stability,
                throughput,
            });
        }

        history
    }

    /// Calculate batch size efficiency
    fn calculate_batch_size_efficiency(&self, batch_history: &[BatchSizePoint]) -> f32 {
        if batch_history.is_empty() {
            return 0.0;
        }

        let avg_improvement =
            batch_history.iter().map(|p| p.loss_improvement.max(0.0)).sum::<f32>()
                / batch_history.len() as f32;

        let avg_stability = batch_history.iter().map(|p| p.gradient_stability).sum::<f32>()
            / batch_history.len() as f32;

        (avg_improvement * 0.6 + avg_stability * 0.4).min(1.0)
    }

    /// Estimate gradient noise level
    fn estimate_gradient_noise_level(&self) -> f32 {
        let gradient_norms: Vec<f32> =
            self.metrics_history.iter().filter_map(|m| m.gradient_norm).collect();

        if gradient_norms.is_empty() {
            return 0.5;
        }

        let std = self.calculate_std(&gradient_norms);
        let mean = gradient_norms.iter().sum::<f32>() / gradient_norms.len() as f32;

        if mean > 1e-8 {
            (std / mean).min(1.0)
        } else {
            0.5
        }
    }

    /// Estimate convergence speed
    fn estimate_convergence_speed(&self) -> f32 {
        let losses: Vec<f32> = self.metrics_history.iter().map(|m| m.train_loss).collect();

        if losses.len() < 2 {
            return 0.0;
        }

        let improvement_per_epoch = (losses[0] - losses[losses.len() - 1]) / losses.len() as f32;
        improvement_per_epoch.max(0.0).min(1.0)
    }

    /// Estimate memory utilization
    fn estimate_memory_utilization(&self, batch_size: usize) -> f32 {
        // Simplified memory utilization based on batch size
        let normalized_batch_size = batch_size as f32 / 1024.0; // Normalize by typical large batch size
        normalized_batch_size.min(1.0)
    }

    /// Estimate optimal batch size
    fn estimate_optimal_batch_size(&self, batch_history: &[BatchSizePoint]) -> usize {
        if batch_history.is_empty() {
            return 32;
        }

        // Find batch size with best balance of improvement and stability
        batch_history
            .iter()
            .max_by(|a, b| {
                let score_a = a.loss_improvement * 0.6 + a.gradient_stability * 0.4;
                let score_b = b.loss_improvement * 0.6 + b.gradient_stability * 0.4;
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|p| p.batch_size)
            .unwrap_or(32)
    }

    /// Generate batch size recommendations
    fn generate_batch_size_recommendations(
        &self,
        current_batch_size: usize,
        _batch_history: &[BatchSizePoint],
    ) -> Vec<BatchSizeRecommendation> {
        let mut recommendations = Vec::new();

        if current_batch_size < 16 {
            recommendations.push(BatchSizeRecommendation {
                suggested_batch_size: 32,
                confidence: 0.7,
                rationale: "Very small batch size may lead to noisy gradients".to_string(),
                expected_benefits: vec![
                    "More stable gradients".to_string(),
                    "Better convergence".to_string(),
                ],
            });
        } else if current_batch_size > 512 {
            recommendations.push(BatchSizeRecommendation {
                suggested_batch_size: 256,
                confidence: 0.6,
                rationale: "Large batch size may slow convergence".to_string(),
                expected_benefits: vec![
                    "Faster convergence".to_string(),
                    "Lower memory usage".to_string(),
                ],
            });
        }

        recommendations
    }

    /// Detect convergence
    async fn detect_convergence(&self) -> Result<ConvergenceAnalysis> {
        if self.metrics_history.len() < self.config.min_epochs_for_convergence {
            return Ok(ConvergenceAnalysis {
                convergence_status: ConvergenceStatus::TooEarly,
                convergence_probability: 0.0,
                epochs_to_convergence_estimate: None,
                convergence_criteria: Vec::new(),
                early_stopping_recommendation: None,
            });
        }

        let convergence_criteria = self.evaluate_convergence_criteria();
        let convergence_status = self.determine_convergence_status(&convergence_criteria);
        let convergence_probability = self.calculate_convergence_probability(&convergence_criteria);
        let epochs_to_convergence_estimate = self.estimate_epochs_to_convergence();
        let early_stopping_recommendation =
            self.generate_early_stopping_recommendation(&convergence_criteria);

        Ok(ConvergenceAnalysis {
            convergence_status,
            convergence_probability,
            epochs_to_convergence_estimate,
            convergence_criteria,
            early_stopping_recommendation,
        })
    }

    /// Evaluate convergence criteria
    fn evaluate_convergence_criteria(&self) -> Vec<ConvergenceCriterion> {
        let mut criteria = Vec::new();

        // Loss stability criterion
        let losses: Vec<f32> = self.metrics_history.iter().map(|m| m.train_loss).collect();
        let recent_window = 10.min(losses.len());
        let recent_losses = &losses[losses.len() - recent_window..];
        let loss_std = self.calculate_std(recent_losses);
        let loss_mean = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
        let loss_stability = loss_std / loss_mean.abs().max(1e-8);

        criteria.push(ConvergenceCriterion {
            criterion_type: ConvergenceCriterionType::LossStability,
            current_value: loss_stability,
            threshold: self.config.convergence_tolerance,
            satisfied: loss_stability < self.config.convergence_tolerance,
            confidence: 0.8,
        });

        // Gradient magnitude criterion
        if let Some(recent_grad_norm) = self.metrics_history.back().and_then(|m| m.gradient_norm) {
            criteria.push(ConvergenceCriterion {
                criterion_type: ConvergenceCriterionType::GradientMagnitude,
                current_value: recent_grad_norm,
                threshold: 1e-4,
                satisfied: recent_grad_norm < 1e-4,
                confidence: 0.7,
            });
        }

        // Loss improvement criterion
        if losses.len() >= 10 {
            let old_window = &losses[losses.len() - 20..losses.len() - 10];
            let new_window = &losses[losses.len() - 10..];
            let old_mean = old_window.iter().sum::<f32>() / old_window.len() as f32;
            let new_mean = new_window.iter().sum::<f32>() / new_window.len() as f32;
            let improvement = (old_mean - new_mean) / old_mean.abs().max(1e-8);

            criteria.push(ConvergenceCriterion {
                criterion_type: ConvergenceCriterionType::LossImprovement,
                current_value: improvement,
                threshold: 1e-3,
                satisfied: improvement < 1e-3,
                confidence: 0.6,
            });
        }

        criteria
    }

    /// Determine convergence status
    fn determine_convergence_status(&self, criteria: &[ConvergenceCriterion]) -> ConvergenceStatus {
        let satisfied_count = criteria.iter().filter(|c| c.satisfied).count();
        let total_count = criteria.len();

        if total_count == 0 {
            return ConvergenceStatus::TooEarly;
        }

        let satisfaction_rate = satisfied_count as f32 / total_count as f32;

        if satisfaction_rate > 0.8 {
            ConvergenceStatus::Converged
        } else if satisfaction_rate > 0.5 {
            ConvergenceStatus::Converging
        } else {
            // Check for divergence
            let losses: Vec<f32> = self.metrics_history.iter().map(|m| m.train_loss).collect();
            let recent_trend =
                self.calculate_linear_trend(&losses[losses.len().saturating_sub(20)..]);

            if recent_trend > 0.01 {
                ConvergenceStatus::Diverging
            } else {
                ConvergenceStatus::Oscillating
            }
        }
    }

    /// Calculate convergence probability
    fn calculate_convergence_probability(&self, criteria: &[ConvergenceCriterion]) -> f32 {
        if criteria.is_empty() {
            return 0.0;
        }

        let weighted_satisfaction: f32 =
            criteria.iter().map(|c| if c.satisfied { c.confidence } else { 0.0 }).sum();

        let total_weight: f32 = criteria.iter().map(|c| c.confidence).sum();

        if total_weight > 0.0 {
            weighted_satisfaction / total_weight
        } else {
            0.0
        }
    }

    /// Estimate epochs to convergence
    fn estimate_epochs_to_convergence(&self) -> Option<usize> {
        let losses: Vec<f32> = self.metrics_history.iter().map(|m| m.train_loss).collect();

        if losses.len() < 5 {
            return None;
        }

        let improvement_rate = self.calculate_improvement_rate(&losses);

        if improvement_rate <= 0.0 {
            return None;
        }

        let current_loss = *losses.last().unwrap();
        let target_loss = current_loss * (1.0 - self.config.convergence_tolerance);
        let remaining_improvement = current_loss - target_loss;

        let epochs_needed = (remaining_improvement / improvement_rate).ceil() as usize;

        Some(epochs_needed.min(1000)) // Cap at reasonable number
    }

    /// Generate early stopping recommendation
    fn generate_early_stopping_recommendation(
        &self,
        criteria: &[ConvergenceCriterion],
    ) -> Option<EarlyStoppingRecommendation> {
        let convergence_probability = self.calculate_convergence_probability(criteria);

        if convergence_probability > 0.9 {
            Some(EarlyStoppingRecommendation {
                should_stop: true,
                confidence: convergence_probability,
                rationale: "High convergence probability detected".to_string(),
                suggested_epochs_remaining: 0,
            })
        } else if convergence_probability > 0.7 {
            Some(EarlyStoppingRecommendation {
                should_stop: false,
                confidence: convergence_probability,
                rationale: "Approaching convergence, continue for a few more epochs".to_string(),
                suggested_epochs_remaining: 5,
            })
        } else {
            None
        }
    }

    /// Identify plateaus
    async fn identify_plateau(&self) -> Result<PlateauAnalysis> {
        let losses: Vec<f32> = self.metrics_history.iter().map(|m| m.train_loss).collect();

        if losses.len() < 10 {
            return Ok(PlateauAnalysis {
                plateau_detected: false,
                plateau_duration: 0,
                plateau_level: 0.0,
                plateau_type: PlateauType::LossPlayteau,
                escape_probability: 0.0,
                plateau_characteristics: PlateauCharacteristics {
                    stability: 0.0,
                    noise_level: 0.0,
                    gradient_magnitude: 0.0,
                    overfitting_risk: 0.0,
                },
                recommendations: Vec::new(),
            });
        }

        let window_size = 10.min(losses.len());
        let recent_losses = &losses[losses.len() - window_size..];

        let plateau_detected = self.detect_plateau_in_window(recent_losses);
        let plateau_duration =
            if plateau_detected { self.calculate_plateau_duration(&losses) } else { 0 };

        let plateau_level = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
        let plateau_type = PlateauType::LossPlayteau; // Simplified
        let escape_probability =
            self.estimate_plateau_escape_probability(&losses, plateau_duration);
        let plateau_characteristics = self.analyze_plateau_characteristics(recent_losses);
        let recommendations =
            self.generate_plateau_recommendations(plateau_detected, plateau_duration);

        Ok(PlateauAnalysis {
            plateau_detected,
            plateau_duration,
            plateau_level,
            plateau_type,
            escape_probability,
            plateau_characteristics,
            recommendations,
        })
    }

    /// Detect plateau in a window of values
    fn detect_plateau_in_window(&self, values: &[f32]) -> bool {
        if values.len() < 3 {
            return false;
        }

        let std = self.calculate_std(values);
        let mean = values.iter().sum::<f32>() / values.len() as f32;

        std / mean.abs().max(1e-8) < self.config.plateau_threshold
    }

    /// Calculate plateau duration
    fn calculate_plateau_duration(&self, losses: &[f32]) -> usize {
        let threshold = self.config.plateau_threshold;
        let mut duration = 0;

        for window in losses.windows(10).rev() {
            let std = self.calculate_std(window);
            let mean = window.iter().sum::<f32>() / window.len() as f32;

            if std / mean.abs().max(1e-8) < threshold {
                duration += 1;
            } else {
                break;
            }
        }

        duration
    }

    /// Estimate plateau escape probability
    fn estimate_plateau_escape_probability(&self, losses: &[f32], plateau_duration: usize) -> f32 {
        if plateau_duration == 0 {
            return 1.0;
        }

        // Longer plateaus are harder to escape
        let duration_factor = 1.0 / (1.0 + plateau_duration as f32 * 0.1);

        // Recent trend might indicate escape potential
        let recent_trend = if losses.len() >= 5 {
            self.calculate_linear_trend(&losses[losses.len() - 5..])
        } else {
            0.0
        };

        let trend_factor = if recent_trend < 0.0 { 0.8 } else { 0.3 };

        (duration_factor * trend_factor).max(0.1).min(0.9)
    }

    /// Analyze plateau characteristics
    fn analyze_plateau_characteristics(&self, plateau_values: &[f32]) -> PlateauCharacteristics {
        let stability = 1.0 - self.calculate_std(plateau_values);
        let noise_level = self.calculate_std(plateau_values);

        let gradient_magnitude =
            self.metrics_history.back().and_then(|m| m.gradient_norm).unwrap_or(0.0);

        // Simple overfitting risk estimation
        let overfitting_risk =
            if let Some(val_loss) = self.metrics_history.back().and_then(|m| m.validation_loss) {
                let train_loss = self.metrics_history.back().unwrap().train_loss;
                ((val_loss - train_loss) / train_loss.abs().max(1e-8)).max(0.0).min(1.0)
            } else {
                0.5
            };

        PlateauCharacteristics {
            stability: stability.max(0.0).min(1.0),
            noise_level: noise_level.min(1.0),
            gradient_magnitude,
            overfitting_risk,
        }
    }

    /// Generate plateau recommendations
    fn generate_plateau_recommendations(
        &self,
        plateau_detected: bool,
        plateau_duration: usize,
    ) -> Vec<PlateauRecommendation> {
        let mut recommendations = Vec::new();

        if !plateau_detected {
            return recommendations;
        }

        if plateau_duration > 20 {
            recommendations.push(PlateauRecommendation {
                action: PlateauAction::IncreaseLearningRate,
                priority: Priority::High,
                description: "Long plateau detected, consider increasing learning rate".to_string(),
                implementation: "Multiply current learning rate by 2-5x temporarily".to_string(),
            });
        } else if plateau_duration > 10 {
            recommendations.push(PlateauRecommendation {
                action: PlateauAction::ChangeBatchSize,
                priority: Priority::Medium,
                description: "Moderate plateau detected, try changing batch size".to_string(),
                implementation: "Increase or decrease batch size by 50%".to_string(),
            });
        }

        if plateau_duration > 30 {
            recommendations.push(PlateauRecommendation {
                action: PlateauAction::EarlyStopping,
                priority: Priority::Critical,
                description: "Very long plateau, consider early stopping".to_string(),
                implementation: "Stop training and use best checkpoint".to_string(),
            });
        }

        recommendations
    }

    /// Generate training summary
    fn generate_training_summary(&self, report: &mut TrainingDynamicsReport) {
        let total_epochs = self.metrics_history.back().map(|m| m.epoch).unwrap_or(0);
        let total_steps = self.metrics_history.back().map(|m| m.step).unwrap_or(0);

        let training_efficiency = if let Some(loss_analysis) = &report.loss_curve_analysis {
            loss_analysis.improvement_rate.max(0.0).min(1.0)
        } else {
            0.0
        };

        let convergence_health = if let Some(conv_analysis) = &report.convergence_analysis {
            conv_analysis.convergence_probability
        } else {
            0.0
        };

        let stability_score = if let Some(loss_analysis) = &report.loss_curve_analysis {
            loss_analysis.smoothness
        } else {
            0.0
        };

        let overall_progress =
            (training_efficiency * 0.4 + convergence_health * 0.3 + stability_score * 0.3)
                .max(0.0)
                .min(1.0);

        report.training_summary = TrainingSummary {
            total_epochs,
            total_steps,
            training_efficiency,
            convergence_health,
            stability_score,
            overall_progress,
        };
    }

    /// Generate training recommendations
    fn generate_training_recommendations(&self, report: &mut TrainingDynamicsReport) {
        let mut recommendations = Vec::new();

        // Learning rate recommendations
        if let Some(lr_analysis) = &report.learning_rate_analysis {
            for lr_rec in &lr_analysis.recommendations {
                recommendations.push(TrainingRecommendation {
                    category: TrainingCategory::LearningRate,
                    priority: if lr_rec.confidence > 0.8 {
                        Priority::High
                    } else {
                        Priority::Medium
                    },
                    description: lr_rec.rationale.clone(),
                    implementation: format!("{:?} learning rate", lr_rec.action),
                    expected_impact: lr_rec.expected_improvement,
                });
            }
        }

        // Plateau recommendations
        if let Some(plateau_analysis) = &report.plateau_analysis {
            for plateau_rec in &plateau_analysis.recommendations {
                recommendations.push(TrainingRecommendation {
                    category: TrainingCategory::Optimization,
                    priority: plateau_rec.priority.clone(),
                    description: plateau_rec.description.clone(),
                    implementation: plateau_rec.implementation.clone(),
                    expected_impact: 0.5, // Default impact
                });
            }
        }

        // Convergence recommendations
        if let Some(conv_analysis) = &report.convergence_analysis {
            if let Some(early_stop) = &conv_analysis.early_stopping_recommendation {
                if early_stop.should_stop {
                    recommendations.push(TrainingRecommendation {
                        category: TrainingCategory::EarlyStopping,
                        priority: Priority::High,
                        description: early_stop.rationale.clone(),
                        implementation: "Stop training and save current model".to_string(),
                        expected_impact: 0.8,
                    });
                }
            }
        }

        report.recommendations = recommendations;
    }

    /// Generate a comprehensive report
    pub async fn generate_report(&self) -> Result<TrainingDynamicsReport> {
        let mut temp_analyzer = TrainingDynamicsAnalyzer {
            config: self.config.clone(),
            metrics_history: self.metrics_history.clone(),
            analysis_cache: HashMap::new(),
        };

        temp_analyzer.analyze().await
    }

    /// Clear all recorded metrics
    pub fn clear(&mut self) {
        self.metrics_history.clear();
        self.analysis_cache.clear();
    }

    /// Get summary of current training state
    pub fn get_training_summary(&self) -> TrainingStateSummary {
        let current_metrics = self.metrics_history.back();

        TrainingStateSummary {
            total_epochs: current_metrics.map(|m| m.epoch).unwrap_or(0),
            total_steps: current_metrics.map(|m| m.step).unwrap_or(0),
            current_loss: current_metrics.map(|m| m.train_loss).unwrap_or(0.0),
            current_lr: current_metrics.map(|m| m.learning_rate).unwrap_or(0.0),
            metrics_collected: self.metrics_history.len(),
        }
    }
}

/// Summary of current training state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStateSummary {
    pub total_epochs: usize,
    pub total_steps: usize,
    pub current_loss: f32,
    pub current_lr: f32,
    pub metrics_collected: usize,
}
