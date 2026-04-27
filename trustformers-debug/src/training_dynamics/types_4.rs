//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};

use super::types::{
    BatchSizePoint, ConvergenceCriterionType, EarlyStoppingRecommendation, LRAction,
    LearningRatePoint, MovingAverages, TrainingCategory,
};

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
pub struct TrainingRecommendation {
    pub category: TrainingCategory,
    pub priority: Priority,
    pub description: String,
    pub implementation: String,
    pub expected_impact: f32,
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
pub struct LRRecommendation {
    pub action: LRAction,
    pub confidence: f32,
    pub rationale: String,
    pub expected_improvement: f32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvergenceStatus {
    Converging,
    Converged,
    Diverging,
    Oscillating,
    TooEarly,
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
pub enum LossTrend {
    Decreasing,
    Increasing,
    Oscillating,
    Plateaued,
    Unknown,
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
pub struct ConvergenceCriterion {
    pub criterion_type: ConvergenceCriterionType,
    pub current_value: f32,
    pub threshold: f32,
    pub satisfied: bool,
    pub confidence: f32,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSizeRecommendation {
    pub suggested_batch_size: usize,
    pub confidence: f32,
    pub rationale: String,
    pub expected_benefits: Vec<String>,
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
