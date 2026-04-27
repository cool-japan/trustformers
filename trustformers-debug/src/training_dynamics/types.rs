//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};

use super::types_4::{
    BatchSizeAnalysis, ConvergenceAnalysis, LearningRateAnalysis, LossCurveAnalysis, PlateauAction,
    PlateauCharacteristics, PlateauType, Priority, TrainingRecommendation, TrainingSummary,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRatePoint {
    pub epoch: usize,
    pub learning_rate: f32,
    pub loss_change: f32,
    pub gradient_norm: Option<f32>,
    pub effectiveness: f32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovingAverages {
    pub short_term: f32,
    pub medium_term: f32,
    pub long_term: f32,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingCategory {
    LearningRate,
    BatchSize,
    Optimization,
    Regularization,
    EarlyStopping,
    Architecture,
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
pub struct PlateauRecommendation {
    pub action: PlateauAction,
    pub priority: Priority,
    pub description: String,
    pub implementation: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingRecommendation {
    pub should_stop: bool,
    pub confidence: f32,
    pub rationale: String,
    pub suggested_epochs_remaining: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSizePoint {
    pub epoch: usize,
    pub batch_size: usize,
    pub loss_improvement: f32,
    pub gradient_stability: f32,
    pub throughput: f32,
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
