//! Curriculum learning: stage progression, data selection/filtering, difficulty assessment and pacing.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Curriculum learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumLearningConfig {
    /// Curriculum strategy
    pub strategy: CurriculumStrategy,
    /// Difficulty assessment
    pub difficulty_assessment: DifficultyAssessment,
    /// Pacing function
    pub pacing: PacingFunction,
    /// Curriculum scheduling
    pub scheduling: CurriculumScheduling,
}
pub struct CurriculumLearningManager {
    pub config: CurriculumLearningConfig,
    pub current_stage: usize,
    pub stats: CurriculumStats,
}
impl CurriculumLearningManager {
    pub fn new() -> Self {
        Self {
            config: CurriculumLearningConfig {
                strategy: CurriculumStrategy::Manual { stages: vec![] },
                difficulty_assessment: DifficultyAssessment::Static {
                    score_field: "difficulty".to_string(),
                },
                pacing: PacingFunction {
                    pacing_type: PacingType::Linear,
                    parameters: HashMap::new(),
                },
                scheduling: CurriculumScheduling {
                    strategy: CurriculumSchedulingStrategy::EpochBased,
                    update_frequency: 1,
                },
            },
            current_stage: 0,
            stats: CurriculumStats {
                current_difficulty: 0.0,
                stage_progress: 0.0,
                competency_scores: HashMap::new(),
            },
        }
    }
}
impl Default for CurriculumLearningManager {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumScheduling {
    /// Scheduling strategy
    pub strategy: CurriculumSchedulingStrategy,
    /// Update frequency
    pub update_frequency: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CurriculumSchedulingStrategy {
    /// Epoch-based scheduling
    EpochBased,
    /// Step-based scheduling
    StepBased,
    /// Performance-based scheduling
    PerformanceBased { trigger_metric: String },
    /// Time-based scheduling
    TimeBased { interval: Duration },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumStage {
    /// Stage name
    pub name: String,
    /// Data selection criteria
    pub criteria: DataSelectionCriteria,
    /// Duration in epochs
    pub duration_epochs: usize,
    /// Success criteria to move to next stage
    pub success_criteria: SuccessCriteria,
}
#[derive(Debug, Clone)]
pub struct CurriculumStats {
    pub current_difficulty: f64,
    pub stage_progress: f64,
    pub competency_scores: HashMap<String, f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CurriculumStrategy {
    /// Manual curriculum with predefined stages
    Manual { stages: Vec<CurriculumStage> },
    /// Automatic curriculum based on model performance
    Automatic {
        difficulty_increase_threshold: f64,
        competency_threshold: f64,
    },
    /// Self-paced curriculum
    SelfPaced { lambda: f64 },
    /// Anti-curriculum (hard to easy)
    AntiCurriculum,
    /// Random curriculum
    Random,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFilter {
    /// Filter type
    pub filter_type: FilterType,
    /// Filter parameters
    pub parameters: HashMap<String, String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSelectionCriteria {
    /// Difficulty range
    pub difficulty_range: (f64, f64),
    /// Quality threshold
    pub quality_threshold: f64,
    /// Data filters
    pub filters: Vec<DataFilter>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifficultyAssessment {
    /// Static difficulty scores
    Static { score_field: String },
    /// Dynamic difficulty based on model performance
    Dynamic {
        assessment_method: DynamicAssessmentMethod,
    },
    /// Learned difficulty function
    Learned { model_path: String },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DynamicAssessmentMethod {
    /// Loss-based difficulty
    LossBased,
    /// Gradient-based difficulty
    GradientBased,
    /// Uncertainty-based difficulty
    UncertaintyBased,
    /// Attention-based difficulty
    AttentionBased,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterType {
    /// Length-based filter
    Length {
        min_length: usize,
        max_length: usize,
    },
    /// Complexity-based filter
    Complexity { complexity_metric: String },
    /// Topic-based filter
    Topic { topics: Vec<String> },
    /// Language-based filter
    Language { languages: Vec<String> },
    /// Custom filter
    Custom { filter_name: String },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacingFunction {
    /// Pacing type
    pub pacing_type: PacingType,
    /// Pacing parameters
    pub parameters: HashMap<String, f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PacingType {
    /// Linear pacing
    Linear,
    /// Exponential pacing
    Exponential,
    /// Root pacing
    Root,
    /// Logarithmic pacing
    Logarithmic,
    /// Custom pacing function
    Custom { function_name: String },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    /// Success metric
    pub metric: String,
    /// Target value
    pub target_value: f64,
    /// Minimum epochs before advancement
    pub min_epochs: usize,
    /// Patience for achieving target
    pub patience: usize,
}
