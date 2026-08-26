//! Dynamic data augmentation: strategies, per-modality augmentation types, adaptive configuration and scheduling.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    /// Performance-based adaptation
    PerformanceBased {
        target_metric: String,
        threshold: f64,
    },
    /// Loss-based adaptation
    LossBased { loss_threshold: f64 },
    /// Gradient-based adaptation
    GradientBased { gradient_threshold: f64 },
    /// Uncertainty-based adaptation
    UncertaintyBased { uncertainty_threshold: f64 },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveAugmentationConfig {
    /// Enable adaptive augmentation
    pub enabled: bool,
    /// Adaptation strategy
    pub strategy: AdaptationStrategy,
    /// Update frequency
    pub update_frequency: usize,
    /// Performance metrics to track
    pub metrics: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioAugmentationType {
    /// Noise injection
    NoiseInjection,
    /// Time stretching
    TimeStretching,
    /// Pitch shifting
    PitchShifting,
    /// Volume adjustment
    VolumeAdjustment,
    /// Speed perturbation
    SpeedPerturbation,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationScheduling {
    /// Scheduling type
    pub schedule_type: ScheduleType,
    /// Schedule parameters
    pub parameters: HashMap<String, f64>,
}
#[derive(Debug, Clone)]
pub struct AugmentationStats {
    pub augmentations_applied: HashMap<String, usize>,
    pub processing_time: Duration,
    pub performance_impact: HashMap<String, f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationStrategy {
    /// Strategy name
    pub name: String,
    /// Strategy type
    pub strategy_type: AugmentationStrategyType,
    /// Probability of applying this augmentation
    pub probability: f64,
    /// Intensity parameter
    pub intensity: f64,
    /// Strategy-specific parameters
    pub parameters: HashMap<String, f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AugmentationStrategyType {
    /// Text augmentations
    Text {
        augmentation_type: TextAugmentationType,
    },
    /// Image augmentations
    Image {
        augmentation_type: ImageAugmentationType,
    },
    /// Audio augmentations
    Audio {
        augmentation_type: AudioAugmentationType,
    },
    /// Token-level augmentations
    Token {
        augmentation_type: TokenAugmentationType,
    },
}
/// Dynamic data augmentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicAugmentationConfig {
    /// Augmentation strategies
    pub strategies: Vec<AugmentationStrategy>,
    /// Adaptive augmentation settings
    pub adaptive: AdaptiveAugmentationConfig,
    /// Augmentation scheduling
    pub scheduling: AugmentationScheduling,
}
pub struct DynamicAugmentationManager {
    pub config: DynamicAugmentationConfig,
    pub strategies: Vec<AugmentationStrategy>,
    pub stats: AugmentationStats,
}
impl DynamicAugmentationManager {
    pub fn new() -> Self {
        Self {
            config: DynamicAugmentationConfig {
                strategies: vec![],
                adaptive: AdaptiveAugmentationConfig {
                    enabled: false,
                    strategy: AdaptationStrategy::PerformanceBased {
                        target_metric: "accuracy".to_string(),
                        threshold: 0.8,
                    },
                    update_frequency: 100,
                    metrics: vec!["accuracy".to_string()],
                },
                scheduling: AugmentationScheduling {
                    schedule_type: ScheduleType::Fixed,
                    parameters: HashMap::new(),
                },
            },
            strategies: vec![],
            stats: AugmentationStats {
                augmentations_applied: HashMap::new(),
                processing_time: Duration::from_secs(0),
                performance_impact: HashMap::new(),
            },
        }
    }
}
impl Default for DynamicAugmentationManager {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageAugmentationType {
    /// Rotation
    Rotation,
    /// Scaling
    Scaling,
    /// Translation
    Translation,
    /// Color jittering
    ColorJitter,
    /// Gaussian noise
    GaussianNoise,
    /// Cutout
    Cutout,
    /// Mixup
    Mixup,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleType {
    /// Fixed schedule
    Fixed,
    /// Linear schedule
    Linear {
        start_value: f64,
        end_value: f64,
        total_steps: usize,
    },
    /// Exponential schedule
    Exponential { initial_value: f64, decay_rate: f64 },
    /// Cosine schedule
    Cosine {
        max_value: f64,
        min_value: f64,
        period: usize,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextAugmentationType {
    /// Synonym replacement
    SynonymReplacement,
    /// Random insertion
    RandomInsertion,
    /// Random swap
    RandomSwap,
    /// Random deletion
    RandomDeletion,
    /// Back translation
    BackTranslation { target_language: String },
    /// Paraphrasing
    Paraphrasing,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenAugmentationType {
    /// Token dropout
    TokenDropout,
    /// Token replacement
    TokenReplacement,
    /// Token insertion
    TokenInsertion,
    /// Span masking
    SpanMasking,
}
