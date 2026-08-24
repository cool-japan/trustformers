//! Multi-modal data handling: per-modality preprocessing, normalization, feature extraction, fusion and alignment.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use trustformers_core::tensor::Tensor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentConfig {
    /// Alignment method
    pub method: AlignmentMethod,
    /// Temporal alignment for time-series modalities
    pub temporal_alignment: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlignmentMethod {
    /// Timestamp-based alignment
    Timestamp,
    /// Learned alignment
    Learned,
    /// Manual alignment
    Manual {
        alignment_map: HashMap<String, String>,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionConfig {
    /// Extraction method
    pub method: FeatureExtractionMethod,
    /// Output dimension
    pub output_dim: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureExtractionMethod {
    /// Pre-trained model
    PretrainedModel { model_path: String },
    /// Custom extraction
    Custom { extractor_name: String },
    /// Raw features
    Raw,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Early fusion (feature level)
    EarlyFusion,
    /// Late fusion (decision level)
    LateFusion,
    /// Intermediate fusion
    IntermediateFusion { fusion_layers: Vec<usize> },
    /// Attention-based fusion
    AttentionFusion,
    /// Cross-modal attention
    CrossModalAttention,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MissingModalityHandling {
    /// Skip samples with missing modalities
    Skip,
    /// Use default values
    DefaultValue,
    /// Impute missing modalities
    Impute { imputation_method: String },
    /// Train separate models
    SeparateModels,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Modality {
    /// Modality type
    pub modality_type: ModalityType,
    /// Preprocessing configuration
    pub preprocessing: PreprocessingConfig,
    /// Feature extraction
    pub feature_extraction: FeatureExtractionConfig,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModalityType {
    Text,
    Image,
    Audio,
    Video,
    Tabular,
    Graph,
    Custom { modality_name: String },
}
/// Multi-modal data handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalConfig {
    /// Supported modalities
    pub modalities: Vec<Modality>,
    /// Fusion strategy
    pub fusion_strategy: FusionStrategy,
    /// Alignment configuration
    pub alignment: AlignmentConfig,
    /// Preprocessing configuration
    pub preprocessing: MultiModalPreprocessing,
}
pub struct MultiModalHandler {
    pub config: MultiModalConfig,
    pub modality_processors: HashMap<String, Box<dyn ModalityProcessor>>,
    pub stats: MultiModalStats,
}
impl MultiModalHandler {
    pub fn new() -> Self {
        Self {
            config: MultiModalConfig {
                modalities: vec![],
                fusion_strategy: FusionStrategy::EarlyFusion,
                alignment: AlignmentConfig {
                    method: AlignmentMethod::Timestamp,
                    temporal_alignment: false,
                },
                preprocessing: MultiModalPreprocessing {
                    synchronization: SynchronizationConfig {
                        require_all: true,
                        sync_window: Duration::from_secs(1),
                    },
                    missing_modality_handling: MissingModalityHandling::Skip,
                },
            },
            modality_processors: HashMap::new(),
            stats: MultiModalStats {
                modalities_processed: HashMap::new(),
                fusion_efficiency: 0.0,
                alignment_accuracy: 0.0,
            },
        }
    }
}
impl Default for MultiModalHandler {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalPreprocessing {
    /// Synchronization requirements
    pub synchronization: SynchronizationConfig,
    /// Missing modality handling
    pub missing_modality_handling: MissingModalityHandling,
}
#[derive(Debug, Clone)]
pub struct MultiModalStats {
    pub modalities_processed: HashMap<String, usize>,
    pub fusion_efficiency: f64,
    pub alignment_accuracy: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    /// Normalization type
    pub normalization_type: NormalizationType,
    /// Parameters
    pub parameters: HashMap<String, f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationType {
    /// Min-max normalization
    MinMax,
    /// Z-score normalization
    ZScore,
    /// Robust normalization
    Robust,
    /// Unit normalization
    Unit,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Preprocessing steps
    pub steps: Vec<PreprocessingStep>,
    /// Normalization
    pub normalization: NormalizationConfig,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingStep {
    /// Step name
    pub name: String,
    /// Step type
    pub step_type: PreprocessingStepType,
    /// Parameters
    pub parameters: HashMap<String, String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreprocessingStepType {
    /// Tokenization
    Tokenization,
    /// Resize
    Resize,
    /// Crop
    Crop,
    /// Filter
    Filter,
    /// Transform
    Transform,
    /// Custom step
    Custom { step_name: String },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationConfig {
    /// Require all modalities
    pub require_all: bool,
    /// Synchronization window
    pub sync_window: Duration,
}
pub trait ModalityProcessor: Send + Sync {
    fn process(&self, data: &Tensor) -> Result<Tensor>;
    fn get_features(&self, data: &Tensor) -> Result<Tensor>;
}
