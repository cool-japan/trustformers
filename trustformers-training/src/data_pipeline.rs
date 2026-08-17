//! Data Pipeline Enhancements for TrustformeRS Training
//!
//! This module provides advanced data pipeline capabilities including streaming datasets,
//! dynamic augmentation, curriculum learning, active learning, and multi-modal data handling.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use trustformers_core::tensor::Tensor;

/// Streaming dataset configuration and management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingDatasetConfig {
    /// Data sources for streaming
    pub sources: Vec<DataSource>,
    /// Buffer size for streaming
    pub buffer_size: usize,
    /// Prefetch buffer size
    pub prefetch_size: usize,
    /// Shuffle configuration
    pub shuffle: ShuffleConfig,
    /// Batching configuration
    pub batching: BatchingConfig,
    /// Caching configuration
    pub caching: CachingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSource {
    /// Source identifier
    pub id: String,
    /// Source type
    pub source_type: DataSourceType,
    /// Source-specific configuration
    pub config: HashMap<String, String>,
    /// Weight for sampling from this source
    pub weight: f64,
    /// Quality score
    pub quality_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSourceType {
    /// Local file system
    LocalFiles { patterns: Vec<String> },
    /// Remote HTTP/HTTPS endpoints
    Http { urls: Vec<String> },
    /// Database connection
    Database { connection_string: String },
    /// Cloud storage (S3, GCS, Azure)
    CloudStorage { bucket: String, prefix: String },
    /// Kafka stream
    Kafka {
        topics: Vec<String>,
        brokers: Vec<String>,
    },
    /// Custom data source
    Custom { source_name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShuffleConfig {
    /// Whether to shuffle data
    pub enabled: bool,
    /// Shuffle buffer size
    pub buffer_size: usize,
    /// Shuffle strategy
    pub strategy: ShuffleStrategy,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShuffleStrategy {
    /// Random shuffle
    Random,
    /// Reservoir sampling
    Reservoir,
    /// Block-wise shuffle
    BlockWise { block_size: usize },
    /// Hash-based shuffle
    HashBased,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchingConfig {
    /// Batch size
    pub batch_size: usize,
    /// Dynamic batching enabled
    pub dynamic: bool,
    /// Maximum batch size for dynamic batching
    pub max_batch_size: usize,
    /// Batching strategy
    pub strategy: BatchingStrategy,
    /// Drop last incomplete batch
    pub drop_last: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchingStrategy {
    /// Fixed size batches
    Fixed,
    /// Variable size based on sequence length
    SequenceLength { max_tokens: usize },
    /// Variable size based on memory usage
    MemoryAware { max_memory_mb: usize },
    /// Adaptive batching based on throughput
    Adaptive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    /// Enable caching
    pub enabled: bool,
    /// Cache type
    pub cache_type: CacheType,
    /// Cache size limit
    pub max_size_gb: f64,
    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Cache compression
    pub compression: CompressionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheType {
    /// In-memory cache
    Memory,
    /// Disk-based cache
    Disk { directory: PathBuf },
    /// Redis cache
    Redis { connection_string: String },
    /// Hybrid memory + disk
    Hybrid { memory_ratio: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In First Out
    FIFO,
    /// Random replacement
    Random,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level
    pub level: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Zstd,
    Lz4,
    Snappy,
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
pub struct AugmentationScheduling {
    /// Scheduling type
    pub schedule_type: ScheduleType,
    /// Schedule parameters
    pub parameters: HashMap<String, f64>,
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
    SelfPaced {
        lambda: f64, // Self-paced regularization parameter
    },
    /// Anti-curriculum (hard to easy)
    AntiCurriculum,
    /// Random curriculum
    Random,
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
pub struct DataFilter {
    /// Filter type
    pub filter_type: FilterType,
    /// Filter parameters
    pub parameters: HashMap<String, String>,
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

/// Active learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveLearningConfig {
    /// Query strategy
    pub query_strategy: QueryStrategy,
    /// Sampling configuration
    pub sampling: SamplingConfig,
    /// Annotation configuration
    pub annotation: AnnotationConfig,
    /// Integration settings
    pub integration: ActiveLearningIntegration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryStrategy {
    /// Uncertainty sampling
    UncertaintySampling {
        uncertainty_measure: UncertaintyMeasure,
    },
    /// Query by committee
    QueryByCommittee {
        committee_size: usize,
        disagreement_measure: DisagreementMeasure,
    },
    /// Expected gradient length
    ExpectedGradientLength,
    /// Bayesian active learning by disagreement
    BALD,
    /// Core-set selection
    CoreSet { selection_method: CoreSetMethod },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintyMeasure {
    /// Least confidence
    LeastConfidence,
    /// Margin sampling
    MarginSampling,
    /// Entropy
    Entropy,
    /// Variation ratios
    VariationRatios,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DisagreementMeasure {
    /// Vote entropy
    VoteEntropy,
    /// KL divergence
    KLDivergence,
    /// Average KL divergence
    AverageKLDivergence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoreSetMethod {
    /// K-center greedy
    KCenterGreedy,
    /// K-means++
    KMeansPlusPlus,
    /// Facility location
    FacilityLocation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Batch size for active learning queries
    pub batch_size: usize,
    /// Sampling budget
    pub budget: usize,
    /// Diversity constraint
    pub diversity_constraint: Option<DiversityConstraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityConstraint {
    /// Diversity measure
    pub measure: DiversityMeasure,
    /// Minimum diversity threshold
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiversityMeasure {
    /// Cosine similarity
    CosineSimilarity,
    /// Euclidean distance
    EuclideanDistance,
    /// Jaccard similarity
    JaccardSimilarity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationConfig {
    /// Annotation source
    pub source: AnnotationSource,
    /// Quality control
    pub quality_control: QualityControl,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnnotationSource {
    /// Human annotators
    Human { annotator_pool: Vec<String> },
    /// Automatic annotation
    Automatic {
        model_path: String,
        confidence_threshold: f64,
    },
    /// Hybrid human + automatic
    Hybrid {
        automatic_threshold: f64,
        human_verification: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityControl {
    /// Multiple annotations per sample
    pub multi_annotation: bool,
    /// Agreement threshold
    pub agreement_threshold: f64,
    /// Quality assessment method
    pub assessment_method: QualityAssessmentMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityAssessmentMethod {
    /// Inter-annotator agreement
    InterAnnotatorAgreement,
    /// Gold standard comparison
    GoldStandard { gold_set_path: String },
    /// Model-based quality assessment
    ModelBased { quality_model_path: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveLearningIntegration {
    /// Update frequency
    pub update_frequency: usize,
    /// Minimum new samples before update
    pub min_new_samples: usize,
    /// Retrain from scratch
    pub retrain_from_scratch: bool,
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
pub struct MultiModalPreprocessing {
    /// Synchronization requirements
    pub synchronization: SynchronizationConfig,
    /// Missing modality handling
    pub missing_modality_handling: MissingModalityHandling,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynchronizationConfig {
    /// Require all modalities
    pub require_all: bool,
    /// Synchronization window
    pub sync_window: Duration,
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

/// Data validation framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataValidationConfig {
    /// Validation rules
    pub rules: Vec<ValidationRule>,
    /// Validation strategy
    pub strategy: ValidationStrategy,
    /// Error handling
    pub error_handling: ErrorHandling,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule type
    pub rule_type: ValidationRuleType,
    /// Severity level
    pub severity: ValidationSeverity,
    /// Parameters
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRuleType {
    /// Schema validation
    Schema,
    /// Range validation
    Range,
    /// Format validation
    Format,
    /// Consistency validation
    Consistency,
    /// Quality validation
    Quality,
    /// Custom validation
    Custom { validator_name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationSeverity {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStrategy {
    /// Validate all data
    All,
    /// Sample-based validation
    Sample { sample_rate: f64 },
    /// Batch-based validation
    Batch { batch_interval: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandling {
    /// Fail on any error
    Strict,
    /// Skip invalid samples
    Skip,
    /// Attempt to fix errors
    Fix,
    /// Log and continue
    LogAndContinue,
}

/// Main data pipeline orchestrator
pub struct DataPipeline {
    /// Pipeline configuration
    config: DataPipelineConfig,
    /// Active streaming datasets
    streaming_datasets: Arc<Mutex<HashMap<String, StreamingDataset>>>,
    /// Dynamic augmentation manager
    augmentation_manager: Arc<Mutex<DynamicAugmentationManager>>,
    /// Curriculum learning manager
    curriculum_manager: Arc<Mutex<CurriculumLearningManager>>,
    /// Active learning manager
    active_learning_manager: Arc<Mutex<ActiveLearningManager>>,
    /// Multi-modal data handler
    multimodal_handler: Arc<Mutex<MultiModalHandler>>,
    /// Data validator
    validator: Arc<Mutex<DataValidator>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPipelineConfig {
    /// Streaming dataset configuration
    pub streaming: StreamingDatasetConfig,
    /// Dynamic augmentation configuration
    pub augmentation: DynamicAugmentationConfig,
    /// Curriculum learning configuration
    pub curriculum: CurriculumLearningConfig,
    /// Active learning configuration
    pub active_learning: ActiveLearningConfig,
    /// Multi-modal configuration
    pub multimodal: MultiModalConfig,
    /// Data validation configuration
    pub validation: DataValidationConfig,
    /// Distributed processing configuration
    pub distributed: DistributedProcessingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedProcessingConfig {
    /// Number of worker processes
    pub num_workers: usize,
    /// Processing backend
    pub backend: ProcessingBackend,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingBackend {
    /// Thread-based processing
    Threading,
    /// Process-based processing
    Multiprocessing,
    /// Ray distributed processing
    Ray { ray_config: HashMap<String, String> },
    /// Dask distributed processing
    Dask {
        dask_config: HashMap<String, String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin
    RoundRobin,
    /// Work-stealing
    WorkStealing,
    /// Dynamic load balancing
    Dynamic,
}

// Implementation structs (simplified for space)
pub struct StreamingDataset {
    pub config: StreamingDatasetConfig,
    pub buffer: VecDeque<DataSample>,
    pub stats: StreamingStats,
    /// Whether [`DataPipeline::get_batch`] may draw from this dataset.
    pub active: bool,
}

pub struct DynamicAugmentationManager {
    pub config: DynamicAugmentationConfig,
    pub strategies: Vec<AugmentationStrategy>,
    pub stats: AugmentationStats,
}

pub struct CurriculumLearningManager {
    pub config: CurriculumLearningConfig,
    pub current_stage: usize,
    pub stats: CurriculumStats,
}

pub struct ActiveLearningManager {
    pub config: ActiveLearningConfig,
    pub query_pool: Vec<DataSample>,
    pub stats: ActiveLearningStats,
}

pub struct MultiModalHandler {
    pub config: MultiModalConfig,
    pub modality_processors: HashMap<String, Box<dyn ModalityProcessor>>,
    pub stats: MultiModalStats,
}

pub struct DataValidator {
    pub config: DataValidationConfig,
    pub validators: Vec<Box<dyn Validator>>,
    pub stats: ValidationStats,
}

#[derive(Debug, Clone)]
pub struct DataSample {
    pub id: String,
    pub data: HashMap<String, Tensor>,
    pub metadata: HashMap<String, String>,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone)]
pub struct StreamingStats {
    pub samples_processed: usize,
    pub bytes_processed: u64,
    pub processing_time: Duration,
    pub error_count: usize,
}

#[derive(Debug, Clone)]
pub struct AugmentationStats {
    pub augmentations_applied: HashMap<String, usize>,
    pub processing_time: Duration,
    pub performance_impact: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct CurriculumStats {
    pub current_difficulty: f64,
    pub stage_progress: f64,
    pub competency_scores: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct ActiveLearningStats {
    pub queries_made: usize,
    pub annotations_received: usize,
    pub model_improvement: f64,
}

#[derive(Debug, Clone)]
pub struct MultiModalStats {
    pub modalities_processed: HashMap<String, usize>,
    pub fusion_efficiency: f64,
    pub alignment_accuracy: f64,
}

#[derive(Debug, Clone)]
pub struct ValidationStats {
    pub samples_validated: usize,
    pub errors_detected: HashMap<String, usize>,
    pub validation_time: Duration,
}

// Traits for extensibility
pub trait ModalityProcessor: Send + Sync {
    fn process(&self, data: &Tensor) -> Result<Tensor>;
    fn get_features(&self, data: &Tensor) -> Result<Tensor>;
}

pub trait Validator: Send + Sync {
    fn validate(&self, sample: &DataSample) -> Result<ValidationResult>;
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
}

#[derive(Debug, Clone)]
pub struct ValidationError {
    pub rule_name: String,
    pub message: String,
    pub severity: ValidationSeverity,
}

#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub rule_name: String,
    pub message: String,
}

impl DataPipeline {
    pub fn new(config: DataPipelineConfig) -> Self {
        Self {
            config,
            streaming_datasets: Arc::new(Mutex::new(HashMap::new())),
            augmentation_manager: Arc::new(Mutex::new(DynamicAugmentationManager::new())),
            curriculum_manager: Arc::new(Mutex::new(CurriculumLearningManager::new())),
            active_learning_manager: Arc::new(Mutex::new(ActiveLearningManager::new())),
            multimodal_handler: Arc::new(Mutex::new(MultiModalHandler::new())),
            validator: Arc::new(Mutex::new(DataValidator::new())),
        }
    }

    /// Register a streaming dataset under `dataset_id`.
    ///
    /// The dataset starts inactive; call [`DataPipeline::start_streaming`] to make it eligible
    /// for [`DataPipeline::get_batch`].
    pub fn register_dataset(&self, dataset_id: &str, config: StreamingDatasetConfig) -> Result<()> {
        let mut datasets = self
            .streaming_datasets
            .lock()
            .map_err(|_| anyhow::anyhow!("streaming dataset registry lock poisoned"))?;
        if datasets.contains_key(dataset_id) {
            return Err(anyhow::anyhow!(
                "dataset '{dataset_id}' is already registered"
            ));
        }
        datasets.insert(dataset_id.to_string(), StreamingDataset::new(config));
        Ok(())
    }

    /// Ids of every registered dataset.
    pub fn dataset_ids(&self) -> Result<Vec<String>> {
        let datasets = self
            .streaming_datasets
            .lock()
            .map_err(|_| anyhow::anyhow!("streaming dataset registry lock poisoned"))?;
        let mut ids: Vec<String> = datasets.keys().cloned().collect();
        ids.sort();
        Ok(ids)
    }

    /// Push samples into a registered dataset's buffer.
    ///
    /// This is how a producer (file reader, network stream, generator, …) hands data to the
    /// pipeline. Samples beyond `buffer_size` are rejected rather than silently dropped.
    pub fn push_samples(&self, dataset_id: &str, samples: Vec<DataSample>) -> Result<usize> {
        let mut datasets = self
            .streaming_datasets
            .lock()
            .map_err(|_| anyhow::anyhow!("streaming dataset registry lock poisoned"))?;
        let dataset = datasets.get_mut(dataset_id).ok_or_else(|| {
            anyhow::anyhow!("unknown dataset '{dataset_id}'; register it with register_dataset")
        })?;
        dataset.push_samples(samples)
    }

    /// Mark a registered dataset as streaming.
    ///
    /// # Errors
    ///
    /// Fails when `dataset_id` names a dataset that was never registered — previously this
    /// returned `Ok(())` for any string at all.
    pub async fn start_streaming(&self, dataset_id: &str) -> Result<()> {
        let mut datasets = self
            .streaming_datasets
            .lock()
            .map_err(|_| anyhow::anyhow!("streaming dataset registry lock poisoned"))?;
        let dataset = datasets.get_mut(dataset_id).ok_or_else(|| {
            anyhow::anyhow!(
                "cannot start streaming: dataset '{dataset_id}' is not registered \
                 (register it with DataPipeline::register_dataset first)"
            )
        })?;
        dataset.active = true;
        Ok(())
    }

    /// Stop streaming a dataset without discarding its buffered samples.
    pub async fn stop_streaming(&self, dataset_id: &str) -> Result<()> {
        let mut datasets = self
            .streaming_datasets
            .lock()
            .map_err(|_| anyhow::anyhow!("streaming dataset registry lock poisoned"))?;
        let dataset = datasets
            .get_mut(dataset_id)
            .ok_or_else(|| anyhow::anyhow!("unknown dataset '{dataset_id}'"))?;
        dataset.active = false;
        Ok(())
    }

    /// Draw up to `batch_size` samples from the active streaming datasets.
    ///
    /// Samples are taken round-robin from every active dataset (so no single source starves
    /// the others), validated with [`DataPipeline::validate_batch`], and filtered according to
    /// the validator's `error_handling` policy:
    ///
    /// * `Strict` — the first invalid sample aborts the batch with an error;
    /// * `Skip` / `Fix` — invalid samples are dropped from the batch;
    /// * `LogAndContinue` — invalid samples are kept and logged.
    ///
    /// # Errors
    ///
    /// Fails when `batch_size` is zero, when no dataset is streaming, or under a `Strict`
    /// validation policy when a sample is invalid.
    pub async fn get_batch(&self, batch_size: usize) -> Result<Vec<DataSample>> {
        if batch_size == 0 {
            return Err(anyhow::anyhow!("batch_size must be greater than zero"));
        }

        let collected = {
            let mut datasets = self
                .streaming_datasets
                .lock()
                .map_err(|_| anyhow::anyhow!("streaming dataset registry lock poisoned"))?;

            let mut active_ids: Vec<String> =
                datasets.iter().filter(|(_, d)| d.active).map(|(id, _)| id.clone()).collect();
            if active_ids.is_empty() {
                return Err(anyhow::anyhow!(
                    "no dataset is streaming; call DataPipeline::start_streaming first"
                ));
            }
            active_ids.sort();

            let mut collected = Vec::with_capacity(batch_size);
            let mut exhausted = 0usize;
            let mut cursor = 0usize;
            while collected.len() < batch_size && exhausted < active_ids.len() {
                let id = &active_ids[cursor % active_ids.len()];
                cursor += 1;
                let Some(dataset) = datasets.get_mut(id) else {
                    exhausted += 1;
                    continue;
                };
                match dataset.pop_sample() {
                    Some(sample) => {
                        exhausted = 0;
                        collected.push(sample);
                    },
                    None => exhausted += 1,
                }
            }
            collected
        };

        if collected.is_empty() {
            return Ok(Vec::new());
        }

        let results = self.validate_batch(&collected).await?;
        let error_handling = {
            let validator =
                self.validator.lock().map_err(|_| anyhow::anyhow!("validator lock poisoned"))?;
            validator.config.error_handling.clone()
        };

        let mut batch = Vec::with_capacity(collected.len());
        for (sample, result) in collected.into_iter().zip(results) {
            if result.is_valid {
                batch.push(sample);
                continue;
            }
            match error_handling {
                ErrorHandling::Strict => {
                    return Err(anyhow::anyhow!(
                        "sample '{}' failed validation: {}",
                        sample.id,
                        result
                            .errors
                            .iter()
                            .map(|e| e.message.clone())
                            .collect::<Vec<_>>()
                            .join("; ")
                    ));
                },
                ErrorHandling::Skip | ErrorHandling::Fix => {},
                ErrorHandling::LogAndContinue => {
                    log::warn!(
                        "data_pipeline: keeping invalid sample '{}': {}",
                        sample.id,
                        result
                            .errors
                            .iter()
                            .map(|e| e.message.clone())
                            .collect::<Vec<_>>()
                            .join("; ")
                    );
                    batch.push(sample);
                },
            }
        }

        Ok(batch)
    }

    /// Validate a batch of samples against the configured validator.
    ///
    /// Every sample is checked against the declarative rules in `DataValidationConfig.rules`
    /// **and** against every registered [`Validator`] trait object; the returned vector is
    /// aligned one-to-one with `samples`.
    pub async fn validate_batch(&self, samples: &[DataSample]) -> Result<Vec<ValidationResult>> {
        let mut validator =
            self.validator.lock().map_err(|_| anyhow::anyhow!("validator lock poisoned"))?;
        let mut results = Vec::with_capacity(samples.len());
        for sample in samples {
            results.push(validator.validate_sample(sample)?);
        }
        Ok(results)
    }

    /// Borrow the pipeline configuration.
    pub fn config(&self) -> &DataPipelineConfig {
        &self.config
    }
}

impl StreamingDataset {
    /// Create an inactive streaming dataset with an empty buffer.
    pub fn new(config: StreamingDatasetConfig) -> Self {
        Self {
            config,
            buffer: VecDeque::new(),
            stats: StreamingStats {
                samples_processed: 0,
                bytes_processed: 0,
                processing_time: Duration::from_secs(0),
                error_count: 0,
            },
            active: false,
        }
    }

    /// Append samples, respecting `config.buffer_size`.
    ///
    /// Returns the number of samples accepted. A full buffer is reported as an error rather
    /// than silently discarding data.
    pub fn push_samples(&mut self, samples: Vec<DataSample>) -> Result<usize> {
        let capacity = self.config.buffer_size.max(1);
        let free = capacity.saturating_sub(self.buffer.len());
        if samples.len() > free {
            return Err(anyhow::anyhow!(
                "streaming buffer is full: {} free slot(s) for {} sample(s) (buffer_size = {})",
                free,
                samples.len(),
                capacity
            ));
        }
        let accepted = samples.len();
        for sample in samples {
            let bytes: u64 = sample
                .data
                .values()
                .map(|t| (t.size() * std::mem::size_of::<f32>()) as u64)
                .sum();
            self.stats.bytes_processed += bytes;
            self.buffer.push_back(sample);
        }
        Ok(accepted)
    }

    /// Pop the oldest buffered sample, counting it as processed.
    pub fn pop_sample(&mut self) -> Option<DataSample> {
        let sample = self.buffer.pop_front()?;
        self.stats.samples_processed += 1;
        Some(sample)
    }

    /// Number of samples currently buffered.
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }
}

impl DataValidator {
    /// Validate one sample against the declarative rules and the registered validators.
    ///
    /// Rule semantics (all parameters come from `ValidationRule::parameters`):
    ///
    /// | rule | parameters | check |
    /// |------|------------|-------|
    /// | `Schema` | `field` | the sample must carry that tensor |
    /// | `Range` | `field`, `min`, `max` | every element must lie inside `[min, max]` |
    /// | `Format` | `field`, `shape` (comma-separated) | the tensor must have that shape |
    /// | `Quality` | `field` | the tensor must be non-empty and free of `NaN`/`inf` |
    /// | `Consistency` | `field`, `matches` | the two tensors must share a shape |
    /// | `Custom` | `validator_name` | delegated to a registered [`Validator`] of that name |
    ///
    /// `ValidationSeverity::Error` produces a [`ValidationError`] (and marks the sample
    /// invalid); `Warning` and `Info` produce a [`ValidationWarning`].
    pub fn validate_sample(&mut self, sample: &DataSample) -> Result<ValidationResult> {
        let started = std::time::Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        for rule in &self.config.rules {
            if let Some(message) = Self::check_rule(rule, sample) {
                match rule.severity {
                    ValidationSeverity::Error => errors.push(ValidationError {
                        rule_name: rule.name.clone(),
                        message,
                        severity: ValidationSeverity::Error,
                    }),
                    ValidationSeverity::Warning | ValidationSeverity::Info => {
                        warnings.push(ValidationWarning {
                            rule_name: rule.name.clone(),
                            message,
                        })
                    },
                }
            }
        }

        for validator in &self.validators {
            let result = validator.validate(sample)?;
            errors.extend(result.errors);
            warnings.extend(result.warnings);
        }

        self.stats.samples_validated += 1;
        self.stats.validation_time += started.elapsed();
        for error in &errors {
            *self.stats.errors_detected.entry(error.rule_name.clone()).or_insert(0) += 1;
        }

        Ok(ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        })
    }

    /// Evaluate one declarative rule, returning a message when it fails.
    fn check_rule(rule: &ValidationRule, sample: &DataSample) -> Option<String> {
        let field = rule.parameters.get("field").map(String::as_str);

        match &rule.rule_type {
            ValidationRuleType::Schema => {
                let field = field?;
                if sample.data.contains_key(field) {
                    None
                } else {
                    Some(format!("required field `{field}` is missing"))
                }
            },
            ValidationRuleType::Range => {
                let field = field?;
                let tensor = sample.data.get(field)?;
                let min = rule.parameters.get("min").and_then(|v| v.parse::<f32>().ok());
                let max = rule.parameters.get("max").and_then(|v| v.parse::<f32>().ok());
                let values = tensor.data().ok()?;
                for value in values {
                    if let Some(min) = min {
                        if value < min {
                            return Some(format!("`{field}` value {value} is below minimum {min}"));
                        }
                    }
                    if let Some(max) = max {
                        if value > max {
                            return Some(format!("`{field}` value {value} exceeds maximum {max}"));
                        }
                    }
                }
                None
            },
            ValidationRuleType::Format => {
                let field = field?;
                let tensor = sample.data.get(field)?;
                let expected: Vec<usize> = rule
                    .parameters
                    .get("shape")?
                    .split(',')
                    .filter_map(|part| part.trim().parse::<usize>().ok())
                    .collect();
                if tensor.shape() == expected.as_slice() {
                    None
                } else {
                    Some(format!(
                        "`{field}` has shape {:?}, expected {:?}",
                        tensor.shape(),
                        expected
                    ))
                }
            },
            ValidationRuleType::Quality => {
                let field = field?;
                let tensor = sample.data.get(field)?;
                let values = tensor.data().ok()?;
                if values.is_empty() {
                    return Some(format!("`{field}` is empty"));
                }
                if values.iter().any(|v| !v.is_finite()) {
                    return Some(format!("`{field}` contains NaN or inf"));
                }
                None
            },
            ValidationRuleType::Consistency => {
                let field = field?;
                let other_name = rule.parameters.get("matches")?;
                let a = sample.data.get(field)?;
                let b = sample.data.get(other_name)?;
                if a.shape() == b.shape() {
                    None
                } else {
                    Some(format!(
                        "`{field}` shape {:?} is inconsistent with `{other_name}` shape {:?}",
                        a.shape(),
                        b.shape()
                    ))
                }
            },
            ValidationRuleType::Custom { validator_name } => Some(format!(
                "custom rule `{validator_name}` has no registered Validator; \
                 add one to DataValidator::validators"
            )),
        }
    }

    /// Register a [`Validator`] trait object that runs on every sample.
    pub fn add_validator(&mut self, validator: Box<dyn Validator>) {
        self.validators.push(validator);
    }

    /// Replace the declarative validation configuration.
    pub fn set_config(&mut self, config: DataValidationConfig) {
        self.config = config;
    }
}

// Default implementations
impl Default for DynamicAugmentationManager {
    fn default() -> Self {
        Self::new()
    }
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

impl Default for CurriculumLearningManager {
    fn default() -> Self {
        Self::new()
    }
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

impl Default for ActiveLearningManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ActiveLearningManager {
    pub fn new() -> Self {
        Self {
            config: ActiveLearningConfig {
                query_strategy: QueryStrategy::UncertaintySampling {
                    uncertainty_measure: UncertaintyMeasure::Entropy,
                },
                sampling: SamplingConfig {
                    batch_size: 10,
                    budget: 1000,
                    diversity_constraint: None,
                },
                annotation: AnnotationConfig {
                    source: AnnotationSource::Human {
                        annotator_pool: vec![],
                    },
                    quality_control: QualityControl {
                        multi_annotation: false,
                        agreement_threshold: 0.8,
                        assessment_method: QualityAssessmentMethod::InterAnnotatorAgreement,
                    },
                },
                integration: ActiveLearningIntegration {
                    update_frequency: 100,
                    min_new_samples: 10,
                    retrain_from_scratch: false,
                },
            },
            query_pool: vec![],
            stats: ActiveLearningStats {
                queries_made: 0,
                annotations_received: 0,
                model_improvement: 0.0,
            },
        }
    }
}

impl Default for MultiModalHandler {
    fn default() -> Self {
        Self::new()
    }
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

impl Default for DataValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl DataValidator {
    pub fn new() -> Self {
        Self {
            config: DataValidationConfig {
                rules: vec![],
                strategy: ValidationStrategy::All,
                error_handling: ErrorHandling::LogAndContinue,
            },
            validators: vec![],
            stats: ValidationStats {
                samples_validated: 0,
                errors_detected: HashMap::new(),
                validation_time: Duration::from_secs(0),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_with(id: &str, field: &str, values: Vec<f32>, shape: &[usize]) -> DataSample {
        let mut data = HashMap::new();
        data.insert(
            field.to_string(),
            Tensor::from_vec(values, shape).expect("tensor creation failed"),
        );
        DataSample {
            id: id.to_string(),
            data,
            metadata: HashMap::new(),
            timestamp: SystemTime::now(),
        }
    }

    fn minimal_streaming_config(buffer_size: usize) -> StreamingDatasetConfig {
        StreamingDatasetConfig {
            sources: vec![],
            buffer_size,
            prefetch_size: 0,
            shuffle: ShuffleConfig {
                enabled: false,
                buffer_size: 0,
                seed: None,
                strategy: ShuffleStrategy::Random,
            },
            batching: BatchingConfig {
                batch_size: 2,
                dynamic: false,
                max_batch_size: 2,
                strategy: BatchingStrategy::Fixed,
                drop_last: false,
            },
            caching: CachingConfig {
                enabled: false,
                cache_type: CacheType::Memory,
                max_size_gb: 0.0,
                eviction_policy: EvictionPolicy::LRU,
                compression: CompressionConfig {
                    enabled: false,
                    algorithm: CompressionAlgorithm::Zstd,
                    level: 0,
                },
            },
        }
    }

    fn pipeline_with_dataset(buffer_size: usize) -> DataPipeline {
        let config = DataPipelineConfig {
            streaming: minimal_streaming_config(buffer_size),
            augmentation: DynamicAugmentationConfig {
                strategies: vec![],
                adaptive: AdaptiveAugmentationConfig {
                    enabled: false,
                    strategy: AdaptationStrategy::PerformanceBased {
                        target_metric: "loss".to_string(),
                        threshold: 0.0,
                    },
                    update_frequency: 0,
                    metrics: vec![],
                },
                scheduling: AugmentationScheduling {
                    schedule_type: ScheduleType::Fixed,
                    parameters: HashMap::new(),
                },
            },
            curriculum: CurriculumLearningConfig {
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
            active_learning: ActiveLearningConfig {
                query_strategy: QueryStrategy::UncertaintySampling {
                    uncertainty_measure: UncertaintyMeasure::Entropy,
                },
                sampling: SamplingConfig {
                    batch_size: 1,
                    budget: 1,
                    diversity_constraint: None,
                },
                annotation: AnnotationConfig {
                    source: AnnotationSource::Human {
                        annotator_pool: vec![],
                    },
                    quality_control: QualityControl {
                        multi_annotation: false,
                        agreement_threshold: 0.0,
                        assessment_method: QualityAssessmentMethod::InterAnnotatorAgreement,
                    },
                },
                integration: ActiveLearningIntegration {
                    update_frequency: 1,
                    min_new_samples: 1,
                    retrain_from_scratch: false,
                },
            },
            multimodal: MultiModalConfig {
                modalities: vec![],
                fusion_strategy: FusionStrategy::EarlyFusion,
                alignment: AlignmentConfig {
                    method: AlignmentMethod::Timestamp,
                    temporal_alignment: false,
                },
                preprocessing: MultiModalPreprocessing {
                    synchronization: SynchronizationConfig {
                        require_all: false,
                        sync_window: Duration::from_secs(1),
                    },
                    missing_modality_handling: MissingModalityHandling::Skip,
                },
            },
            validation: DataValidationConfig {
                rules: vec![],
                strategy: ValidationStrategy::All,
                error_handling: ErrorHandling::Skip,
            },
            distributed: DistributedProcessingConfig {
                num_workers: 1,
                backend: ProcessingBackend::Threading,
                load_balancing: LoadBalancingStrategy::RoundRobin,
            },
        };
        let pipeline = DataPipeline::new(config);
        pipeline
            .register_dataset("train", minimal_streaming_config(buffer_size))
            .expect("register failed");
        pipeline
    }

    // ── Streaming / batching / validation ────────────────────────────────────

    #[tokio::test]
    async fn test_start_streaming_rejects_an_unknown_dataset() {
        // Regression: `start_streaming` used to return Ok(()) for any string.
        let pipeline = pipeline_with_dataset(8);
        assert!(
            pipeline.start_streaming("does-not-exist").await.is_err(),
            "an unknown dataset id must be an error"
        );
        assert!(pipeline.start_streaming("train").await.is_ok());
    }

    #[tokio::test]
    async fn test_get_batch_returns_the_pushed_samples() {
        // Regression: `get_batch` returned an empty vector no matter what.
        let pipeline = pipeline_with_dataset(8);
        pipeline
            .push_samples(
                "train",
                vec![
                    sample_with("a", "x", vec![1.0, 2.0], &[2]),
                    sample_with("b", "x", vec![3.0, 4.0], &[2]),
                    sample_with("c", "x", vec![5.0, 6.0], &[2]),
                ],
            )
            .expect("push failed");
        pipeline.start_streaming("train").await.expect("start failed");

        let batch = pipeline.get_batch(2).await.expect("get_batch failed");
        assert_eq!(batch.len(), 2, "the batch must contain the pushed samples");
        assert_eq!(batch[0].id, "a");
        assert_eq!(batch[1].id, "b");

        let rest = pipeline.get_batch(2).await.expect("get_batch failed");
        assert_eq!(rest.len(), 1, "only one sample is left");
        assert_eq!(rest[0].id, "c");

        let empty = pipeline.get_batch(2).await.expect("get_batch failed");
        assert!(empty.is_empty(), "a drained stream yields an empty batch");
    }

    #[tokio::test]
    async fn test_get_batch_errors_when_nothing_is_streaming() {
        let pipeline = pipeline_with_dataset(8);
        pipeline
            .push_samples("train", vec![sample_with("a", "x", vec![1.0], &[1])])
            .expect("push failed");
        assert!(
            pipeline.get_batch(1).await.is_err(),
            "get_batch must not silently return an empty stream"
        );
    }

    #[tokio::test]
    async fn test_get_batch_rejects_zero_batch_size() {
        let pipeline = pipeline_with_dataset(8);
        pipeline.start_streaming("train").await.expect("start failed");
        assert!(pipeline.get_batch(0).await.is_err());
    }

    #[tokio::test]
    async fn test_validate_batch_runs_the_configured_rules() {
        // Regression: `validate_batch` returned an empty vector, so nothing was ever checked.
        let pipeline = pipeline_with_dataset(8);
        {
            let mut validator = pipeline.validator.lock().expect("validator lock");
            validator.set_config(DataValidationConfig {
                rules: vec![
                    ValidationRule {
                        name: "x_present".to_string(),
                        rule_type: ValidationRuleType::Schema,
                        severity: ValidationSeverity::Error,
                        parameters: HashMap::from([("field".to_string(), "x".to_string())]),
                    },
                    ValidationRule {
                        name: "x_range".to_string(),
                        rule_type: ValidationRuleType::Range,
                        severity: ValidationSeverity::Error,
                        parameters: HashMap::from([
                            ("field".to_string(), "x".to_string()),
                            ("min".to_string(), "0.0".to_string()),
                            ("max".to_string(), "1.0".to_string()),
                        ]),
                    },
                ],
                strategy: ValidationStrategy::All,
                error_handling: ErrorHandling::Skip,
            });
        }

        let samples = vec![
            sample_with("good", "x", vec![0.25, 0.75], &[2]),
            sample_with("out_of_range", "x", vec![0.25, 9.0], &[2]),
            sample_with("wrong_field", "y", vec![0.25], &[1]),
        ];
        let results = pipeline.validate_batch(&samples).await.expect("validate_batch failed");

        assert_eq!(results.len(), 3, "one result per sample");
        assert!(results[0].is_valid, "in-range sample must pass");
        assert!(!results[1].is_valid, "out-of-range sample must fail");
        assert!(!results[2].is_valid, "missing field must fail");
        assert!(results[1].errors.iter().any(|e| e.rule_name == "x_range"));
        assert!(results[2].errors.iter().any(|e| e.rule_name == "x_present"));
    }

    #[tokio::test]
    async fn test_get_batch_skips_invalid_samples_under_the_skip_policy() {
        let pipeline = pipeline_with_dataset(8);
        {
            let mut validator = pipeline.validator.lock().expect("validator lock");
            validator.set_config(DataValidationConfig {
                rules: vec![ValidationRule {
                    name: "finite".to_string(),
                    rule_type: ValidationRuleType::Quality,
                    severity: ValidationSeverity::Error,
                    parameters: HashMap::from([("field".to_string(), "x".to_string())]),
                }],
                strategy: ValidationStrategy::All,
                error_handling: ErrorHandling::Skip,
            });
        }
        pipeline
            .push_samples(
                "train",
                vec![
                    sample_with("ok", "x", vec![1.0], &[1]),
                    sample_with("nan", "x", vec![f32::NAN], &[1]),
                ],
            )
            .expect("push failed");
        pipeline.start_streaming("train").await.expect("start failed");

        let batch = pipeline.get_batch(2).await.expect("get_batch failed");
        assert_eq!(batch.len(), 1, "the NaN sample must be skipped");
        assert_eq!(batch[0].id, "ok");
    }

    #[tokio::test]
    async fn test_get_batch_aborts_under_the_strict_policy() {
        let pipeline = pipeline_with_dataset(8);
        {
            let mut validator = pipeline.validator.lock().expect("validator lock");
            validator.set_config(DataValidationConfig {
                rules: vec![ValidationRule {
                    name: "finite".to_string(),
                    rule_type: ValidationRuleType::Quality,
                    severity: ValidationSeverity::Error,
                    parameters: HashMap::from([("field".to_string(), "x".to_string())]),
                }],
                strategy: ValidationStrategy::All,
                error_handling: ErrorHandling::Strict,
            });
        }
        pipeline
            .push_samples("train", vec![sample_with("nan", "x", vec![f32::NAN], &[1])])
            .expect("push failed");
        pipeline.start_streaming("train").await.expect("start failed");
        assert!(pipeline.get_batch(1).await.is_err());
    }

    #[test]
    fn test_push_samples_respects_the_buffer_size() {
        let pipeline = pipeline_with_dataset(2);
        assert!(pipeline
            .push_samples(
                "train",
                vec![
                    sample_with("a", "x", vec![1.0], &[1]),
                    sample_with("b", "x", vec![1.0], &[1]),
                    sample_with("c", "x", vec![1.0], &[1]),
                ]
            )
            .is_err());
        assert!(pipeline
            .push_samples("train", vec![sample_with("a", "x", vec![1.0], &[1])])
            .is_ok());
    }

    #[test]
    fn test_push_samples_rejects_unknown_dataset() {
        let pipeline = pipeline_with_dataset(4);
        assert!(pipeline
            .push_samples("nope", vec![sample_with("a", "x", vec![1.0], &[1])])
            .is_err());
    }

    #[test]
    fn test_data_pipeline_creation() {
        let config = DataPipelineConfig {
            streaming: StreamingDatasetConfig {
                sources: vec![],
                buffer_size: 1000,
                prefetch_size: 100,
                shuffle: ShuffleConfig {
                    enabled: true,
                    buffer_size: 1000,
                    strategy: ShuffleStrategy::Random,
                    seed: Some(42),
                },
                batching: BatchingConfig {
                    batch_size: 32,
                    dynamic: false,
                    max_batch_size: 64,
                    strategy: BatchingStrategy::Fixed,
                    drop_last: false,
                },
                caching: CachingConfig {
                    enabled: false,
                    cache_type: CacheType::Memory,
                    max_size_gb: 1.0,
                    eviction_policy: EvictionPolicy::LRU,
                    compression: CompressionConfig {
                        enabled: false,
                        algorithm: CompressionAlgorithm::Gzip,
                        level: 6,
                    },
                },
            },
            augmentation: DynamicAugmentationConfig {
                strategies: vec![],
                adaptive: AdaptiveAugmentationConfig {
                    enabled: false,
                    strategy: AdaptationStrategy::PerformanceBased {
                        target_metric: "accuracy".to_string(),
                        threshold: 0.8,
                    },
                    update_frequency: 100,
                    metrics: vec![],
                },
                scheduling: AugmentationScheduling {
                    schedule_type: ScheduleType::Fixed,
                    parameters: HashMap::new(),
                },
            },
            curriculum: CurriculumLearningConfig {
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
            active_learning: ActiveLearningConfig {
                query_strategy: QueryStrategy::UncertaintySampling {
                    uncertainty_measure: UncertaintyMeasure::Entropy,
                },
                sampling: SamplingConfig {
                    batch_size: 10,
                    budget: 1000,
                    diversity_constraint: None,
                },
                annotation: AnnotationConfig {
                    source: AnnotationSource::Human {
                        annotator_pool: vec![],
                    },
                    quality_control: QualityControl {
                        multi_annotation: false,
                        agreement_threshold: 0.8,
                        assessment_method: QualityAssessmentMethod::InterAnnotatorAgreement,
                    },
                },
                integration: ActiveLearningIntegration {
                    update_frequency: 100,
                    min_new_samples: 10,
                    retrain_from_scratch: false,
                },
            },
            multimodal: MultiModalConfig {
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
            validation: DataValidationConfig {
                rules: vec![],
                strategy: ValidationStrategy::All,
                error_handling: ErrorHandling::LogAndContinue,
            },
            distributed: DistributedProcessingConfig {
                num_workers: 4,
                backend: ProcessingBackend::Threading,
                load_balancing: LoadBalancingStrategy::RoundRobin,
            },
        };

        let pipeline = DataPipeline::new(config);
        assert!(pipeline
            .streaming_datasets
            .lock()
            .expect("lock should not be poisoned")
            .is_empty());
    }

    #[test]
    fn test_augmentation_manager() {
        let manager = DynamicAugmentationManager::new();
        assert!(manager.strategies.is_empty());
        assert_eq!(manager.stats.augmentations_applied.len(), 0);
    }

    #[test]
    fn test_curriculum_manager() {
        let manager = CurriculumLearningManager::new();
        assert_eq!(manager.current_stage, 0);
        assert_eq!(manager.stats.current_difficulty, 0.0);
    }

    #[test]
    fn test_active_learning_manager() {
        let manager = ActiveLearningManager::new();
        assert_eq!(manager.stats.queries_made, 0);
        assert_eq!(manager.stats.annotations_received, 0);
    }

    #[test]
    fn test_multimodal_handler() {
        let handler = MultiModalHandler::new();
        assert!(handler.modality_processors.is_empty());
        assert_eq!(handler.stats.fusion_efficiency, 0.0);
    }

    #[test]
    fn test_data_validator() {
        let validator = DataValidator::new();
        assert!(validator.validators.is_empty());
        assert_eq!(validator.stats.samples_validated, 0);
    }
}
