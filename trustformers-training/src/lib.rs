// Allow large error types in Result (TrustformersError is large by design)
#![allow(clippy::result_large_err)]
// Allow common patterns in training code
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::excessive_nesting)]

pub mod adaptive_gradient_scaling;
pub mod adaptive_learning_rate;
pub mod advanced_stability_monitor;
pub mod auto_parallelism;
pub mod config_validation;
pub mod continual;
pub mod cost_tracking;
pub mod data_pipeline;
pub mod distributed;
pub mod elastic_training;
pub mod error_codes;
pub mod error_handling;
pub mod experiment_management;
pub mod expert_parallelism;
pub mod few_shot;
pub mod framework_integration;
pub mod gradient;
pub mod gradient_anomaly_recovery;
pub mod hyperopt;
pub mod losses;
pub mod memory_optimization;
pub mod metrics;
pub mod mixed_precision;
pub mod model_versioning;
pub mod multicloud;
pub mod nas_integration;
pub mod online_learning;
pub mod parallelism_3d;
pub mod qat;
pub mod resource_scheduling;
pub mod ring_attention;
pub mod rlhf;
pub mod sequence_parallelism;
pub mod simplified_trainer;
pub mod tensor_parallelism;
pub mod trainer;
pub mod training_args;
pub mod training_dynamics;
pub mod training_monitor;
pub mod training_orchestration;

pub use continual::{
    CatastrophicPreventionStrategy, ContinualLearningConfig, ContinualLearningManager, EWCConfig,
    EWCTrainer, ExperienceBuffer, FisherInformation, MemoryReplay, MemoryReplayConfig,
    ProgressiveConfig, ProgressiveNetwork, RegularizationMethod, TaskBoundaryDetector, TaskInfo,
    TaskModule, TaskTransition,
};
pub use distributed::{
    init_distributed_training, utils as distributed_utils, DataParallelTrainer, DistributedBackend,
    DistributedConfig, ProcessGroup,
};
pub use experiment_management::{
    ABTestConfig, ABTestResults, ABTestStatus, ArtifactType, DataLineage, DataSplit,
    EnvironmentInfo, ExperimentFilters, ExperimentManager, ExperimentMetadata, ExperimentReport,
    ExperimentResults, ExperimentStatus, GPUInfo, HardwareInfo, HyperparameterComparison,
    HyperparameterConfig, ModelArtifact, ModelLineage, ModelProvenance, ModelSizeInfo,
    ParameterChange, PipelineStep, QualityAssuranceStep, SystemInfo, TrainingPipeline,
};
pub use few_shot::{
    AdaptationConfig, CrossTaskGeneralizer, FewShotConfig, FewShotExample, FewShotMethod,
    GeneralizationConfig, ICLExample, InContextConfig, InContextLearner, MAMLConfig, MAMLTrainer,
    MetaLearningAlgorithm, PromptConfig, PromptTuner, ReptileConfig, ReptileTrainer, SoftPrompt,
    SupportSet, TaskAdapter, TaskDescriptor, TaskEmbedding,
};
pub use gradient::GradientUtils;
pub use hyperopt::{
    // Efficiency features
    AcquisitionFunction,
    AcquisitionFunctionType,
    AdvancedEarlyStoppingConfig,
    ArmGenerationStrategy,
    ArmStatistics,
    BanditAlgorithm,
    BanditConfig,
    BanditOptimizer,
    BayesianOptimization,
    CategoricalParameter,
    ContinuousParameter,
    Direction,
    DiscreteParameter,
    // Configuration
    EarlyStoppingConfig,
    EarlyStoppingStrategy,
    EvaluationJob,
    EvaluationResult,
    ExplorationStrategy,
    FaultToleranceConfig,
    GPSampler,
    GPUAllocation,
    GridSearch,
    HalvingStrategy,
    HyperParameter,
    Hyperband,
    // Core types
    HyperparameterTuner,
    JobStatus,
    KernelType,
    LoadBalancer,
    LogParameter,
    OptimizationDirection,
    OptimizationResult,
    // PBT (Population-based Training)
    PBTConfig,
    PBTMember,
    PBTStats,
    ParallelEvaluationConfig,
    ParallelEvaluator,
    ParallelStrategy,
    ParameterValue,
    PopulationBasedTraining,
    PriorityLevel,
    PruningConfig,
    PruningStrategy,
    RandomSampler,
    RandomSearch,
    ResourceAllocation,
    ResourceUsage,
    RewardFunction,
    // Samplers
    Sampler,
    SamplerConfig,
    // Search space
    SearchSpace,
    // Strategies
    SearchStrategy,
    StudyStatistics,
    SuccessiveHalving,
    SurrogateConfig,
    SurrogateModel,
    SurrogateModelType,
    SurrogateOptimizer,
    TPESampler,
    // Trials
    Trial,
    TrialHistory,
    TrialMetrics,
    TrialResult,
    TrialState,
    TunerConfig,
    WarmStartConfig,
    WarmStartDataSource,
    WarmStartStrategy,
};
pub use losses::{CrossEntropyLoss, Loss, MSELoss};
pub use metrics::{Accuracy, F1Score, Metric, MetricCollection, Perplexity};
pub use mixed_precision::{
    utils as mixed_precision_utils, AMPManager, AdvancedMixedPrecisionConfig,
    AdvancedMixedPrecisionManager, ComputeOptimizationManager, ComputeOptimizationReport,
    DynamicBatchingConfig, DynamicBatchingManager, DynamicBatchingReport, LayerScalingConfig,
    LossScaler, MixedPrecisionConfig, MixedPrecisionReport,
};
pub use qat::{
    fake_quantize, fake_quantize_mixed_bit, qat_loss, ActivationQuantizer, CalibrationDataset,
    LayerQuantConfig, MixedBitQATTrainer, MixedBitStrategy, QATConfig, QATConv2d, QATLinear,
    QATModel, QATTrainer, QuantStats, QuantizationGradients, QuantizationParams, QuantizedModel,
};
pub use rlhf::{
    ConstitutionalPrinciple, GenerationResult, HumanFeedback, PPOConfig, PPOStepResult, PPOTrainer,
    PolicyModel, PreferencePair, RLHFConfig, RLHFMetrics, RLHFPhase, RewardModel,
    RewardModelConfig, RewardPrediction, ValueModel,
};
pub use trainer::{EarlyStoppingCallback, LogEntry, Trainer, TrainerCallback, TrainingState};
pub use training_args::{EvaluationStrategy, SaveStrategy, TrainingArguments};
pub use training_dynamics::{
    ConvergenceMetrics, GradientFlowMetrics, LossLandscapeMetrics, TrainingDynamicsAnalyzer,
    TrainingDynamicsConfig, TrainingDynamicsReport, TrainingDynamicsSnapshot,
    WeightEvolutionMetrics,
};

// New module exports
pub use adaptive_gradient_scaling::{
    AdaptiveGradientScaler, AdaptiveGradientScalingConfig, AdaptiveScalingStatistics,
    GradientScalingResult, LayerGradientStats as AdaptiveLayerGradientStats, StabilityTrend,
};
pub use adaptive_learning_rate::{
    AdaptationStrategy as LRAdaptationStrategy, AdaptiveLRStatistics, AdaptiveLearningRateConfig,
    AdaptiveLearningRateScheduler, LearningRateUpdate, PerformanceTrend, SchedulerState,
    TrainingDynamics as LRTrainingDynamics,
};
pub use auto_parallelism::{
    utils as auto_parallelism_utils, ArchitectureType, AutoParallelismConfig,
    AutoParallelismSelector, DeviceType, EvaluationMethod, HardwareConstraints, ModelConstraints,
    NetworkTopology, OptimizationObjective, ParallelismStrategy, PerformanceRequirements,
    SelectionAlgorithm,
};
pub use data_pipeline::{
    ActiveLearningConfig, ActiveLearningIntegration, ActiveLearningManager, ActiveLearningStats,
    AdaptationStrategy, AdaptiveAugmentationConfig, AlignmentConfig, AlignmentMethod,
    AnnotationConfig, AnnotationSource, AudioAugmentationType, AugmentationScheduling,
    AugmentationStats, AugmentationStrategy, AugmentationStrategyType, BatchingConfig,
    BatchingStrategy, CacheType, CachingConfig,
    CompressionAlgorithm as DataPipelineCompressionAlgorithm, CoreSetMethod,
    CurriculumLearningConfig, CurriculumLearningManager, CurriculumScheduling,
    CurriculumSchedulingStrategy, CurriculumStage, CurriculumStats, CurriculumStrategy, DataFilter,
    DataPipeline, DataPipelineConfig, DataSample, DataSelectionCriteria, DataSource,
    DataSourceType, DataValidationConfig, DataValidator, DifficultyAssessment, DisagreementMeasure,
    DistributedProcessingConfig, DiversityConstraint, DiversityMeasure, DynamicAssessmentMethod,
    DynamicAugmentationConfig, DynamicAugmentationManager, ErrorHandling, EvictionPolicy,
    FeatureExtractionConfig, FeatureExtractionMethod, FilterType, FusionStrategy,
    ImageAugmentationType, LoadBalancingStrategy as DataPipelineLoadBalancingStrategy,
    MissingModalityHandling, Modality, ModalityProcessor, ModalityType, MultiModalConfig,
    MultiModalHandler, MultiModalPreprocessing, MultiModalStats, NormalizationConfig,
    NormalizationType, PacingFunction, PacingType, PreprocessingConfig, PreprocessingStep,
    PreprocessingStepType, ProcessingBackend, QualityAssessmentMethod, QualityControl,
    QueryStrategy, SamplingConfig, ScheduleType, ShuffleConfig, ShuffleStrategy, StreamingDataset,
    StreamingDatasetConfig, StreamingStats, SuccessCriteria, SynchronizationConfig,
    TextAugmentationType, TokenAugmentationType, UncertaintyMeasure, ValidationError,
    ValidationResult, ValidationRule, ValidationRuleType, ValidationSeverity, ValidationStats,
    ValidationStrategy, ValidationWarning, Validator,
};
pub use elastic_training::{
    ElasticTrainingConfig, ElasticTrainingCoordinator, ScalingDecision, ScalingType, SystemStatus,
    WorkerInfo, WorkerStatus,
};
pub use expert_parallelism::{
    utils as expert_parallelism_utils, ExpertAssignment, ExpertCommunicationPattern,
    ExpertParallelism, ExpertParallelismConfig, ExpertRoutingStrategy, LoadBalancingStats,
    LoadBalancingStrategy, TokenRouting,
};
pub use framework_integration::{
    AggregationFunction, ArtifactConfig, ArtifactInfo, AudioLoggingConfig, AutoConnectConfig,
    ChartType, ClearMLArtifactConfig, ClearMLConfig, ClearMLTaskType, ColorFormat,
    ConflictResolution, CustomArtifact, CustomMetric, CustomMonitoring, CustomScalar,
    ExperimentMetadata as FrameworkExperimentMetadata,
    ExperimentStatus as FrameworkExperimentStatus, ExperimentTracker, ExportConfig, ExportFormat,
    ExportFrequency, FrameworkIntegrationManager, GraphLoggingConfig, HistogramConfig,
    ImageLoggingConfig, IntegrationConfig, IntegrationType, MLflowAdvancedConfig, MLflowAuth,
    MLflowAuthType, MLflowConfig, MLflowTracker, MetricType, MetricValue, ModelRegistrationConfig,
    ModelStage, NeptuneConfig, NeptuneExperimentConfig, NeptuneMonitoringConfig,
    ParameterValue as FrameworkParameterValue, ProfilingConfig, ResumeConfig, ScalarLayout,
    SyncConfig, SyncFrequency, TensorBoardAdvancedConfig, TensorBoardConfig, TensorBoardTracker,
    UpdateFrequency, WandBAdvancedConfig, WandBConfig, WandBTracker, WatchModelConfig,
};
pub use memory_optimization::{
    CPUOffloadManager, GradientCheckpointWrapper, MemoryOptimizationConfig,
    MemoryOptimizationStats, MemoryOptimizer,
};
pub use multicloud::{
    AlertType, AuthConfig, AuthType, BudgetAlert, CloudProvider, CloudScheduler,
    CommunicationPattern, CompressionAlgorithm, CompressionConfig, CostConfig, CostEntry,
    CostOptimizationStrategy, InstanceType, MultiCloudConfig, MultiCloudOrchestrator,
    MultiCloudProcessGroup, NodeInfo, NodeStatus, OrchestrationStrategy,
    PerformanceMetrics as MultiCloudPerformanceMetrics, RecoveryStrategy, SchedulingAlgorithm,
};
pub use nas_integration::{
    Architecture, NASAlgorithm, NASConfig, NASController, Operation, PerformanceMetrics,
    SearchSpaceConfig, TargetPlatform,
};
pub use parallelism_3d::{
    AggregateParallelismStats, CommBackend, MemoryOptimization, Parallelism3D,
    Parallelism3DManager, Parallelism3DStats, ParallelismConfig, PipelineSchedule,
};
pub use sequence_parallelism::{
    utils as sequence_parallelism_utils, AttentionCommunication, SequenceChunk,
    SequenceCommunicationPattern, SequenceMemoryOptimization, SequenceParallelism,
    SequenceParallelismConfig, SequenceParallelismStats, SequenceSplittingStrategy,
};
pub use tensor_parallelism::{
    utils as tensor_parallelism_utils, CommunicationRequirement, TensorCommunicationPattern,
    TensorMemoryOptimization, TensorOperation, TensorOperationType, TensorParallelism,
    TensorParallelismConfig, TensorParallelismStatistics, TensorPartition,
    TensorPartitioningStrategy,
};
pub use training_monitor::{
    AnomalyReport, AnomalyType, HealthStatus, PerformanceStats, TrainingHealthStatus,
    TrainingMonitor, TrainingMonitorConfig, TrainingReport,
};

// Advanced stability monitoring exports
pub use advanced_stability_monitor::{
    AdvancedStabilityConfig, AdvancedStabilityMonitor, LossLandscapeAnalysis, PatternDetector,
    PredictedAnomalyType, PredictiveAnomaly, PreventiveAction, RiskLevel, StabilityReport,
    StabilityScore, TrainerParameters, TrainingDynamics, TrendDirection,
};

// Gradient anomaly recovery exports
pub use gradient_anomaly_recovery::{
    AdaptiveThresholds, GradientAnomaly, GradientAnomalyType, GradientRecoveryConfig,
    GradientRecoveryManager, GradientRecoveryStrategy, GradientSeverity, LayerGradientStats,
    RecoveryResult, RecoveryStatistics,
};

// Production feature exports
pub use cost_tracking::{
    AlertThreshold, BillingModel, Budget, BudgetFilters, BudgetPeriod, BudgetStatus, CostBreakdown,
    CostDataPoint, CostDriver, CostEntry as CostTrackingCostEntry, CostForecastingModel,
    CostRecommendation, CostReport, CostStatistics, CostTracker, CostTrend, EfficiencyMetrics,
    ForecastingAccuracy, ForecastingParameters, ImplementationEffort, NotificationType,
    RecommendationCategory, RecommendationPriority, ReportType, TimeRange,
};
pub use model_versioning::{
    ModelRegistry, ModelStatus, ModelVersion, ModelVersioningManager,
    PerformanceMetrics as ModelVersioningPerformanceMetrics, TrainingConfig, VersionComparison,
};
pub use online_learning::{
    ConceptDrift, DriftType, OnlineDataPoint, OnlineLearningConfig, OnlineLearningError,
    OnlineLearningManager, OnlineStatistics, PerformanceWindow,
};
pub use resource_scheduling::{
    AlertSeverity, AllocationStatus, CostAlert, CostOptimizationRecommendation, CostSnapshot,
    LocalityPreference, Priority, RecommendationType,
    ResourceAllocation as SchedulingResourceAllocation, ResourceConstraints, ResourcePool,
    ResourceRequest, ResourceScheduler, ResourceType,
    SchedulingAlgorithm as ResourceSchedulingAlgorithm, SchedulingStatistics, StorageSpeed,
};
pub use ring_attention::{
    utils as ring_attention_utils, ModelParams, RingAttentionBlock, RingAttentionConfig,
    RingAttentionManager, RingAttentionStats, RingCommunicationPattern, RingKVPair,
};
pub use training_orchestration::{
    CheckpointConfig, CheckpointInfo, EarlyStoppingConfig as OrchestrationEarlyStoppingConfig,
    JobEvent, JobPriority, JobScheduler, JobStatus as OrchestrationJobStatus, ModelConfig,
    OrchestrationStatistics, ResourceNode, ResourceRequirements, SchedulingStrategy, TrainingJob,
    TrainingJobConfig, TrainingMetrics, TrainingOrchestrator,
};

// API improvement exports
pub use config_validation::{
    ConfigSchema, ConfigValidator, Constraint, FieldSchema, FieldType, Severity, Validatable,
    ValidatedConfig, ValidationError as ConfigValidationError, ValidationReport,
    ValidationRule as ConfigValidationRule,
};
pub use error_codes::{
    get_error_info, get_recovery_actions, is_critical_error, ErrorCodeInfo, ErrorCodeRegistry,
};
pub use error_handling::{
    ErrorContext, ErrorManager, ErrorPattern, ErrorSeverity, ErrorStatistics, ErrorTrend,
    ErrorType, RecoveryAction, RecoveryStrategy as ErrorRecoveryStrategy, RecoverySuggestion,
    SystemInfo as ErrorSystemInfo, TrainingError, TrainingErrorExt, TrainingResult,
};
pub use simplified_trainer::{
    CheckpointCallback, EarlyStoppingMode, EpochResult, LogLevel, LoggingCallback, MetricsCallback,
    ProgressCallback, SimpleCallback, SimpleTrainer, SimpleTrainerBuilder, SimpleTrainingConfig,
    TrainingResults,
};
