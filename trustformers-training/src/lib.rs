//! # TrustformeRS Training
//!
//! Training infrastructure and utilities for transformer models in Rust.
//!
//! This crate provides comprehensive tools for training transformer models, including:
//!
//! - **Distributed training**: Data parallelism, model parallelism, pipeline parallelism
//! - **Memory optimization**: Gradient checkpointing, mixed precision, ZeRO optimizer
//! - **Training strategies**: Few-shot learning, continual learning, curriculum learning
//! - **Stability**: Gradient clipping, loss scaling, NaN detection and recovery
//! - **Monitoring**: Real-time metrics, experiment tracking, resource monitoring
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use trustformers_training::{Trainer, TrainingArguments, MSELoss};
//! use trustformers_training::trainer::TaskType;
//! use trustformers_optim::Adam;
//! # use trustformers_core::tensor::Tensor;
//! # use trustformers_core::traits::{Config, Model};
//! # use trustformers_core::TrustformersError;
//! #
//! # // A minimal stand-in model; substitute a real architecture from `trustformers-models`.
//! # #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
//! # struct TinyConfig;
//! # impl Config for TinyConfig {
//! #     fn architecture(&self) -> &'static str { "tiny" }
//! # }
//! # #[derive(Debug, Clone)]
//! # struct TinyModel { config: TinyConfig }
//! # impl Model for TinyModel {
//! #     type Config = TinyConfig;
//! #     type Input = Tensor;
//! #     type Output = Tensor;
//! #     fn forward(&self, input: Self::Input) -> Result<Self::Output, TrustformersError> { Ok(input) }
//! #     fn load_pretrained(&mut self, _r: &mut dyn std::io::Read) -> Result<(), TrustformersError> { Ok(()) }
//! #     fn get_config(&self) -> &Self::Config { &self.config }
//! #     fn num_parameters(&self) -> usize { 0 }
//! # }
//! # let model = TinyModel { config: TinyConfig };
//! // Configure training (see `TrainingArguments` for the full set of options).
//! let args = TrainingArguments::default();
//!
//! // Choose an optimizer and a loss function.
//! let optimizer = Box::new(Adam::new(1e-4, (0.9, 0.999), 1e-8, 0.0));
//! let loss_fn = Box::new(MSELoss::new());
//!
//! // Build the trainer; `trainer.train(..)` then runs the loop over your datasets.
//! let _trainer = Trainer::new(model, args, optimizer, loss_fn, TaskType::Classification)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Distributed Training
//!
//! ```rust,no_run
//! use trustformers_training::distributed::{
//!     DataParallelTrainer, DistributedBackend, DistributedConfig, SimulatedProcessGroup,
//! };
//! use std::sync::Arc;
//! # use trustformers_core::tensor::Tensor;
//! # use trustformers_core::traits::{Config, Model};
//! # use trustformers_core::TrustformersError;
//! #
//! # #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
//! # struct TinyConfig;
//! # impl Config for TinyConfig {
//! #     fn architecture(&self) -> &'static str { "tiny" }
//! # }
//! # #[derive(Debug, Clone)]
//! # struct TinyModel { config: TinyConfig }
//! # impl Model for TinyModel {
//! #     type Config = TinyConfig;
//! #     type Input = Tensor;
//! #     type Output = Tensor;
//! #     fn forward(&self, input: Self::Input) -> Result<Self::Output, TrustformersError> { Ok(input) }
//! #     fn load_pretrained(&mut self, _r: &mut dyn std::io::Read) -> Result<(), TrustformersError> { Ok(()) }
//! #     fn get_config(&self) -> &Self::Config { &self.config }
//! #     fn num_parameters(&self) -> usize { 0 }
//! # }
//! # let model = TinyModel { config: TinyConfig };
//! // Data-parallel setup. The simulated backend needs no real cluster, so this
//! // example is runnable anywhere; swap in `DistributedBackend::NCCL` on real hardware.
//! let config = DistributedConfig {
//!     world_size: 1,
//!     rank: 0,
//!     backend: DistributedBackend::Simulated,
//!     master_addr: "localhost".to_string(),
//!     master_port: 29500,
//!     gradient_compression: false,
//!     bucket_size_mb: 25,
//! };
//!
//! let process_group = Arc::new(SimulatedProcessGroup::new(0, 1));
//! let _trainer = DataParallelTrainer::new(model, process_group, config)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Memory Optimization
//!
//! - **Gradient Checkpointing**: Trade compute for memory
//! - **Mixed Precision**: FP16/BF16 training with loss scaling
//! - **ZeRO**: Sharded optimizer states and gradients
//! - **Activation Checkpointing**: Recompute activations during backward
//!
//! ## Features
//!
//! - `distributed`: Multi-GPU and multi-node training
//! - `mixed-precision`: FP16/BF16 training support
//! - `gradient-checkpointing`: Memory-efficient training
//! - `wandb`: Weights & Biases integration
//! - `tensorboard`: TensorBoard logging

// reason: this crate ships a large amount of forward-looking training infrastructure
// (distributed/tensor/sequence/3D parallelism, RLHF, NAS, elastic training, ...) whose
// struct fields and helper methods are intentionally part of the public-facing scaffolding
// but are not all wired into an active code path yet. A single consolidated allow keeps the
// build warning-free without scattering ~110 per-item `#[allow(dead_code)]` attributes.
#![allow(dead_code)]
// Allow large error types in Result (TrustformersError is large by design)
#![allow(clippy::result_large_err)]
// Allow common patterns in training code
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::excessive_nesting)]
// Allow training-specific patterns
#![allow(clippy::await_holding_lock)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::ptr_arg)]

#[cfg(test)]
mod trainer_tests;

pub mod adaptive_gradient_scaling;
pub mod adaptive_learning_rate;
pub mod advanced_stability_monitor;
pub mod async_checkpoint;
pub mod auto_parallelism;
pub mod config_validation;
pub mod constitutional_ai;
pub mod continual;
pub mod contrastive_search;
pub mod cost_tracking;
pub mod curriculum;
pub mod data_pipeline;
pub mod distillation;
pub mod distributed;
/// Real collective backends (shared-memory and TCP) implementing
/// [`distributed::ProcessGroup`].
pub mod distributed_collective;
pub mod distributed_overlap;
/// ZeRO stage-1 optimizer-state sharding over a real
/// [`distributed::ProcessGroup`].
pub mod distributed_zero;
pub mod dpo;
pub mod elastic_training;
pub mod error_codes;
pub mod error_handling;
pub mod ewc;
pub mod experiment_management;
pub mod expert_parallelism;
pub mod few_shot;
pub mod framework_integration;
pub mod gradient;
pub mod gradient_anomaly_recovery;
pub mod grpo;
pub mod hpo;
pub mod hyperopt;
pub mod ipo;
pub mod kto;
pub mod lora;
pub mod losses;
pub mod memory_optimization;
pub mod metrics;
pub mod mixed_precision;
pub mod model_merging;
pub mod model_versioning;
pub mod multicloud;
pub mod nas_integration;
pub mod online_dpo;
pub mod online_learning;
pub mod orpo;
pub mod parallelism_3d;
pub mod ppo;
pub mod qat;
pub mod raft;
pub mod reinforce;
pub mod resource_scheduling;
pub mod reward_modeling;
pub mod ring_attention;
pub mod rlhf;
pub mod sequence_parallelism;
pub mod simplified_trainer;
pub mod simpo;
pub mod spin;
pub mod tensor_parallelism;
pub mod token_dpo;
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
    DistributedConfig, GradientCompressionConfig, ProcessGroup,
};
pub use distributed_collective::{
    run_in_process, CollectiveProcessGroup, InProcessProcessGroup, TcpProcessGroup,
};
pub use distributed_zero::{ShardAssignment, ZeroMemoryReport, ZeroStage, ZeroStage1Optimizer};
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
pub use hpo::{
    compute_pareto_front, hypervolume_indicator, non_domination_sort, AutoLrConfig,
    AutoLrModelType, AutoLrResult, AutoLrSelector, AutoLrStrategy, HpConfig, HpSearchSpace,
    HpValue, LrRangeTest, MultiObjectiveHpo, MultiObjectiveHpoConfig, MultiObjectiveResult,
    ObjectiveDirection, ParetoFront, RecommendedSchedule, TrainingLrConfig,
};
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
    fake_quantize, fake_quantize_mixed_bit, qat_loss, qat_loss_with, ActivationQuantizer,
    CalibrationDataset, LayerQuantConfig, MixedBitQATTrainer, MixedBitStrategy, QATConfig,
    QATConv2d, QATLinear, QATModel, QATTrainer, QuantStats, QuantizableLayer,
    QuantizationGradients, QuantizationParams, QuantizedModel,
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
    ConfigSchema, ConfigValidator, Constraint, CustomConstraintEvaluator, FieldSchema, FieldType,
    Severity, Validatable, ValidatedConfig, ValidationError as ConfigValidationError,
    ValidationReport, ValidationRule as ConfigValidationRule,
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
    TensorDataset, TrainableModel, TrainingDataset, TrainingResults,
};
