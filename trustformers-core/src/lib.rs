#![allow(clippy::excessive_nesting)] // Algorithm-heavy code often requires deep nesting
#![allow(clippy::result_large_err)] // Large error enums are intentional for rich error context
#![allow(clippy::excessive_precision)] // High-precision floats needed for ML computations
#![allow(clippy::module_name_repetitions)] // Module names often repeat for clarity
#![allow(clippy::similar_names)] // Similar variable names are common in mathematical code
#![allow(clippy::too_many_arguments)] // ML functions often require many parameters
#![allow(clippy::too_many_lines)] // Complex algorithms require longer functions

pub mod ab_testing;
pub mod adaptive_computation;
pub mod autodiff;
pub mod blas;
pub mod cache;
pub mod checkpoint;
pub mod compiler;
pub mod compression;
pub mod error;
pub mod errors;
pub mod evaluation;
pub mod export;
pub mod generation;
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod gpu_accelerated;
pub mod hardware;
pub mod hardware_acceleration;
pub mod kernel_fusion;
pub mod kernels;
pub mod layers;
pub mod leaderboard;
pub mod memory;
pub mod monitoring;
pub mod neuromorphic;
pub mod numa_optimization;
pub mod ops;
pub mod optical;
pub mod parallel;
pub mod patterns;
pub mod peft;
pub mod performance;
pub mod plugins;
pub mod quantization;
pub mod quantum;
pub mod sparse_tensor;
pub mod tensor;
pub mod testing;
#[cfg(test)]
pub mod tests;
pub mod traits;
pub mod utils;
pub mod versioning;
pub mod visualization;

pub use ab_testing::{
    ABTestManager, ABTestSummary, ConfidenceLevel, DeploymentStrategy, Experiment,
    ExperimentConfig, ExperimentStatus, HealthCheck, HealthCheckType, MetricCollector,
    MetricSummary, MetricType, MetricValue, Recommendation, RollbackCondition, RolloutController,
    RolloutStatus, RoutingStrategy, StatisticalAnalyzer, TestRecommendation, TestResult,
    TrafficSplitter, UserSegment, Variant,
};
pub use adaptive_computation::{
    AdaptiveComputationConfig, AdaptiveComputationManager, AdaptiveComputationStrategy,
    ComplexityEstimationMethod, ComplexityEstimator, ComputationBudget, ComputationPath,
    ConfidenceBasedStrategy, DynamicDepthStrategy, EntropyBasedComplexityEstimator, LayerMetrics,
    LayerSkipPattern, PerformanceTracker, ResourceAllocation, UncertaintyBasedStrategy,
};
pub use blas::{
    blas_optimizer, init_blas, optimized_dot, optimized_gemm, optimized_gemv, BlasBackend,
    BlasConfig, BlasOperation, BlasOptimizer,
};
pub use cache::{
    CacheConfig, CacheEntry, CacheKey, CacheKeyBuilder, CacheMetrics, EvictionPolicy,
    InferenceCache, LRUEviction, SizeBasedEviction, TTLEviction,
};
pub use checkpoint::{
    convert_checkpoint, detect_format, load_checkpoint, save_checkpoint, CheckpointConverter,
    CheckpointFormat, ConversionConfig, ConversionResult, JaxCheckpoint, LayerMapping,
    PyTorchCheckpoint, TensorFlowCheckpoint, TrustformersCheckpoint, WeightMapping,
    WeightMappingRule,
};
pub use compiler::{
    CompilationResult, CompilerConfig, CompilerOptimizer, ComputationGraph, DeviceType, GraphEdge,
    GraphNode, HardwareTarget, OptimizationLevel, OptimizationRecommendation,
    OptimizationRecommendations, OptimizationResult, PassResult, RecommendationCategory,
    RecommendationPriority,
};
pub use compression::{
    // Convenience functions
    create_compression_pipeline,
    AccuracyRetention,
    AttentionDistiller,
    ChannelPruner,
    CompressionConfig as CompressionPipelineConfig,
    CompressionEvaluator,
    // Metrics exports
    CompressionMetrics,
    // Pipeline exports
    CompressionPipeline,
    CompressionRatio,
    CompressionReport,
    CompressionResult,
    CompressionStage,
    CompressionTargets,
    // Distillation exports
    DistillationConfig,
    DistillationLoss,
    DistillationResult,
    DistillationStrategy,
    FeatureDistiller,
    FilterPruner,
    GradualPruner,
    HeadPruner,
    HiddenStateDistiller,
    InferenceSpeedup,
    KnowledgeDistiller,
    LayerDistiller,
    LayerPruner,
    MagnitudePruner,
    ModelSizeReduction,
    PipelineBuilder,
    PruningConfig,
    PruningResult,
    PruningSchedule,
    PruningStats,
    // Pruning exports
    PruningStrategy,
    ResponseDistiller,
    SparsityMetric,
    StructuredPruner,
    StudentModel,
    TeacherModel,
    UnstructuredPruner,
};
pub use errors::{Result, TrustformersError};
pub use evaluation::{
    Accuracy, DatasetLoader, DatasetManager, DatasetSample, EvaluationConfig, EvaluationDataset,
    EvaluationHarness, EvaluationResult, EvaluationSuite, Evaluator, ExactMatch, F1Average,
    F1Score, FileDatasetLoader, GLUEEvaluator, GLUETask, MemoryDatasetLoader, Metric,
    MetricCollection, OtherBenchmark, Perplexity, SuperGLUEEvaluator, SuperGLUETask, BLEU,
};
pub use export::{
    CoreMLExporter, ExportConfig, ExportFormat, ExportPrecision, ExportQuantization, GGMLExporter,
    GGUFExporter, ModelExporter, ONNXExporter, TensorRTExporter, UniversalExporter,
};
pub use generation::{
    FinishReason,
    GenerationConfig,
    GenerationStrategy,
    GenerationStream,
    KVCache,
    // SpeculativeDecoder, TextGenerator,  // Temporarily disabled due to missing modules
};
#[cfg(feature = "cuda")]
pub use gpu_accelerated::{GpuAcceleratedOps, GpuOpsConfig, GpuPrecision};
pub use hardware::{
    AsicBackend, AsicDevice, AsicOperationSet, DataType, HardwareBackend, HardwareCapabilities,
    HardwareConfig, HardwareDevice, HardwareManager, HardwareMetrics, HardwareOperation,
    HardwareRegistry, HardwareResult, HardwareType, OperationMode, PrecisionMode,
};
pub use kernel_fusion::{
    ComputationGraph as FusionComputationGraph, DataType as FusionDataType, Device as FusionDevice,
    FusedKernel, FusionConstraint, FusionOpportunity, FusionPattern, FusionStatistics,
    GraphNode as FusionGraphNode, KernelFusionEngine, KernelImplementation, MemoryLayout,
    NodeMetadata, OperationType, TensorInfo,
};
pub use kernels::fused_ops::ActivationType;
pub use kernels::{
    FusedAttentionDropout, FusedBiasActivation, FusedGELU, FusedLinear, FusedMatmulScale,
    OptimizedRoPE, RoPEConfig, RoPEScalingType, SIMDLayerNorm, SIMDSoftmax, VectorizedRoPE,
};
// Import hardware traits separately
pub use hardware::traits::{
    AsyncHardwareOperation, AsyncOperationHandle, AsyncOperationStatus, DeviceMemory, DeviceStatus,
    HardwareScheduler, MemoryType, MemoryUsage as HardwareMemoryUsage, OperationParameter,
    OperationRequirements, PerformanceRequirements, SchedulerStatistics,
};
// Import ASIC types from asic submodule
pub use autodiff::{
    AnalysisResult,
    AutodiffEngine,
    ComputationGraph as AutodiffComputationGraph,
    DebuggerConfig,
    GradientFlowStats,
    GradientMode,
    GradientTape,
    // Debugger exports
    GraphDebugger,
    GraphIssue,
    GraphNode as AutodiffGraphNode,
    GraphOutputFormat,
    IssueSeverity,
    IssueType,
    MemoryStats,
    NodeDebugInfo,
    NodeId,
    OperationType as AutodiffOperationType,
    TapeEntry,
    TraversalInfo,
    Variable,
    VariableRef,
};
pub use hardware::asic::{
    AsicDeviceConfig, AsicDriver, AsicDriverFactory, AsicMemoryConfig, AsicPerformanceMonitor,
    AsicSpec, AsicType, AsicVendor, CacheConfig as AsicCacheConfig,
};
pub use hardware_acceleration::{
    api as hardware_acceleration_api, AccelerationBackend, AccelerationConfig, AccelerationStats,
    HardwareAccelerator,
};
pub use leaderboard::{
    LeaderboardCategory, LeaderboardClient, LeaderboardEntry, LeaderboardFilter,
    LeaderboardManager, LeaderboardQuery, LeaderboardRanking, LeaderboardStats, LeaderboardStorage,
    LeaderboardSubmission, RankingCriteria, SubmissionValidator,
};
pub use memory::{
    get_memory_manager, get_tensor, init_memory_manager, return_tensor, MemoryConfig,
    MemoryMappedTensor, MemoryPoolStats, TensorMemoryPool, TensorView,
};
pub use monitoring::{
    AttentionPattern, AttentionPatternType, AttentionReport, AttentionVisualizer,
    AttentionVisualizerConfig, Counter, Gauge, Histogram, MemoryReport, MemorySnapshot,
    MemoryTracker, MemoryTrackerConfig, MemoryUsage, MetricsCollector, MetricsCollectorConfig,
    MetricsSummary, ModelMonitor, ModelProfiler, MonitoringConfig, MonitoringReport,
    MonitoringSession, OptimizationSuggestion, OptimizationType, ProfilerConfig, ProfilingReport,
};
pub use numa_optimization::{
    get_numa_allocator, init_numa_allocator, numa_alloc, numa_free, AllocationStats,
    HotspotSeverity, NumaAllocation, NumaAllocator, NumaNode, NumaPerformanceMonitor, NumaPolicy,
    NumaStrategy, NumaTopology, NumaTrafficAnalysis, ThreadAffinity, ThreadPriority,
    TrafficHotspot,
};
pub use parallel::{
    init_parallelism,
    parallel_chunk_map,
    parallel_context,
    parallel_execute,
    parallel_map,
    ActivationType as ParallelActivationType,
    AsyncTensorParallel,
    // Parallel layers exports
    ColumnParallelLinear,
    CommunicationBackend,
    Communicator,
    DeviceMesh,
    DistributedTensor,
    InitMethod,
    MemoryPolicy,
    MicrobatchManager,
    // Model parallel exports
    ModelParallelConfig,
    ModelParallelContext,
    ModelParallelStrategy,
    NumaConfig,
    ParallelContext,
    ParallelMLP,
    ParallelMultiHeadAttention,
    ParallelOps,
    ParallelismStrategy,
    PipelineExecutor,
    // Pipeline parallel exports
    PipelineLayer,
    PipelineModel,
    PipelineOp,
    PipelineOptimizer,
    PipelineSchedule,
    PipelineScheduleType,
    PipelineStage,
    RowParallelLinear,
    TensorParallelInit,
    // Tensor parallel exports
    TensorParallelOps,
    TensorParallelShapes,
    TensorPartition,
};
pub use patterns::{
    Buildable, Builder, BuilderError, BuilderResult, ConfigBuilder, ConfigBuilderImpl,
    ConfigManager, ConfigMetadata, ConfigSerializable, CpuLimits, EnvironmentConfig, GpuLimits,
    LoggingConfig, MemoryLimits, PatternError, PatternResult, PerformanceConfig, ResourceConfig,
    SecurityConfig, StandardBuilder, StandardConfig, UnifiedConfig, ValidatedBuilder,
};
pub use peft::{
    AdapterLayer, LoRALayer, PeftConfig, PeftMethod, PeftModel, PrefixTuningLayer,
    PromptTuningEmbedding, QLoRALayer,
};
pub use performance::{
    BenchmarkBuilder, BenchmarkCategory, BenchmarkConfig, BenchmarkDSL, BenchmarkMetadata,
    BenchmarkRegistry, BenchmarkReport, BenchmarkResult, BenchmarkRunner, BenchmarkRunnerBuilder,
    BenchmarkSpec, BenchmarkSuite, ComparisonResult, ContinuousBenchmark,
    ContinuousBenchmarkConfig, CustomBenchmark, Framework, HuggingFaceBenchmark, LatencyMetrics,
    MemoryMetrics, MemoryProfiler, MemorySnapshot as PerformanceMemorySnapshot,
    MemoryTracker as PerformanceMemoryTracker, MetricsTracker, ModelComparison,
    PerformanceProfiler, PerformanceRegression, ProfileResult, PytorchBenchmark, ReportFormat,
    Reporter, RunConfig, RunMode, ThroughputMetrics,
};
pub use plugins::{
    Dependency, GpuRequirements, Plugin, PluginContext, PluginEvent, PluginEventHandler,
    PluginInfo, PluginLoader, PluginManager, PluginRegistry, SystemRequirements,
};
pub use quantization::{
    dequantize_bitsandbytes,
    from_bitsandbytes_format,
    quantize_4bit,
    quantize_dynamic_tree,
    quantize_int8,
    to_bitsandbytes_format,
    AWQQuantizer,
    ActivationLayerQuantConfig,
    // Activation quantization
    ActivationQuantConfig,
    ActivationQuantScheme,
    ActivationQuantizer,
    ActivationStats,
    // Mixed-bit quantization
    AutoBitAllocationStrategy,
    // BitsAndBytes compatibility
    BitsAndBytesConfig,
    BnBComputeType,
    BnBConfig,
    BnBQuantType,
    BnBQuantizer,
    BnBStorageType,
    FakeQuantize,
    GPTQQuantizer,
    LayerQuantConfig,
    MixedBitConfig,
    MixedBitQuantizedTensor,
    MixedBitQuantizer,
    Observer,
    QATConfig,
    QuantState,
    QuantizationConfig,
    QuantizationScheme,
    QuantizedActivation,
    QuantizedBlock,
    QuantizedTensor,
    Quantizer,
    SensitivityConfig,
    SensitivityMetric,
};
pub use sparse_tensor::{SparseFormat, SparseIndices, SparseTensor};
pub use tensor::{
    DType, EvalContext, ExprNode, OpType, OptimizationHints, Tensor, TensorExpr, TensorType,
};
pub use traits::{Config, Layer, Model};
pub use versioning::{
    ActiveDeployment,
    // Storage types
    Artifact,
    ArtifactType,
    DateRange,
    DeploymentConfig,
    DeploymentEvent,
    DeploymentEventType,
    // Deployment types
    DeploymentManager,
    DeploymentStatistics,
    DeploymentStatus,
    DeploymentStrategy as VersioningDeploymentStrategy,
    Environment,
    FileSystemStorage,
    HealthStatus,
    InMemoryStorage,
    LifecycleEvent,
    LifecyclePolicies,
    LifecycleStatistics,
    // Metadata types
    ModelMetadata,
    ModelRegistry,
    ModelRoutingResult,
    ModelSource,
    ModelStorage,
    ModelTag,
    // Core versioning types
    ModelVersionManager,
    PromotionResult,
    RegistryStatistics,
    SortBy,
    SortOrder,
    TagMatchMode,
    VersionExperimentConfig,
    VersionExperimentResult,
    VersionFilter,
    VersionLifecycle,
    VersionMetricType,
    // Registry types
    VersionQuery,
    VersionStats,
    // Lifecycle types
    VersionStatus,
    VersionTransition,
    // Integration types
    VersionedABTestManager,
    VersionedExperiment,
    VersionedExperimentStatus,
    VersionedModel,
};
pub use visualization::{
    ColorScheme, OutputFormat, TensorHeatmap, TensorHistogram, TensorSliceView, TensorStats,
    TensorVisualizer, VisualizationConfig,
};
