//! TrustformeRS Inference Server
//!
//! High-performance inference server with dynamic batching, model versioning,
//! and production-ready features for deploying TrustformeRS models.

pub mod alerting;
pub mod async_performance;
pub mod async_runtime_chaos;
pub mod audit;
pub mod auth;
pub mod batching;
pub mod caching;
pub mod chaos_testing;
pub mod cloud_providers;
pub mod contract_testing;
pub mod cpu_gpu_load_balancer;
pub mod custom_metrics;
pub mod dashboard;
pub mod disaster_recovery;
pub mod distributed_tracing;
pub mod dynamic_gpu_allocation;
pub mod edge_deployment;
pub mod encryption;
pub mod error_tracking;
pub mod gdpr_compliance;
pub mod gdpr_compliance_modules;
pub mod gpu_profiler;
pub mod gpu_scheduler;
pub mod graph_optimization;
pub mod graphql;
pub mod grpc;
pub mod health;
pub mod kernel_fusion;
pub mod load_balancer;
pub mod load_testing;
pub mod logging;
pub mod memory_pressure;
pub mod message_queue;
pub mod message_queue_middleware;
pub mod metrics;
pub mod migration;
pub mod model_management;
pub mod mtls;
pub mod multi_model_serving;
pub mod multimodal;
pub mod openapi;
pub mod operator_scheduling;
pub mod parallel_execution_engine;
pub mod performance_optimizer;
pub mod pipeline_parallelism;
pub mod polling;
pub mod production;
pub mod rate_limit;
pub mod request_profiling;
pub mod resource_management;
pub mod resource_manager;
pub mod server;
pub mod serverless;
pub mod service_mesh;
pub mod shadow;
pub mod slo;
pub mod speculative_decoding;
pub mod streaming;
pub mod test_cicd_integration;
pub mod test_config_manager;
pub mod test_independence_analyzer;
pub mod test_parallelization;
pub mod test_performance_monitor;
pub mod test_performance_monitoring;
pub mod test_timeout_optimization;
pub mod test_utilities;
pub mod tracing;
pub mod validation;

// Re-export main components
pub use alerting::{
    Alert, AlertCondition, AlertEvent, AlertEventType, AlertGroup, AlertRule, AlertSeverity,
    AlertState, AlertTemplate, AlertingConfig, AlertingError, AlertingService, AlertingStats,
    AlertingStatsSummary, ComparisonOperator, NotificationChannel, NotificationChannelType,
};
pub use async_runtime_chaos::{
    AsyncExperimentResult, AsyncMemoryPressureConfig, AsyncNetworkFailureConfig,
    AsyncResourceExhaustionConfig, AsyncRuntimeChaosFramework, AsyncRuntimeChaosType,
    AsyncTestSuiteResult, CancellationStrategy, ConcurrentAccessConfig, ConcurrentAccessPattern,
    DeadlockConfig, PanicRecoveryConfig, PanicType, RuntimeShutdownConfig, TaskCancellationConfig,
};
pub use audit::{
    AuditConfig, AuditEvent, AuditEventType, AuditLogger, AuditOutcome, AuditSeverity,
};
pub use auth::{
    create_api_key_handler, list_api_keys_handler, login_handler, oauth2_authorize_handler,
    oauth2_callback_handler, oauth2_providers_handler, oauth2_refresh_handler,
    revoke_api_key_handler, AuthConfig, AuthMiddleware, AuthService, Claims, OAuth2AuthRequest,
    OAuth2AuthResponse, OAuth2CallbackQuery, OAuth2Config, OAuth2ProviderInfo,
    OAuth2ProvidersResponse, OAuth2RefreshRequest, OAuth2RefreshResponse, OAuth2State, OAuth2Token,
    OAuth2TokenResponse, OAuth2UserInfo, TokenRequest, TokenResponse,
};
pub use batching::{BatchingConfig, BatchingStats, DynamicBatchingService, Request, RequestId};
pub use caching::{
    CacheConfig, CacheStats, CacheWarmer, CachingService, EmbeddingCacheService, KVCacheManager,
    ResultCacheService,
};
pub use chaos_testing::{
    BusinessImpact, ChaosExperiment, ChaosExperimentType, ChaosTestingFramework,
    ComparisonOperator as ChaosComparisonOperator, ConditionType, EventType, ExperimentConfig,
    ExperimentResults, ExperimentScope, ExperimentStatus, ImpactAnalysis, Observation,
    ObservationCategory, PreCondition, SafetyCheck, SafetyCheckType, SafetyConfig, Severity,
    SuccessCriterion, TimelineEvent, UserImpactEstimate,
};
pub use cloud_providers::{
    AlertThresholds, AutoScalingConfig as CloudAutoScalingConfig, CloudInferenceRequest,
    CloudInferenceResponse, CloudProvider, CloudProviderConfig, CloudProviderManager,
    CloudProviderStats, CloudProviderType, CostMetrics, CostOptimizationConfig, CredentialsConfig,
    DeploymentConfig, DeploymentStatus, EndpointConfig, FeatureConfig,
    HealthStatus as CloudHealthStatus, InputData, LimitsConfig, LoadBalancingStrategy,
    ModelDeploymentRequest, ModelDeploymentResponse, ModelInfo as CloudModelInfo, MonitoringConfig,
    OutputConfig, OutputData, OutputFormat, PerformanceCharacteristics, PerformanceEstimate,
    PerformanceMetrics, PricingInfo, ProviderConfig, ProviderMetrics, RequestPriority,
    ResourceRequirements, ResourceUtilization, ResponseMetadata, RetryPolicy as CloudRetryPolicy,
};
pub use contract_testing::{
    ApiContract, AuthContract, AuthType as ContractAuthType, ContractError, ContractTestConfig,
    ContractTestResult, ContractTestingFramework, ContractWarning, DataModelContract,
    EndpointContract, EndpointTestResult, ErrorSeverity as ContractErrorSeverity,
    ErrorType as ContractErrorType, HeaderContract, HttpMethod as ContractHttpMethod,
    ModelTestResult, PathParam, QueryParam, RateLimit as ContractRateLimit, TestStatus,
    TestSummary, ValidationResult as ContractValidationResult,
    ValidationRule as ContractValidationRule, ValidationType, VersionTolerance, WarningType,
};
pub use cpu_gpu_load_balancer::types::{PowerState, ProcessorStatus};
pub use cpu_gpu_load_balancer::{
    ComputeTask, CpuGpuLoadBalancer, ExecutionStatus,
    LoadBalancerConfig as CpuGpuLoadBalancerConfig, LoadBalancerEvent, LoadBalancerStats,
    LoadBalancingStrategy as CpuGpuLoadBalancingStrategy, MemoryPattern, PowerEfficiencyMetrics,
    PowerEfficiencyMode, PowerEfficiencyReport, PowerScalingFactors, ProcessorResource,
    ProcessorStats, ProcessorType, TaskExecutionResult, TaskPriority, TaskType,
};
pub use custom_metrics::{
    AnalyticsResult, Anomaly, ApplicationMetricType, CustomMetric, CustomMetricsCollector,
    CustomMetricsConfig, CustomMetricsError, MetricsSummary, RealTimeAnalytics, SystemMetricType,
    Trend,
};
pub use dashboard::{
    ChartType, DashboardAlert, DashboardConfig, DashboardError, DashboardLayout, DashboardService,
    DashboardStatsSummary, DashboardWidget, WidgetType,
};
pub use disaster_recovery::{
    BackupConfig, BackupStrategy, BackupTarget, CompressionConfig, ConflictResolution,
    ConsistencyLevel, DRError, DREventType, DRMonitoringConfig, DRStats, DRStatus, DRTestingConfig,
    DisasterRecoveryConfig, DisasterRecoveryManager, EncryptionConfig, FailoverConfig,
    FailoverStrategy, FailoverTrigger, HealthCheckConfig,
    NotificationChannel as DRNotificationChannel, NotificationConfig, ReplicationConfig,
    ReplicationMode, ReplicationTarget, ReplicationType, RetentionPolicy, RollbackConfig,
    SiteCapacity, SiteConfig, SiteStatus, SiteType, StorageType, TestScenario as DRTestScenario,
    TestSchedule, TrafficSplittingConfig, VerificationConfig,
};
pub use distributed_tracing::{
    create_tracing_middleware, trace_async, ActiveSpan, BatchConfig, DistributedSpan,
    SamplingStrategy, SpanBuilder, SpanEvent, SpanKind, SpanStatus, TraceContext, TracingBackend,
    TracingConfig, TracingEvent, TracingManager, TracingStats,
};
pub use dynamic_gpu_allocation::{
    AllocationError, AllocationMetrics, AllocationPriority, AllocationRequest, AllocationResult,
    AllocationStrategy, AllocationType, AutoScalingStrategy, DynamicAllocationConfig,
    DynamicGpuAllocator, GpuAllocation, GpuResource, GpuStatus, PerformanceProfile,
};
pub use edge_deployment::{
    BandwidthStrategy, DeploymentResult, EdgeConfig, EdgeError, EdgeMetrics, EdgeMode, EdgeModel,
    EdgeNode, EdgeNodeStatus, EdgeOptimization, EdgeOptimizationAction, EdgeOrchestrator,
    EdgeResources, EdgeStatistics, InferenceRequest as EdgeInferenceRequest, InferenceResponse,
    ModelFormat, ModelPriority, NodeDeploymentResult, NodeSyncResult, OptimizationAction,
    OptimizationLevel, OptimizationResult as EdgeOptimizationResult, SyncConfig, SyncEvent,
    SyncResult,
};
pub use encryption::service::{EncryptionService, EncryptionStats};
pub use encryption::EncryptionResult;
pub use encryption::{
    ColumnEncryptionConfig, ComplianceConfig, ComplianceStandard, DatabaseEncryptionConfig,
    DatabaseEncryptionScope, EncryptionAlgorithm, EncryptionConfig as ServiceEncryptionConfig,
    EncryptionError, FilesystemEncryptionConfig, KeyBackupConfig, KeyGenerationMethod,
    KeyManagementConfig, KeyManagementSystem, KeyRotationConfig, MasterKeyConfig,
    MemoryEncryptionConfig, PerformanceConfig, RotationSchedule, RotationTrigger,
};
pub use error_tracking::{
    ErrorCategory, ErrorContext, ErrorEntry, ErrorExportFormat, ErrorGroup, ErrorNotification,
    ErrorSeverity, ErrorStatistics, ErrorTrackingConfig, ErrorTrackingError, ErrorTrackingSystem,
    NotificationType, ResolutionStatus, StackFrame,
};
pub use gdpr_compliance::{
    ComplianceMonitoringConfig, ComplianceStatus, ConsentManagementConfig, ConsentRecord,
    ConsentStatus, ConsentStorageConfig, ConsentVerificationConfig, CrossBorderTransferConfig,
    DataBreachManagementConfig, DataCategory, DataProcessingConfig, DataRetentionConfig,
    DataSubjectRequest, DataSubjectRightsConfig, GdprComplianceConfig, GdprComplianceError,
    GdprComplianceService, GdprComplianceStats, LegalBasis, PrivacyByDesignConfig,
    PrivacyImpactAssessmentConfig, ProcessingPurpose, RequestProcessingResult, RequestStatus,
    RequestType, RetentionPolicy as DataRetentionPolicy,
};
pub use gpu_profiler::{
    AllocationPattern, BandwidthUtilization, BottleneckType, ComputeThroughput,
    EfficiencyMetrics as KernelEfficiencyMetrics, GpuAlert, GpuAlertThresholds, GpuAlertType,
    GpuClockSpeeds, GpuHealthStatus, GpuMemoryProfile, GpuMonitorConfig, GpuPerformanceProfile,
    GpuProcess, GpuProfiler, GpuProfilerConfig, GpuProfilerError, GpuProfilerStats,
    GpuProfilingReport, GpuUtilizationMetrics, KernelExecutionStats, MemoryAccessPattern,
    MemoryFragmentation, MemorySegment, MemorySegmentType, PerformanceBottleneck, ThermalEvent,
    ThermalEventType,
};
pub use gpu_scheduler::{
    GpuConfig, GpuMemoryStatus, GpuScheduler, GpuSchedulerConfig, GpuSchedulerEvent,
    GpuSchedulerStats, GpuStats, GpuTask, SchedulingAlgorithm, TaskResult, TaskStatus,
};
pub use graph_optimization::{
    ActivationType, AttributeValue, ComputationGraph, DataType, GraphMetadata, GraphNode,
    GraphOptimizationConfig, GraphOptimizationError, GraphOptimizationService,
    GraphOptimizationStatsSummary, Operation, OptimizationResult, OptimizationStats,
    OptimizationStep, OptimizationTarget, PoolType,
};
pub use serverless::{
    CostBreakdown, CostOptimizationResult, DetailedMetrics, OptimizationRecommendation,
    PerformanceBreakdown, RecommendationEffort, RecommendationPriority,
};
pub use test_cicd_integration::RotationStrategy;
// pub use graphql::{
//     create_context, create_schema, GraphQLContext, HealthInfo, DetailedHealthInfo,
//     InferenceInput, InferenceResult, BatchInferenceInput, BatchInferenceResult,
//     StatsInfo, ModelInfo, QueryRoot, MutationRoot
// }; // Temporarily disabled due to axum compatibility
pub use grpc::{inference, InferenceServiceImpl};
pub use health::{
    CircuitBreaker, FailoverManager, HAConfig, HealthCheckService, HealthStatus,
    HighAvailabilityService, RetryPolicy,
};
pub use kernel_fusion::{
    ComputeKernel, DeviceType, FusedKernel, FusionOpportunity, FusionStrategy, KernelFusionConfig,
    KernelFusionError, KernelFusionService, KernelFusionStatsSummary, KernelOperationType,
    TensorMetadata,
};
pub use load_balancer::{
    AutoScalingConfig, BackendInstance, CircuitBreakerSettings, ConnectionPoolConfig,
    InstanceHealth, LoadBalancer, LoadBalancerConfig, LoadBalancerMetrics, LoadBalancingAlgorithm,
    RequestContext, RetryPolicy as LoadBalancerRetryPolicy, SessionAffinityConfig,
};
pub use load_testing::{
    AdvancedConfig, AuthConfig as LoadTestAuthConfig, AuthType, HttpMethod, LoadTestConfig,
    LoadTestConfigBuilder, LoadTestResults, LoadTestService, MetricsCollector,
    OutputConfig as LoadTestOutputConfig, OutputFormat as LoadTestOutputFormat, TestScenario,
    TimeoutConfig, ValidationRule, ValidationRuleType,
};
pub use logging::{
    ErrorContext as LoggingErrorContext, LogFormat, LogLevel, LoggingConfig,
    PerformanceMetrics as LoggingPerformanceMetrics, RequestContext as LoggingRequestContext,
    StructuredLogger,
};
pub use memory_pressure::{
    AllocationInfo, CleanupHandler, CleanupStrategy, MemoryPressureConfig, MemoryPressureEvent,
    MemoryPressureHandler, MemoryPressureLevel, MemoryPressureThresholds, MemoryStats,
};
pub use message_queue::{
    AcknowledgmentMode, AutoOffsetReset, BatchResult, CompressionAlgorithm, ConsumerConfig,
    HealthStatus as MessageQueueHealthStatus, Message, MessageBatch, MessageQueueBackend,
    MessageQueueConfig, MessageQueueConsumer, MessageQueueEvent, MessageQueueHealth,
    MessageQueueManager, MessageQueueProducer, MessageQueueStats, MessageResult,
    PerformanceConfig as MessageQueuePerformanceConfig, ProducerCallback, ProducerConfig,
    RetryPolicy as MessageQueueRetryPolicy, SecurityConfig as MessageQueueSecurityConfig,
    SerializationFormat, TransactionId,
};
pub use message_queue_middleware::{
    AsyncInferenceRequest, AsyncInferenceResponse, AsyncTopicConfig, BatchMessageRequest,
    BatchMessageResponse, CommitRequest, ConsumeRequest, ConsumeResponse, ConsumedMessage,
    EventTopicConfig, MessageQueueMiddleware, MessageQueueMiddlewareConfig, MessageQueueRequest,
    MessageQueueResponse, SubscriptionRequest,
};
pub use metrics::{MetricsCollector as ServerMetricsCollector, MetricsService, RequestMetrics};
pub use migration::{
    BackupInfo, BackupManager, BackupType, LogLevel as MigrationLogLevel, MigrationConfig,
    MigrationExecution, MigrationExecutor, MigrationLogEntry, MigrationManager, MigrationOptions,
    MigrationPlan, MigrationPlanner, MigrationProgress, MigrationStatus, MigrationStep,
    MigrationStepResult, MigrationStepStatus, MigrationStepType, MigrationType,
    ValidationResult as MigrationValidationResult, ValidationRule as MigrationValidationRule,
    ValidationRuleType as MigrationValidationRuleType, ValidationSeverity,
    ValidationStatus as MigrationValidationStatus,
};
pub use model_management::{
    DeploymentManager, DeploymentStrategy, ModelError, ModelLoadConfig, ModelManagementConfig,
    ModelManager, ModelMetadata, ModelRegistry, ModelStatus, VersionManager,
};
pub use mtls::{
    mtls_middleware, CertValidationResult, CertificateInfo, CipherSuite, ClientCertValidation,
    MTlsConfig, MTlsError, MTlsService, MTlsStatsSummary, OcspConfig, TlsVersion,
};
pub use multi_model_serving::{
    ABTestingConfig, EnsembleConfig, EnsembleMethod, InferenceRequest, ModelCascadingConfig,
    ModelInfo as MultiModelInfo, ModelRoutingConfig, MultiModelConfig, MultiModelMetrics,
    MultiModelServer, RoutingResult, RoutingStrategy,
    TrafficSplittingConfig as MultiModelTrafficSplittingConfig,
};
pub use multimodal::{
    AudioFormat, DocumentFormat, DocumentMetadata, DocumentProcessingConfig, ImageFormat,
    MediaData, MultiModalConfig, MultiModalError, MultiModalInput, MultiModalRequest,
    MultiModalResponse, MultiModalService, MultiModalStatsSummary, OcrConfig, OcrEngine,
    ProcessingOptions, TextPreprocessingConfig, VideoFormat,
};
pub use openapi::ApiDoc;
pub use operator_scheduling::DeviceType as OpSchedulerDeviceType;
pub use operator_scheduling::{
    DeviceResource, OperationType, OperatorSchedulingConfig, OperatorSchedulingError,
    OperatorSchedulingService, OperatorSchedulingStatsSummary, OperatorTask,
    SchedulingAlgorithm as OpSchedulingAlgorithm, SchedulingDecision, SchedulingStats,
    TaskExecutionResult as OpTaskExecutionResult, TaskPriority as OpTaskPriority, TaskState,
};
pub use parallel_execution_engine::{
    ExecutionMonitor, LoadBalancer as ParallelLoadBalancer, ParallelExecutionEngine,
    ResourceManager as ParallelResourceManager, TestScheduler, WorkStealingConfig,
};
pub use performance_optimizer::{
    BatchingRecommendation, OptimizationEvent, OptimizationEventType, OptimizationHistory,
    OptimizationRecommendations, ParallelismEstimate, PerformanceMeasurement, PerformanceOptimizer,
    RealTimeMetrics, ResourceOptimizationRecommendation, TestCharacteristics,
};
pub use pipeline_parallelism::DeviceType as PipelineDeviceType;
pub use pipeline_parallelism::{
    LoadBalancingStrategy as PipelineLoadBalancingStrategy, PipelineConfig, PipelineError,
    PipelineParallelismManager, PipelineRequest, PipelineStatsSummary, RequestMetadata,
    RequestPriority as PipelineRequestPriority, StageConfig, StageStats,
};
pub use polling::{
    LongPollRequest, LongPollResponse, LongPollingConfig, LongPollingService, LongPollingStats,
    PollEvent, PollEventWithId,
};
pub use production::{
    CanaryMetrics, DeploymentStatus as ProductionDeploymentStatus, GracefulShutdownConfig,
    HealthStatus as ProductionHealthStatus, ProductionConfig, ProductionManager, ProductionMetrics,
    ProductionStatus, RollingUpdateConfig, ShutdownSignal, ShutdownState, UpdateInfo,
    UpdateStrategy,
};
pub use rate_limit::{RateLimitConfig, RateLimitError, RateLimitService, RateLimitStats};
pub use request_profiling::{
    CallStackEntry, CpuProfile, IoProfile, MemoryProfile, PerformanceIssue,
    PerformanceRecommendation, ProfileExportFormat, ProfileStatus, ProfilingStats,
    ProfilingStatsSummary, RequestProfile, RequestProfilingConfig, RequestProfilingError,
    RequestProfilingService, ResourceUsage, TimingBreakdown,
};
pub use resource_management::{
    AlertSystem, AnalyticsEngine, CleanupEvent, CleanupManager, CleanupTask, CustomResourceManager,
    DatabaseConnectionManager, GpuAllocation as RMGpuAllocation, GpuDeviceInfo,
    GpuMonitoringSystem, GpuPerformanceTracker, GpuResourceManager, HealthChecker, LoadMetrics,
    MetricsAggregator, NetworkPortManager as ModularNetworkPortManager, PerformanceAnomaly,
    PerformanceBottleneck as RMPerformanceBottleneck, PerformancePrediction,
    PortAllocation as ModularPortAllocation, ReportGenerator, ResourceAllocator,
    ResourceManagementSystem as ModularResourceManagementSystem, ResourceMonitor,
    ResourceUtilizationSnapshot, StatisticsCollector, SystemMetrics,
    TempDirectoryInfo as ModularTempDirectoryInfo,
    TempDirectoryManager as ModularTempDirectoryManager, WorkerPool,
};
pub use resource_management::{NetworkPortManager, PortAllocation, PortReservationSystem};
pub use resource_manager::{
    DirectoryStatus, PortUsageType, ResourceManagementSystem, TempDirectoryInfo,
    TempDirectoryManager,
};
pub use server::TrustformerServer;
pub use serverless::{
    AwsLambdaProvider, AzureFunctionsProvider, DeploymentLog, DeploymentPackage,
    DeploymentResult as ServerlessDeploymentResult, DeploymentStatus as ServerlessDeploymentStatus,
    EventSourceMapping, GoogleCloudFunctionsProvider,
    MonitoringConfig as ServerlessMonitoringConfig, PackageType, ScalingConfig, ServerlessConfig,
    ServerlessDeployment, ServerlessMetrics, ServerlessOrchestrator, ServerlessProvider,
    ServerlessProviderTrait, Trigger, TriggerType, VpcConfig,
};
pub use service_mesh::{
    HealthCheckConfig as ServiceMeshHealthCheckConfig, HealthStatus as ServiceMeshHealthStatus,
    LoadBalancingStrategy as ServiceMeshLoadBalancingStrategy, ReliabilityConfig,
    SecurityConfig as ServiceMeshSecurityConfig, ServiceEndpoint, ServiceMeshConfig,
    ServiceMeshManager, ServiceMeshMetrics, ServiceMeshType, TrafficManagementConfig, TrafficRule,
};
pub use shadow::{
    ComparisonMetrics, ModelShadowStats, ShadowComparison, ShadowConfig, ShadowEvent,
    ShadowModelConfig, ShadowRequest, ShadowResponse, ShadowStats, ShadowTestingService,
};
pub use slo::{
    AggregationMethod, AlertCondition as SloAlertCondition, AlertSeverity as SloAlertSeverity,
    BreachImpact, DataSource, ErrorBudgetConfig, SliDataPoint, SliDefinition, SliMeasurement,
    SliType, SloAlert, SloBreachEvent, SloConfig, SloCriticality, SloDefinition, SloError,
    SloPerformance, SloStats, SloTracker, SloWindow, TrendDirection, WindowType,
};
pub use speculative_decoding::{
    DraftModel, DraftToken, ModelInfo as SpeculativeModelInfo, SpeculativeDecoder,
    SpeculativeDecodingConfig, SpeculativeDecodingManager, SpeculativeStats, TargetModel,
    VerificationResult,
};
pub use streaming::{
    ChunkStream, SseEvent, SseHandler, StreamType, StreamingConfig, StreamingService,
    StreamingStats, TokenStream, WebSocketHandler, WsMessage,
};
pub use test_cicd_integration::{
    AuthConfig as CicdAuthConfig, CicdFeature, CicdIntegrationConfig, CicdIntegrationManager,
    EnvironmentConfig, EnvironmentMonitoringConfig, EnvironmentOptimizationSettings,
    EnvironmentResourceLimits, EnvironmentSecuritySettings, EnvironmentType, ExportFormat,
    ExportTarget, ExportTargetType, MetricsExportConfig, OptimizationConfig,
    PipelineIntegrationConfig, PipelineType, ReportingConfig, RetryConfig,
    SecurityConfig as CicdSecurityConfig, TlsConfig,
};
pub use test_independence_analyzer::ResourceRequirement;
pub use test_independence_analyzer::{TestGroup, TestIndependenceAnalyzer};
pub use test_parallelization::ResourceAllocation;
pub use test_parallelization::{
    CpuScalingConfig, LoadBalancingStrategy as TestLoadBalancingStrategy, MemoryOptimizationConfig,
    ParallelPerformanceMonitoringConfig, PerformanceOptimizationConfig, ResourceLimits,
    SchedulingConfig, SchedulingStrategy, TestBatchingConfig, TestParallelizationConfig,
    WarmupOptimizationConfig,
};
pub use test_parallelization::{
    DependencyType, TestDependency, TestParallelizationMetadata, TestResourceUsage,
};
pub use test_performance_monitoring::{
    CurrentPerformanceMetrics, EfficiencyMetrics as TestEfficiencyMetrics, EventSeverity,
    MonitoringConfig as TestMonitoringConfig, ParallelizationMetrics, PerformanceEvent,
    PerformanceEventData, PerformanceEventType, PerformanceReport, PerformanceStream,
    PerformanceThreshold, RealTimePerformanceMonitor, ReliabilityMetrics, ReportRequest,
    ReportSummary, ReportType, SystemResourceMetrics, TestExecutionMetrics,
    TestPerformanceMonitoringSystem, TimestampedMetrics,
};
pub use test_timeout_optimization::TestCategory;
pub use test_timeout_optimization::TestExecutionResult;
pub use tracing::{
    DistributedTracer, Span, SpanGuard, SpanLog, SpanStatus as LegacySpanStatus,
    TraceContext as LegacyTraceContext, TraceExportFormat, TraceMetrics,
    TracingConfig as LegacyTracingConfig, TracingError,
};
pub use validation::{
    GenerationParams, InferenceRequest as ValidationInferenceRequest, ValidationConfig,
    ValidationError, ValidationService,
};

/// Server version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Server configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
pub struct ServerConfig {
    /// Host to bind to
    pub host: String,

    /// Port to listen on
    pub port: u16,

    /// Number of worker threads
    pub num_workers: usize,

    /// Model configuration
    pub model_config: ModelConfig,

    /// Batching configuration
    pub batching_config: batching::BatchingConfig,

    /// Caching configuration
    pub caching_config: caching::CacheConfig,

    /// Cloud providers configuration
    pub cloud_providers_config: cloud_providers::CloudProviderConfig,

    /// Streaming configuration
    pub streaming_config: streaming::StreamingConfig,

    /// Validation configuration
    pub validation_config: validation::ValidationConfig,

    /// Rate limiting configuration
    pub rate_limit_config: rate_limit::RateLimitConfig,

    /// Audit configuration
    pub audit_config: audit::AuditConfig,

    /// Logging configuration
    pub logging_config: logging::LoggingConfig,

    /// Model management configuration
    pub model_management_config: model_management::ModelManagementConfig,

    /// Long polling configuration
    pub polling_config: polling::LongPollingConfig,

    /// Shadow testing configuration
    pub shadow_config: shadow::ShadowConfig,

    /// GPU scheduler configuration
    pub gpu_scheduler_config: gpu_scheduler::GpuSchedulerConfig,

    /// GPU profiler configuration
    pub gpu_profiler_config: gpu_profiler::GpuProfilerConfig,

    /// Dynamic GPU allocation configuration
    pub dynamic_gpu_allocation_config: dynamic_gpu_allocation::DynamicAllocationConfig,

    /// Edge deployment configuration
    pub edge_deployment_config: edge_deployment::EdgeConfig,

    /// Memory pressure configuration
    pub memory_pressure_config: memory_pressure::MemoryPressureConfig,

    /// Message queue configuration
    pub message_queue_config: message_queue::MessageQueueConfig,

    /// Message queue middleware configuration
    pub message_queue_middleware_config: message_queue_middleware::MessageQueueMiddlewareConfig,

    /// Tracing configuration
    pub tracing_config: tracing::TracingConfig,

    /// Distributed tracing configuration
    pub distributed_tracing_config: distributed_tracing::TracingConfig,

    /// Error tracking configuration
    pub error_tracking_config: error_tracking::ErrorTrackingConfig,

    /// CPU/GPU load balancer configuration
    pub cpu_gpu_load_balancer_config: cpu_gpu_load_balancer::LoadBalancerConfig,

    /// Speculative decoding configuration
    pub speculative_decoding_config: speculative_decoding::SpeculativeDecodingConfig,

    /// Pipeline parallelism configuration
    pub pipeline_parallelism_config: pipeline_parallelism::PipelineConfig,

    /// Kernel fusion configuration
    pub kernel_fusion_config: kernel_fusion::KernelFusionConfig,

    /// Custom metrics configuration
    pub custom_metrics_config: custom_metrics::CustomMetricsConfig,

    /// Dashboard configuration
    pub dashboard_config: dashboard::DashboardConfig,

    /// mTLS configuration
    pub mtls_config: mtls::MTlsConfig,

    /// Multi-modal configuration
    pub multimodal_config: multimodal::MultiModalConfig,

    /// Graph optimization configuration
    pub graph_optimization_config: graph_optimization::GraphOptimizationConfig,

    /// Operator scheduling configuration
    pub operator_scheduling_config: operator_scheduling::OperatorSchedulingConfig,

    /// Alerting configuration
    pub alerting_config: alerting::AlertingConfig,

    /// Request profiling configuration
    pub request_profiling_config: request_profiling::RequestProfilingConfig,

    /// Service mesh configuration
    pub service_mesh_config: service_mesh::ServiceMeshConfig,

    /// Load balancer configuration
    pub load_balancer_config: load_balancer::LoadBalancerConfig,

    /// Multi-model serving configuration
    pub multi_model_config: multi_model_serving::MultiModelConfig,

    /// Production management configuration
    pub production_config: production::ProductionConfig,

    /// Disaster recovery configuration
    pub disaster_recovery_config: disaster_recovery::DisasterRecoveryConfig,

    /// Encryption configuration
    pub encryption_config: encryption::EncryptionConfig,

    /// GDPR compliance configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gdpr_compliance_config: Option<gdpr_compliance::GdprComplianceConfig>,

    /// SLO tracking configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slo_config: Option<slo::SloConfig>,

    /// Serverless deployment configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub serverless_config: Option<serverless::ServerlessConfig>,

    /// Enable metrics endpoint
    pub enable_metrics: bool,

    /// Enable health check endpoint
    pub enable_health_check: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            num_workers: num_cpus::get(),
            model_config: ModelConfig::default(),
            batching_config: batching::BatchingConfig::default(),
            caching_config: caching::CacheConfig::default(),
            cloud_providers_config: cloud_providers::CloudProviderConfig::default(),
            streaming_config: streaming::StreamingConfig::default(),
            validation_config: validation::ValidationConfig::default(),
            rate_limit_config: rate_limit::RateLimitConfig::default(),
            audit_config: audit::AuditConfig::default(),
            logging_config: logging::LoggingConfig::default(),
            model_management_config: model_management::ModelManagementConfig::default(),
            polling_config: polling::LongPollingConfig::default(),
            shadow_config: shadow::ShadowConfig::default(),
            gpu_scheduler_config: gpu_scheduler::GpuSchedulerConfig::default(),
            gpu_profiler_config: gpu_profiler::GpuProfilerConfig::default(),
            dynamic_gpu_allocation_config: dynamic_gpu_allocation::DynamicAllocationConfig::default(
            ),
            edge_deployment_config: edge_deployment::EdgeConfig::default(),
            memory_pressure_config: memory_pressure::MemoryPressureConfig::default(),
            message_queue_config: message_queue::MessageQueueConfig::default(),
            message_queue_middleware_config:
                message_queue_middleware::MessageQueueMiddlewareConfig::default(),
            tracing_config: tracing::TracingConfig::default(),
            distributed_tracing_config: distributed_tracing::TracingConfig::default(),
            error_tracking_config: error_tracking::ErrorTrackingConfig::default(),
            cpu_gpu_load_balancer_config: cpu_gpu_load_balancer::LoadBalancerConfig::default(),
            speculative_decoding_config: speculative_decoding::SpeculativeDecodingConfig::default(),
            pipeline_parallelism_config: pipeline_parallelism::PipelineConfig::default(),
            kernel_fusion_config: kernel_fusion::KernelFusionConfig::default(),
            custom_metrics_config: custom_metrics::CustomMetricsConfig::default(),
            dashboard_config: dashboard::DashboardConfig::default(),
            mtls_config: mtls::MTlsConfig::default(),
            multimodal_config: multimodal::MultiModalConfig::default(),
            graph_optimization_config: graph_optimization::GraphOptimizationConfig::default(),
            operator_scheduling_config: operator_scheduling::OperatorSchedulingConfig::default(),
            alerting_config: alerting::AlertingConfig::default(),
            request_profiling_config: request_profiling::RequestProfilingConfig::default(),
            service_mesh_config: service_mesh::ServiceMeshConfig::default(),
            load_balancer_config: load_balancer::LoadBalancerConfig::default(),
            multi_model_config: multi_model_serving::MultiModelConfig::default(),
            production_config: production::ProductionConfig::default(),
            disaster_recovery_config: disaster_recovery::DisasterRecoveryConfig::default(),
            encryption_config: encryption::EncryptionConfig::default(),
            gdpr_compliance_config: Some(gdpr_compliance::GdprComplianceConfig::default()),
            slo_config: Some(slo::SloConfig::default()),
            serverless_config: Some(serverless::ServerlessConfig {
                provider: serverless::ServerlessProvider::AwsLambda,
                function_name: "trustformers-inference".to_string(),
                runtime: "provided.al2".to_string(),
                memory_mb: 512,
                timeout_seconds: 30,
                environment_variables: std::collections::HashMap::new(),
                vpc_config: None,
                deployment_package: serverless::DeploymentPackage {
                    package_type: serverless::PackageType::Zip,
                    source_location: "".to_string(),
                    handler: "main".to_string(),
                    layers: vec![],
                },
                triggers: vec![],
                scaling: serverless::ScalingConfig {
                    min_instances: 0,
                    max_instances: 100,
                    target_utilization: 0.7,
                    scale_down_delay_seconds: 300,
                    scale_up_delay_seconds: 60,
                    concurrency_limit: Some(10),
                },
                monitoring: serverless::MonitoringConfig {
                    enable_logging: true,
                    log_level: "INFO".to_string(),
                    enable_tracing: true,
                    enable_metrics: true,
                    custom_metrics: vec![],
                    enable_xray: false,
                    enable_insights: false,
                    log_retention_days: None,
                },
                cold_start: None,
                cost_optimization: None,
                region: Some("us-east-1".to_string()),
                tags: std::collections::HashMap::new(),
            }),
            enable_metrics: true,
            enable_health_check: true,
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    /// Model name or path
    pub model_name: String,

    /// Model version
    pub model_version: Option<String>,

    /// Device to run on
    pub device: Device,

    /// Maximum sequence length
    pub max_sequence_length: usize,

    /// Enable model caching
    pub enable_caching: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_name: String::new(),
            model_version: None,
            device: Device::Cpu,
            max_sequence_length: 2048,
            enable_caching: true,
        }
    }
}

/// Device type
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum Device {
    Cpu,
    Cuda(usize),
    Metal,
}

/// Server error types
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Server overloaded")]
    Overloaded,

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

impl axum::response::IntoResponse for ServerError {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;
        use axum::response::Json;

        let (status, message) = match self {
            ServerError::ModelNotFound(model) => {
                (StatusCode::NOT_FOUND, format!("Model not found: {}", model))
            },
            ServerError::InvalidRequest(msg) => {
                (StatusCode::BAD_REQUEST, format!("Invalid request: {}", msg))
            },
            ServerError::Overloaded => (
                StatusCode::SERVICE_UNAVAILABLE,
                "Server overloaded".to_string(),
            ),
            ServerError::NotFound(msg) => (StatusCode::NOT_FOUND, format!("Not found: {}", msg)),
            ServerError::Internal(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Internal error: {}", err),
            ),
        };

        let error_response = serde_json::json!({
            "error": message,
            "status": status.as_u16()
        });

        (status, Json(error_response)).into_response()
    }
}

/// Result type alias
pub type Result<T> = std::result::Result<T, ServerError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert!(config.enable_metrics);
    }
}
