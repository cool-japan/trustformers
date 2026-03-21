//! TrustformeRS Mobile Deployment
//!
//! This crate provides optimized mobile deployment support for TrustformeRS,
//! including iOS and Android platform integrations with platform-specific
//! optimizations and inference backends.

#![allow(deprecated)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
// Allow large error types in Result (TrustformersError is large by design)
#![allow(clippy::result_large_err)]
// Allow common patterns in mobile deployment code
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::excessive_nesting)]
#![allow(unused_assignments)]
#![allow(private_interfaces)]
#![allow(unused_must_use)]
// Allow FFI patterns - raw pointer handling in Unity interop is intentional
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::useless_vec)]
// Allow mobile-specific patterns
#![allow(clippy::await_holding_lock)]
#![allow(clippy::arc_with_non_send_sync)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::borrowed_box)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::redundant_pattern_matching)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::explicit_counter_loop)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::vec_init_then_push)]
#![allow(clippy::to_string_trait_impl)]
#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::format_in_format_args)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::empty_line_after_doc_comments)]

pub mod abi_checker;
pub mod benchmarks;
pub mod hardware;
pub mod inference;
pub mod optimization;
pub mod ui_testing;

// SciRS2 compatibility layer
pub mod scirs2_compat;

// Re-export SciRS2 compatibility types for seamless integration
pub use scirs2_compat::{DefaultRng, DistributionOps, LinalgOps, SimdOps, StatisticalOps, Tensor};

// C API for FFI interoperability
pub mod c_api;

#[cfg(target_os = "ios")]
pub mod ios;

#[cfg(target_os = "ios")]
pub mod ios_background;

#[cfg(target_os = "ios")]
pub mod ios_app_extensions;

#[cfg(target_os = "ios")]
pub mod ios_icloud;

#[cfg(target_os = "android")]
pub mod android;

#[cfg(target_os = "android")]
pub mod android_renderscript;

#[cfg(target_os = "android")]
pub mod android_work_manager;

#[cfg(target_os = "android")]
pub mod android_content_provider;

#[cfg(target_os = "android")]
pub mod android_doze_compatibility;

#[cfg(feature = "coreml")]
pub mod coreml;

#[cfg(feature = "coreml")]
pub mod coreml_converter;

#[cfg(feature = "nnapi")]
pub mod nnapi;

#[cfg(feature = "nnapi")]
pub mod nnapi_converter;

#[cfg(feature = "on-device-training")]
pub mod training;

#[cfg(feature = "on-device-training")]
pub mod federated;

#[cfg(feature = "on-device-training")]
pub mod differential_privacy;

#[cfg(feature = "on-device-training")]
pub mod advanced_training;

#[cfg(feature = "on-device-training")]
pub mod advanced_privacy_mechanisms;

pub mod advanced_security;
pub mod crash_reporter;
pub mod inference_visualizer;
pub mod integration_testing;
pub mod memory_leak_detector;
pub mod mobile_performance_profiler;
pub mod mobile_performance_profiler_legacy;
pub mod mobile_testing;
pub mod model_debugger;
pub mod privacy_preserving_inference;

#[cfg(target_os = "ios")]
pub mod arkit_integration;

#[cfg(target_os = "android")]
pub mod edge_tpu_support;

pub mod battery;
pub mod compression;
pub mod device_info;
pub mod lifecycle;
pub mod model_management;
pub mod network_adaptation;
pub mod network_optimization;
pub mod profiler;
pub mod thermal_power;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// WebAssembly SIMD optimization module
pub mod wasm_simd;

// Apple Neural Engine V3 optimization module
#[cfg(target_os = "ios")]
pub mod neural_engine_v3;

// Advanced Neural Engine V4 optimization for latest Apple hardware
#[cfg(target_os = "ios")]
pub mod advanced_neural_engine_v4;

// MLX framework integration for Apple Silicon
#[cfg(any(target_os = "ios", target_os = "macos"))]
pub mod mlx_integration;

// Federated Learning v2.0 with advanced differential privacy (moved to federated module)
// pub mod federated_learning_v2;

#[cfg(feature = "react-native")]
pub mod react_native;

#[cfg(feature = "react-native")]
pub mod react_native_turbo;

#[cfg(feature = "react-native")]
pub mod react_native_fabric;

#[cfg(feature = "flutter")]
pub mod flutter;

#[cfg(feature = "expo")]
pub mod expo_plugin;

#[cfg(feature = "unity")]
pub mod unity_interop;

// WebNN integration for web/hybrid platforms
pub mod webnn_integration;

// Advanced thermal management with predictive algorithms
pub mod advanced_thermal;

use serde::{Deserialize, Serialize};
use trustformers_core::errors::{Result, TrustformersError};

// Re-export AI-powered optimization types
pub use optimization::{
    ActivationType, ArchitectureMetrics, ConnectionType, DeviceConstraints, DeviceEnvironment,
    EarlyStoppingConfig, LayerConfig, LayerType, MobileArchitecture, MobileNAS, NASConfig,
    OptimizationTarget, PerformanceRecord, QualityTradeoffs,
    QuantizationConfig as NASQuantizationConfig, QuantizationScheme as NASQuantizationScheme,
    ReinforcementLearningAgent, SearchStrategy, SkipConnection, UsagePattern, UserContext,
    UserPreferences,
};

// Re-export GGUF mobile optimization types
pub use optimization::{
    MobileGGUFConfig, MobileGGUFQuantizer, MobileGGUFStats, MobileGGUFType, MobileGGUFUtils,
};

// Re-export WebNN integration types
pub use webnn_integration::{
    BrowserInfo, WebNNBackend, WebNNCapabilities, WebNNCompiledGraph, WebNNDataType, WebNNDevice,
    WebNNExecutionContext, WebNNGraphConfig, WebNNOperation, WebNNPowerPreference,
    WebNNSupportLevel, WebNNTensorDescriptor, WebNNUtils,
};

// Re-export advanced thermal management types
pub use advanced_thermal::{
    AdaptiveCoolingStrategy, AdvancedThermalManager, CoolingMode, MultiSensorThermalFusion,
    PlannedWorkload, SensorWeights, ThermalCoefficients, ThermalHotspot, ThermalPredictionModel,
    WorkloadExecutionPlan, WorkloadPriority, WorkloadThermalPlanner,
};

// Re-export key device info types
pub use device_info::{
    BasicDeviceInfo, ChargingStatus, CpuInfo, GpuInfo, MemoryInfo, MobileDeviceDetector,
    MobileDeviceInfo, NpuInfo, PerformanceScores, PerformanceTier, PowerInfo, SimdSupport,
    ThermalInfo, ThermalState,
};

// Re-export key thermal and power management types
pub use thermal_power::{
    InferencePriority, PowerOptimizationStrategy, PowerThresholds, ThermalPowerConfig,
    ThermalPowerManager, ThermalPowerStats, ThermalPowerUtils, ThermalThresholds, ThrottleLevel,
    ThrottlingStrategy,
};

// Re-export key compression types
pub use compression::{
    CompressionBenefits, CompressionConfig, CompressionStats, CompressionUtils, DistillationConfig,
    DistillationStrategy, MobileCompressionEngine, PruningStrategy, QualityMetric,
    QualityRecoveryStrategy, QuantizationPrecision, QuantizationStrategy,
};

// Re-export key battery management types
pub use battery::{
    BatteryConfig, BatteryLevel, BatteryReading, BatteryStats, BatteryThresholds, BatteryUtils,
    DailySummary, MobileBatteryManager, OptimizationRecommendation, PowerPrediction,
    PowerUsageLimits, UsageSession, WeeklyUsagePattern,
};

// Re-export key profiler types
pub use profiler::{
    BottleneckConfig, BottleneckSeverity, BottleneckType, CpuMetrics, GpuMetrics, InferenceMetrics,
    MemoryMetrics, MetricsConfig, MetricsSnapshot, MobileProfilerUtils, NetworkMetrics,
    OptimizationCategory, OptimizationSuggestion, PerformanceBottleneck, ProfileSnapshot,
    ProfilerCapability, ProfilerConfig, ProfilingStatistics,
};

// Re-export key network adaptation types
pub use network_adaptation::{NetworkAdaptationManager, NetworkAdaptationStats};

// Re-export key network optimization types
pub use network_optimization::{
    AdaptiveBehaviorConfig, BandwidthAdaptationConfig, BandwidthThresholds, BandwidthTier,
    CacheEvictionStrategy, CompressionAlgorithm, DownloadCompressionConfig, DownloadConstraints,
    DownloadOptimizationConfig, DownloadPriority, DownloadProgress, DownloadRetryConfig,
    DownloadStatus, EdgeCachingConfig, EdgeFailoverConfig, EdgeLoadBalancingStrategy,
    EdgeServerConfig, EdgeServerEndpoint, NetworkMetric, NetworkOptimizationConfig,
    NetworkOptimizationManager, NetworkQualityConfig, OfflineFirstConfig, OfflineRetentionPolicy,
    OfflineSyncStrategy, P2PConfig, P2PProtocol, P2PResourceLimits, P2PSecurityConfig,
    P2PSecurityLevel, P2PSharingPolicy, P2PTimeRestrictions, QualityAdaptationConfig,
    QualityAdaptationStrategy, QualityLevel, QualityThresholds, ResumableDownloadRequest,
    TimeWindow,
};

// Re-export privacy-preserving inference types
pub use privacy_preserving_inference::{
    AggregationMethod, InferenceBudgetConfig, InferencePrivacyConfig, InferencePrivacyGuarantees,
    InferenceQualityMetrics, InferenceUseCase, InputPerturbationMethod, InputPrivacyConfig,
    OutputPrivacyConfig, OutputPrivacyMethod, PrivacyPreservingInferenceEngine,
    PrivacyPreservingInferenceUtils, PrivacyUtilityTradeoff, PrivateInferenceResult,
    SecureAggregationConfig,
};

// Re-export advanced security types
pub use advanced_security::{
    AdvancedSecurityConfig, AdvancedSecurityManager, EncryptedTensor, EncryptionOptimization,
    HomomorphicConfig, HomomorphicEncryptionEngine, HomomorphicScheme, MPCCommunication,
    MPCProtocol, QuantumResistantAlgorithm, QuantumResistantConfig, QuantumResistantEngine,
    QuantumResistantKeyExchange, QuantumResistantSignature, SecureInferenceResult,
    SecureMultipartyConfig, SecureMultipartyEngine, SecurityLevel, ZKProof, ZKProofConfig,
    ZKProofSystem, ZKVerificationConfig, ZeroKnowledgeProofEngine,
};

// Re-export mobile performance profiler types
pub use mobile_performance_profiler::{
    AlertSeverity, AlertThresholds, AlertType, BatteryMetrics, ChartType, CpuProfilingConfig,
    EventData, EventMetrics, EventType, ExportConfig, ExportFormat, GpuProfilingConfig,
    HealthStatus, ImpactLevel, MemoryProfilingConfig, MetricTrend, MobileMetricsSnapshot,
    MobilePerformanceProfiler, MobileProfilerConfig, NetworkProfilingConfig, PerformanceAlert,
    PlatformMetrics, ProfilingData, ProfilingEvent, ProfilingMode, ProfilingSummary,
    RealTimeConfig, RealTimeState, SamplingConfig, SessionInfo, SessionMetadata, SystemHealth,
    TemperatureTrend, ThermalMetrics, TrendDirection, TrendingMetrics,
};

// Re-export crash reporter types
pub use crash_reporter::{
    AppCrashInfo, AuthType, BatteryCrashInfo, CpuCrashInfo, CrashAnalysis, CrashAnalysisConfig,
    CrashContext, CrashInfo, CrashPattern, CrashPrivacyConfig, CrashRecoveryConfig, CrashReport,
    CrashReporterConfig, CrashReportingConfig, CrashSeverity, CrashStatistics, CrashStorageConfig,
    CrashType, EncryptionKeySource, ErrorLogEntry, ExceptionInfo, ForegroundStatus, GpuCrashInfo,
    HeapInfo, ImpactAssessment, LogLevel, MemoryDump, MemoryPermissions, MemoryRegion,
    MemoryRegionType, MemoryUsageInfo, MobileCrashReporter, ModelCrashInfo,
    ModelPerformanceMetrics, NetworkConnectionType, NetworkCrashInfo, PatternType,
    PlatformCrashConfig, RecentOperation, RecoveryImpact, RecoveryStrategy, RecoverySuggestion,
    ReportingAuthConfig, ResolutionStatus, RiskAssessment, RiskLevel, SafeModeConfig, SignalInfo,
    SimilarCrash, StackFrame, StackInfo, StackTrace, SystemCrashInfo, SystemEvent, ThreadInfo,
    ThreadState, UrgencyLevel, UserAction,
};

// Re-export key lifecycle management types
pub use lifecycle::{
    AppCheckpoint, AppLifecycleManager, AppState, BackgroundTask, BackgroundTaskConfig,
    LifecycleConfig, LifecycleStats, LifecycleUtils, MemoryPressureLevel, MemoryWarningConfig,
    ModelState, NetworkInterruptionConfig, NotificationConfig, NotificationType,
    ResourceManagementConfig, ResourceRequirements, SchedulingStrategy, StateTransition, TaskType,
    ThermalLevel, ThermalWarningConfig, TransitionReason, UserSessionState,
};

// Re-export integration testing types
pub use integration_testing::{
    BackendTestResults, BackendTestingConfig, CompatibilityTestResults, CompatibilityTestingConfig,
    CrossPlatformComparison, ErrorAnalysis, IntegrationTestConfig, IntegrationTestResults,
    MobileIntegrationTestFramework, PerformanceBenchmarkResults, PerformanceTestingConfig,
    PlatformTestResults, PlatformTestingConfig, RecommendationPriority, RecommendationType,
    ReportFormat, TestCategory, TestConfiguration, TestEnvironmentInfo, TestMetrics,
    TestRecommendation, TestReportingConfig, TestResult, TestStatus, TestSummary,
};

// Re-export WebAssembly SIMD optimization types
pub use wasm_simd::{
    SimdInstructionSet, SimdLaneWidth, SimdOperationType, SimdPerformanceMetrics, WasmSimdConfig,
    WasmSimdEngine,
};

// Re-export Apple Neural Engine V3 optimization types
#[cfg(target_os = "ios")]
pub use neural_engine_v3::{
    AdvancedFeatures, AppleDeviceType, BatchOptimizationConfig, CachingStrategy, CalibrationStats,
    DataType, HardwareCapabilities, NeuralEngineOperation, NeuralEngineV3Config,
    NeuralEngineV3Engine, NeuralEngineV3Metrics, NeuralEngineVersion, PerformanceProfile,
    PowerEfficiencyMode, QuantizationProfile,
};

// Re-export MLX framework integration types
#[cfg(any(target_os = "ios", target_os = "macos"))]
pub use mlx_integration::{
    AppleSiliconDevice, CompilationStrategy, CompiledMlxModel, ComputeUnitConfig,
    DeviceCapabilities, GraphOptimizationConfig, MemoryRequirements, MlxConfig, MlxEngine,
    MlxOperation, MlxPerformanceMetrics, MlxPrecision, ModelPerformanceProfile, OptimizedGraph,
    PrecisionConfig, ProfilingConfig, UnifiedMemoryConfig,
};

// Re-export Federated Learning types
#[cfg(feature = "on-device-training")]
pub use federated::{
    AggregationMetadata, AggregationStrategy, ClientMetrics, ClientSelectionStrategy,
    ComputationCapability, DifferentialPrivacyConfig, FederatedLearningClient,
    FederatedLearningConfig, FederatedLearningStats, FederatedLearningUtils, GlobalModelUpdate,
    LocalTrainingResult, NetworkQuality, NoiseMechanism, SecureAggregator,
};

// Re-export Flutter integration types
#[cfg(feature = "flutter")]
pub use flutter::{
    FlutterChannelManager, FlutterDeviceInfo, FlutterInferenceRequest, FlutterInferenceResponse,
    FlutterMethodCall, FlutterMethodResult, FlutterPerformanceMetrics, FlutterQuantizationConfig,
    FlutterTrustformersConfig,
};

// Re-export Expo plugin types
#[cfg(feature = "expo")]
pub use expo_plugin::{
    AssetBundlingStrategy, AssetConfig, AssetLoadingStrategy, BuildConfig,
    CacheEvictionStrategy as ExpoCacheEvictionStrategy,
    CompressionAlgorithm as ExpoCompressionAlgorithm, DevelopmentConfig, ExpoConfig, ExpoContext,
    ExpoInferenceRequest, ExpoInferenceResponse, ExpoMetrics, ExpoModule, ExpoPlugin, ExpoValue,
    LogFormat, LogLevel as ExpoLogLevel, ModuleRegistryConfig, PerformanceConfig, TargetPlatform,
};

// Re-export iOS background processing types
#[cfg(target_os = "ios")]
pub use ios_background::{
    iOSBackgroundConfig, iOSBackgroundManager, BackgroundInferenceResult, BackgroundPriority,
    BackgroundState, BackgroundStats, BackgroundTask, BackgroundTaskState, BackgroundTaskType,
};

// Re-export iOS App Extension types
#[cfg(target_os = "ios")]
pub use ios_app_extensions::{
    iOSAppExtensionManager, iOSExtensionConfig, iOSExtensionType, ExtensionBatchConfig,
    ExtensionBatteryConfig, ExtensionCacheLocation, ExtensionContext, ExtensionDataRetentionConfig,
    ExtensionError, ExtensionErrorCategory, ExtensionInferenceRequest, ExtensionInferenceResponse,
    ExtensionMetrics, ExtensionModelCacheConfig, ExtensionPerformanceConfig, ExtensionPriority,
    ExtensionPrivacyConfig, ExtensionResourceConfig,
};

// Re-export iOS iCloud sync types
#[cfg(target_os = "ios")]
pub use ios_icloud::{
    iCloudModelSync, iCloudSyncConfig, ConflictResolution, DatabaseScope, ModelMetadata,
    ModelSyncResult, SyncOperation, SyncResult, SyncStatistics, SyncStatus,
};

// Re-export Android RenderScript types
#[cfg(target_os = "android")]
pub use android_renderscript::{
    AndroidRenderScriptEngine, RSAllocationStrategy, RSKernelParams, RSKernelType,
    RenderScriptConfig, RenderScriptStats,
};

// Re-export Android Work Manager types
#[cfg(target_os = "android")]
pub use android_work_manager::{
    AndroidWorkManager, AndroidWorkManagerConfig, BackgroundExecutionConfig, CompressionAlgorithm,
    ConflictResolutionStrategy, DataCompressionConfig, DataSyncConfig, ExistingWorkPolicy,
    ForegroundNotificationConfig, StorageConstraints, TaskExecutionOrder, TaskPrioritizationConfig,
    WorkConstraintsConfig, WorkError, WorkErrorCategory, WorkExecutionMetrics, WorkFrequency,
    WorkInputData, WorkNetworkType, WorkPriority, WorkRequest, WorkRequestType, WorkResult,
    WorkResultData, WorkRetryInfo, WorkRetryPolicy, WorkRetryPolicyConfig, WorkStatus,
    WorkTaskType,
};

// Re-export Android Content Provider types
#[cfg(target_os = "android")]
pub use android_content_provider::{
    AccessLevel, AccessResult, AndroidModelContentProvider,
    CompressionAlgorithm as ContentCompressionAlgorithm, ContentProviderConfig,
    EncryptionAlgorithm, EncryptionConfig, KeyManagementConfig, ModelInfo, ModelMetadata,
    ModelPermissions, ModelStream, ModelType, Operation, PerformanceConfig, QueryParams,
    QueryResult, SecurityConfig, SortOrder,
};

// Re-export Android Doze compatibility types
#[cfg(target_os = "android")]
pub use android_doze_compatibility::{
    AndroidDozeCompatibilityManager, DeferredTask, DeviceState, DozeCompatibilityConfig,
    DozeCompatibilityStats, DozeCompatibilityUtils, DozeDetectionConfig, DozeMemoryStrategy,
    DozeNetworkConfig, DozePerformanceConfig, DozePerformanceImpact, DozeState, DozeSyncStrategy,
    DozeTaskConfig, DozeTaskType, InferenceResult, MaintenanceWindow, MaintenanceWindowType,
    NetworkRequest, NetworkResponse, TaskPriority, UserImpactLevel, WhitelistReason,
};

// Re-export ARKit integration types
#[cfg(target_os = "ios")]
pub use arkit_integration::{
    ARFrame, ARKitConfig, ARKitInferenceEngine, ARPerformanceConfig, ARProcessingResult,
    ARRenderingConfig, ARSessionConfig, ARSessionState, ARSessionStats, ARWorldMap, BoundingBox2D,
    BoundingBox3D, CameraImage, DepthBuffer, DepthFormat, DetectedPlane, Detection,
    DirectionalLight, FacePose, HandChirality, HandPose, ImageFormat, Joint, JointType,
    LightEstimate, LightEstimationConfig, LightEstimationMode, Matrix4x4, ObjectDetectionConfig,
    ObjectDetectionModel, PlaneClassification, PlaneDetectionConfig, PlaneOrientation, Pose,
    PoseEstimationConfig, PoseEstimationModel, QualityThresholds, Quaternion, RenderTarget,
    RenderingBackend, SceneAnalysis, SphericalHarmonics, Surface, SurfaceMaterial, TextureFormat,
    TrackingQuality, TrackingState, Vec2, Vec3,
};

// Re-export Edge TPU support types
#[cfg(target_os = "android")]
pub use edge_tpu_support::{
    BatchOptimizationConfig, CacheSettings, CompilationConfig, CompiledTPUModel, ComputeCapability,
    ConcurrencyConfig, CoolingSettings, DataType, DebugLevel, DefragmentationConfig,
    DeviceSelectionStrategy, DeviceStatus, EdgeTPUConfig, EdgeTPUEngine, InferenceResult,
    InferenceTask, LatencyOptimizationConfig, MemoryAllocationStrategy, MemoryLayout,
    MemoryRequirements, ModelMetadata, OptimizationLevel, PerformanceMode, PerformanceProfile,
    PipelineConfig, PowerBudgetConfig, PowerMode, SchedulingStrategy, TPUDebugConfig, TPUDevice,
    TPUDeviceConfig, TPUDeviceType, TPUMemoryConfig, TPUPerformanceConfig, TPUPowerConfig,
    TPUPrecision, TPUStats, TPUThermalConfig, TaskPriority, TemperatureThresholds, Tensor,
    TensorResult, TensorSpec, ThermalManagementStrategy,
};

/// Mobile deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileConfig {
    /// Target platform
    pub platform: MobilePlatform,
    /// Inference backend
    pub backend: MobileBackend,
    /// Memory optimization level
    pub memory_optimization: MemoryOptimization,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Use fp16 precision for inference
    pub use_fp16: bool,
    /// Enable quantization
    pub quantization: Option<MobileQuantizationConfig>,
    /// Thread pool size (0 = auto-detect)
    pub num_threads: usize,
    /// Enable batch processing
    pub enable_batching: bool,
    /// Maximum batch size
    pub max_batch_size: usize,
}

/// Target mobile platform
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MobilePlatform {
    /// iOS platform
    Ios,
    /// Android platform
    Android,
    /// Generic mobile (cross-platform)
    Generic,
}

/// Mobile inference backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MobileBackend {
    /// CPU-only inference
    CPU,
    /// Core ML (iOS)
    CoreML,
    /// Neural Networks API (Android)
    NNAPI,
    /// GPU acceleration (Metal/Vulkan)
    GPU,
    /// Metal acceleration (iOS specific)
    Metal,
    /// Vulkan acceleration (cross-platform)
    Vulkan,
    /// OpenCL acceleration (cross-platform)
    OpenCL,
    /// Custom optimized backend
    Custom,
}

/// Memory optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryOptimization {
    /// Minimal optimization (fastest inference)
    Minimal,
    /// Balanced optimization
    Balanced,
    /// Maximum optimization (lowest memory usage)
    Maximum,
}

/// Mobile quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileQuantizationConfig {
    /// Quantization scheme optimized for mobile
    pub scheme: MobileQuantizationScheme,
    /// Use dynamic quantization
    pub dynamic: bool,
    /// Per-channel quantization
    pub per_channel: bool,
}

/// Mobile-optimized quantization schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MobileQuantizationScheme {
    /// 8-bit integer quantization
    Int8,
    /// 4-bit integer quantization (ultra-low memory)
    Int4,
    /// Float16 quantization
    FP16,
    /// Dynamic quantization
    Dynamic,
}

impl Default for MobileConfig {
    fn default() -> Self {
        Self {
            platform: MobilePlatform::Generic,
            backend: MobileBackend::CPU,
            memory_optimization: MemoryOptimization::Balanced,
            max_memory_mb: 512, // Conservative default for mobile
            use_fp16: true,
            quantization: Some(MobileQuantizationConfig {
                scheme: MobileQuantizationScheme::Int8,
                dynamic: true,
                per_channel: false,
            }),
            num_threads: 0,         // Auto-detect
            enable_batching: false, // Conservative for mobile
            max_batch_size: 1,
        }
    }
}

impl MobileConfig {
    /// Create optimized configuration based on device detection
    pub fn auto_detect_optimized() -> Result<Self> {
        let device_info = crate::device_info::MobileDeviceDetector::detect()?;
        Ok(crate::device_info::MobileDeviceDetector::generate_optimized_config(&device_info))
    }

    /// Create optimized configuration for iOS
    pub fn ios_optimized() -> Self {
        Self {
            platform: MobilePlatform::Ios,
            backend: MobileBackend::CoreML,
            memory_optimization: MemoryOptimization::Balanced,
            max_memory_mb: 1024, // iOS devices typically have more RAM
            use_fp16: true,
            quantization: Some(MobileQuantizationConfig {
                scheme: MobileQuantizationScheme::FP16,
                dynamic: false,
                per_channel: true,
            }),
            num_threads: 0,
            enable_batching: true,
            max_batch_size: 4,
        }
    }

    /// Create optimized configuration for Android
    pub fn android_optimized() -> Self {
        Self {
            platform: MobilePlatform::Android,
            backend: MobileBackend::NNAPI,
            memory_optimization: MemoryOptimization::Balanced,
            max_memory_mb: 768, // Conservative for Android diversity
            use_fp16: true,
            quantization: Some(MobileQuantizationConfig {
                scheme: MobileQuantizationScheme::Int8,
                dynamic: true,
                per_channel: false,
            }),
            num_threads: 0,
            enable_batching: false,
            max_batch_size: 1,
        }
    }

    /// Create ultra-low memory configuration
    pub fn ultra_low_memory() -> Self {
        Self {
            platform: MobilePlatform::Generic,
            backend: MobileBackend::CPU,
            memory_optimization: MemoryOptimization::Maximum,
            max_memory_mb: 256,
            use_fp16: true,
            quantization: Some(MobileQuantizationConfig {
                scheme: MobileQuantizationScheme::Int4,
                dynamic: true,
                per_channel: true,
            }),
            num_threads: 1, // Single-threaded for minimum memory
            enable_batching: false,
            max_batch_size: 1,
        }
    }

    /// Validate configuration for mobile deployment
    pub fn validate(&self) -> Result<()> {
        // Check memory constraints
        if self.max_memory_mb < 64 {
            return Err(TrustformersError::config_error(
                "Mobile deployment requires at least 64MB memory",
                "validate",
            ));
        }

        if self.max_memory_mb > 4096 {
            return Err(TrustformersError::config_error(
                "Mobile deployment should not exceed 4GB memory",
                "validate",
            ));
        }

        // Validate platform-backend compatibility
        match (self.platform, self.backend) {
            (MobilePlatform::Ios, MobileBackend::NNAPI) => {
                return Err(TrustformersError::config_error(
                    "NNAPI backend is not available on iOS",
                    "validate",
                ));
            },
            (MobilePlatform::Android, MobileBackend::CoreML) => {
                return Err(TrustformersError::config_error(
                    "Core ML backend is not available on Android",
                    "validate",
                ));
            },
            _ => {},
        }

        // Validate batch size
        if self.enable_batching && self.max_batch_size == 0 {
            return Err(TrustformersError::config_error(
                "Batch size must be > 0 when batching is enabled",
                "validate",
            ));
        }

        // Validate thread count
        if self.num_threads > 16 {
            return Err(TrustformersError::config_error(
                "Mobile deployment should not use more than 16 threads",
                "validate",
            ));
        }

        Ok(())
    }

    /// Estimate memory usage for the configuration
    pub fn estimate_memory_usage(&self, model_size_mb: usize) -> usize {
        let mut total_memory = model_size_mb;

        // Add quantization overhead
        if let Some(ref quant_config) = self.quantization {
            let reduction_factor = match quant_config.scheme {
                MobileQuantizationScheme::Int4 => 8,    // 4-bit = 1/8 of FP32
                MobileQuantizationScheme::Int8 => 4,    // 8-bit = 1/4 of FP32
                MobileQuantizationScheme::FP16 => 2,    // 16-bit = 1/2 of FP32
                MobileQuantizationScheme::Dynamic => 3, // Approximate
            };
            total_memory = model_size_mb / reduction_factor;
        } else if self.use_fp16 {
            total_memory = model_size_mb / 2; // FP16 is half of FP32
        }

        // Add runtime overhead
        let runtime_overhead = match self.memory_optimization {
            MemoryOptimization::Minimal => total_memory / 2, // 50% overhead
            MemoryOptimization::Balanced => total_memory / 4, // 25% overhead
            MemoryOptimization::Maximum => total_memory / 8, // 12.5% overhead
        };

        total_memory + runtime_overhead
    }

    /// Get recommended thread count for the platform
    pub fn get_thread_count(&self) -> usize {
        if self.num_threads > 0 {
            return self.num_threads;
        }

        // Auto-detect based on platform and optimization level
        let base_threads = match self.platform {
            MobilePlatform::Ios => 4,     // iOS devices typically have 4-8 cores
            MobilePlatform::Android => 2, // Conservative for Android diversity
            MobilePlatform::Generic => 2, // Conservative default
        };

        match self.memory_optimization {
            MemoryOptimization::Maximum => 1, // Single-threaded for minimum memory
            MemoryOptimization::Balanced => base_threads, // Balanced approach
            MemoryOptimization::Minimal => base_threads * 2, // More threads for speed
        }
    }
}

/// Mobile deployment statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileStats {
    /// Platform information
    pub platform: MobilePlatform,
    /// Backend in use
    pub backend: MobileBackend,
    /// Current memory usage (MB)
    pub memory_usage_mb: usize,
    /// Peak memory usage (MB)
    pub peak_memory_mb: usize,
    /// Average inference time (ms)
    pub avg_inference_time_ms: f32,
    /// Total inferences performed
    pub total_inferences: usize,
    /// Current thread count
    pub thread_count: usize,
    /// Quantization status
    pub quantization_enabled: bool,
    /// FP16 status
    pub fp16_enabled: bool,
}

impl MobileStats {
    /// Create new statistics tracker
    pub fn new(config: &MobileConfig) -> Self {
        Self {
            platform: config.platform,
            backend: config.backend,
            memory_usage_mb: 0,
            peak_memory_mb: 0,
            avg_inference_time_ms: 0.0,
            total_inferences: 0,
            thread_count: config.get_thread_count(),
            quantization_enabled: config.quantization.is_some(),
            fp16_enabled: config.use_fp16,
        }
    }

    /// Update inference statistics
    pub fn update_inference(&mut self, inference_time_ms: f32) {
        self.total_inferences += 1;

        // Update running average
        let alpha = 0.1; // Smoothing factor
        if self.total_inferences == 1 {
            self.avg_inference_time_ms = inference_time_ms;
        } else {
            self.avg_inference_time_ms =
                alpha * inference_time_ms + (1.0 - alpha) * self.avg_inference_time_ms;
        }
    }

    /// Update memory statistics
    pub fn update_memory(&mut self, current_memory_mb: usize) {
        self.memory_usage_mb = current_memory_mb;
        self.peak_memory_mb = self.peak_memory_mb.max(current_memory_mb);
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> String {
        format!(
            "Mobile Performance Summary:\n\
             Platform: {:?}\n\
             Backend: {:?}\n\
             Memory: {} MB (peak: {} MB)\n\
             Avg Inference: {:.2} ms\n\
             Total Inferences: {}\n\
             Threads: {}\n\
             Optimizations: FP16={}, Quantization={}",
            self.platform,
            self.backend,
            self.memory_usage_mb,
            self.peak_memory_mb,
            self.avg_inference_time_ms,
            self.total_inferences,
            self.thread_count,
            self.fp16_enabled,
            self.quantization_enabled
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobile_config_defaults() {
        let config = MobileConfig::default();
        assert_eq!(config.platform, MobilePlatform::Generic);
        assert_eq!(config.backend, MobileBackend::CPU);
        assert!(config.use_fp16);
        assert!(config.quantization.is_some());
    }

    #[test]
    fn test_ios_optimized_config() {
        let config = MobileConfig::ios_optimized();
        assert_eq!(config.platform, MobilePlatform::Ios);
        assert_eq!(config.backend, MobileBackend::CoreML);
        assert!(config.enable_batching);
        assert_eq!(config.max_batch_size, 4);
    }

    #[test]
    fn test_android_optimized_config() {
        let config = MobileConfig::android_optimized();
        assert_eq!(config.platform, MobilePlatform::Android);
        assert_eq!(config.backend, MobileBackend::NNAPI);
        assert!(!config.enable_batching);
    }

    #[test]
    fn test_ultra_low_memory_config() {
        let config = MobileConfig::ultra_low_memory();
        assert_eq!(config.memory_optimization, MemoryOptimization::Maximum);
        assert_eq!(config.max_memory_mb, 256);
        assert_eq!(config.num_threads, 1);

        if let Some(ref quant) = config.quantization {
            assert_eq!(quant.scheme, MobileQuantizationScheme::Int4);
        }
    }

    #[test]
    fn test_config_validation() {
        let mut config = MobileConfig::default();
        assert!(config.validate().is_ok());

        // Test invalid memory
        config.max_memory_mb = 32;
        assert!(config.validate().is_err());

        config.max_memory_mb = 512;
        assert!(config.validate().is_ok());

        // Test platform-backend mismatch
        config.platform = MobilePlatform::Ios;
        config.backend = MobileBackend::NNAPI;
        assert!(config.validate().is_err());

        config.backend = MobileBackend::CoreML;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_memory_estimation() {
        let config = MobileConfig::default();
        let model_size = 100; // 100MB

        let estimated = config.estimate_memory_usage(model_size);
        assert!(estimated < model_size); // Should be smaller due to quantization
        assert!(estimated > 0);
    }

    #[test]
    fn test_thread_count_detection() {
        let mut config = MobileConfig::default();

        // Auto-detect
        config.num_threads = 0;
        assert!(config.get_thread_count() > 0);

        // Manual setting
        config.num_threads = 4;
        assert_eq!(config.get_thread_count(), 4);
    }

    #[test]
    fn test_mobile_stats() {
        let config = MobileConfig::default();
        let mut stats = MobileStats::new(&config);

        assert_eq!(stats.total_inferences, 0);
        assert_eq!(stats.memory_usage_mb, 0);

        // Update with inference
        stats.update_inference(50.0);
        assert_eq!(stats.total_inferences, 1);
        assert_eq!(stats.avg_inference_time_ms, 50.0);

        // Update with memory
        stats.update_memory(128);
        assert_eq!(stats.memory_usage_mb, 128);
        assert_eq!(stats.peak_memory_mb, 128);

        // Check summary
        let summary = stats.get_performance_summary();
        assert!(summary.contains("Platform"));
        assert!(summary.contains("Memory"));
    }
}
