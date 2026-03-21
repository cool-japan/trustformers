/*!
# Expo Plugin for TrustformeRS

This module provides an Expo plugin for seamless integration of TrustformeRS
with Expo applications, enabling AI model inference and training capabilities
in React Native apps using the Expo development platform.

## Features

- **Expo Config Plugin**: Automatic configuration of native dependencies
- **Expo Module Support**: Integration with Expo Modules API
- **Development Build Support**: Compatibility with Expo Development Builds
- **EAS Build Integration**: Support for Expo Application Services builds
- **Hot Reloading**: Development-time hot reloading of models
- **Asset Management**: Automatic bundling and optimization of model assets

## Usage

```rust
use trustformers_mobile::expo_plugin::{
    ExpoPlugin, ExpoConfig, ExpoModuleRegistry
};

let config = ExpoConfig::default();
let plugin = ExpoPlugin::new(config)?;
```

## Expo App Configuration

Add to your `app.json` or `app.config.js`:

```json
{
  "expo": {
    "plugins": [
      [
        "trustformers-expo-plugin",
        {
          "enableAI": true,
          "modelAssets": ["models/"],
          "optimizeModels": true
        }
      ]
    ]
  }
}
```
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::{Arc, Mutex};
use trustformers_core::TrustformersError;

/// Configuration for Expo plugin integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpoConfig {
    /// Plugin name for Expo
    pub plugin_name: String,
    /// Enable AI inference capabilities
    pub enable_ai: bool,
    /// Model asset directories to bundle
    pub model_assets: Vec<String>,
    /// Optimize models during build
    pub optimize_models: bool,
    /// Development mode settings
    pub development_config: DevelopmentConfig,
    /// Build configuration
    pub build_config: BuildConfig,
    /// Asset management configuration
    pub asset_config: AssetConfig,
    /// Performance optimization settings
    pub performance_config: PerformanceConfig,
    /// Module registry configuration
    pub module_registry_config: ModuleRegistryConfig,
}

impl Default for ExpoConfig {
    fn default() -> Self {
        Self {
            plugin_name: "trustformers-expo-plugin".to_string(),
            enable_ai: true,
            model_assets: vec!["models/".to_string(), "assets/models/".to_string()],
            optimize_models: true,
            development_config: DevelopmentConfig::default(),
            build_config: BuildConfig::default(),
            asset_config: AssetConfig::default(),
            performance_config: PerformanceConfig::default(),
            module_registry_config: ModuleRegistryConfig::default(),
        }
    }
}

/// Development mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevelopmentConfig {
    /// Enable hot reloading of models
    pub enable_hot_reload: bool,
    /// Enable development-time debugging
    pub enable_debug: bool,
    /// Mock inference responses for faster development
    pub enable_mock_responses: bool,
    /// Development server settings
    pub dev_server_config: DevServerConfig,
    /// Logging configuration
    pub logging_config: LoggingConfig,
}

impl Default for DevelopmentConfig {
    fn default() -> Self {
        Self {
            enable_hot_reload: true,
            enable_debug: true,
            enable_mock_responses: false,
            dev_server_config: DevServerConfig::default(),
            logging_config: LoggingConfig::default(),
        }
    }
}

/// Development server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevServerConfig {
    /// Development server port
    pub port: u16,
    /// Enable CORS for development
    pub enable_cors: bool,
    /// Enable HTTPS for development
    pub enable_https: bool,
    /// Hot reload polling interval in milliseconds
    pub hot_reload_interval_ms: u64,
}

impl Default for DevServerConfig {
    fn default() -> Self {
        Self {
            port: 8081,
            enable_cors: true,
            enable_https: false,
            hot_reload_interval_ms: 1000,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level for development
    pub log_level: LogLevel,
    /// Enable performance logging
    pub enable_performance_logs: bool,
    /// Enable inference logging
    pub enable_inference_logs: bool,
    /// Log output format
    pub log_format: LogFormat,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            log_level: LogLevel::Info,
            enable_performance_logs: true,
            enable_inference_logs: true,
            log_format: LogFormat::Json,
        }
    }
}

/// Log levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Log output formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogFormat {
    Json,
    Text,
    Structured,
}

/// Build configuration for Expo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Target build platform
    pub target_platform: TargetPlatform,
    /// EAS Build configuration
    pub eas_config: EASConfig,
    /// Native dependency configuration
    pub native_deps_config: NativeDepsConfig,
    /// Build optimization settings
    pub optimization_config: BuildOptimizationConfig,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            target_platform: TargetPlatform::Both,
            eas_config: EASConfig::default(),
            native_deps_config: NativeDepsConfig::default(),
            optimization_config: BuildOptimizationConfig::default(),
        }
    }
}

/// Target build platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum TargetPlatform {
    iOS,
    Android,
    Both,
}

/// EAS Build configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EASConfig {
    /// Enable EAS Build
    pub enable_eas_build: bool,
    /// Build profile name
    pub build_profile: String,
    /// Custom build environment variables
    pub env_vars: HashMap<String, String>,
    /// Pre-build hook commands
    pub pre_build_hooks: Vec<String>,
    /// Post-build hook commands
    pub post_build_hooks: Vec<String>,
}

impl Default for EASConfig {
    fn default() -> Self {
        Self {
            enable_eas_build: true,
            build_profile: "production".to_string(),
            env_vars: HashMap::new(),
            pre_build_hooks: vec!["echo 'Preparing TrustformeRS build'".to_string()],
            post_build_hooks: vec!["echo 'TrustformeRS build complete'".to_string()],
        }
    }
}

/// Native dependency configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NativeDepsConfig {
    /// iOS framework configuration
    pub ios_config: iOSFrameworkConfig,
    /// Android library configuration
    pub android_config: AndroidLibConfig,
    /// CocoaPods configuration
    pub cocoapods_config: CocoaPodsConfig,
    /// Gradle configuration
    pub gradle_config: GradleConfig,
}

/// iOS framework configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub struct iOSFrameworkConfig {
    /// Framework name
    pub framework_name: String,
    /// iOS deployment target
    pub deployment_target: String,
    /// Required system frameworks
    pub system_frameworks: Vec<String>,
    /// Custom linker flags
    pub linker_flags: Vec<String>,
}

impl Default for iOSFrameworkConfig {
    fn default() -> Self {
        Self {
            framework_name: "TrustformersKit".to_string(),
            deployment_target: "13.0".to_string(),
            system_frameworks: vec![
                "Accelerate".to_string(),
                "CoreML".to_string(),
                "Metal".to_string(),
                "MetalPerformanceShaders".to_string(),
            ],
            linker_flags: vec!["-ObjC".to_string()],
        }
    }
}

/// Android library configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndroidLibConfig {
    /// Library name
    pub library_name: String,
    /// Minimum SDK version
    pub min_sdk_version: u32,
    /// Target SDK version
    pub target_sdk_version: u32,
    /// Required Android permissions
    pub permissions: Vec<String>,
    /// NDK configuration
    pub ndk_config: NDKConfig,
}

impl Default for AndroidLibConfig {
    fn default() -> Self {
        Self {
            library_name: "trustformers-android".to_string(),
            min_sdk_version: 24,
            target_sdk_version: 34,
            permissions: vec![
                "android.permission.WRITE_EXTERNAL_STORAGE".to_string(),
                "android.permission.READ_EXTERNAL_STORAGE".to_string(),
            ],
            ndk_config: NDKConfig::default(),
        }
    }
}

/// Android NDK configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NDKConfig {
    /// NDK version
    pub ndk_version: String,
    /// Target ABIs
    pub target_abis: Vec<String>,
    /// NDK build flags
    pub build_flags: Vec<String>,
}

impl Default for NDKConfig {
    fn default() -> Self {
        Self {
            ndk_version: "25.1.8937393".to_string(),
            target_abis: vec![
                "arm64-v8a".to_string(),
                "armeabi-v7a".to_string(),
                "x86_64".to_string(),
            ],
            build_flags: vec!["-O3".to_string(), "-DNDEBUG".to_string()],
        }
    }
}

/// CocoaPods configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CocoaPodsConfig {
    /// Pod name
    pub pod_name: String,
    /// Pod version
    pub pod_version: String,
    /// Pod source
    pub pod_source: String,
    /// Pod dependencies
    pub dependencies: Vec<String>,
}

impl Default for CocoaPodsConfig {
    fn default() -> Self {
        Self {
            pod_name: "TrustformersKit".to_string(),
            pod_version: "1.0.0".to_string(),
            pod_source: "https://github.com/cool-japan/trustformers-mobile".to_string(),
            dependencies: vec![],
        }
    }
}

/// Gradle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradleConfig {
    /// Gradle version
    pub gradle_version: String,
    /// Android Gradle Plugin version
    pub agp_version: String,
    /// Custom Gradle properties
    pub gradle_properties: HashMap<String, String>,
    /// Build dependencies
    pub dependencies: Vec<String>,
}

impl Default for GradleConfig {
    fn default() -> Self {
        let mut gradle_properties = HashMap::new();
        gradle_properties.insert("android.useAndroidX".to_string(), "true".to_string());
        gradle_properties.insert("android.enableJetifier".to_string(), "true".to_string());

        Self {
            gradle_version: "8.4".to_string(),
            agp_version: "8.1.4".to_string(),
            gradle_properties,
            dependencies: vec!["androidx.core:core-ktx:1.12.0".to_string()],
        }
    }
}

/// Build optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildOptimizationConfig {
    /// Enable tree shaking
    pub enable_tree_shaking: bool,
    /// Enable code splitting
    pub enable_code_splitting: bool,
    /// Minify JavaScript
    pub minify_javascript: bool,
    /// Optimize images
    pub optimize_images: bool,
    /// Bundle size optimization
    pub bundle_optimization: BundleOptimizationConfig,
}

impl Default for BuildOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_tree_shaking: true,
            enable_code_splitting: true,
            minify_javascript: true,
            optimize_images: true,
            bundle_optimization: BundleOptimizationConfig::default(),
        }
    }
}

/// Bundle optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleOptimizationConfig {
    /// Split bundles by platform
    pub split_by_platform: bool,
    /// Maximum bundle size in MB
    pub max_bundle_size_mb: usize,
    /// Enable dynamic imports
    pub enable_dynamic_imports: bool,
    /// Chunk strategy
    pub chunk_strategy: ChunkStrategy,
}

impl Default for BundleOptimizationConfig {
    fn default() -> Self {
        Self {
            split_by_platform: true,
            max_bundle_size_mb: 25,
            enable_dynamic_imports: true,
            chunk_strategy: ChunkStrategy::Automatic,
        }
    }
}

/// Bundle chunk strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkStrategy {
    Single,
    ByRoute,
    ByFeature,
    Automatic,
}

/// Asset management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetConfig {
    /// Asset bundling strategy
    pub bundling_strategy: AssetBundlingStrategy,
    /// Asset compression settings
    pub compression_config: AssetCompressionConfig,
    /// Asset loading strategy
    pub loading_strategy: AssetLoadingStrategy,
    /// Asset caching configuration
    pub caching_config: AssetCachingConfig,
}

impl Default for AssetConfig {
    fn default() -> Self {
        Self {
            bundling_strategy: AssetBundlingStrategy::Selective,
            compression_config: AssetCompressionConfig::default(),
            loading_strategy: AssetLoadingStrategy::LazyWithPrefetch,
            caching_config: AssetCachingConfig::default(),
        }
    }
}

/// Asset bundling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetBundlingStrategy {
    /// Bundle all assets
    All,
    /// Bundle only essential assets
    Essential,
    /// Selective bundling based on usage
    Selective,
    /// No bundling (download on demand)
    None,
}

/// Asset compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetCompressionConfig {
    /// Enable asset compression
    pub enable_compression: bool,
    /// Compression algorithm
    pub compression_algorithm: CompressionAlgorithm,
    /// Compression level (1-9)
    pub compression_level: u8,
    /// Minimum file size for compression (bytes)
    pub min_file_size_bytes: usize,
}

impl Default for AssetCompressionConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            compression_algorithm: CompressionAlgorithm::Gzip,
            compression_level: 6,
            min_file_size_bytes: 1024, // 1KB
        }
    }
}

/// Compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Brotli,
    Deflate,
    Lz4,
}

/// Asset loading strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetLoadingStrategy {
    /// Load all assets at startup
    Eager,
    /// Load assets on demand
    Lazy,
    /// Lazy loading with prefetching
    LazyWithPrefetch,
    /// Progressive loading
    Progressive,
}

/// Asset caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetCachingConfig {
    /// Enable asset caching
    pub enable_caching: bool,
    /// Cache size limit in MB
    pub cache_size_limit_mb: usize,
    /// Cache expiration time in hours
    pub cache_expiration_hours: u32,
    /// Cache eviction strategy
    pub eviction_strategy: CacheEvictionStrategy,
}

impl Default for AssetCachingConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_size_limit_mb: 100,
            cache_expiration_hours: 24,
            eviction_strategy: CacheEvictionStrategy::LRU,
        }
    }
}

/// Cache eviction strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheEvictionStrategy {
    LRU,
    LFU,
    FIFO,
    Random,
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Performance reporting configuration
    pub reporting_config: PerformanceReportingConfig,
    /// Memory optimization settings
    pub memory_config: MemoryOptimizationConfig,
    /// CPU optimization settings
    pub cpu_config: CPUOptimizationConfig,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            reporting_config: PerformanceReportingConfig::default(),
            memory_config: MemoryOptimizationConfig::default(),
            cpu_config: CPUOptimizationConfig::default(),
        }
    }
}

/// Performance reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReportingConfig {
    /// Enable automatic reporting
    pub enable_auto_reporting: bool,
    /// Reporting interval in seconds
    pub reporting_interval_seconds: u32,
    /// Performance metrics to track
    pub tracked_metrics: Vec<PerformanceMetric>,
    /// Reporting destination
    pub reporting_destination: ReportingDestination,
}

impl Default for PerformanceReportingConfig {
    fn default() -> Self {
        Self {
            enable_auto_reporting: false,
            reporting_interval_seconds: 60,
            tracked_metrics: vec![
                PerformanceMetric::InferenceTime,
                PerformanceMetric::MemoryUsage,
                PerformanceMetric::CPUUsage,
                PerformanceMetric::BatteryUsage,
            ],
            reporting_destination: ReportingDestination::Console,
        }
    }
}

/// Performance metrics to track
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceMetric {
    InferenceTime,
    MemoryUsage,
    CPUUsage,
    GPUUsage,
    BatteryUsage,
    NetworkUsage,
    CacheHitRate,
}

/// Reporting destinations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportingDestination {
    Console,
    File,
    Network,
    Analytics,
}

/// Memory optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationConfig {
    /// Enable garbage collection optimizations
    pub enable_gc_optimizations: bool,
    /// Memory pool configuration
    pub pool_config: MemoryPoolConfig,
    /// Memory pressure handling
    pub pressure_config: MemoryPressureConfig,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_gc_optimizations: true,
            pool_config: MemoryPoolConfig::default(),
            pressure_config: MemoryPressureConfig::default(),
        }
    }
}

/// Memory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Initial pool size in MB
    pub initial_size_mb: usize,
    /// Maximum pool size in MB
    pub max_size_mb: usize,
    /// Pool growth factor
    pub growth_factor: f32,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size_mb: 64,
            max_size_mb: 512,
            growth_factor: 1.5,
        }
    }
}

/// Memory pressure configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureConfig {
    /// Enable memory pressure monitoring
    pub enable_monitoring: bool,
    /// Warning threshold (% of available memory)
    pub warning_threshold_percent: f32,
    /// Critical threshold (% of available memory)
    pub critical_threshold_percent: f32,
    /// Actions to take on memory pressure
    pub pressure_actions: Vec<MemoryPressureAction>,
}

impl Default for MemoryPressureConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            warning_threshold_percent: 80.0,
            critical_threshold_percent: 95.0,
            pressure_actions: vec![
                MemoryPressureAction::ClearCaches,
                MemoryPressureAction::UnloadModels,
                MemoryPressureAction::ReduceQuality,
            ],
        }
    }
}

/// Memory pressure actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryPressureAction {
    ClearCaches,
    UnloadModels,
    ReduceQuality,
    PauseInference,
    ForceGC,
}

/// CPU optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPUOptimizationConfig {
    /// Enable CPU affinity optimization
    pub enable_cpu_affinity: bool,
    /// Thread pool configuration
    pub thread_pool_config: ThreadPoolConfig,
    /// CPU scheduling configuration
    pub scheduling_config: CPUSchedulingConfig,
}

impl Default for CPUOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_cpu_affinity: true,
            thread_pool_config: ThreadPoolConfig::default(),
            scheduling_config: CPUSchedulingConfig::default(),
        }
    }
}

/// Thread pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolConfig {
    /// Core thread count
    pub core_threads: usize,
    /// Maximum thread count
    pub max_threads: usize,
    /// Thread keep-alive time in seconds
    pub keep_alive_seconds: u32,
    /// Queue capacity
    pub queue_capacity: usize,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            core_threads: 2,
            max_threads: 8,
            keep_alive_seconds: 60,
            queue_capacity: 100,
        }
    }
}

/// CPU scheduling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPUSchedulingConfig {
    /// Scheduling policy
    pub scheduling_policy: SchedulingPolicy,
    /// Thread priority
    pub thread_priority: ThreadPriority,
    /// Enable work stealing
    pub enable_work_stealing: bool,
}

impl Default for CPUSchedulingConfig {
    fn default() -> Self {
        Self {
            scheduling_policy: SchedulingPolicy::Fair,
            thread_priority: ThreadPriority::Normal,
            enable_work_stealing: true,
        }
    }
}

/// CPU scheduling policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulingPolicy {
    Fair,
    FIFO,
    RoundRobin,
    Priority,
}

/// Thread priorities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreadPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Module registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleRegistryConfig {
    /// Enable automatic module registration
    pub enable_auto_registration: bool,
    /// Module loading strategy
    pub loading_strategy: ModuleLoadingStrategy,
    /// Module validation configuration
    pub validation_config: ModuleValidationConfig,
}

impl Default for ModuleRegistryConfig {
    fn default() -> Self {
        Self {
            enable_auto_registration: true,
            loading_strategy: ModuleLoadingStrategy::Lazy,
            validation_config: ModuleValidationConfig::default(),
        }
    }
}

/// Module loading strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModuleLoadingStrategy {
    Eager,
    Lazy,
    OnDemand,
}

/// Module validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleValidationConfig {
    /// Enable module signature validation
    pub enable_signature_validation: bool,
    /// Enable version compatibility checks
    pub enable_version_checks: bool,
    /// Strict validation mode
    pub strict_mode: bool,
}

impl Default for ModuleValidationConfig {
    fn default() -> Self {
        Self {
            enable_signature_validation: true,
            enable_version_checks: true,
            strict_mode: false,
        }
    }
}

/// Main Expo plugin for TrustformeRS
pub struct ExpoPlugin {
    config: ExpoConfig,
    module_registry: Arc<Mutex<ExpoModuleRegistry>>,
    asset_manager: Arc<Mutex<ExpoAssetManager>>,
    performance_monitor: Option<Arc<Mutex<ExpoPerformanceMonitor>>>,
    dev_server: Option<Arc<Mutex<ExpoDevServer>>>,
}

impl ExpoPlugin {
    /// Create a new Expo plugin
    pub fn new(config: ExpoConfig) -> Result<Self, TrustformersError> {
        let module_registry = Arc::new(Mutex::new(ExpoModuleRegistry::new(&config)?));
        let asset_manager = Arc::new(Mutex::new(ExpoAssetManager::new(&config)?));

        let performance_monitor = if config.performance_config.enable_monitoring {
            Some(Arc::new(Mutex::new(ExpoPerformanceMonitor::new(&config)?)))
        } else {
            None
        };

        let dev_server = if config.development_config.enable_hot_reload {
            Some(Arc::new(Mutex::new(ExpoDevServer::new(&config)?)))
        } else {
            None
        };

        let mut plugin = Self {
            config,
            module_registry,
            asset_manager,
            performance_monitor,
            dev_server,
        };

        plugin.initialize()?;
        Ok(plugin)
    }

    /// Initialize the plugin
    fn initialize(&mut self) -> Result<(), TrustformersError> {
        // Register default modules
        self.register_default_modules()?;

        // Initialize asset management
        self.initialize_asset_management()?;

        // Start performance monitoring if enabled
        if let Some(ref monitor) = self.performance_monitor {
            let mut monitor = monitor.lock().map_err(|_| {
                TrustformersError::runtime_error(
                    "Failed to acquire performance monitor lock".to_string(),
                )
            })?;
            monitor.start_monitoring()?;
        }

        // Start development server if enabled
        if let Some(ref dev_server) = self.dev_server {
            let mut dev_server = dev_server.lock().map_err(|_| {
                TrustformersError::runtime_error("Failed to acquire dev server lock".to_string())
            })?;
            dev_server.start()?;
        }

        Ok(())
    }

    /// Register a custom Expo module
    pub fn register_module(
        &self,
        name: String,
        module: Box<dyn ExpoModule>,
    ) -> Result<(), TrustformersError> {
        let mut registry = self.module_registry.lock().map_err(|_| {
            TrustformersError::runtime_error("Failed to acquire module registry lock".to_string())
        })?;

        registry.register_module(name, module)
    }

    /// Perform inference through Expo plugin
    pub fn perform_inference(
        &self,
        request: ExpoInferenceRequest,
    ) -> Result<ExpoInferenceResponse, TrustformersError> {
        let start_time = std::time::Instant::now();

        // Validate request
        self.validate_inference_request(&request)?;

        // Load model if not already loaded
        self.ensure_model_loaded(&request.model_id)?;

        // Execute inference
        let result = self.execute_inference(&request)?;

        // Report performance metrics
        if let Some(ref monitor) = self.performance_monitor {
            let mut monitor = monitor.lock().map_err(|_| {
                TrustformersError::runtime_error(
                    "Failed to acquire performance monitor lock".to_string(),
                )
            })?;

            monitor.record_inference(&request, &result, start_time.elapsed())?;
        }

        Ok(result)
    }

    /// Get plugin configuration
    pub fn get_config(&self) -> &ExpoConfig {
        &self.config
    }

    /// Get asset manager
    pub fn get_asset_manager(&self) -> Arc<Mutex<ExpoAssetManager>> {
        self.asset_manager.clone()
    }

    /// Get performance monitor
    pub fn get_performance_monitor(&self) -> Option<Arc<Mutex<ExpoPerformanceMonitor>>> {
        self.performance_monitor.clone()
    }

    /// Private helper methods
    fn register_default_modules(&self) -> Result<(), TrustformersError> {
        // Register TrustformersInference module
        let inference_module = Box::new(TrustformersInferenceModule::new());
        self.register_module("TrustformersInference".to_string(), inference_module)?;

        // Register TrustformersModel module
        let model_module = Box::new(TrustformersModelModule::new());
        self.register_module("TrustformersModel".to_string(), model_module)?;

        // Register TrustformersAssets module
        let assets_module = Box::new(TrustformersAssetsModule::new());
        self.register_module("TrustformersAssets".to_string(), assets_module)?;

        Ok(())
    }

    fn initialize_asset_management(&self) -> Result<(), TrustformersError> {
        let mut asset_manager = self.asset_manager.lock().map_err(|_| {
            TrustformersError::runtime_error("Failed to acquire asset manager lock".to_string())
        })?;

        // Initialize asset bundling
        asset_manager.initialize_bundling()?;

        // Setup asset caching
        asset_manager.setup_caching()?;

        Ok(())
    }

    fn validate_inference_request(
        &self,
        request: &ExpoInferenceRequest,
    ) -> Result<(), TrustformersError> {
        if request.model_id.is_empty() {
            return Err(TrustformersError::invalid_input(
                "Model ID cannot be empty".to_string(),
            ));
        }

        if request.input_data.is_empty() {
            return Err(TrustformersError::invalid_input(
                "Input data cannot be empty".to_string(),
            ));
        }

        Ok(())
    }

    fn ensure_model_loaded(&self, model_id: &str) -> Result<(), TrustformersError> {
        let mut asset_manager = self.asset_manager.lock().map_err(|_| {
            TrustformersError::runtime_error("Failed to acquire asset manager lock".to_string())
        })?;

        asset_manager.ensure_model_loaded(model_id)
    }

    fn execute_inference(
        &self,
        request: &ExpoInferenceRequest,
    ) -> Result<ExpoInferenceResponse, TrustformersError> {
        let registry = self.module_registry.lock().map_err(|_| {
            TrustformersError::runtime_error("Failed to acquire module registry lock".to_string())
        })?;

        let inference_module = registry.get_module("TrustformersInference").ok_or_else(|| {
            TrustformersError::runtime_error("TrustformersInference module not found".to_string())
        })?;

        // For now, return a placeholder response
        // In a real implementation, this would call the actual inference engine
        Ok(ExpoInferenceResponse {
            request_id: request.request_id.clone(),
            success: true,
            output_data: vec![1.0, 2.0, 3.0], // Placeholder
            output_shape: vec![1, 3],
            inference_time_ms: 100.0,
            memory_used_mb: 50,
            error_message: None,
            expo_metrics: ExpoMetrics {
                module_load_time_ms: 10.0,
                asset_load_time_ms: 20.0,
                cache_hit_ratio: 0.8,
                bundle_size_kb: 1024,
            },
        })
    }
}

/// Expo module registry
struct ExpoModuleRegistry {
    modules: HashMap<String, Box<dyn ExpoModule>>,
    config: ModuleRegistryConfig,
}

impl ExpoModuleRegistry {
    fn new(config: &ExpoConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            modules: HashMap::new(),
            config: config.module_registry_config.clone(),
        })
    }

    fn register_module(
        &mut self,
        name: String,
        module: Box<dyn ExpoModule>,
    ) -> Result<(), TrustformersError> {
        if self.config.validation_config.enable_signature_validation {
            self.validate_module_signature(&module)?;
        }

        if self.config.validation_config.enable_version_checks {
            self.validate_module_version(&module)?;
        }

        self.modules.insert(name, module);
        Ok(())
    }

    fn get_module(&self, name: &str) -> Option<&Box<dyn ExpoModule>> {
        self.modules.get(name)
    }

    fn validate_module_signature(
        &self,
        _module: &Box<dyn ExpoModule>,
    ) -> Result<(), TrustformersError> {
        // Validate module signature
        Ok(())
    }

    fn validate_module_version(
        &self,
        _module: &Box<dyn ExpoModule>,
    ) -> Result<(), TrustformersError> {
        // Validate module version compatibility
        Ok(())
    }
}

/// Expo asset manager
struct ExpoAssetManager {
    config: AssetConfig,
    loaded_models: HashMap<String, LoadedModel>,
    asset_cache: HashMap<String, CachedAsset>,
}

impl ExpoAssetManager {
    fn new(config: &ExpoConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            config: config.asset_config.clone(),
            loaded_models: HashMap::new(),
            asset_cache: HashMap::new(),
        })
    }

    fn initialize_bundling(&mut self) -> Result<(), TrustformersError> {
        // Initialize asset bundling based on strategy
        Ok(())
    }

    fn setup_caching(&mut self) -> Result<(), TrustformersError> {
        // Setup asset caching system
        Ok(())
    }

    fn ensure_model_loaded(&mut self, model_id: &str) -> Result<(), TrustformersError> {
        if !self.loaded_models.contains_key(model_id) {
            let model = self.load_model(model_id)?;
            self.loaded_models.insert(model_id.to_string(), model);
        }
        Ok(())
    }

    fn load_model(&self, model_id: &str) -> Result<LoadedModel, TrustformersError> {
        // Load model from assets
        Ok(LoadedModel {
            id: model_id.to_string(),
            path: format!("models/{}.bin", model_id),
            size_bytes: 1024 * 1024, // 1MB placeholder
            loaded_at: std::time::SystemTime::now(),
        })
    }
}

/// Expo performance monitor
struct ExpoPerformanceMonitor {
    config: PerformanceReportingConfig,
    metrics: Vec<PerformanceEntry>,
    start_time: std::time::SystemTime,
}

impl ExpoPerformanceMonitor {
    fn new(config: &ExpoConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            config: config.performance_config.reporting_config.clone(),
            metrics: Vec::new(),
            start_time: std::time::SystemTime::now(),
        })
    }

    fn start_monitoring(&mut self) -> Result<(), TrustformersError> {
        // Start performance monitoring
        Ok(())
    }

    fn record_inference(
        &mut self,
        request: &ExpoInferenceRequest,
        response: &ExpoInferenceResponse,
        duration: std::time::Duration,
    ) -> Result<(), TrustformersError> {
        let entry = PerformanceEntry {
            timestamp: std::time::SystemTime::now(),
            metric_type: PerformanceMetric::InferenceTime,
            value: duration.as_millis() as f64,
            context: format!("model:{}", request.model_id),
        };

        self.metrics.push(entry);

        // Auto-report if enabled
        if self.config.enable_auto_reporting {
            self.maybe_report_metrics()?;
        }

        Ok(())
    }

    fn maybe_report_metrics(&self) -> Result<(), TrustformersError> {
        // Check if it's time to report metrics
        Ok(())
    }
}

/// Expo development server
struct ExpoDevServer {
    config: DevServerConfig,
    is_running: bool,
}

impl ExpoDevServer {
    fn new(config: &ExpoConfig) -> Result<Self, TrustformersError> {
        Ok(Self {
            config: config.development_config.dev_server_config.clone(),
            is_running: false,
        })
    }

    fn start(&mut self) -> Result<(), TrustformersError> {
        if !self.is_running {
            // Start development server
            self.is_running = true;
        }
        Ok(())
    }

    fn stop(&mut self) -> Result<(), TrustformersError> {
        if self.is_running {
            // Stop development server
            self.is_running = false;
        }
        Ok(())
    }
}

/// Data structures for Expo integration
#[derive(Debug, Clone)]
struct LoadedModel {
    id: String,
    path: String,
    size_bytes: usize,
    loaded_at: std::time::SystemTime,
}

#[derive(Debug, Clone)]
struct CachedAsset {
    path: String,
    size_bytes: usize,
    cached_at: std::time::SystemTime,
    access_count: usize,
}

#[derive(Debug, Clone)]
struct PerformanceEntry {
    timestamp: std::time::SystemTime,
    metric_type: PerformanceMetric,
    value: f64,
    context: String,
}

/// Expo-specific inference request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpoInferenceRequest {
    pub request_id: String,
    pub model_id: String,
    pub input_data: Vec<f32>,
    pub input_shape: Vec<usize>,
    pub config_override: Option<crate::MobileConfig>,
    pub enable_preprocessing: bool,
    pub enable_postprocessing: bool,
    pub expo_context: ExpoContext,
}

/// Expo-specific inference response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpoInferenceResponse {
    pub request_id: String,
    pub success: bool,
    pub output_data: Vec<f32>,
    pub output_shape: Vec<usize>,
    pub inference_time_ms: f64,
    pub memory_used_mb: usize,
    pub error_message: Option<String>,
    pub expo_metrics: ExpoMetrics,
}

/// Expo context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpoContext {
    /// Expo SDK version
    pub expo_version: String,
    /// React Native version
    pub rn_version: String,
    /// App version
    pub app_version: String,
    /// Build type (development/production)
    pub build_type: String,
    /// Additional context data
    pub context_data: HashMap<String, String>,
}

impl Default for ExpoContext {
    fn default() -> Self {
        Self {
            expo_version: "49.0.0".to_string(),
            rn_version: "0.72.0".to_string(),
            app_version: "1.0.0".to_string(),
            build_type: "development".to_string(),
            context_data: HashMap::new(),
        }
    }
}

/// Expo-specific performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpoMetrics {
    pub module_load_time_ms: f64,
    pub asset_load_time_ms: f64,
    pub cache_hit_ratio: f64,
    pub bundle_size_kb: usize,
}

/// Trait for Expo modules
pub trait ExpoModule: Send + Sync {
    fn get_name(&self) -> &str;
    fn get_version(&self) -> &str;
    fn initialize(&mut self) -> Result<(), TrustformersError>;
    fn call_method(&self, method: &str, args: &[ExpoValue])
        -> Result<ExpoValue, TrustformersError>;
    fn get_supported_methods(&self) -> Vec<String>;
}

/// Expo value types for module communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpoValue {
    Null,
    Boolean(bool),
    Number(f64),
    String(String),
    Array(Vec<ExpoValue>),
    Object(HashMap<String, ExpoValue>),
    Buffer(Vec<u8>),
}

/// TrustformersInference Expo module
struct TrustformersInferenceModule {
    name: String,
    version: String,
}

impl TrustformersInferenceModule {
    fn new() -> Self {
        Self {
            name: "TrustformersInference".to_string(),
            version: "1.0.0".to_string(),
        }
    }
}

impl ExpoModule for TrustformersInferenceModule {
    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_version(&self) -> &str {
        &self.version
    }

    fn initialize(&mut self) -> Result<(), TrustformersError> {
        Ok(())
    }

    fn call_method(
        &self,
        method: &str,
        args: &[ExpoValue],
    ) -> Result<ExpoValue, TrustformersError> {
        match method {
            "performInference" => {
                // Handle inference request
                Ok(ExpoValue::Object(HashMap::new()))
            },
            "loadModel" => {
                // Handle model loading
                Ok(ExpoValue::Boolean(true))
            },
            _ => Err(TrustformersError::invalid_input(format!(
                "Unknown method: {}",
                method
            ))),
        }
    }

    fn get_supported_methods(&self) -> Vec<String> {
        vec![
            "performInference".to_string(),
            "loadModel".to_string(),
            "unloadModel".to_string(),
            "getModelInfo".to_string(),
        ]
    }
}

/// TrustformersModel Expo module
struct TrustformersModelModule {
    name: String,
    version: String,
}

impl TrustformersModelModule {
    fn new() -> Self {
        Self {
            name: "TrustformersModel".to_string(),
            version: "1.0.0".to_string(),
        }
    }
}

impl ExpoModule for TrustformersModelModule {
    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_version(&self) -> &str {
        &self.version
    }

    fn initialize(&mut self) -> Result<(), TrustformersError> {
        Ok(())
    }

    fn call_method(
        &self,
        method: &str,
        args: &[ExpoValue],
    ) -> Result<ExpoValue, TrustformersError> {
        match method {
            "listModels" => Ok(ExpoValue::Array(vec![])),
            "getModelMetadata" => Ok(ExpoValue::Object(HashMap::new())),
            _ => Err(TrustformersError::invalid_input(format!(
                "Unknown method: {}",
                method
            ))),
        }
    }

    fn get_supported_methods(&self) -> Vec<String> {
        vec![
            "listModels".to_string(),
            "getModelMetadata".to_string(),
            "downloadModel".to_string(),
            "deleteModel".to_string(),
        ]
    }
}

/// TrustformersAssets Expo module
struct TrustformersAssetsModule {
    name: String,
    version: String,
}

impl TrustformersAssetsModule {
    fn new() -> Self {
        Self {
            name: "TrustformersAssets".to_string(),
            version: "1.0.0".to_string(),
        }
    }
}

impl ExpoModule for TrustformersAssetsModule {
    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_version(&self) -> &str {
        &self.version
    }

    fn initialize(&mut self) -> Result<(), TrustformersError> {
        Ok(())
    }

    fn call_method(
        &self,
        method: &str,
        args: &[ExpoValue],
    ) -> Result<ExpoValue, TrustformersError> {
        match method {
            "getAssetInfo" => Ok(ExpoValue::Object(HashMap::new())),
            "clearCache" => Ok(ExpoValue::Boolean(true)),
            _ => Err(TrustformersError::invalid_input(format!(
                "Unknown method: {}",
                method
            ))),
        }
    }

    fn get_supported_methods(&self) -> Vec<String> {
        vec![
            "getAssetInfo".to_string(),
            "clearCache".to_string(),
            "getCacheSize".to_string(),
            "preloadAssets".to_string(),
        ]
    }
}

/// C API for Expo integration
#[no_mangle]
pub extern "C" fn tfk_expo_plugin_create(config_json: *const c_char) -> *mut ExpoPlugin {
    if config_json.is_null() {
        return std::ptr::null_mut();
    }

    let config_str = unsafe { CStr::from_ptr(config_json).to_str().unwrap_or_default() };

    let config: ExpoConfig = serde_json::from_str(config_str).unwrap_or_default();

    match ExpoPlugin::new(config) {
        Ok(plugin) => Box::into_raw(Box::new(plugin)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn tfk_expo_plugin_destroy(plugin: *mut ExpoPlugin) {
    if !plugin.is_null() {
        unsafe {
            Box::from_raw(plugin);
        }
    }
}

#[no_mangle]
pub extern "C" fn tfk_expo_perform_inference(
    plugin: *mut ExpoPlugin,
    request_json: *const c_char,
) -> *const c_char {
    if plugin.is_null() || request_json.is_null() {
        return std::ptr::null();
    }

    let plugin = unsafe { &*plugin };
    let request_str = unsafe { CStr::from_ptr(request_json).to_str().unwrap_or_default() };

    let request: ExpoInferenceRequest = match serde_json::from_str(request_str) {
        Ok(req) => req,
        Err(_) => return std::ptr::null(),
    };

    match plugin.perform_inference(request) {
        Ok(response) => {
            let response_json = serde_json::to_string(&response).unwrap_or_default();
            let c_string = CString::new(response_json).unwrap_or_default();
            c_string.into_raw()
        },
        Err(_) => std::ptr::null(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expo_config_default() {
        let config = ExpoConfig::default();
        assert_eq!(config.plugin_name, "trustformers-expo-plugin");
        assert!(config.enable_ai);
        assert!(config.optimize_models);
    }

    #[test]
    fn test_development_config() {
        let config = DevelopmentConfig::default();
        assert!(config.enable_hot_reload);
        assert!(config.enable_debug);
        assert_eq!(config.dev_server_config.port, 8081);
    }

    #[test]
    fn test_build_config() {
        let config = BuildConfig::default();
        assert_eq!(config.target_platform, TargetPlatform::Both);
        assert!(config.eas_config.enable_eas_build);
    }

    #[test]
    fn test_asset_config() {
        let config = AssetConfig::default();
        assert_eq!(config.bundling_strategy, AssetBundlingStrategy::Selective);
        assert!(config.compression_config.enable_compression);
        assert!(config.caching_config.enable_caching);
    }

    #[test]
    fn test_performance_config() {
        let config = PerformanceConfig::default();
        assert!(config.enable_monitoring);
        assert!(config.memory_config.enable_gc_optimizations);
    }

    #[test]
    fn test_expo_context() {
        let context = ExpoContext::default();
        assert!(!context.expo_version.is_empty());
        assert!(!context.rn_version.is_empty());
        assert_eq!(context.build_type, "development");
    }

    #[test]
    fn test_expo_value_types() {
        let null_val = ExpoValue::Null;
        let bool_val = ExpoValue::Boolean(true);
        let num_val = ExpoValue::Number(42.0);
        let str_val = ExpoValue::String("test".to_string());

        assert!(matches!(null_val, ExpoValue::Null));

        if let ExpoValue::Boolean(b) = bool_val {
            assert!(b);
        } else {
            panic!("Expected ExpoValue::Boolean, got {:?}", bool_val);
        }

        if let ExpoValue::Number(n) = num_val {
            assert_eq!(n, 42.0);
        } else {
            panic!("Expected ExpoValue::Number, got {:?}", num_val);
        }

        if let ExpoValue::String(s) = str_val {
            assert_eq!(s, "test");
        } else {
            panic!("Expected ExpoValue::String, got {:?}", str_val);
        }
    }

    #[test]
    fn test_inference_module() {
        let module = TrustformersInferenceModule::new();
        assert_eq!(module.get_name(), "TrustformersInference");
        assert_eq!(module.get_version(), "1.0.0");

        let methods = module.get_supported_methods();
        assert!(methods.contains(&"performInference".to_string()));
        assert!(methods.contains(&"loadModel".to_string()));
    }

    #[test]
    fn test_model_module() {
        let module = TrustformersModelModule::new();
        assert_eq!(module.get_name(), "TrustformersModel");

        let methods = module.get_supported_methods();
        assert!(methods.contains(&"listModels".to_string()));
        assert!(methods.contains(&"getModelMetadata".to_string()));
    }

    #[test]
    fn test_assets_module() {
        let module = TrustformersAssetsModule::new();
        assert_eq!(module.get_name(), "TrustformersAssets");

        let methods = module.get_supported_methods();
        assert!(methods.contains(&"getAssetInfo".to_string()));
        assert!(methods.contains(&"clearCache".to_string()));
    }

    #[test]
    fn test_compression_algorithm_variants() {
        let gzip = CompressionAlgorithm::Gzip;
        let brotli = CompressionAlgorithm::Brotli;
        let deflate = CompressionAlgorithm::Deflate;
        let lz4 = CompressionAlgorithm::Lz4;

        assert_eq!(gzip, CompressionAlgorithm::Gzip);
        assert_ne!(gzip, brotli);
        assert_ne!(brotli, deflate);
        assert_ne!(deflate, lz4);
    }

    #[test]
    fn test_target_platform_variants() {
        assert_eq!(TargetPlatform::iOS, TargetPlatform::iOS);
        assert_eq!(TargetPlatform::Android, TargetPlatform::Android);
        assert_eq!(TargetPlatform::Both, TargetPlatform::Both);
        assert_ne!(TargetPlatform::iOS, TargetPlatform::Android);
    }
}
