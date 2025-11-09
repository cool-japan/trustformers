//! iOS App Extension Support for TrustformeRS Mobile
//!
//! This module provides comprehensive iOS App Extension support, enabling TrustformeRS
//! to run efficiently within various iOS extension contexts including widgets,
//! notification service extensions, share extensions, and more.

use crate::{
    device_info::DeviceInfo, inference::MobileInferenceEngine, MemoryOptimization, MobileConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::Tensor;

/// iOS App Extension types supported by TrustformeRS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum iOSExtensionType {
    /// Today Extension (deprecated in iOS 14+)
    TodayExtension,
    /// Widget Extension (iOS 14+)
    WidgetExtension,
    /// Notification Service Extension
    NotificationServiceExtension,
    /// Notification Content Extension
    NotificationContentExtension,
    /// Share Extension
    ShareExtension,
    /// Action Extension
    ActionExtension,
    /// Keyboard Extension
    KeyboardExtension,
    /// Photo Editing Extension
    PhotoEditingExtension,
    /// Document Provider Extension
    DocumentProviderExtension,
    /// Custom Keyboard Extension
    CustomKeyboardExtension,
    /// Siri Intents Extension
    IntentsExtension,
    /// Intents UI Extension
    IntentsUIExtension,
    /// Spotlight Index Extension
    SpotlightIndexExtension,
}

/// App Extension configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct iOSExtensionConfig {
    /// Extension type
    pub extension_type: iOSExtensionType,
    /// Memory limit in MB (extensions have strict limits)
    pub memory_limit_mb: usize,
    /// Execution time limit in seconds
    pub execution_time_limit_seconds: f64,
    /// Enable background processing
    pub enable_background_processing: bool,
    /// Model cache configuration
    pub model_cache: ExtensionModelCacheConfig,
    /// Performance optimization settings
    pub performance: ExtensionPerformanceConfig,
    /// Privacy settings
    pub privacy: ExtensionPrivacyConfig,
    /// Resource management
    pub resource_management: ExtensionResourceConfig,
}

/// Model cache configuration for extensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionModelCacheConfig {
    /// Enable model caching between extension launches
    pub enable_persistent_cache: bool,
    /// Maximum cache size in MB
    pub max_cache_size_mb: usize,
    /// Cache expiration time in hours
    pub cache_expiration_hours: f64,
    /// Enable model compression in cache
    pub enable_compression: bool,
    /// Cache location strategy
    pub cache_location: ExtensionCacheLocation,
}

/// Extension cache location strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExtensionCacheLocation {
    /// App group shared container
    AppGroupContainer,
    /// Extension bundle
    ExtensionBundle,
    /// Temporary directory
    TemporaryDirectory,
    /// User defaults
    UserDefaults,
}

/// Performance configuration for extensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionPerformanceConfig {
    /// Optimize for memory (vs speed)
    pub optimize_for_memory: bool,
    /// Maximum inference time in milliseconds
    pub max_inference_time_ms: f64,
    /// Enable aggressive memory cleanup
    pub aggressive_memory_cleanup: bool,
    /// Use minimal model loading
    pub use_minimal_model_loading: bool,
    /// Batch size optimization
    pub batch_optimization: ExtensionBatchConfig,
}

/// Batch configuration for extensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionBatchConfig {
    /// Enable batching
    pub enable_batching: bool,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: f64,
    /// Dynamic batch sizing
    pub dynamic_batch_sizing: bool,
}

/// Privacy configuration for extensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionPrivacyConfig {
    /// Enable on-device processing only
    pub on_device_only: bool,
    /// Enable data anonymization
    pub enable_anonymization: bool,
    /// Disable telemetry
    pub disable_telemetry: bool,
    /// Secure memory handling
    pub secure_memory_handling: bool,
    /// Data retention policy
    pub data_retention: ExtensionDataRetentionConfig,
}

/// Data retention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionDataRetentionConfig {
    /// Retain inference results (hours)
    pub inference_results_retention_hours: f64,
    /// Retain model data (hours)
    pub model_data_retention_hours: f64,
    /// Retain cache data (hours)
    pub cache_data_retention_hours: f64,
    /// Auto-cleanup frequency (hours)
    pub cleanup_frequency_hours: f64,
}

/// Resource management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionResourceConfig {
    /// Maximum CPU usage percentage
    pub max_cpu_usage_percent: f64,
    /// Memory warning threshold
    pub memory_warning_threshold_mb: usize,
    /// Thermal throttling threshold
    pub thermal_throttling_threshold: f64,
    /// Battery usage optimization
    pub battery_optimization: ExtensionBatteryConfig,
}

/// Battery optimization for extensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionBatteryConfig {
    /// Enable battery-aware processing
    pub enable_battery_awareness: bool,
    /// Suspend on low battery
    pub suspend_on_low_battery: bool,
    /// Low battery threshold percentage
    pub low_battery_threshold_percent: f64,
    /// Reduce performance on battery
    pub reduce_performance_on_battery: bool,
}

/// Extension inference request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionInferenceRequest {
    /// Request ID
    pub request_id: String,
    /// Model ID
    pub model_id: String,
    /// Input data
    pub input_data: Vec<f32>,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Extension context
    pub extension_context: ExtensionContext,
    /// Priority level
    pub priority: ExtensionPriority,
    /// Timeout override
    pub timeout_override_ms: Option<f64>,
}

/// Extension execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionContext {
    /// Extension type
    pub extension_type: iOSExtensionType,
    /// Available memory in MB
    pub available_memory_mb: usize,
    /// Time remaining in seconds
    pub time_remaining_seconds: f64,
    /// Is in background
    pub is_background: bool,
    /// User interaction required
    pub user_interaction_required: bool,
    /// Additional context data
    pub context_data: HashMap<String, serde_json::Value>,
}

/// Extension priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtensionPriority {
    /// Critical priority (user-facing)
    Critical,
    /// High priority
    High,
    /// Normal priority
    Normal,
    /// Low priority (background)
    Low,
    /// Deferred (when resources available)
    Deferred,
}

/// Extension inference response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionInferenceResponse {
    /// Request ID
    pub request_id: String,
    /// Success flag
    pub success: bool,
    /// Output data
    pub output_data: Vec<f32>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Execution metrics
    pub metrics: ExtensionMetrics,
    /// Error information
    pub error: Option<ExtensionError>,
}

/// Extension execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionMetrics {
    /// Inference time in milliseconds
    pub inference_time_ms: f64,
    /// Memory used in MB
    pub memory_used_mb: usize,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Cache hit
    pub cache_hit: bool,
    /// Model load time (if applicable)
    pub model_load_time_ms: Option<f64>,
    /// Extension lifecycle time
    pub extension_lifecycle_ms: f64,
}

/// Extension error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Error category
    pub category: ExtensionErrorCategory,
    /// Recovery suggestions
    pub recovery_suggestions: Vec<String>,
}

/// Extension error categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExtensionErrorCategory {
    /// Memory limit exceeded
    MemoryLimitExceeded,
    /// Time limit exceeded
    TimeLimitExceeded,
    /// Model not available
    ModelNotAvailable,
    /// Invalid input
    InvalidInput,
    /// Extension terminated
    ExtensionTerminated,
    /// Resource unavailable
    ResourceUnavailable,
    /// Configuration error
    ConfigurationError,
}

/// iOS App Extension Manager
pub struct iOSAppExtensionManager {
    config: iOSExtensionConfig,
    inference_engine: Arc<Mutex<MobileInferenceEngine>>,
    model_cache: Arc<Mutex<ExtensionModelCache>>,
    resource_monitor: Arc<Mutex<ExtensionResourceMonitor>>,
    statistics: Arc<Mutex<ExtensionStatistics>>,
}

/// Extension model cache
#[derive(Debug)]
struct ExtensionModelCache {
    cached_models: HashMap<String, CachedModel>,
    cache_size_mb: usize,
    last_cleanup: std::time::Instant,
}

/// Cached model information
#[derive(Debug, Clone)]
struct CachedModel {
    model_id: String,
    model_data: Vec<u8>,
    size_mb: usize,
    last_accessed: std::time::Instant,
    access_count: usize,
}

/// Extension resource monitor
#[derive(Debug)]
struct ExtensionResourceMonitor {
    memory_usage_mb: usize,
    cpu_usage_percent: f64,
    last_memory_warning: Option<std::time::Instant>,
    thermal_state: f64,
    battery_level: f64,
    is_low_power_mode: bool,
}

/// Extension statistics
#[derive(Debug, Clone)]
struct ExtensionStatistics {
    total_requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    average_response_time_ms: f64,
    cache_hit_rate: f64,
    memory_warnings: usize,
    extension_terminations: usize,
}

impl iOSAppExtensionManager {
    /// Create new extension manager
    pub fn new(config: iOSExtensionConfig, mobile_config: MobileConfig) -> Result<Self> {
        config.validate()?;

        let inference_engine = Arc::new(Mutex::new(MobileInferenceEngine::new(mobile_config)?));
        let model_cache = Arc::new(Mutex::new(ExtensionModelCache::new(&config.model_cache)));
        let resource_monitor = Arc::new(Mutex::new(ExtensionResourceMonitor::new()));
        let statistics = Arc::new(Mutex::new(ExtensionStatistics::new()));

        Ok(Self {
            config,
            inference_engine,
            model_cache,
            resource_monitor,
            statistics,
        })
    }

    /// Initialize extension for specific context
    pub async fn initialize_extension(&self, context: &ExtensionContext) -> Result<String> {
        tracing::info!("Initializing extension: {:?}", context.extension_type);

        // Update resource monitor
        {
            let mut monitor = self.resource_monitor.lock().unwrap();
            monitor.update_context(context);
        }

        // Validate memory constraints
        if context.available_memory_mb < self.config.memory_limit_mb {
            return Err(TrustformersError::runtime_error(
                format!(
                    "Insufficient memory: {} MB < {} MB required",
                    context.available_memory_mb, self.config.memory_limit_mb
                )
                .into(),
            )
            .into());
        }

        // Pre-load common models if configured
        if self.config.model_cache.enable_persistent_cache {
            self.preload_models_for_extension(context.extension_type).await?;
        }

        let init_result = serde_json::json!({
            "extension_type": context.extension_type,
            "available_memory_mb": context.available_memory_mb,
            "time_remaining_seconds": context.time_remaining_seconds,
            "cache_enabled": self.config.model_cache.enable_persistent_cache,
            "performance_optimized": self.config.performance.optimize_for_memory
        });

        Ok(init_result.to_string())
    }

    /// Perform inference in extension context
    pub async fn extension_inference(
        &self,
        request: ExtensionInferenceRequest,
    ) -> Result<ExtensionInferenceResponse> {
        let start_time = std::time::Instant::now();

        // Check resource constraints
        self.check_resource_constraints(&request.extension_context)?;

        // Check priority and queue management
        if !self.should_process_request(&request) {
            return Ok(ExtensionInferenceResponse {
                request_id: request.request_id.clone(),
                success: false,
                output_data: Vec::new(),
                output_shape: Vec::new(),
                metrics: ExtensionMetrics {
                    inference_time_ms: 0.0,
                    memory_used_mb: 0,
                    cpu_usage_percent: 0.0,
                    cache_hit: false,
                    model_load_time_ms: None,
                    extension_lifecycle_ms: start_time.elapsed().as_millis() as f64,
                },
                error: Some(ExtensionError {
                    code: "REQUEST_REJECTED".to_string(),
                    message: "Request rejected due to resource constraints".to_string(),
                    category: ExtensionErrorCategory::ResourceUnavailable,
                    recovery_suggestions: vec!["Retry with lower priority".to_string()],
                }),
            });
        }

        // Check model cache first
        let model_load_start = std::time::Instant::now();
        let model_load_time_ms = if self.ensure_model_loaded(&request.model_id).await? {
            Some(model_load_start.elapsed().as_millis() as f64)
        } else {
            None
        };

        // Perform inference
        let inference_start = std::time::Instant::now();
        let input_tensor = Tensor::from_vec(request.input_data, &request.input_shape)?;

        let inference_result = {
            let mut engine = self.inference_engine.lock().unwrap();
            engine.inference(&request.model_id, &input_tensor)
        };

        let inference_time = inference_start.elapsed().as_millis() as f64;
        let total_time = start_time.elapsed().as_millis() as f64;

        // Update resource monitoring
        let current_memory = self.get_current_memory_usage();
        let cpu_usage = self.get_current_cpu_usage();

        let metrics = ExtensionMetrics {
            inference_time_ms: inference_time,
            memory_used_mb: current_memory,
            cpu_usage_percent: cpu_usage,
            cache_hit: model_load_time_ms.is_none(),
            model_load_time_ms,
            extension_lifecycle_ms: total_time,
        };

        match inference_result {
            Ok(output_tensor) => {
                let output_data = output_tensor.data_f32()?.to_vec();
                let output_shape = output_tensor.shape().to_vec();

                // Update statistics
                self.update_statistics(true, total_time, model_load_time_ms.is_none());

                Ok(ExtensionInferenceResponse {
                    request_id: request.request_id,
                    success: true,
                    output_data,
                    output_shape,
                    metrics,
                    error: None,
                })
            },
            Err(error) => {
                // Update statistics
                self.update_statistics(false, total_time, false);

                Ok(ExtensionInferenceResponse {
                    request_id: request.request_id,
                    success: false,
                    output_data: Vec::new(),
                    output_shape: Vec::new(),
                    metrics,
                    error: Some(ExtensionError {
                        code: "INFERENCE_ERROR".to_string(),
                        message: error.to_string(),
                        category: ExtensionErrorCategory::ResourceUnavailable,
                        recovery_suggestions: vec![
                            "Check model availability".to_string(),
                            "Verify input format".to_string(),
                        ],
                    }),
                })
            },
        }
    }

    /// Clean up extension resources
    pub async fn cleanup_extension(&self) -> Result<()> {
        tracing::info!("Cleaning up extension resources");

        // Aggressive cleanup if configured
        if self.config.performance.aggressive_memory_cleanup {
            // Clear model cache if memory pressure
            if self.is_under_memory_pressure() {
                let mut cache = self.model_cache.lock().unwrap();
                cache.clear_cache();
            }

            // Unload models from inference engine
            let mut engine = self.inference_engine.lock().unwrap();
            engine.cleanup_memory()?;
        }

        // Update cleanup statistics
        let mut cache = self.model_cache.lock().unwrap();
        cache.perform_cleanup(&self.config.model_cache);

        Ok(())
    }

    /// Get extension statistics
    pub fn get_extension_statistics(&self) -> Result<String> {
        let stats = self.statistics.lock().unwrap();

        let stats_json = serde_json::json!({
            "total_requests": stats.total_requests,
            "successful_requests": stats.successful_requests,
            "failed_requests": stats.failed_requests,
            "success_rate": if stats.total_requests > 0 {
                stats.successful_requests as f64 / stats.total_requests as f64
            } else { 0.0 },
            "average_response_time_ms": stats.average_response_time_ms,
            "cache_hit_rate": stats.cache_hit_rate,
            "memory_warnings": stats.memory_warnings,
            "extension_terminations": stats.extension_terminations
        });

        Ok(stats_json.to_string())
    }

    // Private helper methods

    async fn preload_models_for_extension(&self, extension_type: iOSExtensionType) -> Result<()> {
        // Define common models for different extension types
        let common_models = match extension_type {
            iOSExtensionType::WidgetExtension => {
                vec!["lightweight_summarizer", "sentiment_classifier"]
            },
            iOSExtensionType::NotificationServiceExtension => vec!["notification_classifier"],
            iOSExtensionType::ShareExtension => vec!["content_analyzer", "text_classifier"],
            iOSExtensionType::KeyboardExtension => vec!["text_predictor", "autocomplete"],
            _ => vec![],
        };

        for model_id in common_models {
            let _ = self.ensure_model_loaded(model_id).await;
        }

        Ok(())
    }

    fn check_resource_constraints(&self, context: &ExtensionContext) -> Result<()> {
        // Check memory constraints
        if context.available_memory_mb < self.config.memory_limit_mb / 2 {
            return Err(TrustformersError::runtime_error(
                "Insufficient memory for inference".into(),
            )
            .into());
        }

        // Check time constraints
        if context.time_remaining_seconds < self.config.execution_time_limit_seconds {
            return Err(
                TrustformersError::runtime_error("Insufficient time for inference".into()).into(),
            );
        }

        // Check thermal state
        let monitor = self.resource_monitor.lock().unwrap();
        if monitor.thermal_state > self.config.resource_management.thermal_throttling_threshold {
            return Err(
                TrustformersError::runtime_error("Thermal throttling active".into()).into(),
            );
        }

        Ok(())
    }

    fn should_process_request(&self, request: &ExtensionInferenceRequest) -> bool {
        let monitor = self.resource_monitor.lock().unwrap();

        // Always process critical requests
        if request.priority == ExtensionPriority::Critical {
            return true;
        }

        // Check battery constraints
        if self.config.resource_management.battery_optimization.enable_battery_awareness {
            if monitor.battery_level
                < self
                    .config
                    .resource_management
                    .battery_optimization
                    .low_battery_threshold_percent
            {
                return request.priority == ExtensionPriority::Critical
                    || request.priority == ExtensionPriority::High;
            }
        }

        // Check memory pressure
        if self.is_under_memory_pressure() {
            return request.priority != ExtensionPriority::Low
                && request.priority != ExtensionPriority::Deferred;
        }

        true
    }

    async fn ensure_model_loaded(&self, model_id: &str) -> Result<bool> {
        // Check if model is already loaded in inference engine
        {
            let engine = self.inference_engine.lock().unwrap();
            if engine.is_model_loaded(model_id) {
                // Update cache access
                let mut cache = self.model_cache.lock().unwrap();
                cache.update_access(model_id);
                return Ok(false); // Model was already loaded
            }
        }

        // Try to load from cache
        {
            let mut cache = self.model_cache.lock().unwrap();
            if let Some(cached_model) = cache.get_model(model_id) {
                // Load model from cache into inference engine
                let mut engine = self.inference_engine.lock().unwrap();
                engine.load_model_from_data(model_id, &cached_model.model_data)?;
                cache.update_access(model_id);
                return Ok(true); // Model was loaded from cache
            }
        }

        // Model not in cache - this would typically load from app bundle or download
        // For now, we'll return an error as extensions should use pre-cached models
        Err(TrustformersError::runtime_error(
            format!("Model not available in cache: {}", model_id).into(),
        ))
    }

    fn get_current_memory_usage(&self) -> usize {
        // Platform-specific memory usage detection would go here
        // For now, return a placeholder value
        let monitor = self.resource_monitor.lock().unwrap();
        monitor.memory_usage_mb
    }

    fn get_current_cpu_usage(&self) -> f64 {
        let monitor = self.resource_monitor.lock().unwrap();
        monitor.cpu_usage_percent
    }

    fn is_under_memory_pressure(&self) -> bool {
        let monitor = self.resource_monitor.lock().unwrap();
        monitor.memory_usage_mb > self.config.resource_management.memory_warning_threshold_mb
    }

    fn update_statistics(&self, success: bool, response_time_ms: f64, cache_hit: bool) {
        let mut stats = self.statistics.lock().unwrap();

        stats.total_requests += 1;
        if success {
            stats.successful_requests += 1;
        } else {
            stats.failed_requests += 1;
        }

        // Update running averages
        let alpha = 0.1;
        if stats.total_requests == 1 {
            stats.average_response_time_ms = response_time_ms;
            stats.cache_hit_rate = if cache_hit { 1.0 } else { 0.0 };
        } else {
            stats.average_response_time_ms =
                alpha * response_time_ms + (1.0 - alpha) * stats.average_response_time_ms;

            let cache_rate = if cache_hit { 1.0 } else { 0.0 };
            stats.cache_hit_rate = alpha * cache_rate + (1.0 - alpha) * stats.cache_hit_rate;
        }
    }
}

// Implementation details for helper structs

impl ExtensionModelCache {
    fn new(config: &ExtensionModelCacheConfig) -> Self {
        Self {
            cached_models: HashMap::new(),
            cache_size_mb: 0,
            last_cleanup: std::time::Instant::now(),
        }
    }

    fn get_model(&mut self, model_id: &str) -> Option<&CachedModel> {
        self.cached_models.get(model_id)
    }

    fn update_access(&mut self, model_id: &str) {
        if let Some(model) = self.cached_models.get_mut(model_id) {
            model.last_accessed = std::time::Instant::now();
            model.access_count += 1;
        }
    }

    fn clear_cache(&mut self) {
        self.cached_models.clear();
        self.cache_size_mb = 0;
    }

    fn perform_cleanup(&mut self, config: &ExtensionModelCacheConfig) {
        let now = std::time::Instant::now();

        // Check if cleanup is needed
        if now.duration_since(self.last_cleanup).as_secs_f64()
            < config.cache_expiration_hours * 3600.0
        {
            return;
        }

        // Remove expired models
        let expiration_duration =
            std::time::Duration::from_secs_f64(config.cache_expiration_hours * 3600.0);
        self.cached_models
            .retain(|_, model| now.duration_since(model.last_accessed) < expiration_duration);

        // Recalculate cache size
        self.cache_size_mb = self.cached_models.values().map(|m| m.size_mb).sum();
        self.last_cleanup = now;
    }
}

impl ExtensionResourceMonitor {
    fn new() -> Self {
        Self {
            memory_usage_mb: 0,
            cpu_usage_percent: 0.0,
            last_memory_warning: None,
            thermal_state: 0.0,
            battery_level: 100.0,
            is_low_power_mode: false,
        }
    }

    fn update_context(&mut self, context: &ExtensionContext) {
        self.memory_usage_mb = context.available_memory_mb;
        // Other system metrics would be updated from iOS APIs
    }
}

impl ExtensionStatistics {
    fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            average_response_time_ms: 0.0,
            cache_hit_rate: 0.0,
            memory_warnings: 0,
            extension_terminations: 0,
        }
    }
}

impl Default for iOSExtensionConfig {
    fn default() -> Self {
        Self {
            extension_type: iOSExtensionType::WidgetExtension,
            memory_limit_mb: 30, // Conservative limit for extensions
            execution_time_limit_seconds: 2.0,
            enable_background_processing: false,
            model_cache: ExtensionModelCacheConfig {
                enable_persistent_cache: true,
                max_cache_size_mb: 20,
                cache_expiration_hours: 24.0,
                enable_compression: true,
                cache_location: ExtensionCacheLocation::AppGroupContainer,
            },
            performance: ExtensionPerformanceConfig {
                optimize_for_memory: true,
                max_inference_time_ms: 500.0,
                aggressive_memory_cleanup: true,
                use_minimal_model_loading: true,
                batch_optimization: ExtensionBatchConfig {
                    enable_batching: false,
                    max_batch_size: 1,
                    batch_timeout_ms: 100.0,
                    dynamic_batch_sizing: false,
                },
            },
            privacy: ExtensionPrivacyConfig {
                on_device_only: true,
                enable_anonymization: true,
                disable_telemetry: true,
                secure_memory_handling: true,
                data_retention: ExtensionDataRetentionConfig {
                    inference_results_retention_hours: 1.0,
                    model_data_retention_hours: 24.0,
                    cache_data_retention_hours: 24.0,
                    cleanup_frequency_hours: 4.0,
                },
            },
            resource_management: ExtensionResourceConfig {
                max_cpu_usage_percent: 50.0,
                memory_warning_threshold_mb: 25,
                thermal_throttling_threshold: 0.8,
                battery_optimization: ExtensionBatteryConfig {
                    enable_battery_awareness: true,
                    suspend_on_low_battery: true,
                    low_battery_threshold_percent: 20.0,
                    reduce_performance_on_battery: true,
                },
            },
        }
    }
}

impl iOSExtensionConfig {
    /// Validate extension configuration
    pub fn validate(&self) -> Result<()> {
        // Check memory constraints
        if self.memory_limit_mb < 10 {
            return Err(TrustformersError::config_error {
                message: "Memory limit too low for extensions".to_string(),
                context: trustformers_core::error::ErrorContext::new(
                    trustformers_core::error::ErrorCode::E4001,
                    "validate".to_string(),
                ),
            });
        }

        if self.memory_limit_mb > 100 {
            return Err(TrustformersError::config_error {
                message: "Memory limit too high for extensions".to_string(),
                context: trustformers_core::error::ErrorContext::new(
                    trustformers_core::error::ErrorCode::E4001,
                    "validate".to_string(),
                ),
            });
        }

        // Check execution time
        if self.execution_time_limit_seconds < 0.1 {
            return Err(TrustformersError::config_error {
                message: "Execution time limit too low".to_string(),
                context: trustformers_core::error::ErrorContext::new(
                    trustformers_core::error::ErrorCode::E4001,
                    "validate".to_string(),
                ),
            });
        }

        if self.execution_time_limit_seconds > 30.0 {
            return Err(TrustformersError::config_error {
                message: "Execution time limit too high for extensions".to_string(),
                context: trustformers_core::error::ErrorContext::new(
                    trustformers_core::error::ErrorCode::E4001,
                    "validate".to_string(),
                ),
            });
        }

        // Validate cache configuration
        if self.model_cache.max_cache_size_mb > self.memory_limit_mb {
            return Err(TrustformersError::config_error {
                message: "Cache size exceeds memory limit".to_string(),
                context: trustformers_core::error::ErrorContext::new(
                    trustformers_core::error::ErrorCode::E4001,
                    "validate".to_string(),
                ),
            });
        }

        Ok(())
    }

    /// Create configuration optimized for specific extension type
    pub fn for_extension_type(extension_type: iOSExtensionType) -> Self {
        let mut config = Self::default();
        config.extension_type = extension_type;

        match extension_type {
            iOSExtensionType::WidgetExtension => {
                config.memory_limit_mb = 30;
                config.execution_time_limit_seconds = 1.0;
                config.performance.max_inference_time_ms = 300.0;
            },
            iOSExtensionType::NotificationServiceExtension => {
                config.memory_limit_mb = 50;
                config.execution_time_limit_seconds = 30.0;
                config.performance.max_inference_time_ms = 1000.0;
            },
            iOSExtensionType::ShareExtension => {
                config.memory_limit_mb = 120;
                config.execution_time_limit_seconds = 10.0;
                config.performance.max_inference_time_ms = 2000.0;
            },
            iOSExtensionType::KeyboardExtension => {
                config.memory_limit_mb = 48;
                config.execution_time_limit_seconds = 0.5;
                config.performance.max_inference_time_ms = 100.0;
            },
            _ => {
                // Use defaults for other types
            },
        }

        config
    }
}

// Mock implementation for MobileInferenceEngine extension methods
impl MobileInferenceEngine {
    fn load_model_from_data(&mut self, _model_id: &str, _model_data: &[u8]) -> Result<()> {
        // Load model from byte data
        Ok(())
    }

    fn cleanup_memory(&mut self) -> Result<()> {
        // Perform aggressive memory cleanup
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_config_default() {
        let config = iOSExtensionConfig::default();
        assert_eq!(config.extension_type, iOSExtensionType::WidgetExtension);
        assert!(config.memory_limit_mb <= 50);
        assert!(config.privacy.on_device_only);
    }

    #[test]
    fn test_extension_config_validation() {
        let mut config = iOSExtensionConfig::default();
        assert!(config.validate().is_ok());

        config.memory_limit_mb = 5;
        assert!(config.validate().is_err());

        config.memory_limit_mb = 30;
        config.execution_time_limit_seconds = 0.05;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_extension_type_specific_configs() {
        let widget_config =
            iOSExtensionConfig::for_extension_type(iOSExtensionType::WidgetExtension);
        assert_eq!(widget_config.memory_limit_mb, 30);
        assert_eq!(widget_config.execution_time_limit_seconds, 1.0);

        let notification_config =
            iOSExtensionConfig::for_extension_type(iOSExtensionType::NotificationServiceExtension);
        assert_eq!(notification_config.memory_limit_mb, 50);
        assert_eq!(notification_config.execution_time_limit_seconds, 30.0);
    }

    #[tokio::test]
    async fn test_extension_manager_creation() {
        let ext_config = iOSExtensionConfig::default();
        let mobile_config = MobileConfig::ios_optimized();

        let result = iOSAppExtensionManager::new(ext_config, mobile_config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_extension_priorities() {
        assert_eq!(ExtensionPriority::Critical as u8, 0);
        assert!(ExtensionPriority::Critical > ExtensionPriority::Low);
    }
}
