//! React Native Native Module for TrustformeRS Mobile
//!
//! This module provides React Native bindings for TrustformeRS mobile functionality,
//! enabling JavaScript/TypeScript applications to use TrustformeRS models with
//! optimal performance through native execution.

use crate::{
    inference::MobileInferenceEngine,
    mobile_testing::DeviceInfo,
    model_management::{ModelManager, ModelManagerConfig},
    MobileConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::Tensor;
use trustformers_core::TrustformersError;

/// React Native module configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactNativeConfig {
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
    /// Enable debug logging
    pub enable_debug_logging: bool,
    /// Maximum concurrent inferences
    pub max_concurrent_inferences: usize,
    /// JavaScript bridge optimization
    pub optimize_js_bridge: bool,
    /// Use background thread for inference
    pub use_background_thread: bool,
    /// Cache inference results
    pub enable_result_caching: bool,
    /// Maximum cache size (MB)
    pub max_cache_size_mb: usize,
}

/// React Native inference request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Request ID for tracking
    pub request_id: String,
    /// Model ID to use
    pub model_id: String,
    /// Input data (serialized tensor)
    pub input_data: Vec<f32>,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Configuration overrides
    pub config_override: Option<MobileConfig>,
    /// Enable preprocessing
    pub enable_preprocessing: bool,
    /// Enable postprocessing
    pub enable_postprocessing: bool,
}

/// React Native inference response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Request ID
    pub request_id: String,
    /// Success flag
    pub success: bool,
    /// Output data (serialized tensor)
    pub output_data: Vec<f32>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Inference time in milliseconds
    pub inference_time_ms: f64,
    /// Memory used in MB
    pub memory_used_mb: usize,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
}

/// Performance metrics for React Native
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Preprocessing time (ms)
    pub preprocessing_time_ms: f64,
    /// Inference time (ms)
    pub inference_time_ms: f64,
    /// Postprocessing time (ms)
    pub postprocessing_time_ms: f64,
    /// Memory allocation (MB)
    pub memory_allocation_mb: usize,
    /// Cache hit ratio
    pub cache_hit_ratio: f32,
}

/// Model information for React Native
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model ID
    pub model_id: String,
    /// Model type
    pub model_type: String,
    /// Model version
    pub version: String,
    /// Model size in bytes
    pub size_bytes: usize,
    /// Whether model is loaded
    pub is_loaded: bool,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Supported features
    pub supported_features: Vec<String>,
}

/// Device capabilities for React Native
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Platform (iOS/Android)
    pub platform: String,
    /// Device model
    pub device_model: String,
    /// Available memory (MB)
    pub available_memory_mb: usize,
    /// CPU cores
    pub cpu_cores: usize,
    /// Has GPU acceleration
    pub has_gpu_acceleration: bool,
    /// Has neural processing unit
    pub has_npu: bool,
    /// Supported optimizations
    pub supported_optimizations: Vec<String>,
}

/// React Native TrustformeRS module
pub struct TrustformersReactNative {
    config: ReactNativeConfig,
    inference_engine: Arc<Mutex<MobileInferenceEngine>>,
    model_manager: Arc<Mutex<ModelManager>>,
    request_cache: Arc<Mutex<HashMap<String, InferenceResponse>>>,
    performance_stats: Arc<Mutex<PerformanceStats>>,
    device_capabilities: DeviceCapabilities,
}

/// Performance statistics tracking
#[derive(Debug, Clone)]
struct PerformanceStats {
    total_requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    average_inference_time_ms: f64,
    cache_hits: usize,
    cache_misses: usize,
}

impl TrustformersReactNative {
    /// Create new React Native module
    pub fn new(config: ReactNativeConfig, mobile_config: MobileConfig) -> Result<Self> {
        config.validate()?;

        let inference_engine = Arc::new(Mutex::new(MobileInferenceEngine::new(mobile_config)?));

        let model_manager_config = ModelManagerConfig::default();
        let model_manager = Arc::new(Mutex::new(ModelManager::new(model_manager_config)?));

        let request_cache = Arc::new(Mutex::new(HashMap::new()));
        let performance_stats = Arc::new(Mutex::new(PerformanceStats::new()));

        let device_capabilities = Self::detect_device_capabilities()?;

        Ok(Self {
            config,
            inference_engine,
            model_manager,
            request_cache,
            performance_stats,
            device_capabilities,
        })
    }

    /// Initialize the React Native module
    pub fn initialize(&self) -> Result<String> {
        tracing::info!("Initializing TrustformeRS React Native module");

        // Initialize inference engine
        let mut engine = self.inference_engine.lock().expect("Failed to acquire lock");
        engine.initialize()?;

        // Initialize model manager
        let model_manager = self.model_manager.lock().expect("Failed to acquire lock");
        tracing::info!(
            "Model manager initialized with {} models",
            model_manager.list_models().len()
        );

        let init_info = serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "platform": self.device_capabilities.platform,
            "device_model": self.device_capabilities.device_model,
            "available_memory_mb": self.device_capabilities.available_memory_mb,
            "supported_optimizations": self.device_capabilities.supported_optimizations
        });

        Ok(init_info.to_string())
    }

    /// Load a model for inference
    pub async fn load_model(&self, model_id: &str, model_path: &str) -> Result<String> {
        tracing::info!("Loading model: {} from path: {}", model_id, model_path);

        let mut engine = self.inference_engine.lock().expect("Failed to acquire lock");
        engine.load_model_from_path(model_id, model_path)?;

        let model_info = self.get_model_info(model_id)?;
        Ok(serde_json::to_string(&model_info)?)
    }

    /// Perform inference with caching and performance tracking
    pub async fn inference(&self, request_json: &str) -> Result<String> {
        let request: InferenceRequest = serde_json::from_str(request_json)?;

        // Check cache first
        if self.config.enable_result_caching {
            if let Some(cached_response) = self.check_cache(&request) {
                self.update_cache_stats(true);
                return Ok(serde_json::to_string(&cached_response)?);
            }
        }

        self.update_cache_stats(false);

        // Perform inference
        let response = if self.config.use_background_thread {
            self.inference_background(request).await?
        } else {
            self.inference_sync(request)?
        };

        // Cache result if enabled
        if self.config.enable_result_caching && response.success {
            self.cache_response(&response);
        }

        // Update performance statistics
        self.update_performance_stats(&response);

        Ok(serde_json::to_string(&response)?)
    }

    /// Perform batch inference
    pub async fn batch_inference(&self, requests_json: &str) -> Result<String> {
        let requests: Vec<InferenceRequest> = serde_json::from_str(requests_json)?;

        if requests.len() > self.config.max_concurrent_inferences {
            return Err(TrustformersError::runtime_error(format!(
                "Too many concurrent requests: {} > {}",
                requests.len(),
                self.config.max_concurrent_inferences
            ))
            .into());
        }

        let mut responses = Vec::new();

        // Process requests in parallel if background threading is enabled
        if self.config.use_background_thread {
            let futures: Vec<_> =
                requests.into_iter().map(|req| self.inference_background(req)).collect();

            for future in futures {
                responses.push(future.await?);
            }
        } else {
            // Process sequentially
            for request in requests {
                responses.push(self.inference_sync(request)?);
            }
        }

        Ok(serde_json::to_string(&responses)?)
    }

    /// Get available models
    pub fn get_available_models(&self) -> Result<String> {
        let model_manager = self.model_manager.lock().expect("Failed to acquire lock");
        let models = model_manager.list_models();

        let model_infos: Vec<ModelInfo> = models
            .iter()
            .map(|metadata| {
                ModelInfo {
                    model_id: metadata.model_id.clone(),
                    model_type: metadata.model_type.clone(),
                    version: metadata.version.clone(),
                    size_bytes: metadata.size_bytes,
                    is_loaded: self.is_model_loaded(&metadata.model_id),
                    input_shape: vec![1, 224, 224, 3], // Placeholder
                    output_shape: vec![1, 1000],       // Placeholder
                    supported_features: vec!["inference".to_string()],
                }
            })
            .collect();

        Ok(serde_json::to_string(&model_infos)?)
    }

    /// Download model from server
    pub async fn download_model(&self, model_id: &str) -> Result<String> {
        tracing::info!("Downloading model: {}", model_id);

        let mut model_manager = self.model_manager.lock().expect("Failed to acquire lock");

        // Create progress callback for React Native
        let progress_callback =
            Box::new(move |progress: crate::model_management::DownloadProgress| {
                // This would emit progress events to React Native
                tracing::debug!(
                    "Download progress: {:.1}%",
                    (progress.downloaded_bytes as f64 / progress.total_bytes as f64) * 100.0
                );
            });

        model_manager.download_model(model_id, Some(progress_callback)).await?;

        let download_result = serde_json::json!({
            "model_id": model_id,
            "status": "completed",
            "message": "Model downloaded successfully"
        });

        Ok(download_result.to_string())
    }

    /// Remove model from device
    pub fn remove_model(&self, model_id: &str) -> Result<String> {
        tracing::info!("Removing model: {}", model_id);

        // Unload from inference engine if loaded
        {
            let mut engine = self.inference_engine.lock().expect("Failed to acquire lock");
            let _ = engine.unload_model(model_id);
        }

        // Remove from model manager
        {
            let mut model_manager = self.model_manager.lock().expect("Failed to acquire lock");
            model_manager.remove_model(model_id)?;
        }

        let removal_result = serde_json::json!({
            "model_id": model_id,
            "status": "removed",
            "message": "Model removed successfully"
        });

        Ok(removal_result.to_string())
    }

    /// Get device capabilities
    pub fn get_device_capabilities(&self) -> Result<String> {
        Ok(serde_json::to_string(&self.device_capabilities)?)
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> Result<String> {
        let stats = self.performance_stats.lock().expect("Failed to acquire lock");

        let stats_json = serde_json::json!({
            "total_requests": stats.total_requests,
            "successful_requests": stats.successful_requests,
            "failed_requests": stats.failed_requests,
            "success_rate": if stats.total_requests > 0 {
                stats.successful_requests as f64 / stats.total_requests as f64
            } else { 0.0 },
            "average_inference_time_ms": stats.average_inference_time_ms,
            "cache_hit_rate": if stats.cache_hits + stats.cache_misses > 0 {
                stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64
            } else { 0.0 }
        });

        Ok(stats_json.to_string())
    }

    /// Clear cache
    pub fn clear_cache(&self) -> Result<String> {
        let mut cache = self.request_cache.lock().expect("Failed to acquire lock");
        let cache_size = cache.len();
        cache.clear();

        let result = serde_json::json!({
            "cleared_entries": cache_size,
            "message": "Cache cleared successfully"
        });

        Ok(result.to_string())
    }

    /// Configure model settings
    pub fn configure_model(&self, model_id: &str, config_json: &str) -> Result<String> {
        let config: MobileConfig = serde_json::from_str(config_json)?;

        let mut engine = self.inference_engine.lock().expect("Failed to acquire lock");
        engine.configure_model(model_id, config)?;

        let result = serde_json::json!({
            "model_id": model_id,
            "status": "configured",
            "message": "Model configuration updated"
        });

        Ok(result.to_string())
    }

    /// Enable/disable performance monitoring
    pub fn set_performance_monitoring(&mut self, enabled: bool) -> Result<String> {
        self.config.enable_performance_monitoring = enabled;

        let result = serde_json::json!({
            "performance_monitoring": enabled,
            "message": if enabled { "Performance monitoring enabled" } else { "Performance monitoring disabled" }
        });

        Ok(result.to_string())
    }

    // Private helper methods

    async fn inference_background(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        // Run inference on background thread
        let engine = self.inference_engine.clone();
        let config = self.config.clone();

        tokio::task::spawn_blocking(move || {
            Self::perform_inference_internal(engine, request, config)
        })
        .await
        .map_err(|e| CoreError::from(TrustformersError::runtime_error(e.to_string())))?
    }

    fn inference_sync(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        Self::perform_inference_internal(
            self.inference_engine.clone(),
            request,
            self.config.clone(),
        )
    }

    fn perform_inference_internal(
        engine: Arc<Mutex<MobileInferenceEngine>>,
        request: InferenceRequest,
        config: ReactNativeConfig,
    ) -> Result<InferenceResponse> {
        let start_time = std::time::Instant::now();

        let mut metrics = PerformanceMetrics {
            preprocessing_time_ms: 0.0,
            inference_time_ms: 0.0,
            postprocessing_time_ms: 0.0,
            memory_allocation_mb: 0,
            cache_hit_ratio: 0.0,
        };

        // Preprocessing
        let preprocess_start = std::time::Instant::now();
        let input_tensor = Tensor::from_vec(request.input_data, &request.input_shape)?;
        metrics.preprocessing_time_ms = preprocess_start.elapsed().as_millis() as f64;

        // Inference
        let inference_start = std::time::Instant::now();
        let result = {
            let mut engine_lock = engine.lock().expect("Failed to acquire lock");
            engine_lock.run_inference(&request.model_id, &input_tensor)
        };
        metrics.inference_time_ms = inference_start.elapsed().as_millis() as f64;

        match result {
            Ok(output_tensor) => {
                // Postprocessing
                let postprocess_start = std::time::Instant::now();
                let output_data = output_tensor.data_f32()?;
                let output_shape = output_tensor.shape().to_vec();
                metrics.postprocessing_time_ms = postprocess_start.elapsed().as_millis() as f64;

                let total_time = start_time.elapsed().as_millis() as f64;

                Ok(InferenceResponse {
                    request_id: request.request_id,
                    success: true,
                    output_data: output_data.to_vec(),
                    output_shape,
                    inference_time_ms: total_time,
                    memory_used_mb: 50, // Placeholder
                    error_message: None,
                    metrics,
                })
            },
            Err(error) => {
                let total_time = start_time.elapsed().as_millis() as f64;

                Ok(InferenceResponse {
                    request_id: request.request_id,
                    success: false,
                    output_data: Vec::new(),
                    output_shape: Vec::new(),
                    inference_time_ms: total_time,
                    memory_used_mb: 0,
                    error_message: Some(error.to_string()),
                    metrics,
                })
            },
        }
    }

    fn check_cache(&self, request: &InferenceRequest) -> Option<InferenceResponse> {
        let cache = self.request_cache.lock().expect("Failed to acquire lock");

        // Simple cache key based on model_id and input hash
        let cache_key = format!(
            "{}_{}_{:?}",
            request.model_id,
            request.input_shape.len(),
            request.input_data.len()
        );

        cache.get(&cache_key).cloned()
    }

    fn cache_response(&self, response: &InferenceResponse) {
        if !self.config.enable_result_caching {
            return;
        }

        let mut cache = self.request_cache.lock().expect("Failed to acquire lock");

        // Simple cache eviction if size limit exceeded
        if cache.len() >= self.config.max_cache_size_mb * 100 {
            // Rough estimation
            cache.clear();
        }

        let cache_key = format!("{}_response", response.request_id);
        cache.insert(cache_key, response.clone());
    }

    fn update_cache_stats(&self, cache_hit: bool) {
        let mut stats = self.performance_stats.lock().expect("Failed to acquire lock");
        if cache_hit {
            stats.cache_hits += 1;
        } else {
            stats.cache_misses += 1;
        }
    }

    fn update_performance_stats(&self, response: &InferenceResponse) {
        let mut stats = self.performance_stats.lock().expect("Failed to acquire lock");

        stats.total_requests += 1;
        if response.success {
            stats.successful_requests += 1;
        } else {
            stats.failed_requests += 1;
        }

        // Update running average
        let alpha = 0.1;
        if stats.total_requests == 1 {
            stats.average_inference_time_ms = response.inference_time_ms;
        } else {
            stats.average_inference_time_ms = alpha * response.inference_time_ms
                + (1.0 - alpha) * stats.average_inference_time_ms;
        }
    }

    fn get_model_info(&self, model_id: &str) -> Result<ModelInfo> {
        let model_manager = self.model_manager.lock().expect("Failed to acquire lock");

        if let Some(metadata) = model_manager.get_model(model_id) {
            Ok(ModelInfo {
                model_id: metadata.model_id.clone(),
                model_type: metadata.model_type.clone(),
                version: metadata.version.clone(),
                size_bytes: metadata.size_bytes,
                is_loaded: self.is_model_loaded(model_id),
                input_shape: vec![1, 224, 224, 3], // Would get from actual model
                output_shape: vec![1, 1000],       // Would get from actual model
                supported_features: vec!["inference".to_string()],
            })
        } else {
            Err(TrustformersError::runtime_error(format!("Model not found: {}", model_id)).into())
        }
    }

    fn is_model_loaded(&self, model_id: &str) -> bool {
        let engine = self.inference_engine.lock().expect("Failed to acquire lock");
        engine.is_model_loaded(model_id)
    }

    fn detect_device_capabilities() -> Result<DeviceCapabilities> {
        let device_info = DeviceInfo::detect_current_device()?;

        Ok(DeviceCapabilities {
            platform: if cfg!(target_os = "ios") {
                "iOS".to_string()
            } else if cfg!(target_os = "android") {
                "Android".to_string()
            } else {
                "Unknown".to_string()
            },
            device_model: device_info.hardware_model,
            available_memory_mb: device_info.ram_mb,
            cpu_cores: num_cpus::get(),
            has_gpu_acceleration: cfg!(any(target_os = "ios", target_os = "android")),
            has_npu: cfg!(target_os = "ios"), // Neural Engine is iOS-specific
            supported_optimizations: vec![
                "quantization".to_string(),
                "pruning".to_string(),
                "batching".to_string(),
            ],
        })
    }
}

impl PerformanceStats {
    fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            average_inference_time_ms: 0.0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }
}

impl Default for ReactNativeConfig {
    fn default() -> Self {
        Self {
            enable_performance_monitoring: true,
            enable_debug_logging: false,
            max_concurrent_inferences: 4,
            optimize_js_bridge: true,
            use_background_thread: true,
            enable_result_caching: true,
            max_cache_size_mb: 50,
        }
    }
}

impl ReactNativeConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.max_concurrent_inferences == 0 {
            return Err(TrustformersError::config_error(
                "Max concurrent inferences must be > 0",
                "validate",
            )
            .into());
        }

        if self.max_concurrent_inferences > 10 {
            return Err(TrustformersError::config_error(
                "Too many concurrent inferences",
                "validate",
            )
            .into());
        }

        if self.max_cache_size_mb == 0 {
            return Err(
                TrustformersError::config_error("Cache size must be > 0", "validate").into(),
            );
        }

        Ok(())
    }

    /// Create performance-optimized configuration
    pub fn performance_optimized() -> Self {
        Self {
            enable_performance_monitoring: true,
            enable_debug_logging: false,
            max_concurrent_inferences: 8,
            optimize_js_bridge: true,
            use_background_thread: true,
            enable_result_caching: true,
            max_cache_size_mb: 100,
        }
    }

    /// Create memory-optimized configuration
    pub fn memory_optimized() -> Self {
        Self {
            enable_performance_monitoring: false,
            enable_debug_logging: false,
            max_concurrent_inferences: 2,
            optimize_js_bridge: true,
            use_background_thread: false,
            enable_result_caching: false,
            max_cache_size_mb: 10,
        }
    }
}

// Mock implementation of MobileInferenceEngine methods for React Native
impl MobileInferenceEngine {
    fn initialize(&mut self) -> Result<()> {
        // Initialize inference engine
        Ok(())
    }

    fn load_model_from_path(&mut self, _model_id: &str, _model_path: &str) -> Result<()> {
        // Load model implementation
        Ok(())
    }

    fn unload_model(&mut self, _model_id: &str) -> Result<()> {
        // Unload model implementation
        Ok(())
    }

    fn run_inference(&mut self, _model_id: &str, input: &Tensor) -> Result<Tensor> {
        // Placeholder inference - return input tensor as output
        Ok(input.clone())
    }

    fn is_model_loaded(&self, _model_id: &str) -> bool {
        // Check if model is loaded
        true // Placeholder
    }

    fn configure_model(&mut self, _model_id: &str, _config: MobileConfig) -> Result<()> {
        // Configure model with new settings
        Ok(())
    }
}

/// Export functions for React Native bridge
pub mod react_native_exports {
    use super::*;
    use std::ffi::{CStr, CString};
    use std::os::raw::c_char;

    static mut TRUSTFORMERS_RN: Option<TrustformersReactNative> = None;

    /// Initialize TrustformeRS React Native module
    #[no_mangle]
    pub extern "C" fn trustformers_rn_initialize(config_json: *const c_char) -> *mut c_char {
        unsafe {
            let config_str = CStr::from_ptr(config_json).to_str().unwrap_or("{}");

            let rn_config: ReactNativeConfig = serde_json::from_str(config_str).unwrap_or_default();
            let mobile_config = MobileConfig::default();

            match TrustformersReactNative::new(rn_config, mobile_config) {
                Ok(module) => {
                    let init_result = module.initialize().unwrap_or_else(|e| e.to_string());
                    TRUSTFORMERS_RN = Some(module);
                    CString::new(init_result)
                        .unwrap_or_else(|_| {
                            CString::new("initialization complete")
                                .expect("Failed to create CString")
                        })
                        .into_raw()
                },
                Err(e) => {
                    let error = serde_json::json!({"error": e.to_string()});
                    CString::new(error.to_string())
                        .unwrap_or_else(|_| {
                            CString::new("error").expect("Failed to create CString")
                        })
                        .into_raw()
                },
            }
        }
    }

    /// Perform inference
    #[no_mangle]
    pub extern "C" fn trustformers_rn_inference(request_json: *const c_char) -> *mut c_char {
        unsafe {
            if let Some(ref module) = TRUSTFORMERS_RN {
                let request_str = CStr::from_ptr(request_json).to_str().unwrap_or("{}");

                // Note: This is a synchronous wrapper for the async function
                // In a real implementation, you'd use a runtime like tokio
                let result = module
                    .inference_sync(serde_json::from_str(request_str).unwrap_or_default())
                    .unwrap_or_else(|e| InferenceResponse {
                        request_id: "error".to_string(),
                        success: false,
                        output_data: Vec::new(),
                        output_shape: Vec::new(),
                        inference_time_ms: 0.0,
                        memory_used_mb: 0,
                        error_message: Some(e.to_string()),
                        metrics: PerformanceMetrics {
                            preprocessing_time_ms: 0.0,
                            inference_time_ms: 0.0,
                            postprocessing_time_ms: 0.0,
                            memory_allocation_mb: 0,
                            cache_hit_ratio: 0.0,
                        },
                    });

                let response_json = serde_json::to_string(&result).unwrap_or_default();
                CString::new(response_json)
                    .unwrap_or_else(|_| CString::new("response").expect("Failed to create CString"))
                    .into_raw()
            } else {
                let error = serde_json::json!({"error": "Module not initialized"});
                CString::new(error.to_string())
                    .unwrap_or_else(|_| CString::new("error").expect("Failed to create CString"))
                    .into_raw()
            }
        }
    }

    /// Get available models
    #[no_mangle]
    pub extern "C" fn trustformers_rn_get_models() -> *mut c_char {
        unsafe {
            if let Some(ref module) = TRUSTFORMERS_RN {
                let result = module
                    .get_available_models()
                    .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()}).to_string());
                CString::new(result)
                    .unwrap_or_else(|_| CString::new("models").expect("Failed to create CString"))
                    .into_raw()
            } else {
                let error = serde_json::json!({"error": "Module not initialized"});
                CString::new(error.to_string())
                    .unwrap_or_else(|_| CString::new("error").expect("Failed to create CString"))
                    .into_raw()
            }
        }
    }

    /// Get device capabilities
    #[no_mangle]
    pub extern "C" fn trustformers_rn_get_device_capabilities() -> *mut c_char {
        unsafe {
            if let Some(ref module) = TRUSTFORMERS_RN {
                let result = module
                    .get_device_capabilities()
                    .unwrap_or_else(|e| serde_json::json!({"error": e.to_string()}).to_string());
                CString::new(result)
                    .unwrap_or_else(|_| {
                        CString::new("capabilities").expect("Failed to create CString")
                    })
                    .into_raw()
            } else {
                let error = serde_json::json!({"error": "Module not initialized"});
                CString::new(error.to_string())
                    .unwrap_or_else(|_| CString::new("error").expect("Failed to create CString"))
                    .into_raw()
            }
        }
    }

    /// Free string allocated by Rust
    #[no_mangle]
    pub extern "C" fn trustformers_rn_free_string(ptr: *mut c_char) {
        if !ptr.is_null() {
            unsafe {
                let _ = CString::from_raw(ptr);
            }
        }
    }
}

impl Default for InferenceRequest {
    fn default() -> Self {
        Self {
            request_id: "default".to_string(),
            model_id: "default_model".to_string(),
            input_data: Vec::new(),
            input_shape: Vec::new(),
            config_override: None,
            enable_preprocessing: true,
            enable_postprocessing: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_react_native_config_validation() {
        let config = ReactNativeConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.max_concurrent_inferences = 0;
        assert!(invalid_config.validate().is_err());

        invalid_config.max_concurrent_inferences = 15;
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_optimized_configs() {
        let perf_config = ReactNativeConfig::performance_optimized();
        assert_eq!(perf_config.max_concurrent_inferences, 8);
        assert!(perf_config.enable_result_caching);
        assert_eq!(perf_config.max_cache_size_mb, 100);

        let mem_config = ReactNativeConfig::memory_optimized();
        assert_eq!(mem_config.max_concurrent_inferences, 2);
        assert!(!mem_config.enable_result_caching);
        assert_eq!(mem_config.max_cache_size_mb, 10);
    }

    #[test]
    fn test_performance_stats() {
        let stats = PerformanceStats::new();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.successful_requests, 0);
        assert_eq!(stats.failed_requests, 0);
    }

    #[tokio::test]
    async fn test_react_native_module_creation() {
        let rn_config = ReactNativeConfig::default();
        let mobile_config = MobileConfig::default();

        let result = TrustformersReactNative::new(rn_config, mobile_config);
        assert!(result.is_ok());
    }
}
