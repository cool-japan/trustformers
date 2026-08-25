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
    /// Shape of the tensor most recently passed to [`TrustformersReactNative::inference`]
    /// for this model, or `None` before any inference has run. Model
    /// checkpoint formats (safetensors/PyTorch/ONNX weight maps) do not
    /// declare a fixed input signature the way a compiled graph does --
    /// shape only becomes knowable from a request that actually named one
    /// -- so this is observed, not declared, and absent rather than a
    /// fabricated architecture-agnostic guess until then.
    pub input_shape: Option<Vec<usize>>,
    /// Shape of the tensor most recently produced by inference for this
    /// model, or `None` before any inference has run. Same reasoning as
    /// `input_shape`.
    pub output_shape: Option<Vec<usize>>,
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
    /// Input/output shapes actually observed from the most recent
    /// successful inference for each `model_id`, keyed by that id. Nothing
    /// in a loaded checkpoint (safetensors/PyTorch/ONNX weight maps)
    /// declares a fixed input signature the way a compiled graph would, so
    /// there is no shape to report until a real request has run through
    /// `perform_inference_internal` -- see that function's write to this
    /// map, and `get_model_info`'s read of it. Previously
    /// `get_model_info`/`get_available_models` returned a hardcoded
    /// ImageNet-classifier shape (`[1, 224, 224, 3]` / `[1, 1000]`) for
    /// every model regardless of architecture.
    observed_shapes: Arc<Mutex<HashMap<String, ObservedShapes>>>,
}

/// Real input/output tensor shapes captured from an actual inference call.
#[derive(Debug, Clone)]
struct ObservedShapes {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
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
        let observed_shapes = Arc::new(Mutex::new(HashMap::new()));

        let device_capabilities = Self::detect_device_capabilities()?;

        Ok(Self {
            config,
            inference_engine,
            model_manager,
            request_cache,
            performance_stats,
            device_capabilities,
            observed_shapes,
        })
    }

    /// Initialize the React Native module
    pub fn initialize(&self) -> Result<String> {
        tracing::info!("Initializing TrustformeRS React Native module");

        // Initialize inference engine
        let mut engine = self.inference_engine.lock().unwrap_or_else(|p| p.into_inner());
        engine.initialize()?;

        // Initialize model manager
        let model_manager = self.model_manager.lock().unwrap_or_else(|p| p.into_inner());
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

        let mut engine = self.inference_engine.lock().unwrap_or_else(|p| p.into_inner());
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
        let model_manager = self.model_manager.lock().unwrap_or_else(|p| p.into_inner());
        let models = model_manager.list_models();

        let model_infos: Vec<ModelInfo> = models
            .iter()
            .map(|metadata| {
                let (input_shape, output_shape) = self.observed_shapes_for(&metadata.model_id);
                ModelInfo {
                    model_id: metadata.model_id.clone(),
                    model_type: metadata.model_type.clone(),
                    version: metadata.version.clone(),
                    size_bytes: metadata.size_bytes,
                    is_loaded: self.is_model_loaded(&metadata.model_id),
                    input_shape,
                    output_shape,
                    supported_features: vec!["inference".to_string()],
                }
            })
            .collect();

        Ok(serde_json::to_string(&model_infos)?)
    }

    /// Download model from server
    pub async fn download_model(&self, model_id: &str) -> Result<String> {
        tracing::info!("Downloading model: {}", model_id);

        let mut model_manager = self.model_manager.lock().unwrap_or_else(|p| p.into_inner());

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
            let mut engine = self.inference_engine.lock().unwrap_or_else(|p| p.into_inner());
            let _ = engine.unload_model(model_id);
        }

        // Remove from model manager
        {
            let mut model_manager = self.model_manager.lock().unwrap_or_else(|p| p.into_inner());
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
        let stats = self.performance_stats.lock().unwrap_or_else(|p| p.into_inner());

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
        let mut cache = self.request_cache.lock().unwrap_or_else(|p| p.into_inner());
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

        let mut engine = self.inference_engine.lock().unwrap_or_else(|p| p.into_inner());
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
        let observed_shapes = self.observed_shapes.clone();

        tokio::task::spawn_blocking(move || {
            Self::perform_inference_internal(engine, request, config, observed_shapes)
        })
        .await
        .map_err(|e| CoreError::from(TrustformersError::runtime_error(e.to_string())))?
    }

    fn inference_sync(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        Self::perform_inference_internal(
            self.inference_engine.clone(),
            request,
            self.config.clone(),
            self.observed_shapes.clone(),
        )
    }

    fn perform_inference_internal(
        engine: Arc<Mutex<MobileInferenceEngine>>,
        request: InferenceRequest,
        config: ReactNativeConfig,
        observed_shapes: Arc<Mutex<HashMap<String, ObservedShapes>>>,
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
        let input_shape = request.input_shape.clone();
        let input_tensor = Tensor::from_vec(request.input_data, &request.input_shape)?;
        metrics.preprocessing_time_ms = preprocess_start.elapsed().as_millis() as f64;

        // Inference
        let inference_start = std::time::Instant::now();
        let (result, memory_used_mb) = {
            let mut engine_lock = engine.lock().unwrap_or_else(|p| p.into_inner());
            let result = engine_lock.run_inference(&request.model_id, &input_tensor);
            // Real estimate from the engine's own loaded-weight parameter
            // count and quantization scheme (`get_memory_info` ->
            // `estimate_memory_footprint`), read while still holding the
            // lock so it reflects the state this exact call just observed.
            // Previously a hardcoded `50` (success) / `0` (failure)
            // regardless of the model's actual size.
            let memory_used_mb = engine_lock.get_memory_info().total_memory_mb;
            (result, memory_used_mb)
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

                // Record the real shapes this call observed so
                // `get_model_info`/`get_available_models` can report them
                // instead of a fabricated architecture-agnostic guess.
                {
                    let mut shapes = observed_shapes.lock().unwrap_or_else(|p| p.into_inner());
                    shapes.insert(
                        request.model_id.clone(),
                        ObservedShapes {
                            input_shape: input_shape.clone(),
                            output_shape: output_shape.clone(),
                        },
                    );
                }

                Ok(InferenceResponse {
                    request_id: request.request_id,
                    success: true,
                    output_data: output_data.to_vec(),
                    output_shape,
                    inference_time_ms: total_time,
                    memory_used_mb,
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
                    // The engine's real memory footprint, not the previous
                    // hardcoded `0` -- a failed *inference* (e.g. a shape
                    // mismatch) does not mean the loaded model stopped
                    // occupying memory.
                    memory_used_mb,
                    error_message: Some(error.to_string()),
                    metrics,
                })
            },
        }
    }

    fn check_cache(&self, request: &InferenceRequest) -> Option<InferenceResponse> {
        let cache = self.request_cache.lock().unwrap_or_else(|p| p.into_inner());

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

        let mut cache = self.request_cache.lock().unwrap_or_else(|p| p.into_inner());

        // Simple cache eviction if size limit exceeded
        if cache.len() >= self.config.max_cache_size_mb * 100 {
            // Rough estimation
            cache.clear();
        }

        let cache_key = format!("{}_response", response.request_id);
        cache.insert(cache_key, response.clone());
    }

    fn update_cache_stats(&self, cache_hit: bool) {
        let mut stats = self.performance_stats.lock().unwrap_or_else(|p| p.into_inner());
        if cache_hit {
            stats.cache_hits += 1;
        } else {
            stats.cache_misses += 1;
        }
    }

    fn update_performance_stats(&self, response: &InferenceResponse) {
        let mut stats = self.performance_stats.lock().unwrap_or_else(|p| p.into_inner());

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
        let model_manager = self.model_manager.lock().unwrap_or_else(|p| p.into_inner());

        if let Some(metadata) = model_manager.get_model(model_id) {
            let (input_shape, output_shape) = self.observed_shapes_for(model_id);
            Ok(ModelInfo {
                model_id: metadata.model_id.clone(),
                model_type: metadata.model_type.clone(),
                version: metadata.version.clone(),
                size_bytes: metadata.size_bytes,
                is_loaded: self.is_model_loaded(model_id),
                input_shape,
                output_shape,
                supported_features: vec!["inference".to_string()],
            })
        } else {
            Err(TrustformersError::runtime_error(format!("Model not found: {}", model_id)).into())
        }
    }

    /// Real shapes from the most recent successful inference for
    /// `model_id`, or `(None, None)` before any inference has run. See
    /// [`ObservedShapes`] and the write site in
    /// `perform_inference_internal`.
    fn observed_shapes_for(&self, model_id: &str) -> (Option<Vec<usize>>, Option<Vec<usize>>) {
        let observed = self.observed_shapes.lock().unwrap_or_else(|p| p.into_inner());
        match observed.get(model_id) {
            Some(shapes) => (
                Some(shapes.input_shape.clone()),
                Some(shapes.output_shape.clone()),
            ),
            None => (None, None),
        }
    }

    fn is_model_loaded(&self, model_id: &str) -> bool {
        let engine = self.inference_engine.lock().unwrap_or_else(|p| p.into_inner());
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

// React-Native-facing wrappers around the real `MobileInferenceEngine` API
// (`inference.rs`). This used to be a "mock implementation... for React
// Native" that shadowed the same method names with no-ops: `load_model_from_path`
// did nothing, `run_inference` returned `input.clone()` unchanged,
// `is_model_loaded` always answered `true`, `unload_model`/`configure_model`
// were no-ops -- every one of `ReactNativeMobileModule`'s public methods
// above (`load_model`, `inference`, `remove_model`, `configure_model`)
// ultimately calls through here, so none of them ever touched real model
// weights. `MobileInferenceEngine` holds one active model at a time (see
// its `model_weights: Option<HashMap<String, Tensor>>`); `model_id` here
// identifies *which* model the caller believes is active for logging
// purposes; per-ID metadata (size, version, availability) is tracked
// separately by this module's own `ModelManager`.
impl MobileInferenceEngine {
    fn initialize(&mut self) -> Result<()> {
        // `MobileInferenceEngine::new` already performs all real
        // initialization (config validation, optimizer setup); there is
        // nothing further to do before a model is loaded.
        Ok(())
    }

    fn load_model_from_path(&mut self, model_id: &str, model_path: &str) -> Result<()> {
        self.load_model_from_file(model_path).map_err(|e| {
            CoreError::from(TrustformersError::runtime_error(format!(
                "failed to load model '{model_id}' from '{model_path}': {e}"
            )))
        })
    }

    fn unload_model(&mut self, _model_id: &str) -> Result<()> {
        self.clear_loaded_model();
        Ok(())
    }

    fn run_inference(&mut self, model_id: &str, input: &Tensor) -> Result<Tensor> {
        if !self.has_loaded_model() {
            return Err(CoreError::from(TrustformersError::runtime_error(format!(
                "cannot run inference for model '{model_id}': no model is currently loaded"
            ))));
        }
        self.inference(input).map_err(|e| {
            CoreError::from(TrustformersError::runtime_error(format!(
                "inference failed for model '{model_id}': {e}"
            )))
        })
    }

    fn is_model_loaded(&self, _model_id: &str) -> bool {
        self.has_loaded_model()
    }

    fn configure_model(&mut self, _model_id: &str, config: MobileConfig) -> Result<()> {
        self.update_config(config).map_err(CoreError::from)
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
                            CString::new("initialization complete").unwrap_or_default()
                        })
                        .into_raw()
                },
                Err(e) => {
                    let error = serde_json::json!({"error": e.to_string()});
                    CString::new(error.to_string())
                        .unwrap_or_else(|_| CString::new("error").unwrap_or_default())
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

                // This C ABI entry point cannot be `async fn` (the C
                // caller has no executor to poll it), so it calls
                // `inference_sync` -- the synchronous path that runs
                // straight on this thread -- rather than the async
                // `inference()` used by non-FFI callers. `tokio` is a real
                // workspace dependency and this crate does use it
                // elsewhere (`inference_background`'s
                // `tokio::task::spawn_blocking`, reached from `inference()`
                // when `use_background_thread` is set); it is simply not
                // reachable from a plain `extern "C" fn` with no `Runtime`
                // handle in scope, which is why this specific entry point
                // is intentionally synchronous rather than an omission.
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
                    .unwrap_or_else(|_| CString::new("response").unwrap_or_default())
                    .into_raw()
            } else {
                let error = serde_json::json!({"error": "Module not initialized"});
                CString::new(error.to_string())
                    .unwrap_or_else(|_| CString::new("error").unwrap_or_default())
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
                    .unwrap_or_else(|_| CString::new("models").unwrap_or_default())
                    .into_raw()
            } else {
                let error = serde_json::json!({"error": "Module not initialized"});
                CString::new(error.to_string())
                    .unwrap_or_else(|_| CString::new("error").unwrap_or_default())
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
                    .unwrap_or_else(|_| CString::new("capabilities").unwrap_or_default())
                    .into_raw()
            } else {
                let error = serde_json::json!({"error": "Module not initialized"});
                CString::new(error.to_string())
                    .unwrap_or_else(|_| CString::new("error").unwrap_or_default())
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

    /// Regression test for the previous "Mock implementation of
    /// MobileInferenceEngine methods for React Native": `is_model_loaded`
    /// always returned the literal `true`, `run_inference` always returned
    /// `input.clone()` regardless of whether any model was loaded, and
    /// `load_model_from_path` was a no-op that succeeded for any path
    /// (including nonexistent ones). All three must now reflect real
    /// engine state.
    #[test]
    fn test_bridge_is_model_loaded_and_run_inference_reflect_real_state() {
        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("engine creation failed");

        // Before any model is loaded: old mock said `true` unconditionally.
        assert!(!MobileInferenceEngine::is_model_loaded(&engine, "model-a"));

        // Running inference with nothing loaded: old mock happily returned
        // the input tensor back as a "successful" result.
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor");
        let result = MobileInferenceEngine::run_inference(&mut engine, "model-a", &input);
        assert!(
            result.is_err(),
            "inference with no loaded model must not fabricate success"
        );

        // A nonexistent model file: old mock's `load_model_from_path`
        // returned `Ok(())` for literally any path string.
        let load_result = MobileInferenceEngine::load_model_from_path(
            &mut engine,
            "model-a",
            "/nonexistent/definitely-not-a-real-model-file.safetensors",
        );
        assert!(
            load_result.is_err(),
            "loading a nonexistent file must not fabricate success"
        );
        assert!(
            !MobileInferenceEngine::is_model_loaded(&engine, "model-a"),
            "a failed load must not leave the engine reporting a loaded model"
        );
    }

    /// After a real model load, `run_inference` must produce real computed
    /// output (not an identity copy of the input) and `is_model_loaded`
    /// must report `true`; after `unload_model`, both must revert.
    #[test]
    fn test_bridge_load_and_unload_round_trip_with_real_computation() {
        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("engine creation failed");

        let mut weights = HashMap::new();
        weights.insert(
            "layer.weight".to_string(),
            Tensor::from_vec(vec![2.0, 0.0, 0.0, 2.0], &[2, 2]).expect("weight tensor"),
        );
        engine.load_model(weights).expect("direct load_model failed");
        assert!(MobileInferenceEngine::is_model_loaded(&engine, "model-a"));

        let input = Tensor::from_vec(vec![1.0, 3.0], &[1, 2]).expect("input tensor");
        let output = MobileInferenceEngine::run_inference(&mut engine, "model-a", &input)
            .expect("inference should succeed with a loaded model");
        assert_ne!(
            output.data().expect("output data"),
            input.data().expect("input data"),
            "output must be real computed data, not the input echoed back"
        );

        MobileInferenceEngine::unload_model(&mut engine, "model-a").expect("unload failed");
        assert!(!MobileInferenceEngine::is_model_loaded(&engine, "model-a"));
    }

    /// Regression test for the previous fabrication: `get_model_info` and
    /// `get_available_models` returned a hardcoded ImageNet-classifier
    /// shape (`input_shape: vec![1, 224, 224, 3]`,
    /// `output_shape: vec![1, 1000]`) for *every* model, regardless of
    /// architecture. `observed_shapes_for` must report `(None, None)`
    /// before any inference has run for a given `model_id`, and the real
    /// shapes that `perform_inference_internal` actually observed once one
    /// has.
    #[test]
    fn test_observed_shapes_are_absent_until_a_real_inference_populates_them() {
        let rn_config = ReactNativeConfig::default();
        let mobile_config = MobileConfig::default();
        let module =
            TrustformersReactNative::new(rn_config, mobile_config).expect("module creation failed");

        // Before any inference for this model_id: honestly absent, not a
        // fabricated architecture-agnostic guess.
        let (before_input, before_output) = module.observed_shapes_for("model-a");
        assert_eq!(before_input, None);
        assert_eq!(before_output, None);

        // Load a real (tiny) model directly into the shared engine so
        // `run_inference` has real weights to compute against.
        {
            let mut engine = module.inference_engine.lock().expect("lock poisoned");
            let mut weights = HashMap::new();
            weights.insert(
                "layer.weight".to_string(),
                Tensor::from_vec(vec![2.0, 0.0, 0.0, 2.0], &[2, 2]).expect("weight tensor"),
            );
            engine.load_model(weights).expect("load_model failed");
        }

        let request = InferenceRequest {
            request_id: "req-1".to_string(),
            model_id: "model-a".to_string(),
            input_data: vec![1.0, 3.0],
            input_shape: vec![1, 2],
            config_override: None,
            enable_preprocessing: true,
            enable_postprocessing: true,
        };
        let response = TrustformersReactNative::perform_inference_internal(
            module.inference_engine.clone(),
            request,
            module.config.clone(),
            module.observed_shapes.clone(),
        )
        .expect("perform_inference_internal should not itself error");
        assert!(
            response.success,
            "inference with a loaded model should succeed"
        );

        // After a real inference: the observed shapes must be the actual
        // request/response shapes, never the old hardcoded
        // `[1, 224, 224, 3]` / `[1, 1000]` (which would not even match --
        // this test's tensors are 2-D `[1, 2]`).
        let (after_input, after_output) = module.observed_shapes_for("model-a");
        assert_eq!(after_input, Some(vec![1, 2]));
        assert_eq!(after_output, Some(response.output_shape.clone()));
        assert_ne!(after_input, Some(vec![1, 224, 224, 3]));
        assert_ne!(after_output, Some(vec![1, 1000]));

        // A different, never-run model_id must still report absent --
        // shapes are per-model, not a global fallback once anything has
        // run once.
        let (other_input, other_output) = module.observed_shapes_for("model-b");
        assert_eq!(other_input, None);
        assert_eq!(other_output, None);
    }

    /// Regression test for the previous `memory_used_mb: 50 // Placeholder`
    /// (success path) and `memory_used_mb: 0` (failure path, coincidentally
    /// honest for an unloaded engine but for the wrong reason -- it was a
    /// constant either way). Both paths must now report the engine's real
    /// `get_memory_info().total_memory_mb`.
    #[test]
    fn test_memory_used_mb_reflects_real_engine_footprint_not_a_constant() {
        let rn_config = ReactNativeConfig::default();
        let mobile_config = MobileConfig::default();
        let module =
            TrustformersReactNative::new(rn_config, mobile_config).expect("module creation failed");

        // Failure path: no model loaded, so `run_inference` errors. The
        // real footprint of an empty engine is whatever
        // `get_memory_info()` reports for zero parameters (not necessarily
        // the old hardcoded `0`, though it may coincide -- the point is
        // this now genuinely reads the engine rather than asserting a
        // constant).
        let expected_empty_mb = {
            let engine = module.inference_engine.lock().expect("lock poisoned");
            engine.get_memory_info().total_memory_mb
        };
        let request = InferenceRequest {
            request_id: "req-fail".to_string(),
            model_id: "model-a".to_string(),
            input_data: vec![1.0, 2.0, 3.0],
            input_shape: vec![3],
            config_override: None,
            enable_preprocessing: true,
            enable_postprocessing: true,
        };
        let response = TrustformersReactNative::perform_inference_internal(
            module.inference_engine.clone(),
            request,
            module.config.clone(),
            module.observed_shapes.clone(),
        )
        .expect("perform_inference_internal should not itself error");
        assert!(!response.success, "no model is loaded: inference must fail");
        assert_eq!(response.memory_used_mb, expected_empty_mb);

        // Success path: load a real model with known parameter count and
        // confirm the reported figure tracks the *loaded* footprint, which
        // must differ from the empty-engine figure above for a nonzero
        // weight tensor.
        {
            let mut engine = module.inference_engine.lock().expect("lock poisoned");
            let mut weights = HashMap::new();
            weights.insert(
                "layer.weight".to_string(),
                Tensor::from_vec(vec![2.0, 0.0, 0.0, 2.0], &[2, 2]).expect("weight tensor"),
            );
            engine.load_model(weights).expect("load_model failed");
        }
        let expected_loaded_mb = {
            let engine = module.inference_engine.lock().expect("lock poisoned");
            engine.get_memory_info().total_memory_mb
        };
        let request = InferenceRequest {
            request_id: "req-ok".to_string(),
            model_id: "model-a".to_string(),
            input_data: vec![1.0, 3.0],
            input_shape: vec![1, 2],
            config_override: None,
            enable_preprocessing: true,
            enable_postprocessing: true,
        };
        let response = TrustformersReactNative::perform_inference_internal(
            module.inference_engine.clone(),
            request,
            module.config.clone(),
            module.observed_shapes.clone(),
        )
        .expect("perform_inference_internal should not itself error");
        assert!(
            response.success,
            "inference with a loaded model should succeed"
        );
        assert_eq!(response.memory_used_mb, expected_loaded_mb);
    }
}
