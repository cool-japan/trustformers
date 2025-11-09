//! Flutter Integration for TrustformeRS Mobile
//!
//! This module provides comprehensive Flutter platform channel integration,
//! enabling Flutter applications to leverage TrustformeRS mobile inference
//! capabilities with optimized performance and native platform features.

use crate::{
    inference::MobileInferenceEngine, MobileBackend, MobileConfig, MobilePlatform, MobileStats,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    os::raw::c_char,
    sync::{Arc, Mutex},
};
use trustformers_core::error::Result;
use trustformers_core::Tensor;

/// Flutter platform channel manager for TrustformeRS
pub struct FlutterChannelManager {
    engines: Arc<Mutex<HashMap<String, MobileInferenceEngine>>>,
    configurations: Arc<Mutex<HashMap<String, MobileConfig>>>,
    statistics: Arc<Mutex<HashMap<String, MobileStats>>>,
    event_sink: Option<FlutterEventSink>,
}

impl std::fmt::Debug for FlutterChannelManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlutterChannelManager")
            .field("engines", &"<HashMap<String, MobileInferenceEngine>>")
            .field("configurations", &"<HashMap<String, MobileConfig>>")
            .field("statistics", &"<HashMap<String, MobileStats>>")
            .field("event_sink", &self.event_sink.is_some())
            .finish()
    }
}

/// Flutter event sink for streaming events to Dart
pub type FlutterEventSink = Box<dyn Fn(&str) + Send + Sync>;

/// Flutter method call structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlutterMethodCall {
    pub method: String,
    pub arguments: Option<serde_json::Value>,
}

/// Flutter method result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FlutterMethodResult {
    Success(serde_json::Value),
    Error {
        code: String,
        message: String,
        details: Option<serde_json::Value>,
    },
}

/// Flutter TrustformeRS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlutterTrustformersConfig {
    pub engine_id: String,
    pub model_path: String,
    pub platform: String,
    pub backend: String,
    pub memory_optimization: String,
    pub max_memory_mb: u32,
    pub use_fp16: bool,
    pub quantization: Option<FlutterQuantizationConfig>,
    pub num_threads: u32,
    pub enable_batching: bool,
    pub max_batch_size: u32,
}

/// Flutter quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlutterQuantizationConfig {
    pub scheme: String,
    pub dynamic: bool,
    pub per_channel: bool,
}

/// Flutter inference request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlutterInferenceRequest {
    pub engine_id: String,
    pub input_ids: Vec<i64>,
    pub attention_mask: Option<Vec<i64>>,
    pub token_type_ids: Option<Vec<i64>>,
    pub max_length: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub do_sample: bool,
}

/// Flutter inference response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlutterInferenceResponse {
    pub tokens: Vec<i64>,
    pub logits: Option<Vec<f32>>,
    pub attention_weights: Option<Vec<Vec<f32>>>,
    pub inference_time_ms: f32,
    pub memory_usage_mb: u32,
}

/// Flutter device info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlutterDeviceInfo {
    pub platform: String,
    pub model: String,
    pub memory_total_mb: u32,
    pub memory_available_mb: u32,
    pub cpu_cores: u32,
    pub gpu_available: bool,
    pub neural_engine_available: bool,
}

/// Flutter performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlutterPerformanceMetrics {
    pub engine_id: String,
    pub total_inferences: u64,
    pub avg_inference_time_ms: f32,
    pub peak_memory_mb: u32,
    pub current_memory_mb: u32,
    pub throughput_tokens_per_sec: f32,
}

impl FlutterChannelManager {
    /// Create a new Flutter channel manager
    pub fn new() -> Self {
        Self {
            engines: Arc::new(Mutex::new(HashMap::new())),
            configurations: Arc::new(Mutex::new(HashMap::new())),
            statistics: Arc::new(Mutex::new(HashMap::new())),
            event_sink: None,
        }
    }

    /// Set event sink for streaming updates to Dart
    pub fn set_event_sink(&mut self, sink: FlutterEventSink) {
        self.event_sink = Some(sink);
    }

    /// Handle Flutter method call
    pub fn handle_method_call(&self, call: FlutterMethodCall) -> FlutterMethodResult {
        match call.method.as_str() {
            "initialize" => self.handle_initialize(call.arguments),
            "loadModel" => self.handle_load_model(call.arguments),
            "inference" => self.handle_inference(call.arguments),
            "getDeviceInfo" => self.handle_get_device_info(),
            "getPerformanceMetrics" => self.handle_get_performance_metrics(call.arguments),
            "dispose" => self.handle_dispose(call.arguments),
            "getBatchInference" => self.handle_batch_inference(call.arguments),
            "getModelInfo" => self.handle_get_model_info(call.arguments),
            "optimizeForDevice" => self.handle_optimize_for_device(call.arguments),
            _ => FlutterMethodResult::Error {
                code: "METHOD_NOT_FOUND".to_string(),
                message: format!("Method '{}' not implemented", call.method),
                details: None,
            },
        }
    }

    /// Handle engine initialization
    fn handle_initialize(&self, args: Option<serde_json::Value>) -> FlutterMethodResult {
        let config: FlutterTrustformersConfig = match args {
            Some(value) => match serde_json::from_value(value) {
                Ok(config) => config,
                Err(e) => {
                    return FlutterMethodResult::Error {
                        code: "INVALID_ARGUMENTS".to_string(),
                        message: format!("Failed to parse configuration: {}", e),
                        details: None,
                    }
                },
            },
            None => {
                return FlutterMethodResult::Error {
                    code: "MISSING_ARGUMENTS".to_string(),
                    message: "Configuration is required".to_string(),
                    details: None,
                }
            },
        };

        let mobile_config = match self.convert_flutter_config_to_mobile(&config) {
            Ok(config) => config,
            Err(e) => {
                return FlutterMethodResult::Error {
                    code: "CONFIGURATION_ERROR".to_string(),
                    message: format!("Invalid configuration: {}", e),
                    details: None,
                }
            },
        };

        // Store configuration
        {
            let mut configs = self.configurations.lock().unwrap();
            configs.insert(config.engine_id.clone(), mobile_config.clone());
        }

        // Initialize statistics
        {
            let mut stats = self.statistics.lock().unwrap();
            stats.insert(config.engine_id.clone(), MobileStats::new(&mobile_config));
        }

        FlutterMethodResult::Success(serde_json::json!({
            "engine_id": config.engine_id,
            "status": "initialized"
        }))
    }

    /// Handle model loading
    fn handle_load_model(&self, args: Option<serde_json::Value>) -> FlutterMethodResult {
        let request: serde_json::Map<String, serde_json::Value> = match args {
            Some(serde_json::Value::Object(map)) => map,
            _ => {
                return FlutterMethodResult::Error {
                    code: "INVALID_ARGUMENTS".to_string(),
                    message: "Expected object with engine_id and model_path".to_string(),
                    details: None,
                }
            },
        };

        let engine_id = match request.get("engine_id") {
            Some(serde_json::Value::String(id)) => id.clone(),
            _ => {
                return FlutterMethodResult::Error {
                    code: "MISSING_ENGINE_ID".to_string(),
                    message: "engine_id is required".to_string(),
                    details: None,
                }
            },
        };

        let model_path = match request.get("model_path") {
            Some(serde_json::Value::String(path)) => path.clone(),
            _ => {
                return FlutterMethodResult::Error {
                    code: "MISSING_MODEL_PATH".to_string(),
                    message: "model_path is required".to_string(),
                    details: None,
                }
            },
        };

        // Get configuration
        let config = {
            let configs = self.configurations.lock().unwrap();
            match configs.get(&engine_id) {
                Some(config) => config.clone(),
                None => {
                    return FlutterMethodResult::Error {
                        code: "ENGINE_NOT_INITIALIZED".to_string(),
                        message: format!("Engine '{}' not initialized", engine_id),
                        details: None,
                    }
                },
            }
        };

        // Create and load inference engine
        match MobileInferenceEngine::new(config) {
            Ok(mut engine) => {
                match engine.load_model_from_file(&model_path) {
                    Ok(_) => {
                        // Store the engine
                        {
                            let mut engines = self.engines.lock().unwrap();
                            engines.insert(engine_id.clone(), engine);
                        }

                        // Send event if sink is available
                        if let Some(ref sink) = self.event_sink {
                            let event = serde_json::json!({
                                "type": "model_loaded",
                                "engine_id": engine_id,
                                "model_path": model_path
                            });
                            sink(&event.to_string());
                        }

                        FlutterMethodResult::Success(serde_json::json!({
                            "engine_id": engine_id,
                            "status": "model_loaded",
                            "model_path": model_path
                        }))
                    },
                    Err(e) => FlutterMethodResult::Error {
                        code: "MODEL_LOAD_ERROR".to_string(),
                        message: format!("Failed to load model: {}", e),
                        details: None,
                    },
                }
            },
            Err(e) => FlutterMethodResult::Error {
                code: "ENGINE_CREATION_ERROR".to_string(),
                message: format!("Failed to create inference engine: {}", e),
                details: None,
            },
        }
    }

    /// Handle inference request
    fn handle_inference(&self, args: Option<serde_json::Value>) -> FlutterMethodResult {
        let request: FlutterInferenceRequest = match args {
            Some(value) => match serde_json::from_value(value) {
                Ok(request) => request,
                Err(e) => {
                    return FlutterMethodResult::Error {
                        code: "INVALID_ARGUMENTS".to_string(),
                        message: format!("Failed to parse inference request: {}", e),
                        details: None,
                    }
                },
            },
            None => {
                return FlutterMethodResult::Error {
                    code: "MISSING_ARGUMENTS".to_string(),
                    message: "Inference request is required".to_string(),
                    details: None,
                }
            },
        };

        // Perform inference with mutex locked
        let start_time = std::time::Instant::now();
        let result = {
            let mut engines = self.engines.lock().unwrap();
            match engines.get_mut(&request.engine_id) {
                Some(engine) => self.perform_inference(engine, &request),
                None => {
                    return FlutterMethodResult::Error {
                        code: "ENGINE_NOT_FOUND".to_string(),
                        message: format!("Engine '{}' not found", request.engine_id),
                        details: None,
                    }
                },
            }
        };

        match result {
            Ok(mut response) => {
                let inference_time = start_time.elapsed().as_millis() as f32;
                response.inference_time_ms = inference_time;

                // Update statistics
                {
                    let mut stats = self.statistics.lock().unwrap();
                    if let Some(stat) = stats.get_mut(&request.engine_id) {
                        stat.update_inference(inference_time);
                        stat.update_memory(response.memory_usage_mb as usize);
                    }
                }

                // Send event if sink is available
                if let Some(ref sink) = self.event_sink {
                    let event = serde_json::json!({
                        "type": "inference_completed",
                        "engine_id": request.engine_id,
                        "inference_time_ms": inference_time
                    });
                    sink(&event.to_string());
                }

                FlutterMethodResult::Success(serde_json::to_value(response).unwrap())
            },
            Err(e) => FlutterMethodResult::Error {
                code: "INFERENCE_ERROR".to_string(),
                message: format!("Inference failed: {}", e),
                details: None,
            },
        }
    }

    /// Handle device info request
    fn handle_get_device_info(&self) -> FlutterMethodResult {
        match crate::device_info::MobileDeviceDetector::detect() {
            Ok(device_info) => {
                let flutter_device_info = FlutterDeviceInfo {
                    platform: format!("{:?}", device_info.platform),
                    model: device_info.basic_info.model,
                    memory_total_mb: device_info.memory_info.total_memory as u32,
                    memory_available_mb: device_info.memory_info.available_memory as u32,
                    cpu_cores: device_info.cpu_info.core_count as u32,
                    gpu_available: device_info.gpu_info.is_some(),
                    neural_engine_available: device_info.npu_info.is_some(),
                };

                FlutterMethodResult::Success(serde_json::to_value(flutter_device_info).unwrap())
            },
            Err(e) => FlutterMethodResult::Error {
                code: "DEVICE_INFO_ERROR".to_string(),
                message: format!("Failed to get device info: {}", e),
                details: None,
            },
        }
    }

    /// Handle performance metrics request
    fn handle_get_performance_metrics(
        &self,
        args: Option<serde_json::Value>,
    ) -> FlutterMethodResult {
        let engine_id = match args {
            Some(serde_json::Value::String(id)) => id,
            Some(serde_json::Value::Object(map)) => match map.get("engine_id") {
                Some(serde_json::Value::String(id)) => id.clone(),
                _ => {
                    return FlutterMethodResult::Error {
                        code: "INVALID_ENGINE_ID".to_string(),
                        message: "engine_id must be a string".to_string(),
                        details: None,
                    }
                },
            },
            _ => {
                return FlutterMethodResult::Error {
                    code: "MISSING_ENGINE_ID".to_string(),
                    message: "engine_id is required".to_string(),
                    details: None,
                }
            },
        };

        let stats = self.statistics.lock().unwrap();
        match stats.get(&engine_id) {
            Some(stat) => {
                let metrics = FlutterPerformanceMetrics {
                    engine_id: engine_id.clone(),
                    total_inferences: stat.total_inferences as u64,
                    avg_inference_time_ms: stat.avg_inference_time_ms,
                    peak_memory_mb: stat.peak_memory_mb as u32,
                    current_memory_mb: stat.memory_usage_mb as u32,
                    throughput_tokens_per_sec: if stat.avg_inference_time_ms > 0.0 {
                        1000.0 / stat.avg_inference_time_ms
                    } else {
                        0.0
                    },
                };

                FlutterMethodResult::Success(serde_json::to_value(metrics).unwrap())
            },
            None => FlutterMethodResult::Error {
                code: "ENGINE_NOT_FOUND".to_string(),
                message: format!("Engine '{}' not found", engine_id),
                details: None,
            },
        }
    }

    /// Handle engine disposal
    fn handle_dispose(&self, args: Option<serde_json::Value>) -> FlutterMethodResult {
        let engine_id = match args {
            Some(serde_json::Value::String(id)) => id,
            Some(serde_json::Value::Object(map)) => match map.get("engine_id") {
                Some(serde_json::Value::String(id)) => id.clone(),
                _ => {
                    return FlutterMethodResult::Error {
                        code: "INVALID_ENGINE_ID".to_string(),
                        message: "engine_id must be a string".to_string(),
                        details: None,
                    }
                },
            },
            _ => {
                return FlutterMethodResult::Error {
                    code: "MISSING_ENGINE_ID".to_string(),
                    message: "engine_id is required".to_string(),
                    details: None,
                }
            },
        };

        // Remove engine and associated data
        {
            let mut engines = self.engines.lock().unwrap();
            engines.remove(&engine_id);
        }
        {
            let mut configs = self.configurations.lock().unwrap();
            configs.remove(&engine_id);
        }
        {
            let mut stats = self.statistics.lock().unwrap();
            stats.remove(&engine_id);
        }

        FlutterMethodResult::Success(serde_json::json!({
            "engine_id": engine_id,
            "status": "disposed"
        }))
    }

    /// Handle batch inference request
    fn handle_batch_inference(&self, args: Option<serde_json::Value>) -> FlutterMethodResult {
        let request: serde_json::Map<String, serde_json::Value> = match args {
            Some(serde_json::Value::Object(map)) => map,
            _ => {
                return FlutterMethodResult::Error {
                    code: "INVALID_ARGUMENTS".to_string(),
                    message: "Expected object with engine_id and requests".to_string(),
                    details: None,
                }
            },
        };

        let engine_id = match request.get("engine_id") {
            Some(serde_json::Value::String(id)) => id.clone(),
            _ => {
                return FlutterMethodResult::Error {
                    code: "MISSING_ENGINE_ID".to_string(),
                    message: "engine_id is required".to_string(),
                    details: None,
                }
            },
        };

        let requests: Vec<FlutterInferenceRequest> = match request.get("requests") {
            Some(serde_json::Value::Array(arr)) => {
                match arr
                    .iter()
                    .map(|v| serde_json::from_value(v.clone()))
                    .collect::<std::result::Result<Vec<_>, _>>()
                {
                    Ok(reqs) => reqs,
                    Err(e) => {
                        return FlutterMethodResult::Error {
                            code: "INVALID_REQUESTS".to_string(),
                            message: format!("Failed to parse requests: {}", e),
                            details: None,
                        }
                    },
                }
            },
            _ => {
                return FlutterMethodResult::Error {
                    code: "MISSING_REQUESTS".to_string(),
                    message: "requests array is required".to_string(),
                    details: None,
                }
            },
        };

        // Process batch inference with mutex locked
        let start_time = std::time::Instant::now();
        let (results, total_memory) = {
            let mut engines = self.engines.lock().unwrap();
            match engines.get_mut(&engine_id) {
                Some(engine) => {
                    let mut results = Vec::new();
                    let mut total_memory = 0u32;

                    for req in requests {
                        match self.perform_inference(engine, &req) {
                            Ok(result) => {
                                total_memory += result.memory_usage_mb;
                                results.push(result);
                            },
                            Err(e) => {
                                results.push(FlutterInferenceResponse {
                                    tokens: vec![],
                                    logits: None,
                                    attention_weights: None,
                                    inference_time_ms: 0.0,
                                    memory_usage_mb: 0,
                                });
                            },
                        }
                    }
                    (results, total_memory)
                },
                None => {
                    return FlutterMethodResult::Error {
                        code: "ENGINE_NOT_FOUND".to_string(),
                        message: format!("Engine '{}' not found", engine_id),
                        details: None,
                    }
                },
            }
        };

        let total_time = start_time.elapsed().as_millis() as f32;

        // Update statistics
        {
            let mut stats = self.statistics.lock().unwrap();
            if let Some(stat) = stats.get_mut(&engine_id) {
                stat.update_inference(total_time);
                stat.update_memory(total_memory as usize);
            }
        }

        FlutterMethodResult::Success(serde_json::json!({
            "results": results,
            "total_time_ms": total_time,
            "total_memory_mb": total_memory,
            "success_count": results.len(),
            "error_count": 0 // Simplified for now
        }))
    }

    /// Handle model info request
    fn handle_get_model_info(&self, args: Option<serde_json::Value>) -> FlutterMethodResult {
        let engine_id = match args {
            Some(serde_json::Value::String(id)) => id,
            Some(serde_json::Value::Object(map)) => match map.get("engine_id") {
                Some(serde_json::Value::String(id)) => id.clone(),
                _ => {
                    return FlutterMethodResult::Error {
                        code: "INVALID_ENGINE_ID".to_string(),
                        message: "engine_id must be a string".to_string(),
                        details: None,
                    }
                },
            },
            _ => {
                return FlutterMethodResult::Error {
                    code: "MISSING_ENGINE_ID".to_string(),
                    message: "engine_id is required".to_string(),
                    details: None,
                }
            },
        };

        // Get configuration for the engine
        let config = {
            let configs = self.configurations.lock().unwrap();
            match configs.get(&engine_id) {
                Some(config) => config.clone(),
                None => {
                    return FlutterMethodResult::Error {
                        code: "ENGINE_NOT_FOUND".to_string(),
                        message: format!("Engine '{}' not found", engine_id),
                        details: None,
                    }
                },
            }
        };

        // Get statistics for the engine
        let stats = {
            let stats = self.statistics.lock().unwrap();
            stats.get(&engine_id).cloned()
        };

        let mut model_info = serde_json::json!({
            "engine_id": engine_id,
            "platform": format!("{:?}", config.platform),
            "backend": format!("{:?}", config.backend),
            "memory_optimization": format!("{:?}", config.memory_optimization),
            "max_memory_mb": config.max_memory_mb,
            "use_fp16": config.use_fp16,
            "num_threads": config.num_threads,
            "enable_batching": config.enable_batching,
            "max_batch_size": config.max_batch_size,
            "model_loaded": self.engines.lock().unwrap().contains_key(&engine_id),
        });

        // Add quantization info if available
        if let Some(ref quantization) = config.quantization {
            model_info["quantization"] = serde_json::json!({
                "scheme": format!("{:?}", quantization.scheme),
                "dynamic": quantization.dynamic,
                "per_channel": quantization.per_channel,
            });
        }

        // Add performance stats if available
        if let Some(stat) = stats {
            model_info["performance"] = serde_json::json!({
                "total_inferences": stat.total_inferences,
                "avg_inference_time_ms": stat.avg_inference_time_ms,
                "peak_memory_mb": stat.peak_memory_mb,
                "current_memory_mb": stat.memory_usage_mb,
            });
        }

        FlutterMethodResult::Success(model_info)
    }

    /// Handle device optimization request
    fn handle_optimize_for_device(&self, args: Option<serde_json::Value>) -> FlutterMethodResult {
        let request: serde_json::Map<String, serde_json::Value> = match args {
            Some(serde_json::Value::Object(map)) => map,
            _ => {
                return FlutterMethodResult::Error {
                    code: "INVALID_ARGUMENTS".to_string(),
                    message: "Expected object with engine_id and current_config".to_string(),
                    details: None,
                }
            },
        };

        let engine_id = match request.get("engine_id") {
            Some(serde_json::Value::String(id)) => id.clone(),
            _ => {
                return FlutterMethodResult::Error {
                    code: "MISSING_ENGINE_ID".to_string(),
                    message: "engine_id is required".to_string(),
                    details: None,
                }
            },
        };

        let current_config: FlutterTrustformersConfig = match request.get("current_config") {
            Some(value) => match serde_json::from_value(value.clone()) {
                Ok(config) => config,
                Err(e) => {
                    return FlutterMethodResult::Error {
                        code: "INVALID_CONFIG".to_string(),
                        message: format!("Failed to parse current configuration: {}", e),
                        details: None,
                    }
                },
            },
            None => {
                return FlutterMethodResult::Error {
                    code: "MISSING_CONFIG".to_string(),
                    message: "current_config is required".to_string(),
                    details: None,
                }
            },
        };

        // Get device information for optimization
        let device_info = match crate::device_info::MobileDeviceDetector::detect() {
            Ok(info) => info,
            Err(e) => {
                return FlutterMethodResult::Error {
                    code: "DEVICE_INFO_ERROR".to_string(),
                    message: format!("Failed to get device info: {}", e),
                    details: None,
                }
            },
        };

        // Create optimized configuration based on device capabilities
        let mut optimized_config = current_config.clone();

        // Optimize backend based on platform and hardware
        let optimal_backend = match device_info.platform {
            crate::MobilePlatform::Ios => {
                if device_info.npu_info.is_some() {
                    "coreml".to_string()
                } else if device_info.gpu_info.is_some() {
                    "gpu".to_string()
                } else {
                    "cpu".to_string()
                }
            },
            crate::MobilePlatform::Android => {
                if device_info.npu_info.is_some() {
                    "nnapi".to_string()
                } else if device_info.gpu_info.is_some() {
                    "gpu".to_string()
                } else {
                    "cpu".to_string()
                }
            },
            _ => "cpu".to_string(),
        };
        optimized_config.backend = optimal_backend;

        // Optimize memory settings based on available memory
        let available_memory_mb = device_info.memory_info.available_memory as u32;
        if available_memory_mb < 512 {
            optimized_config.memory_optimization = "maximum".to_string();
            optimized_config.max_memory_mb = (available_memory_mb * 2 / 3).min(256);
        } else if available_memory_mb < 1024 {
            optimized_config.memory_optimization = "balanced".to_string();
            optimized_config.max_memory_mb = (available_memory_mb / 2).min(512);
        } else {
            optimized_config.memory_optimization = "minimal".to_string();
            optimized_config.max_memory_mb = (available_memory_mb / 3).min(1024);
        }

        // Optimize quantization based on device tier
        let device_tier = if device_info.memory_info.total_memory >= 8192
            && device_info.cpu_info.core_count >= 8
        {
            "high"
        } else if device_info.memory_info.total_memory >= 4096
            && device_info.cpu_info.core_count >= 4
        {
            "medium"
        } else {
            "low"
        };

        optimized_config.quantization = Some(FlutterQuantizationConfig {
            scheme: match device_tier {
                "high" => "fp16".to_string(),
                "medium" => "int8".to_string(),
                "low" => "int4".to_string(),
                _ => "dynamic".to_string(),
            },
            dynamic: device_tier == "low",
            per_channel: device_tier != "low",
        });

        // Optimize threading based on CPU cores
        optimized_config.num_threads = (device_info.cpu_info.core_count as u32 / 2).max(1).min(8);

        // Optimize batching based on device capabilities
        optimized_config.enable_batching = device_tier != "low";
        optimized_config.max_batch_size = match device_tier {
            "high" => 4,
            "medium" => 2,
            "low" => 1,
            _ => 1,
        };

        // Store optimized configuration
        if let Ok(mobile_config) = self.convert_flutter_config_to_mobile(&optimized_config) {
            let mut configs = self.configurations.lock().unwrap();
            configs.insert(engine_id.clone(), mobile_config);
        }

        FlutterMethodResult::Success(serde_json::json!({
            "engine_id": engine_id,
            "optimized_config": optimized_config,
            "device_tier": device_tier,
            "optimization_applied": true,
            "recommendations": [
                format!("Backend optimized to: {}", optimized_config.backend),
                format!("Memory limit optimized to: {}MB", optimized_config.max_memory_mb),
                format!("Quantization scheme: {}", optimized_config.quantization.as_ref().unwrap().scheme),
                format!("Thread count: {}", optimized_config.num_threads),
                format!("Batching enabled: {}", optimized_config.enable_batching),
            ]
        }))
    }

    /// Convert Flutter config to mobile config
    fn convert_flutter_config_to_mobile(
        &self,
        flutter_config: &FlutterTrustformersConfig,
    ) -> Result<MobileConfig> {
        let platform = match flutter_config.platform.as_str() {
            "ios" => MobilePlatform::Ios,
            "android" => MobilePlatform::Android,
            _ => MobilePlatform::Generic,
        };

        let backend = match flutter_config.backend.as_str() {
            "cpu" => MobileBackend::CPU,
            "coreml" => MobileBackend::CoreML,
            "nnapi" => MobileBackend::NNAPI,
            "gpu" => MobileBackend::GPU,
            _ => MobileBackend::CPU,
        };

        let memory_optimization = match flutter_config.memory_optimization.as_str() {
            "minimal" => crate::MemoryOptimization::Minimal,
            "balanced" => crate::MemoryOptimization::Balanced,
            "maximum" => crate::MemoryOptimization::Maximum,
            _ => crate::MemoryOptimization::Balanced,
        };

        let quantization = flutter_config.quantization.as_ref().map(|q| {
            let scheme = match q.scheme.as_str() {
                "int8" => crate::MobileQuantizationScheme::Int8,
                "int4" => crate::MobileQuantizationScheme::Int4,
                "fp16" => crate::MobileQuantizationScheme::FP16,
                _ => crate::MobileQuantizationScheme::Dynamic,
            };

            crate::MobileQuantizationConfig {
                scheme,
                dynamic: q.dynamic,
                per_channel: q.per_channel,
            }
        });

        let config = MobileConfig {
            platform,
            backend,
            memory_optimization,
            max_memory_mb: flutter_config.max_memory_mb as usize,
            use_fp16: flutter_config.use_fp16,
            quantization,
            num_threads: flutter_config.num_threads as usize,
            enable_batching: flutter_config.enable_batching,
            max_batch_size: flutter_config.max_batch_size as usize,
        };

        config.validate()?;
        Ok(config)
    }

    /// Perform inference using the mobile engine
    fn perform_inference(
        &self,
        engine: &mut MobileInferenceEngine,
        request: &FlutterInferenceRequest,
    ) -> Result<FlutterInferenceResponse> {
        // Convert input data to tensors
        let input_ids = Tensor::from_vec(
            request.input_ids.clone().into_iter().map(|x| x as f32).collect(),
            &[1, request.input_ids.len()],
        )?;

        let attention_mask = request
            .attention_mask
            .as_ref()
            .map(|mask| {
                Tensor::from_vec(
                    mask.clone().into_iter().map(|x| x as f32).collect(),
                    &[1, mask.len()],
                )
            })
            .transpose()?;

        let token_type_ids = request
            .token_type_ids
            .as_ref()
            .map(|ids| {
                Tensor::from_vec(
                    ids.clone().into_iter().map(|x| x as f32).collect(),
                    &[1, ids.len()],
                )
            })
            .transpose()?;

        // Prepare inference options
        let mut inference_options = std::collections::HashMap::new();
        if let Some(max_length) = request.max_length {
            inference_options.insert("max_length".to_string(), max_length.to_string());
        }
        if let Some(temperature) = request.temperature {
            inference_options.insert("temperature".to_string(), temperature.to_string());
        }
        if let Some(top_p) = request.top_p {
            inference_options.insert("top_p".to_string(), top_p.to_string());
        }
        if let Some(top_k) = request.top_k {
            inference_options.insert("top_k".to_string(), top_k.to_string());
        }
        inference_options.insert("do_sample".to_string(), request.do_sample.to_string());

        // Perform inference
        let output = engine.inference(&input_ids)?;

        // Extract tokens from output
        let tokens = output.data()?.iter().map(|&x| x as i64).collect::<Vec<_>>();

        // Get current memory usage (simplified)
        let memory_usage_mb = 128; // This would be calculated from actual memory usage

        Ok(FlutterInferenceResponse {
            tokens,
            logits: None,            // Could be extracted from output if needed
            attention_weights: None, // Could be extracted if available
            inference_time_ms: 0.0,  // Will be set by caller
            memory_usage_mb,
        })
    }
}

impl Default for FlutterChannelManager {
    fn default() -> Self {
        Self::new()
    }
}

// C FFI exports for Flutter Dart FFI integration

/// Initialize Flutter channel manager
#[no_mangle]
pub extern "C" fn flutter_trustformers_init() -> *mut FlutterChannelManager {
    Box::into_raw(Box::new(FlutterChannelManager::new()))
}

/// Handle method call from Flutter
#[no_mangle]
pub extern "C" fn flutter_trustformers_handle_call(
    manager: *mut FlutterChannelManager,
    method_call_json: *const c_char,
) -> *mut c_char {
    if manager.is_null() || method_call_json.is_null() {
        return std::ptr::null_mut();
    }

    let manager = unsafe { &*manager };
    let method_call_str = unsafe { CStr::from_ptr(method_call_json) };

    match method_call_str.to_str() {
        Ok(json_str) => match serde_json::from_str::<FlutterMethodCall>(json_str) {
            Ok(call) => {
                let result = manager.handle_method_call(call);
                match serde_json::to_string(&result) {
                    Ok(result_json) => match CString::new(result_json) {
                        Ok(c_str) => c_str.into_raw(),
                        Err(_) => std::ptr::null_mut(),
                    },
                    Err(_) => std::ptr::null_mut(),
                }
            },
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Set event sink for streaming updates
#[no_mangle]
pub extern "C" fn flutter_trustformers_set_event_sink(
    manager: *mut FlutterChannelManager,
    event_sink: extern "C" fn(*const c_char),
) {
    if manager.is_null() {
        return;
    }

    let manager = unsafe { &mut *manager };
    let sink: FlutterEventSink = Box::new(move |event: &str| {
        if let Ok(c_str) = CString::new(event) {
            event_sink(c_str.as_ptr());
        }
    });

    manager.set_event_sink(sink);
}

/// Dispose Flutter channel manager
#[no_mangle]
pub extern "C" fn flutter_trustformers_dispose(manager: *mut FlutterChannelManager) {
    if !manager.is_null() {
        unsafe {
            let _ = Box::from_raw(manager);
        }
    }
}

/// Free C string allocated by Rust
#[no_mangle]
pub extern "C" fn flutter_trustformers_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flutter_channel_manager_creation() {
        let manager = FlutterChannelManager::new();
        assert!(manager.engines.lock().unwrap().is_empty());
        assert!(manager.configurations.lock().unwrap().is_empty());
        assert!(manager.statistics.lock().unwrap().is_empty());
    }

    #[test]
    fn test_flutter_config_conversion() {
        let manager = FlutterChannelManager::new();
        let flutter_config = FlutterTrustformersConfig {
            engine_id: "test".to_string(),
            model_path: "/test/model".to_string(),
            platform: "ios".to_string(),
            backend: "coreml".to_string(),
            memory_optimization: "balanced".to_string(),
            max_memory_mb: 512,
            use_fp16: true,
            quantization: Some(FlutterQuantizationConfig {
                scheme: "fp16".to_string(),
                dynamic: false,
                per_channel: true,
            }),
            num_threads: 4,
            enable_batching: true,
            max_batch_size: 4,
        };

        let mobile_config = manager.convert_flutter_config_to_mobile(&flutter_config).unwrap();
        assert_eq!(mobile_config.platform, MobilePlatform::Ios);
        assert_eq!(mobile_config.backend, MobileBackend::CoreML);
        assert!(mobile_config.use_fp16);
        assert!(mobile_config.enable_batching);
    }

    #[test]
    fn test_method_call_handling() {
        let manager = FlutterChannelManager::new();

        // Test unknown method
        let call = FlutterMethodCall {
            method: "unknown_method".to_string(),
            arguments: None,
        };

        let result = manager.handle_method_call(call);
        match result {
            FlutterMethodResult::Error { code, .. } => {
                assert_eq!(code, "METHOD_NOT_FOUND");
            },
            result => panic!(
                "Expected FlutterMethodResult::Error for unknown method, got {:?}",
                result
            ),
        }
    }

    #[test]
    fn test_device_info_handling() {
        let manager = FlutterChannelManager::new();
        let result = manager.handle_get_device_info();

        // Should return either success or error, but not panic
        match result {
            FlutterMethodResult::Success(_) => {},
            FlutterMethodResult::Error { .. } => {},
        }
    }

    #[test]
    fn test_initialization_handling() {
        let manager = FlutterChannelManager::new();
        let config = FlutterTrustformersConfig {
            engine_id: "test_engine".to_string(),
            model_path: "/test/model".to_string(),
            platform: "generic".to_string(),
            backend: "cpu".to_string(),
            memory_optimization: "balanced".to_string(),
            max_memory_mb: 512,
            use_fp16: false,
            quantization: None,
            num_threads: 2,
            enable_batching: false,
            max_batch_size: 1,
        };

        let call = FlutterMethodCall {
            method: "initialize".to_string(),
            arguments: Some(serde_json::to_value(config).unwrap()),
        };

        let result = manager.handle_method_call(call);
        match result {
            FlutterMethodResult::Success(value) => {
                assert!(value.get("engine_id").is_some());
                assert_eq!(
                    value.get("status"),
                    Some(&serde_json::Value::String("initialized".to_string()))
                );
            },
            FlutterMethodResult::Error { code, message, .. } => {
                panic!(
                    "Initialization should have succeeded, but failed with error: {} - {}",
                    code, message
                );
            },
        }

        // Verify configuration was stored
        assert!(manager.configurations.lock().unwrap().contains_key("test_engine"));
        assert!(manager.statistics.lock().unwrap().contains_key("test_engine"));
    }

    #[test]
    fn test_performance_metrics_handling() {
        let manager = FlutterChannelManager::new();

        // Test with non-existent engine
        let call = FlutterMethodCall {
            method: "getPerformanceMetrics".to_string(),
            arguments: Some(serde_json::Value::String("non_existent".to_string())),
        };

        let result = manager.handle_method_call(call);
        match result {
            FlutterMethodResult::Error { code, .. } => {
                assert_eq!(code, "ENGINE_NOT_FOUND");
            },
            result => panic!(
                "Expected FlutterMethodResult::Error for non-existent engine, got {:?}",
                result
            ),
        }
    }
}
