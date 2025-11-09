//! WebAssembly Support for Mobile Web Deployment
//!
//! This module provides WebAssembly integration for running TrustformeRS models
//! in mobile web browsers with optimizations for mobile constraints.

#[cfg(all(target_arch = "wasm32", feature = "web"))]
use crate::{MemoryOptimization, MobileConfig, MobileStats};
#[cfg(all(target_arch = "wasm32", feature = "web"))]
use serde::{Deserialize, Serialize};
#[cfg(all(target_arch = "wasm32", feature = "web"))]
use std::collections::HashMap;
use trustformers_core::error::{CoreError, Result};
#[cfg(all(target_arch = "wasm32", feature = "web"))]
use trustformers_core::Tensor;

#[cfg(all(target_arch = "wasm32", feature = "web"))]
use js_sys::{Array, ArrayBuffer, Promise, Uint8Array};
#[cfg(all(target_arch = "wasm32", feature = "web"))]
use wasm_bindgen::prelude::*;
#[cfg(all(target_arch = "wasm32", feature = "web"))]
use wasm_bindgen_futures::JsFuture;
#[cfg(all(target_arch = "wasm32", feature = "web"))]
use web_sys::{console, window, Navigator, Performance, WorkerGlobalScope};

/// WebAssembly inference engine for mobile web
#[cfg(all(target_arch = "wasm32", feature = "web"))]
#[wasm_bindgen]
pub struct WasmMobileEngine {
    config: WasmMobileConfig,
    model_weights: Option<HashMap<String, Tensor>>,
    stats: WasmMobileStats,
    browser_info: BrowserInfo,
    worker_pool: Option<WorkerPool>,
}

/// WebAssembly mobile configuration
#[cfg(all(target_arch = "wasm32", feature = "web"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmMobileConfig {
    /// Use Web Workers for parallel computation
    pub use_web_workers: bool,
    /// Number of Web Workers to spawn
    pub num_workers: usize,
    /// Use WebGL for GPU acceleration
    pub use_webgl: bool,
    /// Use WebGPU if available
    pub use_webgpu: bool,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Memory optimization for web constraints
    pub memory_optimization: MemoryOptimization,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Enable streaming execution
    pub enable_streaming: bool,
    /// Batch size for web inference
    pub batch_size: usize,
}

/// Browser capability information
#[cfg(all(target_arch = "wasm32", feature = "web"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserInfo {
    /// User agent string
    pub user_agent: String,
    /// Available memory (estimate)
    pub memory_mb: Option<usize>,
    /// Hardware concurrency (CPU cores)
    pub hardware_concurrency: usize,
    /// WebGL support
    pub has_webgl: bool,
    /// WebGL2 support
    pub has_webgl2: bool,
    /// WebGPU support
    pub has_webgpu: bool,
    /// SIMD support
    pub has_simd: bool,
    /// Web Workers support
    pub has_web_workers: bool,
    /// Service Worker support
    pub has_service_workers: bool,
    /// Mobile device detection
    pub is_mobile: bool,
    /// Touch support
    pub has_touch: bool,
}

/// WebAssembly statistics
#[cfg(all(target_arch = "wasm32", feature = "web"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmMobileStats {
    /// Total inferences performed
    pub total_inferences: usize,
    /// Average inference time (ms)
    pub avg_inference_time_ms: f32,
    /// Memory usage in MB
    pub memory_usage_mb: usize,
    /// Web Worker utilization
    pub worker_utilization: f32,
    /// WebGL/WebGPU usage
    pub gpu_utilization: f32,
    /// Model load time (ms)
    pub model_load_time_ms: f32,
    /// Compilation time for WASM (ms)
    pub wasm_compilation_time_ms: f32,
}

/// Web Worker pool for parallel computation
#[cfg(all(target_arch = "wasm32", feature = "web"))]
struct WorkerPool {
    workers: Vec<web_sys::Worker>,
    task_queue: Vec<WorkerTask>,
    available_workers: Vec<usize>,
}

/// Task for Web Worker execution
#[cfg(all(target_arch = "wasm32", feature = "web"))]
struct WorkerTask {
    input_data: Vec<u8>,
    callback: Box<dyn FnOnce(Vec<u8>)>,
}

#[cfg(all(target_arch = "wasm32", feature = "web"))]
#[wasm_bindgen]
impl WasmMobileEngine {
    /// Create new WebAssembly mobile engine
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<WasmMobileEngine, JsValue> {
        console_error_panic_hook::set_once();

        let config: WasmMobileConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("Config parse error: {}", e)))?;

        let browser_info = Self::detect_browser_info()
            .map_err(|e| JsValue::from_str(&format!("Browser detection error: {}", e)))?;

        let stats = WasmMobileStats::new();

        let mut engine = Self {
            config,
            model_weights: None,
            stats,
            browser_info,
            worker_pool: None,
        };

        // Initialize Web Workers if enabled
        if engine.config.use_web_workers && engine.browser_info.has_web_workers {
            engine
                .init_worker_pool()
                .map_err(|e| JsValue::from_str(&format!("Worker pool init error: {}", e)))?;
        }

        Ok(engine)
    }

    /// Load model weights from ArrayBuffer
    #[wasm_bindgen]
    pub async fn load_model(&mut self, model_data: ArrayBuffer) -> Result<(), JsValue> {
        let start_time = Self::get_performance_now();

        let uint8_array = Uint8Array::new(&model_data);
        let data_vec = uint8_array.to_vec();

        console::log_1(&JsValue::from_str(&format!(
            "Loading WASM model ({} bytes)",
            data_vec.len()
        )));

        // Parse model weights (simplified)
        let weights = self
            .parse_model_weights(&data_vec)
            .map_err(|e| JsValue::from_str(&format!("Model parse error: {}", e)))?;

        self.model_weights = Some(weights);

        let load_time = Self::get_performance_now() - start_time;
        self.stats.model_load_time_ms = load_time;

        console::log_1(&JsValue::from_str(&format!(
            "WASM model loaded in {:.2}ms",
            load_time
        )));

        Ok(())
    }

    /// Perform inference with input data
    #[wasm_bindgen]
    pub async fn inference(&mut self, input_data: ArrayBuffer) -> Result<ArrayBuffer, JsValue> {
        if self.model_weights.is_none() {
            return Err(JsValue::from_str("Model not loaded"));
        }

        let start_time = Self::get_performance_now();

        let uint8_array = Uint8Array::new(&input_data);
        let input_vec = uint8_array.to_vec();

        // Parse input tensors
        let input_tensors = self
            .parse_input_data(&input_vec)
            .map_err(|e| JsValue::from_str(&format!("Input parse error: {}", e)))?;

        // Perform inference
        let output_tensors = if self.config.use_web_workers && self.worker_pool.is_some() {
            self.inference_with_workers(&input_tensors).await
        } else {
            self.inference_single_threaded(&input_tensors)
        }
        .map_err(|e| JsValue::from_str(&format!("Inference error: {}", e)))?;

        // Serialize output
        let output_data = self
            .serialize_output(&output_tensors)
            .map_err(|e| JsValue::from_str(&format!("Output serialize error: {}", e)))?;

        let inference_time = Self::get_performance_now() - start_time;
        self.stats.update_inference(inference_time);

        // Convert to ArrayBuffer
        let output_array = Uint8Array::from(&output_data[..]);
        Ok(output_array.buffer())
    }

    /// Get current statistics as JSON string
    #[wasm_bindgen]
    pub fn get_stats(&self) -> String {
        serde_json::to_string(&self.stats).unwrap_or_default()
    }

    /// Get browser information as JSON string
    #[wasm_bindgen]
    pub fn get_browser_info(&self) -> String {
        serde_json::to_string(&self.browser_info).unwrap_or_default()
    }

    /// Optimize configuration for current browser
    #[wasm_bindgen]
    pub fn optimize_for_browser(&mut self) -> Result<(), JsValue> {
        // Adjust configuration based on browser capabilities
        if !self.browser_info.has_web_workers {
            self.config.use_web_workers = false;
            self.config.num_workers = 0;
        }

        if !self.browser_info.has_webgl2 && !self.browser_info.has_webgpu {
            self.config.use_webgl = false;
            self.config.use_webgpu = false;
        }

        if !self.browser_info.has_simd {
            self.config.enable_simd = false;
        }

        // Adjust memory limits for mobile
        if self.browser_info.is_mobile {
            self.config.max_memory_mb = self.config.max_memory_mb.min(512);
            self.config.batch_size = self.config.batch_size.min(2);
            self.config.memory_optimization = MemoryOptimization::Maximum;
        }

        // Reduce worker count for low-end devices
        if self.browser_info.hardware_concurrency <= 2 {
            self.config.num_workers = 1;
        }

        console::log_1(&JsValue::from_str(&format!(
            "Optimized WASM config for {} (mobile: {})",
            self.browser_info.user_agent.split_whitespace().next().unwrap_or("Unknown"),
            self.browser_info.is_mobile
        )));

        Ok(())
    }

    // Private methods

    fn detect_browser_info() -> Result<BrowserInfo, Box<dyn std::error::Error>> {
        let window = window().ok_or("No window object")?;
        let navigator = window.navigator();

        let user_agent = navigator.user_agent().unwrap_or_default();
        let hardware_concurrency = navigator.hardware_concurrency() as usize;

        // Detect mobile
        let is_mobile = user_agent.to_lowercase().contains("mobile")
            || user_agent.to_lowercase().contains("android")
            || user_agent.to_lowercase().contains("iphone");

        // Check capabilities
        let has_web_workers = window.worker().is_ok();
        let has_touch = window.navigator().max_touch_points() > 0;

        // Memory estimation (very rough)
        let memory_mb = if is_mobile {
            Some(2048) // Assume 2GB for mobile
        } else {
            Some(8192) // Assume 8GB for desktop
        };

        Ok(BrowserInfo {
            user_agent,
            memory_mb,
            hardware_concurrency: hardware_concurrency.max(1),
            has_webgl: true,   // Assume modern browser
            has_webgl2: true,  // Assume modern browser
            has_webgpu: false, // Conservative assumption
            has_simd: true,    // Assume WASM SIMD support
            has_web_workers,
            has_service_workers: true, // Assume modern browser
            is_mobile,
            has_touch,
        })
    }

    fn get_performance_now() -> f32 {
        if let Some(window) = window() {
            if let Ok(performance) = window.performance() {
                return performance.now() as f32;
            }
        }
        0.0
    }

    fn init_worker_pool(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let num_workers = self.config.num_workers.min(self.browser_info.hardware_concurrency);
        let mut workers = Vec::with_capacity(num_workers);

        for i in 0..num_workers {
            // In a real implementation, this would create actual Web Workers
            // For now, we'll just track that we would create them
            console::log_1(&JsValue::from_str(&format!("Creating worker {}", i)));
        }

        self.worker_pool = Some(WorkerPool {
            workers,
            task_queue: Vec::new(),
            available_workers: (0..num_workers).collect(),
        });

        Ok(())
    }

    fn parse_model_weights(
        &self,
        _data: &[u8],
    ) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
        // Simplified model weight parsing
        let mut weights = HashMap::new();
        weights.insert("layer1".to_string(), Tensor::ones(&[10, 10])?);
        Ok(weights)
    }

    fn parse_input_data(
        &self,
        _data: &[u8],
    ) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
        // Simplified input parsing
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), Tensor::ones(&[1, 10])?);
        Ok(inputs)
    }

    async fn inference_with_workers(
        &mut self,
        input: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
        // Simplified worker-based inference
        // In a real implementation, this would distribute work across Web Workers
        self.inference_single_threaded(input)
    }

    fn inference_single_threaded(
        &self,
        input: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
        // Simplified single-threaded inference
        let mut output = HashMap::new();

        if let Some(input_tensor) = input.get("input") {
            // Simple computation (in practice would be model inference)
            let output_tensor = input_tensor.clone();
            output.insert("output".to_string(), output_tensor);
        }

        Ok(output)
    }

    fn serialize_output(
        &self,
        _output: &HashMap<String, Tensor>,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        // Simplified output serialization
        Ok(vec![0u8; 32]) // Placeholder
    }
}

#[cfg(all(target_arch = "wasm32", feature = "web"))]
impl Default for WasmMobileConfig {
    fn default() -> Self {
        Self {
            use_web_workers: true,
            num_workers: 2,
            use_webgl: true,
            use_webgpu: false,
            enable_simd: true,
            memory_optimization: MemoryOptimization::Balanced,
            max_memory_mb: 512,
            enable_streaming: true,
            batch_size: 1,
        }
    }
}

#[cfg(all(target_arch = "wasm32", feature = "web"))]
impl WasmMobileStats {
    fn new() -> Self {
        Self {
            total_inferences: 0,
            avg_inference_time_ms: 0.0,
            memory_usage_mb: 0,
            worker_utilization: 0.0,
            gpu_utilization: 0.0,
            model_load_time_ms: 0.0,
            wasm_compilation_time_ms: 0.0,
        }
    }

    fn update_inference(&mut self, inference_time_ms: f32) {
        self.total_inferences += 1;

        // Update running average
        let alpha = 0.1;
        if self.total_inferences == 1 {
            self.avg_inference_time_ms = inference_time_ms;
        } else {
            self.avg_inference_time_ms =
                alpha * inference_time_ms + (1.0 - alpha) * self.avg_inference_time_ms;
        }
    }
}

/// Convert mobile config to WASM config
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub fn mobile_config_to_wasm(mobile_config: &MobileConfig) -> WasmMobileConfig {
    WasmMobileConfig {
        use_web_workers: mobile_config.enable_batching && mobile_config.num_threads > 1,
        num_workers: mobile_config.num_threads.max(1).min(4),
        use_webgl: true,
        use_webgpu: false,
        enable_simd: true,
        memory_optimization: mobile_config.memory_optimization,
        max_memory_mb: mobile_config.max_memory_mb,
        enable_streaming: true,
        batch_size: mobile_config.max_batch_size,
    }
}

/// JavaScript utility functions
#[cfg(all(target_arch = "wasm32", feature = "web"))]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

/// High-level JavaScript API
#[cfg(all(target_arch = "wasm32", feature = "web"))]
#[wasm_bindgen]
pub async fn create_mobile_engine(config_json: &str) -> Result<WasmMobileEngine, JsValue> {
    let mut engine = WasmMobileEngine::new(config_json)?;
    engine.optimize_for_browser()?;
    Ok(engine)
}

#[cfg(all(target_arch = "wasm32", feature = "web"))]
#[wasm_bindgen]
pub fn get_browser_capabilities() -> String {
    match WasmMobileEngine::detect_browser_info() {
        Ok(info) => serde_json::to_string(&info).unwrap_or_default(),
        Err(_) => "{}".to_string(),
    }
}

// Stub implementations for non-WASM platforms
#[cfg(not(all(target_arch = "wasm32", feature = "web")))]
pub struct WasmMobileEngine;

#[cfg(not(all(target_arch = "wasm32", feature = "web")))]
impl WasmMobileEngine {
    pub fn new(_config: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Err("WebAssembly features only available when compiled to WASM".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_config_defaults() {
        #[cfg(all(target_arch = "wasm32", feature = "web"))]
        {
            let config = WasmMobileConfig::default();
            assert!(config.use_web_workers);
            assert_eq!(config.num_workers, 2);
            assert!(config.use_webgl);
            assert!(!config.use_webgpu);
            assert!(config.enable_simd);
            assert_eq!(config.max_memory_mb, 512);
        }
    }

    #[test]
    fn test_mobile_to_wasm_config_conversion() {
        #[cfg(all(target_arch = "wasm32", feature = "web"))]
        {
            let mobile_config = crate::MobileConfig {
                memory_optimization: MemoryOptimization::Maximum,
                max_memory_mb: 256,
                num_threads: 4,
                enable_batching: true,
                max_batch_size: 2,
                ..Default::default()
            };

            let wasm_config = mobile_config_to_wasm(&mobile_config);
            assert_eq!(wasm_config.memory_optimization, MemoryOptimization::Maximum);
            assert_eq!(wasm_config.max_memory_mb, 256);
            assert_eq!(wasm_config.num_workers, 4);
            assert!(wasm_config.use_web_workers);
            assert_eq!(wasm_config.batch_size, 2);
        }
    }
}
