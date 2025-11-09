//! Edge runtime compatibility module
//!
//! This module provides compatibility features for various edge computing platforms:
//! - Cloudflare Workers
//! - Deno Deploy
//! - Vercel Edge Functions
//! - AWS Lambda@Edge
//! - Fastly Compute@Edge

use wasm_bindgen::prelude::*;

/// Edge runtime types supported
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeRuntime {
    /// Cloudflare Workers
    CloudflareWorkers,
    /// Deno Deploy
    DenoDeploy,
    /// Vercel Edge Functions
    VercelEdge,
    /// AWS Lambda@Edge
    LambdaEdge,
    /// Fastly Compute@Edge
    FastlyCompute,
    /// Generic/Unknown runtime
    Generic,
}

/// Edge runtime capabilities and limitations
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct EdgeCapabilities {
    runtime_type: EdgeRuntime,
    memory_limit_mb: u32,
    cpu_time_limit_ms: u32,
    supports_streaming: bool,
    supports_webgpu: bool,
    cold_start_optimization: bool,
}

#[wasm_bindgen]
impl EdgeCapabilities {
    #[wasm_bindgen(getter)]
    pub fn runtime_type(&self) -> EdgeRuntime {
        self.runtime_type
    }

    #[wasm_bindgen(getter)]
    pub fn memory_limit_mb(&self) -> u32 {
        self.memory_limit_mb
    }

    #[wasm_bindgen(getter)]
    pub fn cpu_time_limit_ms(&self) -> u32 {
        self.cpu_time_limit_ms
    }

    #[wasm_bindgen(getter)]
    pub fn supports_streaming(&self) -> bool {
        self.supports_streaming
    }

    #[wasm_bindgen(getter)]
    pub fn supports_webgpu(&self) -> bool {
        self.supports_webgpu
    }

    #[wasm_bindgen(getter)]
    pub fn cold_start_optimization(&self) -> bool {
        self.cold_start_optimization
    }
}

/// Edge runtime detector and optimizer
#[wasm_bindgen]
pub struct EdgeRuntimeDetector {
    capabilities: EdgeCapabilities,
}

#[wasm_bindgen]
impl EdgeRuntimeDetector {
    /// Create a new edge runtime detector
    pub fn new() -> EdgeRuntimeDetector {
        let capabilities = Self::detect_runtime_capabilities();
        EdgeRuntimeDetector { capabilities }
    }

    /// Get the detected edge runtime capabilities
    #[wasm_bindgen(getter)]
    pub fn capabilities(&self) -> EdgeCapabilities {
        self.capabilities.clone()
    }

    /// Check if the current runtime is suitable for ML inference
    pub fn is_ml_suitable(&self) -> bool {
        // Check memory and CPU time limits
        self.capabilities.memory_limit_mb >= 128 && self.capabilities.cpu_time_limit_ms >= 30000
        // 30 seconds minimum
    }

    /// Get recommended model size for current runtime
    pub fn recommended_model_size_mb(&self) -> u32 {
        // Use 50% of available memory for model
        (self.capabilities.memory_limit_mb / 2).clamp(16, 256)
    }

    /// Check if WebGPU acceleration is available
    pub fn webgpu_available(&self) -> bool {
        self.capabilities.supports_webgpu
    }

    /// Get cold start optimization recommendations
    pub fn get_cold_start_optimizations(&self) -> js_sys::Array {
        let optimizations = js_sys::Array::new();

        if self.capabilities.cold_start_optimization {
            optimizations.push(&JsValue::from_str("lazy_loading"));
            optimizations.push(&JsValue::from_str("module_splitting"));
            optimizations.push(&JsValue::from_str("precompiled_models"));
        }

        if self.capabilities.memory_limit_mb < 256 {
            optimizations.push(&JsValue::from_str("memory_pooling"));
            optimizations.push(&JsValue::from_str("quantization"));
        }

        if self.capabilities.cpu_time_limit_ms < 60000 {
            optimizations.push(&JsValue::from_str("async_execution"));
            optimizations.push(&JsValue::from_str("result_caching"));
        }

        optimizations
    }
}

impl EdgeRuntimeDetector {
    /// Detect the current edge runtime and its capabilities
    fn detect_runtime_capabilities() -> EdgeCapabilities {
        let runtime_type = Self::detect_runtime_type();

        // Set capabilities based on runtime type
        match runtime_type {
            EdgeRuntime::CloudflareWorkers => EdgeCapabilities {
                runtime_type,
                memory_limit_mb: 128,
                cpu_time_limit_ms: 30000,
                supports_streaming: true,
                supports_webgpu: false, // WebGPU not yet supported in Workers
                cold_start_optimization: true,
            },
            EdgeRuntime::DenoDeploy => EdgeCapabilities {
                runtime_type,
                memory_limit_mb: 512,
                cpu_time_limit_ms: 50000,
                supports_streaming: true,
                supports_webgpu: false, // Limited WebGPU support
                cold_start_optimization: true,
            },
            EdgeRuntime::VercelEdge => EdgeCapabilities {
                runtime_type,
                memory_limit_mb: 64,
                cpu_time_limit_ms: 25000,
                supports_streaming: true,
                supports_webgpu: false,
                cold_start_optimization: true,
            },
            EdgeRuntime::LambdaEdge => EdgeCapabilities {
                runtime_type,
                memory_limit_mb: 128,
                cpu_time_limit_ms: 5000, // Very limited execution time
                supports_streaming: false,
                supports_webgpu: false,
                cold_start_optimization: true,
            },
            EdgeRuntime::FastlyCompute => EdgeCapabilities {
                runtime_type,
                memory_limit_mb: 50,
                cpu_time_limit_ms: 50000,
                supports_streaming: true,
                supports_webgpu: false,
                cold_start_optimization: true,
            },
            EdgeRuntime::Generic => EdgeCapabilities {
                runtime_type,
                memory_limit_mb: 256,
                cpu_time_limit_ms: 60000,
                supports_streaming: true,
                supports_webgpu: true, // Assume full browser environment
                cold_start_optimization: false,
            },
        }
    }

    /// Detect the specific edge runtime type
    fn detect_runtime_type() -> EdgeRuntime {
        // Check for Cloudflare Workers
        if Self::check_global_exists("CloudflareWorkersGlobalScope")
            || Self::check_global_exists("WorkerGlobalScope") && Self::check_global_exists("caches")
        {
            return EdgeRuntime::CloudflareWorkers;
        }

        // Check for Deno Deploy
        if Self::check_global_exists("Deno") {
            return EdgeRuntime::DenoDeploy;
        }

        // Check for Vercel Edge (look for specific globals)
        if Self::check_global_exists("EdgeRuntime") {
            return EdgeRuntime::VercelEdge;
        }

        // Check for AWS Lambda@Edge (limited detection)
        if Self::check_global_exists("awslambda") {
            return EdgeRuntime::LambdaEdge;
        }

        // Check for Fastly Compute@Edge
        if Self::check_global_exists("fastly") {
            return EdgeRuntime::FastlyCompute;
        }

        // Default to generic runtime (browser or Node.js)
        EdgeRuntime::Generic
    }

    /// Check if a global variable exists
    fn check_global_exists(name: &str) -> bool {
        if let Some(global) = js_sys::global().dyn_ref::<web_sys::Window>() {
            js_sys::Reflect::has(global, &JsValue::from_str(name)).unwrap_or(false)
        } else if let Some(global) = js_sys::global().dyn_ref::<js_sys::Object>() {
            js_sys::Reflect::has(global, &JsValue::from_str(name)).unwrap_or(false)
        } else {
            false
        }
    }
}

impl Default for EdgeRuntimeDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Edge-optimized inference configuration
#[wasm_bindgen]
#[derive(Clone)]
pub struct EdgeInferenceConfig {
    max_batch_size: u32,
    streaming_enabled: bool,
    memory_pool_size: u32,
    use_quantization: bool,
    async_execution: bool,
}

#[wasm_bindgen]
impl EdgeInferenceConfig {
    /// Create optimal configuration for detected edge runtime
    pub fn for_runtime(capabilities: &EdgeCapabilities) -> EdgeInferenceConfig {
        let max_batch_size = if capabilities.memory_limit_mb < 128 { 1 } else { 4 };
        let streaming_enabled = capabilities.supports_streaming;
        let memory_pool_size =
            (capabilities.memory_limit_mb * 1024 * 1024 / 4).min(64 * 1024 * 1024); // Max 64MB pool
        let use_quantization = capabilities.memory_limit_mb < 256;
        let async_execution = capabilities.cpu_time_limit_ms < 60000;

        EdgeInferenceConfig {
            max_batch_size,
            streaming_enabled,
            memory_pool_size,
            use_quantization,
            async_execution,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn max_batch_size(&self) -> u32 {
        self.max_batch_size
    }

    #[wasm_bindgen(getter)]
    pub fn streaming_enabled(&self) -> bool {
        self.streaming_enabled
    }

    #[wasm_bindgen(getter)]
    pub fn memory_pool_size(&self) -> u32 {
        self.memory_pool_size
    }

    #[wasm_bindgen(getter)]
    pub fn use_quantization(&self) -> bool {
        self.use_quantization
    }

    #[wasm_bindgen(getter)]
    pub fn async_execution(&self) -> bool {
        self.async_execution
    }
}

/// Utility functions for edge runtime detection
#[wasm_bindgen]
pub fn detect_edge_runtime() -> EdgeRuntime {
    EdgeRuntimeDetector::detect_runtime_type()
}

#[wasm_bindgen]
pub fn get_edge_capabilities() -> EdgeCapabilities {
    EdgeRuntimeDetector::detect_runtime_capabilities()
}

#[wasm_bindgen]
pub fn is_edge_runtime_suitable() -> bool {
    let detector = EdgeRuntimeDetector::new();
    detector.is_ml_suitable()
}
