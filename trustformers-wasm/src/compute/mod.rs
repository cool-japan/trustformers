// Compute Module - Hardware acceleration and computation
//
// This module provides hardware-accelerated computation capabilities
// including WebGPU, WebGL, Web Workers, and SIMD operations.

use wasm_bindgen::JsCast;

#[cfg(feature = "webgpu")]
pub mod webgpu;

#[cfg(feature = "webgpu")]
pub mod gpu_tensor;

#[cfg(feature = "webgpu")]
pub mod webgpu_simple;

#[cfg(feature = "shared-memory")]
pub mod threads;

#[cfg(feature = "web-workers")]
pub mod web_workers;

// WebNN integration for NPU acceleration
pub mod webnn;

// Re-export main types for convenience
#[cfg(feature = "webgpu")]
pub use webgpu::{
    AsyncExecutor, DeviceCapabilities, DeviceSelector, DeviceType, ExecutionStatus, FusableOp,
    KernelFusion, OperationType, Priority, ShaderManager, WorkgroupTuner,
};

#[cfg(feature = "webgpu")]
pub use gpu_tensor::GpuTensor;

#[cfg(feature = "web-workers")]
pub use web_workers::{
    get_optimal_worker_count, is_web_workers_supported, WorkerCoordinator, WorkerPool,
    WorkerPriority, WorkerTaskType,
};

#[cfg(feature = "shared-memory")]
pub use threads::{
    get_optimal_thread_count, is_cross_origin_isolated, is_threading_supported, AtomicOperations,
    ThreadPool, ThreadSync, ThreadTaskType,
};

// WebNN exports
pub use webnn::{
    WebNNCapabilities, WebNNContext, WebNNDeviceType, WebNNExecutionPlan, WebNNGraphBuilder,
    WebNNModelAdapter, WebNNPowerPreference, WebNNTensor,
};

/// Compute module initialization
pub fn initialize() -> Result<(), ComputeError> {
    web_sys::console::log_1(&"Initializing TrustformeRS WASM compute module".into());

    #[cfg(feature = "webgpu")]
    {
        webgpu::initialize()?;
        web_sys::console::log_1(&"WebGPU subsystem initialized".into());
    }

    #[cfg(feature = "web-workers")]
    {
        web_workers::initialize()?;
        web_sys::console::log_1(&"Web Workers subsystem initialized".into());
    }

    #[cfg(feature = "shared-memory")]
    {
        threads::initialize()?;
        web_sys::console::log_1(&"Threading subsystem initialized".into());
    }

    web_sys::console::log_1(&"TrustformeRS WASM compute module initialized successfully".into());
    Ok(())
}

/// Compute module error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeError {
    WebGpuError(String),
    WorkerError(String),
    ThreadError(String),
    DeviceError(String),
    UnsupportedOperation(String),
    InitializationError(String),
}

impl core::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ComputeError::WebGpuError(msg) => write!(f, "WebGPU error: {}", msg),
            ComputeError::WorkerError(msg) => write!(f, "Worker error: {}", msg),
            ComputeError::ThreadError(msg) => write!(f, "Thread error: {}", msg),
            ComputeError::DeviceError(msg) => write!(f, "Device error: {}", msg),
            ComputeError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            ComputeError::InitializationError(msg) => write!(f, "Initialization error: {}", msg),
        }
    }
}

impl std::error::Error for ComputeError {}

/// Compute module configuration
#[derive(Debug, Clone)]
pub struct ComputeConfig {
    pub prefer_gpu: bool,
    pub max_workers: Option<u32>,
    pub max_threads: Option<u32>,
    pub enable_simd: bool,
    pub power_preference: PowerPreference,
    pub enable_webnn: bool,
    pub webnn_device_type: WebNNDeviceType,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            prefer_gpu: true,
            max_workers: None,
            max_threads: None,
            enable_simd: true,
            power_preference: PowerPreference::HighPerformance,
            enable_webnn: true,
            webnn_device_type: WebNNDeviceType::Auto,
        }
    }
}

/// Power preference for compute operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerPreference {
    LowPower,
    HighPerformance,
}

/// Compute capabilities detection
#[derive(Debug, Clone)]
pub struct ComputeCapabilities {
    pub has_webgpu: bool,
    pub has_webgl: bool,
    pub has_web_workers: bool,
    pub has_shared_array_buffer: bool,
    pub has_simd: bool,
    pub max_texture_size: Option<u32>,
    pub max_compute_workgroup_size: Option<[u32; 3]>,
    pub max_worker_count: u32,
    pub supports_f16: bool,
    pub has_webnn: bool,
    pub webnn_supports_npu: bool,
}

impl ComputeCapabilities {
    /// Detect current environment compute capabilities
    pub async fn detect() -> Self {
        let mut capabilities = Self {
            has_webgpu: false,
            has_webgl: false,
            has_web_workers: false,
            has_shared_array_buffer: false,
            has_simd: false,
            max_texture_size: None,
            max_compute_workgroup_size: None,
            max_worker_count: 0,
            supports_f16: false,
            has_webnn: false,
            webnn_supports_npu: false,
        };

        // Detect WebGPU
        #[cfg(feature = "webgpu")]
        {
            capabilities.has_webgpu = webgpu::is_webgpu_supported().await;
            if capabilities.has_webgpu {
                if let Ok(device_caps) = webgpu::get_device_capabilities().await {
                    let size = device_caps.max_compute_workgroup_size;
                    capabilities.max_compute_workgroup_size = Some([size, size, size]);
                    capabilities.supports_f16 = device_caps.supports_f16;
                }
            }
        }

        // Detect WebGL
        capabilities.has_webgl = Self::detect_webgl();

        // Detect Web Workers
        #[cfg(feature = "web-workers")]
        {
            capabilities.has_web_workers = web_workers::is_web_workers_supported();
            capabilities.max_worker_count = web_workers::get_optimal_worker_count() as u32;
        }

        // Detect SharedArrayBuffer
        #[cfg(feature = "shared-memory")]
        {
            capabilities.has_shared_array_buffer = threads::is_threading_supported();
        }

        // Detect SIMD
        capabilities.has_simd = Self::detect_simd();

        // Detect WebNN
        capabilities.has_webnn = WebNNContext::is_available();
        if capabilities.has_webnn {
            let context = WebNNContext::new(WebNNDeviceType::Auto, WebNNPowerPreference::Default);
            capabilities.webnn_supports_npu = context.capabilities().has_npu();
        }

        capabilities
    }

    fn detect_webgl() -> bool {
        let window = match web_sys::window() {
            Some(w) => w,
            None => return false,
        };
        let document = match window.document() {
            Some(d) => d,
            None => return false,
        };
        let canvas_element = match document.create_element("canvas") {
            Ok(e) => e,
            Err(_) => return false,
        };
        let canvas = match canvas_element.dyn_into::<web_sys::HtmlCanvasElement>() {
            Ok(c) => c,
            Err(_) => return false,
        };

        canvas.get_context("webgl").unwrap_or(None).is_some()
            || canvas.get_context("webgl2").unwrap_or(None).is_some()
    }

    fn detect_simd() -> bool {
        // Check if SIMD is available
        cfg!(target_feature = "simd128")
    }
}

/// Automatic compute backend selection
pub struct ComputeManager {
    config: ComputeConfig,
    capabilities: ComputeCapabilities,
    #[cfg(feature = "webgpu")]
    webgpu_device: Option<webgpu::DeviceSelector>,
    #[cfg(feature = "web-workers")]
    worker_pool: Option<web_workers::WorkerPool>,
}

impl ComputeManager {
    /// Create a new compute manager
    pub async fn new(config: ComputeConfig) -> Result<Self, ComputeError> {
        let capabilities = ComputeCapabilities::detect().await;

        let mut manager = Self {
            config,
            capabilities,
            #[cfg(feature = "webgpu")]
            webgpu_device: None,
            #[cfg(feature = "web-workers")]
            worker_pool: None,
        };

        manager.initialize_backends().await?;
        Ok(manager)
    }

    /// Initialize compute backends based on capabilities
    async fn initialize_backends(&mut self) -> Result<(), ComputeError> {
        #[cfg(feature = "webgpu")]
        if self.capabilities.has_webgpu && self.config.prefer_gpu {
            match webgpu::DeviceSelector::new().await {
                Ok(mut device) => {
                    device
                        .initialize_device()
                        .await
                        .map_err(|e| ComputeError::WebGpuError(format!("{:?}", e)))?;
                    self.webgpu_device = Some(device);
                    web_sys::console::log_1(&"WebGPU backend initialized".into());
                },
                Err(e) => {
                    web_sys::console::warn_1(
                        &format!("Failed to initialize WebGPU: {:?}", e).into(),
                    );
                },
            }
        }

        #[cfg(feature = "web-workers")]
        if self.capabilities.has_web_workers {
            let worker_count = self
                .config
                .max_workers
                .unwrap_or(self.capabilities.max_worker_count)
                .min(self.capabilities.max_worker_count);

            // Default worker script URL - should be provided by the application
            let worker_script_url = "trustformers_worker.js".to_string();

            match web_workers::WorkerPool::new(worker_count as usize, worker_script_url) {
                Ok(pool) => {
                    self.worker_pool = Some(pool);
                    web_sys::console::log_1(
                        &format!("Worker pool initialized with {} workers", worker_count).into(),
                    );
                },
                Err(e) => {
                    web_sys::console::warn_1(
                        &format!("Failed to initialize worker pool: {:?}", e).into(),
                    );
                },
            }
        }

        Ok(())
    }

    /// Get the best compute backend for a given operation
    pub fn get_optimal_backend(
        &self,
        operation_size: usize,
        _operation_type: &str,
    ) -> ComputeBackend {
        // Simple heuristic for backend selection
        if self.capabilities.has_webgpu && self.config.prefer_gpu && operation_size > 1024 {
            ComputeBackend::WebGpu
        } else if self.capabilities.has_web_workers && operation_size > 10000 {
            ComputeBackend::WebWorkers
        } else if self.capabilities.has_simd && self.config.enable_simd {
            ComputeBackend::Simd
        } else {
            ComputeBackend::Cpu
        }
    }

    /// Get current capabilities
    pub fn capabilities(&self) -> &ComputeCapabilities {
        &self.capabilities
    }

    /// Get current configuration
    pub fn config(&self) -> &ComputeConfig {
        &self.config
    }
}

/// Available compute backends
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    Cpu,
    Simd,
    WebGpu,
    WebGl,
    WebWorkers,
    Threads,
    WebNN,
}

impl core::fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            ComputeBackend::Cpu => "CPU",
            ComputeBackend::Simd => "SIMD",
            ComputeBackend::WebGpu => "WebGPU",
            ComputeBackend::WebGl => "WebGL",
            ComputeBackend::WebWorkers => "Web Workers",
            ComputeBackend::Threads => "Threads",
            ComputeBackend::WebNN => "WebNN",
        };
        write!(f, "{}", name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_config_default() {
        let config = ComputeConfig::default();
        assert!(config.prefer_gpu);
        assert!(config.enable_simd);
        assert_eq!(config.power_preference, PowerPreference::HighPerformance);
    }

    #[test]
    fn test_compute_backend_display() {
        assert_eq!(format!("{}", ComputeBackend::Cpu), "CPU");
        assert_eq!(format!("{}", ComputeBackend::WebGpu), "WebGPU");
        assert_eq!(format!("{}", ComputeBackend::WebWorkers), "Web Workers");
    }
}
