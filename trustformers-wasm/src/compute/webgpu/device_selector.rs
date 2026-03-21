//! Automatic device selection for optimal performance
//!
//! This module provides intelligent CPU/GPU selection based on:
//! - WebGPU availability and capabilities
//! - Device performance characteristics
//! - Memory constraints
//! - Model size and complexity

use super::types::{Gpu, GpuAdapter, GpuAdapterExt, GpuDevice, GpuExt};
use std::format;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{console, Navigator};

/// Device types available for computation
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceType {
    /// CPU-only execution using SIMD where available
    CPU,
    /// GPU execution via WebGPU
    GPU,
    /// Hybrid execution (CPU for small ops, GPU for large)
    Hybrid,
}

/// Device capabilities and performance metrics
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub webgpu_available: bool,
    pub gpu_memory_limit: u64,
    pub max_compute_workgroup_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub simd_support: bool,
    pub performance_tier: u32,
    pub is_mobile: bool,
    pub supports_f16: bool,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            webgpu_available: false,
            gpu_memory_limit: 0,
            max_compute_workgroup_size: 256,
            max_compute_invocations_per_workgroup: 256,
            simd_support: false,
            performance_tier: 0,
            is_mobile: false,
            supports_f16: false,
        }
    }
}

#[wasm_bindgen]
impl DeviceCapabilities {
    #[wasm_bindgen(getter)]
    pub fn webgpu_available(&self) -> bool {
        self.webgpu_available
    }

    #[wasm_bindgen(getter)]
    pub fn gpu_memory_limit(&self) -> f64 {
        self.gpu_memory_limit as f64
    }

    #[wasm_bindgen(getter)]
    pub fn max_compute_workgroup_size(&self) -> u32 {
        self.max_compute_workgroup_size
    }

    #[wasm_bindgen(getter)]
    pub fn simd_support(&self) -> bool {
        self.simd_support
    }

    #[wasm_bindgen(getter)]
    pub fn performance_tier(&self) -> u32 {
        self.performance_tier
    }

    #[wasm_bindgen(getter)]
    pub fn is_mobile(&self) -> bool {
        self.is_mobile
    }
}

/// Intelligent device selector
#[wasm_bindgen]
pub struct DeviceSelector {
    capabilities: DeviceCapabilities,
    selected_device: DeviceType,
    adapter: Option<GpuAdapter>,
    device: Option<GpuDevice>,
}

#[wasm_bindgen]
impl DeviceSelector {
    /// Create a new device selector and analyze available capabilities
    pub async fn new() -> Result<DeviceSelector, JsValue> {
        let capabilities = Self::analyze_device_capabilities().await?;
        let selected_device = Self::select_optimal_device(&capabilities);

        console::log_1(
            &format!(
                "Device selector initialized: WebGPU={}, SIMD={}, Selected={:?}",
                capabilities.webgpu_available, capabilities.simd_support, selected_device
            )
            .into(),
        );

        Ok(DeviceSelector {
            capabilities,
            selected_device,
            adapter: None,
            device: None,
        })
    }

    /// Get device capabilities
    #[wasm_bindgen(getter)]
    pub fn capabilities(&self) -> DeviceCapabilities {
        self.capabilities.clone()
    }

    /// Get selected device type
    #[wasm_bindgen(getter)]
    pub fn selected_device(&self) -> DeviceType {
        self.selected_device
    }

    /// Initialize the selected device for computation
    pub async fn initialize_device(&mut self) -> Result<(), JsValue> {
        match self.selected_device {
            DeviceType::GPU | DeviceType::Hybrid => {
                self.initialize_gpu().await?;
            },
            DeviceType::CPU => {
                // CPU initialization is minimal
                console::log_1(&"CPU device initialized".into());
            },
        }
        Ok(())
    }

    /// Check if GPU should be used for given operation
    pub fn should_use_gpu(&self, operation_size: usize, complexity: f32) -> bool {
        match self.selected_device {
            DeviceType::CPU => false,
            DeviceType::GPU => true,
            DeviceType::Hybrid => {
                // Use GPU for larger operations or complex computations
                let size_threshold = if self.capabilities.is_mobile { 1024 } else { 512 };
                let complexity_threshold = 0.5;

                operation_size > size_threshold || complexity > complexity_threshold
            },
        }
    }

    /// Get the GPU device if available
    pub fn get_gpu_device(&self) -> Option<GpuDevice> {
        self.device.clone()
    }
}

// Private implementation methods
impl DeviceSelector {
    /// Analyze device capabilities and performance characteristics
    async fn analyze_device_capabilities() -> Result<DeviceCapabilities, JsValue> {
        // Check WebGPU availability
        let webgpu_available = Self::check_webgpu_available();

        // Check SIMD support
        let simd_support = cfg!(target_feature = "simd128");

        // Detect if running on mobile device
        let is_mobile = Self::detect_mobile_device();

        // Initialize default capabilities
        let mut capabilities = DeviceCapabilities {
            webgpu_available,
            gpu_memory_limit: 0,
            max_compute_workgroup_size: 0,
            max_compute_invocations_per_workgroup: 0,
            simd_support,
            performance_tier: if is_mobile { 1 } else { 2 },
            is_mobile,
            supports_f16: false,
        };

        // If WebGPU is available, get detailed GPU capabilities
        if webgpu_available {
            if let Ok(gpu_caps) = Self::get_gpu_capabilities().await {
                capabilities.gpu_memory_limit = gpu_caps.0;
                capabilities.max_compute_workgroup_size = gpu_caps.1;
                capabilities.max_compute_invocations_per_workgroup = gpu_caps.2;
                capabilities.performance_tier = Self::calculate_performance_tier(&capabilities);
            }
        }

        Ok(capabilities)
    }

    /// Check if WebGPU is available in the current environment
    fn check_webgpu_available() -> bool {
        web_sys::window()
            .and_then(|w| {
                js_sys::Reflect::get(&w.navigator(), &JsValue::from_str("gpu"))
                    .ok()
                    .filter(|v| !v.is_undefined() && !v.is_null())
            })
            .is_some()
    }

    /// Detect if running on a mobile device
    #[allow(clippy::excessive_nesting)]
    fn detect_mobile_device() -> bool {
        if let Some(window) = web_sys::window() {
            if let Ok(navigator) = js_sys::Reflect::get(&window, &JsValue::from_str("navigator")) {
                if let Ok(user_agent) =
                    js_sys::Reflect::get(&navigator, &JsValue::from_str("userAgent"))
                {
                    if let Some(ua_string) = user_agent.as_string() {
                        let ua_lower = ua_string.to_lowercase();
                        return ua_lower.contains("mobile")
                            || ua_lower.contains("android")
                            || ua_lower.contains("iphone")
                            || ua_lower.contains("ipad");
                    }
                }
            }
        }
        false
    }

    /// Get detailed GPU capabilities
    async fn get_gpu_capabilities() -> Result<(u64, u32, u32), JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let navigator: Navigator = window.navigator();

        let gpu_val = js_sys::Reflect::get(&navigator, &JsValue::from_str("gpu"))?;
        let gpu: Gpu = gpu_val.dyn_into()?;

        let adapter_promise = gpu.request_adapter();
        let adapter_result = JsFuture::from(adapter_promise).await?;

        if adapter_result.is_null() {
            return Err("No GPU adapter available".into());
        }

        let adapter: GpuAdapter = adapter_result.dyn_into()?;

        // Request limits information
        let limits = adapter.limits();

        let max_buffer_size = js_sys::Reflect::get(&limits, &JsValue::from_str("maxBufferSize"))
            .unwrap_or(JsValue::from(268435456u32)) // 256MB default
            .as_f64()
            .unwrap_or(268435456.0) as u64;

        let max_compute_workgroup_size_x =
            js_sys::Reflect::get(&limits, &JsValue::from_str("maxComputeWorkgroupSizeX"))
                .unwrap_or(JsValue::from(256u32))
                .as_f64()
                .unwrap_or(256.0) as u32;

        let max_compute_invocations_per_workgroup = js_sys::Reflect::get(
            &limits,
            &JsValue::from_str("maxComputeInvocationsPerWorkgroup"),
        )
        .unwrap_or(JsValue::from(256u32))
        .as_f64()
        .unwrap_or(256.0) as u32;

        Ok((
            max_buffer_size,
            max_compute_workgroup_size_x,
            max_compute_invocations_per_workgroup,
        ))
    }

    /// Calculate performance tier based on capabilities
    fn calculate_performance_tier(caps: &DeviceCapabilities) -> u32 {
        let mut tier = if caps.is_mobile { 1 } else { 2 };

        // Boost tier based on GPU memory
        if caps.gpu_memory_limit > 1_000_000_000 {
            // > 1GB
            tier += 1;
        }

        // Boost tier based on compute capabilities
        if caps.max_compute_workgroup_size > 512 {
            tier += 1;
        }

        tier.min(4) // Cap at tier 4
    }

    /// Select optimal device based on capabilities
    fn select_optimal_device(caps: &DeviceCapabilities) -> DeviceType {
        if !caps.webgpu_available {
            return DeviceType::CPU;
        }

        // On mobile devices with limited capabilities, prefer hybrid approach
        if caps.is_mobile && caps.performance_tier < 3 {
            return DeviceType::Hybrid;
        }

        // For high-performance devices, prefer GPU
        if caps.performance_tier >= 3 && caps.gpu_memory_limit > 500_000_000 {
            return DeviceType::GPU;
        }

        // Default to hybrid for balanced performance
        DeviceType::Hybrid
    }

    /// Initialize GPU device and adapter
    async fn initialize_gpu(&mut self) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("No window object")?;
        let navigator: Navigator = window.navigator();

        let gpu_val = js_sys::Reflect::get(&navigator, &JsValue::from_str("gpu"))?;
        let gpu: Gpu = gpu_val.dyn_into()?;

        // Request adapter
        let adapter_promise = gpu.request_adapter();
        let adapter_result = JsFuture::from(adapter_promise).await?;

        if adapter_result.is_null() {
            return Err("Failed to get GPU adapter".into());
        }

        let adapter: GpuAdapter = adapter_result.dyn_into()?;

        // Request device
        let device_promise = adapter.request_device();
        let device_result = JsFuture::from(device_promise).await?;
        let device: GpuDevice = device_result.dyn_into()?;

        self.adapter = Some(adapter);
        self.device = Some(device);

        console::log_1(
            &format!(
                "GPU device initialized successfully with {} memory limit",
                self.capabilities.gpu_memory_limit
            )
            .into(),
        );

        Ok(())
    }
}

/// Utility function to create device selector (for JavaScript)
#[wasm_bindgen]
pub async fn create_device_selector() -> Result<DeviceSelector, JsValue> {
    DeviceSelector::new().await
}

/// Check if current environment supports WebGPU
#[wasm_bindgen]
pub fn check_webgpu_support() -> bool {
    DeviceSelector::check_webgpu_available()
}
