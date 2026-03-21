//! Device capability struct definitions and implementations

use super::types::*;
use serde::{Deserialize, Serialize};
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecommendation {
    pub complexity: ModelComplexity,
    pub max_sequence_length: u32,
    pub batch_size: u32,
    pub precision: String,
    pub estimated_speed: InferenceSpeed,
    pub memory_usage_mb: f64,
    pub confidence_score: f64,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileCapabilities {
    pub is_low_power_mode: bool,
    pub(crate) screen_orientation: String,
    pub supports_orientation_lock: bool,
    pub supports_vibration: bool,
    pub supports_fullscreen: bool,
    pub supports_wake_lock: bool,
    pub supports_picture_in_picture: bool,
    pub(crate) thermal_state: String,
    pub network_save_data: bool,
    pub prefers_reduced_motion: bool,
    pub(crate) prefers_color_scheme: String,
    pub viewport_width: u32,
    pub viewport_height: u32,
    pub(crate) safe_area_insets: Vec<f64>,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmark {
    pub tensor_creation_ms: f64,
    pub matrix_multiply_ms: f64,
    pub webgl_draw_calls_per_second: f64,
    pub webgpu_compute_ms: f64,
    pub memory_allocation_ms: f64,
    pub overall_score: u32,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    pub cpu_cores: u32,
    pub memory_gb: f64,
    pub device_memory_estimate: f64,
    pub cpu_architecture: CPUArchitecture,
    pub gpu_vendor: GPUVendor,
    pub(crate) gpu_model: String,
    pub screen_width: u32,
    pub screen_height: u32,
    pub screen_density: f64,
    pub color_depth: u32,
    pub max_touch_points: u32,
    pub has_accelerometer: bool,
    pub has_gyroscope: bool,
    pub has_magnetometer: bool,
}

#[wasm_bindgen]
impl HardwareCapabilities {
    #[wasm_bindgen(getter)]
    pub fn cpu_cores(&self) -> u32 {
        self.cpu_cores
    }

    #[wasm_bindgen(getter)]
    pub fn memory_gb(&self) -> f64 {
        self.memory_gb
    }

    #[wasm_bindgen(getter)]
    pub fn device_memory_estimate(&self) -> f64 {
        self.device_memory_estimate
    }

    #[wasm_bindgen(getter)]
    pub fn cpu_architecture(&self) -> CPUArchitecture {
        self.cpu_architecture
    }

    #[wasm_bindgen(getter)]
    pub fn gpu_vendor(&self) -> GPUVendor {
        self.gpu_vendor
    }

    #[wasm_bindgen(getter)]
    pub fn gpu_model(&self) -> String {
        self.gpu_model.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn screen_width(&self) -> u32 {
        self.screen_width
    }

    #[wasm_bindgen(getter)]
    pub fn screen_height(&self) -> u32 {
        self.screen_height
    }

    #[wasm_bindgen(getter)]
    pub fn screen_density(&self) -> f64 {
        self.screen_density
    }

    #[wasm_bindgen(getter)]
    pub fn max_touch_points(&self) -> u32 {
        self.max_touch_points
    }

    #[wasm_bindgen(getter)]
    pub fn has_accelerometer(&self) -> bool {
        self.has_accelerometer
    }

    #[wasm_bindgen(getter)]
    pub fn has_gyroscope(&self) -> bool {
        self.has_gyroscope
    }

    #[wasm_bindgen(getter)]
    pub fn has_magnetometer(&self) -> bool {
        self.has_magnetometer
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftwareCapabilities {
    pub supports_webgpu: bool,
    pub supports_webgl2: bool,
    pub supports_webgl1: bool,
    pub supports_webassembly: bool,
    pub supports_wasm_simd: bool,
    pub supports_wasm_threads: bool,
    pub supports_wasm_bulk_memory: bool,
    pub supports_shared_array_buffer: bool,
    pub supports_service_workers: bool,
    pub supports_web_workers: bool,
    pub supports_offscreen_canvas: bool,
    pub supports_indexeddb: bool,
    pub supports_websockets: bool,
    pub supports_webrtc: bool,
    pub supports_geolocation: bool,
    pub supports_device_orientation: bool,
    pub supports_device_motion: bool,
    pub supports_bluetooth: bool,
    pub supports_usb: bool,
    pub supports_nfc: bool,
    pub supports_camera: bool,
    pub supports_microphone: bool,
    pub supports_speakers: bool,
    pub max_webgl_texture_size: u32,
    pub max_webgl_renderbuffer_size: u32,
    pub(crate) webgl_extensions: Vec<String>,
}

#[wasm_bindgen]
impl SoftwareCapabilities {
    #[wasm_bindgen(getter)]
    pub fn supports_webgpu(&self) -> bool {
        self.supports_webgpu
    }

    #[wasm_bindgen(getter)]
    pub fn supports_webgl2(&self) -> bool {
        self.supports_webgl2
    }

    #[wasm_bindgen(getter)]
    pub fn supports_webgl1(&self) -> bool {
        self.supports_webgl1
    }

    #[wasm_bindgen(getter)]
    pub fn supports_webassembly(&self) -> bool {
        self.supports_webassembly
    }

    #[wasm_bindgen(getter)]
    pub fn supports_wasm_simd(&self) -> bool {
        self.supports_wasm_simd
    }

    #[wasm_bindgen(getter)]
    pub fn supports_wasm_threads(&self) -> bool {
        self.supports_wasm_threads
    }

    #[wasm_bindgen(getter)]
    pub fn supports_shared_array_buffer(&self) -> bool {
        self.supports_shared_array_buffer
    }

    #[wasm_bindgen(getter)]
    pub fn supports_service_workers(&self) -> bool {
        self.supports_service_workers
    }

    #[wasm_bindgen(getter)]
    pub fn supports_offscreen_canvas(&self) -> bool {
        self.supports_offscreen_canvas
    }

    #[wasm_bindgen(getter)]
    pub fn supports_indexeddb(&self) -> bool {
        self.supports_indexeddb
    }

    #[wasm_bindgen(getter)]
    pub fn max_webgl_texture_size(&self) -> u32 {
        self.max_webgl_texture_size
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    pub device_type: DeviceType,
    pub operating_system: OperatingSystem,
    pub(crate) os_version: String,
    pub browser: Browser,
    pub(crate) browser_version: String,
    pub(crate) user_agent: String,
    pub is_mobile: bool,
    pub is_tablet: bool,
    pub is_desktop: bool,
    pub is_touch_device: bool,
    pub is_standalone_app: bool,
    pub is_webview: bool,
}

#[wasm_bindgen]
impl PlatformInfo {
    pub fn get_device_type(&self) -> DeviceType {
        self.device_type
    }

    pub fn get_operating_system(&self) -> OperatingSystem {
        self.operating_system
    }

    pub fn get_os_version(&self) -> String {
        self.os_version.clone()
    }

    pub fn get_browser(&self) -> Browser {
        self.browser
    }

    pub fn get_browser_version(&self) -> String {
        self.browser_version.clone()
    }

    pub fn get_user_agent(&self) -> String {
        self.user_agent.clone()
    }

    pub fn get_is_mobile(&self) -> bool {
        self.is_mobile
    }

    pub fn get_is_tablet(&self) -> bool {
        self.is_tablet
    }

    pub fn get_is_desktop(&self) -> bool {
        self.is_desktop
    }

    pub fn get_is_touch_device(&self) -> bool {
        self.is_touch_device
    }

    pub fn get_is_standalone_app(&self) -> bool {
        self.is_standalone_app
    }

    pub fn get_is_webview(&self) -> bool {
        self.is_webview
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub memory_used_mb: f64,
    pub memory_total_mb: f64,
    pub memory_limit_mb: f64,
    pub timing_navigation_start: f64,
    pub timing_dom_loading: f64,
    pub timing_dom_complete: f64,
    pub timing_load_event_end: f64,
    pub(crate) connection_type: String,
    pub connection_downlink: f64,
    pub connection_rtt: u32,
    pub battery_level: f64,
    pub battery_charging: bool,
}

#[wasm_bindgen]
impl PerformanceMetrics {
    #[wasm_bindgen(getter)]
    pub fn memory_used_mb(&self) -> f64 {
        self.memory_used_mb
    }

    #[wasm_bindgen(getter)]
    pub fn memory_total_mb(&self) -> f64 {
        self.memory_total_mb
    }

    #[wasm_bindgen(getter)]
    pub fn memory_limit_mb(&self) -> f64 {
        self.memory_limit_mb
    }

    #[wasm_bindgen(getter)]
    pub fn connection_type(&self) -> String {
        self.connection_type.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn connection_downlink(&self) -> f64 {
        self.connection_downlink
    }

    #[wasm_bindgen(getter)]
    pub fn connection_rtt(&self) -> u32 {
        self.connection_rtt
    }

    #[wasm_bindgen(getter)]
    pub fn battery_level(&self) -> f64 {
        self.battery_level
    }

    #[wasm_bindgen(getter)]
    pub fn battery_charging(&self) -> bool {
        self.battery_charging
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    pub hardware: HardwareCapabilities,
    pub software: SoftwareCapabilities,
    pub platform: PlatformInfo,
    pub performance: PerformanceMetrics,
    pub mobile_capabilities: Option<MobileCapabilities>,
    pub benchmark_results: Option<PerformanceBenchmark>,
    pub model_recommendations: Vec<ModelRecommendation>,
    pub detection_timestamp: f64,
    pub detection_version: String,
}

#[wasm_bindgen]
pub struct DeviceCapabilitiesWasm {
    inner: DeviceCapabilities,
}

#[wasm_bindgen]
impl DeviceCapabilitiesWasm {
    #[wasm_bindgen(getter)]
    pub fn detection_version(&self) -> String {
        self.inner.detection_version.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn detection_timestamp(&self) -> f64 {
        self.inner.detection_timestamp
    }

    #[wasm_bindgen(getter)]
    pub fn hardware(&self) -> HardwareCapabilities {
        self.inner.hardware.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn software(&self) -> SoftwareCapabilities {
        self.inner.software.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn platform(&self) -> PlatformInfo {
        self.inner.platform.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn performance(&self) -> PerformanceMetrics {
        self.inner.performance.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn mobile_capabilities(&self) -> Option<MobileCapabilities> {
        self.inner.mobile_capabilities.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn benchmark_results(&self) -> Option<PerformanceBenchmark> {
        self.inner.benchmark_results.clone()
    }
}

// Non-WASM impl for internal use
impl DeviceCapabilitiesWasm {
    pub fn new(inner: DeviceCapabilities) -> Self {
        Self { inner }
    }
}
