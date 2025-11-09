//! Core Android Platform Types and Enumerations
//!
//! This module contains fundamental data structures for Android platform support,
//! including device information, capabilities, and feature detection.

use serde::{Deserialize, Serialize};
use std::os::raw::c_void;

/// Android GPU acceleration backend selection
#[cfg(target_os = "android")]
#[derive(Debug, Clone, Copy)]
pub enum AndroidGPUBackend {
    /// OpenGL ES compute shaders
    OpenGLES,
    /// Vulkan compute API
    Vulkan,
}

/// Android thermal management status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AndroidThermalStatus {
    /// No thermal throttling
    None,
    /// Light thermal throttling
    Light,
    /// Moderate thermal throttling
    Moderate,
    /// Severe thermal throttling
    Severe,
    /// Critical thermal throttling
    Critical,
    /// Emergency thermal throttling
    Emergency,
    /// System shutdown due to thermal
    Shutdown,
}

/// Android performance class (Android 12+)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AndroidPerformanceClass {
    /// Performance class R (Android 11 level)
    R,
    /// Performance class S (Android 12 level)
    S,
    /// Performance class T (Android 13 level)
    T,
    /// Performance class U (Android 14 level)
    U,
}

/// Android-specific hardware and software features
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AndroidFeature {
    /// Neural Networks API support
    NNAPI,
    /// Vulkan GPU acceleration
    VulkanGPU,
    /// OpenGL ES 3.0+ support
    OpenGLES3,
    /// FP16 inference support
    FP16Inference,
    /// INT8 quantization support
    Int8Quantization,
    /// On-device training capabilities
    OnDeviceTraining,
}

/// Comprehensive Android device information and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndroidDeviceInfo {
    /// Device manufacturer (e.g., "Samsung", "Google")
    pub manufacturer: String,
    /// Device model (e.g., "Pixel 6", "Galaxy S22")
    pub model: String,
    /// Android API level
    pub api_level: u32,
    /// Android version string
    pub android_version: String,
    /// Available memory in MB
    pub total_memory_mb: usize,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// CPU architecture (e.g., "arm64-v8a", "x86_64")
    pub cpu_architecture: String,
    /// GPU information and capabilities
    pub gpu_info: AndroidGPUInfo,
    /// NNAPI availability and configuration
    pub nnapi_info: Option<NNAPIInfo>,
    /// Current thermal throttling status
    pub thermal_status: AndroidThermalStatus,
    /// Performance class (if available, Android 12+)
    pub performance_class: Option<AndroidPerformanceClass>,
}

/// Android GPU hardware information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndroidGPUInfo {
    /// GPU vendor (e.g., "Qualcomm", "ARM", "Imagination")
    pub vendor: String,
    /// GPU renderer name
    pub renderer: String,
    /// OpenGL ES version supported
    pub opengl_es_version: String,
    /// Vulkan API support availability
    pub vulkan_supported: bool,
    /// GPU dedicated memory (if available)
    pub gpu_memory_mb: Option<usize>,
}

/// Neural Networks API information and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNAPIInfo {
    /// NNAPI feature level supported
    pub feature_level: u32,
    /// List of available acceleration devices
    pub available_devices: Vec<String>,
    /// Supported neural network operations
    pub supported_operations: Vec<String>,
    /// Hardware acceleration devices detected
    pub hardware_devices: Vec<NNAPIHardwareDevice>,
    /// Best recommended device type for inference
    pub best_device_type: String,
}

/// NNAPI hardware acceleration device (serializable)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNAPIHardwareDevice {
    /// Device name
    pub name: String,
    /// Device type classification
    pub device_type: String,
    /// NNAPI feature level supported
    pub feature_level: u32,
    /// Performance metrics
    pub exec_time: f32,
    pub power_usage: f32,
    /// Vendor-specific extensions supported
    pub vendor_extensions: Vec<String>,
}

/// NNAPI device information (internal use)
#[cfg(target_os = "android")]
#[derive(Debug, Clone)]
pub struct NNAPIDeviceInfo {
    /// Device index in NNAPI enumeration
    pub index: u32,
    /// Native device pointer
    pub device_ptr: *mut c_void,
    /// Human-readable device name
    pub name: String,
    /// NNAPI device type constant
    pub device_type: i32,
    /// NNAPI feature level
    pub feature_level: i32,
    /// Performance characteristics
    pub performance_info: ANeuralNetworksPerformanceInfo,
    /// Supported vendor extensions
    pub vendor_extensions: Vec<String>,
}

/// NNAPI performance information structure
#[cfg(target_os = "android")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ANeuralNetworksPerformanceInfo {
    /// Execution time estimate
    pub exec_time: f32,
    /// Power usage estimate
    pub power_usage: f32,
}

/// NNAPI model representation
#[cfg(target_os = "android")]
pub struct NNAPIModel {
    /// NNAPI model handle
    pub model: *mut super::nnapi::bindings::ANeuralNetworksModel,
    /// Compiled model handle
    pub compilation: *mut super::nnapi::bindings::ANeuralNetworksCompilation,
    /// Execution context handle
    pub execution: *mut super::nnapi::bindings::ANeuralNetworksExecution,
    /// Number of model inputs
    pub input_count: usize,
    /// Number of model outputs
    pub output_count: usize,
    /// Input operand indices
    pub input_operands: Vec<u32>,
    /// Output operand indices
    pub output_operands: Vec<u32>,
}

/// Android GPU compute state management
#[cfg(target_os = "android")]
pub struct AndroidGPUComputeState {
    /// Selected GPU backend
    pub backend: AndroidGPUBackend,

    // OpenGL ES state
    /// EGL display handle
    pub egl_display: Option<super::gpu::opengl_es::EGLDisplay>,
    /// EGL context handle
    pub egl_context: Option<super::gpu::opengl_es::EGLContext>,
    /// EGL surface handle
    pub egl_surface: Option<super::gpu::opengl_es::EGLSurface>,
    /// OpenGL compute program
    pub compute_program: Option<u32>,

    // Vulkan state
    /// Vulkan instance handle
    pub vk_instance: Option<super::gpu::vulkan::VkInstance>,
    /// Vulkan logical device
    pub vk_device: Option<super::gpu::vulkan::VkDevice>,
    /// Vulkan physical device
    pub vk_physical_device: Option<super::gpu::vulkan::VkPhysicalDevice>,
    /// Vulkan compute queue
    pub vk_queue: Option<super::gpu::vulkan::VkQueue>,
    /// Vulkan command buffer
    pub vk_command_buffer: Option<super::gpu::vulkan::VkCommandBuffer>,
    /// Vulkan Conv2D compute pipeline
    pub vk_conv2d_pipeline: Option<super::gpu::vulkan::VkPipeline>,
    /// Vulkan ReLU compute pipeline
    pub vk_relu_pipeline: Option<super::gpu::vulkan::VkPipeline>,
    /// Vulkan matrix multiplication pipeline
    pub vk_matmul_pipeline: Option<super::gpu::vulkan::VkPipeline>,
}

impl Default for AndroidThermalStatus {
    fn default() -> Self {
        Self::None
    }
}

impl Default for AndroidGPUInfo {
    fn default() -> Self {
        Self {
            vendor: "Unknown".to_string(),
            renderer: "Unknown".to_string(),
            opengl_es_version: "2.0".to_string(),
            vulkan_supported: false,
            gpu_memory_mb: None,
        }
    }
}

impl Default for AndroidDeviceInfo {
    fn default() -> Self {
        Self {
            manufacturer: "Unknown".to_string(),
            model: "Android Device".to_string(),
            api_level: 21, // Android 5.0 minimum
            android_version: "5.0".to_string(),
            total_memory_mb: 1024,
            cpu_cores: 2,
            cpu_architecture: "arm64-v8a".to_string(),
            gpu_info: AndroidGPUInfo::default(),
            nnapi_info: None,
            thermal_status: AndroidThermalStatus::None,
            performance_class: None,
        }
    }
}

#[cfg(target_os = "android")]
unsafe impl Send for NNAPIDeviceInfo {}
#[cfg(target_os = "android")]
unsafe impl Sync for NNAPIDeviceInfo {}

#[cfg(target_os = "android")]
unsafe impl Send for AndroidGPUComputeState {}
#[cfg(target_os = "android")]
unsafe impl Sync for AndroidGPUComputeState {}