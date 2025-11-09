//! Android Platform Support for TrustformeRS
//!
//! This module provides comprehensive Android platform integration including
//! NNAPI acceleration, GPU compute, device detection, and JNI bindings.

pub mod types;
pub mod device;
pub mod nnapi;
pub mod gpu;
pub mod jni;
pub mod engine;

// Re-export main types and interfaces for convenience
pub use types::*;
pub use device::AndroidDeviceInfo;
pub use engine::AndroidInferenceEngine;

// Re-export NNAPI types
pub use nnapi::{NNAPIDeviceManager, NNAPIExecutor, NNAPIModelBuilder};

// Re-export GPU types
pub use gpu::{VulkanComputeContext, OpenGLESComputeContext};

// JNI functions are exported directly from the jni module
pub use jni::utils;

/// Initialize Android platform support
pub fn initialize() -> crate::Result<()> {
    tracing::info!("Initializing Android platform support");

    // Check device capabilities
    let device_info = AndroidDeviceInfo::detect();
    tracing::info!("Device: {} {}", device_info.manufacturer, device_info.model);
    tracing::info!("Android {} (API {})", device_info.android_version, device_info.api_level);
    tracing::info!("Memory: {}MB, CPU cores: {}", device_info.total_memory_mb, device_info.cpu_cores);

    // Log hardware acceleration availability
    if device_info.nnapi_info.is_some() {
        tracing::info!("NNAPI hardware acceleration available");
    }
    if device_info.gpu_info.vulkan_supported {
        tracing::info!("Vulkan GPU acceleration supported");
    }
    if device_info.gpu_info.opengl_es_version >= "3.1" {
        tracing::info!("OpenGL ES compute shaders supported");
    }

    Ok(())
}

/// Check if the current device supports hardware acceleration
pub fn supports_hardware_acceleration() -> bool {
    AndroidInferenceEngine::has_hardware_acceleration()
}

/// Get recommended configuration for the current device
pub fn get_recommended_config() -> crate::MobileConfig {
    let device_info = AndroidDeviceInfo::detect();
    device_info.get_recommended_config()
}