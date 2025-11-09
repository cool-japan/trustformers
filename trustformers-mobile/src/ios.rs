//! iOS Platform Support for TrustformeRS
//!
//! This module provides iOS-specific functionality including Core ML integration,
//! iOS framework bindings, and platform-specific optimizations.

use crate::{MobileBackend, MobileConfig, MobilePlatform, MobileStats};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};
use std::ptr;
use std::slice;
use trustformers_core::error::{CoreError, Result};
use trustformers_core::Tensor;

#[cfg(target_os = "ios")]
use core_foundation::{
    base::{CFRelease, CFTypeRef},
    string::{CFString, CFStringRef},
};
#[cfg(target_os = "ios")]
use objc::runtime::{Class, Object};

// Import submodules
pub mod engine;
pub mod metal;
pub mod mps;

// Re-export key types from submodules
pub use engine::{
    iOSInferenceEngine, LoadDistributionStrategy, MetalComputeState, MultiGPUManager,
};
pub use metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLFunction, MTLLibrary, MTLOrigin, MTLRegion, MTLSize,
};
pub use mps::{
    MPSDataType, MPSGraph, MPSGraphConvolution2DOpDescriptor,
    MPSGraphDepthwiseConvolution2DOpDescriptor, MPSGraphDevice, MPSGraphExecutable,
    MPSGraphExecutionDescriptor, MPSGraphMatrixMultiplicationDescriptor,
    MPSGraphPooling2DOpDescriptor, MPSGraphTensor, MPSGraphTensorData, MPSShape,
};

/// iOS device information
#[derive(Debug, Clone)]
pub struct IOsDeviceInfo {
    pub device_name: String,
    pub system_name: String,
    pub system_version: String,
    pub model: String,
    pub localized_model: String,
    pub identifer_for_vendor: Option<String>,
    pub is_simulator: bool,
    pub total_memory_gb: f64,
    pub available_memory_gb: f64,
    pub battery_level: f32,
    pub thermal_state: ThermalState,
    pub processor_count: usize,
    pub cpu_type: String,
    pub gpu_family: u32,
    pub max_threads_per_group: usize,
    pub supports_neural_engine: bool,
}

/// iOS thermal states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    Nominal = 0,
    Fair = 1,
    Serious = 2,
    Critical = 3,
}

impl IOsDeviceInfo {
    /// Get device information on iOS
    #[cfg(target_os = "ios")]
    pub fn detect() -> Result<Self> {
        // This is a simplified version - real implementation would use iOS APIs
        Ok(Self {
            device_name: "iPhone".to_string(),
            system_name: "iOS".to_string(),
            system_version: "17.0".to_string(),
            model: "iPhone15,2".to_string(),
            localized_model: "iPhone".to_string(),
            identifer_for_vendor: None,
            is_simulator: false,
            total_memory_gb: 6.0,
            available_memory_gb: 4.5,
            battery_level: 0.85,
            thermal_state: ThermalState::Nominal,
            processor_count: 6,
            cpu_type: "A16 Bionic".to_string(),
            gpu_family: metal::MTL_GPU_FAMILY_APPLE_7,
            max_threads_per_group: 1024,
            supports_neural_engine: true,
        })
    }

    /// Get device information on non-iOS platforms (stub)
    #[cfg(not(target_os = "ios"))]
    pub fn detect() -> Result<Self> {
        // Stub implementation for non-iOS platforms
        Ok(Self {
            device_name: "iOS Device".to_string(),
            system_name: "iOS".to_string(),
            system_version: "17.0".to_string(),
            model: "iPhone".to_string(),
            localized_model: "iPhone".to_string(),
            identifer_for_vendor: None,
            is_simulator: true,
            total_memory_gb: 4.0,
            available_memory_gb: 3.0,
            battery_level: 0.5,
            thermal_state: ThermalState::Nominal,
            processor_count: 4,
            cpu_type: "Simulated".to_string(),
            gpu_family: metal::MTL_GPU_FAMILY_APPLE_5,
            max_threads_per_group: 512,
            supports_neural_engine: false,
        })
    }

    /// Get device performance tier
    pub fn performance_tier(&self) -> crate::device_info::PerformanceTier {
        if self.gpu_family >= metal::MTL_GPU_FAMILY_APPLE_7 {
            crate::device_info::PerformanceTier::Flagship
        } else if self.gpu_family >= metal::MTL_GPU_FAMILY_APPLE_5 {
            crate::device_info::PerformanceTier::HighEnd
        } else if self.gpu_family >= metal::MTL_GPU_FAMILY_APPLE_3 {
            crate::device_info::PerformanceTier::MidRange
        } else {
            crate::device_info::PerformanceTier::Budget
        }
    }

    /// Check if device supports feature
    pub fn supports_feature(&self, feature: &iOSFeature) -> bool {
        match feature {
            iOSFeature::CoreML => true,
            iOSFeature::MetalPerformanceShaders => true,
            iOSFeature::NeuralEngine => self.supports_neural_engine,
            iOSFeature::ARKit => self.gpu_family >= metal::MTL_GPU_FAMILY_APPLE_3,
            iOSFeature::CreateML => self.gpu_family >= metal::MTL_GPU_FAMILY_APPLE_5,
            iOSFeature::VisionFramework => true,
            iOSFeature::SpeechFramework => true,
            iOSFeature::NaturalLanguage => true,
        }
    }
}

/// iOS-specific features
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum iOSFeature {
    CoreML,
    MetalPerformanceShaders,
    NeuralEngine,
    ARKit,
    CreateML,
    VisionFramework,
    SpeechFramework,
    NaturalLanguage,
}

/// C API structures for FFI
#[repr(C)]
pub struct CInferenceResult {
    pub success: bool,
    pub output_data: *mut c_float,
    pub output_size: usize,
    pub inference_time_ms: f32,
    pub error_message: *mut c_char,
}

#[repr(C)]
pub struct CTrustformersConfig {
    pub use_coreml: bool,
    pub use_nnapi: bool,
    pub use_gpu: bool,
    pub use_fp16: bool,
    pub num_threads: c_int,
    pub max_memory_mb: c_int,
}

#[repr(C)]
pub struct CTrustformersInferenceResult {
    pub data: *mut f32,
    pub size: usize,
    pub success: bool,
}

#[repr(C)]
pub struct CTrustformersTensor {
    pub data: *mut c_void,
    pub shape: *mut usize,
    pub ndim: usize,
    pub dtype: c_int,
}

/// C API functions
#[no_mangle]
pub unsafe extern "C" fn trustformers_ios_inference_engine_new(
    config: *const CTrustformersConfig,
) -> *mut iOSInferenceEngine {
    if config.is_null() {
        return ptr::null_mut();
    }

    let config_ref = &*config;
    let mobile_config = MobileConfig {
        platform: MobilePlatform::Ios,
        backend: if config_ref.use_coreml {
            MobileBackend::CoreML
        } else if config_ref.use_gpu {
            MobileBackend::Metal
        } else {
            MobileBackend::CPU
        },
        memory_optimization: crate::MemoryOptimization::Balanced,
        max_memory_mb: config_ref.max_memory_mb as usize,
        use_fp16: config_ref.use_fp16,
        quantization: None,
        num_threads: config_ref.num_threads as usize,
        enable_batching: false,
        max_batch_size: 1,
    };

    match iOSInferenceEngine::new(mobile_config) {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn trustformers_ios_inference_engine_free(engine: *mut iOSInferenceEngine) {
    if !engine.is_null() {
        let _ = Box::from_raw(engine);
    }
}

#[no_mangle]
pub unsafe extern "C" fn trustformers_ios_load_model(
    engine: *mut iOSInferenceEngine,
    model_path: *const c_char,
) -> bool {
    if engine.is_null() || model_path.is_null() {
        return false;
    }

    let engine_ref = &mut *engine;
    let path_cstr = CStr::from_ptr(model_path);

    match path_cstr.to_str() {
        Ok(path_str) => engine_ref.load_model_from_path(path_str).is_ok(),
        Err(_) => false,
    }
}

#[no_mangle]
pub unsafe extern "C" fn trustformers_ios_inference(
    engine: *mut iOSInferenceEngine,
    input_data: *const f32,
    input_size: usize,
) -> CInferenceResult {
    let mut result = CInferenceResult {
        success: false,
        output_data: ptr::null_mut(),
        output_size: 0,
        inference_time_ms: 0.0,
        error_message: ptr::null_mut(),
    };

    if engine.is_null() || input_data.is_null() || input_size == 0 {
        let error_msg = CString::new("Invalid input parameters").unwrap_or_default();
        result.error_message = error_msg.into_raw();
        return result;
    }

    let engine_ref = &mut *engine;
    let input_slice = slice::from_raw_parts(input_data, input_size);

    // Convert input to tensor format - simplified for C API
    let shape = vec![1, input_size]; // Assume batch size 1
    let tensor_result = Tensor::from_vec(input_slice.to_vec(), &shape);

    match tensor_result {
        Ok(input_tensor) => {
            let start_time = std::time::Instant::now();
            match engine_ref.inference(&input_tensor) {
                Ok(output_tensor) => {
                    let inference_time = start_time.elapsed().as_secs_f32() * 1000.0;

                    match output_tensor.data() {
                        Ok(output_data_vec) => {
                            let output_size = output_data_vec.len();
                            let output_ptr = output_data_vec.as_ptr() as *mut f32;
                            std::mem::forget(output_data_vec); // Prevent deallocation

                            result.success = true;
                            result.output_data = output_ptr;
                            result.output_size = output_size;
                            result.inference_time_ms = inference_time;
                        },
                        Err(_) => {
                            let error_msg =
                                CString::new("Failed to extract output data").unwrap_or_default();
                            result.error_message = error_msg.into_raw();
                        },
                    }
                },
                Err(_) => {
                    let error_msg = CString::new("Inference failed").unwrap_or_default();
                    result.error_message = error_msg.into_raw();
                },
            }
        },
        Err(_) => {
            let error_msg = CString::new("Failed to create input tensor").unwrap_or_default();
            result.error_message = error_msg.into_raw();
        },
    }

    result
}

#[no_mangle]
pub unsafe extern "C" fn trustformers_ios_free_inference_result(result: *mut CInferenceResult) {
    if result.is_null() {
        return;
    }

    let result_ref = &*result;

    if !result_ref.output_data.is_null() {
        let _ = Vec::from_raw_parts(
            result_ref.output_data,
            result_ref.output_size,
            result_ref.output_size,
        );
    }

    if !result_ref.error_message.is_null() {
        let _ = CString::from_raw(result_ref.error_message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ios_device_info_detection() {
        let device_info = IOsDeviceInfo::detect();
        assert!(device_info.is_ok());

        let info = device_info.unwrap();
        assert!(!info.device_name.is_empty());
        assert!(!info.system_name.is_empty());
        assert!(info.processor_count > 0);
    }

    #[test]
    fn test_performance_tier_classification() {
        let mut device_info = IOsDeviceInfo::detect().unwrap();

        // Test flagship tier
        device_info.gpu_family = metal::MTL_GPU_FAMILY_APPLE_7;
        assert_eq!(
            device_info.performance_tier(),
            crate::device_info::PerformanceTier::Flagship
        );

        // Test high-end tier
        device_info.gpu_family = metal::MTL_GPU_FAMILY_APPLE_5;
        assert_eq!(
            device_info.performance_tier(),
            crate::device_info::PerformanceTier::HighEnd
        );
    }

    #[test]
    fn test_ios_feature_support() {
        let device_info = IOsDeviceInfo::detect().unwrap();

        // Core ML should be supported on all iOS devices
        assert!(device_info.supports_feature(&iOSFeature::CoreML));
        assert!(device_info.supports_feature(&iOSFeature::MetalPerformanceShaders));
    }

    #[test]
    fn test_thermal_state_variants() {
        // Test all thermal state variants
        let states = [
            ThermalState::Nominal,
            ThermalState::Fair,
            ThermalState::Serious,
            ThermalState::Critical,
        ];

        for (i, state) in states.iter().enumerate() {
            assert_eq!(*state as usize, i);
        }
    }

    #[test]
    fn test_c_api_structures() {
        // Test C API structure creation
        let config = CTrustformersConfig {
            use_coreml: true,
            use_nnapi: false,
            use_gpu: true,
            use_fp16: true,
            num_threads: 4,
            max_memory_mb: 512,
        };

        assert!(config.use_coreml);
        assert!(!config.use_nnapi);
        assert_eq!(config.num_threads, 4);
    }

    #[test]
    fn test_load_distribution_strategies() {
        let strategies = [
            LoadDistributionStrategy::RoundRobin,
            LoadDistributionStrategy::PerformanceBased,
            LoadDistributionStrategy::MemoryBased,
            LoadDistributionStrategy::Adaptive,
        ];

        for strategy in strategies {
            // Test that strategies can be cloned and debugged
            let cloned = strategy.clone();
            let debug_str = format!("{:?}", cloned);
            assert!(!debug_str.is_empty());
        }
    }
}
