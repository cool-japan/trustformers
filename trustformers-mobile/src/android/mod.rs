//! Android Platform Support for TrustformeRS
//!
//! This module provides comprehensive Android support including NNAPI integration,
//! JNI bindings, GPU acceleration, and device-specific optimizations.

pub mod device_info;
pub mod engine;
pub mod gpu;
pub mod jni;
pub mod nnapi;

// Re-export main public types
pub use device_info::{
    AndroidDeviceInfo, AndroidFeature, AndroidGPUInfo, AndroidPerformanceClass,
    AndroidThermalStatus, DevicePerformanceTier, NNAPIHardwareDevice, NNAPIInfo,
};
pub use engine::AndroidInferenceEngine;
pub use gpu::{AndroidGPUBackend, AndroidGPUComputeState};
pub use nnapi::{NNAPIDeviceInfo, NNAPIModel};

// Re-export JNI functions for external use
#[cfg(target_os = "android")]
pub use jni::{
    Java_com_trustformers_TrustformersEngine_createEngine,
    Java_com_trustformers_TrustformersEngine_getDeviceInfo,
    Java_com_trustformers_TrustformersEngine_getStats,
    Java_com_trustformers_TrustformersEngine_inference,
    Java_com_trustformers_TrustformersEngine_loadModel,
    Java_com_trustformers_TrustformersEngine_releaseEngine,
    Java_com_trustformers_TrustformersEngine_updateConfig,
};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::{MobileBackend, MobileConfig, MobilePlatform};
    use trustformers_core::Tensor;

    #[test]
    fn test_android_integration_flow() {
        // Test complete Android integration flow
        let config = MobileConfig::android_optimized();
        let mut engine = AndroidInferenceEngine::new(config).unwrap();

        // Check device capabilities
        let device_info = AndroidInferenceEngine::check_device_capabilities();
        assert!(!device_info.manufacturer.is_empty());

        // Load CPU model for testing
        assert!(engine.load_model("test_model.tflite").is_ok());
        assert!(engine.is_model_loaded());

        // Test inference
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input_tensor = Tensor::from_vec(input_data, &[4]).unwrap();

        let result = engine.inference(&input_tensor);
        assert!(result.is_ok());

        // Check stats
        let stats = engine.get_stats();
        assert!(stats.total_inferences > 0);
    }

    #[test]
    fn test_nnapi_device_detection() {
        // Test NNAPI device detection
        let devices = AndroidInferenceEngine::detect_nnapi_devices();

        // Should either find devices or return empty list gracefully
        // This test mainly ensures no panics occur during detection
        println!("Found {} NNAPI devices", devices.len());
    }

    #[test]
    fn test_android_feature_support() {
        let device_info = AndroidDeviceInfo::detect();

        // Test that feature detection works
        let int8_support = device_info.supports_feature(AndroidFeature::Int8Quantization);
        assert!(int8_support); // Should always be true

        let fp16_support = device_info.supports_feature(AndroidFeature::FP16Inference);
        println!("FP16 support: {}", fp16_support);

        let nnapi_support = device_info.supports_feature(AndroidFeature::NNAPI);
        println!("NNAPI support: {}", nnapi_support);
    }

    #[test]
    fn test_gpu_backend_creation() {
        // Test GPU backend creation
        let opengl_result = AndroidGPUComputeState::new(AndroidGPUBackend::OpenGLES);
        assert!(opengl_result.is_ok());

        let vulkan_result = AndroidGPUComputeState::new(AndroidGPUBackend::Vulkan);
        assert!(vulkan_result.is_ok());
    }

    #[test]
    fn test_recommended_config_generation() {
        let device_info = AndroidDeviceInfo::detect();
        let config = device_info.get_recommended_config();

        // Validate the recommended configuration
        assert_eq!(config.platform, MobilePlatform::Android);
        assert!(config.max_memory_mb >= 256);
        assert!(config.max_memory_mb <= 1024);
        assert!(config.thread_count > 0);
    }

    #[test]
    fn test_performance_tier_calculation() {
        let device_info = AndroidDeviceInfo::detect();
        let tier = device_info.get_performance_tier();

        // Should return a valid tier
        match tier {
            DevicePerformanceTier::Low
            | DevicePerformanceTier::Medium
            | DevicePerformanceTier::High
            | DevicePerformanceTier::Premium => {
                println!("Device performance tier: {:?}", tier);
            },
        }
    }

    #[test]
    fn test_multiple_backend_support() {
        // Test that different backends can be configured
        let mut cpu_config = MobileConfig::android_optimized();
        cpu_config.backend = MobileBackend::CPU;
        let cpu_engine = AndroidInferenceEngine::new(cpu_config);
        assert!(cpu_engine.is_ok());

        let mut nnapi_config = MobileConfig::android_optimized();
        nnapi_config.backend = MobileBackend::NNAPI;
        let nnapi_engine = AndroidInferenceEngine::new(nnapi_config);
        assert!(nnapi_engine.is_ok());

        let mut gpu_config = MobileConfig::android_optimized();
        gpu_config.backend = MobileBackend::GPU;
        let gpu_engine = AndroidInferenceEngine::new(gpu_config);
        assert!(gpu_engine.is_ok());
    }

    #[test]
    fn test_engine_lifecycle() {
        let config = MobileConfig::android_optimized();
        let mut engine = AndroidInferenceEngine::new(config).unwrap();

        // Test initial state
        assert!(!engine.is_model_loaded());
        assert_eq!(engine.get_stats().total_inferences, 0);

        // Load model
        assert!(engine.load_model("test_model.tflite").is_ok());
        assert!(engine.is_model_loaded());

        // Perform inference
        let input_data = vec![1.0; 100];
        let input_tensor = Tensor::from_vec(input_data, &[100]).unwrap();
        assert!(engine.inference(&input_tensor).is_ok());
        assert!(engine.get_stats().total_inferences > 0);

        // Update config
        let mut new_config = MobileConfig::android_optimized();
        new_config.max_memory_mb = 1024;
        assert!(engine.update_config(new_config).is_ok());
        assert_eq!(engine.get_config().max_memory_mb, 1024);

        // Reset stats
        engine.reset_stats();
        assert_eq!(engine.get_stats().total_inferences, 0);
    }
}
