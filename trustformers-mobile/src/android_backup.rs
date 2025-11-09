//! Android Platform Support for TrustformeRS
//!
//! This module provides comprehensive Android platform integration including NNAPI hardware
//! acceleration, GPU compute with Vulkan and OpenGL ES, device detection and capabilities,
//! JNI bindings for Android applications, and intelligent performance optimization.
//!
//! ## Features
//!
//! - **NNAPI Hardware Acceleration**: Full Neural Networks API integration with device detection
//! - **GPU Compute**: Vulkan and OpenGL ES compute shader support for ML workloads
//! - **Device Detection**: Comprehensive Android device capabilities and feature detection
//! - **JNI Integration**: Complete Java Native Interface bindings for Android apps
//! - **Performance Optimization**: Intelligent backend selection and thermal management
//! - **Memory Management**: Advanced memory optimization for diverse Android devices
//!
//! ## Architecture
//!
//! The Android platform support is organized into focused modules:
//!
//! ```text
//! android_backup/
//! ├── types.rs           # Core types, enums, and device information
//! ├── device.rs          # Device detection and capabilities analysis
//! ├── nnapi/             # Neural Networks API integration
//! │   ├── bindings.rs    # C API bindings and constants
//! │   ├── model.rs       # Model management and building
//! │   └── execution.rs   # Execution and device management
//! ├── gpu/               # GPU acceleration support
//! │   ├── vulkan.rs      # Vulkan compute API integration
//! │   └── opengl_es.rs   # OpenGL ES compute shader support
//! ├── jni.rs             # JNI bindings for Android applications
//! └── engine.rs          # Main inference engine orchestrator
//! ```
//!
//! ## Usage Examples
//!
//! ### Basic Android Inference Engine
//!
//! ```rust
//! use trustformers_mobile::android_backup::*;
//! use trustformers_mobile::MobileConfig;
//! use trustformers_core::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Detect device capabilities
//! let device_info = AndroidDeviceInfo::detect();
//! println!("Device: {} {}", device_info.manufacturer, device_info.model);
//!
//! // Get optimized configuration for this device
//! let config = device_info.get_recommended_config();
//!
//! // Create inference engine
//! let mut engine = AndroidInferenceEngine::new(config)?;
//!
//! // Load model (supports NNAPI, CPU, and GPU backends)
//! engine.load_model("model.onnx")?;
//!
//! // Perform inference
//! let input = Tensor::ones(&[1, 224, 224, 3])?;
//! let output = engine.inference(&input)?;
//!
//! // Get performance statistics
//! let stats = engine.get_stats();
//! println!("Average inference time: {:.2}ms", stats.avg_inference_time_ms);
//! # Ok(())
//! # }
//! ```
//!
//! ### NNAPI Hardware Acceleration
//!
//! ```rust
//! use trustformers_mobile::android_backup::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Detect available NNAPI devices
//! let devices = AndroidInferenceEngine::detect_nnapi_devices();
//! for device in &devices {
//!     println!("NNAPI Device: {} ({})", device.name, device.device_type);
//! }
//!
//! // Get best device for inference
//! if let Some(best_device) = AndroidInferenceEngine::get_best_nnapi_device() {
//!     println!("Best device: {} ({})", best_device.name, best_device.device_type);
//! }
//!
//! // Check hardware acceleration availability
//! let has_hw_accel = AndroidInferenceEngine::has_hardware_acceleration();
//! println!("Hardware acceleration: {}", has_hw_accel);
//! # Ok(())
//! # }
//! ```
//!
//! ### GPU Compute with Vulkan
//!
//! ```rust
//! use trustformers_mobile::android_backup::gpu::*;
//!
//! # #[cfg(target_os = "android")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Check Vulkan availability
//! if is_vulkan_available() {
//!     // Create Vulkan compute context
//!     let context = VulkanComputeContext::new()?;
//!
//!     // Create compute pipeline for Conv2D
//!     let pipeline = context.create_compute_pipeline(ComputeOperation::Conv2D)?;
//!
//!     // Execute compute operation
//!     context.execute_compute(pipeline, 16, 16, 1)?;
//!
//!     println!("Vulkan compute operation completed");
//! }
//! # Ok(())
//! # }
//! # #[cfg(not(target_os = "android"))]
//! # fn main() {}
//! ```
//!
//! ### Device Feature Detection
//!
//! ```rust
//! use trustformers_mobile::android_backup::*;
//!
//! # fn main() {
//! let device_info = AndroidDeviceInfo::detect();
//!
//! // Check specific features
//! if device_info.supports_feature(AndroidFeature::NNAPI) {
//!     println!("NNAPI supported");
//! }
//! if device_info.supports_feature(AndroidFeature::VulkanGPU) {
//!     println!("Vulkan GPU acceleration supported");
//! }
//! if device_info.supports_feature(AndroidFeature::FP16Inference) {
//!     println!("FP16 inference supported");
//! }
//!
//! // Device classification
//! if device_info.is_flagship_device() {
//!     println!("High-end flagship device detected");
//! }
//! if device_info.is_ml_capable() {
//!     println!("Device capable of machine learning workloads");
//! }
//! # }
//! ```
//!
//! ### Thermal Management
//!
//! ```rust
//! use trustformers_mobile::android_backup::*;
//!
//! # fn main() {
//! let device_info = AndroidDeviceInfo::detect();
//!
//! match device_info.thermal_status {
//!     AndroidThermalStatus::Normal => {
//!         println!("Thermal state normal - full performance available");
//!     },
//!     AndroidThermalStatus::Moderate => {
//!         println!("Moderate thermal throttling - reducing batch size");
//!     },
//!     AndroidThermalStatus::Critical => {
//!         println!("Critical thermal state - switching to minimal mode");
//!     },
//!     _ => {}
//! }
//!
//! // Get thermal recommendations
//! let recommendations = device_info.get_thermal_recommendations();
//! for rec in recommendations {
//!     println!("Recommendation: {}", rec);
//! }
//! # }
//! ```
//!
//! ## JNI Integration
//!
//! For Android applications, the module provides complete JNI bindings:
//!
//! ```java
//! // Java/Kotlin usage example
//! public class TrustformersEngine {
//!     static {
//!         System.loadLibrary("trustformers_mobile");
//!     }
//!
//!     public static native long createEngine(String configJson);
//!     public static native boolean loadModel(long enginePtr, String modelPath);
//!     public static native byte[] inference(long enginePtr, byte[] inputData);
//!     public static native String getDeviceInfo();
//!     public static native String checkCapabilities();
//!     public static native void releaseEngine(long enginePtr);
//! }
//! ```
//!
//! ## Performance Characteristics
//!
//! ### NNAPI Performance
//! - **NPU/Accelerator**: 10-100x faster than CPU for supported operations
//! - **GPU**: 5-20x faster than CPU for parallel workloads
//! - **CPU**: Optimized with NEON/ARM64 instructions
//!
//! ### Memory Efficiency
//! - **Adaptive batching**: Automatically adjusts based on device memory
//! - **FP16 optimization**: Reduces memory usage by 50% on compatible devices
//! - **Quantization**: INT8 support for additional memory savings
//!
//! ### Power Management
//! - **Thermal monitoring**: Real-time thermal state tracking
//! - **Dynamic frequency scaling**: DVFS integration for power efficiency
//! - **Background optimization**: Automatic performance adjustments
//!
//! ## Platform Requirements
//!
//! - **Android API Level**: 21+ (Android 5.0+)
//! - **NNAPI**: API Level 27+ (Android 8.1+) for hardware acceleration
//! - **Vulkan**: API Level 24+ (Android 7.0+) for Vulkan GPU support
//! - **OpenGL ES**: 3.1+ for compute shader support
//!
//! ## Integration with Android Applications
//!
//! ### Gradle Dependencies
//! ```gradle
//! android {
//!     ndkVersion "21.4.7075529"
//!
//!     defaultConfig {
//!         ndk {
//!             abiFilters 'arm64-v8a', 'armeabi-v7a'
//!         }
//!     }
//! }
//! ```
//!
//! ### Proguard Rules
//! ```proguard
//! -keep class com.trustformers.TrustformersEngine { *; }
//! -keepclassmembers class com.trustformers.TrustformersEngine {
//!     native <methods>;
//! }
//! ```
//!
//! ## Error Handling
//!
//! The module provides comprehensive error handling for Android-specific scenarios:
//!
//! ```rust
//! use trustformers_mobile::android_backup::*;
//! use CoreError;
//!
//! # fn example() -> Result<(), CoreError> {
//! match AndroidInferenceEngine::new(config) {
//!     Ok(engine) => {
//!         // Success
//!     },
//!     Err(TrustformersError::config_error(msg)) => {
//!         // Invalid configuration
//!     },
//!     Err(TrustformersError::runtime_error(msg)) => {
//!         // Runtime issues (NNAPI unavailable, GPU init failed, etc.)
//!     },
//!     Err(e) => {
//!         // Other errors
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Best Practices
//!
//! 1. **Device Detection**: Always check device capabilities before selecting backends
//! 2. **Thermal Management**: Monitor thermal state and adjust performance accordingly
//! 3. **Memory Management**: Use recommended configurations based on device specs
//! 4. **Error Handling**: Gracefully fall back to CPU when hardware acceleration fails
//! 5. **JNI Safety**: Properly manage engine lifecycle and native memory
//!
//! ## Troubleshooting
//!
//! ### NNAPI Issues
//! - Ensure Android 8.1+ for NNAPI support
//! - Check device manufacturer NNAPI implementation quality
//! - Fall back to CPU if NNAPI execution fails
//!
//! ### GPU Issues
//! - Verify Vulkan/OpenGL ES support on device
//! - Check for driver compatibility issues
//! - Monitor memory usage for GPU operations
//!
//! ### JNI Issues
//! - Ensure proper library loading in Java/Kotlin
//! - Check NDK version compatibility
//! - Verify architecture-specific builds (arm64-v8a, armeabi-v7a)

mod android_backup;

// Re-export the main public API for easy access
pub use android_backup::*;

// Re-export all core types for convenience
pub use android_backup::{
    AndroidDeviceInfo, AndroidInferenceEngine, AndroidGPUBackend, AndroidThermalStatus,
    AndroidPerformanceClass, AndroidFeature, AndroidGPUInfo, NNAPIInfo, NNAPIHardwareDevice,
};

// Re-export NNAPI functionality
pub use android_backup::nnapi::{
    NNAPIDeviceManager, NNAPIExecutor, NNAPIModelBuilder, is_nnapi_available,
};

// Re-export GPU functionality
pub use android_backup::gpu::{
    VulkanComputeContext, OpenGLESComputeContext, is_vulkan_available,
    is_opengl_es_compute_available, ComputeOperation,
};

// Re-export JNI utilities
pub use android_backup::jni::utils;

// Re-export initialization functions
pub use android_backup::{initialize, supports_hardware_acceleration, get_recommended_config};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MobileConfig;
    use trustformers_core::Tensor;

    #[test]
    fn test_android_platform_initialization() {
        let result = initialize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_device_detection() {
        let device_info = AndroidDeviceInfo::detect();
        assert!(!device_info.manufacturer.is_empty());
        assert!(!device_info.model.is_empty());
        assert!(device_info.api_level > 0);
        assert!(device_info.total_memory_mb > 0);
        assert!(device_info.cpu_cores > 0);
    }

    #[test]
    fn test_recommended_config() {
        let config = get_recommended_config();
        assert_eq!(config.platform, crate::MobilePlatform::Android);
        assert!(config.max_memory_mb > 0);
        assert!(config.get_thread_count() > 0);
    }

    #[test]
    fn test_hardware_acceleration_check() {
        let has_hw_accel = supports_hardware_acceleration();
        // This will be false on non-Android platforms, true if NNAPI/GPU available
        println!("Hardware acceleration available: {}", has_hw_accel);
    }

    #[test]
    fn test_inference_engine_creation() {
        let config = MobileConfig::android_optimized();
        let engine = AndroidInferenceEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_nnapi_device_detection() {
        let devices = AndroidInferenceEngine::detect_nnapi_devices();
        // Will be empty on non-Android platforms
        println!("Found {} NNAPI devices", devices.len());
    }

    #[test]
    fn test_feature_support() {
        let device_info = AndroidDeviceInfo::detect();

        // These should always be supported
        assert!(device_info.supports_feature(AndroidFeature::Int8Quantization));

        // Test device classification
        let _is_flagship = device_info.is_flagship_device();
        let _is_ml_capable = device_info.is_ml_capable();
    }

    #[test]
    fn test_thermal_management() {
        let device_info = AndroidDeviceInfo::detect();
        let recommendations = device_info.get_thermal_recommendations();
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_gpu_availability() {
        let vulkan_available = is_vulkan_available();
        let opengl_available = is_opengl_es_compute_available();

        println!("Vulkan available: {}", vulkan_available);
        println!("OpenGL ES compute available: {}", opengl_available);
    }

    #[cfg(target_os = "android")]
    #[test]
    fn test_android_specific_features() {
        // These tests only run on actual Android devices
        let config = MobileConfig::android_optimized();
        let mut engine = AndroidInferenceEngine::new(config).unwrap();

        // Test model loading (may fail without actual model file)
        let _result = engine.load_model("test_model.onnx");

        // Test GPU context creation
        let _vulkan_context = VulkanComputeContext::new();
        let _opengl_context = OpenGLESComputeContext::new();
    }

    #[test]
    fn test_comprehensive_integration() {
        // Test full integration workflow
        let device_info = AndroidDeviceInfo::detect();
        let config = device_info.get_recommended_config();
        let engine = AndroidInferenceEngine::new(config);

        assert!(engine.is_ok());
        let engine = engine.unwrap();

        // Check capabilities match expectations
        if device_info.supports_feature(AndroidFeature::NNAPI) {
            assert!(AndroidInferenceEngine::has_hardware_acceleration());
        }

        // Test stats collection
        let stats = engine.get_stats();
        assert_eq!(stats.total_inferences, 0); // No inferences performed yet
    }
}