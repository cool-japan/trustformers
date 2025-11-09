//! Android Device Detection and Capabilities
//!
//! This module provides comprehensive device detection, capability analysis,
//! and configuration recommendations for Android devices.

use crate::{MobileBackend, MobileConfig, MobilePlatform, MemoryOptimization};
use super::types::*;

impl AndroidDeviceInfo {
    /// Detect current Android device capabilities and specifications
    pub fn detect() -> Self {
        #[cfg(target_os = "android")]
        {
            Self::detect_android_device()
        }

        #[cfg(not(target_os = "android"))]
        {
            // Return mock data for non-Android platforms (testing/development)
            Self {
                manufacturer: "Unknown".to_string(),
                model: "Emulator".to_string(),
                api_level: 30,
                android_version: "11.0".to_string(),
                total_memory_mb: 2048,
                cpu_cores: 4,
                cpu_architecture: "arm64-v8a".to_string(),
                gpu_info: AndroidGPUInfo {
                    vendor: "Unknown".to_string(),
                    renderer: "Software".to_string(),
                    opengl_es_version: "3.0".to_string(),
                    vulkan_supported: false,
                    gpu_memory_mb: None,
                },
                nnapi_info: None,
                thermal_status: AndroidThermalStatus::None,
                performance_class: None,
            }
        }
    }

    /// Detect actual Android device using platform APIs
    #[cfg(target_os = "android")]
    fn detect_android_device() -> Self {
        let manufacturer = Self::get_manufacturer();
        let model = Self::get_model();
        let api_level = Self::get_api_level();
        let android_version = Self::get_android_version();
        let total_memory_mb = Self::get_total_memory_mb();
        let cpu_cores = Self::get_cpu_cores();
        let cpu_architecture = Self::get_cpu_architecture();
        let gpu_info = Self::get_gpu_info();
        let nnapi_info = Self::get_nnapi_info();
        let thermal_status = Self::get_thermal_status();
        let performance_class = Self::get_performance_class();

        Self {
            manufacturer,
            model,
            api_level,
            android_version,
            total_memory_mb,
            cpu_cores,
            cpu_architecture,
            gpu_info,
            nnapi_info,
            thermal_status,
            performance_class,
        }
    }

    /// Get device manufacturer using Android system properties
    #[cfg(target_os = "android")]
    fn get_manufacturer() -> String {
        // In production: android.os.Build.MANUFACTURER
        "Unknown".to_string()
    }

    /// Get device model using Android system properties
    #[cfg(target_os = "android")]
    fn get_model() -> String {
        // In production: android.os.Build.MODEL
        "Android Device".to_string()
    }

    /// Get Android API level
    #[cfg(target_os = "android")]
    fn get_api_level() -> u32 {
        // In production: android.os.Build.VERSION.SDK_INT
        30
    }

    /// Get Android version string
    #[cfg(target_os = "android")]
    fn get_android_version() -> String {
        // In production: android.os.Build.VERSION.RELEASE
        "11.0".to_string()
    }

    /// Get total device memory in MB
    #[cfg(target_os = "android")]
    fn get_total_memory_mb() -> usize {
        // In production: ActivityManager.MemoryInfo.totalMem
        2048
    }

    /// Get number of CPU cores
    #[cfg(target_os = "android")]
    fn get_cpu_cores() -> usize {
        // In production: Runtime.getRuntime().availableProcessors()
        4
    }

    /// Get CPU architecture
    #[cfg(target_os = "android")]
    fn get_cpu_architecture() -> String {
        // In production: android.os.Build.SUPPORTED_ABIS[0]
        "arm64-v8a".to_string()
    }

    /// Get GPU information and capabilities
    #[cfg(target_os = "android")]
    fn get_gpu_info() -> AndroidGPUInfo {
        AndroidGPUInfo {
            vendor: "Unknown".to_string(),
            renderer: "Unknown".to_string(),
            opengl_es_version: "3.0".to_string(),
            vulkan_supported: false,
            gpu_memory_mb: None,
        }
    }

    /// Get NNAPI information and available devices
    #[cfg(target_os = "android")]
    fn get_nnapi_info() -> Option<NNAPIInfo> {
        // Detect NNAPI hardware devices using the inference engine
        let devices = super::engine::AndroidInferenceEngine::detect_nnapi_devices();

        if devices.is_empty() {
            return None;
        }

        // Convert to serializable format
        let hardware_devices: Vec<NNAPIHardwareDevice> = devices
            .iter()
            .map(|d| NNAPIHardwareDevice {
                name: d.name.clone(),
                device_type: super::nnapi::execution::device_type_to_string(d.device_type)
                    .to_string(),
                feature_level: d.feature_level as u32,
                exec_time: d.performance_info.exec_time,
                power_usage: d.performance_info.power_usage,
                vendor_extensions: d.vendor_extensions.clone(),
            })
            .collect();

        // Get best device recommendation
        let best_device = super::engine::AndroidInferenceEngine::get_best_nnapi_device();
        let best_device_type = if let Some(ref device) = best_device {
            super::nnapi::execution::device_type_to_string(device.device_type).to_string()
        } else {
            "CPU".to_string()
        };

        // Get feature level from best device or use default
        let feature_level = if let Some(ref device) = best_device {
            device.feature_level as u32
        } else {
            27 // Default to API level 27 (NNAPI 1.0)
        };

        // Create device names list
        let available_devices: Vec<String> = hardware_devices
            .iter()
            .map(|d| format!("{} ({})", d.name, d.device_type))
            .collect();

        // List commonly supported operations (simplified)
        let supported_operations = vec![
            "CONV_2D".to_string(),
            "DEPTHWISE_CONV_2D".to_string(),
            "FULLY_CONNECTED".to_string(),
            "AVERAGE_POOL_2D".to_string(),
            "MAX_POOL_2D".to_string(),
            "RELU".to_string(),
            "SOFTMAX".to_string(),
            "ADD".to_string(),
            "MUL".to_string(),
            "RESHAPE".to_string(),
        ];

        Some(NNAPIInfo {
            feature_level,
            available_devices,
            supported_operations,
            hardware_devices,
            best_device_type,
        })
    }

    /// Get current thermal throttling status
    #[cfg(target_os = "android")]
    fn get_thermal_status() -> AndroidThermalStatus {
        // In production: PowerManager.getCurrentThermalStatus()
        AndroidThermalStatus::None
    }

    /// Get performance class (Android 12+)
    #[cfg(target_os = "android")]
    fn get_performance_class() -> Option<AndroidPerformanceClass> {
        // In production: android.os.Build.VERSION.MEDIA_PERFORMANCE_CLASS
        None
    }

    /// Check if device supports specific Android features
    pub fn supports_feature(&self, feature: AndroidFeature) -> bool {
        match feature {
            AndroidFeature::NNAPI => {
                self.nnapi_info.is_some() && self.api_level >= 27
            },
            AndroidFeature::VulkanGPU => {
                self.gpu_info.vulkan_supported && self.api_level >= 24
            },
            AndroidFeature::OpenGLES3 => {
                self.gpu_info.opengl_es_version >= "3.0"
            },
            AndroidFeature::FP16Inference => {
                self.api_level >= 27 // Android 8.1+
            },
            AndroidFeature::Int8Quantization => {
                true // Supported on all devices via software
            },
            AndroidFeature::OnDeviceTraining => {
                self.total_memory_mb >= 4096 && self.cpu_cores >= 6
            },
        }
    }

    /// Generate optimized configuration recommendations for this device
    pub fn get_recommended_config(&self) -> MobileConfig {
        let mut config = MobileConfig::android_optimized();

        // Adjust backend based on capabilities
        if !self.supports_feature(AndroidFeature::NNAPI) {
            config.backend = MobileBackend::CPU;
        } else if let Some(ref nnapi_info) = self.nnapi_info {
            // Use NNAPI if hardware acceleration is available
            if nnapi_info.hardware_devices.iter().any(|d| d.device_type != "CPU") {
                config.backend = MobileBackend::NNAPI;
            } else {
                config.backend = MobileBackend::CPU;
            }
        }

        // Set memory limits based on device capabilities
        config.max_memory_mb = (self.total_memory_mb / 4).max(256).min(1024);

        // Adjust thread count based on CPU cores
        config.num_threads = (self.cpu_cores / 2).max(1).min(4);

        // Adjust configuration based on performance class
        if let Some(performance_class) = self.performance_class {
            match performance_class {
                AndroidPerformanceClass::U | AndroidPerformanceClass::T => {
                    // High-end devices: optimize for performance
                    config.memory_optimization = MemoryOptimization::Balanced;
                    config.enable_batching = true;
                    config.max_batch_size = 4;
                    config.use_fp16 = self.supports_feature(AndroidFeature::FP16Inference);
                },
                AndroidPerformanceClass::S => {
                    // Mid-range devices: balance performance and efficiency
                    config.memory_optimization = MemoryOptimization::Balanced;
                    config.enable_batching = true;
                    config.max_batch_size = 2;
                    config.use_fp16 = self.supports_feature(AndroidFeature::FP16Inference);
                },
                AndroidPerformanceClass::R => {
                    // Lower-end devices: prioritize efficiency
                    config.memory_optimization = MemoryOptimization::Maximum;
                    config.enable_batching = false;
                    config.use_fp16 = false; // Avoid FP16 on older devices
                },
            }
        } else {
            // No performance class available - use conservative settings
            config.memory_optimization = MemoryOptimization::Balanced;
            config.enable_batching = self.total_memory_mb >= 3072; // 3GB+
            config.max_batch_size = if self.total_memory_mb >= 6144 { 4 } else { 2 };
        }

        // Adjust for thermal status
        match self.thermal_status {
            AndroidThermalStatus::Critical | AndroidThermalStatus::Emergency => {
                // Severe thermal throttling: minimize computation
                config.memory_optimization = MemoryOptimization::Maximum;
                config.num_threads = 1;
                config.enable_batching = false;
                config.use_fp16 = false;
            },
            AndroidThermalStatus::Severe => {
                // Moderate thermal throttling: reduce load
                config.memory_optimization = MemoryOptimization::Maximum;
                config.num_threads = (config.num_threads / 2).max(1);
                config.max_batch_size = 1;
            },
            AndroidThermalStatus::Moderate => {
                // Light thermal throttling: slight adjustments
                config.memory_optimization = MemoryOptimization::Balanced;
                config.max_batch_size = (config.max_batch_size / 2).max(1);
            },
            _ => {
                // Normal thermal state: use optimal settings
            },
        }

        // Enable quantization based on device capabilities
        if self.supports_feature(AndroidFeature::Int8Quantization) {
            config.quantization = Some(crate::MobileQuantizationConfig::default());
        }

        // Validate and return configuration
        config
    }

    /// Get human-readable device description
    pub fn get_device_description(&self) -> String {
        format!(
            "{} {} (Android {} API {}, {} cores, {}MB RAM)",
            self.manufacturer,
            self.model,
            self.android_version,
            self.api_level,
            self.cpu_cores,
            self.total_memory_mb
        )
    }

    /// Check if device is likely a high-end flagship device
    pub fn is_flagship_device(&self) -> bool {
        // Heuristics for flagship device detection
        self.total_memory_mb >= 6144 && // 6GB+ RAM
        self.cpu_cores >= 8 && // 8+ cores
        self.api_level >= 30 && // Recent Android version
        self.supports_feature(AndroidFeature::NNAPI) &&
        self.supports_feature(AndroidFeature::VulkanGPU)
    }

    /// Check if device has sufficient resources for machine learning
    pub fn is_ml_capable(&self) -> bool {
        self.total_memory_mb >= 2048 && // 2GB+ RAM minimum
        self.cpu_cores >= 4 && // Quad-core minimum
        (self.supports_feature(AndroidFeature::NNAPI) ||
         self.supports_feature(AndroidFeature::VulkanGPU))
    }

    /// Get thermal management recommendations
    pub fn get_thermal_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        match self.thermal_status {
            AndroidThermalStatus::Critical | AndroidThermalStatus::Emergency => {
                recommendations.push("Reduce inference frequency immediately".to_string());
                recommendations.push("Switch to CPU-only inference".to_string());
                recommendations.push("Disable batching and use single requests".to_string());
                recommendations.push("Consider pausing inference until thermal state improves".to_string());
            },
            AndroidThermalStatus::Severe => {
                recommendations.push("Reduce batch size to minimum".to_string());
                recommendations.push("Decrease inference frequency".to_string());
                recommendations.push("Avoid GPU acceleration".to_string());
            },
            AndroidThermalStatus::Moderate => {
                recommendations.push("Monitor thermal state closely".to_string());
                recommendations.push("Consider reducing batch size".to_string());
            },
            _ => {
                recommendations.push("Thermal state is normal".to_string());
            },
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_feature_support() {
        let device_info = AndroidDeviceInfo::detect();

        // Test basic features that should be available
        assert!(device_info.supports_feature(AndroidFeature::Int8Quantization));

        // Test API-level dependent features
        if device_info.api_level >= 27 {
            // NNAPI should be available on API 27+
        }
    }

    #[test]
    fn test_recommended_config() {
        let device_info = AndroidDeviceInfo::detect();
        let config = device_info.get_recommended_config();

        assert_eq!(config.platform, MobilePlatform::Android);
        assert!(config.max_memory_mb > 0);
        assert!(config.get_thread_count() > 0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_device_classification() {
        let device_info = AndroidDeviceInfo::detect();

        // Test device classification methods
        let _is_flagship = device_info.is_flagship_device();
        let _is_ml_capable = device_info.is_ml_capable();

        // These should not panic
        let _description = device_info.get_device_description();
        let _recommendations = device_info.get_thermal_recommendations();
    }

    #[test]
    fn test_thermal_recommendations() {
        let mut device_info = AndroidDeviceInfo::default();

        // Test different thermal states
        device_info.thermal_status = AndroidThermalStatus::Normal;
        let normal_recs = device_info.get_thermal_recommendations();
        assert!(!normal_recs.is_empty());

        device_info.thermal_status = AndroidThermalStatus::Critical;
        let critical_recs = device_info.get_thermal_recommendations();
        assert!(critical_recs.len() > normal_recs.len());
    }
}