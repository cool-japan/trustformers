//! Android Device Information and Capabilities Detection
//!
//! This module provides comprehensive device detection and capability analysis
//! for Android devices to optimize inference configurations.

use crate::{MemoryOptimization, MobileBackend, MobileConfig};
use serde::{Deserialize, Serialize};

/// Android device information and capabilities
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
    /// GPU information
    pub gpu_info: AndroidGPUInfo,
    /// NNAPI availability and version
    pub nnapi_info: Option<NNAPIInfo>,
    /// Thermal throttling status
    pub thermal_status: AndroidThermalStatus,
    /// Performance class (if available, Android 12+)
    pub performance_class: Option<AndroidPerformanceClass>,
}

/// Android GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndroidGPUInfo {
    /// GPU vendor (e.g., "Qualcomm", "ARM", "Imagination")
    pub vendor: String,
    /// GPU renderer name
    pub renderer: String,
    /// OpenGL ES version
    pub opengl_es_version: String,
    /// Vulkan support
    pub vulkan_supported: bool,
    /// GPU memory (if available)
    pub gpu_memory_mb: Option<usize>,
}

/// NNAPI information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNAPIInfo {
    /// NNAPI feature level
    pub feature_level: u32,
    /// Available devices
    pub available_devices: Vec<String>,
    /// Supported operations
    pub supported_operations: Vec<String>,
    /// Hardware acceleration devices
    pub hardware_devices: Vec<NNAPIHardwareDevice>,
    /// Best recommended device
    pub best_device_type: String,
}

/// NNAPI Hardware Device Information (serializable version)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNAPIHardwareDevice {
    /// Device name
    pub name: String,
    /// Device type
    pub device_type: String,
    /// Feature level
    pub feature_level: u32,
    /// Performance metrics
    pub exec_time: f32,
    pub power_usage: f32,
    /// Vendor extensions
    pub vendor_extensions: Vec<String>,
}

/// Android thermal status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AndroidThermalStatus {
    None,
    Light,
    Moderate,
    Severe,
    Critical,
    Emergency,
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

/// Android-specific features
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AndroidFeature {
    NNAPI,
    VulkanGPU,
    OpenGLES3,
    FP16Inference,
    Int8Quantization,
    OnDeviceTraining,
}

impl AndroidDeviceInfo {
    /// Detect current Android device capabilities
    pub fn detect() -> Self {
        #[cfg(target_os = "android")]
        {
            Self::detect_android_device()
        }

        #[cfg(not(target_os = "android"))]
        {
            // Return mock data for non-Android platforms
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

    #[cfg(target_os = "android")]
    fn detect_android_device() -> Self {
        // Detect actual Android device capabilities using Android APIs
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

    #[cfg(target_os = "android")]
    fn get_manufacturer() -> String {
        "Unknown".to_string() // Would use Android APIs
    }

    #[cfg(target_os = "android")]
    fn get_model() -> String {
        "Android Device".to_string() // Would use Android APIs
    }

    #[cfg(target_os = "android")]
    fn get_api_level() -> u32 {
        30 // Would use Android APIs
    }

    #[cfg(target_os = "android")]
    fn get_android_version() -> String {
        "11.0".to_string() // Would use Android APIs
    }

    #[cfg(target_os = "android")]
    fn get_total_memory_mb() -> usize {
        2048 // Would use Android APIs
    }

    #[cfg(target_os = "android")]
    fn get_cpu_cores() -> usize {
        4 // Would use Android APIs
    }

    #[cfg(target_os = "android")]
    fn get_cpu_architecture() -> String {
        "arm64-v8a".to_string() // Would use Android APIs
    }

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

    #[cfg(target_os = "android")]
    fn get_nnapi_info() -> Option<NNAPIInfo> {
        // Mock NNAPI info for compilation - would detect real devices
        let hardware_devices = vec![NNAPIHardwareDevice {
            name: "CPU".to_string(),
            device_type: "CPU".to_string(),
            feature_level: 30,
            exec_time: 10.0,
            power_usage: 5.0,
            vendor_extensions: vec![],
        }];

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
            feature_level: 30,
            available_devices: vec!["CPU".to_string()],
            supported_operations,
            hardware_devices,
            best_device_type: "CPU".to_string(),
        })
    }

    #[cfg(target_os = "android")]
    fn get_thermal_status() -> AndroidThermalStatus {
        AndroidThermalStatus::None // Would use thermal APIs
    }

    #[cfg(target_os = "android")]
    fn get_performance_class() -> Option<AndroidPerformanceClass> {
        None // Would check for Android 12+ performance class
    }

    /// Check if device supports specific features
    pub fn supports_feature(&self, feature: AndroidFeature) -> bool {
        match feature {
            AndroidFeature::NNAPI => self.nnapi_info.is_some(),
            AndroidFeature::VulkanGPU => self.gpu_info.vulkan_supported,
            AndroidFeature::OpenGLES3 => self.gpu_info.opengl_es_version >= "3.0",
            AndroidFeature::FP16Inference => self.api_level >= 27, // Android 8.1+
            AndroidFeature::Int8Quantization => true,              // Supported on all devices
            AndroidFeature::OnDeviceTraining => self.total_memory_mb >= 4096 && self.cpu_cores >= 6,
        }
    }

    /// Get recommended configuration for this device
    pub fn get_recommended_config(&self) -> MobileConfig {
        let mut config = MobileConfig::android_optimized();

        // Adjust based on device capabilities
        if !self.supports_feature(AndroidFeature::NNAPI) {
            config.backend = MobileBackend::CPU;
        }

        // Adjust memory limits based on device
        config.max_memory_mb = (self.total_memory_mb / 4).max(256).min(1024);

        // Adjust for device performance class
        if let Some(performance_class) = self.performance_class {
            match performance_class {
                AndroidPerformanceClass::U | AndroidPerformanceClass::T => {
                    // High-end devices
                    config.memory_optimization = MemoryOptimization::Balanced;
                    config.enable_batching = true;
                    config.max_batch_size = 4;
                },
                AndroidPerformanceClass::S => {
                    // Mid-range devices
                    config.memory_optimization = MemoryOptimization::Aggressive;
                    config.enable_batching = true;
                    config.max_batch_size = 2;
                },
                AndroidPerformanceClass::R => {
                    // Lower-end devices
                    config.memory_optimization = MemoryOptimization::Aggressive;
                    config.enable_batching = false;
                    config.max_batch_size = 1;
                },
            }
        }

        // Thermal-aware configuration
        match self.thermal_status {
            AndroidThermalStatus::Severe | AndroidThermalStatus::Critical => {
                config.enable_thermal_throttling = true;
                config.thermal_threshold = 80.0;
            },
            AndroidThermalStatus::Moderate => {
                config.enable_thermal_throttling = true;
                config.thermal_threshold = 85.0;
            },
            _ => {
                config.enable_thermal_throttling = false;
            },
        }

        config
    }

    /// Get device performance tier based on hardware specs
    pub fn get_performance_tier(&self) -> DevicePerformanceTier {
        let memory_score = match self.total_memory_mb {
            mb if mb >= 8192 => 3,
            mb if mb >= 4096 => 2,
            mb if mb >= 2048 => 1,
            _ => 0,
        };

        let cpu_score = match self.cpu_cores {
            cores if cores >= 8 => 3,
            cores if cores >= 6 => 2,
            cores if cores >= 4 => 1,
            _ => 0,
        };

        let gpu_score = if self.gpu_info.vulkan_supported { 2 } else { 1 };
        let api_score = if self.api_level >= 30 { 1 } else { 0 };

        let total_score = memory_score + cpu_score + gpu_score + api_score;

        match total_score {
            score if score >= 8 => DevicePerformanceTier::Premium,
            score if score >= 5 => DevicePerformanceTier::High,
            score if score >= 3 => DevicePerformanceTier::Medium,
            _ => DevicePerformanceTier::Low,
        }
    }
}

/// Device performance tiers for configuration optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePerformanceTier {
    Low,
    Medium,
    High,
    Premium,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_info_detection() {
        let device_info = AndroidDeviceInfo::detect();
        assert!(!device_info.manufacturer.is_empty());
        assert!(!device_info.model.is_empty());
        assert!(device_info.api_level > 0);
        assert!(device_info.cpu_cores > 0);
        assert!(device_info.total_memory_mb > 0);
    }

    #[test]
    fn test_feature_support() {
        let device_info = AndroidDeviceInfo::detect();

        // Int8 quantization should always be supported
        assert!(device_info.supports_feature(AndroidFeature::Int8Quantization));

        // FP16 should be supported on API 27+
        if device_info.api_level >= 27 {
            assert!(device_info.supports_feature(AndroidFeature::FP16Inference));
        }
    }

    #[test]
    fn test_performance_tier_calculation() {
        let mut device_info = AndroidDeviceInfo::detect();

        // Test premium tier device
        device_info.total_memory_mb = 8192;
        device_info.cpu_cores = 8;
        device_info.gpu_info.vulkan_supported = true;
        device_info.api_level = 33;
        assert_eq!(
            device_info.get_performance_tier(),
            DevicePerformanceTier::Premium
        );

        // Test low tier device
        device_info.total_memory_mb = 1024;
        device_info.cpu_cores = 2;
        device_info.gpu_info.vulkan_supported = false;
        device_info.api_level = 23;
        assert_eq!(
            device_info.get_performance_tier(),
            DevicePerformanceTier::Low
        );
    }

    #[test]
    fn test_recommended_config() {
        let device_info = AndroidDeviceInfo::detect();
        let config = device_info.get_recommended_config();

        // Config should be valid
        assert!(config.max_memory_mb >= 256);
        assert!(config.max_memory_mb <= 1024);
        assert!(config.thread_count > 0);
    }

    #[test]
    fn test_thermal_status_values() {
        use AndroidThermalStatus::*;
        let statuses = [None, Light, Moderate, Severe, Critical, Emergency, Shutdown];

        for status in &statuses {
            // Should be able to serialize/deserialize
            let serialized = serde_json::to_string(status).unwrap();
            let deserialized: AndroidThermalStatus = serde_json::from_str(&serialized).unwrap();
            assert_eq!(*status, deserialized);
        }
    }

    #[test]
    fn test_performance_class_ordering() {
        use AndroidPerformanceClass::*;

        // Performance classes should have logical ordering
        assert!(U as u8 > T as u8);
        assert!(T as u8 > S as u8);
        assert!(S as u8 > R as u8);
    }
}
