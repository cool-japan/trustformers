//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::{
    CompleteHardwareProfile, CpuVendorFeatures, GpuDeviceModel, GpuVendorFeatures,
    ValidationRuleResult,
};
use anyhow::Result;
use async_trait::async_trait;

/// CPU vendor-specific detector trait
#[async_trait]
pub trait CpuVendorDetector {
    /// Get vendor name
    fn vendor_name(&self) -> &str;
    /// Detect vendor-specific CPU features
    async fn detect_vendor_features(&self) -> Result<CpuVendorFeatures>;
}
/// GPU vendor-specific detector trait
#[async_trait]
pub trait GpuVendorDetector {
    /// Get vendor name
    fn vendor_name(&self) -> &str;
    /// Detect GPU devices
    async fn detect_gpu_devices(&self) -> Result<Vec<GpuDeviceModel>>;
    /// Detect vendor-specific features
    async fn detect_vendor_features(&self) -> Result<GpuVendorFeatures>;
}
/// Hardware validation rule trait
#[async_trait]
pub trait ValidationRule {
    /// Rule name
    fn rule_name(&self) -> &str;
    /// Validate hardware profile
    async fn validate(&self, profile: &CompleteHardwareProfile) -> Result<ValidationRuleResult>;
}
/// Macro to generate default cache types
macro_rules! default_cache_type {
    ($name:ident) => {
        #[derive(Debug, Default)]
        pub struct $name {}
        impl $name {
            pub fn new() -> Self {
                Self::default()
            }
        }
    };
}
default_cache_type!(StorageDetectionCache);
default_cache_type!(NetworkDetectionCache);
default_cache_type!(GpuDetectionCache);
default_cache_type!(MotherboardDetectionCache);
default_cache_type!(VendorOptimizationCache);
default_cache_type!(CapabilityAssessmentCache);
default_cache_type!(ValidationResultCache);
/// Macro to generate default config types
macro_rules! default_config_type {
    ($name:ident) => {
        #[derive(Debug, Clone, Default)]
        pub struct $name {
            pub enable_caching: bool,
        }
    };
}
default_config_type!(MemoryDetectionConfig);
default_config_type!(StorageDetectionConfig);
default_config_type!(NetworkDetectionConfig);
default_config_type!(GpuDetectionConfig);
default_config_type!(MotherboardDetectionConfig);
default_config_type!(VendorOptimizationConfig);
default_config_type!(CapabilityAssessmentConfig);
default_config_type!(HardwareValidationConfig);
#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance_optimizer::resource_modeling::{
        CpuDetectionConfig, CpuDetector, HardwareDetectionConfig, HardwareDetector, MemoryDetector,
    };
    use std::time::Duration;
    #[tokio::test]
    async fn test_hardware_detector_creation() {
        let config = HardwareDetectionConfig::default();
        let detector = HardwareDetector::new(config).await.expect("Failed to create detector");
        let cpu_frequencies = detector
            .detect_cpu_frequencies()
            .await
            .expect("Failed to detect CPU frequencies");
        assert!(cpu_frequencies.0 > 0);
        assert!(cpu_frequencies.1 >= cpu_frequencies.0);
    }
    #[tokio::test]
    async fn test_cpu_detector() {
        let config = CpuDetectionConfig::default();
        let cpu_detector = CpuDetector::new(config).await.expect("Failed to create CPU detector");
        let cpu_profile =
            cpu_detector.detect_cpu_hardware().await.expect("Failed to detect CPU hardware");
        assert!(cpu_profile.core_count > 0);
        assert!(cpu_profile.thread_count >= cpu_profile.core_count);
    }
    #[tokio::test]
    async fn test_memory_detector() {
        let config = MemoryDetectionConfig::default();
        let memory_detector =
            MemoryDetector::new(config).await.expect("Failed to create memory detector");
        let memory_profile = memory_detector
            .detect_memory_hardware()
            .await
            .expect("Failed to detect memory hardware");
        assert!(memory_profile.total_memory_mb > 0);
    }
    #[tokio::test]
    async fn test_cache_size_parsing() {
        let config = CpuDetectionConfig::default();
        let cpu_detector = CpuDetector::new(config).await.expect("Failed to create CPU detector");
        assert_eq!(cpu_detector.parse_cache_size("32K"), 32);
        assert_eq!(cpu_detector.parse_cache_size("256KB"), 256);
        assert_eq!(cpu_detector.parse_cache_size("8M"), 8192);
        assert_eq!(cpu_detector.parse_cache_size("1GB"), 1048576);
    }
    #[tokio::test]
    async fn test_complete_hardware_detection() {
        let config = HardwareDetectionConfig::default();
        let detector = HardwareDetector::new(config).await.expect("Failed to create detector");
        let profile = detector
            .detect_complete_hardware()
            .await
            .expect("Failed to detect complete hardware");
        assert!(profile.detection_duration > Duration::from_nanos(0));
        assert!(profile.cpu_profile.core_count > 0);
    }
}
