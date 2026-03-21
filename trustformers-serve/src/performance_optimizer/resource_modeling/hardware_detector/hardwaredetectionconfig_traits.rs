//! # HardwareDetectionConfig - Trait Implementations
//!
//! This module contains trait implementations for `HardwareDetectionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::{
    CapabilityAssessmentConfig, GpuDetectionConfig, HardwareValidationConfig,
    MemoryDetectionConfig, MotherboardDetectionConfig, NetworkDetectionConfig,
    StorageDetectionConfig, VendorOptimizationConfig,
};
use std::time::Duration;

use super::types::{CpuDetectionConfig, HardwareDetectionConfig};

impl Default for HardwareDetectionConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_ttl: Duration::from_secs(300),
            cpu_config: CpuDetectionConfig::default(),
            memory_config: MemoryDetectionConfig::default(),
            storage_config: StorageDetectionConfig::default(),
            network_config: NetworkDetectionConfig::default(),
            gpu_config: GpuDetectionConfig::default(),
            motherboard_config: MotherboardDetectionConfig::default(),
            vendor_config: VendorOptimizationConfig::default(),
            capability_config: CapabilityAssessmentConfig::default(),
            validation_config: HardwareValidationConfig::default(),
        }
    }
}
