//! # GpuMonitorConfig - Trait Implementations
//!
//! This module contains trait implementations for `GpuMonitorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::GpuMonitorConfig;

impl Default for GpuMonitorConfig {
    fn default() -> Self {
        Self {
            per_device_monitoring: true,
            memory_utilization_tracking: true,
            temperature_monitoring: true,
            power_monitoring: true,
            kernel_execution_tracking: true,
            max_tracked_kernels: 1000,
        }
    }
}
