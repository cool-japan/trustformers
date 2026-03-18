//! # GpuAlertThresholds - Trait Implementations
//!
//! This module contains trait implementations for `GpuAlertThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::GpuAlertThresholds;

impl Default for GpuAlertThresholds {
    fn default() -> Self {
        Self {
            temperature_threshold: 80.0,
            memory_utilization_threshold: 0.9,
            compute_utilization_threshold: 0.95,
            power_threshold: 250.0,
            memory_fragmentation_threshold: 0.3,
            error_rate_threshold: 0.01,
        }
    }
}
