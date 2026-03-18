//! # TemperatureThresholds - Trait Implementations
//!
//! This module contains trait implementations for `TemperatureThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TemperatureThresholds;

impl Default for TemperatureThresholds {
    fn default() -> Self {
        Self {
            normal_temperature: 40.0,
            warning_temperature: 75.0,
            critical_temperature: 85.0,
            emergency_temperature: 95.0,
            throttling_threshold: 90.0,
            fan_adjustment_threshold: 60.0,
            gpu_warning_temperature: 80.0,
            gpu_critical_temperature: 90.0,
        }
    }
}
