//! # CpuScalingConfig - Trait Implementations
//!
//! This module contains trait implementations for `CpuScalingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::CpuScalingConfig;

impl Default for CpuScalingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_cpu_utilization: 0.4,
            max_cpu_utilization: 0.8,
            scaling_factor: 1.2,
            adjustment_interval: Duration::from_secs(30),
        }
    }
}
