//! # CpuDetectionConfig - Trait Implementations
//!
//! This module contains trait implementations for `CpuDetectionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::CpuDetectionConfig;

impl Default for CpuDetectionConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            enable_intel_detection: true,
            enable_amd_detection: true,
            enable_arm_detection: true,
            enable_performance_profiling: true,
        }
    }
}
