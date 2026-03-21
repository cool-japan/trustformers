//! # PerformanceThresholds - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::PerformanceThresholds;

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_analysis_duration: Duration::from_secs(60),
            max_memory_usage: 1024 * 1024 * 1024,
            max_cpu_usage: 0.8,
            target_accuracy: 0.95,
        }
    }
}
