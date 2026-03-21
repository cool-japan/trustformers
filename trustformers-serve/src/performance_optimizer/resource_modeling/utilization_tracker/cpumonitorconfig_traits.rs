//! # CpuMonitorConfig - Trait Implementations
//!
//! This module contains trait implementations for `CpuMonitorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::CpuMonitorConfig;

impl Default for CpuMonitorConfig {
    fn default() -> Self {
        Self {
            per_core_monitoring: true,
            per_thread_monitoring: true,
            frequency_monitoring: true,
            temperature_correlation: true,
            thread_monitoring_threshold: 1.0,
            max_tracked_threads: 100,
        }
    }
}
