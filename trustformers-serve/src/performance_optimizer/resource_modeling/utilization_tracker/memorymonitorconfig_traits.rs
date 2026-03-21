//! # MemoryMonitorConfig - Trait Implementations
//!
//! This module contains trait implementations for `MemoryMonitorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MemoryMonitorConfig;

impl Default for MemoryMonitorConfig {
    fn default() -> Self {
        Self {
            allocation_pattern_analysis: true,
            bandwidth_monitoring: true,
            page_fault_tracking: true,
            swap_monitoring: true,
            pressure_threshold: 80.0,
            pattern_analysis_window: 1000,
        }
    }
}
