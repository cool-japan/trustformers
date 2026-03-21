//! # PerformanceAnalyzerConfig - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceAnalyzerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::PerformanceAnalyzerConfig;

impl Default for PerformanceAnalyzerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            analysis_interval: Duration::from_secs(60),
            history_retention: Duration::from_secs(3600),
            enable_detailed_tracking: true,
            enable_cpu_monitoring: true,
            enable_memory_monitoring: true,
        }
    }
}
