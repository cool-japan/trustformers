//! # SynchronizationAnalyzerConfig - Trait Implementations
//!
//! This module contains trait implementations for `SynchronizationAnalyzerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::SynchronizationAnalyzerConfig;

impl Default for SynchronizationAnalyzerConfig {
    fn default() -> Self {
        Self {
            max_analysis_depth: 10,
            deadlock_detection_sensitivity: 0.85,
            critical_section_threshold_us: 1000,
            wait_time_threshold_ms: 10,
            pattern_recognition_threshold: 0.80,
            lock_ordering_optimization: true,
            real_time_metrics: true,
            cache_size_limit: 1000,
            max_analysis_duration: Duration::from_secs(30),
            parallel_workers: 8,
            advanced_deadlock_prevention: true,
            ml_pattern_recognition: false,
        }
    }
}
