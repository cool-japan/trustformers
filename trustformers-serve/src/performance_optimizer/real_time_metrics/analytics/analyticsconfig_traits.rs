//! # AnalyticsConfig - Trait Implementations
//!
//! This module contains trait implementations for `AnalyticsConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{AnalyticsConfig, PerformanceThresholds, QualityThresholds};

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            max_concurrent_analyses: num_cpus::get(),
            analysis_timeout: Duration::from_secs(300),
            confidence_level: 0.95,
            enable_advanced_analytics: true,
            enable_realtime_processing: true,
            enable_result_caching: true,
            max_cache_size: 1000,
            batch_size: 1000,
            quality_thresholds: QualityThresholds::default(),
            performance_thresholds: PerformanceThresholds::default(),
        }
    }
}
