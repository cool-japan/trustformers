//! # ResourceModelingConfig - Trait Implementations
//!
//! This module contains trait implementations for `ResourceModelingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{AnalysisQuality, ResourceModelingConfig};

impl Default for ResourceModelingConfig {
    fn default() -> Self {
        Self {
            detailed_detection: true,
            enable_profiling: true,
            enable_temperature_monitoring: true,
            enable_numa_analysis: true,
            update_interval: Duration::from_secs(60),
            profiling_samples: 10,
            temperature_threshold: 85.0,
            cache_profiling_results: true,
            max_concurrent_tasks: 8,
            task_timeout: Duration::from_secs(300),
            enable_predictive_analysis: true,
            cache_size_limit_mb: 512,
            enable_error_recovery: true,
            reporting_interval: Duration::from_secs(300),
            analysis_quality: AnalysisQuality::Balanced,
        }
    }
}
