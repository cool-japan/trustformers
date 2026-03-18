//! # ParallelPerformanceMonitoringConfig - Trait Implementations
//!
//! This module contains trait implementations for `ParallelPerformanceMonitoringConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::ParallelPerformanceMonitoringConfig;

impl Default for ParallelPerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval: Duration::from_millis(500),
            regression_detection: true,
            bottleneck_detection: true,
            resource_utilization_tracking: true,
            parallel_efficiency_metrics: true,
        }
    }
}
