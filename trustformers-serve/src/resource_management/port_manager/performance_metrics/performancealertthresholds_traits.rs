//! # PerformanceAlertThresholds - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceAlertThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

use super::types::PerformanceAlertThresholds;

impl Default for PerformanceAlertThresholds {
    fn default() -> Self {
        Self {
            max_avg_allocation_time_ms: 100.0,
            max_p95_allocation_time_ms: 500.0,
            min_success_rate_percent: 95.0,
            max_conflict_rate_percent: 5.0,
            min_throughput_ops_per_minute: 60.0,
        }
    }
}

