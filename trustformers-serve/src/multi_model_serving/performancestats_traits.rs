//! # PerformanceStats - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::PerformanceStats;

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            avg_latency: Duration::from_millis(0),
            p95_latency: Duration::from_millis(0),
            p99_latency: Duration::from_millis(0),
            throughput: 0.0,
            error_rate: 0.0,
            accuracy: None,
            request_count: 0,
        }
    }
}
