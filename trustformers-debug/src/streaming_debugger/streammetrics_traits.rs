//! # StreamMetrics - Trait Implementations
//!
//! This module contains trait implementations for `StreamMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use std::time::SystemTime;

impl Default for StreamMetrics {
    fn default() -> Self {
        Self {
            total_events_sent: 0,
            events_per_second: 0.0,
            active_subscribers: 0,
            dropped_events: 0,
            rate_limited_events: 0,
            average_latency_ms: 0.0,
            buffer_utilization: 0.0,
            error_count: 0,
            last_update: SystemTime::now(),
        }
    }
}
