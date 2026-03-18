//! # SubscriptionPerformanceStats - Trait Implementations
//!
//! This module contains trait implementations for `SubscriptionPerformanceStats`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::{AtomicU64, Ordering};

use super::types::SubscriptionPerformanceStats;

impl Clone for SubscriptionPerformanceStats {
    fn clone(&self) -> Self {
        Self {
            total_events_delivered: AtomicU64::new(
                self.total_events_delivered.load(Ordering::Relaxed),
            ),
            total_events_failed: AtomicU64::new(self.total_events_failed.load(Ordering::Relaxed)),
            total_delivery_time: AtomicU64::new(self.total_delivery_time.load(Ordering::Relaxed)),
            average_delivery_latency: AtomicU64::new(
                self.average_delivery_latency.load(Ordering::Relaxed),
            ),
            last_delivery_duration: AtomicU64::new(
                self.last_delivery_duration.load(Ordering::Relaxed),
            ),
            throughput_events_per_second: self.throughput_events_per_second,
        }
    }
}
