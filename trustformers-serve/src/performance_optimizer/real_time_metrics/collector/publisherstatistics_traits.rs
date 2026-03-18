//! # PublisherStatistics - Trait Implementations
//!
//! This module contains trait implementations for `PublisherStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use parking_lot::RwLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use super::types::PublisherStatistics;

impl Clone for PublisherStatistics {
    fn clone(&self) -> Self {
        Self {
            messages_published: AtomicU64::new(self.messages_published.load(Ordering::Relaxed)),
            messages_failed: AtomicU64::new(self.messages_failed.load(Ordering::Relaxed)),
            avg_publish_latency: AtomicU32::new(self.avg_publish_latency.load(Ordering::Relaxed)),
            throughput: AtomicU32::new(self.throughput.load(Ordering::Relaxed)),
            error_rate: AtomicU32::new(self.error_rate.load(Ordering::Relaxed)),
            last_publish_time: Arc::new(RwLock::new(*self.last_publish_time.read())),
        }
    }
}
