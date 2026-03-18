//! # TracingManager - Trait Implementations
//!
//! This module contains trait implementations for `TracingManager`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::{AtomicU64, Ordering};

use super::types::TracingManager;

impl Clone for TracingManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            active_spans: self.active_spans.clone(),
            export_queue: self.export_queue.clone(),
            stats: self.stats.clone(),
            span_counter: AtomicU64::new(self.span_counter.load(Ordering::Relaxed)),
            sample_counter: AtomicU64::new(self.sample_counter.load(Ordering::Relaxed)),
            last_sample_time: self.last_sample_time.clone(),
            event_sender: self.event_sender.clone(),
        }
    }
}
