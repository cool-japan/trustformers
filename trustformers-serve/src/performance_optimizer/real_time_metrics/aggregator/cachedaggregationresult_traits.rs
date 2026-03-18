//! # CachedAggregationResult - Trait Implementations
//!
//! This module contains trait implementations for `CachedAggregationResult`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::{AtomicUsize, Ordering};

use super::types::CachedAggregationResult;

impl Clone for CachedAggregationResult {
    fn clone(&self) -> Self {
        Self {
            result: self.result.clone(),
            cached_at: self.cached_at,
            cache_duration: self.cache_duration,
            access_count: AtomicUsize::new(self.access_count.load(Ordering::Relaxed)),
        }
    }
}
