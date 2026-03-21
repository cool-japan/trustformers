//! # CollectionStatistics - Trait Implementations
//!
//! This module contains trait implementations for `CollectionStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::{AtomicU64, Ordering};

use super::types::CollectionStatistics;

impl Clone for CollectionStatistics {
    fn clone(&self) -> Self {
        Self {
            total_snapshots: AtomicU64::new(self.total_snapshots.load(Ordering::Relaxed)),
            collection_started: AtomicU64::new(self.collection_started.load(Ordering::Relaxed)),
            collection_stopped: AtomicU64::new(self.collection_stopped.load(Ordering::Relaxed)),
            failed_collections: AtomicU64::new(self.failed_collections.load(Ordering::Relaxed)),
        }
    }
}

impl Default for CollectionStatistics {
    fn default() -> Self {
        Self::new()
    }
}
