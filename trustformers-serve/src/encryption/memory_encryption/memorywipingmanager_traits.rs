//! # MemoryWipingManager - Trait Implementations
//!
//! This module contains trait implementations for `MemoryWipingManager`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::Arc;

use super::types::MemoryWipingManager;

impl Clone for MemoryWipingManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            wiping_queue: Arc::clone(&self.wiping_queue),
            wiping_patterns: Arc::clone(&self.wiping_patterns),
            stats: Arc::clone(&self.stats),
        }
    }
}

