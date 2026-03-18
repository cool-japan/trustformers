//! # CollectionState - Trait Implementations
//!
//! This module contains trait implementations for `CollectionState`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::CollectionState;

impl Default for CollectionState {
    fn default() -> Self {
        Self {
            active: false,
            task_handles: Vec::new(),
            start_time: None,
            last_collection: None,
        }
    }
}
