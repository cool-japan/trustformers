//! # DistributedEventBus - Trait Implementations
//!
//! This module contains trait implementations for `DistributedEventBus`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::DistributedEventBus;

impl Default for DistributedEventBus {
    fn default() -> Self {
        Self::new()
    }
}
