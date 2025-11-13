//! # BufferId - Trait Implementations
//!
//! This module contains trait implementations for `BufferId`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::BufferId;

#[cfg(feature = "metal")]
impl Default for BufferId {
    fn default() -> Self {
        Self::new()
    }
}
