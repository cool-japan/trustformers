//! # OverflowStrategy - Trait Implementations
//!
//! This module contains trait implementations for `OverflowStrategy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::OverflowStrategy;

impl Default for OverflowStrategy {
    fn default() -> Self {
        Self::Overwrite
    }
}
