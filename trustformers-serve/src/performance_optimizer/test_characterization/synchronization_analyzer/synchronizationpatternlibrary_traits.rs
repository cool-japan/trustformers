//! # SynchronizationPatternLibrary - Trait Implementations
//!
//! This module contains trait implementations for `SynchronizationPatternLibrary`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SynchronizationPatternLibrary;

impl std::fmt::Debug for SynchronizationPatternLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SynchronizationPatternLibrary")
            .field("patterns", &format!("<{} pattern(s)>", self.patterns.len()))
            .finish()
    }
}

impl Default for SynchronizationPatternLibrary {
    fn default() -> Self {
        Self::new()
    }
}
