//! # PerformanceTracker - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceTracker`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Default`
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

use super::types::{PerformanceTracker, TypingPatterns, TypingPersonality};

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TypingPatterns {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TypingPersonality {
    fn default() -> Self {
        Self::balanced()
    }
}
