//! # PowerPriority - Trait Implementations
//!
//! This module contains trait implementations for `PowerPriority`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PowerPriority;

impl std::fmt::Display for PowerPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "Low Power"),
            Self::Balanced => write!(f, "Balanced"),
            Self::High => write!(f, "High Performance"),
        }
    }
}
