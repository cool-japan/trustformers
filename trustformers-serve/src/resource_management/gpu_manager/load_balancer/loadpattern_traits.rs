//! # LoadPattern - Trait Implementations
//!
//! This module contains trait implementations for `LoadPattern`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::LoadPattern;

impl std::fmt::Display for LoadPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Steady => write!(f, "Steady"),
            Self::Ramping => write!(f, "Ramping"),
            Self::Declining => write!(f, "Declining"),
            Self::Bursty => write!(f, "Bursty"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}
