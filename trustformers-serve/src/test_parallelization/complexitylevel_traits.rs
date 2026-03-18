//! # ComplexityLevel - Trait Implementations
//!
//! This module contains trait implementations for `ComplexityLevel`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::ComplexityLevel;

impl fmt::Display for ComplexityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComplexityLevel::Low => write!(f, "Low"),
            ComplexityLevel::Medium => write!(f, "Medium"),
            ComplexityLevel::High => write!(f, "High"),
            ComplexityLevel::VeryHigh => write!(f, "Very High"),
        }
    }
}
