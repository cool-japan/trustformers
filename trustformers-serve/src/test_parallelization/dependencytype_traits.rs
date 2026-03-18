//! # DependencyType - Trait Implementations
//!
//! This module contains trait implementations for `DependencyType`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::DependencyType;

impl fmt::Display for DependencyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DependencyType::Hard => write!(f, "Hard"),
            DependencyType::Soft => write!(f, "Soft"),
            DependencyType::Conflict => write!(f, "Conflict"),
            DependencyType::Ordering => write!(f, "Ordering"),
            DependencyType::Setup => write!(f, "Setup"),
        }
    }
}
