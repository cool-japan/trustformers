//! # RecommendationPriority - Trait Implementations
//!
//! This module contains trait implementations for `RecommendationPriority`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::RecommendationPriority;

impl fmt::Display for RecommendationPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecommendationPriority::Critical => write!(f, "Critical"),
            RecommendationPriority::High => write!(f, "High"),
            RecommendationPriority::Medium => write!(f, "Medium"),
            RecommendationPriority::Low => write!(f, "Low"),
        }
    }
}
