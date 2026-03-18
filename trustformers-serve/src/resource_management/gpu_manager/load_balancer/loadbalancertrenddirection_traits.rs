//! # LoadBalancerTrendDirection - Trait Implementations
//!
//! This module contains trait implementations for `LoadBalancerTrendDirection`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::LoadBalancerTrendDirection;

impl std::fmt::Display for LoadBalancerTrendDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Increasing => write!(f, "Increasing"),
            Self::Decreasing => write!(f, "Decreasing"),
            Self::Stable => write!(f, "Stable"),
            Self::Volatile => write!(f, "Volatile"),
        }
    }
}
