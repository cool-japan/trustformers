//! # SchedulingStrategy - Trait Implementations
//!
//! This module contains trait implementations for `SchedulingStrategy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::SchedulingStrategy;

impl Default for SchedulingStrategy {
    fn default() -> Self {
        SchedulingStrategy::ResourceAware
    }
}

impl fmt::Display for SchedulingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchedulingStrategy::Fifo => write!(f, "FIFO"),
            SchedulingStrategy::ShortestJobFirst => write!(f, "Shortest Job First"),
            SchedulingStrategy::Priority => write!(f, "Priority"),
            SchedulingStrategy::ResourceAware => write!(f, "Resource Aware"),
            SchedulingStrategy::Adaptive => write!(f, "Adaptive"),
            SchedulingStrategy::Custom(name) => write!(f, "Custom: {}", name),
        }
    }
}
