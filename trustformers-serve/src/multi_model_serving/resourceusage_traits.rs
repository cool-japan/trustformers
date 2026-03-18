//! # ResourceUsage - Trait Implementations
//!
//! This module contains trait implementations for `ResourceUsage`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ResourceUsage;

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            gpu_memory_usage: 0,
            network_io: 0.0,
        }
    }
}
