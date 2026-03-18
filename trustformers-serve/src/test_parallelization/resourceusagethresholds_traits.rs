//! # ResourceUsageThresholds - Trait Implementations
//!
//! This module contains trait implementations for `ResourceUsageThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ResourceUsageThresholds;

impl Default for ResourceUsageThresholds {
    fn default() -> Self {
        Self {
            cpu_warning: 0.7,
            cpu_critical: 0.9,
            memory_warning: 0.8,
            memory_critical: 0.95,
            gpu_warning: Some(0.8),
            gpu_critical: Some(0.95),
        }
    }
}
