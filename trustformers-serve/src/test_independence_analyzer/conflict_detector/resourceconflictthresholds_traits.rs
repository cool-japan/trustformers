//! # ResourceConflictThresholds - Trait Implementations
//!
//! This module contains trait implementations for `ResourceConflictThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::ResourceConflictThresholds;

impl Default for ResourceConflictThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 0.8,
            memory_threshold: 0.85,
            network_threshold: 0.7,
            disk_io_threshold: 0.75,
            gpu_threshold: 0.9,
            custom_thresholds: HashMap::new(),
        }
    }
}
