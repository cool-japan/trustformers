//! # ConflictPriorityThresholds - Trait Implementations
//!
//! This module contains trait implementations for `ConflictPriorityThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use crate::resource_management::types::*;

use super::types::ConflictPriorityThresholds;

impl Default for ConflictPriorityThresholds {
    fn default() -> Self {
        Self {
            force_allocation_threshold: 9.0,
            critical_test_boost: 2.0,
            long_running_penalty: 0.5,
            base_priority: 1.0,
        }
    }
}

