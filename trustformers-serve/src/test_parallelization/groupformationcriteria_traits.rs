//! # GroupFormationCriteria - Trait Implementations
//!
//! This module contains trait implementations for `GroupFormationCriteria`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::GroupFormationCriteria;

impl Default for GroupFormationCriteria {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            resource_compatibility: true,
            duration_balance: true,
            priority_clustering: true,
        }
    }
}
