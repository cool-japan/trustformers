//! # PriorityFactors - Trait Implementations
//!
//! This module contains trait implementations for `PriorityFactors`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PriorityFactors;

impl Default for PriorityFactors {
    fn default() -> Self {
        Self {
            category_weight: 0.3,
            duration_weight: 0.2,
            resource_weight: 0.25,
            success_rate_weight: 0.15,
            dependency_weight: 0.1,
        }
    }
}
