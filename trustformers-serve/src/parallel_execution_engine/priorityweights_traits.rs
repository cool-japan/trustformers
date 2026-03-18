//! # PriorityWeights - Trait Implementations
//!
//! This module contains trait implementations for `PriorityWeights`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PriorityWeights;

/// Additional type definitions and implementations would continue here...
impl Default for PriorityWeights {
    fn default() -> Self {
        Self {
            category_weight: 0.25,
            duration_weight: 0.20,
            resource_weight: 0.20,
            dependency_weight: 0.15,
            performance_weight: 0.15,
            failure_rate_weight: 0.05,
        }
    }
}
