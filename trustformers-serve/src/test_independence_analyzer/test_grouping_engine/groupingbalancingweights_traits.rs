//! # GroupingBalancingWeights - Trait Implementations
//!
//! This module contains trait implementations for `GroupingBalancingWeights`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::GroupingBalancingWeights;

impl Default for GroupingBalancingWeights {
    fn default() -> Self {
        Self {
            resource_utilization_weight: 0.25,
            execution_time_weight: 0.25,
            dependency_weight: 0.2,
            category_similarity_weight: 0.1,
            conflict_avoidance_weight: 0.15,
            complexity_balance_weight: 0.05,
        }
    }
}
