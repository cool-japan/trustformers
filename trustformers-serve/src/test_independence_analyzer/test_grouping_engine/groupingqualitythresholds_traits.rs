//! # GroupingQualityThresholds - Trait Implementations
//!
//! This module contains trait implementations for `GroupingQualityThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::GroupingQualityThresholds;

impl Default for GroupingQualityThresholds {
    fn default() -> Self {
        Self {
            min_homogeneity: 0.6,
            min_resource_compatibility: 0.7,
            min_duration_balance: 0.5,
            max_dependency_complexity: 0.8,
            min_parallelization_potential: 0.6,
        }
    }
}
