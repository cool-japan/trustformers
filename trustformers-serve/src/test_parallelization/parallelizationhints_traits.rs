//! # ParallelizationHints - Trait Implementations
//!
//! This module contains trait implementations for `ParallelizationHints`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ParallelizationHints, ResourceSharingCapabilities};

impl Default for ParallelizationHints {
    fn default() -> Self {
        Self {
            parallel_within_category: true,
            parallel_with_any: false,
            sequential_only: false,
            preferred_batch_size: None,
            optimal_concurrency: None,
            resource_sharing: ResourceSharingCapabilities::default(),
        }
    }
}
