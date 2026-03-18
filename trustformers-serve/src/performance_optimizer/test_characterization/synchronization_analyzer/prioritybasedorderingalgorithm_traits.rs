//! # PriorityBasedOrderingAlgorithm - Trait Implementations
//!
//! This module contains trait implementations for `PriorityBasedOrderingAlgorithm`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `DeadlockOrderingAlgorithm`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::LockDependency;
use anyhow::Result;

use super::functions::DeadlockOrderingAlgorithm;
use super::types::PriorityBasedOrderingAlgorithm;

impl Default for PriorityBasedOrderingAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl DeadlockOrderingAlgorithm for PriorityBasedOrderingAlgorithm {
    fn generate_ordering(&self, _dependencies: &[LockDependency]) -> Result<Vec<String>> {
        Ok(vec![])
    }
    fn validate_ordering(
        &self,
        _ordering: &[String],
        _dependencies: &[LockDependency],
    ) -> Result<bool> {
        Ok(true)
    }
    fn name(&self) -> &str {
        "PriorityBased"
    }
    fn performance_score(&self) -> f64 {
        0.7
    }
}
