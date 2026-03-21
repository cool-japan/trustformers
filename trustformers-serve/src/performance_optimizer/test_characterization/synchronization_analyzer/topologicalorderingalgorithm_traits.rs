//! # TopologicalOrderingAlgorithm - Trait Implementations
//!
//! This module contains trait implementations for `TopologicalOrderingAlgorithm`.
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
use super::types::TopologicalOrderingAlgorithm;

impl Default for TopologicalOrderingAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl DeadlockOrderingAlgorithm for TopologicalOrderingAlgorithm {
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
        "Topological"
    }
    fn performance_score(&self) -> f64 {
        0.8
    }
}
