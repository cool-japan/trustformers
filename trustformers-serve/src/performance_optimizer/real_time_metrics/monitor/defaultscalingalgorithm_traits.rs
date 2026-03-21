//! # DefaultScalingAlgorithm - Trait Implementations
//!
//! This module contains trait implementations for `DefaultScalingAlgorithm`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `ThreadScalingAlgorithm`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::functions::ThreadScalingAlgorithm;
use super::types::{DefaultScalingAlgorithm, ScalingDecision, ThreadPoolConfig, ThreadPoolMetrics};

impl Default for DefaultScalingAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreadScalingAlgorithm for DefaultScalingAlgorithm {
    fn should_scale(
        &self,
        _metrics: &ThreadPoolMetrics,
        _config: &ThreadPoolConfig,
    ) -> ScalingDecision {
        ScalingDecision::NoChange
    }
    fn calculate_optimal_threads(&self, _metrics: &ThreadPoolMetrics) -> usize {
        4
    }
    fn name(&self) -> &str {
        "default"
    }
    fn update_parameters(&mut self, _config: &ThreadPoolConfig) -> Result<()> {
        Ok(())
    }
}
