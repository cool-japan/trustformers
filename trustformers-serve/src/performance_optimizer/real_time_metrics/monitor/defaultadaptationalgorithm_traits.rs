//! # DefaultAdaptationAlgorithm - Trait Implementations
//!
//! This module contains trait implementations for `DefaultAdaptationAlgorithm`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `BaselineAdaptationAlgorithm`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::TimestampedMetrics;
use anyhow::Result;

use super::functions::BaselineAdaptationAlgorithm;
use super::types::{
    BaselineConfig, BaselineValidationResult, DefaultAdaptationAlgorithm, PerformanceBaseline,
};

impl Default for DefaultAdaptationAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl BaselineAdaptationAlgorithm for DefaultAdaptationAlgorithm {
    fn adapt_baseline(
        &self,
        current: &PerformanceBaseline,
        _new_data: &[TimestampedMetrics],
    ) -> Result<PerformanceBaseline> {
        Ok(current.clone())
    }
    fn name(&self) -> &str {
        "default"
    }
    fn validate_baseline(&self, _baseline: &PerformanceBaseline) -> BaselineValidationResult {
        BaselineValidationResult::Valid
    }
    fn update_parameters(&mut self, _config: &BaselineConfig) -> Result<()> {
        Ok(())
    }
}
