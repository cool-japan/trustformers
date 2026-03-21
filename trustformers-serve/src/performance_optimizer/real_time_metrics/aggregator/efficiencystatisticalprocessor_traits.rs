//! # EfficiencyStatisticalProcessor - Trait Implementations
//!
//! This module contains trait implementations for `EfficiencyStatisticalProcessor`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `StatisticalProcessor`
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::{StatisticalResult, TimestampedMetrics};
use anyhow::Result;

use super::functions::StatisticalProcessor;
use super::types::{EfficiencyStatisticalProcessor, ProcessorConfig};

impl Default for EfficiencyStatisticalProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalProcessor for EfficiencyStatisticalProcessor {
    fn process(&self, _metrics: &[TimestampedMetrics]) -> Result<StatisticalResult> {
        Ok(StatisticalResult::default())
    }
    fn name(&self) -> &str {
        "efficiency_statistical"
    }
    fn config(&self) -> ProcessorConfig {
        ProcessorConfig::default()
    }
    fn validate_input(&self, _metrics: &[TimestampedMetrics]) -> Result<()> {
        Ok(())
    }
}

impl std::fmt::Debug for EfficiencyStatisticalProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EfficiencyStatisticalProcessor").finish()
    }
}
