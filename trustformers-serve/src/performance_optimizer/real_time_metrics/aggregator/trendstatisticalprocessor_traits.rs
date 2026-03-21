//! # TrendStatisticalProcessor - Trait Implementations
//!
//! This module contains trait implementations for `TrendStatisticalProcessor`.
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
use super::types::{ProcessorConfig, TrendStatisticalProcessor};

impl Default for TrendStatisticalProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalProcessor for TrendStatisticalProcessor {
    fn process(&self, _metrics: &[TimestampedMetrics]) -> Result<StatisticalResult> {
        Ok(StatisticalResult::default())
    }
    fn name(&self) -> &str {
        "trend_statistical"
    }
    fn config(&self) -> ProcessorConfig {
        ProcessorConfig::default()
    }
    fn validate_input(&self, _metrics: &[TimestampedMetrics]) -> Result<()> {
        Ok(())
    }
}

impl std::fmt::Debug for TrendStatisticalProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrendStatisticalProcessor").finish()
    }
}
