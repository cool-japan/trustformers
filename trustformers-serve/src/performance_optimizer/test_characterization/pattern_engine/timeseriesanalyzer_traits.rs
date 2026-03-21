//! # TimeSeriesAnalyzer - Trait Implementations
//!
//! This module contains trait implementations for `TimeSeriesAnalyzer`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `StatisticalAlgorithm`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use std::future::Future;
use std::pin::Pin;

use super::functions::StatisticalAlgorithm;
use super::types::{
    StatisticalAnalysisConfig, StatisticalDataPoint, StatisticalPattern, TimeSeriesAnalyzer,
};

impl Default for TimeSeriesAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalAlgorithm for TimeSeriesAnalyzer {
    fn analyze(
        &self,
        _data: &[StatisticalDataPoint],
        _config: &StatisticalAnalysisConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<StatisticalPattern>>> + Send + '_>> {
        Box::pin(async move { Ok(Vec::new()) })
    }
    fn name(&self) -> &str {
        "Time Series Analyzer"
    }
    fn is_applicable(&self, data: &[StatisticalDataPoint]) -> bool {
        data.len() >= 5
    }
}
