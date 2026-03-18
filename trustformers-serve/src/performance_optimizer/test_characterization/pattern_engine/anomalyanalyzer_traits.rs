//! # AnomalyAnalyzer - Trait Implementations
//!
//! This module contains trait implementations for `AnomalyAnalyzer`.
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
    AnomalyAnalyzer, StatisticalAnalysisConfig, StatisticalDataPoint, StatisticalPattern,
};

impl Default for AnomalyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalAlgorithm for AnomalyAnalyzer {
    fn analyze(
        &self,
        _data: &[StatisticalDataPoint],
        _config: &StatisticalAnalysisConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<StatisticalPattern>>> + Send + '_>> {
        Box::pin(async move { Ok(Vec::new()) })
    }
    fn name(&self) -> &str {
        "Anomaly Analyzer"
    }
    fn is_applicable(&self, data: &[StatisticalDataPoint]) -> bool {
        data.len() >= 10
    }
}
