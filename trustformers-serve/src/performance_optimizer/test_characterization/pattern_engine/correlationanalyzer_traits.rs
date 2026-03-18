//! # CorrelationAnalyzer - Trait Implementations
//!
//! This module contains trait implementations for `CorrelationAnalyzer`.
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
    CorrelationAnalyzer, StatisticalAnalysisConfig, StatisticalDataPoint, StatisticalPattern,
};

impl Default for CorrelationAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalAlgorithm for CorrelationAnalyzer {
    fn analyze(
        &self,
        _data: &[StatisticalDataPoint],
        _config: &StatisticalAnalysisConfig,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<StatisticalPattern>>> + Send + '_>> {
        Box::pin(async move { Ok(Vec::new()) })
    }
    fn name(&self) -> &str {
        "Correlation Analyzer"
    }
    fn is_applicable(&self, data: &[StatisticalDataPoint]) -> bool {
        data.len() >= 2
    }
}
