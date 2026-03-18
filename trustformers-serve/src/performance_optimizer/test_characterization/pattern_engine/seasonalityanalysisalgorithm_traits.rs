//! # SeasonalityAnalysisAlgorithm - Trait Implementations
//!
//! This module contains trait implementations for `SeasonalityAnalysisAlgorithm`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `TemporalAnalysisAlgorithm`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use std::future::Future;
use std::pin::Pin;

use super::functions::TemporalAnalysisAlgorithm;
use super::types::{PatternEvolutionData, SeasonalityAnalysisAlgorithm, TemporalInsight};

impl Default for SeasonalityAnalysisAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalAnalysisAlgorithm for SeasonalityAnalysisAlgorithm {
    fn analyze_temporal_patterns(
        &self,
        _evolution_data: &PatternEvolutionData,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<TemporalInsight>>> + Send + '_>> {
        Box::pin(async move { Ok(Vec::new()) })
    }
    fn name(&self) -> &str {
        "Seasonality Analysis Algorithm"
    }
}
