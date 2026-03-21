//! # ConcurrencyRecommendationGenerator - Trait Implementations
//!
//! This module contains trait implementations for `ConcurrencyRecommendationGenerator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `RecommendationGenerator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::test_characterization::types::{
    DetectedPattern, OptimizationRecommendation, PatternType,
};
use anyhow::Result;
use std::future::Future;
use std::pin::Pin;

use super::functions::RecommendationGenerator;
use super::types::ConcurrencyRecommendationGenerator;

impl Default for ConcurrencyRecommendationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl RecommendationGenerator for ConcurrencyRecommendationGenerator {
    fn generate_recommendations(
        &self,
        _patterns: &[DetectedPattern],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<OptimizationRecommendation>>> + Send + '_>> {
        Box::pin(async move { Ok(Vec::new()) })
    }
    fn name(&self) -> &str {
        "Concurrency Recommendation Generator"
    }
    fn supported_pattern_types(&self) -> Vec<PatternType> {
        vec![PatternType::Concurrency]
    }
}
