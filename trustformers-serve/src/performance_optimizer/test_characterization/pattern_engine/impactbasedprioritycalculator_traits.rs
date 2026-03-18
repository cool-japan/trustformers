//! # ImpactBasedPriorityCalculator - Trait Implementations
//!
//! This module contains trait implementations for `ImpactBasedPriorityCalculator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `PriorityCalculator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::test_characterization::types::OptimizationRecommendation;

use super::functions::PriorityCalculator;
use super::types::{ImpactBasedPriorityCalculator, RecommendationContext};

impl Default for ImpactBasedPriorityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl PriorityCalculator for ImpactBasedPriorityCalculator {
    fn calculate_priority(
        &self,
        recommendation: &OptimizationRecommendation,
        _context: &RecommendationContext,
    ) -> f64 {
        recommendation.expected_benefit * (1.0 - recommendation.risk)
    }
    fn name(&self) -> &str {
        "Impact-Based Priority Calculator"
    }
}
