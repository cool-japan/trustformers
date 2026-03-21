//! # EffortBasedPriorityCalculator - Trait Implementations
//!
//! This module contains trait implementations for `EffortBasedPriorityCalculator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `PriorityCalculator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::test_characterization::types::OptimizationRecommendation;

use super::functions::PriorityCalculator;
use super::types::{EffortBasedPriorityCalculator, RecommendationContext};

impl Default for EffortBasedPriorityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl PriorityCalculator for EffortBasedPriorityCalculator {
    fn calculate_priority(
        &self,
        recommendation: &OptimizationRecommendation,
        _context: &RecommendationContext,
    ) -> f64 {
        recommendation.expected_benefit / recommendation.complexity.max(0.1)
    }
    fn name(&self) -> &str {
        "Effort-Based Priority Calculator"
    }
}
