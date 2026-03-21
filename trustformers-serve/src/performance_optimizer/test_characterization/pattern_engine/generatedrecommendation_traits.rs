//! # GeneratedRecommendation - Trait Implementations
//!
//! This module contains trait implementations for `GeneratedRecommendation`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::test_characterization::types::OptimizationRecommendation;
use std::time::Instant;

use super::types::GeneratedRecommendation;

impl Default for GeneratedRecommendation {
    fn default() -> Self {
        Self {
            recommendation: OptimizationRecommendation::default(),
            generated_at: Instant::now(),
            applied: false,
            effectiveness_score: None,
        }
    }
}
