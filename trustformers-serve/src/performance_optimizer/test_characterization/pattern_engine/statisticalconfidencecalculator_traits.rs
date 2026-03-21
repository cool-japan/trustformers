//! # StatisticalConfidenceCalculator - Trait Implementations
//!
//! This module contains trait implementations for `StatisticalConfidenceCalculator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `ConfidenceCalculator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::test_characterization::types::DetectedPattern;

use super::functions::ConfidenceCalculator;
use super::types::{ClassificationContext, StatisticalConfidenceCalculator};

impl Default for StatisticalConfidenceCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceCalculator for StatisticalConfidenceCalculator {
    fn calculate_confidence(
        &self,
        pattern: &DetectedPattern,
        _context: &ClassificationContext,
    ) -> f64 {
        pattern.confidence * 0.9
    }
    fn name(&self) -> &str {
        "Statistical Confidence Calculator"
    }
}
