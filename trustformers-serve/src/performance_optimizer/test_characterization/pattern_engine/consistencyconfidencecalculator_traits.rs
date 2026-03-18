//! # ConsistencyConfidenceCalculator - Trait Implementations
//!
//! This module contains trait implementations for `ConsistencyConfidenceCalculator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `ConfidenceCalculator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::test_characterization::types::DetectedPattern;

use super::functions::ConfidenceCalculator;
use super::types::{ClassificationContext, ConsistencyConfidenceCalculator};

impl Default for ConsistencyConfidenceCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceCalculator for ConsistencyConfidenceCalculator {
    fn calculate_confidence(
        &self,
        pattern: &DetectedPattern,
        _context: &ClassificationContext,
    ) -> f64 {
        pattern.confidence * pattern.predictive_power
    }
    fn name(&self) -> &str {
        "Consistency Confidence Calculator"
    }
}
