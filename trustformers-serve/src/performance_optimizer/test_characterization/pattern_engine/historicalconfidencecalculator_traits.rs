//! # HistoricalConfidenceCalculator - Trait Implementations
//!
//! This module contains trait implementations for `HistoricalConfidenceCalculator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `ConfidenceCalculator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::test_characterization::types::DetectedPattern;

use super::functions::ConfidenceCalculator;
use super::types::{ClassificationContext, HistoricalConfidenceCalculator};

impl Default for HistoricalConfidenceCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceCalculator for HistoricalConfidenceCalculator {
    fn calculate_confidence(
        &self,
        pattern: &DetectedPattern,
        _context: &ClassificationContext,
    ) -> f64 {
        pattern.confidence * pattern.stability
    }
    fn name(&self) -> &str {
        "Historical Confidence Calculator"
    }
}
