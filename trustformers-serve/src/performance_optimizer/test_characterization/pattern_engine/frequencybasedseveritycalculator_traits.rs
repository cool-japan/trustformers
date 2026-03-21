//! # FrequencyBasedSeverityCalculator - Trait Implementations
//!
//! This module contains trait implementations for `FrequencyBasedSeverityCalculator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `SeverityCalculator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::SeverityCalculator;
use super::types::{DetectedAntiPattern, FrequencyBasedSeverityCalculator, SeverityContext};

impl Default for FrequencyBasedSeverityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl SeverityCalculator for FrequencyBasedSeverityCalculator {
    fn calculate_severity(
        &self,
        anti_pattern: &DetectedAntiPattern,
        _context: &SeverityContext,
    ) -> f64 {
        anti_pattern.severity * 1.2
    }
    fn name(&self) -> &str {
        "Frequency-Based Severity Calculator"
    }
}
