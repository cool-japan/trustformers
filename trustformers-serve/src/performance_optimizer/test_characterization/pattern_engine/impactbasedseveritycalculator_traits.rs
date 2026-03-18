//! # ImpactBasedSeverityCalculator - Trait Implementations
//!
//! This module contains trait implementations for `ImpactBasedSeverityCalculator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `SeverityCalculator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::SeverityCalculator;
use super::types::{DetectedAntiPattern, ImpactBasedSeverityCalculator, SeverityContext};

impl Default for ImpactBasedSeverityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl SeverityCalculator for ImpactBasedSeverityCalculator {
    fn calculate_severity(
        &self,
        anti_pattern: &DetectedAntiPattern,
        _context: &SeverityContext,
    ) -> f64 {
        anti_pattern.severity * anti_pattern.potential_impact.overall_impact
    }
    fn name(&self) -> &str {
        "Impact-Based Severity Calculator"
    }
}
