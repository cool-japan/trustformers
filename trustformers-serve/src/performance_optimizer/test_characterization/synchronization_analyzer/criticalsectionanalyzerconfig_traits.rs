//! # CriticalSectionAnalyzerConfig - Trait Implementations
//!
//! This module contains trait implementations for `CriticalSectionAnalyzerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::CriticalSectionAnalyzerConfig;

impl Default for CriticalSectionAnalyzerConfig {
    fn default() -> Self {
        Self {
            min_section_duration_us: 100,
            contention_threshold: 0.75,
            optimization_threshold: 0.65,
            granularity_analysis: true,
            performance_impact_threshold: 0.10,
            history_depth: 100,
        }
    }
}
