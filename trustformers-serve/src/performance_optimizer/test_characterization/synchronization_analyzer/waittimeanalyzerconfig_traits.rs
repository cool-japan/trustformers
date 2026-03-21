//! # WaitTimeAnalyzerConfig - Trait Implementations
//!
//! This module contains trait implementations for `WaitTimeAnalyzerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::WaitTimeAnalyzerConfig;

impl Default for WaitTimeAnalyzerConfig {
    fn default() -> Self {
        Self {
            measurement_precision_us: 10,
            hotspot_threshold: 0.75,
            queue_window_size: 200,
            fairness_assessment: true,
            optimization_recommendations: true,
            history_depth: 50,
        }
    }
}
