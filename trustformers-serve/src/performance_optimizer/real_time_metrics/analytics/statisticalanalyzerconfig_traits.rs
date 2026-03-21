//! # StatisticalAnalyzerConfig - Trait Implementations
//!
//! This module contains trait implementations for `StatisticalAnalyzerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{OutlierDetectionMethod, StatisticalAnalyzerConfig};

impl Default for StatisticalAnalyzerConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            enable_robust_stats: true,
            bootstrap_iterations: 1000,
            outlier_method: OutlierDetectionMethod::Iqr,
            calculation_precision: 1e-10,
        }
    }
}
