//! # IndependenceAnalysisConfig - Trait Implementations
//!
//! This module contains trait implementations for `IndependenceAnalysisConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::IndependenceAnalysisConfig;

impl Default for IndependenceAnalysisConfig {
    fn default() -> Self {
        Self {
            auto_dependency_detection: true,
            resource_conflict_detection: true,
            test_ordering_optimization: true,
            dependency_analysis_depth: 5,
            cache_analysis_results: true,
            analysis_cache_ttl: Duration::from_secs(3600),
        }
    }
}
