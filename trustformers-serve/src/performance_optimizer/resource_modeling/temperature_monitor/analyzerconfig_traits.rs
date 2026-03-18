//! # AnalyzerConfig - Trait Implementations
//!
//! This module contains trait implementations for `AnalyzerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::AnalyzerConfig;

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            analysis_interval: Duration::from_secs(60),
        }
    }
}
