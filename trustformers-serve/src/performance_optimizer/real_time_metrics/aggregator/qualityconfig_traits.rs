//! # QualityConfig - Trait Implementations
//!
//! This module contains trait implementations for `QualityConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::QualityConfig;

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            enable_quality_scoring: true,
            enable_outlier_detection: true,
            outlier_threshold: 3.0,
            quality_threshold: 0.8,
            validation_rules: Vec::new(),
        }
    }
}
