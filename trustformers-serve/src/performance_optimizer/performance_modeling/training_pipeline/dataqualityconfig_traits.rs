//! # DataQualityConfig - Trait Implementations
//!
//! This module contains trait implementations for `DataQualityConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::DataQualityConfig;

impl Default for DataQualityConfig {
    fn default() -> Self {
        Self {
            min_sample_size: 20,
            max_missing_percentage: 0.1,
            enable_duplicate_detection: true,
            correlation_threshold: 0.95,
        }
    }
}
