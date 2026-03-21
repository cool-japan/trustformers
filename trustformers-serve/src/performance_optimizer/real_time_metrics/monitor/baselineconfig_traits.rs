//! # BaselineConfig - Trait Implementations
//!
//! This module contains trait implementations for `BaselineConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

use super::types::BaselineConfig;

impl Default for BaselineConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_secs(3600),
            min_samples: 100,
            confidence_level: 0.95,
            adaptation_rate: 0.1,
            validation_threshold: 0.8,
            auto_refresh: true,
        }
    }
}
