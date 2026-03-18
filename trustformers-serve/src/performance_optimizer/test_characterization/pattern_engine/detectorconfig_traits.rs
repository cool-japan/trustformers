//! # DetectorConfig - Trait Implementations
//!
//! This module contains trait implementations for `DetectorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::Duration;

use super::types::DetectorConfig;

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 0.7,
            max_processing_time: Duration::from_secs(5),
            cache_results: true,
            priority: 1,
            parameters: HashMap::new(),
        }
    }
}
