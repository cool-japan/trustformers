//! # CompressionEngine - Trait Implementations
//!
//! This module contains trait implementations for `CompressionEngine`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::CompressionEngine;

impl fmt::Debug for CompressionEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let algorithm_count = self.compression_algorithms.len();
        let strategy_count = self
            .compression_strategies
            .try_read()
            .map(|strategies| strategies.len())
            .unwrap_or(0);
        f.debug_struct("CompressionEngine")
            .field("algorithm_count", &algorithm_count)
            .field("strategy_count", &strategy_count)
            .field("compression_scheduler", &self.compression_scheduler)
            .field("compression_statistics", &self.compression_statistics)
            .finish()
    }
}
