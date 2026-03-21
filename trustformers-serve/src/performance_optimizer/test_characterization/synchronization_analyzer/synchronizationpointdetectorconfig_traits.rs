//! # SynchronizationPointDetectorConfig - Trait Implementations
//!
//! This module contains trait implementations for `SynchronizationPointDetectorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SynchronizationPointDetectorConfig;

impl Default for SynchronizationPointDetectorConfig {
    fn default() -> Self {
        Self {
            barrier_detection_sensitivity: 0.85,
            producer_consumer_threshold: 0.80,
            reader_writer_threshold: 0.75,
            custom_pattern_detection: true,
            bottleneck_threshold: 0.70,
            detection_window_size: 1000,
        }
    }
}
