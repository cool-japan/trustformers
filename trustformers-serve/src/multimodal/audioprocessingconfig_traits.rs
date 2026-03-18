//! # AudioProcessingConfig - Trait Implementations
//!
//! This module contains trait implementations for `AudioProcessingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AudioProcessingConfig;

impl Default for AudioProcessingConfig {
    fn default() -> Self {
        Self {
            max_duration_seconds: 600,
            target_sample_rate: 16000,
            target_bit_rate: 128000,
            target_channels: 1,
            normalize_audio: true,
            noise_reduction: false,
            auto_convert: false,
            target_format: None,
            silence_detection: true,
            silence_threshold: -40.0,
        }
    }
}
