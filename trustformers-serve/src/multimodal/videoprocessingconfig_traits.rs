//! # VideoProcessingConfig - Trait Implementations
//!
//! This module contains trait implementations for `VideoProcessingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::VideoProcessingConfig;

impl Default for VideoProcessingConfig {
    fn default() -> Self {
        Self {
            max_duration_seconds: 600,
            max_resolution: (1920, 1080),
            target_resolution: Some((720, 480)),
            target_fps: 24.0,
            quality: 75,
            auto_convert: false,
            target_format: None,
            extract_keyframes: true,
            keyframe_interval: 1.0,
            generate_preview: true,
            preview_duration: 10.0,
        }
    }
}
