//! # ImageProcessingConfig - Trait Implementations
//!
//! This module contains trait implementations for `ImageProcessingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ImageProcessingConfig;

impl Default for ImageProcessingConfig {
    fn default() -> Self {
        Self {
            max_dimensions: (4096, 4096),
            auto_resize: true,
            resize_dimensions: Some((1024, 1024)),
            quality: 85,
            auto_convert: false,
            target_format: None,
            strip_metadata: true,
            generate_thumbnails: true,
            thumbnail_size: (256, 256),
        }
    }
}
