//! # ProcessingOptions - Trait Implementations
//!
//! This module contains trait implementations for `ProcessingOptions`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ProcessingOptions, ProcessingPriority};

impl Default for ProcessingOptions {
    fn default() -> Self {
        Self {
            preprocess: true,
            return_processed_media: false,
            include_metadata: true,
            priority: ProcessingPriority::Normal,
            custom_pipeline: None,
        }
    }
}
