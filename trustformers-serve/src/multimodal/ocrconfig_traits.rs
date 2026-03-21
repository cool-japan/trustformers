//! # OcrConfig - Trait Implementations
//!
//! This module contains trait implementations for `OcrConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{OcrConfig, OcrEngine};

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            engine: OcrEngine::Tesseract,
            languages: vec!["eng".to_string()],
            confidence_threshold: 0.6,
            preprocess_images: true,
            target_dpi: 300,
            auto_rotate: true,
            denoise: true,
        }
    }
}
