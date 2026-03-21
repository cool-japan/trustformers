//! # DocumentProcessingConfig - Trait Implementations
//!
//! This module contains trait implementations for `DocumentProcessingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{DocumentProcessingConfig, OcrConfig, TextPreprocessingConfig};

impl Default for DocumentProcessingConfig {
    fn default() -> Self {
        Self {
            max_file_size: 100 * 1024 * 1024,
            max_pages: 1000,
            extract_text: true,
            extract_metadata: true,
            extract_tables: true,
            extract_images: false,
            ocr_config: OcrConfig::default(),
            text_preprocessing: TextPreprocessingConfig::default(),
            analyze_structure: true,
            semantic_analysis: false,
            detect_language: true,
            convert_to_format: None,
            enable_summarization: false,
            summarization_model: None,
            max_summary_length: 500,
        }
    }
}
