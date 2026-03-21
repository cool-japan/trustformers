//! # ContentValidationConfig - Trait Implementations
//!
//! This module contains trait implementations for `ContentValidationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ContentValidationConfig;

impl Default for ContentValidationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scan_inappropriate_content: true,
            scan_malware: false,
            allowed_mime_types: vec![
                "image/jpeg".to_string(),
                "image/png".to_string(),
                "image/webp".to_string(),
                "audio/wav".to_string(),
                "audio/mpeg".to_string(),
                "video/mp4".to_string(),
                "application/pdf".to_string(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    .to_string(),
                "application/msword".to_string(),
                "text/plain".to_string(),
                "text/rtf".to_string(),
                "text/html".to_string(),
                "text/markdown".to_string(),
                "application/vnd.oasis.opendocument.text".to_string(),
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    .to_string(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet".to_string(),
                "text/csv".to_string(),
                "application/json".to_string(),
                "application/xml".to_string(),
                "text/xml".to_string(),
            ],
            safety_model: None,
            safety_threshold: 0.8,
            content_fingerprinting: true,
            block_duplicates: false,
        }
    }
}
