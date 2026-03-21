//! # MultiModalConfig - Trait Implementations
//!
//! This module contains trait implementations for `MultiModalConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    AudioFormat, AudioProcessingConfig, ContentValidationConfig, DocumentFormat,
    DocumentProcessingConfig, ImageFormat, ImageProcessingConfig, MultiModalConfig, StorageConfig,
    VideoFormat, VideoProcessingConfig,
};

impl Default for MultiModalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_file_size: 50 * 1024 * 1024,
            supported_image_formats: vec![
                ImageFormat::JPEG,
                ImageFormat::PNG,
                ImageFormat::WebP,
                ImageFormat::GIF,
                ImageFormat::BMP,
            ],
            supported_audio_formats: vec![
                AudioFormat::WAV,
                AudioFormat::MP3,
                AudioFormat::FLAC,
                AudioFormat::OGG,
                AudioFormat::AAC,
            ],
            supported_video_formats: vec![
                VideoFormat::MP4,
                VideoFormat::AVI,
                VideoFormat::MOV,
                VideoFormat::WebM,
                VideoFormat::MKV,
            ],
            supported_document_formats: vec![
                DocumentFormat::PDF,
                DocumentFormat::DOCX,
                DocumentFormat::TXT,
                DocumentFormat::RTF,
                DocumentFormat::HTML,
                DocumentFormat::MD,
                DocumentFormat::ODT,
                DocumentFormat::PPTX,
                DocumentFormat::XLSX,
            ],
            image_processing: ImageProcessingConfig::default(),
            audio_processing: AudioProcessingConfig::default(),
            video_processing: VideoProcessingConfig::default(),
            document_processing: DocumentProcessingConfig::default(),
            content_validation: ContentValidationConfig::default(),
            storage_config: StorageConfig::default(),
            processing_timeout_seconds: 300,
        }
    }
}
