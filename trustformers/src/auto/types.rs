//! # Common Types for TrustformeRS Auto Module
//!
//! This module provides the foundational types and structures used across all components
//! of the TrustformeRS auto framework, including feature extractors, data collators,
//! metrics, and optimizers.
//!
//! ## Organization
//!
//! The types are organized into the following categories:
//!
//! - **Format Enums**: Define supported data formats for images and documents
//! - **Input/Output Types**: Core types for feature extraction pipelines
//! - **Common Structures**: Reusable structures like special tokens
//! - **Metadata Structures**: Rich metadata for different modalities
//! - **Data Collation Types**: Types for batch processing and padding strategies
//!
//! ## Usage
//!
//! These types are designed to be used throughout the TrustformeRS ecosystem:
//!
//! ```rust
//! use trustformers::auto::types::{FeatureInput, ImageFormat, ImageMetadata};
//!
//! let input = FeatureInput::Image {
//!     data: image_bytes,
//!     format: ImageFormat::Jpeg,
//!     metadata: Some(ImageMetadata {
//!         width: 640,
//!         height: 480,
//!         channels: 3,
//!         dpi: Some(96),
//!     }),
//! };
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Format Enums
// =============================================================================

/// Supported image formats for vision feature extractors
///
/// This enum defines the image formats that can be processed by vision
/// feature extractors in the TrustformeRS framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageFormat {
    /// JPEG image format
    Jpeg,
    /// PNG image format
    Png,
    /// WebP image format
    Webp,
    /// BMP image format
    Bmp,
    /// TIFF image format
    Tiff,
}

impl ImageFormat {
    /// Get the file extension for this image format
    pub fn extension(&self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "jpg",
            ImageFormat::Png => "png",
            ImageFormat::Webp => "webp",
            ImageFormat::Bmp => "bmp",
            ImageFormat::Tiff => "tiff",
        }
    }

    /// Get the MIME type for this image format
    pub fn mime_type(&self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "image/jpeg",
            ImageFormat::Png => "image/png",
            ImageFormat::Webp => "image/webp",
            ImageFormat::Bmp => "image/bmp",
            ImageFormat::Tiff => "image/tiff",
        }
    }
}

/// Supported document formats for document feature extractors
///
/// This enum defines the document formats that can be processed by document
/// feature extractors in the TrustformeRS framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DocumentFormat {
    /// PDF document format
    Pdf,
    /// HTML document format
    Html,
    /// Plain text format
    Text,
    /// Markdown format
    Markdown,
    /// Microsoft Word document format
    Docx,
    /// Image-based document format
    Image,
}

impl DocumentFormat {
    /// Get the file extension for this document format
    pub fn extension(&self) -> &'static str {
        match self {
            DocumentFormat::Pdf => "pdf",
            DocumentFormat::Html => "html",
            DocumentFormat::Text => "txt",
            DocumentFormat::Markdown => "md",
            DocumentFormat::Docx => "docx",
            DocumentFormat::Image => "png",
        }
    }

    /// Get the MIME type for this document format
    pub fn mime_type(&self) -> &'static str {
        match self {
            DocumentFormat::Pdf => "application/pdf",
            DocumentFormat::Html => "text/html",
            DocumentFormat::Text => "text/plain",
            DocumentFormat::Markdown => "text/markdown",
            DocumentFormat::Docx => {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            },
            DocumentFormat::Image => "image/png",
        }
    }
}

// =============================================================================
// Input/Output Types
// =============================================================================

/// Input types for feature extraction
///
/// This enum represents all possible input types that can be processed by
/// feature extractors in the TrustformeRS framework. Each variant contains
/// the raw data and associated metadata for proper processing.
#[derive(Debug, Clone)]
pub enum FeatureInput {
    /// Image input with binary data and format information
    Image {
        /// Raw image data as bytes
        data: Vec<u8>,
        /// Image format specification
        format: ImageFormat,
        /// Optional metadata about the image
        metadata: Option<ImageMetadata>,
    },
    /// Audio input with sample data and rate information
    Audio {
        /// Audio samples as floating-point values
        samples: Vec<f32>,
        /// Sample rate in Hz
        sample_rate: u32,
        /// Optional metadata about the audio
        metadata: Option<AudioMetadata>,
    },
    /// Text input with content string
    Text {
        /// Text content as a string
        content: String,
        /// Optional metadata about the text
        metadata: Option<TextMetadata>,
    },
    /// Document input with binary data and format information
    Document {
        /// Raw document data as bytes
        content: Vec<u8>,
        /// Document format specification
        format: DocumentFormat,
        /// Optional metadata about the document
        metadata: Option<DocumentMetadata>,
    },
    /// Multimodal input combining multiple modalities
    Multimodal {
        /// Collection of inputs from different modalities
        inputs: Vec<FeatureInput>,
        /// Optional metadata about the multimodal combination
        metadata: Option<MultimodalMetadata>,
    },
}

impl FeatureInput {
    /// Get the modality type of this input
    pub fn modality(&self) -> &'static str {
        match self {
            FeatureInput::Image { .. } => "image",
            FeatureInput::Audio { .. } => "audio",
            FeatureInput::Text { .. } => "text",
            FeatureInput::Document { .. } => "document",
            FeatureInput::Multimodal { .. } => "multimodal",
        }
    }

    /// Check if this input has metadata
    pub fn has_metadata(&self) -> bool {
        match self {
            FeatureInput::Image { metadata, .. } => metadata.is_some(),
            FeatureInput::Audio { metadata, .. } => metadata.is_some(),
            FeatureInput::Text { metadata, .. } => metadata.is_some(),
            FeatureInput::Document { metadata, .. } => metadata.is_some(),
            FeatureInput::Multimodal { metadata, .. } => metadata.is_some(),
        }
    }
}

/// Output from feature extraction operations
///
/// This structure contains the extracted features along with metadata about
/// the extraction process and any special tokens that were identified.
#[derive(Debug, Clone)]
pub struct FeatureOutput {
    /// Extracted feature vector as floating-point values
    pub features: Vec<f32>,
    /// Shape of the feature tensor [batch_size, seq_len, feature_dim, ...]
    pub shape: Vec<usize>,
    /// Additional metadata from the extraction process
    pub metadata: HashMap<String, serde_json::Value>,
    /// Optional attention mask for variable-length sequences
    pub attention_mask: Option<Vec<u32>>,
    /// Special tokens identified during extraction
    pub special_tokens: Vec<SpecialToken>,
}

impl FeatureOutput {
    /// Create a new FeatureOutput with basic parameters
    pub fn new(features: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            features,
            shape,
            metadata: HashMap::new(),
            attention_mask: None,
            special_tokens: Vec::new(),
        }
    }

    /// Get the total number of features
    pub fn feature_count(&self) -> usize {
        self.features.len()
    }

    /// Get the dimensionality of the features
    pub fn feature_dimension(&self) -> usize {
        self.shape.iter().product()
    }

    /// Add metadata to the output
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Add an attention mask to the output
    pub fn with_attention_mask(mut self, mask: Vec<u32>) -> Self {
        self.attention_mask = Some(mask);
        self
    }

    /// Add special tokens to the output
    pub fn with_special_tokens(mut self, tokens: Vec<SpecialToken>) -> Self {
        self.special_tokens = tokens;
        self
    }
}

// =============================================================================
// Common Structures
// =============================================================================

/// Represents special tokens identified during feature extraction
///
/// Special tokens are tokens with specific semantic meaning such as
/// classification tokens, separator tokens, or padding tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecialToken {
    /// Type of the special token (e.g., "CLS", "SEP", "PAD", "MASK")
    pub token_type: String,
    /// Position of the token in the sequence
    pub position: usize,
    /// String representation of the token
    pub value: String,
}

impl SpecialToken {
    /// Create a new special token
    pub fn new(token_type: impl Into<String>, position: usize, value: impl Into<String>) -> Self {
        Self {
            token_type: token_type.into(),
            position,
            value: value.into(),
        }
    }

    /// Check if this is a classification token
    pub fn is_cls_token(&self) -> bool {
        self.token_type.eq_ignore_ascii_case("cls")
    }

    /// Check if this is a separator token
    pub fn is_sep_token(&self) -> bool {
        self.token_type.eq_ignore_ascii_case("sep")
    }

    /// Check if this is a padding token
    pub fn is_pad_token(&self) -> bool {
        self.token_type.eq_ignore_ascii_case("pad")
    }

    /// Check if this is a mask token
    pub fn is_mask_token(&self) -> bool {
        self.token_type.eq_ignore_ascii_case("mask")
    }
}

// =============================================================================
// Metadata Structures
// =============================================================================

/// Metadata for image inputs
///
/// Contains information about image dimensions, format, and quality parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageMetadata {
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Number of color channels (e.g., 3 for RGB, 4 for RGBA)
    pub channels: u32,
    /// Optional dots per inch (DPI) information
    pub dpi: Option<u32>,
}

impl ImageMetadata {
    /// Create new image metadata
    pub fn new(width: u32, height: u32, channels: u32) -> Self {
        Self {
            width,
            height,
            channels,
            dpi: None,
        }
    }

    /// Get the total number of pixels
    pub fn pixel_count(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Get the aspect ratio
    pub fn aspect_ratio(&self) -> f64 {
        self.width as f64 / self.height as f64
    }

    /// Check if the image is grayscale
    pub fn is_grayscale(&self) -> bool {
        self.channels == 1
    }

    /// Check if the image has an alpha channel
    pub fn has_alpha(&self) -> bool {
        self.channels == 2 || self.channels == 4
    }
}

/// Metadata for audio inputs
///
/// Contains information about audio duration, quality, and format parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioMetadata {
    /// Duration of the audio in seconds
    pub duration: f64,
    /// Number of audio channels (1 for mono, 2 for stereo)
    pub channels: u32,
    /// Optional bit depth information
    pub bit_depth: Option<u32>,
}

impl AudioMetadata {
    /// Create new audio metadata
    pub fn new(duration: f64, channels: u32) -> Self {
        Self {
            duration,
            channels,
            bit_depth: None,
        }
    }

    /// Check if the audio is mono
    pub fn is_mono(&self) -> bool {
        self.channels == 1
    }

    /// Check if the audio is stereo
    pub fn is_stereo(&self) -> bool {
        self.channels == 2
    }

    /// Get the duration in milliseconds
    pub fn duration_ms(&self) -> f64 {
        self.duration * 1000.0
    }
}

/// Metadata for text inputs
///
/// Contains information about text language, encoding, and content statistics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextMetadata {
    /// Language of the text (ISO 639-1 code)
    pub language: Option<String>,
    /// Text encoding (e.g., "utf-8", "ascii")
    pub encoding: Option<String>,
    /// Number of words in the text
    pub word_count: Option<usize>,
}

impl TextMetadata {
    /// Create new text metadata
    pub fn new() -> Self {
        Self {
            language: None,
            encoding: None,
            word_count: None,
        }
    }

    /// Set the language
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set the encoding
    pub fn with_encoding(mut self, encoding: impl Into<String>) -> Self {
        self.encoding = Some(encoding.into());
        self
    }

    /// Set the word count
    pub fn with_word_count(mut self, count: usize) -> Self {
        self.word_count = Some(count);
        self
    }
}

impl Default for TextMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata for document inputs
///
/// Contains information about document structure, authorship, and content organization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Number of pages in the document
    pub page_count: Option<usize>,
    /// Document author
    pub author: Option<String>,
    /// Document title
    pub title: Option<String>,
    /// Document creation date (ISO 8601 format)
    pub creation_date: Option<String>,
}

impl DocumentMetadata {
    /// Create new document metadata
    pub fn new() -> Self {
        Self {
            page_count: None,
            author: None,
            title: None,
            creation_date: None,
        }
    }

    /// Set the page count
    pub fn with_page_count(mut self, count: usize) -> Self {
        self.page_count = Some(count);
        self
    }

    /// Set the author
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Set the title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the creation date
    pub fn with_creation_date(mut self, date: impl Into<String>) -> Self {
        self.creation_date = Some(date.into());
        self
    }
}

impl Default for DocumentMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata for multimodal inputs
///
/// Contains information about the combination of different modalities and fusion strategies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultimodalMetadata {
    /// List of modalities present in the input
    pub modalities: Vec<String>,
    /// Strategy used for fusing different modalities
    pub fusion_strategy: Option<String>,
}

impl MultimodalMetadata {
    /// Create new multimodal metadata
    pub fn new(modalities: Vec<String>) -> Self {
        Self {
            modalities,
            fusion_strategy: None,
        }
    }

    /// Set the fusion strategy
    pub fn with_fusion_strategy(mut self, strategy: impl Into<String>) -> Self {
        self.fusion_strategy = Some(strategy.into());
        self
    }

    /// Check if a specific modality is present
    pub fn has_modality(&self, modality: &str) -> bool {
        self.modalities.contains(&modality.to_string())
    }

    /// Get the number of modalities
    pub fn modality_count(&self) -> usize {
        self.modalities.len()
    }
}

// =============================================================================
// Data Collation Types
// =============================================================================

/// Padding strategies for data collation
///
/// Defines how sequences should be padded when creating batches of variable-length data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum PaddingStrategy {
    /// Do not apply any padding
    None,
    /// Pad to the length of the longest sequence in the batch
    #[default]
    Longest,
    /// Pad to a fixed maximum length
    MaxLength,
    /// Explicitly do not pad (same as None but more explicit)
    DoNotPad,
}

impl PaddingStrategy {
    /// Check if padding should be applied
    pub fn should_pad(&self) -> bool {
        matches!(self, PaddingStrategy::Longest | PaddingStrategy::MaxLength)
    }

    /// Check if dynamic padding is used
    pub fn is_dynamic(&self) -> bool {
        matches!(self, PaddingStrategy::Longest)
    }
}

/// Input example for data collation
///
/// Represents a single training/inference example that can be batched together
/// with other examples during data collation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataExample {
    /// Token IDs for the input sequence
    pub input_ids: Vec<u32>,
    /// Attention mask indicating which tokens should be attended to
    pub attention_mask: Option<Vec<u32>>,
    /// Token type IDs for distinguishing different segments
    pub token_type_ids: Option<Vec<u32>>,
    /// Labels for supervised learning tasks
    pub labels: Option<Vec<i64>>,
    /// Additional metadata for the example
    pub metadata: HashMap<String, serde_json::Value>,
}

impl DataExample {
    /// Create a new data example with basic input IDs
    pub fn new(input_ids: Vec<u32>) -> Self {
        Self {
            input_ids,
            attention_mask: None,
            token_type_ids: None,
            labels: None,
            metadata: HashMap::new(),
        }
    }

    /// Get the sequence length
    pub fn sequence_length(&self) -> usize {
        self.input_ids.len()
    }

    /// Add an attention mask
    pub fn with_attention_mask(mut self, mask: Vec<u32>) -> Self {
        self.attention_mask = Some(mask);
        self
    }

    /// Add token type IDs
    pub fn with_token_type_ids(mut self, type_ids: Vec<u32>) -> Self {
        self.token_type_ids = Some(type_ids);
        self
    }

    /// Add labels
    pub fn with_labels(mut self, labels: Vec<i64>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Check if the example has labels
    pub fn has_labels(&self) -> bool {
        self.labels.is_some()
    }
}

/// Collated batch output from data collation
///
/// Represents a batch of examples that have been collated together with
/// appropriate padding and alignment for efficient processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollatedBatch {
    /// Batched input IDs with shape [batch_size, sequence_length]
    pub input_ids: Vec<Vec<u32>>,
    /// Batched attention masks with shape [batch_size, sequence_length]
    pub attention_mask: Vec<Vec<u32>>,
    /// Optional token type IDs with shape [batch_size, sequence_length]
    pub token_type_ids: Option<Vec<Vec<u32>>>,
    /// Optional labels with shape [batch_size, num_labels]
    pub labels: Option<Vec<Vec<i64>>>,
    /// Number of examples in the batch
    pub batch_size: usize,
    /// Length of the padded sequences
    pub sequence_length: usize,
    /// Aggregated metadata from all examples
    pub metadata: HashMap<String, serde_json::Value>,
}

impl CollatedBatch {
    /// Create a new collated batch
    pub fn new(
        input_ids: Vec<Vec<u32>>,
        attention_mask: Vec<Vec<u32>>,
        batch_size: usize,
        sequence_length: usize,
    ) -> Self {
        Self {
            input_ids,
            attention_mask,
            token_type_ids: None,
            labels: None,
            batch_size,
            sequence_length,
            metadata: HashMap::new(),
        }
    }

    /// Get the total number of tokens in the batch
    pub fn total_tokens(&self) -> usize {
        self.batch_size * self.sequence_length
    }

    /// Get the shape of the input tensors
    pub fn input_shape(&self) -> (usize, usize) {
        (self.batch_size, self.sequence_length)
    }

    /// Check if the batch has token type IDs
    pub fn has_token_type_ids(&self) -> bool {
        self.token_type_ids.is_some()
    }

    /// Check if the batch has labels
    pub fn has_labels(&self) -> bool {
        self.labels.is_some()
    }

    /// Add token type IDs to the batch
    pub fn with_token_type_ids(mut self, token_type_ids: Vec<Vec<u32>>) -> Self {
        self.token_type_ids = Some(token_type_ids);
        self
    }

    /// Add labels to the batch
    pub fn with_labels(mut self, labels: Vec<Vec<i64>>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Add metadata to the batch
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

// =============================================================================
// Helper Functions and Utilities
// =============================================================================

/// Utility functions for working with common types
pub mod utils {
    use super::*;

    /// Calculate the total memory footprint of a FeatureOutput
    pub fn feature_output_memory_size(output: &FeatureOutput) -> usize {
        let features_size = output.features.len() * std::mem::size_of::<f32>();
        let shape_size = output.shape.len() * std::mem::size_of::<usize>();
        let attention_mask_size = output
            .attention_mask
            .as_ref()
            .map(|mask| mask.len() * std::mem::size_of::<u32>())
            .unwrap_or(0);
        let special_tokens_size = output.special_tokens.len() * std::mem::size_of::<SpecialToken>();

        features_size + shape_size + attention_mask_size + special_tokens_size
    }

    /// Calculate the total memory footprint of a CollatedBatch
    pub fn collated_batch_memory_size(batch: &CollatedBatch) -> usize {
        let input_ids_size = batch.batch_size * batch.sequence_length * std::mem::size_of::<u32>();
        let attention_mask_size =
            batch.batch_size * batch.sequence_length * std::mem::size_of::<u32>();
        let token_type_ids_size = batch
            .token_type_ids
            .as_ref()
            .map(|_| batch.batch_size * batch.sequence_length * std::mem::size_of::<u32>())
            .unwrap_or(0);
        let labels_size = batch
            .labels
            .as_ref()
            .map(|labels| {
                labels.iter().map(|l| l.len()).sum::<usize>() * std::mem::size_of::<i64>()
            })
            .unwrap_or(0);

        input_ids_size + attention_mask_size + token_type_ids_size + labels_size
    }

    /// Validate that a FeatureOutput has consistent dimensions
    pub fn validate_feature_output(output: &FeatureOutput) -> Result<(), String> {
        let expected_size: usize = output.shape.iter().product();
        if output.features.len() != expected_size {
            return Err(format!(
                "Feature vector size {} does not match shape {:?} (expected {})",
                output.features.len(),
                output.shape,
                expected_size
            ));
        }

        if let Some(mask) = &output.attention_mask {
            if !output.shape.is_empty() && mask.len() != output.shape[0] {
                return Err(format!(
                    "Attention mask length {} does not match first dimension of shape {:?}",
                    mask.len(),
                    output.shape
                ));
            }
        }

        Ok(())
    }

    /// Validate that a CollatedBatch has consistent dimensions
    pub fn validate_collated_batch(batch: &CollatedBatch) -> Result<(), String> {
        if batch.input_ids.len() != batch.batch_size {
            return Err(format!(
                "Input IDs batch size {} does not match expected batch size {}",
                batch.input_ids.len(),
                batch.batch_size
            ));
        }

        if batch.attention_mask.len() != batch.batch_size {
            return Err(format!(
                "Attention mask batch size {} does not match expected batch size {}",
                batch.attention_mask.len(),
                batch.batch_size
            ));
        }

        for (i, input_ids) in batch.input_ids.iter().enumerate() {
            if input_ids.len() != batch.sequence_length {
                return Err(format!(
                    "Input IDs sequence {} has length {} but expected {}",
                    i,
                    input_ids.len(),
                    batch.sequence_length
                ));
            }
        }

        for (i, attention_mask) in batch.attention_mask.iter().enumerate() {
            if attention_mask.len() != batch.sequence_length {
                return Err(format!(
                    "Attention mask sequence {} has length {} but expected {}",
                    i,
                    attention_mask.len(),
                    batch.sequence_length
                ));
            }
        }

        if let Some(token_type_ids) = &batch.token_type_ids {
            if token_type_ids.len() != batch.batch_size {
                return Err(format!(
                    "Token type IDs batch size {} does not match expected batch size {}",
                    token_type_ids.len(),
                    batch.batch_size
                ));
            }

            for (i, type_ids) in token_type_ids.iter().enumerate() {
                if type_ids.len() != batch.sequence_length {
                    return Err(format!(
                        "Token type IDs sequence {} has length {} but expected {}",
                        i,
                        type_ids.len(),
                        batch.sequence_length
                    ));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_format_properties() {
        assert_eq!(ImageFormat::Jpeg.extension(), "jpg");
        assert_eq!(ImageFormat::Jpeg.mime_type(), "image/jpeg");
        assert_eq!(ImageFormat::Png.extension(), "png");
        assert_eq!(ImageFormat::Png.mime_type(), "image/png");
    }

    #[test]
    fn test_document_format_properties() {
        assert_eq!(DocumentFormat::Pdf.extension(), "pdf");
        assert_eq!(DocumentFormat::Pdf.mime_type(), "application/pdf");
        assert_eq!(DocumentFormat::Html.extension(), "html");
        assert_eq!(DocumentFormat::Html.mime_type(), "text/html");
    }

    #[test]
    fn test_feature_input_modality() {
        let image_input = FeatureInput::Image {
            data: vec![1, 2, 3],
            format: ImageFormat::Jpeg,
            metadata: None,
        };
        assert_eq!(image_input.modality(), "image");

        let text_input = FeatureInput::Text {
            content: "Hello world".to_string(),
            metadata: None,
        };
        assert_eq!(text_input.modality(), "text");
    }

    #[test]
    fn test_special_token_creation() {
        let token = SpecialToken::new("CLS", 0, "[CLS]");
        assert_eq!(token.token_type, "CLS");
        assert_eq!(token.position, 0);
        assert_eq!(token.value, "[CLS]");
        assert!(token.is_cls_token());
        assert!(!token.is_sep_token());
    }

    #[test]
    fn test_image_metadata_properties() {
        let metadata = ImageMetadata::new(640, 480, 3);
        assert_eq!(metadata.pixel_count(), 307200);
        assert!((metadata.aspect_ratio() - 1.333333).abs() < 0.000001);
        assert!(!metadata.is_grayscale());
        assert!(!metadata.has_alpha());

        let grayscale = ImageMetadata::new(100, 100, 1);
        assert!(grayscale.is_grayscale());

        let rgba = ImageMetadata::new(100, 100, 4);
        assert!(rgba.has_alpha());
    }

    #[test]
    fn test_audio_metadata_properties() {
        let metadata = AudioMetadata::new(5.5, 2);
        assert!(metadata.is_stereo());
        assert!(!metadata.is_mono());
        assert_eq!(metadata.duration_ms(), 5500.0);

        let mono = AudioMetadata::new(3.0, 1);
        assert!(mono.is_mono());
        assert!(!mono.is_stereo());
    }

    #[test]
    fn test_padding_strategy_properties() {
        assert!(PaddingStrategy::Longest.should_pad());
        assert!(PaddingStrategy::MaxLength.should_pad());
        assert!(!PaddingStrategy::None.should_pad());
        assert!(!PaddingStrategy::DoNotPad.should_pad());

        assert!(PaddingStrategy::Longest.is_dynamic());
        assert!(!PaddingStrategy::MaxLength.is_dynamic());
    }

    #[test]
    fn test_data_example_creation() {
        let example = DataExample::new(vec![1, 2, 3, 4])
            .with_attention_mask(vec![1, 1, 1, 1])
            .with_labels(vec![0]);

        assert_eq!(example.sequence_length(), 4);
        assert!(example.has_labels());
        assert_eq!(example.attention_mask, Some(vec![1, 1, 1, 1]));
    }

    #[test]
    fn test_collated_batch_properties() {
        let batch = CollatedBatch::new(
            vec![vec![1, 2, 3], vec![4, 5, 6]],
            vec![vec![1, 1, 1], vec![1, 1, 1]],
            2,
            3,
        );

        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.sequence_length, 3);
        assert_eq!(batch.total_tokens(), 6);
        assert_eq!(batch.input_shape(), (2, 3));
        assert!(!batch.has_labels());
        assert!(!batch.has_token_type_ids());
    }

    #[test]
    fn test_feature_output_validation() {
        let output = FeatureOutput::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(utils::validate_feature_output(&output).is_ok());

        let invalid_output = FeatureOutput::new(vec![1.0, 2.0, 3.0], vec![2, 2]);
        assert!(utils::validate_feature_output(&invalid_output).is_err());
    }

    #[test]
    fn test_collated_batch_validation() {
        let valid_batch = CollatedBatch::new(
            vec![vec![1, 2], vec![3, 4]],
            vec![vec![1, 1], vec![1, 1]],
            2,
            2,
        );
        assert!(utils::validate_collated_batch(&valid_batch).is_ok());

        let invalid_batch = CollatedBatch::new(
            vec![vec![1, 2, 3], vec![4, 5]],
            vec![vec![1, 1], vec![1, 1]],
            2,
            2,
        );
        assert!(utils::validate_collated_batch(&invalid_batch).is_err());
    }

    #[test]
    fn test_multimodal_metadata() {
        let metadata = MultimodalMetadata::new(vec!["text".to_string(), "image".to_string()])
            .with_fusion_strategy("late_fusion");

        assert_eq!(metadata.modality_count(), 2);
        assert!(metadata.has_modality("text"));
        assert!(metadata.has_modality("image"));
        assert!(!metadata.has_modality("audio"));
        assert_eq!(metadata.fusion_strategy, Some("late_fusion".to_string()));
    }
}
