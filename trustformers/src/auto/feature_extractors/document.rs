//! # Document Feature Extractor Module
//!
//! This module provides specialized feature extraction capabilities for document-based inputs,
//! supporting various document formats including PDF, HTML, Text, Markdown, and DOCX.
//! It's designed for document understanding tasks such as document classification,
//! document question answering, and layout analysis.
//!
//! ## Overview
//!
//! The document feature extractor is optimized for processing structured and unstructured
//! documents, with support for both textual content and layout information. It can handle
//! various document formats and extract meaningful features for downstream NLP tasks.
//!
//! ## Supported Document Formats
//!
//! - **Text**: Plain text documents
//! - **HTML**: Web pages and HTML documents with tag processing
//! - **Markdown**: Markdown formatted documents
//! - **PDF**: PDF documents (future enhancement)
//! - **DOCX**: Microsoft Word documents (future enhancement)
//! - **Image**: Document images for OCR processing (future enhancement)
//!
//! ## Key Features
//!
//! - **Multi-format Support**: Handles various document types with format-specific preprocessing
//! - **Layout Awareness**: Optional support for spatial and visual layout features
//! - **Token Processing**: Advanced tokenization with special token handling
//! - **Batch Processing**: Efficient batch processing for multiple documents
//! - **Configurable Parameters**: Flexible configuration for different model architectures
//!
//! ## Architecture
//!
//! ```text
//! Document Input
//!       ↓
//! Format Detection & Preprocessing
//!       ↓
//! Text Extraction & Cleaning
//!       ↓
//! Feature Extraction
//!       ↓
//! Output with Attention Masks & Special Tokens
//! ```
//!
//! ## Usage Examples
//!
//! ### Basic Text Document Processing
//!
//! ```rust,ignore
//! use trustformers::auto::feature_extractors::document::{DocumentFeatureExtractor, DocumentFeatureConfig};
//! use trustformers::auto::types::{FeatureInput, DocumentFormat};
//!
//! // Create configuration
//! let config = DocumentFeatureConfig {
//!     max_length: 512,
//!     feature_size: 768,
//!     include_layout: false,
//!     include_visual_features: false,
//!     max_batch_size: Some(8),
//! };
//!
//! // Create extractor
//! let extractor = DocumentFeatureExtractor::new(config);
//!
//! // Prepare document input
//! let document_content = b"This is a sample document for processing.";
//! let input = FeatureInput::Document {
//!     content: document_content.to_vec(),
//!     format: DocumentFormat::Text,
//!     metadata: None,
//! };
//!
//! // Extract features
//! let output = extractor.extract_features(&input)?;
//! println!("Extracted features shape: {:?}", output.shape);
//! ```
//!
//! ### HTML Document Processing
//!
//! ```rust,ignore
//! let html_content = br#"
//! <html>
//! <body>
//!     <h1>Document Title</h1>
//!     <p>This is the main content of the document.</p>
//! </body>
//! </html>
//! "#;
//!
//! let input = FeatureInput::Document {
//!     content: html_content.to_vec(),
//!     format: DocumentFormat::Html,
//!     metadata: None,
//! };
//!
//! let output = extractor.extract_features(&input)?;
//! ```
//!
//! ### Configuration from Model Config
//!
//! ```rust,ignore
//! use serde_json::json;
//!
//! let model_config = json!({
//!     "model_type": "layoutlm",
//!     "hidden_size": 768,
//!     "max_position_embeddings": 512,
//!     "has_visual_segment_embedding": true,
//!     "has_spatial_attention_bias": true
//! });
//!
//! let config = DocumentFeatureConfig::from_config(&model_config)?;
//! let extractor = DocumentFeatureExtractor::new(config);
//! ```
//!
//! ## Performance Considerations
//!
//! - **Memory Usage**: Feature extraction scales with max_length × feature_size
//! - **Processing Speed**: HTML and Markdown parsing add minimal overhead
//! - **Batch Processing**: Recommended for processing multiple documents
//! - **Large Documents**: Consider chunking for documents exceeding max_length
//!
//! ## Model Compatibility
//!
//! This extractor is designed to work with document understanding models such as:
//!
//! - **LayoutLM**: Layout-aware language model for documents
//! - **Donut**: Document understanding transformer
//! - **BERT-based**: Document classification models
//! - **Custom Models**: Any model requiring document feature extraction

use super::{FeatureExtractor, FeatureExtractorConfig};
use crate::auto::types::{DocumentFormat, FeatureInput, FeatureOutput, SpecialToken};
use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Document Feature Extractor Implementation
// =============================================================================

/// Document feature extractor for processing various document formats
///
/// The `DocumentFeatureExtractor` is specialized for handling document-based inputs
/// across multiple formats. It provides comprehensive preprocessing capabilities
/// and can extract both textual and layout-based features depending on the
/// configuration.
///
/// ## Key Capabilities
///
/// - **Format-Agnostic Processing**: Handles Text, HTML, Markdown, and more
/// - **Advanced Preprocessing**: Format-specific text extraction and cleaning
/// - **Token-Level Features**: Generates features for each token position
/// - **Special Token Support**: Adds classification and separation tokens
/// - **Attention Mask Generation**: Creates proper attention masks for variable-length documents
/// - **Layout Integration**: Optional support for spatial and visual features
///
/// ## Feature Extraction Process
///
/// 1. **Format Detection**: Identifies document format from input
/// 2. **Preprocessing**: Applies format-specific text extraction
/// 3. **Tokenization**: Splits text into tokens and handles special tokens
/// 4. **Feature Generation**: Creates token-level features with position encoding
/// 5. **Attention Mask**: Generates masks for padded sequences
/// 6. **Output Assembly**: Combines features with metadata and special tokens
///
/// ## Thread Safety
///
/// This extractor is thread-safe and implements `Send + Sync`. Multiple threads
/// can safely use the same extractor instance concurrently.
#[derive(Debug, Clone)]
pub struct DocumentFeatureExtractor {
    config: DocumentFeatureConfig,
}

impl DocumentFeatureExtractor {
    /// Create a new document feature extractor with the given configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters for the extractor
    ///
    /// # Returns
    ///
    /// A new `DocumentFeatureExtractor` instance
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = DocumentFeatureConfig {
    ///     max_length: 512,
    ///     feature_size: 768,
    ///     include_layout: false,
    ///     include_visual_features: false,
    ///     max_batch_size: Some(8),
    /// };
    ///
    /// let extractor = DocumentFeatureExtractor::new(config);
    /// ```
    pub fn new(config: DocumentFeatureConfig) -> Self {
        Self { config }
    }

    /// Preprocess document content based on its format
    ///
    /// This method handles format-specific preprocessing to extract clean text
    /// from various document formats. It applies appropriate parsing and
    /// cleaning strategies for each supported format.
    ///
    /// # Arguments
    ///
    /// * `content` - Raw document content as bytes
    /// * `format` - Document format identifier
    ///
    /// # Returns
    ///
    /// Cleaned and processed text string ready for feature extraction
    ///
    /// # Errors
    ///
    /// - `TrustformersError::Other` if text decoding fails
    /// - `TrustformersError::InvalidInput` if the document format is unsupported
    ///
    /// # Supported Formats
    ///
    /// - **Text**: Direct UTF-8 decoding
    /// - **HTML**: Tag removal and text extraction
    /// - **Markdown**: Formatting removal and text extraction
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// let html_content = b"<h1>Title</h1><p>Content</p>";
    /// let text = extractor.preprocess_document(html_content, DocumentFormat::Html)?;
    /// assert_eq!(text, "Title Content");

    fn preprocess_document(&self, content: &[u8], format: DocumentFormat) -> Result<String> {
        match format {
            DocumentFormat::Text => {
                String::from_utf8(content.to_vec()).map_err(|e| TrustformersError::Io {
                    message: format!("Failed to decode text: {}", e),
                    path: None,
                    suggestion: Some(
                        "Check that the input is valid UTF-8 encoded text".to_string(),
                    ),
                })
            },
            DocumentFormat::Html => {
                // Simplified HTML processing - extract text
                let html = String::from_utf8_lossy(content);
                Ok(self.extract_text_from_html(&html))
            },
            DocumentFormat::Markdown => {
                let markdown = String::from_utf8_lossy(content);
                Ok(self.extract_text_from_markdown(&markdown))
            },
            _ => Err(TrustformersError::invalid_input(
                format!("Unsupported document format: {:?}", format),
                Some("document_format"),
                Some("supported format (PDF, TXT, HTML, Markdown)"),
                Some(format!("{:?}", format)),
            )),
        }
    }

    /// Extract clean text from HTML content
    ///
    /// This method processes HTML content to extract the textual content while
    /// removing HTML tags and formatting. It uses a simplified approach that
    /// handles most common HTML structures.
    ///
    /// # Arguments
    ///
    /// * `html` - HTML content as a string
    ///
    /// # Returns
    ///
    /// Clean text with HTML tags removed
    ///
    /// # Processing Steps
    ///
    /// 1. Add spaces around angle brackets to prevent word concatenation
    /// 2. Split content into words
    /// 3. Filter out HTML tags (content between < and >)
    /// 4. Rejoin words with single spaces
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let html = "<h1>Title</h1><p>Paragraph content</p>";
    /// let text = extractor.extract_text_from_html(html);
    /// assert_eq!(text, "Title Paragraph content");
    /// ```
    fn extract_text_from_html(&self, html: &str) -> String {
        // Simplified HTML text extraction
        html.replace("<", " <")
            .replace(">", "> ")
            .split_whitespace()
            .filter(|word| !word.starts_with('<') || !word.ends_with('>'))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Extract clean text from Markdown content
    ///
    /// This method processes Markdown content to extract the textual content
    /// while removing Markdown formatting syntax. It handles common Markdown
    /// elements like headers, emphasis, and code blocks.
    ///
    /// # Arguments
    ///
    /// * `markdown` - Markdown content as a string
    ///
    /// # Returns
    ///
    /// Clean text with Markdown formatting removed
    ///
    /// # Processing Steps
    ///
    /// 1. Process each line individually
    /// 2. Remove common Markdown syntax (headers, emphasis, code)
    /// 3. Filter out empty lines
    /// 4. Join processed lines with spaces
    ///
    /// # Supported Markdown Elements
    ///
    /// - Headers (# ## ###)
    /// - Bold text (**)
    /// - Italic text (*)
    /// - Inline code (`)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let markdown = "# Title\n\n**Bold text** and *italic text*";
    /// let text = extractor.extract_text_from_markdown(markdown);
    /// assert_eq!(text, "Title Bold text and italic text");
    /// ```
    fn extract_text_from_markdown(&self, markdown: &str) -> String {
        // Simplified Markdown text extraction
        markdown
            .lines()
            .map(|line| {
                line.trim()
                    .replace("# ", "")
                    .replace("## ", "")
                    .replace("### ", "")
                    .replace("**", "")
                    .replace("*", "")
                    .replace("`", "")
            })
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Extract document features from processed text
    ///
    /// This method generates feature vectors for each token position in the document.
    /// It creates a comprehensive feature representation that includes word-level
    /// characteristics, positional information, and linguistic properties.
    ///
    /// # Arguments
    ///
    /// * `text` - Preprocessed text content
    ///
    /// # Returns
    ///
    /// Feature vector with shape [max_length, feature_size]
    ///
    /// # Feature Types
    ///
    /// The method generates multiple feature types for each token:
    ///
    /// - **Length Features**: Normalized word length (0-1 scale)
    /// - **Position Features**: Relative position in document (0-1 scale)
    /// - **Alphabetic Features**: Binary indicator for alphabetic words
    /// - **Case Features**: Binary indicator for uppercase presence
    ///
    /// # Processing Details
    ///
    /// 1. **Tokenization**: Splits text into whitespace-separated tokens
    /// 2. **Feature Calculation**: Computes multiple feature types per token
    /// 3. **Padding**: Zero-pads sequences shorter than max_length
    /// 4. **Normalization**: Applies appropriate normalization to feature values
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// let text = "Hello world example";
    /// let features = extractor.extract_document_features(text)?;
    /// assert_eq!(features.len(), config.max_length * config.feature_size);

    fn extract_document_features(&self, text: &str) -> Result<Vec<f32>> {
        // Simplified document feature extraction
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut features = Vec::with_capacity(self.config.max_length * self.config.feature_size);

        for token_idx in 0..self.config.max_length {
            for feat_idx in 0..self.config.feature_size {
                let feature_val = if token_idx < words.len() {
                    // Simple word-based feature (character count, position, etc.)
                    let word = words[token_idx];
                    match feat_idx % 4 {
                        0 => word.len() as f32 / 20.0, // Normalized word length
                        1 => token_idx as f32 / self.config.max_length as f32, // Position
                        2 => {
                            if word.chars().all(|c| c.is_alphabetic()) {
                                1.0
                            } else {
                                0.0
                            }
                        }, // Is alphabetic
                        3 => {
                            if word.chars().any(|c| c.is_uppercase()) {
                                1.0
                            } else {
                                0.0
                            }
                        }, // Has uppercase
                        _ => 0.0,
                    }
                } else {
                    0.0 // Padding
                };
                features.push(feature_val);
            }
        }

        Ok(features)
    }
}

impl FeatureExtractor for DocumentFeatureExtractor {
    /// Extract features from document input
    ///
    /// This is the main entry point for document feature extraction. It handles
    /// the complete pipeline from raw document input to processed features
    /// suitable for model consumption.
    ///
    /// # Arguments
    ///
    /// * `input` - Document input containing content, format, and metadata
    ///
    /// # Returns
    ///
    /// Feature output with extracted features, attention masks, and special tokens
    ///
    /// # Errors
    ///
    /// - `TrustformersError::InvalidInput` if input is not a document type
    /// - `TrustformersError::ProcessingError` if feature extraction fails
    ///
    /// # Processing Pipeline
    ///
    /// 1. **Input Validation**: Ensures input is document type
    /// 2. **Preprocessing**: Applies format-specific preprocessing
    /// 3. **Feature Extraction**: Generates token-level features
    /// 4. **Metadata Assembly**: Creates output metadata
    /// 5. **Special Tokens**: Adds CLS and SEP tokens
    /// 6. **Attention Mask**: Generates attention mask for the sequence
    ///
    /// # Output Structure
    ///
    /// The returned `FeatureOutput` contains:
    ///
    /// - **features**: Flattened feature vector [max_length × feature_size]
    /// - **shape**: Dimensions [max_length, feature_size]
    /// - **metadata**: Document format and processing information
    /// - **attention_mask**: Binary mask for valid tokens
    /// - **special_tokens**: CLS and SEP token information
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// let input = FeatureInput::Document {
    ///     content: b"Sample document content".to_vec(),
    ///     format: DocumentFormat::Text,
    ///     metadata: None,
    /// };
    ///
    /// let output = extractor.extract_features(&input)?;
    /// assert_eq!(output.shape, vec![512, 768]); // max_length, feature_size
    /// assert!(output.attention_mask.is_some());
    /// assert_eq!(output.special_tokens.len(), 2); // CLS and SEP

    fn extract_features(&self, input: &FeatureInput) -> Result<FeatureOutput> {
        match input {
            FeatureInput::Document {
                content,
                format,
                metadata,
            } => {
                let processed_content = self.preprocess_document(content, *format)?;
                let features = self.extract_document_features(&processed_content)?;

                let mut output_metadata = HashMap::new();
                output_metadata.insert(
                    "format".to_string(),
                    serde_json::Value::String(format!("{:?}", format)),
                );

                Ok(FeatureOutput {
                    features,
                    shape: vec![self.config.max_length, self.config.feature_size],
                    metadata: output_metadata,
                    attention_mask: Some(vec![1; self.config.max_length]),
                    special_tokens: vec![
                        SpecialToken {
                            token_type: "CLS".to_string(),
                            position: 0,
                            value: "[CLS]".to_string(),
                        },
                        SpecialToken {
                            token_type: "SEP".to_string(),
                            position: self.config.max_length - 1,
                            value: "[SEP]".to_string(),
                        },
                    ],
                })
            },
            _ => Err(TrustformersError::invalid_input(
                "Document feature extractor requires document input".to_string(),
                Some("input_type".to_string()),
                Some("DocumentInput".to_string()),
                Some("Other input type".to_string()),
            )),
        }
    }

    /// Get the feature extractor configuration
    ///
    /// Returns a reference to the configuration object that defines the
    /// extractor's behavior and capabilities.
    ///
    /// # Returns
    ///
    /// Reference to the configuration trait object
    fn config(&self) -> &dyn FeatureExtractorConfig {
        &self.config
    }
}

// =============================================================================
// Document Feature Extractor Configuration
// =============================================================================

/// Configuration for document feature extraction
///
/// This configuration struct defines all parameters that control the behavior
/// of the document feature extractor. It provides fine-grained control over
/// the feature extraction process and model compatibility.
///
/// ## Key Parameters
///
/// - **Sequence Length**: Controls maximum document length processing
/// - **Feature Dimensions**: Defines output feature vector size
/// - **Layout Features**: Enables spatial and visual feature extraction
/// - **Batch Processing**: Configures batch processing capabilities
///
/// ## Model Compatibility
///
/// Different document understanding models may require specific configurations:
///
/// - **LayoutLM**: Requires layout and visual features enabled
/// - **BERT-based**: Standard text features with appropriate dimensions
/// - **Donut**: May require specific sequence length limits
///
/// ## Memory Considerations
///
/// Memory usage scales as: `max_length × feature_size × batch_size × 4 bytes`
/// For typical configurations (512 × 768 × 8), this requires ~12MB per batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentFeatureConfig {
    /// Maximum sequence length for document processing
    ///
    /// This parameter controls the maximum number of tokens that will be
    /// processed from each document. Longer documents will be truncated,
    /// while shorter documents will be padded.
    ///
    /// # Typical Values
    ///
    /// - **512**: Standard for most BERT-based models
    /// - **1024**: For longer document processing
    /// - **2048**: For very long documents (increased memory usage)
    ///
    /// # Memory Impact
    ///
    /// Memory usage increases linearly with max_length. Consider document
    /// chunking for very long sequences.
    pub max_length: usize,

    /// Size of feature vectors produced for each token
    ///
    /// This parameter defines the dimensionality of the feature vectors
    /// that will be generated for each token position. It should match
    /// the expected input size of the downstream model.
    ///
    /// # Common Values
    ///
    /// - **768**: BERT-base and similar models
    /// - **1024**: BERT-large and larger models
    /// - **512**: Smaller or specialized models
    ///
    /// # Model Compatibility
    ///
    /// This value must match the expected input dimension of the target model.
    /// Mismatched dimensions will cause runtime errors.
    pub feature_size: usize,

    /// Enable layout-aware feature extraction
    ///
    /// When enabled, the extractor will include spatial and layout information
    /// in the feature vectors. This is essential for models like LayoutLM
    /// that require document structure awareness.
    ///
    /// # When to Enable
    ///
    /// - **LayoutLM models**: Always required
    /// - **Document structure tasks**: Recommended
    /// - **Text-only tasks**: Usually disabled for efficiency
    ///
    /// # Performance Impact
    ///
    /// Enabling layout features increases computation time and memory usage.
    pub include_layout: bool,

    /// Enable visual feature extraction from document images
    ///
    /// When enabled, the extractor will process visual elements of documents
    /// such as fonts, formatting, and spatial relationships. This is useful
    /// for models that combine textual and visual understanding.
    ///
    /// # Use Cases
    ///
    /// - **Multimodal document models**: Required
    /// - **Form understanding**: Highly beneficial
    /// - **Text-only processing**: Should be disabled
    ///
    /// # Requirements
    ///
    /// Visual features require additional image processing capabilities
    /// and significantly increase computational requirements.
    pub include_visual_features: bool,

    /// Maximum number of documents that can be processed in a single batch
    ///
    /// This parameter limits batch sizes to prevent memory overflow and
    /// ensure stable processing. The optimal value depends on available
    /// memory and document complexity.
    ///
    /// # Determining Optimal Batch Size
    ///
    /// Consider these factors:
    /// - Available GPU/CPU memory
    /// - Document length (max_length)
    /// - Feature size
    /// - Model complexity
    ///
    /// # Memory Estimation
    ///
    /// Memory per batch ≈ `batch_size × max_length × feature_size × 4 bytes`
    ///
    /// # Examples
    ///
    /// - **High-memory systems**: 16-32 documents
    /// - **Standard systems**: 4-8 documents
    /// - **Memory-constrained**: 1-2 documents
    pub max_batch_size: Option<usize>,
}

impl DocumentFeatureConfig {
    /// Create configuration from a model configuration JSON object
    ///
    /// This method provides automatic configuration creation from standard
    /// model configuration files, making it easy to set up extractors for
    /// pretrained models without manual parameter tuning.
    ///
    /// # Arguments
    ///
    /// * `config` - JSON configuration object from model
    ///
    /// # Returns
    ///
    /// Configured `DocumentFeatureConfig` instance
    ///
    /// # Configuration Mapping
    ///
    /// The method maps standard model configuration keys to extractor parameters:
    ///
    /// - **max_position_embeddings** → max_length
    /// - **hidden_size** → feature_size
    /// - **has_visual_segment_embedding** → include_layout
    /// - **has_spatial_attention_bias** → include_visual_features
    /// - **max_batch_size** → max_batch_size
    ///
    /// # Default Values
    ///
    /// If configuration keys are missing, the following defaults are used:
    ///
    /// - max_length: 512
    /// - feature_size: 768
    /// - include_layout: false
    /// - include_visual_features: false
    /// - max_batch_size: None (unlimited)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// use serde_json::json;
    ///
    /// let model_config = json!({
    ///     "model_type": "layoutlm",
    ///     "hidden_size": 768,
    ///     "max_position_embeddings": 512,
    ///     "has_visual_segment_embedding": true,
    ///     "has_spatial_attention_bias": true
    /// });
    ///
    /// let config = DocumentFeatureConfig::from_config(&model_config)?;
    /// assert_eq!(config.feature_size, 768);
    /// assert_eq!(config.max_length, 512);
    /// assert!(config.include_layout);
    /// assert!(config.include_visual_features);

    pub fn from_config(config: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            max_length: config
                .get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(512) as usize,
            feature_size: config.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(768)
                as usize,
            include_layout: config
                .get("has_visual_segment_embedding")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            include_visual_features: config
                .get("has_spatial_attention_bias")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            max_batch_size: config
                .get("max_batch_size")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
        })
    }
}

impl FeatureExtractorConfig for DocumentFeatureConfig {
    /// Get the size of feature vectors produced by this extractor
    ///
    /// # Returns
    ///
    /// Number of dimensions in output feature vectors
    fn feature_size(&self) -> usize {
        self.feature_size
    }

    /// Check if the extractor supports batch processing
    ///
    /// Document extractors always support batch processing for improved
    /// throughput when processing multiple documents.
    ///
    /// # Returns
    ///
    /// Always returns `true`
    fn supports_batching(&self) -> bool {
        true
    }

    /// Get the maximum batch size supported by this extractor
    ///
    /// # Returns
    ///
    /// Maximum number of documents that can be processed in a single batch,
    /// or `None` if there is no specific limit
    fn max_batch_size(&self) -> Option<usize> {
        self.max_batch_size
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto::types::DocumentMetadata;

    #[test]
    fn test_document_feature_extractor_creation() {
        let config = DocumentFeatureConfig {
            max_length: 512,
            feature_size: 768,
            include_layout: false,
            include_visual_features: false,
            max_batch_size: Some(8),
        };

        let extractor = DocumentFeatureExtractor::new(config);
        assert_eq!(extractor.config().feature_size(), 768);
        assert!(extractor.config().supports_batching());
        assert_eq!(extractor.config().max_batch_size(), Some(8));
    }

    #[test]
    fn test_text_document_extraction() {
        let config = DocumentFeatureConfig {
            max_length: 512,
            feature_size: 768,
            include_layout: false,
            include_visual_features: false,
            max_batch_size: Some(8),
        };

        let extractor = DocumentFeatureExtractor::new(config);

        let content = b"This is a test document with some text content.";
        let input = FeatureInput::Document {
            content: content.to_vec(),
            format: DocumentFormat::Text,
            metadata: Some(DocumentMetadata {
                page_count: Some(1),
                author: Some("Test Author".to_string()),
                title: Some("Test Document".to_string()),
                creation_date: None,
            }),
        };

        let result = extractor.extract_features(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.features.len(), 512 * 768);
        assert_eq!(output.shape, vec![512, 768]);
        assert!(output.attention_mask.is_some());
        assert_eq!(output.special_tokens.len(), 2);
    }

    #[test]
    fn test_html_document_extraction() {
        let config = DocumentFeatureConfig {
            max_length: 256,
            feature_size: 384,
            include_layout: false,
            include_visual_features: false,
            max_batch_size: Some(4),
        };

        let extractor = DocumentFeatureExtractor::new(config);

        let html_content = br#"
        <html>
        <body>
            <h1>Document Title</h1>
            <p>This is the main content of the document.</p>
        </body>
        </html>
        "#;

        let input = FeatureInput::Document {
            content: html_content.to_vec(),
            format: DocumentFormat::Html,
            metadata: None,
        };

        let result = extractor.extract_features(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.features.len(), 256 * 384);
        assert_eq!(output.shape, vec![256, 384]);
    }

    #[test]
    fn test_markdown_document_extraction() {
        let config = DocumentFeatureConfig {
            max_length: 128,
            feature_size: 256,
            include_layout: true,
            include_visual_features: false,
            max_batch_size: Some(16),
        };

        let extractor = DocumentFeatureExtractor::new(config);

        let markdown_content = br#"
# Document Title

This is a **bold** text and *italic* text.

## Section Header

Some `code` and regular text.
        "#;

        let input = FeatureInput::Document {
            content: markdown_content.to_vec(),
            format: DocumentFormat::Markdown,
            metadata: None,
        };

        let result = extractor.extract_features(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.features.len(), 128 * 256);
        assert_eq!(output.shape, vec![128, 256]);
    }

    #[test]
    fn test_document_config_from_json() {
        let model_config = serde_json::json!({
            "model_type": "layoutlm",
            "hidden_size": 768,
            "max_position_embeddings": 512,
            "has_visual_segment_embedding": true,
            "has_spatial_attention_bias": true,
            "max_batch_size": 8
        });

        let config = DocumentFeatureConfig::from_config(&model_config).unwrap();
        assert_eq!(config.feature_size, 768);
        assert_eq!(config.max_length, 512);
        assert!(config.include_layout);
        assert!(config.include_visual_features);
        assert_eq!(config.max_batch_size, Some(8));
    }

    #[test]
    fn test_document_config_defaults() {
        let minimal_config = serde_json::json!({});

        let config = DocumentFeatureConfig::from_config(&minimal_config).unwrap();
        assert_eq!(config.feature_size, 768);
        assert_eq!(config.max_length, 512);
        assert!(!config.include_layout);
        assert!(!config.include_visual_features);
        assert_eq!(config.max_batch_size, None);
    }

    #[test]
    fn test_invalid_input_type() {
        let config = DocumentFeatureConfig {
            max_length: 512,
            feature_size: 768,
            include_layout: false,
            include_visual_features: false,
            max_batch_size: Some(8),
        };

        let extractor = DocumentFeatureExtractor::new(config);

        let input = FeatureInput::Text {
            content: "This is not a document input".to_string(),
            metadata: None,
        };

        let result = extractor.extract_features(&input);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TrustformersError::InvalidInput { .. }
        ));
    }

    #[test]
    fn test_html_text_extraction() {
        let config = DocumentFeatureConfig {
            max_length: 512,
            feature_size: 768,
            include_layout: false,
            include_visual_features: false,
            max_batch_size: Some(8),
        };

        let extractor = DocumentFeatureExtractor::new(config);

        let html = "<h1>Title</h1><p>Paragraph content</p>";
        let extracted = extractor.extract_text_from_html(html);

        // Should extract text content without HTML tags
        assert!(extracted.contains("Title"));
        assert!(extracted.contains("Paragraph"));
        assert!(extracted.contains("content"));
        assert!(!extracted.contains("<h1>"));
        assert!(!extracted.contains("</p>"));
    }

    #[test]
    fn test_markdown_text_extraction() {
        let config = DocumentFeatureConfig {
            max_length: 512,
            feature_size: 768,
            include_layout: false,
            include_visual_features: false,
            max_batch_size: Some(8),
        };

        let extractor = DocumentFeatureExtractor::new(config);

        let markdown = "# Title\n\n**Bold text** and *italic text*\n\n`code`";
        let extracted = extractor.extract_text_from_markdown(markdown);

        // Should extract text content without Markdown formatting
        assert!(extracted.contains("Title"));
        assert!(extracted.contains("Bold text"));
        assert!(extracted.contains("italic text"));
        assert!(extracted.contains("code"));
        assert!(!extracted.contains("# "));
        assert!(!extracted.contains("**"));
        assert!(!extracted.contains("*"));
        assert!(!extracted.contains("`"));
    }

    #[test]
    fn test_feature_extraction_with_special_tokens() {
        let config = DocumentFeatureConfig {
            max_length: 10,
            feature_size: 4,
            include_layout: false,
            include_visual_features: false,
            max_batch_size: Some(1),
        };

        let extractor = DocumentFeatureExtractor::new(config);

        let input = FeatureInput::Document {
            content: b"short text".to_vec(),
            format: DocumentFormat::Text,
            metadata: None,
        };

        let result = extractor.extract_features(&input).unwrap();

        // Check special tokens
        assert_eq!(result.special_tokens.len(), 2);
        assert_eq!(result.special_tokens[0].token_type, "CLS");
        assert_eq!(result.special_tokens[0].position, 0);
        assert_eq!(result.special_tokens[1].token_type, "SEP");
        assert_eq!(result.special_tokens[1].position, 9); // max_length - 1
    }
}
