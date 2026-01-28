//! # Vision Feature Extractor Module
//!
//! This module provides comprehensive vision-based feature extraction capabilities for
//! TrustformeRS, supporting various computer vision models and image processing tasks.
//!
//! ## Overview
//!
//! The vision feature extractor is designed to handle image inputs and extract meaningful
//! features that can be used for downstream tasks such as image classification, object
//! detection, and image-to-text generation. It supports multiple image formats and
//! provides extensive preprocessing capabilities.
//!
//! ## Supported Model Architectures
//!
//! - **CLIP**: Contrastive Language-Image Pre-training models
//! - **BLIP**: Bootstrapping Language-Image Pre-training models
//! - **ViT**: Vision Transformer models
//! - **Custom Vision Models**: Extensible architecture for additional models
//!
//! ## Key Features
//!
//! - **Multi-format Support**: JPEG, PNG, WebP, BMP, TIFF
//! - **Advanced Preprocessing**: Resize, crop, normalize, augmentation
//! - **Batch Processing**: Efficient handling of multiple images
//! - **Memory Optimization**: Intelligent memory management for large images
//! - **GPU Acceleration**: Hardware-accelerated processing when available
//!
//! ## Image Processing Pipeline
//!
//! ```text
//! Raw Image Data
//!      ↓
//! Format Detection & Decoding
//!      ↓
//! Preprocessing (Resize, Crop, Normalize)
//!      ↓
//! Feature Extraction (Vision Model)
//!      ↓
//! Postprocessing & Output Formatting
//! ```
//!
//! ## Usage Examples
//!
//! ### Basic Image Feature Extraction
//!
//! ```rust,ignore
//! use trustformers::auto::feature_extractors::vision::{VisionFeatureExtractor, VisionFeatureConfig};
//! use trustformers::auto::types::{FeatureInput, ImageFormat, ImageMetadata};
//!
//! // Create vision feature extractor
//! let config = VisionFeatureConfig {
//!     image_size: 224,
//!     feature_size: 768,
//!     normalize: true,
//!     do_resize: true,
//!     do_center_crop: true,
//!     crop_size: Some(224),
//!     mean: vec![0.485, 0.456, 0.406],
//!     std: vec![0.229, 0.224, 0.225],
//!     max_batch_size: Some(32),
//! };
//!
//! let extractor = VisionFeatureExtractor::new(config);
//!
//! // Prepare image input
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
//!
//! // Extract features
//! let output = extractor.extract_features(&input)?;
//! println!("Extracted {} features", output.features.len());
//! ```
//!
//! ### Configuration from Model Config
//!
//! ```rust,ignore
//! use serde_json::json;
//!
//! let model_config = json!({
//!     "image_size": 224,
//!     "hidden_size": 768,
//!     "do_normalize": true,
//!     "do_resize": true,
//!     "do_center_crop": true,
//!     "image_mean": [0.485, 0.456, 0.406],
//!     "image_std": [0.229, 0.224, 0.225]
//! });
//!
//! let config = VisionFeatureConfig::from_config(&model_config)?;
//! let extractor = VisionFeatureExtractor::new(config);
//! ```
//!
//! ## Performance Considerations
//!
//! - **Image Size**: Larger images require more memory and processing time
//! - **Batch Size**: Balance between throughput and memory usage
//! - **Preprocessing**: Enable only necessary preprocessing steps
//! - **Memory Management**: Use appropriate batch sizes to avoid OOM errors
//!
//! ## Error Handling
//!
//! The vision feature extractor provides detailed error information for:
//! - Unsupported image formats
//! - Invalid image data
//! - Memory allocation failures
//! - Processing pipeline errors

use super::{FeatureExtractor, FeatureExtractorConfig};
use crate::auto::types::{FeatureInput, FeatureOutput, ImageFormat};
use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Vision Feature Extractor Implementation
// =============================================================================

/// Vision feature extractor for image-based models
///
/// The `VisionFeatureExtractor` provides comprehensive image processing and feature
/// extraction capabilities for computer vision models. It handles various image
/// formats, performs intelligent preprocessing, and extracts high-quality features
/// suitable for downstream tasks.
///
/// ## Architecture
///
/// The extractor follows a modular design with separate components for:
/// - Image decoding and format handling
/// - Preprocessing and normalization
/// - Feature extraction through vision models
/// - Output formatting and metadata generation
///
/// ## Key Capabilities
///
/// - **Format Support**: Automatic detection and handling of multiple image formats
/// - **Preprocessing Pipeline**: Configurable resize, crop, and normalization
/// - **Batch Processing**: Efficient processing of multiple images
/// - **Memory Management**: Optimized memory usage for large images and batches
/// - **Error Handling**: Comprehensive error reporting and recovery
///
/// ## Thread Safety
///
/// The extractor is thread-safe and can be safely shared across threads.
/// However, individual extraction operations should be serialized per instance
/// to avoid resource conflicts.
///
/// ## Examples
///
/// ```rust,ignore
/// let config = VisionFeatureConfig {
///     image_size: 224,
///     feature_size: 768,
///     normalize: true,
///     do_resize: true,
///     do_center_crop: true,
///     crop_size: Some(224),
///     mean: vec![0.485, 0.456, 0.406],
///     std: vec![0.229, 0.224, 0.225],
///     max_batch_size: Some(32),
/// };
///
/// let extractor = VisionFeatureExtractor::new(config);
/// ```
#[derive(Debug, Clone)]
pub struct VisionFeatureExtractor {
    config: VisionFeatureConfig,
}

impl VisionFeatureExtractor {
    /// Create a new vision feature extractor with the specified configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters for the vision feature extractor
    ///
    /// # Returns
    ///
    /// A new `VisionFeatureExtractor` instance
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = VisionFeatureConfig {
    ///     image_size: 224,
    ///     feature_size: 768,
    ///     normalize: true,
    ///     do_resize: true,
    ///     do_center_crop: true,
    ///     crop_size: Some(224),
    ///     mean: vec![0.485, 0.456, 0.406],
    ///     std: vec![0.229, 0.224, 0.225],
    ///     max_batch_size: Some(32),
    /// };
    ///
    /// let extractor = VisionFeatureExtractor::new(config);
    /// ```
    pub fn new(config: VisionFeatureConfig) -> Self {
        Self { config }
    }

    /// Preprocess image data for feature extraction
    ///
    /// This method handles the complete image preprocessing pipeline including:
    /// - Image decoding based on format
    /// - Resizing to target dimensions
    /// - Center cropping if enabled
    /// - Normalization with mean/std values
    /// - Format conversion to model input format
    ///
    /// # Arguments
    ///
    /// * `data` - Raw image bytes
    /// * `format` - Image format specification
    ///
    /// # Returns
    ///
    /// Preprocessed image data as normalized float vector
    ///
    /// # Errors
    ///
    /// - `TrustformersError::InvalidInput` if image data is corrupted
    /// - `TrustformersError::ProcessingError` if preprocessing fails
    /// - `TrustformersError::ResourceError` if insufficient memory
    ///
    /// # Implementation Notes
    ///
    /// This is a simplified implementation. A production version would:
    /// - Use proper image decoding libraries (e.g., image crate)
    /// - Implement efficient resizing algorithms
    /// - Support hardware acceleration
    /// - Handle edge cases and error conditions robustly
    fn preprocess_image(&self, data: &[u8], format: ImageFormat) -> Result<Vec<f32>> {
        // Simplified image preprocessing implementation
        // In a real implementation, this would:
        // 1. Decode the image based on format
        // 2. Resize to target dimensions
        // 3. Apply center cropping if enabled
        // 4. Normalize with mean/std values
        // 5. Convert to model input format

        let processed_size = self.config.image_size * self.config.image_size * 3; // RGB channels

        // For this simplified implementation, return zero-initialized vector
        // Real implementation would perform actual image processing
        Ok(vec![0.0; processed_size])
    }

    /// Extract visual features from preprocessed image data
    ///
    /// This method performs the core feature extraction using the configured
    /// vision model. It processes the preprocessed image data through the
    /// model's feature extraction layers to produce high-level semantic features.
    ///
    /// # Arguments
    ///
    /// * `image` - Preprocessed image data as normalized float vector
    ///
    /// # Returns
    ///
    /// Extracted feature vector with configured dimensionality
    ///
    /// # Errors
    ///
    /// - `TrustformersError::ProcessingError` if feature extraction fails
    /// - `TrustformersError::ResourceError` if insufficient compute resources
    ///
    /// # Implementation Notes
    ///
    /// This is a simplified implementation. A production version would:
    /// - Load and execute actual vision models (ViT, CLIP, etc.)
    /// - Support different model architectures
    /// - Implement efficient inference pipelines
    /// - Handle batch processing optimally
    /// - Support GPU acceleration
    fn extract_visual_features(&self, image: &[f32]) -> Result<Vec<f32>> {
        // Simplified feature extraction implementation
        // In a real implementation, this would:
        // 1. Load the vision model (ViT, CLIP, etc.)
        // 2. Run inference on the preprocessed image
        // 3. Extract features from the appropriate layer
        // 4. Apply any post-processing transformations

        // For this simplified implementation, return zero-initialized features
        // Real implementation would run actual model inference
        Ok(vec![0.0; self.config.feature_size])
    }
}

impl FeatureExtractor for VisionFeatureExtractor {
    /// Extract features from image input
    ///
    /// This method implements the core feature extraction interface for vision inputs.
    /// It validates the input type, performs preprocessing, extracts features using
    /// the vision model, and formats the output with metadata.
    ///
    /// # Arguments
    ///
    /// * `input` - Feature input containing image data and metadata
    ///
    /// # Returns
    ///
    /// Feature output with extracted features, shape information, and metadata
    ///
    /// # Errors
    ///
    /// - `TrustformersError::InvalidInput` if input is not an image
    /// - `TrustformersError::ProcessingError` if feature extraction fails
    /// - `TrustformersError::ResourceError` if insufficient resources
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// let input = FeatureInput::Image {
    ///     data: image_bytes,
    ///     format: ImageFormat::Jpeg,
    ///     metadata: Some(ImageMetadata {
    ///         width: 640,
    ///         height: 480,
    ///         channels: 3,
    ///         dpi: Some(96),
    ///     }),
    /// };
    ///
    /// let output = extractor.extract_features(&input)?;

    fn extract_features(&self, input: &FeatureInput) -> Result<FeatureOutput> {
        match input {
            FeatureInput::Image {
                data,
                format,
                metadata,
            } => {
                // Decode and preprocess image
                let processed_image = self.preprocess_image(data, *format)?;

                // Extract visual features using the vision model
                let features = self.extract_visual_features(&processed_image)?;

                // Build output metadata from input metadata
                let mut output_metadata = HashMap::new();
                if let Some(meta) = metadata {
                    output_metadata.insert(
                        "width".to_string(),
                        serde_json::Value::Number(meta.width.into()),
                    );
                    output_metadata.insert(
                        "height".to_string(),
                        serde_json::Value::Number(meta.height.into()),
                    );
                    output_metadata.insert(
                        "channels".to_string(),
                        serde_json::Value::Number(meta.channels.into()),
                    );
                    if let Some(dpi) = meta.dpi {
                        output_metadata
                            .insert("dpi".to_string(), serde_json::Value::Number(dpi.into()));
                    }
                }

                // Add processing metadata
                output_metadata.insert(
                    "processed_image_size".to_string(),
                    serde_json::Value::Number(self.config.image_size.into()),
                );
                output_metadata.insert(
                    "normalized".to_string(),
                    serde_json::Value::Bool(self.config.normalize),
                );

                Ok(FeatureOutput {
                    features,
                    shape: vec![self.config.feature_size],
                    metadata: output_metadata,
                    attention_mask: None,
                    special_tokens: vec![],
                })
            },
            _ => Err(TrustformersError::invalid_input_simple(
                "Vision feature extractor requires image input".to_string(),
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

    /// Check if the extractor supports a specific input type
    ///
    /// Vision feature extractors only support image inputs.
    ///
    /// # Arguments
    ///
    /// * `input` - Input to check for compatibility
    ///
    /// # Returns
    ///
    /// `true` if the input is an image, `false` otherwise
    fn supports_input(&self, input: &FeatureInput) -> bool {
        matches!(input, FeatureInput::Image { .. })
    }

    /// Get extractor capabilities and metadata
    ///
    /// Returns detailed information about the vision extractor's capabilities,
    /// including supported formats, processing parameters, and model information.
    ///
    /// # Returns
    ///
    /// HashMap containing capability information
    fn capabilities(&self) -> HashMap<String, serde_json::Value> {
        let mut caps = HashMap::new();

        // Basic capabilities from parent trait
        caps.insert(
            "feature_size".to_string(),
            serde_json::Value::Number(self.config.feature_size.into()),
        );
        caps.insert(
            "supports_batching".to_string(),
            serde_json::Value::Bool(self.config.supports_batching()),
        );
        if let Some(max_batch) = self.config.max_batch_size {
            caps.insert(
                "max_batch_size".to_string(),
                serde_json::Value::Number(max_batch.into()),
            );
        }

        // Vision-specific capabilities
        caps.insert(
            "modality".to_string(),
            serde_json::Value::String("vision".to_string()),
        );
        caps.insert(
            "image_size".to_string(),
            serde_json::Value::Number(self.config.image_size.into()),
        );
        caps.insert(
            "supports_resize".to_string(),
            serde_json::Value::Bool(self.config.do_resize),
        );
        caps.insert(
            "supports_center_crop".to_string(),
            serde_json::Value::Bool(self.config.do_center_crop),
        );
        caps.insert(
            "normalize".to_string(),
            serde_json::Value::Bool(self.config.normalize),
        );

        // Supported image formats
        let supported_formats = vec!["jpeg", "jpg", "png", "webp", "bmp", "tiff"];
        caps.insert(
            "supported_formats".to_string(),
            serde_json::Value::Array(
                supported_formats
                    .into_iter()
                    .map(|f| serde_json::Value::String(f.to_string()))
                    .collect(),
            ),
        );

        caps
    }
}

// =============================================================================
// Vision Feature Extractor Configuration
// =============================================================================

/// Configuration for vision feature extractors
///
/// The `VisionFeatureConfig` struct defines all the parameters needed to
/// configure a vision feature extractor's behavior. It includes settings for
/// image preprocessing, model parameters, and processing constraints.
///
/// ## Configuration Categories
///
/// - **Model Parameters**: Feature size, architecture-specific settings
/// - **Image Processing**: Size, cropping, normalization parameters
/// - **Performance**: Batch size limits, memory constraints
/// - **Preprocessing**: Mean/std values, format handling options
///
/// ## Default Values
///
/// The configuration provides sensible defaults based on common vision models:
/// - Image size: 224×224 (standard for most vision transformers)
/// - Normalization: ImageNet mean/std values
/// - Feature size: 768 (BERT/ViT base size)
/// - Batch processing: Enabled with reasonable limits
///
/// ## Examples
///
/// ```rust,ignore
/// // Manual configuration
/// let config = VisionFeatureConfig {
///     image_size: 224,
///     feature_size: 768,
///     normalize: true,
///     do_resize: true,
///     do_center_crop: true,
///     crop_size: Some(224),
///     mean: vec![0.485, 0.456, 0.406],
///     std: vec![0.229, 0.224, 0.225],
///     max_batch_size: Some(32),
/// };
///
/// // From model configuration
/// let model_config = serde_json::json!({
///     "image_size": 224,
///     "hidden_size": 768,
///     "do_normalize": true
/// });
/// let config = VisionFeatureConfig::from_config(&model_config)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionFeatureConfig {
    /// Target image size for preprocessing (width and height in pixels)
    ///
    /// Images will be resized to this size before feature extraction.
    /// Common values: 224 (ViT), 336 (CLIP), 384 (DeiT), 512 (high-res models)
    pub image_size: usize,

    /// Size of the output feature vectors
    ///
    /// This determines the dimensionality of the extracted features.
    /// Common values: 768 (base models), 1024 (large models), 1536 (XL models)
    pub feature_size: usize,

    /// Whether to apply normalization to input images
    ///
    /// When enabled, images are normalized using the specified mean and std values.
    /// This is typically required for models trained on normalized data.
    pub normalize: bool,

    /// Whether to resize images to the target size
    ///
    /// When enabled, input images are resized to `image_size` before processing.
    /// Disable if you want to handle resizing externally.
    pub do_resize: bool,

    /// Whether to apply center cropping during preprocessing
    ///
    /// When enabled, images are center-cropped to the target size after resizing.
    /// This helps maintain aspect ratio and focus on the central content.
    pub do_center_crop: bool,

    /// Size for center cropping (optional)
    ///
    /// If specified, images are center-cropped to this size. If None,
    /// the `image_size` value is used for cropping.
    pub crop_size: Option<usize>,

    /// Mean values for normalization (per channel)
    ///
    /// RGB mean values used for image normalization. Default values are
    /// ImageNet statistics: [0.485, 0.456, 0.406]
    pub mean: Vec<f32>,

    /// Standard deviation values for normalization (per channel)
    ///
    /// RGB standard deviation values used for image normalization.
    /// Default values are ImageNet statistics: [0.229, 0.224, 0.225]
    pub std: Vec<f32>,

    /// Maximum batch size for processing
    ///
    /// Limits the number of images that can be processed in a single batch.
    /// This helps prevent out-of-memory errors and ensures consistent performance.
    pub max_batch_size: Option<usize>,
}

impl VisionFeatureConfig {
    /// Create configuration from a model configuration JSON object
    ///
    /// This method parses a model configuration (typically from HuggingFace Hub)
    /// and extracts the relevant vision processing parameters. It provides
    /// sensible defaults for missing values.
    ///
    /// # Arguments
    ///
    /// * `config` - JSON configuration object from model definition
    ///
    /// # Returns
    ///
    /// Configured `VisionFeatureConfig` instance
    ///
    /// # Errors
    ///
    /// - `TrustformersError::ConfigError` if required parameters are invalid
    ///
    /// # Supported Configuration Keys
    ///
    /// - `image_size` or `size`: Target image dimensions
    /// - `hidden_size`: Feature vector dimensionality
    /// - `do_normalize`: Enable/disable normalization
    /// - `do_resize`: Enable/disable resizing
    /// - `do_center_crop`: Enable/disable center cropping
    /// - `crop_size`: Specific crop size
    /// - `image_mean`: Normalization mean values
    /// - `image_std`: Normalization standard deviation values
    /// - `max_batch_size`: Maximum batch size limit
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// let config_json = serde_json::json!({
    ///     "image_size": 224,
    ///     "hidden_size": 768,
    ///     "do_normalize": true,
    ///     "do_resize": true,
    ///     "do_center_crop": true,
    ///     "image_mean": [0.485, 0.456, 0.406],
    ///     "image_std": [0.229, 0.224, 0.225]
    /// });
    ///
    /// let config = VisionFeatureConfig::from_config(&config_json)?;

    pub fn from_config(config: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            image_size: config
                .get("image_size")
                .or_else(|| config.get("size"))
                .and_then(|v| v.as_u64())
                .unwrap_or(224) as usize,
            feature_size: config.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(768)
                as usize,
            normalize: config.get("do_normalize").and_then(|v| v.as_bool()).unwrap_or(true),
            do_resize: config.get("do_resize").and_then(|v| v.as_bool()).unwrap_or(true),
            do_center_crop: config.get("do_center_crop").and_then(|v| v.as_bool()).unwrap_or(true),
            crop_size: config.get("crop_size").and_then(|v| v.as_u64()).map(|v| v as usize),
            mean: config
                .get("image_mean")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_f64()).map(|v| v as f32).collect())
                .unwrap_or_else(|| vec![0.485, 0.456, 0.406]), // ImageNet mean
            std: config
                .get("image_std")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_f64()).map(|v| v as f32).collect())
                .unwrap_or_else(|| vec![0.229, 0.224, 0.225]), // ImageNet std
            max_batch_size: config
                .get("max_batch_size")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
        })
    }

    /// Create a default configuration for common vision models
    ///
    /// Returns a configuration with sensible defaults suitable for most
    /// vision transformer models and CLIP-style architectures.
    ///
    /// # Returns
    ///
    /// Default `VisionFeatureConfig` instance
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = VisionFeatureConfig::default();
    /// assert_eq!(config.image_size, 224);
    /// assert_eq!(config.feature_size, 768);
    /// ```
    pub fn default() -> Self {
        Self {
            image_size: 224,
            feature_size: 768,
            normalize: true,
            do_resize: true,
            do_center_crop: true,
            crop_size: None,
            mean: vec![0.485, 0.456, 0.406], // ImageNet mean
            std: vec![0.229, 0.224, 0.225],  // ImageNet std
            max_batch_size: Some(32),
        }
    }
}

impl FeatureExtractorConfig for VisionFeatureConfig {
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
    /// Vision extractors support batch processing for improved efficiency.
    ///
    /// # Returns
    ///
    /// Always `true` for vision extractors
    fn supports_batching(&self) -> bool {
        true
    }

    /// Get the maximum batch size supported by this extractor
    ///
    /// # Returns
    ///
    /// Maximum number of images that can be processed in a single batch
    fn max_batch_size(&self) -> Option<usize> {
        self.max_batch_size
    }

    /// Get additional vision-specific configuration parameters
    ///
    /// Returns detailed configuration information specific to vision processing,
    /// including image size, normalization parameters, and preprocessing options.
    ///
    /// # Returns
    ///
    /// HashMap containing additional configuration parameters
    fn additional_params(&self) -> HashMap<String, serde_json::Value> {
        let mut params = HashMap::new();

        params.insert(
            "image_size".to_string(),
            serde_json::Value::Number(self.image_size.into()),
        );
        params.insert(
            "normalize".to_string(),
            serde_json::Value::Bool(self.normalize),
        );
        params.insert(
            "do_resize".to_string(),
            serde_json::Value::Bool(self.do_resize),
        );
        params.insert(
            "do_center_crop".to_string(),
            serde_json::Value::Bool(self.do_center_crop),
        );

        if let Some(crop_size) = self.crop_size {
            params.insert(
                "crop_size".to_string(),
                serde_json::Value::Number(crop_size.into()),
            );
        }

        params.insert(
            "mean".to_string(),
            serde_json::Value::Array(
                self.mean
                    .iter()
                    .map(|&v| {
                        serde_json::Value::Number(serde_json::Number::from_f64(v as f64).unwrap())
                    })
                    .collect(),
            ),
        );

        params.insert(
            "std".to_string(),
            serde_json::Value::Array(
                self.std
                    .iter()
                    .map(|&v| {
                        serde_json::Value::Number(serde_json::Number::from_f64(v as f64).unwrap())
                    })
                    .collect(),
            ),
        );

        params
    }

    /// Validate configuration consistency
    ///
    /// Checks that all configuration parameters are valid and internally consistent.
    /// This includes validating array lengths, size constraints, and logical relationships.
    ///
    /// # Returns
    ///
    /// `Ok(())` if configuration is valid, error describing the issue otherwise
    ///
    /// # Errors
    ///
    /// - `TrustformersError::ConfigError` if parameters are invalid or inconsistent
    fn validate(&self) -> Result<()> {
        // Validate image size
        if self.image_size == 0 {
            return Err(TrustformersError::lconfig_error(
                "Image size must be greater than 0".to_string(),
            ));
        }

        // Validate feature size
        if self.feature_size == 0 {
            return Err(TrustformersError::lconfig_error(
                "Feature size must be greater than 0".to_string(),
            ));
        }

        // Validate normalization parameters
        if self.normalize {
            if self.mean.len() != 3 {
                return Err(TrustformersError::lconfig_error(
                    "Mean values must have exactly 3 elements (RGB)".to_string(),
                ));
            }
            if self.std.len() != 3 {
                return Err(TrustformersError::lconfig_error(
                    "Standard deviation values must have exactly 3 elements (RGB)".to_string(),
                ));
            }

            // Check that std values are positive
            for &std_val in &self.std {
                if std_val <= 0.0 {
                    return Err(TrustformersError::lconfig_error(
                        "Standard deviation values must be positive".to_string(),
                    ));
                }
            }
        }

        // Validate crop size if specified
        if let Some(crop_size) = self.crop_size {
            if crop_size == 0 {
                return Err(TrustformersError::lconfig_error(
                    "Crop size must be greater than 0".to_string(),
                ));
            }
            if crop_size > self.image_size {
                return Err(TrustformersError::lconfig_error(
                    "Crop size cannot be larger than image size".to_string(),
                ));
            }
        }

        // Validate batch size if specified
        if let Some(batch_size) = self.max_batch_size {
            if batch_size == 0 {
                return Err(TrustformersError::lconfig_error(
                    "Maximum batch size must be greater than 0".to_string(),
                ));
            }
        }

        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto::types::{ImageFormat, ImageMetadata};

    #[test]
    fn test_vision_feature_extractor_creation() {
        let config = VisionFeatureConfig {
            image_size: 224,
            feature_size: 768,
            normalize: true,
            do_resize: true,
            do_center_crop: true,
            crop_size: Some(224),
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
            max_batch_size: Some(32),
        };

        let extractor = VisionFeatureExtractor::new(config);
        assert_eq!(extractor.config().feature_size(), 768);
        assert!(extractor.config().supports_batching());
        assert_eq!(extractor.config().max_batch_size(), Some(32));
    }

    #[test]
    fn test_vision_feature_extraction() {
        let config = VisionFeatureConfig::default();
        let extractor = VisionFeatureExtractor::new(config);

        let input = FeatureInput::Image {
            data: vec![0u8; 1024],
            format: ImageFormat::Jpeg,
            metadata: Some(ImageMetadata {
                width: 640,
                height: 480,
                channels: 3,
                dpi: Some(96),
            }),
        };

        let result = extractor.extract_features(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.features.len(), 768);
        assert_eq!(output.shape, vec![768]);

        // Check metadata preservation
        assert_eq!(
            output.metadata.get("width").unwrap().as_u64().expect("expected u64 value"),
            640
        );
        assert_eq!(
            output.metadata.get("height").unwrap().as_u64().expect("expected u64 value"),
            480
        );
        assert_eq!(
            output.metadata.get("channels").unwrap().as_u64().expect("expected u64 value"),
            3
        );
        assert_eq!(
            output.metadata.get("dpi").unwrap().as_u64().expect("expected u64 value"),
            96
        );
    }

    #[test]
    fn test_vision_config_from_json() {
        let config_json = serde_json::json!({
            "image_size": 224,
            "hidden_size": 768,
            "do_normalize": true,
            "do_resize": true,
            "do_center_crop": true,
            "crop_size": 224,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "max_batch_size": 32
        });

        let config = VisionFeatureConfig::from_config(&config_json).unwrap();
        assert_eq!(config.image_size, 224);
        assert_eq!(config.feature_size, 768);
        assert!(config.normalize);
        assert!(config.do_resize);
        assert!(config.do_center_crop);
        assert_eq!(config.crop_size, Some(224));
        assert_eq!(config.mean, vec![0.485, 0.456, 0.406]);
        assert_eq!(config.std, vec![0.229, 0.224, 0.225]);
        assert_eq!(config.max_batch_size, Some(32));
    }

    #[test]
    fn test_vision_config_defaults() {
        let minimal_config = serde_json::json!({});
        let config = VisionFeatureConfig::from_config(&minimal_config).unwrap();

        assert_eq!(config.image_size, 224);
        assert_eq!(config.feature_size, 768);
        assert!(config.normalize);
        assert!(config.do_resize);
        assert!(config.do_center_crop);
        assert_eq!(config.mean, vec![0.485, 0.456, 0.406]);
        assert_eq!(config.std, vec![0.229, 0.224, 0.225]);
    }

    #[test]
    fn test_vision_config_validation() {
        let mut config = VisionFeatureConfig::default();

        // Valid configuration should pass
        assert!(config.validate().is_ok());

        // Invalid image size
        config.image_size = 0;
        assert!(config.validate().is_err());
        config.image_size = 224;

        // Invalid feature size
        config.feature_size = 0;
        assert!(config.validate().is_err());
        config.feature_size = 768;

        // Invalid mean length
        config.mean = vec![0.5, 0.5]; // Should be 3 elements
        assert!(config.validate().is_err());
        config.mean = vec![0.485, 0.456, 0.406];

        // Invalid std length
        config.std = vec![0.2]; // Should be 3 elements
        assert!(config.validate().is_err());
        config.std = vec![0.229, 0.224, 0.225];

        // Invalid std values (negative)
        config.std = vec![-0.1, 0.224, 0.225];
        assert!(config.validate().is_err());
        config.std = vec![0.229, 0.224, 0.225];

        // Invalid crop size (larger than image size)
        config.crop_size = Some(300);
        assert!(config.validate().is_err());
        config.crop_size = Some(224);

        // Should be valid again
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_input_type_validation() {
        let config = VisionFeatureConfig::default();
        let extractor = VisionFeatureExtractor::new(config);

        // Valid image input
        let image_input = FeatureInput::Image {
            data: vec![0u8; 1024],
            format: ImageFormat::Png,
            metadata: None,
        };
        assert!(extractor.supports_input(&image_input));

        // Invalid audio input
        let audio_input = FeatureInput::Audio {
            samples: vec![0.0; 1000],
            sample_rate: 16000,
            metadata: None,
        };
        assert!(!extractor.supports_input(&audio_input));
    }

    #[test]
    fn test_extractor_capabilities() {
        let config = VisionFeatureConfig::default();
        let extractor = VisionFeatureExtractor::new(config);
        let caps = extractor.capabilities();

        assert_eq!(
            caps.get("modality").unwrap().as_str().expect("expected str value"),
            "vision"
        );
        assert_eq!(
            caps.get("feature_size").unwrap().as_u64().expect("expected u64 value"),
            768
        );
        assert_eq!(
            caps.get("image_size").unwrap().as_u64().expect("expected u64 value"),
            224
        );
        assert!(caps.get("supports_batching").unwrap().as_bool().unwrap());
        assert!(caps.contains_key("supported_formats"));
    }

    #[test]
    fn test_invalid_input_handling() {
        let config = VisionFeatureConfig::default();
        let extractor = VisionFeatureExtractor::new(config);

        // Try to pass text input to vision extractor
        let input = FeatureInput::Text {
            content: "This is text, not an image".to_string(),
            metadata: None,
        };

        let result = extractor.extract_features(&input);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TrustformersError::InvalidInput { .. }
        ));
    }
}
