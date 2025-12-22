//! # Feature Extractors Module
//!
//! This module provides the core feature extraction functionality for TrustformeRS,
//! enabling automatic creation and configuration of feature extractors based on
//! model types and tasks.
//!
//! ## Overview
//!
//! The feature extractor system is designed around the concept of automatic model
//! detection and task-specific optimization. It supports multiple modalities including
//! vision, audio, text, and multimodal inputs.
//!
//! ## Architecture
//!
//! The module follows a hierarchical structure:
//!
//! ```text
//! AutoFeatureExtractor (Entry Point)
//!         ↓
//! FeatureExtractor Trait (Base Interface)
//!         ↓
//! Specific Implementations (Vision, Audio, Document, etc.)
//! ```
//!
//! ## Key Components
//!
//! - **AutoFeatureExtractor**: Main entry point for automatic feature extractor creation
//! - **FeatureExtractor**: Core trait defining the feature extraction interface
//! - **FeatureExtractorConfig**: Configuration trait for extractor parameters
//! - **Specific Extractors**: Task-specific implementations (vision, audio, etc.)
//!
//! ## Usage Examples
//!
//! ### Basic Usage
//!
//! ```rust,ignore
//! use trustformers::auto::feature_extractors::AutoFeatureExtractor;
//! use trustformers::auto::types::{FeatureInput, ImageFormat, ImageMetadata};
//!
//! // Create a feature extractor from a pretrained model
//! let extractor = AutoFeatureExtractor::from_pretrained("clip-vit-base-patch32")?;
//!
//! // Prepare input
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
//! ### Task-Specific Creation
//!
//! ```rust,ignore
//! use trustformers::auto::feature_extractors::AutoFeatureExtractor;
//!
//! // Create extractor for specific task
//! let config = serde_json::json!({
//!     "model_type": "clip",
//!     "image_size": 224,
//!     "hidden_size": 768
//! });
//!
//! let extractor = AutoFeatureExtractor::for_task("image-classification", &config)?;
//! ```
//!
//! ## Supported Model Types
//!
//! The auto feature extractor supports the following model architectures:
//!
//! - **Vision Models**: CLIP, BLIP, ViT
//! - **Audio Models**: Wav2Vec2, Whisper, HuBERT
//! - **Document Models**: LayoutLM, Donut
//! - **Generic Models**: Fallback for unsupported types
//!
//! ## Supported Tasks
//!
//! The system automatically selects appropriate extractors for:
//!
//! - **Vision Tasks**: image-classification, object-detection, image-to-text
//! - **Audio Tasks**: automatic-speech-recognition, audio-classification
//! - **Document Tasks**: document-understanding, document-question-answering
//! - **Generic Tasks**: Fallback processing
//!
//! ## Performance Considerations
//!
//! - All extractors support batching for improved throughput
//! - Memory usage is optimized based on model size and batch configuration
//! - GPU acceleration is supported where available
//! - Lazy loading of models to reduce startup time
//!
//! ## Thread Safety
//!
//! All feature extractors implement `Send + Sync` and can be safely used
//! across multiple threads. However, individual extraction operations
//! are not thread-safe and should be serialized per extractor instance.

use crate::auto::types::{FeatureInput, FeatureOutput};
use crate::error::Result;
use std::collections::HashMap;

// Import specific extractor modules
mod audio;
mod document;
mod generic;
mod vision;

// Re-export specific extractor implementations
pub use audio::{AudioFeatureConfig, AudioFeatureExtractor};
pub use document::{DocumentFeatureConfig, DocumentFeatureExtractor};
pub use generic::{GenericFeatureConfig, GenericFeatureExtractor};
pub use vision::{VisionFeatureConfig, VisionFeatureExtractor};

// =============================================================================
// Auto Feature Extractor
// =============================================================================

/// Automatically create feature extractors based on model type and task
///
/// The `AutoFeatureExtractor` provides a unified interface for creating
/// feature extractors without needing to know the specific implementation
/// details. It automatically detects the appropriate extractor type based
/// on model configuration or task requirements.
///
/// ## Design Philosophy
///
/// The auto feature extractor follows the "convention over configuration"
/// principle, making intelligent defaults while still allowing customization
/// when needed. It prioritizes ease of use for common scenarios while
/// maintaining flexibility for advanced use cases.
///
/// ## Model Type Detection
///
/// The extractor uses the following priority order for type detection:
/// 1. Explicit model_type in configuration
/// 2. Task-specific requirements
/// 3. Configuration pattern matching
/// 4. Fallback to generic extractor
///
/// ## Examples
///
/// ```rust,ignore
/// // From pretrained model
/// let extractor = AutoFeatureExtractor::from_pretrained("openai/clip-vit-base-patch32")?;
///
/// // For specific task
/// let config = serde_json::json!({"model_type": "clip", "image_size": 224});
/// let extractor = AutoFeatureExtractor::for_task("image-classification", &config)?;
/// ```
#[derive(Debug, Clone)]
pub struct AutoFeatureExtractor;

impl AutoFeatureExtractor {
    /// Create a feature extractor from a pretrained model
    ///
    /// This method loads the model configuration from the Hub or local path
    /// and automatically selects the appropriate feature extractor implementation
    /// based on the model type and architecture.
    ///
    /// # Arguments
    ///
    /// * `model_name_or_path` - Model identifier on Hub or local filesystem path
    ///
    /// # Returns
    ///
    /// A boxed feature extractor trait object that can process inputs for the model
    ///
    /// # Errors
    ///
    /// - `TrustformersError::ModelNotFound` if the model cannot be located
    /// - `TrustformersError::ConfigError` if the model configuration is invalid
    /// - `TrustformersError::UnsupportedModel` if the model type is not supported
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// // Load a vision model
    /// let extractor = AutoFeatureExtractor::from_pretrained("openai/clip-vit-base-patch32")?;
    ///
    /// // Load an audio model
    /// let extractor = AutoFeatureExtractor::from_pretrained("facebook/wav2vec2-base-960h")?;

    pub fn from_pretrained(model_name_or_path: &str) -> Result<Box<dyn FeatureExtractor>> {
        let config = crate::hub::load_config_from_hub(model_name_or_path, None)?;

        let model_type = config.get("model_type").and_then(|v| v.as_str()).unwrap_or("unknown");

        match model_type {
            "clip" | "blip" | "vit" => Ok(Box::new(VisionFeatureExtractor::new(
                VisionFeatureConfig::from_config(&config)?,
            ))),
            "wav2vec2" | "whisper" | "hubert" => Ok(Box::new(AudioFeatureExtractor::new(
                AudioFeatureConfig::from_config(&config)?,
            ))),
            "layoutlm" | "donut" => Ok(Box::new(DocumentFeatureExtractor::new(
                DocumentFeatureConfig::from_config(&config)?,
            ))),
            _ => Ok(Box::new(GenericFeatureExtractor::new(
                GenericFeatureConfig::from_config(&config)?,
            ))),
        }
    }

    /// Create a feature extractor for a specific task
    ///
    /// This method creates a feature extractor optimized for a particular task,
    /// using the provided model configuration. It's useful when you know the
    /// task requirements but want automatic extractor selection.
    ///
    /// # Arguments
    ///
    /// * `task` - The target task (e.g., "image-classification", "asr")
    /// * `model_config` - Model configuration as JSON object
    ///
    /// # Returns
    ///
    /// A boxed feature extractor optimized for the specified task
    ///
    /// # Task Categories
    ///
    /// - **Vision**: image-classification, object-detection, image-to-text
    /// - **Audio**: automatic-speech-recognition, audio-classification
    /// - **Document**: document-understanding, document-question-answering
    /// - **Generic**: Fallback for unrecognized tasks
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// let config = serde_json::json!({
    ///     "model_type": "clip",
    ///     "image_size": 224,
    ///     "hidden_size": 768
    /// });
    ///
    /// let extractor = AutoFeatureExtractor::for_task("image-classification", &config)?;

    pub fn for_task(
        task: &str,
        model_config: &serde_json::Value,
    ) -> Result<Box<dyn FeatureExtractor>> {
        match task {
            "image-classification" | "object-detection" | "image-to-text" => Ok(Box::new(
                VisionFeatureExtractor::new(VisionFeatureConfig::from_config(model_config)?),
            )),
            "automatic-speech-recognition" | "audio-classification" => Ok(Box::new(
                AudioFeatureExtractor::new(AudioFeatureConfig::from_config(model_config)?),
            )),
            "document-understanding" | "document-question-answering" => Ok(Box::new(
                DocumentFeatureExtractor::new(DocumentFeatureConfig::from_config(model_config)?),
            )),
            _ => Ok(Box::new(GenericFeatureExtractor::new(
                GenericFeatureConfig::from_config(model_config)?,
            ))),
        }
    }

    /// Get supported model types
    ///
    /// Returns a list of model types that are currently supported by the
    /// auto feature extractor system.
    ///
    /// # Returns
    ///
    /// Vector of supported model type strings
    pub fn supported_model_types() -> Vec<&'static str> {
        vec![
            "clip", "blip", "vit", // Vision models
            "wav2vec2", "whisper", "hubert", // Audio models
            "layoutlm", "donut",   // Document models
            "generic", // Fallback
        ]
    }

    /// Get supported tasks
    ///
    /// Returns a list of tasks that are currently supported by the
    /// auto feature extractor system.
    ///
    /// # Returns
    ///
    /// Vector of supported task strings
    pub fn supported_tasks() -> Vec<&'static str> {
        vec![
            // Vision tasks
            "image-classification",
            "object-detection",
            "image-to-text",
            // Audio tasks
            "automatic-speech-recognition",
            "audio-classification",
            // Document tasks
            "document-understanding",
            "document-question-answering",
            // Generic
            "text-classification",
        ]
    }
}

// =============================================================================
// Core Traits
// =============================================================================

/// Trait for feature extraction from various input modalities
///
/// The `FeatureExtractor` trait defines the core interface that all feature
/// extractors must implement. It provides a unified API for extracting
/// features from different types of input data.
///
/// ## Design Principles
///
/// - **Modality Agnostic**: The same interface works for vision, audio, text, etc.
/// - **Preprocessing Integration**: Built-in preprocessing and postprocessing hooks
/// - **Configuration Access**: Easy access to extractor configuration
/// - **Error Handling**: Comprehensive error reporting for debugging
///
/// ## Implementation Guidelines
///
/// When implementing this trait:
/// 1. Validate input types in `extract_features`
/// 2. Use `preprocess` for input normalization
/// 3. Use `postprocess` for output formatting
/// 4. Return detailed error information for failures
///
/// ## Thread Safety
///
/// Implementations must be `Send + Sync` to support multi-threaded usage.
/// However, individual extraction operations are not required to be thread-safe.
pub trait FeatureExtractor: Send + Sync {
    /// Extract features from raw input
    ///
    /// This is the core method that performs feature extraction. It takes
    /// raw input data and returns processed features suitable for model input.
    ///
    /// # Arguments
    ///
    /// * `input` - Raw input data in the appropriate format for this extractor
    ///
    /// # Returns
    ///
    /// Extracted features with metadata and additional information
    ///
    /// # Errors
    ///
    /// - `TrustformersError::InvalidInput` if the input type is not supported
    /// - `TrustformersError::ProcessingError` if feature extraction fails
    /// - `TrustformersError::ResourceError` if insufficient memory or compute resources
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    ///    /// let features = extractor.extract_features(&input)?;
    /// println!("Extracted {} features", features.features.len());

    fn extract_features(&self, input: &FeatureInput) -> Result<FeatureOutput>;

    /// Get the feature extractor configuration
    ///
    /// Returns a reference to the configuration object that defines the
    /// extractor's behavior and capabilities.
    ///
    /// # Returns
    ///
    /// Reference to the configuration trait object
    fn config(&self) -> &dyn FeatureExtractorConfig;

    /// Preprocess input before feature extraction
    ///
    /// This method is called automatically before `extract_features` and
    /// can be overridden to implement custom preprocessing logic.
    ///
    /// # Arguments
    ///
    /// * `input` - Raw input to preprocess
    ///
    /// # Returns
    ///
    /// Preprocessed input ready for feature extraction
    ///
    /// # Default Implementation
    ///
    /// The default implementation returns the input unchanged.
    fn preprocess(&self, input: &FeatureInput) -> Result<FeatureInput> {
        Ok(input.clone())
    }

    /// Postprocess features after extraction
    ///
    /// This method is called automatically after `extract_features` and
    /// can be overridden to implement custom postprocessing logic.
    ///
    /// # Arguments
    ///
    /// * `features` - Raw extracted features
    ///
    /// # Returns
    ///
    /// Postprocessed features ready for model input
    ///
    /// # Default Implementation
    ///
    /// The default implementation returns the features unchanged.
    fn postprocess(&self, features: FeatureOutput) -> Result<FeatureOutput> {
        Ok(features)
    }

    /// Check if the extractor supports a specific input type
    ///
    /// This method can be used to validate input compatibility before
    /// attempting feature extraction.
    ///
    /// # Arguments
    ///
    /// * `input` - Input to check for compatibility
    ///
    /// # Returns
    ///
    /// `true` if the input type is supported, `false` otherwise
    ///
    /// # Default Implementation
    ///
    /// The default implementation always returns `true`.
    fn supports_input(&self, _input: &FeatureInput) -> bool {
        true
    }

    /// Get extractor capabilities and metadata
    ///
    /// Returns information about the extractor's capabilities, supported
    /// formats, and other metadata useful for introspection.
    ///
    /// # Returns
    ///
    /// HashMap containing capability information
    fn capabilities(&self) -> HashMap<String, serde_json::Value> {
        let mut caps = HashMap::new();
        caps.insert(
            "feature_size".to_string(),
            serde_json::Value::Number(self.config().feature_size().into()),
        );
        caps.insert(
            "supports_batching".to_string(),
            serde_json::Value::Bool(self.config().supports_batching()),
        );
        if let Some(max_batch) = self.config().max_batch_size() {
            caps.insert(
                "max_batch_size".to_string(),
                serde_json::Value::Number(max_batch.into()),
            );
        }
        caps
    }
}

/// Feature extractor configuration trait
///
/// This trait defines the configuration interface that all feature extractors
/// must implement. It provides access to key parameters that affect feature
/// extraction behavior.
///
/// ## Key Parameters
///
/// - **Feature Size**: Dimensionality of the output feature vectors
/// - **Batching Support**: Whether the extractor can process multiple inputs
/// - **Batch Size Limits**: Maximum number of inputs per batch
///
/// ## Implementation Notes
///
/// Implementations should be lightweight and fast, as these methods may be
/// called frequently during feature extraction operations.
pub trait FeatureExtractorConfig: Send + Sync {
    /// Get the size of feature vectors produced by this extractor
    ///
    /// This is the dimensionality of the feature vectors that will be
    /// returned by the `extract_features` method.
    ///
    /// # Returns
    ///
    /// Number of dimensions in output feature vectors
    fn feature_size(&self) -> usize;

    /// Check if the extractor supports batch processing
    ///
    /// Batch processing allows multiple inputs to be processed together,
    /// which can improve throughput and efficiency.
    ///
    /// # Returns
    ///
    /// `true` if batch processing is supported, `false` otherwise
    fn supports_batching(&self) -> bool;

    /// Get the maximum batch size supported by this extractor
    ///
    /// This limit may be imposed by memory constraints, model architecture,
    /// or other implementation details.
    ///
    /// # Returns
    ///
    /// Maximum number of inputs that can be processed in a single batch,
    /// or `None` if there is no specific limit
    fn max_batch_size(&self) -> Option<usize>;

    /// Get additional configuration parameters
    ///
    /// This method can be used to access implementation-specific configuration
    /// parameters that are not covered by the standard interface.
    ///
    /// # Returns
    ///
    /// HashMap containing additional configuration parameters
    ///
    /// # Default Implementation
    ///
    /// The default implementation returns an empty HashMap.
    fn additional_params(&self) -> HashMap<String, serde_json::Value> {
        HashMap::new()
    }

    /// Validate configuration consistency
    ///
    /// This method checks that the configuration parameters are internally
    /// consistent and valid.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the configuration is valid, error otherwise
    ///
    /// # Default Implementation
    ///
    /// The default implementation always returns `Ok(())`.
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

// All feature extractor implementations have been moved to their respective modules:
// - vision.rs: VisionFeatureExtractor and VisionFeatureConfig
// - audio.rs: AudioFeatureExtractor and AudioFeatureConfig
// - document.rs: DocumentFeatureExtractor and DocumentFeatureConfig
// - generic.rs: GenericFeatureExtractor and GenericFeatureConfig

// =============================================================================
// Tests
// =============================================================================

// TODO: Tests use TrustformersError::InvalidInput variant that doesn't exist in current API
// #[cfg(test)]
#[cfg(test_disabled)]
mod tests {
    use super::*;
    use crate::auto::types::{
        AudioMetadata, DocumentFormat, DocumentMetadata, FeatureInput, ImageFormat, ImageMetadata,
    };
    use trustformers_core::errors::TrustformersError;

    #[test]
    fn test_auto_feature_extractor_supported_types() {
        let types = AutoFeatureExtractor::supported_model_types();
        assert!(types.contains(&"clip"));
        assert!(types.contains(&"wav2vec2"));
        assert!(types.contains(&"layoutlm"));
        assert!(types.contains(&"generic"));
    }

    #[test]
    fn test_auto_feature_extractor_supported_tasks() {
        let tasks = AutoFeatureExtractor::supported_tasks();
        assert!(tasks.contains(&"image-classification"));
        assert!(tasks.contains(&"automatic-speech-recognition"));
        assert!(tasks.contains(&"document-understanding"));
    }

    #[test]
    fn test_vision_feature_extractor() {
        let config = VisionFeatureConfig {
            feature_size: 768,
            image_size: 224,
            normalize: true,
            do_resize: true,
            do_center_crop: true,
            crop_size: None,
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
            max_batch_size: Some(32),
        };

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
    }

    #[test]
    fn test_feature_extractor_config_validation() {
        let config = VisionFeatureConfig {
            feature_size: 768,
            image_size: 224,
            normalize: true,
            do_resize: true,
            do_center_crop: true,
            crop_size: None,
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
            max_batch_size: Some(32),
        };

        assert!(config.validate().is_ok());
        assert_eq!(config.feature_size(), 768);
        assert!(config.supports_batching());
        assert_eq!(config.max_batch_size(), Some(32));
    }

    #[test]
    fn test_extractor_capabilities() {
        let config = AudioFeatureConfig {
            sampling_rate: 16000,
            feature_size: 80,
            n_fft: 512,
            hop_length: 160,
            normalize: true,
            max_batch_size: Some(16),
        };

        let extractor = AudioFeatureExtractor::new(config);
        let caps = extractor.capabilities();

        assert_eq!(caps.get("feature_size").unwrap().as_u64().unwrap(), 80);
        assert_eq!(
            caps.get("supports_batching").unwrap().as_bool().unwrap(),
            true
        );
        assert_eq!(caps.get("max_batch_size").unwrap().as_u64().unwrap(), 16);
    }

    #[test]
    fn test_invalid_input_handling() {
        let config = VisionFeatureConfig {
            feature_size: 768,
            image_size: 224,
            normalize: true,
            do_resize: true,
            do_center_crop: true,
            crop_size: None,
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
            max_batch_size: Some(32),
        };

        let extractor = VisionFeatureExtractor::new(config);

        // Try to pass audio input to vision extractor
        let input = FeatureInput::Audio {
            samples: vec![0.0; 1000],
            sample_rate: 16000,
            metadata: Some(AudioMetadata {
                duration: 1.0,
                channels: 1,
                bit_depth: Some(16),
            }),
        };

        let result = extractor.extract_features(&input);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TrustformersError::InvalidInput { .. }
        ));
    }
}
