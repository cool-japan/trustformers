//! # Generic Feature Extractor
//!
//! This module provides a generic feature extractor that serves as a fallback for
//! unsupported model types or when specific feature extractors are not available.
//! The generic extractor provides basic text feature extraction using simple
//! techniques like bag-of-words and TF-IDF-like scoring.
//!
//! ## Overview
//!
//! The generic feature extractor is designed to handle text inputs and provide
//! meaningful feature representations even for models that don't have specialized
//! extractors. It uses position-weighted word embeddings and normalization to
//! create consistent feature vectors.
//!
//! ## Key Features
//!
//! - **Text Processing**: Tokenization and normalization of text input
//! - **Position Weighting**: Earlier words get higher weights in the feature vector
//! - **Hash-Based Features**: Uses simple hashing for vocabulary-independent features
//! - **Normalization**: L2 normalization for consistent feature magnitudes
//! - **Configurable Size**: Supports different feature vector dimensions
//!
//! ## Usage
//!
//! ```rust
//! use trustformers::auto::feature_extractors::generic::{GenericFeatureExtractor, GenericFeatureConfig};
//! use trustformers::auto::types::FeatureInput;
//!
//! let config = GenericFeatureConfig {
//!     feature_size: 768,
//!     max_batch_size: Some(32),
//! };
//!
//! let extractor = GenericFeatureExtractor::new(config);
//!
//! let input = FeatureInput::Text {
//!     content: "This is a sample text for feature extraction.".to_string(),
//!     metadata: None,
//! };
//!
//! let features = extractor.extract_features(&input)?;
//! ```
//!
//! ## Implementation Details
//!
//! The generic extractor uses a simple but effective approach:
//!
//! 1. **Tokenization**: Split text on whitespace and convert to lowercase
//! 2. **Hashing**: Map each word to a feature dimension using simple hash function
//! 3. **Weighting**: Apply position-based weighting (earlier words get higher weight)
//! 4. **Normalization**: Apply L2 normalization to the final feature vector
//!
//! This approach is vocabulary-independent and provides reasonable features for
//! many text classification and similarity tasks.

use crate::auto::feature_extractors::{FeatureExtractor, FeatureExtractorConfig};
use crate::auto::types::{FeatureInput, FeatureOutput};
use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Generic Feature Extractor Implementation
// =============================================================================

/// Generic feature extractor for fallback scenarios
///
/// The `GenericFeatureExtractor` provides basic feature extraction capabilities
/// for models that don't have specialized extractors. It focuses on text input
/// processing using simple but effective techniques.
///
/// ## Design Philosophy
///
/// This extractor prioritizes:
/// - **Simplicity**: Easy to understand and debug
/// - **Robustness**: Works with any text input without dependencies
/// - **Consistency**: Produces stable features across different inputs
/// - **Efficiency**: Minimal computational overhead
///
/// ## Limitations
///
/// As a generic fallback, this extractor has some limitations:
/// - Only supports text input (other modalities will return errors)
/// - Uses simple hashing rather than learned embeddings
/// - No semantic understanding of text content
/// - Fixed feature size regardless of input length
///
/// ## Examples
///
/// ```rust
/// let config = GenericFeatureConfig {
///     feature_size: 512,
///     max_batch_size: Some(16),
/// };
///
/// let extractor = GenericFeatureExtractor::new(config);
///
/// // Extract features from text
/// let input = FeatureInput::Text {
///     content: "Machine learning is fascinating!".to_string(),
///     metadata: None,
/// };
///
/// let output = extractor.extract_features(&input)?;
/// assert_eq!(output.features.len(), 512);
/// ```
#[derive(Debug, Clone)]
pub struct GenericFeatureExtractor {
    config: GenericFeatureConfig,
}

impl GenericFeatureExtractor {
    /// Create a new generic feature extractor
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the feature extractor
    ///
    /// # Returns
    ///
    /// A new instance of the generic feature extractor
    ///
    /// # Examples
    ///
    /// ```rust
    /// let config = GenericFeatureConfig {
    ///     feature_size: 768,
    ///     max_batch_size: Some(32),
    /// };
    /// let extractor = GenericFeatureExtractor::new(config);
    /// ```
    pub fn new(config: GenericFeatureConfig) -> Self {
        Self { config }
    }

    /// Extract text features using bag-of-words with position weighting
    ///
    /// This method implements the core text feature extraction logic:
    /// 1. Tokenize the input text
    /// 2. Hash each word to a feature dimension
    /// 3. Apply position-based weighting
    /// 4. Normalize the resulting feature vector
    ///
    /// # Arguments
    ///
    /// * `text` - Input text string to process
    ///
    /// # Returns
    ///
    /// Feature vector of the configured size
    ///
    /// # Algorithm
    ///
    /// For each word at position i:
    /// - hash = simple_hash(word) % feature_size
    /// - weight = 1.0 / (i + 1)  # Position weighting
    /// - features[hash] += weight
    ///
    /// Finally, L2 normalize the entire vector.
    fn extract_text_features(&self, text: &str) -> Result<Vec<f32>> {
        // Convert to lowercase and tokenize
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();
        let mut features = vec![0.0f32; self.config.feature_size];

        // Handle empty input
        if words.is_empty() {
            return Ok(features);
        }

        // Process each word with position weighting
        for (i, word) in words.iter().enumerate() {
            let hash = self.simple_hash(word) % self.config.feature_size;
            let position_weight = 1.0 / (i + 1) as f32; // Earlier words get higher weight
            features[hash] += position_weight;
        }

        // Apply L2 normalization
        let norm: f32 = features.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for feat in &mut features {
                *feat /= norm;
            }
        }

        Ok(features)
    }

    /// Simple hash function for mapping words to feature dimensions
    ///
    /// This is a basic hash function that provides reasonable distribution
    /// of words across the feature space. It uses a simple polynomial
    /// rolling hash algorithm.
    ///
    /// # Arguments
    ///
    /// * `word` - Word to hash
    ///
    /// # Returns
    ///
    /// Hash value as usize
    ///
    /// # Implementation
    ///
    /// Uses a polynomial rolling hash with prime number 31:
    /// hash = hash * 31 + byte_value
    fn simple_hash(&self, word: &str) -> usize {
        let mut hash = 0usize;
        for byte in word.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as usize);
        }
        hash
    }

    /// Validate input compatibility
    ///
    /// Checks if the provided input is compatible with this extractor.
    /// Currently only supports text input.
    ///
    /// # Arguments
    ///
    /// * `input` - Input to validate
    ///
    /// # Returns
    ///
    /// `Ok(())` if input is valid, error otherwise
    fn validate_input(&self, input: &FeatureInput) -> Result<()> {
        match input {
            FeatureInput::Text { .. } => Ok(()),
            _ => Err(TrustformersError::invalid_input_simple(
                "Generic feature extractor only supports text input".to_string(),
            )),
        }
    }
}

impl FeatureExtractor for GenericFeatureExtractor {
    /// Extract features from the input
    ///
    /// Processes the input and returns feature vectors. Currently supports
    /// only text input; other input types will return an error.
    ///
    /// # Arguments
    ///
    /// * `input` - Input data to process
    ///
    /// # Returns
    ///
    /// Feature output containing the extracted features and metadata
    ///
    /// # Errors
    ///
    /// - `TrustformersError::InvalidInput` if input type is not supported
    /// - `TrustformersError::RuntimeError` if feature extraction fails
    fn extract_features(&self, input: &FeatureInput) -> Result<FeatureOutput> {
        // Validate input type
        self.validate_input(input)?;

        match input {
            FeatureInput::Text { content, metadata } => {
                // Extract features from text
                let features = self.extract_text_features(content)?;

                // Prepare metadata
                let mut output_metadata = HashMap::new();
                output_metadata.insert(
                    "input_type".to_string(),
                    serde_json::Value::String("text".to_string()),
                );
                output_metadata.insert(
                    "word_count".to_string(),
                    serde_json::Value::Number(content.split_whitespace().count().into()),
                );
                output_metadata.insert(
                    "character_count".to_string(),
                    serde_json::Value::Number(content.len().into()),
                );

                // Include input metadata if present
                if let Some(text_meta) = metadata {
                    if let Some(lang) = &text_meta.language {
                        output_metadata.insert(
                            "language".to_string(),
                            serde_json::Value::String(lang.clone()),
                        );
                    }
                    if let Some(encoding) = &text_meta.encoding {
                        output_metadata.insert(
                            "encoding".to_string(),
                            serde_json::Value::String(encoding.clone()),
                        );
                    }
                }

                Ok(FeatureOutput {
                    features,
                    shape: vec![self.config.feature_size],
                    metadata: output_metadata,
                    attention_mask: None,
                    special_tokens: vec![],
                })
            },
            _ => Err(TrustformersError::invalid_input_simple(
                "Generic feature extractor requires text input".to_string(),
            )),
        }
    }

    /// Get the feature extractor configuration
    ///
    /// Returns a reference to the configuration object.
    ///
    /// # Returns
    ///
    /// Reference to the configuration trait object
    fn config(&self) -> &dyn FeatureExtractorConfig {
        &self.config
    }

    /// Check if the extractor supports a specific input type
    ///
    /// The generic extractor only supports text input.
    ///
    /// # Arguments
    ///
    /// * `input` - Input to check
    ///
    /// # Returns
    ///
    /// `true` if input is supported, `false` otherwise
    fn supports_input(&self, input: &FeatureInput) -> bool {
        matches!(input, FeatureInput::Text { .. })
    }

    /// Preprocess input before feature extraction
    ///
    /// For text input, this performs basic cleaning and normalization.
    ///
    /// # Arguments
    ///
    /// * `input` - Input to preprocess
    ///
    /// # Returns
    ///
    /// Preprocessed input
    fn preprocess(&self, input: &FeatureInput) -> Result<FeatureInput> {
        match input {
            FeatureInput::Text { content, metadata } => {
                // Basic text cleaning
                let cleaned_content = content
                    .chars()
                    .filter(|c| {
                        c.is_alphanumeric() || c.is_whitespace() || c.is_ascii_punctuation()
                    })
                    .collect::<String>()
                    .trim()
                    .to_string();

                Ok(FeatureInput::Text {
                    content: cleaned_content,
                    metadata: metadata.clone(),
                })
            },
            _ => Ok(input.clone()),
        }
    }

    /// Get extractor capabilities
    ///
    /// Returns information about what this extractor can do.
    ///
    /// # Returns
    ///
    /// HashMap containing capability information
    fn capabilities(&self) -> HashMap<String, serde_json::Value> {
        let mut caps = HashMap::new();
        caps.insert(
            "supported_modalities".to_string(),
            serde_json::Value::Array(vec![serde_json::Value::String("text".to_string())]),
        );
        caps.insert(
            "feature_size".to_string(),
            serde_json::Value::Number(self.config.feature_size.into()),
        );
        caps.insert(
            "supports_batching".to_string(),
            serde_json::Value::Bool(self.config.supports_batching()),
        );
        if let Some(max_batch) = self.config.max_batch_size() {
            caps.insert(
                "max_batch_size".to_string(),
                serde_json::Value::Number(max_batch.into()),
            );
        }
        caps.insert(
            "extraction_method".to_string(),
            serde_json::Value::String("bag_of_words_with_position_weighting".to_string()),
        );
        caps.insert(
            "normalization".to_string(),
            serde_json::Value::String("l2".to_string()),
        );
        caps
    }
}

// =============================================================================
// Generic Feature Extractor Configuration
// =============================================================================

/// Configuration for the generic feature extractor
///
/// This configuration controls the behavior of the generic feature extractor,
/// including the output feature size and batch processing parameters.
///
/// ## Parameters
///
/// - **feature_size**: Dimensionality of output feature vectors
/// - **max_batch_size**: Maximum number of inputs to process in a single batch
///
/// ## Examples
///
/// ```rust
/// // Create configuration from JSON
/// let config_json = serde_json::json!({
///     "hidden_size": 512,
///     "max_batch_size": 16
/// });
/// let config = GenericFeatureConfig::from_config(&config_json)?;
///
/// // Create configuration manually
/// let config = GenericFeatureConfig {
///     feature_size: 768,
///     max_batch_size: Some(32),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericFeatureConfig {
    /// Size of the output feature vectors
    pub feature_size: usize,
    /// Maximum batch size for processing multiple inputs
    pub max_batch_size: Option<usize>,
}

impl GenericFeatureConfig {
    /// Create configuration from a JSON object
    ///
    /// Parses model configuration and extracts relevant parameters for
    /// the generic feature extractor.
    ///
    /// # Arguments
    ///
    /// * `config` - JSON configuration object
    ///
    /// # Returns
    ///
    /// Parsed configuration or error
    ///
    /// # Configuration Keys
    ///
    /// - `hidden_size` or `feature_size`: Feature vector dimension (default: 768)
    /// - `max_batch_size`: Maximum batch size (optional)
    ///
    /// # Examples
    ///
    /// ```rust
    /// let config = serde_json::json!({
    ///     "hidden_size": 1024,
    ///     "max_batch_size": 64
    /// });
    /// let feature_config = GenericFeatureConfig::from_config(&config)?;
    /// ```
    pub fn from_config(config: &serde_json::Value) -> Result<Self> {
        let feature_size = config
            .get("hidden_size")
            .or_else(|| config.get("feature_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(768) as usize;

        let max_batch_size =
            config.get("max_batch_size").and_then(|v| v.as_u64()).map(|v| v as usize);

        Ok(Self {
            feature_size,
            max_batch_size,
        })
    }

    /// Create a default configuration
    ///
    /// Returns a configuration with sensible defaults suitable for most
    /// general-purpose feature extraction tasks.
    ///
    /// # Returns
    ///
    /// Default configuration with feature_size=768, max_batch_size=32
    pub fn default() -> Self {
        Self {
            feature_size: 768,
            max_batch_size: Some(32),
        }
    }

    /// Validate the configuration
    ///
    /// Checks that all parameters are within valid ranges.
    ///
    /// # Returns
    ///
    /// `Ok(())` if configuration is valid, error otherwise
    ///
    /// # Validation Rules
    ///
    /// - feature_size must be > 0
    /// - max_batch_size must be > 0 if specified
    pub fn validate(&self) -> Result<()> {
        if self.feature_size == 0 {
            return Err(TrustformersError::lconfig_error(
                "Feature size must be greater than 0".to_string(),
            ));
        }

        if let Some(batch_size) = self.max_batch_size {
            if batch_size == 0 {
                return Err(TrustformersError::lconfig_error(
                    "Max batch size must be greater than 0".to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl FeatureExtractorConfig for GenericFeatureConfig {
    /// Get the feature vector size
    ///
    /// # Returns
    ///
    /// Number of dimensions in output feature vectors
    fn feature_size(&self) -> usize {
        self.feature_size
    }

    /// Check if batch processing is supported
    ///
    /// The generic extractor supports batching.
    ///
    /// # Returns
    ///
    /// Always returns `true`
    fn supports_batching(&self) -> bool {
        true
    }

    /// Get the maximum batch size
    ///
    /// # Returns
    ///
    /// Maximum number of inputs per batch, or None if unlimited
    fn max_batch_size(&self) -> Option<usize> {
        self.max_batch_size
    }

    /// Validate configuration consistency
    ///
    /// # Returns
    ///
    /// `Ok(())` if configuration is valid, error otherwise
    fn validate(&self) -> Result<()> {
        self.validate()
    }

    /// Get additional configuration parameters
    ///
    /// # Returns
    ///
    /// HashMap with additional parameters specific to the generic extractor
    fn additional_params(&self) -> HashMap<String, serde_json::Value> {
        let mut params = HashMap::new();
        params.insert(
            "extraction_method".to_string(),
            serde_json::Value::String("bag_of_words".to_string()),
        );
        params.insert(
            "normalization".to_string(),
            serde_json::Value::String("l2".to_string()),
        );
        params.insert(
            "position_weighting".to_string(),
            serde_json::Value::Bool(true),
        );
        params
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto::types::TextMetadata;

    #[test]
    fn test_generic_feature_extractor_creation() {
        let config = GenericFeatureConfig {
            feature_size: 512,
            max_batch_size: Some(16),
        };

        let extractor = GenericFeatureExtractor::new(config);
        assert_eq!(extractor.config().feature_size(), 512);
        assert!(extractor.config().supports_batching());
        assert_eq!(extractor.config().max_batch_size(), Some(16));
    }

    #[test]
    fn test_text_feature_extraction() {
        let config = GenericFeatureConfig {
            feature_size: 128,
            max_batch_size: Some(8),
        };

        let extractor = GenericFeatureExtractor::new(config);

        let input = FeatureInput::Text {
            content: "This is a test sentence for feature extraction.".to_string(),
            metadata: Some(TextMetadata::new().with_language("en")),
        };

        let result = extractor.extract_features(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.features.len(), 128);
        assert_eq!(output.shape, vec![128]);
        assert!(output.metadata.contains_key("word_count"));
        assert!(output.metadata.contains_key("language"));
    }

    #[test]
    fn test_empty_text_handling() {
        let config = GenericFeatureConfig {
            feature_size: 64,
            max_batch_size: None,
        };

        let extractor = GenericFeatureExtractor::new(config);

        let input = FeatureInput::Text {
            content: "".to_string(),
            metadata: None,
        };

        let result = extractor.extract_features(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.features.len(), 64);
        // All features should be 0.0 for empty input
        assert!(output.features.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_invalid_input_type() {
        let config = GenericFeatureConfig::default();
        let extractor = GenericFeatureExtractor::new(config);

        let input = FeatureInput::Audio {
            samples: vec![0.0; 1000],
            sample_rate: 16000,
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
    fn test_supports_input() {
        let config = GenericFeatureConfig::default();
        let extractor = GenericFeatureExtractor::new(config);

        let text_input = FeatureInput::Text {
            content: "test".to_string(),
            metadata: None,
        };
        assert!(extractor.supports_input(&text_input));

        let audio_input = FeatureInput::Audio {
            samples: vec![0.0],
            sample_rate: 16000,
            metadata: None,
        };
        assert!(!extractor.supports_input(&audio_input));
    }

    #[test]
    fn test_config_from_json() {
        let config_json = serde_json::json!({
            "hidden_size": 1024,
            "max_batch_size": 64
        });

        let config = GenericFeatureConfig::from_config(&config_json);
        assert!(config.is_ok());

        let config = config.unwrap();
        assert_eq!(config.feature_size, 1024);
        assert_eq!(config.max_batch_size, Some(64));
    }

    #[test]
    fn test_config_validation() {
        let valid_config = GenericFeatureConfig {
            feature_size: 768,
            max_batch_size: Some(32),
        };
        assert!(valid_config.validate().is_ok());

        let invalid_config = GenericFeatureConfig {
            feature_size: 0,
            max_batch_size: Some(32),
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_simple_hash_consistency() {
        let config = GenericFeatureConfig::default();
        let extractor = GenericFeatureExtractor::new(config);

        // Same word should always hash to same value
        let hash1 = extractor.simple_hash("test");
        let hash2 = extractor.simple_hash("test");
        assert_eq!(hash1, hash2);

        // Different words should (usually) hash to different values
        let hash3 = extractor.simple_hash("different");
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_feature_normalization() {
        let config = GenericFeatureConfig {
            feature_size: 100,
            max_batch_size: None,
        };

        let extractor = GenericFeatureExtractor::new(config);

        let input = FeatureInput::Text {
            content: "word1 word2 word3".to_string(),
            metadata: None,
        };

        let result = extractor.extract_features(&input);
        assert!(result.is_ok());

        let output = result.unwrap();

        // Check L2 normalization: sum of squares should be approximately 1.0
        let norm_squared: f32 = output.features.iter().map(|&x| x * x).sum();
        assert!((norm_squared - 1.0).abs() < 1e-6 || norm_squared == 0.0);
    }

    #[test]
    fn test_text_preprocessing() {
        let config = GenericFeatureConfig::default();
        let extractor = GenericFeatureExtractor::new(config);

        let input = FeatureInput::Text {
            content: "  Hello,    World!   \n\t  ".to_string(),
            metadata: None,
        };

        let preprocessed = extractor.preprocess(&input);
        assert!(preprocessed.is_ok());

        if let FeatureInput::Text { content, .. } = preprocessed.unwrap() {
            assert_eq!(content, "Hello,    World!");
        } else {
            panic!("Expected text input after preprocessing");
        }
    }

    #[test]
    fn test_extractor_capabilities() {
        let config = GenericFeatureConfig {
            feature_size: 256,
            max_batch_size: Some(16),
        };

        let extractor = GenericFeatureExtractor::new(config);
        let caps = extractor.capabilities();

        assert!(caps.contains_key("supported_modalities"));
        assert!(caps.contains_key("extraction_method"));
        assert!(caps.contains_key("normalization"));
        assert_eq!(caps.get("feature_size").unwrap().as_u64().unwrap(), 256);
    }
}
