use crate::auto::data_collators::{
    CausalLanguageModelingCollatorConfig, CausalLanguageModelingDataCollator,
    ClassificationCollatorConfig, ClassificationDataCollator, DataCollator, DefaultCollatorConfig,
    DefaultDataCollator, LanguageModelingCollatorConfig, LanguageModelingDataCollator,
    QuestionAnsweringCollatorConfig, QuestionAnsweringDataCollator, Seq2SeqCollatorConfig,
    Seq2SeqDataCollator,
};
use crate::auto::feature_extractors::{
    AudioFeatureConfig, AudioFeatureExtractor, DocumentFeatureConfig, DocumentFeatureExtractor,
    FeatureExtractor, GenericFeatureConfig, GenericFeatureExtractor, VisionFeatureConfig,
    VisionFeatureExtractor,
};
use crate::error::Result;

// =============================================================================
// TrustFormeRS Auto Classes - Main Entry Point
// =============================================================================
//
// This module provides the complete TrustFormeRS auto classes API, bringing together
// all automatic component creation functionality in a single, convenient location.
//
// Available Auto Classes:
// - AutoFeatureExtractor: Automatic feature extraction for multimodal inputs
// - AutoDataCollator: Automatic data collation for various ML tasks
// - AutoMetric: Automatic metric selection for evaluation (imported from auto::metrics)
// - AutoOptimizer: Automatic optimizer selection and configuration (imported from auto::optimizers)
//
// All auto classes follow the pattern of intelligent defaults with flexible overrides,
// minimizing configuration overhead while maintaining full control when needed.

/// Automatically create feature extractors based on model type and task
#[derive(Debug, Clone)]
pub struct AutoFeatureExtractor;

impl AutoFeatureExtractor {
    /// Create a feature extractor from a pretrained model
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
}

/// Automatically create data collators based on task and data format
#[derive(Debug, Clone)]
pub struct AutoDataCollator;

impl AutoDataCollator {
    /// Create a data collator from model configuration
    pub fn from_pretrained(model_name_or_path: &str) -> Result<Box<dyn DataCollator>> {
        let config = crate::hub::load_config_from_hub(model_name_or_path, None)?;
        Self::from_config(&config)
    }

    /// Create a data collator from configuration
    pub fn from_config(config: &serde_json::Value) -> Result<Box<dyn DataCollator>> {
        let model_type = config.get("model_type").and_then(|v| v.as_str()).unwrap_or("default");

        match model_type {
            "bert" | "roberta" | "electra" => Ok(Box::new(LanguageModelingDataCollator::new(
                LanguageModelingCollatorConfig::from_config(config)?,
            ))),
            "gpt2" | "gpt_neo" | "gpt_j" => Ok(Box::new(CausalLanguageModelingDataCollator::new(
                CausalLanguageModelingCollatorConfig::from_config(config)?,
            ))),
            "t5" | "bart" | "pegasus" => Ok(Box::new(Seq2SeqDataCollator::new(
                Seq2SeqCollatorConfig::from_config(config)?,
            ))),
            _ => Ok(Box::new(DefaultDataCollator::new(
                DefaultCollatorConfig::from_config(config)?,
            ))),
        }
    }

    /// Create a data collator for a specific task
    pub fn for_task(task: &str, config: &serde_json::Value) -> Result<Box<dyn DataCollator>> {
        match task {
            "masked-lm" | "fill-mask" => Ok(Box::new(LanguageModelingDataCollator::new(
                LanguageModelingCollatorConfig::from_config(config)?,
            ))),
            "causal-lm" | "text-generation" => {
                Ok(Box::new(CausalLanguageModelingDataCollator::new(
                    CausalLanguageModelingCollatorConfig::from_config(config)?,
                )))
            },
            "text2text-generation" | "translation" | "summarization" => Ok(Box::new(
                Seq2SeqDataCollator::new(Seq2SeqCollatorConfig::from_config(config)?),
            )),
            "text-classification" | "sentiment-analysis" => Ok(Box::new(
                ClassificationDataCollator::new(ClassificationCollatorConfig::from_config(config)?),
            )),
            "question-answering" => Ok(Box::new(QuestionAnsweringDataCollator::new(
                QuestionAnsweringCollatorConfig::from_config(config)?,
            ))),
            _ => Ok(Box::new(DefaultDataCollator::new(
                DefaultCollatorConfig::from_config(config)?,
            ))),
        }
    }
}

// =============================================================================
// Data Collator Implementations
// =============================================================================

// All data collator implementations have been moved to dedicated modules
// in the auto::data_collators package for better organization:
//
// - seq2seq.rs: Seq2SeqDataCollator for sequence-to-sequence tasks
// - classification.rs: ClassificationDataCollator for text classification
// - question_answering.rs: QuestionAnsweringDataCollator for extractive QA
// - default.rs: DefaultDataCollator as a fallback for unknown tasks
// - language_modeling.rs: LanguageModelingDataCollator and CausalLanguageModelingDataCollator
//
// This refactoring improves code maintainability and provides comprehensive
// documentation and testing for each collator type.

// Note: All metric implementations have been moved to dedicated modules
// in the auto::metrics package for better organization and maintainability.

// =============================================================================
// Optimizer System - Module Integration
// =============================================================================

// All optimizer implementations have been moved to dedicated modules
// in the auto::optimizers package for better organization:
//
// - AutoOptimizer: Main entry point for automatic optimizer creation
// - Optimizer trait: Base interface for all optimizers
// - OptimizerGradients/OptimizerUpdate: Data structures for optimization
// - LearningRateSchedule: Various learning rate scheduling strategies
// - AdamWOptimizer/AdamOptimizer: Concrete optimizer implementations
// - ScheduledOptimizer: Optimizer wrapper with learning rate scheduling
//
// This refactoring improves code maintainability and provides comprehensive
// documentation and testing for each optimizer type.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto::types::{
        AudioMetadata, DocumentFormat, DocumentMetadata, FeatureInput, ImageFormat, ImageMetadata,
    };

    #[test]
    fn test_vision_feature_extractor() {
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
    fn test_audio_feature_extractor() {
        let config = AudioFeatureConfig {
            sampling_rate: 16000,
            feature_size: 80,
            n_fft: 512,
            hop_length: 160,
            normalize: true,
            max_batch_size: Some(16),
        };

        let extractor = AudioFeatureExtractor::new(config);

        let samples: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.001).sin()).collect();
        let input = FeatureInput::Audio {
            samples,
            sample_rate: 16000,
            metadata: Some(AudioMetadata {
                duration: 1.0,
                channels: 1,
                bit_depth: Some(16),
            }),
        };

        let result = extractor.extract_features(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_document_feature_extractor() {
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
    fn test_auto_feature_extractor_from_pretrained() {
        // Test with mock config
        let config = serde_json::json!({
            "model_type": "clip",
            "image_size": 224,
            "hidden_size": 768
        });

        let extractor = AutoFeatureExtractor::for_task("image-classification", &config);
        assert!(extractor.is_ok());

        let fe = extractor.unwrap();
        assert_eq!(fe.config().feature_size(), 768);
        assert!(fe.config().supports_batching());
    }
}
