//! # Auto Module for TrustformeRS
//!
//! This module provides automatic configuration and instantiation of components
//! in the TrustformeRS framework based on model types, tasks, and configurations.
//!
//! ## Features
//!
//! - **Common Types**: Foundational types and structures used across all components
//! - **Feature Extractors**: Automatic feature extraction for different modalities
//! - **Data Collators**: Automatic data collation and batching strategies
//! - **Metrics**: Automatic evaluation metrics for different tasks
//! - **Optimizers**: Automatic optimizer selection and configuration
//!
//! ## Usage
//!
//! The auto module is designed to simplify the setup of machine learning pipelines
//! by automatically selecting appropriate components based on the task and model configuration:
//!
//! ```rust,ignore
//! use trustformers::auto::{
//!     AutoFeatureExtractor, AutoMetric, AutoOptimizer, // Available now
//!     // AutoDataCollator, // TODO: Available in future
//!     FeatureInput, ImageFormat, ImageMetadata
//! };
//!
//! // Create feature extractor automatically from model
//! let feature_extractor = AutoFeatureExtractor::from_pretrained("clip-vit-base-patch32")?;
//!
//! // Create input for feature extraction
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
//! let features = feature_extractor.extract_features(&input)?;
//!
//! // Create metric automatically from task
//! let mut metric = AutoMetric::for_task("text-classification")?;
//!
//! // Add evaluation data
//! let predictions = MetricInput::Classifications(vec![0, 1, 0, 1]);
//! let references = MetricInput::Classifications(vec![0, 0, 1, 1]);
//! metric.add_batch(&predictions, &references)?;
//!
//! // Compute results
//! let result = metric.compute()?;
//! println!("Accuracy: {}", result.details.get("accuracy").unwrap());
//!
//! // Create optimizer automatically from model
//! let optimizer = AutoOptimizer::from_pretrained("bert-base-uncased")?;
//!
//! // Or create for specific task
//! let task_optimizer = AutoOptimizer::for_task("text-classification", &model_config)?;
//!
//! // Add learning rate scheduling
//! let schedule = LearningRateSchedule::LinearWarmup {
//!     warmup_steps: 1000,
//!     max_lr: 5e-5,
//! };
//! let scheduled_optimizer = AutoOptimizer::with_schedule(task_optimizer, schedule);
//! ```

pub mod types;

// Re-export all common types for easy access
pub use types::{
    // Utility functions
    utils,
    AudioMetadata,
    CollatedBatch,

    DataExample,
    DocumentFormat,

    DocumentMetadata,
    // Input/Output types
    FeatureInput,
    FeatureOutput,

    // Format enums
    ImageFormat,
    // Metadata structures
    ImageMetadata,
    MultimodalMetadata,

    // Data collation types
    PaddingStrategy,
    // Common structures
    SpecialToken,

    TextMetadata,
};

// Auto submodules
pub mod feature_extractors;
// TODO: Add remaining auto submodules when they are implemented
pub mod data_collators;
pub mod metrics;
pub mod optimizers;

// Re-export auto classes
pub use feature_extractors::{
    AudioFeatureExtractor, AutoFeatureExtractor, DocumentFeatureExtractor, FeatureExtractor,
    FeatureExtractorConfig, GenericFeatureExtractor, VisionFeatureExtractor,
};
// TODO: Re-export remaining auto classes when modules are restructured
pub use data_collators::*;
pub use metrics::*;
pub use optimizers::{
    AdamConfig, AdamOptimizer, AdamWConfig, AdamWOptimizer, AutoOptimizer, LearningRateSchedule,
    Optimizer, OptimizerGradients, OptimizerUpdate, ScheduledOptimizer,
};

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Re-export smoke tests: verify types are reachable from the auto module
    // -------------------------------------------------------------------------

    #[test]
    fn test_padding_strategy_reachable() {
        let ps = PaddingStrategy::Longest;
        assert!(ps.should_pad(), "PaddingStrategy::Longest should pad");
    }

    #[test]
    fn test_padding_strategy_none_reachable() {
        let ps = PaddingStrategy::None;
        assert!(!ps.should_pad(), "PaddingStrategy::None should not pad");
    }

    #[test]
    fn test_image_format_reachable() {
        let fmt = ImageFormat::Jpeg;
        assert_eq!(fmt.extension(), "jpg", "Jpeg extension should be 'jpg'");
    }

    #[test]
    fn test_image_format_mime_type() {
        let fmt = ImageFormat::Png;
        assert_eq!(fmt.mime_type(), "image/png");
    }

    #[test]
    fn test_document_format_reachable() {
        let fmt = DocumentFormat::Pdf;
        assert_eq!(fmt.extension(), "pdf");
        assert_eq!(fmt.mime_type(), "application/pdf");
    }

    #[test]
    fn test_image_metadata_reachable() {
        let meta = ImageMetadata {
            width: 640,
            height: 480,
            channels: 3,
            dpi: Some(96),
        };
        assert_eq!(meta.width, 640);
        assert_eq!(meta.height, 480);
    }

    #[test]
    fn test_data_example_reachable() {
        let example = DataExample::new(vec![1, 2, 3]);
        assert_eq!(example.sequence_length(), 3);
    }

    #[test]
    fn test_collated_batch_reachable() {
        let batch = CollatedBatch::new(
            vec![vec![1_u32, 2], vec![3_u32, 4]],
            vec![vec![1_u32, 1], vec![1_u32, 1]],
            2,
            2,
        );
        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.total_tokens(), 4);
    }

    #[test]
    fn test_collated_batch_input_shape() {
        let batch = CollatedBatch::new(vec![vec![10_u32, 20, 30]], vec![vec![1_u32, 1, 1]], 1, 3);
        let shape = batch.input_shape();
        assert_eq!(shape, (1, 3));
    }

    #[test]
    fn test_adam_config_reachable() {
        let config = AdamConfig {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            amsgrad: false,
        };
        let diff = (config.learning_rate - 1e-3).abs();
        assert!(diff < 1e-12);
    }

    #[test]
    fn test_adamw_config_reachable() {
        let config = AdamWConfig {
            learning_rate: 2e-5,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
            amsgrad: false,
        };
        let diff = (config.weight_decay - 0.01).abs();
        assert!(diff < 1e-12);
    }

    #[test]
    fn test_lr_schedule_constant_reachable() {
        let s = LearningRateSchedule::Constant;
        assert!(matches!(s, LearningRateSchedule::Constant));
    }

    #[test]
    fn test_optimizer_gradients_reachable() {
        let gradients = OptimizerGradients {
            parameters: std::collections::HashMap::new(),
            parameter_shapes: std::collections::HashMap::new(),
        };
        assert!(gradients.parameters.is_empty());
    }

    #[test]
    fn test_optimizer_update_reachable() {
        let update = OptimizerUpdate {
            parameter_updates: std::collections::HashMap::new(),
            learning_rate: 1e-4,
            step_count: 0,
        };
        assert_eq!(update.step_count, 0);
        let diff = (update.learning_rate - 1e-4).abs();
        assert!(diff < 1e-12);
    }

    #[test]
    fn test_adam_optimizer_create_and_get_lr() {
        let optimizer = AdamOptimizer::new(AdamConfig {
            learning_rate: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            amsgrad: false,
        });
        let diff = (optimizer.get_lr() - 3e-4).abs();
        assert!(diff < 1e-12);
    }

    #[test]
    fn test_adamw_optimizer_create_and_get_lr() {
        let optimizer = AdamWOptimizer::new(AdamWConfig {
            learning_rate: 2e-5,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            eps: 1e-8,
            amsgrad: false,
        });
        let diff = (optimizer.get_lr() - 2e-5).abs();
        assert!(diff < 1e-12);
    }

    #[test]
    fn test_special_token_reachable() {
        let tok = SpecialToken::new("CLS", 0, "[CLS]");
        assert!(tok.is_cls_token(), "Should be identified as CLS token");
    }

    #[test]
    fn test_special_token_sep() {
        let tok = SpecialToken::new("SEP", 5, "[SEP]");
        assert!(tok.is_sep_token(), "Should be identified as SEP token");
    }

    #[test]
    fn test_text_metadata_reachable() {
        let meta = TextMetadata::new()
            .with_language("en")
            .with_encoding("utf-8")
            .with_word_count(20);
        assert_eq!(meta.language.as_deref(), Some("en"));
        assert_eq!(meta.word_count, Some(20));
    }

    #[test]
    fn test_auto_data_collator_for_task_fill_mask() {
        let config = serde_json::json!({"pad_token_id": 0, "mask_token_id": 103});
        let result = AutoDataCollator::for_task("fill-mask", &config);
        assert!(
            result.is_ok(),
            "fill-mask task should produce a valid collator"
        );
    }

    #[test]
    fn test_auto_optimizer_from_config_default_config() {
        let config = serde_json::json!({});
        let result = AutoOptimizer::from_config(&config);
        assert!(result.is_ok(), "from_config with empty JSON should succeed");
        if let Ok(opt) = result {
            assert!(opt.get_lr() > 0.0, "Optimizer LR should be positive");
        }
    }
}
