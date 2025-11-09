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
//! ```rust
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
