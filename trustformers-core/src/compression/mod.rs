//! Model Compression Toolkit for TrustformeRS
//!
//! This module provides tools for reducing model size and inference time:
//! - Pruning: Remove unnecessary weights/neurons
//! - Knowledge Distillation: Train smaller models from larger ones
//! - Compression Pipelines: Combine multiple techniques

pub mod distillation;
pub mod metrics;
pub mod pipeline;
pub mod pruning;

pub use pruning::{
    ChannelPruner, FilterPruner, GradualPruner, HeadPruner, LayerPruner, MagnitudePruner,
    PruningConfig, PruningResult, PruningSchedule, PruningStats, PruningStrategy, StructuredPruner,
    UnstructuredPruner,
};

pub use distillation::{
    AttentionDistiller, DistillationConfig, DistillationLoss, DistillationResult,
    DistillationStrategy, FeatureDistiller, HiddenStateDistiller, KnowledgeDistiller,
    LayerDistiller, ResponseDistiller, StudentModel, TeacherModel,
};

pub use pipeline::{
    CompressionConfig, CompressionPipeline, CompressionReport, CompressionResult, CompressionStage,
    PipelineBuilder,
};

pub use metrics::{
    AccuracyRetention, CompressionEvaluator, CompressionMetrics, CompressionRatio,
    CompressionTargets, InferenceSpeedup, ModelSizeReduction, SparsityMetric,
};

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub original_params: usize,
    pub compressed_params: usize,
    pub sparsity: f32,
    pub compression_ratio: f32,
    pub accuracy_retention: Option<f32>,
    pub speedup: Option<f32>,
}

impl CompressionStats {
    pub fn new(original_params: usize, compressed_params: usize) -> Self {
        let sparsity = 1.0 - (compressed_params as f32 / original_params as f32);
        let compression_ratio = original_params as f32 / compressed_params.max(1) as f32;

        Self {
            original_params,
            compressed_params,
            sparsity,
            compression_ratio,
            accuracy_retention: None,
            speedup: None,
        }
    }
}

/// Convenience function to create a compression pipeline
pub fn create_compression_pipeline() -> PipelineBuilder {
    PipelineBuilder::new()
}
