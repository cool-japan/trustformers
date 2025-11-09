//! TrustformeRS - Main integration crate
//!
//! This crate provides high-level APIs and convenience functions for working with transformer models.

// Allow deprecated APIs for backward compatibility with trustformers-core
#![allow(deprecated)]
// Allow unexpected cfg conditions for future/disabled features
#![allow(unexpected_cfgs)]
// Allow ambiguous glob re-exports (intentional design for convenience API)
#![allow(ambiguous_glob_reexports)]
// Allow dead code, unused variables, unused imports, and unused assignments in incomplete experimental features
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unused_assignments)]
// Allow private types in public APIs for incomplete features
#![allow(private_interfaces)]
// Allow large error types in Result (TrustformersError is large by design for detailed error info)
#![allow(clippy::result_large_err)]
// Allow common patterns in complex integration code
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::excessive_nesting)]
#![allow(clippy::large_enum_variant)]

pub mod auto;
pub mod auto_classes;
pub mod automodel;
pub mod automodel_tasks;
pub mod config_management;
pub mod enhanced_profiler;
pub mod error;
pub mod hub;
pub mod hub_differential;
pub mod hub_local_mirror;
pub mod hub_offline_packs;
pub mod hub_p2p;
#[cfg(feature = "async")]
pub mod hub_ui;
pub mod memory_pool;
pub mod pipeline;
pub mod processor;
pub mod profiler;
pub mod validation;
pub mod zero_copy;

pub use trustformers_core as core;
pub use trustformers_models as models;
pub use trustformers_optim as optim;
pub use trustformers_tokenizers as tokenizers;

pub use core::{
    errors::{Result as CoreResult, TrustformersError},
    gpu::GpuContext,
    performance::{
        BenchmarkConfig, BenchmarkResult, BenchmarkSuite, LatencyMetrics, MemoryMetrics,
        MemoryProfiler, MemorySnapshot as PerformanceMemorySnapshot, MetricsTracker,
        OptimizationAdvisor, OptimizationSuggestion, PerformanceProfiler, ProfileResult,
        ThroughputMetrics,
    },
    tensor::Tensor,
    traits::{Config, Layer, Model, Optimizer, TokenizedInput, Tokenizer},
};

pub use error::{RecoveryAction, RecoveryContext, Result};

pub use processor::{AutoProcessor, Modality, ProcessingResult, ProcessorConfig, ValidationResult};

pub use config_management::{
    ConfigComparison, ConfigDiffer, ConfigFormat, ConfigPreset, ConfigRecommendation,
    ConfigRecommender, ConfigSchema, ConfigValidator, ConfigurationManager, FieldConstraint,
    FieldDiff, FieldSchema, FieldType, Migration, PerformanceRequirements, RecommendationContext,
    RecommendationImpact, ValidationError, ValidationErrorType,
    ValidationResult as ConfigValidationResult, ValidationSeverity, ValidationWarning,
};

pub use tokenizers::{
    BPETokenizer, SentencePieceTokenizer, TokenizerImpl, TokenizerWrapper, WordPieceTokenizer,
};

// Type alias for AutoTokenizer to maintain compatibility with C API
pub type AutoTokenizer = TokenizerWrapper;

#[cfg(feature = "bert")]
pub use models::bert::{
    BertConfig, BertForMaskedLM, BertForQuestionAnswering, BertForSequenceClassification,
    BertForTokenClassification, BertModel,
};

#[cfg(feature = "roberta")]
pub use models::roberta::{
    RobertaConfig, RobertaForMaskedLM, RobertaForQuestionAnswering,
    RobertaForSequenceClassification, RobertaForTokenClassification, RobertaModel,
};

#[cfg(feature = "gpt2")]
pub use models::gpt2::{Gpt2Config, Gpt2LMHeadModel, Gpt2Model};

#[cfg(feature = "gpt_neo")]
pub use models::gpt_neo::{GptNeoConfig, GptNeoLMHeadModel, GptNeoModel};

#[cfg(feature = "gpt_j")]
pub use models::gpt_j::{GptJConfig, GptJLMHeadModel, GptJModel};

#[cfg(feature = "t5")]
pub use models::t5::{T5Config, T5ForConditionalGeneration, T5Model};

pub use optim::{Adam, AdamW, CosineScheduler, LinearScheduler, SGD};

// Re-export auto types for convenience
pub use auto::{
    AudioMetadata, CollatedBatch, DataExample, DocumentFormat, DocumentMetadata, FeatureInput,
    FeatureOutput, ImageFormat, ImageMetadata, MultimodalMetadata, PaddingStrategy, SpecialToken,
    TextMetadata,
};

pub use auto::data_collators::DataCollator;
pub use auto::feature_extractors::FeatureExtractor;
pub use auto::metrics::AutoMetric;
pub use auto::metrics::{CompositeMetric, Metric, MetricInput, MetricResult};
pub use auto::optimizers::AutoOptimizer;
pub use auto::optimizers::{
    AdamOptimizer, AdamWOptimizer, LearningRateSchedule, Optimizer as AutoOptimizerTrait,
    ScheduledOptimizer,
};
pub use auto_classes::{AutoDataCollator, AutoFeatureExtractor};
pub use automodel::{AutoConfig, AutoModel};
pub use automodel_tasks::{
    AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForTokenClassification,
};
pub use enhanced_profiler::{
    global_profiler, init_global_profiler, EnhancedProfiler, ExportFormat as EnhancedExportFormat,
    GlobalMetrics, HardwareInfo, MemoryTracker,
    OptimizationSuggestion as EnhancedOptimizationSuggestion, PerformanceAnalysis,
    PerformanceSample, PerformanceThresholds, ProfilerConfig as EnhancedProfilerConfig,
    SessionSummary,
};
pub use hub_local_mirror::{
    get_hub_mirror, get_model_from_mirror, init_hub_mirror, CachedModel, DownloadProgress,
    DownloadStatus, HubMirror, MirrorConfig, MirrorStats, ModelMetadata,
};
#[cfg(feature = "async")]
pub use hub_ui::{
    start_hub_ui, start_hub_ui_with_config, BenchmarkResult as HubBenchmarkResult,
    CompatibilityInfo, FeatureFlags, HubUiConfig, HubUiServer, HubUiState, ModelMetrics,
    ModelRepository, ModelVersion, PerformanceDiff, ThemeConfig, VersionComparison, VersionStatus,
};
pub use memory_pool::{
    allocate as pool_allocate, deallocate as pool_deallocate, global_pool, init_global_pool,
    MemoryPool, MemoryPoolConfig, MemoryPoolStats, MemoryUsage, PreallocationStrategy,
    ThreadLocalMemoryPool,
};
pub use pipeline::{
    compose_pipelines, pipeline, ComposedPipeline, DocumentUnderstandingPipeline, EnsemblePipeline,
    FillMaskPipeline, MultiModalPipeline, OutputConverter, PipelineChain, PipelineComposer,
    QuestionAnsweringPipeline, SummarizationPipeline, TextClassificationPipeline, TextConverter,
    TextGenerationPipeline, TokenClassificationPipeline, TranslationPipeline,
};

#[cfg(feature = "async")]
pub use pipeline::ConversationalPipeline;
pub use profiler::{
    profile_async, profile_fn, ExportFormat, ProfileResults, ProfileSummary, Profiler,
    ProfilerConfig,
};
pub use validation::{
    ClassificationOutput, ClassificationOutputValidator, ClassificationValidationConfig,
    CompositeValidator, OutputValidator, TextOutputValidator, TextValidationConfig,
    ValidationError as OutputValidationError, ValidationErrorType as OutputValidationErrorType,
    ValidationImpact, ValidationManager, ValidationManagerConfig, ValidationMetrics,
    ValidationResult as OutputValidationResult, ValidationSeverity as OutputValidationSeverity,
    ValidationSuggestion, ValidationSuggestionType, ValidationWarning as OutputValidationWarning,
    ValidationWarningType,
};

pub mod prelude {
    pub use crate::{
        compose_pipelines, get_hub_mirror, get_model_from_mirror, global_pool, init_global_pool,
        init_hub_mirror, pipeline, pool_allocate, pool_deallocate, profile_async, profile_fn,
        AudioMetadata, AutoConfig, AutoDataCollator, AutoFeatureExtractor, AutoMetric, AutoModel,
        AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForTokenClassification,
        AutoOptimizer, AutoProcessor, AutoTokenizer, BenchmarkSuite, CollatedBatch,
        ComposedPipeline, Config, DataCollator, DataExample, DocumentFormat, DocumentMetadata,
        DownloadProgress, FeatureExtractor, FeatureInput, FeatureOutput, HubMirror, ImageFormat,
        ImageMetadata, MemoryPool, MemoryPoolConfig, MemoryPoolStats, MemoryUsage, Metric,
        MirrorConfig, MirrorStats, Modality, Model, MultimodalMetadata, OptimizationAdvisor,
        Optimizer, OutputValidator, PaddingStrategy, PerformanceProfiler, PipelineChain,
        PipelineComposer, PreallocationStrategy, ProcessorConfig, ProfileResult, ProfileResults,
        Profiler, RecoveryAction, RecoveryContext, Result, SpecialToken, TextMetadata,
        TextOutputValidator, Tokenizer, TrustformersError, ValidationError, ValidationManager,
        ValidationManagerConfig, ValidationMetrics, ValidationResult,
    };

    #[cfg(feature = "async")]
    pub use crate::{
        start_hub_ui, start_hub_ui_with_config, HubUiConfig, HubUiServer, ModelRepository,
        ModelVersion, VersionStatus,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
