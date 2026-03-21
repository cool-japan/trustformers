use crate::error::{Result, TrustformersError};
use crate::{AutoModel, AutoTokenizer};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use trustformers_core::cache::{CacheConfig, InferenceCache};
use trustformers_models::GenerativeModel;

/// Common input format for pipelines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineInput {
    /// Text input
    Text(String),
    /// Token input
    Tokens(Vec<u32>),
    /// Batch text input
    BatchText(Vec<String>),
    /// Batch token input
    BatchTokens(Vec<Vec<u32>>),
}

pub mod adaptive_batching;
pub mod adaptive_inference;
pub mod advanced_caching;
pub mod advanced_rag;
pub mod composition;
#[cfg(feature = "async")]
pub mod conversational;
pub mod coreml_backend;
pub mod custom_backend;
pub mod document_understanding;
pub mod dynamic_batching;
pub mod early_exit;
pub mod ensemble;
pub mod fill_mask;
pub mod image_to_text;
pub mod jit_compilation;
pub mod mamba2_pipeline;
pub mod metal_backend;
pub mod mixture_of_depths;
pub mod multimodal;
pub mod onnx_backend;
pub mod openvino_backend;
pub mod question_answering;
pub mod speculative_decoding;
pub mod speech_to_text;
pub mod streaming;
pub mod summarization;
pub mod tensorrt_backend;
pub mod text_classification;
pub mod text_generation;
pub mod text_to_speech;
pub mod token_classification;
pub mod translation;
pub mod visual_question_answering;

pub use adaptive_batching::{
    AdaptiveBatchConfig, AdaptiveBatchOptimizer, BatchComparison, BatchSizeStats,
    PerformanceReport, PerformanceSample,
};
pub use adaptive_inference::{
    create_adaptive_inference_pipeline, create_balanced_adaptive_pipeline,
    create_energy_efficient_pipeline, create_latency_optimized_pipeline,
    create_memory_efficient_pipeline, AdaptationDecision, AdaptiveInferenceConfig,
    AdaptiveInferenceEngine, AdaptiveInferenceResult, ConditionalStrategy, InputAnalysis,
    LayerAnalysis, PerformanceMetrics, PrecisionMode, ResourceAllocation, ResourceStrategy,
};
pub use advanced_caching::{
    AccessPattern, AdvancedCacheConfig, AdvancedLRUCache, CacheEntry, CachePriority, CacheStats,
    PipelineCacheKeyBuilder,
};
pub use composition::{
    compose_pipelines, ComposedPipeline, OutputConverter, PipelineChain, PipelineComposer,
    TextConverter,
};
#[cfg(feature = "async")]
pub use conversational::{
    ConversationMode, ConversationRole, ConversationState, ConversationStats, ConversationTurn,
    ConversationalConfig, ConversationalInput, ConversationalOutput, ConversationalPipeline,
    GenerationStats, PersonaConfig, SafetyFilter,
};
pub use custom_backend::{
    create_backend, create_custom_backend_pipeline, create_custom_text_classification_pipeline,
    create_custom_text_generation_pipeline, get_backend, list_available_backends,
    list_available_factories, register_backend_factory, BackendCapabilities, BackendConfig,
    BackendFactory, BackendHealth, BackendMetrics, BackendModel, BackendRegistry, BackendTensor,
    CustomBackend, CustomBackendPipeline, DataType, FactoryInfo, HealthStatus, MemoryConfig,
    MemoryLayout, MemoryStats, MemoryUsage, ModelMetadata, ModelPerformanceStats,
    OptimizationLevel, PerformanceConfig, PerformanceIndicators, QuantizationMode,
    TensorConstraints, TensorSpec, GLOBAL_BACKEND_REGISTRY,
};
pub use document_understanding::{
    BoundingBox as DocumentBoundingBox,
    DocumentEntity,
    DocumentMetadata,
    DocumentUnderstandingConfig,
    DocumentUnderstandingInput,
    DocumentUnderstandingOutput,
    DocumentUnderstandingPipeline,
    KeyValuePair,
    OCRResult,
    Table,
    TextBlock,
    TextBlockType, // document_understanding_pipeline, // Removed to avoid duplicate with function definition
};
pub use dynamic_batching::{
    BatchRequest,
    BatchingStats,
    DynamicBatchPipeline,
    DynamicBatcher,
    DynamicBatchingConfig,
    // PerformanceMetrics, // Removed duplicate import (already imported above)
    RequestPriority,
};
pub use early_exit::{
    create_adaptive_early_exit, create_budget_constrained_early_exit,
    create_confidence_based_early_exit, create_early_exit_pipeline, EarlyExitConfig,
    EarlyExitPipeline, EarlyExitPredictor, EarlyExitResult, ExitPoint, ExitStrategy, LayerOutput,
};
pub use ensemble::{
    create_adaptive_voting_ensemble, create_cascade_ensemble, create_classification_ensemble,
    create_consensus_ensemble, create_dynamic_ensemble, create_dynamic_routing_ensemble,
    create_efficient_ensemble, create_ensemble_pipeline, create_generation_ensemble,
    create_high_performance_ensemble, create_qa_ensemble, create_quality_latency_ensemble,
    create_resource_aware_ensemble, create_uncertainty_ensemble, CascadeResult, EnsembleConfig,
    EnsembleModel, EnsemblePipeline, EnsemblePrediction, EnsembleStrategy, InputCharacteristics,
    ModelSelectionInfo, ModelSelectionStrategy, ModelWeight,
};
pub use fill_mask::FillMaskPipeline;
#[cfg(feature = "vision")]
pub use image_to_text::{ImageToTextInput, ImageToTextOutput, ImageToTextPipeline};
pub use jit_compilation::{
    AnomalyDetector, AnomalySeverity, AnomalyType, CompilationPriority, CompilationStrategy,
    CompilationThresholds, CompiledPipeline, DataLayout, ExecutionPercentiles, ExecutionStats,
    OptimizationHints, OptimizationType, PerformanceAnomaly,
    PerformanceSample as JitPerformanceSample, PerformanceTracker, PerformanceTrend,
    PipelineJitCompiler, PipelineJitConfig, PipelinePerformanceMetrics, TargetHardware,
    ThermalMetrics, TrendDirection,
};
pub use mamba2_pipeline::{
    create_high_performance_mamba2_pipeline, create_memory_efficient_mamba2_pipeline,
    create_ultra_long_sequence_mamba2_pipeline, ChunkingStrategy, HardwareStrategy, Mamba2Config,
    Mamba2Output, Mamba2PerformanceMetrics, Mamba2Pipeline, MemoryOptimization,
};
pub use multimodal::{
    multimodal_pipeline, AttentionConfig, AttentionWeights, ClassificationResult,
    FusionStrategy as MultiModalFusionStrategy, ModalityFeatures, MultiModalConfig,
    MultiModalInput, MultiModalOutput, MultiModalPipeline, ProcessingMetadata,
};
pub use onnx_backend::{
    onnx_text_classification_pipeline, onnx_text_generation_pipeline, ONNXBackendConfig,
    ONNXBasePipeline, ONNXModel, ONNXPipelineManager, ONNXPipelineOptions,
    ONNXTextClassificationPipeline, ONNXTextGenerationPipeline, ONNXTokenizer,
};
pub use openvino_backend::{
    openvino_text_classification_pipeline, openvino_text_generation_pipeline, ExecutionPriority,
    OpenVINOBackendConfig, OpenVINOBasePipeline, OpenVINOModel, OpenVINOPipelineManager,
    OpenVINOPipelineOptions, OpenVINOTextClassificationPipeline, OpenVINOTextGenerationPipeline,
    OpenVINOTokenizer, PerformanceHint,
};
pub use tensorrt_backend::{
    tensorrt_text_classification_pipeline, tensorrt_text_generation_pipeline,
    TensorRTBackendConfig, TensorRTBasePipeline, TensorRTModel, TensorRTPipelineManager,
    TensorRTPipelineOptions, TensorRTTextClassificationPipeline, TensorRTTextGenerationPipeline,
    TensorRTTokenizer,
};

// Export enhanced pipeline factory and backend types
// Backend is defined in this module, so no need to import
// pub use enhanced_pipeline; // Commented out to avoid duplicate definition with function below
pub use question_answering::QuestionAnsweringPipeline;
#[cfg(feature = "audio")]
pub use speech_to_text::{
    AudioInput, SpeechTask, SpeechToTextConfig, SpeechToTextOutput, SpeechToTextPipeline,
};
pub use streaming::{
    AdaptiveBatcher, AggregatorConfig, BackpressureController, PartialResult,
    PartialResultAggregator, PriorityItem, RealTimeConfig, RealTimeProcessor, RealTimeStats,
    StreamConfig, StreamProcessor, StreamResult, StreamResultStream, StreamStats,
    StreamTransformer, StreamingPipeline,
};
pub use summarization::SummarizationPipeline;
pub use text_classification::TextClassificationPipeline;
pub use text_generation::TextGenerationPipeline;
pub use text_to_speech::{
    AudioFormat, EmphasisInfo, EmphasisType, PauseInfo, PauseType, PhonemeTimings, ProsodyInfo,
    ProsodyMarker, ProsodyType, TextToSpeechConfig, TextToSpeechInput, TextToSpeechOutput,
    TextToSpeechPipeline,
};
pub use token_classification::TokenClassificationPipeline;
pub use translation::TranslationPipeline;
#[cfg(feature = "vision")]
pub use visual_question_answering::{
    AnswerCandidate, AnswerGenerationStrategy, AttentionVisualization, BoundingBox, DetectedObject,
    FusionStrategy, ImageFeatures, ImageInput, ReasoningStep, ReasoningStepType,
    VisualQuestionAnsweringConfig, VisualQuestionAnsweringInput, VisualQuestionAnsweringOutput,
    VisualQuestionAnsweringPipeline,
};

/// Base trait for all pipelines
pub trait Pipeline: Send + Sync {
    type Input;
    type Output;

    /// Main entry point for pipeline processing
    fn __call__(&self, inputs: Self::Input) -> Result<Self::Output>;

    /// Process multiple inputs in a batch
    fn batch(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        inputs.into_iter().map(|input| self.__call__(input)).collect()
    }

    /// Process multiple inputs with adaptive batch sizing
    fn adaptive_batch(
        &self,
        inputs: Vec<Self::Input>,
        config: Option<DynamicBatchingConfig>,
    ) -> Result<Vec<Self::Output>>
    where
        Self::Input: Clone,
    {
        let config = config.unwrap_or_default();

        // For now, implement a simple adaptive strategy
        // In a real implementation, this would use performance monitoring
        let optimal_batch_size = std::cmp::min(inputs.len(), config.max_batch_size);

        if inputs.len() <= optimal_batch_size {
            self.batch(inputs)
        } else {
            // Process in chunks
            let mut results = Vec::with_capacity(inputs.len());
            for chunk in inputs.chunks(optimal_batch_size) {
                let chunk_results = self.batch(chunk.to_vec())?;
                results.extend(chunk_results);
            }
            Ok(results)
        }
    }

    /// Create a streaming processor with enhanced capabilities
    fn create_stream_processor(
        &self,
        config: StreamConfig,
    ) -> StreamProcessor<Self::Input, Self::Output, String>
    where
        Self: Clone + 'static,
        Self::Input: Send + Sync + 'static,
        Self::Output: Send + Sync + 'static,
    {
        // This default implementation assumes the pipeline also implements StreamingPipeline
        // Implementers can override this method for custom behavior
        StreamProcessor::<Self::Input, Self::Output, String>::new_from_pipeline(
            self.clone(),
            config,
        )
    }

    /// Create a real-time processor for low-latency scenarios
    fn create_realtime_processor(
        &self,
        config: RealTimeConfig,
    ) -> Result<RealTimeProcessor<Self::Input, Self::Output, String>>
    where
        Self: Clone + 'static,
        Self::Input: Send + Sync + 'static,
        Self::Output: Send + Sync + 'static,
    {
        // This is a placeholder implementation. Real implementations should
        // implement StreamingPipeline and use RealTimeProcessor::new
        Err(TrustformersError::feature_unavailable(
            "Real-time processor not implemented for this pipeline".to_string(),
            "real_time_processing".to_string(),
        ))
    }
}

/// Async pipeline trait for concurrent processing
#[cfg(feature = "async")]
#[async_trait::async_trait]
pub trait AsyncPipeline: Send + Sync {
    type Input: Send + Sync;
    type Output: Send;

    /// Async processing of single input
    async fn __call_async__(&self, input: Self::Input) -> Result<Self::Output>;

    /// Async batch processing with concurrent execution
    async fn batch_async(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        use futures::future::join_all;

        let futures = inputs.into_iter().map(|input| self.__call_async__(input));

        let results = join_all(futures).await;
        results.into_iter().collect()
    }

    /// Async adaptive batch processing with dynamic sizing
    async fn adaptive_batch_async(
        &self,
        inputs: Vec<Self::Input>,
        config: Option<DynamicBatchingConfig>,
    ) -> Result<Vec<Self::Output>>
    where
        Self::Input: Clone,
    {
        let config = config.unwrap_or_default();
        let optimal_batch_size = std::cmp::min(inputs.len(), config.max_batch_size);

        if inputs.len() <= optimal_batch_size {
            self.batch_async(inputs).await
        } else {
            // Process in chunks concurrently but with controlled parallelism
            let mut results = Vec::with_capacity(inputs.len());
            for chunk in inputs.chunks(optimal_batch_size) {
                let chunk_results = self.batch_async(chunk.to_vec()).await?;
                results.extend(chunk_results);
            }
            Ok(results)
        }
    }

    /// Create a dynamic batcher for this async pipeline
    fn create_async_batcher(&self, config: DynamicBatchingConfig) -> DynamicBatcher<Self::Input>
    where
        Self::Input: Clone + Send + Sync + 'static,
    {
        DynamicBatcher::new(config)
    }
}

/// Options for pipeline creation
#[derive(Clone, Debug)]
pub struct PipelineOptions {
    pub model: Option<String>,
    pub tokenizer: Option<String>,
    pub device: Option<Device>,
    pub batch_size: Option<usize>,
    pub max_length: Option<usize>,
    pub truncation: bool,
    pub padding: PaddingStrategy,
    pub num_threads: Option<usize>,
    pub cache_config: Option<CacheConfig>,
    pub backend: Option<Backend>,
    pub onnx_config: Option<ONNXBackendConfig>,
    pub tensorrt_config: Option<TensorRTBackendConfig>,
    pub streaming: bool,
}

/// Backend specification for pipeline execution
#[derive(Clone, Debug)]
pub enum Backend {
    /// Native TrustformeRS backend
    Native,
    /// ONNX Runtime backend
    ONNX { model_path: std::path::PathBuf },
    /// TensorRT backend
    TensorRT { model_path: std::path::PathBuf },
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            model: None,
            tokenizer: None,
            device: None,
            batch_size: Some(1),
            max_length: Some(512),
            truncation: true,
            padding: PaddingStrategy::Longest,
            num_threads: None,
            cache_config: Some(CacheConfig::default()),
            backend: Some(Backend::Native),
            onnx_config: None,
            tensorrt_config: None,
            streaming: false,
        }
    }
}

/// Padding strategy for batch processing
#[derive(Clone, Debug)]
pub enum PaddingStrategy {
    /// No padding
    None,
    /// Pad to the longest sequence in the batch
    Longest,
    /// Pad to a fixed length
    MaxLength(usize),
}

/// Device specification for pipeline execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Gpu(usize),
}

/// Factory function to create pipelines
pub fn pipeline(
    task: &str,
    model: Option<&str>,
    options: Option<PipelineOptions>,
) -> Result<Box<dyn Pipeline<Input = String, Output = PipelineOutput>>> {
    let opts = options.unwrap_or_default();
    let model_name = model.or(opts.model.as_deref());

    match task {
        "sentiment-analysis" | "text-classification" => {
            let model_name = model_name.unwrap_or("bert-base-uncased");
            let model = AutoModel::from_pretrained(model_name)?;
            let tokenizer = AutoTokenizer::from_pretrained(model_name)?;

            Ok(Box::new(TextClassificationPipeline::new(model, tokenizer)?))
        },
        "text-generation" => {
            let model_name = model_name.unwrap_or("gpt2");
            let model = AutoModel::from_pretrained(model_name)?;
            let tokenizer = AutoTokenizer::from_pretrained(model_name)?;

            Ok(Box::new(TextGenerationPipeline::new(model, tokenizer)?))
        },
        "ner" | "token-classification" => {
            let model_name = model_name.unwrap_or("bert-base-cased");
            let model = AutoModel::from_pretrained(model_name)?;
            let tokenizer = AutoTokenizer::from_pretrained(model_name)?;

            Ok(Box::new(TokenClassificationPipeline::new(
                model, tokenizer,
            )?))
        },
        "question-answering" => {
            let model_name = model_name.unwrap_or("bert-base-cased");
            let model = AutoModel::from_pretrained(model_name)?;
            let tokenizer = AutoTokenizer::from_pretrained(model_name)?;

            Ok(Box::new(QuestionAnsweringPipeline::new(model, tokenizer)?))
        },
        "fill-mask" => {
            let model_name = model_name.unwrap_or("bert-base-uncased");
            let model = AutoModel::from_pretrained(model_name)?;
            let tokenizer = AutoTokenizer::from_pretrained(model_name)?;

            Ok(Box::new(FillMaskPipeline::new(model, tokenizer)?))
        },
        "summarization" => {
            let model_name = model_name.unwrap_or("t5-small");
            let model = AutoModel::from_pretrained(model_name)?;
            let tokenizer = AutoTokenizer::from_pretrained(model_name)?;

            Ok(Box::new(SummarizationPipeline::new(model, tokenizer)?))
        },
        "translation" => {
            let model_name = model_name.unwrap_or("t5-small");
            let model = AutoModel::from_pretrained(model_name)?;
            let tokenizer = AutoTokenizer::from_pretrained(model_name)?;

            Ok(Box::new(TranslationPipeline::new(model, tokenizer)?))
        },
        // Note: Speech pipelines use different Input/Output types than the generic pipeline
        // They should be instantiated directly rather than through this function
        // #[cfg(feature = "audio")]
        // "automatic-speech-recognition" | "speech-to-text" => {
        //     let model_name = model_name.unwrap_or("openai/whisper-base");
        //     let model = AutoModel::from_pretrained(model_name)?;
        //     let tokenizer = AutoTokenizer::from_pretrained(model_name)?;
        //     Ok(Box::new(SpeechToTextPipeline::new(model, tokenizer)?))
        // },
        // #[cfg(feature = "audio")]
        // "text-to-speech" | "tts" => {
        //     let model_name = model_name.unwrap_or("microsoft/speecht5_tts");
        //     let model = AutoModel::from_pretrained(model_name)?;
        //     let tokenizer = AutoTokenizer::from_pretrained(model_name)?;
        //     Ok(Box::new(TextToSpeechPipeline::new(model, tokenizer)?))
        // },
        #[cfg(feature = "vision")]
        "visual-question-answering" | "vqa" => {
            let model_name = model_name.unwrap_or("dandelin/vilt-b32-finetuned-vqa");
            let model = AutoModel::from_pretrained(model_name)?;
            let tokenizer = AutoTokenizer::from_pretrained(model_name)?;

            let pipeline = VisualQuestionAnsweringPipeline::new(model, tokenizer)?;
            Ok(Box::new(VisualQuestionAnsweringPipelineWrapper(pipeline)))
        },
        "document-understanding" | "document-ai" => {
            let model_name = model_name.unwrap_or("microsoft/layoutlmv3-base");
            let model = AutoModel::from_pretrained(model_name)?;
            let tokenizer = AutoTokenizer::from_pretrained(model_name)?;

            let pipeline = DocumentUnderstandingPipeline::new(model, tokenizer)?;
            Ok(Box::new(DocumentUnderstandingPipelineWrapper(pipeline)))
        },
        "multimodal" | "multi-modal" => {
            let model_name = model_name.unwrap_or("openai/clip-vit-base-patch32");
            let model = AutoModel::from_pretrained(model_name)?;
            let tokenizer = AutoTokenizer::from_pretrained(model_name)?;

            let pipeline = MultiModalPipeline::new(model, tokenizer)?;
            Ok(Box::new(MultiModalPipelineWrapper(pipeline)))
        },
#[cfg(feature = "async")]
        "conversational" | "chat" | "dialogue" => {
            let model_name = model_name.unwrap_or("microsoft/DialoGPT-medium");
            let model = AutoModel::from_pretrained(model_name)?;
            let tokenizer = AutoTokenizer::from_pretrained(model_name)?;

            let pipeline = ConversationalPipeline::new(model, tokenizer)?;
            Ok(Box::new(ConversationalPipelineWrapper(pipeline)))
        },
        "adaptive-inference" | "adaptive" => {
            let model_name = model_name.unwrap_or("bert-base-uncased");
            let model = AutoModel::from_pretrained(model_name)?;
            let tokenizer = AutoTokenizer::from_pretrained(model_name)?;

            // Create a base text classification pipeline
            let base_pipeline = TextClassificationPipeline::new(model, tokenizer)?;

            // Wrap with adaptive inference
            let adaptive_config = AdaptiveInferenceConfig::default();
            let adaptive_pipeline = create_adaptive_inference_pipeline(base_pipeline, adaptive_config);

            Ok(Box::new(AdaptivePipelineWrapper::new(adaptive_pipeline)))
        },
        _ => Err(TrustformersError::invalid_input_simple(format!(
            "Unknown pipeline task: {}. For image-to-text pipelines, use image_to_text_pipeline() function instead. For speech-to-text pipelines, use speech_to_text_pipeline() function instead. For text-to-speech pipelines, use text_to_speech_pipeline() function instead. For visual question answering pipelines, use visual_question_answering_pipeline() function instead. For document understanding pipelines, use document_understanding_pipeline() function instead. For multimodal pipelines, use multimodal_pipeline() function instead. For conversational pipelines, use conversational_pipeline() function instead.",
            task
        ))),
    }
}

/// Enhanced pipeline factory that supports both native and ONNX backends
pub fn enhanced_pipeline(
    task: &str,
    model: Option<&str>,
    options: Option<PipelineOptions>,
) -> Result<Box<dyn Pipeline<Input = String, Output = PipelineOutput>>> {
    let opts = options.unwrap_or_default();

    match opts.backend.as_ref().unwrap_or(&Backend::Native) {
        Backend::Native => {
            // Use existing native pipeline factory
            pipeline(task, model, Some(opts))
        },
        Backend::ONNX { model_path } => create_onnx_pipeline(task, model_path, &opts),
        Backend::TensorRT { model_path } => create_tensorrt_pipeline(task, model_path, &opts),
    }
}

/// Create ONNX-backed pipeline
fn create_onnx_pipeline(
    task: &str,
    model_path: &std::path::Path,
    opts: &PipelineOptions,
) -> Result<Box<dyn Pipeline<Input = String, Output = PipelineOutput>>> {
    use crate::AutoTokenizer;

    // Load tokenizer (still use native tokenizer)
    let tokenizer_name = opts
        .tokenizer
        .as_deref()
        .or(opts.model.as_deref())
        .unwrap_or("bert-base-uncased");
    let tokenizer = AutoTokenizer::from_pretrained(tokenizer_name)?;

    // Create ONNX config based on options
    let onnx_config = if let Some(config) = &opts.onnx_config {
        config.clone()
    } else {
        match opts.device.as_ref().unwrap_or(&Device::Cpu) {
            Device::Cpu => ONNXBackendConfig::cpu_optimized(model_path.to_path_buf()),
            Device::Gpu(device_id) => {
                ONNXBackendConfig::gpu_optimized(model_path.to_path_buf(), Some(*device_id as i32))
            },
        }
    };

    match task {
        "sentiment-analysis" | "text-classification" => {
            let pipeline =
                onnx_text_classification_pipeline(model_path, tokenizer, Some(onnx_config))?;
            Ok(Box::new(OnnxPipelineWrapper::Classification(pipeline)))
        },
        "text-generation" => {
            let mut pipeline =
                onnx_text_generation_pipeline(model_path, tokenizer, Some(onnx_config))?;

            if let Some(max_length) = opts.max_length {
                pipeline = pipeline.with_max_new_tokens(max_length);
            }

            Ok(Box::new(OnnxPipelineWrapper::Generation(pipeline)))
        },
        _ => Err(TrustformersError::invalid_input_simple(format!(
            "ONNX backend not yet implemented for task: {}",
            task
        ))),
    }
}

/// Create TensorRT-backed pipeline
fn create_tensorrt_pipeline(
    task: &str,
    model_path: &std::path::Path,
    opts: &PipelineOptions,
) -> Result<Box<dyn Pipeline<Input = String, Output = PipelineOutput>>> {
    use crate::AutoTokenizer;

    // Load tokenizer (still use native tokenizer)
    let tokenizer_name = opts
        .tokenizer
        .as_deref()
        .or(opts.model.as_deref())
        .unwrap_or("bert-base-uncased");
    let tokenizer = AutoTokenizer::from_pretrained(tokenizer_name)?;

    // Create TensorRT config based on options
    let tensorrt_config = if let Some(config) = &opts.tensorrt_config {
        config.clone()
    } else {
        match opts.device.as_ref().unwrap_or(&Device::Cpu) {
            Device::Cpu => {
                // TensorRT typically runs on GPU, but we can create a CPU-fallback config
                TensorRTBackendConfig::latency_optimized(model_path.to_path_buf())
            },
            Device::Gpu(device_id) => {
                let mut config = TensorRTBackendConfig::latency_optimized(model_path.to_path_buf());
                config.device_id = *device_id as i32;
                config
            },
        }
    };

    match task {
        "sentiment-analysis" | "text-classification" => {
            let pipeline = tensorrt_text_classification_pipeline(
                model_path,
                tokenizer,
                Some(tensorrt_config),
            )?;
            Ok(Box::new(TensorRTPipelineWrapper::Classification(pipeline)))
        },
        "text-generation" => {
            let mut pipeline =
                tensorrt_text_generation_pipeline(model_path, tokenizer, Some(tensorrt_config))?;

            if let Some(max_length) = opts.max_length {
                pipeline = pipeline.with_max_new_tokens(max_length);
            }

            Ok(Box::new(TensorRTPipelineWrapper::Generation(pipeline)))
        },
        _ => Err(TrustformersError::invalid_input_simple(format!(
            "TensorRT backend not yet implemented for task: {}",
            task
        ))),
    }
}

/// Wrapper to make ONNX pipelines compatible with the unified Pipeline trait
enum OnnxPipelineWrapper<T: crate::core::traits::Tokenizer + Clone> {
    Classification(ONNXTextClassificationPipeline<T>),
    Generation(ONNXTextGenerationPipeline<T>),
}

impl<T: crate::core::traits::Tokenizer + Clone> Pipeline for OnnxPipelineWrapper<T> {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        match self {
            OnnxPipelineWrapper::Classification(pipeline) => pipeline.__call__(input),
            OnnxPipelineWrapper::Generation(pipeline) => pipeline.__call__(input),
        }
    }
}

/// Wrapper to make TensorRT pipelines compatible with the unified Pipeline trait
enum TensorRTPipelineWrapper<T: crate::core::traits::Tokenizer + Clone> {
    Classification(TensorRTTextClassificationPipeline<T>),
    Generation(TensorRTTextGenerationPipeline<T>),
}

impl<T: crate::core::traits::Tokenizer + Clone> Pipeline for TensorRTPipelineWrapper<T> {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        match self {
            TensorRTPipelineWrapper::Classification(pipeline) => pipeline.__call__(input),
            TensorRTPipelineWrapper::Generation(pipeline) => pipeline.__call__(input),
        }
    }
}

/// Wrapper to make DocumentUnderstanding pipeline compatible with the unified Pipeline trait
pub struct DocumentUnderstandingPipelineWrapper<M, T>(DocumentUnderstandingPipeline<M, T>);

impl<M, T> Pipeline for DocumentUnderstandingPipelineWrapper<M, T>
where
    M: crate::core::traits::Model + Clone,
    T: crate::core::traits::Tokenizer + Clone,
{
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        // DocumentUnderstanding requires image data, not text
        // For string input, we'll return an error indicating this pipeline needs proper input
        Err(TrustformersError::invalid_input_simple(
            "DocumentUnderstanding pipeline requires DocumentUnderstandingInput with image data, not string input".to_string()
        ))
    }
}

/// Wrapper to make VisualQuestionAnswering pipeline compatible with the unified Pipeline trait
#[cfg(feature = "vision")]
pub struct VisualQuestionAnsweringPipelineWrapper<M, T>(VisualQuestionAnsweringPipeline<M, T>)
where
    M: crate::core::traits::Model + Clone + Send + Sync + 'static,
    T: crate::core::traits::Tokenizer + Clone + Send + Sync + 'static;

#[cfg(feature = "vision")]
impl<M, T> Pipeline for VisualQuestionAnsweringPipelineWrapper<M, T>
where
    M: crate::core::traits::Model + Clone + Send + Sync + 'static,
    T: crate::core::traits::Tokenizer + Clone + Send + Sync + 'static,
{
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        // VQA requires both image and question, but we only have string input
        // Return an error indicating this pipeline needs proper input
        Err(TrustformersError::invalid_input_simple(
            "VisualQuestionAnswering pipeline requires VisualQuestionAnsweringInput with image and question data, not string input".to_string()
        ))
    }
}

/// Wrapper to make MultiModal pipeline compatible with the unified Pipeline trait
pub struct MultiModalPipelineWrapper<M, T>(MultiModalPipeline<M, T>);

impl<M, T> Pipeline for MultiModalPipelineWrapper<M, T>
where
    M: crate::core::traits::Model + Clone + 'static,
    T: crate::core::traits::Tokenizer + Clone + 'static,
{
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        use std::collections::HashMap;
        let output = self.0.__call__(MultiModalInput {
            text: Some(input),
            image: None,
            audio: None,
            video: None,
            metadata: HashMap::new(),
            modality_weights: None,
        })?;
        Ok(PipelineOutput::MultiModal(output))
    }
}

#[cfg(feature = "async")]
/// Wrapper to make Conversational pipeline compatible with the unified Pipeline trait
pub struct ConversationalPipelineWrapper<M, T>(ConversationalPipeline<M, T>);

#[cfg(feature = "async")]
impl<M, T> Pipeline for ConversationalPipelineWrapper<M, T>
where
    M: crate::core::traits::Model + GenerativeModel + Clone + Send + Sync,
    T: crate::core::traits::Tokenizer + Clone,
{
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        let output = self.0.__call__(ConversationalInput {
            message: input,
            conversation_id: None,
            context: None,
            config_override: None,
        })?;
        Ok(PipelineOutput::Conversational(output))
    }
}

/// Wrapper to make adaptive inference pipelines compatible with the unified Pipeline trait
pub struct AdaptivePipelineWrapper<P> {
    engine: AdaptiveInferenceEngine<P>,
}

impl<P> AdaptivePipelineWrapper<P> {
    pub fn new(engine: AdaptiveInferenceEngine<P>) -> Self {
        Self { engine }
    }
}

impl<P> Pipeline for AdaptivePipelineWrapper<P>
where
    P: Pipeline<Output = PipelineOutput> + Clone,
    P::Input: Clone,
{
    type Input = P::Input;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        // For now, we'll need to make the engine mutable
        // In a real implementation, this would require interior mutability
        let mut engine = self.engine.clone();
        let result = engine.adaptive_inference(input)?;
        Ok(result.prediction)
    }
}

#[cfg(feature = "vision")]
/// Factory function specifically for image-to-text pipelines
pub fn image_to_text_pipeline(
    model: Option<&str>,
    options: Option<PipelineOptions>,
) -> Result<ImageToTextPipeline> {
    let opts = options.unwrap_or_default();
    let model_name = model.or(opts.model.as_deref()).unwrap_or("blip-image-captioning-base");

    let model = AutoModel::from_pretrained(model_name)?;
    let tokenizer = AutoTokenizer::from_pretrained(model_name)?;

    let mut pipeline = ImageToTextPipeline::new(model, tokenizer)?;

    // Apply options
    if let Some(max_length) = opts.max_length {
        pipeline = pipeline.with_max_new_tokens(max_length);
    }

    // Configure device if specified
    match opts.device {
        Some(Device::Cpu) => {
            // Already default
        },
        Some(Device::Gpu(_)) => {
            // Would configure GPU device in real implementation
        },
        None => {
            // Use default (CPU)
        },
    }

    Ok(pipeline)
}

#[cfg(feature = "audio")]
/// Factory function specifically for speech-to-text pipelines
pub fn speech_to_text_pipeline(
    model: Option<&str>,
    options: Option<PipelineOptions>,
) -> Result<SpeechToTextPipeline> {
    let opts = options.unwrap_or_default();
    let model_name = model.or(opts.model.as_deref()).unwrap_or("openai/whisper-base");

    let model = AutoModel::from_pretrained(model_name)?;
    let tokenizer = AutoTokenizer::from_pretrained(model_name)?;

    let mut pipeline = SpeechToTextPipeline::new(model, tokenizer)?;

    // Apply options
    let mut config = SpeechToTextConfig::default();

    if let Some(max_length) = opts.max_length {
        config.max_duration = Some(max_length as f64); // Convert to seconds
    }

    pipeline = pipeline.with_config(config);

    // Configure device if specified
    match opts.device {
        Some(Device::Cpu) => {
            // Already default
        },
        Some(Device::Gpu(_)) => {
            // Would configure GPU device in real implementation
        },
        None => {
            // Use default (CPU)
        },
    }

    Ok(pipeline)
}

#[cfg(feature = "audio")]
/// Factory function specifically for text-to-speech pipelines
pub fn text_to_speech_pipeline(
    model: Option<&str>,
    options: Option<PipelineOptions>,
) -> Result<TextToSpeechPipeline<crate::AutoModel, crate::AutoTokenizer>> {
    let opts = options.unwrap_or_default();
    let model_name = model.or(opts.model.as_deref()).unwrap_or("microsoft/speecht5_tts");

    let model = crate::AutoModel::from_pretrained(model_name)?;
    let tokenizer = crate::AutoTokenizer::from_pretrained(model_name)?;

    let mut pipeline = TextToSpeechPipeline::new(model, tokenizer)?;

    // Apply configuration from options
    let mut config = TextToSpeechConfig::default();

    if let Some(max_length) = opts.max_length {
        config.max_duration = Some(max_length as f64); // Convert to seconds
    }

    pipeline = pipeline.with_config(config);

    // Configure device if specified
    match opts.device {
        Some(Device::Cpu) => {
            // Already default
        },
        Some(Device::Gpu(_)) => {
            // Would configure GPU device in real implementation
        },
        None => {
            // Use default (CPU)
        },
    }

    Ok(pipeline)
}

#[cfg(feature = "vision")]
/// Factory function specifically for visual question answering pipelines
pub fn visual_question_answering_pipeline(
    model: Option<&str>,
    options: Option<PipelineOptions>,
) -> Result<VisualQuestionAnsweringPipeline<crate::AutoModel, crate::AutoTokenizer>> {
    let opts = options.unwrap_or_default();
    let model_name = model.or(opts.model.as_deref()).unwrap_or("dandelin/vilt-b32-finetuned-vqa");

    let model = crate::AutoModel::from_pretrained(model_name)?;
    let tokenizer = crate::AutoTokenizer::from_pretrained(model_name)?;

    let mut pipeline = VisualQuestionAnsweringPipeline::new(model, tokenizer)?;

    // Apply configuration from options
    let mut config = VisualQuestionAnsweringConfig::default();

    if let Some(max_length) = opts.max_length {
        config.max_question_length = max_length;
    }

    if let Some(batch_size) = opts.batch_size {
        config.top_k_answers = batch_size;
    }

    pipeline = pipeline.with_config(config);

    // Configure device if specified
    match opts.device {
        Some(Device::Cpu) => {
            // Already default
        },
        Some(Device::Gpu(_)) => {
            // Would configure GPU device in real implementation
        },
        None => {
            // Use default (CPU)
        },
    }

    Ok(pipeline)
}

/// Factory function specifically for document understanding pipelines
pub fn document_understanding_pipeline(
    model: Option<&str>,
    options: Option<PipelineOptions>,
) -> Result<DocumentUnderstandingPipeline<crate::AutoModel, crate::AutoTokenizer>> {
    let opts = options.unwrap_or_default();
    let model_name = model.or(opts.model.as_deref()).unwrap_or("microsoft/layoutlmv3-base");

    let model = crate::AutoModel::from_pretrained(model_name)?;
    let tokenizer = crate::AutoTokenizer::from_pretrained(model_name)?;

    let mut pipeline = DocumentUnderstandingPipeline::new(model, tokenizer)?;

    // Apply configuration from options
    let mut config = DocumentUnderstandingConfig::default();

    if let Some(max_length) = opts.max_length {
        config.max_length = max_length;
    }

    if let Some(batch_size) = opts.batch_size {
        config.confidence_threshold = (batch_size as f32) / 100.0; // Use batch_size as confidence threshold
    }

    pipeline = pipeline.with_config(config);

    // Configure device if specified
    match opts.device {
        Some(Device::Cpu) => {
            // Already default
        },
        Some(Device::Gpu(_)) => {
            // Would configure GPU device in real implementation
        },
        None => {
            // Use default (CPU)
        },
    }

    Ok(pipeline)
}

/// Common output format for pipelines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineOutput {
    /// Classification output with labels and scores
    Classification(Vec<ClassificationOutput>),
    /// Text generation output
    Generation(GenerationOutput),
    /// Token classification output (NER)
    TokenClassification(Vec<TokenClassificationOutput>),
    /// Question answering output
    QuestionAnswering(QuestionAnsweringOutput),
    /// Fill mask output
    FillMask(Vec<FillMaskOutput>),
    /// Summarization output
    Summarization(String),
    /// Translation output
    Translation(String),
    /// Image-to-text output
    #[cfg(feature = "vision")]
    ImageToText(ImageToTextOutput),
    /// Speech-to-text output
    #[cfg(feature = "audio")]
    SpeechToText(SpeechToTextOutput),
    /// Text-to-speech output
    #[cfg(feature = "audio")]
    TextToSpeech(TextToSpeechOutput),
    /// Visual question answering output
    #[cfg(feature = "vision")]
    VisualQuestionAnswering(VisualQuestionAnsweringOutput),
    /// Document understanding output
    DocumentUnderstanding(DocumentUnderstandingOutput),
    /// Multi-modal output
    MultiModal(MultiModalOutput),
    /// Conversational output
    #[cfg(feature = "async")]
    Conversational(ConversationalOutput),
    /// Advanced RAG output
    AdvancedRAG(advanced_rag::AdvancedRAGOutput),
    /// Mixture of Depths output
    MixtureOfDepths(mixture_of_depths::MoDExecutionResult),
    /// Speculative Decoding output
    SpeculativeDecoding(speculative_decoding::SpeculativeDecodingResult),
    /// Mamba-2 State Space Model output
    Mamba2(mamba2_pipeline::Mamba2Output),
    /// Simple text output
    Text(String),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClassificationOutput {
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationOutput {
    pub generated_text: String,
    pub sequences: Option<Vec<Vec<u32>>>,
    pub scores: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenClassificationOutput {
    pub entity: String,
    pub score: f32,
    pub index: usize,
    pub word: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionAnsweringOutput {
    pub answer: String,
    pub score: f32,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillMaskOutput {
    pub sequence: String,
    pub score: f32,
    pub token: u32,
    pub token_str: String,
}

/// Base struct for pipelines that need a model and tokenizer
#[derive(Clone)]
pub struct BasePipeline<M, T> {
    pub model: Arc<M>,
    pub tokenizer: Arc<T>,
    pub device: Device,
    pub batch_size: usize,
    pub max_length: usize,
    pub truncation: bool,
    pub padding: PaddingStrategy,
    pub cache: Option<Arc<InferenceCache>>,
    pub advanced_cache: Option<Arc<AdvancedLRUCache<String>>>,
    pub cache_key_builder: PipelineCacheKeyBuilder,
}

impl<M, T> BasePipeline<M, T> {
    pub fn new(model: M, tokenizer: T) -> Self {
        Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            device: Device::Cpu,
            batch_size: 1,
            max_length: 512,
            truncation: true,
            padding: PaddingStrategy::Longest,
            cache: None,
            advanced_cache: None,
            cache_key_builder: PipelineCacheKeyBuilder::new(),
        }
    }

    pub fn to_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    pub fn with_padding(mut self, padding: PaddingStrategy) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_cache(mut self, cache_config: CacheConfig) -> Self {
        self.cache = Some(Arc::new(InferenceCache::new(cache_config)));
        self
    }

    pub fn with_existing_cache(mut self, cache: Arc<InferenceCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    pub fn get_cache(&self) -> Option<Arc<InferenceCache>> {
        self.cache.clone()
    }

    /// Create a dynamic batching configuration based on pipeline settings
    pub fn create_dynamic_config(&self) -> DynamicBatchingConfig {
        DynamicBatchingConfig {
            initial_batch_size: self.batch_size,
            max_batch_size: std::cmp::max(self.batch_size * 4, 32),
            min_batch_size: 1,
            target_latency_ms: match self.device {
                Device::Gpu(_) => 50, // Lower latency target for GPU
                Device::Cpu => 100,   // Higher latency target for CPU
            },
            max_wait_time_ms: 25,
            throughput_threshold: 10.0,
            performance_window_size: 10,
            adjustment_factor: 1.2,
        }
    }

    /// Get optimal batch size based on input characteristics
    pub fn get_optimal_batch_size(&self, input_count: usize, estimated_memory_mb: f64) -> usize {
        let base_size = self.batch_size;

        // Adjust based on available memory (simplified calculation)
        let memory_factor = if estimated_memory_mb > 1000.0 {
            0.8 // Reduce batch size for large memory usage
        } else if estimated_memory_mb < 100.0 {
            1.5 // Increase batch size for small memory usage
        } else {
            1.0
        };

        // Adjust based on device capabilities
        let device_factor = match self.device {
            Device::Gpu(_) => 2.0, // GPU can handle larger batches
            Device::Cpu => 1.0,
        };

        let calculated_size = (base_size as f64 * memory_factor * device_factor) as usize;

        // Ensure we don't exceed input count
        std::cmp::min(calculated_size, input_count)
    }

    /// Create an adaptive batch configuration based on pipeline settings
    pub fn create_adaptive_config(&self) -> AdaptiveBatchConfig {
        AdaptiveBatchConfig {
            min_batch_size: 1,
            max_batch_size: std::cmp::max(self.batch_size * 8, 64),
            samples_per_size: 10,
            warmup_iterations: 3,
            target_latency_percentile: 95.0,
            target_latency_ms: match self.device {
                Device::Gpu(_) => 50.0, // Lower latency target for GPU
                Device::Cpu => 100.0,   // Higher latency target for CPU
            },
            throughput_weight: 0.4,
            latency_weight: 0.4,
            memory_weight: 0.2,
            reevaluation_interval_secs: 300,
        }
    }

    /// Create an adaptive batch optimizer for this pipeline
    pub fn create_adaptive_optimizer(&self) -> AdaptiveBatchOptimizer {
        let config = self.create_adaptive_config();
        AdaptiveBatchOptimizer::new(config)
    }

    /// Helper method to create a performance sample
    pub fn create_performance_sample(
        &self,
        batch_size: usize,
        latency_ms: f64,
        throughput_rps: f64,
        memory_usage_mb: f64,
    ) -> PerformanceSample {
        PerformanceSample {
            batch_size,
            latency_ms,
            throughput_rps,
            memory_usage_mb,
            gpu_memory_mb: memory_usage_mb * 0.8, // Estimate GPU memory
            cpu_utilization: 0.7,                 // Placeholder
            gpu_utilization: match self.device {
                Device::Gpu(_) => 0.8,
                Device::Cpu => 0.0,
            },
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Enable advanced caching with configuration
    pub fn with_advanced_cache(mut self, config: AdvancedCacheConfig) -> Self {
        self.advanced_cache = Some(Arc::new(AdvancedLRUCache::new(config)));
        self
    }

    /// Enable advanced caching with existing cache
    pub fn with_existing_advanced_cache(mut self, cache: Arc<AdvancedLRUCache<String>>) -> Self {
        self.advanced_cache = Some(cache);
        self
    }

    /// Create default advanced cache configuration
    pub fn create_advanced_cache_config(&self) -> AdvancedCacheConfig {
        AdvancedCacheConfig {
            max_entries: match self.device {
                Device::Gpu(_) => 50000, // GPU can handle more entries
                Device::Cpu => 10000,    // CPU more conservative
            },
            max_memory_bytes: match self.device {
                Device::Gpu(_) => 2 * 1024 * 1024 * 1024, // 2GB for GPU
                Device::Cpu => 1024 * 1024 * 1024,        // 1GB for CPU
            },
            ttl_seconds: 3600,             // 1 hour
            cleanup_interval_seconds: 300, // 5 minutes
            lru_eviction_threshold: 0.8,
            smart_eviction_threshold: 0.9,
            enable_hit_rate_tracking: true,
            enable_memory_pressure_monitoring: true,
            enable_access_pattern_analysis: true,
        }
    }

    /// Get cache from advanced cache
    pub fn cache_get(&self, input: &str, model_id: &str, config_hash: u64) -> Option<String> {
        if let Some(cache) = &self.advanced_cache {
            let key = self.cache_key_builder.build_key(&input, model_id, config_hash);
            cache.get(&key)
        } else {
            None
        }
    }

    /// Put value in advanced cache
    pub fn cache_put(
        &self,
        input: &str,
        model_id: &str,
        config_hash: u64,
        output: String,
        memory_size: u64,
        priority: CachePriority,
        tags: std::collections::HashSet<String>,
        ttl: Option<std::time::Duration>,
    ) -> Result<()> {
        if let Some(cache) = &self.advanced_cache {
            let key = self.cache_key_builder.build_key(&input, model_id, config_hash);
            cache.insert(key, output, memory_size, priority, tags, ttl)?;
        }
        Ok(())
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> Option<CacheStats> {
        self.advanced_cache.as_ref().map(|cache| cache.get_stats())
    }

    /// Clear cache entries by tag
    pub fn clear_cache_by_tag(&self, tag: &str) -> usize {
        if let Some(cache) = &self.advanced_cache {
            cache.remove_by_tag(tag)
        } else {
            0
        }
    }

    /// Get cache size information
    pub fn get_cache_size_info(&self) -> Option<(usize, u64)> {
        self.advanced_cache.as_ref().map(|cache| cache.size_info())
    }
}
