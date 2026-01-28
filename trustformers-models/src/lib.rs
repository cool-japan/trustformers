//! # TrustformeRS Models
//!
//! This crate provides pre-trained transformer model implementations optimized for Rust,
//! offering high-performance alternatives to Python-based implementations.

// Temporary clippy allows for alpha release - to be addressed in 0.1.0
#![allow(clippy::needless_range_loop)]
#![allow(clippy::useless_vec)]
#![allow(clippy::redundant_locals)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::await_holding_lock)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::wrong_self_convention)]
#![allow(clippy::same_item_push)]
#![allow(clippy::vec_init_then_push)]
#![allow(clippy::ptr_arg)]
//!
//! ## Overview
//!
//! TrustformeRS Models includes implementations of popular transformer architectures:
//!
//! - **BERT Family**: BERT, RoBERTa, DistilBERT, ALBERT, ELECTRA, DeBERTa
//! - **GPT Family**: GPT-2, GPT-Neo, GPT-J
//! - **Modern LLMs**: LLaMA, Mistral, Gemma, Qwen, Falcon, StableLM, Command R, Claude
//! - **Encoder-Decoder**: T5
//! - **Vision Models**: Vision Transformer (ViT), CLIP
//! - **Multimodal Models**: BLIP-2, LLaVA, DALL-E, Flamingo
//! - **Efficient Models**: Mamba, RWKV, S4
//!
//! ## Features
//!
//! Each model implementation includes:
//! - Pre-trained weight loading from Hugging Face Hub
//! - Task-specific heads (classification, generation, etc.)
//! - Efficient inference with SciRS2 backend
//! - Support for quantization and optimization
//! - Performance optimization utilities for production deployment
//! - Model serving infrastructure with load balancing and health monitoring
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_models::{BertModel, BertConfig};
//! use trustformers_core::Result;
//!
//! fn main() -> Result<()> {
//!     // Load a pre-trained BERT model
//!     let config = BertConfig::bert_base_uncased();
//!     let mut model = BertModel::new(config)?;
//!
//!     // Load weights from Hugging Face
//!     model.load_from_hub("bert-base-uncased")?;
//!
//!     // Use the model for inference
//!     // ... tokenization and forward pass
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! Models are gated behind feature flags to reduce compilation time:
//!
//! ```toml
//! [dependencies]
//! trustformers-models = { version = "*", features = ["bert", "gpt2"] }
//! ```
//!
//! Available features:
//! - `bert`: BERT model family
//! - `roberta`: RoBERTa models
//! - `distilbert`: DistilBERT models
//! - `gpt2`: GPT-2 models
//! - `gpt_neo`: GPT-Neo models
//! - `gpt_j`: GPT-J models
//! - `t5`: T5 encoder-decoder models
//! - `albert`: ALBERT models
//! - `electra`: ELECTRA models
//! - `deberta`: DeBERTa models
//! - `vit`: Vision Transformer models
//! - `llama`: LLaMA models
//! - `mistral`: Mistral models
//! - `clip`: CLIP multimodal models
//! - `llava`: LLaVA visual instruction tuning models
//! - `dalle`: DALL-E text-to-image generation models
//! - `flamingo`: Flamingo few-shot vision-language models
//! - `gemma`: Gemma models
//! - `qwen`: Qwen models
//! - `phi3`: Phi-3 small language models
//! - `mamba`: Mamba state-space models
//! - `rwkv`: RWKV linear complexity models
//! - `falcon`: Falcon high-performance language models
//! - `claude`: Claude constitutional AI models
//!
//! ## Model Categories
//!
//! ### Encoder Models (BERT Family)
//!
//! These models use bidirectional attention and are best for:
//! - Text classification
//! - Named entity recognition
//! - Question answering
//! - Feature extraction
//!
//! ### Decoder Models (GPT Family)
//!
//! These models use causal (left-to-right) attention and excel at:
//! - Text generation
//! - Code completion
//! - Creative writing
//! - Conversational AI
//!
//! ### Encoder-Decoder Models (T5)
//!
//! These models combine both architectures and are ideal for:
//! - Translation
//! - Summarization
//! - Question answering
//! - Text-to-text generation
//!
//! ### Vision Models
//!
//! - **ViT**: Image classification and feature extraction
//! - **CLIP**: Multimodal understanding (text + image)

// Allow large error types in Result (TrustformersError is large by design)
#![allow(clippy::result_large_err)]
// Allow nested blocks (complex model implementations require deep nesting)
#![allow(clippy::excessive_nesting)]
// Allow common patterns in complex model code
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::large_enum_variant)]

#[cfg(feature = "bert")]
pub mod bert;

#[cfg(feature = "roberta")]
pub mod roberta;

#[cfg(feature = "distilbert")]
pub mod distilbert;

#[cfg(feature = "gpt2")]
pub mod gpt2;

#[cfg(feature = "gpt_neo")]
pub mod gpt_neo;

#[cfg(feature = "gpt_j")]
pub mod gpt_j;

#[cfg(feature = "t5")]
pub mod t5;

#[cfg(feature = "albert")]
pub mod albert;

#[cfg(feature = "electra")]
pub mod electra;

#[cfg(feature = "deberta")]
pub mod deberta;

#[cfg(feature = "vit")]
pub mod vit;

#[cfg(feature = "llama")]
pub mod llama;

#[cfg(feature = "gpt_neox")]
pub mod gpt_neox;

#[cfg(feature = "mistral")]
pub mod mistral;

#[cfg(feature = "clip")]
pub mod clip;
pub mod cogvlm;
pub mod recursive;

// Multimodal models
#[cfg(feature = "blip2")]
pub mod blip2;

#[cfg(feature = "llava")]
pub mod llava;

#[cfg(feature = "dalle")]
pub mod dalle;

#[cfg(feature = "flamingo")]
pub mod flamingo;

#[cfg(feature = "gemma")]
pub mod gemma;

#[cfg(feature = "qwen")]
pub mod qwen;

#[cfg(feature = "phi3")]
pub mod phi3;

// State-space and efficient models
pub mod hyena;
pub mod mamba;
pub mod retnet;
pub mod rwkv;
pub mod s4;

// Modern LLM models
pub mod falcon;
pub mod stablelm;

// Cohere Models
pub mod command_r;

// Anthropic Models
pub mod claude;

// Mixture of Experts infrastructure
pub mod moe;

// Efficient attention architectures
pub mod fnet;
pub mod linformer;
pub mod performer;

// Sparse attention patterns library
pub mod sparse_attention;

// Cross-attention variants
pub mod cross_attention;

// Hierarchical transformers
pub mod hierarchical;

// Weight loading infrastructure
pub mod advanced_quantization;
pub mod ring_attention;
pub mod weight_loading;

// Generation utilities for text generation
pub mod generation_utils;

// Batch inference optimization utilities
pub mod batch_inference;

// Dynamic token pruning for efficiency
pub mod dynamic_pruning;

// Knowledge distillation framework
pub mod knowledge_distillation;

// Model compression toolkit
pub mod model_compression;

// Continual learning framework
pub mod continual_learning;

// Curriculum learning framework
pub mod curriculum_learning;

// Multi-task learning framework
pub mod multi_task_learning;

// Progressive training framework
pub mod progressive_training;

// Meta-learning framework
pub mod meta_learning;

// Code-specialized models
#[cfg(feature = "llama")]
pub mod code_specialized;

// Mathematics-specialized models
#[cfg(feature = "llama")]
pub mod math_specialized;

// Scientific domain-specialized models
pub mod scientific_specialized;

// Legal and medical domain-specialized models
pub mod legal_medical_specialized;

// Creative writing domain-specialized models
pub mod creative_writing_specialized;

// Common model patterns and traits
pub mod common_patterns;

// Comprehensive testing and validation framework
pub mod comprehensive_testing;

// Model documentation and ethics
pub mod model_cards;

// Neural Architecture Search framework
pub mod neural_architecture_search;

// Automated Model Design framework
pub mod automated_model_design;

// Hybrid Architectures framework
pub mod hybrid_architectures;

// Memory profiling and performance analysis
pub mod memory_profiling;

// Comprehensive error recovery framework
pub mod error_recovery;

// Mixed-bit quantization framework
pub mod mixed_bit_quantization;

// Performance optimization utilities
pub mod performance_optimization;

// Model serving utilities
pub mod model_serving;

// Extended LSTM (xLSTM) models with exponential gating and matrix memory
pub mod xlstm;

// Biologically-inspired models (temporarily disabled due to compilation errors)
pub mod biologically_inspired;

// Quantum-classical hybrid models (temporarily disabled due to compiler crash)
pub mod quantum_classical_hybrids;

// Benchmarking and performance analysis tools
pub mod benchmarking;

// Numerical parity testing with reference implementations
pub mod numerical_parity_tests;

// Developer tools and utilities
pub mod developer_tools;

#[cfg(feature = "bert")]
pub use bert::{BertConfig, BertForMaskedLM, BertForSequenceClassification, BertModel};

#[cfg(feature = "roberta")]
pub use roberta::{
    RobertaConfig, RobertaForMaskedLM, RobertaForQuestionAnswering,
    RobertaForSequenceClassification, RobertaForTokenClassification, RobertaModel,
};

#[cfg(feature = "distilbert")]
pub use distilbert::{
    DistilBertConfig, DistilBertForMaskedLM, DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification, DistilBertForTokenClassification, DistilBertModel,
};

#[cfg(feature = "gpt2")]
pub use gpt2::{Gpt2Config, Gpt2LMHeadModel, Gpt2Model};

#[cfg(feature = "gpt_neo")]
pub use gpt_neo::{GptNeoConfig, GptNeoLMHeadModel, GptNeoModel};

#[cfg(feature = "gpt_j")]
pub use gpt_j::{GptJConfig, GptJLMHeadModel, GptJModel};

#[cfg(feature = "t5")]
pub use t5::{T5Config, T5ForConditionalGeneration, T5Model};

#[cfg(feature = "albert")]
pub use albert::{
    AlbertConfig, AlbertForMaskedLM, AlbertForQuestionAnswering, AlbertForSequenceClassification,
    AlbertForTokenClassification, AlbertModel,
};

#[cfg(feature = "electra")]
pub use electra::{
    ElectraConfig, ElectraForMultipleChoice, ElectraForPreTraining, ElectraForQuestionAnswering,
    ElectraForSequenceClassification, ElectraForTokenClassification, ElectraModel,
};

#[cfg(feature = "deberta")]
pub use deberta::{
    DebertaConfig, DebertaForMaskedLM, DebertaForMultipleChoice, DebertaForQuestionAnswering,
    DebertaForSequenceClassification, DebertaForTokenClassification, DebertaModel,
};

#[cfg(feature = "vit")]
pub use vit::{ViTConfig, ViTForImageClassification, ViTModel};

#[cfg(feature = "llama")]
pub use llama::{LlamaConfig, LlamaForCausalLM, LlamaModel};

#[cfg(feature = "gpt_neox")]
pub use gpt_neox::{GPTNeoXConfig, GPTNeoXForCausalLM, GPTNeoXModel};

#[cfg(feature = "mistral")]
pub use mistral::{MistralConfig, MistralForCausalLM, MistralModel};

#[cfg(feature = "clip")]
pub use clip::{CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig};

#[cfg(feature = "blip2")]
pub use blip2::{
    Blip2ConditionalGenerationOutput, Blip2Config, Blip2ForConditionalGeneration, Blip2Model,
    Blip2Output, Blip2QFormerConfig, Blip2QFormerModel, Blip2QFormerOutput, Blip2TextConfig,
    Blip2VisionConfig, Blip2VisionModel, LanguageModelOutput,
};

#[cfg(feature = "llava")]
pub use llava::{LlavaConfig, LlavaForConditionalGeneration, LlavaVisionConfig};

#[cfg(feature = "dalle")]
pub use dalle::{
    DalleConfig, DalleDiffusionConfig, DalleImageConfig, DalleImageEncoder, DalleMLP, DalleModel,
    DalleModelOutput, DalleTextConfig, DalleTextEncoder, DalleTimeEmbedding, DalleUNet, DalleVAE,
    DalleVisionConfig,
};

#[cfg(feature = "flamingo")]
pub use flamingo::{
    FlamingoConfig, FlamingoLanguageConfig, FlamingoLanguageModel, FlamingoLanguageOutput,
    FlamingoModel, FlamingoOutput, FlamingoPerceiverConfig, FlamingoVisionConfig,
    FlamingoVisionEncoder, FlamingoXAttentionConfig, PerceiverResampler,
};

#[cfg(feature = "gemma")]
pub use gemma::{GemmaConfig, GemmaForCausalLM, GemmaModel};

#[cfg(feature = "qwen")]
pub use qwen::{QwenConfig, QwenForCausalLM, QwenModel};

#[cfg(feature = "phi3")]
pub use phi3::{Phi3Config, Phi3ForCausalLM, Phi3Model};

pub use automated_model_design::{
    ArchitectureTemplate, ConstraintSolver, DeploymentEnvironment, DesignPatternLibrary,
    DesignRequirements, DesignRequirementsBuilder, Modality, ModelDesign, ModelDesignMetadata,
    ModelDesigner, ModelMetrics, PerformanceTarget, ResourceConstraints,
    TaskType as DesignTaskType, TemplateMetadata,
};
pub use claude::{ClaudeConfig, ClaudeForCausalLM, ClaudeModel};
#[cfg(feature = "llama")]
pub use code_specialized::{
    CodeLlamaConfig, CodeLlamaForCausalLM, CodeLlamaModel, CodeModelVariant, CodeSpecialTokens,
    CodeSpecializedConfig, CodeSpecializedForCausalLM, CodeSpecializedModel, DeepSeekCoderConfig,
    DeepSeekCoderForCausalLM, DeepSeekCoderModel, QwenCoderConfig, QwenCoderForCausalLM,
    QwenCoderModel, StarCoderConfig, StarCoderForCausalLM, StarCoderModel,
};
pub use command_r::{CommandRConfig, CommandRForCausalLM, CommandRModel};
pub use common_patterns::{
    components, get_global_registry, ArchitectureType, ComputeRequirements, EvaluableModel,
    EvaluationData, EvaluationMetric, EvaluationResults, GenerationConfig, GenerationStrategy,
    GenerativeModel, InitializationStrategy, MemoryEstimate, ModelFamily, ModelFamilyMetadata,
    ModelRegistry, ModelUtils, TaskType as CommonTaskType,
};
pub use comprehensive_testing::{
    reporting, BiasMetric, BiasmitigationStrategy, FairnessAssessment, FairnessConfig,
    FairnessMetricType, FairnessResult, FairnessTestData, FairnessViolation, GroupData,
    LayerPerformance, MemoryAnalysis, ModelTestSuite, NumericalDifferences, NumericalParityResults,
    OverallPerformance, PerformanceProfiler, PerformanceResults, ReferenceComparator,
    StatisticalTest, TestDataType, TestInputConfig, TestResult, TestStatistics,
    ThroughputMeasurements, TimingInfo, ValidationConfig,
};
pub use continual_learning::{
    utils as continual_learning_utils, ContinualLearningConfig, ContinualLearningMetrics,
    ContinualLearningOutput, ContinualLearningTrainer, ContinualStrategy, LearningRateSchedule,
    MemoryBuffer, MemorySelectionStrategy, TaskEvaluation, TaskInfo,
};
pub use creative_writing_specialized::{
    CreativeWritingConfig, CreativeWritingForCausalLM, CreativeWritingModel,
    CreativeWritingSpecialTokens, EmotionalTone, ImprovementType, LiteraryDevice,
    NarrativePerspective, PoetryStyle, StyleAnalysis, WritingGenre, WritingImprovement,
    WritingStyle,
};
pub use cross_attention::{
    AdaptiveCrossAttention, CrossAttention, CrossAttentionConfig, GatedCrossAttention,
    HierarchicalCrossAttention, MultiHeadCrossAttention, SparseCrossAttention,
};
pub use curriculum_learning::{
    utils as curriculum_learning_utils, CurriculumAnalysis, CurriculumConfig,
    CurriculumEpochOutput, CurriculumExample, CurriculumLearningOutput, CurriculumLearningTrainer,
    CurriculumStats, CurriculumStrategy, DifficultyMeasure, PacingFunction,
};
pub use dynamic_pruning::*;
pub use error_recovery::{
    ErrorCategory, ErrorRecoveryManager, ErrorTrends, ModelCheckpoint, RecoverableOperation,
    RecoveryAttempt, RecoveryConfig, RecoveryMetrics, RecoveryReport, RecoveryStrategy,
};
pub use falcon::{FalconConfig, FalconForCausalLM, FalconModel};
pub use fnet::{FNetConfig, FNetForMaskedLM, FNetForSequenceClassification, FNetModel};
pub use hierarchical::{
    HierarchicalConfig, HierarchicalForLanguageModeling, HierarchicalForSequenceClassification,
    HierarchicalTransformer, NestedTransformer, PyramidTransformer, TreeTransformer,
};
pub use hybrid_architectures::{
    AdaptiveConfig, ArchitecturalComponent, ArchitectureSummary, AttentionType, CNNArchitecture,
    CrossModalConfig, EnsembleMethod, FusionStrategy, GlobalParams, HierarchyType,
    HybridArchitecture, HybridConfig, HybridConfigBuilder, MemoryType, ParallelFusionMethod,
    RNNCellType, StateSpaceType, SwitchingCriteria, TransformerVariant,
};
pub use hyena::{
    HyenaConfig, HyenaForLanguageModeling, HyenaForSequenceClassification, HyenaModel,
};
pub use knowledge_distillation::{
    utils as knowledge_distillation_utils, DistillationConfig, DistillationOutput,
    DistillationStrategy, KnowledgeDistillationTrainer, ProgressiveStage, StudentOutputs,
    TeacherOutputs,
};
pub use legal_medical_specialized::{
    Citation, CitationType, ComplianceReport, ComplianceViolation, DocumentAnalysis,
    LegalMedicalConfig, LegalMedicalDomain, LegalMedicalForCausalLM, LegalMedicalModel,
    LegalMedicalSpecialTokens, LegalSystem, MedicalStandard, PrivacyRequirement,
};
pub use linformer::{
    LinformerConfig, LinformerForMaskedLM, LinformerForSequenceClassification, LinformerModel,
};
pub use mamba::{MambaConfig, MambaModel};
#[cfg(feature = "llama")]
pub use math_specialized::{
    ChainOfThoughtConfig, DeepSeekMathConfig, DeepSeekMathForCausalLM, DeepSeekMathModel,
    MammothConfig, MammothForCausalLM, MammothModel, MathDomain, MathLlamaConfig,
    MathLlamaForCausalLM, MathLlamaModel, MathModelVariant, MathProblemType, MathReasoningOutput,
    MathSpecialTokens, MathSpecializedConfig, MathSpecializedForCausalLM, MathSpecializedModel,
    MinervaConfig, MinervaForCausalLM, MinervaModel, ReasoningStep, ReasoningStrategy,
};
pub use meta_learning::{
    utils as meta_learning_utils, ConvergenceMetrics, EpisodeResult, EvaluationResult, Example,
    ExampleSet, MetaAlgorithm, MetaLearner, MetaLearningConfig, MetaLearningModel, MetaOptimizer,
    MetaStatistics, PerformanceMetrics, Task, TaskBatch, TaskResult, TaskSampler,
    TaskType as MetaTaskType,
};
pub use mixed_bit_quantization::{
    BitAllocationStrategy, CalibrationConfig, CalibrationMethod,
    HardwareConstraints as QuantizationHardwareConstraints,
    HardwarePlatform as QuantizationHardwarePlatform, LayerQuantizationConstraints,
    MixedBitQuantizationConfig, MixedBitQuantizer, ProgressiveQuantizationConfig,
    QuantizationFormat, QuantizationParams, QuantizationQualityMetrics, QuantizationResults,
    QuantizedLayerInfo, SensitivityAnalysisResults,
};
pub use model_compression::{
    utils as model_compression_utils, ClusteringMethod, CompressedModel, CompressionAnalysis,
    CompressionConfig, CompressionPipeline, CompressionStrategy, CompressionSummary,
    DecompositionType, LayerCompressionStats, OptimizationObjective, PruningStrategy,
    StructuredPruningGranularity,
};
pub use model_serving::{
    InferenceRequest, InferenceResponse, LoadBalancer, LoadBalancingStrategy, ModelInstance,
    ModelServingManager, RequestPriority, RequestQueue, ServingConfig, ServingMetrics,
};
pub use moe::{
    glam_config, switch_config, Expert, ExpertParallel, MLPExpert, MoEConfig, RouterOutput,
    RoutingStats, SparseMoE, SwitchMoE, TopKRouter,
};
pub use multi_task_learning::{
    utils as multi_task_learning_utils, LossBalancingStrategy, MTLAnalysis, MTLArchitecture,
    MTLConfig, MTLStats, MultiTaskEvaluation, MultiTaskLearningTrainer, MultiTaskOutput,
    TaskConfig, TaskEvaluation as MTLTaskEvaluation, TaskPriority, TaskType as MTLTaskType,
};
pub use neural_architecture_search::{
    Architecture, ArchitectureConstraint, ArchitectureEvaluation, ArchitectureMetadata,
    DimensionRange, HardwareConstraints, HardwarePlatform, NASConfig, NeuralArchitectureSearcher,
    OptimizationObjective as NASOptimizationObjective, SearchSpace, SearchStatistics,
    SearchStrategy,
};
pub use performance_optimization::{
    BatchProcessor, BatchingStrategy, CachedTensor, DynamicBatchManager, GpuCacheStatistics,
    GpuMemoryChunk, GpuMemoryOptimizer, GpuMemoryPool, GpuMemoryStats,
    GpuOptimizationRecommendations, GpuTensorCache, MemoryOptimizer, PerformanceConfig,
    PerformanceMonitor, PerformanceStatistics,
};
pub use performer::{
    PerformerConfig, PerformerForMaskedLM, PerformerForSequenceClassification, PerformerModel,
};
pub use progressive_training::{
    utils as progressive_training_utils, GrowthDimension, GrowthEvent, GrowthInfo, GrowthResult,
    GrowthSchedule, GrowthStrategy, LearningProgress, ProgressiveConfig, ProgressiveModel,
    ProgressiveTrainer,
};
pub use retnet::{
    RetNetConfig, RetNetForLanguageModeling, RetNetForSequenceClassification, RetNetModel,
};
pub use rwkv::{RwkvConfig, RwkvModel};
pub use s4::{S4Config, S4ForLanguageModeling, S4Model};
pub use scientific_specialized::{
    CitationStyle, ScientificAnalysis, ScientificConfig, ScientificDomain, ScientificForCausalLM,
    ScientificModel, ScientificSpecialTokens,
};
pub use sparse_attention::{
    utils as sparse_attention_utils, SparseAttention, SparseAttentionConfig, SparseAttentionMask,
    SparsePattern,
};
pub use stablelm::{StableLMConfig, StableLMForCausalLM, StableLMModel};
pub use weight_loading::{
    auto_create_loader, create_distributed_loader, create_gguf_loader, create_huggingface_loader,
    create_memory_mapped_loader, DistributedStats, DistributedWeightLoader, GGMLType, GGUFLoader,
    HuggingFaceLoader, LazyTensor, MemoryMappedLoader, QuantizationConfig, StreamingLoader,
    TensorMetadata, WeightDataType, WeightFormat, WeightLoader, WeightLoadingConfig,
};

pub use xlstm::{
    ExponentialGatingConfig, FeedForward, MLstmBlock, MLstmConfig, SLstmBlock, SLstmConfig,
    XLSTMBlockConfig, XLSTMBlockType, XLSTMConfig, XLSTMForCausalLM,
    XLSTMForSequenceClassification, XLSTMLayer, XLSTMModel, XLSTMState,
};

// pub use biologically_inspired::{
//     BiologicalConfig, BiologicalArchitecture, NeuronModel, PlasticityType, MemoryType as BiologicalMemoryType,
//     BiologicalModel, BiologicalModelOutput, BiologicalModelForCausalLM, BiologicalModelForSequenceClassification,
//     BiologicalMemoryStats, SpikingNeuralNetwork, HopfieldNetwork, LiquidTimeConstantNetwork,
//     NeuralTuringMachine, ReservoirComputing, CapsuleNetwork, DendriticComputation, BiologicalMemory,
// }; // Temporarily disabled due to compilation errors
// pub use quantum_classical_hybrids::{
//     QuantumClassicalConfig, QuantumHybridArchitecture, QuantumAnsatzConfig, QuantumMeasurementStrategy,
//     QuantumErrorMitigation, HybridTrainingStrategy, QuantumConnectivity,
//     QuantumClassicalModel, QuantumClassicalModelOutput, QuantumClassicalModelForCausalLM,
//     QuantumClassicalModelForSequenceClassification, QuantumClassicalModelStats,
//     QuantumTransformer, QuantumGraphNeuralNetwork, QuantumConvolutionalNN, QuantumRecurrentNN,
//     QuantumAttentionLayer, QuantumEmbeddingModel, QuantumOptimizer, QuantumTrainingManager,
//     QuantumTrainingMetrics, QuantumTrainingStats,
// }; // Temporarily disabled due to compiler crash

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
