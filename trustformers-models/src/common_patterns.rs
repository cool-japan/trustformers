//! # Common Model Architecture Patterns and Traits
//!
//! This module provides common patterns, traits, and abstractions that are shared
//! across different model implementations in the TrustformeRS Models crate.
//!
//! ## Features
//!
//! - **Common Model Traits**: Standardized interfaces for all models
//! - **Architecture Patterns**: Reusable architectural components
//! - **Weight Initialization**: Standardized weight initialization strategies
//! - **Generation Interfaces**: Unified text generation APIs
//! - **Evaluation Interfaces**: Common evaluation and testing interfaces
//! - **Configuration Patterns**: Shared configuration management
//!
//! ## Common Traits
//!
//! ### ModelFamily
//! Groups related model configurations and provides family-level operations
//!
//! ### GenerativeModel
//! Unified interface for text generation across all model types
//!
//! ### EvaluableModel
//! Common evaluation interface for model testing and validation
//!
//! ## Example Usage
//!
//! ```rust
//! use trustformers_models::common_patterns::{ModelFamily, GenerativeModel, GenerationConfig};
//!
//! // Use any model through common interface
//! fn generate_text<M: GenerativeModel>(model: &M, prompt: &str) -> Result<String> {
//!     let config = GenerationConfig::default();
//!     model.generate(prompt, &config)
//! }
//! ```

use anyhow::Result;
use scirs2_core::random::*; // SciRS2 Integration Policy - includes Rng, Distribution, Normal, Uniform
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use trustformers_core::errors::Result as CoreResult;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Config;

/// Common generation configuration for all models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub max_length: Option<usize>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub length_penalty: f32,
    pub do_sample: bool,
    pub early_stopping: bool,
    pub num_beams: Option<usize>,
    pub num_return_sequences: usize,
    pub pad_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub use_cache: bool,
    pub stream: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            max_length: None,
            temperature: 1.0,
            top_p: 0.9,
            top_k: None,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            do_sample: true,
            early_stopping: false,
            num_beams: None,
            num_return_sequences: 1,
            pad_token_id: None,
            eos_token_id: None,
            use_cache: true,
            stream: false,
        }
    }
}

/// Weight initialization strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InitializationStrategy {
    /// Normal distribution with mean=0, std=initializer_range
    Normal { std: f32 },
    /// Xavier/Glorot uniform initialization
    XavierUniform,
    /// Xavier/Glorot normal initialization
    XavierNormal,
    /// Kaiming/He uniform initialization
    KaimingUniform,
    /// Kaiming/He normal initialization
    KaimingNormal,
    /// Truncated normal initialization
    TruncatedNormal { std: f32, bounds: f32 },
    /// Custom initialization function
    Custom(String),
}

impl Default for InitializationStrategy {
    fn default() -> Self {
        Self::Normal { std: 0.02 }
    }
}

/// A dyn-compatible version of Config trait for runtime usage
pub trait DynConfig {
    /// Validates the configuration for correctness
    fn validate(&self) -> CoreResult<()>;

    /// Returns the architecture name for this configuration
    fn architecture(&self) -> &'static str;

    /// Returns the configuration as Any for downcasting
    fn as_any(&self) -> &dyn Any;
}

/// Blanket implementation for any type that implements Config
impl<T: Config + 'static> DynConfig for T {
    fn validate(&self) -> CoreResult<()> {
        self.validate()
    }

    fn architecture(&self) -> &'static str {
        self.architecture()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Model family trait for grouping related models
pub trait ModelFamily: Send + Sync {
    /// Get the family name (e.g., "LLaMA", "BERT", "GPT")
    fn family_name() -> &'static str
    where
        Self: Sized;

    /// Get available model sizes
    fn available_sizes() -> Vec<&'static str>
    where
        Self: Sized;

    /// Get available variants (base, instruct, chat, etc.)
    fn available_variants() -> Vec<&'static str>
    where
        Self: Sized;

    /// Create configuration for a specific size and variant
    fn create_config(size: &str, variant: Option<&str>) -> Result<Box<dyn DynConfig>>
    where
        Self: Sized;

    /// Get recommended use cases for this family
    fn use_cases() -> Vec<&'static str>
    where
        Self: Sized;

    /// Get model family metadata
    fn metadata() -> ModelFamilyMetadata
    where
        Self: Sized;
}

/// Metadata about a model family
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFamilyMetadata {
    pub family_name: String,
    pub description: String,
    pub paper_reference: Option<String>,
    pub organization: Option<String>,
    pub license: Option<String>,
    pub release_date: Option<String>,
    pub architecture_type: ArchitectureType,
    pub supported_tasks: Vec<TaskType>,
    pub compute_requirements: ComputeRequirements,
}

/// Architecture type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ArchitectureType {
    /// Encoder-only (BERT-style)
    EncoderOnly,
    /// Decoder-only (GPT-style)
    DecoderOnly,
    /// Encoder-decoder (T5-style)
    EncoderDecoder,
    /// State-space models (Mamba-style)
    StateSpace,
    /// Hybrid architectures
    Hybrid,
    /// Multimodal architectures
    Multimodal,
}

/// Supported task types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskType {
    /// Text generation
    TextGeneration,
    /// Text classification
    TextClassification,
    /// Question answering
    QuestionAnswering,
    /// Summarization
    Summarization,
    /// Translation
    Translation,
    /// Code generation
    CodeGeneration,
    /// Image understanding
    ImageUnderstanding,
    /// Multimodal understanding
    MultimodalUnderstanding,
    /// Custom task
    Custom(String),
}

/// Compute requirements information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeRequirements {
    pub minimum_vram_gb: f32,
    pub recommended_vram_gb: f32,
    pub minimum_ram_gb: f32,
    pub cpu_requirements: String,
    pub gpu_requirements: Option<String>,
    pub supports_cpu_inference: bool,
    pub supports_quantization: bool,
}

/// Common generative model interface
pub trait GenerativeModel {
    /// Generate text from a prompt
    fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String>;

    /// Generate multiple completions
    fn generate_batch(&self, prompts: &[&str], config: &GenerationConfig) -> Result<Vec<String>>;

    /// Stream generation token by token
    fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<Box<dyn Iterator<Item = Result<String>>>>;

    /// Get the maximum context length
    fn max_context_length(&self) -> usize;

    /// Get the model configuration
    fn config(&self) -> &dyn DynConfig;

    /// Check if the model supports a specific task
    fn supports_task(&self, task: &TaskType) -> bool;
}

/// Model evaluation interface
pub trait EvaluableModel {
    /// Compute perplexity on a text corpus
    fn compute_perplexity(&self, text: &str) -> Result<f32>;

    /// Compute log likelihood of text
    fn compute_log_likelihood(&self, text: &str) -> Result<f32>;

    /// Get model embeddings for text
    fn get_embeddings(&self, text: &str) -> Result<Tensor>;

    /// Run model-specific evaluation metrics
    fn evaluate(&self, evaluation_data: &EvaluationData) -> Result<EvaluationResults>;
}

/// Evaluation data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationData {
    pub prompts: Vec<String>,
    pub expected_outputs: Option<Vec<String>>,
    pub task_type: TaskType,
    pub metrics: Vec<EvaluationMetric>,
}

/// Evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EvaluationMetric {
    /// Perplexity
    Perplexity,
    /// BLEU score (for translation/generation)
    BLEU,
    /// ROUGE score (for summarization)
    ROUGE,
    /// Exact match accuracy
    ExactMatch,
    /// F1 score
    F1Score,
    /// Semantic similarity
    SemanticSimilarity,
    /// Custom metric
    Custom(String),
}

/// Evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResults {
    pub overall_score: f32,
    pub metric_scores: HashMap<EvaluationMetric, f32>,
    pub per_sample_scores: Option<Vec<f32>>,
    pub metadata: HashMap<String, String>,
}

/// Common model utilities
pub struct ModelUtils;

impl ModelUtils {
    /// Initialize weights using the specified strategy
    pub fn initialize_weights(
        tensor: &mut Tensor,
        strategy: &InitializationStrategy,
    ) -> Result<()> {
        match strategy {
            InitializationStrategy::Normal { std } => {
                // Initialize with normal distribution
                let mut rng = thread_rng();
                let normal = Normal::new(0.0, *std)
                    .map_err(|e| anyhow::anyhow!("Failed to create normal distribution: {}", e))?;

                match tensor {
                    Tensor::F32(data) => {
                        for value in data.iter_mut() {
                            *value = normal.sample(&mut rng);
                        }
                        Ok(())
                    },
                    _ => Err(anyhow::anyhow!(
                        "Normal initialization only supports F32 tensors"
                    )),
                }
            },
            InitializationStrategy::XavierUniform => {
                // Xavier uniform initialization
                let mut rng = thread_rng();
                let shape = tensor.shape();

                match tensor {
                    Tensor::F32(data) => {
                        let (fan_in, fan_out) = if shape.len() >= 2 {
                            (shape[shape.len() - 1], shape[shape.len() - 2])
                        } else {
                            (1, data.len())
                        };

                        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
                        let uniform = Uniform::new(-limit, limit).map_err(|e| {
                            anyhow::anyhow!("Failed to create uniform distribution: {}", e)
                        })?;

                        for value in data.iter_mut() {
                            *value = uniform.sample(&mut rng);
                        }
                        Ok(())
                    },
                    _ => Err(anyhow::anyhow!(
                        "Xavier uniform initialization only supports F32 tensors"
                    )),
                }
            },
            InitializationStrategy::XavierNormal => {
                // Xavier normal initialization
                let mut rng = thread_rng();
                let shape = tensor.shape();

                match tensor {
                    Tensor::F32(data) => {
                        let (fan_in, fan_out) = if shape.len() >= 2 {
                            (shape[shape.len() - 1], shape[shape.len() - 2])
                        } else {
                            (1, data.len())
                        };

                        let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
                        let normal = Normal::new(0.0, std).map_err(|e| {
                            anyhow::anyhow!("Failed to create normal distribution: {}", e)
                        })?;

                        for value in data.iter_mut() {
                            *value = normal.sample(&mut rng);
                        }
                        Ok(())
                    },
                    _ => Err(anyhow::anyhow!(
                        "Xavier normal initialization only supports F32 tensors"
                    )),
                }
            },
            InitializationStrategy::KaimingUniform => {
                // Kaiming uniform initialization
                let mut rng = thread_rng();
                let shape = tensor.shape();

                match tensor {
                    Tensor::F32(data) => {
                        let fan_in = if shape.len() >= 2 { shape[shape.len() - 1] } else { 1 };

                        let limit = (6.0 / fan_in as f32).sqrt();
                        let uniform = Uniform::new(-limit, limit).map_err(|e| {
                            anyhow::anyhow!("Failed to create uniform distribution: {}", e)
                        })?;

                        for value in data.iter_mut() {
                            *value = uniform.sample(&mut rng);
                        }
                        Ok(())
                    },
                    _ => Err(anyhow::anyhow!(
                        "Kaiming uniform initialization only supports F32 tensors"
                    )),
                }
            },
            InitializationStrategy::KaimingNormal => {
                // Kaiming normal initialization
                let mut rng = thread_rng();
                let shape = tensor.shape();

                match tensor {
                    Tensor::F32(data) => {
                        let fan_in = if shape.len() >= 2 { shape[shape.len() - 1] } else { 1 };

                        let std = (2.0 / fan_in as f32).sqrt();
                        let normal = Normal::new(0.0, std).map_err(|e| {
                            anyhow::anyhow!("Failed to create normal distribution: {}", e)
                        })?;

                        for value in data.iter_mut() {
                            *value = normal.sample(&mut rng);
                        }
                        Ok(())
                    },
                    _ => Err(anyhow::anyhow!(
                        "Kaiming normal initialization only supports F32 tensors"
                    )),
                }
            },
            InitializationStrategy::TruncatedNormal { std, bounds } => {
                // Truncated normal initialization
                let mut rng = thread_rng();

                match tensor {
                    Tensor::F32(data) => {
                        let normal = Normal::new(0.0, *std).map_err(|e| {
                            anyhow::anyhow!("Failed to create normal distribution: {}", e)
                        })?;

                        for value in data.iter_mut() {
                            loop {
                                let sample = normal.sample(&mut rng);
                                if sample.abs() <= *bounds {
                                    *value = sample;
                                    break;
                                }
                                // Resample if outside bounds
                            }
                        }
                        Ok(())
                    },
                    _ => Err(anyhow::anyhow!(
                        "Truncated normal initialization only supports F32 tensors"
                    )),
                }
            },
            InitializationStrategy::Custom(name) => {
                // Custom initialization - lookup by name
                match name.as_str() {
                    "zero" => {
                        // Initialize all weights to zero
                        match tensor {
                            Tensor::F32(data) => {
                                data.fill(0.0);
                                Ok(())
                            },
                            _ => Err(anyhow::anyhow!(
                                "Custom zero initialization only supports F32 tensors"
                            )),
                        }
                    },
                    "ones" => {
                        // Initialize all weights to one
                        match tensor {
                            Tensor::F32(data) => {
                                data.fill(1.0);
                                Ok(())
                            },
                            _ => Err(anyhow::anyhow!(
                                "Custom ones initialization only supports F32 tensors"
                            )),
                        }
                    },
                    "identity" => {
                        // Initialize as identity matrix (for square matrices)
                        let shape = tensor.shape();
                        match tensor {
                            Tensor::F32(data) => {
                                if shape.len() == 2 && shape[0] == shape[1] {
                                    // Square matrix - initialize as identity
                                    data.fill(0.0);
                                    let dim = shape[0];
                                    for i in 0..dim {
                                        data[i * dim + i] = 1.0;
                                    }
                                    Ok(())
                                } else {
                                    Err(anyhow::anyhow!(
                                        "Identity initialization requires square matrix"
                                    ))
                                }
                            },
                            _ => Err(anyhow::anyhow!(
                                "Custom identity initialization only supports F32 tensors"
                            )),
                        }
                    },
                    _ => Err(anyhow::anyhow!(
                        "Unknown custom initialization strategy: {}",
                        name
                    )),
                }
            },
        }
    }

    /// Create a standard generation configuration for a task
    pub fn generation_config_for_task(task: &TaskType) -> GenerationConfig {
        match task {
            TaskType::TextGeneration => GenerationConfig {
                max_new_tokens: 512,
                temperature: 0.7,
                top_p: 0.9,
                repetition_penalty: 1.1,
                ..GenerationConfig::default()
            },
            TaskType::CodeGeneration => GenerationConfig {
                max_new_tokens: 1024,
                temperature: 0.2,
                top_p: 0.95,
                repetition_penalty: 1.05,
                ..GenerationConfig::default()
            },
            TaskType::Summarization => GenerationConfig {
                max_new_tokens: 256,
                temperature: 0.3,
                top_p: 0.9,
                repetition_penalty: 1.2,
                ..GenerationConfig::default()
            },
            TaskType::QuestionAnswering => GenerationConfig {
                max_new_tokens: 128,
                temperature: 0.1,
                top_p: 0.95,
                repetition_penalty: 1.0,
                early_stopping: true,
                ..GenerationConfig::default()
            },
            _ => GenerationConfig::default(),
        }
    }

    /// Validate model configuration
    pub fn validate_config(config: &dyn DynConfig) -> Result<Vec<String>> {
        let warnings = Vec::new();

        // Basic validation
        config.validate()?;

        // Additional checks can be added here
        // For example, checking if model size is reasonable, etc.

        Ok(warnings)
    }

    /// Estimate model memory requirements
    pub fn estimate_memory_requirements(
        vocab_size: usize,
        hidden_size: usize,
        num_layers: usize,
        context_length: usize,
    ) -> MemoryEstimate {
        // Rough estimation for transformer models
        let embedding_params = vocab_size * hidden_size;
        let layer_params = num_layers
            * (
                // Self-attention weights
                hidden_size * hidden_size * 4 +
            // Feed-forward weights (assuming 4x expansion)
            hidden_size * hidden_size * 4 * 2 +
            // Layer norms
            hidden_size * 2
            );
        let output_head_params = vocab_size * hidden_size;

        let total_params = embedding_params + layer_params + output_head_params;

        // Estimate memory usage (parameters + gradients + optimizer states + activations)
        let model_memory_gb = (total_params * 4) as f32 / 1_000_000_000.0; // 4 bytes per param (FP32)
        let activation_memory_gb =
            (context_length * hidden_size * num_layers * 4) as f32 / 1_000_000_000.0;

        MemoryEstimate {
            total_parameters: total_params,
            model_memory_gb,
            activation_memory_gb,
            total_memory_gb: model_memory_gb + activation_memory_gb,
            inference_memory_gb: model_memory_gb + activation_memory_gb * 0.5,
        }
    }
}

/// Memory usage estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEstimate {
    pub total_parameters: usize,
    pub model_memory_gb: f32,
    pub activation_memory_gb: f32,
    pub total_memory_gb: f32,
    pub inference_memory_gb: f32,
}

/// Generation strategy patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenerationStrategy {
    /// Greedy decoding
    Greedy,
    /// Sampling with temperature
    Sampling { temperature: f32 },
    /// Top-k sampling
    TopK { k: usize, temperature: f32 },
    /// Nucleus (top-p) sampling
    TopP { p: f32, temperature: f32 },
    /// Beam search
    BeamSearch { num_beams: usize },
    /// Diverse beam search
    DiverseBeamSearch {
        num_beams: usize,
        diversity_penalty: f32,
    },
    /// Contrastive search
    ContrastiveSearch { penalty_alpha: f32, top_k: usize },
}

/// Common architectural components
pub mod components {
    use super::*;

    /// Standard transformer layer interface
    pub trait TransformerLayer {
        /// Forward pass through the layer
        fn forward(&self, input: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor>;

        /// Get layer configuration
        fn config(&self) -> &dyn DynConfig;
    }

    /// Standard attention mechanism interface
    pub trait AttentionMechanism {
        /// Compute attention weights
        fn compute_attention(&self, query: &Tensor, key: &Tensor, value: &Tensor)
            -> Result<Tensor>;

        /// Apply attention mask
        fn apply_mask(&self, attention_weights: &Tensor, mask: &Tensor) -> Result<Tensor>;
    }

    /// Standard feed-forward network interface
    pub trait FeedForwardNetwork {
        /// Forward pass through FFN
        fn forward(&self, input: &Tensor) -> Result<Tensor>;

        /// Get hidden dimensions
        fn hidden_size(&self) -> usize;
        fn intermediate_size(&self) -> usize;
    }

    /// Standard embedding layer interface
    pub trait EmbeddingLayer {
        /// Forward pass for token embeddings
        fn forward(&self, input_ids: &Tensor) -> Result<Tensor>;

        /// Get vocabulary size
        fn vocab_size(&self) -> usize;

        /// Get embedding dimension
        fn embedding_dim(&self) -> usize;
    }
}

/// Model family implementation for registry
pub trait ModelFamilyImpl: Send + Sync {
    /// Create configuration for a specific size and variant
    fn create_config(&self, size: &str, variant: Option<&str>) -> Result<Box<dyn DynConfig>>;

    /// Get family name
    fn family_name(&self) -> &'static str;

    /// Get available model sizes
    fn available_sizes(&self) -> Vec<&'static str>;

    /// Get available variants (base, instruct, chat, etc.)
    fn available_variants(&self) -> Vec<&'static str>;

    /// Get recommended use cases for this family
    fn use_cases(&self) -> Vec<&'static str>;

    /// Get model family metadata
    fn metadata(&self) -> ModelFamilyMetadata;
}

/// Wrapper for ModelFamily trait to make it work with ModelRegistry
pub struct ModelFamilyWrapper<T: ModelFamily>(std::marker::PhantomData<T>);

impl<T: ModelFamily> Default for ModelFamilyWrapper<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ModelFamily> ModelFamilyWrapper<T> {
    pub fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T: ModelFamily> ModelFamilyImpl for ModelFamilyWrapper<T> {
    fn create_config(&self, size: &str, variant: Option<&str>) -> Result<Box<dyn DynConfig>> {
        T::create_config(size, variant)
    }

    fn family_name(&self) -> &'static str {
        T::family_name()
    }

    fn available_sizes(&self) -> Vec<&'static str> {
        T::available_sizes()
    }

    fn available_variants(&self) -> Vec<&'static str> {
        T::available_variants()
    }

    fn use_cases(&self) -> Vec<&'static str> {
        T::use_cases()
    }

    fn metadata(&self) -> ModelFamilyMetadata {
        T::metadata()
    }
}

/// Model registry for dynamic model creation
pub struct ModelRegistry {
    families: HashMap<String, Box<dyn ModelFamilyImpl>>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            families: HashMap::new(),
        }
    }

    pub fn register_family<F: ModelFamilyImpl + 'static>(&mut self, family: F) {
        let name = family.family_name().to_string();
        self.families.insert(name, Box::new(family));
    }

    pub fn register_model_family<F: ModelFamily + 'static>(&mut self) {
        let wrapper = ModelFamilyWrapper::<F>::new();
        self.register_family(wrapper);
    }

    pub fn get_family(&self, name: &str) -> Option<&dyn ModelFamilyImpl> {
        self.families.get(name).map(|f| f.as_ref())
    }

    pub fn list_families(&self) -> Vec<&str> {
        self.families.keys().map(|s| s.as_str()).collect()
    }

    pub fn create_model(
        &self,
        family: &str,
        size: &str,
        variant: Option<&str>,
    ) -> Result<Box<dyn DynConfig>> {
        let family_impl = self
            .get_family(family)
            .ok_or_else(|| anyhow::anyhow!("Unknown model family: {}", family))?;

        family_impl.create_config(size, variant)
    }
}

/// Global model registry instance
static MODEL_REGISTRY: OnceLock<Mutex<ModelRegistry>> = OnceLock::new();

pub fn get_global_registry() -> &'static Mutex<ModelRegistry> {
    MODEL_REGISTRY.get_or_init(|| Mutex::new(ModelRegistry::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 100);
        assert_eq!(config.temperature, 1.0);
        assert!(config.do_sample);
    }

    #[test]
    fn test_task_specific_generation_config() {
        let code_config = ModelUtils::generation_config_for_task(&TaskType::CodeGeneration);
        assert_eq!(code_config.temperature, 0.2);
        assert_eq!(code_config.max_new_tokens, 1024);

        let qa_config = ModelUtils::generation_config_for_task(&TaskType::QuestionAnswering);
        assert_eq!(qa_config.temperature, 0.1);
        assert!(qa_config.early_stopping);
    }

    #[test]
    fn test_memory_estimation() {
        let estimate = ModelUtils::estimate_memory_requirements(32000, 4096, 32, 2048);
        assert!(estimate.total_parameters > 0);
        assert!(estimate.model_memory_gb > 0.0);
        assert!(estimate.total_memory_gb >= estimate.model_memory_gb);
    }

    #[test]
    fn test_model_registry() {
        let registry = ModelRegistry::new();
        // Test basic registry operations
        assert_eq!(registry.list_families().len(), 0);
    }

    #[test]
    fn test_initialization_strategy() {
        let strategy = InitializationStrategy::Normal { std: 0.02 };
        match strategy {
            InitializationStrategy::Normal { std } => assert_eq!(std, 0.02),
            _ => panic!("Wrong strategy type"),
        }
    }

    #[test]
    fn test_architecture_types() {
        let encoder_only = ArchitectureType::EncoderOnly;
        let decoder_only = ArchitectureType::DecoderOnly;
        assert_ne!(encoder_only, decoder_only);
    }
}
