// ONNX Pipeline Backend Integration for TrustformeRS
// Provides seamless ONNX Runtime integration with the existing pipeline system

use crate::core::traits::TokenizedInput;
use crate::error::Result;
use crate::pipeline::{BasePipeline, Device, Pipeline, PipelineOptions, PipelineOutput};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use trustformers_core::errors::{runtime_error, Result as CoreResult};
use trustformers_core::traits::{Model, Tokenizer};
// Note: Using mock types since actual ONNX runtime types need implementation
use trustformers_core::export::onnx_runtime::{
    ExecutionMode, ExecutionProvider, GraphOptimizationLevel, LogLevel, MemoryInfo,
    ONNXRuntimeConfig, ONNXRuntimeSession,
};

// Mock ONNX types - replace with actual implementation
#[derive(Debug, Clone)]
pub struct ONNXRuntimeBackend;

#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub avg_latency_ms: f64,
    pub throughput: f64,
    pub memory_usage: u64,
}

// Mock implementations
impl ONNXRuntimeBackend {
    pub fn new(_config: ONNXRuntimeConfig) -> CoreResult<Self> {
        Ok(Self)
    }

    pub fn load_model(&self, path: &PathBuf) -> CoreResult<ONNXRuntimeSession> {
        let backend = crate::core::export::onnx_runtime::ONNXRuntimeBackend::new();
        backend
            .load_model(path)
            .map_err(|e| runtime_error(format!("Failed to load ONNX model: {}", e)))
    }

    pub fn get_available_providers(&self) -> CoreResult<Vec<ExecutionProvider>> {
        Ok(vec![ExecutionProvider::CPU])
    }

    pub fn get_device_properties(
        &self,
        _provider: &ExecutionProvider,
    ) -> CoreResult<HashMap<String, String>> {
        let mut props = HashMap::new();
        props.insert("type".to_string(), "mock".to_string());
        Ok(props)
    }
}

/// Trait for ONNX session operations
pub trait ONNXSessionOps {
    fn input_names(&self) -> &[String];
    fn output_names(&self) -> &[String];
    fn run(
        &self,
        inputs: HashMap<String, trustformers_core::tensor::Tensor>,
    ) -> Result<HashMap<String, trustformers_core::tensor::Tensor>>;
    fn run_with_provider(
        &self,
        inputs: HashMap<String, trustformers_core::tensor::Tensor>,
        provider: ExecutionProvider,
    ) -> Result<HashMap<String, trustformers_core::tensor::Tensor>>;
    fn run_async(
        &self,
        inputs: HashMap<String, trustformers_core::tensor::Tensor>,
    ) -> impl std::future::Future<Output = Result<HashMap<String, trustformers_core::tensor::Tensor>>>
           + Send;
    fn benchmark(
        &self,
        inputs: HashMap<String, trustformers_core::tensor::Tensor>,
        num_runs: usize,
        warmup_runs: usize,
    ) -> Result<BenchmarkResults>;
    fn get_memory_info(&self) -> Result<MemoryInfo>;
}

impl ONNXSessionOps for ONNXRuntimeSession {
    fn input_names(&self) -> &[String] {
        self.input_names()
    }

    fn output_names(&self) -> &[String] {
        self.output_names()
    }

    fn run(
        &self,
        _inputs: HashMap<String, trustformers_core::tensor::Tensor>,
    ) -> Result<HashMap<String, trustformers_core::tensor::Tensor>> {
        // Mock implementation - return empty result
        let mut outputs = HashMap::new();
        let mock_tensor = trustformers_core::tensor::Tensor::zeros(&[1, 10])
            .map_err(crate::error::TrustformersError::from)?;
        outputs.insert("logits".to_string(), mock_tensor);
        Ok(outputs)
    }

    fn run_with_provider(
        &self,
        inputs: HashMap<String, trustformers_core::tensor::Tensor>,
        _provider: ExecutionProvider,
    ) -> Result<HashMap<String, trustformers_core::tensor::Tensor>> {
        ONNXSessionOps::run(self, inputs)
    }

    async fn run_async(
        &self,
        inputs: HashMap<String, trustformers_core::tensor::Tensor>,
    ) -> Result<HashMap<String, trustformers_core::tensor::Tensor>> {
        ONNXSessionOps::run(self, inputs)
    }

    fn benchmark(
        &self,
        _inputs: HashMap<String, trustformers_core::tensor::Tensor>,
        _num_runs: usize,
        _warmup_runs: usize,
    ) -> Result<BenchmarkResults> {
        Ok(BenchmarkResults {
            avg_latency_ms: 30.0,
            throughput: 33.0,
            memory_usage: 512 * 1024 * 1024, // 512MB
        })
    }

    fn get_memory_info(&self) -> Result<MemoryInfo> {
        Ok(MemoryInfo {
            total_memory_bytes: 4 * 1024 * 1024 * 1024,     // 4GB
            model_memory_bytes: 1024 * 1024 * 1024,         // 1GB
            available_memory_bytes: 3 * 1024 * 1024 * 1024, // 3GB
        })
    }
}
use trustformers_core::tensor::Tensor;

/// ONNX backend configuration for pipelines
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ONNXBackendConfig {
    pub model_path: PathBuf,
    pub execution_providers: Vec<ExecutionProvider>,
    pub optimization_level: GraphOptimizationLevel,
    pub execution_mode: ExecutionMode,
    pub inter_op_threads: Option<usize>,
    pub intra_op_threads: Option<usize>,
    pub enable_memory_pattern: bool,
    pub enable_cpu_mem_arena: bool,
    pub log_level: LogLevel,
    pub enable_profiling: bool,
    pub profile_output_path: Option<PathBuf>,
}

impl Default for ONNXBackendConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            execution_providers: vec![ExecutionProvider::CPU],
            optimization_level: GraphOptimizationLevel::All,
            execution_mode: ExecutionMode::Sequential,
            inter_op_threads: None,
            intra_op_threads: None,
            enable_memory_pattern: true,
            enable_cpu_mem_arena: true,
            log_level: LogLevel::Warning,
            enable_profiling: false,
            profile_output_path: None,
        }
    }
}

impl crate::core::traits::Config for ONNXBackendConfig {
    fn validate(&self) -> CoreResult<()> {
        if !self.model_path.exists() {
            return Err(runtime_error(format!(
                "ONNX model file not found: {:?}",
                self.model_path
            )));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "onnx"
    }
}

impl ONNXBackendConfig {
    /// Create config optimized for CPU inference
    pub fn cpu_optimized(model_path: PathBuf) -> Self {
        Self {
            model_path,
            execution_providers: vec![ExecutionProvider::CPU],
            optimization_level: GraphOptimizationLevel::All,
            execution_mode: ExecutionMode::Parallel,
            inter_op_threads: Some(num_cpus::get()),
            intra_op_threads: Some(num_cpus::get()),
            enable_memory_pattern: true,
            enable_cpu_mem_arena: true,
            log_level: LogLevel::Warning,
            enable_profiling: false,
            profile_output_path: None,
        }
    }

    /// Create config optimized for GPU inference
    pub fn gpu_optimized(model_path: PathBuf, device_id: Option<i32>) -> Self {
        let mut providers = vec![ExecutionProvider::CUDA { device_id }];

        // Add TensorRT if available
        if std::env::var("TENSORRT_ROOT").is_ok() {
            providers.insert(0, ExecutionProvider::TensorRT { device_id });
        }

        // Fallback to CPU
        providers.push(ExecutionProvider::CPU);

        Self {
            model_path,
            execution_providers: providers,
            optimization_level: GraphOptimizationLevel::All,
            execution_mode: ExecutionMode::Sequential, // GPU typically better with sequential
            inter_op_threads: Some(1),
            intra_op_threads: Some(1),
            enable_memory_pattern: true,
            enable_cpu_mem_arena: false, // Not needed for GPU
            log_level: LogLevel::Warning,
            enable_profiling: false,
            profile_output_path: None,
        }
    }

    /// Create config for production deployment
    pub fn production(model_path: PathBuf) -> Self {
        Self {
            model_path,
            execution_providers: vec![
                ExecutionProvider::CUDA { device_id: Some(0) },
                ExecutionProvider::CPU,
            ],
            optimization_level: GraphOptimizationLevel::All,
            execution_mode: ExecutionMode::Sequential,
            inter_op_threads: Some(1),
            intra_op_threads: Some(1),
            enable_memory_pattern: true,
            enable_cpu_mem_arena: true,
            log_level: LogLevel::Error, // Minimal logging in production
            enable_profiling: false,
            profile_output_path: None,
        }
    }

    /// Enable profiling with output path
    pub fn with_profiling(mut self, output_path: PathBuf) -> Self {
        self.enable_profiling = true;
        self.profile_output_path = Some(output_path);
        self
    }

    /// Convert to ONNX Runtime config
    pub fn to_runtime_config(&self) -> ONNXRuntimeConfig {
        ONNXRuntimeConfig {
            inter_op_num_threads: self.inter_op_threads,
            intra_op_num_threads: self.intra_op_threads,
            enable_cpu_mem_arena: self.enable_cpu_mem_arena,
            enable_mem_pattern: self.enable_memory_pattern,
            execution_mode: self.execution_mode.clone(),
            graph_optimization_level: self.optimization_level.clone(),
            log_severity_level: self.log_level.clone(),
        }
    }
}

/// ONNX-backed model wrapper
#[derive(Clone)]
pub struct ONNXModel {
    session: Arc<ONNXRuntimeSession>,
    config: ONNXBackendConfig,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl ONNXModel {
    /// Create new ONNX model from config
    pub fn from_config(config: ONNXBackendConfig) -> CoreResult<Self> {
        if !config.model_path.exists() {
            return Err(runtime_error(format!(
                "ONNX model file not found: {:?}",
                config.model_path
            )));
        }

        let runtime_config = config.to_runtime_config();
        let backend = ONNXRuntimeBackend::new(runtime_config)?;
        let session = backend.load_model(&config.model_path)?;

        let input_names = session.input_names().to_vec();
        let output_names = session.output_names().to_vec();

        Ok(Self {
            session: Arc::new(session),
            config,
            input_names,
            output_names,
        })
    }

    /// Load from ONNX file with default config
    pub fn from_pretrained<P: AsRef<Path>>(model_path: P) -> CoreResult<Self> {
        let config = ONNXBackendConfig {
            model_path: model_path.as_ref().to_path_buf(),
            ..Default::default()
        };
        Self::from_config(config)
    }

    /// Load with specific execution providers
    pub fn from_pretrained_with_providers<P: AsRef<Path>>(
        model_path: P,
        providers: Vec<ExecutionProvider>,
    ) -> CoreResult<Self> {
        let config = ONNXBackendConfig {
            model_path: model_path.as_ref().to_path_buf(),
            execution_providers: providers,
            ..Default::default()
        };
        Self::from_config(config)
    }

    /// Get input names
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Get output names
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Run inference
    pub fn forward(&self, inputs: HashMap<String, Tensor>) -> CoreResult<HashMap<String, Tensor>> {
        self.session.run(inputs).map_err(Into::into)
    }

    /// Run inference with specific provider
    pub fn forward_with_provider(
        &self,
        inputs: HashMap<String, Tensor>,
        provider: ExecutionProvider,
    ) -> CoreResult<HashMap<String, Tensor>> {
        self.session.run_with_provider(inputs, provider).map_err(Into::into)
    }

    /// Benchmark the model
    pub fn benchmark(
        &self,
        inputs: HashMap<String, Tensor>,
        num_runs: usize,
    ) -> CoreResult<BenchmarkResults> {
        // Mock benchmark implementation
        Ok(BenchmarkResults {
            avg_latency_ms: 10.0,
            throughput: 100.0,
            memory_usage: 1024 * 1024, // 1MB
        })
    }

    /// Get memory usage information
    pub fn memory_info(&self) -> CoreResult<MemoryInfo> {
        // Mock memory info implementation
        Ok(MemoryInfo {
            total_memory_bytes: 1024 * 1024 * 1024,    // 1GB
            model_memory_bytes: 512 * 1024 * 1024,     // 512MB
            available_memory_bytes: 512 * 1024 * 1024, // 512MB
        })
    }

    /// Get available execution providers
    pub fn execution_providers(&self) -> &[ExecutionProvider] {
        self.session.execution_providers()
    }

    /// Get model path
    pub fn model_path(&self) -> &Path {
        &self.config.model_path
    }
}

impl Model for ONNXModel {
    type Config = ONNXBackendConfig;
    type Input = HashMap<String, Tensor>;
    type Output = HashMap<String, Tensor>;

    /// Forward pass implementation for Model trait
    fn forward(&self, inputs: Self::Input) -> CoreResult<Self::Output> {
        // Run inference using the ONNX session
        self.session.run(inputs).map_err(Into::into)
    }

    /// Load pretrained weights (not applicable for ONNX models as they're already loaded)
    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> CoreResult<()> {
        // ONNX models are already loaded from file, so this is a no-op
        Ok(())
    }

    /// Get model configuration
    fn get_config(&self) -> &ONNXBackendConfig {
        &self.config
    }

    /// Get the number of parameters in the model
    fn num_parameters(&self) -> usize {
        // For ONNX models, we can't easily determine this without parsing the model
        // Return a placeholder value or implement actual parameter counting if needed
        0 // Placeholder - would need ONNX model introspection
    }
}

/// ONNX tokenizer wrapper (can wrap existing tokenizers)
#[derive(Clone)]
pub struct ONNXTokenizer<T> {
    inner: T,
}

impl<T: Tokenizer> ONNXTokenizer<T> {
    pub fn new(tokenizer: T) -> Self {
        Self { inner: tokenizer }
    }
}

impl<T: Tokenizer> Tokenizer for ONNXTokenizer<T> {
    fn encode(&self, text: &str) -> CoreResult<TokenizedInput> {
        self.inner.encode(text)
    }

    fn encode_pair(&self, text: &str, text2: &str) -> CoreResult<TokenizedInput> {
        self.inner.encode_pair(text, text2)
    }

    fn decode(&self, ids: &[u32]) -> CoreResult<String> {
        self.inner.decode(ids)
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn get_vocab(&self) -> std::collections::HashMap<String, u32> {
        self.inner.get_vocab()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
}

/// ONNX-backed pipeline base
pub type ONNXBasePipeline<T> = BasePipeline<ONNXModel, ONNXTokenizer<T>>;

/// Text classification pipeline with ONNX backend
pub struct ONNXTextClassificationPipeline<T> {
    base: ONNXBasePipeline<T>,
    return_all_scores: bool,
}

impl<T: Tokenizer + Clone> ONNXTextClassificationPipeline<T> {
    pub fn new(model: ONNXModel, tokenizer: ONNXTokenizer<T>) -> CoreResult<Self> {
        Ok(Self {
            base: BasePipeline::new(model, tokenizer),
            return_all_scores: false,
        })
    }

    pub fn with_return_all_scores(mut self, return_all: bool) -> Self {
        self.return_all_scores = return_all;
        self
    }

    /// Benchmark this pipeline
    pub fn benchmark(&self, input: &str, num_runs: usize) -> CoreResult<BenchmarkResults> {
        let tokenized = self.base.tokenizer.encode(input)?;
        let inputs = self.prepare_inputs(&tokenized)?;
        self.base.model.benchmark(inputs, num_runs)
    }

    /// Get memory usage
    pub fn memory_info(&self) -> CoreResult<MemoryInfo> {
        self.base.model.memory_info()
    }

    fn prepare_inputs(&self, tokenized: &TokenizedInput) -> CoreResult<HashMap<String, Tensor>> {
        let mut inputs = HashMap::new();

        let batch_size = 1;
        let seq_len = tokenized.input_ids.len();

        // Input IDs
        let input_ids = Tensor::from_vec(
            tokenized.input_ids.iter().map(|&x| x as f32).collect(),
            &[batch_size, seq_len],
        )?;
        inputs.insert("input_ids".to_string(), input_ids);

        // Attention mask
        let attention_mask = if !tokenized.attention_mask.is_empty() {
            Tensor::from_vec(
                tokenized.attention_mask.iter().map(|&x| x as f32).collect(),
                &[batch_size, seq_len],
            )?
        } else {
            Tensor::from_vec(vec![1.0f32; batch_size * seq_len], &[batch_size, seq_len])?
        };
        inputs.insert("attention_mask".to_string(), attention_mask);

        Ok(inputs)
    }
}

impl<T: Tokenizer + Clone> Pipeline for ONNXTextClassificationPipeline<T> {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        let tokenized = self
            .base
            .tokenizer
            .encode(&input)
            .map_err(crate::error::TrustformersError::from)?;
        let inputs = self.prepare_inputs(&tokenized)?;
        let outputs =
            self.base.model.forward(inputs).map_err(crate::error::TrustformersError::from)?;

        // Get logits (assuming first output)
        let logits = outputs.into_values().next().ok_or_else(|| {
            crate::error::TrustformersError::from(runtime_error("No logits output"))
        })?;

        // Apply softmax to get probabilities (simplified)
        let logits_data = logits.data();
        let flat_data: Vec<f32> = logits_data.iter().flatten().cloned().collect();
        let max_logit = flat_data.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
        let exp_logits: Vec<f32> = flat_data.iter().map(|x| (*x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

        // Create classification outputs
        let mut results = Vec::new();
        if self.return_all_scores {
            for (i, &score) in probs.iter().enumerate() {
                results.push(crate::pipeline::ClassificationOutput {
                    label: format!("LABEL_{}", i),
                    score,
                });
            }
        } else {
            // Return only the highest scoring label
            let (max_idx, &max_score) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .ok_or_else(|| crate::error::TrustformersError::Pipeline {
                    message: "Cannot find maximum probability: empty classification output"
                        .to_string(),
                    pipeline_type: "text-classification".to_string(),
                    suggestion: Some("Ensure the model produces non-empty logits".to_string()),
                    recovery_actions: vec![],
                })?;
            results.push(crate::pipeline::ClassificationOutput {
                label: format!("LABEL_{}", max_idx),
                score: max_score,
            });
        }

        Ok(PipelineOutput::Classification(results))
    }
}

/// Text generation pipeline with ONNX backend
pub struct ONNXTextGenerationPipeline<T> {
    base: ONNXBasePipeline<T>,
    max_new_tokens: usize,
    do_sample: bool,
    temperature: f32,
    top_p: f32,
}

impl<T: Tokenizer + Clone> ONNXTextGenerationPipeline<T> {
    pub fn new(model: ONNXModel, tokenizer: ONNXTokenizer<T>) -> CoreResult<Self> {
        Ok(Self {
            base: BasePipeline::new(model, tokenizer),
            max_new_tokens: 50,
            do_sample: false,
            temperature: 1.0,
            top_p: 1.0,
        })
    }

    pub fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }

    pub fn with_do_sample(mut self, do_sample: bool) -> Self {
        self.do_sample = do_sample;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }
}

impl<T: Tokenizer + Clone> Pipeline for ONNXTextGenerationPipeline<T> {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        let tokenized = self
            .base
            .tokenizer
            .encode(&input)
            .map_err(crate::error::TrustformersError::from)?;
        let mut input_ids = tokenized.input_ids.clone();

        // Simple autoregressive generation
        for _ in 0..self.max_new_tokens {
            let mut inputs = HashMap::new();

            let batch_size = 1;
            let seq_len = input_ids.len();

            let input_ids_tensor = Tensor::from_vec(
                input_ids.iter().map(|&x| x as f32).collect(),
                &[batch_size, seq_len],
            )
            .map_err(crate::error::TrustformersError::from)?;
            inputs.insert("input_ids".to_string(), input_ids_tensor);

            let attention_mask =
                Tensor::from_vec(vec![1.0f32; batch_size * seq_len], &[batch_size, seq_len])
                    .map_err(crate::error::TrustformersError::from)?;
            inputs.insert("attention_mask".to_string(), attention_mask);

            let outputs =
                self.base.model.forward(inputs).map_err(crate::error::TrustformersError::from)?;
            let logits = outputs.into_values().next().ok_or_else(|| {
                crate::error::TrustformersError::from(runtime_error("No logits output"))
            })?;

            // Get next token (simplified greedy decoding)
            let logits_data = logits.data();
            let flat_data: Vec<f32> = logits_data.iter().flatten().cloned().collect();
            let vocab_size = flat_data.len() / (batch_size * seq_len);
            let last_token_logits = &flat_data[(seq_len - 1) * vocab_size..seq_len * vocab_size];

            let next_token = if self.do_sample {
                // Apply temperature and sampling (simplified)
                let scaled_logits: Vec<f32> =
                    last_token_logits.iter().map(|&x| x / self.temperature).collect();
                let max_logit = scaled_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_logits: Vec<f32> =
                    scaled_logits.iter().map(|&x| (x - max_logit).exp()).collect();
                let sum_exp: f32 = exp_logits.iter().sum();
                let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

                // Sample from distribution (simplified random selection)
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                input_ids.hash(&mut hasher);
                let hash = hasher.finish();
                let random_val = (hash % 1000) as f32 / 1000.0;

                let mut cumulative = 0.0;
                let mut selected_token = 0;
                for (i, &prob) in probs.iter().enumerate() {
                    cumulative += prob;
                    if random_val <= cumulative {
                        selected_token = i;
                        break;
                    }
                }
                selected_token as u32
            } else {
                // Greedy decoding
                last_token_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as u32)
                    .ok_or_else(|| crate::error::TrustformersError::Pipeline {
                        message: "Cannot find maximum logit: empty token logits".to_string(),
                        pipeline_type: "text-generation".to_string(),
                        suggestion: Some(
                            "Ensure the model produces non-empty logits for greedy decoding"
                                .to_string(),
                        ),
                        recovery_actions: vec![],
                    })?
            };

            input_ids.push(next_token);

            // Simple stopping condition (can be improved)
            if next_token == 0 || next_token == 2 {
                // Common EOS tokens
                break;
            }
        }

        let generated_text = self
            .base
            .tokenizer
            .decode(&input_ids)
            .map_err(crate::error::TrustformersError::from)?;

        Ok(PipelineOutput::Generation(
            crate::pipeline::GenerationOutput {
                generated_text,
                sequences: Some(vec![input_ids]),
                scores: None,
            },
        ))
    }
}

/// Factory functions for ONNX pipelines
pub fn onnx_text_classification_pipeline<T: Tokenizer + Clone>(
    model_path: impl AsRef<Path>,
    tokenizer: T,
    config: Option<ONNXBackendConfig>,
) -> CoreResult<ONNXTextClassificationPipeline<T>> {
    let config = config
        .unwrap_or_else(|| ONNXBackendConfig::cpu_optimized(model_path.as_ref().to_path_buf()));
    let model = ONNXModel::from_config(config)?;
    let onnx_tokenizer = ONNXTokenizer::new(tokenizer);
    ONNXTextClassificationPipeline::new(model, onnx_tokenizer)
}

pub fn onnx_text_generation_pipeline<T: Tokenizer + Clone>(
    model_path: impl AsRef<Path>,
    tokenizer: T,
    config: Option<ONNXBackendConfig>,
) -> CoreResult<ONNXTextGenerationPipeline<T>> {
    let config = config
        .unwrap_or_else(|| ONNXBackendConfig::cpu_optimized(model_path.as_ref().to_path_buf()));
    let model = ONNXModel::from_config(config)?;
    let onnx_tokenizer = ONNXTokenizer::new(tokenizer);
    ONNXTextGenerationPipeline::new(model, onnx_tokenizer)
}

/// Enhanced pipeline options with ONNX backend support
#[derive(Clone, Debug)]
pub struct ONNXPipelineOptions {
    pub base_options: PipelineOptions,
    pub onnx_config: ONNXBackendConfig,
    pub enable_profiling: bool,
    pub warmup_runs: usize,
}

impl Default for ONNXPipelineOptions {
    fn default() -> Self {
        Self {
            base_options: PipelineOptions::default(),
            onnx_config: ONNXBackendConfig::default(),
            enable_profiling: false,
            warmup_runs: 3,
        }
    }
}

impl ONNXPipelineOptions {
    pub fn cpu_optimized(model_path: PathBuf) -> Self {
        Self {
            base_options: PipelineOptions::default(),
            onnx_config: ONNXBackendConfig::cpu_optimized(model_path),
            enable_profiling: false,
            warmup_runs: 3,
        }
    }

    pub fn gpu_optimized(model_path: PathBuf, device_id: Option<i32>) -> Self {
        Self {
            base_options: PipelineOptions {
                device: Some(Device::Gpu(device_id.unwrap_or(0) as usize)),
                ..Default::default()
            },
            onnx_config: ONNXBackendConfig::gpu_optimized(model_path, device_id),
            enable_profiling: false,
            warmup_runs: 3,
        }
    }

    pub fn with_profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }

    pub fn with_warmup_runs(mut self, runs: usize) -> Self {
        self.warmup_runs = runs;
        self
    }
}

/// ONNX pipeline manager for coordinating multiple backends
pub struct ONNXPipelineManager {
    models: HashMap<String, ONNXModel>,
    default_config: ONNXBackendConfig,
}

impl ONNXPipelineManager {
    pub fn new(default_config: ONNXBackendConfig) -> Self {
        Self {
            models: HashMap::new(),
            default_config,
        }
    }

    /// Register a model with the manager
    pub fn register_model(&mut self, name: String, model: ONNXModel) {
        self.models.insert(name, model);
    }

    /// Load and register a model from path
    pub fn load_model<P: AsRef<Path>>(&mut self, name: String, model_path: P) -> CoreResult<()> {
        let mut config = self.default_config.clone();
        config.model_path = model_path.as_ref().to_path_buf();
        let model = ONNXModel::from_config(config)?;
        self.register_model(name, model);
        Ok(())
    }

    /// Get a registered model
    pub fn get_model(&self, name: &str) -> Option<&ONNXModel> {
        self.models.get(name)
    }

    /// List all registered models
    pub fn list_models(&self) -> Vec<&String> {
        self.models.keys().collect()
    }

    /// Benchmark all registered models
    pub fn benchmark_all(
        &self,
        inputs: HashMap<String, Tensor>,
        num_runs: usize,
    ) -> CoreResult<HashMap<String, BenchmarkResults>> {
        let mut results = HashMap::new();
        for (name, model) in &self.models {
            let benchmark = model.benchmark(inputs.clone(), num_runs)?;
            results.insert(name.clone(), benchmark);
        }
        Ok(results)
    }

    /// Get memory info for all models
    pub fn memory_info_all(&self) -> CoreResult<HashMap<String, MemoryInfo>> {
        let mut results = HashMap::new();
        for (name, model) in &self.models {
            let info = model.memory_info()?;
            results.insert(name.clone(), info);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_onnx_backend_config() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.onnx");

        let config = ONNXBackendConfig::cpu_optimized(model_path.clone());
        assert_eq!(config.model_path, model_path);
        assert!(matches!(
            config.execution_providers[0],
            ExecutionProvider::CPU
        ));

        let runtime_config = config.to_runtime_config();
        assert!(runtime_config.inter_op_num_threads.is_some());
    }

    #[test]
    fn test_onnx_backend_config_gpu() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.onnx");

        let config = ONNXBackendConfig::gpu_optimized(model_path.clone(), Some(0));
        assert_eq!(config.model_path, model_path);
        assert!(matches!(
            config.execution_providers[0],
            ExecutionProvider::CUDA { .. }
        ));
    }

    #[test]
    fn test_onnx_pipeline_options() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.onnx");

        let options = ONNXPipelineOptions::cpu_optimized(model_path.clone());
        assert_eq!(options.onnx_config.model_path, model_path);
        assert_eq!(options.warmup_runs, 3);

        let gpu_options = ONNXPipelineOptions::gpu_optimized(model_path.clone(), Some(0));
        assert!(matches!(
            gpu_options.base_options.device,
            Some(Device::Gpu(0))
        ));
    }

    #[test]
    fn test_onnx_pipeline_manager() {
        let config = ONNXBackendConfig::default();
        let manager = ONNXPipelineManager::new(config);

        assert_eq!(manager.list_models().len(), 0);

        // In a real test, we would create a mock ONNX model
        // For now, just test the basic structure
    }
}
