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
// This module used to define its own "mock ONNX types" with hardcoded outputs.
// It now thinly wraps trustformers-core's real pure-Rust ONNX CPU interpreter
// (`trustformers_core::export::onnx_runtime` / `onnx_cpu`): `ONNXRuntimeBackend`
// below is a pipeline-facing handle over the core backend, and `ONNXRuntimeSession`
// (imported directly, not reimplemented) really parses `.onnx` protobufs and
// really executes their graphs.
use trustformers_core::export::onnx_runtime::{
    ExecutionMode, ExecutionProvider, GraphOptimizationLevel, LogLevel, MemoryInfo,
    ONNXRuntimeConfig, ONNXRuntimeSession,
};

/// Pipeline-facing handle over the real core ONNX backend.
#[derive(Debug, Clone)]
pub struct ONNXRuntimeBackend;

#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub avg_latency_ms: f64,
    pub throughput: f64,
    pub memory_usage: u64,
}

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

    /// Execution providers this build can actually run a graph on.
    ///
    /// Only [`ExecutionProvider::CPU`] is listed because only the CPU
    /// interpreter exists; this mirrors
    /// [`trustformers_core::export::onnx_runtime::ONNXRuntimeBackend::get_available_providers`]
    /// rather than advertising accelerators nothing here can drive.
    pub fn get_available_providers(&self) -> CoreResult<Vec<ExecutionProvider>> {
        Ok(vec![ExecutionProvider::CPU])
    }

    /// Real host properties for the CPU provider; a structured error for every
    /// other provider, since no execution backend exists for them.
    pub fn get_device_properties(
        &self,
        provider: &ExecutionProvider,
    ) -> CoreResult<HashMap<String, String>> {
        match provider {
            ExecutionProvider::CPU => {
                let mut props = HashMap::new();
                props.insert("type".to_string(), "cpu".to_string());
                props.insert("logical_cores".to_string(), num_cpus::get().to_string());
                props.insert(
                    "physical_cores".to_string(),
                    num_cpus::get_physical().to_string(),
                );
                let mut system = sysinfo::System::new();
                system.refresh_memory();
                props.insert(
                    "total_memory_bytes".to_string(),
                    system.total_memory().to_string(),
                );
                props.insert(
                    "available_memory_bytes".to_string(),
                    system.available_memory().to_string(),
                );
                Ok(props)
            },
            other => Err(runtime_error(format!(
                "no device properties are available for {other:?}: this build executes ONNX \
                 graphs with a pure-Rust CPU interpreter only, no accelerator execution \
                 provider is linked in"
            ))),
        }
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
        ONNXRuntimeSession::input_names(self)
    }

    fn output_names(&self) -> &[String] {
        ONNXRuntimeSession::output_names(self)
    }

    fn run(
        &self,
        inputs: HashMap<String, trustformers_core::tensor::Tensor>,
    ) -> Result<HashMap<String, trustformers_core::tensor::Tensor>> {
        // Delegates to the real interpreter in trustformers-core: it parses the
        // loaded `.onnx` graph and executes every supported node against `inputs`.
        ONNXRuntimeSession::run(self, inputs).map_err(crate::error::TrustformersError::from)
    }

    fn run_with_provider(
        &self,
        inputs: HashMap<String, trustformers_core::tensor::Tensor>,
        provider: ExecutionProvider,
    ) -> Result<HashMap<String, trustformers_core::tensor::Tensor>> {
        ONNXRuntimeSession::run_with_provider(self, inputs, provider)
            .map_err(crate::error::TrustformersError::from)
    }

    async fn run_async(
        &self,
        inputs: HashMap<String, trustformers_core::tensor::Tensor>,
    ) -> Result<HashMap<String, trustformers_core::tensor::Tensor>> {
        ONNXSessionOps::run(self, inputs)
    }

    fn benchmark(
        &self,
        inputs: HashMap<String, trustformers_core::tensor::Tensor>,
        num_runs: usize,
        warmup_runs: usize,
    ) -> Result<BenchmarkResults> {
        // Real warm-up: run and discard, exactly like ONNX Runtime's own benchmark
        // protocol, so caches/allocators are primed before the timed runs.
        for _ in 0..warmup_runs {
            ONNXRuntimeSession::run(self, inputs.clone())
                .map_err(crate::error::TrustformersError::from)?;
        }

        let core_results = ONNXRuntimeSession::benchmark(self, inputs, num_runs.max(1))
            .map_err(crate::error::TrustformersError::from)?;
        let memory = ONNXRuntimeSession::get_memory_info(self)
            .map_err(crate::error::TrustformersError::from)?;

        Ok(BenchmarkResults {
            avg_latency_ms: core_results.mean_latency_ms,
            throughput: if core_results.mean_latency_ms > 0.0 {
                1000.0 / core_results.mean_latency_ms
            } else {
                0.0
            },
            memory_usage: memory.model_memory_bytes as u64,
        })
    }

    fn get_memory_info(&self) -> Result<MemoryInfo> {
        ONNXRuntimeSession::get_memory_info(self).map_err(crate::error::TrustformersError::from)
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
            graph_optimization_level: self.optimization_level,
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

        // Fail loudly here rather than building a pipeline that would only
        // discover mid-inference (or, worse, silently on a lucky graph prefix)
        // that some operator has no CPU-interpreter implementation.
        let unsupported = session.unsupported_operators();
        if !unsupported.is_empty() {
            return Err(runtime_error(format!(
                "ONNX model {:?} uses operators this build's CPU interpreter cannot execute: {}. \
                 Refusing to build a pipeline on top of it.",
                config.model_path,
                unsupported.join(", ")
            )));
        }

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

    /// Benchmark the model by timing real graph execution.
    pub fn benchmark(
        &self,
        inputs: HashMap<String, Tensor>,
        num_runs: usize,
    ) -> CoreResult<BenchmarkResults> {
        ONNXSessionOps::benchmark(self.session.as_ref(), inputs, num_runs, 0).map_err(Into::into)
    }

    /// Get memory usage information: the model's real weight bytes plus the
    /// host's real total/available memory.
    pub fn memory_info(&self) -> CoreResult<MemoryInfo> {
        ONNXSessionOps::get_memory_info(self.session.as_ref()).map_err(Into::into)
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

    /// Get the number of parameters in the model.
    ///
    /// Computed from the real parsed graph: the total element count across every
    /// initializer tensor (weights, biases, embeddings, ...), which is the
    /// standard definition of "parameter count" for a static computation graph.
    fn num_parameters(&self) -> usize {
        self.session
            .executor()
            .graph()
            .initializers
            .iter()
            .map(|tensor| tensor.dims.iter().map(|&d| d.max(0) as usize).product::<usize>())
            .sum()
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
        let flat_data: Vec<f32> = logits.data().map_err(crate::error::TrustformersError::from)?;
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
            let flat_data: Vec<f32> =
                logits.data().map_err(crate::error::TrustformersError::from)?;
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

    // ── Default config fields ─────────────────────────────────────────────────

    #[test]
    fn test_default_config_cpu_provider() {
        let cfg = ONNXBackendConfig::default();
        assert!(
            !cfg.execution_providers.is_empty(),
            "must have at least one provider"
        );
        assert!(matches!(cfg.execution_providers[0], ExecutionProvider::CPU));
    }

    #[test]
    fn test_default_config_memory_pattern_enabled() {
        let cfg = ONNXBackendConfig::default();
        assert!(cfg.enable_memory_pattern);
        assert!(cfg.enable_cpu_mem_arena);
    }

    #[test]
    fn test_default_config_profiling_disabled() {
        let cfg = ONNXBackendConfig::default();
        assert!(!cfg.enable_profiling);
        assert!(cfg.profile_output_path.is_none());
    }

    #[test]
    fn test_default_config_threads_none() {
        let cfg = ONNXBackendConfig::default();
        assert!(cfg.inter_op_threads.is_none());
        assert!(cfg.intra_op_threads.is_none());
    }

    // ── CPU optimised config ──────────────────────────────────────────────────

    #[test]
    fn test_cpu_optimized_threads_set() {
        let temp_dir = tempdir().expect("temp dir");
        let model_path = temp_dir.path().join("model.onnx");
        let cfg = ONNXBackendConfig::cpu_optimized(model_path);
        assert!(
            cfg.inter_op_threads.is_some(),
            "CPU-optimised config should set thread count"
        );
    }

    #[test]
    fn test_cpu_optimized_optimization_level_all() {
        let temp_dir = tempdir().expect("temp dir");
        let model_path = temp_dir.path().join("model.onnx");
        let cfg = ONNXBackendConfig::cpu_optimized(model_path);
        assert!(matches!(
            cfg.optimization_level,
            GraphOptimizationLevel::All
        ));
    }

    // ── GPU optimised config ──────────────────────────────────────────────────

    #[test]
    fn test_gpu_optimized_has_cuda_provider() {
        let temp_dir = tempdir().expect("temp dir");
        let model_path = temp_dir.path().join("model.onnx");
        let cfg = ONNXBackendConfig::gpu_optimized(model_path, Some(0));
        let has_cuda = cfg
            .execution_providers
            .iter()
            .any(|p| matches!(p, ExecutionProvider::CUDA { .. }));
        assert!(has_cuda, "GPU config must include CUDA provider");
    }

    // ── Pipeline manager operations ───────────────────────────────────────────

    #[test]
    fn test_pipeline_manager_list_models_starts_empty() {
        let cfg = ONNXBackendConfig::default();
        let manager = ONNXPipelineManager::new(cfg);
        assert_eq!(manager.list_models().len(), 0);
    }

    #[test]
    fn test_pipeline_manager_get_nonexistent_model() {
        let cfg = ONNXBackendConfig::default();
        let manager = ONNXPipelineManager::new(cfg);
        assert!(
            manager.get_model("does_not_exist").is_none(),
            "get_model for non-registered name must return None"
        );
    }

    // ── Profiling pipeline option ─────────────────────────────────────────────

    #[test]
    fn test_pipeline_options_with_profiling() {
        let temp_dir = tempdir().expect("temp dir");
        let model_path = temp_dir.path().join("model.onnx");
        let options = ONNXPipelineOptions::cpu_optimized(model_path).with_profiling(true);
        assert!(options.enable_profiling);
    }

    #[test]
    fn test_pipeline_options_with_warmup_runs() {
        let temp_dir = tempdir().expect("temp dir");
        let model_path = temp_dir.path().join("model.onnx");
        let options = ONNXPipelineOptions::cpu_optimized(model_path).with_warmup_runs(10);
        assert_eq!(options.warmup_runs, 10);
    }

    // ── Real ONNX execution fixtures ──────────────────────────────────────────
    //
    // These build a tiny, real, binary `.onnx` protobuf in-test (the same
    // `y = Relu(x @ w1) @ w2` MLP trustformers-core's own interpreter tests use)
    // and drive it end to end through this *pipeline* module's public API, to
    // prove `ONNXModel`/`ONNXSessionOps` really call the CPU interpreter instead
    // of returning `Tensor::zeros`.

    fn onnx_fixture_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create fixture dir");
        dir
    }

    fn onnx_value_info(name: &str, dims: &[i64]) -> trustformers_core::export::onnx::ONNXValueInfo {
        use trustformers_core::export::onnx::{
            ONNXDataType, ONNXDimension, ONNXTensorShape, ONNXTensorType, ONNXTypeInfo,
            ONNXValueInfo,
        };
        ONNXValueInfo {
            name: name.to_string(),
            type_info: ONNXTypeInfo {
                tensor_type: ONNXTensorType {
                    elem_type: ONNXDataType::Float,
                    shape: ONNXTensorShape {
                        dims: dims.iter().map(|&d| ONNXDimension::Value(d)).collect(),
                    },
                },
            },
        }
    }

    fn onnx_initializer(
        name: &str,
        dims: &[i64],
        values: &[f32],
    ) -> trustformers_core::export::onnx::ONNXTensor {
        use trustformers_core::export::onnx::{ONNXDataType, ONNXTensor};
        ONNXTensor {
            name: name.to_string(),
            data_type: ONNXDataType::Float,
            dims: dims.to_vec(),
            raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    fn onnx_node(
        op: &str,
        name: &str,
        inputs: &[&str],
        outputs: &[&str],
    ) -> trustformers_core::export::onnx::ONNXNode {
        use trustformers_core::export::onnx::ONNXNode;
        ONNXNode {
            op_type: op.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attributes: HashMap::new(),
            name: name.to_string(),
        }
    }

    /// Fixture weights for the tiny MLP: `y = Relu(x @ w1) @ w2`, `x: [1, 64]`.
    fn onnx_fixture_weights() -> (Vec<f32>, Vec<f32>) {
        let w1: Vec<f32> = (0..64 * 8).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let w2: Vec<f32> = (0..8 * 4).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        (w1, w2)
    }

    fn write_real_onnx_mlp(path: &std::path::Path) {
        use trustformers_core::export::onnx::{ONNXExporter, ONNXGraph};
        use trustformers_core::export::ExportConfig;

        let (w1, w2) = onnx_fixture_weights();
        let graph = ONNXGraph {
            nodes: vec![
                onnx_node("MatMul", "mm1", &["x", "w1"], &["h"]),
                onnx_node("Relu", "relu", &["h"], &["a"]),
                onnx_node("MatMul", "mm2", &["a", "w2"], &["y"]),
            ],
            inputs: vec![onnx_value_info("x", &[1, 64])],
            outputs: vec![onnx_value_info("y", &[1, 4])],
            initializers: vec![
                onnx_initializer("w1", &[64, 8], &w1),
                onnx_initializer("w2", &[8, 4], &w2),
            ],
            name: "mlp".to_string(),
        };

        let exporter = ONNXExporter::new().with_opset_version(17);
        let model = exporter.wrap_graph(graph, &ExportConfig::default());
        exporter.export_graph(&model, path).expect("write fixture onnx model");
    }

    /// A graph whose single node uses an operator the CPU interpreter does not
    /// implement, to exercise the "fail loudly" path.
    fn write_unsupported_op_onnx(path: &std::path::Path) {
        use trustformers_core::export::onnx::{ONNXExporter, ONNXGraph};
        use trustformers_core::export::ExportConfig;

        let graph = ONNXGraph {
            nodes: vec![onnx_node("TotallyMadeUpVendorOp", "n1", &["x"], &["y"])],
            inputs: vec![onnx_value_info("x", &[1, 4])],
            outputs: vec![onnx_value_info("y", &[1, 4])],
            initializers: vec![],
            name: "unsupported".to_string(),
        };
        let exporter = ONNXExporter::new().with_opset_version(17);
        let model = exporter.wrap_graph(graph, &ExportConfig::default());
        exporter.export_graph(&model, path).expect("write fixture onnx model");
    }

    fn onnx_fixture_input() -> HashMap<String, Tensor> {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            Tensor::from_vec(values, &[1, 64]).expect("fixture input tensor"),
        );
        inputs
    }

    /// Minimal `Tokenizer` for driving the pipeline-level factory functions
    /// without touching the network: always emits 64 fixed token ids, matching
    /// the fixture MLP's `x: [1, 64]` input.
    #[derive(Clone)]
    struct FixedTokenizer;

    impl Tokenizer for FixedTokenizer {
        fn encode(&self, _text: &str) -> CoreResult<TokenizedInput> {
            Ok(TokenizedInput::new(vec![1u32; 64], vec![1u8; 64]))
        }
        fn encode_pair(&self, text: &str, _text2: &str) -> CoreResult<TokenizedInput> {
            self.encode(text)
        }
        fn decode(&self, ids: &[u32]) -> CoreResult<String> {
            Ok(format!("{ids:?}"))
        }
        fn vocab_size(&self) -> usize {
            100
        }
        fn get_vocab(&self) -> std::collections::HashMap<String, u32> {
            std::collections::HashMap::new()
        }
        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }
        fn id_to_token(&self, _id: u32) -> Option<String> {
            None
        }
    }

    /// Regression test for the `Tensor::zeros(&[1, 10])` mock: the pipeline's
    /// `ONNXModel::forward` must return the interpreter's real, deterministic
    /// answer for the fixture graph, matching a hand-computed reference.
    #[test]
    fn real_onnx_execution_matches_hand_computed_reference() {
        let dir = onnx_fixture_dir("trustformers_pipeline_onnx_run");
        let path = dir.join("mlp.onnx");
        write_real_onnx_mlp(&path);

        let config = ONNXBackendConfig::cpu_optimized(path);
        let model = ONNXModel::from_config(config).expect("load real onnx fixture");

        let inputs = onnx_fixture_input();
        let x = inputs["x"].to_vec_f32().expect("f32");
        let outputs = model.forward(inputs).expect("forward must execute the real graph");
        let y = outputs.get("y").expect("y output").to_vec_f32().expect("f32");

        let (w1, w2) = onnx_fixture_weights();
        let mut hidden = [0.0f32; 8];
        for (column, cell) in hidden.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (row, &value) in x.iter().enumerate() {
                acc += value * w1[row * 8 + column];
            }
            *cell = acc.max(0.0);
        }
        let mut expected = [0.0f32; 4];
        for (column, cell) in expected.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (row, &value) in hidden.iter().enumerate() {
                acc += value * w2[row * 4 + column];
            }
            *cell = acc;
        }

        assert_eq!(y.len(), 4);
        for (actual, expected) in y.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
        }
        // The old mock always answered zeros([1, 10]), which fails both the
        // shape and value assertions above.

        let _ = fs::remove_dir_all(&dir);
    }

    /// Regression test for `from_config` silently building a pipeline on top of
    /// operators the interpreter cannot run.
    #[test]
    fn from_config_refuses_a_graph_with_an_unsupported_operator() {
        let dir = onnx_fixture_dir("trustformers_pipeline_onnx_unsupported");
        let path = dir.join("bad.onnx");
        write_unsupported_op_onnx(&path);

        let config = ONNXBackendConfig::cpu_optimized(path);
        let result = ONNXModel::from_config(config);
        let message = match result {
            Ok(_) => panic!("a graph with an unimplemented operator must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            message.contains("TotallyMadeUpVendorOp"),
            "error should name the unsupported operator: {message}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// Regression test: the public pipeline factory must fail loudly (not
    /// build a classifier that will misbehave later) for the same reason.
    #[test]
    fn onnx_text_classification_pipeline_fails_loudly_for_unsupported_ops() {
        let dir = onnx_fixture_dir("trustformers_pipeline_onnx_factory_unsupported");
        let path = dir.join("bad.onnx");
        write_unsupported_op_onnx(&path);

        let result = onnx_text_classification_pipeline(&path, FixedTokenizer, None);
        assert!(
            result.is_err(),
            "the factory must refuse to build a pipeline it cannot run"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// Regression test for `Model::num_parameters` returning a hardcoded `0`.
    #[test]
    fn num_parameters_counts_real_initializer_elements() {
        let dir = onnx_fixture_dir("trustformers_pipeline_onnx_params");
        let path = dir.join("mlp.onnx");
        write_real_onnx_mlp(&path);

        let config = ONNXBackendConfig::cpu_optimized(path);
        let model = ONNXModel::from_config(config).expect("load real onnx fixture");

        assert_eq!(model.num_parameters(), 64 * 8 + 8 * 4);

        let _ = fs::remove_dir_all(&dir);
    }

    /// Regression test for the fixed `avg_latency_ms: 30.0, throughput: 33.0,
    /// memory_usage: 512MB` mock: values must come from real measurement.
    #[test]
    fn benchmark_measures_real_execution_not_fixed_mock_values() {
        let dir = onnx_fixture_dir("trustformers_pipeline_onnx_benchmark");
        let path = dir.join("mlp.onnx");
        write_real_onnx_mlp(&path);

        let config = ONNXBackendConfig::cpu_optimized(path);
        let model = ONNXModel::from_config(config).expect("load real onnx fixture");

        let result = model.benchmark(onnx_fixture_input(), 5).expect("benchmark");
        assert!(
            result.avg_latency_ms > 0.0,
            "latency must be a real positive measurement"
        );
        assert!(
            result.throughput > 0.0,
            "throughput must be a real positive measurement"
        );
        // Real weight bytes for this fixture: (64*8 + 8*4) f32 elements.
        assert_eq!(result.memory_usage, ((64 * 8 + 8 * 4) * 4) as u64);
        assert_ne!(
            (
                result.avg_latency_ms,
                result.throughput,
                result.memory_usage
            ),
            (30.0, 33.0, 512 * 1024 * 1024),
            "must not be the old hardcoded mock triple"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// Regression test for `memory_info` returning fixed 1GB/512MB/512MB.
    #[test]
    fn memory_info_reports_real_model_weight_bytes() {
        let dir = onnx_fixture_dir("trustformers_pipeline_onnx_memory");
        let path = dir.join("mlp.onnx");
        write_real_onnx_mlp(&path);

        let config = ONNXBackendConfig::cpu_optimized(path);
        let model = ONNXModel::from_config(config).expect("load real onnx fixture");

        let info = model.memory_info().expect("memory info");
        assert_eq!(info.model_memory_bytes, (64 * 8 + 8 * 4) * 4);
        assert!(info.total_memory_bytes > 0);
        assert!(info.available_memory_bytes <= info.total_memory_bytes);

        let _ = fs::remove_dir_all(&dir);
    }

    /// Regression test for `get_device_properties` always answering
    /// `{"type": "mock"}` regardless of the requested provider.
    #[test]
    fn get_device_properties_reports_real_cpu_info_not_mock() {
        let backend = ONNXRuntimeBackend::new(ONNXRuntimeConfig::default())
            .expect("runtime backend creation failed");

        let cpu_props =
            backend.get_device_properties(&ExecutionProvider::CPU).expect("cpu properties");
        assert_eq!(cpu_props.get("type").map(String::as_str), Some("cpu"));
        assert_ne!(cpu_props.get("type").map(String::as_str), Some("mock"));
        let cores: usize = cpu_props
            .get("logical_cores")
            .expect("logical_cores")
            .parse()
            .expect("numeric core count");
        assert!(cores > 0);

        let err = backend
            .get_device_properties(&ExecutionProvider::CUDA { device_id: Some(0) })
            .expect_err("no CUDA provider exists in this build");
        assert!(!err.to_string().is_empty());
    }

    // ── Session ops on a real fixture ─────────────────────────────────────────

    #[test]
    fn session_benchmark_and_memory_info_are_real() {
        let dir = onnx_fixture_dir("trustformers_pipeline_onnx_session_ops");
        let path = dir.join("mlp.onnx");
        write_real_onnx_mlp(&path);

        let backend = ONNXRuntimeBackend::new(ONNXRuntimeConfig::default())
            .expect("runtime backend creation failed");
        let session = backend.load_model(&path).expect("load real onnx fixture");

        let bench =
            ONNXSessionOps::benchmark(&session, onnx_fixture_input(), 5, 2).expect("benchmark");
        assert!(bench.avg_latency_ms > 0.0, "avg latency must be positive");
        assert!(bench.throughput > 0.0, "throughput must be positive");
        assert_eq!(bench.memory_usage, ((64 * 8 + 8 * 4) * 4) as u64);

        let info = ONNXSessionOps::get_memory_info(&session).expect("memory info");
        assert_eq!(info.model_memory_bytes, (64 * 8 + 8 * 4) * 4);
        assert!(info.total_memory_bytes > 0);
        assert!(info.available_memory_bytes <= info.total_memory_bytes);

        let _ = fs::remove_dir_all(&dir);
    }

    // ── Benchmark results struct ──────────────────────────────────────────────

    #[test]
    fn test_benchmark_results_struct_fields() {
        let b = BenchmarkResults {
            avg_latency_ms: 25.5,
            throughput: 40.0,
            memory_usage: 256 * 1024 * 1024,
        };
        assert!(b.avg_latency_ms > 0.0);
        assert!(b.throughput > 0.0);
        assert!(b.memory_usage > 0);
    }

    // ── GPU options device field ──────────────────────────────────────────────

    #[test]
    fn test_gpu_options_device_id_zero() {
        let temp_dir = tempdir().expect("temp dir");
        let model_path = temp_dir.path().join("model.onnx");
        let opts = ONNXPipelineOptions::gpu_optimized(model_path, Some(0));
        assert!(matches!(opts.base_options.device, Some(Device::Gpu(0))));
    }

    #[test]
    fn test_cpu_options_no_gpu_device() {
        let temp_dir = tempdir().expect("temp dir");
        let model_path = temp_dir.path().join("model.onnx");
        let opts = ONNXPipelineOptions::cpu_optimized(model_path);
        // CPU config should not force a GPU device
        assert!(
            !matches!(opts.base_options.device, Some(Device::Gpu(_))),
            "CPU-optimised options should not set a GPU device"
        );
    }
}
