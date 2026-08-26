// Core ML Pipeline Backend Integration for TrustformeRS
// Provides native Core ML inference optimized for iOS and macOS deployment
//
// # Why prediction always fails
//
// Running an actual `.mlmodel`/`.mlpackage` requires Apple's Core ML framework,
// reachable from Rust only through `objc2`/`objc2-metal`-style bindings to the
// Objective-C runtime. No such binding is linked into this crate (it would be a
// macOS-only, non-pure-source dependency, and this workspace keeps that class of
// dependency feature-gated and off by default; see `trustformers-core`'s `metal`
// feature). `trustformers-core::export::coreml` is a *topology writer* for
// exporting a trained model's parameters into Core ML's protobuf shape — it
// contains no execution engine, and even it refuses to fabricate the topology it
// cannot recover (see its module docs).
//
// An earlier revision of this file `load_model`ed unconditionally and had
// `predict` return `sin(i * 0.001 + input[0])` dressed up as classifier logits,
// while `detect_device_capabilities` hardcoded Apple Silicon numbers regardless
// of the host. Neither survives: `load_model`/`predict` return a structured
// [`TrustformersError::FeatureUnavailable`] naming the missing framework, and
// capability detection reports only what `cfg!` / `num_cpus` / `sysinfo` can
// really observe about the host running the build.

use crate::core::traits::Tokenizer;
use crate::error::{Result, TrustformersError};
use crate::pipeline::{ClassificationOutput, GenerationOutput, Pipeline, PipelineOutput};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use trustformers_core::tensor::Tensor;

/// Human-readable explanation attached to every Core ML "unavailable" error.
const COREML_UNAVAILABLE_REASON: &str =
    "Core ML inference requires Apple's Core ML framework via objc2-style Objective-C \
     bindings, which this pure-Rust build does not link. No pure-Rust Core ML execution \
     engine exists in trustformers-core to fall back to.";

// Core ML backend types
#[derive(Debug, Clone)]
pub struct CoreMLBackend {
    model: Option<CoreMLModel>,
    config: CoreMLBackendConfig,
    device_capabilities: CoreMLDeviceCapabilities,
}

#[derive(Debug, Clone)]
pub struct CoreMLModel;

#[derive(Debug, Clone)]
pub struct CoreMLPrediction;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoreMLComputeUnit {
    /// CPU execution only
    CPUOnly,
    /// CPU and GPU execution
    CPUAndGPU,
    /// All available compute units (CPU, GPU, Neural Engine)
    All,
    /// Neural Engine only (Apple Silicon)
    NeuralEngineOnly,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoreMLPrecision {
    /// 32-bit floating point
    Float32,
    /// 16-bit floating point
    Float16,
    /// 8-bit integer quantization
    Int8,
    /// Automatic precision selection
    Auto,
}

#[derive(Debug, Clone, Copy)]
pub enum CoreMLOptimizationLevel {
    /// No optimizations
    None,
    /// Basic optimizations
    Basic,
    /// Standard optimizations
    Standard,
    /// Aggressive optimizations
    Aggressive,
}

#[derive(Debug, Clone)]
pub struct CoreMLBackendConfig {
    /// Path to Core ML model (.mlmodel or .mlpackage)
    pub model_path: PathBuf,
    /// Preferred compute unit
    pub compute_unit: CoreMLComputeUnit,
    /// Model precision
    pub precision: CoreMLPrecision,
    /// Optimization level
    pub optimization_level: CoreMLOptimizationLevel,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Enable Neural Engine (Apple Silicon only)
    pub enable_neural_engine: bool,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Allow low precision inference
    pub allow_low_precision: bool,
    /// Enable model compilation
    pub enable_compilation: bool,
    /// Compiled model cache directory
    pub cache_directory: Option<PathBuf>,
    /// Model timeout in seconds
    pub timeout_seconds: f64,
}

impl Default for CoreMLBackendConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            compute_unit: CoreMLComputeUnit::All,
            precision: CoreMLPrecision::Auto,
            optimization_level: CoreMLOptimizationLevel::Standard,
            max_batch_size: 1,
            enable_neural_engine: true,
            enable_gpu: true,
            allow_low_precision: true,
            enable_compilation: true,
            cache_directory: None,
            timeout_seconds: 30.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoreMLDeviceCapabilities {
    pub has_neural_engine: bool,
    pub has_gpu: bool,
    pub supports_float16: bool,
    pub supports_int8: bool,
    pub max_memory_mb: usize,
    pub cpu_core_count: usize,
    pub gpu_core_count: Option<usize>,
    pub neural_engine_core_count: Option<usize>,
}

impl CoreMLBackend {
    /// Create a new Core ML backend instance
    pub fn new(config: CoreMLBackendConfig) -> Result<Self> {
        let device_capabilities = Self::detect_device_capabilities();

        Ok(Self {
            model: None,
            config,
            device_capabilities,
        })
    }

    /// Detect device capabilities.
    ///
    /// Only facts this build can actually establish are reported:
    /// - `has_neural_engine` / `has_gpu` / `supports_float16` / `supports_int8`
    ///   are derived from `cfg!(target_os, target_arch)`. Every shipped Apple
    ///   Silicon Mac (aarch64 macOS) has a Neural Engine and GPU; Core ML's
    ///   float16/int8 execution modes exist only on macOS. On any other target
    ///   these are honestly `false` rather than the old unconditional `true`.
    /// - `cpu_core_count` comes from `num_cpus::get()`.
    /// - `max_memory_mb` comes from `sysinfo`'s real host memory reading.
    /// - `gpu_core_count` / `neural_engine_core_count` are genuinely
    ///   unknowable without private Apple APIs this crate does not call, so
    ///   they are `None` rather than a fabricated guess.
    fn detect_device_capabilities() -> CoreMLDeviceCapabilities {
        let is_apple_silicon = cfg!(target_os = "macos") && cfg!(target_arch = "aarch64");
        let is_macos = cfg!(target_os = "macos");

        let mut system = sysinfo::System::new();
        system.refresh_memory();
        let max_memory_mb = (system.total_memory() / (1024 * 1024)) as usize;

        CoreMLDeviceCapabilities {
            has_neural_engine: is_apple_silicon,
            has_gpu: is_macos,
            supports_float16: is_macos,
            supports_int8: is_macos,
            max_memory_mb,
            cpu_core_count: num_cpus::get(),
            gpu_core_count: None,
            neural_engine_core_count: None,
        }
    }

    /// Load and compile a Core ML model.
    ///
    /// Always fails: see the module docs for why no build of this crate can
    /// actually load a `.mlmodel`/`.mlpackage`.
    pub fn load_model(&mut self, _model_path: &Path) -> Result<()> {
        Err(TrustformersError::feature_unavailable(
            COREML_UNAVAILABLE_REASON,
            "coreml_inference",
        ))
    }

    /// Run inference with Core ML.
    ///
    /// Always fails, honestly: there is no path to real Core ML execution in
    /// this build, and this method must never again return `sin(...)` dressed
    /// up as logits. `self.model` can never legitimately be `Some` (only
    /// `load_model` sets it, and `load_model` always errors), so both branches
    /// report the same real reason.
    pub fn predict(&self, _inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        Err(TrustformersError::feature_unavailable(
            COREML_UNAVAILABLE_REASON,
            "coreml_inference",
        ))
    }

    /// Get model metadata
    pub fn model_description(&self) -> Option<CoreMLModelDescription> {
        if self.model.is_some() {
            Some(CoreMLModelDescription {
                name: "TrustformeRS Model".to_string(),
                description: "Core ML model for transformer inference".to_string(),
                version: "1.0.0".to_string(),
                author: "TrustformeRS".to_string(),
                input_names: vec!["input_ids".to_string(), "attention_mask".to_string()],
                output_names: vec!["logits".to_string()],
                compute_units: self.config.compute_unit,
            })
        } else {
            None
        }
    }

    /// Get device capabilities
    pub fn device_capabilities(&self) -> &CoreMLDeviceCapabilities {
        &self.device_capabilities
    }

    /// Optimize model for target device
    pub fn optimize_for_device(&mut self) -> Result<()> {
        // In real implementation would:
        // 1. Analyze model architecture
        // 2. Apply device-specific optimizations
        // 3. Configure compute unit preferences
        // 4. Set precision preferences
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CoreMLModelDescription {
    pub name: String,
    pub description: String,
    pub version: String,
    pub author: String,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub compute_units: CoreMLComputeUnit,
}

/// Core ML Text Classification Pipeline
pub struct CoreMLTextClassificationPipeline<T: Tokenizer> {
    tokenizer: T,
    backend: CoreMLBackend,
    config: CoreMLBackendConfig,
}

impl<T: Tokenizer + Clone> CoreMLTextClassificationPipeline<T> {
    /// Create a new Core ML text classification pipeline
    pub fn new(tokenizer: T, config: CoreMLBackendConfig) -> Result<Self> {
        let mut backend = CoreMLBackend::new(config.clone())?;

        // Load and optimize Core ML model
        backend.load_model(&config.model_path)?;
        backend.optimize_for_device()?;

        Ok(Self {
            tokenizer,
            backend,
            config,
        })
    }

    /// Get model description
    pub fn model_description(&self) -> Option<CoreMLModelDescription> {
        self.backend.model_description()
    }

    /// Get device capabilities
    pub fn device_capabilities(&self) -> &CoreMLDeviceCapabilities {
        self.backend.device_capabilities()
    }

    /// Get the Core ML backend configuration this pipeline was built with.
    pub fn config(&self) -> &CoreMLBackendConfig {
        &self.config
    }
}

impl<T: Tokenizer + Clone> Pipeline for CoreMLTextClassificationPipeline<T> {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        // Tokenize input
        let tokenized = self.tokenizer.encode(&input)?;
        let input_ids = tokenized.input_ids;
        let attention_mask = tokenized.attention_mask;

        // Prepare inputs for Core ML
        let mut inputs = HashMap::new();
        inputs.insert(
            "input_ids".to_string(),
            Tensor::from_vec(
                input_ids.iter().map(|&x| x as f32).collect(),
                &[1, input_ids.len()],
            )?,
        );
        inputs.insert(
            "attention_mask".to_string(),
            Tensor::from_vec(
                attention_mask.iter().map(|&x| x as f32).collect(),
                &[1, attention_mask.len()],
            )?,
        );

        // Run Core ML inference
        let outputs = self.backend.predict(inputs)?;

        // Process outputs
        if let Some(output_tensor) = outputs.get("output") {
            let logits = output_tensor.data()?;

            // Apply softmax
            let exp_logits: Vec<f32> = logits.iter().map(|x| x.exp()).collect();
            let sum_exp: f32 = exp_logits.iter().sum();
            let probabilities: Vec<f32> = exp_logits.iter().map(|x| x / sum_exp).collect();

            // Create classification results
            let mut results = Vec::new();
            for (i, &prob) in probabilities.iter().enumerate().take(5) {
                // Top 5 results
                results.push(ClassificationOutput {
                    label: format!("LABEL_{}", i),
                    score: prob,
                });
            }

            // Sort by score (descending)
            results
                .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

            Ok(PipelineOutput::Classification(results))
        } else {
            Err(TrustformersError::invalid_input_simple(
                "No output from Core ML model".to_string(),
            ))
        }
    }
}

/// Core ML Text Generation Pipeline
pub struct CoreMLTextGenerationPipeline<T: Tokenizer> {
    tokenizer: T,
    backend: CoreMLBackend,
    config: CoreMLBackendConfig,
}

impl<T: Tokenizer + Clone> CoreMLTextGenerationPipeline<T> {
    /// Create a new Core ML text generation pipeline
    pub fn new(tokenizer: T, config: CoreMLBackendConfig) -> Result<Self> {
        let mut backend = CoreMLBackend::new(config.clone())?;

        // Load and optimize Core ML model
        backend.load_model(&config.model_path)?;
        backend.optimize_for_device()?;

        Ok(Self {
            tokenizer,
            backend,
            config,
        })
    }

    /// Get the Core ML backend configuration this pipeline was built with.
    pub fn config(&self) -> &CoreMLBackendConfig {
        &self.config
    }
}

impl<T: Tokenizer + Clone> Pipeline for CoreMLTextGenerationPipeline<T> {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        // Tokenize input
        let tokenized = self.tokenizer.encode(&input)?;
        let input_ids = tokenized.input_ids;

        // Prepare inputs for Core ML
        let mut inputs = HashMap::new();
        inputs.insert(
            "input_ids".to_string(),
            Tensor::from_vec(
                input_ids.iter().map(|&x| x as f32).collect(),
                &[1, input_ids.len()],
            )?,
        );

        // Run Core ML inference
        let outputs = self.backend.predict(inputs)?;

        // Process generation output
        if let Some(output_tensor) = outputs.get("output") {
            let logits = output_tensor.data()?;

            // Simple greedy decoding
            let next_token_id = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, _)| index as u32)
                .unwrap_or(0);

            // Decode generated token
            let generated_text = self.tokenizer.decode(&[next_token_id])?;

            Ok(PipelineOutput::Generation(GenerationOutput {
                generated_text: input + &generated_text,
                sequences: Some(vec![vec![next_token_id]]),
                scores: Some(logits.clone()),
            }))
        } else {
            Err(TrustformersError::invalid_input_simple(
                "No output from Core ML model".to_string(),
            ))
        }
    }
}

/// Configuration presets for different deployment scenarios
impl CoreMLBackendConfig {
    /// Configuration optimized for iOS devices
    pub fn for_ios() -> Self {
        Self {
            compute_unit: CoreMLComputeUnit::All,
            precision: CoreMLPrecision::Float16,
            optimization_level: CoreMLOptimizationLevel::Aggressive,
            max_batch_size: 1,
            enable_neural_engine: true,
            enable_gpu: true,
            allow_low_precision: true,
            enable_compilation: true,
            timeout_seconds: 10.0,
            ..Default::default()
        }
    }

    /// Configuration optimized for macOS
    pub fn for_macos() -> Self {
        Self {
            compute_unit: CoreMLComputeUnit::All,
            precision: CoreMLPrecision::Float32,
            optimization_level: CoreMLOptimizationLevel::Standard,
            max_batch_size: 4,
            enable_neural_engine: true,
            enable_gpu: true,
            allow_low_precision: false,
            enable_compilation: true,
            timeout_seconds: 30.0,
            ..Default::default()
        }
    }

    /// Configuration for maximum performance (may sacrifice accuracy)
    pub fn for_maximum_performance() -> Self {
        Self {
            compute_unit: CoreMLComputeUnit::NeuralEngineOnly,
            precision: CoreMLPrecision::Int8,
            optimization_level: CoreMLOptimizationLevel::Aggressive,
            max_batch_size: 1,
            enable_neural_engine: true,
            enable_gpu: false,
            allow_low_precision: true,
            enable_compilation: true,
            timeout_seconds: 5.0,
            ..Default::default()
        }
    }

    /// Configuration for best accuracy (may sacrifice performance)
    pub fn for_best_accuracy() -> Self {
        Self {
            compute_unit: CoreMLComputeUnit::CPUOnly,
            precision: CoreMLPrecision::Float32,
            optimization_level: CoreMLOptimizationLevel::None,
            max_batch_size: 1,
            enable_neural_engine: false,
            enable_gpu: false,
            allow_low_precision: false,
            enable_compilation: false,
            timeout_seconds: 60.0,
            ..Default::default()
        }
    }
}

/// Factory functions for creating Core ML pipelines
pub fn create_coreml_text_classification_pipeline<T: Tokenizer + Clone>(
    tokenizer: T,
    config: Option<CoreMLBackendConfig>,
) -> Result<CoreMLTextClassificationPipeline<T>> {
    let config = config.unwrap_or_else(CoreMLBackendConfig::for_ios);
    CoreMLTextClassificationPipeline::new(tokenizer, config)
}

pub fn create_coreml_text_generation_pipeline<T: Tokenizer + Clone>(
    tokenizer: T,
    config: Option<CoreMLBackendConfig>,
) -> Result<CoreMLTextGenerationPipeline<T>> {
    let config = config.unwrap_or_else(CoreMLBackendConfig::for_ios);
    CoreMLTextGenerationPipeline::new(tokenizer, config)
}

/// Utility functions for Core ML model conversion.
///
/// Every method here used to return `Ok(())` without writing an output file —
/// callers were told their model had been converted when nothing had happened.
/// None of them can honestly succeed: writing a real `.mlmodel`/`.mlpackage`
/// needs the same Core ML protobuf writer
/// `trustformers_core::export::coreml::CoreMLExporter` already refuses to drive
/// (it can recover a model's parameters but not its topology; see that module's
/// docs). Use the GGUF or GGML exporters for a real, honest weight container
/// instead.
pub struct CoreMLModelConverter;

impl CoreMLModelConverter {
    /// Convert a PyTorch model to Core ML format. Always fails; see the type
    /// docs for why no build of this crate can produce a real `.mlmodel`.
    pub fn from_pytorch(
        _model_path: &Path,
        _output_path: &Path,
        _input_shapes: HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        Err(TrustformersError::feature_unavailable(
            "PyTorch -> Core ML conversion needs both a PyTorch model reader and the Core ML \
             protobuf writer; neither is implemented in this pure-Rust build.",
            "coreml_conversion",
        ))
    }

    /// Convert an ONNX model to Core ML format. Always fails.
    pub fn from_onnx(_model_path: &Path, _output_path: &Path) -> Result<()> {
        Err(TrustformersError::feature_unavailable(
            "ONNX -> Core ML conversion needs the Core ML protobuf writer, which is not \
             implemented in this pure-Rust build (trustformers-core's CoreML exporter writes \
             parameters only, never topology).",
            "coreml_conversion",
        ))
    }

    /// Convert a TensorFlow model to Core ML format. Always fails.
    pub fn from_tensorflow(_model_path: &Path, _output_path: &Path) -> Result<()> {
        Err(TrustformersError::feature_unavailable(
            "TensorFlow -> Core ML conversion needs both a TensorFlow model reader and the \
             Core ML protobuf writer; neither is implemented in this pure-Rust build.",
            "coreml_conversion",
        ))
    }

    /// Optimize a Core ML model for a specific device. Always fails: there is
    /// no Core ML model reader/writer in this build to optimize with.
    pub fn optimize_for_device(
        _model_path: &Path,
        _output_path: &Path,
        _target_device: CoreMLComputeUnit,
    ) -> Result<()> {
        Err(TrustformersError::feature_unavailable(
            "Core ML model optimization needs the Core ML protobuf reader/writer, which is \
             not implemented in this pure-Rust build.",
            "coreml_conversion",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coreml_backend_creation() {
        let config = CoreMLBackendConfig::for_ios();
        let backend = CoreMLBackend::new(config);
        assert!(backend.is_ok());
    }

    /// Regression test for hardcoded `has_neural_engine: true, // Assume Apple
    /// Silicon`: capabilities must reflect the *actual* host, not an assumption.
    #[test]
    fn test_device_capabilities() {
        let config = CoreMLBackendConfig::for_ios();
        let backend = CoreMLBackend::new(config).expect("operation failed in test");
        let capabilities = backend.device_capabilities();

        let is_apple_silicon = cfg!(target_os = "macos") && cfg!(target_arch = "aarch64");
        let is_macos = cfg!(target_os = "macos");
        assert_eq!(capabilities.has_neural_engine, is_apple_silicon);
        assert_eq!(capabilities.has_gpu, is_macos);
        assert_eq!(capabilities.supports_float16, is_macos);
        assert_eq!(capabilities.cpu_core_count, num_cpus::get());
        assert!(capabilities.cpu_core_count > 0);
    }

    #[test]
    fn test_configuration_presets() {
        let ios_config = CoreMLBackendConfig::for_ios();
        assert_eq!(ios_config.precision, CoreMLPrecision::Float16);
        assert!(ios_config.enable_neural_engine);
        assert_eq!(ios_config.max_batch_size, 1);

        let macos_config = CoreMLBackendConfig::for_macos();
        assert_eq!(macos_config.precision, CoreMLPrecision::Float32);
        assert_eq!(macos_config.max_batch_size, 4);

        let performance_config = CoreMLBackendConfig::for_maximum_performance();
        assert_eq!(
            performance_config.compute_unit,
            CoreMLComputeUnit::NeuralEngineOnly
        );
        assert_eq!(performance_config.precision, CoreMLPrecision::Int8);

        let accuracy_config = CoreMLBackendConfig::for_best_accuracy();
        assert_eq!(accuracy_config.compute_unit, CoreMLComputeUnit::CPUOnly);
        assert_eq!(accuracy_config.precision, CoreMLPrecision::Float32);
    }

    /// Regression test: converters used to return `Ok(())` while writing no
    /// file at all. They must now honestly refuse.
    #[test]
    fn test_model_converter() {
        let input_path = Path::new("input.pt");
        let output_path = Path::new("output.mlmodel");
        let input_shapes = HashMap::new();

        let result = CoreMLModelConverter::from_pytorch(input_path, output_path, input_shapes);
        assert!(
            result.is_err(),
            "no PyTorch -> Core ML conversion exists in this build"
        );
        assert!(!output_path.exists());
    }

    // ── Default config ────────────────────────────────────────────────────────

    #[test]
    fn test_default_config_fields() {
        let cfg = CoreMLBackendConfig::default();
        assert_eq!(cfg.compute_unit, CoreMLComputeUnit::All);
        assert_eq!(cfg.precision, CoreMLPrecision::Auto);
        assert_eq!(cfg.max_batch_size, 1);
        assert!(cfg.enable_neural_engine);
        assert!(cfg.enable_gpu);
        assert!(cfg.allow_low_precision);
        assert!(cfg.enable_compilation);
        assert!(cfg.cache_directory.is_none());
    }

    #[test]
    fn test_default_config_timeout_positive() {
        let cfg = CoreMLBackendConfig::default();
        assert!(cfg.timeout_seconds > 0.0);
    }

    // ── Compute unit variants ─────────────────────────────────────────────────

    #[test]
    fn test_compute_unit_variants_distinct() {
        assert_ne!(CoreMLComputeUnit::CPUOnly, CoreMLComputeUnit::All);
        assert_ne!(
            CoreMLComputeUnit::CPUAndGPU,
            CoreMLComputeUnit::NeuralEngineOnly
        );
    }

    // ── Precision variants ────────────────────────────────────────────────────

    #[test]
    fn test_precision_variants_distinct() {
        assert_ne!(CoreMLPrecision::Float32, CoreMLPrecision::Float16);
        assert_ne!(CoreMLPrecision::Int8, CoreMLPrecision::Auto);
    }

    // ── Model load and description ────────────────────────────────────────────

    #[test]
    fn test_model_description_none_before_load() {
        let config = CoreMLBackendConfig::for_ios();
        let backend = CoreMLBackend::new(config).expect("backend creation failed");
        // model not yet loaded — must be None
        assert!(backend.model_description().is_none());
    }

    /// Regression test: `load_model` used to unconditionally set
    /// `self.model = Some(CoreMLModel)` and return `Ok(())`, so
    /// `model_description()` would report an invented model afterwards. It
    /// must now fail loudly instead, and the description must stay `None`.
    #[test]
    fn test_load_model_fails_and_description_stays_none() {
        let config = CoreMLBackendConfig::for_ios();
        let mut backend = CoreMLBackend::new(config).expect("backend creation failed");
        let dummy = std::env::temp_dir().join("test.mlpackage");

        let err = backend
            .load_model(&dummy)
            .expect_err("no Core ML runtime is linked into this build");
        assert!(!err.to_string().is_empty());
        assert!(
            backend.model_description().is_none(),
            "a failed load must never leave a fabricated model description behind"
        );
    }

    // ── Predict before load returns error ────────────────────────────────────

    #[test]
    fn test_predict_without_model_returns_error() {
        let config = CoreMLBackendConfig::default();
        let backend = CoreMLBackend::new(config).expect("backend creation failed");
        // Model is None (not loaded)
        let result = backend.predict(HashMap::new());
        assert!(
            result.is_err(),
            "predict must fail when model is not loaded"
        );
    }

    // ── Predict never fabricates output ───────────────────────────────────────

    /// Regression test for `predict` returning
    /// `(i as f32 * 0.001 + input[0]).sin()` dressed up as 1000-way classifier
    /// logits. `load_model` cannot succeed either, so `predict` must fail even
    /// after attempting to load — and it must never synthesize any tensor.
    #[test]
    fn test_predict_never_fabricates_output() {
        let config = CoreMLBackendConfig::for_ios();
        let mut backend = CoreMLBackend::new(config).expect("backend creation failed");
        let _ = backend.load_model(&std::env::temp_dir().join("m.mlmodel"));

        let input_tensor =
            trustformers_core::tensor::Tensor::zeros(&[1, 10]).expect("tensor creation ok");
        let mut inputs = HashMap::new();
        inputs.insert("input_ids".to_string(), input_tensor);
        let err = backend
            .predict(inputs)
            .expect_err("no Core ML runtime is linked into this build");
        assert!(!err.to_string().is_empty());
    }

    // ── Optimize for device ───────────────────────────────────────────────────

    #[test]
    fn test_optimize_for_device_ok() {
        let config = CoreMLBackendConfig::for_macos();
        let mut backend = CoreMLBackend::new(config).expect("backend creation failed");
        let result = backend.optimize_for_device();
        assert!(
            result.is_ok(),
            "optimize_for_device mock should always succeed"
        );
    }

    // ── Device capabilities detailed ──────────────────────────────────────────

    #[test]
    fn test_device_capabilities_memory_positive() {
        let config = CoreMLBackendConfig::for_ios();
        let backend = CoreMLBackend::new(config).expect("backend ok");
        let cap = backend.device_capabilities();
        assert!(cap.max_memory_mb > 0, "max_memory_mb must be positive");
    }

    /// Regression test for the hardcoded `gpu_core_count: Some(8)`: this crate
    /// has no way to really learn a Mac's GPU core count without private Apple
    /// APIs, so it must honestly report `None` rather than guessing.
    #[test]
    fn test_device_capabilities_gpu_core_count() {
        let config = CoreMLBackendConfig::for_ios();
        let backend = CoreMLBackend::new(config).expect("backend ok");
        let cap = backend.device_capabilities();
        assert!(
            cap.gpu_core_count.is_none(),
            "gpu core count is not obtainable in this build and must not be fabricated"
        );
    }

    /// Regression test for the hardcoded `neural_engine_core_count: Some(16)`.
    #[test]
    fn test_device_capabilities_neural_engine_core_count() {
        let config = CoreMLBackendConfig::for_ios();
        let backend = CoreMLBackend::new(config).expect("backend ok");
        let cap = backend.device_capabilities();
        assert!(
            cap.neural_engine_core_count.is_none(),
            "neural engine core count is not obtainable in this build and must not be fabricated"
        );
    }

    // ── iOS preset timeout ────────────────────────────────────────────────────

    #[test]
    fn test_ios_config_timeout_10s() {
        let cfg = CoreMLBackendConfig::for_ios();
        assert!((cfg.timeout_seconds - 10.0).abs() < 1e-6);
    }

    // ── Best accuracy preset disables compilation ─────────────────────────────

    #[test]
    fn test_best_accuracy_config_no_compilation() {
        let cfg = CoreMLBackendConfig::for_best_accuracy();
        assert!(
            !cfg.enable_compilation,
            "best_accuracy should disable compilation"
        );
        assert!(
            !cfg.enable_neural_engine,
            "best_accuracy should disable neural engine"
        );
        assert!(!cfg.enable_gpu, "best_accuracy should disable GPU");
    }

    // ── Maximum performance preset  ───────────────────────────────────────────

    #[test]
    fn test_maximum_performance_short_timeout() {
        let cfg = CoreMLBackendConfig::for_maximum_performance();
        assert!(
            cfg.timeout_seconds <= 10.0,
            "maximum performance config should have a tight timeout"
        );
    }

    // ── Converter additional methods ──────────────────────────────────────────

    /// Regression test for `from_onnx` returning `Ok(())` while writing no
    /// `.mlmodel` file at all.
    #[test]
    fn test_converter_from_onnx_is_unavailable() {
        let output_path = Path::new("model.mlmodel");
        let result = CoreMLModelConverter::from_onnx(Path::new("model.onnx"), output_path);
        assert!(result.is_err());
        assert!(!output_path.exists());
    }

    #[test]
    fn test_converter_from_tensorflow_is_unavailable() {
        let output_path = Path::new("model.mlmodel");
        let result = CoreMLModelConverter::from_tensorflow(Path::new("model.pb"), output_path);
        assert!(result.is_err());
        assert!(!output_path.exists());
    }

    #[test]
    fn test_converter_optimize_for_device_is_unavailable() {
        let output_path = Path::new("model_opt.mlmodel");
        let result = CoreMLModelConverter::optimize_for_device(
            Path::new("model.mlmodel"),
            output_path,
            CoreMLComputeUnit::All,
        );
        assert!(result.is_err());
        assert!(!output_path.exists());
    }

    // ── MacOS preset ──────────────────────────────────────────────────────────

    #[test]
    fn test_macos_config_no_low_precision() {
        let cfg = CoreMLBackendConfig::for_macos();
        assert!(
            !cfg.allow_low_precision,
            "macOS accuracy preset disallows low precision"
        );
    }

    #[test]
    fn test_macos_config_timeout_30s() {
        let cfg = CoreMLBackendConfig::for_macos();
        assert!((cfg.timeout_seconds - 30.0).abs() < 1e-6);
    }

    /// Regression test for the hardcoded `max_memory_mb: 8192`: the reported
    /// value must track the host's real memory, not a fixed constant.
    #[test]
    fn test_device_capabilities_memory_matches_real_host() {
        let config = CoreMLBackendConfig::for_ios();
        let backend = CoreMLBackend::new(config).expect("backend ok");
        let cap = backend.device_capabilities();

        let mut system = sysinfo::System::new();
        system.refresh_memory();
        let expected_mb = (system.total_memory() / (1024 * 1024)) as usize;

        // Allow a little drift: the host's memory reading can shift by a few MB
        // between the two `refresh_memory()` calls under real-world jitter.
        let diff = cap.max_memory_mb.abs_diff(expected_mb);
        assert!(
            diff <= expected_mb / 100 + 8,
            "reported {} MB, host reports {} MB",
            cap.max_memory_mb,
            expected_mb
        );
    }

    /// A minimal `Tokenizer` for exercising the pipeline-level factory
    /// functions without any model/tokenizer files.
    #[derive(Clone)]
    struct StubTokenizer;

    impl Tokenizer for StubTokenizer {
        fn encode(
            &self,
            _text: &str,
        ) -> crate::core::errors::Result<crate::core::traits::TokenizedInput> {
            Ok(crate::core::traits::TokenizedInput::new(
                vec![1, 2, 3],
                vec![1, 1, 1],
            ))
        }
        fn encode_pair(
            &self,
            text: &str,
            _text2: &str,
        ) -> crate::core::errors::Result<crate::core::traits::TokenizedInput> {
            self.encode(text)
        }
        fn decode(&self, ids: &[u32]) -> crate::core::errors::Result<String> {
            Ok(format!("{ids:?}"))
        }
        fn vocab_size(&self) -> usize {
            10
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

    /// Regression test: the public factory functions must fail loudly, not
    /// build a pipeline whose `__call__` only fails later (or worse, whose
    /// `predict` used to succeed with fabricated logits).
    #[test]
    fn factory_functions_fail_loudly_instead_of_building_a_dead_pipeline() {
        let classification = create_coreml_text_classification_pipeline(
            StubTokenizer,
            Some(CoreMLBackendConfig::for_ios()),
        );
        assert!(classification.is_err());

        let generation = create_coreml_text_generation_pipeline(
            StubTokenizer,
            Some(CoreMLBackendConfig::for_ios()),
        );
        assert!(generation.is_err());
    }
}
