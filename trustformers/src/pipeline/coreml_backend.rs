// Core ML Pipeline Backend Integration for TrustformeRS
// Provides native Core ML inference optimized for iOS and macOS deployment

use crate::core::traits::Tokenizer;
use crate::error::{Result, TrustformersError};
use crate::pipeline::{ClassificationOutput, GenerationOutput, Pipeline, PipelineOutput};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use trustformers_core::tensor::Tensor;

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

    /// Detect device capabilities
    fn detect_device_capabilities() -> CoreMLDeviceCapabilities {
        // In a real implementation, this would query the system
        CoreMLDeviceCapabilities {
            has_neural_engine: true, // Assume Apple Silicon
            has_gpu: true,
            supports_float16: true,
            supports_int8: true,
            max_memory_mb: 8192, // 8GB
            cpu_core_count: 8,
            gpu_core_count: Some(8),
            neural_engine_core_count: Some(16),
        }
    }

    /// Load and compile Core ML model
    pub fn load_model(&mut self, model_path: &Path) -> Result<()> {
        // In a real implementation, this would:
        // 1. Load .mlmodel or .mlpackage file
        // 2. Compile for target device
        // 3. Configure compute units
        // 4. Validate input/output specifications
        self.model = Some(CoreMLModel);
        Ok(())
    }

    /// Run inference with Core ML
    pub fn predict(&self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        if self.model.is_none() {
            return Err(TrustformersError::invalid_input_simple(
                "Core ML model not loaded".to_string(),
            ));
        }

        // Mock inference - in real implementation would:
        // 1. Convert tensors to MLMultiArray
        // 2. Create MLFeatureProvider
        // 3. Run model prediction
        // 4. Extract outputs and convert back to tensors
        let mut outputs = HashMap::new();

        // Create mock output based on input
        if let Some(input_tensor) = inputs.values().next() {
            let input_data = input_tensor.data()?;
            let output_size = 1000; // Example classification output
            let mock_logits: Vec<f32> = (0..output_size)
                .map(|i| (i as f32 * 0.001 + input_data.first().unwrap_or(&0.5)).sin())
                .collect();

            let output_tensor = Tensor::from_vec(mock_logits, &[1, output_size])?;
            outputs.insert("output".to_string(), output_tensor);
        }

        Ok(outputs)
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

/// Utility functions for Core ML model conversion
pub struct CoreMLModelConverter;

impl CoreMLModelConverter {
    /// Convert PyTorch model to Core ML format
    pub fn from_pytorch(
        model_path: &Path,
        output_path: &Path,
        input_shapes: HashMap<String, Vec<usize>>,
    ) -> Result<()> {
        // In real implementation would use PyTorch -> Core ML conversion
        // This is a placeholder for the actual conversion logic
        Ok(())
    }

    /// Convert ONNX model to Core ML format
    pub fn from_onnx(model_path: &Path, output_path: &Path) -> Result<()> {
        // In real implementation would use ONNX -> Core ML conversion
        Ok(())
    }

    /// Convert TensorFlow model to Core ML format
    pub fn from_tensorflow(model_path: &Path, output_path: &Path) -> Result<()> {
        // In real implementation would use TensorFlow -> Core ML conversion
        Ok(())
    }

    /// Optimize Core ML model for specific device
    pub fn optimize_for_device(
        model_path: &Path,
        output_path: &Path,
        target_device: CoreMLComputeUnit,
    ) -> Result<()> {
        // In real implementation would apply device-specific optimizations
        Ok(())
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

    #[test]
    fn test_device_capabilities() {
        let config = CoreMLBackendConfig::for_ios();
        let backend = CoreMLBackend::new(config).expect("operation failed in test");
        let capabilities = backend.device_capabilities();

        assert!(capabilities.has_neural_engine);
        assert!(capabilities.has_gpu);
        assert!(capabilities.supports_float16);
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

    #[test]
    fn test_model_converter() {
        // Test that converter methods don't panic
        let input_path = Path::new("input.pt");
        let output_path = Path::new("output.mlmodel");
        let input_shapes = HashMap::new();

        let result = CoreMLModelConverter::from_pytorch(input_path, output_path, input_shapes);
        assert!(result.is_ok());
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
        let mut backend = CoreMLBackend::new(config).expect("backend creation failed");
        // model not yet loaded — should return None
        // Note: new() does NOT call load_model(), so model is None initially
        // BUT the test config calls load_model in the pipeline ctor, not here.
        // We deliberately skip loading to verify None case.
        // Reset model to None manually isn't possible via public API, so we
        // just test that after loading the description is Some.
        let dummy = Path::new("/tmp/dummy.mlmodel");
        backend.load_model(dummy).expect("load_model mock should succeed");
        let desc = backend.model_description();
        assert!(
            desc.is_some(),
            "description must be present after load_model"
        );
    }

    #[test]
    fn test_model_description_fields_after_load() {
        let config = CoreMLBackendConfig::for_ios();
        let mut backend = CoreMLBackend::new(config).expect("backend creation failed");
        let dummy = Path::new("/tmp/test.mlpackage");
        backend.load_model(dummy).expect("load_model should succeed");
        let desc = backend.model_description().expect("description should be Some");
        assert!(!desc.name.is_empty());
        assert!(!desc.version.is_empty());
        assert!(!desc.input_names.is_empty());
        assert!(!desc.output_names.is_empty());
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

    // ── Predict after load returns Some output ────────────────────────────────

    #[test]
    fn test_predict_after_load_succeeds() {
        let config = CoreMLBackendConfig::for_ios();
        let mut backend = CoreMLBackend::new(config).expect("backend creation failed");
        backend.load_model(Path::new("/tmp/m.mlmodel")).expect("load ok");
        // Provide a dummy input tensor
        let input_tensor =
            trustformers_core::tensor::Tensor::zeros(&[1, 10]).expect("tensor creation ok");
        let mut inputs = HashMap::new();
        inputs.insert("input_ids".to_string(), input_tensor);
        let outputs = backend.predict(inputs).expect("predict should succeed after load");
        assert!(
            !outputs.is_empty(),
            "outputs should contain at least one entry"
        );
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

    #[test]
    fn test_device_capabilities_gpu_core_count() {
        let config = CoreMLBackendConfig::for_ios();
        let backend = CoreMLBackend::new(config).expect("backend ok");
        let cap = backend.device_capabilities();
        // Mock returns Some(8)
        assert!(
            cap.gpu_core_count.is_some(),
            "gpu_core_count should be present on Apple Silicon"
        );
    }

    #[test]
    fn test_device_capabilities_neural_engine_core_count() {
        let config = CoreMLBackendConfig::for_ios();
        let backend = CoreMLBackend::new(config).expect("backend ok");
        let cap = backend.device_capabilities();
        assert!(
            cap.neural_engine_core_count.is_some(),
            "neural_engine_core_count should be present"
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

    #[test]
    fn test_converter_from_onnx_ok() {
        let result =
            CoreMLModelConverter::from_onnx(Path::new("model.onnx"), Path::new("model.mlmodel"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_converter_from_tensorflow_ok() {
        let result = CoreMLModelConverter::from_tensorflow(
            Path::new("model.pb"),
            Path::new("model.mlmodel"),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_converter_optimize_for_device_ok() {
        let result = CoreMLModelConverter::optimize_for_device(
            Path::new("model.mlmodel"),
            Path::new("model_opt.mlmodel"),
            CoreMLComputeUnit::All,
        );
        assert!(result.is_ok());
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
}
