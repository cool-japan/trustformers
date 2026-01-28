// TensorRT Pipeline Backend Integration for TrustformeRS
// Provides high-performance TensorRT inference with optimized execution and memory management

use crate::core::traits::TokenizedInput;
use crate::error::{Result, TrustformersError};
use crate::pipeline::{BasePipeline, Device, Pipeline, PipelineOptions, PipelineOutput};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use trustformers_core::errors::{Result as CoreResult, TrustformersError as CoreTrustformersError};
use trustformers_core::traits::{Model, Tokenizer};
// Note: Using mock types since actual TensorRT runtime types need implementation
use trustformers_core::export::tensorrt::TensorRTConfig;

// Mock TensorRT types - replace with actual implementation
#[derive(Debug, Clone)]
pub struct TensorRTBackend;

#[derive(Debug, Clone)]
pub struct TensorRTEngine;

#[derive(Debug, Clone)]
pub struct TensorRTBuilder;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRTOptimizationProfile;

#[derive(Debug, Clone)]
pub struct TensorRTLogger;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PrecisionMode {
    FP32,
    FP16,
    INT8,
    INT4,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum MemoryStrategy {
    Conservative,
    Balanced,
    Aggressive,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DeviceType {
    GPU,
    DLA,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryPoolType {
    Workspace,
    DLAManagedSRAM,
    DLALocalDRAM,
    DLAGlobalDRAM,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    FastLatency,
    HighThroughput,
    MemoryOptimized,
    Balanced,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub avg_latency_ms: f64,
    pub throughput: f64,
    pub memory_usage: u64,
}

#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_memory: u64,
    pub used_memory: u64,
    pub free_memory: u64,
}

#[derive(Debug, Clone)]
pub struct DynamicShape {
    pub min_shape: Vec<i32>,
    pub opt_shape: Vec<i32>,
    pub max_shape: Vec<i32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationLevel {
    O0, // No optimization
    O1, // Basic optimization
    O2, // Standard optimization
    O3, // Aggressive optimization
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LogLevel {
    InternalError,
    Error,
    Warning,
    Info,
    Verbose,
}

// Mock implementations
impl TensorRTBackend {
    pub fn new(_config: TensorRTConfig) -> Result<Self> {
        Ok(Self)
    }

    pub fn load_engine(&self, _path: &PathBuf) -> Result<TensorRTEngine> {
        Ok(TensorRTEngine)
    }

    pub fn build_engine(&self, _builder: TensorRTBuilder) -> Result<TensorRTEngine> {
        Ok(TensorRTEngine)
    }

    pub fn get_available_devices(&self) -> Result<Vec<String>> {
        Ok(vec!["GPU:0".to_string(), "DLA:0".to_string()])
    }

    pub fn get_device_properties(&self, _device: &str) -> Result<HashMap<String, String>> {
        let mut props = HashMap::new();
        props.insert("type".to_string(), "mock".to_string());
        Ok(props)
    }

    pub fn save_engine(&self, _engine: &TensorRTEngine, _path: &Path) -> Result<()> {
        // Mock implementation - would serialize engine to file
        Ok(())
    }
}

impl TensorRTBuilder {
    pub fn from_path<P: AsRef<Path>>(_path: P) -> Result<Self> {
        Ok(TensorRTBuilder)
    }
}

impl TensorRTEngine {
    pub fn input_names(&self) -> Vec<String> {
        vec!["input_ids".to_string(), "attention_mask".to_string()]
    }

    pub fn output_names(&self) -> Vec<String> {
        vec!["logits".to_string()]
    }

    pub fn input_shapes(&self) -> HashMap<String, Vec<usize>> {
        let mut shapes = HashMap::new();
        shapes.insert("input_ids".to_string(), vec![1, 512]);
        shapes.insert("attention_mask".to_string(), vec![1, 512]);
        shapes
    }

    pub fn output_shapes(&self) -> HashMap<String, Vec<usize>> {
        let mut shapes = HashMap::new();
        shapes.insert("logits".to_string(), vec![1, 1000]); // Common classification size
        shapes
    }

    pub fn execute(
        &self,
        _inputs: HashMap<String, trustformers_core::tensor::Tensor>,
    ) -> Result<HashMap<String, trustformers_core::tensor::Tensor>> {
        // Mock implementation - return empty result
        let mut outputs = HashMap::new();
        let mock_tensor = trustformers_core::tensor::Tensor::zeros(&[1, 10])?;
        outputs.insert("logits".to_string(), mock_tensor);
        Ok(outputs)
    }

    pub fn execute_with_device(
        &self,
        inputs: HashMap<String, trustformers_core::tensor::Tensor>,
        _device: DeviceType,
    ) -> Result<HashMap<String, trustformers_core::tensor::Tensor>> {
        self.execute(inputs)
    }

    pub async fn execute_async(
        &self,
        inputs: HashMap<String, trustformers_core::tensor::Tensor>,
    ) -> Result<HashMap<String, trustformers_core::tensor::Tensor>> {
        self.execute(inputs)
    }

    pub fn benchmark(
        &self,
        _inputs: HashMap<String, trustformers_core::tensor::Tensor>,
        _num_runs: usize,
        _warmup_runs: usize,
    ) -> Result<BenchmarkResults> {
        Ok(BenchmarkResults {
            avg_latency_ms: 20.0,
            throughput: 50.0,
            memory_usage: 2 * 1024 * 1024 * 1024, // 2GB
        })
    }

    pub fn get_memory_info(&self) -> Result<MemoryInfo> {
        Ok(MemoryInfo {
            total_memory: 12 * 1024 * 1024 * 1024, // 12GB
            used_memory: 4 * 1024 * 1024 * 1024,   // 4GB
            free_memory: 8 * 1024 * 1024 * 1024,   // 8GB
        })
    }

    pub fn execute_with_context(
        &self,
        inputs: HashMap<String, trustformers_core::tensor::Tensor>,
        _context_id: usize,
    ) -> Result<HashMap<String, trustformers_core::tensor::Tensor>> {
        // Mock implementation - context_id is ignored
        self.execute(inputs)
    }

    pub fn get_device_info(&self) -> Result<HashMap<String, String>> {
        let mut info = HashMap::new();
        info.insert(
            "device_name".to_string(),
            "NVIDIA GeForce RTX 4090".to_string(),
        );
        info.insert("driver_version".to_string(), "535.104.05".to_string());
        info.insert("cuda_version".to_string(), "12.2".to_string());
        info.insert("compute_capability".to_string(), "8.9".to_string());
        Ok(info)
    }

    pub fn optimize_for_shapes(&self, _shapes: HashMap<String, Vec<i32>>) -> Result<()> {
        // Mock implementation - shapes optimization
        Ok(())
    }

    pub fn create_execution_context(&self) -> Result<usize> {
        // Mock implementation - return a context ID
        Ok(42)
    }

    pub fn get_performance_metrics(&self) -> Result<HashMap<String, f32>> {
        let mut metrics = HashMap::new();
        metrics.insert("throughput_ops_per_sec".to_string(), 1250.0);
        metrics.insert("memory_bandwidth_gb_per_sec".to_string(), 1008.0);
        metrics.insert("gpu_utilization_percent".to_string(), 85.0);
        metrics.insert("power_usage_watts".to_string(), 350.0);
        Ok(metrics)
    }

    pub fn run(
        &self,
        inputs: HashMap<String, trustformers_core::tensor::Tensor>,
    ) -> Result<HashMap<String, trustformers_core::tensor::Tensor>> {
        // Delegate to execute method
        self.execute(inputs)
    }
}
use serde::{Deserialize, Serialize};
use trustformers_core::tensor::Tensor;

/// TensorRT backend configuration for pipelines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRTBackendConfig {
    /// Path to the TensorRT engine file or ONNX model file
    pub model_path: PathBuf,
    /// TensorRT engine path (if different from model_path)
    pub engine_path: Option<PathBuf>,
    /// Target device type
    pub device_type: DeviceType,
    /// Device ID for multi-GPU setups
    pub device_id: i32,
    /// Precision mode for inference
    pub precision_mode: PrecisionMode,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Workspace size in bytes
    pub workspace_size: usize,
    /// Memory allocation strategy
    pub memory_strategy: MemoryStrategy,
    /// Optimization profiles for dynamic shapes
    pub optimization_profiles: Vec<TensorRTOptimizationProfile>,
    /// Memory pool configuration
    pub memory_pool_type: MemoryPoolType,
    /// Execution strategy
    pub execution_strategy: ExecutionStrategy,
    /// Enable DLA (Deep Learning Accelerator) if available
    pub enable_dla: bool,
    /// DLA core ID
    pub dla_core: Option<i32>,
    /// Enable profiling
    pub enable_profiling: bool,
    /// Profile output path
    pub profile_output_path: Option<PathBuf>,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Enable FP16 kernel selection
    pub enable_fp16: bool,
    /// Enable INT8 calibration
    pub enable_int8: bool,
    /// Calibration dataset path
    pub calibration_dataset_path: Option<PathBuf>,
    /// Minimum timing iterations
    pub min_timing_iterations: i32,
    /// Average timing iterations
    pub avg_timing_iterations: i32,
    /// Enable strict type constraints
    pub enable_strict_types: bool,
    /// Engine cache path
    pub engine_cache_path: Option<PathBuf>,
    /// Log level
    pub log_level: LogLevel,
}

impl Default for TensorRTBackendConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            engine_path: None,
            device_type: DeviceType::GPU,
            device_id: 0,
            precision_mode: PrecisionMode::FP32,
            max_batch_size: 1,
            workspace_size: 1024 * 1024 * 1024, // 1GB
            memory_strategy: MemoryStrategy::Balanced,
            optimization_profiles: Vec::new(),
            memory_pool_type: MemoryPoolType::Workspace,
            execution_strategy: ExecutionStrategy::Balanced,
            enable_dla: false,
            dla_core: None,
            enable_profiling: false,
            profile_output_path: None,
            optimization_level: OptimizationLevel::O3,
            enable_fp16: false,
            enable_int8: false,
            calibration_dataset_path: None,
            min_timing_iterations: 1,
            avg_timing_iterations: 8,
            enable_strict_types: false,
            engine_cache_path: None,
            log_level: LogLevel::Warning,
        }
    }
}

impl TensorRTBackendConfig {
    /// Create config optimized for latency
    pub fn latency_optimized(model_path: PathBuf) -> Self {
        Self {
            model_path,
            device_type: DeviceType::GPU,
            device_id: 0,
            precision_mode: PrecisionMode::FP16,
            max_batch_size: 1,
            workspace_size: 2 * 1024 * 1024 * 1024, // 2GB
            memory_pool_type: MemoryPoolType::Workspace,
            execution_strategy: ExecutionStrategy::FastLatency,
            enable_dla: false,
            optimization_level: OptimizationLevel::O3,
            enable_fp16: true,
            enable_int8: false,
            min_timing_iterations: 3,
            avg_timing_iterations: 16,
            enable_strict_types: true,
            log_level: LogLevel::Error,
            ..Default::default()
        }
    }

    /// Create config optimized for throughput
    pub fn throughput_optimized(model_path: PathBuf, max_batch_size: usize) -> Self {
        Self {
            model_path,
            device_type: DeviceType::GPU,
            device_id: 0,
            precision_mode: PrecisionMode::FP16,
            max_batch_size,
            workspace_size: 4 * 1024 * 1024 * 1024, // 4GB
            memory_pool_type: MemoryPoolType::Workspace,
            execution_strategy: ExecutionStrategy::HighThroughput,
            enable_dla: false,
            optimization_level: OptimizationLevel::O3,
            enable_fp16: true,
            enable_int8: false,
            min_timing_iterations: 1,
            avg_timing_iterations: 8,
            enable_strict_types: false,
            log_level: LogLevel::Warning,
            ..Default::default()
        }
    }

    /// Create config optimized for memory efficiency
    pub fn memory_optimized(model_path: PathBuf) -> Self {
        Self {
            model_path,
            device_type: DeviceType::GPU,
            device_id: 0,
            precision_mode: PrecisionMode::INT8,
            max_batch_size: 1,
            workspace_size: 512 * 1024 * 1024, // 512MB
            memory_pool_type: MemoryPoolType::Workspace,
            execution_strategy: ExecutionStrategy::MemoryOptimized,
            enable_dla: false,
            optimization_level: OptimizationLevel::O2,
            enable_fp16: false,
            enable_int8: true,
            min_timing_iterations: 1,
            avg_timing_iterations: 4,
            enable_strict_types: true,
            log_level: LogLevel::Error,
            ..Default::default()
        }
    }

    /// Create config for production deployment
    pub fn production(model_path: PathBuf, max_batch_size: usize) -> Self {
        Self {
            model_path,
            device_type: DeviceType::GPU,
            device_id: 0,
            precision_mode: PrecisionMode::FP16,
            max_batch_size,
            workspace_size: 2 * 1024 * 1024 * 1024, // 2GB
            memory_pool_type: MemoryPoolType::Workspace,
            execution_strategy: ExecutionStrategy::Balanced,
            enable_dla: false,
            optimization_level: OptimizationLevel::O3,
            enable_fp16: true,
            enable_int8: false,
            min_timing_iterations: 2,
            avg_timing_iterations: 8,
            enable_strict_types: true,
            log_level: LogLevel::Error,
            ..Default::default()
        }
    }

    /// Create config with INT8 quantization
    pub fn int8_optimized(model_path: PathBuf, calibration_dataset_path: PathBuf) -> Self {
        Self {
            model_path,
            device_type: DeviceType::GPU,
            device_id: 0,
            precision_mode: PrecisionMode::INT8,
            max_batch_size: 1,
            workspace_size: 1024 * 1024 * 1024, // 1GB
            memory_pool_type: MemoryPoolType::Workspace,
            execution_strategy: ExecutionStrategy::Balanced,
            enable_dla: false,
            optimization_level: OptimizationLevel::O3,
            enable_fp16: false,
            enable_int8: true,
            calibration_dataset_path: Some(calibration_dataset_path),
            min_timing_iterations: 1,
            avg_timing_iterations: 4,
            enable_strict_types: true,
            log_level: LogLevel::Warning,
            ..Default::default()
        }
    }

    /// Enable DLA acceleration if available
    pub fn with_dla(mut self, dla_core: i32) -> Self {
        self.enable_dla = true;
        self.dla_core = Some(dla_core);
        self
    }

    /// Enable profiling with output path
    pub fn with_profiling(mut self, output_path: PathBuf) -> Self {
        self.enable_profiling = true;
        self.profile_output_path = Some(output_path);
        self
    }

    /// Add optimization profile for dynamic shapes
    pub fn with_optimization_profile(mut self, profile: TensorRTOptimizationProfile) -> Self {
        self.optimization_profiles.push(profile);
        self
    }

    /// Set engine cache path
    pub fn with_engine_cache(mut self, cache_path: PathBuf) -> Self {
        self.engine_cache_path = Some(cache_path);
        self
    }

    /// Convert to TensorRT runtime config
    pub fn to_runtime_config(&self) -> trustformers_core::export::tensorrt::TensorRTConfig {
        // Map backend config to core TensorRT config
        trustformers_core::export::tensorrt::TensorRTConfig {
            max_batch_size: self.max_batch_size,
            max_sequence_length: 2048, // Default sequence length
            workspace_size: match self.memory_strategy {
                MemoryStrategy::Conservative => 512, // 512MB
                MemoryStrategy::Balanced => 1024,    // 1GB
                MemoryStrategy::Aggressive => 2048,  // 2GB
            },
            fp16_enabled: self.precision_mode == PrecisionMode::FP16,
            int8_enabled: self.precision_mode == PrecisionMode::INT8,
            dynamic_shapes: true, // Always enable dynamic shapes
            optimization_level: match self.optimization_level {
                OptimizationLevel::O0 => 0,
                OptimizationLevel::O1 => 1,
                OptimizationLevel::O2 => 2,
                OptimizationLevel::O3 => 3,
            },
        }
    }
}

/// TensorRT-backed model wrapper
#[derive(Clone)]
pub struct TensorRTModel {
    engine: Arc<TensorRTEngine>,
    config: TensorRTBackendConfig,
    input_names: Vec<String>,
    output_names: Vec<String>,
    input_shapes: HashMap<String, Vec<i32>>,
    output_shapes: HashMap<String, Vec<i32>>,
}

impl TensorRTModel {
    /// Create new TensorRT model from config
    pub fn from_config(config: TensorRTBackendConfig) -> Result<Self> {
        if !config.model_path.exists() {
            return Err(TrustformersError::Core(CoreTrustformersError::other(
                format!(
                    "TensorRT model file not found: {}",
                    config.model_path.to_string_lossy()
                ),
            )));
        }

        let runtime_config = config.to_runtime_config();
        let backend = TensorRTBackend::new(runtime_config)?;

        // Try to load existing engine or build new one
        let engine = if let Some(ref engine_path) = config.engine_path {
            if engine_path.exists() {
                backend.load_engine(engine_path)?
            } else {
                let engine =
                    backend.build_engine(TensorRTBuilder::from_path(&config.model_path)?)?;
                backend.save_engine(&engine, engine_path)?;
                engine
            }
        } else {
            // Check cache first
            if let Some(ref cache_path) = config.engine_cache_path {
                let engine_path = cache_path.join(format!(
                    "{}.engine",
                    config.model_path.file_stem().unwrap().to_string_lossy()
                ));
                if engine_path.exists() {
                    backend.load_engine(&engine_path)?
                } else {
                    let engine =
                        backend.build_engine(TensorRTBuilder::from_path(&config.model_path)?)?;
                    std::fs::create_dir_all(cache_path)?;
                    backend.save_engine(&engine, &engine_path)?;
                    engine
                }
            } else {
                backend.build_engine(TensorRTBuilder::from_path(&config.model_path)?)?
            }
        };

        let input_names = engine.input_names().to_vec();
        let output_names = engine.output_names().to_vec();
        let input_shapes = engine
            .input_shapes()
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().map(|&x| x as i32).collect()))
            .collect();
        let output_shapes = engine
            .output_shapes()
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().map(|&x| x as i32).collect()))
            .collect();

        Ok(Self {
            engine: Arc::new(engine),
            config,
            input_names,
            output_names,
            input_shapes,
            output_shapes,
        })
    }

    /// Load from ONNX file with default config
    pub fn from_pretrained<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let config = TensorRTBackendConfig {
            model_path: model_path.as_ref().to_path_buf(),
            ..Default::default()
        };
        Self::from_config(config)
    }

    /// Load with specific precision mode
    pub fn from_pretrained_with_precision<P: AsRef<Path>>(
        model_path: P,
        precision_mode: PrecisionMode,
    ) -> Result<Self> {
        let config = TensorRTBackendConfig {
            model_path: model_path.as_ref().to_path_buf(),
            precision_mode,
            enable_fp16: matches!(precision_mode, PrecisionMode::FP16),
            enable_int8: matches!(precision_mode, PrecisionMode::INT8),
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

    /// Get input shapes
    pub fn input_shapes(&self) -> &HashMap<String, Vec<i32>> {
        &self.input_shapes
    }

    /// Get output shapes
    pub fn output_shapes(&self) -> &HashMap<String, Vec<i32>> {
        &self.output_shapes
    }

    /// Run inference
    pub fn forward(&self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        self.engine.execute(inputs)
    }

    /// Run inference with specific execution context
    pub fn forward_with_context(
        &self,
        inputs: HashMap<String, Tensor>,
        context_id: usize,
    ) -> Result<HashMap<String, Tensor>> {
        self.engine.execute_with_context(inputs, context_id)
    }

    /// Run inference asynchronously
    pub async fn forward_async(
        &self,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        self.engine
            .execute_async(inputs)
            .await
            .map_err(|e| TrustformersError::FeatureUnavailable {
                message: format!("Execute async failed: {}", e),
                feature: "async_execution".to_string(),
                suggestion: Some("Check TensorRT async execution support".to_string()),
                alternatives: vec!["Use synchronous execution".to_string()],
            })
    }

    /// Benchmark the model
    pub fn benchmark(
        &self,
        inputs: HashMap<String, Tensor>,
        num_runs: usize,
    ) -> Result<BenchmarkResults> {
        self.engine.benchmark(inputs, num_runs, 5) // Default 5 warmup runs
    }

    /// Get memory usage information
    pub fn memory_info(&self) -> Result<MemoryInfo> {
        self.engine
            .get_memory_info()
            .map_err(|e| TrustformersError::FeatureUnavailable {
                message: format!("Get memory info failed: {}", e),
                feature: "memory_info".to_string(),
                suggestion: Some("Check TensorRT runtime status".to_string()),
                alternatives: vec!["Use default memory estimates".to_string()],
            })
    }

    /// Get device information
    pub fn device_info(&self) -> Result<HashMap<String, String>> {
        self.engine
            .get_device_info()
            .map_err(|e| TrustformersError::FeatureUnavailable {
                message: format!("Get device info failed: {}", e),
                feature: "device_info".to_string(),
                suggestion: Some("Check TensorRT device availability".to_string()),
                alternatives: vec!["Use default device settings".to_string()],
            })
    }

    /// Get model path
    pub fn model_path(&self) -> &Path {
        &self.config.model_path
    }

    /// Get engine path
    pub fn engine_path(&self) -> Option<&Path> {
        self.config.engine_path.as_deref()
    }

    /// Get precision mode
    pub fn precision_mode(&self) -> &PrecisionMode {
        &self.config.precision_mode
    }

    /// Get max batch size
    pub fn max_batch_size(&self) -> usize {
        self.config.max_batch_size
    }

    /// Optimize for specific input shapes
    pub fn optimize_for_shapes(&mut self, shapes: HashMap<String, Vec<i32>>) -> Result<()> {
        self.engine
            .optimize_for_shapes(shapes)
            .map_err(|e| TrustformersError::FeatureUnavailable {
                message: format!("Optimize for shapes failed: {}", e),
                feature: "shape_optimization".to_string(),
                suggestion: Some(
                    "Check input shapes and TensorRT dynamic shape support".to_string(),
                ),
                alternatives: vec!["Use fixed shapes".to_string()],
            })
    }

    /// Create execution context
    pub fn create_execution_context(&self) -> Result<usize> {
        self.engine
            .create_execution_context()
            .map_err(|e| TrustformersError::FeatureUnavailable {
                message: format!("Create execution context failed: {}", e),
                feature: "execution_context".to_string(),
                suggestion: Some("Check TensorRT runtime and memory availability".to_string()),
                alternatives: vec!["Use default execution context".to_string()],
            })
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> Result<HashMap<String, f32>> {
        self.engine
            .get_performance_metrics()
            .map_err(|e| TrustformersError::FeatureUnavailable {
                message: format!("Get performance metrics failed: {}", e),
                feature: "performance_metrics".to_string(),
                suggestion: Some("Check TensorRT profiling support".to_string()),
                alternatives: vec!["Use benchmark results instead".to_string()],
            })
    }
}

impl crate::core::traits::Config for TensorRTBackendConfig {
    fn validate(&self) -> CoreResult<()> {
        if !self.model_path.exists() {
            return Err(CoreTrustformersError::other(format!(
                "TensorRT model file not found: {}",
                self.model_path.to_string_lossy()
            )));
        }
        if self.max_batch_size == 0 {
            return Err(CoreTrustformersError::other(
                "max_batch_size must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "tensorrt"
    }
}

impl Model for TensorRTModel {
    type Config = TensorRTBackendConfig;
    type Input = HashMap<String, Tensor>;
    type Output = HashMap<String, Tensor>;

    /// Forward pass implementation for Model trait
    fn forward(&self, inputs: Self::Input) -> CoreResult<Self::Output> {
        // Run inference using the TensorRT engine
        self.engine.run(inputs).map_err(Into::into)
    }

    /// Load pretrained weights (not applicable for TensorRT models as they're already loaded)
    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> CoreResult<()> {
        // TensorRT models are already optimized and loaded from engine, so this is a no-op
        Ok(())
    }

    /// Get model configuration
    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    /// Get the number of parameters in the model
    fn num_parameters(&self) -> usize {
        // For TensorRT models, we can't easily determine this without parsing the model
        // Return a placeholder value or implement actual parameter counting if needed
        0 // Placeholder - would need TensorRT model introspection
    }
}

/// TensorRT tokenizer wrapper (can wrap existing tokenizers)
#[derive(Clone)]
pub struct TensorRTTokenizer<T> {
    inner: T,
}

impl<T: Tokenizer> TensorRTTokenizer<T> {
    pub fn new(tokenizer: T) -> Self {
        Self { inner: tokenizer }
    }
}

impl<T: Tokenizer> Tokenizer for TensorRTTokenizer<T> {
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

/// TensorRT-backed pipeline base
pub type TensorRTBasePipeline<T> = BasePipeline<TensorRTModel, TensorRTTokenizer<T>>;

/// Text classification pipeline with TensorRT backend
pub struct TensorRTTextClassificationPipeline<T> {
    base: TensorRTBasePipeline<T>,
    return_all_scores: bool,
}

impl<T: Tokenizer + Clone> TensorRTTextClassificationPipeline<T> {
    pub fn new(model: TensorRTModel, tokenizer: TensorRTTokenizer<T>) -> Result<Self> {
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
    pub fn benchmark(&self, input: &str, num_runs: usize) -> Result<BenchmarkResults> {
        let tokenized = self.base.tokenizer.encode(input)?;
        let inputs = self.prepare_inputs(&tokenized)?;
        self.base.model.benchmark(inputs, num_runs)
    }

    /// Get memory usage
    pub fn memory_info(&self) -> Result<MemoryInfo> {
        self.base.model.memory_info()
    }

    /// Get device info
    pub fn device_info(&self) -> Result<HashMap<String, String>> {
        self.base.model.device_info()
    }

    /// Get performance metrics
    pub fn performance_metrics(&self) -> Result<HashMap<String, f32>> {
        self.base.model.get_performance_metrics()
    }

    fn prepare_inputs(&self, tokenized: &TokenizedInput) -> Result<HashMap<String, Tensor>> {
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
        let attention_mask = Tensor::from_vec(
            tokenized.attention_mask.iter().map(|&x| x as f32).collect(),
            &[batch_size, seq_len],
        )?;
        inputs.insert("attention_mask".to_string(), attention_mask);

        Ok(inputs)
    }
}

impl<T: Tokenizer + Clone> Pipeline for TensorRTTextClassificationPipeline<T> {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        let tokenized = self.base.tokenizer.encode(&input)?;
        let inputs = self.prepare_inputs(&tokenized)?;
        let outputs = self.base.model.forward(inputs)?;

        // Get logits (assuming first output)
        let logits = outputs.into_values().next().ok_or_else(|| {
            TrustformersError::invalid_input(
                "No logits output found in model inference results",
                Some("model_outputs"),
                Some("at least one output tensor"),
                Some("empty outputs"),
            )
        })?;

        // Apply softmax to get probabilities
        let logits_data = logits.data()?;
        let max_logit = logits_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits_data.iter().map(|&x| (x - max_logit).exp()).collect();
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
                .unwrap();
            results.push(crate::pipeline::ClassificationOutput {
                label: format!("LABEL_{}", max_idx),
                score: max_score,
            });
        }

        Ok(PipelineOutput::Classification(results))
    }

    fn batch(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        // TensorRT excels at batch processing
        let tokenized_inputs: Result<Vec<TokenizedInput>> = inputs
            .iter()
            .map(|input| self.base.tokenizer.encode(input).map_err(Into::into))
            .collect();
        let tokenized_inputs = tokenized_inputs?;

        // Prepare batch inputs
        let batch_size = inputs.len();
        let max_seq_len = tokenized_inputs.iter().map(|t| t.input_ids.len()).max().unwrap_or(0);

        let mut batch_input_ids = Vec::new();
        let mut batch_attention_mask = Vec::new();

        for tokenized in &tokenized_inputs {
            let mut input_ids = tokenized.input_ids.clone();
            let mut attention_mask = tokenized.attention_mask.clone();

            // Pad to max length
            while input_ids.len() < max_seq_len {
                input_ids.push(0);
                attention_mask.push(0);
            }

            batch_input_ids.extend(input_ids.iter().map(|&x| x as f32));
            batch_attention_mask.extend(attention_mask.iter().map(|&x| x as f32));
        }

        let mut batch_inputs = HashMap::new();
        batch_inputs.insert(
            "input_ids".to_string(),
            Tensor::from_vec(batch_input_ids, &[batch_size, max_seq_len])?,
        );
        batch_inputs.insert(
            "attention_mask".to_string(),
            Tensor::from_vec(batch_attention_mask, &[batch_size, max_seq_len])?,
        );

        let outputs = self.base.model.forward(batch_inputs)?;
        let logits = outputs.into_values().next().ok_or_else(|| {
            TrustformersError::invalid_input(
                "No logits output found in model inference results",
                Some("model_outputs"),
                Some("at least one output tensor"),
                Some("empty outputs"),
            )
        })?;

        // Process batch outputs
        let logits_data = logits.data()?;
        let vocab_size = logits_data.len() / batch_size;

        let mut results = Vec::new();
        for i in 0..batch_size {
            let start_idx = i * vocab_size;
            let end_idx = (i + 1) * vocab_size;
            let sample_logits = &logits_data[start_idx..end_idx];

            // Apply softmax
            let max_logit = sample_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_logits: Vec<f32> =
                sample_logits.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_exp: f32 = exp_logits.iter().sum();
            let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

            let mut sample_results = Vec::new();
            if self.return_all_scores {
                for (j, &score) in probs.iter().enumerate() {
                    sample_results.push(crate::pipeline::ClassificationOutput {
                        label: format!("LABEL_{}", j),
                        score,
                    });
                }
            } else {
                let (max_idx, &max_score) = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                sample_results.push(crate::pipeline::ClassificationOutput {
                    label: format!("LABEL_{}", max_idx),
                    score: max_score,
                });
            }

            results.push(PipelineOutput::Classification(sample_results));
        }

        Ok(results)
    }
}

/// Text generation pipeline with TensorRT backend
pub struct TensorRTTextGenerationPipeline<T> {
    base: TensorRTBasePipeline<T>,
    max_new_tokens: usize,
    do_sample: bool,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
}

impl<T: Tokenizer + Clone> TensorRTTextGenerationPipeline<T> {
    pub fn new(model: TensorRTModel, tokenizer: TensorRTTokenizer<T>) -> Result<Self> {
        Ok(Self {
            base: BasePipeline::new(model, tokenizer),
            max_new_tokens: 50,
            do_sample: false,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.0,
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

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = repetition_penalty;
        self
    }

    /// Benchmark this pipeline
    pub fn benchmark(&self, input: &str, num_runs: usize) -> Result<BenchmarkResults> {
        let tokenized = self.base.tokenizer.encode(input)?;
        let inputs = self.prepare_inputs(&tokenized.input_ids)?;
        self.base.model.benchmark(inputs, num_runs)
    }

    fn prepare_inputs(&self, input_ids: &[u32]) -> Result<HashMap<String, Tensor>> {
        let mut inputs = HashMap::new();

        let batch_size = 1;
        let seq_len = input_ids.len();

        let input_ids_tensor = Tensor::from_vec(
            input_ids.iter().map(|&x| x as f32).collect(),
            &[batch_size, seq_len],
        )?;
        inputs.insert("input_ids".to_string(), input_ids_tensor);

        let attention_mask =
            Tensor::from_vec(vec![1.0f32; batch_size * seq_len], &[batch_size, seq_len])?;
        inputs.insert("attention_mask".to_string(), attention_mask);

        Ok(inputs)
    }

    fn apply_repetition_penalty(&self, logits: &mut [f32], input_ids: &[u32]) {
        if self.repetition_penalty == 1.0 {
            return;
        }

        for &token_id in input_ids {
            if token_id < logits.len() as u32 {
                let idx = token_id as usize;
                if logits[idx] > 0.0 {
                    logits[idx] /= self.repetition_penalty;
                } else {
                    logits[idx] *= self.repetition_penalty;
                }
            }
        }
    }

    fn sample_token(&self, logits: &[f32], input_ids: &[u32]) -> u32 {
        let mut logits = logits.to_vec();
        self.apply_repetition_penalty(&mut logits, input_ids);

        if self.do_sample {
            // Apply temperature
            if self.temperature != 1.0 {
                for logit in &mut logits {
                    *logit /= self.temperature;
                }
            }

            // Apply top-k filtering
            if self.top_k < logits.len() {
                let mut indexed_logits: Vec<(usize, f32)> =
                    logits.iter().enumerate().map(|(i, &logit)| (i, logit)).collect();
                indexed_logits
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                for (i, _) in indexed_logits.iter().skip(self.top_k) {
                    logits[*i] = f32::NEG_INFINITY;
                }
            }

            // Apply softmax
            let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_exp: f32 = exp_logits.iter().sum();
            let mut probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

            // Apply top-p (nucleus) sampling
            if self.top_p < 1.0 {
                let mut indexed_probs: Vec<(usize, f32)> =
                    probs.iter().enumerate().map(|(i, &prob)| (i, prob)).collect();
                indexed_probs
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let mut cumulative_prob = 0.0;
                for (i, &(_, prob)) in indexed_probs.iter().enumerate() {
                    cumulative_prob += prob;
                    if cumulative_prob > self.top_p {
                        for (j, _) in indexed_probs.iter().skip(i + 1) {
                            probs[*j] = 0.0;
                        }
                        break;
                    }
                }

                // Renormalize
                let sum_probs: f32 = probs.iter().sum();
                for prob in &mut probs {
                    *prob /= sum_probs;
                }
            }

            // Sample from distribution
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            input_ids.hash(&mut hasher);
            let hash = hasher.finish();
            let random_val = (hash % 1000000) as f32 / 1000000.0;

            let mut cumulative = 0.0;
            for (i, &prob) in probs.iter().enumerate() {
                cumulative += prob;
                if random_val <= cumulative {
                    return i as u32;
                }
            }

            // Fallback to last token
            (probs.len() - 1) as u32
        } else {
            // Greedy decoding
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0)
        }
    }
}

impl<T: Tokenizer + Clone> Pipeline for TensorRTTextGenerationPipeline<T> {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        let tokenized = self.base.tokenizer.encode(&input)?;
        let mut input_ids = tokenized.input_ids.clone();

        // Autoregressive generation with TensorRT optimization
        for _ in 0..self.max_new_tokens {
            let inputs = self.prepare_inputs(&input_ids)?;
            let outputs = self.base.model.forward(inputs)?;
            let logits = outputs.into_values().next().ok_or_else(|| {
                TrustformersError::invalid_input(
                    "No logits output found in model inference results",
                    Some("model_outputs"),
                    Some("at least one output tensor"),
                    Some("empty outputs"),
                )
            })?;

            // Get next token logits
            let logits_data = logits.data()?;
            let vocab_size = logits_data.len() / input_ids.len();
            let last_token_logits =
                &logits_data[(input_ids.len() - 1) * vocab_size..input_ids.len() * vocab_size];

            let next_token = self.sample_token(last_token_logits, &input_ids);
            input_ids.push(next_token);

            // Simple stopping condition
            if next_token == 0 || next_token == 2 {
                // Common EOS tokens
                break;
            }
        }

        let generated_text = self.base.tokenizer.decode(&input_ids)?;

        Ok(PipelineOutput::Generation(
            crate::pipeline::GenerationOutput {
                generated_text,
                sequences: Some(vec![input_ids]),
                scores: None,
            },
        ))
    }
}

/// Factory functions for TensorRT pipelines
pub fn tensorrt_text_classification_pipeline<T: Tokenizer + Clone>(
    model_path: impl AsRef<Path>,
    tokenizer: T,
    config: Option<TensorRTBackendConfig>,
) -> Result<TensorRTTextClassificationPipeline<T>> {
    let config = config.unwrap_or_else(|| {
        TensorRTBackendConfig::latency_optimized(model_path.as_ref().to_path_buf())
    });
    let model = TensorRTModel::from_config(config)?;
    let tensorrt_tokenizer = TensorRTTokenizer::new(tokenizer);
    TensorRTTextClassificationPipeline::new(model, tensorrt_tokenizer)
}

pub fn tensorrt_text_generation_pipeline<T: Tokenizer + Clone>(
    model_path: impl AsRef<Path>,
    tokenizer: T,
    config: Option<TensorRTBackendConfig>,
) -> Result<TensorRTTextGenerationPipeline<T>> {
    let config = config.unwrap_or_else(|| {
        TensorRTBackendConfig::throughput_optimized(model_path.as_ref().to_path_buf(), 4)
    });
    let model = TensorRTModel::from_config(config)?;
    let tensorrt_tokenizer = TensorRTTokenizer::new(tokenizer);
    TensorRTTextGenerationPipeline::new(model, tensorrt_tokenizer)
}

/// Enhanced pipeline options with TensorRT backend support
#[derive(Clone, Debug)]
pub struct TensorRTPipelineOptions {
    pub base_options: PipelineOptions,
    pub tensorrt_config: TensorRTBackendConfig,
    pub enable_profiling: bool,
    pub warmup_runs: usize,
    pub enable_engine_cache: bool,
    pub cache_path: Option<PathBuf>,
}

impl Default for TensorRTPipelineOptions {
    fn default() -> Self {
        Self {
            base_options: PipelineOptions::default(),
            tensorrt_config: TensorRTBackendConfig::default(),
            enable_profiling: false,
            warmup_runs: 3,
            enable_engine_cache: true,
            cache_path: None,
        }
    }
}

impl TensorRTPipelineOptions {
    pub fn latency_optimized(model_path: PathBuf) -> Self {
        Self {
            base_options: PipelineOptions::default(),
            tensorrt_config: TensorRTBackendConfig::latency_optimized(model_path),
            enable_profiling: false,
            warmup_runs: 5,
            enable_engine_cache: true,
            cache_path: Some(PathBuf::from("./tensorrt_cache")),
        }
    }

    pub fn throughput_optimized(model_path: PathBuf, max_batch_size: usize) -> Self {
        Self {
            base_options: PipelineOptions {
                batch_size: Some(max_batch_size),
                device: Some(Device::Gpu(0)),
                ..Default::default()
            },
            tensorrt_config: TensorRTBackendConfig::throughput_optimized(
                model_path,
                max_batch_size,
            ),
            enable_profiling: false,
            warmup_runs: 3,
            enable_engine_cache: true,
            cache_path: Some(PathBuf::from("./tensorrt_cache")),
        }
    }

    pub fn memory_optimized(model_path: PathBuf) -> Self {
        Self {
            base_options: PipelineOptions::default(),
            tensorrt_config: TensorRTBackendConfig::memory_optimized(model_path),
            enable_profiling: false,
            warmup_runs: 2,
            enable_engine_cache: true,
            cache_path: Some(PathBuf::from("./tensorrt_cache")),
        }
    }

    pub fn production(model_path: PathBuf, max_batch_size: usize) -> Self {
        Self {
            base_options: PipelineOptions {
                batch_size: Some(max_batch_size),
                device: Some(Device::Gpu(0)),
                ..Default::default()
            },
            tensorrt_config: TensorRTBackendConfig::production(model_path, max_batch_size),
            enable_profiling: false,
            warmup_runs: 10,
            enable_engine_cache: true,
            cache_path: Some(PathBuf::from("./tensorrt_cache")),
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

    pub fn with_cache_path(mut self, path: PathBuf) -> Self {
        self.cache_path = Some(path);
        self.enable_engine_cache = true;
        self
    }

    pub fn with_engine_cache(mut self, enable: bool) -> Self {
        self.enable_engine_cache = enable;
        self
    }
}

/// TensorRT pipeline manager for coordinating multiple engines
pub struct TensorRTPipelineManager {
    models: HashMap<String, TensorRTModel>,
    default_config: TensorRTBackendConfig,
    engine_cache_path: Option<PathBuf>,
}

impl TensorRTPipelineManager {
    pub fn new(default_config: TensorRTBackendConfig) -> Self {
        Self {
            models: HashMap::new(),
            default_config,
            engine_cache_path: None,
        }
    }

    pub fn with_engine_cache(mut self, cache_path: PathBuf) -> Self {
        self.engine_cache_path = Some(cache_path);
        self
    }

    /// Register a model with the manager
    pub fn register_model(&mut self, name: String, model: TensorRTModel) {
        self.models.insert(name, model);
    }

    /// Load and register a model from path
    pub fn load_model<P: AsRef<Path>>(&mut self, name: String, model_path: P) -> Result<()> {
        let mut config = self.default_config.clone();
        config.model_path = model_path.as_ref().to_path_buf();

        if let Some(ref cache_path) = self.engine_cache_path {
            config.engine_cache_path = Some(cache_path.clone());
        }

        let model = TensorRTModel::from_config(config)?;
        self.register_model(name, model);
        Ok(())
    }

    /// Load model with specific config
    pub fn load_model_with_config<P: AsRef<Path>>(
        &mut self,
        name: String,
        model_path: P,
        config: TensorRTBackendConfig,
    ) -> Result<()> {
        let mut config = config;
        config.model_path = model_path.as_ref().to_path_buf();

        if let Some(ref cache_path) = self.engine_cache_path {
            config.engine_cache_path = Some(cache_path.clone());
        }

        let model = TensorRTModel::from_config(config)?;
        self.register_model(name, model);
        Ok(())
    }

    /// Get a registered model
    pub fn get_model(&self, name: &str) -> Option<&TensorRTModel> {
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
    ) -> Result<HashMap<String, BenchmarkResults>> {
        let mut results = HashMap::new();
        for (name, model) in &self.models {
            let benchmark = model.benchmark(inputs.clone(), num_runs)?;
            results.insert(name.clone(), benchmark);
        }
        Ok(results)
    }

    /// Get memory info for all models
    pub fn memory_info_all(&self) -> Result<HashMap<String, MemoryInfo>> {
        let mut results = HashMap::new();
        for (name, model) in &self.models {
            let info = model.memory_info()?;
            results.insert(name.clone(), info);
        }
        Ok(results)
    }

    /// Get device info for all models
    pub fn device_info_all(&self) -> Result<HashMap<String, HashMap<String, String>>> {
        let mut results = HashMap::new();
        for (name, model) in &self.models {
            let info = model.device_info()?;
            results.insert(name.clone(), info);
        }
        Ok(results)
    }

    /// Get performance metrics for all models
    pub fn performance_metrics_all(&self) -> Result<HashMap<String, HashMap<String, f32>>> {
        let mut results = HashMap::new();
        for (name, model) in &self.models {
            let metrics = model.get_performance_metrics()?;
            results.insert(name.clone(), metrics);
        }
        Ok(results)
    }

    /// Clear engine cache
    pub fn clear_engine_cache(&self) -> Result<()> {
        if let Some(ref cache_path) = self.engine_cache_path {
            if cache_path.exists() {
                std::fs::remove_dir_all(cache_path)?;
                std::fs::create_dir_all(cache_path)?;
            }
        }
        Ok(())
    }

    /// Get cache size
    pub fn cache_size(&self) -> Result<u64> {
        if let Some(ref cache_path) = self.engine_cache_path {
            if cache_path.exists() {
                let mut size = 0;
                for entry in std::fs::read_dir(cache_path)? {
                    let entry = entry?;
                    if let Ok(metadata) = entry.metadata() {
                        size += metadata.len();
                    }
                }
                Ok(size)
            } else {
                Ok(0)
            }
        } else {
            Ok(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_tensorrt_backend_config() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.onnx");

        let config = TensorRTBackendConfig::latency_optimized(model_path.clone());
        assert_eq!(config.model_path, model_path);
        assert!(matches!(config.precision_mode, PrecisionMode::FP16));
        assert!(config.enable_fp16);
        assert_eq!(config.max_batch_size, 1);

        let runtime_config = config.to_runtime_config();
        assert!(runtime_config.fp16_enabled);
        assert!(!runtime_config.int8_enabled);
    }

    #[test]
    fn test_tensorrt_backend_config_throughput() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.onnx");

        let config = TensorRTBackendConfig::throughput_optimized(model_path.clone(), 8);
        assert_eq!(config.model_path, model_path);
        assert_eq!(config.max_batch_size, 8);
        assert!(matches!(
            config.execution_strategy,
            ExecutionStrategy::HighThroughput
        ));
    }

    #[test]
    fn test_tensorrt_backend_config_memory() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.onnx");

        let config = TensorRTBackendConfig::memory_optimized(model_path.clone());
        assert_eq!(config.model_path, model_path);
        assert!(matches!(config.precision_mode, PrecisionMode::INT8));
        assert!(config.enable_int8);
        assert_eq!(config.workspace_size, 512 * 1024 * 1024);
    }

    #[test]
    fn test_tensorrt_backend_config_int8() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.onnx");
        let calib_path = temp_dir.path().join("calibration.txt");

        let config = TensorRTBackendConfig::int8_optimized(model_path.clone(), calib_path.clone());
        assert_eq!(config.model_path, model_path);
        assert_eq!(config.calibration_dataset_path, Some(calib_path));
        assert!(matches!(config.precision_mode, PrecisionMode::INT8));
        assert!(config.enable_int8);
    }

    #[test]
    fn test_tensorrt_pipeline_options() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.onnx");

        let options = TensorRTPipelineOptions::latency_optimized(model_path.clone());
        assert_eq!(options.tensorrt_config.model_path, model_path);
        assert_eq!(options.warmup_runs, 5);
        assert!(options.enable_engine_cache);

        let throughput_options =
            TensorRTPipelineOptions::throughput_optimized(model_path.clone(), 8);
        assert_eq!(throughput_options.tensorrt_config.max_batch_size, 8);
        assert!(matches!(
            throughput_options.base_options.device,
            Some(Device::Gpu(0))
        ));
    }

    #[test]
    fn test_tensorrt_pipeline_manager() {
        let config = TensorRTBackendConfig::default();
        let manager = TensorRTPipelineManager::new(config);

        assert_eq!(manager.list_models().len(), 0);
        assert_eq!(manager.cache_size().unwrap(), 0);
    }

    #[test]
    fn test_tensorrt_config_with_dla() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.onnx");

        let config = TensorRTBackendConfig::latency_optimized(model_path.clone()).with_dla(0);

        assert!(config.enable_dla);
        assert_eq!(config.dla_core, Some(0));
    }

    #[test]
    fn test_tensorrt_config_with_profiling() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.onnx");
        let profile_path = temp_dir.path().join("profile.json");

        let config = TensorRTBackendConfig::latency_optimized(model_path.clone())
            .with_profiling(profile_path.clone());

        assert!(config.enable_profiling);
        assert_eq!(config.profile_output_path, Some(profile_path));
    }

    #[test]
    fn test_tensorrt_config_with_engine_cache() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.onnx");
        let cache_path = temp_dir.path().join("cache");

        let config = TensorRTBackendConfig::latency_optimized(model_path.clone())
            .with_engine_cache(cache_path.clone());

        assert_eq!(config.engine_cache_path, Some(cache_path));
    }
}
