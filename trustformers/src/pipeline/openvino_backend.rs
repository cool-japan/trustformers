// OpenVINO Pipeline Backend Integration for TrustformeRS
// Provides seamless Intel OpenVINO integration with the existing pipeline system

use crate::core::traits::TokenizedInput;
use crate::error::{Result, TrustformersError};
use crate::pipeline::{BasePipeline, Device, Pipeline, PipelineOptions, PipelineOutput};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use trustformers_core::errors::{Result as CoreResult, TrustformersError as CoreTrustformersError};
use trustformers_core::traits::{Model, Tokenizer};
// # Why every real code path here fails
//
// Intel's OpenVINO is a closed-source C++ runtime; there is no pure-Rust
// implementation of it anywhere in this workspace (compare
// `trustformers_core::export::openvino`, which only *writes* an IR pair from a
// model's parameters and refuses to invent one — see that module's docs). This
// file used to construct a `OpenVINOModel` that echoed its inputs back as
// "inference", reported a fixed `avg_latency_ms: 1.0` benchmark regardless of
// what ran, and answered device queries with `{"type": "mock"}`. None of that
// survives: every method that would need the real runtime now returns a
// structured [`TrustformersError::FeatureUnavailable`] naming it, and no
// method fabricates a tensor, a timing, or a device fact.
const OPENVINO_UNAVAILABLE_REASON: &str =
    "OpenVINO inference requires Intel's OpenVINO runtime (C++), which this pure-Rust build \
     does not link. No pure-Rust OpenVINO IR interpreter exists in trustformers-core.";

fn openvino_unavailable(feature: &str) -> TrustformersError {
    TrustformersError::FeatureUnavailable {
        message: OPENVINO_UNAVAILABLE_REASON.to_string(),
        feature: feature.to_string(),
        suggestion: Some(
            "Export the model to ONNX and use trustformers::pipeline::onnx_backend instead; \
             it runs on a real pure-Rust CPU interpreter."
                .to_string(),
        ),
        alternatives: vec!["onnx_backend".to_string()],
    }
}

/// OpenVINO client handle. Holding one claims nothing about model or device
/// state — only [`OpenVINORuntime::load_model`] and friends would make (and
/// honestly refuse) that claim.
#[derive(Debug, Clone)]
pub struct OpenVINORuntime;

#[derive(Debug, Clone)]
pub struct OpenVINOConfig {
    pub device: OpenVINODevice,
    pub precision: OpenVINOPrecision,
    pub execution_mode: OpenVINOExecutionMode,
    pub num_threads: Option<usize>,
    pub num_streams: Option<usize>,
    pub enable_profiling: bool,
    pub cache_dir: Option<PathBuf>,
    pub performance_hint: String,
    pub execution_priority: String,
    pub inference_timeout: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct OpenVINOModel {
    model: Option<Arc<OpenVINOModelWrapper>>,
    config: OpenVINOBackendConfig,
    input_names: Vec<String>,
    output_names: Vec<String>,
    runtime: Arc<OpenVINORuntime>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OpenVINODevice {
    CPU,
    GPU(i32),
    VPU,
    AUTO,
}

#[derive(Debug, Clone, Copy)]
pub enum OpenVINOPrecision {
    FP32,
    FP16,
    INT8,
}

#[derive(Debug, Clone, Copy)]
pub enum OpenVINOExecutionMode {
    Sync,
    Async,
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

impl OpenVINORuntime {
    /// Constructing a client handle claims nothing about a specific model or
    /// device, so this alone can honestly stay infallible.
    pub fn new(_config: OpenVINOConfig) -> Result<Self> {
        Ok(Self)
    }
}

impl Default for OpenVINORuntime {
    fn default() -> Self {
        OpenVINORuntime
    }
}

impl OpenVINORuntime {
    /// Always fails: see the module docs for why no build of this crate can
    /// actually load an OpenVINO IR model.
    pub fn load_model(&self, _path: &PathBuf) -> Result<OpenVINOModel> {
        Err(openvino_unavailable("openvino_model_load"))
    }

    pub fn load_model_with_weights(
        &self,
        model_path: &PathBuf,
        _weights_path: &PathBuf,
    ) -> Result<OpenVINOModel> {
        self.load_model(model_path)
    }

    /// Always fails: there is no OpenVINO runtime here to enumerate devices
    /// with. The old `vec!["CPU", "GPU"]` claimed hardware nothing here can
    /// actually drive.
    pub fn get_available_devices(&self) -> Result<Vec<String>> {
        Err(openvino_unavailable("device_enumeration"))
    }

    pub fn get_device_properties(&self, _device: &str) -> Result<HashMap<String, String>> {
        Err(openvino_unavailable("device_properties"))
    }
}

impl OpenVINOModel {
    pub fn new_mock() -> Self {
        Self::default()
    }

    pub fn new_minimal() -> Self {
        use std::sync::Arc;
        let config = OpenVINOBackendConfig::default();
        let input_names = vec!["input_ids".to_string()];
        let output_names = vec!["logits".to_string()];
        let runtime = Arc::new(OpenVINORuntime);

        // Create a simple model without circular references
        OpenVINOModel {
            model: None, // Simplified: no circular dependency
            config,
            input_names,
            output_names,
            runtime,
        }
    }

    pub fn new_base_minimal() -> Self {
        use std::sync::Arc;
        let config = OpenVINOBackendConfig::default();
        let input_names = vec!["input_ids".to_string()];
        let output_names = vec!["logits".to_string()];
        let runtime = Arc::new(OpenVINORuntime);

        // Create a truly minimal model that doesn't reference itself
        OpenVINOModel {
            model: None, // Simplified: no circular dependency
            config,
            input_names,
            output_names,
            runtime,
        }
    }

    pub fn new_leaf() -> Self {
        use std::sync::Arc;
        let config = OpenVINOBackendConfig::default();
        let input_names = vec!["input_ids".to_string()];
        let output_names = vec!["logits".to_string()];
        let runtime = Arc::new(OpenVINORuntime);

        // Create a leaf model that doesn't reference any other models
        OpenVINOModel {
            model: None, // Simplified: no circular dependency
            config,
            input_names,
            output_names,
            runtime,
        }
    }

    /// Create a mock model with a wrapper but no further nesting (no circular reference).
    pub fn new_with_wrapper() -> Self {
        OpenVINOModel {
            model: Some(Arc::new(OpenVINOModelWrapper::new_base())),
            config: OpenVINOBackendConfig::default(),
            input_names: vec!["input_ids".to_string()],
            output_names: vec!["logits".to_string()],
            runtime: Arc::new(OpenVINORuntime),
        }
    }

    /// Create the simplest possible model without any nested wrappers.
    pub fn new_empty_model() -> Self {
        OpenVINOModel {
            model: None,
            config: OpenVINOBackendConfig::default(),
            input_names: vec!["input_ids".to_string()],
            output_names: vec!["logits".to_string()],
            runtime: Arc::new(OpenVINORuntime),
        }
    }

    pub fn new_absolute_final() -> Self {
        // Create the most basic model without any wrapper complications
        OpenVINOModel {
            model: Some(Arc::new(OpenVINOModelWrapper::new_absolute_final())),
            config: OpenVINOBackendConfig::default(),
            input_names: vec!["input_ids".to_string()],
            output_names: vec!["logits".to_string()],
            runtime: Arc::new(OpenVINORuntime),
        }
    }

    pub fn new_ultimate_final() -> Self {
        // Create the ultimate final model that ends the chain
        OpenVINOModel {
            model: Some(Arc::new(OpenVINOModelWrapper::new_ultimate_final())),
            config: OpenVINOBackendConfig::default(),
            input_names: vec!["input_ids".to_string()],
            output_names: vec!["logits".to_string()],
            runtime: Arc::new(OpenVINORuntime),
        }
    }

    pub fn new_chain_breaker() -> Self {
        // Create a chain breaker model that doesn't create any more nested structures
        OpenVINOModel {
            model: Some(Arc::new(OpenVINOModelWrapper::new_simple())),
            config: OpenVINOBackendConfig::default(),
            input_names: vec!["input_ids".to_string()],
            output_names: vec!["logits".to_string()],
            runtime: Arc::new(OpenVINORuntime),
        }
    }

    pub fn new_placeholder() -> Self {
        // Create a placeholder model that doesn't create any nested structures
        OpenVINOModel {
            model: None,
            config: OpenVINOBackendConfig::default(),
            input_names: vec!["input_ids".to_string()],
            output_names: vec!["logits".to_string()],
            runtime: Arc::new(OpenVINORuntime),
        }
    }
}

impl Default for OpenVINOModel {
    fn default() -> Self {
        // Simple default: no nested wrapper to avoid deeply nested structures
        OpenVINOModel {
            model: None,
            config: OpenVINOBackendConfig::default(),
            input_names: vec!["input_ids".to_string()],
            output_names: vec!["logits".to_string()],
            runtime: Arc::new(OpenVINORuntime),
        }
    }
}

impl OpenVINOModel {
    /// Always fails: the old body returned `Tensor::zeros(&[1, 10])` as
    /// "inference". `self.model` is always `None` (only
    /// [`OpenVINORuntime::load_model`] would populate it, and that always
    /// errors), so there is nothing here to run inference with.
    pub fn infer(
        &self,
        _inputs: HashMap<String, trustformers_core::tensor::Tensor>,
    ) -> Result<HashMap<String, trustformers_core::tensor::Tensor>> {
        Err(openvino_unavailable("openvino_inference"))
    }

    pub fn infer_with_device(
        &self,
        inputs: HashMap<String, trustformers_core::tensor::Tensor>,
        _device: OpenVINODevice,
    ) -> Result<HashMap<String, trustformers_core::tensor::Tensor>> {
        self.infer(inputs)
    }

    pub async fn infer_async(
        &self,
        inputs: HashMap<String, trustformers_core::tensor::Tensor>,
    ) -> Result<HashMap<String, trustformers_core::tensor::Tensor>> {
        self.infer(inputs)
    }

    /// Always fails: there is no real execution to time.
    pub fn benchmark_mock(
        &self,
        _inputs: HashMap<String, trustformers_core::tensor::Tensor>,
        _num_runs: usize,
        _warmup_runs: usize,
    ) -> Result<BenchmarkResults> {
        Err(openvino_unavailable("openvino_benchmark"))
    }

    /// Always fails: there is no loaded OpenVINO model to report memory for.
    pub fn get_memory_info(&self) -> Result<MemoryInfo> {
        Err(openvino_unavailable("memory_info"))
    }

    /// The nested model wrapper, when this instance was built with one (see
    /// [`Self::new_with_wrapper`] and friends). `infer`/`benchmark_mock`
    /// never consult it — there is no real execution engine to hand it to —
    /// so this is purely for callers that want to inspect what a
    /// wrapper-carrying constructor actually built.
    pub fn wrapper(&self) -> Option<&OpenVINOModelWrapper> {
        self.model.as_deref()
    }
}
use trustformers_core::tensor::Tensor;

/// OpenVINO backend configuration for pipelines
#[derive(Debug, Clone)]
pub struct OpenVINOBackendConfig {
    pub model_path: PathBuf,
    pub weights_path: Option<PathBuf>,
    pub device: OpenVINODevice,
    pub precision: OpenVINOPrecision,
    pub execution_mode: OpenVINOExecutionMode,
    pub num_threads: Option<usize>,
    pub num_streams: Option<usize>,
    pub enable_profiling: bool,
    pub cache_dir: Option<PathBuf>,
    pub performance_hint: PerformanceHint,
    pub execution_priority: ExecutionPriority,
    pub inference_timeout: Option<u64>,
}

/// Performance optimization hints for OpenVINO
#[derive(Debug, Clone, Copy)]
pub enum PerformanceHint {
    /// Optimize for latency (default)
    Latency,
    /// Optimize for throughput
    Throughput,
    /// Optimize for cumulative throughput
    CumulativeThroughput,
    /// No specific hint
    None,
}

/// Execution priority levels
#[derive(Debug, Clone, Copy)]
pub enum ExecutionPriority {
    Low,
    Medium,
    High,
}

impl Default for OpenVINOBackendConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            weights_path: None,
            device: OpenVINODevice::CPU,
            precision: OpenVINOPrecision::FP32,
            execution_mode: OpenVINOExecutionMode::Sync,
            num_threads: None,
            num_streams: Some(1),
            enable_profiling: false,
            cache_dir: None,
            performance_hint: PerformanceHint::Latency,
            execution_priority: ExecutionPriority::Medium,
            inference_timeout: None,
        }
    }
}

impl OpenVINOBackendConfig {
    /// Create config optimized for CPU inference
    pub fn cpu_optimized(model_path: PathBuf) -> Self {
        Self {
            model_path,
            weights_path: None,
            device: OpenVINODevice::CPU,
            precision: OpenVINOPrecision::FP32,
            execution_mode: OpenVINOExecutionMode::Sync,
            num_threads: Some(num_cpus::get()),
            num_streams: Some(1),
            enable_profiling: false,
            cache_dir: None,
            performance_hint: PerformanceHint::Latency,
            execution_priority: ExecutionPriority::Medium,
            inference_timeout: None,
        }
    }

    /// Create config optimized for GPU inference
    pub fn gpu_optimized(model_path: PathBuf, device_id: Option<i32>) -> Self {
        Self {
            model_path,
            weights_path: None,
            device: OpenVINODevice::GPU(device_id.unwrap_or(0)),
            precision: OpenVINOPrecision::FP16, // FP16 is more efficient on GPU
            execution_mode: OpenVINOExecutionMode::Async,
            num_threads: Some(1),
            num_streams: Some(2), // Better for GPU parallelism
            enable_profiling: false,
            cache_dir: None,
            performance_hint: PerformanceHint::Throughput,
            execution_priority: ExecutionPriority::High,
            inference_timeout: None,
        }
    }

    /// Create config optimized for Intel integrated GPU
    pub fn igpu_optimized(model_path: PathBuf) -> Self {
        Self {
            model_path,
            weights_path: None,
            device: OpenVINODevice::GPU(0),
            precision: OpenVINOPrecision::FP16,
            execution_mode: OpenVINOExecutionMode::Sync,
            num_threads: Some(1),
            num_streams: Some(1),
            enable_profiling: false,
            cache_dir: None,
            performance_hint: PerformanceHint::Latency,
            execution_priority: ExecutionPriority::Medium,
            inference_timeout: None,
        }
    }

    /// Create config optimized for VPU (Movidius/Myriad) inference
    pub fn vpu_optimized(model_path: PathBuf) -> Self {
        Self {
            model_path,
            weights_path: None,
            device: OpenVINODevice::VPU,
            precision: OpenVINOPrecision::FP16, // VPU works best with FP16
            execution_mode: OpenVINOExecutionMode::Sync,
            num_threads: Some(1),
            num_streams: Some(1),
            enable_profiling: false,
            cache_dir: None,
            performance_hint: PerformanceHint::Latency,
            execution_priority: ExecutionPriority::Medium,
            inference_timeout: Some(10000), // 10 second timeout for VPU
        }
    }

    /// Create config for production deployment
    pub fn production(model_path: PathBuf) -> Self {
        Self {
            model_path,
            weights_path: None,
            device: OpenVINODevice::AUTO, // Auto-detect best device
            precision: OpenVINOPrecision::FP32,
            execution_mode: OpenVINOExecutionMode::Async,
            num_threads: Some(num_cpus::get()),
            num_streams: Some(2),
            enable_profiling: false,
            cache_dir: Some(std::env::temp_dir().join("openvino_cache")),
            performance_hint: PerformanceHint::Throughput,
            execution_priority: ExecutionPriority::High,
            inference_timeout: Some(30000), // 30 second timeout
        }
    }

    /// Create config for low-latency applications
    pub fn low_latency(model_path: PathBuf) -> Self {
        Self {
            model_path,
            weights_path: None,
            device: OpenVINODevice::CPU,
            precision: OpenVINOPrecision::FP32,
            execution_mode: OpenVINOExecutionMode::Sync,
            num_threads: Some(1), // Single thread for lower latency
            num_streams: Some(1),
            enable_profiling: false,
            cache_dir: None,
            performance_hint: PerformanceHint::Latency,
            execution_priority: ExecutionPriority::High,
            inference_timeout: None,
        }
    }

    /// Create config for high-throughput applications
    pub fn high_throughput(model_path: PathBuf) -> Self {
        Self {
            model_path,
            weights_path: None,
            device: OpenVINODevice::CPU,
            precision: OpenVINOPrecision::FP32,
            execution_mode: OpenVINOExecutionMode::Async,
            num_threads: Some(num_cpus::get()),
            num_streams: Some(num_cpus::get()), // More streams for throughput
            enable_profiling: false,
            cache_dir: None,
            performance_hint: PerformanceHint::Throughput,
            execution_priority: ExecutionPriority::Medium,
            inference_timeout: None,
        }
    }

    /// Enable profiling with optional cache directory
    pub fn with_profiling(mut self, cache_dir: Option<PathBuf>) -> Self {
        self.enable_profiling = true;
        self.cache_dir = cache_dir;
        self
    }

    /// Set weights file path (for .bin files)
    pub fn with_weights(mut self, weights_path: PathBuf) -> Self {
        self.weights_path = Some(weights_path);
        self
    }

    /// Set cache directory for compiled models
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = Some(cache_dir);
        self
    }

    /// Set inference timeout in milliseconds
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.inference_timeout = Some(timeout_ms);
        self
    }

    /// Convert to OpenVINO runtime config
    pub fn to_runtime_config(&self) -> OpenVINOConfig {
        OpenVINOConfig {
            device: self.device.clone(),
            precision: self.precision,
            execution_mode: self.execution_mode,
            num_threads: self.num_threads,
            num_streams: self.num_streams,
            enable_profiling: self.enable_profiling,
            cache_dir: self.cache_dir.clone(),
            performance_hint: match self.performance_hint {
                PerformanceHint::Latency => "LATENCY",
                PerformanceHint::Throughput => "THROUGHPUT",
                PerformanceHint::CumulativeThroughput => "CUMULATIVE_THROUGHPUT",
                PerformanceHint::None => "NONE",
            }
            .to_string(),
            execution_priority: match self.execution_priority {
                ExecutionPriority::Low => "LOW",
                ExecutionPriority::Medium => "MEDIUM",
                ExecutionPriority::High => "HIGH",
            }
            .to_string(),
            inference_timeout: self.inference_timeout,
        }
    }
}

/// OpenVINO-backed model wrapper
#[derive(Debug, Clone)]
pub struct OpenVINOModelWrapper {
    model: Option<Arc<OpenVINOModel>>,
    config: OpenVINOBackendConfig,
    input_names: Vec<String>,
    output_names: Vec<String>,
    runtime: Arc<OpenVINORuntime>,
}

impl Default for OpenVINOModelWrapper {
    fn default() -> Self {
        Self::new_empty()
    }
}

impl OpenVINOModelWrapper {
    /// The `OpenVINOModel` this wrapper was nested under, when constructed
    /// with one.
    pub fn inner_model(&self) -> Option<&OpenVINOModel> {
        self.model.as_deref()
    }

    /// The backend configuration this wrapper was built with.
    pub fn config(&self) -> &OpenVINOBackendConfig {
        &self.config
    }

    /// Input tensor names this wrapper reports.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Output tensor names this wrapper reports.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// The (always-unavailable, see [`OpenVINORuntime`]) runtime this
    /// wrapper is bound to.
    pub fn runtime(&self) -> &Arc<OpenVINORuntime> {
        &self.runtime
    }
}

impl OpenVINOModelWrapper {
    /// Create a wrapper with no inner model (simplest form, no nesting).
    pub fn new_empty() -> Self {
        OpenVINOModelWrapper {
            model: None,
            config: OpenVINOBackendConfig::default(),
            input_names: vec!["input_ids".to_string()],
            output_names: vec!["logits".to_string()],
            runtime: Arc::new(OpenVINORuntime),
        }
    }

    pub fn new_base() -> Self {
        use std::sync::Arc;
        let config = OpenVINOBackendConfig::default();
        let input_names = vec!["input_ids".to_string()];
        let output_names = vec!["logits".to_string()];
        let runtime = Arc::new(OpenVINORuntime);

        // Create a simple base wrapper without circular references
        OpenVINOModelWrapper {
            model: None, // Simplified: no circular dependency
            config,
            input_names,
            output_names,
            runtime,
        }
    }

    /// Create a wrapper that holds an empty model (no further nesting).
    pub fn new_with_empty_model() -> Self {
        OpenVINOModelWrapper {
            model: Some(Arc::new(OpenVINOModel::new_empty_model())),
            config: OpenVINOBackendConfig::default(),
            input_names: vec!["input_ids".to_string()],
            output_names: vec!["logits".to_string()],
            runtime: Arc::new(OpenVINORuntime),
        }
    }

    /// Create a wrapper holding only an inner model without further wrapper nesting.
    pub fn new_terminal() -> Self {
        OpenVINOModelWrapper {
            model: Some(Arc::new(OpenVINOModel::new_empty_model())),
            config: OpenVINOBackendConfig::default(),
            input_names: vec!["input_ids".to_string()],
            output_names: vec!["logits".to_string()],
            runtime: Arc::new(OpenVINORuntime),
        }
    }

    pub fn new_absolute_final() -> Self {
        use std::sync::Arc;
        let config = OpenVINOBackendConfig::default();
        let input_names = vec!["input_ids".to_string()];
        let output_names = vec!["logits".to_string()];
        let runtime = Arc::new(OpenVINORuntime);

        // Create the absolute final wrapper that doesn't create any more nested models
        OpenVINOModelWrapper {
            model: Some(Arc::new(OpenVINOModel::new_ultimate_final())),
            config,
            input_names,
            output_names,
            runtime,
        }
    }

    pub fn new_ultimate_final() -> Self {
        use std::sync::Arc;
        let config = OpenVINOBackendConfig::default();
        let input_names = vec!["input_ids".to_string()];
        let output_names = vec!["logits".to_string()];
        let runtime = Arc::new(OpenVINORuntime);

        // Create the ultimate final wrapper that ends the chain
        OpenVINOModelWrapper {
            model: Some(Arc::new(OpenVINOModel::new_chain_breaker())),
            config,
            input_names,
            output_names,
            runtime,
        }
    }

    pub fn new_simple() -> Self {
        use std::sync::Arc;
        let config = OpenVINOBackendConfig::default();
        let input_names = vec!["input_ids".to_string()];
        let output_names = vec!["logits".to_string()];
        let runtime = Arc::new(OpenVINORuntime);

        // Create a simple model that doesn't create any more nested structures -
        // we'll use a placeholder struct that doesn't actually do anything
        OpenVINOModelWrapper {
            model: Some(Arc::new(OpenVINOModel::new_placeholder())),
            config,
            input_names,
            output_names,
            runtime,
        }
    }
}

impl OpenVINOModel {
    /// Create new OpenVINO model from config
    pub fn from_config(config: OpenVINOBackendConfig) -> Result<Self> {
        if !config.model_path.exists() {
            return Err(TrustformersError::Core(CoreTrustformersError::other(
                format!(
                    "OpenVINO model file not found: {}",
                    config.model_path.to_string_lossy()
                ),
            )));
        }

        let runtime_config = config.to_runtime_config();
        let runtime = OpenVINORuntime::new(runtime_config)?;

        let model = if let Some(weights_path) = &config.weights_path {
            runtime.load_model_with_weights(&config.model_path, weights_path)?
        } else {
            runtime.load_model(&config.model_path)?
        };

        let input_names = model.input_names().to_vec();
        let output_names = model.output_names().to_vec();

        Ok(Self {
            model: None, // Simplified: avoid circular dependency
            config,
            input_names,
            output_names,
            runtime: Arc::new(runtime),
        })
    }

    /// Load from OpenVINO IR files with default config
    pub fn from_pretrained<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let config = OpenVINOBackendConfig {
            model_path: model_path.as_ref().to_path_buf(),
            ..Default::default()
        };
        Self::from_config(config)
    }

    /// Load with specific device
    pub fn from_pretrained_with_device<P: AsRef<Path>>(
        model_path: P,
        device: OpenVINODevice,
    ) -> Result<Self> {
        let config = OpenVINOBackendConfig {
            model_path: model_path.as_ref().to_path_buf(),
            device,
            ..Default::default()
        };
        Self::from_config(config)
    }

    /// Load from separate .xml and .bin files
    pub fn from_xml_bin<P: AsRef<Path>>(xml_path: P, bin_path: P) -> Result<Self> {
        let config = OpenVINOBackendConfig {
            model_path: xml_path.as_ref().to_path_buf(),
            weights_path: Some(bin_path.as_ref().to_path_buf()),
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

    /// Run inference. Always fails: see the module docs. The previous body
    /// echoed `inputs` straight back as "output", which downstream code (this
    /// pipeline's `sample_next_token`, softmax, ...) would happily decode into
    /// a confident-looking, entirely fabricated classification or generation.
    pub fn forward(&self, _inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        Err(openvino_unavailable("openvino_inference"))
    }

    /// Run inference with specific device. Always fails, for the same reason.
    pub fn forward_with_device(
        &self,
        _inputs: HashMap<String, Tensor>,
        _device: OpenVINODevice,
    ) -> Result<HashMap<String, Tensor>> {
        Err(openvino_unavailable("openvino_inference"))
    }

    /// Run asynchronous inference
    pub async fn forward_async(
        &self,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        self.infer_async(inputs).await
    }

    /// Benchmark the model. Always fails: with no real backend there is
    /// nothing to time. The previous body reported a fixed `1.0 ms` /
    /// `1000.0 ops/sec` regardless of `inputs`, `num_runs`, or the host.
    pub fn benchmark(
        &self,
        _inputs: HashMap<String, Tensor>,
        _num_runs: usize,
        _warmup_runs: usize,
    ) -> Result<BenchmarkResults> {
        Err(openvino_unavailable("openvino_benchmark"))
    }

    /// Get memory usage information. Always fails: there is no loaded model
    /// to report memory for.
    pub fn memory_info(&self) -> Result<MemoryInfo> {
        Err(openvino_unavailable("memory_info"))
    }

    /// Get supported devices
    pub fn supported_devices(&self) -> Result<Vec<String>> {
        self.runtime
            .get_available_devices()
            .map_err(|e| TrustformersError::FeatureUnavailable {
                message: format!("Failed to get available devices: {}", e),
                feature: "device_enumeration".to_string(),
                suggestion: Some("Check OpenVINO runtime installation and permissions".to_string()),
                alternatives: vec!["Use default CPU device".to_string()],
            })
    }

    /// Get device properties
    pub fn device_properties(&self, device: &str) -> Result<HashMap<String, String>> {
        self.runtime.get_device_properties(device).map_err(|e| {
            TrustformersError::FeatureUnavailable {
                message: format!("Failed to get device properties: {}", e),
                feature: "device_properties".to_string(),
                suggestion: Some("Check device availability and OpenVINO runtime".to_string()),
                alternatives: vec!["Use default device settings".to_string()],
            }
        })
    }

    /// Get model path
    pub fn model_path(&self) -> &Path {
        &self.config.model_path
    }

    /// Get current device
    pub fn device(&self) -> &OpenVINODevice {
        &self.config.device
    }

    /// Get current precision
    pub fn precision(&self) -> OpenVINOPrecision {
        self.config.precision
    }

    /// Create model optimized for specific use case
    pub fn optimize_for_device(&self, device: OpenVINODevice) -> Result<Self> {
        let mut config = self.config.clone();
        config.device = device.clone();

        // Adjust other settings based on device
        match device {
            OpenVINODevice::CPU => {
                config.precision = OpenVINOPrecision::FP32;
                config.num_threads = Some(num_cpus::get());
                config.num_streams = Some(1);
                config.performance_hint = PerformanceHint::Latency;
            },
            OpenVINODevice::GPU(_) => {
                config.precision = OpenVINOPrecision::FP16;
                config.num_streams = Some(2);
                config.performance_hint = PerformanceHint::Throughput;
            },
            OpenVINODevice::VPU => {
                config.precision = OpenVINOPrecision::FP16;
                config.num_streams = Some(1);
                config.performance_hint = PerformanceHint::Latency;
                config.inference_timeout = Some(10000);
            },
            OpenVINODevice::AUTO => {
                config.precision = OpenVINOPrecision::FP32;
                config.num_streams = Some(2);
                config.performance_hint = PerformanceHint::Throughput;
            },
        }

        Self::from_config(config)
    }
}

// Placeholder Config for OpenVINO mock implementation
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct OpenVINOModelConfig {
    pub backend: String,
}

impl crate::core::traits::Config for OpenVINOModelConfig {
    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "openvino"
    }
}

impl Model for OpenVINOModel {
    type Config = OpenVINOModelConfig;
    type Input = HashMap<String, Tensor>;
    type Output = HashMap<String, Tensor>;

    /// Forward pass implementation for Model trait. Always fails: see the
    /// module docs for why no OpenVINO execution path exists in this build.
    fn forward(&self, _inputs: Self::Input) -> CoreResult<Self::Output> {
        Err(CoreTrustformersError::other(
            OPENVINO_UNAVAILABLE_REASON.to_string(),
        ))
    }

    /// Load pretrained weights. No-op: an `OpenVINOModel` can never be
    /// constructed with real state in this build (see [`OpenVINOModel::from_config`]),
    /// so there is nothing for this to load into.
    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> CoreResult<()> {
        Ok(())
    }

    /// Get model configuration
    fn get_config(&self) -> &Self::Config {
        // Return a static default config for the mock implementation
        static CONFIG: OpenVINOModelConfig = OpenVINOModelConfig {
            backend: String::new(),
        };
        &CONFIG
    }

    /// Get the number of parameters in the model.
    ///
    /// Honestly `0`: no build of this crate can load real OpenVINO IR weights
    /// (see [`OpenVINOModel::from_config`]), so there are never any real
    /// parameters behind an `OpenVINOModel` value to count.
    fn num_parameters(&self) -> usize {
        0
    }
}

/// OpenVINO tokenizer wrapper (can wrap existing tokenizers)
#[derive(Clone)]
pub struct OpenVINOTokenizer<T> {
    inner: T,
}

impl<T: Tokenizer> OpenVINOTokenizer<T> {
    pub fn new(tokenizer: T) -> Self {
        Self { inner: tokenizer }
    }
}

impl<T: Tokenizer> Tokenizer for OpenVINOTokenizer<T> {
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

/// OpenVINO-backed pipeline base
pub type OpenVINOBasePipeline<T> = BasePipeline<OpenVINOModel, OpenVINOTokenizer<T>>;

/// Text classification pipeline with OpenVINO backend
pub struct OpenVINOTextClassificationPipeline<T> {
    base: OpenVINOBasePipeline<T>,
    return_all_scores: bool,
}

impl<T: Tokenizer + Clone> OpenVINOTextClassificationPipeline<T> {
    pub fn new(model: OpenVINOModel, tokenizer: OpenVINOTokenizer<T>) -> Result<Self> {
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
    pub fn benchmark(
        &self,
        input: &str,
        num_runs: usize,
        warmup_runs: usize,
    ) -> Result<BenchmarkResults> {
        let tokenized = self.base.tokenizer.encode(input)?;
        let inputs = self.prepare_inputs(&tokenized)?;
        self.base.model.benchmark(inputs, num_runs, warmup_runs)
    }

    /// Get memory usage
    pub fn memory_info(&self) -> Result<MemoryInfo> {
        self.base.model.memory_info()
    }

    /// Get supported devices
    pub fn supported_devices(&self) -> Result<Vec<String>> {
        self.base.model.supported_devices()
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
        let attention_mask = if !tokenized.attention_mask.is_empty() {
            let mask = &tokenized.attention_mask;
            Tensor::from_vec(
                mask.iter().map(|&x| x as f32).collect(),
                &[batch_size, seq_len],
            )?
        } else {
            Tensor::from_vec(vec![1.0f32; batch_size * seq_len], &[batch_size, seq_len])?
        };
        inputs.insert("attention_mask".to_string(), attention_mask);

        Ok(inputs)
    }
}

impl<T: Tokenizer + Clone> Pipeline for OpenVINOTextClassificationPipeline<T> {
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
                .ok_or_else(|| {
                    TrustformersError::invalid_input(
                        "No probabilities available for classification",
                        Some("model_logits"),
                        Some("at least one probability value"),
                        Some("empty probability vector"),
                    )
                })?;
            results.push(crate::pipeline::ClassificationOutput {
                label: format!("LABEL_{}", max_idx),
                score: max_score,
            });
        }

        Ok(PipelineOutput::Classification(results))
    }
}

/// Text generation pipeline with OpenVINO backend
pub struct OpenVINOTextGenerationPipeline<T> {
    base: OpenVINOBasePipeline<T>,
    max_new_tokens: usize,
    do_sample: bool,
    temperature: f32,
    top_p: f32,
    use_async: bool,
}

impl<T: Tokenizer + Clone> OpenVINOTextGenerationPipeline<T> {
    pub fn new(model: OpenVINOModel, tokenizer: OpenVINOTokenizer<T>) -> Result<Self> {
        Ok(Self {
            base: BasePipeline::new(model, tokenizer),
            max_new_tokens: 50,
            do_sample: false,
            temperature: 1.0,
            top_p: 1.0,
            use_async: false,
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

    pub fn with_async(mut self, use_async: bool) -> Self {
        self.use_async = use_async;
        self
    }

    /// Generate text asynchronously
    pub async fn generate_async(&self, input: String) -> Result<PipelineOutput> {
        let tokenized = self.base.tokenizer.encode(&input)?;
        let mut input_ids = tokenized.input_ids.clone();

        // Simple autoregressive generation with async inference
        for _ in 0..self.max_new_tokens {
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

            let outputs = self.base.model.forward_async(inputs).await?;
            let logits = outputs.into_values().next().ok_or_else(|| {
                TrustformersError::invalid_input(
                    "No logits output found in model inference results".to_string(),
                    Some("model_outputs".to_string()),
                    Some("tensor with logits".to_string()),
                    Some("empty outputs".to_string()),
                )
            })?;

            // Get next token (simplified greedy decoding)
            let next_token = self.sample_next_token(&logits, seq_len)?;
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

    fn sample_next_token(&self, logits: &Tensor, seq_len: usize) -> Result<u32> {
        let logits_data = logits.data()?;
        let vocab_size = logits_data.len() / seq_len;
        let last_token_logits = &logits_data[(seq_len - 1) * vocab_size..seq_len * vocab_size];

        let next_token = if self.do_sample {
            // Apply temperature and sampling
            let scaled_logits: Vec<f32> =
                last_token_logits.iter().map(|&x| x / self.temperature).collect();
            let max_logit = scaled_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_logits: Vec<f32> =
                scaled_logits.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_exp: f32 = exp_logits.iter().sum();
            let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

            // Simple random sampling
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            logits_data.len().hash(&mut hasher);
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
                .ok_or_else(|| {
                    TrustformersError::invalid_input(
                        "No logits available for token selection",
                        Some("token_logits"),
                        Some("at least one logit value"),
                        Some("empty logits vector"),
                    )
                })?
        };

        Ok(next_token)
    }
}

impl<T: Tokenizer + Clone> Pipeline for OpenVINOTextGenerationPipeline<T> {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        let tokenized = self.base.tokenizer.encode(&input)?;
        let mut input_ids = tokenized.input_ids.clone();

        // Simple autoregressive generation
        for _ in 0..self.max_new_tokens {
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

            let outputs = self.base.model.forward(inputs)?;
            let logits = outputs.into_values().next().ok_or_else(|| {
                TrustformersError::invalid_input(
                    "No logits output found in model inference results".to_string(),
                    Some("model_outputs".to_string()),
                    Some("tensor with logits".to_string()),
                    Some("empty outputs".to_string()),
                )
            })?;

            // Get next token
            let next_token = self.sample_next_token(&logits, seq_len)?;
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

/// Factory functions for OpenVINO pipelines
pub fn openvino_text_classification_pipeline<T: Tokenizer + Clone>(
    model_path: impl AsRef<Path>,
    tokenizer: T,
    config: Option<OpenVINOBackendConfig>,
) -> Result<OpenVINOTextClassificationPipeline<T>> {
    let config = config
        .unwrap_or_else(|| OpenVINOBackendConfig::cpu_optimized(model_path.as_ref().to_path_buf()));
    let model = OpenVINOModel::from_config(config)?;
    let openvino_tokenizer = OpenVINOTokenizer::new(tokenizer);
    OpenVINOTextClassificationPipeline::new(model, openvino_tokenizer)
}

pub fn openvino_text_generation_pipeline<T: Tokenizer + Clone>(
    model_path: impl AsRef<Path>,
    tokenizer: T,
    config: Option<OpenVINOBackendConfig>,
) -> Result<OpenVINOTextGenerationPipeline<T>> {
    let config = config
        .unwrap_or_else(|| OpenVINOBackendConfig::cpu_optimized(model_path.as_ref().to_path_buf()));
    let model = OpenVINOModel::from_config(config)?;
    let openvino_tokenizer = OpenVINOTokenizer::new(tokenizer);
    OpenVINOTextGenerationPipeline::new(model, openvino_tokenizer)
}

/// Enhanced pipeline options with OpenVINO backend support
#[derive(Clone, Debug)]
pub struct OpenVINOPipelineOptions {
    pub base_options: PipelineOptions,
    pub openvino_config: OpenVINOBackendConfig,
    pub enable_profiling: bool,
    pub warmup_runs: usize,
    pub enable_async: bool,
}

impl Default for OpenVINOPipelineOptions {
    fn default() -> Self {
        Self {
            base_options: PipelineOptions::default(),
            openvino_config: OpenVINOBackendConfig::default(),
            enable_profiling: false,
            warmup_runs: 3,
            enable_async: false,
        }
    }
}

impl OpenVINOPipelineOptions {
    pub fn cpu_optimized(model_path: PathBuf) -> Self {
        Self {
            base_options: PipelineOptions::default(),
            openvino_config: OpenVINOBackendConfig::cpu_optimized(model_path),
            enable_profiling: false,
            warmup_runs: 3,
            enable_async: false,
        }
    }

    pub fn gpu_optimized(model_path: PathBuf, device_id: Option<i32>) -> Self {
        Self {
            base_options: PipelineOptions {
                device: Some(Device::Gpu(device_id.unwrap_or(0) as usize)),
                ..Default::default()
            },
            openvino_config: OpenVINOBackendConfig::gpu_optimized(model_path, device_id),
            enable_profiling: false,
            warmup_runs: 3,
            enable_async: true, // Enable async for GPU
        }
    }

    pub fn vpu_optimized(model_path: PathBuf) -> Self {
        Self {
            base_options: PipelineOptions::default(),
            openvino_config: OpenVINOBackendConfig::vpu_optimized(model_path),
            enable_profiling: false,
            warmup_runs: 1, // VPU doesn't need many warmup runs
            enable_async: false,
        }
    }

    pub fn production(model_path: PathBuf) -> Self {
        Self {
            base_options: PipelineOptions::default(),
            openvino_config: OpenVINOBackendConfig::production(model_path),
            enable_profiling: false,
            warmup_runs: 5,
            enable_async: true,
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

    pub fn with_async(mut self, enable: bool) -> Self {
        self.enable_async = enable;
        self
    }
}

/// OpenVINO pipeline manager for coordinating multiple backends
pub struct OpenVINOPipelineManager {
    models: HashMap<String, OpenVINOModel>,
    default_config: OpenVINOBackendConfig,
    runtime: Arc<OpenVINORuntime>,
}

impl OpenVINOPipelineManager {
    pub fn new(default_config: OpenVINOBackendConfig) -> Result<Self> {
        let runtime_config = default_config.to_runtime_config();
        let runtime = OpenVINORuntime::new(runtime_config)?;

        Ok(Self {
            models: HashMap::new(),
            default_config,
            runtime: Arc::new(runtime),
        })
    }

    /// Register a model with the manager
    pub fn register_model(&mut self, name: String, model: OpenVINOModel) {
        self.models.insert(name, model);
    }

    /// Load and register a model from path
    pub fn load_model<P: AsRef<Path>>(&mut self, name: String, model_path: P) -> Result<()> {
        let mut config = self.default_config.clone();
        config.model_path = model_path.as_ref().to_path_buf();
        let model = OpenVINOModel::from_config(config)?;
        self.register_model(name, model);
        Ok(())
    }

    /// Load and register a model from XML/BIN files
    pub fn load_model_xml_bin<P: AsRef<Path>>(
        &mut self,
        name: String,
        xml_path: P,
        bin_path: P,
    ) -> Result<()> {
        let model = OpenVINOModel::from_xml_bin(xml_path, bin_path)?;
        self.register_model(name, model);
        Ok(())
    }

    /// Get a registered model
    pub fn get_model(&self, name: &str) -> Option<&OpenVINOModel> {
        self.models.get(name)
    }

    /// List all registered models
    pub fn list_models(&self) -> Vec<&String> {
        self.models.keys().collect()
    }

    /// Get available devices
    pub fn available_devices(&self) -> Result<Vec<String>> {
        self.runtime
            .get_available_devices()
            .map_err(|e| TrustformersError::FeatureUnavailable {
                message: format!("Failed to get available devices: {}", e),
                feature: "device_enumeration".to_string(),
                suggestion: Some("Check OpenVINO runtime installation and permissions".to_string()),
                alternatives: vec!["Use default CPU device".to_string()],
            })
    }

    /// Get device properties
    pub fn device_properties(&self, device: &str) -> Result<HashMap<String, String>> {
        self.runtime.get_device_properties(device).map_err(|e| {
            TrustformersError::FeatureUnavailable {
                message: format!("Failed to get device properties: {}", e),
                feature: "device_properties".to_string(),
                suggestion: Some("Check device availability and OpenVINO runtime".to_string()),
                alternatives: vec!["Use default device settings".to_string()],
            }
        })
    }

    /// Benchmark all registered models
    pub fn benchmark_all(
        &self,
        inputs: HashMap<String, Tensor>,
        num_runs: usize,
    ) -> Result<HashMap<String, BenchmarkResults>> {
        let mut results = HashMap::new();
        for (name, model) in &self.models {
            let benchmark = model.benchmark(inputs.clone(), num_runs, 3)?;
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

    /// Optimize all models for specific device
    pub fn optimize_all_for_device(&mut self, device: OpenVINODevice) -> Result<()> {
        let model_names: Vec<String> = self.models.keys().cloned().collect();
        for name in model_names {
            if let Some(model) = self.models.remove(&name) {
                let optimized_model = model.optimize_for_device(device.clone())?;
                self.models.insert(name, optimized_model);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_openvino_model_wrapper_accessor_returns_nested_wrapper() {
        let model = OpenVINOModel::new_with_wrapper();
        let wrapper = model.wrapper().expect("new_with_wrapper must nest a wrapper");
        assert_eq!(
            wrapper.input_names().to_vec(),
            vec!["input_ids".to_string()]
        );
        assert_eq!(wrapper.output_names().to_vec(), vec!["logits".to_string()]);
        assert!(
            wrapper.inner_model().is_none(),
            "the nested wrapper must not itself nest a model"
        );
        let _ = wrapper.config();
        let _ = wrapper.runtime();
    }

    #[test]
    fn test_openvino_model_wrapper_accessor_is_none_without_a_wrapper() {
        let model = OpenVINOModel::new_empty_model();
        assert!(model.wrapper().is_none());
    }

    #[test]
    fn test_openvino_backend_config() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.xml");

        let config = OpenVINOBackendConfig::cpu_optimized(model_path.clone());
        assert_eq!(config.model_path, model_path);
        assert!(matches!(config.device, OpenVINODevice::CPU));
        assert!(matches!(config.precision, OpenVINOPrecision::FP32));

        let runtime_config = config.to_runtime_config();
        assert_eq!(runtime_config.device, OpenVINODevice::CPU);
    }

    #[test]
    fn test_openvino_backend_config_gpu() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.xml");

        let config = OpenVINOBackendConfig::gpu_optimized(model_path.clone(), Some(0));
        assert_eq!(config.model_path, model_path);
        assert!(matches!(config.device, OpenVINODevice::GPU(0)));
        assert!(matches!(config.precision, OpenVINOPrecision::FP16));
    }

    #[test]
    fn test_openvino_backend_config_vpu() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.xml");

        let config = OpenVINOBackendConfig::vpu_optimized(model_path.clone());
        assert_eq!(config.model_path, model_path);
        assert!(matches!(config.device, OpenVINODevice::VPU));
        assert!(matches!(config.precision, OpenVINOPrecision::FP16));
        assert!(config.inference_timeout.is_some());
    }

    #[test]
    fn test_openvino_pipeline_options() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.xml");

        let options = OpenVINOPipelineOptions::cpu_optimized(model_path.clone());
        assert_eq!(options.openvino_config.model_path, model_path);
        assert_eq!(options.warmup_runs, 3);
        assert!(!options.enable_async);

        let gpu_options = OpenVINOPipelineOptions::gpu_optimized(model_path.clone(), Some(0));
        assert!(matches!(
            gpu_options.base_options.device,
            Some(Device::Gpu(0))
        ));
        assert!(gpu_options.enable_async);
    }

    #[test]
    fn test_performance_hints() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.xml");

        let latency_config = OpenVINOBackendConfig::low_latency(model_path.clone());
        assert!(matches!(
            latency_config.performance_hint,
            PerformanceHint::Latency
        ));

        let throughput_config = OpenVINOBackendConfig::high_throughput(model_path.clone());
        assert!(matches!(
            throughput_config.performance_hint,
            PerformanceHint::Throughput
        ));
    }

    #[test]
    fn test_openvino_pipeline_manager() {
        let config = OpenVINOBackendConfig::default();
        // Note: This would fail without actual OpenVINO runtime
        // let manager = OpenVINOPipelineManager::new(config);
        // assert!(manager.is_ok());

        // For now, just test the config
        let runtime_config = config.to_runtime_config();
        assert!(matches!(runtime_config.device, OpenVINODevice::CPU));
    }

    #[test]
    fn test_config_with_weights() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.xml");
        let weights_path = temp_dir.path().join("model.bin");

        let config = OpenVINOBackendConfig::cpu_optimized(model_path.clone())
            .with_weights(weights_path.clone());

        assert_eq!(config.model_path, model_path);
        assert_eq!(config.weights_path, Some(weights_path));
    }

    #[test]
    fn test_config_with_cache() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.xml");
        let cache_dir = temp_dir.path().join("cache");

        let config = OpenVINOBackendConfig::cpu_optimized(model_path.clone())
            .with_cache_dir(cache_dir.clone());

        assert_eq!(config.cache_dir, Some(cache_dir));
    }

    #[test]
    fn test_config_with_timeout() {
        let temp_dir = tempdir().expect("failed to create temp dir");
        let model_path = temp_dir.path().join("model.xml");

        let config = OpenVINOBackendConfig::cpu_optimized(model_path.clone()).with_timeout(5000);

        assert_eq!(config.inference_timeout, Some(5000));
    }

    // ── Honesty regression tests ──────────────────────────────────────────────
    //
    // None of these can be driven through a *successfully constructed*
    // `OpenVINOModel`, because no build of this crate can construct one: that
    // is exactly the point being tested.

    /// Regression test for `OpenVINORuntime::load_model` unconditionally
    /// returning `Ok`.
    #[test]
    fn load_model_always_fails() {
        let runtime = OpenVINORuntime::new(OpenVINOConfig {
            device: OpenVINODevice::CPU,
            precision: OpenVINOPrecision::FP32,
            execution_mode: OpenVINOExecutionMode::Sync,
            num_threads: None,
            num_streams: None,
            enable_profiling: false,
            cache_dir: None,
            performance_hint: "LATENCY".to_string(),
            execution_priority: "MEDIUM".to_string(),
            inference_timeout: None,
        })
        .expect("client handle construction is infallible");

        let err = runtime
            .load_model(&PathBuf::from("model.xml"))
            .expect_err("no OpenVINO runtime is linked into this build");
        assert!(!err.to_string().is_empty());
    }

    /// Regression test for `get_available_devices` returning a fixed
    /// `["CPU", "GPU"]` regardless of what is actually installed.
    #[test]
    fn get_available_devices_is_honest_about_having_none() {
        let runtime = OpenVINORuntime;
        assert!(runtime.get_available_devices().is_err());
    }

    /// Regression test for `get_device_properties` returning `{"type": "mock"}`.
    #[test]
    fn get_device_properties_is_honest() {
        let runtime = OpenVINORuntime;
        assert!(runtime.get_device_properties("CPU").is_err());
    }

    /// Regression test: `from_config` must refuse every real model path, since
    /// no build of this crate can load one.
    #[test]
    fn from_config_always_fails_even_for_an_existing_file() {
        let temp_dir = tempdir().expect("temp dir");
        let model_path = temp_dir.path().join("model.xml");
        fs::write(&model_path, b"<net></net>").expect("write dummy IR file");

        let config = OpenVINOBackendConfig::cpu_optimized(model_path);
        let err = OpenVINOModel::from_config(config)
            .expect_err("no OpenVINO runtime is linked into this build");
        assert!(!err.to_string().is_empty());
    }

    /// Regression test: the public pipeline factory must fail loudly rather
    /// than build a classifier whose `__call__` only fails (or, previously,
    /// echoed inputs as fake logits) on first use.
    #[test]
    fn openvino_text_classification_pipeline_fails_loudly() {
        let temp_dir = tempdir().expect("temp dir");
        let model_path = temp_dir.path().join("model.xml");
        fs::write(&model_path, b"<net></net>").expect("write dummy IR file");

        struct StubTokenizer;
        impl Clone for StubTokenizer {
            fn clone(&self) -> Self {
                StubTokenizer
            }
        }
        impl Tokenizer for StubTokenizer {
            fn encode(&self, _text: &str) -> CoreResult<TokenizedInput> {
                Ok(TokenizedInput::new(vec![1, 2, 3], vec![1, 1, 1]))
            }
            fn encode_pair(&self, text: &str, _text2: &str) -> CoreResult<TokenizedInput> {
                self.encode(text)
            }
            fn decode(&self, ids: &[u32]) -> CoreResult<String> {
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

        let result = openvino_text_classification_pipeline(&model_path, StubTokenizer, None);
        assert!(result.is_err());
    }

    /// Regression test for `OpenVINOModel::forward` returning `Ok(inputs)`
    /// unchanged ("echo as inference") and for `benchmark`/`memory_info`/
    /// `infer` returning fixed mock numbers. Exercised directly against the
    /// zero-nesting `new_mock()` constructor, since `from_config` can never
    /// succeed (see above).
    #[test]
    fn model_methods_never_fabricate_output_or_metrics() {
        let model = OpenVINOModel::new_mock();

        let mut inputs = HashMap::new();
        inputs.insert(
            "input_ids".to_string(),
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).expect("tensor"),
        );

        let forward_err = model
            .forward(inputs.clone())
            .expect_err("forward must not echo inputs back as inference");
        assert!(!forward_err.to_string().is_empty());

        assert!(model.benchmark(inputs.clone(), 5, 2).is_err());
        assert!(model.memory_info().is_err());
        assert!(model.infer(inputs).is_err());
    }
}
