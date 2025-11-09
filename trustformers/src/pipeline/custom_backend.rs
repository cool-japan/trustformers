// Custom Backend API for TrustformeRS
// Provides an extensible framework for implementing custom inference backends

use crate::core::traits::{Model, Tokenizer};
use crate::error::{Result, TrustformersError};
use crate::pipeline::{Device, PaddingStrategy, PipelineOptions};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Trait for custom backend implementations
pub trait CustomBackend: Send + Sync + Debug {
    /// Backend name/identifier
    fn name(&self) -> &str;

    /// Backend version
    fn version(&self) -> &str;

    /// Initialize the backend with configuration
    fn initialize(&mut self, config: &BackendConfig) -> Result<()>;

    /// Load a model from the given path
    fn load_model(&self, path: &PathBuf) -> Result<Box<dyn BackendModel>>;

    /// Get supported device types
    fn supported_devices(&self) -> Vec<Device>;

    /// Get backend-specific capabilities
    fn capabilities(&self) -> BackendCapabilities;

    /// Validate backend health
    fn health_check(&self) -> Result<BackendHealth>;

    /// Get backend metrics
    fn get_metrics(&self) -> BackendMetrics;

    /// Cleanup resources
    fn cleanup(&mut self) -> Result<()>;

    /// Get backend as Any for dynamic casting
    fn as_any(&self) -> &dyn Any;
}

/// Trait for models loaded by custom backends
pub trait BackendModel: Send + Sync + Debug {
    /// Run inference on the model
    fn predict(
        &self,
        inputs: &HashMap<String, BackendTensor>,
    ) -> Result<HashMap<String, BackendTensor>>;

    /// Get model metadata
    fn metadata(&self) -> &ModelMetadata;

    /// Get input specifications
    fn input_specs(&self) -> &HashMap<String, TensorSpec>;

    /// Get output specifications
    fn output_specs(&self) -> &HashMap<String, TensorSpec>;

    /// Warm up the model for faster inference
    fn warmup(&self) -> Result<()>;

    /// Get model performance statistics
    fn performance_stats(&self) -> ModelPerformanceStats;

    /// Get model as Any for dynamic casting
    fn as_any(&self) -> &dyn Any;
}

/// Backend configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Backend name
    pub name: String,
    /// Device to use
    pub device: Device,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Memory settings
    pub memory_config: MemoryConfig,
    /// Performance settings
    pub performance_config: PerformanceConfig,
    /// Custom backend-specific settings
    pub custom_settings: HashMap<String, serde_json::Value>,
}

/// Backend capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    /// Supported data types
    pub supported_dtypes: Vec<DataType>,
    /// Supported operations
    pub supported_ops: Vec<String>,
    /// Maximum tensor dimensions
    pub max_dimensions: u32,
    /// Maximum batch size
    pub max_batch_size: Option<u32>,
    /// Dynamic shape support
    pub dynamic_shapes: bool,
    /// In-place operations support
    pub in_place_ops: bool,
    /// Quantization support
    pub quantization: Vec<QuantizationMode>,
    /// Memory mapping support
    pub memory_mapping: bool,
}

/// Backend health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendHealth {
    /// Overall status
    pub status: HealthStatus,
    /// Device availability
    pub device_available: bool,
    /// Memory status
    pub memory_usage: MemoryUsage,
    /// Last error if any
    pub last_error: Option<String>,
    /// Performance indicators
    pub performance_indicators: PerformanceIndicators,
}

/// Backend performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendMetrics {
    /// Total inference count
    pub total_inferences: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Throughput (inferences per second)
    pub throughput: f64,
    /// Memory statistics
    pub memory_stats: MemoryStats,
    /// Error rate
    pub error_rate: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Utilization percentage
    pub utilization_percent: f64,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model format
    pub format: String,
    /// Input shapes
    pub input_shapes: HashMap<String, Vec<i64>>,
    /// Output shapes
    pub output_shapes: HashMap<String, Vec<i64>>,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Number of parameters
    pub num_parameters: u64,
    /// Required memory in bytes
    pub memory_required: u64,
}

/// Tensor specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Tensor shape
    pub shape: Vec<i64>,
    /// Data type
    pub dtype: DataType,
    /// Memory layout
    pub layout: MemoryLayout,
    /// Optional constraints
    pub constraints: Option<TensorConstraints>,
}

/// Generic tensor representation for custom backends
#[derive(Debug, Clone)]
pub struct BackendTensor {
    /// Tensor data (generic byte buffer)
    pub data: Vec<u8>,
    /// Tensor shape
    pub shape: Vec<i64>,
    /// Data type
    pub dtype: DataType,
    /// Memory layout
    pub layout: MemoryLayout,
}

/// Backend registry for managing custom backends
pub struct BackendRegistry {
    backends: RwLock<HashMap<String, Arc<dyn CustomBackend>>>,
    factories: RwLock<HashMap<String, Box<dyn BackendFactory>>>,
}

/// Factory trait for creating backend instances
pub trait BackendFactory: Send + Sync + std::fmt::Debug {
    /// Create a new backend instance
    fn create_backend(&self, config: &BackendConfig) -> Result<Box<dyn CustomBackend>>;

    /// Get factory metadata
    fn factory_info(&self) -> FactoryInfo;
}

/// Custom pipeline implementation that uses custom backends
pub struct CustomBackendPipeline {
    backend: Arc<dyn CustomBackend>,
    model: Arc<dyn BackendModel>,
    tokenizer: Option<Arc<dyn Tokenizer>>,
    config: BackendConfig,
    options: PipelineOptions,
}

// Supporting enums and structs

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Standard,
    Aggressive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub max_memory_mb: Option<u64>,
    pub cache_size_mb: Option<u64>,
    pub prefetch_enabled: bool,
    pub memory_mapping: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub max_batch_size: Option<u32>,
    pub num_threads: Option<u32>,
    pub enable_profiling: bool,
    pub warmup_runs: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DataType {
    Float32,
    Float16,
    Int32,
    Int16,
    Int8,
    UInt8,
    Bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QuantizationMode {
    None,
    Dynamic,
    Static,
    QAT,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub total_mb: u64,
    pub used_mb: u64,
    pub available_mb: u64,
    pub fragmentation_percent: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceIndicators {
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub queue_depth: u32,
    pub active_requests: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub peak_usage_mb: u64,
    pub current_usage_mb: u64,
    pub allocations_count: u64,
    pub deallocations_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceStats {
    pub total_inferences: u64,
    pub avg_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub throughput: f64,
    pub memory_usage_mb: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
    NHWC,
    NCHW,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorConstraints {
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub positive_only: bool,
    pub normalized: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactoryInfo {
    pub name: String,
    pub version: String,
    pub description: String,
    pub supported_formats: Vec<String>,
    pub required_features: Vec<String>,
}

// Implementation of BackendRegistry
impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendRegistry {
    /// Create a new backend registry
    pub fn new() -> Self {
        Self {
            backends: RwLock::new(HashMap::new()),
            factories: RwLock::new(HashMap::new()),
        }
    }

    /// Register a backend factory
    pub fn register_factory(&self, name: String, factory: Box<dyn BackendFactory>) -> Result<()> {
        let mut factories = self.factories.write().map_err(|_| {
            trustformers_core::errors::runtime_error("Failed to acquire write lock for factories")
        })?;
        factories.insert(name, factory);
        Ok(())
    }

    /// Create and register a backend instance
    pub fn create_backend(&self, name: &str, config: &BackendConfig) -> Result<()> {
        let factories = self.factories.read().map_err(|_| {
            trustformers_core::errors::runtime_error("Failed to acquire read lock for factories")
        })?;

        let factory = factories.get(name).ok_or_else(|| {
            trustformers_core::errors::runtime_error(format!(
                "Backend factory '{}' not found",
                name
            ))
        })?;

        let backend = factory.create_backend(config)?;

        let mut backends = self.backends.write().map_err(|_| {
            trustformers_core::errors::runtime_error("Failed to acquire write lock for backends")
        })?;
        backends.insert(name.to_string(), Arc::from(backend));
        Ok(())
    }

    /// Get a backend instance
    pub fn get_backend(&self, name: &str) -> Result<Arc<dyn CustomBackend>> {
        let backends = self.backends.read().map_err(|_| {
            trustformers_core::errors::runtime_error("Failed to acquire read lock for backends")
        })?;

        backends.get(name).cloned().ok_or_else(|| {
            TrustformersError::runtime_error(format!("Backend '{}' not found", name))
        })
    }

    /// List all registered backend names
    pub fn list_backends(&self) -> Result<Vec<String>> {
        let backends = self.backends.read().map_err(|_| {
            trustformers_core::errors::runtime_error("Failed to acquire read lock for backends")
        })?;
        Ok(backends.keys().cloned().collect())
    }

    /// List all registered factory names
    pub fn list_factories(&self) -> Result<Vec<String>> {
        let factories = self.factories.read().map_err(|_| {
            trustformers_core::errors::runtime_error("Failed to acquire read lock for factories")
        })?;
        Ok(factories.keys().cloned().collect())
    }

    /// Remove a backend
    pub fn remove_backend(&self, name: &str) -> Result<()> {
        let mut backends = self.backends.write().map_err(|_| {
            trustformers_core::errors::runtime_error("Failed to acquire write lock for backends")
        })?;
        backends.remove(name);
        Ok(())
    }

    /// Get backend health status for all backends
    pub fn health_check_all(&self) -> Result<HashMap<String, BackendHealth>> {
        let backends = self.backends.read().map_err(|_| {
            trustformers_core::errors::runtime_error("Failed to acquire read lock for backends")
        })?;

        let mut health_map = HashMap::new();
        for (name, backend) in backends.iter() {
            match backend.health_check() {
                Ok(health) => {
                    health_map.insert(name.clone(), health);
                },
                Err(_) => {
                    health_map.insert(
                        name.clone(),
                        BackendHealth {
                            status: HealthStatus::Critical,
                            device_available: false,
                            memory_usage: MemoryUsage {
                                total_mb: 0,
                                used_mb: 0,
                                available_mb: 0,
                                fragmentation_percent: 0.0,
                            },
                            last_error: Some("Health check failed".to_string()),
                            performance_indicators: PerformanceIndicators {
                                latency_p50_ms: 0.0,
                                latency_p95_ms: 0.0,
                                latency_p99_ms: 0.0,
                                queue_depth: 0,
                                active_requests: 0,
                            },
                        },
                    );
                },
            }
        }
        Ok(health_map)
    }
}

// Implementation of CustomBackendPipeline
impl CustomBackendPipeline {
    /// Create a new custom backend pipeline
    pub fn new(
        backend: Arc<dyn CustomBackend>,
        model_path: &PathBuf,
        config: BackendConfig,
        options: PipelineOptions,
    ) -> Result<Self> {
        let model = backend.load_model(model_path)?;

        Ok(Self {
            backend,
            model: Arc::from(model),
            tokenizer: None,
            config,
            options,
        })
    }

    /// Set tokenizer for the pipeline
    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Get backend reference
    pub fn backend(&self) -> &Arc<dyn CustomBackend> {
        &self.backend
    }

    /// Get model reference
    pub fn model(&self) -> &Arc<dyn BackendModel> {
        &self.model
    }

    /// Get pipeline configuration
    pub fn config(&self) -> &BackendConfig {
        &self.config
    }

    /// Warm up the pipeline
    pub fn warmup(&self) -> Result<()> {
        self.model.warmup()
    }

    /// Get pipeline metrics
    pub fn get_metrics(&self) -> (BackendMetrics, ModelPerformanceStats) {
        (self.backend.get_metrics(), self.model.performance_stats())
    }
}

// Global backend registry instance
lazy_static::lazy_static! {
    pub static ref GLOBAL_BACKEND_REGISTRY: BackendRegistry = BackendRegistry::new();
}

// Convenience functions for working with the global registry
pub fn register_backend_factory(name: String, factory: Box<dyn BackendFactory>) -> Result<()> {
    GLOBAL_BACKEND_REGISTRY.register_factory(name, factory)
}

pub fn create_backend(name: &str, config: &BackendConfig) -> Result<()> {
    GLOBAL_BACKEND_REGISTRY.create_backend(name, config)
}

pub fn get_backend(name: &str) -> Result<Arc<dyn CustomBackend>> {
    GLOBAL_BACKEND_REGISTRY.get_backend(name)
}

pub fn list_available_backends() -> Result<Vec<String>> {
    GLOBAL_BACKEND_REGISTRY.list_backends()
}

pub fn list_available_factories() -> Result<Vec<String>> {
    GLOBAL_BACKEND_REGISTRY.list_factories()
}

// Factory functions for creating custom backend pipelines
pub fn create_custom_backend_pipeline(
    backend_name: &str,
    model_path: &PathBuf,
    config: BackendConfig,
    options: PipelineOptions,
) -> Result<CustomBackendPipeline> {
    let backend = get_backend(backend_name)?;
    CustomBackendPipeline::new(backend, model_path, config, options)
}

pub fn create_custom_text_generation_pipeline(
    backend_name: &str,
    model_path: &PathBuf,
    tokenizer: Arc<dyn Tokenizer>,
) -> Result<CustomBackendPipeline> {
    let config = BackendConfig {
        name: backend_name.to_string(),
        device: Device::Cpu,
        optimization_level: OptimizationLevel::Standard,
        memory_config: MemoryConfig {
            max_memory_mb: None,
            cache_size_mb: Some(512),
            prefetch_enabled: true,
            memory_mapping: false,
        },
        performance_config: PerformanceConfig {
            max_batch_size: Some(8),
            num_threads: None,
            enable_profiling: false,
            warmup_runs: 3,
        },
        custom_settings: HashMap::new(),
    };

    let options = PipelineOptions {
        model: None,
        tokenizer: None,
        device: Some(Device::Cpu),
        batch_size: Some(1),
        max_length: None,
        truncation: false,
        padding: PaddingStrategy::None,
        num_threads: None,
        cache_config: None,
        backend: None,
        onnx_config: None,
        tensorrt_config: None,
        streaming: false,
    };

    let backend = get_backend(backend_name)?;
    let pipeline =
        CustomBackendPipeline::new(backend, model_path, config, options)?.with_tokenizer(tokenizer);

    Ok(pipeline)
}

pub fn create_custom_text_classification_pipeline(
    backend_name: &str,
    model_path: &PathBuf,
    tokenizer: Arc<dyn Tokenizer>,
) -> Result<CustomBackendPipeline> {
    let config = BackendConfig {
        name: backend_name.to_string(),
        device: Device::Cpu,
        optimization_level: OptimizationLevel::Standard,
        memory_config: MemoryConfig {
            max_memory_mb: None,
            cache_size_mb: Some(256),
            prefetch_enabled: true,
            memory_mapping: false,
        },
        performance_config: PerformanceConfig {
            max_batch_size: Some(16),
            num_threads: None,
            enable_profiling: false,
            warmup_runs: 3,
        },
        custom_settings: HashMap::new(),
    };

    let options = PipelineOptions {
        model: None,
        tokenizer: None,
        device: Some(Device::Cpu),
        batch_size: Some(1),
        max_length: None,
        truncation: false,
        padding: PaddingStrategy::None,
        num_threads: None,
        cache_config: None,
        backend: None,
        onnx_config: None,
        tensorrt_config: None,
        streaming: false,
    };

    let backend = get_backend(backend_name)?;
    let pipeline =
        CustomBackendPipeline::new(backend, model_path, config, options)?.with_tokenizer(tokenizer);

    Ok(pipeline)
}

// Default implementations for common backend types
impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            device: Device::Cpu,
            optimization_level: OptimizationLevel::Standard,
            memory_config: MemoryConfig::default(),
            performance_config: PerformanceConfig::default(),
            custom_settings: HashMap::new(),
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: None,
            cache_size_mb: Some(512),
            prefetch_enabled: true,
            memory_mapping: false,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_batch_size: Some(8),
            num_threads: None,
            enable_profiling: false,
            warmup_runs: 3,
        }
    }
}

// Implementation for BackendTensor utility methods
impl BackendTensor {
    /// Create a new tensor
    pub fn new(data: Vec<u8>, shape: Vec<i64>, dtype: DataType, layout: MemoryLayout) -> Self {
        Self {
            data,
            shape,
            dtype,
            layout,
        }
    }

    /// Get tensor element count
    pub fn element_count(&self) -> usize {
        self.shape.iter().map(|&dim| dim as usize).product()
    }

    /// Get tensor size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Get element size in bytes for the data type
    pub fn element_size(&self) -> usize {
        match self.dtype {
            DataType::Float32 | DataType::Int32 => 4,
            DataType::Float16 | DataType::Int16 => 2,
            DataType::Int8 | DataType::UInt8 | DataType::Bool => 1,
        }
    }

    /// Validate tensor consistency
    pub fn validate(&self) -> Result<()> {
        let expected_size = self.element_count() * self.element_size();
        if self.data.len() != expected_size {
            return Err(TrustformersError::runtime_error(format!(
                "Tensor data size {} doesn't match expected size {}",
                self.data.len(),
                expected_size
            )));
        }
        Ok(())
    }
}
