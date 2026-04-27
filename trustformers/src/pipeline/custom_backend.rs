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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_default_config(name: &str) -> BackendConfig {
        BackendConfig {
            name: name.to_string(),
            ..BackendConfig::default()
        }
    }

    // Mock backend for registration tests
    #[derive(Debug)]
    struct MockBackend {
        name: String,
    }

    impl CustomBackend for MockBackend {
        fn name(&self) -> &str {
            &self.name
        }
        fn version(&self) -> &str {
            "1.0.0"
        }
        fn initialize(&mut self, _config: &BackendConfig) -> Result<()> {
            Ok(())
        }
        fn load_model(&self, _path: &PathBuf) -> Result<Box<dyn BackendModel>> {
            Ok(Box::new(MockModel::new()))
        }
        fn supported_devices(&self) -> Vec<Device> {
            vec![Device::Cpu]
        }
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                supported_dtypes: vec![DataType::Float32],
                supported_ops: vec!["matmul".to_string()],
                max_dimensions: 4,
                max_batch_size: Some(16),
                dynamic_shapes: true,
                in_place_ops: false,
                quantization: vec![QuantizationMode::None],
                memory_mapping: false,
            }
        }
        fn health_check(&self) -> Result<BackendHealth> {
            Ok(BackendHealth {
                status: HealthStatus::Healthy,
                device_available: true,
                memory_usage: MemoryUsage {
                    total_mb: 8192,
                    used_mb: 1024,
                    available_mb: 7168,
                    fragmentation_percent: 0.0,
                },
                last_error: None,
                performance_indicators: PerformanceIndicators {
                    latency_p50_ms: 10.0,
                    latency_p95_ms: 20.0,
                    latency_p99_ms: 30.0,
                    queue_depth: 0,
                    active_requests: 0,
                },
            })
        }
        fn get_metrics(&self) -> BackendMetrics {
            BackendMetrics {
                total_inferences: 42,
                avg_latency_ms: 15.0,
                throughput: 66.0,
                memory_stats: MemoryStats {
                    peak_usage_mb: 2048,
                    current_usage_mb: 1024,
                    allocations_count: 100,
                    deallocations_count: 90,
                },
                error_rate: 0.0,
                cache_hit_rate: 0.5,
                utilization_percent: 60.0,
            }
        }
        fn cleanup(&mut self) -> Result<()> {
            Ok(())
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[derive(Debug)]
    struct MockModel {
        metadata: ModelMetadata,
    }

    impl MockModel {
        fn new() -> Self {
            let mut input_shapes = HashMap::new();
            input_shapes.insert("input_ids".to_string(), vec![1, 128]);
            let mut output_shapes = HashMap::new();
            output_shapes.insert("logits".to_string(), vec![1, 128, 32000]);
            Self {
                metadata: ModelMetadata {
                    name: "mock_model".to_string(),
                    version: "1.0".to_string(),
                    format: "onnx".to_string(),
                    input_shapes,
                    output_shapes,
                    size_bytes: 1_000_000,
                    num_parameters: 125_000_000,
                    memory_required: 2_000_000,
                },
            }
        }
    }

    impl BackendModel for MockModel {
        fn predict(
            &self,
            inputs: &HashMap<String, BackendTensor>,
        ) -> Result<HashMap<String, BackendTensor>> {
            let mut outputs = HashMap::new();
            for (key, tensor) in inputs {
                outputs.insert(format!("out_{}", key), tensor.clone());
            }
            Ok(outputs)
        }
        fn metadata(&self) -> &ModelMetadata {
            &self.metadata
        }
        fn input_specs(&self) -> &HashMap<String, TensorSpec> {
            // Return empty for test simplicity
            static EMPTY: std::sync::OnceLock<HashMap<String, TensorSpec>> =
                std::sync::OnceLock::new();
            EMPTY.get_or_init(HashMap::new)
        }
        fn output_specs(&self) -> &HashMap<String, TensorSpec> {
            static EMPTY: std::sync::OnceLock<HashMap<String, TensorSpec>> =
                std::sync::OnceLock::new();
            EMPTY.get_or_init(HashMap::new)
        }
        fn warmup(&self) -> Result<()> {
            Ok(())
        }
        fn performance_stats(&self) -> ModelPerformanceStats {
            ModelPerformanceStats {
                total_inferences: 10,
                avg_latency_ms: 12.5,
                min_latency_ms: 8.0,
                max_latency_ms: 20.0,
                throughput: 80.0,
                memory_usage_mb: 512,
            }
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[derive(Debug)]
    struct MockFactory;

    impl BackendFactory for MockFactory {
        fn create_backend(&self, config: &BackendConfig) -> Result<Box<dyn CustomBackend>> {
            Ok(Box::new(MockBackend {
                name: config.name.clone(),
            }))
        }
        fn factory_info(&self) -> FactoryInfo {
            FactoryInfo {
                name: "mock_factory".to_string(),
                version: "1.0.0".to_string(),
                description: "Test factory".to_string(),
                supported_formats: vec!["onnx".to_string()],
                required_features: vec![],
            }
        }
    }

    // ── BackendRegistry tests ─────────────────────────────────────────────────

    #[test]
    fn test_registry_starts_empty() {
        let registry = BackendRegistry::new();
        let backends = registry.list_backends().expect("list_backends should succeed");
        assert!(backends.is_empty(), "new registry should have no backends");
    }

    #[test]
    fn test_registry_register_factory() {
        let registry = BackendRegistry::new();
        registry
            .register_factory("mock".to_string(), Box::new(MockFactory))
            .expect("register_factory should succeed");
        let factories = registry.list_factories().expect("list_factories should succeed");
        assert!(factories.contains(&"mock".to_string()));
    }

    #[test]
    fn test_registry_create_and_retrieve_backend() {
        let registry = BackendRegistry::new();
        registry
            .register_factory("mock".to_string(), Box::new(MockFactory))
            .expect("register_factory should succeed");
        let config = make_default_config("mock");
        registry.create_backend("mock", &config).expect("create_backend should succeed");
        let backend = registry
            .get_backend("mock")
            .expect("get_backend should return the registered backend");
        assert_eq!(backend.name(), "mock");
    }

    #[test]
    fn test_registry_get_missing_backend_errors() {
        let registry = BackendRegistry::new();
        let result = registry.get_backend("nonexistent");
        assert!(
            result.is_err(),
            "getting a non-existent backend should fail"
        );
    }

    #[test]
    fn test_registry_create_backend_missing_factory_errors() {
        let registry = BackendRegistry::new();
        let config = make_default_config("ghost");
        let result = registry.create_backend("ghost", &config);
        assert!(
            result.is_err(),
            "creating backend without factory should fail"
        );
    }

    #[test]
    fn test_registry_remove_backend() {
        let registry = BackendRegistry::new();
        registry
            .register_factory("mock".to_string(), Box::new(MockFactory))
            .expect("register_factory should succeed");
        registry
            .create_backend("mock", &make_default_config("mock"))
            .expect("create_backend should succeed");
        registry.remove_backend("mock").expect("remove_backend should succeed");
        assert!(
            registry.get_backend("mock").is_err(),
            "removed backend should not be retrievable"
        );
    }

    #[test]
    fn test_registry_health_check_all_healthy() {
        let registry = BackendRegistry::new();
        registry
            .register_factory("mock".to_string(), Box::new(MockFactory))
            .expect("register_factory should succeed");
        registry
            .create_backend("mock", &make_default_config("mock"))
            .expect("create_backend should succeed");
        let health_map = registry.health_check_all().expect("health_check_all should succeed");
        assert!(health_map.contains_key("mock"));
        let health = &health_map["mock"];
        assert!(matches!(health.status, HealthStatus::Healthy));
    }

    // ── Capability declaration tests ──────────────────────────────────────────

    #[test]
    fn test_backend_capabilities_quantization() {
        let backend = MockBackend {
            name: "test".to_string(),
        };
        let caps = backend.capabilities();
        assert!(
            !caps.quantization.is_empty(),
            "capabilities must declare quantization support"
        );
    }

    #[test]
    fn test_backend_capabilities_max_batch_size() {
        let backend = MockBackend {
            name: "test".to_string(),
        };
        let caps = backend.capabilities();
        assert!(
            caps.max_batch_size.is_some(),
            "capabilities should declare max_batch_size"
        );
    }

    #[test]
    fn test_backend_capabilities_supported_dtypes() {
        let backend = MockBackend {
            name: "test".to_string(),
        };
        let caps = backend.capabilities();
        assert!(
            !caps.supported_dtypes.is_empty(),
            "should declare at least one dtype"
        );
    }

    // ── BackendTensor tests ───────────────────────────────────────────────────

    #[test]
    fn test_tensor_element_count() {
        let tensor = BackendTensor::new(
            vec![0u8; 24],
            vec![2, 3],
            DataType::Float32,
            MemoryLayout::RowMajor,
        );
        // 2*3=6 elements, 4 bytes each = 24 bytes
        assert_eq!(tensor.element_count(), 6);
    }

    #[test]
    fn test_tensor_element_size_f32() {
        let tensor = BackendTensor::new(vec![], vec![], DataType::Float32, MemoryLayout::RowMajor);
        assert_eq!(tensor.element_size(), 4);
    }

    #[test]
    fn test_tensor_element_size_int8() {
        let tensor = BackendTensor::new(vec![], vec![], DataType::Int8, MemoryLayout::RowMajor);
        assert_eq!(tensor.element_size(), 1);
    }

    #[test]
    fn test_tensor_element_size_float16() {
        let tensor = BackendTensor::new(vec![], vec![], DataType::Float16, MemoryLayout::RowMajor);
        assert_eq!(tensor.element_size(), 2);
    }

    #[test]
    fn test_tensor_validate_correct() {
        // 2x3 f32 = 24 bytes
        let data = vec![0u8; 24];
        let tensor =
            BackendTensor::new(data, vec![2, 3], DataType::Float32, MemoryLayout::RowMajor);
        assert!(
            tensor.validate().is_ok(),
            "valid tensor should pass validation"
        );
    }

    #[test]
    fn test_tensor_validate_incorrect_size() {
        // 2x3 f32 = 24 bytes; give 20 → mismatch
        let data = vec![0u8; 20];
        let tensor =
            BackendTensor::new(data, vec![2, 3], DataType::Float32, MemoryLayout::RowMajor);
        assert!(
            tensor.validate().is_err(),
            "tensor with wrong data size should fail validation"
        );
    }

    #[test]
    fn test_tensor_size_bytes() {
        let data = vec![0u8; 32];
        let tensor =
            BackendTensor::new(data, vec![4, 2], DataType::Float32, MemoryLayout::RowMajor);
        assert_eq!(tensor.size_bytes(), 32);
    }

    // ── Config validation tests ───────────────────────────────────────────────

    #[test]
    fn test_backend_config_default_name() {
        let config = BackendConfig::default();
        assert!(
            !config.name.is_empty(),
            "default config should have a non-empty name"
        );
    }

    #[test]
    fn test_memory_config_default_prefetch() {
        let config = MemoryConfig::default();
        assert!(
            config.prefetch_enabled,
            "prefetch should be enabled by default"
        );
    }

    #[test]
    fn test_performance_config_default_warmup_runs() {
        let config = PerformanceConfig::default();
        assert!(
            config.warmup_runs > 0,
            "default should have at least one warmup run"
        );
    }

    // ── Error propagation ─────────────────────────────────────────────────────

    #[test]
    fn test_backend_metrics_fields_non_negative() {
        let backend = MockBackend {
            name: "test".to_string(),
        };
        let metrics = backend.get_metrics();
        assert!(metrics.avg_latency_ms >= 0.0);
        assert!(metrics.throughput >= 0.0);
        assert!(metrics.error_rate >= 0.0 && metrics.error_rate <= 1.0);
    }

    #[test]
    fn test_model_performance_stats_consistency() {
        let model = MockModel::new();
        let stats = model.performance_stats();
        assert!(
            stats.min_latency_ms <= stats.avg_latency_ms,
            "min latency must not exceed average"
        );
        assert!(
            stats.avg_latency_ms <= stats.max_latency_ms,
            "average latency must not exceed max"
        );
    }
}
