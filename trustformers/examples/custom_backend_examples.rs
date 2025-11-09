// Custom Backend Examples for TrustformeRS
#![allow(unused_variables)]
// This file demonstrates how to implement custom backends using the TrustformeRS framework

use serde_json::Value;
use std::any::Any;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use trustformers::error::{Result, TrustformersError};
use trustformers::pipeline::custom_backend::*;
use trustformers::pipeline::Device;

// Helper function for creating errors
fn runtime_error(msg: impl Into<String>) -> TrustformersError {
    TrustformersError::InvalidInput {
        message: msg.into(),
        parameter: None,
        expected: None,
        received: None,
        suggestion: None,
    }
}

// =============================================================================
// Example 1: Mock Backend (Simple Reference Implementation)
// =============================================================================

/// A simple mock backend for testing and demonstration purposes
#[derive(Debug)]
pub struct MockBackend {
    name: String,
    version: String,
    initialized: bool,
    config: Option<BackendConfig>,
    metrics: Arc<Mutex<BackendMetrics>>,
}

impl MockBackend {
    pub fn new() -> Self {
        Self {
            name: "mock-backend".to_string(),
            version: "1.0.0".to_string(),
            initialized: false,
            config: None,
            metrics: Arc::new(Mutex::new(BackendMetrics {
                total_inferences: 0,
                avg_latency_ms: 0.0,
                throughput: 0.0,
                memory_stats: MemoryStats {
                    peak_usage_mb: 64,
                    current_usage_mb: 32,
                    allocations_count: 0,
                    deallocations_count: 0,
                },
                error_rate: 0.0,
                cache_hit_rate: 0.95,
                utilization_percent: 15.0,
            })),
        }
    }
}

impl CustomBackend for MockBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn initialize(&mut self, config: &BackendConfig) -> Result<()> {
        println!("Initializing Mock Backend with config: {:?}", config.name);
        self.config = Some(config.clone());
        self.initialized = true;
        Ok(())
    }

    fn load_model(&self, path: &PathBuf) -> Result<Box<dyn BackendModel>> {
        if !self.initialized {
            return Err(runtime_error("Backend not initialized"));
        }

        println!("Loading mock model from: {:?}", path);
        Ok(Box::new(MockModel::new(path.clone())))
    }

    fn supported_devices(&self) -> Vec<Device> {
        vec![Device::Cpu, Device::Gpu(0)]
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supported_dtypes: vec![DataType::Float32, DataType::Float16, DataType::Int32],
            supported_ops: vec![
                "matmul".to_string(),
                "add".to_string(),
                "relu".to_string(),
                "softmax".to_string(),
            ],
            max_dimensions: 4,
            max_batch_size: Some(64),
            dynamic_shapes: true,
            in_place_ops: true,
            quantization: vec![
                QuantizationMode::None,
                QuantizationMode::Dynamic,
                QuantizationMode::Static,
            ],
            memory_mapping: false,
        }
    }

    fn health_check(&self) -> Result<BackendHealth> {
        Ok(BackendHealth {
            status: if self.initialized { HealthStatus::Healthy } else { HealthStatus::Warning },
            device_available: true,
            memory_usage: MemoryUsage {
                total_mb: 1024,
                used_mb: 256,
                available_mb: 768,
                fragmentation_percent: 5.0,
            },
            last_error: None,
            performance_indicators: PerformanceIndicators {
                latency_p50_ms: 12.5,
                latency_p95_ms: 25.0,
                latency_p99_ms: 40.0,
                queue_depth: 2,
                active_requests: 1,
            },
        })
    }

    fn get_metrics(&self) -> BackendMetrics {
        self.metrics.lock().unwrap().clone()
    }

    fn cleanup(&mut self) -> Result<()> {
        println!("Cleaning up Mock Backend");
        self.initialized = false;
        self.config = None;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Mock model implementation
#[derive(Debug)]
pub struct MockModel {
    _path: PathBuf,
    metadata: ModelMetadata,
    input_specs: HashMap<String, TensorSpec>,
    output_specs: HashMap<String, TensorSpec>,
    performance_stats: Arc<Mutex<ModelPerformanceStats>>,
}

impl MockModel {
    pub fn new(path: PathBuf) -> Self {
        let mut input_specs = HashMap::new();
        input_specs.insert(
            "input_ids".to_string(),
            TensorSpec {
                shape: vec![-1, -1], // Dynamic batch and sequence length
                dtype: DataType::Int32,
                layout: MemoryLayout::RowMajor,
                constraints: Some(TensorConstraints {
                    min_value: Some(0.0),
                    max_value: Some(50000.0),
                    positive_only: true,
                    normalized: false,
                }),
            },
        );

        let mut output_specs = HashMap::new();
        output_specs.insert(
            "logits".to_string(),
            TensorSpec {
                shape: vec![-1, -1, 768], // Dynamic batch, sequence, hidden size
                dtype: DataType::Float32,
                layout: MemoryLayout::RowMajor,
                constraints: None,
            },
        );

        Self {
            _path: path.clone(),
            metadata: ModelMetadata {
                name: "mock-model".to_string(),
                version: "1.0.0".to_string(),
                format: "mock".to_string(),
                input_shapes: {
                    let mut shapes = HashMap::new();
                    shapes.insert("input_ids".to_string(), vec![-1, -1]);
                    shapes
                },
                output_shapes: {
                    let mut shapes = HashMap::new();
                    shapes.insert("logits".to_string(), vec![-1, -1, 768]);
                    shapes
                },
                size_bytes: 1024 * 1024 * 100,      // 100MB
                num_parameters: 125_000_000,        // 125M parameters
                memory_required: 1024 * 1024 * 500, // 500MB
            },
            input_specs,
            output_specs,
            performance_stats: Arc::new(Mutex::new(ModelPerformanceStats {
                total_inferences: 0,
                avg_latency_ms: 0.0,
                min_latency_ms: f64::MAX,
                max_latency_ms: 0.0,
                throughput: 0.0,
                memory_usage_mb: 256,
            })),
        }
    }
}

impl BackendModel for MockModel {
    fn predict(
        &self,
        inputs: &HashMap<String, BackendTensor>,
    ) -> Result<HashMap<String, BackendTensor>> {
        let start_time = Instant::now();

        // Simulate processing delay
        std::thread::sleep(Duration::from_millis(10));

        // Validate inputs
        if !inputs.contains_key("input_ids") {
            return Err(runtime_error("Missing required input: input_ids"));
        }

        let input_tensor = &inputs["input_ids"];

        // Mock inference: create output tensor with same batch size
        let batch_size = input_tensor.shape[0];
        let seq_len = input_tensor.shape[1];
        let hidden_size = 768i64;

        // Create mock output data (zeros for simplicity)
        let output_size = (batch_size * seq_len * hidden_size) as usize * 4; // 4 bytes per float32
        let output_data = vec![0u8; output_size];

        let output_tensor = BackendTensor::new(
            output_data,
            vec![batch_size, seq_len, hidden_size],
            DataType::Float32,
            MemoryLayout::RowMajor,
        );

        let mut outputs = HashMap::new();
        outputs.insert("logits".to_string(), output_tensor);

        // Update performance stats
        let elapsed = start_time.elapsed().as_millis() as f64;
        if let Ok(mut stats) = self.performance_stats.lock() {
            stats.total_inferences += 1;
            stats.min_latency_ms = stats.min_latency_ms.min(elapsed);
            stats.max_latency_ms = stats.max_latency_ms.max(elapsed);
            stats.avg_latency_ms = (stats.avg_latency_ms * (stats.total_inferences - 1) as f64
                + elapsed)
                / stats.total_inferences as f64;
            stats.throughput = 1000.0 / stats.avg_latency_ms;
        }

        Ok(outputs)
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn input_specs(&self) -> &HashMap<String, TensorSpec> {
        &self.input_specs
    }

    fn output_specs(&self) -> &HashMap<String, TensorSpec> {
        &self.output_specs
    }

    fn warmup(&self) -> Result<()> {
        println!("Warming up mock model...");
        // Simulate warmup with a dummy inference
        let mut dummy_inputs = HashMap::new();
        dummy_inputs.insert(
            "input_ids".to_string(),
            BackendTensor::new(
                vec![0u8; 8], // 2 int32 values (1x2 tensor)
                vec![1, 2],
                DataType::Int32,
                MemoryLayout::RowMajor,
            ),
        );
        self.predict(&dummy_inputs)?;
        println!("Mock model warmup complete");
        Ok(())
    }

    fn performance_stats(&self) -> ModelPerformanceStats {
        self.performance_stats.lock().unwrap().clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Factory for creating mock backends
#[derive(Debug)]
pub struct MockBackendFactory;

impl BackendFactory for MockBackendFactory {
    fn create_backend(&self, config: &BackendConfig) -> Result<Box<dyn CustomBackend>> {
        let mut backend = MockBackend::new();
        backend.initialize(config)?;
        Ok(Box::new(backend))
    }

    fn factory_info(&self) -> FactoryInfo {
        FactoryInfo {
            name: "mock".to_string(),
            version: "1.0.0".to_string(),
            description: "Mock backend for testing and development".to_string(),
            supported_formats: vec!["mock".to_string()],
            required_features: vec![],
        }
    }
}

// =============================================================================
// Example 2: File-Based Backend (Reads pre-computed outputs from files)
// =============================================================================

/// A file-based backend that reads pre-computed outputs from files
#[derive(Debug)]
pub struct FileBasedBackend {
    name: String,
    version: String,
    initialized: bool,
    base_path: Option<PathBuf>,
    cache: Arc<Mutex<HashMap<String, HashMap<String, BackendTensor>>>>,
}

impl FileBasedBackend {
    pub fn new() -> Self {
        Self {
            name: "file-backend".to_string(),
            version: "1.0.0".to_string(),
            initialized: false,
            base_path: None,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl CustomBackend for FileBasedBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn initialize(&mut self, config: &BackendConfig) -> Result<()> {
        // Extract base path from custom settings
        if let Some(base_path_value) = config.custom_settings.get("base_path") {
            if let Some(base_path_str) = base_path_value.as_str() {
                self.base_path = Some(PathBuf::from(base_path_str));
            }
        }

        if self.base_path.is_none() {
            return Err(runtime_error(
                "base_path required in custom_settings for file-based backend",
            ));
        }

        self.initialized = true;
        Ok(())
    }

    fn load_model(&self, path: &PathBuf) -> Result<Box<dyn BackendModel>> {
        if !self.initialized {
            return Err(runtime_error("Backend not initialized"));
        }

        Ok(Box::new(FileBasedModel::new(
            path.clone(),
            self.base_path.clone().unwrap(),
            self.cache.clone(),
        )))
    }

    fn supported_devices(&self) -> Vec<Device> {
        vec![Device::Cpu] // File-based backend only supports CPU
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supported_dtypes: vec![DataType::Float32, DataType::Int32],
            supported_ops: vec!["lookup".to_string()],
            max_dimensions: 4,
            max_batch_size: Some(32),
            dynamic_shapes: true,
            in_place_ops: false,
            quantization: vec![QuantizationMode::None],
            memory_mapping: true,
        }
    }

    fn health_check(&self) -> Result<BackendHealth> {
        let base_path_exists = self.base_path.as_ref().map(|p| p.exists()).unwrap_or(false);

        Ok(BackendHealth {
            status: if self.initialized && base_path_exists {
                HealthStatus::Healthy
            } else {
                HealthStatus::Critical
            },
            device_available: true,
            memory_usage: MemoryUsage {
                total_mb: 512,
                used_mb: 128,
                available_mb: 384,
                fragmentation_percent: 2.0,
            },
            last_error: if !base_path_exists {
                Some("Base path does not exist".to_string())
            } else {
                None
            },
            performance_indicators: PerformanceIndicators {
                latency_p50_ms: 5.0,
                latency_p95_ms: 15.0,
                latency_p99_ms: 30.0,
                queue_depth: 0,
                active_requests: 0,
            },
        })
    }

    fn get_metrics(&self) -> BackendMetrics {
        let cache_size = self.cache.lock().unwrap().len() as u64;

        BackendMetrics {
            total_inferences: cache_size * 10, // Estimate
            avg_latency_ms: 8.0,
            throughput: 125.0,
            memory_stats: MemoryStats {
                peak_usage_mb: 256,
                current_usage_mb: 128,
                allocations_count: cache_size,
                deallocations_count: 0,
            },
            error_rate: 0.001,
            cache_hit_rate: 0.85,
            utilization_percent: 25.0,
        }
    }

    fn cleanup(&mut self) -> Result<()> {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
        self.initialized = false;
        self.base_path = None;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// File-based model implementation
#[derive(Debug)]
pub struct FileBasedModel {
    _path: PathBuf,
    base_path: PathBuf,
    cache: Arc<Mutex<HashMap<String, HashMap<String, BackendTensor>>>>,
    metadata: ModelMetadata,
}

impl FileBasedModel {
    pub fn new(
        path: PathBuf,
        base_path: PathBuf,
        cache: Arc<Mutex<HashMap<String, HashMap<String, BackendTensor>>>>,
    ) -> Self {
        Self {
            _path: path.clone(),
            base_path,
            cache,
            metadata: ModelMetadata {
                name: "file-model".to_string(),
                version: "1.0.0".to_string(),
                format: "file".to_string(),
                input_shapes: {
                    let mut shapes = HashMap::new();
                    shapes.insert("input_key".to_string(), vec![-1]);
                    shapes
                },
                output_shapes: {
                    let mut shapes = HashMap::new();
                    shapes.insert("output".to_string(), vec![-1, 10]);
                    shapes
                },
                size_bytes: 1024 * 1024, // 1MB
                num_parameters: 1000,
                memory_required: 1024 * 1024 * 10, // 10MB
            },
        }
    }

    fn generate_cache_key(&self, inputs: &HashMap<String, BackendTensor>) -> String {
        // Simple hash-based cache key generation
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for (key, tensor) in inputs {
            key.hash(&mut hasher);
            tensor.shape.hash(&mut hasher);
            // Hash a sample of the data to avoid hashing large tensors
            if !tensor.data.is_empty() {
                let sample_size = tensor.data.len().min(64);
                tensor.data[..sample_size].hash(&mut hasher);
            }
        }
        format!("{:x}", hasher.finish())
    }
}

impl BackendModel for FileBasedModel {
    fn predict(
        &self,
        inputs: &HashMap<String, BackendTensor>,
    ) -> Result<HashMap<String, BackendTensor>> {
        let cache_key = self.generate_cache_key(inputs);

        // Check cache first
        if let Ok(cache) = self.cache.lock() {
            if let Some(cached_result) = cache.get(&cache_key) {
                return Ok(cached_result.clone());
            }
        }

        // Simulate file-based lookup
        let output_file_path = self.base_path.join(format!("{}.bin", cache_key));

        let output_tensor = if output_file_path.exists() {
            // In a real implementation, you would read the tensor from file
            // For this example, we'll create a mock tensor
            let data_size = 10 * 4; // 10 float32 values
            let mock_data = vec![0u8; data_size];

            BackendTensor::new(
                mock_data,
                vec![1, 10],
                DataType::Float32,
                MemoryLayout::RowMajor,
            )
        } else {
            // Generate default output if file doesn't exist
            let data_size = 10 * 4; // 10 float32 values
            let mock_data = vec![0u8; data_size];

            BackendTensor::new(
                mock_data,
                vec![1, 10],
                DataType::Float32,
                MemoryLayout::RowMajor,
            )
        };

        let mut outputs = HashMap::new();
        outputs.insert("output".to_string(), output_tensor);

        // Cache the result
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(cache_key, outputs.clone());
        }

        Ok(outputs)
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn input_specs(&self) -> &HashMap<String, TensorSpec> {
        static SPECS: std::sync::OnceLock<HashMap<String, TensorSpec>> = std::sync::OnceLock::new();
        SPECS.get_or_init(|| {
            let mut specs = HashMap::new();
            specs.insert(
                "input_key".to_string(),
                TensorSpec {
                    shape: vec![-1],
                    dtype: DataType::Int32,
                    layout: MemoryLayout::RowMajor,
                    constraints: None,
                },
            );
            specs
        })
    }

    fn output_specs(&self) -> &HashMap<String, TensorSpec> {
        static SPECS: std::sync::OnceLock<HashMap<String, TensorSpec>> = std::sync::OnceLock::new();
        SPECS.get_or_init(|| {
            let mut specs = HashMap::new();
            specs.insert(
                "output".to_string(),
                TensorSpec {
                    shape: vec![-1, 10],
                    dtype: DataType::Float32,
                    layout: MemoryLayout::RowMajor,
                    constraints: None,
                },
            );
            specs
        })
    }

    fn warmup(&self) -> Result<()> {
        // Create dummy input for warmup
        let mut dummy_inputs = HashMap::new();
        dummy_inputs.insert(
            "input_key".to_string(),
            BackendTensor::new(
                vec![0u8; 4], // 1 int32 value
                vec![1],
                DataType::Int32,
                MemoryLayout::RowMajor,
            ),
        );
        self.predict(&dummy_inputs)?;
        Ok(())
    }

    fn performance_stats(&self) -> ModelPerformanceStats {
        ModelPerformanceStats {
            total_inferences: 100,
            avg_latency_ms: 8.0,
            min_latency_ms: 2.0,
            max_latency_ms: 25.0,
            throughput: 125.0,
            memory_usage_mb: 64,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Factory for creating file-based backends
#[derive(Debug)]
pub struct FileBasedBackendFactory;

impl BackendFactory for FileBasedBackendFactory {
    fn create_backend(&self, config: &BackendConfig) -> Result<Box<dyn CustomBackend>> {
        let mut backend = FileBasedBackend::new();
        backend.initialize(config)?;
        Ok(Box::new(backend))
    }

    fn factory_info(&self) -> FactoryInfo {
        FactoryInfo {
            name: "file".to_string(),
            version: "1.0.0".to_string(),
            description: "File-based backend for pre-computed outputs".to_string(),
            supported_formats: vec!["bin".to_string(), "npy".to_string()],
            required_features: vec!["filesystem".to_string()],
        }
    }
}

// =============================================================================
// Example 3: HTTP API Backend (Network-based inference)
// =============================================================================

/// HTTP API backend that sends requests to a remote inference server
#[derive(Debug)]
pub struct HttpApiBackend {
    name: String,
    version: String,
    initialized: bool,
    api_endpoint: Option<String>,
    client: Option<Arc<reqwest::Client>>,
    timeout_seconds: u64,
}

impl HttpApiBackend {
    pub fn new() -> Self {
        Self {
            name: "http-api-backend".to_string(),
            version: "1.0.0".to_string(),
            initialized: false,
            api_endpoint: None,
            client: None,
            timeout_seconds: 30,
        }
    }
}

impl CustomBackend for HttpApiBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn initialize(&mut self, config: &BackendConfig) -> Result<()> {
        // Extract API endpoint from custom settings
        if let Some(endpoint_value) = config.custom_settings.get("api_endpoint") {
            if let Some(endpoint_str) = endpoint_value.as_str() {
                self.api_endpoint = Some(endpoint_str.to_string());
            }
        }

        if let Some(timeout_value) = config.custom_settings.get("timeout_seconds") {
            if let Some(timeout) = timeout_value.as_u64() {
                self.timeout_seconds = timeout;
            }
        }

        if self.api_endpoint.is_none() {
            return Err(runtime_error(
                "api_endpoint required in custom_settings for HTTP API backend",
            ));
        }

        // Create HTTP client
        self.client = Some(Arc::new(
            reqwest::Client::builder()
                .timeout(Duration::from_secs(self.timeout_seconds))
                .build()
                .map_err(|e| runtime_error(format!("Failed to create HTTP client: {}", e)))?,
        ));

        self.initialized = true;
        Ok(())
    }

    fn load_model(&self, path: &PathBuf) -> Result<Box<dyn BackendModel>> {
        if !self.initialized {
            return Err(runtime_error("Backend not initialized"));
        }

        Ok(Box::new(HttpApiModel::new(
            path.clone(),
            self.api_endpoint.clone().unwrap(),
            self.client.clone().unwrap(),
        )))
    }

    fn supported_devices(&self) -> Vec<Device> {
        vec![Device::Cpu] // HTTP API backend is device-agnostic
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supported_dtypes: vec![
                DataType::Float32,
                DataType::Float16,
                DataType::Int32,
                DataType::Int8,
            ],
            supported_ops: vec!["http_inference".to_string()],
            max_dimensions: 8,
            max_batch_size: Some(128),
            dynamic_shapes: true,
            in_place_ops: false,
            quantization: vec![QuantizationMode::None, QuantizationMode::Dynamic],
            memory_mapping: false,
        }
    }

    fn health_check(&self) -> Result<BackendHealth> {
        let api_available =
            if let (Some(client), Some(endpoint)) = (&self.client, &self.api_endpoint) {
                // Try to ping the API endpoint
                tokio::runtime::Runtime::new().unwrap().block_on(async {
                    client
                        .get(&format!("{}/health", endpoint))
                        .send()
                        .await
                        .map(|resp| resp.status().is_success())
                        .unwrap_or(false)
                })
            } else {
                false
            };

        Ok(BackendHealth {
            status: if self.initialized && api_available {
                HealthStatus::Healthy
            } else if self.initialized {
                HealthStatus::Warning
            } else {
                HealthStatus::Critical
            },
            device_available: api_available,
            memory_usage: MemoryUsage {
                total_mb: 128,
                used_mb: 32,
                available_mb: 96,
                fragmentation_percent: 1.0,
            },
            last_error: if !api_available && self.initialized {
                Some("API endpoint not reachable".to_string())
            } else {
                None
            },
            performance_indicators: PerformanceIndicators {
                latency_p50_ms: 150.0,
                latency_p95_ms: 500.0,
                latency_p99_ms: 1000.0,
                queue_depth: 5,
                active_requests: 2,
            },
        })
    }

    fn get_metrics(&self) -> BackendMetrics {
        BackendMetrics {
            total_inferences: 1000,
            avg_latency_ms: 200.0,
            throughput: 5.0,
            memory_stats: MemoryStats {
                peak_usage_mb: 64,
                current_usage_mb: 32,
                allocations_count: 100,
                deallocations_count: 95,
            },
            error_rate: 0.02,
            cache_hit_rate: 0.0, // No caching for HTTP backend
            utilization_percent: 40.0,
        }
    }

    fn cleanup(&mut self) -> Result<()> {
        self.initialized = false;
        self.api_endpoint = None;
        self.client = None;
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// HTTP API model implementation
#[derive(Debug)]
pub struct HttpApiModel {
    _path: PathBuf,
    _api_endpoint: String,
    _client: Arc<reqwest::Client>,
    metadata: ModelMetadata,
}

impl HttpApiModel {
    pub fn new(path: PathBuf, api_endpoint: String, client: Arc<reqwest::Client>) -> Self {
        Self {
            _path: path.clone(),
            _api_endpoint: api_endpoint,
            _client: client,
            metadata: ModelMetadata {
                name: "http-api-model".to_string(),
                version: "1.0.0".to_string(),
                format: "http".to_string(),
                input_shapes: {
                    let mut shapes = HashMap::new();
                    shapes.insert("text".to_string(), vec![-1]);
                    shapes
                },
                output_shapes: {
                    let mut shapes = HashMap::new();
                    shapes.insert("predictions".to_string(), vec![-1, -1]);
                    shapes
                },
                size_bytes: 0,      // Remote model
                num_parameters: 0,  // Unknown
                memory_required: 0, // Remote
            },
        }
    }
}

impl BackendModel for HttpApiModel {
    fn predict(
        &self,
        inputs: &HashMap<String, BackendTensor>,
    ) -> Result<HashMap<String, BackendTensor>> {
        // For this example, we'll simulate the HTTP request
        // In a real implementation, you would serialize the tensors and send HTTP requests

        tokio::runtime::Runtime::new().unwrap().block_on(async {
            // Simulate HTTP request delay
            tokio::time::sleep(Duration::from_millis(100)).await;

            // Create mock response
            let response_size = 100 * 4; // 100 float32 values
            let response_data = vec![0u8; response_size];

            let output_tensor = BackendTensor::new(
                response_data,
                vec![1, 100],
                DataType::Float32,
                MemoryLayout::RowMajor,
            );

            let mut outputs = HashMap::new();
            outputs.insert("predictions".to_string(), output_tensor);

            Ok(outputs)
        })
    }

    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn input_specs(&self) -> &HashMap<String, TensorSpec> {
        static SPECS: std::sync::OnceLock<HashMap<String, TensorSpec>> = std::sync::OnceLock::new();
        SPECS.get_or_init(|| {
            let mut specs = HashMap::new();
            specs.insert(
                "text".to_string(),
                TensorSpec {
                    shape: vec![-1],
                    dtype: DataType::Int32,
                    layout: MemoryLayout::RowMajor,
                    constraints: None,
                },
            );
            specs
        })
    }

    fn output_specs(&self) -> &HashMap<String, TensorSpec> {
        static SPECS: std::sync::OnceLock<HashMap<String, TensorSpec>> = std::sync::OnceLock::new();
        SPECS.get_or_init(|| {
            let mut specs = HashMap::new();
            specs.insert(
                "predictions".to_string(),
                TensorSpec {
                    shape: vec![-1, -1],
                    dtype: DataType::Float32,
                    layout: MemoryLayout::RowMajor,
                    constraints: None,
                },
            );
            specs
        })
    }

    fn warmup(&self) -> Result<()> {
        // Create dummy input for warmup
        let mut dummy_inputs = HashMap::new();
        dummy_inputs.insert(
            "text".to_string(),
            BackendTensor::new(
                vec![0u8; 4], // 1 int32 value
                vec![1],
                DataType::Int32,
                MemoryLayout::RowMajor,
            ),
        );
        self.predict(&dummy_inputs)?;
        Ok(())
    }

    fn performance_stats(&self) -> ModelPerformanceStats {
        ModelPerformanceStats {
            total_inferences: 500,
            avg_latency_ms: 200.0,
            min_latency_ms: 50.0,
            max_latency_ms: 2000.0,
            throughput: 5.0,
            memory_usage_mb: 16, // Local memory usage only
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Factory for creating HTTP API backends
#[derive(Debug)]
pub struct HttpApiBackendFactory;

impl BackendFactory for HttpApiBackendFactory {
    fn create_backend(&self, config: &BackendConfig) -> Result<Box<dyn CustomBackend>> {
        let mut backend = HttpApiBackend::new();
        backend.initialize(config)?;
        Ok(Box::new(backend))
    }

    fn factory_info(&self) -> FactoryInfo {
        FactoryInfo {
            name: "http-api".to_string(),
            version: "1.0.0".to_string(),
            description: "HTTP API backend for remote inference servers".to_string(),
            supported_formats: vec!["http".to_string(), "rest".to_string()],
            required_features: vec!["network".to_string(), "http".to_string()],
        }
    }
}

// =============================================================================
// Demo and Testing Functions
// =============================================================================

pub async fn demo_custom_backends() -> Result<()> {
    println!("🚀 TrustformeRS Custom Backend Examples Demo");
    println!("===============================================\n");

    // Register all example backend factories
    register_backend_factory("mock".to_string(), Box::new(MockBackendFactory))?;
    register_backend_factory("file".to_string(), Box::new(FileBasedBackendFactory))?;
    register_backend_factory("http-api".to_string(), Box::new(HttpApiBackendFactory))?;

    println!("✅ Registered backend factories:");
    for factory in list_available_factories()? {
        println!("   - {}", factory);
    }
    println!();

    // Demo 1: Mock Backend
    demo_mock_backend().await?;

    // Demo 2: File-Based Backend
    demo_file_based_backend().await?;

    // Demo 3: HTTP API Backend
    demo_http_api_backend().await?;

    println!("🎉 All custom backend demos completed successfully!");
    Ok(())
}

async fn demo_mock_backend() -> Result<()> {
    println!("📋 Demo 1: Mock Backend");
    println!("-----------------------");

    let config = BackendConfig {
        name: "mock".to_string(),
        device: Device::Cpu,
        optimization_level: OptimizationLevel::Standard,
        memory_config: MemoryConfig::default(),
        performance_config: PerformanceConfig::default(),
        custom_settings: HashMap::new(),
    };

    // Create backend instance
    create_backend("mock", &config)?;
    let backend = get_backend("mock")?;

    println!(
        "✅ Created mock backend: {} v{}",
        backend.name(),
        backend.version()
    );

    // Check health
    let health = backend.health_check()?;
    println!("🏥 Backend health: {:?}", health.status);

    // Load model
    let model_path = PathBuf::from("/tmp/mock_model");
    let model = backend.load_model(&model_path)?;

    println!("📦 Loaded model: {}", model.metadata().name);

    // Warmup
    model.warmup()?;
    println!("🔥 Model warmed up");

    // Run inference
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_ids".to_string(),
        BackendTensor::new(
            vec![1, 0, 0, 0, 2, 0, 0, 0], // Two int32 values: [1, 2]
            vec![1, 2],
            DataType::Int32,
            MemoryLayout::RowMajor,
        ),
    );

    let outputs = model.predict(&inputs)?;
    println!(
        "🔮 Inference complete! Output shape: {:?}",
        outputs.get("logits").map(|t| &t.shape)
    );

    // Get performance stats
    let stats = model.performance_stats();
    println!(
        "📊 Performance: {:.2}ms avg latency, {:.1} inferences/sec",
        stats.avg_latency_ms, stats.throughput
    );

    println!();
    Ok(())
}

async fn demo_file_based_backend() -> Result<()> {
    println!("📁 Demo 2: File-Based Backend");
    println!("------------------------------");

    let mut custom_settings = HashMap::new();
    custom_settings.insert(
        "base_path".to_string(),
        Value::String("/tmp/file_backend_cache".to_string()),
    );

    let config = BackendConfig {
        name: "file".to_string(),
        device: Device::Cpu,
        optimization_level: OptimizationLevel::Basic,
        memory_config: MemoryConfig::default(),
        performance_config: PerformanceConfig::default(),
        custom_settings,
    };

    // Create backend instance
    create_backend("file", &config)?;
    let backend = get_backend("file")?;

    println!(
        "✅ Created file-based backend: {} v{}",
        backend.name(),
        backend.version()
    );

    // Check capabilities
    let capabilities = backend.capabilities();
    println!(
        "🔧 Capabilities: {} supported ops, max batch size: {:?}",
        capabilities.supported_ops.len(),
        capabilities.max_batch_size
    );

    // Load model
    let model_path = PathBuf::from("/tmp/file_model");
    let model = backend.load_model(&model_path)?;

    // Run inference
    let mut inputs = HashMap::new();
    inputs.insert(
        "input_key".to_string(),
        BackendTensor::new(
            vec![42, 0, 0, 0], // int32 value: 42
            vec![1],
            DataType::Int32,
            MemoryLayout::RowMajor,
        ),
    );

    let outputs = model.predict(&inputs)?;
    println!(
        "🗃️  File-based inference complete! Output shape: {:?}",
        outputs.get("output").map(|t| &t.shape)
    );

    println!();
    Ok(())
}

async fn demo_http_api_backend() -> Result<()> {
    println!("🌐 Demo 3: HTTP API Backend");
    println!("----------------------------");

    let mut custom_settings = HashMap::new();
    custom_settings.insert(
        "api_endpoint".to_string(),
        Value::String("http://localhost:8080".to_string()),
    );
    custom_settings.insert(
        "timeout_seconds".to_string(),
        Value::Number(serde_json::Number::from(10)),
    );

    let config = BackendConfig {
        name: "http-api".to_string(),
        device: Device::Cpu,
        optimization_level: OptimizationLevel::Standard,
        memory_config: MemoryConfig::default(),
        performance_config: PerformanceConfig::default(),
        custom_settings,
    };

    // Create backend instance
    create_backend("http-api", &config)?;
    let backend = get_backend("http-api")?;

    println!(
        "✅ Created HTTP API backend: {} v{}",
        backend.name(),
        backend.version()
    );

    // Check health (this will show warning since we don't have a real server)
    let health = backend.health_check()?;
    println!(
        "🏥 Backend health: {:?} (expected warning - no real server)",
        health.status
    );

    // Load model
    let model_path = PathBuf::from("/tmp/http_model");
    let model = backend.load_model(&model_path)?;

    // Run inference
    let mut inputs = HashMap::new();
    inputs.insert(
        "text".to_string(),
        BackendTensor::new(
            vec![72, 101, 108, 108, 111], // "Hello" as bytes
            vec![5],
            DataType::Int32,
            MemoryLayout::RowMajor,
        ),
    );

    let outputs = model.predict(&inputs)?;
    println!(
        "📡 HTTP API inference complete! Output shape: {:?}",
        outputs.get("predictions").map(|t| &t.shape)
    );

    // Get metrics
    let metrics = backend.get_metrics();
    println!(
        "📈 Network metrics: {:.1}ms avg latency, {:.1}% error rate",
        metrics.avg_latency_ms,
        metrics.error_rate * 100.0
    );

    println!();
    Ok(())
}

// Integration test functions
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_backend() {
        let mut backend = MockBackend::new();
        let config = BackendConfig::default();

        backend.initialize(&config).unwrap();
        assert_eq!(backend.name(), "mock-backend");

        let health = backend.health_check().unwrap();
        assert!(matches!(health.status, HealthStatus::Healthy));
    }

    #[test]
    fn test_backend_tensor() {
        let tensor = BackendTensor::new(
            vec![0u8; 16], // 4 float32 values
            vec![2, 2],
            DataType::Float32,
            MemoryLayout::RowMajor,
        );

        assert_eq!(tensor.element_count(), 4);
        assert_eq!(tensor.size_bytes(), 16);
        assert_eq!(tensor.element_size(), 4);
        assert!(tensor.validate().is_ok());
    }

    #[tokio::test]
    async fn test_backend_registry() {
        let registry = BackendRegistry::new();

        // Register mock factory
        registry
            .register_factory("test-mock".to_string(), Box::new(MockBackendFactory))
            .unwrap();

        let factories = registry.list_factories().unwrap();
        assert!(factories.contains(&"test-mock".to_string()));

        // Create backend
        let config = BackendConfig::default();
        registry.create_backend("test-mock", &config).unwrap();

        let backends = registry.list_backends().unwrap();
        assert!(backends.contains(&"test-mock".to_string()));

        // Get backend
        let backend = registry.get_backend("test-mock").unwrap();
        assert_eq!(backend.name(), "mock-backend");
    }
}

fn main() -> Result<()> {
    println!("🔧 TrustformeRS Custom Backend Examples");
    println!("========================================\n");

    println!("This file contains example implementations of custom backends:");
    println!("1. MockBackend - Simple testing backend with in-memory mock inference");
    println!("2. FileBasedBackend - Reads pre-computed outputs from files");
    println!("3. HttpApiBackend - Sends requests to remote inference servers");
    println!();
    println!("To run the demos, use: cargo run --example custom_backend_examples");
    println!("To run tests, use: cargo test custom_backend_examples");

    Ok(())
}
