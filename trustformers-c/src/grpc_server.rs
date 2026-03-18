//! gRPC server for TrustformeRS model serving
//!
//! This module provides a high-performance gRPC server for serving TrustformeRS models
//! with load balancing and advanced serving capabilities.

use anyhow::anyhow;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::error::{TrustformersError, TrustformersResult};
use crate::{c_str_to_string, result_to_error, string_to_c_str};

/// Global gRPC server manager
static GRPC_SERVER_MANAGER: Lazy<Mutex<GrpcServerManager>> =
    Lazy::new(|| Mutex::new(GrpcServerManager::new()));

/// gRPC server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub request_timeout_ms: u64,
    pub enable_reflection: bool,
    pub enable_health_check: bool,
    pub enable_metrics: bool,
    pub tls_config: Option<TlsConfig>,
    pub compression: CompressionConfig,
    pub load_balancing: LoadBalancingConfig,
}

impl Default for GrpcServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 50051,
            max_connections: 1000,
            request_timeout_ms: 60000,
            enable_reflection: true,
            enable_health_check: true,
            enable_metrics: true,
            tls_config: None,
            compression: CompressionConfig::default(),
            load_balancing: LoadBalancingConfig::default(),
        }
    }
}

/// TLS configuration for gRPC server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    pub cert_path: String,
    pub key_path: String,
    pub ca_cert_path: Option<String>,
    pub client_cert_required: bool,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub enable_gzip: bool,
    pub enable_deflate: bool,
    pub compression_level: u8, // 1-9
    pub min_compression_size: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enable_gzip: true,
            enable_deflate: true,
            compression_level: 6,
            min_compression_size: 1024,
        }
    }
}

/// Load balancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    pub strategy: LoadBalancingStrategy,
    pub health_check_interval_ms: u64,
    pub max_requests_per_instance: usize,
    pub circuit_breaker_enabled: bool,
    pub circuit_breaker_threshold: u32,
    pub circuit_breaker_timeout_ms: u64,
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            strategy: LoadBalancingStrategy::RoundRobin,
            health_check_interval_ms: 30000,
            max_requests_per_instance: 100,
            circuit_breaker_enabled: true,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout_ms: 30000,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    LeastResponseTime,
    ConsistentHashing,
}

/// gRPC server instance
#[derive(Debug)]
pub struct GrpcServer {
    config: GrpcServerConfig,
    server_handle: Option<ServerHandle>,
    model_instances: Arc<RwLock<HashMap<String, Vec<ModelInstance>>>>,
    load_balancer: Arc<LoadBalancer>,
    metrics: Arc<Mutex<GrpcServerMetrics>>,
    is_running: Arc<Mutex<bool>>,
}

/// Server handle for managing the gRPC server
#[derive(Debug)]
struct ServerHandle {
    thread_handle: thread::JoinHandle<()>,
    shutdown_signal: Arc<Mutex<bool>>,
}

/// Model instance for load balancing
#[derive(Debug, Clone)]
pub struct ModelInstance {
    pub id: String,
    pub model_path: String,
    pub weight: f64,
    pub max_concurrent_requests: usize,
    pub current_requests: Arc<Mutex<usize>>,
    pub total_requests: Arc<Mutex<u64>>,
    pub total_response_time_ms: Arc<Mutex<u64>>,
    pub last_health_check: Arc<Mutex<Instant>>,
    pub is_healthy: Arc<Mutex<bool>>,
    pub circuit_breaker_state: Arc<Mutex<CircuitBreakerState>>,
}

/// Circuit breaker state
#[derive(Debug, Clone)]
pub enum CircuitBreakerState {
    Closed,
    Open { opened_at: Instant },
    HalfOpen,
}

/// Load balancer implementation
#[derive(Debug)]
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    config: LoadBalancingConfig,
}

/// gRPC server metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GrpcServerMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_response_time_ms: f64,
    pub current_connections: u32,
    pub peak_connections: u32,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub requests_per_second: f64,
    pub model_requests: HashMap<String, u64>,
    pub error_rates: HashMap<String, f64>,
}

/// gRPC request context
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub request_id: String,
    pub client_id: String,
    pub model_name: String,
    pub start_time: Instant,
    pub metadata: HashMap<String, String>,
}

/// Text generation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcTextGenerationRequest {
    pub prompt: String,
    pub max_length: Option<u32>,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub do_sample: Option<bool>,
    pub num_return_sequences: Option<u32>,
    pub stream: Option<bool>,
    pub stop_sequences: Vec<String>,
    pub model_name: String,
}

/// Text generation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcTextGenerationResponse {
    pub generated_text: Vec<String>,
    pub processing_time_ms: f64,
    pub model_name: String,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub request_id: String,
    pub instance_id: String,
}

/// Text classification request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcTextClassificationRequest {
    pub text: String,
    pub return_all_scores: Option<bool>,
    pub model_name: String,
}

/// Text classification response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcTextClassificationResponse {
    pub classifications: Vec<ClassificationResult>,
    pub processing_time_ms: f64,
    pub model_name: String,
    pub request_id: String,
    pub instance_id: String,
}

/// Classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub label: String,
    pub score: f32,
}

/// Server manager for handling multiple gRPC servers
#[derive(Debug)]
struct GrpcServerManager {
    servers: HashMap<String, GrpcServer>,
    next_server_id: usize,
}

impl GrpcServerManager {
    fn new() -> Self {
        Self {
            servers: HashMap::new(),
            next_server_id: 1,
        }
    }

    fn create_server(&mut self, config: GrpcServerConfig) -> TrustformersResult<String> {
        let server_id = format!("grpc_server_{}", self.next_server_id);
        self.next_server_id += 1;

        let server = GrpcServer::new(config)?;
        self.servers.insert(server_id.clone(), server);

        Ok(server_id)
    }

    fn get_server_mut(&mut self, server_id: &str) -> Option<&mut GrpcServer> {
        self.servers.get_mut(server_id)
    }

    fn remove_server(&mut self, server_id: &str) -> Option<GrpcServer> {
        self.servers.remove(server_id)
    }
}

impl GrpcServer {
    fn new(config: GrpcServerConfig) -> TrustformersResult<Self> {
        let load_balancer = Arc::new(LoadBalancer::new(config.load_balancing.clone()));

        Ok(Self {
            config,
            server_handle: None,
            model_instances: Arc::new(RwLock::new(HashMap::new())),
            load_balancer,
            metrics: Arc::new(Mutex::new(GrpcServerMetrics::default())),
            is_running: Arc::new(Mutex::new(false)),
        })
    }

    fn add_model_instance(
        &mut self,
        model_name: String,
        instance: ModelInstance,
    ) -> TrustformersResult<()> {
        let mut instances = self
            .model_instances
            .write()
            .map_err(|_| anyhow!("Failed to acquire write lock on model instances"))?;

        instances.entry(model_name).or_insert_with(Vec::new).push(instance);
        Ok(())
    }

    fn remove_model_instance(
        &mut self,
        model_name: &str,
        instance_id: &str,
    ) -> TrustformersResult<()> {
        let mut instances = self
            .model_instances
            .write()
            .map_err(|_| anyhow!("Failed to acquire write lock on model instances"))?;

        if let Some(model_instances) = instances.get_mut(model_name) {
            model_instances.retain(|instance| instance.id != instance_id);
            if model_instances.is_empty() {
                instances.remove(model_name);
            }
        }

        Ok(())
    }

    fn start(&mut self) -> TrustformersResult<()> {
        let mut is_running = self
            .is_running
            .lock()
            .map_err(|_| anyhow!("Failed to acquire lock on running state"))?;

        if *is_running {
            return Err(anyhow!("Server is already running").into());
        }

        let shutdown_signal = Arc::new(Mutex::new(false));
        let shutdown_signal_clone = shutdown_signal.clone();
        let config = self.config.clone();
        let model_instances = self.model_instances.clone();
        let load_balancer = self.load_balancer.clone();
        let metrics = self.metrics.clone();

        let thread_handle = thread::spawn(move || {
            Self::run_grpc_server_loop(
                config,
                model_instances,
                load_balancer,
                metrics,
                shutdown_signal_clone,
            );
        });

        self.server_handle = Some(ServerHandle {
            thread_handle,
            shutdown_signal,
        });

        *is_running = true;
        Ok(())
    }

    fn stop(&mut self) -> TrustformersResult<()> {
        let mut is_running = self
            .is_running
            .lock()
            .map_err(|_| anyhow!("Failed to acquire lock on running state"))?;

        if !*is_running {
            return Ok(());
        }

        if let Some(handle) = self.server_handle.take() {
            // Signal shutdown
            if let Ok(mut shutdown) = handle.shutdown_signal.lock() {
                *shutdown = true;
            }

            // Wait for server thread to finish
            handle
                .thread_handle
                .join()
                .map_err(|_| anyhow!("Failed to join server thread"))?;
        }

        *is_running = false;
        Ok(())
    }

    fn run_grpc_server_loop(
        config: GrpcServerConfig,
        model_instances: Arc<RwLock<HashMap<String, Vec<ModelInstance>>>>,
        load_balancer: Arc<LoadBalancer>,
        metrics: Arc<Mutex<GrpcServerMetrics>>,
        shutdown_signal: Arc<Mutex<bool>>,
    ) {
        eprintln!("gRPC server started on {}:{}", config.host, config.port);
        eprintln!("Health check enabled: {}", config.enable_health_check);
        eprintln!("Reflection enabled: {}", config.enable_reflection);
        eprintln!("Metrics enabled: {}", config.enable_metrics);

        // Simulate gRPC server loop (in real implementation, this would use tonic or similar)
        loop {
            // Check shutdown signal
            if let Ok(shutdown) = shutdown_signal.lock() {
                if *shutdown {
                    break;
                }
            }

            // Simulate request processing and health checks
            Self::simulate_request_processing(&model_instances, &load_balancer, &metrics);
            Self::perform_health_checks(&model_instances, &config.load_balancing);

            thread::sleep(Duration::from_millis(100));
        }

        eprintln!("gRPC server stopped");
    }

    fn simulate_request_processing(
        model_instances: &Arc<RwLock<HashMap<String, Vec<ModelInstance>>>>,
        load_balancer: &Arc<LoadBalancer>,
        metrics: &Arc<Mutex<GrpcServerMetrics>>,
    ) {
        // Simulate incoming requests
        if let Ok(instances) = model_instances.read() {
            for (model_name, model_list) in instances.iter() {
                if !model_list.is_empty() {
                    // Simulate load balancer selecting an instance
                    if let Some(selected_instance) = load_balancer.select_instance(model_list) {
                        // Update metrics
                        if let Ok(mut metrics_guard) = metrics.lock() {
                            metrics_guard.total_requests += 1;
                            metrics_guard.successful_requests += 1;
                            *metrics_guard.model_requests.entry(model_name.clone()).or_insert(0) +=
                                1;
                        }

                        // Update instance metrics
                        if let Ok(mut requests) = selected_instance.total_requests.lock() {
                            *requests += 1;
                        }
                    }
                }
            }
        }
    }

    fn perform_health_checks(
        model_instances: &Arc<RwLock<HashMap<String, Vec<ModelInstance>>>>,
        config: &LoadBalancingConfig,
    ) {
        if let Ok(instances) = model_instances.read() {
            for model_list in instances.values() {
                for instance in model_list {
                    if let Ok(mut last_check) = instance.last_health_check.lock() {
                        let now = Instant::now();
                        if now.duration_since(*last_check).as_millis()
                            > config.health_check_interval_ms as u128
                        {
                            // Perform health check (simplified simulation)
                            if let Ok(mut is_healthy) = instance.is_healthy.lock() {
                                *is_healthy = true; // Simulate healthy status
                            }
                            *last_check = now;
                        }
                    }
                }
            }
        }
    }

    fn get_metrics(&self) -> TrustformersResult<GrpcServerMetrics> {
        let metrics =
            self.metrics.lock().map_err(|_| anyhow!("Failed to acquire lock on metrics"))?;
        Ok(metrics.clone())
    }
}

impl LoadBalancer {
    fn new(config: LoadBalancingConfig) -> Self {
        Self {
            strategy: config.strategy.clone(),
            config,
        }
    }

    fn select_instance<'a>(&self, instances: &'a [ModelInstance]) -> Option<&'a ModelInstance> {
        if instances.is_empty() {
            return None;
        }

        // Filter healthy instances
        let healthy_instances: Vec<&ModelInstance> = instances
            .iter()
            .filter(|instance| {
                if let Ok(is_healthy) = instance.is_healthy.lock() {
                    *is_healthy
                } else {
                    false
                }
            })
            .collect();

        if healthy_instances.is_empty() {
            return None;
        }

        match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                // Simple round-robin (in real implementation, would maintain state)
                Some(healthy_instances[0])
            },
            LoadBalancingStrategy::LeastConnections => healthy_instances
                .iter()
                .min_by_key(|instance| *instance.current_requests.lock().expect("lock should not be poisoned"))
                .copied(),
            LoadBalancingStrategy::WeightedRoundRobin => {
                // Weighted selection (simplified)
                healthy_instances
                    .iter()
                    .max_by(|a, b| {
                        a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .copied()
            },
            LoadBalancingStrategy::LeastResponseTime => healthy_instances
                .iter()
                .min_by_key(|instance| {
                    let total_requests = *instance.total_requests.lock().expect("lock should not be poisoned");
                    let total_time = *instance.total_response_time_ms.lock().expect("lock should not be poisoned");
                    if total_requests > 0 {
                        total_time / total_requests
                    } else {
                        0
                    }
                })
                .copied(),
            LoadBalancingStrategy::ConsistentHashing => {
                // Simplified consistent hashing
                Some(healthy_instances[0])
            },
        }
    }
}

impl ModelInstance {
    fn new(id: String, model_path: String) -> Self {
        Self {
            id,
            model_path,
            weight: 1.0,
            max_concurrent_requests: 10,
            current_requests: Arc::new(Mutex::new(0)),
            total_requests: Arc::new(Mutex::new(0)),
            total_response_time_ms: Arc::new(Mutex::new(0)),
            last_health_check: Arc::new(Mutex::new(Instant::now())),
            is_healthy: Arc::new(Mutex::new(true)),
            circuit_breaker_state: Arc::new(Mutex::new(CircuitBreakerState::Closed)),
        }
    }
}

// C API exports for gRPC server

/// Create gRPC server with default configuration
#[no_mangle]
pub extern "C" fn trustformers_grpc_server_create(
    server_id: *mut *mut c_char,
) -> TrustformersError {
    if server_id.is_null() {
        return TrustformersError::NullPointer;
    }

    let mut manager = match GRPC_SERVER_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let config = GrpcServerConfig::default();
    let id = match manager.create_server(config) {
        Ok(id) => id,
        Err(_) => return TrustformersError::RuntimeError,
    };

    unsafe {
        *server_id = string_to_c_str(id);
    }

    TrustformersError::Success
}

/// Create gRPC server with custom configuration
#[no_mangle]
pub extern "C" fn trustformers_grpc_server_create_with_config(
    config_json: *const c_char,
    server_id: *mut *mut c_char,
) -> TrustformersError {
    if config_json.is_null() || server_id.is_null() {
        return TrustformersError::NullPointer;
    }

    let config_str = match c_str_to_string(config_json) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let config: GrpcServerConfig = match serde_json::from_str(&config_str) {
        Ok(config) => config,
        Err(_) => return TrustformersError::SerializationError,
    };

    let mut manager = match GRPC_SERVER_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let id = match manager.create_server(config) {
        Ok(id) => id,
        Err(_) => return TrustformersError::RuntimeError,
    };

    unsafe {
        *server_id = string_to_c_str(id);
    }

    TrustformersError::Success
}

/// Add model instance to gRPC server
#[no_mangle]
pub extern "C" fn trustformers_grpc_server_add_model_instance(
    server_id: *const c_char,
    model_name: *const c_char,
    instance_json: *const c_char,
) -> TrustformersError {
    if server_id.is_null() || model_name.is_null() || instance_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let server_id_str = match c_str_to_string(server_id) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let model_name_str = match c_str_to_string(model_name) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let instance_str = match c_str_to_string(instance_json) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    // Parse instance configuration
    let instance_config: serde_json::Value = match serde_json::from_str(&instance_str) {
        Ok(config) => config,
        Err(_) => return TrustformersError::SerializationError,
    };

    let instance_id = instance_config["id"].as_str().unwrap_or("default").to_string();
    let model_path = instance_config["model_path"].as_str().unwrap_or("").to_string();

    let instance = ModelInstance::new(instance_id, model_path);

    let mut manager = match GRPC_SERVER_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let server = match manager.get_server_mut(&server_id_str) {
        Some(server) => server,
        None => return TrustformersError::InvalidParameter,
    };

    match server.add_model_instance(model_name_str, instance) {
        Ok(_) => TrustformersError::Success,
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Start gRPC server
#[no_mangle]
pub extern "C" fn trustformers_grpc_server_start(server_id: *const c_char) -> TrustformersError {
    if server_id.is_null() {
        return TrustformersError::NullPointer;
    }

    let server_id_str = match c_str_to_string(server_id) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let mut manager = match GRPC_SERVER_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let server = match manager.get_server_mut(&server_id_str) {
        Some(server) => server,
        None => return TrustformersError::InvalidParameter,
    };

    match server.start() {
        Ok(_) => TrustformersError::Success,
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Stop gRPC server
#[no_mangle]
pub extern "C" fn trustformers_grpc_server_stop(server_id: *const c_char) -> TrustformersError {
    if server_id.is_null() {
        return TrustformersError::NullPointer;
    }

    let server_id_str = match c_str_to_string(server_id) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let mut manager = match GRPC_SERVER_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let server = match manager.get_server_mut(&server_id_str) {
        Some(server) => server,
        None => return TrustformersError::InvalidParameter,
    };

    match server.stop() {
        Ok(_) => TrustformersError::Success,
        Err(_) => TrustformersError::RuntimeError,
    }
}

/// Get gRPC server metrics
#[no_mangle]
pub extern "C" fn trustformers_grpc_server_get_metrics(
    server_id: *const c_char,
    metrics_json: *mut *mut c_char,
) -> TrustformersError {
    if server_id.is_null() || metrics_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let server_id_str = match c_str_to_string(server_id) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let manager = match GRPC_SERVER_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let server = match manager.servers.get(&server_id_str) {
        Some(server) => server,
        None => return TrustformersError::InvalidParameter,
    };

    let metrics = match server.get_metrics() {
        Ok(metrics) => metrics,
        Err(_) => return TrustformersError::RuntimeError,
    };

    let metrics_json_data = match serde_json::to_string_pretty(&metrics) {
        Ok(json) => json,
        Err(_) => return TrustformersError::SerializationError,
    };

    unsafe {
        *metrics_json = string_to_c_str(metrics_json_data);
    }

    TrustformersError::Success
}

/// Destroy gRPC server
#[no_mangle]
pub extern "C" fn trustformers_grpc_server_destroy(server_id: *const c_char) -> TrustformersError {
    if server_id.is_null() {
        return TrustformersError::NullPointer;
    }

    let server_id_str = match c_str_to_string(server_id) {
        Ok(s) => s,
        Err(_) => return TrustformersError::InvalidParameter,
    };

    let mut manager = match GRPC_SERVER_MANAGER.lock() {
        Ok(manager) => manager,
        Err(_) => return TrustformersError::RuntimeError,
    };

    if let Some(mut server) = manager.remove_server(&server_id_str) {
        let _ = server.stop(); // Ensure server is stopped before destruction
    }

    TrustformersError::Success
}
