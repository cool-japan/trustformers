//! Test Environment Isolation and Helper Utilities
//!
//! This module provides comprehensive test environment isolation, fixtures, and helper
//! utilities to ensure reliable and independent test execution. It includes isolated
//! test servers, cleanup mechanisms, and test data management.

// Allow dead code for test infrastructure under development
#![allow(dead_code)]

use anyhow::Result;
use axum_test::TestServer;
use serde_json::Value;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU16, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tokio::sync::RwLock;
use trustformers_serve::{
    batching::BatchingConfig, caching::CacheConfig, shadow::ShadowConfig,
    streaming::StreamingConfig, Device, ModelConfig, ServerConfig, TrustformerServer,
};

/// Global port allocator to ensure test isolation
static PORT_ALLOCATOR: AtomicU16 = AtomicU16::new(8000);

/// Global test registry to track active test environments
static TEST_REGISTRY: std::sync::LazyLock<Mutex<HashMap<String, TestEnvironment>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Test environment isolation levels
#[derive(Debug, Clone, PartialEq)]
pub enum IsolationLevel {
    /// Basic isolation - separate ports and configs
    Basic,
    /// Process isolation - separate processes
    Process,
    /// Container isolation - separate containers
    Container,
    /// Full isolation - separate networks and filesystems
    Full,
}

/// Test environment configuration
#[derive(Debug, Clone)]
pub struct TestEnvironmentConfig {
    /// Test name for identification
    pub test_name: String,
    /// Isolation level
    pub isolation_level: IsolationLevel,
    /// Enable authentication
    pub enable_auth: bool,
    /// Enable caching
    pub enable_caching: bool,
    /// Enable streaming
    pub enable_streaming: bool,
    /// Enable shadow testing
    pub enable_shadow: bool,
    /// Custom model configuration
    pub model_config: Option<ModelConfig>,
    /// Resource limits
    pub resource_limits: TestResourceLimits,
    /// Cleanup timeout
    pub cleanup_timeout: Duration,
}

impl Default for TestEnvironmentConfig {
    fn default() -> Self {
        Self {
            test_name: "default".to_string(),
            isolation_level: IsolationLevel::Basic,
            enable_auth: false,
            enable_caching: true,
            enable_streaming: false,
            enable_shadow: false,
            model_config: None,
            resource_limits: TestResourceLimits::default(),
            cleanup_timeout: Duration::from_secs(30),
        }
    }
}

/// Test resource limits
#[derive(Debug, Clone)]
pub struct TestResourceLimits {
    /// Maximum memory usage (MB)
    pub max_memory_mb: Option<u64>,
    /// Maximum CPU usage (percentage)
    pub max_cpu_percent: Option<f32>,
    /// Maximum concurrent requests
    pub max_concurrent_requests: Option<usize>,
    /// Request timeout
    pub request_timeout: Duration,
}

impl Default for TestResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: Some(512),
            max_cpu_percent: Some(80.0),
            max_concurrent_requests: Some(100),
            request_timeout: Duration::from_secs(30),
        }
    }
}

/// Isolated test environment
pub struct TestEnvironment {
    /// Environment configuration
    pub config: TestEnvironmentConfig,
    /// Test server instance
    pub server: TestServer,
    /// Server configuration
    pub server_config: ServerConfig,
    /// Allocated port
    pub port: u16,
    /// Environment state
    pub state: Arc<RwLock<TestEnvironmentState>>,
    /// Cleanup handlers
    pub cleanup_handlers: Vec<Box<dyn Fn() -> Result<()> + Send + Sync>>,
}

impl std::fmt::Debug for TestEnvironment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestEnvironment")
            .field("config", &self.config)
            .field("server", &"TestServer { .. }")
            .field("server_config", &self.server_config)
            .field("port", &self.port)
            .field("state", &self.state)
            .field("cleanup_handlers_count", &self.cleanup_handlers.len())
            .finish()
    }
}

/// Test environment state
#[derive(Debug, Clone)]
pub struct TestEnvironmentState {
    /// Environment status
    pub status: EnvironmentStatus,
    /// Start time
    pub started_at: std::time::Instant,
    /// Active requests
    pub active_requests: usize,
    /// Total requests handled
    pub total_requests: u64,
    /// Error count
    pub error_count: u64,
    /// Resource usage
    pub resource_usage: ResourceUsage,
}

/// Environment status
#[derive(Debug, Clone, PartialEq)]
pub enum EnvironmentStatus {
    /// Environment is starting up
    Starting,
    /// Environment is ready for testing
    Ready,
    /// Environment is running tests
    Running,
    /// Environment is shutting down
    ShuttingDown,
    /// Environment has been cleaned up
    Cleaned,
    /// Environment failed to start or crashed
    Failed,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Memory usage in MB
    pub memory_mb: f64,
    /// CPU usage percentage
    pub cpu_percent: f32,
    /// Network I/O bytes
    pub network_bytes: u64,
    /// Disk I/O bytes
    pub disk_bytes: u64,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            memory_mb: 0.0,
            cpu_percent: 0.0,
            network_bytes: 0,
            disk_bytes: 0,
        }
    }
}

/// Test environment builder
pub struct TestEnvironmentBuilder {
    config: TestEnvironmentConfig,
}

impl TestEnvironmentBuilder {
    /// Create a new test environment builder
    pub fn new(test_name: &str) -> Self {
        Self {
            config: TestEnvironmentConfig {
                test_name: test_name.to_string(),
                ..Default::default()
            },
        }
    }

    /// Set isolation level
    pub fn isolation_level(mut self, level: IsolationLevel) -> Self {
        self.config.isolation_level = level;
        self
    }

    /// Enable authentication
    pub fn with_auth(mut self, enable: bool) -> Self {
        self.config.enable_auth = enable;
        self
    }

    /// Enable caching
    pub fn with_caching(mut self, enable: bool) -> Self {
        self.config.enable_caching = enable;
        self
    }

    /// Enable streaming
    pub fn with_streaming(mut self, enable: bool) -> Self {
        self.config.enable_streaming = enable;
        self
    }

    /// Enable shadow testing
    pub fn with_shadow(mut self, enable: bool) -> Self {
        self.config.enable_shadow = enable;
        self
    }

    /// Set custom model configuration
    pub fn with_model_config(mut self, model_config: ModelConfig) -> Self {
        self.config.model_config = Some(model_config);
        self
    }

    /// Set resource limits
    pub fn with_resource_limits(mut self, limits: TestResourceLimits) -> Self {
        self.config.resource_limits = limits;
        self
    }

    /// Set cleanup timeout
    pub fn with_cleanup_timeout(mut self, timeout: Duration) -> Self {
        self.config.cleanup_timeout = timeout;
        self
    }

    /// Build the test environment
    pub async fn build(self) -> Result<TestEnvironment> {
        create_isolated_test_environment(self.config).await
    }
}

/// Create an isolated test environment
pub async fn create_isolated_test_environment(
    config: TestEnvironmentConfig,
) -> Result<TestEnvironment> {
    // Allocate unique port
    let port = PORT_ALLOCATOR.fetch_add(1, Ordering::SeqCst);

    // Create server configuration
    let mut server_config = create_test_server_config(&config, port)?;

    // Apply isolation settings
    apply_isolation_settings(&mut server_config, &config)?;

    // Create server instance with or without auth
    let server = if config.enable_auth {
        create_test_server_instance_with_auth(server_config.clone(), true).await?
    } else {
        create_test_server_instance(server_config.clone()).await?
    };

    // Initialize environment state
    let state = Arc::new(RwLock::new(TestEnvironmentState {
        status: EnvironmentStatus::Starting,
        started_at: std::time::Instant::now(),
        active_requests: 0,
        total_requests: 0,
        error_count: 0,
        resource_usage: ResourceUsage::default(),
    }));

    let environment = TestEnvironment {
        config,
        server,
        server_config,
        port,
        state: state.clone(),
        cleanup_handlers: Vec::new(),
    };

    // Update state to ready
    {
        let mut state_guard = state.write().await;
        state_guard.status = EnvironmentStatus::Ready;
    }

    Ok(environment)
}

/// Create test server configuration
fn create_test_server_config(config: &TestEnvironmentConfig, port: u16) -> Result<ServerConfig> {
    let mut server_config = ServerConfig::default();

    // Basic server settings
    server_config.host = "127.0.0.1".to_string();
    server_config.port = port;

    // Model configuration
    server_config.model_config = config.model_config.clone().unwrap_or_else(|| ModelConfig {
        model_name: format!("test-model-{}", config.test_name),
        model_version: Some("1.0.0".to_string()),
        device: Device::Cpu,
        max_sequence_length: 2048,
        enable_caching: config.enable_caching,
    });

    // Authentication configuration
    // Note: Authentication is now handled at the server level using with_auth() method
    // This will be configured when creating the server instance

    // Caching configuration
    if config.enable_caching {
        server_config.caching_config = CacheConfig {
            result_cache: trustformers_serve::caching::config::TierConfig {
                max_size_bytes: 1024 * 1024, // 1MB for tests
                max_entries: 1000,
                default_ttl: Duration::from_secs(300),
                eviction_policy: trustformers_serve::caching::config::EvictionPolicy::LRU,
                compression_enabled: false,
                tier_name: "test_result_cache".to_string(),
            },
            ..Default::default()
        };
    }

    // Streaming configuration
    if config.enable_streaming {
        server_config.streaming_config = StreamingConfig {
            buffer_size: 100,
            max_concurrent_streams: 10,
            heartbeat_interval: Duration::from_secs(30),
            ..Default::default()
        };
    }

    // Shadow testing configuration
    if config.enable_shadow {
        server_config.shadow_config = ShadowConfig {
            traffic_percentage: 10.0,
            ..Default::default()
        };
    }

    // Batching configuration for tests
    server_config.batching_config = BatchingConfig {
        max_batch_size: 8,
        min_batch_size: 1,
        max_wait_time: Duration::from_millis(100),
        ..Default::default()
    };

    Ok(server_config)
}

/// Apply isolation settings to server configuration
fn apply_isolation_settings(
    server_config: &mut ServerConfig,
    config: &TestEnvironmentConfig,
) -> Result<()> {
    match config.isolation_level {
        IsolationLevel::Basic => {
            // Basic isolation - just separate ports and test-specific configs
        },
        IsolationLevel::Process => {
            // Process isolation - additional process-level configurations
            // Process isolation would be configured here
        },
        IsolationLevel::Container => {
            // Container isolation - container-specific settings
            // Container isolation would be configured here
        },
        IsolationLevel::Full => {
            // Full isolation - network and filesystem isolation
            // Full isolation would be configured here
        },
    }

    // Apply resource limits to memory pressure configuration
    if let Some(max_memory) = config.resource_limits.max_memory_mb {
        server_config.memory_pressure_config.memory_buffer_mb = max_memory as usize;
    }

    if let Some(max_concurrent) = config.resource_limits.max_concurrent_requests {
        // Set max concurrent requests in batching configuration
        server_config.batching_config.max_batch_size = max_concurrent;
    }

    Ok(())
}

/// Create test server instance
async fn create_test_server_instance(config: ServerConfig) -> Result<TestServer> {
    let server = TrustformerServer::new(config);
    let router = server.create_test_router().await;
    TestServer::new(router)
}

/// Create test server instance with authentication
async fn create_test_server_instance_with_auth(
    config: ServerConfig,
    enable_auth: bool,
) -> Result<TestServer> {
    let mut server = TrustformerServer::new(config);

    if enable_auth {
        // Create a test authentication service with default configuration
        let auth_config = trustformers_serve::auth::AuthConfig::default();
        let auth_service = trustformers_serve::auth::AuthService::new(auth_config);
        server = server.with_auth(auth_service);
    }

    let router = server.create_test_router().await;
    TestServer::new(router)
}

/// Test fixture manager
#[derive(Debug)]
pub struct TestFixtures {
    /// Test data sets
    pub datasets: HashMap<String, Vec<Value>>,
    /// Mock responses
    pub mock_responses: HashMap<String, Value>,
    /// Test configurations
    pub configs: HashMap<String, Value>,
}

impl TestFixtures {
    /// Create new test fixtures
    pub fn new() -> Self {
        Self {
            datasets: HashMap::new(),
            mock_responses: HashMap::new(),
            configs: HashMap::new(),
        }
    }

    /// Load test dataset
    pub fn load_dataset(&mut self, name: &str, data: Vec<Value>) {
        self.datasets.insert(name.to_string(), data);
    }

    /// Get test dataset
    pub fn get_dataset(&self, name: &str) -> Option<&Vec<Value>> {
        self.datasets.get(name)
    }

    /// Set mock response
    pub fn set_mock_response(&mut self, endpoint: &str, response: Value) {
        self.mock_responses.insert(endpoint.to_string(), response);
    }

    /// Get mock response
    pub fn get_mock_response(&self, endpoint: &str) -> Option<&Value> {
        self.mock_responses.get(endpoint)
    }

    /// Load default fixtures
    pub fn load_defaults(&mut self) {
        // Default inference test data
        self.load_dataset("inference_inputs", vec![
            serde_json::json!({"text": "Test input 1", "max_tokens": 50}),
            serde_json::json!({"text": "Test input 2", "max_tokens": 100}),
            serde_json::json!({"text": "Long test input for testing model capabilities", "max_tokens": 200}),
        ]);

        // Default batch test data
        self.load_dataset(
            "batch_inputs",
            vec![serde_json::json!({"inputs": ["Batch test 1", "Batch test 2", "Batch test 3"]})],
        );

        // Default mock responses
        self.set_mock_response(
            "/health",
            serde_json::json!({
                "status": "healthy",
                "timestamp": "2024-01-01T00:00:00Z",
                "version": "1.0.0"
            }),
        );

        self.set_mock_response(
            "/v1/inference",
            serde_json::json!({
                "request_id": "test-request-123",
                "output": "Test response output",
                "metadata": {
                    "model": "test-model",
                    "inference_time_ms": 100
                }
            }),
        );
    }
}

/// Test cleanup manager
pub struct TestCleanupManager {
    /// Cleanup tasks
    cleanup_tasks: Vec<Box<dyn Fn() -> Result<()> + Send + Sync>>,
    /// Timeout for cleanup
    timeout: Duration,
}

impl std::fmt::Debug for TestCleanupManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestCleanupManager")
            .field("cleanup_tasks_count", &self.cleanup_tasks.len())
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl TestCleanupManager {
    /// Create new cleanup manager
    pub fn new(timeout: Duration) -> Self {
        Self {
            cleanup_tasks: Vec::new(),
            timeout,
        }
    }

    /// Add cleanup task
    pub fn add_cleanup<F>(&mut self, task: F)
    where
        F: Fn() -> Result<()> + Send + Sync + 'static,
    {
        self.cleanup_tasks.push(Box::new(task));
    }

    /// Execute all cleanup tasks
    pub async fn cleanup(&self) -> Result<()> {
        for task in &self.cleanup_tasks {
            if let Err(e) = task() {
                eprintln!("Cleanup task failed: {}", e);
            }
        }
        Ok(())
    }
}

/// Environment registry for tracking test environments
pub struct EnvironmentRegistry;

impl EnvironmentRegistry {
    /// Register test environment
    pub fn register(name: String, environment: TestEnvironment) -> Result<()> {
        let mut registry = TEST_REGISTRY
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire registry lock"))?;

        if registry.contains_key(&name) {
            return Err(anyhow::anyhow!(
                "Test environment '{}' already registered",
                name
            ));
        }

        registry.insert(name, environment);
        Ok(())
    }

    /// Unregister test environment
    pub fn unregister(name: &str) -> Result<()> {
        let mut registry = TEST_REGISTRY
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire registry lock"))?;

        registry.remove(name);
        Ok(())
    }

    /// Get test environment by removing it from registry
    pub fn take(name: &str) -> Result<Option<TestEnvironment>> {
        let mut registry = TEST_REGISTRY
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire registry lock"))?;

        // Remove and return the environment from registry
        Ok(registry.remove(name))
    }

    /// Check if test environment exists
    pub fn contains(name: &str) -> Result<bool> {
        let registry = TEST_REGISTRY
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire registry lock"))?;

        Ok(registry.contains_key(name))
    }

    /// Cleanup all registered environments
    pub async fn cleanup_all() -> Result<()> {
        let registry = TEST_REGISTRY
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire registry lock"))?;

        for (name, _env) in registry.iter() {
            println!("Cleaning up test environment: {}", name);
            // Cleanup logic would go here
        }

        Ok(())
    }
}

/// Convenience macros for test environment setup
#[macro_export]
macro_rules! isolated_test {
    ($test_name:expr, $config:expr, $test_body:expr) => {{
        let env = create_isolated_test_environment($config).await?;
        let result = $test_body(&env).await;

        // Cleanup
        let cleanup_manager = TestCleanupManager::new(Duration::from_secs(30));
        cleanup_manager.cleanup().await?;

        result
    }};
}

#[macro_export]
macro_rules! basic_test_env {
    ($test_name:expr) => {
        TestEnvironmentBuilder::new($test_name)
            .isolation_level(IsolationLevel::Basic)
            .build()
            .await?
    };
}

#[macro_export]
macro_rules! auth_test_env {
    ($test_name:expr) => {
        TestEnvironmentBuilder::new($test_name)
            .isolation_level(IsolationLevel::Basic)
            .with_auth(true)
            .build()
            .await?
    };
}

#[macro_export]
macro_rules! full_test_env {
    ($test_name:expr) => {
        TestEnvironmentBuilder::new($test_name)
            .isolation_level(IsolationLevel::Basic)
            .with_auth(true)
            .with_caching(true)
            .with_streaming(true)
            .with_shadow(true)
            .build()
            .await?
    };
}

impl Default for TestFixtures {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions for common test scenarios
pub fn create_test_inference_request() -> Value {
    serde_json::json!({
        "text": "Test inference request",
        "max_length": 100,
        "temperature": 0.7
    })
}

pub fn create_test_batch_request() -> Value {
    serde_json::json!({
        "requests": [
            {
                "text": "Batch test 1",
                "max_length": 50,
                "temperature": 0.5
            },
            {
                "text": "Batch test 2",
                "max_length": 50,
                "temperature": 0.5
            },
            {
                "text": "Batch test 3",
                "max_length": 50,
                "temperature": 0.5
            }
        ]
    })
}

pub fn create_test_auth_request() -> Value {
    serde_json::json!({
        "username": "test-user",
        "password": "test-password"
    })
}
