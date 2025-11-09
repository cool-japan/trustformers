//! Test Environment Isolation Framework for TrustformeRS Serve
//!
//! Provides comprehensive test environment isolation to prevent test interference,
//! ensure reproducible results, and improve test reliability through resource
//! isolation, temporary directories, port management, and cleanup mechanisms.

// Allow dead code for test infrastructure under development
#![allow(dead_code)]

use std::collections::HashMap;
use std::env;
use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::sync::RwLock;
use tokio::time::sleep;
use uuid::Uuid;

/// Manages isolated test environments with automatic cleanup
#[derive(Debug)]
pub struct TestEnvironmentManager {
    /// Active test environments
    environments: Arc<RwLock<HashMap<String, TestEnvironment>>>,
    /// Global resource locks to prevent conflicts
    resource_locks: Arc<Mutex<ResourceLocks>>,
    /// Configuration for environment management
    config: EnvironmentConfig,
}

/// Individual test environment with isolated resources
pub struct TestEnvironment {
    /// Unique environment ID
    pub id: String,
    /// Temporary directory for this test
    pub temp_dir: TempDir,
    /// Isolated port assignments
    pub ports: PortAllocation,
    /// Environment variables specific to this test
    pub env_vars: HashMap<String, String>,
    /// Database/storage isolation
    pub storage: StorageIsolation,
    /// Network isolation settings
    pub network: NetworkIsolation,
    /// Cleanup handlers
    cleanup_handlers: Vec<Box<dyn FnOnce() + Send + Sync>>,
    /// Creation timestamp
    created_at: Instant,
}

impl std::fmt::Debug for TestEnvironment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestEnvironment")
            .field("id", &self.id)
            .field("temp_dir", &self.temp_dir)
            .field("ports", &self.ports)
            .field("env_vars", &self.env_vars)
            .field("storage", &self.storage)
            .field("network", &self.network)
            .field("cleanup_handlers_count", &self.cleanup_handlers.len())
            .field("created_at", &self.created_at)
            .finish()
    }
}

/// Port allocation for isolated testing
#[derive(Debug, Clone)]
pub struct PortAllocation {
    /// Main server port
    pub server_port: u16,
    /// Metrics endpoint port
    pub metrics_port: u16,
    /// Health check port
    pub health_port: u16,
    /// Admin interface port
    pub admin_port: u16,
    /// Message queue port
    pub message_queue_port: u16,
    /// Redis cache port (if used)
    pub redis_port: Option<u16>,
    /// Database port (if used)
    pub database_port: Option<u16>,
}

/// Storage isolation configuration
#[derive(Debug)]
pub struct StorageIsolation {
    /// Model registry path
    pub model_registry_path: PathBuf,
    /// Cache directory
    pub cache_directory: PathBuf,
    /// Log directory
    pub log_directory: PathBuf,
    /// Temporary model storage
    pub model_storage_path: PathBuf,
    /// Configuration files directory
    pub config_directory: PathBuf,
}

/// Network isolation settings
#[derive(Debug)]
pub struct NetworkIsolation {
    /// Isolated network namespace (if supported)
    pub network_namespace: Option<String>,
    /// Custom hosts file entries
    pub hosts_entries: HashMap<String, String>,
    /// Firewall rules (test-specific)
    pub firewall_rules: Vec<String>,
}

/// Global resource locks to prevent conflicts
#[derive(Debug, Default)]
struct ResourceLocks {
    /// Reserved ports across all tests
    reserved_ports: std::collections::HashSet<u16>,
    /// Reserved directories
    reserved_directories: std::collections::HashSet<PathBuf>,
    /// Active database instances
    active_databases: std::collections::HashSet<String>,
}

/// Configuration for environment management
#[derive(Debug, Clone)]
pub struct EnvironmentConfig {
    /// Base port range for allocation
    pub port_range: (u16, u16),
    /// Maximum number of concurrent environments
    pub max_concurrent_environments: usize,
    /// Environment timeout (auto-cleanup)
    pub environment_timeout: Duration,
    /// Enable network isolation
    pub enable_network_isolation: bool,
    /// Enable storage isolation
    pub enable_storage_isolation: bool,
    /// Base directory for temporary files
    pub temp_base_dir: Option<PathBuf>,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            port_range: (30000, 40000),
            max_concurrent_environments: 50,
            environment_timeout: Duration::from_secs(300), // 5 minutes
            enable_network_isolation: true,
            enable_storage_isolation: true,
            temp_base_dir: None,
        }
    }
}

impl TestEnvironmentManager {
    /// Create a new test environment manager
    pub fn new(config: EnvironmentConfig) -> Self {
        Self {
            environments: Arc::new(RwLock::new(HashMap::new())),
            resource_locks: Arc::new(Mutex::new(ResourceLocks::default())),
            config,
        }
    }

    /// Create a new isolated test environment
    pub async fn create_environment(&self, test_name: &str) -> Result<String, IsolationError> {
        let env_id = format!("{}_{}", test_name, Uuid::new_v4());

        // Check concurrent environment limit
        {
            let environments = self.environments.read().await;
            if environments.len() >= self.config.max_concurrent_environments {
                return Err(IsolationError::TooManyEnvironments);
            }
        }

        // Allocate resources
        let temp_dir = self.create_temp_directory(&env_id)?;
        let ports = self.allocate_ports().await?;
        let storage = self.create_storage_isolation(&temp_dir)?;
        let network = self.create_network_isolation(&env_id)?;
        let env_vars = self.create_environment_variables(&env_id, &ports, &storage)?;

        let environment = TestEnvironment {
            id: env_id.clone(),
            temp_dir,
            ports,
            env_vars,
            storage,
            network,
            cleanup_handlers: Vec::new(),
            created_at: Instant::now(),
        };

        // Register environment
        {
            let mut environments = self.environments.write().await;
            environments.insert(env_id.clone(), environment);
        }

        Ok(env_id)
    }

    /// Get environment by ID
    pub async fn get_environment(&self, env_id: &str) -> Option<TestEnvironment> {
        let environments = self.environments.read().await;
        // Note: This returns a copy/clone of the environment
        // In a real implementation, you'd want to return a reference or handle
        environments.get(env_id).cloned()
    }

    /// Clean up a specific environment
    pub async fn cleanup_environment(&self, env_id: &str) -> Result<(), IsolationError> {
        let environment = {
            let mut environments = self.environments.write().await;
            environments.remove(env_id)
        };

        if let Some(mut env) = environment {
            // Run cleanup handlers
            for handler in env.cleanup_handlers.drain(..) {
                handler();
            }

            // Release ports
            self.release_ports(&env.ports).await?;

            // Clean up network isolation
            self.cleanup_network_isolation(&env.network)?;

            // Temporary directory is automatically cleaned up when dropped
        }

        Ok(())
    }

    /// Clean up all environments
    pub async fn cleanup_all(&self) -> Result<(), IsolationError> {
        let env_ids: Vec<String> = {
            let environments = self.environments.read().await;
            environments.keys().cloned().collect()
        };

        for env_id in env_ids {
            self.cleanup_environment(&env_id).await?;
        }

        Ok(())
    }

    /// Clean up expired environments
    pub async fn cleanup_expired(&self) -> Result<usize, IsolationError> {
        let now = Instant::now();
        let expired_ids: Vec<String> = {
            let environments = self.environments.read().await;
            environments
                .iter()
                .filter(|(_, env)| {
                    now.duration_since(env.created_at) > self.config.environment_timeout
                })
                .map(|(id, _)| id.clone())
                .collect()
        };

        let count = expired_ids.len();
        for env_id in expired_ids {
            self.cleanup_environment(&env_id).await?;
        }

        Ok(count)
    }

    // Private helper methods

    fn create_temp_directory(&self, env_id: &str) -> Result<TempDir, IsolationError> {
        let base_dir = self.config.temp_base_dir.as_deref().unwrap_or_else(|| Path::new("/tmp"));

        tempfile::Builder::new()
            .prefix(&format!("trustformers_test_{}_", env_id))
            .tempdir_in(base_dir)
            .map_err(IsolationError::TempDirCreation)
    }

    async fn allocate_ports(&self) -> Result<PortAllocation, IsolationError> {
        let mut resource_locks = self.resource_locks.lock().unwrap();

        let mut allocated_ports = Vec::new();
        let (start_port, end_port) = self.config.port_range;

        // Allocate required ports
        for _ in 0..7 {
            // We need 7 ports (5 main + 2 optional)
            let port =
                self.find_available_port(start_port, end_port, &resource_locks.reserved_ports)?;
            resource_locks.reserved_ports.insert(port);
            allocated_ports.push(port);
        }

        Ok(PortAllocation {
            server_port: allocated_ports[0],
            metrics_port: allocated_ports[1],
            health_port: allocated_ports[2],
            admin_port: allocated_ports[3],
            message_queue_port: allocated_ports[4],
            redis_port: Some(allocated_ports[5]),
            database_port: Some(allocated_ports[6]),
        })
    }

    fn find_available_port(
        &self,
        start: u16,
        end: u16,
        reserved: &std::collections::HashSet<u16>,
    ) -> Result<u16, IsolationError> {
        for port in start..=end {
            if reserved.contains(&port) {
                continue;
            }

            // Check if port is actually available
            if let Ok(listener) = TcpListener::bind(("127.0.0.1", port)) {
                drop(listener);
                return Ok(port);
            }
        }

        Err(IsolationError::NoAvailablePorts)
    }

    async fn release_ports(&self, ports: &PortAllocation) -> Result<(), IsolationError> {
        let mut resource_locks = self.resource_locks.lock().unwrap();

        resource_locks.reserved_ports.remove(&ports.server_port);
        resource_locks.reserved_ports.remove(&ports.metrics_port);
        resource_locks.reserved_ports.remove(&ports.health_port);
        resource_locks.reserved_ports.remove(&ports.admin_port);
        resource_locks.reserved_ports.remove(&ports.message_queue_port);

        if let Some(port) = ports.redis_port {
            resource_locks.reserved_ports.remove(&port);
        }
        if let Some(port) = ports.database_port {
            resource_locks.reserved_ports.remove(&port);
        }

        Ok(())
    }

    fn create_storage_isolation(
        &self,
        temp_dir: &TempDir,
    ) -> Result<StorageIsolation, IsolationError> {
        let base_path = temp_dir.path();

        let storage = StorageIsolation {
            model_registry_path: base_path.join("model_registry"),
            cache_directory: base_path.join("cache"),
            log_directory: base_path.join("logs"),
            model_storage_path: base_path.join("models"),
            config_directory: base_path.join("config"),
        };

        // Create directories
        fs::create_dir_all(&storage.model_registry_path)
            .map_err(IsolationError::DirectoryCreation)?;
        fs::create_dir_all(&storage.cache_directory).map_err(IsolationError::DirectoryCreation)?;
        fs::create_dir_all(&storage.log_directory).map_err(IsolationError::DirectoryCreation)?;
        fs::create_dir_all(&storage.model_storage_path)
            .map_err(IsolationError::DirectoryCreation)?;
        fs::create_dir_all(&storage.config_directory).map_err(IsolationError::DirectoryCreation)?;

        Ok(storage)
    }

    fn create_network_isolation(&self, env_id: &str) -> Result<NetworkIsolation, IsolationError> {
        let mut hosts_entries = HashMap::new();

        // Add test-specific host entries
        hosts_entries.insert("test-redis".to_string(), "127.0.0.1".to_string());
        hosts_entries.insert("test-database".to_string(), "127.0.0.1".to_string());
        hosts_entries.insert(format!("test-server-{}", env_id), "127.0.0.1".to_string());

        Ok(NetworkIsolation {
            network_namespace: if self.config.enable_network_isolation {
                Some(format!("trustformers-test-{}", env_id))
            } else {
                None
            },
            hosts_entries,
            firewall_rules: Vec::new(),
        })
    }

    fn cleanup_network_isolation(&self, _network: &NetworkIsolation) -> Result<(), IsolationError> {
        // In a real implementation, this would clean up network namespaces,
        // restore original hosts file, remove firewall rules, etc.
        Ok(())
    }

    fn create_environment_variables(
        &self,
        env_id: &str,
        ports: &PortAllocation,
        storage: &StorageIsolation,
    ) -> Result<HashMap<String, String>, IsolationError> {
        let mut env_vars = HashMap::new();

        // Test environment identification
        env_vars.insert("TRUSTFORMERS_TEST_ENV_ID".to_string(), env_id.to_string());
        env_vars.insert("TRUSTFORMERS_TEST_MODE".to_string(), "true".to_string());

        // Port configuration
        env_vars.insert(
            "TRUSTFORMERS_SERVER_PORT".to_string(),
            ports.server_port.to_string(),
        );
        env_vars.insert(
            "TRUSTFORMERS_METRICS_PORT".to_string(),
            ports.metrics_port.to_string(),
        );
        env_vars.insert(
            "TRUSTFORMERS_HEALTH_PORT".to_string(),
            ports.health_port.to_string(),
        );
        env_vars.insert(
            "TRUSTFORMERS_ADMIN_PORT".to_string(),
            ports.admin_port.to_string(),
        );
        env_vars.insert(
            "TRUSTFORMERS_QUEUE_PORT".to_string(),
            ports.message_queue_port.to_string(),
        );

        // Storage paths
        env_vars.insert(
            "TRUSTFORMERS_MODEL_REGISTRY".to_string(),
            storage.model_registry_path.to_string_lossy().to_string(),
        );
        env_vars.insert(
            "TRUSTFORMERS_CACHE_DIR".to_string(),
            storage.cache_directory.to_string_lossy().to_string(),
        );
        env_vars.insert(
            "TRUSTFORMERS_LOG_DIR".to_string(),
            storage.log_directory.to_string_lossy().to_string(),
        );
        env_vars.insert(
            "TRUSTFORMERS_MODEL_STORAGE".to_string(),
            storage.model_storage_path.to_string_lossy().to_string(),
        );
        env_vars.insert(
            "TRUSTFORMERS_CONFIG_DIR".to_string(),
            storage.config_directory.to_string_lossy().to_string(),
        );

        // Redis configuration (if enabled)
        if let Some(redis_port) = ports.redis_port {
            env_vars.insert(
                "REDIS_URL".to_string(),
                format!("redis://127.0.0.1:{}", redis_port),
            );
        }

        // Database configuration (if enabled)
        if let Some(db_port) = ports.database_port {
            env_vars.insert(
                "DATABASE_URL".to_string(),
                format!(
                    "postgresql://test:test@127.0.0.1:{}/test_{}",
                    db_port, env_id
                ),
            );
        }

        // Disable external dependencies in test mode
        env_vars.insert(
            "TRUSTFORMERS_DISABLE_TELEMETRY".to_string(),
            "true".to_string(),
        );
        env_vars.insert(
            "TRUSTFORMERS_DISABLE_ANALYTICS".to_string(),
            "true".to_string(),
        );
        env_vars.insert(
            "TRUSTFORMERS_MOCK_EXTERNAL_SERVICES".to_string(),
            "true".to_string(),
        );

        Ok(env_vars)
    }
}

// Clone implementation for TestEnvironment (simplified for demonstration)
impl Clone for TestEnvironment {
    fn clone(&self) -> Self {
        // Note: In a real implementation, you'd want a more sophisticated
        // approach to sharing test environments or using reference counting
        Self {
            id: self.id.clone(),
            temp_dir: tempfile::tempdir().unwrap(), // Create new temp dir
            ports: self.ports.clone(),
            env_vars: self.env_vars.clone(),
            storage: StorageIsolation {
                model_registry_path: self.storage.model_registry_path.clone(),
                cache_directory: self.storage.cache_directory.clone(),
                log_directory: self.storage.log_directory.clone(),
                model_storage_path: self.storage.model_storage_path.clone(),
                config_directory: self.storage.config_directory.clone(),
            },
            network: NetworkIsolation {
                network_namespace: self.network.network_namespace.clone(),
                hosts_entries: self.network.hosts_entries.clone(),
                firewall_rules: self.network.firewall_rules.clone(),
            },
            cleanup_handlers: Vec::new(),
            created_at: self.created_at,
        }
    }
}

/// Errors that can occur during environment isolation
#[derive(Debug, thiserror::Error)]
pub enum IsolationError {
    #[error("Too many concurrent test environments")]
    TooManyEnvironments,

    #[error("No available ports in the specified range")]
    NoAvailablePorts,

    #[error("Failed to create temporary directory: {0}")]
    TempDirCreation(std::io::Error),

    #[error("Failed to create directory: {0}")]
    DirectoryCreation(std::io::Error),

    #[error("Network isolation setup failed: {0}")]
    NetworkSetup(String),

    #[error("Environment not found: {0}")]
    EnvironmentNotFound(String),

    #[error("Resource cleanup failed: {0}")]
    CleanupFailed(String),
}

/// Test helper macros and utilities
pub mod test_helpers {
    use super::*;
    use std::future::Future;

    /// Macro to run a test with isolated environment
    #[macro_export]
    macro_rules! env_isolated_test {
        ($test_name:ident, $test_fn:expr) => {
            #[tokio::test]
            async fn $test_name() {
                let manager = TestEnvironmentManager::new(EnvironmentConfig::default());
                let env_id = manager
                    .create_environment(stringify!($test_name))
                    .await
                    .expect("Failed to create test environment");

                let env =
                    manager.get_environment(&env_id).await.expect("Failed to get test environment");

                // Set environment variables
                for (key, value) in &env.env_vars {
                    std::env::set_var(key, value);
                }

                // Run the test
                let result = std::panic::AssertUnwindSafe($test_fn(env)).catch_unwind().await;

                // Clean up environment
                manager
                    .cleanup_environment(&env_id)
                    .await
                    .expect("Failed to cleanup test environment");

                // Propagate test result
                match result {
                    Ok(Ok(())) => (),
                    Ok(Err(e)) => panic!("Test failed: {:?}", e),
                    Err(panic) => std::panic::resume_unwind(panic),
                }
            }
        };
    }

    /// Create an isolated test environment for manual management
    pub async fn create_isolated_environment(
        test_name: &str,
    ) -> Result<(TestEnvironmentManager, String), IsolationError> {
        let manager = TestEnvironmentManager::new(EnvironmentConfig::default());
        let env_id = manager.create_environment(test_name).await?;
        Ok((manager, env_id))
    }

    /// Run a function with environment variables temporarily set
    pub async fn with_env_vars<F, Fut, T>(env_vars: &HashMap<String, String>, f: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        // Save original environment
        let original_env: HashMap<String, String> = env_vars
            .keys()
            .filter_map(|key| env::var(key).ok().map(|value| (key.clone(), value)))
            .collect();

        // Set test environment variables
        for (key, value) in env_vars {
            env::set_var(key, value);
        }

        // Run the function
        let result = f().await;

        // Restore original environment
        for key in env_vars.keys() {
            match original_env.get(key) {
                Some(original_value) => env::set_var(key, original_value),
                None => env::remove_var(key),
            }
        }

        result
    }

    /// Wait for a port to become available
    pub async fn wait_for_port(port: u16, timeout: Duration) -> Result<(), IsolationError> {
        let start = Instant::now();

        while start.elapsed() < timeout {
            if TcpListener::bind(("127.0.0.1", port)).is_ok() {
                return Ok(());
            }
            sleep(Duration::from_millis(100)).await;
        }

        Err(IsolationError::CleanupFailed(format!(
            "Port {} did not become available within timeout",
            port
        )))
    }

    /// Create a test-specific configuration file
    pub fn create_test_config_file(
        storage: &StorageIsolation,
        ports: &PortAllocation,
        filename: &str,
        content: &str,
    ) -> Result<PathBuf, IsolationError> {
        let config_path = storage.config_directory.join(filename);

        // Replace port placeholders in content
        let processed_content = content
            .replace("{{SERVER_PORT}}", &ports.server_port.to_string())
            .replace("{{METRICS_PORT}}", &ports.metrics_port.to_string())
            .replace("{{HEALTH_PORT}}", &ports.health_port.to_string())
            .replace("{{ADMIN_PORT}}", &ports.admin_port.to_string())
            .replace("{{QUEUE_PORT}}", &ports.message_queue_port.to_string());

        fs::write(&config_path, processed_content).map_err(IsolationError::DirectoryCreation)?;

        Ok(config_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_environment_creation_and_cleanup() {
        let manager = TestEnvironmentManager::new(EnvironmentConfig::default());

        let env_id = manager
            .create_environment("test_basic")
            .await
            .expect("Failed to create environment");

        assert!(!env_id.is_empty());
        assert!(env_id.starts_with("test_basic_"));

        let env = manager.get_environment(&env_id).await.expect("Failed to get environment");

        // Verify environment has required resources
        assert!(env.temp_dir.path().exists());
        assert!(env.ports.server_port > 0);
        assert!(env.env_vars.contains_key("TRUSTFORMERS_TEST_ENV_ID"));

        // Clean up
        manager
            .cleanup_environment(&env_id)
            .await
            .expect("Failed to cleanup environment");
    }

    #[tokio::test]
    async fn test_concurrent_environments() {
        let manager = TestEnvironmentManager::new(EnvironmentConfig {
            max_concurrent_environments: 3,
            ..Default::default()
        });

        // Create multiple environments
        let env1 = manager.create_environment("test_concurrent_1").await.unwrap();
        let env2 = manager.create_environment("test_concurrent_2").await.unwrap();
        let env3 = manager.create_environment("test_concurrent_3").await.unwrap();

        // Fourth should fail
        let env4_result = manager.create_environment("test_concurrent_4").await;
        assert!(matches!(
            env4_result,
            Err(IsolationError::TooManyEnvironments)
        ));

        // Clean up
        manager.cleanup_environment(&env1).await.unwrap();
        manager.cleanup_environment(&env2).await.unwrap();
        manager.cleanup_environment(&env3).await.unwrap();
    }

    #[tokio::test]
    async fn test_port_allocation_isolation() {
        let manager = TestEnvironmentManager::new(EnvironmentConfig::default());

        let env1_id = manager.create_environment("test_ports_1").await.unwrap();
        let env2_id = manager.create_environment("test_ports_2").await.unwrap();

        let env1 = manager.get_environment(&env1_id).await.unwrap();
        let env2 = manager.get_environment(&env2_id).await.unwrap();

        // Ports should be different
        assert_ne!(env1.ports.server_port, env2.ports.server_port);
        assert_ne!(env1.ports.metrics_port, env2.ports.metrics_port);
        assert_ne!(env1.ports.health_port, env2.ports.health_port);

        // Clean up
        manager.cleanup_environment(&env1_id).await.unwrap();
        manager.cleanup_environment(&env2_id).await.unwrap();
    }

    #[tokio::test]
    async fn test_storage_isolation() {
        let manager = TestEnvironmentManager::new(EnvironmentConfig::default());

        let env1_id = manager.create_environment("test_storage_1").await.unwrap();
        let env2_id = manager.create_environment("test_storage_2").await.unwrap();

        let env1 = manager.get_environment(&env1_id).await.unwrap();
        let env2 = manager.get_environment(&env2_id).await.unwrap();

        // Storage paths should be different
        assert_ne!(
            env1.storage.model_registry_path,
            env2.storage.model_registry_path
        );
        assert_ne!(env1.storage.cache_directory, env2.storage.cache_directory);

        // Directories should exist
        assert!(env1.storage.model_registry_path.exists());
        assert!(env2.storage.cache_directory.exists());

        // Clean up
        manager.cleanup_environment(&env1_id).await.unwrap();
        manager.cleanup_environment(&env2_id).await.unwrap();
    }

    #[tokio::test]
    async fn test_environment_timeout_cleanup() {
        let manager = TestEnvironmentManager::new(EnvironmentConfig {
            environment_timeout: Duration::from_millis(100), // Very short timeout
            ..Default::default()
        });

        let env_id = manager.create_environment("test_timeout").await.unwrap();

        // Wait for timeout
        sleep(Duration::from_millis(200)).await;

        // Clean up expired environments
        let cleaned_count = manager.cleanup_expired().await.unwrap();
        assert_eq!(cleaned_count, 1);

        // Environment should no longer exist
        assert!(manager.get_environment(&env_id).await.is_none());
    }

    #[tokio::test]
    async fn test_helper_functions() {
        use test_helpers::*;

        let (manager, env_id) = create_isolated_environment("test_helpers")
            .await
            .expect("Failed to create isolated environment");

        let env = manager.get_environment(&env_id).await.unwrap();

        // Test with_env_vars helper
        let result = with_env_vars(&env.env_vars, || async {
            std::env::var("TRUSTFORMERS_TEST_ENV_ID").unwrap()
        })
        .await;

        assert_eq!(result, env_id);

        // Test port waiting helper
        let port_result = timeout(
            Duration::from_secs(1),
            wait_for_port(env.ports.server_port, Duration::from_millis(500)),
        )
        .await;

        assert!(port_result.is_ok());

        // Clean up
        manager.cleanup_environment(&env_id).await.unwrap();
    }
}
