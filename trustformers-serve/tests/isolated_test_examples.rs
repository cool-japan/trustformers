//! Examples of using the test environment isolation framework
//!
//! Demonstrates practical usage of the isolation framework with actual
//! TrustformeRS Serve tests to ensure reliable, non-interfering test execution.

mod test_environment_isolation;
mod test_helpers;

use axum_test::{http::StatusCode, TestServer};
use serde_json::{json, Value};
use std::{sync::Arc, time::Duration};
use test_environment_isolation::{
    test_helpers::{
        create_isolated_environment, create_test_config_file, wait_for_port, with_env_vars,
    },
    EnvironmentConfig, TestEnvironmentManager,
};
use tokio::time::sleep;
use trustformers_serve::{Device, ModelConfig, ServerConfig, TrustformerServer};

/// Example of a test using the isolation framework
#[tokio::test]
async fn test_isolated_auth_flow() {
    let (manager, env_id) = create_isolated_environment("auth_flow")
        .await
        .expect("Failed to create isolated environment");

    let env = manager.get_environment(&env_id).await.unwrap();

    // Set environment variables for this test
    with_env_vars(&env.env_vars, || async {
        // Create server config using isolated ports and paths
        let mut config = ServerConfig::default();
        config.host = "127.0.0.1".to_string();
        config.port = env.ports.server_port;

        config.model_config = ModelConfig {
            model_name: "isolated-test-model".to_string(),
            model_version: Some("1.0.0".to_string()),
            device: Device::Cpu,
            max_sequence_length: 1024,
            enable_caching: true,
        };

        // Note: Authentication is now handled at router level, not in ServerConfig

        let server = TrustformerServer::new(config);
        let router = server.create_test_router().await;
        let test_server = TestServer::new(router).unwrap();

        // Test authentication flow in isolation
        let response = test_server
            .post("/auth/token")
            .json(&json!({
                "username": "test_user",
                "password": "test_password"
            }))
            .await;

        assert_eq!(response.status_code(), StatusCode::OK);
        let token_response: Value = response.json();
        assert!(token_response["access_token"].is_string());

        // Test that this token works for protected endpoints
        let token = token_response["access_token"].as_str().unwrap();
        let metrics_response = test_server
            .get("/metrics")
            .add_header("Authorization", &format!("Bearer {}", token))
            .await;

        assert_eq!(metrics_response.status_code(), StatusCode::OK);
    })
    .await;

    // Clean up
    manager.cleanup_environment(&env_id).await.unwrap();
}

/// Example showing parallel isolated tests that don't interfere
#[tokio::test]
async fn test_parallel_isolated_inference() {
    let manager = Arc::new(TestEnvironmentManager::new(EnvironmentConfig::default()));

    // Create multiple isolated environments for parallel testing
    let env1_id = manager.create_environment("parallel_test_1").await.unwrap();
    let env2_id = manager.create_environment("parallel_test_2").await.unwrap();
    let env3_id = manager.create_environment("parallel_test_3").await.unwrap();

    let env1 = manager.get_environment(&env1_id).await.unwrap();
    let env2 = manager.get_environment(&env2_id).await.unwrap();
    let env3 = manager.get_environment(&env3_id).await.unwrap();

    // Run tests sequentially to avoid Send issues with TestServer
    // Each test is still isolated due to different environments
    run_isolated_inference_test(env1, "model-1".to_string()).await.unwrap();
    run_isolated_inference_test(env2, "model-2".to_string()).await.unwrap();
    run_isolated_inference_test(env3, "model-3".to_string()).await.unwrap();

    // Clean up all environments
    manager.cleanup_environment(&env1_id).await.unwrap();
    manager.cleanup_environment(&env2_id).await.unwrap();
    manager.cleanup_environment(&env3_id).await.unwrap();
}

async fn run_isolated_inference_test(
    env: crate::test_environment_isolation::TestEnvironment,
    model_name: String,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    with_env_vars(&env.env_vars, || async {
        let mut config = ServerConfig::default();
        config.host = "127.0.0.1".to_string();
        config.port = env.ports.server_port;

        config.model_config = ModelConfig {
            model_name: model_name.clone(),
            model_version: Some("1.0.0".to_string()),
            device: Device::Cpu,
            max_sequence_length: 512,
            enable_caching: false, // Disable caching to avoid interference
        };

        let server = TrustformerServer::new(config);
        let router = server.create_test_router().await;
        let test_server = TestServer::new(router).unwrap();

        // Each test uses different model and isolated resources
        let response = test_server
            .post("/inference")
            .json(&json!({
                "text": format!("Test input for {}", model_name),
                "model": model_name,
                "max_length": 50
            }))
            .await;

        assert_eq!(response.status_code(), StatusCode::OK);
        let result: Value = response.json();
        assert!(result["text"].is_string());

        Ok(())
    })
    .await
}

/// Example showing isolation of file system resources
#[tokio::test]
async fn test_isolated_model_management() {
    let (manager, env_id) = create_isolated_environment("model_management")
        .await
        .expect("Failed to create isolated environment");

    let env = manager.get_environment(&env_id).await.unwrap();

    with_env_vars(&env.env_vars, || async {
        // Create test model files in isolated storage
        let model_path = env.storage.model_storage_path.join("test_model.bin");
        std::fs::write(&model_path, b"fake model data").unwrap();

        let config_content = format!(
            r#"
        [server]
        host = "127.0.0.1"
        port = {{{{SERVER_PORT}}}}

        [model]
        model_name = "isolated_model"
        model_path = "{}"
        device = "cpu"

        [storage]
        model_registry = "{}"
        cache_directory = "{}"
        "#,
            model_path.to_string_lossy(),
            env.storage.model_registry_path.to_string_lossy(),
            env.storage.cache_directory.to_string_lossy()
        );

        let config_path = create_test_config_file(
            &env.storage,
            &env.ports,
            "test_config.toml",
            &config_content,
        )
        .unwrap();

        // Verify isolated file system
        assert!(config_path.exists());
        assert!(model_path.exists());
        assert!(env.storage.model_registry_path.exists());
        assert!(env.storage.cache_directory.exists());

        // Each test environment has completely separate file systems
        assert!(config_path.to_string_lossy().contains(&env.id));

        // Test that the configuration works
        let config_contents = std::fs::read_to_string(&config_path).unwrap();
        assert!(config_contents.contains(&env.ports.server_port.to_string()));
    })
    .await;

    manager.cleanup_environment(&env_id).await.unwrap();
}

/// Example showing database isolation
#[tokio::test]
async fn test_isolated_database_operations() {
    let (manager, env_id) = create_isolated_environment("database_ops")
        .await
        .expect("Failed to create isolated environment");

    let env = manager.get_environment(&env_id).await.unwrap();

    with_env_vars(&env.env_vars, || async {
        // Each test gets its own database URL
        let db_url = std::env::var("DATABASE_URL").unwrap();
        assert!(db_url.contains(&env.id)); // Database name includes environment ID

        let mut config = ServerConfig::default();
        config.host = "127.0.0.1".to_string();
        config.port = env.ports.server_port;

        // In a real implementation, you would:
        // 1. Create isolated database instance
        // 2. Run migrations for this test environment
        // 3. Test database operations
        // 4. Database gets cleaned up with environment

        // For demonstration, we'll just verify the URL format
        assert!(db_url.starts_with("postgresql://"));
        assert!(db_url.contains("test_"));
        assert!(db_url.contains(&env.id));
    })
    .await;

    manager.cleanup_environment(&env_id).await.unwrap();
}

/// Example showing network isolation and port management
#[tokio::test]
async fn test_isolated_network_services() {
    let (manager, env_id) = create_isolated_environment("network_services")
        .await
        .expect("Failed to create isolated environment");

    let env = manager.get_environment(&env_id).await.unwrap();

    with_env_vars(&env.env_vars, || async {
        // Each test environment gets isolated ports for all services
        let server_port = env.ports.server_port;
        let metrics_port = env.ports.metrics_port;
        let health_port = env.ports.health_port;
        let admin_port = env.ports.admin_port;

        // Verify all ports are different and available
        assert_ne!(server_port, metrics_port);
        assert_ne!(server_port, health_port);
        assert_ne!(server_port, admin_port);
        assert_ne!(metrics_port, health_port);

        // Test that we can bind to these ports (they should be available)
        let server_listener = std::net::TcpListener::bind(("127.0.0.1", server_port)).unwrap();
        let metrics_listener = std::net::TcpListener::bind(("127.0.0.1", metrics_port)).unwrap();

        // Clean up listeners
        drop(server_listener);
        drop(metrics_listener);

        // Wait for ports to be released
        wait_for_port(server_port, Duration::from_millis(500)).await.unwrap();
        wait_for_port(metrics_port, Duration::from_millis(500)).await.unwrap();
    })
    .await;

    manager.cleanup_environment(&env_id).await.unwrap();
}

/// Example showing caching isolation
#[tokio::test]
async fn test_isolated_caching() {
    let manager = Arc::new(TestEnvironmentManager::new(EnvironmentConfig::default()));

    // Create two environments to test cache isolation
    let env1_id = manager.create_environment("cache_test_1").await.unwrap();
    let env2_id = manager.create_environment("cache_test_2").await.unwrap();

    let env1 = manager.get_environment(&env1_id).await.unwrap();
    let env2 = manager.get_environment(&env2_id).await.unwrap();

    // Test that caches are isolated between environments
    let result1 = with_env_vars(&env1.env_vars, || async {
        test_cache_operations("cache-key-1", "cache-value-1", &env1).await
    })
    .await;

    let result2 = with_env_vars(&env2.env_vars, || async {
        test_cache_operations("cache-key-1", "cache-value-2", &env2).await // Same key, different value
    })
    .await;

    // Both should succeed with their own isolated caches
    assert_eq!(result1, "cache-value-1");
    assert_eq!(result2, "cache-value-2");

    manager.cleanup_environment(&env1_id).await.unwrap();
    manager.cleanup_environment(&env2_id).await.unwrap();
}

async fn test_cache_operations(
    key: &str,
    value: &str,
    env: &crate::test_environment_isolation::TestEnvironment,
) -> String {
    // In a real implementation, this would:
    // 1. Set up cache with isolated storage directory
    // 2. Store the key-value pair
    // 3. Retrieve and return the value
    // 4. Verify isolation by using env.storage.cache_directory

    // For demonstration, we'll just verify the cache directory is isolated
    assert!(env.storage.cache_directory.to_string_lossy().contains(&env.id));

    // Simulate cache operation
    let cache_file = env.storage.cache_directory.join(format!("{}.cache", key));
    std::fs::write(&cache_file, value).unwrap();

    // Read back the cached value
    std::fs::read_to_string(&cache_file).unwrap()
}

/// Stress test with many concurrent isolated environments
#[tokio::test]
async fn test_concurrent_environment_stress() {
    let manager = Arc::new(TestEnvironmentManager::new(EnvironmentConfig {
        max_concurrent_environments: 20,
        ..Default::default()
    }));

    let mut env_ids = Vec::new();
    let mut handles = Vec::new();

    // Create many concurrent isolated environments
    for i in 0..15 {
        let env_id = manager.create_environment(&format!("stress_test_{}", i)).await.unwrap();
        env_ids.push(env_id.clone());

        let manager_clone = Arc::clone(&manager);
        let handle = tokio::spawn(async move {
            let env = manager_clone.get_environment(&env_id).await.unwrap();

            // Simulate some work in the isolated environment
            with_env_vars(&env.env_vars, || async {
                // Verify environment isolation
                let test_env_id = std::env::var("TRUSTFORMERS_TEST_ENV_ID").unwrap();
                assert_eq!(test_env_id, env_id);

                // Create some files in isolated storage
                let test_file = env.storage.cache_directory.join("stress_test.txt");
                std::fs::write(&test_file, format!("stress test data for {}", env_id)).unwrap();

                // Simulate work
                sleep(Duration::from_millis(100)).await;

                // Verify file still exists and has correct content
                let content = std::fs::read_to_string(&test_file).unwrap();
                assert!(content.contains(&env_id));
            })
            .await;
        });

        handles.push(handle);
    }

    // Wait for all concurrent tests to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Clean up all environments
    for env_id in env_ids {
        manager.cleanup_environment(&env_id).await.unwrap();
    }
}

// Using the isolation macro (if implemented)
/*
isolated_test!(
    test_with_isolation_macro,
    async |env: TestEnvironment| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // This test runs in complete isolation automatically
        let server_port = env.ports.server_port;

        // Test logic here - everything is automatically isolated
        assert!(server_port > 0);
        assert!(env.temp_dir.path().exists());

        Ok(())
    }
);
*/
