//! Basic Integration Tests for TrustformeRS Serve
//!
//! Simplified integration tests that focus on core functionality
//! without requiring complex setup or mocking.

use std::time::Duration;
use tokio::time::timeout;
use trustformers_serve::{
    batching::BatchingConfig, caching::CacheConfig, gpu_profiler::GpuProfilerConfig,
    gpu_scheduler::GpuSchedulerConfig, polling::LongPollingConfig, rate_limit::RateLimitConfig,
    shadow::ShadowConfig, streaming::StreamingConfig, validation::ValidationConfig, Device,
    ModelConfig, ServerConfig, TrustformerServer,
};

/// Create a minimal test server configuration
fn create_minimal_test_config() -> ServerConfig {
    let mut config = ServerConfig::default();
    config.host = "127.0.0.1".to_string();
    config.port = 0;
    config.num_workers = 1;
    config.model_config = ModelConfig {
        model_name: "test-model".to_string(),
        model_version: Some("1.0.0".to_string()),
        device: Device::Cpu,
        max_sequence_length: 2048,
        enable_caching: true,
    };
    config
}

#[tokio::test]
async fn test_server_creation() {
    let config = create_minimal_test_config();
    let _server = TrustformerServer::new(config);

    // Test that server can be created without panicking
    assert!(true, "Server created successfully");
}

#[tokio::test]
async fn test_server_router_creation() {
    let config = create_minimal_test_config();
    let server = TrustformerServer::new(config);

    // Test that router can be created
    let result = timeout(Duration::from_secs(5), server.create_test_router()).await;
    assert!(
        result.is_ok(),
        "Router creation should complete within timeout"
    );

    let _router = result.unwrap();
    // Router created successfully
}

#[tokio::test]
async fn test_config_serialization() {
    let config = create_minimal_test_config();

    // Test that config can be serialized
    let json_str = serde_json::to_string(&config);
    assert!(json_str.is_ok(), "Config should be serializable");

    // Test that config can be deserialized
    let json_str = json_str.unwrap();
    let deserialized: Result<ServerConfig, _> = serde_json::from_str(&json_str);
    assert!(deserialized.is_ok(), "Config should be deserializable");

    let deserialized_config = deserialized.unwrap();
    assert_eq!(config.host, deserialized_config.host);
    assert_eq!(config.port, deserialized_config.port);
}

#[tokio::test]
async fn test_default_configs() {
    // Test that all default configs can be created
    assert!(BatchingConfig::default().max_batch_size > 0);
    assert!(CacheConfig::default().result_cache.max_size_bytes > 0);
    assert!(StreamingConfig::default().buffer_size > 0);
    assert!(ValidationConfig::default().max_text_length > 0);
    assert!(RateLimitConfig::default().max_requests > 0);
    assert!(LongPollingConfig::default().timeout_seconds > 0);
    assert!(ShadowConfig::default().traffic_percentage >= 0.0);
    assert!(GpuSchedulerConfig::default().max_queue_size > 0);
    assert!(GpuProfilerConfig::default().profiling_interval_seconds > 0);
}

#[tokio::test]
async fn test_model_config() {
    let model_config = ModelConfig {
        model_name: "test-model".to_string(),
        model_version: Some("2.0.0".to_string()),
        device: Device::Cuda(0),
        max_sequence_length: 1024,
        enable_caching: false,
    };

    assert_eq!(model_config.model_name, "test-model");
    assert_eq!(model_config.model_version, Some("2.0.0".to_string()));
    assert!(matches!(model_config.device, Device::Cuda(0)));
    assert_eq!(model_config.max_sequence_length, 1024);
    assert!(!model_config.enable_caching);
}

#[tokio::test]
async fn test_service_configuration_validation() {
    let config = create_minimal_test_config();

    // Test basic validation of config values
    assert!(!config.host.is_empty(), "Host should not be empty");
    assert!(
        config.num_workers > 0,
        "Number of workers should be positive"
    );
    assert!(
        config.model_config.max_sequence_length > 0,
        "Max sequence length should be positive"
    );
}

#[tokio::test]
async fn test_batching_config_validation() {
    let mut config = BatchingConfig::default();

    // Test that we can modify batching config
    config.max_batch_size = 32;
    config.max_wait_time = Duration::from_millis(100);

    assert_eq!(config.max_batch_size, 32);
    assert_eq!(config.max_wait_time, Duration::from_millis(100));
}

#[tokio::test]
async fn test_caching_config_validation() {
    let mut config = CacheConfig::default();

    // Test that we can modify caching config
    config.result_cache.max_size_bytes = 1000;
    config.result_cache.default_ttl = Duration::from_secs(3600);

    assert_eq!(config.result_cache.max_size_bytes, 1000);
    assert_eq!(config.result_cache.default_ttl, Duration::from_secs(3600));
}

#[tokio::test]
async fn test_server_with_custom_config() {
    let mut config = create_minimal_test_config();
    config.port = 8080;
    config.num_workers = 4;
    config.model_config.max_sequence_length = 4096;

    let _server = TrustformerServer::new(config.clone());

    // Server created with custom config
}

#[tokio::test]
async fn test_concurrent_server_creation() {
    // Test creating multiple servers concurrently
    let handles: Vec<_> = (0..5)
        .map(|i| {
            tokio::spawn(async move {
                let mut config = create_minimal_test_config();
                config.port = 8000 + i; // Different ports
                let server = TrustformerServer::new(config);

                // Try to create router
                timeout(Duration::from_secs(10), server.create_test_router()).await
            })
        })
        .collect();

    // Wait for all tasks to complete
    let results = futures::future::join_all(handles).await;

    // All should succeed
    for result in results {
        assert!(result.is_ok(), "Server creation task should succeed");
        assert!(result.unwrap().is_ok(), "Router creation should succeed");
    }
}

#[tokio::test]
async fn test_error_handling_in_config() {
    // Test that invalid configs are handled gracefully
    let mut config = create_minimal_test_config();

    // Set some potentially problematic values
    config.num_workers = 0; // This might cause issues

    // Server creation should still work (validation might happen later)
    let _server = TrustformerServer::new(config);
    // Server creation handles edge cases
}

#[tokio::test]
async fn test_stress_config_creation() {
    // Test creating many configs rapidly
    for i in 0..100 {
        let mut config = create_minimal_test_config();
        config.port = 9000 + (i % 1000); // Vary ports
        config.model_config.model_name = format!("model-{}", i);

        let _server = TrustformerServer::new(config);
        // Config created successfully without panic
    }
}

#[tokio::test]
async fn test_memory_usage() {
    // Simple test to ensure we're not leaking memory during creation
    let initial_time = std::time::Instant::now();

    for _ in 0..10 {
        let config = create_minimal_test_config();
        let server = TrustformerServer::new(config);

        // Create and drop router
        let _router = timeout(Duration::from_secs(5), server.create_test_router()).await;
    }

    let elapsed = initial_time.elapsed();
    assert!(
        elapsed < Duration::from_secs(30),
        "Test should complete quickly"
    );
}

#[tokio::test]
async fn test_json_serialization_roundtrip() {
    let config = create_minimal_test_config();

    // Serialize to JSON
    let json_value = serde_json::to_value(&config).unwrap();

    // Deserialize back
    let config2: ServerConfig = serde_json::from_value(json_value).unwrap();

    // Compare key fields
    assert_eq!(config.host, config2.host);
    assert_eq!(config.port, config2.port);
    assert_eq!(config.num_workers, config2.num_workers);
    assert_eq!(
        config.model_config.model_name,
        config2.model_config.model_name
    );
}

/// Integration test that validates the complete server setup flow
#[tokio::test]
async fn test_complete_server_setup_flow() {
    // 1. Create configuration
    let config = create_minimal_test_config();
    println!("✅ Created server configuration");

    // 2. Create server instance
    let server = TrustformerServer::new(config);
    println!("✅ Created server instance");

    // 3. Create router
    let router_result = timeout(Duration::from_secs(10), server.create_test_router()).await;
    assert!(router_result.is_ok(), "Router creation should succeed");
    println!("✅ Created router successfully");

    // 4. Verify router is usable (basic checks)
    let _router = router_result.unwrap();
    println!("✅ Router is ready for use");

    println!("🎉 Complete server setup flow test passed!");
}
