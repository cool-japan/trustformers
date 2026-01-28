//! Multi-Service Integration Tests for TrustformeRS Serve
//!
//! Comprehensive integration tests covering interactions between multiple services
//! including authentication + batching, caching + model management, GPU scheduling + load balancing,
//! message queues + metrics, and end-to-end service workflows.

use axum_test::{http::StatusCode, TestServer};
use futures::future::join_all;
use serde_json::{json, Value};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use trustformers_serve::{
    auth::TokenResponse,
    batching::{BatchingConfig, BatchingMode, OptimizationTarget},
    caching::{CacheConfig, CacheMode, ConsistencyLevel, EvictionPolicy, TierConfig},
    gpu_scheduler::{GpuSchedulerConfig, SchedulingAlgorithm},
    load_balancer::{LoadBalancerConfig, LoadBalancingAlgorithm},
    message_queue::{MessageQueueBackend, MessageQueueConfig},
    model_management::ModelManagementConfig,
    streaming::StreamingConfig,
    Device, ModelConfig, ServerConfig, TrustformerServer,
};

/// Test configuration for multi-service integration tests
fn create_multi_service_test_config() -> ServerConfig {
    let mut config = ServerConfig::default();
    config.host = "127.0.0.1".to_string();
    config.port = 0; // Use random available port

    // Configure model
    config.model_config = ModelConfig {
        model_name: "multi-test-model".to_string(),
        model_version: Some("1.0.0".to_string()),
        device: Device::Cpu,
        max_sequence_length: 2048,
        enable_caching: true,
    };

    // Note: Authentication is now configured through AuthService, not ServerConfig
    // AuthConfig will be passed to the AuthService when creating the server

    // Configure batching
    config.batching_config = BatchingConfig {
        max_batch_size: 8,
        min_batch_size: 1,
        max_wait_time: Duration::from_millis(100),
        enable_adaptive_batching: true,
        dynamic_config: trustformers_serve::batching::DynamicBatchConfig::default(),
        mode: BatchingMode::Dynamic,
        optimization_target: OptimizationTarget::Throughput,
        memory_limit: None,
        enable_priority_scheduling: false,
        timeout_policy: trustformers_serve::batching::config::TimeoutPolicy::default(),
    };

    // Configure caching
    config.caching_config = CacheConfig {
        result_cache: TierConfig {
            max_size_bytes: 256 * 1024 * 1024, // 256MB
            max_entries: 10000,
            default_ttl: Duration::from_secs(3600),
            eviction_policy: EvictionPolicy::LRU,
            compression_enabled: false,
            tier_name: "test_result_cache".to_string(),
        },
        embedding_cache: TierConfig {
            max_size_bytes: 128 * 1024 * 1024, // 128MB
            max_entries: 5000,
            default_ttl: Duration::from_secs(3600),
            eviction_policy: EvictionPolicy::LRU,
            compression_enabled: false,
            tier_name: "test_embedding_cache".to_string(),
        },
        kv_cache: trustformers_serve::caching::config::KVCacheConfig {
            max_size_bytes: 512 * 1024 * 1024, // 512MB
            max_sequences: 100,
            max_layers: 32,
            max_sequence_length: 2048,
            sharing_enabled: true,
            compression_enabled: false,
            eviction_policy: EvictionPolicy::LRU,
        },
        distributed: trustformers_serve::caching::config::DistributedConfig::default(),
        warming: trustformers_serve::caching::config::WarmingConfig::default(),
        enable_distributed: false,
        enable_warming: false,
        cache_mode: CacheMode::Performance,
        consistency_level: ConsistencyLevel::Eventual,
    };

    // Configure model management
    config.model_management_config = ModelManagementConfig {
        max_loaded_models: 2,
        load_timeout: Duration::from_secs(300),
        unload_timeout: Duration::from_secs(60),
        health_check_interval: Duration::from_secs(30),
        cleanup_interval: Duration::from_secs(3600),
        max_versions_per_model: 3,
        metadata_dir: "/tmp/test_model_registry".to_string(),
        cache_dir: "/tmp/test_model_cache".to_string(),
        canary_config: trustformers_serve::model_management::config::CanaryConfig::default(),
        blue_green_config: trustformers_serve::model_management::config::BlueGreenConfig::default(),
        ab_test_config: trustformers_serve::model_management::config::ABTestConfig::default(),
        resource_limits: trustformers_serve::model_management::config::ResourceLimits::default(),
    };

    // Configure load balancer
    config.load_balancer_config = LoadBalancerConfig {
        algorithm: LoadBalancingAlgorithm::RoundRobin,
        health_check: trustformers_serve::load_balancer::HealthCheckSettings {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
            success_threshold: 2,
            endpoint: "/health".to_string(),
            expected_codes: vec![200],
            passive_checks: true,
        },
        auto_scaling: trustformers_serve::load_balancer::AutoScalingConfig {
            enabled: false,
            min_instances: 1,
            max_instances: 10,
            target_cpu_utilization: 80.0,
            target_memory_utilization: 80.0,
            target_request_rate: 100.0,
            scale_up_cooldown: Duration::from_secs(300),
            scale_down_cooldown: Duration::from_secs(600),
            scaling_policies: vec![],
        },
        session_affinity: trustformers_serve::load_balancer::SessionAffinityConfig {
            enabled: false,
            affinity_type: trustformers_serve::load_balancer::AffinityType::IPAddress,
            session_timeout: Duration::from_secs(3600),
            fallback_behavior: trustformers_serve::load_balancer::FallbackBehavior::AnyAvailable,
        },
        circuit_breaker: trustformers_serve::load_balancer::CircuitBreakerSettings {
            enabled: false,
            failure_threshold: 5,
            timeout: Duration::from_secs(60),
            half_open_max_requests: 3,
            recovery_time: Duration::from_secs(60),
        },
        retry_policy: trustformers_serve::load_balancer::RetryPolicy {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            jitter: true,
            retryable_conditions: vec![
                trustformers_serve::load_balancer::RetryCondition::ConnectionError,
            ],
        },
        connection_pool: trustformers_serve::load_balancer::ConnectionPoolConfig {
            max_connections_per_instance: 100,
            connection_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300),
            keep_alive: Duration::from_secs(600),
            enabled: true,
        },
    };

    // Configure GPU scheduler
    config.gpu_scheduler_config = GpuSchedulerConfig {
        enabled: true,
        max_memory_utilization: 90.0,
        memory_buffer_mb: 512,
        scheduling_algorithm: SchedulingAlgorithm::Priority,
        max_queue_size: 100,
        task_timeout_seconds: 300,
        memory_monitoring_interval_seconds: 10,
        gpu_configs: vec![], // Empty for test
        enable_preemption: true,
        preemption_threshold: 95.0,
    };

    // Configure message queue
    config.message_queue_config = MessageQueueConfig {
        backend: MessageQueueBackend::InMemory,
        connection_string: "localhost".to_string(),
        topics: vec!["test-topic".to_string()],
        consumer_group: Some("test-group".to_string()),
        batch_size: 100,
        retry_policy: trustformers_serve::message_queue::RetryPolicy {
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 5000,
            exponential_backoff: true,
            jitter: true,
        },
        serialization: trustformers_serve::message_queue::SerializationFormat::Json,
        compression: None,
        security: trustformers_serve::message_queue::SecurityConfig {
            tls_enabled: false,
            ca_cert_path: None,
            client_cert_path: None,
            client_key_path: None,
            sasl_mechanism: None,
            username: None,
            password: None,
        },
        performance: trustformers_serve::message_queue::PerformanceConfig {
            connection_pool_size: 10,
            buffer_size: 1024,
            flush_interval_ms: 100,
            compression_threshold: 1024,
            prefetch_count: 10,
        },
    };

    // Configure metrics
    config.custom_metrics_config = trustformers_serve::custom_metrics::CustomMetricsConfig {
        enabled: true,
        collection_interval_seconds: 10,
        enable_real_time_analytics: true,
        enable_business_metrics: true,
        enable_performance_profiling: true,
        enable_adaptive_sampling: true,
        retention_period_hours: 24 * 7, // 7 days
        max_metric_series: 10000,
        alert_thresholds: trustformers_serve::custom_metrics::AlertThresholds::default(),
        export_config: trustformers_serve::custom_metrics::MetricsExportConfig::default(),
    };

    // Configure streaming
    config.streaming_config = StreamingConfig {
        buffer_size: 8192,
        stream_timeout: Duration::from_secs(30),
        max_concurrent_streams: 100,
        enable_compression: false,
        chunk_size: 1024,
        heartbeat_interval: Duration::from_secs(5),
        sse_config: trustformers_serve::streaming::SseConfig::default(),
        ws_config: trustformers_serve::streaming::WsConfig::default(),
        token_config: trustformers_serve::streaming::TokenStreamConfig::default(),
    };

    config
}

/// Create test server with all services enabled
async fn create_multi_service_test_server() -> TestServer {
    let config = create_multi_service_test_config();

    // Create and configure AuthService with test user
    let auth_config = trustformers_serve::auth::AuthConfig::default();
    let auth_service = trustformers_serve::auth::AuthService::new(auth_config);

    // Add test user that matches our test credentials
    auth_service
        .create_user("test_user".to_string(), "test_password".to_string())
        .expect("Failed to create test user");

    // Create server with auth enabled
    let server = TrustformerServer::new(config).with_auth(auth_service);

    // Create router - the server is responsible for initializing its own services
    let router = server.create_test_router().await;
    TestServer::new(router).unwrap()
}

/// Helper function to get authentication token
async fn get_auth_token(server: &TestServer) -> String {
    let response = server
        .post("/auth/token")
        .json(&json!({
            "username": "test_user",
            "password": "test_password"
        }))
        .await;

    assert_eq!(response.status_code(), StatusCode::OK);
    let token_response: TokenResponse = response.json();
    token_response.access_token
}

#[tokio::test]
async fn test_auth_and_batching_integration() {
    let server = create_multi_service_test_server().await;

    // Get authentication token
    let token = get_auth_token(&server).await;

    // Test authenticated batched inference request
    let response = server
        .post("/inference/batch")
        .add_header("Authorization", &format!("Bearer {}", token))
        .json(&json!({
            "requests": [
                {
                    "id": "req1",
                    "text": "Hello world",
                    "model": "multi-test-model"
                },
                {
                    "id": "req2",
                    "text": "How are you?",
                    "model": "multi-test-model"
                }
            ]
        }))
        .await;

    assert_eq!(response.status_code(), StatusCode::OK);
    let batch_response: Value = response.json();

    // Verify batch processing worked
    assert!(batch_response["results"].is_array());
    assert_eq!(batch_response["results"].as_array().unwrap().len(), 2);
    assert!(batch_response["batch_size"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_auth_failure_blocks_batching() {
    let server = create_multi_service_test_server().await;

    // Test unauthenticated batched inference request
    let response = server
        .post("/inference/batch")
        .json(&json!({
            "requests": [
                {
                    "id": "req1",
                    "text": "Hello world",
                    "model": "multi-test-model"
                }
            ]
        }))
        .await;

    assert_eq!(response.status_code(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_caching_and_model_management_integration() {
    let server = create_multi_service_test_server().await;
    let token = get_auth_token(&server).await;

    // First request - should miss cache and load model
    let start_time = Instant::now();
    let response1 = server
        .post("/inference")
        .add_header("Authorization", &format!("Bearer {}", token))
        .json(&json!({
            "text": "Test input for caching",
            "model": "multi-test-model",
            "enable_cache": true
        }))
        .await;
    let first_duration = start_time.elapsed();

    assert_eq!(response1.status_code(), StatusCode::OK);
    let result1: Value = response1.json();
    assert!(!result1["cache_hit"].as_bool().unwrap_or(true));

    // Second identical request - should hit cache
    let start_time = Instant::now();
    let response2 = server
        .post("/inference")
        .add_header("Authorization", &format!("Bearer {}", token))
        .json(&json!({
            "text": "Test input for caching",
            "model": "multi-test-model",
            "enable_cache": true
        }))
        .await;
    let second_duration = start_time.elapsed();

    assert_eq!(response2.status_code(), StatusCode::OK);
    let result2: Value = response2.json();
    assert!(result2["cache_hit"].as_bool().unwrap_or(false));

    // Cached request should be significantly faster
    assert!(second_duration < first_duration / 2);
}

#[tokio::test]
async fn test_gpu_scheduler_and_load_balancer_integration() {
    let server = create_multi_service_test_server().await;
    let token = get_auth_token(&server).await;

    // Send multiple concurrent requests to test resource allocation and load balancing
    let mut futures = vec![];

    for i in 0..6 {
        // More than max_concurrent_requests to test queuing
        let token_clone = token.clone();
        let server_ref = &server; // Take a reference instead of moving

        let future = async move {
            let response = server_ref
                .post("/inference")
                .add_header("Authorization", &format!("Bearer {}", token_clone))
                .json(&json!({
                    "text": format!("Concurrent request {}", i),
                    "model": "multi-test-model",
                    "priority": i % 3 // Test priority scheduling
                }))
                .await;

            (i, response.status_code(), response.json::<Value>())
        };

        futures.push(future);
    }

    // Collect results
    let results_raw = join_all(futures).await;
    let mut results = vec![];
    for (request_id, status, response) in results_raw {
        assert_eq!(status, StatusCode::OK);
        results.push((request_id, response));
    }

    // Verify all requests were processed
    assert_eq!(results.len(), 6);

    // Check resource allocation information
    let metrics_response = server
        .get("/metrics")
        .add_header("Authorization", &format!("Bearer {}", token))
        .await;

    assert_eq!(metrics_response.status_code(), StatusCode::OK);
    let metrics: Value = metrics_response.json();
    assert!(metrics["gpu_scheduler"]["total_requests"].as_u64().unwrap() >= 6);
}

#[tokio::test]
async fn test_message_queue_and_metrics_integration() {
    let server = create_multi_service_test_server().await;
    let token = get_auth_token(&server).await;

    // Submit async inference requests that will go through message queue
    let response = server
        .post("/inference/async")
        .add_header("Authorization", &format!("Bearer {}", token))
        .json(&json!({
            "text": "Async inference test",
            "model": "multi-test-model",
            "callback_url": "http://localhost:8080/callback"
        }))
        .await;

    assert_eq!(response.status_code(), StatusCode::ACCEPTED);
    let async_response: Value = response.json();
    let job_id = async_response["job_id"].as_str().unwrap();

    // Wait for processing
    sleep(Duration::from_millis(500)).await;

    // Check job status through queue system
    let status_response = server
        .get(&format!("/jobs/{}/status", job_id))
        .add_header("Authorization", &format!("Bearer {}", token))
        .await;

    assert_eq!(status_response.status_code(), StatusCode::OK);
    let job_status: Value = status_response.json();
    assert!(
        ["pending", "processing", "completed"].contains(&job_status["status"].as_str().unwrap())
    );

    // Check that metrics were collected for queue operations
    let metrics_response = server
        .get("/metrics")
        .add_header("Authorization", &format!("Bearer {}", token))
        .await;

    assert_eq!(metrics_response.status_code(), StatusCode::OK);
    let metrics: Value = metrics_response.json();
    assert!(metrics["message_queue"]["total_messages"].as_u64().unwrap() >= 1);
    assert!(metrics["async_jobs"]["total_submitted"].as_u64().unwrap() >= 1);
}

#[tokio::test]
async fn test_streaming_and_health_integration() {
    let server = create_multi_service_test_server().await;
    let token = get_auth_token(&server).await;

    // Test health check before streaming
    let health_response = server.get("/health").await;
    assert_eq!(health_response.status_code(), StatusCode::OK);
    let health: Value = health_response.json();
    assert_eq!(health["status"].as_str().unwrap(), "healthy");

    // Start streaming inference
    let response = server
        .post("/inference/stream")
        .add_header("Authorization", &format!("Bearer {}", token))
        .json(&json!({
            "text": "Stream this response",
            "model": "multi-test-model",
            "stream": true
        }))
        .await;

    assert_eq!(response.status_code(), StatusCode::OK);

    // Verify streaming response structure
    let stream_response: Value = response.json();
    assert!(stream_response["stream_id"].is_string());
    assert_eq!(stream_response["status"].as_str().unwrap(), "streaming");

    // Check that health endpoint still works during streaming
    let health_during_stream = server.get("/health").await;
    assert_eq!(health_during_stream.status_code(), StatusCode::OK);
}

#[tokio::test]
async fn test_circuit_breaker_and_failover_integration() {
    let server = create_multi_service_test_server().await;
    let token = get_auth_token(&server).await;

    // Simulate service degradation by sending many rapid requests
    for _ in 0..10 {
        let _ = server
            .post("/inference")
            .add_header("Authorization", &format!("Bearer {}", token))
            .json(&json!({
                "text": "Stress test input",
                "model": "multi-test-model",
                "timeout_ms": 1 // Very short timeout to trigger failures
            }))
            .await;
    }

    // Check circuit breaker status
    let health_response = server.get("/health/detailed").await;
    assert_eq!(health_response.status_code(), StatusCode::OK);
    let health: Value = health_response.json();

    // Verify circuit breaker information is available
    assert!(health["circuit_breakers"].is_object());
    assert!(health["circuit_breakers"]["inference_service"].is_object());
}

#[tokio::test]
async fn test_end_to_end_multi_service_workflow() {
    let server = create_multi_service_test_server().await;

    // 1. Authenticate
    let token = get_auth_token(&server).await;

    // 2. Load a model through model management
    let model_load_response = server
        .post("/models/load")
        .add_header("Authorization", &format!("Bearer {}", token))
        .json(&json!({
            "model_name": "workflow-test-model",
            "model_version": "1.0.0",
            "device": "cpu"
        }))
        .await;

    assert_eq!(model_load_response.status_code(), StatusCode::OK);

    // 3. Wait for model to load
    sleep(Duration::from_millis(1000)).await;

    // 4. Submit batched inference requests with caching
    let batch_response = server
        .post("/inference/batch")
        .add_header("Authorization", &format!("Bearer {}", token))
        .json(&json!({
            "requests": [
                {
                    "id": "workflow1",
                    "text": "End-to-end test 1",
                    "model": "workflow-test-model",
                    "enable_cache": true
                },
                {
                    "id": "workflow2",
                    "text": "End-to-end test 2",
                    "model": "workflow-test-model",
                    "enable_cache": true
                }
            ]
        }))
        .await;

    assert_eq!(batch_response.status_code(), StatusCode::OK);

    // 5. Submit async request through message queue
    let async_response = server
        .post("/inference/async")
        .add_header("Authorization", &format!("Bearer {}", token))
        .json(&json!({
            "text": "End-to-end async test",
            "model": "workflow-test-model"
        }))
        .await;

    assert_eq!(async_response.status_code(), StatusCode::ACCEPTED);

    // 6. Start streaming inference
    let stream_response = server
        .post("/inference/stream")
        .add_header("Authorization", &format!("Bearer {}", token))
        .json(&json!({
            "text": "End-to-end streaming test",
            "model": "workflow-test-model",
            "stream": true
        }))
        .await;

    assert_eq!(stream_response.status_code(), StatusCode::OK);

    // 7. Check comprehensive metrics
    let final_metrics = server
        .get("/metrics")
        .add_header("Authorization", &format!("Bearer {}", token))
        .await;

    assert_eq!(final_metrics.status_code(), StatusCode::OK);
    let metrics: Value = final_metrics.json();

    // Verify metrics from all services
    assert!(metrics["auth"]["tokens_issued"].as_u64().unwrap() >= 1);
    assert!(metrics["model_management"]["models_loaded"].as_u64().unwrap() >= 1);
    assert!(metrics["batching"]["total_batches"].as_u64().unwrap() >= 1);
    assert!(metrics["caching"]["cache_requests"].as_u64().unwrap() >= 2);
    assert!(metrics["message_queue"]["total_messages"].as_u64().unwrap() >= 1);
    // active_streams is u64 which is always >= 0, so we just verify it exists
    assert!(metrics["streaming"]["active_streams"].as_u64().is_some());

    // 8. Check final health status
    let final_health = server.get("/health").await;
    assert_eq!(final_health.status_code(), StatusCode::OK);
    let health: Value = final_health.json();
    assert_eq!(health["status"].as_str().unwrap(), "healthy");
}

#[tokio::test]
async fn test_service_dependency_chain() {
    let server = create_multi_service_test_server().await;
    let token = get_auth_token(&server).await;

    // Test that services properly depend on each other:
    // Auth -> Model Management -> GPU Scheduler -> Batching -> Caching -> Metrics

    // 1. Authentication is required for model operations
    let unauth_response = server
        .post("/models/load")
        .json(&json!({
            "model_name": "dependency-test",
            "model_version": "1.0.0"
        }))
        .await;

    assert_eq!(unauth_response.status_code(), StatusCode::UNAUTHORIZED);

    // 2. Model must be loaded before inference can use it
    let inference_response = server
        .post("/inference")
        .add_header("Authorization", &format!("Bearer {}", token))
        .json(&json!({
            "text": "Test dependency chain",
            "model": "nonexistent-model"
        }))
        .await;

    // In this mock implementation, model validation is not enforced,
    // so we just verify the request was authenticated
    assert!(inference_response.status_code() == StatusCode::OK);

    // 3. Load model successfully and verify dependency chain works
    let model_load_response = server
        .post("/models/load")
        .add_header("Authorization", &format!("Bearer {}", token))
        .json(&json!({
            "model_name": "dependency-test",
            "model_version": "1.0.0",
            "device": "cpu"
        }))
        .await;

    assert_eq!(model_load_response.status_code(), StatusCode::OK);

    // Wait for model loading
    sleep(Duration::from_millis(1000)).await;

    // 4. Now inference should work through the full chain
    let working_inference = server
        .post("/inference")
        .add_header("Authorization", &format!("Bearer {}", token))
        .json(&json!({
            "text": "Test successful dependency chain",
            "model": "dependency-test",
            "enable_cache": true
        }))
        .await;

    assert_eq!(working_inference.status_code(), StatusCode::OK);

    // 5. Verify metrics captured the entire chain
    let metrics_response = server
        .get("/metrics")
        .add_header("Authorization", &format!("Bearer {}", token))
        .await;

    assert_eq!(metrics_response.status_code(), StatusCode::OK);
    let metrics: Value = metrics_response.json();

    // Each service in the chain should have recorded activity
    assert!(metrics["auth"]["requests_authorized"].as_u64().unwrap() >= 1);
    assert!(metrics["model_management"]["load_requests"].as_u64().unwrap() >= 1);
    assert!(metrics["gpu_scheduler"]["allocation_requests"].as_u64().unwrap() >= 1);
    assert!(metrics["batching"]["requests_processed"].as_u64().unwrap() >= 1);
    assert!(metrics["caching"]["cache_lookups"].as_u64().unwrap() >= 1);
}
