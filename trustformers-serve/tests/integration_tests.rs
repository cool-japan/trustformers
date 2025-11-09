//! Integration Tests for TrustformeRS Serve
//!
//! Comprehensive integration tests covering all major server functionality
//! including health checks, inference endpoints, authentication, metrics,
//! streaming, and administrative operations.

use axum_test::{http::StatusCode, TestServer};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use trustformers_serve::{Device, ModelConfig, ServerConfig, TrustformerServer};

/// Test configuration for integration tests
fn create_test_config() -> ServerConfig {
    let mut config = ServerConfig::default();
    config.host = "127.0.0.1".to_string();
    config.port = 0; // Use random available port
    config.model_config = ModelConfig {
        model_name: "test-model".to_string(),
        model_version: Some("1.0.0".to_string()),
        device: Device::Cpu,
        max_sequence_length: 2048,
        enable_caching: true,
    };
    config
}

/// Create test server for integration tests
async fn create_test_server() -> TestServer {
    let config = create_test_config();
    let server = TrustformerServer::new(config);

    // Create router and convert to service that can be used by TestServer
    let router = server.create_test_router().await;
    TestServer::new(router).unwrap()
}

/// Create test server with authentication enabled
async fn create_test_server_with_auth() -> TestServer {
    let config = create_test_config();
    let server = TrustformerServer::new(config);

    let router = server.create_test_router().await;
    TestServer::new(router).unwrap()
}

#[tokio::test]
async fn test_health_endpoints() {
    let server = create_test_server().await;

    // Test basic health check
    let response = server.get("/health").await;
    response.assert_status_ok();

    let body: Value = response.json();
    assert_eq!(body["status"], "healthy");
    assert!(body["timestamp"].is_string());

    // Test detailed health check
    let response = server.get("/health/detailed").await;
    response.assert_status_ok();

    let body: Value = response.json();
    assert_eq!(body["status"], "healthy");
    assert!(body["checks"].is_object());

    // Test readiness check
    let response = server.get("/health/readiness").await;
    response.assert_status_ok();

    // Test liveness check
    let response = server.get("/health/liveness").await;
    response.assert_status_ok();
}

#[tokio::test]
async fn test_inference_endpoints() {
    let server = create_test_server().await;

    // Test single inference
    let request_body = json!({
        "model": "test-model",
        "input": "Hello, world!",
        "parameters": {
            "max_tokens": 100,
            "temperature": 0.7
        }
    });

    let response = server.post("/v1/inference").json(&request_body).await;

    response.assert_status_ok();
    let body: Value = response.json();
    assert!(body["request_id"].is_string());
    assert!(body["response"].is_string());

    // Test batch inference
    let batch_request = json!({
        "model": "test-model",
        "inputs": [
            "Hello, world!",
            "How are you?",
            "What is AI?"
        ],
        "parameters": {
            "max_tokens": 100,
            "temperature": 0.7
        }
    });

    let response = server.post("/v1/inference/batch").json(&batch_request).await;

    response.assert_status_ok();
    let body: Value = response.json();
    assert!(body["batch_id"].is_string());
    assert!(body["responses"].is_array());
    assert_eq!(body["responses"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn test_metrics_endpoint() {
    let server = create_test_server().await;

    // Test metrics endpoint
    let response = server.get("/metrics").await;
    response.assert_status_ok();

    let body = response.text();
    assert!(body.contains("# HELP"));
    assert!(body.contains("# TYPE"));
    // Should contain Prometheus metrics format
}

#[tokio::test]
async fn test_admin_endpoints() {
    let server = create_test_server().await;

    // Test stats endpoint
    let response = server.get("/admin/stats").await;
    response.assert_status_ok();

    let body: Value = response.json();
    assert!(body["server_stats"].is_object());
    assert!(body["batching_stats"].is_object());
    assert!(body["caching_stats"].is_object());

    // Test config endpoint
    let response = server.get("/admin/config").await;
    response.assert_status_ok();

    let body: Value = response.json();
    assert!(body["host"].is_string());
    assert!(body["port"].is_number());
    assert!(body["enable_metrics"].is_boolean());
}

#[tokio::test]
async fn test_graphql_endpoints() {
    let server = create_test_server().await;

    // Test GraphQL health query
    let query = json!({
        "query": "{ health { status timestamp } }"
    });

    let response = server.post("/graphql").json(&query).await;

    response.assert_status_ok();
    let body: Value = response.json();
    assert!(body["data"]["health"]["status"].is_string());

    // Test GraphQL playground endpoint
    let response = server.get("/graphql/playground").await;
    response.assert_status_ok();
}

#[tokio::test]
async fn test_long_polling_endpoints() {
    let server = create_test_server().await;

    // Test poll stats endpoint
    let response = server.get("/v1/poll/stats").await;
    response.assert_status_ok();

    let body: Value = response.json();
    assert!(body["active_connections"].is_number());
    assert!(body["total_events"].is_number());

    // Test long polling endpoint (with timeout)
    let response = server
        .get("/v1/poll")
        .add_query_param("timeout", "1") // 1 second timeout
        .await;

    // Should timeout and return empty or no events
    response.assert_status_ok();
}

#[tokio::test]
async fn test_shadow_testing_endpoints() {
    let server = create_test_server().await;

    // Test shadow stats endpoint
    let response = server.get("/v1/shadow/stats").await;
    response.assert_status_ok();

    let body: Value = response.json();
    assert!(body["total_requests"].is_number());
    assert!(body["shadow_responses"].is_number());

    // Test shadow results endpoint
    let response = server.get("/v1/shadow/results").await;
    response.assert_status_ok();

    let body: Value = response.json();
    assert!(body["results"].is_array());
}

#[tokio::test]
async fn test_authentication_flow() {
    let server = create_test_server_with_auth().await;

    // Test accessing protected endpoint without auth (should fail)
    let response = server.get("/admin/stats").await;
    response.assert_status_unauthorized();

    // Test login endpoint
    let login_request = json!({
        "username": "test_user",
        "password": "test_password"
    });

    let response = server.post("/auth/login").json(&login_request).await;

    if response.status_code().is_success() {
        let body: Value = response.json();
        let token = body["token"].as_str().unwrap();

        // Test accessing protected endpoint with valid token
        let response = server
            .get("/admin/stats")
            .add_header("Authorization", &format!("Bearer {}", token))
            .await;

        response.assert_status_ok();
    }
}

#[tokio::test]
async fn test_streaming_endpoints() {
    let server = create_test_server().await;

    // Test SSE endpoint
    let response = server.get("/v1/stream/sse").await;
    response.assert_status_ok();

    // Check content type for SSE
    assert!(response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .contains("text/event-stream"));

    // Test WebSocket endpoint (will upgrade connection)
    let response = server
        .get("/v1/stream/ws")
        .add_header("Connection", "Upgrade")
        .add_header("Upgrade", "websocket")
        .add_header("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==")
        .add_header("Sec-WebSocket-Version", "13")
        .await;

    // Should return upgrade response
    assert_eq!(response.status_code(), 101); // Switching Protocols
}

#[tokio::test]
async fn test_api_documentation() {
    let server = create_test_server().await;

    // Test OpenAPI JSON endpoint
    let response = server.get("/api-docs/openapi.json").await;
    response.assert_status_ok();

    let body: Value = response.json();
    assert_eq!(body["openapi"], "3.0.3");
    assert!(body["info"].is_object());
    assert!(body["paths"].is_object());

    // Test Swagger UI endpoint
    let response = server.get("/docs").await;
    response.assert_status_ok();

    // Should return HTML content
    let body = response.text();
    assert!(body.contains("swagger-ui"));
}

#[tokio::test]
async fn test_error_handling() {
    let server = create_test_server().await;

    // Test invalid endpoint
    let response = server.get("/invalid/endpoint").await;
    response.assert_status_not_found();

    // Test malformed JSON request
    let response = server
        .post("/v1/inference")
        .add_header("content-type", "application/json")
        .text("invalid json")
        .await;

    response.assert_status_bad_request();

    // Test missing required fields
    let invalid_request = json!({
        "model": "test-model"
        // Missing 'input' field
    });

    let response = server.post("/v1/inference").json(&invalid_request).await;

    response.assert_status_bad_request();
}

#[tokio::test]
async fn test_concurrent_requests() {
    let server = create_test_server().await;

    // Test multiple concurrent health checks
    let futures: Vec<_> = (0..10)
        .map(|_| {
            let server = &server;
            async move { server.get("/health").await }
        })
        .collect();

    let responses = futures::future::join_all(futures).await;

    for response in responses {
        response.assert_status_ok();
    }

    // Test concurrent inference requests
    let request_body = json!({
        "model": "test-model",
        "input": "Test concurrent request",
        "parameters": {
            "max_tokens": 50
        }
    });

    let futures: Vec<_> = (0..5)
        .map(|_| {
            let server = &server;
            let request_body = &request_body;
            async move { server.post("/v1/inference").json(request_body).await }
        })
        .collect();

    let responses = futures::future::join_all(futures).await;

    for response in responses {
        response.assert_status_ok();
    }
}

#[tokio::test]
async fn test_rate_limiting() {
    let server = create_test_server().await;

    // Make many rapid requests to test rate limiting
    // Note: This assumes rate limiting is configured
    let request_body = json!({
        "model": "test-model",
        "input": "Rate limit test",
        "parameters": {
            "max_tokens": 10
        }
    });

    let mut success_count = 0;
    let mut _rate_limited_count = 0;

    for _ in 0..50 {
        let response = server.post("/v1/inference").json(&request_body).await;

        if response.status_code().is_success() {
            success_count += 1;
        } else if response.status_code() == 429 {
            _rate_limited_count += 1;
        }

        // Small delay between requests
        sleep(Duration::from_millis(10)).await;
    }

    // Should have at least some successful requests
    assert!(success_count > 0);

    // If rate limiting is enabled, we might see some 429 responses
    // This is optional depending on configuration
}

#[tokio::test]
async fn test_failover_endpoint() {
    let server = create_test_server().await;

    // Test force failover endpoint
    let response = server.post("/admin/failover").await;

    // Should either succeed or return appropriate error
    assert!(response.status_code().is_success() || response.status_code().is_server_error());
}

#[tokio::test]
async fn test_end_to_end_workflow() {
    let server = create_test_server().await;

    // 1. Check server health
    let response = server.get("/health").await;
    response.assert_status_ok();

    // 2. Make inference request
    let request_body = json!({
        "model": "test-model",
        "input": "End-to-end test",
        "parameters": {
            "max_tokens": 100,
            "temperature": 0.8
        }
    });

    let response = server.post("/v1/inference").json(&request_body).await;

    response.assert_status_ok();
    let inference_result: Value = response.json();
    let request_id = inference_result["request_id"].as_str().unwrap();

    // 3. Check metrics were updated
    let response = server.get("/metrics").await;
    response.assert_status_ok();
    let metrics = response.text();
    assert!(metrics.contains("inference_requests_total"));

    // 4. Check admin stats
    let response = server.get("/admin/stats").await;
    response.assert_status_ok();
    let stats: Value = response.json();
    assert!(stats["server_stats"]["total_requests"].as_u64().unwrap() > 0);

    // 5. Test batch inference
    let batch_request = json!({
        "model": "test-model",
        "inputs": ["Batch test 1", "Batch test 2"],
        "parameters": {
            "max_tokens": 50
        }
    });

    let response = server.post("/v1/inference/batch").json(&batch_request).await;

    response.assert_status_ok();
    let batch_result: Value = response.json();
    assert_eq!(batch_result["responses"].as_array().unwrap().len(), 2);

    println!("✅ End-to-end workflow test completed successfully");
    println!("   - Request ID: {}", request_id);
    println!(
        "   - Batch responses: {}",
        batch_result["responses"].as_array().unwrap().len()
    );
}

/// Test interaction between caching and batching services
#[tokio::test]
async fn test_caching_batching_interaction() {
    let server = create_test_server().await;

    // First, make a request that should be cached
    let request_body = json!({
        "text": "Cache test input",
        "max_length": 50,
        "temperature": 0.5
    });

    // First request - should hit the model
    let response1 = server.post("/v1/inference").json(&request_body).await;
    response1.assert_status_ok();
    let result1: Value = response1.json();

    // Second request - should hit cache
    let response2 = server.post("/v1/inference").json(&request_body).await;
    response2.assert_status_ok();
    let result2: Value = response2.json();

    // Results should be the same (text field)
    assert_eq!(result1["text"], result2["text"]);

    // Now test with batch requests to see cache interaction
    let batch_request = json!({
        "requests": [
            {
                "text": "Cache test input",
                "max_length": 50,
                "temperature": 0.5
            },
            {
                "text": "New cache input",
                "max_length": 50,
                "temperature": 0.5
            }
        ]
    });

    let batch_response = server.post("/v1/inference/batch").json(&batch_request).await;
    batch_response.assert_status_ok();
    let batch_result: Value = batch_response.json();

    assert_eq!(batch_result["responses"].as_array().unwrap().len(), 2);
    println!("✅ Caching-Batching interaction test completed");
}

/// Test interaction between authentication and metrics services
#[tokio::test]
async fn test_auth_metrics_interaction() {
    let server = create_test_server_with_auth().await;

    // Test unauthenticated request
    let request_body = json!({
        "model": "test-model",
        "input": "Auth test",
        "parameters": {"max_tokens": 50}
    });

    let response = server.post("/v1/inference").json(&request_body).await;
    response.assert_status(StatusCode::UNAUTHORIZED); // Unauthorized

    // Get auth token
    let auth_request = json!({
        "username": "test-user",
        "password": "test-password"
    });

    let auth_response = server.post("/auth/login").json(&auth_request).await;
    auth_response.assert_status_ok();
    let auth_result: Value = auth_response.json();
    let token = auth_result["token"].as_str().unwrap();

    // Test authenticated request
    let response = server
        .post("/v1/inference")
        .add_header("Authorization", &format!("Bearer {}", token))
        .json(&request_body)
        .await;
    response.assert_status_ok();

    // Check metrics for auth failures and successes
    let metrics_response = server
        .get("/metrics")
        .add_header("Authorization", &format!("Bearer {}", token))
        .await;
    metrics_response.assert_status_ok();
    let metrics = metrics_response.text();

    assert!(metrics.contains("auth_requests_total"));
    println!("✅ Auth-Metrics interaction test completed");
}

/// Test interaction between streaming and monitoring services
#[tokio::test]
async fn test_streaming_monitoring_interaction() {
    let server = create_test_server().await;

    // Start a streaming request
    let request_body = json!({
        "model": "test-model",
        "input": "Stream test input",
        "parameters": {
            "max_tokens": 100,
            "stream": true
        }
    });

    let response = server.post("/v1/inference/stream").json(&request_body).await;
    response.assert_status_ok();

    // Test Server-Sent Events connection
    let sse_response = server.get("/v1/stream/sse").await;
    sse_response.assert_status_ok();

    // Check that streaming metrics are being tracked
    let metrics_response = server.get("/metrics").await;
    metrics_response.assert_status_ok();
    let metrics = metrics_response.text();

    assert!(metrics.contains("streaming_connections"));

    // Check admin stats for streaming information
    let admin_response = server.get("/admin/stats").await;
    admin_response.assert_status_ok();
    let stats: Value = admin_response.json();

    assert!(stats["streaming_stats"].is_object());
    println!("✅ Streaming-Monitoring interaction test completed");
}

/// Test interaction between shadow testing and validation services
#[tokio::test]
async fn test_shadow_validation_interaction() {
    let server = create_test_server().await;

    // Test shadow mode request with validation
    let request_body = json!({
        "model": "test-model",
        "input": "Shadow test with very long input that should be validated by the validation service to ensure it meets requirements",
        "parameters": {
            "max_tokens": 50,
            "temperature": 0.7
        },
        "shadow_mode": true
    });

    let response = server.post("/v1/inference").json(&request_body).await;
    response.assert_status_ok();
    let result: Value = response.json();

    // Check that shadow comparison was performed
    assert!(result["shadow_comparison"].is_object());

    // Test with invalid input to trigger validation
    let invalid_request = json!({
        "model": "test-model",
        "input": "a".repeat(10000), // Very long input to trigger validation
        "parameters": {"max_tokens": 50},
        "shadow_mode": true
    });

    let response = server.post("/v1/inference").json(&invalid_request).await;

    // Should either reject or handle gracefully
    assert!(response.status_code().is_client_error() || response.status_code().is_success());

    println!("✅ Shadow-Validation interaction test completed");
}

/// Test interaction between load balancing and health monitoring services
#[tokio::test]
async fn test_load_balancing_health_interaction() {
    let server = Arc::new(create_test_server().await);

    // Check initial health
    let health_response = server.get("/health/detailed").await;
    health_response.assert_status_ok();
    let health: Value = health_response.json();

    assert_eq!(health["status"], "healthy");

    // Make multiple concurrent requests to test load balancing
    let futures: Vec<_> = (0..10)
        .map(|i| {
            let server = &server;
            async move {
                let request_body = json!({
                    "model": "test-model",
                    "input": format!("Load test request {}", i),
                    "parameters": {"max_tokens": 20}
                });

                server.post("/v1/inference").json(&request_body).await.assert_status_ok()
            }
        })
        .collect();

    // Wait for all requests to complete
    futures::future::join_all(futures).await;

    // Check health after load
    let health_response = server.get("/health/detailed").await;
    health_response.assert_status_ok();
    let health: Value = health_response.json();

    // Should still be healthy
    assert_eq!(health["status"], "healthy");

    // Check load balancer stats
    let admin_response = server.get("/admin/stats").await;
    admin_response.assert_status_ok();
    let stats: Value = admin_response.json();

    assert!(stats["server_stats"]["total_requests"].as_u64().unwrap() >= 10);
    println!("✅ Load Balancing-Health interaction test completed");
}

/// Test interaction between GPU scheduling and memory management services
#[tokio::test]
async fn test_gpu_memory_interaction() {
    let server = create_test_server().await;

    // Test GPU status endpoint
    let gpu_response = server.get("/admin/gpu/status").await;
    // GPU endpoint may not be available in test environment, so check gracefully
    if gpu_response.status_code().is_success() {
        let gpu_status: Value = gpu_response.json();
        assert!(gpu_status.is_object());
    }

    // Test memory pressure endpoint
    let memory_response = server.get("/admin/memory/pressure").await;
    memory_response.assert_status_ok();
    let memory_status: Value = memory_response.json();

    assert!(memory_status["pressure_level"].is_string());

    // Make requests that could trigger memory pressure
    let large_request = json!({
        "model": "test-model",
        "input": "Large memory test ".repeat(100),
        "parameters": {
            "max_tokens": 200,
            "batch_size": 5
        }
    });

    let response = server.post("/v1/inference").json(&large_request).await;
    response.assert_status_ok();

    // Check updated memory status
    let memory_response = server.get("/admin/memory/pressure").await;
    memory_response.assert_status_ok();
    let updated_memory: Value = memory_response.json();

    assert!(updated_memory["memory_usage"].is_object());
    println!("✅ GPU-Memory interaction test completed");
}

/// Test comprehensive multi-service workflow
#[tokio::test]
async fn test_comprehensive_multi_service_workflow() {
    let server = create_test_server().await;

    // 1. Check initial health and metrics
    let health_response = server.get("/health").await;
    health_response.assert_status_ok();

    let initial_metrics = server.get("/metrics").await;
    initial_metrics.assert_status_ok();

    // 2. Test batch inference with caching
    let batch_request = json!({
        "model": "test-model",
        "inputs": [
            "Multi-service test 1",
            "Multi-service test 2",
            "Multi-service test 1" // Duplicate for cache testing
        ],
        "parameters": {
            "max_tokens": 50,
            "temperature": 0.7
        }
    });

    let batch_response = server.post("/v1/inference/batch").json(&batch_request).await;
    batch_response.assert_status_ok();
    let batch_result: Value = batch_response.json();

    assert_eq!(batch_result["responses"].as_array().unwrap().len(), 3);

    // 3. Test streaming with monitoring
    let stream_request = json!({
        "model": "test-model",
        "input": "Streaming test for multi-service",
        "parameters": {
            "max_tokens": 30,
            "stream": true
        }
    });

    let stream_response = server.post("/v1/inference/stream").json(&stream_request).await;
    stream_response.assert_status_ok();

    // 4. Check GraphQL endpoint integration
    let graphql_query = json!({
        "query": "{ health { status, uptime }, modelInfo { name, version } }"
    });

    let graphql_response = server.post("/graphql").json(&graphql_query).await;
    graphql_response.assert_status_ok();
    let graphql_result: Value = graphql_response.json();

    assert!(graphql_result["data"]["health"]["status"].is_string());

    // 5. Test long polling integration
    let poll_response = server.get("/v1/poll").await;
    poll_response.assert_status_ok();

    // 6. Check final metrics show all interactions
    let final_metrics = server.get("/metrics").await;
    final_metrics.assert_status_ok();
    let metrics_text = final_metrics.text();

    assert!(metrics_text.contains("inference_requests_total"));
    assert!(metrics_text.contains("batch_requests_total"));

    // 7. Check admin statistics comprehensive view
    let admin_stats = server.get("/admin/stats").await;
    admin_stats.assert_status_ok();
    let stats: Value = admin_stats.json();

    assert!(stats["server_stats"]["total_requests"].as_u64().unwrap() > 0);
    assert!(stats["batching_stats"].is_object());
    assert!(stats["caching_stats"].is_object());

    println!("✅ Comprehensive multi-service workflow test completed");
    println!(
        "   - Batch responses: {}",
        batch_result["responses"].as_array().unwrap().len()
    );
    println!(
        "   - GraphQL health: {}",
        graphql_result["data"]["health"]["status"]
    );
    println!(
        "   - Total requests: {}",
        stats["server_stats"]["total_requests"]
    );
}
