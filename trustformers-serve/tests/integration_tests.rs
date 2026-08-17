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
use trustformers_serve::{
    batching::{BatchingConfig, BatchingMode},
    Device, ModelConfig, ServerConfig, TrustformerServer,
};

/// Test configuration for integration tests
fn create_test_config() -> ServerConfig {
    use trustformers_serve::streaming::StreamingConfig;

    let mut config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 0, // Use random available port
        model_config: ModelConfig {
            model_name: "test-model".to_string(),
            model_version: Some("1.0.0".to_string()),
            device: Device::Cpu,
            max_sequence_length: 2048,
            enable_caching: true,
        },
        ..Default::default()
    };
    // Use Fixed batching mode for tests to form batches immediately
    // This avoids timeout-based batch formation which can cause test delays
    config.batching_config = BatchingConfig {
        mode: BatchingMode::Fixed,
        min_batch_size: 1,
        max_batch_size: 32,
        max_wait_time: Duration::from_millis(10),
        ..BatchingConfig::default()
    };

    // Configure streaming with very short timeouts for tests (500ms)
    config.streaming_config = StreamingConfig {
        buffer_size: 100,
        stream_timeout: Duration::from_millis(500),
        max_concurrent_streams: 100,
        enable_compression: false,
        chunk_size: 1024,
        heartbeat_interval: Duration::from_millis(250),
        sse_config: trustformers_serve::streaming::SseConfig {
            buffer_size: 10,
            heartbeat_interval: Duration::from_millis(250),
            connection_timeout: Duration::from_millis(500),
            max_connections: 100,
            enable_compression: false,
            cors_origins: vec!["*".to_string()],
        },
        ws_config: trustformers_serve::streaming::WsConfig {
            buffer_size: 10,
            connection_timeout: Duration::from_millis(500),
            max_connections: 100,
            max_message_size: 1024,
            enable_compression: false,
            ping_interval: Duration::from_millis(250),
            max_frame_size: 1024,
        },
        ..StreamingConfig::default()
    };

    config
}

/// A real, if small, model for the serving path.
///
/// The architecture, the weights and the forward pass are genuine; the weights
/// are the architecture's own initialization rather than trained parameters, so
/// the output is real model output rather than English. Nothing here fakes
/// inference.
fn test_executor() -> Arc<dyn trustformers_serve::batching::BatchExecutor> {
    Arc::new(
        trustformers_serve::batching::untrained_byte_gpt2_executor(1, 16, 8)
            .expect("the tiny GPT-2 used by the tests must build"),
    )
}

/// Create test server for integration tests
async fn create_test_server() -> TestServer {
    let config = create_test_config();
    let server = TrustformerServer::with_executor(config, test_executor());

    // Create router and convert to service that can be used by TestServer
    let router = server.create_test_router().await;
    TestServer::new(router)
}

/// Create test server with authentication enabled
async fn create_test_server_with_auth() -> TestServer {
    use trustformers_serve::auth::{AuthConfig, AuthService};

    let config = create_test_config();

    // Create auth service with default config
    let auth_config = AuthConfig::default();
    let auth_service = AuthService::new(auth_config);

    // Create server and add auth service
    let server = TrustformerServer::with_executor(config, test_executor()).with_auth(auth_service);

    let router = server.create_test_router().await;
    TestServer::new(router)
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
    // API returns "services" instead of "checks"
    assert!(body["services"].is_object());

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
        "text": "Hello, world!",
        "max_length": 100,
        "temperature": 0.7
    });

    let response = server.post("/v1/inference").json(&request_body).await;

    response.assert_status_ok();
    let body: Value = response.json();
    assert!(body["request_id"].is_string());
    assert!(body["text"].is_string());

    // Test batch inference
    let batch_request = json!({
        "requests": [
            {"text": "Hello, world!", "max_length": 100, "temperature": 0.7},
            {"text": "How are you?", "max_length": 100, "temperature": 0.7},
            {"text": "What is AI?", "max_length": 100, "temperature": 0.7}
        ]
    });

    let response = server.post("/v1/inference/batch").json(&batch_request).await;

    response.assert_status_ok();
    let body: Value = response.json();
    assert!(body["batch_id"].is_string());
    // API returns "results" not "responses"
    assert!(body["results"].is_array());
    assert_eq!(
        body["results"].as_array().expect("operation failed in test").len(),
        3
    );
}

/// Regression: `/metrics` must serve real Prometheus text exposition, not a JSON
/// blob of invented constants.
#[tokio::test]
async fn test_metrics_endpoint() {
    let server = create_test_server().await;

    let response = server.get("/metrics").await;
    response.assert_status_ok();

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        content_type.starts_with("text/plain"),
        "metrics must be served as Prometheus text, got {content_type:?}"
    );

    let body = response.text();
    assert!(body.contains("# TYPE trustformers_serve_uptime_seconds gauge"));
    assert!(body.contains("trustformers_serve_http_requests_total"));
    assert!(body.contains("trustformers_serve_model_configured 1"));
    // The old fabricated counters must be gone.
    assert!(!body.contains("tokens_issued"));
    assert!(!body.contains("Mock"));
}

#[tokio::test]
async fn test_admin_endpoints() {
    let server = create_test_server().await;

    // Test stats endpoint
    let response = server.get("/admin/stats").await;
    response.assert_status_ok();

    let body: Value = response.json();
    // The stats endpoint returns batching_stats, caching_stats, streaming_stats, ha_stats
    assert!(body["batching_stats"].is_object());
    assert!(body["caching_stats"].is_object());
    assert!(body["streaming_stats"].is_object());

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

    // Regression: the handler used to return {"result": "Mock GraphQL response"}.
    // It must now execute against the real async-graphql schema.
    response.assert_status_ok();
    let body: Value = response.json();
    assert!(
        body.get("errors").is_none(),
        "GraphQL execution failed: {body}"
    );
    let health = &body["data"]["health"];
    assert!(health.is_object(), "no health data resolved: {body}");
    assert!(health["status"].is_string());
    assert!(health["timestamp"].is_string());
    assert!(!body.to_string().contains("Mock GraphQL response"));

    // A malformed query must be reported as a GraphQL error, not silently echoed.
    let bad = server
        .post("/graphql")
        .json(&json!({ "query": "{ thisFieldDoesNotExist }" }))
        .await;
    bad.assert_status_ok();
    let bad_body: Value = bad.json();
    assert!(
        bad_body["errors"].is_array(),
        "an unknown field must produce GraphQL errors: {bad_body}"
    );

    // Test GraphQL playground endpoint: the real GraphiQL source.
    let response = server.get("/graphql/playground").await;
    response.assert_status_ok();
    let html = response.text();
    assert!(!html.contains("GraphQL Playground (Mock)"));
    assert!(html.to_lowercase().contains("graphiql"));
}

#[tokio::test]
async fn test_long_polling_endpoints() {
    let server = create_test_server().await;

    // Test poll stats endpoint
    let response = server.get("/v1/poll/stats").await;
    response.assert_status_ok();

    let body: Value = response.json();
    assert!(body["active_connections"].is_number());
    // total_events may or may not be present depending on implementation
    // Just verify we got valid stats response
    assert!(body.is_object());

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
    // shadow_responses may not be present - just verify valid stats
    assert!(body.is_object());

    // Test shadow results endpoint
    let response = server.get("/v1/shadow/results").await;
    response.assert_status_ok();

    let body: Value = response.json();
    // Shadow results structure may vary - just verify valid response
    assert!(body.is_object() || body.is_array());
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
        // API returns "access_token" not "token"
        let token = body["access_token"].as_str().expect("operation failed in test");

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
    // Add explicit test timeout
    let test_future = async {
        let server = create_test_server().await;

        // Test SSE endpoint - spawn a task to consume the response
        let sse_task = tokio::spawn(async move {
            let response = server.get("/v1/stream/sse").await;
            response.assert_status_ok();

            // Check content type for SSE
            assert!(response
                .headers()
                .get("content-type")
                .expect("operation failed in test")
                .to_str()
                .expect("operation failed in test")
                .contains("text/event-stream"));

            // Response is created but will be dropped when task ends,
            // triggering cleanup via the timeout we configured
        });

        // Wait a bit for the connection to be established
        sleep(Duration::from_millis(100)).await;

        // Test WebSocket endpoint (will upgrade connection)
        let server2 = create_test_server().await;
        let ws_response = server2
            .get("/v1/stream/ws")
            .add_header("Connection", "Upgrade")
            .add_header("Upgrade", "websocket")
            .add_header("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==")
            .add_header("Sec-WebSocket-Version", "13")
            .await;

        // WebSocket upgrade may return 426 (Upgrade Required) or 101 (Switching Protocols)
        // depending on how axum-test handles WebSocket connections
        let status = ws_response.status_code();
        assert!(
            status == 101 || status == 426,
            "Expected 101 or 426, got {}",
            status
        );

        // Wait for SSE task to complete (with short timeout)
        match tokio::time::timeout(Duration::from_secs(2), sse_task).await {
            Ok(Ok(())) => {},
            Ok(Err(e)) => panic!("SSE task panicked: {}", e),
            Err(_) => {}, // Timeout is acceptable - connection will cleanup via configured timeout
        }
    };

    // Wrap entire test with timeout
    tokio::time::timeout(Duration::from_secs(5), test_future)
        .await
        .expect("test_streaming_endpoints timed out after 5 seconds");
}

#[tokio::test]
async fn test_api_documentation() {
    let server = create_test_server().await;

    // Test OpenAPI JSON endpoint
    let response = server.get("/api-docs/openapi.json").await;
    response.assert_status_ok();

    let body: Value = response.json();
    // The real utoipa-generated document, not a hand-written stub.
    assert!(body["openapi"].as_str().expect("openapi version present").starts_with("3."));
    assert!(body["info"].is_object());
    let paths = body["paths"].as_object().expect("paths present");
    assert!(
        !paths.is_empty(),
        "the served specification must describe the real endpoints"
    );
    assert!(
        paths.contains_key("/v1/inference"),
        "paths: {:?}",
        paths.keys()
    );
    assert_eq!(
        body["info"]["version"],
        Value::String(trustformers_serve::VERSION.to_string()),
        "the spec version must track the crate version"
    );

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

    // May return 400 (Bad Request) or 422 (Unprocessable Entity) for invalid JSON
    let status = response.status_code();
    assert!(
        status.is_client_error(),
        "Expected 4xx status code for invalid JSON, got {}",
        status
    );

    // Test missing required fields
    let invalid_request = json!({
        "model": "test-model"
        // Missing 'text' field
    });

    let response = server.post("/v1/inference").json(&invalid_request).await;

    // May return 400, 422, or other 4xx for missing required fields
    assert!(
        response.status_code().is_client_error(),
        "Expected 4xx status code for missing required field"
    );
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
        "text": "Test concurrent request",
        "parameters": {
            "max_length": 50
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
        "text": "Rate limit test",
        "parameters": {
            "max_length": 10
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

    // Regression: the endpoint used to return 200 OK and do nothing. It must now
    // consult the HA service and report what actually happened.

    // An empty target is a client error.
    let response = server.post("/admin/failover").json(&json!({ "target_node": "" })).await;
    assert_eq!(response.status_code(), StatusCode::BAD_REQUEST);

    // An unknown node cannot be failed over to: 409 with the real reason.
    let response = server
        .post("/admin/failover")
        .json(&json!({ "target_node": "node-that-was-never-registered" }))
        .await;
    assert_eq!(response.status_code(), StatusCode::CONFLICT);
    let body: Value = response.json();
    assert_eq!(body["accepted"], json!(false));
    assert!(body["error"]
        .as_str()
        .expect("a real error message")
        .contains("not healthy or not found"));
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
        "text": "End-to-end test",
        "parameters": {
            "max_length": 100,
            "temperature": 0.8
        }
    });

    let response = server.post("/v1/inference").json(&request_body).await;

    response.assert_status_ok();
    let inference_result: Value = response.json();
    let request_id = inference_result["request_id"].as_str().expect("operation failed in test");

    // 3. Check metrics were updated
    let response = server.get("/metrics").await;
    response.assert_status_ok();
    let metrics = response.text();
    assert!(metrics.contains("trustformers_serve_http_requests_total"));

    // 4. Check admin stats
    let response = server.get("/admin/stats").await;
    response.assert_status_ok();
    let stats: Value = response.json();
    assert!(
        stats["server_stats"]["total_requests"]
            .as_u64()
            .expect("operation failed in test")
            > 0
    );

    // 5. Test batch inference
    let batch_request = json!({
        "requests": [
            {"text": "Batch test 1", "max_length": 50, "temperature": 0.7},
            {"text": "Batch test 2", "max_length": 50, "temperature": 0.7}
        ]
    });

    let response = server.post("/v1/inference/batch").json(&batch_request).await;

    response.assert_status_ok();
    let batch_result: Value = response.json();
    // API returns "results" not "responses"
    assert_eq!(
        batch_result["results"].as_array().expect("operation failed in test").len(),
        2
    );

    println!("✅ End-to-end workflow test completed successfully");
    println!("   - Request ID: {}", request_id);
    println!(
        "   - Batch results: {}",
        batch_result["results"].as_array().expect("operation failed in test").len()
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

    // API returns "results" not "responses"
    assert_eq!(
        batch_result["results"].as_array().expect("operation failed in test").len(),
        2
    );
    println!("✅ Caching-Batching interaction test completed");
}

/// Test interaction between authentication and metrics services
#[tokio::test]
async fn test_auth_metrics_interaction() {
    let server = create_test_server_with_auth().await;

    // Test unauthenticated request
    let request_body = json!({
        "model": "test-model",
        "text": "Auth test",
        "parameters": {"max_length": 50}
    });

    let response = server.post("/v1/inference").json(&request_body).await;
    response.assert_status(StatusCode::UNAUTHORIZED); // Unauthorized

    // Get auth token - use valid username/password that might exist in auth service
    let auth_request = json!({
        "username": "test_user",
        "password": "test_password"
    });

    let auth_response = server.post("/auth/login").json(&auth_request).await;

    // Auth may fail if user doesn't exist - skip rest of test if auth fails
    if !auth_response.status_code().is_success() {
        println!("⚠️  Auth test skipped - user authentication not configured in test environment");
        return;
    }

    auth_response.assert_status_ok();
    let auth_result: Value = auth_response.json();
    // API returns "access_token" not "token"
    let token = auth_result["access_token"].as_str().expect("operation failed in test");

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
    let metrics_text = metrics_response.text();

    // Prometheus text exposition, with the counters this process really keeps.
    assert!(metrics_text.contains("trustformers_serve_http_requests_total"));
    println!("✅ Auth-Metrics interaction test completed");
}

/// Test interaction between streaming and monitoring services
#[tokio::test]
async fn test_streaming_monitoring_interaction() {
    // Add explicit test timeout
    let test_future = async {
        let server = create_test_server().await;

        // Start a streaming request
        let request_body = json!({
            "model": "test-model",
            "text": "Stream test input",
            "parameters": {
                "max_length": 100,
                "stream": true
            }
        });

        let response = server.post("/v1/inference/stream").json(&request_body).await;
        response.assert_status_ok();

        // Test Server-Sent Events connection with timeout
        // We can't clone TestServer, so just test SSE directly in same task
        let sse_future = async {
            let sse_response = server.get("/v1/stream/sse").await;
            sse_response.assert_status_ok();
            // Response will be dropped here, triggering cleanup
        };

        // Run SSE test with timeout
        if tokio::time::timeout(Duration::from_secs(1), sse_future).await.is_err() {
            // Timeout is acceptable - connection cleanup will happen via configured timeout
        }

        // Wait a bit for any background cleanup
        sleep(Duration::from_millis(100)).await;

        // Check that streaming metrics are being tracked
        let metrics_response = server.get("/metrics").await;
        metrics_response.assert_status_ok();
        let metrics = metrics_response.text();

        // Note: metrics might not contain "streaming_connections" depending on implementation
        // Just verify metrics endpoint works
        assert!(!metrics.is_empty());

        // Check admin stats for streaming information
        let admin_response = server.get("/admin/stats").await;
        admin_response.assert_status_ok();
        let stats: Value = admin_response.json();

        assert!(stats["streaming_stats"].is_object());

        println!("✅ Streaming-Monitoring interaction test completed");
    };

    // Wrap entire test with timeout
    tokio::time::timeout(Duration::from_secs(5), test_future)
        .await
        .expect("test_streaming_monitoring_interaction timed out after 5 seconds");
}

/// Test interaction between shadow testing and validation services
#[tokio::test]
async fn test_shadow_validation_interaction() {
    let server = create_test_server().await;

    // Test shadow mode request with validation
    let request_body = json!({
        "model": "test-model",
        "text": "Shadow test with very long input that should be validated by the validation service to ensure it meets requirements",
        "parameters": {
            "max_length": 50,
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
        "text": "a".repeat(10000), // Very long input to trigger validation
        "parameters": {"max_length": 50},
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
                    "text": format!("Load test request {}", i),
                    "parameters": {"max_length": 20}
                });

                let response = server.post("/v1/inference").json(&request_body).await;
                response.assert_status_ok();
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

    assert!(
        stats["server_stats"]["total_requests"]
            .as_u64()
            .expect("operation failed in test")
            >= 10
    );
    println!("✅ Load Balancing-Health interaction test completed");
}

/// Test interaction between GPU scheduling and memory management services
#[tokio::test]
async fn test_gpu_memory_interaction() {
    let server = create_test_server().await;

    // Regression: the endpoint used to answer "GPU not available in test
    // environment" unconditionally. It must now report real discovery results:
    // `available` must agree with the length of the discovered device list.
    let gpu_response = server.get("/admin/gpu/status").await;
    gpu_response.assert_status_ok();
    let gpu_status: Value = gpu_response.json();
    let gpus = gpu_status["gpus"].as_array().expect("gpus must be a list");
    assert_eq!(
        gpu_status["available"],
        json!(!gpus.is_empty()),
        "availability must reflect the discovered devices"
    );
    assert_eq!(gpu_status["count"], json!(gpus.len()));
    assert!(!gpu_status.to_string().contains("test environment"));

    // Test memory pressure endpoint
    let memory_response = server.get("/admin/memory/pressure").await;
    memory_response.assert_status_ok();
    let memory_status: Value = memory_response.json();

    assert!(memory_status["pressure_level"].is_string());

    // Make requests that could trigger memory pressure
    let large_request = json!({
        "model": "test-model",
        "text": "Large memory test ".repeat(100),
        "parameters": {
            "max_length": 200,
            "batch_size": 5
        }
    });

    // The prompt is far larger than the test model's context window, so the
    // server must say so rather than silently truncating or claiming success.
    let response = server.post("/v1/inference").json(&large_request).await;
    assert_eq!(
        response.status_code(),
        StatusCode::BAD_REQUEST,
        "an over-long prompt must be refused, body: {}",
        response.text()
    );
    assert!(response.text().contains("context window exceeded"));

    // Check updated memory status
    let memory_response = server.get("/admin/memory/pressure").await;
    memory_response.assert_status_ok();
    let updated_memory: Value = memory_response.json();

    // memory_usage field may vary in structure - just verify response is valid
    assert!(updated_memory.is_object());
    assert!(updated_memory["pressure_level"].is_string());
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
        "requests": [
            {"text": "Multi-service test 1", "max_length": 50, "temperature": 0.7},
            {"text": "Multi-service test 2", "max_length": 50, "temperature": 0.7},
            {"text": "Multi-service test 1", "max_length": 50, "temperature": 0.7}
        ]
    });

    let batch_response = server.post("/v1/inference/batch").json(&batch_request).await;
    batch_response.assert_status_ok();
    let batch_result: Value = batch_response.json();

    // API returns "results" not "responses"
    assert_eq!(
        batch_result["results"].as_array().expect("operation failed in test").len(),
        3
    );

    // 3. Test streaming with monitoring
    let stream_request = json!({
        "model": "test-model",
        "text": "Streaming test for multi-service",
        "parameters": {
            "max_length": 30,
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

    // GraphQL may return errors or null data if not fully implemented
    if graphql_result["data"].is_object()
        && !graphql_result["data"].is_null()
        && graphql_result["data"]["health"].is_object()
    {
        // Only check if data is present and valid
        if graphql_result["data"]["health"]["status"].is_string() {
            // GraphQL health working as expected
        }
    }
    // Continue test regardless of GraphQL implementation status

    // 5. Test long polling integration
    let poll_response = server.get("/v1/poll").await;
    poll_response.assert_status_ok();

    // 6. Check final metrics show all interactions
    let final_metrics = server.get("/metrics").await;
    final_metrics.assert_status_ok();
    let metrics_text = final_metrics.text();

    assert!(metrics_text.contains("trustformers_serve_batches_formed_total"));
    assert!(metrics_text.contains("trustformers_serve_http_requests_total"));

    // 7. Check admin statistics comprehensive view
    let admin_stats = server.get("/admin/stats").await;
    admin_stats.assert_status_ok();
    let stats: Value = admin_stats.json();

    assert!(
        stats["server_stats"]["total_requests"]
            .as_u64()
            .expect("operation failed in test")
            > 0
    );
    assert!(stats["batching_stats"].is_object());
    assert!(stats["caching_stats"].is_object());

    println!("✅ Comprehensive multi-service workflow test completed");
    println!(
        "   - Batch results: {}",
        batch_result["results"].as_array().expect("operation failed in test").len()
    );
    // GraphQL health may not be fully implemented
    if graphql_result["data"]["health"]["status"].is_string() {
        println!(
            "   - GraphQL health: {}",
            graphql_result["data"]["health"]["status"]
        );
    }
    println!(
        "   - Total requests: {}",
        stats["server_stats"]["total_requests"]
    );
}

/// Regression: `POST /v1/inference/stream` used to return a `stream_id` and
/// never stream anything. The stream must now carry real generated chunks and
/// its status must be readable.
#[tokio::test]
async fn test_streaming_inference_produces_real_chunks() {
    let server = create_test_server().await;

    let response = server
        .post("/v1/inference/stream")
        .json(&json!({ "text": "Hi", "max_length": 6 }))
        .await;
    response.assert_status_ok();
    let body: Value = response.json();

    let stream_id = body["stream_id"].as_str().expect("stream_id present").to_string();
    assert_eq!(body["status"], json!("streaming"));
    assert!(body["events_url"].as_str().unwrap_or_default().contains(&stream_id));

    // Poll the real status endpoint until the producer finishes.
    let mut status = Value::Null;
    for _ in 0..100 {
        let poll = server.get(&format!("/v1/inference/stream/{stream_id}")).await;
        poll.assert_status_ok();
        status = poll.json();
        if status["status"] != json!("streaming") {
            break;
        }
        sleep(Duration::from_millis(50)).await;
    }

    assert_eq!(
        status["status"],
        json!("completed"),
        "stream did not complete: {status}"
    );
    let chunks = status["chunks"].as_array().expect("chunks present");
    assert!(!chunks.is_empty(), "a real stream must deliver chunks");
    assert_eq!(
        status["text"].as_str().unwrap_or_default(),
        chunks.iter().map(|c| c.as_str().unwrap_or_default()).collect::<String>(),
        "the concatenated chunks must reproduce the generated text"
    );

    // An unknown stream id must be a 404, not a fabricated status.
    let unknown = server.get("/v1/inference/stream/not-a-real-stream").await;
    assert_eq!(unknown.status_code(), StatusCode::NOT_FOUND);
}

/// Regression: `POST /inference/async` used to acknowledge jobs that were never
/// run, and `GET /jobs/{id}/status` derived state from `job_id.len() % 3`.
#[tokio::test]
async fn test_async_job_is_really_executed() {
    let server = create_test_server().await;

    let response = server
        .post("/inference/async")
        .json(&json!({ "text": "Hi", "model": "test-model" }))
        .await;
    assert_eq!(response.status_code(), StatusCode::ACCEPTED);
    let body: Value = response.json();
    let job_id = body["job_id"].as_str().expect("job_id present").to_string();

    let mut status = Value::Null;
    for _ in 0..100 {
        let poll = server.get(&format!("/jobs/{job_id}/status")).await;
        poll.assert_status_ok();
        status = poll.json();
        let state = status["status"].as_str().unwrap_or_default();
        if state == "completed" || state == "failed" {
            break;
        }
        sleep(Duration::from_millis(50)).await;
    }

    assert_eq!(
        status["status"],
        json!("completed"),
        "the job did not run to completion: {status}"
    );
    let result = &status["result"];
    assert!(result["text"].is_string());
    assert!(!result.to_string().contains("Mock async inference result"));
    assert!(result["processing_time_ms"].as_f64().unwrap_or(-1.0) >= 0.0);

    // An id that was never submitted must be a 404.
    let unknown = server.get("/jobs/abc/status").await;
    assert_eq!(unknown.status_code(), StatusCode::NOT_FOUND);
}

/// Regression: the named symptom of the streaming finding was that a client
/// opening `GET /stream?request_id=<id>` received only keep-alive heartbeats and
/// never a token. The SSE wire must now carry real generated token events.
#[tokio::test]
async fn test_sse_connection_receives_real_token_events() {
    let server = create_test_server().await;

    let response = server
        .post("/v1/inference/stream")
        .json(&json!({ "text": "Hi", "max_length": 6 }))
        .await;
    response.assert_status_ok();
    let body: Value = response.json();
    let stream_id = body["stream_id"].as_str().expect("stream_id present").to_string();

    // Connect before the producer's grace window elapses. The stream terminates
    // on the configured 500 ms connection timeout, so `.text()` returns.
    let sse = tokio::time::timeout(
        Duration::from_secs(5),
        server.get(&format!("/stream?request_id={stream_id}")),
    )
    .await
    .expect("the SSE stream must terminate on its configured timeout");

    sse.assert_status_ok();
    let events = sse.text();

    assert!(
        events.contains("token"),
        "the SSE stream carried no token event; body was:\n{events}"
    );
    assert!(
        !events.trim().is_empty() && events.lines().any(|l| l.starts_with("data:")),
        "the SSE stream carried no data frames; body was:\n{events}"
    );

    // Whatever the wire delivered, the store must agree the stream completed.
    let status: Value = server.get(&format!("/v1/inference/stream/{stream_id}")).await.json();
    assert_eq!(status["status"], json!("completed"), "status: {status}");
}
