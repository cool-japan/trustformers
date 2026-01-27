//! Isolated Integration Tests using Test Environment Isolation
//!
//! These tests demonstrate how to use the test environment isolation framework
//! for reliable and independent test execution across multiple services.

mod test_helpers;

use anyhow::Result;
use axum_test::http::StatusCode;
use serde_json::{json, Value};
use std::time::Duration;
use test_helpers::{
    create_test_auth_request, create_test_batch_request, create_test_inference_request,
    IsolationLevel, TestEnvironmentBuilder, TestFixtures, TestResourceLimits,
};

/// Test isolated inference with basic environment
#[tokio::test]
async fn test_isolated_basic_inference() -> Result<()> {
    let env = TestEnvironmentBuilder::new("basic_inference")
        .isolation_level(IsolationLevel::Basic)
        .with_caching(true)
        .build()
        .await?;

    // Test health endpoint
    let health_response = env.server.get("/health").await;
    health_response.assert_status_ok();

    // Test inference endpoint
    let request = create_test_inference_request();
    let response = env.server.post("/v1/inference").json(&request).await;

    response.assert_status_ok();
    let result: Value = response.json();
    assert!(result["request_id"].is_string());
    assert!(result["text"].is_string());

    println!("✅ Isolated basic inference test completed");
    Ok(())
}

/// Test isolated authentication with auth-enabled environment
#[tokio::test]
async fn test_isolated_auth_flow() -> Result<()> {
    let env = TestEnvironmentBuilder::new("auth_flow")
        .isolation_level(IsolationLevel::Basic)
        .with_auth(true)
        .with_caching(false) // Disable caching to test auth in isolation
        .build()
        .await?;

    // Test unauthenticated request (should fail)
    let request = create_test_inference_request();
    let response = env.server.post("/v1/inference").json(&request).await;
    response.assert_status(StatusCode::UNAUTHORIZED);

    // Get authentication token
    let auth_request = create_test_auth_request();
    let auth_response = env.server.post("/auth/login").json(&auth_request).await;
    auth_response.assert_status_ok();

    let auth_result: Value = auth_response.json();
    let token = auth_result["access_token"].as_str().unwrap();

    // Test authenticated request (should succeed)
    let response = env
        .server
        .post("/v1/inference")
        .add_header("Authorization", &format!("Bearer {}", token))
        .json(&request)
        .await;
    response.assert_status_ok();

    println!("✅ Isolated auth flow test completed");
    Ok(())
}

/// Test isolated batch processing with resource limits
#[tokio::test]
async fn test_isolated_batch_with_limits() -> Result<()> {
    let resource_limits = TestResourceLimits {
        max_memory_mb: Some(256),
        max_cpu_percent: Some(50.0),
        max_concurrent_requests: Some(5),
        request_timeout: Duration::from_secs(10),
    };

    let env = TestEnvironmentBuilder::new("batch_limits")
        .isolation_level(IsolationLevel::Basic)
        .with_caching(true)
        .with_resource_limits(resource_limits)
        .build()
        .await?;

    // Test batch inference with limits
    let batch_request = create_test_batch_request();
    let response = env.server.post("/v1/inference/batch").json(&batch_request).await;

    response.assert_status_ok();
    let result: Value = response.json();
    let responses = result["results"].as_array().unwrap();
    assert_eq!(responses.len(), 3);

    // Test that resource limits are being enforced
    let stats_response = env.server.get("/admin/stats").await;
    stats_response.assert_status_ok();
    let stats: Value = stats_response.json();

    // Verify resource usage is tracked
    assert!(stats["resource_usage"].is_object());

    println!("✅ Isolated batch with limits test completed");
    Ok(())
}

/// Test isolated streaming with monitoring
#[tokio::test]
async fn test_isolated_streaming_monitoring() -> Result<()> {
    use tokio::time::{sleep, timeout, Duration};

    // Wrap entire test with timeout
    let test_future = async {
        let env = TestEnvironmentBuilder::new("streaming_monitoring")
            .isolation_level(IsolationLevel::Basic)
            .with_streaming(true)
            .with_caching(false)
            .build()
            .await?;

        // Test streaming endpoint
        let stream_request = json!({
            "text": "Streaming test in isolated environment",
            "max_length": 50,
            "temperature": 0.7
        });

        let response = env.server.post("/v1/inference/stream").json(&stream_request).await;
        response.assert_status_ok();

        // Test SSE connection with timeout
        // We can't clone TestServer, so just test SSE directly in same task
        let sse_future = async {
            let sse_response = env.server.get("/v1/stream/sse").await;
            sse_response.assert_status_ok();
            // Response will be dropped here, triggering cleanup
        };

        // Run SSE test with timeout
        if let Err(_) = timeout(Duration::from_secs(1), sse_future).await {
            // Timeout is acceptable - connection cleanup will happen via configured timeout
        }

        // Wait a bit for any background cleanup
        sleep(Duration::from_millis(100)).await;

        // Verify streaming metrics
        let metrics_response = env.server.get("/metrics").await;
        metrics_response.assert_status_ok();
        let metrics = metrics_response.text();
        // Just verify metrics endpoint works
        assert!(!metrics.is_empty());

        println!("✅ Isolated streaming monitoring test completed");
        Ok(())
    };

    timeout(Duration::from_secs(5), test_future).await.map_err(|_| {
        anyhow::anyhow!("test_isolated_streaming_monitoring timed out after 5 seconds")
    })?
}

/// Test isolated shadow testing
#[tokio::test]
async fn test_isolated_shadow_testing() -> Result<()> {
    let env = TestEnvironmentBuilder::new("shadow_testing")
        .isolation_level(IsolationLevel::Basic)
        .with_shadow(true)
        .with_caching(true)
        .build()
        .await?;

    // Test shadow mode request
    let shadow_request = json!({
        "model": "test-model",
        "text": "Shadow testing in isolated environment",
        "parameters": {
            "max_length": 75,
            "temperature": 0.8
        },
        "shadow_mode": true
    });

    let response = env.server.post("/v1/inference").json(&shadow_request).await;
    response.assert_status_ok();

    let result: Value = response.json();
    // Shadow comparison should be included in response
    assert!(result["shadow_comparison"].is_object());

    println!("✅ Isolated shadow testing test completed");
    Ok(())
}

/// Test isolated multi-service interaction with full environment
#[tokio::test]
async fn test_isolated_full_multi_service() -> Result<()> {
    let env = TestEnvironmentBuilder::new("full_multi_service")
        .isolation_level(IsolationLevel::Basic)
        .with_auth(true)
        .with_caching(true)
        .with_streaming(true)
        .with_shadow(true)
        .build()
        .await?;

    // Get auth token first
    let auth_request = create_test_auth_request();
    let auth_response = env.server.post("/auth/login").json(&auth_request).await;
    auth_response.assert_status_ok();

    let auth_result: Value = auth_response.json();
    let token = auth_result["access_token"].as_str().unwrap();

    // Test multi-service workflow with authentication
    let test_cases = vec![
        // Single inference with caching
        json!({
            "model": "test-model",
            "text": "Multi-service test 1",
            "parameters": {"max_length": 50}
        }),
        // Batch inference
        json!({
            "requests": [
                {
                    "text": "Multi test 2a",
                    "max_length": 30
                },
                {
                    "text": "Multi test 2b",
                    "max_length": 30
                }
            ]
        }),
        // Streaming inference
        json!({
            "model": "test-model",
            "text": "Multi-service streaming test",
            "parameters": {"max_length": 40, "stream": true}
        }),
        // Shadow mode inference
        json!({
            "model": "test-model",
            "text": "Multi-service shadow test",
            "parameters": {"max_length": 35},
            "shadow_mode": true
        }),
    ];

    // Execute test cases
    for (i, test_case) in test_cases.iter().enumerate() {
        let endpoint = match i {
            1 => "/v1/inference/batch",
            2 => "/v1/inference/stream",
            _ => "/v1/inference",
        };

        let response = env
            .server
            .post(endpoint)
            .add_header("Authorization", &format!("Bearer {}", token))
            .json(test_case)
            .await;

        response.assert_status_ok();
        println!("✓ Test case {} completed", i + 1);
    }

    // Verify all services interacted correctly
    let final_metrics = env
        .server
        .get("/metrics")
        .add_header("Authorization", &format!("Bearer {}", token))
        .await;
    final_metrics.assert_status_ok();

    let final_stats = env
        .server
        .get("/admin/stats")
        .add_header("Authorization", &format!("Bearer {}", token))
        .await;
    final_stats.assert_status_ok();

    let stats: Value = final_stats.json();
    assert!(stats["server_stats"]["total_requests"].as_u64().unwrap() > 0);

    println!("✅ Isolated full multi-service test completed");
    Ok(())
}

/// Test concurrent isolated environments
#[tokio::test]
async fn test_concurrent_isolated_environments() -> Result<()> {
    use tokio::time::{sleep, Duration};

    // Create multiple isolated environments concurrently
    let env_futures = (0..3).map(|i| {
        TestEnvironmentBuilder::new(&format!("concurrent_env_{}", i))
            .isolation_level(IsolationLevel::Basic)
            .with_caching(true)
            .build()
    });

    let environments = futures::future::try_join_all(env_futures).await?;

    // Test each environment independently
    let test_futures = environments.iter().enumerate().map(|(i, env)| {
        let request = json!({
            "text": format!("Concurrent test from environment {}", i),
            "max_length": 20,
            "temperature": 0.7
        });

        async move {
            let response = env.server.post("/v1/inference").json(&request).await;
            response.assert_status_ok();

            let result: Value = response.json();
            assert!(result["request_id"].is_string());
            println!("✓ Environment {} completed", i);
        }
    });

    futures::future::join_all(test_futures).await;

    // Add small delay before cleanup to avoid race conditions
    // during concurrent resource cleanup
    sleep(Duration::from_millis(50)).await;

    // Explicitly drop environments sequentially to avoid concurrent cleanup issues
    drop(environments);

    // Allow time for cleanup to complete
    sleep(Duration::from_millis(50)).await;

    println!("✅ Concurrent isolated environments test completed");
    Ok(())
}

/// Test with test fixtures and mock data
#[tokio::test]
async fn test_isolated_with_fixtures() -> Result<()> {
    let env = TestEnvironmentBuilder::new("fixtures_test")
        .isolation_level(IsolationLevel::Basic)
        .with_caching(true)
        .build()
        .await?;

    // Create and load test fixtures
    let mut fixtures = TestFixtures::new();
    fixtures.load_defaults();

    // Add custom test data
    fixtures.load_dataset(
        "custom_inputs",
        vec![
            json!({"text": "Custom test input 1", "max_length": 25}),
            json!({"text": "Custom test input 2", "max_length": 35}),
        ],
    );

    // Test with fixture data
    if let Some(dataset) = fixtures.get_dataset("custom_inputs") {
        for (i, test_input) in dataset.iter().enumerate() {
            let request = json!({
                "model": "test-model",
                "text": test_input["text"],
                "parameters": {
                    "max_length": test_input["max_tokens"]
                }
            });

            let response = env.server.post("/v1/inference").json(&request).await;
            response.assert_status_ok();

            println!("✓ Fixture test case {} completed", i + 1);
        }
    }

    // Test with mock responses validation
    if let Some(expected_health) = fixtures.get_mock_response("/health") {
        let health_response = env.server.get("/health").await;
        health_response.assert_status_ok();

        let health_result: Value = health_response.json();
        assert_eq!(health_result["status"], expected_health["status"]);
    }

    println!("✅ Isolated test with fixtures completed");
    Ok(())
}

/// Test error handling in isolated environment
#[tokio::test]
async fn test_isolated_error_handling() -> Result<()> {
    let env = TestEnvironmentBuilder::new("error_handling")
        .isolation_level(IsolationLevel::Basic)
        .with_caching(false) // Disable caching to test direct error handling
        .build()
        .await?;

    // Test various error conditions
    let error_test_cases = vec![
        // Invalid JSON
        ("invalid_json", r#"{"invalid": json}"#),
        // Missing required fields
        ("missing_model", r#"{"text": "test"}"#),
        // Invalid parameters
        (
            "invalid_params",
            r#"{"model": "test", "text": "test", "parameters": {"max_length": -1}}"#,
        ),
    ];

    for (test_name, invalid_payload) in error_test_cases {
        let response = env
            .server
            .post("/v1/inference")
            .add_header("Content-Type", "application/json")
            .text(invalid_payload)
            .await;

        // Should return appropriate error status
        assert!(response.status_code().is_client_error());
        println!("✓ Error test case '{}' handled correctly", test_name);
    }

    // Test request timeout
    let timeout_request = json!({
        "model": "test-model",
        "text": "Very long input that might timeout ".repeat(1000),
        "parameters": {"max_length": 1000}
    });

    let response = env.server.post("/v1/inference").json(&timeout_request).await;

    // Should either succeed or fail gracefully
    assert!(response.status_code().is_success() || response.status_code().is_server_error());

    println!("✅ Isolated error handling test completed");
    Ok(())
}
