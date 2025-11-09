/*!
# Distributed Tracing Demo

This example demonstrates how to use the comprehensive distributed tracing system
in trustformers-serve for production observability.

## Features Demonstrated

- Setting up tracing with different backends (Console, Jaeger, Zipkin, OTLP)
- Creating spans for inference operations
- Trace context propagation across HTTP boundaries
- Custom span attributes and events
- Sampling strategies for production environments
- Real-time tracing event monitoring

## Usage

```bash
# Run with console output
cargo run --example distributed_tracing_demo

# Run with Jaeger backend (requires Jaeger running)
JAEGER_ENDPOINT=http://localhost:14268/api/traces cargo run --example distributed_tracing_demo

# Run with custom sampling
SAMPLING_RATE=0.1 cargo run --example distributed_tracing_demo
```

## Starting Jaeger (optional)

```bash
# Using Docker
docker run -d --name jaeger \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 5775:5775/udp \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 14268:14268 \
  -p 14250:14250 \
  -p 9411:9411 \
  jaegertracing/all-in-one:latest

# View traces at http://localhost:16686
```
*/

// Allow unused code for example/demo purposes
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use anyhow::Result;
use std::{collections::HashMap, env, sync::Arc, time::Duration};
use tokio::time::sleep;
use tracing::{info, warn};
use trustformers_serve::distributed_tracing::{
    SamplingStrategy, SpanKind, SpanStatus, TracingConfig, TracingEvent, TracingManager,
};

/// Simulated inference request
#[derive(Debug, Clone)]
struct InferenceRequest {
    id: String,
    model_name: String,
    input_text: String,
    max_tokens: usize,
}

/// Simulated inference response
#[derive(Debug, Clone)]
struct InferenceResponse {
    request_id: String,
    generated_text: String,
    tokens_generated: usize,
    latency_ms: u64,
}

/// Simulated model service
struct ModelService {
    tracing_manager: Arc<TracingManager>,
}

impl ModelService {
    fn new(tracing_manager: Arc<TracingManager>) -> Self {
        Self { tracing_manager }
    }

    /// Simulate inference with distributed tracing
    async fn inference(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        // Start inference span
        let span = self.tracing_manager.start_inference_span(&request.model_name, &request.id)?;

        // Add request attributes
        span.set_attribute("input_length", request.input_text.len().to_string());
        span.set_attribute("max_tokens", request.max_tokens.to_string());
        span.set_attribute("model.version", "1.0.0");
        span.set_attribute("request.type", "text_generation");

        // Add model loading event
        span.add_event(
            "model_loading_started",
            vec![("model_size_gb", "7.5"), ("device", "cuda:0")],
        );

        // Simulate model loading time
        sleep(Duration::from_millis(100)).await;

        span.add_event(
            "model_loaded",
            vec![("load_time_ms", "100"), ("memory_used_gb", "6.2")],
        );

        // Simulate tokenization
        let tokenization_span = self.tracing_manager.start_span("tokenization")?;
        tokenization_span.set_attribute("tokenizer", "sentencepiece");
        tokenization_span.set_attribute("vocab_size", "32000");

        sleep(Duration::from_millis(10)).await;

        let input_tokens = request.input_text.split_whitespace().count();
        tokenization_span.set_attribute("input_tokens", input_tokens.to_string());
        tokenization_span.finish();

        // Simulate inference computation
        let inference_span =
            self.tracing_manager.start_span_with_kind("forward_pass", SpanKind::Internal)?;
        inference_span.set_attribute("batch_size", "1");
        inference_span.set_attribute("sequence_length", input_tokens.to_string());

        // Add computation events
        inference_span.add_event(
            "attention_computation",
            vec![("num_heads", "32"), ("head_dim", "128")],
        );

        sleep(Duration::from_millis(200)).await;

        inference_span.add_event(
            "feedforward_computation",
            vec![("hidden_size", "4096"), ("intermediate_size", "11008")],
        );

        sleep(Duration::from_millis(150)).await;

        let generated_tokens = std::cmp::min(request.max_tokens, 50);
        inference_span.set_attribute("generated_tokens", generated_tokens.to_string());
        inference_span.finish();

        // Simulate post-processing
        let postprocess_span = self.tracing_manager.start_span("post_processing")?;
        postprocess_span.set_attribute("detokenization", "true");
        postprocess_span.set_attribute("output_filtering", "enabled");

        sleep(Duration::from_millis(20)).await;
        postprocess_span.finish();

        // Create response
        let response = InferenceResponse {
            request_id: request.id.clone(),
            generated_text: format!("Generated text for: {}", request.input_text),
            tokens_generated: generated_tokens,
            latency_ms: 480, // Total simulated latency
        };

        // Add final span attributes
        span.set_attribute("output_length", response.generated_text.len().to_string());
        span.set_attribute(
            "total_tokens",
            (input_tokens + generated_tokens).to_string(),
        );
        span.set_attribute("latency_ms", response.latency_ms.to_string());
        span.set_attribute(
            "throughput_tokens_per_sec",
            ((input_tokens + generated_tokens) as f64 / (response.latency_ms as f64 / 1000.0))
                .to_string(),
        );

        // Set success status
        span.set_status(SpanStatus::Ok);

        span.add_event(
            "inference_completed",
            vec![("success", "true"), ("tokens_per_second", "104.2")],
        );

        span.finish();

        info!("Inference completed for request {}", request.id);
        Ok(response)
    }

    /// Simulate batch inference with distributed tracing
    async fn batch_inference(
        &self,
        requests: Vec<InferenceRequest>,
    ) -> Result<Vec<InferenceResponse>> {
        let batch_span =
            self.tracing_manager.start_span_with_kind("batch_inference", SpanKind::Server)?;
        batch_span.set_attribute("batch_size", requests.len().to_string());
        batch_span.set_attribute("operation.type", "batch_inference");

        let mut responses = Vec::new();

        for (i, request) in requests.into_iter().enumerate() {
            // Create child span for each request in the batch
            let request_span = self
                .tracing_manager
                .start_span_with_kind(format!("batch_request_{}", i), SpanKind::Internal)?;
            request_span.set_attribute("batch_index", i.to_string());
            request_span.set_attribute("request_id", request.id.clone());

            let response = self.inference(request).await?;
            responses.push(response);

            request_span.set_status(SpanStatus::Ok);
            request_span.finish();
        }

        batch_span.set_attribute("completed_requests", responses.len().to_string());
        batch_span.set_status(SpanStatus::Ok);
        batch_span.finish();

        Ok(responses)
    }
}

/// HTTP request simulation with trace context propagation
async fn simulate_http_request(
    tracing_manager: Arc<TracingManager>,
    request_id: String,
) -> Result<()> {
    // Simulate incoming HTTP request with trace context
    let http_span = tracing_manager.start_http_span("POST", "/v1/inference")?;
    http_span.set_attribute("request_id", request_id.clone());
    http_span.set_attribute("user_agent", "TrustformersClient/1.0");
    http_span.set_attribute("client_ip", "192.168.1.100");

    // Get trace context for propagation
    let trace_context = http_span.get_context();

    // Simulate context propagation to downstream services
    let mut headers = HashMap::new();
    tracing_manager.inject_context(&trace_context, &mut headers);

    info!(
        "Propagating trace context: traceparent={}",
        headers.get("traceparent").unwrap_or(&"none".to_string())
    );

    // Simulate authentication span
    let auth_span = tracing_manager.start_span_with_kind("authentication", SpanKind::Internal)?;
    auth_span.set_attribute("auth_method", "api_key");
    auth_span.set_attribute("user_id", "user_123");
    sleep(Duration::from_millis(50)).await;
    auth_span.set_status(SpanStatus::Ok);
    auth_span.finish();

    // Simulate validation span
    let validation_span =
        tracing_manager.start_span_with_kind("request_validation", SpanKind::Internal)?;
    validation_span.set_attribute("validation_rules", "length,profanity,pii");
    sleep(Duration::from_millis(20)).await;
    validation_span.set_status(SpanStatus::Ok);
    validation_span.finish();

    // Create inference request
    let inference_request = InferenceRequest {
        id: request_id.clone(),
        model_name: "llama-7b-chat".to_string(),
        input_text: "What is the capital of France?".to_string(),
        max_tokens: 100,
    };

    // Call model service
    let model_service = ModelService::new(tracing_manager.clone());
    let response = model_service.inference(inference_request).await?;

    // Add response attributes
    http_span.set_attribute("response_size", response.generated_text.len().to_string());
    http_span.set_attribute("status_code", "200");
    http_span.set_attribute("tokens_generated", response.tokens_generated.to_string());

    http_span.set_status(SpanStatus::Ok);
    http_span.finish();

    info!("HTTP request completed: {}", request_id);
    Ok(())
}

/// Setup tracing based on environment variables
fn setup_tracing_config() -> TracingConfig {
    let mut config = TracingConfig::new("trustformers-serve")
        .with_service_version("1.0.0")
        .with_resource_attribute("environment", "demo")
        .with_resource_attribute("region", "us-west-2")
        .with_resource_attribute("instance_id", "demo-001");

    // Configure backend from environment
    if let Ok(jaeger_endpoint) = env::var("JAEGER_ENDPOINT") {
        info!("Using Jaeger backend: {}", jaeger_endpoint);
        config = config.with_jaeger_endpoint(jaeger_endpoint);
    } else if let Ok(zipkin_endpoint) = env::var("ZIPKIN_ENDPOINT") {
        info!("Using Zipkin backend: {}", zipkin_endpoint);
        config = config.with_zipkin_endpoint(zipkin_endpoint);
    } else if let Ok(otlp_endpoint) = env::var("OTLP_ENDPOINT") {
        info!("Using OTLP backend: {}", otlp_endpoint);
        config = config.with_otlp_endpoint(otlp_endpoint);
    } else {
        info!("Using console backend for demo");
        // Console backend is already default
    }

    // Configure sampling from environment
    if let Ok(sampling_rate) = env::var("SAMPLING_RATE") {
        if let Ok(rate) = sampling_rate.parse::<f64>() {
            info!("Using probabilistic sampling: {}", rate);
            config = config.with_sampling(SamplingStrategy::Probabilistic(rate));
        }
    } else if env::var("DISABLE_SAMPLING").is_ok() {
        info!("Disabling sampling");
        config = config.with_sampling(SamplingStrategy::Never);
    } else {
        info!("Using always sampling for demo");
        config = config.with_sampling(SamplingStrategy::Always);
    }

    config
}

/// Monitor tracing events in real-time
async fn monitor_tracing_events(tracing_manager: Arc<TracingManager>) {
    let mut event_receiver = tracing_manager.subscribe_events();

    info!("Starting tracing event monitor...");

    while let Ok(event) = event_receiver.recv().await {
        match event {
            TracingEvent::SpanStarted { span_id, trace_id } => {
                info!("🟢 Span started: {} (trace: {})", span_id, trace_id);
            },
            TracingEvent::SpanFinished {
                span_id,
                duration_ms,
            } => {
                info!("🔴 Span finished: {} ({}ms)", span_id, duration_ms);
            },
            TracingEvent::ExportCompleted { span_count } => {
                info!("📤 Exported {} spans", span_count);
            },
            TracingEvent::ExportFailed { error } => {
                warn!("❌ Export failed: {}", error);
            },
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    info!("🚀 Starting distributed tracing demo");

    // Setup tracing configuration
    let config = setup_tracing_config();

    // Create tracing manager
    let tracing_manager = TracingManager::new(config).await?;

    info!("✅ Distributed tracing initialized");

    // Start event monitoring in background
    let manager_for_monitor = tracing_manager.clone();
    tokio::spawn(async move {
        monitor_tracing_events(manager_for_monitor).await;
    });

    // Display current statistics
    let stats = tracing_manager.get_stats();
    info!("📊 Initial stats: {:?}", stats);

    // Demo 1: Single inference request
    info!("\n🔍 Demo 1: Single inference request");
    simulate_http_request(tracing_manager.clone(), "req-001".to_string()).await?;

    // Demo 2: Batch inference
    info!("\n📦 Demo 2: Batch inference");
    let batch_requests = vec![
        InferenceRequest {
            id: "batch-001".to_string(),
            model_name: "llama-7b-chat".to_string(),
            input_text: "Explain quantum computing".to_string(),
            max_tokens: 150,
        },
        InferenceRequest {
            id: "batch-002".to_string(),
            model_name: "llama-7b-chat".to_string(),
            input_text: "What is machine learning?".to_string(),
            max_tokens: 100,
        },
        InferenceRequest {
            id: "batch-003".to_string(),
            model_name: "llama-7b-chat".to_string(),
            input_text: "Describe neural networks".to_string(),
            max_tokens: 200,
        },
    ];

    let model_service = ModelService::new(tracing_manager.clone());
    let _batch_responses = model_service.batch_inference(batch_requests).await?;

    // Demo 3: Concurrent requests
    info!("\n🔄 Demo 3: Concurrent requests");
    let mut handles = Vec::new();
    for i in 0..5 {
        let manager = tracing_manager.clone();
        let handle = tokio::spawn(async move {
            simulate_http_request(manager, format!("concurrent-{:03}", i)).await
        });
        handles.push(handle);
    }

    // Wait for all concurrent requests
    for handle in handles {
        handle.await??;
    }

    // Demo 4: Error simulation
    info!("\n❌ Demo 4: Error simulation");
    let error_span = tracing_manager.start_inference_span("llama-7b-chat", "error-001")?;
    error_span.set_attribute("input_length", "5000");
    error_span.add_event("validation_started", vec![] as Vec<(String, String)>);

    // Simulate error
    sleep(Duration::from_millis(50)).await;
    error_span.set_error("Input text too long: 5000 tokens exceeds limit of 4096");
    error_span.add_event(
        "error_occurred",
        vec![("error_type", "validation_error"), ("error_code", "E001")],
    );
    error_span.finish();

    // Wait a bit for export
    sleep(Duration::from_secs(2)).await;

    // Display final statistics
    let final_stats = tracing_manager.get_stats();
    info!("\n📊 Final statistics:");
    info!("  Spans created: {}", final_stats.spans_created);
    info!("  Spans exported: {}", final_stats.spans_exported);
    info!("  Export failures: {}", final_stats.export_failures);
    info!("  Queue size: {}", final_stats.queue_size);
    info!(
        "  Average span duration: {:.2}ms",
        final_stats.avg_span_duration_ms
    );
    info!("  Sampling rate: {:.2}", final_stats.sampling_rate);
    info!("  Export latency: {:.2}ms", final_stats.export_latency_ms);

    // Flush remaining spans
    info!("\n🔄 Flushing remaining spans...");
    tracing_manager.flush().await?;

    // Shutdown gracefully
    info!("🛑 Shutting down tracing...");
    tracing_manager.shutdown().await?;

    info!("✅ Demo completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_service_inference() {
        let config = TracingConfig::default();
        let tracing_manager = TracingManager::new(config).await.unwrap();

        let model_service = ModelService::new(tracing_manager);

        let request = InferenceRequest {
            id: "test-001".to_string(),
            model_name: "test-model".to_string(),
            input_text: "Test input".to_string(),
            max_tokens: 50,
        };

        let response = model_service.inference(request).await.unwrap();
        assert_eq!(response.request_id, "test-001");
        assert!(!response.generated_text.is_empty());
    }

    #[tokio::test]
    async fn test_trace_context_propagation() {
        let config = TracingConfig::default();
        let tracing_manager = TracingManager::new(config).await.unwrap();

        let span = tracing_manager.start_http_span("GET", "/test").unwrap();
        let context = span.get_context();

        let mut headers = HashMap::new();
        tracing_manager.inject_context(&context, &mut headers);

        assert!(headers.contains_key("traceparent"));

        let extracted_context = tracing_manager.extract_context(&headers);
        assert!(extracted_context.is_some());

        let extracted = extracted_context.unwrap();
        assert_eq!(extracted.trace_id, context.trace_id);
    }
}
