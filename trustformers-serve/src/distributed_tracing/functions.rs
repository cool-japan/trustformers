//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

use super::types::{SpanBuilder, SpanKind, SpanStatus, TracingManager};

/// Create distributed tracing middleware for HTTP requests
pub fn create_tracing_middleware(
    manager: Arc<TracingManager>,
) -> impl tower::Layer<axum::routing::Router> {
    tower_http::trace::TraceLayer::new_for_http().make_span_with(
        move |request: &axum::http::Request<axum::body::Body>| {
            let method = request.method().to_string();
            let uri = request.uri().to_string();
            let headers: HashMap<String, String> = request
                .headers()
                .iter()
                .filter_map(|(name, value)| {
                    value.to_str().ok().map(|v| (name.to_string(), v.to_string()))
                })
                .collect();
            let context = manager.extract_context(&headers);
            let span_result = if let Some(context) = context {
                SpanBuilder::new(format!("{} {}", method, uri))
                    .with_kind(SpanKind::Server)
                    .with_parent_context(context)
                    .with_attribute("http.method", method.clone())
                    .with_attribute("http.uri", uri.clone())
                    .start(&manager)
            } else {
                manager.start_http_span(method, uri)
            };
            match span_result {
                Ok(span) => {
                    tracing::info_span!(
                        "http_request", span_id = % span.get_context().span_id
                    )
                },
                Err(e) => {
                    tracing::error!("Failed to create tracing span: {}", e);
                    tracing::info_span!("http_request_fallback")
                },
            }
        },
    )
}
/// Trace an async function
pub async fn trace_async<F, T>(
    manager: &TracingManager,
    operation_name: &str,
    attributes: Vec<(&str, &str)>,
    f: F,
) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    let span = manager.start_span(operation_name)?;
    for (key, value) in attributes {
        span.set_attribute(key, value);
    }
    let result = f.await;
    match &result {
        Ok(_) => span.set_status(SpanStatus::Ok),
        Err(e) => span.set_error(e.to_string()),
    }
    span.finish();
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SamplingStrategy, TraceContext, TracingConfig, TracingEvent};
    use std::collections::HashMap;
    use std::time::Duration;
    #[tokio::test]
    async fn test_tracing_manager_creation() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config)
            .await
            .expect("TracingManager creation should succeed");
        let stats = manager.get_stats();
        assert_eq!(stats.spans_created, 0);
        assert_eq!(stats.spans_exported, 0);
    }
    #[tokio::test]
    async fn test_span_creation_and_attributes() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config)
            .await
            .expect("TracingManager creation should succeed");
        let span = manager.start_span("test_operation").expect("Start span should succeed");
        span.set_attribute("test.key", "test.value");
        span.add_event("test_event", vec![("event.key", "event.value")]);
        span.finish();
        let stats = manager.get_stats();
        assert_eq!(stats.spans_created, 1);
    }
    #[tokio::test]
    async fn test_trace_context_propagation() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config)
            .await
            .expect("TracingManager creation should succeed");
        let parent_span = manager
            .start_span("parent_operation")
            .expect("Start parent span should succeed");
        let parent_context = parent_span.get_context();
        let child_span = SpanBuilder::new("child_operation")
            .with_parent_context(parent_context.clone())
            .start(&manager)
            .expect("Start child span should succeed");
        let child_context = child_span.get_context();
        assert_eq!(child_context.trace_id, parent_context.trace_id);
        assert_ne!(child_context.span_id, parent_context.span_id);
        child_span.finish();
        parent_span.finish();
    }
    #[tokio::test]
    async fn test_sampling_strategies() {
        let config = TracingConfig::default().with_sampling(SamplingStrategy::Probabilistic(0.5));
        let manager = TracingManager::new(config)
            .await
            .expect("TracingManager creation should succeed");
        for i in 0..100 {
            if let Ok(span) = manager.start_span(format!("test_span_{}", i)) {
                span.finish();
            }
        }
        let stats = manager.get_stats();
        assert!(stats.spans_created <= 100);
    }
    #[tokio::test]
    async fn test_inference_span() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config)
            .await
            .expect("TracingManager creation should succeed");
        let span = manager
            .start_inference_span("llama-7b", "req-123")
            .expect("Start inference span should succeed");
        let span_guard = span.span.lock();
        assert_eq!(span_guard.operation_name, "inference");
        assert!(span_guard.attributes.contains_key("model.name"));
        assert!(span_guard.attributes.contains_key("request.id"));
        drop(span_guard);
        span.finish();
    }
    #[tokio::test]
    async fn test_context_extraction() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config)
            .await
            .expect("TracingManager creation should succeed");
        let mut headers = HashMap::new();
        headers.insert(
            "traceparent".to_string(),
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01".to_string(),
        );
        let context = manager.extract_context(&headers).expect("Extract context should succeed");
        assert_eq!(context.trace_id, "0af7651916cd43dd8448eb211c80319c");
        assert_eq!(context.span_id, "b7ad6b7169203331");
        assert_eq!(context.flags, 1);
    }
    #[tokio::test]
    async fn test_context_injection() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config)
            .await
            .expect("TracingManager creation should succeed");
        let context = TraceContext {
            trace_id: "0af7651916cd43dd8448eb211c80319c".to_string(),
            span_id: "b7ad6b7169203331".to_string(),
            flags: 1,
            state: HashMap::new(),
            baggage: HashMap::new(),
        };
        let mut headers = HashMap::new();
        manager.inject_context(&context, &mut headers);
        assert!(headers.contains_key("traceparent"));
        assert_eq!(
            headers["traceparent"],
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        );
    }
    #[tokio::test]
    async fn test_event_subscription() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config)
            .await
            .expect("TracingManager creation should succeed");
        let mut event_receiver = manager.subscribe_events();
        let span = manager.start_span("test_operation").expect("Start span should succeed");
        if let Ok(event) =
            tokio::time::timeout(Duration::from_millis(100), event_receiver.recv()).await
        {
            match event.expect("Event should be received") {
                TracingEvent::SpanStarted { span_id, trace_id } => {
                    assert!(!span_id.is_empty());
                    assert!(!trace_id.is_empty());
                },
                other => {
                    assert_eq!(
                        std::mem::discriminant(&other),
                        std::mem::discriminant(&TracingEvent::SpanStarted {
                            span_id: String::new(),
                            trace_id: String::new()
                        }),
                        "Expected SpanStarted event, got: {:?}",
                        other
                    );
                },
            }
        }
        span.finish();
    }
    #[tokio::test]
    async fn test_flush_and_shutdown() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config)
            .await
            .expect("TracingManager creation should succeed");
        let span = manager.start_span("test_operation").expect("Start span should succeed");
        span.finish();
        manager.flush().await.expect("Flush should succeed");
        manager.shutdown().await.expect("Shutdown should succeed");
    }
}
