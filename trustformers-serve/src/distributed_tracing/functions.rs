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
    use crate::{
        BatchConfig, DistributedSpan, SamplingStrategy, SpanEvent, TraceContext, TracingBackend,
        TracingConfig, TracingEvent,
    };
    use std::collections::HashMap;
    use std::time::Duration;

    /// A span with just enough shape to exercise the format converters:
    /// callers set only the fields the assertion under test cares about.
    fn make_span(
        parent_span_id: Option<&str>,
        attributes: Vec<(&str, &str)>,
        events: Vec<SpanEvent>,
    ) -> DistributedSpan {
        DistributedSpan {
            span_id: "span1".to_string(),
            trace_id: "trace1".to_string(),
            parent_span_id: parent_span_id.map(|s| s.to_string()),
            operation_name: "test_op".to_string(),
            kind: SpanKind::Server,
            start_time: chrono::Utc::now(),
            end_time: Some(chrono::Utc::now()),
            status: SpanStatus::Ok,
            attributes: attributes
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            events,
            service_name: "trustformers-serve-test".to_string(),
            resource_attributes: HashMap::new(),
        }
    }

    /// Regression: Jaeger's real `model.KeyValue` tag is `{key, type, value}`;
    /// `tags`/`process.tags` used to serialize `span.attributes` (a
    /// `HashMap<String, String>`) directly as a flat JSON object instead,
    /// which a Jaeger-format reader could not parse as tags at all.
    #[tokio::test]
    async fn test_convert_to_jaeger_format_tags_have_key_type_value_shape() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config)
            .await
            .expect("TracingManager creation should succeed");
        let span = make_span(None, vec![("http.method", "GET")], Vec::new());
        let jaeger_spans =
            manager.convert_to_jaeger_format(vec![span]).expect("conversion should succeed");
        let tags = jaeger_spans[0]["tags"].as_array().expect("tags should be a JSON array");
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0]["key"], "http.method");
        assert_eq!(tags[0]["type"], "string");
        assert_eq!(tags[0]["value"], "GET");
    }

    /// Regression: `parentSpanID` used to default to `""` for a root span
    /// (`unwrap_or_default()`); a Jaeger reader has no way to distinguish
    /// that from a span whose parent really is the empty string.
    #[tokio::test]
    async fn test_convert_to_jaeger_format_root_span_parent_is_null() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config)
            .await
            .expect("TracingManager creation should succeed");
        let span = make_span(None, Vec::new(), Vec::new());
        let jaeger_spans =
            manager.convert_to_jaeger_format(vec![span]).expect("conversion should succeed");
        assert!(jaeger_spans[0]["parentSpanID"].is_null());
    }

    /// Regression: this used to nest spans under the pre-1.0 OTLP field names
    /// `instrumentationLibrarySpans`/`instrumentationLibrary`, which a current
    /// (stable OTLP) collector does not recognize, and never included
    /// `service.name` -- the one resource attribute OTLP semantic conventions
    /// require to identify the emitting service.
    #[tokio::test]
    async fn test_convert_to_otlp_format_uses_stable_field_names_and_service_name() {
        let config = TracingConfig::default().with_service_name("my-service");
        let manager = TracingManager::new(config)
            .await
            .expect("TracingManager creation should succeed");
        let span = make_span(None, Vec::new(), Vec::new());
        let otlp = manager.convert_to_otlp_format(vec![span]).expect("conversion should succeed");
        let resource_span = &otlp["resourceSpans"][0];
        assert!(
            resource_span.get("instrumentationLibrarySpans").is_none(),
            "must not use the pre-1.0 OTLP field name"
        );
        let scope_spans =
            resource_span["scopeSpans"].as_array().expect("scopeSpans should be present");
        assert_eq!(scope_spans.len(), 1);
        assert!(scope_spans[0].get("instrumentationLibrary").is_none());
        assert_eq!(scope_spans[0]["scope"]["name"], "trustformers-serve");
        let resource_attributes = resource_span["resource"]["attributes"]
            .as_array()
            .expect("resource attributes should be an array");
        let service_name_attr = resource_attributes
            .iter()
            .find(|attr| attr["key"] == "service.name")
            .expect("resource attributes must include service.name");
        assert_eq!(service_name_attr["value"]["stringValue"], "my-service");
    }

    /// Regression: OTLP/JSON represents protobuf `fixed64` fields (which
    /// `startTimeUnixNano`/`endTimeUnixNano` are) as JSON strings precisely so
    /// a 64-bit nanosecond timestamp survives a round-trip through a language
    /// whose numbers are `f64`-precision; these used to be emitted as JSON
    /// numbers.
    #[tokio::test]
    async fn test_convert_to_otlp_format_nano_timestamps_are_strings() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config)
            .await
            .expect("TracingManager creation should succeed");
        let span = make_span(None, Vec::new(), Vec::new());
        let otlp = manager.convert_to_otlp_format(vec![span]).expect("conversion should succeed");
        let converted_span = &otlp["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        assert!(
            converted_span["startTimeUnixNano"].is_string(),
            "startTimeUnixNano must be a JSON string, got {:?}",
            converted_span["startTimeUnixNano"]
        );
        assert!(converted_span["endTimeUnixNano"].is_string());
    }

    /// Regression: `parentSpanId` used to default to `""` for a root span,
    /// same defect as the Jaeger converter above.
    #[tokio::test]
    async fn test_convert_to_otlp_format_root_span_parent_is_null() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config)
            .await
            .expect("TracingManager creation should succeed");
        let span = make_span(None, Vec::new(), Vec::new());
        let otlp = manager.convert_to_otlp_format(vec![span]).expect("conversion should succeed");
        let converted_span = &otlp["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        assert!(converted_span["parentSpanId"].is_null());
    }

    /// Regression: `SamplingStrategy::Adaptive` used to read `let
    /// current_load = 0.5;` -- a hardcoded constant, so the branch was in
    /// fact always exactly `min_rate` or always exactly `max_rate` for any
    /// given config, never actually reading system load. The real reading is
    /// cached (see `CpuLoadMonitor`); calling it twice back to back, well
    /// inside the refresh interval, must return the exact same value rather
    /// than two independent (and here, indistinguishable-from-fabricated)
    /// numbers -- this is the one property of "a real, cached system
    /// reading" a hermetic test can pin down without asserting anything
    /// about the host's actual CPU load.
    #[tokio::test]
    async fn test_current_cpu_load_fraction_is_cached_and_bounded() {
        let config = TracingConfig::default();
        let manager = TracingManager::new(config)
            .await
            .expect("TracingManager creation should succeed");
        let first = manager.current_cpu_load_fraction();
        let second = manager.current_cpu_load_fraction();
        assert!(first.is_finite() && first >= 0.0, "got {first}");
        assert_eq!(
            first, second,
            "back-to-back reads within the cache window must agree"
        );
    }

    /// Regression: a failed export used to clear the queue *before*
    /// attempting the export, so a failure lost the spans for good. Points
    /// at a syntactically invalid endpoint so `reqwest` fails the request at
    /// URL-parse time -- no real network I/O, so this cannot hang or flake on
    /// an unreachable host.
    ///
    /// `export_loop`'s background task ticks immediately on creation, so this
    /// deliberately: (1) creates the manager and lets that first, immediate
    /// tick fire on an empty queue before anything is queued, and (2) sets
    /// `max_batch_timeout` far beyond this test's lifetime so no *second*
    /// tick can race the explicit `flush()` call below -- `flush()` is then
    /// deterministically the only thing draining the queue.
    #[tokio::test]
    async fn test_flush_requeues_spans_on_export_failure() {
        let config = TracingConfig {
            backend: TracingBackend::Jaeger {
                endpoint: "not a valid url".to_string(),
                username: None,
                password: None,
            },
            batch_config: BatchConfig {
                max_batch_timeout: Duration::from_secs(3600),
                ..BatchConfig::default()
            },
            ..TracingConfig::default()
        };
        let manager = TracingManager::new(config)
            .await
            .expect("TracingManager creation should succeed");
        tokio::time::sleep(Duration::from_millis(200)).await;
        let span = manager.start_span("test_operation").expect("Start span should succeed");
        span.finish();
        let result = manager.flush().await;
        assert!(
            result.is_err(),
            "flushing to a broken endpoint must report the failure"
        );
        let stats = manager.get_stats();
        assert_eq!(
            stats.queue_size, 1,
            "the span must be requeued, not lost, on a failed flush"
        );
        assert_eq!(
            stats.spans_exported, 0,
            "a permanently failing endpoint exports nothing"
        );
    }
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
