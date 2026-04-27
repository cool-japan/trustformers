//! Request Tracing for Observability
//!
//! Provides distributed tracing capabilities, including:
//! - Legacy distributed tracing via `legacy` submodule
//! - New OpenTelemetry-compatible request tracer via `request_tracer` submodule

pub mod legacy;
pub mod request_tracer;

// Re-export legacy types
pub use legacy::{
    DistributedTracer, OpenTelemetryExport, ProcessInfo, SpanGuard, SpanLog, SpanReference,
    SpanReferenceType, TraceContext, TraceExportFormat, TraceMetrics, TracingError, ZipkinSpan,
};

// Legacy types that have name conflicts are re-exported with aliases
pub use legacy::Span as LegacySpan;
pub use legacy::SpanStatus as LegacySpanStatus;
pub use legacy::TracingConfig as LegacyTracingConfig;

// Re-export request_tracer types
pub use request_tracer::{
    AttributeValue, RequestTracer, Span, SpanAttribute, SpanEvent, SpanId, SpanKind, SpanStatus,
    Trace, TraceError, TraceId, TracingConfig, TracingStats,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_id_deterministic() {
        let id1 = TraceId::new_deterministic(42);
        let id2 = TraceId::new_deterministic(42);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_trace_id_different_seeds_differ() {
        let id1 = TraceId::new_deterministic(1);
        let id2 = TraceId::new_deterministic(2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_trace_id_hex_roundtrip() {
        let original = TraceId::new_deterministic(999);
        let hex = original.to_hex();
        assert_eq!(hex.len(), 32);
        let decoded = TraceId::from_hex(&hex).unwrap_or_else(|_| TraceId::new_deterministic(0));
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_trace_id_invalid_hex_length_error() {
        let result = TraceId::from_hex("abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_span_id_deterministic() {
        let id1 = SpanId::new_deterministic(100);
        let id2 = SpanId::new_deterministic(100);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_span_id_hex_roundtrip() {
        let original = SpanId::new_deterministic(77);
        let hex = original.to_hex();
        assert_eq!(hex.len(), 16);
        let decoded = SpanId::from_hex(&hex).unwrap_or_else(|_| SpanId::new_deterministic(0));
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_span_status_is_error() {
        let err = SpanStatus::Error {
            message: "boom".to_string(),
        };
        assert!(err.is_error());
        assert!(!SpanStatus::Ok.is_error());
        assert!(!SpanStatus::Unset.is_error());
    }

    #[test]
    fn test_span_attribute_new() {
        let attr = SpanAttribute::new("http.method", AttributeValue::String("GET".to_string()));
        assert_eq!(attr.key, "http.method");
        assert_eq!(attr.value, AttributeValue::String("GET".to_string()));
    }

    #[test]
    fn test_span_event_new() {
        let event = SpanEvent::new("cache_hit", vec![]);
        assert_eq!(event.name, "cache_hit");
        assert!(event.attributes.is_empty());
    }

    #[test]
    fn test_span_lifecycle() {
        let trace_id = TraceId::new_deterministic(1);
        let mut span = Span::new("test_op".to_string(), SpanKind::Internal, trace_id, None);
        assert!(span.end_time.is_none());
        assert!(!span.is_error());
        span.end();
        assert!(span.end_time.is_some());
        assert!(span.duration_ms().is_some());
    }

    #[test]
    fn test_span_add_attribute_and_event() {
        let trace_id = TraceId::new_deterministic(2);
        let mut span = Span::new("db_query".to_string(), SpanKind::Client, trace_id, None);
        span.add_attribute("db.table", AttributeValue::String("users".to_string()));
        span.add_event("row_fetched", vec![]);
        assert_eq!(span.attributes.len(), 1);
        assert_eq!(span.events.len(), 1);
    }

    #[test]
    fn test_trace_add_span_and_root() {
        let trace_id = TraceId::new_deterministic(3);
        let mut trace = Trace::new(trace_id.clone());
        let root = Span::new("root".to_string(), SpanKind::Server, trace_id.clone(), None);
        trace.add_span(root);
        assert!(trace.root_span().is_some());
        assert_eq!(trace.spans.len(), 1);
    }

    #[test]
    fn test_trace_error_spans() {
        let trace_id = TraceId::new_deterministic(4);
        let mut trace = Trace::new(trace_id.clone());
        let mut span = Span::new("failing".to_string(), SpanKind::Internal, trace_id, None);
        span.set_status(SpanStatus::Error {
            message: "fail".to_string(),
        });
        trace.add_span(span);
        assert_eq!(trace.error_spans().len(), 1);
    }

    #[test]
    fn test_trace_to_json_contains_trace_id() {
        let trace_id = TraceId::new_deterministic(5);
        let hex = trace_id.to_hex();
        let mut trace = Trace::new(trace_id.clone());
        let span = Span::new("op".to_string(), SpanKind::Server, trace_id, None);
        trace.add_span(span);
        let json = trace.to_json();
        assert!(json.contains(&hex));
    }

    #[test]
    fn test_request_tracer_start_and_end_trace() {
        let config = TracingConfig::default();
        let mut tracer = RequestTracer::new(config);
        let trace_id = tracer.start_trace("req-1".to_string(), "handle_request".to_string());
        assert_eq!(tracer.stats.total_traces, 1);
        let completed = tracer.end_trace("req-1");
        assert!(completed.is_some());
        if let Some(t) = completed {
            assert_eq!(t.trace_id, trace_id);
        }
    }

    #[test]
    fn test_request_tracer_start_span() {
        let config = TracingConfig::default();
        let mut tracer = RequestTracer::new(config);
        tracer.start_trace("req-2".to_string(), "root_op".to_string());
        let span_id = tracer.start_span("req-2", "child_op".to_string(), SpanKind::Internal);
        assert!(span_id.is_some());
        assert_eq!(tracer.stats.total_spans, 1);
    }

    #[test]
    fn test_request_tracer_end_span_with_error() {
        let config = TracingConfig::default();
        let mut tracer = RequestTracer::new(config);
        tracer.start_trace("req-3".to_string(), "root".to_string());
        let span_id_opt = tracer.start_span("req-3", "failing_child".to_string(), SpanKind::Client);
        if let Some(span_id) = span_id_opt {
            tracer.end_span(
                "req-3",
                &span_id,
                SpanStatus::Error {
                    message: "timeout".to_string(),
                },
            );
        }
        let completed = tracer.end_trace("req-3");
        assert!(completed.is_some());
        if let Some(trace) = completed {
            let error_spans = trace.error_spans();
            assert!(!error_spans.is_empty());
        }
    }

    #[test]
    fn test_request_tracer_recent_traces() {
        let config = TracingConfig::default();
        let mut tracer = RequestTracer::new(config);
        tracer.start_trace("req-a".to_string(), "op_a".to_string());
        tracer.end_trace("req-a");
        let recent = tracer.recent_traces();
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_tracing_stats_error_rate() {
        let mut stats = TracingStats::default();
        assert_eq!(stats.error_rate(), 0.0);
        stats.total_traces = 4;
        stats.error_traces = 2;
        assert!((stats.error_rate() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_span_kind_display() {
        assert_eq!(SpanKind::Client.to_string(), "CLIENT");
        assert_eq!(SpanKind::Server.to_string(), "SERVER");
        assert_eq!(SpanKind::Internal.to_string(), "INTERNAL");
    }

    #[test]
    fn test_attribute_value_variants() {
        let s = AttributeValue::String("hello".to_string());
        let i = AttributeValue::Int(42);
        let f = AttributeValue::Float(3.14);
        let b = AttributeValue::Bool(true);
        assert_eq!(s, AttributeValue::String("hello".to_string()));
        assert_eq!(i, AttributeValue::Int(42));
        assert_eq!(f, AttributeValue::Float(3.14));
        assert_eq!(b, AttributeValue::Bool(true));
    }

    #[test]
    fn test_tracing_config_defaults() {
        let config = TracingConfig::default();
        assert_eq!(config.service_name, "trustformers-serve");
        assert_eq!(config.sample_rate, 1.0);
        assert_eq!(config.max_spans_per_trace, 1000);
        assert!(config.export_endpoint.is_none());
    }
}
