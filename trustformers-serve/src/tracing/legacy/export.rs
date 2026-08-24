//! Trace export wire formats: Jaeger, Zipkin v2 and OTLP/JSON.
//!
//! Each `TraceExportFormat` variant emits a document that genuinely conforms to
//! the format it is named after, so the bytes `DistributedTracer::export_traces`
//! returns can be POSTed to that collector, or loaded by that UI, unchanged.
//!
//! NOTE (2026-08-24): before this revision the names promised wire formats the
//! bytes did not implement. `TraceExportFormat::Jaeger` returned
//! `serde_json::to_string_pretty(&completed_spans)` — this module's own `Span`
//! struct verbatim; `OpenTelemetryExport` wrapped that same internal shape in a
//! `{"format": "opentelemetry", "version": "1.0"}` metadata map; and `ZipkinSpan`
//! serialised `trace_id`/`parent_id` where Zipkin v2 requires `traceId`/`parentId`,
//! with no `localEndpoint` at all. No collector of the three would have accepted
//! its own document.
//!
//! This is a plain `serde_json` implementation and deliberately depends on no
//! OpenTelemetry SDK crate: `opentelemetry`, `opentelemetry-otlp`,
//! `opentelemetry-jaeger` (the root of RUSTSEC-2025-0123, deprecated upstream)
//! and `opentelemetry-zipkin` were removed from the workspace on 2026-08-24
//! because nothing here ever called them. See `trustformers-serve/Cargo.toml`.

use super::{Span, SpanLog, SpanReferenceType, SpanStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Width, in hex digits, of a trace id on the wire.
///
/// W3C Trace Context, OTLP and Jaeger all use a 128-bit trace id; Zipkin accepts
/// either 128-bit or 64-bit.
const TRACE_ID_HEX_WIDTH: usize = 32;

/// Width, in hex digits, of a span id on the wire (64 bits everywhere).
const SPAN_ID_HEX_WIDTH: usize = 16;

/// FNV-1a, 64-bit. Used only to derive wire ids deterministically from id
/// strings that carry too little hex material to be used directly (see
/// [`normalize_hex_id`]); never used as a security primitive.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Map an internal id string onto the fixed-width lowercase hex id every trace
/// wire format requires.
///
/// This module mints ids with `Uuid::new_v4().to_string()`, so the common cases
/// are exact: a UUID has 32 hex digits, which is precisely a trace id, and its
/// leading 16 (`time_low` + `time_mid`) are 64 random bits, precisely a span id.
/// Ids that arrive from `TraceContext::from_headers` are arbitrary strings, so
/// anything with too few hex digits is extended with FNV-1a over the original
/// string. That keeps the mapping a deterministic function of its input, which
/// is what parent/child links depend on: the same id always yields the same wire
/// id, so a child's `parentSpanId` still matches its parent's `spanId`.
///
/// An all-zero result is replaced, because W3C Trace Context, OTLP and Jaeger
/// all treat an all-zero id as invalid rather than as a value.
fn normalize_hex_id(raw: &str, width: usize) -> String {
    let mut hex: String = raw
        .chars()
        .filter(char::is_ascii_hexdigit)
        .map(|c| c.to_ascii_lowercase())
        .take(width)
        .collect();

    if hex.len() < width {
        let mut seed = fnv1a64(raw.as_bytes());
        while hex.len() < width {
            hex.push_str(&format!("{seed:016x}"));
            seed = fnv1a64(&seed.to_be_bytes());
        }
        hex.truncate(width);
    }

    if hex.bytes().all(|byte| byte == b'0') {
        hex = format!("{:0width$x}", fnv1a64(raw.as_bytes()) | 1, width = width);
    }

    hex
}

/// Render a span's `logs` fields in a stable order, so exported documents are
/// byte-reproducible for the same span (the source is a `HashMap`).
fn sorted_pairs(fields: &HashMap<String, String>) -> Vec<(&String, &String)> {
    let mut pairs: Vec<(&String, &String)> = fields.iter().collect();
    pairs.sort_by(|left, right| left.0.cmp(right.0));
    pairs
}

/// Nanoseconds since the Unix epoch, falling back to microsecond precision for
/// timestamps outside `timestamp_nanos_opt`'s representable range (before 1677
/// or after 2262) instead of silently reporting the epoch.
fn unix_nanos(timestamp: chrono::DateTime<chrono::Utc>) -> i64 {
    timestamp
        .timestamp_nanos_opt()
        .unwrap_or_else(|| timestamp.timestamp_micros().saturating_mul(1_000))
}

/// A Jaeger `model.KeyValue`: a tag carries its value type alongside its value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerTag {
    pub key: String,

    #[serde(rename = "type")]
    pub value_type: String,

    pub value: serde_json::Value,
}

impl JaegerTag {
    fn text(key: &str, value: &str) -> Self {
        Self {
            key: key.to_string(),
            value_type: "string".to_string(),
            value: serde_json::Value::String(value.to_string()),
        }
    }

    fn boolean(key: &str, value: bool) -> Self {
        Self {
            key: key.to_string(),
            value_type: "bool".to_string(),
            value: serde_json::Value::Bool(value),
        }
    }
}

/// A Jaeger `model.Log`: a timestamped bag of key/value fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerLog {
    /// Microseconds since the Unix epoch.
    pub timestamp: i64,

    pub fields: Vec<JaegerTag>,
}

/// A Jaeger `model.SpanRef`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerReference {
    #[serde(rename = "refType")]
    pub ref_type: String,

    #[serde(rename = "traceID")]
    pub trace_id: String,

    #[serde(rename = "spanID")]
    pub span_id: String,
}

/// A Jaeger `model.Process`: the service that emitted a set of spans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerProcess {
    #[serde(rename = "serviceName")]
    pub service_name: String,

    pub tags: Vec<JaegerTag>,
}

/// A Jaeger `model.Span`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerSpan {
    #[serde(rename = "traceID")]
    pub trace_id: String,

    #[serde(rename = "spanID")]
    pub span_id: String,

    #[serde(rename = "operationName")]
    pub operation_name: String,

    pub references: Vec<JaegerReference>,

    /// Microseconds since the Unix epoch.
    #[serde(rename = "startTime")]
    pub start_time: i64,

    /// Microseconds. Jaeger's model makes this mandatory, so an unfinished span
    /// reports `0` *and* carries a `span.unfinished` tag saying so — a reader
    /// must never mistake the placeholder for a measured duration.
    pub duration: u64,

    pub tags: Vec<JaegerTag>,

    pub logs: Vec<JaegerLog>,

    #[serde(rename = "processID")]
    pub process_id: String,
}

impl JaegerSpan {
    fn from_span(span: &Span, trace_id: &str, process_id: String) -> Self {
        let mut references: Vec<JaegerReference> = span
            .references
            .iter()
            .map(|reference| JaegerReference {
                ref_type: match reference.ref_type {
                    SpanReferenceType::ChildOf => "CHILD_OF",
                    SpanReferenceType::FollowsFrom => "FOLLOWS_FROM",
                }
                .to_string(),
                trace_id: normalize_hex_id(&reference.trace_id, TRACE_ID_HEX_WIDTH),
                span_id: normalize_hex_id(&reference.span_id, SPAN_ID_HEX_WIDTH),
            })
            .collect();

        if let Some(parent_span_id) = &span.parent_span_id {
            let parent = normalize_hex_id(parent_span_id, SPAN_ID_HEX_WIDTH);
            if !references.iter().any(|reference| reference.span_id == parent) {
                references.push(JaegerReference {
                    ref_type: "CHILD_OF".to_string(),
                    trace_id: trace_id.to_string(),
                    span_id: parent,
                });
            }
        }

        let mut tags: Vec<JaegerTag> = sorted_pairs(&span.tags)
            .into_iter()
            .map(|(key, value)| JaegerTag::text(key, value))
            .collect();

        if !matches!(span.status, SpanStatus::Ok) {
            tags.push(JaegerTag::boolean("error", true));
            tags.push(JaegerTag::text("otel.status_code", "ERROR"));
            tags.push(JaegerTag::text(
                "otel.status_description",
                &format!("{:?}", span.status),
            ));
        }

        if span.end_time.is_none() {
            tags.push(JaegerTag::boolean("span.unfinished", true));
        }

        Self {
            trace_id: trace_id.to_string(),
            span_id: normalize_hex_id(&span.span_id, SPAN_ID_HEX_WIDTH),
            operation_name: span.operation_name.clone(),
            references,
            start_time: span.start_time.timestamp_micros(),
            duration: span.duration_us.unwrap_or(0),
            tags,
            logs: span
                .logs
                .iter()
                .map(|log| JaegerLog {
                    timestamp: log.timestamp.timestamp_micros(),
                    fields: sorted_pairs(&log.fields)
                        .into_iter()
                        .map(|(key, value)| JaegerTag::text(key, value))
                        .collect(),
                })
                .collect(),
            process_id,
        }
    }
}

/// One trace: its spans plus the process registry those spans reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerTrace {
    #[serde(rename = "traceID")]
    pub trace_id: String,

    pub spans: Vec<JaegerSpan>,

    pub processes: HashMap<String, JaegerProcess>,
}

/// The document `jaeger-query` serves from `/api/traces`, which is also exactly
/// what the Jaeger UI accepts from its "Load JSON File" control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerExport {
    pub data: Vec<JaegerTrace>,
    pub total: usize,
    pub limit: usize,
    pub offset: usize,
    pub errors: Option<serde_json::Value>,
}

impl JaegerExport {
    /// Group spans into traces, interning one `process` entry per service.
    pub fn from_spans(spans: &[Span]) -> Self {
        let mut data: Vec<JaegerTrace> = Vec::new();
        let mut trace_slots: HashMap<String, usize> = HashMap::new();

        for span in spans {
            let trace_id = normalize_hex_id(&span.trace_id, TRACE_ID_HEX_WIDTH);
            let slot = match trace_slots.get(&trace_id) {
                Some(index) => *index,
                None => {
                    data.push(JaegerTrace {
                        trace_id: trace_id.clone(),
                        spans: Vec::new(),
                        processes: HashMap::new(),
                    });
                    let index = data.len() - 1;
                    trace_slots.insert(trace_id.clone(), index);
                    index
                },
            };

            let trace = &mut data[slot];
            let process_id = Self::intern_process(trace, span);
            trace.spans.push(JaegerSpan::from_span(span, &trace_id, process_id));
        }

        let total = data.len();
        Self {
            data,
            total,
            limit: 0,
            offset: 0,
            errors: None,
        }
    }

    /// Return the `processID` for this span's service, registering it first if
    /// this is the service's first span in the trace.
    fn intern_process(trace: &mut JaegerTrace, span: &Span) -> String {
        if let Some((process_id, _)) = trace
            .processes
            .iter()
            .find(|(_, process)| process.service_name == span.service_name)
        {
            return process_id.clone();
        }

        let process_id = format!("p{}", trace.processes.len() + 1);
        let mut tags: Vec<JaegerTag> = sorted_pairs(&span.process.tags)
            .into_iter()
            .map(|(key, value)| JaegerTag::text(key, value))
            .collect();

        // `Span::new` fills `process` from `ProcessInfo::default()`, so its
        // service name can differ from the span's own. Keep both rather than
        // silently dropping one.
        if span.process.service_name != span.service_name {
            tags.push(JaegerTag::text(
                "process.service_name",
                &span.process.service_name,
            ));
        }

        trace.processes.insert(
            process_id.clone(),
            JaegerProcess {
                service_name: span.service_name.clone(),
                tags,
            },
        );
        process_id
    }
}

/// The `localEndpoint` of a Zipkin v2 span.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZipkinEndpoint {
    #[serde(rename = "serviceName")]
    pub service_name: String,
}

/// A Zipkin v2 annotation: a timestamped string, not a structured bag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZipkinAnnotation {
    /// Microseconds since the Unix epoch.
    pub timestamp: i64,

    pub value: String,
}

impl ZipkinAnnotation {
    fn from_log(log: &SpanLog) -> Self {
        let rendered: Vec<String> = sorted_pairs(&log.fields)
            .into_iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect();

        Self {
            timestamp: log.timestamp.timestamp_micros(),
            value: if rendered.is_empty() { "log".to_string() } else { rendered.join(" ") },
        }
    }
}

/// A span in Zipkin's v2 JSON encoding, the body `POST /api/v2/spans` accepts.
///
/// The wire names are camelCase (`traceId`, `parentId`, `localEndpoint`) and the
/// ids are fixed-width lowercase hex; both are enforced by the `serde` renames
/// and [`normalize_hex_id`] rather than left to the caller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZipkinSpan {
    #[serde(rename = "traceId")]
    pub trace_id: String,

    #[serde(rename = "parentId", skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,

    pub id: String,

    pub name: String,

    /// Start time, microseconds since the Unix epoch.
    pub timestamp: i64,

    /// Duration in microseconds; omitted entirely while the span is unfinished,
    /// which Zipkin reads as "still open" rather than as a zero-length span.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<u64>,

    #[serde(rename = "localEndpoint")]
    pub local_endpoint: ZipkinEndpoint,

    pub tags: HashMap<String, String>,

    pub annotations: Vec<ZipkinAnnotation>,
}

impl ZipkinSpan {
    pub fn from_span(span: &Span) -> Self {
        let mut tags = span.tags.clone();
        if !matches!(span.status, SpanStatus::Ok) {
            tags.insert("error".to_string(), format!("{:?}", span.status));
        }

        Self {
            trace_id: normalize_hex_id(&span.trace_id, TRACE_ID_HEX_WIDTH),
            parent_id: span
                .parent_span_id
                .as_deref()
                .map(|parent| normalize_hex_id(parent, SPAN_ID_HEX_WIDTH)),
            id: normalize_hex_id(&span.span_id, SPAN_ID_HEX_WIDTH),
            name: span.operation_name.clone(),
            timestamp: span.start_time.timestamp_micros(),
            duration: span.duration_us,
            local_endpoint: ZipkinEndpoint {
                service_name: span.service_name.clone(),
            },
            tags,
            annotations: span.logs.iter().map(ZipkinAnnotation::from_log).collect(),
        }
    }
}

/// An OTLP `AnyValue`. Every attribute this module produces is a string, so
/// only `stringValue` is modelled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpAnyValue {
    #[serde(rename = "stringValue")]
    pub string_value: String,
}

/// An OTLP `KeyValue`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpKeyValue {
    pub key: String,
    pub value: OtlpAnyValue,
}

impl OtlpKeyValue {
    fn text(key: &str, value: &str) -> Self {
        Self {
            key: key.to_string(),
            value: OtlpAnyValue {
                string_value: value.to_string(),
            },
        }
    }
}

/// An OTLP `Resource`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpResource {
    pub attributes: Vec<OtlpKeyValue>,
}

/// An OTLP `InstrumentationScope`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpScope {
    pub name: String,
    pub version: String,
}

/// An OTLP `Span.Event`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpEvent {
    /// Nanoseconds since the Unix epoch, decimal-encoded as a string per the
    /// proto3 JSON mapping for 64-bit integers.
    #[serde(rename = "timeUnixNano")]
    pub time_unix_nano: String,

    pub name: String,

    pub attributes: Vec<OtlpKeyValue>,
}

/// An OTLP `Span.Link`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpLink {
    #[serde(rename = "traceId")]
    pub trace_id: String,

    #[serde(rename = "spanId")]
    pub span_id: String,
}

/// An OTLP `Status`. `code` follows `StatusCode`: 0 unset, 1 ok, 2 error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpStatus {
    pub code: u8,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// An OTLP `Span`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpSpan {
    #[serde(rename = "traceId")]
    pub trace_id: String,

    #[serde(rename = "spanId")]
    pub span_id: String,

    #[serde(rename = "parentSpanId", skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,

    pub name: String,

    /// `SPAN_KIND_UNSPECIFIED`. This module records no span kind, so it reports
    /// the unset code rather than guessing `SERVER` or `CLIENT`.
    pub kind: u8,

    #[serde(rename = "startTimeUnixNano")]
    pub start_time_unix_nano: String,

    /// Omitted while the span is unfinished rather than back-filled with the
    /// start time, which would read as a zero-duration span.
    #[serde(rename = "endTimeUnixNano", skip_serializing_if = "Option::is_none")]
    pub end_time_unix_nano: Option<String>,

    pub attributes: Vec<OtlpKeyValue>,

    pub events: Vec<OtlpEvent>,

    pub links: Vec<OtlpLink>,

    pub status: OtlpStatus,
}

impl OtlpSpan {
    fn from_span(span: &Span) -> Self {
        let parent_span_id = span
            .parent_span_id
            .as_deref()
            .map(|parent| normalize_hex_id(parent, SPAN_ID_HEX_WIDTH));

        // OTLP models the parent edge as a field, not a link; everything else in
        // `references` becomes a link.
        let links: Vec<OtlpLink> = span
            .references
            .iter()
            .map(|reference| OtlpLink {
                trace_id: normalize_hex_id(&reference.trace_id, TRACE_ID_HEX_WIDTH),
                span_id: normalize_hex_id(&reference.span_id, SPAN_ID_HEX_WIDTH),
            })
            .filter(|link| Some(&link.span_id) != parent_span_id.as_ref())
            .collect();

        let status = if matches!(span.status, SpanStatus::Ok) {
            OtlpStatus {
                code: 1,
                message: None,
            }
        } else {
            OtlpStatus {
                code: 2,
                message: Some(format!("{:?}", span.status)),
            }
        };

        Self {
            trace_id: normalize_hex_id(&span.trace_id, TRACE_ID_HEX_WIDTH),
            span_id: normalize_hex_id(&span.span_id, SPAN_ID_HEX_WIDTH),
            parent_span_id,
            name: span.operation_name.clone(),
            kind: 0,
            start_time_unix_nano: unix_nanos(span.start_time).to_string(),
            end_time_unix_nano: span.end_time.map(|end| unix_nanos(end).to_string()),
            attributes: sorted_pairs(&span.tags)
                .into_iter()
                .map(|(key, value)| OtlpKeyValue::text(key, value))
                .collect(),
            events: span
                .logs
                .iter()
                .map(|log| OtlpEvent {
                    time_unix_nano: unix_nanos(log.timestamp).to_string(),
                    // OpenTracing's `event` field is OTel's event name; without
                    // one there is no name to report, so say "log".
                    name: log.fields.get("event").cloned().unwrap_or_else(|| "log".to_string()),
                    attributes: sorted_pairs(&log.fields)
                        .into_iter()
                        .map(|(key, value)| OtlpKeyValue::text(key, value))
                        .collect(),
                })
                .collect(),
            links,
            status,
        }
    }
}

/// An OTLP `ScopeSpans`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpScopeSpans {
    pub scope: OtlpScope,
    pub spans: Vec<OtlpSpan>,
}

/// An OTLP `ResourceSpans`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpResourceSpans {
    pub resource: OtlpResource,

    #[serde(rename = "scopeSpans")]
    pub scope_spans: Vec<OtlpScopeSpans>,
}

/// An OTLP/JSON `ExportTraceServiceRequest` — the body an OTLP/HTTP collector
/// accepts at `POST /v1/traces` with `Content-Type: application/json`.
///
/// Trace and span ids are hex-encoded (the OTLP/JSON encoding overrides proto3
/// JSON's base64 default for these two fields) and the nanosecond timestamps are
/// decimal strings (proto3 JSON's mapping for 64-bit integers).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenTelemetryExport {
    #[serde(rename = "resourceSpans")]
    pub resource_spans: Vec<OtlpResourceSpans>,
}

impl OpenTelemetryExport {
    /// Group spans by `service.name` into one `ResourceSpans` per service.
    pub fn from_spans(spans: &[Arc<Span>]) -> Self {
        let mut resource_spans: Vec<OtlpResourceSpans> = Vec::new();
        let mut resource_slots: HashMap<String, usize> = HashMap::new();

        for span in spans {
            let slot = match resource_slots.get(&span.service_name) {
                Some(index) => *index,
                None => {
                    let mut attributes =
                        vec![OtlpKeyValue::text("service.name", &span.service_name)];
                    attributes.extend(
                        sorted_pairs(&span.process.tags)
                            .into_iter()
                            .map(|(key, value)| OtlpKeyValue::text(key, value)),
                    );

                    resource_spans.push(OtlpResourceSpans {
                        resource: OtlpResource { attributes },
                        scope_spans: vec![OtlpScopeSpans {
                            scope: OtlpScope {
                                name: "trustformers-serve/tracing".to_string(),
                                version: env!("CARGO_PKG_VERSION").to_string(),
                            },
                            spans: Vec::new(),
                        }],
                    });
                    let index = resource_spans.len() - 1;
                    resource_slots.insert(span.service_name.clone(), index);
                    index
                },
            };

            if let Some(scope_spans) = resource_spans[slot].scope_spans.first_mut() {
                scope_spans.spans.push(OtlpSpan::from_span(span));
            }
        }

        Self { resource_spans }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracing::legacy::{
        DistributedTracer, TraceContext, TraceExportFormat, TracingConfig,
    };
    use std::time::Duration;

    /// A span with fully controlled ids, so the tests can assert on exact wire
    /// ids rather than on whatever `Uuid::new_v4` happened to produce.
    fn sample_span(trace_id: &str, span_id: &str, parent: Option<&str>, service: &str) -> Span {
        let context = TraceContext {
            trace_id: trace_id.to_string(),
            span_id: span_id.to_string(),
            parent_span_id: parent.map(str::to_string),
            trace_flags: 1,
            trace_state: None,
            baggage: HashMap::new(),
        };
        Span::new(&context, "generate".to_string(), service.to_string())
    }

    fn finished_span(trace_id: &str, span_id: &str, parent: Option<&str>, service: &str) -> Span {
        let mut span = sample_span(trace_id, span_id, parent, service);
        span.finish();
        span
    }

    fn is_wire_id(value: &str, width: usize) -> bool {
        value.len() == width
            && value.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    }

    const UUID_A: &str = "550e8400-e29b-41d4-a716-446655440000";
    const UUID_B: &str = "6ba7b810-9dad-11d1-80b4-00c04fd430c8";
    const UUID_C: &str = "01234567-89ab-cdef-0123-456789abcdef";

    #[test]
    fn test_normalize_hex_id_uses_a_uuid_verbatim() {
        // A UUID already carries exactly 32 hex digits; nothing is invented.
        assert_eq!(
            normalize_hex_id(UUID_A, TRACE_ID_HEX_WIDTH),
            "550e8400e29b41d4a716446655440000"
        );
        // The span id is the leading 64 bits of the same UUID.
        assert_eq!(
            normalize_hex_id(UUID_A, SPAN_ID_HEX_WIDTH),
            "550e8400e29b41d4"
        );
    }

    #[test]
    fn test_normalize_hex_id_is_deterministic_for_opaque_ids() {
        // Ids arriving through `TraceContext::from_headers` need not be hex.
        let first = normalize_hex_id("request-from-upstream", SPAN_ID_HEX_WIDTH);
        let second = normalize_hex_id("request-from-upstream", SPAN_ID_HEX_WIDTH);

        assert_eq!(
            first, second,
            "the same id must always map to the same wire id"
        );
        assert!(is_wire_id(&first, SPAN_ID_HEX_WIDTH), "got {first}");
        assert_ne!(
            normalize_hex_id("a-different-id", SPAN_ID_HEX_WIDTH),
            first,
            "distinct ids must not collapse onto one wire id"
        );
    }

    #[test]
    fn test_normalize_hex_id_never_returns_the_invalid_all_zero_id() {
        // W3C Trace Context, OTLP and Jaeger all reject an all-zero id.
        for raw in ["00000000-0000-0000-0000-000000000000", "0", ""] {
            let trace_id = normalize_hex_id(raw, TRACE_ID_HEX_WIDTH);
            assert!(is_wire_id(&trace_id, TRACE_ID_HEX_WIDTH), "got {trace_id}");
            assert!(
                trace_id.bytes().any(|byte| byte != b'0'),
                "{raw:?} produced the invalid all-zero id"
            );
        }
    }

    #[test]
    fn test_jaeger_export_is_a_jaeger_trace_document() {
        let spans = vec![finished_span(UUID_A, UUID_B, None, "inference")];
        let document = JaegerExport::from_spans(&spans);
        let json = serde_json::to_value(&document).expect("Jaeger document should serialize");

        let trace = &json["data"][0];
        assert_eq!(trace["traceID"], "550e8400e29b41d4a716446655440000");

        let span = &trace["spans"][0];
        // Jaeger's key names are `traceID`/`spanID`/`operationName`, not the
        // internal `trace_id`/`span_id`/`operation_name`.
        assert!(span.get("traceID").is_some(), "missing traceID: {span}");
        assert!(span.get("spanID").is_some(), "missing spanID: {span}");
        assert!(
            span.get("operationName").is_some(),
            "missing operationName: {span}"
        );
        assert!(
            span.get("trace_id").is_none(),
            "leaked internal field name: {span}"
        );
        assert!(
            span.get("operation_name").is_none(),
            "leaked internal field name: {span}"
        );

        // `startTime`/`duration` are microseconds, and both are numbers.
        assert!(
            span["startTime"].is_i64(),
            "startTime must be numeric: {span}"
        );
        assert!(
            span["duration"].is_u64(),
            "duration must be numeric: {span}"
        );

        // The span's process must resolve inside the trace's own registry.
        let process_id = span["processID"].as_str().expect("processID should be a string");
        assert_eq!(trace["processes"][process_id]["serviceName"], "inference");
    }

    #[test]
    fn test_jaeger_export_tags_carry_an_explicit_value_type() {
        let mut span = finished_span(UUID_A, UUID_B, None, "inference");
        span.set_tag("model".to_string(), "gpt2".to_string());

        let document = JaegerExport::from_spans(&[span]);
        let json = serde_json::to_value(&document).expect("Jaeger document should serialize");
        let tag = &json["data"][0]["spans"][0]["tags"][0];

        // Jaeger's `model.KeyValue` is {key, type, value} — not a JSON object map.
        assert_eq!(tag["key"], "model");
        assert_eq!(tag["type"], "string");
        assert_eq!(tag["value"], "gpt2");
    }

    #[test]
    fn test_jaeger_export_links_child_to_parent() {
        let spans = vec![
            finished_span(UUID_A, UUID_B, None, "inference"),
            finished_span(UUID_A, UUID_C, Some(UUID_B), "inference"),
        ];
        let document = JaegerExport::from_spans(&spans);

        assert_eq!(document.data.len(), 1, "both spans share one trace");
        let parent_span_id = document.data[0].spans[0].span_id.clone();
        let child_reference = &document.data[0].spans[1].references[0];

        assert_eq!(child_reference.ref_type, "CHILD_OF");
        assert_eq!(
            child_reference.span_id, parent_span_id,
            "the normalised child->parent edge must still resolve"
        );
        assert_eq!(child_reference.trace_id, document.data[0].trace_id);
    }

    #[test]
    fn test_jaeger_export_groups_traces_and_interns_one_process_per_service() {
        let spans = vec![
            finished_span(UUID_A, UUID_B, None, "inference"),
            finished_span(UUID_A, UUID_C, None, "inference"),
            finished_span(UUID_B, UUID_C, None, "tokenizer"),
        ];
        let document = JaegerExport::from_spans(&spans);

        assert_eq!(document.data.len(), 2, "two distinct trace ids");
        assert_eq!(document.total, 2);
        assert_eq!(document.data[0].spans.len(), 2);
        assert_eq!(
            document.data[0].processes.len(),
            1,
            "two spans of one service share one process entry"
        );
        assert_eq!(document.data[1].processes.len(), 1);
    }

    #[test]
    fn test_jaeger_export_marks_an_unfinished_span_instead_of_reporting_zero_duration() {
        // Jaeger's model makes `duration` mandatory, so the placeholder 0 must be
        // labelled rather than left to read as a measurement.
        let spans = vec![sample_span(UUID_A, UUID_B, None, "inference")];
        let document = JaegerExport::from_spans(&spans);
        let span = &document.data[0].spans[0];

        assert_eq!(span.duration, 0);
        assert!(
            span.tags.iter().any(|tag| tag.key == "span.unfinished"),
            "an unfinished span must say so: {:?}",
            span.tags
        );
    }

    #[test]
    fn test_jaeger_export_reports_a_failed_span_as_an_error() {
        let mut span = finished_span(UUID_A, UUID_B, None, "inference");
        span.set_status(SpanStatus::DeadlineExceeded);

        let document = JaegerExport::from_spans(&[span]);
        let tags = &document.data[0].spans[0].tags;

        assert!(tags.iter().any(|tag| tag.key == "error" && tag.value == true));
        assert!(tags
            .iter()
            .any(|tag| tag.key == "otel.status_description" && tag.value == "DeadlineExceeded"));
    }

    #[test]
    fn test_zipkin_export_uses_zipkin_v2_wire_names() {
        let span = finished_span(UUID_A, UUID_C, Some(UUID_B), "inference");
        let zipkin = ZipkinSpan::from_span(&span);
        let json = serde_json::to_value(&zipkin).expect("Zipkin span should serialize");

        // Zipkin v2 rejects snake_case: the keys are traceId/parentId/localEndpoint.
        assert!(json.get("traceId").is_some(), "missing traceId: {json}");
        assert!(json.get("parentId").is_some(), "missing parentId: {json}");
        assert!(
            json.get("trace_id").is_none(),
            "leaked internal field name: {json}"
        );
        assert!(
            json.get("parent_id").is_none(),
            "leaked internal field name: {json}"
        );
        assert_eq!(json["localEndpoint"]["serviceName"], "inference");
    }

    #[test]
    fn test_zipkin_export_ids_are_fixed_width_hex() {
        let span = finished_span(UUID_A, UUID_C, Some(UUID_B), "inference");
        let zipkin = ZipkinSpan::from_span(&span);

        // Zipkin validates id widths; a 36-character dashed UUID is not an id.
        assert!(
            is_wire_id(&zipkin.trace_id, TRACE_ID_HEX_WIDTH),
            "{}",
            zipkin.trace_id
        );
        assert!(is_wire_id(&zipkin.id, SPAN_ID_HEX_WIDTH), "{}", zipkin.id);
        let parent_id = zipkin.parent_id.clone().expect("parent id should be present");
        assert!(is_wire_id(&parent_id, SPAN_ID_HEX_WIDTH), "{parent_id}");
        assert_eq!(parent_id, normalize_hex_id(UUID_B, SPAN_ID_HEX_WIDTH));
    }

    #[test]
    fn test_zipkin_export_omits_duration_while_the_span_is_open() {
        let open = ZipkinSpan::from_span(&sample_span(UUID_A, UUID_B, None, "inference"));
        let json = serde_json::to_value(&open).expect("Zipkin span should serialize");
        assert!(
            json.get("duration").is_none(),
            "an open span has no duration: {json}"
        );
        assert!(
            json.get("parentId").is_none(),
            "a root span has no parent: {json}"
        );
    }

    #[test]
    fn test_zipkin_annotations_render_span_logs() {
        let mut span = finished_span(UUID_A, UUID_B, None, "inference");
        let mut fields = HashMap::new();
        fields.insert("event".to_string(), "first_token".to_string());
        fields.insert("index".to_string(), "0".to_string());
        span.add_log(SpanLog::new(fields));

        let zipkin = ZipkinSpan::from_span(&span);
        assert_eq!(zipkin.annotations.len(), 1);
        // Zipkin annotation values are plain strings, rendered in a stable order.
        assert_eq!(zipkin.annotations[0].value, "event=first_token index=0");
    }

    #[test]
    fn test_otlp_export_is_an_export_trace_service_request() {
        let spans = vec![Arc::new(finished_span(UUID_A, UUID_B, None, "inference"))];
        let export = OpenTelemetryExport::from_spans(&spans);
        let json = serde_json::to_value(&export).expect("OTLP request should serialize");

        let resource_spans = &json["resourceSpans"][0];
        assert_eq!(
            resource_spans["resource"]["attributes"][0]["key"],
            "service.name"
        );
        assert_eq!(
            resource_spans["resource"]["attributes"][0]["value"]["stringValue"],
            "inference"
        );

        let span = &resource_spans["scopeSpans"][0]["spans"][0];
        // OTLP/JSON hex-encodes the ids and decimal-string-encodes the nanos.
        let trace_id = span["traceId"].as_str().expect("traceId should be a string");
        assert!(is_wire_id(trace_id, TRACE_ID_HEX_WIDTH), "{trace_id}");
        let start = span["startTimeUnixNano"].as_str().expect("nanos are JSON strings");
        assert!(start.chars().all(|c| c.is_ascii_digit()), "{start}");

        // The old exporter emitted `{"spans": [...], "metadata": {...}}`.
        assert!(
            json.get("spans").is_none(),
            "leaked pre-OTLP envelope: {json}"
        );
        assert!(
            json.get("metadata").is_none(),
            "leaked pre-OTLP envelope: {json}"
        );
    }

    #[test]
    fn test_otlp_export_groups_spans_by_service() {
        let spans = vec![
            Arc::new(finished_span(UUID_A, UUID_B, None, "inference")),
            Arc::new(finished_span(UUID_A, UUID_C, None, "inference")),
            Arc::new(finished_span(UUID_B, UUID_C, None, "tokenizer")),
        ];
        let export = OpenTelemetryExport::from_spans(&spans);

        assert_eq!(
            export.resource_spans.len(),
            2,
            "one ResourceSpans per service"
        );
        assert_eq!(export.resource_spans[0].scope_spans[0].spans.len(), 2);
        assert_eq!(export.resource_spans[1].scope_spans[0].spans.len(), 1);
    }

    #[test]
    fn test_otlp_status_and_open_span_encoding() {
        let mut failed = finished_span(UUID_A, UUID_B, None, "inference");
        failed.set_status(SpanStatus::Internal);
        let open = sample_span(UUID_A, UUID_C, None, "inference");

        let export = OpenTelemetryExport::from_spans(&[Arc::new(failed), Arc::new(open)]);
        let spans = &export.resource_spans[0].scope_spans[0].spans;

        // StatusCode: 0 unset, 1 ok, 2 error.
        assert_eq!(spans[0].status.code, 2);
        assert_eq!(spans[0].status.message.as_deref(), Some("Internal"));
        assert!(spans[0].end_time_unix_nano.is_some());

        assert_eq!(spans[1].status.code, 1);
        assert!(
            spans[1].end_time_unix_nano.is_none(),
            "an open span must not claim an end time"
        );
    }

    #[tokio::test]
    async fn test_export_traces_emits_each_format_for_real() {
        let tracer = DistributedTracer::new(TracingConfig::default());
        let context = tracer
            .start_span("generate".to_string(), None)
            .await
            .expect("async operation should succeed in test");
        tracer
            .finish_span(&context.span_id)
            .await
            .expect("async operation should succeed in test");
        tokio::time::sleep(Duration::from_millis(100)).await;

        let jaeger = tracer
            .export_traces(TraceExportFormat::Jaeger)
            .await
            .expect("Jaeger export should succeed");
        let jaeger: serde_json::Value =
            serde_json::from_slice(&jaeger).expect("Jaeger export should be JSON");
        // The pre-2026-08-24 exporter returned a bare array of internal spans here.
        assert!(
            jaeger.is_object(),
            "Jaeger export must be a trace document: {jaeger}"
        );
        assert_eq!(jaeger["data"][0]["spans"][0]["operationName"], "generate");

        let zipkin = tracer
            .export_traces(TraceExportFormat::Zipkin)
            .await
            .expect("Zipkin export should succeed");
        let zipkin: serde_json::Value =
            serde_json::from_slice(&zipkin).expect("Zipkin export should be JSON");
        assert!(
            zipkin.is_array(),
            "Zipkin v2 posts a bare span array: {zipkin}"
        );
        assert!(zipkin[0].get("traceId").is_some(), "{zipkin}");

        let otlp = tracer
            .export_traces(TraceExportFormat::OpenTelemetry)
            .await
            .expect("OTLP export should succeed");
        let otlp: serde_json::Value =
            serde_json::from_slice(&otlp).expect("OTLP export should be JSON");
        assert!(otlp["resourceSpans"].is_array(), "{otlp}");
        assert_eq!(
            otlp["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["name"],
            "generate"
        );
    }
}
