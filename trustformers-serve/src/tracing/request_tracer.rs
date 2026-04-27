//! OpenTelemetry-compatible request tracing infrastructure.
//!
//! Implements pure-Rust distributed request tracing with TraceId/SpanId,
//! span lifecycle management, and a RequestTracer that manages active and
//! completed traces.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::SystemTime;

// ─────────────────────────────────────────────────────────────────────────────
// TraceError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during tracing operations.
#[derive(Debug)]
pub enum TraceError {
    /// The hex string had an invalid length.
    InvalidHexLength { expected: usize, got: usize },
    /// The hex string contained non-hex characters.
    InvalidHexChar(char),
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceError::InvalidHexLength { expected, got } => {
                write!(f, "invalid hex length: expected {}, got {}", expected, got)
            },
            TraceError::InvalidHexChar(c) => {
                write!(f, "invalid hex character: '{}'", c)
            },
        }
    }
}

impl std::error::Error for TraceError {}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Encode a byte slice as a lowercase hex string.
fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(char::from_digit((b >> 4) as u32, 16).unwrap_or('0'));
        out.push(char::from_digit((b & 0x0f) as u32, 16).unwrap_or('0'));
    }
    out
}

/// Decode a hex string into bytes.  Returns `Err` on length mismatch or bad chars.
fn hex_to_bytes(s: &str, buf: &mut [u8]) -> Result<(), TraceError> {
    if s.len() != buf.len() * 2 {
        return Err(TraceError::InvalidHexLength {
            expected: buf.len() * 2,
            got: s.len(),
        });
    }
    let chars: Vec<char> = s.chars().collect();
    for (i, chunk) in chars.chunks(2).enumerate() {
        let hi = chunk[0].to_digit(16).ok_or(TraceError::InvalidHexChar(chunk[0]))?;
        let lo = chunk[1].to_digit(16).ok_or(TraceError::InvalidHexChar(chunk[1]))?;
        buf[i] = ((hi << 4) | lo) as u8;
    }
    Ok(())
}

/// Linear congruential generator used to create deterministic IDs from a seed.
/// Uses Knuth's constants (64-bit Schrage method).
fn lcg_hash(seed: u64) -> [u8; 16] {
    const A: u64 = 6364136223846793005;
    const C: u64 = 1442695040888963407;

    let s0 = seed.wrapping_mul(A).wrapping_add(C);
    let s1 = s0.wrapping_mul(A).wrapping_add(C);

    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&s0.to_le_bytes());
    out[8..].copy_from_slice(&s1.to_le_bytes());
    out
}

/// Same LCG, but produces 8 bytes (for SpanId).
fn lcg_hash_8(seed: u64) -> [u8; 8] {
    const A: u64 = 6364136223846793005;
    const C: u64 = 1442695040888963407;

    let s0 = seed.wrapping_mul(A).wrapping_add(C);
    s0.to_le_bytes()
}

// ─────────────────────────────────────────────────────────────────────────────
// TraceId
// ─────────────────────────────────────────────────────────────────────────────

/// 128-bit trace identifier (OpenTelemetry-compatible).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TraceId([u8; 16]);

impl TraceId {
    /// Construct a deterministic `TraceId` by hashing `seed` with an LCG.
    pub fn new_deterministic(seed: u64) -> Self {
        TraceId(lcg_hash(seed))
    }

    /// Encode the trace ID as a 32-character lowercase hex string.
    pub fn to_hex(&self) -> String {
        bytes_to_hex(&self.0)
    }

    /// Decode a 32-character hex string into a `TraceId`.
    pub fn from_hex(s: &str) -> Result<Self, TraceError> {
        let mut buf = [0u8; 16];
        hex_to_bytes(s, &mut buf)?;
        Ok(TraceId(buf))
    }
}

impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpanId
// ─────────────────────────────────────────────────────────────────────────────

/// 64-bit span identifier (OpenTelemetry-compatible).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpanId([u8; 8]);

impl SpanId {
    /// Construct a deterministic `SpanId` by hashing `seed` with an LCG.
    pub fn new_deterministic(seed: u64) -> Self {
        SpanId(lcg_hash_8(seed))
    }

    /// Encode the span ID as a 16-character lowercase hex string.
    pub fn to_hex(&self) -> String {
        bytes_to_hex(&self.0)
    }

    /// Decode a 16-character hex string into a `SpanId`.
    pub fn from_hex(s: &str) -> Result<Self, TraceError> {
        let mut buf = [0u8; 8];
        hex_to_bytes(s, &mut buf)?;
        Ok(SpanId(buf))
    }
}

impl fmt::Display for SpanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpanKind / SpanStatus / AttributeValue / SpanAttribute / SpanEvent
// ─────────────────────────────────────────────────────────────────────────────

/// The role of a span in a distributed trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpanKind {
    /// A request sent by this service to another service.
    Client,
    /// A request received by this service from another service.
    Server,
    /// A message published to a queue/topic.
    Producer,
    /// A message consumed from a queue/topic.
    Consumer,
    /// An internal operation with no cross-service interaction.
    Internal,
}

impl fmt::Display for SpanKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SpanKind::Client => "CLIENT",
            SpanKind::Server => "SERVER",
            SpanKind::Producer => "PRODUCER",
            SpanKind::Consumer => "CONSUMER",
            SpanKind::Internal => "INTERNAL",
        };
        f.write_str(s)
    }
}

/// The completion status of a span.
#[derive(Debug, Clone, PartialEq)]
pub enum SpanStatus {
    /// Default, no explicit status.
    Unset,
    /// The operation completed successfully.
    Ok,
    /// The operation failed with a human-readable message.
    Error { message: String },
}

impl SpanStatus {
    /// Returns `true` when this is an error status.
    pub fn is_error(&self) -> bool {
        matches!(self, SpanStatus::Error { .. })
    }
}

/// A typed attribute value.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl AttributeValue {
    fn to_json_value(&self) -> String {
        match self {
            AttributeValue::String(s) => format!("\"{}\"", s.replace('"', "\\\"")),
            AttributeValue::Int(i) => i.to_string(),
            AttributeValue::Float(f) => f.to_string(),
            AttributeValue::Bool(b) => b.to_string(),
        }
    }
}

/// A key-value attribute attached to a span.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanAttribute {
    pub key: String,
    pub value: AttributeValue,
}

impl SpanAttribute {
    /// Create a new attribute.
    pub fn new(key: impl Into<String>, value: AttributeValue) -> Self {
        SpanAttribute {
            key: key.into(),
            value,
        }
    }
}

/// A timestamped event recorded within a span.
#[derive(Debug, Clone)]
pub struct SpanEvent {
    pub name: String,
    pub timestamp: SystemTime,
    pub attributes: Vec<SpanAttribute>,
}

impl SpanEvent {
    /// Create a new span event with the current wall-clock time.
    pub fn new(name: impl Into<String>, attributes: Vec<SpanAttribute>) -> Self {
        SpanEvent {
            name: name.into(),
            timestamp: SystemTime::now(),
            attributes,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Span
// ─────────────────────────────────────────────────────────────────────────────

/// A single timed operation within a distributed trace.
#[derive(Debug, Clone)]
pub struct Span {
    pub trace_id: TraceId,
    pub span_id: SpanId,
    pub parent_span_id: Option<SpanId>,
    pub name: String,
    pub kind: SpanKind,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub status: SpanStatus,
    pub attributes: Vec<SpanAttribute>,
    pub events: Vec<SpanEvent>,
}

impl Span {
    /// Create a new span.  The span is *not* ended until [`Span::end`] is called.
    pub fn new(name: String, kind: SpanKind, trace_id: TraceId, parent: Option<SpanId>) -> Self {
        // Derive a deterministic span_id from the trace_id bytes XOR'd with a
        // simple counter approximation (name hash) so spans within a trace differ.
        let name_hash =
            name.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let trace_seed = u64::from_le_bytes(trace_id.0[..8].try_into().unwrap_or([0u8; 8]));
        let span_seed = trace_seed ^ name_hash;

        Span {
            trace_id,
            span_id: SpanId::new_deterministic(span_seed),
            parent_span_id: parent,
            name,
            kind,
            start_time: SystemTime::now(),
            end_time: None,
            status: SpanStatus::Unset,
            attributes: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Mark the span as ended with the current wall-clock time.
    pub fn end(&mut self) {
        if self.end_time.is_none() {
            self.end_time = Some(SystemTime::now());
        }
    }

    /// Update the span's completion status.
    pub fn set_status(&mut self, status: SpanStatus) {
        self.status = status;
    }

    /// Append a key-value attribute.
    pub fn add_attribute(&mut self, key: impl Into<String>, value: AttributeValue) {
        self.attributes.push(SpanAttribute {
            key: key.into(),
            value,
        });
    }

    /// Record a named event with optional attributes.
    pub fn add_event(&mut self, name: impl Into<String>, attributes: Vec<SpanAttribute>) {
        self.events.push(SpanEvent::new(name, attributes));
    }

    /// Return the span duration in milliseconds, or `None` if the span has not ended.
    pub fn duration_ms(&self) -> Option<f64> {
        let end = self.end_time?;
        end.duration_since(self.start_time).ok().map(|d| d.as_secs_f64() * 1_000.0)
    }

    /// Returns `true` when the span has an error status.
    pub fn is_error(&self) -> bool {
        self.status.is_error()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trace
// ─────────────────────────────────────────────────────────────────────────────

/// A complete request trace containing one or more spans.
#[derive(Debug, Clone)]
pub struct Trace {
    pub trace_id: TraceId,
    pub spans: Vec<Span>,
}

impl Trace {
    /// Create an empty trace with the given ID.
    pub fn new(trace_id: TraceId) -> Self {
        Trace {
            trace_id,
            spans: Vec::new(),
        }
    }

    /// Return the root span (the span with no parent), if any.
    pub fn root_span(&self) -> Option<&Span> {
        self.spans.iter().find(|s| s.parent_span_id.is_none())
    }

    /// Append a span to the trace.
    pub fn add_span(&mut self, span: Span) {
        self.spans.push(span);
    }

    /// Return the total duration of the trace in milliseconds, derived from
    /// the root span's start/end times.
    pub fn total_duration_ms(&self) -> Option<f64> {
        self.root_span()?.duration_ms()
    }

    /// Return all spans that have an error status.
    pub fn error_spans(&self) -> Vec<&Span> {
        self.spans.iter().filter(|s| s.is_error()).collect()
    }

    /// Serialise the trace to a simple JSON string (no external dependencies).
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(512);
        out.push_str("{\"traceId\":\"");
        out.push_str(&self.trace_id.to_hex());
        out.push_str("\",\"spans\":[");

        for (i, span) in self.spans.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&span_to_json(span));
        }
        out.push_str("]}");
        out
    }
}

/// Serialise a single span to JSON (helper for `Trace::to_json`).
fn span_to_json(span: &Span) -> String {
    let mut out = String::with_capacity(256);
    out.push('{');
    out.push_str("\"traceId\":\"");
    out.push_str(&span.trace_id.to_hex());
    out.push_str("\",\"spanId\":\"");
    out.push_str(&span.span_id.to_hex());
    out.push_str("\",\"name\":\"");
    out.push_str(&span.name.replace('"', "\\\""));
    out.push_str("\",\"kind\":\"");
    out.push_str(&span.kind.to_string());
    out.push('"');

    if let Some(ref p) = span.parent_span_id {
        out.push_str(",\"parentSpanId\":\"");
        out.push_str(&p.to_hex());
        out.push('"');
    }

    // status
    out.push_str(",\"status\":");
    match &span.status {
        SpanStatus::Unset => out.push_str("{\"code\":\"UNSET\"}"),
        SpanStatus::Ok => out.push_str("{\"code\":\"OK\"}"),
        SpanStatus::Error { message } => {
            out.push_str("{\"code\":\"ERROR\",\"message\":\"");
            out.push_str(&message.replace('"', "\\\""));
            out.push_str("\"}");
        },
    }

    // duration
    if let Some(ms) = span.duration_ms() {
        out.push_str(",\"durationMs\":");
        out.push_str(&ms.to_string());
    }

    // attributes
    out.push_str(",\"attributes\":[");
    for (i, attr) in span.attributes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"key\":\"");
        out.push_str(&attr.key.replace('"', "\\\""));
        out.push_str("\",\"value\":");
        out.push_str(&attr.value.to_json_value());
        out.push('}');
    }
    out.push(']');

    // events
    out.push_str(",\"events\":[");
    for (i, event) in span.events.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"name\":\"");
        out.push_str(&event.name.replace('"', "\\\""));
        out.push_str("\"}");
    }
    out.push(']');

    out.push('}');
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// TracingConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the [`RequestTracer`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TracingConfig {
    /// Human-readable name of the service being traced.
    pub service_name: String,
    /// Version of the service being traced.
    pub service_version: String,
    /// Fraction of requests to sample, in [0.0, 1.0].  Default: 1.0.
    pub sample_rate: f32,
    /// Maximum number of spans allowed per trace before new spans are silently
    /// dropped.  Default: 1000.
    pub max_spans_per_trace: usize,
    /// Optional Jaeger/Zipkin endpoint for future export use.
    pub export_endpoint: Option<String>,
}

impl Default for TracingConfig {
    fn default() -> Self {
        TracingConfig {
            service_name: "trustformers-serve".into(),
            service_version: "0.0.0".into(),
            sample_rate: 1.0,
            max_spans_per_trace: 1000,
            export_endpoint: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TracingStats
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics collected by [`RequestTracer`].
#[derive(Debug, Clone, Default)]
pub struct TracingStats {
    pub total_traces: u64,
    pub error_traces: u64,
    pub total_spans: u64,
    pub mean_trace_duration_ms: f64,
}

impl TracingStats {
    /// Fraction of traces that ended in error, in [0.0, 1.0].
    pub fn error_rate(&self) -> f32 {
        if self.total_traces == 0 {
            return 0.0;
        }
        self.error_traces as f32 / self.total_traces as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RequestTracer
// ─────────────────────────────────────────────────────────────────────────────

/// Manages the lifecycle of request traces and their constituent spans.
///
/// Traces are keyed by a caller-supplied *request ID* (typically a UUID
/// string or some other request-scoped identifier).
pub struct RequestTracer {
    pub config: TracingConfig,
    /// In-progress traces, keyed by request_id.
    pub active_traces: HashMap<String, Trace>,
    /// Ring-buffer of recently completed traces.
    pub completed_traces: VecDeque<Trace>,
    /// Maximum number of completed traces to retain.
    pub max_completed: usize,
    pub stats: TracingStats,

    /// Monotonically increasing counter used to generate unique span seeds.
    span_counter: u64,
}

impl RequestTracer {
    /// Create a new tracer with the given configuration.
    pub fn new(config: TracingConfig) -> Self {
        RequestTracer {
            config,
            active_traces: HashMap::new(),
            completed_traces: VecDeque::new(),
            max_completed: 100,
            stats: TracingStats::default(),
            span_counter: 0,
        }
    }

    /// Begin a new trace for `request_id`, naming the root operation `operation`.
    ///
    /// Returns the `TraceId` of the newly created trace, or the existing one if
    /// a trace for `request_id` was already started.
    pub fn start_trace(&mut self, request_id: String, operation: String) -> TraceId {
        // If one already exists, return its id.
        if let Some(t) = self.active_traces.get(&request_id) {
            return t.trace_id.clone();
        }

        // Derive a deterministic trace ID from the request ID's FNV hash.
        let seed = fnv1a_hash(request_id.as_bytes());
        let trace_id = TraceId::new_deterministic(seed);

        let mut trace = Trace::new(trace_id.clone());

        // Create root span.
        let root_span = Span::new(operation, SpanKind::Server, trace_id.clone(), None);
        trace.add_span(root_span);

        self.active_traces.insert(request_id, trace);
        self.stats.total_traces += 1;

        trace_id
    }

    /// Start a new child span within an existing trace.
    ///
    /// Returns the new `SpanId`, or `None` if no trace exists for `request_id`
    /// or the per-trace span limit has been reached.
    pub fn start_span(&mut self, request_id: &str, name: String, kind: SpanKind) -> Option<SpanId> {
        let trace = self.active_traces.get_mut(request_id)?;

        if trace.spans.len() >= self.config.max_spans_per_trace {
            return None;
        }

        // Parent is the most recently started (last) span without an end_time.
        let parent_id = trace
            .spans
            .iter()
            .rev()
            .find(|s| s.end_time.is_none())
            .map(|s| s.span_id.clone());

        self.span_counter = self.span_counter.wrapping_add(1);
        let seed = fnv1a_hash(request_id.as_bytes())
            ^ name.bytes().fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64))
            ^ self.span_counter;

        let mut span = Span::new(name, kind, trace.trace_id.clone(), parent_id);
        // Override the default deterministic span_id with a seed that also
        // incorporates the counter so spans with the same name stay distinct.
        span.span_id = SpanId::new_deterministic(seed);

        let span_id = span.span_id.clone();
        trace.spans.push(span);
        self.stats.total_spans += 1;

        Some(span_id)
    }

    /// End a span, setting its status and recording its end time.
    pub fn end_span(&mut self, request_id: &str, span_id: &SpanId, status: SpanStatus) {
        if let Some(trace) = self.active_traces.get_mut(request_id) {
            if let Some(span) = trace.spans.iter_mut().find(|s| &s.span_id == span_id) {
                span.end();
                span.set_status(status);
            }
        }
    }

    /// Finalise the trace for `request_id`, ending the root span if it has not
    /// been ended already, and moving the trace to the completed buffer.
    ///
    /// Returns the completed `Trace`, or `None` if no trace was found.
    pub fn end_trace(&mut self, request_id: &str) -> Option<Trace> {
        let mut trace = self.active_traces.remove(request_id)?;

        // End any spans that are still open.
        for span in trace.spans.iter_mut() {
            if span.end_time.is_none() {
                span.end();
            }
        }

        // Collect stats.
        let has_errors = trace.spans.iter().any(|s| s.is_error());
        if has_errors {
            self.stats.error_traces += 1;
        }

        if let Some(dur) = trace.total_duration_ms() {
            let n = self.stats.total_traces as f64;
            // Running mean update (Welford-ish approximation using completed count).
            let completed = (self.stats.total_traces - self.active_traces.len() as u64) as f64;
            if completed <= 1.0 {
                self.stats.mean_trace_duration_ms = dur;
            } else {
                self.stats.mean_trace_duration_ms =
                    (self.stats.mean_trace_duration_ms * (completed - 1.0) + dur) / completed;
            }
            let _ = n; // suppress unused warning
        }

        // Evict oldest if at capacity.
        while self.completed_traces.len() >= self.max_completed {
            self.completed_traces.pop_front();
        }
        self.completed_traces.push_back(trace.clone());

        Some(trace)
    }

    /// Return references to all recently completed traces (in completion order).
    pub fn recent_traces(&self) -> Vec<&Trace> {
        self.completed_traces.iter().collect()
    }

    /// Fraction of completed traces that contained at least one error span.
    pub fn error_rate(&self) -> f32 {
        self.stats.error_rate()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FNV-1a hash (64-bit) — used to derive seeds from string request IDs
// ─────────────────────────────────────────────────────────────────────────────

fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut hash = FNV_OFFSET;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. TraceId hex round-trip
    #[test]
    fn test_trace_id_hex_roundtrip() {
        let id = TraceId::new_deterministic(42);
        let hex = id.to_hex();
        assert_eq!(hex.len(), 32, "TraceId hex must be 32 chars");
        let id2 = TraceId::from_hex(&hex).expect("valid hex should parse");
        assert_eq!(id, id2);
    }

    // 2. SpanId hex round-trip
    #[test]
    fn test_span_id_hex_roundtrip() {
        let id = SpanId::new_deterministic(99);
        let hex = id.to_hex();
        assert_eq!(hex.len(), 16, "SpanId hex must be 16 chars");
        let id2 = SpanId::from_hex(&hex).expect("valid hex should parse");
        assert_eq!(id, id2);
    }

    // 3. from_hex rejects bad length
    #[test]
    fn test_trace_id_from_hex_bad_length() {
        let result = TraceId::from_hex("deadbeef");
        assert!(result.is_err(), "short hex should fail");
    }

    // 4. Span creation and end
    #[test]
    fn test_span_create_and_end() {
        let trace_id = TraceId::new_deterministic(1);
        let mut span = Span::new("my-op".to_string(), SpanKind::Server, trace_id, None);
        assert!(span.end_time.is_none());
        span.end();
        assert!(span.end_time.is_some());
    }

    // 5. Duration calculation
    #[test]
    fn test_span_duration() {
        let trace_id = TraceId::new_deterministic(2);
        let mut span = Span::new("latency".to_string(), SpanKind::Internal, trace_id, None);
        // Sleep a tiny bit so duration > 0
        std::thread::sleep(std::time::Duration::from_millis(5));
        span.end();
        let dur = span.duration_ms().expect("ended span should have duration");
        assert!(dur >= 0.0, "duration must be non-negative");
    }

    // 6. Add attributes
    #[test]
    fn test_span_add_attributes() {
        let trace_id = TraceId::new_deterministic(3);
        let mut span = Span::new("attrs".to_string(), SpanKind::Client, trace_id, None);
        span.add_attribute("http.method", AttributeValue::String("GET".to_string()));
        span.add_attribute("http.status_code", AttributeValue::Int(200));
        span.add_attribute("latency", AttributeValue::Float(1.5));
        span.add_attribute("cached", AttributeValue::Bool(true));
        assert_eq!(span.attributes.len(), 4);
    }

    // 7. Add events
    #[test]
    fn test_span_add_events() {
        let trace_id = TraceId::new_deterministic(4);
        let mut span = Span::new("events".to_string(), SpanKind::Producer, trace_id, None);
        span.add_event("cache_miss", vec![]);
        span.add_event(
            "db_query",
            vec![SpanAttribute::new("rows", AttributeValue::Int(5))],
        );
        assert_eq!(span.events.len(), 2);
        assert_eq!(span.events[0].name, "cache_miss");
    }

    // 8. Trace JSON output contains "traceId"
    #[test]
    fn test_trace_to_json_contains_trace_id() {
        let trace_id = TraceId::new_deterministic(5);
        let mut trace = Trace::new(trace_id.clone());
        let mut span = Span::new("root".to_string(), SpanKind::Server, trace_id, None);
        span.end();
        trace.add_span(span);
        let json = trace.to_json();
        assert!(json.contains("traceId"), "JSON must contain 'traceId' key");
    }

    // 9. Error span detection
    #[test]
    fn test_error_span_detection() {
        let trace_id = TraceId::new_deterministic(6);
        let mut trace = Trace::new(trace_id.clone());
        let mut good_span = Span::new("ok".to_string(), SpanKind::Internal, trace_id.clone(), None);
        good_span.set_status(SpanStatus::Ok);
        good_span.end();

        let mut bad_span = Span::new("fail".to_string(), SpanKind::Internal, trace_id, None);
        bad_span.set_status(SpanStatus::Error {
            message: "something broke".to_string(),
        });
        bad_span.end();

        trace.add_span(good_span);
        trace.add_span(bad_span);

        let errors = trace.error_spans();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].is_error());
    }

    // 10. RequestTracer start/end trace
    #[test]
    fn test_request_tracer_start_end_trace() {
        let config = TracingConfig::default();
        let mut tracer = RequestTracer::new(config);

        let request_id = "req-001".to_string();
        let trace_id = tracer.start_trace(request_id.clone(), "handle_request".to_string());
        assert_eq!(trace_id.to_hex().len(), 32);

        let completed = tracer.end_trace(&request_id).expect("trace must exist");
        assert!(!completed.spans.is_empty());
        assert_eq!(tracer.recent_traces().len(), 1);
    }

    // 11. Error rate calculation
    #[test]
    fn test_error_rate_calculation() {
        let config = TracingConfig::default();
        let mut tracer = RequestTracer::new(config);

        // Complete two traces: one with error, one without
        for (id, with_error) in [("req-a", true), ("req-b", false)] {
            let rid = id.to_string();
            tracer.start_trace(rid.clone(), "op".to_string());
            if with_error {
                let span_id = tracer
                    .start_span(&rid, "inner".to_string(), SpanKind::Internal)
                    .expect("should create span");
                tracer.end_span(
                    &rid,
                    &span_id,
                    SpanStatus::Error {
                        message: "boom".to_string(),
                    },
                );
            }
            tracer.end_trace(&rid);
        }

        let rate = tracer.error_rate();
        assert!(rate > 0.0 && rate <= 1.0, "error rate must be in (0,1]");
    }

    // 12. Completed trace eviction (ring-buffer)
    #[test]
    fn test_completed_trace_eviction() {
        let config = TracingConfig::default();
        let mut tracer = RequestTracer::new(config);
        tracer.max_completed = 3; // small buffer for the test

        for i in 0..5u64 {
            let rid = format!("req-{}", i);
            tracer.start_trace(rid.clone(), "op".to_string());
            tracer.end_trace(&rid);
        }

        // Only the last 3 traces should be retained.
        assert_eq!(tracer.recent_traces().len(), 3);
    }

    // 13. Span count tracking
    #[test]
    fn test_span_count_tracking() {
        let config = TracingConfig::default();
        let mut tracer = RequestTracer::new(config);

        let rid = "req-x".to_string();
        tracer.start_trace(rid.clone(), "root".to_string());
        // The root span is created by start_trace; stats counter is NOT incremented
        // for the root span (it is added silently).  Subsequent start_span calls
        // increment the counter.
        tracer.start_span(&rid, "child-a".to_string(), SpanKind::Client);
        tracer.start_span(&rid, "child-b".to_string(), SpanKind::Internal);
        tracer.end_trace(&rid);

        // 2 explicit child spans recorded via start_span
        assert_eq!(tracer.stats.total_spans, 2);
    }

    // 14. SpanId from_hex rejects bad chars
    #[test]
    fn test_span_id_from_hex_bad_char() {
        let result = SpanId::from_hex("zzzzzzzzzzzzzzzz");
        assert!(result.is_err());
    }

    // 15. Deterministic IDs are stable across calls
    #[test]
    fn test_deterministic_ids_stable() {
        let a = TraceId::new_deterministic(777);
        let b = TraceId::new_deterministic(777);
        assert_eq!(a, b);

        let c = SpanId::new_deterministic(777);
        let d = SpanId::new_deterministic(777);
        assert_eq!(c, d);
    }
}
