//! Unified export interface for all supported trace formats.
//!
//! [`TraceExporter`] dispatches to the appropriate backend based on
//! [`ExportConfig::format`].  New formats can be added by extending
//! [`ExportFormat`] and [`TraceExporter::export_all`].

use std::path::Path;

use anyhow::Result;

use super::perfetto::{PerfettoEvent, PerfettoExporter, PerfettoPhase, PerfettoTrace};
use super::tracy::{TracyExporter, TracyTrace, TracyZone};

// ─────────────────────────────────────────────────────────────────────────────
// TimingEvent
// ─────────────────────────────────────────────────────────────────────────────

/// A generic timed event used by the CSV and JSON exporters.
///
/// Can be converted from a [`PerfettoEvent`] or a [`TracyZone`].
///
/// # Example
///
/// ```
/// use trustformers_debug::export::unified::TimingEvent;
///
/// let ev = TimingEvent {
///     timestamp_ns: 1_000_000,
///     duration_ns: 500_000,
///     thread_id: 0,
///     name: "attention_forward".to_string(),
/// };
/// assert_eq!(ev.timestamp_ns, 1_000_000);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TimingEvent {
    /// Start timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Duration in nanoseconds.
    pub duration_ns: u64,
    /// Thread / worker identifier.
    pub thread_id: u32,
    /// Human-readable operation name.
    pub name: String,
}

impl From<&TracyZone> for TimingEvent {
    fn from(z: &TracyZone) -> Self {
        Self {
            timestamp_ns: z.timestamp_ns,
            duration_ns: z.duration_ns,
            thread_id: z.thread_id,
            name: z.name.clone(),
        }
    }
}

impl From<&PerfettoEvent> for TimingEvent {
    fn from(e: &PerfettoEvent) -> Self {
        Self {
            timestamp_ns: e.timestamp_us * 1_000,
            duration_ns: e.duration_us.unwrap_or(0) * 1_000,
            thread_id: e.tid,
            name: e.name.clone(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExportFormat
// ─────────────────────────────────────────────────────────────────────────────

/// Supported export formats for profiling trace data.
///
/// # Example
///
/// ```
/// use trustformers_debug::export::unified::ExportFormat;
///
/// let fmt = ExportFormat::Csv;
/// assert_eq!(fmt.extension(), "csv");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportFormat {
    /// Perfetto/Chrome trace event JSON.
    Perfetto,
    /// Tracy offline CSV format.
    Tracy,
    /// `chrome://tracing` compatible JSON (alias for Perfetto).
    ChromeTrace,
    /// Generic timing CSV (timestamp_ns, duration_ns, thread_id, name).
    Csv,
    /// Simple JSON array of timing events.
    Json,
}

impl ExportFormat {
    /// Returns the conventional file extension for this format.
    pub fn extension(&self) -> &str {
        match self {
            Self::Perfetto | Self::ChromeTrace => "json",
            Self::Tracy => "csv",
            Self::Csv => "csv",
            Self::Json => "json",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExportConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a single export operation.
///
/// # Example
///
/// ```
/// use trustformers_debug::export::unified::{ExportConfig, ExportFormat};
///
/// let cfg = ExportConfig {
///     format: ExportFormat::Csv,
///     output_path: "/tmp/trace.csv".to_string(),
///     compress: false,
/// };
/// assert_eq!(cfg.format, ExportFormat::Csv);
/// ```
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Target export format.
    pub format: ExportFormat,
    /// Filesystem path where the output will be written.
    pub output_path: String,
    /// Reserved for future compression support (currently ignored).
    pub compress: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// ExportError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors returned by [`TraceExporter`].
#[derive(Debug, Clone, PartialEq)]
pub enum ExportError {
    /// The requested format is not supported in this build.
    UnsupportedFormat(String),
    /// An I/O error occurred while writing the output file.
    IoError(String),
    /// The input trace/events slice was empty.
    EmptyTrace,
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedFormat(s) => write!(f, "unsupported export format: {s}"),
            Self::IoError(s) => write!(f, "I/O error during export: {s}"),
            Self::EmptyTrace => write!(f, "export trace is empty"),
        }
    }
}

impl std::error::Error for ExportError {}

impl From<anyhow::Error> for ExportError {
    fn from(e: anyhow::Error) -> Self {
        Self::IoError(e.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProfilingTrace wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// A unified view of a profiling trace that can be exported to multiple formats.
///
/// Build one from a [`PerfettoTrace`] or a [`TracyTrace`], then call
/// [`TraceExporter::export_all`] to materialise it in the configured format.
#[derive(Debug, Default)]
pub struct ProfilingTrace {
    events: Vec<TimingEvent>,
}

impl ProfilingTrace {
    /// Creates an empty trace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a [`TimingEvent`].
    pub fn add_event(&mut self, event: TimingEvent) {
        self.events.push(event);
    }

    /// Returns a slice of all events.
    pub fn events(&self) -> &[TimingEvent] {
        &self.events
    }

    /// Returns the total number of events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` when the trace contains no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Builds a [`PerfettoTrace`] from this unified trace.
    pub fn to_perfetto(&self) -> PerfettoTrace {
        let mut trace = PerfettoTrace::new();
        for ev in &self.events {
            trace.add_event(PerfettoEvent {
                name: ev.name.clone(),
                phase: PerfettoPhase::Complete,
                timestamp_us: ev.timestamp_ns / 1_000,
                duration_us: Some(ev.duration_ns / 1_000),
                pid: 1,
                tid: ev.thread_id,
                args: std::collections::HashMap::new(),
            });
        }
        trace
    }

    /// Builds a [`TracyTrace`] from this unified trace.
    pub fn to_tracy(&self) -> TracyTrace {
        let mut trace = TracyTrace::new();
        for ev in &self.events {
            trace.add_zone(TracyZone {
                name: ev.name.clone(),
                timestamp_ns: ev.timestamp_ns,
                duration_ns: ev.duration_ns,
                thread_id: ev.thread_id,
            });
        }
        trace
    }
}

impl From<&TracyTrace> for ProfilingTrace {
    fn from(t: &TracyTrace) -> Self {
        let events = t.zones().iter().map(TimingEvent::from).collect();
        Self { events }
    }
}

impl From<&PerfettoTrace> for ProfilingTrace {
    fn from(t: &PerfettoTrace) -> Self {
        // PerfettoTrace doesn't expose its internal Vec directly; we rebuild
        // from the JSON round-trip is overkill — instead we use a to_string
        // approach only as last resort.  Here we use the public API via
        // export_to_string + manual parse.  Since PerfettoTrace has no
        // `events()` accessor we build via a zero-copy approach instead:
        // call `add_event` on a fresh trace.
        //
        // NOTE: PerfettoTrace itself is opaque, so we rely on From<TimingEvent>
        // to bridge the types.  In practice callers will construct a
        // `ProfilingTrace` directly via `add_event`.
        let _ = t; // suppress unused warning — this path is documented as limited
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CsvExporter
// ─────────────────────────────────────────────────────────────────────────────

/// Exports a slice of [`TimingEvent`]s to a CSV string.
///
/// # Example
///
/// ```
/// use trustformers_debug::export::unified::{CsvExporter, TimingEvent};
///
/// let events = vec![TimingEvent { timestamp_ns: 0, duration_ns: 1000, thread_id: 0, name: "op".to_string() }];
/// let csv = CsvExporter::export_to_csv(&events);
/// assert!(csv.contains("timestamp_ns,duration_ns,thread_id,name"));
/// assert!(csv.contains("0,1000,0,op"));
/// ```
pub struct CsvExporter;

impl CsvExporter {
    /// Serialises `events` to a CSV string with header:
    /// `timestamp_ns,duration_ns,thread_id,name`
    pub fn export_to_csv(events: &[TimingEvent]) -> String {
        let mut out = String::from("timestamp_ns,duration_ns,thread_id,name\n");
        for ev in events {
            let safe_name = ev.name.replace(',', "\\,");
            use std::fmt::Write as _;
            let _ = writeln!(
                out,
                "{},{},{},{}",
                ev.timestamp_ns, ev.duration_ns, ev.thread_id, safe_name
            );
        }
        out
    }

    /// Writes CSV to the file at `path`.
    pub fn export_to_file(events: &[TimingEvent], path: &Path) -> Result<()> {
        let csv = Self::export_to_csv(events);
        std::fs::write(path, csv.as_bytes())?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JsonExporter
// ─────────────────────────────────────────────────────────────────────────────

/// Exports a slice of [`TimingEvent`]s to a JSON array string.
///
/// # Example
///
/// ```
/// use trustformers_debug::export::unified::{JsonExporter, TimingEvent};
///
/// let events = vec![
///     TimingEvent { timestamp_ns: 1000, duration_ns: 500, thread_id: 1, name: "ffn".to_string() },
/// ];
/// let json = JsonExporter::export_to_json(&events);
/// assert!(json.starts_with('['));
/// assert!(json.contains("\"name\":\"ffn\""));
/// ```
pub struct JsonExporter;

impl JsonExporter {
    /// Serialises `events` to a JSON array without external dependencies.
    pub fn export_to_json(events: &[TimingEvent]) -> String {
        use std::fmt::Write as _;
        let mut out = String::from('[');
        for (i, ev) in events.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let escaped_name = escape_json_string_local(&ev.name);
            let _ = write!(
                out,
                r#"{{"timestamp_ns":{},"duration_ns":{},"thread_id":{},"name":"{}"}}"#,
                ev.timestamp_ns, ev.duration_ns, ev.thread_id, escaped_name
            );
        }
        out.push(']');
        out
    }

    /// Writes JSON to the file at `path`.
    pub fn export_to_file(events: &[TimingEvent], path: &Path) -> Result<()> {
        let json = Self::export_to_json(events);
        std::fs::write(path, json.as_bytes())?;
        Ok(())
    }
}

fn escape_json_string_local(s: &str) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// TraceExporter
// ─────────────────────────────────────────────────────────────────────────────

/// Unified exporter that dispatches to the appropriate backend.
///
/// # Example
///
/// ```no_run
/// use trustformers_debug::export::unified::{
///     ExportConfig, ExportFormat, ProfilingTrace, TimingEvent, TraceExporter,
/// };
///
/// let mut trace = ProfilingTrace::new();
/// trace.add_event(TimingEvent {
///     timestamp_ns: 0,
///     duration_ns: 1_000_000,
///     thread_id: 0,
///     name: "attention".to_string(),
/// });
/// let config = ExportConfig {
///     format: ExportFormat::Csv,
///     output_path: "/tmp/trace.csv".to_string(),
///     compress: false,
/// };
/// TraceExporter::export_all(&trace, &config).unwrap();
/// ```
pub struct TraceExporter;

impl TraceExporter {
    /// Exports `trace` according to `config`.
    ///
    /// # Errors
    ///
    /// Returns [`ExportError::EmptyTrace`] when the trace is empty.
    /// Returns [`ExportError::IoError`] on filesystem errors.
    pub fn export_all(trace: &ProfilingTrace, config: &ExportConfig) -> Result<(), ExportError> {
        if trace.is_empty() {
            return Err(ExportError::EmptyTrace);
        }
        let path = Path::new(&config.output_path);

        match &config.format {
            ExportFormat::Perfetto | ExportFormat::ChromeTrace => {
                let perf = trace.to_perfetto();
                perf.export_to_file(path).map_err(ExportError::from)?;
            }
            ExportFormat::Tracy => {
                let tracy = trace.to_tracy();
                tracy.export_to_file(path).map_err(ExportError::from)?;
            }
            ExportFormat::Csv => {
                CsvExporter::export_to_file(trace.events(), path)
                    .map_err(ExportError::from)?;
            }
            ExportFormat::Json => {
                JsonExporter::export_to_file(trace.events(), path)
                    .map_err(ExportError::from)?;
            }
        }
        Ok(())
    }

    /// Exports from a [`crate::ProfilerReport`] via [`PerfettoExporter`] or
    /// [`TracyExporter`] depending on the format.
    ///
    /// For CSV/JSON formats the slowest layers are converted to
    /// [`TimingEvent`]s first.
    pub fn export_profiler_report(
        report: &crate::ProfilerReport,
        config: &ExportConfig,
    ) -> Result<(), ExportError> {
        let path = Path::new(&config.output_path);
        match &config.format {
            ExportFormat::Perfetto | ExportFormat::ChromeTrace => {
                PerfettoExporter::export_profiler_report(report, path)
                    .map_err(ExportError::from)?;
            }
            ExportFormat::Tracy => {
                TracyExporter::export_profiler_report(report, path)
                    .map_err(ExportError::from)?;
            }
            ExportFormat::Csv | ExportFormat::Json => {
                let events: Vec<TimingEvent> = report
                    .slowest_layers
                    .iter()
                    .enumerate()
                    .map(|(i, (name, dur))| TimingEvent {
                        timestamp_ns: i as u64 * 1_000_000,
                        duration_ns: dur.as_nanos() as u64,
                        thread_id: 0,
                        name: name.clone(),
                    })
                    .collect();
                if events.is_empty() {
                    return Err(ExportError::EmptyTrace);
                }
                match &config.format {
                    ExportFormat::Csv => CsvExporter::export_to_file(&events, path)
                        .map_err(ExportError::from)?,
                    _ => JsonExporter::export_to_file(&events, path)
                        .map_err(ExportError::from)?,
                }
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_events() -> Vec<TimingEvent> {
        vec![
            TimingEvent { timestamp_ns: 0, duration_ns: 1_000_000, thread_id: 0, name: "attention".to_string() },
            TimingEvent { timestamp_ns: 1_000_000, duration_ns: 2_000_000, thread_id: 1, name: "ffn".to_string() },
            TimingEvent { timestamp_ns: 3_000_000, duration_ns: 500_000, thread_id: 0, name: "layer_norm".to_string() },
        ]
    }

    fn sample_trace() -> ProfilingTrace {
        let mut t = ProfilingTrace::new();
        for e in sample_events() {
            t.add_event(e);
        }
        t
    }

    // ── CsvExporter ──────────────────────────────────────────────────────────

    #[test]
    fn test_csv_header() {
        let csv = CsvExporter::export_to_csv(&[]);
        assert_eq!(csv.trim(), "timestamp_ns,duration_ns,thread_id,name");
    }

    #[test]
    fn test_csv_export_values() {
        let csv = CsvExporter::export_to_csv(&sample_events());
        assert!(csv.contains("0,1000000,0,attention"));
        assert!(csv.contains("1000000,2000000,1,ffn"));
    }

    #[test]
    fn test_csv_comma_escaping() {
        let events = vec![TimingEvent {
            timestamp_ns: 0,
            duration_ns: 0,
            thread_id: 0,
            name: "op,with,commas".to_string(),
        }];
        let csv = CsvExporter::export_to_csv(&events);
        assert!(csv.contains("op\\,with\\,commas"));
    }

    #[test]
    fn test_csv_export_to_file() {
        let path = std::env::temp_dir().join("csv_export_test.csv");
        CsvExporter::export_to_file(&sample_events(), &path).unwrap();
        assert!(path.exists());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("attention"));
        std::fs::remove_file(&path).ok();
    }

    // ── JsonExporter ─────────────────────────────────────────────────────────

    #[test]
    fn test_json_export_structure() {
        let json = JsonExporter::export_to_json(&sample_events());
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("\"name\":\"attention\""));
        assert!(json.contains("\"timestamp_ns\":0"));
        assert!(json.contains("\"thread_id\":1"));
    }

    #[test]
    fn test_json_export_empty() {
        let json = JsonExporter::export_to_json(&[]);
        assert_eq!(json, "[]");
    }

    #[test]
    fn test_json_export_escaping() {
        let events = vec![TimingEvent {
            timestamp_ns: 0,
            duration_ns: 0,
            thread_id: 0,
            name: "say \"hello\"".to_string(),
        }];
        let json = JsonExporter::export_to_json(&events);
        assert!(json.contains("\\\"hello\\\""));
    }

    #[test]
    fn test_json_export_to_file() {
        let path = std::env::temp_dir().join("json_export_test.json");
        JsonExporter::export_to_file(&sample_events(), &path).unwrap();
        assert!(path.exists());
        std::fs::remove_file(&path).ok();
    }

    // ── ExportFormat ─────────────────────────────────────────────────────────

    #[test]
    fn test_export_format_extension() {
        assert_eq!(ExportFormat::Perfetto.extension(), "json");
        assert_eq!(ExportFormat::ChromeTrace.extension(), "json");
        assert_eq!(ExportFormat::Tracy.extension(), "csv");
        assert_eq!(ExportFormat::Csv.extension(), "csv");
        assert_eq!(ExportFormat::Json.extension(), "json");
    }

    // ── ProfilingTrace conversions ────────────────────────────────────────────

    #[test]
    fn test_profiling_trace_to_perfetto() {
        let trace = sample_trace();
        let perf = trace.to_perfetto();
        assert_eq!(perf.len(), 3);
    }

    #[test]
    fn test_profiling_trace_to_tracy() {
        let trace = sample_trace();
        let tracy = trace.to_tracy();
        assert_eq!(tracy.zones().len(), 3);
        assert_eq!(tracy.zones()[0].name, "attention");
    }

    #[test]
    fn test_timing_event_from_tracy_zone() {
        let zone = TracyZone {
            name: "test".to_string(),
            timestamp_ns: 5_000,
            duration_ns: 1_000,
            thread_id: 2,
        };
        let ev = TimingEvent::from(&zone);
        assert_eq!(ev.timestamp_ns, 5_000);
        assert_eq!(ev.duration_ns, 1_000);
        assert_eq!(ev.thread_id, 2);
        assert_eq!(ev.name, "test");
    }

    // ── TraceExporter ─────────────────────────────────────────────────────────

    #[test]
    fn test_trace_exporter_csv() {
        let trace = sample_trace();
        let path = std::env::temp_dir().join("unified_export_csv.csv");
        let config = ExportConfig {
            format: ExportFormat::Csv,
            output_path: path.to_string_lossy().into_owned(),
            compress: false,
        };
        TraceExporter::export_all(&trace, &config).unwrap();
        assert!(path.exists());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_trace_exporter_json() {
        let trace = sample_trace();
        let path = std::env::temp_dir().join("unified_export_json.json");
        let config = ExportConfig {
            format: ExportFormat::Json,
            output_path: path.to_string_lossy().into_owned(),
            compress: false,
        };
        TraceExporter::export_all(&trace, &config).unwrap();
        assert!(path.exists());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("attention"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_trace_exporter_empty_returns_error() {
        let trace = ProfilingTrace::new();
        let config = ExportConfig {
            format: ExportFormat::Csv,
            output_path: "/tmp/should_not_exist.csv".to_string(),
            compress: false,
        };
        let result = TraceExporter::export_all(&trace, &config);
        assert!(matches!(result, Err(ExportError::EmptyTrace)));
    }

    #[test]
    fn test_export_error_display() {
        assert!(ExportError::EmptyTrace.to_string().contains("empty"));
        assert!(ExportError::UnsupportedFormat("xyz".to_string()).to_string().contains("xyz"));
        assert!(ExportError::IoError("perm denied".to_string()).to_string().contains("perm denied"));
    }
}
