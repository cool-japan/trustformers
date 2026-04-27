//! Perfetto/Chrome Trace Event Format export
//!
//! Exports profiling data in Perfetto JSON format compatible with
//! `chrome://tracing` and the Perfetto UI at <https://ui.perfetto.dev>.
//!
//! # Example
//!
//! ```no_run
//! use trustformers_debug::export::perfetto::{PerfettoTrace, PerfettoEvent, PerfettoPhase};
//! use std::collections::HashMap;
//!
//! let mut trace = PerfettoTrace::new();
//! let mut args = HashMap::new();
//! args.insert("layer".to_string(), serde_json::json!("attention"));
//! trace.add_event(PerfettoEvent {
//!     name: "layer_forward".to_string(),
//!     phase: PerfettoPhase::Complete,
//!     timestamp_us: 12345,
//!     duration_us: Some(1500),
//!     pid: 1,
//!     tid: 1,
//!     args,
//! });
//! let json = trace.export_to_string().unwrap();
//! println!("{}", json);
//! ```

use std::collections::HashMap;
use std::io::Write;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ProfilerReport;

// ─────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────

/// Phase codes used in the Perfetto/Chrome trace-event format.
///
/// Each variant corresponds to the `ph` field in a trace-event object.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PerfettoPhase {
    /// `B` — duration begin.
    Begin,
    /// `E` — duration end.
    End,
    /// `X` — complete (begin + duration).
    Complete,
    /// `i` — instant event.
    Instant,
    /// `C` — counter.
    Counter,
}

impl PerfettoPhase {
    /// Returns the single-character phase code.
    pub fn as_code(&self) -> &'static str {
        match self {
            Self::Begin => "B",
            Self::End => "E",
            Self::Complete => "X",
            Self::Instant => "i",
            Self::Counter => "C",
        }
    }
}

/// A single trace event in Perfetto/Chrome format.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use trustformers_debug::export::perfetto::{PerfettoEvent, PerfettoPhase};
///
/// let event = PerfettoEvent {
///     name: "forward".to_string(),
///     phase: PerfettoPhase::Complete,
///     timestamp_us: 0,
///     duration_us: Some(500),
///     pid: 1,
///     tid: 1,
///     args: HashMap::new(),
/// };
/// assert_eq!(event.phase.as_code(), "X");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfettoEvent {
    /// Human-readable name shown in the trace viewer.
    pub name: String,
    /// Event phase (Begin, End, Complete, Instant, Counter).
    pub phase: PerfettoPhase,
    /// Timestamp in **microseconds** since the trace start.
    pub timestamp_us: u64,
    /// Duration in microseconds (required for `Complete` events).
    pub duration_us: Option<u64>,
    /// Process ID.
    pub pid: u32,
    /// Thread ID.
    pub tid: u32,
    /// Arbitrary key-value metadata.
    pub args: HashMap<String, Value>,
}

/// An in-memory collection of [`PerfettoEvent`]s that can be serialised to
/// the Chrome trace-event JSON format.
///
/// # Example
///
/// ```
/// use trustformers_debug::export::perfetto::PerfettoTrace;
///
/// let trace = PerfettoTrace::new();
/// assert_eq!(trace.len(), 0);
/// ```
#[derive(Debug, Default)]
pub struct PerfettoTrace {
    events: Vec<PerfettoEvent>,
}

impl PerfettoTrace {
    /// Creates an empty trace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a single event.
    pub fn add_event(&mut self, event: PerfettoEvent) {
        self.events.push(event);
    }

    /// Returns the number of events in the trace.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` if the trace contains no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Serialises the trace to a JSON string in Perfetto format.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialisation fails.
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::export::perfetto::PerfettoTrace;
    ///
    /// let trace = PerfettoTrace::new();
    /// let json = trace.export_to_string().unwrap();
    /// assert!(json.contains("traceEvents"));
    /// ```
    pub fn export_to_string(&self) -> Result<String> {
        let doc = self.build_doc();
        Ok(serde_json::to_string_pretty(&doc)?)
    }

    /// Writes the trace to a file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or written to, or if
    /// JSON serialisation fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_debug::export::perfetto::PerfettoTrace;
    ///
    /// let trace = PerfettoTrace::new();
    /// trace.export_to_file(std::path::Path::new("/tmp/trace.json")).unwrap();
    /// ```
    pub fn export_to_file(&self, path: &std::path::Path) -> Result<()> {
        let json = self.export_to_string()?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(json.as_bytes())?;
        tracing::debug!("Perfetto trace written to {}", path.display());
        Ok(())
    }

    // ── helpers ──────────────────────────────────────────────

    fn build_doc(&self) -> Value {
        let events: Vec<Value> = self.events.iter().map(event_to_value).collect();
        serde_json::json!({
            "traceEvents": events,
            "displayTimeUnit": "ms",
        })
    }
}

// ─────────────────────────────────────────────────────────────
// PerfettoExporter
// ─────────────────────────────────────────────────────────────

/// Converts a [`ProfilerReport`] to a [`PerfettoTrace`] and writes it to disk.
///
/// # Example
///
/// ```no_run
/// use trustformers_debug::export::perfetto::PerfettoExporter;
/// use trustformers_debug::ProfilerReport;
/// use std::collections::HashMap;
/// use std::time::Duration;
///
/// // Build a minimal report for demonstration.
/// let report = ProfilerReport {
///     total_events: 0,
///     total_runtime: Duration::from_millis(0),
///     statistics: HashMap::new(),
///     bottlenecks: vec![],
///     slowest_layers: vec![],
///     memory_efficiency: Default::default(),
///     recommendations: vec![],
/// };
/// PerfettoExporter::export_profiler_report(
///     &report,
///     std::path::Path::new("/tmp/report.json"),
/// ).unwrap();
/// ```
pub struct PerfettoExporter;

impl PerfettoExporter {
    /// Converts a [`ProfilerReport`] into a Perfetto trace file.
    ///
    /// Each layer in [`ProfilerReport::slowest_layers`] becomes a `Complete`
    /// trace event.  Bottlenecks are appended as `Instant` events.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn export_profiler_report(
        report: &ProfilerReport,
        path: &std::path::Path,
    ) -> Result<()> {
        let mut trace = PerfettoTrace::new();
        let mut cursor_us: u64 = 0;

        for (layer_name, duration) in &report.slowest_layers {
            let dur_us = duration.as_micros() as u64;
            let mut args = HashMap::new();
            args.insert("layer_name".to_string(), Value::String(layer_name.clone()));
            trace.add_event(PerfettoEvent {
                name: layer_name.clone(),
                phase: PerfettoPhase::Complete,
                timestamp_us: cursor_us,
                duration_us: Some(dur_us),
                pid: 1,
                tid: 1,
                args,
            });
            cursor_us += dur_us;
        }

        for bottleneck in &report.bottlenecks {
            let mut args = HashMap::new();
            args.insert(
                "description".to_string(),
                Value::String(bottleneck.description.clone()),
            );
            args.insert(
                "suggestion".to_string(),
                Value::String(bottleneck.suggestion.clone()),
            );
            trace.add_event(PerfettoEvent {
                name: format!("bottleneck:{}", bottleneck.location),
                phase: PerfettoPhase::Instant,
                timestamp_us: cursor_us,
                duration_us: None,
                pid: 1,
                tid: 1,
                args,
            });
        }

        trace.export_to_file(path)
    }
}

// ─────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────

fn event_to_value(e: &PerfettoEvent) -> Value {
    let mut obj = serde_json::json!({
        "name": e.name,
        "ph": e.phase.as_code(),
        "ts": e.timestamp_us,
        "pid": e.pid,
        "tid": e.tid,
    });

    if let Some(dur) = e.duration_us {
        obj["dur"] = Value::Number(dur.into());
    }

    if !e.args.is_empty() {
        obj["args"] = Value::Object(
            e.args
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        );
    }

    obj
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_event(name: &str, phase: PerfettoPhase, ts: u64, dur: Option<u64>) -> PerfettoEvent {
        PerfettoEvent {
            name: name.to_string(),
            phase,
            timestamp_us: ts,
            duration_us: dur,
            pid: 1,
            tid: 1,
            args: HashMap::new(),
        }
    }

    #[test]
    fn test_empty_trace_roundtrip() {
        let trace = PerfettoTrace::new();
        let json = trace.export_to_string().unwrap();
        let parsed: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["traceEvents"].as_array().unwrap().len(), 0);
        assert_eq!(parsed["displayTimeUnit"], "ms");
    }

    #[test]
    fn test_add_complete_event() {
        let mut trace = PerfettoTrace::new();
        trace.add_event(make_event(
            "forward",
            PerfettoPhase::Complete,
            1000,
            Some(500),
        ));
        assert_eq!(trace.len(), 1);

        let json = trace.export_to_string().unwrap();
        let parsed: Value = serde_json::from_str(&json).unwrap();
        let ev = &parsed["traceEvents"][0];
        assert_eq!(ev["ph"], "X");
        assert_eq!(ev["ts"], 1000_u64);
        assert_eq!(ev["dur"], 500_u64);
        assert_eq!(ev["name"], "forward");
    }

    #[test]
    fn test_begin_end_phases() {
        let mut trace = PerfettoTrace::new();
        trace.add_event(make_event("op", PerfettoPhase::Begin, 0, None));
        trace.add_event(make_event("op", PerfettoPhase::End, 200, None));
        let json = trace.export_to_string().unwrap();
        let parsed: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["traceEvents"][0]["ph"], "B");
        assert_eq!(parsed["traceEvents"][1]["ph"], "E");
    }

    #[test]
    fn test_export_to_file() {
        let mut dir = std::env::temp_dir();
        dir.push("perfetto_test_trace.json");

        let mut trace = PerfettoTrace::new();
        let mut args = HashMap::new();
        args.insert("batch_size".to_string(), serde_json::json!(32));
        trace.add_event(PerfettoEvent {
            name: "attention_forward".to_string(),
            phase: PerfettoPhase::Complete,
            timestamp_us: 0,
            duration_us: Some(1500),
            pid: 1,
            tid: 2,
            args,
        });
        trace.export_to_file(&dir).unwrap();
        assert!(dir.exists());

        let content = std::fs::read_to_string(&dir).unwrap();
        let parsed: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed["traceEvents"].as_array().unwrap().len(), 1);

        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn test_exporter_from_profiler_report() {
        use crate::profiler::{MemoryEfficiencyAnalysis, PerformanceBottleneck};

        let mut dir = std::env::temp_dir();
        dir.push("perfetto_profiler_report.json");

        let report = ProfilerReport {
            total_events: 2,
            total_runtime: Duration::from_millis(100),
            statistics: HashMap::new(),
            bottlenecks: vec![PerformanceBottleneck {
                bottleneck_type: crate::profiler::BottleneckType::CpuBound,
                location: "attention".to_string(),
                severity: crate::profiler::BottleneckSeverity::Medium,
                description: "CPU saturated".to_string(),
                suggestion: "Use flash attention".to_string(),
                metrics: HashMap::new(),
            }],
            slowest_layers: vec![
                ("attention".to_string(), Duration::from_millis(10)),
                ("ffn".to_string(), Duration::from_millis(15)),
            ],
            memory_efficiency: MemoryEfficiencyAnalysis::default(),
            recommendations: vec![],
        };

        PerfettoExporter::export_profiler_report(&report, &dir).unwrap();
        assert!(dir.exists());

        let content = std::fs::read_to_string(&dir).unwrap();
        let parsed: Value = serde_json::from_str(&content).unwrap();
        let events = parsed["traceEvents"].as_array().unwrap();
        // 2 layer events + 1 bottleneck instant event
        assert_eq!(events.len(), 3);
        assert_eq!(events[2]["ph"], "i");

        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn test_instant_and_counter_phases() {
        let mut trace = PerfettoTrace::new();
        trace.add_event(make_event("checkpoint", PerfettoPhase::Instant, 500, None));
        trace.add_event(make_event("loss", PerfettoPhase::Counter, 600, None));
        let json = trace.export_to_string().unwrap();
        let parsed: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["traceEvents"][0]["ph"], "i");
        assert_eq!(parsed["traceEvents"][1]["ph"], "C");
    }
}
