//! Tracy profiler offline trace export
//!
//! Serialises profiling data to Tracy's text-based CSV/log format for
//! offline analysis with Tracy's viewer or compatible tools.
//!
//! Format reference (line-oriented, one record per line):
//! ```text
//! ZoneBegin,<name>,<file>,<line>,<timestamp_ns>
//! ZoneEnd,<timestamp_ns>
//! Message,<text>,<timestamp_ns>
//! Plot,<name>,<value>,<timestamp_ns>
//! ```
//!
//! # Example
//!
//! ```no_run
//! use trustformers_debug::export::tracy::{TracyTrace, TracyZone};
//!
//! let mut trace = TracyTrace::new();
//! trace.add_zone(TracyZone {
//!     name: "attention_forward".to_string(),
//!     timestamp_ns: 0,
//!     duration_ns: 1_500_000,
//!     thread_id: 1,
//! });
//! trace.add_message("training started", 0);
//! trace.add_plot("loss", 0.345, 0);
//! let path = std::path::Path::new("/tmp/trace.tracy.csv");
//! trace.export_to_file(path).unwrap();
//! ```

use std::io::Write;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::ProfilerReport;

// ─────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────

/// A single profiling zone recorded in a Tracy trace.
///
/// A zone corresponds to one timed code region (e.g., a model layer
/// forward pass).
///
/// # Example
///
/// ```
/// use trustformers_debug::export::tracy::TracyZone;
///
/// let zone = TracyZone {
///     name: "ffn_forward".to_string(),
///     timestamp_ns: 1_000_000,
///     duration_ns: 500_000,
///     thread_id: 0,
/// };
/// assert_eq!(zone.end_timestamp_ns(), 1_500_000);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracyZone {
    /// Human-readable zone name.
    pub name: String,
    /// Start timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Duration in nanoseconds.
    pub duration_ns: u64,
    /// Thread identifier.
    pub thread_id: u32,
}

impl TracyZone {
    /// Returns the end timestamp (start + duration) in nanoseconds.
    pub fn end_timestamp_ns(&self) -> u64 {
        self.timestamp_ns.saturating_add(self.duration_ns)
    }
}

/// In-memory collection of Tracy trace records.
///
/// Holds zones, text messages, and numeric plot entries that can be
/// exported to a CSV file understood by Tracy-compatible tooling.
///
/// # Example
///
/// ```
/// use trustformers_debug::export::tracy::TracyTrace;
///
/// let trace = TracyTrace::new();
/// assert!(trace.is_empty());
/// ```
#[derive(Debug, Default)]
pub struct TracyTrace {
    zones: Vec<TracyZone>,
    messages: Vec<(String, u64)>,
    plots: Vec<(String, f64, u64)>,
}

impl TracyTrace {
    /// Creates an empty trace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if the trace contains no records of any kind.
    pub fn is_empty(&self) -> bool {
        self.zones.is_empty() && self.messages.is_empty() && self.plots.is_empty()
    }

    /// Returns the total number of records (zones + messages + plots).
    pub fn total_records(&self) -> usize {
        self.zones.len() + self.messages.len() + self.plots.len()
    }

    /// Appends a profiling zone.
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::export::tracy::{TracyTrace, TracyZone};
    ///
    /// let mut trace = TracyTrace::new();
    /// trace.add_zone(TracyZone {
    ///     name: "forward".to_string(),
    ///     timestamp_ns: 0,
    ///     duration_ns: 1000,
    ///     thread_id: 0,
    /// });
    /// assert_eq!(trace.zones().len(), 1);
    /// ```
    pub fn add_zone(&mut self, zone: TracyZone) {
        self.zones.push(zone);
    }

    /// Appends a text message with a timestamp.
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::export::tracy::TracyTrace;
    ///
    /// let mut trace = TracyTrace::new();
    /// trace.add_message("epoch started", 1_000_000);
    /// assert_eq!(trace.messages().len(), 1);
    /// ```
    pub fn add_message(&mut self, msg: &str, timestamp_ns: u64) {
        self.messages.push((msg.to_string(), timestamp_ns));
    }

    /// Appends a named numeric plot value with a timestamp.
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::export::tracy::TracyTrace;
    ///
    /// let mut trace = TracyTrace::new();
    /// trace.add_plot("loss", 0.42, 2_000_000);
    /// assert_eq!(trace.plots().len(), 1);
    /// ```
    pub fn add_plot(&mut self, name: &str, value: f64, timestamp_ns: u64) {
        self.plots.push((name.to_string(), value, timestamp_ns));
    }

    /// Read-only view of the recorded zones.
    pub fn zones(&self) -> &[TracyZone] {
        &self.zones
    }

    /// Read-only view of the recorded messages.
    pub fn messages(&self) -> &[(String, u64)] {
        &self.messages
    }

    /// Read-only view of the recorded plot entries.
    pub fn plots(&self) -> &[(String, f64, u64)] {
        &self.plots
    }

    /// Writes the trace to `path` in Tracy CSV format.
    ///
    /// Records are emitted in the order: zones (each zone becomes a
    /// `ZoneBegin` / `ZoneEnd` pair), messages, then plots.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or written.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_debug::export::tracy::TracyTrace;
    ///
    /// let trace = TracyTrace::new();
    /// trace.export_to_file(std::path::Path::new("/tmp/trace.csv")).unwrap();
    /// ```
    pub fn export_to_file(&self, path: &std::path::Path) -> Result<()> {
        let mut file = std::fs::File::create(path)?;

        // Header comment
        writeln!(file, "# TracyTrace export — generated by trustformers-debug")?;

        for zone in &self.zones {
            writeln!(
                file,
                "ZoneBegin,{},{},0,{}",
                zone.name, zone.name, zone.timestamp_ns
            )?;
            writeln!(file, "ZoneEnd,{}", zone.end_timestamp_ns())?;
        }

        for (msg, ts) in &self.messages {
            // Escape commas inside the message to keep CSV parseable
            let safe_msg = msg.replace(',', "\\,");
            writeln!(file, "Message,{},{}", safe_msg, ts)?;
        }

        for (name, value, ts) in &self.plots {
            writeln!(file, "Plot,{},{},{}", name, value, ts)?;
        }

        tracing::debug!("Tracy trace written to {}", path.display());
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────
// TracyExporter
// ─────────────────────────────────────────────────────────────

/// Converts a [`ProfilerReport`] to a [`TracyTrace`] and writes it to disk.
///
/// # Example
///
/// ```no_run
/// use trustformers_debug::export::tracy::TracyExporter;
/// use trustformers_debug::ProfilerReport;
/// use std::collections::HashMap;
/// use std::time::Duration;
///
/// let report = ProfilerReport {
///     total_events: 0,
///     total_runtime: Duration::from_millis(0),
///     statistics: HashMap::new(),
///     bottlenecks: vec![],
///     slowest_layers: vec![],
///     memory_efficiency: Default::default(),
///     recommendations: vec![],
/// };
/// TracyExporter::export_profiler_report(
///     &report,
///     std::path::Path::new("/tmp/report.csv"),
/// ).unwrap();
/// ```
pub struct TracyExporter;

impl TracyExporter {
    /// Converts a [`ProfilerReport`] into a Tracy CSV trace file.
    ///
    /// Slowest layers become `TracyZone`s ordered by appearance.
    /// Recommendations are recorded as text messages.
    /// Per-event statistics are emitted as plots.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn export_profiler_report(
        report: &ProfilerReport,
        path: &std::path::Path,
    ) -> Result<()> {
        let mut trace = TracyTrace::new();
        let mut cursor_ns: u64 = 0;

        // Zones from slowest-layer list
        for (layer_name, duration) in &report.slowest_layers {
            let dur_ns = duration.as_nanos() as u64;
            trace.add_zone(TracyZone {
                name: layer_name.clone(),
                timestamp_ns: cursor_ns,
                duration_ns: dur_ns,
                thread_id: 0,
            });
            cursor_ns += dur_ns;
        }

        // Recommendations as messages
        for (idx, rec) in report.recommendations.iter().enumerate() {
            trace.add_message(rec.as_str(), idx as u64 * 1_000);
        }

        // Statistics as plots
        for (name, stats) in &report.statistics {
            let avg_us = stats.avg_duration.as_micros() as f64;
            trace.add_plot(name, avg_us, cursor_ns);
        }

        trace.export_to_file(path)
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::Duration;

    #[test]
    fn test_empty_trace() {
        let trace = TracyTrace::new();
        assert!(trace.is_empty());
        assert_eq!(trace.total_records(), 0);
    }

    #[test]
    fn test_add_zone() {
        let mut trace = TracyTrace::new();
        trace.add_zone(TracyZone {
            name: "attention".to_string(),
            timestamp_ns: 1000,
            duration_ns: 500,
            thread_id: 0,
        });
        assert_eq!(trace.zones().len(), 1);
        assert_eq!(trace.zones()[0].end_timestamp_ns(), 1500);
    }

    #[test]
    fn test_add_message_and_plot() {
        let mut trace = TracyTrace::new();
        trace.add_message("hello", 42);
        trace.add_plot("loss", 0.5, 100);
        assert_eq!(trace.messages().len(), 1);
        assert_eq!(trace.plots().len(), 1);
        assert_eq!(trace.total_records(), 2);
    }

    #[test]
    fn test_export_to_file() {
        let mut path = std::env::temp_dir();
        path.push("tracy_test_trace.csv");

        let mut trace = TracyTrace::new();
        trace.add_zone(TracyZone {
            name: "ffn".to_string(),
            timestamp_ns: 0,
            duration_ns: 2_000_000,
            thread_id: 1,
        });
        trace.add_message("epoch start", 0);
        trace.add_plot("loss", 0.42, 2_000_000);

        trace.export_to_file(&path).unwrap();
        assert!(path.exists());

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("ZoneBegin,ffn"));
        assert!(content.contains("ZoneEnd,2000000"));
        assert!(content.contains("Message,epoch start,0"));
        assert!(content.contains("Plot,loss,0.42,2000000"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_exporter_from_profiler_report() {
        use crate::profiler::MemoryEfficiencyAnalysis;

        let mut path = std::env::temp_dir();
        path.push("tracy_profiler_report.csv");

        let report = ProfilerReport {
            total_events: 3,
            total_runtime: Duration::from_millis(50),
            statistics: HashMap::new(),
            bottlenecks: vec![],
            slowest_layers: vec![
                ("attn".to_string(), Duration::from_millis(20)),
                ("ffn".to_string(), Duration::from_millis(30)),
            ],
            memory_efficiency: MemoryEfficiencyAnalysis::default(),
            recommendations: vec!["Use flash attention".to_string()],
        };

        TracyExporter::export_profiler_report(&report, &path).unwrap();
        assert!(path.exists());

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("ZoneBegin,attn"));
        assert!(content.contains("ZoneBegin,ffn"));
        assert!(content.contains("Message,Use flash attention"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_comma_escaping_in_message() {
        let mut path = std::env::temp_dir();
        path.push("tracy_comma_test.csv");

        let mut trace = TracyTrace::new();
        trace.add_message("loss, accuracy: 0.9", 0);
        trace.export_to_file(&path).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        // commas inside the message must be escaped
        assert!(content.contains("loss\\, accuracy: 0.9"));

        std::fs::remove_file(&path).ok();
    }
}
