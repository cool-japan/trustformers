//! Unified experiment tracking interface and in-memory backend.
//!
//! Defines the [`ExperimentTracker`] trait and provides [`InMemoryTracker`]
//! for testing, offline training runs, or as a composable building block for
//! multi-backend dispatchers.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

// ─────────────────────────────────────────────────────────────────────────────
// Supporting types
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque run identifier — a plain string for maximum compatibility.
pub type RunId = String;

/// Status of an experiment run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrackerRunStatus {
    /// The run is currently in progress.
    Running,
    /// The run completed without errors.
    Finished,
    /// The run terminated with an error.
    Failed,
    /// The run was forcibly stopped.
    Killed,
}

impl TrackerRunStatus {
    fn as_str(&self) -> &str {
        match self {
            Self::Running => "running",
            Self::Finished => "finished",
            Self::Failed => "failed",
            Self::Killed => "killed",
        }
    }
}

/// Errors returned by [`ExperimentTracker`] implementations.
#[derive(Debug, Clone, PartialEq)]
pub enum TrackingError {
    /// An operation was attempted without an active run.
    NoActiveRun,
    /// [`ExperimentTracker::start_run`] was called while a run was already active.
    AlreadyStarted,
    /// An I/O error occurred (path is included in the message).
    IoError(String),
    /// A value was invalid (e.g. NaN metric, empty key).
    InvalidValue(String),
}

impl std::fmt::Display for TrackingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoActiveRun => write!(f, "no active run — call start_run() first"),
            Self::AlreadyStarted => write!(f, "a run is already in progress"),
            Self::IoError(msg) => write!(f, "tracking I/O error: {msg}"),
            Self::InvalidValue(msg) => write!(f, "invalid tracking value: {msg}"),
        }
    }
}

impl std::error::Error for TrackingError {}

// ─────────────────────────────────────────────────────────────────────────────
// ExperimentTracker trait
// ─────────────────────────────────────────────────────────────────────────────

/// Unified interface for experiment tracking backends.
///
/// Implementations must be consistent: most methods require an active run
/// (started via [`start_run`](Self::start_run)) and return
/// [`TrackingError::NoActiveRun`] otherwise.
///
/// # Example
///
/// ```
/// use trustformers_debug::tracking::tracker::{ExperimentTracker, InMemoryTracker, TrackerRunStatus};
///
/// let mut tracker = InMemoryTracker::new();
/// tracker.start_run("my_run").unwrap();
/// tracker.log_param("lr", "0.001").unwrap();
/// tracker.log_metric("loss", 1.5, 0).unwrap();
/// tracker.log_metric("loss", 1.2, 1).unwrap();
/// tracker.end_run(TrackerRunStatus::Finished).unwrap();
/// assert_eq!(tracker.get_metric_history("loss").map(|v| v.len()), Some(2));
/// ```
pub trait ExperimentTracker {
    /// Logs a scalar metric at the given step.
    fn log_metric(&mut self, key: &str, value: f64, step: u64) -> Result<(), TrackingError>;

    /// Logs a string parameter (key/value pair).
    fn log_param(&mut self, key: &str, value: &str) -> Result<(), TrackingError>;

    /// Logs an artifact file path.  `artifact_path` optionally overrides the
    /// subdirectory name used to store the artifact.
    fn log_artifact(
        &mut self,
        local_path: &str,
        artifact_path: Option<&str>,
    ) -> Result<(), TrackingError>;

    /// Sets one or more string tags on the current run.
    fn set_tags(&mut self, tags: &[(&str, &str)]) -> Result<(), TrackingError>;

    /// Starts a new run with the given name.  Returns the run's unique ID.
    fn start_run(&mut self, run_name: &str) -> Result<RunId, TrackingError>;

    /// Finishes the current run with the given status.
    fn end_run(&mut self, status: TrackerRunStatus) -> Result<(), TrackingError>;

    /// Returns the ID of the currently active run, or `None`.
    fn get_run_id(&self) -> Option<&RunId>;
}

// ─────────────────────────────────────────────────────────────────────────────
// TrackingSummary
// ─────────────────────────────────────────────────────────────────────────────

/// High-level summary of an [`InMemoryTracker`] session.
#[derive(Debug, Clone)]
pub struct TrackingSummary {
    /// The active run ID, if any.
    pub run_id: Option<RunId>,
    /// Number of distinct metric keys logged.
    pub total_metrics: usize,
    /// Number of distinct parameters logged.
    pub total_params: usize,
    /// Number of artifact paths logged.
    pub total_artifacts: usize,
    /// Human-readable run status (e.g. `"running"`, `"finished"`).
    pub status: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// InMemoryTracker
// ─────────────────────────────────────────────────────────────────────────────

/// A fully in-memory experiment tracker.
///
/// Suitable for testing, offline training, and as a drop-in replacement when
/// no external tracking service is available.
///
/// All data lives in `HashMap`s and is discarded when the tracker is dropped.
/// Use [`InMemoryTracker::export_to_json`] to persist the contents.
pub struct InMemoryTracker {
    run_id: Option<RunId>,
    /// Metric series: key → list of (step, value) pairs in insertion order.
    metrics: HashMap<String, Vec<(u64, f64)>>,
    params: HashMap<String, String>,
    tags: HashMap<String, String>,
    artifacts: Vec<String>,
    status: Option<TrackerRunStatus>,
}

impl InMemoryTracker {
    /// Creates a new empty tracker with no active run.
    pub fn new() -> Self {
        Self {
            run_id: None,
            metrics: HashMap::new(),
            params: HashMap::new(),
            tags: HashMap::new(),
            artifacts: Vec::new(),
            status: None,
        }
    }

    /// Returns the full history of `(step, value)` pairs for `key`, or `None`.
    pub fn get_metric_history(&self, key: &str) -> Option<&Vec<(u64, f64)>> {
        self.metrics.get(key)
    }

    /// Returns the stored value for a parameter key, or `None`.
    pub fn get_param(&self, key: &str) -> Option<&str> {
        self.params.get(key).map(|s| s.as_str())
    }

    /// Returns the number of distinct metric keys logged so far.
    pub fn metric_count(&self) -> usize {
        self.metrics.len()
    }

    /// Exports the full tracker state to a compact JSON string.
    ///
    /// The format is:
    /// ```json
    /// {
    ///   "run_id": "...",
    ///   "status": "...",
    ///   "params": { "key": "value", ... },
    ///   "tags": { "key": "value", ... },
    ///   "artifacts": ["..."],
    ///   "metrics": { "loss": [[0, 1.5], [1, 1.2]], ... }
    /// }
    /// ```
    pub fn export_to_json(&self) -> String {
        let run_id_json = self
            .run_id
            .as_deref()
            .map(|id| format!("\"{}\"", escape_json(id)))
            .unwrap_or_else(|| "null".to_string());

        let status_str = self
            .status
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("null");
        let status_json = if status_str == "null" {
            "null".to_string()
        } else {
            format!("\"{}\"", status_str)
        };

        let params_json = map_to_json_obj(&self.params);
        let tags_json = map_to_json_obj(&self.tags);

        let artifacts_json = {
            let mut out = String::from('[');
            for (i, a) in self.artifacts.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let _ = write!(out, "\"{}\"", escape_json(a));
            }
            out.push(']');
            out
        };

        let metrics_json = {
            let mut out = String::from('{');
            for (i, (key, series)) in self.metrics.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let _ = write!(out, "\"{}\":[", escape_json(key));
                for (j, (step, val)) in series.iter().enumerate() {
                    if j > 0 {
                        out.push(',');
                    }
                    let _ = write!(out, "[{step},{val}]");
                }
                out.push(']');
            }
            out.push('}');
            out
        };

        format!(
            r#"{{"run_id":{run_id_json},"status":{status_json},"params":{params_json},"tags":{tags_json},"artifacts":{artifacts_json},"metrics":{metrics_json}}}"#
        )
    }

    /// Returns a high-level summary of the current tracking state.
    pub fn to_summary(&self) -> TrackingSummary {
        TrackingSummary {
            run_id: self.run_id.clone(),
            total_metrics: self.metrics.len(),
            total_params: self.params.len(),
            total_artifacts: self.artifacts.len(),
            status: self
                .status
                .as_ref()
                .map(|s| s.as_str().to_string())
                .unwrap_or_else(|| "none".to_string()),
        }
    }

    /// Returns all logged artifact paths.
    pub fn artifacts(&self) -> &[String] {
        &self.artifacts
    }

    /// Returns all logged tags as a reference to the inner map.
    pub fn tags(&self) -> &HashMap<String, String> {
        &self.tags
    }
}

impl ExperimentTracker for InMemoryTracker {
    fn start_run(&mut self, run_name: &str) -> Result<RunId, TrackingError> {
        if self.run_id.is_some() {
            return Err(TrackingError::AlreadyStarted);
        }
        if run_name.is_empty() {
            return Err(TrackingError::InvalidValue("run_name must not be empty".to_string()));
        }
        // Generate a deterministic-ish ID from the name + a counter.
        let id = format!("run_{}", run_name.replace(' ', "_"));
        self.run_id = Some(id.clone());
        self.status = Some(TrackerRunStatus::Running);
        Ok(id)
    }

    fn end_run(&mut self, status: TrackerRunStatus) -> Result<(), TrackingError> {
        if self.run_id.is_none() {
            return Err(TrackingError::NoActiveRun);
        }
        self.status = Some(status);
        Ok(())
    }

    fn log_metric(&mut self, key: &str, value: f64, step: u64) -> Result<(), TrackingError> {
        if self.run_id.is_none() {
            return Err(TrackingError::NoActiveRun);
        }
        if key.is_empty() {
            return Err(TrackingError::InvalidValue("metric key must not be empty".to_string()));
        }
        if !value.is_finite() {
            return Err(TrackingError::InvalidValue(format!(
                "metric value for '{key}' is not finite: {value}"
            )));
        }
        self.metrics.entry(key.to_string()).or_default().push((step, value));
        Ok(())
    }

    fn log_param(&mut self, key: &str, value: &str) -> Result<(), TrackingError> {
        if self.run_id.is_none() {
            return Err(TrackingError::NoActiveRun);
        }
        if key.is_empty() {
            return Err(TrackingError::InvalidValue("param key must not be empty".to_string()));
        }
        self.params.insert(key.to_string(), value.to_string());
        Ok(())
    }

    fn log_artifact(
        &mut self,
        local_path: &str,
        _artifact_path: Option<&str>,
    ) -> Result<(), TrackingError> {
        if self.run_id.is_none() {
            return Err(TrackingError::NoActiveRun);
        }
        if local_path.is_empty() {
            return Err(TrackingError::InvalidValue("artifact path must not be empty".to_string()));
        }
        self.artifacts.push(local_path.to_string());
        Ok(())
    }

    fn set_tags(&mut self, tags: &[(&str, &str)]) -> Result<(), TrackingError> {
        if self.run_id.is_none() {
            return Err(TrackingError::NoActiveRun);
        }
        for (k, v) in tags {
            if k.is_empty() {
                return Err(TrackingError::InvalidValue("tag key must not be empty".to_string()));
            }
            self.tags.insert(k.to_string(), v.to_string());
        }
        Ok(())
    }

    fn get_run_id(&self) -> Option<&RunId> {
        self.run_id.as_ref()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn map_to_json_obj(map: &HashMap<String, String>) -> String {
    let mut out = String::from('{');
    for (i, (k, v)) in map.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "\"{}\":\"{}\"", escape_json(k), escape_json(v));
    }
    out.push('}');
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn started_tracker() -> InMemoryTracker {
        let mut t = InMemoryTracker::new();
        t.start_run("test_run").unwrap();
        t
    }

    // ── start_run / end_run ──────────────────────────────────────────────────

    #[test]
    fn test_start_run_returns_id() {
        let mut t = InMemoryTracker::new();
        let id = t.start_run("training").unwrap();
        assert!(!id.is_empty());
        assert_eq!(t.get_run_id(), Some(&id));
    }

    #[test]
    fn test_start_run_already_started() {
        let mut t = started_tracker();
        let err = t.start_run("second_run");
        assert!(matches!(err, Err(TrackingError::AlreadyStarted)));
    }

    #[test]
    fn test_start_run_empty_name_rejected() {
        let mut t = InMemoryTracker::new();
        assert!(matches!(t.start_run(""), Err(TrackingError::InvalidValue(_))));
    }

    #[test]
    fn test_end_run_no_active_run() {
        let mut t = InMemoryTracker::new();
        assert!(matches!(t.end_run(TrackerRunStatus::Finished), Err(TrackingError::NoActiveRun)));
    }

    #[test]
    fn test_end_run_sets_status() {
        let mut t = started_tracker();
        t.end_run(TrackerRunStatus::Failed).unwrap();
        let summary = t.to_summary();
        assert_eq!(summary.status, "failed");
    }

    // ── log_metric ────────────────────────────────────────────────────────────

    #[test]
    fn test_log_metric_basic() {
        let mut t = started_tracker();
        t.log_metric("loss", 1.5, 0).unwrap();
        t.log_metric("loss", 1.2, 1).unwrap();
        let hist = t.get_metric_history("loss").unwrap();
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0], (0, 1.5));
        assert_eq!(hist[1], (1, 1.2));
    }

    #[test]
    fn test_log_metric_no_run() {
        let mut t = InMemoryTracker::new();
        assert!(matches!(t.log_metric("x", 1.0, 0), Err(TrackingError::NoActiveRun)));
    }

    #[test]
    fn test_log_metric_nan_rejected() {
        let mut t = started_tracker();
        assert!(matches!(t.log_metric("x", f64::NAN, 0), Err(TrackingError::InvalidValue(_))));
    }

    #[test]
    fn test_log_metric_inf_rejected() {
        let mut t = started_tracker();
        assert!(matches!(t.log_metric("x", f64::INFINITY, 0), Err(TrackingError::InvalidValue(_))));
    }

    #[test]
    fn test_log_metric_empty_key_rejected() {
        let mut t = started_tracker();
        assert!(matches!(t.log_metric("", 1.0, 0), Err(TrackingError::InvalidValue(_))));
    }

    // ── log_param ─────────────────────────────────────────────────────────────

    #[test]
    fn test_log_param_and_retrieve() {
        let mut t = started_tracker();
        t.log_param("lr", "0.001").unwrap();
        assert_eq!(t.get_param("lr"), Some("0.001"));
    }

    #[test]
    fn test_log_param_overwrite() {
        let mut t = started_tracker();
        t.log_param("lr", "0.001").unwrap();
        t.log_param("lr", "0.0001").unwrap();
        assert_eq!(t.get_param("lr"), Some("0.0001"));
    }

    #[test]
    fn test_log_param_no_run() {
        let mut t = InMemoryTracker::new();
        assert!(matches!(t.log_param("k", "v"), Err(TrackingError::NoActiveRun)));
    }

    // ── log_artifact ──────────────────────────────────────────────────────────

    #[test]
    fn test_log_artifact_stored() {
        let mut t = started_tracker();
        t.log_artifact("/tmp/model.bin", None).unwrap();
        assert_eq!(t.artifacts(), &["/tmp/model.bin".to_string()]);
    }

    #[test]
    fn test_log_artifact_no_run() {
        let mut t = InMemoryTracker::new();
        assert!(matches!(t.log_artifact("/tmp/x", None), Err(TrackingError::NoActiveRun)));
    }

    // ── set_tags ──────────────────────────────────────────────────────────────

    #[test]
    fn test_set_tags() {
        let mut t = started_tracker();
        t.set_tags(&[("framework", "trustformers"), ("version", "0.1.0")]).unwrap();
        assert_eq!(t.tags().get("framework").map(|s| s.as_str()), Some("trustformers"));
    }

    #[test]
    fn test_set_tags_no_run() {
        let mut t = InMemoryTracker::new();
        assert!(matches!(t.set_tags(&[("k", "v")]), Err(TrackingError::NoActiveRun)));
    }

    // ── metric_count ──────────────────────────────────────────────────────────

    #[test]
    fn test_metric_count() {
        let mut t = started_tracker();
        t.log_metric("loss", 1.0, 0).unwrap();
        t.log_metric("acc", 0.9, 0).unwrap();
        assert_eq!(t.metric_count(), 2);
    }

    // ── to_summary ────────────────────────────────────────────────────────────

    #[test]
    fn test_summary_fields() {
        let mut t = started_tracker();
        t.log_metric("loss", 0.5, 0).unwrap();
        t.log_param("epochs", "5").unwrap();
        t.log_artifact("/tmp/ckpt.bin", None).unwrap();
        let s = t.to_summary();
        assert!(s.run_id.is_some());
        assert_eq!(s.total_metrics, 1);
        assert_eq!(s.total_params, 1);
        assert_eq!(s.total_artifacts, 1);
        assert_eq!(s.status, "running");
    }

    // ── export_to_json ────────────────────────────────────────────────────────

    #[test]
    fn test_export_json_structure() {
        let mut t = started_tracker();
        t.log_metric("loss", 2.0, 0).unwrap();
        t.log_param("lr", "0.01").unwrap();
        t.set_tags(&[("model", "gpt2")]).unwrap();
        t.log_artifact("/tmp/weights.bin", None).unwrap();
        let json = t.export_to_json();
        assert!(json.contains("\"run_id\""));
        assert!(json.contains("\"params\""));
        assert!(json.contains("\"metrics\""));
        assert!(json.contains("\"tags\""));
        assert!(json.contains("\"artifacts\""));
        assert!(json.contains("\"loss\""));
        assert!(json.contains("\"lr\""));
        assert!(json.contains("gpt2"));
    }

    #[test]
    fn test_export_json_no_run() {
        let t = InMemoryTracker::new();
        let json = t.export_to_json();
        assert!(json.contains("\"run_id\":null"));
        assert!(json.contains("\"status\":null"));
    }

    // ── TrackingError display ─────────────────────────────────────────────────

    #[test]
    fn test_tracking_error_display() {
        assert!(TrackingError::NoActiveRun.to_string().contains("no active run"));
        assert!(TrackingError::AlreadyStarted.to_string().contains("already in progress"));
        assert!(TrackingError::IoError("disk full".to_string()).to_string().contains("disk full"));
        assert!(TrackingError::InvalidValue("nan".to_string()).to_string().contains("nan"));
    }
}
