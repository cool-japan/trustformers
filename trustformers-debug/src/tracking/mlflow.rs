//! MLflow experiment tracking integration.
//!
//! Supports both remote MLflow server (HTTP, future) and local file-based tracking.
//! The local file backend writes MLflow-compatible files to an `mlruns/` directory,
//! mirroring the exact on-disk layout that MLflow itself produces.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Configuration
// ============================================================================

/// MLflow tracking configuration.
#[derive(Debug, Clone)]
pub struct MlflowConfig {
    /// Tracking URI: `"mlruns"` (local directory) or `"http://localhost:5000"` (server).
    pub tracking_uri: String,
    /// Default experiment name.
    pub experiment_name: String,
    /// Request timeout in seconds (reserved for future HTTP mode).
    pub timeout_secs: u64,
}

impl Default for MlflowConfig {
    fn default() -> Self {
        Self {
            tracking_uri: "mlruns".to_string(),
            experiment_name: "Default".to_string(),
            timeout_secs: 10,
        }
    }
}

// ============================================================================
// Value types
// ============================================================================

/// Status of an MLflow run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RunStatus {
    Running,
    Scheduled,
    Finished,
    Failed,
    Killed,
}

impl RunStatus {
    fn as_str(&self) -> &'static str {
        match self {
            RunStatus::Running => "RUNNING",
            RunStatus::Scheduled => "SCHEDULED",
            RunStatus::Finished => "FINISHED",
            RunStatus::Failed => "FAILED",
            RunStatus::Killed => "KILLED",
        }
    }
}

/// A single metric data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub key: String,
    pub value: f64,
    pub timestamp_ms: i64,
    pub step: i64,
}

/// An MLflow run parameter (key/value string pair).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunParam {
    pub key: String,
    pub value: String,
}

/// MLflow run info.
#[derive(Debug, Clone)]
pub struct RunInfo {
    pub run_id: String,
    pub experiment_id: String,
    pub status: RunStatus,
    pub start_time_ms: i64,
    pub end_time_ms: Option<i64>,
    pub artifact_uri: String,
}

// ============================================================================
// Internal helpers
// ============================================================================

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn short_uuid() -> String {
    // Generate a compact 32-char hex UUID without hyphens.
    uuid::Uuid::new_v4().to_string().replace('-', "")
}

fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' || c == '.' { c } else { '_' })
        .collect()
}

// ============================================================================
// Local-file backend
// ============================================================================

/// Writes MLflow-compatible files to `base_path/`.
///
/// Directory layout:
/// ```text
/// {base_path}/
///   {experiment_id}/
///     meta.yaml
///     {run_id}/
///       meta.yaml
///       metrics/{key}      — one line per point: "{timestamp} {step} {value}\n"
///       params/{key}       — file content = value
///       tags/{key}         — file content = value
///       artifacts/
/// ```
struct LocalFileBackend {
    base_path: PathBuf,
}

impl LocalFileBackend {
    fn new(base_path: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&base_path)
            .with_context(|| format!("failed to create mlruns directory: {}", base_path.display()))?;
        Ok(Self { base_path })
    }

    // ---------- experiments ----------

    fn experiment_dir(&self, experiment_id: &str) -> PathBuf {
        self.base_path.join(experiment_id)
    }

    fn create_experiment(&self, name: &str) -> Result<String> {
        // Check if an experiment with this name already exists.
        if let Some(id) = self.find_experiment_by_name(name)? {
            return Ok(id);
        }

        let experiment_id = format!("{}", now_secs() % 1_000_000); // short numeric-ish id
        let dir = self.experiment_dir(&experiment_id);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create experiment directory: {}", dir.display()))?;

        let meta = format!(
            "artifact_location: {}\nexperiment_id: '{}'\nlifecycle_stage: active\nname: {}\n",
            dir.join("artifacts").display(),
            experiment_id,
            name,
        );
        std::fs::write(dir.join("meta.yaml"), meta)
            .context("failed to write experiment meta.yaml")?;

        tracing::debug!(experiment_id = %experiment_id, name = %name, "created mlflow experiment");
        Ok(experiment_id)
    }

    fn find_experiment_by_name(&self, name: &str) -> Result<Option<String>> {
        let rd = match std::fs::read_dir(&self.base_path) {
            Ok(rd) => rd,
            Err(_) => return Ok(None),
        };
        for entry in rd {
            let entry = entry.context("failed to read mlruns directory entry")?;
            let meta_path = entry.path().join("meta.yaml");
            if meta_path.exists() {
                let content = std::fs::read_to_string(&meta_path)
                    .context("failed to read experiment meta.yaml")?;
                // Simple line scan — no YAML library needed.
                if content.lines().any(|l| l.starts_with("name: ") && l[6..].trim() == name) {
                    let id = entry.file_name().to_string_lossy().into_owned();
                    return Ok(Some(id));
                }
            }
        }
        Ok(None)
    }

    // ---------- runs ----------

    fn run_dir(&self, experiment_id: &str, run_id: &str) -> PathBuf {
        self.experiment_dir(experiment_id).join(run_id)
    }

    fn start_run(&self, experiment_id: &str, run_name: Option<&str>) -> Result<String> {
        let run_id = short_uuid();
        let dir = self.run_dir(experiment_id, &run_id);
        for sub in &["metrics", "params", "tags", "artifacts"] {
            std::fs::create_dir_all(dir.join(sub))
                .with_context(|| format!("failed to create run subdirectory: {}", sub))?;
        }

        let name = run_name.unwrap_or("unnamed");
        let start_ms = now_ms();
        let artifact_uri = dir.join("artifacts").to_string_lossy().into_owned();

        let meta = format!(
            "artifact_uri: {artifact_uri}\nexperiment_id: '{experiment_id}'\nrun_id: {run_id}\nrun_name: {name}\nstart_time: {start_ms}\nend_time: null\nstatus: RUNNING\nlifecycle_stage: active\n"
        );
        std::fs::write(dir.join("meta.yaml"), meta).context("failed to write run meta.yaml")?;

        tracing::debug!(run_id = %run_id, experiment_id = %experiment_id, "started mlflow run");
        Ok(run_id)
    }

    fn end_run(&self, experiment_id: &str, run_id: &str, status: &RunStatus) -> Result<()> {
        let dir = self.run_dir(experiment_id, run_id);
        let meta_path = dir.join("meta.yaml");
        let content = std::fs::read_to_string(&meta_path).context("failed to read run meta.yaml")?;

        let end_ms = now_ms();
        let mut new_lines: Vec<String> = Vec::new();
        for line in content.lines() {
            if line.starts_with("end_time:") {
                new_lines.push(format!("end_time: {end_ms}"));
            } else if line.starts_with("status:") {
                new_lines.push(format!("status: {}", status.as_str()));
            } else {
                new_lines.push(line.to_owned());
            }
        }
        new_lines.push(String::new()); // trailing newline
        std::fs::write(&meta_path, new_lines.join("\n")).context("failed to update run meta.yaml")?;

        tracing::debug!(run_id = %run_id, status = ?status, "ended mlflow run");
        Ok(())
    }

    // ---------- metrics ----------

    fn log_metric(
        &self,
        experiment_id: &str,
        run_id: &str,
        key: &str,
        value: f64,
        timestamp_ms: i64,
        step: i64,
    ) -> Result<()> {
        let dir = self.run_dir(experiment_id, run_id);
        let file_path = dir.join("metrics").join(sanitize_filename(key));

        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .with_context(|| format!("failed to open metric file: {}", file_path.display()))?;
        writeln!(f, "{timestamp_ms} {step} {value}")
            .context("failed to write metric line")?;
        Ok(())
    }

    fn get_metrics(&self, experiment_id: &str, run_id: &str) -> Result<Vec<MetricPoint>> {
        let dir = self.run_dir(experiment_id, run_id).join("metrics");
        let mut points = Vec::new();

        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(_) => return Ok(points),
        };

        for entry in rd {
            let entry = entry.context("failed to read metrics directory")?;
            let key = entry.file_name().to_string_lossy().into_owned();
            let content = std::fs::read_to_string(entry.path())
                .context("failed to read metric file")?;
            for line in content.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let timestamp_ms: i64 = parts[0].parse().unwrap_or(0);
                    let step: i64 = parts[1].parse().unwrap_or(0);
                    let value: f64 = parts[2].parse().unwrap_or(0.0);
                    points.push(MetricPoint { key: key.clone(), value, timestamp_ms, step });
                }
            }
        }
        Ok(points)
    }

    // ---------- params ----------

    fn log_param(&self, experiment_id: &str, run_id: &str, key: &str, value: &str) -> Result<()> {
        let dir = self.run_dir(experiment_id, run_id);
        let file_path = dir.join("params").join(sanitize_filename(key));
        std::fs::write(&file_path, value)
            .with_context(|| format!("failed to write param: {key}"))?;
        Ok(())
    }

    // ---------- tags ----------

    fn set_tag(&self, experiment_id: &str, run_id: &str, key: &str, value: &str) -> Result<()> {
        let dir = self.run_dir(experiment_id, run_id);
        let file_path = dir.join("tags").join(sanitize_filename(key));
        std::fs::write(&file_path, value)
            .with_context(|| format!("failed to write tag: {key}"))?;
        Ok(())
    }

    // ---------- artifacts ----------

    fn log_artifact(
        &self,
        experiment_id: &str,
        run_id: &str,
        local_path: &Path,
    ) -> Result<()> {
        let artifacts_dir = self.run_dir(experiment_id, run_id).join("artifacts");
        std::fs::create_dir_all(&artifacts_dir).context("failed to create artifacts directory")?;

        let file_name = local_path
            .file_name()
            .context("artifact path must have a filename")?;
        let dest = artifacts_dir.join(file_name);
        std::fs::copy(local_path, &dest)
            .with_context(|| format!("failed to copy artifact: {}", local_path.display()))?;
        Ok(())
    }

    // ---------- list runs ----------

    fn list_runs(&self, experiment_id: &str) -> Result<Vec<RunInfo>> {
        let exp_dir = self.experiment_dir(experiment_id);
        let mut runs = Vec::new();

        let rd = match std::fs::read_dir(&exp_dir) {
            Ok(rd) => rd,
            Err(_) => return Ok(runs),
        };

        for entry in rd {
            let entry = entry.context("failed to read experiment directory")?;
            let meta_path = entry.path().join("meta.yaml");
            if !meta_path.exists() {
                continue;
            }
            let content =
                std::fs::read_to_string(&meta_path).context("failed to read run meta.yaml")?;

            let mut run_id_opt: Option<String> = None;
            let mut status = RunStatus::Running;
            let mut start_time_ms: i64 = 0;
            let mut end_time_ms: Option<i64> = None;
            let mut artifact_uri = String::new();

            for line in content.lines() {
                if let Some(v) = line.strip_prefix("run_id: ") {
                    run_id_opt = Some(v.trim().to_string());
                } else if let Some(v) = line.strip_prefix("start_time: ") {
                    start_time_ms = v.trim().parse().unwrap_or(0);
                } else if let Some(v) = line.strip_prefix("end_time: ") {
                    let trimmed = v.trim();
                    if trimmed != "null" {
                        end_time_ms = trimmed.parse().ok();
                    }
                } else if let Some(v) = line.strip_prefix("status: ") {
                    status = match v.trim() {
                        "FINISHED" => RunStatus::Finished,
                        "FAILED" => RunStatus::Failed,
                        "KILLED" => RunStatus::Killed,
                        "SCHEDULED" => RunStatus::Scheduled,
                        _ => RunStatus::Running,
                    };
                } else if let Some(v) = line.strip_prefix("artifact_uri: ") {
                    artifact_uri = v.trim().to_string();
                }
            }

            if let Some(run_id) = run_id_opt {
                runs.push(RunInfo {
                    run_id,
                    experiment_id: experiment_id.to_string(),
                    status,
                    start_time_ms,
                    end_time_ms,
                    artifact_uri,
                });
            }
        }
        Ok(runs)
    }
}

// ============================================================================
// Backend enum (extensible)
// ============================================================================

enum MlflowBackend {
    LocalFile(LocalFileBackend),
}

// ============================================================================
// Public MlflowClient
// ============================================================================

/// MLflow tracking client.
pub struct MlflowClient {
    config: MlflowConfig,
    backend: MlflowBackend,
    /// Cache: experiment name → experiment_id
    experiment_cache: std::sync::Mutex<HashMap<String, String>>,
}

impl MlflowClient {
    /// Create a new client.
    ///
    /// If the tracking URI looks like a filesystem path (starts with `.` or `/` or `mlruns`)
    /// the local file backend is used; otherwise the URI is treated as local for now.
    pub fn new(config: MlflowConfig) -> Result<Self> {
        let backend = {
            let base = if config.tracking_uri.starts_with("http://")
                || config.tracking_uri.starts_with("https://")
            {
                // HTTP mode is not yet implemented — fall back to local.
                tracing::warn!("HTTP MLflow backend not implemented; using local file backend");
                PathBuf::from("mlruns")
            } else {
                PathBuf::from(&config.tracking_uri)
            };
            MlflowBackend::LocalFile(LocalFileBackend::new(base)?)
        };

        Ok(Self {
            config,
            backend,
            experiment_cache: std::sync::Mutex::new(HashMap::new()),
        })
    }

    /// Create (or get) an experiment by name. Returns the experiment_id.
    pub fn create_experiment(&self, name: &str) -> Result<String> {
        // Check local cache first.
        {
            let cache = self
                .experiment_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("experiment_cache mutex poisoned"))?;
            if let Some(id) = cache.get(name) {
                return Ok(id.clone());
            }
        }

        let id = match &self.backend {
            MlflowBackend::LocalFile(b) => b.create_experiment(name)?,
        };

        {
            let mut cache = self
                .experiment_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("experiment_cache mutex poisoned"))?;
            cache.insert(name.to_string(), id.clone());
        }
        Ok(id)
    }

    /// Start a new run. Returns the run_id.
    pub fn start_run(&self, experiment_id: &str, run_name: Option<&str>) -> Result<String> {
        match &self.backend {
            MlflowBackend::LocalFile(b) => b.start_run(experiment_id, run_name),
        }
    }

    /// End a run with the given status.
    pub fn end_run(&self, experiment_id: &str, run_id: &str, status: RunStatus) -> Result<()> {
        match &self.backend {
            MlflowBackend::LocalFile(b) => b.end_run(experiment_id, run_id, &status),
        }
    }

    /// Log a single metric.
    pub fn log_metric(
        &self,
        experiment_id: &str,
        run_id: &str,
        key: &str,
        value: f64,
        step: i64,
    ) -> Result<()> {
        let ts = now_ms();
        match &self.backend {
            MlflowBackend::LocalFile(b) => b.log_metric(experiment_id, run_id, key, value, ts, step),
        }
    }

    /// Log multiple metrics at once.
    pub fn log_metrics(
        &self,
        experiment_id: &str,
        run_id: &str,
        metrics: &[MetricPoint],
    ) -> Result<()> {
        for m in metrics {
            match &self.backend {
                MlflowBackend::LocalFile(b) => {
                    b.log_metric(experiment_id, run_id, &m.key, m.value, m.timestamp_ms, m.step)?;
                }
            }
        }
        Ok(())
    }

    /// Log a parameter.
    pub fn log_param(&self, experiment_id: &str, run_id: &str, key: &str, value: &str) -> Result<()> {
        match &self.backend {
            MlflowBackend::LocalFile(b) => b.log_param(experiment_id, run_id, key, value),
        }
    }

    /// Log multiple parameters at once.
    pub fn log_params(
        &self,
        experiment_id: &str,
        run_id: &str,
        params: &[RunParam],
    ) -> Result<()> {
        for p in params {
            self.log_param(experiment_id, run_id, &p.key, &p.value)?;
        }
        Ok(())
    }

    /// Set a tag.
    pub fn set_tag(&self, experiment_id: &str, run_id: &str, key: &str, value: &str) -> Result<()> {
        match &self.backend {
            MlflowBackend::LocalFile(b) => b.set_tag(experiment_id, run_id, key, value),
        }
    }

    /// Log an artifact file.
    pub fn log_artifact(
        &self,
        experiment_id: &str,
        run_id: &str,
        local_path: &Path,
    ) -> Result<()> {
        match &self.backend {
            MlflowBackend::LocalFile(b) => b.log_artifact(experiment_id, run_id, local_path),
        }
    }

    /// Get all metrics logged for a run.
    pub fn get_metrics(&self, experiment_id: &str, run_id: &str) -> Result<Vec<MetricPoint>> {
        match &self.backend {
            MlflowBackend::LocalFile(b) => b.get_metrics(experiment_id, run_id),
        }
    }

    /// List all runs in an experiment.
    pub fn list_runs(&self, experiment_id: &str) -> Result<Vec<RunInfo>> {
        match &self.backend {
            MlflowBackend::LocalFile(b) => b.list_runs(experiment_id),
        }
    }

    /// Return the tracking URI (useful for informational messages).
    pub fn tracking_uri(&self) -> &str {
        &self.config.tracking_uri
    }
}

// ============================================================================
// High-level experiment runner
// ============================================================================

/// High-level convenience wrapper: create an experiment, start a run, and act
/// as the single object for the duration of that run.
pub struct MlflowExperiment {
    client: Arc<MlflowClient>,
    experiment_id: String,
    run_id: String,
}

impl MlflowExperiment {
    /// Create a new experiment and start a run.
    pub fn new(config: MlflowConfig, run_name: Option<&str>) -> Result<Self> {
        let experiment_name = config.experiment_name.clone();
        let client = Arc::new(MlflowClient::new(config)?);
        let experiment_id = client.create_experiment(&experiment_name)?;
        let run_id = client.start_run(&experiment_id, run_name)?;
        Ok(Self { client, experiment_id, run_id })
    }

    /// Log a metric at a given step.
    pub fn log_metric(&self, key: &str, value: f64, step: i64) -> Result<()> {
        self.client.log_metric(&self.experiment_id, &self.run_id, key, value, step)
    }

    /// Log a parameter.
    pub fn log_param(&self, key: &str, value: &str) -> Result<()> {
        self.client.log_param(&self.experiment_id, &self.run_id, key, value)
    }

    /// Set a tag.
    pub fn set_tag(&self, key: &str, value: &str) -> Result<()> {
        self.client.set_tag(&self.experiment_id, &self.run_id, key, value)
    }

    /// Mark the run as finished.
    pub fn finish(self) -> Result<()> {
        self.client.end_run(&self.experiment_id, &self.run_id, RunStatus::Finished)
    }

    /// Mark the run as failed.
    pub fn fail(self) -> Result<()> {
        self.client.end_run(&self.experiment_id, &self.run_id, RunStatus::Failed)
    }

    /// Expose the underlying `run_id`.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Expose the underlying `experiment_id`.
    pub fn experiment_id(&self) -> &str {
        &self.experiment_id
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn temp_config(name: &str) -> MlflowConfig {
        let path = temp_dir().join(format!("mlflow_test_{name}_{}", uuid::Uuid::new_v4()));
        MlflowConfig {
            tracking_uri: path.to_string_lossy().into_owned(),
            experiment_name: "test_experiment".to_string(),
            timeout_secs: 5,
        }
    }

    #[test]
    fn test_create_client() {
        let config = temp_config("create_client");
        let client = MlflowClient::new(config);
        assert!(client.is_ok(), "should create client without error");
    }

    #[test]
    fn test_create_experiment() -> Result<()> {
        let config = temp_config("create_experiment");
        let client = MlflowClient::new(config)?;
        let id = client.create_experiment("my_exp")?;
        assert!(!id.is_empty());
        Ok(())
    }

    #[test]
    fn test_idempotent_experiment_creation() -> Result<()> {
        let config = temp_config("idempotent_exp");
        let client = MlflowClient::new(config)?;
        let id1 = client.create_experiment("same_name")?;
        let id2 = client.create_experiment("same_name")?;
        assert_eq!(id1, id2, "same experiment name must return same id");
        Ok(())
    }

    #[test]
    fn test_start_and_end_run() -> Result<()> {
        let config = temp_config("start_end_run");
        let client = MlflowClient::new(config)?;
        let exp_id = client.create_experiment("e1")?;
        let run_id = client.start_run(&exp_id, Some("run1"))?;
        assert!(!run_id.is_empty());
        client.end_run(&exp_id, &run_id, RunStatus::Finished)?;
        Ok(())
    }

    #[test]
    fn test_log_and_read_metrics() -> Result<()> {
        let config = temp_config("metrics");
        let client = MlflowClient::new(config)?;
        let exp_id = client.create_experiment("e_metrics")?;
        let run_id = client.start_run(&exp_id, None)?;

        client.log_metric(&exp_id, &run_id, "loss", 1.5, 0)?;
        client.log_metric(&exp_id, &run_id, "loss", 1.2, 1)?;
        client.log_metric(&exp_id, &run_id, "acc", 0.9, 0)?;

        let pts = client.get_metrics(&exp_id, &run_id)?;
        let loss_pts: Vec<_> = pts.iter().filter(|p| p.key == "loss").collect();
        assert_eq!(loss_pts.len(), 2);
        Ok(())
    }

    #[test]
    fn test_log_metrics_batch() -> Result<()> {
        let config = temp_config("metrics_batch");
        let client = MlflowClient::new(config)?;
        let exp_id = client.create_experiment("e_batch")?;
        let run_id = client.start_run(&exp_id, None)?;

        let batch = vec![
            MetricPoint { key: "lr".to_string(), value: 0.001, timestamp_ms: 0, step: 0 },
            MetricPoint { key: "lr".to_string(), value: 0.0009, timestamp_ms: 1, step: 1 },
        ];
        client.log_metrics(&exp_id, &run_id, &batch)?;

        let pts = client.get_metrics(&exp_id, &run_id)?;
        assert_eq!(pts.len(), 2);
        Ok(())
    }

    #[test]
    fn test_log_param() -> Result<()> {
        let config = temp_config("log_param");
        let client = MlflowClient::new(config.clone())?;
        let exp_id = client.create_experiment("e_param")?;
        let run_id = client.start_run(&exp_id, None)?;
        client.log_param(&exp_id, &run_id, "batch_size", "32")?;

        // Check the file exists with correct content
        let MlflowBackend::LocalFile(ref b) = client.backend;
        let param_path = b.run_dir(&exp_id, &run_id).join("params").join("batch_size");
        let val = std::fs::read_to_string(param_path)?;
        assert_eq!(val.trim(), "32");
        Ok(())
    }

    #[test]
    fn test_log_params_batch() -> Result<()> {
        let config = temp_config("params_batch");
        let client = MlflowClient::new(config)?;
        let exp_id = client.create_experiment("e_params")?;
        let run_id = client.start_run(&exp_id, None)?;

        let params = vec![
            RunParam { key: "lr".to_string(), value: "0.01".to_string() },
            RunParam { key: "optimizer".to_string(), value: "adam".to_string() },
        ];
        client.log_params(&exp_id, &run_id, &params)?;
        Ok(())
    }

    #[test]
    fn test_set_tag() -> Result<()> {
        let config = temp_config("set_tag");
        let client = MlflowClient::new(config.clone())?;
        let exp_id = client.create_experiment("e_tag")?;
        let run_id = client.start_run(&exp_id, None)?;
        client.set_tag(&exp_id, &run_id, "model_type", "transformer")?;

        let MlflowBackend::LocalFile(ref b) = client.backend;
        let tag_path = b.run_dir(&exp_id, &run_id).join("tags").join("model_type");
        let val = std::fs::read_to_string(tag_path)?;
        assert_eq!(val.trim(), "transformer");
        Ok(())
    }

    #[test]
    fn test_log_artifact() -> Result<()> {
        let config = temp_config("artifact");
        let client = MlflowClient::new(config)?;
        let exp_id = client.create_experiment("e_art")?;
        let run_id = client.start_run(&exp_id, None)?;

        // Create a temp file to log
        let tmp = temp_dir().join("artifact_test.txt");
        std::fs::write(&tmp, "model data")?;
        client.log_artifact(&exp_id, &run_id, &tmp)?;

        let MlflowBackend::LocalFile(ref b) = client.backend;
        let dest = b.run_dir(&exp_id, &run_id).join("artifacts").join("artifact_test.txt");
        assert!(dest.exists());
        Ok(())
    }

    #[test]
    fn test_list_runs() -> Result<()> {
        let config = temp_config("list_runs");
        let client = MlflowClient::new(config)?;
        let exp_id = client.create_experiment("e_list")?;
        let run_id_1 = client.start_run(&exp_id, Some("r1"))?;
        let run_id_2 = client.start_run(&exp_id, Some("r2"))?;
        client.end_run(&exp_id, &run_id_1, RunStatus::Finished)?;
        client.end_run(&exp_id, &run_id_2, RunStatus::Failed)?;

        let runs = client.list_runs(&exp_id)?;
        assert_eq!(runs.len(), 2);
        let statuses: Vec<_> = runs.iter().map(|r| &r.status).collect();
        assert!(statuses.contains(&&RunStatus::Finished));
        assert!(statuses.contains(&&RunStatus::Failed));
        Ok(())
    }

    #[test]
    fn test_mlflow_experiment_high_level() -> Result<()> {
        let config = temp_config("high_level");
        let exp = MlflowExperiment::new(config, Some("my_run"))?;
        exp.log_param("epochs", "10")?;
        exp.log_metric("loss", 0.42, 0)?;
        exp.log_metric("loss", 0.38, 1)?;
        exp.set_tag("framework", "trustformers")?;
        exp.finish()?;
        Ok(())
    }

    #[test]
    fn test_mlflow_experiment_failure_path() -> Result<()> {
        let config = temp_config("failure_path");
        let exp = MlflowExperiment::new(config, None)?;
        exp.log_param("seed", "42")?;
        exp.fail()?;
        Ok(())
    }
}
