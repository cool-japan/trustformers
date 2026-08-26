//! MLflow Integration for Experiment Tracking
//!
//! Implements the real MLflow REST API 2.0 (create experiment/run, log
//! metrics/params/tags, set terminated status) over `reqwest`, gated behind
//! the `http-integrations` Cargo feature (see `Cargo.toml`) since it pulls a
//! non-pure-Rust TLS stack into the build. `metrics`/`params`/`tags` are
//! cached locally as they're logged (unchanged from before) and delivered
//! to the tracking server in a real REST call from [`MLflowClient::flush`]
//! and [`MLflowClient::end_run`] -- never fabricated, and never silently
//! dropped.
//!
//! [`TrackingMode::LocalOnly`] is available for offline/test use, but must
//! be selected explicitly via [`MLflowConfig::mode`]; the default
//! ([`TrackingMode::Http`]) always attempts a real connection and surfaces
//! a real error if the tracking server is unreachable, rather than quietly
//! doing nothing.
//!
//! # Scope: artifacts are local-only
//!
//! [`MLflowClient::log_artifact`] (and [`MLflowClient::log_model`] /
//! [`MLflowClient::log_plot`] / [`MLflowClient::log_report`], which are
//! built on it) only ever writes into the local `artifact_dir` cache --
//! never to the remote tracking server's artifact store, even in
//! [`TrackingMode::Http`]. MLflow's artifact backend is
//! deployment-specific (local disk, S3, GCS, Azure Blob, DBFS, ...) with no
//! single REST endpoint to target, so uploading real artifact bytes is
//! explicitly out of scope for this client.

use anyhow::{Context, Result};
use parking_lot::RwLock;
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use trustformers_core::tensor::Tensor;

/// How an [`MLflowClient`] talks to a tracking server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TrackingMode {
    /// Make real MLflow REST API calls against `tracking_uri`. The default:
    /// matches the historical default `tracking_uri` of
    /// `http://localhost:5000`, and fails with a real connection/transport
    /// error (rather than silently) if no server is reachable there.
    #[default]
    Http,
    /// Never contact a server; everything stays in the in-memory caches
    /// until read back via [`MLflowClient::get_metrics`] /
    /// [`MLflowClient::get_params`]. Must be selected explicitly -- this
    /// client never falls back to it silently.
    LocalOnly,
}

/// MLflow client for experiment tracking
#[derive(Debug)]
pub struct MLflowClient {
    /// MLflow tracking URI
    tracking_uri: String,
    /// Current experiment ID
    experiment_id: Option<String>,
    /// Current run ID
    run_id: Option<String>,
    /// Name of the current run, set by [`MLflowClient::start_run`].
    run_name: Option<String>,
    /// Wall-clock start time (ms since epoch) of the current run.
    run_start_time: Option<i64>,
    /// Configuration
    config: MLflowConfig,
    /// Cached metrics, flushed to the tracking server by
    /// [`MLflowClient::flush`] / [`MLflowClient::end_run`].
    metrics_cache: Arc<RwLock<HashMap<String, Vec<MetricPoint>>>>,
    /// Cached parameters, flushed the same way.
    params_cache: Arc<RwLock<HashMap<String, String>>>,
    /// Cached tags, flushed the same way.
    tags_cache: Arc<RwLock<HashMap<String, String>>>,
}

/// Configuration for MLflow integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLflowConfig {
    /// MLflow tracking server URI (default: http://localhost:5000)
    pub tracking_uri: String,
    /// Default experiment name
    pub experiment_name: String,
    /// Enable automatic metric logging
    pub auto_log: bool,
    /// Metric logging interval (steps)
    pub log_interval: usize,
    /// Maximum number of cached metrics before flush
    pub max_cache_size: usize,
    /// Enable artifact logging
    pub log_artifacts: bool,
    /// Artifact storage directory
    pub artifact_dir: PathBuf,
    /// Whether to actually contact a tracking server, or stay local-only.
    /// See [`TrackingMode`].
    #[serde(default)]
    pub mode: TrackingMode,
}

impl Default for MLflowConfig {
    fn default() -> Self {
        Self {
            tracking_uri: "http://localhost:5000".to_string(),
            experiment_name: "trustformers-debug".to_string(),
            auto_log: true,
            log_interval: 10,
            max_cache_size: 1000,
            log_artifacts: true,
            artifact_dir: PathBuf::from("./mlflow_artifacts"),
            mode: TrackingMode::Http,
        }
    }
}

/// Real MLflow REST API 2.0 transport, compiled only when
/// `http-integrations` is enabled (it pulls `reqwest` and a non-pure-Rust
/// TLS stack). See the module docs.
#[cfg(feature = "http-integrations")]
mod rest {
    use super::Result;
    use anyhow::Context;

    fn endpoint(base: &str, path: &str) -> String {
        format!("{}/api/2.0/mlflow/{path}", base.trim_end_matches('/'))
    }

    async fn parse_response(url: &str, response: reqwest::Response) -> Result<serde_json::Value> {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("MLflow request to {url} returned {status}: {text}");
        }
        serde_json::from_str(&text)
            .with_context(|| format!("MLflow response from {url} was not valid JSON: {text}"))
    }

    pub(super) async fn post(
        base: &str,
        path: &str,
        body: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let url = endpoint(base, path);
        let response = reqwest::Client::new()
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("MLflow request to {url} failed"))?;
        parse_response(&url, response).await
    }

    /// `GET`, treating a `404` response as `Ok(None)` (the resource
    /// genuinely does not exist) rather than an error -- everything else
    /// (network failure, `5xx`, malformed JSON) is still a real `Err`, so
    /// callers can't confuse "doesn't exist" with "couldn't find out".
    pub(super) async fn get_optional(
        base: &str,
        path: &str,
        query: &[(&str, &str)],
    ) -> Result<Option<serde_json::Value>> {
        let url = endpoint(base, path);
        let response = reqwest::Client::new()
            .get(&url)
            .query(query)
            .send()
            .await
            .with_context(|| format!("MLflow request to {url} failed"))?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        parse_response(&url, response).await.map(Some)
    }
}

/// Without `http-integrations`, no HTTP client exists in this build: fail
/// honestly instead of pretending to contact a tracking server.
#[cfg(not(feature = "http-integrations"))]
mod rest {
    use super::Result;

    const DISABLED_MESSAGE: &str = "MLflow HTTP tracking is not enabled: rebuild \
         trustformers-debug with `--features http-integrations`, or select \
         `TrackingMode::LocalOnly`";

    pub(super) async fn post(
        _base: &str,
        _path: &str,
        _body: serde_json::Value,
    ) -> Result<serde_json::Value> {
        anyhow::bail!(DISABLED_MESSAGE)
    }

    pub(super) async fn get_optional(
        _base: &str,
        _path: &str,
        _query: &[(&str, &str)],
    ) -> Result<Option<serde_json::Value>> {
        anyhow::bail!(DISABLED_MESSAGE)
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// A single metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    /// Metric value
    pub value: f64,
    /// Step number
    pub step: i64,
    /// Timestamp (milliseconds since epoch)
    pub timestamp: i64,
}

/// MLflow run information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunInfo {
    /// Run ID
    pub run_id: String,
    /// Experiment ID
    pub experiment_id: String,
    /// Run name
    pub run_name: String,
    /// Start time (milliseconds since epoch)
    pub start_time: i64,
    /// End time (milliseconds since epoch, None if active)
    pub end_time: Option<i64>,
    /// Run status
    pub status: RunStatus,
}

/// Status of an MLflow run
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    /// Run is active
    Running,
    /// Run completed successfully
    Finished,
    /// Run failed
    Failed,
    /// Run was killed
    Killed,
}

impl RunStatus {
    /// The exact string MLflow's REST API expects for `runs/update`'s
    /// `status` field.
    fn as_mlflow_str(self) -> &'static str {
        match self {
            RunStatus::Running => "RUNNING",
            RunStatus::Finished => "FINISHED",
            RunStatus::Failed => "FAILED",
            RunStatus::Killed => "KILLED",
        }
    }
}

/// Artifact type for logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
    /// Model weights/checkpoints
    Model,
    /// Visualization plots
    Plot,
    /// Text reports
    Report,
    /// Raw data
    Data,
    /// Configuration files
    Config,
}

impl MLflowClient {
    /// Create a new MLflow client
    ///
    /// # Arguments
    /// * `config` - MLflow configuration
    ///
    /// # Example
    /// ```rust
    /// use trustformers_debug::{MLflowClient, MLflowConfig};
    ///
    /// let config = MLflowConfig::default();
    /// let client = MLflowClient::new(config);
    /// ```
    pub fn new(config: MLflowConfig) -> Self {
        Self {
            tracking_uri: config.tracking_uri.clone(),
            experiment_id: None,
            run_id: None,
            run_name: None,
            run_start_time: None,
            config,
            metrics_cache: Arc::new(RwLock::new(HashMap::new())),
            params_cache: Arc::new(RwLock::new(HashMap::new())),
            tags_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set the tracking URI
    ///
    /// # Arguments
    /// * `uri` - MLflow tracking server URI
    pub fn set_tracking_uri(&mut self, uri: impl Into<String>) {
        self.tracking_uri = uri.into();
    }

    /// Start a new experiment.
    ///
    /// In [`TrackingMode::Http`] this looks the experiment up by name
    /// (`GET experiments/get-by-name`) and creates it
    /// (`POST experiments/create`) if it doesn't exist yet -- a real round
    /// trip to `tracking_uri`, which fails with a real transport/HTTP error
    /// if the server is unreachable or rejects the request. In
    /// [`TrackingMode::LocalOnly`] no request is made and a synthetic
    /// `local-*` id is generated instead, so callers can always tell the
    /// two apart.
    ///
    /// # Arguments
    /// * `name` - Experiment name
    ///
    /// # Returns
    /// Experiment ID
    pub async fn start_experiment(&mut self, name: impl Into<String>) -> Result<String> {
        let experiment_name = name.into();

        let experiment_id = match self.config.mode {
            TrackingMode::LocalOnly => {
                let id = format!("local-exp-{}", uuid::Uuid::new_v4());
                tracing::info!(
                    experiment_id = %id,
                    experiment_name = %experiment_name,
                    "Started MLflow experiment (LocalOnly: not sent to a tracking server)"
                );
                id
            },
            TrackingMode::Http => {
                let query = [("experiment_name", experiment_name.as_str())];
                // `get_optional` only ever returns `Ok(None)` for a genuine
                // "no such experiment" (HTTP 404); any other failure --
                // network error, auth failure, the `http-integrations`
                // feature being disabled -- is a real `Err` that must
                // propagate here rather than being silently reinterpreted
                // as "doesn't exist yet, so create a new one".
                let existing =
                    rest::get_optional(&self.tracking_uri, "experiments/get-by-name", &query)
                        .await
                        .with_context(|| {
                            format!("failed to look up MLflow experiment {experiment_name:?}")
                        })?;
                let id = match existing {
                    Some(response) => response
                        .get("experiment")
                        .and_then(|e| e.get("experiment_id"))
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .context("MLflow experiments/get-by-name response missing experiment_id")?,
                    None => {
                        let body = serde_json::json!({ "name": experiment_name });
                        let response = rest::post(&self.tracking_uri, "experiments/create", body)
                            .await
                            .with_context(|| {
                                format!("failed to create MLflow experiment {experiment_name:?}")
                            })?;
                        response
                            .get("experiment_id")
                            .and_then(|v| v.as_str())
                            .map(str::to_string)
                            .context("MLflow experiments/create response missing experiment_id")?
                    },
                };
                tracing::info!(
                    experiment_id = %id,
                    experiment_name = %experiment_name,
                    tracking_uri = %self.tracking_uri,
                    "Started MLflow experiment"
                );
                id
            },
        };

        self.experiment_id = Some(experiment_id.clone());
        Ok(experiment_id)
    }

    /// Start a new run within the current experiment.
    ///
    /// Real `POST runs/create` in [`TrackingMode::Http`]; a synthetic
    /// `local-*` id with no network access in [`TrackingMode::LocalOnly`].
    ///
    /// # Arguments
    /// * `run_name` - Optional run name
    ///
    /// # Returns
    /// Run ID
    pub async fn start_run(&mut self, run_name: Option<&str>) -> Result<String> {
        let experiment_id = self
            .experiment_id
            .clone()
            .context("No active experiment. Call start_experiment() first")?;

        let run_name = run_name.unwrap_or("debug_run").to_string();
        let start_time = now_ms();

        let run_id = match self.config.mode {
            TrackingMode::LocalOnly => {
                let id = format!("local-run-{}", uuid::Uuid::new_v4());
                tracing::info!(
                    run_id = %id,
                    run_name = %run_name,
                    experiment_id = %experiment_id,
                    "Started MLflow run (LocalOnly: not sent to a tracking server)"
                );
                id
            },
            TrackingMode::Http => {
                let body = serde_json::json!({
                    "experiment_id": experiment_id,
                    "run_name": run_name,
                    "start_time": start_time,
                    "tags": [{"key": "mlflow.runName", "value": run_name}],
                });
                let response =
                    rest::post(&self.tracking_uri, "runs/create", body).await.with_context(
                        || format!("failed to create MLflow run in experiment {experiment_id}"),
                    )?;
                let id = response
                    .get("run")
                    .and_then(|r| r.get("info"))
                    .and_then(|i| i.get("run_id"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .context("MLflow runs/create response missing run.info.run_id")?;
                tracing::info!(
                    run_id = %id,
                    run_name = %run_name,
                    experiment_id = %experiment_id,
                    "Started MLflow run"
                );
                id
            },
        };

        self.run_id = Some(run_id.clone());
        self.run_name = Some(run_name);
        self.run_start_time = Some(start_time);

        // Clear caches for new run
        self.metrics_cache.write().clear();
        self.params_cache.write().clear();
        self.tags_cache.write().clear();

        Ok(run_id)
    }

    /// End the current run: flushes any cached metrics/params/tags, then
    /// (in [`TrackingMode::Http`]) sets the run's terminal status via a real
    /// `POST runs/update` call.
    ///
    /// # Arguments
    /// * `status` - Final run status
    pub async fn end_run(&mut self, status: RunStatus) -> Result<()> {
        let run_id = self.run_id.clone().context("No active run")?;

        self.flush().await?;

        if self.config.mode == TrackingMode::Http {
            let body = serde_json::json!({
                "run_id": run_id,
                "status": status.as_mlflow_str(),
                "end_time": now_ms(),
            });
            rest::post(&self.tracking_uri, "runs/update", body).await.with_context(|| {
                format!("failed to set MLflow run {run_id} status to {status:?}")
            })?;
        }

        tracing::info!(
            run_id = %run_id,
            status = ?status,
            "Ended MLflow run"
        );

        self.run_id = None;
        self.run_name = None;
        self.run_start_time = None;

        Ok(())
    }

    /// Log a parameter. Cached locally; delivered to the tracking server by
    /// [`MLflowClient::flush`] / [`MLflowClient::end_run`] (never on every
    /// call -- a per-call network round trip for every logged value would
    /// be prohibitively slow for training loops that log every step).
    ///
    /// # Arguments
    /// * `key` - Parameter name
    /// * `value` - Parameter value
    pub fn log_param(&mut self, key: impl Into<String>, value: impl ToString) -> Result<()> {
        let key = key.into();
        let value = value.to_string();

        self.run_id.as_ref().context("No active run. Call start_run() first")?;

        self.params_cache.write().insert(key.clone(), value.clone());

        tracing::debug!(key = %key, value = %value, "Logged parameter");

        Ok(())
    }

    /// Log multiple parameters at once
    ///
    /// # Arguments
    /// * `params` - Map of parameter names to values
    pub fn log_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            self.log_param(key, value)?;
        }
        Ok(())
    }

    /// Log a tag. Cached locally, delivered the same way as parameters.
    ///
    /// # Arguments
    /// * `key` - Tag name
    /// * `value` - Tag value
    pub fn log_tag(&mut self, key: impl Into<String>, value: impl ToString) -> Result<()> {
        let key = key.into();
        let value = value.to_string();

        self.run_id.as_ref().context("No active run. Call start_run() first")?;

        self.tags_cache.write().insert(key.clone(), value.clone());

        tracing::debug!(key = %key, value = %value, "Logged tag");

        Ok(())
    }

    /// Log multiple tags at once
    ///
    /// # Arguments
    /// * `tags` - Map of tag names to values
    pub fn log_tags(&mut self, tags: HashMap<String, String>) -> Result<()> {
        for (key, value) in tags {
            self.log_tag(key, value)?;
        }
        Ok(())
    }

    /// Log a metric at a specific step. Cached locally; see
    /// [`MLflowClient::log_param`] for why this does not hit the network
    /// directly.
    ///
    /// # Arguments
    /// * `key` - Metric name
    /// * `value` - Metric value
    /// * `step` - Step number
    pub fn log_metric(&mut self, key: impl Into<String>, value: f64, step: i64) -> Result<()> {
        let key = key.into();

        self.run_id.as_ref().context("No active run. Call start_run() first")?;

        let metric = MetricPoint {
            value,
            step,
            timestamp: now_ms(),
        };

        self.metrics_cache.write().entry(key.clone()).or_default().push(metric);

        tracing::debug!(key = %key, value = %value, step = %step, "Logged metric");

        let cached_count = self.metrics_cache.read().values().map(|v| v.len()).sum::<usize>();
        if cached_count >= self.config.max_cache_size {
            // Delivery is a real, fallible network call now, so it cannot
            // happen implicitly inside a sync fn: surface the backlog
            // instead of silently dropping or blocking on a runtime here.
            tracing::warn!(
                cached_metric_count = cached_count,
                max_cache_size = self.config.max_cache_size,
                "MLflow metric cache exceeds max_cache_size; call `MLflowClient::flush().await` \
                 to deliver cached metrics to the tracking server"
            );
        }

        Ok(())
    }

    /// Log multiple metrics at once
    ///
    /// # Arguments
    /// * `metrics` - Map of metric names to values
    /// * `step` - Step number
    pub fn log_metrics(&mut self, metrics: HashMap<String, f64>, step: i64) -> Result<()> {
        for (key, value) in metrics {
            self.log_metric(key, value, step)?;
        }
        Ok(())
    }

    /// Log tensor statistics as metrics
    ///
    /// # Arguments
    /// * `prefix` - Metric name prefix
    /// * `tensor` - Tensor to analyze
    /// * `step` - Step number
    pub fn log_tensor_stats(&mut self, prefix: &str, tensor: &Tensor, step: i64) -> Result<()> {
        // Log tensor element count and shape info
        self.log_metric(
            format!("{}/element_count", prefix),
            tensor.len() as f64,
            step,
        )?;
        self.log_metric(
            format!("{}/memory_bytes", prefix),
            tensor.memory_usage() as f64,
            step,
        )?;

        let shape = tensor.shape();
        self.log_metric(format!("{}/ndim", prefix), shape.len() as f64, step)?;

        Ok(())
    }

    /// Log array statistics as metrics
    ///
    /// # Arguments
    /// * `prefix` - Metric name prefix
    /// * `array` - Array to analyze
    /// * `step` - Step number
    pub fn log_array_stats(&mut self, prefix: &str, array: &Array1<f64>, step: i64) -> Result<()> {
        let mean = array.mean().unwrap_or(0.0);
        let std = array.std(0.0);
        let min = array.iter().copied().fold(f64::INFINITY, f64::min);
        let max = array.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        self.log_metric(format!("{}/mean", prefix), mean, step)?;
        self.log_metric(format!("{}/std", prefix), std, step)?;
        self.log_metric(format!("{}/min", prefix), min, step)?;
        self.log_metric(format!("{}/max", prefix), max, step)?;

        Ok(())
    }

    /// Flush cached metrics, parameters and tags to the MLflow tracking
    /// server via a real `POST runs/log-batch` call (chunked to respect
    /// MLflow's per-request limits of 1000 metrics / 100 params / 100 tags).
    /// A no-op in [`TrackingMode::LocalOnly`] -- the caches themselves are
    /// the record in that mode -- and a no-op when there is nothing cached
    /// or no active run.
    pub async fn flush(&self) -> Result<()> {
        if self.config.mode == TrackingMode::LocalOnly {
            return Ok(());
        }

        let Some(run_id) = self.run_id.clone() else {
            return Ok(());
        };

        let metrics: Vec<serde_json::Value> = self
            .metrics_cache
            .read()
            .iter()
            .flat_map(|(key, points)| {
                points.iter().map(move |p| {
                    serde_json::json!({
                        "key": key,
                        "value": p.value,
                        "timestamp": p.timestamp,
                        "step": p.step,
                    })
                })
            })
            .collect();
        let params: Vec<serde_json::Value> = self
            .params_cache
            .read()
            .iter()
            .map(|(k, v)| serde_json::json!({"key": k, "value": v}))
            .collect();
        let tags: Vec<serde_json::Value> = self
            .tags_cache
            .read()
            .iter()
            .map(|(k, v)| serde_json::json!({"key": k, "value": v}))
            .collect();

        if metrics.is_empty() && params.is_empty() && tags.is_empty() {
            return Ok(());
        }

        const METRIC_CHUNK: usize = 1000;
        const PARAM_CHUNK: usize = 100;
        const TAG_CHUNK: usize = 100;

        let metric_chunks: Vec<Vec<serde_json::Value>> =
            metrics.chunks(METRIC_CHUNK).map(|c| c.to_vec()).collect();
        let param_chunks: Vec<Vec<serde_json::Value>> =
            params.chunks(PARAM_CHUNK).map(|c| c.to_vec()).collect();
        let tag_chunks: Vec<Vec<serde_json::Value>> =
            tags.chunks(TAG_CHUNK).map(|c| c.to_vec()).collect();
        let rounds = metric_chunks.len().max(param_chunks.len()).max(tag_chunks.len()).max(1);

        for i in 0..rounds {
            let body = serde_json::json!({
                "run_id": run_id,
                "metrics": metric_chunks.get(i).cloned().unwrap_or_default(),
                "params": param_chunks.get(i).cloned().unwrap_or_default(),
                "tags": tag_chunks.get(i).cloned().unwrap_or_default(),
            });
            rest::post(&self.tracking_uri, "runs/log-batch", body).await.with_context(|| {
                format!(
                    "failed to flush batch {}/{rounds} to MLflow run {run_id}",
                    i + 1
                )
            })?;
        }

        tracing::debug!(
            run_id = %run_id,
            metric_count = metrics.len(),
            param_count = params.len(),
            tag_count = tags.len(),
            "Flushed metrics/params/tags to MLflow"
        );

        Ok(())
    }

    /// Log an artifact (file) to the *local* `artifact_dir` cache.
    ///
    /// Unlike [`Self::log_metric`]/[`Self::log_param`]/[`Self::log_tag`],
    /// this method does **not** upload to the remote MLflow tracking
    /// server's artifact store, even in [`TrackingMode::Http`] -- MLflow's
    /// artifact storage is backend-dependent (local disk, S3, GCS, Azure
    /// Blob, DBFS, ...) and there is no single REST endpoint this client
    /// can target without knowing which backend the server is configured
    /// with. Uploading real artifact bytes to a remote MLflow server is
    /// intentionally out of scope for this client; only the experiment/run
    /// lifecycle and metrics/params/tags are delivered over HTTP. Callers
    /// that need artifacts on the actual tracking server must copy from
    /// `artifact_dir` themselves via whatever channel the server's artifact
    /// backend expects.
    ///
    /// # Arguments
    /// * `local_path` - Path to local file
    /// * `artifact_path` - Optional path within the local artifact cache
    /// * `artifact_type` - Type of artifact
    pub fn log_artifact(
        &self,
        local_path: impl AsRef<Path>,
        artifact_path: Option<&str>,
        artifact_type: ArtifactType,
    ) -> Result<()> {
        let _run_id = self.run_id.as_ref().context("No active run")?;

        let local_path = local_path.as_ref();

        if !self.config.log_artifacts {
            tracing::debug!("Artifact logging disabled");
            return Ok(());
        }

        // Copy to artifact directory
        let artifact_dir = &self.config.artifact_dir;
        std::fs::create_dir_all(artifact_dir)?;

        let dest_path = if let Some(rel_path) = artifact_path {
            artifact_dir.join(rel_path)
        } else {
            artifact_dir.join(local_path.file_name().context("local_path must have a filename")?)
        };

        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::copy(local_path, &dest_path).context("Failed to copy artifact")?;

        if self.config.mode == TrackingMode::Http {
            tracing::warn!(
                local_path = ?local_path,
                artifact_path = ?dest_path,
                artifact_type = ?artifact_type,
                "Artifact cached locally only -- NOT uploaded to the remote MLflow tracking \
                 server at {}; MLflow artifact storage backends vary and are not implemented \
                 by this client",
                self.tracking_uri
            );
        } else {
            tracing::info!(
                local_path = ?local_path,
                artifact_path = ?dest_path,
                artifact_type = ?artifact_type,
                "Logged artifact to local cache"
            );
        }

        Ok(())
    }

    /// Log a model artifact
    ///
    /// # Arguments
    /// * `model_path` - Path to model file
    /// * `model_name` - Optional model name
    pub fn log_model(&self, model_path: impl AsRef<Path>, model_name: Option<&str>) -> Result<()> {
        let artifact_path = if let Some(name) = model_name {
            format!("models/{}", name)
        } else {
            "models/model".to_string()
        };

        self.log_artifact(model_path, Some(&artifact_path), ArtifactType::Model)
    }

    /// Log a plot/visualization
    ///
    /// # Arguments
    /// * `plot_path` - Path to plot file
    /// * `plot_name` - Optional plot name
    pub fn log_plot(&self, plot_path: impl AsRef<Path>, plot_name: Option<&str>) -> Result<()> {
        let artifact_path = if let Some(name) = plot_name {
            format!("plots/{}", name)
        } else {
            "plots/plot".to_string()
        };

        self.log_artifact(plot_path, Some(&artifact_path), ArtifactType::Plot)
    }

    /// Log a text report
    ///
    /// # Arguments
    /// * `content` - Report content
    /// * `filename` - Report filename
    pub fn log_report(&self, content: &str, filename: &str) -> Result<()> {
        let temp_path = std::env::temp_dir().join(filename);
        std::fs::write(&temp_path, content)?;

        self.log_artifact(
            &temp_path,
            Some(&format!("reports/{}", filename)),
            ArtifactType::Report,
        )?;

        std::fs::remove_file(&temp_path)?;

        Ok(())
    }

    /// Get current run information. The `start_time` reflects when
    /// [`MLflowClient::start_run`] was actually called (real wall-clock ms
    /// since epoch), not a placeholder.
    pub fn get_run_info(&self) -> Option<RunInfo> {
        let run_id = self.run_id.as_ref()?;
        let experiment_id = self.experiment_id.as_ref()?;

        Some(RunInfo {
            run_id: run_id.clone(),
            experiment_id: experiment_id.clone(),
            run_name: self.run_name.clone().unwrap_or_else(|| "debug_run".to_string()),
            start_time: self.run_start_time.unwrap_or(0),
            end_time: None,
            status: RunStatus::Running,
        })
    }

    /// Get all logged parameters
    pub fn get_params(&self) -> HashMap<String, String> {
        self.params_cache.read().clone()
    }

    /// Get all logged metrics
    pub fn get_metrics(&self) -> HashMap<String, Vec<MetricPoint>> {
        self.metrics_cache.read().clone()
    }

    /// Get all logged tags
    pub fn get_tags(&self) -> HashMap<String, String> {
        self.tags_cache.read().clone()
    }
}

/// Integration with TrustformeRS debug session
pub struct MLflowDebugSession {
    /// MLflow client
    pub client: MLflowClient,
    /// Current step
    step: i64,
}

impl MLflowDebugSession {
    /// Create a new MLflow debug session
    pub fn new(config: MLflowConfig) -> Self {
        Self {
            client: MLflowClient::new(config),
            step: 0,
        }
    }

    /// Start debugging with MLflow tracking
    pub async fn start(&mut self, experiment_name: &str, run_name: Option<&str>) -> Result<()> {
        self.client.start_experiment(experiment_name).await?;
        self.client.start_run(run_name).await?;
        self.step = 0;
        Ok(())
    }

    /// Log debugging metrics for current step
    pub fn log_debug_metrics(&mut self, metrics: HashMap<String, f64>) -> Result<()> {
        self.client.log_metrics(metrics, self.step)?;
        self.step += 1;
        Ok(())
    }

    /// End debugging session
    pub async fn end(&mut self, status: RunStatus) -> Result<()> {
        self.client.end_run(status).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    /// A config that never touches the network -- used by tests that only
    /// exercise local caching behavior, not transport.
    fn local_only_config() -> MLflowConfig {
        let mut config = MLflowConfig::default();
        config.mode = TrackingMode::LocalOnly;
        config
    }

    #[test]
    fn test_mlflow_client_creation() {
        let config = MLflowConfig::default();
        let _client = MLflowClient::new(config);
    }

    #[test]
    fn test_default_mode_is_http_not_silent_local_only() {
        // The mission requires LocalOnly to be an explicit opt-in, never a
        // silent default.
        assert_eq!(MLflowConfig::default().mode, TrackingMode::Http);
    }

    #[tokio::test]
    async fn test_local_only_start_experiment_and_run_never_touch_network() {
        let mut client = MLflowClient::new(local_only_config());

        let exp_id = client
            .start_experiment("test_experiment")
            .await
            .expect("LocalOnly must not require a server");
        assert!(exp_id.starts_with("local-exp-"), "got {exp_id:?}");

        let run_id = client
            .start_run(Some("test_run"))
            .await
            .expect("LocalOnly must not require a server");
        assert!(run_id.starts_with("local-run-"), "got {run_id:?}");
    }

    #[tokio::test]
    async fn test_log_params() -> Result<()> {
        let mut client = MLflowClient::new(local_only_config());

        client.start_experiment("test").await?;
        client.start_run(None).await?;

        client.log_param("learning_rate", "0.001")?;
        client.log_param("batch_size", "32")?;

        let params = client.get_params();
        assert_eq!(params.get("learning_rate"), Some(&"0.001".to_string()));
        assert_eq!(params.get("batch_size"), Some(&"32".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_log_tags() -> Result<()> {
        let mut client = MLflowClient::new(local_only_config());

        client.start_experiment("test").await?;
        client.start_run(None).await?;

        client.log_tag("owner", "trustformers-debug")?;
        let tags = client.get_tags();
        assert_eq!(tags.get("owner"), Some(&"trustformers-debug".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_log_metrics() -> Result<()> {
        let mut client = MLflowClient::new(local_only_config());

        client.start_experiment("test").await?;
        client.start_run(None).await?;

        client.log_metric("loss", 0.5, 0)?;
        client.log_metric("loss", 0.4, 1)?;
        client.log_metric("accuracy", 0.8, 0)?;

        let metrics = client.get_metrics();
        assert_eq!(
            metrics.get("loss").expect("expected value not found").len(),
            2
        );
        assert_eq!(
            metrics.get("accuracy").expect("expected value not found").len(),
            1
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_log_array_stats() -> Result<()> {
        let mut client = MLflowClient::new(local_only_config());

        client.start_experiment("test").await?;
        client.start_run(None).await?;

        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        client.log_array_stats("weights", &array, 0)?;

        let metrics = client.get_metrics();
        assert!(metrics.contains_key("weights/mean"));
        assert!(metrics.contains_key("weights/std"));
        assert!(metrics.contains_key("weights/min"));
        assert!(metrics.contains_key("weights/max"));

        Ok(())
    }

    #[tokio::test]
    async fn test_log_artifact_really_copies_the_file_to_local_cache() -> Result<()> {
        let artifact_dir =
            std::env::temp_dir().join(format!("mlflow_test_artifacts_{}", uuid::Uuid::new_v4()));
        let mut config = local_only_config();
        config.artifact_dir = artifact_dir.clone();
        let mut client = MLflowClient::new(config);

        client.start_experiment("test").await?;
        client.start_run(None).await?;

        let source_path =
            std::env::temp_dir().join(format!("mlflow_test_source_{}.txt", uuid::Uuid::new_v4()));
        std::fs::write(&source_path, b"real artifact bytes")?;

        client.log_artifact(
            &source_path,
            Some("checkpoints/model.txt"),
            ArtifactType::Model,
        )?;

        let dest_path = artifact_dir.join("checkpoints/model.txt");
        assert!(
            dest_path.exists(),
            "artifact must really be copied, not just logged"
        );
        assert_eq!(std::fs::read(&dest_path)?, b"real artifact bytes");

        std::fs::remove_file(&source_path).ok();
        std::fs::remove_dir_all(&artifact_dir).ok();

        Ok(())
    }

    #[tokio::test]
    async fn test_end_run() -> Result<()> {
        let mut client = MLflowClient::new(local_only_config());

        client.start_experiment("test").await?;
        client.start_run(None).await?;
        client.log_metric("loss", 0.5, 0)?;
        client.end_run(RunStatus::Finished).await?;

        assert!(client.run_id.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_get_run_info_reports_real_start_time_not_zero_placeholder() -> Result<()> {
        let mut client = MLflowClient::new(local_only_config());
        client.start_experiment("test").await?;

        let before = now_ms();
        client.start_run(Some("named-run")).await?;
        let after = now_ms();

        let info = client.get_run_info().expect("run should be active");
        assert_eq!(info.run_name, "named-run");
        // The old implementation hardcoded `start_time: 0`.
        assert!(
            info.start_time >= before && info.start_time <= after,
            "start_time {} should fall within [{before}, {after}]",
            info.start_time
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_mlflow_debug_session() -> Result<()> {
        let mut session = MLflowDebugSession::new(local_only_config());

        session.start("test_debug", Some("debug_run_1")).await?;

        let mut metrics = HashMap::new();
        metrics.insert("gradient_norm".to_string(), 0.1);
        metrics.insert("activation_mean".to_string(), 0.5);

        session.log_debug_metrics(metrics)?;

        session.end(RunStatus::Finished).await?;

        Ok(())
    }

    // ------------------------------------------------------------------
    // Real HTTP delivery against a local mock MLflow server, and an
    // honest, structured failure when the feature is disabled.
    // ------------------------------------------------------------------

    #[cfg(feature = "http-integrations")]
    mod http_delivery {
        use super::*;
        use axum::extract::{Query, State};
        use axum::http::StatusCode;
        use axum::routing::{get, post};
        use axum::{Json, Router};
        use std::sync::Arc;
        use tokio::net::TcpListener;
        use tokio::sync::Mutex as AsyncMutex;

        #[derive(Default, Clone)]
        struct MockState {
            /// (path, body-or-query) for every request the mock server saw,
            /// in arrival order.
            requests: Vec<(&'static str, serde_json::Value)>,
        }

        async fn get_by_name(
            State(state): State<Arc<AsyncMutex<MockState>>>,
            Query(params): Query<HashMap<String, String>>,
        ) -> (StatusCode, Json<serde_json::Value>) {
            state.lock().await.requests.push((
                "experiments/get-by-name",
                serde_json::to_value(&params).unwrap_or_default(),
            ));
            // Simulate "experiment does not exist yet" so the client
            // exercises the real create fallback too.
            (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error_code": "RESOURCE_DOES_NOT_EXIST"})),
            )
        }

        async fn create_experiment(
            State(state): State<Arc<AsyncMutex<MockState>>>,
            Json(body): Json<serde_json::Value>,
        ) -> Json<serde_json::Value> {
            state.lock().await.requests.push(("experiments/create", body));
            Json(serde_json::json!({"experiment_id": "1"}))
        }

        async fn create_run(
            State(state): State<Arc<AsyncMutex<MockState>>>,
            Json(body): Json<serde_json::Value>,
        ) -> Json<serde_json::Value> {
            let experiment_id = body["experiment_id"].clone();
            state.lock().await.requests.push(("runs/create", body));
            Json(serde_json::json!({
                "run": {"info": {"run_id": "run-1", "experiment_id": experiment_id, "status": "RUNNING"}}
            }))
        }

        async fn log_batch(
            State(state): State<Arc<AsyncMutex<MockState>>>,
            Json(body): Json<serde_json::Value>,
        ) -> Json<serde_json::Value> {
            state.lock().await.requests.push(("runs/log-batch", body));
            Json(serde_json::json!({}))
        }

        async fn update_run(
            State(state): State<Arc<AsyncMutex<MockState>>>,
            Json(body): Json<serde_json::Value>,
        ) -> Json<serde_json::Value> {
            state.lock().await.requests.push(("runs/update", body));
            Json(serde_json::json!({}))
        }

        /// A real local HTTP server (127.0.0.1, OS-assigned port) speaking
        /// just enough of the MLflow REST API 2.0 shape to exercise the
        /// client end to end.
        async fn start_mock_server() -> (String, Arc<AsyncMutex<MockState>>) {
            let state = Arc::new(AsyncMutex::new(MockState::default()));
            let app = Router::new()
                .route("/api/2.0/mlflow/experiments/get-by-name", get(get_by_name))
                .route(
                    "/api/2.0/mlflow/experiments/create",
                    post(create_experiment),
                )
                .route("/api/2.0/mlflow/runs/create", post(create_run))
                .route("/api/2.0/mlflow/runs/log-batch", post(log_batch))
                .route("/api/2.0/mlflow/runs/update", post(update_run))
                .with_state(state.clone());
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock MLflow server");
            let addr = listener.local_addr().expect("mock server local addr");
            tokio::spawn(async move {
                let _ = axum::serve(listener, app).await;
            });
            (format!("http://{addr}"), state)
        }

        #[tokio::test]
        async fn test_full_http_run_lifecycle_hits_real_mock_server() {
            let (base_url, state) = start_mock_server().await;
            let mut config = MLflowConfig::default();
            config.tracking_uri = base_url;
            let mut client = MLflowClient::new(config);

            let exp_id =
                client.start_experiment("exp").await.expect("mock server should accept create");
            assert_eq!(exp_id, "1");

            let run_id = client
                .start_run(Some("my-run"))
                .await
                .expect("mock server should accept create");
            assert_eq!(run_id, "run-1");

            client.log_metric("loss", 0.5, 0).expect("cache metric");
            client.log_param("lr", "0.01").expect("cache param");
            client.log_tag("owner", "ci").expect("cache tag");

            client
                .end_run(RunStatus::Finished)
                .await
                .expect("end_run should flush and update status via the mock server");

            let requests = state.lock().await.requests.clone();
            let paths: Vec<&str> = requests.iter().map(|(p, _)| *p).collect();
            assert!(paths.contains(&"experiments/get-by-name"));
            assert!(paths.contains(&"experiments/create"));
            assert!(paths.contains(&"runs/create"));
            assert!(paths.contains(&"runs/log-batch"));
            assert!(paths.contains(&"runs/update"));

            let log_batch_body = requests
                .iter()
                .find(|(p, _)| *p == "runs/log-batch")
                .map(|(_, b)| b.clone())
                .expect("log-batch request captured");
            assert_eq!(log_batch_body["run_id"], "run-1");
            assert_eq!(log_batch_body["metrics"][0]["key"], "loss");
            assert_eq!(log_batch_body["metrics"][0]["value"], 0.5);
            assert_eq!(log_batch_body["params"][0]["key"], "lr");
            assert_eq!(log_batch_body["tags"][0]["key"], "owner");

            let update_body = requests
                .iter()
                .find(|(p, _)| *p == "runs/update")
                .map(|(_, b)| b.clone())
                .expect("update request captured");
            assert_eq!(update_body["status"], "FINISHED");
        }

        #[tokio::test]
        async fn test_http_mode_returns_real_transport_error_when_server_unreachable() {
            // Port 1 refuses connections. The old implementation fabricated
            // `exp_<uuid>` regardless of whether any server existed; the
            // fixed client must surface a real error instead.
            let mut config = MLflowConfig::default();
            config.tracking_uri = "http://127.0.0.1:1".to_string();
            let mut client = MLflowClient::new(config);

            let err = client
                .start_experiment("exp")
                .await
                .expect_err("unreachable server must not fabricate an experiment id");
            assert!(!err.to_string().is_empty());
        }
    }

    #[cfg(not(feature = "http-integrations"))]
    mod http_disabled {
        use super::*;

        #[tokio::test]
        async fn test_http_mode_fails_honestly_without_feature() {
            // Default mode (Http) with the feature compiled out must not
            // silently pretend to have created anything.
            let mut client = MLflowClient::new(MLflowConfig::default());

            let err = client
                .start_experiment("exp")
                .await
                .expect_err("must not silently pretend to have created an experiment");
            // `start_experiment` wraps the transport error in an
            // operation-level `with_context` (e.g. "failed to look up
            // MLflow experiment ..."), so check the full `anyhow` cause
            // chain -- not just the top `Display` frame -- for the actual
            // "rebuild with http-integrations" message.
            let full_chain = err.chain().map(|c| c.to_string()).collect::<Vec<_>>().join(" | ");
            assert!(
                full_chain.contains("http-integrations"),
                "expected the disabled-feature message somewhere in the error chain, got: {full_chain}"
            );
        }
    }
}
