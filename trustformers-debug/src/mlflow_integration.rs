//! MLflow Integration for Experiment Tracking
//!
//! This module provides integration with MLflow for tracking experiments, logging metrics,
//! parameters, and artifacts during model training and debugging.

use anyhow::{Context, Result};
use parking_lot::RwLock;
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use trustformers_core::tensor::Tensor;

/// MLflow client for experiment tracking
#[derive(Debug)]
pub struct MLflowClient {
    /// MLflow tracking URI
    tracking_uri: String,
    /// Current experiment ID
    experiment_id: Option<String>,
    /// Current run ID
    run_id: Option<String>,
    /// Configuration
    config: MLflowConfig,
    /// Cached metrics
    metrics_cache: Arc<RwLock<HashMap<String, Vec<MetricPoint>>>>,
    /// Cached parameters
    params_cache: Arc<RwLock<HashMap<String, String>>>,
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
        }
    }
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
            config,
            metrics_cache: Arc::new(RwLock::new(HashMap::new())),
            params_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set the tracking URI
    ///
    /// # Arguments
    /// * `uri` - MLflow tracking server URI
    pub fn set_tracking_uri(&mut self, uri: impl Into<String>) {
        self.tracking_uri = uri.into();
    }

    /// Start a new experiment
    ///
    /// # Arguments
    /// * `name` - Experiment name
    ///
    /// # Returns
    /// Experiment ID
    pub fn start_experiment(&mut self, name: impl Into<String>) -> Result<String> {
        let experiment_name = name.into();

        // In a real implementation, this would make an HTTP request to MLflow
        // For now, we'll simulate it
        let experiment_id = format!("exp_{}", uuid::Uuid::new_v4());

        self.experiment_id = Some(experiment_id.clone());

        tracing::info!(
            experiment_id = %experiment_id,
            experiment_name = %experiment_name,
            "Started MLflow experiment"
        );

        Ok(experiment_id)
    }

    /// Start a new run within the current experiment
    ///
    /// # Arguments
    /// * `run_name` - Optional run name
    ///
    /// # Returns
    /// Run ID
    pub fn start_run(&mut self, run_name: Option<&str>) -> Result<String> {
        let experiment_id = self
            .experiment_id
            .as_ref()
            .context("No active experiment. Call start_experiment() first")?;

        let run_id = format!("run_{}", uuid::Uuid::new_v4());
        let run_name = run_name.unwrap_or("debug_run").to_string();

        self.run_id = Some(run_id.clone());

        // Clear caches for new run
        self.metrics_cache.write().clear();
        self.params_cache.write().clear();

        tracing::info!(
            run_id = %run_id,
            run_name = %run_name,
            experiment_id = %experiment_id,
            "Started MLflow run"
        );

        Ok(run_id)
    }

    /// End the current run
    ///
    /// # Arguments
    /// * `status` - Final run status
    pub fn end_run(&mut self, status: RunStatus) -> Result<()> {
        let run_id = self.run_id.as_ref().context("No active run")?;

        // Flush any cached metrics
        self.flush_metrics()?;

        tracing::info!(
            run_id = %run_id,
            status = ?status,
            "Ended MLflow run"
        );

        self.run_id = None;

        Ok(())
    }

    /// Log a parameter
    ///
    /// # Arguments
    /// * `key` - Parameter name
    /// * `value` - Parameter value
    pub fn log_param(&mut self, key: impl Into<String>, value: impl ToString) -> Result<()> {
        let key = key.into();
        let value = value.to_string();

        let _run_id = self.run_id.as_ref().context("No active run. Call start_run() first")?;

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

    /// Log a metric at a specific step
    ///
    /// # Arguments
    /// * `key` - Metric name
    /// * `value` - Metric value
    /// * `step` - Step number
    pub fn log_metric(&mut self, key: impl Into<String>, value: f64, step: i64) -> Result<()> {
        let key = key.into();

        let _run_id = self.run_id.as_ref().context("No active run. Call start_run() first")?;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        let metric = MetricPoint {
            value,
            step,
            timestamp,
        };

        self.metrics_cache.write().entry(key.clone()).or_default().push(metric);

        tracing::debug!(key = %key, value = %value, step = %step, "Logged metric");

        // Auto-flush if cache is too large
        if self.metrics_cache.read().values().map(|v| v.len()).sum::<usize>()
            >= self.config.max_cache_size
        {
            self.flush_metrics()?;
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

    /// Flush cached metrics to MLflow server
    fn flush_metrics(&self) -> Result<()> {
        let metrics = self.metrics_cache.read();

        if metrics.is_empty() {
            return Ok(());
        }

        // In a real implementation, this would make HTTP requests to MLflow
        tracing::debug!(metric_count = metrics.len(), "Flushed metrics to MLflow");

        Ok(())
    }

    /// Log an artifact (file)
    ///
    /// # Arguments
    /// * `local_path` - Path to local file
    /// * `artifact_path` - Optional path within artifact storage
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
            artifact_dir.join(local_path.file_name().unwrap())
        };

        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::copy(local_path, &dest_path).context("Failed to copy artifact")?;

        tracing::info!(
            local_path = ?local_path,
            artifact_path = ?dest_path,
            artifact_type = ?artifact_type,
            "Logged artifact"
        );

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

    /// Get current run information
    pub fn get_run_info(&self) -> Option<RunInfo> {
        let run_id = self.run_id.as_ref()?;
        let experiment_id = self.experiment_id.as_ref()?;

        Some(RunInfo {
            run_id: run_id.clone(),
            experiment_id: experiment_id.clone(),
            run_name: "debug_run".to_string(),
            start_time: 0, // Would be tracked in real implementation
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
    pub fn start(&mut self, experiment_name: &str, run_name: Option<&str>) -> Result<()> {
        self.client.start_experiment(experiment_name)?;
        self.client.start_run(run_name)?;
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
    pub fn end(&mut self, status: RunStatus) -> Result<()> {
        self.client.end_run(status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_mlflow_client_creation() {
        let config = MLflowConfig::default();
        let _client = MLflowClient::new(config);
    }

    #[test]
    fn test_start_experiment_and_run() -> Result<()> {
        let config = MLflowConfig::default();
        let mut client = MLflowClient::new(config);

        let _exp_id = client.start_experiment("test_experiment")?;
        let _run_id = client.start_run(Some("test_run"))?;

        Ok(())
    }

    #[test]
    fn test_log_params() -> Result<()> {
        let config = MLflowConfig::default();
        let mut client = MLflowClient::new(config);

        client.start_experiment("test")?;
        client.start_run(None)?;

        client.log_param("learning_rate", "0.001")?;
        client.log_param("batch_size", "32")?;

        let params = client.get_params();
        assert_eq!(params.get("learning_rate"), Some(&"0.001".to_string()));
        assert_eq!(params.get("batch_size"), Some(&"32".to_string()));

        Ok(())
    }

    #[test]
    fn test_log_metrics() -> Result<()> {
        let config = MLflowConfig::default();
        let mut client = MLflowClient::new(config);

        client.start_experiment("test")?;
        client.start_run(None)?;

        client.log_metric("loss", 0.5, 0)?;
        client.log_metric("loss", 0.4, 1)?;
        client.log_metric("accuracy", 0.8, 0)?;

        let metrics = client.get_metrics();
        assert_eq!(metrics.get("loss").unwrap().len(), 2);
        assert_eq!(metrics.get("accuracy").unwrap().len(), 1);

        Ok(())
    }

    #[test]
    fn test_log_array_stats() -> Result<()> {
        let config = MLflowConfig::default();
        let mut client = MLflowClient::new(config);

        client.start_experiment("test")?;
        client.start_run(None)?;

        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        client.log_array_stats("weights", &array, 0)?;

        let metrics = client.get_metrics();
        assert!(metrics.contains_key("weights/mean"));
        assert!(metrics.contains_key("weights/std"));
        assert!(metrics.contains_key("weights/min"));
        assert!(metrics.contains_key("weights/max"));

        Ok(())
    }

    #[test]
    fn test_end_run() -> Result<()> {
        let config = MLflowConfig::default();
        let mut client = MLflowClient::new(config);

        client.start_experiment("test")?;
        client.start_run(None)?;
        client.log_metric("loss", 0.5, 0)?;
        client.end_run(RunStatus::Finished)?;

        assert!(client.run_id.is_none());

        Ok(())
    }

    #[test]
    fn test_mlflow_debug_session() -> Result<()> {
        let config = MLflowConfig::default();
        let mut session = MLflowDebugSession::new(config);

        session.start("test_debug", Some("debug_run_1"))?;

        let mut metrics = HashMap::new();
        metrics.insert("gradient_norm".to_string(), 0.1);
        metrics.insert("activation_mean".to_string(), 0.5);

        session.log_debug_metrics(metrics)?;

        session.end(RunStatus::Finished)?;

        Ok(())
    }
}
