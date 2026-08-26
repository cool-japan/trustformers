//! CI/CD Integration tools for TrustformeRS debugging
//!
//! Provides interfaces for continuous integration and deployment workflows

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

use crate::{DebugConfig, DebugReport, DebugSession};

/// Errors from attempting to deliver a CI/CD notification.
///
/// Distinguishes *why* nothing was sent (never silently -- see
/// `CICDIntegration::send_notifications`) from a genuine transport
/// failure once delivery was actually attempted.
#[derive(Debug, thiserror::Error)]
pub enum NotificationError {
    /// This crate was built without the `http-integrations` feature, so no
    /// HTTP client is available to actually deliver the notification.
    #[error(
        "HTTP notification delivery is not enabled: rebuild trustformers-debug with \
         `--features http-integrations`"
    )]
    HttpFeatureDisabled,
    /// An `Email` channel has no HTTP email API endpoint configured (SMTP is
    /// out of scope for this crate).
    #[error("no HTTP email API endpoint configured for this Email notification channel")]
    EmailNotConfigured,
    /// This channel kind has no delivery implementation at all.
    #[error("{0} notification channel has no delivery implementation")]
    NotImplemented(&'static str),
    /// The HTTP request was sent but failed (network error or non-success
    /// status).
    #[error("notification delivery failed: {0}")]
    Transport(String),
}

/// POST a JSON payload to `url` with optional extra headers, real HTTP
/// delivery. Only compiled when `http-integrations` is enabled (see
/// `Cargo.toml`) -- this pulls in `reqwest` (and, transitively, a TLS stack
/// that is not pure Rust), so it is intentionally kept out of the default
/// build.
#[cfg(feature = "http-integrations")]
async fn post_json(
    url: &str,
    payload: &serde_json::Value,
    extra_headers: &HashMap<String, String>,
) -> Result<()> {
    let client = reqwest::Client::new();
    let mut request = client.post(url).json(payload);
    for (key, value) in extra_headers {
        request = request.header(key, value);
    }

    let response = request.send().await.map_err(|e| NotificationError::Transport(e.to_string()))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(NotificationError::Transport(format!("HTTP {status}: {body}")).into());
    }
    Ok(())
}

/// Without `http-integrations`, no HTTP client exists in this build: fail
/// honestly instead of pretending to deliver.
#[cfg(not(feature = "http-integrations"))]
async fn post_json(
    _url: &str,
    _payload: &serde_json::Value,
    _extra_headers: &HashMap<String, String>,
) -> Result<()> {
    Err(NotificationError::HttpFeatureDisabled.into())
}

/// CI/CD platform types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CICDPlatform {
    GitHub,
    GitLab,
    Jenkins,
    CircleCI,
    AzureDevOps,
    BitbucketPipelines,
    TeamCity,
    Travis,
    Custom(String),
}

/// CI/CD integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CICDConfig {
    pub platform: CICDPlatform,
    pub project_id: String,
    pub api_token: Option<String>,
    pub base_url: Option<String>,
    pub branch_filters: Vec<String>,
    pub enable_regression_detection: bool,
    pub enable_performance_tracking: bool,
    pub enable_quality_gates: bool,
    pub enable_automated_reports: bool,
    pub enable_alert_systems: bool,
    pub report_formats: Vec<ReportFormat>,
    pub notification_channels: Vec<NotificationChannel>,
}

/// Report formats for CI/CD
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    JSON,
    XML,
    HTML,
    Markdown,
    JUnit,
    SonarQube,
    Custom(String),
}

/// Notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    Email {
        recipients: Vec<String>,
        /// HTTP email API to POST to (e.g. a transactional-email provider's
        /// REST endpoint). SMTP is explicitly out of scope. `None` means
        /// "not configured": `send_email_notification` returns a
        /// structured [`NotificationError::EmailNotConfigured`] rather than
        /// silently doing nothing.
        #[serde(default)]
        api: Option<EmailApiConfig>,
    },
    Slack {
        webhook_url: String,
        channel: String,
    },
    Teams {
        webhook_url: String,
    },
    Discord {
        webhook_url: String,
    },
    Webhook {
        url: String,
        headers: HashMap<String, String>,
    },
    Custom(String),
}

/// HTTP email API configuration (transactional-email provider REST
/// endpoint). See [`NotificationChannel::Email`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailApiConfig {
    /// Full URL of the provider's "send email" REST endpoint.
    pub endpoint: String,
    /// Extra headers to send with the request (e.g. `Authorization`).
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

/// CI/CD pipeline stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineStage {
    Build,
    Test,
    Debug,
    Analysis,
    Deploy,
    Custom(String),
}

/// Quality gate status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityGateStatus {
    Passed,
    Failed,
    Warning,
    Skipped,
}

/// Quality gate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGate {
    pub name: String,
    pub description: String,
    pub metric: QualityMetric,
    pub threshold: f64,
    pub operator: ComparisonOperator,
    pub blocking: bool,
}

/// Quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityMetric {
    TestCoverage,
    ModelAccuracy,
    TrainingLoss,
    GradientNorm,
    MemoryUsage,
    TrainingTime,
    ModelSize,
    InferenceLatency,
    Custom(String),
}

/// Comparison operators for quality gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

/// Regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    pub detected: bool,
    pub severity: RegressionSeverity,
    pub metric: String,
    pub baseline_value: f64,
    pub current_value: f64,
    pub change_percent: f64,
    pub description: String,
}

/// Regression severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Critical,
    Major,
    Minor,
    Info,
}

/// Performance tracking data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceData {
    pub timestamp: DateTime<Utc>,
    pub commit_hash: String,
    pub branch: String,
    pub metrics: HashMap<String, f64>,
    pub benchmark_results: Vec<BenchmarkResult>,
}

/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub baseline: Option<f64>,
    pub improvement_percent: Option<f64>,
}

/// CI/CD pipeline run result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    pub run_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub commit_hash: String,
    pub branch: String,
    pub stage: PipelineStage,
    pub status: PipelineStatus,
    pub debug_report: Option<DebugReport>,
    pub quality_gate_results: Vec<QualityGateResult>,
    pub regression_results: Vec<RegressionResult>,
    pub performance_data: Option<PerformanceData>,
    pub artifacts: Vec<Artifact>,
    pub duration_ms: u64,
}

/// Pipeline execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineStatus {
    Success,
    Failed,
    Warning,
    Cancelled,
    Timeout,
}

impl std::fmt::Display for PipelineStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineStatus::Success => write!(f, "Success"),
            PipelineStatus::Failed => write!(f, "Failed"),
            PipelineStatus::Warning => write!(f, "Warning"),
            PipelineStatus::Cancelled => write!(f, "Cancelled"),
            PipelineStatus::Timeout => write!(f, "Timeout"),
        }
    }
}

/// Quality gate result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGateResult {
    pub gate: QualityGate,
    pub status: QualityGateStatus,
    pub actual_value: f64,
    pub message: String,
}

/// Build artifact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub checksum: String,
    pub artifact_type: ArtifactType,
}

/// Artifact types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
    DebugReport,
    TestResults,
    BenchmarkResults,
    Model,
    Dataset,
    Documentation,
    Custom(String),
}

/// CI/CD integration manager
#[derive(Debug)]
pub struct CICDIntegration {
    config: CICDConfig,
    quality_gates: Vec<QualityGate>,
    baseline_metrics: HashMap<String, f64>,
    performance_history: Vec<PerformanceData>,
    pipeline_history: Vec<PipelineResult>,
}

impl CICDIntegration {
    /// Create a new CI/CD integration
    pub fn new(config: CICDConfig) -> Self {
        Self {
            config,
            quality_gates: Vec::new(),
            baseline_metrics: HashMap::new(),
            performance_history: Vec::new(),
            pipeline_history: Vec::new(),
        }
    }

    /// Add a quality gate
    pub fn add_quality_gate(&mut self, gate: QualityGate) {
        self.quality_gates.push(gate);
    }

    /// Set baseline metrics for regression detection
    pub fn set_baseline_metrics(&mut self, metrics: HashMap<String, f64>) {
        self.baseline_metrics = metrics;
    }

    /// Run debug analysis in CI/CD pipeline
    pub async fn run_debug_analysis(
        &mut self,
        commit_hash: String,
        branch: String,
        debug_config: DebugConfig,
    ) -> Result<PipelineResult> {
        let run_id = Uuid::new_v4();
        let start_time = Utc::now();

        tracing::info!(
            "Starting debug analysis for commit {} on branch {}",
            commit_hash,
            branch
        );

        // Create debug session
        let mut debug_session = DebugSession::new(debug_config);
        debug_session.start().await?;

        // Generate debug report -- this performs the real analysis (tensor
        // inspection, gradient debugging, memory profiling, ...); there is
        // nothing left to simulate here.
        let debug_report = debug_session.stop().await?;
        let analysis_duration_ms = (Utc::now() - start_time).num_milliseconds().max(0) as u64;

        // Extract metrics for quality gates and regression detection
        let metrics = self.extract_metrics_from_report(&debug_report, analysis_duration_ms);

        // Run quality gates
        let quality_gate_results = self.evaluate_quality_gates(&metrics);

        // Check for regressions
        let regression_results = self.detect_regressions(&metrics, &commit_hash);

        // Determine overall status
        let status = self.determine_pipeline_status(&quality_gate_results, &regression_results);

        // Create performance data
        let performance_data = PerformanceData {
            timestamp: start_time,
            commit_hash: commit_hash.clone(),
            branch: branch.clone(),
            metrics: metrics.clone(),
            benchmark_results: self.generate_benchmark_results(&metrics),
        };

        // Generate artifacts
        let artifacts = self.generate_artifacts(&debug_report, &performance_data)?;

        let duration_ms = (Utc::now() - start_time).num_milliseconds() as u64;

        let result = PipelineResult {
            run_id,
            timestamp: start_time,
            commit_hash,
            branch,
            stage: PipelineStage::Debug,
            status: status.clone(),
            debug_report: Some(debug_report),
            quality_gate_results,
            regression_results,
            performance_data: Some(performance_data.clone()),
            artifacts,
            duration_ms,
        };

        // Store results
        self.performance_history.push(performance_data);
        self.pipeline_history.push(result.clone());

        // Send notifications if configured
        if self.config.enable_alert_systems {
            self.send_notifications(&result).await?;
        }

        // Generate reports if configured
        if self.config.enable_automated_reports {
            self.generate_reports(&result).await?;
        }

        tracing::info!("Debug analysis completed with status: {:?}", status);

        Ok(result)
    }

    /// Extract metrics from debug report.
    ///
    /// `analysis_duration_ms` is the real wall-clock time
    /// [`Self::run_debug_analysis`] spent running the debug session, passed
    /// in explicitly since this function has no other way to measure it.
    fn extract_metrics_from_report(
        &self,
        report: &DebugReport,
        analysis_duration_ms: u64,
    ) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        // Extract tensor metrics
        if let Some(ref tensor_report) = report.tensor_report {
            metrics.insert(
                "tensor_nan_count".to_string(),
                tensor_report.total_nan_count() as f64,
            );
            metrics.insert(
                "tensor_inf_count".to_string(),
                tensor_report.total_inf_count() as f64,
            );
        }

        // Extract gradient metrics
        if let Some(ref gradient_report) = report.gradient_report {
            metrics.insert(
                "gradient_norm".to_string(),
                gradient_report.average_gradient_norm(),
            );
            metrics.insert(
                "vanishing_gradients".to_string(),
                gradient_report.vanishing_gradient_layers().len() as f64,
            );
            metrics.insert(
                "exploding_gradients".to_string(),
                gradient_report.exploding_gradient_layers().len() as f64,
            );
        }

        // Extract memory metrics
        if let Some(ref memory_report) = report.memory_profiler_report {
            metrics.insert(
                "peak_memory_mb".to_string(),
                memory_report.peak_memory_usage() / (1024.0 * 1024.0),
            );
            metrics.insert(
                "memory_efficiency".to_string(),
                memory_report.memory_efficiency(),
            );
        }

        // Extract performance metrics. `total_parameters` is only inserted
        // when a real count is available (see `count_model_parameters`) --
        // never a fabricated constant.
        if let Some(total_parameters) = self.count_model_parameters(report) {
            metrics.insert("total_parameters".to_string(), total_parameters as f64);
        }
        metrics.insert("training_time_ms".to_string(), analysis_duration_ms as f64);

        metrics
    }

    /// Evaluate quality gates against metrics
    fn evaluate_quality_gates(&self, metrics: &HashMap<String, f64>) -> Vec<QualityGateResult> {
        let mut results = Vec::new();

        for gate in &self.quality_gates {
            let metric_name = self.get_metric_name(&gate.metric);
            let actual_value = metrics.get(&metric_name).copied().unwrap_or(0.0);

            let passed = match gate.operator {
                ComparisonOperator::GreaterThan => actual_value > gate.threshold,
                ComparisonOperator::LessThan => actual_value < gate.threshold,
                ComparisonOperator::GreaterThanOrEqual => actual_value >= gate.threshold,
                ComparisonOperator::LessThanOrEqual => actual_value <= gate.threshold,
                ComparisonOperator::Equal => (actual_value - gate.threshold).abs() < f64::EPSILON,
                ComparisonOperator::NotEqual => {
                    (actual_value - gate.threshold).abs() >= f64::EPSILON
                },
            };

            let status = if passed { QualityGateStatus::Passed } else { QualityGateStatus::Failed };

            let message = format!(
                "Quality gate '{}': {} {} {} (actual: {})",
                gate.name,
                metric_name,
                self.operator_symbol(&gate.operator),
                gate.threshold,
                actual_value
            );

            results.push(QualityGateResult {
                gate: gate.clone(),
                status,
                actual_value,
                message,
            });
        }

        results
    }

    /// Detect regressions by comparing current metrics with baseline
    fn detect_regressions(
        &self,
        metrics: &HashMap<String, f64>,
        _commit_hash: &str,
    ) -> Vec<RegressionResult> {
        let mut results = Vec::new();

        if !self.config.enable_regression_detection {
            return results;
        }

        for (metric_name, &current_value) in metrics {
            if let Some(&baseline_value) = self.baseline_metrics.get(metric_name) {
                let change_percent = ((current_value - baseline_value) / baseline_value) * 100.0;

                // Determine if this is a regression based on metric type and change
                let (detected, severity) = self.analyze_regression(metric_name, change_percent);

                if detected {
                    results.push(RegressionResult {
                        detected: true,
                        severity,
                        metric: metric_name.clone(),
                        baseline_value,
                        current_value,
                        change_percent,
                        description: format!(
                            "Regression detected in {}: {:.2}% change from baseline (baseline: {:.4}, current: {:.4})",
                            metric_name, change_percent, baseline_value, current_value
                        ),
                    });
                }
            }
        }

        results
    }

    /// Analyze if a metric change constitutes a regression
    fn analyze_regression(
        &self,
        metric_name: &str,
        change_percent: f64,
    ) -> (bool, RegressionSeverity) {
        let abs_change = change_percent.abs();

        // Define regression thresholds based on metric type
        let (minor_threshold, major_threshold, critical_threshold) = match metric_name {
            name if name.contains("accuracy") => (2.0, 5.0, 10.0),
            name if name.contains("loss") => (5.0, 15.0, 30.0),
            name if name.contains("memory") => (10.0, 25.0, 50.0),
            name if name.contains("time") => (15.0, 30.0, 60.0),
            _ => (5.0, 15.0, 30.0), // Default thresholds
        };

        if abs_change >= critical_threshold {
            (true, RegressionSeverity::Critical)
        } else if abs_change >= major_threshold {
            (true, RegressionSeverity::Major)
        } else if abs_change >= minor_threshold {
            (true, RegressionSeverity::Minor)
        } else {
            (false, RegressionSeverity::Info)
        }
    }

    /// Determine overall pipeline status
    fn determine_pipeline_status(
        &self,
        quality_gate_results: &[QualityGateResult],
        regression_results: &[RegressionResult],
    ) -> PipelineStatus {
        // Check for blocking quality gate failures
        for result in quality_gate_results {
            if result.gate.blocking && matches!(result.status, QualityGateStatus::Failed) {
                return PipelineStatus::Failed;
            }
        }

        // Check for critical regressions
        for regression in regression_results {
            if matches!(regression.severity, RegressionSeverity::Critical) {
                return PipelineStatus::Failed;
            }
        }

        // Check for major regressions or non-blocking quality gate failures
        let has_warnings = quality_gate_results
            .iter()
            .any(|r| matches!(r.status, QualityGateStatus::Failed))
            || regression_results
                .iter()
                .any(|r| matches!(r.severity, RegressionSeverity::Major));

        if has_warnings {
            PipelineStatus::Warning
        } else {
            PipelineStatus::Success
        }
    }

    /// Generate benchmark results
    fn generate_benchmark_results(&self, metrics: &HashMap<String, f64>) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();

        for (name, &value) in metrics {
            let baseline = self.baseline_metrics.get(name).copied();
            let improvement_percent = baseline.map(|b| ((value - b) / b) * 100.0);

            let unit = match name.as_str() {
                name if name.contains("time") || name.contains("latency") => "ms",
                name if name.contains("memory") => "MB",
                name if name.contains("accuracy") => "%",
                name if name.contains("loss") => "loss",
                _ => "units",
            };

            results.push(BenchmarkResult {
                name: name.clone(),
                value,
                unit: unit.to_string(),
                baseline,
                improvement_percent,
            });
        }

        results
    }

    /// Generate artifacts from analysis results
    fn generate_artifacts(
        &self,
        debug_report: &DebugReport,
        performance_data: &PerformanceData,
    ) -> Result<Vec<Artifact>> {
        let mut artifacts = Vec::new();

        // Generate debug report artifact
        let debug_report_json = serde_json::to_string_pretty(debug_report)?;
        let debug_report_path = PathBuf::from("debug_report.json");
        std::fs::write(&debug_report_path, &debug_report_json)?;

        artifacts.push(Artifact {
            name: "Debug Report".to_string(),
            path: debug_report_path,
            size_bytes: debug_report_json.len() as u64,
            checksum: format!("{:x}", md5::compute(&debug_report_json)),
            artifact_type: ArtifactType::DebugReport,
        });

        // Generate performance data artifact
        let performance_json = serde_json::to_string_pretty(performance_data)?;
        let performance_path = PathBuf::from("performance_data.json");
        std::fs::write(&performance_path, &performance_json)?;

        artifacts.push(Artifact {
            name: "Performance Data".to_string(),
            path: performance_path,
            size_bytes: performance_json.len() as u64,
            checksum: format!("{:x}", md5::compute(&performance_json)),
            artifact_type: ArtifactType::BenchmarkResults,
        });

        Ok(artifacts)
    }

    /// Send notifications based on pipeline result
    async fn send_notifications(&self, result: &PipelineResult) -> Result<()> {
        for channel in &self.config.notification_channels {
            match channel {
                NotificationChannel::Slack {
                    webhook_url,
                    channel: slack_channel,
                } => {
                    self.send_slack_notification(webhook_url, slack_channel, result).await?;
                },
                NotificationChannel::Email { recipients, api } => {
                    self.send_email_notification(recipients, api.as_ref(), result).await?;
                },
                NotificationChannel::Teams { webhook_url } => {
                    self.send_teams_notification(webhook_url, result).await?;
                },
                NotificationChannel::Discord { webhook_url } => {
                    self.send_discord_notification(webhook_url, result).await?;
                },
                NotificationChannel::Webhook { url, headers } => {
                    self.send_webhook_notification(url, headers, result).await?;
                },
                NotificationChannel::Custom(_) => {
                    return Err(NotificationError::NotImplemented("Custom").into());
                },
            }
        }

        Ok(())
    }

    /// Generate reports in various formats
    async fn generate_reports(&self, result: &PipelineResult) -> Result<()> {
        for format in &self.config.report_formats {
            match format {
                ReportFormat::JSON => {
                    let json_report = serde_json::to_string_pretty(result)?;
                    std::fs::write("cicd_report.json", json_report)?;
                },
                ReportFormat::HTML => {
                    let html_report = self.generate_html_report(result)?;
                    std::fs::write("cicd_report.html", html_report)?;
                },
                ReportFormat::Markdown => {
                    let md_report = self.generate_markdown_report(result)?;
                    std::fs::write("cicd_report.md", md_report)?;
                },
                ReportFormat::JUnit => {
                    let junit_report = self.generate_junit_report(result)?;
                    std::fs::write("cicd_report.xml", junit_report)?;
                },
                _ => {
                    tracing::info!("Report format {:?} not implemented", format);
                },
            }
        }

        Ok(())
    }

    /// Helper methods for notification and report generation.
    ///
    /// Every sender builds its real payload unconditionally (so callers,
    /// tests and mocks all see the exact JSON that would be sent), then
    /// delivers it via [`post_json`] when the `http-integrations` feature is
    /// enabled. Without that feature no HTTP client exists in this build,
    /// so delivery honestly fails with
    /// [`NotificationError::HttpFeatureDisabled`] rather than pretending to
    /// have sent anything.
    async fn send_slack_notification(
        &self,
        webhook_url: &str,
        channel: &str,
        result: &PipelineResult,
    ) -> Result<()> {
        let color = match result.status {
            PipelineStatus::Success => "good",
            PipelineStatus::Warning => "warning",
            PipelineStatus::Failed => "danger",
            _ => "warning",
        };

        let message = serde_json::json!({
            "channel": channel,
            "attachments": [{
                "color": color,
                "title": format!("Debug Analysis - {}", result.commit_hash),
                "text": format!("Branch: {} | Status: {:?} | Duration: {}ms",
                    result.branch, result.status, result.duration_ms),
                "fields": [
                    {
                        "title": "Quality Gates",
                        "value": format!("{} passed, {} failed",
                            result.quality_gate_results.iter().filter(|r| matches!(r.status, QualityGateStatus::Passed)).count(),
                            result.quality_gate_results.iter().filter(|r| matches!(r.status, QualityGateStatus::Failed)).count()),
                        "short": true
                    },
                    {
                        "title": "Regressions",
                        "value": format!("{} detected", result.regression_results.len()),
                        "short": true
                    }
                ]
            }]
        });

        post_json(webhook_url, &message, &HashMap::new()).await?;
        tracing::info!(webhook_url = %webhook_url, "Slack notification delivered");
        Ok(())
    }

    async fn send_email_notification(
        &self,
        recipients: &[String],
        api: Option<&EmailApiConfig>,
        result: &PipelineResult,
    ) -> Result<()> {
        let Some(api) = api else {
            return Err(NotificationError::EmailNotConfigured.into());
        };

        let subject = format!(
            "Debug Analysis Report - {} ({})",
            result.commit_hash, result.status
        );
        let body = format!(
            "Debug analysis completed for commit {} on branch {}.\n\nStatus: {:?}\nDuration: {}ms\n\nQuality Gates: {} passed, {} failed\nRegressions: {} detected",
            result.commit_hash,
            result.branch,
            result.status,
            result.duration_ms,
            result.quality_gate_results.iter().filter(|r| matches!(r.status, QualityGateStatus::Passed)).count(),
            result.quality_gate_results.iter().filter(|r| matches!(r.status, QualityGateStatus::Failed)).count(),
            result.regression_results.len()
        );

        let payload = serde_json::json!({
            "to": recipients,
            "subject": subject,
            "body": body,
        });

        post_json(&api.endpoint, &payload, &api.headers).await?;
        tracing::info!(recipients = ?recipients, endpoint = %api.endpoint, "Email notification delivered");
        Ok(())
    }

    async fn send_teams_notification(
        &self,
        webhook_url: &str,
        result: &PipelineResult,
    ) -> Result<()> {
        // Office 365 Connector "MessageCard" format.
        let theme_color = match result.status {
            PipelineStatus::Success => "28A745",
            PipelineStatus::Warning => "FFC107",
            PipelineStatus::Failed => "DC3545",
            _ => "6C757D",
        };

        let message = serde_json::json!({
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": theme_color,
            "summary": format!("Debug Analysis - {}", result.commit_hash),
            "sections": [{
                "activityTitle": format!("Debug Analysis - {}", result.commit_hash),
                "facts": [
                    {"name": "Branch", "value": result.branch},
                    {"name": "Status", "value": format!("{:?}", result.status)},
                    {"name": "Duration", "value": format!("{}ms", result.duration_ms)},
                    {"name": "Quality Gates", "value": format!("{} passed, {} failed",
                        result.quality_gate_results.iter().filter(|r| matches!(r.status, QualityGateStatus::Passed)).count(),
                        result.quality_gate_results.iter().filter(|r| matches!(r.status, QualityGateStatus::Failed)).count())},
                    {"name": "Regressions", "value": format!("{} detected", result.regression_results.len())},
                ],
            }],
        });

        post_json(webhook_url, &message, &HashMap::new()).await?;
        tracing::info!(webhook_url = %webhook_url, "Teams notification delivered");
        Ok(())
    }

    async fn send_discord_notification(
        &self,
        webhook_url: &str,
        result: &PipelineResult,
    ) -> Result<()> {
        // Discord webhook "embeds" format.
        let color = match result.status {
            PipelineStatus::Success => 0x28_A7_45,
            PipelineStatus::Warning => 0xFF_C1_07,
            PipelineStatus::Failed => 0xDC_35_45,
            _ => 0x6C_75_7D,
        };

        let message = serde_json::json!({
            "embeds": [{
                "title": format!("Debug Analysis - {}", result.commit_hash),
                "description": format!("Branch: {} | Status: {:?} | Duration: {}ms",
                    result.branch, result.status, result.duration_ms),
                "color": color,
                "fields": [
                    {
                        "name": "Quality Gates",
                        "value": format!("{} passed, {} failed",
                            result.quality_gate_results.iter().filter(|r| matches!(r.status, QualityGateStatus::Passed)).count(),
                            result.quality_gate_results.iter().filter(|r| matches!(r.status, QualityGateStatus::Failed)).count()),
                        "inline": true
                    },
                    {
                        "name": "Regressions",
                        "value": format!("{} detected", result.regression_results.len()),
                        "inline": true
                    }
                ]
            }]
        });

        post_json(webhook_url, &message, &HashMap::new()).await?;
        tracing::info!(webhook_url = %webhook_url, "Discord notification delivered");
        Ok(())
    }

    async fn send_webhook_notification(
        &self,
        url: &str,
        headers: &HashMap<String, String>,
        result: &PipelineResult,
    ) -> Result<()> {
        let payload = serde_json::to_value(result)?;
        post_json(url, &payload, headers).await?;
        tracing::info!(url = %url, "Generic webhook notification delivered");
        Ok(())
    }

    /// Generate HTML report
    fn generate_html_report(&self, result: &PipelineResult) -> Result<String> {
        let html = format!(r#"
<!DOCTYPE html>
<html>
<head>
    <title>Debug Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .status-success {{ color: green; }}
        .status-warning {{ color: orange; }}
        .status-failed {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Debug Analysis Report</h1>
    <h2>Overview</h2>
    <p><strong>Commit:</strong> {}</p>
    <p><strong>Branch:</strong> {}</p>
    <p><strong>Status:</strong> <span class="status-{}">{:?}</span></p>
    <p><strong>Duration:</strong> {}ms</p>
    <p><strong>Timestamp:</strong> {}</p>

    <h2>Quality Gates</h2>
    <table>
        <tr><th>Gate</th><th>Status</th><th>Actual Value</th><th>Threshold</th><th>Message</th></tr>
        {}
    </table>

    <h2>Regression Analysis</h2>
    <table>
        <tr><th>Metric</th><th>Severity</th><th>Change %</th><th>Baseline</th><th>Current</th><th>Description</th></tr>
        {}
    </table>
</body>
</html>
"#,
            result.commit_hash,
            result.branch,
            format!("{:?}", result.status).to_lowercase(),
            result.status,
            result.duration_ms,
            result.timestamp,
            result.quality_gate_results.iter().map(|r| format!(
                "<tr><td>{}</td><td>{:?}</td><td>{:.4}</td><td>{:.4}</td><td>{}</td></tr>",
                r.gate.name, r.status, r.actual_value, r.gate.threshold, r.message
            )).collect::<Vec<_>>().join(""),
            result.regression_results.iter().map(|r| format!(
                "<tr><td>{}</td><td>{:?}</td><td>{:.2}%</td><td>{:.4}</td><td>{:.4}</td><td>{}</td></tr>",
                r.metric, r.severity, r.change_percent, r.baseline_value, r.current_value, r.description
            )).collect::<Vec<_>>().join("")
        );

        Ok(html)
    }

    /// Generate Markdown report
    fn generate_markdown_report(&self, result: &PipelineResult) -> Result<String> {
        let status_emoji = match result.status {
            PipelineStatus::Success => "✅",
            PipelineStatus::Warning => "⚠️",
            PipelineStatus::Failed => "❌",
            _ => "❓",
        };

        let markdown = format!(
            r#"# Debug Analysis Report

## Overview
- **Commit:** {}
- **Branch:** {}
- **Status:** {} {:?}
- **Duration:** {}ms
- **Timestamp:** {}

## Quality Gates
| Gate | Status | Actual Value | Threshold | Message |
|------|--------|--------------|-----------|---------|
{}

## Regression Analysis
| Metric | Severity | Change % | Baseline | Current | Description |
|--------|----------|----------|----------|---------|-------------|
{}

## Artifacts
{}
"#,
            result.commit_hash,
            result.branch,
            status_emoji,
            result.status,
            result.duration_ms,
            result.timestamp,
            result
                .quality_gate_results
                .iter()
                .map(|r| format!(
                    "| {} | {:?} | {:.4} | {:.4} | {} |",
                    r.gate.name, r.status, r.actual_value, r.gate.threshold, r.message
                ))
                .collect::<Vec<_>>()
                .join("\n"),
            result
                .regression_results
                .iter()
                .map(|r| format!(
                    "| {} | {:?} | {:.2}% | {:.4} | {:.4} | {} |",
                    r.metric,
                    r.severity,
                    r.change_percent,
                    r.baseline_value,
                    r.current_value,
                    r.description
                ))
                .collect::<Vec<_>>()
                .join("\n"),
            result
                .artifacts
                .iter()
                .map(|a| format!(
                    "- **{}:** {} ({} bytes)",
                    a.name,
                    a.path.display(),
                    a.size_bytes
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(markdown)
    }

    /// Generate JUnit XML report
    fn generate_junit_report(&self, result: &PipelineResult) -> Result<String> {
        let test_cases = result
            .quality_gate_results
            .iter()
            .map(|r| {
                let status = match r.status {
                    QualityGateStatus::Passed => "",
                    QualityGateStatus::Failed => {
                        r#"<failure message="Quality gate failed"></failure>"#
                    },
                    QualityGateStatus::Warning => {
                        r#"<error message="Quality gate warning"></error>"#
                    },
                    QualityGateStatus::Skipped => r#"<skipped/>"#,
                };
                format!(
                    r#"<testcase classname="QualityGates" name="{}" time="0">{}</testcase>"#,
                    r.gate.name, status
                )
            })
            .collect::<Vec<_>>()
            .join("\n    ");

        let junit = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="DebugAnalysis" tests="{}" failures="{}" errors="0" time="{:.3}">
    {}
</testsuite>
"#,
            result.quality_gate_results.len(),
            result
                .quality_gate_results
                .iter()
                .filter(|r| matches!(r.status, QualityGateStatus::Failed))
                .count(),
            result.duration_ms as f64 / 1000.0,
            test_cases
        );

        Ok(junit)
    }

    /// Helper methods
    fn get_metric_name(&self, metric: &QualityMetric) -> String {
        match metric {
            QualityMetric::TestCoverage => "test_coverage".to_string(),
            QualityMetric::ModelAccuracy => "model_accuracy".to_string(),
            QualityMetric::TrainingLoss => "training_loss".to_string(),
            QualityMetric::GradientNorm => "gradient_norm".to_string(),
            QualityMetric::MemoryUsage => "peak_memory_mb".to_string(),
            QualityMetric::TrainingTime => "training_time_ms".to_string(),
            QualityMetric::ModelSize => "total_parameters".to_string(),
            QualityMetric::InferenceLatency => "inference_latency_ms".to_string(),
            QualityMetric::Custom(name) => name.clone(),
        }
    }

    fn operator_symbol(&self, op: &ComparisonOperator) -> &'static str {
        match op {
            ComparisonOperator::GreaterThan => ">",
            ComparisonOperator::LessThan => "<",
            ComparisonOperator::GreaterThanOrEqual => ">=",
            ComparisonOperator::LessThanOrEqual => "<=",
            ComparisonOperator::Equal => "==",
            ComparisonOperator::NotEqual => "!=",
        }
    }

    /// Real parameter count, if one is reachable from the debug report.
    /// `DebugReport` does not currently carry model architecture metadata
    /// (see `ModelDiagnosticsReport`/`ArchitecturalAnalysis`), so this
    /// honestly returns `None` rather than fabricating a number; callers
    /// must simply omit the `total_parameters` metric until a real source
    /// is wired through the debug session.
    fn count_model_parameters(&self, _report: &DebugReport) -> Option<u64> {
        None
    }

    /// Get pipeline history
    pub fn get_pipeline_history(&self) -> &[PipelineResult] {
        &self.pipeline_history
    }

    /// Get performance history
    pub fn get_performance_history(&self) -> &[PerformanceData] {
        &self.performance_history
    }

    /// Get quality gates
    pub fn get_quality_gates(&self) -> &[QualityGate] {
        &self.quality_gates
    }
}

impl Default for CICDConfig {
    fn default() -> Self {
        Self {
            platform: CICDPlatform::GitHub,
            project_id: "default".to_string(),
            api_token: None,
            base_url: None,
            branch_filters: vec!["main".to_string(), "develop".to_string()],
            enable_regression_detection: true,
            enable_performance_tracking: true,
            enable_quality_gates: true,
            enable_automated_reports: true,
            enable_alert_systems: true,
            report_formats: vec![
                ReportFormat::JSON,
                ReportFormat::HTML,
                ReportFormat::Markdown,
            ],
            notification_channels: Vec::new(),
        }
    }
}

// Additional accessor implementations used when extracting CI/CD metrics
// from a debug report. `DebugReport` itself intentionally has no
// `total_nan_count`/`total_inf_count` here: those real, data-derived
// methods already live on `TensorInspectionReport`
// (`report.tensor_report.total_nan_count()`), which is what
// `extract_metrics_from_report` actually calls.

impl crate::GradientDebugReport {
    /// Real mean of the latest per-layer gradient norms. `0.0` when no layer
    /// status has been recorded yet (rather than a fabricated `1.0`).
    pub fn average_gradient_norm(&self) -> f64 {
        let norms: Vec<f64> =
            self.status.layer_statuses.values().map(|s| s.latest_gradient_norm).collect();
        mean_gradient_norm(&norms)
    }

    /// Real layer names whose latest gradient norm is vanishing, using the
    /// same `1e-8` threshold as [`Self::has_vanishing_gradients`].
    pub fn vanishing_gradient_layers(&self) -> Vec<String> {
        let norms: HashMap<String, f64> = self
            .status
            .layer_statuses
            .iter()
            .map(|(name, status)| (name.clone(), status.latest_gradient_norm))
            .collect();
        layers_matching(&norms, |norm| norm < 1e-8)
    }

    /// Real layer names whose latest gradient norm is exploding, using the
    /// same `100.0` threshold as [`Self::has_exploding_gradients`].
    pub fn exploding_gradient_layers(&self) -> Vec<String> {
        let norms: HashMap<String, f64> = self
            .status
            .layer_statuses
            .iter()
            .map(|(name, status)| (name.clone(), status.latest_gradient_norm))
            .collect();
        layers_matching(&norms, |norm| norm > 100.0)
    }
}

/// Real mean of a set of gradient norms. `0.0` when empty (rather than a
/// fabricated default).
fn mean_gradient_norm(norms: &[f64]) -> f64 {
    if norms.is_empty() {
        0.0
    } else {
        norms.iter().sum::<f64>() / norms.len() as f64
    }
}

/// Names of layers whose norm satisfies `predicate` -- shared by both
/// vanishing- and exploding-gradient detection so they stay in lock-step.
fn layers_matching(norms: &HashMap<String, f64>, predicate: impl Fn(f64) -> bool) -> Vec<String> {
    norms
        .iter()
        .filter(|(_, &norm)| predicate(norm))
        .map(|(name, _)| name.clone())
        .collect()
}

impl crate::MemoryProfilingReport {
    /// Real peak memory usage in bytes, derived from the already-real
    /// `peak_memory_mb` field this report is built from.
    pub fn peak_memory_usage(&self) -> f64 {
        self.peak_memory_mb * 1024.0 * 1024.0
    }

    /// Real memory-efficiency proxy: `1 - fragmentation_ratio`, both drawn
    /// from this report's already-real fragmentation analysis.
    pub fn memory_efficiency(&self) -> f64 {
        (1.0 - self.fragmentation_analysis.fragmentation_ratio).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result(status: PipelineStatus) -> PipelineResult {
        PipelineResult {
            run_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            commit_hash: "abc123".to_string(),
            branch: "main".to_string(),
            stage: PipelineStage::Debug,
            status,
            debug_report: None,
            quality_gate_results: Vec::new(),
            regression_results: Vec::new(),
            performance_data: None,
            artifacts: Vec::new(),
            duration_ms: 42,
        }
    }

    // ------------------------------------------------------------------
    // Real per-report accessor methods (no longer hardcoded placeholders)
    // ------------------------------------------------------------------

    #[test]
    fn test_mean_gradient_norm_is_real_not_fixed_one() {
        // The old implementation always returned 1.0 regardless of input.
        assert_eq!(mean_gradient_norm(&[]), 0.0);
        assert_eq!(mean_gradient_norm(&[2.0, 4.0, 6.0]), 4.0);
        assert_ne!(mean_gradient_norm(&[2.0, 4.0, 6.0]), 1.0);
    }

    #[test]
    fn test_layers_matching_finds_real_vanishing_and_exploding_layers() {
        let mut norms = HashMap::new();
        norms.insert("healthy".to_string(), 0.5);
        norms.insert("vanished".to_string(), 1e-10);
        norms.insert("exploded".to_string(), 500.0);

        // The old implementation always returned an empty Vec regardless of
        // input, for both vanishing and exploding layers.
        let vanishing = layers_matching(&norms, |n| n < 1e-8);
        assert_eq!(vanishing, vec!["vanished".to_string()]);

        let exploding = layers_matching(&norms, |n| n > 100.0);
        assert_eq!(exploding, vec!["exploded".to_string()]);
    }

    #[tokio::test]
    async fn test_memory_report_accessors_reflect_real_profiler_fields() {
        use crate::memory_profiler::{AllocationType, MemoryProfiler, MemoryProfilingConfig};

        let mut profiler = MemoryProfiler::new(MemoryProfilingConfig::default());
        profiler.start().await.expect("profiler should start");
        let id = profiler
            .record_allocation(4096, AllocationType::Tensor, vec!["test".to_string()])
            .expect("allocation should record");
        profiler.record_deallocation(id).expect("deallocation should record");
        let report = profiler.stop().await.expect("profiler should stop");

        // The old implementation ignored the report entirely and always
        // returned a hardcoded 100MB / 0.85, regardless of its real fields.
        assert_eq!(
            report.peak_memory_usage(),
            report.peak_memory_mb * 1024.0 * 1024.0
        );
        assert_eq!(
            report.memory_efficiency(),
            (1.0 - report.fragmentation_analysis.fragmentation_ratio).clamp(0.0, 1.0)
        );
    }

    #[tokio::test]
    async fn test_count_model_parameters_is_honest_absence_not_a_million() {
        let integration = CICDIntegration::new(CICDConfig::default());
        let mut session = DebugSession::new(DebugConfig::default());
        session.start().await.expect("session should start");
        let report = session.stop().await.expect("session should stop");

        // The old implementation always returned 1_000_000 regardless of
        // whether any model was ever involved.
        assert_eq!(integration.count_model_parameters(&report), None);
    }

    // ------------------------------------------------------------------
    // Notification delivery: real payload shape via a local mock server,
    // and an honest, structured failure when the feature is disabled.
    // ------------------------------------------------------------------

    #[cfg(feature = "http-integrations")]
    mod http_delivery {
        use super::*;
        use axum::{extract::State, routing::post, Json, Router};
        use std::sync::Arc;
        use tokio::net::TcpListener;
        use tokio::sync::Mutex as AsyncMutex;

        #[derive(Default, Clone)]
        struct Captured {
            body: Option<serde_json::Value>,
        }

        async fn capture(
            State(state): State<Arc<AsyncMutex<Captured>>>,
            Json(body): Json<serde_json::Value>,
        ) -> &'static str {
            state.lock().await.body = Some(body);
            "ok"
        }

        /// Start a real local HTTP server (127.0.0.1, OS-assigned port; never
        /// touches the network) that captures whatever JSON body is POSTed
        /// to `/hook`.
        async fn start_mock_server() -> (String, Arc<AsyncMutex<Captured>>) {
            let state = Arc::new(AsyncMutex::new(Captured::default()));
            let app = Router::new().route("/hook", post(capture)).with_state(state.clone());
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock server");
            let addr = listener.local_addr().expect("mock server local addr");
            tokio::spawn(async move {
                let _ = axum::serve(listener, app).await;
            });
            (format!("http://{addr}/hook"), state)
        }

        #[tokio::test]
        async fn test_slack_notification_sends_real_payload() {
            let (url, state) = start_mock_server().await;
            let integration = CICDIntegration::new(CICDConfig::default());
            let result = sample_result(PipelineStatus::Success);

            integration
                .send_slack_notification(&url, "#ci-alerts", &result)
                .await
                .expect("mock server should accept the request");

            let captured = state.lock().await.body.clone().expect("payload should be captured");
            assert_eq!(captured["channel"], "#ci-alerts");
            assert_eq!(captured["attachments"][0]["color"], "good");
            assert!(captured["attachments"][0]["title"]
                .as_str()
                .expect("title should be a string")
                .contains(&result.commit_hash));
        }

        #[tokio::test]
        async fn test_teams_notification_sends_real_payload() {
            let (url, state) = start_mock_server().await;
            let integration = CICDIntegration::new(CICDConfig::default());
            let result = sample_result(PipelineStatus::Failed);

            integration
                .send_teams_notification(&url, &result)
                .await
                .expect("mock server should accept the request");

            let captured = state.lock().await.body.clone().expect("payload should be captured");
            assert_eq!(captured["@type"], "MessageCard");
            assert_eq!(captured["themeColor"], "DC3545");
        }

        #[tokio::test]
        async fn test_discord_notification_sends_real_payload() {
            let (url, state) = start_mock_server().await;
            let integration = CICDIntegration::new(CICDConfig::default());
            let result = sample_result(PipelineStatus::Warning);

            integration
                .send_discord_notification(&url, &result)
                .await
                .expect("mock server should accept the request");

            let captured = state.lock().await.body.clone().expect("payload should be captured");
            assert!(captured["embeds"][0]["title"]
                .as_str()
                .expect("title should be a string")
                .contains(&result.commit_hash));
        }

        #[tokio::test]
        async fn test_generic_webhook_sends_real_pipeline_result() {
            let (url, state) = start_mock_server().await;
            let integration = CICDIntegration::new(CICDConfig::default());
            let result = sample_result(PipelineStatus::Success);

            integration
                .send_webhook_notification(&url, &HashMap::new(), &result)
                .await
                .expect("mock server should accept the request");

            let captured = state.lock().await.body.clone().expect("payload should be captured");
            assert_eq!(captured["commit_hash"], result.commit_hash);
            assert_eq!(captured["duration_ms"], result.duration_ms);
        }

        #[tokio::test]
        async fn test_email_notification_sends_real_payload_when_configured() {
            let (url, state) = start_mock_server().await;
            let integration = CICDIntegration::new(CICDConfig::default());
            let result = sample_result(PipelineStatus::Success);
            let api = EmailApiConfig {
                endpoint: url,
                headers: HashMap::new(),
            };

            integration
                .send_email_notification(&["dev@example.com".to_string()], Some(&api), &result)
                .await
                .expect("mock server should accept the request");

            let captured = state.lock().await.body.clone().expect("payload should be captured");
            assert_eq!(captured["to"][0], "dev@example.com");
            assert!(captured["subject"]
                .as_str()
                .expect("subject should be a string")
                .contains(&result.commit_hash));
        }
    }

    #[cfg(not(feature = "http-integrations"))]
    mod http_disabled {
        use super::*;

        #[tokio::test]
        async fn test_slack_notification_fails_honestly_without_feature() {
            let integration = CICDIntegration::new(CICDConfig::default());
            let result = sample_result(PipelineStatus::Success);

            let err = integration
                .send_slack_notification("http://127.0.0.1:1/hook", "#ci", &result)
                .await
                .expect_err("must not silently pretend to have sent anything");
            assert!(err.to_string().contains("http-integrations"));
        }
    }

    #[tokio::test]
    async fn test_email_notification_is_honestly_not_configured_without_api() {
        let integration = CICDIntegration::new(CICDConfig::default());
        let result = sample_result(PipelineStatus::Success);

        let err = integration
            .send_email_notification(&["dev@example.com".to_string()], None, &result)
            .await
            .expect_err("must not silently pretend to have sent an email");
        assert!(err.to_string().contains("endpoint configured"));
    }

    #[tokio::test]
    async fn test_custom_notification_channel_errors_instead_of_silently_succeeding() {
        let mut config = CICDConfig::default();
        config.enable_alert_systems = true;
        config.notification_channels = vec![NotificationChannel::Custom("pagerduty".to_string())];
        let integration = CICDIntegration::new(config);
        let result = sample_result(PipelineStatus::Success);

        // The old implementation logged "not implemented" and then returned
        // `Ok(())` from the whole dispatch loop -- a caller had no way to
        // tell the notification never went anywhere.
        let err = integration
            .send_notifications(&result)
            .await
            .expect_err("an unimplemented channel must not report success");
        assert!(err.to_string().contains("no delivery implementation"));
    }
}
