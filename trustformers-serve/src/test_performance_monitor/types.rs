//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::test_timeout_optimization::{
    FrameworkStats, PerformanceRegression, TestCategory, TestExecutionResult, TestOutcome,
    TestTimeoutFramework,
};
use anyhow::Result;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::{atomic::Ordering, Arc},
    time::Duration,
};
use tokio::task::JoinHandle;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

/// Alert state
#[derive(Debug, Clone)]
pub struct AlertState {
    /// Last triggered time
    pub last_triggered: Option<DateTime<Utc>>,
    /// Trigger count
    pub trigger_count: usize,
    /// Current status
    pub status: AlertStatus,
    /// Escalation level
    pub escalation_level: usize,
}
/// Performance trends summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendsSummary {
    /// Execution time trend
    pub execution_time_trend: TrendDirection,
    /// Success rate trend
    pub success_rate_trend: TrendDirection,
    /// Optimization effectiveness trend
    pub optimization_effectiveness_trend: TrendDirection,
    /// Resource usage trend
    pub resource_usage_trend: TrendDirection,
}
/// Dashboard server
pub struct DashboardServer {
    /// Server configuration
    config: DashboardConfig,
    /// Data store reference
    _data_store: Arc<PerformanceDataStore>,
    /// Server handle
    server_handle: std::sync::Mutex<Option<JoinHandle<()>>>,
}
impl DashboardServer {
    fn new(config: DashboardConfig, data_store: Arc<PerformanceDataStore>) -> Result<Self> {
        Ok(Self {
            config,
            _data_store: data_store,
            server_handle: std::sync::Mutex::new(None),
        })
    }
    async fn start(&self) -> Result<()> {
        info!(
            host = % self.config.host, port = self.config.port,
            "Dashboard server would start here"
        );
        Ok(())
    }
    async fn stop(&self) -> Result<()> {
        if let Some(handle) = self.server_handle.lock().expect("lock should not be poisoned").take()
        {
            handle.abort();
        }
        info!("Dashboard server stopped");
        Ok(())
    }
}
/// Email distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailDistributionConfig {
    /// Enable email distribution
    pub enabled: bool,
    /// Recipients
    pub recipients: Vec<String>,
    /// Email template
    pub template: String,
}
/// Notification channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    /// Channel name
    pub name: String,
    /// Channel type
    pub channel_type: NotificationChannelType,
    /// Channel configuration
    pub config: NotificationChannelConfig,
    /// Channel enabled flag
    pub enabled: bool,
}
/// Optimization summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSummary {
    /// Total optimizations applied
    pub total_optimizations: usize,
    /// Time saved through optimizations
    pub time_saved: Duration,
    /// Optimization success rate
    pub optimization_success_rate: f32,
    /// Most effective optimizations
    pub top_optimizations: Vec<(String, f32)>,
}
/// Report generator
pub struct ReportGenerator {
    /// Report templates
    templates: Arc<Mutex<HashMap<String, ReportTemplate>>>,
    /// Generated reports history
    reports_history: Arc<Mutex<VecDeque<GeneratedReport>>>,
    /// Distribution channels
    _distribution_channels: Arc<Mutex<Vec<DistributionChannel>>>,
}
impl ReportGenerator {
    fn new(config: &ReportingConfig) -> Result<Self> {
        let mut templates = HashMap::new();
        for template in &config.templates {
            templates.insert(template.name.clone(), template.clone());
        }
        Ok(Self {
            templates: Arc::new(Mutex::new(templates)),
            reports_history: Arc::new(Mutex::new(VecDeque::new())),
            _distribution_channels: Arc::new(Mutex::new(Vec::new())),
        })
    }
    async fn check_and_generate_reports(&self, data_store: &PerformanceDataStore) -> Result<()> {
        let should_generate_daily = true;
        if should_generate_daily {
            self.generate_daily_report(data_store).await?;
        }
        Ok(())
    }
    async fn generate_daily_report(&self, data_store: &PerformanceDataStore) -> Result<()> {
        let default_template = {
            let templates = self.templates.lock();
            templates
                .get("default")
                .ok_or_else(|| anyhow::anyhow!("Default template not found"))?
                .clone()
        };
        let report_content = self.render_template(&default_template, data_store).await?;
        let generated_report = GeneratedReport {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            template: "default".to_string(),
            format: ReportFormat::Html,
            content: report_content,
            metadata: HashMap::new(),
        };
        {
            let mut reports_history = self.reports_history.lock();
            reports_history.push_back(generated_report.clone());
        }
        self.distribute_report(&generated_report).await?;
        info!(report_id = % generated_report.id, "Generated daily report");
        Ok(())
    }
    async fn render_template(
        &self,
        _template: &ReportTemplate,
        data_store: &PerformanceDataStore,
    ) -> Result<String> {
        let execution_results = data_store.execution_results.lock();
        let total_tests = execution_results.len();
        let successful_tests = execution_results
            .iter()
            .filter(|r| matches!(r.outcome, TestOutcome::Success))
            .count();
        let success_rate = if total_tests > 0 {
            successful_tests as f32 / total_tests as f32 * 100.0
        } else {
            0.0
        };
        let report = format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>Test Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
        .success {{ background-color: #d4edda; }}
        .warning {{ background-color: #fff3cd; }}
        .danger {{ background-color: #f8d7da; }}
    </style>
</head>
<body>
    <h1>Test Performance Report</h1>
    <p>Generated: {}</p>

    <div class="metric success">
        <h3>Total Tests Executed</h3>
        <p>{}</p>
    </div>

    <div class="metric {}">
        <h3>Success Rate</h3>
        <p>{:.1}%</p>
    </div>

    <div class="metric">
        <h3>Test Categories</h3>
        <ul>
            <li>Unit Tests: {}</li>
            <li>Integration Tests: {}</li>
            <li>End-to-End Tests: {}</li>
        </ul>
    </div>
</body>
</html>
            "#,
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
            total_tests,
            if success_rate >= 95.0 {
                "success"
            } else if success_rate >= 80.0 {
                "warning"
            } else {
                "danger"
            },
            success_rate,
            execution_results
                .iter()
                .filter(|r| r.context.category == TestCategory::Unit)
                .count(),
            execution_results
                .iter()
                .filter(|r| r.context.category == TestCategory::Integration)
                .count(),
            execution_results
                .iter()
                .filter(|r| r.context.category == TestCategory::EndToEnd)
                .count(),
        );
        Ok(report)
    }
    async fn distribute_report(&self, report: &GeneratedReport) -> Result<()> {
        let filename = format!(
            "test_report_{}_{}.html",
            report.timestamp.format("%Y%m%d_%H%M%S"),
            report.id
        );
        tokio::fs::create_dir_all("./reports").await?;
        tokio::fs::write(format!("./reports/{}", filename), &report.content).await?;
        info!(
            report_id = % report.id, filename = % filename,
            "Report distributed to filesystem"
        );
        Ok(())
    }
}
/// Report generation schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSchedule {
    /// Daily reports
    pub daily: bool,
    /// Weekly reports
    pub weekly: bool,
    /// Monthly reports
    pub monthly: bool,
    /// Custom schedules
    pub custom_schedules: Vec<CustomSchedule>,
}
/// Alert event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    /// Alert ID
    pub id: String,
    /// Alert timestamp
    pub timestamp: DateTime<Utc>,
    /// Alert rule name
    pub rule_name: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Alert metadata
    pub metadata: HashMap<String, String>,
    /// Alert status
    pub status: AlertStatus,
}
/// Distribution channel
#[derive(Debug, Clone)]
pub struct DistributionChannel {
    /// Channel name
    pub name: String,
    /// Channel type
    pub channel_type: DistributionChannelType,
    /// Channel configuration
    pub config: HashMap<String, String>,
}
/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Rule condition
    pub condition: AlertCondition,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Notification channels to use
    pub notification_channels: Vec<String>,
    /// Rule enabled flag
    pub enabled: bool,
}
/// Chart type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    LineChart,
    BarChart,
    PieChart,
    AreaChart,
    ScatterPlot,
    Histogram,
    Heatmap,
}
/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Alert rules
    pub rules: Vec<AlertRule>,
    /// Notification channels
    pub notification_channels: Vec<NotificationChannel>,
    /// Alert throttling
    pub throttling: AlertThrottlingConfig,
}
/// Performance data store
#[derive(Debug)]
pub struct PerformanceDataStore {
    /// Execution results history
    pub execution_results: Mutex<VecDeque<TestExecutionResult>>,
    /// Performance metrics time series
    pub performance_metrics: Mutex<HashMap<String, VecDeque<TimestampedMetric>>>,
    /// Alert history
    pub alert_history: Mutex<VecDeque<AlertEvent>>,
    /// Regression events
    pub regression_events: Mutex<VecDeque<PerformanceRegression>>,
    /// Framework statistics history
    pub framework_stats_history: Mutex<VecDeque<TimestampedFrameworkStats>>,
}
impl PerformanceDataStore {
    fn new() -> Self {
        Self {
            execution_results: Mutex::new(VecDeque::new()),
            performance_metrics: Mutex::new(HashMap::new()),
            alert_history: Mutex::new(VecDeque::new()),
            regression_events: Mutex::new(VecDeque::new()),
            framework_stats_history: Mutex::new(VecDeque::new()),
        }
    }
    async fn cleanup_old_data(&self, retention_config: &DataRetentionConfig) {
        let now = Utc::now();
        {
            let mut execution_results = self.execution_results.lock();
            let _retention_cutoff = now
                - ChronoDuration::from_std(retention_config.execution_results_retention)
                    .unwrap_or_default();
            if execution_results.len() > 10000 {
                execution_results.truncate(5000);
            }
        }
        {
            let mut alert_history = self.alert_history.lock();
            let retention_cutoff = now
                - ChronoDuration::from_std(retention_config.alert_history_retention)
                    .unwrap_or_default();
            alert_history.retain(|alert| alert.timestamp > retention_cutoff);
        }
        {
            let mut stats_history = self.framework_stats_history.lock();
            if stats_history.len() > 10000 {
                stats_history.truncate(5000);
            }
        }
        debug!("Completed data cleanup");
    }
}
/// Alert summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertSummary {
    /// Total alerts in period
    pub total_alerts: usize,
    /// Alerts by severity
    pub alerts_by_severity: HashMap<AlertSeverity, usize>,
    /// Active alerts
    pub active_alerts: usize,
    /// Most frequent alert types
    pub top_alert_types: Vec<(String, usize)>,
}
/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Enable web dashboard
    pub enabled: bool,
    /// Dashboard host
    pub host: String,
    /// Dashboard port
    pub port: u16,
    /// Refresh interval
    pub refresh_interval: Duration,
    /// Dashboard theme
    pub theme: DashboardTheme,
    /// Chart configurations
    pub charts: Vec<ChartConfig>,
}
/// Report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Html,
    Pdf,
    Json,
    Csv,
    Markdown,
}
/// Data retention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRetentionConfig {
    /// Execution results retention
    pub execution_results_retention: Duration,
    /// Performance metrics retention
    pub performance_metrics_retention: Duration,
    /// Alert history retention
    pub alert_history_retention: Duration,
    /// Cleanup interval
    pub cleanup_interval: Duration,
}
/// Filesystem distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemDistributionConfig {
    /// Enable filesystem distribution
    pub enabled: bool,
    /// Output directory
    pub output_directory: String,
    /// File naming pattern
    pub filename_pattern: String,
}
/// Report template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportTemplate {
    /// Template name
    pub name: String,
    /// Template content
    pub content: String,
    /// Template variables
    pub variables: Vec<TemplateVariable>,
}
/// Alert status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertStatus {
    Active,
    Acknowledged,
    Resolved,
    Suppressed,
}
/// Test performance monitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitorConfig {
    /// Enable performance monitoring
    pub enabled: bool,
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Dashboard configuration
    pub dashboard: DashboardConfig,
    /// Alerting configuration
    pub alerting: AlertingConfig,
    /// Report generation configuration
    pub reporting: ReportingConfig,
    /// Data retention settings
    pub retention: DataRetentionConfig,
}
/// Alert manager
pub struct AlertManager {
    /// Alert rules
    rules: Arc<Mutex<Vec<AlertRule>>>,
    /// Notification channels
    channels: Arc<Mutex<Vec<NotificationChannel>>>,
    /// Alert state tracking
    _alert_state: Arc<Mutex<HashMap<String, AlertState>>>,
    /// Alert history
    alert_history: Arc<Mutex<VecDeque<AlertEvent>>>,
}
impl AlertManager {
    fn new(config: &AlertingConfig) -> Result<Self> {
        Ok(Self {
            rules: Arc::new(Mutex::new(config.rules.clone())),
            channels: Arc::new(Mutex::new(config.notification_channels.clone())),
            _alert_state: Arc::new(Mutex::new(HashMap::new())),
            alert_history: Arc::new(Mutex::new(VecDeque::new())),
        })
    }
    async fn evaluate_alerts(&self, data_store: &PerformanceDataStore) -> Result<()> {
        let rules = self.rules.lock().clone();
        for rule in rules.iter().filter(|r| r.enabled) {
            if let Some(alert_event) = self.evaluate_alert_rule(rule, data_store).await? {
                self.process_alert_event(alert_event).await?;
            }
        }
        Ok(())
    }
    async fn evaluate_alert_rule(
        &self,
        rule: &AlertRule,
        data_store: &PerformanceDataStore,
    ) -> Result<Option<AlertEvent>> {
        match &rule.condition {
            AlertCondition::TimeoutFailureRate {
                threshold,
                window: _,
            } => {
                let execution_results = data_store.execution_results.lock();
                let timeout_count = execution_results
                    .iter()
                    .filter(|r| matches!(r.outcome, TestOutcome::Timeout))
                    .count();
                let total_count = execution_results.len();
                if total_count > 0 {
                    let timeout_rate = timeout_count as f32 / total_count as f32;
                    if timeout_rate > *threshold {
                        return Ok(Some(AlertEvent {
                            id: uuid::Uuid::new_v4().to_string(),
                            timestamp: Utc::now(),
                            rule_name: rule.name.clone(),
                            severity: rule.severity.clone(),
                            message: format!(
                                "Timeout failure rate ({:.1}%) exceeds threshold ({:.1}%)",
                                timeout_rate * 100.0,
                                threshold * 100.0
                            ),
                            metadata: HashMap::new(),
                            status: AlertStatus::Active,
                        }));
                    }
                }
            },
            _ => {},
        }
        Ok(None)
    }
    async fn process_alert_event(&self, alert_event: AlertEvent) -> Result<()> {
        {
            let mut alert_history = self.alert_history.lock();
            alert_history.push_back(alert_event.clone());
        }
        self.send_alert_notifications(&alert_event).await?;
        info!(
            alert_id = % alert_event.id, rule_name = % alert_event.rule_name, severity =
            ? alert_event.severity, "Alert triggered"
        );
        Ok(())
    }
    async fn send_alert_notifications(&self, alert_event: &AlertEvent) -> Result<()> {
        let channels = self.channels.lock();
        for channel in channels.iter().filter(|c| c.enabled) {
            match channel.channel_type {
                NotificationChannelType::Console => {
                    println!(
                        "[ALERT] {} - {} ({}): {}",
                        alert_event.timestamp.format("%Y-%m-%d %H:%M:%S"),
                        alert_event.severity,
                        alert_event.rule_name,
                        alert_event.message
                    );
                },
                _ => {
                    warn!(
                        "Notification channel type {:?} not implemented",
                        channel.channel_type
                    );
                },
            }
        }
        Ok(())
    }
}
/// Data source for charts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    ExecutionTimes,
    TimeoutEvents,
    SuccessRates,
    OptimizationEffectiveness,
    ResourceUsage,
    RegressionDetection,
}
/// Test execution summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSummary {
    /// Total tests executed
    pub total_tests: usize,
    /// Success rate
    pub success_rate: f32,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Timeout rate
    pub timeout_rate: f32,
    /// Early termination rate
    pub early_termination_rate: f32,
    /// Execution time by category
    pub execution_times_by_category: HashMap<TestCategory, Duration>,
}
/// Alert throttling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThrottlingConfig {
    /// Minimum time between identical alerts
    pub min_interval: Duration,
    /// Maximum alerts per hour
    pub max_alerts_per_hour: usize,
    /// Escalation settings
    pub escalation: EscalationConfig,
}
/// Escalation level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLevel {
    /// Time to escalate
    pub escalate_after: Duration,
    /// Target notification channels
    pub channels: Vec<String>,
    /// Escalation message template
    pub message_template: String,
}
/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Enable automatic report generation
    pub enabled: bool,
    /// Report generation schedule
    pub schedule: ReportSchedule,
    /// Report formats
    pub formats: Vec<ReportFormat>,
    /// Report distribution
    pub distribution: DistributionConfig,
    /// Report templates
    pub templates: Vec<ReportTemplate>,
}
/// Chart configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartConfig {
    /// Chart title
    pub title: String,
    /// Chart type
    pub chart_type: ChartType,
    /// Data source
    pub data_source: DataSource,
    /// Time window
    pub time_window: Duration,
    /// Update interval
    pub update_interval: Duration,
}
/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}
/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Unknown,
}
/// S3 distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3DistributionConfig {
    /// S3 bucket
    pub bucket: String,
    /// S3 prefix
    pub prefix: String,
    /// AWS region
    pub region: String,
}
/// Metric value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Float(f64),
    Integer(i64),
    Boolean(bool),
    String(String),
    Duration(Duration),
}
/// Dashboard theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardTheme {
    Light,
    Dark,
    Auto,
}
/// Distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionConfig {
    /// Email distribution
    pub email: EmailDistributionConfig,
    /// File system distribution
    pub filesystem: FilesystemDistributionConfig,
    /// S3 distribution
    pub s3: Option<S3DistributionConfig>,
}
/// Template variable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateVariable {
    /// Variable name
    pub name: String,
    /// Variable type
    pub variable_type: VariableType,
    /// Default value
    pub default_value: Option<String>,
}
/// Template variable types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableType {
    String,
    Number,
    Boolean,
    Date,
    Duration,
    Chart,
    Table,
}
/// Alert escalation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationConfig {
    /// Enable escalation
    pub enabled: bool,
    /// Escalation levels
    pub levels: Vec<EscalationLevel>,
}
/// Generated report
#[derive(Debug, Clone)]
pub struct GeneratedReport {
    /// Report ID
    pub id: String,
    /// Generation timestamp
    pub timestamp: DateTime<Utc>,
    /// Report template used
    pub template: String,
    /// Report format
    pub format: ReportFormat,
    /// Report content
    pub content: String,
    /// Report metadata
    pub metadata: HashMap<String, String>,
}
/// Notification channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannelConfig {
    /// Channel-specific settings
    pub settings: HashMap<String, String>,
}
/// Distribution channel types
#[derive(Debug, Clone)]
pub enum DistributionChannelType {
    Email,
    Filesystem,
    S3,
    Http,
}
/// Test performance monitor
pub struct TestPerformanceMonitor {
    /// Monitor configuration
    config: Arc<Mutex<PerformanceMonitorConfig>>,
    /// Framework reference
    framework: Arc<TestTimeoutFramework>,
    /// Performance data store
    data_store: Arc<PerformanceDataStore>,
    /// Alert manager
    alert_manager: Arc<AlertManager>,
    /// Report generator
    report_generator: Arc<ReportGenerator>,
    /// Dashboard server
    dashboard_server: Option<Arc<DashboardServer>>,
    /// Background tasks
    background_tasks: Vec<JoinHandle<()>>,
    /// Shutdown signal
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}
impl TestPerformanceMonitor {
    /// Create a new test performance monitor
    pub fn new(
        config: PerformanceMonitorConfig,
        framework: Arc<TestTimeoutFramework>,
    ) -> Result<Self> {
        let data_store = Arc::new(PerformanceDataStore::new());
        let alert_manager = Arc::new(AlertManager::new(&config.alerting)?);
        let report_generator = Arc::new(ReportGenerator::new(&config.reporting)?);
        let dashboard_server = if config.dashboard.enabled {
            Some(Arc::new(DashboardServer::new(
                config.dashboard.clone(),
                data_store.clone(),
            )?))
        } else {
            None
        };
        Ok(Self {
            config: Arc::new(Mutex::new(config)),
            framework,
            data_store,
            alert_manager,
            report_generator,
            dashboard_server,
            background_tasks: Vec::new(),
            shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }
    /// Start the performance monitor
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting test performance monitor");
        let monitoring_task = self.start_monitoring_task().await?;
        self.background_tasks.push(monitoring_task);
        let alert_task = self.start_alert_evaluation_task().await?;
        self.background_tasks.push(alert_task);
        let report_task = self.start_report_generation_task().await?;
        self.background_tasks.push(report_task);
        let cleanup_task = self.start_cleanup_task().await?;
        self.background_tasks.push(cleanup_task);
        if let Some(dashboard) = &mut self.dashboard_server {
            dashboard.start().await?;
        }
        info!("Test performance monitor started successfully");
        Ok(())
    }
    /// Stop the performance monitor
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping test performance monitor");
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(dashboard) = &mut self.dashboard_server {
            dashboard.stop().await?;
        }
        for task in self.background_tasks.drain(..) {
            let _ = task.await;
        }
        info!("Test performance monitor stopped");
        Ok(())
    }
    /// Record test execution result
    pub async fn record_execution_result(&self, result: &TestExecutionResult) -> Result<()> {
        let mut execution_results = self.data_store.execution_results.lock();
        execution_results.push_back(result.clone());
        let config = self.config.lock();
        let retention_limit =
            Utc::now() - ChronoDuration::from_std(config.retention.execution_results_retention)?;
        execution_results.retain(|_r| Utc::now() > retention_limit);
        debug!(
            test_name = % result.context.test_name, outcome = ? result.outcome,
            "Recorded test execution result"
        );
        Ok(())
    }
    /// Get performance summary for a time window
    pub async fn get_performance_summary(
        &self,
        time_window: Duration,
    ) -> Result<PerformanceSummary> {
        let execution_results = self.data_store.execution_results.lock();
        let alert_history = self.data_store.alert_history.lock();
        let window_start = Utc::now() - ChronoDuration::from_std(time_window)?;
        let relevant_results: Vec<_> =
            execution_results.iter().filter(|_r| Utc::now() > window_start).collect();
        let total_tests = relevant_results.len();
        let successful_tests = relevant_results
            .iter()
            .filter(|r| matches!(r.outcome, TestOutcome::Success))
            .count();
        let timeout_tests = relevant_results
            .iter()
            .filter(|r| matches!(r.outcome, TestOutcome::Timeout))
            .count();
        let early_termination_tests = relevant_results
            .iter()
            .filter(|r| matches!(r.outcome, TestOutcome::EarlyTermination(_)))
            .count();
        let success_rate =
            if total_tests > 0 { successful_tests as f32 / total_tests as f32 } else { 0.0 };
        let timeout_rate =
            if total_tests > 0 { timeout_tests as f32 / total_tests as f32 } else { 0.0 };
        let early_termination_rate = if total_tests > 0 {
            early_termination_tests as f32 / total_tests as f32
        } else {
            0.0
        };
        let avg_execution_time = if total_tests > 0 {
            let total_time: Duration = relevant_results.iter().map(|r| r.execution_time).sum();
            total_time / total_tests as u32
        } else {
            Duration::ZERO
        };
        let mut execution_times_by_category = HashMap::new();
        for category in [
            TestCategory::Unit,
            TestCategory::Integration,
            TestCategory::EndToEnd,
            TestCategory::Stress,
            TestCategory::Property,
            TestCategory::Chaos,
            TestCategory::LongRunning,
        ] {
            let category_results: Vec<_> =
                relevant_results.iter().filter(|r| r.context.category == category).collect();
            if !category_results.is_empty() {
                let avg_time: Duration =
                    category_results.iter().map(|r| r.execution_time).sum::<Duration>()
                        / category_results.len() as u32;
                execution_times_by_category.insert(category, avg_time);
            }
        }
        let execution_summary = ExecutionSummary {
            total_tests,
            success_rate,
            avg_execution_time,
            timeout_rate,
            early_termination_rate,
            execution_times_by_category,
        };
        let total_optimizations =
            relevant_results.iter().map(|r| r.optimizations_applied.len()).sum();
        let optimization_summary = OptimizationSummary {
            total_optimizations,
            time_saved: Duration::ZERO,
            optimization_success_rate: 0.8,
            top_optimizations: Vec::new(),
        };
        let trends = TrendsSummary {
            execution_time_trend: TrendDirection::Stable,
            success_rate_trend: TrendDirection::Stable,
            optimization_effectiveness_trend: TrendDirection::Improving,
            resource_usage_trend: TrendDirection::Stable,
        };
        let relevant_alerts: Vec<_> =
            alert_history.iter().filter(|a| a.timestamp > window_start).collect();
        let total_alerts = relevant_alerts.len();
        let active_alerts = relevant_alerts
            .iter()
            .filter(|a| matches!(a.status, AlertStatus::Active))
            .count();
        let mut alerts_by_severity = HashMap::new();
        for alert in &relevant_alerts {
            *alerts_by_severity.entry(alert.severity.clone()).or_insert(0) += 1;
        }
        let alert_summary = AlertSummary {
            total_alerts,
            alerts_by_severity,
            active_alerts,
            top_alert_types: Vec::new(),
        };
        Ok(PerformanceSummary {
            timestamp: Utc::now(),
            time_window,
            execution_summary,
            optimization_summary,
            trends,
            alert_summary,
        })
    }
    /// Start monitoring background task
    async fn start_monitoring_task(&self) -> Result<JoinHandle<()>> {
        let framework = self.framework.clone();
        let data_store = self.data_store.clone();
        let shutdown = self.shutdown.clone();
        let config = self.config.lock().clone();
        let handle = tokio::spawn(async move {
            let mut interval = interval(config.monitoring_interval);
            while !shutdown.load(Ordering::SeqCst) {
                interval.tick().await;
                let stats = framework.get_statistics().await;
                let timestamped_stats = TimestampedFrameworkStats {
                    timestamp: Utc::now(),
                    stats,
                };
                let mut stats_history = data_store.framework_stats_history.lock();
                stats_history.push_back(timestamped_stats);
                if stats_history.len() > 1000 {
                    stats_history.pop_front();
                }
            }
        });
        Ok(handle)
    }
    /// Start alert evaluation background task
    async fn start_alert_evaluation_task(&self) -> Result<JoinHandle<()>> {
        let alert_manager = self.alert_manager.clone();
        let data_store = self.data_store.clone();
        let shutdown = self.shutdown.clone();
        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            while !shutdown.load(Ordering::SeqCst) {
                interval.tick().await;
                if let Err(e) = alert_manager.evaluate_alerts(&data_store).await {
                    error!("Error evaluating alerts: {}", e);
                }
            }
        });
        Ok(handle)
    }
    /// Start report generation background task
    async fn start_report_generation_task(&self) -> Result<JoinHandle<()>> {
        let report_generator = self.report_generator.clone();
        let data_store = self.data_store.clone();
        let shutdown = self.shutdown.clone();
        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(3600));
            while !shutdown.load(Ordering::SeqCst) {
                interval.tick().await;
                if let Err(e) = report_generator.check_and_generate_reports(&data_store).await {
                    error!("Error generating reports: {}", e);
                }
            }
        });
        Ok(handle)
    }
    /// Start cleanup background task
    async fn start_cleanup_task(&self) -> Result<JoinHandle<()>> {
        let data_store = self.data_store.clone();
        let shutdown = self.shutdown.clone();
        let config = self.config.lock().clone();
        let handle = tokio::spawn(async move {
            let mut interval = interval(config.retention.cleanup_interval);
            while !shutdown.load(Ordering::SeqCst) {
                interval.tick().await;
                data_store.cleanup_old_data(&config.retention).await;
            }
        });
        Ok(handle)
    }
}
/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Summary timestamp
    pub timestamp: DateTime<Utc>,
    /// Time window covered
    pub time_window: Duration,
    /// Test execution summary
    pub execution_summary: ExecutionSummary,
    /// Timeout optimization summary
    pub optimization_summary: OptimizationSummary,
    /// Performance trends
    pub trends: TrendsSummary,
    /// Alert summary
    pub alert_summary: AlertSummary,
}
/// Alert condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    /// High timeout failure rate
    TimeoutFailureRate { threshold: f32, window: Duration },
    /// Performance regression detected
    PerformanceRegression {
        threshold: f32,
        category: Option<TestCategory>,
    },
    /// Optimization effectiveness below threshold
    OptimizationEffectiveness { threshold: f32, window: Duration },
    /// High resource usage
    ResourceUsage {
        cpu_threshold: Option<f32>,
        memory_threshold: Option<u64>,
    },
    /// Framework error rate
    FrameworkErrorRate { threshold: f32, window: Duration },
}
/// Timestamped framework statistics
#[derive(Debug, Clone)]
pub struct TimestampedFrameworkStats {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Framework statistics
    pub stats: FrameworkStats,
}
/// Notification channel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannelType {
    Email,
    Slack,
    Discord,
    Webhook,
    Console,
}
/// Custom report schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomSchedule {
    /// Schedule name
    pub name: String,
    /// Cron expression
    pub cron_expression: String,
    /// Report template to use
    pub template: String,
}
/// Timestamped metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedMetric {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: MetricValue,
    /// Metric tags
    pub tags: HashMap<String, String>,
}
