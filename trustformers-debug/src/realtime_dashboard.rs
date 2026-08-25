//! Real-Time Debugging Dashboard
//!
//! This module provides a modern, real-time debugging dashboard with WebSocket support,
//! interactive visualizations, and live data streaming for comprehensive neural network monitoring.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::System;
use tokio::sync::broadcast;
use tokio::time::interval;
use tokio_stream::wrappers::BroadcastStream;
use uuid::Uuid;

/// Number of most-recent samples of a category examined for a *sustained*
/// threshold breach (as opposed to a single noisy spike).
const SUSTAINED_BREACH_WINDOW: usize = 5;

/// Configuration for the real-time dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Port for WebSocket server
    pub websocket_port: u16,
    /// Update frequency in milliseconds
    pub update_frequency_ms: u64,
    /// Maximum number of data points to keep in memory
    pub max_data_points: usize,
    /// Enable GPU monitoring
    pub enable_gpu_monitoring: bool,
    /// Enable memory profiling
    pub enable_memory_profiling: bool,
    /// Enable network traffic monitoring
    pub enable_network_monitoring: bool,
    /// Enable performance alerts
    pub enable_performance_alerts: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            websocket_port: 8080,
            update_frequency_ms: 100,
            max_data_points: 1000,
            enable_gpu_monitoring: true,
            enable_memory_profiling: true,
            enable_network_monitoring: false,
            enable_performance_alerts: true,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Memory usage threshold (percentage)
    pub memory_threshold: f64,
    /// GPU utilization threshold (percentage)
    pub gpu_utilization_threshold: f64,
    /// Temperature threshold (Celsius)
    pub temperature_threshold: f64,
    /// Loss spike threshold
    pub loss_spike_threshold: f64,
    /// Gradient norm threshold
    pub gradient_norm_threshold: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            memory_threshold: 90.0,
            gpu_utilization_threshold: 95.0,
            temperature_threshold: 80.0,
            loss_spike_threshold: 2.0,
            gradient_norm_threshold: 10.0,
        }
    }
}

/// Real-time metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDataPoint {
    pub timestamp: u64,
    pub value: f64,
    pub label: String,
    pub category: MetricCategory,
}

/// Categories of metrics for organization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MetricCategory {
    Training,
    Memory,
    GPU,
    Network,
    Performance,
    Custom(String),
}

/// Dashboard alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardAlert {
    pub id: String,
    pub timestamp: u64,
    pub severity: AlertSeverity,
    pub category: MetricCategory,
    pub title: String,
    pub message: String,
    pub value: Option<f64>,
    pub threshold: Option<f64>,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    MetricUpdate {
        data: Vec<MetricDataPoint>,
    },
    Alert {
        alert: DashboardAlert,
    },
    ConfigUpdate {
        config: DashboardConfig,
    },
    SessionInfo {
        session_id: String,
        uptime: u64,
    },
    HistoricalData {
        category: MetricCategory,
        data: Vec<MetricDataPoint>,
    },
    SystemStats {
        stats: SystemStats,
    },
    #[serde(untagged)]
    Generic {
        message_type: String,
        data: serde_json::Value,
        timestamp: u64,
        session_id: String,
    },
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    pub timestamp: u64,
    pub value: f64,
    pub expected_range: (f64, f64),
    pub anomaly_type: AnomalyType,
    pub confidence_score: f64,
    pub category: MetricCategory,
    pub description: String,
}

/// Types of anomalies that can be detected
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AnomalyType {
    Spike,
    Drop,
    GradualIncrease,
    GradualDecrease,
    Outlier,
}

/// Advanced dashboard visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardVisualizationData {
    pub heatmap_data: HashMap<MetricCategory, HeatmapData>,
    pub time_series_data: HashMap<MetricCategory, Vec<TimeSeriesPoint>>,
    pub correlation_matrix: Vec<Vec<f64>>,
    pub performance_distribution: HashMap<MetricCategory, HistogramData>,
    pub generated_at: u64,
    pub session_id: String,
}

/// Heatmap data for metric visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapData {
    pub intensity: f64,
    pub normalized_intensity: f64,
    pub data_points: usize,
    pub timestamp: u64,
}

/// Time series data point for trend visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    pub timestamp: u64,
    pub value: f64,
    pub label: String,
}

/// Histogram data for performance distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramData {
    pub bins: Vec<HistogramBin>,
    pub max_frequency: usize,
}

/// Individual histogram bin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBin {
    pub range_start: f64,
    pub range_end: f64,
    pub frequency: usize,
}

/// Performance prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePrediction {
    pub category: MetricCategory,
    pub predicted_value: f64,
    pub confidence_interval: (f64, f64),
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub prediction_horizon_hours: u64,
    pub model_accuracy: f64,
    pub generated_at: u64,
    pub recommendations: Vec<String>,
}

/// Trend direction for predictions
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Dashboard theme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTheme {
    pub name: String,
    pub primary_color: String,
    pub secondary_color: String,
    pub background_color: String,
    pub text_color: String,
    pub accent_color: String,
    pub chart_colors: Vec<String>,
    pub dark_mode: bool,
    pub font_family: String,
    pub border_radius: u8,
}

impl Default for DashboardTheme {
    fn default() -> Self {
        Self {
            name: "Default".to_string(),
            primary_color: "#3b82f6".to_string(),
            secondary_color: "#64748b".to_string(),
            background_color: "#ffffff".to_string(),
            text_color: "#1f2937".to_string(),
            accent_color: "#10b981".to_string(),
            chart_colors: vec![
                "#3b82f6".to_string(),
                "#ef4444".to_string(),
                "#10b981".to_string(),
                "#f59e0b".to_string(),
                "#8b5cf6".to_string(),
            ],
            dark_mode: false,
            font_family: "Inter, sans-serif".to_string(),
            border_radius: 8,
        }
    }
}

/// Export format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    JSON,
    CSV,
    MessagePack,
}

/// System statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub uptime: u64,
    pub total_alerts: usize,
    pub active_connections: usize,
    pub data_points_collected: usize,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

/// Real-time dashboard server
#[derive(Debug)]
pub struct RealtimeDashboard {
    config: Arc<Mutex<DashboardConfig>>,
    session_id: String,
    start_time: Instant,
    metric_data: Arc<Mutex<HashMap<MetricCategory, VecDeque<MetricDataPoint>>>>,
    alert_history: Arc<Mutex<VecDeque<DashboardAlert>>>,
    websocket_sender: broadcast::Sender<WebSocketMessage>,
    active_connections: Arc<Mutex<usize>>,
    total_data_points: Arc<Mutex<usize>>,
    is_running: Arc<Mutex<bool>>,
    /// Long-lived host telemetry handle (real `sysinfo` readings). Kept
    /// long-lived rather than recreated per call because CPU-usage
    /// measurement is delta-based: `sysinfo` needs two refreshes spaced
    /// apart in real time to report a meaningful percentage.
    system_info: Arc<Mutex<System>>,
}

impl RealtimeDashboard {
    /// Create new real-time dashboard
    pub fn new(config: DashboardConfig) -> Self {
        let (websocket_sender, _) = broadcast::channel(1000);

        Self {
            config: Arc::new(Mutex::new(config)),
            session_id: Uuid::new_v4().to_string(),
            start_time: Instant::now(),
            metric_data: Arc::new(Mutex::new(HashMap::new())),
            alert_history: Arc::new(Mutex::new(VecDeque::new())),
            websocket_sender,
            active_connections: Arc::new(Mutex::new(0)),
            total_data_points: Arc::new(Mutex::new(0)),
            is_running: Arc::new(Mutex::new(false)),
            system_info: Arc::new(Mutex::new(System::new_all())),
        }
    }

    /// Start the dashboard server
    pub async fn start(&self) -> Result<()> {
        {
            let mut running = self
                .is_running
                .lock()
                .map_err(|_| anyhow::anyhow!("Failed to acquire running state lock"))?;
            if *running {
                return Ok(());
            }
            *running = true;
        }

        // Start periodic data collection
        self.start_data_collection().await?;

        // Start periodic system stats updates
        self.start_system_stats_updates().await?;

        // Start alert monitoring
        self.start_alert_monitoring().await?;

        Ok(())
    }

    /// Stop the dashboard server
    pub fn stop(&self) {
        if let Ok(mut running) = self.is_running.lock() {
            *running = false;
        }
    }

    /// Add a metric data point
    pub fn add_metric(&self, category: MetricCategory, label: String, value: f64) -> Result<()> {
        self.add_metrics(vec![(category, label, value)])
    }

    /// Add multiple metrics at once. Stores each point (bounded by
    /// `max_data_points`), broadcasts the batch, and evaluates each real
    /// value against the configured alert thresholds -- the same path used
    /// by the periodic host-telemetry collector, so manually-supplied and
    /// automatically-collected metrics raise alerts identically.
    pub fn add_metrics(&self, metrics: Vec<(MetricCategory, String, f64)>) -> Result<()> {
        let thresholds = self
            .config
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire config lock"))?
            .alert_thresholds
            .clone();
        let max_points = self
            .config
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire config lock"))?
            .max_data_points;

        ingest_metrics(
            &self.metric_data,
            &self.total_data_points,
            max_points,
            &self.alert_history,
            &thresholds,
            &self.websocket_sender,
            metrics,
        )
    }

    /// Create an alert
    pub fn create_alert(
        &self,
        severity: AlertSeverity,
        category: MetricCategory,
        title: String,
        message: String,
        value: Option<f64>,
        threshold: Option<f64>,
    ) -> Result<()> {
        let alert = DashboardAlert {
            id: Uuid::new_v4().to_string(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64,
            severity,
            category,
            title,
            message,
            value,
            threshold,
        };

        // Add to alert history
        {
            let mut history =
                self.alert_history.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            history.push_back(alert.clone());

            // Keep only last 100 alerts
            while history.len() > 100 {
                history.pop_front();
            }
        }

        // Broadcast alert
        let message = WebSocketMessage::Alert { alert };
        let _ = self.websocket_sender.send(message);

        Ok(())
    }

    /// Get historical data for a category
    pub fn get_historical_data(&self, category: &MetricCategory) -> Vec<MetricDataPoint> {
        let data = self.metric_data.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        data.get(category)
            .map(|deque| deque.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get current system stats
    pub fn get_system_stats(&self) -> SystemStats {
        let uptime = self.start_time.elapsed().as_secs();
        let total_alerts =
            self.alert_history.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).len();
        let active_connections =
            *self.active_connections.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let data_points_collected =
            *self.total_data_points.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        // Simple memory and CPU usage estimation
        let memory_usage_mb = self.estimate_memory_usage();
        let cpu_usage_percent = self.estimate_cpu_usage();

        SystemStats {
            uptime,
            total_alerts,
            active_connections,
            data_points_collected,
            memory_usage_mb,
            cpu_usage_percent,
        }
    }

    /// Subscribe to WebSocket messages
    pub fn subscribe(&self) -> BroadcastStream<WebSocketMessage> {
        // Increment connection counter
        {
            let mut connections =
                self.active_connections.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            *connections += 1;
        }

        BroadcastStream::new(self.websocket_sender.subscribe())
    }

    /// Update dashboard configuration
    pub fn update_config(&self, new_config: DashboardConfig) -> Result<()> {
        {
            let mut config = self.config.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            *config = new_config.clone();
        }

        // Broadcast configuration update
        let message = WebSocketMessage::ConfigUpdate { config: new_config };
        let _ = self.websocket_sender.send(message);

        Ok(())
    }

    /// Get current configuration
    pub fn get_config(&self) -> DashboardConfig {
        self.config.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
    }

    /// Start periodic data collection
    async fn start_data_collection(&self) -> Result<()> {
        let config = self.config.clone();
        let metric_data = self.metric_data.clone();
        let total_data_points = self.total_data_points.clone();
        let alert_history = self.alert_history.clone();
        let system_info = self.system_info.clone();
        let websocket_sender = self.websocket_sender.clone();
        let is_running = self.is_running.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(
                config
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .update_frequency_ms,
            ));

            while *is_running.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) {
                interval.tick().await;

                // Collect real host telemetry, then store + alert on it via the
                // exact same path `add_metric`/`add_metrics` use -- real values
                // that never reach `metric_data`/the alert thresholds would be
                // pointless to have collected in the first place.
                let (metrics, max_points, thresholds) = {
                    let cfg = config.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    let metrics = collect_system_metrics(&cfg, &system_info);
                    (metrics, cfg.max_data_points, cfg.alert_thresholds.clone())
                };

                if !metrics.is_empty() {
                    let _ = ingest_metrics(
                        &metric_data,
                        &total_data_points,
                        max_points,
                        &alert_history,
                        &thresholds,
                        &websocket_sender,
                        metrics,
                    );
                }
            }
        });

        Ok(())
    }

    /// Start system stats updates
    async fn start_system_stats_updates(&self) -> Result<()> {
        let websocket_sender = self.websocket_sender.clone();
        let start_time = self.start_time;
        let alert_history = self.alert_history.clone();
        let active_connections = self.active_connections.clone();
        let total_data_points = self.total_data_points.clone();
        let metric_data = self.metric_data.clone();
        let system_info = self.system_info.clone();
        let is_running = self.is_running.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5)); // Update every 5 seconds

            while *is_running.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) {
                interval.tick().await;

                let stats = SystemStats {
                    uptime: start_time.elapsed().as_secs(),
                    total_alerts: alert_history
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .len(),
                    active_connections: *active_connections
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                    data_points_collected: *total_data_points
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                    memory_usage_mb: dashboard_footprint_mb(&metric_data),
                    cpu_usage_percent: estimate_cpu_usage(&system_info),
                };

                let message = WebSocketMessage::SystemStats { stats };
                let _ = websocket_sender.send(message);
            }
        });

        Ok(())
    }

    /// Start alert monitoring
    async fn start_alert_monitoring(&self) -> Result<()> {
        let metric_data = self.metric_data.clone();
        let config = self.config.clone();
        let alert_history = self.alert_history.clone();
        let websocket_sender = self.websocket_sender.clone();
        let is_running = self.is_running.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));

            while *is_running.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) {
                interval.tick().await;

                // Monitor the real, now-populated history for sustained
                // threshold breaches and create alerts.
                let thresholds = {
                    let cfg = config.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    cfg.alert_thresholds.clone()
                };
                check_threshold_breaches(
                    &metric_data,
                    &thresholds,
                    &alert_history,
                    &websocket_sender,
                );
            }
        });

        Ok(())
    }

    /// Estimate memory usage of dashboard
    fn estimate_memory_usage(&self) -> f64 {
        dashboard_footprint_mb(&self.metric_data)
    }

    /// Real, host-wide CPU usage percentage (see [`estimate_cpu_usage`] free function).
    fn estimate_cpu_usage(&self) -> f64 {
        estimate_cpu_usage(&self.system_info)
    }

    /// AI-powered anomaly detection for metric patterns
    pub async fn detect_metric_anomalies(
        &self,
        category: &MetricCategory,
    ) -> Result<Vec<AnomalyDetection>> {
        let data = self.get_historical_data(category);
        let mut anomalies = Vec::new();

        if data.len() < 10 {
            return Ok(anomalies); // Need sufficient data for anomaly detection
        }

        // Calculate statistical thresholds
        let values: Vec<f64> = data.iter().map(|d| d.value).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        // Z-score based anomaly detection
        let z_threshold = 2.0; // 2 standard deviations
        for point in data.iter() {
            let z_score = (point.value - mean).abs() / std_dev;
            if z_score > z_threshold {
                let anomaly_type =
                    if point.value > mean { AnomalyType::Spike } else { AnomalyType::Drop };

                anomalies.push(AnomalyDetection {
                    timestamp: point.timestamp,
                    value: point.value,
                    expected_range: (mean - std_dev, mean + std_dev),
                    anomaly_type,
                    confidence_score: (z_score - z_threshold) / z_threshold,
                    category: category.clone(),
                    description: format!(
                        "Detected {} in {} metrics: value {} (Z-score: {:.2})",
                        match anomaly_type {
                            AnomalyType::Spike => "spike",
                            AnomalyType::Drop => "drop",
                            _ => "anomaly",
                        },
                        match category {
                            MetricCategory::Training => "training",
                            MetricCategory::Memory => "memory",
                            MetricCategory::GPU => "GPU",
                            MetricCategory::Network => "network",
                            MetricCategory::Performance => "performance",
                            MetricCategory::Custom(name) => name,
                        },
                        point.value,
                        z_score
                    ),
                });
            }
        }

        // Advanced pattern detection - look for gradual trends
        if data.len() >= 20 {
            let recent_window = &data[data.len() - 10..];
            let earlier_window = &data[data.len() - 20..data.len() - 10];

            let recent_avg =
                recent_window.iter().map(|d| d.value).sum::<f64>() / recent_window.len() as f64;
            let earlier_avg =
                earlier_window.iter().map(|d| d.value).sum::<f64>() / earlier_window.len() as f64;

            let trend_change = (recent_avg - earlier_avg) / earlier_avg;

            if trend_change.abs() > 0.3 {
                // 30% change
                if let Some(last_point) = recent_window.last() {
                    anomalies.push(AnomalyDetection {
                        timestamp: last_point.timestamp,
                        value: recent_avg,
                        expected_range: (earlier_avg * 0.9, earlier_avg * 1.1),
                        anomaly_type: if trend_change > 0.0 {
                            AnomalyType::GradualIncrease
                        } else {
                            AnomalyType::GradualDecrease
                        },
                        confidence_score: trend_change.abs(),
                        category: category.clone(),
                        description: format!(
                            "Detected gradual {} trend: {:.1}% change over recent measurements",
                            if trend_change > 0.0 { "increase" } else { "decrease" },
                            trend_change.abs() * 100.0
                        ),
                    });
                }
            }
        }

        Ok(anomalies)
    }

    /// Generate advanced visualization data for modern dashboard components
    pub fn generate_advanced_visualizations(&self) -> Result<DashboardVisualizationData> {
        let mut heatmap_data = HashMap::new();
        let mut time_series_data = HashMap::new();
        let mut correlation_matrix = Vec::new();
        let mut performance_distribution = HashMap::new();

        // Generate heatmap data for different metric categories
        for (category, data) in
            self.metric_data.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).iter()
        {
            if data.len() >= 10 {
                let recent_data: Vec<f64> = data.iter().rev().take(10).map(|d| d.value).collect();
                let avg_value = recent_data.iter().sum::<f64>() / recent_data.len() as f64;

                heatmap_data.insert(
                    category.clone(),
                    HeatmapData {
                        intensity: avg_value,
                        normalized_intensity: (avg_value / (avg_value + 1.0)).min(1.0), // Normalize to 0-1
                        data_points: recent_data.len(),
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    },
                );

                // Time series data for trend visualization
                let time_series: Vec<TimeSeriesPoint> = data
                    .iter()
                    .map(|d| TimeSeriesPoint {
                        timestamp: d.timestamp,
                        value: d.value,
                        label: d.label.clone(),
                    })
                    .collect();

                time_series_data.insert(category.clone(), time_series);

                // Performance distribution data
                let values: Vec<f64> = data.iter().map(|d| d.value).collect();
                let histogram = self.create_histogram(&values, 10);
                performance_distribution.insert(category.clone(), histogram);
            }
        }

        // Generate correlation matrix for different metrics
        let categories: Vec<&MetricCategory> = heatmap_data.keys().collect();
        for (i, cat1) in categories.iter().enumerate() {
            let mut row = Vec::new();
            for (j, cat2) in categories.iter().enumerate() {
                if i == j {
                    row.push(1.0); // Perfect correlation with itself
                } else {
                    // Real Pearson correlation over the aligned recent
                    // samples; see `calculate_correlation_coefficient` for the
                    // pairing rule it assumes.
                    let corr = self.calculate_correlation_coefficient(cat1, cat2);
                    row.push(corr);
                }
            }
            correlation_matrix.push(row);
        }

        Ok(DashboardVisualizationData {
            heatmap_data,
            time_series_data,
            correlation_matrix,
            performance_distribution,
            generated_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            session_id: self.session_id.clone(),
        })
    }

    /// AI-powered performance prediction based on historical trends
    pub async fn predict_performance_trends(
        &self,
        category: &MetricCategory,
        hours_ahead: u64,
    ) -> Result<PerformancePrediction> {
        let data = self.get_historical_data(category);

        if data.len() < 20 {
            return Err(anyhow::anyhow!(
                "Insufficient data for prediction (need at least 20 points)"
            ));
        }

        let values: Vec<f64> = data.iter().map(|d| d.value).collect();
        let timestamps: Vec<u64> = data.iter().map(|d| d.timestamp).collect();

        // Simple linear regression for trend prediction
        let n = values.len() as f64;
        let sum_x = timestamps.iter().sum::<u64>() as f64;
        let sum_y = values.iter().sum::<f64>();
        let sum_xy = timestamps.iter().zip(&values).map(|(x, y)| *x as f64 * y).sum::<f64>();
        let sum_x2 = timestamps.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
        let intercept = (sum_y - slope * sum_x) / n;

        // Generate predictions
        let current_time =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        let prediction_time = current_time + (hours_ahead * 3600);
        let predicted_value = slope * prediction_time as f64 + intercept;

        // 95% interval around the fitted line from the standard error of the
        // MEAN (`sigma / sqrt(n)`) times the normal 1.96 quantile.
        //
        // Explicitly NOT a prediction interval: a real one would add the
        // residual variance and the leverage term for a point `hours_ahead`
        // outside the observed range, and would therefore be wider -- see
        // `PerformancePrediction::confidence_interval`.
        let mean = sum_y / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std_error = (variance / n).sqrt();
        let confidence_interval = std_error * 1.96;

        // Analyze trend direction and strength
        let trend_strength = slope.abs() / mean.abs();
        let trend_direction = if slope > 0.01 {
            TrendDirection::Increasing
        } else if slope < -0.01 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        Ok(PerformancePrediction {
            category: category.clone(),
            predicted_value,
            confidence_interval: (
                predicted_value - confidence_interval,
                predicted_value + confidence_interval,
            ),
            trend_direction,
            trend_strength,
            prediction_horizon_hours: hours_ahead,
            model_accuracy: 1.0 - (std_error / mean.abs()).min(1.0), // Simplified accuracy
            generated_at: current_time,
            recommendations: self.generate_performance_recommendations(
                &trend_direction,
                trend_strength,
                predicted_value,
            ),
        })
    }

    /// Advanced dashboard theme and customization support
    pub fn apply_dashboard_theme(&self, theme: DashboardTheme) -> Result<()> {
        // This would typically update UI styling, but we'll store theme preferences
        let theme_message = WebSocketMessage::Generic {
            message_type: "theme_update".to_string(),
            data: serde_json::to_value(&theme)?,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
            session_id: self.session_id.clone(),
        };

        if self.websocket_sender.send(theme_message).is_err() {
            // No active subscribers, but that's okay
        }

        Ok(())
    }

    /// Export dashboard data in various formats
    pub async fn export_dashboard_data(
        &self,
        format: ExportFormat,
        time_range: Option<(u64, u64)>,
    ) -> Result<Vec<u8>> {
        let data = if let Some((start, end)) = time_range {
            self.get_filtered_data(start, end)
        } else {
            self.get_all_data()
        };

        match format {
            ExportFormat::JSON => {
                let json_data = serde_json::to_string_pretty(&data)?;
                Ok(json_data.into_bytes())
            },
            ExportFormat::CSV => {
                let mut csv_data = String::from("timestamp,category,label,value\n");
                for (category, points) in data {
                    for point in points {
                        csv_data.push_str(&format!(
                            "{},{:?},{},{}\n",
                            point.timestamp, category, point.label, point.value
                        ));
                    }
                }
                Ok(csv_data.into_bytes())
            },
            ExportFormat::MessagePack => {
                // Would use rmp_serde for MessagePack serialization
                // For now, return JSON as fallback
                let json_data = serde_json::to_string(&data)?;
                Ok(json_data.into_bytes())
            },
        }
    }

    // Helper methods for advanced features

    fn create_histogram(&self, values: &[f64], bins: usize) -> HistogramData {
        if values.is_empty() {
            return HistogramData {
                bins: Vec::new(),
                max_frequency: 0,
            };
        }

        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let bin_width = (max_val - min_val) / bins as f64;

        let mut histogram_bins = vec![0; bins];

        for &value in values {
            let bin_idx = ((value - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(bins - 1); // Ensure we don't go out of bounds
            histogram_bins[bin_idx] += 1;
        }

        let max_frequency = *histogram_bins.iter().max().unwrap_or(&0);

        let bins_data: Vec<HistogramBin> = histogram_bins
            .into_iter()
            .enumerate()
            .map(|(i, count)| HistogramBin {
                range_start: min_val + i as f64 * bin_width,
                range_end: min_val + (i + 1) as f64 * bin_width,
                frequency: count,
            })
            .collect();

        HistogramData {
            bins: bins_data,
            max_frequency,
        }
    }

    /// Pearson correlation between the most recent samples of two metric
    /// categories.
    ///
    /// The two series are paired by RECENCY RANK (newest with newest, and so
    /// on) over at most 50 points, not by timestamp. That is exact while both
    /// categories are sampled on the dashboard's single refresh tick -- the
    /// only way [`ingest_metrics`] feeds them -- and would misalign if a caller
    /// ever ingested two categories at different cadences.
    ///
    /// Returns `0.0` when either category has fewer than two samples or has
    /// zero variance, i.e. when the coefficient is undefined.
    fn calculate_correlation_coefficient(
        &self,
        cat1: &MetricCategory,
        cat2: &MetricCategory,
    ) -> f64 {
        let data = self.metric_data.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        let data1 = match data.get(cat1) {
            Some(d) => d,
            None => return 0.0,
        };

        let data2 = match data.get(cat2) {
            Some(d) => d,
            None => return 0.0,
        };

        if data1.len() < 2 || data2.len() < 2 {
            return 0.0;
        }

        // Take the minimum length to align the datasets
        let min_len = data1.len().min(data2.len()).min(50); // Use at most 50 points for performance
        let values1: Vec<f64> = data1.iter().rev().take(min_len).map(|d| d.value).collect();
        let values2: Vec<f64> = data2.iter().rev().take(min_len).map(|d| d.value).collect();

        // Calculate Pearson correlation coefficient
        let n = values1.len() as f64;
        let mean1 = values1.iter().sum::<f64>() / n;
        let mean2 = values2.iter().sum::<f64>() / n;

        let covariance = values1
            .iter()
            .zip(&values2)
            .map(|(v1, v2)| (v1 - mean1) * (v2 - mean2))
            .sum::<f64>()
            / n;

        let std1 = (values1.iter().map(|v| (v - mean1).powi(2)).sum::<f64>() / n).sqrt();
        let std2 = (values2.iter().map(|v| (v - mean2).powi(2)).sum::<f64>() / n).sqrt();

        if std1 == 0.0 || std2 == 0.0 {
            0.0
        } else {
            covariance / (std1 * std2)
        }
    }

    fn generate_performance_recommendations(
        &self,
        trend: &TrendDirection,
        strength: f64,
        predicted_value: f64,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        match trend {
            TrendDirection::Increasing => {
                if strength > 0.1 {
                    recommendations.push(
                        "Monitor for potential resource exhaustion due to increasing trend"
                            .to_string(),
                    );
                    recommendations.push("Consider scaling resources proactively".to_string());
                }
                if predicted_value > 90.0 {
                    recommendations.push(
                        "Critical threshold approaching - immediate action recommended".to_string(),
                    );
                }
            },
            TrendDirection::Decreasing => {
                if strength > 0.05 {
                    recommendations
                        .push("Investigate potential performance degradation".to_string());
                    recommendations.push("Check for resource leaks or inefficiencies".to_string());
                }
            },
            TrendDirection::Stable => {
                recommendations
                    .push("Performance trend is stable - continue monitoring".to_string());
            },
        }

        if recommendations.is_empty() {
            recommendations.push("No specific recommendations at this time".to_string());
        }

        recommendations
    }

    fn get_filtered_data(
        &self,
        start: u64,
        end: u64,
    ) -> HashMap<MetricCategory, VecDeque<MetricDataPoint>> {
        let data = self.metric_data.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut filtered_data = HashMap::new();

        for (category, points) in data.iter() {
            let filtered_points: VecDeque<MetricDataPoint> = points
                .iter()
                .filter(|p| p.timestamp >= start && p.timestamp <= end)
                .cloned()
                .collect();

            if !filtered_points.is_empty() {
                filtered_data.insert(category.clone(), filtered_points);
            }
        }

        filtered_data
    }

    fn get_all_data(&self) -> HashMap<MetricCategory, VecDeque<MetricDataPoint>> {
        self.metric_data.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
    }
}

// ============================================================================
// Free functions
//
// These take explicit `&Mutex<_>` / `&Arc<_>` references rather than `&self`
// so they can be shared between `&self` methods (`add_metric`, `get_system_stats`,
// ...) and the background tasks spawned by `start()`, which only hold cloned
// `Arc`s of individual fields (not the whole `RealtimeDashboard`).
// ============================================================================

/// Store metric points (bounded by `max_data_points`), broadcast the batch,
/// and evaluate each real value against the alert thresholds. The single
/// path used by `add_metric`/`add_metrics` and by the periodic host-telemetry
/// collector, so manually-supplied and automatically-collected metrics are
/// treated identically.
#[allow(clippy::too_many_arguments)]
fn ingest_metrics(
    metric_data: &Mutex<HashMap<MetricCategory, VecDeque<MetricDataPoint>>>,
    total_data_points: &Mutex<usize>,
    max_data_points: usize,
    alert_history: &Mutex<VecDeque<DashboardAlert>>,
    thresholds: &AlertThresholds,
    websocket_sender: &broadcast::Sender<WebSocketMessage>,
    metrics: Vec<(MetricCategory, String, f64)>,
) -> Result<()> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64;
    let mut data_points = Vec::with_capacity(metrics.len());

    for (category, label, value) in metrics {
        let data_point = MetricDataPoint {
            timestamp,
            value,
            label,
            category: category.clone(),
        };

        {
            let mut data = metric_data.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let category_data = data.entry(category.clone()).or_default();
            category_data.push_back(data_point.clone());
            while category_data.len() > max_data_points {
                category_data.pop_front();
            }
        }

        data_points.push(data_point);
        evaluate_alert(
            alert_history,
            websocket_sender,
            thresholds,
            &category,
            value,
        );
    }

    {
        let mut total = total_data_points.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        *total += data_points.len();
    }

    let message = WebSocketMessage::MetricUpdate { data: data_points };
    let _ = websocket_sender.send(message);

    Ok(())
}

/// Evaluate one real metric value against the configured thresholds and, on
/// breach, record + broadcast a real alert (real value, real threshold --
/// never derived from randomness).
fn evaluate_alert(
    alert_history: &Mutex<VecDeque<DashboardAlert>>,
    websocket_sender: &broadcast::Sender<WebSocketMessage>,
    thresholds: &AlertThresholds,
    category: &MetricCategory,
    value: f64,
) {
    let (severity, title, message, threshold) = match category {
        MetricCategory::Memory if value > thresholds.memory_threshold => (
            AlertSeverity::Warning,
            "High Memory Usage".to_string(),
            format!(
                "Memory usage is {:.1}% (threshold: {:.1}%)",
                value, thresholds.memory_threshold
            ),
            thresholds.memory_threshold,
        ),
        MetricCategory::GPU if value > thresholds.gpu_utilization_threshold => (
            AlertSeverity::Warning,
            "High GPU Utilization".to_string(),
            format!(
                "GPU utilization is {:.1}% (threshold: {:.1}%)",
                value, thresholds.gpu_utilization_threshold
            ),
            thresholds.gpu_utilization_threshold,
        ),
        MetricCategory::Training if value > thresholds.loss_spike_threshold => (
            AlertSeverity::Error,
            "Training Loss Spike".to_string(),
            format!(
                "Loss spike detected: {:.4} (threshold: {:.4})",
                value, thresholds.loss_spike_threshold
            ),
            thresholds.loss_spike_threshold,
        ),
        _ => return,
    };

    let alert = DashboardAlert {
        id: Uuid::new_v4().to_string(),
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
        severity,
        category: category.clone(),
        title,
        message,
        value: Some(value),
        threshold: Some(threshold),
    };

    {
        let mut history = alert_history.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        history.push_back(alert.clone());
        while history.len() > 100 {
            history.pop_front();
        }
    }

    let _ = websocket_sender.send(WebSocketMessage::Alert { alert });
}

/// Look for a *sustained* threshold breach: the last [`SUSTAINED_BREACH_WINDOW`]
/// real samples of a category all above threshold (as opposed to a single
/// noisy spike, which `evaluate_alert` already covers per-sample). Reads the
/// real, now-populated `metric_data` history -- this used to lock both
/// mutexes and do nothing.
fn check_threshold_breaches(
    metric_data: &Mutex<HashMap<MetricCategory, VecDeque<MetricDataPoint>>>,
    thresholds: &AlertThresholds,
    alert_history: &Mutex<VecDeque<DashboardAlert>>,
    websocket_sender: &broadcast::Sender<WebSocketMessage>,
) {
    let breaches: Vec<(MetricCategory, f64, f64)> = {
        let data = metric_data.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        [
            (MetricCategory::Memory, thresholds.memory_threshold),
            (MetricCategory::GPU, thresholds.gpu_utilization_threshold),
        ]
        .into_iter()
        .filter_map(|(category, threshold)| {
            let points = data.get(&category)?;
            if points.len() < SUSTAINED_BREACH_WINDOW {
                return None;
            }
            let recent: Vec<f64> =
                points.iter().rev().take(SUSTAINED_BREACH_WINDOW).map(|p| p.value).collect();
            if recent.iter().all(|&v| v > threshold) {
                let avg = recent.iter().sum::<f64>() / recent.len() as f64;
                Some((category, avg, threshold))
            } else {
                None
            }
        })
        .collect()
    };

    for (category, avg_value, threshold) in breaches {
        let alert = DashboardAlert {
            id: Uuid::new_v4().to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            severity: AlertSeverity::Error,
            category: category.clone(),
            title: format!("Sustained {category:?} threshold breach"),
            message: format!(
                "{category:?} has stayed above {threshold:.1} for the last \
                 {SUSTAINED_BREACH_WINDOW} samples (avg {avg_value:.1})"
            ),
            value: Some(avg_value),
            threshold: Some(threshold),
        };

        {
            let mut history = alert_history.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            history.push_back(alert.clone());
            while history.len() > 100 {
                history.pop_front();
            }
        }
        let _ = websocket_sender.send(WebSocketMessage::Alert { alert });
    }
}

/// Collect real host telemetry (`sysinfo`) for the categories enabled in
/// `cfg`, as `(category, label, value)` tuples ready for [`ingest_metrics`].
/// GPU readings are honestly omitted (not fabricated) when no GPU telemetry
/// backend is available -- see [`get_gpu_utilization`].
fn collect_system_metrics(
    cfg: &DashboardConfig,
    system_info: &Mutex<System>,
) -> Vec<(MetricCategory, String, f64)> {
    let mut metrics = Vec::new();

    if cfg.enable_memory_profiling {
        if let Some(memory_usage) = get_memory_usage(system_info) {
            metrics.push((
                MetricCategory::Memory,
                "Memory Usage".to_string(),
                memory_usage,
            ));
        }
    }

    if cfg.enable_gpu_monitoring {
        match get_gpu_utilization() {
            Some(gpu_utilization) => {
                metrics.push((
                    MetricCategory::GPU,
                    "GPU Utilization".to_string(),
                    gpu_utilization,
                ));
            },
            None => tracing::debug!(
                "GPU monitoring is enabled but no GPU telemetry backend is available on this \
                 build/machine; skipping (not fabricating a reading)."
            ),
        }
        if let Some(gpu_memory) = get_gpu_memory_usage() {
            metrics.push((MetricCategory::GPU, "GPU Memory".to_string(), gpu_memory));
        }
    }

    metrics
}

/// Real system memory usage, as a percentage of total memory.
fn get_memory_usage(system_info: &Mutex<System>) -> Option<f64> {
    let mut sys = system_info.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    sys.refresh_memory();
    let total = sys.total_memory();
    if total == 0 {
        return None;
    }
    Some(sys.used_memory() as f64 / total as f64 * 100.0)
}

/// Real GPU utilization percentage -- always `None`. No pure-Rust,
/// C/C++-free GPU telemetry backend (NVML/ROCm-SMI) is wired into this
/// build (see the `cuda`/`rocm`/`tpu` Cargo features, which are placeholders
/// today), so this honestly reports "unavailable on this machine" rather
/// than fabricating a number.
fn get_gpu_utilization() -> Option<f64> {
    None
}

/// Real GPU memory usage percentage -- see [`get_gpu_utilization`].
fn get_gpu_memory_usage() -> Option<f64> {
    None
}

/// Real host-wide CPU usage percentage via `sysinfo`. CPU usage is
/// delta-based: the very first reading in a process is not meaningful, but
/// `system_info` is long-lived on `RealtimeDashboard` and refreshed on every
/// call, so accuracy improves as the dashboard runs.
fn estimate_cpu_usage(system_info: &Mutex<System>) -> f64 {
    let mut sys = system_info.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    sys.refresh_cpu_usage();
    sys.global_cpu_usage() as f64
}

/// Rough estimate (in MB) of the dashboard's own retained metric-history
/// footprint, not host telemetry.
fn dashboard_footprint_mb(
    metric_data: &Mutex<HashMap<MetricCategory, VecDeque<MetricDataPoint>>>,
) -> f64 {
    let data = metric_data.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let total_points: usize = data.values().map(|deque| deque.len()).sum();
    // Rough estimate: ~100 bytes per data point.
    (total_points * 100) as f64 / (1024.0 * 1024.0)
}

/// Dashboard builder for easier configuration
#[derive(Debug, Default)]
pub struct DashboardBuilder {
    config: DashboardConfig,
}

impl DashboardBuilder {
    /// Create new dashboard builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set WebSocket port
    pub fn port(mut self, port: u16) -> Self {
        self.config.websocket_port = port;
        self
    }

    /// Set update frequency
    pub fn update_frequency(mut self, frequency_ms: u64) -> Self {
        self.config.update_frequency_ms = frequency_ms;
        self
    }

    /// Set maximum data points
    pub fn max_data_points(mut self, max_points: usize) -> Self {
        self.config.max_data_points = max_points;
        self
    }

    /// Enable/disable GPU monitoring
    pub fn gpu_monitoring(mut self, enabled: bool) -> Self {
        self.config.enable_gpu_monitoring = enabled;
        self
    }

    /// Enable/disable memory profiling
    pub fn memory_profiling(mut self, enabled: bool) -> Self {
        self.config.enable_memory_profiling = enabled;
        self
    }

    /// Set alert thresholds
    pub fn alert_thresholds(mut self, thresholds: AlertThresholds) -> Self {
        self.config.alert_thresholds = thresholds;
        self
    }

    /// Build the dashboard
    pub fn build(self) -> RealtimeDashboard {
        RealtimeDashboard::new(self.config)
    }
}

#[cfg(test)]
#[path = "realtime_dashboard_tests.rs"]
mod realtime_dashboard_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use std::time::Duration;

    #[tokio::test]
    async fn test_dashboard_creation() {
        let dashboard = DashboardBuilder::new()
            .port(8081)
            .update_frequency(50)
            .max_data_points(500)
            .build();

        assert_eq!(dashboard.get_config().websocket_port, 8081);
        assert_eq!(dashboard.get_config().update_frequency_ms, 50);
        assert_eq!(dashboard.get_config().max_data_points, 500);
    }

    #[tokio::test]
    async fn test_metric_addition() {
        let dashboard = DashboardBuilder::new().build();

        let result = dashboard.add_metric(MetricCategory::Training, "loss".to_string(), 0.5);

        assert!(result.is_ok());

        let historical_data = dashboard.get_historical_data(&MetricCategory::Training);
        assert_eq!(historical_data.len(), 1);
        assert_eq!(historical_data[0].value, 0.5);
        assert_eq!(historical_data[0].label, "loss");
    }

    #[tokio::test]
    async fn test_batch_metrics() {
        let dashboard = DashboardBuilder::new().build();

        let metrics = vec![
            (MetricCategory::Training, "loss".to_string(), 0.5),
            (MetricCategory::Training, "accuracy".to_string(), 0.9),
            (MetricCategory::GPU, "utilization".to_string(), 75.0),
        ];

        let result = dashboard.add_metrics(metrics);
        assert!(result.is_ok());

        let training_data = dashboard.get_historical_data(&MetricCategory::Training);
        assert_eq!(training_data.len(), 2);

        let gpu_data = dashboard.get_historical_data(&MetricCategory::GPU);
        assert_eq!(gpu_data.len(), 1);
    }

    #[tokio::test]
    async fn test_alert_creation() {
        let dashboard = DashboardBuilder::new().build();

        let result = dashboard.create_alert(
            AlertSeverity::Warning,
            MetricCategory::Memory,
            "High Memory".to_string(),
            "Memory usage is high".to_string(),
            Some(95.0),
            Some(90.0),
        );

        assert!(result.is_ok());

        let history = dashboard.alert_history.lock().expect("lock should not be poisoned");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].title, "High Memory");
    }

    #[tokio::test]
    async fn test_websocket_subscription() {
        let dashboard = DashboardBuilder::new().build();

        let mut stream = dashboard.subscribe();

        // Start the dashboard
        let dashboard_clone = Arc::new(dashboard);
        let dashboard_for_task = dashboard_clone.clone();

        tokio::spawn(async move {
            let _ = dashboard_for_task.start().await;
        });

        // Add a metric to trigger a message
        let _ =
            dashboard_clone.add_metric(MetricCategory::Training, "test_metric".to_string(), 42.0);

        // The subscriber legitimately also sees other real broadcasts (host
        // telemetry from the periodic collector, `SystemStats` updates, ...),
        // so scan the stream for the specific update rather than assuming
        // it's the very first message -- a real client would do the same.
        let found = tokio::time::timeout(Duration::from_millis(500), async {
            loop {
                match stream.next().await {
                    Some(Ok(WebSocketMessage::MetricUpdate { data })) => {
                        if let Some(point) = data.iter().find(|p| p.label == "test_metric") {
                            return Some(point.clone());
                        }
                    },
                    Some(_) => continue,
                    None => return None,
                }
            }
        })
        .await;

        dashboard_clone.stop();

        let point = found
            .expect("should not time out waiting for the test_metric update")
            .expect("stream should not end before the test_metric update arrives");
        assert_eq!(point.value, 42.0);
        assert_eq!(point.label, "test_metric");
    }

    #[tokio::test]
    async fn test_system_stats() {
        let dashboard = DashboardBuilder::new().build();

        // Add some data
        let _ = dashboard.add_metric(MetricCategory::Training, "loss".to_string(), 0.5);
        let _ = dashboard.create_alert(
            AlertSeverity::Info,
            MetricCategory::Training,
            "Test Alert".to_string(),
            "Test message".to_string(),
            None,
            None,
        );

        let stats = dashboard.get_system_stats();

        assert_eq!(stats.data_points_collected, 1);
        assert_eq!(stats.total_alerts, 1);
        // uptime is a Duration which is always >= 0
    }

    #[tokio::test]
    async fn test_data_point_limit() {
        let dashboard = DashboardBuilder::new().max_data_points(2).build();

        // Add 3 data points
        let _ = dashboard.add_metric(MetricCategory::Training, "metric1".to_string(), 1.0);
        let _ = dashboard.add_metric(MetricCategory::Training, "metric2".to_string(), 2.0);
        let _ = dashboard.add_metric(MetricCategory::Training, "metric3".to_string(), 3.0);

        let data = dashboard.get_historical_data(&MetricCategory::Training);

        // Should only keep the last 2 data points
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].value, 2.0); // First of the remaining two
        assert_eq!(data[1].value, 3.0); // Last added
    }

    #[test]
    fn test_get_memory_usage_is_real_not_random() {
        let system_info = Mutex::new(System::new_all());
        let a = get_memory_usage(&system_info).expect("memory usage should be available");
        let b = get_memory_usage(&system_info).expect("memory usage should be available");
        assert!((0.0..=100.0).contains(&a));
        assert!((0.0..=100.0).contains(&b));
        // The old implementation was `50.0 + thread_rng().random::<f64>() * 40.0`,
        // i.e. uniformly random over a 40-point-wide band on every call. Two
        // real readings taken back-to-back on the same host must be far more
        // stable than that.
        assert!(
            (a - b).abs() < 10.0,
            "consecutive real memory readings should be close: {a} vs {b}"
        );
    }

    #[test]
    fn test_estimate_cpu_usage_is_real_not_random() {
        let system_info = Mutex::new(System::new_all());
        let cpu = estimate_cpu_usage(&system_info);
        // A real reading is a finite, non-negative percentage. The old
        // implementation was a fixed `5.0..=15.0` band regardless of the
        // host; a real one is unbounded above (though in practice well
        // under a few hundred) since it comes straight from the OS.
        assert!(cpu.is_finite() && cpu >= 0.0, "got {cpu}");
    }

    #[test]
    fn test_gpu_metrics_are_honestly_absent_not_fabricated() {
        // The old implementation always returned a random number (e.g. in
        // [30, 90] for utilization) regardless of whether a GPU -- or any
        // GPU telemetry backend -- was actually present.
        assert_eq!(get_gpu_utilization(), None);
        assert_eq!(get_gpu_memory_usage(), None);

        let cfg = DashboardConfig {
            enable_gpu_monitoring: true,
            ..Default::default()
        };
        let system_info = Mutex::new(System::new_all());
        let metrics = collect_system_metrics(&cfg, &system_info);
        assert!(
            metrics.iter().all(|(category, _, _)| *category != MetricCategory::GPU),
            "no fabricated GPU metric should be emitted when no GPU backend is available"
        );
    }

    #[test]
    fn test_check_threshold_breaches_raises_a_real_alert() {
        let metric_data: Mutex<HashMap<MetricCategory, VecDeque<MetricDataPoint>>> =
            Mutex::new(HashMap::new());
        let alert_history: Mutex<VecDeque<DashboardAlert>> = Mutex::new(VecDeque::new());
        let (websocket_sender, _rx) = broadcast::channel(16);
        let thresholds = AlertThresholds::default();

        {
            let mut data = metric_data.lock().expect("lock should not be poisoned");
            let points = data.entry(MetricCategory::Memory).or_default();
            for i in 0..SUSTAINED_BREACH_WINDOW {
                points.push_back(MetricDataPoint {
                    timestamp: i as u64,
                    value: thresholds.memory_threshold + 5.0,
                    label: "Memory Usage".to_string(),
                    category: MetricCategory::Memory,
                });
            }
        }

        // The old implementation locked both mutexes and returned without
        // ever inspecting the data or creating an alert.
        check_threshold_breaches(&metric_data, &thresholds, &alert_history, &websocket_sender);

        let history = alert_history.lock().expect("lock should not be poisoned");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].category, MetricCategory::Memory);
        assert!(
            history[0].value.expect("alert should carry a value") > thresholds.memory_threshold
        );
    }

    #[test]
    fn test_check_threshold_breaches_ignores_non_sustained_spikes() {
        let metric_data: Mutex<HashMap<MetricCategory, VecDeque<MetricDataPoint>>> =
            Mutex::new(HashMap::new());
        let alert_history: Mutex<VecDeque<DashboardAlert>> = Mutex::new(VecDeque::new());
        let (websocket_sender, _rx) = broadcast::channel(16);
        let thresholds = AlertThresholds::default();

        {
            let mut data = metric_data.lock().expect("lock should not be poisoned");
            let points = data.entry(MetricCategory::Memory).or_default();
            // One spike above threshold, rest comfortably below it.
            points.push_back(MetricDataPoint {
                timestamp: 0,
                value: thresholds.memory_threshold + 5.0,
                label: "Memory Usage".to_string(),
                category: MetricCategory::Memory,
            });
            for i in 1..SUSTAINED_BREACH_WINDOW {
                points.push_back(MetricDataPoint {
                    timestamp: i as u64,
                    value: thresholds.memory_threshold - 20.0,
                    label: "Memory Usage".to_string(),
                    category: MetricCategory::Memory,
                });
            }
        }

        check_threshold_breaches(&metric_data, &thresholds, &alert_history, &websocket_sender);

        let history = alert_history.lock().expect("lock should not be poisoned");
        assert!(
            history.is_empty(),
            "a single spike must not be treated as a sustained breach"
        );
    }

    #[tokio::test]
    async fn test_automatic_collection_actually_records_real_memory_metrics() {
        let dashboard =
            Arc::new(DashboardBuilder::new().update_frequency(20).memory_profiling(true).build());
        let handle = dashboard.clone();

        tokio::spawn(async move {
            let _ = handle.start().await;
        });

        // Long enough for several 20ms collection ticks.
        tokio::time::sleep(Duration::from_millis(150)).await;
        dashboard.stop();

        // The old implementation captured `_metric_data` (a deliberately
        // unused clone) in the collection task and only ever broadcast
        // metrics over the WebSocket channel, so `metric_data` -- and
        // therefore `get_historical_data` -- stayed empty forever for
        // automatically-collected categories.
        let memory_data = dashboard.get_historical_data(&MetricCategory::Memory);
        assert!(
            !memory_data.is_empty(),
            "the periodic collector must actually store real telemetry, not just broadcast it"
        );
        for point in &memory_data {
            assert!((0.0..=100.0).contains(&point.value));
        }
    }
}
