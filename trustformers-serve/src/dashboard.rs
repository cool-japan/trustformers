//! Real-time Monitoring Dashboard
//!
//! Provides web-based real-time monitoring dashboards with interactive charts,
//! live metrics, and customizable visualizations for the inference server.

use anyhow::Result;
use axum::{
    extract::{ws::WebSocket, Path, WebSocketUpgrade},
    response::{Html, IntoResponse},
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Enable dashboard service
    pub enabled: bool,
    /// Dashboard web server port
    pub port: u16,
    /// Dashboard host
    pub host: String,
    /// WebSocket update interval in milliseconds
    pub update_interval_ms: u64,
    /// Maximum number of data points to retain
    pub max_data_points: usize,
    /// Enable real-time alerts
    pub enable_alerts: bool,
    /// Dashboard theme
    pub theme: DashboardTheme,
    /// Available dashboard layouts
    pub layouts: Vec<DashboardLayout>,
    /// Auto-refresh interval in seconds
    pub auto_refresh_seconds: u64,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 8081,
            host: "0.0.0.0".to_string(),
            update_interval_ms: 1000,
            max_data_points: 1000,
            enable_alerts: true,
            theme: DashboardTheme::Dark,
            layouts: vec![
                DashboardLayout::default_system_overview(),
                DashboardLayout::default_performance_dashboard(),
                DashboardLayout::default_model_monitoring(),
            ],
            auto_refresh_seconds: 30,
        }
    }
}

/// Dashboard theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardTheme {
    Light,
    Dark,
    Blue,
    Green,
    Custom {
        primary_color: String,
        secondary_color: String,
    },
}

/// Dashboard layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardLayout {
    /// Layout identifier
    pub id: String,
    /// Layout name
    pub name: String,
    /// Layout description
    pub description: String,
    /// Dashboard widgets
    pub widgets: Vec<DashboardWidget>,
    /// Layout grid configuration
    pub grid_config: GridConfig,
    /// Auto-update enabled
    pub auto_update: bool,
}

impl DashboardLayout {
    pub fn default_system_overview() -> Self {
        Self {
            id: "system_overview".to_string(),
            name: "System Overview".to_string(),
            description: "High-level system metrics and health indicators".to_string(),
            widgets: vec![
                DashboardWidget::default_cpu_widget(),
                DashboardWidget::default_memory_widget(),
                DashboardWidget::default_gpu_widget(),
                DashboardWidget::default_requests_widget(),
            ],
            grid_config: GridConfig {
                rows: 2,
                columns: 2,
            },
            auto_update: true,
        }
    }

    pub fn default_performance_dashboard() -> Self {
        Self {
            id: "performance".to_string(),
            name: "Performance Dashboard".to_string(),
            description: "Detailed performance metrics and analytics".to_string(),
            widgets: vec![
                DashboardWidget::default_latency_widget(),
                DashboardWidget::default_throughput_widget(),
                DashboardWidget::default_batch_size_widget(),
                DashboardWidget::default_queue_depth_widget(),
            ],
            grid_config: GridConfig {
                rows: 2,
                columns: 2,
            },
            auto_update: true,
        }
    }

    pub fn default_model_monitoring() -> Self {
        Self {
            id: "model_monitoring".to_string(),
            name: "Model Monitoring".to_string(),
            description: "Model-specific metrics and performance tracking".to_string(),
            widgets: vec![
                DashboardWidget::default_model_latency_widget(),
                DashboardWidget::default_model_accuracy_widget(),
                DashboardWidget::default_model_load_widget(),
                DashboardWidget::default_cache_hit_rate_widget(),
            ],
            grid_config: GridConfig {
                rows: 2,
                columns: 2,
            },
            auto_update: true,
        }
    }
}

/// Grid configuration for layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    pub rows: usize,
    pub columns: usize,
}

/// Dashboard widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardWidget {
    /// Widget identifier
    pub id: String,
    /// Widget title
    pub title: String,
    /// Widget type
    pub widget_type: WidgetType,
    /// Data source configuration
    pub data_source: DataSourceConfig,
    /// Display configuration
    pub display_config: DisplayConfig,
    /// Widget size and position
    pub layout: WidgetLayout,
    /// Update interval in seconds
    pub update_interval: u64,
    /// Widget-specific options
    pub options: HashMap<String, serde_json::Value>,
}

impl DashboardWidget {
    pub fn default_cpu_widget() -> Self {
        Self {
            id: "cpu_usage".to_string(),
            title: "CPU Usage".to_string(),
            widget_type: WidgetType::Gauge,
            data_source: DataSourceConfig {
                metric_name: "system_cpu_usage".to_string(),
                aggregation: AggregationType::Average,
                time_window: Duration::from_secs(60),
            },
            display_config: DisplayConfig {
                chart_type: ChartType::Gauge,
                color_scheme: ColorScheme::BlueGreen,
                show_legend: false,
                show_grid: false,
                min_value: Some(0.0),
                max_value: Some(100.0),
            },
            layout: WidgetLayout {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
            update_interval: 5,
            options: HashMap::new(),
        }
    }

    pub fn default_memory_widget() -> Self {
        Self {
            id: "memory_usage".to_string(),
            title: "Memory Usage".to_string(),
            widget_type: WidgetType::Gauge,
            data_source: DataSourceConfig {
                metric_name: "system_memory_usage".to_string(),
                aggregation: AggregationType::Average,
                time_window: Duration::from_secs(60),
            },
            display_config: DisplayConfig {
                chart_type: ChartType::Gauge,
                color_scheme: ColorScheme::OrangeRed,
                show_legend: false,
                show_grid: false,
                min_value: Some(0.0),
                max_value: Some(100.0),
            },
            layout: WidgetLayout {
                x: 1,
                y: 0,
                width: 1,
                height: 1,
            },
            update_interval: 5,
            options: HashMap::new(),
        }
    }

    pub fn default_gpu_widget() -> Self {
        Self {
            id: "gpu_usage".to_string(),
            title: "GPU Usage".to_string(),
            widget_type: WidgetType::Gauge,
            data_source: DataSourceConfig {
                metric_name: "system_gpu_usage".to_string(),
                aggregation: AggregationType::Average,
                time_window: Duration::from_secs(60),
            },
            display_config: DisplayConfig {
                chart_type: ChartType::Gauge,
                color_scheme: ColorScheme::Purple,
                show_legend: false,
                show_grid: false,
                min_value: Some(0.0),
                max_value: Some(100.0),
            },
            layout: WidgetLayout {
                x: 0,
                y: 1,
                width: 1,
                height: 1,
            },
            update_interval: 5,
            options: HashMap::new(),
        }
    }

    pub fn default_requests_widget() -> Self {
        Self {
            id: "request_rate".to_string(),
            title: "Request Rate".to_string(),
            widget_type: WidgetType::LineChart,
            data_source: DataSourceConfig {
                metric_name: "inference_requests_total".to_string(),
                aggregation: AggregationType::Rate,
                time_window: Duration::from_secs(300),
            },
            display_config: DisplayConfig {
                chart_type: ChartType::Line,
                color_scheme: ColorScheme::Blue,
                show_legend: true,
                show_grid: true,
                min_value: None,
                max_value: None,
            },
            layout: WidgetLayout {
                x: 1,
                y: 1,
                width: 1,
                height: 1,
            },
            update_interval: 5,
            options: HashMap::new(),
        }
    }

    pub fn default_latency_widget() -> Self {
        Self {
            id: "latency_percentiles".to_string(),
            title: "Latency Percentiles".to_string(),
            widget_type: WidgetType::LineChart,
            data_source: DataSourceConfig {
                metric_name: "inference_request_duration_seconds".to_string(),
                aggregation: AggregationType::Percentile(95.0),
                time_window: Duration::from_secs(300),
            },
            display_config: DisplayConfig {
                chart_type: ChartType::Line,
                color_scheme: ColorScheme::Green,
                show_legend: true,
                show_grid: true,
                min_value: Some(0.0),
                max_value: None,
            },
            layout: WidgetLayout {
                x: 0,
                y: 0,
                width: 2,
                height: 1,
            },
            update_interval: 5,
            options: HashMap::new(),
        }
    }

    pub fn default_throughput_widget() -> Self {
        Self {
            id: "throughput".to_string(),
            title: "Throughput (RPS)".to_string(),
            widget_type: WidgetType::AreaChart,
            data_source: DataSourceConfig {
                metric_name: "inference_throughput".to_string(),
                aggregation: AggregationType::Average,
                time_window: Duration::from_secs(300),
            },
            display_config: DisplayConfig {
                chart_type: ChartType::Area,
                color_scheme: ColorScheme::Blue,
                show_legend: false,
                show_grid: true,
                min_value: Some(0.0),
                max_value: None,
            },
            layout: WidgetLayout {
                x: 0,
                y: 1,
                width: 1,
                height: 1,
            },
            update_interval: 5,
            options: HashMap::new(),
        }
    }

    pub fn default_batch_size_widget() -> Self {
        Self {
            id: "batch_size_dist".to_string(),
            title: "Batch Size Distribution".to_string(),
            widget_type: WidgetType::Histogram,
            data_source: DataSourceConfig {
                metric_name: "inference_batch_size".to_string(),
                aggregation: AggregationType::Distribution,
                time_window: Duration::from_secs(300),
            },
            display_config: DisplayConfig {
                chart_type: ChartType::Bar,
                color_scheme: ColorScheme::OrangeRed,
                show_legend: false,
                show_grid: true,
                min_value: None,
                max_value: None,
            },
            layout: WidgetLayout {
                x: 1,
                y: 1,
                width: 1,
                height: 1,
            },
            update_interval: 10,
            options: HashMap::new(),
        }
    }

    pub fn default_queue_depth_widget() -> Self {
        Self {
            id: "queue_depth".to_string(),
            title: "Queue Depth".to_string(),
            widget_type: WidgetType::Gauge,
            data_source: DataSourceConfig {
                metric_name: "inference_queue_size".to_string(),
                aggregation: AggregationType::Current,
                time_window: Duration::from_secs(60),
            },
            display_config: DisplayConfig {
                chart_type: ChartType::Gauge,
                color_scheme: ColorScheme::Yellow,
                show_legend: false,
                show_grid: false,
                min_value: Some(0.0),
                max_value: Some(1000.0),
            },
            layout: WidgetLayout {
                x: 2,
                y: 0,
                width: 1,
                height: 1,
            },
            update_interval: 2,
            options: HashMap::new(),
        }
    }

    pub fn default_model_latency_widget() -> Self {
        Self {
            id: "model_latency".to_string(),
            title: "Model Latency by Type".to_string(),
            widget_type: WidgetType::LineChart,
            data_source: DataSourceConfig {
                metric_name: "model_inference_latency".to_string(),
                aggregation: AggregationType::Average,
                time_window: Duration::from_secs(300),
            },
            display_config: DisplayConfig {
                chart_type: ChartType::Line,
                color_scheme: ColorScheme::Multicolor,
                show_legend: true,
                show_grid: true,
                min_value: None,
                max_value: None,
            },
            layout: WidgetLayout {
                x: 0,
                y: 0,
                width: 2,
                height: 1,
            },
            update_interval: 5,
            options: HashMap::new(),
        }
    }

    pub fn default_model_accuracy_widget() -> Self {
        Self {
            id: "model_accuracy".to_string(),
            title: "Model Accuracy".to_string(),
            widget_type: WidgetType::Gauge,
            data_source: DataSourceConfig {
                metric_name: "model_accuracy_score".to_string(),
                aggregation: AggregationType::Average,
                time_window: Duration::from_secs(300),
            },
            display_config: DisplayConfig {
                chart_type: ChartType::Gauge,
                color_scheme: ColorScheme::Green,
                show_legend: false,
                show_grid: false,
                min_value: Some(0.0),
                max_value: Some(1.0),
            },
            layout: WidgetLayout {
                x: 0,
                y: 1,
                width: 1,
                height: 1,
            },
            update_interval: 30,
            options: HashMap::new(),
        }
    }

    pub fn default_model_load_widget() -> Self {
        Self {
            id: "model_load_time".to_string(),
            title: "Model Load Time".to_string(),
            widget_type: WidgetType::BarChart,
            data_source: DataSourceConfig {
                metric_name: "model_load_duration".to_string(),
                aggregation: AggregationType::Average,
                time_window: Duration::from_secs(3600),
            },
            display_config: DisplayConfig {
                chart_type: ChartType::Bar,
                color_scheme: ColorScheme::Purple,
                show_legend: true,
                show_grid: true,
                min_value: Some(0.0),
                max_value: None,
            },
            layout: WidgetLayout {
                x: 1,
                y: 1,
                width: 1,
                height: 1,
            },
            update_interval: 60,
            options: HashMap::new(),
        }
    }

    pub fn default_cache_hit_rate_widget() -> Self {
        Self {
            id: "cache_hit_rate".to_string(),
            title: "Cache Hit Rate".to_string(),
            widget_type: WidgetType::Gauge,
            data_source: DataSourceConfig {
                metric_name: "cache_hit_rate".to_string(),
                aggregation: AggregationType::Average,
                time_window: Duration::from_secs(300),
            },
            display_config: DisplayConfig {
                chart_type: ChartType::Gauge,
                color_scheme: ColorScheme::BlueGreen,
                show_legend: false,
                show_grid: false,
                min_value: Some(0.0),
                max_value: Some(1.0),
            },
            layout: WidgetLayout {
                x: 2,
                y: 0,
                width: 1,
                height: 1,
            },
            update_interval: 10,
            options: HashMap::new(),
        }
    }
}

/// Widget types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    LineChart,
    AreaChart,
    BarChart,
    Gauge,
    Counter,
    Table,
    Heatmap,
    Histogram,
    PieChart,
    ScatterPlot,
}

/// Data source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceConfig {
    /// Metric name to query
    pub metric_name: String,
    /// Data aggregation method
    pub aggregation: AggregationType,
    /// Time window for data
    pub time_window: Duration,
}

/// Aggregation types for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationType {
    Average,
    Sum,
    Min,
    Max,
    Count,
    Rate,
    Current,
    Percentile(f64),
    Distribution,
}

/// Display configuration for widgets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayConfig {
    /// Chart type for rendering
    pub chart_type: ChartType,
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Show legend
    pub show_legend: bool,
    /// Show grid
    pub show_grid: bool,
    /// Minimum value for Y-axis
    pub min_value: Option<f64>,
    /// Maximum value for Y-axis
    pub max_value: Option<f64>,
}

/// Chart types for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    Line,
    Area,
    Bar,
    Gauge,
    Pie,
    Scatter,
    Heatmap,
}

/// Color schemes for charts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorScheme {
    Blue,
    Green,
    Red,
    Purple,
    Orange,
    Yellow,
    BlueGreen,
    OrangeRed,
    Multicolor,
}

/// Widget layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetLayout {
    /// X position in grid
    pub x: usize,
    /// Y position in grid
    pub y: usize,
    /// Width in grid units
    pub width: usize,
    /// Height in grid units
    pub height: usize,
}

/// Real-time data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp
    pub timestamp: u64,
    /// Value
    pub value: f64,
    /// Optional labels
    pub labels: HashMap<String, String>,
}

/// Widget data update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetUpdate {
    /// Widget ID
    pub widget_id: String,
    /// New data points
    pub data: Vec<DataPoint>,
    /// Update timestamp
    pub timestamp: u64,
}

/// Dashboard alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardAlert {
    /// Alert ID
    pub id: String,
    /// Alert title
    pub title: String,
    /// Alert message
    pub message: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert timestamp
    pub timestamp: u64,
    /// Related metric
    pub metric_name: String,
    /// Alert value
    pub value: f64,
    /// Alert threshold
    pub threshold: f64,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WsMessage {
    /// Widget data update
    WidgetUpdate { data: WidgetUpdate },
    /// Dashboard alert
    Alert { alert: DashboardAlert },
    /// Heartbeat/ping
    Ping { timestamp: u64 },
    /// Client subscription to widget
    Subscribe { widget_id: String },
    /// Client unsubscription from widget
    Unsubscribe { widget_id: String },
    /// Error message
    Error { message: String },
}

/// Dashboard service state
#[derive(Clone)]
pub struct DashboardService {
    /// Configuration
    config: DashboardConfig,
    /// Widget data storage
    widget_data: Arc<RwLock<HashMap<String, VecDeque<DataPoint>>>>,
    /// Active WebSocket connections
    connections: Arc<RwLock<HashMap<Uuid, WebSocketConnection>>>,
    /// Broadcast channel for real-time updates
    broadcast_tx: broadcast::Sender<WsMessage>,
    /// Dashboard statistics
    stats: Arc<DashboardStats>,
}

/// WebSocket connection info
#[derive(Debug)]
pub struct WebSocketConnection {
    /// Connection ID
    pub id: Uuid,
    /// Subscribed widgets
    pub subscribed_widgets: HashSet<String>,
    /// Connection start time
    pub connected_at: Instant,
    /// Last activity time
    pub last_activity: Instant,
}

use std::collections::HashSet;

/// Dashboard statistics
#[derive(Debug, Default)]
pub struct DashboardStats {
    /// Total widget updates sent
    pub updates_sent: AtomicU64,
    /// Active connections
    pub active_connections: AtomicU64,
    /// Total alerts generated
    pub alerts_generated: AtomicU64,
    /// Data points collected
    pub data_points_collected: AtomicU64,
}

impl DashboardService {
    /// Create a new dashboard service
    pub fn new(config: DashboardConfig) -> Result<Self> {
        let (broadcast_tx, _) = broadcast::channel(1000);

        Ok(Self {
            config,
            widget_data: Arc::new(RwLock::new(HashMap::new())),
            connections: Arc::new(RwLock::new(HashMap::new())),
            broadcast_tx,
            stats: Arc::new(DashboardStats::default()),
        })
    }

    /// Start the dashboard service
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Start data collection task
        self.start_data_collection_task().await?;

        // Start WebSocket cleanup task
        self.start_cleanup_task().await?;

        // Start dashboard web server
        self.start_web_server().await?;

        Ok(())
    }

    /// Create dashboard router
    pub fn create_router(&self) -> Router {
        Router::new()
            .route("/", get(Self::dashboard_home))
            .route("/dashboard/:layout_id", get(Self::dashboard_layout))
            .route("/api/layouts", get(Self::get_layouts))
            .route("/api/widgets/:widget_id/data", get(Self::get_widget_data))
            .route("/api/alerts", get(Self::get_alerts))
            .route("/ws", get(Self::websocket_handler))
            .with_state(self.clone())
    }

    /// Update widget data
    pub async fn update_widget_data(&self, widget_id: &str, data_point: DataPoint) -> Result<()> {
        // Store data point
        {
            let mut widget_data = self.widget_data.write().await;
            let series = widget_data.entry(widget_id.to_string()).or_insert_with(VecDeque::new);
            series.push_back(data_point.clone());

            // Limit series size
            while series.len() > self.config.max_data_points {
                series.pop_front();
            }
        }

        // Broadcast update to WebSocket clients
        let update = WidgetUpdate {
            widget_id: widget_id.to_string(),
            data: vec![data_point],
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        };

        let message = WsMessage::WidgetUpdate { data: update };
        let _ = self.broadcast_tx.send(message);

        self.stats.data_points_collected.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Send alert to dashboard
    pub async fn send_alert(&self, alert: DashboardAlert) -> Result<()> {
        if !self.config.enable_alerts {
            return Ok(());
        }

        let message = WsMessage::Alert { alert };
        let _ = self.broadcast_tx.send(message);

        self.stats.alerts_generated.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Get dashboard statistics
    pub async fn get_stats(&self) -> DashboardStatsSummary {
        let connections = self.connections.read().await;

        DashboardStatsSummary {
            active_connections: connections.len() as u64,
            updates_sent: self.stats.updates_sent.load(Ordering::Relaxed),
            alerts_generated: self.stats.alerts_generated.load(Ordering::Relaxed),
            data_points_collected: self.stats.data_points_collected.load(Ordering::Relaxed),
            widget_count: self.config.layouts.iter().map(|l| l.widgets.len()).sum::<usize>() as u64,
        }
    }

    // HTTP handlers

    async fn dashboard_home() -> impl IntoResponse {
        Html(include_str!("../static/dashboard.html"))
    }

    async fn dashboard_layout(Path(layout_id): Path<String>) -> impl IntoResponse {
        // Return dashboard HTML for specific layout
        Html(format!(
            r#"
            <!DOCTYPE html>
            <html>
            <head>
                <title>Dashboard - {}</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/luxon@3.4.4/build/global/luxon.min.js"></script>
                <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1/dist/chartjs-adapter-luxon.bundle.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: #fff; }}
                    .dashboard-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
                    .widget {{ background: #2a2a2a; border-radius: 8px; padding: 20px; }}
                    .widget h3 {{ margin-top: 0; color: #4CAF50; }}
                    .chart-container {{ position: relative; height: 300px; }}
                </style>
            </head>
            <body>
                <h1>Dashboard: {}</h1>
                <div class="dashboard-grid" id="dashboard">
                    <!-- Widgets will be loaded here -->
                </div>
                <script>
                    const ws = new WebSocket('ws://localhost:8081/ws');
                    const charts = {{}};

                    ws.onmessage = function(event) {{
                        const message = JSON.parse(event.data);
                        if (message.type === 'WidgetUpdate') {{
                            updateWidget(message.data);
                        }}
                    }};

                    function updateWidget(update) {{
                        // Update chart with new data
                        console.log('Widget update:', update);
                    }}

                    // Load initial dashboard layout
                    loadDashboard('{}');

                    async function loadDashboard(layoutId) {{
                        try {{
                            const response = await fetch(`/api/layouts`);
                            const layouts = await response.json();
                            const layout = layouts.find(l => l.id === layoutId);
                            if (layout) {{
                                renderLayout(layout);
                            }}
                        }} catch (error) {{
                            console.error('Failed to load dashboard:', error);
                        }}
                    }}

                    function renderLayout(layout) {{
                        const dashboard = document.getElementById('dashboard');
                        layout.widgets.forEach(widget => {{
                            const widgetElement = createWidget(widget);
                            dashboard.appendChild(widgetElement);
                        }});
                    }}

                    function createWidget(widget) {{
                        const div = document.createElement('div');
                        div.className = 'widget';
                        div.innerHTML = `
                            <h3>${{widget.title}}</h3>
                            <div class="chart-container">
                                <canvas id="chart-${{widget.id}}"></canvas>
                            </div>
                        `;

                        // Initialize chart
                        const ctx = div.querySelector(`#chart-${{widget.id}}`);
                        charts[widget.id] = new Chart(ctx, {{
                            type: getChartType(widget.display_config.chart_type),
                            data: {{
                                datasets: [{{
                                    label: widget.title,
                                    data: [],
                                    borderColor: getColor(widget.display_config.color_scheme),
                                    backgroundColor: getColor(widget.display_config.color_scheme, 0.1),
                                }}]
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                scales: {{
                                    x: {{
                                        type: 'time',
                                        time: {{
                                            unit: 'minute'
                                        }}
                                    }}
                                }}
                            }}
                        }});

                        return div;
                    }}

                    function getChartType(type) {{
                        const typeMap = {{
                            'Line': 'line',
                            'Area': 'line',
                            'Bar': 'bar',
                            'Gauge': 'doughnut'
                        }};
                        return typeMap[type] || 'line';
                    }}

                    function getColor(scheme, alpha = 1) {{
                        const colors = {{
                            'Blue': `rgba(54, 162, 235, ${{alpha}})`,
                            'Green': `rgba(75, 192, 192, ${{alpha}})`,
                            'Red': `rgba(255, 99, 132, ${{alpha}})`,
                            'Purple': `rgba(153, 102, 255, ${{alpha}})`,
                            'Orange': `rgba(255, 159, 64, ${{alpha}})`
                        }};
                        return colors[scheme] || colors['Blue'];
                    }}
                </script>
            </body>
            </html>
            "#,
            layout_id, layout_id, layout_id
        ))
    }

    async fn get_layouts() -> impl IntoResponse {
        let config = DashboardConfig::default();
        Json(config.layouts)
    }

    async fn get_widget_data(Path(widget_id): Path<String>) -> impl IntoResponse {
        // Return widget data (placeholder)
        Json(serde_json::json!({
            "widget_id": widget_id,
            "data": []
        }))
    }

    async fn get_alerts() -> impl IntoResponse {
        // Return current alerts (placeholder)
        Json(serde_json::json!({
            "alerts": []
        }))
    }

    async fn websocket_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
        ws.on_upgrade(Self::handle_websocket)
    }

    async fn handle_websocket(_socket: WebSocket) {
        let connection_id = Uuid::new_v4();
        println!("New WebSocket connection: {}", connection_id);

        // Handle WebSocket messages
        // This is simplified - in practice would handle subscriptions, etc.
    }

    // Private helper methods

    async fn start_data_collection_task(&self) -> Result<()> {
        let service = self.clone();
        let interval = Duration::from_millis(service.config.update_interval_ms);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                // Collect and update widget data
                if let Err(e) = service.collect_widget_data().await {
                    eprintln!("Failed to collect widget data: {}", e);
                }
            }
        });

        Ok(())
    }

    async fn start_cleanup_task(&self) -> Result<()> {
        let service = self.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));

            loop {
                interval.tick().await;

                if let Err(e) = service.cleanup_connections().await {
                    eprintln!("Failed to cleanup connections: {}", e);
                }
            }
        });

        Ok(())
    }

    async fn start_web_server(&self) -> Result<()> {
        let app = self.create_router();
        let addr = format!("{}:{}", self.config.host, self.config.port);

        println!("Dashboard server starting on {}", addr);

        let _service = self.clone();
        tokio::spawn(async move {
            let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
            axum::serve(listener, app).await.unwrap();
        });

        Ok(())
    }

    async fn collect_widget_data(&self) -> Result<()> {
        // Collect data for all widgets in all layouts
        for layout in &self.config.layouts {
            for widget in &layout.widgets {
                // Generate sample data (in practice, would query metrics)
                let data_point = DataPoint {
                    timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                    value: (SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() % 100) as f64, // Simple sample data
                    labels: HashMap::new(),
                };

                self.update_widget_data(&widget.id, data_point).await?;
            }
        }

        Ok(())
    }

    async fn cleanup_connections(&self) -> Result<()> {
        let mut connections = self.connections.write().await;
        let now = Instant::now();
        let timeout = Duration::from_secs(300); // 5 minutes

        connections.retain(|_, conn| now.duration_since(conn.last_activity) < timeout);

        Ok(())
    }
}

/// Dashboard statistics summary
#[derive(Debug, Serialize)]
pub struct DashboardStatsSummary {
    pub active_connections: u64,
    pub updates_sent: u64,
    pub alerts_generated: u64,
    pub data_points_collected: u64,
    pub widget_count: u64,
}

/// Dashboard error types
#[derive(Debug, thiserror::Error)]
pub enum DashboardError {
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    #[error("WebSocket error: {message}")]
    WebSocketError { message: String },

    #[error("Data collection error: {message}")]
    DataCollectionError { message: String },

    #[error("Rendering error: {message}")]
    RenderingError { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dashboard_service_creation() {
        let config = DashboardConfig::default();
        let service = DashboardService::new(config).unwrap();
        assert!(service.config.enabled);
    }

    #[tokio::test]
    async fn test_widget_data_update() {
        let config = DashboardConfig::default();
        let service = DashboardService::new(config).unwrap();

        let data_point = DataPoint {
            timestamp: 1234567890,
            value: 42.0,
            labels: HashMap::new(),
        };

        let result = service.update_widget_data("test_widget", data_point).await;
        assert!(result.is_ok());

        let stats = service.get_stats().await;
        assert_eq!(stats.data_points_collected, 1);
    }

    #[test]
    fn test_dashboard_layout_creation() {
        let layout = DashboardLayout::default_system_overview();
        assert_eq!(layout.id, "system_overview");
        assert_eq!(layout.widgets.len(), 4);
    }
}
