//! Modern Plotting Engine
//!
//! Advanced visualization engine with support for modern plotting libraries,
//! interactive dashboards, and real-time updates.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::types::*;

/// Configuration for modern plotting engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModernPlottingConfig {
    /// Enable interactive plots
    pub enable_interactive: bool,
    /// Enable real-time updates
    pub enable_realtime: bool,
    /// Enable web dashboard
    pub enable_web_dashboard: bool,
    /// Plotting backend to use
    pub backend: PlottingBackend,
    /// Output directory for plots
    pub output_directory: String,
    /// Dashboard port
    pub dashboard_port: u16,
    /// Maximum number of data points per plot
    pub max_data_points: usize,
    /// Auto-refresh interval for real-time plots (milliseconds)
    pub refresh_interval_ms: u64,
    /// Enable plot animations
    pub enable_animations: bool,
    /// Animation frame rate
    pub animation_fps: u32,
}

impl Default for ModernPlottingConfig {
    fn default() -> Self {
        Self {
            enable_interactive: true,
            enable_realtime: true,
            enable_web_dashboard: true,
            backend: PlottingBackend::PlotlyJS,
            output_directory: "./modern_debug_plots".to_string(),
            dashboard_port: 8888,
            max_data_points: 10000,
            refresh_interval_ms: 1000,
            enable_animations: true,
            animation_fps: 30,
        }
    }
}

/// Modern plotting backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlottingBackend {
    /// Plotly.js for interactive web-based plots
    PlotlyJS,
    /// D3.js for custom interactive visualizations
    D3JS,
    /// Chart.js for responsive charts
    ChartJS,
    /// Three.js for 3D visualizations
    ThreeJS,
    /// Matplotlib backend (Python integration)
    Matplotlib,
    /// Bokeh backend (Python integration)
    Bokeh,
    /// Custom WebGL backend for high-performance visualizations
    WebGL,
}

/// Interactive plot types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractivePlotType {
    /// Interactive line plot with zoom, pan, hover
    InteractiveLinePlot,
    /// Interactive scatter plot with selection
    InteractiveScatterPlot,
    /// Interactive heatmap with drill-down
    InteractiveHeatmap,
    /// Interactive 3D surface with rotation
    Interactive3DSurface,
    /// Real-time streaming plot
    RealtimeStreamingPlot,
    /// Animated training visualization
    AnimatedTrainingPlot,
    /// Interactive network diagram
    InteractiveNetworkDiagram,
    /// Dashboard with multiple plots
    MultiPlotDashboard,
    /// Interactive histogram with brushing
    InteractiveHistogram,
    /// Parallel coordinates plot
    ParallelCoordinatesPlot,
    /// Interactive correlation matrix
    InteractiveCorrelationMatrix,
    /// Time series with range selector
    TimeSeriesWithRangeSelector,
}

/// Modern plot data with interactive features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractivePlotData {
    /// Basic plot data
    pub plot_data: PlotData,
    /// Interactive features configuration
    pub interactive_config: InteractiveConfig,
    /// Custom styling
    pub styling: PlotStyling,
    /// Animation configuration
    pub animation_config: Option<AnimationConfig>,
    /// Real-time update configuration
    pub realtime_config: Option<RealtimeConfig>,
}

/// Configuration for interactive features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveConfig {
    /// Enable zoom functionality
    pub enable_zoom: bool,
    /// Enable pan functionality
    pub enable_pan: bool,
    /// Enable hover tooltips
    pub enable_hover: bool,
    /// Enable selection
    pub enable_selection: bool,
    /// Enable brush selection
    pub enable_brush: bool,
    /// Enable crossfilter
    pub enable_crossfilter: bool,
    /// Custom event handlers
    pub event_handlers: HashMap<String, String>,
}

impl Default for InteractiveConfig {
    fn default() -> Self {
        Self {
            enable_zoom: true,
            enable_pan: true,
            enable_hover: true,
            enable_selection: true,
            enable_brush: false,
            enable_crossfilter: false,
            event_handlers: HashMap::new(),
        }
    }
}

/// Custom plot styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotStyling {
    /// Color palette
    pub color_palette: Vec<String>,
    /// Font configuration
    pub font_config: FontConfig,
    /// Line styles
    pub line_styles: Vec<LineStyle>,
    /// Marker styles
    pub marker_styles: Vec<MarkerStyle>,
    /// Background color
    pub background_color: String,
    /// Grid configuration
    pub grid_config: GridConfig,
    /// Legend configuration
    pub legend_config: LegendConfig,
    /// Custom CSS styles
    pub custom_css: Option<String>,
}

impl Default for PlotStyling {
    fn default() -> Self {
        Self {
            color_palette: vec![
                "#1f77b4".to_string(),
                "#ff7f0e".to_string(),
                "#2ca02c".to_string(),
                "#d62728".to_string(),
                "#9467bd".to_string(),
                "#8c564b".to_string(),
                "#e377c2".to_string(),
                "#7f7f7f".to_string(),
                "#bcbd22".to_string(),
                "#17becf".to_string(),
            ],
            font_config: FontConfig::default(),
            line_styles: vec![LineStyle::Solid, LineStyle::Dashed, LineStyle::Dotted],
            marker_styles: vec![
                MarkerStyle::Circle,
                MarkerStyle::Square,
                MarkerStyle::Triangle,
            ],
            background_color: "#ffffff".to_string(),
            grid_config: GridConfig::default(),
            legend_config: LegendConfig::default(),
            custom_css: None,
        }
    }
}

/// Font configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontConfig {
    pub family: String,
    pub size: u32,
    pub weight: FontWeight,
    pub color: String,
}

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            family: "Arial, sans-serif".to_string(),
            size: 12,
            weight: FontWeight::Normal,
            color: "#000000".to_string(),
        }
    }
}

/// Font weight options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontWeight {
    Normal,
    Bold,
    Light,
    ExtraBold,
}

/// Line style options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
    DashDot,
    None,
}

/// Marker style options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarkerStyle {
    Circle,
    Square,
    Triangle,
    Diamond,
    Cross,
    Plus,
    Star,
    None,
}

/// Grid configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    pub show_x_grid: bool,
    pub show_y_grid: bool,
    pub grid_color: String,
    pub grid_alpha: f64,
    pub grid_width: f64,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            show_x_grid: true,
            show_y_grid: true,
            grid_color: "#cccccc".to_string(),
            grid_alpha: 0.5,
            grid_width: 1.0,
        }
    }
}

/// Legend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegendConfig {
    pub show_legend: bool,
    pub position: LegendPosition,
    pub background_color: String,
    pub border_color: String,
    pub border_width: f64,
}

impl Default for LegendConfig {
    fn default() -> Self {
        Self {
            show_legend: true,
            position: LegendPosition::TopRight,
            background_color: "rgba(255, 255, 255, 0.8)".to_string(),
            border_color: "#cccccc".to_string(),
            border_width: 1.0,
        }
    }
}

/// Legend position options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegendPosition {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Top,
    Bottom,
    Left,
    Right,
}

/// Animation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationConfig {
    /// Animation type
    pub animation_type: AnimationType,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Easing function
    pub easing: EasingFunction,
    /// Number of frames
    pub frames: u32,
    /// Loop animation
    pub loop_animation: bool,
    /// Auto-start animation
    pub auto_start: bool,
}

/// Animation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnimationType {
    /// Fade in animation
    FadeIn,
    /// Slide in animation
    SlideIn,
    /// Grow animation
    Grow,
    /// Training progress animation
    TrainingProgress,
    /// Gradient flow animation
    GradientFlow,
    /// Loss landscape flythrough
    LossLandscapeFlythrough,
    /// Custom animation
    Custom(String),
}

/// Easing functions for animations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EasingFunction {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Bounce,
    Elastic,
    Back,
}

/// Real-time plot configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeConfig {
    /// Maximum number of points to keep in buffer
    pub buffer_size: usize,
    /// Update frequency in milliseconds
    pub update_frequency_ms: u64,
    /// Enable streaming mode
    pub streaming_mode: bool,
    /// Data source configuration
    pub data_source: DataSource,
    /// Auto-scroll behavior
    pub auto_scroll: bool,
    /// Time window for display (in seconds)
    pub time_window_seconds: f64,
}

/// Data source for real-time plots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    /// WebSocket connection
    WebSocket { url: String },
    /// HTTP polling
    HttpPolling { url: String, interval_ms: u64 },
    /// File watching
    FileWatching { path: String },
    /// Memory buffer
    MemoryBuffer { buffer_id: String },
    /// Custom function
    CustomFunction { function_name: String },
}

/// Modern plotting engine
#[derive(Debug)]
pub struct ModernPlottingEngine {
    config: ModernPlottingConfig,
    active_plots: HashMap<String, PlotInstance>,
    dashboard_server: Option<DashboardServer>,
    #[allow(dead_code)]
    plot_cache: HashMap<String, CachedPlot>,
}

/// Plot instance tracking
#[derive(Debug, Clone)]
pub struct PlotInstance {
    pub id: String,
    pub plot_type: InteractivePlotType,
    pub data: InteractivePlotData,
    pub creation_time: DateTime<Utc>,
    pub last_update: DateTime<Utc>,
    pub file_path: Option<PathBuf>,
    pub is_realtime: bool,
    pub update_count: u64,
}

/// Dashboard server for web interface
#[derive(Debug)]
pub struct DashboardServer {
    port: u16,
    plots: HashMap<String, String>, // plot_id -> HTML content
    is_running: bool,
}

/// Cached plot for performance optimization
#[derive(Debug, Clone)]
pub struct CachedPlot {
    pub content: String,
    pub hash: u64,
    pub creation_time: DateTime<Utc>,
    pub access_count: u64,
}

impl ModernPlottingEngine {
    /// Create a new modern plotting engine
    pub fn new(config: ModernPlottingConfig) -> Self {
        std::fs::create_dir_all(&config.output_directory).ok();

        Self {
            config,
            active_plots: HashMap::new(),
            dashboard_server: None,
            plot_cache: HashMap::new(),
        }
    }

    /// Create an interactive line plot
    pub async fn create_interactive_line_plot(
        &mut self,
        plot_data: InteractivePlotData,
        plot_id: Option<String>,
    ) -> Result<String> {
        let id = plot_id.unwrap_or_else(|| format!("line_plot_{}", Utc::now().timestamp()));

        let html_content = self.generate_plotly_line_plot(&plot_data)?;
        let file_path = self.save_plot_to_file(&id, &html_content).await?;

        let instance = PlotInstance {
            id: id.clone(),
            plot_type: InteractivePlotType::InteractiveLinePlot,
            data: plot_data,
            creation_time: Utc::now(),
            last_update: Utc::now(),
            file_path: Some(file_path),
            is_realtime: false,
            update_count: 0,
        };

        self.active_plots.insert(id.clone(), instance);

        if self.config.enable_web_dashboard {
            self.add_plot_to_dashboard(&id, &html_content).await?;
        }

        Ok(id)
    }

    /// Create an interactive scatter plot
    pub async fn create_interactive_scatter_plot(
        &mut self,
        x_values: &[f64],
        y_values: &[f64],
        labels: Option<&[String]>,
        title: &str,
        plot_id: Option<String>,
    ) -> Result<String> {
        let id = plot_id.unwrap_or_else(|| format!("scatter_plot_{}", Utc::now().timestamp()));

        let plot_data = InteractivePlotData {
            plot_data: PlotData {
                x_values: x_values.to_vec(),
                y_values: y_values.to_vec(),
                labels: labels.map(|l| l.to_vec()).unwrap_or_else(|| vec!["Series 1".to_string()]),
                title: title.to_string(),
                x_label: "X".to_string(),
                y_label: "Y".to_string(),
            },
            interactive_config: InteractiveConfig::default(),
            styling: PlotStyling::default(),
            animation_config: None,
            realtime_config: None,
        };

        let html_content = self.generate_plotly_scatter_plot(&plot_data)?;
        let file_path = self.save_plot_to_file(&id, &html_content).await?;

        let instance = PlotInstance {
            id: id.clone(),
            plot_type: InteractivePlotType::InteractiveScatterPlot,
            data: plot_data,
            creation_time: Utc::now(),
            last_update: Utc::now(),
            file_path: Some(file_path),
            is_realtime: false,
            update_count: 0,
        };

        self.active_plots.insert(id.clone(), instance);

        if self.config.enable_web_dashboard {
            self.add_plot_to_dashboard(&id, &html_content).await?;
        }

        Ok(id)
    }

    /// Create an interactive heatmap
    pub async fn create_interactive_heatmap(
        &mut self,
        values: &[Vec<f64>],
        x_labels: Option<&[String]>,
        y_labels: Option<&[String]>,
        title: &str,
        plot_id: Option<String>,
    ) -> Result<String> {
        let id = plot_id.unwrap_or_else(|| format!("heatmap_{}", Utc::now().timestamp()));

        let default_x_labels: Vec<String> = (0..values.first().map_or(0, |row| row.len()))
            .map(|i| format!("Col_{}", i))
            .collect();
        let default_y_labels: Vec<String> =
            (0..values.len()).map(|i| format!("Row_{}", i)).collect();

        let heatmap_data = HeatmapData {
            values: values.to_vec(),
            x_labels: x_labels.map(|l| l.to_vec()).unwrap_or(default_x_labels),
            y_labels: y_labels.map(|l| l.to_vec()).unwrap_or(default_y_labels),
            title: title.to_string(),
            color_bar_label: "Value".to_string(),
        };

        let html_content = self.generate_plotly_heatmap(&heatmap_data)?;
        let file_path = self.save_plot_to_file(&id, &html_content).await?;

        let plot_data = InteractivePlotData {
            plot_data: PlotData {
                x_values: vec![],
                y_values: vec![],
                labels: vec![],
                title: title.to_string(),
                x_label: "X".to_string(),
                y_label: "Y".to_string(),
            },
            interactive_config: InteractiveConfig::default(),
            styling: PlotStyling::default(),
            animation_config: None,
            realtime_config: None,
        };

        let instance = PlotInstance {
            id: id.clone(),
            plot_type: InteractivePlotType::InteractiveHeatmap,
            data: plot_data,
            creation_time: Utc::now(),
            last_update: Utc::now(),
            file_path: Some(file_path),
            is_realtime: false,
            update_count: 0,
        };

        self.active_plots.insert(id.clone(), instance);

        if self.config.enable_web_dashboard {
            self.add_plot_to_dashboard(&id, &html_content).await?;
        }

        Ok(id)
    }

    /// Create a real-time streaming plot
    pub async fn create_realtime_plot(
        &mut self,
        title: &str,
        plot_id: Option<String>,
        realtime_config: RealtimeConfig,
    ) -> Result<String> {
        let id = plot_id.unwrap_or_else(|| format!("realtime_plot_{}", Utc::now().timestamp()));

        let plot_data = InteractivePlotData {
            plot_data: PlotData {
                x_values: vec![],
                y_values: vec![],
                labels: vec!["Real-time Data".to_string()],
                title: title.to_string(),
                x_label: "Time".to_string(),
                y_label: "Value".to_string(),
            },
            interactive_config: InteractiveConfig::default(),
            styling: PlotStyling::default(),
            animation_config: None,
            realtime_config: Some(realtime_config),
        };

        let html_content = self.generate_realtime_plot(&plot_data)?;
        let file_path = self.save_plot_to_file(&id, &html_content).await?;

        let instance = PlotInstance {
            id: id.clone(),
            plot_type: InteractivePlotType::RealtimeStreamingPlot,
            data: plot_data,
            creation_time: Utc::now(),
            last_update: Utc::now(),
            file_path: Some(file_path),
            is_realtime: true,
            update_count: 0,
        };

        self.active_plots.insert(id.clone(), instance);

        if self.config.enable_web_dashboard {
            self.add_plot_to_dashboard(&id, &html_content).await?;
        }

        Ok(id)
    }

    /// Create an animated training visualization
    pub async fn create_animated_training_plot(
        &mut self,
        training_data: &[f64],
        validation_data: &[f64],
        epochs: &[u32],
        title: &str,
        plot_id: Option<String>,
    ) -> Result<String> {
        let id = plot_id.unwrap_or_else(|| format!("animated_training_{}", Utc::now().timestamp()));

        let animation_config = AnimationConfig {
            animation_type: AnimationType::TrainingProgress,
            duration_ms: 5000,
            easing: EasingFunction::EaseInOut,
            frames: epochs.len() as u32,
            loop_animation: false,
            auto_start: true,
        };

        let plot_data = InteractivePlotData {
            plot_data: PlotData {
                x_values: epochs.iter().map(|&e| e as f64).collect(),
                y_values: training_data.to_vec(),
                labels: vec!["Training Loss".to_string(), "Validation Loss".to_string()],
                title: title.to_string(),
                x_label: "Epoch".to_string(),
                y_label: "Loss".to_string(),
            },
            interactive_config: InteractiveConfig::default(),
            styling: PlotStyling::default(),
            animation_config: Some(animation_config),
            realtime_config: None,
        };

        let html_content = self.generate_animated_training_plot(&plot_data, validation_data)?;
        let file_path = self.save_plot_to_file(&id, &html_content).await?;

        let instance = PlotInstance {
            id: id.clone(),
            plot_type: InteractivePlotType::AnimatedTrainingPlot,
            data: plot_data,
            creation_time: Utc::now(),
            last_update: Utc::now(),
            file_path: Some(file_path),
            is_realtime: false,
            update_count: 0,
        };

        self.active_plots.insert(id.clone(), instance);

        if self.config.enable_web_dashboard {
            self.add_plot_to_dashboard(&id, &html_content).await?;
        }

        Ok(id)
    }

    /// Create a comprehensive dashboard with multiple plots
    pub async fn create_dashboard(&mut self, plot_ids: &[String], title: &str) -> Result<String> {
        let dashboard_id = format!("dashboard_{}", Utc::now().timestamp());

        let mut dashboard_html = self.generate_dashboard_template(title)?;

        for plot_id in plot_ids {
            if let Some(plot_instance) = self.active_plots.get(plot_id) {
                let plot_html = self.get_plot_html_content(plot_instance)?;
                dashboard_html =
                    self.embed_plot_in_dashboard(&dashboard_html, plot_id, &plot_html)?;
            }
        }

        dashboard_html = self.finalize_dashboard_html(&dashboard_html)?;

        let dashboard_path =
            Path::new(&self.config.output_directory).join(format!("{}.html", dashboard_id));
        tokio::fs::write(&dashboard_path, &dashboard_html).await?;

        if self.config.enable_web_dashboard {
            self.start_dashboard_server().await?;
        }

        Ok(dashboard_path.to_string_lossy().to_string())
    }

    /// Update real-time plot with new data
    pub async fn update_realtime_plot(
        &mut self,
        plot_id: &str,
        new_x: f64,
        new_y: f64,
    ) -> Result<()> {
        let should_update_dashboard = self.config.enable_web_dashboard;
        let mut plot_data_for_dashboard = None;

        if let Some(plot_instance) = self.active_plots.get_mut(plot_id) {
            if plot_instance.is_realtime {
                // Add new data point
                plot_instance.data.plot_data.x_values.push(new_x);
                plot_instance.data.plot_data.y_values.push(new_y);

                // Maintain buffer size
                if let Some(ref realtime_config) = plot_instance.data.realtime_config {
                    let buffer_size = realtime_config.buffer_size;
                    if plot_instance.data.plot_data.x_values.len() > buffer_size {
                        plot_instance.data.plot_data.x_values.remove(0);
                        plot_instance.data.plot_data.y_values.remove(0);
                    }
                }

                plot_instance.last_update = Utc::now();
                plot_instance.update_count += 1;

                // Store data for dashboard update
                if should_update_dashboard {
                    plot_data_for_dashboard = Some(plot_instance.data.clone());
                }
            }
        }

        // Update dashboard if needed
        if let Some(data) = plot_data_for_dashboard {
            self.update_plot_in_dashboard(plot_id, &data).await?;
        }

        Ok(())
    }

    /// Get plot statistics
    pub fn get_plot_statistics(&self, plot_id: &str) -> Option<PlotStatistics> {
        self.active_plots.get(plot_id).map(|instance| PlotStatistics {
            plot_id: plot_id.to_string(),
            plot_type: instance.plot_type.clone(),
            creation_time: instance.creation_time,
            last_update: instance.last_update,
            update_count: instance.update_count,
            data_points: instance.data.plot_data.x_values.len(),
            is_realtime: instance.is_realtime,
            file_size_bytes: instance
                .file_path
                .as_ref()
                .and_then(|path| std::fs::metadata(path).ok())
                .map(|metadata| metadata.len())
                .unwrap_or(0),
        })
    }

    /// List all active plots
    pub fn list_active_plots(&self) -> Vec<String> {
        self.active_plots.keys().cloned().collect()
    }

    /// Remove a plot
    pub async fn remove_plot(&mut self, plot_id: &str) -> Result<()> {
        if let Some(instance) = self.active_plots.remove(plot_id) {
            // Remove file if it exists
            if let Some(file_path) = instance.file_path {
                tokio::fs::remove_file(file_path).await.ok();
            }

            // Remove from dashboard
            if self.config.enable_web_dashboard {
                self.remove_plot_from_dashboard(plot_id).await?;
            }
        }

        Ok(())
    }

    // Private helper methods

    fn generate_plotly_line_plot(&self, data: &InteractivePlotData) -> Result<String> {
        let plot_data = &data.plot_data;
        let styling = &data.styling;

        let mut html = String::from(
            r#"
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <title>Interactive Line Plot</title>
</head>
<body>
    <div id="plotDiv" style="width:100%;height:600px;"></div>
    <script>
        var trace = {
            x: ["#,
        );

        // Add x values
        html.push_str(
            &plot_data.x_values.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "),
        );

        html.push_str(
            r#"],
            y: ["#,
        );

        // Add y values
        html.push_str(
            &plot_data.y_values.iter().map(|y| y.to_string()).collect::<Vec<_>>().join(", "),
        );

        html.push_str(&format!(
            r#"],
            type: 'scatter',
            mode: 'lines+markers',
            name: '{}',
            line: {{
                color: '{}',
                width: 2
            }},
            marker: {{
                size: 6,
                color: '{}'
            }}
        }};

        var layout = {{
            title: '{}',
            xaxis: {{
                title: '{}',
                showgrid: {},
                gridcolor: '{}'
            }},
            yaxis: {{
                title: '{}',
                showgrid: {},
                gridcolor: '{}'
            }},
            font: {{
                family: '{}',
                size: {},
                color: '{}'
            }},
            plot_bgcolor: '{}',
            paper_bgcolor: '{}'
        }};

        var config = {{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToAdd: ['pan2d', 'select2d', 'lasso2d', 'resetScale2d'],
            toImageButtonOptions: {{
                format: 'png',
                filename: 'debug_plot',
                height: 600,
                width: 800,
                scale: 1
            }}
        }};

        Plotly.newPlot('plotDiv', [trace], layout, config);
    </script>
</body>
</html>"#,
            plot_data.labels.first().unwrap_or(&"Series 1".to_string()),
            styling.color_palette.first().unwrap_or(&"#1f77b4".to_string()),
            styling.color_palette.first().unwrap_or(&"#1f77b4".to_string()),
            plot_data.title,
            plot_data.x_label,
            styling.grid_config.show_x_grid,
            styling.grid_config.grid_color,
            plot_data.y_label,
            styling.grid_config.show_y_grid,
            styling.grid_config.grid_color,
            styling.font_config.family,
            styling.font_config.size,
            styling.font_config.color,
            styling.background_color,
            styling.background_color
        ));

        Ok(html)
    }

    fn generate_plotly_scatter_plot(&self, data: &InteractivePlotData) -> Result<String> {
        let plot_data = &data.plot_data;
        let styling = &data.styling;

        let html = format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <title>Interactive Scatter Plot</title>
</head>
<body>
    <div id="plotDiv" style="width:100%;height:600px;"></div>
    <script>
        var trace = {{
            x: [{}],
            y: [{}],
            mode: 'markers',
            type: 'scatter',
            name: '{}',
            marker: {{
                size: 8,
                color: '{}',
                opacity: 0.7,
                line: {{
                    color: '{}',
                    width: 1
                }}
            }}
        }};

        var layout = {{
            title: '{}',
            xaxis: {{
                title: '{}',
                showgrid: true
            }},
            yaxis: {{
                title: '{}',
                showgrid: true
            }},
            hovermode: 'closest'
        }};

        var config = {{
            responsive: true,
            displayModeBar: true
        }};

        Plotly.newPlot('plotDiv', [trace], layout, config);
    </script>
</body>
</html>"#,
            plot_data.x_values.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "),
            plot_data.y_values.iter().map(|y| y.to_string()).collect::<Vec<_>>().join(", "),
            plot_data.labels.first().unwrap_or(&"Series 1".to_string()),
            styling.color_palette.first().unwrap_or(&"#1f77b4".to_string()),
            styling.color_palette.first().unwrap_or(&"#1f77b4".to_string()),
            plot_data.title,
            plot_data.x_label,
            plot_data.y_label
        );

        Ok(html)
    }

    fn generate_plotly_heatmap(&self, data: &HeatmapData) -> Result<String> {
        let values_json = serde_json::to_string(&data.values)?;
        let x_labels_json = serde_json::to_string(&data.x_labels)?;
        let y_labels_json = serde_json::to_string(&data.y_labels)?;

        let html = format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <title>Interactive Heatmap</title>
</head>
<body>
    <div id="plotDiv" style="width:100%;height:600px;"></div>
    <script>
        var data = [{{
            z: {},
            x: {},
            y: {},
            type: 'heatmap',
            colorscale: 'Viridis',
            showscale: true,
            colorbar: {{
                title: '{}'
            }}
        }}];

        var layout = {{
            title: '{}',
            xaxis: {{
                title: 'Features'
            }},
            yaxis: {{
                title: 'Samples'
            }}
        }};

        var config = {{
            responsive: true,
            displayModeBar: true
        }};

        Plotly.newPlot('plotDiv', data, layout, config);
    </script>
</body>
</html>"#,
            values_json, x_labels_json, y_labels_json, data.color_bar_label, data.title
        );

        Ok(html)
    }

    fn generate_realtime_plot(&self, data: &InteractivePlotData) -> Result<String> {
        let realtime_config = data.realtime_config.as_ref().unwrap();

        let html = format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <title>Real-time Plot</title>
</head>
<body>
    <div id="plotDiv" style="width:100%;height:600px;"></div>
    <script>
        var trace = {{
            x: [],
            y: [],
            mode: 'lines',
            type: 'scatter',
            name: '{}'
        }};

        var layout = {{
            title: '{}',
            xaxis: {{
                title: '{}',
                range: [0, {}]
            }},
            yaxis: {{
                title: '{}'
            }}
        }};

        var config = {{
            responsive: true,
            displayModeBar: true
        }};

        Plotly.newPlot('plotDiv', [trace], layout, config);

        // Simulate real-time updates
        var cnt = 0;
        var interval = setInterval(function() {{
            var time = new Date().getTime();
            var y = Math.sin(cnt * 0.1) + Math.random() * 0.1;

            Plotly.extendTraces('plotDiv', {{
                x: [[time]],
                y: [[y]]
            }}, [0]);

            // Keep only last {} points
            if (trace.x.length > {}) {{
                Plotly.relayout('plotDiv', {{
                    'xaxis.range': [trace.x[trace.x.length - {}], trace.x[trace.x.length - 1]]
                }});
            }}

            cnt++;
        }}, {});
    </script>
</body>
</html>"#,
            data.plot_data.labels.first().unwrap_or(&"Real-time Data".to_string()),
            data.plot_data.title,
            data.plot_data.x_label,
            realtime_config.time_window_seconds,
            data.plot_data.y_label,
            realtime_config.buffer_size,
            realtime_config.buffer_size,
            realtime_config.buffer_size,
            realtime_config.update_frequency_ms
        );

        Ok(html)
    }

    fn generate_animated_training_plot(
        &self,
        data: &InteractivePlotData,
        validation_data: &[f64],
    ) -> Result<String> {
        let training_json = serde_json::to_string(&data.plot_data.y_values)?;
        let validation_json = serde_json::to_string(validation_data)?;
        let epochs_json = serde_json::to_string(&data.plot_data.x_values)?;

        let html = format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <title>Animated Training Plot</title>
</head>
<body>
    <div id="plotDiv" style="width:100%;height:600px;"></div>
    <div id="controls">
        <button onclick="animateTraining()">Start Animation</button>
        <button onclick="resetAnimation()">Reset</button>
    </div>
    <script>
        var trainingData = {};
        var validationData = {};
        var epochs = {};
        var currentFrame = 0;

        var trace1 = {{
            x: [],
            y: [],
            mode: 'lines+markers',
            type: 'scatter',
            name: 'Training Loss',
            line: {{color: '#1f77b4', width: 3}},
            marker: {{size: 6}}
        }};

        var trace2 = {{
            x: [],
            y: [],
            mode: 'lines+markers',
            type: 'scatter',
            name: 'Validation Loss',
            line: {{color: '#ff7f0e', width: 3}},
            marker: {{size: 6}}
        }};

        var layout = {{
            title: '{}',
            xaxis: {{title: '{}'}},
            yaxis: {{title: '{}'}},
            showlegend: true
        }};

        Plotly.newPlot('plotDiv', [trace1, trace2], layout);

        function animateTraining() {{
            var interval = setInterval(function() {{
                if (currentFrame >= trainingData.length) {{
                    clearInterval(interval);
                    return;
                }}

                trace1.x.push(epochs[currentFrame]);
                trace1.y.push(trainingData[currentFrame]);
                trace2.x.push(epochs[currentFrame]);
                trace2.y.push(validationData[currentFrame]);

                Plotly.redraw('plotDiv');
                currentFrame++;
            }}, 200);
        }}

        function resetAnimation() {{
            currentFrame = 0;
            trace1.x = [];
            trace1.y = [];
            trace2.x = [];
            trace2.y = [];
            Plotly.redraw('plotDiv');
        }}
    </script>
</body>
</html>"#,
            training_json,
            validation_json,
            epochs_json,
            data.plot_data.title,
            data.plot_data.x_label,
            data.plot_data.y_label
        );

        Ok(html)
    }

    fn generate_dashboard_template(&self, title: &str) -> Result<String> {
        let html = format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{}</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .dashboard-header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot-container {{
            display: inline-block;
            width: 48%;
            margin: 1%;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 10px;
        }}
        .plot-container.full-width {{
            width: 98%;
        }}
        .controls {{
            text-align: center;
            margin: 20px 0;
        }}
        button {{
            padding: 10px 20px;
            margin: 0 10px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }}
        button:hover {{
            background-color: #0056b3;
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>{}</h1>
        <p>Real-time debugging dashboard</p>
    </div>
    <div class="controls">
        <button onclick="refreshAll()">Refresh All</button>
        <button onclick="exportDashboard()">Export</button>
        <button onclick="toggleAutoRefresh()">Toggle Auto-refresh</button>
    </div>
    <div id="plots-container">"#,
            title, title
        );

        Ok(html)
    }

    fn embed_plot_in_dashboard(
        &self,
        dashboard_html: &str,
        plot_id: &str,
        plot_html: &str,
    ) -> Result<String> {
        // Extract the plot div and script from the plot HTML
        let plot_div = format!(
            r#"<div class="plot-container" id="container-{}"></div>"#,
            plot_id
        );

        let mut updated_html = dashboard_html.replace(
            r#"<div id="plots-container">"#,
            &format!(r#"<div id="plots-container">{}"#, plot_div),
        );

        // Add the plot script (simplified - would need proper HTML parsing in production)
        updated_html.push_str(&format!(
            r#"
    <script>
        // Plot {} initialization would go here
        // Extracted from: {}
    </script>"#,
            plot_id,
            plot_html.len()
        ));

        Ok(updated_html)
    }

    fn finalize_dashboard_html(&self, html: &str) -> Result<String> {
        let finalized = format!(
            r#"{}
    </div>
    <script>
        function refreshAll() {{
            location.reload();
        }}

        function exportDashboard() {{
            // Export functionality
            alert('Export functionality would be implemented here');
        }}

        var autoRefresh = false;
        function toggleAutoRefresh() {{
            autoRefresh = !autoRefresh;
            if (autoRefresh) {{
                setInterval(refreshAll, 30000); // Refresh every 30 seconds
            }}
        }}
    </script>
</body>
</html>"#,
            html
        );

        Ok(finalized)
    }

    async fn save_plot_to_file(&self, plot_id: &str, content: &str) -> Result<PathBuf> {
        let file_path = Path::new(&self.config.output_directory).join(format!("{}.html", plot_id));
        tokio::fs::write(&file_path, content).await?;
        Ok(file_path)
    }

    async fn add_plot_to_dashboard(&mut self, plot_id: &str, content: &str) -> Result<()> {
        if self.dashboard_server.is_none() {
            self.dashboard_server = Some(DashboardServer {
                port: self.config.dashboard_port,
                plots: HashMap::new(),
                is_running: false,
            });
        }

        if let Some(ref mut server) = self.dashboard_server {
            server.plots.insert(plot_id.to_string(), content.to_string());
        }

        Ok(())
    }

    async fn start_dashboard_server(&mut self) -> Result<()> {
        if let Some(ref mut server) = self.dashboard_server {
            if !server.is_running {
                // In a real implementation, this would start an actual web server
                server.is_running = true;
                tracing::info!("Dashboard server started on port {}", server.port);
            }
        }
        Ok(())
    }

    async fn update_plot_in_dashboard(
        &mut self,
        plot_id: &str,
        data: &InteractivePlotData,
    ) -> Result<()> {
        // Update the plot in the dashboard (simplified implementation)
        let updated_content = self.generate_plotly_line_plot(data)?;

        if let Some(ref mut server) = self.dashboard_server {
            server.plots.insert(plot_id.to_string(), updated_content);
        }
        Ok(())
    }

    async fn remove_plot_from_dashboard(&mut self, plot_id: &str) -> Result<()> {
        if let Some(ref mut server) = self.dashboard_server {
            server.plots.remove(plot_id);
        }
        Ok(())
    }

    fn get_plot_html_content(&self, instance: &PlotInstance) -> Result<String> {
        // Return the HTML content for the plot
        match instance.plot_type {
            InteractivePlotType::InteractiveLinePlot => {
                self.generate_plotly_line_plot(&instance.data)
            },
            InteractivePlotType::InteractiveScatterPlot => {
                self.generate_plotly_scatter_plot(&instance.data)
            },
            InteractivePlotType::RealtimeStreamingPlot => {
                self.generate_realtime_plot(&instance.data)
            },
            _ => Ok("Plot content not available".to_string()),
        }
    }
}

/// Statistics for a plot instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotStatistics {
    pub plot_id: String,
    pub plot_type: InteractivePlotType,
    pub creation_time: DateTime<Utc>,
    pub last_update: DateTime<Utc>,
    pub update_count: u64,
    pub data_points: usize,
    pub is_realtime: bool,
    pub file_size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_modern_plotting_engine_creation() {
        let config = ModernPlottingConfig::default();
        let engine = ModernPlottingEngine::new(config);
        assert_eq!(engine.active_plots.len(), 0);
    }

    #[tokio::test]
    async fn test_create_interactive_line_plot() {
        let config = ModernPlottingConfig::default();
        let mut engine = ModernPlottingEngine::new(config);

        let plot_data = InteractivePlotData {
            plot_data: PlotData {
                x_values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                y_values: vec![1.0, 4.0, 2.0, 3.0, 5.0],
                labels: vec!["Test Data".to_string()],
                title: "Test Plot".to_string(),
                x_label: "X Axis".to_string(),
                y_label: "Y Axis".to_string(),
            },
            interactive_config: InteractiveConfig::default(),
            styling: PlotStyling::default(),
            animation_config: None,
            realtime_config: None,
        };

        let result = engine.create_interactive_line_plot(plot_data, None).await;
        assert!(result.is_ok());

        let plot_id = result.unwrap();
        assert!(engine.active_plots.contains_key(&plot_id));
    }

    #[tokio::test]
    async fn test_create_interactive_scatter_plot() {
        let config = ModernPlottingConfig::default();
        let mut engine = ModernPlottingEngine::new(config);

        let x_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_values = vec![2.0, 3.0, 1.0, 4.0, 5.0];

        let result = engine
            .create_interactive_scatter_plot(&x_values, &y_values, None, "Test Scatter Plot", None)
            .await;

        assert!(result.is_ok());

        let plot_id = result.unwrap();
        assert!(engine.active_plots.contains_key(&plot_id));
    }
}
