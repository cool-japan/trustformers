//! Basic visualization types, enums, and data structures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Plot type for visualization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlotType {
    Line,
    Scatter,
    Bar,
    Histogram,
    Heatmap,
    ThreeDimensional,
}

/// Configuration for visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    pub output_directory: String,
    pub image_format: ImageFormat,
    pub plot_width: u32,
    pub plot_height: u32,
    pub font_size: u32,
    pub color_scheme: ColorScheme,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            output_directory: "./debug_plots".to_string(),
            image_format: ImageFormat::PNG,
            plot_width: 800,
            plot_height: 600,
            font_size: 12,
            color_scheme: ColorScheme::Default,
        }
    }
}

/// Image format options for visualization output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageFormat {
    PNG,
    SVG,
    PDF,
    HTML,
    LaTeX,
    JSON,
    /// MP4 video format for animated visualizations
    MP4,
    /// GIF format for animated visualizations
    GIF,
    /// WebM video format for web-compatible animations
    WebM,
}

/// Color scheme options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorScheme {
    Default,
    Dark,
    Colorblind,
    Viridis,
    Plasma,
}

/// Visualization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationType {
    /// Line plot for time series data
    LinePlot,
    /// Histogram for distribution analysis
    Histogram,
    /// Heatmap for 2D tensor visualization
    Heatmap,
    /// Scatter plot for correlation analysis
    ScatterPlot,
    /// Box plot for statistical summaries
    BoxPlot,
    /// 3D surface plot for advanced visualization
    SurfacePlot,
    /// 3D loss landscape visualization
    LossLandscape,
    /// 3D optimization trajectory
    OptimizationTrajectory,
    /// 3D weight space exploration
    WeightSpaceExploration,
    /// 3D embedding projections
    EmbeddingProjection,
    /// Architecture diagram
    ArchitectureDiagram,
}

/// Plot data structure for 2D plots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotData {
    pub x_values: Vec<f64>,
    pub y_values: Vec<f64>,
    pub labels: Vec<String>,
    pub title: String,
    pub x_label: String,
    pub y_label: String,
}

/// 3D Plot data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plot3DData {
    pub x_values: Vec<f64>,
    pub y_values: Vec<f64>,
    pub z_values: Vec<f64>,
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub z_label: String,
    pub point_labels: Vec<String>,
    pub color_values: Option<Vec<f64>>,
    pub size_values: Option<Vec<f64>>,
}

/// Architecture node for diagram visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureNode {
    pub id: String,
    pub name: String,
    pub node_type: String,
    pub position: (f64, f64, f64),
    pub size: (f64, f64, f64),
    pub color: String,
    pub metadata: HashMap<String, String>,
}

/// Architecture connection for diagram visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConnection {
    pub from_node: String,
    pub to_node: String,
    pub connection_type: String,
    pub weight: f64,
    pub color: String,
    pub style: ConnectionStyle,
}

/// Connection style for architecture diagrams
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionStyle {
    Solid,
    Dashed,
    Dotted,
    Thick,
    Arrow,
}

/// Heatmap data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapData {
    pub values: Vec<Vec<f64>>,
    pub x_labels: Vec<String>,
    pub y_labels: Vec<String>,
    pub title: String,
    pub color_bar_label: String,
}

/// Histogram data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramData {
    pub values: Vec<f64>,
    pub bins: usize,
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub density: bool,
}
