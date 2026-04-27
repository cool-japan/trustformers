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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_visualization_config_default() {
        let cfg = VisualizationConfig::default();
        assert_eq!(cfg.plot_width, 800);
        assert_eq!(cfg.plot_height, 600);
        assert_eq!(cfg.font_size, 12);
    }

    #[test]
    fn test_visualization_config_clone() {
        let cfg = VisualizationConfig::default();
        let cloned = cfg.clone();
        assert_eq!(cloned.plot_width, cfg.plot_width);
    }

    #[test]
    fn test_plot_type_equality() {
        assert_eq!(PlotType::Line, PlotType::Line);
        assert_ne!(PlotType::Line, PlotType::Bar);
    }

    #[test]
    fn test_plot_type_all_variants() {
        let _ = [
            PlotType::Line,
            PlotType::Scatter,
            PlotType::Bar,
            PlotType::Histogram,
            PlotType::Heatmap,
            PlotType::ThreeDimensional,
        ];
    }

    #[test]
    fn test_plot_data_creation() {
        let data = PlotData {
            x_values: vec![1.0, 2.0, 3.0],
            y_values: vec![10.0, 20.0, 30.0],
            labels: vec!["s1".to_string()],
            title: "Test".to_string(),
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
        };
        assert_eq!(data.x_values.len(), 3);
    }

    #[test]
    fn test_heatmap_data_creation() {
        let hm = HeatmapData {
            values: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            x_labels: vec!["a".to_string(), "b".to_string()],
            y_labels: vec!["r1".to_string(), "r2".to_string()],
            title: "HM".to_string(),
            color_bar_label: "val".to_string(),
        };
        assert_eq!(hm.values.len(), 2);
    }

    #[test]
    fn test_histogram_data_creation() {
        let hd = HistogramData {
            values: vec![1.0, 2.0, 3.0],
            bins: 10,
            title: "Dist".to_string(),
            x_label: "Value".to_string(),
            y_label: "Count".to_string(),
            density: false,
        };
        assert_eq!(hd.bins, 10);
    }

    #[test]
    fn test_architecture_node_creation() {
        let node = ArchitectureNode {
            id: "n1".to_string(),
            name: "Linear".to_string(),
            node_type: "layer".to_string(),
            position: (0.0, 0.0, 0.0),
            size: (1.0, 1.0, 1.0),
            color: "#ff0000".to_string(),
            metadata: HashMap::new(),
        };
        assert_eq!(node.id, "n1");
    }

    #[test]
    fn test_architecture_connection_creation() {
        let conn = ArchitectureConnection {
            from_node: "n1".to_string(),
            to_node: "n2".to_string(),
            connection_type: "forward".to_string(),
            weight: 1.0,
            color: "#00ff00".to_string(),
            style: ConnectionStyle::Solid,
        };
        assert_eq!(conn.from_node, "n1");
    }

    #[test]
    fn test_connection_style_variants() {
        let _ = ConnectionStyle::Solid;
        let _ = ConnectionStyle::Dashed;
        let _ = ConnectionStyle::Arrow;
    }

    #[test]
    fn test_image_format_variants() {
        let _ = [
            ImageFormat::PNG,
            ImageFormat::SVG,
            ImageFormat::HTML,
            ImageFormat::MP4,
            ImageFormat::GIF,
            ImageFormat::WebM,
        ];
    }

    #[test]
    fn test_color_scheme_variants() {
        let _ = [
            ColorScheme::Default,
            ColorScheme::Dark,
            ColorScheme::Viridis,
            ColorScheme::Plasma,
        ];
    }

    #[test]
    fn test_plot_3d_data_creation() {
        let data = Plot3DData {
            x_values: vec![0.0, 1.0],
            y_values: vec![0.0, 1.0],
            z_values: vec![0.0, 2.0],
            title: "3D".to_string(),
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
            z_label: "Z".to_string(),
            point_labels: vec!["p1".to_string()],
            color_values: None,
            size_values: None,
        };
        assert_eq!(data.x_values.len(), 2);
        assert!(data.color_values.is_none());
    }
}
