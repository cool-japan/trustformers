//! Tensor visualization utilities for TrustformeRS
//!
//! This module provides various visualization tools for tensors, helping with debugging,
//! model analysis, and development. It supports different output formats including ASCII art,
//! HTML, and export to visualization libraries.

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use scirs2_core::ArrayD;
use serde::{Deserialize, Serialize};
use std::fmt::Write;

/// Configuration for tensor visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Maximum number of elements to display in each dimension
    pub max_display_elements: usize,
    /// Color scheme for heatmaps
    pub color_scheme: ColorScheme,
    /// Precision for floating point display
    pub precision: usize,
    /// Whether to show tensor metadata
    pub show_metadata: bool,
    /// Whether to show statistics
    pub show_statistics: bool,
    /// Format for tensor output
    pub output_format: OutputFormat,
    /// Threshold for highlighting values
    pub highlight_threshold: Option<f32>,
}

/// Color schemes for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorScheme {
    /// Grayscale colors
    Grayscale,
    /// Blue to red color map
    BlueRed,
    /// Viridis color map
    Viridis,
    /// Plasma color map
    Plasma,
    /// Custom RGB values
    Custom,
}

/// Output formats for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Plain ASCII text
    ASCII,
    /// Unicode with box drawing characters
    Unicode,
    /// HTML with inline CSS
    HTML,
    /// JSON format for programmatic use
    JSON,
    /// SVG format for scalable graphics
    SVG,
}

/// Tensor statistics for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorStats {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub element_count: usize,
    pub memory_usage: usize,
    pub min_value: f32,
    pub max_value: f32,
    pub mean_value: f32,
    pub std_value: f32,
    pub zeros_count: usize,
    pub nans_count: usize,
    pub infs_count: usize,
}

/// 2D tensor visualization as heatmap
#[derive(Debug, Clone)]
pub struct TensorHeatmap {
    pub data: Vec<Vec<f32>>,
    pub width: usize,
    pub height: usize,
    pub min_value: f32,
    pub max_value: f32,
    pub color_scheme: ColorScheme,
}

/// 1D tensor visualization as histogram
#[derive(Debug, Clone)]
pub struct TensorHistogram {
    pub bins: Vec<f32>,
    pub counts: Vec<usize>,
    pub bin_edges: Vec<f32>,
    pub total_count: usize,
}

/// Multi-dimensional tensor slice visualization
#[derive(Debug, Clone)]
pub struct TensorSliceView {
    pub slices: Vec<TensorHeatmap>,
    pub slice_indices: Vec<Vec<usize>>,
    pub original_shape: Vec<usize>,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            max_display_elements: 20,
            color_scheme: ColorScheme::Viridis,
            precision: 4,
            show_metadata: true,
            show_statistics: true,
            output_format: OutputFormat::Unicode,
            highlight_threshold: None,
        }
    }
}

/// Main tensor visualizer
pub struct TensorVisualizer {
    config: VisualizationConfig,
}

impl Default for TensorVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorVisualizer {
    /// Create a new tensor visualizer with default configuration
    pub fn new() -> Self {
        Self {
            config: VisualizationConfig::default(),
        }
    }

    /// Create a new tensor visualizer with custom configuration
    pub fn with_config(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Visualize a tensor as text representation
    pub fn visualize_tensor(&self, tensor: &Tensor) -> Result<String> {
        match self.config.output_format {
            OutputFormat::ASCII | OutputFormat::Unicode => self.visualize_as_text(tensor),
            OutputFormat::HTML => self.visualize_as_html(tensor),
            OutputFormat::JSON => self.visualize_as_json(tensor),
            OutputFormat::SVG => self.visualize_as_svg(tensor),
        }
    }

    /// Generate tensor statistics
    pub fn compute_statistics(&self, tensor: &Tensor) -> Result<TensorStats> {
        match tensor {
            Tensor::F32(arr) => {
                let data: Vec<f32> = arr.iter().cloned().collect();
                let shape = arr.shape().to_vec();
                let element_count = data.len();

                if data.is_empty() {
                    return Ok(TensorStats {
                        shape,
                        dtype: "f32".to_string(),
                        device: tensor.device(),
                        element_count: 0,
                        memory_usage: 0,
                        min_value: 0.0,
                        max_value: 0.0,
                        mean_value: 0.0,
                        std_value: 0.0,
                        zeros_count: 0,
                        nans_count: 0,
                        infs_count: 0,
                    });
                }

                let finite_values: Vec<f32> =
                    data.iter().filter(|&&x| x.is_finite()).cloned().collect();

                let min_value = finite_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_value = finite_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                let mean_value = if finite_values.is_empty() {
                    0.0
                } else {
                    finite_values.iter().sum::<f32>() / finite_values.len() as f32
                };

                let std_value = if finite_values.len() <= 1 {
                    0.0
                } else {
                    let variance =
                        finite_values.iter().map(|&x| (x - mean_value).powi(2)).sum::<f32>()
                            / (finite_values.len() - 1) as f32;
                    variance.sqrt()
                };

                let zeros_count = data.iter().filter(|&&x| x == 0.0).count();
                let nans_count = data.iter().filter(|&&x| x.is_nan()).count();
                let infs_count = data.iter().filter(|&&x| x.is_infinite()).count();

                Ok(TensorStats {
                    shape,
                    dtype: "f32".to_string(),
                    device: tensor.device(),
                    element_count,
                    memory_usage: element_count * 4, // 4 bytes per f32
                    min_value,
                    max_value,
                    mean_value,
                    std_value,
                    zeros_count,
                    nans_count,
                    infs_count,
                })
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Statistics computation only supported for F32 tensors",
                "compute_statistics",
            )),
        }
    }

    /// Create a heatmap visualization for 2D tensors
    pub fn create_heatmap(&self, tensor: &Tensor) -> Result<TensorHeatmap> {
        match tensor {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                if shape.len() != 2 {
                    return Err(TrustformersError::tensor_op_error(
                        "Heatmap visualization requires 2D tensors",
                        "create_heatmap",
                    ));
                }

                let height = shape[0];
                let width = shape[1];
                let mut data = Vec::with_capacity(height);

                for i in 0..height {
                    let mut row = Vec::with_capacity(width);
                    for j in 0..width {
                        let value = arr[[i, j]];
                        row.push(value);
                    }
                    data.push(row);
                }

                let flat_data: Vec<f32> = data.iter().flatten().cloned().collect();
                let min_value = flat_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_value = flat_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                Ok(TensorHeatmap {
                    data,
                    width,
                    height,
                    min_value,
                    max_value,
                    color_scheme: self.config.color_scheme,
                })
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Heatmap only supported for F32 tensors",
                "create_heatmap",
            )),
        }
    }

    /// Create a histogram for 1D tensor or flattened tensor
    pub fn create_histogram(&self, tensor: &Tensor, num_bins: usize) -> Result<TensorHistogram> {
        match tensor {
            Tensor::F32(arr) => {
                let data: Vec<f32> = arr.iter().filter(|&&x| x.is_finite()).cloned().collect();

                if data.is_empty() {
                    return Ok(TensorHistogram {
                        bins: vec![0.0; num_bins],
                        counts: vec![0; num_bins],
                        bin_edges: vec![0.0; num_bins + 1],
                        total_count: 0,
                    });
                }

                let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                if min_val == max_val {
                    let mut counts = vec![0; num_bins];
                    counts[0] = data.len();

                    return Ok(TensorHistogram {
                        bins: vec![min_val; num_bins],
                        counts,
                        bin_edges: (0..=num_bins).map(|i| min_val + i as f32 * 0.1).collect(),
                        total_count: data.len(),
                    });
                }

                let bin_width = (max_val - min_val) / num_bins as f32;
                let mut counts = vec![0; num_bins];
                let mut bins = Vec::with_capacity(num_bins);
                let mut bin_edges = Vec::with_capacity(num_bins + 1);

                // Create bin edges and centers
                for i in 0..=num_bins {
                    bin_edges.push(min_val + i as f32 * bin_width);
                }

                for i in 0..num_bins {
                    bins.push(min_val + (i as f32 + 0.5) * bin_width);
                }

                // Count values in each bin
                for &value in &data {
                    let bin_index = ((value - min_val) / bin_width) as usize;
                    let bin_index = bin_index.min(num_bins - 1);
                    counts[bin_index] += 1;
                }

                Ok(TensorHistogram {
                    bins,
                    counts,
                    bin_edges,
                    total_count: data.len(),
                })
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Histogram only supported for F32 tensors",
                "create_histogram",
            )),
        }
    }

    /// Create slice view for multi-dimensional tensors
    pub fn create_slice_view(&self, tensor: &Tensor, max_slices: usize) -> Result<TensorSliceView> {
        let shape = tensor.shape();

        if shape.len() < 3 {
            return Err(TrustformersError::tensor_op_error(
                "Slice view requires tensors with 3 or more dimensions",
                "create_slice_view",
            ));
        }

        let mut slices = Vec::new();
        let mut slice_indices = Vec::new();

        // For now, create slices along the first dimension
        let num_slices = shape[0].min(max_slices);
        let step = if shape[0] <= max_slices { 1 } else { shape[0] / max_slices };

        for i in (0..shape[0]).step_by(step).take(num_slices) {
            // Create a 2D slice by taking the first two remaining dimensions
            if let Ok(slice_tensor) = self.extract_2d_slice(tensor, i) {
                if let Ok(heatmap) = self.create_heatmap(&slice_tensor) {
                    slices.push(heatmap);
                    slice_indices.push(vec![i]);
                }
            }
        }

        Ok(TensorSliceView {
            slices,
            slice_indices,
            original_shape: shape,
        })
    }

    /// Visualize tensor as text (ASCII/Unicode)
    fn visualize_as_text(&self, tensor: &Tensor) -> Result<String> {
        let mut output = String::new();

        if self.config.show_metadata {
            writeln!(output, "Tensor Information:")?;
            writeln!(output, "Shape: {:?}", tensor.shape())?;
            writeln!(output, "Device: {}", tensor.device())?;
            writeln!(output, "Memory: {} bytes", tensor.memory_usage())?;
            writeln!(output)?;
        }

        if self.config.show_statistics {
            let stats = self.compute_statistics(tensor)?;
            writeln!(output, "Statistics:")?;
            writeln!(
                output,
                "  Min: {:.precision$}",
                stats.min_value,
                precision = self.config.precision
            )?;
            writeln!(
                output,
                "  Max: {:.precision$}",
                stats.max_value,
                precision = self.config.precision
            )?;
            writeln!(
                output,
                "  Mean: {:.precision$}",
                stats.mean_value,
                precision = self.config.precision
            )?;
            writeln!(
                output,
                "  Std: {:.precision$}",
                stats.std_value,
                precision = self.config.precision
            )?;
            writeln!(output, "  Zeros: {}", stats.zeros_count)?;
            writeln!(output, "  NaNs: {}", stats.nans_count)?;
            writeln!(output, "  Infs: {}", stats.infs_count)?;
            writeln!(output)?;
        }

        writeln!(output, "Data:")?;
        let data_repr = self.format_tensor_data(tensor)?;
        write!(output, "{}", data_repr)?;

        Ok(output)
    }

    /// Visualize tensor as HTML
    fn visualize_as_html(&self, tensor: &Tensor) -> Result<String> {
        let mut html = String::new();

        html.push_str("<div class='tensor-visualization'>\n");
        html.push_str("<style>\n");
        html.push_str("  .tensor-visualization { font-family: monospace; margin: 10px; }\n");
        html.push_str(
            "  .tensor-metadata { background: #f5f5f5; padding: 10px; margin-bottom: 10px; }\n",
        );
        html.push_str(
            "  .tensor-stats { background: #e8f4fd; padding: 10px; margin-bottom: 10px; }\n",
        );
        html.push_str(
            "  .tensor-data { background: #fff; border: 1px solid #ddd; padding: 10px; }\n",
        );
        html.push_str("  .tensor-heatmap { display: inline-block; margin: 2px; padding: 4px; text-align: center; }\n");
        html.push_str("</style>\n");

        if self.config.show_metadata {
            html.push_str("<div class='tensor-metadata'>\n");
            html.push_str("<h3>Tensor Information</h3>\n");
            html.push_str(&format!(
                "<p><strong>Shape:</strong> {:?}</p>\n",
                tensor.shape()
            ));
            html.push_str(&format!(
                "<p><strong>Device:</strong> {}</p>\n",
                tensor.device()
            ));
            html.push_str(&format!(
                "<p><strong>Memory:</strong> {} bytes</p>\n",
                tensor.memory_usage()
            ));
            html.push_str("</div>\n");
        }

        if self.config.show_statistics {
            let stats = self.compute_statistics(tensor)?;
            html.push_str("<div class='tensor-stats'>\n");
            html.push_str("<h3>Statistics</h3>\n");
            html.push_str(&format!(
                "<p><strong>Min:</strong> {:.precision$}</p>\n",
                stats.min_value,
                precision = self.config.precision
            ));
            html.push_str(&format!(
                "<p><strong>Max:</strong> {:.precision$}</p>\n",
                stats.max_value,
                precision = self.config.precision
            ));
            html.push_str(&format!(
                "<p><strong>Mean:</strong> {:.precision$}</p>\n",
                stats.mean_value,
                precision = self.config.precision
            ));
            html.push_str(&format!(
                "<p><strong>Std:</strong> {:.precision$}</p>\n",
                stats.std_value,
                precision = self.config.precision
            ));
            html.push_str("</div>\n");
        }

        html.push_str("<div class='tensor-data'>\n");
        html.push_str("<h3>Data</h3>\n");

        // For 2D tensors, create a visual heatmap
        if tensor.shape().len() == 2 {
            let heatmap = self.create_heatmap(tensor)?;
            html.push_str(&self.heatmap_to_html(&heatmap)?);
        } else {
            let data_repr = self.format_tensor_data(tensor)?;
            html.push_str(&format!("<pre>{}</pre>\n", data_repr));
        }

        html.push_str("</div>\n");
        html.push_str("</div>\n");

        Ok(html)
    }

    /// Visualize tensor as JSON
    fn visualize_as_json(&self, tensor: &Tensor) -> Result<String> {
        let stats = self.compute_statistics(tensor)?;
        let json_data = serde_json::json!({
            "shape": tensor.shape(),
            "device": tensor.device(),
            "memory_usage": tensor.memory_usage(),
            "statistics": stats,
            "data": self.tensor_to_nested_vec(tensor)?,
        });

        Ok(serde_json::to_string_pretty(&json_data)?)
    }

    /// Visualize tensor as SVG
    fn visualize_as_svg(&self, tensor: &Tensor) -> Result<String> {
        if tensor.shape().len() != 2 {
            return Err(TrustformersError::tensor_op_error(
                "SVG visualization currently only supports 2D tensors",
                "visualize_as_svg",
            ));
        }

        let heatmap = self.create_heatmap(tensor)?;
        self.heatmap_to_svg(&heatmap)
    }

    /// Convert heatmap to HTML table with colors
    fn heatmap_to_html(&self, heatmap: &TensorHeatmap) -> Result<String> {
        let mut html = String::new();
        html.push_str("<table style='border-collapse: collapse;'>\n");

        for row in &heatmap.data {
            html.push_str("<tr>\n");
            for &value in row {
                let color = self.value_to_color(value, heatmap.min_value, heatmap.max_value);
                html.push_str(&format!(
                    "<td class='tensor-heatmap' style='background-color: {}; color: {}; min-width: 40px;'>{:.precision$}</td>\n",
                    color.background,
                    color.text,
                    value,
                    precision = self.config.precision
                ));
            }
            html.push_str("</tr>\n");
        }

        html.push_str("</table>\n");
        Ok(html)
    }

    /// Convert heatmap to SVG
    fn heatmap_to_svg(&self, heatmap: &TensorHeatmap) -> Result<String> {
        let cell_size = 20;
        let width = heatmap.width * cell_size;
        let height = heatmap.height * cell_size;

        let mut svg = String::new();
        svg.push_str(&format!(
            "<svg width='{}' height='{}' xmlns='http://www.w3.org/2000/svg'>\n",
            width, height
        ));

        for (i, row) in heatmap.data.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                let color = self.value_to_color(value, heatmap.min_value, heatmap.max_value);
                let x = j * cell_size;
                let y = i * cell_size;

                svg.push_str(&format!(
                    "<rect x='{}' y='{}' width='{}' height='{}' fill='{}' stroke='#ddd' stroke-width='0.5'/>\n",
                    x, y, cell_size, cell_size, color.background
                ));

                // Add text if cell is large enough
                if cell_size >= 30 {
                    svg.push_str(&format!(
                        "<text x='{}' y='{}' text-anchor='middle' dominant-baseline='middle' font-size='10' fill='{}'>{:.1}</text>\n",
                        x + cell_size / 2, y + cell_size / 2, color.text, value
                    ));
                }
            }
        }

        svg.push_str("</svg>\n");
        Ok(svg)
    }

    /// Convert value to color based on color scheme
    fn value_to_color(&self, value: f32, min_val: f32, max_val: f32) -> Color {
        let normalized =
            if max_val > min_val { (value - min_val) / (max_val - min_val) } else { 0.5 };

        let normalized = normalized.clamp(0.0, 1.0);

        match self.config.color_scheme {
            ColorScheme::Grayscale => {
                let intensity = (255.0 * normalized) as u8;
                Color {
                    background: format!("rgb({},{},{})", intensity, intensity, intensity),
                    text: if intensity > 127 { "#000".to_string() } else { "#fff".to_string() },
                }
            },
            ColorScheme::BlueRed => {
                let red = (255.0 * normalized) as u8;
                let blue = (255.0 * (1.0 - normalized)) as u8;
                Color {
                    background: format!("rgb({},0,{})", red, blue),
                    text: "#fff".to_string(),
                }
            },
            ColorScheme::Viridis => {
                // Simplified viridis approximation
                let r = (255.0 * (0.267 + 0.004 * normalized)) as u8;
                let g = (255.0 * (-0.000 + 0.857 * normalized)) as u8;
                let b = (255.0 * (0.329 + 0.584 * normalized)) as u8;
                Color {
                    background: format!("rgb({},{},{})", r, g, b),
                    text: if normalized > 0.5 { "#000".to_string() } else { "#fff".to_string() },
                }
            },
            ColorScheme::Plasma => {
                // Simplified plasma approximation
                let r = (255.0 * (0.050 + 0.839 * normalized)) as u8;
                let g = (255.0 * (0.004 + 0.728 * normalized)) as u8;
                let b = (255.0 * (0.506 + 0.494 * normalized)) as u8;
                Color {
                    background: format!("rgb({},{},{})", r, g, b),
                    text: if normalized > 0.5 { "#000".to_string() } else { "#fff".to_string() },
                }
            },
            ColorScheme::Custom => Color {
                background: "#ffffff".to_string(),
                text: "#000000".to_string(),
            },
        }
    }

    /// Format tensor data as string
    fn format_tensor_data(&self, tensor: &Tensor) -> Result<String> {
        match tensor {
            Tensor::F32(arr) => {
                let shape = arr.shape();

                match shape.len() {
                    1 => self.format_1d_tensor(arr),
                    2 => self.format_2d_tensor(arr),
                    _ => self.format_nd_tensor(arr),
                }
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Tensor formatting only supported for F32 tensors",
                "format_tensor_data",
            )),
        }
    }

    /// Format 1D tensor
    fn format_1d_tensor(&self, arr: &ArrayD<f32>) -> Result<String> {
        let mut output = String::new();
        let len = arr.len();
        let max_display = self.config.max_display_elements;

        output.push('[');

        if len <= max_display {
            for (i, &value) in arr.iter().enumerate() {
                if i > 0 {
                    output.push_str(", ");
                }
                write!(
                    output,
                    "{:.precision$}",
                    value,
                    precision = self.config.precision
                )?;
            }
        } else {
            let half = max_display / 2;

            // Show first half
            for i in 0..half {
                if i > 0 {
                    output.push_str(", ");
                }
                write!(
                    output,
                    "{:.precision$}",
                    arr[[i]],
                    precision = self.config.precision
                )?;
            }

            output.push_str(", ..., ");

            // Show last half
            for i in (len - half)..len {
                if i > len - half {
                    output.push_str(", ");
                }
                write!(
                    output,
                    "{:.precision$}",
                    arr[[i]],
                    precision = self.config.precision
                )?;
            }
        }

        output.push(']');
        Ok(output)
    }

    /// Format 2D tensor
    fn format_2d_tensor(&self, arr: &ArrayD<f32>) -> Result<String> {
        let mut output = String::new();
        let shape = arr.shape();
        let rows = shape[0];
        let cols = shape[1];
        let max_display = self.config.max_display_elements;

        output.push_str("[\n");

        let rows_to_show = if rows <= max_display {
            (0..rows).collect()
        } else {
            let half = max_display / 2;
            let mut indices = (0..half).collect::<Vec<_>>();
            indices.extend((rows - half)..rows);
            indices
        };

        for (idx, &i) in rows_to_show.iter().enumerate() {
            if idx > 0
                && rows > max_display
                && i >= rows - max_display / 2
                && idx == max_display / 2
            {
                output.push_str("  ...\n");
            }

            output.push_str("  [");

            if cols <= max_display {
                for j in 0..cols {
                    if j > 0 {
                        output.push_str(", ");
                    }
                    write!(
                        output,
                        "{:>8.precision$}",
                        arr[[i, j]],
                        precision = self.config.precision
                    )?;
                }
            } else {
                let half = max_display / 2;

                for j in 0..half {
                    if j > 0 {
                        output.push_str(", ");
                    }
                    write!(
                        output,
                        "{:>8.precision$}",
                        arr[[i, j]],
                        precision = self.config.precision
                    )?;
                }

                output.push_str(", ..., ");

                for j in (cols - half)..cols {
                    if j > cols - half {
                        output.push_str(", ");
                    }
                    write!(
                        output,
                        "{:>8.precision$}",
                        arr[[i, j]],
                        precision = self.config.precision
                    )?;
                }
            }

            if i == rows - 1 {
                output.push_str("]\n");
            } else {
                output.push_str("],\n");
            }
        }

        output.push(']');
        Ok(output)
    }

    /// Format N-dimensional tensor
    fn format_nd_tensor(&self, arr: &ArrayD<f32>) -> Result<String> {
        let shape = arr.shape();
        Ok(format!(
            "Tensor with shape {:?} (displaying shape only for >2D tensors)",
            shape
        ))
    }

    /// Convert tensor to nested vector for JSON serialization
    fn tensor_to_nested_vec(&self, tensor: &Tensor) -> Result<serde_json::Value> {
        match tensor {
            Tensor::F32(arr) => {
                let shape = arr.shape();

                match shape.len() {
                    1 => {
                        let vec: Vec<f32> = arr.iter().cloned().collect();
                        Ok(serde_json::Value::Array(
                            vec.into_iter().map(|x| serde_json::json!(x)).collect(),
                        ))
                    },
                    2 => {
                        let mut rows = Vec::new();
                        for i in 0..shape[0] {
                            let mut row = Vec::new();
                            for j in 0..shape[1] {
                                row.push(serde_json::json!(arr[[i, j]]));
                            }
                            rows.push(serde_json::Value::Array(row));
                        }
                        Ok(serde_json::Value::Array(rows))
                    },
                    _ => Ok(serde_json::json!({
                        "shape": shape,
                        "note": "Multi-dimensional tensor data not fully serialized"
                    })),
                }
            },
            _ => Ok(serde_json::json!({
                "error": "Only F32 tensors supported for JSON serialization"
            })),
        }
    }

    /// Extract 2D slice from multi-dimensional tensor
    fn extract_2d_slice(&self, tensor: &Tensor, index: usize) -> Result<Tensor> {
        match tensor {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                if shape.len() < 3 {
                    return Err(TrustformersError::tensor_op_error(
                        "Need at least 3D tensor for slicing",
                        "extract_2d_slice",
                    ));
                }

                // Create a 2D slice by taking index along first dimension
                let slice_shape = &shape[1..];
                let mut slice_data = Vec::new();

                // For simplicity, assume we're taking a 2D slice from first two remaining dimensions
                if slice_shape.len() >= 2 {
                    for i in 0..slice_shape[0] {
                        for j in 0..slice_shape[1] {
                            let mut indices = vec![index, i, j];
                            indices.extend(vec![0; shape.len() - 3]); // Fill remaining dimensions with 0

                            // Ensure indices match tensor dimensions
                            indices.truncate(shape.len());
                            slice_data.push(arr[indices.as_slice()]);
                        }
                    }

                    Tensor::from_vec(slice_data, &[slice_shape[0], slice_shape[1]])
                } else {
                    Err(TrustformersError::tensor_op_error(
                        "Cannot create 2D slice from tensor",
                        "extract_2d_slice",
                    ))
                }
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Slice extraction only supported for F32 tensors",
                "extract_2d_slice",
            )),
        }
    }

    /// Save visualization to file
    pub fn save_to_file(&self, tensor: &Tensor, path: &str) -> Result<()> {
        let content = self.visualize_tensor(tensor)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Generate comparison visualization for two tensors
    pub fn compare_tensors(&self, tensor1: &Tensor, tensor2: &Tensor) -> Result<String> {
        let mut output = String::new();

        writeln!(output, "Tensor Comparison\n")?;
        writeln!(output, "=== Tensor 1 ===")?;
        writeln!(output, "{}", self.visualize_tensor(tensor1)?)?;
        writeln!(output, "\n=== Tensor 2 ===")?;
        writeln!(output, "{}", self.visualize_tensor(tensor2)?)?;

        // Add difference statistics if tensors have same shape
        if tensor1.shape() == tensor2.shape() {
            writeln!(output, "\n=== Difference Analysis ===")?;
            if let (Tensor::F32(arr1), Tensor::F32(arr2)) = (tensor1, tensor2) {
                let diff_data: Vec<f32> =
                    arr1.iter().zip(arr2.iter()).map(|(&a, &b)| (a - b).abs()).collect();

                if !diff_data.is_empty() {
                    let max_diff = diff_data.iter().fold(0.0f32, |a, &b| a.max(b));
                    let mean_diff = diff_data.iter().sum::<f32>() / diff_data.len() as f32;
                    let mse =
                        diff_data.iter().map(|&x| x * x).sum::<f32>() / diff_data.len() as f32;

                    writeln!(output, "Max Absolute Difference: {:.6}", max_diff)?;
                    writeln!(output, "Mean Absolute Difference: {:.6}", mean_diff)?;
                    writeln!(output, "Mean Squared Error: {:.6}", mse)?;
                }
            }
        }

        Ok(output)
    }
}

/// Color representation for visualization
#[derive(Debug, Clone)]
struct Color {
    background: String,
    text: String,
}

// From<std::fmt::Error> implementation is already provided in error.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualizer_creation() {
        let visualizer = TensorVisualizer::new();
        assert_eq!(visualizer.config.max_display_elements, 20);
        assert_eq!(visualizer.config.color_scheme, ColorScheme::Viridis);
    }

    #[test]
    fn test_statistics_computation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, &[5]).unwrap();

        let visualizer = TensorVisualizer::new();
        let stats = visualizer.compute_statistics(&tensor).unwrap();

        assert_eq!(stats.element_count, 5);
        assert_eq!(stats.min_value, 1.0);
        assert_eq!(stats.max_value, 5.0);
        assert_eq!(stats.mean_value, 3.0);
        assert_eq!(stats.zeros_count, 0);
    }

    #[test]
    fn test_heatmap_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, &[2, 2]).unwrap();

        let visualizer = TensorVisualizer::new();
        let heatmap = visualizer.create_heatmap(&tensor).unwrap();

        assert_eq!(heatmap.width, 2);
        assert_eq!(heatmap.height, 2);
        assert_eq!(heatmap.min_value, 1.0);
        assert_eq!(heatmap.max_value, 4.0);
    }

    #[test]
    fn test_histogram_creation() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let tensor = Tensor::from_vec(data, &[6]).unwrap();

        let visualizer = TensorVisualizer::new();
        let histogram = visualizer.create_histogram(&tensor, 3).unwrap();

        assert_eq!(histogram.bins.len(), 3);
        assert_eq!(histogram.counts.len(), 3);
        assert_eq!(histogram.total_count, 6);
    }

    #[test]
    fn test_text_visualization() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, &[2, 2]).unwrap();

        let visualizer = TensorVisualizer::new();
        let result = visualizer.visualize_tensor(&tensor).unwrap();

        assert!(result.contains("Shape: [2, 2]"));
        assert!(result.contains("Statistics:"));
        assert!(result.contains("Data:"));
    }

    #[test]
    fn test_json_visualization() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, &[4]).unwrap();

        let config = VisualizationConfig {
            output_format: OutputFormat::JSON,
            ..Default::default()
        };
        let visualizer = TensorVisualizer::with_config(config);

        let result = visualizer.visualize_tensor(&tensor).unwrap();
        assert!(result.contains("\"shape\""));
        assert!(result.contains("\"statistics\""));
    }

    #[test]
    fn test_html_visualization() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec(data, &[2, 2]).unwrap();

        let config = VisualizationConfig {
            output_format: OutputFormat::HTML,
            ..Default::default()
        };
        let visualizer = TensorVisualizer::with_config(config);

        let result = visualizer.visualize_tensor(&tensor).unwrap();
        assert!(result.contains("<div class='tensor-visualization'>"));
        assert!(result.contains("<table"));
    }

    #[test]
    fn test_color_schemes() {
        let visualizer = TensorVisualizer::new();

        let color = visualizer.value_to_color(0.5, 0.0, 1.0);
        assert!(!color.background.is_empty());
        assert!(!color.text.is_empty());
    }

    #[test]
    fn test_tensor_comparison() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![1.1, 2.1, 3.1, 4.1];
        let tensor1 = Tensor::from_vec(data1, &[4]).unwrap();
        let tensor2 = Tensor::from_vec(data2, &[4]).unwrap();

        let visualizer = TensorVisualizer::new();
        let comparison = visualizer.compare_tensors(&tensor1, &tensor2).unwrap();

        assert!(comparison.contains("Tensor Comparison"));
        assert!(comparison.contains("=== Tensor 1 ==="));
        assert!(comparison.contains("=== Tensor 2 ==="));
        assert!(comparison.contains("=== Difference Analysis ==="));
    }

    #[test]
    fn test_config_serialization() {
        let config = VisualizationConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: VisualizationConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            config.max_display_elements,
            deserialized.max_display_elements
        );
        assert_eq!(config.color_scheme, deserialized.color_scheme);
    }
}
