//! Custom Visualization Plugin System
//!
//! This module provides an extensible plugin system for custom visualizations,
//! allowing users to create and register their own visualization tools.

use anyhow::Result;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Visualization data that can be passed to plugins
#[derive(Debug, Clone)]
pub enum VisualizationData {
    /// 1D array data
    Array1D(Vec<f64>),
    /// 2D array data
    Array2D(Vec<Vec<f64>>),
    /// Tensor data with shape information
    Tensor { data: Vec<f64>, shape: Vec<usize> },
    /// Key-value pairs (for metadata, metrics, etc.)
    KeyValue(HashMap<String, String>),
    /// Time series data
    TimeSeries {
        timestamps: Vec<f64>,
        values: Vec<f64>,
        labels: Vec<String>,
    },
    /// Custom JSON data
    Json(serde_json::Value),
}

/// Output format for visualizations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// PNG image (not supported — requires binary encoder; use Svg instead)
    Png,
    /// SVG vector graphics
    Svg,
    /// HTML interactive visualization (SVG embedded in HTML)
    Html,
    /// Plain text/ASCII
    Text,
    /// JSON data
    Json,
    /// CSV data
    Csv,
}

/// Plugin capabilities and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Author
    pub author: String,
    /// Supported input data types
    pub supported_inputs: Vec<String>,
    /// Supported output formats
    pub supported_outputs: Vec<OutputFormat>,
    /// Tags/categories
    pub tags: Vec<String>,
}

/// Configuration for visualization plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Output format
    pub output_format: OutputFormat,
    /// Output path (file or directory)
    pub output_path: Option<String>,
    /// Width in pixels (for image outputs)
    pub width: usize,
    /// Height in pixels (for image outputs)
    pub height: usize,
    /// Color scheme
    pub color_scheme: String,
    /// Additional custom parameters
    pub custom_params: HashMap<String, serde_json::Value>,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            output_format: OutputFormat::Svg,
            output_path: None,
            width: 800,
            height: 600,
            color_scheme: "viridis".to_string(),
            custom_params: HashMap::new(),
        }
    }
}

/// Result of plugin execution
#[derive(Debug)]
pub struct PluginResult {
    /// Success status
    pub success: bool,
    /// Output file path (if file was created)
    pub output_path: Option<String>,
    /// Raw output data (if applicable)
    pub output_data: Option<Vec<u8>>,
    /// Metadata about the visualization
    pub metadata: HashMap<String, String>,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// Trait for visualization plugins
///
/// Implement this trait to create custom visualization plugins.
pub trait VisualizationPlugin: Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> PluginMetadata;

    /// Execute the visualization
    ///
    /// # Arguments
    /// * `data` - Input data to visualize
    /// * `config` - Configuration for the visualization
    ///
    /// # Returns
    /// Result containing the plugin output
    fn execute(&self, data: VisualizationData, config: PluginConfig) -> Result<PluginResult>;

    /// Validate input data
    ///
    /// # Arguments
    /// * `data` - Data to validate
    ///
    /// # Returns
    /// True if data is valid for this plugin
    fn validate(&self, data: &VisualizationData) -> bool {
        // Default implementation: accept all data
        let _ = data;
        true
    }

    /// Get configuration schema (for UI generation)
    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }
}

/// Plugin manager for registering and executing visualization plugins
pub struct PluginManager {
    /// Registered plugins
    plugins: Arc<RwLock<HashMap<String, Box<dyn VisualizationPlugin>>>>,
    /// Plugin execution history
    history: Arc<RwLock<Vec<PluginExecution>>>,
}

/// Record of plugin execution
#[derive(Debug, Clone)]
struct PluginExecution {
    plugin_name: String,
    timestamp: std::time::SystemTime,
    success: bool,
    duration_ms: u128,
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        let manager = Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
        };

        // Register built-in plugins
        manager.register_builtin_plugins();

        manager
    }

    /// Register built-in plugins
    fn register_builtin_plugins(&self) {
        // Register histogram plugin
        self.register_plugin(Box::new(HistogramPlugin)).ok();

        // Register heatmap plugin
        self.register_plugin(Box::new(HeatmapPlugin)).ok();

        // Register line plot plugin
        self.register_plugin(Box::new(LinePlotPlugin)).ok();

        // Register scatter plot plugin
        self.register_plugin(Box::new(ScatterPlotPlugin)).ok();
    }

    /// Register a new plugin
    ///
    /// # Arguments
    /// * `plugin` - Plugin to register
    pub fn register_plugin(&self, plugin: Box<dyn VisualizationPlugin>) -> Result<()> {
        let name = plugin.metadata().name.clone();

        self.plugins.write().insert(name.clone(), plugin);

        tracing::info!(plugin_name = %name, "Registered visualization plugin");

        Ok(())
    }

    /// Unregister a plugin
    ///
    /// # Arguments
    /// * `name` - Name of plugin to unregister
    pub fn unregister_plugin(&self, name: &str) -> Result<()> {
        self.plugins.write().remove(name);

        tracing::info!(plugin_name = %name, "Unregistered visualization plugin");

        Ok(())
    }

    /// Get list of registered plugins
    pub fn list_plugins(&self) -> Vec<PluginMetadata> {
        self.plugins.read().values().map(|p| p.metadata()).collect()
    }

    /// Execute a plugin
    ///
    /// # Arguments
    /// * `plugin_name` - Name of plugin to execute
    /// * `data` - Input data
    /// * `config` - Configuration
    pub fn execute(
        &self,
        plugin_name: &str,
        data: VisualizationData,
        config: PluginConfig,
    ) -> Result<PluginResult> {
        let start_time = std::time::Instant::now();

        let result = {
            let plugins = self.plugins.read();
            let plugin = plugins
                .get(plugin_name)
                .ok_or_else(|| anyhow::anyhow!("Plugin not found: {}", plugin_name))?;

            // Validate data
            if !plugin.validate(&data) {
                anyhow::bail!("Invalid data for plugin: {}", plugin_name);
            }

            plugin.execute(data, config)?
        };

        let duration = start_time.elapsed().as_millis();

        // Record execution
        self.history.write().push(PluginExecution {
            plugin_name: plugin_name.to_string(),
            timestamp: std::time::SystemTime::now(),
            success: result.success,
            duration_ms: duration,
        });

        Ok(result)
    }

    /// Get plugin by name
    pub fn get_plugin(&self, name: &str) -> Option<PluginMetadata> {
        self.plugins.read().get(name).map(|p| p.metadata())
    }

    /// Get execution history
    pub fn get_history(&self) -> Vec<String> {
        self.history
            .read()
            .iter()
            .map(|e| {
                format!(
                    "{}: {} ({}ms) - {}",
                    e.timestamp.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
                    e.plugin_name,
                    e.duration_ms,
                    if e.success { "success" } else { "failed" }
                )
            })
            .collect()
    }
}

// ============================================================================
// SVG rendering — pure handwritten, no external drawing dep
// ============================================================================

mod svg_render {
    //! Pure-Rust handwritten SVG generators.
    //!
    //! Each function returns a complete, self-contained SVG document.
    //! Layout constants and a basic coordinate mapper are shared across helpers.

    use super::PluginConfig;

    // Layout constants (pixels inside the SVG viewBox)
    const MARGIN_TOP: f64 = 40.0;
    const MARGIN_BOTTOM: f64 = 50.0;
    const MARGIN_LEFT: f64 = 60.0;
    const MARGIN_RIGHT: f64 = 20.0;
    const AXIS_TICK_LEN: f64 = 5.0;

    // SVG attribute strings (literal, not format arguments)
    const FONT_ATTR: &str = r#"font-family="sans-serif""#;
    const AXIS_COLOR: &str = "#555";
    const BAR_COLOR: &str = "#4878CF";

    /// Clamp a value to [lo, hi]
    fn clamp_f64(v: f64, lo: f64, hi: f64) -> f64 {
        if v < lo {
            lo
        } else if v > hi {
            hi
        } else {
            v
        }
    }

    /// Map a data value to a pixel coordinate within the plot area.
    fn map_x(v: f64, data_min: f64, data_max: f64, px_left: f64, px_right: f64) -> f64 {
        let range = data_max - data_min;
        if range == 0.0 {
            return (px_left + px_right) / 2.0;
        }
        let t = (v - data_min) / range;
        clamp_f64(px_left + t * (px_right - px_left), px_left, px_right)
    }

    /// Map a data value to a pixel y-coordinate (data grows up, SVG y grows down).
    fn map_y(v: f64, data_min: f64, data_max: f64, px_top: f64, px_bottom: f64) -> f64 {
        let range = data_max - data_min;
        if range == 0.0 {
            return (px_top + px_bottom) / 2.0;
        }
        let t = (v - data_min) / range;
        clamp_f64(px_bottom - t * (px_bottom - px_top), px_top, px_bottom)
    }

    /// SVG header with explicit width/height and a white background.
    fn svg_open(width: usize, height: usize) -> String {
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">\n\
             <rect width=\"{w}\" height=\"{h}\" fill=\"white\"/>\n",
            w = width,
            h = height,
        )
    }

    fn svg_close() -> &'static str {
        "</svg>"
    }

    /// Render a title centred at the top of the SVG.
    fn svg_title(text: &str, width: usize) -> String {
        let cx = width / 2;
        let escaped = escape_xml(text);
        format!(
            "<text x=\"{cx}\" y=\"24\" text-anchor=\"middle\" {font} font-size=\"16\" fill=\"#333\">{text}</text>\n",
            cx = cx,
            font = FONT_ATTR,
            text = escaped,
        )
    }

    /// Render axis lines, tick marks and numeric tick labels.
    #[allow(clippy::too_many_arguments)]
    fn svg_axes(
        px_left: f64,
        px_right: f64,
        px_top: f64,
        px_bottom: f64,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        x_label: &str,
        y_label: &str,
        _width: usize,
        height: usize,
        n_ticks: usize,
    ) -> String {
        let mut out = String::new();
        let c = AXIS_COLOR;

        // Vertical axis line (left)
        out.push_str(&format!(
            "<line x1=\"{x1:.2}\" y1=\"{y1:.2}\" x2=\"{x2:.2}\" y2=\"{y2:.2}\" stroke=\"{c}\" stroke-width=\"1\"/>\n",
            x1 = px_left, y1 = px_top, x2 = px_left, y2 = px_bottom, c = c,
        ));
        // Horizontal axis line (bottom)
        out.push_str(&format!(
            "<line x1=\"{x1:.2}\" y1=\"{y1:.2}\" x2=\"{x2:.2}\" y2=\"{y2:.2}\" stroke=\"{c}\" stroke-width=\"1\"/>\n",
            x1 = px_left, y1 = px_bottom, x2 = px_right, y2 = px_bottom, c = c,
        ));

        // X ticks
        let x_range = x_max - x_min;
        for i in 0..=n_ticks {
            let frac = i as f64 / n_ticks as f64;
            let val = x_min + frac * x_range;
            let px = px_left + frac * (px_right - px_left);
            let ty = px_bottom + AXIS_TICK_LEN + 12.0;
            let y2 = px_bottom + AXIS_TICK_LEN;
            out.push_str(&format!(
                "<line x1=\"{px:.2}\" y1=\"{y1:.2}\" x2=\"{px:.2}\" y2=\"{y2:.2}\" stroke=\"{c}\" stroke-width=\"1\"/>\n\
                 <text x=\"{px:.2}\" y=\"{ty:.2}\" text-anchor=\"middle\" {font} font-size=\"10\" fill=\"{c}\">{val:.2}</text>\n",
                px = px, y1 = px_bottom, y2 = y2, ty = ty, c = c, font = FONT_ATTR, val = val,
            ));
        }

        // Y ticks
        let y_range = y_max - y_min;
        for i in 0..=n_ticks {
            let frac = i as f64 / n_ticks as f64;
            let val = y_min + frac * y_range;
            let py = px_bottom - frac * (px_bottom - px_top);
            let tick_x1 = px_left - AXIS_TICK_LEN;
            let tx = tick_x1 - 2.0;
            let py_t = py + 4.0;
            out.push_str(&format!(
                "<line x1=\"{tx1:.2}\" y1=\"{py:.2}\" x2=\"{x2:.2}\" y2=\"{py:.2}\" stroke=\"{c}\" stroke-width=\"1\"/>\n\
                 <text x=\"{tx:.2}\" y=\"{pyt:.2}\" text-anchor=\"end\" {font} font-size=\"10\" fill=\"{c}\">{val:.2}</text>\n",
                tx1 = tick_x1, py = py, x2 = px_left, tx = tx, pyt = py_t,
                c = c, font = FONT_ATTR, val = val,
            ));
        }

        // Axis labels
        if !x_label.is_empty() {
            let lx = (px_left + px_right) / 2.0;
            let ly = height as f64 - 4.0;
            let label = escape_xml(x_label);
            out.push_str(&format!(
                "<text x=\"{lx:.2}\" y=\"{ly:.2}\" text-anchor=\"middle\" {font} font-size=\"12\" fill=\"{c}\">{label}</text>\n",
                lx = lx, ly = ly, font = FONT_ATTR, c = c, label = label,
            ));
        }
        if !y_label.is_empty() {
            let ry_x = -((px_top + px_bottom) / 2.0);
            let label = escape_xml(y_label);
            out.push_str(&format!(
                "<text transform=\"rotate(-90)\" x=\"{rx:.2}\" y=\"14\" text-anchor=\"middle\" {font} font-size=\"12\" fill=\"{c}\">{label}</text>\n",
                rx = ry_x, font = FONT_ATTR, c = c, label = label,
            ));
        }

        out
    }

    /// Escape XML special characters in text content.
    fn escape_xml(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }

    // -------------------------------------------------------------------------
    // Public renderers
    // -------------------------------------------------------------------------

    /// Render a histogram. `bins` is a slice of `(bin_left, bin_right, count)`.
    pub fn histogram(bins: &[(f64, f64, usize)], config: &PluginConfig) -> String {
        let w = config.width;
        let h = config.height;
        let px_left = MARGIN_LEFT;
        let px_right = w as f64 - MARGIN_RIGHT;
        let px_top = MARGIN_TOP;
        let px_bottom = h as f64 - MARGIN_BOTTOM;

        let title = config
            .custom_params
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Histogram");
        let x_label =
            config.custom_params.get("x_label").and_then(|v| v.as_str()).unwrap_or("Value");
        let y_label =
            config.custom_params.get("y_label").and_then(|v| v.as_str()).unwrap_or("Count");

        let max_count = bins.iter().map(|b| b.2).max().unwrap_or(1).max(1);
        let x_min = bins.first().map(|b| b.0).unwrap_or(0.0);
        let x_max = bins.last().map(|b| b.1).unwrap_or(1.0);

        let mut out = svg_open(w, h);
        out.push_str(&svg_title(title, w));
        out.push_str(&svg_axes(
            px_left,
            px_right,
            px_top,
            px_bottom,
            x_min,
            x_max,
            0.0,
            max_count as f64,
            x_label,
            y_label,
            w,
            h,
            5,
        ));

        // Draw bars
        let fill = BAR_COLOR;
        for (bin_left, bin_right, count) in bins {
            let bx1 = map_x(*bin_left, x_min, x_max, px_left, px_right);
            let bx2 = map_x(*bin_right, x_min, x_max, px_left, px_right);
            let by_top = map_y(*count as f64, 0.0, max_count as f64, px_top, px_bottom);
            let bar_h = px_bottom - by_top;
            let bar_w = (bx2 - bx1).max(1.0);
            out.push_str(&format!(
                "<rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{bw:.2}\" height=\"{bh:.2}\" fill=\"{fill}\" stroke=\"white\" stroke-width=\"1\"/>\n",
                x = bx1, y = by_top, bw = bar_w, bh = bar_h, fill = fill,
            ));
        }

        out.push_str(svg_close());
        out
    }

    /// Render a heatmap. `values` is row-major with dimensions `rows × cols`.
    pub fn heatmap(rows: usize, cols: usize, values: &[f64], config: &PluginConfig) -> String {
        let w = config.width;
        let h = config.height;
        let px_left = MARGIN_LEFT;
        let px_right = w as f64 - MARGIN_RIGHT;
        let px_top = MARGIN_TOP;
        let px_bottom = h as f64 - MARGIN_BOTTOM;

        let title = config.custom_params.get("title").and_then(|v| v.as_str()).unwrap_or("Heatmap");

        let cell_w = if cols > 0 { (px_right - px_left) / cols as f64 } else { 1.0 };
        let cell_h = if rows > 0 { (px_bottom - px_top) / rows as f64 } else { 1.0 };

        let (v_min, v_max) =
            values.iter().copied().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                (lo.min(v), hi.max(v))
            });
        let v_range = (v_max - v_min).max(f64::EPSILON);

        let mut out = svg_open(w, h);
        out.push_str(&svg_title(title, w));

        // Draw cells
        for row_idx in 0..rows {
            for col_idx in 0..cols {
                let idx = row_idx * cols + col_idx;
                let val = values.get(idx).copied().unwrap_or(0.0);
                let t = ((val - v_min) / v_range).clamp(0.0, 1.0);
                // Viridis-like gradient: dark purple to yellow
                let red = (255.0 * (t * t)) as u8;
                let green = (255.0 * t * (1.0 - t * 0.5)) as u8;
                let blue = (255.0 * (1.0 - t)) as u8;
                let cx = px_left + col_idx as f64 * cell_w;
                let cy = px_top + row_idx as f64 * cell_h;
                out.push_str(&format!(
                    "<rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{cw:.2}\" height=\"{ch:.2}\" fill=\"rgb({r},{g},{b})\"/>\n",
                    x = cx, y = cy, cw = cell_w, ch = cell_h,
                    r = red, g = green, b = blue,
                ));
            }
        }

        out.push_str(svg_close());
        out
    }

    /// Render a line plot. `points` is a slice of `(x, y)` pairs, ordered by x.
    pub fn line_plot(points: &[(f64, f64)], config: &PluginConfig) -> String {
        let w = config.width;
        let h = config.height;
        let px_left = MARGIN_LEFT;
        let px_right = w as f64 - MARGIN_RIGHT;
        let px_top = MARGIN_TOP;
        let px_bottom = h as f64 - MARGIN_BOTTOM;

        let title = config
            .custom_params
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Line Plot");
        let x_label = config.custom_params.get("x_label").and_then(|v| v.as_str()).unwrap_or("X");
        let y_label = config.custom_params.get("y_label").and_then(|v| v.as_str()).unwrap_or("Y");

        let (x_min, x_max, y_min, y_max) = data_bounds(points);

        let mut out = svg_open(w, h);
        out.push_str(&svg_title(title, w));
        out.push_str(&svg_axes(
            px_left, px_right, px_top, px_bottom, x_min, x_max, y_min, y_max, x_label, y_label, w,
            h, 5,
        ));

        if !points.is_empty() {
            // Build a polyline
            let pts: Vec<String> = points
                .iter()
                .map(|(x, y)| {
                    let px = map_x(*x, x_min, x_max, px_left, px_right);
                    let py = map_y(*y, y_min, y_max, px_top, px_bottom);
                    format!("{:.2},{:.2}", px, py)
                })
                .collect();
            let stroke = BAR_COLOR;
            out.push_str(&format!(
                "<polyline points=\"{pts}\" fill=\"none\" stroke=\"{stroke}\" stroke-width=\"2\"/>\n",
                pts = pts.join(" "),
                stroke = stroke,
            ));
        }

        out.push_str(svg_close());
        out
    }

    /// Render a scatter plot. `points` is a slice of `(x, y)` pairs.
    pub fn scatter(points: &[(f64, f64)], config: &PluginConfig) -> String {
        let w = config.width;
        let h = config.height;
        let px_left = MARGIN_LEFT;
        let px_right = w as f64 - MARGIN_RIGHT;
        let px_top = MARGIN_TOP;
        let px_bottom = h as f64 - MARGIN_BOTTOM;

        let title = config
            .custom_params
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Scatter Plot");
        let x_label = config.custom_params.get("x_label").and_then(|v| v.as_str()).unwrap_or("X");
        let y_label = config.custom_params.get("y_label").and_then(|v| v.as_str()).unwrap_or("Y");

        let (x_min, x_max, y_min, y_max) = data_bounds(points);

        let mut out = svg_open(w, h);
        out.push_str(&svg_title(title, w));
        out.push_str(&svg_axes(
            px_left, px_right, px_top, px_bottom, x_min, x_max, y_min, y_max, x_label, y_label, w,
            h, 5,
        ));

        let fill = BAR_COLOR;
        for (x, y) in points {
            let px = map_x(*x, x_min, x_max, px_left, px_right);
            let py = map_y(*y, y_min, y_max, px_top, px_bottom);
            out.push_str(&format!(
                "<circle cx=\"{cx:.2}\" cy=\"{cy:.2}\" r=\"4\" fill=\"{fill}\" fill-opacity=\"0.7\"/>\n",
                cx = px, cy = py, fill = fill,
            ));
        }

        out.push_str(svg_close());
        out
    }

    // -------------------------------------------------------------------------
    // JSON / CSV renderers
    // -------------------------------------------------------------------------

    /// Render histogram data as JSON.
    pub fn histogram_json(
        bins: &[(f64, f64, usize)],
        min: f64,
        max: f64,
        n_values: usize,
    ) -> String {
        let bins_arr: Vec<serde_json::Value> = bins
            .iter()
            .map(|(lo, hi, cnt)| {
                serde_json::json!({
                    "bin_left": lo,
                    "bin_right": hi,
                    "count": cnt,
                })
            })
            .collect();
        serde_json::to_string_pretty(&serde_json::json!({
            "type": "histogram",
            "n_values": n_values,
            "min": min,
            "max": max,
            "bins": bins_arr,
        }))
        .unwrap_or_else(|_| "{}".to_string())
    }

    /// Render histogram data as CSV.
    pub fn histogram_csv(bins: &[(f64, f64, usize)]) -> String {
        let mut out = String::from("bin_left,bin_right,count\n");
        for (lo, hi, cnt) in bins {
            out.push_str(&format!("{},{},{}\n", lo, hi, cnt));
        }
        out
    }

    /// Render 2D point data as JSON.
    pub fn points_json(kind: &str, points: &[(f64, f64)]) -> String {
        let pts: Vec<serde_json::Value> =
            points.iter().map(|(x, y)| serde_json::json!({ "x": x, "y": y })).collect();
        serde_json::to_string_pretty(&serde_json::json!({
            "type": kind,
            "n_points": points.len(),
            "points": pts,
        }))
        .unwrap_or_else(|_| "{}".to_string())
    }

    /// Render 2D point data as CSV.
    pub fn points_csv(points: &[(f64, f64)]) -> String {
        let mut out = String::from("x,y\n");
        for (x, y) in points {
            out.push_str(&format!("{},{}\n", x, y));
        }
        out
    }

    /// Wrap an SVG string in a minimal HTML document.
    pub fn wrap_html(svg: &str) -> String {
        format!(
            "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/></head><body>{svg}</body></html>",
            svg = svg
        )
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /// Compute (x_min, x_max, y_min, y_max) for a slice of (x, y) points.
    fn data_bounds(points: &[(f64, f64)]) -> (f64, f64, f64, f64) {
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        for (x, y) in points {
            if *x < x_min {
                x_min = *x;
            }
            if *x > x_max {
                x_max = *x;
            }
            if *y < y_min {
                y_min = *y;
            }
            if *y > y_max {
                y_max = *y;
            }
        }
        // Pad a little so points at edge are visible
        let x_pad = if (x_max - x_min).abs() < f64::EPSILON { 1.0 } else { (x_max - x_min) * 0.05 };
        let y_pad = if (y_max - y_min).abs() < f64::EPSILON { 1.0 } else { (y_max - y_min) * 0.05 };
        (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)
    }
}

// ============================================================================
// Helper: write bytes to output_path when configured
// ============================================================================

fn write_output_path(path: &str, bytes: &[u8]) -> Result<()> {
    let p = PathBuf::from(path);
    std::fs::write(&p, bytes)
        .map_err(|e| anyhow::anyhow!("Failed to write output to {}: {}", p.display(), e))
}

// ============================================================================
// Built-in Plugins
// ============================================================================

/// Histogram visualization plugin
struct HistogramPlugin;

impl VisualizationPlugin for HistogramPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "histogram".to_string(),
            version: "1.0.0".to_string(),
            description: "Generates histogram visualizations".to_string(),
            author: "TrustformeRS".to_string(),
            supported_inputs: vec!["Array1D".to_string(), "Tensor".to_string()],
            supported_outputs: vec![
                OutputFormat::Svg,
                OutputFormat::Html,
                OutputFormat::Text,
                OutputFormat::Json,
                OutputFormat::Csv,
            ],
            tags: vec!["distribution".to_string(), "statistics".to_string()],
        }
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "width": 800,
            "height": 600,
            "title": "Histogram",
            "x_label": "Value",
            "y_label": "Count",
            "bins": 20
        })
    }

    fn execute(&self, data: VisualizationData, config: PluginConfig) -> Result<PluginResult> {
        let values = match data {
            VisualizationData::Array1D(v) => v,
            VisualizationData::Tensor { data, .. } => data,
            _ => anyhow::bail!("Unsupported data type for histogram"),
        };

        if values.is_empty() {
            anyhow::bail!("Histogram requires non-empty input data");
        }

        // Calculate histogram bins
        let n_bins =
            config.custom_params.get("bins").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        let n_bins = n_bins.max(1);

        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let bin_width =
            if (max - min).abs() < f64::EPSILON { 1.0 } else { (max - min) / n_bins as f64 };

        let mut counts = vec![0usize; n_bins];
        for &value in &values {
            let bin_idx = ((value - min) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(n_bins - 1);
            counts[bin_idx] += 1;
        }

        let bins: Vec<(f64, f64, usize)> = counts
            .iter()
            .enumerate()
            .map(|(i, &cnt)| {
                let lo = min + i as f64 * bin_width;
                let hi = lo + bin_width;
                (lo, hi, cnt)
            })
            .collect();

        let bytes = match config.output_format {
            OutputFormat::Png => {
                anyhow::bail!(
                    "PNG output is not supported (requires binary encoder); use Svg instead"
                )
            },
            OutputFormat::Svg => {
                let svg = svg_render::histogram(&bins, &config);
                svg.into_bytes()
            },
            OutputFormat::Html => {
                let svg = svg_render::histogram(&bins, &config);
                svg_render::wrap_html(&svg).into_bytes()
            },
            OutputFormat::Text => {
                let output_text = format!(
                    "Histogram (bins={}):\nMin={:.4}, Max={:.4}\nBin counts: {:?}",
                    n_bins, min, max, counts
                );
                output_text.into_bytes()
            },
            OutputFormat::Json => {
                svg_render::histogram_json(&bins, min, max, values.len()).into_bytes()
            },
            OutputFormat::Csv => svg_render::histogram_csv(&bins).into_bytes(),
        };

        let out_path_str = if let Some(ref path) = config.output_path {
            write_output_path(path, &bytes)?;
            Some(path.clone())
        } else {
            None
        };

        Ok(PluginResult {
            success: true,
            output_path: out_path_str,
            output_data: Some(bytes),
            metadata: {
                let mut m = HashMap::new();
                m.insert("bins".to_string(), n_bins.to_string());
                m.insert("min".to_string(), min.to_string());
                m.insert("max".to_string(), max.to_string());
                m.insert("n_values".to_string(), values.len().to_string());
                m
            },
            error: None,
        })
    }

    fn validate(&self, data: &VisualizationData) -> bool {
        matches!(
            data,
            VisualizationData::Array1D(_) | VisualizationData::Tensor { .. }
        )
    }
}

/// Heatmap visualization plugin
struct HeatmapPlugin;

impl VisualizationPlugin for HeatmapPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "heatmap".to_string(),
            version: "1.0.0".to_string(),
            description: "Generates heatmap visualizations for 2D data".to_string(),
            author: "TrustformeRS".to_string(),
            supported_inputs: vec!["Array2D".to_string(), "Tensor".to_string()],
            supported_outputs: vec![
                OutputFormat::Svg,
                OutputFormat::Html,
                OutputFormat::Json,
                OutputFormat::Csv,
                OutputFormat::Text,
            ],
            tags: vec!["matrix".to_string(), "2d".to_string()],
        }
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "width": 800,
            "height": 600,
            "title": "Heatmap",
            "x_label": "",
            "y_label": ""
        })
    }

    fn execute(&self, data: VisualizationData, config: PluginConfig) -> Result<PluginResult> {
        let (rows, cols, flat_values) = match &data {
            VisualizationData::Array2D(v) => {
                let r = v.len();
                let c = v.first().map(|row| row.len()).unwrap_or(0);
                let flat: Vec<f64> = v.iter().flat_map(|row| row.iter().copied()).collect();
                (r, c, flat)
            },
            VisualizationData::Tensor { shape, data } if shape.len() == 2 => {
                (shape[0], shape[1], data.clone())
            },
            _ => anyhow::bail!("Heatmap requires 2D data"),
        };

        let bytes = match config.output_format {
            OutputFormat::Png => {
                anyhow::bail!(
                    "PNG output is not supported (requires binary encoder); use Svg instead"
                )
            },
            OutputFormat::Svg => {
                let svg = svg_render::heatmap(rows, cols, &flat_values, &config);
                svg.into_bytes()
            },
            OutputFormat::Html => {
                let svg = svg_render::heatmap(rows, cols, &flat_values, &config);
                svg_render::wrap_html(&svg).into_bytes()
            },
            OutputFormat::Text => format!("Heatmap {}x{}", rows, cols).into_bytes(),
            OutputFormat::Json => {
                let cells: Vec<serde_json::Value> = flat_values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        serde_json::json!({
                            "row": i / cols.max(1),
                            "col": i % cols.max(1),
                            "value": v,
                        })
                    })
                    .collect();
                serde_json::to_string_pretty(&serde_json::json!({
                    "type": "heatmap",
                    "rows": rows,
                    "cols": cols,
                    "cells": cells,
                }))
                .unwrap_or_else(|_| "{}".to_string())
                .into_bytes()
            },
            OutputFormat::Csv => {
                let mut out = String::from("row,col,value\n");
                for (i, v) in flat_values.iter().enumerate() {
                    let r = i / cols.max(1);
                    let c = i % cols.max(1);
                    out.push_str(&format!("{},{},{}\n", r, c, v));
                }
                out.into_bytes()
            },
        };

        let out_path_str = if let Some(ref path) = config.output_path {
            write_output_path(path, &bytes)?;
            Some(path.clone())
        } else {
            None
        };

        Ok(PluginResult {
            success: true,
            output_path: out_path_str,
            output_data: Some(bytes),
            metadata: {
                let mut m = HashMap::new();
                m.insert("rows".to_string(), rows.to_string());
                m.insert("cols".to_string(), cols.to_string());
                m
            },
            error: None,
        })
    }

    fn validate(&self, data: &VisualizationData) -> bool {
        match data {
            VisualizationData::Array2D(_) => true,
            VisualizationData::Tensor { shape, .. } => shape.len() == 2,
            _ => false,
        }
    }
}

/// Line plot visualization plugin
struct LinePlotPlugin;

impl VisualizationPlugin for LinePlotPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "lineplot".to_string(),
            version: "1.0.0".to_string(),
            description: "Generates line plots for time series data".to_string(),
            author: "TrustformeRS".to_string(),
            supported_inputs: vec!["TimeSeries".to_string(), "Array1D".to_string()],
            supported_outputs: vec![
                OutputFormat::Svg,
                OutputFormat::Html,
                OutputFormat::Text,
                OutputFormat::Json,
                OutputFormat::Csv,
            ],
            tags: vec!["timeseries".to_string(), "trend".to_string()],
        }
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "width": 800,
            "height": 600,
            "title": "Line Plot",
            "x_label": "X",
            "y_label": "Y"
        })
    }

    fn execute(&self, data: VisualizationData, config: PluginConfig) -> Result<PluginResult> {
        let points: Vec<(f64, f64)> = match &data {
            VisualizationData::TimeSeries {
                timestamps, values, ..
            } => timestamps.iter().zip(values.iter()).map(|(t, v)| (*t, *v)).collect(),
            VisualizationData::Array1D(v) => {
                v.iter().enumerate().map(|(i, val)| (i as f64, *val)).collect()
            },
            _ => anyhow::bail!("Line plot requires time series or 1D array data"),
        };

        let n_points = points.len();

        let bytes = match config.output_format {
            OutputFormat::Png => {
                anyhow::bail!(
                    "PNG output is not supported (requires binary encoder); use Svg instead"
                )
            },
            OutputFormat::Svg => {
                let svg = svg_render::line_plot(&points, &config);
                svg.into_bytes()
            },
            OutputFormat::Html => {
                let svg = svg_render::line_plot(&points, &config);
                svg_render::wrap_html(&svg).into_bytes()
            },
            OutputFormat::Text => format!("Line plot with {} points", n_points).into_bytes(),
            OutputFormat::Json => svg_render::points_json("lineplot", &points).into_bytes(),
            OutputFormat::Csv => svg_render::points_csv(&points).into_bytes(),
        };

        let out_path_str = if let Some(ref path) = config.output_path {
            write_output_path(path, &bytes)?;
            Some(path.clone())
        } else {
            None
        };

        Ok(PluginResult {
            success: true,
            output_path: out_path_str,
            output_data: Some(bytes),
            metadata: {
                let mut m = HashMap::new();
                m.insert("points".to_string(), n_points.to_string());
                m
            },
            error: None,
        })
    }

    fn validate(&self, data: &VisualizationData) -> bool {
        matches!(
            data,
            VisualizationData::TimeSeries { .. } | VisualizationData::Array1D(_)
        )
    }
}

/// Scatter plot visualization plugin
struct ScatterPlotPlugin;

impl VisualizationPlugin for ScatterPlotPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "scatterplot".to_string(),
            version: "1.0.0".to_string(),
            description: "Generates scatter plots for 2D point data".to_string(),
            author: "TrustformeRS".to_string(),
            supported_inputs: vec!["Array2D".to_string()],
            supported_outputs: vec![
                OutputFormat::Svg,
                OutputFormat::Html,
                OutputFormat::Text,
                OutputFormat::Json,
                OutputFormat::Csv,
            ],
            tags: vec!["correlation".to_string(), "distribution".to_string()],
        }
    }

    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "width": 800,
            "height": 600,
            "title": "Scatter Plot",
            "x_label": "X",
            "y_label": "Y"
        })
    }

    fn execute(&self, data: VisualizationData, config: PluginConfig) -> Result<PluginResult> {
        let points: Vec<(f64, f64)> = match &data {
            VisualizationData::Array2D(v) => v
                .iter()
                .filter_map(
                    |row| {
                        if row.len() >= 2 {
                            Some((row[0], row[1]))
                        } else {
                            None
                        }
                    },
                )
                .collect(),
            _ => anyhow::bail!("Scatter plot requires 2D array data (each row = [x, y])"),
        };

        let n_points = points.len();

        let bytes = match config.output_format {
            OutputFormat::Png => {
                anyhow::bail!(
                    "PNG output is not supported (requires binary encoder); use Svg instead"
                )
            },
            OutputFormat::Svg => {
                let svg = svg_render::scatter(&points, &config);
                svg.into_bytes()
            },
            OutputFormat::Html => {
                let svg = svg_render::scatter(&points, &config);
                svg_render::wrap_html(&svg).into_bytes()
            },
            OutputFormat::Text => format!("Scatter plot with {} points", n_points).into_bytes(),
            OutputFormat::Json => svg_render::points_json("scatterplot", &points).into_bytes(),
            OutputFormat::Csv => svg_render::points_csv(&points).into_bytes(),
        };

        let out_path_str = if let Some(ref path) = config.output_path {
            write_output_path(path, &bytes)?;
            Some(path.clone())
        } else {
            None
        };

        Ok(PluginResult {
            success: true,
            output_path: out_path_str,
            output_data: Some(bytes),
            metadata: {
                let mut m = HashMap::new();
                m.insert("points".to_string(), n_points.to_string());
                m
            },
            error: None,
        })
    }

    fn validate(&self, data: &VisualizationData) -> bool {
        matches!(data, VisualizationData::Array2D(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Existing tests (updated for Svg default)
    // -----------------------------------------------------------------------

    #[test]
    fn test_plugin_manager_creation() {
        let manager = PluginManager::new();
        let plugins = manager.list_plugins();
        assert!(!plugins.is_empty());
    }

    #[test]
    fn test_histogram_plugin() {
        let manager = PluginManager::new();
        let data = VisualizationData::Array1D(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let config = PluginConfig::default(); // default is now Svg

        let result = manager.execute("histogram", data, config).expect("operation failed in test");

        assert!(result.success);
        assert!(result.output_data.is_some());
    }

    #[test]
    fn test_plugin_validation() {
        let manager = PluginManager::new();

        // Valid data for histogram
        let data = VisualizationData::Array1D(vec![1.0, 2.0, 3.0]);
        let config = PluginConfig::default();
        assert!(manager.execute("histogram", data, config.clone()).is_ok());

        // Invalid data for heatmap (needs 2D)
        let data = VisualizationData::Array1D(vec![1.0, 2.0, 3.0]);
        assert!(manager.execute("heatmap", data, config).is_err());
    }

    #[test]
    fn test_custom_plugin_registration() {
        let manager = PluginManager::new();
        let count_before = manager.list_plugins().len();

        // Register histogram again (should replace)
        manager
            .register_plugin(Box::new(HistogramPlugin))
            .expect("operation failed in test");

        let count_after = manager.list_plugins().len();
        assert_eq!(count_before, count_after);
    }

    // -----------------------------------------------------------------------
    // New SVG rendering tests
    // -----------------------------------------------------------------------

    fn svg_config() -> PluginConfig {
        PluginConfig {
            output_format: OutputFormat::Svg,
            ..PluginConfig::default()
        }
    }

    #[test]
    fn test_histogram_svg_contains_rect() {
        let plugin = HistogramPlugin;
        let data = VisualizationData::Array1D(vec![1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 5.5]);
        let result = plugin.execute(data, svg_config()).expect("histogram SVG render failed");
        assert!(result.success);
        let bytes = result.output_data.expect("no output data");
        let svg = String::from_utf8(bytes).expect("invalid UTF-8");
        assert!(
            svg.contains("<rect"),
            "SVG histogram should contain <rect elements; got: {}",
            &svg[..svg.len().min(300)]
        );
        assert!(svg.starts_with("<svg"), "output should start with <svg tag");
    }

    #[test]
    fn test_heatmap_svg_contains_rect_cells() {
        let plugin = HeatmapPlugin;
        let data = VisualizationData::Array2D(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);
        let result = plugin.execute(data, svg_config()).expect("heatmap SVG render failed");
        assert!(result.success);
        let bytes = result.output_data.expect("no output data");
        let svg = String::from_utf8(bytes).expect("invalid UTF-8");
        assert!(
            svg.contains("<rect"),
            "SVG heatmap should contain <rect cell elements"
        );
        assert!(svg.starts_with("<svg"));
    }

    #[test]
    fn test_line_plot_svg_contains_polyline() {
        let plugin = LinePlotPlugin;
        let data = VisualizationData::TimeSeries {
            timestamps: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            values: vec![0.1, 0.4, 0.9, 0.3, 0.7],
            labels: vec![],
        };
        let result = plugin.execute(data, svg_config()).expect("line plot SVG render failed");
        assert!(result.success);
        let bytes = result.output_data.expect("no output data");
        let svg = String::from_utf8(bytes).expect("invalid UTF-8");
        assert!(
            svg.contains("<polyline"),
            "SVG line plot should contain <polyline element"
        );
        assert!(svg.starts_with("<svg"));
    }

    #[test]
    fn test_scatter_svg_contains_circle() {
        let plugin = ScatterPlotPlugin;
        let data = VisualizationData::Array2D(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]);
        let result = plugin.execute(data, svg_config()).expect("scatter plot SVG render failed");
        assert!(result.success);
        let bytes = result.output_data.expect("no output data");
        let svg = String::from_utf8(bytes).expect("invalid UTF-8");
        assert!(
            svg.contains("<circle"),
            "SVG scatter plot should contain <circle elements"
        );
        assert!(svg.starts_with("<svg"));
    }

    #[test]
    fn test_output_path_writes_file() {
        let plugin = HistogramPlugin;
        let data = VisualizationData::Array1D(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let tmp_dir = std::env::temp_dir();
        let out_file = tmp_dir.join("test_histogram_visp.svg");
        let out_path_str = out_file.to_str().expect("temp path is not valid UTF-8").to_string();

        let config = PluginConfig {
            output_format: OutputFormat::Svg,
            output_path: Some(out_path_str.clone()),
            ..PluginConfig::default()
        };

        let result = plugin.execute(data, config).expect("histogram with output_path failed");
        assert!(result.success);
        assert_eq!(result.output_path.as_deref(), Some(out_path_str.as_str()));

        let written = std::fs::read_to_string(&out_file).expect("output file not found on disk");
        assert!(
            written.contains("<rect"),
            "Written SVG file should contain <rect"
        );

        // Cleanup
        std::fs::remove_file(&out_file).ok();
    }

    #[test]
    fn test_png_returns_error() {
        let plugin = HistogramPlugin;
        let data = VisualizationData::Array1D(vec![1.0, 2.0, 3.0]);
        let config = PluginConfig {
            output_format: OutputFormat::Png,
            ..PluginConfig::default()
        };
        let result = plugin.execute(data, config);
        assert!(result.is_err(), "PNG format should return an error, not Ok");
        let err = result.unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("png") || msg.contains("not supported"),
            "Error message should mention PNG or not supported; got: {}",
            err
        );
    }

    #[test]
    fn test_html_wraps_svg() {
        let plugin = ScatterPlotPlugin;
        let data = VisualizationData::Array2D(vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
        let config = PluginConfig {
            output_format: OutputFormat::Html,
            ..PluginConfig::default()
        };
        let result = plugin.execute(data, config).expect("HTML render failed");
        let bytes = result.output_data.expect("no output data");
        let html = String::from_utf8(bytes).expect("invalid UTF-8");
        assert!(
            html.contains("<!DOCTYPE html>"),
            "HTML output should contain DOCTYPE"
        );
        assert!(html.contains("<svg"), "HTML output should embed SVG");
    }

    #[test]
    fn test_histogram_json_output() {
        let plugin = HistogramPlugin;
        let data = VisualizationData::Array1D(vec![1.0, 2.0, 3.0, 4.0]);
        let config = PluginConfig {
            output_format: OutputFormat::Json,
            ..PluginConfig::default()
        };
        let result = plugin.execute(data, config).expect("JSON render failed");
        let bytes = result.output_data.expect("no output data");
        let json_str = String::from_utf8(bytes).expect("invalid UTF-8");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("output is not valid JSON");
        assert_eq!(parsed["type"], "histogram");
        assert!(parsed["bins"].is_array());
    }

    #[test]
    fn test_histogram_csv_output() {
        let plugin = HistogramPlugin;
        let data = VisualizationData::Array1D(vec![1.0, 2.0, 3.0, 4.0]);
        let config = PluginConfig {
            output_format: OutputFormat::Csv,
            ..PluginConfig::default()
        };
        let result = plugin.execute(data, config).expect("CSV render failed");
        let bytes = result.output_data.expect("no output data");
        let csv_str = String::from_utf8(bytes).expect("invalid UTF-8");
        assert!(
            csv_str.starts_with("bin_left,bin_right,count"),
            "CSV should start with header"
        );
    }

    #[test]
    fn test_config_schema_fields() {
        let hist_schema = HistogramPlugin.config_schema();
        assert!(hist_schema["width"].is_number());
        assert!(hist_schema["height"].is_number());

        let heat_schema = HeatmapPlugin.config_schema();
        assert!(heat_schema["width"].is_number());

        let line_schema = LinePlotPlugin.config_schema();
        assert!(line_schema["x_label"].is_string());

        let scatter_schema = ScatterPlotPlugin.config_schema();
        assert!(scatter_schema["title"].is_string());
    }
}
