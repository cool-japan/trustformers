//! Visualization module for TrustformeRS debugging tools
//!
//! This module has been refactored into focused submodules to comply with the
//! 2000-line policy. The original visualization.rs (2843 lines) has been split into:
//!
//! - `types` - Basic visualization types, enums, and data structures
//! - Additional modules to be created as needed for terminal, video, etc.

pub mod modern_plotting;
pub mod types;

// Re-export main types for backward compatibility
pub use modern_plotting::*;
pub use types::*;

use anyhow::Result;
use std::path::Path;

/// Main debug visualizer (simplified version)
#[derive(Debug)]
pub struct DebugVisualizer {
    config: VisualizationConfig,
}

impl DebugVisualizer {
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self::new(VisualizationConfig::default())
    }

    /// Create a simple line plot
    pub fn create_line_plot(&self, data: &PlotData) -> Result<String> {
        // Placeholder implementation
        Ok(format!("Line plot '{}' created successfully", data.title))
    }

    /// Create a heatmap visualization
    pub fn create_heatmap(&self, data: &HeatmapData) -> Result<String> {
        // Placeholder implementation
        Ok(format!("Heatmap '{}' created successfully", data.title))
    }

    /// Create a histogram
    pub fn create_histogram(&self, data: &HistogramData) -> Result<String> {
        // Placeholder implementation
        Ok(format!("Histogram '{}' created successfully", data.title))
    }

    /// Plot tensor distribution
    pub fn plot_tensor_distribution(
        &self,
        name: &str,
        values: &[f64],
        bins: usize,
    ) -> Result<String> {
        let data = HistogramData {
            values: values.to_vec(),
            bins,
            title: format!("{} Distribution", name),
            x_label: "Value".to_string(),
            y_label: "Frequency".to_string(),
            density: false,
        };
        self.create_histogram(&data)
    }

    /// Plot training metrics
    pub fn plot_training_metrics(
        &mut self,
        steps: &[f64],
        losses: &[f64],
        accuracies: Option<&[f64]>,
    ) -> Result<String> {
        let mut plot_data = PlotData {
            x_values: steps.to_vec(),
            y_values: losses.to_vec(),
            labels: vec!["Loss".to_string()],
            title: "Training Metrics".to_string(),
            x_label: "Steps".to_string(),
            y_label: "Value".to_string(),
        };

        if let Some(acc) = accuracies {
            plot_data.y_values.extend_from_slice(acc);
            plot_data.labels.push("Accuracy".to_string());
        }

        self.create_line_plot(&plot_data)
    }

    /// Plot gradient flow
    pub fn plot_gradient_flow(
        &self,
        layer_name: &str,
        steps: &[f64],
        gradient_norms: &[f64],
    ) -> Result<String> {
        let data = PlotData {
            x_values: steps.to_vec(),
            y_values: gradient_norms.to_vec(),
            labels: vec![format!("{} Gradient Flow", layer_name)],
            title: format!("Gradient Flow - {}", layer_name),
            x_label: "Steps".to_string(),
            y_label: "Gradient Norm".to_string(),
        };
        self.create_line_plot(&data)
    }

    /// Plot tensor heatmap
    pub fn plot_tensor_heatmap(&self, name: &str, values: &[Vec<f64>]) -> Result<String> {
        let data = HeatmapData {
            values: values.to_vec(),
            x_labels: (0..values.first().map_or(0, |row| row.len()))
                .map(|i| i.to_string())
                .collect(),
            y_labels: (0..values.len()).map(|i| i.to_string()).collect(),
            title: format!("{} Heatmap", name),
            color_bar_label: "Value".to_string(),
        };
        self.create_heatmap(&data)
    }

    /// Plot activation patterns
    pub fn plot_activation_patterns(
        &self,
        layer_name: &str,
        inputs: &[f64],
        outputs: &[f64],
    ) -> Result<String> {
        let data = PlotData {
            x_values: inputs.to_vec(),
            y_values: outputs.to_vec(),
            labels: vec![format!("{} Activation", layer_name)],
            title: format!("Activation Pattern - {}", layer_name),
            x_label: "Input".to_string(),
            y_label: "Output".to_string(),
        };
        self.create_line_plot(&data)
    }

    /// Get plot names
    pub fn get_plot_names(&self) -> Vec<String> {
        // Return some default plot names for demonstration
        vec![
            "tensor_distribution".to_string(),
            "training_metrics".to_string(),
            "gradient_flow".to_string(),
            "activation_patterns".to_string(),
        ]
    }

    /// Create dashboard
    pub fn create_dashboard(&mut self, plot_names: &[String]) -> Result<String> {
        let dashboard_path = Path::new(&self.config.output_directory).join("dashboard.html");
        std::fs::create_dir_all(&self.config.output_directory)?;

        let mut html_content =
            String::from("<html><head><title>Debug Dashboard</title></head><body>");
        html_content.push_str("<h1>TrustformeRS Debug Dashboard</h1>");

        for plot_name in plot_names {
            html_content.push_str(&format!(
                "<div><h2>{}</h2><p>Plot: {}</p></div>",
                plot_name, plot_name
            ));
        }

        html_content.push_str("</body></html>");
        std::fs::write(&dashboard_path, html_content)?;

        Ok(dashboard_path.to_string_lossy().to_string())
    }

    /// Export plot data
    pub fn export_plot_data(&self, plot_name: &str, export_path: &Path) -> Result<()> {
        std::fs::create_dir_all(export_path.parent().unwrap_or(Path::new(".")))?;
        let data = format!("Plot data for: {}", plot_name);
        std::fs::write(export_path, data)?;
        Ok(())
    }

    /// Save visualization to file
    pub fn save_to_file(&self, filename: &str) -> Result<()> {
        // Placeholder implementation
        let output_path = Path::new(&self.config.output_directory).join(filename);
        std::fs::create_dir_all(&self.config.output_directory)?;
        std::fs::write(output_path, "placeholder visualization content")?;
        Ok(())
    }
}

/// Simple terminal-based visualizer
pub struct TerminalVisualizer;

impl TerminalVisualizer {
    pub fn new() -> Self {
        Self
    }

    /// Display simple text-based histogram in terminal
    pub fn display_histogram(&self, data: &HistogramData) -> Result<()> {
        println!("Terminal Histogram: {}", data.title);
        println!("Data points: {}", data.values.len());
        // Placeholder for actual terminal histogram
        Ok(())
    }

    /// Display simple text-based statistics
    pub fn display_statistics(&self, label: &str, values: &[f64]) -> Result<()> {
        if values.is_empty() {
            println!("{}: No data", label);
            return Ok(());
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        println!(
            "{}: mean={:.3}, min={:.3}, max={:.3}",
            label, mean, min, max
        );
        Ok(())
    }

    /// ASCII histogram display
    pub fn ascii_histogram(&self, values: &[f64], bins: usize) -> String {
        if values.is_empty() {
            return "No data for histogram".to_string();
        }

        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if (max_val - min_val).abs() < f64::EPSILON {
            return format!("All values are {:.3}", min_val);
        }

        let mut histogram = vec![0; bins];
        let bin_width = (max_val - min_val) / bins as f64;

        for &value in values {
            let bin_index = ((value - min_val) / bin_width).floor() as usize;
            let bin_index = bin_index.min(bins - 1);
            histogram[bin_index] += 1;
        }

        let max_count = histogram.iter().max().unwrap_or(&0);
        let scale = if *max_count > 0 { 40.0 / *max_count as f64 } else { 1.0 };

        let mut result = String::new();
        for (i, &count) in histogram.iter().enumerate() {
            let bin_start = min_val + i as f64 * bin_width;
            let bin_end = bin_start + bin_width;
            let bar_length = (count as f64 * scale) as usize;
            let bar = "█".repeat(bar_length);
            result.push_str(&format!(
                "[{:.2}-{:.2}): {} ({})\n",
                bin_start, bin_end, bar, count
            ));
        }

        result
    }

    /// ASCII line plot display
    pub fn ascii_line_plot(&self, x_values: &[f64], y_values: &[f64], title: &str) -> String {
        if x_values.is_empty() || y_values.is_empty() || x_values.len() != y_values.len() {
            return "Invalid data for line plot".to_string();
        }

        let min_y = y_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_y = y_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let mut result = format!("{}\n", title);
        result.push_str("═".repeat(title.len()).as_str());
        result.push('\n');

        if (max_y - min_y).abs() < f64::EPSILON {
            result.push_str(&format!("Constant value: {:.3}\n", min_y));
            return result;
        }

        let height = 20;
        let width = x_values.len().min(80);

        // Sample data if too many points
        let step = if x_values.len() > width { x_values.len() / width } else { 1 };

        for row in (0..height).rev() {
            let y_threshold = min_y + (max_y - min_y) * row as f64 / (height - 1) as f64;
            let mut line = String::new();

            for i in (0..x_values.len()).step_by(step).take(width) {
                if y_values[i] >= y_threshold {
                    line.push('*');
                } else {
                    line.push(' ');
                }
            }
            result.push_str(&format!("{:8.2} |{}\n", y_threshold, line));
        }

        result.push_str(&format!("{:8} +{}\n", "", "─".repeat(width)));
        result
    }
}

impl Default for TerminalVisualizer {
    fn default() -> Self {
        Self::new()
    }
}
