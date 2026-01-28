//! Activation visualization tools for layer-wise debugging
//!
//! This module provides tools to inspect and visualize activations from different layers
//! of a neural network, including heatmaps, distributions, and statistical analysis.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// Note: scirs2_core types available for advanced operations if needed in future

/// Activation visualizer for inspecting layer outputs
#[derive(Debug)]
pub struct ActivationVisualizer {
    /// Stored activations by layer name
    activations: HashMap<String, ActivationData>,
    /// Configuration for visualization
    config: ActivationConfig,
}

/// Configuration for activation visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationConfig {
    /// Number of histogram bins
    pub num_bins: usize,
    /// Whether to compute detailed statistics
    pub detailed_stats: bool,
    /// Threshold for outlier detection (in standard deviations)
    pub outlier_threshold: f64,
    /// Maximum number of activations to store (to prevent memory overflow)
    pub max_stored_activations: usize,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            num_bins: 50,
            detailed_stats: true,
            outlier_threshold: 3.0,
            max_stored_activations: 10000,
        }
    }
}

/// Stored activation data for a layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationData {
    /// Layer name
    pub layer_name: String,
    /// Activation values (flattened)
    pub values: Vec<f32>,
    /// Original shape of the activation tensor
    pub shape: Vec<usize>,
    /// Statistics computed from the activations
    pub statistics: ActivationStatistics,
    /// Timestamp when captured
    pub timestamp: u64,
}

/// Statistical summary of activations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationStatistics {
    /// Mean activation value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Median value
    pub median: f64,
    /// 25th percentile
    pub q25: f64,
    /// 75th percentile
    pub q75: f64,
    /// Number of zero activations
    pub num_zeros: usize,
    /// Number of negative activations
    pub num_negative: usize,
    /// Number of positive activations
    pub num_positive: usize,
    /// Outlier count (values beyond threshold std devs)
    pub num_outliers: usize,
    /// Sparsity ratio (fraction of zeros)
    pub sparsity: f64,
}

/// Histogram data for activation distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationHistogram {
    /// Bin edges
    pub bin_edges: Vec<f64>,
    /// Bin counts
    pub bin_counts: Vec<usize>,
    /// Total count
    pub total_count: usize,
}

/// Heatmap data for 2D activation visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationHeatmap {
    /// Layer name
    pub layer_name: String,
    /// 2D values for heatmap
    pub values: Vec<Vec<f64>>,
    /// Row labels (optional)
    pub row_labels: Option<Vec<String>>,
    /// Column labels (optional)
    pub col_labels: Option<Vec<String>>,
}

impl ActivationVisualizer {
    /// Create a new activation visualizer
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::ActivationVisualizer;
    ///
    /// let visualizer = ActivationVisualizer::new();
    /// ```
    pub fn new() -> Self {
        Self {
            activations: HashMap::new(),
            config: ActivationConfig::default(),
        }
    }

    /// Create a new activation visualizer with custom configuration
    pub fn with_config(config: ActivationConfig) -> Self {
        Self {
            activations: HashMap::new(),
            config,
        }
    }

    /// Register activations from a layer
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Name of the layer
    /// * `values` - Flattened activation values
    /// * `shape` - Original shape of the activation tensor
    ///
    /// # Example
    ///
    /// ```
    /// # use trustformers_debug::ActivationVisualizer;
    /// # let mut visualizer = ActivationVisualizer::new();
    /// let activations = vec![0.1, 0.5, 0.3, 0.8];
    /// visualizer.register("layer1", activations, vec![2, 2]).unwrap();
    /// ```
    pub fn register(
        &mut self,
        layer_name: &str,
        values: Vec<f32>,
        shape: Vec<usize>,
    ) -> Result<()> {
        // Limit stored activations to prevent memory overflow
        let values = if values.len() > self.config.max_stored_activations {
            values.into_iter().take(self.config.max_stored_activations).collect()
        } else {
            values
        };

        let statistics = self.compute_statistics(&values)?;

        let timestamp =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();

        let activation_data = ActivationData {
            layer_name: layer_name.to_string(),
            values,
            shape,
            statistics,
            timestamp,
        };

        self.activations.insert(layer_name.to_string(), activation_data);
        Ok(())
    }

    /// Get activations for a specific layer
    pub fn get_activations(&self, layer_name: &str) -> Option<&ActivationData> {
        self.activations.get(layer_name)
    }

    /// Get all layer names with registered activations
    pub fn get_layer_names(&self) -> Vec<String> {
        self.activations.keys().cloned().collect()
    }

    /// Compute statistics for activation values
    fn compute_statistics(&self, values: &[f32]) -> Result<ActivationStatistics> {
        if values.is_empty() {
            return Ok(ActivationStatistics {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
                q25: 0.0,
                q75: 0.0,
                num_zeros: 0,
                num_negative: 0,
                num_positive: 0,
                num_outliers: 0,
                sparsity: 0.0,
            });
        }

        let mean: f64 = values.iter().map(|&x| x as f64).sum::<f64>() / values.len() as f64;

        let variance: f64 = values
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / values.len() as f64;

        let std_dev = variance.sqrt();

        let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b)) as f64;
        let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;

        // Count zeros, negatives, positives
        let num_zeros = values.iter().filter(|&&x| x.abs() < 1e-8).count();
        let num_negative = values.iter().filter(|&&x| x < 0.0).count();
        let num_positive = values.iter().filter(|&&x| x > 0.0).count();

        // Count outliers
        let num_outliers = values
            .iter()
            .filter(|&&x| (x as f64 - mean).abs() > self.config.outlier_threshold * std_dev)
            .count();

        // Compute percentiles
        let mut sorted_values: Vec<f32> = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = percentile(&sorted_values, 50.0);
        let q25 = percentile(&sorted_values, 25.0);
        let q75 = percentile(&sorted_values, 75.0);

        let sparsity = num_zeros as f64 / values.len() as f64;

        Ok(ActivationStatistics {
            mean,
            std_dev,
            min,
            max,
            median,
            q25,
            q75,
            num_zeros,
            num_negative,
            num_positive,
            num_outliers,
            sparsity,
        })
    }

    /// Create a histogram of activation values
    pub fn create_histogram(&self, layer_name: &str) -> Result<ActivationHistogram> {
        let activation = self
            .activations
            .get(layer_name)
            .ok_or_else(|| anyhow::anyhow!("Layer {} not found", layer_name))?;

        let min = activation.statistics.min;
        let max = activation.statistics.max;

        let bin_width = (max - min) / self.config.num_bins as f64;
        let mut bin_counts = vec![0; self.config.num_bins];

        for &value in &activation.values {
            let bin_idx = if bin_width > 0.0 {
                ((value as f64 - min) / bin_width).floor() as usize
            } else {
                0
            };
            let bin_idx = bin_idx.min(self.config.num_bins - 1);
            bin_counts[bin_idx] += 1;
        }

        let bin_edges: Vec<f64> =
            (0..=self.config.num_bins).map(|i| min + i as f64 * bin_width).collect();

        Ok(ActivationHistogram {
            bin_edges,
            bin_counts,
            total_count: activation.values.len(),
        })
    }

    /// Create a heatmap from 2D activations
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Name of the layer
    /// * `reshape` - Optional reshape dimensions (e.g., [height, width])
    pub fn create_heatmap(
        &self,
        layer_name: &str,
        reshape: Option<(usize, usize)>,
    ) -> Result<ActivationHeatmap> {
        let activation = self
            .activations
            .get(layer_name)
            .ok_or_else(|| anyhow::anyhow!("Layer {} not found", layer_name))?;

        let (rows, cols) = if let Some((r, c)) = reshape {
            (r, c)
        } else {
            // Try to infer 2D shape
            if activation.shape.len() >= 2 {
                let rows = activation.shape[activation.shape.len() - 2];
                let cols = activation.shape[activation.shape.len() - 1];
                (rows, cols)
            } else {
                // Fallback: make it as square as possible
                let total = activation.values.len();
                let cols = (total as f64).sqrt().ceil() as usize;
                let rows = total.div_ceil(cols);
                (rows, cols)
            }
        };

        let mut values = vec![vec![0.0; cols]; rows];
        for (i, &val) in activation.values.iter().enumerate().take(rows * cols) {
            let row = i / cols;
            let col = i % cols;
            if row < rows {
                values[row][col] = val as f64;
            }
        }

        Ok(ActivationHeatmap {
            layer_name: layer_name.to_string(),
            values,
            row_labels: None,
            col_labels: None,
        })
    }

    /// Export activation statistics to JSON
    pub fn export_statistics(&self, layer_name: &str, output_path: &Path) -> Result<()> {
        let activation = self
            .activations
            .get(layer_name)
            .ok_or_else(|| anyhow::anyhow!("Layer {} not found", layer_name))?;

        let json = serde_json::to_string_pretty(&activation.statistics)?;
        std::fs::write(output_path, json)?;

        Ok(())
    }

    /// Plot distribution as ASCII histogram
    pub fn plot_distribution_ascii(&self, layer_name: &str) -> Result<String> {
        let histogram = self.create_histogram(layer_name)?;

        let max_count = histogram.bin_counts.iter().max().unwrap_or(&0);
        let scale = if *max_count > 0 { 50.0 / *max_count as f64 } else { 1.0 };

        let mut output = String::new();
        output.push_str(&format!("Activation Distribution: {}\n", layer_name));
        output.push_str(&"=".repeat(60));
        output.push('\n');

        for i in 0..histogram.bin_counts.len() {
            let bar_length = (histogram.bin_counts[i] as f64 * scale) as usize;
            let bar = "█".repeat(bar_length);
            output.push_str(&format!(
                "{:8.3} - {:8.3} | {} ({})\n",
                histogram.bin_edges[i],
                histogram.bin_edges[i + 1],
                bar,
                histogram.bin_counts[i]
            ));
        }

        Ok(output)
    }

    /// Print summary statistics for all layers
    pub fn print_summary(&self) -> Result<String> {
        let mut output = String::new();
        output.push_str("Activation Summary\n");
        output.push_str(&"=".repeat(80));
        output.push('\n');

        for (layer_name, activation) in &self.activations {
            output.push_str(&format!("\nLayer: {}\n", layer_name));
            output.push_str(&format!("  Shape: {:?}\n", activation.shape));
            output.push_str(&format!("  Mean: {:.6}\n", activation.statistics.mean));
            output.push_str(&format!(
                "  Std Dev: {:.6}\n",
                activation.statistics.std_dev
            ));
            output.push_str(&format!("  Min: {:.6}\n", activation.statistics.min));
            output.push_str(&format!("  Max: {:.6}\n", activation.statistics.max));
            output.push_str(&format!("  Median: {:.6}\n", activation.statistics.median));
            output.push_str(&format!(
                "  Sparsity: {:.2}%\n",
                activation.statistics.sparsity * 100.0
            ));
            output.push_str(&format!(
                "  Outliers: {} ({:.2}%)\n",
                activation.statistics.num_outliers,
                activation.statistics.num_outliers as f64 / activation.values.len() as f64 * 100.0
            ));
        }

        Ok(output)
    }

    /// Clear all stored activations
    pub fn clear(&mut self) {
        self.activations.clear();
    }

    /// Get number of stored activations
    pub fn num_layers(&self) -> usize {
        self.activations.len()
    }
}

impl Default for ActivationVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to compute percentile
fn percentile(sorted_values: &[f32], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    let index = (p / 100.0 * (sorted_values.len() - 1) as f64).round() as usize;
    sorted_values[index.min(sorted_values.len() - 1)] as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_activation_visualizer_creation() {
        let visualizer = ActivationVisualizer::new();
        assert_eq!(visualizer.num_layers(), 0);
    }

    #[test]
    fn test_register_activations() {
        let mut visualizer = ActivationVisualizer::new();
        let values = vec![0.1, 0.5, 0.3, 0.8, -0.2];

        visualizer.register("layer1", values.clone(), vec![5]).unwrap();
        assert_eq!(visualizer.num_layers(), 1);

        let activation = visualizer.get_activations("layer1").unwrap();
        assert_eq!(activation.values, values);
        assert_eq!(activation.shape, vec![5]);
    }

    #[test]
    fn test_compute_statistics() {
        let visualizer = ActivationVisualizer::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let stats = visualizer.compute_statistics(&values).unwrap();
        assert_eq!(stats.mean, 3.0);
        assert!(stats.std_dev > 0.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.num_zeros, 0);
        assert_eq!(stats.num_positive, 5);
    }

    #[test]
    fn test_create_histogram() {
        let mut visualizer = ActivationVisualizer::new();
        let values: Vec<f32> = (0..100).map(|x| x as f32).collect();

        visualizer.register("layer1", values, vec![100]).unwrap();

        let histogram = visualizer.create_histogram("layer1").unwrap();
        assert_eq!(histogram.bin_edges.len(), visualizer.config.num_bins + 1);
        assert_eq!(histogram.total_count, 100);
    }

    #[test]
    fn test_create_heatmap() {
        let mut visualizer = ActivationVisualizer::new();
        let values: Vec<f32> = (0..16).map(|x| x as f32).collect();

        visualizer.register("layer1", values, vec![4, 4]).unwrap();

        let heatmap = visualizer.create_heatmap("layer1", Some((4, 4))).unwrap();
        assert_eq!(heatmap.values.len(), 4);
        assert_eq!(heatmap.values[0].len(), 4);
    }

    #[test]
    fn test_export_statistics() {
        let temp_dir = env::temp_dir();
        let output_path = temp_dir.join("activation_stats.json");

        let mut visualizer = ActivationVisualizer::new();
        let values = vec![1.0, 2.0, 3.0];

        visualizer.register("layer1", values, vec![3]).unwrap();
        visualizer.export_statistics("layer1", &output_path).unwrap();

        assert!(output_path.exists());

        // Clean up
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn test_plot_distribution_ascii() {
        let mut visualizer = ActivationVisualizer::new();
        let values: Vec<f32> = (0..100).map(|x| x as f32 / 100.0).collect();

        visualizer.register("layer1", values, vec![100]).unwrap();

        let ascii_plot = visualizer.plot_distribution_ascii("layer1").unwrap();
        assert!(ascii_plot.contains("Activation Distribution"));
        assert!(ascii_plot.contains("layer1"));
    }

    #[test]
    fn test_print_summary() {
        let mut visualizer = ActivationVisualizer::new();

        visualizer.register("layer1", vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        visualizer.register("layer2", vec![4.0, 5.0, 6.0], vec![3]).unwrap();

        let summary = visualizer.print_summary().unwrap();
        assert!(summary.contains("layer1"));
        assert!(summary.contains("layer2"));
        assert!(summary.contains("Mean"));
        assert!(summary.contains("Std Dev"));
    }

    #[test]
    fn test_sparsity_calculation() {
        let visualizer = ActivationVisualizer::new();
        let values = vec![0.0, 0.0, 0.0, 1.0, 0.0];

        let stats = visualizer.compute_statistics(&values).unwrap();
        assert_eq!(stats.num_zeros, 4);
        assert_eq!(stats.sparsity, 0.8);
    }

    #[test]
    fn test_clear_activations() {
        let mut visualizer = ActivationVisualizer::new();

        visualizer.register("layer1", vec![1.0], vec![1]).unwrap();
        visualizer.register("layer2", vec![2.0], vec![1]).unwrap();

        assert_eq!(visualizer.num_layers(), 2);

        visualizer.clear();
        assert_eq!(visualizer.num_layers(), 0);
    }
}
