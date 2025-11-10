//! Weight distribution analysis tools
//!
//! This module provides tools to analyze weight distributions in neural networks,
//! including dead neuron detection, initialization validation, and statistical analysis.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Weight distribution analyzer for model inspection
#[derive(Debug)]
pub struct WeightAnalyzer {
    /// Stored weight analyses by layer name
    analyses: HashMap<String, WeightAnalysis>,
    /// Configuration
    config: WeightAnalyzerConfig,
}

/// Configuration for weight analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightAnalyzerConfig {
    /// Threshold for dead neuron detection (absolute value)
    pub dead_neuron_threshold: f64,
    /// Number of histogram bins
    pub num_bins: usize,
    /// Check for initialization issues
    pub check_initialization: bool,
    /// Expected initialization schemes to validate against
    pub expected_init_schemes: Vec<InitializationScheme>,
}

impl Default for WeightAnalyzerConfig {
    fn default() -> Self {
        Self {
            dead_neuron_threshold: 1e-8,
            num_bins: 50,
            check_initialization: true,
            expected_init_schemes: vec![
                InitializationScheme::XavierUniform,
                InitializationScheme::HeNormal,
            ],
        }
    }
}

/// Initialization schemes for validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InitializationScheme {
    /// Xavier/Glorot uniform initialization
    XavierUniform,
    /// Xavier/Glorot normal initialization
    XavierNormal,
    /// He uniform initialization
    HeUniform,
    /// He normal initialization
    HeNormal,
    /// LeCun normal initialization
    LeCunNormal,
    /// Orthogonal initialization
    Orthogonal,
    /// Uniform initialization
    Uniform,
    /// Normal initialization
    Normal,
}

/// Analysis results for a layer's weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightAnalysis {
    /// Layer name
    pub layer_name: String,
    /// Weight statistics
    pub statistics: WeightStatistics,
    /// Dead neurons detected
    pub dead_neurons: Vec<usize>,
    /// Histogram data
    pub histogram: WeightHistogram,
    /// Likely initialization scheme
    pub likely_init_scheme: Option<InitializationScheme>,
    /// Initialization warnings
    pub init_warnings: Vec<String>,
}

/// Statistical summary of weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightStatistics {
    /// Mean weight value
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
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
    /// L1 norm
    pub l1_norm: f64,
    /// L2 norm
    pub l2_norm: f64,
    /// Number of zero weights
    pub num_zeros: usize,
    /// Sparsity ratio
    pub sparsity: f64,
}

/// Histogram of weight distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightHistogram {
    /// Bin edges
    pub bin_edges: Vec<f64>,
    /// Bin counts
    pub bin_counts: Vec<usize>,
    /// Total count
    pub total_count: usize,
}

impl WeightAnalyzer {
    /// Create a new weight analyzer
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::WeightAnalyzer;
    ///
    /// let analyzer = WeightAnalyzer::new();
    /// ```
    pub fn new() -> Self {
        Self {
            analyses: HashMap::new(),
            config: WeightAnalyzerConfig::default(),
        }
    }

    /// Create a weight analyzer with custom configuration
    pub fn with_config(config: WeightAnalyzerConfig) -> Self {
        Self {
            analyses: HashMap::new(),
            config,
        }
    }

    /// Analyze weights from a layer
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Name of the layer
    /// * `weights` - Weight values
    ///
    /// # Example
    ///
    /// ```
    /// # use trustformers_debug::WeightAnalyzer;
    /// # let mut analyzer = WeightAnalyzer::new();
    /// let weights = vec![0.1, 0.2, 0.05, 0.15, 0.3];
    /// let analysis = analyzer.analyze("layer1", &weights).unwrap();
    /// ```
    pub fn analyze(&mut self, layer_name: &str, weights: &[f64]) -> Result<&WeightAnalysis> {
        let statistics = self.compute_statistics(weights)?;
        let dead_neurons = self.detect_dead_neurons(weights);
        let histogram = self.compute_histogram(weights)?;
        let (likely_init_scheme, init_warnings) = if self.config.check_initialization {
            self.check_initialization(&statistics)
        } else {
            (None, Vec::new())
        };

        let analysis = WeightAnalysis {
            layer_name: layer_name.to_string(),
            statistics,
            dead_neurons,
            histogram,
            likely_init_scheme,
            init_warnings,
        };

        self.analyses.insert(layer_name.to_string(), analysis);
        Ok(self.analyses.get(layer_name).unwrap())
    }

    /// Compute statistics for weights
    fn compute_statistics(&self, weights: &[f64]) -> Result<WeightStatistics> {
        if weights.is_empty() {
            anyhow::bail!("Cannot compute statistics for empty weight array");
        }

        let n = weights.len() as f64;
        let mean = weights.iter().sum::<f64>() / n;

        let variance = weights
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt();

        let mut sorted = weights.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let median = percentile(&sorted, 50.0);
        let q25 = percentile(&sorted, 25.0);
        let q75 = percentile(&sorted, 75.0);

        // Compute skewness
        let skewness = if std_dev > 0.0 {
            weights
                .iter()
                .map(|&x| {
                    let z = (x - mean) / std_dev;
                    z * z * z
                })
                .sum::<f64>()
                / n
        } else {
            0.0
        };

        // Compute kurtosis
        let kurtosis = if std_dev > 0.0 {
            weights
                .iter()
                .map(|&x| {
                    let z = (x - mean) / std_dev;
                    z * z * z * z
                })
                .sum::<f64>()
                / n
                - 3.0
        } else {
            0.0
        };

        let l1_norm = weights.iter().map(|x| x.abs()).sum::<f64>();
        let l2_norm = weights.iter().map(|x| x * x).sum::<f64>().sqrt();

        let num_zeros = weights.iter().filter(|&&x| x.abs() < 1e-10).count();
        let sparsity = num_zeros as f64 / n;

        Ok(WeightStatistics {
            mean,
            std_dev,
            min,
            max,
            median,
            q25,
            q75,
            skewness,
            kurtosis,
            l1_norm,
            l2_norm,
            num_zeros,
            sparsity,
        })
    }

    /// Detect dead neurons (weights close to zero)
    fn detect_dead_neurons(&self, weights: &[f64]) -> Vec<usize> {
        weights
            .iter()
            .enumerate()
            .filter_map(
                |(i, &w)| {
                    if w.abs() < self.config.dead_neuron_threshold {
                        Some(i)
                    } else {
                        None
                    }
                },
            )
            .collect()
    }

    /// Compute histogram of weight distribution
    fn compute_histogram(&self, weights: &[f64]) -> Result<WeightHistogram> {
        if weights.is_empty() {
            anyhow::bail!("Cannot compute histogram for empty weight array");
        }

        let min = weights.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = weights.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let bin_width = (max - min) / self.config.num_bins as f64;
        let mut bin_counts = vec![0; self.config.num_bins];

        for &weight in weights {
            let bin_idx =
                if bin_width > 0.0 { ((weight - min) / bin_width).floor() as usize } else { 0 };
            let bin_idx = bin_idx.min(self.config.num_bins - 1);
            bin_counts[bin_idx] += 1;
        }

        let bin_edges: Vec<f64> =
            (0..=self.config.num_bins).map(|i| min + i as f64 * bin_width).collect();

        Ok(WeightHistogram {
            bin_edges,
            bin_counts,
            total_count: weights.len(),
        })
    }

    /// Check initialization scheme and detect issues
    fn check_initialization(
        &self,
        stats: &WeightStatistics,
    ) -> (Option<InitializationScheme>, Vec<String>) {
        let mut warnings = Vec::new();
        let mut likely_scheme = None;

        // Check if weights are all zero (not initialized)
        if stats.sparsity > 0.99 {
            warnings.push("Weights appear to be uninitialized (all zeros)".to_string());
            return (None, warnings);
        }

        // Check if weights are too large
        if stats.std_dev > 1.0 {
            warnings.push(format!(
                "Weights have high variance (std_dev={:.4}), may cause gradient explosion",
                stats.std_dev
            ));
        }

        // Check if weights are too small
        if stats.std_dev < 0.001 {
            warnings.push(format!(
                "Weights have very low variance (std_dev={:.4}), may cause gradient vanishing",
                stats.std_dev
            ));
        }

        // Infer initialization scheme based on statistics
        // Xavier: mean ~ 0, std ~ sqrt(2 / (fan_in + fan_out))
        // He: mean ~ 0, std ~ sqrt(2 / fan_in)
        // Normal: mean ~ 0, std ~ some constant

        if stats.mean.abs() < 0.01 {
            // Mean is close to zero, likely a good initialization
            if stats.std_dev > 0.01 && stats.std_dev < 0.2 {
                // Check distribution shape
                if stats.skewness.abs() < 0.5 && stats.kurtosis.abs() < 1.0 {
                    likely_scheme = Some(InitializationScheme::XavierNormal);
                } else {
                    likely_scheme = Some(InitializationScheme::Normal);
                }
            } else if stats.std_dev < 0.01 {
                likely_scheme = Some(InitializationScheme::Uniform);
            }
        }

        (likely_scheme, warnings)
    }

    /// Get analysis for a specific layer
    pub fn get_analysis(&self, layer_name: &str) -> Option<&WeightAnalysis> {
        self.analyses.get(layer_name)
    }

    /// Get all layer names with analyses
    pub fn get_layer_names(&self) -> Vec<String> {
        self.analyses.keys().cloned().collect()
    }

    /// Print summary of all analyses
    pub fn print_summary(&self) -> String {
        let mut output = String::new();
        output.push_str("Weight Distribution Summary\n");
        output.push_str(&"=".repeat(80));
        output.push('\n');

        for (layer_name, analysis) in &self.analyses {
            output.push_str(&format!("\nLayer: {}\n", layer_name));
            output.push_str(&format!("  Mean: {:.6}\n", analysis.statistics.mean));
            output.push_str(&format!("  Std Dev: {:.6}\n", analysis.statistics.std_dev));
            output.push_str(&format!(
                "  Range: [{:.6}, {:.6}]\n",
                analysis.statistics.min, analysis.statistics.max
            ));
            output.push_str(&format!("  Median: {:.6}\n", analysis.statistics.median));
            output.push_str(&format!(
                "  Sparsity: {:.2}%\n",
                analysis.statistics.sparsity * 100.0
            ));
            output.push_str(&format!(
                "  Dead Neurons: {} ({:.2}%)\n",
                analysis.dead_neurons.len(),
                analysis.dead_neurons.len() as f64 / analysis.histogram.total_count as f64 * 100.0
            ));

            if let Some(scheme) = analysis.likely_init_scheme {
                output.push_str(&format!("  Likely Init: {:?}\n", scheme));
            }

            if !analysis.init_warnings.is_empty() {
                output.push_str("  Warnings:\n");
                for warning in &analysis.init_warnings {
                    output.push_str(&format!("    - {}\n", warning));
                }
            }
        }

        output
    }

    /// Export analysis to JSON
    pub fn export_to_json(&self, layer_name: &str, output_path: &Path) -> Result<()> {
        let analysis = self
            .analyses
            .get(layer_name)
            .ok_or_else(|| anyhow::anyhow!("Layer {} not found", layer_name))?;

        let json = serde_json::to_string_pretty(analysis)?;
        std::fs::write(output_path, json)?;

        Ok(())
    }

    /// Plot weight distribution as ASCII histogram
    pub fn plot_distribution_ascii(&self, layer_name: &str) -> Result<String> {
        let analysis = self
            .analyses
            .get(layer_name)
            .ok_or_else(|| anyhow::anyhow!("Layer {} not found", layer_name))?;

        let histogram = &analysis.histogram;
        let max_count = histogram.bin_counts.iter().max().unwrap_or(&0);
        let scale = if *max_count > 0 { 50.0 / *max_count as f64 } else { 1.0 };

        let mut output = String::new();
        output.push_str(&format!("Weight Distribution: {}\n", layer_name));
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

        output.push_str(&format!("\nStatistics:\n"));
        output.push_str(&format!("  Mean: {:.6}\n", analysis.statistics.mean));
        output.push_str(&format!("  Std Dev: {:.6}\n", analysis.statistics.std_dev));
        output.push_str(&format!(
            "  Skewness: {:.6}\n",
            analysis.statistics.skewness
        ));
        output.push_str(&format!(
            "  Kurtosis: {:.6}\n",
            analysis.statistics.kurtosis
        ));

        Ok(output)
    }

    /// Clear all stored analyses
    pub fn clear(&mut self) {
        self.analyses.clear();
    }

    /// Get number of analyzed layers
    pub fn num_layers(&self) -> usize {
        self.analyses.len()
    }
}

impl Default for WeightAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to compute percentile
fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    let index = (p / 100.0 * (sorted_values.len() - 1) as f64).round() as usize;
    sorted_values[index.min(sorted_values.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_weight_analyzer_creation() {
        let analyzer = WeightAnalyzer::new();
        assert_eq!(analyzer.num_layers(), 0);
    }

    #[test]
    fn test_analyze_weights() {
        let mut analyzer = WeightAnalyzer::new();
        let weights = vec![0.1, 0.2, 0.15, 0.3, 0.25];

        let analysis = analyzer.analyze("layer1", &weights).unwrap();
        assert_eq!(analysis.layer_name, "layer1");
        assert!(analysis.statistics.mean > 0.0);
        assert!(analysis.statistics.std_dev > 0.0);
    }

    #[test]
    fn test_dead_neuron_detection() {
        let mut analyzer = WeightAnalyzer::new();
        let weights = vec![0.1, 0.0, 0.2, 0.0, 0.3]; // Two dead neurons

        let analysis = analyzer.analyze("layer1", &weights).unwrap();
        assert_eq!(analysis.dead_neurons.len(), 2);
    }

    #[test]
    fn test_compute_histogram() {
        let analyzer = WeightAnalyzer::new();
        let weights: Vec<f64> = (0..100).map(|x| x as f64 / 100.0).collect();

        let histogram = analyzer.compute_histogram(&weights).unwrap();
        assert_eq!(histogram.bin_edges.len(), analyzer.config.num_bins + 1);
        assert_eq!(histogram.total_count, 100);
    }

    #[test]
    fn test_weight_statistics() {
        let analyzer = WeightAnalyzer::new();
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let stats = analyzer.compute_statistics(&weights).unwrap();
        assert_eq!(stats.mean, 3.0);
        assert!(stats.std_dev > 0.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_initialization_check() {
        let analyzer = WeightAnalyzer::new();

        // Simulate Xavier-like initialization
        let stats = WeightStatistics {
            mean: 0.001,
            std_dev: 0.05,
            min: -0.15,
            max: 0.15,
            median: 0.0,
            q25: -0.03,
            q75: 0.03,
            skewness: 0.1,
            kurtosis: 0.2,
            l1_norm: 10.0,
            l2_norm: 5.0,
            num_zeros: 0,
            sparsity: 0.0,
        };

        let (scheme, warnings) = analyzer.check_initialization(&stats);
        assert!(scheme.is_some());
        assert!(warnings.is_empty() || warnings.len() <= 1);
    }

    #[test]
    fn test_export_to_json() {
        let temp_dir = env::temp_dir();
        let output_path = temp_dir.join("weight_analysis.json");

        let mut analyzer = WeightAnalyzer::new();
        analyzer.analyze("layer1", &vec![1.0, 2.0, 3.0]).unwrap();

        analyzer.export_to_json("layer1", &output_path).unwrap();
        assert!(output_path.exists());

        // Clean up
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn test_plot_distribution_ascii() {
        let mut analyzer = WeightAnalyzer::new();
        let weights: Vec<f64> = (0..100).map(|x| x as f64 / 100.0).collect();

        analyzer.analyze("layer1", &weights).unwrap();

        let ascii_plot = analyzer.plot_distribution_ascii("layer1").unwrap();
        assert!(ascii_plot.contains("Weight Distribution"));
        assert!(ascii_plot.contains("layer1"));
        assert!(ascii_plot.contains("Statistics"));
    }

    #[test]
    fn test_print_summary() {
        let mut analyzer = WeightAnalyzer::new();

        analyzer.analyze("layer1", &vec![1.0, 2.0, 3.0]).unwrap();
        analyzer.analyze("layer2", &vec![0.5, 1.0, 1.5]).unwrap();

        let summary = analyzer.print_summary();
        assert!(summary.contains("layer1"));
        assert!(summary.contains("layer2"));
        assert!(summary.contains("Mean"));
        assert!(summary.contains("Std Dev"));
    }

    #[test]
    fn test_sparsity_calculation() {
        let analyzer = WeightAnalyzer::new();
        let weights = vec![0.0, 0.0, 0.0, 1.0, 0.0];

        let stats = analyzer.compute_statistics(&weights).unwrap();
        assert_eq!(stats.num_zeros, 4);
        assert_eq!(stats.sparsity, 0.8);
    }

    #[test]
    fn test_clear_analyses() {
        let mut analyzer = WeightAnalyzer::new();

        analyzer.analyze("layer1", &vec![1.0]).unwrap();
        analyzer.analyze("layer2", &vec![2.0]).unwrap();

        assert_eq!(analyzer.num_layers(), 2);

        analyzer.clear();
        assert_eq!(analyzer.num_layers(), 0);
    }
}
