// Gradient flow analysis for debugging vanishing/exploding gradients
use crate::tensor::Tensor;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Gradient flow analyzer for tracking gradient statistics through backpropagation
#[derive(Debug, Clone)]
pub struct GradientFlowAnalyzer {
    config: GradientFlowConfig,
    layer_stats: HashMap<String, LayerGradientStats>,
    history: Vec<GradientSnapshot>,
    current_step: u64,
}

/// Configuration for gradient flow analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFlowConfig {
    /// Track gradient magnitudes
    pub track_magnitudes: bool,
    /// Track gradient distributions
    pub track_distributions: bool,
    /// Threshold for detecting vanishing gradients
    pub vanishing_threshold: f32,
    /// Threshold for detecting exploding gradients
    pub exploding_threshold: f32,
    /// Number of histogram bins for gradient distributions
    pub histogram_bins: usize,
    /// Maximum history to keep
    pub max_history: usize,
}

impl Default for GradientFlowConfig {
    fn default() -> Self {
        Self {
            track_magnitudes: true,
            track_distributions: true,
            vanishing_threshold: 1e-6,
            exploding_threshold: 100.0,
            histogram_bins: 50,
            max_history: 1000,
        }
    }
}

/// Gradient statistics for a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerGradientStats {
    pub layer_name: String,
    pub magnitude_stats: GradientMagnitudeStats,
    pub distribution_stats: Option<GradientDistributionStats>,
    pub flow_health: GradientFlowHealth,
    pub update_count: u64,
}

/// Gradient magnitude statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientMagnitudeStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub l1_norm: f32,
    pub l2_norm: f32,
    pub max_norm: f32,
    pub rms: f32,
}

/// Gradient distribution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientDistributionStats {
    pub histogram: Vec<u32>,
    pub bin_edges: Vec<f32>,
    pub skewness: f32,
    pub kurtosis: f32,
    pub percentiles: Vec<f32>, // P5, P25, P50, P75, P95
}

/// Gradient flow health assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFlowHealth {
    pub is_vanishing: bool,
    pub is_exploding: bool,
    pub vanishing_ratio: f32,
    pub exploding_ratio: f32,
    pub flow_score: f32, // 0.0 (poor) to 1.0 (excellent)
    pub recommendations: Vec<String>,
}

/// Snapshot of gradient flow at a specific step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientSnapshot {
    pub step: u64,
    pub timestamp: f64,
    pub layer_stats: HashMap<String, LayerGradientStats>,
    pub global_stats: GlobalGradientStats,
}

/// Global gradient statistics across all layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalGradientStats {
    pub total_parameters: usize,
    pub parameters_with_gradients: usize,
    pub dead_parameters: usize, // Parameters with zero gradients
    pub avg_gradient_norm: f32,
    pub max_layer_norm: f32,
    pub min_layer_norm: f32,
    pub gradient_variance_across_layers: f32,
}

impl Default for GradientFlowAnalyzer {
    fn default() -> Self {
        Self::new(GradientFlowConfig::default())
    }
}

impl GradientFlowAnalyzer {
    /// Create a new gradient flow analyzer
    pub fn new(config: GradientFlowConfig) -> Self {
        Self {
            config,
            layer_stats: HashMap::new(),
            history: Vec::new(),
            current_step: 0,
        }
    }

    /// Analyze gradients for a specific layer
    pub fn analyze_layer_gradients(&mut self, layer_name: &str, gradients: &Tensor) -> Result<()> {
        let grad_data = gradients.data()?;

        // Calculate magnitude statistics
        let magnitude_stats = self.calculate_magnitude_stats(&grad_data);

        // Calculate distribution statistics if enabled
        let distribution_stats = if self.config.track_distributions {
            Some(self.calculate_distribution_stats(&grad_data)?)
        } else {
            None
        };

        // Assess gradient flow health
        let flow_health = self.assess_flow_health(&magnitude_stats);

        // Update layer statistics
        let layer_stats = LayerGradientStats {
            layer_name: layer_name.to_string(),
            magnitude_stats,
            distribution_stats,
            flow_health,
            update_count: self.layer_stats.get(layer_name).map(|s| s.update_count + 1).unwrap_or(1),
        };

        self.layer_stats.insert(layer_name.to_string(), layer_stats);

        Ok(())
    }

    /// Analyze gradients for multiple layers
    pub fn analyze_model_gradients(
        &mut self,
        layer_gradients: HashMap<String, Tensor>,
    ) -> Result<()> {
        for (layer_name, gradients) in layer_gradients {
            self.analyze_layer_gradients(&layer_name, &gradients)?;
        }

        // Create snapshot if we have layer statistics
        if !self.layer_stats.is_empty() {
            self.create_snapshot()?;
        }

        Ok(())
    }

    /// Create a snapshot of current gradient flow state
    pub fn create_snapshot(&mut self) -> Result<()> {
        let global_stats = self.calculate_global_stats();

        let snapshot = GradientSnapshot {
            step: self.current_step,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs_f64(),
            layer_stats: self.layer_stats.clone(),
            global_stats,
        };

        self.history.push(snapshot);

        // Maintain history size limit
        while self.history.len() > self.config.max_history {
            self.history.remove(0);
        }

        self.current_step += 1;
        Ok(())
    }

    /// Calculate magnitude statistics for gradient values
    fn calculate_magnitude_stats(&self, gradients: &[f32]) -> GradientMagnitudeStats {
        let n = gradients.len() as f32;

        if n == 0.0 {
            return GradientMagnitudeStats {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                l1_norm: 0.0,
                l2_norm: 0.0,
                max_norm: 0.0,
                rms: 0.0,
            };
        }

        // Basic statistics
        let sum: f32 = gradients.iter().sum();
        let mean = sum / n;

        let variance: f32 = gradients.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();

        let min = gradients.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = gradients.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Norms
        let l1_norm: f32 = gradients.iter().map(|&x| x.abs()).sum();
        let l2_norm: f32 = gradients.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let max_norm = gradients.iter().map(|&x| x.abs()).fold(0.0f32, |a, b| a.max(b));
        let rms = (gradients.iter().map(|&x| x * x).sum::<f32>() / n).sqrt();

        GradientMagnitudeStats {
            mean,
            std,
            min,
            max,
            l1_norm,
            l2_norm,
            max_norm,
            rms,
        }
    }

    /// Calculate distribution statistics for gradient values
    fn calculate_distribution_stats(&self, gradients: &[f32]) -> Result<GradientDistributionStats> {
        if gradients.is_empty() {
            return Ok(GradientDistributionStats {
                histogram: vec![0; self.config.histogram_bins],
                bin_edges: vec![0.0; self.config.histogram_bins + 1],
                skewness: 0.0,
                kurtosis: 0.0,
                percentiles: vec![0.0; 5],
            });
        }

        // Sort gradients for percentile calculation
        let mut sorted_grads = gradients.to_vec();
        sorted_grads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate percentiles (P5, P25, P50, P75, P95)
        let percentiles = vec![
            Self::percentile(&sorted_grads, 5.0),
            Self::percentile(&sorted_grads, 25.0),
            Self::percentile(&sorted_grads, 50.0),
            Self::percentile(&sorted_grads, 75.0),
            Self::percentile(&sorted_grads, 95.0),
        ];

        // Create histogram
        let min_val = sorted_grads[0];
        let max_val = sorted_grads[sorted_grads.len() - 1];
        let range = max_val - min_val;

        let mut histogram = vec![0u32; self.config.histogram_bins];
        let mut bin_edges = Vec::with_capacity(self.config.histogram_bins + 1);

        // Calculate bin edges
        for i in 0..=self.config.histogram_bins {
            let edge = min_val + (i as f32 * range / self.config.histogram_bins as f32);
            bin_edges.push(edge);
        }

        // Fill histogram
        for &value in gradients {
            if range > 0.0 {
                let bin_idx = ((value - min_val) / range * self.config.histogram_bins as f32)
                    .floor() as usize;
                let bin_idx = bin_idx.min(self.config.histogram_bins - 1);
                histogram[bin_idx] += 1;
            } else {
                // All values are the same
                histogram[0] = gradients.len() as u32;
                break;
            }
        }

        // Calculate skewness and kurtosis
        let mean = gradients.iter().sum::<f32>() / gradients.len() as f32;
        let variance =
            gradients.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / gradients.len() as f32;
        let std_dev = variance.sqrt();

        let skewness = if std_dev > 0.0 {
            gradients.iter().map(|&x| ((x - mean) / std_dev).powi(3)).sum::<f32>()
                / gradients.len() as f32
        } else {
            0.0
        };

        let kurtosis = if std_dev > 0.0 {
            gradients.iter().map(|&x| ((x - mean) / std_dev).powi(4)).sum::<f32>()
                / gradients.len() as f32
                - 3.0 // Excess kurtosis
        } else {
            0.0
        };

        Ok(GradientDistributionStats {
            histogram,
            bin_edges,
            skewness,
            kurtosis,
            percentiles,
        })
    }

    /// Calculate percentile value
    fn percentile(sorted_values: &[f32], percentile: f32) -> f32 {
        if sorted_values.is_empty() {
            return 0.0;
        }

        let idx = (percentile / 100.0 * (sorted_values.len() - 1) as f32).round() as usize;
        let idx = idx.min(sorted_values.len() - 1);
        sorted_values[idx]
    }

    /// Assess gradient flow health
    fn assess_flow_health(&self, stats: &GradientMagnitudeStats) -> GradientFlowHealth {
        let is_vanishing = stats.l2_norm < self.config.vanishing_threshold;
        let is_exploding = stats.l2_norm > self.config.exploding_threshold;

        // Calculate ratios
        let vanishing_ratio = if stats.l2_norm > 0.0 {
            (self.config.vanishing_threshold / stats.l2_norm).min(1.0)
        } else {
            1.0
        };

        let exploding_ratio = if self.config.exploding_threshold > 0.0 {
            (stats.l2_norm / self.config.exploding_threshold).max(0.0)
        } else {
            0.0
        };

        // Calculate flow score (0.0 to 1.0)
        let flow_score = if is_vanishing {
            0.1 * (1.0 - vanishing_ratio)
        } else if is_exploding {
            0.1 * (1.0 / exploding_ratio).min(1.0)
        } else {
            // Good gradient flow
            let ideal_range = 0.01..10.0;
            if ideal_range.contains(&stats.l2_norm) {
                1.0
            } else if stats.l2_norm < ideal_range.start {
                0.5 + 0.5 * (stats.l2_norm / ideal_range.start)
            } else {
                0.5 + 0.5 * (ideal_range.end / stats.l2_norm)
            }
        };

        // Generate recommendations
        let mut recommendations = Vec::new();

        if is_vanishing {
            recommendations.push("Consider increasing learning rate".to_string());
            recommendations.push("Check for saturating activations".to_string());
            recommendations.push("Consider gradient clipping threshold".to_string());
            recommendations.push("Review network depth and skip connections".to_string());
        }

        if is_exploding {
            recommendations.push("Apply gradient clipping".to_string());
            recommendations.push("Reduce learning rate".to_string());
            recommendations.push("Check weight initialization".to_string());
            recommendations.push("Consider batch normalization".to_string());
        }

        if !is_vanishing && !is_exploding && flow_score < 0.7 {
            recommendations.push("Monitor gradient flow closely".to_string());
            recommendations.push("Consider learning rate scheduling".to_string());
        }

        GradientFlowHealth {
            is_vanishing,
            is_exploding,
            vanishing_ratio,
            exploding_ratio,
            flow_score,
            recommendations,
        }
    }

    /// Calculate global gradient statistics
    fn calculate_global_stats(&self) -> GlobalGradientStats {
        let total_layers = self.layer_stats.len();

        if total_layers == 0 {
            return GlobalGradientStats {
                total_parameters: 0,
                parameters_with_gradients: 0,
                dead_parameters: 0,
                avg_gradient_norm: 0.0,
                max_layer_norm: 0.0,
                min_layer_norm: 0.0,
                gradient_variance_across_layers: 0.0,
            };
        }

        let layer_norms: Vec<f32> =
            self.layer_stats.values().map(|stats| stats.magnitude_stats.l2_norm).collect();

        let avg_gradient_norm = layer_norms.iter().sum::<f32>() / layer_norms.len() as f32;
        let max_layer_norm = layer_norms.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_layer_norm = layer_norms.iter().fold(f32::INFINITY, |a, &b| a.min(b));

        // Calculate variance across layers
        let variance =
            layer_norms.iter().map(|&norm| (norm - avg_gradient_norm).powi(2)).sum::<f32>()
                / layer_norms.len() as f32;

        // Count dead parameters (simplified - would need actual parameter counts)
        let dead_parameters = self
            .layer_stats
            .values()
            .filter(|stats| stats.magnitude_stats.l2_norm < 1e-8)
            .count();

        GlobalGradientStats {
            total_parameters: total_layers * 1000, // Placeholder
            parameters_with_gradients: total_layers * 1000 - dead_parameters * 1000,
            dead_parameters: dead_parameters * 1000,
            avg_gradient_norm,
            max_layer_norm,
            min_layer_norm,
            gradient_variance_across_layers: variance,
        }
    }

    /// Get current layer statistics
    pub fn get_layer_stats(&self, layer_name: &str) -> Option<&LayerGradientStats> {
        self.layer_stats.get(layer_name)
    }

    /// Get all layer statistics
    pub fn get_all_layer_stats(&self) -> &HashMap<String, LayerGradientStats> {
        &self.layer_stats
    }

    /// Get gradient flow history
    pub fn get_history(&self) -> &[GradientSnapshot] {
        &self.history
    }

    /// Get the most recent snapshot
    pub fn get_latest_snapshot(&self) -> Option<&GradientSnapshot> {
        self.history.last()
    }

    /// Clear history and reset analyzer
    pub fn reset(&mut self) {
        self.layer_stats.clear();
        self.history.clear();
        self.current_step = 0;
    }

    /// Generate a comprehensive gradient flow report
    pub fn generate_report(&self) -> GradientFlowReport {
        let global_health = self.assess_global_health();
        let problematic_layers = self.identify_problematic_layers();
        let recommendations =
            self.generate_global_recommendations(&global_health, &problematic_layers);

        GradientFlowReport {
            global_health,
            layer_count: self.layer_stats.len(),
            problematic_layers,
            recommendations,
            latest_snapshot: self.get_latest_snapshot().cloned(),
        }
    }

    /// Assess overall gradient flow health
    fn assess_global_health(&self) -> GlobalGradientHealth {
        let vanishing_layers =
            self.layer_stats.values().filter(|stats| stats.flow_health.is_vanishing).count();

        let exploding_layers =
            self.layer_stats.values().filter(|stats| stats.flow_health.is_exploding).count();

        let total_layers = self.layer_stats.len();

        let avg_flow_score = if total_layers > 0 {
            self.layer_stats.values().map(|stats| stats.flow_health.flow_score).sum::<f32>()
                / total_layers as f32
        } else {
            0.0
        };

        let health_status = if avg_flow_score >= 0.8 {
            "Excellent"
        } else if avg_flow_score >= 0.6 {
            "Good"
        } else if avg_flow_score >= 0.4 {
            "Fair"
        } else if avg_flow_score >= 0.2 {
            "Poor"
        } else {
            "Critical"
        };

        GlobalGradientHealth {
            overall_score: avg_flow_score,
            health_status: health_status.to_string(),
            vanishing_layers,
            exploding_layers,
            total_layers,
            vanishing_percentage: (vanishing_layers as f32 / total_layers as f32) * 100.0,
            exploding_percentage: (exploding_layers as f32 / total_layers as f32) * 100.0,
        }
    }

    /// Identify layers with gradient flow problems
    fn identify_problematic_layers(&self) -> Vec<ProblematicLayer> {
        let mut problematic = Vec::new();

        for (layer_name, stats) in &self.layer_stats {
            let mut issues = Vec::new();

            if stats.flow_health.is_vanishing {
                issues.push("Vanishing gradients".to_string());
            }
            if stats.flow_health.is_exploding {
                issues.push("Exploding gradients".to_string());
            }
            if stats.flow_health.flow_score < 0.3 {
                issues.push("Poor gradient flow".to_string());
            }

            if !issues.is_empty() {
                problematic.push(ProblematicLayer {
                    layer_name: layer_name.clone(),
                    issues,
                    flow_score: stats.flow_health.flow_score,
                    gradient_norm: stats.magnitude_stats.l2_norm,
                });
            }
        }

        // Sort by severity (lowest flow score first)
        problematic.sort_by(|a, b| {
            a.flow_score.partial_cmp(&b.flow_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        problematic
    }

    /// Generate global recommendations based on analysis
    fn generate_global_recommendations(
        &self,
        global_health: &GlobalGradientHealth,
        problematic_layers: &[ProblematicLayer],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if global_health.vanishing_percentage > 30.0 {
            recommendations
                .push("High percentage of vanishing gradients detected. Consider:".to_string());
            recommendations.push("- Using residual connections or skip connections".to_string());
            recommendations.push("- Increasing learning rate".to_string());
            recommendations.push("- Using better weight initialization (Xavier, He)".to_string());
            recommendations.push("- Adding batch normalization or layer normalization".to_string());
        }

        if global_health.exploding_percentage > 10.0 {
            recommendations.push("Exploding gradients detected. Consider:".to_string());
            recommendations.push("- Applying gradient clipping".to_string());
            recommendations.push("- Reducing learning rate".to_string());
            recommendations.push("- Using gradient normalization".to_string());
        }

        if global_health.overall_score < 0.5 {
            recommendations.push("Overall poor gradient flow. Consider:".to_string());
            recommendations.push("- Learning rate scheduling".to_string());
            recommendations.push("- Network architecture review".to_string());
            recommendations.push("- Optimizer tuning (momentum, weight decay)".to_string());
        }

        if problematic_layers.len() > 3 {
            recommendations.push(format!(
                "{} layers need attention. Focus on:",
                problematic_layers.len()
            ));
            for layer in problematic_layers.iter().take(3) {
                recommendations.push(format!(
                    "- {} (score: {:.3})",
                    layer.layer_name, layer.flow_score
                ));
            }
        }

        recommendations
    }
}

/// Global gradient flow health assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalGradientHealth {
    pub overall_score: f32,
    pub health_status: String,
    pub vanishing_layers: usize,
    pub exploding_layers: usize,
    pub total_layers: usize,
    pub vanishing_percentage: f32,
    pub exploding_percentage: f32,
}

/// Information about a problematic layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblematicLayer {
    pub layer_name: String,
    pub issues: Vec<String>,
    pub flow_score: f32,
    pub gradient_norm: f32,
}

/// Comprehensive gradient flow report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientFlowReport {
    pub global_health: GlobalGradientHealth,
    pub layer_count: usize,
    pub problematic_layers: Vec<ProblematicLayer>,
    pub recommendations: Vec<String>,
    pub latest_snapshot: Option<GradientSnapshot>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_flow_analyzer_creation() {
        let analyzer = GradientFlowAnalyzer::default();
        assert_eq!(analyzer.current_step, 0);
        assert!(analyzer.layer_stats.is_empty());
    }

    #[test]
    fn test_magnitude_stats_calculation() -> Result<()> {
        let analyzer = GradientFlowAnalyzer::default();
        let gradients = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = analyzer.calculate_magnitude_stats(&gradients);

        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!(stats.l2_norm > 0.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);

        Ok(())
    }

    #[test]
    fn test_layer_gradient_analysis() -> Result<()> {
        let mut analyzer = GradientFlowAnalyzer::default();
        let gradient_data = vec![0.1, 0.05, -0.02, 0.08, -0.03];
        let gradients = Tensor::from_vec(gradient_data, &[5])?;

        analyzer.analyze_layer_gradients("test_layer", &gradients)?;

        let stats = analyzer.get_layer_stats("test_layer").unwrap();
        assert_eq!(stats.layer_name, "test_layer");
        assert_eq!(stats.update_count, 1);

        Ok(())
    }

    #[test]
    fn test_vanishing_gradient_detection() -> Result<()> {
        let mut config = GradientFlowConfig::default();
        config.vanishing_threshold = 0.1;

        let mut analyzer = GradientFlowAnalyzer::new(config);

        // Create very small gradients (vanishing)
        let gradient_data = vec![1e-8, 2e-8, -1e-8, 1e-9];
        let gradients = Tensor::from_vec(gradient_data, &[4])?;

        analyzer.analyze_layer_gradients("vanishing_layer", &gradients)?;

        let stats = analyzer.get_layer_stats("vanishing_layer").unwrap();
        assert!(stats.flow_health.is_vanishing);
        assert!(!stats.flow_health.recommendations.is_empty());

        Ok(())
    }

    #[test]
    fn test_exploding_gradient_detection() -> Result<()> {
        let mut config = GradientFlowConfig::default();
        config.exploding_threshold = 10.0;

        let mut analyzer = GradientFlowAnalyzer::new(config);

        // Create very large gradients (exploding)
        let gradient_data = vec![50.0, 60.0, -40.0, 80.0];
        let gradients = Tensor::from_vec(gradient_data, &[4])?;

        analyzer.analyze_layer_gradients("exploding_layer", &gradients)?;

        let stats = analyzer.get_layer_stats("exploding_layer").unwrap();
        assert!(stats.flow_health.is_exploding);
        assert!(!stats.flow_health.recommendations.is_empty());

        Ok(())
    }

    #[test]
    fn test_gradient_flow_report() -> Result<()> {
        let mut analyzer = GradientFlowAnalyzer::default();

        // Add some layers with different gradient characteristics
        let normal_grads = Tensor::from_vec(vec![0.1, 0.05, -0.02, 0.08], &[4])?;
        let vanishing_grads = Tensor::from_vec(vec![1e-8, 2e-8, -1e-8], &[3])?;

        analyzer.analyze_layer_gradients("normal_layer", &normal_grads)?;
        analyzer.analyze_layer_gradients("vanishing_layer", &vanishing_grads)?;

        let report = analyzer.generate_report();
        assert_eq!(report.layer_count, 2);
        assert!(!report.problematic_layers.is_empty());
        assert!(!report.recommendations.is_empty());

        Ok(())
    }

    #[test]
    fn test_percentile_calculation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(GradientFlowAnalyzer::percentile(&values, 50.0), 3.0);
        assert_eq!(GradientFlowAnalyzer::percentile(&values, 0.0), 1.0);
        assert_eq!(GradientFlowAnalyzer::percentile(&values, 100.0), 5.0);
    }

    #[test]
    fn test_distribution_stats() -> Result<()> {
        let analyzer = GradientFlowAnalyzer::default();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0];
        let stats = analyzer.calculate_distribution_stats(&values)?;

        assert_eq!(stats.histogram.len(), analyzer.config.histogram_bins);
        assert_eq!(stats.bin_edges.len(), analyzer.config.histogram_bins + 1);
        assert_eq!(stats.percentiles.len(), 5);

        Ok(())
    }
}
