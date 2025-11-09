//! Weight and gradient analysis utilities

use anyhow::Result;
use scirs2_core::ndarray::*; // SciRS2 Integration Policy - was: use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

/// Layer exploding gradient information
#[derive(Debug, Serialize, Deserialize)]
pub struct ExplodingLayer {
    pub layer_index: usize,
    pub gradient_norm: f32,
    pub severity: ExplosionSeverity,
    pub recommended_action: String,
}

/// Severity levels for gradient explosion
#[derive(Debug, Serialize, Deserialize)]
pub enum ExplosionSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Comprehensive gradient explosion analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct GradientExplosionAnalysis {
    pub exploding_layers: Vec<ExplodingLayer>,
    pub max_gradient_norm: f32,
    pub mean_gradient_norm: f32,
    pub std_gradient_norm: f32,
    pub explosion_ratio: f32,
    pub overall_severity: ExplosionSeverity,
    pub mitigation_recommendations: Vec<String>,
}

/// Weight distribution analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct WeightDistributionAnalysis {
    pub layer_analyses: Vec<LayerWeightAnalysis>,
    pub overall_statistics: WeightStatistics,
    pub distribution_health: DistributionHealth,
    pub outlier_detection: Vec<WeightOutlier>,
}

/// Individual layer weight analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct LayerWeightAnalysis {
    pub layer_index: usize,
    pub statistics: WeightStatistics,
    pub health_score: f32,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Weight statistics for a layer or model
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WeightStatistics {
    pub mean: f32,
    pub std_dev: f32,
    pub skewness: f32,
    pub kurtosis: f32,
    pub entropy: f32,
    pub min: f32,
    pub max: f32,
    pub zero_fraction: f32,
}

impl WeightStatistics {
    pub fn accumulate(&mut self, other: &WeightStatistics) {
        // Simple accumulation for overall statistics
        self.mean += other.mean;
        self.std_dev += other.std_dev;
        self.skewness += other.skewness;
        self.kurtosis += other.kurtosis;
        self.entropy += other.entropy;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.zero_fraction += other.zero_fraction;
    }

    pub fn finalize(&mut self, count: usize) {
        if count > 0 {
            let count_f32 = count as f32;
            self.mean /= count_f32;
            self.std_dev /= count_f32;
            self.skewness /= count_f32;
            self.kurtosis /= count_f32;
            self.entropy /= count_f32;
            self.zero_fraction /= count_f32;
        }
    }
}

/// Weight health assessment
#[derive(Debug, Serialize, Deserialize)]
pub struct WeightHealth {
    pub score: f32,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Distribution health status
#[derive(Debug, Serialize, Deserialize)]
pub struct DistributionHealth {
    pub score: f32,
    pub status: DistributionHealthStatus,
}

/// Health status levels for weight distributions
#[derive(Debug, Serialize, Deserialize)]
pub enum DistributionHealthStatus {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

/// Weight outlier information
#[derive(Debug, Serialize, Deserialize)]
pub struct WeightOutlier {
    pub layer_index: usize,
    pub weight_index: usize,
    pub value: f32,
    pub z_score: f32,
    pub severity: OutlierSeverity,
}

/// Severity levels for weight outliers
#[derive(Debug, Serialize, Deserialize)]
pub enum OutlierSeverity {
    Medium,
    High,
}

/// Weight drift analysis between model states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightDriftAnalysis {
    pub mean_drift: f32,
    pub max_drift: f32,
    pub severity: WeightDriftSeverity,
    pub affected_layers: Vec<usize>,
}

/// Drift severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightDriftSeverity {
    Minimal,
    Low,
    Medium,
    High,
}

/// Weight and gradient analysis utilities
pub struct WeightAnalyzer;

impl WeightAnalyzer {
    /// Detect gradient explosion patterns in a set of gradients
    pub fn detect_gradient_explosion(
        gradients: &[ArrayD<f32>],
        threshold: f32,
    ) -> GradientExplosionAnalysis {
        let mut exploding_layers = Vec::new();
        let mut max_gradient_norm = 0.0f32;
        let mut gradient_norms = Vec::new();

        for (layer_idx, gradient) in gradients.iter().enumerate() {
            let l2_norm = Self::compute_l2_norm(gradient);
            gradient_norms.push(l2_norm);

            if l2_norm > max_gradient_norm {
                max_gradient_norm = l2_norm;
            }

            if l2_norm > threshold {
                exploding_layers.push(ExplodingLayer {
                    layer_index: layer_idx,
                    gradient_norm: l2_norm,
                    severity: Self::classify_explosion_severity(l2_norm, &gradient_norms),
                    recommended_action: Self::recommend_explosion_mitigation(l2_norm),
                });
            }
        }

        let mean_norm = gradient_norms.iter().sum::<f32>() / gradient_norms.len() as f32;
        let std_norm = {
            let variance: f32 =
                gradient_norms.iter().map(|&x| (x - mean_norm).powi(2)).sum::<f32>()
                    / gradient_norms.len() as f32;
            variance.sqrt()
        };

        let explosion_ratio = exploding_layers.len() as f32 / gradients.len() as f32;

        let overall_severity = if explosion_ratio > 0.5 || max_gradient_norm > threshold * 10.0 {
            ExplosionSeverity::Critical
        } else if explosion_ratio > 0.3 || max_gradient_norm > threshold * 5.0 {
            ExplosionSeverity::High
        } else if explosion_ratio > 0.1 || max_gradient_norm > threshold * 2.0 {
            ExplosionSeverity::Medium
        } else {
            ExplosionSeverity::Low
        };

        GradientExplosionAnalysis {
            exploding_layers,
            max_gradient_norm,
            mean_gradient_norm: mean_norm,
            std_gradient_norm: std_norm,
            explosion_ratio,
            overall_severity,
            mitigation_recommendations: Self::generate_explosion_recommendations(
                explosion_ratio,
                max_gradient_norm,
            ),
        }
    }

    /// Analyze weight distributions across model layers
    pub fn analyze_weight_distribution(
        weights: &[ArrayD<f32>],
    ) -> Result<WeightDistributionAnalysis> {
        let mut layer_analyses = Vec::new();
        let mut overall_stats = WeightStatistics::default();
        let mut all_outliers = Vec::new();

        for (layer_idx, weight_tensor) in weights.iter().enumerate() {
            let layer_stats = Self::compute_weight_statistics(weight_tensor)?;
            let health_score = Self::compute_weight_health_score(&layer_stats);
            let outliers = Self::detect_weight_outliers(weight_tensor, layer_idx)?;

            let issues = Self::identify_weight_issues(&layer_stats);
            let recommendations = Self::generate_weight_recommendations(&issues);

            layer_analyses.push(LayerWeightAnalysis {
                layer_index: layer_idx,
                statistics: layer_stats.clone(),
                health_score,
                issues,
                recommendations,
            });

            overall_stats.accumulate(&layer_stats);
            all_outliers.extend(outliers);
        }

        overall_stats.finalize(weights.len());

        let distribution_health = Self::assess_distribution_health(&overall_stats);

        Ok(WeightDistributionAnalysis {
            layer_analyses,
            overall_statistics: overall_stats,
            distribution_health,
            outlier_detection: all_outliers,
        })
    }

    /// Compute L2 norm of a tensor
    pub fn compute_l2_norm(tensor: &ArrayD<f32>) -> f32 {
        tensor.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    /// Compute comprehensive weight statistics for a tensor
    fn compute_weight_statistics(tensor: &ArrayD<f32>) -> Result<WeightStatistics> {
        let data: Vec<f32> = tensor.iter().cloned().collect();
        let count = data.len();

        if count == 0 {
            return Ok(WeightStatistics::default());
        }

        // Basic statistics
        let sum: f32 = data.iter().sum();
        let mean = sum / count as f32;

        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / count as f32;
        let std_dev = variance.sqrt();

        // Min/max
        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Zero fraction
        let zero_count = data.iter().filter(|&&x| x == 0.0).count();
        let zero_fraction = zero_count as f32 / count as f32;

        // Higher order moments
        let skewness = Self::compute_skewness(&data, mean, std_dev);
        let kurtosis = Self::compute_kurtosis(&data, mean, std_dev);
        let entropy = Self::compute_entropy(&data);

        Ok(WeightStatistics {
            mean,
            std_dev,
            skewness,
            kurtosis,
            entropy,
            min,
            max,
            zero_fraction,
        })
    }

    /// Classify gradient explosion severity
    fn classify_explosion_severity(norm: f32, all_norms: &[f32]) -> ExplosionSeverity {
        let mean_norm = all_norms.iter().sum::<f32>() / all_norms.len() as f32;
        let ratio = norm / (mean_norm + 1e-8);

        if ratio > 100.0 {
            ExplosionSeverity::Critical
        } else if ratio > 50.0 {
            ExplosionSeverity::High
        } else if ratio > 10.0 {
            ExplosionSeverity::Medium
        } else {
            ExplosionSeverity::Low
        }
    }

    /// Recommend mitigation for gradient explosion
    fn recommend_explosion_mitigation(norm: f32) -> String {
        if norm > 100.0 {
            "Critical gradient explosion: Reduce learning rate by 10x and implement gradient clipping".to_string()
        } else if norm > 10.0 {
            "High gradient explosion: Reduce learning rate and implement gradient clipping"
                .to_string()
        } else if norm > 5.0 {
            "Moderate gradient explosion: Consider gradient clipping or learning rate reduction"
                .to_string()
        } else {
            "Monitor gradients for stability".to_string()
        }
    }

    /// Generate recommendations for gradient explosion mitigation
    fn generate_explosion_recommendations(explosion_ratio: f32, max_norm: f32) -> Vec<String> {
        let mut recommendations = Vec::new();

        if explosion_ratio > 0.3 {
            recommendations.push("High proportion of exploding gradients detected".to_string());
            recommendations.push("Consider significant learning rate reduction".to_string());
        }

        if max_norm > 100.0 {
            recommendations.push("Extremely large gradients detected".to_string());
            recommendations.push("Implement gradient clipping with threshold < 1.0".to_string());
        }

        recommendations.push("Monitor gradient norms during training".to_string());
        recommendations.push("Consider batch normalization or layer normalization".to_string());

        recommendations
    }

    /// Compute weight health score
    fn compute_weight_health_score(stats: &WeightStatistics) -> f32 {
        let mut score: f32 = 100.0;

        // Penalize extreme values
        if stats.max.abs() > 10.0 || stats.min.abs() > 10.0 {
            score -= 20.0;
        }

        // Penalize high zero fraction (dead neurons)
        if stats.zero_fraction > 0.5 {
            score -= 30.0;
        }

        // Penalize extreme skewness or kurtosis
        if stats.skewness.abs() > 2.0 {
            score -= 15.0;
        }
        if stats.kurtosis > 10.0 {
            score -= 15.0;
        }

        score.max(0.0)
    }

    /// Detect weight outliers in a tensor
    fn detect_weight_outliers(
        tensor: &ArrayD<f32>,
        layer_idx: usize,
    ) -> Result<Vec<WeightOutlier>> {
        let data: Vec<f32> = tensor.iter().cloned().collect();
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let std_dev = {
            let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
            variance.sqrt()
        };

        let mut outliers = Vec::new();

        for (idx, &value) in data.iter().enumerate() {
            let z_score = ((value - mean) / std_dev).abs();

            if z_score > 3.0 {
                let severity =
                    if z_score > 5.0 { OutlierSeverity::High } else { OutlierSeverity::Medium };

                outliers.push(WeightOutlier {
                    layer_index: layer_idx,
                    weight_index: idx,
                    value,
                    z_score,
                    severity,
                });
            }
        }

        Ok(outliers)
    }

    /// Assess overall distribution health
    fn assess_distribution_health(stats: &WeightStatistics) -> DistributionHealth {
        let mut score = 100.0;

        // Factor in various metrics
        if stats.zero_fraction > 0.3 {
            score -= 25.0;
        }
        if stats.skewness.abs() > 1.0 {
            score -= 15.0;
        }
        if stats.kurtosis > 5.0 {
            score -= 15.0;
        }
        if stats.max.abs() > 5.0 || stats.min.abs() > 5.0 {
            score -= 20.0;
        }

        let status = match score {
            s if s >= 90.0 => DistributionHealthStatus::Excellent,
            s if s >= 75.0 => DistributionHealthStatus::Good,
            s if s >= 60.0 => DistributionHealthStatus::Fair,
            s if s >= 40.0 => DistributionHealthStatus::Poor,
            _ => DistributionHealthStatus::Critical,
        };

        DistributionHealth { score, status }
    }

    /// Identify issues in weight statistics
    fn identify_weight_issues(stats: &WeightStatistics) -> Vec<String> {
        let mut issues = Vec::new();

        if stats.zero_fraction > 0.5 {
            issues.push("High proportion of zero weights (dead neurons)".to_string());
        }

        if stats.skewness.abs() > 2.0 {
            issues.push("Highly skewed weight distribution".to_string());
        }

        if stats.kurtosis > 10.0 {
            issues.push("Heavy-tailed weight distribution".to_string());
        }

        if stats.max.abs() > 10.0 || stats.min.abs() > 10.0 {
            issues.push("Extreme weight values detected".to_string());
        }

        issues
    }

    /// Generate recommendations based on weight issues
    fn generate_weight_recommendations(issues: &[String]) -> Vec<String> {
        let mut recommendations = Vec::new();

        for issue in issues {
            match issue.as_str() {
                s if s.contains("dead neurons") => {
                    recommendations.push(
                        "Consider reducing learning rate or changing activation function"
                            .to_string(),
                    );
                },
                s if s.contains("skewed") => {
                    recommendations.push(
                        "Consider weight normalization or different initialization".to_string(),
                    );
                },
                s if s.contains("heavy-tailed") => {
                    recommendations.push("Monitor for gradient instability".to_string());
                },
                s if s.contains("extreme") => {
                    recommendations.push("Implement weight clipping or regularization".to_string());
                },
                _ => {},
            }
        }

        recommendations
    }

    /// Compute skewness of data
    fn compute_skewness(data: &[f32], mean: f32, std_dev: f32) -> f32 {
        if std_dev == 0.0 || data.len() < 3 {
            return 0.0;
        }

        let n = data.len() as f32;
        data.iter().map(|&x| ((x - mean) / std_dev).powi(3)).sum::<f32>() / n
    }

    /// Compute kurtosis of data
    fn compute_kurtosis(data: &[f32], mean: f32, std_dev: f32) -> f32 {
        if std_dev == 0.0 || data.len() < 4 {
            return 0.0;
        }

        let n = data.len() as f32;
        data.iter().map(|&x| ((x - mean) / std_dev).powi(4)).sum::<f32>() / n - 3.0
        // Excess kurtosis
    }

    /// Compute entropy of data (simplified)
    fn compute_entropy(data: &[f32]) -> f32 {
        // Simplified entropy computation
        // In practice, this would discretize the data and compute proper entropy
        let std_dev = {
            let mean = data.iter().sum::<f32>() / data.len() as f32;
            let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
            variance.sqrt()
        };

        // Higher std_dev implies higher entropy (roughly)
        std_dev.log2().max(0.0)
    }
}
