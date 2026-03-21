//! Tensor analysis utilities and statistical functions

use anyhow::Result;
use scirs2_core::ndarray::*; // SciRS2 Integration Policy - was: use ndarray::{Array, ArrayD};
use serde::{Deserialize, Serialize};

/// Batch tensor analysis result
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchTensorAnalysis {
    pub individual_results: Vec<TensorAnalysisResult>,
    pub overall_statistics: TensorStatistics,
    pub batch_size: usize,
    pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Individual tensor analysis result
#[derive(Debug, Serialize, Deserialize)]
pub struct TensorAnalysisResult {
    pub tensor_index: usize,
    pub shape: Vec<usize>,
    pub statistics: TensorStatistics,
    pub anomalies: Vec<TensorAnomaly>,
}

/// Comprehensive tensor statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorStatistics {
    pub count: usize,
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub median: f32,
    pub p25: f32,
    pub p75: f32,
    pub nan_count: usize,
    pub inf_count: usize,
    pub zero_count: usize,
    pub skewness: f32,
    pub kurtosis: f32,
}

impl Default for TensorStatistics {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
            p25: 0.0,
            p75: 0.0,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
            skewness: 0.0,
            kurtosis: 0.0,
        }
    }
}

impl TensorStatistics {
    pub fn accumulate(&mut self, other: &TensorStatistics) {
        self.count += other.count;
        self.mean += other.mean;
        self.std_dev += other.std_dev;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.nan_count += other.nan_count;
        self.inf_count += other.inf_count;
        self.zero_count += other.zero_count;
    }

    pub fn finalize(&mut self, batch_size: usize) {
        if batch_size > 0 {
            self.mean /= batch_size as f32;
            self.std_dev /= batch_size as f32;
        }
    }
}

/// Tensor anomaly detection result
#[derive(Debug, Serialize, Deserialize)]
pub struct TensorAnomaly {
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub description: String,
    pub suggested_fix: String,
}

/// Types of tensor anomalies
#[derive(Debug, Serialize, Deserialize)]
pub enum AnomalyType {
    NanValues,
    InfiniteValues,
    ExtremeSkewness,
    ExtremeKurtosis,
    DeadNeurons,
    ExtremeValues,
    Saturation,
    Outliers,
}

/// Severity levels for anomalies
#[derive(Debug, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Advanced tensor analysis utilities
pub struct TensorAnalyzer;

impl TensorAnalyzer {
    /// Batch tensor analysis with statistical insights
    pub fn analyze_tensors_batch(tensors: &[ArrayD<f32>]) -> Result<BatchTensorAnalysis> {
        let mut results = Vec::new();
        let mut overall_stats = TensorStatistics::default();

        for (i, tensor) in tensors.iter().enumerate() {
            let stats = Self::compute_tensor_statistics(tensor)?;
            let anomalies = Self::detect_tensor_anomalies(&stats);

            results.push(TensorAnalysisResult {
                tensor_index: i,
                shape: tensor.shape().to_vec(),
                statistics: stats.clone(),
                anomalies,
            });

            overall_stats.accumulate(&stats);
        }

        overall_stats.finalize(tensors.len());

        Ok(BatchTensorAnalysis {
            individual_results: results,
            overall_statistics: overall_stats,
            batch_size: tensors.len(),
            analysis_timestamp: chrono::Utc::now(),
        })
    }

    /// Compute comprehensive statistics for a tensor
    pub fn compute_tensor_statistics(tensor: &ArrayD<f32>) -> Result<TensorStatistics> {
        let data: Vec<f32> = tensor.iter().cloned().collect();
        let count = data.len();

        if count == 0 {
            return Ok(TensorStatistics::default());
        }

        // Basic statistics
        let sum: f32 = data.iter().sum();
        let mean = sum / count as f32;

        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / count as f32;
        let std_dev = variance.sqrt();

        // Min/max
        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Percentiles
        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = Self::percentile(&sorted_data, 50.0);
        let p25 = Self::percentile(&sorted_data, 25.0);
        let p75 = Self::percentile(&sorted_data, 75.0);

        // Count special values
        let nan_count = data.iter().filter(|&&x| x.is_nan()).count();
        let inf_count = data.iter().filter(|&&x| x.is_infinite()).count();
        let zero_count = data.iter().filter(|&&x| x == 0.0).count();

        // Higher order moments
        let skewness = Self::compute_skewness(&data, mean, std_dev);
        let kurtosis = Self::compute_kurtosis(&data, mean, std_dev);

        Ok(TensorStatistics {
            count,
            mean,
            std_dev,
            min,
            max,
            median,
            p25,
            p75,
            nan_count,
            inf_count,
            zero_count,
            skewness,
            kurtosis,
        })
    }

    /// Detect anomalies in tensor statistics
    pub fn detect_tensor_anomalies(stats: &TensorStatistics) -> Vec<TensorAnomaly> {
        let mut anomalies = Vec::new();

        // Check for NaN values
        if stats.nan_count > 0 {
            anomalies.push(TensorAnomaly {
                anomaly_type: AnomalyType::NanValues,
                severity: AnomalySeverity::Critical,
                description: format!("Found {} NaN values in tensor", stats.nan_count),
                suggested_fix: "Check for division by zero or invalid operations".to_string(),
            });
        }

        // Check for infinite values
        if stats.inf_count > 0 {
            anomalies.push(TensorAnomaly {
                anomaly_type: AnomalyType::InfiniteValues,
                severity: AnomalySeverity::High,
                description: format!("Found {} infinite values in tensor", stats.inf_count),
                suggested_fix: "Check for overflow or division by zero".to_string(),
            });
        }

        // Check for extreme skewness
        if stats.skewness.abs() > 3.0 {
            anomalies.push(TensorAnomaly {
                anomaly_type: AnomalyType::ExtremeSkewness,
                severity: AnomalySeverity::Medium,
                description: format!("Extreme skewness detected: {:.2}", stats.skewness),
                suggested_fix: "Consider data normalization or outlier removal".to_string(),
            });
        }

        // Check for extreme kurtosis
        if stats.kurtosis > 10.0 {
            anomalies.push(TensorAnomaly {
                anomaly_type: AnomalyType::ExtremeKurtosis,
                severity: AnomalySeverity::Medium,
                description: format!("High kurtosis detected: {:.2}", stats.kurtosis),
                suggested_fix: "Check for outliers or distribution issues".to_string(),
            });
        }

        // Check for dead neurons (too many zeros)
        let zero_ratio = stats.zero_count as f32 / stats.count as f32;
        if zero_ratio > 0.5 {
            anomalies.push(TensorAnomaly {
                anomaly_type: AnomalyType::DeadNeurons,
                severity: AnomalySeverity::High,
                description: format!("High zero ratio: {:.2}%", zero_ratio * 100.0),
                suggested_fix:
                    "Check learning rate, weight initialization, or activation functions"
                        .to_string(),
            });
        }

        // Check for extreme values
        let range = stats.max - stats.min;
        if range > 1000.0 || stats.max.abs() > 100.0 || stats.min.abs() > 100.0 {
            anomalies.push(TensorAnomaly {
                anomaly_type: AnomalyType::ExtremeValues,
                severity: AnomalySeverity::Medium,
                description: format!("Extreme value range: [{:.2}, {:.2}]", stats.min, stats.max),
                suggested_fix: "Consider gradient clipping or weight regularization".to_string(),
            });
        }

        anomalies
    }

    /// Calculate percentile of sorted data
    fn percentile(sorted_data: &[f32], percentile: f32) -> f32 {
        if sorted_data.is_empty() {
            return 0.0;
        }

        let index = (percentile / 100.0) * (sorted_data.len() - 1) as f32;
        let lower_index = index.floor() as usize;
        let upper_index = (index.ceil() as usize).min(sorted_data.len() - 1);

        if lower_index == upper_index {
            sorted_data[lower_index]
        } else {
            let weight = index - lower_index as f32;
            sorted_data[lower_index] * (1.0 - weight) + sorted_data[upper_index] * weight
        }
    }

    /// Compute skewness
    fn compute_skewness(data: &[f32], mean: f32, std_dev: f32) -> f32 {
        if std_dev == 0.0 || data.len() < 3 {
            return 0.0;
        }

        let n = data.len() as f32;
        let skewness = data.iter().map(|&x| ((x - mean) / std_dev).powi(3)).sum::<f32>() / n;

        skewness
    }

    /// Compute kurtosis
    fn compute_kurtosis(data: &[f32], mean: f32, std_dev: f32) -> f32 {
        if std_dev == 0.0 || data.len() < 4 {
            return 0.0;
        }

        let n = data.len() as f32;
        let kurtosis = data.iter().map(|&x| ((x - mean) / std_dev).powi(4)).sum::<f32>() / n;

        kurtosis - 3.0 // Excess kurtosis
    }

    /// Compare tensors for drift detection
    pub fn compare_tensors(
        baseline: &ArrayD<f32>,
        current: &ArrayD<f32>,
    ) -> Result<TensorComparisonResult> {
        let baseline_stats = Self::compute_tensor_statistics(baseline)?;
        let current_stats = Self::compute_tensor_statistics(current)?;

        // Calculate various drift metrics
        let mean_drift = (current_stats.mean - baseline_stats.mean).abs();
        let std_drift = (current_stats.std_dev - baseline_stats.std_dev).abs();
        let distribution_shift = Self::compute_distribution_shift(&baseline_stats, &current_stats);

        let drift_severity = if mean_drift > 1.0 || std_drift > 1.0 || distribution_shift > 0.5 {
            TensorDriftSeverity::High
        } else if mean_drift > 0.5 || std_drift > 0.5 || distribution_shift > 0.3 {
            TensorDriftSeverity::Medium
        } else {
            TensorDriftSeverity::Low
        };

        Ok(TensorComparisonResult {
            baseline_stats,
            current_stats,
            mean_drift,
            std_drift,
            distribution_shift,
            drift_severity: drift_severity.clone(),
            recommendations: Self::generate_drift_recommendations(
                drift_severity,
                mean_drift,
                std_drift,
            ),
        })
    }

    /// Compute distribution shift between two sets of statistics
    fn compute_distribution_shift(baseline: &TensorStatistics, current: &TensorStatistics) -> f32 {
        // Simple distribution shift metric based on statistical differences
        let mean_diff = ((current.mean - baseline.mean) / (baseline.std_dev + 1e-8)).abs();
        let std_diff = ((current.std_dev - baseline.std_dev) / (baseline.std_dev + 1e-8)).abs();
        let skew_diff = (current.skewness - baseline.skewness).abs();

        (mean_diff + std_diff + skew_diff * 0.5) / 2.5
    }

    /// Generate recommendations based on drift severity
    fn generate_drift_recommendations(
        severity: TensorDriftSeverity,
        mean_drift: f32,
        std_drift: f32,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        match severity {
            TensorDriftSeverity::High => {
                recommendations.push("Significant tensor drift detected".to_string());
                if mean_drift > 1.0 {
                    recommendations.push("Consider retraining or data rebalancing".to_string());
                }
                if std_drift > 1.0 {
                    recommendations.push("Check for changes in data preprocessing".to_string());
                }
            },
            TensorDriftSeverity::Medium => {
                recommendations.push("Moderate tensor drift detected".to_string());
                recommendations.push("Monitor closely for further changes".to_string());
            },
            TensorDriftSeverity::Low => {
                recommendations.push("Minimal tensor drift - within acceptable range".to_string());
            },
        }

        recommendations
    }
}

/// Result of tensor comparison for drift detection
#[derive(Debug, Serialize, Deserialize)]
pub struct TensorComparisonResult {
    pub baseline_stats: TensorStatistics,
    pub current_stats: TensorStatistics,
    pub mean_drift: f32,
    pub std_drift: f32,
    pub distribution_shift: f32,
    pub drift_severity: TensorDriftSeverity,
    pub recommendations: Vec<String>,
}

/// Drift severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorDriftSeverity {
    Low,
    Medium,
    High,
}
