//! Advanced memory analytics components
//!
//! This module contains sophisticated analytics components for memory analysis
//! including adaptive thresholds, leak detection, and memory prediction.

use super::types::MemoryMetrics;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// Pre-allocated recommendations to reduce string allocations in hot paths
#[derive(Debug, Clone)]
pub struct AlertRecommendations {
    pub high_memory: Vec<String>,
    pub rapid_growth: Vec<String>,
    pub fragmentation: Vec<String>,
    pub memory_leak: Vec<String>,
    pub gc_pressure: Vec<String>,
}

impl Default for AlertRecommendations {
    fn default() -> Self {
        Self::new()
    }
}

impl AlertRecommendations {
    pub fn new() -> Self {
        Self {
            high_memory: vec![
                "Consider reducing batch size".to_string(),
                "Enable gradient checkpointing".to_string(),
                "Use mixed precision training".to_string(),
                "Implement tensor sharding for large models".to_string(),
            ],
            rapid_growth: vec![
                "Check for memory leaks".to_string(),
                "Review tensor lifetime management".to_string(),
                "Consider explicit garbage collection".to_string(),
                "Monitor unclosed file handles and resources".to_string(),
            ],
            fragmentation: vec![
                "Restart the process to defragment memory".to_string(),
                "Use memory pools for frequent allocations".to_string(),
                "Implement custom allocators for large tensors".to_string(),
            ],
            memory_leak: vec![
                "Use weak references for circular dependencies".to_string(),
                "Implement explicit cleanup for model components".to_string(),
                "Review closure captures for tensor references".to_string(),
                "Consider using RAII patterns for resource management".to_string(),
            ],
            gc_pressure: vec![
                "Increase heap size if possible".to_string(),
                "Reduce allocation frequency in hot paths".to_string(),
                "Implement object pooling for frequent allocations".to_string(),
                "Use pre-allocated buffers for intermediate results".to_string(),
            ],
        }
    }
}

/// Adaptive threshold system that adjusts based on model characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveThresholds {
    pub base_memory_threshold: f64,
    pub growth_rate_threshold: f64,
    pub fragmentation_threshold: f64,
    pub adaptation_factor: f64,
    pub last_updated: SystemTime,
}

impl Default for AdaptiveThresholds {
    fn default() -> Self {
        Self {
            base_memory_threshold: 1024.0,
            growth_rate_threshold: 50.0,
            fragmentation_threshold: 0.3,
            adaptation_factor: 0.1,
            last_updated: SystemTime::now(),
        }
    }
}

impl AdaptiveThresholds {
    /// Update thresholds based on current memory usage patterns
    pub fn update_thresholds(&mut self, recent_metrics: &[MemoryMetrics]) {
        if recent_metrics.is_empty() {
            return;
        }

        let avg_memory = recent_metrics.iter().map(|m| m.total_memory_mb).sum::<f64>()
            / recent_metrics.len() as f64;

        let avg_growth =
            recent_metrics.iter().map(|m| m.memory_growth_rate_mb_per_sec).sum::<f64>()
                / recent_metrics.len() as f64;

        let avg_fragmentation =
            recent_metrics.iter().map(|m| m.memory_fragmentation_ratio).sum::<f64>()
                / recent_metrics.len() as f64;

        // Adaptive adjustment
        self.base_memory_threshold = self.base_memory_threshold * (1.0 - self.adaptation_factor)
            + avg_memory * 1.2 * self.adaptation_factor;
        self.growth_rate_threshold = self.growth_rate_threshold * (1.0 - self.adaptation_factor)
            + avg_growth * 2.0 * self.adaptation_factor;
        self.fragmentation_threshold = self.fragmentation_threshold
            * (1.0 - self.adaptation_factor)
            + avg_fragmentation * 1.5 * self.adaptation_factor;

        self.last_updated = SystemTime::now();
    }
}

/// Memory leak detection heuristics
#[derive(Debug, Clone)]
pub struct LeakDetectionHeuristics {
    pub sustained_growth_threshold: f64,
    pub growth_duration_threshold: Duration,
    pub allocation_pattern_threshold: usize,
    pub false_positive_filter: f64,
}

impl Default for LeakDetectionHeuristics {
    fn default() -> Self {
        Self {
            sustained_growth_threshold: 10.0,                    // MB/sec
            growth_duration_threshold: Duration::from_secs(300), // 5 minutes
            allocation_pattern_threshold: 1000,
            false_positive_filter: 0.8, // Confidence threshold
        }
    }
}

impl LeakDetectionHeuristics {
    /// Analyze memory patterns for potential leaks
    pub fn detect_potential_leaks(&self, metrics: &[MemoryMetrics]) -> Vec<LeakAlert> {
        let mut alerts = Vec::new();

        if metrics.len() < 10 {
            return alerts; // Not enough data
        }

        // Check for sustained growth
        let recent_metrics = &metrics[metrics.len().saturating_sub(60)..];
        let growth_rates: Vec<f64> =
            recent_metrics.iter().map(|m| m.memory_growth_rate_mb_per_sec).collect();

        let avg_growth = growth_rates.iter().sum::<f64>() / growth_rates.len() as f64;

        if avg_growth > self.sustained_growth_threshold {
            let confidence = self.calculate_leak_confidence(&growth_rates);
            if confidence > self.false_positive_filter {
                alerts.push(LeakAlert {
                    alert_type: LeakAlertType::SustainedGrowth,
                    confidence,
                    description: format!(
                        "Sustained memory growth detected: {:.2} MB/sec",
                        avg_growth
                    ),
                });
            }
        }

        // Check for allocation patterns
        let total_allocations: u64 = recent_metrics.iter().map(|m| m.allocated_objects).sum();
        let total_deallocations: u64 = recent_metrics.iter().map(|m| m.deallocated_objects).sum();

        if total_allocations > total_deallocations + self.allocation_pattern_threshold as u64 {
            let imbalance_ratio = total_allocations as f64 / (total_deallocations + 1) as f64;
            if imbalance_ratio > 1.2 {
                alerts.push(LeakAlert {
                    alert_type: LeakAlertType::AllocationImbalance,
                    confidence: (imbalance_ratio - 1.0).min(1.0),
                    description: format!(
                        "Allocation/deallocation imbalance: ratio {:.2}",
                        imbalance_ratio
                    ),
                });
            }
        }

        alerts
    }

    fn calculate_leak_confidence(&self, growth_rates: &[f64]) -> f64 {
        if growth_rates.is_empty() {
            return 0.0;
        }

        // Calculate consistency of growth (low variance = higher confidence)
        let mean = growth_rates.iter().sum::<f64>() / growth_rates.len() as f64;
        let variance = growth_rates.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / growth_rates.len() as f64;

        let consistency = 1.0 / (1.0 + variance);
        let magnitude = (mean / self.sustained_growth_threshold).min(1.0);

        (consistency * magnitude).min(1.0)
    }
}

#[derive(Debug, Clone)]
pub struct LeakAlert {
    pub alert_type: LeakAlertType,
    pub confidence: f64,
    pub description: String,
}

#[derive(Debug, Clone)]
pub enum LeakAlertType {
    SustainedGrowth,
    AllocationImbalance,
    PatternAnomaly,
}

/// Memory usage predictor using trend analysis
#[derive(Debug, Clone)]
pub struct MemoryPredictor {
    trend_window: usize,
    prediction_horizon_secs: u64,
    confidence_threshold: f64,
    linear_regression_cache: Option<LinearRegression>,
}

impl Default for MemoryPredictor {
    fn default() -> Self {
        Self {
            trend_window: 60,             // Last 60 data points
            prediction_horizon_secs: 300, // 5 minutes ahead
            confidence_threshold: 0.7,
            linear_regression_cache: None,
        }
    }
}

impl MemoryPredictor {
    /// Predict future memory usage based on current trends
    pub fn predict_memory_usage(
        &mut self,
        metrics: &[MemoryMetrics],
        horizon_secs: Option<u64>,
    ) -> Option<MemoryPrediction> {
        if metrics.len() < self.trend_window {
            return None;
        }

        let recent_metrics = &metrics[metrics.len() - self.trend_window..];
        let horizon = horizon_secs.unwrap_or(self.prediction_horizon_secs);

        // Convert to time series data
        let data_points: Vec<(f64, f64)> = recent_metrics
            .iter()
            .enumerate()
            .map(|(i, m)| (i as f64, m.total_memory_mb))
            .collect();

        let regression = self.calculate_linear_regression(&data_points);
        let confidence = regression.correlation.abs();

        if confidence < self.confidence_threshold {
            return None; // Low confidence prediction
        }

        let future_time = data_points.len() as f64 + (horizon as f64 / 60.0); // Assuming 1-minute intervals
        let predicted_memory = regression.slope * future_time + regression.intercept;

        self.linear_regression_cache = Some(regression.clone());

        Some(MemoryPrediction {
            predicted_memory_mb: predicted_memory,
            confidence,
            horizon_secs: horizon,
            trend_slope: regression.slope,
        })
    }

    fn calculate_linear_regression(&self, data: &[(f64, f64)]) -> LinearRegression {
        let n = data.len() as f64;
        let sum_x: f64 = data.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = data.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = data.iter().map(|(x, _)| x * x).sum();
        let sum_y2: f64 = data.iter().map(|(_, y)| y * y).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Calculate correlation coefficient
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        let correlation = if denominator != 0.0 { numerator / denominator } else { 0.0 };

        LinearRegression {
            slope,
            intercept,
            correlation,
            last_computed: SystemTime::now(),
        }
    }
}

/// Simple linear regression for memory prediction
#[derive(Debug, Clone)]
pub struct LinearRegression {
    pub slope: f64,
    pub intercept: f64,
    pub correlation: f64,
    pub last_computed: SystemTime,
}

#[derive(Debug, Clone)]
pub struct MemoryPrediction {
    pub predicted_memory_mb: f64,
    pub confidence: f64,
    pub horizon_secs: u64,
    pub trend_slope: f64,
}

/// Advanced statistical analyzer for memory patterns
#[derive(Debug, Clone)]
pub struct StatisticalAnalyzer {
    #[allow(dead_code)]
    confidence_interval: f64,
    outlier_threshold: f64,
}

impl StatisticalAnalyzer {
    pub fn new(confidence_interval: f64) -> Self {
        Self {
            confidence_interval,
            outlier_threshold: 2.0, // 2 standard deviations
        }
    }

    /// Calculate memory usage statistics with confidence intervals
    pub fn calculate_usage_statistics(&self, metrics: &[MemoryMetrics]) -> MemoryStatistics {
        if metrics.is_empty() {
            return MemoryStatistics::default();
        }

        let memory_values: Vec<f64> = metrics.iter().map(|m| m.total_memory_mb).collect();

        let mean = self.calculate_mean(&memory_values);
        let std_dev = self.calculate_std_dev(&memory_values, mean);
        let median = self.calculate_median(&memory_values);
        let (p25, p75) = self.calculate_quartiles(&memory_values);

        let confidence_lower = mean - (1.96 * std_dev / (memory_values.len() as f64).sqrt());
        let confidence_upper = mean + (1.96 * std_dev / (memory_values.len() as f64).sqrt());

        let outliers = self.detect_outliers(&memory_values, mean, std_dev);
        let trend = self.calculate_trend(&memory_values);

        MemoryStatistics {
            mean,
            median,
            std_dev,
            min: memory_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max: memory_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            percentile_25: p25,
            percentile_75: p75,
            confidence_interval_lower: confidence_lower,
            confidence_interval_upper: confidence_upper,
            outlier_count: outliers.len(),
            trend_slope: trend,
            coefficient_of_variation: if mean != 0.0 { std_dev / mean } else { 0.0 },
        }
    }

    /// Detect memory usage anomalies using statistical methods
    pub fn detect_anomalies(&self, metrics: &[MemoryMetrics]) -> Vec<MemoryAnomaly> {
        let mut anomalies = Vec::new();

        if metrics.len() < 10 {
            return anomalies; // Need sufficient data for anomaly detection
        }

        let memory_values: Vec<f64> = metrics.iter().map(|m| m.total_memory_mb).collect();
        let mean = self.calculate_mean(&memory_values);
        let std_dev = self.calculate_std_dev(&memory_values, mean);

        // Detect sudden spikes
        for (i, &value) in memory_values.iter().enumerate() {
            let z_score = (value - mean).abs() / std_dev;

            if z_score > self.outlier_threshold {
                anomalies.push(MemoryAnomaly {
                    timestamp: metrics[i].timestamp,
                    anomaly_type: if value > mean {
                        AnomalyType::SuddenSpike
                    } else {
                        AnomalyType::SuddenDrop
                    },
                    severity: if z_score > 3.0 {
                        AnomalySeverity::High
                    } else {
                        AnomalySeverity::Medium
                    },
                    value,
                    expected_value: mean,
                    confidence_score: ((z_score - self.outlier_threshold) / self.outlier_threshold).min(1.0),
                    description: format!("Memory usage {} ({}MB) deviates significantly from expected value ({}MB). Z-score: {:.2}",
                                       if value > mean { "spike" } else { "drop" },
                                       value, mean, z_score),
                });
            }
        }

        // Detect sustained growth patterns
        if let Some(growth_anomaly) = self.detect_sustained_growth(&memory_values, metrics) {
            anomalies.push(growth_anomaly);
        }

        anomalies
    }

    fn calculate_mean(&self, values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
    }

    fn calculate_std_dev(&self, values: &[f64], mean: f64) -> f64 {
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    fn calculate_median(&self, values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    fn calculate_quartiles(&self, values: &[f64]) -> (f64, f64) {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;
        (sorted[q1_idx], sorted[q3_idx])
    }

    fn detect_outliers(&self, values: &[f64], mean: f64, std_dev: f64) -> Vec<f64> {
        values
            .iter()
            .filter(|&&x| (x - mean).abs() / std_dev > self.outlier_threshold)
            .copied()
            .collect()
    }

    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = self.calculate_mean(values);

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    fn detect_sustained_growth(
        &self,
        values: &[f64],
        metrics: &[MemoryMetrics],
    ) -> Option<MemoryAnomaly> {
        if values.len() < 20 {
            return None;
        }

        let recent_window = &values[values.len() - 10..];
        let earlier_window = &values[values.len() - 20..values.len() - 10];

        let recent_mean = self.calculate_mean(recent_window);
        let earlier_mean = self.calculate_mean(earlier_window);

        let growth_rate = (recent_mean - earlier_mean) / earlier_mean;

        if growth_rate > 0.2 {
            // 20% growth is considered anomalous
            Some(MemoryAnomaly {
                timestamp: metrics[metrics.len() - 1].timestamp,
                anomaly_type: AnomalyType::SustainedGrowth,
                severity: if growth_rate > 0.5 {
                    AnomalySeverity::High
                } else {
                    AnomalySeverity::Medium
                },
                value: recent_mean,
                expected_value: earlier_mean,
                confidence_score: (growth_rate / 0.5).min(1.0),
                description: format!(
                    "Sustained memory growth detected: {:.1}% increase over recent observations",
                    growth_rate * 100.0
                ),
            })
        } else {
            None
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatistics {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentile_25: f64,
    pub percentile_75: f64,
    pub confidence_interval_lower: f64,
    pub confidence_interval_upper: f64,
    pub outlier_count: usize,
    pub trend_slope: f64,
    pub coefficient_of_variation: f64,
}

impl Default for MemoryStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            percentile_25: 0.0,
            percentile_75: 0.0,
            confidence_interval_lower: 0.0,
            confidence_interval_upper: 0.0,
            outlier_count: 0,
            trend_slope: 0.0,
            coefficient_of_variation: 0.0,
        }
    }
}

/// Memory anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnomaly {
    pub timestamp: SystemTime,
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub value: f64,
    pub expected_value: f64,
    pub confidence_score: f64,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnomalyType {
    SuddenSpike,
    SuddenDrop,
    SustainedGrowth,
    UnusualPattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}
