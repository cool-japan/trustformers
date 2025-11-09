//! Performance metrics and analysis functionality.
//!
//! This module provides comprehensive performance monitoring and analysis
//! capabilities including throughput analysis, memory usage tracking,
//! performance trend analysis, and performance optimization recommendations.

use super::types::{ModelPerformanceMetrics, PerformanceSummary};

/// Performance analyzer for tracking and analyzing model performance metrics.
#[derive(Debug)]
pub struct PerformanceAnalyzer {
    /// Historical performance metrics
    performance_history: Vec<ModelPerformanceMetrics>,
    /// Maximum history length to maintain
    max_history_length: usize,
    /// Performance thresholds for alerts
    thresholds: PerformanceThresholds,
}

/// Performance thresholds for triggering alerts.
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Maximum acceptable memory usage in MB
    pub max_memory_mb: f64,
    /// Minimum acceptable throughput in samples/sec
    pub min_throughput: f64,
    /// Maximum acceptable loss increase percentage
    pub max_loss_increase_percent: f64,
    /// Maximum acceptable variance in loss
    pub max_loss_variance: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_memory_mb: 8192.0, // 8GB
            min_throughput: 100.0,
            max_loss_increase_percent: 10.0,
            max_loss_variance: 0.1,
        }
    }
}

impl PerformanceAnalyzer {
    /// Create a new performance analyzer.
    pub fn new() -> Self {
        Self {
            performance_history: Vec::new(),
            max_history_length: 1000,
            thresholds: PerformanceThresholds::default(),
        }
    }

    /// Create a new performance analyzer with custom thresholds.
    pub fn with_thresholds(thresholds: PerformanceThresholds) -> Self {
        Self {
            performance_history: Vec::new(),
            max_history_length: 1000,
            thresholds,
        }
    }

    /// Set maximum history length.
    pub fn set_max_history_length(&mut self, length: usize) {
        self.max_history_length = length;
        if self.performance_history.len() > length {
            self.performance_history.drain(0..self.performance_history.len() - length);
        }
    }

    /// Record a new performance measurement.
    pub fn record_performance(&mut self, metrics: ModelPerformanceMetrics) {
        self.performance_history.push(metrics);

        // Maintain maximum history length
        if self.performance_history.len() > self.max_history_length {
            self.performance_history.remove(0);
        }
    }

    /// Record metrics (alias for record_performance).
    pub fn record_metrics(&mut self, metrics: ModelPerformanceMetrics) {
        self.record_performance(metrics);
    }

    /// Get the complete performance history.
    pub fn get_performance_history(&self) -> &[ModelPerformanceMetrics] {
        &self.performance_history
    }

    /// Generate a performance summary.
    pub fn generate_performance_summary(&self) -> PerformanceSummary {
        if self.performance_history.is_empty() {
            return PerformanceSummary::default();
        }

        let total_steps = self.performance_history.len();
        let current_metrics = self.performance_history.last().unwrap();

        let losses: Vec<f64> = self.performance_history.iter().map(|m| m.loss).collect();
        let throughputs: Vec<f64> =
            self.performance_history.iter().map(|m| m.throughput_samples_per_sec).collect();
        let memory_usages: Vec<f64> =
            self.performance_history.iter().map(|m| m.memory_usage_mb).collect();

        let best_loss = losses.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let avg_loss = losses.iter().sum::<f64>() / losses.len() as f64;
        let avg_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let peak_memory_mb = memory_usages.iter().fold(0.0f64, |acc, &x| acc.max(x));
        let avg_memory_mb = memory_usages.iter().sum::<f64>() / memory_usages.len() as f64;

        PerformanceSummary {
            total_steps,
            current_loss: current_metrics.loss,
            best_loss,
            avg_loss,
            current_throughput: current_metrics.throughput_samples_per_sec,
            avg_throughput,
            peak_memory_mb,
            avg_memory_mb,
        }
    }

    /// Analyze performance trends.
    pub fn analyze_performance_trends(&self) -> PerformanceTrends {
        if self.performance_history.len() < 10 {
            return PerformanceTrends::default();
        }

        let losses: Vec<f64> = self.performance_history.iter().map(|m| m.loss).collect();
        let throughputs: Vec<f64> =
            self.performance_history.iter().map(|m| m.throughput_samples_per_sec).collect();
        let memory_usages: Vec<f64> =
            self.performance_history.iter().map(|m| m.memory_usage_mb).collect();

        let loss_trend = self.compute_trend(&losses);
        let throughput_trend = self.compute_trend(&throughputs);
        let memory_trend = self.compute_trend(&memory_usages);

        let loss_volatility = self.compute_volatility(&losses);
        let throughput_volatility = self.compute_volatility(&throughputs);

        PerformanceTrends {
            loss_trend,
            throughput_trend,
            memory_trend,
            loss_volatility,
            throughput_volatility,
            trend_confidence: self.compute_trend_confidence(&losses),
        }
    }

    /// Check for performance anomalies.
    pub fn detect_performance_anomalies(&self) -> Vec<PerformanceAnomaly> {
        let mut anomalies = Vec::new();

        if self.performance_history.len() < 5 {
            return anomalies;
        }

        // Check for memory leaks
        if let Some(anomaly) = self.detect_memory_leak() {
            anomalies.push(anomaly);
        }

        // Check for performance degradation
        if let Some(anomaly) = self.detect_performance_degradation() {
            anomalies.push(anomaly);
        }

        // Check for training instability
        if let Some(anomaly) = self.detect_training_instability() {
            anomalies.push(anomaly);
        }

        // Check for throughput drops
        if let Some(anomaly) = self.detect_throughput_drops() {
            anomalies.push(anomaly);
        }

        anomalies
    }

    /// Generate performance optimization recommendations.
    pub fn generate_optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        let summary = self.generate_performance_summary();

        // Memory optimization recommendations
        if summary.peak_memory_mb > self.thresholds.max_memory_mb {
            recommendations.push(OptimizationRecommendation {
                category: "Memory".to_string(),
                priority: PerformanceRecommendationPriority::High,
                description: "High memory usage detected".to_string(),
                suggestion: "Consider reducing batch size or using gradient checkpointing"
                    .to_string(),
                expected_improvement: 0.3,
            });
        }

        // Throughput optimization recommendations
        if summary.avg_throughput < self.thresholds.min_throughput {
            recommendations.push(OptimizationRecommendation {
                category: "Throughput".to_string(),
                priority: PerformanceRecommendationPriority::Medium,
                description: "Low throughput detected".to_string(),
                suggestion: "Consider increasing batch size or optimizing data loading".to_string(),
                expected_improvement: 0.4,
            });
        }

        // Loss optimization recommendations
        let trends = self.analyze_performance_trends();
        if trends.loss_trend > 0.01 {
            recommendations.push(OptimizationRecommendation {
                category: "Training".to_string(),
                priority: PerformanceRecommendationPriority::High,
                description: "Loss is increasing".to_string(),
                suggestion: "Consider reducing learning rate or adding regularization".to_string(),
                expected_improvement: 0.5,
            });
        }

        recommendations
    }

    /// Compute linear trend for a series of values.
    fn compute_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Compute volatility (coefficient of variation) for a series of values.
    fn compute_volatility(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        let std_dev = variance.sqrt();

        if mean == 0.0 {
            0.0
        } else {
            std_dev / mean.abs()
        }
    }

    /// Compute confidence in trend analysis.
    fn compute_trend_confidence(&self, values: &[f64]) -> f64 {
        if values.len() < 10 {
            return 0.0;
        }

        let trend = self.compute_trend(values);
        let volatility = self.compute_volatility(values);

        // Higher confidence for stronger trends with lower volatility
        let trend_strength = trend.abs();
        let confidence = trend_strength / (1.0 + volatility);
        confidence.min(1.0)
    }

    /// Detect memory leak patterns.
    fn detect_memory_leak(&self) -> Option<PerformanceAnomaly> {
        if self.performance_history.len() < 10 {
            return None;
        }

        let recent_metrics = &self.performance_history[self.performance_history.len() - 10..];
        let memory_usages: Vec<f64> = recent_metrics.iter().map(|m| m.memory_usage_mb).collect();
        let memory_trend = self.compute_trend(&memory_usages);

        // Consider it a memory leak if memory is consistently growing
        if memory_trend > 10.0 {
            // More than 10MB increase per step on average
            Some(PerformanceAnomaly {
                anomaly_type: AnomalyType::MemoryLeak,
                severity: AnomalySeverity::High,
                description: format!("Memory usage increasing at {:.1} MB/step", memory_trend),
                detected_at_step: self.performance_history.last().unwrap().training_step,
                confidence: 0.8,
            })
        } else {
            None
        }
    }

    /// Detect performance degradation.
    fn detect_performance_degradation(&self) -> Option<PerformanceAnomaly> {
        if self.performance_history.len() < 20 {
            return None;
        }

        let recent_metrics = &self.performance_history[self.performance_history.len() - 10..];
        let previous_metrics = &self.performance_history
            [self.performance_history.len() - 20..self.performance_history.len() - 10];

        let recent_avg_loss: f64 =
            recent_metrics.iter().map(|m| m.loss).sum::<f64>() / recent_metrics.len() as f64;
        let previous_avg_loss: f64 =
            previous_metrics.iter().map(|m| m.loss).sum::<f64>() / previous_metrics.len() as f64;

        let degradation_percent =
            ((recent_avg_loss - previous_avg_loss) / previous_avg_loss) * 100.0;

        if degradation_percent > self.thresholds.max_loss_increase_percent {
            Some(PerformanceAnomaly {
                anomaly_type: AnomalyType::PerformanceDegradation,
                severity: AnomalySeverity::High,
                description: format!("Performance degraded by {:.1}%", degradation_percent),
                detected_at_step: self.performance_history.last().unwrap().training_step,
                confidence: 0.9,
            })
        } else {
            None
        }
    }

    /// Detect training instability.
    fn detect_training_instability(&self) -> Option<PerformanceAnomaly> {
        if self.performance_history.len() < 10 {
            return None;
        }

        let recent_metrics = &self.performance_history[self.performance_history.len() - 10..];
        let losses: Vec<f64> = recent_metrics.iter().map(|m| m.loss).collect();
        let volatility = self.compute_volatility(&losses);

        if volatility > self.thresholds.max_loss_variance {
            Some(PerformanceAnomaly {
                anomaly_type: AnomalyType::TrainingInstability,
                severity: AnomalySeverity::Medium,
                description: format!("High loss volatility: {:.3}", volatility),
                detected_at_step: self.performance_history.last().unwrap().training_step,
                confidence: 0.7,
            })
        } else {
            None
        }
    }

    /// Detect throughput drops.
    fn detect_throughput_drops(&self) -> Option<PerformanceAnomaly> {
        if self.performance_history.len() < 10 {
            return None;
        }

        let recent_metrics = &self.performance_history[self.performance_history.len() - 5..];
        let avg_recent_throughput: f64 =
            recent_metrics.iter().map(|m| m.throughput_samples_per_sec).sum::<f64>()
                / recent_metrics.len() as f64;

        if avg_recent_throughput < self.thresholds.min_throughput {
            Some(PerformanceAnomaly {
                anomaly_type: AnomalyType::ThroughputDrop,
                severity: AnomalySeverity::Medium,
                description: format!("Low throughput: {:.1} samples/sec", avg_recent_throughput),
                detected_at_step: self.performance_history.last().unwrap().training_step,
                confidence: 0.8,
            })
        } else {
            None
        }
    }

    /// Clear performance history.
    pub fn clear(&mut self) {
        self.performance_history.clear();
    }
}

impl Default for PerformanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance trends analysis results.
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    /// Loss trend (slope)
    pub loss_trend: f64,
    /// Throughput trend (slope)
    pub throughput_trend: f64,
    /// Memory usage trend (slope)
    pub memory_trend: f64,
    /// Loss volatility (coefficient of variation)
    pub loss_volatility: f64,
    /// Throughput volatility (coefficient of variation)
    pub throughput_volatility: f64,
    /// Confidence in trend analysis
    pub trend_confidence: f64,
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            loss_trend: 0.0,
            throughput_trend: 0.0,
            memory_trend: 0.0,
            loss_volatility: 0.0,
            throughput_volatility: 0.0,
            trend_confidence: 0.0,
        }
    }
}

/// Performance anomaly detection results.
#[derive(Debug, Clone)]
pub struct PerformanceAnomaly {
    /// Type of anomaly detected
    pub anomaly_type: AnomalyType,
    /// Severity of the anomaly
    pub severity: AnomalySeverity,
    /// Description of the anomaly
    pub description: String,
    /// Training step when anomaly was detected
    pub detected_at_step: usize,
    /// Confidence in the detection
    pub confidence: f64,
}

/// Types of performance anomalies.
#[derive(Debug, Clone)]
pub enum AnomalyType {
    /// Memory leak detected
    MemoryLeak,
    /// Performance degradation detected
    PerformanceDegradation,
    /// Training instability detected
    TrainingInstability,
    /// Throughput drop detected
    ThroughputDrop,
}

/// Severity levels for anomalies.
#[derive(Debug, Clone)]
pub enum AnomalySeverity {
    /// Low severity anomaly
    Low,
    /// Medium severity anomaly
    Medium,
    /// High severity anomaly
    High,
    /// Critical severity anomaly
    Critical,
}

/// Performance optimization recommendation.
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Category of optimization
    pub category: String,
    /// Priority of the recommendation
    pub priority: PerformanceRecommendationPriority,
    /// Description of the issue
    pub description: String,
    /// Suggested optimization
    pub suggestion: String,
    /// Expected improvement (0.0 to 1.0)
    pub expected_improvement: f64,
}

/// Priority levels for recommendations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum PerformanceRecommendationPriority {
    /// Low priority recommendation
    Low,
    /// Medium priority recommendation
    Medium,
    /// High priority recommendation
    High,
    /// Critical priority recommendation
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_metrics(
        step: usize,
        loss: f64,
        memory: f64,
        throughput: f64,
    ) -> ModelPerformanceMetrics {
        ModelPerformanceMetrics {
            training_step: step,
            loss,
            accuracy: Some(0.8),
            learning_rate: 0.001,
            batch_size: 32,
            throughput_samples_per_sec: throughput,
            memory_usage_mb: memory,
            gpu_utilization: Some(0.9),
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_performance_analyzer_creation() {
        let analyzer = PerformanceAnalyzer::new();
        assert_eq!(analyzer.performance_history.len(), 0);
        assert_eq!(analyzer.max_history_length, 1000);
    }

    #[test]
    fn test_record_performance() {
        let mut analyzer = PerformanceAnalyzer::new();
        let metrics = create_test_metrics(1, 0.5, 1000.0, 100.0);

        analyzer.record_performance(metrics);
        assert_eq!(analyzer.performance_history.len(), 1);
    }

    #[test]
    fn test_performance_summary() {
        let mut analyzer = PerformanceAnalyzer::new();

        // Add some test data
        for i in 1..=5 {
            let metrics = create_test_metrics(i, 1.0 / i as f64, 1000.0, 100.0);
            analyzer.record_performance(metrics);
        }

        let summary = analyzer.generate_performance_summary();
        assert_eq!(summary.total_steps, 5);
        assert!(summary.best_loss < summary.avg_loss);
    }

    #[test]
    fn test_trend_computation() {
        let analyzer = PerformanceAnalyzer::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend = analyzer.compute_trend(&values);
        assert!(trend > 0.0); // Should be positive trend
    }

    #[test]
    fn test_memory_leak_detection() {
        let mut analyzer = PerformanceAnalyzer::new();

        // Add metrics with increasing memory usage
        for i in 1..=15 {
            let metrics = create_test_metrics(i, 0.5, 1000.0 + (i as f64 * 50.0), 100.0);
            analyzer.record_performance(metrics);
        }

        let anomalies = analyzer.detect_performance_anomalies();
        assert!(!anomalies.is_empty());
        assert!(matches!(anomalies[0].anomaly_type, AnomalyType::MemoryLeak));
    }
}
