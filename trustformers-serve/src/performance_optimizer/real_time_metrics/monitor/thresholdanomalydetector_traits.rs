//! # ThresholdAnomalyDetector - Trait Implementations
//!
//! This module contains trait implementations for `ThresholdAnomalyDetector`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `AnomalyDetectionAlgorithm`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::TimestampedMetrics;
use super::types::*;
use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;

use super::functions::AnomalyDetectionAlgorithm;
use super::types::{
    AnomalyAlgorithmStats, AnomalyDetectionConfig, AnomalyEvent, PerformanceBaseline,
    ThresholdAnomalyDetector,
};

impl Default for ThresholdAnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyDetectionAlgorithm for ThresholdAnomalyDetector {
    fn detect_anomaly(
        &self,
        metrics: &TimestampedMetrics,
        baseline: &PerformanceBaseline,
    ) -> Result<Option<AnomalyEvent>> {
        let current_throughput = metrics.metrics.current_throughput;
        let current_latency = metrics.metrics.current_latency.as_secs_f64() * 1000.0;
        let current_cpu = metrics.metrics.current_cpu_utilization as f64;
        let current_memory = metrics.metrics.current_memory_utilization as f64;
        let baseline_latency_ms = baseline.baseline_latency.as_secs_f64() * 1000.0;
        let mut violated_thresholds = Vec::new();
        let mut max_deviation = 0.0f64;
        let mut primary_metric = String::new();
        if self.exceeds_threshold(
            current_throughput,
            baseline.baseline_throughput,
            self.throughput_threshold,
        ) {
            violated_thresholds.push("throughput".to_string());
            let deviation = (current_throughput - baseline.baseline_throughput).abs()
                / baseline.baseline_throughput;
            if deviation > max_deviation {
                max_deviation = deviation;
                primary_metric = "throughput".to_string();
            }
        }
        if self.exceeds_threshold(current_latency, baseline_latency_ms, self.latency_threshold) {
            violated_thresholds.push("latency".to_string());
            let deviation = (current_latency - baseline_latency_ms).abs() / baseline_latency_ms;
            if deviation > max_deviation {
                max_deviation = deviation;
                primary_metric = "latency".to_string();
            }
        }
        if self.exceeds_threshold(
            current_cpu,
            baseline.baseline_cpu as f64,
            self.cpu_threshold,
        ) {
            violated_thresholds.push("cpu".to_string());
            let deviation =
                (current_cpu - baseline.baseline_cpu as f64).abs() / baseline.baseline_cpu as f64;
            if deviation > max_deviation {
                max_deviation = deviation;
                primary_metric = "cpu".to_string();
            }
        }
        if self.exceeds_threshold(
            current_memory,
            baseline.baseline_memory as f64,
            self.memory_threshold,
        ) {
            violated_thresholds.push("memory".to_string());
            let deviation = (current_memory - baseline.baseline_memory as f64).abs()
                / baseline.baseline_memory as f64;
            if deviation > max_deviation {
                max_deviation = deviation;
                primary_metric = "memory".to_string();
            }
        }
        if !violated_thresholds.is_empty() {
            let severity = match max_deviation {
                d if d > 1.0 => SeverityLevel::Critical,
                d if d > 0.5 => SeverityLevel::High,
                d if d > 0.3 => SeverityLevel::Medium,
                _ => SeverityLevel::Low,
            };
            let anomaly = AnomalyEvent {
                timestamp: Utc::now(),
                anomaly_type: "threshold_violation".to_string(),
                severity,
                description: format!(
                    "Threshold violation detected for {} with {:.1}% deviation",
                    primary_metric,
                    max_deviation * 100.0
                ),
                affected_metrics: violated_thresholds,
                score: (max_deviation / 2.0).min(1.0) as f32,
                confidence: 0.9,
                expected_value: match primary_metric.as_str() {
                    "throughput" => baseline.baseline_throughput,
                    "latency" => baseline_latency_ms,
                    "cpu" => baseline.baseline_cpu as f64,
                    "memory" => baseline.baseline_memory as f64,
                    _ => 0.0,
                },
                actual_value: match primary_metric.as_str() {
                    "throughput" => current_throughput,
                    "latency" => current_latency,
                    "cpu" => current_cpu,
                    "memory" => current_memory,
                    _ => 0.0,
                },
                deviation: max_deviation,
                detection_algorithm: "threshold".to_string(),
                context: {
                    let mut ctx = HashMap::new();
                    ctx.insert("primary_metric".to_string(), primary_metric.clone());
                    ctx.insert(
                        "deviation_percent".to_string(),
                        format!("{:.1}", max_deviation * 100.0),
                    );
                    ctx
                },
                recommendations: vec![
                    format!("Investigate {} performance degradation", primary_metric),
                    "Check system resources and external dependencies".to_string(),
                    "Consider adjusting thresholds if this is expected behavior".to_string(),
                ],
            };
            Ok(Some(anomaly))
        } else {
            Ok(None)
        }
    }
    fn name(&self) -> &str {
        "threshold"
    }
    fn confidence(&self) -> f32 {
        0.95
    }
    fn update_parameters(&mut self, config: &AnomalyDetectionConfig) -> Result<()> {
        let base_sensitivity = config.sensitivity;
        self.throughput_threshold = 0.3 * (1.0 - base_sensitivity * 0.5);
        self.latency_threshold = 0.5 * (1.0 - base_sensitivity * 0.5);
        self.cpu_threshold = 0.2 * (1.0 - base_sensitivity * 0.5);
        self.memory_threshold = 0.25 * (1.0 - base_sensitivity * 0.5);
        Ok(())
    }
    fn get_statistics(&self) -> AnomalyAlgorithmStats {
        self.stats.clone()
    }
}
