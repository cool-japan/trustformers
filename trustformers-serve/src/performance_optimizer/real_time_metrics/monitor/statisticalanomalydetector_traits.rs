//! # StatisticalAnomalyDetector - Trait Implementations
//!
//! This module contains trait implementations for `StatisticalAnomalyDetector`.
//!
//! ## Implemented Traits
//!
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
    StatisticalAnomalyDetector,
};

impl AnomalyDetectionAlgorithm for StatisticalAnomalyDetector {
    fn detect_anomaly(
        &self,
        metrics: &TimestampedMetrics,
        baseline: &PerformanceBaseline,
    ) -> Result<Option<AnomalyEvent>> {
        let current_throughput = metrics.metrics.current_throughput;
        let current_latency = metrics.metrics.current_latency.as_secs_f64() * 1000.0;
        let current_cpu = metrics.metrics.current_cpu_utilization as f64;
        let current_memory = metrics.metrics.current_memory_utilization as f64;
        let throughput_z = self.calculate_z_score(
            current_throughput,
            baseline.baseline_throughput,
            (baseline.variability_bounds.throughput_upper
                - baseline.variability_bounds.throughput_lower)
                / 4.0,
        );
        let latency_z = self.calculate_z_score(
            current_latency,
            baseline.baseline_latency.as_secs_f64() * 1000.0,
            (baseline.variability_bounds.latency_upper - baseline.variability_bounds.latency_lower)
                * 1000.0
                / 4.0,
        );
        let cpu_z = self.calculate_z_score(
            current_cpu,
            baseline.baseline_cpu as f64,
            (baseline.variability_bounds.cpu_upper - baseline.variability_bounds.cpu_lower) as f64
                / 4.0,
        );
        let memory_z = self.calculate_z_score(
            current_memory,
            baseline.baseline_memory as f64,
            (baseline.variability_bounds.memory_upper - baseline.variability_bounds.memory_lower)
                as f64
                / 4.0,
        );
        let max_z = [
            throughput_z.abs(),
            latency_z.abs(),
            cpu_z.abs(),
            memory_z.abs(),
        ]
        .iter()
        .fold(0.0_f64, |a, &b| a.max(b));
        if max_z > self.threshold as f64 {
            let severity = match max_z {
                z if z > 4.0 => SeverityLevel::Critical,
                z if z > 3.0 => SeverityLevel::High,
                z if z > 2.5 => SeverityLevel::Medium,
                _ => SeverityLevel::Low,
            };
            let affected_metrics = vec![
                if throughput_z.abs() > self.threshold as f64 {
                    Some("throughput".to_string())
                } else {
                    None
                },
                if latency_z.abs() > self.threshold as f64 {
                    Some("latency".to_string())
                } else {
                    None
                },
                if cpu_z.abs() > self.threshold as f64 { Some("cpu".to_string()) } else { None },
                if memory_z.abs() > self.threshold as f64 {
                    Some("memory".to_string())
                } else {
                    None
                },
            ]
            .into_iter()
            .flatten()
            .collect();
            let anomaly = AnomalyEvent {
                timestamp: Utc::now(),
                anomaly_type: "statistical_deviation".to_string(),
                severity,
                description: format!("Statistical anomaly detected with Z-score: {:.2}", max_z),
                affected_metrics,
                score: (max_z / 5.0).min(1.0) as f32,
                confidence: ((max_z - self.threshold as f64) / (5.0 - self.threshold as f64))
                    .clamp(0.0, 1.0) as f32,
                expected_value: baseline.baseline_throughput,
                actual_value: current_throughput,
                deviation: max_z,
                detection_algorithm: "statistical".to_string(),
                context: {
                    let mut ctx = HashMap::new();
                    ctx.insert("throughput_z".to_string(), throughput_z.to_string());
                    ctx.insert("latency_z".to_string(), latency_z.to_string());
                    ctx.insert("cpu_z".to_string(), cpu_z.to_string());
                    ctx.insert("memory_z".to_string(), memory_z.to_string());
                    ctx
                },
                recommendations: vec![
                    "Review system load and resource allocation".to_string(),
                    "Check for external factors affecting performance".to_string(),
                    "Consider scaling resources if needed".to_string(),
                ],
            };
            Ok(Some(anomaly))
        } else {
            Ok(None)
        }
    }
    fn name(&self) -> &str {
        "statistical"
    }
    fn confidence(&self) -> f32 {
        0.85
    }
    fn update_parameters(&mut self, config: &AnomalyDetectionConfig) -> Result<()> {
        self.threshold = config.sensitivity * 3.0;
        Ok(())
    }
    fn get_statistics(&self) -> AnomalyAlgorithmStats {
        self.stats.clone()
    }
}
