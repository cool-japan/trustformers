//! Metrics collection and tracking for A/B tests

use super::Variant;
use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Types of metrics that can be tracked
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    /// Response latency in milliseconds
    Latency,
    /// Throughput in requests per second
    Throughput,
    /// Error rate (0.0 to 1.0)
    ErrorRate,
    /// Model accuracy (0.0 to 1.0)
    Accuracy,
    /// Memory usage in MB
    MemoryUsage,
    /// User engagement score
    EngagementScore,
    /// Conversion rate (0.0 to 1.0)
    ConversionRate,
    /// Custom metric with name
    Custom(String),
}

impl MetricType {
    /// Returns true if lower values are better for this metric
    pub fn lower_is_better(&self) -> bool {
        match self {
            MetricType::Latency => true,
            MetricType::ErrorRate => true,
            MetricType::MemoryUsage => true,
            MetricType::Throughput => false,
            MetricType::Accuracy => false,
            MetricType::EngagementScore => false,
            MetricType::ConversionRate => false,
            MetricType::Custom(_) => false, // Default to higher is better for custom metrics
        }
    }
}

/// Metric value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    /// Numeric value
    Numeric(f64),
    /// Boolean value
    Boolean(bool),
    /// Count/integer value
    Count(u64),
    /// Duration in milliseconds
    Duration(u64),
}

impl MetricValue {
    /// Convert to f64 for analysis
    pub fn as_f64(&self) -> f64 {
        match self {
            MetricValue::Numeric(v) => *v,
            MetricValue::Boolean(v) => {
                if *v {
                    1.0
                } else {
                    0.0
                }
            },
            MetricValue::Count(v) => *v as f64,
            MetricValue::Duration(v) => *v as f64,
        }
    }
}

/// A single metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDataPoint {
    /// Timestamp when metric was recorded
    pub timestamp: DateTime<Utc>,
    /// The metric value
    pub value: MetricValue,
    /// Optional metadata
    pub metadata: Option<HashMap<String, String>>,
}

/// Aggregated metrics for a variant
#[derive(Debug, Clone)]
pub struct AggregatedMetrics {
    /// Variant this is for
    pub variant: Variant,
    /// Metric type
    pub metric_type: MetricType,
    /// Number of data points
    pub count: usize,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Percentiles (50th, 90th, 95th, 99th)
    pub percentiles: HashMap<u8, f64>,
}

/// Metric collector for experiments
pub struct MetricCollector {
    /// Storage for metrics by experiment ID
    metrics: Arc<RwLock<HashMap<String, ExperimentMetrics>>>,
    /// Buffer for batch writes
    buffer: Arc<RwLock<Vec<BufferedMetric>>>,
    /// Buffer flush size
    buffer_size: usize,
}

/// Metrics for a single experiment
#[derive(Default)]
struct ExperimentMetrics {
    /// Metrics by variant and type
    data: HashMap<(String, MetricType), Vec<MetricDataPoint>>,
}

/// Buffered metric for batch processing
struct BufferedMetric {
    experiment_id: String,
    variant: Variant,
    metric_type: MetricType,
    data_point: MetricDataPoint,
}

impl Default for MetricCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricCollector {
    /// Create a new metric collector
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            buffer: Arc::new(RwLock::new(Vec::new())),
            buffer_size: 100,
        }
    }

    /// Record a metric
    pub fn record(
        &self,
        experiment_id: &str,
        variant: &Variant,
        metric_type: MetricType,
        value: MetricValue,
    ) -> Result<()> {
        let data_point = MetricDataPoint {
            timestamp: Utc::now(),
            value,
            metadata: None,
        };

        self.record_with_metadata(experiment_id, variant, metric_type, data_point)
    }

    /// Record a metric with metadata
    pub fn record_with_metadata(
        &self,
        experiment_id: &str,
        variant: &Variant,
        metric_type: MetricType,
        data_point: MetricDataPoint,
    ) -> Result<()> {
        let buffered = BufferedMetric {
            experiment_id: experiment_id.to_string(),
            variant: variant.clone(),
            metric_type,
            data_point,
        };

        let mut buffer = self.buffer.write();
        buffer.push(buffered);

        // Flush if buffer is full
        if buffer.len() >= self.buffer_size {
            drop(buffer); // Release lock before flushing
            self.flush_buffer()?;
        }

        Ok(())
    }

    /// Flush buffered metrics
    pub fn flush_buffer(&self) -> Result<()> {
        let mut buffer = self.buffer.write();
        if buffer.is_empty() {
            return Ok(());
        }

        let mut metrics = self.metrics.write();

        for buffered in buffer.drain(..) {
            let experiment_metrics = metrics.entry(buffered.experiment_id).or_default();

            let key = (buffered.variant.name().to_string(), buffered.metric_type);
            experiment_metrics.data.entry(key).or_default().push(buffered.data_point);
        }

        Ok(())
    }

    /// Get all metrics for an experiment
    pub fn get_metrics(
        &self,
        experiment_id: &str,
    ) -> Result<HashMap<(Variant, MetricType), Vec<MetricDataPoint>>> {
        // Flush buffer first
        self.flush_buffer()?;

        let metrics = self.metrics.read();
        let experiment_metrics = metrics
            .get(experiment_id)
            .ok_or_else(|| anyhow::anyhow!("No metrics found for experiment"))?;

        let mut result = HashMap::new();
        for ((variant_name, metric_type), data_points) in &experiment_metrics.data {
            // Reconstruct variant (simplified - in practice would store full variant)
            let variant = Variant::new(variant_name, "");
            result.insert((variant, metric_type.clone()), data_points.clone());
        }

        Ok(result)
    }

    /// Get aggregated metrics for a specific variant and metric type
    pub fn get_aggregated_metrics(
        &self,
        experiment_id: &str,
        variant: &Variant,
        metric_type: &MetricType,
    ) -> Result<AggregatedMetrics> {
        self.flush_buffer()?;

        let metrics = self.metrics.read();
        let experiment_metrics = metrics
            .get(experiment_id)
            .ok_or_else(|| anyhow::anyhow!("No metrics found for experiment"))?;

        let key = (variant.name().to_string(), metric_type.clone());
        let data_points = experiment_metrics
            .data
            .get(&key)
            .ok_or_else(|| anyhow::anyhow!("No metrics found for variant and type"))?;

        self.calculate_aggregates(variant.clone(), metric_type.clone(), data_points)
    }

    /// Calculate aggregate statistics
    fn calculate_aggregates(
        &self,
        variant: Variant,
        metric_type: MetricType,
        data_points: &[MetricDataPoint],
    ) -> Result<AggregatedMetrics> {
        if data_points.is_empty() {
            anyhow::bail!("No data points to aggregate");
        }

        let values: Vec<f64> = data_points.iter().map(|dp| dp.value.as_f64()).collect();

        let count = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;

        let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Calculate percentiles
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).expect("Partial comparison failed"));

        let percentiles = vec![50, 90, 95, 99]
            .into_iter()
            .map(|p| {
                let index = ((p as f64 / 100.0) * (count - 1) as f64) as usize;
                (p, sorted_values[index])
            })
            .collect();

        Ok(AggregatedMetrics {
            variant,
            metric_type,
            count,
            mean,
            std_dev,
            min,
            max,
            percentiles,
        })
    }

    /// Clear metrics for an experiment
    pub fn clear_experiment_metrics(&self, experiment_id: &str) -> Result<()> {
        self.flush_buffer()?;
        self.metrics.write().remove(experiment_id);
        Ok(())
    }

    /// Get metric time series for visualization
    pub fn get_time_series(
        &self,
        experiment_id: &str,
        variant: &Variant,
        metric_type: &MetricType,
    ) -> Result<Vec<(DateTime<Utc>, f64)>> {
        self.flush_buffer()?;

        let metrics = self.metrics.read();
        let experiment_metrics = metrics
            .get(experiment_id)
            .ok_or_else(|| anyhow::anyhow!("No metrics found for experiment"))?;

        let key = (variant.name().to_string(), metric_type.clone());
        let data_points = experiment_metrics
            .data
            .get(&key)
            .ok_or_else(|| anyhow::anyhow!("No metrics found for variant and type"))?;

        Ok(data_points.iter().map(|dp| (dp.timestamp, dp.value.as_f64())).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_recording() {
        let collector = MetricCollector::new();
        let variant = Variant::new("test", "model-v1");

        // Record some metrics
        collector
            .record(
                "exp1",
                &variant,
                MetricType::Latency,
                MetricValue::Duration(100),
            )
            .unwrap();
        collector
            .record(
                "exp1",
                &variant,
                MetricType::Latency,
                MetricValue::Duration(150),
            )
            .unwrap();
        collector
            .record(
                "exp1",
                &variant,
                MetricType::Latency,
                MetricValue::Duration(120),
            )
            .unwrap();

        // Get aggregated metrics
        let aggregated = collector
            .get_aggregated_metrics("exp1", &variant, &MetricType::Latency)
            .unwrap();

        assert_eq!(aggregated.count, 3);
        assert_eq!(aggregated.mean, 123.33333333333333);
        assert_eq!(aggregated.min, 100.0);
        assert_eq!(aggregated.max, 150.0);
    }

    #[test]
    fn test_metric_types() {
        let collector = MetricCollector::new();
        let variant = Variant::new("test", "model-v1");

        // Test different metric types
        collector
            .record(
                "exp1",
                &variant,
                MetricType::Accuracy,
                MetricValue::Numeric(0.95),
            )
            .unwrap();
        collector
            .record(
                "exp1",
                &variant,
                MetricType::ErrorRate,
                MetricValue::Numeric(0.02),
            )
            .unwrap();
        collector
            .record(
                "exp1",
                &variant,
                MetricType::ConversionRate,
                MetricValue::Boolean(true),
            )
            .unwrap();

        // Verify metrics were recorded
        let metrics = collector.get_metrics("exp1").unwrap();
        assert!(metrics.len() >= 3);
    }

    #[test]
    fn test_time_series() {
        let collector = MetricCollector::new();
        let variant = Variant::new("test", "model-v1");

        // Record metrics over time
        for i in 0..10 {
            collector
                .record(
                    "exp1",
                    &variant,
                    MetricType::Throughput,
                    MetricValue::Numeric(100.0 + i as f64),
                )
                .unwrap();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Get time series
        let time_series =
            collector.get_time_series("exp1", &variant, &MetricType::Throughput).unwrap();

        assert_eq!(time_series.len(), 10);

        // Verify values are in order
        for (i, (_, value)) in time_series.iter().enumerate().take(10) {
            assert_eq!(*value, 100.0 + i as f64);
        }
    }
}
