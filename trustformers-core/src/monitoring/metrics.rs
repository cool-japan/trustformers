// Metrics collection and aggregation system
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Metrics collector for tracking various performance and operational metrics
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    config: MetricsCollectorConfig,
    metrics: HashMap<String, MetricTimeSeries>,
    counters: HashMap<String, Counter>,
    gauges: HashMap<String, Gauge>,
    histograms: HashMap<String, Histogram>,
    start_time: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsCollectorConfig {
    pub enabled: bool,
    pub collection_interval_ms: u64,
    pub max_time_series_points: usize,
    pub enable_histograms: bool,
    pub histogram_buckets: Vec<f64>,
    pub retention_duration_hours: u64,
}

impl Default for MetricsCollectorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval_ms: 1000,
            max_time_series_points: 10000,
            enable_histograms: true,
            histogram_buckets: vec![
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ],
            retention_duration_hours: 24,
        }
    }
}

/// Time series metric with timestamped values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTimeSeries {
    pub name: String,
    pub metric_type: MetricType,
    pub values: Vec<TimestampedValue>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedValue {
    pub timestamp: u64,
    pub value: f64,
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
    Timer,
}

/// Counter metric for tracking cumulative values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counter {
    pub name: String,
    pub value: f64,
    pub labels: HashMap<String, String>,
    pub last_updated: u64,
}

/// Gauge metric for tracking instantaneous values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gauge {
    pub name: String,
    pub value: f64,
    pub labels: HashMap<String, String>,
    pub last_updated: u64,
}

/// Histogram metric for tracking value distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    pub name: String,
    pub buckets: Vec<HistogramBucket>,
    pub count: u64,
    pub sum: f64,
    pub labels: HashMap<String, String>,
    pub last_updated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBucket {
    pub upper_bound: f64,
    pub count: u64,
}

/// Aggregated metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub collection_period: Duration,
    pub total_metrics: usize,
    pub counter_metrics: Vec<CounterSummary>,
    pub gauge_metrics: Vec<GaugeSummary>,
    pub histogram_metrics: Vec<HistogramSummary>,
    pub time_series_stats: TimeSeriesStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterSummary {
    pub name: String,
    pub value: f64,
    pub rate_per_second: f64,
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaugeSummary {
    pub name: String,
    pub current_value: f64,
    pub min_value: f64,
    pub max_value: f64,
    pub average_value: f64,
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramSummary {
    pub name: String,
    pub count: u64,
    pub sum: f64,
    pub mean: f64,
    pub percentiles: HashMap<String, f64>, // P50, P90, P95, P99
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesStats {
    pub total_data_points: usize,
    pub oldest_timestamp: u64,
    pub newest_timestamp: u64,
    pub average_points_per_series: f64,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            config: MetricsCollectorConfig::default(),
            metrics: HashMap::new(),
            counters: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
            start_time: Instant::now(),
        }
    }

    pub fn with_config(config: MetricsCollectorConfig) -> Self {
        Self {
            config,
            metrics: HashMap::new(),
            counters: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
            start_time: Instant::now(),
        }
    }

    /// Increment a counter metric
    pub fn increment_counter(
        &mut self,
        name: &str,
        value: f64,
        labels: Option<HashMap<String, String>>,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let timestamp = self.get_current_timestamp();
        let labels = labels.unwrap_or_default();
        let metric_key = self.create_metric_key(name, &labels);

        let counter_value = {
            let counter = self.counters.entry(metric_key.clone()).or_insert_with(|| Counter {
                name: name.to_string(),
                value: 0.0,
                labels: labels.clone(),
                last_updated: timestamp,
            });

            counter.value += value;
            counter.last_updated = timestamp;
            counter.value
        };

        // Also add to time series
        self.add_time_series_point(name, MetricType::Counter, counter_value, labels)?;

        Ok(())
    }

    /// Set a gauge metric value
    pub fn set_gauge(
        &mut self,
        name: &str,
        value: f64,
        labels: Option<HashMap<String, String>>,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let timestamp = self.get_current_timestamp();
        let labels = labels.unwrap_or_default();
        let metric_key = self.create_metric_key(name, &labels);

        let gauge = self.gauges.entry(metric_key.clone()).or_insert_with(|| Gauge {
            name: name.to_string(),
            value: 0.0,
            labels: labels.clone(),
            last_updated: timestamp,
        });

        gauge.value = value;
        gauge.last_updated = timestamp;

        // Also add to time series
        self.add_time_series_point(name, MetricType::Gauge, value, labels)?;

        Ok(())
    }

    /// Record a value in a histogram
    pub fn record_histogram(
        &mut self,
        name: &str,
        value: f64,
        labels: Option<HashMap<String, String>>,
    ) -> Result<()> {
        if !self.config.enabled || !self.config.enable_histograms {
            return Ok(());
        }

        let timestamp = self.get_current_timestamp();
        let labels = labels.unwrap_or_default();
        let metric_key = self.create_metric_key(name, &labels);

        let histogram = self.histograms.entry(metric_key.clone()).or_insert_with(|| {
            let buckets = self
                .config
                .histogram_buckets
                .iter()
                .map(|&upper_bound| HistogramBucket {
                    upper_bound,
                    count: 0,
                })
                .collect();

            Histogram {
                name: name.to_string(),
                buckets,
                count: 0,
                sum: 0.0,
                labels: labels.clone(),
                last_updated: timestamp,
            }
        });

        // Update histogram
        histogram.count += 1;
        histogram.sum += value;
        histogram.last_updated = timestamp;

        // Update buckets
        for bucket in &mut histogram.buckets {
            if value <= bucket.upper_bound {
                bucket.count += 1;
            }
        }

        // Also add to time series
        self.add_time_series_point(name, MetricType::Histogram, value, labels)?;

        Ok(())
    }

    /// Record execution time
    pub fn record_timing(
        &mut self,
        name: &str,
        duration: Duration,
        labels: Option<HashMap<String, String>>,
    ) -> Result<()> {
        let duration_seconds = duration.as_secs_f64();
        self.record_histogram(name, duration_seconds, labels)
    }

    /// Time a function execution
    pub fn time_function<T, F>(
        &mut self,
        name: &str,
        labels: Option<HashMap<String, String>>,
        func: F,
    ) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        let start = Instant::now();
        let result = func()?;
        let duration = start.elapsed();

        self.record_timing(name, duration, labels)?;
        Ok(result)
    }

    /// Get current metrics summary
    pub fn collect_metrics(&self) -> Result<HashMap<String, f64>> {
        let mut metrics = HashMap::new();

        // Add counter metrics
        for counter in self.counters.values() {
            metrics.insert(counter.name.clone(), counter.value);
        }

        // Add gauge metrics
        for gauge in self.gauges.values() {
            metrics.insert(gauge.name.clone(), gauge.value);
        }

        // Add histogram summary metrics
        for histogram in self.histograms.values() {
            let mean =
                if histogram.count > 0 { histogram.sum / histogram.count as f64 } else { 0.0 };

            metrics.insert(format!("{}_count", histogram.name), histogram.count as f64);
            metrics.insert(format!("{}_sum", histogram.name), histogram.sum);
            metrics.insert(format!("{}_mean", histogram.name), mean);
        }

        Ok(metrics)
    }

    /// Generate comprehensive metrics summary
    pub fn generate_summary(&self) -> Result<MetricsSummary> {
        let collection_period = self.start_time.elapsed();

        // Counter summaries
        let mut counter_metrics = Vec::new();
        for counter in self.counters.values() {
            let rate_per_second = if collection_period.as_secs() > 0 {
                counter.value / collection_period.as_secs_f64()
            } else {
                0.0
            };

            counter_metrics.push(CounterSummary {
                name: counter.name.clone(),
                value: counter.value,
                rate_per_second,
                labels: counter.labels.clone(),
            });
        }

        // Gauge summaries
        let mut gauge_metrics = Vec::new();
        for gauge in self.gauges.values() {
            // Get statistics from time series if available
            let (min_value, max_value, average_value) =
                if let Some(series) = self.metrics.get(&gauge.name) {
                    self.calculate_gauge_stats(series)
                } else {
                    (gauge.value, gauge.value, gauge.value)
                };

            gauge_metrics.push(GaugeSummary {
                name: gauge.name.clone(),
                current_value: gauge.value,
                min_value,
                max_value,
                average_value,
                labels: gauge.labels.clone(),
            });
        }

        // Histogram summaries
        let mut histogram_metrics = Vec::new();
        for histogram in self.histograms.values() {
            let mean =
                if histogram.count > 0 { histogram.sum / histogram.count as f64 } else { 0.0 };

            let percentiles = self.calculate_histogram_percentiles(histogram);

            histogram_metrics.push(HistogramSummary {
                name: histogram.name.clone(),
                count: histogram.count,
                sum: histogram.sum,
                mean,
                percentiles,
                labels: histogram.labels.clone(),
            });
        }

        // Time series statistics
        let time_series_stats = self.calculate_time_series_stats();

        Ok(MetricsSummary {
            collection_period,
            total_metrics: self.counters.len() + self.gauges.len() + self.histograms.len(),
            counter_metrics,
            gauge_metrics,
            histogram_metrics,
            time_series_stats,
        })
    }

    /// Clear all metrics
    pub fn clear(&mut self) -> Result<()> {
        self.metrics.clear();
        self.counters.clear();
        self.gauges.clear();
        self.histograms.clear();
        self.start_time = Instant::now();
        Ok(())
    }

    /// Export metrics in Prometheus format
    pub fn export_prometheus(&self) -> String {
        let mut output = String::new();

        // Export counters
        for counter in self.counters.values() {
            output.push_str(&format!("# TYPE {} counter\n", counter.name));
            let labels_str = if counter.labels.is_empty() {
                String::new()
            } else {
                let labels: Vec<String> =
                    counter.labels.iter().map(|(k, v)| format!("{}=\"{}\"", k, v)).collect();
                format!("{{{}}}", labels.join(","))
            };
            output.push_str(&format!(
                "{}{} {}\n",
                counter.name, labels_str, counter.value
            ));
        }

        // Export gauges
        for gauge in self.gauges.values() {
            output.push_str(&format!("# TYPE {} gauge\n", gauge.name));
            let labels_str = if gauge.labels.is_empty() {
                String::new()
            } else {
                let labels: Vec<String> =
                    gauge.labels.iter().map(|(k, v)| format!("{}=\"{}\"", k, v)).collect();
                format!("{{{}}}", labels.join(","))
            };
            output.push_str(&format!("{}{} {}\n", gauge.name, labels_str, gauge.value));
        }

        // Export histograms
        for histogram in self.histograms.values() {
            output.push_str(&format!("# TYPE {} histogram\n", histogram.name));

            let base_labels = if histogram.labels.is_empty() {
                String::new()
            } else {
                let labels: Vec<String> =
                    histogram.labels.iter().map(|(k, v)| format!("{}=\"{}\"", k, v)).collect();
                format!("{{{}}}", labels.join(","))
            };

            // Export buckets
            for bucket in &histogram.buckets {
                let bucket_labels = if base_labels.is_empty() {
                    format!("{{le=\"{}\"}}", bucket.upper_bound)
                } else {
                    format!(
                        "{{le=\"{}\",{}}}",
                        bucket.upper_bound,
                        &base_labels[1..base_labels.len() - 1]
                    )
                };
                output.push_str(&format!(
                    "{}_bucket{} {}\n",
                    histogram.name, bucket_labels, bucket.count
                ));
            }

            // Export count and sum
            output.push_str(&format!(
                "{}_count{} {}\n",
                histogram.name, base_labels, histogram.count
            ));
            output.push_str(&format!(
                "{}_sum{} {}\n",
                histogram.name, base_labels, histogram.sum
            ));
        }

        output
    }

    /// Add a point to time series
    fn add_time_series_point(
        &mut self,
        name: &str,
        metric_type: MetricType,
        value: f64,
        labels: HashMap<String, String>,
    ) -> Result<()> {
        let timestamp = self.get_current_timestamp();

        let series = self.metrics.entry(name.to_string()).or_insert_with(|| MetricTimeSeries {
            name: name.to_string(),
            metric_type,
            values: Vec::new(),
            metadata: HashMap::new(),
        });

        // Add new value
        series.values.push(TimestampedValue {
            timestamp,
            value,
            labels,
        });

        // Trim old values if necessary
        if series.values.len() > self.config.max_time_series_points {
            series.values.remove(0);
        }

        // Remove old values based on retention policy
        let retention_cutoff =
            timestamp.saturating_sub(self.config.retention_duration_hours * 3600);
        series.values.retain(|v| v.timestamp >= retention_cutoff);

        Ok(())
    }

    /// Get current timestamp
    fn get_current_timestamp(&self) -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
    }

    /// Create unique metric key
    fn create_metric_key(&self, name: &str, labels: &HashMap<String, String>) -> String {
        if labels.is_empty() {
            name.to_string()
        } else {
            let mut key_parts = vec![name.to_string()];
            let mut sorted_labels: Vec<_> = labels.iter().collect();
            sorted_labels.sort_by_key(|(k, _)| *k);

            for (k, v) in sorted_labels {
                key_parts.push(format!("{}={}", k, v));
            }

            key_parts.join("|")
        }
    }

    /// Calculate gauge statistics from time series
    fn calculate_gauge_stats(&self, series: &MetricTimeSeries) -> (f64, f64, f64) {
        if series.values.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let values: Vec<f64> = series.values.iter().map(|v| v.value).collect();
        let min_value = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_value = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let average_value = values.iter().sum::<f64>() / values.len() as f64;

        (min_value, max_value, average_value)
    }

    /// Calculate histogram percentiles
    fn calculate_histogram_percentiles(&self, histogram: &Histogram) -> HashMap<String, f64> {
        let mut percentiles = HashMap::new();

        if histogram.count == 0 {
            return percentiles;
        }

        let percentile_targets = [50.0, 90.0, 95.0, 99.0];

        for &percentile in &percentile_targets {
            let target_count = (histogram.count as f64 * percentile / 100.0).ceil() as u64;

            // Find the bucket containing the target count
            let mut cumulative_count = 0;
            for bucket in &histogram.buckets {
                cumulative_count += bucket.count;
                if cumulative_count >= target_count {
                    percentiles.insert(format!("p{}", percentile as u32), bucket.upper_bound);
                    break;
                }
            }
        }

        percentiles
    }

    /// Calculate time series statistics
    fn calculate_time_series_stats(&self) -> TimeSeriesStats {
        let mut total_data_points = 0;
        let mut oldest_timestamp = u64::MAX;
        let mut newest_timestamp = 0;

        for series in self.metrics.values() {
            total_data_points += series.values.len();

            if let Some(first) = series.values.first() {
                oldest_timestamp = oldest_timestamp.min(first.timestamp);
            }

            if let Some(last) = series.values.last() {
                newest_timestamp = newest_timestamp.max(last.timestamp);
            }
        }

        let average_points_per_series = if !self.metrics.is_empty() {
            total_data_points as f64 / self.metrics.len() as f64
        } else {
            0.0
        };

        TimeSeriesStats {
            total_data_points,
            oldest_timestamp,
            newest_timestamp,
            average_points_per_series,
        }
    }
}

impl MetricsSummary {
    /// Print a summary of the metrics
    pub fn print_summary(&self) {
        println!("Metrics Summary");
        println!("===============");
        println!(
            "Collection Period: {:.2}s",
            self.collection_period.as_secs_f64()
        );
        println!("Total Metrics: {}", self.total_metrics);

        if !self.counter_metrics.is_empty() {
            println!("\nCounters:");
            for counter in &self.counter_metrics {
                println!(
                    "  {}: {} (rate: {:.2}/s)",
                    counter.name, counter.value, counter.rate_per_second
                );
            }
        }

        if !self.gauge_metrics.is_empty() {
            println!("\nGauges:");
            for gauge in &self.gauge_metrics {
                println!(
                    "  {}: {} (min: {:.2}, max: {:.2}, avg: {:.2})",
                    gauge.name,
                    gauge.current_value,
                    gauge.min_value,
                    gauge.max_value,
                    gauge.average_value
                );
            }
        }

        if !self.histogram_metrics.is_empty() {
            println!("\nHistograms:");
            for histogram in &self.histogram_metrics {
                println!(
                    "  {}: count={}, mean={:.4}",
                    histogram.name, histogram.count, histogram.mean
                );
                if let Some(p95) = histogram.percentiles.get("p95") {
                    println!("    P95: {:.4}", p95);
                }
            }
        }

        println!("\nTime Series Stats:");
        println!(
            "  Total Data Points: {}",
            self.time_series_stats.total_data_points
        );
        println!(
            "  Average Points per Series: {:.1}",
            self.time_series_stats.average_points_per_series
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        assert!(collector.config.enabled);
        assert_eq!(collector.config.collection_interval_ms, 1000);
    }

    #[test]
    fn test_counter_metrics() -> Result<()> {
        let mut collector = MetricsCollector::new();

        collector.increment_counter("requests_total", 1.0, None)?;
        collector.increment_counter("requests_total", 2.0, None)?;

        let metrics = collector.collect_metrics()?;
        assert_eq!(metrics.get("requests_total"), Some(&3.0));

        Ok(())
    }

    #[test]
    fn test_gauge_metrics() -> Result<()> {
        let mut collector = MetricsCollector::new();

        collector.set_gauge("cpu_usage", 0.75, None)?;
        collector.set_gauge("cpu_usage", 0.80, None)?;

        let metrics = collector.collect_metrics()?;
        assert_eq!(metrics.get("cpu_usage"), Some(&0.80));

        Ok(())
    }

    #[test]
    fn test_histogram_metrics() -> Result<()> {
        let mut collector = MetricsCollector::new();

        collector.record_histogram("request_duration", 0.1, None)?;
        collector.record_histogram("request_duration", 0.2, None)?;
        collector.record_histogram("request_duration", 0.15, None)?;

        let metrics = collector.collect_metrics()?;
        assert_eq!(metrics.get("request_duration_count"), Some(&3.0));
        assert!((metrics.get("request_duration_sum").unwrap() - 0.45).abs() < 1e-10);
        assert!((metrics.get("request_duration_mean").unwrap() - 0.15).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_labeled_metrics() -> Result<()> {
        let mut collector = MetricsCollector::new();

        let mut labels = HashMap::new();
        labels.insert("method".to_string(), "GET".to_string());
        labels.insert("status".to_string(), "200".to_string());

        collector.increment_counter("http_requests_total", 1.0, Some(labels.clone()))?;
        collector.increment_counter("http_requests_total", 1.0, Some(labels))?;

        // Should have separate counter for different label combinations
        let mut different_labels = HashMap::new();
        different_labels.insert("method".to_string(), "POST".to_string());
        different_labels.insert("status".to_string(), "404".to_string());

        collector.increment_counter("http_requests_total", 1.0, Some(different_labels))?;

        // We should have 2 different counter instances
        assert_eq!(collector.counters.len(), 2);

        Ok(())
    }

    #[test]
    fn test_timing_function() -> Result<()> {
        let mut collector = MetricsCollector::new();

        let result = collector.time_function("test_operation", None, || {
            std::thread::sleep(Duration::from_millis(10));
            Ok(42)
        })?;

        assert_eq!(result, 42);

        let metrics = collector.collect_metrics()?;
        assert!(metrics.contains_key("test_operation_count"));
        assert!(metrics.get("test_operation_count").unwrap() == &1.0);

        Ok(())
    }

    #[test]
    fn test_metrics_summary() -> Result<()> {
        let mut collector = MetricsCollector::new();

        collector.increment_counter("requests", 10.0, None)?;
        collector.set_gauge("cpu_usage", 0.75, None)?;
        collector.record_histogram("latency", 0.1, None)?;

        let summary = collector.generate_summary()?;

        assert_eq!(summary.total_metrics, 3);
        assert_eq!(summary.counter_metrics.len(), 1);
        assert_eq!(summary.gauge_metrics.len(), 1);
        assert_eq!(summary.histogram_metrics.len(), 1);

        Ok(())
    }

    #[test]
    fn test_prometheus_export() -> Result<()> {
        let mut collector = MetricsCollector::new();

        collector.increment_counter("test_counter", 5.0, None)?;
        collector.set_gauge("test_gauge", 0.8, None)?;

        let prometheus_output = collector.export_prometheus();

        assert!(prometheus_output.contains("# TYPE test_counter counter"));
        assert!(prometheus_output.contains("test_counter 5"));
        assert!(prometheus_output.contains("# TYPE test_gauge gauge"));
        assert!(prometheus_output.contains("test_gauge 0.8"));

        Ok(())
    }

    #[test]
    fn test_metric_key_generation() {
        let collector = MetricsCollector::new();

        // Test without labels
        let key1 = collector.create_metric_key("test_metric", &HashMap::new());
        assert_eq!(key1, "test_metric");

        // Test with labels
        let mut labels = HashMap::new();
        labels.insert("method".to_string(), "GET".to_string());
        labels.insert("status".to_string(), "200".to_string());

        let key2 = collector.create_metric_key("test_metric", &labels);
        // Should be deterministic ordering
        assert!(key2.contains("test_metric"));
        assert!(key2.contains("method=GET"));
        assert!(key2.contains("status=200"));
    }

    #[test]
    fn test_histogram_percentiles() {
        let collector = MetricsCollector::new();

        // Create a histogram with known values
        // Note: buckets should contain individual counts per bucket, not cumulative
        let histogram = Histogram {
            name: "test".to_string(),
            buckets: vec![
                HistogramBucket {
                    upper_bound: 0.1,
                    count: 10,
                }, // 10 values in [0, 0.1]
                HistogramBucket {
                    upper_bound: 0.5,
                    count: 20,
                }, // 20 values in (0.1, 0.5]
                HistogramBucket {
                    upper_bound: 1.0,
                    count: 30,
                }, // 30 values in (0.5, 1.0]
                HistogramBucket {
                    upper_bound: 5.0,
                    count: 40,
                }, // 40 values in (1.0, 5.0]
            ],
            count: 100,
            sum: 150.0,
            labels: HashMap::new(),
            last_updated: 0,
        };

        let percentiles = collector.calculate_histogram_percentiles(&histogram);

        // With cumulative counts: 10, 30, 60, 100
        // P50 (50th sample) lands in the 1.0 bucket (cumulative 60)
        assert_eq!(percentiles.get("p50"), Some(&1.0));
        // P90 (90th sample) lands in the 5.0 bucket (cumulative 100)
        assert_eq!(percentiles.get("p90"), Some(&5.0));
    }
}
