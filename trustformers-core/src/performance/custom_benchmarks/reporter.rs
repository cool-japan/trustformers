//! Benchmark result reporting and visualization

use super::BenchmarkIteration;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Benchmark report containing results and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Benchmark name
    pub name: String,
    /// Description
    pub description: String,
    /// Tags
    pub tags: Vec<String>,
    /// Number of iterations
    pub iterations: usize,
    /// Total duration
    pub total_duration: Duration,
    /// Raw iteration data
    pub raw_data: Option<Vec<BenchmarkIteration>>,
    /// Duration statistics
    pub duration_stats: Option<DurationStats>,
    /// Aggregated metrics
    pub aggregate_metrics: HashMap<String, MetricStats>,
    /// Performance summary
    pub summary: PerformanceSummary,
}

/// Duration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub percentiles: HashMap<String, f64>,
}

/// Statistics for a metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub avg_throughput: Option<f64>,
    pub avg_latency_ms: f64,
    pub total_samples: usize,
    pub success_rate: f64,
}

impl BenchmarkReport {
    /// Create report from iterations
    pub fn from_iterations(
        name: String,
        description: String,
        tags: Vec<String>,
        iterations: Vec<BenchmarkIteration>,
        total_duration: Duration,
    ) -> Self {
        let duration_stats = Self::calculate_duration_stats(&iterations);
        let aggregate_metrics = Self::aggregate_metrics(&iterations);
        let summary = Self::create_summary(&iterations, &aggregate_metrics);

        Self {
            name,
            description,
            tags,
            iterations: iterations.len(),
            total_duration,
            raw_data: None, // Don't store by default to save memory
            duration_stats: Some(duration_stats),
            aggregate_metrics,
            summary,
        }
    }

    /// Calculate duration statistics
    fn calculate_duration_stats(iterations: &[BenchmarkIteration]) -> DurationStats {
        let durations: Vec<f64> = iterations.iter().map(|i| i.duration.as_secs_f64()).collect();

        let mean = durations.iter().sum::<f64>() / durations.len() as f64;
        let variance =
            durations.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / durations.len() as f64;
        let std_dev = variance.sqrt();

        let mut sorted = durations.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("Partial comparison failed"));

        let percentiles = vec![
            ("p50".to_string(), Self::percentile(&sorted, 0.50)),
            ("p90".to_string(), Self::percentile(&sorted, 0.90)),
            ("p95".to_string(), Self::percentile(&sorted, 0.95)),
            ("p99".to_string(), Self::percentile(&sorted, 0.99)),
            ("p999".to_string(), Self::percentile(&sorted, 0.999)),
        ]
        .into_iter()
        .collect();

        DurationStats {
            mean,
            std_dev,
            min: *sorted.first().unwrap_or(&0.0),
            max: *sorted.last().unwrap_or(&0.0),
            median: Self::percentile(&sorted, 0.50),
            percentiles,
        }
    }

    /// Calculate percentile
    fn percentile(sorted: &[f64], p: f64) -> f64 {
        let idx = (p * (sorted.len() - 1) as f64) as usize;
        sorted[idx]
    }

    /// Aggregate metrics across iterations
    fn aggregate_metrics(iterations: &[BenchmarkIteration]) -> HashMap<String, MetricStats> {
        let mut metrics_map: HashMap<String, Vec<f64>> = HashMap::new();

        // Collect all metrics
        for iteration in iterations {
            // Standard metrics
            if let Some(throughput) = iteration.metrics.throughput {
                metrics_map.entry("throughput".to_string()).or_default().push(throughput);
            }

            if let Some(memory) = iteration.metrics.memory_bytes {
                metrics_map
                    .entry("memory_mb".to_string())
                    .or_default()
                    .push(memory as f64 / 1_048_576.0);
            }

            if let Some(gpu) = iteration.metrics.gpu_utilization {
                metrics_map.entry("gpu_utilization".to_string()).or_default().push(gpu);
            }

            // Model metrics
            if let Some(model_metrics) = &iteration.metrics.model_metrics {
                if let Some(tps) = model_metrics.tokens_per_second {
                    metrics_map.entry("tokens_per_second".to_string()).or_default().push(tps);
                }
            }

            // Custom metrics
            for (key, value) in &iteration.metrics.custom {
                metrics_map.entry(key.clone()).or_default().push(*value);
            }
        }

        // Calculate stats for each metric
        metrics_map
            .into_iter()
            .map(|(name, values)| {
                let stats = Self::calculate_metric_stats(&values);
                (name, stats)
            })
            .collect()
    }

    /// Calculate statistics for a metric
    fn calculate_metric_stats(values: &[f64]) -> MetricStats {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("Partial comparison failed"));

        MetricStats {
            mean,
            std_dev,
            min: *sorted.first().unwrap_or(&0.0),
            max: *sorted.last().unwrap_or(&0.0),
            median: Self::percentile(&sorted, 0.50),
        }
    }

    /// Create performance summary
    fn create_summary(
        iterations: &[BenchmarkIteration],
        metrics: &HashMap<String, MetricStats>,
    ) -> PerformanceSummary {
        let avg_latency_ms =
            iterations.iter().map(|i| i.duration.as_secs_f64() * 1000.0).sum::<f64>()
                / iterations.len() as f64;

        let success_rate = iterations.iter().filter(|i| i.validation_passed.unwrap_or(true)).count()
            as f64
            / iterations.len() as f64;

        PerformanceSummary {
            avg_throughput: metrics.get("throughput").map(|s| s.mean),
            avg_latency_ms,
            total_samples: iterations.len(),
            success_rate,
        }
    }
}

/// Report format options
#[derive(Debug, Clone, Copy)]
pub enum ReportFormat {
    /// Human-readable text
    Text,
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Markdown table
    Markdown,
    /// HTML report
    Html,
}

/// Report generator
pub struct Reporter;

impl Reporter {
    /// Generate report in specified format
    pub fn generate(report: &BenchmarkReport, format: ReportFormat) -> Result<String> {
        match format {
            ReportFormat::Text => Self::generate_text(report),
            ReportFormat::Json => Self::generate_json(report),
            ReportFormat::Csv => Self::generate_csv(report),
            ReportFormat::Markdown => Self::generate_markdown(report),
            ReportFormat::Html => Self::generate_html(report),
        }
    }

    /// Generate text report
    fn generate_text(report: &BenchmarkReport) -> Result<String> {
        let mut output = String::new();

        output.push_str(&format!("Benchmark: {}\n", report.name));
        output.push_str(&format!("Description: {}\n", report.description));
        output.push_str(&format!("Tags: {}\n", report.tags.join(", ")));
        output.push_str(&format!("Iterations: {}\n", report.iterations));
        output.push_str(&format!(
            "Total Duration: {:.2}s\n\n",
            report.total_duration.as_secs_f64()
        ));

        output.push_str("Performance Summary:\n");
        output.push_str(&format!(
            "  Average Latency: {:.2}ms\n",
            report.summary.avg_latency_ms
        ));
        if let Some(throughput) = report.summary.avg_throughput {
            output.push_str(&format!(
                "  Average Throughput: {:.2} items/sec\n",
                throughput
            ));
        }
        output.push_str(&format!(
            "  Success Rate: {:.1}%\n\n",
            report.summary.success_rate * 100.0
        ));

        if let Some(stats) = &report.duration_stats {
            output.push_str("Duration Statistics:\n");
            output.push_str(&format!("  Mean: {:.2}ms\n", stats.mean * 1000.0));
            output.push_str(&format!("  Std Dev: {:.2}ms\n", stats.std_dev * 1000.0));
            output.push_str(&format!("  Min: {:.2}ms\n", stats.min * 1000.0));
            output.push_str(&format!("  Max: {:.2}ms\n", stats.max * 1000.0));
            output.push_str(&format!("  Median: {:.2}ms\n", stats.median * 1000.0));
            output.push_str("  Percentiles:\n");
            for (name, value) in &stats.percentiles {
                output.push_str(&format!("    {}: {:.2}ms\n", name, value * 1000.0));
            }
            output.push('\n');
        }

        if !report.aggregate_metrics.is_empty() {
            output.push_str("Metrics:\n");
            for (name, stats) in &report.aggregate_metrics {
                output.push_str(&format!("  {}:\n", name));
                output.push_str(&format!("    Mean: {:.2}\n", stats.mean));
                output.push_str(&format!("    Std Dev: {:.2}\n", stats.std_dev));
                output.push_str(&format!("    Min: {:.2}\n", stats.min));
                output.push_str(&format!("    Max: {:.2}\n", stats.max));
            }
        }

        Ok(output)
    }

    /// Generate JSON report
    fn generate_json(report: &BenchmarkReport) -> Result<String> {
        Ok(serde_json::to_string_pretty(report)?)
    }

    /// Generate CSV report
    fn generate_csv(report: &BenchmarkReport) -> Result<String> {
        let mut output = String::new();

        // Header
        output.push_str("Metric,Mean,StdDev,Min,Max,Median\n");

        // Duration
        if let Some(stats) = &report.duration_stats {
            output.push_str(&format!(
                "Duration (ms),{:.2},{:.2},{:.2},{:.2},{:.2}\n",
                stats.mean * 1000.0,
                stats.std_dev * 1000.0,
                stats.min * 1000.0,
                stats.max * 1000.0,
                stats.median * 1000.0
            ));
        }

        // Other metrics
        for (name, stats) in &report.aggregate_metrics {
            output.push_str(&format!(
                "{},{:.2},{:.2},{:.2},{:.2},{:.2}\n",
                name, stats.mean, stats.std_dev, stats.min, stats.max, stats.median
            ));
        }

        Ok(output)
    }

    /// Generate Markdown report
    fn generate_markdown(report: &BenchmarkReport) -> Result<String> {
        let mut output = String::new();

        output.push_str(&format!("# Benchmark: {}\n\n", report.name));
        output.push_str(&format!("{}\n\n", report.description));
        output.push_str(&format!("**Tags:** {}\n\n", report.tags.join(", ")));

        output.push_str("## Summary\n\n");
        output.push_str("| Metric | Value |\n");
        output.push_str("|--------|-------|\n");
        output.push_str(&format!("| Iterations | {} |\n", report.iterations));
        output.push_str(&format!(
            "| Total Duration | {:.2}s |\n",
            report.total_duration.as_secs_f64()
        ));
        output.push_str(&format!(
            "| Average Latency | {:.2}ms |\n",
            report.summary.avg_latency_ms
        ));
        if let Some(throughput) = report.summary.avg_throughput {
            output.push_str(&format!(
                "| Average Throughput | {:.2} items/sec |\n",
                throughput
            ));
        }
        output.push_str(&format!(
            "| Success Rate | {:.1}% |\n\n",
            report.summary.success_rate * 100.0
        ));

        if let Some(stats) = &report.duration_stats {
            output.push_str("## Duration Statistics\n\n");
            output.push_str("| Statistic | Value (ms) |\n");
            output.push_str("|-----------|------------|\n");
            output.push_str(&format!("| Mean | {:.2} |\n", stats.mean * 1000.0));
            output.push_str(&format!("| Std Dev | {:.2} |\n", stats.std_dev * 1000.0));
            output.push_str(&format!("| Min | {:.2} |\n", stats.min * 1000.0));
            output.push_str(&format!("| Max | {:.2} |\n", stats.max * 1000.0));
            output.push_str(&format!("| Median | {:.2} |\n", stats.median * 1000.0));
            for (name, value) in &stats.percentiles {
                output.push_str(&format!("| {} | {:.2} |\n", name, value * 1000.0));
            }
            output.push('\n');
        }

        if !report.aggregate_metrics.is_empty() {
            output.push_str("## Metrics\n\n");
            output.push_str("| Metric | Mean | Std Dev | Min | Max |\n");
            output.push_str("|--------|------|---------|-----|-----|\n");
            for (name, stats) in &report.aggregate_metrics {
                output.push_str(&format!(
                    "| {} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
                    name, stats.mean, stats.std_dev, stats.min, stats.max
                ));
            }
        }

        Ok(output)
    }

    /// Generate HTML report
    fn generate_html(report: &BenchmarkReport) -> Result<String> {
        let mut output = String::new();

        output.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        output.push_str("<title>Benchmark Report</title>\n");
        output.push_str("<style>\n");
        output.push_str("body { font-family: Arial, sans-serif; margin: 20px; }\n");
        output.push_str("table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n");
        output.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        output.push_str("th { background-color: #f2f2f2; }\n");
        output.push_str("</style>\n</head>\n<body>\n");

        output.push_str(&format!("<h1>{}</h1>\n", report.name));
        output.push_str(&format!("<p>{}</p>\n", report.description));
        output.push_str(&format!(
            "<p><strong>Tags:</strong> {}</p>\n",
            report.tags.join(", ")
        ));

        output.push_str("<h2>Summary</h2>\n");
        output.push_str("<table>\n");
        output.push_str(&format!(
            "<tr><td>Iterations</td><td>{}</td></tr>\n",
            report.iterations
        ));
        output.push_str(&format!(
            "<tr><td>Total Duration</td><td>{:.2}s</td></tr>\n",
            report.total_duration.as_secs_f64()
        ));
        output.push_str(&format!(
            "<tr><td>Average Latency</td><td>{:.2}ms</td></tr>\n",
            report.summary.avg_latency_ms
        ));
        if let Some(throughput) = report.summary.avg_throughput {
            output.push_str(&format!(
                "<tr><td>Average Throughput</td><td>{:.2} items/sec</td></tr>\n",
                throughput
            ));
        }
        output.push_str(&format!(
            "<tr><td>Success Rate</td><td>{:.1}%</td></tr>\n",
            report.summary.success_rate * 100.0
        ));
        output.push_str("</table>\n");

        output.push_str("</body>\n</html>");

        Ok(output)
    }
}

/// Compare multiple benchmark reports
pub struct ReportComparator;

impl ReportComparator {
    /// Compare two reports
    pub fn compare(baseline: &BenchmarkReport, candidate: &BenchmarkReport) -> ComparisonResult {
        let latency_change = (candidate.summary.avg_latency_ms - baseline.summary.avg_latency_ms)
            / baseline.summary.avg_latency_ms
            * 100.0;

        let throughput_change = if let (Some(base), Some(cand)) = (
            baseline.summary.avg_throughput,
            candidate.summary.avg_throughput,
        ) {
            Some((cand - base) / base * 100.0)
        } else {
            None
        };

        ComparisonResult {
            baseline_name: baseline.name.clone(),
            candidate_name: candidate.name.clone(),
            latency_change_percent: latency_change,
            throughput_change_percent: throughput_change,
            metrics_comparison: Self::compare_metrics(
                &baseline.aggregate_metrics,
                &candidate.aggregate_metrics,
            ),
        }
    }

    /// Compare metrics
    fn compare_metrics(
        baseline: &HashMap<String, MetricStats>,
        candidate: &HashMap<String, MetricStats>,
    ) -> HashMap<String, f64> {
        let mut comparison = HashMap::new();

        for (name, base_stats) in baseline {
            if let Some(cand_stats) = candidate.get(name) {
                let change = (cand_stats.mean - base_stats.mean) / base_stats.mean * 100.0;
                comparison.insert(name.clone(), change);
            }
        }

        comparison
    }
}

/// Comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub baseline_name: String,
    pub candidate_name: String,
    pub latency_change_percent: f64,
    pub throughput_change_percent: Option<f64>,
    pub metrics_comparison: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance::custom_benchmarks::BenchmarkMetrics;

    fn create_test_report() -> BenchmarkReport {
        let iterations = vec![
            BenchmarkIteration {
                duration: Duration::from_millis(10),
                metrics: BenchmarkMetrics {
                    throughput: Some(100.0),
                    ..Default::default()
                },
                validation_passed: Some(true),
                metadata: None,
            },
            BenchmarkIteration {
                duration: Duration::from_millis(12),
                metrics: BenchmarkMetrics {
                    throughput: Some(83.3),
                    ..Default::default()
                },
                validation_passed: Some(true),
                metadata: None,
            },
        ];

        BenchmarkReport::from_iterations(
            "test_benchmark".to_string(),
            "Test benchmark".to_string(),
            vec!["test".to_string()],
            iterations,
            Duration::from_millis(22),
        )
    }

    #[test]
    fn test_report_generation() {
        let report = create_test_report();

        assert_eq!(report.iterations, 2);
        assert_eq!(report.summary.avg_latency_ms, 11.0);
        assert!(report.summary.avg_throughput.is_some());
    }

    #[test]
    fn test_text_format() {
        let report = create_test_report();
        let text = Reporter::generate(&report, ReportFormat::Text).unwrap();

        assert!(text.contains("Benchmark: test_benchmark"));
        assert!(text.contains("Average Latency: 11.00ms"));
    }

    #[test]
    fn test_json_format() {
        let report = create_test_report();
        let json = Reporter::generate(&report, ReportFormat::Json).unwrap();

        let parsed: BenchmarkReport =
            serde_json::from_str(&json).expect("JSON deserialization failed");
        assert_eq!(parsed.name, "test_benchmark");
    }

    #[test]
    fn test_comparison() {
        let baseline = create_test_report();
        let mut candidate = create_test_report();
        candidate.summary.avg_latency_ms = 9.0;

        let comparison = ReportComparator::compare(&baseline, &candidate);

        // Latency improved by ~18%
        assert!(comparison.latency_change_percent < 0.0);
        assert!((comparison.latency_change_percent - (-18.18)).abs() < 0.1);
    }
}
