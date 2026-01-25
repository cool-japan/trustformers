//! Statistical analysis and trend tracking for leaderboard data

use super::LeaderboardEntry;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Leaderboard statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardStats {
    /// Total number of entries
    pub total_entries: usize,
    /// Number of unique models
    pub unique_models: usize,
    /// Number of unique submitters
    pub unique_submitters: usize,
    /// Number of entries by category
    pub entries_by_category: HashMap<String, usize>,
    /// Average metrics across all entries
    pub average_metrics: AverageMetrics,
    /// Best metrics across all entries
    pub best_metrics: BestMetrics,
    /// Entries by precision
    pub entries_by_precision: HashMap<String, usize>,
    /// Entries by hardware platform
    pub entries_by_platform: HashMap<String, usize>,
    /// Date range of entries
    pub date_range: DateRange,
    /// Most active submitters
    pub top_submitters: Vec<(String, usize)>,
    /// Most popular benchmarks
    pub top_benchmarks: Vec<(String, usize)>,
}

/// Average metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AverageMetrics {
    pub latency_ms: f64,
    pub throughput: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub memory_mb: Option<f64>,
    pub gpu_utilization: Option<f64>,
    pub accuracy: Option<f64>,
}

/// Best metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestMetrics {
    pub lowest_latency: MetricRecord,
    pub highest_throughput: Option<MetricRecord>,
    pub highest_tokens_per_second: Option<MetricRecord>,
    pub lowest_memory: Option<MetricRecord>,
    pub highest_accuracy: Option<MetricRecord>,
    pub lowest_energy: Option<MetricRecord>,
}

/// Record of a metric with associated entry info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRecord {
    pub value: f64,
    pub model_name: String,
    pub model_version: String,
    pub entry_id: uuid::Uuid,
    pub timestamp: DateTime<Utc>,
}

/// Date range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

impl LeaderboardStats {
    /// Calculate statistics from entries
    pub fn from_entries(entries: &[LeaderboardEntry]) -> Result<Self> {
        if entries.is_empty() {
            return Ok(Self::empty());
        }

        let total_entries = entries.len();

        // Unique counts
        let unique_models: std::collections::HashSet<_> =
            entries.iter().map(|e| &e.model_name).collect();
        let unique_submitters: std::collections::HashSet<_> =
            entries.iter().map(|e| &e.submitter.name).collect();

        // Entries by category
        let mut entries_by_category = HashMap::new();
        for entry in entries {
            *entries_by_category.entry(entry.category.to_string()).or_insert(0) += 1;
        }

        // Entries by precision
        let mut entries_by_precision = HashMap::new();
        for entry in entries {
            let precision = format!("{:?}", entry.software.precision);
            *entries_by_precision.entry(precision).or_insert(0) += 1;
        }

        // Entries by platform
        let mut entries_by_platform = HashMap::new();
        for entry in entries {
            *entries_by_platform.entry(entry.hardware.platform.clone()).or_insert(0) += 1;
        }

        // Calculate average metrics
        let average_metrics = Self::calculate_averages(entries);

        // Find best metrics
        let best_metrics = Self::find_best_metrics(entries);

        // Date range
        let dates: Vec<_> = entries.iter().map(|e| e.timestamp).collect();
        let date_range = DateRange {
            start: *dates.iter().min().expect("Min failed"),
            end: *dates.iter().max().expect("Max failed"),
        };

        // Top submitters
        let mut submitter_counts: HashMap<String, usize> = HashMap::new();
        for entry in entries {
            *submitter_counts.entry(entry.submitter.name.clone()).or_insert(0) += 1;
        }
        let mut top_submitters: Vec<_> = submitter_counts.into_iter().collect();
        top_submitters.sort_by(|a, b| b.1.cmp(&a.1));
        top_submitters.truncate(10);

        // Top benchmarks
        let mut benchmark_counts: HashMap<String, usize> = HashMap::new();
        for entry in entries {
            *benchmark_counts.entry(entry.benchmark_name.clone()).or_insert(0) += 1;
        }
        let mut top_benchmarks: Vec<_> = benchmark_counts.into_iter().collect();
        top_benchmarks.sort_by(|a, b| b.1.cmp(&a.1));
        top_benchmarks.truncate(10);

        Ok(Self {
            total_entries,
            unique_models: unique_models.len(),
            unique_submitters: unique_submitters.len(),
            entries_by_category,
            average_metrics,
            best_metrics,
            entries_by_precision,
            entries_by_platform,
            date_range,
            top_submitters,
            top_benchmarks,
        })
    }

    /// Create empty statistics
    fn empty() -> Self {
        Self {
            total_entries: 0,
            unique_models: 0,
            unique_submitters: 0,
            entries_by_category: HashMap::new(),
            average_metrics: AverageMetrics {
                latency_ms: 0.0,
                throughput: None,
                tokens_per_second: None,
                memory_mb: None,
                gpu_utilization: None,
                accuracy: None,
            },
            best_metrics: BestMetrics {
                lowest_latency: MetricRecord {
                    value: f64::MAX,
                    model_name: String::new(),
                    model_version: String::new(),
                    entry_id: uuid::Uuid::nil(),
                    timestamp: Utc::now(),
                },
                highest_throughput: None,
                highest_tokens_per_second: None,
                lowest_memory: None,
                highest_accuracy: None,
                lowest_energy: None,
            },
            entries_by_precision: HashMap::new(),
            entries_by_platform: HashMap::new(),
            date_range: DateRange {
                start: Utc::now(),
                end: Utc::now(),
            },
            top_submitters: Vec::new(),
            top_benchmarks: Vec::new(),
        }
    }

    /// Calculate average metrics
    fn calculate_averages(entries: &[LeaderboardEntry]) -> AverageMetrics {
        let mut latency_sum = 0.0;
        let mut throughput_sum = 0.0;
        let mut throughput_count = 0;
        let mut tps_sum = 0.0;
        let mut tps_count = 0;
        let mut memory_sum = 0.0;
        let mut memory_count = 0;
        let mut gpu_sum = 0.0;
        let mut gpu_count = 0;
        let mut accuracy_sum = 0.0;
        let mut accuracy_count = 0;

        for entry in entries {
            latency_sum += entry.metrics.latency_ms;

            if let Some(throughput) = entry.metrics.throughput {
                throughput_sum += throughput;
                throughput_count += 1;
            }

            if let Some(tps) = entry.metrics.tokens_per_second {
                tps_sum += tps;
                tps_count += 1;
            }

            if let Some(memory) = entry.metrics.memory_mb {
                memory_sum += memory;
                memory_count += 1;
            }

            if let Some(gpu) = entry.metrics.gpu_utilization {
                gpu_sum += gpu;
                gpu_count += 1;
            }

            if let Some(accuracy) = entry.metrics.accuracy {
                accuracy_sum += accuracy;
                accuracy_count += 1;
            }
        }

        AverageMetrics {
            latency_ms: latency_sum / entries.len() as f64,
            throughput: if throughput_count > 0 {
                Some(throughput_sum / throughput_count as f64)
            } else {
                None
            },
            tokens_per_second: if tps_count > 0 { Some(tps_sum / tps_count as f64) } else { None },
            memory_mb: if memory_count > 0 { Some(memory_sum / memory_count as f64) } else { None },
            gpu_utilization: if gpu_count > 0 { Some(gpu_sum / gpu_count as f64) } else { None },
            accuracy: if accuracy_count > 0 {
                Some(accuracy_sum / accuracy_count as f64)
            } else {
                None
            },
        }
    }

    /// Find best metrics
    fn find_best_metrics(entries: &[LeaderboardEntry]) -> BestMetrics {
        let mut best = BestMetrics {
            lowest_latency: MetricRecord {
                value: f64::MAX,
                model_name: String::new(),
                model_version: String::new(),
                entry_id: uuid::Uuid::nil(),
                timestamp: Utc::now(),
            },
            highest_throughput: None,
            highest_tokens_per_second: None,
            lowest_memory: None,
            highest_accuracy: None,
            lowest_energy: None,
        };

        for entry in entries {
            // Lowest latency
            if entry.metrics.latency_ms < best.lowest_latency.value {
                best.lowest_latency = MetricRecord {
                    value: entry.metrics.latency_ms,
                    model_name: entry.model_name.clone(),
                    model_version: entry.model_version.clone(),
                    entry_id: entry.id,
                    timestamp: entry.timestamp,
                };
            }

            // Highest throughput
            if let Some(throughput) = entry.metrics.throughput {
                if best.highest_throughput.is_none()
                    || throughput > best.highest_throughput.as_ref().unwrap().value
                {
                    best.highest_throughput = Some(MetricRecord {
                        value: throughput,
                        model_name: entry.model_name.clone(),
                        model_version: entry.model_version.clone(),
                        entry_id: entry.id,
                        timestamp: entry.timestamp,
                    });
                }
            }

            // Highest tokens per second
            if let Some(tps) = entry.metrics.tokens_per_second {
                if best.highest_tokens_per_second.is_none()
                    || tps > best.highest_tokens_per_second.as_ref().unwrap().value
                {
                    best.highest_tokens_per_second = Some(MetricRecord {
                        value: tps,
                        model_name: entry.model_name.clone(),
                        model_version: entry.model_version.clone(),
                        entry_id: entry.id,
                        timestamp: entry.timestamp,
                    });
                }
            }

            // Lowest memory
            if let Some(memory) = entry.metrics.memory_mb {
                if best.lowest_memory.is_none()
                    || memory < best.lowest_memory.as_ref().unwrap().value
                {
                    best.lowest_memory = Some(MetricRecord {
                        value: memory,
                        model_name: entry.model_name.clone(),
                        model_version: entry.model_version.clone(),
                        entry_id: entry.id,
                        timestamp: entry.timestamp,
                    });
                }
            }

            // Highest accuracy
            if let Some(accuracy) = entry.metrics.accuracy {
                if best.highest_accuracy.is_none()
                    || accuracy > best.highest_accuracy.as_ref().unwrap().value
                {
                    best.highest_accuracy = Some(MetricRecord {
                        value: accuracy,
                        model_name: entry.model_name.clone(),
                        model_version: entry.model_version.clone(),
                        entry_id: entry.id,
                        timestamp: entry.timestamp,
                    });
                }
            }

            // Lowest energy
            if let Some(energy) = entry.metrics.energy_watts {
                if best.lowest_energy.is_none()
                    || energy < best.lowest_energy.as_ref().unwrap().value
                {
                    best.lowest_energy = Some(MetricRecord {
                        value: energy,
                        model_name: entry.model_name.clone(),
                        model_version: entry.model_version.clone(),
                        entry_id: entry.id,
                        timestamp: entry.timestamp,
                    });
                }
            }
        }

        best
    }
}

/// Trend analysis for performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Model name
    pub model_name: String,
    /// Metric name
    pub metric_name: String,
    /// Time period
    pub period_days: usize,
    /// Data points
    pub data_points: Vec<TrendDataPoint>,
    /// Overall trend
    pub trend: PerformanceTrend,
    /// Average improvement per day
    pub daily_change_percent: f64,
    /// Statistical significance
    pub r_squared: f64,
}

/// Trend data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendDataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub entry_id: uuid::Uuid,
}

/// Performance trend direction
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PerformanceTrend {
    /// Performance is improving
    Improving,
    /// Performance is degrading
    Degrading,
    /// Performance is stable
    Stable,
    /// Not enough data
    Insufficient,
}

impl TrendAnalysis {
    /// Analyze trend for a specific metric
    pub fn analyze(entries: &[LeaderboardEntry], metric_name: &str) -> Result<Self> {
        if entries.is_empty() {
            anyhow::bail!("No entries provided for trend analysis");
        }

        let model_name = entries[0].model_name.clone();
        let mut data_points = Vec::new();

        // Extract data points
        for entry in entries {
            let value = match metric_name {
                "latency" => Some(entry.metrics.latency_ms),
                "throughput" => entry.metrics.throughput,
                "tokens_per_second" => entry.metrics.tokens_per_second,
                "memory" => entry.metrics.memory_mb,
                "gpu_utilization" => entry.metrics.gpu_utilization,
                "accuracy" => entry.metrics.accuracy,
                "energy" => entry.metrics.energy_watts,
                _ => entry.metrics.custom_metrics.get(metric_name).copied(),
            };

            if let Some(val) = value {
                data_points.push(TrendDataPoint {
                    timestamp: entry.timestamp,
                    value: val,
                    entry_id: entry.id,
                });
            }
        }

        if data_points.len() < 2 {
            return Ok(Self {
                model_name,
                metric_name: metric_name.to_string(),
                period_days: 0,
                data_points,
                trend: PerformanceTrend::Insufficient,
                daily_change_percent: 0.0,
                r_squared: 0.0,
            });
        }

        // Sort by timestamp
        data_points.sort_by_key(|p| p.timestamp);

        // Calculate period
        let start = data_points.first().unwrap().timestamp;
        let end = data_points.last().unwrap().timestamp;
        let period_days = (end - start).num_days() as usize;

        // Calculate linear regression
        let (slope, r_squared) = Self::linear_regression(&data_points);

        // Determine trend
        let trend = if r_squared < 0.3 {
            PerformanceTrend::Stable
        } else if slope > 0.01 {
            // For metrics where higher is better
            if matches!(metric_name, "throughput" | "tokens_per_second" | "accuracy") {
                PerformanceTrend::Improving
            } else {
                PerformanceTrend::Degrading
            }
        } else if slope < -0.01 {
            // For metrics where lower is better
            if matches!(metric_name, "latency" | "memory" | "energy") {
                PerformanceTrend::Improving
            } else {
                PerformanceTrend::Degrading
            }
        } else {
            PerformanceTrend::Stable
        };

        // Calculate daily change percentage
        let first_value = data_points.first().unwrap().value;
        let last_value = data_points.last().unwrap().value;
        let total_change_percent = ((last_value - first_value) / first_value) * 100.0;
        let daily_change_percent =
            if period_days > 0 { total_change_percent / period_days as f64 } else { 0.0 };

        Ok(Self {
            model_name,
            metric_name: metric_name.to_string(),
            period_days,
            data_points,
            trend,
            daily_change_percent,
            r_squared,
        })
    }

    /// Calculate linear regression
    fn linear_regression(points: &[TrendDataPoint]) -> (f64, f64) {
        let n = points.len() as f64;
        if n < 2.0 {
            return (0.0, 0.0);
        }

        // Convert timestamps to days since first point
        let first_timestamp = points[0].timestamp;
        let x_values: Vec<f64> = points
            .iter()
            .map(|p| (p.timestamp - first_timestamp).num_seconds() as f64 / 86400.0)
            .collect();

        let y_values: Vec<f64> = points.iter().map(|p| p.value).collect();

        // Calculate means
        let x_mean = x_values.iter().sum::<f64>() / n;
        let y_mean = y_values.iter().sum::<f64>() / n;

        // Calculate slope and intercept
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..points.len() {
            numerator += (x_values[i] - x_mean) * (y_values[i] - y_mean);
            denominator += (x_values[i] - x_mean).powi(2);
        }

        if denominator == 0.0 {
            return (0.0, 0.0);
        }

        let slope = numerator / denominator;

        // Calculate R-squared
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;

        for i in 0..points.len() {
            let y_pred = slope * (x_values[i] - x_mean) + y_mean;
            ss_tot += (y_values[i] - y_mean).powi(2);
            ss_res += (y_values[i] - y_pred).powi(2);
        }

        let r_squared = if ss_tot == 0.0 { 0.0 } else { 1.0 - (ss_res / ss_tot) };

        (slope, r_squared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaderboard::*;
    use chrono::Duration;
    use std::collections::HashMap;

    fn create_test_entries() -> Vec<LeaderboardEntry> {
        vec![
            LeaderboardEntry {
                id: uuid::Uuid::new_v4(),
                timestamp: Utc::now() - Duration::days(7),
                model_name: "model1".to_string(),
                model_version: "1.0".to_string(),
                benchmark_name: "benchmark1".to_string(),
                category: LeaderboardCategory::Inference,
                hardware: HardwareInfo {
                    cpu: "Intel Xeon".to_string(),
                    cpu_cores: 32,
                    gpu: Some("NVIDIA A100".to_string()),
                    gpu_count: Some(1),
                    memory_gb: 256.0,
                    accelerator: Some(AcceleratorType::CUDA),
                    platform: "x86_64".to_string(),
                },
                software: SoftwareInfo {
                    framework_version: "0.1.0".to_string(),
                    rust_version: "1.75".to_string(),
                    os: "Linux".to_string(),
                    optimization_level: OptimizationLevel::O3,
                    precision: Precision::FP16,
                    quantization: None,
                    compiler_flags: vec![],
                },
                metrics: PerformanceMetrics {
                    latency_ms: 15.0,
                    latency_percentiles: LatencyPercentiles {
                        p50: 14.0,
                        p90: 18.0,
                        p95: 20.0,
                        p99: 25.0,
                        p999: 35.0,
                    },
                    throughput: Some(66.7),
                    tokens_per_second: None,
                    memory_mb: Some(1024.0),
                    peak_memory_mb: Some(1536.0),
                    gpu_utilization: Some(85.0),
                    accuracy: Some(0.90),
                    energy_watts: None,
                    custom_metrics: HashMap::new(),
                },
                metadata: HashMap::new(),
                validated: true,
                submitter: SubmitterInfo {
                    name: "User1".to_string(),
                    organization: None,
                    email: None,
                    github: None,
                },
                tags: vec![],
            },
            LeaderboardEntry {
                id: uuid::Uuid::new_v4(),
                timestamp: Utc::now(),
                model_name: "model1".to_string(),
                model_version: "1.1".to_string(),
                benchmark_name: "benchmark1".to_string(),
                category: LeaderboardCategory::Inference,
                hardware: HardwareInfo {
                    cpu: "Intel Xeon".to_string(),
                    cpu_cores: 32,
                    gpu: Some("NVIDIA A100".to_string()),
                    gpu_count: Some(1),
                    memory_gb: 256.0,
                    accelerator: Some(AcceleratorType::CUDA),
                    platform: "x86_64".to_string(),
                },
                software: SoftwareInfo {
                    framework_version: "0.1.0".to_string(),
                    rust_version: "1.75".to_string(),
                    os: "Linux".to_string(),
                    optimization_level: OptimizationLevel::O3,
                    precision: Precision::FP16,
                    quantization: None,
                    compiler_flags: vec![],
                },
                metrics: PerformanceMetrics {
                    latency_ms: 10.0,
                    latency_percentiles: LatencyPercentiles {
                        p50: 9.0,
                        p90: 12.0,
                        p95: 14.0,
                        p99: 18.0,
                        p999: 25.0,
                    },
                    throughput: Some(100.0),
                    tokens_per_second: None,
                    memory_mb: Some(1024.0),
                    peak_memory_mb: Some(1536.0),
                    gpu_utilization: Some(90.0),
                    accuracy: Some(0.95),
                    energy_watts: None,
                    custom_metrics: HashMap::new(),
                },
                metadata: HashMap::new(),
                validated: true,
                submitter: SubmitterInfo {
                    name: "User1".to_string(),
                    organization: None,
                    email: None,
                    github: None,
                },
                tags: vec![],
            },
        ]
    }

    #[test]
    fn test_leaderboard_stats() {
        let entries = create_test_entries();
        let stats = LeaderboardStats::from_entries(&entries).unwrap();

        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.unique_models, 1);
        assert_eq!(stats.unique_submitters, 1);
        assert_eq!(stats.entries_by_category["Inference"], 2);
        assert_eq!(stats.average_metrics.latency_ms, 12.5);
        assert_eq!(stats.best_metrics.lowest_latency.value, 10.0);
    }

    #[test]
    fn test_trend_analysis() {
        let entries = create_test_entries();
        let trend = TrendAnalysis::analyze(&entries, "latency").unwrap();

        assert_eq!(trend.model_name, "model1");
        assert_eq!(trend.metric_name, "latency");
        assert_eq!(trend.data_points.len(), 2);

        // Latency improved from 15ms to 10ms
        match trend.trend {
            PerformanceTrend::Improving => {}, // Expected
            _ => panic!("Expected improving trend for latency"),
        }
    }
}
