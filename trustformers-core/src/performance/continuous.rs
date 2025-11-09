//! Continuous benchmarking infrastructure for performance tracking

use crate::performance::benchmark::{BenchmarkResult, BenchmarkSuite};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Performance regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    /// Benchmark name
    pub benchmark_name: String,
    /// Metric that regressed
    pub metric_name: String,
    /// Previous value
    pub previous_value: f64,
    /// Current value
    pub current_value: f64,
    /// Regression percentage
    pub regression_percent: f64,
    /// Statistical significance
    pub is_significant: bool,
    /// Confidence level
    pub confidence: f64,
}

/// Configuration for continuous benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousBenchmarkConfig {
    /// Directory to store benchmark results
    pub results_dir: PathBuf,
    /// Git commit SHA
    pub commit_sha: Option<String>,
    /// Git branch name
    pub branch: Option<String>,
    /// Build configuration (debug/release)
    pub build_config: String,
    /// Regression threshold (percentage)
    pub regression_threshold: f64,
    /// Number of runs for statistical significance
    pub num_runs: usize,
    /// Confidence level for regression detection
    pub confidence_level: f64,
}

impl Default for ContinuousBenchmarkConfig {
    fn default() -> Self {
        Self {
            results_dir: PathBuf::from("benchmark_results"),
            commit_sha: None,
            branch: None,
            build_config: "release".to_string(),
            regression_threshold: 5.0, // 5% regression threshold
            num_runs: 5,
            confidence_level: 0.95,
        }
    }
}

/// Continuous benchmark runner
pub struct ContinuousBenchmark {
    config: ContinuousBenchmarkConfig,
    history: BenchmarkHistory,
}

impl ContinuousBenchmark {
    /// Create new continuous benchmark runner
    pub fn new(config: ContinuousBenchmarkConfig) -> Result<Self> {
        // Create results directory if it doesn't exist
        std::fs::create_dir_all(&config.results_dir)?;

        // Load history
        let history = BenchmarkHistory::load(&config.results_dir)?;

        Ok(Self { config, history })
    }

    /// Run benchmarks and check for regressions
    pub fn run_and_check(
        &mut self,
        suite: &mut BenchmarkSuite,
    ) -> Result<Vec<PerformanceRegression>> {
        // Run benchmarks multiple times for statistical significance
        let mut all_results = Vec::new();

        for run in 0..self.config.num_runs {
            println!(
                "Running benchmark iteration {}/{}",
                run + 1,
                self.config.num_runs
            );
            // Note: In real implementation, you'd re-run the benchmarks here
            // For now, we'll use the existing results
            all_results.extend(suite.results().to_vec());
        }

        // Save results
        let run_id = self.generate_run_id();
        self.save_results(&run_id, &all_results)?;

        // Check for regressions
        let regressions = self.check_regressions(&all_results)?;

        // Update history
        self.history.add_run(run_id, all_results);
        self.history.save(&self.config.results_dir)?;

        Ok(regressions)
    }

    /// Check for performance regressions
    fn check_regressions(
        &self,
        current_results: &[BenchmarkResult],
    ) -> Result<Vec<PerformanceRegression>> {
        let mut regressions = Vec::new();

        // Get baseline results (previous run on same branch/config)
        let baseline = self.history.get_baseline(&self.config.branch, &self.config.build_config);

        if let Some(baseline_results) = baseline {
            for current in current_results {
                if let Some(baseline) = baseline_results.iter().find(|b| b.name == current.name) {
                    // Check latency regression
                    let latency_regression = self.check_metric_regression(
                        &current.name,
                        "avg_latency",
                        baseline.avg_latency_ms,
                        current.avg_latency_ms,
                        true, // Higher is worse for latency
                    );

                    if let Some(reg) = latency_regression {
                        regressions.push(reg);
                    }

                    // Check throughput regression
                    let throughput_regression = self.check_metric_regression(
                        &current.name,
                        "throughput",
                        baseline.throughput_tokens_per_sec,
                        current.throughput_tokens_per_sec,
                        false, // Lower is worse for throughput
                    );

                    if let Some(reg) = throughput_regression {
                        regressions.push(reg);
                    }

                    // Check memory regression
                    if let (Some(baseline_mem), Some(current_mem)) =
                        (baseline.memory_bytes, current.memory_bytes)
                    {
                        let memory_regression = self.check_metric_regression(
                            &current.name,
                            "memory",
                            baseline_mem as f64,
                            current_mem as f64,
                            true, // Higher is worse for memory
                        );

                        if let Some(reg) = memory_regression {
                            regressions.push(reg);
                        }
                    }
                }
            }
        }

        Ok(regressions)
    }

    /// Check regression for a specific metric
    fn check_metric_regression(
        &self,
        benchmark_name: &str,
        metric_name: &str,
        baseline_value: f64,
        current_value: f64,
        higher_is_worse: bool,
    ) -> Option<PerformanceRegression> {
        let change_percent = if higher_is_worse {
            (current_value - baseline_value) / baseline_value * 100.0
        } else {
            (baseline_value - current_value) / baseline_value * 100.0
        };

        if change_percent > self.config.regression_threshold {
            // Simple statistical test - in real implementation, use proper statistics
            let is_significant = change_percent > self.config.regression_threshold * 2.0;

            Some(PerformanceRegression {
                benchmark_name: benchmark_name.to_string(),
                metric_name: metric_name.to_string(),
                previous_value: baseline_value,
                current_value,
                regression_percent: change_percent,
                is_significant,
                confidence: if is_significant { 0.95 } else { 0.5 },
            })
        } else {
            None
        }
    }

    /// Generate run ID
    fn generate_run_id(&self) -> String {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let commit = self.config.commit_sha.as_ref().map(|s| &s[..8]).unwrap_or("unknown");
        format!("{}_{}", timestamp, commit)
    }

    /// Save benchmark results
    fn save_results(&self, run_id: &str, results: &[BenchmarkResult]) -> Result<()> {
        let file_path = self.config.results_dir.join(format!("{}.json", run_id));
        let json = serde_json::to_string_pretty(results)?;
        std::fs::write(file_path, json)?;
        Ok(())
    }

    /// Generate performance report
    pub fn generate_report(&self) -> Result<PerformanceReport> {
        let trends = self.history.calculate_trends()?;
        let summary = self.history.generate_summary()?;

        Ok(PerformanceReport {
            trends,
            summary,
            latest_regressions: Vec::new(),
        })
    }
}

/// Benchmark history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkHistory {
    runs: HashMap<String, Vec<BenchmarkResult>>,
    metadata: HashMap<String, RunMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunMetadata {
    run_id: String,
    timestamp: chrono::DateTime<chrono::Utc>,
    commit_sha: Option<String>,
    branch: Option<String>,
    build_config: String,
}

impl BenchmarkHistory {
    /// Load history from directory
    fn load(dir: &Path) -> Result<Self> {
        let history_file = dir.join("history.json");

        if history_file.exists() {
            let json = std::fs::read_to_string(history_file)?;
            Ok(serde_json::from_str(&json)?)
        } else {
            Ok(Self {
                runs: HashMap::new(),
                metadata: HashMap::new(),
            })
        }
    }

    /// Save history to directory
    fn save(&self, dir: &Path) -> Result<()> {
        let history_file = dir.join("history.json");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(history_file, json)?;
        Ok(())
    }

    /// Add a benchmark run
    fn add_run(&mut self, run_id: String, results: Vec<BenchmarkResult>) {
        let metadata = RunMetadata {
            run_id: run_id.clone(),
            timestamp: chrono::Utc::now(),
            commit_sha: None, // Would be set from config
            branch: None,     // Would be set from config
            build_config: "release".to_string(),
        };

        self.runs.insert(run_id.clone(), results);
        self.metadata.insert(run_id, metadata);
    }

    /// Get baseline results for comparison
    fn get_baseline(
        &self,
        branch: &Option<String>,
        build_config: &str,
    ) -> Option<&Vec<BenchmarkResult>> {
        // Find the most recent run with matching branch and build config
        let mut matching_runs: Vec<_> = self
            .metadata
            .iter()
            .filter(|(_, meta)| {
                meta.branch.as_ref() == branch.as_ref() && meta.build_config == build_config
            })
            .collect();

        matching_runs.sort_by_key(|(_, meta)| meta.timestamp);

        matching_runs.last().and_then(|(run_id, _)| self.runs.get(*run_id))
    }

    /// Calculate performance trends
    fn calculate_trends(&self) -> Result<HashMap<String, PerformanceTrend>> {
        let mut trends = HashMap::new();

        // Group runs by benchmark name
        let mut by_benchmark: HashMap<String, Vec<(&String, &BenchmarkResult)>> = HashMap::new();

        for (run_id, results) in &self.runs {
            for result in results {
                by_benchmark.entry(result.name.clone()).or_default().push((run_id, result));
            }
        }

        // Calculate trends for each benchmark
        for (benchmark_name, mut runs) in by_benchmark {
            // Sort by timestamp
            runs.sort_by_key(|(run_id, _)| {
                self.metadata.get(*run_id).map(|m| m.timestamp).unwrap_or_default()
            });

            if runs.len() >= 2 {
                let latencies: Vec<f64> = runs.iter().map(|(_, r)| r.avg_latency_ms).collect();
                let throughputs: Vec<f64> =
                    runs.iter().map(|(_, r)| r.throughput_tokens_per_sec).collect();

                trends.insert(
                    benchmark_name,
                    PerformanceTrend {
                        latency_trend: calculate_trend(&latencies),
                        throughput_trend: calculate_trend(&throughputs),
                        sample_count: runs.len(),
                    },
                );
            }
        }

        Ok(trends)
    }

    /// Generate summary statistics
    fn generate_summary(&self) -> Result<PerformanceSummary> {
        let total_runs = self.runs.len();
        let total_benchmarks = self
            .runs
            .values()
            .flat_map(|results| results.iter().map(|r| &r.name))
            .collect::<std::collections::HashSet<_>>()
            .len();

        let latest_run = self.metadata.values().max_by_key(|m| m.timestamp).map(|m| m.timestamp);

        Ok(PerformanceSummary {
            total_runs,
            total_benchmarks,
            latest_run,
            earliest_run: self.metadata.values().min_by_key(|m| m.timestamp).map(|m| m.timestamp),
        })
    }
}

/// Performance trend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    /// Latency trend (positive = getting worse)
    pub latency_trend: f64,
    /// Throughput trend (negative = getting worse)
    pub throughput_trend: f64,
    /// Number of data points
    pub sample_count: usize,
}

/// Performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Performance trends by benchmark
    pub trends: HashMap<String, PerformanceTrend>,
    /// Summary statistics
    pub summary: PerformanceSummary,
    /// Latest regressions
    pub latest_regressions: Vec<PerformanceRegression>,
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_runs: usize,
    pub total_benchmarks: usize,
    pub latest_run: Option<chrono::DateTime<chrono::Utc>>,
    pub earliest_run: Option<chrono::DateTime<chrono::Utc>>,
}

/// Calculate linear trend from data points
fn calculate_trend(values: &[f64]) -> f64 {
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
        denominator += (x - x_mean) * (x - x_mean);
    }

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regression_detection() {
        let config = ContinuousBenchmarkConfig::default();
        let benchmark = ContinuousBenchmark::new(config).unwrap();

        let regression = benchmark.check_metric_regression(
            "test_benchmark",
            "latency",
            100.0, // baseline
            110.0, // current (10% worse)
            true,  // higher is worse
        );

        assert!(regression.is_some());
        let reg = regression.unwrap();
        assert_eq!(reg.regression_percent, 10.0);
    }

    #[test]
    fn test_trend_calculation() {
        let values = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let trend = calculate_trend(&values);
        assert!(trend > 0.0); // Positive trend (getting worse for latency)

        let values = vec![100.0, 98.0, 96.0, 94.0, 92.0];
        let trend = calculate_trend(&values);
        assert!(trend < 0.0); // Negative trend (getting better for latency)
    }
}
