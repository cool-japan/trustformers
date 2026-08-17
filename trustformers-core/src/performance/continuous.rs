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

    /// Deprecated entry point that cannot actually re-run benchmarks.
    ///
    /// A `BenchmarkSuite` holds results, not the model that produced them, so
    /// this signature has no way to execute the benchmarks `num_runs` times.
    /// It previously duplicated the existing result set `num_runs` times, which
    /// made every downstream variance and significance figure identically zero.
    ///
    /// Use [`Self::run_and_check_with`] and pass a closure that re-runs the
    /// benchmarks.
    pub fn run_and_check(
        &mut self,
        _suite: &mut BenchmarkSuite,
    ) -> Result<Vec<PerformanceRegression>> {
        Err(anyhow::anyhow!(
            "run_and_check cannot re-run benchmarks: a BenchmarkSuite carries results, not the \
             model that produced them. Use run_and_check_with(suite, |suite| \
             suite.benchmark_inference(&model, \"name\")) so each of the {} runs is really \
             executed.",
            self.config.num_runs
        ))
    }

    /// Run the benchmarks `num_runs` times and check for regressions.
    ///
    /// `run_once` is invoked once per run against a freshly cleared suite, so
    /// every run produces independent measurements. The per-benchmark samples
    /// collected this way are what feed the regression test, giving it real
    /// variance to work with.
    pub fn run_and_check_with<F>(
        &mut self,
        suite: &mut BenchmarkSuite,
        mut run_once: F,
    ) -> Result<Vec<PerformanceRegression>>
    where
        F: FnMut(&mut BenchmarkSuite) -> Result<()>,
    {
        if self.config.num_runs == 0 {
            return Err(anyhow::anyhow!(
                "ContinuousBenchmarkConfig::num_runs must be at least 1"
            ));
        }

        let mut all_results: Vec<BenchmarkResult> = Vec::new();

        for run in 0..self.config.num_runs {
            tracing::info!(
                run = run + 1,
                total = self.config.num_runs,
                "running benchmark iteration"
            );

            // Each iteration starts from an empty suite so the results really
            // come from this run.
            suite.clear_results();
            run_once(suite)?;

            let results = suite.results();
            if results.is_empty() {
                return Err(anyhow::anyhow!(
                    "benchmark run {} produced no results; refusing to report a regression \
                     verdict with no measurements",
                    run + 1
                ));
            }
            all_results.extend(results.to_vec());
        }

        // Save results
        let run_id = self.generate_run_id();
        self.save_results(&run_id, &all_results)?;

        // Check for regressions
        let regressions = self.check_regressions(&all_results)?;

        // Update history
        self.history.add_run(
            run_id,
            all_results,
            self.config.commit_sha.clone(),
            self.config.branch.clone(),
            self.config.build_config.clone(),
        );
        self.history.save(&self.config.results_dir)?;

        Ok(regressions)
    }

    /// Check for performance regressions.
    ///
    /// Results are grouped by benchmark name so repeated runs of the same
    /// benchmark form a sample; the comparison against the baseline run is a
    /// Welch t-test over those samples, not a single-point ratio.
    fn check_regressions(
        &self,
        current_results: &[BenchmarkResult],
    ) -> Result<Vec<PerformanceRegression>> {
        let mut regressions = Vec::new();

        // Get baseline results (previous run on same branch/config)
        let Some(baseline_results) =
            self.history.get_baseline(&self.config.branch, &self.config.build_config)
        else {
            return Ok(regressions);
        };

        let current_by_name = group_by_name(current_results);
        let baseline_by_name = group_by_name(baseline_results);

        let mut names: Vec<&String> = current_by_name.keys().collect();
        names.sort();

        for name in names {
            let Some(current_group) = current_by_name.get(name) else {
                continue;
            };
            let Some(baseline_group) = baseline_by_name.get(name) else {
                continue;
            };

            let metrics: [(&str, MetricExtractor, bool); 3] = [
                ("avg_latency", |r| Some(r.avg_latency_ms), true),
                ("throughput", |r| Some(r.throughput_tokens_per_sec), false),
                ("memory", |r| r.memory_bytes.map(|bytes| bytes as f64), true),
            ];

            for (metric_name, extract, higher_is_worse) in metrics {
                let current_samples: Vec<f64> =
                    current_group.iter().filter_map(|result| extract(result)).collect();
                let baseline_samples: Vec<f64> =
                    baseline_group.iter().filter_map(|result| extract(result)).collect();

                if current_samples.is_empty() || baseline_samples.is_empty() {
                    continue;
                }

                if let Some(regression) = self.check_metric_regression(
                    name,
                    metric_name,
                    &baseline_samples,
                    &current_samples,
                    higher_is_worse,
                ) {
                    regressions.push(regression);
                }
            }
        }

        Ok(regressions)
    }

    /// Check regression for a specific metric using the observed samples.
    ///
    /// `confidence` is `1 - p` from a Welch t-test over the two samples, so it
    /// reflects the measured distributions. With a single observation per side
    /// no test is possible and `confidence` is reported as `0.0` with
    /// `is_significant = false`, instead of a hardcoded 0.95/0.5.
    fn check_metric_regression(
        &self,
        benchmark_name: &str,
        metric_name: &str,
        baseline_samples: &[f64],
        current_samples: &[f64],
        higher_is_worse: bool,
    ) -> Option<PerformanceRegression> {
        let baseline_value = crate::statistics::mean(baseline_samples)?;
        let current_value = crate::statistics::mean(current_samples)?;

        if baseline_value == 0.0 {
            return None;
        }

        let change_percent = if higher_is_worse {
            (current_value - baseline_value) / baseline_value * 100.0
        } else {
            (baseline_value - current_value) / baseline_value * 100.0
        };

        if change_percent <= self.config.regression_threshold {
            return None;
        }

        let test = crate::statistics::welch_t_test(current_samples, baseline_samples);
        let (confidence, p_value) = match test {
            Some(result) => (result.confidence(), result.p_value),
            // Not enough data (or zero variance on both sides) to run a test.
            None => (0.0, 1.0),
        };
        let alpha = (1.0 - self.config.confidence_level).max(f64::EPSILON);
        let is_significant = p_value < alpha;

        Some(PerformanceRegression {
            benchmark_name: benchmark_name.to_string(),
            metric_name: metric_name.to_string(),
            previous_value: baseline_value,
            current_value,
            regression_percent: change_percent,
            is_significant,
            confidence,
        })
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

    /// Add a benchmark run, recording the real provenance of the run.
    fn add_run(
        &mut self,
        run_id: String,
        results: Vec<BenchmarkResult>,
        commit_sha: Option<String>,
        branch: Option<String>,
        build_config: String,
    ) {
        let metadata = RunMetadata {
            run_id: run_id.clone(),
            timestamp: chrono::Utc::now(),
            commit_sha,
            branch,
            build_config,
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

/// Pulls one comparable metric out of a benchmark result.
type MetricExtractor = fn(&BenchmarkResult) -> Option<f64>;

/// Group benchmark results by benchmark name.
fn group_by_name(results: &[BenchmarkResult]) -> HashMap<String, Vec<&BenchmarkResult>> {
    let mut grouped: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
    for result in results {
        grouped.entry(result.name.clone()).or_default().push(result);
    }
    grouped
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
    use crate::performance::benchmark::BenchmarkConfig;
    use std::time::Duration;

    fn test_config(name: &str) -> ContinuousBenchmarkConfig {
        ContinuousBenchmarkConfig {
            results_dir: std::env::temp_dir().join(format!(
                "trustformers_continuous_{}_{}",
                name,
                std::process::id()
            )),
            ..Default::default()
        }
    }

    fn make_result(name: &str, latency_ms: f64) -> BenchmarkResult {
        BenchmarkResult::from_timings(
            name.to_string(),
            "test".to_string(),
            vec![Duration::from_secs_f64(latency_ms / 1000.0)],
            1,
            1,
            None,
            None,
        )
    }

    #[test]
    fn test_regression_detection() {
        let config = test_config("regression");
        let benchmark = ContinuousBenchmark::new(config).expect("operation failed in test");

        let regression = benchmark.check_metric_regression(
            "test_benchmark",
            "latency",
            &[100.0, 100.0, 100.0], // baseline
            &[110.0, 110.0, 110.0], // current (10% worse)
            true,                   // higher is worse
        );

        assert!(regression.is_some());
        let reg = regression.expect("operation failed in test");
        assert!((reg.regression_percent - 10.0).abs() < 1e-9);
    }

    /// Regression test: `check_metric_regression` used to stamp
    /// `confidence: 0.95` whenever the change exceeded twice the threshold and
    /// `0.5` otherwise, regardless of the data.
    #[test]
    fn test_confidence_comes_from_the_observed_distribution() {
        let benchmark =
            ContinuousBenchmark::new(test_config("confidence")).expect("operation failed in test");

        // Tight, clearly separated samples: high confidence.
        let tight = benchmark
            .check_metric_regression(
                "bench",
                "avg_latency",
                &[100.0, 100.1, 99.9, 100.05],
                &[120.0, 120.1, 119.9, 120.05],
                true,
            )
            .expect("a 20% regression must be reported");
        assert!(tight.is_significant);
        assert!(
            tight.confidence > 0.999,
            "confidence {} should be near 1 for cleanly separated samples",
            tight.confidence
        );

        // Same mean shift but huge overlap: the same 20% change must not carry
        // the same confidence.
        let noisy = benchmark
            .check_metric_regression(
                "bench",
                "avg_latency",
                &[40.0, 160.0, 60.0, 140.0],
                &[60.0, 180.0, 80.0, 160.0],
                true,
            )
            .expect("a 20% mean regression must still be reported");
        assert!(
            noisy.confidence < tight.confidence,
            "noisy samples ({}) must not be as confident as tight ones ({})",
            noisy.confidence,
            tight.confidence
        );
        assert_ne!(noisy.confidence, 0.95);
        assert_ne!(noisy.confidence, 0.5);
        assert!(!noisy.is_significant);
    }

    /// Regression test: `run_and_check` used to copy one result set `num_runs`
    /// times. It must now refuse, and the replacement must really re-run.
    #[test]
    fn test_run_and_check_refuses_to_duplicate_results() {
        let mut benchmark =
            ContinuousBenchmark::new(test_config("norerun")).expect("operation failed in test");
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::default());

        let error = benchmark
            .run_and_check(&mut suite)
            .expect_err("must not fabricate repeated runs");
        assert!(
            error.to_string().contains("run_and_check_with"),
            "unexpected error: {}",
            error
        );
    }

    #[test]
    fn test_run_and_check_with_executes_every_run() {
        let mut config = test_config("rerun");
        config.num_runs = 4;
        let results_dir = config.results_dir.clone();
        let _ = std::fs::remove_dir_all(&results_dir);

        let mut benchmark = ContinuousBenchmark::new(config).expect("operation failed in test");
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::default());

        let mut invocations = 0usize;
        let regressions = benchmark
            .run_and_check_with(&mut suite, |suite| {
                invocations += 1;
                // Each run reports a different latency, so the collected
                // samples must have non-zero variance.
                suite.push_result(make_result("synthetic", 100.0 + invocations as f64));
                Ok(())
            })
            .expect("run_and_check_with failed");

        assert_eq!(invocations, 4, "every configured run must be executed");
        // No baseline exists yet, so no regression can be reported.
        assert!(regressions.is_empty());

        let stored = benchmark.history.runs.values().next().expect("a run must have been recorded");
        assert_eq!(stored.len(), 4, "each run contributes one result");
        let latencies: Vec<f64> = stored.iter().map(|r| r.avg_latency_ms).collect();
        let variance = crate::statistics::sample_variance(&latencies).expect("4 samples");
        assert!(
            variance > 0.0,
            "duplicated results would give zero variance; got {:?}",
            latencies
        );

        let _ = std::fs::remove_dir_all(&results_dir);
    }

    #[test]
    fn test_run_and_check_with_rejects_empty_runs() {
        let mut benchmark =
            ContinuousBenchmark::new(test_config("empty")).expect("operation failed in test");
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::default());

        let error = benchmark
            .run_and_check_with(&mut suite, |_suite| Ok(()))
            .expect_err("a run that measured nothing must not produce a verdict");
        assert!(error.to_string().contains("no results"));
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
