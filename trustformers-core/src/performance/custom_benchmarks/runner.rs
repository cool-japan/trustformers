//! Benchmark runner for executing custom benchmarks

use super::{BenchmarkReport, CustomBenchmark};
use anyhow::Result;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for running benchmarks
#[derive(Clone, Serialize, Deserialize)]
pub struct RunConfig {
    /// Run mode
    pub mode: RunMode,
    /// Whether to run in parallel
    pub parallel: bool,
    /// Number of parallel workers
    pub num_workers: usize,
    /// Output directory for results
    pub output_dir: Option<String>,
    /// Whether to save raw data
    pub save_raw_data: bool,
    /// Maximum duration for benchmarks
    pub max_duration: Option<Duration>,
    /// Minimum duration for benchmarks
    pub min_duration: Option<Duration>,
    /// Whether to validate results
    pub validate_results: bool,
    /// Progress reporting
    #[serde(skip)]
    pub progress_callback: Option<Arc<dyn Fn(ProgressUpdate) + Send + Sync>>,
}

impl std::fmt::Debug for RunConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunConfig")
            .field("mode", &self.mode)
            .field("parallel", &self.parallel)
            .field("num_workers", &self.num_workers)
            .field("output_dir", &self.output_dir)
            .field("save_raw_data", &self.save_raw_data)
            .field("max_duration", &self.max_duration)
            .field("min_duration", &self.min_duration)
            .field("validate_results", &self.validate_results)
            .field("progress_callback", &self.progress_callback.is_some())
            .finish()
    }
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            mode: RunMode::Standard,
            parallel: false,
            num_workers: num_cpus::get(),
            output_dir: None,
            save_raw_data: false,
            max_duration: None,
            min_duration: None,
            validate_results: true,
            progress_callback: None,
        }
    }
}

/// Run mode for benchmarks
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RunMode {
    /// Standard run with configured iterations
    Standard,
    /// Quick run for smoke testing
    Quick,
    /// Extended run for detailed analysis
    Extended,
    /// Continuous run until stopped
    Continuous,
    /// Profile mode with detailed tracing
    Profile,
}

/// Progress update during benchmark execution
#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    pub benchmark_name: String,
    pub current_iteration: usize,
    pub total_iterations: usize,
    pub elapsed_time: Duration,
    pub estimated_remaining: Option<Duration>,
}

/// Benchmark runner
pub struct BenchmarkRunner {
    config: RunConfig,
    benchmarks: Vec<Box<dyn CustomBenchmark>>,
    results: Arc<Mutex<Vec<BenchmarkReport>>>,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(config: RunConfig) -> Self {
        Self {
            config,
            benchmarks: Vec::new(),
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Add a benchmark to run
    pub fn add_benchmark(mut self, benchmark: Box<dyn CustomBenchmark>) -> Self {
        self.benchmarks.push(benchmark);
        self
    }

    /// Add multiple benchmarks
    pub fn add_benchmarks(mut self, benchmarks: Vec<Box<dyn CustomBenchmark>>) -> Self {
        self.benchmarks.extend(benchmarks);
        self
    }

    /// Run all benchmarks
    pub fn run(mut self) -> Result<Vec<BenchmarkReport>> {
        if self.benchmarks.is_empty() {
            anyhow::bail!("No benchmarks to run");
        }

        println!("Running {} benchmarks...", self.benchmarks.len());

        if self.config.parallel && self.benchmarks.len() > 1 {
            self.run_parallel()?;
        } else {
            self.run_sequential()?;
        }

        let output_dir = self.config.output_dir.clone();
        let save_raw_data = self.config.save_raw_data;
        let results = Arc::try_unwrap(self.results)
            .map(|mutex| mutex.into_inner())
            .unwrap_or_else(|arc| (*arc.lock()).clone());

        if let Some(output_dir) = output_dir {
            Self::save_results_static(&results, &output_dir, save_raw_data)?;
        }

        Ok(results)
    }

    /// Run benchmarks sequentially
    fn run_sequential(&mut self) -> Result<()> {
        for benchmark in &mut self.benchmarks {
            let report = Self::run_single_benchmark(&self.config, benchmark.as_mut())?;
            self.results.lock().push(report);
        }
        Ok(())
    }

    /// Run benchmarks in parallel
    fn run_parallel(&mut self) -> Result<()> {
        use rayon::prelude::*;

        let config = &self.config;
        let results = self
            .benchmarks
            .par_iter_mut()
            .map(|benchmark| Self::run_single_benchmark(config, benchmark.as_mut()))
            .collect::<Result<Vec<_>>>()?;

        *self.results.lock() = results;
        Ok(())
    }

    /// Run a single benchmark
    fn run_single_benchmark(
        config: &RunConfig,
        benchmark: &mut dyn CustomBenchmark,
    ) -> Result<BenchmarkReport> {
        let start_time = Instant::now();

        println!("\nRunning benchmark: {}", benchmark.name());
        println!("Description: {}", benchmark.description());

        // Setup
        benchmark.setup()?;

        let benchmark_config = benchmark.config();
        let (warmup_iterations, measurement_iterations) = match config.mode {
            RunMode::Quick => (5, 10),
            RunMode::Standard => (
                benchmark_config.warmup_iterations,
                benchmark_config.iterations,
            ),
            RunMode::Extended => (
                benchmark_config.warmup_iterations * 2,
                benchmark_config.iterations * 5,
            ),
            RunMode::Continuous => (benchmark_config.warmup_iterations, usize::MAX),
            RunMode::Profile => (3, 10),
        };

        // Warmup
        if warmup_iterations > 0 {
            println!("  Warming up ({} iterations)...", warmup_iterations);
            benchmark.warmup(warmup_iterations)?;
        }

        // Measurement
        println!("  Measuring ({} iterations)...", measurement_iterations);
        let mut iterations = Vec::new();
        let mut total_duration = Duration::ZERO;

        for i in 0..measurement_iterations {
            // Check time limits
            if let Some(max_duration) = config.max_duration {
                if total_duration > max_duration {
                    println!("  Reached maximum duration, stopping early");
                    break;
                }
            }

            if config.mode != RunMode::Continuous && i > measurement_iterations {
                if let Some(min_duration) = config.min_duration {
                    if total_duration > min_duration {
                        break;
                    }
                }
            }

            // Run iteration
            let iteration = benchmark.run_iteration()?;
            total_duration += iteration.duration;

            // Validate if configured
            if config.validate_results {
                let valid = benchmark.validate(&iteration)?;
                if !valid {
                    eprintln!("  Warning: Iteration {} failed validation", i);
                }
            }

            // Progress callback
            if let Some(callback) = &config.progress_callback {
                let update = ProgressUpdate {
                    benchmark_name: benchmark.name().to_string(),
                    current_iteration: i + 1,
                    total_iterations: measurement_iterations,
                    elapsed_time: start_time.elapsed(),
                    estimated_remaining: Self::estimate_remaining_time(
                        i + 1,
                        measurement_iterations,
                        start_time.elapsed(),
                    ),
                };
                callback(update);
            }

            iterations.push(iteration);

            // Break for continuous mode if interrupted
            if config.mode == RunMode::Continuous {
                // Note: In static context, we can't check should_stop()
                // This would need to be handled by the caller
                break;
            }
        }

        // Teardown
        benchmark.teardown()?;

        // Create report
        let report = BenchmarkReport::from_iterations(
            benchmark.name().to_string(),
            benchmark.description().to_string(),
            benchmark.tags(),
            iterations,
            start_time.elapsed(),
        );

        println!("  Completed in {:.2}s", start_time.elapsed().as_secs_f64());
        // Note: print_summary is an instance method, can't call from static context

        Ok(report)
    }

    /// Estimate remaining time
    fn estimate_remaining_time(
        current: usize,
        total: usize,
        elapsed: Duration,
    ) -> Option<Duration> {
        if current == 0 || current >= total {
            return None;
        }

        let per_iteration = elapsed / current as u32;
        let remaining_iterations = total - current;
        Some(per_iteration * remaining_iterations as u32)
    }

    /// Check if continuous mode should stop
    fn should_stop(&self) -> bool {
        // In a real implementation, check for interrupt signal
        false
    }

    /// Print benchmark summary
    fn print_summary(&self, report: &BenchmarkReport) {
        println!("\n  Summary:");
        println!("    Total iterations: {}", report.iterations);
        println!(
            "    Total duration: {:.2}s",
            report.total_duration.as_secs_f64()
        );

        if let Some(stats) = &report.duration_stats {
            println!("    Duration stats:");
            println!("      Mean: {:.2}ms", stats.mean * 1000.0);
            println!("      Std Dev: {:.2}ms", stats.std_dev * 1000.0);
            println!("      Min: {:.2}ms", stats.min * 1000.0);
            println!("      Max: {:.2}ms", stats.max * 1000.0);
        }

        if let Some(throughput) = report.aggregate_metrics.get("throughput") {
            println!("    Throughput: {:.2} items/sec", throughput.mean);
        }
    }

    /// Save results to disk
    fn save_results(&self, results: &[BenchmarkReport], output_dir: &str) -> Result<()> {
        use std::fs;
        use std::path::Path;

        fs::create_dir_all(output_dir)?;

        // Save summary
        let summary_path = Path::new(output_dir).join("summary.json");
        let summary_json = serde_json::to_string_pretty(results)?;
        fs::write(summary_path, summary_json)?;

        // Save individual reports if configured
        if self.config.save_raw_data {
            for report in results {
                let filename = format!("{}_raw.json", report.name.replace(' ', "_"));
                let path = Path::new(output_dir).join(filename);
                let json = serde_json::to_string_pretty(report)?;
                fs::write(path, json)?;
            }
        }

        println!("\nResults saved to: {}", output_dir);
        Ok(())
    }

    /// Static version of save_results for use after Arc::try_unwrap
    fn save_results_static(
        results: &[BenchmarkReport],
        output_dir: &str,
        save_raw_data: bool,
    ) -> Result<()> {
        use std::fs;
        use std::path::Path;

        fs::create_dir_all(output_dir)?;

        // Save summary
        let summary_path = Path::new(output_dir).join("summary.json");
        let summary_json = serde_json::to_string_pretty(results)?;
        fs::write(summary_path, summary_json)?;

        // Save individual reports if configured
        if save_raw_data {
            for report in results {
                let filename = format!("{}_raw.json", report.name.replace(' ', "_"));
                let path = Path::new(output_dir).join(filename);
                let json = serde_json::to_string_pretty(report)?;
                fs::write(path, json)?;
            }
        }

        println!("\nResults saved to: {}", output_dir);
        Ok(())
    }
}

/// Builder pattern for benchmark runner
pub struct BenchmarkRunnerBuilder {
    config: RunConfig,
    benchmarks: Vec<Box<dyn CustomBenchmark>>,
}

impl Default for BenchmarkRunnerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkRunnerBuilder {
    pub fn new() -> Self {
        Self {
            config: RunConfig::default(),
            benchmarks: Vec::new(),
        }
    }

    pub fn mode(mut self, mode: RunMode) -> Self {
        self.config.mode = mode;
        self
    }

    pub fn parallel(mut self, workers: usize) -> Self {
        self.config.parallel = true;
        self.config.num_workers = workers;
        self
    }

    pub fn output_dir(mut self, dir: impl Into<String>) -> Self {
        self.config.output_dir = Some(dir.into());
        self
    }

    pub fn save_raw_data(mut self, save: bool) -> Self {
        self.config.save_raw_data = save;
        self
    }

    pub fn with_progress<F>(mut self, callback: F) -> Self
    where
        F: Fn(ProgressUpdate) + Send + Sync + 'static,
    {
        self.config.progress_callback = Some(Arc::new(callback));
        self
    }

    pub fn add_benchmark(mut self, benchmark: Box<dyn CustomBenchmark>) -> Self {
        self.benchmarks.push(benchmark);
        self
    }

    pub fn build(self) -> BenchmarkRunner {
        BenchmarkRunner::new(self.config).add_benchmarks(self.benchmarks)
    }

    pub fn run(self) -> Result<Vec<BenchmarkReport>> {
        self.build().run()
    }
}

/// Comparison runner for A/B testing benchmarks
pub struct ComparisonRunner {
    baseline: Box<dyn CustomBenchmark>,
    candidates: Vec<Box<dyn CustomBenchmark>>,
    config: RunConfig,
}

impl ComparisonRunner {
    pub fn new(
        baseline: Box<dyn CustomBenchmark>,
        candidates: Vec<Box<dyn CustomBenchmark>>,
    ) -> Self {
        Self {
            baseline,
            candidates,
            config: RunConfig::default(),
        }
    }

    pub fn run(mut self) -> Result<ComparisonReport> {
        let runner = BenchmarkRunner::new(self.config.clone());

        // Run baseline
        let baseline_report =
            BenchmarkRunner::run_single_benchmark(&runner.config, self.baseline.as_mut())?;

        // Run candidates
        let mut candidate_reports = Vec::new();
        for mut candidate in self.candidates {
            let report = BenchmarkRunner::run_single_benchmark(&runner.config, candidate.as_mut())?;
            candidate_reports.push(report);
        }

        Ok(ComparisonReport {
            baseline: baseline_report,
            candidates: candidate_reports,
            timestamp: chrono::Utc::now(),
        })
    }
}

/// Comparison report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub baseline: BenchmarkReport,
    pub candidates: Vec<BenchmarkReport>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance::custom_benchmarks::{BenchmarkIteration, BenchmarkMetrics};

    struct TestBenchmark {
        iterations_run: Arc<Mutex<usize>>,
    }

    impl CustomBenchmark for TestBenchmark {
        fn name(&self) -> &str {
            "test_benchmark"
        }

        fn description(&self) -> &str {
            "Test benchmark"
        }

        fn run_iteration(&mut self) -> Result<BenchmarkIteration> {
            *self.iterations_run.lock() += 1;

            Ok(BenchmarkIteration {
                duration: Duration::from_millis(10),
                metrics: BenchmarkMetrics::default(),
                validation_passed: Some(true),
                metadata: None,
            })
        }
    }

    #[test]
    fn test_runner_sequential() {
        let benchmark = Box::new(TestBenchmark {
            iterations_run: Arc::new(Mutex::new(0)),
        });

        let results = BenchmarkRunnerBuilder::new()
            .mode(RunMode::Quick)
            .add_benchmark(benchmark)
            .run()
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "test_benchmark");
        assert!(results[0].iterations > 0);
    }

    #[test]
    fn test_runner_parallel() {
        let benchmarks: Vec<Box<dyn CustomBenchmark>> = (0..4)
            .map(|_| {
                Box::new(TestBenchmark {
                    iterations_run: Arc::new(Mutex::new(0)),
                }) as Box<dyn CustomBenchmark>
            })
            .collect();

        let mut builder = BenchmarkRunnerBuilder::new().mode(RunMode::Quick).parallel(2);

        for benchmark in benchmarks {
            builder = builder.add_benchmark(benchmark);
        }

        let results = builder.run().unwrap();

        assert_eq!(results.len(), 4);
    }
}
