//! # Simplified Model Benchmarking Suite
//!
//! This module provides basic benchmarking capabilities for comparing model performance.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use trustformers_core::tensor::Tensor;

/// Basic benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
    /// Test batch sizes
    pub batch_sizes: Vec<usize>,
    /// Test sequence lengths
    pub sequence_lengths: Vec<usize>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            benchmark_iterations: 20,
            batch_sizes: vec![1, 4, 8, 16],
            sequence_lengths: vec![128, 256, 512],
        }
    }
}

/// Benchmark results for a single test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Test name
    pub test_name: String,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Standard deviation of latency
    pub std_latency_ms: f64,
    /// Throughput in samples per second
    pub throughput: f64,
    /// Memory usage in MB (if measured)
    pub memory_mb: Option<f64>,
}

/// Model benchmark function type
pub type ModelBenchmarkFn = Box<dyn Fn(&Tensor) -> Result<Tensor>>;

/// Simplified benchmarking suite
pub struct BenchmarkSuite {
    config: BenchmarkConfig,
    models: HashMap<String, ModelBenchmarkFn>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            models: HashMap::new(),
        }
    }

    /// Add a model function to benchmark
    pub fn add_model<F>(&mut self, name: &str, model_fn: F)
    where
        F: Fn(&Tensor) -> Result<Tensor> + 'static,
    {
        self.models.insert(name.to_string(), Box::new(model_fn));
    }

    /// Run all benchmarks
    pub fn run_benchmarks(&self) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        for (model_name, model_fn) in &self.models {
            for &batch_size in &self.config.batch_sizes {
                for &seq_len in &self.config.sequence_lengths {
                    let test_name =
                        format!("{}_{}_{}x{}", model_name, "forward", batch_size, seq_len);
                    let input = Tensor::randn(&[batch_size, seq_len])?;

                    let result = self.benchmark_model_with_input(&test_name, model_fn, &input)?;
                    results.push(result);
                }
            }
        }

        Ok(results)
    }

    /// Benchmark a single model with specific input
    fn benchmark_model_with_input(
        &self,
        test_name: &str,
        model_fn: &ModelBenchmarkFn,
        input: &Tensor,
    ) -> Result<BenchmarkResult> {
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            model_fn(input)?;
        }

        // Benchmark
        let mut durations = Vec::new();
        for _ in 0..self.config.benchmark_iterations {
            let start = Instant::now();
            model_fn(input)?;
            durations.push(start.elapsed().as_millis() as f64);
        }

        // Calculate statistics
        let avg_latency_ms = durations.iter().sum::<f64>() / durations.len() as f64;
        let variance = durations.iter().map(|&x| (x - avg_latency_ms).powi(2)).sum::<f64>()
            / durations.len() as f64;
        let std_latency_ms = variance.sqrt();
        let throughput = 1000.0 / avg_latency_ms; // samples per second

        Ok(BenchmarkResult {
            test_name: test_name.to_string(),
            avg_latency_ms,
            std_latency_ms,
            throughput,
            memory_mb: None, // Memory measurement would require platform-specific code
        })
    }

    /// Generate a simple benchmark report
    pub fn generate_report(&self, results: &[BenchmarkResult]) -> String {
        let mut report = String::new();
        report.push_str("# Benchmark Report\n\n");

        report
            .push_str("| Test Name | Avg Latency (ms) | Std Dev (ms) | Throughput (samples/s) |\n");
        report.push_str("|-----------|------------------|--------------|----------------------|\n");

        for result in results {
            report.push_str(&format!(
                "| {} | {:.2} | {:.2} | {:.2} |\n",
                result.test_name, result.avg_latency_ms, result.std_latency_ms, result.throughput
            ));
        }

        report
    }
}

/// Model performance comparison utilities
pub struct ModelComparator;

impl ModelComparator {
    /// Compare two sets of benchmark results
    pub fn compare_results(baseline: &[BenchmarkResult], comparison: &[BenchmarkResult]) -> String {
        let mut report = String::new();
        report.push_str("# Performance Comparison\n\n");

        // Match results by test name
        let baseline_map: HashMap<_, _> =
            baseline.iter().map(|r| (r.test_name.clone(), r)).collect();

        report
            .push_str("| Test | Baseline (ms) | Comparison (ms) | Speedup | Throughput Change |\n");
        report
            .push_str("|------|---------------|-----------------|---------|------------------|\n");

        for comp_result in comparison {
            if let Some(base_result) = baseline_map.get(&comp_result.test_name) {
                let speedup = base_result.avg_latency_ms / comp_result.avg_latency_ms;
                let throughput_change = (comp_result.throughput - base_result.throughput)
                    / base_result.throughput
                    * 100.0;

                report.push_str(&format!(
                    "| {} | {:.2} | {:.2} | {:.2}x | {:.1}% |\n",
                    comp_result.test_name,
                    base_result.avg_latency_ms,
                    comp_result.avg_latency_ms,
                    speedup,
                    throughput_change
                ));
            }
        }

        report
    }
}

/// Simple benchmarking utilities
pub struct BenchmarkUtils;

impl BenchmarkUtils {
    /// Create test inputs for benchmarking
    pub fn create_test_inputs(batch_sizes: &[usize], seq_lengths: &[usize]) -> Result<Vec<Tensor>> {
        let mut inputs = Vec::new();
        for &batch_size in batch_sizes {
            for &seq_len in seq_lengths {
                inputs.push(Tensor::randn(&[batch_size, seq_len])?);
            }
        }
        Ok(inputs)
    }

    /// Measure single function execution time
    pub fn measure_execution_time<F, T>(mut f: F) -> (T, Duration)
    where
        F: FnMut() -> T,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_creation() {
        let config = BenchmarkConfig::default();
        let suite = BenchmarkSuite::new(config);
        // Suite should be created successfully
        assert_eq!(suite.models.len(), 0);
    }

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 5);
        assert_eq!(config.benchmark_iterations, 20);
        assert!(!config.batch_sizes.is_empty());
        assert!(!config.sequence_lengths.is_empty());
    }

    #[test]
    fn test_benchmark_utils() -> Result<()> {
        let inputs = BenchmarkUtils::create_test_inputs(&[2, 4], &[10, 20])?;
        assert_eq!(inputs.len(), 4);
        Ok(())
    }

    #[test]
    fn test_execution_time_measurement() {
        let (result, duration) = BenchmarkUtils::measure_execution_time(|| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);
        assert!(duration.as_millis() >= 10);
    }
}
