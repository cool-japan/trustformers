//! Core benchmarking infrastructure for TrustformeRS

use crate::tensor::Tensor;
use crate::traits::Model;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Model input structure for benchmarking
#[derive(Clone)]
pub struct ModelInput {
    pub input_ids: Tensor,
    pub attention_mask: Option<Tensor>,
    pub token_type_ids: Option<Tensor>,
    pub position_ids: Option<Tensor>,
}

/// Model output structure for benchmarking
#[derive(Default)]
pub struct ModelOutput {
    pub hidden_states: Option<Tensor>,
    pub logits: Option<Tensor>,
    pub attentions: Option<Vec<Tensor>>,
}

/// Configuration for benchmark runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Batch sizes to test
    pub batch_sizes: Vec<usize>,
    /// Sequence lengths to test
    pub sequence_lengths: Vec<usize>,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of benchmark iterations
    pub num_iterations: usize,
    /// Whether to measure memory usage
    pub measure_memory: bool,
    /// Device to run benchmarks on (cpu, cuda, etc.)
    pub device: String,
    /// Whether to use mixed precision
    pub use_fp16: bool,
    /// Whether to benchmark generation tasks
    pub include_generation: bool,
    /// Maximum generation length for generation benchmarks
    pub max_generation_length: Option<usize>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            batch_sizes: vec![1, 4, 8, 16, 32],
            sequence_lengths: vec![128, 256, 512, 1024, 2048],
            warmup_iterations: 10,
            num_iterations: 100,
            measure_memory: true,
            device: "cpu".to_string(),
            use_fp16: false,
            include_generation: false,
            max_generation_length: Some(256),
        }
    }
}

/// Result of a single benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Name of the benchmark
    pub name: String,
    /// Model type (BERT, GPT-2, etc.)
    pub model_type: String,
    /// Average latency per forward pass
    pub avg_latency_ms: f64,
    /// P50 latency
    pub p50_latency_ms: f64,
    /// P95 latency
    pub p95_latency_ms: f64,
    /// P99 latency
    pub p99_latency_ms: f64,
    /// Minimum latency
    pub min_latency_ms: f64,
    /// Maximum latency
    pub max_latency_ms: f64,
    /// Standard deviation of latency
    pub std_dev_ms: f64,
    /// Throughput in tokens per second
    pub throughput_tokens_per_sec: f64,
    /// Throughput in batches per second
    pub throughput_batches_per_sec: f64,
    /// Memory usage in bytes
    pub memory_bytes: Option<usize>,
    /// Peak memory usage
    pub peak_memory_bytes: Option<usize>,
    /// Configuration parameters
    pub parameters: HashMap<String, String>,
    /// Raw timing data
    pub raw_timings: Vec<Duration>,
    /// Timestamp of the benchmark
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl BenchmarkResult {
    /// Calculate percentile from sorted timings
    fn percentile(sorted_timings: &[Duration], percentile: f64) -> Duration {
        let index = ((sorted_timings.len() - 1) as f64 * percentile / 100.0) as usize;
        sorted_timings[index]
    }

    /// Create result from raw timings
    pub fn from_timings(
        name: String,
        model_type: String,
        timings: Vec<Duration>,
        batch_size: usize,
        seq_len: usize,
        memory_bytes: Option<usize>,
        peak_memory_bytes: Option<usize>,
    ) -> Self {
        let mut sorted_timings = timings.clone();
        sorted_timings.sort();

        let total_duration: Duration = timings.iter().sum();
        let avg_duration = total_duration / timings.len() as u32;

        let avg_ms = avg_duration.as_secs_f64() * 1000.0;
        let variance = timings
            .iter()
            .map(|t| {
                let diff = t.as_secs_f64() - avg_duration.as_secs_f64();
                diff * diff
            })
            .sum::<f64>()
            / timings.len() as f64;
        let std_dev_ms = variance.sqrt() * 1000.0;

        let tokens_per_batch = batch_size * seq_len;
        let batches_per_sec = 1.0 / avg_duration.as_secs_f64();
        let tokens_per_sec = tokens_per_batch as f64 * batches_per_sec;

        let mut parameters = HashMap::new();
        parameters.insert("batch_size".to_string(), batch_size.to_string());
        parameters.insert("seq_len".to_string(), seq_len.to_string());
        parameters.insert("num_iterations".to_string(), timings.len().to_string());

        Self {
            name,
            model_type,
            avg_latency_ms: avg_ms,
            p50_latency_ms: Self::percentile(&sorted_timings, 50.0).as_secs_f64() * 1000.0,
            p95_latency_ms: Self::percentile(&sorted_timings, 95.0).as_secs_f64() * 1000.0,
            p99_latency_ms: Self::percentile(&sorted_timings, 99.0).as_secs_f64() * 1000.0,
            min_latency_ms: sorted_timings[0].as_secs_f64() * 1000.0,
            max_latency_ms: sorted_timings[sorted_timings.len() - 1].as_secs_f64() * 1000.0,
            std_dev_ms,
            throughput_tokens_per_sec: tokens_per_sec,
            throughput_batches_per_sec: batches_per_sec,
            memory_bytes,
            peak_memory_bytes,
            parameters,
            raw_timings: timings,
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Main benchmark suite for running performance tests
pub struct BenchmarkSuite {
    results: Vec<BenchmarkResult>,
    config: BenchmarkConfig,
}

impl BenchmarkSuite {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            results: Vec::new(),
            config,
        }
    }

    /// Run inference benchmark for a model that takes ModelInput
    pub fn benchmark_inference<M>(&mut self, model: &M, model_name: &str) -> Result<()>
    where
        M: Model<Input = ModelInput, Output = ModelOutput>,
    {
        println!("Benchmarking {} inference...", model_name);

        for &batch_size in &self.config.batch_sizes {
            for &seq_len in &self.config.sequence_lengths {
                let result =
                    self.run_single_inference_benchmark(model, model_name, batch_size, seq_len)?;
                self.results.push(result);
            }
        }

        Ok(())
    }

    /// Run a single inference benchmark
    fn run_single_inference_benchmark<M>(
        &self,
        model: &M,
        model_name: &str,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<BenchmarkResult>
    where
        M: Model<Input = ModelInput, Output = ModelOutput>,
    {
        println!("  Batch size: {}, Sequence length: {}", batch_size, seq_len);

        // Create dummy input
        let input_ids = Tensor::zeros(&[batch_size, seq_len])?;
        let attention_mask = Some(Tensor::ones(&[batch_size, seq_len])?);

        let model_input = ModelInput {
            input_ids,
            attention_mask,
            token_type_ids: None,
            position_ids: None,
        };

        // Initial RSS. `None` when memory measurement is off, or when the host
        // does not report this process: either way no memory figure is derived.
        let initial_memory =
            if self.config.measure_memory { self.get_memory_usage() } else { None };

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = model.forward(model_input.clone())?;
        }

        // Benchmark
        let mut timings = Vec::with_capacity(self.config.num_iterations);
        let mut peak_memory = initial_memory;

        for _ in 0..self.config.num_iterations {
            let start = Instant::now();
            let _ = model.forward(model_input.clone())?;
            let duration = start.elapsed();
            timings.push(duration);

            if self.config.measure_memory {
                if let (Some(peak), Some(current)) = (peak_memory.as_mut(), self.get_memory_usage())
                {
                    *peak = (*peak).max(current);
                }
            }
        }

        // Memory growth over the run, only when both readings are real.
        let memory_usage = match (self.config.measure_memory, initial_memory) {
            (true, Some(initial)) => {
                self.get_memory_usage().map(|final_rss| final_rss.saturating_sub(initial))
            },
            _ => None,
        };
        let peak_growth = match (peak_memory, initial_memory) {
            (Some(peak), Some(initial)) => Some(peak.saturating_sub(initial)),
            _ => None,
        };

        Ok(BenchmarkResult::from_timings(
            format!("{}_inference_b{}_s{}", model_name, batch_size, seq_len),
            model_name.to_string(),
            timings,
            batch_size,
            seq_len,
            memory_usage,
            peak_growth,
        ))
    }

    /// Resident set size of this process, in bytes, from `sysinfo`.
    ///
    /// Returns `None` when the host does not report the process (in which case
    /// the benchmark records no memory figure at all, rather than an estimate:
    /// this value feeds the regression statistics in
    /// [`crate::performance::ContinuousBenchmark`]).
    fn get_memory_usage(&self) -> Option<usize> {
        use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};

        let mut system = System::new();
        let pid = sysinfo::Pid::from_u32(std::process::id());
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_memory(),
        );
        system.process(pid).map(|process| process.memory() as usize)
    }

    /// Print benchmark summary
    pub fn print_summary(&self) {
        println!("\n=== Benchmark Results Summary ===");
        println!(
            "{:<40} {:>12} {:>12} {:>12} {:>12} {:>15}",
            "Benchmark", "Avg (ms)", "P50 (ms)", "P95 (ms)", "P99 (ms)", "Throughput (tok/s)"
        );
        println!("{}", "-".repeat(103));

        for result in &self.results {
            println!(
                "{:<40} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.0}",
                result.name,
                result.avg_latency_ms,
                result.p50_latency_ms,
                result.p95_latency_ms,
                result.p99_latency_ms,
                result.throughput_tokens_per_sec,
            );
        }
    }

    /// Export results to JSON
    pub fn export_json(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.results)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Export results to CSV
    pub fn export_csv(&self, path: &str) -> Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        // Write header
        writeln!(file, "name,model_type,batch_size,seq_len,avg_latency_ms,p50_ms,p95_ms,p99_ms,min_ms,max_ms,std_dev_ms,throughput_tokens_sec,throughput_batches_sec,memory_bytes,timestamp")?;

        // Write data
        for result in &self.results {
            writeln!(
                file,
                "{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.0},{:.2},{},{}",
                result.name,
                result.model_type,
                result.parameters.get("batch_size").unwrap_or(&"0".to_string()),
                result.parameters.get("seq_len").unwrap_or(&"0".to_string()),
                result.avg_latency_ms,
                result.p50_latency_ms,
                result.p95_latency_ms,
                result.p99_latency_ms,
                result.min_latency_ms,
                result.max_latency_ms,
                result.std_dev_ms,
                result.throughput_tokens_per_sec,
                result.throughput_batches_per_sec,
                result.memory_bytes.unwrap_or(0),
                result.timestamp.to_rfc3339(),
            )?;
        }

        Ok(())
    }

    /// Get results
    /// Record a benchmark result produced outside the built-in runners.
    pub fn push_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Drop all recorded results so the suite can be re-run from a clean state.
    ///
    /// Used by [`crate::performance::ContinuousBenchmark::run_and_check_with`]
    /// to make each of the N runs independent.
    pub fn clear_results(&mut self) {
        self.results.clear();
    }

    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Compare with baseline results
    pub fn compare_with_baseline(&self, baseline: &[BenchmarkResult]) -> Vec<ComparisonSummary> {
        let mut comparisons = Vec::new();

        for result in &self.results {
            if let Some(baseline_result) = baseline.iter().find(|b| b.name == result.name) {
                let speedup = baseline_result.avg_latency_ms / result.avg_latency_ms;
                let throughput_improvement =
                    result.throughput_tokens_per_sec / baseline_result.throughput_tokens_per_sec;

                comparisons.push(ComparisonSummary {
                    benchmark_name: result.name.clone(),
                    speedup,
                    throughput_improvement,
                    latency_reduction_percent: (1.0
                        - result.avg_latency_ms / baseline_result.avg_latency_ms)
                        * 100.0,
                    memory_reduction_percent: if let (Some(current), Some(baseline)) =
                        (result.memory_bytes, baseline_result.memory_bytes)
                    {
                        Some((1.0 - current as f64 / baseline as f64) * 100.0)
                    } else {
                        None
                    },
                });
            }
        }

        comparisons
    }
}

/// Summary of performance comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    pub benchmark_name: String,
    pub speedup: f64,
    pub throughput_improvement: f64,
    pub latency_reduction_percent: f64,
    pub memory_reduction_percent: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: `get_memory_usage` shelled out to `ps` on macOS and
    /// otherwise fell back to `results.len() * 50MB + 100MB` — a fabricated
    /// figure that feeds `BenchmarkResult::memory_bytes` and, from there, the
    /// regression statistics in `ContinuousBenchmark`.
    #[test]
    fn test_memory_usage_is_a_real_rss_reading() {
        let suite = BenchmarkSuite::new(BenchmarkConfig::default());
        let rss = suite.get_memory_usage().expect("sysinfo reports this process");

        assert!(rss > 0, "resident set size must be positive");
        // The old fallback for an empty suite was exactly 100 MiB.
        assert_ne!(
            rss,
            100 * 1024 * 1024,
            "the fabricated fallback must be gone"
        );
        // A test process is far below 4 GiB resident.
        assert!(rss < 4 * 1024 * 1024 * 1024, "implausible RSS: {rss}");

        // Two readings of the same process must be in the same ballpark: a
        // synthetic value derived from `results.len()` would jump.
        let second = suite.get_memory_usage().expect("second reading");
        let ratio = rss.max(second) as f64 / rss.min(second).max(1) as f64;
        assert!(
            ratio < 4.0,
            "readings {rss} and {second} are not the same process"
        );
    }

    #[test]
    fn test_benchmark_result_from_timings() {
        let timings = vec![
            Duration::from_millis(10),
            Duration::from_millis(12),
            Duration::from_millis(11),
            Duration::from_millis(15),
            Duration::from_millis(13),
        ];

        let result = BenchmarkResult::from_timings(
            "test_benchmark".to_string(),
            "TestModel".to_string(),
            timings,
            4,
            128,
            Some(1024 * 1024),
            Some(2048 * 1024),
        );

        assert_eq!(result.name, "test_benchmark");
        assert_eq!(result.model_type, "TestModel");
        assert!(result.avg_latency_ms > 0.0);
        assert!(result.throughput_tokens_per_sec > 0.0);
        assert_eq!(
            result.parameters.get("batch_size").expect("expected value not found"),
            "4"
        );
        assert_eq!(
            result.parameters.get("seq_len").expect("expected value not found"),
            "128"
        );
    }

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.batch_sizes, vec![1, 4, 8, 16, 32]);
        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.num_iterations, 100);
        assert!(config.measure_memory);
    }
}
