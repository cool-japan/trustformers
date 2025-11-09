//! Performance comparison with other frameworks (PyTorch, HuggingFace)

#![allow(dead_code)] // Comparison framework with reserved features for future benchmarking

use crate::performance::benchmark::BenchmarkResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Framework for comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Framework {
    TrustformeRS,
    PyTorch,
    HuggingFace,
    TensorFlow,
    ONNX,
}

impl std::fmt::Display for Framework {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Framework::TrustformeRS => write!(f, "TrustformeRS"),
            Framework::PyTorch => write!(f, "PyTorch"),
            Framework::HuggingFace => write!(f, "HuggingFace"),
            Framework::TensorFlow => write!(f, "TensorFlow"),
            Framework::ONNX => write!(f, "ONNX"),
        }
    }
}

/// Comparison result between frameworks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Benchmark name
    pub benchmark_name: String,
    /// Model type
    pub model_type: String,
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub sequence_length: usize,
    /// Results by framework
    pub framework_results: HashMap<Framework, FrameworkMetrics>,
    /// Relative performance (TrustformeRS vs others)
    pub relative_performance: HashMap<Framework, RelativePerformance>,
}

/// Performance metrics for a framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkMetrics {
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// P95 latency
    pub p95_latency_ms: f64,
    /// P99 latency
    pub p99_latency_ms: f64,
    /// Throughput in tokens/second
    pub throughput_tokens_per_sec: f64,
    /// Memory usage in MB
    pub memory_mb: Option<f64>,
    /// GPU memory usage in MB
    pub gpu_memory_mb: Option<f64>,
    /// Framework version
    pub version: String,
}

/// Relative performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelativePerformance {
    /// Speedup factor (>1 means TrustformeRS is faster)
    pub speedup: f64,
    /// Throughput ratio
    pub throughput_ratio: f64,
    /// Memory efficiency ratio
    pub memory_efficiency: Option<f64>,
    /// Latency improvement percentage
    pub latency_improvement_percent: f64,
}

/// Model comparison manager
pub struct ModelComparison {
    results: Vec<ComparisonResult>,
}

impl Default for ModelComparison {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelComparison {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Add TrustformeRS benchmark results
    pub fn add_trustformers_results(&mut self, results: &[BenchmarkResult]) {
        for result in results {
            let framework_metrics = FrameworkMetrics {
                avg_latency_ms: result.avg_latency_ms,
                p95_latency_ms: result.p95_latency_ms,
                p99_latency_ms: result.p99_latency_ms,
                throughput_tokens_per_sec: result.throughput_tokens_per_sec,
                memory_mb: result.memory_bytes.map(|b| b as f64 / (1024.0 * 1024.0)),
                gpu_memory_mb: None, // Would be set if using GPU
                version: env!("CARGO_PKG_VERSION").to_string(),
            };

            // Extract batch size and sequence length from parameters
            let batch_size =
                result.parameters.get("batch_size").and_then(|s| s.parse().ok()).unwrap_or(1);
            let seq_len =
                result.parameters.get("seq_len").and_then(|s| s.parse().ok()).unwrap_or(128);

            // Check if we already have a comparison for this benchmark
            if let Some(comparison) = self.results.iter_mut().find(|c| {
                c.benchmark_name == result.name
                    && c.batch_size == batch_size
                    && c.sequence_length == seq_len
            }) {
                comparison.framework_results.insert(Framework::TrustformeRS, framework_metrics);
            } else {
                let mut framework_results = HashMap::new();
                framework_results.insert(Framework::TrustformeRS, framework_metrics);

                self.results.push(ComparisonResult {
                    benchmark_name: result.name.clone(),
                    model_type: result.model_type.clone(),
                    batch_size,
                    sequence_length: seq_len,
                    framework_results,
                    relative_performance: HashMap::new(),
                });
            }
        }
    }

    /// Add PyTorch benchmark results
    pub fn add_pytorch_results(&mut self, pytorch_results: &[PytorchBenchmark]) {
        for result in pytorch_results {
            let framework_metrics = FrameworkMetrics {
                avg_latency_ms: result.avg_latency_ms,
                p95_latency_ms: result.p95_latency_ms,
                p99_latency_ms: result.p99_latency_ms,
                throughput_tokens_per_sec: result.throughput_tokens_per_sec,
                memory_mb: result.memory_mb,
                gpu_memory_mb: result.gpu_memory_mb,
                version: result.torch_version.clone(),
            };

            // Find or create comparison
            if let Some(comparison) = self.results.iter_mut().find(|c| {
                c.benchmark_name == result.name
                    && c.batch_size == result.batch_size
                    && c.sequence_length == result.sequence_length
            }) {
                comparison.framework_results.insert(Framework::PyTorch, framework_metrics);
            } else {
                let mut framework_results = HashMap::new();
                framework_results.insert(Framework::PyTorch, framework_metrics);

                self.results.push(ComparisonResult {
                    benchmark_name: result.name.clone(),
                    model_type: result.model_type.clone(),
                    batch_size: result.batch_size,
                    sequence_length: result.sequence_length,
                    framework_results,
                    relative_performance: HashMap::new(),
                });
            }
        }

        // Calculate relative performance
        self.calculate_relative_performance();
    }

    /// Add HuggingFace benchmark results
    pub fn add_huggingface_results(&mut self, hf_results: &[HuggingFaceBenchmark]) {
        for result in hf_results {
            let framework_metrics = FrameworkMetrics {
                avg_latency_ms: result.avg_latency_ms,
                p95_latency_ms: result.p95_latency_ms,
                p99_latency_ms: result.p99_latency_ms,
                throughput_tokens_per_sec: result.throughput_tokens_per_sec,
                memory_mb: result.memory_mb,
                gpu_memory_mb: result.gpu_memory_mb,
                version: result.transformers_version.clone(),
            };

            // Find or create comparison
            if let Some(comparison) = self.results.iter_mut().find(|c| {
                c.benchmark_name == result.name
                    && c.batch_size == result.batch_size
                    && c.sequence_length == result.sequence_length
            }) {
                comparison.framework_results.insert(Framework::HuggingFace, framework_metrics);
            } else {
                let mut framework_results = HashMap::new();
                framework_results.insert(Framework::HuggingFace, framework_metrics);

                self.results.push(ComparisonResult {
                    benchmark_name: result.name.clone(),
                    model_type: result.model_type.clone(),
                    batch_size: result.batch_size,
                    sequence_length: result.sequence_length,
                    framework_results,
                    relative_performance: HashMap::new(),
                });
            }
        }

        // Calculate relative performance
        self.calculate_relative_performance();
    }

    /// Calculate relative performance metrics
    fn calculate_relative_performance(&mut self) {
        for comparison in &mut self.results {
            if let Some(trustformers) = comparison.framework_results.get(&Framework::TrustformeRS) {
                // Compare with each other framework
                for (framework, metrics) in &comparison.framework_results {
                    if framework != &Framework::TrustformeRS {
                        let speedup = metrics.avg_latency_ms / trustformers.avg_latency_ms;
                        let throughput_ratio = trustformers.throughput_tokens_per_sec
                            / metrics.throughput_tokens_per_sec;
                        let latency_improvement =
                            (1.0 - trustformers.avg_latency_ms / metrics.avg_latency_ms) * 100.0;

                        let memory_efficiency = match (trustformers.memory_mb, metrics.memory_mb) {
                            (Some(tf_mem), Some(other_mem)) => Some(other_mem / tf_mem),
                            _ => None,
                        };

                        comparison.relative_performance.insert(
                            *framework,
                            RelativePerformance {
                                speedup,
                                throughput_ratio,
                                memory_efficiency,
                                latency_improvement_percent: latency_improvement,
                            },
                        );
                    }
                }
            }
        }
    }

    /// Generate comparison report
    pub fn generate_report(&self) -> ComparisonReport {
        let mut summary = ComparisonSummary::default();

        // Calculate average performance across all benchmarks
        for comparison in &self.results {
            for (framework, perf) in &comparison.relative_performance {
                summary
                    .avg_speedup
                    .entry(*framework)
                    .and_modify(|v| v.0 += perf.speedup)
                    .or_insert((perf.speedup, 1));
                summary.avg_speedup.entry(*framework).and_modify(|v| v.1 += 1);

                if perf.speedup > 1.0 {
                    *summary.benchmarks_faster.entry(*framework).or_insert(0) += 1;
                } else {
                    *summary.benchmarks_slower.entry(*framework).or_insert(0) += 1;
                }
            }
        }

        // Calculate averages
        for (_, (sum, count)) in summary.avg_speedup.iter_mut() {
            if *count > 0 {
                *sum /= *count as f64;
            }
        }

        ComparisonReport {
            comparisons: self.results.clone(),
            summary,
        }
    }

    /// Print comparison summary
    pub fn print_summary(&self) {
        println!("\n=== Performance Comparison Summary ===");

        for comparison in &self.results {
            println!(
                "\n{} (batch={}, seq_len={})",
                comparison.benchmark_name, comparison.batch_size, comparison.sequence_length
            );

            // Print metrics for each framework
            println!(
                "  {:15} {:>10} {:>10} {:>15} {:>10}",
                "Framework", "Avg (ms)", "P95 (ms)", "Throughput", "Memory (MB)"
            );
            println!("  {}", "-".repeat(65));

            for (framework, metrics) in &comparison.framework_results {
                println!(
                    "  {:15} {:>10.2} {:>10.2} {:>15.0} {:>10}",
                    framework.to_string(),
                    metrics.avg_latency_ms,
                    metrics.p95_latency_ms,
                    metrics.throughput_tokens_per_sec,
                    metrics.memory_mb.map(|m| format!("{:.1}", m)).unwrap_or("-".to_string()),
                );
            }

            // Print relative performance
            if !comparison.relative_performance.is_empty() {
                println!("\n  Relative Performance (TrustformeRS vs others):");
                for (framework, perf) in &comparison.relative_performance {
                    println!(
                        "    vs {}: {:.2}x speedup, {:.1}% latency improvement",
                        framework, perf.speedup, perf.latency_improvement_percent
                    );
                }
            }
        }
    }
}

/// PyTorch benchmark result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PytorchBenchmark {
    pub name: String,
    pub model_type: String,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_tokens_per_sec: f64,
    pub memory_mb: Option<f64>,
    pub gpu_memory_mb: Option<f64>,
    pub torch_version: String,
}

/// HuggingFace benchmark result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceBenchmark {
    pub name: String,
    pub model_type: String,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_tokens_per_sec: f64,
    pub memory_mb: Option<f64>,
    pub gpu_memory_mb: Option<f64>,
    pub transformers_version: String,
}

/// Comparison report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub comparisons: Vec<ComparisonResult>,
    pub summary: ComparisonSummary,
}

/// Comparison summary statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComparisonSummary {
    /// Average speedup by framework (value, count)
    pub avg_speedup: HashMap<Framework, (f64, usize)>,
    /// Number of benchmarks where TrustformeRS is faster
    pub benchmarks_faster: HashMap<Framework, usize>,
    /// Number of benchmarks where TrustformeRS is slower
    pub benchmarks_slower: HashMap<Framework, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison() {
        let mut comparison = ModelComparison::new();

        // Add TrustformeRS result
        let tf_result = BenchmarkResult {
            name: "bert_inference".to_string(),
            model_type: "BERT".to_string(),
            avg_latency_ms: 50.0,
            p50_latency_ms: 48.0,
            p95_latency_ms: 55.0,
            p99_latency_ms: 60.0,
            min_latency_ms: 45.0,
            max_latency_ms: 65.0,
            std_dev_ms: 5.0,
            throughput_tokens_per_sec: 2560.0, // 4 * 128 * 1000 / 50
            throughput_batches_per_sec: 20.0,
            memory_bytes: Some(100 * 1024 * 1024),
            peak_memory_bytes: Some(150 * 1024 * 1024),
            parameters: {
                let mut params = HashMap::new();
                params.insert("batch_size".to_string(), "4".to_string());
                params.insert("seq_len".to_string(), "128".to_string());
                params
            },
            raw_timings: vec![],
            timestamp: chrono::Utc::now(),
        };

        comparison.add_trustformers_results(&[tf_result]);

        // Add PyTorch result
        let pytorch_result = PytorchBenchmark {
            name: "bert_inference".to_string(),
            model_type: "BERT".to_string(),
            batch_size: 4,
            sequence_length: 128,
            avg_latency_ms: 60.0,
            p95_latency_ms: 65.0,
            p99_latency_ms: 70.0,
            throughput_tokens_per_sec: 2133.3,
            memory_mb: Some(120.0),
            gpu_memory_mb: None,
            torch_version: "2.0.0".to_string(),
        };

        comparison.add_pytorch_results(&[pytorch_result]);

        // Check comparison results
        assert_eq!(comparison.results.len(), 1);
        let result = &comparison.results[0];
        assert_eq!(result.framework_results.len(), 2);

        // Check relative performance
        let pytorch_perf = result.relative_performance.get(&Framework::PyTorch).unwrap();
        assert!(pytorch_perf.speedup > 1.0); // TrustformeRS should be faster
        assert!(pytorch_perf.latency_improvement_percent > 0.0);
    }
}
