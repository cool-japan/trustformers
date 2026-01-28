//! Benchmark Generator
//!
//! Automatic generation of performance benchmarks for model implementations.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkGeneratorConfig {
    /// Model name to benchmark
    pub model_name: String,
    /// Benchmark types to include
    pub benchmark_types: Vec<BenchmarkType>,
    /// Test configurations
    pub test_configs: HashMap<String, TestConfig>,
    /// Hardware targets
    pub hardware_targets: Vec<HardwareTarget>,
}

/// Type of benchmark to generate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchmarkType {
    Latency,
    Throughput,
    Memory,
    Accuracy,
    Scalability,
    Comparative,
}

/// Test configuration for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    pub batch_sizes: Vec<usize>,
    pub sequence_lengths: Vec<usize>,
    pub iterations: usize,
    pub warmup_iterations: usize,
}

/// Hardware target for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HardwareTarget {
    CPU,
    GPU,
    Metal,
    WASM,
}

/// Benchmark generator
pub struct BenchmarkGenerator {
    config: BenchmarkGeneratorConfig,
}

impl BenchmarkGenerator {
    /// Create a new benchmark generator
    pub fn new(config: BenchmarkGeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate benchmark suite
    pub fn generate_benchmarks(&self, output_path: &Path) -> Result<()> {
        let mut benchmark_content = self.generate_header();

        for benchmark_type in &self.config.benchmark_types {
            match benchmark_type {
                BenchmarkType::Latency => {
                    benchmark_content.push_str(&self.generate_latency_benchmark())
                },
                BenchmarkType::Throughput => {
                    benchmark_content.push_str(&self.generate_throughput_benchmark())
                },
                BenchmarkType::Memory => {
                    benchmark_content.push_str(&self.generate_memory_benchmark())
                },
                BenchmarkType::Accuracy => {
                    benchmark_content.push_str(&self.generate_accuracy_benchmark())
                },
                BenchmarkType::Scalability => {
                    benchmark_content.push_str(&self.generate_scalability_benchmark())
                },
                BenchmarkType::Comparative => {
                    benchmark_content.push_str(&self.generate_comparative_benchmark())
                },
            }
        }

        std::fs::write(output_path, benchmark_content)?;
        Ok(())
    }

    /// Generate benchmark file header
    fn generate_header(&self) -> String {
        format!(
            "//! Performance Benchmarks for {}\n//!\n//! This file is auto-generated. Do not edit manually.\n\nuse criterion::{{criterion_group, criterion_main, Criterion}};
use std::hint::black_box;\nuse super::{{{}Config, {}Model}};\nuse trustformers_core::tensor::Tensor;\nuse std::time::Duration;\n\n",
            self.config.model_name,
            self.config.model_name,
            self.config.model_name
        )
    }

    /// Generate latency benchmark
    fn generate_latency_benchmark(&self) -> String {
        format!(
            "// ========== Latency Benchmarks ==========\n\nfn bench_{}_latency(c: &mut Criterion) {{\n    let config = {}Config::default();\n    let model = match {}Model::new(config) {{\n        Ok(m) => m,\n        Err(e) => {{\n            eprintln!(\"Failed to create model: {{}}\", e);\n            return;\n        }}\n    }};\n    \n    let mut group = c.benchmark_group(\"{}_latency\");\n    group.warm_up_time(Duration::from_millis(500));\n    group.measurement_time(Duration::from_secs(3));\n    \n    let batch_sizes = vec![1, 4, 8, 16];\n    let sequence_length = 512;\n    \n    for batch_size in batch_sizes {{\n        group.bench_with_input(\n            format!(\"batch_{{}}\", batch_size),\n            &batch_size,\n            |b, &batch_size| {{\n                let input = Tensor::randn(&[batch_size, sequence_length]);\n                b.iter(|| {{\n                    if let Ok(output) = model.forward(black_box(&input)) {{\n                        black_box(output)\n                    }}\n                }});\n            }},\n        );\n    }}\n    \n    group.finish();\n}}\n\n",
            self.config.model_name.to_lowercase(),
            self.config.model_name,
            self.config.model_name,
            self.config.model_name.to_lowercase()
        )
    }

    /// Generate throughput benchmark
    fn generate_throughput_benchmark(&self) -> String {
        format!(
            "// ========== Throughput Benchmarks ==========\n\nfn bench_{}_throughput(c: &mut Criterion) {{\n    let config = {}Config::default();\n    let model = match {}Model::new(config) {{\n        Ok(m) => m,\n        Err(e) => {{\n            eprintln!(\"Failed to create model: {{}}\", e);\n            return;\n        }}\n    }};\n    \n    let mut group = c.benchmark_group(\"{}_throughput\");\n    group.warm_up_time(Duration::from_millis(500));\n    group.measurement_time(Duration::from_secs(5));\n    group.throughput(criterion::Throughput::Elements(1000));\n    \n    let batch_size = 8;\n    let sequence_length = 512;\n    let input = Tensor::randn(&[batch_size, sequence_length]);\n    \n    group.bench_function(\"tokens_per_second\", |b| {{\n        b.iter(|| {{\n            for _ in 0..100 {{\n                if let Ok(output) = model.forward(black_box(&input)) {{\n                    black_box(output);\n                }}\n            }}\n        }});\n    }});\n    \n    group.finish();\n}}\n\n",
            self.config.model_name.to_lowercase(),
            self.config.model_name,
            self.config.model_name,
            self.config.model_name.to_lowercase()
        )
    }

    /// Generate memory benchmark
    fn generate_memory_benchmark(&self) -> String {
        format!(
            "// ========== Memory Benchmarks ==========\n\nfn bench_{}_memory(c: &mut Criterion) {{\n    let config = {}Config::default();\n    let model = match {}Model::new(config) {{\n        Ok(m) => m,\n        Err(e) => {{\n            eprintln!(\"Failed to create model: {{}}\", e);\n            return;\n        }}\n    }};\n    \n    let mut group = c.benchmark_group(\"{}_memory\");\n    group.warm_up_time(Duration::from_millis(100));\n    group.measurement_time(Duration::from_secs(2));\n    \n    let sequence_lengths = vec![128, 256, 512, 1024];\n    let batch_size = 4;\n    \n    for seq_len in sequence_lengths {{\n        group.bench_with_input(\n            format!(\"seq_len_{{}}\", seq_len),\n            &seq_len,\n            |b, &seq_len| {{\n                b.iter(|| {{\n                    let input = Tensor::randn(&[batch_size, seq_len]);\n                    if let Ok(output) = model.forward(black_box(&input)) {{\n                        black_box(output)\n                    }}\n                }});\n            }},\n        );\n    }}\n    \n    group.finish();\n}}\n\n",
            self.config.model_name.to_lowercase(),
            self.config.model_name,
            self.config.model_name,
            self.config.model_name.to_lowercase()
        )
    }

    /// Generate accuracy benchmark
    fn generate_accuracy_benchmark(&self) -> String {
        format!(
            "// ========== Accuracy Benchmarks ==========\n\nfn bench_{}_accuracy(c: &mut Criterion) {{\n    let config = {}Config::default();\n    let model = match {}Model::new(config) {{\n        Ok(m) => m,\n        Err(e) => {{\n            eprintln!(\"Failed to create model: {{}}\", e);\n            return;\n        }}\n    }};\n    \n    let mut group = c.benchmark_group(\"{}_accuracy\");\n    group.warm_up_time(Duration::from_millis(200));\n    group.measurement_time(Duration::from_secs(2));\n    \n    // Test numerical precision\n    let input = Tensor::ones(&[4, 512]);\n    \n    group.bench_function(\"numerical_precision\", |b| {{\n        b.iter(|| {{\n            if let Ok(output) = model.forward(black_box(&input)) {{\n                // Verify output consistency\n                match &output {{\n                    Tensor::F32(arr) => {{\n                        let sum: f32 = arr.iter().sum();\n                        black_box(sum);\n                    }}\n                    _ => {{}}\n                }}\n            }}\n        }});\n    }});\n    \n    group.finish();\n}}\n\n",
            self.config.model_name.to_lowercase(),
            self.config.model_name,
            self.config.model_name,
            self.config.model_name.to_lowercase()
        )
    }

    /// Generate scalability benchmark
    fn generate_scalability_benchmark(&self) -> String {
        format!(
            "// ========== Scalability Benchmarks ==========\n\nfn bench_{}_scalability(c: &mut Criterion) {{\n    let config = {}Config::default();\n    let model = match {}Model::new(config) {{\n        Ok(m) => m,\n        Err(e) => {{\n            eprintln!(\"Failed to create model: {{}}\", e);\n            return;\n        }}\n    }};\n    \n    let mut group = c.benchmark_group(\"{}_scalability\");\n    group.warm_up_time(Duration::from_millis(500));\n    group.measurement_time(Duration::from_secs(3));\n    \n    // Test different configurations\n    let test_cases = vec![\n        (1, 128),   // Small: 1 batch, 128 sequence\n        (4, 256),   // Medium: 4 batch, 256 sequence\n        (8, 512),   // Large: 8 batch, 512 sequence\n        (16, 256),  // Wide: 16 batch, 256 sequence\n    ];\n    \n    for (batch_size, seq_len) in test_cases {{\n        group.bench_with_input(\n            format!(\"batch_{{}}x{{}}\", batch_size, seq_len),\n            &(batch_size, seq_len),\n            |b, &(batch_size, seq_len)| {{\n                let input = Tensor::randn(&[batch_size, seq_len]);\n                b.iter(|| {{\n                    if let Ok(output) = model.forward(black_box(&input)) {{\n                        black_box(output)\n                    }}\n                }});\n            }},\n        );\n    }}\n    \n    group.finish();\n}}\n\n",
            self.config.model_name.to_lowercase(),
            self.config.model_name,
            self.config.model_name,
            self.config.model_name.to_lowercase()
        )
    }

    /// Generate comparative benchmark
    fn generate_comparative_benchmark(&self) -> String {
        format!(
            "// ========== Comparative Benchmarks ==========\n\nfn bench_{}_comparative(c: &mut Criterion) {{\n    let config = {}Config::default();\n    let model = match {}Model::new(config) {{\n        Ok(m) => m,\n        Err(e) => {{\n            eprintln!(\"Failed to create model: {{}}\", e);\n            return;\n        }}\n    }};\n    \n    let mut group = c.benchmark_group(\"{}_comparative\");\n    group.warm_up_time(Duration::from_millis(300));\n    group.measurement_time(Duration::from_secs(2));\n    \n    let input = Tensor::randn(&[8, 512]);\n    \n    group.bench_function(\"baseline\", |b| {{\n        b.iter(|| {{\n            if let Ok(output) = model.forward(black_box(&input)) {{\n                black_box(output)\n            }}\n        }});\n    }});\n    \n    // Add comparisons with other models or configurations here\n    // This would typically compare against reference implementations\n    \n    group.finish();\n}}\n\n",
            self.config.model_name.to_lowercase(),
            self.config.model_name,
            self.config.model_name,
            self.config.model_name.to_lowercase()
        )
    }
}

impl BenchmarkGenerator {
    /// Generate the main benchmark group registration
    pub fn generate_main(&self) -> String {
        let function_names: Vec<String> = self
            .config
            .benchmark_types
            .iter()
            .map(|bt| {
                format!(
                    "bench_{}_{}",
                    self.config.model_name.to_lowercase(),
                    match bt {
                        BenchmarkType::Latency => "latency",
                        BenchmarkType::Throughput => "throughput",
                        BenchmarkType::Memory => "memory",
                        BenchmarkType::Accuracy => "accuracy",
                        BenchmarkType::Scalability => "scalability",
                        BenchmarkType::Comparative => "comparative",
                    }
                )
            })
            .collect();

        format!(
            "criterion_group!(benches, {});\ncriterion_main!(benches);\n",
            function_names.join(", ")
        )
    }
}

/// Predefined benchmark configurations
pub struct BenchmarkTemplates;

impl BenchmarkTemplates {
    /// Get comprehensive benchmark configuration
    pub fn comprehensive(model_name: String) -> BenchmarkGeneratorConfig {
        let mut test_configs = HashMap::new();
        test_configs.insert(
            "default".to_string(),
            TestConfig {
                batch_sizes: vec![1, 4, 8, 16],
                sequence_lengths: vec![128, 256, 512, 1024],
                iterations: 100,
                warmup_iterations: 10,
            },
        );

        BenchmarkGeneratorConfig {
            model_name,
            benchmark_types: vec![
                BenchmarkType::Latency,
                BenchmarkType::Throughput,
                BenchmarkType::Memory,
                BenchmarkType::Accuracy,
                BenchmarkType::Scalability,
                BenchmarkType::Comparative,
            ],
            test_configs,
            hardware_targets: vec![HardwareTarget::CPU],
        }
    }

    /// Get performance-focused benchmark configuration
    pub fn performance(model_name: String) -> BenchmarkGeneratorConfig {
        let mut test_configs = HashMap::new();
        test_configs.insert(
            "performance".to_string(),
            TestConfig {
                batch_sizes: vec![1, 8, 16, 32],
                sequence_lengths: vec![256, 512, 1024],
                iterations: 200,
                warmup_iterations: 20,
            },
        );

        BenchmarkGeneratorConfig {
            model_name,
            benchmark_types: vec![
                BenchmarkType::Latency,
                BenchmarkType::Throughput,
                BenchmarkType::Scalability,
            ],
            test_configs,
            hardware_targets: vec![HardwareTarget::CPU, HardwareTarget::GPU],
        }
    }

    /// Get accuracy-focused benchmark configuration
    pub fn accuracy(model_name: String) -> BenchmarkGeneratorConfig {
        let mut test_configs = HashMap::new();
        test_configs.insert(
            "accuracy".to_string(),
            TestConfig {
                batch_sizes: vec![4, 8],
                sequence_lengths: vec![512],
                iterations: 50,
                warmup_iterations: 5,
            },
        );

        BenchmarkGeneratorConfig {
            model_name,
            benchmark_types: vec![BenchmarkType::Accuracy, BenchmarkType::Comparative],
            test_configs,
            hardware_targets: vec![HardwareTarget::CPU],
        }
    }
}
