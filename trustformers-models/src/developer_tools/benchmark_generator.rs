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

    /// The test configuration the generated benchmarks should use.
    ///
    /// Falls back to a single small shape when none was configured, so the
    /// generated file never silently ignores what the caller asked for.
    fn test_config(&self) -> TestConfig {
        let mut names: Vec<&String> = self.config.test_configs.keys().collect();
        names.sort();
        names
            .first()
            .and_then(|name| self.config.test_configs.get(*name))
            .cloned()
            .unwrap_or(TestConfig {
                batch_sizes: vec![1],
                sequence_lengths: vec![128],
                iterations: 100,
                warmup_iterations: 10,
            })
    }

    /// Render a Rust literal for a list of sizes.
    fn size_list(values: &[usize]) -> String {
        let rendered: Vec<String> = values.iter().map(usize::to_string).collect();
        format!("vec![{}]", rendered.join(", "))
    }

    /// Generate benchmark file header
    fn generate_header(&self) -> String {
        format!(
            "//! Performance Benchmarks for {name}\n\
             //!\n\
             //! This file is auto-generated. Do not edit manually.\n\
             //!\n\
             //! Every benchmark below runs a real forward pass of the model; a failed\n\
             //! model construction or forward pass aborts the benchmark instead of\n\
             //! silently measuring nothing.\n\n\
             use criterion::{{criterion_group, criterion_main, Criterion}};\n\
             use std::hint::black_box;\n\
             use std::time::Duration;\n\
             use super::{{{name}Config, {name}Model}};\n\
             use trustformers_core::tensor::Tensor;\n\n\
             /// Build the model under test, panicking with a clear message on failure.\n\
             fn build_model() -> {name}Model {{\n\
             \x20   match {name}Model::new({name}Config::default()) {{\n\
             \x20       Ok(model) => model,\n\
             \x20       Err(error) => panic!(\"failed to build {name}Model: {{error}}\"),\n\
             \x20   }}\n\
             }}\n\n\
             /// Build an input tensor, panicking with a clear message on failure.\n\
             fn build_input(shape: &[usize]) -> Tensor {{\n\
             \x20   match Tensor::randn(shape) {{\n\
             \x20       Ok(tensor) => tensor,\n\
             \x20       Err(error) => panic!(\"failed to build a {{shape:?}} input: {{error}}\"),\n\
             \x20   }}\n\
             }}\n\n\
             /// Run one forward pass, panicking if the model reports an error so a\n\
             /// broken model can never be recorded as fast.\n\
             fn forward(model: &{name}Model, input: &Tensor) -> Tensor {{\n\
             \x20   match model.forward(input) {{\n\
             \x20       Ok(output) => output,\n\
             \x20       Err(error) => panic!(\"forward pass failed: {{error}}\"),\n\
             \x20   }}\n\
             }}\n\n",
            name = self.config.model_name
        )
    }

    /// Generate latency benchmark
    fn generate_latency_benchmark(&self) -> String {
        let test_config = self.test_config();
        format!(
            "// ========== Latency Benchmarks ==========\n\n\
             fn bench_{lower}_latency(c: &mut Criterion) {{\n\
             \x20   let model = build_model();\n\
             \x20   let mut group = c.benchmark_group(\"{lower}_latency\");\n\
             \x20   group.warm_up_time(Duration::from_millis(500));\n\
             \x20   group.measurement_time(Duration::from_secs(3));\n\
             \x20   // Sample count from the generator's TestConfig (criterion's floor is 10).\n\
             \x20   group.sample_size({samples});\n\n\
             \x20   let batch_sizes = {batches};\n\
             \x20   let sequence_length = {seq_len};\n\n\
             \x20   for batch_size in batch_sizes {{\n\
             \x20       let input = build_input(&[batch_size, sequence_length]);\n\
             \x20       group.throughput(criterion::Throughput::Elements(\n\
             \x20           (batch_size * sequence_length) as u64,\n\
             \x20       ));\n\
             \x20       group.bench_with_input(\n\
             \x20           format!(\"batch_{{}}\", batch_size),\n\
             \x20           &batch_size,\n\
             \x20           |b, _| {{\n\
             \x20               b.iter(|| black_box(forward(&model, black_box(&input))));\n\
             \x20           }},\n\
             \x20       );\n\
             \x20   }}\n\n\
             \x20   group.finish();\n\
             }}\n\n",
            lower = self.config.model_name.to_lowercase(),
            batches = Self::size_list(&test_config.batch_sizes),
            seq_len = test_config.sequence_lengths.first().copied().unwrap_or(128),
            samples = test_config.iterations.max(10),
        )
    }

    /// Generate throughput benchmark
    fn generate_throughput_benchmark(&self) -> String {
        let test_config = self.test_config();
        let batch_size = test_config.batch_sizes.last().copied().unwrap_or(1);
        let sequence_length = test_config.sequence_lengths.last().copied().unwrap_or(128);
        format!(
            "// ========== Throughput Benchmarks ==========\n\n\
             fn bench_{lower}_throughput(c: &mut Criterion) {{\n\
             \x20   let model = build_model();\n\
             \x20   let mut group = c.benchmark_group(\"{lower}_throughput\");\n\
             \x20   group.warm_up_time(Duration::from_millis({warmup}));\n\
             \x20   group.measurement_time(Duration::from_secs(5));\n\
             \x20   group.sample_size({samples});\n\n\
             \x20   let batch_size = {batch_size};\n\
             \x20   let sequence_length = {sequence_length};\n\
             \x20   let input = build_input(&[batch_size, sequence_length]);\n\
             \x20   // Criterion divides the measured time by this element count, so the\n\
             \x20   // reported throughput is tokens per second of real work.\n\
             \x20   group.throughput(criterion::Throughput::Elements(\n\
             \x20       (batch_size * sequence_length) as u64,\n\
             \x20   ));\n\n\
             \x20   group.bench_function(\"tokens_per_second\", |b| {{\n\
             \x20       b.iter(|| black_box(forward(&model, black_box(&input))));\n\
             \x20   }});\n\n\
             \x20   group.finish();\n\
             }}\n\n",
            lower = self.config.model_name.to_lowercase(),
            batch_size = batch_size,
            sequence_length = sequence_length,
            samples = test_config.iterations.max(10),
            warmup = (test_config.warmup_iterations.max(1) * 100).min(5_000),
        )
    }

    /// Generate memory benchmark
    fn generate_memory_benchmark(&self) -> String {
        let test_config = self.test_config();
        format!(
            "// ========== Sequence-Length Scaling ==========\n\n\
             fn bench_{lower}_memory(c: &mut Criterion) {{\n\
             \x20   let model = build_model();\n\
             \x20   let mut group = c.benchmark_group(\"{lower}_memory\");\n\
             \x20   group.warm_up_time(Duration::from_millis(100));\n\
             \x20   group.measurement_time(Duration::from_secs(2));\n\n\
             \x20   let sequence_lengths = {seq_lens};\n\
             \x20   let batch_size = {batch_size};\n\n\
             \x20   for seq_len in sequence_lengths {{\n\
             \x20       let input = build_input(&[batch_size, seq_len]);\n\
             \x20       group.bench_with_input(\n\
             \x20           format!(\"seq_len_{{}}\", seq_len),\n\
             \x20           &seq_len,\n\
             \x20           |b, _| {{\n\
             \x20               b.iter(|| black_box(forward(&model, black_box(&input))));\n\
             \x20           }},\n\
             \x20       );\n\
             \x20   }}\n\n\
             \x20   group.finish();\n\
             }}\n\n",
            lower = self.config.model_name.to_lowercase(),
            seq_lens = Self::size_list(&test_config.sequence_lengths),
            batch_size = test_config.batch_sizes.first().copied().unwrap_or(1),
        )
    }

    /// Generate accuracy benchmark
    fn generate_accuracy_benchmark(&self) -> String {
        let test_config = self.test_config();
        format!(
            "// ========== Numerical Consistency ==========\n\n\
             fn bench_{lower}_accuracy(c: &mut Criterion) {{\n\
             \x20   let model = build_model();\n\
             \x20   let mut group = c.benchmark_group(\"{lower}_accuracy\");\n\
             \x20   group.warm_up_time(Duration::from_millis(200));\n\
             \x20   group.measurement_time(Duration::from_secs(2));\n\n\
             \x20   let input = build_input(&[{batch_size}, {seq_len}]);\n\
             \x20   // Reference output: every iteration must reproduce it exactly.\n\
             \x20   let reference = match forward(&model, &input).data() {{\n\
             \x20       Ok(values) => values,\n\
             \x20       Err(error) => panic!(\"failed to read the reference output: {{error}}\"),\n\
             \x20   }};\n\n\
             \x20   group.bench_function(\"deterministic_forward\", |b| {{\n\
             \x20       b.iter(|| {{\n\
             \x20           let output = forward(&model, black_box(&input));\n\
             \x20           let values = match output.data() {{\n\
             \x20               Ok(values) => values,\n\
             \x20               Err(error) => panic!(\"failed to read the output: {{error}}\"),\n\
             \x20           }};\n\
             \x20           assert_eq!(values.len(), reference.len());\n\
             \x20           for (actual, expected) in values.iter().zip(reference.iter()) {{\n\
             \x20               assert!((actual - expected).abs() < 1e-5);\n\
             \x20           }}\n\
             \x20           black_box(values);\n\
             \x20       }});\n\
             \x20   }});\n\n\
             \x20   group.finish();\n\
             }}\n\n",
            lower = self.config.model_name.to_lowercase(),
            batch_size = test_config.batch_sizes.first().copied().unwrap_or(1),
            seq_len = test_config.sequence_lengths.first().copied().unwrap_or(128),
        )
    }

    /// Generate scalability benchmark
    fn generate_scalability_benchmark(&self) -> String {
        let test_config = self.test_config();
        let mut cases = Vec::new();
        for batch in &test_config.batch_sizes {
            for seq_len in &test_config.sequence_lengths {
                cases.push(format!("({batch}, {seq_len})"));
            }
        }
        format!(
            "// ========== Scalability Benchmarks ==========\n\n\
             fn bench_{lower}_scalability(c: &mut Criterion) {{\n\
             \x20   let model = build_model();\n\
             \x20   let mut group = c.benchmark_group(\"{lower}_scalability\");\n\
             \x20   group.warm_up_time(Duration::from_millis(500));\n\
             \x20   group.measurement_time(Duration::from_secs(3));\n\n\
             \x20   let test_cases = vec![{cases}];\n\n\
             \x20   for (batch_size, seq_len) in test_cases {{\n\
             \x20       let input = build_input(&[batch_size, seq_len]);\n\
             \x20       group.throughput(criterion::Throughput::Elements(\n\
             \x20           (batch_size * seq_len) as u64,\n\
             \x20       ));\n\
             \x20       group.bench_with_input(\n\
             \x20           format!(\"batch_{{}}x{{}}\", batch_size, seq_len),\n\
             \x20           &(batch_size, seq_len),\n\
             \x20           |b, _| {{\n\
             \x20               b.iter(|| black_box(forward(&model, black_box(&input))));\n\
             \x20           }},\n\
             \x20       );\n\
             \x20   }}\n\n\
             \x20   group.finish();\n\
             }}\n\n",
            lower = self.config.model_name.to_lowercase(),
            cases = cases.join(", "),
        )
    }

    /// Generate comparative benchmark
    fn generate_comparative_benchmark(&self) -> String {
        let test_config = self.test_config();
        format!(
            "// ========== Comparative Benchmarks ==========\n\n\
             fn bench_{lower}_comparative(c: &mut Criterion) {{\n\
             \x20   let model = build_model();\n\
             \x20   let mut group = c.benchmark_group(\"{lower}_comparative\");\n\
             \x20   group.warm_up_time(Duration::from_millis(300));\n\
             \x20   group.measurement_time(Duration::from_secs(2));\n\n\
             \x20   let input = build_input(&[{batch_size}, {seq_len}]);\n\n\
             \x20   group.bench_function(\"baseline\", |b| {{\n\
             \x20       b.iter(|| black_box(forward(&model, black_box(&input))));\n\
             \x20   }});\n\n\
             \x20   // Add further configurations here to compare against this baseline.\n\n\
             \x20   group.finish();\n\
             }}\n\n",
            lower = self.config.model_name.to_lowercase(),
            batch_size = test_config.batch_sizes.last().copied().unwrap_or(1),
            seq_len = test_config.sequence_lengths.last().copied().unwrap_or(128),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn generator() -> BenchmarkGenerator {
        let mut test_configs = HashMap::new();
        test_configs.insert(
            "default".to_string(),
            TestConfig {
                batch_sizes: vec![2, 7],
                sequence_lengths: vec![33, 64],
                iterations: 5,
                warmup_iterations: 1,
            },
        );

        BenchmarkGenerator::new(BenchmarkGeneratorConfig {
            model_name: "Demo".to_string(),
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
        })
    }

    #[test]
    fn test_generated_benchmarks_use_the_configured_shapes() {
        let generator = generator();
        let code = generator.generate_latency_benchmark();

        assert!(
            code.contains("vec![2, 7]"),
            "the configured batch sizes must be used:\n{code}"
        );
        assert!(
            code.contains("let sequence_length = 33"),
            "the configured sequence length must be used:\n{code}"
        );
        assert!(
            !code.contains("vec![1, 4, 8, 16]"),
            "the generator must not fall back to hardcoded shapes:\n{code}"
        );
        assert!(
            code.contains("group.sample_size(10)"),
            "the configured iteration count must reach criterion:\n{code}"
        );

        let scalability = generator.generate_scalability_benchmark();
        for case in ["(2, 33)", "(2, 64)", "(7, 33)", "(7, 64)"] {
            assert!(
                scalability.contains(case),
                "missing case {case}:\n{scalability}"
            );
        }
    }

    #[test]
    fn test_generated_benchmarks_run_real_forward_passes() {
        let generator = generator();
        let mut code = generator.generate_header();
        code.push_str(&generator.generate_latency_benchmark());
        code.push_str(&generator.generate_throughput_benchmark());
        code.push_str(&generator.generate_accuracy_benchmark());

        // The measured region contains a real forward pass, not an empty closure.
        assert!(code.contains("model.forward(input)"));
        assert!(code.contains("b.iter(|| black_box(forward(&model, black_box(&input))))"));
        // Throughput is declared so criterion reports tokens/second of real work.
        assert!(code.contains("criterion::Throughput::Elements"));
    }

    #[test]
    fn test_generated_code_handles_fallible_constructors() {
        let code = generator().generate_header();

        // `Tensor::randn` and `Model::new` are fallible: the generated file must
        // unwrap them explicitly instead of assigning a `Result` to a `Tensor`.
        assert!(code.contains("fn build_input(shape: &[usize]) -> Tensor"));
        assert!(code.contains("match Tensor::randn(shape)"));
        assert!(code.contains("match DemoModel::new(DemoConfig::default())"));
        assert!(
            !code.contains("let input = Tensor::randn(&[batch_size, sequence_length]);"),
            "a raw Result must never be passed off as a Tensor"
        );
    }

    #[test]
    fn test_failed_forward_passes_are_not_silently_ignored() {
        let code = generator().generate_latency_benchmark();
        assert!(
            !code.contains("if let Ok(output) = model.forward"),
            "a failing forward pass must abort the benchmark, not measure an empty loop:\n{code}"
        );
    }

    #[test]
    fn test_generate_benchmarks_writes_a_file() {
        let generator = generator();
        let path = std::env::temp_dir().join("trustformers_generated_benchmark.rs");
        let _ = std::fs::remove_file(&path);

        generator.generate_benchmarks(&path).expect("benchmark generation");
        let written = std::fs::read_to_string(&path).expect("generated file");

        assert!(written.contains("fn bench_demo_latency"));
        assert!(written.contains("fn bench_demo_comparative"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_main_registration_lists_every_benchmark() {
        let main = generator().generate_main();
        for name in [
            "bench_demo_latency",
            "bench_demo_throughput",
            "bench_demo_memory",
            "bench_demo_accuracy",
            "bench_demo_scalability",
            "bench_demo_comparative",
        ] {
            assert!(main.contains(name), "{main}");
        }
    }
}
