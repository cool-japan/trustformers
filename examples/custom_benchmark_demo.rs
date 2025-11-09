use anyhow::Result;
#![allow(unused_variables)]
use std::time::{Duration, Instant};
use trustformers_core::performance::{
    BenchmarkBuilder, BenchmarkCategory, BenchmarkDSL, BenchmarkMetrics, BenchmarkRegistry,
    BenchmarkRunnerBuilder, CustomBenchmark, ExampleBenchmark, ReportFormat, Reporter, RunMode,
};

/// Custom inference benchmark
struct InferenceBenchmark {
    model_name: String,
    batch_sizes: Vec<usize>,
    current_batch_idx: usize,
}

impl InferenceBenchmark {
    fn new(model_name: String) -> Self {
        Self {
            model_name,
            batch_sizes: vec![1, 8, 16, 32],
            current_batch_idx: 0,
        }
    }
}

impl CustomBenchmark for InferenceBenchmark {
    fn name(&self) -> &str {
        "inference_benchmark"
    }

    fn description(&self) -> &str {
        "Benchmark model inference with various batch sizes"
    }

    fn tags(&self) -> Vec<String> {
        vec!["inference".to_string(), "latency".to_string()]
    }

    fn setup(&mut self) -> Result<()> {
        println!("Loading model: {}", self.model_name);
        // Simulate model loading
        std::thread::sleep(Duration::from_millis(100));
        Ok(())
    }

    fn run_iteration(&mut self) -> Result<trustformers_core::performance::BenchmarkIteration> {
        let batch_size = self.batch_sizes[self.current_batch_idx % self.batch_sizes.len()];
        self.current_batch_idx += 1;

        let start = Instant::now();

        // Simulate inference
        std::thread::sleep(Duration::from_millis(10 + batch_size as u64));

        let duration = start.elapsed();
        let tokens_per_second = (batch_size * 128) as f64 / duration.as_secs_f64();

        let mut metrics = BenchmarkMetrics::default();
        metrics.throughput = Some(batch_size as f64 / duration.as_secs_f64());
        metrics.model_metrics = Some(trustformers_core::performance::ModelMetrics {
            tokens_per_second: Some(tokens_per_second),
            flops_utilization: Some(0.75),
            batch_size: Some(batch_size),
            sequence_length: Some(128),
        });
        metrics.custom.insert("batch_size".to_string(), batch_size as f64);

        Ok(trustformers_core::performance::BenchmarkIteration {
            duration,
            metrics,
            validation_passed: Some(true),
            metadata: None,
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("TrustformeRS Custom Benchmark Framework Demo");
    println!("==========================================\n");

    // Example 1: Using the pre-built example benchmark
    println!("=== Example 1: Pre-built Benchmark ===");
    run_example_benchmark()?;

    // Example 2: Using custom benchmark
    println!("\n=== Example 2: Custom Inference Benchmark ===");
    run_custom_benchmark()?;

    // Example 3: Using benchmark builder
    println!("\n=== Example 3: Builder Pattern ===");
    run_builder_benchmark()?;

    // Example 4: Using DSL for specific benchmark types
    println!("\n=== Example 4: Domain-Specific Language ===");
    run_dsl_benchmarks()?;

    // Example 5: Using the registry
    println!("\n=== Example 5: Benchmark Registry ===");
    run_registry_example()?;

    Ok(())
}

fn run_example_benchmark() -> Result<()> {
    let benchmark = Box::new(ExampleBenchmark::new(
        "bert-base".to_string(),
        32,
        128,
    ));

    let results = BenchmarkRunnerBuilder::new()
        .mode(RunMode::Quick)
        .add_benchmark(benchmark)
        .run()?;

    let report = &results[0];
    println!("Results for: {}", report.name);
    println!("  Iterations: {}", report.iterations);
    println!("  Avg latency: {:.2}ms", report.summary.avg_latency_ms);

    Ok(())
}

fn run_custom_benchmark() -> Result<()> {
    let benchmark = Box::new(InferenceBenchmark::new("gpt2".to_string()));

    let results = BenchmarkRunnerBuilder::new()
        .mode(RunMode::Standard)
        .output_dir("benchmark_results")
        .save_raw_data(true)
        .with_progress(|update| {
            print!("\rProgress: {}/{}", update.current_iteration, update.total_iterations);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        })
        .add_benchmark(benchmark)
        .run()?;

    println!("\n");

    // Generate different report formats
    let report = &results[0];
    let text_report = Reporter::generate(report, ReportFormat::Text)?;
    println!("{}", text_report);

    Ok(())
}

fn run_builder_benchmark() -> Result<()> {
    let benchmark = BenchmarkBuilder::new("multi_stage_benchmark")
        .description("Benchmark with multiple stages")
        .tags(vec!["multi-stage".to_string()])
        .add_stage("preprocessing", || {
            let start = Instant::now();
            std::thread::sleep(Duration::from_millis(5));
            let duration = start.elapsed();

            let mut metrics = BenchmarkMetrics::default();
            metrics.custom.insert("stage".to_string(), 1.0);

            Ok(trustformers_core::performance::BenchmarkIteration {
                duration,
                metrics,
                validation_passed: Some(true),
                metadata: None,
            })
        })
        .add_weighted_stage("inference", 3.0, || {
            let start = Instant::now();
            std::thread::sleep(Duration::from_millis(20));
            let duration = start.elapsed();

            let mut metrics = BenchmarkMetrics::default();
            metrics.custom.insert("stage".to_string(), 2.0);
            metrics.throughput = Some(50.0);

            Ok(trustformers_core::performance::BenchmarkIteration {
                duration,
                metrics,
                validation_passed: Some(true),
                metadata: None,
            })
        })
        .add_stage("postprocessing", || {
            let start = Instant::now();
            std::thread::sleep(Duration::from_millis(3));
            let duration = start.elapsed();

            let mut metrics = BenchmarkMetrics::default();
            metrics.custom.insert("stage".to_string(), 3.0);

            Ok(trustformers_core::performance::BenchmarkIteration {
                duration,
                metrics,
                validation_passed: Some(true),
                metadata: None,
            })
        })
        .build()?;

    let results = BenchmarkRunnerBuilder::new()
        .mode(RunMode::Quick)
        .add_benchmark(Box::new(benchmark))
        .run()?;

    println!("Multi-stage benchmark completed");
    println!("  Total time: {:.2}s", results[0].total_duration.as_secs_f64());

    Ok(())
}

fn run_dsl_benchmarks() -> Result<()> {
    // Latency benchmark
    let latency_bench = BenchmarkDSL::latency_benchmark("tokenizer_latency")
        .measure("encode", || {
            let start = Instant::now();
            // Simulate tokenization
            std::thread::sleep(Duration::from_millis(2));
            Ok(start.elapsed())
        })
        .measure("decode", || {
            let start = Instant::now();
            // Simulate detokenization
            std::thread::sleep(Duration::from_millis(1));
            Ok(start.elapsed())
        })
        .build()?;

    // Throughput benchmark
    let throughput_bench = BenchmarkDSL::throughput_benchmark("model_throughput")
        .batch_size(64)
        .measure("forward_pass", 64, || {
            let start = Instant::now();
            // Simulate processing batch
            std::thread::sleep(Duration::from_millis(50));
            Ok(start.elapsed())
        })
        .build()?;

    // Memory benchmark
    let memory_bench = BenchmarkDSL::memory_benchmark("model_memory")
        .measure("load_weights", || {
            let start = Instant::now();
            // Simulate loading model weights
            std::thread::sleep(Duration::from_millis(100));
            let memory_bytes = 512 * 1024 * 1024; // 512 MB
            Ok((start.elapsed(), memory_bytes))
        })
        .build()?;

    // Run all DSL benchmarks
    let results = BenchmarkRunnerBuilder::new()
        .mode(RunMode::Quick)
        .parallel(3)
        .add_benchmark(Box::new(latency_bench))
        .add_benchmark(Box::new(throughput_bench))
        .add_benchmark(Box::new(memory_bench))
        .run()?;

    println!("DSL benchmarks completed:");
    for report in &results {
        println!("  {}: {:.2}ms avg latency", report.name, report.summary.avg_latency_ms);
    }

    Ok(())
}

fn run_registry_example() -> Result<()> {
    let registry = BenchmarkRegistry::new();

    // Register benchmarks
    registry
        .register_with_builder()
        .name("bert_inference")
        .description("BERT model inference benchmark")
        .category(BenchmarkCategory::Inference)
        .tags(vec!["bert".to_string(), "transformer".to_string()])
        .author("TrustformeRS Team")
        .version("1.0.0")
        .register(|| Box::new(ExampleBenchmark::new("bert".to_string(), 16, 512)))?;

    registry
        .register_with_builder()
        .name("memory_stress")
        .description("Memory allocation stress test")
        .category(BenchmarkCategory::Memory)
        .tags(vec!["memory".to_string(), "stress".to_string()])
        .register(|| Box::new(ExampleBenchmark::new("memory".to_string(), 1, 1024)))?;

    // List all benchmarks
    println!("Registered benchmarks:");
    for metadata in registry.list() {
        println!("  - {} ({})", metadata.name, metadata.category.clone());
        println!("    {}", metadata.description);
        println!("    Tags: {}", metadata.tags.join(", "));
    }

    // Search by category
    println!("\nInference benchmarks:");
    for metadata in registry.list_by_category(BenchmarkCategory::Inference) {
        println!("  - {}", metadata.name);
    }

    // Create and run a benchmark from registry
    let benchmark = registry.create("bert_inference")?;
    let results = BenchmarkRunnerBuilder::new()
        .mode(RunMode::Quick)
        .add_benchmark(benchmark)
        .run()?;

    println!("\nRegistry benchmark completed: {} iterations", results[0].iterations);

    Ok(())
}

// Helper to demonstrate comparison between benchmarks
fn compare_benchmarks() -> Result<()> {
    use trustformers_core::performance::{ComparisonRunner, ReportComparator};

    let baseline = Box::new(ExampleBenchmark::new("model-v1".to_string(), 32, 128));
    let candidates = vec![
        Box::new(ExampleBenchmark::new("model-v2".to_string(), 32, 128)) as Box<dyn CustomBenchmark>,
        Box::new(ExampleBenchmark::new("model-v3".to_string(), 32, 128)),
    ];

    let comparison = ComparisonRunner::new(baseline, candidates).run()?;

    println!("Baseline: {}", comparison.baseline.name);
    for candidate in &comparison.candidates {
        let result = ReportComparator::compare(&comparison.baseline, candidate);
        println!(
            "  vs {}: {:.1}% latency change",
            candidate.name,
            result.latency_change_percent
        );
    }

    Ok(())
}