//! Example demonstrating comprehensive performance benchmarking
#![allow(unused_variables)]

use anyhow::Result;
use trustformers::{
    BertConfig, BertModel,
    pipeline::{Pipeline, TextClassificationPipeline},
    automodel::{AutoModel, AutoTokenizer},
};
use trustformers_core::{
    BenchmarkSuite, BenchmarkConfig,
    PerformanceProfiler, profile_fn,
    MetricsTracker,
    ModelComparison, Framework, PytorchBenchmark, HuggingFaceBenchmark,
    ContinuousBenchmark, ContinuousBenchmarkConfig,
    MemoryProfiler, PerformanceMemoryTracker,
};
use std::sync::Arc;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("TrustformeRS Performance Benchmark Example");
    println!("=========================================\n");

    // 1. Basic latency and throughput benchmarking
    run_basic_benchmarks()?;

    // 2. Detailed performance profiling
    run_performance_profiling()?;

    // 3. Real-time metrics tracking
    run_metrics_tracking()?;

    // 4. Memory profiling
    run_memory_profiling()?;

    // 5. Framework comparison
    run_framework_comparison()?;

    // 6. Continuous benchmarking
    run_continuous_benchmarking()?;

    Ok(())
}

/// Run basic latency and throughput benchmarks
fn run_basic_benchmarks() -> Result<()> {
    println!("1. Running Basic Benchmarks");
    println!("===========================\n");

    // Configure benchmarks
    let config = BenchmarkConfig {
        batch_sizes: vec![1, 4, 8, 16],
        sequence_lengths: vec![128, 256, 512],
        warmup_iterations: 5,
        num_iterations: 50,
        measure_memory: true,
        device: "cpu".to_string(),
        use_fp16: false,
        include_generation: false,
        max_generation_length: None,
    };

    let mut suite = BenchmarkSuite::new(config);

    // Create a BERT model
    let bert_config = BertConfig::default();
    let bert_model = BertModel::new(bert_config)?;

    // Run inference benchmarks
    suite.benchmark_inference(&bert_model, "BERT-base")?;

    // Print results
    suite.print_summary();

    // Export results
    suite.export_json("benchmark_results.json")?;
    suite.export_csv("benchmark_results.csv")?;

    println!("\nBenchmark results exported to benchmark_results.json and benchmark_results.csv\n");

    Ok(())
}

/// Run detailed performance profiling
fn run_performance_profiling() -> Result<()> {
    println!("2. Running Performance Profiling");
    println!("================================\n");

    let profiler = PerformanceProfiler::new();
    profiler.enable();

    // Profile model creation
    let model = profile_fn("model_creation", || {
        BertModel::new(BertConfig::default())
    })?;

    // Profile tokenization
    let tokenizer = profile_fn("tokenizer_creation", || {
        AutoTokenizer::from_pretrained("bert-base-uncased")
    })?;

    // Profile pipeline operations
    {
        let _guard = profiler.start_operation("pipeline_operations");

        // Create pipeline
        let pipeline = {
            let _guard = profiler.start_operation("pipeline_creation");
            TextClassificationPipeline::new(model, tokenizer)?
        };

        // Run inference
        let texts = vec![
            "This is a positive example.",
            "This is a negative example.",
            "This is another test.",
        ];

        for (i, text) in texts.iter().enumerate() {
            let _guard = profiler.start_operation(&format!("inference_{}", i));
            let _ = pipeline.predict(text);
        }
    }

    // Print profiling results
    profiler.print_summary();

    // Export flamegraph
    profiler.export_flamegraph("profile_flamegraph.txt")?;
    println!("\nFlamegraph data exported to profile_flamegraph.txt\n");

    Ok(())
}

/// Run real-time metrics tracking
fn run_metrics_tracking() -> Result<()> {
    println!("3. Running Real-time Metrics Tracking");
    println!("=====================================\n");

    let mut tracker = MetricsTracker::new(100); // Window size of 100

    // Create model and run some inferences
    let model = BertModel::new(BertConfig::default())?;

    // Simulate multiple inferences
    for i in 0..20 {
        let start = std::time::Instant::now();

        // Simulate inference with varying batch sizes
        let batch_size = ((i % 4) + 1) * 2; // 2, 4, 6, 8
        let seq_length = 128;

        // Dummy inference (in real scenario, this would be actual model inference)
        std::thread::sleep(std::time::Duration::from_millis(50 + (i % 20)));

        let latency = start.elapsed();
        tracker.record_inference(latency, batch_size, seq_length);

        // Record memory snapshot periodically
        if i % 5 == 0 {
            let memory_metrics = trustformers_core::MemoryMetrics::new(
                100 * 1024 * 1024 + i * 1024 * 1024, // Current
                150 * 1024 * 1024,                    // Peak
                90 * 1024 * 1024,                     // Allocated
                120 * 1024 * 1024,                    // Reserved
            );
            tracker.record_memory(memory_metrics);
        }
    }

    // Get metrics
    let latency_metrics = tracker.latency_metrics();
    let throughput_metrics = tracker.throughput_metrics();
    let memory_metrics = tracker.memory_metrics();

    // Print metrics
    println!("Latency Metrics:");
    println!("  Count: {}", latency_metrics.count);
    println!("  Mean: {:.2}ms", latency_metrics.mean_ms);
    println!("  P50: {:.2}ms", latency_metrics.p50_ms);
    println!("  P95: {:.2}ms", latency_metrics.p95_ms);
    println!("  P99: {:.2}ms", latency_metrics.p99_ms);

    println!("\nThroughput Metrics:");
    println!("  Tokens/sec: {:.0}", throughput_metrics.tokens_per_second);
    println!("  Batches/sec: {:.2}", throughput_metrics.batches_per_second);
    println!("  Avg batch size: {:.1}", throughput_metrics.avg_batch_size);

    if let Some(mem) = memory_metrics {
        println!("\nMemory Metrics:");
        println!("  Current: {:.1}MB", mem.current_mb());
        println!("  Peak: {:.1}MB", mem.peak_mb());
    }

    println!();

    Ok(())
}

/// Run memory profiling
fn run_memory_profiling() -> Result<()> {
    println!("4. Running Memory Profiling");
    println!("===========================\n");

    let profiler = Arc::new(MemoryProfiler::new());
    profiler.enable();

    let mut tracker = PerformanceMemoryTracker::new(profiler.clone());

    // Track model creation memory
    tracker.start_tracking("model_creation");
    let _model = BertModel::new(BertConfig::default())?;
    let model_delta = tracker.stop_tracking();

    if let Some(delta) = model_delta {
        println!("Model Creation Memory:");
        println!("  Allocated: {:.2}MB", delta.allocated_delta_mb());
        println!("  Peak: {:.2}MB", delta.peak_allocated_mb());
    }

    // Track pipeline creation memory
    tracker.start_tracking("pipeline_creation");
    let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;
    let pipeline_delta = tracker.stop_tracking();

    if let Some(delta) = pipeline_delta {
        println!("\nPipeline Creation Memory:");
        println!("  Allocated: {:.2}MB", delta.allocated_delta_mb());
        println!("  Peak: {:.2}MB", delta.peak_allocated_mb());
    }

    // Get overall memory stats
    let stats = profiler.get_stats();
    println!("\nOverall Memory Statistics:");
    println!("  Total allocated: {:.2}MB", stats.total_allocated as f64 / (1024.0 * 1024.0));
    println!("  Number of allocations: {}", stats.num_allocations);
    println!("  Average allocation size: {:.2}KB", stats.avg_allocation_size as f64 / 1024.0);

    println!();

    Ok(())
}

/// Run framework comparison
fn run_framework_comparison() -> Result<()> {
    println!("5. Running Framework Comparison");
    println!("===============================\n");

    let mut comparison = ModelComparison::new();

    // Run TrustformeRS benchmarks
    let config = BenchmarkConfig {
        batch_sizes: vec![1, 4, 8],
        sequence_lengths: vec![128, 256],
        warmup_iterations: 5,
        num_iterations: 20,
        measure_memory: true,
        device: "cpu".to_string(),
        use_fp16: false,
        include_generation: false,
        max_generation_length: None,
    };

    let mut suite = BenchmarkSuite::new(config);
    let model = BertModel::new(BertConfig::default())?;
    suite.benchmark_inference(&model, "BERT")?;

    comparison.add_trustformers_results(suite.results());

    // Simulate PyTorch results (in real scenario, these would come from actual PyTorch benchmarks)
    let pytorch_results = vec![
        PytorchBenchmark {
            name: "BERT_inference_b1_s128".to_string(),
            model_type: "BERT".to_string(),
            batch_size: 1,
            sequence_length: 128,
            avg_latency_ms: 25.0,
            p95_latency_ms: 28.0,
            p99_latency_ms: 30.0,
            throughput_tokens_per_sec: 5120.0,
            memory_mb: Some(450.0),
            gpu_memory_mb: None,
            torch_version: "2.0.0".to_string(),
        },
        PytorchBenchmark {
            name: "BERT_inference_b4_s128".to_string(),
            model_type: "BERT".to_string(),
            batch_size: 4,
            sequence_length: 128,
            avg_latency_ms: 80.0,
            p95_latency_ms: 85.0,
            p99_latency_ms: 90.0,
            throughput_tokens_per_sec: 6400.0,
            memory_mb: Some(600.0),
            gpu_memory_mb: None,
            torch_version: "2.0.0".to_string(),
        },
    ];

    comparison.add_pytorch_results(&pytorch_results);

    // Simulate HuggingFace results
    let hf_results = vec![
        HuggingFaceBenchmark {
            name: "BERT_inference_b1_s128".to_string(),
            model_type: "BERT".to_string(),
            batch_size: 1,
            sequence_length: 128,
            avg_latency_ms: 30.0,
            p95_latency_ms: 33.0,
            p99_latency_ms: 35.0,
            throughput_tokens_per_sec: 4266.0,
            memory_mb: Some(500.0),
            gpu_memory_mb: None,
            transformers_version: "4.30.0".to_string(),
        },
        HuggingFaceBenchmark {
            name: "BERT_inference_b4_s128".to_string(),
            model_type: "BERT".to_string(),
            batch_size: 4,
            sequence_length: 128,
            avg_latency_ms: 90.0,
            p95_latency_ms: 95.0,
            p99_latency_ms: 100.0,
            throughput_tokens_per_sec: 5688.0,
            memory_mb: Some(650.0),
            gpu_memory_mb: None,
            transformers_version: "4.30.0".to_string(),
        },
    ];

    comparison.add_huggingface_results(&hf_results);

    // Print comparison
    comparison.print_summary();

    // Generate report
    let report = comparison.generate_report();
    println!("\nComparison Summary:");
    for (framework, (avg_speedup, count)) in &report.summary.avg_speedup {
        if *count > 0 {
            println!("  vs {}: {:.2}x average speedup", framework, avg_speedup);
        }
    }

    println!();

    Ok(())
}

/// Run continuous benchmarking
fn run_continuous_benchmarking() -> Result<()> {
    println!("6. Running Continuous Benchmarking");
    println!("==================================\n");

    // Configure continuous benchmarking
    let config = ContinuousBenchmarkConfig {
        results_dir: PathBuf::from("continuous_benchmarks"),
        commit_sha: Some("abc123def456".to_string()),
        branch: Some("main".to_string()),
        build_config: "release".to_string(),
        regression_threshold: 5.0,
        num_runs: 3,
        confidence_level: 0.95,
    };

    let mut continuous = ContinuousBenchmark::new(config)?;

    // Run benchmarks
    let bench_config = BenchmarkConfig {
        batch_sizes: vec![1, 4],
        sequence_lengths: vec![128],
        warmup_iterations: 3,
        num_iterations: 10,
        measure_memory: true,
        device: "cpu".to_string(),
        use_fp16: false,
        include_generation: false,
        max_generation_length: None,
    };

    let mut suite = BenchmarkSuite::new(bench_config);
    let model = BertModel::new(BertConfig::default())?;
    suite.benchmark_inference(&model, "BERT")?;

    // Check for regressions
    let regressions = continuous.run_and_check(&mut suite)?;

    if regressions.is_empty() {
        println!("✅ No performance regressions detected!");
    } else {
        println!("⚠️  Found {} performance regressions:", regressions.len());
        for regression in &regressions {
            println!("  - {} {}: {:.1}% regression (was {:.2}, now {:.2})",
                regression.benchmark_name,
                regression.metric_name,
                regression.regression_percent,
                regression.previous_value,
                regression.current_value,
            );
        }
    }

    // Generate performance report
    let report = continuous.generate_report()?;
    println!("\nPerformance Trends:");
    for (benchmark, trend) in &report.trends {
        println!("  {}: latency trend={:.2}, throughput trend={:.2}",
            benchmark, trend.latency_trend, trend.throughput_trend);
    }

    println!();

    Ok(())
}

/// Simulated results for examples (in real usage, these would be replaced with actual benchmark runs)