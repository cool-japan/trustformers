//! Simple demonstration of performance benchmarking capabilities
#![allow(unused_variables)]

use anyhow::Result;
use trustformers_core::{BenchmarkConfig, MetricsTracker, PerformanceProfiler};

fn main() -> Result<()> {
    println!("TrustformeRS Performance Benchmarking Demo");
    println!("==========================================\n");

    // 1. Basic benchmarking configuration
    println!("1. Configuring benchmarks...");
    let config = BenchmarkConfig {
        batch_sizes: vec![1, 4, 8],
        sequence_lengths: vec![128, 256],
        warmup_iterations: 3,
        num_iterations: 10,
        measure_memory: true,
        device: "cpu".to_string(),
        use_fp16: false,
        include_generation: false,
        max_generation_length: None,
    };

    println!("   Batch sizes: {:?}", config.batch_sizes);
    println!("   Sequence lengths: {:?}", config.sequence_lengths);
    println!(
        "   Iterations: {} (warmup: {})\n",
        config.num_iterations, config.warmup_iterations
    );

    // 2. Performance profiler example
    println!("2. Performance Profiling Example:");
    let profiler = PerformanceProfiler::new();
    profiler.enable();

    {
        let _guard = profiler.start_operation("data_loading");
        std::thread::sleep(std::time::Duration::from_millis(50));

        {
            let _guard = profiler.start_operation("preprocessing");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    {
        let _guard = profiler.start_operation("model_inference");
        std::thread::sleep(std::time::Duration::from_millis(100));

        {
            let _guard = profiler.start_operation("attention_computation");
            std::thread::sleep(std::time::Duration::from_millis(40));
        }

        {
            let _guard = profiler.start_operation("feedforward_computation");
            std::thread::sleep(std::time::Duration::from_millis(30));
        }
    }

    profiler.print_summary();

    // 3. Real-time metrics tracking
    println!("\n3. Real-time Metrics Tracking:");
    let mut tracker = MetricsTracker::new(10);

    // Simulate some inference operations
    for i in 0..5 {
        let batch_size = ((i % 3) + 1) * 2;
        let seq_length = 128;
        let latency = std::time::Duration::from_millis((20 + (i * 5)) as u64);

        tracker.record_inference(latency, batch_size, seq_length);
        println!(
            "   Recorded inference {}: batch_size={}, latency={:?}",
            i + 1,
            batch_size,
            latency
        );
    }

    let latency_metrics = tracker.latency_metrics();
    let throughput_metrics = tracker.throughput_metrics();

    println!("\n   Latency Summary:");
    println!("     Mean: {:.2}ms", latency_metrics.mean_ms);
    println!("     P50: {:.2}ms", latency_metrics.p50_ms);
    println!("     P95: {:.2}ms", latency_metrics.p95_ms);

    println!("\n   Throughput Summary:");
    println!(
        "     Tokens/sec: {:.0}",
        throughput_metrics.tokens_per_second
    );
    println!(
        "     Batches/sec: {:.2}",
        throughput_metrics.batches_per_second
    );

    println!("\n✅ Performance benchmarking infrastructure is ready!");
    println!("   - Comprehensive latency and throughput metrics");
    println!("   - Detailed performance profiling");
    println!("   - Memory usage tracking");
    println!("   - Framework comparison capabilities");
    println!("   - Continuous benchmarking with regression detection");

    Ok(())
}
