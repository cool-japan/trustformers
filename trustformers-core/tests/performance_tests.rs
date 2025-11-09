//! Tests for performance benchmarking infrastructure

use anyhow::Result;
use std::collections::HashMap;
use std::time::Duration;
use trustformers_core::performance::benchmark::{ModelInput, ModelOutput};
use trustformers_core::*;

#[test]
fn test_benchmark_suite() -> Result<()> {
    // Create a simple test model
    struct TestModel {
        config: TestConfig,
    }

    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    struct TestConfig {
        hidden_size: usize,
    }

    impl Config for TestConfig {
        fn architecture(&self) -> &'static str {
            "test"
        }
    }

    impl Model for TestModel {
        type Config = TestConfig;
        type Input = ModelInput;
        type Output = ModelOutput;

        fn forward(&self, _input: Self::Input) -> Result<Self::Output, TrustformersError> {
            // Simulate some work
            std::thread::sleep(Duration::from_micros(100));
            Ok(ModelOutput {
                hidden_states: Some(Tensor::zeros(&[1, 128, 768])?),
                ..Default::default()
            })
        }

        fn load_pretrained(
            &mut self,
            _reader: &mut dyn std::io::Read,
        ) -> Result<(), TrustformersError> {
            Ok(())
        }

        fn get_config(&self) -> &Self::Config {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            // Return a dummy parameter count for testing
            768 * self.config.hidden_size
        }
    }

    let config = BenchmarkConfig {
        batch_sizes: vec![1, 2],
        sequence_lengths: vec![64, 128],
        warmup_iterations: 2,
        num_iterations: 5,
        measure_memory: false,
        device: "cpu".to_string(),
        use_fp16: false,
        include_generation: false,
        max_generation_length: None,
    };

    let mut suite = BenchmarkSuite::new(config);
    let model = TestModel {
        config: TestConfig { hidden_size: 768 },
    };

    suite.benchmark_inference(&model, "TestModel")?;

    let results = suite.results();
    assert_eq!(results.len(), 4); // 2 batch sizes * 2 sequence lengths

    for result in results {
        assert!(result.avg_latency_ms > 0.0);
        assert!(result.throughput_tokens_per_sec > 0.0);
        assert_eq!(result.model_type, "TestModel");
    }

    Ok(())
}

#[test]
fn test_metrics_tracker() {
    let mut tracker = MetricsTracker::new(10);

    // Record some measurements
    for i in 0..5 {
        let latency = Duration::from_millis(10 + i);
        tracker.record_inference(latency, 4, 128);
    }

    let latency_metrics = tracker.latency_metrics();
    assert_eq!(latency_metrics.count, 5);
    assert!(latency_metrics.mean_ms > 0.0);
    assert!(latency_metrics.p50_ms > 0.0);

    let throughput_metrics = tracker.throughput_metrics();
    assert_eq!(throughput_metrics.total_batches, 5);
    assert_eq!(throughput_metrics.total_tokens, 5 * 4 * 128);
    assert!(throughput_metrics.tokens_per_second > 0.0);
}

#[test]
fn test_performance_profiler() {
    let profiler = PerformanceProfiler::new();
    profiler.enable();

    // Profile some operations
    {
        let _guard = profiler.start_operation("outer_operation");
        std::thread::sleep(Duration::from_millis(5));

        {
            let _guard = profiler.start_operation("inner_operation");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    let results = profiler.get_results();
    assert!(results.contains_key("outer_operation"));

    let outer = &results["outer_operation"];
    assert_eq!(outer.call_count, 1);
    assert!(outer.total_time >= Duration::from_millis(15));
    assert!(!outer.children.is_empty());
}

#[test]
fn test_memory_profiler() {
    let profiler = MemoryProfiler::new();
    profiler.enable();

    // Record some allocations
    let id1 = profiler.record_allocation(1024);
    let _id2 = profiler.record_allocation(2048);

    let snapshot = profiler.take_snapshot();
    assert_eq!(snapshot.num_allocations, 2);
    assert_eq!(snapshot.allocated_bytes, 3072);

    // Deallocate one
    profiler.record_deallocation(id1);

    let snapshot = profiler.take_snapshot();
    assert_eq!(snapshot.num_allocations, 1);
    assert_eq!(snapshot.allocated_bytes, 2048);
}

#[test]
fn test_model_comparison() {
    let mut comparison = ModelComparison::new();

    // Create test TrustformeRS result
    let tf_result = BenchmarkResult {
        name: "test_benchmark".to_string(),
        model_type: "TestModel".to_string(),
        avg_latency_ms: 20.0,
        p50_latency_ms: 19.0,
        p95_latency_ms: 22.0,
        p99_latency_ms: 25.0,
        min_latency_ms: 18.0,
        max_latency_ms: 26.0,
        std_dev_ms: 2.0,
        throughput_tokens_per_sec: 6400.0,
        throughput_batches_per_sec: 50.0,
        memory_bytes: Some(50 * 1024 * 1024),
        peak_memory_bytes: Some(60 * 1024 * 1024),
        parameters: {
            let mut params = HashMap::new();
            params.insert("batch_size".to_string(), "1".to_string());
            params.insert("seq_len".to_string(), "128".to_string());
            params
        },
        raw_timings: vec![],
        timestamp: chrono::Utc::now(),
    };

    comparison.add_trustformers_results(&[tf_result]);

    // Create test PyTorch result
    let pytorch_result = PytorchBenchmark {
        name: "test_benchmark".to_string(),
        model_type: "TestModel".to_string(),
        batch_size: 1,
        sequence_length: 128,
        avg_latency_ms: 30.0,
        p95_latency_ms: 33.0,
        p99_latency_ms: 35.0,
        throughput_tokens_per_sec: 4266.0,
        memory_mb: Some(60.0),
        gpu_memory_mb: None,
        torch_version: "2.0.0".to_string(),
    };

    comparison.add_pytorch_results(&[pytorch_result]);

    let report = comparison.generate_report();
    assert_eq!(report.comparisons.len(), 1);

    let comp = &report.comparisons[0];
    assert_eq!(comp.framework_results.len(), 2);
    assert!(comp.relative_performance.contains_key(&Framework::PyTorch));

    let perf = &comp.relative_performance[&Framework::PyTorch];
    assert!(perf.speedup > 1.0); // TrustformeRS should be faster in this test
}

#[test]
fn test_continuous_benchmarking() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let config = ContinuousBenchmarkConfig {
        results_dir: temp_dir.path().to_path_buf(),
        commit_sha: Some("test123".to_string()),
        branch: Some("test".to_string()),
        build_config: "debug".to_string(),
        regression_threshold: 10.0,
        num_runs: 1,
        confidence_level: 0.95,
    };

    let continuous = ContinuousBenchmark::new(config)?;
    assert!(continuous.generate_report().is_ok());

    Ok(())
}

#[test]
fn test_latency_metrics_calculation() {
    let durations = vec![
        Duration::from_millis(10),
        Duration::from_millis(20),
        Duration::from_millis(15),
        Duration::from_millis(25),
        Duration::from_millis(30),
    ];

    let metrics = LatencyMetrics::from_durations(&durations);

    assert_eq!(metrics.count, 5);
    assert_eq!(metrics.mean_ms, 20.0);
    assert_eq!(metrics.min_ms, 10.0);
    assert_eq!(metrics.max_ms, 30.0);
    assert_eq!(metrics.median_ms, 20.0); // Middle value when sorted
    assert!(metrics.std_dev_ms > 0.0);
}

#[test]
fn test_throughput_metrics_calculation() {
    let metrics = ThroughputMetrics::calculate(
        10000, // total tokens
        100,   // total batches
        400,   // total samples
        Duration::from_secs(10),
    );

    assert_eq!(metrics.tokens_per_second, 1000.0);
    assert_eq!(metrics.batches_per_second, 10.0);
    assert_eq!(metrics.samples_per_second, 40.0);
    assert_eq!(metrics.avg_batch_size, 4.0);
    assert_eq!(metrics.avg_sequence_length, 25.0);
}
