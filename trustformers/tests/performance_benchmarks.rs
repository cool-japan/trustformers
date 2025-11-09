use rstest::*;
use std::time::{Duration, Instant};
use trustformers::config_management::ConfigurationManager;

/// Performance benchmark tests for TrustformeRS components
/// These tests ensure that the library meets performance requirements

/// Benchmark constants
#[allow(dead_code)]
const PERFORMANCE_TIMEOUT: Duration = Duration::from_secs(10);
#[allow(dead_code)]
const LARGE_BATCH_SIZE: usize = 100;
#[allow(dead_code)]
const MEDIUM_BATCH_SIZE: usize = 32;
#[allow(dead_code)]
const SMALL_BATCH_SIZE: usize = 8;

#[rstest]
fn benchmark_config_validation_performance() {
    let config_manager = ConfigurationManager::new();

    let config = serde_json::json!({
        "model_type": "bert",
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "vocab_size": 30522,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "layer_norm_eps": 1e-12,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1
    });

    // Warm up
    for _ in 0..10 {
        let _ = config_manager.validate_config("model", &config);
    }

    // Benchmark validation speed
    let iterations = 1000;
    let start = Instant::now();

    for _ in 0..iterations {
        let result = config_manager.validate_config("model", &config);
        assert!(result.is_valid, "Config validation should succeed");
    }

    let duration = start.elapsed();
    let per_validation = duration / iterations;

    println!(
        "Config validation: {} validations in {:?} ({:?} per validation)",
        iterations, duration, per_validation
    );

    // Should validate at least 1000 configs per second
    assert!(
        per_validation < Duration::from_micros(1000),
        "Config validation too slow: {:?} per validation",
        per_validation
    );
}

#[rstest]
fn benchmark_error_creation_performance() {
    use trustformers::error::TrustformersError;

    let iterations = 10000;
    let start = Instant::now();

    for i in 0..iterations {
        let _error = TrustformersError::pipeline(&format!("Benchmark error {}", i), "benchmark");
    }

    let duration = start.elapsed();
    let per_error = duration / iterations;

    println!(
        "Error creation: {} errors in {:?} ({:?} per error)",
        iterations, duration, per_error
    );

    // Should create at least 100,000 errors per second
    assert!(
        per_error < Duration::from_nanos(10000),
        "Error creation too slow: {:?} per error",
        per_error
    );
}

#[rstest]
fn benchmark_profiler_overhead() {
    let profiler = trustformers::profiler::get_global_profiler();
    let session_id = profiler.start_session("benchmark_session").unwrap();

    let iterations = 1000;

    // Benchmark without profiling
    let start = Instant::now();
    for i in 0..iterations {
        // Simulate some work
        let _result = format!("operation_{}", i);
        std::hint::black_box(_result);
    }
    let baseline_duration = start.elapsed();

    // Benchmark with profiling
    let start = Instant::now();
    for i in 0..iterations {
        let _result = profiler.profile_function_lightweight(&format!("operation_{}", i), || {});
        // Simulate some work
        let _result = format!("operation_{}", i);
        std::hint::black_box(_result);
    }
    let profiled_duration = start.elapsed();

    let overhead = profiled_duration.saturating_sub(baseline_duration);
    let overhead_per_op = overhead / iterations;

    println!(
        "Profiler overhead: {:?} total, {:?} per operation",
        overhead, overhead_per_op
    );

    // Profiler overhead should be minimal
    assert!(
        overhead_per_op < Duration::from_micros(10),
        "Profiler overhead too high: {:?} per operation",
        overhead_per_op
    );

    profiler.end_session(&session_id).unwrap();
}

#[rstest]
fn benchmark_caching_performance() {
    use trustformers::pipeline::advanced_caching::{AdvancedCache, CacheConfig};

    let config = CacheConfig {
        max_entries: 10000,
        max_memory_bytes: 100 * 1024 * 1024, // 100MB in bytes
        ttl_seconds: 3600,
        cleanup_interval_seconds: 300,
        lru_eviction_threshold: 0.8,
        smart_eviction_threshold: 0.9,
        enable_hit_rate_tracking: true,
        enable_memory_pressure_monitoring: true,
        enable_access_pattern_analysis: true,
    };

    let cache = AdvancedCache::new(config);

    // Benchmark cache insertions
    let iterations = 1000;
    let start = Instant::now();

    for i in 0..iterations {
        let key = format!("key_{}", i);
        let value = vec![i as u8; 64]; // 64 bytes per entry
        let value_len = value.len() as u64;
        let result = cache.insert(
            key.clone(),
            value,
            value_len,
            trustformers::pipeline::advanced_caching::CachePriority::Normal,
            std::collections::HashSet::new(),
            None,
        );
        assert!(result.is_ok(), "Cache insertion should succeed");
    }

    let insert_duration = start.elapsed();
    let per_insert = insert_duration / iterations;

    println!(
        "Cache insertions: {} in {:?} ({:?} per insert)",
        iterations, insert_duration, per_insert
    );

    // Benchmark cache lookups
    let start = Instant::now();

    for i in 0..iterations {
        let key = format!("key_{}", i);
        let result = cache.get(&key);
        assert!(result.is_some(), "Cache lookup should succeed");
    }

    let lookup_duration = start.elapsed();
    let per_lookup = lookup_duration / iterations;

    println!(
        "Cache lookups: {} in {:?} ({:?} per lookup)",
        iterations, lookup_duration, per_lookup
    );

    // Cache operations should be fast
    assert!(
        per_insert < Duration::from_micros(100),
        "Cache insert too slow: {:?}",
        per_insert
    );
    assert!(
        per_lookup < Duration::from_micros(50),
        "Cache lookup too slow: {:?}",
        per_lookup
    );
}

#[rstest]
fn benchmark_dynamic_batching_performance() {
    use trustformers::pipeline::dynamic_batching::{DynamicBatchConfig, DynamicBatchManager};

    let config = DynamicBatchConfig {
        initial_batch_size: 8,
        min_batch_size: 1,
        max_batch_size: 64,
        target_latency_ms: 100,
        max_wait_time_ms: 50,
        throughput_threshold: 10.0,
        performance_window_size: 10,
        adjustment_factor: 1.2,
    };

    let batch_manager: DynamicBatchManager<String> = DynamicBatchManager::new(config);

    // Benchmark batch size calculations
    let iterations = 1000;
    let start = Instant::now();

    for i in 0..iterations {
        let _throughput = 100.0 + (i as f64 * 0.1);
        let _latency = 50.0 + (i as f64 * 0.05);
        let _stats = batch_manager.get_performance_stats();
    }

    let duration = start.elapsed();
    let per_calculation = duration / iterations;

    println!(
        "Batch size calculations: {} in {:?} ({:?} per calculation)",
        iterations, duration, per_calculation
    );

    // Batch size calculation should be very fast
    assert!(
        per_calculation < Duration::from_micros(10),
        "Batch size calculation too slow: {:?}",
        per_calculation
    );
}

#[rstest]
fn benchmark_adaptive_batching_performance() {
    use trustformers::pipeline::adaptive_batching::{AdaptiveBatchConfig, AdaptiveBatchManager};

    let config = AdaptiveBatchConfig {
        min_batch_size: 1,
        max_batch_size: 64,
        samples_per_size: 10,
        warmup_iterations: 3,
        target_latency_percentile: 95.0,
        target_latency_ms: 100.0,
        throughput_weight: 0.6,
        latency_weight: 0.4,
        memory_weight: 0.0,
        reevaluation_interval_secs: 300,
    };

    let batch_manager = AdaptiveBatchManager::new(config);

    // Benchmark performance metric recording
    let iterations = 1000;
    let start = Instant::now();

    for i in 0..iterations {
        let sample = trustformers::pipeline::adaptive_batching::PerformanceSample {
            batch_size: 8,
            latency_ms: 100.0 + (i as f64 * 0.1),
            throughput_rps: 80.0 + (i as f64 * 0.2),
            memory_usage_mb: 512.0,
            gpu_memory_mb: 1024.0,
            cpu_utilization: 0.60,
            gpu_utilization: 0.75,
            timestamp: std::time::SystemTime::now(),
        };
        let _ = batch_manager.record_sample(sample);
    }

    let record_duration = start.elapsed();
    let per_record = record_duration / iterations;

    // Benchmark batch size suggestions
    let start = Instant::now();

    for _ in 0..iterations {
        let _suggestion = batch_manager.get_optimal_batch_size();
    }

    let suggest_duration = start.elapsed();
    let per_suggest = suggest_duration / iterations;

    println!(
        "Performance recording: {} in {:?} ({:?} per record)",
        iterations, record_duration, per_record
    );
    println!(
        "Batch suggestions: {} in {:?} ({:?} per suggestion)",
        iterations, suggest_duration, per_suggest
    );

    // Adaptive batching operations should be fast
    assert!(
        per_record < Duration::from_micros(50),
        "Performance recording too slow: {:?}",
        per_record
    );
    assert!(
        per_suggest < Duration::from_micros(100),
        "Batch suggestion too slow: {:?}",
        per_suggest
    );
}

// #[rstest]
// fn benchmark_processor_modality_detection() {
//     use trustformers::processor::{AutoProcessor, ProcessorInput};
//
//     let processor = AutoProcessor::new();
//     let inputs = vec![
//         ProcessorInput::Text("Hello world".to_string()),
//         ProcessorInput::Text("Another text input for testing".to_string()),
//         ProcessorInput::Text("Short text".to_string()),
//         ProcessorInput::Text(
//             "This is a longer text input that should still be processed quickly".to_string(),
//         ),
//     ];
//
//     // Warm up
//     for input in &inputs {
//         let _ = processor.detect_modality(input);
//     }
//
//     // Benchmark modality detection
//     let iterations = 1000;
//     let start = Instant::now();
//
//     for _ in 0..iterations {
//         for input in &inputs {
//             let result = processor.detect_modality(input);
//             assert!(result.is_ok(), "Modality detection should succeed");
//         }
//     }
//
//     let duration = start.elapsed();
//     let total_detections = iterations * inputs.len();
//     let per_detection = duration / total_detections as u32;
//
//     println!(
//         "Modality detection: {} detections in {:?} ({:?} per detection)",
//         total_detections, duration, per_detection
//     );
//
//     // Modality detection should be very fast
//     assert!(
//         per_detection < Duration::from_micros(10),
//         "Modality detection too slow: {:?}",
//         per_detection
//     );
// }

#[rstest]
fn benchmark_hub_options_creation() {
    use trustformers::hub::HubOptions;

    let iterations = 10000;
    let start = Instant::now();

    for _ in 0..iterations {
        let _options = HubOptions::default();
    }

    let duration = start.elapsed();
    let per_creation = duration / iterations;

    println!(
        "HubOptions creation: {} in {:?} ({:?} per creation)",
        iterations, duration, per_creation
    );

    // Options creation should be very fast
    assert!(
        per_creation < Duration::from_nanos(1000),
        "HubOptions creation too slow: {:?}",
        per_creation
    );
}

#[rstest]
fn benchmark_pipeline_config_creation() {
    use trustformers::pipeline::text_classification::TextClassificationConfig;

    let iterations = 10000;
    let start = Instant::now();

    for i in 0..iterations {
        let _config = TextClassificationConfig {
            max_length: 512,
            labels: vec!["POSITIVE".to_string(), "NEGATIVE".to_string()],
            return_all_scores: (i % 2) == 0,
        };
    }

    let duration = start.elapsed();
    let per_creation = duration / iterations;

    println!(
        "Pipeline config creation: {} in {:?} ({:?} per creation)",
        iterations, duration, per_creation
    );

    // Config creation should be very fast
    assert!(
        per_creation < Duration::from_nanos(1000),
        "Pipeline config creation too slow: {:?}",
        per_creation
    );
}

#[rstest]
#[tokio::test]
async fn benchmark_async_operations() {
    use tokio::task::yield_now;

    // Benchmark simple async operations without timeout overhead
    let iterations = 1000;
    let start = Instant::now();

    for _ in 0..iterations {
        // Minimal async work without expensive timeout
        yield_now().await;
        std::hint::black_box(42);
    }

    let duration = start.elapsed();
    let per_operation = duration / iterations;

    println!(
        "Async operations: {} in {:?} ({:?} per operation)",
        iterations, duration, per_operation
    );

    // Async overhead should be minimal
    assert!(
        per_operation < Duration::from_micros(100),
        "Async operation overhead too high: {:?}",
        per_operation
    );
}

#[rstest]
fn benchmark_memory_allocation_patterns() {
    // Test different allocation patterns that might be used in pipelines

    // Small frequent allocations
    let iterations = 10000;
    let start = Instant::now();

    for i in 0..iterations {
        let _vec: Vec<u8> = vec![i as u8; 64];
        std::hint::black_box(_vec);
    }

    let small_alloc_duration = start.elapsed();

    // Medium allocations
    let iterations = 1000;
    let start = Instant::now();

    for i in 0..iterations {
        let _vec: Vec<u32> = vec![i; 1024];
        std::hint::black_box(_vec);
    }

    let medium_alloc_duration = start.elapsed();

    // Large allocations
    let iterations = 100;
    let start = Instant::now();

    for i in 0..iterations {
        let _vec: Vec<f32> = vec![i as f32; 1024 * 1024];
        std::hint::black_box(_vec);
    }

    let large_alloc_duration = start.elapsed();

    println!(
        "Small allocations (64B): {} in {:?}",
        10000, small_alloc_duration
    );
    println!(
        "Medium allocations (4KB): {} in {:?}",
        1000, medium_alloc_duration
    );
    println!(
        "Large allocations (4MB): {} in {:?}",
        100, large_alloc_duration
    );

    // Allocations should complete in reasonable time
    assert!(small_alloc_duration < Duration::from_millis(100));
    assert!(medium_alloc_duration < Duration::from_millis(100));
    assert!(large_alloc_duration < Duration::from_millis(1000));
}

#[rstest]
fn benchmark_string_operations() {
    // Test string operations commonly used in text processing

    let test_text = "This is a test sentence with various words and punctuation marks.";
    let iterations = 10000;

    // String cloning
    let start = Instant::now();
    for _ in 0..iterations {
        let _cloned = test_text.to_string();
        std::hint::black_box(_cloned);
    }
    let clone_duration = start.elapsed();

    // String splitting
    let start = Instant::now();
    for _ in 0..iterations {
        let _words: Vec<&str> = test_text.split_whitespace().collect();
        std::hint::black_box(_words);
    }
    let split_duration = start.elapsed();

    // String formatting
    let start = Instant::now();
    for i in 0..iterations {
        let _formatted = format!("{}: {}", i, test_text);
        std::hint::black_box(_formatted);
    }
    let format_duration = start.elapsed();

    println!("String cloning: {} in {:?}", iterations, clone_duration);
    println!("String splitting: {} in {:?}", iterations, split_duration);
    println!("String formatting: {} in {:?}", iterations, format_duration);

    // String operations should be reasonably fast
    assert!(clone_duration < Duration::from_millis(10));
    assert!(split_duration < Duration::from_millis(50));
    assert!(format_duration < Duration::from_millis(100));
}

#[rstest]
fn benchmark_json_operations() {
    use serde_json::{json, Value};

    let test_config = json!({
        "model_type": "bert",
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "vocab_size": 30522,
        "max_position_embeddings": 512,
        "nested": {
            "layer_norm_eps": 1e-12,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1
        }
    });

    let iterations = 1000;

    // JSON serialization
    let start = Instant::now();
    for _ in 0..iterations {
        let _serialized = serde_json::to_string(&test_config).unwrap();
        std::hint::black_box(_serialized);
    }
    let serialize_duration = start.elapsed();

    // JSON deserialization
    let json_string = serde_json::to_string(&test_config).unwrap();
    let start = Instant::now();
    for _ in 0..iterations {
        let _deserialized: Value = serde_json::from_str(&json_string).unwrap();
        std::hint::black_box(_deserialized);
    }
    let deserialize_duration = start.elapsed();

    println!(
        "JSON serialization: {} in {:?}",
        iterations, serialize_duration
    );
    println!(
        "JSON deserialization: {} in {:?}",
        iterations, deserialize_duration
    );

    // JSON operations should be reasonably fast
    assert!(serialize_duration < Duration::from_millis(100));
    assert!(deserialize_duration < Duration::from_millis(200));
}

/// Performance regression tests
/// These tests help detect performance regressions in critical paths
mod regression_tests {
    use super::*;

    #[rstest]
    fn test_no_performance_regression_config_validation() {
        let config_manager = ConfigurationManager::new();
        let config = serde_json::json!({
            "model_type": "bert",
            "hidden_size": 768
        });

        // Baseline: should validate 1000 configs in under 100ms
        let iterations = 1000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = config_manager.validate_config("model", &config);
        }

        let duration = start.elapsed();
        assert!(
            duration < Duration::from_millis(100),
            "Performance regression detected in config validation: {:?}",
            duration
        );
    }

    #[rstest]
    fn test_no_performance_regression_error_handling() {
        use trustformers::error::TrustformersError;

        // Baseline: should create 10000 errors in under 50ms
        let iterations = 10000;
        let start = Instant::now();

        for i in 0..iterations {
            let _error = TrustformersError::pipeline(&format!("Error {}", i), "regression_test");
        }

        let duration = start.elapsed();
        assert!(
            duration < Duration::from_millis(50),
            "Performance regression detected in error creation: {:?}",
            duration
        );
    }

    #[rstest]
    fn test_no_performance_regression_profiler() {
        let profiler = trustformers::profiler::get_global_profiler();
        let session_id = profiler.start_session("regression_test").unwrap();

        // Baseline: profiler overhead should be minimal
        let iterations = 1000;

        let start = Instant::now();
        for _ in 0..iterations {
            // Simulate work without profiling
            std::hint::black_box(42);
        }
        let baseline = start.elapsed();

        let start = Instant::now();
        for _ in 0..iterations {
            let _result = profiler.profile_function_lightweight("test_op", || {});
            // Simulate work with profiling
            std::hint::black_box(42);
        }
        let with_profiling = start.elapsed();

        let overhead = with_profiling.saturating_sub(baseline);
        let overhead_per_op = overhead / iterations;

        // Overhead should be less than 10 microseconds per operation
        assert!(
            overhead_per_op < Duration::from_micros(10),
            "Performance regression detected in profiler overhead: {:?}",
            overhead_per_op
        );

        profiler.end_session(&session_id).unwrap();
    }
}

/// Load testing for concurrent operations
mod load_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[rstest]
    fn test_concurrent_config_validation() {
        let config_manager = Arc::new(std::sync::Mutex::new(ConfigurationManager::new()));
        let config = Arc::new(serde_json::json!({
            "model_type": "bert",
            "hidden_size": 768
        }));

        let num_threads = 4;
        let iterations_per_thread = 250;

        let start = Instant::now();

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let config_manager: Arc<std::sync::Mutex<ConfigurationManager>> =
                    Arc::clone(&config_manager);
                let config = Arc::clone(&config);

                thread::spawn(move || {
                    for _ in 0..iterations_per_thread {
                        let manager = config_manager.lock().unwrap();
                        let _ = manager.validate_config("model", &config);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let duration = start.elapsed();
        let total_operations = num_threads * iterations_per_thread;
        let per_operation = duration / total_operations as u32;

        println!("Concurrent config validation: {} operations across {} threads in {:?} ({:?} per operation)",
                 total_operations, num_threads, duration, per_operation);

        // Should handle concurrent access efficiently
        assert!(
            per_operation < Duration::from_millis(1),
            "Concurrent config validation too slow: {:?}",
            per_operation
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_concurrent_async_operations() {
        use tokio::task::JoinSet;

        let mut set = JoinSet::new();
        let num_tasks = 100;
        let operations_per_task = 10;

        let start = Instant::now();

        for task_id in 0..num_tasks {
            set.spawn(async move {
                for op_id in 0..operations_per_task {
                    // Simulate async work
                    tokio::time::sleep(Duration::from_nanos(1)).await;
                    std::hint::black_box(task_id * operations_per_task + op_id);
                }
                task_id
            });
        }

        let mut results = Vec::new();
        while let Some(res) = set.join_next().await {
            results.push(res.unwrap());
        }

        let duration = start.elapsed();
        let total_operations = num_tasks * operations_per_task;

        println!(
            "Concurrent async operations: {} operations across {} tasks in {:?}",
            total_operations, num_tasks, duration
        );

        assert_eq!(results.len(), num_tasks);
        assert!(
            duration < Duration::from_millis(100),
            "Concurrent async operations too slow: {:?}",
            duration
        );
    }
}
