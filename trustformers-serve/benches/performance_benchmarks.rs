//! Performance Benchmarks for TrustformeRS Serve
//!
//! Comprehensive benchmarks covering inference performance, batching efficiency,
//! caching performance, streaming throughput, and resource utilization.

// Allow unused code for benchmarks
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::random::*; // Replaces rand - SciRS2 Integration Policy
use std::time::Duration;
use tokio::runtime::Runtime;
use trustformers_serve::batching::aggregator::RequestInput;
use trustformers_serve::batching::config::Priority;
use trustformers_serve::{
    batching::{BatchingConfig, DynamicBatchingService, Request, RequestId},
    caching::{CacheConfig, CacheKey, CacheResult, CachingService},
    metrics::MetricsService,
    rate_limit::{RateLimitConfig, RateLimitService},
    streaming::{StreamData, StreamType, StreamingConfig, StreamingService},
    validation::{ValidationConfig, ValidationService},
    Device, ServerConfig, TrustformerServer,
};

/// Create a benchmark-optimized server configuration
fn create_benchmark_config() -> ServerConfig {
    let mut config = ServerConfig::default();
    config.host = "127.0.0.1".to_string();
    config.port = 0;
    config.num_workers = 4;
    config.model_config.model_name = "benchmark-model".to_string();
    config.model_config.model_version = Some("1.0.0".to_string());
    config.model_config.device = Device::Cpu;
    config.model_config.max_sequence_length = 1024;
    config.model_config.enable_caching = true;

    // Configure batching with correct field names
    config.batching_config.max_batch_size = 32;
    config.batching_config.max_wait_time = std::time::Duration::from_millis(10);
    config.batching_config.enable_adaptive_batching = true;

    config
}

/// Create test request data for benchmarks
fn create_test_request(input_size: usize) -> Request {
    let input_text = "a".repeat(input_size);
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("model".to_string(), "benchmark-model".to_string());
    metadata.insert("max_tokens".to_string(), "100".to_string());
    metadata.insert("temperature".to_string(), "0.7".to_string());

    Request {
        id: RequestId::new(),
        input: RequestInput::Text {
            text: input_text,
            max_length: Some(input_size),
        },
        priority: Priority::Normal,
        submitted_at: std::time::Instant::now(),
        deadline: None,
        metadata,
    }
}

/// Benchmark basic server creation and initialization
fn bench_server_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("server_creation");

    group.bench_function("create_server", |b| {
        b.iter(|| {
            let config = create_benchmark_config();
            let server = TrustformerServer::new(black_box(config));
            black_box(server)
        })
    });

    group.bench_function("create_server_with_router", |b| {
        let rt = Runtime::new().unwrap();
        b.to_async(&rt).iter(|| async {
            let config = create_benchmark_config();
            let server = TrustformerServer::new(black_box(config));
            let router = server.create_test_router().await;
            black_box(router)
        })
    });

    group.finish();
}

/// Benchmark batching service performance
fn bench_batching_service(c: &mut Criterion) {
    let mut group = c.benchmark_group("batching_service");
    let rt = Runtime::new().unwrap();

    // Benchmark single request processing
    group.bench_function("single_request", |b| {
        let batching_config = BatchingConfig::default();
        let service = DynamicBatchingService::new(batching_config);

        b.to_async(&rt).iter(|| async {
            let request = create_test_request(100);
            let result = service.submit_request(black_box(request)).await;
            black_box(result)
        })
    });

    // Benchmark batch processing with different batch sizes
    for batch_size in [1, 4, 8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_processing", batch_size),
            batch_size,
            |b, &batch_size| {
                let mut batching_config = BatchingConfig::default();
                batching_config.max_batch_size = batch_size;
                batching_config.max_wait_time = std::time::Duration::from_millis(5);
                let service = DynamicBatchingService::new(batching_config);

                b.to_async(&rt).iter(|| async {
                    let mut results = Vec::new();
                    for _i in 0..batch_size {
                        let request = create_test_request(100);
                        let result = service.submit_request(black_box(request)).await;
                        results.push(result);
                    }
                    black_box(results)
                })
            },
        );
    }

    // Benchmark with different input sizes
    for input_size in [50, 100, 500, 1000, 2000].iter() {
        group.throughput(Throughput::Bytes(*input_size as u64));
        group.bench_with_input(
            BenchmarkId::new("input_size", input_size),
            input_size,
            |b, &input_size| {
                let batching_config = BatchingConfig::default();
                let service = DynamicBatchingService::new(batching_config);

                b.to_async(&rt).iter(|| async {
                    let request = create_test_request(input_size);
                    let result = service.submit_request(black_box(request)).await;
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark caching service performance
fn bench_caching_service(c: &mut Criterion) {
    let mut group = c.benchmark_group("caching_service");
    let rt = Runtime::new().unwrap();

    // Benchmark cache operations
    group.bench_function("cache_set", |b| {
        let cache_config = CacheConfig::default();
        let service = CachingService::new(cache_config);

        b.to_async(&rt).iter(|| async {
            let cache_key = CacheKey::new(
                "benchmark-model".to_string(),
                &format!("bench-input-{}", random::<u64>()),
                &std::collections::HashMap::new(),
                Some("1.0.0".to_string()),
            );
            let cache_result = CacheResult {
                output: "benchmark-value".repeat(100),
                tokens: Some(vec![1, 2, 3, 4, 5]),
                logits: None,
                metadata: std::collections::HashMap::new(),
                processing_time_ms: 50,
                model_used: "benchmark-model".to_string(),
            };
            let result =
                service.result_cache().put(black_box(cache_key), black_box(cache_result)).await;
            black_box(result)
        })
    });

    group.bench_function("cache_get", |b| {
        let cache_config = CacheConfig::default();
        let service = CachingService::new(cache_config);
        let rt_handle = rt.handle().clone();

        // Pre-populate cache
        rt_handle.block_on(async {
            for i in 0..1000 {
                let cache_key = CacheKey::new(
                    "benchmark-model".to_string(),
                    &format!("bench-input-{}", i),
                    &std::collections::HashMap::new(),
                    Some("1.0.0".to_string()),
                );
                let cache_result = CacheResult {
                    output: format!("benchmark-value-{}", i),
                    tokens: Some(vec![1, 2, 3, 4, 5]),
                    logits: None,
                    metadata: std::collections::HashMap::new(),
                    processing_time_ms: 50,
                    model_used: "benchmark-model".to_string(),
                };
                let _ = service.result_cache().put(cache_key, cache_result).await;
            }
        });

        b.to_async(&rt).iter(|| async {
            let cache_key = CacheKey::new(
                "benchmark-model".to_string(),
                &format!("bench-input-{}", random::<u64>() % 1000),
                &std::collections::HashMap::new(),
                Some("1.0.0".to_string()),
            );
            let result = service.result_cache().get(black_box(&cache_key)).await;
            black_box(result)
        })
    });

    // Benchmark cache performance with different value sizes
    for value_size in [100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Bytes(*value_size as u64));
        group.bench_with_input(
            BenchmarkId::new("cache_value_size", value_size),
            value_size,
            |b, &value_size| {
                let cache_config = CacheConfig::default();
                let service = CachingService::new(cache_config);

                b.to_async(&rt).iter(|| async {
                    let cache_key = CacheKey::new(
                        "benchmark-model".to_string(),
                        &format!("bench-input-{}", random::<u64>()),
                        &std::collections::HashMap::new(),
                        Some("1.0.0".to_string()),
                    );
                    let cache_result = CacheResult {
                        output: "x".repeat(value_size),
                        tokens: Some(vec![1; value_size.min(100)]),
                        logits: None,
                        metadata: std::collections::HashMap::new(),
                        processing_time_ms: 50,
                        model_used: "benchmark-model".to_string(),
                    };
                    let result = service
                        .result_cache()
                        .put(black_box(cache_key), black_box(cache_result))
                        .await;
                    black_box(result)
                })
            },
        );
    }

    // Benchmark cache hit/miss ratios
    group.bench_function("cache_hit_miss_ratio", |b| {
        let mut cache_config = CacheConfig::default();
        cache_config.result_cache.max_size_bytes = 1000; // Small cache to force evictions
        let service = CachingService::new(cache_config);

        b.to_async(&rt).iter(|| async {
            // Mix of operations that will cause hits and misses
            for i in 0..150 {
                let cache_key = CacheKey::new(
                    "benchmark-model".to_string(),
                    &format!("bench-input-{}", i % 120), // Overlap to cause hits
                    &std::collections::HashMap::new(),
                    Some("1.0.0".to_string()),
                );
                let cache_result = CacheResult {
                    output: format!("value-{}", i),
                    tokens: Some(vec![1, 2, 3]),
                    logits: None,
                    metadata: std::collections::HashMap::new(),
                    processing_time_ms: 50,
                    model_used: "benchmark-model".to_string(),
                };
                let _ = service.result_cache().put(cache_key.clone(), cache_result).await;
                let _ = service.result_cache().get(&cache_key).await;
            }
        })
    });

    group.finish();
}

/// Benchmark streaming service performance
fn bench_streaming_service(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_service");
    let rt = Runtime::new().unwrap();

    group.bench_function("stream_creation", |b| {
        let streaming_config = StreamingConfig::default();
        let service = StreamingService::new(streaming_config);

        b.to_async(&rt).iter(|| async {
            let request_id = uuid::Uuid::new_v4();
            let result = service
                .start_stream(black_box(StreamType::TokenStream), black_box(request_id))
                .await;
            black_box(result)
        })
    });

    // Benchmark streaming throughput with different chunk sizes
    for chunk_size in [64, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Bytes(*chunk_size as u64));
        group.bench_with_input(
            BenchmarkId::new("stream_throughput", chunk_size),
            chunk_size,
            |b, &chunk_size| {
                let streaming_config = StreamingConfig::default();
                let service = StreamingService::new(streaming_config);

                b.to_async(&rt).iter(|| async {
                    let request_id = uuid::Uuid::new_v4();
                    let stream_data = StreamData::Token("x".repeat(chunk_size));

                    if let Ok(handle) =
                        service.start_stream(StreamType::TokenStream, request_id).await
                    {
                        let result = service
                            .send_to_stream(black_box(handle.id), black_box(stream_data))
                            .await;
                        let _ = service.close_stream(handle.id).await;
                        black_box(result)
                    } else {
                        black_box(Ok(()))
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark validation service performance
fn bench_validation_service(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_service");
    let rt = Runtime::new().unwrap();

    let validation_config = ValidationConfig::default();
    let service = ValidationService::new(validation_config).unwrap();

    // Benchmark input validation with different input sizes
    for input_size in [100, 500, 1000, 5000, 10000].iter() {
        group.throughput(Throughput::Bytes(*input_size as u64));
        group.bench_with_input(
            BenchmarkId::new("validate_input", input_size),
            input_size,
            |b, &input_size| {
                let input_text = "Hello world! ".repeat(input_size / 13);

                b.iter(|| {
                    let result = service.validate_text(black_box(&input_text));
                    black_box(result)
                })
            },
        );
    }

    // Benchmark profanity detection
    group.bench_function("profanity_detection", |b| {
        let input_text = "This is a clean text with no profanity for testing purposes";

        b.iter(|| {
            let result = service.validate_text(black_box(input_text));
            black_box(result)
        })
    });

    // Benchmark PII detection
    group.bench_function("pii_detection", |b| {
        let input_text = "My email is john.doe@example.com and my phone is 555-1234";

        b.iter(|| {
            let result = service.validate_text(black_box(input_text));
            black_box(result)
        })
    });

    group.finish();
}

/// Benchmark rate limiting service performance
fn bench_rate_limiting_service(c: &mut Criterion) {
    let mut group = c.benchmark_group("rate_limiting_service");
    let rt = Runtime::new().unwrap();

    group.bench_function("rate_limit_check", |b| {
        let rate_limit_config = RateLimitConfig::default();
        let service = RateLimitService::new(rate_limit_config).unwrap();

        b.to_async(&rt).iter(|| async {
            let client_id = format!("client-{}", random::<u64>() % 100);
            let result = service.check_rate_limit(black_box(&client_id)).await;
            black_box(result)
        })
    });

    // Benchmark rate limiting with different client loads
    for num_clients in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_clients", num_clients),
            num_clients,
            |b, &num_clients| {
                let rate_limit_config = RateLimitConfig::default();
                let service = RateLimitService::new(rate_limit_config).unwrap();

                b.to_async(&rt).iter(|| async {
                    for i in 0..num_clients {
                        let client_id = format!("client-{}", i);
                        let _ = service.check_rate_limit(black_box(&client_id)).await;
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark metrics collection performance
fn bench_metrics_service(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics_service");
    let rt = Runtime::new().unwrap();

    group.bench_function("metrics_collection", |b| {
        let service = MetricsService::new();

        b.iter(|| {
            service.collector().increment_requests();
            service.collector().observe_request_duration(0.123);
            service.collector().set_queue_size(42);
            black_box(())
        })
    });

    // Benchmark metrics export
    group.bench_function("metrics_export", |b| {
        let service = MetricsService::new();

        // Pre-populate some metrics
        for i in 0..1000 {
            service.collector().increment_requests();
            service.collector().observe_request_duration(i as f64);
            service.collector().set_queue_size(i);
        }

        b.to_async(&rt).iter(|| async {
            let metrics_text = service.get_metrics().await;
            black_box(metrics_text)
        })
    });

    group.finish();
}

/// Comprehensive end-to-end benchmark
fn bench_end_to_end_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");
    let rt = Runtime::new().unwrap();

    group.bench_function("complete_request_flow", |b| {
        let config = create_benchmark_config();

        b.to_async(&rt).iter(|| async {
            // Simulate complete request processing flow
            let request = create_test_request(500);

            // 1. Validation
            // let validation_result = validate_request(&request).await;

            // 2. Rate limiting check
            // let rate_limit_result = check_rate_limit("test-client").await;

            // 3. Caching check
            // let cache_result = check_cache(&request.input).await;

            // 4. Batching
            // let batch_result = add_to_batch(request).await;

            // 5. Metrics recording
            // record_metrics().await;

            // For now, just measure server router creation as proxy
            let server = TrustformerServer::new(config.clone());
            let router = server.create_test_router().await;
            black_box(router)
        })
    });

    group.finish();
}

/// Memory usage benchmark
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    group.bench_function("server_memory_footprint", |b| {
        b.iter(|| {
            let config = create_benchmark_config();
            let server = TrustformerServer::new(black_box(config));

            // Measure memory usage (approximation)
            // Memory profiling is complex and platform specific
            // For benchmarking purposes, we just measure the server creation time
            black_box(server);
        })
    });

    group.finish();
}

/// Stress test benchmark
fn bench_stress_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_test");
    let rt = Runtime::new().unwrap();

    group.sample_size(10); // Fewer samples for stress tests
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("high_concurrency", |b| {
        let config = create_benchmark_config();

        b.to_async(&rt).iter(|| async {
            let mut results = Vec::new();
            for _i in 0..100 {
                let server = TrustformerServer::new(config.clone());
                let router = server.create_test_router().await;
                results.push(black_box(router));
            }
            black_box(results)
        })
    });

    group.finish();
}

// Group all benchmarks
criterion_group!(
    benches,
    bench_server_creation,
    bench_batching_service,
    bench_caching_service,
    bench_streaming_service,
    bench_validation_service,
    bench_rate_limiting_service,
    bench_metrics_service,
    bench_end_to_end_performance,
    bench_memory_usage,
    bench_stress_test
);

criterion_main!(benches);
