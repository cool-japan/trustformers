//! Basic Performance Benchmarks for TrustformeRS Serve
//!
//! Focused benchmarks for core functionality that doesn't require
//! complex service interactions or mocking.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde_json::json;
use std::hint::black_box;
use std::time::Duration;
use tokio::runtime::Runtime;
use trustformers_serve::{Device, ModelConfig, ServerConfig, TrustformerServer};

/// Create a benchmark-optimized server configuration
fn create_benchmark_config() -> ServerConfig {
    let mut config = ServerConfig::default();
    config.host = "127.0.0.1".to_string();
    config.port = 0;
    config.num_workers = 4;
    config.model_config = ModelConfig {
        model_name: "benchmark-model".to_string(),
        model_version: Some("1.0.0".to_string()),
        device: Device::Cpu,
        max_sequence_length: 1024,
        enable_caching: true,
    };
    config
}

/// Benchmark server creation and configuration
fn bench_server_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("server_creation");

    group.bench_function("create_server_config", |b| {
        b.iter(|| {
            let config = create_benchmark_config();
            black_box(config)
        })
    });

    group.bench_function("create_server_instance", |b| {
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

/// Benchmark configuration serialization/deserialization
fn bench_config_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_serialization");

    let config = create_benchmark_config();

    group.bench_function("serialize_config", |b| {
        b.iter(|| {
            let json_str = serde_json::to_string(&config);
            black_box(json_str)
        })
    });

    let json_str = serde_json::to_string(&config).unwrap();
    group.bench_function("deserialize_config", |b| {
        b.iter(|| {
            let config: Result<ServerConfig, _> = serde_json::from_str(black_box(&json_str));
            black_box(config)
        })
    });

    group.bench_function("serialize_deserialize_roundtrip", |b| {
        b.iter(|| {
            let json_str = serde_json::to_string(&config).unwrap();
            let config2: ServerConfig = serde_json::from_str(black_box(&json_str)).unwrap();
            black_box(config2)
        })
    });

    group.finish();
}

/// Benchmark different configuration variations
fn bench_config_variations(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_variations");

    // Benchmark with different model configurations
    for seq_length in [512, 1024, 2048, 4096].iter() {
        group.bench_with_input(
            BenchmarkId::new("model_seq_length", seq_length),
            seq_length,
            |b, &seq_length| {
                b.iter(|| {
                    let mut config = create_benchmark_config();
                    config.model_config.max_sequence_length = seq_length;
                    let server = TrustformerServer::new(black_box(config));
                    black_box(server)
                })
            },
        );
    }

    // Benchmark with different worker counts
    for num_workers in [1, 2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("num_workers", num_workers),
            num_workers,
            |b, &num_workers| {
                b.iter(|| {
                    let mut config = create_benchmark_config();
                    config.num_workers = num_workers;
                    let server = TrustformerServer::new(black_box(config));
                    black_box(server)
                })
            },
        );
    }

    // Benchmark with different device types
    let devices = vec![
        ("cpu", Device::Cpu),
        ("cuda_0", Device::Cuda(0)),
        ("metal", Device::Metal),
    ];

    for (name, device) in devices {
        group.bench_with_input(
            BenchmarkId::new("device_type", name),
            &device,
            |b, device| {
                b.iter(|| {
                    let mut config = create_benchmark_config();
                    config.model_config.device = device.clone();
                    let server = TrustformerServer::new(black_box(config));
                    black_box(server)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark JSON processing performance
fn bench_json_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_processing");

    // Benchmark JSON creation for different request sizes
    for input_size in [100, 500, 1000, 5000, 10000].iter() {
        group.throughput(Throughput::Bytes(*input_size as u64));
        group.bench_with_input(
            BenchmarkId::new("create_request_json", input_size),
            input_size,
            |b, &input_size| {
                let input_text = "a".repeat(input_size);
                b.iter(|| {
                    let request_json = json!({
                        "model": "benchmark-model",
                        "input": black_box(&input_text),
                        "parameters": {
                            "max_tokens": 100,
                            "temperature": 0.7,
                            "top_p": 0.95
                        }
                    });
                    black_box(request_json)
                })
            },
        );
    }

    // Benchmark JSON parsing
    let large_json = json!({
        "model": "benchmark-model",
        "input": "a".repeat(1000),
        "parameters": {
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": ["\\n", ".", "!", "?"],
            "logit_bias": {
                "50256": -100
            }
        },
        "metadata": {
            "user_id": "benchmark-user",
            "session_id": "benchmark-session",
            "timestamp": "2025-07-16T12:00:00Z",
            "request_source": "benchmark"
        }
    });

    let json_str = serde_json::to_string(&large_json).unwrap();

    group.bench_function("parse_large_json", |b| {
        b.iter(|| {
            let parsed: serde_json::Value = serde_json::from_str(black_box(&json_str)).unwrap();
            black_box(parsed)
        })
    });

    group.finish();
}

/// Benchmark concurrent server operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_operations");
    let rt = Runtime::new().unwrap();

    // Benchmark concurrent server creation
    for concurrency in [1, 2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_server_creation", concurrency),
            concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let handles: Vec<_> = (0..concurrency)
                        .map(|i| {
                            tokio::spawn(async move {
                                let mut config = create_benchmark_config();
                                config.port = 8000 + i;
                                let server = TrustformerServer::new(config);
                                black_box(server)
                            })
                        })
                        .collect();

                    futures::future::join_all(handles).await
                })
            },
        );
    }

    // Benchmark concurrent router creation
    group.bench_function("concurrent_router_creation", |b| {
        b.to_async(&rt).iter(|| async {
            let handles: Vec<_> = (0..4)
                .map(|i| {
                    tokio::spawn(async move {
                        let mut config = create_benchmark_config();
                        config.port = 9000 + i;
                        let server = TrustformerServer::new(config);
                        let router = server.create_test_router().await;
                        black_box(router)
                    })
                })
                .collect();

            futures::future::join_all(handles).await
        })
    });

    group.finish();
}

/// Benchmark memory-intensive operations
fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");

    // Benchmark creating multiple server instances
    group.bench_function("multiple_server_instances", |b| {
        b.iter(|| {
            let mut servers = Vec::new();
            for i in 0..10 {
                let mut config = create_benchmark_config();
                config.port = 10000 + i;
                let server = TrustformerServer::new(config);
                servers.push(server);
            }
            black_box(servers)
        })
    });

    // Benchmark large configuration objects
    group.bench_function("large_config_objects", |b| {
        b.iter(|| {
            let mut configs = Vec::new();
            for i in 0..100 {
                let mut config = create_benchmark_config();
                config.model_config.model_name = format!("model-{}", i);
                config.port = 11000 + i as u16;
                configs.push(config);
            }
            black_box(configs)
        })
    });

    group.finish();
}

/// Benchmark string operations
fn bench_string_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_operations");

    // Benchmark string creation for model names
    group.bench_function("model_name_generation", |b| {
        b.iter(|| {
            let model_names: Vec<String> = (0..1000)
                .map(|i| format!("model-{}-v{}.{}.{}", i, i % 10, (i * 3) % 10, (i * 7) % 10))
                .collect();
            black_box(model_names)
        })
    });

    // Benchmark input text processing
    for text_size in [100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Bytes(*text_size as u64));
        group.bench_with_input(
            BenchmarkId::new("text_processing", text_size),
            text_size,
            |b, &text_size| {
                b.iter(|| {
                    let text = "Hello world! ".repeat(text_size / 13);
                    let processed = text.to_lowercase();
                    let word_count = processed.split_whitespace().count();
                    black_box((processed, word_count))
                })
            },
        );
    }

    group.finish();
}

/// Stress test benchmark
fn bench_stress_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_tests");
    let rt = Runtime::new().unwrap();

    group.sample_size(10); // Fewer samples for stress tests
    group.measurement_time(Duration::from_secs(5));

    // Stress test: Rapid server creation and destruction
    group.bench_function("rapid_server_lifecycle", |b| {
        b.iter(|| {
            for i in 0..50 {
                let mut config = create_benchmark_config();
                config.port = 12000 + i;
                let server = TrustformerServer::new(config);
                black_box(server);
                // Server is dropped here
            }
        })
    });

    // Stress test: Memory pressure
    group.bench_function("memory_pressure", |b| {
        b.to_async(&rt).iter(|| async {
            let mut routers = Vec::new();
            for i in 0..20 {
                let mut config = create_benchmark_config();
                config.port = 13000 + i;
                config.model_config.model_name = "x".repeat(1000); // Large model name
                let server = TrustformerServer::new(config);
                let router = server.create_test_router().await;
                routers.push(router);
            }
            black_box(routers)
        })
    });

    group.finish();
}

/// End-to-end performance benchmark
fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");
    let rt = Runtime::new().unwrap();

    group.bench_function("complete_setup_flow", |b| {
        b.to_async(&rt).iter(|| async {
            // Complete flow: config -> server -> router -> cleanup
            let config = create_benchmark_config();
            let server = TrustformerServer::new(config);
            let router = server.create_test_router().await;

            // Simulate some operations
            let json_data = json!({
                "test": "data",
                "timestamp": chrono::Utc::now().timestamp(),
                "size": 12345
            });
            let _serialized = serde_json::to_string(&json_data).unwrap();

            black_box(router)
        })
    });

    group.finish();
}

// Group all benchmarks
criterion_group!(
    benches,
    bench_server_creation,
    bench_config_serialization,
    bench_config_variations,
    bench_json_processing,
    bench_concurrent_operations,
    bench_memory_operations,
    bench_string_operations,
    bench_stress_tests,
    bench_end_to_end
);

criterion_main!(benches);
