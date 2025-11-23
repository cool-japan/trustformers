#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]
//! Hardware acceleration performance benchmarks
//!
//! This module benchmarks different hardware acceleration backends
//! to demonstrate performance improvements and help select optimal backends.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use trustformers_core::hardware_acceleration::{
    api, AccelerationBackend, AccelerationConfig, HardwareAccelerator,
};
use trustformers_core::tensor::Tensor;

// Helper function to create test tensors
fn create_test_tensor(shape: &[usize]) -> Tensor {
    Tensor::randn(shape).unwrap()
}

// Test different matrix multiplication sizes common in ML
fn benchmark_accelerated_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("accelerated_matmul");

    // Initialize hardware acceleration
    let _ = api::init_hardware_acceleration();

    // Test various matrix sizes common in transformer models
    let matrix_sizes = vec![
        (256, 256, 256),    // Small matrices
        (512, 512, 512),    // Medium matrices
        (1024, 1024, 1024), // Large matrices
        (32, 768, 768),     // BERT-base dimensions
        (16, 1024, 1024),   // GPT-style dimensions
        (8, 2048, 2048),    // Large transformer dimensions
        (64, 512, 2048),    // MLP-style dimensions
        (128, 768, 3072),   // Transformer feed-forward
    ];

    for (batch, m, k) in matrix_sizes {
        let tensor_a = create_test_tensor(&[batch, m]);
        let tensor_b = create_test_tensor(&[m, k]);
        let mut result = Tensor::zeros(&[batch, k]).unwrap();

        group.bench_with_input(
            BenchmarkId::new("hardware_accelerated", format!("{}x{}x{}", batch, m, k)),
            &(&tensor_a, &tensor_b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let mut result = Tensor::zeros(&[batch, k]).unwrap();
                    api::accelerated_matmul(a, b, &mut result).unwrap();
                    black_box(result)
                })
            },
        );

        // Compare with standard tensor matmul
        group.bench_with_input(
            BenchmarkId::new("standard_tensor", format!("{}x{}x{}", batch, m, k)),
            &(&tensor_a, &tensor_b),
            |bench, (a, b)| bench.iter(|| black_box(a.matmul(b).unwrap())),
        );
    }

    group.finish();
}

// Benchmark Flash Attention performance
fn benchmark_accelerated_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("accelerated_attention");

    // Initialize hardware acceleration
    let _ = api::init_hardware_acceleration();

    // Attention sizes common in transformer models
    let attention_configs = vec![
        (32, 128, 64),  // Small: (batch_size, seq_len, head_dim)
        (16, 512, 64),  // Medium sequence
        (8, 1024, 64),  // Long sequence
        (4, 2048, 64),  // Very long sequence
        (32, 256, 128), // Larger head dimension
        (16, 512, 128), // BERT-large style
    ];

    for (batch_size, seq_len, head_dim) in attention_configs {
        let query = create_test_tensor(&[batch_size, seq_len, head_dim]);
        let key = create_test_tensor(&[batch_size, seq_len, head_dim]);
        let value = create_test_tensor(&[batch_size, seq_len, head_dim]);
        let mut output = Tensor::zeros(&[batch_size, seq_len, head_dim]).unwrap();

        group.bench_with_input(
            BenchmarkId::new(
                "flash_attention",
                format!("{}x{}x{}", batch_size, seq_len, head_dim),
            ),
            &(&query, &key, &value),
            |bench, (q, k, v)| {
                bench.iter(|| {
                    let mut output = Tensor::zeros(&[batch_size, seq_len, head_dim]).unwrap();
                    api::accelerated_flash_attention(q, k, v, &mut output).unwrap();
                    black_box(output)
                })
            },
        );

        // Compare with manual attention computation
        group.bench_with_input(
            BenchmarkId::new(
                "manual_attention",
                format!("{}x{}x{}", batch_size, seq_len, head_dim),
            ),
            &(&query, &key, &value),
            |bench, (q, k, v)| {
                bench.iter(|| {
                    // Manual attention: softmax(QK^T/sqrt(d_k))V
                    let scale = 1.0 / (head_dim as f32).sqrt();
                    let key_t = k.transpose(1, 2).unwrap(); // transpose last two dimensions
                    let scores = q.matmul(&key_t).unwrap();
                    let scaled_scores = scores.mul_scalar(scale).unwrap();
                    let attention_weights = scaled_scores.softmax(-1).unwrap();
                    let result = attention_weights.matmul(v).unwrap();
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

// Benchmark backend availability and initialization overhead
fn benchmark_backend_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("backend_performance");

    // Test backend availability checking
    group.bench_function("check_all_backends", |b| {
        b.iter(|| {
            let backends = api::list_available_backends();
            black_box(backends)
        })
    });

    // Test individual backend availability
    let test_backends = vec![
        AccelerationBackend::Cpu,
        AccelerationBackend::Cuda,
        AccelerationBackend::Rocm,
        AccelerationBackend::Intel,
        AccelerationBackend::Vulkan,
        AccelerationBackend::Metal,
    ];

    for backend in test_backends {
        group.bench_with_input(
            BenchmarkId::new("backend_available", format!("{:?}", backend)),
            &backend,
            |b, backend| b.iter(|| black_box(api::is_backend_available(*backend))),
        );
    }

    // Test hardware accelerator initialization
    group.bench_function("init_default_config", |b| {
        b.iter(|| {
            let config = AccelerationConfig::default();
            black_box(config)
        })
    });

    group.bench_function("init_acceleration", |b| {
        b.iter(|| black_box(api::init_hardware_acceleration().is_ok()))
    });

    group.finish();
}

// Benchmark memory and device operations
fn benchmark_device_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_operations");

    // Initialize hardware acceleration
    let _ = api::init_hardware_acceleration();

    // Test device information queries
    group.bench_function("get_active_backend", |b| {
        b.iter(|| black_box(api::get_active_backend().unwrap()))
    });

    group.bench_function("get_device_info", |b| {
        b.iter(|| black_box(api::get_device_info().unwrap()))
    });

    group.bench_function("get_memory_stats", |b| {
        b.iter(|| black_box(api::get_memory_stats().unwrap()))
    });

    group.bench_function("get_performance_stats", |b| {
        b.iter(|| black_box(api::get_performance_stats().unwrap()))
    });

    group.finish();
}

// Benchmark throughput for different operation sizes
fn benchmark_operation_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("operation_throughput");
    group.sample_size(20); // Reduce sample size for large operations
    group.measurement_time(Duration::from_secs(10)); // Longer measurement time

    // Initialize hardware acceleration
    let _ = api::init_hardware_acceleration();

    // Test throughput with different batch sizes
    let batch_configs = vec![
        (1, 512, 512),   // Single sample
        (8, 512, 512),   // Small batch
        (32, 512, 512),  // Medium batch
        (128, 512, 512), // Large batch
        (512, 512, 512), // Very large batch
    ];

    for (batch_size, m, k) in batch_configs {
        let tensor_a = create_test_tensor(&[batch_size, m]);
        let tensor_b = create_test_tensor(&[m, k]);
        let mut result = Tensor::zeros(&[batch_size, k]).unwrap();

        // Calculate theoretical FLOPS for matrix multiplication
        let flops = 2 * batch_size * m * k; // Multiply-add operations

        group.bench_with_input(
            BenchmarkId::new(
                "matmul_throughput",
                format!("batch_{}_{}x{}", batch_size, m, k),
            ),
            &(&tensor_a, &tensor_b, flops),
            |bench, (a, b, _flops)| {
                bench.iter(|| {
                    let mut result = Tensor::zeros(&[batch_size, k]).unwrap();
                    api::accelerated_matmul(a, b, &mut result).unwrap();
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

// Benchmark memory pressure scenarios
fn benchmark_memory_pressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pressure");
    group.sample_size(10); // Reduce sample size for memory-intensive tests
    group.measurement_time(Duration::from_secs(15)); // Longer measurement time

    // Initialize hardware acceleration
    let _ = api::init_hardware_acceleration();

    // Test with increasingly large tensors to stress memory
    let memory_sizes = vec![
        (1024, 1024), // ~4 MB (1M floats)
        (2048, 2048), // ~16 MB (4M floats)
        (4096, 4096), // ~64 MB (16M floats)
        (8192, 8192), // ~256 MB (64M floats)
    ];

    for (rows, cols) in memory_sizes {
        let size_mb = (rows * cols * 4) / (1024 * 1024); // Size in MB

        group.bench_with_input(
            BenchmarkId::new(
                "large_tensor_matmul",
                format!("{}MB_{}x{}", size_mb, rows, cols),
            ),
            &(rows, cols),
            |bench, (rows, cols)| {
                bench.iter(|| {
                    let a = create_test_tensor(&[*rows, *cols]);
                    let b = create_test_tensor(&[*cols, *rows]);
                    let mut result = Tensor::zeros(&[*rows, *rows]).unwrap();
                    api::accelerated_matmul(&a, &b, &mut result).unwrap();
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    hardware_benches,
    benchmark_accelerated_matmul,
    benchmark_accelerated_attention,
    benchmark_backend_performance,
    benchmark_device_operations,
    benchmark_operation_throughput,
    benchmark_memory_pressure
);

criterion_main!(hardware_benches);
