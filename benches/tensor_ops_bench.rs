//! Benchmarks for core tensor operations

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::hint::black_box;
use trustformers_core::tensor::Tensor;

fn tensor_creation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");

    for size in [128, 512, 1024, 2048, 4096].iter() {
        let shape = vec![*size, *size];

        group.throughput(Throughput::Elements((*size * *size) as u64));

        group.bench_with_input(BenchmarkId::new("zeros", size), &shape, |b, shape| {
            b.iter(|| {
                let tensor = Tensor::zeros(shape);
                black_box(tensor)
            })
        });

        group.bench_with_input(BenchmarkId::new("ones", size), &shape, |b, shape| {
            b.iter(|| {
                let tensor = Tensor::ones(shape);
                black_box(tensor)
            })
        });

        group.bench_with_input(BenchmarkId::new("randn", size), &shape, |b, shape| {
            b.iter(|| {
                let tensor = Tensor::randn(shape);
                black_box(tensor)
            })
        });
    }

    group.finish();
}

fn tensor_arithmetic_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_arithmetic");

    for size in [256, 512, 1024, 2048].iter() {
        let shape = vec![*size, *size];
        let a = Tensor::randn(&shape).expect("Failed to create tensor");
        let b = Tensor::randn(&shape).expect("Failed to create tensor");

        group.throughput(Throughput::Elements((*size * *size) as u64));

        group.bench_with_input(BenchmarkId::new("add", size), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| {
                let result = a.add(b);
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("mul", size), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| {
                let result = a.mul(b);
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("matmul", size), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| {
                let result = a.matmul(b);
                black_box(result)
            })
        });
    }

    group.finish();
}

fn tensor_reduction_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_reduction");

    for (batch, seq_len, hidden) in [(1, 128, 768), (8, 512, 768), (32, 128, 768)].iter() {
        let shape = vec![*batch, *seq_len, *hidden];
        let tensor = Tensor::randn(&shape).expect("Failed to create tensor");
        let size_str = format!("{}x{}x{}", batch, seq_len, hidden);

        group.throughput(Throughput::Elements((batch * seq_len * hidden) as u64));

        group.bench_with_input(BenchmarkId::new("sum", &size_str), &tensor, |b, tensor| {
            b.iter(|| {
                let result = tensor.sum();
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("mean", &size_str), &tensor, |b, tensor| {
            b.iter(|| {
                let result = tensor.mean();
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("softmax", &size_str), &tensor, |b, tensor| {
            b.iter(|| {
                let result = tensor.softmax(-1);
                black_box(result)
            })
        });
    }

    group.finish();
}

fn tensor_manipulation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_manipulation");

    for size in [512, 1024, 2048].iter() {
        let shape = vec![*size, *size];
        let tensor = Tensor::randn(&shape).expect("Failed to create tensor");

        group.throughput(Throughput::Elements((*size * *size) as u64));

        group.bench_with_input(BenchmarkId::new("transpose", size), &tensor, |b, tensor| {
            b.iter(|| {
                let result = tensor.transpose();
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("reshape", size), &tensor, |b, tensor| {
            b.iter(|| {
                let new_shape = vec![1, *size * *size];
                let result = tensor.reshape(&new_shape);
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("slice", size), &tensor, |b, tensor| {
            b.iter(|| {
                let result = tensor.slice(&[(0, *size/2), (0, *size/2)]);
                black_box(result)
            })
        });
    }

    group.finish();
}

#[cfg(feature = "gpu")]
fn tensor_gpu_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_gpu");

    for size in [1024, 2048, 4096].iter() {
        let shape = vec![*size, *size];
        let cpu_tensor = Tensor::randn(&shape).expect("Failed to create tensor");

        group.throughput(Throughput::Bytes((*size * *size * 4) as u64));

        group.bench_with_input(BenchmarkId::new("to_gpu", size), &cpu_tensor, |b, tensor| {
            b.iter(|| {
                let gpu_tensor = tensor.to_gpu();
                black_box(gpu_tensor)
            })
        });

        group.bench_with_input(BenchmarkId::new("gpu_matmul", size), &cpu_tensor, |b, tensor| {
            let a_gpu = tensor.to_gpu().expect("Failed to move to GPU");
            let b_gpu = tensor.to_gpu().expect("Failed to move to GPU");

            b.iter(|| {
                let result = a_gpu.matmul(&b_gpu);
                black_box(result)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    tensor_creation_benchmarks,
    tensor_arithmetic_benchmarks,
    tensor_reduction_benchmarks,
    tensor_manipulation_benchmarks
);

#[cfg(feature = "gpu")]
criterion_group!(gpu_benches, tensor_gpu_benchmarks);

#[cfg(not(feature = "gpu"))]
criterion_main!(benches);

#[cfg(feature = "gpu")]
criterion_main!(benches, gpu_benches);