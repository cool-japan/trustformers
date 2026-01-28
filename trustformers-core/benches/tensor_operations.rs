//! Micro-benchmarks for tensor operations.
//!
//! This module contains comprehensive benchmarks for all tensor operations
//! to track performance regressions and optimize critical paths.

#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use trustformers_core::tensor::Tensor;

// Helper function to create test tensors of various sizes
fn create_f32_tensor(shape: &[usize]) -> Tensor {
    let size: usize = shape.iter().product();
    let values: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
    Tensor::from_vec(values, shape).unwrap()
}

// Helper function to create random test tensors
fn create_random_f32_tensor(shape: &[usize]) -> Tensor {
    Tensor::randn(shape).unwrap()
}

// Helper function to create positive test tensors for operations like sqrt, log
fn create_positive_f32_tensor(shape: &[usize]) -> Tensor {
    let size: usize = shape.iter().product();
    let values: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 + 1.0).collect();
    Tensor::from_vec(values, shape).unwrap()
}

fn benchmark_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");

    let shapes = vec![
        vec![100],
        vec![32, 32],
        vec![16, 16, 16],
        vec![8, 8, 8, 8],
        vec![1000],
        vec![100, 100],
        vec![50, 50, 50],
    ];

    for shape in shapes {
        let size: usize = shape.iter().product();

        group.bench_with_input(
            BenchmarkId::new("zeros_f32", format!("{:?}", shape)),
            &shape,
            |b, shape| b.iter(|| black_box(Tensor::zeros(shape).unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new("ones_f32", format!("{:?}", shape)),
            &shape,
            |b, shape| b.iter(|| black_box(Tensor::ones(shape).unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new("randn_f32", format!("{:?}", shape)),
            &shape,
            |b, shape| b.iter(|| black_box(Tensor::randn(shape).unwrap())),
        );

        let values: Vec<f32> = (0..size).map(|i| i as f32).collect();
        group.bench_with_input(
            BenchmarkId::new("from_vec_f32", format!("{:?}", shape)),
            &(values, shape),
            |b, (values, shape)| {
                b.iter(|| black_box(Tensor::from_vec(values.clone(), shape).unwrap()))
            },
        );
    }

    group.finish();
}

fn benchmark_element_wise_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("element_wise_operations");

    let shapes = vec![
        vec![1000],
        vec![100, 100],
        vec![50, 50, 50],
        vec![25, 25, 25, 25],
    ];

    for shape in shapes {
        let tensor_a = create_f32_tensor(&shape);
        let tensor_b = create_f32_tensor(&shape);
        let positive_tensor = create_positive_f32_tensor(&shape);

        // Addition
        group.bench_with_input(
            BenchmarkId::new("add", format!("{:?}", shape)),
            &(&tensor_a, &tensor_b),
            |bench, (a, b)| bench.iter(|| black_box(a.add(b).unwrap())),
        );

        // Subtraction
        group.bench_with_input(
            BenchmarkId::new("sub", format!("{:?}", shape)),
            &(&tensor_a, &tensor_b),
            |bench, (a, b)| bench.iter(|| black_box(a.sub(b).unwrap())),
        );

        // Multiplication
        group.bench_with_input(
            BenchmarkId::new("mul", format!("{:?}", shape)),
            &(&tensor_a, &tensor_b),
            |bench, (a, b)| bench.iter(|| black_box(a.mul(b).unwrap())),
        );

        // Division
        group.bench_with_input(
            BenchmarkId::new("div", format!("{:?}", shape)),
            &(&tensor_a, &positive_tensor),
            |bench, (a, b)| bench.iter(|| black_box(a.div(b).unwrap())),
        );

        // Scalar operations
        group.bench_with_input(
            BenchmarkId::new("add_scalar", format!("{:?}", shape)),
            &tensor_a,
            |b, tensor| b.iter(|| black_box(tensor.add_scalar(2.5).unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new("mul_scalar", format!("{:?}", shape)),
            &tensor_a,
            |b, tensor| b.iter(|| black_box(tensor.mul_scalar(2.5).unwrap())),
        );

        // Mathematical functions
        group.bench_with_input(
            BenchmarkId::new("sqrt", format!("{:?}", shape)),
            &positive_tensor,
            |b, tensor| b.iter(|| black_box(tensor.sqrt().unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new("exp", format!("{:?}", shape)),
            &tensor_a,
            |b, tensor| b.iter(|| black_box(tensor.exp().unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new("log", format!("{:?}", shape)),
            &positive_tensor,
            |b, tensor| b.iter(|| black_box(tensor.log().unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new("pow", format!("{:?}", shape)),
            &positive_tensor,
            |b, tensor| b.iter(|| black_box(tensor.pow(2.0).unwrap())),
        );
    }

    group.finish();
}

fn benchmark_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_operations");

    let matrix_sizes = vec![
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (16, 1024, 1024),
        (1024, 16, 1024),
        (1024, 1024, 16),
    ];

    for (m, k, n) in matrix_sizes {
        let tensor_a = create_random_f32_tensor(&[m, k]);
        let tensor_b = create_random_f32_tensor(&[k, n]);

        group.bench_with_input(
            BenchmarkId::new("matmul", format!("{}x{}x{}", m, k, n)),
            &(&tensor_a, &tensor_b),
            |bench, (a, b)| bench.iter(|| black_box(a.matmul(b).unwrap())),
        );
    }

    // Batch matrix multiplication
    let batch_sizes = vec![(4, 32, 32, 32), (8, 64, 64, 64), (16, 128, 128, 128)];

    for (batch, m, k, n) in batch_sizes {
        let tensor_a = create_random_f32_tensor(&[batch, m, k]);
        let tensor_b = create_random_f32_tensor(&[batch, k, n]);

        group.bench_with_input(
            BenchmarkId::new("batch_matmul", format!("{}x{}x{}x{}", batch, m, k, n)),
            &(&tensor_a, &tensor_b),
            |bench, (a, b)| bench.iter(|| black_box(a.matmul(b).unwrap())),
        );
    }

    group.finish();
}

fn benchmark_activation_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_functions");

    let shapes = vec![
        vec![1000],
        vec![100, 100],
        vec![32, 768], // Common transformer dimensions
        vec![16, 1024],
        vec![8, 2048],
    ];

    for shape in shapes {
        let tensor = create_random_f32_tensor(&shape);

        group.bench_with_input(
            BenchmarkId::new("relu", format!("{:?}", shape)),
            &tensor,
            |b, tensor| b.iter(|| black_box(tensor.relu().unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new("sigmoid", format!("{:?}", shape)),
            &tensor,
            |b, tensor| b.iter(|| black_box(tensor.sigmoid().unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new("tanh", format!("{:?}", shape)),
            &tensor,
            |b, tensor| b.iter(|| black_box(tensor.tanh().unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new("gelu", format!("{:?}", shape)),
            &tensor,
            |b, tensor| b.iter(|| black_box(tensor.gelu().unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new("softmax", format!("{:?}", shape)),
            &tensor,
            |b, tensor| b.iter(|| black_box(tensor.softmax(-1).unwrap())),
        );
    }

    group.finish();
}

fn benchmark_shape_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_operations");

    let tensor_2d = create_random_f32_tensor(&[64, 128]);
    let tensor_3d = create_random_f32_tensor(&[8, 16, 32]);
    let tensor_4d = create_random_f32_tensor(&[4, 8, 16, 32]);

    // Transpose
    group.bench_function("transpose_2d", |b| {
        b.iter(|| black_box(tensor_2d.transpose(1, 0).unwrap()))
    });

    group.bench_function("transpose_3d", |b| {
        b.iter(|| black_box(tensor_3d.transpose(0, 2).unwrap()))
    });

    // Reshape
    group.bench_function("reshape_2d_to_1d", |b| {
        b.iter(|| black_box(tensor_2d.reshape(&[64 * 128]).unwrap()))
    });

    group.bench_function("reshape_3d_to_2d", |b| {
        b.iter(|| black_box(tensor_3d.reshape(&[8, 16 * 32]).unwrap()))
    });

    group.bench_function("reshape_4d_to_2d", |b| {
        b.iter(|| black_box(tensor_4d.reshape(&[4 * 8, 16 * 32]).unwrap()))
    });

    // Slice operations
    group.bench_function("slice_2d", |b| {
        b.iter(|| black_box(tensor_2d.slice(0, 0, 32).unwrap()))
    });

    group.bench_function("slice_3d", |b| {
        b.iter(|| black_box(tensor_3d.slice(0, 0, 4).unwrap()))
    });

    // Concatenation
    let tensor_a = create_random_f32_tensor(&[32, 64]);
    let tensor_b = create_random_f32_tensor(&[32, 64]);
    let tensors = vec![tensor_a.clone(), tensor_b.clone()];

    group.bench_function("concat_2d_axis0", |b| {
        b.iter(|| black_box(Tensor::concat(&tensors, 0).unwrap()))
    });

    group.bench_function("concat_2d_axis1", |b| {
        b.iter(|| black_box(Tensor::concat(&tensors, 1).unwrap()))
    });

    group.finish();
}

fn benchmark_reduction_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_operations");

    let shapes = vec![
        vec![10000],
        vec![100, 100],
        vec![32, 768],
        vec![16, 1024],
        vec![8, 16, 128],
    ];

    for shape in shapes {
        let tensor = create_random_f32_tensor(&shape);

        // Sum operations
        group.bench_with_input(
            BenchmarkId::new("sum_all", format!("{:?}", shape)),
            &tensor,
            |b, tensor| b.iter(|| black_box(tensor.sum(None, false).unwrap())),
        );

        if shape.len() >= 2 {
            group.bench_with_input(
                BenchmarkId::new("sum_axis_last", format!("{:?}", shape)),
                &tensor,
                |b, tensor| {
                    b.iter(|| black_box(tensor.sum_axes(&[tensor.shape().len() - 1]).unwrap()))
                },
            );

            group.bench_with_input(
                BenchmarkId::new("sum_axis_first", format!("{:?}", shape)),
                &tensor,
                |b, tensor| b.iter(|| black_box(tensor.sum_axes(&[0]).unwrap())),
            );
        }

        // Mean operations
        group.bench_with_input(
            BenchmarkId::new("mean_all", format!("{:?}", shape)),
            &tensor,
            |b, tensor| b.iter(|| black_box(tensor.mean().unwrap())),
        );

        if shape.len() >= 2 {
            group.bench_with_input(
                BenchmarkId::new("mean_axis_last", format!("{:?}", shape)),
                &tensor,
                |b, tensor| {
                    b.iter(|| black_box(tensor.mean_axes(&[tensor.shape().len() - 1]).unwrap()))
                },
            );
        }
    }

    group.finish();
}

fn benchmark_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");

    let shapes = vec![vec![1000], vec![100, 100], vec![32, 768], vec![16, 1024]];

    for shape in shapes {
        let tensor = create_random_f32_tensor(&shape);

        // Clone operation
        group.bench_with_input(
            BenchmarkId::new("clone", format!("{:?}", shape)),
            &tensor,
            |b, tensor| b.iter(|| black_box(tensor.clone())),
        );

        // Data extraction
        group.bench_with_input(
            BenchmarkId::new("to_vec", format!("{:?}", shape)),
            &tensor,
            |b, tensor| b.iter(|| black_box(tensor.data())),
        );

        // Shape query
        group.bench_with_input(
            BenchmarkId::new("shape", format!("{:?}", shape)),
            &tensor,
            |b, tensor| b.iter(|| black_box(tensor.shape())),
        );

        // Memory usage calculation
        group.bench_with_input(
            BenchmarkId::new("memory_usage", format!("{:?}", shape)),
            &tensor,
            |b, tensor| b.iter(|| black_box(tensor.memory_usage())),
        );
    }

    group.finish();
}

// Complex tensor operations benchmark (placeholder - complex operations not yet implemented)
fn benchmark_complex_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_operations");

    // Complex operations are not yet fully implemented, so this is a placeholder
    // for future complex number support in tensors
    let tensor = create_random_f32_tensor(&[100, 100]);

    group.bench_function("complex_placeholder", |b| {
        b.iter(|| {
            // Placeholder for complex operations
            black_box(tensor.clone())
        })
    });

    group.finish();
}

// Quantization operations benchmark (placeholder - quantization not fully integrated yet)
fn benchmark_quantization_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_operations");

    // Quantization operations are not yet fully integrated with tensor operations,
    // so this is a placeholder for future quantization benchmarks
    let tensor = create_random_f32_tensor(&[1000, 100]);

    group.bench_function("quantization_placeholder", |b| {
        b.iter(|| {
            // Placeholder for quantization operations
            black_box(tensor.clone())
        })
    });

    group.finish();
}

// Sparse tensor operations benchmark
fn benchmark_sparse_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_operations");

    let shapes = vec![
        vec![100, 100],
        vec![500, 500],
        vec![1000, 1000],
        vec![32, 768],
    ];

    for shape in shapes {
        let dense_tensor = create_random_f32_tensor(&shape);

        // Convert to sparse (with sparsity)
        group.bench_with_input(
            BenchmarkId::new("to_sparse_coo", format!("{:?}", shape)),
            &dense_tensor,
            |b, tensor| {
                b.iter(|| {
                    black_box(tensor.to_sparse(0.1).unwrap()) // 10% sparsity threshold
                })
            },
        );

        // Create a sparse tensor for other operations
        let sparse_tensor = dense_tensor.to_sparse(0.1).unwrap();

        group.bench_with_input(
            BenchmarkId::new("sparse_to_dense", format!("{:?}", shape)),
            &sparse_tensor,
            |b, tensor| b.iter(|| black_box(tensor.to_dense().unwrap())),
        );

        // Sparse matrix multiplication
        if shape.len() == 2 && shape[0] == shape[1] {
            let sparse_b = create_random_f32_tensor(&shape).to_sparse(0.1).unwrap();

            group.bench_with_input(
                BenchmarkId::new("sparse_matmul", format!("{:?}", shape)),
                &(&sparse_tensor, &sparse_b),
                |bench, (a, b)| bench.iter(|| black_box(a.matmul(b).unwrap())),
            );
        }
    }

    group.finish();
}

// Autodiff operations benchmark (placeholder - autodiff not fully implemented yet)
fn benchmark_autodiff_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("autodiff_operations");

    // Autodiff operations are not yet fully implemented, so this is a placeholder
    // for future automatic differentiation benchmarks
    let tensor = create_random_f32_tensor(&[100, 100]);

    group.bench_function("autodiff_placeholder", |b| {
        b.iter(|| {
            // Placeholder for autodiff operations
            black_box(tensor.clone())
        })
    });

    group.finish();
}

// Broadcasting operations benchmark
fn benchmark_broadcasting_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcasting_operations");

    let test_cases = vec![
        (vec![100, 1], vec![100, 50]),       // Column vector + matrix
        (vec![1, 50], vec![100, 50]),        // Row vector + matrix
        (vec![100], vec![100, 50]),          // Vector + matrix
        (vec![1, 1, 50], vec![10, 20, 50]),  // 3D broadcasting
        (vec![10, 1, 50], vec![10, 20, 50]), // Partial 3D broadcasting
    ];

    for (small_shape, large_shape) in test_cases {
        let small_tensor = create_random_f32_tensor(&small_shape);
        let large_tensor = create_random_f32_tensor(&large_shape);

        group.bench_with_input(
            BenchmarkId::new(
                "broadcast_add",
                format!("{:?}+{:?}", small_shape, large_shape),
            ),
            &(&small_tensor, &large_tensor),
            |b, (small, large)| b.iter(|| black_box(small.add(large).unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new(
                "broadcast_mul",
                format!("{:?}*{:?}", small_shape, large_shape),
            ),
            &(&small_tensor, &large_tensor),
            |b, (small, large)| b.iter(|| black_box(small.mul(large).unwrap())),
        );
    }

    group.finish();
}

// Normalization operations benchmark
fn benchmark_normalization_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalization_operations");

    let shapes = vec![
        vec![32, 768], // Common transformer dimensions
        vec![16, 1024],
        vec![8, 2048],
        vec![64, 512],
    ];

    for shape in shapes {
        let tensor = create_random_f32_tensor(&shape);

        // Layer normalization
        group.bench_with_input(
            BenchmarkId::new("layer_norm", format!("{:?}", shape)),
            &tensor,
            |b, tensor| b.iter(|| black_box(tensor.layer_norm(-1, 1e-5).unwrap())),
        );

        // L2 normalization
        group.bench_with_input(
            BenchmarkId::new("l2_norm", format!("{:?}", shape)),
            &tensor,
            |b, tensor| b.iter(|| black_box(tensor.norm().unwrap())),
        );

        // Batch normalization (if implemented)
        if shape.len() >= 2 {
            group.bench_with_input(
                BenchmarkId::new("batch_norm", format!("{:?}", shape)),
                &tensor,
                |b, tensor| {
                    b.iter(|| {
                        // Simulate batch norm: (x - mean) / sqrt(var + eps)
                        let mean = tensor.mean_axis(0).unwrap();
                        let var =
                            tensor.sub(&mean).unwrap().pow(2.0).unwrap().mean_axis(0).unwrap();
                        let normalized = tensor
                            .sub(&mean)
                            .unwrap()
                            .div(&var.add_scalar(1e-5).unwrap().sqrt().unwrap())
                            .unwrap();
                        black_box(normalized)
                    })
                },
            );
        }
    }

    group.finish();
}

// Advanced tensor operations benchmark
fn benchmark_advanced_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_operations");

    let tensor_2d = create_random_f32_tensor(&[64, 128]);
    let tensor_3d = create_random_f32_tensor(&[8, 16, 32]);

    // Advanced indexing and slicing
    group.bench_function("advanced_slice_2d", |b| {
        b.iter(|| black_box(tensor_2d.slice_multi(&[(10, 50), (20, 100)]).unwrap()))
    });

    // Tensor splitting
    group.bench_function("split_2d", |b| {
        b.iter(|| {
            black_box(tensor_2d.split(32, 0).unwrap()) // Split along first dimension
        })
    });

    // Tensor permutation (complex transpose)
    group.bench_function("permute_3d", |b| {
        b.iter(|| black_box(tensor_3d.permute(&[2, 0, 1]).unwrap()))
    });

    // Advanced reduction with keepdims
    group.bench_function("sum_keepdims", |b| {
        b.iter(|| {
            // Simulate keepdims by reshaping after reduction
            let sum = tensor_2d.sum_axis(1).unwrap();
            black_box(sum.reshape(&[64, 1]).unwrap())
        })
    });

    // Tensor comparison operations (if implemented)
    let tensor_a = create_random_f32_tensor(&[100, 100]);
    let tensor_b = create_random_f32_tensor(&[100, 100]);

    group.bench_function("tensor_gt", |b| {
        b.iter(|| {
            // Element-wise comparison simulation
            black_box(tensor_a.sub(&tensor_b).unwrap().relu().unwrap()) // Simulate >
        })
    });

    group.finish();
}

// Enhanced tensor operations benchmark (new operations)
fn benchmark_enhanced_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("enhanced_operations");

    // Classification and similarity shapes
    let prediction_shapes = vec![
        vec![32, 10],    // Small classification
        vec![64, 100],   // Medium classification
        vec![128, 1000], // Large classification (ImageNet-like)
        vec![16, 50257], // Large vocabulary (GPT-like)
    ];

    // Vector similarity shapes
    let vector_shapes = vec![
        vec![100, 128], // Small embeddings
        vec![32, 768],  // BERT-base hidden size
        vec![16, 1024], // Large embeddings
        vec![8, 2048],  // Very large embeddings
    ];

    // Statistical computation shapes
    let stats_shapes = vec![
        vec![1000],        // 1D statistics
        vec![100, 100],    // 2D statistics
        vec![32, 768],     // Transformer dimensions
        vec![16, 16, 256], // 3D statistics
    ];

    // Cross-entropy loss benchmarks
    for shape in &prediction_shapes {
        let predictions = create_random_f32_tensor(shape);
        let num_classes = shape[1];
        let batch_size = shape[0];

        // Create target labels (as f32 for compatibility)
        let targets_data: Vec<f32> = (0..batch_size).map(|i| (i % num_classes) as f32).collect();
        let targets = Tensor::from_vec(targets_data, &[batch_size]).unwrap();

        group.bench_with_input(
            BenchmarkId::new("cross_entropy_mean", format!("{:?}", shape)),
            &(&predictions, &targets),
            |b, (preds, targets)| {
                b.iter(|| black_box(preds.cross_entropy(targets, "mean").unwrap()))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cross_entropy_sum", format!("{:?}", shape)),
            &(&predictions, &targets),
            |b, (preds, targets)| {
                b.iter(|| black_box(preds.cross_entropy(targets, "sum").unwrap()))
            },
        );
    }

    // Cosine similarity benchmarks
    for shape in &vector_shapes {
        let tensor_a = create_random_f32_tensor(shape);
        let tensor_b = create_random_f32_tensor(shape);

        group.bench_with_input(
            BenchmarkId::new("cosine_similarity", format!("{:?}", shape)),
            &(&tensor_a, &tensor_b),
            |bench, (a, b)| bench.iter(|| black_box(a.cosine_similarity(b, -1, 1e-8).unwrap())),
        );
    }

    // Log softmax benchmarks
    for shape in &prediction_shapes {
        let tensor = create_random_f32_tensor(shape);

        group.bench_with_input(
            BenchmarkId::new("log_softmax", format!("{:?}", shape)),
            &tensor,
            |b, tensor| b.iter(|| black_box(tensor.log_softmax(-1).unwrap())),
        );
    }

    // Variance benchmarks
    for shape in &stats_shapes {
        let tensor = create_random_f32_tensor(shape);

        // All axes variance
        group.bench_with_input(
            BenchmarkId::new("variance_all", format!("{:?}", shape)),
            &tensor,
            |b, tensor| b.iter(|| black_box(tensor.variance(None, false).unwrap())),
        );

        // Last axis variance (common for feature statistics)
        if shape.len() > 1 {
            let last_axis = shape.len() - 1;
            group.bench_with_input(
                BenchmarkId::new("variance_last_axis", format!("{:?}", shape)),
                &tensor,
                |b, tensor| {
                    b.iter(|| black_box(tensor.variance(Some(&[last_axis]), false).unwrap()))
                },
            );
        }
    }

    // Standard deviation benchmarks
    for shape in &stats_shapes {
        let tensor = create_random_f32_tensor(shape);

        // All axes standard deviation
        group.bench_with_input(
            BenchmarkId::new("std_dev_all", format!("{:?}", shape)),
            &tensor,
            |b, tensor| b.iter(|| black_box(tensor.std_dev(None, false).unwrap())),
        );

        // Last axis standard deviation
        if shape.len() > 1 {
            let last_axis = shape.len() - 1;
            group.bench_with_input(
                BenchmarkId::new("std_dev_last_axis", format!("{:?}", shape)),
                &tensor,
                |b, tensor| {
                    b.iter(|| black_box(tensor.std_dev(Some(&[last_axis]), false).unwrap()))
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_tensor_creation,
    benchmark_element_wise_operations,
    benchmark_matrix_operations,
    benchmark_activation_functions,
    benchmark_shape_operations,
    benchmark_reduction_operations,
    benchmark_memory_operations,
    benchmark_complex_operations,
    benchmark_quantization_operations,
    benchmark_sparse_operations,
    benchmark_autodiff_operations,
    benchmark_broadcasting_operations,
    benchmark_normalization_operations,
    benchmark_advanced_operations,
    benchmark_enhanced_operations
);

criterion_main!(benches);
