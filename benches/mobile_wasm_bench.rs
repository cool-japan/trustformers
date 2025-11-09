//! Benchmarks for mobile and WebAssembly performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};

#[cfg(feature = "mobile")]
use trustformers_mobile::{
    MobileOptimizer, OptimizationLevel,
    DeviceProfile, ThermalState,
    quantize_for_mobile, optimize_graph_for_mobile,
};

#[cfg(feature = "wasm")]
use trustformers_wasm::{
    WasmTensor, WasmModel,
    WebGPUBackend, SimdBackend,
};

use trustformers_core::tensor::Tensor;
use trustformers_models::bert::{BertConfig, BertModel};

#[cfg(feature = "mobile")]
fn mobile_optimization_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("mobile_optimization");

    // Create a small model suitable for mobile
    let config = BertConfig {
        vocab_size: 30522,
        hidden_size: 256,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        intermediate_size: 1024,
        ..Default::default()
    };

    let model = BertModel::new(config).expect("Failed to create model");

    let optimization_levels = vec![
        ("baseline", OptimizationLevel::None),
        ("basic", OptimizationLevel::Basic),
        ("aggressive", OptimizationLevel::Aggressive),
        ("extreme", OptimizationLevel::Extreme),
    ];

    for (name, opt_level) in optimization_levels {
        group.bench_with_input(
            BenchmarkId::new("optimize_model", name),
            &(&model, opt_level),
            |b, (model, opt_level)| {
                b.iter(|| {
                    let optimizer = MobileOptimizer::new(*opt_level);
                    let optimized = optimizer.optimize_model(model);
                    black_box(optimized)
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "mobile")]
fn mobile_quantization_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("mobile_quantization");

    let tensor_sizes = vec![
        ("small", vec![64, 64]),
        ("medium", vec![256, 256]),
        ("large", vec![512, 512]),
    ];

    for (name, shape) in tensor_sizes {
        let tensor = Tensor::randn(&shape).expect("Failed to create tensor");

        let num_elements: usize = shape.iter().product();
        group.throughput(Throughput::Elements(num_elements as u64));

        group.bench_with_input(
            BenchmarkId::new("quantize_int8", name),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    let quantized = quantize_for_mobile(tensor, QuantizationType::Int8);
                    black_box(quantized)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("quantize_int4", name),
            &tensor,
            |b, tensor| {
                b.iter(|| {
                    let quantized = quantize_for_mobile(tensor, QuantizationType::Int4);
                    black_box(quantized)
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "mobile")]
fn device_adaptation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_adaptation");

    let device_profiles = vec![
        ("low_end", DeviceProfile {
            cpu_cores: 4,
            ram_mb: 2048,
            gpu_available: false,
            npu_available: false,
        }),
        ("mid_range", DeviceProfile {
            cpu_cores: 6,
            ram_mb: 4096,
            gpu_available: true,
            npu_available: false,
        }),
        ("high_end", DeviceProfile {
            cpu_cores: 8,
            ram_mb: 8192,
            gpu_available: true,
            npu_available: true,
        }),
    ];

    let model = create_test_model();

    for (name, profile) in device_profiles {
        group.bench_with_input(
            BenchmarkId::new("adapt_model", name),
            &(&model, profile),
            |b, (model, profile)| {
                b.iter(|| {
                    let adapted = adapt_model_for_device(model, profile);
                    black_box(adapted)
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "mobile")]
fn thermal_throttling_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("thermal_throttling");

    let thermal_states = vec![
        ("cool", ThermalState::Cool),
        ("nominal", ThermalState::Nominal),
        ("warm", ThermalState::Warm),
        ("hot", ThermalState::Hot),
        ("critical", ThermalState::Critical),
    ];

    let input = Tensor::randn(&[1, 128, 256]).expect("Failed to create input");
    let model = create_test_model();

    for (name, thermal_state) in thermal_states {
        group.bench_with_input(
            BenchmarkId::new("inference", name),
            &(&model, &input, thermal_state),
            |b, (model, input, thermal_state)| {
                b.iter(|| {
                    let runtime = MobileRuntime::new();
                    runtime.set_thermal_state(*thermal_state);
                    let output = runtime.run_inference(model, input);
                    black_box(output)
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "wasm")]
fn wasm_tensor_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("wasm_tensor");

    let sizes = vec![
        ("small", vec![64, 64]),
        ("medium", vec![256, 256]),
        ("large", vec![512, 512]),
    ];

    for (name, shape) in sizes {
        let num_elements: usize = shape.iter().product();
        group.throughput(Throughput::Elements(num_elements as u64));

        // Creation benchmarks
        group.bench_with_input(
            BenchmarkId::new("create", name),
            &shape,
            |b, shape| {
                b.iter(|| {
                    let tensor = WasmTensor::zeros(shape);
                    black_box(tensor)
                })
            },
        );

        // Operation benchmarks
        let a = WasmTensor::randn(&shape);
        let b = WasmTensor::randn(&shape);

        group.bench_with_input(
            BenchmarkId::new("add", name),
            &(&a, &b),
            |b, (a, b)| {
                b.iter(|| {
                    let result = a.add(b);
                    black_box(result)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("matmul", name),
            &(&a, &b),
            |b, (a, b)| {
                b.iter(|| {
                    let result = a.matmul(b);
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "wasm")]
fn wasm_backend_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("wasm_backend");

    let tensor_size = vec![256, 256];
    let a = Tensor::randn(&tensor_size).expect("Failed to create tensor");
    let b = Tensor::randn(&tensor_size).expect("Failed to create tensor");

    // CPU backend (baseline)
    group.bench_function("cpu_matmul", |bench| {
        bench.iter(|| {
            let result = a.matmul(&b);
            black_box(result)
        })
    });

    // SIMD backend
    if SimdBackend::is_available() {
        let simd_backend = SimdBackend::new();
        let a_simd = simd_backend.from_tensor(&a);
        let b_simd = simd_backend.from_tensor(&b);

        group.bench_function("simd_matmul", |bench| {
            bench.iter(|| {
                let result = simd_backend.matmul(&a_simd, &b_simd);
                black_box(result)
            })
        });
    }

    // WebGPU backend
    if WebGPUBackend::is_available() {
        let gpu_backend = WebGPUBackend::new().expect("Failed to create WebGPU backend");
        let a_gpu = gpu_backend.from_tensor(&a);
        let b_gpu = gpu_backend.from_tensor(&b);

        group.bench_function("webgpu_matmul", |bench| {
            bench.iter(|| {
                let result = gpu_backend.matmul(&a_gpu, &b_gpu);
                black_box(result)
            })
        });
    }

    group.finish();
}

#[cfg(feature = "wasm")]
fn wasm_model_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("wasm_model");

    // Small model configuration for WASM
    let config = BertConfig {
        vocab_size: 10000,
        hidden_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        intermediate_size: 512,
        ..Default::default()
    };

    let model = WasmModel::from_config(config).expect("Failed to create WASM model");

    let sequence_lengths = vec![32, 64, 128];

    for seq_len in sequence_lengths {
        let input_ids = vec![1; seq_len];

        group.throughput(Throughput::Elements(seq_len as u64));

        group.bench_with_input(
            BenchmarkId::new("inference", seq_len),
            &input_ids,
            |b, input_ids| {
                b.iter(|| {
                    let output = model.forward(input_ids);
                    black_box(output)
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "wasm")]
fn wasm_memory_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("wasm_memory");

    // Test memory allocation patterns in WASM
    let allocation_sizes = vec![
        ("1kb", 1024),
        ("64kb", 65536),
        ("1mb", 1048576),
        ("10mb", 10485760),
    ];

    for (name, size) in allocation_sizes {
        group.bench_with_input(
            BenchmarkId::new("allocate", name),
            &size,
            |b, &size| {
                b.iter(|| {
                    let buffer = WasmBuffer::allocate(size);
                    black_box(buffer)
                })
            },
        );
    }

    // Test memory copy operations
    let src_size = 1048576; // 1MB
    let src_buffer = WasmBuffer::allocate(src_size);

    group.bench_function("memory_copy_1mb", |b| {
        b.iter(|| {
            let dst_buffer = src_buffer.clone();
            black_box(dst_buffer)
        })
    });

    group.finish();
}

// Helper functions
#[cfg(any(feature = "mobile", feature = "wasm"))]
fn create_test_model() -> BertModel {
    let config = BertConfig {
        vocab_size: 10000,
        hidden_size: 256,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        intermediate_size: 1024,
        ..Default::default()
    };

    BertModel::new(config).expect("Failed to create test model")
}

// Conditional compilation for benchmark groups
#[cfg(all(feature = "mobile", feature = "wasm"))]
criterion_group!(
    benches,
    mobile_optimization_benchmarks,
    mobile_quantization_benchmarks,
    device_adaptation_benchmarks,
    thermal_throttling_benchmarks,
    wasm_tensor_benchmarks,
    wasm_backend_benchmarks,
    wasm_model_benchmarks,
    wasm_memory_benchmarks
);

#[cfg(all(feature = "mobile", not(feature = "wasm")))]
criterion_group!(
    benches,
    mobile_optimization_benchmarks,
    mobile_quantization_benchmarks,
    device_adaptation_benchmarks,
    thermal_throttling_benchmarks
);

#[cfg(all(not(feature = "mobile"), feature = "wasm"))]
criterion_group!(
    benches,
    wasm_tensor_benchmarks,
    wasm_backend_benchmarks,
    wasm_model_benchmarks,
    wasm_memory_benchmarks
);

#[cfg(not(any(feature = "mobile", feature = "wasm")))]
criterion_group!(benches,);

criterion_main!(benches);