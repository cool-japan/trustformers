//! Benchmarks for quantization performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use trustformers_core::{
    tensor::Tensor,
    quantization::{
        QuantizationConfig, QuantizationType,
        quantize_tensor, dequantize_tensor,
        quantize_model, QuantizedLinear,
    },
};
use trustformers_models::bert::{BertConfig, BertModel};

fn tensor_quantization_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_quantization");

    let quantization_types = vec![
        ("int8", QuantizationType::Int8),
        ("uint8", QuantizationType::UInt8),
        ("int4", QuantizationType::Int4),
        ("fp16", QuantizationType::FP16),
        ("bf16", QuantizationType::BF16),
    ];

    for (name, quant_type) in quantization_types {
        for size in [256, 512, 1024, 2048].iter() {
            let shape = vec![*size, *size];
            let tensor = Tensor::randn(&shape).expect("Failed to create tensor");

            let num_elements = size * size;
            group.throughput(Throughput::Elements(num_elements as u64));

            // Quantization benchmark
            group.bench_with_input(
                BenchmarkId::new(format!("{}_quantize", name), size),
                &tensor,
                |b, tensor| {
                    b.iter(|| {
                        let (quantized, scale, zero_point) = quantize_tensor(tensor, quant_type);
                        black_box((quantized, scale, zero_point))
                    })
                },
            );

            // Dequantization benchmark
            let (quantized, scale, zero_point) = quantize_tensor(&tensor, quant_type)
                .expect("Failed to quantize");

            group.bench_with_input(
                BenchmarkId::new(format!("{}_dequantize", name), size),
                &(quantized, scale, zero_point),
                |b, (quantized, scale, zero_point)| {
                    b.iter(|| {
                        let dequantized = dequantize_tensor(quantized, *scale, *zero_point, quant_type);
                        black_box(dequantized)
                    })
                },
            );
        }
    }

    group.finish();
}

fn quantized_matmul_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantized_matmul");

    for size in [256, 512, 1024].iter() {
        let a = Tensor::randn(&[*size, *size]).expect("Failed to create tensor");
        let b = Tensor::randn(&[*size, *size]).expect("Failed to create tensor");

        // Quantize tensors
        let (a_q, a_scale, a_zp) = quantize_tensor(&a, QuantizationType::Int8)
            .expect("Failed to quantize");
        let (b_q, b_scale, b_zp) = quantize_tensor(&b, QuantizationType::Int8)
            .expect("Failed to quantize");

        let num_ops = 2 * size * size * size; // Matrix multiplication ops
        group.throughput(Throughput::Elements(num_ops as u64));

        // Regular matmul
        group.bench_with_input(
            BenchmarkId::new("fp32_matmul", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let result = a.matmul(b);
                    black_box(result)
                })
            },
        );

        // Quantized matmul
        group.bench_with_input(
            BenchmarkId::new("int8_matmul", size),
            &(&a_q, &b_q, a_scale, a_zp, b_scale, b_zp),
            |bench, (a_q, b_q, a_scale, a_zp, b_scale, b_zp)| {
                bench.iter(|| {
                    // Quantized matrix multiplication
                    let result = quantized_matmul(a_q, b_q, *a_scale, *a_zp, *b_scale, *b_zp);
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

fn quantized_linear_layer_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantized_linear");

    let configs = vec![
        ("small", 768, 768),
        ("medium", 768, 3072),
        ("large", 1024, 4096),
    ];

    for (name, in_features, out_features) in configs {
        // Create linear layer
        let weight = Tensor::randn(&[out_features, in_features]).expect("Failed to create weight");
        let bias = Tensor::randn(&[out_features]).expect("Failed to create bias");

        // Create quantized linear layer
        let quantized_linear = QuantizedLinear::from_float(
            weight.clone(),
            Some(bias.clone()),
            QuantizationType::Int8,
        ).expect("Failed to create quantized linear");

        let batch_size = 32;
        let input = Tensor::randn(&[batch_size, in_features]).expect("Failed to create input");

        let num_ops = batch_size * in_features * out_features;
        group.throughput(Throughput::Elements(num_ops as u64));

        // Float linear
        group.bench_with_input(
            BenchmarkId::new("float_linear", name),
            &(&input, &weight, &bias),
            |b, (input, weight, bias)| {
                b.iter(|| {
                    let output = input.matmul(&weight.transpose()) + bias;
                    black_box(output)
                })
            },
        );

        // Quantized linear
        group.bench_with_input(
            BenchmarkId::new("quantized_linear", name),
            &(&input, &quantized_linear),
            |b, (input, quantized_linear)| {
                b.iter(|| {
                    let output = quantized_linear.forward(input);
                    black_box(output)
                })
            },
        );
    }

    group.finish();
}

fn model_quantization_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_quantization");

    let config = BertConfig {
        vocab_size: 30522,
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        ..Default::default()
    };

    let model = BertModel::new(config).expect("Failed to create model");

    let quant_configs = vec![
        ("int8_symmetric", QuantizationConfig {
            quantization_type: QuantizationType::Int8,
            symmetric: true,
            per_channel: false,
            calibration_method: CalibrationMethod::MinMax,
        }),
        ("int8_asymmetric", QuantizationConfig {
            quantization_type: QuantizationType::Int8,
            symmetric: false,
            per_channel: false,
            calibration_method: CalibrationMethod::MinMax,
        }),
        ("int8_per_channel", QuantizationConfig {
            quantization_type: QuantizationType::Int8,
            symmetric: true,
            per_channel: true,
            calibration_method: CalibrationMethod::MinMax,
        }),
        ("int4_symmetric", QuantizationConfig {
            quantization_type: QuantizationType::Int4,
            symmetric: true,
            per_channel: false,
            calibration_method: CalibrationMethod::MinMax,
        }),
    ];

    for (name, quant_config) in quant_configs {
        group.bench_with_input(
            BenchmarkId::new("quantize_model", name),
            &(&model, &quant_config),
            |b, (model, quant_config)| {
                b.iter(|| {
                    let quantized_model = quantize_model(model, quant_config);
                    black_box(quantized_model)
                })
            },
        );
    }

    group.finish();
}

fn dynamic_quantization_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_quantization");

    // Test dynamic quantization during inference
    let sizes = vec![
        ("small", 256),
        ("medium", 512),
        ("large", 1024),
    ];

    for (name, size) in sizes {
        let input = Tensor::randn(&[1, size, size]).expect("Failed to create input");

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("dynamic_quant", name),
            &input,
            |b, input| {
                b.iter(|| {
                    // Simulate dynamic quantization
                    let (quantized, scale, zero_point) = quantize_tensor(input, QuantizationType::Int8)
                        .expect("Failed to quantize");

                    // Perform some operation (e.g., a simple transformation)
                    let processed = quantized.add(&quantized);

                    // Dequantize back
                    let result = dequantize_tensor(&processed, scale * 2.0, zero_point, QuantizationType::Int8);
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

fn mixed_precision_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_precision");

    let sizes = vec![512, 1024, 2048];

    for size in sizes {
        let a_fp32 = Tensor::randn(&[size, size]).expect("Failed to create tensor");
        let b_fp32 = Tensor::randn(&[size, size]).expect("Failed to create tensor");

        // Convert to FP16
        let a_fp16 = a_fp32.to_dtype(DType::F16);
        let b_fp16 = b_fp32.to_dtype(DType::F16);

        // Convert to BF16
        let a_bf16 = a_fp32.to_dtype(DType::BF16);
        let b_bf16 = b_fp32.to_dtype(DType::BF16);

        let num_ops = 2 * size * size * size;
        group.throughput(Throughput::Elements(num_ops as u64));

        // FP32 baseline
        group.bench_with_input(
            BenchmarkId::new("fp32", size),
            &(&a_fp32, &b_fp32),
            |b, (a, b)| {
                b.iter(|| {
                    let result = a.matmul(b);
                    black_box(result)
                })
            },
        );

        // FP16
        group.bench_with_input(
            BenchmarkId::new("fp16", size),
            &(&a_fp16, &b_fp16),
            |b, (a, b)| {
                b.iter(|| {
                    let result = a.matmul(b);
                    black_box(result)
                })
            },
        );

        // BF16
        group.bench_with_input(
            BenchmarkId::new("bf16", size),
            &(&a_bf16, &b_bf16),
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

// Helper function for quantized matmul (simplified)
fn quantized_matmul(
    a: &Tensor,
    b: &Tensor,
    a_scale: f32,
    a_zp: i32,
    b_scale: f32,
    b_zp: i32,
) -> Tensor {
    // Simplified quantized matmul implementation
    let result = a.matmul(b);
    let output_scale = a_scale * b_scale;
    result.mul_scalar(output_scale)
}

criterion_group!(
    benches,
    tensor_quantization_benchmarks,
    quantized_matmul_benchmarks,
    quantized_linear_layer_benchmarks,
    model_quantization_benchmarks,
    dynamic_quantization_benchmarks,
    mixed_precision_benchmarks
);
criterion_main!(benches);