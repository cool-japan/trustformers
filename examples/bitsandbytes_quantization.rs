//! Example demonstrating BitsAndBytes-compatible quantization
#![allow(unused_variables)]

use trustformers_core::{
    Tensor,
    BitsAndBytesConfig, quantize_int8, quantize_4bit, dequantize_bitsandbytes,
    to_bitsandbytes_format, from_bitsandbytes_format,
};
use trustformers_models::llama::{LlamaConfig, LlamaModel};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== BitsAndBytes Quantization Example ===\n");

    // Example 1: Basic INT8 quantization
    basic_int8_quantization()?;

    // Example 2: 4-bit quantization (NF4)
    nf4_quantization()?;

    // Example 3: Dynamic tree quantization
    dynamic_tree_quantization()?;

    // Example 4: Model weight quantization
    model_weight_quantization()?;

    // Example 5: BitsAndBytes format conversion
    format_conversion()?;

    Ok(())
}

fn basic_int8_quantization() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic INT8 Quantization");
    println!("-" * 40);

    // Create a sample tensor
    let tensor = Tensor::randn(&[128, 768])?;
    let original_size = 128 * 768 * 4; // float32

    // Configure INT8 quantization
    let config = BitsAndBytesConfig {
        bits: 8,
        dynamic_tree: false,
        block_size: 256,
        stochastic: false,
        outlier_threshold: 0.99,
        nested_quantization: false,
    };

    // Quantize the tensor
    let start = Instant::now();
    let quantized = quantize_int8(&tensor, &config)?;
    let quantize_time = start.elapsed();

    // Calculate compression ratio
    let quantized_size = quantized.data.shape().iter().product::<usize>();
    let compression_ratio = original_size as f32 / quantized_size as f32;

    println!("Original tensor shape: {:?}", tensor.shape());
    println!("Quantized data shape: {:?}", quantized.data.shape());
    println!("Number of scale factors: {}", quantized.scale.shape()[0]);
    println!("Compression ratio: {:.2}x", compression_ratio);
    println!("Quantization time: {:?}", quantize_time);

    // Dequantize and check reconstruction error
    let start = Instant::now();
    let dequantized = dequantize_bitsandbytes(&quantized, &config)?;
    let dequantize_time = start.elapsed();

    let error = tensor.sub(&dequantized)?.abs()?.mean()?;
    println!("Reconstruction error (MAE): {:.6}", error.to_vec_f32()?[0]);
    println!("Dequantization time: {:?}", dequantize_time);
    println!();

    Ok(())
}

fn nf4_quantization() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. 4-bit Quantization (NF4)");
    println!("-" * 40);

    // Create a sample tensor with normal distribution
    let tensor = Tensor::randn(&[64, 512])?;
    let original_size = 64 * 512 * 4; // float32

    // Configure 4-bit quantization
    let config = BitsAndBytesConfig {
        bits: 4,
        dynamic_tree: false,
        block_size: 128,
        stochastic: false,
        outlier_threshold: 0.99,
        nested_quantization: true, // Quantize scales too
    };

    // Quantize the tensor
    let start = Instant::now();
    let quantized = quantize_4bit(&tensor, &config)?;
    let quantize_time = start.elapsed();

    // Calculate effective bits per element
    let quantized_bytes = quantized.data.shape().iter().product::<usize>();
    let scale_bytes = quantized.scale.shape().iter().product::<usize>() * 4;
    let total_bytes = quantized_bytes + scale_bytes;
    let bits_per_element = (total_bytes * 8) as f32 / (64 * 512) as f32;

    println!("Original tensor shape: {:?}", tensor.shape());
    println!("Quantized data shape: {:?}", quantized.data.shape());
    println!("Number of blocks: {}", quantized.scale.shape()[0]);
    println!("Original size: {} bytes", original_size);
    println!("Quantized size: {} bytes", total_bytes);
    println!("Effective bits per element: {:.2}", bits_per_element);
    println!("Compression ratio: {:.2}x", original_size as f32 / total_bytes as f32);
    println!("Quantization time: {:?}", quantize_time);

    // Dequantize and check quality
    let dequantized = dequantize_bitsandbytes(&quantized, &config)?;
    let error = tensor.sub(&dequantized)?.abs()?.mean()?;
    println!("Reconstruction error (MAE): {:.6}", error.to_vec_f32()?[0]);

    // Calculate signal-to-noise ratio
    let signal_power = tensor.pow(2.0)?.mean()?;
    let noise_power = tensor.sub(&dequantized)?.pow(2.0)?.mean()?;
    let snr = 10.0 * (signal_power.to_vec_f32()?[0] / noise_power.to_vec_f32()?[0]).log10();
    println!("Signal-to-Noise Ratio: {:.2} dB", snr);
    println!();

    Ok(())
}

fn dynamic_tree_quantization() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Dynamic Tree Quantization");
    println!("-" * 40);

    // Create a tensor with non-uniform distribution
    let uniform = Tensor::rand(&[32, 256])?;
    let gaussian = Tensor::randn(&[32, 256])?;
    let tensor = uniform.mul_scalar(0.5)?.add(&gaussian.mul_scalar(2.0)?)?;

    // Configure dynamic tree quantization
    let config = BitsAndBytesConfig {
        bits: 8,
        dynamic_tree: true,
        block_size: 256,
        stochastic: false,
        outlier_threshold: 0.99,
        nested_quantization: false,
    };

    // Quantize using dynamic tree
    let start = Instant::now();
    let quantized = quantize_dynamic_tree(&tensor, &config)?;
    let quantize_time = start.elapsed();

    println!("Original tensor shape: {:?}", tensor.shape());
    println!("Tree structure stored in scale tensor");
    println!("Scale tensor shape: {:?}", quantized.scale.shape());
    println!("Quantization time: {:?}", quantize_time);

    // Compare with standard quantization
    let standard_config = BitsAndBytesConfig {
        dynamic_tree: false,
        ..config
    };
    let standard_quantized = quantize_int8(&tensor, &standard_config)?;

    // Dequantize both
    let tree_dequantized = dequantize_bitsandbytes(&quantized, &config)?;
    let standard_dequantized = dequantize_bitsandbytes(&standard_quantized, &standard_config)?;

    // Compare errors
    let tree_error = tensor.sub(&tree_dequantized)?.abs()?.mean()?;
    let standard_error = tensor.sub(&standard_dequantized)?.abs()?.mean()?;

    println!("\nReconstruction errors:");
    println!("  Dynamic tree: {:.6}", tree_error.to_vec_f32()?[0]);
    println!("  Standard:     {:.6}", standard_error.to_vec_f32()?[0]);

    let improvement = (1.0 - tree_error.to_vec_f32()?[0] / standard_error.to_vec_f32()?[0]) * 100.0;
    println!("  Improvement:  {:.1}%", improvement);
    println!();

    Ok(())
}

fn model_weight_quantization() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Model Weight Quantization");
    println!("-" * 40);

    // Create a small model for demonstration
    let config = LlamaConfig {
        vocab_size: 1000,
        hidden_size: 256,
        intermediate_size: 512,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        ..Default::default()
    };

    println!("Creating small LLaMA model...");
    let model = LlamaModel::new(config)?;

    // Quantize model weights
    let bnb_config = BitsAndBytesConfig {
        bits: 8,
        dynamic_tree: false,
        block_size: 256,
        stochastic: false,
        outlier_threshold: 0.99,
        nested_quantization: false,
    };

    // Simulate quantizing a weight matrix
    let weight_tensor = Tensor::randn(&[256, 256])?;
    let original_bytes = 256 * 256 * 4;

    println!("\nQuantizing weight matrix [{}, {}]...", 256, 256);
    let start = Instant::now();
    let quantized = quantize_int8(&weight_tensor, &bnb_config)?;
    let quantize_time = start.elapsed();

    let quantized_bytes = quantized.data.shape().iter().product::<usize>() +
                         quantized.scale.shape().iter().product::<usize>() * 4;

    println!("Original size: {} KB", original_bytes / 1024);
    println!("Quantized size: {} KB", quantized_bytes / 1024);
    println!("Compression ratio: {:.2}x", original_bytes as f32 / quantized_bytes as f32);
    println!("Quantization time: {:?}", quantize_time);

    // Detect outliers
    if let Some(outliers) = &quantized.outliers {
        println!("Number of outliers detected: {} ({:.2}%)",
                outliers.len(),
                outliers.len() as f32 / (256 * 256) as f32 * 100.0);
    }
    println!();

    Ok(())
}

fn format_conversion() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. BitsAndBytes Format Conversion");
    println!("-" * 40);

    // Create a tensor
    let tensor = Tensor::randn(&[64, 128])?;

    // Configure quantization
    let config = BitsAndBytesConfig {
        bits: 8,
        dynamic_tree: false,
        block_size: 256,
        stochastic: false,
        outlier_threshold: 0.95, // Lower threshold to generate outliers
        nested_quantization: false,
    };

    // Convert to bitsandbytes format
    println!("Converting to bitsandbytes format...");
    let start = Instant::now();
    let bnb_format = to_bitsandbytes_format(&tensor, &config)?;
    let convert_time = start.elapsed();

    println!("BitsAndBytes format components:");
    for (key, tensor) in &bnb_format {
        println!("  {}: shape {:?}, dtype {:?}", key, tensor.shape(), tensor.dtype());
    }
    println!("Conversion time: {:?}", convert_time);

    // Convert back from bitsandbytes format
    println!("\nConverting back from bitsandbytes format...");
    let start = Instant::now();
    let reconstructed = from_bitsandbytes_format(bnb_format, &config)?;
    let reconstruct_time = start.elapsed();

    println!("Reconstructed shape: {:?}", reconstructed.shape());
    println!("Reconstruction time: {:?}", reconstruct_time);

    // Check fidelity
    let error = tensor.sub(&reconstructed)?.abs()?.mean()?;
    println!("Round-trip error (MAE): {:.6}", error.to_vec_f32()?[0]);

    // Performance summary
    println!("\nPerformance Summary:");
    println!("  Forward conversion: {:?}", convert_time);
    println!("  Backward conversion: {:?}", reconstruct_time);
    println!("  Total round-trip: {:?}", convert_time + reconstruct_time);

    Ok(())
}

// Helper function to print tensor statistics
fn print_tensor_stats(name: &str, tensor: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let mean = tensor.mean()?.to_vec_f32()?[0];
    let std = tensor.std()?.to_vec_f32()?[0];
    let min = tensor.min()?.to_vec_f32()?[0];
    let max = tensor.max()?.to_vec_f32()?[0];

    println!("{} statistics:", name);
    println!("  Mean: {:.4}", mean);
    println!("  Std:  {:.4}", std);
    println!("  Min:  {:.4}", min);
    println!("  Max:  {:.4}", max);

    Ok(())
}