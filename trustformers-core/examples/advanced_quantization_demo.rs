//! Demonstration of advanced quantization methods: SmoothQuant and GGML Q5/Q6 variants
#![allow(unused_variables)]

use scirs2_core::random::*;

use anyhow::Result;
use trustformers_core::quantization::{
    AdvancedGGMLQuantizer, GGMLQuantType, MigrationAnalyzer, SmoothQuantConfig, SmoothQuantizer,
};
use trustformers_core::tensor::Tensor;

fn main() -> Result<()> {
    println!("TrustformeRS Advanced Quantization Demo");
    println!("======================================\n");

    // Example 1: SmoothQuant demonstration
    println!("Example 1: SmoothQuant W8A8 Quantization");
    demo_smoothquant()?;

    // Example 2: GGML Q5 variants
    println!("\nExample 2: GGML Q5 Quantization");
    demo_ggml_q5()?;

    // Example 3: GGML Q6_K quantization
    println!("\nExample 3: GGML Q6_K Quantization");
    demo_ggml_q6k()?;

    // Example 4: SmoothQuant migration analysis
    println!("\nExample 4: SmoothQuant Migration Analysis");
    demo_migration_analysis()?;

    // Example 5: Compression comparison
    println!("\nExample 5: Compression Ratio Comparison");
    demo_compression_comparison()?;

    Ok(())
}

fn demo_smoothquant() -> Result<()> {
    println!("Creating test weight and activation tensors...");

    // Create a linear layer's weights (simulating a 768x768 transformer layer)
    let weights = Tensor::randn(&[768, 768])?;

    // Create activation tensors with outliers (common in LLMs)
    let mut activations = Vec::new();
    for _ in 0..10 {
        let mut act = Tensor::randn(&[32, 768])?;
        // Add some outliers
        if let Tensor::F32(data) = &mut act {
            if let Some(values) = data.as_slice_mut() {
                if values.len() > 200 {
                    values[100] = 50.0; // Outlier
                    values[200] = -45.0; // Outlier
                }
            }
        }
        activations.push(act);
    }

    // Configure SmoothQuant
    let config = SmoothQuantConfig {
        alpha: 0.5, // Balanced migration
        num_calibration_samples: 10,
        activation_percentile: 99.9,
        per_channel: true,
        migration_strength: 0.8,
        quantize_activations: true,
    };

    let mut quantizer = SmoothQuantizer::new(config);

    // Calibrate and quantize
    println!("Calibrating SmoothQuant...");
    let quantized_layer =
        quantizer.quantize_linear_layer("transformer.layer.0", &weights, &activations)?;

    // Test forward pass
    println!("Testing quantized forward pass...");
    let test_input = Tensor::randn(&[1, 768])?;
    let output = quantized_layer.forward(&test_input)?;

    println!("✓ SmoothQuant successfully applied!");
    println!("  - Input shape: {:?}", test_input.shape());
    println!("  - Output shape: {:?}", output.shape());
    println!("  - Activation quantization: enabled");

    // Calculate memory savings
    let original_size = 768 * 768 * 4; // f32
    let quantized_size = 768 * 768; // int8
    let compression_ratio = original_size as f32 / quantized_size as f32;
    println!("  - Compression ratio: {:.1}x", compression_ratio);

    Ok(())
}

fn demo_ggml_q5() -> Result<()> {
    println!("Creating test tensor for Q5 quantization...");

    // Create a weight tensor
    let original = Tensor::randn(&[2048, 768])?;

    // Test Q5_0 quantization
    println!("\nQ5_0 Quantization:");
    let quantizer_q5_0 = AdvancedGGMLQuantizer::new(GGMLQuantType::Q5_0);
    let quantized_q5_0 = quantizer_q5_0.quantize(&original)?;

    let memory_q5_0 = quantized_q5_0.memory_usage();
    let original_memory = 2048 * 768 * 4;
    let compression_q5_0 = quantizer_q5_0.compression_ratio(2048 * 768);

    println!(
        "  - Original size: {:.2} MB",
        original_memory as f32 / 1024.0 / 1024.0
    );
    println!(
        "  - Q5_0 size: {:.2} MB",
        memory_q5_0 as f32 / 1024.0 / 1024.0
    );
    println!("  - Compression ratio: {:.2}x", compression_q5_0);
    println!(
        "  - Bits per weight: {:.1}",
        GGMLQuantType::Q5_0.bits_per_weight()
    );

    // Test Q5_1 quantization
    println!("\nQ5_1 Quantization:");
    let quantizer_q5_1 = AdvancedGGMLQuantizer::new(GGMLQuantType::Q5_1);
    let quantized_q5_1 = quantizer_q5_1.quantize(&original)?;

    let memory_q5_1 = quantized_q5_1.memory_usage();
    let compression_q5_1 = quantizer_q5_1.compression_ratio(2048 * 768);

    println!(
        "  - Q5_1 size: {:.2} MB",
        memory_q5_1 as f32 / 1024.0 / 1024.0
    );
    println!("  - Compression ratio: {:.2}x", compression_q5_1);
    println!("  - Additional precision from minimum values");

    // Test reconstruction quality
    println!("\nTesting reconstruction quality...");
    let reconstructed = quantized_q5_0.dequantize()?;
    let error = calculate_reconstruction_error(&original, &reconstructed)?;
    println!("  - Q5_0 reconstruction RMSE: {:.4}", error);

    Ok(())
}

fn demo_ggml_q6k() -> Result<()> {
    println!("Creating large tensor for Q6_K super-block quantization...");

    // Q6_K uses 256-element super-blocks
    let original = Tensor::randn(&[4096, 4096])?;

    let quantizer = AdvancedGGMLQuantizer::new(GGMLQuantType::Q6K);
    let quantized = quantizer.quantize(&original)?;

    let memory_usage = quantized.memory_usage();
    let original_memory = 4096 * 4096 * 4;
    let compression_ratio = quantizer.compression_ratio(4096 * 4096);

    println!("Q6_K Quantization Results:");
    println!(
        "  - Original size: {:.2} MB",
        original_memory as f32 / 1024.0 / 1024.0
    );
    println!(
        "  - Q6_K size: {:.2} MB",
        memory_usage as f32 / 1024.0 / 1024.0
    );
    println!("  - Compression ratio: {:.2}x", compression_ratio);
    println!(
        "  - Bits per weight: {:.2}",
        GGMLQuantType::Q6K.bits_per_weight()
    );
    println!(
        "  - Super-block size: {} elements",
        GGMLQuantType::Q6K.block_size()
    );

    // Test reconstruction
    let reconstructed = quantized.dequantize()?;
    let error = calculate_reconstruction_error(&original, &reconstructed)?;
    println!("  - Reconstruction RMSE: {:.4}", error);
    println!("  - Quality: Higher than Q4/Q5 variants");

    Ok(())
}

fn demo_migration_analysis() -> Result<()> {
    println!("Analyzing optimal alpha values for SmoothQuant migration...");

    // Create test layer
    let weights = Tensor::randn(&[1024, 1024])?;
    let mut activations = Vec::new();

    // Create activations with varying outlier patterns
    for i in 0..20 {
        let mut act = Tensor::randn(&[32, 1024])?;

        // Add outliers with increasing magnitude
        if let Tensor::F32(data) = &mut act {
            if let Some(values) = data.as_slice_mut() {
                for j in 0..i {
                    if j * 100 < values.len() {
                        values[j * 100] = (i as f32) * 5.0; // Increasing outliers
                    }
                }
            }
        }
        activations.push(act);
    }

    // Create migration analyzer
    let analyzer = MigrationAnalyzer::new("accuracy");

    // Define evaluation function (simplified - in practice would evaluate on real task)
    let eval_fn = |quantized: &trustformers_core::quantization::SmoothQuantizedLinear| -> f32 {
        // Simulate accuracy score based on quantization
        // In practice, this would run actual model evaluation
        let test_input = Tensor::randn(&[1, 1024]).unwrap();
        match quantized.forward(&test_input) {
            Ok(_) => 0.95 - thread_rng().random::<f32>() * 0.1, // Simulated accuracy
            Err(_) => 0.0,
        }
    };

    // Find optimal alpha
    println!("Testing different alpha values...");
    let optimal_alpha = analyzer.find_optimal_alpha(&weights, &activations, eval_fn)?;

    println!("✓ Migration analysis complete!");
    println!("  - Optimal alpha: {:.2}", optimal_alpha);
    println!("  - Interpretation:");
    if optimal_alpha < 0.3 {
        println!("    → Low alpha: Most quantization difficulty on weights");
        println!("    → Good for models with activation outliers");
    } else if optimal_alpha > 0.7 {
        println!("    → High alpha: Most quantization difficulty on activations");
        println!("    → Good for models with smooth activations");
    } else {
        println!("    → Balanced alpha: Equal distribution of difficulty");
        println!("    → Good general-purpose setting");
    }

    Ok(())
}

fn demo_compression_comparison() -> Result<()> {
    println!("Comparing compression ratios across quantization methods...");

    let tensor_size = 1024 * 1024; // 1M parameters
    let tensor = Tensor::randn(&[1024, 1024])?;

    // Calculate sizes for different methods
    let original_mb = (tensor_size * 4) as f32 / 1024.0 / 1024.0;

    println!("\nModel size comparison (1M parameters):");
    println!("  Original (FP32):     {:>6.2} MB", original_mb);
    println!("  ----------------");

    // INT8 quantization
    let int8_mb = tensor_size as f32 / 1024.0 / 1024.0;
    println!(
        "  INT8:                {:>6.2} MB  ({:.1}x compression)",
        int8_mb,
        original_mb / int8_mb
    );

    // INT4 quantization
    let int4_mb = (tensor_size / 2) as f32 / 1024.0 / 1024.0;
    println!(
        "  INT4:                {:>6.2} MB  ({:.1}x compression)",
        int4_mb,
        original_mb / int4_mb
    );

    // Q5_0 quantization
    let q5_0_quantizer = AdvancedGGMLQuantizer::new(GGMLQuantType::Q5_0);
    let q5_0_ratio = q5_0_quantizer.compression_ratio(tensor_size);
    let q5_0_mb = original_mb / q5_0_ratio;
    println!(
        "  GGML Q5_0:           {:>6.2} MB  ({:.1}x compression)",
        q5_0_mb, q5_0_ratio
    );

    // Q6_K quantization
    let q6_k_quantizer = AdvancedGGMLQuantizer::new(GGMLQuantType::Q6K);
    let q6_k_ratio = q6_k_quantizer.compression_ratio(tensor_size);
    let q6_k_mb = original_mb / q6_k_ratio;
    println!(
        "  GGML Q6_K:           {:>6.2} MB  ({:.1}x compression)",
        q6_k_mb, q6_k_ratio
    );

    // SmoothQuant W8A8
    let smoothquant_mb = int8_mb; // Weights are INT8
    println!(
        "  SmoothQuant W8A8:    {:>6.2} MB  ({:.1}x compression)",
        smoothquant_mb,
        original_mb / smoothquant_mb
    );

    println!("\nQuality vs Compression Trade-offs:");
    println!("  INT4:        Highest compression, lowest quality");
    println!("  Q5_0/Q5_1:   Good compression, better quality than INT4");
    println!("  Q6_K:        Moderate compression, high quality");
    println!("  INT8:        Lower compression, good quality");
    println!("  SmoothQuant: INT8 compression + activation quantization");

    Ok(())
}

fn calculate_reconstruction_error(original: &Tensor, reconstructed: &Tensor) -> Result<f32> {
    match (original, reconstructed) {
        (Tensor::F32(orig_data), Tensor::F32(recon_data)) => {
            let orig_values = orig_data
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("Failed to get original data"))?;
            let recon_values = recon_data
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("Failed to get reconstructed data"))?;

            let mse: f32 = orig_values
                .iter()
                .zip(recon_values.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                / orig_values.len() as f32;

            Ok(mse.sqrt())
        },
        _ => Err(anyhow::anyhow!("Unsupported tensor type")),
    }
}

// Helper function to demonstrate quantization on a real model layer
#[allow(dead_code)]
fn quantize_transformer_layer() -> Result<()> {
    println!("\n=== Bonus: Quantizing a Transformer Layer ===");

    // Simulate transformer layer weights
    let q_weight = Tensor::randn(&[768, 768])?;
    let k_weight = Tensor::randn(&[768, 768])?;
    let v_weight = Tensor::randn(&[768, 768])?;
    let o_weight = Tensor::randn(&[768, 768])?;

    // Use Q6_K for attention weights (higher quality needed)
    let attention_quantizer = AdvancedGGMLQuantizer::new(GGMLQuantType::Q6K);

    let q_quantized = attention_quantizer.quantize(&q_weight)?;
    let k_quantized = attention_quantizer.quantize(&k_weight)?;
    let v_quantized = attention_quantizer.quantize(&v_weight)?;

    // Use Q5_0 for output projection (can tolerate more compression)
    let output_quantizer = AdvancedGGMLQuantizer::new(GGMLQuantType::Q5_0);
    let o_quantized = output_quantizer.quantize(&o_weight)?;

    let original_size = 4 * 768 * 768 * 4; // 4 matrices, f32
    let quantized_size = q_quantized.memory_usage() * 3 + o_quantized.memory_usage();

    println!("Transformer attention layer quantization:");
    println!(
        "  - Original size: {:.2} MB",
        original_size as f32 / 1024.0 / 1024.0
    );
    println!(
        "  - Quantized size: {:.2} MB",
        quantized_size as f32 / 1024.0 / 1024.0
    );
    println!(
        "  - Overall compression: {:.1}x",
        original_size as f32 / quantized_size as f32
    );
    println!("  - Strategy: Q6_K for Q/K/V (quality), Q5_0 for O (efficiency)");

    Ok(())
}
