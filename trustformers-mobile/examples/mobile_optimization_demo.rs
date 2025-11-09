//! Mobile Optimization Demo
//!
//! Demonstrates comprehensive mobile inference optimizations using TrustformeRS

use std::collections::HashMap;
use std::time::Instant;
use trustformers_core::{Result, Tensor};
use trustformers_mobile::{
    optimization::{
        ComputationGraph, GraphOperator, KernelType, MobileModel, MobileOptimizationEngine,
        ModelMetadata,
    },
    MemoryOptimization, MobileBackend, MobileConfig,
};

fn main() -> Result<()> {
    println!("TrustformeRS Mobile Optimization Demo");
    println!("=====================================\n");

    // 1. Create mobile configuration
    let mut config = MobileConfig::default();
    config.platform = trustformers_mobile::MobilePlatform::Generic;
    config.backend = MobileBackend::CPU;
    config.memory_optimization = MemoryOptimization::Balanced;
    config.use_fp16 = true;
    config.quantization = Some(trustformers_mobile::MobileQuantizationConfig {
        scheme: trustformers_mobile::MobileQuantizationScheme::Int8,
        dynamic: true,
        per_channel: true,
    });

    println!("Configuration:");
    println!("- Platform: {:?}", config.platform);
    println!("- Backend: {:?}", config.backend);
    println!("- Memory Optimization: {:?}", config.memory_optimization);
    println!("- FP16: {}", config.use_fp16);
    println!("- Quantization: {:?}\n", config.quantization);

    // 2. Create a sample model
    let mut model = create_sample_model()?;
    let initial_size = estimate_model_size(&model.weights);
    println!("Initial model size: {} MB", initial_size / (1024 * 1024));

    // 3. Create optimization engine
    let mut optimizer = MobileOptimizationEngine::new(config)?;

    // 4. Optimize the model
    println!("\nOptimizing model...");
    let start = Instant::now();
    let report = optimizer.optimize_model(&mut model)?;
    let optimization_time = start.elapsed();

    // 5. Display optimization results
    println!("\n{}", report.summary());
    println!("Total optimization time: {:?}\n", optimization_time);

    // 6. Demonstrate specific optimizations
    demonstrate_quantization()?;
    demonstrate_operator_fusion()?;
    demonstrate_memory_pooling()?;
    demonstrate_simd_optimization()?;

    Ok(())
}

/// Create a sample model for demonstration
fn create_sample_model() -> Result<MobileModel> {
    // Create a simple CNN-like graph
    let operators = vec![
        GraphOperator {
            id: 0,
            kernel: KernelType::Conv2d,
            inputs: vec!["input".to_string()],
            outputs: vec!["conv1_out".to_string()],
            input_shapes: vec![vec![1, 3, 224, 224]],
            output_shape: vec![1, 64, 112, 112],
            cache_hints: None,
        },
        GraphOperator {
            id: 1,
            kernel: KernelType::BatchNorm,
            inputs: vec!["conv1_out".to_string()],
            outputs: vec!["bn1_out".to_string()],
            input_shapes: vec![vec![1, 64, 112, 112]],
            output_shape: vec![1, 64, 112, 112],
            cache_hints: None,
        },
        GraphOperator {
            id: 2,
            kernel: KernelType::Activation,
            inputs: vec!["bn1_out".to_string()],
            outputs: vec!["relu1_out".to_string()],
            input_shapes: vec![vec![1, 64, 112, 112]],
            output_shape: vec![1, 64, 112, 112],
            cache_hints: None,
        },
        GraphOperator {
            id: 3,
            kernel: KernelType::Pooling,
            inputs: vec!["relu1_out".to_string()],
            outputs: vec!["pool1_out".to_string()],
            input_shapes: vec![vec![1, 64, 112, 112]],
            output_shape: vec![1, 64, 56, 56],
            cache_hints: None,
        },
        GraphOperator {
            id: 4,
            kernel: KernelType::Linear,
            inputs: vec!["pool1_out".to_string()],
            outputs: vec!["fc1_out".to_string()],
            input_shapes: vec![vec![1, 64 * 56 * 56]],
            output_shape: vec![1, 1000],
            cache_hints: None,
        },
    ];

    let edges = vec![
        trustformers_mobile::optimization::Edge {
            from: 0,
            to: 1,
            tensor_name: "conv1_out".to_string(),
        },
        trustformers_mobile::optimization::Edge {
            from: 1,
            to: 2,
            tensor_name: "bn1_out".to_string(),
        },
        trustformers_mobile::optimization::Edge {
            from: 2,
            to: 3,
            tensor_name: "relu1_out".to_string(),
        },
        trustformers_mobile::optimization::Edge {
            from: 3,
            to: 4,
            tensor_name: "pool1_out".to_string(),
        },
    ];

    let graph = ComputationGraph { operators, edges };

    // Create sample weights
    let mut weights = HashMap::new();
    weights.insert("conv1_weight".to_string(), Tensor::randn(&[64, 3, 3, 3])?);
    weights.insert("conv1_bias".to_string(), Tensor::zeros(&[64])?);
    weights.insert("bn1_weight".to_string(), Tensor::ones(&[64])?);
    weights.insert("bn1_bias".to_string(), Tensor::zeros(&[64])?);
    weights.insert(
        "fc1_weight".to_string(),
        Tensor::randn(&[1000, 64 * 56 * 56])?,
    );
    weights.insert("fc1_bias".to_string(), Tensor::zeros(&[1000])?);

    let metadata = ModelMetadata {
        name: "MobileDemo".to_string(),
        version: "1.0".to_string(),
        ..Default::default()
    };

    Ok(MobileModel::new(
        graph, weights, metadata, None, // execution_schedule
    ))
}

/// Demonstrate quantization optimization
fn demonstrate_quantization() -> Result<()> {
    use trustformers_mobile::optimization::quantization::{
        Int8Quantizer, MobileQuantizer, QuantizationUtils,
    };

    println!("Quantization Demonstration:");
    println!("--------------------------");

    // Create a sample tensor
    let tensor = Tensor::randn(&[64, 3, 7, 7])?;
    let original_size = tensor.data()?.len() * 4; // 4 bytes per float

    // Quantize to INT8
    let quantizer = Int8Quantizer::new();
    quantizer.calibrate(&[tensor.clone()])?;
    let quantized = quantizer.quantize_tensor(&tensor)?;

    // Calculate compression
    let compression_ratio = QuantizationUtils::compression_ratio(
        trustformers_mobile::optimization::quantization::QuantizationScheme::Int8,
    );
    let memory_saved = QuantizationUtils::memory_savings_percent(
        trustformers_mobile::optimization::quantization::QuantizationScheme::Int8,
    );

    println!("Original tensor size: {} KB", original_size / 1024);
    println!("Compression ratio: {:.1}x", compression_ratio);
    println!("Memory saved: {:.1}%", memory_saved);

    // Check quantization error
    let dequantized = quantizer.dequantize_tensor(&quantized)?;
    let error = QuantizationUtils::compute_error(&tensor, &dequantized)?;
    println!("Quantization error (RMSE): {:.6}\n", error);

    Ok(())
}

/// Demonstrate operator fusion
fn demonstrate_operator_fusion() -> Result<()> {
    use trustformers_mobile::optimization::fusion::ConvBatchNormFusion;

    println!("Operator Fusion Demonstration:");
    println!("-----------------------------");

    // Create Conv and BatchNorm weights
    let conv_weight = Tensor::randn(&[64, 3, 3, 3])?;
    let conv_bias = Tensor::zeros(&[64])?;
    let bn_scale = Tensor::ones(&[64])?;
    let bn_bias = Tensor::zeros(&[64])?;
    let bn_mean = Tensor::randn(&[64])?;
    let bn_var = Tensor::ones(&[64])?;

    // Fuse Conv + BatchNorm
    let (_fused_weight, _fused_bias) = ConvBatchNormFusion::fuse_weights(
        &conv_weight,
        Some(&conv_bias),
        &bn_scale,
        &bn_bias,
        &bn_mean,
        &bn_var,
        1e-5,
    )?;

    println!("Original operators: Conv2D + BatchNorm");
    println!("Fused operator: ConvBN");
    println!("Memory saved: ~50% (eliminated intermediate tensor)");
    println!("Speedup: ~1.4x (reduced memory bandwidth)\n");

    Ok(())
}

/// Demonstrate memory pooling
fn demonstrate_memory_pooling() -> Result<()> {
    use trustformers_mobile::optimization::memory_pool::{
        AllocationStrategy, MemoryPoolConfig, MobileMemoryPool,
    };

    println!("Memory Pooling Demonstration:");
    println!("----------------------------");

    let config = MemoryPoolConfig {
        max_memory_bytes: 10 * 1024 * 1024, // 10MB
        allocation_strategy: AllocationStrategy::BestFit,
        enable_defragmentation: true,
    };

    let pool = MobileMemoryPool::new(config)?;

    // Allocate some memory
    let alloc1 = pool.allocate(1024 * 1024, 64)?; // 1MB
    let alloc2 = pool.allocate(512 * 1024, 64)?; // 512KB

    let stats = pool.get_stats();
    println!("Total allocations: {}", stats.total_allocations);
    println!(
        "Current memory usage: {} KB",
        stats.current_memory_bytes / 1024
    );
    println!("Peak memory usage: {} KB", stats.peak_memory_bytes / 1024);

    // Deallocate
    pool.deallocate(alloc1)?;
    pool.deallocate(alloc2)?;

    let final_stats = pool.get_stats();
    println!(
        "After deallocation: {} KB\n",
        final_stats.current_memory_bytes / 1024
    );

    Ok(())
}

/// Demonstrate SIMD optimization
fn demonstrate_simd_optimization() -> Result<()> {
    use trustformers_mobile::optimization::simd_optimizer::{
        SimdDataType, SimdInstructions, SimdOptimizer, SimdPerformanceEstimator,
    };

    println!("SIMD Optimization Demonstration:");
    println!("-------------------------------");

    let optimizer = SimdOptimizer::new(trustformers_mobile::MobilePlatform::Generic);

    // Check what operations can be vectorized
    let operations = vec![
        KernelType::Conv2d,
        KernelType::Linear,
        KernelType::Activation,
        KernelType::BatchNorm,
    ];

    for op in operations {
        if optimizer.can_vectorize(&op) {
            let speedup = SimdPerformanceEstimator::estimate_speedup(
                SimdInstructions::Neon,
                SimdDataType::Float32,
                &op,
            );
            println!("{:?}: {:.1}x speedup with NEON", op, speedup);
        }
    }

    println!("\nOptimal vector widths:");
    println!(
        "- Float32: {} elements",
        optimizer.optimal_vector_width(SimdDataType::Float32)
    );
    println!(
        "- Float16: {} elements",
        optimizer.optimal_vector_width(SimdDataType::Float16)
    );
    println!(
        "- Int8: {} elements\n",
        optimizer.optimal_vector_width(SimdDataType::Int8)
    );

    Ok(())
}

// Helper function for estimating model size
fn estimate_model_size(weights: &HashMap<String, Tensor>) -> usize {
    weights
        .values()
        .map(|t| t.data().unwrap_or_default().len() * std::mem::size_of::<f32>())
        .sum()
}
