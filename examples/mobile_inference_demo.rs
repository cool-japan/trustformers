//! Mobile Inference Demo
#![allow(unused_variables)]
//!
//! This example demonstrates how to use TrustformeRS for mobile deployment
//! with optimizations for iOS and Android platforms.

use std::collections::HashMap;
use trustformers_core::Tensor;
use trustformers_mobile::{
    MobileInferenceBuilder, MobileConfig, MobilePlatform, MobileBackend,
    MobileQuantizationScheme, MemoryOptimization,
    optimization::MobileInferenceOptimizer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 TrustformeRS Mobile Inference Demo");
    println!("=====================================\n");

    // Demo 1: Basic mobile inference
    demo_basic_mobile_inference()?;

    // Demo 2: Platform-specific optimization
    demo_platform_optimization()?;

    // Demo 3: Memory optimization strategies
    demo_memory_optimization()?;

    // Demo 4: Batch inference on mobile
    demo_batch_inference()?;

    // Demo 5: Memory monitoring
    demo_memory_monitoring()?;

    println!("✅ All mobile inference demos completed successfully!");

    Ok(())
}

fn demo_basic_mobile_inference() -> Result<(), Box<dyn std::error::Error>> {
    println!("📱 Demo 1: Basic Mobile Inference");
    println!("----------------------------------");

    // Create mobile inference engine with default settings
    let mut engine = MobileInferenceBuilder::new()
        .memory_limit_mb(512)
        .fp16(true)
        .quantization(MobileQuantizationScheme::Int8)
        .threads(4)
        .build()?;

    // Create a simple model (in practice, would load actual model weights)
    let mut model_weights = HashMap::new();
    model_weights.insert("embedding".to_string(), Tensor::randn(&[1000, 256])?);
    model_weights.insert("transformer_layer_0".to_string(), Tensor::randn(&[256, 256])?);
    model_weights.insert("transformer_layer_1".to_string(), Tensor::randn(&[256, 256])?);
    model_weights.insert("output_projection".to_string(), Tensor::randn(&[256, 1000])?);

    // Load model
    engine.load_model(model_weights)?;

    // Create input tensor
    let input = Tensor::randn(&[1, 128])?; // Batch size 1, sequence length 128

    // Perform inference
    let output = engine.inference(&input)?;

    println!("✅ Input shape: {:?}", input.shape());
    println!("✅ Output shape: {:?}", output.shape());

    // Get performance statistics
    let stats = engine.get_stats();
    println!("📊 Inference time: {:.2}ms", stats.avg_inference_time_ms);
    println!("💾 Memory usage: {}MB", stats.memory_usage_mb);

    let memory_info = engine.get_memory_info();
    println!("💾 Memory utilization: {:.1}%", memory_info.memory_utilization_percent());
    println!("💾 Memory savings: {:.1}%", memory_info.memory_savings_percent);

    println!();
    Ok(())
}

fn demo_platform_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏗️ Demo 2: Platform-Specific Optimization");
    println!("-------------------------------------------");

    // iOS optimized configuration
    println!("📱 iOS Configuration:");
    let ios_config = MobileConfig::ios_optimized();
    println!("   Platform: {:?}", ios_config.platform);
    println!("   Backend: {:?}", ios_config.backend);
    println!("   Memory limit: {}MB", ios_config.max_memory_mb);
    println!("   Batching enabled: {}", ios_config.enable_batching);

    let mut ios_engine = MobileInferenceBuilder::new()
        .platform(MobilePlatform::iOS)
        .backend(MobileBackend::CoreML)
        .memory_limit_mb(1024)
        .fp16(true)
        .quantization(MobileQuantizationScheme::FP16)
        .batching(true, 4)
        .build()?;

    // Android optimized configuration
    println!("\n🤖 Android Configuration:");
    let android_config = MobileConfig::android_optimized();
    println!("   Platform: {:?}", android_config.platform);
    println!("   Backend: {:?}", android_config.backend);
    println!("   Memory limit: {}MB", android_config.max_memory_mb);
    println!("   Batching enabled: {}", android_config.enable_batching);

    let mut android_engine = MobileInferenceBuilder::new()
        .platform(MobilePlatform::Android)
        .backend(MobileBackend::NNAPI)
        .memory_limit_mb(768)
        .fp16(true)
        .quantization(MobileQuantizationScheme::Int8)
        .batching(false, 1)
        .build()?;

    // Load the same model on both platforms
    let mut model_weights = HashMap::new();
    model_weights.insert("layer1".to_string(), Tensor::randn(&[128, 128])?);
    model_weights.insert("layer2".to_string(), Tensor::randn(&[128, 64])?);

    ios_engine.load_model(model_weights.clone())?;
    android_engine.load_model(model_weights)?;

    // Compare performance
    let input = Tensor::randn(&[1, 64])?;

    let ios_output = ios_engine.inference(&input)?;
    let android_output = android_engine.inference(&input)?;

    println!("\n📊 Performance Comparison:");
    println!("   iOS inference time: {:.2}ms", ios_engine.get_stats().avg_inference_time_ms);
    println!("   Android inference time: {:.2}ms", android_engine.get_stats().avg_inference_time_ms);
    println!("   iOS memory usage: {}MB", ios_engine.get_memory_info().total_memory_mb);
    println!("   Android memory usage: {}MB", android_engine.get_memory_info().total_memory_mb);

    println!();
    Ok(())
}

fn demo_memory_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Demo 3: Memory Optimization Strategies");
    println!("------------------------------------------");

    // Create configurations for different optimization levels
    let configs = vec![
        ("Battery Life", create_battery_optimized_config()),
        ("Balanced", create_balanced_config()),
        ("Performance", create_performance_config()),
        ("Ultra Low Memory", create_ultra_low_memory_config()),
    ];

    for (name, config) in configs {
        println!("🔧 {} Configuration:", name);
        println!("   Memory optimization: {:?}", config.memory_optimization);
        println!("   Memory limit: {}MB", config.max_memory_mb);
        println!("   Threads: {}", config.get_thread_count());
        println!("   FP16: {}", config.use_fp16);

        if let Some(ref quant) = config.quantization {
            println!("   Quantization: {:?}", quant.scheme);
        }

        // Create engine and test memory footprint
        let mut engine = MobileInferenceBuilder::new()
            .memory_optimization(config.memory_optimization)
            .memory_limit_mb(config.max_memory_mb)
            .fp16(config.use_fp16)
            .threads(config.get_thread_count())
            .build()?;

        // Load model and check memory usage
        let model_weights = create_test_model(1000); // 1K parameters
        engine.load_model(model_weights)?;

        let memory_info = engine.get_memory_info();
        println!("   Memory footprint: {}MB", memory_info.total_memory_mb);
        println!("   Memory savings: {:.1}%", memory_info.memory_savings_percent);
        println!();
    }

    Ok(())
}

fn demo_batch_inference() -> Result<(), Box<dyn std::error::Error>> {
    println!("📦 Demo 4: Batch Inference on Mobile");
    println!("-------------------------------------");

    // Create engine with batching enabled
    let mut engine = MobileInferenceBuilder::new()
        .memory_limit_mb(512)
        .batching(true, 4)
        .fp16(true)
        .quantization(MobileQuantizationScheme::Int8)
        .build()?;

    // Load model
    let model_weights = create_test_model(500);
    engine.load_model(model_weights)?;

    // Create batch inputs
    let batch_inputs = vec![
        Tensor::randn(&[32])?,
        Tensor::randn(&[32])?,
        Tensor::randn(&[32])?,
        Tensor::randn(&[32])?,
        Tensor::randn(&[32])?, // This should be limited by max_batch_size
    ];

    println!("📝 Input batch size: {}", batch_inputs.len());

    // Perform batch inference
    let batch_outputs = engine.batch_inference(batch_inputs)?;

    println!("📤 Output batch size: {}", batch_outputs.len());
    println!("📊 Average inference time: {:.2}ms", engine.get_stats().avg_inference_time_ms);
    println!("💾 Memory usage: {}MB", engine.get_stats().memory_usage_mb);

    println!();
    Ok(())
}

fn demo_memory_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Demo 5: Memory Monitoring");
    println!("-----------------------------");

    // Create engine with memory monitoring
    let mut engine = MobileInferenceBuilder::new()
        .memory_limit_mb(256) // Low memory limit for demo
        .memory_optimization(MemoryOptimization::Maximum)
        .fp16(true)
        .quantization(MobileQuantizationScheme::Int4)
        .build()?;

    // Load progressively larger models and monitor memory
    let model_sizes = vec![100, 500, 1000, 2000];

    for &size in &model_sizes {
        println!("🧪 Testing model with {} parameters:", size);

        let model_weights = create_test_model(size);

        match engine.load_model(model_weights) {
            Ok(_) => {
                let memory_info = engine.get_memory_info();
                println!("   ✅ Model loaded successfully");
                println!("   💾 Memory usage: {}MB / {}MB",
                    memory_info.total_memory_mb, memory_info.memory_limit_mb);
                println!("   📊 Memory utilization: {:.1}%", memory_info.memory_utilization_percent());
                println!("   💡 Available memory: {}MB", memory_info.available_memory_mb());

                // Test inference
                let input = Tensor::randn(&[16])?;
                let _output = engine.inference(&input)?;

                println!("   🚀 Inference completed in {:.2}ms",
                    engine.get_stats().avg_inference_time_ms);
            }
            Err(e) => {
                println!("   ❌ Model too large for memory limit: {}", e);
                break;
            }
        }

        println!();
    }

    // Demonstrate memory management operations
    println!("🧹 Memory Management Operations:");

    // Clear cache
    engine.clear_cache();
    println!("   ✅ Cache cleared");

    // Force garbage collection
    engine.force_gc();
    println!("   ✅ Garbage collection completed");

    let final_memory = engine.get_memory_info();
    println!("   💾 Final memory usage: {}MB", final_memory.total_memory_mb);

    Ok(())
}

// Helper functions

fn create_battery_optimized_config() -> MobileConfig {
    let mut config = MobileConfig::default();
    MobileInferenceOptimizer::optimize_for_battery_life(&mut config);
    config
}

fn create_balanced_config() -> MobileConfig {
    let mut config = MobileConfig::default();
    MobileInferenceOptimizer::optimize_balanced(&mut config);
    config
}

fn create_performance_config() -> MobileConfig {
    let mut config = MobileConfig::default();
    MobileInferenceOptimizer::optimize_for_speed(&mut config);
    config
}

fn create_ultra_low_memory_config() -> MobileConfig {
    let mut config = MobileConfig::default();
    MobileInferenceOptimizer::optimize_for_memory(&mut config);
    config
}

fn create_test_model(param_count: usize) -> HashMap<String, Tensor> {
    let mut weights = HashMap::new();

    // Create layers that sum to approximately param_count parameters
    let layer_size = (param_count as f64).sqrt() as usize;

    weights.insert("layer1".to_string(),
        Tensor::randn(&[layer_size, layer_size]).unwrap());

    if param_count > layer_size * layer_size {
        let remaining = param_count - layer_size * layer_size;
        weights.insert("layer2".to_string(),
            Tensor::randn(&[remaining]).unwrap());
    }

    weights
}