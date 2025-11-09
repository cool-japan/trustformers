use std::path::PathBuf;
#[allow(unused_variables)]
use std::time::Instant;
use trustformers::pipeline::tensorrt_backend::{
    DeviceType, ExecutionStrategy, MemoryPoolType, OptimizationLevel, PrecisionMode,
};
use trustformers::pipeline::{
    pipeline, tensorrt_text_classification_pipeline, tensorrt_text_generation_pipeline, Backend,
    Device, Pipeline, PipelineOptions, TensorRTBackendConfig, TensorRTPipelineManager,
    TensorRTPipelineOptions,
};
use trustformers::AutoTokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 TrustformeRS TensorRT Backend Demo");
    println!("====================================");

    // Example model paths (in real usage, these would be actual ONNX/TensorRT model files)
    let model_path = PathBuf::from("./models/bert-base-uncased.onnx");
    let generation_model_path = PathBuf::from("./models/gpt2.onnx");

    // Test inputs with different characteristics
    let test_inputs = vec![
        ("This is fantastic!", "Simple positive sentiment"),
        ("I absolutely love this amazing product with its incredible features and outstanding quality.", "Complex positive sentiment"),
        ("The service was okay, not terrible but not great either.", "Neutral sentiment"),
        ("This product is terrible and I hate it completely.", "Negative sentiment"),
        ("The performance is excellent and the interface is very user-friendly.", "Positive technical review"),
    ];

    println!("\n📊 Testing Different TensorRT Configurations");
    println!("===========================================");

    // 1. Latency-optimized configuration
    println!("\n1. 🏃 Latency-Optimized Configuration (FP16)");
    demo_latency_optimized(&model_path, &test_inputs)?;

    // 2. Throughput-optimized configuration
    println!("\n2. 🚀 Throughput-Optimized Configuration (Batch Size 8)");
    demo_throughput_optimized(&model_path, &test_inputs)?;

    // 3. Memory-optimized configuration
    println!("\n3. 💾 Memory-Optimized Configuration (INT8)");
    demo_memory_optimized(&model_path, &test_inputs)?;

    // 4. Production configuration
    println!("\n4. 🏭 Production Configuration");
    demo_production_config(&model_path, &test_inputs)?;

    // 5. Custom configuration with profiling
    println!("\n5. 🔧 Custom Configuration with Profiling");
    demo_custom_config(&model_path, &test_inputs)?;

    // 6. Text generation with TensorRT
    println!("\n6. 📝 Text Generation with TensorRT");
    demo_text_generation(&generation_model_path)?;

    // 7. Precision mode comparison
    println!("\n7. 🎯 Precision Mode Comparison");
    demo_precision_comparison(&model_path, &test_inputs)?;

    // 8. Pipeline manager demonstration
    println!("\n8. 🎛️ Pipeline Manager Demo");
    demo_pipeline_manager(&model_path, &generation_model_path)?;

    // 9. Unified pipeline API with TensorRT backend
    println!("\n9. 🔗 Unified Pipeline API with TensorRT Backend");
    demo_unified_api(&model_path)?;

    // 10. Benchmarking
    println!("\n10. 📈 TensorRT Benchmarking");
    demo_benchmarking(&model_path, &test_inputs)?;

    println!("\n✅ TensorRT Backend Demo Complete!");
    println!("================================");

    Ok(())
}

fn demo_latency_optimized(
    model_path: &PathBuf,
    test_inputs: &[(&str, &str)],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Creating latency-optimized TensorRT pipeline...");

    // Create latency-optimized config
    let config = TensorRTBackendConfig::latency_optimized(model_path.clone());

    // Load tokenizer
    let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;

    // Create TensorRT pipeline
    let pipeline = tensorrt_text_classification_pipeline(model_path, tokenizer, Some(config))?;

    // Test with various inputs
    for (input_text, description) in test_inputs {
        let start = Instant::now();
        let result = pipeline.__call__(input_text.to_string())?;
        let duration = start.elapsed();

        println!("   📝 {}: {}", description, input_text);
        println!("      ⏱️ Time: {:.2}ms", duration.as_millis());
        println!("      📊 Result: {:?}", result);

        // Get performance metrics
        let performance = pipeline.performance_metrics()?;
        println!("      🚀 Performance: {:?}", performance);

        // Get memory info
        let memory_info = pipeline.memory_info()?;
        println!("      💾 Memory: {:?}", memory_info);
        println!();
    }

    Ok(())
}

fn demo_throughput_optimized(
    model_path: &PathBuf,
    test_inputs: &[(&str, &str)],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Creating throughput-optimized TensorRT pipeline...");

    let config = TensorRTBackendConfig::throughput_optimized(model_path.clone(), 8);
    let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;

    let pipeline = tensorrt_text_classification_pipeline(model_path, tokenizer, Some(config))?;

    // Test batch processing
    let batch_inputs: Vec<String> =
        test_inputs.iter().map(|(input, _)| input.to_string()).collect();

    let start = Instant::now();
    let results = pipeline.batch(batch_inputs)?;
    let duration = start.elapsed();

    println!("   📦 Batch processing {} inputs", test_inputs.len());
    println!("   ⏱️ Total time: {:.2}ms", duration.as_millis());
    println!(
        "   📊 Average time per item: {:.2}ms",
        duration.as_millis() as f64 / test_inputs.len() as f64
    );

    for (i, result) in results.iter().enumerate() {
        println!("   📝 Input {}: {:?}", i + 1, result);
    }

    Ok(())
}

fn demo_memory_optimized(
    model_path: &PathBuf,
    test_inputs: &[(&str, &str)],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Creating memory-optimized TensorRT pipeline...");

    let config = TensorRTBackendConfig::memory_optimized(model_path.clone());
    let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;

    let pipeline = tensorrt_text_classification_pipeline(model_path, tokenizer, Some(config))?;

    // Test a representative input
    let test_input = test_inputs[0].0;
    let start = Instant::now();
    let result = pipeline.__call__(test_input.to_string())?;
    let duration = start.elapsed();

    println!("   📝 Input: {}", test_input);
    println!("   ⏱️ Time: {:.2}ms", duration.as_millis());
    println!("   📊 Result: {:?}", result);

    let memory_info = pipeline.memory_info()?;
    println!("   💾 Memory Usage: {:?}", memory_info);
    println!("   🎯 Precision: INT8 (Memory Optimized)");

    Ok(())
}

fn demo_production_config(
    model_path: &PathBuf,
    test_inputs: &[(&str, &str)],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Creating production-ready TensorRT pipeline...");

    let config = TensorRTBackendConfig::production(model_path.clone(), 4);
    let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;

    let pipeline = tensorrt_text_classification_pipeline(model_path, tokenizer, Some(config))?;

    // Test with batch processing
    let batch_inputs: Vec<String> =
        test_inputs.iter().take(4).map(|(input, _)| input.to_string()).collect();

    let start = Instant::now();
    let results = pipeline.batch(batch_inputs)?;
    let duration = start.elapsed();

    println!("   📦 Production batch processing: {} items", results.len());
    println!("   ⏱️ Time: {:.2}ms", duration.as_millis());
    println!("   🎯 Precision: FP16 (Production Optimized)");

    // Show device information
    let device_info = pipeline.device_info()?;
    println!("   🖥️ Device Info: {:?}", device_info);

    Ok(())
}

fn demo_custom_config(
    model_path: &PathBuf,
    test_inputs: &[(&str, &str)],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Creating custom TensorRT configuration...");

    let profile_path = PathBuf::from("/tmp/tensorrt_profile.json");
    let cache_path = PathBuf::from("/tmp/tensorrt_cache");

    let config = TensorRTBackendConfig {
        model_path: model_path.clone(),
        device_type: DeviceType::GPU,
        device_id: 0,
        precision_mode: PrecisionMode::FP16,
        max_batch_size: 2,
        workspace_size: 512 * 1024 * 1024, // 512MB
        memory_pool_type: MemoryPoolType::Workspace,
        execution_strategy: ExecutionStrategy::Balanced,
        optimization_level: OptimizationLevel::O3,
        enable_fp16: true,
        enable_profiling: true,
        profile_output_path: Some(profile_path),
        engine_cache_path: Some(cache_path),
        // log_level: LogLevel::Info, // TODO: Fix LogLevel import
        ..Default::default()
    };

    let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;
    let pipeline = tensorrt_text_classification_pipeline(model_path, tokenizer, Some(config))?;

    // Test with profiling enabled
    let test_input = test_inputs[0].0;
    let start = Instant::now();
    let result = pipeline.__call__(test_input.to_string())?;
    let duration = start.elapsed();

    println!("   📝 Input: {}", test_input);
    println!("   ⏱️ Time: {:.2}ms", duration.as_millis());
    println!("   📊 Result: {:?}", result);
    println!("   🔧 Custom Config: FP16, 512MB workspace, profiling enabled");

    Ok(())
}

fn demo_text_generation(model_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Creating TensorRT text generation pipeline...");

    let config = TensorRTBackendConfig::latency_optimized(model_path.clone());
    let tokenizer = AutoTokenizer::from_pretrained("gpt2")?;

    let pipeline = tensorrt_text_generation_pipeline(model_path, tokenizer, Some(config))?
        .with_max_new_tokens(50)
        .with_temperature(0.8)
        .with_top_p(0.9)
        .with_do_sample(true);

    let prompts = vec![
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The most important thing about machine learning is",
    ];

    for prompt in prompts {
        let start = Instant::now();
        let result = pipeline.__call__(prompt.to_string())?;
        let duration = start.elapsed();

        println!("   📝 Prompt: {}", prompt);
        println!("   ⏱️ Time: {:.2}ms", duration.as_millis());
        println!("   📊 Generated: {:?}", result);

        // Benchmark generation
        let benchmark = pipeline.benchmark(prompt, 5)?;
        println!("   📈 Benchmark: {:?}", benchmark);
        println!();
    }

    Ok(())
}

fn demo_precision_comparison(
    model_path: &PathBuf,
    test_inputs: &[(&str, &str)],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Comparing different precision modes...");

    let precision_modes = vec![
        (PrecisionMode::FP32, "FP32 (Full Precision)"),
        (PrecisionMode::FP16, "FP16 (Half Precision)"),
        (PrecisionMode::INT8, "INT8 (8-bit Quantization)"),
    ];

    let test_input = test_inputs[0].0;
    let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;

    for (precision_mode, description) in precision_modes {
        println!("   📏 Testing {}", description);

        let config = TensorRTBackendConfig {
            model_path: model_path.clone(),
            precision_mode: precision_mode.clone(),
            enable_fp16: matches!(precision_mode, PrecisionMode::FP16),
            enable_int8: matches!(precision_mode, PrecisionMode::INT8),
            ..TensorRTBackendConfig::latency_optimized(model_path.clone())
        };

        let pipeline =
            tensorrt_text_classification_pipeline(model_path, tokenizer.clone(), Some(config))?;

        let start = Instant::now();
        let result = pipeline.__call__(test_input.to_string())?;
        let duration = start.elapsed();

        println!("      ⏱️ Time: {:.2}ms", duration.as_millis());
        println!("      📊 Result: {:?}", result);

        let memory_info = pipeline.memory_info()?;
        println!("      💾 Memory: {:?}", memory_info);
        println!();
    }

    Ok(())
}

fn demo_pipeline_manager(
    model_path: &PathBuf,
    generation_model_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Creating TensorRT pipeline manager...");

    let default_config = TensorRTBackendConfig::production(model_path.clone(), 4);
    let mut manager = TensorRTPipelineManager::new(default_config)
        .with_engine_cache(PathBuf::from("/tmp/tensorrt_cache"));

    // Load classification model
    manager.load_model("classifier".to_string(), model_path)?;

    // Load generation model with custom config
    let gen_config = TensorRTBackendConfig::throughput_optimized(generation_model_path.clone(), 8);
    manager.load_model_with_config("generator".to_string(), generation_model_path, gen_config)?;

    println!("   📋 Registered models: {:?}", manager.list_models());

    // Get model information
    if let Some(classifier) = manager.get_model("classifier") {
        println!(
            "   📊 Classifier precision: {:?}",
            classifier.precision_mode()
        );
        println!(
            "   📦 Classifier batch size: {}",
            classifier.max_batch_size()
        );
    }

    // Show cache information
    let cache_size = manager.cache_size()?;
    println!("   🗂️ Cache size: {} bytes", cache_size);

    Ok(())
}

fn demo_unified_api(model_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing unified pipeline API with TensorRT backend...");

    // Create TensorRT backend pipeline through unified API
    let tensorrt_config = TensorRTBackendConfig::latency_optimized(model_path.clone());
    let options = PipelineOptions {
        model: Some("bert-base-uncased".to_string()),
        device: Some(Device::Gpu(0)),
        batch_size: Some(1),
        max_length: Some(512),
        backend: Some(Backend::TensorRT {
            model_path: model_path.clone(),
        }),
        tensorrt_config: Some(tensorrt_config),
        ..Default::default()
    };

    let pipeline = pipeline("sentiment-analysis", None, Some(options))?;

    // Test with unified API
    let test_input = "This TensorRT integration is amazing!";
    let start = Instant::now();
    let result = pipeline.__call__(test_input.to_string())?;
    let duration = start.elapsed();

    println!("   📝 Input: {}", test_input);
    println!("   ⏱️ Time: {:.2}ms", duration.as_millis());
    println!("   📊 Result: {:?}", result);
    println!("   🔧 Backend: TensorRT through unified API");

    Ok(())
}

fn demo_benchmarking(
    model_path: &PathBuf,
    test_inputs: &[(&str, &str)],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Running comprehensive TensorRT benchmarks...");

    let config = TensorRTBackendConfig::production(model_path.clone(), 4);
    let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;

    let pipeline = tensorrt_text_classification_pipeline(model_path, tokenizer, Some(config))?;

    let test_input = test_inputs[0].0;

    // Benchmark different run counts
    let run_counts = vec![1, 10, 50, 100];

    for &num_runs in &run_counts {
        println!("   🔄 Benchmarking {} runs...", num_runs);

        let start = Instant::now();
        let benchmark = pipeline.benchmark(test_input, num_runs)?;
        let duration = start.elapsed();

        println!("      ⏱️ Benchmark time: {:.2}ms", duration.as_millis());
        println!("      📊 Results: {:?}", benchmark);

        // Calculate statistics
        let avg_time = duration.as_millis() as f64 / num_runs as f64;
        let throughput = 1000.0 / avg_time; // requests per second

        println!("      📈 Average time per run: {:.2}ms", avg_time);
        println!("      🚀 Throughput: {:.1} requests/second", throughput);
        println!();
    }

    Ok(())
}

/// Helper function to demonstrate TensorRT options
#[allow(dead_code)]
fn demonstrate_tensorrt_options() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎛️ TensorRT Pipeline Options Demo");
    println!("=================================");

    let model_path = PathBuf::from("./models/bert-base-uncased.onnx");

    // Different TensorRT pipeline options
    let options = vec![
        (
            "Latency Optimized",
            TensorRTPipelineOptions::latency_optimized(model_path.clone()),
        ),
        (
            "Throughput Optimized",
            TensorRTPipelineOptions::throughput_optimized(model_path.clone(), 8),
        ),
        (
            "Memory Optimized",
            TensorRTPipelineOptions::memory_optimized(model_path.clone()),
        ),
        (
            "Production",
            TensorRTPipelineOptions::production(model_path.clone(), 4),
        ),
    ];

    for (name, option) in options {
        println!("   🔧 {}: {:?}", name, option);
        println!("      - Enable profiling: {}", option.enable_profiling);
        println!("      - Warmup runs: {}", option.warmup_runs);
        println!("      - Engine cache: {}", option.enable_engine_cache);
        println!(
            "      - Precision: {:?}",
            option.tensorrt_config.precision_mode
        );
        println!(
            "      - Batch size: {}",
            option.tensorrt_config.max_batch_size
        );
        println!(
            "      - Workspace: {}MB",
            option.tensorrt_config.workspace_size / (1024 * 1024)
        );
        println!();
    }

    Ok(())
}

// Additional test functions
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_tensorrt_config_creation() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("model.onnx");

        let config = TensorRTBackendConfig::latency_optimized(model_path.clone());
        assert_eq!(config.model_path, model_path);
        assert!(matches!(config.precision_mode, PrecisionMode::FP16));
        assert!(config.enable_fp16);
        assert_eq!(config.max_batch_size, 1);
    }

    #[test]
    fn test_tensorrt_pipeline_options() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("model.onnx");

        let options = TensorRTPipelineOptions::latency_optimized(model_path.clone());
        assert_eq!(options.tensorrt_config.model_path, model_path);
        assert_eq!(options.warmup_runs, 5);
        assert!(options.enable_engine_cache);
    }

    #[test]
    fn test_tensorrt_manager_creation() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path().join("model.onnx");

        let config = TensorRTBackendConfig::production(model_path, 4);
        let manager = TensorRTPipelineManager::new(config);

        assert_eq!(manager.list_models().len(), 0);
        assert_eq!(manager.cache_size().unwrap(), 0);
    }
}
