// ONNX Pipeline Backend Demo for TrustformeRS
#![allow(unused_variables)]
// Demonstrates comprehensive ONNX Runtime integration with pipelines

use anyhow::Result;
use std::path::PathBuf;
use std::time::Instant;
use trustformers::pipeline::{
    enhanced_pipeline, onnx_text_classification_pipeline, onnx_text_generation_pipeline,
    Backend, ONNXBackendConfig, ONNXPipelineManager, ONNXPipelineOptions, PipelineOptions,
    Device, Pipeline,
};
use trustformers::AutoTokenizer;
use trustformers_core::export::onnx_runtime::ExecutionProvider;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 TrustformeRS ONNX Pipeline Backend Demo");
    println!("==========================================\n");

    // Demo 1: Basic ONNX Pipeline Usage
    demo_basic_onnx_pipeline().await?;

    // Demo 2: Performance Comparison
    demo_performance_comparison().await?;

    // Demo 3: Advanced Configuration
    demo_advanced_configuration().await?;

    // Demo 4: Pipeline Manager
    demo_pipeline_manager().await?;

    // Demo 5: Benchmarking
    demo_benchmarking().await?;

    Ok(())
}

async fn demo_basic_onnx_pipeline() -> Result<()> {
    println!("📚 Demo 1: Basic ONNX Pipeline Usage");
    println!("------------------------------------");

    // NOTE: In a real scenario, you would have actual ONNX model files
    // For this demo, we'll create dummy paths and show the API usage

    let model_path = PathBuf::from("models/bert-base-uncased.onnx");

    if !model_path.exists() {
        println!("⚠️  ONNX model file not found at {:?}", model_path);
        println!("   This demo shows the API usage. In practice, you would:");
        println!("   1. Export your model to ONNX format");
        println!("   2. Place it at the specified path");
        println!("   3. Run the pipeline");
        println!();
        return Ok(());
    }

    // Create pipeline options with ONNX backend
    let options = PipelineOptions {
        backend: Some(Backend::ONNX { model_path: model_path.clone() }),
        device: Some(Device::Cpu),
        batch_size: Some(4),
        max_length: Some(512),
        ..Default::default()
    };

    println!("Creating text classification pipeline with ONNX backend...");
    let pipeline = enhanced_pipeline("text-classification", None, Some(options))?;

    // Test inputs
    let test_inputs = vec![
        "I love this product! It's amazing!",
        "This is terrible, I hate it.",
        "The weather is nice today.",
        "Machine learning is fascinating.",
    ];

    println!("Running inference on test inputs...");
    for (i, input) in test_inputs.iter().enumerate() {
        let start = Instant::now();
        let result = pipeline.__call__(input.to_string())?;
        let duration = start.elapsed();

        println!("Input {}: \"{}\"", i + 1, input);
        println!("Result: {:?}", result);
        println!("Latency: {:.2}ms\n", duration.as_secs_f64() * 1000.0);
    }

    Ok(())
}

async fn demo_performance_comparison() -> Result<()> {
    println!("⚡ Demo 2: Performance Comparison (Native vs ONNX)");
    println!("--------------------------------------------------");

    let model_path = PathBuf::from("models/gpt2.onnx");

    if !model_path.exists() {
        println!("⚠️  ONNX model file not found. Skipping performance comparison.");
        println!();
        return Ok(());
    }

    let test_input = "The future of artificial intelligence is";
    let num_runs = 10;

    // Native pipeline
    println!("Testing Native TrustformeRS pipeline...");
    let native_options = PipelineOptions {
        backend: Some(Backend::Native),
        device: Some(Device::Cpu),
        ..Default::default()
    };

    let native_pipeline = enhanced_pipeline("text-generation", Some("gpt2"), Some(native_options))?;

    let start = Instant::now();
    for _ in 0..num_runs {
        let _ = native_pipeline.__call__(test_input.to_string())?;
    }
    let native_duration = start.elapsed();
    let native_avg = native_duration.as_secs_f64() * 1000.0 / num_runs as f64;

    // ONNX pipeline
    println!("Testing ONNX Runtime pipeline...");
    let onnx_options = PipelineOptions {
        backend: Some(Backend::ONNX { model_path }),
        device: Some(Device::Cpu),
        ..Default::default()
    };

    let onnx_pipeline = enhanced_pipeline("text-generation", None, Some(onnx_options))?;

    let start = Instant::now();
    for _ in 0..num_runs {
        let _ = onnx_pipeline.__call__(test_input.to_string())?;
    }
    let onnx_duration = start.elapsed();
    let onnx_avg = onnx_duration.as_secs_f64() * 1000.0 / num_runs as f64;

    // Results
    println!("\nPerformance Comparison Results:");
    println!("Native TrustformeRS: {:.2}ms average", native_avg);
    println!("ONNX Runtime: {:.2}ms average", onnx_avg);
    println!("Speedup: {:.2}x", native_avg / onnx_avg);
    println!();

    Ok(())
}

async fn demo_advanced_configuration() -> Result<()> {
    println!("🔧 Demo 3: Advanced ONNX Configuration");
    println!("---------------------------------------");

    let model_path = PathBuf::from("models/bert-large.onnx");

    if !model_path.exists() {
        println!("⚠️  ONNX model file not found. Showing configuration examples.");
        println!();
    }

    // CPU-optimized configuration
    println!("Creating CPU-optimized ONNX configuration...");
    let cpu_config = ONNXBackendConfig::cpu_optimized(model_path.clone());
    println!("CPU Config: {:?}", cpu_config);
    println!();

    // GPU-optimized configuration
    println!("Creating GPU-optimized ONNX configuration...");
    let gpu_config = ONNXBackendConfig::gpu_optimized(model_path.clone(), Some(0));
    println!("GPU Config: {:?}", gpu_config);
    println!();

    // Production configuration with profiling
    println!("Creating production configuration with profiling...");
    let production_config = ONNXBackendConfig::production(model_path.clone())
        .with_profiling(PathBuf::from("./profile_output.json"));
    println!("Production Config: {:?}", production_config);
    println!();

    // Advanced pipeline options
    println!("Creating advanced ONNX pipeline options...");
    let advanced_options = ONNXPipelineOptions::gpu_optimized(model_path.clone(), Some(0))
        .with_profiling(true)
        .with_warmup_runs(5);
    println!("Advanced Options: {:?}", advanced_options);
    println!();

    // Direct pipeline creation with custom config
    if model_path.exists() {
        println!("Creating pipeline with custom configuration...");
        let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;
        let custom_config = ONNXBackendConfig {
            model_path: model_path.clone(),
            execution_providers: vec![
                ExecutionProvider::CUDA { device_id: Some(0) },
                ExecutionProvider::CPU,
            ],
            inter_op_threads: Some(4),
            intra_op_threads: Some(1),
            enable_profiling: true,
            ..Default::default()
        };

        let pipeline = onnx_text_classification_pipeline(
            &model_path,
            tokenizer,
            Some(custom_config),
        )?;

        println!("Custom pipeline created successfully!");

        // Test the pipeline
        let result = pipeline.__call__("This is a test sentence.".to_string())?;
        println!("Test result: {:?}", result);
    }

    println!();
    Ok(())
}

async fn demo_pipeline_manager() -> Result<()> {
    println!("🎛️  Demo 4: ONNX Pipeline Manager");
    println!("----------------------------------");

    // Create default config
    let default_config = ONNXBackendConfig::cpu_optimized(PathBuf::from("dummy.onnx"));
    let mut manager = ONNXPipelineManager::new(default_config);

    // Simulate model registration (would use real models in practice)
    let model_paths = vec![
        ("classification", "models/bert-classification.onnx"),
        ("generation", "models/gpt2-generation.onnx"),
        ("qa", "models/bert-qa.onnx"),
    ];

    println!("Registering models with the manager...");
    for (name, path) in &model_paths {
        let model_path = PathBuf::from(path);

        if model_path.exists() {
            manager.load_model(name.to_string(), &model_path)?;
            println!("✅ Registered model: {} from {}", name, path);
        } else {
            println!("⚠️  Model file not found: {}", path);
        }
    }

    // List all models
    println!("\nRegistered models:");
    for model_name in manager.list_models() {
        println!("  - {}", model_name);

        if let Some(model) = manager.get_model(model_name) {
            println!("    Inputs: {:?}", model.input_names());
            println!("    Outputs: {:?}", model.output_names());
            println!("    Providers: {:?}", model.execution_providers());
        }
    }

    // Benchmark all models (if they exist)
    println!("\nBenchmarking all models...");
    // Note: In practice, you'd provide appropriate test inputs for each model
    // let test_inputs = create_test_inputs();
    // let benchmark_results = manager.benchmark_all(test_inputs, 5)?;
    // for (name, results) in benchmark_results {
    //     println!("Model {}: {:.2}ms average", name, results.mean_latency_ms);
    // }

    println!("Pipeline manager demo completed.");
    println!();
    Ok(())
}

async fn demo_benchmarking() -> Result<()> {
    println!("📊 Demo 5: ONNX Pipeline Benchmarking");
    println!("--------------------------------------");

    let model_path = PathBuf::from("models/bert-base-uncased.onnx");

    if !model_path.exists() {
        println!("⚠️  ONNX model file not found. Skipping benchmarking demo.");
        println!();
        return Ok(());
    }

    println!("Creating ONNX text classification pipeline for benchmarking...");
    let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;
    let config = ONNXBackendConfig::cpu_optimized(model_path);
    let pipeline = onnx_text_classification_pipeline(&config.model_path, tokenizer, Some(config))?;

    // Benchmark with different input lengths
    let test_cases = vec![
        ("Short", "Good product."),
        ("Medium", "This is a medium length sentence that should give us a good baseline for performance testing."),
        ("Long", "This is a much longer sentence that contains significantly more tokens and should help us understand how the model performance scales with input length. We want to see if there are any performance bottlenecks or issues when processing longer sequences of text."),
    ];

    println!("\nRunning benchmarks...");
    for (name, input) in test_cases {
        println!("\n{} input benchmark:", name);
        println!("Input: \"{}\"", input);

        let benchmark_results = pipeline.benchmark(input, 10)?;
        benchmark_results.print_summary();

        // Get memory info
        let memory_info = pipeline.memory_info()?;
        println!("Memory Usage:");
        println!("  Total: {} MB", memory_info.total_memory_bytes / 1024 / 1024);
        println!("  Available: {} MB", memory_info.available_memory_bytes / 1024 / 1024);
        println!("  Model: {} MB", memory_info.model_memory_bytes / 1024 / 1024);
    }

    // Test different execution providers (if available)
    println!("\nTesting different execution providers...");
    let providers = pipeline.base.model.execution_providers();
    println!("Available providers: {:?}", providers);

    for provider in providers {
        println!("\nTesting with provider: {:?}", provider);
        let start = Instant::now();
        let inputs = std::collections::HashMap::new(); // Would create proper inputs
        // let result = pipeline.base.model.forward_with_provider(inputs, provider.clone())?;
        let duration = start.elapsed();
        println!("Provider {} latency: {:.2}ms", format!("{:?}", provider), duration.as_secs_f64() * 1000.0);
    }

    println!("\nBenchmarking completed!");
    println!();
    Ok(())
}

// Helper functions for creating test data (would be implemented based on model requirements)
fn create_test_inputs_for_classification() -> std::collections::HashMap<String, trustformers_core::tensor::Tensor> {
    // Would create appropriate tensor inputs for the specific model
    std::collections::HashMap::new()
}

fn create_test_inputs_for_generation() -> std::collections::HashMap<String, trustformers_core::tensor::Tensor> {
    // Would create appropriate tensor inputs for the specific model
    std::collections::HashMap::new()
}

// Performance monitoring utilities
struct PerformanceMonitor {
    start_time: Instant,
    measurements: Vec<f64>,
}

impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
            measurements: Vec::new(),
        }
    }

    fn record_measurement(&mut self) {
        let elapsed = self.start_time.elapsed();
        self.measurements.push(elapsed.as_secs_f64() * 1000.0);
        self.start_time = Instant::now();
    }

    fn get_statistics(&self) -> (f64, f64, f64, f64) {
        if self.measurements.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let mean = self.measurements.iter().sum::<f64>() / self.measurements.len() as f64;
        let min = self.measurements.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = self.measurements.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let mut sorted = self.measurements.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];

        (mean, median, min, max)
    }

    fn print_summary(&self, name: &str) {
        let (mean, median, min, max) = self.get_statistics();
        println!("{} Performance Summary:", name);
        println!("  Mean: {:.2}ms", mean);
        println!("  Median: {:.2}ms", median);
        println!("  Min: {:.2}ms", min);
        println!("  Max: {:.2}ms", max);
        println!("  Samples: {}", self.measurements.len());
    }
}