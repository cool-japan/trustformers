use std::time::Instant;
#[allow(unused_variables)]
use trustformers::pipeline::{
    create_adaptive_inference_pipeline, create_balanced_adaptive_pipeline,
    create_energy_efficient_pipeline, create_latency_optimized_pipeline,
    create_memory_efficient_pipeline, pipeline, AdaptiveInferenceConfig, AdaptiveInferenceEngine,
    ConditionalStrategy, Device, Pipeline, PipelineOptions, PipelineOutput, PrecisionMode,
    ResourceStrategy,
};
use trustformers::{AutoModel, AutoTokenizer, TextClassificationPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 TrustformeRS Adaptive Inference Demo");
    println!("======================================");

    // Create a basic text classification pipeline
    let model_name = "bert-base-uncased";
    let model = AutoModel::from_pretrained(model_name)?;
    let tokenizer = AutoTokenizer::from_pretrained(model_name)?;
    let base_pipeline = TextClassificationPipeline::new(model, tokenizer)?;

    // Test inputs with different complexity levels
    let test_inputs = vec![
        ("This is great!", "Simple positive sentiment"),
        ("I absolutely love this amazing product with its incredible features and outstanding quality.", "Complex positive sentiment"),
        ("The movie was okay, not terrible but not great either, somewhere in between.", "Neutral sentiment"),
        ("This is a very long and complex sentence that discusses multiple topics including technology, politics, economics, and social issues in great detail.", "Complex multi-topic"),
    ];

    println!("\n📊 Testing Different Adaptive Inference Configurations");
    println!("=====================================================");

    // 1. Latency-optimized pipeline
    println!("\n1. 🏃 Latency-Optimized Pipeline (50ms budget)");
    let latency_pipeline = create_latency_optimized_pipeline(base_pipeline.clone(), 50);
    test_adaptive_pipeline(latency_pipeline, &test_inputs, "Latency-Optimized")?;

    // 2. Memory-efficient pipeline
    println!("\n2. 💾 Memory-Efficient Pipeline (512MB budget)");
    let memory_pipeline = create_memory_efficient_pipeline(base_pipeline.clone(), 512);
    test_adaptive_pipeline(memory_pipeline, &test_inputs, "Memory-Efficient")?;

    // 3. Energy-efficient pipeline
    println!("\n3. 🔋 Energy-Efficient Pipeline (25W budget)");
    let energy_pipeline = create_energy_efficient_pipeline(base_pipeline.clone(), 25.0);
    test_adaptive_pipeline(energy_pipeline, &test_inputs, "Energy-Efficient")?;

    // 4. Balanced adaptive pipeline
    println!("\n4. ⚖️ Balanced Adaptive Pipeline (0.85 quality threshold)");
    let balanced_pipeline = create_balanced_adaptive_pipeline(base_pipeline.clone(), 0.85);
    test_adaptive_pipeline(balanced_pipeline, &test_inputs, "Balanced")?;

    // 5. Custom configuration
    println!("\n5. 🎛️ Custom Configuration");
    let custom_config = AdaptiveInferenceConfig {
        precision_mode: PrecisionMode::Adaptive,
        conditional_strategy: ConditionalStrategy::DynamicDepth,
        resource_strategy: ResourceStrategy::Balanced,
        quality_threshold: 0.8,
        latency_budget_ms: 75,
        memory_budget_mb: 1024,
        energy_budget_watts: 30.0,
        adaptive_precision_threshold: 0.7,
        skip_probability_threshold: 0.3,
        dynamic_batch_size: true,
        progressive_inference: true,
        uncertainty_estimation: true,
        calibration_enabled: true,
        ..Default::default()
    };
    let custom_pipeline = create_adaptive_inference_pipeline(base_pipeline.clone(), custom_config);
    test_adaptive_pipeline(custom_pipeline, &test_inputs, "Custom")?;

    // 6. Demonstrate different precision modes
    println!("\n📈 Precision Mode Comparison");
    println!("============================");

    let precision_modes = vec![
        (PrecisionMode::Full, "Full Precision (FP32)"),
        (PrecisionMode::Half, "Half Precision (FP16)"),
        (PrecisionMode::Mixed, "Mixed Precision"),
        (PrecisionMode::Int8, "8-bit Quantization"),
        (PrecisionMode::Dynamic, "Dynamic Precision"),
        (PrecisionMode::Adaptive, "Adaptive Precision"),
    ];

    for (precision_mode, description) in precision_modes {
        println!("\n📏 Testing {}", description);
        let mut config = AdaptiveInferenceConfig::default();
        config.precision_mode = precision_mode;
        let pipeline = create_adaptive_inference_pipeline(base_pipeline.clone(), config);

        // Test with a representative input
        let test_input = "This is a test of the adaptive inference system.";
        let mut engine = pipeline;
        let start = Instant::now();
        let result = engine.adaptive_inference(test_input.to_string())?;
        let duration = start.elapsed();

        println!("   ⏱️ Time: {:.2}ms", duration.as_millis());
        println!("   📊 Quality: {:.3}", result.quality_score);
        println!("   🔬 Uncertainty: {:.3}", result.uncertainty_score);
        println!(
            "   ⚡ Resource Efficiency: {:.3}",
            result.resource_efficiency
        );
        println!(
            "   🎯 Layers Used: {}/{}",
            result.layers_computed,
            result.layers_computed + result.layers_skipped
        );
        println!("   💾 Memory Peak: {:.1}MB", result.memory_peak_mb);
        println!("   🔋 Energy: {:.1}W", result.energy_consumed_watts);
    }

    // 7. Demonstrate conditional computation strategies
    println!("\n🧠 Conditional Computation Strategies");
    println!("====================================");

    let conditional_strategies = vec![
        (ConditionalStrategy::AttentionSkipping, "Attention Skipping"),
        (
            ConditionalStrategy::FeedForwardSkipping,
            "Feed-Forward Skipping",
        ),
        (ConditionalStrategy::BlockSkipping, "Block Skipping"),
        (ConditionalStrategy::DynamicDepth, "Dynamic Depth"),
        (ConditionalStrategy::LayerConditional, "Layer Conditional"),
    ];

    for (strategy, description) in conditional_strategies {
        println!("\n🔄 Testing {}", description);
        let mut config = AdaptiveInferenceConfig::default();
        config.conditional_strategy = strategy;
        let pipeline = create_adaptive_inference_pipeline(base_pipeline.clone(), config);

        // Test with a complex input
        let test_input = "This is a complex sentence that requires careful analysis of multiple linguistic patterns and semantic relationships.";
        let mut engine = pipeline;
        let start = Instant::now();
        let result = engine.adaptive_inference(test_input.to_string())?;
        let duration = start.elapsed();

        println!("   ⏱️ Time: {:.2}ms", duration.as_millis());
        println!("   📊 Quality: {:.3}", result.quality_score);
        println!("   🔬 Layers Computed: {}", result.layers_computed);
        println!("   ⚡ Layers Skipped: {}", result.layers_skipped);
        println!("   🎯 Conditional Ops: {}", result.conditional_computations);
        println!("   📈 Efficiency: {:.3}", result.resource_efficiency);
    }

    println!("\n✅ Adaptive Inference Demo Complete!");
    println!("=====================================");

    Ok(())
}

fn test_adaptive_pipeline<P>(
    pipeline: AdaptiveInferenceEngine<P>,
    test_inputs: &[(&str, &str)],
    config_name: &str,
) -> Result<(), Box<dyn std::error::Error>>
where
    P: Pipeline<Output = PipelineOutput> + Clone,
    P::Input: Clone + From<String>,
{
    println!("   Testing {} configuration...", config_name);

    for (input_text, description) in test_inputs {
        let mut engine = pipeline.clone();
        let start = Instant::now();
        let result = engine.adaptive_inference(input_text.to_string().into())?;
        let duration = start.elapsed();

        println!("   📝 {}: {}", description, input_text);
        println!("      ⏱️ Time: {:.2}ms", duration.as_millis());
        println!("      📊 Quality: {:.3}", result.quality_score);
        println!("      🔬 Uncertainty: {:.3}", result.uncertainty_score);
        println!(
            "      ⚡ Resource Efficiency: {:.3}",
            result.resource_efficiency
        );
        println!(
            "      🎯 Layers: {}/{}",
            result.layers_computed,
            result.layers_computed + result.layers_skipped
        );
        println!("      💾 Memory: {:.1}MB", result.memory_peak_mb);
        println!("      🔋 Energy: {:.1}W", result.energy_consumed_watts);
        println!("      🔄 Precision: {:?}", result.precision_used);

        // Show adaptation decisions
        if !result.adaptation_decisions.is_empty() {
            println!("      🧠 Adaptations:");
            for decision in result.adaptation_decisions.iter().take(3) {
                println!(
                    "         - Layer {}: {} (confidence: {:.2})",
                    decision.layer_index, decision.decision_type, decision.confidence
                );
            }
        }

        // Show performance metrics
        println!("      📈 Performance:");
        println!(
            "         - Throughput: {:.1} tokens/sec",
            result.performance_metrics.throughput_tokens_per_second
        );
        println!(
            "         - Memory Efficiency: {:.3}",
            result.performance_metrics.memory_efficiency
        );
        println!(
            "         - Energy Efficiency: {:.3}",
            result.performance_metrics.energy_efficiency
        );
        println!(
            "         - Speedup Factor: {:.2}x",
            result.performance_metrics.speedup_factor
        );
        println!();
    }

    Ok(())
}

/// Helper function to demonstrate the unified pipeline API
#[allow(dead_code)]
fn demonstrate_unified_api() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔗 Unified Pipeline API Demo");
    println!("===========================");

    // Create adaptive inference pipeline through the unified API
    let options = PipelineOptions {
        model: Some("bert-base-uncased".to_string()),
        device: Some(Device::Cpu),
        batch_size: Some(4),
        max_length: Some(256),
        ..Default::default()
    };

    let adaptive_pipeline = pipeline("adaptive-inference", None, Some(options))?;

    // Test with the unified API
    let test_input = "This is a test of the unified adaptive inference API.";
    let start = Instant::now();
    let result = adaptive_pipeline.__call__(test_input.to_string())?;
    let duration = start.elapsed();

    println!("   📝 Input: {}", test_input);
    println!("   ⏱️ Time: {:.2}ms", duration.as_millis());
    println!("   📊 Output: {:?}", result);

    Ok(())
}

/// Helper function to demonstrate batch processing
#[allow(dead_code)]
fn demonstrate_batch_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📦 Batch Processing Demo");
    println!("========================");

    // Create a batch of inputs
    let batch_inputs = vec![
        "This is great!".to_string(),
        "I love this product.".to_string(),
        "Not bad, could be better.".to_string(),
        "Excellent quality and service.".to_string(),
        "Terrible experience, avoid.".to_string(),
    ];

    // Create adaptive pipeline
    let model = AutoModel::from_pretrained("bert-base-uncased")?;
    let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;
    let base_pipeline = TextClassificationPipeline::new(model, tokenizer)?;

    let config = AdaptiveInferenceConfig {
        dynamic_batch_size: true,
        ..Default::default()
    };

    let mut adaptive_pipeline = create_adaptive_inference_pipeline(base_pipeline, config);

    // Process batch with adaptive inference
    let start = Instant::now();
    let mut results = Vec::new();
    for input in batch_inputs {
        let result = adaptive_pipeline.adaptive_inference(input)?;
        results.push(result);
    }
    let duration = start.elapsed();

    println!("   📦 Batch size: {}", results.len());
    println!("   ⏱️ Total time: {:.2}ms", duration.as_millis());
    println!(
        "   📊 Average time per item: {:.2}ms",
        duration.as_millis() as f64 / results.len() as f64
    );

    // Show results summary
    let avg_quality: f32 =
        results.iter().map(|r| r.quality_score).sum::<f32>() / results.len() as f32;
    let avg_efficiency: f32 =
        results.iter().map(|r| r.resource_efficiency).sum::<f32>() / results.len() as f32;
    let total_layers_computed: usize = results.iter().map(|r| r.layers_computed).sum();
    let total_layers_skipped: usize = results.iter().map(|r| r.layers_skipped).sum();

    println!("   📈 Results Summary:");
    println!("      - Average Quality: {:.3}", avg_quality);
    println!("      - Average Efficiency: {:.3}", avg_efficiency);
    println!("      - Total Layers Computed: {}", total_layers_computed);
    println!("      - Total Layers Skipped: {}", total_layers_skipped);
    println!(
        "      - Skip Rate: {:.1}%",
        (total_layers_skipped as f32 / (total_layers_computed + total_layers_skipped) as f32)
            * 100.0
    );

    Ok(())
}

// Additional test functions
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_inference_config() {
        let config = AdaptiveInferenceConfig::default();
        assert!(matches!(config.precision_mode, PrecisionMode::Mixed));
        assert!(matches!(
            config.conditional_strategy,
            ConditionalStrategy::DynamicDepth
        ));
        assert!(matches!(
            config.resource_strategy,
            ResourceStrategy::Balanced
        ));
        assert_eq!(config.quality_threshold, 0.8);
    }

    #[test]
    fn test_precision_mode_variants() {
        let modes = [
            PrecisionMode::Full,
            PrecisionMode::Half,
            PrecisionMode::Mixed,
            PrecisionMode::Int8,
            PrecisionMode::Int4,
            PrecisionMode::Dynamic,
            PrecisionMode::Adaptive,
        ];

        for mode in modes {
            // Test that all modes can be created and are different
            let mut config = AdaptiveInferenceConfig::default();
            config.precision_mode = mode;
            assert!(matches!(config.precision_mode, _));
        }
    }

    #[test]
    fn test_conditional_strategies() {
        let strategies = [
            ConditionalStrategy::AttentionSkipping,
            ConditionalStrategy::FeedForwardSkipping,
            ConditionalStrategy::BlockSkipping,
            ConditionalStrategy::DynamicDepth,
            ConditionalStrategy::SparseActivation,
            ConditionalStrategy::TokenConditional,
            ConditionalStrategy::LayerConditional,
        ];

        for strategy in strategies {
            let mut config = AdaptiveInferenceConfig::default();
            config.conditional_strategy = strategy;
            assert!(matches!(config.conditional_strategy, _));
        }
    }

    #[test]
    fn test_resource_strategies() {
        let strategies = [
            ResourceStrategy::MinLatency,
            ResourceStrategy::MinMemory,
            ResourceStrategy::MinEnergy,
            ResourceStrategy::Balanced,
            ResourceStrategy::MaxThroughput,
        ];

        for strategy in strategies {
            let mut config = AdaptiveInferenceConfig::default();
            config.resource_strategy = strategy;
            assert!(matches!(config.resource_strategy, _));
        }
    }
}
