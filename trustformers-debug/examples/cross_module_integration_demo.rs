//! Cross-Module Integration Demonstration
#![allow(unused_variables)]
//!
//! This example demonstrates how trustformers-debug integrates with other
//! TrustformeRS crates to provide comprehensive ML debugging capabilities.
//! It showcases integration with models, training, optimization, and tokenization.

use anyhow::Result;
use scirs2_core::ndarray;
use scirs2_core::random::*;
use std::time::Duration;
use tokio::time::sleep;

use trustformers_debug::model_diagnostics::ModelArchitectureInfo;
use trustformers_debug::{
    debug_session_with_config, DebugConfig, DebugSession, DebugVisualizer, HookAction, HookBuilder,
    HookTrigger, ModelPerformanceMetrics, VisualizationConfig,
};

// Mock types to demonstrate integration patterns
// In a real implementation, these would come from other trustformers crates

#[derive(Debug, Clone)]
pub struct MockModel {
    pub name: String,
    pub architecture: String,
    pub parameters: usize,
    pub layers: Vec<MockLayer>,
}

#[derive(Debug, Clone)]
pub struct MockLayer {
    pub name: String,
    pub layer_type: String,
    pub parameters: usize,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct MockTrainingConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub optimizer: String,
}

#[derive(Debug, Clone)]
pub struct MockTokenizer {
    pub vocab_size: usize,
    pub model_max_length: usize,
    pub pad_token: String,
    pub unk_token: String,
}

#[derive(Debug, Clone)]
pub struct MockOptimizer {
    pub name: String,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub momentum: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct TrainingState {
    pub epoch: usize,
    pub step: usize,
    pub loss: f64,
    pub learning_rate: f64,
    pub gradient_norm: f64,
}

/// Comprehensive integration example demonstrating cross-module debugging
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("TrustformeRS Cross-Module Integration Demo");
    println!("========================================\n");

    // Example 1: Model architecture analysis
    model_architecture_analysis().await?;

    // Example 2: Training pipeline debugging
    training_pipeline_debugging().await?;

    // Example 3: Tokenization and preprocessing analysis
    tokenization_analysis().await?;

    // Example 4: Optimization algorithm monitoring
    optimization_monitoring().await?;

    // Example 5: End-to-end workflow debugging
    end_to_end_workflow_debugging().await?;

    println!("✅ All integration examples completed successfully!");
    Ok(())
}

async fn model_architecture_analysis() -> Result<()> {
    println!("=== Example 1: Model Architecture Analysis ===");

    let config = DebugConfig {
        max_tracked_tensors: 1000,
        output_dir: Some("./integration_debug/model_analysis".to_string()),
        ..DebugConfig::default()
    };

    let mut debug_session = DebugSession::new(config);
    debug_session.start().await?;

    // Create a mock transformer model
    let model = create_mock_transformer_model();
    println!("✓ Created mock transformer model: {}", model.name);
    println!("  - Architecture: {}", model.architecture);
    println!("  - Parameters: {}", format_number(model.parameters));
    println!("  - Layers: {}", model.layers.len());

    // Record model architecture in debug session
    let arch_info = ModelArchitectureInfo {
        total_parameters: model.parameters,
        trainable_parameters: model.parameters,
        model_size_mb: (model.parameters * 4) as f64 / (1024.0 * 1024.0), // 4 bytes per param
        layer_count: model.layers.len(),
        layer_types: {
            let mut types = std::collections::HashMap::new();
            for layer in &model.layers {
                *types.entry(layer.layer_type.clone()).or_insert(0) += 1;
            }
            types
        },
        depth: model.layers.len(),
        width: model
            .layers
            .first()
            .map(|l| l.output_shape.last().copied().unwrap_or(0))
            .unwrap_or(0),
        activation_functions: {
            let mut activations = std::collections::HashMap::new();
            activations.insert("GELU".to_string(), 12);
            activations.insert("Softmax".to_string(), 12);
            activations
        },
    };

    debug_session.model_diagnostics_mut().record_architecture(arch_info);

    // Simulate layer-wise analysis
    println!("\n--- Analyzing layer-wise properties ---");
    for (i, layer) in model.layers.iter().enumerate() {
        // Simulate tensor inspection for each layer
        let layer_data = simulate_layer_output(layer);

        let tensor_id = debug_session.tensor_inspector_mut().inspect_tensor(
            &layer_data,
            &format!("{}_output", layer.name),
            Some(&layer.name),
            Some("forward"),
        )?;

        // Record layer statistics
        let layer_stats = create_layer_activation_stats(layer, i);
        debug_session
            .model_diagnostics_mut()
            .record_layer_stats(layer_stats)
            .expect("Failed to record layer stats");

        if i % 3 == 0 {
            println!(
                "  ✓ Analyzed layer {}: {} (tensor_id: {})",
                i, layer.name, tensor_id
            );
        }
    }

    // Analyze model complexity
    let complexity_analysis = analyze_model_complexity(&model);
    println!("\n--- Model Complexity Analysis ---");
    println!(
        "  FLOPs per forward pass: {}",
        format_number(complexity_analysis.flops)
    );
    println!(
        "  Memory footprint: {:.1} MB",
        complexity_analysis.memory_mb
    );
    println!(
        "  Computational intensity: {:.2}",
        complexity_analysis.compute_intensity
    );

    // Generate architecture report
    let report = debug_session.stop().await?;

    if let Some(ref diagnostics_report) = report.diagnostics_report {
        println!("\n--- Architecture Analysis Report ---");
        println!("Layers analyzed: {}", model.layers.len());
        println!("Issues detected: {}", diagnostics_report.alerts.len());

        for recommendation in &diagnostics_report.recommendations {
            println!("  💡 {}", recommendation);
        }
    }

    println!("✅ Model architecture analysis completed\n");
    Ok(())
}

async fn training_pipeline_debugging() -> Result<()> {
    println!("=== Example 2: Training Pipeline Debugging ===");

    let config = DebugConfig {
        max_gradient_history: 500,
        output_dir: Some("./integration_debug/training".to_string()),
        ..DebugConfig::default()
    };

    let mut debug_session = DebugSession::new(config);
    debug_session.start().await?;

    // Setup training configuration
    let training_config = MockTrainingConfig {
        learning_rate: 1e-4,
        batch_size: 32,
        max_epochs: 10,
        optimizer: "AdamW".to_string(),
    };

    println!("✓ Training configuration:");
    println!("  - Learning rate: {:.0e}", training_config.learning_rate);
    println!("  - Batch size: {}", training_config.batch_size);
    println!("  - Optimizer: {}", training_config.optimizer);

    // Set up training monitoring hooks
    let training_hook = HookBuilder::new("Training Monitor")
        .trigger(HookTrigger::EveryForward)
        .action(HookAction::TrackGradients)
        .layer_patterns(vec!["attention.*".to_string(), "linear.*".to_string()])
        .build();

    let hook_id = debug_session.hooks_mut().register_hook(training_hook)?;
    println!("✓ Registered training monitoring hook: {}", hook_id);

    // Simulate training loop
    println!("\n--- Simulating training loop ---");

    for epoch in 0..3 {
        for step in 0..20 {
            let global_step = epoch * 20 + step;

            // Simulate training state
            let training_state = TrainingState {
                epoch,
                step: global_step,
                loss: 2.5 * (-0.1 * global_step as f64).exp()
                    + 0.1
                    + (global_step as f64 * 0.05).sin() * 0.05,
                learning_rate: training_config.learning_rate
                    * (1.0 - global_step as f64 / 60.0).max(0.1),
                gradient_norm: 0.5 * (-0.05 * global_step as f64).exp()
                    + 0.1 * (global_step as f64 * 0.1).cos().abs(),
            };

            // Record performance metrics
            let metrics = ModelPerformanceMetrics {
                training_step: global_step,
                loss: training_state.loss,
                accuracy: Some((1.0 - training_state.loss / 3.0).max(0.0).min(1.0)),
                learning_rate: training_state.learning_rate,
                batch_size: training_config.batch_size,
                throughput_samples_per_sec: 150.0 + global_step as f64 * 2.0,
                memory_usage_mb: 2048.0 + global_step as f64 * 5.0,
                gpu_utilization: Some(85.0 + (global_step as f64 * 0.3).sin() * 10.0),
                timestamp: chrono::Utc::now(),
            };

            debug_session.model_diagnostics_mut().record_performance(metrics)?;

            // Record gradient flow
            let layer_names = vec!["attention_0", "linear_1", "attention_1", "linear_2"];
            for layer_name in &layer_names {
                let gradient_values = simulate_gradients(&training_state, layer_name);
                let gradient_norm = (gradient_values.iter().map(|&x| x * x).sum::<f64>()).sqrt();
                let gradient_mean =
                    gradient_values.iter().sum::<f64>() / gradient_values.len() as f64;
                let gradient_variance =
                    gradient_values.iter().map(|&x| (x - gradient_mean).powi(2)).sum::<f64>()
                        / gradient_values.len() as f64;
                let gradient_std = gradient_variance.sqrt();

                debug_session.gradient_debugger_mut().record_gradient_flow(
                    layer_name,
                    gradient_norm,
                    gradient_mean,
                    gradient_std,
                )?;
            }

            debug_session.gradient_debugger_mut().next_step();

            // Trigger hooks for monitoring
            debug_session.hooks_mut().set_step(global_step);

            if global_step % 10 == 0 {
                println!(
                    "  Epoch {}, Step {}: loss={:.4}, lr={:.2e}, grad_norm={:.4}",
                    epoch,
                    step,
                    training_state.loss,
                    training_state.learning_rate,
                    training_state.gradient_norm
                );
            }
        }
    }

    // Analyze training dynamics
    let training_dynamics = debug_session.model_diagnostics().analyze_training_dynamics();
    println!("\n--- Training Dynamics Analysis ---");
    println!(
        "Convergence status: {:?}",
        training_dynamics.convergence_status
    );
    println!(
        "Training stability: {:?}",
        training_dynamics.training_stability
    );
    println!(
        "Learning efficiency: {:.6}",
        training_dynamics.learning_efficiency
    );

    // Analyze gradient flow
    let gradient_analysis = debug_session.gradient_debugger().analyze_gradient_conflicts();
    println!("\n--- Gradient Flow Analysis ---");
    println!(
        "Total conflicts detected: {}",
        gradient_analysis.total_conflicts
    );
    println!(
        "Overall conflict level: {:?}",
        gradient_analysis.overall_conflict_level
    );
    println!("Conflicts detected: {}", gradient_analysis.conflicts.len());
    println!(
        "Mitigation strategies: {}",
        gradient_analysis.mitigation_strategies.len()
    );

    let report = debug_session.stop().await?;
    println!("✅ Training pipeline debugging completed\n");
    Ok(())
}

async fn tokenization_analysis() -> Result<()> {
    println!("=== Example 3: Tokenization and Preprocessing Analysis ===");

    let mut debug_session = debug_session_with_config(DebugConfig::default());
    debug_session.start().await?;

    // Create mock tokenizer
    let tokenizer = MockTokenizer {
        vocab_size: 50000,
        model_max_length: 512,
        pad_token: "[PAD]".to_string(),
        unk_token: "[UNK]".to_string(),
    };

    println!("✓ Tokenizer configuration:");
    println!(
        "  - Vocabulary size: {}",
        format_number(tokenizer.vocab_size)
    );
    println!("  - Max length: {}", tokenizer.model_max_length);
    println!(
        "  - Special tokens: {} / {}",
        tokenizer.pad_token, tokenizer.unk_token
    );

    // Simulate tokenization and preprocessing
    let sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is revolutionizing artificial intelligence.",
        "TrustformeRS provides comprehensive debugging tools for ML models.",
        "Attention is all you need for transformer architectures.",
    ];

    println!("\n--- Analyzing tokenization patterns ---");

    for (i, text) in sample_texts.iter().enumerate() {
        // Simulate tokenization
        let tokens = simulate_tokenization(text, &tokenizer);
        let token_ids = simulate_token_ids(&tokens, &tokenizer);

        // Convert to tensor for debugging
        let token_tensor = ndarray::Array::from_vec(token_ids.iter().map(|&x| x as f32).collect())
            .into_shape_with_order(ndarray::IxDyn(&[1, token_ids.len()]))
            .unwrap();

        let tensor_id = debug_session.tensor_inspector_mut().inspect_tensor(
            &token_tensor,
            &format!("tokens_sample_{}", i),
            Some("tokenizer"),
            Some("encode"),
        )?;

        // Analyze tokenization statistics
        let stats = analyze_tokenization_stats(&tokens, &token_ids, &tokenizer);

        if i == 0 {
            println!("  ✓ Sample analysis:");
            println!("    Text: \"{}\"", text);
            println!("    Tokens: {} (tensor_id: {})", tokens.len(), tensor_id);
            println!("    OOV rate: {:.2}%", stats.oov_rate * 100.0);
            println!("    Compression ratio: {:.2}", stats.compression_ratio);
        }
    }

    // Simulate embedding analysis
    println!("\n--- Analyzing embeddings ---");
    let embedding_dim = 768;

    for i in 0..3 {
        let embedding_data = simulate_embeddings(embedding_dim);

        let tensor_id = debug_session.tensor_inspector_mut().inspect_tensor(
            &embedding_data,
            &format!("embeddings_{}", i),
            Some("embedding_layer"),
            Some("forward"),
        )?;

        if i == 0 {
            println!("  ✓ Embedding analysis (tensor_id: {})", tensor_id);
            println!("    Dimension: {}", embedding_dim);
            println!("    Mean: {:.4}", embedding_data.mean().unwrap_or(0.0));
            println!(
                "    Std: {:.4}",
                embedding_data.std_axis(ndarray::Axis(1), 0.0).mean().unwrap_or(0.0)
            );
        }
    }

    let report = debug_session.stop().await?;
    println!("✅ Tokenization analysis completed\n");
    Ok(())
}

async fn optimization_monitoring() -> Result<()> {
    println!("=== Example 4: Optimization Algorithm Monitoring ===");

    let mut debug_session = debug_session_with_config(DebugConfig::default());
    debug_session.start().await?;

    // Create mock optimizer
    let optimizer = MockOptimizer {
        name: "AdamW".to_string(),
        learning_rate: 1e-4,
        weight_decay: 0.01,
        momentum: Some(0.9),
    };

    println!("✓ Optimizer configuration:");
    println!("  - Algorithm: {}", optimizer.name);
    println!("  - Learning rate: {:.0e}", optimizer.learning_rate);
    println!("  - Weight decay: {}", optimizer.weight_decay);

    // Set up optimization monitoring
    let opt_hook = HookBuilder::new("Optimizer Monitor")
        .trigger(HookTrigger::EveryBackward)
        .action(HookAction::TrackGradients)
        .build();

    let hook_id = debug_session.hooks_mut().register_hook(opt_hook)?;
    println!("✓ Registered optimization monitoring hook: {}", hook_id);

    // Simulate optimization steps
    println!("\n--- Monitoring optimization steps ---");

    for step in 0..50 {
        // Simulate parameter updates
        let layer_names = vec![
            "linear_1",
            "linear_2",
            "attention_q",
            "attention_k",
            "attention_v",
        ];

        for layer_name in &layer_names {
            // Simulate gradient computation
            {
                let profiler = debug_session.profiler_mut();
                profiler.start_timer(&format!("{}_gradient", layer_name));
            }
            sleep(Duration::from_millis(5)).await;

            let gradient_norm =
                0.1 * (-0.02 * step as f64).exp() + 0.05 * (step as f64 * 0.1).sin().abs();
            let grad_time = {
                let profiler = debug_session.profiler_mut();
                profiler
                    .end_timer(&format!("{}_gradient", layer_name))
                    .unwrap_or(Duration::ZERO)
            };

            {
                let profiler = debug_session.profiler_mut();
                profiler.record_gradient_computation(layer_name, gradient_norm, grad_time);
            }

            // Simulate parameter update
            let param_data = simulate_parameters(layer_name, step);
            debug_session.tensor_inspector_mut().inspect_tensor(
                &param_data,
                &format!("{}_params_step_{}", layer_name, step),
                Some(layer_name),
                Some("parameter_update"),
            )?;
        }

        // Record optimization metrics
        let learning_rate = optimizer.learning_rate * (1.0 - step as f64 / 100.0).max(0.1);

        if step % 10 == 0 {
            println!(
                "  Step {}: lr={:.2e}, monitoring {} layers",
                step,
                learning_rate,
                layer_names.len()
            );
        }
    }

    // Analyze optimization performance
    let bottlenecks = debug_session.profiler_mut().analyze_performance();
    println!("\n--- Optimization Performance Analysis ---");
    println!("Performance bottlenecks detected: {}", bottlenecks.len());

    for bottleneck in &bottlenecks {
        println!("  {:?}: {}", bottleneck.severity, bottleneck.description);
    }

    let report = debug_session.stop().await?;
    println!("✅ Optimization monitoring completed\n");
    Ok(())
}

async fn end_to_end_workflow_debugging() -> Result<()> {
    println!("=== Example 5: End-to-End Workflow Debugging ===");

    let config = DebugConfig {
        max_tracked_tensors: 2000,
        max_gradient_history: 1000,
        output_dir: Some("./integration_debug/e2e".to_string()),
        ..DebugConfig::default()
    };

    let mut debug_session = DebugSession::new(config);
    debug_session.start().await?;

    println!("✓ Started comprehensive end-to-end debugging session");

    // Set up comprehensive monitoring
    let hooks = vec![
        HookBuilder::new("Input Monitor")
            .trigger(HookTrigger::EveryForward)
            .action(HookAction::InspectTensor)
            .layer_patterns(vec!["input.*".to_string(), "embedding.*".to_string()])
            .build(),
        HookBuilder::new("Attention Monitor")
            .trigger(HookTrigger::EveryForward)
            .action(HookAction::InspectTensor)
            .layer_patterns(vec!["attention.*".to_string()])
            .build(),
        HookBuilder::new("Output Monitor")
            .trigger(HookTrigger::EveryForward)
            .action(HookAction::InspectTensor)
            .layer_patterns(vec!["output.*".to_string(), "classifier.*".to_string()])
            .build(),
        HookBuilder::new("Gradient Monitor")
            .trigger(HookTrigger::EveryBackward)
            .action(HookAction::TrackGradients)
            .build(),
    ];

    let mut hook_ids = Vec::new();
    for hook in hooks {
        let hook_id = debug_session.hooks_mut().register_hook(hook)?;
        hook_ids.push(hook_id);
    }

    println!(
        "✓ Registered {} comprehensive monitoring hooks",
        hook_ids.len()
    );

    // Simulate complete ML pipeline
    println!("\n--- Simulating complete ML pipeline ---");

    // Execute stages directly
    println!("  🔄 Executing stage: Data Loading");
    {
        let profiler = debug_session.profiler_mut();
        profiler.start_timer("stage_data_loading");
    }
    simulate_data_loading(&mut debug_session).await?;
    let stage_time = {
        let profiler = debug_session.profiler_mut();
        profiler.end_timer("stage_data_loading").unwrap_or(Duration::ZERO)
    };
    println!("    ✓ Completed in {:.1}ms", stage_time.as_millis());

    println!("  🔄 Executing stage: Tokenization");
    {
        let profiler = debug_session.profiler_mut();
        profiler.start_timer("stage_tokenization");
    }
    simulate_tokenization_stage(&mut debug_session).await?;
    let stage_time = {
        let profiler = debug_session.profiler_mut();
        profiler.end_timer("stage_tokenization").unwrap_or(Duration::ZERO)
    };
    println!("    ✓ Completed in {:.1}ms", stage_time.as_millis());

    println!("  🔄 Executing stage: Model Forward");
    {
        let profiler = debug_session.profiler_mut();
        profiler.start_timer("stage_model_forward");
    }
    simulate_model_forward(&mut debug_session).await?;
    let stage_time = {
        let profiler = debug_session.profiler_mut();
        profiler.end_timer("stage_model_forward").unwrap_or(Duration::ZERO)
    };
    println!("    ✓ Completed in {:.1}ms", stage_time.as_millis());

    println!("  🔄 Executing stage: Loss Computation");
    {
        let profiler = debug_session.profiler_mut();
        profiler.start_timer("stage_loss_computation");
    }
    simulate_loss_computation(&mut debug_session).await?;
    let stage_time = {
        let profiler = debug_session.profiler_mut();
        profiler.end_timer("stage_loss_computation").unwrap_or(Duration::ZERO)
    };
    println!("    ✓ Completed in {:.1}ms", stage_time.as_millis());

    println!("  🔄 Executing stage: Backward Pass");
    {
        let profiler = debug_session.profiler_mut();
        profiler.start_timer("stage_backward_pass");
    }
    simulate_backward_pass(&mut debug_session).await?;
    let stage_time = {
        let profiler = debug_session.profiler_mut();
        profiler.end_timer("stage_backward_pass").unwrap_or(Duration::ZERO)
    };
    println!("    ✓ Completed in {:.1}ms", stage_time.as_millis());

    println!("  🔄 Executing stage: Optimization");
    {
        let profiler = debug_session.profiler_mut();
        profiler.start_timer("stage_optimization");
    }
    simulate_optimization_step(&mut debug_session).await?;
    let stage_time = {
        let profiler = debug_session.profiler_mut();
        profiler.end_timer("stage_optimization").unwrap_or(Duration::ZERO)
    };
    println!("    ✓ Completed in {:.1}ms", stage_time.as_millis());

    // Generate comprehensive visualization
    println!("\n--- Generating comprehensive visualizations ---");

    let viz_config = VisualizationConfig {
        output_directory: "./integration_debug/e2e/visualizations".to_string(),
        ..Default::default()
    };

    let mut visualizer = DebugVisualizer::new(viz_config);

    // Create multiple visualizations
    create_comprehensive_visualizations(&mut visualizer).await?;

    println!("✓ Generated comprehensive visualizations");

    // Generate final comprehensive report
    let report = debug_session.stop().await?;

    println!("\n--- End-to-End Integration Report ---");
    println!("Session ID: {}", report.session_id);

    let summary = report.summary();
    println!("Total components analyzed: {}", summary.total_issues); // Simplified - using total_issues as proxy
    println!("Total issues detected: {}", summary.total_issues);
    println!("Critical issues: {}", summary.critical_issues);

    println!("\nKey recommendations:");
    for (i, recommendation) in summary.recommendations.iter().take(5).enumerate() {
        println!("  {}. {}", i + 1, recommendation);
    }

    // Hook execution statistics
    println!("\n--- Hook Execution Statistics ---");
    let hook_stats = debug_session.hooks().get_all_stats();
    for stats in hook_stats {
        println!(
            "  {}: {} executions, {:.1}ms avg",
            stats.hook_name, stats.total_executions, stats.avg_execution_time_ms
        );
    }

    println!("✅ End-to-end workflow debugging completed\n");
    Ok(())
}

// Helper functions for simulation

fn create_mock_transformer_model() -> MockModel {
    let mut layers = Vec::new();

    // Embedding layer
    layers.push(MockLayer {
        name: "embedding".to_string(),
        layer_type: "Embedding".to_string(),
        parameters: 50_000 * 768,         // vocab_size * hidden_size
        input_shape: vec![32, 512],       // batch_size, seq_len
        output_shape: vec![32, 512, 768], // batch_size, seq_len, hidden_size
    });

    // Transformer blocks
    for i in 0..12 {
        // Multi-head attention
        layers.push(MockLayer {
            name: format!("transformer_block_{}_attention", i),
            layer_type: "MultiHeadAttention".to_string(),
            parameters: 768 * 768 * 4, // 4 * (hidden_size * hidden_size)
            input_shape: vec![32, 512, 768],
            output_shape: vec![32, 512, 768],
        });

        // Feed forward
        layers.push(MockLayer {
            name: format!("transformer_block_{}_ffn", i),
            layer_type: "FeedForward".to_string(),
            parameters: 768 * 3072 + 3072 * 768, // hidden_size * intermediate + intermediate * hidden_size
            input_shape: vec![32, 512, 768],
            output_shape: vec![32, 512, 768],
        });
    }

    // Output layer
    layers.push(MockLayer {
        name: "output_projection".to_string(),
        layer_type: "Linear".to_string(),
        parameters: 768 * 50_000, // hidden_size * vocab_size
        input_shape: vec![32, 512, 768],
        output_shape: vec![32, 512, 50_000],
    });

    let total_params = layers.iter().map(|l| l.parameters).sum();

    MockModel {
        name: "MockTransformer-Base".to_string(),
        architecture: "Transformer".to_string(),
        parameters: total_params,
        layers,
    }
}

fn simulate_layer_output(layer: &MockLayer) -> ndarray::ArrayD<f32> {
    let mut rng = thread_rng();
    let total_elements: usize = layer.output_shape.iter().product();
    let data: Vec<f32> = (0..total_elements)
        .map(|i| (i as f32 * 0.01).sin() * 0.5 + (rng.random::<f32>() - 0.5) * 0.1)
        .collect();

    ndarray::Array::from_vec(data)
        .into_shape_with_order(ndarray::IxDyn(&layer.output_shape))
        .unwrap()
}

fn create_layer_activation_stats(
    layer: &MockLayer,
    layer_idx: usize,
) -> trustformers_debug::LayerActivationStats {
    trustformers_debug::LayerActivationStats {
        layer_name: layer.name.clone(),
        mean_activation: 0.5 + (layer_idx as f64 * 0.1).sin() * 0.1,
        std_activation: 0.2 + (layer_idx as f64 * 0.05).cos() * 0.05,
        min_activation: -0.2,
        max_activation: 1.2,
        dead_neurons_ratio: 0.05 + (layer_idx as f64 * 0.01),
        saturated_neurons_ratio: 0.02,
        sparsity: 0.3 + (layer_idx as f64 * 0.02),
        output_shape: layer.output_shape.clone(),
    }
}

#[derive(Debug)]
struct ModelComplexity {
    flops: usize,
    memory_mb: f64,
    compute_intensity: f64,
}

fn analyze_model_complexity(model: &MockModel) -> ModelComplexity {
    let mut total_flops = 0;

    for layer in &model.layers {
        match layer.layer_type.as_str() {
            "MultiHeadAttention" => {
                // Simplified FLOP calculation for attention
                let seq_len = layer.input_shape[1];
                let hidden_size = layer.input_shape[2];
                total_flops += seq_len * seq_len * hidden_size * 4; // Q*K^T, softmax, *V, etc.
            },
            "FeedForward" => {
                // FLOP calculation for FFN
                let hidden_size = layer.input_shape[2];
                total_flops += hidden_size * 3072 * 2; // forward and backward
            },
            "Linear" => {
                total_flops += layer.parameters * 2; // matrix multiplication
            },
            _ => {
                total_flops += layer.parameters; // rough estimate
            },
        }
    }

    let memory_mb = (model.parameters * 4) as f64 / (1024.0 * 1024.0); // 4 bytes per param
    let compute_intensity = total_flops as f64 / (memory_mb * 1024.0 * 1024.0);

    ModelComplexity {
        flops: total_flops,
        memory_mb,
        compute_intensity,
    }
}

fn simulate_gradients(training_state: &TrainingState, layer_name: &str) -> Vec<f64> {
    let base_norm = training_state.gradient_norm;
    let layer_factor = match layer_name {
        name if name.contains("attention") => 1.2,
        name if name.contains("linear") => 0.8,
        _ => 1.0,
    };

    (0..1000)
        .map(|i| {
            base_norm * layer_factor * (i as f64 * 0.01 + training_state.step as f64 * 0.1).sin()
        })
        .collect()
}

fn simulate_tokenization(text: &str, _tokenizer: &MockTokenizer) -> Vec<String> {
    // Simple whitespace tokenization for demo
    text.split_whitespace()
        .map(|s| s.trim_matches(|c: char| c.is_ascii_punctuation()))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

fn simulate_token_ids(tokens: &[String], tokenizer: &MockTokenizer) -> Vec<i64> {
    tokens
        .iter()
        .map(|token| {
            // Simple hash-based ID generation for demo
            let hash =
                token.chars().fold(0u32, |acc, c| acc.wrapping_mul(31).wrapping_add(c as u32));
            (hash % tokenizer.vocab_size as u32) as i64
        })
        .collect()
}

#[derive(Debug)]
struct TokenizationStats {
    oov_rate: f64,
    compression_ratio: f64,
}

fn analyze_tokenization_stats(
    tokens: &[String],
    token_ids: &[i64],
    _tokenizer: &MockTokenizer,
) -> TokenizationStats {
    let oov_count = token_ids.iter().filter(|&&id| id == 0).count(); // Assume 0 is UNK
    let oov_rate = oov_count as f64 / token_ids.len() as f64;

    let original_chars: usize = tokens.iter().map(|t| t.len()).sum();
    let compression_ratio = original_chars as f64 / tokens.len() as f64;

    TokenizationStats {
        oov_rate,
        compression_ratio,
    }
}

fn simulate_embeddings(embedding_dim: usize) -> ndarray::ArrayD<f32> {
    let mut rng = thread_rng();
    let data: Vec<f32> = (0..embedding_dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.5 + (rng.random::<f32>() - 0.5) * 0.1)
        .collect();

    ndarray::Array::from_vec(data)
        .into_shape_with_order(ndarray::IxDyn(&[1, embedding_dim]))
        .unwrap()
}

fn simulate_parameters(layer_name: &str, step: usize) -> ndarray::ArrayD<f32> {
    let size = match layer_name {
        name if name.contains("linear") => 1000,
        name if name.contains("attention") => 1500,
        _ => 500,
    };

    let data: Vec<f32> =
        (0..size).map(|i| (i as f32 * 0.01 + step as f32 * 0.001).sin() * 0.1).collect();

    ndarray::Array::from_vec(data)
        .into_shape_with_order(ndarray::IxDyn(&[size]))
        .unwrap()
}

async fn simulate_data_loading(debug_session: &mut DebugSession) -> Result<()> {
    // Simulate loading batch data
    sleep(Duration::from_millis(20)).await;

    let batch_data = ndarray::Array::linspace(0.0, 1.0, 32 * 512)
        .into_shape_with_order(ndarray::IxDyn(&[32, 512]))
        .unwrap();

    debug_session.tensor_inspector_mut().inspect_tensor(
        &batch_data,
        "input_batch",
        Some("data_loader"),
        Some("load"),
    )?;

    Ok(())
}

async fn simulate_tokenization_stage(debug_session: &mut DebugSession) -> Result<()> {
    sleep(Duration::from_millis(15)).await;

    let token_ids = ndarray::Array::from_vec((0..32 * 512).map(|i| (i % 50000) as f32).collect())
        .into_shape_with_order(ndarray::IxDyn(&[32, 512]))
        .unwrap();

    debug_session.tensor_inspector_mut().inspect_tensor(
        &token_ids,
        "token_ids",
        Some("tokenizer"),
        Some("encode"),
    )?;

    Ok(())
}

async fn simulate_model_forward(debug_session: &mut DebugSession) -> Result<()> {
    sleep(Duration::from_millis(100)).await;

    let layer_names = vec!["embedding", "attention_0", "attention_1", "ffn_0", "ffn_1"];

    for layer_name in layer_names {
        let output = ndarray::Array::linspace(0.0, 1.0, 32 * 512 * 768)
            .into_shape_with_order(ndarray::IxDyn(&[32, 512, 768]))
            .unwrap();

        debug_session.tensor_inspector_mut().inspect_tensor(
            &output,
            &format!("{}_output", layer_name),
            Some(layer_name),
            Some("forward"),
        )?;
    }

    Ok(())
}

async fn simulate_loss_computation(debug_session: &mut DebugSession) -> Result<()> {
    sleep(Duration::from_millis(10)).await;

    let loss = ndarray::Array::from_vec(vec![2.5])
        .into_shape_with_order(ndarray::IxDyn(&[1]))
        .unwrap();

    debug_session.tensor_inspector_mut().inspect_tensor(
        &loss,
        "loss",
        Some("loss_function"),
        Some("compute"),
    )?;

    Ok(())
}

async fn simulate_backward_pass(debug_session: &mut DebugSession) -> Result<()> {
    sleep(Duration::from_millis(80)).await;

    let layer_names = vec!["ffn_1", "ffn_0", "attention_1", "attention_0", "embedding"];

    for layer_name in layer_names {
        let gradients: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin() * 0.1).collect();
        let gradient_norm = (gradients.iter().map(|&x| x * x).sum::<f64>()).sqrt();
        let gradient_mean = gradients.iter().sum::<f64>() / gradients.len() as f64;
        let gradient_variance = gradients.iter().map(|&x| (x - gradient_mean).powi(2)).sum::<f64>()
            / gradients.len() as f64;
        let gradient_std = gradient_variance.sqrt();

        debug_session.gradient_debugger_mut().record_gradient_flow(
            layer_name,
            gradient_norm,
            gradient_mean,
            gradient_std,
        )?;
    }

    debug_session.gradient_debugger_mut().next_step();
    Ok(())
}

async fn simulate_optimization_step(_debug_session: &mut DebugSession) -> Result<()> {
    sleep(Duration::from_millis(30)).await;
    Ok(())
}

async fn create_comprehensive_visualizations(visualizer: &mut DebugVisualizer) -> Result<()> {
    // Create sample data for visualizations
    let steps: Vec<f64> = (0..50).map(|x| x as f64).collect();
    let losses: Vec<f64> = steps.iter().map(|&s| 2.0 * (-0.05 * s).exp() + 0.1).collect();
    let accuracies: Vec<f64> =
        losses.iter().map(|&loss| (1.0 - loss / 2.5).max(0.0).min(1.0)).collect();

    // Training metrics
    visualizer.plot_training_metrics(&steps, &losses, Some(&accuracies))?;

    // Gradient flow
    let gradient_norms: Vec<f64> = steps.iter().map(|&s| 0.1 * (-0.02 * s).exp() + 0.01).collect();
    visualizer.plot_gradient_flow("overall", &steps, &gradient_norms)?;

    // Tensor distribution
    let mut rng = thread_rng();
    let tensor_values: Vec<f64> = (0..1000)
        .map(|i| (i as f64 * 0.01).sin() + (rng.random::<f64>() - 0.5) * 0.1)
        .collect();
    visualizer.plot_tensor_distribution("model_weights", &tensor_values, 30)?;

    // Create dashboard
    let plot_names = visualizer.get_plot_names();
    visualizer.create_dashboard(&plot_names)?;

    Ok(())
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
