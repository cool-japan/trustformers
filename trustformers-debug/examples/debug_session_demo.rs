//! Comprehensive debugging session demonstration
#![allow(unused_variables)]
//!
//! This example shows how to use all debugging tools together in a typical
//! machine learning training scenario.

use anyhow::Result;
use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_core::random::*;
use std::time::Duration;
use tokio::time::sleep;

use trustformers_debug::model_diagnostics::ModelArchitectureInfo;
use trustformers_debug::{
    debug_session_with_config, hooks::AlertSeverity, DebugConfig, DebugSession, DebugVisualizer,
    HookAction, HookBuilder, HookTrigger, LayerActivationStats, ModelPerformanceMetrics,
    VisualizationConfig,
};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("TrustformeRS Advanced Debugging Tools Demo");
    println!("==========================================\n");

    // Example 1: Basic debugging session
    basic_debugging_session().await?;

    // Example 2: Comprehensive model debugging
    comprehensive_model_debugging().await?;

    // Example 3: Automated debugging with hooks
    automated_debugging_with_hooks().await?;

    // Example 4: Performance profiling
    performance_profiling_demo().await?;

    // Example 5: Visualization examples
    visualization_demo().await?;

    Ok(())
}

async fn basic_debugging_session() -> Result<()> {
    println!("=== Example 1: Basic Debugging Session ===");

    let config = DebugConfig {
        enable_model_diagnostics: false, // Keep it simple for this example
        max_tracked_tensors: 100,
        max_gradient_history: 50,
        output_dir: Some("./debug_output".to_string()),
        ..DebugConfig::default()
    };

    let mut debug_session = DebugSession::new(config);
    debug_session.start().await?;

    println!("✓ Started debugging session: {}", debug_session.id());

    // Simulate some tensor operations
    println!("\n--- Simulating tensor operations ---");

    // Create some sample tensors
    let input_tensor = Array::linspace(0.0, 1.0, 1000)
        .into_shape_with_order(IxDyn(&[10, 10, 10]))
        .unwrap();
    let weight_tensor = Array::<f32, _>::ones(IxDyn(&[10, 10])) * 0.5;

    // Inspect tensors
    let input_id = debug_session.tensor_inspector_mut().inspect_tensor(
        &input_tensor,
        "input",
        Some("input_layer"),
        Some("forward"),
    )?;

    let weight_id = debug_session.tensor_inspector_mut().inspect_tensor(
        &weight_tensor,
        "weights",
        Some("linear_layer"),
        Some("parameter"),
    )?;

    println!(
        "✓ Inspected tensors: input={}, weights={}",
        input_id, weight_id
    );

    // Simulate gradient computation
    let gradients = Array::from_elem(IxDyn(&[10, 10]), 0.01); // Small gradients
    debug_session.tensor_inspector_mut().inspect_gradients(weight_id, &gradients)?;

    println!("✓ Recorded gradients for weight tensor");

    // Record gradient flow for debugging
    let gradient_values: Vec<f64> = gradients.iter().cloned().collect();
    let gradient_norm = (gradient_values.iter().map(|&x| x * x).sum::<f64>()).sqrt();
    let gradient_mean = gradient_values.iter().sum::<f64>() / gradient_values.len() as f64;
    let gradient_variance =
        gradient_values.iter().map(|&x| (x - gradient_mean).powi(2)).sum::<f64>()
            / gradient_values.len() as f64;
    let gradient_std = gradient_variance.sqrt();

    debug_session.gradient_debugger_mut().record_gradient_flow(
        "linear_layer",
        gradient_norm,
        gradient_mean,
        gradient_std,
    )?;

    debug_session.gradient_debugger_mut().next_step();

    println!("✓ Recorded gradient flow");

    // Compare tensors
    let comparison = debug_session.tensor_inspector_mut().compare_tensors(input_id, weight_id)?;

    println!(
        "✓ Tensor comparison - MSE: {:.6}, Shape match: {}",
        comparison.mse, comparison.shape_match
    );

    // Generate and display report
    let report = debug_session.stop().await?;

    println!("\n--- Debug Session Report ---");
    println!("Session ID: {}", report.session_id);

    if let Some(ref tensor_report) = report.tensor_report {
        println!("Tensors tracked: {}", tensor_report.total_tensors);
        println!("Tensors with issues: {}", tensor_report.tensors_with_issues);
        println!("Alerts: {}", tensor_report.alerts.len());

        for alert in &tensor_report.alerts {
            println!("  - {:?}: {}", alert.severity, alert.message);
        }
    }

    if let Some(ref gradient_report) = report.gradient_report {
        println!(
            "Gradient flow tracked for {} layers",
            gradient_report.status.active_layers
        );
        println!("Current step: {}", gradient_report.status.current_step);
        println!("Alerts: {}", gradient_report.status.recent_alerts.len());
    }

    let summary = report.summary();
    println!("\nSummary:");
    println!("  Total issues: {}", summary.total_issues);
    println!("  Critical issues: {}", summary.critical_issues);

    for recommendation in &summary.recommendations {
        println!("  💡 {}", recommendation);
    }

    println!("✅ Basic debugging session completed\n");
    Ok(())
}

async fn comprehensive_model_debugging() -> Result<()> {
    println!("=== Example 2: Comprehensive Model Debugging ===");

    let config = DebugConfig {
        max_tracked_tensors: 500,
        max_gradient_history: 200,
        output_dir: Some("./debug_comprehensive".to_string()),
        ..DebugConfig::default()
    };

    let mut debug_session = DebugSession::new(config);
    debug_session.start().await?;

    println!("✓ Started comprehensive debugging session");

    // Record model architecture
    let arch_info = ModelArchitectureInfo {
        total_parameters: 125_000_000,
        trainable_parameters: 125_000_000,
        model_size_mb: 500.0,
        layer_count: 24,
        layer_types: {
            let mut types = std::collections::HashMap::new();
            types.insert("TransformerBlock".to_string(), 12);
            types.insert("Linear".to_string(), 36);
            types.insert("LayerNorm".to_string(), 24);
            types.insert("Attention".to_string(), 12);
            types
        },
        depth: 24,
        width: 1024,
        activation_functions: {
            let mut activations = std::collections::HashMap::new();
            activations.insert("GELU".to_string(), 12);
            activations.insert("Softmax".to_string(), 12);
            activations
        },
    };

    debug_session.model_diagnostics_mut().record_architecture(arch_info);
    println!("✓ Recorded model architecture (125M parameters, 24 layers)");

    // Simulate training loop with performance metrics
    println!("\n--- Simulating training loop ---");

    for step in 0..20 {
        // Simulate performance metrics
        let loss = 2.5 * (-0.1 * step as f64).exp() + 0.1 + (step as f64 * 0.01).sin() * 0.05;
        let accuracy = 1.0 - loss / 3.0;

        let metrics = ModelPerformanceMetrics {
            training_step: step,
            loss,
            accuracy: Some(accuracy.max(0.0).min(1.0)),
            learning_rate: 1e-4,
            batch_size: 32,
            throughput_samples_per_sec: 150.0 + (step as f64 * 2.0),
            memory_usage_mb: 2048.0 + step as f64 * 10.0,
            gpu_utilization: Some(85.0 + (step as f64 * 0.5).sin() * 5.0),
            timestamp: chrono::Utc::now(),
        };

        debug_session.model_diagnostics_mut().record_performance(metrics)?;

        // Simulate layer activations
        for layer_idx in 0..12 {
            let layer_name = format!("transformer_block_{}", layer_idx);
            let activation_stats = LayerActivationStats {
                layer_name: layer_name.clone(),
                mean_activation: 0.5 + (step as f64 * 0.1 + layer_idx as f64).sin() * 0.1,
                std_activation: 0.2 + (step as f64 * 0.05).cos() * 0.05,
                min_activation: -0.1,
                max_activation: 1.2,
                dead_neurons_ratio: 0.05 + (layer_idx as f64 * 0.01),
                saturated_neurons_ratio: 0.02,
                sparsity: 0.3 + (layer_idx as f64 * 0.02),
                output_shape: vec![32, 512, 1024],
            };

            debug_session
                .model_diagnostics_mut()
                .record_layer_stats(activation_stats)
                .expect("Failed to record layer stats");
        }

        // Simulate gradient recording
        for layer_idx in 0..12 {
            let layer_name = format!("transformer_block_{}", layer_idx);

            // Create realistic gradient patterns
            let base_gradient_norm =
                0.1 * (step as f64 * 0.2).exp() * (1.0 + layer_idx as f64 * 0.1);
            let gradient_values: Vec<f64> =
                (0..1000).map(|i| base_gradient_norm * (i as f64 * 0.01).sin()).collect();

            let gradient_norm = (gradient_values.iter().map(|&x| x * x).sum::<f64>()).sqrt();
            let gradient_mean = gradient_values.iter().sum::<f64>() / gradient_values.len() as f64;
            let gradient_variance =
                gradient_values.iter().map(|&x| (x - gradient_mean).powi(2)).sum::<f64>()
                    / gradient_values.len() as f64;
            let gradient_std = gradient_variance.sqrt();

            debug_session.gradient_debugger_mut().record_gradient_flow(
                &layer_name,
                gradient_norm,
                gradient_mean,
                gradient_std,
            )?;
        }

        debug_session.gradient_debugger_mut().next_step();

        if step % 5 == 0 {
            println!(
                "  Step {}: loss={:.4}, accuracy={:.3}",
                step,
                loss,
                accuracy.max(0.0)
            );
        }
    }

    println!("✓ Completed 20 training steps");

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
    println!(
        "Overfitting indicators: {}",
        training_dynamics.overfitting_indicators.len()
    );
    println!(
        "Underfitting indicators: {}",
        training_dynamics.underfitting_indicators.len()
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

    // Generate final report
    let report = debug_session.stop().await?;

    println!("\n--- Comprehensive Debug Report ---");
    if let Some(ref diagnostics_report) = report.diagnostics_report {
        println!(
            "Training analyzed for {} steps",
            diagnostics_report.current_step
        );
        println!("Performance alerts: {}", diagnostics_report.alerts.len());

        for recommendation in &diagnostics_report.recommendations {
            println!("  💡 {}", recommendation);
        }
    }

    println!("✅ Comprehensive model debugging completed\n");
    Ok(())
}

async fn automated_debugging_with_hooks() -> Result<()> {
    println!("=== Example 3: Automated Debugging with Hooks ===");

    let mut debug_session = debug_session_with_config(DebugConfig::default());
    debug_session.start().await?;

    println!("✓ Started debugging session with hooks");

    // Create various debugging hooks
    println!("\n--- Setting up debugging hooks ---");

    // 1. Tensor inspection hook for specific layers
    let tensor_hook = HookBuilder::new("Tensor Inspector")
        .trigger(HookTrigger::EveryForward)
        .action(HookAction::InspectTensor)
        .layer_patterns(vec!["attention.*".to_string(), "linear.*".to_string()])
        .build();

    let tensor_hook_id = debug_session.hooks_mut().register_hook(tensor_hook)?;
    println!("✓ Registered tensor inspection hook: {}", tensor_hook_id);

    // 2. Gradient tracking hook
    let gradient_hook = HookBuilder::new("Gradient Tracker")
        .trigger(HookTrigger::EveryBackward)
        .action(HookAction::TrackGradients)
        .layer_patterns(vec![".*".to_string()])
        .max_executions(100)
        .build();

    let gradient_hook_id = debug_session.hooks_mut().register_hook(gradient_hook)?;
    println!("✓ Registered gradient tracking hook: {}", gradient_hook_id);

    // 3. Alert hook for extreme values
    use trustformers_debug::HookCondition;
    let alert_hook = HookBuilder::new("NaN Alert")
        .trigger(HookTrigger::Conditional(HookCondition::Custom(
            "check_nan".to_string(),
        )))
        .action(HookAction::Alert {
            message: "NaN values detected in tensor!".to_string(),
            severity: AlertSeverity::Critical,
        })
        .build();

    let alert_hook_id = debug_session.hooks_mut().register_hook(alert_hook)?;
    println!("✓ Registered NaN detection hook: {}", alert_hook_id);

    // 4. Snapshot saving hook
    let snapshot_hook = HookBuilder::new("Tensor Snapshots")
        .trigger(HookTrigger::EveryNSteps(5))
        .action(HookAction::SaveSnapshot {
            path: "./debug_snapshots/tensor".to_string(),
        })
        .layer_patterns(vec!["attention_output".to_string()])
        .max_executions(10)
        .build();

    let snapshot_hook_id = debug_session.hooks_mut().register_hook(snapshot_hook)?;
    println!("✓ Registered tensor snapshot hook: {}", snapshot_hook_id);

    // Simulate model execution with hook triggers
    println!("\n--- Simulating model execution ---");

    for step in 0..15 {
        debug_session.hooks_mut().set_step(step);

        // Simulate forward pass through different layers
        let layers = vec![
            "attention_input",
            "attention_output",
            "linear_1",
            "linear_2",
        ];

        for layer_name in &layers {
            // Create sample tensor data
            let tensor_data: Vec<f32> =
                (0..100).map(|i| (i as f32 * 0.01 + step as f32 * 0.1).sin()).collect();

            let tensor_shape = vec![10, 10];
            let metadata = std::collections::HashMap::new();

            // Execute hooks for forward pass
            let hook_results = debug_session.hooks_mut().execute_hooks(
                layer_name,
                &tensor_data,
                &tensor_shape,
                true, // is_forward
                Some(metadata.clone()),
            );

            // Log hook execution results
            for (hook_id, result) in hook_results {
                match result {
                    trustformers_debug::hooks::HookResult::Success => {
                        if step == 0 {
                            println!(
                                "  ✓ Hook {} executed successfully for {}",
                                hook_id, layer_name
                            );
                        }
                    },
                    trustformers_debug::hooks::HookResult::Error(err) => {
                        println!("  ❌ Hook {} failed for {}: {}", hook_id, layer_name, err);
                    },
                    trustformers_debug::hooks::HookResult::Skipped(reason) => {
                        if step == 0 {
                            println!(
                                "  ⏭ Hook {} skipped for {}: {}",
                                hook_id, layer_name, reason
                            );
                        }
                    },
                }
            }

            // Simulate backward pass for some layers
            if layer_name.contains("linear") {
                let hook_results = debug_session.hooks_mut().execute_hooks(
                    layer_name,
                    &tensor_data,
                    &tensor_shape,
                    false, // is_forward = false (backward pass)
                    Some(metadata),
                );

                // Process backward pass hook results (similar to forward)
                for (hook_id, result) in hook_results {
                    if matches!(result, trustformers_debug::hooks::HookResult::Success) && step == 0
                    {
                        println!("  ✓ Gradient hook {} executed for {}", hook_id, layer_name);
                    }
                }
            }
        }

        if step % 3 == 0 {
            println!("  Step {} completed with hook execution", step);
        }
    }

    // Display hook statistics
    println!("\n--- Hook Execution Statistics ---");
    let all_stats = debug_session.hooks().get_all_stats();
    for stats in all_stats {
        println!(
            "Hook '{}': {} executions, avg time: {:.2}ms",
            stats.hook_name, stats.total_executions, stats.avg_execution_time_ms
        );
    }

    // Generate report
    let report = debug_session.stop().await?;
    println!("\n--- Automated Debugging Report ---");

    let summary = report.summary();
    println!("Issues detected by hooks: {}", summary.total_issues);
    for issue in &summary.issues {
        println!("  ⚠ {}", issue);
    }

    println!("✅ Automated debugging with hooks completed\n");
    Ok(())
}

async fn performance_profiling_demo() -> Result<()> {
    println!("=== Example 4: Performance Profiling ===");

    let mut debug_session = debug_session_with_config(DebugConfig::default());
    debug_session.start().await?;

    println!("✓ Started debugging session with profiler");

    let profiler = debug_session.profiler_mut();

    // Simulate different types of operations
    println!("\n--- Simulating model operations ---");

    // 1. Layer execution profiling
    for layer_idx in 0..8 {
        let layer_name = format!("transformer_layer_{}", layer_idx);

        profiler.start_timer(&format!("{}_forward", layer_name));

        // Simulate computation time (varies by layer)
        let computation_time = Duration::from_millis(50 + layer_idx * 10);
        sleep(computation_time).await;

        let forward_time =
            profiler.end_timer(&format!("{}_forward", layer_name)).unwrap_or(Duration::ZERO);

        // Simulate backward pass
        profiler.start_timer(&format!("{}_backward", layer_name));
        sleep(Duration::from_millis(30 + layer_idx * 5)).await;
        let backward_time = profiler.end_timer(&format!("{}_backward", layer_name));

        // Record layer execution
        profiler.record_layer_execution(
            &layer_name,
            "TransformerBlock",
            forward_time,
            backward_time,
            (1024 * 1024 * (layer_idx + 1)) as usize, // Memory usage
            (1_000_000 + layer_idx * 100_000) as usize, // Parameter count
        );

        println!(
            "  ✓ Profiled layer {}: forward={:.1}ms",
            layer_idx,
            forward_time.as_millis()
        );
    }

    // 2. Tensor operation profiling
    let tensor_ops = vec![
        ("matrix_multiply", vec![512, 512], 25),
        ("softmax", vec![32, 128], 15),
        ("layer_norm", vec![32, 512], 10),
        ("attention", vec![32, 512, 64], 40),
    ];

    for (op_name, shape, duration_ms) in tensor_ops {
        profiler.record_tensor_operation(
            op_name,
            &shape,
            Duration::from_millis(duration_ms),
            shape.iter().product::<usize>() * 4, // 4 bytes per float
        );

        println!("  ✓ Profiled tensor op '{}': {:.1}ms", op_name, duration_ms);
    }

    // 3. Model inference profiling
    for batch_size in [1, 8, 32] {
        let sequence_length = 128;
        let inference_time = Duration::from_millis(batch_size as u64 * 50);

        profiler.record_model_inference(batch_size, sequence_length, inference_time);
        println!(
            "  ✓ Profiled inference: batch_size={}, time={:.1}ms",
            batch_size,
            inference_time.as_millis()
        );
    }

    // 4. Gradient computation profiling
    for layer_idx in 0..5 {
        let layer_name = format!("linear_{}", layer_idx);
        let gradient_norm = 0.1 * (layer_idx as f64 + 1.0);
        let grad_time = Duration::from_millis(20 + layer_idx as u64 * 5);

        profiler.record_gradient_computation(&layer_name, gradient_norm, grad_time);
    }

    // 5. Take memory snapshots
    for _ in 0..5 {
        profiler.take_memory_snapshot();
        sleep(Duration::from_millis(100)).await;
    }

    // Analyze performance
    println!("\n--- Performance Analysis ---");
    let bottlenecks = profiler.analyze_performance();

    println!("Detected {} performance bottlenecks:", bottlenecks.len());
    for bottleneck in &bottlenecks {
        println!(
            "  {:?} in {}: {}",
            bottleneck.severity, bottleneck.location, bottleneck.description
        );
        println!("    💡 {}", bottleneck.suggestion);
    }

    // Get profiling statistics
    let statistics = profiler.get_statistics();
    println!("\n--- Profiling Statistics ---");
    for (event_type, stats) in &statistics {
        println!(
            "{}: {} events, avg time: {:.1}ms",
            event_type,
            stats.count,
            stats.avg_duration.as_millis()
        );
    }

    // Get slowest layers
    let slowest_layers = profiler.get_layer_profiles();
    println!("\n--- Layer Performance ---");
    let mut layer_times: Vec<(String, Duration)> = slowest_layers
        .iter()
        .map(|(name, profile)| {
            let avg_time = if profile.forward_times().is_empty() {
                Duration::ZERO
            } else {
                profile.forward_times().iter().sum::<Duration>()
                    / profile.forward_times().len() as u32
            };
            (name.clone(), avg_time)
        })
        .collect();

    layer_times.sort_by(|a, b| b.1.cmp(&a.1));

    for (i, (layer_name, avg_time)) in layer_times.iter().take(5).enumerate() {
        println!(
            "  {}: {} - {:.1}ms avg",
            i + 1,
            layer_name,
            avg_time.as_millis()
        );
    }

    // Generate profiler report
    let report = debug_session.stop().await?;
    println!("\n--- Profiler Report ---");

    let profiler_report = &report.profiler_report;
    println!("Total events: {}", profiler_report.total_events);
    println!(
        "Total runtime: {:.1}s",
        profiler_report.total_runtime.as_secs_f64()
    );
    println!("Performance recommendations:");
    for recommendation in &profiler_report.recommendations {
        println!("  💡 {}", recommendation);
    }

    println!("✅ Performance profiling completed\n");
    Ok(())
}

async fn visualization_demo() -> Result<()> {
    println!("=== Example 5: Visualization Demo ===");

    let viz_config = VisualizationConfig {
        output_directory: "./debug_visualizations".to_string(),
        ..Default::default()
    };

    let mut visualizer = DebugVisualizer::new(viz_config);

    println!("✓ Created debug visualizer");

    // 1. Plot tensor distribution
    println!("\n--- Creating tensor distribution plot ---");
    let mut rng = thread_rng();
    let tensor_values: Vec<f64> = (0..1000)
        .map(|i| {
            let x = i as f64 * 0.01;
            x.sin() + 0.5 * (2.0 * x).cos() + (rng.random::<f64>() - 0.5) * 0.1
        })
        .collect();

    let dist_plot = visualizer.plot_tensor_distribution("sample_tensor", &tensor_values, 30)?;
    println!("✓ Created tensor distribution plot: {}", dist_plot);

    // 2. Plot gradient flow
    println!("\n--- Creating gradient flow plot ---");
    let steps: Vec<f64> = (0..50).map(|x| x as f64).collect();
    let gradient_norms: Vec<f64> = steps
        .iter()
        .map(|&step| 0.1 * (-0.02 * step).exp() + 0.01 * (step * 0.3).sin().abs())
        .collect();

    let grad_plot = visualizer.plot_gradient_flow("attention_layer", &steps, &gradient_norms)?;
    println!("✓ Created gradient flow plot: {}", grad_plot);

    // 3. Plot training metrics
    println!("\n--- Creating training metrics plot ---");
    let train_steps: Vec<f64> = (0..100).map(|x| x as f64).collect();
    let losses: Vec<f64> =
        train_steps.iter().map(|&step| 2.0 * (-0.05 * step).exp() + 0.1).collect();
    let accuracies: Vec<f64> =
        losses.iter().map(|&loss| (1.0 - loss / 2.5).max(0.0).min(1.0)).collect();

    let metrics_plot =
        visualizer.plot_training_metrics(&train_steps, &losses, Some(&accuracies))?;
    println!("✓ Created training metrics plot: {}", metrics_plot);

    // 4. Plot tensor heatmap
    println!("\n--- Creating tensor heatmap ---");
    let heatmap_data: Vec<Vec<f64>> = (0..20)
        .map(|i| (0..20).map(|j| (i as f64 * 0.1).sin() * (j as f64 * 0.1).cos()).collect())
        .collect();

    let heatmap_plot = visualizer.plot_tensor_heatmap("weight_matrix", &heatmap_data)?;
    println!("✓ Created tensor heatmap: {}", heatmap_plot);

    // 5. Plot activation patterns
    println!("\n--- Creating activation patterns plot ---");
    let activation_steps: Vec<usize> = (0..30).collect();
    let mean_activations: Vec<f64> = activation_steps
        .iter()
        .map(|&step| 0.5 + 0.2 * (step as f64 * 0.2).sin())
        .collect();
    let std_activations: Vec<f64> = activation_steps
        .iter()
        .map(|&step| 0.1 + 0.05 * (step as f64 * 0.15).cos().abs())
        .collect();

    let activation_plot =
        visualizer.plot_activation_patterns("hidden_layer", &mean_activations, &std_activations)?;
    println!("✓ Created activation patterns plot: {}", activation_plot);

    // 6. Create dashboard
    println!("\n--- Creating debug dashboard ---");
    let plot_names = visualizer.get_plot_names();
    let dashboard_path = visualizer.create_dashboard(&plot_names)?;
    println!("✓ Created debug dashboard: {}", dashboard_path);

    // 7. Terminal visualization demo
    println!("\n--- Terminal Visualization Demo ---");
    use trustformers_debug::TerminalVisualizer;

    let terminal_viz = TerminalVisualizer::new();

    // ASCII histogram
    let histogram = terminal_viz.ascii_histogram(&tensor_values[0..100], 10);
    println!("ASCII Histogram:");
    println!("{}", histogram);

    // ASCII line plot
    let line_plot = terminal_viz.ascii_line_plot(
        &steps[0..20].iter().map(|&x| x).collect::<Vec<_>>(),
        &gradient_norms[0..20],
        "Gradient Norm Over Time",
    );
    println!("ASCII Line Plot:");
    println!("{}", line_plot);

    // Export plot data
    for plot_name in &plot_names {
        let export_path = format!("./debug_visualizations/{}.json", plot_name);
        let export_path = std::path::Path::new(&export_path);
        visualizer.export_plot_data(plot_name, export_path)?;
    }
    println!("✓ Exported plot data as JSON files");

    println!("✅ Visualization demo completed\n");
    Ok(())
}

// Helper function to simulate some delay
#[allow(dead_code)]
async fn simulate_computation(duration_ms: u64) {
    sleep(Duration::from_millis(duration_ms)).await;
}
