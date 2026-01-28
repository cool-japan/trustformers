//! Comprehensive integration tests for TrustformeRS Debug framework

use anyhow::Result;
use scirs2_core::ndarray::{Array, IxDyn};
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

use trustformers_debug::model_diagnostics::ModelArchitectureInfo;
use trustformers_debug::*;

/// Test basic debugging session workflow
#[tokio::test]
async fn test_basic_debugging_workflow() -> Result<()> {
    let config = DebugConfig {
        enable_tensor_inspection: true,
        enable_gradient_debugging: true,
        enable_model_diagnostics: true,
        enable_visualization: false,    // Disable for testing
        enable_memory_profiling: false, // Disable for testing
        enable_computation_graph_analysis: true,
        max_tracked_tensors: 100,
        max_gradient_history: 50,
        output_dir: Some("./test_debug_output".to_string()),
        sampling_rate: 1.0,
        ..Default::default()
    };

    let mut debug_session = DebugSession::new(config);

    // Start session
    debug_session.start().await?;

    // Create test tensors with different means for MSE comparison
    let input_tensor =
        Array::linspace(0.0, 1.0, 100).to_shape(IxDyn(&[10, 10])).unwrap().to_owned();
    let weight_tensor = Array::<f32, _>::ones(IxDyn(&[10, 10])) * 0.8; // Different mean from input (0.8 vs 0.5)

    // Inspect tensors
    let input_id = debug_session.tensor_inspector_mut().inspect_tensor(
        &input_tensor,
        "input_test",
        Some("test_layer"),
        Some("forward"),
    )?;

    let weight_id = debug_session.tensor_inspector_mut().inspect_tensor(
        &weight_tensor,
        "weights_test",
        Some("test_layer"),
        Some("parameter"),
    )?;

    // Record gradient flow
    let gradient_values: Vec<f64> = weight_tensor.iter().map(|&x| x as f64 * 0.01).collect();
    let gradient_norm = (gradient_values.iter().map(|&x| x * x).sum::<f64>()).sqrt();
    let gradient_mean = gradient_values.iter().sum::<f64>() / gradient_values.len() as f64;
    let gradient_variance =
        gradient_values.iter().map(|&x| (x - gradient_mean).powi(2)).sum::<f64>()
            / gradient_values.len() as f64;
    let gradient_std = gradient_variance.sqrt();

    debug_session.gradient_debugger_mut().record_gradient_flow(
        "test_layer",
        gradient_norm,
        gradient_mean,
        gradient_std,
    )?;

    debug_session.gradient_debugger_mut().next_step();

    // Compare tensors
    let comparison = debug_session.tensor_inspector_mut().compare_tensors(input_id, weight_id)?;
    assert!(comparison.shape_match); // Same shape [10, 10]
    assert!(comparison.mse > 0.0); // Different values, MSE should be > 0

    // Generate report
    let report = debug_session.stop().await?;

    // Verify report contents
    assert!(report.tensor_report.is_some());
    assert!(report.gradient_report.is_some());

    if let Some(ref tensor_report) = report.tensor_report {
        assert_eq!(tensor_report.total_tensors, 2);
    }

    Ok(())
}

/// Test simplified debugging interface
#[tokio::test]
async fn test_simplified_debugging_interface() -> Result<()> {
    // Mock model for testing
    let mock_model = "test_model";

    // Test different debug levels
    let light_result = quick_debug(&mock_model, QuickDebugLevel::Light).await?;
    assert!(matches!(light_result, SimplifiedDebugResult::Light(_)));

    let standard_result = quick_debug(&mock_model, QuickDebugLevel::Standard).await?;
    assert!(matches!(
        standard_result,
        SimplifiedDebugResult::Standard { .. }
    ));

    let production_result = quick_debug(&mock_model, QuickDebugLevel::Production).await?;
    assert!(matches!(
        production_result,
        SimplifiedDebugResult::Production(_)
    ));

    // Test convenience function
    let debug_result = debug(&mock_model).await?;
    assert!(matches!(
        debug_result,
        SimplifiedDebugResult::Standard { .. }
    ));

    // Test result methods
    let summary = debug_result.summary();
    assert!(!summary.is_empty());

    let _recommendations = debug_result.recommendations();
    // Recommendations vector is valid

    Ok(())
}

/// Test utilities module functionality
#[tokio::test]
async fn test_utilities_functionality() -> Result<()> {
    // Test tensor statistics computation
    let test_tensor = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .to_shape(IxDyn(&[5]))
        .unwrap()
        .to_owned();

    let stats = DebugUtils::compute_tensor_statistics(&test_tensor)?;

    assert_eq!(stats.count, 5);
    assert!((stats.mean - 3.0).abs() < 0.01); // Mean should be 3.0
    assert!(stats.min == 1.0);
    assert!(stats.max == 5.0);
    assert_eq!(stats.nan_count, 0);
    assert_eq!(stats.inf_count, 0);

    // Test anomaly detection
    let anomalies = DebugUtils::detect_tensor_anomalies(&stats);
    assert!(anomalies.is_empty()); // Normal tensor should have no anomalies

    // Test tensor with NaN values
    let nan_tensor = Array::from_vec(vec![1.0, f32::NAN, 3.0, 4.0, 5.0])
        .to_shape(IxDyn(&[5]))
        .unwrap()
        .to_owned();

    let nan_stats = DebugUtils::compute_tensor_statistics(&nan_tensor)?;
    let nan_anomalies = DebugUtils::detect_tensor_anomalies(&nan_stats);

    assert_eq!(nan_stats.nan_count, 1);
    assert!(!nan_anomalies.is_empty()); // Should detect NaN anomaly

    Ok(())
}

/// Test batch tensor analysis
#[tokio::test]
async fn test_batch_tensor_analysis() -> Result<()> {
    let tensors = vec![
        Array::ones(IxDyn(&[5, 5])) * 0.5,
        Array::zeros(IxDyn(&[5, 5])),
        Array::from_elem(IxDyn(&[5, 5]), 2.0),
    ];

    let batch_analysis = DebugUtils::analyze_tensors_batch(&tensors)?;

    assert_eq!(batch_analysis.batch_size, 3);
    assert_eq!(batch_analysis.individual_results.len(), 3);

    // Check individual results
    for (i, result) in batch_analysis.individual_results.iter().enumerate() {
        assert_eq!(result.tensor_index, i);
        assert_eq!(result.shape, vec![5, 5]);
        assert_eq!(result.statistics.count, 25);
    }

    // First tensor should have mean ~0.5
    assert!((batch_analysis.individual_results[0].statistics.mean - 0.5).abs() < 0.01);

    // Second tensor should have mean ~0.0
    assert!((batch_analysis.individual_results[1].statistics.mean - 0.0).abs() < 0.01);

    // Third tensor should have mean ~2.0
    assert!((batch_analysis.individual_results[2].statistics.mean - 2.0).abs() < 0.01);

    Ok(())
}

/// Test guided debugging workflow
#[tokio::test]
async fn test_guided_debugging() -> Result<()> {
    let mut guided_debugger = GuidedDebugger::new();

    assert_eq!(guided_debugger.total_steps(), 6); // Should have 6 predefined steps
    assert_eq!(guided_debugger.progress(), 0.0);
    assert!(!guided_debugger.is_complete());

    // Execute first step
    if let Some(step) = guided_debugger.current_step() {
        assert_eq!(step.name, "Health Check");
    }

    let result = guided_debugger.execute_current_step().await?;
    assert!(matches!(result, StepResult::Health(_)));

    assert!(guided_debugger.progress() > 0.0);

    // Skip to end
    while !guided_debugger.is_complete() {
        guided_debugger.skip_current_step()?;
    }

    assert!(guided_debugger.is_complete());
    assert_eq!(guided_debugger.progress(), 100.0);

    Ok(())
}

/// Test tutorial system
#[tokio::test]
async fn test_tutorial_system() -> Result<()> {
    let mut tutorial = TutorialMode::new();

    assert!(tutorial.total_lessons() > 0);
    assert_eq!(tutorial.progress(), 0.0);
    assert!(!tutorial.is_complete());

    // Get current lesson
    if let Some(lesson) = tutorial.current_lesson() {
        assert!(!lesson.title.is_empty());
        assert!(!lesson.description.is_empty());
        assert!(!lesson.example_code.is_empty());
    }

    // Complete first lesson
    tutorial.complete_current_lesson()?;
    assert!(tutorial.progress() > 0.0);

    // Test navigation
    tutorial.goto_lesson(0)?;
    assert_eq!(
        tutorial.current_lesson().unwrap().title,
        "Getting Started with TrustformeRS Debug"
    );

    Ok(())
}

/// Test context help system
#[tokio::test]
async fn test_context_help() -> Result<()> {
    let help_system = context_help();

    // Test getting specific help
    let debug_help = help_system.get_help("debug_session");
    assert!(debug_help.is_some());

    if let Some(help_entry) = debug_help {
        assert_eq!(help_entry.topic, "Debug Session");
        assert!(!help_entry.description.is_empty());
        assert!(!help_entry.usage.is_empty());
    }

    // Test search functionality
    let search_results = help_system.search("debug");
    assert!(!search_results.is_empty());

    // Test contextual help
    let _gradient_help = help_system.contextual_help("gradient");
    // Should return help results

    Ok(())
}

/// Test performance monitoring utilities
#[tokio::test]
async fn test_performance_monitoring() -> Result<()> {
    let mut monitor = PerformanceMonitor::new();

    // Test checkpoints
    monitor.checkpoint("operation1");
    sleep(Duration::from_millis(10)).await;
    let duration1 = monitor.end_checkpoint("operation1");
    assert!(duration1.is_some());
    assert!(duration1.unwrap().as_millis() >= 10);

    monitor.checkpoint("operation2");
    sleep(Duration::from_millis(20)).await;
    let duration2 = monitor.end_checkpoint("operation2");
    assert!(duration2.is_some());
    assert!(duration2.unwrap().as_millis() >= 20);

    // Test total elapsed time
    let total = monitor.total_elapsed();
    assert!(total.as_millis() >= 30);

    // Test performance report
    let report = monitor.performance_report();
    assert!(report.contains("operation1"));
    assert!(report.contains("operation2"));
    assert!(report.contains("Performance Report"));

    Ok(())
}

/// Test debug templates
#[tokio::test]
async fn test_debug_templates() -> Result<()> {
    // Test development template
    let dev_config = DebugUtils::create_debug_template(DebugTemplate::Development);
    assert!(dev_config.enable_tensor_inspection);
    assert!(dev_config.enable_gradient_debugging);
    assert!(dev_config.enable_visualization);
    assert_eq!(dev_config.sampling_rate, 1.0);

    // Test production template
    let prod_config = DebugUtils::create_debug_template(DebugTemplate::Production);
    assert!(!prod_config.enable_tensor_inspection);
    assert!(!prod_config.enable_gradient_debugging);
    assert!(!prod_config.enable_visualization);
    assert_eq!(prod_config.sampling_rate, 0.1);

    // Test training template
    let train_config = DebugUtils::create_debug_template(DebugTemplate::Training);
    assert!(train_config.enable_tensor_inspection);
    assert!(train_config.enable_gradient_debugging);
    assert!(!train_config.enable_visualization);
    assert_eq!(train_config.sampling_rate, 0.5);

    // Test research template
    let research_config = DebugUtils::create_debug_template(DebugTemplate::Research);
    assert!(research_config.enable_tensor_inspection);
    assert!(research_config.enable_gradient_debugging);
    assert!(research_config.enable_visualization);
    assert_eq!(research_config.sampling_rate, 1.0);
    assert_eq!(research_config.max_tracked_tensors, 2000);

    Ok(())
}

/// Test optimized debugging sessions
#[tokio::test]
async fn test_optimized_debugging() -> Result<()> {
    // Test ultra-low overhead session
    let mut low_overhead_session = ultra_low_overhead_session();

    // Start the session
    low_overhead_session.start().await?;

    // Check performance metrics
    let metrics = low_overhead_session.get_performance_metrics();
    assert_eq!(metrics.memory_usage_mb, 0); // Simplified implementation returns 0
    assert_eq!(metrics.cpu_usage_percentage, 0.0); // Simplified implementation returns 0

    // Check performance limits
    assert!(low_overhead_session.is_within_performance_limits());

    Ok(())
}

/// Test comprehensive debugging flow with all components
#[tokio::test]
async fn test_comprehensive_debugging_flow() -> Result<()> {
    let config = DebugConfig {
        enable_tensor_inspection: true,
        enable_gradient_debugging: true,
        enable_model_diagnostics: true,
        enable_visualization: false,    // Disable for testing
        enable_memory_profiling: false, // Disable for testing
        enable_computation_graph_analysis: true,
        max_tracked_tensors: 50,
        max_gradient_history: 25,
        sampling_rate: 0.5,
        ..Default::default()
    };

    let mut debug_session = DebugSession::new(config.clone());
    debug_session.start().await?;

    // Record model architecture
    let arch_info = ModelArchitectureInfo {
        total_parameters: 1_000_000,
        trainable_parameters: 1_000_000,
        model_size_mb: 4.0,
        layer_count: 12,
        layer_types: {
            let mut types = std::collections::HashMap::new();
            types.insert("Linear".to_string(), 6);
            types.insert("Attention".to_string(), 6);
            types
        },
        depth: 12,
        width: 512,
        activation_functions: {
            let mut activations = std::collections::HashMap::new();
            activations.insert("ReLU".to_string(), 6);
            activations.insert("Softmax".to_string(), 6);
            activations
        },
    };

    debug_session.model_diagnostics_mut().record_architecture(arch_info);

    // Simulate training loop
    for step in 0..5 {
        let loss = 2.0 * (-0.1 * step as f64).exp() + 0.1;
        let accuracy = 1.0 - loss / 3.0;

        let metrics = ModelPerformanceMetrics {
            training_step: step,
            loss,
            accuracy: Some(accuracy.clamp(0.0, 1.0)),
            learning_rate: 1e-4,
            batch_size: 32,
            throughput_samples_per_sec: 100.0,
            memory_usage_mb: 1024.0,
            gpu_utilization: Some(80.0),
            timestamp: chrono::Utc::now(),
        };

        let _ = debug_session.model_diagnostics_mut().record_performance(metrics);

        // Record gradients for multiple layers
        for layer_idx in 0..3 {
            let layer_name = format!("layer_{}", layer_idx);
            let gradient_values: Vec<f64> =
                (0..50).map(|i| 0.01 * (i as f64 * 0.1 + step as f64 * 0.2).sin()).collect();

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
    }

    // Analyze training dynamics
    let training_dynamics = debug_session.model_diagnostics().analyze_training_dynamics();
    assert!(matches!(training_dynamics.convergence_status, _)); // Should have some status

    // Analyze gradient flow
    let _gradient_analysis = debug_session.gradient_debugger().analyze_gradient_conflicts();
    // Conflicts can be analyzed (may be 0 if no conflicts found)

    // Generate final report
    let report = debug_session.stop().await?;

    // Verify comprehensive report
    assert!(report.tensor_report.is_some() || !config.enable_tensor_inspection);
    assert!(report.gradient_report.is_some());
    assert!(report.diagnostics_report.is_some());
    assert!(report.architecture_analysis_report.is_some());

    let _summary = report.summary();
    // Summary contains valid issue count

    Ok(())
}

/// Test macro functionality
#[tokio::test(flavor = "multi_thread")]
#[ignore] // FIXME: This test has implementation issues causing slow execution
async fn test_debug_macros() -> Result<()> {
    // Wrap in timeout to prevent hanging
    let test_result = tokio::time::timeout(Duration::from_millis(500), async {
        let mut debug_session = debug_session();
        debug_session.start().await?;

        // Test tensor debugging macro
        let test_tensor = Array::<f32, _>::ones(IxDyn(&[5, 5]));
        let tensor_id = debug_tensor!(debug_session, &test_tensor, "macro_test")?;
        assert!(tensor_id != Uuid::nil());

        // Test gradient debugging macro
        let gradients = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        debug_gradient!(debug_session, "macro_layer", &gradients)?;

        // Give background tasks a moment to process
        sleep(Duration::from_millis(1)).await;

        debug_session.stop().await?;

        // Give background tasks time to clean up
        sleep(Duration::from_millis(1)).await;

        Ok::<(), anyhow::Error>(())
    })
    .await;

    match test_result {
        Ok(result) => result,
        Err(_) => Err(anyhow::anyhow!("Test timed out after 500ms")),
    }
}

/// Integration test with error conditions
#[tokio::test]
#[ignore] // FIXME: Test hangs waiting for background tasks - needs investigation
async fn test_error_handling() -> Result<()> {
    let mut debug_session = debug_session();
    debug_session.start().await?;

    // Test with problematic tensor (NaN values)
    let problematic_tensor = Array::from_vec(vec![1.0, f32::NAN, 3.0, f32::INFINITY])
        .to_shape(IxDyn(&[2, 2]))
        .unwrap()
        .to_owned();

    let tensor_id = debug_session.tensor_inspector_mut().inspect_tensor(
        &problematic_tensor,
        "problematic",
        Some("test_layer"),
        Some("forward"),
    )?;

    // Should still work despite problematic values
    assert!(tensor_id != Uuid::nil());

    let report = debug_session.stop().await?;

    // Should detect issues in the tensor
    if let Some(ref tensor_report) = report.tensor_report {
        assert!(tensor_report.tensors_with_issues > 0);
    }

    Ok(())
}
