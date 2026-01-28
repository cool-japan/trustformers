//! MLflow Integration Demo
//!
//! This example demonstrates how to use the MLflow integration for experiment tracking
//! in TrustformeRS debugging workflows.

use anyhow::Result;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;
use trustformers_debug::{MLflowClient, MLflowConfig, MLflowDebugSession, RunStatus};

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("=== MLflow Integration Demo ===\n");

    // Demo 1: Basic MLflow Client Usage
    demo_basic_client()?;

    // Demo 2: MLflow Debug Session
    demo_debug_session()?;

    // Demo 3: Logging Metrics Over Time
    demo_metrics_tracking()?;

    // Demo 4: Artifact Logging
    demo_artifact_logging()?;

    println!("\n=== Demo Complete ===");

    Ok(())
}

fn demo_basic_client() -> Result<()> {
    println!("--- Demo 1: Basic MLflow Client ---\n");

    // Create MLflow client with default config
    let config = MLflowConfig {
        tracking_uri: "http://localhost:5000".to_string(),
        experiment_name: "trustformers-demo".to_string(),
        auto_log: true,
        log_interval: 10,
        max_cache_size: 1000,
        log_artifacts: true,
        artifact_dir: std::env::temp_dir().join("mlflow_artifacts"),
    };

    let mut client = MLflowClient::new(config);

    // Start an experiment and run
    client.start_experiment("transformer-training")?;
    client.start_run(Some("initial-run"))?;

    // Log hyperparameters
    client.log_param("learning_rate", "0.001")?;
    client.log_param("batch_size", "32")?;
    client.log_param("model", "gpt2")?;
    client.log_param("optimizer", "adamw")?;

    println!("✓ Logged 4 hyperparameters");

    // Log training metrics
    for step in 0..10 {
        let loss = 2.0 * (-0.1 * step as f64).exp(); // Simulated decreasing loss
        let accuracy = 0.5 + 0.4 * (1.0 - (-0.1 * step as f64).exp()); // Increasing accuracy

        client.log_metric("train/loss", loss, step)?;
        client.log_metric("train/accuracy", accuracy, step)?;

        if step % 5 == 0 {
            println!(
                "  Step {}: loss={:.4}, accuracy={:.4}",
                step, loss, accuracy
            );
        }
    }

    println!("✓ Logged 10 training steps");

    // End the run
    client.end_run(RunStatus::Finished)?;

    println!("✓ Run completed successfully\n");

    Ok(())
}

fn demo_debug_session() -> Result<()> {
    println!("--- Demo 2: MLflow Debug Session ---\n");

    let config = MLflowConfig::default();
    let mut session = MLflowDebugSession::new(config);

    // Start debugging session
    session.start("debugging-experiment", Some("gradient-check"))?;

    println!("✓ Started debug session");

    // Log debugging metrics
    for step in 0..5 {
        let mut metrics = HashMap::new();
        metrics.insert("gradient_norm".to_string(), 0.1 * (1.0 - step as f64 * 0.1));
        metrics.insert("activation_mean".to_string(), 0.5 + step as f64 * 0.01);
        metrics.insert("weight_variance".to_string(), 1.0 - step as f64 * 0.05);

        session.log_debug_metrics(metrics)?;
    }

    println!("✓ Logged debug metrics for 5 steps");

    // End session
    session.end(RunStatus::Finished)?;

    println!("✓ Debug session completed\n");

    Ok(())
}

fn demo_metrics_tracking() -> Result<()> {
    println!("--- Demo 3: Metrics Tracking ---\n");

    let config = MLflowConfig::default();
    let mut client = MLflowClient::new(config);

    client.start_experiment("metrics-demo")?;
    client.start_run(Some("array-stats"))?;

    // Create sample data
    let data = Array1::from_vec(vec![0.5, 1.2, 0.8, 1.5, 0.9, 1.1, 0.7, 1.3]);

    println!("✓ Created sample data array (8 elements)");

    // Log array statistics
    client.log_array_stats("layer_weights", &data, 0)?;

    println!("✓ Logged array statistics (mean, std, min, max)");

    // Log multiple metrics at once
    let mut batch_metrics = HashMap::new();
    batch_metrics.insert("throughput".to_string(), 1000.0);
    batch_metrics.insert("latency_ms".to_string(), 15.5);
    batch_metrics.insert("memory_usage_mb".to_string(), 512.0);

    client.log_metrics(batch_metrics, 0)?;

    println!("✓ Logged batch metrics (throughput, latency, memory)");

    client.end_run(RunStatus::Finished)?;

    println!("✓ Metrics tracking completed\n");

    Ok(())
}

fn demo_artifact_logging() -> Result<()> {
    println!("--- Demo 4: Artifact Logging ---\n");

    let config = MLflowConfig::default();
    let _client = MLflowClient::new(config);

    // Note: We don't start experiment/run here to show artifact logging
    // can be done independently or you would need to start a run first

    // Create temporary files for demo
    let temp_dir = std::env::temp_dir();

    // Create a sample report
    let report_path = temp_dir.join("debug_report.txt");
    let report_content = "=== Debug Report ===\n\
                          Model: GPT-2\n\
                          Status: Healthy\n\
                          Issues: None detected\n\
                          Recommendations: Continue training";

    std::fs::write(&report_path, report_content)?;

    println!("✓ Created sample debug report");

    // In a real scenario with an active run, you would log it:
    // client.log_report(report_content, "debug_report.txt")?;

    println!("  (Artifact logging requires active run - skipping actual upload)");

    // Clean up
    std::fs::remove_file(&report_path)?;

    println!("✓ Artifact demo completed\n");

    Ok(())
}
