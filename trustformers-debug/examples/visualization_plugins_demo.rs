//! Visualization Plugin System Demo
//!
//! This example demonstrates how to use the custom visualization plugin system
//! to create and execute visualization plugins.

use anyhow::Result;
use std::collections::HashMap;
use trustformers_debug::{
    PluginConfig, PluginManager, PluginMetadata, PluginOutputFormat, PluginResult,
    VisualizationData, VisualizationPlugin,
};

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("=== Visualization Plugin System Demo ===\n");

    // Demo 1: List built-in plugins
    demo_list_plugins()?;

    // Demo 2: Execute histogram plugin
    demo_histogram_plugin()?;

    // Demo 3: Execute heatmap plugin
    demo_heatmap_plugin()?;

    // Demo 4: Create and register custom plugin
    demo_custom_plugin()?;

    // Demo 5: Plugin validation
    demo_plugin_validation()?;

    println!("\n=== Demo Complete ===");

    Ok(())
}

fn demo_list_plugins() -> Result<()> {
    println!("--- Demo 1: List Built-in Plugins ---\n");

    let manager = PluginManager::new();
    let plugins = manager.list_plugins();

    println!("Found {} registered plugins:\n", plugins.len());

    for plugin in plugins {
        println!("  • {} v{}", plugin.name, plugin.version);
        println!("    Description: {}", plugin.description);
        println!("    Supported inputs: {:?}", plugin.supported_inputs);
        println!("    Supported outputs: {:?}", plugin.supported_outputs);
        println!("    Tags: {:?}\n", plugin.tags);
    }

    Ok(())
}

fn demo_histogram_plugin() -> Result<()> {
    println!("--- Demo 2: Histogram Plugin ---\n");

    let manager = PluginManager::new();

    // Create sample data (simulated weights or activations)
    let data = VisualizationData::Array1D(vec![
        0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.28, 0.12, 0.19, 0.21, 0.16, 0.24, 0.27, 0.14,
        0.23, 0.17, 0.26, 0.13, 0.20,
    ]);

    // Configure plugin
    let mut config = PluginConfig::default();
    config.custom_params.insert("bins".to_string(), serde_json::json!(10));

    println!("Executing histogram plugin on {} data points...", 20);

    // Execute plugin
    let result = manager.execute("histogram", data, config)?;

    println!("\nResult:");
    println!("  Success: {}", result.success);

    if let Some(output) = result.output_data {
        let output_str = String::from_utf8_lossy(&output);
        println!("  Output:\n{}", output_str);
    }

    println!("  Metadata: {:?}\n", result.metadata);

    Ok(())
}

fn demo_heatmap_plugin() -> Result<()> {
    println!("--- Demo 3: Heatmap Plugin ---\n");

    let manager = PluginManager::new();

    // Create 2D attention matrix data
    let data = VisualizationData::Array2D(vec![
        vec![0.8, 0.1, 0.05, 0.05],
        vec![0.1, 0.7, 0.15, 0.05],
        vec![0.05, 0.2, 0.6, 0.15],
        vec![0.05, 0.1, 0.15, 0.7],
    ]);

    let config = PluginConfig::default();

    println!("Executing heatmap plugin on 4x4 matrix...");

    let result = manager.execute("heatmap", data, config)?;

    println!("\nResult:");
    println!("  Success: {}", result.success);

    if let Some(output) = result.output_data {
        let output_str = String::from_utf8_lossy(&output);
        println!("  Output: {}", output_str);
    }

    println!("  Metadata: {:?}\n", result.metadata);

    Ok(())
}

fn demo_custom_plugin() -> Result<()> {
    println!("--- Demo 4: Custom Plugin ---\n");

    let manager = PluginManager::new();

    // Register custom statistics plugin
    manager.register_plugin(Box::new(StatisticsPlugin))?;

    println!("✓ Registered custom 'statistics' plugin");

    // Use the custom plugin
    let data = VisualizationData::Array1D(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    let config = PluginConfig::default();

    let result = manager.execute("statistics", data, config)?;

    println!("\nResult:");
    println!("  Success: {}", result.success);

    if let Some(output) = result.output_data {
        let output_str = String::from_utf8_lossy(&output);
        println!("  Output:\n{}", output_str);
    }

    println!();

    Ok(())
}

fn demo_plugin_validation() -> Result<()> {
    println!("--- Demo 5: Plugin Validation ---\n");

    let manager = PluginManager::new();

    // Try to execute heatmap with 1D data (should fail validation)
    let invalid_data = VisualizationData::Array1D(vec![1.0, 2.0, 3.0]);
    let config = PluginConfig::default();

    println!("Attempting to execute heatmap plugin with 1D data (invalid)...");

    match manager.execute("heatmap", invalid_data, config.clone()) {
        Ok(_) => println!("  Unexpected success"),
        Err(e) => println!("  ✓ Validation correctly failed: {}", e),
    }

    // Try with valid 2D data
    let valid_data = VisualizationData::Array2D(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    println!("\nAttempting to execute heatmap plugin with 2D data (valid)...");

    match manager.execute("heatmap", valid_data, config) {
        Ok(_) => println!("  ✓ Execution succeeded"),
        Err(e) => println!("  Unexpected error: {}", e),
    }

    println!();

    Ok(())
}

// ============================================================================
// Custom Plugin Implementation
// ============================================================================

/// Custom statistics plugin that computes basic statistics
struct StatisticsPlugin;

impl VisualizationPlugin for StatisticsPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "statistics".to_string(),
            version: "1.0.0".to_string(),
            description: "Computes statistical measures (mean, median, std, etc.)".to_string(),
            author: "Demo".to_string(),
            supported_inputs: vec!["Array1D".to_string()],
            supported_outputs: vec![PluginOutputFormat::Text, PluginOutputFormat::Json],
            tags: vec!["statistics".to_string(), "analysis".to_string()],
        }
    }

    fn execute(&self, data: VisualizationData, _config: PluginConfig) -> Result<PluginResult> {
        let values = match data {
            VisualizationData::Array1D(v) => v,
            _ => anyhow::bail!("Statistics plugin requires 1D array data"),
        };

        // Compute statistics
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;

        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let min = sorted.first().copied().unwrap_or(0.0);
        let max = sorted.last().copied().unwrap_or(0.0);

        // Generate output
        let output = format!(
            "Statistics:\n\
             Count:   {}\n\
             Mean:    {:.4}\n\
             Median:  {:.4}\n\
             Std Dev: {:.4}\n\
             Min:     {:.4}\n\
             Max:     {:.4}",
            sorted.len(),
            mean,
            median,
            std,
            min,
            max
        );

        let mut metadata = HashMap::new();
        metadata.insert("count".to_string(), sorted.len().to_string());
        metadata.insert("mean".to_string(), format!("{:.4}", mean));
        metadata.insert("median".to_string(), format!("{:.4}", median));
        metadata.insert("std".to_string(), format!("{:.4}", std));

        Ok(PluginResult {
            success: true,
            output_path: None,
            output_data: Some(output.into_bytes()),
            metadata,
            error: None,
        })
    }

    fn validate(&self, data: &VisualizationData) -> bool {
        matches!(data, VisualizationData::Array1D(_))
    }
}
