//! Configuration Management Demo
#![allow(unused_variables)]
//!
//! This example demonstrates the configuration management features in TrustformeRS:
//! - Configuration validation
//! - Configuration migration between versions
//! - Configuration recommendations and optimization
//! - Configuration templates and presets
//! - Configuration comparison

use std::collections::HashMap;
use trustformers::{
    ConfigurationManager, RecommendationContext, PerformanceRequirements, ConfigFormat,
};

fn main() -> anyhow::Result<()> {
    println!("🔧 TrustformeRS Configuration Management Demo");
    println!("============================================\n");

    // Create a configuration manager
    let mut config_manager = ConfigurationManager::new();

    // 1. Generate configuration template
    println!("1. 📝 Generating configuration template for training:");
    if let Some(template) = config_manager.generate_template("training") {
        println!("{}\n", serde_json::to_string_pretty(&template)?);
    }

    // 2. Create configuration from preset
    println!("2. 🎨 Creating configuration from preset:");
    let config_from_preset = config_manager.create_from_preset(
        "training",
        "fast_development",
        Some(HashMap::from([
            ("batch_size".to_string(), serde_json::Value::Number(serde_json::Number::from(16)))
        ]))
    )?;
    println!("Fast development preset (with custom batch size):");
    println!("{}\n", serde_json::to_string_pretty(&config_from_preset)?);

    // 3. Validate configuration
    println!("3. ✅ Validating configuration:");

    // Valid configuration
    let valid_config = serde_json::json!({
        "num_epochs": 5,
        "batch_size": 32,
        "learning_rate": 2e-5
    });

    let validation_result = config_manager.validate_config("training", &valid_config);
    println!("Valid configuration result: {}", if validation_result.is_valid { "✅ PASSED" } else { "❌ FAILED" });

    // Invalid configuration (missing required field)
    let invalid_config = serde_json::json!({
        "num_epochs": 5,
        "batch_size": 32
        // missing learning_rate
    });

    let validation_result = config_manager.validate_config("training", &invalid_config);
    println!("Invalid configuration result: {}", if validation_result.is_valid { "✅ PASSED" } else { "❌ FAILED" });
    if !validation_result.is_valid {
        for error in &validation_result.errors {
            println!("  ❌ Error: {}", error.message);
            if let Some(suggestion) = &error.suggestion {
                println!("     💡 Suggestion: {}", suggestion);
            }
        }
    }

    // Configuration with warnings
    let config_with_warnings = serde_json::json!({
        "num_epochs": 5,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "unknown_field": "some_value"
    });

    let validation_result = config_manager.validate_config("training", &config_with_warnings);
    if !validation_result.warnings.is_empty() {
        println!("Configuration with warnings:");
        for warning in &validation_result.warnings {
            println!("  ⚠️  Warning: {}", warning.message);
            if let Some(suggestion) = &warning.suggestion {
                println!("     💡 Suggestion: {}", suggestion);
            }
        }
    }
    println!();

    // 4. Configuration migration
    println!("4. 🔄 Configuration migration:");
    let old_config = serde_json::json!({
        "num_epochs": 5,
        "batch_size": 32,
        "learning_rate": 2e-5
    });

    println!("Original config (v1.0.0):");
    println!("{}", serde_json::to_string_pretty(&old_config)?);

    let migrated_config = config_manager.migrate_config("training", &old_config, "1.0.0", "2.0.0")?;
    println!("Migrated config (v2.0.0):");
    println!("{}\n", serde_json::to_string_pretty(&migrated_config)?);

    // 5. Configuration recommendations
    println!("5. 💡 Configuration recommendations:");

    let suboptimal_config = serde_json::json!({
        "batch_size": 8,
        "learning_rate": 1e-2, // Very high learning rate
        "num_epochs": 5
    });

    let context = RecommendationContext {
        hardware_info: HashMap::from([
            ("gpu_memory_gb".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(16.0).unwrap())),
            ("gpu_count".to_string(), serde_json::Value::Number(serde_json::Number::from(2))),
        ]),
        use_case: "production".to_string(),
        performance_requirements: PerformanceRequirements {
            max_latency_ms: Some(100.0),
            min_throughput: Some(50.0),
            memory_budget_gb: Some(8.0),
            power_budget_watts: None,
        },
        constraints: vec!["must_fit_in_memory".to_string()],
    };

    let recommendations = config_manager.get_recommendations("training", &suboptimal_config, &context);

    if !recommendations.is_empty() {
        println!("Recommendations for optimizing configuration:");
        for rec in &recommendations {
            println!("  🎯 Field: {}", rec.field);
            if let Some(current) = &rec.current_value {
                println!("     Current: {}", current);
            }
            println!("     Recommended: {}", rec.recommended_value);
            println!("     Reason: {}", rec.reason);
            println!("     Impact: {:?}", rec.impact);
            println!("     Confidence: {:.1}%", rec.confidence * 100.0);
            println!();
        }
    }

    // 6. Configuration comparison
    println!("6. 🔍 Configuration comparison:");

    let config1 = serde_json::json!({
        "num_epochs": 5,
        "batch_size": 32,
        "learning_rate": 2e-5
    });

    let config2 = serde_json::json!({
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "warmup_steps": 1000
    });

    let comparison = config_manager.compare_configs(&config1, &config2);
    println!("{}", comparison);

    // 7. Available presets
    println!("7. 🎭 Available presets:");
    let training_presets = config_manager.get_presets("training");
    for preset in training_presets {
        println!("  📋 {}: {}", preset.name, preset.description);
    }

    let conversational_presets = config_manager.get_presets("conversational");
    for preset in conversational_presets {
        println!("  💬 {}: {}", preset.name, preset.description);
    }
    println!();

    // 8. Save and load configuration
    println!("8. 💾 Save and load configuration:");

    // Save configuration to JSON
    config_manager.save_config_file(&valid_config, "/tmp/training_config.json", ConfigFormat::Json)?;
    println!("✅ Saved configuration to /tmp/training_config.json");

    // Save configuration to YAML
    config_manager.save_config_file(&valid_config, "/tmp/training_config.yaml", ConfigFormat::Yaml)?;
    println!("✅ Saved configuration to /tmp/training_config.yaml");

    // Load configuration from file
    let loaded_config = config_manager.load_config_file("/tmp/training_config.json", "training")?;
    println!("✅ Loaded and validated configuration from file");
    println!("Loaded config: {}", serde_json::to_string_pretty(&loaded_config)?);

    println!("\n🎉 Configuration management demo completed successfully!");
    Ok(())
}