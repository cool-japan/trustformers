use anyhow::Result;
#![allow(unused_variables)]
use std::sync::Arc;
use trustformers_core::{
    // Versioning types
    ModelVersionManager, ModelMetadata, VersionedModel, ModelTag, ModelSource,
    Artifact, ArtifactType, FileSystemStorage, InMemoryStorage,
    VersionStatus, VersionTransition,
    DeploymentManager, DeploymentConfig, Environment, DeploymentStrategy,
    // A/B testing integration
    VersionedABTestManager, VersionExperimentConfig, VersionMetricType,
};
use uuid::Uuid;
use std::collections::HashMap;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    println!("TrustformeRS Model Versioning and A/B Testing Demo");
    println!("=================================================\n");

    // Initialize storage and version manager
    let storage = Arc::new(InMemoryStorage::new()); // Use filesystem storage in production
    let version_manager = Arc::new(ModelVersionManager::new(storage));

    // Demo 1: Model Version Registration
    println!("📦 Demo 1: Model Version Registration");
    println!("-------------------------------------");

    let gpt2_v1_id = register_model_version(
        &version_manager,
        "gpt2",
        "1.0.0",
        "Initial GPT-2 model release",
        "training_team",
        vec![
            ModelTag::with_value("model_type", "transformer"),
            ModelTag::with_value("size", "125M"),
            ModelTag::new("production_ready"),
        ],
    ).await?;

    let gpt2_v11_id = register_model_version(
        &version_manager,
        "gpt2",
        "1.1.0",
        "Improved GPT-2 with better attention",
        "training_team",
        vec![
            ModelTag::with_value("model_type", "transformer"),
            ModelTag::with_value("size", "125M"),
            ModelTag::with_value("improvement", "attention_optimization"),
        ],
    ).await?;

    let gpt2_v20_id = register_model_version(
        &version_manager,
        "gpt2",
        "2.0.0",
        "Major GPT-2 upgrade with new architecture",
        "research_team",
        vec![
            ModelTag::with_value("model_type", "transformer"),
            ModelTag::with_value("size", "350M"),
            ModelTag::with_value("architecture", "improved_layers"),
            ModelTag::new("experimental"),
        ],
    ).await?;

    println!("✅ Registered 3 model versions\n");

    // Demo 2: Version Lifecycle Management
    println!("🔄 Demo 2: Version Lifecycle Management");
    println!("---------------------------------------");

    // Promote v1.0.0 to production
    version_manager.lifecycle().transition(gpt2_v1_id, VersionTransition::ToStaging).await?;
    version_manager.promote_to_production(gpt2_v1_id).await?;
    println!("✅ Promoted gpt2:1.0.0 to production");

    // Move v1.1.0 to staging
    version_manager.lifecycle().transition(gpt2_v11_id, VersionTransition::ToStaging).await?;
    println!("✅ Moved gpt2:1.1.0 to staging");

    // Keep v2.0.0 in development for now
    println!("✅ Kept gpt2:2.0.0 in development");

    // Show version statistics
    let stats = version_manager.get_version_stats("gpt2").await?;
    println!("\n📊 Version Statistics:");
    println!("  Total versions: {}", stats.total_versions);
    println!("  Production: {}", stats.production_versions);
    println!("  Staging: {}", stats.staging_versions);
    println!("  Development: {}", stats.development_versions);
    println!("  Latest: {}", stats.latest_version.unwrap_or("None".to_string()));
    println!();

    // Demo 3: Model Deployment
    println!("🚀 Demo 3: Model Deployment Strategies");
    println!("--------------------------------------");

    // Deploy v1.1.0 using canary strategy
    let v11_model = version_manager.get_version(gpt2_v11_id).await?.unwrap();
    let deployment_config = DeploymentConfig {
        environment: Environment::Production,
        strategy: DeploymentStrategy::Canary,
        initial_traffic_percentage: Some(10.0),
        health_check_url: Some("http://localhost:8080/health".to_string()),
        config_overrides: HashMap::new(),
        min_sample_size: None,
        max_duration_hours: None,
    };

    let deployment_id = version_manager.deployment_manager()
        .deploy_with_strategy(gpt2_v11_id, &v11_model, deployment_config).await?;
    println!("✅ Started canary deployment: {}", deployment_id);

    // Gradually increase traffic
    for percentage in [25.0, 50.0, 75.0, 100.0] {
        version_manager.deployment_manager()
            .update_traffic_percentage(&deployment_id, percentage).await?;
        println!("  📈 Updated traffic to {:.0}%", percentage);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    // Check deployment health
    let health = version_manager.deployment_manager()
        .health_check(&deployment_id).await?;
    println!("  🏥 Health check: {} ({}ms response time)",
             if health.is_healthy { "✅ Healthy" } else { "❌ Unhealthy" },
             health.response_time_ms);
    println!();

    // Demo 4: A/B Testing Integration
    println!("🧪 Demo 4: A/B Testing Integration");
    println!("---------------------------------");

    let ab_manager = VersionedABTestManager::new(version_manager.clone());

    // Create A/B test between v1.0.0 (control) and v1.1.0 (treatment)
    let experiment_config = VersionExperimentConfig {
        name: "GPT-2 v1.1 Performance Test".to_string(),
        description: "Testing improved attention mechanism".to_string(),
        control_version_id: gpt2_v1_id,
        treatment_version_ids: vec![gpt2_v11_id],
        traffic_percentage: 50.0,
        min_sample_size: 100,
        max_duration_hours: 24,
    };

    let experiment_id = ab_manager.create_version_experiment(experiment_config).await?;
    println!("✅ Created A/B test experiment: {}", experiment_id);

    // Simulate user requests and collect metrics
    println!("  📊 Simulating user requests...");
    for i in 0..50 {
        let user_id = format!("user_{}", i);

        // Route request to appropriate version
        let routing_result = ab_manager.route_request(&experiment_id, &user_id).await?;

        // Simulate different performance characteristics
        let (latency, accuracy) = match routing_result.variant.name() {
            "control" => (120.0 + rand::random::<f64>() * 20.0, 0.85 + rand::random::<f64>() * 0.10),
            "treatment_0" => (100.0 + rand::random::<f64>() * 15.0, 0.88 + rand::random::<f64>() * 0.08), // Improved
            _ => (120.0, 0.85),
        };

        // Record metrics
        ab_manager.record_version_metric(
            &experiment_id,
            &user_id,
            VersionMetricType::Latency,
            latency,
            None,
        ).await?;

        ab_manager.record_version_metric(
            &experiment_id,
            &user_id,
            VersionMetricType::Accuracy,
            accuracy,
            None,
        ).await?;
    }

    // Analyze experiment results
    let experiment_result = ab_manager.analyze_version_experiment(&experiment_id).await?;
    println!("\n  📈 Experiment Results:");
    println!("    Control version: {}:{}",
             experiment_result.control_version.model_name(),
             experiment_result.control_version.version());
    println!("    Treatment versions: {}",
             experiment_result.treatment_versions.iter()
                 .map(|v| format!("{}:{}", v.model_name(), v.version()))
                 .collect::<Vec<_>>()
                 .join(", "));
    println!("    Total requests: {}", experiment_result.total_requests);
    println!("    Duration: {} minutes", experiment_result.experiment_duration.num_minutes());

    // Show performance comparison
    for (metric_key, metrics) in &experiment_result.version_performance_comparison {
        println!("    {}: mean={:.2}, p95={:.2}", metric_key, metrics.mean, metrics.p95);
    }

    // Try to promote winning version
    let promotion_result = ab_manager.promote_winning_version(&experiment_id).await?;
    if promotion_result.promoted {
        println!("  🎉 Promoted winning version: {:?} ({})",
                 promotion_result.version_id, promotion_result.reason);
    } else {
        println!("  ⏳ No promotion: {}", promotion_result.reason);
    }
    println!();

    // Demo 5: Advanced Querying
    println!("🔍 Demo 5: Advanced Querying");
    println!("----------------------------");

    // Query versions by tag
    let transformer_versions = version_manager.registry()
        .get_versions_by_tag("model_type").await?;
    println!("✅ Found {} transformer models", transformer_versions.len());

    // Query latest version
    let latest = version_manager.get_latest_version("gpt2").await?;
    if let Some(latest_version) = latest {
        println!("✅ Latest version: {}:{}",
                 latest_version.model_name(), latest_version.version());
    }

    // Get registry statistics
    let registry_stats = version_manager.registry().get_statistics().await?;
    println!("✅ Registry contains {} versions across {} models",
             registry_stats.total_versions, registry_stats.total_models);
    println!();

    // Demo 6: Rollback Scenario
    println!("⏪ Demo 6: Rollback Scenario");
    println!("---------------------------");

    // Simulate a problem with the promoted version and rollback
    println!("  ⚠️  Detected issue with current production version");
    println!("  🔄 Rolling back to previous stable version...");

    version_manager.rollback_to_version("gpt2", "1.0.0").await?;
    println!("  ✅ Successfully rolled back to gpt2:1.0.0");

    // Check current production version
    let current_deployment = version_manager.deployment_manager()
        .get_active_deployment("gpt2").await?;

    if let Some(deployment) = current_deployment {
        let current_version = version_manager.get_version(deployment.version_id).await?.unwrap();
        println!("  📦 Current production version: {}:{}",
                 current_version.model_name(), current_version.version());
    }
    println!();

    // Demo 7: Cleanup and Archive
    println!("🗄️  Demo 7: Cleanup and Archive");
    println!("-----------------------------");

    // Archive old development version
    version_manager.archive_version(gpt2_v20_id).await?;
    println!("✅ Archived experimental version gpt2:2.0.0");

    // Get final statistics
    let final_stats = version_manager.get_version_stats("gpt2").await?;
    println!("📊 Final Statistics:");
    println!("  Total versions: {}", final_stats.total_versions);
    println!("  Production: {}", final_stats.production_versions);
    println!("  Staging: {}", final_stats.staging_versions);
    println!("  Development: {}", final_stats.development_versions);
    println!("  Archived: {}", final_stats.archived_versions);

    println!("\n🎯 Demo completed successfully!");
    println!("   This demo showcased:");
    println!("   ✅ Model version registration and metadata management");
    println!("   ✅ Version lifecycle management (dev → staging → production)");
    println!("   ✅ Advanced deployment strategies (canary, blue-green)");
    println!("   ✅ A/B testing integration with automated promotion");
    println!("   ✅ Advanced querying and version discovery");
    println!("   ✅ Production rollback capabilities");
    println!("   ✅ Version archival and cleanup");

    Ok(())
}

/// Helper function to register a model version
async fn register_model_version(
    manager: &ModelVersionManager,
    model_name: &str,
    version: &str,
    description: &str,
    created_by: &str,
    tags: Vec<ModelTag>,
) -> Result<Uuid> {
    // Create metadata
    let mut metadata = ModelMetadata::builder()
        .description(description.to_string())
        .created_by(created_by.to_string())
        .model_type("transformer".to_string())
        .architecture("gpt2".to_string())
        .metric("perplexity".to_string(), 15.2)
        .metric("bleu_score".to_string(), 0.78)
        .source(ModelSource {
            source_type: "training".to_string(),
            dataset: Some("openwebtext".to_string()),
            training_run_id: Some(Uuid::new_v4().to_string()),
            base_model: None,
            config_ref: None,
            metadata: std::collections::HashMap::new(),
        })
        .build();

    // Add tags
    for tag in tags {
        metadata.add_tag(tag);
    }

    // Create mock artifacts
    let artifacts = vec![
        Artifact::new(
            ArtifactType::Model,
            PathBuf::from("model.bin"),
            b"mock model weights".to_vec(),
        ),
        Artifact::new(
            ArtifactType::Config,
            PathBuf::from("config.json"),
            b"{}".to_vec(),
        ),
        Artifact::new(
            ArtifactType::Tokenizer,
            PathBuf::from("tokenizer.json"),
            b"{}".to_vec(),
        ),
    ];

    // Register the version
    let version_id = manager.register_version(model_name, version, metadata, artifacts).await?;
    println!("  📦 Registered {}:{} ({})", model_name, version, version_id);

    Ok(version_id)
}