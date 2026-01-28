#![allow(unused_variables)]
use anyhow::Result;
use std::collections::HashMap;
use trustformers_serve::chaos_testing::{
    ChaosExperiment, ChaosExperimentType, ChaosTestingFramework, ComparisonOperator, ConditionType,
    ExperimentConfig, ExperimentScope, ExperimentStatus, PreCondition, SafetyCheck,
    SafetyCheckType, SafetyConfig, SuccessCriterion,
};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("🔬 TrustformeRS Chaos Testing Framework Demo");
    println!("============================================");

    // Create chaos testing framework
    let framework = ChaosTestingFramework::new();

    println!("✅ Initialized Chaos Testing Framework");

    // Example 1: Network Latency Experiment
    println!("\n🌐 Example 1: Network Latency Chaos Experiment");
    let network_experiment = create_network_latency_experiment();
    let network_id = framework.create_experiment(network_experiment).await?;
    println!("📋 Created network latency experiment: {}", network_id);

    // Start the experiment
    framework.start_experiment(network_id).await?;
    println!("🚀 Started network latency experiment");

    // Wait a bit for the experiment to run
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

    // Check experiment status
    if let Some(experiment) = framework.get_experiment(network_id).await {
        println!("📊 Experiment status: {:?}", experiment.status);
    }

    // Example 2: CPU Exhaustion Experiment
    println!("\n💻 Example 2: CPU Exhaustion Chaos Experiment");
    let cpu_experiment = create_cpu_exhaustion_experiment();
    let cpu_id = framework.create_experiment(cpu_experiment).await?;
    println!("📋 Created CPU exhaustion experiment: {}", cpu_id);

    framework.start_experiment(cpu_id).await?;
    println!("🚀 Started CPU exhaustion experiment");

    // Example 3: Service Kill Experiment
    println!("\n⚡ Example 3: Service Kill Chaos Experiment");
    let service_experiment = create_service_kill_experiment();
    let service_id = framework.create_experiment(service_experiment).await?;
    println!("📋 Created service kill experiment: {}", service_id);

    framework.start_experiment(service_id).await?;
    println!("🚀 Started service kill experiment");

    // Example 4: Model Load Failure Experiment
    println!("\n🤖 Example 4: Model Load Failure Chaos Experiment");
    let model_experiment = create_model_load_failure_experiment();
    let model_id = framework.create_experiment(model_experiment).await?;
    println!("📋 Created model load failure experiment: {}", model_id);

    framework.start_experiment(model_id).await?;
    println!("🚀 Started model load failure experiment");

    // Wait for experiments to complete
    println!("\n⏳ Waiting for experiments to complete...");
    tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;

    // Example 5: List All Experiments
    println!("\n📋 Example 5: All Chaos Experiments");
    let all_experiments = framework.list_experiments().await;
    println!("📊 Total experiments: {}", all_experiments.len());

    for experiment in &all_experiments {
        println!(
            "  • {} ({:?}) - Status: {:?}",
            experiment.name, experiment.experiment_type, experiment.status
        );
        println!(
            "    Duration: {}s, Intensity: {:.1}%",
            experiment.config.duration_seconds,
            experiment.config.intensity * 100.0
        );
    }

    // Example 6: Stop All Experiments
    println!("\n🛑 Example 6: Stopping All Active Experiments");
    for experiment in &all_experiments {
        if matches!(experiment.status, ExperimentStatus::Running) {
            framework.stop_experiment(experiment.id).await?;
            println!("⏹️  Stopped experiment: {}", experiment.name);
        }
    }

    // Example 7: Collect Results
    println!("\n📈 Example 7: Experiment Results Analysis");
    for experiment in &all_experiments {
        if let Some(results) = framework.get_experiment_results(experiment.id).await {
            println!("\n📊 Results for '{}' experiment:", experiment.name);
            println!("  ✅ Success: {}", results.success);
            println!(
                "  🎯 Overall Impact: {:?}",
                results.impact_analysis.overall_impact
            );
            println!("  📈 Metrics:");
            for (metric, value) in &results.metrics {
                println!("    • {}: {:.2}", metric, value);
            }

            println!("  🔍 Impact Analysis:");
            println!(
                "    • Affected Services: {:?}",
                results.impact_analysis.affected_services
            );
            println!(
                "    • Recovery Time: {}s",
                results.impact_analysis.recovery_time_seconds
            );
            println!(
                "    • Error Increase: {:.1}%",
                results.impact_analysis.error_increase_percentage
            );
            println!(
                "    • Latency Increase: {:.1}%",
                results.impact_analysis.latency_increase_percentage
            );
            println!(
                "    • Availability Reduction: {:.1}%",
                results.impact_analysis.availability_reduction_percentage
            );

            println!("  💡 Recommendations:");
            for (i, recommendation) in results.recommendations.iter().enumerate() {
                println!("    {}. {}", i + 1, recommendation);
            }

            println!("  👀 Observations:");
            for observation in &results.observations {
                println!(
                    "    • [{:?}] {:?}: {}",
                    observation.severity, observation.category, observation.description
                );
            }
        }
    }

    // Example 8: Emergency Stop Demo
    println!("\n🚨 Example 8: Emergency Stop Demonstration");

    // Create a high-impact experiment for demo
    let emergency_experiment = create_high_impact_experiment();
    let emergency_id = framework.create_experiment(emergency_experiment).await?;
    framework.start_experiment(emergency_id).await?;
    println!("⚡ Started high-impact experiment for emergency stop demo");

    // Wait a moment
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Trigger emergency stop
    framework.emergency_stop_all().await?;
    println!("🚨 Emergency stop triggered - all experiments stopped");

    // Example 9: Safety Features Demonstration
    println!("\n🛡️  Example 9: Safety Features");
    let safe_experiment = create_experiment_with_safety_features();
    let safe_id = framework.create_experiment(safe_experiment).await?;
    println!("🔒 Created experiment with comprehensive safety features");

    if let Some(safe_exp) = framework.get_experiment(safe_id).await {
        println!("🛡️  Safety Configuration:");
        println!(
            "  • Max Duration: {}s",
            safe_exp.safety_config.max_duration_seconds
        );
        println!(
            "  • Health Check Interval: {}s",
            safe_exp.safety_config.health_check_interval_seconds
        );
        println!(
            "  • Automatic Rollback: {}",
            safe_exp.safety_config.enable_automatic_rollback
        );
        println!(
            "  • Safety Checks: {}",
            safe_exp.safety_config.safety_checks.len()
        );

        for check in &safe_exp.safety_config.safety_checks {
            println!(
                "    - {} ({:?}): threshold {:.2}",
                check.name, check.check_type, check.threshold
            );
        }
    }

    // Example 10: Experiment Configuration Analysis
    println!("\n⚙️  Example 10: Configuration Analysis");
    analyze_experiment_configurations(&all_experiments).await;

    println!("\n🎉 Chaos testing demo completed successfully!");
    println!("📊 Summary:");
    println!("  • Total experiments created: {}", all_experiments.len());
    println!("  • Experiment types tested: Network, CPU, Service, Model failures");
    println!("  • Safety features demonstrated: Emergency stop, rollback, health checks");
    println!("  • Analysis completed: Results, recommendations, observations");

    Ok(())
}

fn create_network_latency_experiment() -> ChaosExperiment {
    ChaosExperiment {
        id: Uuid::new_v4(),
        name: "Network Latency Resilience Test".to_string(),
        description: "Test system resilience to increased network latency between services"
            .to_string(),
        experiment_type: ChaosExperimentType::NetworkLatency,
        config: ExperimentConfig {
            duration_seconds: 30,
            intensity: 0.3,
            scope: ExperimentScope::Percentage(20.0),
            parameters: HashMap::from([
                ("latency_ms".to_string(), serde_json::json!(150)),
                ("jitter_ms".to_string(), serde_json::json!(20)),
            ]),
            pre_conditions: vec![
                PreCondition {
                    name: "System Health Check".to_string(),
                    condition_type: ConditionType::HealthCheck,
                    value: serde_json::json!(true),
                    required: true,
                },
                PreCondition {
                    name: "Low Error Rate".to_string(),
                    condition_type: ConditionType::ErrorRate,
                    value: serde_json::json!(0.01),
                    required: true,
                },
            ],
            success_criteria: vec![
                SuccessCriterion {
                    name: "Response Time Degradation".to_string(),
                    metric: "response_time_ms".to_string(),
                    threshold: 500.0,
                    comparison: ComparisonOperator::LessThan,
                    measurement_window_seconds: 60,
                },
                SuccessCriterion {
                    name: "Error Rate Increase".to_string(),
                    metric: "error_rate".to_string(),
                    threshold: 0.05,
                    comparison: ComparisonOperator::LessThan,
                    measurement_window_seconds: 60,
                },
            ],
        },
        safety_config: SafetyConfig {
            max_duration_seconds: 120,
            rollback_timeout_seconds: 30,
            health_check_interval_seconds: 5,
            failure_threshold: 0.1,
            enable_automatic_rollback: true,
            safety_checks: vec![
                SafetyCheck {
                    name: "Error Rate Monitor".to_string(),
                    check_type: SafetyCheckType::ErrorRate,
                    threshold: 0.1,
                    enabled: true,
                },
                SafetyCheck {
                    name: "Response Time Monitor".to_string(),
                    check_type: SafetyCheckType::ResponseTime,
                    threshold: 1000.0,
                    enabled: true,
                },
            ],
            emergency_contacts: vec![
                "ops-team@example.com".to_string(),
                "sre-team@example.com".to_string(),
            ],
        },
        status: ExperimentStatus::Created,
        created_at: chrono::Utc::now(),
        started_at: None,
        ended_at: None,
        results: None,
        tags: HashMap::from([
            ("environment".to_string(), "staging".to_string()),
            ("component".to_string(), "network".to_string()),
            ("team".to_string(), "reliability".to_string()),
        ]),
    }
}

fn create_cpu_exhaustion_experiment() -> ChaosExperiment {
    ChaosExperiment {
        id: Uuid::new_v4(),
        name: "CPU Exhaustion Recovery Test".to_string(),
        description: "Test system behavior under high CPU load conditions".to_string(),
        experiment_type: ChaosExperimentType::CpuExhaustion,
        config: ExperimentConfig {
            duration_seconds: 45,
            intensity: 0.7,
            scope: ExperimentScope::SingleInstance,
            parameters: HashMap::from([
                ("cpu_load".to_string(), serde_json::json!(0.85)),
                ("cores".to_string(), serde_json::json!(2)),
            ]),
            pre_conditions: vec![PreCondition {
                name: "CPU Utilization Below Threshold".to_string(),
                condition_type: ConditionType::ResourceUtilization,
                value: serde_json::json!(0.6),
                required: true,
            }],
            success_criteria: vec![SuccessCriterion {
                name: "Service Availability".to_string(),
                metric: "availability".to_string(),
                threshold: 0.95,
                comparison: ComparisonOperator::GreaterThan,
                measurement_window_seconds: 60,
            }],
        },
        safety_config: SafetyConfig {
            max_duration_seconds: 90,
            rollback_timeout_seconds: 15,
            health_check_interval_seconds: 3,
            failure_threshold: 0.15,
            enable_automatic_rollback: true,
            safety_checks: vec![
                SafetyCheck {
                    name: "Service Health".to_string(),
                    check_type: SafetyCheckType::ServiceHealth,
                    threshold: 1.0,
                    enabled: true,
                },
                SafetyCheck {
                    name: "Resource Usage".to_string(),
                    check_type: SafetyCheckType::ResourceUsage,
                    threshold: 0.95,
                    enabled: true,
                },
            ],
            emergency_contacts: vec!["platform-team@example.com".to_string()],
        },
        status: ExperimentStatus::Created,
        created_at: chrono::Utc::now(),
        started_at: None,
        ended_at: None,
        results: None,
        tags: HashMap::from([
            ("type".to_string(), "resource".to_string()),
            ("severity".to_string(), "medium".to_string()),
        ]),
    }
}

fn create_service_kill_experiment() -> ChaosExperiment {
    ChaosExperiment {
        id: Uuid::new_v4(),
        name: "Service Recovery Test".to_string(),
        description: "Test automatic service recovery and failover mechanisms".to_string(),
        experiment_type: ChaosExperimentType::ServiceKill,
        config: ExperimentConfig {
            duration_seconds: 60,
            intensity: 1.0,
            scope: ExperimentScope::SpecificTargets(vec!["inference-service-1".to_string()]),
            parameters: HashMap::from([
                (
                    "service_name".to_string(),
                    serde_json::json!("inference-service"),
                ),
                ("kill_signal".to_string(), serde_json::json!("SIGTERM")),
                ("grace_period".to_string(), serde_json::json!(10)),
            ]),
            pre_conditions: vec![PreCondition {
                name: "Multiple Service Instances".to_string(),
                condition_type: ConditionType::ServiceAvailability,
                value: serde_json::json!(2),
                required: true,
            }],
            success_criteria: vec![
                SuccessCriterion {
                    name: "Service Recovery Time".to_string(),
                    metric: "recovery_time_seconds".to_string(),
                    threshold: 30.0,
                    comparison: ComparisonOperator::LessThan,
                    measurement_window_seconds: 120,
                },
                SuccessCriterion {
                    name: "Overall Availability".to_string(),
                    metric: "availability".to_string(),
                    threshold: 0.99,
                    comparison: ComparisonOperator::GreaterThan,
                    measurement_window_seconds: 120,
                },
            ],
        },
        safety_config: SafetyConfig {
            max_duration_seconds: 180,
            rollback_timeout_seconds: 60,
            health_check_interval_seconds: 5,
            failure_threshold: 0.05,
            enable_automatic_rollback: true,
            safety_checks: vec![
                SafetyCheck {
                    name: "Minimum Service Instances".to_string(),
                    check_type: SafetyCheckType::ServiceHealth,
                    threshold: 1.0,
                    enabled: true,
                },
                SafetyCheck {
                    name: "Availability Threshold".to_string(),
                    check_type: SafetyCheckType::Availability,
                    threshold: 0.95,
                    enabled: true,
                },
            ],
            emergency_contacts: vec![
                "devops@example.com".to_string(),
                "on-call@example.com".to_string(),
            ],
        },
        status: ExperimentStatus::Created,
        created_at: chrono::Utc::now(),
        started_at: None,
        ended_at: None,
        results: None,
        tags: HashMap::from([
            ("component".to_string(), "service".to_string()),
            ("impact".to_string(), "high".to_string()),
        ]),
    }
}

fn create_model_load_failure_experiment() -> ChaosExperiment {
    ChaosExperiment {
        id: Uuid::new_v4(),
        name: "Model Load Failure Resilience".to_string(),
        description: "Test system behavior when model loading fails".to_string(),
        experiment_type: ChaosExperimentType::ModelLoadFailure,
        config: ExperimentConfig {
            duration_seconds: 90,
            intensity: 0.5,
            scope: ExperimentScope::Percentage(25.0),
            parameters: HashMap::from([
                ("model_name".to_string(), serde_json::json!("llama-7b")),
                ("failure_type".to_string(), serde_json::json!("corruption")),
                ("retry_attempts".to_string(), serde_json::json!(3)),
            ]),
            pre_conditions: vec![PreCondition {
                name: "Model Available".to_string(),
                condition_type: ConditionType::ServiceAvailability,
                value: serde_json::json!(true),
                required: true,
            }],
            success_criteria: vec![SuccessCriterion {
                name: "Fallback Model Usage".to_string(),
                metric: "fallback_model_usage".to_string(),
                threshold: 0.8,
                comparison: ComparisonOperator::GreaterThan,
                measurement_window_seconds: 120,
            }],
        },
        safety_config: SafetyConfig {
            max_duration_seconds: 300,
            rollback_timeout_seconds: 45,
            health_check_interval_seconds: 10,
            failure_threshold: 0.2,
            enable_automatic_rollback: true,
            safety_checks: vec![SafetyCheck {
                name: "Inference Success Rate".to_string(),
                check_type: SafetyCheckType::ErrorRate,
                threshold: 0.3,
                enabled: true,
            }],
            emergency_contacts: vec!["ml-team@example.com".to_string()],
        },
        status: ExperimentStatus::Created,
        created_at: chrono::Utc::now(),
        started_at: None,
        ended_at: None,
        results: None,
        tags: HashMap::from([
            ("component".to_string(), "ml-model".to_string()),
            ("team".to_string(), "ml-platform".to_string()),
        ]),
    }
}

fn create_high_impact_experiment() -> ChaosExperiment {
    ChaosExperiment {
        id: Uuid::new_v4(),
        name: "High Impact Emergency Stop Demo".to_string(),
        description: "Demonstrate emergency stop functionality with high-impact experiment"
            .to_string(),
        experiment_type: ChaosExperimentType::ServiceCrash,
        config: ExperimentConfig {
            duration_seconds: 300, // Long duration for demo
            intensity: 0.9,
            scope: ExperimentScope::AllInstances,
            parameters: HashMap::new(),
            pre_conditions: vec![],
            success_criteria: vec![],
        },
        safety_config: SafetyConfig {
            max_duration_seconds: 600,
            rollback_timeout_seconds: 30,
            health_check_interval_seconds: 2,
            failure_threshold: 0.5,
            enable_automatic_rollback: true,
            safety_checks: vec![],
            emergency_contacts: vec!["emergency@example.com".to_string()],
        },
        status: ExperimentStatus::Created,
        created_at: chrono::Utc::now(),
        started_at: None,
        ended_at: None,
        results: None,
        tags: HashMap::from([
            ("demo".to_string(), "emergency-stop".to_string()),
            ("severity".to_string(), "critical".to_string()),
        ]),
    }
}

fn create_experiment_with_safety_features() -> ChaosExperiment {
    ChaosExperiment {
        id: Uuid::new_v4(),
        name: "Comprehensive Safety Features Demo".to_string(),
        description: "Demonstrate all safety features and monitoring capabilities".to_string(),
        experiment_type: ChaosExperimentType::NetworkPartition,
        config: ExperimentConfig {
            duration_seconds: 120,
            intensity: 0.6,
            scope: ExperimentScope::MultipleInstances(3),
            parameters: HashMap::from([
                ("partition_duration".to_string(), serde_json::json!(30)),
                ("affected_percentage".to_string(), serde_json::json!(0.3)),
            ]),
            pre_conditions: vec![
                PreCondition {
                    name: "Cluster Health".to_string(),
                    condition_type: ConditionType::HealthCheck,
                    value: serde_json::json!(true),
                    required: true,
                },
                PreCondition {
                    name: "Network Baseline".to_string(),
                    condition_type: ConditionType::MetricThreshold,
                    value: serde_json::json!({"latency_ms": 100}),
                    required: true,
                },
            ],
            success_criteria: vec![
                SuccessCriterion {
                    name: "Partition Recovery".to_string(),
                    metric: "partition_recovery_time".to_string(),
                    threshold: 60.0,
                    comparison: ComparisonOperator::LessThan,
                    measurement_window_seconds: 180,
                },
                SuccessCriterion {
                    name: "Data Consistency".to_string(),
                    metric: "consistency_score".to_string(),
                    threshold: 0.95,
                    comparison: ComparisonOperator::GreaterThan,
                    measurement_window_seconds: 240,
                },
            ],
        },
        safety_config: SafetyConfig {
            max_duration_seconds: 180,
            rollback_timeout_seconds: 45,
            health_check_interval_seconds: 3,
            failure_threshold: 0.08,
            enable_automatic_rollback: true,
            safety_checks: vec![
                SafetyCheck {
                    name: "Error Rate Guard".to_string(),
                    check_type: SafetyCheckType::ErrorRate,
                    threshold: 0.1,
                    enabled: true,
                },
                SafetyCheck {
                    name: "Response Time Guard".to_string(),
                    check_type: SafetyCheckType::ResponseTime,
                    threshold: 2000.0,
                    enabled: true,
                },
                SafetyCheck {
                    name: "Availability Guard".to_string(),
                    check_type: SafetyCheckType::Availability,
                    threshold: 0.9,
                    enabled: true,
                },
                SafetyCheck {
                    name: "Resource Guard".to_string(),
                    check_type: SafetyCheckType::ResourceUsage,
                    threshold: 0.9,
                    enabled: true,
                },
                SafetyCheck {
                    name: "Service Health Guard".to_string(),
                    check_type: SafetyCheckType::ServiceHealth,
                    threshold: 1.0,
                    enabled: true,
                },
            ],
            emergency_contacts: vec![
                "primary-oncall@example.com".to_string(),
                "secondary-oncall@example.com".to_string(),
                "engineering-lead@example.com".to_string(),
            ],
        },
        status: ExperimentStatus::Created,
        created_at: chrono::Utc::now(),
        started_at: None,
        ended_at: None,
        results: None,
        tags: HashMap::from([
            ("feature".to_string(), "safety-demo".to_string()),
            ("complexity".to_string(), "comprehensive".to_string()),
            ("monitoring".to_string(), "full".to_string()),
        ]),
    }
}

async fn analyze_experiment_configurations(experiments: &[ChaosExperiment]) {
    println!("🔍 Configuration Analysis:");

    // Analyze experiment types
    let mut type_counts = HashMap::new();
    for exp in experiments {
        *type_counts.entry(format!("{:?}", exp.experiment_type)).or_insert(0) += 1;
    }

    println!("📊 Experiment Types:");
    for (exp_type, count) in type_counts {
        println!("  • {}: {} experiments", exp_type, count);
    }

    // Analyze durations
    let durations: Vec<u64> = experiments.iter().map(|e| e.config.duration_seconds).collect();
    if !durations.is_empty() {
        let avg_duration = durations.iter().sum::<u64>() as f64 / durations.len() as f64;
        let min_duration = *durations.iter().min().unwrap();
        let max_duration = *durations.iter().max().unwrap();

        println!("⏱️  Duration Analysis:");
        println!("  • Average: {:.1}s", avg_duration);
        println!("  • Range: {}s - {}s", min_duration, max_duration);
    }

    // Analyze safety features
    let total_safety_checks: usize =
        experiments.iter().map(|e| e.safety_config.safety_checks.len()).sum();
    let auto_rollback_count =
        experiments.iter().filter(|e| e.safety_config.enable_automatic_rollback).count();

    println!("🛡️  Safety Analysis:");
    println!("  • Total Safety Checks: {}", total_safety_checks);
    println!(
        "  • Auto-Rollback Enabled: {}/{}",
        auto_rollback_count,
        experiments.len()
    );

    // Analyze scopes
    let mut scope_types = HashMap::new();
    for exp in experiments {
        let scope_type = match &exp.config.scope {
            ExperimentScope::SingleInstance => "Single Instance",
            ExperimentScope::MultipleInstances(_) => "Multiple Instances",
            ExperimentScope::Percentage(_) => "Percentage",
            ExperimentScope::AllInstances => "All Instances",
            ExperimentScope::SpecificTargets(_) => "Specific Targets",
        };
        *scope_types.entry(scope_type).or_insert(0) += 1;
    }

    println!("🎯 Scope Analysis:");
    for (scope_type, count) in scope_types {
        println!("  • {}: {} experiments", scope_type, count);
    }

    // Analyze intensities
    let intensities: Vec<f64> = experiments.iter().map(|e| e.config.intensity).collect();
    if !intensities.is_empty() {
        let avg_intensity = intensities.iter().sum::<f64>() / intensities.len() as f64;
        let min_intensity = intensities.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_intensity = intensities.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        println!("🔥 Intensity Analysis:");
        println!("  • Average: {:.1}%", avg_intensity * 100.0);
        println!(
            "  • Range: {:.1}% - {:.1}%",
            min_intensity * 100.0,
            max_intensity * 100.0
        );
    }
}
