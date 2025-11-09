use anyhow::Result;
#![allow(unused_variables)]
use trustformers_core::{
    ABTestManager, ConfidenceLevel, ExperimentConfig, MetricType, MetricValue, RoutingStrategy,
    StatisticalAnalyzer, TrafficSplitter, Variant,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("TrustformeRS A/B Testing Framework Demo");
    println!("======================================\n");

    // Create A/B test manager
    let ab_manager = ABTestManager::new();

    // Define experiment configuration
    let config = ExperimentConfig {
        name: "GPT-2 vs GPT-2-Optimized".to_string(),
        description: "Testing optimized GPT-2 model for latency improvements".to_string(),
        control_variant: Variant::new("gpt2-base", "gpt2-base-model"),
        treatment_variants: vec![Variant::new("gpt2-optimized", "gpt2-optimized-model")],
        traffic_percentage: 50.0,
        min_sample_size: 100,
        max_duration_hours: 24,
    };

    // Create experiment
    let experiment_id = ab_manager.create_experiment(config)?;
    println!("Created experiment: {}", experiment_id);

    // Simulate user requests and collect metrics
    println!("\nSimulating user requests...");
    simulate_experiment(&ab_manager, &experiment_id, 200)?;

    // Analyze results
    println!("\nAnalyzing experiment results...");
    let test_result = ab_manager.analyze_experiment(&experiment_id)?;

    // Display results
    println!("\n=== Experiment Results ===");
    println!(
        "Control ({}): mean={:.2}ms, std={:.2}ms, n={}",
        test_result.control_stats.variant.name(),
        test_result.control_stats.mean,
        test_result.control_stats.std_dev,
        test_result.control_stats.sample_size
    );

    for treatment in &test_result.treatment_stats {
        println!(
            "Treatment ({}): mean={:.2}ms, std={:.2}ms, n={}",
            treatment.variant.name(),
            treatment.mean,
            treatment.std_dev,
            treatment.sample_size
        );
    }

    println!("\n=== Statistical Analysis ===");
    println!("P-value: {:.4}", test_result.test_stats.p_value);
    println!("Effect size: {:.4}", test_result.test_stats.effect_size);
    println!("Statistical power: {:.2}%", test_result.test_stats.power * 100.0);

    println!("\n=== Recommendation ===");
    match &test_result.recommendation {
        trustformers_core::TestRecommendation::AdoptTreatment { variant, improvement } => {
            println!("✅ Adopt treatment '{}' (improvement: {:.1}%)", variant, improvement);
        }
        trustformers_core::TestRecommendation::KeepControl { degradation } => {
            println!("❌ Keep control (degradation: {:.1}%)", degradation);
        }
        trustformers_core::TestRecommendation::NoSignificantDifference => {
            println!("➖ No significant difference between variants");
        }
        trustformers_core::TestRecommendation::InsufficientData { required_sample_size } => {
            println!("⏳ Need more data (required: {} samples)", required_sample_size);
        }
    }

    // Demonstrate different routing strategies
    println!("\n\n=== Routing Strategy Demo ===");
    demonstrate_routing_strategies()?;

    Ok(())
}

/// Simulate an experiment with synthetic data
fn simulate_experiment(manager: &ABTestManager, experiment_id: &str, num_requests: usize) -> Result<()> {
    use rand::distributions::{Distribution, Normal};
    use rand::thread_rng;

    // Define performance characteristics for each variant
    let control_latency = Normal::new(100.0, 15.0).unwrap(); // mean=100ms, std=15ms
    let treatment_latency = Normal::new(85.0, 12.0).unwrap(); // mean=85ms, std=12ms (15% improvement)

    let mut rng = thread_rng();

    for i in 0..num_requests {
        let user_id = format!("user-{}", i);

        // Route request to variant
        let variant = manager.route_request(experiment_id, &user_id)?;

        // Simulate latency based on variant
        let latency = match variant.name() {
            "gpt2-base" => control_latency.sample(&mut rng),
            "gpt2-optimized" => treatment_latency.sample(&mut rng),
            _ => 100.0,
        };

        // Record metric
        manager.record_metric(
            experiment_id,
            &variant,
            MetricType::Latency,
            MetricValue::Duration(latency as u64),
        )?;

        // Also record some accuracy metrics (simulated)
        let accuracy = if rand::random::<f64>() > 0.05 { 0.95 } else { 0.90 };
        manager.record_metric(
            experiment_id,
            &variant,
            MetricType::Accuracy,
            MetricValue::Numeric(accuracy),
        )?;
    }

    Ok(())
}

/// Demonstrate different routing strategies
fn demonstrate_routing_strategies() -> Result<()> {
    println!("1. Hash-based routing (consistent assignment):");
    let hash_splitter = TrafficSplitter::new();
    demonstrate_routing(&hash_splitter, "hash")?;

    println!("\n2. Round-robin routing (alternating assignment):");
    let rr_splitter = TrafficSplitter::with_strategy(RoutingStrategy::RoundRobin);
    demonstrate_routing(&rr_splitter, "round-robin")?;

    println!("\n3. Sticky session routing (persistent assignment):");
    let sticky_splitter = TrafficSplitter::with_strategy(RoutingStrategy::Sticky);
    demonstrate_routing(&sticky_splitter, "sticky")?;

    Ok(())
}

fn demonstrate_routing(splitter: &TrafficSplitter, strategy_name: &str) -> Result<()> {
    // Create a simple experiment
    let config = ExperimentConfig {
        name: format!("{} Demo", strategy_name),
        description: "Demonstrating routing strategy".to_string(),
        control_variant: Variant::new("A", "variant-a"),
        treatment_variants: vec![Variant::new("B", "variant-b")],
        traffic_percentage: 100.0,
        min_sample_size: 10,
        max_duration_hours: 1,
    };

    let mut experiment = trustformers_core::Experiment::new(config)?;
    experiment.start()?;

    // Route 10 requests and show assignments
    for i in 0..10 {
        let user_id = format!("demo-user-{}", i);
        let variant = splitter.route(&experiment, &user_id)?;
        println!("  {} -> Variant {}", user_id, variant.name());
    }

    Ok(())
}