use std::collections::HashMap;
#![allow(unused_variables)]
use std::time::Duration;
use trustformers_training::hyperopt::{
    AutomatedHyperparameterTuner, BayesianOptimizationTuner, HyperparameterSpace,
    OptimizationDirection as AutoTunerOptimizationDirection, ParameterScale, ParameterSpec,
    RandomSearchTuner, ResourceAllocation as AutoTunerResourceAllocation, ResourceSharingStrategy,
    SearchAlgorithm, TuningConfig, AcquisitionFunction as AutoTunerAcquisitionFunction,
    HyperparameterConfig, ParameterValue,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 TrustformeRS Automated Hyperparameter Tuning Demo");
    println!("=" .repeat(50));

    // Define hyperparameter search space
    let search_space = create_transformer_search_space();
    println!("📊 Created search space with {} parameters", search_space.parameters.len());

    // Demo 1: Random Search
    println!("\n🎲 Demo 1: Random Search Optimization");
    let random_result = run_random_search_demo(&search_space).await?;
    print_optimization_summary("Random Search", &random_result);

    // Demo 2: Bayesian Optimization
    println!("\n🧠 Demo 2: Bayesian Optimization");
    let bayesian_result = run_bayesian_optimization_demo(&search_space).await?;
    print_optimization_summary("Bayesian Optimization", &bayesian_result);

    // Demo 3: Compare Results
    println!("\n📈 Comparison Summary");
    compare_results(&random_result, &bayesian_result);

    println!("\n✅ Demo completed successfully!");
    Ok(())
}

fn create_transformer_search_space() -> HyperparameterSpace {
    let mut parameters = HashMap::new();

    // Learning rate with logarithmic scale
    parameters.insert(
        "learning_rate".to_string(),
        ParameterSpec::Float {
            min: 1e-6,
            max: 1e-2,
            scale: ParameterScale::Logarithmic,
        },
    );

    // Batch size (power of 2)
    parameters.insert(
        "batch_size".to_string(),
        ParameterSpec::Int {
            min: 8,
            max: 128,
        },
    );

    // Number of attention heads
    parameters.insert(
        "num_attention_heads".to_string(),
        ParameterSpec::Int {
            min: 4,
            max: 16,
        },
    );

    // Hidden size
    parameters.insert(
        "hidden_size".to_string(),
        ParameterSpec::Int {
            min: 256,
            max: 1024,
        },
    );

    // Dropout rate
    parameters.insert(
        "dropout".to_string(),
        ParameterSpec::Float {
            min: 0.0,
            max: 0.5,
            scale: ParameterScale::Linear,
        },
    );

    // Optimizer choice
    parameters.insert(
        "optimizer".to_string(),
        ParameterSpec::Categorical {
            choices: vec![
                "adam".to_string(),
                "adamw".to_string(),
                "sgd".to_string(),
                "rmsprop".to_string(),
            ],
        },
    );

    // Weight decay
    parameters.insert(
        "weight_decay".to_string(),
        ParameterSpec::Float {
            min: 0.0,
            max: 0.1,
            scale: ParameterScale::Linear,
        },
    );

    // Warmup steps ratio
    parameters.insert(
        "warmup_ratio".to_string(),
        ParameterSpec::Float {
            min: 0.0,
            max: 0.2,
            scale: ParameterScale::Linear,
        },
    );

    // Use gradient clipping
    parameters.insert(
        "use_gradient_clipping".to_string(),
        ParameterSpec::Boolean,
    );

    HyperparameterSpace {
        parameters,
        constraints: Vec::new(),
    }
}

// Mock objective function that simulates model training
fn mock_objective_function(config: &HyperparameterConfig) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
    // Extract parameters
    let learning_rate = config.values.get("learning_rate").unwrap().as_f64().unwrap();
    let batch_size = config.values.get("batch_size").unwrap().as_i64().unwrap();
    let dropout = config.values.get("dropout").unwrap().as_f64().unwrap();
    let hidden_size = config.values.get("hidden_size").unwrap().as_i64().unwrap();
    let num_heads = config.values.get("num_attention_heads").unwrap().as_i64().unwrap();
    let weight_decay = config.values.get("weight_decay").unwrap().as_f64().unwrap();
    let warmup_ratio = config.values.get("warmup_ratio").unwrap().as_f64().unwrap();
    let optimizer = config.values.get("optimizer").unwrap().as_string().unwrap();
    let use_clipping = config.values.get("use_gradient_clipping").unwrap().as_bool().unwrap();

    // Simulate training time (in practice this would be actual model training)
    std::thread::sleep(Duration::from_millis(100 + (batch_size as u64 * 2)));

    // Mock performance based on parameter relationships
    // This simulates realistic relationships between hyperparameters and performance

    // Optimal learning rate is around 5e-4
    let lr_score = 1.0 - ((learning_rate.ln() + 7.6).abs() / 2.0).min(1.0);

    // Larger batch sizes are generally better up to a point
    let batch_score = (batch_size as f64 / 128.0).min(1.0);

    // Moderate dropout is best
    let dropout_score = 1.0 - (dropout - 0.1).abs() * 2.0;

    // Hidden size should be divisible by num_heads
    let head_compatibility = if hidden_size % num_heads == 0 { 1.0 } else { 0.8 };

    // Some optimizers perform better than others
    let optimizer_score = match optimizer.as_str() {
        "adamw" => 1.0,
        "adam" => 0.95,
        "rmsprop" => 0.85,
        "sgd" => 0.7,
        _ => 0.5,
    };

    // Gradient clipping usually helps
    let clipping_score = if use_clipping { 1.0 } else { 0.9 };

    // Weight decay should be moderate
    let wd_score = 1.0 - (weight_decay - 0.01).abs() * 10.0;

    // Warmup is generally helpful
    let warmup_score = (warmup_ratio * 10.0).min(1.0);

    // Combine scores with some noise
    let base_accuracy = lr_score * 0.25 +
                       batch_score * 0.15 +
                       dropout_score * 0.15 +
                       head_compatibility * 0.1 +
                       optimizer_score * 0.15 +
                       clipping_score * 0.05 +
                       wd_score * 0.1 +
                       warmup_score * 0.05;

    // Add some noise to make it realistic
    let noise = (rand::random::<f64>() - 0.5) * 0.1;
    let accuracy = (base_accuracy + noise).max(0.0).min(1.0);

    // Create additional realistic metrics
    let mut metrics = HashMap::new();
    metrics.insert("accuracy".to_string(), accuracy);
    metrics.insert("loss".to_string(), 1.0 - accuracy + rand::random::<f64>() * 0.1);
    metrics.insert("f1_score".to_string(), accuracy * 0.98 + rand::random::<f64>() * 0.02);
    metrics.insert("training_time".to_string(), batch_size as f64 * 0.1 + hidden_size as f64 * 0.001);

    println!("    Config: lr={:.2e}, batch={}, dropout={:.3}, heads={}, hidden={}, opt={}, acc={:.4}",
             learning_rate, batch_size, dropout, num_heads, hidden_size, optimizer, accuracy);

    Ok(metrics)
}

async fn run_random_search_demo(space: &HyperparameterSpace) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
    println!("  Configuring random search tuner...");

    let tuner = Box::new(RandomSearchTuner::new());
    let config = TuningConfig {
        max_trials: 20,
        max_duration: Some(Duration::from_secs(60)),
        early_stopping_patience: Some(5),
        early_stopping_threshold: Some(0.001),
        primary_metric: "accuracy".to_string(),
        optimization_direction: AutoTunerOptimizationDirection::Maximize,
        search_algorithm: SearchAlgorithm::Random,
        parallel_trials: 3,
        resource_allocation: AutoTunerResourceAllocation {
            max_gpu_memory_per_trial: Some(4 * 1024 * 1024 * 1024), // 4GB
            max_cpu_cores_per_trial: Some(4),
            max_training_time_per_trial: Some(Duration::from_secs(300)),
            resource_sharing_strategy: ResourceSharingStrategy::Shared,
        },
    };

    let mut automated_tuner = AutomatedHyperparameterTuner::new(tuner, config);

    println!("  Starting optimization...");
    let best_result = automated_tuner.optimize(space, mock_objective_function).await?;

    let history = automated_tuner.get_optimization_history();

    Ok(OptimizationResult {
        best_result,
        history,
    })
}

async fn run_bayesian_optimization_demo(space: &HyperparameterSpace) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
    println!("  Configuring Bayesian optimization tuner...");

    let tuner = Box::new(BayesianOptimizationTuner::new(
        AutoTunerAcquisitionFunction::ExpectedImprovement { xi: 0.01 }
    ));
    let config = TuningConfig {
        max_trials: 20,
        max_duration: Some(Duration::from_secs(60)),
        early_stopping_patience: Some(8), // Longer patience for BO
        early_stopping_threshold: Some(0.001),
        primary_metric: "accuracy".to_string(),
        optimization_direction: AutoTunerOptimizationDirection::Maximize,
        search_algorithm: SearchAlgorithm::BayesianOptimization,
        parallel_trials: 2, // Fewer parallel trials for BO
        resource_allocation: AutoTunerResourceAllocation {
            max_gpu_memory_per_trial: Some(4 * 1024 * 1024 * 1024), // 4GB
            max_cpu_cores_per_trial: Some(4),
            max_training_time_per_trial: Some(Duration::from_secs(300)),
            resource_sharing_strategy: ResourceSharingStrategy::Shared,
        },
    };

    let mut automated_tuner = AutomatedHyperparameterTuner::new(tuner, config);

    println!("  Starting optimization...");
    let best_result = automated_tuner.optimize(space, mock_objective_function).await?;

    let history = automated_tuner.get_optimization_history();

    Ok(OptimizationResult {
        best_result,
        history,
    })
}

// Custom result type for this demo
struct OptimizationResult {
    best_result: trustformers_training::hyperopt::TuningResult,
    history: Vec<trustformers_training::hyperopt::TuningResult>,
}

fn print_optimization_summary(method: &str, result: &OptimizationResult) {
    println!("  {} Results:", method);
    println!("    Trials completed: {}", result.history.len());
    println!("    Best accuracy: {:.4}", result.best_result.primary_metric);

    // Print best configuration
    println!("    Best configuration:");
    for (param, value) in &result.best_result.config.values {
        match value {
            ParameterValue::Float(v) => println!("      {}: {:.6}", param, v),
            ParameterValue::Int(v) => println!("      {}: {}", param, v),
            ParameterValue::String(v) => println!("      {}: {}", param, v),
            ParameterValue::Bool(v) => println!("      {}: {}", param, v),
        }
    }

    // Print convergence info
    let accuracies: Vec<f64> = result.history.iter().map(|r| r.primary_metric).collect();
    let best_trial_idx = accuracies.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    println!("    Best result found at trial: {}", best_trial_idx + 1);

    let total_time: Duration = result.history.iter().map(|r| r.training_time).sum();
    println!("    Total optimization time: {:?}", total_time);

    // Calculate improvement over trials
    if result.history.len() > 5 {
        let first_5_avg: f64 = accuracies.iter().take(5).sum::<f64>() / 5.0;
        let improvement = result.best_result.primary_metric - first_5_avg;
        println!("    Improvement over baseline: +{:.4}", improvement);
    }
}

fn compare_results(random_result: &OptimizationResult, bayesian_result: &OptimizationResult) {
    println!("  Method Comparison:");

    let random_best = random_result.best_result.primary_metric;
    let bayesian_best = bayesian_result.best_result.primary_metric;

    println!("    Random Search Best:      {:.4}", random_best);
    println!("    Bayesian Opt Best:       {:.4}", bayesian_best);

    let winner = if bayesian_best > random_best {
        "Bayesian Optimization"
    } else {
        "Random Search"
    };
    let difference = (bayesian_best - random_best).abs();

    println!("    Winner: {} (+{:.4})", winner, difference);

    // Compare efficiency (trials to reach 90% of best)
    let target_random = random_best * 0.9;
    let target_bayesian = bayesian_best * 0.9;

    let random_trials_to_target = random_result.history.iter()
        .position(|r| r.primary_metric >= target_random)
        .unwrap_or(random_result.history.len()) + 1;

    let bayesian_trials_to_target = bayesian_result.history.iter()
        .position(|r| r.primary_metric >= target_bayesian)
        .unwrap_or(bayesian_result.history.len()) + 1;

    println!("    Trials to 90% of best:");
    println!("      Random Search:    {}", random_trials_to_target);
    println!("      Bayesian Opt:     {}", bayesian_trials_to_target);

    if bayesian_trials_to_target < random_trials_to_target {
        println!("    🎯 Bayesian Optimization converged {} trials faster!",
                random_trials_to_target - bayesian_trials_to_target);
    } else if random_trials_to_target < bayesian_trials_to_target {
        println!("    🎯 Random Search converged {} trials faster!",
                bayesian_trials_to_target - random_trials_to_target);
    } else {
        println!("    🤝 Both methods converged at the same rate!");
    }
}