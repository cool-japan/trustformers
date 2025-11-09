//! Multi-Strategy Hyperparameter Optimization Demo
#![allow(unused_variables)]
//!
//! This example demonstrates all available hyperparameter optimization strategies
//! in TrustformeRS and compares their performance across different scenarios.
//!
//! Features demonstrated:
//! - Grid Search with configurable resolution
//! - Bayesian Optimization with Gaussian Process surrogate models
//! - Population-Based Training with multiple generations
//! - Multi-Armed Bandit with successive halving
//! - Performance metrics and timing comparisons
//!
//! Usage:
//! ```bash
//! cd examples/hyperparameter_optimization
//! cargo run --bin multi_strategy_optimization_demo
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::time::{Duration, Instant};
use trustformers_core::{Result, TrustformersError};
use trustformers_training::hyperopt::search_space::SearchSpaceBuilder;
use trustformers_training::hyperopt::{
    BanditAlgorithm, BanditConfig, BanditOptimizer, GridSearch, HyperparameterTuner,
    OptimizationDirection, PBTConfig, ParameterValue, PopulationBasedTraining, SearchSpace,
    TrialMetrics, TrialResult, TunerConfig,
};

/// Configuration for the optimization demo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoConfig {
    pub model_name: String,
    pub dataset_size: usize,
    pub max_trials: usize,
    pub optimization_timeout_seconds: u64,
    pub enable_early_stopping: bool,
    pub warm_start_from_history: bool,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            model_name: "bert-base-uncased".to_string(),
            dataset_size: 10000,
            max_trials: 30,
            optimization_timeout_seconds: 1800,
            enable_early_stopping: true,
            warm_start_from_history: false,
        }
    }
}

/// Configuration for optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    pub enabled: bool,
    pub max_trials: usize,
}

/// Configuration loaded from config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub demo_config: DemoConfig,
    pub optimization_strategies: HashMap<String, serde_json::Value>,
    pub search_space: HashMap<String, serde_json::Value>,
    pub early_stopping: HashMap<String, serde_json::Value>,
    pub logging: HashMap<String, serde_json::Value>,
}

/// Results for a single optimization strategy
#[derive(Debug)]
pub struct StrategyResult {
    pub name: String,
    pub best_accuracy: f64,
    pub best_params: HashMap<String, ParameterValue>,
    pub total_trials: usize,
    pub duration: Duration,
    pub convergence_trial: usize,
}

/// Comprehensive results for all strategies
#[derive(Debug)]
pub struct ComparisonResults {
    pub strategies: Vec<StrategyResult>,
    pub fastest_strategy: String,
    pub most_accurate_strategy: String,
    pub most_efficient_strategy: String,
}

/// Custom objective function that simulates realistic model training
fn demo_objective_function(params: HashMap<String, ParameterValue>) -> Result<TrialResult> {
    let start_time = Instant::now();

    // Extract hyperparameters with defaults
    let learning_rate = params.get("learning_rate").and_then(|v| v.as_float()).unwrap_or(3e-4);

    let batch_size = params.get("batch_size").and_then(|v| v.as_int()).unwrap_or(32) as usize;

    let weight_decay = params.get("weight_decay").and_then(|v| v.as_float()).unwrap_or(1e-2);

    let optimizer = params.get("optimizer").and_then(|v| v.as_string()).unwrap_or("adamw");

    let warmup_ratio = params.get("warmup_ratio").and_then(|v| v.as_float()).unwrap_or(0.1);

    let max_grad_norm = params.get("max_grad_norm").and_then(|v| v.as_float()).unwrap_or(1.0);

    // Simulate training time based on batch size and complexity
    let training_time_ms = 200 + (batch_size as u64 * 5) + fastrand::u64(50..150);
    std::thread::sleep(Duration::from_millis(training_time_ms));

    // Realistic performance simulation based on hyperparameters
    let mut base_accuracy = 0.82;

    // Learning rate effects
    if learning_rate > 1e-2 {
        base_accuracy -= 0.15; // Too high LR hurts performance
    } else if learning_rate < 1e-5 {
        base_accuracy -= 0.08; // Too low LR is slow to converge
    } else if learning_rate >= 1e-4 && learning_rate <= 5e-3 {
        base_accuracy += 0.03; // Sweet spot
    }

    // Batch size effects
    if batch_size < 8 {
        base_accuracy -= 0.05; // Too small batch is noisy
    } else if batch_size > 128 {
        base_accuracy -= 0.03; // Too large batch might be less generalizable
    } else if batch_size >= 16 && batch_size <= 64 {
        base_accuracy += 0.02; // Good batch size range
    }

    // Weight decay effects
    if weight_decay > 1e-1 {
        base_accuracy -= 0.06; // Too much regularization
    } else if weight_decay >= 1e-4 && weight_decay <= 1e-2 {
        base_accuracy += 0.01; // Good regularization
    }

    // Optimizer effects
    match optimizer {
        "adamw" => base_accuracy += 0.02,
        "adam" => base_accuracy += 0.01,
        "adafactor" => base_accuracy += 0.015,
        "sgd" => base_accuracy -= 0.01,
        _ => {},
    }

    // Warmup ratio effects
    if warmup_ratio >= 0.05 && warmup_ratio <= 0.15 {
        base_accuracy += 0.005;
    }

    // Gradient clipping effects
    if max_grad_norm >= 0.5 && max_grad_norm <= 2.0 {
        base_accuracy += 0.005;
    }

    // Add some realistic noise
    let noise = fastrand::f64() * 0.02 - 0.01; // ±1% noise
    let final_accuracy = (base_accuracy + noise).clamp(0.65, 0.95);

    let loss = 1.5 - final_accuracy; // Inverse relationship

    // Create metrics
    let mut metrics = HashMap::new();
    metrics.insert("eval_accuracy".to_string(), final_accuracy);
    metrics.insert("eval_loss".to_string(), loss);
    metrics.insert("train_loss".to_string(), loss + 0.05);
    metrics.insert("perplexity".to_string(), loss.exp());

    // Simulate training progression
    let epochs = 3;
    let mut intermediate_values = Vec::new();
    for epoch in 1..=epochs {
        let progress = epoch as f64 / epochs as f64;
        let intermediate_acc = final_accuracy * (0.6 + 0.4 * progress);
        intermediate_values.push((epoch * 1000, intermediate_acc));
    }

    let trial_metrics = TrialMetrics {
        objective_value: final_accuracy,
        metrics,
        intermediate_values,
    };

    // Add actual training time information
    let actual_duration = start_time.elapsed();

    println!(
        "   📊 Accuracy: {:.4}, Training time: {:?}",
        final_accuracy, actual_duration
    );
    if final_accuracy > 0.85 {
        println!("   🎯 New best accuracy: {:.4}", final_accuracy);
    }

    Ok(TrialResult::success(trial_metrics))
}

/// Run Grid Search optimization
fn run_grid_search(search_space: &SearchSpace, config: &DemoConfig) -> Result<StrategyResult> {
    println!("\n🔍 === Grid Search Optimization Demo ===");
    let start_time = Instant::now();

    let tuner_config = TunerConfig::new("grid_search_demo")
        .direction(OptimizationDirection::Maximize)
        .max_trials(15) // Reduced for demo purposes
        .max_duration(Duration::from_secs(300));

    let grid_strategy = Box::new(
        GridSearch::new(search_space)
            .map_err(|e| TrustformersError::config_error(&e, "grid_search_creation"))?,
    );
    let mut tuner = HyperparameterTuner::new(tuner_config, search_space.clone(), grid_strategy);

    let result = tuner.optimize(demo_objective_function)?;
    let duration = start_time.elapsed();

    let best_accuracy = result
        .best_trial
        .result
        .as_ref()
        .map(|r| r.metrics.objective_value)
        .unwrap_or(0.0);
    let best_params = result.best_trial.params.clone();

    println!(
        "✅ Grid Search completed: {:.4} accuracy in {:?}",
        best_accuracy, duration
    );

    Ok(StrategyResult {
        name: "Grid Search".to_string(),
        best_accuracy,
        best_params,
        total_trials: result.completed_trials,
        duration,
        convergence_trial: result.completed_trials,
    })
}

/// Run Bayesian Optimization
fn run_bayesian_optimization(
    search_space: &SearchSpace,
    config: &DemoConfig,
) -> Result<StrategyResult> {
    println!("\n🧠 === Bayesian Optimization Demo ===");
    let start_time = Instant::now();

    let tuner_config = TunerConfig::new("bayesian_optimization_demo")
        .direction(OptimizationDirection::Maximize)
        .max_trials(25)
        .max_duration(Duration::from_secs(400));

    let mut tuner =
        HyperparameterTuner::with_bayesian_optimization(tuner_config, search_space.clone());

    let result = tuner.optimize(demo_objective_function)?;
    let duration = start_time.elapsed();

    let best_accuracy = result
        .best_trial
        .result
        .as_ref()
        .map(|r| r.metrics.objective_value)
        .unwrap_or(0.0);
    let best_params = result.best_trial.params.clone();

    println!(
        "✅ Bayesian Optimization completed: {:.4} accuracy in {:?}",
        best_accuracy, duration
    );

    Ok(StrategyResult {
        name: "Bayesian Optimization".to_string(),
        best_accuracy,
        best_params,
        total_trials: result.completed_trials,
        duration,
        convergence_trial: result.completed_trials,
    })
}

/// Run Population-Based Training
fn run_population_based_training(
    search_space: &SearchSpace,
    config: &DemoConfig,
) -> Result<StrategyResult> {
    println!("\n📈 === Population-Based Training Demo ===");
    let start_time = Instant::now();

    let tuner_config = TunerConfig::new("pbt_demo")
        .direction(OptimizationDirection::Maximize)
        .max_trials(20)
        .max_duration(Duration::from_secs(350));

    let pbt_config = PBTConfig {
        population_size: 8,
        exploit_interval: 500,
        exploit_fraction: 0.25,
        perturbation_std: 0.2,
        seed: Some(42),
        ..Default::default()
    };

    let pbt_strategy = Box::new(PopulationBasedTraining::new(pbt_config, search_space));
    let mut tuner = HyperparameterTuner::new(tuner_config, search_space.clone(), pbt_strategy);

    let result = tuner.optimize(demo_objective_function)?;
    let duration = start_time.elapsed();

    let best_accuracy = result
        .best_trial
        .result
        .as_ref()
        .map(|r| r.metrics.objective_value)
        .unwrap_or(0.0);
    let best_params = result.best_trial.params.clone();

    println!(
        "✅ Population-Based Training completed: {:.4} accuracy in {:?}",
        best_accuracy, duration
    );

    Ok(StrategyResult {
        name: "Population-Based Training".to_string(),
        best_accuracy,
        best_params,
        total_trials: result.completed_trials,
        duration,
        convergence_trial: result.completed_trials,
    })
}

/// Run Multi-Armed Bandit optimization
fn run_bandit_optimization(
    search_space: &SearchSpace,
    config: &DemoConfig,
) -> Result<StrategyResult> {
    println!("\n🎰 === Multi-Armed Bandit Demo ===");
    let start_time = Instant::now();

    let tuner_config = TunerConfig::new("bandit_demo")
        .direction(OptimizationDirection::Maximize)
        .max_trials(18)
        .max_duration(Duration::from_secs(250));

    let bandit_config = BanditConfig {
        algorithm: BanditAlgorithm::UCB {
            confidence_parameter: 1.5,
        },
        num_arms: 30,
        ..Default::default()
    };

    let bandit_strategy = Box::new(BanditOptimizer::new(bandit_config, search_space)?);
    let mut tuner = HyperparameterTuner::new(tuner_config, search_space.clone(), bandit_strategy);

    let result = tuner.optimize(demo_objective_function)?;
    let duration = start_time.elapsed();

    let best_accuracy = result
        .best_trial
        .result
        .as_ref()
        .map(|r| r.metrics.objective_value)
        .unwrap_or(0.0);
    let best_params = result.best_trial.params.clone();

    println!(
        "✅ Multi-Armed Bandit completed: {:.4} accuracy in {:?}",
        best_accuracy, duration
    );

    Ok(StrategyResult {
        name: "Multi-Armed Bandit".to_string(),
        best_accuracy,
        best_params,
        total_trials: result.completed_trials,
        duration,
        convergence_trial: result.completed_trials,
    })
}

/// Build the search space from configuration
fn build_search_space() -> SearchSpace {
    SearchSpaceBuilder::new()
        .log_uniform("learning_rate", 1e-5, 1e-2)
        .discrete("batch_size", 8, 64, 8)
        .log_uniform("weight_decay", 1e-6, 1e-2)
        .categorical(
            "optimizer",
            vec![
                "adamw".to_string(),
                "adam".to_string(),
                "sgd".to_string(),
                "adafactor".to_string(),
            ],
        )
        .continuous("warmup_ratio", 0.0, 0.2)
        .continuous("max_grad_norm", 0.5, 2.0)
        .build()
}

/// Load configuration from config.json file
fn load_config() -> Result<DemoConfig> {
    match fs::read_to_string("config.json") {
        Ok(content) => {
            let config: OptimizationConfig = serde_json::from_str(&content).map_err(|e| {
                TrustformersError::config_error(
                    &format!("Failed to parse config.json: {}", e),
                    "config_parsing",
                )
            })?;
            Ok(config.demo_config)
        },
        Err(_) => {
            println!("⚠️  config.json not found, using default configuration");
            Ok(DemoConfig::default())
        },
    }
}

/// Analyze and compare results from all strategies
fn analyze_results(results: Vec<StrategyResult>) -> ComparisonResults {
    let fastest_strategy = results
        .iter()
        .min_by_key(|r| r.duration)
        .map(|r| r.name.clone())
        .unwrap_or_else(|| "Unknown".to_string());

    let most_accurate_strategy = results
        .iter()
        .max_by(|a, b| a.best_accuracy.partial_cmp(&b.best_accuracy).unwrap())
        .map(|r| r.name.clone())
        .unwrap_or_else(|| "Unknown".to_string());

    // Efficiency = accuracy / time (simplified metric)
    let most_efficient_strategy = results
        .iter()
        .max_by(|a, b| {
            let eff_a = a.best_accuracy / a.duration.as_secs_f64();
            let eff_b = b.best_accuracy / b.duration.as_secs_f64();
            eff_a.partial_cmp(&eff_b).unwrap()
        })
        .map(|r| r.name.clone())
        .unwrap_or_else(|| "Unknown".to_string());

    ComparisonResults {
        strategies: results,
        fastest_strategy,
        most_accurate_strategy,
        most_efficient_strategy,
    }
}

/// Print comprehensive results comparison
fn print_results(results: &ComparisonResults) {
    println!("\n🏆 === OPTIMIZATION RESULTS COMPARISON ===");
    println!("{}", "=".repeat(50));

    // Individual strategy results
    for result in &results.strategies {
        println!("\n📋 {}", result.name);
        println!("   🎯 Best Accuracy: {:.4}", result.best_accuracy);
        println!("   ⏱️  Total Time: {:.2}s", result.duration.as_secs_f64());
        println!("   🔄 Trials: {}", result.total_trials);
        println!(
            "   ⚡ Efficiency: {:.6} acc/sec",
            result.best_accuracy / result.duration.as_secs_f64()
        );

        // Print best parameters (abbreviated)
        println!("   📊 Best Parameters:");
        for (key, value) in result.best_params.iter().take(3) {
            match value {
                ParameterValue::Float(f) => println!("      {}: {:.6}", key, f),
                ParameterValue::Int(i) => println!("      {}: {}", key, i),
                ParameterValue::String(s) => println!("      {}: {}", key, s),
                ParameterValue::Bool(b) => println!("      {}: {}", key, b),
            }
        }
        if result.best_params.len() > 3 {
            println!("      ... and {} more", result.best_params.len() - 3);
        }
    }

    // Summary comparison
    println!("\n🏁 === FINAL SUMMARY ===");
    println!("🚀 Fastest Strategy: {}", results.fastest_strategy);
    println!(
        "🎯 Most Accurate Strategy: {}",
        results.most_accurate_strategy
    );
    println!(
        "⚡ Most Efficient Strategy: {}",
        results.most_efficient_strategy
    );

    // Performance table
    println!("\n📊 Performance Comparison Table:");
    println!("| Strategy                  | Accuracy | Time (s) | Trials | Efficiency |");
    println!("| ------------------------- | -------- | -------- | ------ | ---------- |");
    for result in &results.strategies {
        let efficiency = result.best_accuracy / result.duration.as_secs_f64();
        println!(
            "| {:<25} | {:.4}   | {:.2}    | {:>6} | {:.6}   |",
            result.name,
            result.best_accuracy,
            result.duration.as_secs_f64(),
            result.total_trials,
            efficiency
        );
    }

    // Recommendations
    println!("\n💡 === RECOMMENDATIONS ===");
    println!("🔬 For Research/Experimentation: Use Bayesian Optimization for thorough exploration");
    println!("⚡ For Quick Results: Use Multi-Armed Bandit for fast convergence");
    println!("🏭 For Production Training: Use Population-Based Training for adaptive schedules");
    println!("📐 For Complete Coverage: Use Grid Search when computational budget allows");
}

/// Main demonstration function
pub fn run_optimization_comparison(config: &DemoConfig) -> Result<()> {
    println!("🚀 TrustformeRS Hyperparameter Optimization Demo");
    println!("{}", "=".repeat(50));
    println!("📝 Model: {}", config.model_name);
    println!("📊 Dataset Size: {}", config.dataset_size);
    println!("🎯 Max Trials per Strategy: {}", config.max_trials);
    println!("⏰ Timeout: {}s", config.optimization_timeout_seconds);
    println!();

    let search_space = build_search_space();
    let mut results = Vec::new();

    // Run all optimization strategies
    results.push(run_grid_search(&search_space, config)?);
    results.push(run_bayesian_optimization(&search_space, config)?);
    results.push(run_population_based_training(&search_space, config)?);
    results.push(run_bandit_optimization(&search_space, config)?);

    // Analyze and print results
    let comparison = analyze_results(results);
    print_results(&comparison);

    println!("\n✅ Hyperparameter optimization demonstration completed!");
    println!("💡 Try modifying config.json to experiment with different settings");

    Ok(())
}

fn main() -> Result<()> {
    // Load configuration
    let config = load_config()?;

    // Run the optimization comparison
    run_optimization_comparison(&config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_objective_function() {
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), ParameterValue::Float(1e-4));
        params.insert("batch_size".to_string(), ParameterValue::Int(32));
        params.insert("weight_decay".to_string(), ParameterValue::Float(1e-3));
        params.insert(
            "optimizer".to_string(),
            ParameterValue::String("adamw".to_string()),
        );

        let result = demo_objective_function(params).unwrap();

        if result.is_success() {
            assert!(result.metrics.objective_value >= 0.65);
            assert!(result.metrics.objective_value <= 0.95);
            assert!(result.metrics.metrics.contains_key("eval_accuracy"));
            assert!(result.metrics.metrics.contains_key("eval_loss"));
        } else {
            panic!("Expected successful trial result");
        }
    }

    #[test]
    fn test_build_search_space() {
        let search_space = build_search_space();
        assert!(search_space.parameters.len() >= 6);
    }

    #[test]
    fn test_analyze_results() {
        let results = vec![StrategyResult {
            name: "Test Strategy".to_string(),
            best_accuracy: 0.85,
            best_params: HashMap::new(),
            total_trials: 10,
            duration: Duration::from_secs(30),
            convergence_trial: 8,
        }];

        let comparison = analyze_results(results);
        assert_eq!(comparison.fastest_strategy, "Test Strategy");
        assert_eq!(comparison.most_accurate_strategy, "Test Strategy");
        assert_eq!(comparison.most_efficient_strategy, "Test Strategy");
    }
}
