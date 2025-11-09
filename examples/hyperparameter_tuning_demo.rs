//! Comprehensive demonstration of the automated hyperparameter tuning framework
#![allow(unused_variables)]
//!
//! This example shows how to use the hyperparameter optimization system to
//! automatically find the best hyperparameters for training transformer models.

use std::collections::HashMap;
use std::time::Duration;
use trustformers_training::{
    // Hyperparameter tuning framework
    HyperparameterTuner, TunerConfig, OptimizationDirection,
    SearchSpaceBuilder, ParameterValue, TrialResult, TrialMetrics,
    GridSearch, RandomSearch, BayesianOptimization,
    EarlyStoppingConfig, PruningConfig, PruningStrategy,
    hyperparams_to_training_args,
    // Training infrastructure
    TrainingArguments, Trainer,
    // Metrics and losses
    Accuracy, F1Score, MetricCollection, CrossEntropyLoss,
};
use trustformers_core::{Result, Tensor};

/// Mock model for demonstration purposes
struct MockModel {
    name: String,
}

impl trustformers_core::traits::Model for MockModel {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Mock forward pass - just return the input
        Ok(input)
    }
}

/// Mock dataset entry
struct DatasetEntry {
    input: Tensor,
    target: Tensor,
}

fn create_mock_dataset() -> Vec<DatasetEntry> {
    // Create a small mock dataset for demonstration
    (0..100)
        .map(|i| DatasetEntry {
            input: Tensor::randn(&[1, 10]).unwrap(),
            target: Tensor::from_f32((i % 2) as f32).unwrap(),
        })
        .collect()
}

/// Simulated training function that evaluates hyperparameters
fn simulate_training(hyperparams: HashMap<String, ParameterValue>) -> Result<TrialResult> {
    println!("Training with hyperparameters:");
    for (name, value) in &hyperparams {
        println!("  {}: {}", name, value);
    }

    // Extract hyperparameters
    let learning_rate = hyperparams.get("learning_rate")
        .and_then(|v| v.as_float())
        .unwrap_or(0.001);

    let batch_size = hyperparams.get("batch_size")
        .and_then(|v| v.as_int())
        .unwrap_or(32) as usize;

    let optimizer = hyperparams.get("optimizer")
        .and_then(|v| v.as_string())
        .unwrap_or("adam");

    // Simulate training performance based on hyperparameters
    // In reality, this would run actual training
    let mut performance = 0.7; // Base performance

    // Learning rate effect
    if learning_rate > 0.0001 && learning_rate < 0.01 {
        performance += 0.1; // Good learning rate range
    } else if learning_rate > 0.1 {
        performance -= 0.2; // Too high, unstable training
    }

    // Batch size effect
    if batch_size >= 16 && batch_size <= 64 {
        performance += 0.05; // Good batch size range
    }

    // Optimizer effect
    match optimizer {
        "adam" | "adamw" => performance += 0.05,
        "sgd" => performance -= 0.02,
        _ => {},
    }

    // Add some randomness to simulate real training variance
    let noise = (rand::random::<f64>() - 0.5) * 0.1;
    performance += noise;

    // Clamp to realistic range
    performance = performance.clamp(0.0, 1.0);

    // Simulate intermediate values (for pruning)
    let mut metrics = TrialMetrics::new(performance);
    for step in (10..=100).step_by(10) {
        let intermediate_perf = performance * (step as f64 / 100.0) +
                               (rand::random::<f64>() - 0.5) * 0.05;
        metrics.add_intermediate_value(step, intermediate_perf.clamp(0.0, 1.0));
    }

    // Add additional metrics
    metrics = metrics
        .add_metric("loss", 1.0 - performance)
        .add_metric("f1_score", performance + 0.05)
        .add_metric("precision", performance - 0.02);

    println!("  → Performance: {:.4}", performance);
    println!();

    Ok(TrialResult::success(metrics))
}

fn demo_random_search() -> Result<()> {
    println!("=== Random Search Demo ===\n");

    // Define search space
    let search_space = SearchSpaceBuilder::new()
        .continuous("learning_rate", 1e-5, 1e-1)
        .discrete("batch_size", 8, 64, 8)
        .categorical("optimizer", vec!["adam", "adamw", "sgd"])
        .continuous("weight_decay", 0.0, 0.1)
        .build();

    // Configure tuner
    let config = TunerConfig::new("random_search_demo")
        .direction(OptimizationDirection::Maximize)
        .objective_metric("accuracy")
        .max_trials(20)
        .seed(42);

    // Create tuner with random search
    let mut tuner = HyperparameterTuner::with_random_search(config, search_space);

    // Run optimization
    let result = tuner.optimize(simulate_training)?;

    println!("Random Search Results:");
    println!("Best trial: {}", result.best_trial.summary());
    println!("Best value: {:.4}", result.best_trial.objective_value().unwrap_or(0.0));
    println!("Total trials: {}", result.trials.len());
    println!("Completed trials: {}", result.completed_trials);
    println!("Total duration: {:?}\n", result.total_duration);

    Ok(())
}

fn demo_bayesian_optimization() -> Result<()> {
    println!("=== Bayesian Optimization Demo ===\n");

    // Define search space
    let search_space = SearchSpaceBuilder::new()
        .log_uniform("learning_rate", 1e-5, 1e-1)
        .discrete("batch_size", 16, 128, 16)
        .categorical("optimizer", vec!["adam", "adamw"])
        .continuous("weight_decay", 0.0, 0.01)
        .continuous("adam_beta1", 0.8, 0.95)
        .continuous("adam_beta2", 0.9, 0.999)
        .build();

    // Configure tuner with early stopping
    let config = TunerConfig::new("bayesian_optimization_demo")
        .direction(OptimizationDirection::Maximize)
        .objective_metric("accuracy")
        .max_trials(30)
        .early_stopping(EarlyStoppingConfig {
            patience: 5,
            min_delta: 0.001,
            restore_best_weights: true,
        })
        .seed(42);

    // Create tuner with Bayesian optimization
    let mut tuner = HyperparameterTuner::with_bayesian_optimization(config, search_space);

    // Run optimization
    let result = tuner.optimize(simulate_training)?;

    println!("Bayesian Optimization Results:");
    println!("Best trial: {}", result.best_trial.summary());
    println!("Best value: {:.4}", result.best_trial.objective_value().unwrap_or(0.0));
    println!("Total trials: {}", result.trials.len());
    println!("Completed trials: {}", result.completed_trials);
    println!("Total duration: {:?}\n", result.total_duration);

    Ok(())
}

fn demo_grid_search() -> Result<()> {
    println!("=== Grid Search Demo ===\n");

    // Define a smaller discrete search space for grid search
    let search_space = SearchSpaceBuilder::new()
        .discrete("batch_size", 16, 32, 16)  // [16, 32]
        .categorical("optimizer", vec!["adam", "adamw"])  // 2 choices
        .categorical("scheduler", vec!["linear", "cosine"])  // 2 choices
        .build();

    let config = TunerConfig::new("grid_search_demo")
        .direction(OptimizationDirection::Maximize)
        .objective_metric("accuracy");

    // Create grid search strategy
    let strategy = Box::new(GridSearch::new(&search_space)?);
    let mut tuner = HyperparameterTuner::new(config, search_space, strategy);

    // Run optimization
    let result = tuner.optimize(simulate_training)?;

    println!("Grid Search Results:");
    println!("Best trial: {}", result.best_trial.summary());
    println!("Best value: {:.4}", result.best_trial.objective_value().unwrap_or(0.0));
    println!("Total trials: {}", result.trials.len());
    println!("Completed trials: {}", result.completed_trials);
    println!("Total duration: {:?}\n", result.total_duration);

    Ok(())
}

fn demo_pruning() -> Result<()> {
    println!("=== Pruning Demo ===\n");

    let search_space = SearchSpaceBuilder::new()
        .log_uniform("learning_rate", 1e-4, 1e-1)
        .discrete("batch_size", 16, 64, 16)
        .categorical("optimizer", vec!["adam", "adamw", "sgd"])
        .build();

    // Configure tuner with pruning
    let config = TunerConfig::new("pruning_demo")
        .direction(OptimizationDirection::Maximize)
        .objective_metric("accuracy")
        .max_trials(25)
        .pruning(PruningConfig {
            strategy: PruningStrategy::Median,
            min_steps: 30,
            percentile: 0.5,
        })
        .seed(42);

    let mut tuner = HyperparameterTuner::with_bayesian_optimization(config, search_space);

    // Run optimization
    let result = tuner.optimize(simulate_training)?;

    println!("Pruning Demo Results:");
    println!("Best trial: {}", result.best_trial.summary());
    println!("Best value: {:.4}", result.best_trial.objective_value().unwrap_or(0.0));
    println!("Total trials: {}", result.trials.len());
    println!("Completed trials: {}", result.completed_trials);
    println!("Pruned trials: {}", result.statistics.pruned_trials);
    println!("Pruning rate: {:.1}%", result.statistics.pruning_rate);
    println!("Total duration: {:?}\n", result.total_duration);

    Ok(())
}

fn demo_integration_with_training_args() -> Result<()> {
    println!("=== Integration with TrainingArguments Demo ===\n");

    let search_space = SearchSpaceBuilder::new()
        .log_uniform("learning_rate", 1e-5, 1e-2)
        .discrete("per_device_train_batch_size", 8, 32, 8)
        .continuous("weight_decay", 0.0, 0.1)
        .discrete("num_train_epochs", 3, 10, 1)
        .continuous("warmup_ratio", 0.0, 0.3)
        .continuous("adam_beta1", 0.85, 0.95)
        .continuous("adam_beta2", 0.9, 0.999)
        .build();

    let config = TunerConfig::new("training_args_demo")
        .direction(OptimizationDirection::Maximize)
        .objective_metric("eval_accuracy")
        .max_trials(15)
        .seed(42);

    let mut tuner = HyperparameterTuner::with_bayesian_optimization(config, search_space);

    // Base training arguments
    let base_args = TrainingArguments::new("./output")
        .do_eval = true;

    // Define objective function that uses TrainingArguments
    let objective_fn = |hyperparams: HashMap<String, ParameterValue>| -> Result<TrialResult> {
        // Convert hyperparameters to training arguments
        let training_args = hyperparams_to_training_args(&base_args, &hyperparams);

        println!("Training with updated arguments:");
        println!("  Learning rate: {}", training_args.learning_rate);
        println!("  Batch size: {}", training_args.per_device_train_batch_size);
        println!("  Weight decay: {}", training_args.weight_decay);
        println!("  Epochs: {}", training_args.num_train_epochs);
        println!("  Warmup ratio: {}", training_args.warmup_ratio);

        // Simulate training with these arguments
        // In a real scenario, you would create and train a model here
        simulate_training(hyperparams)
    };

    // Run optimization
    let result = tuner.optimize(objective_fn)?;

    println!("Training Args Integration Results:");
    println!("Best trial: {}", result.best_trial.summary());
    println!("Best value: {:.4}", result.best_trial.objective_value().unwrap_or(0.0));

    // Show the best hyperparameters converted to training args
    let best_training_args = hyperparams_to_training_args(&base_args, &result.best_trial.params);
    println!("Best training arguments:");
    println!("  Learning rate: {}", best_training_args.learning_rate);
    println!("  Batch size: {}", best_training_args.per_device_train_batch_size);
    println!("  Weight decay: {}", best_training_args.weight_decay);
    println!("  Epochs: {}", best_training_args.num_train_epochs);
    println!("  Warmup ratio: {}", best_training_args.warmup_ratio);
    println!();

    Ok(())
}

fn demo_study_persistence() -> Result<()> {
    println!("=== Study Persistence Demo ===\n");

    let search_space = SearchSpaceBuilder::new()
        .continuous("learning_rate", 1e-4, 1e-2)
        .discrete("batch_size", 16, 64, 16)
        .categorical("optimizer", vec!["adam", "adamw"])
        .build();

    let config = TunerConfig::new("persistence_demo")
        .direction(OptimizationDirection::Maximize)
        .objective_metric("accuracy")
        .max_trials(10)
        .output_dir("./hyperopt_results")
        .save_checkpoints(true)
        .seed(42);

    let mut tuner = HyperparameterTuner::with_random_search(config, search_space);

    // Run optimization (this will save checkpoints and results)
    let result = tuner.optimize(simulate_training)?;

    println!("Study results saved to: ./hyperopt_results/");
    println!("Files created:");
    println!("  - trial_history.json: Complete trial history");
    println!("  - statistics.json: Study statistics");
    println!("  - best_parameters.json: Best hyperparameters found");
    println!("  - checkpoint.json: Checkpoint for resuming");
    println!();

    println!("Persistence Demo Results:");
    println!("Best value: {:.4}", result.best_trial.objective_value().unwrap_or(0.0));
    println!("Total trials: {}", result.trials.len());

    Ok(())
}

fn main() -> Result<()> {
    println!("TrustformeRS Hyperparameter Tuning Framework Demo\n");
    println!("This demo showcases various hyperparameter optimization strategies");
    println!("and features available in the TrustformeRS training framework.\n");

    // Run all demonstrations
    demo_random_search()?;
    demo_bayesian_optimization()?;
    demo_grid_search()?;
    demo_pruning()?;
    demo_integration_with_training_args()?;
    demo_study_persistence()?;

    println!("=== Summary ===");
    println!("The hyperparameter tuning framework provides:");
    println!("✓ Multiple search strategies (Random, Bayesian, Grid, Successive Halving)");
    println!("✓ Flexible search space definitions (continuous, discrete, categorical, log-scale)");
    println!("✓ Advanced features (pruning, early stopping, checkpointing)");
    println!("✓ Integration with TrustformeRS training infrastructure");
    println!("✓ Comprehensive trial tracking and result persistence");
    println!("✓ Statistical analysis and visualization support");
    println!();
    println!("For production use:");
    println!("1. Define your search space based on known good ranges");
    println!("2. Start with Bayesian optimization for efficient exploration");
    println!("3. Enable pruning to save computational resources");
    println!("4. Use checkpointing for long-running studies");
    println!("5. Analyze results to understand hyperparameter importance");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_training() {
        let mut hyperparams = HashMap::new();
        hyperparams.insert("learning_rate".to_string(), ParameterValue::Float(0.001));
        hyperparams.insert("batch_size".to_string(), ParameterValue::Int(32));
        hyperparams.insert("optimizer".to_string(), ParameterValue::String("adam".to_string()));

        let result = simulate_training(hyperparams).unwrap();
        assert!(result.is_success());
        assert!(result.metrics.objective_value >= 0.0 && result.metrics.objective_value <= 1.0);
        assert!(!result.metrics.intermediate_values.is_empty());
    }

    #[test]
    fn test_search_space_creation() {
        let search_space = SearchSpaceBuilder::new()
            .continuous("learning_rate", 1e-5, 1e-1)
            .discrete("batch_size", 8, 64, 8)
            .categorical("optimizer", vec!["adam", "sgd"])
            .build();

        assert_eq!(search_space.parameters.len(), 3);
        assert!(search_space.get_parameter("learning_rate").is_some());
        assert!(search_space.get_parameter("batch_size").is_some());
        assert!(search_space.get_parameter("optimizer").is_some());
    }

    #[test]
    fn test_tuner_config() {
        let config = TunerConfig::new("test")
            .direction(OptimizationDirection::Maximize)
            .max_trials(10)
            .seed(42);

        assert_eq!(config.study_name, "test");
        assert_eq!(config.direction, OptimizationDirection::Maximize);
        assert_eq!(config.max_trials, Some(10));
        assert_eq!(config.seed, Some(42));
    }
}