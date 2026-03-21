# Automated Hyperparameter Tuning Framework

This document describes the comprehensive hyperparameter optimization framework implemented in TrustformeRS for automatically finding optimal hyperparameters for transformer model training.

## Overview

The hyperparameter tuning framework provides a systematic approach to optimize training configurations through:

- **Multiple Search Strategies**: Random search, Bayesian optimization, grid search, successive halving, and Hyperband
- **Flexible Search Spaces**: Support for continuous, discrete, categorical, and log-scale parameters
- **Advanced Features**: Pruning, early stopping, checkpointing, and result persistence
- **Seamless Integration**: Works with existing TrustformeRS training infrastructure

## Architecture

The framework is organized into several key modules:

### Core Components

1. **Search Space (`search_space.rs`)**: Defines hyperparameter ranges and types
2. **Trial Management (`trial.rs`)**: Tracks individual optimization trials
3. **Sampling Strategies (`sampler.rs`)**: Implements different parameter sampling methods
4. **Search Strategies (`strategies.rs`)**: Orchestrates the optimization process
5. **Main Tuner (`tuner.rs`)**: Coordinates the entire optimization workflow

## Key Features

### 🔍 Search Space Definition

```rust
use trustformers_training::{SearchSpaceBuilder, HyperparameterTuner, TunerConfig};

let search_space = SearchSpaceBuilder::new()
    .log_uniform("learning_rate", 1e-5, 1e-1)          // Log-scale parameter
    .discrete("batch_size", 8, 128, 8)                 // Discrete values
    .categorical("optimizer", vec!["adam", "adamw"])    // Categorical choices
    .continuous("weight_decay", 0.0, 0.1)              // Continuous range
    .build();
```

### 🎯 Search Strategies

#### Random Search
- Simple baseline approach
- Good for initial exploration
- Unbiased parameter sampling

#### Bayesian Optimization (TPE)
- Tree-structured Parzen Estimator
- Learns from previous trials
- Efficient for expensive evaluations

#### Grid Search
- Exhaustive search over discrete spaces
- Guarantees finding optimal configuration
- Limited to discrete/categorical parameters

#### Successive Halving & Hyperband
- Early stopping of unpromising trials
- Resource-efficient optimization
- Good for large-scale experiments

### ⚡ Advanced Features

#### Trial Pruning
Automatically stops underperforming trials:

```rust
let config = TunerConfig::new("study")
    .pruning(PruningConfig {
        strategy: PruningStrategy::Median,
        min_steps: 30,
        percentile: 0.5,
    });
```

#### Early Stopping
Prevents overfitting and saves time:

```rust
let config = TunerConfig::new("study")
    .early_stopping(EarlyStoppingConfig {
        patience: 5,
        min_delta: 0.001,
        restore_best_weights: true,
    });
```

#### Study Persistence
Automatic saving and resuming:

```rust
let config = TunerConfig::new("study")
    .output_dir("./hyperopt_results")
    .save_checkpoints(true);
```

## Usage Examples

### Basic Usage

```rust
use trustformers_training::*;

// Define search space
let search_space = SearchSpaceBuilder::new()
    .log_uniform("learning_rate", 1e-5, 1e-2)
    .discrete("batch_size", 16, 64, 16)
    .categorical("optimizer", vec!["adam", "adamw"])
    .build();

// Configure optimization
let config = TunerConfig::new("my_study")
    .direction(OptimizationDirection::Maximize)
    .objective_metric("eval_accuracy")
    .max_trials(50);

// Create tuner
let mut tuner = HyperparameterTuner::with_bayesian_optimization(config, search_space);

// Define objective function
let objective = |hyperparams: HashMap<String, ParameterValue>| -> Result<TrialResult> {
    // Convert hyperparameters to training arguments
    let training_args = hyperparams_to_training_args(&base_args, &hyperparams);
    
    // Train model with these parameters
    let trainer = Trainer::new(model, training_args, optimizer, loss_fn)?;
    trainer.train(&train_dataset, Some(&eval_dataset))?;
    
    // Extract metrics
    let eval_results = trainer.evaluate(&eval_dataset)?;
    let accuracy = eval_results.get("eval_accuracy").unwrap_or(&0.0);
    
    Ok(TrialResult::success(TrialMetrics::new(*accuracy)))
};

// Run optimization
let result = tuner.optimize(objective)?;

println!("Best hyperparameters found:");
for (name, value) in &result.best_trial.params {
    println!("  {}: {}", name, value);
}
```

### Integration with TrainingArguments

The framework seamlessly integrates with TrustformeRS training infrastructure:

```rust
// Base training configuration
let base_args = TrainingArguments::new("./output")
    .do_eval(true)
    .eval_steps(100)
    .save_steps(500);

// Hyperparameter space for training arguments
let search_space = SearchSpaceBuilder::new()
    .log_uniform("learning_rate", 1e-5, 1e-2)
    .discrete("per_device_train_batch_size", 8, 32, 8)
    .continuous("weight_decay", 0.0, 0.1)
    .discrete("num_train_epochs", 3, 10, 1)
    .continuous("warmup_ratio", 0.0, 0.3)
    .build();

// Objective function using training arguments
let objective = |hyperparams| {
    let training_args = hyperparams_to_training_args(&base_args, &hyperparams);
    // ... run training with updated arguments
};
```

## Parameter Types

### Continuous Parameters
For floating-point values within a range:
```rust
.continuous("weight_decay", 0.0, 0.1)
```

### Discrete Parameters
For integer values with step size:
```rust
.discrete("batch_size", 8, 128, 8)  // [8, 16, 24, ..., 128]
```

### Categorical Parameters
For string choices:
```rust
.categorical("optimizer", vec!["adam", "adamw", "sgd"])
```

### Log-Scale Parameters
For parameters that vary across orders of magnitude:
```rust
.log_uniform("learning_rate", 1e-5, 1e-1)  // Good for learning rates
```

## Study Management

### Results and Statistics
```rust
let result = tuner.optimize(objective)?;

println!("Study completed!");
println!("Best value: {:.4}", result.best_trial.objective_value().unwrap());
println!("Total trials: {}", result.trials.len());
println!("Success rate: {:.1}%", result.statistics.success_rate);
println!("Duration: {:?}", result.total_duration);
```

### Checkpointing and Resuming
```rust
// Save study state
tuner.save_checkpoint()?;

// Resume from checkpoint
let mut tuner = HyperparameterTuner::new(config, search_space, strategy);
tuner.load_checkpoint(&checkpoint_path)?;
let result = tuner.optimize(objective)?;  // Continues from where it left off
```

### Custom Callbacks
```rust
struct MyCallback;

impl HyperparameterCallback for MyCallback {
    fn on_trial_complete(&mut self, trial: &Trial) {
        println!("Trial {} completed with value: {:.4}", 
                 trial.number, trial.objective_value().unwrap_or(0.0));
    }
    
    fn on_new_best(&mut self, trial: &Trial, improvement: f64) {
        println!("New best! Trial {} improved by {:.4}", trial.number, improvement);
    }
}

let tuner = tuner.add_callback(Box::new(MyCallback));
```

## Best Practices

### 1. Search Space Design
- **Start Wide**: Begin with broad ranges to explore the space
- **Use Log Scale**: For learning rates, regularization parameters
- **Consider Dependencies**: Some parameters interact (e.g., batch size and learning rate)

### 2. Strategy Selection
- **Random Search**: Good baseline, works well for 1-10 parameters
- **Bayesian Optimization**: Best for expensive evaluations (< 100 trials)
- **Grid Search**: Only for small discrete spaces
- **Hyperband**: Good for cheap evaluations with many trials

### 3. Resource Management
- **Enable Pruning**: Stop bad trials early to save compute
- **Use Early Stopping**: Prevent overfitting in individual trials
- **Checkpoint Frequently**: For long-running studies

### 4. Evaluation Strategy
- **Hold-out Validation**: Use separate validation set for objective
- **Cross-validation**: More robust but computationally expensive
- **Multiple Metrics**: Track accuracy, loss, F1, etc.

## Performance Considerations

### Computational Efficiency
- **Parallel Trials**: Run multiple trials simultaneously when possible
- **Smart Pruning**: Use median or percentile-based pruning
- **Resource Allocation**: Use successive halving for budget constraints

### Memory Management
- **Checkpoint Cleanup**: Remove old checkpoints to save disk space
- **Result Storage**: Compress large trial histories
- **Model Cleanup**: Clear GPU memory between trials

## Example Output Files

When `save_checkpoints=true`, the framework creates:

```
hyperopt_results/
├── trial_history.json      # Complete trial records
├── statistics.json         # Study statistics
├── best_parameters.json    # Best hyperparameters found
└── checkpoint.json         # Resumable checkpoint
```

## Integration with Existing Code

The framework is designed to work with existing TrustformeRS components:

```rust
// Works with any model implementing the Model trait
let model = BertModel::from_pretrained("bert-base-uncased")?;

// Works with existing optimizers
let optimizer = AdamW::new(/* params from hyperopt */);

// Works with existing loss functions
let loss_fn = CrossEntropyLoss::new();

// Works with existing metrics
let metrics = MetricCollection::new()
    .add_metric(Box::new(Accuracy::new()))
    .add_metric(Box::new(F1Score::new()));
```

## Error Handling and Debugging

### Trial Failures
```rust
// Objective function should handle errors gracefully
let objective = |hyperparams| {
    match train_model(hyperparams) {
        Ok(metrics) => Ok(TrialResult::success(metrics)),
        Err(e) => {
            eprintln!("Trial failed: {}", e);
            Ok(TrialResult::failure(e.to_string()))
        }
    }
};
```

### Debugging Tips
- **Enable Logging**: Use the `LoggingCallback` for detailed output
- **Check Search Space**: Validate parameter ranges make sense
- **Monitor Progress**: Watch success rates and pruning statistics
- **Inspect Trials**: Examine failed trials for patterns

## Future Extensions

The framework is designed to be extensible:

- **Multi-objective Optimization**: Optimize multiple metrics simultaneously
- **Distributed Studies**: Run trials across multiple machines
- **Custom Samplers**: Implement domain-specific sampling strategies
- **Visualization**: Add plotting and analysis tools
- **Auto-ML Integration**: Combine with neural architecture search

## Conclusion

The TrustformeRS hyperparameter tuning framework provides a comprehensive solution for optimizing transformer model training. With support for multiple search strategies, advanced features like pruning and early stopping, and seamless integration with existing training infrastructure, it enables efficient and effective hyperparameter optimization at any scale.

For more examples and detailed usage, see the `hyperparameter_tuning_demo.rs` example in the repository.