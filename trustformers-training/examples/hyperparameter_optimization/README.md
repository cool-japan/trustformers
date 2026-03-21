# Hyperparameter Optimization Examples

This directory contains comprehensive examples demonstrating the advanced hyperparameter optimization capabilities of TrustformeRS. These examples showcase different optimization strategies and their practical applications in machine learning model training.

## 📋 Overview

The examples in this directory demonstrate:

- **Multiple Optimization Strategies**: Grid Search, Bayesian Optimization, Population-Based Training, and Multi-Armed Bandit algorithms
- **Advanced Features**: Early stopping, warm starting, surrogate models, and acquisition functions
- **Realistic Scenarios**: Model training simulation with various hyperparameter configurations
- **Performance Comparison**: Side-by-side comparison of different optimization approaches
- **Production-Ready Code**: Complete implementations ready for real-world use

## 🚀 Quick Start

### Running the Multi-Strategy Demo

```bash
# From the trustformers-training directory
cargo run --example hyperparameter_optimization/multi_strategy_optimization_demo

# Or compile and run directly
cd examples/hyperparameter_optimization
cargo run --bin multi_strategy_optimization_demo
```

### Using Custom Configuration

```bash
# Edit config.json to customize the optimization parameters
vim config.json

# Run with custom settings
cargo run --example multi_strategy_optimization_demo -- --config config.json
```

## 📚 Examples

### 1. Multi-Strategy Optimization Demo (`multi_strategy_optimization_demo.rs`)

**Purpose**: Comprehensive demonstration of all available optimization strategies with performance comparison.

**Key Features**:
- Simulated model training with realistic performance characteristics
- Grid Search with configurable resolution
- Bayesian Optimization with Gaussian Process surrogate models
- Population-Based Training with multiple generations
- Multi-Armed Bandit with early stopping
- Performance metrics and timing comparisons

**Usage**:
```rust
use trustformers_training::examples::hyperparameter_optimization::*;

fn main() -> Result<()> {
    // Run all optimization strategies and compare results
    run_optimization_comparison(&DemoConfig::default())
}
```

**Example Output**:
```
🚀 TrustformeRS Hyperparameter Optimization Demo
=================================================

🔍 === Grid Search Optimization Demo ===
Trial 1: {"learning_rate": 0.0001, "batch_size": 32, "weight_decay": 0.0001, "optimizer": "adamw"}
   📊 Accuracy: 0.8542, Training time: 312ms
   🎯 New best accuracy: 0.8542

🧠 === Bayesian Optimization Demo ===
Trial 1 (Bayesian): {"learning_rate": 0.00023, "batch_size": 32, "weight_decay": 8.7e-5, "optimizer": "adamw"}
   📊 Accuracy: 0.8634, Training time: 312ms
   🎯 New best accuracy: 0.8634

📈 Final Comparison:
   Grid Search: 45.2s
   Bayesian Optimization: 32.1s
   Population-Based Training: 28.7s
   Multi-Armed Bandit: 18.3s

🏃‍♂️ Fastest strategy: Multi-Armed Bandit (18.3s)
```

## ⚙️ Configuration

### Configuration File (`config.json`)

The configuration file allows you to customize various aspects of the optimization:

```json
{
  "demo_config": {
    "model_name": "bert-base-uncased",
    "max_trials": 30,
    "optimization_timeout_seconds": 1800,
    "enable_early_stopping": true
  },
  "search_space": {
    "learning_rate": {
      "type": "log_uniform",
      "min": 1e-5,
      "max": 0.01
    },
    "batch_size": {
      "type": "discrete",
      "min": 8,
      "max": 64,
      "step": 8
    }
  }
}
```

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_trials` | Maximum number of optimization trials | 50 |
| `optimization_timeout_seconds` | Maximum time for optimization | 3600 |
| `enable_early_stopping` | Enable early stopping for poor performers | true |
| `population_size` | Size of population for PBT | 8 |
| `n_initial_points` | Initial random points for Bayesian optimization | 5 |

## 📊 Optimization Strategies

### 1. Grid Search

**Best For**: 
- Small search spaces (< 6 dimensions)
- When computational budget is not a constraint
- Need to guarantee coverage of the search space

**Characteristics**:
- Exhaustive search over predefined grid
- Predictable runtime
- No intelligence in search strategy

```rust
let mut tuner: HyperparameterTuner<GridSearch> = HyperparameterTuner::new(
    GridSearch::new(10), // 10 points per dimension
    search_space,
    tuner_config,
)?;
```

### 2. Bayesian Optimization

**Best For**:
- Expensive function evaluations
- Continuous search spaces
- When you want intelligent exploration

**Characteristics**:
- Uses surrogate models (Gaussian Process)
- Balances exploration vs exploitation
- Efficient for expensive evaluations

```rust
let surrogate_config = SurrogateConfig {
    model_type: SurrogateModelType::GaussianProcess,
    acquisition_function: AcquisitionFunctionType::ExpectedImprovement,
    n_initial_points: 5,
};
```

### 3. Population-Based Training (PBT)

**Best For**:
- Training schedules that can be adjusted during training
- When you can checkpoint and resume training
- Resource-efficient hyperparameter adaptation

**Characteristics**:
- Evolves population of configurations
- Can adapt hyperparameters during training
- Combines multiple promising configurations

```rust
let pbt_config = PBTConfig {
    population_size: 8,
    perturbation_factor: 0.2,
    exploitation_strategy: ExploitationStrategy::TournamentSelection,
};
```

### 4. Multi-Armed Bandit

**Best For**:
- Limited computational budget
- When early stopping is beneficial
- Fast convergence requirements

**Characteristics**:
- Successive halving of poor performers
- Resource-aware optimization
- Fast identification of promising regions

```rust
let bandit_config = BanditConfig {
    algorithm: BanditAlgorithm::UCB,
    n_arms: 20,
    evaluation_budget: 100,
    reduction_factor: 3,
};
```

## 🎯 Performance Comparison

Based on our simulation results:

| Strategy | Speed | Efficiency | Best For |
|----------|-------|------------|----------|
| Grid Search | ⭐⭐ | ⭐⭐ | Small spaces, guaranteed coverage |
| Bayesian Opt | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Expensive evaluations, continuous spaces |
| PBT | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Adaptive training schedules |
| Bandit | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Limited budget, fast convergence |

## 🛠️ Customization

### Adding New Objective Functions

```rust
pub struct CustomObjective {
    // Your custom fields
}

impl CustomObjective {
    pub fn evaluate(&self, parameters: &HashMap<String, ParameterValue>) -> Result<f64> {
        // Your custom evaluation logic
        // Return the metric you want to optimize (higher is better)
    }
}
```

### Custom Search Spaces

```rust
let search_space = SearchSpaceBuilder::new()
    .log_uniform("learning_rate", 1e-5, 1e-2)    // Log-uniform distribution
    .continuous("dropout", 0.0, 0.5)             // Linear uniform
    .discrete("layers", 6, 12, 2)                // Integer values: 6, 8, 10, 12
    .categorical("activation", vec![              // Discrete choices
        "relu".to_string(), 
        "gelu".to_string(), 
        "swish".to_string()
    ])
    .build();
```

### Custom Early Stopping

```rust
let early_stopping = EarlyStoppingConfig {
    patience: 10,           // Wait 10 trials without improvement
    min_delta: 0.001,       // Minimum improvement threshold
    mode: EarlyStoppingMode::Maximize,
    baseline: Some(0.85),   // Stop if we can't beat 85% accuracy
};
```

## 📈 Monitoring and Debugging

### Logging

The examples include comprehensive logging:

```rust
println!("🔥 Training model with parameters: {:?}", parameters);
println!("   📊 Accuracy: {:.4}, Training time: {:?}", accuracy, training_time);
println!("   🎯 New best accuracy: {:.4}", accuracy);
```

### Performance Metrics

Each strategy reports:
- Best accuracy achieved
- Best hyperparameters found
- Total time taken
- Number of trials completed
- Convergence statistics

### Visualization

Results can be visualized by:
1. Tracking accuracy over time
2. Comparing strategy performance
3. Analyzing parameter importance

## 🚨 Common Issues and Solutions

### Issue: "Parameter not found or invalid type"
**Solution**: Ensure your search space matches the parameters expected by your objective function.

### Issue: Poor optimization performance
**Solution**: 
- Increase the number of trials
- Adjust the search space bounds
- Try a different optimization strategy
- Check if your objective function is noisy

### Issue: Optimization takes too long
**Solution**:
- Enable early stopping
- Reduce max_trials
- Use bandit optimization for faster convergence
- Implement parallel evaluation (if supported)

## 🤝 Contributing

To add new optimization examples:

1. Create a new `.rs` file in this directory
2. Implement your optimization logic
3. Add appropriate tests
4. Update this README
5. Add configuration options to `config.json`

## 📖 Further Reading

- [Hyperparameter Optimization in Machine Learning](https://en.wikipedia.org/wiki/Hyperparameter_optimization)
- [Bayesian Optimization](https://krasserm.github.io/2018/03/21/bayesian-optimization/)
- [Population-Based Training](https://deepmind.com/blog/article/population-based-training-neural-networks)
- [Multi-Armed Bandit Algorithms](https://en.wikipedia.org/wiki/Multi-armed_bandit)

## 📄 License

This example is part of the TrustformeRS project and follows the same license terms.