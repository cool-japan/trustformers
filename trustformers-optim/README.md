# trustformers-optim

Optimization algorithms and learning rate schedulers for training transformer models.

## Current State

This crate provides **comprehensive optimization infrastructure** including state-of-the-art optimizers, learning rate schedulers, and distributed optimization techniques. It includes implementations of Adam, AdamW, SGD, LAMB, AdaFactor, and all three ZeRO optimization stages.

## Features

### Optimizers
- **SGD**: Stochastic Gradient Descent with momentum and weight decay
- **Adam**: Adaptive Moment Estimation optimizer
- **AdamW**: Adam with decoupled weight decay (recommended for transformers)
- **LAMB**: Layer-wise Adaptive Moments optimizer for large batch training
- **AdaFactor**: Memory-efficient optimizer with adaptive learning rates

### Learning Rate Schedulers
- **Linear**: Linear warmup and decay
- **Cosine**: Cosine annealing with optional warmup
- **Polynomial**: Polynomial decay with configurable power
- **Constant**: Constant learning rate with optional warmup
- **Exponential**: Exponential decay
- **Step**: Step-wise learning rate reduction

### Distributed Optimization
- **ZeRO Stage 1**: Optimizer state partitioning across GPUs
- **ZeRO Stage 2**: Optimizer state + gradient partitioning
- **ZeRO Stage 3**: Full parameter partitioning for maximum memory efficiency
- **Gradient Synchronization**: Efficient all-reduce operations
- **Mixed Precision Support**: Compatible with FP16/BF16 training

### Advanced Features
- **Gradient Clipping**: By value or norm
- **Weight Decay**: L2 regularization and decoupled weight decay
- **Momentum**: Classical and Nesterov momentum
- **Adaptive Learning Rates**: Per-parameter learning rate adaptation
- **Memory Optimization**: Reduced memory footprint for large models

## Usage Example

### Basic Optimizer Usage
```rust
use trustformers_optim::{
    optimizers::{AdamW, AdamWConfig},
    schedulers::{LinearScheduler, SchedulerConfig},
    Optimizer,
};

// Create AdamW optimizer
let config = AdamWConfig {
    lr: 5e-5,
    betas: (0.9, 0.999),
    eps: 1e-8,
    weight_decay: 0.01,
    correct_bias: true,
};
let mut optimizer = AdamW::new(config)?;

// Create learning rate scheduler
let scheduler_config = SchedulerConfig {
    num_warmup_steps: 1000,
    num_training_steps: 10000,
};
let scheduler = LinearScheduler::new(scheduler_config);

// Training loop
for step in 0..num_steps {
    // Forward pass
    let loss = model.forward(&batch)?;
    
    // Backward pass
    let gradients = loss.backward()?;
    
    // Update learning rate
    let lr = scheduler.get_lr(step);
    optimizer.set_lr(lr);
    
    // Optimizer step
    optimizer.step(&mut model.parameters(), &gradients)?;
    optimizer.zero_grad();
}
```

### ZeRO Optimization
```rust
use trustformers_optim::{
    distributed::{ZeroOptimizer, ZeroConfig, ZeroStage},
    optimizers::AdamW,
};

// Configure ZeRO
let zero_config = ZeroConfig {
    stage: ZeroStage::Three,
    partition_gradients: true,
    contiguous_gradients: true,
    overlap_comm: true,
    reduce_scatter: true,
    cpu_offload: false,
};

// Wrap optimizer with ZeRO
let base_optimizer = AdamW::new(adam_config)?;
let optimizer = ZeroOptimizer::new(
    base_optimizer,
    model,
    zero_config,
    process_group,
)?;
```

## Architecture

```
trustformers-optim/
├── src/
│   ├── optimizers/       # Optimizer implementations
│   │   ├── sgd.rs       # SGD optimizer
│   │   ├── adam.rs      # Adam & AdamW
│   │   ├── lamb.rs      # LAMB optimizer
│   │   └── adafactor.rs # AdaFactor optimizer
│   ├── schedulers/       # Learning rate schedulers
│   ├── distributed/      # Distributed optimization
│   │   ├── zero.rs      # ZeRO implementation
│   │   └── utils.rs     # Communication utilities
│   └── traits.rs        # Core traits
```

## Performance

### Memory Savings with ZeRO
| Model Size | Standard | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|------------|----------|---------|---------|---------|
| 1.5B params | 24 GB | 16 GB | 12 GB | 8 GB |
| 7B params | 112 GB | 75 GB | 56 GB | 28 GB |
| 175B params | 2.8 TB | 1.9 TB | 1.4 TB | 700 GB |

### Optimizer Performance
- **AdamW**: Industry standard for transformer training
- **LAMB**: Enables large batch training (up to 64K)
- **AdaFactor**: 75% memory reduction vs Adam
- **ZeRO**: Near-linear scaling across multiple GPUs

## Best Practices

### Choosing an Optimizer
- **AdamW**: Default choice for most transformer models
- **LAMB**: When using very large batch sizes
- **AdaFactor**: Memory-constrained environments
- **SGD**: Simple baseline, rarely optimal for transformers

### Learning Rate Schedules
- **Linear**: Standard for BERT-style pre-training
- **Cosine**: Often better for fine-tuning
- **Constant + Warmup**: Simple and effective
- **Polynomial**: Alternative to linear decay

### Hyperparameters
```rust
// Recommended starting points
AdamW: lr=5e-5, weight_decay=0.01, warmup=10% of steps
LAMB: lr=2e-3, weight_decay=0.01, warmup=10% of steps
AdaFactor: lr=1e-3, no weight_decay, warmup=10% of steps
```

## Testing

- Unit tests for all optimizers and schedulers
- Convergence tests on toy problems
- Numerical stability tests
- Distributed operation tests
- Memory usage profiling

## License

MIT OR Apache-2.0