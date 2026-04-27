# trustformers-optim

**Version:** 0.1.1 | **Status:** Stable | **Tests:** 583 | **SLoC:** 43,888 | **Updated:** 2026-04-25

Comprehensive optimization algorithms, learning rate schedulers, and distributed optimization techniques for training transformer models.

## Overview

This crate provides **comprehensive optimization infrastructure** including state-of-the-art optimizers, learning rate schedulers, quantized optimizers, and distributed optimization techniques. It includes implementations of 20+ optimizers ranging from standard methods (SGD, Adam, AdamW) to cutting-edge research algorithms (Lion, Muon, CAME, MicroAdam, BGE-Adam, HN-Adam, AdEMAMix) and Schedule-Free variants, plus 4-bit/8-bit quantized optimizers and all three ZeRO optimization stages.

## Features

### Standard Optimizers
- **SGD**: Stochastic Gradient Descent with momentum and weight decay
- **Adam**: Adaptive Moment Estimation optimizer
- **AdamW**: Adam with decoupled weight decay (recommended for transformers)
- **LAMB**: Layer-wise Adaptive Moments optimizer for large batch training
- **AdaFactor**: Memory-efficient optimizer with adaptive learning rates
- **RAdam**: Rectified Adam with automatic variance warmup

### Advanced Research Optimizers
- **Lion**: Sign-based optimizer discovered via evolutionary search; memory-efficient, competitive with AdamW
- **Muon**: Momentum + Orthogonalization; combines Nesterov momentum with Gram-Schmidt orthogonalization of updates for improved generalization
- **CAME**: Confidence-guided Adaptive Memory-Efficient optimizer; confidence-guided second moment with memory footprint similar to AdaFactor
- **MicroAdam**: Micro-batch Adam with gradient compression; quantized gradient accumulation for extremely memory-constrained training
- **BGE-Adam**: Bias-corrected Gradient Estimation Adam; improved bias correction for large-batch scenarios
- **HN-Adam**: Hyperbolic Nesterov Adam; Nesterov correction in hyperbolic space for transformer training
- **AdEMAMix**: Adaptive EMA Mixture optimizer; mixes fast and slow EMA of gradients for improved long-range memory
- **Schedule-Free variants**: Schedule-Free Adam and Schedule-Free SGD; eliminate the need for a separate LR scheduler by folding scheduling into the optimizer state

### Quantized Optimizers
- **4-bit Adam/AdamW**: Optimizer state stored in 4-bit quantized format; ~8x memory reduction for optimizer states
- **8-bit Adam/AdamW**: Optimizer state stored in 8-bit quantized format; ~4x memory reduction, negligible accuracy loss
- Dynamic quantization with per-block scaling factors
- Compatible with mixed-precision (FP16/BF16) training

### Learning Rate Schedulers
- **Linear**: Linear warmup and decay
- **Cosine + Restarts**: Cosine annealing with warm restarts (SGDR)
- **Polynomial**: Polynomial decay with configurable power
- **Constant**: Constant learning rate with optional warmup
- **Exponential**: Exponential decay
- **Step**: Step-wise learning rate reduction
- **One-Cycle**: SuperConvergence one-cycle policy (cycles LR and momentum inversely)

### Second-Order Methods
- **Sophia**: Scalable second-order optimizer using Hutchinson's Hessian diagonal estimator
- **Shampoo**: Matrix preconditioning with Kronecker-factored curvature
- Efficient Hessian-vector product computation without full Hessian materialization

### Distributed Optimization (ZeRO)
- **ZeRO Stage 1**: Optimizer state partitioning across data-parallel ranks
- **ZeRO Stage 2**: Optimizer state + gradient partitioning
- **ZeRO Stage 3**: Full parameter partitioning for maximum memory efficiency
- Efficient all-reduce and reduce-scatter communication primitives
- Mixed Precision support (FP16/BF16 parameters, FP32 optimizer state)

### Advanced Features
- **Gradient Clipping**: By global norm or per-parameter value
- **Weight Decay**: L2 regularization and decoupled weight decay
- **Momentum**: Classical and Nesterov momentum
- **Adaptive Learning Rates**: Per-parameter learning rate adaptation
- **Gradient Accumulation**: Simulate large effective batch sizes
- **Parameter Groups**: Layer-wise learning rates and per-group hyperparameters
- **Optimizer State Checkpointing**: Save/load full optimizer state for training resumption

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

### Schedule-Free Training

Eliminates the need for a separate LR scheduler:

```rust
use trustformers_optim::schedule_free::ScheduleFreeAdamW;

let optimizer = ScheduleFreeAdamW::new(
    model.parameters(),
    lr: 3e-4,
    betas: (0.9, 0.999),
    weight_decay: 0.01,
    warmup_steps: 1000,
)?;

// No separate scheduler needed — the optimizer handles it internally
for step in 0..num_steps {
    let loss = model.forward(&batch)?;
    let gradients = loss.backward()?;
    optimizer.step(&mut model.parameters(), &gradients)?;
    optimizer.zero_grad();
}
```

### 8-bit Quantized Optimizer

```rust
use trustformers_optim::quantized::Adam8bit;

// Drop-in replacement for Adam with ~4x reduced optimizer state memory
let optimizer = Adam8bit::new(
    model.parameters(),
    lr: 1e-4,
    betas: (0.9, 0.999),
    eps: 1e-8,
    weight_decay: 0.01,
    block_size: 2048, // Quantization block size
)?;
```

### Cosine Schedule with Warm Restarts

```rust
use trustformers_optim::schedulers::{CosineAnnealingWarmRestarts, WarmRestartConfig};

let scheduler = CosineAnnealingWarmRestarts::new(WarmRestartConfig {
    t_0: 1000,         // Steps in first cycle
    t_mult: 2,         // Cycle length multiplier
    eta_min: 1e-6,     // Minimum learning rate
    warmup_steps: 100,
})?;
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
│   ├── optimizers/           # Optimizer implementations
│   │   ├── sgd.rs           # SGD optimizer
│   │   ├── adam.rs          # Adam & AdamW
│   │   ├── lamb.rs          # LAMB optimizer
│   │   ├── adafactor.rs     # AdaFactor optimizer
│   │   ├── radam.rs         # Rectified Adam
│   │   ├── lion.rs          # Lion sign-based optimizer
│   │   ├── muon.rs          # Muon (momentum + orthogonalization)
│   │   ├── came.rs          # CAME optimizer
│   │   ├── micro_adam.rs    # MicroAdam
│   │   ├── bge_adam.rs      # BGE-Adam
│   │   ├── hn_adam.rs       # HN-Adam
│   │   └── ademamix.rs      # AdEMAMix
│   ├── schedule_free/        # Schedule-Free optimizer variants
│   ├── quantized/            # 4-bit and 8-bit quantized optimizers
│   ├── schedulers/           # Learning rate schedulers
│   ├── distributed/          # Distributed optimization
│   │   ├── zero.rs          # ZeRO stages 1/2/3
│   │   └── utils.rs         # Communication utilities
│   ├── second_order/         # Second-order methods (Sophia, Shampoo)
│   └── traits.rs            # Core optimizer traits
```

## Performance

### Memory Savings with ZeRO
| Model Size | Standard | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|------------|----------|---------|---------|---------|
| 1.5B params | 24 GB | 16 GB | 12 GB | 8 GB |
| 7B params | 112 GB | 75 GB | 56 GB | 28 GB |
| 175B params | 2.8 TB | 1.9 TB | 1.4 TB | 700 GB |

### Quantized Optimizer Memory Reduction
| Optimizer | FP32 State | 8-bit State | 4-bit State |
|-----------|------------|-------------|-------------|
| Adam (7B) | 112 GB | 28 GB | 14 GB |
| AdamW (7B) | 112 GB | 28 GB | 14 GB |

### Optimizer Performance
- **AdamW**: Industry standard for transformer training
- **LAMB**: Enables large batch training (up to 64K)
- **AdaFactor**: 75% memory reduction vs Adam
- **Lion**: Up to 2-3x memory savings over Adam (no variance state)
- **Schedule-Free**: Removes scheduler tuning; competitive final quality
- **8-bit Adam**: ~4x optimizer state reduction with minimal quality loss
- **ZeRO**: Near-linear scaling across multiple GPUs

## Best Practices

### Choosing an Optimizer
- **AdamW**: Default choice for most transformer models
- **Lion**: When GPU memory is constrained (no second moment)
- **Schedule-Free AdamW**: When eliminating LR scheduler complexity
- **LAMB**: When using very large batch sizes
- **AdaFactor**: Memory-constrained environments
- **8-bit Adam**: Large models where optimizer state dominates memory
- **Muon**: Experimental; strong results on vision transformers

### Learning Rate Schedules
- **Linear**: Standard for BERT-style pre-training
- **Cosine + Restarts**: Often better for long fine-tuning runs
- **One-Cycle**: Fast convergence for shorter schedules
- **Constant + Warmup**: Simple and effective
- **Schedule-Free**: Eliminates schedule search entirely

### Hyperparameters
```rust
// Recommended starting points
AdamW:            lr=5e-5,  weight_decay=0.01, warmup=10% of steps
Lion:             lr=1e-4,  weight_decay=0.1,  betas=(0.9, 0.99)
LAMB:             lr=2e-3,  weight_decay=0.01, warmup=10% of steps
AdaFactor:        lr=1e-3,  no weight_decay,   warmup=10% of steps
Schedule-Free AdamW: lr=3e-4, weight_decay=0.01, warmup_steps=1000
8-bit AdamW:      lr=5e-5,  weight_decay=0.01, block_size=2048
```

## Testing

- **583 unit tests** with 100% pass rate
- Convergence tests on toy problems for all optimizers
- Numerical stability tests (NaN, Inf, zero gradients)
- Distributed operation tests for ZeRO stages
- Memory usage profiling and quantization accuracy tests
- Schedule-Free convergence equivalence tests
- State save/load round-trip verification

## License

Apache-2.0
