# trustformers-optim TODO List

**Version:** 0.1.0 | **Status:** Stable | **Tests:** 589 | **SLoC:** ~45,000 | **Updated:** 2026-03-22

## Overview

The `trustformers-optim` crate provides comprehensive optimization algorithms and learning rate schedulers
for training transformer models in the TrustformeRS ecosystem. It implements 20+ state-of-the-art optimizers
including standard methods (SGD, Adam), modern variants (AdamW, LAMB, RAdam), cutting-edge research algorithms
(Lion, Muon, CAME, MicroAdam, BGE-Adam, HN-Adam, AdEMAMix), Schedule-Free variants, 4-bit/8-bit quantized
optimizers, ZeRO stages 1/2/3, and second-order methods.

**Key Responsibilities:**
- Optimization algorithms (SGD, Adam, AdamW, Lion, Muon, CAME, MicroAdam, BGE-Adam, HN-Adam, AdEMAMix, etc.)
- Schedule-Free optimizer variants (eliminate separate LR scheduler)
- 4-bit and 8-bit quantized optimizer states
- Learning rate schedulers (Linear, Cosine+Restarts, OneCycle, Polynomial, etc.)
- ZeRO distributed optimization stages 1, 2, and 3
- Second-order methods (Sophia, Shampoo)
- Gradient clipping and normalization
- Weight decay (L2 regularization and decoupled)
- Optimizer state management (save/load checkpoints)
- Parameter groups for layer-wise learning rates
- Mixed precision training support

---

## Current Status

### Implementation Status
- [x] **PRODUCTION-READY** - All major optimizers implemented and tested
- [x] **COMPREHENSIVE TEST COVERAGE** - 583 tests with 100% pass rate
- [x] **CUTTING-EDGE ALGORITHMS** - Latest research optimizers (2023-2025)
- [x] **ZERO COMPILATION ERRORS** - Clean compilation across all platforms
- [x] **MEMORY EFFICIENT** - Quantized optimizers, ZeRO stages, and sign-based methods
- [x] **SCHEDULE-FREE** - Schedule-Free Adam and SGD variants implemented
- [x] **QUANTIZED OPTIMIZERS** - 4-bit and 8-bit quantized optimizer states
- [x] **ZERO STAGES 1/2/3** - Full distributed optimizer state partitioning

### Test Metrics
- **Test Count:** 583 unit tests
- **Pass Rate:** 100%
- **Coverage:** Optimizer convergence, scheduler validation, gradient clipping, state save/load, quantization accuracy, ZeRO round-trip, Schedule-Free equivalence
- **Numerical Stability:** Extensive testing with edge cases (NaN, Inf, zero gradients)

---

## Completed Optimizer Implementations

### Standard Optimizers

#### SGD (Stochastic Gradient Descent)

**Classic first-order optimization**

- [x] **Algorithm**
  - Update: `θ ← θ - lr * ∇L(θ)`
  - With momentum: `v ← β * v + ∇L(θ), θ ← θ - lr * v`
  - Momentum coefficient β (typically 0.9)

- [x] **Features**
  - Momentum support for acceleration
  - Nesterov momentum option
  - Weight decay (L2 regularization)
  - Dampening for momentum

- [x] **Use Cases**
  - Simple baseline
  - Works well with large batch sizes
  - Computer vision tasks

**Example:**
```rust
use trustformers_optim::SGD;

let optimizer = SGD::new(
    model.parameters(),
    lr: 0.01,
    momentum: 0.9,
    weight_decay: 1e-4,
)?;
```

---

#### Adam (Adaptive Moment Estimation)

**Adaptive learning rate optimizer with momentum**

- [x] **Algorithm**
  - First moment: `m ← β1 * m + (1 - β1) * ∇L`
  - Second moment: `v ← β2 * v + (1 - β2) * ∇L²`
  - Bias correction: `m̂ ← m / (1 - β1^t), v̂ ← v / (1 - β2^t)`
  - Update: `θ ← θ - lr * m̂ / (√v̂ + ε)`

- [x] **Features**
  - Per-parameter adaptive learning rates
  - Momentum on gradients (first moment)
  - Momentum on squared gradients (second moment)
  - Bias correction for initial timesteps

**Example:**
```rust
use trustformers_optim::Adam;

let optimizer = Adam::new(
    model.parameters(),
    lr: 1e-3,
    betas: (0.9, 0.999),
    eps: 1e-8,
    weight_decay: 0.0,
)?;
```

---

#### AdamW (Adam with Decoupled Weight Decay)

**Adam with proper weight decay — recommended for transformers**

- [x] **Algorithm**
  - Same as Adam for moment estimates
  - Decoupled weight decay: `θ ← θ - lr * λ * θ` (applied after Adam update)

- [x] **Features**
  - Proper weight decay (not L2 regularization)
  - Better generalization than Adam
  - Default choice for transformer training

**Example:**
```rust
use trustformers_optim::AdamW;

let optimizer = AdamW::new(
    model.parameters(),
    lr: 1e-4,
    betas: (0.9, 0.999),
    eps: 1e-8,
    weight_decay: 0.01,
)?;
```

---

#### RAdam (Rectified Adam)

**Adam with automatic variance warmup**

- [x] **Algorithm**
  - Rectifies variance estimate in early training steps
  - Computes maximum length of approximated SMA
  - Falls back to SGD update when variance not tractable

- [x] **Features**
  - Automatic warmup (no manual warmup schedule needed)
  - More stable than Adam in early training
  - Better convergence on some tasks

**Example:**
```rust
use trustformers_optim::RAdam;

let optimizer = RAdam::new(
    model.parameters(),
    lr: 1e-3,
    betas: (0.9, 0.999),
    eps: 1e-8,
    weight_decay: 0.0,
)?;
```

---

#### LAMB (Layer-wise Adaptive Moments)

**Optimizer for very large batch training**

- [x] **Algorithm**
  - Compute Adam-like update: `u = m / (√v + ε)`
  - Layer-wise trust ratio: `r = ||θ|| / ||u||`
  - Update: `θ ← θ - lr * r * u`

- [x] **Features**
  - Enables large batch training (32k+)
  - Layer-wise learning rate adaptation
  - Maintains accuracy with large batches

---

#### AdaFactor

**Memory-efficient optimizer**

- [x] **Features**
  - Factored second moment estimate
  - ~75% memory reduction vs Adam
  - Adaptive learning rates without full second moment

---

#### AdaGrad / RMSProp

- [x] AdaGrad: per-parameter adaptive rates based on accumulated squared gradients
- [x] RMSProp: exponential moving average variant that fixes AdaGrad's aggressive decay

---

### Cutting-Edge Research Optimizers

#### Lion (Evolved Sign Momentum)

**Sign-based optimizer from evolutionary search**

- [x] **Algorithm**
  - Update: `θ ← θ - lr * sign(β1 * m + (1 - β1) * ∇L)`
  - Momentum: `m ← β2 * m + (1 - β2) * ∇L`

- [x] **Features**
  - Memory-efficient (only stores first moment, no variance)
  - Discovered via evolutionary algorithm
  - Competitive or better than AdamW

**Example:**
```rust
use trustformers_optim::Lion;

let optimizer = Lion::new(
    model.parameters(),
    lr: 1e-4,
    betas: (0.9, 0.99),
    weight_decay: 0.1,
)?;
```

---

#### Muon (Momentum + Orthogonalization)

**Nesterov momentum with Gram-Schmidt orthogonalization**

- [x] **Algorithm**
  - Applies Nesterov momentum update
  - Orthogonalizes the update matrix via Newton-Schulz iteration
  - Improves generalization by encouraging orthogonal weight updates

- [x] **Features**
  - Reduces gradient interference between filters
  - Strong results on vision transformers
  - Per-layer orthogonal projection step

**Example:**
```rust
use trustformers_optim::Muon;

let optimizer = Muon::new(
    model.parameters(),
    lr: 2e-4,
    momentum: 0.95,
    nesterov: true,
    orthogonalize: true,
)?;
```

---

#### CAME (Confidence-guided Adaptive Memory-Efficient)

**Confidence-guided second moment with AdaFactor-like memory footprint**

- [x] **Algorithm**
  - Uses confidence scores to weight second moment updates
  - Factored representation of the second moment matrix
  - Confidence metric derived from gradient consistency

- [x] **Features**
  - Memory footprint similar to AdaFactor
  - Better convergence than AdaFactor on large models
  - Confidence-guided adaptation avoids noisy gradient updates

**Example:**
```rust
use trustformers_optim::Came;

let optimizer = Came::new(
    model.parameters(),
    lr: 1e-3,
    betas: (0.9, 0.999, 0.9999),
    eps: (1e-30, 1e-16),
    weight_decay: 0.0,
)?;
```

---

#### MicroAdam

**Gradient compression with micro-batch accumulation**

- [x] **Algorithm**
  - Compresses gradients with top-k sparsity or quantization
  - Accumulates error feedback across micro-batches
  - Updates optimizer state with decompressed gradients

- [x] **Features**
  - Extremely memory-constrained training
  - Gradient sparsification with error correction
  - Compatible with ZeRO stage 1

---

#### BGE-Adam (Bias-corrected Gradient Estimation Adam)

**Improved bias correction for large-batch regimes**

- [x] **Algorithm**
  - Enhanced bias correction that accounts for gradient variance at large batch sizes
  - Corrected moment estimates remain accurate even at step 1

- [x] **Features**
  - Better large-batch accuracy than standard Adam
  - Drop-in replacement for Adam in distributed settings

---

#### HN-Adam (Hyperbolic Nesterov Adam)

**Nesterov correction in hyperbolic space**

- [x] **Algorithm**
  - Applies Nesterov lookahead in the hyperbolic tangent-mapped parameter space
  - Hyperbolic projection prevents update divergence in deep models

- [x] **Features**
  - Improved stability in very deep transformer models
  - Compatible with standard AdamW hyperparameters

---

#### AdEMAMix (Adaptive EMA Mixture)

**Mixes fast and slow EMA of gradients for long-range memory**

- [x] **Algorithm**
  - Maintains two EMA buffers: fast (β1) and slow (β3)
  - Linearly interpolates between fast and slow EMA: `m_mix = α * m_fast + (1-α) * m_slow`
  - Denominator uses standard second moment estimate

- [x] **Features**
  - Captures both short-range and long-range gradient trends
  - Improved convergence on long training runs
  - Extra α hyperparameter for mixture weight

**Example:**
```rust
use trustformers_optim::AdEMAMix;

let optimizer = AdEMAMix::new(
    model.parameters(),
    lr: 1e-4,
    betas: (0.9, 0.999, 0.9999), // (β1, β2, β3_slow)
    alpha: 5.0,                   // mixture weight
    weight_decay: 0.01,
)?;
```

---

### Schedule-Free Optimizer Variants

**Eliminate the need for a separate learning rate scheduler**

- [x] **Schedule-Free AdamW**
  - Folds cosine-like scheduling into primal-dual iterate averaging
  - No separate scheduler object required
  - Competitive final loss with properly-tuned scheduled training

- [x] **Schedule-Free SGD**
  - Same approach applied to SGD with momentum
  - Useful for vision tasks without scheduler search

**Example:**
```rust
use trustformers_optim::schedule_free::ScheduleFreeAdamW;

let optimizer = ScheduleFreeAdamW::new(
    model.parameters(),
    lr: 3e-4,
    betas: (0.9, 0.999),
    weight_decay: 0.01,
    warmup_steps: 1000,
)?;

// No separate scheduler needed
for step in 0..num_steps {
    let loss = model.forward(&batch)?;
    let gradients = loss.backward()?;
    optimizer.step(&mut model.parameters(), &gradients)?;
    optimizer.zero_grad();
}
```

---

### Quantized Optimizers

**Reduce optimizer state memory by 4-8x**

#### 8-bit Adam/AdamW

- [x] **Features**
  - Optimizer states stored in 8-bit quantized format
  - Dynamic per-block scaling factors (block_size=2048 default)
  - ~4x memory reduction for optimizer states
  - Negligible accuracy loss on standard benchmarks

**Example:**
```rust
use trustformers_optim::quantized::Adam8bit;

let optimizer = Adam8bit::new(
    model.parameters(),
    lr: 1e-4,
    betas: (0.9, 0.999),
    eps: 1e-8,
    weight_decay: 0.01,
    block_size: 2048,
)?;
```

---

#### 4-bit Adam/AdamW

- [x] **Features**
  - Optimizer states stored in 4-bit quantized format
  - NF4 (NormalFloat4) or FP4 quantization schemes
  - ~8x memory reduction for optimizer states
  - Recommended for models >= 7B parameters

**Example:**
```rust
use trustformers_optim::quantized::{Adam4bit, QuantScheme};

let optimizer = Adam4bit::new(
    model.parameters(),
    lr: 1e-4,
    betas: (0.9, 0.999),
    eps: 1e-8,
    weight_decay: 0.01,
    quant_scheme: QuantScheme::NF4,
    block_size: 64,
)?;
```

---

### Learning Rate Schedulers

#### Linear Scheduler

- [x] Linear warmup + linear decay
- [x] Standard for BERT-style pretraining

**Example:**
```rust
use trustformers_optim::schedulers::LinearScheduler;

let scheduler = LinearScheduler::new(
    optimizer,
    warmup_steps: 10000,
    total_steps: 100000,
)?;
```

---

#### Cosine Annealing with Warm Restarts (SGDR)

- [x] Cosine decay with periodic restarts
- [x] Cycle length multiplier for progressive restart spacing
- [x] Minimum learning rate floor

**Example:**
```rust
use trustformers_optim::schedulers::CosineAnnealingWarmRestarts;

let scheduler = CosineAnnealingWarmRestarts::new(WarmRestartConfig {
    t_0: 1000,
    t_mult: 2,
    eta_min: 1e-6,
    warmup_steps: 100,
})?;
```

---

#### One-Cycle Policy

- [x] Cycles learning rate from low → max → very low in single training run
- [x] Inversely cycles momentum
- [x] SuperConvergence: enables faster convergence with SGD

**Example:**
```rust
use trustformers_optim::schedulers::OneCycleLR;

let scheduler = OneCycleLR::new(OneCycleConfig {
    max_lr: 1e-3,
    total_steps: 10000,
    pct_start: 0.3,      // 30% of steps for increasing phase
    div_factor: 25.0,    // initial_lr = max_lr / div_factor
    final_div_factor: 1e4,
})?;
```

---

#### Polynomial Decay

- [x] `lr(t) = lr * (1 - t / T)^power`
- [x] Power typically 1.0 (linear) or 2.0 (quadratic)

---

#### Step Decay / Exponential Decay / Constant

- [x] Step: multiply by gamma every step_size steps
- [x] Exponential: continuous `lr * gamma^t` decay
- [x] Constant: fixed LR with optional warmup

---

### Second-Order Methods

#### Sophia

**Scalable second-order optimizer using Hutchinson estimator**

- [x] **Algorithm**
  - Hutchinson's estimator for Hessian diagonal
  - Pre-conditioned gradient descent: `θ ← θ - lr * ∇L / (h + ε)`
  - Hessian estimate updated every k steps (typically 10-100)

- [x] **Features**
  - Second-order information without full Hessian
  - Scalable to billion-parameter models
  - Better sample efficiency than first-order methods

---

#### Shampoo

**Matrix preconditioning with Kronecker-factored curvature**

- [x] **Features**
  - Left and right Kronecker factor preconditioning
  - Efficient matrix root computation via coupled Newton iterations
  - Near-second-order convergence with manageable overhead

---

### Advanced Features

#### Gradient Clipping

- [x] **By Global Norm**: `∇L ← ∇L * max_norm / max(||∇L||, max_norm)`
- [x] **By Value**: element-wise clip to `[-max_val, max_val]`

**Example:**
```rust
optimizer.clip_grad_norm(max_norm: 1.0)?;
optimizer.clip_grad_value(max_value: 0.5)?;
```

---

#### Weight Decay

- [x] **L2 Regularization**: gradient includes regularization term
- [x] **Decoupled Weight Decay**: `θ ← θ - lr * λ * θ` (correct for adaptive methods)

---

#### Gradient Accumulation

- [x] Accumulate gradients over N steps before optimizer update
- [x] Effective batch size = `batch_size * accumulation_steps`

---

#### Mixed Precision Training

- [x] FP16/BF16 forward/backward, FP32 optimizer state
- [x] Dynamic loss scaling to prevent FP16 underflow
- [x] Automatic Mixed Precision (AMP) integration

---

#### Optimizer State Management

- [x] **Save State**: `optimizer.save_state("checkpoint.pt")?`
- [x] **Load State**: `optimizer.load_state("checkpoint.pt")?`
- [x] Includes all buffers: momentum, variance, step count, quantization scales

---

#### Parameter Groups

- [x] Per-group learning rates, weight decay, and hyperparameters
- [x] Layer-wise learning rate decay for fine-tuning

---

## Distributed Optimization (ZeRO)

#### ZeRO Stage 1

- [x] Optimizer state partitioned across data-parallel ranks
- [x] Reduces optimizer memory by `world_size` factor

#### ZeRO Stage 2

- [x] Optimizer state + gradient partitioning
- [x] Reduce-scatter for gradient aggregation

#### ZeRO Stage 3

- [x] Full parameter partitioning for maximum memory efficiency
- [x] All-gather before forward pass, re-partition after backward
- [x] Optional CPU offload of optimizer states

**Example:**
```rust
use trustformers_optim::distributed::{ZeroOptimizer, ZeroConfig, ZeroStage};

let zero_config = ZeroConfig {
    stage: ZeroStage::Three,
    partition_gradients: true,
    contiguous_gradients: true,
    overlap_comm: true,
    reduce_scatter: true,
    cpu_offload: false,
};

let base_optimizer = AdamW::new(adam_config)?;
let optimizer = ZeroOptimizer::new(
    base_optimizer,
    model,
    zero_config,
    process_group,
)?;
```

---

## Testing

### Test Coverage

- [x] **583 Unit Tests** - 100% pass rate
- [x] **Optimizer Convergence** - Verify convergence on toy problems
- [x] **Scheduler Validation** - Check learning rate schedules
- [x] **Gradient Clipping** - Verify clipping correctness
- [x] **State Save/Load** - Round-trip state verification
- [x] **Memory Leak Detection** - No memory leaks
- [x] **Numerical Stability** - Edge cases (zero gradients, NaN, Inf)
- [x] **Quantization Accuracy** - 8-bit and 4-bit optimizer convergence tests
- [x] **Schedule-Free Equivalence** - Verify Schedule-Free matches scheduled training
- [x] **ZeRO Round-Trip** - State consistency across ZeRO stage transitions

### Test Categories

1. **Correctness Tests** - Optimizer updates match reference implementations
2. **Convergence Tests** - Optimizers converge on convex problems
3. **State Tests** - Save/load produces identical state
4. **Quantization Tests** - Quantized optimizers achieve acceptable accuracy
5. **Distributed Tests** - ZeRO stages maintain training equivalence
6. **Edge Case Tests** - Zero gradients, NaN/Inf, empty parameter groups

---

## Known Limitations

- Shampoo matrix root computation is expensive for very large layers (>4096 dims)
- ZeRO stage 3 with CPU offload adds host-device transfer overhead
- 4-bit quantized optimizer may diverge on tasks with very noisy gradients

---

## Future Enhancements

### High Priority
- [ ] Additional emerging optimizers as they appear (2025+)
  - **Refinement needed:** List specific optimizers: SOAP, Muon, Adam-mini, Grokfast?
- [ ] Enhanced distributed optimizer state management with async overlap
  - **Refinement needed:** async overlap strategy? Overlap with which compute phases?
- [ ] Automatic hyperparameter tuning integration
  - **Refinement needed:** which tuning framework? Optuna, Ray Tune, custom Bayesian search?

### Performance
- [ ] Fused quantized optimizer kernels for GPU
- [x] **Lazy optimizer state allocation** — `LazyAdam` allocates moment buffers only when first gradient is seen (`lazy_state.rs`)
- [ ] Better async communication overlap for ZeRO stage 3

### Features
- [x] **Cyclic LR with decay** — `CyclicLrScheduler` (Triangular/Triangular2/ExpRange) + `OneCycleLrScheduler` (`cyclic_decay.rs`)
- [x] **Automatic LR Finder** — `LrFinder` + `LrFinderConfig` + `LrFinderResult` + `find_optimal_lr` (`lr_finder.rs`)
- [ ] Optimizer surgery (change optimizer type mid-training)
- [ ] Per-layer quantization bit-width selection

---

## Development Guidelines

### Code Standards
- **Use trustformers-core abstractions only** (no external deps directly)
- **File size limit:** <2000 lines per file
- **Error handling:** Use `Result<T, TrustformersError>` (no unwrap)
- **Testing:** Convergence tests required for new optimizers
- **Naming:** snake_case for all identifiers

### Adding a New Optimizer

**Checklist:**

1. **Implement Optimizer Trait**
   ```rust
   impl StatefulOptimizer for NewOptimizer {
       fn step(&mut self) -> Result<()>;
       fn zero_grad(&mut self) -> Result<()>;
       fn state_dict(&self) -> StateDict;
       fn load_state_dict(&mut self, state: StateDict) -> Result<()>;
   }
   ```

2. **Add State Buffers** - Momentum buffers, variance buffers, per-parameter state

3. **Implement Update Rule** - Follow algorithm from paper, handle edge cases

4. **Add Tests** - Convergence test, state save/load, reference comparison

5. **Document** - Algorithm, hyperparameter recommendations, use cases, example

### Build & Test Commands

```bash
# Run all tests
cargo nextest run -p trustformers-optim --all-features

# Test specific optimizer
cargo test -p trustformers-optim test_adam

# Benchmark
cargo bench -p trustformers-optim

# Check compilation
cargo check -p trustformers-optim --all-features
```

---

**Last Updated:** 2026-03-21 - 0.1.0 Stable Release
**Status:** Production-ready optimization
**Tests:** 583 tests, 100% pass rate
**Optimizers:** SGD, Adam, AdamW, RAdam, LAMB, AdaFactor, Lion, Muon, CAME, MicroAdam, BGE-Adam, HN-Adam, AdEMAMix, Schedule-Free variants, Sophia, Shampoo, and more
**Quantized:** 4-bit and 8-bit Adam/AdamW
**Distributed:** ZeRO stages 1, 2, and 3
