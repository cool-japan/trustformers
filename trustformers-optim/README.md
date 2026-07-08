# trustformers-optim

**Version:** 0.2.0 | **Status:** Stable | **Tests:** ~960 | **SLoC:** 52,189 | **Public API:** ~1,925 items | **Updated:** 2026-07-02

Comprehensive optimization algorithms, learning rate schedulers, and distributed/advanced training
infrastructure for training transformer models in the TrustformeRS ecosystem.

## Overview

`trustformers-optim` is one of the largest crates in TrustformeRS: over 100 modules (~52,200 lines of
source) covering everything from textbook first-order optimizers to 2024/2025 research algorithms,
4-bit/8-bit quantized optimizer states, ZeRO/FSDP-style distributed training, federated and continual
learning, hardware-targeted variants (GPU/TPU/edge/mobile), and cross-framework (PyTorch/JAX/TensorFlow)
compatibility layers.

All functionality ships under the crate's single default feature set (no optional Cargo features are
defined — everything described below is always compiled in).

## Features

### Standard Optimizers
- **SGD** (`SGD`): momentum, Nesterov momentum, weight decay
- **Adam** / **AdamW** (`Adam`, `AdamW`): adaptive moment estimation, with AdamW's decoupled weight decay
- **RAdam**, **NAdam**, **AdaBelief** (`RAdam`, `NAdam`, `AdaBelief`): Adam variants with rectified variance,
  Nesterov-style lookahead, and belief-based second moments respectively
- **LAMB** (`LAMB`): layer-wise adaptive moments for large-batch training
- **AdaFactor** (`AdaFactor`): factored second-moment estimator for memory-efficient training
- **AdaFisher** (`AdaFisher`): block-diagonal Fisher-information preconditioning (ICLR 2025)
- **AdaMaxPlus**, **Adan** (`AdaMaxPlus`, `Adan`): infinity-norm and Nesterov-accelerated adaptive variants
- **Ranger**, **AdaBound**, **AMSBound** (`Ranger`, `AdaBound`, `AMSBound`): RAdam+Lookahead composite and
  bounded adaptive-LR variants

### Cutting-Edge Research Optimizers
- **Lion** (`Lion`): sign-based updates discovered via evolutionary search
- **Muon** (`Muon`): Nesterov momentum + Newton-Schulz orthogonalization of 2D updates
- **CAME** (`CAME` / `CameOptimizer`): confidence-guided, AdaFactor-sized second moment
- **MicroAdam** (`MicroAdam`): gradient-compressed Adam with error feedback
- **BGE-Adam** / **OptimizedBGEAdam** (`BGEAdam`, `OptimizedBGEAdam`): entropy-weighted bias correction,
  with a 3–5x faster vectorized reimplementation
- **HN-Adam** (`HNAdam`): hybrid-norm adaptive step size
- **AdEMAMix** (`AdEMAMix`): dual fast/slow EMA mixture (Apple/EPFL, 2024)
- **Prodigy**, **NovoGrad**, **LancBiO**, **AMacP**, **EVA** — additional 2023–2025 adaptive/variance-reduced
  optimizers
- **Schedule-Free Adam** / **Schedule-Free SGD** (`ScheduleFreeAdam`, `ScheduleFreeSGD`): fold scheduling
  into primal-dual iterate averaging, eliminating the separate LR-scheduler object
- **Research-preview, simplified reference implementations**: `GENIE`, `LoRARITE`, `SOFO` — functional,
  tested momentum-style updates for very recent papers (GENIE, LoRA-RITE, SOFO) whose modules are
  explicitly documented as simplified pending full tensor-op parity with the original papers

### Quantized Optimizers
- **Adam8bit** / **AdamW8bit** (`quantized::Adam8bit`, `quantized::AdamW8bit`): 8-bit optimizer state,
  ~4x memory reduction
- **Adam4bit** (`quantized_advanced::Adam4bit`): 4-bit optimizer state via configurable
  `QuantizationMethod` (e.g. NF4), ~8x memory reduction
- **Per-layer bit-width selection** (`per_layer_quant`): assigns different quantization bit-widths
  (`BitWidth::{Int2,Int4,Int8,Fp16,Fp32}`) per layer based on sensitivity/memory-budget analysis

### Learning Rate Schedulers
- **Linear**, **Polynomial**, **Step**, **Exponential**, **Constant+Warmup** (`LinearScheduler`,
  `PolynomialScheduler`, `StepScheduler`, `ExponentialScheduler`, `ConstantWithWarmupScheduler`)
- **Cosine** and **Cosine with Warm Restarts** (`CosineScheduler`, `CosineWithRestartsScheduler`, SGDR-style)
- **One-Cycle** (`OneCycleScheduler`) and a dedicated cyclic-decay variant
  (`cyclic_decay::{CyclicLrScheduler, OneCycleLrScheduler}`) with `Triangular` / `Triangular2` / `ExpRange`
  amplitude modes
- **Adaptive / Composite / Cyclical / Dynamic / Phase-based / Task-specific** schedulers
  (`AdaptiveScheduler`, `CompositeScheduler`, `CyclicalScheduler`, `DynamicScheduler`,
  `PhaseBasedScheduler`, `TaskSpecificScheduler`)
- **Automatic LR Finder** (`LrFinder`, `find_optimal_lr`): sweeps learning rate to locate a good starting
  point before training

### Second-Order Methods
- **Sophia** (`Sophia` / `SophiaOptimizer`): Hutchinson's-estimator Hessian diagonal preconditioning
- **L-BFGS**, **Newton-CG** (`LBFGS`, `NewtonCG`): classic quasi-Newton and Newton-CG line-search methods
- **Self-Scaled BFGS / Broyden** (`SSBFGS`, `SSBroyden`, 2025): quasi-Newton methods with presets for
  physics-informed neural networks (PINNs) and non-convex problems

### Distributed & Scaled Training
- **ZeRO stages 1/2/3** (`ZeROOptimizer`, `ZeROConfig`, `ZeROStage`): optimizer-state, gradient, and full
  parameter partitioning with configurable bucket size, prefetch depth, and optional gradient compression
- **FSDP-style sharding** (`fsdp` module: `FsdpConfig`, `ShardingStrategy`, `FsdpUnit`, `FsdpState`) —
  present in the source tree, not yet re-exported at the crate root (use `trustformers_optim::fsdp::*`)
- **Multi-node training** (`MultiNodeTrainer`, `MultiNodeConfig`)
- **Enhanced distributed trainer** (`EnhancedDistributedTrainer`, `DistributedConfig`): NCCL-style
  communication, gradient compression (`CompressionType`, e.g. PowerSGD), dynamic batching, fault
  tolerance
- **Advanced distributed features** (`AutoScaler`, `PerformanceMLOptimizer`, `SmartCheckpointManager`):
  auto-scaling, ML-based performance tuning, differential checkpointing
- **Asynchronous / staleness-tolerant training** (`Hogwild`, `ElasticAveraging`, `ParameterServer`,
  `AsyncSGD`, `DelayedGradient`): lock-free and delay-compensated parameter updates
- **Hierarchical gradient aggregation** (`HierarchicalAggregator`: ring / tree / butterfly topologies)
- **Deep distributed QP** (`DeepDistributedQP`)

### Federated & Continual Learning
- **FedAvg**, **FedProx** (`FedAvg`, `FedProx`): federated averaging and proximal-term federated optimization
- **Differential privacy** / **secure aggregation** (`DifferentialPrivacy`, `SecureAggregation`)
- **EWC**, **PackNet**, **memory replay** (`EWC`, `PackNet`, `MemoryReplay`): catastrophic-forgetting
  mitigation for continual learning

### Hardware-Aware & Performance
- **GPU / TPU / edge / mobile variants** (`GPUAdam`, `TPUOptimizer`, `EdgeOptimizer`, `MobileOptimizer`)
- **Kernel fusion** (`kernel_fusion`: `KernelFusedAdam`, tensor-core-aware fused Adam kernels)
- **SIMD optimizations**, **cache-friendly layouts**, **memory layout optimization**
  (`SIMDOptimizer`, `CacheFriendlyAdam`, `LayoutOptimizedAdam`, `AlignedAllocator`)
- **CPU offload** and **lazy state allocation** (`CPUOffloadedOptimizer`, `LazyAdam` — moment buffers
  allocate only once the first gradient is seen)

### Cross-Framework Compatibility
- **PyTorch-style** (`PyTorchAdam`, `PyTorchAdamW`, `PyTorchSGD`, `PyTorchLRScheduler`)
- **JAX/Optax-style** (`JAXAdam`, `JAXAdamW`, `JAXSGD`, `JAXGradientTransformation`, `JAXChain`)
- **TensorFlow-style** (`TensorFlowAdam`, `TensorFlowAdamW`, `TensorFlowCosineDecay`)
- **Universal converter** (`CrossFrameworkConverter`, `UniversalOptimizerConfig`) to translate
  configurations between frameworks

### Tooling
- **Hyperparameter tuning** (`BayesianOptimizer`, `MultiObjectiveOptimizer`, `HyperparameterTuner`)
- **Monitoring & recommendation** (`OptimizerMonitor`, `OptimizerSelector`, `ConvergenceIndicators`)
- **Performance validation & benchmarking harness** (`PerformanceValidator`, `advanced_benchmarking`*)
- **ONNX export** (`ONNXOptimizerExporter`)
- **Optimizer surgery** (`optimizer_surgery`): migrate momentum/variance state between Adam, AdamW, SGD,
  and Lion mid-training

### Advanced Features
- **Gradient clipping/scaling utilities** (`GradientProcessor::{clip_by_norm, clip_by_value,
  scale_gradient, is_finite}`) plus a higher-level `GradientProcessedOptimizer` wrapper with adaptive
  clipping, noise injection, and Hessian-based preconditioning
- **Weight decay**: L2 regularization and AdamW-style decoupling (`WeightDecayMode`)
- **Parameter groups & checkpointing**: per-group hyperparameters; `StatefulOptimizer::state_dict()` /
  `load_state_dict()` for full state save/restore
- **Sparse optimizers** (`SparseAdam`, `SparseSGD`) and **LoRA-aware optimizers**
  (`LoRAOptimizer`, `LoRAAdapter`, `create_lora_adam`/`create_lora_adamw`/`create_lora_sgd`)
- **Task-specific presets** (`BERTOptimizer`, `GANOptimizer`, `RLOptimizer`)

\* `advanced_benchmarking` is present in `src/` but is not currently wired into `lib.rs` (see
[Known Limitations](#known-limitations)).

## Usage Example

### Basic Optimizer Usage

```rust,no_run
use trustformers_optim::AdamW;
use trustformers_core::traits::Optimizer;

let mut optimizer = AdamW::new(
    5e-5,           // learning_rate
    (0.9, 0.999),   // (beta1, beta2)
    1e-8,           // epsilon
    0.01,           // weight_decay
);

# fn train_step(_optimizer: &mut AdamW) -> anyhow::Result<()> {
// Training loop (per-parameter update, matching the `Optimizer` trait from trustformers-core)
// for (parameter, grad) in model.parameters_mut().zip(gradients.iter()) {
//     optimizer.update(parameter, grad)?;
// }
// optimizer.step();
// optimizer.zero_grad();
# Ok(())
# }
```

### Schedule-Free Training

Eliminates the need for a separate LR scheduler:

```rust,no_run
use trustformers_optim::ScheduleFreeAdam;

// Preset tuned for language-model training
let optimizer = ScheduleFreeAdam::for_language_models();

// Or fully custom: (learning_rate, beta1, beta2, epsilon, weight_decay)
let optimizer = ScheduleFreeAdam::new(0.5, 0.9, 0.95, 1e-8, 0.1);
```

### 8-bit and 4-bit Quantized Optimizers

```rust,no_run
use trustformers_optim::{Adam8bit, Adam4bit};

// ~4x reduced optimizer-state memory
let optimizer_8bit = Adam8bit::new(1e-4 /* learning_rate */);

// ~8x reduced optimizer-state memory: (lr, beta1, beta2, eps, weight_decay)
let optimizer_4bit = Adam4bit::new(1e-4, 0.9, 0.999, 1e-8, 0.01);
```

### Cosine Schedule with Warm Restarts

```rust,no_run
use trustformers_optim::CosineWithRestartsScheduler;

// (base_lr, min_lr, t_0 = steps in first cycle, t_mult = cycle-length multiplier)
let scheduler = CosineWithRestartsScheduler::new(1e-3, 1e-6, 1000, 2.0);
```

### Gradient Clipping

```rust
use trustformers_optim::GradientProcessor;

let mut grad = vec![3.0_f32, 4.0, 0.0];
GradientProcessor::clip_by_norm(&mut grad, 1.0);      // clip by global L2 norm
GradientProcessor::clip_by_value(&mut grad, -0.5, 0.5); // element-wise clip
assert!(GradientProcessor::is_finite(&grad));
```

### ZeRO Optimization

```rust,ignore
// Requires a distributed `ModelParallelContext`; illustrates configuration only.
use std::sync::Arc;
use trustformers_optim::{AdamW, ZeROConfig, ZeROOptimizer, ZeROStage};

let zero_config = ZeROConfig {
    stage: ZeROStage::Stage3,
    bucket_size_mb: 25,
    overlap_comm: true,
    reduce_bucket_size: 500_000_000,
    prefetch_depth: 2,
    max_memory_usage_mb: 1024,
    gradient_compression: false,
    pin_memory: true,
};

let base_optimizer = AdamW::new(1e-4, (0.9, 0.999), 1e-8, 0.01);
let mut optimizer = ZeROOptimizer::new(base_optimizer, zero_config, mp_context)?;
optimizer.register_parameters(parameters)?;
```

## Architecture

### Trait hierarchy

`trustformers-optim` layers its trait hierarchy on top of the base `Optimizer` trait from
`trustformers-core`:

```text
Optimizer (trustformers-core: update / zero_grad / step / get_lr / set_lr)
    │
    ├── StatefulOptimizer      (+ config/state access, state_dict/load_state_dict, memory_usage)
    │   ├── MomentumOptimizer
    │   │   ├── AdaptiveMomentumOptimizer   (Adam, AdamW, etc.)
    │   │   └── ClassicalMomentumOptimizer  (SGD with momentum)
    │   └── SecondOrderOptimizer            (L-BFGS, Newton-CG, etc.)
    │
    ├── DistributedOptimizer
    │   ├── GradientCompressionOptimizer
    │   ├── FederatedOptimizer
    │   └── AsyncOptimizer
    │
    ├── HardwareOptimizer
    │   ├── SIMDOptimizer (concrete type, not this trait's name)
    │   ├── GPUOptimizer
    │   └── EdgeOptimizer-style targets
    │
    └── MetaOptimizer
        ├── LookaheadOptimizer
        ├── ScheduledOptimizer
        └── CompositeOptimizer
```

### Source layout

The crate is *not* organized into `optimizers/`/`schedulers/` subdirectories — nearly all ~100 modules
are flat files directly under `src/`, grouped here by role (file names are real, not illustrative):

```
trustformers-optim/src/
├── lib.rs                                  # crate root; re-exports the public API
├── traits.rs                               # extended optimizer trait hierarchy (see above)
├── common.rs                               # OptimizerState, GradientProcessor, WeightDecayMode
├── optimizer.rs                            # base OptimizerState plumbing
│
├── adam.rs, adam_v2.rs, sgd.rs, lamb.rs,    # standard optimizers
│   adafactor_new.rs, adafisher_simple.rs,
│   adamax_plus.rs, adan.rs, adaptive.rs
├── lion.rs, muon.rs, hn_adam.rs,            # cutting-edge research optimizers
│   ademamix.rs, microadam.rs, bge_adam.rs,
│   bge_adam_optimized.rs, prodigy.rs, novograd.rs,
│   lancbio.rs, amacp.rs, eva.rs,
│   advanced_2025_research.rs
├── genie_stub.rs, lora_rite_stub.rs,        # simplified reference implementations
│   sofo_stub.rs
├── schedule_free.rs                         # Schedule-Free Adam/SGD
├── quantized.rs, quantized_advanced.rs,     # 8-bit / 4-bit optimizer state
│   per_layer_quant.rs
├── scheduler.rs, cyclic_decay.rs,           # learning-rate schedulers
│   lr_finder.rs
├── second_order/ (lbfgs.rs, newton_cg.rs,   # second-order methods
│   self_scaled.rs), sophia/ (mod.rs, legacy.rs), came/ (mod.rs, legacy.rs)
├── zero/ (zero_optimizer.rs, zero_stage{1,2,3}.rs, # ZeRO distributed optimization
│   zero_stage3_overlap.rs, zero_utils.rs)
├── fsdp/ (mod.rs)                           # FSDP-style sharding
├── multinode/ (mod.rs)                      # multi-node coordination
├── enhanced_distributed_training.rs,        # distributed training infrastructure
│   advanced_distributed_features.rs,
│   hierarchical_aggregation.rs, deep_distributed_qp.rs,
│   async_optim.rs, compression.rs, cpu_offload.rs, parallel.rs
├── federated.rs, continual_learning.rs      # federated & continual learning
├── hardware_aware.rs, kernel_fusion.rs,     # hardware-aware & low-level performance
│   simd_optimizations.rs, cache_friendly.rs,
│   memory_layout.rs, lazy_state.rs, fusion.rs
├── pytorch_compat.rs, jax_compat.rs,        # cross-framework compatibility
│   tensorflow_compat.rs, cross_framework.rs
├── hyperparameter_tuning.rs, monitoring.rs, # tooling
│   performance_validation.rs, onnx_export.rs,
│   optimizer_surgery.rs
├── sparse.rs, lora.rs, lora_rite.rs,        # sparse / LoRA-aware / task-specific
│   task_specific.rs
├── convergence.rs, gradient_processing.rs,  # gradient/convergence utilities
│   quantum_inspired.rs, pde_aware.rs
└── tests.rs (+ per-module `#[cfg(test)]` files: adam_tests.rs, sgd_tests.rs,
    lookahead_tests.rs, pde_aware_tests.rs, hardware_aware_tests.rs)
```

## Performance

The tables below are standard ZeRO/quantization memory-reduction formulas (illustrative, not
measured on a specific model in this repository):

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

## Best Practices

### Choosing an Optimizer
- **AdamW**: default choice for most transformer models
- **Lion**: when GPU memory is constrained (no second-moment buffer)
- **Schedule-Free Adam**: when eliminating LR-scheduler complexity
- **LAMB**: when using very large batch sizes
- **AdaFactor**: memory-constrained environments
- **Adam8bit / Adam4bit**: large models where optimizer state dominates memory
- **Muon**: experimental; strong results reported on vision transformers and NanoGPT-style training

### Learning Rate Schedules
- **Linear**: standard for BERT-style pre-training
- **Cosine + Restarts**: often better for long fine-tuning runs
- **One-Cycle**: fast convergence for shorter schedules
- **Constant + Warmup**: simple and effective
- **Schedule-Free**: eliminates schedule search entirely

### Hyperparameters
```text
// Recommended starting points
AdamW:                lr=5e-5,  weight_decay=0.01, warmup=10% of steps
Lion:                  lr=1e-4,  weight_decay=0.1,  betas=(0.9, 0.99)
LAMB:                  lr=2e-3,  weight_decay=0.01, warmup=10% of steps
AdaFactor:             lr=1e-3,  no weight_decay,   warmup=10% of steps
Schedule-Free Adam:    lr=0.5,   weight_decay=0.1,  warmup_steps=2000 (for_language_models preset)
Adam8bit:              lr=1e-4  (block-wise quantized internally)
```

## Examples

The `examples/` directory contains 24 runnable programs, including:
`basic_benchmark`, `basic_validation`, `advanced_benchmark_analysis`, `advanced_performance_profiler`,
`auto_optimizer_tuning`, `bge_adam_optimization_benchmark`, `cutting_edge_2025_optimizers_demo`,
`cross_framework_compatibility_test`, `distributed_training_test`,
`comprehensive_distributed_training_benchmarks`, `comprehensive_performance_validation_demo`,
`hyperparameter_optimization_demo`, `memory_efficiency_test`, `memory_optimization_analyzer`,
`multi_gpu_averaged_adam_training`, `optimizer_selection_demo`, `optimizer_visualization_tools`, and
`transformer_training_with_averaged_adam`. A Criterion benchmark harness lives in
`benches/optimizer_benchmark.rs`.

## Testing

- **~960 tests** for this crate (workspace-wide `cargo nextest run --workspace --all-features`:
  18,102 passed / 0 failed / 119 skipped)
- **49 doctests passed, 0 failed, 1 ignored**
- **0 clippy warnings, 0 rustdoc warnings**
- Convergence tests on toy problems for all optimizers
- Numerical stability tests (NaN, Inf, zero gradients)
- Distributed operation tests for ZeRO stages
- Memory usage profiling and quantization accuracy tests
- Schedule-Free convergence equivalence tests
- State save/load round-trip verification

## Known Limitations

- Shampoo-style Kronecker-factored preconditioning is **not implemented** in this crate; the mentioned
  second-order options are Sophia, L-BFGS, Newton-CG, SSBFGS, and SSBroyden.
- `GENIE`, `LoRARITE`, and `SOFO` are simplified, self-described "stub" reference implementations —
  functional and tested, but not yet at full tensor-op parity with their reference papers.
- ~27 modules (e.g. `kernel_fusion`, `federated`, `hyperparameter_tuning`, `zero::zero_stage1`,
  `came`, `prodigy`) carry an explicit `#[allow(dead_code)]` "research-stage module" annotation for
  scaffolding fields/methods not yet on every active call path; the modules compile and are tested but
  should be treated as less battle-hardened than the core SGD/Adam/AdamW/LAMB path.
- ZeRO stage 3 with CPU offload adds host-device transfer overhead.
- 4-bit quantized optimizers may diverge on tasks with very noisy gradients.
- `fsdp`, `optimizer_surgery`, and `per_layer_quant` are implemented but not yet re-exported at the
  crate root (accessible via their full module paths).
- Four source files (`adafactor.rs`, `adafisher.rs`, `advanced_benchmarking.rs`, `second_order_new.rs`)
  are not declared as modules anywhere in the crate and are therefore dead code excluded from the build
  (superseded by `adafactor_new.rs`, `adafisher_simple.rs`, and the `second_order/` directory,
  respectively).

## License

Apache-2.0
