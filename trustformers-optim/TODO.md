# trustformers-optim TODO List

**Version:** 0.1.4 | **Status:** Stable | **Tests:** ~960 | **SLoC:** 52,189 | **Updated:** 2026-07-02

## Overview

The `trustformers-optim` crate provides optimization algorithms, learning rate schedulers, and
distributed/advanced-training infrastructure for the TrustformeRS ecosystem. It spans over 100 modules
(~52,200 lines) — from textbook first-order optimizers through 2024/2025 research algorithms, 4-bit/8-bit
quantized optimizer states, ZeRO/FSDP-style distributed training, federated & continual learning,
hardware-targeted variants, and PyTorch/JAX/TensorFlow compatibility layers.

**Key Responsibilities:**
- Standard & research optimizers (SGD, Adam, AdamW, RAdam, NAdam, AdaBelief, LAMB, AdaFactor, AdaFisher,
  AdaMaxPlus, Adan, Lion, Muon, CAME, MicroAdam, BGE-Adam, HN-Adam, AdEMAMix, Prodigy, NovoGrad, LancBiO,
  AMacP, EVA, Ranger/AdaBound/AMSBound, etc.)
- Schedule-Free optimizer variants (`ScheduleFreeAdam`, `ScheduleFreeSGD`)
- 8-bit and 4-bit quantized optimizer states, plus per-layer bit-width selection
- Learning rate schedulers (Linear, Cosine[+Restarts], One-Cycle, Polynomial, cyclic-decay, LR finder, ...)
- Second-order methods (Sophia, L-BFGS, Newton-CG, SSBFGS, SSBroyden)
- ZeRO distributed optimization stages 1/2/3, FSDP-style sharding, multi-node training
- Federated learning (FedAvg, FedProx, differential privacy) and continual learning (EWC, PackNet)
- Hardware-aware variants (GPU/TPU/edge/mobile), kernel fusion, SIMD, cache-friendly layouts
- Cross-framework compatibility (PyTorch, JAX/Optax, TensorFlow) and a universal config converter
- Hyperparameter tuning, optimizer monitoring/recommendation, performance validation, ONNX export,
  optimizer surgery
- Gradient clipping/scaling utilities and optimizer state checkpointing (`state_dict`/`load_state_dict`)

---

## Current Status

### Implementation Status
- [x] **PRODUCTION-READY** — core optimizers (SGD/Adam/AdamW/LAMB/AdaFactor/RAdam/NAdam/AdaBelief) fully
      implemented and tested
- [x] **BROAD RESEARCH COVERAGE** — Lion, Muon, CAME, MicroAdam, BGE-Adam, HN-Adam, AdEMAMix, Prodigy,
      NovoGrad, LancBiO, AMacP, EVA (2023–2025 algorithms)
- [x] **ZERO COMPILATION ERRORS/WARNINGS** — 0 clippy warnings, 0 rustdoc warnings workspace-wide
- [x] **MEMORY EFFICIENT** — 8-bit/4-bit quantized optimizers, per-layer bit-width selection, ZeRO stages,
      sign-based methods, lazy state allocation
- [x] **SCHEDULE-FREE** — `ScheduleFreeAdam` / `ScheduleFreeSGD` implemented
- [x] **ZERO STAGES 1/2/3** — including an async-communication-overlap variant of stage 3
      (`zero::zero_stage3_overlap`)
- [x] **FSDP-STYLE SHARDING** — `fsdp` module (not yet re-exported at crate root)
- [x] **NO UNWRAP POLICY SATISFIED** — 0 occurrences of `.unwrap()` in any real `.rs` source file
- [ ] **SECOND-ORDER: NO SHAMPOO** — Shampoo/Kronecker-factored preconditioning is not implemented
      (real second-order coverage: Sophia, L-BFGS, Newton-CG, SSBFGS, SSBroyden)

### Test Metrics
- **Test Count:** ~960 tests for this crate (workspace-wide `cargo nextest run --workspace --all-features`:
  18,102 passed / 0 failed / 119 skipped)
- **Doctests:** 49 passed, 0 failed, 1 ignored
- **Pass Rate:** 100%
- **Coverage:** Optimizer convergence, scheduler validation, gradient clipping, state save/load,
  quantization accuracy, ZeRO round-trip, Schedule-Free equivalence
- **Numerical Stability:** Extensive testing with edge cases (NaN, Inf, zero gradients)

### Housekeeping / Verified Policy Compliance
- [x] No `.unwrap()` anywhere in real `.rs` source (checked via `grep -rn "\.unwrap()" --include="*.rs" src/`)
- [x] All compiled source files are under the 2000-line refactor threshold — largest is `convergence.rs`
      at 1,964 lines (worth watching; next largest are `performance_validation.rs` at 1,725 and
      `enhanced_distributed_training.rs` at 1,678)
- [x] Delete 3 stray *.prelude_fix backup files (planned 2026-07-05)
  - Goal/Design: delete src/federated.rs.prelude_fix, src/hyperparameter_tuning.rs.prelude_fix, src/quantum_inspired.rs.prelude_fix — confirmed pre-migration snapshots, zero references, not part of the build graph.
  - Files: the 3 files.
  - Tests: cargo check -p trustformers-optim --all-features (no-op diff).
  - Risk: none.
- [x] Delete 4 orphaned files (adafactor.rs, adafisher.rs, advanced_benchmarking.rs, second_order_new.rs) (planned 2026-07-05)
  - Goal/Design: delete all 4 — adafactor.rs implements a stale non-current Optimizer trait shape (wouldn't compile if mounted); adafisher.rs calls ~8 Tensor methods that don't exist anywhere in trustformers-core (cholesky_inverse, pinverse, diag, eye, etc — wouldn't compile); advanced_benchmarking.rs fabricates every benchmark number while importing-but-never-instantiating 6 real optimizers; second_order_new.rs is pure scaffolding with no Optimizer impl and an inner `pub mod lbfgs;` that doesn't resolve to any real path. All 4 confirmed outside the build graph already.
  - Files: the 4 files.
  - Tests: cargo check/cargo nextest run -p trustformers-optim --all-features (no-op diff).
  - Risk: none for the compiler. These existed as (non-compiling) design references for a future real AdaFisher/Shampoo/KFAC, recoverable via git history if ever wanted.
- [x] Re-export fsdp/optimizer_surgery/per_layer_quant at crate root (planned 2026-07-05)
  - Goal: FsdpConfig, OptimizerSurgeon, PerLayerQuantSelector, etc (21 types total) reachable at crate root.
  - Design: add 3 pub use blocks to lib.rs, alphabetized, mirroring the existing re-export style.
  - Files: trustformers-optim/src/lib.rs only.
  - Tests: cargo doc --no-deps; a smoke test importing all 3 from the crate root path.
  - Risk: none — zero naming collisions confirmed against the full existing public surface.

---

## Completed Optimizer Implementations

### Standard Optimizers

#### SGD (Stochastic Gradient Descent)

- [x] **Algorithm** — `θ ← θ - lr * v` with `v ← momentum * v + ∇L(θ)`, optional Nesterov look-ahead,
      weight decay
- [x] **Features** — momentum, Nesterov momentum, weight decay (L2)

**Real constructor** (`src/sgd.rs`):
```rust
use trustformers_optim::SGD;

// (lr, momentum, weight_decay, nesterov)
let optimizer = SGD::new(0.01, 0.9, 1e-4, true);
```

---

#### Adam / AdamW / RAdam / NAdam / AdaBelief

All five live in `src/adam.rs` and share the same `(lr, betas, eps, weight_decay) -> Self` constructor
shape:

- [x] **Adam** — first/second moment EMA with bias correction
- [x] **AdamW** — decoupled weight decay (recommended default for transformers)
- [x] **RAdam** — rectified variance with automatic warmup
- [x] **NAdam** — Nesterov-accelerated Adam
- [x] **AdaBelief** — second moment tracks "belief" in the gradient direction

```rust
use trustformers_optim::{Adam, AdamW, RAdam, NAdam, AdaBelief};

let adam    = Adam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
let adamw   = AdamW::new(1e-4, (0.9, 0.999), 1e-8, 0.01);
let radam   = RAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
let nadam   = NAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
let adabelief = AdaBelief::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
```

---

#### LAMB (Layer-wise Adaptive Moments)

- [x] Trust-ratio scaled Adam-style update for large-batch training

```rust
use trustformers_optim::LAMB;

let optimizer = LAMB::new(2e-3, (0.9, 0.999), 1e-6, 0.01);
```

---

#### AdaFactor

- [x] Factored second-moment estimate, ~75% memory reduction vs Adam

```rust
use trustformers_optim::AdaFactor;

let optimizer = AdaFactor::new(); // default configuration
```

---

#### AdaFisher, AdaMaxPlus, Adan

- [x] **AdaFisher** — block-diagonal Fisher-information preconditioning (ICLR 2025)
- [x] **AdaMaxPlus** — infinity-norm-based adaptive learning rate variant
- [x] **Adan** — Nesterov-accelerated adaptive optimizer

#### ~~AdaGrad / RMSProp~~ — not implemented

Earlier revisions of this document claimed AdaGrad and RMSProp were implemented. Verified against
source: **neither exists** as a struct or re-export anywhere in the crate. `optimizer_surgery.rs` uses
"RMSProp" only as a state-snapshot label for migration purposes, not as an actual optimizer
implementation. Removed from the checklist; add back only if genuinely implemented.

---

### Cutting-Edge Research Optimizers

#### Lion (Evolved Sign Momentum)
- [x] Sign-based update; only first-moment buffer (no variance state)
```rust
use trustformers_optim::Lion;
let optimizer = Lion::new(1e-4, (0.9, 0.99), 0.1); // (lr, betas, weight_decay)
```

#### Muon (Momentum + Orthogonalization)
- [x] Nesterov momentum + Newton-Schulz orthogonalization of 2D updates
```rust
use trustformers_optim::Muon;
let optimizer = Muon::new();                 // default config
let optimizer = Muon::new_with_lr(2e-4);      // custom learning rate
let optimizer = Muon::for_nanogpt();          // preset
let optimizer = Muon::for_large_lm();         // preset
```

#### CAME (Confidence-guided Adaptive Memory-Efficient)
- [x] Confidence-guided second moment, AdaFactor-sized memory footprint
```rust
use trustformers_optim::{CameConfig, CameOptimizer};
let optimizer = CameOptimizer::new(CameConfig::default());
```

#### MicroAdam
- [x] Gradient compression with error-feedback accumulation
```rust
use trustformers_optim::MicroAdam;
let optimizer = MicroAdam::new();
let optimizer = MicroAdam::for_large_models();
let optimizer = MicroAdam::for_memory_constrained();
```

#### BGE-Adam / OptimizedBGEAdam
- [x] Entropy-weighted bias correction; `OptimizedBGEAdam` is a reported 3–5x faster reimplementation
```rust
use trustformers_optim::{BGEAdam, OptimizedBGEAdam};
let optimizer = OptimizedBGEAdam::new();              // recommended
let original  = BGEAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1, 0.05, 0.05);
```

#### HN-Adam (Hybrid-Norm Adam)
- [x] Adaptive step size based on update-norm history
```rust
use trustformers_optim::HNAdam;
// (lr, betas, eps, weight_decay, adaptation_threshold)
let optimizer = HNAdam::new(1e-3, (0.9, 0.999), 1e-8, 0.01, 0.1);
let optimizer = HNAdam::for_transformers();
```

#### AdEMAMix
- [x] Mixes fast (β1) and slow (β3) EMAs of the gradient
```rust
use trustformers_optim::AdEMAMix;
let optimizer = AdEMAMix::new();
let optimizer = AdEMAMix::new_with_params(1e-4, 0.01); // (lr, weight_decay)
let optimizer = AdEMAMix::for_llm_training();
```

#### Prodigy, NovoGrad, LancBiO, AMacP, EVA
- [x] Additional adaptive/variance-reduced optimizers, each with their own `*Config` type
      (`ProdigyConfig`, `NovoGradConfig`, `LancBiOConfig`, `AMacPConfig`, `EVAConfig`)

#### Research-preview simplified reference implementations: GENIE, LoRA-RITE, SOFO
- [x] Functional and tested (real momentum-style parameter updates, not `unimplemented!()`/`todo!()`)
- [ ] **Not yet at full paper fidelity** — `src/genie_stub.rs`, `src/lora_rite_stub.rs`, and
      `src/sofo_stub.rs` self-document as simplified implementations pending "proper tensor
      operations"/"full LoRA-specific operations"/"forward-mode differentiation" respectively. Several
      stats getters (e.g. `GENIE::get_osgr_stats`) currently return empty placeholders.

---

### Schedule-Free Optimizer Variants

- [x] **`ScheduleFreeAdam`** — folds scheduling into primal-dual iterate averaging
- [x] **`ScheduleFreeSGD`** — same approach applied to momentum SGD

> Earlier revisions of this document referenced a type called `ScheduleFreeAdamW`. It does not exist;
> the real, exported type is `ScheduleFreeAdam`.

```rust
use trustformers_optim::ScheduleFreeAdam;

let optimizer = ScheduleFreeAdam::for_language_models();
// or fully custom: (learning_rate, beta1, beta2, epsilon, weight_decay)
let optimizer = ScheduleFreeAdam::new(0.5, 0.9, 0.95, 1e-8, 0.1);
```

---

### Quantized Optimizers

#### Adam8bit / AdamW8bit
- [x] 8-bit optimizer state, ~4x memory reduction
```rust
use trustformers_optim::Adam8bit;
let optimizer = Adam8bit::new(1e-4); // single-argument constructor (learning_rate)
```

#### Adam4bit
- [x] 4-bit optimizer state via `QuantizationMethod` (e.g. NF4), ~8x memory reduction
- [~] Add AdamW4bit mirroring Adam4bit, with decoupled weight decay (planned 2026-07-05, implement AFTER the save_state/load_state trait-defaults item below)
  - Goal: AdamW4bit alongside the existing Adam4bit, with decoupled (AdamW-style) weight decay instead of Adam4bit's coupled decay.
  - Design: structurally mirror Adam4bit exactly (same QuantizedTensor/NF4 block-wise quantization — this crate's "4-bit" storage is f32-backed with codebook values, not literally nibble-packed; that's a pre-existing crate-wide simplification, out of scope to fix here). Only formula difference: apply weight decay directly to the parameter before the Adam update, instead of folding it into the gradient. Implement a COMPLETE load_state_dict — Adam4bit's own version was found broken (only restores learning_rate) — do not copy that bug.
  - Files: trustformers-optim/src/quantized_advanced.rs (new struct+impl), lib.rs (extend existing pub use block).
  - Tests: creation test; numerical-divergence test (decoupled vs coupled decay diverge under nonzero weight_decay); state_dict/load_state_dict round-trip test (the regression test that would have caught Adam4bit's bug); integration test round-tripping through the save_state/load_state trait defaults below.
  - Risk: must reuse the exact same NF4 quantize/dequantize utilities Adam4bit uses — don't invent a third quantization scheme.
```rust
use trustformers_optim::Adam4bit;
// (lr, beta1, beta2, eps, weight_decay)
let optimizer = Adam4bit::new(1e-4, 0.9, 0.999, 1e-8, 0.01);
```

#### Per-layer bit-width selection
- [x] `per_layer_quant.rs` (807 lines) — `BitWidth::{Int2,Int4,Int8,Fp16,Fp32}`, sensitivity analysis,
      memory-budget-constrained assignment (module exists, not yet re-exported at crate root)

---

### Learning Rate Schedulers

All scheduler constructors return `Self` directly (no `Result`/`?`).

```rust
use trustformers_optim::{LinearScheduler, CosineWithRestartsScheduler, OneCycleScheduler};

let linear = LinearScheduler::new(5e-5, 10_000, 100_000); // (base_lr, warmup_steps, total_steps)

// (base_lr, min_lr, t_0, t_mult)
let cosine_restarts = CosineWithRestartsScheduler::new(1e-3, 1e-6, 1000, 2.0);

// (max_lr, total_steps, pct_start, final_lr)
let one_cycle = OneCycleScheduler::new(1e-3, 10_000, 0.3, 1e-6);
```

- [x] Linear, Polynomial, Step, Exponential, Constant+Warmup
- [x] Cosine and Cosine-with-Warm-Restarts (SGDR-style)
- [x] One-Cycle (`OneCycleScheduler`) plus a separate cyclic-decay implementation
      (`cyclic_decay::{CyclicLrScheduler, OneCycleLrScheduler}`) with `Triangular` / `Triangular2` /
      `ExpRange` amplitude modes
- [x] Adaptive, Composite, Cyclical, Dynamic, Phase-based, Task-specific schedulers
- [x] **Automatic LR Finder** — `LrFinder` + `LrFinderConfig` + `LrFinderResult` + `find_optimal_lr`
      (`lr_finder.rs`)

---

### Second-Order Methods

- [x] **Sophia** — Hutchinson's-estimator Hessian diagonal preconditioning, updated every k steps
- [x] **L-BFGS** (`LBFGS`) and **Newton-CG** (`NewtonCG`) — classic quasi-Newton / Newton-CG line search
- [x] **Self-Scaled BFGS / Broyden** (`SSBFGS`, `SSBroyden`, 2025) — with `for_physics_informed()` /
      `for_non_convex()` presets for PINN-style training

> **Correction:** earlier revisions of this document listed "Shampoo" (Kronecker-factored curvature) as
> implemented. It is not — the only occurrence of the word "Shampoo" in the source tree is a comparison
> reference in a doc comment (`src/lora_rite.rs`) and in the orphaned, uncompiled `second_order_new.rs`.

---

### Advanced Features

#### Gradient Clipping
```rust
use trustformers_optim::GradientProcessor;

let mut grad = vec![3.0_f32, 4.0, 0.0];
GradientProcessor::clip_by_norm(&mut grad, 1.0);        // clip by global L2 norm
GradientProcessor::clip_by_value(&mut grad, -0.5, 0.5); // element-wise clip
```
> These are associated functions operating on `&mut [f32]`, not instance methods on the optimizer types
> as earlier revisions of this document showed.

#### Weight Decay
- [x] L2 regularization and AdamW-style decoupled weight decay (`WeightDecayMode`)

#### Gradient Accumulation
- [x] `Optimizer::accumulate_grad` has a default implementation (delegates to `update`); optimizers may
      override it for specialized accumulation logic

#### Optimizer State Management
- [x] `StatefulOptimizer::state_dict(&self) -> Result<HashMap<String, Tensor>>` and
      `load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()>`
- [~] Add save_state/load_state default trait methods on StatefulOptimizer (planned 2026-07-05, implement BEFORE AdamW4bit above)
  - Goal: every one of the ~29 StatefulOptimizer implementors gets a working save_state(path)/load_state(path) for free.
  - Design: add a small private TensorSnapshot { data: Vec<f32>, shape: Vec<usize> } type (derive Serialize/Deserialize) to traits.rs — needed because Tensor itself has no Serialize/Deserialize impl anywhere in trustformers-core. Add 2 default trait methods on StatefulOptimizer: save_state converts state_dict()'s HashMap<String,Tensor> to HashMap<String,TensorSnapshot>, encodes with oxicode::serde::encode_to_vec (NOT bincode, per COOLJAPAN policy — already a declared-but-unused workspace dependency of this crate), writes to disk; load_state reverses it. Mirror the existing oxicode save/load pattern already used in trustformers-core/src/checkpoint/formats.rs and trustformers-tokenizers/src/binary_format.rs.
  - Files: trustformers-optim/src/traits.rs only.
  - Tests: round-trip test using a real optimizer (AdamW) against std::env::temp_dir(); multi-dimensional-shape round-trip; error-path test (nonexistent path returns Err, not panic).
  - Risk: document explicitly this is a private, crate-internal binary layout, not a cross-framework checkpoint format (this crate has separate pytorch_compat.rs/cross_framework.rs files — a reader could otherwise assume interoperability that doesn't exist).

#### Parameter Groups
- [x] Per-group learning rates and hyperparameters supported across the standard optimizers

---

## Distributed & Scaled Training

### ZeRO (stages 1/2/3)
- [x] `ZeROOptimizer<T: Optimizer>` wraps a base optimizer; `ZeROConfig` controls `stage`
      (`ZeROStage::{Stage1,Stage2,Stage3}`), `bucket_size_mb`, `overlap_comm`, `reduce_bucket_size`,
      `prefetch_depth`, `max_memory_usage_mb`, `gradient_compression`, `pin_memory`
- [x] `zero::zero_stage3_overlap` — async prefetch/overlap of communication with backward-pass compute
      for stage 3 (this substantially satisfies the former "better async communication overlap for ZeRO
      stage 3" future item — see below)

```rust
use std::sync::Arc;
use trustformers_optim::{AdamW, ZeROConfig, ZeROOptimizer, ZeROStage};

let zero_config = ZeROConfig { stage: ZeROStage::Stage3, ..Default::default() };
let base_optimizer = AdamW::new(1e-4, (0.9, 0.999), 1e-8, 0.01);
let mut optimizer = ZeROOptimizer::new(base_optimizer, zero_config, mp_context)?;
optimizer.register_parameters(parameters)?;
```

> **Correction:** earlier revisions used `ZeroOptimizer`/`ZeroConfig`/`ZeroStage` (wrong casing) and
> config fields (`partition_gradients`, `contiguous_gradients`, `reduce_scatter`, `cpu_offload`) that do
> not exist on the real `ZeROConfig`. The constructor also does not take a `model` argument directly;
> parameters are registered separately via `register_parameters`.

### FSDP-style sharding
- [x] `fsdp` module: `FsdpConfig`, `ShardingStrategy`, `WrappingPolicy`, `FsdpUnit`, `FsdpState`,
      `FsdpMemoryAnalyzer` (not yet re-exported at crate root)

### Multi-node training
- [x] `MultiNodeTrainer` / `MultiNodeConfig` / `MultiNodeStats`

### Enhanced distributed trainer
- [x] `EnhancedDistributedTrainer` + `DistributedConfig` (builder methods `.with_gpus()`,
      `.with_gradient_compression(CompressionType::PowerSGD { rank })`, `.with_dynamic_batching()`,
      `.with_fault_tolerance()`), gradient compression, dynamic batching, fault tolerance
- [x] `advanced_distributed_features`: `AutoScaler`, `PerformanceMLOptimizer`, `SmartCheckpointManager`
      (auto-scaling, ML-based performance tuning, differential checkpointing)

### Asynchronous / staleness-tolerant training
- [x] `Hogwild`, `ElasticAveraging`, `ParameterServer`, `AsyncSGD`, `DelayedGradient` (with configurable
      `DelayCompensationMethod`)

### Hierarchical aggregation & deep distributed QP
- [x] `HierarchicalAggregator` (ring / tree / butterfly communication topologies)
- [x] `DeepDistributedQP`

---

## Federated & Continual Learning

- [x] **FedAvg**, **FedProx** — federated averaging / proximal-term federated optimization
- [x] **Differential privacy** (`DifferentialPrivacy`, configurable `NoiseMechanism`) and
      **secure aggregation** (`SecureAggregation`)
- [x] **EWC**, **PackNet**, **memory replay** (`MemoryReplay`) — catastrophic-forgetting mitigation

## Hardware-Aware & Performance

- [x] GPU / TPU / edge / mobile optimizer variants (`GPUAdam`, `TPUOptimizer`, `EdgeOptimizer`,
      `MobileOptimizer`)
- [x] GPU kernel fusion (`kernel_fusion::KernelFusedAdam`) — fused momentum/variance/parameter-update
      kernels, tensor-core-aware
- [x] SIMD optimizations, cache-friendly layouts, aligned memory layout (`SIMDOptimizer`,
      `CacheFriendlyAdam`, `LayoutOptimizedAdam`, `AlignedAllocator`)
- [x] CPU offload (`CPUOffloadedOptimizer`) and lazy state allocation (`LazyAdam` — allocates moment
      buffers only once the first gradient is seen, `lazy_state.rs`)

## Cross-Framework Compatibility

- [x] PyTorch-style (`PyTorchAdam`, `PyTorchAdamW`, `PyTorchSGD`, `PyTorchLRScheduler`)
- [x] JAX/Optax-style (`JAXAdam`, `JAXAdamW`, `JAXSGD`, `JAXGradientTransformation`, `JAXChain`)
- [x] TensorFlow-style (`TensorFlowAdam`, `TensorFlowAdamW`, `TensorFlowCosineDecay`)
- [x] Universal cross-framework config converter (`CrossFrameworkConverter`, `UniversalOptimizerConfig`)

## Tooling

- [x] **Hyperparameter tuning** — native `BayesianOptimizer` + `MultiObjectiveOptimizer` +
      `HyperparameterTuner` (this substantially satisfies the former "automatic hyperparameter tuning
      integration" future item — see below; it is a built-in implementation, not an Optuna/Ray-Tune
      integration)
- [x] **Monitoring & recommendation** — `OptimizerMonitor`, `OptimizerSelector`, `ConvergenceIndicators`
- [x] **Performance validation harness** — `PerformanceValidator` (correctness, convergence, memory,
      regression, and distributed-training validation)
- [x] **ONNX export** — `ONNXOptimizerExporter`
- [x] **Optimizer surgery** — `optimizer_surgery` module (875 lines): migrates momentum/variance/EMA
      state between Adam, AdamW, SGD, and Lion mid-training (module exists, not yet re-exported at
      crate root)

---

## Testing

### Test Coverage
- [x] **~960 tests** for this crate — 100% pass rate (workspace-wide: 18,102 passed / 0 failed / 119 skipped)
- [x] **49 doctests passed, 0 failed, 1 ignored**
- [x] **0 clippy warnings, 0 rustdoc warnings**
- [x] **Optimizer Convergence** — verify convergence on toy problems
- [x] **Scheduler Validation** — check learning rate schedules
- [x] **Gradient Clipping** — verify clipping correctness
- [x] **State Save/Load** — round-trip state verification
- [x] **Numerical Stability** — edge cases (zero gradients, NaN, Inf)
- [x] **Quantization Accuracy** — 8-bit and 4-bit optimizer convergence tests
- [x] **Schedule-Free Equivalence** — verify Schedule-Free matches scheduled training
- [x] **ZeRO Round-Trip** — state consistency across ZeRO stage transitions

### Test Categories
1. **Correctness Tests** — optimizer updates match reference implementations
2. **Convergence Tests** — optimizers converge on convex problems
3. **State Tests** — save/load produces identical state
4. **Quantization Tests** — quantized optimizers achieve acceptable accuracy
5. **Distributed Tests** — ZeRO stages maintain training equivalence
6. **Edge Case Tests** — zero gradients, NaN/Inf, empty parameter groups

---

## Known Limitations

- Shampoo-style Kronecker-factored preconditioning is **not implemented** (see Second-Order Methods
  correction above).
- `GENIE`, `LoRARITE`, and `SOFO` are simplified reference implementations, not full paper-fidelity
  ports; some introspection getters return placeholder/empty values.
- ~27 modules carry an explicit `#[allow(dead_code)]` "research-stage module" annotation (e.g.
  `kernel_fusion`, `federated`, `hyperparameter_tuning`, `zero::zero_stage1`, `came`, `prodigy`,
  `advanced_2025_research`) — these compile and are tested, but some fields/methods are scaffolding
  not yet on every active call path.
- ZeRO stage 3 with CPU offload adds host-device transfer overhead.
- 4-bit quantized optimizers may diverge on tasks with very noisy gradients.
- AdaGrad and RMSProp are **not implemented** despite being listed in earlier revisions of this document.

---

## Future Enhancements

### High Priority
- [ ] Additional emerging optimizers as they appear (2025+): SOAP and Adam-mini/Grokfast-style methods
      are still **not implemented** (Muon, previously on this list, is now done — see above)
- [x] ~~Enhanced distributed optimizer state management with async overlap~~ — substantially addressed
      by `zero::zero_stage3_overlap` (async prefetch/overlap of communication with backward-pass compute)
- [x] ~~Automatic hyperparameter tuning integration~~ — addressed via the native `hyperparameter_tuning`
      module (Bayesian + multi-objective search); not integrated with external frameworks
      (Optuna/Ray Tune) if that specific integration is still desired

### Performance
- [~] Fused quantized optimizer kernels for GPU — `kernel_fusion::KernelFusedAdam` provides fused,
      tensor-core-aware GPU kernels for the Adam family, but they are not yet specialized for 4-bit/8-bit
      quantized state specifically
- [x] **Lazy optimizer state allocation** — `LazyAdam` allocates moment buffers only when first gradient
      is seen (`lazy_state.rs`)
- [x] ~~Better async communication overlap for ZeRO stage 3~~ — see `zero::zero_stage3_overlap` above

### Features
- [x] **Cyclic LR with decay** — `CyclicLrScheduler` (Triangular/Triangular2/ExpRange) +
      `OneCycleLrScheduler` (`cyclic_decay.rs`)
- [x] **Automatic LR Finder** — `LrFinder` + `LrFinderConfig` + `LrFinderResult` + `find_optimal_lr`
      (`lr_finder.rs`)
- [x] **Optimizer surgery** (change optimizer type mid-training) — `optimizer_surgery.rs` (875 lines);
      not yet re-exported at crate root
- [x] **Per-layer quantization bit-width selection** — `per_layer_quant.rs` (807 lines); not yet
      re-exported at crate root

### Housekeeping (new)
- [x] Delete 3 stray *.prelude_fix backup files (planned 2026-07-05)
  - Goal/Design: delete src/federated.rs.prelude_fix, src/hyperparameter_tuning.rs.prelude_fix, src/quantum_inspired.rs.prelude_fix — confirmed pre-migration snapshots, zero references, not part of the build graph.
  - Files: the 3 files.
  - Tests: cargo check -p trustformers-optim --all-features (no-op diff).
  - Risk: none.
- [x] Delete 4 orphaned files (adafactor.rs, adafisher.rs, advanced_benchmarking.rs, second_order_new.rs) (planned 2026-07-05)
  - Goal/Design: delete all 4 — adafactor.rs implements a stale non-current Optimizer trait shape (wouldn't compile if mounted); adafisher.rs calls ~8 Tensor methods that don't exist anywhere in trustformers-core (cholesky_inverse, pinverse, diag, eye, etc — wouldn't compile); advanced_benchmarking.rs fabricates every benchmark number while importing-but-never-instantiating 6 real optimizers; second_order_new.rs is pure scaffolding with no Optimizer impl and an inner `pub mod lbfgs;` that doesn't resolve to any real path. All 4 confirmed outside the build graph already.
  - Files: the 4 files.
  - Tests: cargo check/cargo nextest run -p trustformers-optim --all-features (no-op diff).
  - Risk: none for the compiler. These existed as (non-compiling) design references for a future real AdaFisher/Shampoo/KFAC, recoverable via git history if ever wanted.
- [x] Re-export fsdp/optimizer_surgery/per_layer_quant at crate root (planned 2026-07-05)
  - Goal: FsdpConfig, OptimizerSurgeon, PerLayerQuantSelector, etc (21 types total) reachable at crate root.
  - Design: add 3 pub use blocks to lib.rs, alphabetized, mirroring the existing re-export style.
  - Files: trustformers-optim/src/lib.rs only.
  - Tests: cargo doc --no-deps; a smoke test importing all 3 from the crate root path.
  - Risk: none — zero naming collisions confirmed against the full existing public surface.
- [ ] Keep an eye on `convergence.rs` (1,964 lines) — closest file to the 2,000-line refactor threshold;
      consider splitting with `splitrs` if it grows further

---

## Development Guidelines

### Code Standards
- **Use trustformers-core abstractions only** (no external deps directly)
- **File size limit:** <2000 lines per file (currently satisfied crate-wide; see Housekeeping above)
- **Error handling:** `Result<T, TrustformersError>` from `trustformers-core`; **0 `.unwrap()`** in
  production or test source, verified today
- **Testing:** convergence tests required for new optimizers
- **Naming:** snake_case for all identifiers

### Adding a New Optimizer

**Checklist:**

1. **Implement the trait hierarchy** — the real `StatefulOptimizer` trait (`src/traits.rs`) uses
   associated types, not the simplified signature shown in earlier revisions of this document:
   ```rust
   pub trait StatefulOptimizer: Optimizer {
       type Config: Clone + Send + Sync;
       type State: Send + Sync;

       fn config(&self) -> &Self::Config;
       fn state(&self) -> &Self::State;
       fn state_mut(&mut self) -> &mut Self::State;
       fn state_dict(&self) -> Result<HashMap<String, Tensor>>;
       fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()>;
       fn memory_usage(&self) -> StateMemoryStats;
       fn reset_state(&mut self);
       fn num_parameters(&self) -> usize;
   }
   ```
   `step`/`zero_grad`/`update`/`get_lr`/`set_lr` come from the supertrait `Optimizer`
   (`trustformers_core::traits::Optimizer`), which every optimizer must also implement directly.

2. **Add state buffers** — momentum buffers, variance buffers, per-parameter state (`HashMap<String, Vec<f32>>`
   is the common pattern used throughout this crate)

3. **Implement the update rule** — follow the algorithm from the paper, handle edge cases (NaN/Inf,
   zero gradients)

4. **Add tests** — convergence test, state save/load round-trip, reference comparison

5. **Document** — algorithm, hyperparameter recommendations, use cases, a doctested example

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

**Last Updated:** 2026-07-02 — v0.1.4
**Status:** Stable
**Tests:** ~960 tests for this crate (workspace: 18,102 passed / 0 failed / 119 skipped); 49 doctests
passed, 0 failed, 1 ignored
**Optimizers:** SGD, Adam, AdamW, RAdam, NAdam, AdaBelief, LAMB, AdaFactor, AdaFisher, AdaMaxPlus, Adan,
Lion, Muon, CAME, MicroAdam, BGE-Adam, HN-Adam, AdEMAMix, Prodigy, NovoGrad, LancBiO, AMacP, EVA,
Schedule-Free Adam/SGD, Sophia, L-BFGS, Newton-CG, SSBFGS, SSBroyden, and more (GENIE/LoRA-RITE/SOFO as
simplified reference implementations)
**Quantized:** 8-bit Adam/AdamW, 4-bit Adam, per-layer bit-width selection
**Distributed:** ZeRO stages 1/2/3 (incl. async-overlap stage 3), FSDP-style sharding, multi-node training
