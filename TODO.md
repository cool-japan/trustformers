# TrustformeRS TODO List

## Project Overview

TrustformeRS is a high-performance, memory-safe Rust implementation of Hugging Face Transformers.
The project provides a comprehensive ecosystem for transformer model development, training, and deployment
with support for 21+ architectures and multiple deployment targets.

### Version Information
- **Current Version:** 0.1.3 (Unreleased)
- **Previous Release:** 0.1.2 (Released 2026-06-20)
- **Status:** Active Development (v0.1.3)
- **License:** Apache-2.0
- **Repository:** https://github.com/cool-japan/trustformers

### Project Health (corrected 2026-06-19 — see Code-Quality Audit below)
- ✅ **Compiles cleanly** across all crates (no-warnings policy enforced where checked)
- ✅ **Large test suite** — ~23,700 `#[test]`/`#[tokio::test]` functions (run locally with `cargo nextest`; a Rust CI gate is being restored, so the "100% pass" claim is not yet machine-verified — Task 9 below)
- 🟡 **49+ architectures for CPU inference** — maturity varies; a batch of "fake implementation" defects was fixed 2026-06-19 (see audit), others may remain
- 🟡 **Compute is CPU / `f32`** — F16/BF16 are storage-only (upcast for math); GPU is wired only for GPT-2/RetNet (see audit Tasks 4 & 5)
- ✅ **100% Pure Rust** source (~1.4M SLoC; GPU backends bind to system libraries via Rust crates)

---

## 🔍 Code-Quality Audit (2026-06-19)

A thorough source audit was run against external criticism (Reddit) that the project
over-claims its capabilities. The criticism was found to be **exaggerated but
substantively correct** in several areas. Evidence-backed verdicts:

| Criticism | Verdict | Detail |
|-----------|---------|--------|
| GPU "in name only" | **Partially true** | Real CUDA(`cudarc`)/Metal backends exist, but core `Tensor::matmul` errors on GPU variants and only GPT-2/RetNet wire GPU into `forward` (2 of ~58 models). `gpu.rs::GpuMemoryPool` is a counter-only simulation; device detection is stubbed. |
| "Float32 only" | **True (compute)** | 11 of 17 core ops are F32(+F64) only; F16/BF16 are storage that is upcast to f32. RMSNorm is `f32`-hardcoded. |
| Stubs / `todo!()` | **Partially true** | Only 12 `todo!()` (mostly in doc-comments) + 0 `unimplemented!()`, BUT ~15–25 genuinely *fake* functions returned wrong/placeholder results in real compute paths. |
| Allocation-heavy / non-idiomatic | **Partially true (~50%)** | Real `Vec` round-trips in matmul/SDPA hot loops; stringly-typed activation dispatch in 20+ models. (`[T;N]` const-generic critique does **not** apply to dynamic tensors.) |
| README false advertising ("看板倒れ") | **Several FALSE claims** | Fabricated LLaMA-7B GPU benchmark numbers, a non-compiling GPU code example, overstated test count, TPU listed as supported (empty feature flag). |

### ✅ Resolved 2026-06-19
- **README & this file**: removed fabricated GPU benchmarks, fixed the non-compiling GPU example, corrected GPU/precision/test/TPU claims, added honest maturity notes.
- **Fake model implementations replaced with real algorithms (+ tests, all green):**
  - `mamba`: real S6 selective scan (ZOH discretisation `Ā=exp(ΔA)`, `B̄=ΔB`, recurrent `h_t`) + real causal depthwise `conv1d` (was `silu(x)` / identity).
  - `starcoder2`: real `rotate_half` RoPE + real GQA causal attention (rank-agnostic 2D/3D) (was return-clone RoPE + `q·scale` only).
  - `phi3`: real RoPE (incl. LongRope scaling) + real GQA causal attention (was identity RoPE + "return query").
  - `flamingo`: real cross-attention via existing SDPA (was uniform/all-ones weights).
- **Mathematically-wrong helpers fixed**: privacy-inference `softmax` (was `exp` w/o normalisation), `renormalize_probabilities`, MLX `BatchNorm`/`Embedding`, mobile lifecycle freed-byte reporting, RLHF `compute_rating_std` (was hardcoded `0.1`) & `filter_by_quality`, gradient-checkpoint forward/backward.
- **Dead code removed**: ~11.5k lines of orphaned `*_backup.rs` / `*_backup/` files across core/serve/mobile.

### ✅ Resolved 2026-06-20 (Task 2/3 follow-up)
- **RLHF feedback scoring de-faked** (`trustformers-training/src/rlhf/feedback.rs`): `compute_quality_scores`
  used a constant `mean_rating = 0.5` base for every item — now bases each item's quality on its **own**
  rating (per-item, with group-average / global-mean fallbacks). `compute_batch_statistics` hardcoded
  `mean_rating: 0.5` / `std_rating: 0.1` — now computes the real batch mean and reuses the existing
  `compute_rating_std_single` (sample std). Both excused themselves with "item() method not available",
  but `to_vec_f32()` was already in use 20 lines below. +2 behavioural tests (assert scores track ratings
  and stats ≠ the old constants); `cargo test -p trustformers-training --lib rlhf::feedback` → 20 passed,
  0 warnings. NOTE: the OPEN-sites list below had mis-filed this as `memory_optimization.rs`; corrected.
  `aggregate_ratings` deliberately left as-is — `ratings` is 1-D `[batch_size]` (one rating/item, confirmed
  by `test_feedback_batch_processing`), so `Mean`→clone is correct and a 2-D annotator semantic would be invented.
- **Adaptive-computation stats de-faked** (`trustformers-core/src/adaptive_computation.rs`):
  `compute_entropy`/`compute_variance`/`compute_sparsity` returned hardcoded `0.5`/`0.3`/`0.2` (ignoring their
  `tensor` argument) yet feed `analyze_input_complexity`'s **live** branching/complexity score. Now: entropy →
  a new reusable `Tensor::softmax_entropy_normalized()` (numerically-stable softmax → `H/ln(n)` ∈ [0,1], added
  to `tensor/utils.rs`); variance → existing core `Tensor::variance`; sparsity → existing core `Tensor::sparsity`
  (no duplicate math). +1 entropy test (uniform≈1, peaked≈0, single=0); all 36 `adaptive_computation` tests
  still green; 0 warnings; file kept at 1998 lines (**< 2000** — delegated rather than inlined to respect the
  size policy). `execute_single_path` still returns `input.clone()` (honest scaffold; needs a real model
  plumbed — see tracked future work below), but the metrics that *select* paths are now real.
- **Broken crate-doc examples fixed** (`trustformers-training/src/lib.rs`): the "Quick Start" and "Distributed
  Training" `//!` doctests never compiled — they used a name-collided `TrainingConfig` (resolved to
  `model_versioning::TrainingConfig`), a 3-arg `Trainer::new`, and a non-existent `Trainer::new_distributed`.
  Rewritten against the real API (`Trainer::new(model, TrainingArguments, Box<dyn Optimizer>, Box<dyn Loss>,
  TaskType)` with `trustformers_optim::Adam` + `MSELoss`; `DataParallelTrainer::new` + `DistributedConfig` +
  `SimulatedProcessGroup` for the distributed one), with a hidden minimal mock model so the rendered docs stay
  clean. `cargo test -p trustformers-training --doc` → 2 passed (was 2 failed). Same "看板倒れ" class as the
  README fix in Task 1.
- **CI-incidental code fixes** (kept; GitHub Actions intentionally NOT added — billable): `trustformers-core`
  `hardware_acceleration` bench failed to compile (`criterion_main!` inside `mod benches` → no crate-level
  `main`, E0601) — moved to crate root; `gpu.rs` `detect_devices` `vec_init_then_push` clippy warning fixed
  (with feature-conditional `allow(unused_mut)`).
- **Full local verification (free, no GitHub CI), 2026-06-20 — all green**:
  `cargo fmt --all -- --check` = 0 diffs · `cargo clippy --workspace --all-targets -- -D warnings` = exit 0 ·
  `cargo nextest run --workspace` (default) = **14,498 passed** / 0 failed · `cargo nextest run -p
  trustformers-models --features all --retries 3` = **4,319 passed** / 0 failed (1 timing-flaky recovered:
  `memory_profiling::test_monitoring_performance_stats`, unrelated) · core doctests = 66 passed · training
  doctests = 2 passed · bench builds. The "no-warnings / tests pass" claim is now locally machine-verified.

### 📌 Handoff — codebase facts a successor MUST know

Read this before touching anything; these are non-obvious and cost time to rediscover.

- **`trustformers-models` is feature-gated per model.** `default = ["bert"]`, and each model is its
  own empty feature (`mamba = []`, `phi3 = []`, `starcoder2 = []`, `flamingo = []`, …). A bare
  `cargo check -p trustformers-models` **does not compile most models**. Always pass the features you
  touch, e.g. `--features "mamba,starcoder2,phi3,flamingo"`, or `--features all` for everything.
  This is why a default check can finish in <1s while silently skipping your file.
- **Two GPU systems, don't confuse them.** The *real* backends are in
  `trustformers-core/src/gpu_ops/` (`cuda/` via `cudarc`, `metal/` via `objc2`/MPS, `webgpu.rs`,
  `opencl.rs`, `rocm.rs`) — feature-gated, off by default. The high-level
  `trustformers-core/src/gpu.rs` (`GpuContext`, `GpuMemoryPool`, `detect_*_devices`) is a **CPU
  simulation**: `allocate()` just increments a counter and device detection returns hard-coded
  placeholder devices. Core `Tensor::matmul` **now dispatches** to `gpu_ops` for `Tensor::CUDA`/
  `Tensor::Metal` operands (added 2026-06-20, `#[cfg(feature=…)]`-gated) — but per-model `forward` GPU
  wiring is still only `gpt2`/`retnet`, and `gpu.rs` remains a CPU simulation.
- **Working reference implementations to copy** for attention/RoPE work:
  `trustformers-models/src/mistral/model.rs` and `…/starcoder2/model.rs` (the latter was fixed in
  this campaign and is rank-agnostic for 2D `[seq,h]` and 3D `[batch,seq,h]`). NOTE: mistral leaves
  stray `eprintln!` debug lines — do not copy those.
- **Confirmed `Tensor` API** (`trustformers_core::tensor::Tensor`, an enum `F32(ArrayD<f32>)`, `F16`,
  `BF16`, … plus `Metal`/`CUDA` buffer variants): `reshape(&[usize])`, `transpose(usize,usize)`,
  `matmul(&Tensor)` (4-D batched OK), `mul_scalar(f32)`, `softmax(i32)` (use `-1`), `add(&Tensor)`
  (broadcasts `[1,1,S,S]` against `[B,H,S,S]`), `from_vec(Vec<f32>,&[usize])`, `shape() -> Vec<usize>`,
  `data() -> Result<Vec<f32>>`. The additive causal-mask helper pattern is `fn causal_mask(seq)` in
  `starcoder2/model.rs`.
- **Policies enforced here**: no warnings; no `todo!()`/`unimplemented!()`/placeholder returns; use
  `scirs2_core::ndarray` / `scirs2_core::random` (NOT raw `ndarray`/`rand`); files < 2000 lines.

### ⏳ Remaining work (tracked, prioritised — each item is independently actionable)

> **STATUS 2026-06-20 — parallel batch landed.** #4 (precision F16/BF16 upcast), #5 (GPU matmul→`gpu_ops`
> dispatch), #7 (`ActivationType` enum, 12 models), #10 (TPU façade removed) are **DONE** — run as
> worktree-isolated parallel agents, 3-way-integrated onto HEAD, and re-verified TOGETHER:
> `cargo clippy --workspace --all-targets --features all -- -D warnings` = clean (now lints all ~58
> models, not just `bert`), `fmt --check` = 0 diffs, core `--lib` = 2269 passed, models `--lib
> --features all` = 4322 passed. #9 also done (local verification; **GitHub Actions CI intentionally NOT
> added — billable, per user**). **#6 (efficiency) also DONE 2026-06-20** — all 16 Vec round-trips in
> the matmul/SDPA hot paths removed (borrow `ArrayView2` instead of `from_shape_vec(to_vec())`; `as_slice`
> instead of `iter().copied().collect()`; `as_standard_layout().into_owned()` instead of the triple-copy).
> Verified: core 2269 + models-all 4324 pass, fmt/clippy clean, matmul ~26% faster (67.8µs→~50µs). **The
> entire original-criticism campaign (#1–#10) is now complete.**
> Implementation notes: precision upcast is inline via `half::{f16,bf16}::{to_f32,from_f32}` because
> `conversions.rs::to_f32`/`to_dtype` have NO F16/BF16 support (the handoff above is corrected by this).
> Two PRE-EXISTING F32 bugs were found (not fixed, out of scope): `advanced.rs::layer_norm` general-shape
> indexing and `statistical.rs::variance` axis-wise broadcast. The #4/#5/#7/#10 blocks below are kept for
> historical context.

**4. Precision — wire F16/BF16 through core ops.**
   - Where: `trustformers-core/src/tensor/math_ops/linear_algebra.rs` (`matmul`),
     `…/tensor/activations.rs`, `…/tensor/math_ops/*`, `…/fused/mod.rs` (`rms_norm_slice` is
     `&[f32]`-hardcoded). 11 of 17 surveyed ops are F32(+F64) only.
   - How: add a consistent upcast-compute-downcast path (F16/BF16 → f32 → compute → original dtype)
     so non-F32 tensors stop returning `Err("… not supported for these tensor types")`. Native
     low-precision kernels are a later optimisation.
   - Verify: add tests feeding F16/BF16 tensors to `matmul`/`softmax`/`layer_norm`; assert finite,
     shape-correct output. `cargo test -p trustformers-core --lib`.

**5. GPU — connect real backends to the compute/model path.**
   - Where: `Tensor::matmul` & friends in `…/tensor/math_ops/`; dispatch fns in
     `…/gpu_ops/{cuda,metal,webgpu}` (`dispatch_cuda_matmul`, `dispatch_matmul`, …); per-model
     `forward` (model the wiring on `gpt2/model/model_core.rs` + `model_blocks.rs`, which use
     `to_device_enum` + `*_gpu_to_gpu`).
   - How: in the core ops, detect `Tensor::CUDA`/`Tensor::Metal` variants and route to the matching
     `gpu_ops` dispatch; factor a device-aware self-attention/linear helper and adopt it across
     models so GPU is not GPT-2/RetNet-only. Reconcile/retire the simulated `gpu.rs` layer (or make
     its `GpuMemoryPool`/detection back onto the real backends). Big task — stage it: (a) core matmul
     GPU dispatch + tests, (b) one extra model (e.g. llama), (c) generalise.
   - Verify: feature-gated tests under `--features cuda` / `metal`; keep CPU default green.

**6. Efficiency — kill hot-path allocations.**
   - Where: `…/tensor/math_ops/linear_algebra.rs` and `…/layers/sdpa.rs` — **16** `iter().copied()
     .collect::<Vec<f32>>()` sites that round-trip already-contiguous data through a `Vec` before
     `from_shape_vec`/BLAS, mostly inside per-batch/per-head loops.
   - How: prefer `.as_slice()` when the view is contiguous and pass it straight to BLAS; only allocate
     on the non-contiguous fallback. Remove the `as_standard_layout().to_owned()` double-copy at the
     top of `matmul`. **Benchmark before/after** (`cargo bench`, `benches/`) — `from_shape_vec` needs
     ownership, so confirm each removal is a real win and not load-bearing for the BLAS signature.

**7. Idiom — enum-ise activation dispatch.**
   - Where: **14** files in `trustformers-models/src/` use `match self.<act>.as_str() { "gelu" => … }`
     per forward pass (albert, moe, hyena, linformer, fnet, performer, gpt_neo, retnet, gpt_j, …).
   - How: introduce an `ActivationType` enum (parse once at config load), give it an `apply(&Tensor)`,
     and migrate each model. Mechanical but spread out; do it crate-wide in one pass.

**9. Infra — restore Rust CI (regression guard for everything above).**
   - The Rust workflows are archived in `.github/_archived/workflows/` (`ci.yml`, `code-quality.yml`,
     `code-style.yml`, `pr-tests.yml`, `benchmarks.yml`, …); `.github/workflows/` currently has only
     `npm-publish.yml` + `pypi-publish.yml`.
   - How: restore a `cargo nextest run` (+ per-model feature matrix or `--features all`) and
     `cargo clippy -- -D warnings` gate. This is what makes the "100% pass / no-warnings" claim true
     and stops fake-impl / non-compiling-example regressions. **High leverage — recommend doing first.**

**10. Backends — make TPU honest.**
   - `tpu = []` (empty feature, no deps), `trustformers-core/src/kernels/tpu_impl.rs` is comments only.
   - How: either implement a minimal real TPU path with a smoke test, or delete the `tpu` flag +
     `tpu_impl.rs` and keep README/this-file scoped to CUDA/Metal/WebGPU (already done in prose). The
     same applies to ROCm/Vulkan/OpenCL: real backend code exists but is unverified and non-default —
     either add smoke tests or label clearly experimental.

### 🧪 Known fake/placeholder sites still OPEN (audit-flagged, NOT yet fixed)

Fix under Task 2/3 follow-up — triaged 2026-06-20:
- ~~`trustformers-training/.../rlhf/feedback.rs` — `compute_quality_scores` / `compute_batch_statistics`
  hardcoded `0.5` / `0.1`~~ — **RESOLVED 2026-06-20** (see above; was mis-filed as `memory_optimization.rs`).
- `trustformers-mobile/src/react_native.rs:722` — `run_inference` returns `input.clone()`. This is NOT a
  silent fake: the whole block is openly `// Mock implementation of MobileInferenceEngine methods for
  React Native` (every method — `initialize`/`load_model_from_path`/`run_inference`/`is_model_loaded` — is
  an FFI-boundary stub holding no model/weights). Making it real means plumbing the actual mobile engine
  through the RN bridge — a **feature**, not a quick correctness fix. Tracked as future work, not a hidden defect.
- `trustformers-core/src/adaptive_computation.rs:~881` — `execute_single_path` returns `input.clone()` as a
  path's "neural-network output". Honest scaffold (the struct holds no model); the *path-selection* metrics it
  consumes are now real (resolved above), but actually executing a path needs a real model wired in — a feature.
- `trustformers-wasm/src/layers.rs:194` — dropout disabled ("return input as dropout requires RNG");
  fine for inference but should use `scirs2_core::random` if training on WASM.
- `trustformers/src/profiler.rs:334` — returns a mock dashboard URL (benign, but labelled "in a real
  implementation…").
- `trustformers-c/src/cuda.rs` (legacy, excluded from workspace) — CUDA matmul copies via host memory
  as a placeholder; `cloud/aws_lambda.rs` returns `{"task":"placeholder"}`. Lower priority (legacy crate).
- Re-run the sweep periodically: `grep -rniE "placeholder|return input|simplified|in a real implementation|dummy|hardcoded" --include=*.rs | grep -v /target/` and triage real-compute hits vs. benign error strings.

### ✅ Verification cheat-sheet (what "done" looked like this campaign)

```bash
# Models touched this campaign (compile + behavioural tests, all green):
cargo test -p trustformers-models --lib --features "mamba,starcoder2,phi3,flamingo"
#   → 1560 passed; 0 failed; 0 warnings
# Helper/cleanup crates:
cargo check -p trustformers-core -p trustformers-mobile -p trustformers-training -p trustformers-serve --lib
#   → Finished, exit 0, no warnings
# Full model matrix when validating broad changes:
cargo check -p trustformers-models --lib --features all
```

---

## Workspace Structure

TrustformeRS is organized as a Cargo workspace of specialized crates:

### Per-Crate Status (v0.1.3, 2026-06-24)

| Crate | Tests | Status | SLoC |
|-------|-------|--------|------|
| trustformers-core | 1,077 | Stable | 204,130 |
| trustformers-models | 688 | Alpha | 196,463 |
| trustformers-training | 333 | Stable | 89,413 |
| trustformers-tokenizers | 500 | Stable | 51,211 |
| trustformers-optim | 535 | Stable | 71,429 |
| trustformers-serve | 586 | Stable | 361,251 |
| trustformers-debug | 323 | Alpha | 101,448 |
| trustformers-wasm | 128 | Stable | 55,493 |
| trustformers-mobile | 513 | Alpha | 143,001 |
| trustformers | 675 | Alpha | 134,295 |
| **Total** | **5,358** | | **~1,408,134** |

*(v0.1.0 baseline: 5,007 tests / ~900,000+ SLoC)*

> **Correction (2026-06-19 audit):** the "5,358" figure is a stale/curated snapshot. The actual
> in-tree count is **~23,700** `#[test]` + `#[tokio::test]` functions, but there is currently **no
> Rust CI** to substantiate a "100% pass" claim (CI is archived — see Task 9 in the Code-Quality
> Audit above). Treat per-crate numbers in this table as historical until CI is restored.

### Core Crates
1. **trustformers-core** - Fundamental tensor operations, layers, hardware acceleration (Stable)
2. **trustformers-models** - 27+ transformer model implementations (Alpha)
3. **trustformers-tokenizers** - BPE, WordPiece, SentencePiece tokenizers (Stable)
4. **trustformers-optim** - 20+ optimization algorithms and learning rate schedulers (Stable)
5. **trustformers-training** - Complete training infrastructure, RLHF/DPO support (Stable)
6. **trustformers** - High-level integration crate with unified API (Alpha)

### Deployment Crates
7. **trustformers-wasm** - WebAssembly + WebGPU deployment (Stable)
8. **trustformers-mobile** - iOS/Android deployment with hardware acceleration (Alpha)
9. **trustformers-serve** - REST/gRPC/GraphQL serving, dynamic batching (Stable)
10. **trustformers-debug** - Debugging tools, profilers, TensorBoard integration (Alpha)

---

## Completed Features

### Core Infrastructure

#### Tensor Operations (trustformers-core)
- ✅ **Multi-Backend Tensor Abstraction**
  - Unified tensor API across CPU, CUDA, ROCm, Metal, Vulkan, XLA, TPU
  - Automatic backend selection based on availability
  - Zero-copy operations where possible

- ✅ **Mathematical Operations**
  - Basic arithmetic: add, sub, mul, div, matmul, pow, sqrt
  - Advanced operations: einsum, gather, scatter, index_select
  - Broadcasting support compatible with NumPy/PyTorch semantics
  - Reduction operations: sum, mean, max, min, argmax, argmin, std, var

- ✅ **Activation Functions**
  - ReLU, GELU (exact and approximate), SiLU/Swish, Tanh, Sigmoid
  - Softmax, LogSoftmax with numerical stability
  - Fused operations for performance (bias+activation)

- ✅ **Shape Manipulation**
  - Reshape, transpose, permute, squeeze, unsqueeze, flatten
  - Concatenate, split, chunk operations
  - View operations with zero-copy when possible

- ✅ **Data Types**
  - Full precision: F32, F64
  - Half precision: F16, BF16
  - Integer types: I8, I16, I32, I64, U8, U16, U32, U64
  - Complex numbers: C32, C64, CF16, CBF16
  - Sparse tensor support

#### Memory Management
- ✅ **Advanced Memory Pool** - LRU eviction policy, configurable size limits
- ✅ **Zero-Copy Operations** - Minimize data movement with smart views
- ✅ **Memory-Mapped Loading** - Load large models without RAM overhead
- ✅ **LazyTensor Loading** - On-demand weight loading for memory efficiency
- ✅ **Automatic Optimization** - Dynamic memory allocation strategies
- ✅ **Scoped Allocations** - Mobile-optimized memory management
- ✅ **Memory Profiling** - Track allocations and detect leaks
- ✅ **Custom Allocator** - Integration with jemalloc/mimalloc

#### Model/Layer Abstraction
- ✅ **Model Trait System**
  - Generic `Model` trait for all architectures
  - `Config` trait for hyperparameter management
  - `Layer` trait for composable building blocks
  - `Tokenizer` trait for text processing

- ✅ **Core Layers**
  - Linear (dense) layers with optional bias
  - Embedding layers with padding token support
  - LayerNorm with configurable epsilon
  - RMSNorm (LLaMA-style normalization)
  - Dropout with training/inference modes
  - Residual connections

- ✅ **Attention Mechanisms**
  - Multi-head attention (MHA)
  - Grouped-query attention (GQA)
  - Multi-query attention (MQA)
  - Flash Attention integration
  - Sliding window attention (Mistral)
  - Rotary position embeddings (RoPE)
  - ALiBi positional encoding

- ✅ **Feed-Forward Networks**
  - Standard FFN with configurable activations
  - SwiGLU (LLaMA-style gated FFN)
  - Mixture of Experts (MoE) support

---

### Model Architectures (27+ Models)

#### Encoder Models (BERT Family)
- ✅ **BERT** - Bidirectional Encoder Representations from Transformers
  - Base (110M params) and Large (340M params) variants
  - Absolute position embeddings
  - Segment embeddings for sentence pairs
  - Complete weight loading from HuggingFace

- ✅ **RoBERTa** - Robustly Optimized BERT Pretraining Approach
  - Improved pretraining recipe
  - Dynamic masking
  - Larger batch sizes and learning rates

- ✅ **ALBERT** - A Lite BERT
  - Factorized embedding parameterization
  - Cross-layer parameter sharing
  - Sentence-order prediction (SOP)

- ✅ **DeBERTa** - Decoding-enhanced BERT with disentangled attention
  - Disentangled attention mechanism
  - Enhanced mask decoder
  - Relative position encodings

- ✅ **DistilBERT** - Distilled version of BERT
  - 6-layer student network (vs 12-layer BERT)
  - 40% smaller, 60% faster
  - Knowledge distillation from BERT-base

- ✅ **ELECTRA** - Efficiently Learning an Encoder
  - Replaced token detection pretraining
  - More efficient than masked language modeling

#### Decoder Models (GPT Family & Modern LLMs)
- ✅ **GPT-2** - Generative Pre-trained Transformer 2
  - Small (124M), Medium (355M), Large (774M), XL (1.5B) variants
  - Causal self-attention with learned positional embeddings
  - Byte-level BPE tokenization
  - Generation with temperature, top-k, top-p, beam search

- ✅ **GPT-Neo** - EleutherAI's GPT-3 alternative
  - 125M, 1.3B, 2.7B parameter models
  - Local and global attention patterns
  - Rotary position embeddings option

- ✅ **GPT-J** - 6B parameter autoregressive language model
  - Rotary position embeddings
  - Parallel attention and FFN for efficiency
  - Dense attention across full sequence

- ✅ **LLaMA** - Large Language Model Meta AI
  - 7B, 13B, 30B, 65B parameter models
  - RoPE positional encodings
  - RMSNorm pre-normalization
  - SwiGLU activation function
  - Complete weight loading infrastructure

- ✅ **Mistral** - 7B model with innovations
  - Sliding window attention (4096 window size)
  - Grouped-query attention (GQA)
  - Byte-fallback BPE tokenizer
  - RoPE with theta=10000

- ✅ **Gemma** - Google's lightweight LLM family
  - 2B and 7B variants
  - Multi-query attention
  - GeGLU activation
  - RMSNorm normalization

- ✅ **Qwen** - Alibaba's multilingual large language model
  - Multiple size variants
  - Extended context length support
  - Multilingual pretraining (Chinese, English, etc.)

- ✅ **Phi-3** - Microsoft's small language model
  - High performance at small scale
  - Efficient architecture
  - Specialized training data

- ✅ **Falcon** - Technology Innovation Institute
  - Multi-query attention
  - RoPE positional encodings
  - Parallel attention/FFN architecture
  - Complete QKV weight splitting support

- ✅ **StableLM** - Stability AI language models
  - Multiple variants (base, zephyr, code)
  - 1.6B to 12B parameter range
  - Grouped-query attention
  - RoPE with partial rotary factor

#### Encoder-Decoder Models
- ✅ **T5** - Text-to-Text Transfer Transformer
  - Small, Base, Large, 3B, 11B, XXL (11B) variants
  - Relative position bias
  - Shared embedding for encoder/decoder
  - SentencePiece tokenization
  - Complete encoder-decoder attention

- ✅ **BART** - Bidirectional and Auto-Regressive Transformers
  - Denoising autoencoder pretraining
  - Full encoder-decoder architecture
  - Suitable for sequence-to-sequence tasks

#### Vision & Multimodal Models
- ✅ **Vision Transformer (ViT)** - Image classification transformer
  - Patch embeddings (16x16, 32x32)
  - Position embeddings for spatial information
  - Classification token
  - Multiple size variants (Tiny, Small, Base, Large)

- ✅ **CLIP** - Contrastive Language-Image Pre-training
  - Dual encoder architecture (text + vision)
  - Contrastive learning objective
  - Zero-shot image classification
  - ✅ Complete weight loading for text and vision encoders
  - ✅ HuggingFace model loading support
  - ✅ Load from path, lazy loading, memory-mapped modes

- ✅ **CogVLM** - Visual language model with temporal processing
  - Temporal encoder for video understanding
  - Multi-frame attention mechanisms
  - Vision-language alignment

- ✅ **BLIP-2** - Bootstrap Language-Image Pre-training v2
  - Querying Transformer (Q-Former)
  - Vision-language alignment
  - Frozen vision and language models

- ✅ **LLaVA** - Large Language and Vision Assistant
  - Vision encoder + LLM architecture
  - Visual instruction tuning
  - Multi-modal conversation

- ✅ **DALL-E** - Text-to-image generation
  - VQ-VAE image tokenization
  - Autoregressive generation
  - Discrete codebook

- ✅ **Flamingo** - Visual language model
  - Perceiver Resampler
  - Cross-attention between vision and language

#### State-Space & Linear Attention Models
- ✅ **S4** - Structured State Space model
  - HiPPO initialization (LEGS, LEGT, LAGT, Fourier)
  - Efficient long-range dependencies
  - O(N log N) complexity with FFT
  - Diagonal plus low-rank structure

- ✅ **Mamba** - Selective state-space model
  - Selective scan mechanism
  - Linear time complexity
  - Hardware-efficient implementation
  - Superior long-context performance

- ✅ **RWKV** - Receptance Weighted Key Value
  - Linear attention mechanism
  - Recurrent and parallelizable
  - O(N) time and space complexity
  - Time-mixing and channel-mixing

- ✅ **RetNet** - Retention mechanism
  - Multi-scale retention
  - O(N) inference complexity
  - Chunk-based processing
  - Parallel and recurrent modes

- ✅ **Hyena** - Implicit long convolutions
  - Subquadratic complexity
  - FlashFFT integration
  - Long-context optimization
  - Data-controlled implicit filter

#### Specialized Models
- ✅ **Code-Specialized Models** - Optimized for code generation
- ✅ **Math-Specialized Models** - Mathematical reasoning
- ✅ **Recursive Transformers** - Recursive attention patterns
- ✅ **Spiking Neural Networks** - Neuromorphic computing
- ✅ **Neural Turing Machines** - External memory
- ✅ **Hopfield Networks** - Modern continuous Hopfield
- ✅ **Quantum Transformers** - Quantum-inspired attention

---

### Hardware Acceleration

#### CUDA Backend
- ✅ **Custom Fused Kernels**
  - Fused GELU activation (exact and approximate)
  - Fused bias + activation (ReLU, GELU, SiLU, Tanh)
  - Optimized for NVIDIA GPUs
  - Dynamic kernel compilation

- ✅ **cuBLAS Integration** - Optimized matrix operations
- ✅ **Memory Management** - Efficient GPU memory allocation
- ✅ **Multi-GPU Support** - NCCL for collective operations

#### ROCm/HIP Backend
- ✅ **AMD GPU Support** - Full ROCm/HIP integration
- ✅ **Custom HIP Kernels** - Fused operations for AMD architecture
- ✅ **Memory Management** - Efficient HIP memory APIs
- ✅ **Synchronization** - Proper device synchronization

#### Metal Backend (Apple Silicon)
- ✅ **MPS Integration** - Metal Performance Shaders
- ✅ **Unified Memory** - Efficient memory management
- ✅ **Custom Shaders** - Metal Shading Language kernels
- ✅ **Flash Attention** - MPS graph operations
- ✅ **Platform Support** - macOS 10.15+, iOS 13+

#### Intel oneAPI Backend
- ✅ **DPC++ SYCL** - Data Parallel C++ kernel compilation
- ✅ **oneDNN** - Deep Neural Network Library integration
- ✅ **oneMKL** - Math Kernel Library for linear algebra
- ✅ **Multi-Device** - CPU, GPU, FPGA support
- ✅ **USM** - Unified Shared Memory management

#### Google XLA Integration
- ✅ **HLO Compilation** - High-Level Operations to optimized code
- ✅ **Multi-Platform** - CPU, GPU, TPU execution
- ✅ **Shape Inference** - Automatic output shape inference
- ✅ **Optimization** - Platform-specific optimizations

#### TPU Backend
- ✅ **Multi-Generation** - v2, v3, v4, v5, v5e support
- ✅ **Systolic Array** - Optimized for TPU architecture
- ✅ **BFloat16** - Native bfloat16 precision
- ✅ **HBM Management** - High Bandwidth Memory optimization

#### RISC-V Vector Extensions
- ✅ **RVV 1.0 Compliance** - Full specification support
- ✅ **Vector Length Agnostic** - VLEN 128-1024 bits
- ✅ **LMUL Support** - Vector register grouping
- ✅ **Vector Operations** - Arithmetic, logical, shift, reduction

#### Vulkan Compute
- ✅ **Cross-Platform** - Works on multiple OS/GPU vendors
- ✅ **Compute Shaders** - GLSL-based compute kernels
- ✅ **Memory Management** - Vulkan buffer management

#### Flash Attention
- ✅ **All Backends** - Implemented across CUDA, ROCm, Metal, Vulkan
- ✅ **Memory Efficient** - O(N) memory complexity
- ✅ **IO Aware** - Optimized for GPU memory hierarchy

---

### Tokenizers (trustformers-tokenizers)

#### Implementations
- ✅ **BPE** - Byte-Pair Encoding (GPT-2 style)
  - Merge operations with vocabulary
  - Byte-level encoding
  - Regex-based pre-tokenization

- ✅ **WordPiece** - BERT-style subword tokenization
  - Greedy longest-match first algorithm
  - Special token handling ([CLS], [SEP], [MASK])
  - Vocabulary with ## prefix for continuations

- ✅ **SentencePiece (Unigram)** - T5-style tokenization
  - Unigram language model
  - Reversible tokenization
  - Language-agnostic

- ✅ **Character-Level** - Simple character tokenization
- ✅ **AutoTokenizer** - Automatic detection from model name/path

#### Features
- ✅ **Encoding/Decoding** - Bidirectional text ↔ token IDs
- ✅ **Batch Processing** - Efficient multi-text tokenization
- ✅ **Padding/Truncation** - Length normalization
- ✅ **Special Tokens** - CLS, SEP, PAD, MASK, UNK handling
- ✅ **Attention Masks** - Automatic mask generation
- ✅ **Token Type IDs** - Segment embeddings support

#### Training & Analysis
- ✅ **Training from Files** - Build vocabulary from corpus
- ✅ **Training from Iterator** - Streaming vocabulary building
- ✅ **Vocabulary Intelligence**
  - Semantic clustering and redundancy detection
  - Compression efficiency analysis
  - Cross-lingual coverage assessment
  - Domain adaptability scoring
  - Evolution tracking

#### Python Bindings
- ✅ **PyO3 Integration** - Native Python extension
- ✅ **Maturin Build** - pip-installable package
- ✅ **Pythonic API** - Wrapper classes matching HF Tokenizers
- ✅ **Training Interface** - TokenizerTrainer in Python
- ✅ **Analysis Tools** - Coverage, benchmarking, profiling

---

## Known Limitations

### Security Advisories
- [ ] rsa 0.9.10 — Marvin Attack (RUSTSEC-2023-0071) — no upstream fix available; dep comes via azure_core transitive chain (trustformers-serve → azure_core → rsa). Monitor for upstream resolution.
- [ ] rustls-webpki 0.101.7 — CVEs (RUSTSEC-2023-0052, RUSTSEC-2023-0053, RUSTSEC-2024-0357) — transitive dep via trustformers-serve → aws-smithy-http-client → rustls v0.21.12 → rustls-webpki 0.101.7; cannot be upgraded until AWS SDK drops rustls 0.21 dependency. Monitor aws-smithy-http-client for updates.
- [ ] pyo3 0.28.x — RUSTSEC-2026-0176 / RUSTSEC-2026-0177 — trustformers-py is pinned to pyo3 "0.28" because scirs2-core 0.5.0 requires pyo3 = "0.28.3" (links="python" conflict prevents dual versions). The workspace (trustformers-tokenizers optional dep) was upgraded to pyo3 0.29. Unblock trustformers-py upgrade once scirs2-core publishes with pyo3 0.29 support.

### Platform Limitations
- **Metal Flash Attention:** Requires macOS 10.15+ or iOS 13+
- **TPU Backend:** Requires Google Cloud TPU access
- **Some Hardware Backends:** Platform-specific driver requirements

---

## Future Enhancements

### 🚨 CRITICAL PRIORITY: SciRS2 Policy Compliance & Performance (2025-12-19)

**Status**: ✅ **100% Policy Compliant** - 🔴 **Performance Blocked on SciRS2-Core MPSGraph**

**Performance Status**: ~1 tok/sec vs 50-200 tok/sec target (PyTorch+MPS parity)

**Root Cause Analysis** (Audit completed 2025-12-19):
- ✅ **NOT** due to policy violations - TrustformeRS is 100% compliant
- ✅ Basic MPS working - 100-500x matmul speedup via `scirs2_core::gpu::backends::MPSOperations`
- ❌ **BLOCKER**: MPSGraph not implemented in scirs2-core (all methods return "not yet implemented")
- ❌ Missing automatic kernel fusion (attention, GeLU, LayerNorm) - additional 10-50x speedup

**Audit Results** (145,823 lines audited):
- ✅ ndarray: 0 direct imports, 54 qualified paths ALL via `scirs2_core::ndarray::*`
- ✅ rand: 0 direct imports, all via `scirs2_core::random`
- ✅ rayon: 0 direct imports, all via `scirs2_core::parallel_ops`
- ✅ Tests: 100% compliant
- ✅ Features: All required features enabled (gpu, metal, mpsgraph, linalg, parallel, simd)
- ✅ BLAS: Accelerate framework configured via scirs2-core

#### Parallel Tracks

**Track A: SciRS2-Core MPSGraph Implementation** ⏳ **BLOCKING ITEM** (scirs2-core team)
- [ ] Implement MPSGraph in `~/work/scirs/scirs2-core/src/gpu/backends/metal_mpsgraph.rs`
  - Priority 1: `scaled_dot_product_attention()` (10-50x speedup, most critical)
  - Priority 1: `matmul()` (5-10x vs basic MPS)
  - Priority 1: `softmax()` (10-20x)
  - Priority 2: `gelu()`, `silu()` with operator stitching
  - Priority 2: `layer_norm()`, `rms_norm()` with fusion
  - Priority 3: `rope()` (rotary position embeddings)
- [ ] Release scirs2-core 0.3.0 (full release, graduation from rc.3)
- [ ] Benchmark and verify 50+ tok/sec performance target

**Implementation Request**: `~/work/requests/MPSGRAPH.md` (978 lines, ready for SciRS2 team)
- Comprehensive technical requirements
- PyTorch MPS reference implementation locations
- 3-week implementation plan (target: 2026-01-09)
- Quality gate: 50+ tok/sec verified before release

**Track B: TrustformeRS Policy Compliance** ✅ **COMPLETE**
- ✅ trustformers-core uses `scirs2_core::ndarray`, `scirs2_core::random`, `scirs2_core::parallel_ops`
- ✅ Inline qualified paths verified (all 54 instances compliant)
- ✅ scirs2-core features enabled: `gpu`, `metal`, `mpsgraph`, `linalg`, `parallel`, `simd`
- ✅ BLAS backend verified (Accelerate framework via scirs2-core)
- ✅ Compliance audit complete (zero violations found)
- ✅ Cargo.toml updated (mpsgraph feature enabled for macOS)

**Next Actions**:
1. **Blocked**: Await scirs2-core 0.3.0 release with MPSGraph implementation (Track A)
2. **Then**: Update TrustformeRS to scirs2-core 0.3.0
3. **Then**: Verify 50+ tok/sec performance on rinna-1b model
4. **Then**: Release TrustformeRS 0.3.0 (graduation from alpha)

**Performance Roadmap**:
- Current: ~1 tok/sec (basic MPS working)
- Target: 50-200 tok/sec (requires MPSGraph from scirs2-core)
- Quality Gate: Beta.1 requires verified 50+ tok/sec

**See**: `SCIRS2_INTEGRATION_POLICY.md` for policy details

---

### High Priority
- ✅ Complete CLIP text/vision encoder weight loading (COMPLETED - CLIPEncoderConfig trait, load_weights_chunked, 7 new tests)
- ✅ Conv2D forward pass: Full im2col+matmul with groups/dilation/stride/padding, 12 new tests
- ✅ DPO training loss: sigmoid DPO/KTO loss, log_softmax in get_batch_logps, preference accuracy metric, 8 new tests
- ✅ Safety filter stack overflow: Boxed large fields in SafetyFilter, all 25 safety tests pass
- ✅ Flamingo tensor shape: Fixed gate_proj dimension mismatch, Tensor::contiguous() method
- ✅ Debug integration hang: Fixed mutex deadlock in MemoryProfiler::generate_report()
- ✅ Topology analyzer + NUMA: Platform-aware detection (Linux/macOS/fallback), 28 new tests
- ✅ Custom eviction policy: Replaced panic! with LRU fallback + tracing::warn
- ✅ ROCm/GPU stubs: Replaced eprintln with tracing::debug, CPU fallbacks for conv2d/attention/flash_attention
- ✅ **AudioClassificationPipeline** — Audio input → label with wav2vec2/Whisper support
- ✅ **ImageClassificationPipeline** — Image input → label with ViT/CLIP support
- ✅ **AutoModelForAudioClassification** — Auto class for audio classification
- ✅ **AutoModelForImageClassification** — Auto class for image classification
- ✅ **OpenVINO Backend Stubs Resolved** — 11 stubs cleaned up
- ✅ **LoRA Fine-tuning Helpers** — LoraConfig, LoraLinear, merge/unmerge operations
- ✅ **Adapter Fine-tuning Helpers** — BottleneckAdapter with residual connection
- ✅ **Evaluation Metrics** — BLEU-1/2/4, ROUGE-1/2/L, F1, exact match, perplexity
- ✅ **DeiT (Data-efficient Image Transformers)** — with distillation token, 4 variants (Tiny/Small/Base/Large)
- ✅ **Swin Transformer** — Hierarchical with shifted windows, 4 variants (Tiny/Small/Base/Large)
- ✅ **Perfetto Trace Export** — chrome://tracing compatible JSON export
- ✅ **Tracy Profiler Export** — Tracy CSV format export
- ✅ **Lock-Free Ring Buffer** — SPSC atomic ring buffer for profiling
- ✅ **WebSocket/SSE Streaming Dashboard** — Real-time metrics streaming
- ✅ **trustformers-mobile Test Suite** — 512 new integration tests (1 → 513)
- ✅ **Parallel scaling test threshold** — Lowered to 0.05x to prevent false failures on loaded CI
- [ ] Enhanced multimodal model support and integration examples
  - **Refinement needed:** Which modalities? (audio+vision, video, document understanding?) What integration examples are needed?
- [ ] Latest research architectures (as they emerge)
- [ ] Advanced generation examples and tutorials

### Performance Optimizations
- [ ] Further SIMD optimizations via SciRS2
- [ ] Advanced kernel fusion strategies
- [ ] Enhanced memory pooling
- [ ] Dynamic batching improvements

### Deployment
- [ ] Enhanced edge device support (microcontrollers, embedded)
- [ ] Browser-based fine-tuning capabilities
- [ ] Improved federated learning infrastructure
- [ ] Better mobile quantization strategies

### Developer Tools
- [ ] Interactive model architecture explorer
- [ ] Enhanced debugging visualizations
- [ ] Automated hyperparameter search
- [ ] Performance regression detection dashboard

---

## Proposed follow-ups
- **scirs2-core 0.3.0 MPSGraph (externally blocked):** The 3 checkbox items under Track A are awaiting upstream scirs2-core 0.3.0 release.
- **`trustformers-js` workspace governance gap:** The `trustformers-js/` directory is not declared in root `Cargo.toml` workspace `members` or `exclude`. Consider: add to `exclude` (explicit), or create a bridge Cargo.toml for the npm monorepo.
- **Branch/version gap:** Resolved — workspace `Cargo.toml` and all package files are now at version `0.1.3`.

---

## Development Guidelines

### Dependency Management
- **Workspace Dependencies:** All crates use `workspace = true`
- **Version Policy:** Always use latest stable versions from crates.io
- **SciRS2 Integration:** Strictly enforced (see SCIRS2_INTEGRATION_POLICY.md)
  - Only trustformers-core can use external dependencies directly
  - Other crates must use trustformers-core or scirs2-core abstractions
  - No direct imports of rand, ndarray, tokenizers, etc. in non-core crates

### Code Quality Standards
- **Naming:** snake_case for all identifiers (variables, functions, modules)
- **File Size:** Maximum 2000 lines per file (use splitrs for refactoring)
- **Error Handling:** Always use `Result<T, TrustformersError>` pattern
- **Testing:** Use `std::env::temp_dir()` for temporary file handling
- **Documentation:** Public APIs must have rustdoc with examples
- **No Warnings:** Code must compile with `cargo clippy -- -D warnings`

### Testing Requirements
- Unit tests for all public APIs
- Integration tests for model implementations
- Property-based tests for tensor operations
- Numerical parity tests with HuggingFace reference
- Performance benchmarks for critical paths

### Contributing
- See CONTRIBUTING.md for detailed contribution guidelines
- Use GitHub issue templates in `.github/ISSUE_TEMPLATE/`
- Follow the model implementation checklist for new architectures
- Ensure all tests pass before submitting PR
- Run `make check` (format, clippy, tests, docs) before commit

---

## Project Resources

### Documentation
- **Main README:** Project overview and quick start
- **SCIRS2_INTEGRATION_POLICY.md:** **MANDATORY** dependency policy
- **CONTRIBUTING.md:** Contribution guidelines
- **CLAUDE.md:** Development instructions for Claude Code
- **Architecture Guide:** docs/architecture.md
- **Migration Guides:** docs/migration/

### Build Commands
```bash
# Full workspace check (recommended before commit)
make check

# Run all tests
cargo nextest run --all-features

# Run tests for specific crate
cargo nextest run -p trustformers-core --all-features

# Format code
cargo fmt --all

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings

# Build documentation
cargo doc --all-features --no-deps

# Build release
cargo build --release --all-features
```

### Example Applications
TrustformeRS includes comprehensive examples demonstrating real-world usage:

#### Workspace Examples (examples/)
- **adaptive_inference_demo.rs** - Adaptive inference strategies
- **advanced_composition.rs** - Model composition techniques
- **basic_pipeline.rs** - Simple pipeline usage
- **batch_inference_example.rs** - Efficient batch processing with batch utilities
- **custom_backend_examples.rs** - Custom hardware backend integration
- **dynamic_batching.rs** - Dynamic batch sizing
- **ensemble_models.rs** - Model ensemble techniques
- **generation_advanced_example.rs** - Advanced text generation strategies
- **interactive_cli.rs** - Interactive command-line interface
- **realtime_streaming.rs** - Real-time streaming inference
- **tensorrt_demo.rs** - TensorRT integration
- **web_demo.rs** - Web-based demo applications

#### Trustformers Crate Examples (trustformers/examples/)
- **batch_inference_example.rs** - Batch inference patterns and optimization strategies
- **generation_advanced_example.rs** - Comprehensive text generation showcase
- **clip_multimodal_example.rs** - CLIP multimodal vision-language capabilities

Run examples with:
```bash
# Workspace examples
cargo run --example batch_inference_example --features "bert,gpt2"

# Trustformers crate examples
cargo run -p trustformers --example clip_multimodal_example --features "clip,vit"
```

### Community
- **Issues:** https://github.com/cool-japan/trustformers/issues
- **Discussions:** https://github.com/cool-japan/trustformers/discussions
- **License:** Apache-2.0

---

**Last Updated:** 2026-06-24 - v0.1.3 Development
**Next Milestone:** Beta 1.0 Release (pending scirs2-core 0.3.0 with MPSGraph for 50-200x Metal performance)
**Target Audience:** ML engineers, researchers, and production deployment teams

---

## v0.1.1 Release Summary (2026-04-25)

### New Model Architectures (2 added, total 29+)
- ✅ **DeiT (Data-efficient Image Transformers)** — distillation token, 4 size variants
- ✅ **Swin Transformer** — Hierarchical shifted-window attention, 4 size variants

### New Pipeline & Auto Classes
- ✅ **AudioClassificationPipeline** — wav2vec2/Whisper audio → label pipeline
- ✅ **ImageClassificationPipeline** — ViT/CLIP image → label pipeline
- ✅ **AutoModelForAudioClassification** — Auto class routing for audio classification
- ✅ **AutoModelForImageClassification** — Auto class routing for image classification

### Fine-tuning Infrastructure
- ✅ **LoRA Fine-tuning** — LoraConfig, LoraLinear with merge/unmerge, rank/alpha/dropout support
- ✅ **Adapter Fine-tuning** — BottleneckAdapter with residual connection, hidden dim configuration

### Evaluation Metrics
- ✅ **Text Quality** — BLEU-1/2/4 (n-gram precision with brevity penalty)
- ✅ **Summarization** — ROUGE-1/2/L (recall-oriented evaluation)
- ✅ **QA Metrics** — F1 token overlap, exact match
- ✅ **LM Evaluation** — Perplexity computation

### Profiling & Debug Enhancements (trustformers-debug)
- ✅ **Perfetto Trace Export** — chrome://tracing compatible JSON (trace viewer ready)
- ✅ **Tracy Profiler Export** — Tracy CSV format with zone timing
- ✅ **Lock-Free Ring Buffer** — SPSC atomic ring buffer for zero-overhead profiling
- ✅ **WebSocket/SSE Streaming Dashboard** — Real-time metrics broadcast
- ✅ **debug tests: 216 → 323** (+107 tests)

### Mobile Deployment Expansion (trustformers-mobile)
- ✅ **Comprehensive test suite: 1 → 513** (+512 tests covering all mobile subsystems)
- ✅ Battery optimization, thermal management, network optimization
- ✅ Device detection, memory pressure handling, privacy controls
- ✅ Model management, compression, aggregation integration tests

### Backend Cleanup
- ✅ **OpenVINO Backend** — 11 stub warnings resolved

### Bug Fixes
- ✅ **Parallel scaling test** — threshold lowered to 0.05x for loaded CI environments

### Hub & Cache Enhancements (trustformers)
- ✅ **Hub Upload** — `HubUploader` / `HubUploaderBuilder` for single file, multi-file, and directory uploads to HuggingFace Hub
- ✅ **Automatic Model Card Generation** — `ModelCard`, `ModelCardGenerator`, YAML front matter, benchmark tables, `to_markdown()` / `from_markdown()` / `save()` / `load()`
- ✅ **TTL/Versioned Caching** — `VersionedCache<K,V>` with LRU/LFU/TTL/Size eviction, per-entry TTL override, version invalidation, `Arc<RwLock>` thread safety

### Workstream O: Rich Error Diagnostics + Parallel Model Loading + Speech Recognition (trustformers)
- ✅ **Rich Error Diagnostics** (`trustformers::diagnostics`) — `DiagnosticContext`, `DiagnosticSeverity`, `RichError`, `ErrorSpan`, `Diagnosable` trait, `CommonDiagnostics` (8 built-in patterns: E001–E006, W001–W002), `DiagnosticReport` with JSON export; 15 tests
- ✅ **Parallel Model Loading** (`trustformers::loading`) — `ParallelWeightLoader`, `ParallelLoaderConfig`, `WeightChunk`, `LoadingProgress`, `LoadingStats`; concurrent shard loading via `std::thread::scope`, safetensors header parsing, progress callbacks, `load_model_parallel` convenience function; 11 tests
- ✅ **Speech Recognition Pipeline** (`trustformers::pipeline::speech_recognition`) — `SpeechRecognitionPipeline`, `SpeechRecognitionConfig`, `AudioInput` (raw/file/mel), `TranscriptionResult`, `TranscriptionSegment`, `SpeechTask` (Transcribe/Translate), `ReturnTimestamps` (None/Word/Sentence), Hann-windowed DFT mel spectrogram computation, `compute_mel_spectrogram` public API; 20 tests

### Workspace Metrics Delta (v0.1.0 → v0.1.1)
| Metric | v0.1.0 | v0.1.1 | Delta |
|--------|--------|------------|-------|
| Total tests | 5,007 | 5,404 | +397 |
| Rust SLoC | ~900,000 | ~917,800 | +17,800 |
| Architectures | 27+ | 29+ | +2 |
| trustformers-debug tests | 216 | 323 | +107 |
| trustformers-mobile tests | 1 | 513 | +512 |
| trustformers tests | ~1,740 | 717 | +46 new (diagnostics×15, parallel loader×11, speech recognition×20) |

---

**v0.1.0 Release Summary (2026-03-21):**
- ✅ 27+ transformer architectures: BERT, RoBERTa, ALBERT, DistilBERT, ELECTRA, DeBERTa, GPT-2, GPT-Neo, GPT-J, GPT-NeoX, LLaMA, Mistral, Gemma, Qwen, Phi-3, Falcon, StableLM, T5, ViT, CLIP, BLIP-2, LLaVA, DALL-E, Flamingo, Mamba, RWKV, S4
- ✅ 5,007 tests with 100% pass rate across all crates
- ✅ ~900,000+ SLoC, 100% Pure Rust (COOLJAPAN Policy)
- ✅ Full production serving: REST/gRPC/GraphQL, dynamic batching, Kubernetes
- ✅ RLHF and DPO training support
- ✅ WebAssembly browser inference
- ✅ Mobile deployment (iOS/Android)
- ✅ Safety filtering pipeline
- ✅ SciRS2 policy compliance audit (145,823 lines) - 100% compliant, zero violations
- ✅ BLAS integration verified: Accelerate framework via scirs2-core
- ✅ Multi-backend: CUDA, Metal, ROCm, WebGPU, Vulkan, OpenCL, TPU
- Awaiting scirs2-core 0.3.0 release for 50-200x MPSGraph Metal performance improvement

## Stubs to implement (added 2026-06-12 by /cooljapan-stub-check)

- [ ] `trustformers-serve`: `build.rs:2` — Proto compilation is disabled because `tonic-build` 0.14 API changed; investigate new builder pattern or pre-generated proto files and restore gRPC stub generation.
  - Priority: P2 | Scope: medium | Hint: none

- [ ] `trustformers-serve`: `src/lib.rs:81,335` — Two proto-generated modules are commented out pending build.rs fix; re-enable once proto compilation is restored.
  - Priority: P2 | Scope: trivial | Hint: none

- [x] `trustformers-serve`: `src/resource_management/gpu_manager.rs` — Entire GPU manager module is commented-out because types (`GpuDeviceCapability`, `GpuLoadBalancer`, `GpuPerformanceTrend`, `GpuMonitoringSystem`, etc.) do not exist at the expected import paths; resolve import paths or define the missing types and uncomment.
  - Priority: P2 | Scope: large | Hint: none
  - Locations: :73-131 (large commented-out import block)

- [x] `trustformers-serve`: `src/performance_optimizer/test_characterization/concurrency_detector/deadlock_analyzer.rs` — Deadlock analyzer accumulates API-mismatch workarounds: `PotentialDeadlock` fields (`confidence`, `probability`, `deadlock_type`, `impact`) no longer exist; `DeadlockRiskLevel` changed from enum to struct; lock-acquisition/release signatures changed; fix all field accesses and call sites to match current API.
  - Priority: P2 | Scope: medium | Hint: none
  - Locations: :54,:56,:93,:215,:223,:267,:310,:341,:349,:398,:421,:430,:497,:512,:532,:552 (16 mismatch sites)

- [x] `trustformers-serve`: `src/performance_optimizer/test_characterization/concurrency_detector/thread_analyzer.rs` — `ThreadAnalysis` API changed (`thread_id` removed, `detected_patterns` is now `Vec<String>`); `ExecutionTrace` lost `duration` and `result` fields; `InteractionType`/`PatternType`/`PatternImpact` enum variants renamed; fix all mismatches.
  - Priority: P2 | Scope: medium | Hint: none
  - Locations: :39,:69,:123,:124,:193,:286,:300,:317,:467,:477,:484,:504,:521 (13 mismatch sites)

- [x] `trustformers-serve`: `src/performance_optimizer/test_characterization/concurrency_detector/lock_analyzer.rs` — `ContentionFrequencyAnalysis::new` and `WaitTimeAnalysis::new` signatures changed; `ExecutionTrace` lost `result` field; `contention_summary`/`latency_bounds` require conversion from analyzer structs.
  - Priority: P2 | Scope: small | Hint: none
  - Locations: :36,:39,:120,:121,:157,:173 (6 mismatch sites)

- [x] `trustformers-serve`: `src/performance_optimizer/test_characterization/concurrency_detector/conflict_detector.rs` — `ResourceConflict.resources` field removed (only `resource_id` remains); `ConflictType` enum variants simplified; `ConflictHistory::new()`/`ResourceDependencyGraph::new()` return `Result` now; fix all construction and field access sites.
  - Priority: P2 | Scope: small | Hint: none
  - Locations: :53,:55,:80,:213,:217,:302,:318,:346 (8 mismatch sites)

- [x] `trustformers-serve`: `src/performance_optimizer/test_characterization/concurrency_detector/pattern_detector.rs` — `ScalabilityRating`, `ScalingBehavior`, and `OptimizationComplexity` changed from enums to structs; fix all match arms and construction sites.
  - Priority: P2 | Scope: small | Hint: none
  - Locations: :204,:230,:278,:322,:378,:428,:576,:588 (8 mismatch sites)

- [x] `trustformers-serve`: `src/performance_optimizer/test_characterization/concurrency_detector/sharing_analyzer.rs` — `ResourceSharingCapabilities` lost `sharing_safety_level` enum field and `implementation_complexity`; `performance_overhead` renamed to `sharing_overhead`; fix field names.
  - Priority: P2 | Scope: trivial | Hint: none
  - Locations: :185,:190,:194,:212 (4 sites)

- [x] `trustformers-serve`: `src/performance_optimizer/test_characterization/concurrency_detector/risk_assessment.rs` — `PreventiveMitigation::new` and `ReactiveMitigation::new` required args changed; `assess_risk`, `is_applicable`, `generate_mitigation` argument counts changed; fix all call sites.
  - Priority: P2 | Scope: small | Hint: none
  - Locations: :44,:46,:74,:96,:265,:267 (6 mismatch sites)

- [x] `trustformers-serve`: `src/performance_optimizer/test_characterization/concurrency_detector/detector.rs` — `PatternEstimationConfig` → `EstimationConfig` type mismatch; `SharingCapability` (enum) vs `ResourceSharingCapabilities` (struct) conversion missing; `SynchronizationRequirements`/`ConcurrencyRequirements` field renames; `IsolationLevel::Process` variant missing.
  - Priority: P2 | Scope: medium | Hint: none
  - Locations: :83,:219,:228,:319,:340,:389,:451,:456,:461 (9 mismatch sites)

- [x] `trustformers-serve`: `src/performance_optimizer/performance_modeling/mod.rs:207` — Trained model cannot be stored in `active_models` because field uses `Box` while insertion needs `Arc`; refactor `active_models` to `Arc<dyn PerformanceModel>` and fix the train path.
  - Priority: P2 | Scope: small | Hint: none

- [x] `trustformers-serve`: `src/test_performance_monitoring/historical_data/types.rs` — Multiple data-lifecycle methods (compress, optimize, check_policy, cleanup, archive, retrieve, cache_lookup, query_execute, cache_store, lifecycle_eval) all return `Ok(())`/default stubs; implement each.
  - Priority: P2 | Scope: large | Hint: none
  - Locations: :117,:122,:196,:201,:271,:280,:396,:401,:417,:881 (10 stub methods)

- [ ] `trustformers-c`: `src/containers/deployment.rs:57` — `DockerImageConfig` conversion from `containers::types::DockerImageConfig` to `docker::DockerImageConfig` is a placeholder; implement the struct field mapping.
  - Priority: P2 | Scope: small | Hint: none

- [ ] `trustformers-c`: `src/cloud/aws_lambda.rs,azure_functions.rs,google_cloud_functions.rs` — Cloud function handlers set `TrustformersModel` and `TrustformersPipeline` handles to `0` (null) and return placeholder JSON; wire real model loading and pipeline execution.
  - Priority: P2 | Scope: large | Hint: none
  - Locations: aws_lambda.rs:204,208,324,342,358,383 / azure_functions.rs:285,289,418,424,528 / google_cloud_functions.rs:271,274,431,437,494

- [ ] `trustformers-c`: `src/utils.rs:236` and `src/utils_impl/mod.rs:184` — Tests reference old `validate_string_comprehensive` / `validate_string` / `safe_c_string` signatures that no longer exist; update tests to current API or restore the functions.
  - Priority: P2 | Scope: small | Hint: none

- [ ] `trustformers-core`: `src/gpu_ops/cuda/cuda_split/cuda_backend_ext.rs:457` — `run_fused_transformer_layer` executes operations individually instead of fused; implement a fully fused LayerNorm+QKV+RoPE+Attention+Proj+Residual CUDA kernel path.
  - Priority: P2 | Scope: large | Hint: none

- [ ] `trustformers-wasm`: `src/compute/gpu_tensor.rs:52,85` — WebGPU backend initialization is a stub (no device creation); `Rc<RefCell<>>` wrapper for interior mutability not applied; implement WebGPU device/queue setup and wrap backend.
  - Priority: P2 | Scope: medium | Hint: none

- [x] `trustformers`: `tests/compatibility_tests.rs:34,107,203,349` — Four tests are `#[ignore]`d waiting for `GlobalMemoryPool`, `ZeroCopyTensorView`, and `GlobalProfiler`; either implement these types or delete the placeholder tests.
  - Priority: P2 | Scope: medium | Hint: none
  - Locations: GlobalMemoryPool :34,:107, ZeroCopyTensorView :203, GlobalProfiler :349

- [x] `trustformers-debug`: `src/interpretability_tools.rs:27,34` — Interpretability-tools module is entirely commented out waiting for the module to be implemented; implement `AttentionVisualizer` and `FeatureAttributor` (or the equivalent current API) and re-enable.
  - Priority: P2 | Scope: large | Hint: none

## Stubs to implement (added 2026-06-22 by /cooljapan-stub-check)

- [ ] **trustformers** `trustformers-py`: `src/auto.rs:96` — `TODO`: `"rwkv" | "mamba" => { // State-space models - for now use BERT as fallback`
  - **Priority:** P2  **Scope:** large  **Cross-project:** none
  - **Approach:** Route the `rwkv`/`mamba` arm to the real state-space loaders (`PyRwkvModel`/`PyMambaModel`) instead of the BERT fallback; wire `from_pretrained` to load state-space weights.
  - **Risk:** Known-wrong correctness bug — silently returns a BERT model for RWKV/Mamba checkpoints, producing garbage outputs; needs the model crates' loaders to exist and be Python-exposed.

- [ ] **trustformers** `trustformers-core`: `src/ops/activations.rs:42` — `TODO`: `let device_id = 0; // TODO: Get from tensor metadata`
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Read the CUDA device id from `cuda_data` tensor metadata (e.g. `cuda_data.device_id()`/its `Device::CUDA(id)`) instead of hardcoding `0` before `get_cuda_backend`.
  - **Risk:** Wrong on multi-GPU — activations dispatched to device 0 regardless of where the tensor lives, causing cross-device faults or silent corruption.

- [ ] **trustformers** `trustformers-core`: `src/tensor/utils.rs:440` — `TODO`: `// For now, just return a clone (buffer is reference counted) ... device-to-device transfer (Metal)`
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Implement a real Metal device→device buffer copy (blit encoder) when source and destination Metal devices differ, instead of cloning the reference-counted buffer.
  - **Risk:** Multi-device Metal transfers alias the source buffer rather than copying, so data is not actually moved to the target device.

- [ ] **trustformers** `trustformers-core`: `src/tensor/utils.rs:562` — `TODO`: `// For now, just return a clone (buffer is reference counted) ... device-to-device transfer (CUDA)`
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Implement direct CUDA device→device copy (`cudaMemcpyPeer`/equivalent in the CUDA backend) when the destination device differs; avoid the host round-trip / clone shortcut.
  - **Risk:** Same as the Metal case — CUDA peer transfers silently alias instead of relocating data across GPUs.

- [ ] **trustformers** `trustformers-core`: `src/tensor/math_ops/arithmetic.rs:141` — `TODO`: `// Mixed Metal/CPU - convert to CPU for now // TODO: Could upload CPU tensor to GPU instead`
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** When exactly one operand is on GPU (Metal), upload the CPU operand to the GPU and run the op on-device, rather than downloading the GPU operand to CPU.
  - **Risk:** Perf/correctness — current path forces a GPU→CPU download per mixed op, defeating GPU residency and adding latency in hot arithmetic paths.

- [x] **trustformers** `trustformers-serve`: `src/resource_management/statistics.rs:507` — `TODO`: `active_resources: 0, // TODO: Calculate from system_stats`
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Derive `active_resources` from `system_stats` (e.g. count currently-allocated resource handles) instead of the hardcoded `0`.
  - **Risk:** Reported active-resource count is always 0, misleading dashboards/autoscaling decisions.

- [x] **trustformers** `trustformers-serve`: `src/resource_management/statistics.rs:508` — `TODO`: `peak_usage: recent_snapshots.len() as u64, // TODO: Calculate from system_stats`
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Compute true peak usage as the max observed usage across `recent_snapshots`/`system_stats`, not the snapshot count.
  - **Risk:** `peak_usage` currently equals the number of snapshots (a sampling artifact), not real peak resource consumption.

- [x] **trustformers** `trustformers-serve`: `src/performance_optimizer/performance_modeling/mod.rs:207` — `TODO`: `// TODO: Add trained model to active_models - requires refactoring to use Arc instead of Box`
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Refactor `active_models` storage from `Box<dyn PerformanceModel>` to `Arc<dyn PerformanceModel>`, then insert the freshly trained model instead of returning a default placeholder.
  - **Risk:** Training succeeds but the model is discarded (a default is returned), so trained performance models are never actually served.

- [ ] **trustformers** `trustformers-models`: `src/gpt2/model/model_blocks.rs:953` — `TODO`: `// TODO: Fused matmul+bias+GELU kernel for Metal GPU`
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Integrate the existing Metal fused matmul+bias+GELU kernel (`gpu_ops/metal/metalbackend_matmul_gelu_f32_group.rs`) into the Linear layer with GPU-resident buffers, replacing the current MPS/Accelerate path.
  - **Risk:** Perf only (current MPS/Accelerate path is correct); integration requires GPU-resident buffer ops in the Linear layer.

- [ ] **trustformers** `trustformers-models`: `src/gpt_neox/model.rs:165` — `TODO`: `// Temporary fallback: Convert Metal/CUDA tensors to F32 // TODO: Implement full Tensor::Metal/CUDA support in Attention`
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Implement native Metal/CUDA tensor support in GPT-NeoX attention (QKV split + RoPE on-device) instead of downcasting GPU tensors to CPU F32.
  - **Risk:** Perf/correctness — GPU GPT-NeoX attention silently round-trips to CPU F32, losing GPU residency and precision flexibility.

- [ ] **trustformers** `trustformers-serve`: `src/resource_management/gpu_manager/manager.rs:259` — `TODO`: `// TODO: In production, add real GPU discovery:`
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Implement real GPU discovery (NVIDIA via NVML/Pure-Rust probe, AMD via ROCm, cross-vendor via the project's compute backends) and driver-compatibility checks instead of the placeholder enumeration.
  - **Risk:** Serve cannot see actual GPUs in production; scheduling/placement runs on stubbed device info. (Real bindings should follow the Pure-Rust/noffi policy.)

- [ ] **trustformers** `trustformers-c`: `src/tensor.rs:1153` — `TODO`: `/// Clamp tensor values to [min, max] range (NOT YET IMPLEMENTED) // TODO: Implement clamp() method`
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Add a `clamp(min, max)` method to `trustformers_core::tensor::Tensor`, then uncomment and wire the `trustformers_tensor_clamp` C export.
  - **Risk:** C API advertises clamp but the entry point is commented out; consumers cannot clamp via FFI until the core method lands.

### Known external-blocked placeholders (not actionable)

- `trustformers-serve` `build.rs:2` — Proto compilation disabled: `tonic-build`/`tonic-prost-build` 0.14 API changed; blocked on settling the new builder pattern (also gates `src/lib.rs:81,335` proto module re-enables). Tracked in the 2026-06-12 section.
- `trustformers-core` `src/gpu_ops/rocm.rs:131` — `// TODO: Implement actual HIP kernel execution when HIP bindings are available`; falls back to CPU. Blocked on Pure-Rust/HIP bindings.
- `trustformers-core` `src/gpu_ops/cuda/cuda_split/cuda_backend_ext.rs:457` — `// TODO: Implement fully fused transformer kernel`; needs a hand-written fused CUDA kernel. Already tracked (large) in the 2026-06-12 section.
- `trustformers-wasm` `src/compute/gpu_tensor.rs:52,85` — WebGPU device creation + `Rc<RefCell<>>` interior-mutability wiring; blocked on the WebGPU backend. Already tracked (medium) in the 2026-06-12 section.

## Deferred stubs surfaced by /stub-check (2026-06-24)

Real but non-actionable-now stubs found in workspace member crates during nagare Phase 2; each needs cross-cutting wiring, external resources, GPU hardware, or upstream fixes. Documented for future passes.

- [ ] `trustformers-serve`: `src/openai_compat/mod.rs:742,769` — route_chat/route_completion return hardcoded stub responses; need real model inference wired through serve (reason: crosscut, needs inference path + weights)
- [ ] `trustformers-serve`: `src/graphql.rs:181` — models() returns a single hardcoded entry; model_service not wired into GraphQL context (reason: crosscut)
- [ ] `trustformers-serve`: `src/model_management/manager.rs:47,56` — ModelInstance::infer() returns a placeholder string (reason: crosscut, needs inference)
- [ ] `trustformers-serve`: `src/performance_optimizer/real_time_metrics/optimization/advanced_algorithms.rs:150,285,406,523,651,707,777` — update_with_feedback is a no-op across 7 algorithms; AlgorithmStatistics lost feedback_count/positive_feedback/negative_feedback fields (reason: crosscut, type API drift)
- [ ] `trustformers-serve`: `src/performance_optimizer/real_time_metrics/mod.rs:167` — threshold module disabled (only stub impls); comment cites 1,700+ compile errors to restore from .bak2 (reason: oversized)
- [ ] `trustformers-serve`: `src/test_performance_monitoring/types/storage.rs:35` — StorageManager::get_report returns a stub Report (reason: crosscut)
- [ ] `trustformers-serve`: `src/test_performance_monitoring/types/reporting.rs:36` — ReportExporter::export_report returns a stub ExportResult without writing a file (reason: crosscut)
- [ ] `trustformers-serve`: `src/test_performance_monitoring/mod.rs:205-207,223` and `src/test_performance_monitoring/service.rs:111,115,120` — config fields (compliance_reporting/historical_data_config/event_config/alert_config) absent on the config types; API drift (reason: external/crosscut)
- [ ] `trustformers`: `src/auto/feature_extractors/vision.rs:251,291` — preprocess_image and extract_visual_features return zero vectors (reason: crosscut)
- [ ] `trustformers`: `src/hub_offline_packs.rs:356` — get_model_info returns a mock ModelInfo; needs a HuggingFace Hub HTTP call (reason: external)
- [ ] `trustformers`: `src/pipeline/conversational/config/presets.rs:409,462,488` — references AnalysisConfigBuilder/ReasoningConfigBuilder that may not exist; verify whether live or dead before implementing (reason: needs-clarification)
- [ ] `trustformers-mobile`: `src/react_native_fabric.rs:410` — execute_standard_inference returns a placeholder vec (reason: crosscut)
- [ ] `trustformers-optim`: `src/genie_stub.rs`, `src/sofo_stub.rs`, `src/lora_rite_stub.rs` — simplified GENIE/SOFO/LoRA-RITE optimizer steps; full research algorithms pending API-compat resolution (reason: research/needs-clarification)
- [ ] `trustformers-debug`: `src/data_export.rs:608` — export_sqlite falls back to JSON instead of a real SQLite file; should use oxisql-sqlite-compat per COOLJAPAN policy (reason: small-medium, deferred)
- [ ] `trustformers-debug`: `src/kernel_optimizer.rs:987,1035,1101,1180` — GPU kernel analyzers (LaunchConfig/MemoryAccess/ComputeUtilization/KernelFusion) return empty results (reason: gpu)

> **Skipped as already-tracked:** `trustformers-core` `src/gpu_ops/rocm.rs:131` (HIP kernel execution pending Pure-Rust HIP bindings, reason: external/gpu) — already documented under "Known external-blocked placeholders" in the 2026-06-22 section above.

## Planned campaigns — COOLJAPAN policy debt (scheduled 2026-06-24)

Two dedicated campaigns for pre-existing debt (not introduced by 0.1.3). Each is large and must run as a focused, per-crate, verify-as-you-go pass — NOT a quick fix. Sized from the nagare 0.1.3 policy-check + purity audit.

### Campaign A — No-unwrap + dead-code elimination

- [ ] **Campaign A — No-unwrap + dead-code elimination**
  - **Goal:** eliminate 1,844 `#[allow(...)]` (1,337 `dead_code`, 104 `unused_variables`, 61 `unused_imports`, 41 `unreachable_patterns`, 32 `deprecated`, ~160 clippy) and drive 2,421 production `unwrap`/`expect` (73 unwrap + 2,348 expect) toward zero, per the No-warnings + No-unwrap policies. Build/tests currently pass only because the allows silence the warnings.
  - **Scope:** workspace member `src/` only (exclude `#[cfg(test)]`/`tests/`). Per-crate `#[allow]` counts: debug 484, core 467, models 329, training 127, optim 110, serve 102, wasm 81, trustformers 62, mobile 42, tokenizers 30. Worst `unwrap`/`expect` files: mobile `profiler_impl.rs` 48, mobile `profiler_split/types.rs` 48, core `memory_leak_detector.rs` 41, mobile `adaptive_cache_manager.rs` 36, core `checkpoint/mapping.rs` 36, wasm `webgpu/types.rs` 35, core `hardware/registry.rs` 34, mobile `optimization/memory_pool.rs` 33, debug `realtime_dashboard.rs` 32, trustformers `memory_pool.rs` 31.
  - **Approach (phased, per-crate, leaf-first):** A1 — remove `#[allow(dead_code)]`; for each surfaced item: delete if truly dead, wire it up if it should be used, or make it `pub` if it's intended API. A2 — remove remaining `#[allow(unused_*/unreachable/deprecated/clippy)]`, fix root causes (unused imports/vars, update deprecated APIs). A3 — replace `unwrap()`/`expect()` in src with `Result`/`?`/`ok_or_else`/`unwrap_or_else`; keep a documented `expect` ONLY where the invariant is provably infallible.
  - **Risk:** removing `dead_code` allows may unmask code used only via macros/reflection — verify before deleting; deleting public-looking items can break downstream. Large effort; slice per-crate, PR-sized.
  - **Verify:** per crate `cargo clippy -p <crate> --all-features --all-targets -- -D warnings` clean WITHOUT the removed allows; `cargo nextest run -p <crate> --all-features` green.
  - **Suggested execution:** a dedicated `/recursive` or `/loop` campaign, one crate per slice, leaf-of-dep-graph first, capped iterations, never re-adding an `#[allow]`.

### Campaign B — Pure-Rust default-features dependency hygiene

- [ ] **Campaign B — Pure-Rust default-features dependency hygiene**
  - **Goal:** make DEFAULT features 100% Pure Rust (COOLJAPAN policy). Today every non-wasm member transitively compiles C libs in its default tree; only `trustformers-wasm` is clean. Our own crates are already compliant (use `oxiarc-*`, gate all heavy C backends) — this is third-party transitive leakage.
  - **Scope (default-tree offenders + who pulls them):** `aws-lc-sys` via `reqwest`→`rustls` (reqwest is a non-optional dep of `trustformers-core`); `onig_sys` via `tokenizers` (non-optional dep of core); `zstd-sys`/`zstd` via `jieba-rs`→`include-flate` (tokenizers); banned compression `flate2`+`miniz_oxide` via `plotters`/`image`/`png` (debug `visual`) and cloud-SDK HTTP stacks (serve); `brotli` via `lambda-web` (serve); `zip` (build-dep) via `utoipa-swagger-ui` (serve). NOTE: torch/cuda/opencl/vulkan/mpi/ffmpeg/kafka C deps are ALREADY correctly feature-gated (non-default) — leave them.
  - **Approach:** (1) `reqwest`/`rustls`: switch the default crypto provider off `aws-lc-rs` to a pure-Rust path (e.g. `reqwest` default-features=false + rustls with a RustCrypto provider), dropping `aws-lc-sys`. (2) `tokenizers`: disable its default `onig` feature, use the pure-Rust `fancy-regex` backend, dropping `onig_sys`. (3) `jieba-rs`/`include-flate`: replace or feature-gate so `zstd-sys` leaves default (or store data uncompressed / via `oxiarc`). (4) debug `visual` (plotters/image): move behind a non-default feature, or use a pure-Rust raster backend, dropping `flate2`/`miniz_oxide`. (5) serve: gate `lambda-web` behind a non-default `lambda` feature (drops `brotli`) and `utoipa-swagger-ui` behind a non-default `swagger-ui` feature (drops the `zip` build-dep).
  - **Risk:** changing the TLS crypto provider can affect HTTPS behavior; disabling `onig` can change tokenizer regex semantics for some models; gating debug-visual / serve lambda+swagger changes the default API/feature surface. Each change needs build+test verification and a default-vs-all-features tree diff.
  - **Verify:** `cargo tree -p <member> -e normal,build` shows no `-sys`/banned-compression in the DEFAULT tree for every member; `cargo build`/`cargo nextest run` green on default features; `--all-features` still green.
  - **Suggested execution:** a focused dependency-surgery pass, one offender at a time with a `cargo tree` diff + build/test gate after each; use `trustformers-wasm` (already pure) as the reference.
