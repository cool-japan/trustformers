# TrustformeRS TODO List

## Project Overview

TrustformeRS is a high-performance, memory-safe Rust implementation of Hugging Face Transformers.
The project provides a comprehensive ecosystem for transformer model development, training, and deployment
with support for 49+ architectures and multiple deployment targets.

### Version Information
- **Current Version:** 0.2.0 (Unreleased)
- **Previous Release:** 0.1.4 (Released 2026-07-02)
- **Status:** Active Development (v0.2.0)
- **License:** Apache-2.0
- **Repository:** https://github.com/cool-japan/trustformers

### Project Health (updated 2026-07-01 — full local workspace verification)
- ✅ **Compiles cleanly** across all crates — `cargo clippy --workspace --all-features --all-targets -- -D warnings` = 0 warnings/errors
- ✅ **Full test suite passes, machine-verified locally** — `cargo nextest run --workspace --all-features` = **18,102 passed, 0 failed** (119 skipped), ~565s, verified 2026-07-01. (No GitHub Actions Rust CI — intentionally not added, billable, per user policy — but the "100% pass" claim is now freshly and completely locally verified end-to-end, superseding Task 9's "not yet machine-verified" caveat below.)
- ✅ **0 rustdoc warnings** — `cargo doc --workspace --all-features --no-deps` with `RUSTDOCFLAGS="-D warnings"` = clean; `cargo fmt --all -- --check` = clean
- 🟡 **49+ architectures for CPU inference** — maturity varies; a batch of "fake implementation" defects was fixed 2026-06-19 (see audit), others may remain
- 🟡 **Compute is CPU / `f32`** — F16/BF16 are storage-only (upcast for math); GPU is wired only for GPT-2/RetNet, now via the Pure-Rust `oxicuda`/`oxicuda-metal` backends (the `cudarc` backend was fully removed in 0.1.4 — see audit Tasks 4 & 5, now historical)
- ✅ **100% Pure Rust** default-feature source for every crate except `trustformers-serve` (accepted exception: rustls/aws-lc-rs TLS); 2,983 Rust files, ~1.42M lines / ~1.18M lines of code (via `tokei`, 2026-07-01); optional GPU/hardware backends remain feature-gated FFI

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

> **Update (2026-07-01):** the GPU row above is historical (as of 2026-06-19) and now stale on backend naming — the `cudarc` CUDA backend referenced there was **fully removed** in 0.1.4, replaced by the Pure-Rust `oxicuda` (`oxicuda-blas`/`-dnn`/`-memory`/`-driver`); Metal similarly moved from scirs2 MPS to `oxicuda-metal`, dropping the `scirs2-core` GPU dependency. Both backends now carry 12 CPU↔CUDA golden-parity tests, runtime-verified on a real NVIDIA RTX A4000 (CUDA 12.0). GPU is still wired into per-model `forward` for GPT-2/RetNet only (2 of ~58+ models) — that part of the criticism still stands; see the 0.1.4 CHANGELOG and README's GPU Acceleration section for the current picture.

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
  `trustformers-core/src/gpu_ops/` (`cuda/oxicuda/` via the Pure-Rust `oxicuda` — **not** `cudarc`,
  which was fully removed in 0.1.4 — `metal/` via `objc2`/`oxicuda-metal`, `webgpu.rs`, `opencl.rs`,
  `rocm.rs`) — feature-gated, off by default. The high-level
  `trustformers-core/src/gpu.rs` (`GpuContext`, `GpuMemoryPool`, `detect_*_devices`) is a **CPU
  simulation**: `allocate()` just increments a counter and device detection returns hard-coded
  placeholder devices. Core `Tensor::matmul` **now dispatches** to `gpu_ops` for `Tensor::CUDA`/
  `Tensor::Metal` operands (added 2026-06-20, `#[cfg(feature=…)]`-gated) — but per-model `forward` GPU
  wiring is still only `gpt2`/`retnet`, and `gpu.rs` remains a CPU simulation. (Updated 2026-07-01:
  the CUDA path is now `gpu_ops/cuda/oxicuda/mod.rs`, carrying 10 `_parity`-suffixed CPU↔CUDA
  golden-parity unit tests plus 2 more counted in the 0.1.4 CHANGELOG's "12 golden-parity tests"
  figure; `cargo test -p trustformers-core --features cuda --lib gpu_ops::cuda::oxicuda`.)
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
- `trustformers/src/profiler.rs:368` (line drifted from :334; still open as of 2026-07-01) — `get_dashboard_url`
  returns a mock dashboard URL (benign, but labelled "in a real implementation…").
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

### Per-Crate Status (v0.1.4, 2026-07-01)

| Crate | Tests (approx.) | Status | SLoC |
|-------|-------|--------|------|
| trustformers-core | ~2,353 | Stable | 189,922 |
| trustformers-models | ~4,479 | Alpha | 185,954 |
| trustformers-training | ~930 | Stable | 87,017 |
| trustformers-tokenizers | ~500 | Stable | 48,701 |
| trustformers-optim | ~960 | Stable | 76,662 |
| trustformers-serve | ~4,321 | Stable | 331,151 |
| trustformers-debug | ~899 | Alpha | 100,417 |
| trustformers-wasm | ~130 | Stable | 53,361 |
| trustformers-mobile | ~742 | Alpha | 125,131 |
| trustformers | ~2,261 | Alpha | 131,961 |
| **Sum of per-crate figures above** | **~17,575** | | **~1,330,277** |
| **Full workspace** (`cargo nextest run --workspace --all-features`) | **18,102 passed**, 0 failed, 119 skipped | | 2,983 files / ~1.42M lines / ~1.18M lines of code |

Per-crate test counts are approximate (derived from the full workspace run, attributed per crate;
some integration/cross-crate tests aren't cleanly attributable to one row, hence the ~3% gap between
the per-crate sum and the full-workspace total — this is expected and not a discrepancy to chase).
SLoC is per-crate `tokei` line count (code+comments+blanks); the full-workspace row additionally
covers root-level examples/benches/tests/docs/bindings and the non-member `trustformers-c`/`-js`/`-py`
crates, hence it exceeds the sum of the 10 rows above. **This full-workspace count (18,102 passed, 0
failed) is machine-verified locally, 2026-07-01** — see Project Health above.

*(v0.1.0 baseline: 5,007 tests / ~900,000+ SLoC)*

### Per-Crate Status (v0.1.3, 2026-06-24) — historical

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

> **Correction (2026-06-19 audit):** the "5,358" figure is a stale/curated snapshot. The actual
> in-tree count is **~23,700** `#[test]` + `#[tokio::test]` functions. **Update (2026-07-01): this is
> now resolved** — a full `cargo nextest run --workspace --all-features` was run end-to-end and
> passed completely (18,102 passed, 0 failed, 119 skipped; see the v0.1.4 table above), so the
> "100% pass" claim is machine-verified locally even though no GitHub Actions Rust CI exists (by
> policy, not by gap — see Task 9 in the Code-Quality Audit above).

### Core Crates
1. **trustformers-core** - Fundamental tensor operations, layers, hardware acceleration (Stable)
2. **trustformers-models** - 49+ transformer model implementations (Alpha)
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

### Model Architectures (49+ Models)

> **Note (2026-07-01):** the count in this heading was corrected from the original "27+" to the
> current "49+" (confirmed via `trustformers-models/src/*` module directories and the README Model
> Zoo table). The detailed per-model bullets below only cover the original ~27; the ~22 added since
> (Falcon2, Gemma2, Granite, Hyena, InternLM2, Jamba, Jamba2, Linformer, LLaMA3.2, Mamba2, Nemotron,
> Performer, Phi4, Qwen2.5, RetNet, SD3, StarCoder2, Whisper, xLSTM, Yi, DeiT, Swin — see the v0.1.1/
> v0.1.2 release summaries below) exist and are feature-gated per-model but aren't individually
> written up here yet.

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

> **⚠️ Accuracy correction (2026-07-01):** the "✅ done" bullets in this whole subsection predate the
> 2026-06-19 Code-Quality Audit and the 0.1.4 oxicuda migration, and several are now known-inaccurate —
> do not treat this list at face value; the honest, current status is the README's "Honest maturity
> note" + "GPU Acceleration" section, and the Campaign C/D/E entries elsewhere in this file. Known
> corrections, verified while updating this file (2026-07-01):
> - **TPU Backend**: **not just unimplemented — the `tpu` feature and `kernels/tpu_impl.rs` have been
>   removed from `trustformers-core` entirely** (no file, no feature flag found in `Cargo.toml`). The
>   "Multi-Generation v2-v5e / Systolic Array / HBM Management" bullets below never reflected a real
>   TPU backend (matches the audit's "TPU listed as supported, empty feature flag" finding).
> - **ROCm/HIP Backend**: real backend scaffolding exists (`gpu_ops/rocm.rs`, `rocm = ["dep:libloading"]`)
>   but kernel execution still falls back to CPU pending Pure-Rust HIP bindings
>   (`gpu_ops/rocm.rs:131`, tracked as externally-blocked) — "Full ROCm/HIP integration" overstates it.
> - **Vulkan Compute**: real deps (`vulkano`/`vulkano-shaders`) and code exist, but per the README this
>   backend is feature-gated and experimental, not wired into model `forward`.
> - **Google XLA / Intel oneAPI / RISC-V Vector Extensions**: `xla = []`, `oneapi = []`, `riscv = []`
>   are all **empty feature flags with zero external dependencies** in `trustformers-core/Cargo.toml`
>   — the same red flag the audit used to catch the fake TPU feature. In-tree code exists
>   (`kernels/{xla_impl,oneapi_impl,riscv_impl}.rs`) but, unlike CUDA/Metal, none of it is mentioned as
>   "real" anywhere in the README or the audited GPU status — treat as unverified/likely-simulated
>   until a dedicated stub-check confirms otherwise; not re-audited line-by-line in this pass.
> - **CUDA Backend / Flash Attention** below are also stale on naming (`cudarc`/cuBLAS) — the real,
>   current, runtime-verified backend is the Pure-Rust `oxicuda` (see Campaign C/D/E and the 0.1.4
>   CHANGELOG); Flash Attention is verified for CUDA/Metal only, not "all backends."

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

> **SUPERSEDED (2026-07-01):** the "blocked on scirs2-core MPSGraph" premise below no longer applies.
> In 0.1.4, Metal GPU compute (matmul + resident attention) was migrated from scirs2 MPS to the
> Pure-Rust `oxicuda-metal` backend, and `trustformers-core`'s `scirs2-core` dependency no longer
> requests any GPU-related feature at all (`Cargo.toml`: `features = ["array", "random", "parallel",
> "simd"]`, plus an optional `linalg` — no `gpu`/`metal`/`mpsgraph`). Track A (waiting on the
> scirs2-core team) is therefore moot: TrustformeRS is no longer depending on scirs2-core for Metal
> GPU acceleration at all, and this is not tracked as an open blocker upstream. GPU-resident matmul on
> both CUDA (`oxicuda`) and Metal (`oxicuda-metal`) is now zero-copy and CPU-parity-tested (see the
> Code-Quality Audit update above and the 0.1.4 CHANGELOG). No verified end-to-end tok/sec figure on
> the new oxicuda-metal path exists yet in this repo's docs — do not assume the old "~1 tok/sec" or
> "50-200 tok/sec target" numbers below still apply; re-benchmark before citing a number. The
> historical record below (2025-12-19) is kept for context only.

**Status**: ✅ **100% Policy Compliant** - 🔴 **Performance Blocked on SciRS2-Core MPSGraph** *(historical — see superseded note above)*

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
- **scirs2-core 0.3.0 MPSGraph — SUPERSEDED (2026-07-01):** no longer a follow-up. 0.1.4 migrated Metal GPU compute to `oxicuda-metal` and dropped the `scirs2-core` GPU dependency entirely, so the 3 checkbox items under Track A are moot rather than "awaiting upstream release." See the SciRS2 Policy Compliance section above.
- **`trustformers-js` workspace governance gap:** still open as of 2026-07-01 — the `trustformers-js/` directory is not declared in root `Cargo.toml` workspace `members` or `exclude`. Consider: add to `exclude` (explicit), or create a bridge Cargo.toml for the npm monorepo.
- **Branch/version gap:** Resolved — workspace `Cargo.toml` and all package files are now at version `0.1.4`.
- **Broader oxicuda GPU coverage (new, 2026-07-01):** GPU-resident `forward` is still GPT-2/RetNet-only; the fused CUDA megakernel (CUDA-6) and on-device GPT-NeoX attention residency (CUDA-7) remain future work — see "Deferred stubs" below. **CUDA-7 UPDATE (2026-07-06):** on-device GPT-NeoX attention residency (prefill) landed this session (`cuda_resident_forward`); CUDA-6 fused megakernel is still open.

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

**Last Updated:** 2026-07-06 - v0.2.0 Development
**Next Milestone:** Beta 1.0 Release — no longer pending scirs2-core/MPSGraph (superseded by the 0.1.4 `oxicuda`/`oxicuda-metal` migration, see the SciRS2 Policy Compliance section above); remaining work is broadening GPU-resident `forward` coverage beyond GPT-2/RetNet and re-benchmarking end-to-end tok/sec on the new backend
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

> **Deferred-stubs campaign — DONE (2026-06-29 session):** Implemented 4 of 6 long-deferred stubs — gRPC proto + serving (`build.rs:2`, `lib.rs:81,335`), Metal fused matmul+bias+GELU (`trustformers-models` gpt2 `model_blocks.rs:953`), WebGPU init (`trustformers-wasm` `gpu_tensor.rs:52,85`), and Python RWKV/Mamba (`trustformers-py` `auto.rs:96`). The remaining 2 stay `[DEFERRED]` pending an NVIDIA-GPU box / real hardware: the fused CUDA transformer kernel (`cuda_backend_ext.rs:457`, tracked under Campaign C / C3) and the `trustformers-c` device/cloud bindings (`src/cloud/*`). **Verification:** `cargo nextest run --workspace` = **14,560 passed, 58 skipped, 0 failed** (+20 vs the prior 14,540 — the re-enabled gRPC tests).

- [x] `trustformers-serve`: `build.rs:2` — Proto compilation is disabled because `tonic-build` 0.14 API changed; investigate new builder pattern or pre-generated proto files and restore gRPC stub generation.
  - Priority: P2 | Scope: medium | Hint: none
  - **DONE (2026-06-29 session):** `build.rs` rewritten for the tonic 0.14 split API (`tonic-prost-build::configure().compile_protos(...)`); added the new `tonic-prost` codec dep — proto compilation restored.

- [x] `trustformers-serve`: `src/lib.rs:81,335` — Two proto-generated modules are commented out pending build.rs fix; re-enable once proto compilation is restored.
  - Priority: P2 | Scope: trivial | Hint: none
  - **DONE (2026-06-29 session):** re-enabled `pub mod grpc;` (lib.rs:81) + the re-export (lib.rs:335); 20 gRPC tests now run (also fixed a wrong test — `HealthStatus::Serving` is wire value 1, not 0). 4388 serve tests pass.

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

- [~] Implement proper DockerImageConfig conversion in deployment.rs (planned 2026-07-05)
  - Goal: generate_deployment_artifacts() produces real Dockerfile/compose/dockerignore/build-script content (build_args, env_vars, ports, volumes) instead of ignoring them.
  - Design: add `to_docker_builder_config(&containers::types::DockerImageConfig) -> containers::docker::DockerImageConfig` in deployment.rs; add a `BaseImage::Custom(String)` variant to `docker::BaseImage` (+ one match arm in `get_base_image_name`) since the target enum has no arbitrary-image-reference variant today; route through the existing `DockerImageBuilder::{generate_dockerfile, generate_docker_compose, generate_dockerignore, generate_build_script}` instead of hand-rolled `format!`.
  - Files: trustformers-c/src/containers/deployment.rs, trustformers-c/src/containers/docker.rs.
  - Tests: build a ContainerDeploymentConfig with populated build_args/ports/volumes; assert generated Dockerfile/compose actually contain those values.
  - Risk: do not add `use super::docker::*;` to deployment.rs — it already has `use super::types::*;` and both modules export a type named `DockerImageConfig`, causing an ambiguous-glob compile error (E0659). Use qualified paths.

- [ ] **[DEFERRED — needs NVIDIA GPU box / real hardware]** `trustformers-c`: `src/cloud/aws_lambda.rs,azure_functions.rs,google_cloud_functions.rs` — Cloud function handlers set `TrustformersModel` and `TrustformersPipeline` handles to `0` (null) and return placeholder JSON; wire real model loading and pipeline execution.
  - Priority: P2 | Scope: large | Hint: none
  - Locations: aws_lambda.rs:204,208,324,342,358,383 / azure_functions.rs:285,289,418,424,528 / google_cloud_functions.rs:271,274,431,437,494
  - **DEFERRED (2026-06-29 session):** the `trustformers-c` ASIC/cloud device + cloud-function bindings need FFI to real ASIC/cloud hardware/SDKs that aren't present here (and should follow the Pure-Rust/noffi policy); needs real hardware.

- [~] Reconcile utils.rs/utils_impl tests with current validation API (planned 2026-07-05)
  - Goal: re-enable the 3 disabled test blocks in utils.rs/utils_impl/mod.rs against the current (post-refactor) validation API.
  - Design: uncomment `#[cfg(test)] mod tests` (drop the `/* */` + stale TODO comments) in both files; fix call sites to the real signatures — `validate_string_comprehensive(s: *const c_char, max_len: Option<usize>, allow_unicode: bool)`, `validate_string` = alias for `validate_c_string_safe(s: *const c_char, max_len: Option<usize>)`, and wrap the `safe_c_string!` macro call inside a small `fn() -> *mut c_char` helper.
  - Files: trustformers-c/src/utils.rs, trustformers-c/src/utils_impl/mod.rs.
  - Tests: the re-enabled blocks themselves; fix test_utils_statistics's `>= 0` comparison on a usize field (clippy absurd_extreme_comparisons) while in the file.
  - Risk: leave the separate utils_impl::validate_model_name shadowing bug as a comment/flag only — out of scope for this fix.

- [ ] **[CUDA-6 — oxicuda fused megakernel, future work]** `trustformers-core` — fully fused LayerNorm+QKV+RoPE+Attention+Proj+Residual kernel path. **PATH SUPERSEDED (2026-06-29, Campaign E):** the old cudarc file `src/gpu_ops/cuda/cuda_split/cuda_backend_ext.rs:457` was **deleted** with the cudarc backend; the current oxicuda path executes ops individually (correct). A fused megakernel is now an oxicuda-backend (`oxicuda-dnn`) task, not a cudarc one.
  - Priority: P2 | Scope: large | Hint: implement via oxicuda-ptx template + a resident `*_gpu_to_gpu` chain; optimization only, not a correctness gap.

- [x] `trustformers-wasm`: `src/compute/gpu_tensor.rs:52,85` — WebGPU backend initialization is a stub (no device creation); `Rc<RefCell<>>` wrapper for interior mutability not applied; implement WebGPU device/queue setup and wrap backend.
  - Priority: P2 | Scope: medium | Hint: none
  - **DONE (2026-06-29 session):** implemented WebGPU device/queue setup via the crate's web-sys/js_sys path (`navigator.gpu` → request_adapter → request_device) and wrapped the backend in `Rc<RefCell<>>`. Host + wasm32 builds/clippy clean; runtime device creation needs a browser (compile-validated; falls back to CPU when no adapter).

- [x] `trustformers`: `tests/compatibility_tests.rs:34,107,203,349` — Four tests are `#[ignore]`d waiting for `GlobalMemoryPool`, `ZeroCopyTensorView`, and `GlobalProfiler`; either implement these types or delete the placeholder tests.
  - Priority: P2 | Scope: medium | Hint: none
  - Locations: GlobalMemoryPool :34,:107, ZeroCopyTensorView :203, GlobalProfiler :349

- [x] `trustformers-debug`: `src/interpretability_tools.rs:27,34` — Interpretability-tools module is entirely commented out waiting for the module to be implemented; implement `AttentionVisualizer` and `FeatureAttributor` (or the equivalent current API) and re-enable.
  - Priority: P2 | Scope: large | Hint: none

## Stubs to implement (added 2026-06-22 by /cooljapan-stub-check)

- [x] **trustformers** `trustformers-py`: `src/auto.rs:96` — `TODO`: `"rwkv" | "mamba" => { // State-space models - for now use BERT as fallback`
  - **Priority:** P2  **Scope:** large  **Cross-project:** none
  - **Approach:** Route the `rwkv`/`mamba` arm to the real state-space loaders (`PyRwkvModel`/`PyMambaModel`) instead of the BERT fallback; wire `from_pretrained` to load state-space weights.
  - **Risk:** Known-wrong correctness bug — silently returns a BERT model for RWKV/Mamba checkpoints, producing garbage outputs; needs the model crates' loaders to exist and be Python-exposed.
  - **DONE (2026-06-29 session):** found the entire Python binding layer had been disabled since v0.1.0 (pyo3 0.26→0.28 API drift); modernized + re-enabled it, created real `PyRwkvModel`/`PyMambaModel` classes, and routed the arm to them instead of the BERT fallback. (Weight loading uses the same pre-existing stubbed path as all py model classes.)

- [x] **trustformers** `trustformers-core`: `src/ops/activations.rs:42` — `TODO`: `let device_id = 0; // TODO: Get from tensor metadata`
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Read the CUDA device id from `cuda_data` tensor metadata (e.g. `cuda_data.device_id()`/its `Device::CUDA(id)`) instead of hardcoding `0` before `get_cuda_backend`.
  - **Risk:** Wrong on multi-GPU — activations dispatched to device 0 regardless of where the tensor lives, causing cross-device faults or silent corruption.
  - **DONE (2026-07-06, CUDA leak/device-id session):** GELU now reads `cuda_data.device_id()` and threads it through to `get_cuda_backend`/the resident op instead of hardcoding `0`.

- [ ] **trustformers** `trustformers-core`: `src/tensor/utils.rs:440` — `TODO`: `// For now, just return a clone (buffer is reference counted) ... device-to-device transfer (Metal)`
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Implement a real Metal device→device buffer copy (blit encoder) when source and destination Metal devices differ, instead of cloning the reference-counted buffer.
  - **Risk:** Multi-device Metal transfers alias the source buffer rather than copying, so data is not actually moved to the target device.

- [x] **trustformers** `trustformers-core`: `src/tensor/utils.rs:562` — `TODO`: `// For now, just return a clone (buffer is reference counted) ... device-to-device transfer (CUDA)`
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Implement direct CUDA device→device copy (`cudaMemcpyPeer`/equivalent in the CUDA backend) when the destination device differs; avoid the host round-trip / clone shortcut.
  - **Risk:** Same as the Metal case — CUDA peer transfers silently alias instead of relocating data across GPUs.
  - **DONE (2026-07-06, CUDA leak/device-id session):** CUDA→CUDA transfer now checks `target_device == cuda_data.device_id()`; same-device stays a cheap refcounted clone (no copy needed), cross-device now genuinely bounces through the host (download then re-upload to the target device) instead of silently aliasing the source buffer.

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

- [x] **trustformers** `trustformers-models`: `src/gpt2/model/model_blocks.rs:953` — `TODO`: `// TODO: Fused matmul+bias+GELU kernel for Metal GPU`
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Integrate the existing Metal fused matmul+bias+GELU kernel (`gpu_ops/metal/metalbackend_matmul_gelu_f32_group.rs`) into the Linear layer with GPU-resident buffers, replacing the current MPS/Accelerate path.
  - **Risk:** Perf only (current MPS/Accelerate path is correct); integration requires GPU-resident buffer ops in the Linear layer.
  - **DONE (2026-06-29 session):** wired the existing `MetalBackend::matmul_bias_gelu_f32` kernel into `Gpt2MLP` (gated `metal,gpt2`), collapsing matmul→bias→GELU into one GPU dispatch. GPU parity test on Apple Silicon: bit-identical (max diff 0) vs the separate-ops path.

- [x] **trustformers** `trustformers-models`: `src/gpt_neox/model.rs:165` — `TODO`: `// Temporary fallback: Convert Metal/CUDA tensors to F32 // TODO: Implement full Tensor::Metal/CUDA support in Attention`
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Implement native Metal/CUDA tensor support in GPT-NeoX attention (QKV split + RoPE on-device) instead of downcasting GPU tensors to CPU F32.
  - **Risk:** Perf/correctness — GPU GPT-NeoX attention silently round-trips to CPU F32, losing GPU residency and precision flexibility.
  - **PARTIAL (2026-06-29, Campaign E CUDA-7):** the CPU-download fallback is now **honestly documented** (oxicuda host-in/host-out; attention downloads to CPU; parity-correct). On-device CUDA attention residency (QKV split + RoPE resident on the oxicuda device, no host round-trip) remains the open future-work item — keep this unchecked until residency lands.
  - **DONE (2026-07-06, CUDA-7 resident-attention session):** added `cuda_resident_forward` — a genuinely GPU-resident prefill path (NeoX-packing QKV gather into RoPE layout, device RoPE on Q/K, per-head causal attention, head merge, resident dense projection) that `forward` now tries first, falling back to the existing CPU-download path only when the resident fast path declines (non-2D/non-F32 input, weights not device-cached, etc.). Prefill-only by construction — the NeoX `Layer` trait carries no KV cache. Metal still uses the CPU-download fallback (out of this session's CUDA-only scope).

- [x] **trustformers** `trustformers-serve`: `src/resource_management/gpu_manager/manager.rs:259` — `TODO`: `// TODO: In production, add real GPU discovery:` — **DONE (2026-06-29, Campaign E Tier-2):** real `nvidia-smi`-based GPU discovery with an honest empty fallback when no GPUs are present (Pure-Rust probe, no FFI).
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Implement real GPU discovery (NVIDIA via NVML/Pure-Rust probe, AMD via ROCm, cross-vendor via the project's compute backends) and driver-compatibility checks instead of the placeholder enumeration.
  - **Risk:** Serve cannot see actual GPUs in production; scheduling/placement runs on stubbed device info. (Real bindings should follow the Pure-Rust/noffi policy.)

- [~] Implement Tensor::clamp FFI export (planned 2026-07-05)
  - Goal: trustformers_tensor_clamp FFI function works (TODO was stale — core Tensor::clamp(min, max) already exists).
  - Design: uncomment the stub, implement it by mirroring trustformers_tensor_leaky_relu's body exactly (null-check output, look up handle in TENSOR_REGISTRY, call .clamp(min_val, max_val), register result, write through output). Delete the now-inaccurate "not yet implemented" doc lines for clamp specifically (leave the note for concat/stack, which genuinely still need it).
  - Files: trustformers-c/src/tensor.rs only.
  - Tests: value-level test (clamp [-5.0, 0.5, 10.0] to [0.0,1.0], assert [0.0,0.5,1.0]) plus a null/invalid-handle negative test.
  - Risk: verify via cargo build -p trustformers-c that the new symbol appears in the generated trustformers.h.

### Known external-blocked placeholders (not actionable)

- ~~`trustformers-serve` `build.rs:2` — Proto compilation disabled: `tonic-build`/`tonic-prost-build` 0.14 API changed~~ — **RESOLVED (2026-06-29 session):** migrated to the tonic 0.14 split `tonic-prost-build` API; proto compilation and the `src/lib.rs:81,335` re-enables are done (see the 2026-06-12 section).
- `trustformers-core` `src/gpu_ops/rocm.rs:131` — `// TODO: Implement actual HIP kernel execution when HIP bindings are available`; falls back to CPU. Blocked on Pure-Rust/HIP bindings.
- ~~`trustformers-core` `src/gpu_ops/cuda/cuda_split/cuda_backend_ext.rs:457` — fused transformer kernel~~ — **PATH DELETED (2026-06-29, Campaign E):** this cudarc file was removed with the cudarc backend. Fused megakernel is now CUDA-6 (oxicuda-backend) future work — see the 2026-06-12 section.
- ~~`trustformers-wasm` `src/compute/gpu_tensor.rs:52,85` — WebGPU device creation + `Rc<RefCell<>>` interior-mutability wiring~~ — **RESOLVED (2026-06-29 session):** WebGPU device/queue setup + `Rc<RefCell<>>` wrapping implemented (see the 2026-06-12 section).

## Deferred stubs surfaced by /stub-check (2026-06-24)

Real but non-actionable-now stubs found in workspace member crates during nagare Phase 2; each needs cross-cutting wiring, external resources, GPU hardware, or upstream fixes. Documented for future passes.

- [ ] `trustformers-serve`: `src/openai_compat/mod.rs:742,769` — route_chat/route_completion return hardcoded stub responses; need real model inference wired through serve (reason: crosscut, needs inference path + weights)
- [~] Wire real model_service into GraphQL models() resolver (planned 2026-07-05)
  - Goal: QueryRoot::models() lists real registered/loaded models instead of one hardcoded row.
  - Design: add model_manager: Arc<ModelManager> to TrustformerServer (constructed like the other services) with a pub fn model_manager() accessor. Add the field to GraphQLContext, set it in create_context(). Rewrite models() to enumerate ModelRegistry::list_models() + ModelManager::get_loaded_model(), mapping ModelMetadata's fields onto ModelInfo (use ModelMetadata.created_at for loaded_at, not LoadedModel.loaded_at which is a monotonic Instant).
  - Files: trustformers-serve/src/graphql.rs, src/server/types.rs.
  - Tests: unit test the resolver logic directly against a ModelManager built on an in-memory/tempdir registry — no HTTP needed.
  - Documented caveat, not a blocker: the live /graphql HTTP route uses a separate, still-mock handler in server/functions.rs — this schema is currently unreachable dead code (pub use commented out, "disabled due to axum compatibility"). This fix corrects the schema layer only; re-enabling the live route is a separate follow-up.
  - Risk: ModelManager::new needs an async ModelRegistry::initialize() while TrustformerServer::new() is currently sync — thread this through by making construction async or deferring registry init to start().
- [ ] `trustformers-serve`: `src/model_management/manager.rs:47,56` — ModelInstance::infer() returns a placeholder string (reason: crosscut, needs inference)
- [x] `trustformers-serve`: `src/performance_optimizer/real_time_metrics/optimization/advanced_algorithms.rs:150,285,406,523,651,707,777` — update_with_feedback is a no-op across 7 algorithms; AlgorithmStatistics lost feedback_count/positive_feedback/negative_feedback fields (reason: crosscut, type API drift) — **DONE (2026-06-29, Campaign E Tier-2):** restored the three `AlgorithmStatistics` feedback fields and implemented real `update_with_feedback` across all 7 algorithms.
- [ ] `trustformers-serve`: `src/performance_optimizer/real_time_metrics/mod.rs:167` — threshold module disabled (only stub impls); comment cites 1,700+ compile errors to restore from .bak2 (reason: oversized)
- [~] Implement real ReportStorage store_report/get_report (planned 2026-07-05)
  - Goal: reports can actually be persisted and retrieved instead of both being fabricated stubs.
  - Design: add ReportStorage::store_report(&self, report: &Report) -> Result<()> (serialize to JSON, write to {storage_path}/{report_id}.json); rewrite get_report to read that file back, returning a real Err on a missing id instead of a fabricated stub. Resolve the Report vs GeneratedReport type mismatch by converting GeneratedReport -> Report at generation time inside ReportingSystem::generate_report(), which must be updated to call store_report after generating. Make storage_path configurable rather than hardcoded /tmp/reports.
  - Files: trustformers-serve/src/test_performance_monitoring/types/storage.rs, reporting.rs.
  - Tests: round-trip tests using std::env::temp_dir() for store/get.
  - Risk: concurrent access to a shared fixed path — use a per-instance/per-test unique subdirectory.
- [~] Implement real ExportManager export_report (planned 2026-07-05, depends on the store_report/get_report item above landing first)
  - Goal: exported reports write real content to a real file instead of being fabricated.
  - Design: implement export_report for real, scoped to text-representable formats only (Json via serde_json::to_string_pretty, Csv flattening Report's fields+metadata, Html/Xml minimal string templates). Fix the existing hardcoded ReportFormat::Json default at the ExportFormat -> ReportFormat call site to a real match; return an explicit "unsupported format" error for Excel/PowerPoint (no ReportFormat variant exists for either) rather than silently mapping them to JSON. Explicitly exclude true binary PDF generation — out of scope for this batch.
  - Files: trustformers-serve/src/test_performance_monitoring/types/reporting.rs, reporting.rs.
  - Tests: round-trip tests using std::env::temp_dir() for each export format; a dedicated test for the ExportFormat->ReportFormat mapping covering all 6 source variants.
  - Risk: depends on the store_report/get_report fix landing first (export_report calls get_report).
- [~] Fix TestPerformanceMonitoringConfig field drift (planned 2026-07-05)
  - Goal: add the 6 missing config struct fields (AnalyticsConfig, EventConfig, HistoricalDataConfig, AlertConfig, DashboardConfig, SubscriptionConfig) to TestPerformanceMonitoringConfig, wire real configs into service.rs instead of Default::default() everywhere, uncomment the specialized-service field assignments in mod.rs.
  - Design: add the 6 fields + update Default impls in types/config.rs; add the 2 missing leaf fields (audit_trail_enabled, compliance_logging, rate_limiting_enabled) referenced in the commented-out lines; update TestPerformanceMonitoringService::new() in service.rs to pass the real configs.
  - Files: trustformers-serve/src/test_performance_monitoring/types/config.rs, service.rs, mod.rs.
  - Tests: extend existing config-default tests; extend test_specialized_service_creation to actually assert the effect (not just .is_ok()).
  - Risk: double check there isn't a second, shadowing definition of any of these config types elsewhere (this module was split via SplitRS from a monolithic types.rs, and a stale types.rs.bak_refactored sits alongside).
- [ ] `trustformers`: `src/auto/feature_extractors/vision.rs:251,291` — preprocess_image and extract_visual_features return zero vectors (reason: crosscut)
- [x] Implement real HF Hub HTTP call for get_model_info (planned 2026-07-05) — **DONE (2026-07-05):** split into `#[cfg(feature = "hub")]` (real `reqwest` GET to `/api/models/{model_id}`, mapped via new pure `model_info_from_hub_json` helper) / `#[cfg(not(feature = "hub"))]` (unchanged deterministic mock) bodies in `hub_offline_packs.rs`; added field-mapping unit tests (no network) plus a mock-mode regression test.
  - Goal: get_model_info queries the real HF Hub API instead of returning fabricated metadata.
  - Design: split into #[cfg(feature = "hub")]/#[cfg(not(feature = "hub"))] bodies, mirroring hub.rs::download_file's existing split. Real body: reqwest GET to huggingface.co/api/models/{model_id}, parse as serde_json::Value, map fields, mirroring hub.rs::get_download_stats's existing pattern for this exact endpoint.
  - Files: trustformers/src/hub_offline_packs.rs only.
  - Tests: mock-mode (no hub feature) regression test; real path needs an HTTP-mocking boundary, no live network in unit tests.
  - Risk: must be feature-gated correctly — reqwest is optional; an unconditional `use reqwest` breaks the default (non-hub) build (hard compile error).
- [x] `trustformers`: `src/pipeline/conversational/config/presets.rs:409,462,488` — references AnalysisConfigBuilder/ReasoningConfigBuilder that may not exist; verify whether live or dead before implementing (reason: needs-clarification) — **DONE (2026-06-29, Campaign E Tier-2):** verified dead; removed the references to the non-existent `AnalysisConfigBuilder`/`ReasoningConfigBuilder`.
- [ ] `trustformers-mobile`: `src/react_native_fabric.rs:410` — execute_standard_inference returns a placeholder vec (reason: crosscut)
- [ ] `trustformers-optim`: `src/genie_stub.rs`, `src/sofo_stub.rs`, `src/lora_rite_stub.rs` — simplified GENIE/SOFO/LoRA-RITE optimizer steps; full research algorithms pending API-compat resolution (reason: research/needs-clarification)
- [x] `trustformers-debug`: `src/data_export.rs:608` — export_sqlite falls back to JSON instead of a real SQLite file; should use oxisql-sqlite-compat per COOLJAPAN policy (reason: small-medium, deferred) — **DONE (2026-06-29, Campaign E Tier-2):** real SQLite export via `oxisql-sqlite-compat` 0.3 (added to the workspace), replacing the JSON fallback.
- [x] `trustformers-debug`: `src/kernel_optimizer.rs:987,1035,1101,1180` — GPU kernel analyzers (LaunchConfig/MemoryAccess/ComputeUtilization/KernelFusion) return empty results (reason: gpu) — **DONE (2026-06-29, Campaign E Tier-2):** real occupancy / memory-coalescing / roofline / fusion analyzers implemented in new `src/kernel_optimizer/analysis.rs`; analyzers now return computed recommendations.

> **Skipped as already-tracked:** `trustformers-core` `src/gpu_ops/rocm.rs:131` (HIP kernel execution pending Pure-Rust HIP bindings, reason: external/gpu) — already documented under "Known external-blocked placeholders" in the 2026-06-22 section above.

## Planned campaigns — COOLJAPAN policy debt (scheduled 2026-06-24)

Two dedicated campaigns for pre-existing debt (not introduced by 0.1.3). Each is large and must run as a focused, per-crate, verify-as-you-go pass — NOT a quick fix. Sized from the nagare 0.1.3 policy-check + purity audit.

### Campaign A — No-unwrap + dead-code elimination ✅ DONE (2026-06-29 session)

- [x] **Campaign A — No-unwrap + dead-code elimination** — ✅ DONE (2026-06-29 session); results below
  - **Goal:** eliminate 1,844 `#[allow(...)]` (1,337 `dead_code`, 104 `unused_variables`, 61 `unused_imports`, 41 `unreachable_patterns`, 32 `deprecated`, ~160 clippy) and drive 2,421 production `unwrap`/`expect` (73 unwrap + 2,348 expect) toward zero, per the No-warnings + No-unwrap policies. Build/tests currently pass only because the allows silence the warnings.
  - **Scope:** workspace member `src/` only (exclude `#[cfg(test)]`/`tests/`). Per-crate `#[allow]` counts: debug 484, core 467, models 329, training 127, optim 110, serve 102, wasm 81, trustformers 62, mobile 42, tokenizers 30. Worst `unwrap`/`expect` files: mobile `profiler_impl.rs` 48, mobile `profiler_split/types.rs` 48, core `memory_leak_detector.rs` 41, mobile `adaptive_cache_manager.rs` 36, core `checkpoint/mapping.rs` 36, wasm `webgpu/types.rs` 35, core `hardware/registry.rs` 34, mobile `optimization/memory_pool.rs` 33, debug `realtime_dashboard.rs` 32, trustformers `memory_pool.rs` 31.
  - **Approach (phased, per-crate, leaf-first):** A1 — remove `#[allow(dead_code)]`; for each surfaced item: delete if truly dead, wire it up if it should be used, or make it `pub` if it's intended API. A2 — remove remaining `#[allow(unused_*/unreachable/deprecated/clippy)]`, fix root causes (unused imports/vars, update deprecated APIs). A3 — replace `unwrap()`/`expect()` in src with `Result`/`?`/`ok_or_else`/`unwrap_or_else`; keep a documented `expect` ONLY where the invariant is provably infallible.
  - **Risk:** removing `dead_code` allows may unmask code used only via macros/reflection — verify before deleting; deleting public-looking items can break downstream. Large effort; slice per-crate, PR-sized.
  - **Verify:** per crate `cargo clippy -p <crate> --all-features --all-targets -- -D warnings` clean WITHOUT the removed allows; `cargo nextest run -p <crate> --all-features` green.
  - **Suggested execution:** a dedicated `/recursive` or `/loop` campaign, one crate per slice, leaf-of-dep-graph first, capped iterations, never re-adding an `#[allow]`.

#### Results — Campaign A ✅ DONE (2026-06-29 session)

Across all 10 workspace crates, production-code `unwrap()`/`expect()` was driven to ~0 — only documented genuinely-infallible invariants remain, each carrying a `// reason:` comment. `#[allow]` was reduced wherever the underlying lint could be fixed; genuine load-bearing survivors were kept and documented. Tests and doc-examples keep their `unwrap`/`expect` (policy permits). No public API signatures changed.

| Crate | prod `unwrap`/`expect` before → after | `#[allow]` before → after |
|-------|----------------------------------------|---------------------------|
| tokenizers | 104 → 11 (documented) | 22 → 15 |
| optim | 165 → 0 (+4 documented `expect`) | 105 → 28 |
| mobile | 498 → 10 (documented) | 9 → 5 |
| debug | 80 → 0 | 460 → 35 |
| wasm | 213 → 0 | 81 → 65 |
| training | 123 → 6 (documented) | 127 → 10 |
| trustformers (umbrella) | 208 → 0 | 35 → 30 |
| serve | 301 → 0 | 26 → 17 |
| core | 459 → 0 (+4 documented; default + metal + cuda-oxicuda) | 372 → 367 (survivors verified load-bearing) |
| models | edited; clippy-clean | clippy-clean |

**Replacement patterns used:** lock-poisoning `.lock()/.read()/.write().unwrap()/.expect()` → `.unwrap_or_else(|p| p.into_inner())` (poison recovery); `SystemTime`/`Instant` → `.unwrap_or_default()`; `partial_cmp` → `.unwrap_or(Ordering::Equal)`; `?` / `.map_err(...)` / `.ok_or_else(...)?` propagation to `TrustformersError` in `Result` fns; `unwrap_or` / `match` / `let-else` for `Option` invariants; fire-and-forget sends → `let _ = ...`.

**Survivors are documented, not hidden:** the genuine `#[allow]` that remain (notably ~367 in `core`) are documented load-bearing scaffolding (the lint cannot be fixed without changing real behavior/API); the remaining `expect` are documented infallible invariants, each with a `// reason:` comment.

**Verification (workspace-wide, this session):** `cargo clippy --workspace --all-targets -- -D warnings` clean; `cargo nextest run --workspace` = **14,561 passed, 58 skipped, 0 failed** — identical to the pre-campaign baseline, i.e. zero behavior regression.

### Campaign B — Pure-Rust default-features dependency hygiene ✅ DONE (2026-06-29 session)

- [x] **Campaign B — Pure-Rust default-features dependency hygiene** — ✅ DONE (2026-06-29 session); results below
  - **Goal:** make DEFAULT features 100% Pure Rust (COOLJAPAN policy). Today every non-wasm member transitively compiles C libs in its default tree; only `trustformers-wasm` is clean. Our own crates are already compliant (use `oxiarc-*`, gate all heavy C backends) — this is third-party transitive leakage.
  - **Scope (default-tree offenders + who pulls them):** `aws-lc-sys` via `reqwest`→`rustls` (reqwest is a non-optional dep of `trustformers-core`); `onig_sys` via `tokenizers` (non-optional dep of core); `zstd-sys`/`zstd` via `jieba-rs`→`include-flate` (tokenizers); banned compression `flate2`+`miniz_oxide` via `plotters`/`image`/`png` (debug `visual`) and cloud-SDK HTTP stacks (serve); `brotli` via `lambda-web` (serve); `zip` (build-dep) via `utoipa-swagger-ui` (serve). NOTE: torch/cuda/opencl/vulkan/mpi/ffmpeg/kafka C deps are ALREADY correctly feature-gated (non-default) — leave them.
  - **Approach:** (1) `reqwest`/`rustls`: switch the default crypto provider off `aws-lc-rs` to a pure-Rust path (e.g. `reqwest` default-features=false + rustls with a RustCrypto provider), dropping `aws-lc-sys`. (2) `tokenizers`: disable its default `onig` feature, use the pure-Rust `fancy-regex` backend, dropping `onig_sys`. (3) `jieba-rs`/`include-flate`: replace or feature-gate so `zstd-sys` leaves default (or store data uncompressed / via `oxiarc`). (4) debug `visual` (plotters/image): move behind a non-default feature, or use a pure-Rust raster backend, dropping `flate2`/`miniz_oxide`. (5) serve: gate `lambda-web` behind a non-default `lambda` feature (drops `brotli`) and `utoipa-swagger-ui` behind a non-default `swagger-ui` feature (drops the `zip` build-dep).
  - **Risk:** changing the TLS crypto provider can affect HTTPS behavior; disabling `onig` can change tokenizer regex semantics for some models; gating debug-visual / serve lambda+swagger changes the default API/feature surface. Each change needs build+test verification and a default-vs-all-features tree diff.
  - **Verify:** `cargo tree -p <member> -e normal,build` shows no `-sys`/banned-compression in the DEFAULT tree for every member; `cargo build`/`cargo nextest run` green on default features; `--all-features` still green.
  - **Suggested execution:** a focused dependency-surgery pass, one offender at a time with a `cargo tree` diff + build/test gate after each; use `trustformers-wasm` (already pure) as the reference.

#### Results — Campaign B ✅ DONE (2026-06-29 session)

Every workspace crate now has a Pure-Rust (C/C++/Fortran-free) default dependency tree **except `trustformers-serve`** — an accepted exception: it is the HTTP server and keeps `aws-lc-sys` for rustls TLS, plus pure-Rust `flate2`/`miniz_oxide` pulled transitively by the AWS/Azure cloud SDKs. No default public API broke; the dropped functionality moved behind opt-in features.

| Offender (lang) | Pulled via | Resolution |
|-----------------|------------|------------|
| `onig`/`onig_sys` (C) + `esaxx_fast`/`esaxx-rs` (C++) | `tokenizers` dep of `trustformers-tokenizers` | `tokenizers` set to `default-features=false, features=["fancy-regex","progressbar"]` — pure-Rust regex backend, drop-in, zero code changes |
| `zstd-sys` (C) | `jieba-rs` (`trustformers-tokenizers`) | `jieba-rs` removed — it was entirely unused (`chinese.rs` has its own pure-Rust segmenter) |
| `flate2`/`miniz_oxide` (banned compression) | `trustformers-debug` `visual` (plotters/ratatui/crossterm) | `default = ["visual"]` → `default = []`; those deps were already optional + unreferenced, so `visual` is now opt-in |
| `brotli` + `zip` (banned) | `trustformers-serve` (`lambda-web`, `utoipa-swagger-ui`) | `lambda-web` → `lambda` feature; `utoipa-swagger-ui` → `swagger-ui` feature (both unreferenced in code, now opt-in). The CDN-based `/docs` Swagger page stays default |
| `aws-lc-sys` (C, via reqwest→rustls→aws-lc-rs) | all crates | `reqwest` made optional; core's remote-leaderboard gated behind a `remote-leaderboard` feature, the umbrella's HuggingFace hub downloads behind a `hub` feature. Local/cached model loading still works without `hub`. **`trustformers-serve` keeps reqwest/aws-lc-sys** (accepted) |
| `cc`/`alloca` (C) | `criterion` (`trustformers-models`) | `criterion` was a dead dependency (its "uses" were a local variable + benchmark string-templates) — removed |

Also fixed a pre-existing `--all-features`-only unused-import warning (`trustformers-core/src/parallel/model_parallel.rs:10`).

**New non-default (opt-in) features:** `hub` + `remote-leaderboard` (core/umbrella networking), `visual` (debug visualization — now off by default), `lambda` + `swagger-ui` (serve adapters).

**Accepted serve exception:** `trustformers-serve` (the HTTP server) keeps `aws-lc-sys` (rustls TLS) plus pure-Rust `flate2`/`miniz_oxide` from the AWS/Azure cloud SDKs — accepted, not a regression.

**Verification (this session):** C-free audit — 0 C/banned deps in every default tree except serve. `cargo clippy --workspace --all-targets -- -D warnings` (default) clean. `cargo nextest run --workspace` (default) = **14,540 passed, 58 skipped, 0 failed** (21 fewer than the 14,561 baseline = the now-opt-in feature tests, which run under their own features — not lost coverage). `--all-features` compiles.

## Planned campaign — oxicuda GPU migration (scheduled 2026-06-25)

Migrate the CUDA/GPU compute backend off `cudarc` (plus the lone scirs2 GPU touchpoint) onto **oxicuda** — the COOLJAPAN Pure-Rust CUDA replacement at `~/work/oxicuda`. **Version target (2026-06-25): oxicuda 0.4.0** (path dep `path = "../oxicuda"` during dev; oxicuda was advanced to 0.4.0 this session with the enhancements listed under Progress below; those enhancements are now **committed** in `~/work/oxicuda` (branch `0.4.0`: `1c1d23a bump-040`, `c01abfa "Add PTX kernel templates for bias-add and causal softmax operations"`, `abd66b9 "fmt"` — working tree clean), committed outside the delegated subagent flow). oxicuda needs no CUDA SDK / `nvcc` / `cudarc`; it loads `libcuda.so` at runtime via `libloading`, so this migration is a net **Pure-Rust policy win**.

**Scope decisions:**
- **GPU/CUDA + Metal.** Today's CUDA backend uses `cudarc` (NOT scirs2); scirs2's only *GPU* use is the macOS Metal `MPSOperations` call. So "use oxicuda instead of scirs2-* for GPU" = replace `cudarc` with oxicuda for CUDA AND replace the scirs2 Metal touchpoint with `oxicuda-metal`. [Q1 resolved]
- **Out of scope (do NOT touch):** `scirs2_core::ndarray` (~102 sites), `scirs2_core::random` (~50 sites), `scirs2-linalg`, `oxiblas` — mandatory SciRS2-Integration-Policy substrate, unrelated to GPU. [Q2 resolved]
- **Resolved (owner, 2026-06-25):** Metal IS in scope — `scirs2 MPSOperations` -> `oxicuda-metal` [Q3]. Integration tier = **phased**: low-level `DeviceBuffer`+`oxicuda-blas` FIRST for 1:1 parity with today's cudarc primitives, THEN adopt the purpose-built `transformer_backend` (PagedKvCache/FlashAttention) on the serving path as a net-new capability uplift [Q4 — owner deferred to recommendation]. Toolchain fits — trustformers rust 1.89 >= oxicuda 1.85; edition-2021 may depend on an edition-2024 crate [Q5].
- **Resolved by investigation (2026-06-25) [Q7]:** delete the hand-written PTX — every kernel has a direct oxicuda library equivalent. GEMM -> `oxicuda-blas` `BlasHandle::gemm`; activations (gelu/silu/relu) + softmax -> `oxicuda-blas` (`elementwise`/`reduction`); layernorm/rmsnorm + flash/paged attention + RoPE -> `oxicuda-dnn` (`DnnHandle`, which wraps a `BlasHandle`). The four trustformers PTX kernels (gelu/layernorm/rope/causal-softmax) all map 1:1. Only numeric parity remains, verified by C1 golden tests — not an open design question. **All of Q1-Q7 are now resolved.**
- **Crate map (oxicuda 0.4.0, verified 2026-06-25):** GPU GEMM = `oxicuda-blas`; GPU activations/softmax = `oxicuda-blas` (`elementwise`/`reduction`); GPU norm/attention/RoPE/conv/MoE = `oxicuda-dnn`; Metal = `oxicuda-metal` (`MetalBackend`: MPS `MpsMatrixMultiply` or MSL `simdgroup_gemm_msl`); KV-cache/scheduler/speculative/sampling = `oxicuda` root crate + `transformer-backend` feature (a module, NOT a standalone crate; compute delegates to `oxicuda-dnn::attn`). **OxiBLAS is CPU-only (v0.2.1, no GPU backend) — it stays the CPU substrate; the GPU GEMM path is `oxicuda-blas`, not OxiBLAS.**

### Campaign C — oxicuda CUDA + Metal backend migration

- [x] **Campaign C — oxicuda CUDA + Metal backend migration** — **DONE (2026-06-29, Campaign D + E; see those sections below for the full runtime-verified account).** Metal (C4) GPU-verified on Apple Silicon; CUDA runtime-verified 12/12 on a real NVIDIA RTX A4000 (Campaign D+E); `cudarc` fully removed and `cuda-oxicuda` promoted to be the `cuda` feature itself (C3, done); feature propagated to `trustformers-models` + the `trustformers` umbrella crate (C5, done). Only **C6** (the net-new `transformer_backend` uplift — `PagedKvCache`/`AttentionDispatch`/`ContinuousBatchScheduler`/`SpeculativeDecoder`/`TokenSampler`) remains not started — that is additional capability beyond parity, not a blocker for the migration itself.
  - **Goal:** replace the `cudarc::{driver,nvrtc}` CUDA backend with oxicuda 0.4.0 (`oxicuda-driver`/`-memory`/`-launch` + `oxicuda-blas`/`-dnn`), replace the scirs2 Metal `MPSOperations` touchpoint with `oxicuda-metal`, re-enable the kernels currently disabled "pending cudarc API migration", and drop the `cudarc` dependency and the dormant `scirs2-core/"gpu"` feature.
  - **Scope (cudarc surface, all in `trustformers-core` unless noted):** real backend `src/gpu_ops/cuda/cuda_split/` (cuda_backend.rs 1386, cuda_backend_ext.rs 527, cuda_dispatch.rs 318, cuda_types.rs 73); legacy dupes `gpu_ops/cuda/{backend,types,buffer_ops}.rs`; disabled `gpu_ops/advanced_kernels.rs` + `kernels/{mod,cuda_impl}.rs`; dispatch in `tensor/math_ops/linear_algebra.rs:186-205` + `layers/linear.rs`; Cargo `cuda=["dep:cudarc"]` + `cudarc 0.19` target dep (`Cargo.toml:101,134`). Metal touchpoint `gpu_ops/metal/common.rs:27` (`scirs2_core::gpu::backends::MPSOperations`). Pass-through only: `trustformers-models`/`trustformers` `cuda`/`metal` features. Out-of-workspace legacy `trustformers-c` (cudarc 0.17) handled last. `gpu.rs` is a CPU sim — leave it.
  - **Approach (phased, verify-as-you-go) — tier = low-level parity FIRST, transformer_backend uplift SECOND [Q4]:**
    - **C0 [DONE]** API-fit spike against `../oxicuda` 0.4.0 (the `oxicuda-blas`/`-dnn`/`-metal` APIs compile + link; Q1-Q7 resolved).
    - **C1 [DONE — delivered as an additive `cuda-oxicuda` feature, NOT in-place behind `cuda`]** low-level CUDA backend: built a side-by-side `OxicudaCudaBackend` (`trustformers-core/src/gpu_ops/cuda/oxicuda/`) — GEMM via `oxicuda-blas`, activations/softmax via `oxicuda-blas`, layernorm/rmsnorm + attention + RoPE via `oxicuda-dnn`, plus three new PTX kernels added to oxicuda this session (`causal_softmax`, `rope_neox_half_split`, `bias_add`). Full op-surface parity to the cudarc backend's 18 public ops. GPU-gated golden-parity tests written. **Compile + clippy validated on macOS via oxicuda runtime-loading `libcuda`; NOT runtime-verified (no NVIDIA GPU here).**
    - **C2 [DONE — additive routing]** per-device `OXICUDA_BACKENDS` singleton + dispatch routing wired under the `cuda-oxicuda` feature; the cudarc `cuda` arm (`dispatch_cuda_matmul`/`Tensor::matmul`) is untouched and remains the default CUDA path. (advanced_kernels / kernels/cuda_impl re-enable deferred to the runtime-verification pass.)
    - **C3 [DONE — Campaign E, 2026-06-29, runtime-verified on RTX A4000]** cudarc + legacy dupes deleted (`gpu_ops/cuda/cuda_split/`, `gpu_ops/advanced_kernels.rs`, `kernels/cuda_impl.rs`, fake `kernels/cuda_kernels.rs`, legacy `gpu_ops/cuda/{backend,types,buffer_ops}.rs`); `cuda = ["dep:oxicuda-blas", "dep:oxicuda-dnn", "dep:oxicuda-memory", "dep:oxicuda-driver"]`, `cuda-oxicuda = ["cuda"]` kept only as a deprecated alias. `rg cudarc trustformers-core/src` = comments only.
    - **C4 [DONE — GPU-verified on Apple Silicon]** Metal: `scirs2_core::gpu::backends::MPSOperations` (`gpu_ops/metal/common.rs`) -> `oxicuda-metal`. Stateless `matmul_f32` rerouted from the CPU-OxiBLAS path to oxicuda-metal `ComputeBackend::gemm` (genuine GPU compute); the GPU-resident MPS attention path (`matmul_gpu_to_gpu_mps[_scaled]`) migrated to oxicuda-metal and made zero-copy via oxicuda-metal's new `register_external` (no host round-trip). `scirs2-core/"gpu"` (and `"metal"`/`"mpsgraph"`) DROPPED — the Metal GPU stack no longer uses scirs2. Parity tests pass on the actual GPU.
    - **C5 [DONE — Campaign E]** docs/CHANGELOG updated (CHANGELOG `[0.1.4]`); `cuda` feature propagated to `trustformers-models` and the `trustformers` umbrella crate (confirmed: both crates' `Cargo.toml` now expose `cuda = ["trustformers-core/cuda"]`-style passthrough).
    - **C6 [NOT STARTED]** transformer_backend uplift [Q4 phase 2]: once C1-C5 parity is green, adopt oxicuda's `transformer-backend` (a feature/module of the `oxicuda` root crate, NOT a standalone crate): `PagedKvCache`, `AttentionDispatch` (Flash/Paged/SlidingWindow), `ContinuousBatchScheduler`, `SpeculativeDecoder`, `TokenSampler` — compute delegates to `oxicuda-dnn::attn`; CUDA-only (no Metal path). Wire it on the serving/generation path; net-new capability beyond today's cudarc backend; done as its own phase so a capability regression can't hide behind the backend swap.
  - **Risk:** kernel numeric parity (gelu/layernorm-eps/rope-theta/causal-softmax) must match existing PTX — golden tests mandatory. GPU is Linux/Windows-only + off-by-default, so CI builds but cannot execute kernels (same as today); needs a real NVIDIA box for runtime verification. oxicuda macOS CUDA = UnsupportedPlatform at runtime (Metal path is the macOS story). Metal (C4) is the most-used GPU path — treat as its own benchmarked slice. transformer_backend (C6) changes the serving compute path — bench latency/throughput vs the C2 baseline.
  - **Verify:** per phase — `cargo build -p trustformers-core --features cuda` + `cargo clippy -p trustformers-core --features cuda --all-targets -- -D warnings` (Linux); `cargo build -p trustformers-core` (default) and `--features metal` (macOS) stay green; `grep -rn cudarc trustformers-core/src` empty after C3; `grep -rn 'scirs2.*gpu\|MPSOperations' trustformers-core/src` empty after C4; `cargo tree -p trustformers-core --features cuda -e normal` shows oxicuda and no cudarc; `cargo nextest run -p trustformers-core --features cuda` (on GPU host) with CPU-parity assertions; workspace `--all-features` green.
  - **Suggested execution:** focused per-file campaign on `trustformers-core`, C0->C6 in order, parity-test gated after each kernel; all design questions (Q1-Q7) resolved — C0 is now just an API-fit spike against `../oxicuda` 0.4.0; C4 (Metal) and C6 (transformer_backend) are each their own benchmarked slice.

### Progress (2026-06-25 session)

**Done**
- **Metal — fully migrated off scirs2, GPU-verified on this Apple Silicon Mac.** Stateless `matmul_f32` rerouted from a CPU-OxiBLAS path to oxicuda-metal `ComputeBackend::gemm` (genuine GPU compute). The GPU-resident MPS attention path (`matmul_gpu_to_gpu_mps[_scaled]`) migrated off `scirs2_core::gpu::backends::MPSOperations` to oxicuda-metal, then made zero-copy via oxicuda-metal's new `register_external` (no host round-trip). `scirs2-core/"gpu"` (plus `"metal"`/`"mpsgraph"` from the macOS override) DROPPED — trustformers' Metal GPU stack no longer depends on scirs2. Parity tests pass on the real GPU. (= phase C4 + the Metal half of the C3 `scirs2-core/"gpu"` removal.)
- **oxicuda enhanced to 0.4.0** (in `~/work/oxicuda`, **committed** on branch `0.4.0` — `1c1d23a bump-040` + `c01abfa "Add PTX kernel templates…"` + `abd66b9 "fmt"`, working tree clean): `DeviceBuffer::from_raw` (non-owning import); oxicuda-metal `register_external`/`import_buffer`/`copy_dtod` (GPU-verified); three new PTX kernels — `causal_softmax` (oxicuda-blas), `rope_neox_half_split` (oxicuda-dnn), `bias_add` (oxicuda-blas).
- **CUDA `cuda-oxicuda` backend at full op parity, compile-validated.** New cudarc-free, opt-in `cuda-oxicuda` feature + `OxicudaCudaBackend` (`trustformers-core/src/gpu_ops/cuda/oxicuda/`) with full op-surface parity to the cudarc backend's 18 public ops: host-facing matmul/gelu/layernorm/rope/softmax_causal; resident-buffer cache + matmul/gelu/layernorm/add_bias `_gpu_to_gpu`, `matmul_with_cached_weight`, `device_info`, persistent-buffer management. Validated by `cargo build` + `clippy` on macOS (oxicuda runtime-loads `libcuda`). (= phase C1, delivered additively rather than in-place.)
- **C2 dispatch wired (additive).** Per-device `OXICUDA_BACKENDS` singleton + dispatch routing under the `cuda-oxicuda` feature; the cudarc `cuda` path is fully intact, untouched, and side-by-side (additive). GPU-gated parity tests written.

**Remaining (needs an NVIDIA GPU box) — RESOLVED 2026-06-29, see Campaign D and Campaign E below**
- ~~Runtime parity verification of the whole `cuda-oxicuda` backend against the cudarc path / CPU reference~~ — **DONE**: an NVIDIA RTX A4000 became available (Campaign D), and Campaign E achieved 12/12 oxicuda parity tests passing on it.
- ~~C3 (cudarc removal): DEFERRED pending NVIDIA-GPU verification~~ — **DONE (Campaign E)**: cudarc fully removed; oxicuda is now the `cuda` backend itself, with feature propagation to models/umbrella also complete.

**Follow-ups (separate from Campaign C)**
- **Latent trustformers RoPE bug:** trustformers' RoPE is internally inconsistent — its CPU reference (`rope/mod.rs`) is INTERLEAVED while its CUDA kernel is GPT-NeoX HALF-SPLIT, so CPU vs CUDA rope produce different results. Worth a dedicated fix (the new oxicuda `rope_neox_half_split` kernel matches the CUDA half-split convention, not the CPU interleaved one).
- **Stale refactor clutter:** ✅ DONE (2026-06-29) — removed the leftover `*.backup`/`*.bak` files under `trustformers-core/src/` (`gpu_ops/metal/`, `layers/`, `kernels/simd/`, `tensor/math_ops/`).

## Hardware-gated remaining work — consolidated status (2026-06-29)

> **The CUDA / NVIDIA subset has its own actionable checklist: [`TODO-CUDA.md`](TODO-CUDA.md).**

As of the 2026-06-29 session, **all four planned 0.1.4 campaigns are complete and verified on this hardware**: Campaign C Metal slice (C4 — GPU-verified on Apple Silicon), Campaign A (no-unwrap + dead-code), Campaign B (Pure-Rust default features, `trustformers-serve` excepted), and the deferred-stubs campaign (4 of 6 implemented). Workspace default `cargo nextest run --workspace` = **14,560 passed, 58 skipped, 0 failed**; `cargo clippy --workspace --all-targets -- -D warnings` clean. trustformers' 0.1.4 working tree is intentionally uncommitted (the user's "keep uncommitted" choice). The oxicuda 0.4.0 enhancements are **committed** in `~/work/oxicuda` (branch `0.4.0`: `1c1d23a bump-040`, `c01abfa "Add PTX kernel templates for bias-add and causal softmax operations"`, `abd66b9 "fmt"`; working tree clean) — committed outside the delegated subagent flow (the subagents were instructed to run no git).

**UPDATE 2026-06-29 (Campaign D):** an **NVIDIA RTX A4000 + CUDA 12.0 box is now available**,
so the CUDA rows below are no longer blocked. Items **#1, #2, #3, #4 are RESOLVED** (Campaign D
established #1/#4; **Campaign E (2026-06-29) completed the full oxicuda migration: #1, #2, #3** —
cudarc removed, oxicuda is the `cuda` backend, 12/12 parity on the A4000; see "Campaign E" below).
The AMD/ROCm (#5) and cloud/ASIC (#6) rows remain genuinely hardware/SDK-gated.

| # | Remaining work | Blocked on | Status |
|---|----------------|-----------|----------------|
| 1 | ~~Runtime parity verification of the `cuda-oxicuda` backend~~ | ~~NVIDIA CUDA GPU~~ | ✅ **DONE (Campaign D + E)** — **12/12 oxicuda parity tests pass on A4000**. Both root-caused bugs FIXED in Campaign E: layernorm PTX (oxicuda-dnn `.maxntid` placement + f16/bf16 `.b16`); GEMM RowMajor (fixed by the oxicuda repo owner's in-progress `gemm/` rewrite) |
| 2 | ~~cudarc removal + `cuda-oxicuda` feature propagation~~ | ~~NVIDIA GPU~~ | ✅ **DONE (Campaign E)** — cudarc fully removed (`cuda`→oxicuda; `cuda-oxicuda`=deprecated alias; cudarc dep + `cuda_split/`+`advanced_kernels`+`cuda_impl`+fake `cuda_kernels`+legacy dupes all deleted); feature propagated to models+umbrella. (`gpu_accelerated`/`hardware_acceleration` re-enable on oxicuda = noted follow-up — TODO-CUDA.md CUDA-3.) |
| 3 | ~~CUDA GPU-residency zero-copy via `DeviceBuffer::from_raw` in `matmul_gpu_to_gpu`~~ | ~~NVIDIA GPU~~ | ✅ **DONE (Campaign E)** — resident `matmul_gpu_to_gpu` confirmed on-device (`DeviceBuffer` cache, no host round-trip); resident parity tests green |
| 4 | ~~Fused CUDA transformer-layer kernel — `cuda_backend_ext.rs`~~ | ~~NVIDIA CUDA GPU~~ | ✅ **DONE (Campaign D)** — real GPU-resident layer + CPU-parity test (max diff 5.96e-8) |
| 5 | HIP/ROCm real kernel execution — `gpu_ops/rocm.rs:131` (currently falls back to CPU) | AMD ROCm GPU + Pure-Rust HIP bindings | 🔒 still gated (no AMD GPU here) |
| 6 | `trustformers-c` cloud-function handlers + ASIC/device bindings — return placeholder handles/JSON | Real cloud/ASIC hardware + Pure-Rust/noffi-compliant SDKs | 🔒 still gated |

**Non-hardware follow-up still open (doable here, not blocked):**
- ~~**RoPE CPU/CUDA convention mismatch**~~ — **RESOLVED / re-characterized 2026-06-29 (Campaign D).** Not a live bug: `rope/mod.rs` is an **orphaned, uncompiled file** (no `mod rope;` mounts the `rope/` dir; nothing imports its `RopeFrequencies`/`apply_yarn_rope`/…). The **compiled** CPU RoPE is `kernels/rope.rs` (`VectorizedRoPE`), already GPT-NeoX HALF-SPLIT and matching the CUDA kernel. Aligned the orphaned file to half-split anyway (hygiene) + added a live CPU↔CUDA `rope_f32` parity test. **RESOLVED (Campaign E): the orphaned `rope/mod.rs` (~1693 lines dead) was DELETED** — confirmed no `mod rope;` mount + no `crate::rope::` importers; live `kernels/rope.rs` tests stay 34/34 green.

---

## Campaign D — NVIDIA CUDA runtime verification & enablement (2026-06-29, RTX A4000)

**Environment changed:** this session ran on **Linux x86_64 with a real NVIDIA RTX A4000
(16 GB, Ampere sm_86) + CUDA 12.0 (`nvcc`) + driver 550 + libcuda/cublas/cudnn8/nvrtc12**.
The "everything remaining is hardware-gated, no NVIDIA GPU here" premise above no longer
holds — the CUDA work is now actionable and was runtime-verified on hardware. Scope this run
(user-approved "Verify + fix core"): D1 + D2 + D3 + D6 + cleanup. cudarc stays the default
backend; CPU-default + macOS builds stay green; cudarc NOT removed.

- **Baseline (now verified on hardware):** `cargo build/test -p trustformers-core --features cuda`
  builds (cudarc 0.19 + `cuda-12000` ↔ system CUDA 12.0) and the cudarc backend the TODO
  called "compile-validated, NOT runtime-verified" **passes on the A4000**. The TODO's
  hardware-gated table items #1 (cudarc runtime) and #4 (fused kernel) are RESOLVED below.

- **D1 — CUDA runtime parity suite (DONE).** Added **8 CPU↔CUDA golden-parity tests** in
  `gpu_ops/cuda/cuda_split/cuda_dispatch.rs` (graceful-skip) for the previously-untested ops:
  `rope_f32` (+ partial-rotary), `softmax_causal_f32`, `add_bias_gpu_to_gpu`,
  `matmul_with_cached_weight`, `matmul_gpu_to_gpu`, `layernorm_gpu_to_gpu`, and a large
  non-square (64×96×48) matmul. Each uses an inline CPU reference mirroring the kernel's
  documented math. `cargo test -p trustformers-core --features cuda --lib gpu_ops::cuda` =
  **18 passed, 0 failed** on the A4000 (was 9; +8 new +1 from D3).

- **D2 — RoPE convention (DONE / re-characterized).** See the resolved follow-up above:
  the flagged mismatch was against an **orphaned** file; the live path is half-split and is
  now locked by the D1 `rope_f32` parity test. Orphaned `rope/mod.rs` aligned to half-split.

- **D3 — fused transformer-layer (DONE).** Resolves table item #4. Replaced the
  `cuda_backend_ext.rs` `transformer_layer_forward_optimized` placeholder (was `eprintln!` +
  returns input unchanged) with a **real GPU-resident pre-norm causal self-attention layer**:
  LayerNorm→QKV→bias chained via `BufferId` with no host round-trips, then RoPE + per-head
  causal-softmax attention + output proj + residual. CPU-parity test
  `test_transformer_layer_forward_optimized_matches_cpu` → **max abs diff 5.96e-8**. (A single
  fully-fused megakernel remains a future optimization — the disabled-kernel re-enable D5 was
  deferred this run.)

- **D6 — `cuda-oxicuda` runtime-verified on hardware (DONE, WITH FINDINGS).** Resolves table
  item #1. First-ever execution of the oxicuda backend on NVIDIA hardware:
  **7 / 12 parity tests PASS** (gelu, rope, softmax_causal, add_bias resident, resident gelu,
  device_info, singleton) — these oxicuda ops are now confirmed numerically correct on the
  A4000. **5 fail, two real kernel bugs found + root-caused (in committed oxicuda 0.4.0):**
  1. **oxicuda-blas GEMM layout bug** (3 tests): RowMajor C output partially unwritten
     (`C[1,0]` returns 0, expected 139). Root cause: `oxicuda-blas/src/level3/gemm_api.rs`
     `gemm()` builds `GemmProblem` and dispatches with only `a.ptr/b.ptr/c.ptr` — it **drops
     the `MatrixDesc` `Layout`/leading-dimension**, so RowMajor inputs are mishandled. The
     trustformers call is correct (verified: proper RowMajor `MatrixDesc`, NoTrans, α=1/β=0).
  2. **oxicuda-dnn layer_norm PTX bug** (2 tests): `generate_layer_norm_ptx`
     (`oxicuda-dnn/src/norm/layer_norm.rs:188`) emits PTX that fails to load on sm_86/CUDA12
     (`CUDA: invalid PTX`); the blas-sourced gelu/rope/softmax PTX all load fine.
  The 5 failing tests are now `#[ignore = "BLOCKED on oxicuda … — see Campaign D"]` so the
  opt-in suite is green (7 pass / 5 ignored / 0 failed) and the bugs are documented in-code.
  These are oxicuda-repo defects, NOT trustformers-side — documented, not silently edited.

- **Cleanup (DONE).** `println!` at `cuda_backend.rs:41` → `tracing::debug!`; the misleading
  `eprintln!` fallback notices in `cuda_backend_ext.rs` (which falsely claimed tensor-cores /
  cudarc-streams unavailable — this box is CC 8.6, cudarc 0.19 has streams) → truthful
  `tracing::debug!`; feature-conditional `unused_mut` in `gpu.rs:246` `#[allow]`-annotated.
  Clippy `--features cuda --all-targets -D warnings` = clean; default clippy = clean.

### ⚠️ Blocker flag for the user (NOT caused this session)
The **oxicuda repo at `oxicuda/` has substantial pre-existing UNCOMMITTED changes**
(1106 insertions across 6 files: `oxicuda-ptx/{ir/types.rs,templates/gemm.rs}`,
`oxicuda-dnn/conv/fprop/direct.rs`, `oxicuda-solver/.../cholesky.rs`, `oxicuda-sparse/…`) — a
**half-finished refactor that does NOT compile from scratch** (`templates/gemm.rs` references
undefined `cvt_to_acc`/`cvt_to_out`, E0425). A *clean* `cargo …--features cuda-oxicuda` build
is currently blocked by this; the D6 run above succeeded only by reusing a cached good
`oxicuda-ptx.rlib`. **Left untouched** (it is your in-progress work). Recommend finishing or
`git stash`-ing it; afterward the oxicuda suite will report 7 pass / 5 ignored.

### Campaign D — remaining / recommended follow-ups (next NVIDIA session)
- **Fix the 2 oxicuda kernel bugs** (GEMM layout propagation in `gemm_api.rs`; layer_norm PTX
  in `oxicuda-dnn`), then un-`#[ignore]` the 5 oxicuda parity tests → 12/12 green.
- **D5 (deferred):** re-enable `advanced_kernels.rs` (cudarc 0.17→0.19 migration: `load_ptx`→
  `compile_ptx`+`load_module`, `htod_copy`→`clone_htod`, `get_func`→`load_function`) and port
  Flash-Attention-v2 from disabled `kernels/cuda_impl.rs`; **delete** the fake stub
  `kernels/cuda_kernels.rs` (hardcoded "RTX 4090" device, todo!/zero returns).
- **D4 (deferred):** add `device_id` to `CudaTensorData` (currently only buffer_id/shape/dtype)
  + real CUDA→CUDA copy (`tensor/utils.rs:561` clones) + stop hardcoding device 0 — needs ≥2
  GPUs to fully verify. **DONE (2026-07-06, CUDA leak/device-id session):** `CudaTensorData`
  now carries its device ordinal via the new refcounted `OxiCudaBufferHandle`; `to_device_enum`
  CUDA→CUDA does a real device-id check (same-device clone, cross-device host bounce instead of
  aliasing); GELU/layernorm/linear dispatch use the tensor's device instead of hardcoded `0`.
  Multi-GPU hardware verification still needs ≥2 GPUs.
- **Mount-or-delete the orphaned `rope/mod.rs`** (~1670 lines, not in the module tree).

---

## Campaign E — full oxicuda CUDA migration: cudarc DROPPED (2026-06-29, RTX A4000)

**`/ucont` orchestration loop.** After re-reading `TODO-CUDA.md` the CUDA goal was re-centered from
"extend cudarc" to its real intent: **migrate the CUDA compute path OFF `cudarc` ONTO the Pure-Rust
`oxicuda` and delete cudarc** (COOLJAPAN policy). Run on Linux x86_64 + **RTX A4000 (sm_86) + CUDA
12.0**. User decisions: *fix the oxicuda kernels directly* + *drop cudarc this session*.

- **CUDA-1 — oxicuda parity GREEN (12/12 on A4000).** Ground-truth build+test (with the oxicuda repo
  owner's in-progress `gemm/` rewrite present) showed **10/12 pass, only the 2 layernorm tests fail**
  — the **GEMM RowMajor bug was already fixed by the owner's rewrite** (3/3 GEMM parity green). The
  remaining bug was layernorm "invalid PTX": real cause = **`.maxntid N,1,1;` emitted inside the
  kernel body with a trailing `;`** (must be a directive between `)` and `{`) — fixed in
  `oxicuda-dnn/src/norm/layer_norm.rs`; also fixed the f16/bf16 `ld/st.global.f16`→`.b16`+`cvt`
  defect. Un-`#[ignore]`'d all 5 tests → **12/12 pass by default**. (oxicuda edit confined to
  `layer_norm.rs`; no git; owner's WIP untouched.)
- **CUDA-2/3 — cudarc removed.** `cuda = [oxicuda-blas/dnn/memory/driver]`; `cuda-oxicuda = ["cuda"]`
  (deprecated alias); cudarc dep deleted; all `feature="cuda-oxicuda"` cfgs renamed to `cuda`; the
  oxicuda dispatch unified under `cuda`; `crate::gpu_ops::cuda::{BufferId, get_cuda_backend}`
  re-exported from oxicuda (resident layer paths in `layers/{linear,layernorm}.rs`,
  `tensor/utils.rs`, `ops/activations.rs` now oxicuda-backed, no call-site edits). **Deleted:**
  `gpu_ops/cuda/cuda_split/` (whole dir), `gpu_ops/advanced_kernels.rs`, `kernels/cuda_impl.rs`,
  fake `kernels/cuda_kernels.rs`, legacy dupes `gpu_ops/cuda/{backend,types,buffer_ops}.rs`.
  `rg cudarc trustformers-core/src` = comments only; `cargo tree --features cuda | grep cudarc` empty.
  `gpu_accelerated`/`hardware_acceleration` stay `#[cfg(not(feature="cuda"))]` (they call the removed
  cudarc kernel API; oxicuda port = follow-up; stale "cudarc 0.17.7" comments corrected).
- **CUDA-4** — resident `matmul_gpu_to_gpu` confirmed genuinely on-device (`DeviceBuffer` cache, no
  host round-trip). **CUDA-5** — `cuda` feature propagated to `trustformers-models` + umbrella
  `trustformers`. **CUDA-7** — gpt_neox: oxicuda is host-in/host-out, so GPU operands download to CPU
  F32 for attention (correct, not a stub); misleading TODO replaced with honest status; on-device
  residency = future work. **CUDA-8** — orphaned `rope/mod.rs` (~1693 L) DELETED; live
  `kernels/rope.rs` tests 34/34.
- **Verification:** `cuda` build + clippy `-D warnings` clean (cuda + default); **12/12 oxicuda
  parity on A4000**; trustformers workspace `cargo fmt --check` clean (the 42 fmt diffs are all in the
  oxicuda repo owner's uncommitted WIP — gemm/conv/solver/sparse — left untouched). **Workspace
  default regression GREEN: `cargo clippy --workspace --all-targets -- -D warnings` exit 0;
  `cargo nextest run --workspace` = 14,553 passed / 57 skipped / 0 failed** (within the
  14,540–14,561 baseline band — no regression). trustformers working tree left uncommitted.

### Campaign E — remaining CUDA follow-ups (not blockers)
- **CUDA-6** fused transformer-layer megakernel (perf optimization; the cudarc fused path from
  Campaign D was deleted with `cuda_split` — oxicuda runs the layer as individual ops, correct/unfused).
- **CUDA-7** real on-device gpt_neox attention residency (needs oxicuda-dnn resident QKV-split/RoPE/attn). **DONE (2026-07-06):** `cuda_resident_forward` added (prefill-only, since the NeoX `Layer` trait carries no KV cache); `forward` tries it first and falls back to the CPU-download path otherwise.
- **CUDA-3 follow-up** port `gpu_accelerated`/`hardware_acceleration` onto oxicuda (re-enable under `cuda`).
- **CUDA-9** legacy `trustformers-c` (excluded crate, cudarc 0.17). **SUPERSEDED (2026-07-06):** `trustformers-c` is now deprecated (its `TODO.md` rewritten as an English deprecation notice); this cudarc-retirement task is dropped, not carried forward.
- The oxicuda repo (`oxicuda/`) retains the owner's large uncommitted in-progress rewrite
  (gemm/conv/solver/sparse) with `cargo fmt` diffs — **left untouched; the owner finishes/commits it.**

---

## 0.2.0 release scope — workspace dependency hygiene: OxiCUDA GPU migration tail + PyTorch (tch) removal (planned 2026-07-06)

**Status (2026-07-06): all Track 1 and Track 2 items below landed this session** (workspace check +
clippy + full nextest run green: 12033/12033 passed, 113 skipped availability-gated GPU tests, zero
warnings). The 0.3.x P2 (`torsh-interop`) remains unstarted by design, out of 0.2.0 scope. Additional
hygiene fixes uncovered and landed alongside this work: `PlatformCapabilities`-based GPU detection in
`trustformers-core/src/device.rs` was actually broken (`cuda_available`/`metal_available` never truly
reflected hardware) — `cuda_if_available`/`best_available` now probe real backends
(`oxicuda_cuda_available()` / `metal::Device::system_default()`) instead; a dead
`scirs2_core::linalg`/`tensor` import pair was removed from `trustformers-mobile`'s
`advanced_neural_engine_v4.rs`; the dead `scirs2-core` dependency and `scirs2` feature were removed
from `trustformers-wasm`; `trustformers-c`'s `TODO.md` was rewritten as an English deprecation notice
(the crate's C FFI surface is superseded by the pure-Rust core + language-binding crates).

Two tracks for the 0.2.0 branch, both workspace-root Cargo.toml surgery plus doc alignment.
**Track 1 — OxiCUDA GPU migration (scirs2-core gpu → OxiCUDA):** the compute migration itself is
DONE (Campaign C/D/E above — `oxicuda` is the `cuda` backend, `oxicuda-metal` the Metal one, and
`trustformers-core` dropped scirs2 GPU features in 0.1.4), but the **workspace-level** scirs2-core
dependency still requests the `gpu` feature and two dead scirs2 deps/docs mandates linger — this
track finishes that tail. **Track 2 — PyTorch (tch) dependency removal, DECISION:** delete the
`tch` dependency and the `torch` feature entirely in 0.2.0 (workspace `Cargo.toml:82`,
trustformers-core `torch` feature + ~40 lines of cfg arms, and the forwarder features in
`trustformers`, `trustformers-training`, `trustformers-c`). Do NOT adopt ToRSh as a replacement
now; a P2 task below records evaluating an optional `torsh-interop` feature in 0.3.x once torsh
0.2.0 ships on crates.io. Sub-decision on candle: drop the unused `candle-nn` workspace dep now,
keep the `candle` feature/variant through 0.2.0 (it is in every `full` set), and decide
implement-vs-remove in 0.3.x.

**Rationale (tch):** `Tensor::Torch` (`trustformers-core/src/tensor/mod.rs:220-221`) is never
constructed anywhere in the workspace — every op hits wildcard error arms, and the `cfg(torch)`
"PyTorch validation" is simulated with hardcoded results
(`trustformers-core/src/testing/cross_framework.rs:309-325`) — so zero functionality is lost.
Keeping tch costs a multi-GB libtorch download plus policy violations: `torch-sys` pulls `cc`
(C++), the OxiARC-banned `zip 0.6.6`, `ureq`, and duplicate old ndarray/rand/safetensors pins
(`Cargo.lock:10598-10610`). All real PyTorch interop is pure Rust already (`safetensors`
non-optional at `trustformers-core/Cargo.toml:34`; `checkpoint/formats.rs`,
`utils/weight_loading.rs`, optim `pytorch_compat.rs` — all kept). ToRSh cannot replace it today:
crates.io torsh 0.1.3 pins scirs2 0.5.1 (type-incompatible duplicate of trustformers' scirs2 0.6.0
stack), torsh 0.2.0 is unpublished with a workspace-blocking compile error, and torsh adds no
missing capability (no real pickle `.pt` parser either). Both tch-focused investigation reports
independently reached the same verdict; the only inter-report disagreement (candle) is resolved by
deferring the variant decision to 0.3.x while removing the definitively-dead `candle-nn` dep now.

### Track 1 — OxiCUDA GPU migration tail (scirs2-core gpu → OxiCUDA)

- [x] **[P0] Remove `"gpu"` from the workspace scirs2-core feature list.** — **DONE (2026-07-06):**
  the `"gpu"` entry and its comment were deleted from the workspace-root `scirs2-core` dependency's
  feature list in `Cargo.toml`. Delete the `"gpu",`
  entry from the scirs2-core workspace dependency in the root `Cargo.toml` (line 285, comment
  "GPU acceleration abstractions"). Verified safe: zero scirs2-core GPU API usage exists anywhere
  in the repo (no `scirs2_core::gpu` / `gpu_registry` / `tensor_cores` references, no glob/prelude
  imports, checked against scirs2-core 0.6.0's crate-root `pub use crate::gpu::*` re-export), and
  the only flag-sensitive field trustformers could observe (`PlatformCapabilities.gpu_available`)
  is never read. In scirs2-core 0.6.0 `gpu = ["std"]` pulls zero extra deps, so this is purely a
  build-time win across all 10 members plus `trustformers-c`/`-py` via feature unification.
  Verify: `cargo check --workspace --all-features` green, plus a `cargo check` inside
  `trustformers-c` and `trustformers-py`.
  Evidence: `Cargo.toml:276-287` (feature list, `"gpu"` at :285);
  `trustformers-core/src/device.rs:80-113` (only `PlatformCapabilities` consumer).
- [x] **[P1] Remove the dead scirs2-linalg workspace dependency.** — **DONE (2026-07-06):** the
  `scirs2-linalg` workspace dependency and the `trustformers-mobile` consumer line were both
  deleted (zero usage confirmed via `rg scirs2_linalg`). Delete
  `scirs2-linalg = { version = "0.6.0" }` from the root `Cargo.toml` (line 288).
  `rg scirs2_linalg` over all .rs files returns zero matches — the crate is declared but never
  used. Land together with the trustformers-mobile task that removes its
  `scirs2-linalg.workspace = true` line (`trustformers-mobile/Cargo.toml:33`), the only member
  that declares it. (Note: the Campaign C scope decision above listed `scirs2-linalg` as
  "out of scope substrate" — superseded by this finding: it is declared but has zero usage.)
  Verify: rg confirms zero usage, `cargo check --workspace` green.
  Evidence: `Cargo.toml:288`; `trustformers-mobile/Cargo.toml:33`.
- [x] **[P1] Rewrite GPU sections of SCIRS2_INTEGRATION_POLICY.md and CONTRIBUTING.md for the
  oxicuda reality.** — **DONE (2026-07-06):** both documents rewritten to codify scirs2-core as
  CPU-only substrate and `oxicuda-*`/`oxicuda-metal`/`oxicuda-backend` as the GPU path; policy
  version bumped to 2.1.0. `SCIRS2_INTEGRATION_POLICY.md` (sections around lines 129-154, 234, 339-344,
  363-368, 386, 405-408, 617-628, 697, 798) still mandates scirs2-core GPU features and
  `scirs2_core::gpu_ops::MetalBackend/CudaBackend` — APIs that do not exist under those paths in
  scirs2-core 0.6.0 — and `CONTRIBUTING.md:330` says "Use SciRS2's GPU context management".
  Rewrite both to codify: scirs2-core is the CPU substrate only (ndarray/random/simd_ops/
  parallel_ops); GPU acceleration is `oxicuda-*` (CUDA), `oxicuda-metal`/`oxicuda-backend`
  (Metal), and trustformers' own wgpu backend.
  Verify: rg for `scirs2.*gpu|gpu_ops::.*Backend` in both docs returns no stale mandates.
  Evidence: `SCIRS2_INTEGRATION_POLICY.md:129-154,234,339-344,363-368,617-628`;
  `CONTRIBUTING.md:330`.

### Track 2 — PyTorch (tch) dependency removal

- [x] **[P0] Delete the tch workspace dependency and purge torch-sys from the lockfile.** —
  **DONE (2026-07-06):** `tch` removed from the root `Cargo.toml`, the `torch` feature and all
  `cfg`/match arms removed from `trustformers-core`, and the forwarder features removed from
  `trustformers`, `trustformers-training`, and `trustformers-c`; `.typos.toml`'s `"tch" = "tch"`
  entry removed. Confirmed via `rg '"tch"|torch-sys'` over `Cargo.lock`: no matches — tch/torch-sys
  and their `cc`/`zip 0.6.6`/`ureq`/duplicate-pin baggage are gone. Remove
  `tch = { version = "0.24", features = ["download-libtorch"] }` from the root `Cargo.toml`
  (line 82). Must land in the same change as the `torch`-feature removals in `trustformers-core`,
  `trustformers`, `trustformers-training`, and `trustformers-c` (forwarders referencing a deleted
  feature fail Cargo resolution). After removal, regenerate `Cargo.lock` and verify tch 0.24.0
  (`Cargo.lock:10001`) and torch-sys 0.24.0 (`Cargo.lock:10598`) drop out, taking `cc`, the
  policy-banned `zip 0.6.6`, `ureq`, and duplicate ndarray 0.16.1 / rand 0.8.6 /
  safetensors 0.3.3 pins with them. Optionally remove the `"tch" = "tch"` entry from
  `.typos.toml:163`.
  Evidence: `Cargo.toml:82`; `Cargo.lock:10001, 10598-10610`; `.typos.toml:163`.
- [x] **[P1] Remove the unused candle-nn workspace dependency.** — **DONE (2026-07-06):**
  `candle-nn = "0.11.0"` deleted from the root `Cargo.toml`; `candle-core` and the `candle`
  feature/variant were kept untouched, per the 0.3.x-deferred decision. Delete `candle-nn = "0.11.0"`
  from the root `Cargo.toml` (line 81) — no member crate references it at all. Keep `candle-core`
  (line 80) for 0.2.0 because the `candle` feature remains in the `full` feature sets of
  `trustformers-core` and `trustformers`; its implement-or-remove decision is a recorded P2 in
  `trustformers-core/TODO.md`.
  Verify: `cargo tree` shows no candle-nn; `cargo check --workspace --all-features` green.
  Evidence: `Cargo.toml:80-81`; `trustformers-core/Cargo.toml:133` (`full` includes `candle`);
  `trustformers/Cargo.toml:93`.

### Post-0.2.0 (0.3.x) — deferred follow-ups

- [ ] **[P2] Evaluate an optional `torsh-interop` feature (0.3.x, record-only).** — Not started
  this session (out of 0.2.0 scope by design; blocked on torsh 0.2.0 publishing to crates.io with a
  scirs2 0.6-aligned stack). Once torsh
  0.2.0 is published on crates.io with its scirs2 0.6-aligned stack, evaluate an optional
  `torsh-interop` feature for tensor conversion via the shared scirs2 0.6 ndarray types. Do NOT
  depend on torsh 0.1.3 (pins scirs2 0.5.1 — would duplicate the scirs2 stack at an incompatible
  version), and note torsh 0.2.0-dev currently has a workspace-blocking compile error (torsh-data
  `Tensor::cat` signature). Real pickle `.pt` loading would still need a pure-Rust pickle
  deserializer that neither project has; `safetensors` remains the checkpoint bridge.
  Evidence: `Cargo.toml:83` (safetensors workspace dep); `trustformers-core/Cargo.toml:34`;
  external: `~/work/torsh` `Cargo.toml:134-152`, `TODO.md:56`.
