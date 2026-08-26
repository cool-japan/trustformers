# TODO-CUDA.md — Work requiring a real NVIDIA (CUDA) GPU

> **Branch:** `0.2.1` · **Generated:** 2026-06-29 · **Companion to:** `TODO.md` ("Hardware-gated remaining work" section)

## ✅ STATUS 2026-06-29 (Campaign E) — oxicuda migration COMPLETE on RTX A4000

This checklist was picked up on a **Linux x86_64 + RTX A4000 (Ampere sm_86) + CUDA 12.0** box.
The full migration **OFF cudarc ONTO oxicuda** is done and runtime-verified on hardware:

| Item | Status |
|------|--------|
| CUDA-0 smoke test | ✅ builds + runs on A4000 |
| **CUDA-1** runtime parity (BLOCKER) | ✅ **12/12 oxicuda parity tests pass.** layer_norm PTX bug fixed (`.maxntid` was emitted inside the body with a trailing `;` — moved to a directive; also fixed f16/bf16 `ld/st.global.f16`→`.b16`). The GEMM RowMajor bug was fixed by the repo owner's in-progress `gemm/` rewrite (verified: 3/3 GEMM parity green). |
| **CUDA-2** drop cudarc | ✅ `cuda` feature now pulls oxicuda; `cuda-oxicuda = ["cuda"]` deprecated alias; cudarc dep removed; cudarc gone from `cargo tree`; deleted `cuda_split/`, `advanced_kernels.rs`, `kernels/cuda_impl.rs`, `kernels/cuda_kernels.rs`, legacy `gpu_ops/cuda/{backend,types,buffer_ops}.rs`. `BufferId`/`get_cuda_backend` re-exported from oxicuda. |
| **CUDA-3** disabled kernels | ✅ resolved **by removal** (the cudarc `advanced_kernels`/`cuda_impl`/fake `cuda_kernels` are deleted, replaced by the oxicuda backend). ⏳ FOLLOW-UP: `gpu_accelerated`/`hardware_acceleration` stay `#[cfg(not(feature="cuda"))]` — they call the removed cudarc kernel API; porting them onto oxicuda is future work. |
| **CUDA-4** zero-copy residency | ✅ confirmed: `OxicudaCudaBackend::matmul_gpu_to_gpu` operates on cached `DeviceBuffer<f32>` and allocates output on-device — no host round-trip (resident parity tests green). |
| **CUDA-5** feature propagation | ✅ `trustformers-models` (already `cuda = ["trustformers-core/cuda"]`) + umbrella `trustformers` (`cuda = ["trustformers-models/cuda"]` added); both build `--features cuda`. |
| **CUDA-6** fused transformer-layer kernel | ⏳ OPTIMIZATION, deferred. NOTE: the cudarc `cuda_backend_ext.rs` fused-layer (Campaign D) was deleted with `cuda_split`; the oxicuda path runs the layer as individual ops (correct, unfused). |
| **CUDA-7** gpt_neox GPU residency | ⏳ documented honestly: oxicuda is host-in/host-out (no on-device QKV-split/RoPE/attention), so GPU operands are downloaded to CPU F32 for attention (correct, not a stub). On-device residency = future work. |
| **CUDA-8** RoPE convention | ✅ resolved (Campaign D: live path is half-split). The orphaned interleaved `rope/mod.rs` (~1693 L dead) is now **deleted** this campaign. |
| **CUDA-9** legacy `trustformers-c` | 🔒 out of scope (excluded-from-workspace crate; cudarc 0.17). |

**Remaining CUDA follow-ups (not blockers):** CUDA-6 (fused megakernel — perf), CUDA-7 (on-device
attention residency — needs oxicuda-dnn resident attention), CUDA-3 follow-up (port
`gpu_accelerated`/`hardware_acceleration` onto oxicuda), CUDA-9 (legacy crate). The oxicuda repo
(`oxicuda/`) still has the owner's large uncommitted in-progress rewrite (gemm/conv/
solver/sparse) with `cargo fmt` diffs — left untouched; the owner finishes/commits it.

---

The items below are the original (pre-Campaign-E) checklist, kept for reference. Most are now ✅ per the table above.

Every item below was **implemented and compile/clippy-validated on macOS (Apple Silicon)** but **could not be runtime-verified** — the development machine had no NVIDIA GPU. oxicuda runtime-loads `libcuda` (via `libloading`), so the `cuda-oxicuda` backend *compiles* on macOS but returns `UnsupportedPlatform` at runtime; the legacy `cuda` (cudarc) backend does not build on macOS at all. This file was the pickup checklist for a Linux/Windows host with a real NVIDIA GPU.

**Nothing here was a known bug** (except CUDA-8, a latent RoPE inconsistency). It was verification, plus the changes that are only safe to make once verified on hardware.

---

## Environment prerequisites

- **GPU:** any CUDA-capable NVIDIA GPU with a current driver (`libcuda.so` on Linux / `nvcuda.dll` on Windows).
- **CUDA SDK / nvcc:** **not** required for the `cuda-oxicuda` (oxicuda) path — it `dlopen`s `libcuda` at runtime. The legacy `cuda` (cudarc) path expects a CUDA **12.x** runtime (cudarc feature `cuda-12000`).
- **OS:** Linux or Windows. The parity tests are gated `#[cfg(all(test, feature = "cuda-oxicuda", any(target_os = "linux", target_os = "windows")))]` — they do not even compile on macOS.
- **Rust:** the workspace toolchain (oxicuda requires rustc ≥ 1.85).
- **oxicuda:** path dependency `path = "../oxicuda"`, branch `0.4.0` (already committed there). Ensure `~/work/oxicuda` is checked out next to the trustformers repo.

---

## 0 — Smoke test (run this first)

```bash
# Builds the oxicuda CUDA backend and runs its 10 GPU parity tests.
cargo build      -p trustformers-core --features cuda-oxicuda
cargo nextest run -p trustformers-core --features cuda-oxicuda -E 'test(oxicuda_cuda_)'
```

On macOS these tests are not compiled; on an NVIDIA host they should compile and **execute**. If `OxicudaCudaBackend::new()` errors, the GPU/driver isn't visible — fix the environment before proceeding.

---

## CUDA-1 — [BLOCKER] Runtime parity verification of the `cuda-oxicuda` backend

The Pure-Rust backend `OxicudaCudaBackend` (`trustformers-core/src/gpu_ops/cuda/oxicuda/mod.rs`) has full op-surface parity (18 public ops) with the cudarc backend, and 10 golden-parity tests are written but have **never executed**. Run them and confirm each matches the CPU reference within tolerance:

| # | Test (`gpu_ops::cuda::oxicuda::tests`) | Op |
|---|----------------------------------------|-----|
| 1 | `oxicuda_cuda_matmul_parity` (mod.rs:1276) | host GEMM |
| 2 | `oxicuda_cuda_gelu_parity` (1318) | host GELU |
| 3 | `oxicuda_cuda_layernorm_parity` (1353) | host LayerNorm |
| 4 | `oxicuda_cuda_softmax_causal_parity` (1407) | host causal softmax |
| 5 | `oxicuda_cuda_rope_parity` (1465) | host RoPE (see CUDA-8) |
| 6 | `oxicuda_cuda_resident_matmul_parity` (1526) | GPU-resident GEMM |
| 7 | `oxicuda_cuda_resident_gelu_parity` (1602) | GPU-resident GELU |
| 8 | `oxicuda_cuda_resident_add_bias_parity` (1641) | GPU-resident bias-add |
| 9 | `oxicuda_cuda_resident_layernorm_parity` (1684) | GPU-resident LayerNorm |
| 10 | `oxicuda_cuda_matmul_with_cached_weight_parity` (1753) | cached-weight GEMM |

**Acceptance:** all 10 green on real hardware. If any fails, the likely culprits are kernel numeric conventions (GELU exact-vs-approx, LayerNorm epsilon placement, RoPE theta/half-split, causal-softmax masking) — fix in oxicuda (`~/work/oxicuda`; the kernels added this cycle are `causal_softmax`, `rope_neox_half_split`, `bias_add`) and re-run.

**Then broaden:**
```bash
cargo nextest run -p trustformers-core --features cuda-oxicuda
cargo nextest run --workspace --all-features   # confirm nothing else regressed on a GPU host
```

---

## CUDA-2 — Drop `cudarc` and the legacy duplicate backend (Campaign C / C3)

**Only after CUDA-1 is green.** The working cudarc path must not be removed until oxicuda is hardware-proven. Then:

1. Switch the default CUDA dispatch from the cudarc backend to `OxicudaCudaBackend` (`tensor/math_ops/linear_algebra.rs` dispatch + `layers/linear.rs`).
2. Delete the cudarc dependency and feature:
   - `trustformers-core/Cargo.toml:113` — `cuda = ["dep:cudarc"]`
   - `trustformers-core/Cargo.toml:148` — `cudarc = { version = "0.19", optional = true, features = ["nvrtc", "cuda-12000"] }`
3. Remove the cudarc-using sources (13 files; `gpu_ops/cuda/oxicuda/mod.rs` stays — it only *mentions* cudarc in doc comments):
   `lib.rs`, `gpu_ops/mod.rs`, `gpu_ops/advanced_kernels.rs`, `gpu_ops/cuda/{backend,types,buffer_ops}.rs`, `gpu_ops/cuda/cuda_split/{cuda_backend,cuda_backend_ext,cuda_dispatch,cuda_types}.rs`, `kernels/{mod,cuda_impl}.rs`, `tensor/math_ops/linear_algebra.rs`.

**Acceptance:** `grep -rn cudarc trustformers-core/src` shows only doc-comment mentions (ideally none); `cargo tree -p trustformers-core --features cuda-oxicuda -e normal` shows oxicuda and no cudarc; build/clippy/nextest green on the GPU host.

---

## CUDA-3 — Re-enable the kernels disabled "pending cudarc API migration"

`gpu_ops/advanced_kernels.rs` and `kernels/cuda_impl.rs` were parked during the migration. Re-enable them against oxicuda (not cudarc) and verify on hardware. Fold into the CUDA-2 pass.

---

## CUDA-4 — CUDA GPU-residency zero-copy (`DeviceBuffer::from_raw`) — task #25

oxicuda 0.4.0 added `DeviceBuffer::from_raw` (a non-owning device-pointer view). Confirm `OxicudaCudaBackend`'s resident `matmul_gpu_to_gpu` path is genuinely zero-copy (no host round-trip) on hardware — the resident-buffer parity tests (#6–9 above) cover correctness; profile to confirm residency.

---

## CUDA-5 — Propagate `cuda-oxicuda` to models / umbrella

Once core is verified, thread the `cuda-oxicuda` feature through `trustformers-models` and the `trustformers` umbrella (today only `cuda` is plumbed). Add pass-through features mirroring the existing `cuda` ones.

---

## CUDA-6 — [OPTIMIZATION] Fused transformer-layer CUDA kernel

`gpu_ops/cuda/cuda_split/cuda_backend_ext.rs:457` — `run_fused_transformer_layer` currently executes LayerNorm → QKV → RoPE → Attention → Proj → Residual **individually** (correct but unfused). Implement a single fused kernel path (oxicuda-dnn or a hand-written kernel). Benchmark vs the individual-op path. *Optimization, not a correctness gap.*

---

## CUDA-7 — GPT-NeoX GPU attention residency

`trustformers-models/src/gpt_neox/model.rs:165` — `// TODO: Implement full Tensor::Metal/CUDA support in Attention`. GPU GPT-NeoX attention currently downcasts QKV to CPU F32 (loses GPU residency). Implement on-device QKV split + RoPE for CUDA tensors. (The Metal half is verifiable on the Mac; the CUDA half is NVIDIA-gated.)

---

## CUDA-8 — RoPE CPU/CUDA convention reconciliation (latent bug)

trustformers' RoPE is internally inconsistent: the CPU reference (`rope/mod.rs`) uses the **interleaved** convention, while the CUDA kernel (and the new oxicuda `rope_neox_half_split`) use the **GPT-NeoX half-split** convention — so CPU and CUDA RoPE produce different results. On a GPU host, decide the canonical convention (recommended: **half-split**, matching the kernels and HF GPT-NeoX/LLaMA), align the CPU reference to it, and confirm `oxicuda_cuda_rope_parity` passes against the corrected CPU reference. This is the one item that is *also* a latent correctness bug, not merely unverified.

---

## CUDA-9 — [LOW] Legacy `trustformers-c` CUDA (out-of-workspace)

`trustformers-c/src/cuda.rs` uses the **cudarc 0.17** API (`CudaContext`/`CudaSlice`/`CudaStream`) and copies matmul via host memory. This crate is excluded from the workspace. Lowest priority: eventually migrate it to oxicuda (Pure-Rust policy) or retire it. Not part of the core `cuda-oxicuda` verification.

---

## Verification cheat-sheet

```bash
# Build (no GPU needed — compiles on the host)
cargo build      -p trustformers-core --features cuda-oxicuda
cargo clippy     -p trustformers-core --features cuda-oxicuda --all-targets -- -D warnings

# Run parity (needs the NVIDIA GPU)
cargo nextest run -p trustformers-core --features cuda-oxicuda -E 'test(oxicuda_cuda_)'

# After CUDA-2/3 (cudarc removed)
grep -rn cudarc trustformers-core/src                                              # → doc-comment mentions only, ideally none
cargo tree -p trustformers-core --features cuda-oxicuda -e normal | grep -i cudarc # → empty
cargo nextest run --workspace --all-features
```

## See also
- `TODO.md` → **"Hardware-gated remaining work — consolidated status (2026-06-29)"** — the full blocked set, including the non-CUDA items (ROCm/HIP, cloud/ASIC).
- `TODO.md` → **"Campaign C — oxicuda CUDA + Metal backend migration"** — design context and the C0–C6 phase breakdown.
