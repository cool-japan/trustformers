# Changelog

All notable changes to TrustformeRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-RC.1] - 2026-03-18

### Release Candidate 1

First release candidate for TrustformeRS v0.1.0.

### Changes

#### Pure Rust Compliance
- **Replaced banned compression dependencies** with OxiARC (COOLJAPAN Pure Rust Policy):
  - `flate2` → `oxiarc-deflate`
  - `zstd` → `oxiarc-zstd`
  - `lz4` → `oxiarc-lz4`
- License updated to Apache-2.0 only (COOLJAPAN Policy 2026+)

#### Code Quality
- **Zero clippy warnings** with `-D warnings` across entire workspace
- Fixed ~100+ clippy lints: `sort_by` → `sort_by_key`, `field_reassign_with_default`, `manual_checked_ops`, `module_inception`, `result_large_err`, and more
- Fixed slow `test_failover_trigger` test (180s → instant) by disabling gradual failover in test config

#### Dependencies
- Upgraded to latest stable versions (scirs2-core 0.3.3, scirs2-linalg 0.3.3, etc.)
- All workspace dependencies consolidated (110+ shared)

#### Python Bindings
- Version format updated to PEP 440 compliance (`0.1.0rc1`)

#### Project Structure
- Version consistency enforced across all workspace members
- All 12 publishable crates verified with comprehensive READMEs
- 4940 tests passing, 134 skipped

### Known Issues

1. ~4,928 `unwrap()` calls need refactoring to proper error handling
2. 18 files exceed 2,000 line refactoring policy limit
3. 4 unmaintained transitive dependencies (`tui`, `wee_alloc`, `proc-macro-error`, `rustls-pemfile`)

### Migration Notes

No breaking changes from 0.1.0-alpha.2.

---

## [0.1.0-alpha.2] - 2025-12-19

### 🚀 Performance Improvements

#### CPU Acceleration
- **17x CPU speedup** - Added direct BLAS acceleration via cblas_sgemm for matrix operations
- Integrated Accelerate framework (macOS) and OpenBLAS for optimized BLAS operations
- Added direct BLAS to SDPA (Scaled Dot-Product Attention) and Linear layers
- Added direct BLAS to FlashAttention batched matrix multiplication
- Added direct BLAS to SIMD matrix_ops kernel

#### GPU Acceleration
- **Metal (macOS)**:
  - Implemented macOS Metal support for GPU operations
  - Added MPS (Metal Performance Shaders) integration with 100-500x matmul speedup
  - Implemented Metal-to-MPS bridge via objc2-metal
  - Added GPU-to-GPU operations (eliminating CPU roundtrips)
  - **2.88x overall performance improvement** through GPU optimizations

- **CUDA (NVIDIA)**:
  - Enhanced CUDA support and kernel implementations
  - Improved CUDA feature guards and compilation

#### Memory & Efficiency
- Implemented GPU-resident tensor operations (Metal GPU-to-GPU matmul)
- Added auto-fallback from GPU to CPU when GPU unavailable
- Optimized memory transfers between CPU and GPU

### 🐛 Bug Fixes

#### Critical Fixes
- **Fixed Metal GPU kernel synchronization issues** (3 commits)
  - Flash Attention kernels producing all-zero outputs
  - Standard attention kernels failing tests
  - Fused matmul+GELU kernels returning zeros
  - Added `command_buffer.wait_until_completed()` to 4 critical operations
  - **Result**: All 20/20 flash attention tests now pass
  - **Result**: All 4/4 fused matmul+GELU tests now pass

- **Fixed safety filter stack overflow**
  - Infinite recursion in toxicity scoring causing SIGABRT
  - Enhanced harm pattern detection
  - Improved metadata enrichment logic
  - **Result**: All 19/19 safety tests now pass

- **Fixed tokio runtime conflicts**
  - Marked blocking I/O tests to avoid runtime conflicts

#### SciRS2 Policy Compliance
- Refactored to eliminate direct dependency imports (Track B compliance)
- All crates now use `scirs2_core` abstractions for scientific computing
- Removed direct `rand`, `ndarray`, `rayon` imports from model crates

### ✨ New Features

- Added MPS scaffolding to MetalBackend for Apple Silicon optimization
- Implemented comprehensive test infrastructure with memory leak detection
- Added saved failure cases for proptest regression tests
- Enhanced error handling and type safety across GPU operations

### 🧹 Code Quality

#### Clippy Improvements
- **Fixed 228 manual_div_ceil warnings** - Migrated to Rust 1.73+ `div_ceil()` method
- Fixed explicit_auto_deref warnings (unnecessary manual dereferencing)
- Fixed needless_return warnings (removed unnecessary return statements)
- Fixed manual_slice_size_calculation warnings
- Fixed unnecessary_lazy_evaluations (unwrap_or vs unwrap_or_else)
- Fixed map_entry warnings (HashMap entry API)
- Fixed empty_line_after_doc_comments warnings
- Fixed io_other_error warnings (Error::other() for Rust 1.78+)
- Fixed bind_instead_of_map warnings
- Fixed let_and_return warnings
- Fixed useless_vec warnings (vec! → arrays)
- Fixed field_reassign_with_default warnings (struct initialization)
- Fixed cloned_ref_to_slice_refs warnings (std::slice::from_ref)
- **Total**: Fixed ~300 clippy warnings across 230+ files

#### Configuration
- Updated MSRV to 1.75 (aligned clippy.toml with Cargo.toml)
- Documented remaining lint exceptions for incremental fixes
- Updated TODO.md version to 0.1.0-alpha.2

### 📚 Documentation

- Updated SIMD matrix_ops documentation to reflect direct BLAS usage
- Added comprehensive clippy lint exception documentation
- Documented SciRS2 Policy as CRITICAL PRIORITY

### 🔧 Technical Details

#### Files Changed
- 230+ files modified across the workspace
- Major changes in:
  - `trustformers-core`: GPU operations, kernels, layers, BLAS integration
  - `trustformers-models`: Model implementations
  - `trustformers-serve`: Resource management, batching
  - `trustformers-training`: Parallelism implementations
  - `trustformers-wasm`: WebGPU operations
  - `trustformers-c`: C API and codegen
  - `trustformers-optim`: Optimization algorithms

#### Test Coverage
- trustformers: 526/526 tests passing ✅
- trustformers-core flash attention: 20/20 tests passing ✅
- trustformers-core Metal GPU: 16/16 tests passing ✅
- Safety filters: 19/19 tests passing ✅

### ⚠️ Known Issues

- ~130 `field_reassign_with_default` warnings remaining in `trustformers-debug`
  - Scheduled for incremental fixes in future releases
  - Does not affect core functionality or performance

### 🔄 Migration Notes

No breaking API changes in this release. All changes are backward compatible with 0.1.0-alpha.1.

### 📦 Commit Statistics

- Total commits: 88
- Contributors: Claude Sonnet 4.5 (Co-Authored)
- Lines changed: 2000+ insertions, 600+ deletions

---

## [0.1.0-alpha.1] - 2024-XX-XX

Initial alpha release of TrustformeRS.

### Features
- 21+ transformer model implementations
- Multi-backend tensor support (CPU, CUDA, ROCm, Metal, WebGPU)
- Comprehensive tokenizer support (BPE, WordPiece, SentencePiece)
- Training infrastructure with distributed support
- Deployment targets: WASM, Python, Mobile, Server, C API
- Quantization support (GGML, GGUF, AWQ, GPTQ)
- Hardware acceleration (SIMD, GPU, TPU)

---

[0.1.0-RC.1]: https://github.com/cool-japan/trustformers/compare/0.1.0-alpha.2...0.1.0-RC.1
[0.1.0-alpha.2]: https://github.com/cool-japan/trustformers/compare/0.1.0-alpha.1...0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/cool-japan/trustformers/releases/tag/0.1.0-alpha.1
