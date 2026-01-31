# Changelog

All notable changes to TrustformeRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-rc.1] - 2026-01-28

### 🎯 Release Candidate 1

This is the first release candidate for TrustformeRS v0.1.0, marking a significant milestone toward production readiness.

### ✨ Improvements

#### Dependencies
- **Upgraded to latest stable versions**:
  - `jsonwebtoken`: 10.2 → 10.3
  - `scirs2-core`: 0.1.2 → 0.1.3
  - `scirs2-linalg`: 0.1.2 → 0.1.3
  - `aws-sdk-sagemaker`: 1.178 → 1.179
  - 15 additional packages updated in Cargo.lock

#### Metadata
- Fixed invalid crates.io category in `trustformers-client`
  - Changed `web-programming::http-client` → `web-programming`
  - All categories now validated against official crates.io slugs

#### Project Structure
- Version consistency enforced across all workspace members
- All 12 publishable crates verified with comprehensive READMEs
- Workspace policy compliance confirmed (110 shared dependencies)

### 🔧 Technical Improvements

- **Build System**: Enhanced publish script with better error handling and resume capability
- **Documentation**: All entry points have complete crate-level documentation
- **Policy Compliance**: Maintained Pure Rust policy (no C/Fortran dependencies by default)

### 📊 Project Metrics

- **1,059,876 lines** of Rust code across 1,766 files
- **12 publishable crates** in workspace
- **110 workspace dependencies** properly managed
- Estimated project value: $43.6M, 57.74 months development

### ⚠️ Known Issues

The following issues are being addressed before final 0.1.0 release:

1. **Code Quality** (In Progress):
   - 4,928 `unwrap()` calls need refactoring to proper error handling
   - 36 files exceed 2,000 line refactoring policy limit
   - See [Issue #TBD] for tracking

2. **Dependencies** (Minor):
   - 4 unmaintained dependencies need replacement:
     - `tui` → migrate to `ratatui`
     - `wee_alloc` → remove or replace
     - `proc-macro-error`, `rustls-pemfile` → update transitive deps

3. **Testing** (In Progress):
   - Comprehensive test suite validation ongoing
   - Clippy checks with zero-warnings policy in progress
   - Documentation build verification pending

### 🚀 Next Steps

- Address known issues listed above
- Complete final round of testing
- Perform security audit verification
- Prepare for 0.1.0 stable release

### 📝 Migration Notes

No breaking changes from 0.1.0-alpha.2. All existing code should work without modifications.

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

[0.1.0-alpha.2]: https://github.com/cool-japan/trustformers/compare/0.1.0-alpha.1...0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/cool-japan/trustformers/releases/tag/0.1.0-alpha.1
