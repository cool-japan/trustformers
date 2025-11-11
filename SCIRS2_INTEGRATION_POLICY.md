# TrustformeRS SciRS2 Integration Policy

## 🚨 CRITICAL REQUIREMENT: Complete SciRS2-Core Integration

**TrustformeRS MUST use SciRS2-Core as its complete scientific computing foundation.** This policy establishes mandatory requirements for proper integration following the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md).

**Status**: 🔴 **PARTIAL COMPLIANCE** - Systematic remediation required

## Table of Contents

1. [Core Architectural Principles](#core-architectural-principles)
2. [Dual-Layer Architecture](#dual-layer-architecture)
3. [Forbidden Direct Dependencies](#forbidden-direct-dependencies)
4. [Required Abstractions](#required-abstractions)
5. [Performance & Hardware Acceleration](#performance--hardware-acceleration)
6. [GPU Operations Critical Policy](#gpu-operations-critical-policy)
7. [Implementation Patterns](#implementation-patterns)
8. [Migration from Current State](#migration-from-current-state)
9. [Enforcement](#enforcement)

---

## Core Architectural Principles

### Layered Architecture (MANDATORY)

```
Application Layer (trustformers-models, trustformers-serve, etc.)
                    ↓ MUST use abstractions only
    TrustformeRS-Core (ML-specific: tensors, layers, models, tokenizers)
                    ↓ delegates scientific computing to
         SciRS2-Core (Scientific computing: SIMD, parallel, BLAS, GPU, random)
                    ↓ manages
      External Dependencies (rand, ndarray, rayon, cudarc, metal, etc.)
```

**Critical Rule**: Only `trustformers-core` and `scirs2-core` may import external dependencies directly.

### Compliance Requirement

**ALL TrustformeRS crates EXCEPT `trustformers-core` source code MUST**:
- ✅ Use `trustformers_core::*` for ML/DL operations
- ✅ Use `scirs2_core::*` for scientific computing
- ❌ NEVER import external dependencies directly

**Applies to**:
- All module crates (`trustformers-models`, `trustformers-training`, etc.)
- **ALL tests** (including in `trustformers-core`)
- **ALL examples** and benchmarks
- **ALL documentation code**

---

## Dual-Layer Architecture

### Layer 1: TrustformeRS-Core (ML/DL Framework)

**Purpose**: Transformer-specific operations and model abstractions

```rust
// ✅ Use trustformers-core for:
use trustformers_core::tensor::Tensor;           // Unified tensor type
use trustformers_core::device::Device;           // Device management
use trustformers_core::layers::*;                // Attention, FFN, LayerNorm, etc.
use trustformers_core::models::*;                // AutoModel, model loading
use trustformers_core::tokenizer::*;             // AutoTokenizer, encoding
use trustformers_core::quantization::*;          // Quantization operations
use trustformers_core::error::TrustformersError; // Error types
```

### Layer 2: SciRS2-Core (Scientific Computing)

**Purpose**: Scientific computing primitives (SIMD, parallel, BLAS, GPU, random)

```rust
// ✅ Use scirs2-core for:
use scirs2_core::ndarray::*;              // Arrays (Array, Array1, array!, s!)
use scirs2_core::random::*;               // RNG + distributions (Normal, Uniform, etc.)
use scirs2_core::{Complex, Complex32, Complex64};  // Complex numbers (root level)
use scirs2_core::parallel_ops::*;        // Parallel processing (rayon replacement)
use scirs2_core::simd_ops::*;            // SIMD operations
use scirs2_core::gpu_ops::*;             // GPU operations (with 'gpu' feature)
```

---

## Forbidden Direct Dependencies

### Core Scientific Computing - Use SciRS2-Core Instead

```rust
// ❌ FORBIDDEN in ALL crates (including tests)

// Random Number Generation
use rand::*;
use rand::Rng;
use rand::thread_rng;
use rand_distr::{Normal, Uniform, Beta, Gamma, Exp, Cauchy, StudentT};

// Array Operations
use ndarray::*;
use ndarray::{Array, Array1, Array2, ArrayD, IxDyn, Axis};
use ndarray::{array, s, azip};  // Macros
use ndarray_rand::*;

// Complex Numbers
use num_complex::{Complex, Complex32, Complex64};

// Numerical Traits
use num_traits::{Float, Zero, One, NumCast};
```

### Parallelization - Use SciRS2-Core Instead

```rust
// ❌ FORBIDDEN - Direct parallelization

use rayon::*;
use rayon::prelude::*;
use rayon::iter::ParallelIterator;
use rayon_core::*;
```

**Rationale**: SciRS2-core provides unified, optimized parallel operations. Direct rayon use:
1. Bypasses SciRS2's unified performance layer
2. Creates fragmented parallelization strategies
3. Prevents centralized tuning

### GPU & Hardware Acceleration - Use SciRS2-Core Features

```rust
// ❌ FORBIDDEN - Direct GPU libraries

// CUDA
use cudarc::*;
use cuda_sys::*;

// Metal
use metal::*;
use objc2_metal::*;
use objc2_metal_performance_shaders::*;

// WebGPU
use wgpu::*;
use pollster::*;

// OpenCL
use opencl3::*;

// Vulkan
use vulkano::*;
```

**Rationale**: SciRS2-core manages GPU backend selection. Use scirs2-core with feature flags:
- `features = ["gpu", "cuda"]` for NVIDIA
- `features = ["gpu", "metal"]` for Apple
- `features = ["gpu", "wgpu_backend"]` for WebGPU
- `features = ["gpu", "opencl"]` for OpenCL

### ML/DL Frameworks - Use TrustformeRS-Core Instead

```rust
// ❌ FORBIDDEN - Direct ML framework usage

// Tensor Backends
use tch::*;                    // PyTorch
use tch::Tensor;
use candle_core::*;            // Candle
use candle_core::Tensor;
use ort::*;                    // ONNX Runtime

// Tokenization
use tokenizers::*;
use tokenizers::Tokenizer;
```

**Rationale**: TrustformeRS-core provides unified abstractions. Use `trustformers_core::tensor` and `trustformers_core::tokenizer`.

---

## Required Abstractions

### Complete Import Reference Card

```rust
// ═══════════════════════════════════════════════════════════════════════
// TRUSTFORMERS-CORE USAGE (ML/DL Operations)
// ═══════════════════════════════════════════════════════════════════════

// Tensor Operations
use trustformers_core::tensor::{Tensor, TensorOps};
use trustformers_core::device::Device;

// Model Components
use trustformers_core::layers::{Linear, LayerNorm, Dropout, MultiHeadAttention};
use trustformers_core::models::{AutoModel, Model};
use trustformers_core::tokenizer::{AutoTokenizer, Tokenizer, Encoding};

// Quantization
use trustformers_core::quantization::{QuantizationEngine, QuantizationMethod};

// Error Handling
use trustformers_core::error::{TrustformersError, Result};

// ═══════════════════════════════════════════════════════════════════════
// SCIRS2-CORE USAGE (Scientific Computing)
// ═══════════════════════════════════════════════════════════════════════

// Arrays and Numerical Operations (COMPLETE functionality including macros)
use scirs2_core::ndarray::*;  // Full ndarray + ALL macros (array!, s!, azip!)
// OR selective:
use scirs2_core::ndarray::{Array, Array1, Array2, ArrayD, Axis, IxDyn, array, s};

// Random Number Generation (COMPLETE rand + rand_distr)
use scirs2_core::random::*;  // Full RNG + all distributions
// OR selective:
use scirs2_core::random::{thread_rng, Normal, Uniform, RandBeta, Gamma, Exp};

// Complex Numbers (at ROOT level, not in submodule)
use scirs2_core::{Complex, Complex32, Complex64};

// Parallel Processing (rayon replacement)
use scirs2_core::parallel_ops::*;

// SIMD Operations
use scirs2_core::simd_ops::{SimdUnifiedOps, PlatformCapabilities};

// GPU Operations (with appropriate features enabled)
use scirs2_core::gpu_ops::*;  // Requires 'gpu' feature
```

### Cargo.toml Configuration Examples

#### Application Crate (trustformers-models, etc.)

```toml
[dependencies]
# TrustformeRS abstractions
trustformers-core = { workspace = true }

# SciRS2 scientific computing (NO direct external deps!)
scirs2-core = { workspace = true, features = ["random", "parallel", "simd"] }

# ❌ FORBIDDEN (SciRS2 Policy Violations):
# rand = { workspace = true }         # Use scirs2_core::random
# ndarray = { workspace = true }      # Use scirs2_core::ndarray
# rayon = { workspace = true }        # Use scirs2_core::parallel_ops
# cudarc = "0.17"                     # Use scirs2_core with 'cuda' feature
# metal = "0.32"                      # Use scirs2_core with 'metal' feature
```

#### TrustformeRS-Core (Foundation Layer)

```toml
[dependencies]
# SciRS2 foundation
scirs2-core = { workspace = true, features = ["random", "parallel", "simd", "linalg"] }

# ML/DL frameworks (ONLY in trustformers-core!)
tokenizers = { workspace = true }
safetensors = { workspace = true }

# External dependencies (ONLY in trustformers-core!)
ndarray = { workspace = true, features = ["blas"] }
rand = { workspace = true }
rayon = { workspace = true }

# ⚠️ Note: trustformers-core re-exports these through scirs2-core for modules
```

---

## Performance & Hardware Acceleration

### SIMD Operations (MANDATORY scirs2-core)

```rust
// ✅ REQUIRED: Use scirs2-core SIMD operations
use scirs2_core::simd_ops::SimdUnifiedOps;

let result = f32::simd_add(&a.view(), &b.view());
let dot = f64::simd_dot(&x.view(), &y.view());

// ❌ FORBIDDEN: Direct SIMD
// use wide::f32x8;             // POLICY VIOLATION
// use std::arch::x86_64::*;    // POLICY VIOLATION
```

### Parallel Processing (MANDATORY scirs2-core)

```rust
// ✅ REQUIRED: Use scirs2-core parallel ops
use scirs2_core::parallel_ops::*;

let results: Vec<_> = data
    .par_iter()  // From scirs2_core::parallel_ops
    .map(|x| expensive_computation(x))
    .collect();

// ❌ FORBIDDEN: Direct Rayon
// use rayon::prelude::*;       // POLICY VIOLATION
```

### BLAS Operations (MANDATORY scirs2-core)

All BLAS operations go through scirs2-core's backend selection:

```rust
// ✅ REQUIRED: ndarray with BLAS via scirs2-core
use scirs2_core::ndarray::{Array2, s};

let a = Array2::zeros((1000, 1000));
let b = Array2::ones((1000, 1000));
let c = a.dot(&b);  // Uses Accelerate/OpenBLAS/MKL via scirs2-core

// ❌ FORBIDDEN: Direct BLAS dependency
// use cblas_sys::*;            // POLICY VIOLATION
// use openblas_src::*;         // POLICY VIOLATION
```

---

## GPU Operations Critical Policy

### The Three Rules of GPU Usage

1. **High-level tensor ops** → `trustformers_core::tensor`
2. **Low-level GPU primitives** → `scirs2_core` with GPU features
3. **Direct GPU libraries** → ❌ FORBIDDEN

### Feature Flag Configuration

```toml
# ✅ CORRECT: Enable GPU through scirs2-core features
[dependencies]
scirs2-core = { workspace = true, features = ["gpu", "metal"] }   # Apple
scirs2-core = { workspace = true, features = ["gpu", "cuda"] }    # NVIDIA
scirs2-core = { workspace = true, features = ["gpu", "wgpu_backend"] }  # WebGPU
scirs2-core = { workspace = true, features = ["gpu", "opencl"] }  # OpenCL

# ❌ INCORRECT: Direct GPU dependencies
# metal = "0.32"               # Use scirs2-core with 'metal' feature
# cudarc = "0.17"              # Use scirs2-core with 'cuda' feature
```

### GPU Usage Patterns

```rust
// ✅ High-Level (Recommended)
use trustformers_core::tensor::Tensor;
use trustformers_core::device::Device;

let device = Device::cuda_if_available()?;
let tensor = Tensor::randn(&[1024, 768])?.to_device(&device)?;
let result = tensor.matmul(&weights)?.relu()?;

// ✅ Low-Level (Advanced)
use scirs2_core::gpu_ops::*;
use scirs2_core::simd_ops::PlatformCapabilities;

let caps = PlatformCapabilities::detect();
if caps.metal_available {
    // Use GPU-accelerated operations from scirs2-core
    let result = gpu_matmul(&a, &b)?;
}

// ❌ FORBIDDEN: Direct GPU usage
// use metal::*;                // POLICY VIOLATION
// use cudarc::*;               // POLICY VIOLATION
```

### Current Issue in TrustformeRS

**Problem**: `trustformers-core/src/gpu_ops/metal.rs` implements direct Metal backend

```rust
// ❌ CURRENT (Policy Violation)
use metal::{Device as MetalDevice, CommandQueue};  // Direct Metal usage

// ✅ SHOULD BE (Policy Compliant)
use scirs2_core::gpu_ops::MetalBackend;  // Via scirs2-core
```

**Impact**: Missing SciRS2's optimized GPU operations (MPS framework, cuBLAS, etc.)

---

## Forbidden Direct Dependencies

### Complete Prohibited List

| Category | Forbidden Crates | Use Instead |
|----------|-----------------|-------------|
| **Arrays** | `ndarray`, `ndarray-rand`, `ndarray-stats` | `scirs2_core::ndarray` |
| **Random** | `rand`, `rand_distr`, `rand_chacha` | `scirs2_core::random` |
| **Complex** | `num-complex` | `scirs2_core::Complex*` |
| **Parallel** | `rayon`, `rayon-core` | `scirs2_core::parallel_ops` |
| **SIMD** | `wide`, `packed_simd`, `std::arch` | `scirs2_core::simd_ops` |
| **BLAS** | `cblas-sys`, `openblas-src`, `mkl-src` | `scirs2_core` (auto-selects) |
| **GPU-CUDA** | `cudarc`, `cuda-sys` | `scirs2_core` + `features = ["cuda"]` |
| **GPU-Metal** | `metal`, `objc2-metal`, `objc2-metal-performance-shaders` | `scirs2_core` + `features = ["metal"]` |
| **GPU-WebGPU** | `wgpu`, `pollster` | `scirs2_core` + `features = ["wgpu_backend"]` |
| **GPU-OpenCL** | `opencl3` | `scirs2_core` + `features = ["opencl"]` |
| **Tensors** | `tch`, `candle-core`, `ort` | `trustformers_core::tensor` |
| **Tokenizers** | `tokenizers` | `trustformers_core::tokenizer` |

---

## Required Abstractions

### Typical Model Implementation Imports

```rust
// Complete example for trustformers-models crate

// TrustformeRS Core (ML/DL operations)
use trustformers_core::{
    tensor::Tensor,
    device::Device,
    layers::{Linear, LayerNorm, Dropout, MultiHeadAttention},
    error::{TrustformersError, Result},
};

// SciRS2 Core (Scientific computing)
use scirs2_core::random::*;              // Weight initialization
use scirs2_core::ndarray::{Array2, s};   // If array operations needed

// Now you have access to:
// - Tensor operations (from trustformers_core)
// - Random distributions (from scirs2_core)
// - Array manipulation (from scirs2_core)
// - Parallel processing (from scirs2_core::parallel_ops)
```

### Test Code Imports

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // ✅ REQUIRED test imports
    use trustformers_core::tensor::Tensor;
    use scirs2_core::random::*;        // For test data generation
    use scirs2_core::ndarray::{array, Array1, s};  // For test assertions

    #[test]
    fn test_forward_pass() {
        let mut rng = thread_rng();  // From scirs2_core::random
        let normal = Normal::new(0.0, 1.0).unwrap();

        let input = Tensor::randn(&[2, 4, 512])?;  // From trustformers_core
        let test_arr = array![1.0, 2.0, 3.0];      // From scirs2_core::ndarray

        let model = MyModel::new(&config)?;
        let output = model.forward(input)?;

        assert_eq!(output.shape(), &[2, 4, 512]);
    }
}
```

---

## Implementation Patterns

### Pattern 1: Weight Initialization

```rust
use trustformers_core::layers::Linear;
use scirs2_core::random::*;

pub struct TransformerBlock {
    self_attn: MultiHeadAttention,
    feed_forward: FeedForward,
}

impl TransformerBlock {
    pub fn new(config: &Config) -> Result<Self> {
        // ✅ Use scirs2_core for RNG
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        Ok(Self {
            self_attn: MultiHeadAttention::new(config)?,
            feed_forward: FeedForward::new(config)?,
        })
    }
}
```

### Pattern 2: Batch Processing with Parallelization

```rust
use trustformers_core::tokenizer::AutoTokenizer;
use scirs2_core::parallel_ops::*;

pub fn batch_tokenize(texts: &[String], model: &str) -> Result<Vec<Encoding>> {
    let tokenizer = AutoTokenizer::from_pretrained(model)?;

    // ✅ Parallel processing via scirs2_core
    let results: Vec<_> = texts
        .par_iter()  // From scirs2_core::parallel_ops
        .map(|text| tokenizer.encode(text, true))
        .collect::<Result<Vec<_>>>()?;

    Ok(results)
}
```

### Pattern 3: Platform-Aware Device Selection

```rust
use trustformers_core::device::Device;
use scirs2_core::simd_ops::PlatformCapabilities;

pub fn get_optimal_device() -> Device {
    let caps = PlatformCapabilities::detect();  // From scirs2_core

    if caps.cuda_available {
        Device::cuda(0).unwrap()
    } else if caps.metal_available {
        Device::metal(0).unwrap()
    } else {
        Device::cpu()
    }
}
```

### Pattern 4: Array Operations in Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2, s};  // Complete functionality

    #[test]
    fn test_layer_forward() {
        // ✅ array! macro from scirs2_core
        let input = array![[1.0, 2.0], [3.0, 4.0]];

        // ✅ Slicing with s! macro
        let slice = input.slice(s![0..1, ..]);

        // ✅ Array construction
        let weights = Array2::zeros((4, 8));

        assert_eq!(slice.shape(), &[1, 2]);
    }
}
```

---

## Migration from Current State

### Current Violations in TrustformeRS

**Identified Issues**:

1. **trustformers-core uses direct dependencies**:
   - `use ndarray::*;` - Should use `scirs2_core::ndarray`
   - `use rand::*;` - Should use `scirs2_core::random`
   - `use rayon::*;` - Should use `scirs2_core::parallel_ops`
   - `use metal::*;` - Should use `scirs2_core` with `metal` feature

2. **trustformers-models has inline violations**:
   - `ndarray::Array2::zeros()` - Should import from scirs2_core
   - `ndarray::s![]` - Should import macro from scirs2_core

3. **Performance impact**:
   - Missing SciRS2's Accelerate BLAS integration
   - Missing SciRS2's MPS (Metal Performance Shaders)
   - Missing SciRS2's SIMD optimizations
   - Naive matmul kernel instead of optimized BLAS

### Migration Priority

#### Phase 1: Enable SciRS2 Features in Workspace (HIGH PRIORITY)
```toml
# Cargo.toml workspace dependencies
scirs2-core = { version = "0.1.0-rc.2", features = [
    "random",       # Replaces rand/rand_distr
    "parallel",     # Replaces rayon
    "simd",         # SIMD optimizations
    "linalg",       # BLAS operations
    "gpu",          # GPU base
    "metal",        # Metal + MPS (macOS)
    "cuda",         # CUDA (NVIDIA)
]}
```

#### Phase 2: Update trustformers-core to Re-Export SciRS2

```rust
// trustformers-core/src/lib.rs

// Re-export SciRS2 scientific computing (for module access)
pub use scirs2_core::{
    ndarray,        // Complete ndarray functionality
    random,         // Complete rand + rand_distr
    parallel_ops,   // Parallel processing
    simd_ops,       // SIMD operations
    Complex, Complex32, Complex64,  // Complex numbers
};

// Note: Direct usage of external deps in trustformers-core source is OK,
// but tests and examples should use scirs2_core::* imports
```

#### Phase 3: Replace GPU Backend with SciRS2

```rust
// trustformers-core/src/gpu_ops/mod.rs

#[cfg(feature = "metal")]
pub use scirs2_core::gpu_ops::MetalBackend;  // Use SciRS2's MPS implementation

#[cfg(feature = "cuda")]
pub use scirs2_core::gpu_ops::CudaBackend;   // Use SciRS2's cuBLAS integration

// Remove: trustformers-core/src/gpu_ops/metal.rs (use SciRS2 instead)
```

#### Phase 4: Fix Modules (trustformers-models, etc.)

```rust
// ❌ BEFORE (Policy Violation)
use ndarray::{Array2, s};
let arr = ndarray::Array2::zeros((10, 10));

// ✅ AFTER (Policy Compliant)
use scirs2_core::ndarray::{Array2, s};
let arr = Array2::zeros((10, 10));
```

### Expected Performance Gains from Full SciRS2 Integration

| Component | Current (naive) | With SciRS2 | Speedup |
|-----------|----------------|-------------|---------|
| matmul | Custom Metal kernel | Accelerate/MPS | **100-500x** |
| ndarray.dot() | Pure Rust | Accelerate BLAS | **10-50x** |
| Random generation | Generic impl | Optimized RNG | **2-5x** |
| Parallel ops | Manual rayon | Tuned scirs2 | **1.5-3x** |
| **TOTAL** | ~1 tok/sec | **50-200 tok/sec** | **50-200x** |

---

## Enforcement

### CI/CD Checks

```yaml
# .github/workflows/policy-check.yml
- name: SciRS2 Policy Compliance Check
  run: |
    # Scientific computing violations
    ! grep -r "^use ndarray::" trustformers-models/src trustformers-training/src
    ! grep -r "^use rand::" trustformers-models/src trustformers-training/src
    ! grep -r "^use rayon::" trustformers-models/src trustformers-training/src

    # GPU violations
    ! grep -r "^use metal::" trustformers-models/src
    ! grep -r "^use cudarc::" trustformers-models/src

    # Inline usage violations
    ! grep -r "ndarray::" trustformers-models/src | grep -v "scirs2_core::ndarray"
    ! grep -r "rand::" trustformers-models/src | grep -v "scirs2_core::random"

    # Cargo.toml violations
    ! grep -E '^(ndarray|rand|rayon|metal|cudarc)[[:space:]]*=' \
      trustformers-models/Cargo.toml \
      trustformers-training/Cargo.toml
```

### Code Review Checklist

#### Scientific Computing
- [ ] All array operations use `scirs2_core::ndarray`
- [ ] RNG uses `scirs2_core::random`
- [ ] Complex numbers use `scirs2_core::Complex*`
- [ ] No `ndarray::` inline qualified paths

#### Parallelization
- [ ] No `rayon` dependency in Cargo.toml
- [ ] Parallel ops use `scirs2_core::parallel_ops`
- [ ] No direct rayon imports

#### GPU Operations
- [ ] No direct GPU dependencies in Cargo.toml
- [ ] GPU features via `scirs2-core = { features = ["gpu", "metal"] }`
- [ ] Device management via `trustformers_core::device::Device`
- [ ] No custom Metal/CUDA kernels (use scirs2-core)

#### ML/DL Operations
- [ ] Tensors use `trustformers_core::tensor`
- [ ] Tokenization uses `trustformers_core::tokenizer`
- [ ] Layers use `trustformers_core::layers`

---

## Quick Reference

### Dos and Don'ts

#### ✅ DO

```rust
// Scientific computing
use scirs2_core::ndarray::{Array2, array, s};
use scirs2_core::random::{thread_rng, Normal};
use scirs2_core::parallel_ops::*;

// ML/DL operations
use trustformers_core::tensor::Tensor;
use trustformers_core::layers::Linear;
use trustformers_core::tokenizer::AutoTokenizer;

// Test code
#[cfg(test)]
use scirs2_core::random::*;  // For generating test data
```

#### ❌ DON'T

```rust
// ❌ Direct scientific computing imports
use ndarray::Array2;
use rand::thread_rng;
use rayon::prelude::*;

// ❌ Direct GPU imports
use metal::*;
use cudarc::*;

// ❌ Direct ML framework imports
use tokenizers::Tokenizer;
use tch::Tensor;

// ❌ Inline qualified paths
let arr = ndarray::Array2::zeros((10, 10));
let mut rng = rand::thread_rng();
```

### Import Template for New Modules

```rust
// Standard imports for TrustformeRS modules

// TrustformeRS Core (ML/DL)
use trustformers_core::{
    tensor::Tensor,
    device::Device,
    layers::{Linear, LayerNorm},
    error::{Result, TrustformersError},
};

// SciRS2 Core (Scientific Computing)
use scirs2_core::random::*;           // RNG + distributions
use scirs2_core::ndarray::{Array1, Array2, array, s};  // Arrays + macros
use scirs2_core::parallel_ops::*;    // Parallel processing (if needed)

// Module-specific imports
use crate::config::ModelConfig;
```

---

## Benefits of Full SciRS2 Integration

1. **Performance**: Accelerate BLAS (100-500x faster matmul)
2. **GPU Optimization**: MPS framework, cuBLAS, optimized kernels
3. **Consistency**: Unified APIs across all modules
4. **Maintainability**: Single dependency management point
5. **Type Safety**: No mixing of external types
6. **Cross-Platform**: Automatic backend selection
7. **Future-Proof**: Benefit from SciRS2 improvements automatically

## Current Status & Action Items

### 🔴 Current Compliance: ~30%

**Violations**:
- trustformers-core uses direct `metal`, `ndarray`, `rand`, `rayon`
- trustformers-models has inline `ndarray::` usage
- Missing SciRS2 GPU features
- No Accelerate BLAS via scirs2-core

### 🎯 Target: 100% Compliance

**Action Items** (Priority Order):
1. ✅ Enable `scirs2-core` features: `random`, `parallel`, `simd`, `linalg`, `gpu`, `metal`
2. ⏳ Update trustformers-core to delegate to scirs2-core
3. ⏳ Replace direct Metal backend with `scirs2_core::gpu_ops`
4. ⏳ Fix inline `ndarray::`, `rand::` usage in modules
5. ⏳ Remove direct GPU dependencies
6. ⏳ Benchmark performance gains

---

## Policy Version

- **Version**: 2.0.0 - Complete SciRS2 Integration
- **Effective Date**: 2025-11-11
- **Last Updated**: 2025-11-11
- **Status**: **ACTIVE - REMEDIATION REQUIRED**
- **Based On**:
  - [SciRS2 POLICY v3.0.0](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md)
  - [ToRSh SCIRS2 Policy v3.0](https://github.com/cool-japan/torsh/blob/master/SCIRS2_INTEGRATION_POLICY.md) - **96.7% compliance achieved**
  - [TensorLogic SCIRS2 Policy](https://github.com/cool-japan/tensorlogic/blob/master/SCIRS2_INTEGRATION_POLICY.md)
- **SciRS2 Version**: v0.1.0-RC.2
- **Next Review**: Q1 2026
- **Owner**: TrustformeRS Architecture Team

---

**Remember**: When in doubt, use SciRS2-Core abstractions for scientific computing and TrustformeRS-Core abstractions for ML/DL operations!
