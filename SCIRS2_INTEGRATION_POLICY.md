# TrustformeRS SciRS2 Integration Policy

## Core Architectural Principles

This document establishes the foundational policies for the TrustformeRS deep learning ecosystem to ensure consistency, maintainability, and architectural integrity across all crates. TrustformeRS leverages SciRS2-Core for scientific computing operations while maintaining its own core abstractions for transformer-specific functionality.

## Table of Contents

### Part I: Ecosystem Architecture
1. [Overview](#overview)
2. [Dependency Abstraction Policy](#dependency-abstraction-policy)
3. [Core Architectural Principles](#core-architectural-principles-1)
4. [Implementation Guidelines](#implementation-guidelines)
5. [Migration Strategy](#migration-strategy)

### Part II: Technical Policies
6. [Tensor Operations Policy](#tensor-operations-policy)
7. [SIMD Operations Policy](#simd-operations-policy)
8. [GPU Operations Policy](#gpu-operations-policy)
9. [Parallel Processing Policy](#parallel-processing-policy)
10. [BLAS Operations Policy](#blas-operations-policy)
11. [Platform Detection Policy](#platform-detection-policy)
12. [Performance Optimization Policy](#performance-optimization-policy)
13. [Error Handling Policy](#error-handling-policy)
14. [Memory Management Policy](#memory-management-policy)
15. [Tokenization Policy](#tokenization-policy)
16. [Model Architecture Policy](#model-architecture-policy)

### Part III: Implementation
17. [Refactoring Guidelines](#refactoring-guidelines)
18. [Examples](#examples)
19. [Enforcement](#enforcement)
20. [Benefits](#benefits)

---

## Part I: Ecosystem Architecture

## Overview

The trustformers-core crate serves as the central hub for all common functionality, optimizations, and abstractions used across TrustformeRS modules. For scientific computing operations, TrustformeRS integrates with SciRS2-Core. This dual-layer approach ensures:

- **Consistency**: All modules use the same optimized implementations
- **Maintainability**: Updates and improvements are made in one place
- **Performance**: Optimizations are available to all modules
- **Portability**: Platform-specific code is isolated in core
- **Version Control**: Only core manages external dependency versions
- **Type Safety**: Prevents mixing external types with TrustformeRS types
- **SciRS2 Integration**: Leverages battle-tested scientific computing primitives

## Dependency Abstraction Policy

### Core Principle: Dual-Layered Abstraction Architecture

The TrustformeRS ecosystem follows a strict layered architecture:
1. **trustformers-core**: Transformer-specific operations, models, and abstractions
2. **scirs2-core**: Scientific computing primitives (SIMD, parallel, linear algebra)
3. **Other crates**: Use only trustformers-core and scirs2-core abstractions

### Policy: No Direct External Dependencies in Non-Core Crates

**Applies to:** All TrustformeRS crates except `trustformers-core`
- `trustformers-models`, `trustformers-tokenizers`, `trustformers-serve`, etc.
- All tests, examples, benchmarks in all crates (including trustformers-core)
- All integration tests and documentation examples

#### Prohibited Direct Imports:
```rust
// ❌ FORBIDDEN in non-core crates
use rand::*;
use rand::Rng;
use rand_distr::{Normal, Uniform};  // Use scirs2_core::random instead
use ndarray::*;
use ndarray::{Array, Array1, Array2};
use ndarray::{array, s};  // Use scirs2_core::ndarray instead
use num_complex::Complex;  // Use scirs2_core::complex instead
use tokenizers::*;  // Use trustformers_core::tokenizer instead
use candle_core::*;  // Use trustformers_core::tensor instead
use tch::*;  // Use trustformers_core::tensor instead
```

#### Required TrustformeRS-Core Abstractions:
```rust
// ✅ REQUIRED in non-core crates and all tests / examples
use trustformers_core::tensor::*;        // Unified tensor operations
use trustformers_core::tokenizer::*;     // Tokenization abstractions
use trustformers_core::layers::*;        // Layer primitives (Attention, FFN, etc.)
use trustformers_core::autodiff::*;      // Automatic differentiation
use trustformers_core::quantization::*;  // Quantization operations

// For scientific computing operations, use SciRS2-Core
use scirs2_core::random::*;              // Random number generation
use scirs2_core::ndarray::*;             // Array operations (when needed)
use scirs2_core::complex::*;             // Complex numbers
use scirs2_core::simd_ops::*;            // SIMD operations
use scirs2_core::parallel_ops::*;        // Parallel processing
```

### Exception: TrustformeRS-Core Foundation Layer

**Only `trustformers-core` may use external dependencies directly:**
- ✅ `tokenizers`, `safetensors`, `candle-core`, etc.
- ✅ Direct integration with external ML libraries
- ✅ Platform-specific optimizations and SIMD operations
- ✅ `scirs2-core` for scientific computing primitives

### Benefits of This Architecture

1. **Consistent APIs**: All TrustformeRS crates use the same interfaces
2. **Version Control**: Only core manages external dependency versions
3. **Type Safety**: Prevents mixing external types with TrustformeRS types
4. **Maintainability**: Changes to external APIs only affect core
5. **Performance**: Core can optimize all external library usage
6. **Documentation**: Single source of truth for API documentation
7. **SciRS2 Integration**: Leverages proven scientific computing infrastructure

## Implementation Guidelines

### For Developers

When writing code in non-core TrustformeRS crates:

1. **Never import external crates directly**
2. **Always use TrustformeRS-Core re-exports**
3. **Use SciRS2-Core for scientific computing operations**
4. **Use TrustformeRS tensor types instead of external tensor libraries directly**
5. **Follow existing patterns in other TrustformeRS crates**

### For Tests and Examples

```rust
// ❌ Wrong - direct external usage
use rand::thread_rng;
use ndarray::{Array2, array};
use tokenizers::Tokenizer;
let mut rng = thread_rng();
let arr = array![[1, 2], [3, 4]];

// ✅ Correct - TrustformeRS-Core and SciRS2-Core unified abstractions
use trustformers_core::tokenizer::*;
use scirs2_core::random::*;
use scirs2_core::ndarray::*;

let mut rng = thread_rng();  // Available through scirs2_core
let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;
let arr = array![[1, 2], [3, 4]];  // array! macro through scirs2_core
```

## Migration Strategy

### Current Phase

1. **Phase 1**: Document policy (✅ This document)
2. **Phase 2**: Fix trustformers-c compilation issues (✅ Completed)
3. **Phase 3**: Systematic refactoring of all non-core code (In Progress)
4. **Phase 4**: Update CLAUDE.md and documentation (This step)
5. **Phase 5**: Establish CI checks to enforce policy (Planned)

---

## Part II: Technical Policies

## Tensor Operations Policy

### Mandatory Rules

1. **ALWAYS use `trustformers_core::tensor::Tensor`** for all tensor operations
2. **NEVER use external tensor libraries** (tch, candle, ndarray for tensors) directly in modules
3. **NEVER implement custom tensor operations** outside of trustformers-core
4. **ALWAYS use unified tensor API** that abstracts backend differences

### Required Usage Pattern

```rust
use trustformers_core::tensor::*;

// CORRECT - Uses unified tensor operations
let x = Tensor::randn(&[batch_size, seq_len, hidden_dim])?;
let y = x.matmul(&weights)?;
let z = y.softmax(-1)?;

// INCORRECT - Direct external tensor usage
// use tch::Tensor;  // FORBIDDEN in modules
// let x = Tensor::randn([...]);  // FORBIDDEN
```

### Available Tensor Operations

All operations are available through the `Tensor` type:

- `matmul`, `add`, `sub`, `mul`, `div` - Basic operations
- `softmax`, `log_softmax` - Activation functions
- `layer_norm`, `batch_norm` - Normalization
- `reshape`, `transpose`, `permute` - Shape operations
- `slice`, `index_select`, `gather` - Indexing operations
- `cat`, `stack`, `split` - Concatenation operations
- `mean`, `sum`, `max`, `min` - Reduction operations
- `to_device`, `to_dtype` - Device/dtype conversions

## SIMD Operations Policy

### Mandatory Rules

1. **ALWAYS use `scirs2_core::simd_ops::SimdUnifiedOps` trait** for all SIMD operations
2. **NEVER implement custom SIMD** code in individual modules
3. **NEVER use direct SIMD libraries** (wide, packed_simd, std::arch) in modules
4. **ALWAYS provide scalar fallbacks** through the unified trait

### Required Usage Pattern

```rust
use scirs2_core::simd_ops::SimdUnifiedOps;

// CORRECT - Uses unified SIMD operations
let result = f32::simd_add(&a.view(), &b.view());
let dot_product = f64::simd_dot(&x.view(), &y.view());

// INCORRECT - Direct SIMD implementation
// use wide::f32x8;  // FORBIDDEN in modules
```

### TrustformeRS-Specific SIMD Usage

For transformer-specific operations that benefit from SIMD:
- Attention score computation
- Token embedding lookups
- Layer normalization
- Activation functions (GELU, SiLU, etc.)

All these should use `scirs2_core::simd_ops` when operating on raw arrays.

## GPU Operations Policy

### Mandatory Rules

1. **ALWAYS use `trustformers_core::tensor::Tensor`** for GPU operations
2. **NEVER implement direct CUDA/Metal/ROCm kernels** in modules
3. **NEVER make direct GPU API calls** outside of core
4. **Use `trustformers_core::device::Device`** for device management

### GPU Backend Support

The core tensor module provides unified abstractions for:
- CUDA (NVIDIA)
- ROCm (AMD)
- Metal (Apple)
- WebGPU (Web/cross-platform)
- CPU fallback

### Usage Pattern

```rust
use trustformers_core::tensor::*;
use trustformers_core::device::Device;

// CORRECT - Uses core GPU abstractions
let device = Device::cuda_if_available()?;
let tensor = Tensor::randn(&[1024, 768])?.to_device(&device)?;

// INCORRECT - Direct CUDA usage
// use cuda_sys::*;  // FORBIDDEN in modules
```

## Parallel Processing Policy

### Mandatory Rules

1. **ALWAYS use `scirs2_core::parallel_ops`** for all parallel operations
2. **NEVER add direct `rayon` dependency** to module Cargo.toml files
3. **ALWAYS import via `use scirs2_core::parallel_ops::*`**
4. **NEVER use `rayon::prelude::*` directly** in modules

### Required Usage Pattern

```rust
// CORRECT - Uses scirs2-core parallel abstractions
use scirs2_core::parallel_ops::*;

let results: Vec<i32> = (0..1000)
    .into_par_iter()
    .map(|x| process_batch(x))
    .collect();

// INCORRECT - Direct Rayon usage
// use rayon::prelude::*;  // FORBIDDEN in modules
```

### Parallel Processing for ML Tasks

Common TrustformeRS use cases:
- Batch processing during inference
- Data preprocessing pipelines
- Ensemble model inference
- Tokenization of multiple texts

All should use `scirs2_core::parallel_ops`.

## BLAS Operations Policy

### Mandatory Rules

1. **ALL BLAS operations go through `scirs2-core`**
2. **NEVER add direct BLAS dependencies** to individual modules
3. **Backend selection is handled by core's platform configuration**
4. **Use feature flags through core** for BLAS backend selection

### Supported BLAS Backends

- macOS: Accelerate Framework (default)
- Linux/Windows: OpenBLAS (default)
- Intel MKL (optional)
- Netlib (fallback)

### Module Dependencies

```toml
# CORRECT - Module Cargo.toml
[dependencies]
trustformers-core = { workspace = true }
scirs2-core = { workspace = true, features = ["blas"] }

# INCORRECT - Direct BLAS dependency
# openblas-src = "0.10"  # FORBIDDEN
```

## Platform Detection Policy

### Mandatory Rules

1. **ALWAYS use `scirs2_core::simd_ops::PlatformCapabilities`** for capability detection
2. **NEVER implement custom CPU feature detection**
3. **NEVER duplicate platform detection code**

### Usage Pattern

```rust
use scirs2_core::simd_ops::PlatformCapabilities;
use trustformers_core::device::Device;

// CORRECT - Uses core platform detection
let caps = PlatformCapabilities::detect();
let device = if caps.cuda_available {
    Device::cuda(0)?
} else if caps.metal_available {
    Device::metal(0)?
} else {
    Device::cpu()
};
```

## Performance Optimization Policy

### Automatic Optimization Selection

Use dual-layer optimization:

```rust
use scirs2_core::simd_ops::AutoOptimizer;
use trustformers_core::tensor::Tensor;

let optimizer = AutoOptimizer::new();

// For tensor operations, use trustformers-core
let result = tensor.matmul(&weights)?;

// For low-level scientific operations, use scirs2-core
if optimizer.should_use_simd(problem_size) {
    // Use SIMD implementation from scirs2-core
    f32::simd_dot(&a, &b)
}
```

## Error Handling Policy

### Mandatory Rules

1. **Base all module errors on `trustformers_core::error`**
2. **Provide proper error conversions** to/from core errors
3. **Use core validation functions** for parameter checking

### Usage Pattern

```rust
use trustformers_core::error::TrustformersError;
use trustformers_core::validation::{check_shape, check_device};

// Module-specific error should derive from core
#[derive(Debug, thiserror::Error)]
pub enum ModuleError {
    #[error(transparent)]
    Core(#[from] TrustformersError),
    // Module-specific variants...
}

// Use core validation
check_shape(&tensor, &[batch, seq_len])?;
check_device(&tensor, &expected_device)?;
```

## Memory Management Policy

### Mandatory Rules

1. **Use `trustformers_core::memory` for tensor memory management**
2. **Use `scirs2_core::memory_efficient` algorithms** for large data processing
3. **Use `trustformers_core::cache` for model caching** instead of custom solutions
4. **Follow core memory pooling strategies** when available

### Memory-Efficient Tensor Operations

```rust
use trustformers_core::tensor::*;
use trustformers_core::memory::MemoryPool;

// CORRECT - Uses core memory management
let pool = MemoryPool::new()?;
let tensor = pool.allocate_tensor(&[1024, 768])?;

// Process in chunks for large sequences
for chunk in tensor.chunks(chunk_size) {
    process_chunk(chunk)?;
}

// INCORRECT - Manual memory management
// let mut buffer = vec![0.0; size];  // Don't manage memory manually
```

## Tokenization Policy

### Mandatory Rules (TrustformeRS-Specific)

1. **ALWAYS use `trustformers_core::tokenizer`** for all tokenization
2. **NEVER use tokenizers crate directly** in modules
3. **Use `AutoTokenizer` for automatic tokenizer selection**
4. **Cache tokenizers through core's caching mechanism**

### Required Usage Pattern

```rust
use trustformers_core::tokenizer::*;

// CORRECT - Uses unified tokenizer interface
let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;
let tokens = tokenizer.encode("Hello world", true)?;

// INCORRECT - Direct tokenizers usage
// use tokenizers::Tokenizer;  // FORBIDDEN in modules
```

## Model Architecture Policy

### Mandatory Rules (TrustformeRS-Specific)

1. **ALWAYS use `trustformers_core::layers`** for building models
2. **NEVER implement custom attention/FFN** outside of core
3. **Use `AutoModel` for automatic model loading**
4. **Register custom architectures through core's model registry**

### Required Usage Pattern

```rust
use trustformers_core::layers::*;
use trustformers_core::models::AutoModel;

// CORRECT - Uses core layer abstractions
pub struct MyTransformer {
    attention: MultiHeadAttention,
    ffn: FeedForward,
    norm: LayerNorm,
}

// Load pretrained models
let model = AutoModel::from_pretrained("bert-base-uncased")?;

// INCORRECT - Custom implementations
// struct MyAttention { ... }  // FORBIDDEN - use core layers
```

---

## Part III: Implementation

## Refactoring Guidelines

When encountering code that violates these policies, follow this priority order:

1. **Tensor operations** - Replace all custom tensor code with `trustformers_core::tensor`
2. **Tokenization** - Replace direct tokenizers usage with `trustformers_core::tokenizer`
3. **Model layers** - Replace custom layers with `trustformers_core::layers`
4. **SIMD implementations** - Replace custom SIMD with `scirs2_core::simd_ops`
5. **GPU implementations** - Centralize GPU operations in `trustformers_core::tensor`
6. **Parallel operations** - Replace direct Rayon usage with `scirs2_core::parallel_ops`
7. **Platform detection** - Replace with `PlatformCapabilities::detect()`
8. **Error types** - Base on trustformers-core error types
9. **Memory management** - Use core memory pools and caching

## Examples

### Example 1: Transformer Inference

```rust
use trustformers_core::tensor::*;
use trustformers_core::models::AutoModel;
use trustformers_core::tokenizer::AutoTokenizer;

pub fn run_inference(text: &str) -> Result<Tensor> {
    // Use unified abstractions
    let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;
    let model = AutoModel::from_pretrained("bert-base-uncased")?;

    let tokens = tokenizer.encode(text, true)?;
    let input_ids = Tensor::from_slice(&tokens.input_ids)?;

    let output = model.forward(input_ids)?;
    Ok(output)
}
```

### Example 2: Custom Layer with SIMD

```rust
use trustformers_core::layers::Layer;
use trustformers_core::tensor::Tensor;
use scirs2_core::simd_ops::SimdUnifiedOps;

pub struct OptimizedLayer {
    weights: Tensor,
}

impl Layer for OptimizedLayer {
    fn forward(&self, input: Tensor) -> Result<Tensor> {
        // Use trustformers-core for tensor ops
        let matmul_result = input.matmul(&self.weights)?;

        // Use scirs2-core for low-level SIMD when needed
        // (Only in performance-critical paths)
        Ok(matmul_result)
    }
}
```

### Example 3: Batch Processing

```rust
use trustformers_core::tokenizer::AutoTokenizer;
use scirs2_core::parallel_ops::*;

pub fn batch_tokenize(texts: &[String]) -> Vec<TokenizedInput> {
    let tokenizer = AutoTokenizer::from_pretrained("gpt2").unwrap();

    // Use scirs2-core parallel processing
    texts.par_iter()
        .map(|text| tokenizer.encode(text, true).unwrap())
        .collect()
}
```

### Example 4: Model Quantization

```rust
use trustformers_core::quantization::*;
use trustformers_core::models::AutoModel;

pub fn quantize_model(model_name: &str) -> Result<QuantizedModel> {
    let model = AutoModel::from_pretrained(model_name)?;

    // Use core quantization
    let quantizer = QuantizationEngine::new(QuantizationConfig {
        method: QuantizationMethod::INT8,
        calibration_samples: 100,
        ..Default::default()
    })?;

    quantizer.quantize_model(&model)
}
```

### Example 5: Platform-Aware Inference

```rust
use trustformers_core::device::Device;
use scirs2_core::simd_ops::PlatformCapabilities;

pub fn get_best_device() -> Device {
    let caps = PlatformCapabilities::detect();

    if caps.cuda_available {
        Device::cuda(0).unwrap()
    } else if caps.metal_available {
        Device::metal(0).unwrap()
    } else {
        Device::cpu()
    }
}
```

## Enforcement

### Automated Checks (Future)
- CI pipeline checks for prohibited imports
- `cargo deny` configuration for dependency restrictions
- Custom linting rules for TrustformeRS ecosystem
- Integration tests that verify policy compliance

### Manual Review
- All PRs must follow this policy
- Code reviews must verify TrustformeRS-Core and SciRS2-Core usage
- Examples and tests must demonstrate proper patterns

### Current Enforcement
- Code reviews MUST check for policy compliance
- Regular audits should identify and refactor non-compliant code
- New modules MUST follow these policies from the start

## Benefits

By following these policies, we achieve:

1. **Unified Performance**: All modules benefit from optimizations
2. **Easier Maintenance**: Updates in one place benefit all modules
3. **Consistent Behavior**: Same optimizations across the ecosystem
4. **Better Testing**: Centralized testing of critical operations
5. **Improved Portability**: Platform-specific code is isolated
6. **Reduced Duplication**: No repeated implementation of common operations
7. **Version Control**: Simplified dependency management
8. **Type Safety**: Consistent types across the ecosystem
9. **SciRS2 Integration**: Leverage proven scientific computing infrastructure
10. **ML Optimization**: Transformer-specific optimizations in core

## Integration with SciRS2

TrustformeRS leverages SciRS2-Core for:
- ✅ SIMD operations for performance-critical numeric computations
- ✅ Parallel processing for batch operations
- ✅ BLAS operations for linear algebra
- ✅ Random number generation for initialization and sampling
- ✅ Platform capability detection
- ✅ Memory-efficient algorithms for large-scale processing

TrustformeRS-Core provides:
- ✅ Tensor abstractions with automatic differentiation
- ✅ Tokenization with multiple backend support
- ✅ Transformer layers (Attention, FFN, LayerNorm, etc.)
- ✅ Model loading and serialization (SafeTensors, GGUF, etc.)
- ✅ Quantization algorithms (GGML, AWQ, SmoothQuant, etc.)
- ✅ Device management (CUDA, Metal, ROCm, WebGPU)

## Inspiration

This policy is inspired by:
- **SciRS2 Ecosystem**: Proven dependency abstraction patterns
- **OxiRS**: Graph processing abstraction layers
- **HuggingFace Transformers**: Unified model and tokenizer interfaces
- **PyTorch**: Tensor abstraction and device management

## Questions or Clarifications

If you have questions about these policies or need clarification on specific use cases, please:

1. Check the `trustformers-core` documentation
2. Review existing implementations in other modules
3. Check SciRS2-Core documentation for scientific computing operations
4. Open an issue for discussion
5. Consult with the core team

Remember: When in doubt, use the core abstractions!

## Policy Version
- **Version**: 1.0.0
- **Effective Date**: 2025-10-01
- **Last Updated**: 2025-10-01
- **Status**: Active
- **Based on**: SciRS2 Policy v2.0.0

---

*This policy ensures the TrustformeRS ecosystem remains maintainable, consistent, and high-performance as it scales to support the broader ML and transformer community.*
