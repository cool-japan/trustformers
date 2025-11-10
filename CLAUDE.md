# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TrustformeRS is a high-performance, memory-safe Rust implementation of Hugging Face Transformers. The project includes 15+ transformer architectures (BERT, GPT-2, T5, LLaMA, Mistral, Gemma, Qwen, multimodal models, etc.) with support for multiple deployment targets (WASM, Python, Mobile, Server, C API).

## Build and Development Commands

### Essential Commands

```bash
# Run all checks (format, clippy, tests, docs)
make check

# Run tests with nextest (preferred test runner)
cargo nextest run --all-features

# Run tests for a specific crate
cargo nextest run -p trustformers-core --all-features

# Run a single test
cargo nextest run -p trustformers-models --test '*' test_name

# Run standard tests (for doctests)
cargo test --doc --all-features

# Format code
cargo fmt --all

# Run clippy linting
cargo clippy --all-targets --all-features -- -D warnings

# Build documentation
cargo doc --all-features --no-deps

# Build release
cargo build --release --all-features
```

### Testing Commands

```bash
# Run all tests with no-fail-fast
make test

# Run integration tests only
make integration

# Run benchmarks
cargo criterion

# Generate test coverage
cargo tarpaulin --all-features --workspace --timeout 600 --out html

# Run property tests
cargo test --test property_tests --all-features

# Run fuzz tests
cargo test --test fuzz_tests --all-features
```

### Development Tools

```bash
# Install development tools
make install-tools

# Quick development check
make quick

# Watch for changes and run tests
cargo watch -x "nextest run" -x "clippy -- -D warnings"

# Check for security vulnerabilities
cargo audit

# Check dependencies
cargo deny check

# Check for typos
typos
```

## Architecture Overview

### Workspace Structure

The project is organized as a Cargo workspace with the following crates:

- **trustformers-core**: Core abstractions, tensor operations, traits (Model, Layer, Config), GPU support, quantization, SIMD kernels, autodiff, hardware acceleration
- **trustformers-models**: Model implementations (BERT, GPT-2, T5, LLaMA, Mistral, Gemma, Qwen, CLIP, ViT, BLIP-2, LLaVA, DALL-E, Flamingo, Mamba, RWKV, etc.)
- **trustformers-tokenizers**: Tokenizer implementations (BPE, WordPiece, SentencePiece)
- **trustformers-optim**: Optimizers (Adam, AdamW, SGD) and learning rate schedulers
- **trustformers-training**: Training infrastructure, distributed training, ZeRO optimization
- **trustformers**: Main integration crate with high-level APIs (AutoModel, AutoTokenizer, pipeline functions)
- **trustformers-wasm**: WebAssembly deployment with WebGPU support
- **trustformers-py**: Python bindings (PyO3)
- **trustformers-mobile**: Mobile deployment (iOS/Android)
- **trustformers-serve**: REST API server with Kubernetes deployment
- **trustformers-debug**: Debugging tools, profilers, visualizers
- **trustformers-c**: C API for FFI

### Dual-Layer Architecture

TrustformeRS uses a dual-layer architecture for optimal performance and maintainability:

```
Application Crates (trustformers-models, trustformers-tokenizers, etc.)
                    ↓
    TrustformeRS-Core (ML-specific: tensors, layers, models)
                    ↓
         SciRS2-Core (Scientific computing: SIMD, parallel, BLAS)
                    ↓
      External Dependencies (rand, ndarray, tokenizers, etc.)
```

**Key Principle**: Only `trustformers-core` can use external dependencies directly. All other crates must use abstractions from either `trustformers-core` or `scirs2-core`. **See [Critical Development Policies](#critical-development-policies) below for detailed requirements.**

## Critical Development Policies

**READ THIS FIRST**: TrustformeRS follows **two distinct dependency policies**. Understanding the difference is critical for proper development.

### 📊 Quick Reference: Which Policy Applies?

| Dependency Type | Example Libraries | Policy | Import Via |
|----------------|-------------------|--------|------------|
| **ML/DL Frameworks** | `tch`, `candle_core`, `ort` | **TrustformeRS Core Usage** | `trustformers_core::tensor` |
| **Tokenization** | `tokenizers` | **TrustformeRS Core Usage** | `trustformers_core::Tokenizer` |
| **Scientific Computing** | `rand`, `ndarray` | **SciRS2 Integration** | `scirs2_core::random`, `scirs2_core::ndarray` |
| **Parallelization** | `rayon` | **SciRS2 Integration** | `scirs2_core::parallel_ops` |
| **GPU Libraries** | `cudarc`, `wgpu`, `metal` | **SciRS2 Integration** | `scirs2_core` features |

---

### 1️⃣ TrustformeRS Core Usage Policy

**Scope**: ML/Deep Learning frameworks and model-specific libraries

**Rule**: Only `trustformers-core` can import ML framework dependencies directly. All other crates must use `trustformers_core` abstractions.

#### Prohibited Direct Imports

```rust
// ❌ FORBIDDEN in ALL crates except trustformers-core source

// ML/DL Tensor Backends
use tch::*;                    // PyTorch - Use trustformers_core::tensor instead
use tch::Tensor;
use candle_core::*;            // Candle - Use trustformers_core::tensor instead
use candle_core::Tensor;
use ort::*;                    // ONNX Runtime - Use trustformers_core::tensor instead

// Tokenization Libraries
use tokenizers::*;             // HuggingFace - Use trustformers_core::Tokenizer instead
use tokenizers::Tokenizer;
```

#### Required Abstractions

```rust
// ✅ CORRECT - TrustformeRS Core Usage Policy

// Tensor Operations
use trustformers_core::tensor::Tensor;        // Unified tensor type
use trustformers_core::tensor::*;             // Tensor operations
use trustformers_core::device::Device;        // Device management (CPU/GPU)

// ML Components
use trustformers_core::layers::*;             // Attention, FFN, LayerNorm, etc.
use trustformers_core::models::*;             // AutoModel and model loading
use trustformers_core::quantization::*;       // Quantization operations

// Tokenization
use trustformers_core::{Tokenizer, Encoding}; // Re-exported from tokenizers crate

// Example: Typical model file imports
use trustformers_core::{
    tensor::Tensor,
    device::Device,
    layers::{Linear, LayerNorm, Dropout},
    error::TrustformersError,
};
```

---

### 2️⃣ SciRS2 Integration Policy

**Scope**: Scientific computing and numerical libraries

**Rule**: Only `trustformers-core` and `scirs2-core` can import scientific computing dependencies. All other crates must use `scirs2_core` abstractions.

**Full documentation**: See `SCIRS2_INTEGRATION_POLICY.md`

#### Prohibited Direct Imports

```rust
// ❌ FORBIDDEN in ALL crates except trustformers-core source

// Scientific Computing
use rand::*;
use rand::Rng;
use rand_distr::{Normal, Uniform, Beta};
use ndarray::*;
use ndarray::{Array, Array1, Array2, array, s};
use num_complex::Complex;

// Parallelization and SIMD
use rayon::*;
use rayon::prelude::*;

// GPU and Hardware Acceleration (low-level)
use cudarc::*;       // Use scirs2_core with 'cuda' feature
use wgpu::*;         // Use scirs2_core with 'wgpu_backend' feature
use metal::*;        // Use scirs2_core with 'metal' feature
use opencl3::*;      // Use scirs2_core with 'opencl' feature
use vulkano::*;      // Use scirs2_core with 'vulkan' feature
```

#### Required Abstractions

```rust
// ✅ CORRECT - SciRS2 Integration Policy

// Random Number Generation
use scirs2_core::random::*;              // RNG, distributions (Normal, Beta, etc.)

// Array Operations
use scirs2_core::ndarray::*;             // Arrays, array! macro, s! macro
use scirs2_core::ndarray::{Array1, Array2, ArrayD, Axis, s};

// Complex Numbers
use scirs2_core::Complex;                // Complex numbers (at root level)

// Parallelization
use scirs2_core::parallel_ops::*;        // Parallel processing (via rayon)

// SIMD Operations
use scirs2_core::simd_ops::*;            // SIMD operations

// GPU Operations (via features)
// Enable in Cargo.toml:
// scirs2-core = { workspace = true, features = ["gpu", "cuda"] }    # NVIDIA
// scirs2-core = { workspace = true, features = ["gpu", "metal"] }   # Apple
// scirs2-core = { workspace = true, features = ["gpu", "wgpu_backend"] } # WebGPU
// scirs2-core = { workspace = true, features = ["gpu", "opencl"] }  # OpenCL
use scirs2_core::gpu_ops::*;             // Low-level GPU operations (with 'gpu' feature)
use scirs2_core::simd_ops::PlatformCapabilities; // GPU detection

// Example: Scientific computing with random initialization
use scirs2_core::random::*;
let mut rng = thread_rng();
let dist = Normal::new(0.0, 1.0)?;
```

---

### ⚙️ General Policy Rules

**Applies to BOTH policies:**

1. **In trustformers-models, trustformers-tokenizers, trustformers-serve, etc.**:
   - ❌ NEVER import external dependencies directly
   - ✅ ALWAYS use `trustformers_core` or `scirs2_core` abstractions

2. **In ALL tests and examples (including in trustformers-core)**:
   - ❌ NEVER import external crates directly
   - ✅ ALWAYS use the unified abstractions

3. **Only in trustformers-core source code**:
   - ✅ Can import external dependencies directly
   - ✅ Must re-export them for other crates to use

---

### 💡 Complete Example: Combining Both Policies

```rust
// A typical model implementation file should use BOTH policies:

// TrustformeRS Core Usage Policy (ML frameworks)
use trustformers_core::{
    tensor::Tensor,
    device::Device,
    layers::{Linear, LayerNorm, Dropout},
    error::TrustformersError,
};

// SciRS2 Integration Policy (scientific computing)
use scirs2_core::random::*;              // For weight initialization
use scirs2_core::ndarray::{Array2, s};   // If needed for array operations

// Now you can use unified APIs
let mut rng = thread_rng();              // From scirs2_core
let tensor = Tensor::randn(&[512, 768])?;  // From trustformers_core
let device = Device::cuda_if_available()?; // From trustformers_core
```

### ✅ Recent SciRS2 Policy Remediation (2025)

**Status**: 100% Compliant across all crates

A comprehensive remediation effort was completed to ensure full compliance with the SciRS2 Integration Policy. This involved systematic fixes across the entire codebase.

#### Remediation Statistics

- **Total Inline Violations Fixed**: 250+ instances
- **Files Modified**: 14 files across 2 crates
- **Crates Remediated**: `trustformers-models`, `trustformers-training`
- **Compilation Status**: ✅ All crates compile successfully with no warnings

#### Files Remediated in trustformers-models (13 files)

1. **bert/layers.rs** - Fixed inline `ndarray::s![]` slice macro usage
2. **bert/model.rs** - Added `ArrayD, IxDyn` imports, replaced inline usages
3. **distilbert/model.rs** - Added `IxDyn` import, replaced inline usages
4. **gpt2/model.rs** - Added `s, Axis` imports, replaced inline usages
5. **gpt_j/model.rs** - Fixed model and test code
6. **gpt_neo/model.rs** - Fixed model and test code
7. **roberta/model.rs** - Added `ArrayD, IxDyn` imports
8. **rwkv/model.rs** - Added `Array1` import
9. **vit/tests.rs** - Changed `use ndarray::Array4` to `use scirs2_core::ndarray::Array4`
10. **vit/model.rs** - Added `concatenate` import
11. **t5/model.rs** - Added `Array2` import
12. **falcon/model.rs** - Added `s` import for slice operations
13. **command_r/model.rs** - Comprehensive import additions

#### Files Remediated in trustformers-training (8 files)

1. **few_shot/prompt_tuning.rs** - Fixed `Uniform::new()` Result type handling with `?` operator
2. **few_shot/task_adaptation.rs** - Fixed `Uniform::new()` Result type handling
3. **few_shot/cross_task.rs** - Added `.expect()` to `Uniform::new()` call
4. **hyperopt/search_space.rs** - Removed direct `rand_distr` imports (already via scirs2_core)
5. **hyperopt/sampler.rs** - Removed direct `rand_distr` imports in 2 locations
6. **Cargo.toml** - Added missing `rmp-serde.workspace = true` dependency
7. **mixed_precision.rs** - Replaced `num_complex::Complex` with `scirs2_core::Complex`
8. **losses.rs** - Added `Axis` to ndarray import, replaced inline usages

#### Key Patterns Established

**Pattern 1: Distribution Usage**
```rust
// ✅ CORRECT - Distributions return Result type
use scirs2_core::random::*;

let uniform = Uniform::new(-bound, bound)?;  // Handle Result with ?
let samples = Array2::from_shape_fn(shape, |_| uniform.sample(&mut rng));

// Or in test code
let uniform = Uniform::new(-bound, bound).expect("Invalid bounds");
```

**Pattern 2: Complex Number Usage**
```rust
// ✅ CORRECT - Import from scirs2_core root
use scirs2_core::Complex;  // Not scirs2_core::complex::Complex

let c: Complex<f32> = Complex::new(1.0, 2.0);
```

**Pattern 3: ndarray Types Import**
```rust
// ✅ CORRECT - Add all needed types to single import
use scirs2_core::ndarray::{Array1, Array2, ArrayD, Axis, IxDyn, s};

// Then use without qualification
let arr = Array2::zeros((10, 10));
let slice = arr.slice(s![0..5, ..]);
let sum = arr.sum_axis(Axis(0));
```

**Pattern 4: Avoiding Inline Qualified Paths**
```rust
// ❌ WRONG - Inline qualified paths
let arr = ndarray::Array2::zeros((10, 10));
let slice = arr.slice(ndarray::s![0..5, ..]);

// ✅ CORRECT - Proper imports
use scirs2_core::ndarray::{Array2, s};
let arr = Array2::zeros((10, 10));
let slice = arr.slice(s![0..5, ..]);
```

#### Common Errors Encountered and Solutions

**Error 1: "Distribution trait method `.sample()` not found"**
- **Cause**: `Uniform::new()` returns `Result<Uniform<T>, Error>`, not the distribution directly
- **Solution**: Add `?` operator: `Uniform::new(-bound, bound)?`

**Error 2: "cannot find type `Complex` in crate `num_complex`"**
- **Cause**: Direct `num_complex` usage violates policy
- **Solution**: Use `scirs2_core::Complex` (at root level, not in submodule)

**Error 3: "unresolved import `rand_distr`"**
- **Cause**: Test code had direct `use rand_distr::*` imports
- **Solution**: Remove import - types already available via `use scirs2_core::random::*`

**Error 4: "cannot find `Axis` in this scope"**
- **Cause**: Inline `ndarray::Axis(...)` usage without proper import
- **Solution**: Add `Axis` to ndarray import list

#### Verification Results

```bash
# trustformers-models verification
$ cargo check -p trustformers-models
✅ Finished `dev` profile [unoptimized + debuginfo] target(s) in 8.32s
   Warning: 1 unused import (cosmetic only)

# trustformers-training verification
$ cargo check -p trustformers-training
✅ Finished `dev` profile [unoptimized + debuginfo] target(s) in 6.07s
   No warnings
```

Both crates now compile cleanly with 100% SciRS2 policy compliance.

### GPU Operations Policy (Critical)

**Important**: TrustformeRS uses a dual-layer approach for GPU operations. All GPU operations must go through either `trustformers-core` or `scirs2-core`.

#### Dual-Layer GPU Architecture

1. **High-level (trustformers-core)**: Tensor operations with automatic GPU dispatch
   - Device management: `Device::cuda_if_available()`, `Device::metal()`, etc.
   - Automatic memory transfers
   - Backend-agnostic tensor API

2. **Low-level (scirs2-core)**: Scientific computing primitives with GPU acceleration
   - GPU-accelerated BLAS and SIMD
   - Platform capability detection
   - Multiple GPU backend support

#### SciRS2-Core GPU Features

SciRS2-Core provides GPU backends through optional feature flags:
- `gpu` - Base GPU abstractions
- `cuda` - NVIDIA CUDA support (internally uses `cudarc`)
- `metal` - Apple Metal support (internally uses `metal`, `objc2-metal`, `objc2-metal-performance-shaders`)
- `wgpu_backend` - WebGPU support (internally uses `wgpu`, `pollster`)
- `opencl` - OpenCL support (internally uses `opencl3`)

#### Cargo.toml Configuration for GPU

```toml
# ✅ CORRECT - Use scirs2-core features
[dependencies]
scirs2-core = { workspace = true, features = ["gpu", "cuda"] }    # For NVIDIA
scirs2-core = { workspace = true, features = ["gpu", "metal"] }   # For Apple
scirs2-core = { workspace = true, features = ["gpu", "wgpu_backend"] } # For WebGPU
scirs2-core = { workspace = true, features = ["gpu", "opencl"] }  # For OpenCL

# ❌ INCORRECT - Direct GPU dependencies
# cudarc = "0.17"      # FORBIDDEN
# wgpu = "26.0"        # FORBIDDEN
# metal = "0.32"       # FORBIDDEN
# opencl3 = "0.9"      # FORBIDDEN
```

#### GPU Usage Examples

**High-Level Tensor Operations (Recommended)**:
```rust
use trustformers_core::tensor::*;
use trustformers_core::device::Device;

// Automatic GPU selection
let device = Device::cuda_if_available()?;
let tensor = Tensor::randn(&[1024, 768])?.to_device(&device)?;

// Operations automatically run on GPU
let result = tensor.matmul(&weights)?.relu()?;

// Transfer back to CPU if needed
let cpu_result = result.to_device(&Device::cpu())?;
```

**Low-Level GPU Operations (Advanced)**:
```rust
use scirs2_core::gpu_ops::*;  // With 'gpu' feature enabled
use scirs2_core::simd_ops::PlatformCapabilities;

// Platform detection
let caps = PlatformCapabilities::detect();
if caps.cuda_available {
    // Use GPU-accelerated scientific operations
    let result = gpu_accelerated_operation(&data)?;
}
```

**Device Detection**:
```rust
use trustformers_core::device::Device;
use scirs2_core::simd_ops::PlatformCapabilities;

let caps = PlatformCapabilities::detect();
let device = if caps.cuda_available {
    Device::cuda(0)?
} else if caps.rocm_available {
    Device::rocm(0)?
} else if caps.metal_available {
    Device::metal(0)?
} else {
    Device::cpu()
};
```

#### Why This Architecture?

Using scirs2-core for GPU operations provides:
1. **Consistent API** across all GPU backends
2. **Automatic platform detection** and fallback
3. **Centralized version management** (scirs2-core manages cudarc, wgpu, metal versions)
4. **Unified error handling**
5. **Cross-platform compatibility** guarantees

**Remember**: Never add direct GPU library dependencies. Always use scirs2-core's feature flags.

### Real-World Examples

#### ✅ CORRECT: Model Layer Implementation

```rust
use trustformers_core::tensor::Tensor;
use trustformers_core::layers::{Linear, LayerNorm};
use scirs2_core::random::*;

pub struct TransformerBlock {
    self_attn: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerBlock {
    pub fn new(config: &ModelConfig) -> Result<Self> {
        let mut rng = thread_rng();  // ✅ From scirs2_core::random

        Ok(Self {
            self_attn: MultiHeadAttention::new(config)?,
            feed_forward: FeedForward::new(config)?,
            norm1: LayerNorm::new(config.hidden_size)?,
            norm2: LayerNorm::new(config.hidden_size)?,
        })
    }

    pub fn forward(&self, x: Tensor) -> Result<Tensor> {
        // ✅ Uses trustformers_core::tensor operations
        let attn_out = self.self_attn.forward(x.clone())?;
        let x = self.norm1.forward(x.add(&attn_out)?)?;

        let ffn_out = self.feed_forward.forward(x.clone())?;
        let x = self.norm2.forward(x.add(&ffn_out)?)?;

        Ok(x)
    }
}
```

#### ❌ INCORRECT: Direct External Imports

```rust
// DON'T DO THIS!
use rand::thread_rng;  // ❌ Direct rand usage
use ndarray::Array2;   // ❌ Direct ndarray usage
use tch::Tensor;       // ❌ Direct PyTorch tensor usage

pub struct BadImplementation {
    weights: Array2<f32>,  // ❌ Should use trustformers_core::tensor::Tensor
}

impl BadImplementation {
    pub fn new() -> Self {
        let mut rng = thread_rng();  // ❌ Should use scirs2_core::random::thread_rng
        // ...
    }
}
```

#### ✅ CORRECT: Batch Processing with Parallelization

```rust
use trustformers_core::tokenizer::AutoTokenizer;
use scirs2_core::parallel_ops::*;  // ✅ Parallel processing

pub fn batch_tokenize(texts: &[String], model: &str) -> Result<Vec<TokenizedInput>> {
    let tokenizer = AutoTokenizer::from_pretrained(model)?;

    // ✅ Uses scirs2_core parallel processing
    let results: Vec<_> = texts
        .par_iter()  // ✅ from scirs2_core::parallel_ops
        .map(|text| tokenizer.encode(text, true))
        .collect::<Result<Vec<_>>>()?;

    Ok(results)
}
```

#### ✅ CORRECT: Testing with Random Data

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::tensor::Tensor;
    use scirs2_core::random::*;  // ✅ For test data generation

    #[test]
    fn test_forward_pass() {
        let mut rng = thread_rng();  // ✅ From scirs2_core
        let dist = Normal::new(0.0, 1.0).unwrap();  // ✅ From scirs2_core

        // Generate test data
        let input = Tensor::randn(&[2, 4, 512])?;  // ✅ From trustformers_core

        let model = MyModel::new(&config)?;
        let output = model.forward(input)?;

        assert_eq!(output.shape(), &[2, 4, 512]);
    }
}
```

### Workspace Dependency Management

- All dependencies use `workspace = true` in crate-level `Cargo.toml` files
- Versions are managed centrally in the root `Cargo.toml` `[workspace.dependencies]` section
- Always use latest versions available on crates.io
- Exceptions: `keywords` and `categories` are defined per-crate as they differ

Example crate-level `Cargo.toml`:
```toml
[dependencies]
trustformers-core = { workspace = true }
scirs2-core = { workspace = true, features = ["parallel", "simd"] }
anyhow = { workspace = true }
serde = { workspace = true }

# ❌ NEVER do this in non-core crates:
# rand = { workspace = true }  # FORBIDDEN
# ndarray = { workspace = true }  # FORBIDDEN
```

### Code Quality Standards

- **File size limit**: Single code files should be less than 2000 lines. Refactor if exceeded.
- **Naming conventions**: Use `snake_case` for all identifiers (variables, functions, modules)
- **Error handling**: Use `Result<T, TrustformersError>` for error propagation
- **Documentation**: Public APIs must have doc comments with examples
- **Testing**: Use `std::env::temp_dir()` for temporary file handling in tests
- **No warnings**: Code must compile with `cargo clippy -- -D warnings`

## Model Implementation Guidelines

### Adding a New Model

1. Create a new module in `trustformers-models/src/your_model/`
2. Implement the configuration struct (`YourModelConfig`)
3. Implement the base model (`YourModel`) with the `Model` trait
4. Implement task-specific heads (e.g., `YourModelForCausalLM`)
5. Add feature gate in `trustformers-models/Cargo.toml`
6. Export types in `trustformers-models/src/lib.rs`
7. Add tests comparing outputs with Hugging Face reference implementation
8. **IMPORTANT**: Follow the [Critical Development Policies](#critical-development-policies) - use only `trustformers_core` and `scirs2_core` abstractions

### Model Structure Pattern

```rust
// config.rs
use serde::{Deserialize, Serialize};
use trustformers_core::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YourModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    // ... other config fields
}

impl Config for YourModelConfig {
    fn validate(&self) -> Result<()> {
        // Validation logic
        Ok(())
    }
}

// model.rs
use trustformers_core::{
    tensor::Tensor,
    layers::*,
    traits::Model,
    error::TrustformersError,
};
use scirs2_core::random::*;  // For initialization

pub struct YourModel {
    embeddings: Embedding,
    layers: Vec<YourModelLayer>,
    norm: LayerNorm,
}

impl YourModel {
    pub fn new(config: &YourModelConfig) -> Result<Self> {
        let mut rng = thread_rng();  // ✅ From scirs2_core

        let layers = (0..config.num_hidden_layers)
            .map(|_| YourModelLayer::new(config))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            embeddings: Embedding::new(config.vocab_size, config.hidden_size)?,
            layers,
            norm: LayerNorm::new(config.hidden_size)?,
        })
    }
}

impl Model for YourModel {
    type Config = YourModelConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let mut hidden = self.embeddings.forward(input)?;

        for layer in &self.layers {
            hidden = layer.forward(hidden)?;
        }

        hidden = self.norm.forward(hidden)?;
        Ok(hidden)
    }
}

// tasks.rs
pub struct YourModelForCausalLM {
    model: YourModel,
    lm_head: Linear,
}

impl YourModelForCausalLM {
    pub fn new(config: &YourModelConfig) -> Result<Self> {
        Ok(Self {
            model: YourModel::new(config)?,
            lm_head: Linear::new(config.hidden_size, config.vocab_size, true)?,
        })
    }

    pub fn forward(&self, input_ids: Tensor) -> Result<Tensor> {
        let hidden = self.model.forward(input_ids)?;
        let logits = self.lm_head.forward(hidden)?;
        Ok(logits)
    }
}
```

## Core Traits

All models implement these traits from `trustformers-core::traits`:

- **Model**: Base model trait with `forward()` method
- **Layer**: Building block trait for composable layers
- **Config**: Configuration trait for model hyperparameters
- **Tokenizer**: Tokenization interface

## Tensor Operations

### Tensor Creation

```rust
use trustformers_core::tensor::Tensor;

// Create tensors
let zeros = Tensor::zeros(&[2, 3, 4])?;
let ones = Tensor::ones(&[2, 3, 4])?;
let randn = Tensor::randn(&[2, 3, 4])?;  // Normal(0, 1)
let uniform = Tensor::rand(&[2, 3, 4])?;  // Uniform(0, 1)

// From data
let data = vec![1.0, 2.0, 3.0, 4.0];
let tensor = Tensor::from_slice(&data, &[2, 2])?;
```

### Tensor Operations

```rust
// Math operations
let result = a.add(&b)?;
let result = a.sub(&b)?;
let result = a.mul(&b)?;
let result = a.div(&b)?;
let result = a.matmul(&b)?;

// Activations
let result = tensor.relu()?;
let result = tensor.gelu()?;
let result = tensor.silu()?;
let result = tensor.softmax(-1)?;

// Shape operations
let result = tensor.reshape(&[new_shape])?;
let result = tensor.transpose(0, 1)?;
let result = tensor.permute(&[2, 0, 1])?;

// Reduction operations
let sum = tensor.sum(None)?;
let mean = tensor.mean(None)?;
let max = tensor.max(None)?;
```

### Device Management

```rust
use trustformers_core::device::Device;

// Get best available device
let device = Device::cuda_if_available()?;

// Or specify explicitly
let device = Device::cuda(0)?;  // CUDA device 0
let device = Device::metal(0)?;  // Metal device 0
let device = Device::cpu();     // CPU

// Move tensor to device
let tensor = tensor.to_device(&device)?;
```

## Testing Conventions

- Integration tests in `tests/` directories use `#[cfg(test)]` and `#[test]`
- Property-based tests use `proptest` for generative testing
- Benchmark tests use `criterion` in `benches/` directories
- Tests should use `rstest` for parameterized tests when applicable
- Compare numerical outputs with Hugging Face for model correctness
- **IMPORTANT**: Tests must follow the [Critical Development Policies](#critical-development-policies) - use `scirs2_core::random` for RNG, not `rand` directly

### Test Template

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::tensor::Tensor;
    use scirs2_core::random::*;  // ✅ Required for tests

    #[test]
    fn test_model_forward() -> Result<()> {
        let mut rng = thread_rng();  // ✅ From scirs2_core

        let config = ModelConfig::default();
        let model = Model::new(&config)?;

        let input = Tensor::randn(&[1, 10, 512])?;
        let output = model.forward(input)?;

        assert_eq!(output.shape(), &[1, 10, 512]);
        Ok(())
    }
}
```

## Performance Optimization

### SIMD Operations

```rust
use scirs2_core::simd_ops::SimdUnifiedOps;

// For performance-critical array operations
let result = f32::simd_add(&a.view(), &b.view());
let dot = f64::simd_dot(&x.view(), &y.view());
```

### Parallel Processing

```rust
use scirs2_core::parallel_ops::*;

// Parallel iteration
let results: Vec<_> = data
    .par_iter()  // ✅ From scirs2_core
    .map(|x| expensive_computation(x))
    .collect();

// Conditional parallelism
if is_parallel_enabled() && data.len() > 1000 {
    // Use parallel
} else {
    // Use sequential
}
```

### GPU Acceleration

```rust
use trustformers_core::device::Device;
use trustformers_core::tensor::Tensor;

// Detect best device
let device = Device::cuda_if_available()?;

// Move model and data to GPU
let model = model.to_device(&device)?;
let input = input.to_device(&device)?;

// Forward pass on GPU
let output = model.forward(input)?;
```

### Quantization

```rust
use trustformers_core::quantization::*;

// Load quantized model
let model = AutoModel::from_pretrained_quantized(
    "model-name",
    QuantizationConfig {
        method: QuantizationMethod::GGML_Q4_0,
        ..Default::default()
    }
)?;

// Or quantize existing model
let quantizer = QuantizationEngine::new(config)?;
let quantized = quantizer.quantize_model(&model)?;
```

## Feature Flags

Models are feature-gated to reduce compilation time:
```toml
[features]
default = ["bert", "gpt2"]
bert = []
gpt2 = []
llama = []
mistral = []
multimodal = ["clip", "vit"]
# ... etc
```

Enable specific models when developing:
```bash
cargo build --features "bert,gpt2,llama"
cargo test --features "bert" -p trustformers-models
```

## Pipeline API

High-level pipeline API for common tasks:
```rust
use trustformers::pipeline;

// Text classification
let classifier = pipeline("sentiment-analysis")?;
let result = classifier("I love Rust!")?;

// Text generation
let generator = pipeline("text-generation")?;
let result = generator("Once upon a time")?;

// Question answering
let qa = pipeline("question-answering")?;
let result = qa(&context, &question)?;
```

Available pipelines: text-generation, token-classification, question-answering, fill-mask, summarization, translation, text-classification

## Common Patterns

### Loading Models from Hub

```rust
use trustformers::{AutoModel, AutoTokenizer};

let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;
let model = AutoModel::from_pretrained("bert-base-uncased")?;

// Tokenize
let encoding = tokenizer.encode("Hello world", true)?;

// Forward pass
let input_ids = Tensor::from_slice(&encoding.input_ids)?;
let output = model.forward(input_ids)?;
```

### Custom Model Training

```rust
use trustformers_core::optim::*;
use trustformers_training::Trainer;

let optimizer = AdamW::new(model.parameters(), 1e-4)?;
let trainer = Trainer::new(
    model,
    optimizer,
    train_dataset,
    val_dataset,
)?;

trainer.train()?;
```

## SciRS2-Core Integration Examples

### Random Number Generation

```rust
use scirs2_core::random::*;

// Basic RNG
let mut rng = thread_rng();

// Distributions
let normal = Normal::new(0.0, 1.0)?;
let uniform = Uniform::new(0.0, 1.0)?;
let beta = Beta::new(2.0, 5.0)?;

// Sampling
let samples: Vec<f64> = (0..100)
    .map(|_| normal.sample(&mut rng))
    .collect();

// Shuffling
let mut data = vec![1, 2, 3, 4, 5];
data.shuffle(&mut rng);
```

### Array Operations (when needed)

```rust
use scirs2_core::ndarray::*;

// Create arrays
let arr = array![[1, 2], [3, 4]];
let zeros = Array2::<f32>::zeros((10, 10));

// Slicing
let slice = arr.slice(s![.., 0]);

// Operations
let sum = arr.sum();
let mean = arr.mean().unwrap();
```

## Related Projects

TrustformeRS is part of the cool-japan ecosystem. Related projects are referenced in `~/.claude/CLAUDE.md`:
- **SciRS2** (~/work/scirs/) - Scientific computing in Rust (used for SIMD, parallel ops)
- **ToRSh** (~/work/torsh/) - PyTorch-like framework in Rust
- **NumRS2** (~/work/numrs/) - NumPy-like library in Rust
- **SkleaRS** (~/work/sklears/) - Scikit-learn in Rust
- **HuggingFace Transformers** (~/work/transformers/) - Reference Python implementation

## Additional Documentation

- `README.md` - Project overview and quick start
- `SCIRS2_INTEGRATION_POLICY.md` - **MANDATORY READ**: Dependency abstraction policy
- `CONTRIBUTING.md` - Contribution guidelines
- `TODO.md` - Development roadmap and TODOs

## Troubleshooting Common Issues

### Compilation Error: "cannot find type X"

**Cause**: Direct import of external dependency in non-core crate

**Solution**: Use trustformers-core or scirs2-core abstraction instead
```rust
// ❌ Wrong
use rand::Rng;

// ✅ Correct
use scirs2_core::random::*;
```

### Compilation Error: "trait X is not implemented"

**Cause**: Using wrong tensor type (e.g., ndarray::Array instead of trustformers_core::tensor::Tensor)

**Solution**: Use unified tensor type
```rust
// ❌ Wrong
use ndarray::Array2;
let tensor: Array2<f32> = ...;

// ✅ Correct
use trustformers_core::tensor::Tensor;
let tensor = Tensor::zeros(&[10, 10])?;
```

### Test Failure: "distribution not found"

**Cause**: Not importing scirs2_core::random in tests

**Solution**: Add proper imports to test module
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::*;  // ✅ Add this
}
```

### Compilation Error: "cannot find type `cudarc` / `wgpu` / `metal`"

**Cause**: Direct GPU library usage violates SciRS2 policy

**Solution**: Use scirs2-core with appropriate GPU features
```rust
// ❌ Wrong - Direct GPU dependency
// Cargo.toml:
// cudarc = "0.17"
use cudarc::*;

// ✅ Correct - Use scirs2-core features
// Cargo.toml:
// scirs2-core = { workspace = true, features = ["gpu", "cuda"] }
use scirs2_core::gpu_ops::*;

// Or use high-level tensor API
use trustformers_core::tensor::Tensor;
use trustformers_core::device::Device;
let device = Device::cuda_if_available()?;
```

### Runtime Error: "GPU not available"

**Cause**: GPU features not enabled or GPU not detected

**Solution**:
1. Enable appropriate feature in Cargo.toml:
```toml
scirs2-core = { workspace = true, features = ["gpu", "cuda"] }  # For NVIDIA
# or
scirs2-core = { workspace = true, features = ["gpu", "metal"] }  # For Apple
```

2. Check platform capabilities:
```rust
use scirs2_core::simd_ops::PlatformCapabilities;

let caps = PlatformCapabilities::detect();
if !caps.cuda_available {
    eprintln!("CUDA not available, falling back to CPU");
}
```

3. Use safe device selection:
```rust
let device = Device::cuda_if_available()
    .unwrap_or_else(|_| Device::cpu());
```

## Quick Reference Card

### Imports Cheat Sheet

```rust
// ✅ Tensor operations
use trustformers_core::tensor::{Tensor, Device};

// ✅ Model components
use trustformers_core::layers::{Linear, LayerNorm, Dropout, MultiHeadAttention};
use trustformers_core::models::{AutoModel, Model};
use trustformers_core::tokenizer::AutoTokenizer;

// ✅ Scientific computing
use scirs2_core::random::*;        // RNG and distributions
use scirs2_core::parallel_ops::*;  // Parallel processing
use scirs2_core::simd_ops::*;      // SIMD operations
use scirs2_core::ndarray::*;       // Arrays (Array1, Array2, ArrayD, s!, array!)
use scirs2_core::Complex;          // Complex numbers

// ✅ GPU operations (with appropriate features enabled)
use scirs2_core::gpu_ops::*;                         // Low-level GPU ops
use scirs2_core::simd_ops::PlatformCapabilities;    // Platform/GPU detection

// ✅ Error handling
use trustformers_core::error::{TrustformersError, Result};

// ❌ NEVER import these directly in non-core crates:
// use rand::*;
// use ndarray::*;
// use tokenizers::*;
// use tch::*;
// use candle_core::*;
// use rayon::*;
// use cudarc::*;      // Use scirs2_core with 'cuda' feature
// use wgpu::*;        // Use scirs2_core with 'wgpu_backend' feature
// use metal::*;       // Use scirs2_core with 'metal' feature
// use opencl3::*;     // Use scirs2_core with 'opencl' feature
```

## Minimum Rust Version

MSRV: 1.75 (specified in root `Cargo.toml`)

## Getting Help

1. Check `SCIRS2_INTEGRATION_POLICY.md` for dependency issues
2. Review existing model implementations for patterns
3. Check the test suite for usage examples
4. Open an issue on GitHub for bugs or questions

---

**Remember**: When in doubt, use the core abstractions! Follow the dual-layer architecture strictly to maintain consistency and avoid compilation issues.
