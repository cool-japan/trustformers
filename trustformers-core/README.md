# trustformers-core

![Version](https://img.shields.io/badge/version-0.2.1-blue)
![Status](https://img.shields.io/badge/status-Stable-brightgreen)
![Tests](https://img.shields.io/badge/tests-2%2C353%2B%20passing-brightgreen)
![SLoC](https://img.shields.io/badge/SLoC-153%2C689-informational)
![Date](https://img.shields.io/badge/updated-2026--07--09-lightgrey)

Core infrastructure crate providing fundamental abstractions and utilities for the TrustformeRS ecosystem.

## Current State

**Version 0.2.1 — Stable (2026-07-09)**

This crate is **stable and production-ready**, serving as the foundation for all other TrustformeRS components. It provides high-performance tensor operations, layer implementations, and advanced optimization techniques. ~2,353 tests pass for this crate specifically, with zero stubs or unimplemented items, and zero clippy/rustdoc warnings workspace-wide.

## Features

### Tensor Operations
- **Comprehensive tensor abstraction** supporting multiple backends
- **SciRS2 integration** for SIMD-optimized operations
- **GPU support** through multiple backends (CUDA, Metal, Vulkan, WebGPU, OpenCL, ROCm, OneAPI, XLA, RISC-V), each opt-in behind its own Cargo feature (`cuda`, `metal`, `vulkan`, `wgpu_backend`, `opencl`, `rocm`, `oneapi`, `xla`, `riscv`) — no GPU backend is enabled by default (`default = ["linalg"]`)
- **Automatic differentiation** with reverse-mode and forward-mode autodiff
- **Memory-efficient operations** with zero-copy views

### Layer Implementations
- **Core Layers**: Linear, Embedding, LayerNorm, Dropout, RMSNorm
- **Attention Mechanisms**:
  - Multi-head attention (MHA) with causal masking
  - FlashAttention and FlashAttention-2 for memory efficiency
  - Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
  - PagedAttention for KV cache management
  - Optimized SDPA kernels with adaptive strategies
- **Advanced Layers**: FeedForward (SwiGLU, GeGLU), PositionalEncoding, RoPE, ALiBi

### Performance Optimizations
- **SIMD Operations**: Optimized LayerNorm, Softmax, and RoPE implementations
- **Quantization Support**: INT8, INT4, FP16, FP8, GPTQ, AWQ, GGUF/K-quants with calibration
- **Custom Kernels**: Fused operations for reduced memory bandwidth
- **Kernel Tuning**: Automatic hardware-aware kernel parameter optimization
- **Memory Management**: Adaptive pooling with LRU, LFU, ARC, and Hybrid eviction policies
- **Conv2D**: Full im2col+matmul implementation with groups and dilation
- **GPU Attention**: Scaled dot-product and flash attention (tiled online-softmax); the CUDA backend (`cuda` feature) additionally runs a fully GPU-resident attention pipeline — QKV split, RoPE, causal softmax, KV-cache append — and batched/broadcasting matmul, avoiding host round-trips during prefill and decode

### Export and Interoperability
- **ONNX Export**: Complete graph construction and runtime support
- **GGML/GGUF**: Advanced quantization formats including K-quants for edge deployment
- **CoreML**: iOS/macOS deployment support

### Advanced Features
- **Evaluation Framework**: GLUE, SuperGLUE, MMLU, HellaSwag, HumanEval benchmarks
- **Monitoring**: TensorBoard integration, gradient flow analysis, activation statistics
- **Caching System**: Multiple eviction policies (LRU, LFU, ARC, Hybrid)
- **A/B Testing**: Infrastructure for model comparison
- **Model Compression**: Pruning and distillation support
- **Plugin System**: Extensible architecture for custom kernels and layers
- **Tensor Debugger**: Interactive watchpoints, NaN/Inf detection, operation tracing

### Distributed and Parallel Computing
- **Tensor Parallelism**: Column/row parallel linear layers
- **Pipeline Parallelism**: Stage-based model partitioning
- **Data Parallelism**: Multi-GPU training infrastructure
- **Communication Backends**: NCCL, MPI, Gloo support
- **Process Groups**: All-reduce, broadcast, all-gather, reduce-scatter operations

### PEFT (Parameter-Efficient Fine-Tuning)
- **LoRA**: Low-rank adaptation with weight merging
- **QLoRA**: Quantized LoRA for memory efficiency
- **Adapters**: Bottleneck adapter layers
- **Prefix Tuning**: Trainable prefix embeddings
- **Prompt Tuning**: Virtual token optimization

## Architecture

```
trustformers-core/
├── src/
│   ├── tensor/             # Tensor abstractions and operations
│   ├── layers/             # Neural network layers
│   ├── attention/          # Attention mechanisms
│   ├── quantization/       # Quantization infrastructure (largest module by API surface)
│   ├── export/             # Model export formats (ONNX, GGUF, CoreML)
│   ├── kernels/            # Custom fused and SIMD compute kernels
│   ├── hardware/           # Hardware acceleration abstraction (CUDA, Metal, Vulkan, ROCm, ...)
│   ├── compiler/           # JIT compilation, kernel fusion, graph optimization
│   ├── performance/        # Benchmarking, profiling, and optimization advisor
│   ├── monitoring/         # Profiling and analysis
│   ├── parallel/           # Distributed computing
│   ├── evaluation/         # Benchmark implementations
│   └── peft.rs             # Parameter-efficient fine-tuning
```

## Usage Example

```rust
use trustformers_core::layers::{Linear, LayerNorm, MultiHeadAttention};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Layer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create layers for a transformer block
    let attention = MultiHeadAttention::new(768, 12, 0.1, true)?;
    let norm1 = LayerNorm::new(vec![768], 1e-5)?;
    let ffn = Linear::new(768, 3072, true);
    let norm2 = LayerNorm::new(vec![768], 1e-5)?;

    // Run the attention sub-layer's forward pass
    let input = Tensor::randn(&[2, 128, 768])?;
    let attended = attention.forward(input)?;

    Ok(())
}
```

## Performance

- **FlashAttention**: O(N) memory complexity vs O(N²) standard
- **Quantization**: 50-75% memory reduction with INT8/INT4
- **SIMD**: 2-3x speedup on supported operations
- **PagedAttention**: Eliminates KV cache fragmentation

## Testing

The crate includes comprehensive test coverage:
- **~2,353 unit and integration tests, all passing** (this crate's approximate share of a workspace-wide run completed 2026-07-01: 18,102 passed / 0 failed / 119 skipped via `cargo nextest run --workspace --all-features`, plus 0 clippy warnings and 0 rustdoc warnings)
- Property-based testing with proptest
- Memory leak detection
- Performance benchmarks
- Cross-backend compatibility tests
- Numerical stability tests with adaptive tolerance

## Dependencies

- `scirs2-core`: SIMD operations and parallelism
- `half`: FP16/BF16 support
- `rayon`: Parallel iteration (via SciRS2)
- `oxicuda-{blas,dnn,memory,driver}` (optional, `cuda` feature): CUDA GEMM, resident attention, and refcounted GPU buffer management
- Various serialization and utility crates

## Public API

The crate exposes **~4,533 public API items** (structs, enums, traits, and fns across the public API, including impl blocks — broader than a prior doc revision's narrower top-level-only count) covering tensors, layers, attention, quantization, export, evaluation, monitoring, distributed computing, PEFT, kernel tuning, and memory management.

## License

Apache-2.0