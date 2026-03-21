# trustformers-core

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Status](https://img.shields.io/badge/status-Stable-brightgreen)
![Tests](https://img.shields.io/badge/tests-1%2C140%20passing-brightgreen)
![SLoC](https://img.shields.io/badge/SLoC-121%2C799-informational)
![Date](https://img.shields.io/badge/updated-2026--03--21-lightgrey)

Core infrastructure crate providing fundamental abstractions and utilities for the TrustformeRS ecosystem.

## Current State

**Version 0.1.0 — Stable (2026-03-21)**

This crate is **stable and production-ready**, serving as the foundation for all other TrustformeRS components. It provides high-performance tensor operations, layer implementations, and advanced optimization techniques. All 1,140 tests pass with zero stubs or unimplemented items.

## Features

### Tensor Operations
- **Comprehensive tensor abstraction** supporting multiple backends
- **SciRS2 integration** for SIMD-optimized operations
- **GPU support** through multiple backends (CUDA, Metal, Vulkan, WebGPU, OpenCL, ROCm, Vulkan, TPU, OneAPI, XLA, RISC-V)
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
- **GPU Attention**: Scaled dot-product and flash attention (tiled online-softmax)

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
│   ├── tensor/           # Tensor abstractions and operations
│   ├── layers/           # Neural network layers
│   ├── attention/        # Attention mechanisms
│   ├── optimization/     # Performance optimizations
│   ├── quantization/     # Quantization infrastructure
│   ├── export/           # Model export formats
│   ├── evaluation/       # Benchmark implementations
│   ├── monitoring/       # Profiling and analysis
│   ├── parallel/         # Distributed computing
│   └── peft/            # Parameter-efficient fine-tuning
```

## Usage Example

```rust
use trustformers_core::{
    tensor::Tensor,
    layers::{Linear, Layer},
    attention::FlashAttention,
};

// Create tensors
let input = Tensor::randn(&[32, 512, 768])?;

// Create layers
let linear = Linear::new(768, 768, true)?;
let attention = FlashAttention::new(768, 12)?;

// Forward pass
let output = linear.forward(&input)?;
let attended = attention.forward(&output, None)?;
```

## Performance

- **FlashAttention**: O(N) memory complexity vs O(N²) standard
- **Quantization**: 50-75% memory reduction with INT8/INT4
- **SIMD**: 2-3x speedup on supported operations
- **PagedAttention**: Eliminates KV cache fragmentation

## Testing

The crate includes comprehensive test coverage:
- **1,140 unit and integration tests, all passing**
- Property-based testing with proptest
- Memory leak detection
- Performance benchmarks
- Cross-backend compatibility tests
- Numerical stability tests with adaptive tolerance

## Dependencies

- `scirs2-core`: SIMD operations and parallelism
- `half`: FP16/BF16 support
- `rayon`: Parallel iteration (via SciRS2)
- Various serialization and utility crates

## Public API

The crate exposes **1,596 public API items** covering tensors, layers, attention, quantization, export, evaluation, monitoring, distributed computing, PEFT, kernel tuning, and memory management.

## License

Apache-2.0