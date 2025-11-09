# trustformers-core

Core infrastructure crate providing fundamental abstractions and utilities for the TrustformeRS ecosystem.

## Current State

This crate is **mature and comprehensive**, serving as the foundation for all other TrustformeRS components. It provides high-performance tensor operations, layer implementations, and advanced optimization techniques.

## Features

### Tensor Operations
- **Comprehensive tensor abstraction** supporting multiple backends
- **SciRS2 integration** for SIMD-optimized operations
- **GPU support** through multiple backends (CUDA, Metal, Vulkan, WebGPU)
- **Automatic differentiation** framework (in progress)
- **Memory-efficient operations** with zero-copy views

### Layer Implementations
- **Core Layers**: Linear, Embedding, LayerNorm, Dropout
- **Attention Mechanisms**: 
  - Multi-head attention with causal masking
  - FlashAttention & FlashAttention-2 for memory efficiency
  - Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
  - PagedAttention for KV cache management
  - Optimized SDPA kernels with adaptive strategies
- **Advanced Layers**: FeedForward, PositionalEncoding, RMSNorm

### Performance Optimizations
- **SIMD Operations**: Optimized LayerNorm, Softmax, and RoPE implementations
- **Quantization Support**: INT8, INT4, GPTQ, AWQ with calibration
- **Custom Kernels**: Fused operations for reduced memory bandwidth
- **Memory Management**: Efficient allocation strategies and pooling

### Export and Interoperability
- **ONNX Export**: Complete graph construction and runtime support
- **GGML/GGUF**: Advanced quantization formats for edge deployment
- **CoreML**: iOS deployment support
- **TensorRT**: NVIDIA GPU optimization (framework ready)

### Advanced Features
- **Evaluation Framework**: GLUE, SuperGLUE, MMLU, HellaSwag, HumanEval benchmarks
- **Monitoring**: TensorBoard integration, gradient flow analysis, activation statistics
- **Caching System**: Multiple eviction policies (LRU, LFU, ARC)
- **A/B Testing**: Infrastructure for model comparison
- **Model Compression**: Pruning and distillation support

### Distributed and Parallel Computing
- **Model Parallelism**: Tensor and pipeline parallelism support
- **Data Parallelism**: Multi-GPU training infrastructure
- **Communication Backends**: NCCL, MPI, Gloo support
- **Process Groups**: All-reduce, broadcast, and collective operations

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
- Unit tests for all operations
- Integration tests for complex scenarios
- Property-based testing with proptest
- Memory leak detection
- Performance benchmarks

## Dependencies

- `scirs2-core`: SIMD operations and parallelism
- `ndarray`: Tensor backend (being migrated to SciRS2)
- `half`: FP16/BF16 support
- `rayon`: Parallel iteration (via SciRS2)
- Various serialization and utility crates

## License

MIT OR Apache-2.0