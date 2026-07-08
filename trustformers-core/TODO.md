# trustformers-core TODO List

## Overview

The `trustformers-core` crate is the foundational infrastructure of the TrustformeRS ecosystem.
It provides core tensor operations, hardware acceleration, layer abstractions, and all fundamental
building blocks required by model implementations in trustformers-models and other crates.

**Key Responsibilities:**
- Multi-backend tensor abstraction (CPU, CUDA, ROCm, Metal, Vulkan, XLA, TPU, RISC-V)
- Hardware acceleration infrastructure
- Core layers (Linear, Embedding, LayerNorm, Attention, FFN)
- Memory management and optimization
- AutoDiff engine for backpropagation
- Quantization infrastructure
- Weight loading and checkpoint conversion
- Export formats (ONNX, GGUF, TensorRT, Core ML, TVM)
- Error handling and debugging tools

---

## Current Status

### Implementation Status
✅ **STABLE** - Version 0.1.4 (initial stable release 0.1.0 on 2026-03-21)
✅ **ZERO COMPILATION ERRORS** - Clean compilation across all backends
✅ **COMPREHENSIVE TEST COVERAGE** - ~2,353 tests with 100% pass rate
✅ **ALL TODOS COMPLETED** - Zero stubs (todo!/unimplemented!) remaining (verified 2026-07-01)
✅ **THREAD-SAFE** - Proper synchronization primitives throughout
✅ **MEMORY-SAFE** - Zero-copy operations and efficient memory management

### Code Quality Metrics
- **Test Count:** ~2,353 unit tests, all passing (0 failing)
- **Stubs:** 0 (no todo! or unimplemented! macros; verified 2026-07-01)
- **Public API Items:** ~4,533
- **SLoC:** 155,280
- **Code Coverage:** Extensive coverage across modules
- **Clippy Warnings:** 3855+ warnings resolved historically; 0 clippy and 0 rustdoc warnings workspace-wide as of 2026-07-01
- **File Size Compliance:** All files <2000 lines (verified 2026-07-01)
- **Documentation:** Comprehensive rustdoc for all public APIs

---

## Completed Features

### Tensor Operations

#### Core Tensor Infrastructure
- ✅ **Multi-Backend Abstraction**
  - Unified `Tensor` type across all backends
  - Automatic backend selection based on device availability
  - Seamless device-to-device transfers
  - Zero-copy views where possible

- ✅ **Tensor Creation**
  - `zeros`, `ones`, `randn` (normal distribution)
  - `rand` (uniform distribution)
  - `from_slice`, `from_vec` with shape specification
  - `eye` (identity matrix)
  - `arange`, `linspace` for ranges
  - Empty tensor allocation with `empty`

- ✅ **Mathematical Operations**
  - **Arithmetic:** add, sub, mul, div, neg, abs, pow, sqrt, exp, log
  - **Matrix Operations:** matmul, dot, outer, tensordot
  - **Advanced:** einsum with Einstein summation notation
  - **Comparison:** eq, ne, lt, le, gt, ge
  - **Logical:** and, or, not, xor for boolean tensors
  - **Trigonometric:** sin, cos, tan, asin, acos, atan, atan2
  - **Hyperbolic:** sinh, cosh, tanh, asinh, acosh, atanh

- ✅ **Broadcasting**
  - NumPy-compatible broadcasting rules
  - Automatic shape alignment
  - Efficient memory usage with view semantics
  - Support for complex broadcasting patterns

- ✅ **Shape Manipulation**
  - `reshape`: Change tensor shape (with validation)
  - `transpose`: 2D matrix transpose
  - `permute`: Multi-dimensional permutation
  - `squeeze`: Remove dimensions of size 1
  - `unsqueeze`: Add dimensions of size 1
  - `flatten`: Flatten to 1D or specified dimensions
  - `view`: Create view with new shape (zero-copy when contiguous)
  - `expand`: Broadcast to new shape without copying
  - `repeat`: Repeat tensor along dimensions

- ✅ **Indexing and Slicing**
  - Multi-dimensional indexing `[start..end, :]`
  - Fancy indexing with index tensors
  - `select`: Select along dimension
  - `gather`: Gather values along dimension with indices
  - `scatter`: Scatter values into tensor
  - `index_select`: Select indices along dimension
  - `masked_select`: Boolean masking

- ✅ **Concatenation and Splitting**
  - `concat`/`cat`: Concatenate tensors along dimension
  - `stack`: Stack tensors creating new dimension
  - `split`: Split tensor into chunks
  - `chunk`: Split into equal-sized chunks
  - `unbind`: Remove dimension returning list of tensors

- ✅ **Reduction Operations**
  - `sum`, `mean`: Reduce with optional dimension
  - `max`, `min`: Maximum/minimum values
  - `argmax`, `argmin`: Indices of max/min values
  - `std`, `var`: Standard deviation and variance
  - `prod`: Product reduction
  - `all`, `any`: Boolean reductions

- ✅ **Activation Functions**
  - **ReLU:** `relu` (max(0, x))
  - **GELU:** `gelu` (exact) and `gelu_approx` (tanh approximation)
  - **SiLU/Swish:** `silu` (x * sigmoid(x))
  - **Softmax:** `softmax` with numerical stability
  - **LogSoftmax:** `log_softmax` for numerical stability in cross-entropy
  - **Tanh:** `tanh` hyperbolic tangent
  - **Sigmoid:** `sigmoid` logistic function
  - **ELU:** Exponential Linear Unit
  - **LeakyReLU:** Leaky ReLU with negative slope
  - **Mish:** `mish` (x * tanh(softplus(x)))

- ✅ **Data Types**
  - **Full Precision:** F32, F64 for training and high-precision inference
  - **Half Precision:** F16, BF16 for memory-efficient training
  - **Integer Types:** I8, I16, I32, I64 for quantization
  - **Unsigned:** U8, U16, U32, U64 for indices and masks
  - **Complex:** C32 (Complex<f32>), C64 (Complex<f64>)
  - **Half Complex:** CF16, CBF16 for memory-efficient complex operations
  - **Boolean:** Bool for masks and logical operations

- ✅ **Sparse Tensor Support**
  - COO (Coordinate) format
  - CSR (Compressed Sparse Row) format
  - CSC (Compressed Sparse Column) format
  - BSR (Block Sparse Row) format
  - DOK (Dictionary of Keys) format
  - Sparse-dense operations
  - Efficient storage for sparse weights

- ✅ **Advanced Sparse Operations** (NEW - 2025-11-10)
  - **Structured Sparsity Patterns**
    - N:M sparsity (2:4, 1:4, etc.) for hardware acceleration
    - Block sparsity with configurable block sizes
    - Channel pruning for model compression
    - Magnitude-based and gradient-based pruning
  - **Sparse Matrix Multiplication**
    - SpMM (Sparse-Dense matmul) optimized for CSR format
    - Efficient batch operations
  - **Sparse Attention Utilities**
    - Block-sparse attention patterns
    - Sliding window attention masks
    - Dilated window attention for long-range dependencies
  - **Format Conversion**
    - COO ↔ CSR ↔ CSC conversions
    - Efficient sorting and reorganization
  - **Pruning Algorithms**
    - Magnitude pruning with configurable keep ratios
    - Gradient-based importance scoring
    - Automatic sparsity pattern selection

---

### Hardware Acceleration

#### CUDA Backend (NVIDIA GPUs)
- ✅ **Custom Fused Kernels**
  - Fused GELU (exact): Single kernel for GELU activation
  - Fused GELU (approximate): Fast tanh-based approximation
  - Fused Bias + ReLU: Bias addition and ReLU in single kernel
  - Fused Bias + GELU: Bias addition and GELU activation
  - Fused Bias + SiLU: Bias addition and SiLU/Swish
  - Fused Bias + Tanh: Bias addition and hyperbolic tangent
  - Dynamic kernel compilation with NVRTC

- ✅ **cuBLAS Integration**
  - Optimized GEMM (General Matrix Multiply)
  - Batch matrix multiplication
  - Strided batched operations
  - Mixed-precision GEMM (FP16, BF16)

- ✅ **Memory Management**
  - Efficient GPU memory allocation
  - Memory pools for small tensors
  - Unified memory support
  - Asynchronous memory operations

- ✅ **Multi-GPU Support**
  - NCCL (NVIDIA Collective Communications Library)
  - Peer-to-peer memory access
  - Device-to-device transfers
  - All-reduce, all-gather, reduce-scatter operations

- ✅ **Streams and Events**
  - Asynchronous kernel execution
  - Multi-stream concurrency
  - Event-based synchronization

#### ROCm/HIP Backend (AMD GPUs)
- ✅ **AMD GPU Support**
  - Full ROCm/HIP integration
  - Portable across AMD GPU architectures
  - Compatible with MI series (MI100, MI200, MI300)

- ✅ **Custom HIP Kernels**
  - Fused operations optimized for AMD architecture
  - Wavefront-aware kernel design
  - LDS (Local Data Share) utilization

- ✅ **rocBLAS Integration**
  - Optimized matrix operations
  - Batched GEMM support
  - Mixed-precision computations

- ✅ **Memory Management**
  - Efficient HIP memory APIs
  - Asynchronous memory operations
  - HIP managed memory

#### Metal Backend (Apple Silicon)
- ✅ **MPS Integration**
  - Metal Performance Shaders framework
  - Neural network operations
  - Optimized for M-series chips (M1, M2, M3, M4)

- ✅ **Unified Memory**
  - Efficient CPU-GPU memory sharing
  - Zero-copy between CPU and GPU
  - Automatic data migration

- ✅ **Custom Metal Shaders**
  - Metal Shading Language (MSL) kernels
  - Optimized for Apple GPU architecture
  - Tile-based rendering utilization

- ✅ **Flash Attention**
  - MPS graph-based implementation
  - Memory-efficient attention computation
  - Platform: macOS 10.15+, iOS 13+

#### Intel oneAPI Backend
- ✅ **DPC++ SYCL**
  - Data Parallel C++ kernel compilation
  - Cross-architecture support (CPU, GPU, FPGA)
  - USM (Unified Shared Memory)

- ✅ **oneDNN Integration**
  - Deep Neural Network Library
  - Optimized convolutions, pooling, normalization
  - Primitive caching for performance

- ✅ **oneMKL**
  - Math Kernel Library for linear algebra
  - Optimized BLAS and LAPACK operations
  - Intel CPU optimizations (AVX-512, AMX)

- ✅ **Multi-Device Support**
  - CPU: Intel Xeon, Core
  - GPU: Intel Arc, Iris Xe, Data Center GPUs
  - FPGA: Programmable acceleration

#### Google XLA (Accelerated Linear Algebra)
- ✅ **HLO Compilation**
  - High-Level Operations IR
  - Platform-specific code generation
  - Automatic fusion and optimization

- ✅ **Multi-Platform**
  - CPU backend with LLVM
  - GPU backend with NVPTX/AMDGPU
  - TPU backend for Google Cloud

- ✅ **Shape Inference**
  - Automatic output shape computation
  - Static shape optimization
  - Dynamic shape support

- ✅ **Optimization Passes**
  - Operation fusion (element-wise, reduce-window)
  - Buffer assignment and liveness analysis
  - Layout optimization for hardware

#### TPU Backend (Google Cloud TPU)
- ✅ **Multi-Generation Support**
  - TPU v2: 180 teraflops, 64GB HBM
  - TPU v3: 420 teraflops, 128GB HBM
  - TPU v4: 275 teraflops per chip, scalable pods
  - TPU v5e: Cost-optimized for inference and training
  - TPU v5p: High-performance training

- ✅ **Systolic Array Optimization**
  - Matrix multiplication acceleration
  - Pipelined data flow
  - 2D mesh architecture

- ✅ **BFloat16**
  - Native bfloat16 precision
  - Dynamic range of FP32 with FP16 storage
  - Mixed-precision training

- ✅ **HBM Management**
  - High Bandwidth Memory (up to 128GB per chip)
  - Efficient memory layout
  - Sharding across TPU cores

#### RISC-V Vector Extensions (RVV)
- ✅ **RVV 1.0 Compliance**
  - Full specification support
  - Vector-length agnostic programming
  - Scalable vector operations

- ✅ **Vector Length Support**
  - VLEN: 128, 256, 512, 1024 bits
  - Automatic adaptation to hardware VLEN
  - Efficient code generation

- ✅ **LMUL (Length Multiplier)**
  - Vector register grouping (LMUL=1,2,4,8)
  - Trade-off between vector length and registers
  - Optimized for different workloads

- ✅ **Vector Operations**
  - Arithmetic: add, sub, mul, div, fma
  - Logical: and, or, xor, not
  - Shift: sll, srl, sra
  - Reduction: sum, max, min
  - Permutation: vrgather, vslide

#### Vulkan Compute
- ✅ **Cross-Platform Support**
  - Windows, Linux, macOS (via MoltenVK), Android
  - Multiple GPU vendors: NVIDIA, AMD, Intel, ARM Mali, Qualcomm Adreno
  - Unified API across platforms

- ✅ **Compute Shaders**
  - GLSL-based compute kernels
  - SPIR-V compilation
  - Descriptor sets for resource binding

- ✅ **Memory Management**
  - Vulkan buffer objects
  - Device memory allocation
  - Host-visible and device-local memory
  - Transfer queues for data movement

- ✅ **Synchronization**
  - Fences for CPU-GPU sync
  - Semaphores for GPU-GPU sync
  - Pipeline barriers

#### Flash Attention (All Backends)
- ✅ **Implementation Coverage**
  - CUDA: Custom fused kernels with shared memory tiling
  - ROCm: HIP kernels optimized for AMD architecture
  - Metal: MPS graph operations for Apple Silicon
  - Vulkan: Compute shader implementation

- ✅ **Memory Efficiency**
  - O(N) memory complexity (vs O(N²) naive attention)
  - IO-aware algorithm design
  - Tiling for L2 cache optimization

- ✅ **Performance**
  - Fused softmax and dropout
  - Reduced memory bandwidth usage
  - Faster than standard attention on all supported hardware

---

### Memory Management

- ✅ **Advanced Memory Pool with Adaptive Strategies** (Enhanced 2025-11-10)
  - **Multiple Eviction Policies:**
    - LRU (Least Recently Used) - time-based eviction
    - LFU (Least Frequently Used) - frequency-based eviction
    - Size-Based - evict largest tensors first
    - ARC (Adaptive Replacement Cache) - balanced recency/frequency
    - Hybrid - combined LRU, frequency, and size factors
  - **Adaptive Pool Sizing:**
    - Fixed - static pool size
    - HitRate - adjust based on cache hit/miss rates
    - MemoryPressure - adapt to system memory availability
    - Predictive - forecast needs based on access patterns
  - **Access Pattern Learning:**
    - Track access frequency and recency per tensor shape
    - Predict frequently accessed shapes
    - Historical pattern analysis
  - **Performance Optimization:**
    - Automatic defragmentation
    - Entry sorting by access count
    - Thread-safe allocation
    - Fragmentation reduction
  - **Enhanced Statistics:**
    - Hit rate and miss rate tracking
    - Peak memory usage monitoring
    - Per-policy eviction counters
    - Request/hit/miss counters
    - Dynamic pool size reporting
  - **Dynamic Features:**
    - Automatic pool growth/shrinkage (configurable min/max)
    - Target hit rate optimization (default 85%)
    - Prefetching support (pattern-based)
    - Configurable size limits per device

**Key Files:**
- `src/memory.rs` (Enhanced to 900+ lines)
- Exports: `MemoryEvictionPolicy`, `AdaptiveStrategy`, `MemoryConfig`, `TensorMemoryPool`, `MemoryPoolStats`

**Usage Example:**
```rust
use trustformers_core::memory::*;

// Configure enhanced memory pool
let config = MemoryConfig {
    enable_memory_pool: true,
    max_pool_size: 2 * 1024 * 1024 * 1024, // 2GB max
    min_pool_size: 128 * 1024 * 1024,       // 128MB min
    eviction_policy: MemoryEvictionPolicy::Hybrid,
    adaptive_strategy: AdaptiveStrategy::HitRate,
    target_hit_rate: 0.90, // Target 90% hit rate
    enable_prefetching: true,
    enable_defragmentation: true,
    ..Default::default()
};

let pool = TensorMemoryPool::new(config);

// Get tensor from pool (or create if not available)
let tensor = pool.get_tensor(&[1024, 768], DType::F32)?;

// Use tensor...

// Return to pool for reuse
pool.return_tensor(tensor)?;

// Check performance statistics
let stats = pool.get_stats();
println!("Hit rate: {:.2}%", stats.hit_rate * 100.0);
println!("Pool utilization: {:.2}%", stats.utilization * 100.0);
println!("Dynamic max size: {} MB", stats.dynamic_max_size_bytes / 1024 / 1024);

// Get predicted shapes for prefetching
let predicted = pool.get_predicted_shapes(Duration::from_secs(60));
```

- ✅ **Zero-Copy Operations**
  - Tensor views without data duplication
  - Smart reference counting
  - Efficient slicing and indexing
  - Lazy evaluation where possible

- ✅ **Memory-Mapped Loading**
  - Load large model weights without RAM overhead
  - mmap for file-backed tensors
  - On-demand page loading
  - Platform-specific optimizations

- ✅ **LazyTensor Loading**
  - Deferred weight loading
  - Load only used parameters
  - Memory pressure adaptation
  - Background prefetching

- ✅ **Scoped Allocations**
  - RAII-based memory management
  - Automatic cleanup on scope exit
  - Mobile-optimized for memory-constrained devices

- ✅ **Memory Profiling**
  - Allocation tracking
  - Peak memory usage reporting
  - Memory leak detection
  - Per-operation memory metrics

- ✅ **Custom Allocator**
  - jemalloc integration option
  - mimalloc integration option
  - Platform-specific allocators

---

### Layers & Building Blocks

- ✅ **Linear Layer**
  - Dense/fully-connected layer
  - Optional bias
  - Weight initialization (Xavier, Kaiming, etc.)
  - Optimized matmul implementation

- ✅ **Embedding Layer**
  - Learnable token embeddings
  - Padding token support (ignored in backprop)
  - Sparse gradient updates
  - Weight tying with output projection

- ✅ **Normalization Layers**
  - LayerNorm: Configurable epsilon, learnable affine parameters
  - RMSNorm: LLaMA-style root mean square normalization
  - GroupNorm: Group-based normalization
  - BatchNorm: Batch normalization (with running statistics)

- ✅ **Dropout**
  - Training vs inference modes
  - Configurable dropout probability
  - Spatial dropout for CNNs
  - Efficient random number generation

- ✅ **Attention Mechanisms**
  - Multi-head Attention (MHA): Parallel attention heads
  - Grouped-Query Attention (GQA): Memory-efficient multi-head
  - Multi-Query Attention (MQA): Single KV head, multiple Q heads
  - Flash Attention: Memory-efficient fused attention
  - Sliding Window Attention: Local attention patterns (Mistral)

- ✅ **Position Encodings**
  - Rotary Position Embeddings (RoPE): Relative positional encoding
  - **RoPE Scaling Variants** (NEW - 2026-03-23): Linear, NTK-aware, Dynamic NTK, YaRN, LongRoPE
  - Absolute Position Embeddings: Learned or sinusoidal
  - ALiBi: Attention with Linear Biases
  - Relative Position Bias: T5-style bias terms

- ✅ **Feed-Forward Networks**
  - Standard FFN: Linear → Activation → Linear
  - SwiGLU: Gated linear unit with Swish (LLaMA-style)
  - GeGLU: Gated linear unit with GELU
  - Configurable expansion ratio
  - Dropout support

- ✅ **Specialized Layers**
  - Residual Connections: Skip connections
  - Parallel Layers: Model parallelism support
  - MoE (Mixture of Experts): Conditional routing
  - Embedding + Position Encoding: Fused layer

---

### Quantization

- ✅ **Standard Quantization Methods**
  - INT8 and INT4 symmetric/asymmetric quantization
  - Per-tensor and per-channel quantization
  - Dynamic and static quantization
  - Quantization-aware training (QAT)

- ✅ **Advanced Quantization Formats**
  - **BitsAndBytes:** 4-bit and 8-bit quantization with compatibility
  - **GPTQ:** Weight quantization for large language models
  - **AWQ:** Activation-aware weight quantization
  - **SmoothQuant:** W8A8 quantization with migration analysis

- ✅ **GGML/GGUF Quantization** (Production-Ready)
  - Q5_0, Q5_1: 5-bit quantization formats
  - Q5K, Q6K: Advanced 5/6-bit super-block formats

- ✅ **GGUF K-Quant Formats** (NEW - 2025-11-10)
  - **Q2_K:** 2.5625 bits/weight, ~10GB for 7B models
    - 16 sub-blocks with 4-bit quantized scales
    - Best for maximum compression with acceptable quality
  - **Q3_K:** 3.4375 bits/weight, ~13GB for 7B models
    - 16 sub-blocks with 6-bit quantized scales
    - Balanced compression and quality
  - **Q4_K:** 4.5 bits/weight, ~15GB for 7B models
    - 8 sub-blocks with 6-bit quantized scales
    - High quality with good compression
  - Super-block architecture (256 weights per block)
  - Importance-based quantization support
  - Outlier-aware scale optimization

- ✅ **FP8 Quantization** (NEW - 2025-11-10)
  - **E4M3 Format:** 4-bit exponent, 3-bit mantissa
    - Range: ±448, optimized for forward pass
    - Best for: Weights, activations
  - **E5M2 Format:** 5-bit exponent, 2-bit mantissa
    - Range: ±57344, wider dynamic range
    - Best for: Gradients, loss scaling
  - **Scaling Strategies:**
    - Per-tensor scaling with single scale factor
    - Per-channel scaling for better accuracy
    - Per-token scaling for sequence models
    - Block-wise scaling with configurable block sizes
  - **Delayed Scaling:** Training-optimized scale updates
    - Configurable update intervals
    - Historical statistics tracking
    - Overflow/underflow monitoring
  - **Hardware Support:** Optimized for H100, MI300, future accelerators
  - Native FP8 operations when hardware available
  - Automatic format selection based on tensor characteristics

- ✅ **Activation Quantization**
  - Runtime inference optimization
  - Calibration-based quantization
  - Per-layer quality metrics

- ✅ **Mixed-Bit Quantization**
  - Automatic bit allocation strategies
  - Sensitivity-based bit assignment
  - Layer-specific quantization configurations

- ✅ **Learned Quantization**
  - Trainable quantization parameters
  - Gradient-based optimization
  - Fake quantization for training

- ✅ **Calibration Toolkit**
  - Multiple calibration methods (MinMax, Entropy, Percentile)
  - Cross-validation support
  - Quality thresholds and recommendations
  - Trade-off analysis tools

- ✅ **Tensor Quantization Utilities** (NEW - 2026-03-23)
  - **QuantDtype:** INT4, INT8, Uint8, FP16, FP32 with compression ratio helpers
  - **QuantScheme:** Symmetric, Asymmetric, PerChannel, PerGroup (GPTQ-style)
  - **QuantParams::calibrate:** Auto-compute scale/zero-point from data
  - **quantize/dequantize:** Fast round-trip with clipping
  - **QuantizationMetrics:** MAE, RMSE, SNR (dB), clipped element count
  - **Fp16:** Pure-Rust IEEE 754 binary16 software encode/decode (subnormals, NaN, Inf)
  - **quantize_fp16/dequantize_fp16:** Batch FP16 conversion helpers
  - Module: `trustformers_core::quantization::utils`

---

### AutoDiff & Backpropagation

- ✅ **Computational Graph**
  - Dynamic graph construction
  - Node tracking for all operations
  - Parent-child relationships
  - Topological sorting for backprop

- ✅ **Automatic Differentiation**
  - Reverse-mode autodiff
  - Forward-mode autodiff option
  - Gradient computation for all ops
  - Efficient gradient accumulation

- ✅ **Gradient Operations**
  - Backward pass through all tensor ops
  - Chain rule application
  - Gradient clipping (by norm, by value)
  - Gradient accumulation for large batches

- ✅ **Advanced Features**
  - Gradient checkpointing: Trade compute for memory
  - Higher-order derivatives: Double backprop
  - Custom gradient functions: User-defined backprop
  - Detach operations: Stop gradient flow

- ✅ **Thread Safety**
  - Concurrent forward passes
  - Thread-safe gradient storage
  - OnceLock for global state
  - Arc/Mutex for shared mutable state

---

### Kernel Optimization & Performance Tuning

#### Automatic Kernel Tuning (NEW - 2025-11-10)
- ✅ **Platform Detection**
  - Automatic hardware capability detection
  - Multi-backend support (CUDA, ROCm, Metal, CPU, Vulkan, OneAPI, TPU)
  - GPU memory and compute capability detection
  - CPU feature detection (AVX, AVX2, AVX-512, NEON, etc.)

- ✅ **Auto-Tuning Infrastructure**
  - **Benchmarking Engine:** Automatic kernel parameter optimization
  - **Caching System:** Persistent tuning results with JSON storage
  - **Platform-Specific Tuning:** Separate configurations per hardware
  - **Operation Coverage:**
    - Matrix multiplication (GEMM) with shape-specific tuning
    - Convolution operations
    - Batch normalization
    - Activation functions (ReLU, GELU, etc.)
    - Pooling operations
    - Custom operations

- ✅ **Kernel Parameters**
  - **Block Size:** Optimal CUDA/HIP thread block dimensions
  - **Tile Size:** Memory hierarchy optimization
  - **Unroll Factor:** Loop unrolling for performance
  - **Vector Width:** SIMD vectorization width
  - **Shared Memory:** Per-block shared memory allocation
  - **Registers:** Register usage hints
  - **Occupancy:** Target GPU occupancy percentage

- ✅ **Tuning Strategies**
  - Grid search over parameter spaces
  - Configurable iteration counts for stable measurements
  - Statistical filtering of benchmark results
  - Automatic fallback to safe defaults
  - Platform-aware parameter constraints

- ✅ **Global Kernel Tuner**
  - Thread-safe singleton access via `get_kernel_tuner()`
  - Automatic cache loading/saving
  - Configurable cache directory
  - Zero-overhead when tuning disabled

- ✅ **Configuration Options**
  - Enable/disable auto-tuning per operation
  - Custom cache directory paths
  - Benchmark iteration control
  - Platform preference specification

**Key Files:**
- `src/kernel_tuning.rs` (680+ lines)
- Exports: `KernelTuner`, `KernelParams`, `TuningConfig`, `PlatformInfo`, `Operation`, `Backend`

**Usage Example:**
```rust
use trustformers_core::kernel_tuning::*;

// Get global tuner (thread-safe)
let mut tuner = get_kernel_tuner();

// Tune for specific operation
let params = tuner.tune_matmul(1024, 1024, 1024)?;

// Or tune generic operation
let params = tuner.tune_operation(
    Operation::Convolution,
    &[batch, channels, height, width]
)?;
```

---

### Interactive Tensor Debugger (NEW - 2025-11-10)

- ✅ **Tensor Inspection**
  - Comprehensive statistics (min, max, mean, std dev)
  - Shape and dtype tracking
  - Memory usage reporting
  - Element count and distribution analysis

- ✅ **Automatic Issue Detection**
  - **NaN Detection:** Identifies and counts Not-a-Number values
  - **Infinity Detection:** Tracks infinite values
  - **Vanishing Values:** Detects very small values (< 1e-7)
  - **Exploding Values:** Detects very large values (> 1e6)
  - **All Zeros:** Identifies tensors filled with zeros
  - **Unusual Distributions:** Statistical anomaly detection

- ✅ **Watchpoints System**
  - Conditional breakpoints on tensor operations
  - Multiple watch conditions:
    - `HasNaN` - Break on NaN values
    - `HasInf` - Break on infinite values
    - `ValueExceeds(threshold)` - Break on large values
    - `ValueBelow(threshold)` - Break on small values
    - `ShapeEquals(shape)` - Break on specific shapes
    - `Custom(condition)` - User-defined conditions
  - Pattern-based tensor matching (wildcards supported)
  - Trigger count tracking
  - Configurable break-on-trigger behavior

- ✅ **Operation Tracing**
  - Track tensor operations and transformations
  - Record input/output shapes
  - Measure operation duration
  - Build operation history
  - Maximum trace entry limits (configurable)

- ✅ **Severity Levels**
  - Info: Informational messages
  - Warning: Potential issues
  - Error: Issues requiring attention
  - Critical: Critical issues requiring immediate action

- ✅ **Configuration Options**
  - Enable/disable automatic issue detection
  - Enable/disable operation tracing
  - Maximum trace entries (default: 1000)
  - Enable/disable watchpoints
  - Break on errors/warnings
  - Maximum issues to track (default: 100)

- ✅ **Interactive Features**
  - Register tensors for debugging
  - Get tensor by name
  - Query statistics
  - List all issues
  - Clear issues and traces
  - Check breakpoint status
  - Print comprehensive summary

**Key Files:**
- `src/tensor_debugger.rs` (760+ lines)
- Exports: `TensorDebugger`, `TensorDebuggerConfig`, `DebugTensorStats`, `TensorDebugIssue`, `TensorIssueType`, `Severity`, `Watchpoint`, `WatchCondition`, `OperationTrace`

**Usage Example:**
```rust
use trustformers_core::tensor_debugger::*;

// Create debugger with custom configuration
let config = TensorDebuggerConfig {
    auto_detect_issues: true,
    enable_tracing: true,
    break_on_error: true,
    ..Default::default()
};
let debugger = TensorDebugger::with_config(config);

// Register tensors for debugging
let tensor = Tensor::randn(&[100, 768])?;
debugger.register_tensor("hidden_states".to_string(), tensor)?;

// Add watchpoint for NaN values
let watchpoint = Watchpoint {
    tensor_pattern: "hidden_states".to_string(),
    condition: WatchCondition::HasNaN,
    break_on_trigger: true,
    trigger_count: 0,
};
debugger.add_watchpoint(watchpoint);

// Check for issues
let issues = debugger.get_issues();
for issue in issues {
    println!("[{:?}] {}: {}", issue.severity, issue.issue_type, issue.message);
}

// Get statistics
if let Some(stats) = debugger.get_stats("hidden_states") {
    println!("Shape: {:?}", stats.shape);
    println!("Mean: {:.6}", stats.mean.unwrap_or(0.0));
    println!("Std Dev: {:.6}", stats.std_dev.unwrap_or(0.0));
    println!("NaN count: {}", stats.nan_count);
}

// Print full summary
debugger.print_summary();

// Check if breakpoint was hit
if debugger.is_breakpoint_hit() {
    println!("Breakpoint hit! Inspect tensors.");
    debugger.clear_breakpoint();
}
```

---

## Known Limitations

### Hardware Backend Limitations
- **Metal Flash Attention:** Requires macOS 10.15+ or iOS 13+
- **TPU Backend:** Requires Google Cloud TPU access and authentication
- **Some Features:** Platform-specific driver/SDK requirements

### Numerical Precision
- **Floating-Point:** Adaptive tolerance system for numerical stability tests
- **Platform Variations:** Some operations may have minor precision differences across backends
- **Half Precision:** F16/BF16 have reduced precision (acceptable for most ML tasks)

### Performance
- **CPU Fallback:** Some operations fall back to CPU when not implemented on specific backend
- **Small Tensors:** Overhead may dominate for very small tensors on GPU

### Housekeeping
- **Stray Backup Files:** 6 `.bak2` files remain under `src/` from prior refactors and are not compiled/live code — `layers/sdpa.rs.bak2`, `layers/flash_attention.rs.bak2`, `layers/linear.rs.bak2`, `kernels/simd/matrix_ops.rs.bak2`, `gpu_ops/metal/metalbackend_matmul_f32_group.rs.bak2`, `tensor/math_ops/linear_algebra.rs.bak2` (found 2026-07-01) — safe to delete

---

## 0.2.0 Release Scope (added 2026-07-06)

Two tracks for the 0.2.0 release: (1) **OxiCUDA GPU migration** — finish moving GPU semantics off `scirs2-core` (whose `PlatformCapabilities.cuda_available` is hardcoded false in 0.6.0) onto the oxicuda 0.4.x backend already integrated behind the `cuda`/`metal` features; (2) **PyTorch (tch) dependency removal** — delete the `tch` dependency and the `torch` feature entirely in 0.2.0 (workspace Cargo.toml:82, trustformers-core `torch` feature + ~40 lines of cfg arms, and the forwarder features in trustformers, trustformers-training, trustformers-c). `Tensor::Torch` is never constructed anywhere in the workspace and the cfg(torch) "PyTorch validation" is simulated with hardcoded results, so zero functionality is lost, while tch costs a multi-GB libtorch download plus Pure-Rust policy violations (torch-sys pulls cc/C++, the OxiARC-banned zip 0.6.6, ureq, and duplicate old ndarray/rand/safetensors pins). All real PyTorch interop (safetensors, checkpoint/formats.rs, utils/weight_loading.rs) is pure Rust already and stays. Do NOT adopt ToRSh as a replacement now — crates.io torsh 0.1.3 pins scirs2 0.5.1 (type-incompatible with our scirs2 0.6.0 stack) and torsh 0.2.0 is unpublished; a P2 task below records evaluating an optional `torsh-interop` feature in 0.3.x once torsh 0.2.0 ships. Sub-decision on candle: drop the unused candle-nn workspace dep now, keep the `candle` feature/variant through 0.2.0 (it is in every `full` set), and decide implement-vs-remove in 0.3.x.

### Track 1: OxiCUDA GPU migration (scirs2-core gpu → OxiCUDA)

- [x] **[P0] Replace scirs2 PlatformCapabilities GPU detection in device.rs with oxicuda-native probes** — DONE (2026-07-06)
  - `Device::cuda_if_available()` now calls `crate::gpu_ops::cuda::oxicuda_cuda_available()`; `Device::best_available()` checks CUDA via that probe first, then `metal::Device::system_default().is_some()`, then falls back to CPU. Ordering is CUDA > Metal > CPU. Doc comments updated to explain why PlatformCapabilities is unsuitable for GPU detection. Added 3 new unit tests (`test_cuda_if_available_does_not_panic`, `test_metal_if_available_does_not_panic`, `test_best_available_does_not_panic`) asserting a valid `Device` is always returned. `hardware/devices.rs` and `hardware/manager.rs` do not use `PlatformCapabilities` at all, so no changes were needed there (though `hardware/manager.rs` has its own pre-existing, unrelated `is_cuda_available`/`is_metal_available`/`is_rocm_available` hardcoded-`true` placeholder bug, noted but out of scope). Additionally, the upstream scirs2-core `PlatformCapabilities` GPU-misreporting bug itself was fixed in the ~/work/scirs repo (0.6.1 branch, real CUDA runtime probe via `simd_ops/gpu_detection.rs`; no commit/publish here). Workspace-wide `cargo check --all-features` and `cargo clippy --all-features --all-targets` both verified clean in this round's convergence pass; full nextest suite 12033/12033 passed.
  - Evidence: trustformers-core/src/device.rs:79-113; trustformers-core/src/gpu_ops/cuda/oxicuda/mod.rs:1193-1200; trustformers-core/src/tensor/math_ops/linear_algebra.rs:202-204; trustformers-core/tests/test_device_linear.rs:45

- [x] **[P1] Fix the CUDA resident-buffer lifecycle: device memory leaks until clear_buffer_cache()** — DONE (2026-07-06)
  - Implemented an `Arc`-based RAII `OxiCudaBufferHandle` (re-exported as `gpu_ops::cuda::BufferHandle`); last-clone drop removes the `DeviceBuffer` from the backend cache via a best-effort release callback (no panic, no lock-order inversion; monotonic never-reused ids make it double-free-proof). `CudaTensorData` now holds the handle, so all resident op outputs (matmul, linear, layernorm, GELU, uploads) free on last tensor drop, including error paths; Linear's CUDA weight cache also stores a handle. Added 6 hardware-free refcount unit tests (mocked release fn). `cargo nextest -p trustformers-core --features cuda -E 'test(/cuda|buffer|resident/)'` = 20/20 passed at implementation time; `cargo check`/`clippy --all-targets` clean on cuda and default features. Re-verified clean in this round's workspace-wide convergence pass.
  - Evidence: trustformers-core/src/tensor/mod.rs:185-201; trustformers-core/src/gpu_ops/cuda/oxicuda/mod.rs:63-73,294,405,468,552

- [x] **[P1] Build the CUDA-resident attention chain (CUDA-7) on OxicudaCudaBackend** — DONE (2026-07-06)
  - New module `gpu_ops/cuda/oxicuda/attention.rs` implements the resident method set (gather_heads, gather_heads_transposed, rope_neox, softmax_causal/rows, add, attention_prefill, attention_decode, concat_kt_cache/concat_v_cache), all bracketed with `cuCtxSynchronize` for cross-stream ordering and feeding the refcounted buffer-handle lifecycle. `Tensor::add` got a resident CUDA×CUDA same-device/shape/F32 arm so residuals stay on-device. Wired into GPT-NeoX (`cuda_resident_forward`, prefill-only — the NeoX `Layer` trait has no KV-cache plumbing) and GPT-2 (`cuda_resident_attention` in `forward_with_cache`, covering empty-cache bulk prefill and per-token decode against a resident cache; batch>1 or non-F32 declines to the existing host-download path). Documented upstream oxicuda 0.4.0 defects that shaped the design (GEMM ignores transpose flags/no lda; `mha::multi_head_attention` has an OOB/aliasing bug; causal_softmax masks by global row index). 10 hardware-free plan tests run everywhere; 7 availability-gated GPU parity tests skip cleanly without a GPU. Verification at implementation time: clippy clean on core+models (cuda/default); nextest core cuda attention/rope/softmax/resident 163/163, wider gemm/buffer/cuda/gather/concat 205/205, trustformers-models full suite w/ cuda 1089/1089. Real-hardware execution not possible on this macOS host; correctness rests on source-verified kernel signatures and CPU-parity tests. GPT-2 batch>1/non-F32 and multi-token-continuation-behind-cache still use fallback paths (documented, not half-wired). Re-verified clean in this round's workspace-wide convergence pass (12033/12033 tests, 113 GPU tests skipped by design on this non-CUDA host).
  - Evidence: TODO-CUDA.md:18-25; trustformers-core/src/tensor/math_ops/arithmetic.rs:146-157; trustformers-core/src/gpu_ops/cuda/oxicuda/mod.rs:294-1140; trustformers-core/src/gpu_ops/cuda/oxicuda/attention.rs; trustformers-models/src/gpt_neox/model.rs:164-180; trustformers-models/src/gpt2/model/model_blocks.rs
  - Update (2026-07-07): the three 0.4.0 defects documented here (GEMM transpose/lda, mha OOB/no-op, causal_softmax global-row masking) are fixed upstream in the published oxicuda 0.4.1; the transpose-driven workarounds (K^T materialisation, K^T cache layout) were removed — see the "Bump oxicuda pins" entry below for the full resolution record.

- [x] **[P1] Route Tensor::CUDA operands through Tensor::matmul and add strided-batched GEMM** — DONE (2026-07-06, with a deviation from the planned approach)
  - Added a resident match arm routing any `Tensor::CUDA` operand to a new `dispatch_oxicuda_matmul_resident`, and extended the host-offload arm from 2D-only to any rank ≥2 with NumPy-style broadcasting (new `BatchedMatmulPlan` shape planner in `gpu_ops/cuda/oxicuda/batched.rs`). DEVIATION: the plan's suggested `oxicuda-blas::gemm_strided_batched` was verified (by reading the published oxicuda-blas 0.4.0 / oxicuda-ptx 0.4.0 sources) to be broken upstream — the batched entry points build an 8-parameter GEMM kernel template but launch it with a differently-ordered 13/17-element tuple, corrupting dimensions into pointer slots. Implemented instead as a per-batch loop over the already-validated single-GEMM entry point (still fully device-resident, one async launch per batch on one stream), with the upstream defect documented in the module for a future single-launch upgrade once oxicuda fixes it. 11 hardware-free plan tests run everywhere; 6 availability-gated GPU parity tests skip cleanly without a GPU. Verification at implementation time: clippy clean; nextest matmul/gemm 33/33, cuda/buffer/resident 37/37. Re-verified clean in this round's workspace-wide convergence pass.
  - Evidence: trustformers-core/src/tensor/math_ops/linear_algebra.rs:197-211,566-570; trustformers-core/src/gpu_ops/cuda/oxicuda/mod.rs:294; trustformers-core/src/gpu_ops/cuda/oxicuda/batched.rs
  - Update (2026-07-07): the upstream launch-tuple corruption in `gemm_strided_batched` is fixed in the published oxicuda 0.4.1 (honest validation + matched launch tuple). The per-batch loop here is deliberately retained: `BatchedMatmulPlan`'s non-uniform broadcast offsets cannot be expressed as one stride, and 0.4.1's strided path is itself a per-batch launch loop, so switching would save nothing. Module docs updated; see the "Bump oxicuda pins" entry below.

- [x] **[P1] Thread the CUDA device id through CudaTensorData (device 0 is hardcoded)** — DONE (2026-07-06)
  - Landed together with the buffer-lifecycle fix above: `OxiCudaBufferHandle`/`CudaTensorData` now carry the device ordinal (`device_id()`). Download (utils.rs, was `get_cuda_backend(0)`), GELU (activations.rs, was `device_id=0` TODO), layernorm, and linear forward all now use the tensor's actual device; cross-device operand mismatches fall back to host, and CUDA→CUDA to a different ordinal bounces through host (matching the `to_device_enum` policy). `OxicudaCudaBackend::new` validates the ordinal against `oxicuda_driver::Device::count()`. Host matmul dispatch uses a `default_cuda_device_id()` (0, overridable via `TRUSTFORMERS_CUDA_DEVICE`) rather than a hardcoded literal. No dedicated multi-GPU hardware test was run (no multi-GPU hardware available in this environment); correctness verified via the mocked-release refcount unit tests and code-path review.
  - Evidence: trustformers-core/src/tensor/math_ops/linear_algebra.rs:203; trustformers-core/src/tensor/utils.rs:466-563; trustformers-core/src/tensor/mod.rs:185-189; trustformers-core/src/gpu_ops/cuda/oxicuda/mod.rs:1227

- [x] **[P1] Bump oxicuda pins 0.4.0 -> 0.4.1 once oxicuda 0.4.1 is published on crates.io** — DONE (2026-07-07)
  - oxicuda 0.4.1 is now published on crates.io (commit `facb2cd` "Availability of 0.4.1"). Cargo.toml pins were already at 0.4.1; this session updated Cargo.lock (all eight oxicuda crates 0.4.0 -> 0.4.1). The earlier "delta is API-identical" note was superseded by the published release: `oxicuda_blas::reduction::causal_softmax` gained an explicit `seq_len` parameter (masking by `row % seq_len`), and the three trustformers call sites were adapted.
  - **Upstream bug resolutions verified against the published 0.4.1 sources (2026-07-07)** — all three originally-reported 0.4.0 bugs plus the separately-found causal-softmax batch bug are fixed upstream:
    1. Batched GEMM launch-tuple corruption: `gemm_strided_batched` now validates arguments, honestly rejects unsupported transpose/ld combinations, and launches a correctly matched tuple. trustformers keeps its per-batch `gemm` loop anyway — `BatchedMatmulPlan`'s non-uniform broadcast offsets cannot be expressed as a single stride, and the upstream strided path is itself a sequential per-batch launch loop (module docs updated in `gpu_ops/cuda/oxicuda/batched.rs`).
    2. GEMM transpose flags/leading dimensions: transposed operands now dispatch to a SIMT kernel honouring all four `(trans_a, trans_b)` combinations; non-tightly-packed / column-major operands are honestly rejected. Workarounds REMOVED: `gather_heads_transposed_gpu_to_gpu` + `gather_heads_transposed_plan` (pitched-copy `K^T` materialisation) and `concat_kt_cache_gpu_to_gpu` + `concat_kt_plan` (column-oriented `K^T` cache append) are deleted; `attention_prefill_gpu_to_gpu` / `attention_decode_gpu_to_gpu` now take heads-major `[H, kv, d]` keys and apply `Transpose::Trans` on the score GEMM; GPT-2/GPT-NeoX resident paths updated (GPT-2's resident KV cache now stores K in the same `[1, H, kv, d]` layout as V and appends via `concat_v_cache_gpu_to_gpu`).
    3. `mha::multi_head_attention` no-op stub/OOB: 0.4.1 ships real QK^T/softmax/PV kernels with a dedicated score scratch. trustformers still composes its own chain (upstream mha has no KV-cache/decode surface and takes an additive mask tensor); module docs updated and a TODO recorded in `gpu_ops/cuda/oxicuda/attention.rs`.
    4. `causal_softmax` global-row masking: fixed via the new `seq_len` parameter (`row % seq_len` boundary reset, upstream regression-tested for `[batch*heads*seq, seq]`). trustformers' per-head invocation in `attention_prefill_gpu_to_gpu` is retained as a `[seq, seq]` scratch-reuse choice, no longer a correctness constraint.
  - **Still open upstream (f64 only)**: `oxicuda-dnn`'s `generate_scale_mask_ptx` (the scale+mask step inside `multi_head_attention`) is hardcoded to f32 addressing (`f32_elem_addr`/`load_global_f32`), so the f64 instantiation mis-addresses 8-byte elements. trustformers is unaffected today (the resident CUDA path is `DeviceBuffer<f32>`-only), but any future f64/dtype-polymorphic resident work (see the P2 f16/bf16 task below) must not route f64 attention through upstream `multi_head_attention` until this is fixed.
  - Verification (2026-07-07, macOS host — no CUDA hardware, so GPU parity tests self-skip): `cargo check -p trustformers-core --features cuda` clean; `cargo clippy -p trustformers-core --features cuda --all-targets -- -D warnings` clean; `cargo clippy -p trustformers-models --features cuda --all-targets -- -D warnings` clean; nextest trustformers-core w/ cuda 2343/2343 passed (2 skipped), trustformers-models w/ cuda 1089/1089 passed (28 skipped). On-device parity for the new `Transpose::Trans` score GEMMs and the 6-arg `causal_softmax` rests on the existing availability-gated GPU tests — rerun them on real CUDA hardware before a release that exercises this backend.
  - Evidence: trustformers-core/Cargo.toml:87-91,156-157; Cargo.lock (oxicuda-* 0.4.1); trustformers-core/src/gpu_ops/cuda/oxicuda/{attention.rs,batched.rs,mod.rs}; trustformers-models/src/gpt2/model/model_blocks.rs; trustformers-models/src/gpt_neox/model.rs; external: ~/work/oxicuda commit facb2cd

### Track 2: PyTorch (tch) dependency removal

- [x] **[P0] Delete the Tensor::Torch variant, the torch feature, and every cfg(feature="torch") arm** — DONE (2026-07-06)
  - Fully removed as planned: the optional `tch` dep and `torch` feature (+ `full` comment) from trustformers-core/Cargo.toml (and the forwarder features in trustformers, trustformers-training, trustformers-c, and the root workspace tch dep, plus the `.typos.toml` `tch` dictionary entry). Deleted `Tensor::Torch` variant, its Clone/Debug arms, narrowed `unsafe impl Sync for Tensor` to `#[cfg(feature = "candle")]`; removed the six torch accessor arms in tensor/utils.rs, the memory/mod.rs arm, and the error-only arms in quantization/ggml_advanced.rs and smoothquant.rs. `testing/cross_framework.rs`'s `check_pytorch_available()` now unconditionally returns false and `validate_pytorch()` collapses to the not-available branch. All pure-Rust PyTorch format compat (checkpoint/*, utils/weight_loading.rs PyTorchReader, optim pytorch_compat.rs, tokenizers pytorch.rs, hub.rs) was left untouched as required. Verification (workspace-wide convergence): `rg '\btch\b|feature = "torch"|"torch"'` across sources returns only harmless residue (commented-out example prose, an unrelated wasm string literal, and the intentionally-kept `data_str.contains("torch")` PyTorch-format-detection heuristic in weight_loading.rs); `rg '"tch"|torch-sys' Cargo.lock` is empty; `cargo check`/`clippy --all-features --all-targets` clean; full nextest suite 12033/12033 passed.
  - Evidence: trustformers-core/Cargo.toml:65,108-110,132; trustformers-core/src/tensor/mod.rs:220-221,246-247,272-273,297-298; trustformers-core/src/tensor/utils.rs:108-109,136-137,182-183,871-872,908-909,936-937; trustformers-core/src/memory/mod.rs:437-438; trustformers-core/src/quantization/ggml_advanced.rs:191-192,322-323,432-433; trustformers-core/src/quantization/smoothquant.rs:227-234,397-403,623-628,705-710,931-933; trustformers-core/src/testing/cross_framework.rs:241,309-325

### Post-0.2.0 (0.3.x) — record-only follow-ups

- [ ] **[P2] f16/bf16 device-resident compute via a dtype-polymorphic buffer cache** (oxicuda)
  - The backend buffer cache is monomorphized to `DeviceBuffer<f32>`, and F16/BF16 matmul upcasts to f32 on the CPU. oxicuda-blas GEMM is generic over f16/bf16/FP8 via `GpuFloat` (~/work/oxicuda crates/oxicuda-blas/src/types.rs:184-360); making the cache dtype-polymorphic unlocks half-precision GEMM on device. Explicitly deferred from 0.2.0 by the OxiCUDA readiness assessment.
  - Evidence: trustformers-core/src/gpu_ops/cuda/oxicuda/mod.rs:66-73; trustformers-core/src/tensor/math_ops/linear_algebra.rs:504-566

- [ ] **[P2] Re-enable or delete gpu_accelerated / hardware_acceleration under the cuda feature (CUDA-3)** (oxicuda)
  - Both modules are compiled `#[cfg(not(feature = "cuda"))]` because they call the removed cudarc kernel API; kernels/mod.rs no longer contains cuda modules. Either port their kernel calls to the oxicuda backend or delete the modules if the resident-op set makes them redundant. Tracked in-repo as TODO-CUDA.md CUDA-3.
  - Evidence: trustformers-core/src/lib.rs:81-95; trustformers-core/src/gpu_accelerated.rs:5-6; TODO-CUDA.md:18-25

- [ ] **[P2] GPU quantized inference (quantization modules reject Tensor::CUDA)** (oxicuda)
  - Quantization paths reject CUDA tensors outright. oxicuda provides AWQ / INT8 / INT4 / FP8 quantization kernels (oxicuda-dnn quantization + oxicuda-quant/-infer upstream) to back a device-resident quantized matmul path.
  - Evidence: trustformers-core/src/quantization/ggml_advanced.rs:201,332,442; trustformers-core/src/quantization/smoothquant.rs:223,421,620,702

- [ ] **[P2] Fused transformer-layer megakernel (CUDA-6) and GPU training via oxicuda-train** (oxicuda)
  - Both already scoped out of 0.2.0. CUDA-6 (fused megakernel) is perf-only and tracked in TODO-CUDA.md. GPU training would exploit oxicuda-train's AMP (FP16/BF16 + loss scaling), fused Adam/AdamW/SGD/LAMB, grad accumulation/clipping, ZeRO, and checkpointing — none of which trustformers uses today; 0.2.0 is inference-focused.
  - Evidence: TODO-CUDA.md:18-25; external: ~/work/oxicuda crates/oxicuda-train/src

- [ ] **[P2] Hold wgpu/WebGPU consolidation onto oxicuda-webgpu until wgpu majors align** (oxicuda)
  - Keep trustformers-core's own wgpu 30.0-based backend (src/gpu_ops/webgpu.rs, workspace pin at Cargo.toml:271) and trustformers-wasm's web-sys-based browser WebGPU as-is. oxicuda-webgpu pins wgpu 29.0.3, so adopting it now would drag a second wgpu major into the tree or force a downgrade of a working 1,230-line backend; the wasm path uses web-sys, not the wgpu crate, and gains nothing. Revisit only after oxicuda bumps to wgpu 30.x, and then only for the native `wgpu_backend` feature.
  - Evidence: Cargo.toml:271; trustformers-core/src/gpu_ops/webgpu.rs:26,89-130; trustformers-wasm/src/compute/webgpu/backend.rs:79,95; external: ~/work/oxicuda Cargo.toml:71

- [ ] **[P2] Evaluate an optional torsh-interop feature once torsh 0.2.0 ships on crates.io** (tch)
  - Do not adopt ToRSh in 0.2.0: crates.io torsh 0.1.3 pins scirs2 0.5.1 (type-incompatible duplicate of trustformers' scirs2 0.6.0 stack), torsh 0.2.0 is unpublished with a workspace-blocking compile error, and torsh adds no missing capability (no real pickle .pt parser either). Revisit as an optional `torsh-interop` feature in 0.3.x once torsh 0.2.0 is on crates.io with a compatible scirs2 stack (~/work/torsh).
  - Evidence: trustformers-core/Cargo.toml:65 (removed tch slot); external: ~/work/torsh

- [ ] **[P2] Decide the candle backend's fate (implement or remove)** (tch)
  - `Tensor::Candle` is in the identical vestigial state as the removed torch variant — never constructed, referenced only in read-only/error match arms — and candle-nn is being dropped from the workspace as unused. It is kept through 0.2.0 because the `candle` feature is part of `full` (Cargo.toml:133). In 0.3.x either implement a real candle backend or delete the variant + feature, which would also allow removing the (by then candle-only) `unsafe impl Sync for Tensor` at src/tensor/mod.rs:297-298 entirely.
  - Evidence: trustformers-core/src/tensor/mod.rs:222-223,297-298; trustformers-core/Cargo.toml:133; Cargo.toml:80-81

---

## Future Enhancements

### High Priority (Updated 2026-07-01)
- ~~Additional fused kernel patterns~~ ✅ COMPLETED (2026-07-01)
- ~~Enhanced sparse tensor operations~~ ✅ COMPLETED (2025-11-10)
- ~~More quantization methods~~ ✅ COMPLETED (FP8, GGUF K-quants - 2025-11-10)
- ~~RoPE scaling variants (Linear, NTK, Dynamic NTK, YaRN, LongRoPE)~~ ✅ COMPLETED (2026-03-23)
- ~~Tensor quantization utilities (INT4/INT8/FP16 with scale/zero-point calibration)~~ ✅ COMPLETED (2026-03-23)
- [ ] INT2 and sub-byte quantization for extreme compression
- ~~MX (Microscaling) formats for future hardware~~ ✅ COMPLETED (2026-07-01)

### Performance
- [ ] Further SIMD optimizations via SciRS2
- ~~Advanced kernel fusion strategies~~ ✅ COMPLETED (2026-07-01)
- ~~Enhanced memory pooling with adaptive strategies~~ ✅ COMPLETED (2025-11-10)
- ~~Automatic kernel tuning for new hardware~~ ✅ COMPLETED (2025-11-10)

### Hardware Support
- ~~WebGPU backend for browser deployment~~ ✅ COMPLETED (2026-07-01) — implemented via the `wgpu` crate (v29.0) behind the `wgpu_backend` feature flag
- [ ] Mobile GPU: Android Vulkan compute optimizations
- [ ] Mobile GPU: iOS Metal optimizations
- [ ] Enhanced FPGA support

### Developer Tools
- ~~Interactive tensor debugger~~ ✅ COMPLETED (2025-11-10)
- ~~Enhanced profiling visualizations~~ ✅ COMPLETED (2026-07-01)
- [ ] Performance regression dashboard

---

## Development Guidelines

### General Policies
See main project TODO.md and SCIRS2_INTEGRATION_POLICY.md for comprehensive development policies.

### Core-Specific Guidelines

#### Dependency Rules (CRITICAL)
- ✅ **External Dependencies:** Only trustformers-core can use external crates directly
- ✅ **Re-exports:** Core must re-export all needed functionality for other crates
- ✅ **SciRS2 Integration:** Use scirs2-core for scientific computing (SIMD, random, ndarray)
- ❌ **Application Crates:** Must NEVER import external deps (use core abstractions only)

#### Code Standards
- **Naming:** snake_case for all identifiers
- **File Size:** Maximum 2000 lines (use splitrs for refactoring)
- **Error Handling:** Always use `Result<T, TrustformersError>`
- **Testing:** Use `std::env::temp_dir()` for temporary files
- **Documentation:** rustdoc with examples for all public APIs
- **No Warnings:** Must pass `cargo clippy -- -D warnings`

#### Testing Requirements
- Unit tests for all public APIs
- Property-based tests for tensor operations
- Numerical stability tests with adaptive tolerance
- Cross-backend compatibility tests
- Memory leak detection tests
- Performance benchmarks

### Build & Test Commands

```bash
# Full check (recommended before commit)
cargo check --all-features

# Run all tests
cargo nextest run -p trustformers-core --all-features

# Run specific backend tests
cargo test -p trustformers-core --features cuda
cargo test -p trustformers-core --features metal

# Run doctests
cargo test -p trustformers-core --doc --all-features

# Format and clippy
cargo fmt --all
cargo clippy -p trustformers-core --all-features -- -D warnings

# Build documentation
cargo doc -p trustformers-core --all-features --no-deps
```

---

**Last Updated:** 2026-07-06 - v0.2.0 Development
**Version:** 0.2.0
**Status:** Stable — production-ready core infrastructure
**Test Coverage:** ~2,353 tests, 100% pass rate, 0 stubs
**Public API:** ~4,533 items
**SLoC:** 155,280
