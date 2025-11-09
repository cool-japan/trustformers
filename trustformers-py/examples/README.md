# TrustformeRS Python Examples

This directory contains examples demonstrating the capabilities of the TrustformeRS Python bindings.

## Setup

First, build and install the Python bindings:

```bash
cd trustformers-py
pip install maturin
maturin develop
```

## Examples

### Comprehensive Demo (`comprehensive_demo.py`)

A complete demonstration of all TrustformeRS features:

```bash
python examples/comprehensive_demo.py
```

This demo showcases:

- **Basic Operations**: Matrix multiplication, softmax, layer normalization
- **Activation Functions**: GELU, SiLU, Mish, Swish, ELU, Hardtanh, SELU, Hardswish, Softplus, Softsign
- **Mathematical Operations**: log, exp, sqrt, power, trigonometric functions, abs, sign
- **Shape Operations**: reshape, transpose, squeeze, unsqueeze
- **Reduction Operations**: sum, mean with axis control
- **Element-wise Operations**: add, multiply, maximum, minimum, clamp
- **Tensor Creation**: zeros, ones, full, random normal
- **Quantization**: INT8, INT4, dynamic quantization with analysis
- **Advanced Operations**: concatenation, gather, broadcasting
- **Attention Operations**: scaled dot-product attention, rotary position embedding
- **Performance Benchmarking**: GFLOPS measurement for key operations

### Tensor Operations Demo (`tensor_operations_demo.py`)

Basic tensor operations demonstration:

```bash
python examples/tensor_operations_demo.py
```

## Features Demonstrated

### 🔢 **Quantization Support**
- INT8 quantization (per-tensor and per-channel)
- INT4 quantization with configurable group sizes
- Dynamic quantization for runtime optimization
- Quantization impact analysis with compression ratios and accuracy metrics
- Full dequantization support

### 🧮 **Advanced Activation Functions**
- **GELU**: Gaussian Error Linear Unit
- **SiLU**: Sigmoid Linear Unit (Swish)
- **Mish**: Self-regularized non-monotonic activation
- **Swish**: x * sigmoid(x)
- **ELU**: Exponential Linear Unit with configurable alpha
- **Hardtanh**: Clamped linear activation
- **SELU**: Scaled Exponential Linear Unit
- **Hardswish**: Hardware-efficient Swish approximation
- **Softplus**: Smooth approximation of ReLU
- **Softsign**: Smooth alternative to tanh

### 🔧 **Mathematical Operations**
- Logarithmic and exponential functions
- Power and square root operations
- Trigonometric functions (sin, cos, tan)
- Element-wise abs and sign functions
- Clamping operations

### 📐 **Shape Manipulation**
- Flexible reshape operations
- Matrix transpose
- Squeeze/unsqueeze for dimension manipulation
- Advanced broadcasting for element-wise operations

### 📊 **Reduction Operations**
- Sum and mean with axis control
- Keepdims support for maintaining tensor dimensions

### 🏗️ **Tensor Creation**
- Zeros, ones, and constant-filled tensors
- Random normal distribution with configurable mean/std
- Efficient memory allocation and initialization

### 🚀 **High-Performance Features**
- SIMD-optimized implementations where available
- Parallel processing with Rayon
- Memory-efficient operations
- Zero-copy tensor handling where possible

### 🤖 **Transformer-Specific Operations**
- Scaled dot-product attention
- Rotary position embedding (RoPE)
- Layer normalization
- Optimized matrix operations for transformer architectures

## Performance

The library is optimized for performance with:
- SIMD instructions for vectorized operations
- Parallel processing for large tensors
- Memory pooling for reduced allocations
- Optimized algorithms for transformer workloads

Typical performance on modern hardware:
- Matrix multiplication: >100 GFLOPS for large matrices
- Activation functions: <1μs for 1M elements
- Quantization: Real-time compression with minimal accuracy loss

## Requirements

- Python 3.8+
- NumPy
- Rust toolchain (for building from source)

## API Compatibility

The API is designed to be familiar to PyTorch and NumPy users:

```python
import numpy as np
from trustformers import TensorOptimized

# Create tensors (NumPy arrays)
a = np.random.randn(100, 100).astype(np.float32)
b = np.random.randn(100, 100).astype(np.float32)

# Perform operations
result = TensorOptimized.matmul(a, b)
activated = TensorOptimized.gelu(result)
quantized = TensorOptimized.quantize_int8(activated)
```

All operations return NumPy arrays for seamless integration with existing Python machine learning workflows.