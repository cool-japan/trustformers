# Installation Guide for TrustformeRS Python

This guide provides detailed instructions for installing the TrustformeRS Python package.

## Quick Install (PyPI)

Once the package is published to PyPI, you can install it with:

```bash
pip install trustformers
```

## Install from Source

### Prerequisites

1. **Python 3.8 or higher**
   ```bash
   python --version
   ```

2. **Rust toolchain**
   ```bash
   # Install Rust if not already installed
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

3. **Maturin** (Rust/Python build tool)
   ```bash
   pip install maturin
   ```

### Building from Source

1. **Clone the repository**
   ```bash
   git clone https://github.com/cool-japan/trustformers.git
   cd trustformers/trustformers-py
   ```

2. **Build the package**
   ```bash
   # For development (debug mode, faster compilation)
   maturin develop
   
   # For production (release mode, optimized)
   maturin develop --release
   ```

3. **Or use the build script**
   ```bash
   ./build.sh --install
   ```

### Alternative: Using pip with git

```bash
pip install git+https://github.com/cool-japan/trustformers.git#subdirectory=trustformers-py
```

## Platform-Specific Instructions

### macOS

1. Ensure Xcode Command Line Tools are installed:
   ```bash
   xcode-select --install
   ```

2. For Apple Silicon (M1/M2), Rust will automatically build for ARM64.

### Linux

1. Install build essentials:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install build-essential
   
   # Fedora
   sudo dnf install gcc
   ```

### Windows

1. Install Visual Studio Build Tools or Visual Studio with C++ support
2. Use PowerShell or Command Prompt (not WSL) for building

## Verify Installation

```python
import trustformers
print(trustformers.__version__)

# Test basic functionality
from trustformers import Tensor
import numpy as np

tensor = Tensor(np.array([1, 2, 3], dtype=np.float32))
print(tensor.shape)
```

## Optional Dependencies

TrustformeRS provides extensive integration with the Python ML ecosystem. Install the dependencies you need for your specific use case:

### Core Framework Integration

#### PyTorch Integration
For torch.nn.Module compatibility, tensor conversions, and PyTorch training:
```bash
pip install torch>=1.9.0 torchvision torchaudio
```

#### JAX Integration  
For JAX array support, JIT compilation, and gradient transformations:
```bash
pip install jax jaxlib optax
```

#### NumPy Enhanced Features
For advanced array operations (installed by default):
```bash
pip install numpy>=1.20.0
```

### Data Science and Visualization

#### Jupyter Notebook Support
For rich display, interactive widgets, and notebook integration:
```bash
pip install jupyter ipython matplotlib seaborn
```

#### Data Science Tools
For pandas integration and scikit-learn compatibility:
```bash
pip install pandas>=1.3.0 scikit-learn>=1.0.0
```

#### Visualization Libraries
For plotting and interactive visualizations:
```bash
pip install matplotlib>=3.5.0 seaborn>=0.11.0 plotly>=5.0.0
```

### MLOps and Experiment Tracking

#### MLflow Integration
For experiment tracking and model registry:
```bash
pip install mlflow>=1.20.0
```

#### Weights & Biases
For experiment tracking and visualization:
```bash
pip install wandb>=0.12.0
```

#### TensorBoard
For logging and visualization:
```bash
pip install tensorboard>=2.8.0
```

### Distributed and High-Performance Computing

#### Horovod (Distributed Training)
For multi-GPU and multi-node training:
```bash
pip install horovod>=0.24.0
```

#### Ray (Distributed Computing)
For distributed training and hyperparameter tuning:
```bash
pip install ray[train]>=2.0.0
```

### Model Serving and Production

#### FastAPI Serving
For production model serving with REST API:
```bash
pip install fastapi>=0.95.0 uvicorn[standard]>=0.20.0
```

#### Additional Serving Dependencies
For advanced serving features:
```bash
pip install redis>=4.0.0 prometheus-client>=0.14.0
```

### Development and Testing

#### Development Tools
```bash
pip install pytest>=7.0.0 pytest-cov>=4.0.0 pytest-asyncio>=0.21.0
pip install black>=22.0.0 isort>=5.10.0 mypy>=1.0.0
```

#### Performance Profiling
```bash
pip install psutil>=5.8.0 memory-profiler>=0.60.0
```

### Mobile and Edge Deployment

#### ONNX Export
For model export and edge deployment:
```bash
pip install onnx>=1.12.0 onnxruntime>=1.12.0
```

#### TensorRT (NVIDIA GPUs)
For optimized inference on NVIDIA hardware:
```bash
pip install tensorrt>=8.0.0
```

### Installation Profiles

#### Quick Start (Minimal)
```bash
pip install trustformers
```

#### Data Science Workstation
```bash
pip install trustformers[data-science]
# Includes: pandas, scikit-learn, matplotlib, seaborn, jupyter
```

#### Machine Learning Development
```bash
pip install trustformers[ml-dev]  
# Includes: torch, jax, tensorboard, mlflow, pytest
```

#### Production Serving
```bash
pip install trustformers[serving]
# Includes: fastapi, uvicorn, redis, prometheus-client
```

#### Full Installation (All Features)
```bash
pip install trustformers[all]
# Includes all optional dependencies
```

### Compatibility Matrix

| Feature | Python 3.8 | Python 3.9 | Python 3.10 | Python 3.11 | Python 3.12 |
|---------|-------------|-------------|--------------|--------------|--------------|
| Core Features | ✅ | ✅ | ✅ | ✅ | ✅ |
| PyTorch | ✅ | ✅ | ✅ | ✅ | ✅ |
| JAX | ✅ | ✅ | ✅ | ✅ | ⚠️* |
| MLOps Tools | ✅ | ✅ | ✅ | ✅ | ✅ |
| Serving | ✅ | ✅ | ✅ | ✅ | ✅ |

*JAX support for Python 3.12 may require pre-release versions.

### Platform Support

| Platform | Architecture | Support Level |
|----------|-------------|---------------|
| Linux | x86_64 | ✅ Full |
| Linux | ARM64 | ✅ Full |
| macOS | x86_64 (Intel) | ✅ Full |
| macOS | ARM64 (Apple Silicon) | ✅ Full |
| Windows | x86_64 | ✅ Full |
| Windows | ARM64 | ⚠️ Experimental |

### Checking Installed Features

Verify which optional features are available:
```python
import trustformers
print(trustformers.available_features())

# Check specific integrations
print(f"PyTorch available: {trustformers.HAS_TORCH}")
print(f"JAX available: {trustformers.HAS_JAX}")
print(f"MLflow available: {trustformers.HAS_MLFLOW}")
```

## Troubleshooting

### Rust compilation errors

1. Update Rust:
   ```bash
   rustup update
   ```

2. Clear Cargo cache:
   ```bash
   cargo clean
   ```

### Python import errors

1. Ensure the package is installed:
   ```bash
   pip list | grep trustformers
   ```

2. Check Python path:
   ```python
   import sys
   print(sys.path)
   ```

### Memory issues during compilation

1. Limit parallel compilation:
   ```bash
   export CARGO_BUILD_JOBS=2
   maturin build --release
   ```

## Building Wheels for Distribution

### Build for current platform:
```bash
maturin build --release
```

### Build for multiple Python versions:
```bash
maturin build --release --interpreter python3.8 python3.9 python3.10 python3.11
```

### Build manylinux wheels (Linux):
```bash
docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release
```

## Publishing to PyPI

1. **Build wheels**:
   ```bash
   maturin build --release
   ```

2. **Upload to TestPyPI** (for testing):
   ```bash
   maturin publish --repository-url https://test.pypi.org/legacy/
   ```

3. **Upload to PyPI**:
   ```bash
   maturin publish
   ```

## Development Setup

For contributing to TrustformeRS:

1. **Fork and clone the repository**
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   maturin develop
   pip install -e ".[dev]"
   ```

4. **Run tests**:
   ```bash
   pytest
   cargo test
   ```

## Uninstalling

```bash
pip uninstall trustformers
```

## Getting Help

- GitHub Issues: https://github.com/cool-japan/trustformers/issues
- Documentation: https://trustformers.readthedocs.io
- Discord: [Join our community]

## License

TrustformeRS is licensed under the Apache License 2.0. See LICENSE for details.