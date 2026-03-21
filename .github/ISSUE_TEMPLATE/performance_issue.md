---
name: Performance Issue
about: Report performance problems, regressions, or optimization opportunities
title: "[PERFORMANCE] "
labels: ["performance", "needs-investigation"]
assignees: ''

---

## Performance Issue Summary
A clear description of the performance problem you're experiencing.

## Performance Metrics
Please provide specific metrics if available:

### Current Performance
- **Throughput**: [e.g. 150 tokens/sec, 50 samples/sec]
- **Latency**: [e.g. 200ms per inference, 5s per batch]
- **Memory Usage**: [e.g. 8GB GPU memory, 16GB system RAM]
- **CPU/GPU Utilization**: [e.g. 70% GPU, 40% CPU]

### Expected Performance
- **Expected Throughput**: [e.g. should be >300 tokens/sec]
- **Expected Latency**: [e.g. should be <100ms]
- **Expected Memory**: [e.g. should use <4GB]
- **Baseline**: [e.g. PyTorch achieves X, TensorFlow achieves Y]

## Environment Information
- **TrustformeRS Version**: [e.g. 0.1.0]
- **Rust Version**: [e.g. 1.75.0]
- **Operating System**: [e.g. Ubuntu 22.04, macOS 14.0]
- **Hardware**: 
  - **CPU**: [e.g. AMD Ryzen 9 7950X, Intel i9-13900K, Apple M2 Max]
  - **GPU**: [e.g. NVIDIA RTX 4090 24GB, AMD RX 7900 XTX, Apple M2 Max]
  - **RAM**: [e.g. 32GB DDR5-5600, 64GB unified memory]
- **Backend**: [e.g. CUDA 12.1, ROCm 6.0, Metal, CPU]
- **Compilation Flags**: [e.g. --release, target-cpu=native, CUDA_ARCH]

## Workload Details
Describe the specific workload that's showing performance issues:

### Model Information
- **Model Type**: [e.g. GPT-3, BERT, Llama-2-7B]
- **Model Size**: [e.g. 7B parameters, 175B parameters]
- **Quantization**: [e.g. FP16, INT8, INT4, none]
- **Sequence Length**: [e.g. 2048 tokens, 4096 tokens]

### Batch Configuration
- **Batch Size**: [e.g. 1, 16, 32]
- **Dynamic Batching**: [Yes/No]
- **Sequence Packing**: [Yes/No]

### Operation Details
- **Operation Type**: [e.g. inference, training, fine-tuning]
- **Specific Layer/Component**: [e.g. attention, FFN, embedding]

## Reproduction Steps
Provide code to reproduce the performance issue:

```rust
// Rust code example
use trustformers::prelude::*;

fn main() -> Result<()> {
    // Your reproduction code here
    let model = AutoModel::from_pretrained("model-name")?;
    // ... rest of the code
}
```

or

```python
# Python code example
import trustformers

# Your reproduction code here
```

## Benchmarking
If you've done performance comparisons, please share:

### TrustformeRS vs Other Frameworks
| Framework | Throughput | Latency | Memory | Notes |
|-----------|------------|---------|---------|-------|
| TrustformeRS | 150 tok/s | 200ms | 8GB | Current |
| PyTorch | 300 tok/s | 100ms | 6GB | Reference |
| TensorFlow | 250 tok/s | 120ms | 7GB | Reference |

### Performance Profile
If you have profiling data, please include:
- [ ] CPU profiling results (flamegraph, perf, etc.)
- [ ] GPU profiling results (nsys, nvprof, etc.)
- [ ] Memory profiling results
- [ ] System-level metrics (iostat, etc.)

## Component
Which component is affected? (check all that apply)
- [ ] trustformers-core (tensor operations)
- [ ] trustformers-models (model implementations)
- [ ] trustformers-training (training loops)
- [ ] trustformers-optim (optimizers)
- [ ] trustformers-tokenizers (tokenization)
- [ ] trustformers-serve (serving infrastructure)
- [ ] Hardware acceleration (CUDA/ROCm/Metal)
- [ ] Memory management
- [ ] Kernel fusion
- [ ] Quantization
- [ ] Other: _____________

## Issue Type
What kind of performance issue is this?
- [ ] **Regression** (performance got worse in recent version)
- [ ] **Slower than expected** (compared to other frameworks)
- [ ] **Memory inefficiency** (using too much memory)
- [ ] **Poor scaling** (doesn't scale with batch size/cores/GPUs)
- [ ] **Optimization opportunity** (could be faster with different approach)

## Impact
How does this affect your use case?
- [ ] **Blocking** (prevents usage in production)
- [ ] **Significant** (major impact on user experience)
- [ ] **Moderate** (noticeable but workable)
- [ ] **Minor** (optimization opportunity)

## Additional Information
- **Compilation optimizations tried**: [e.g. LTO, PGO, target-cpu=native]
- **Configuration changes attempted**: [e.g. different batch sizes, backends]
- **Workarounds found**: [describe any workarounds]
- **Related issues**: [link to related performance issues]

## Checklist
- [ ] I have provided specific performance metrics
- [ ] I have included environment details and hardware specifications
- [ ] I have provided a reproducible example
- [ ] I have compared with baseline/reference implementations if possible
- [ ] I have tried basic optimization flags (--release, etc.)