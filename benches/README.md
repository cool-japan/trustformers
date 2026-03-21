# TrustformeRS Benchmark Suite

Comprehensive benchmarks for measuring performance across all components of TrustformeRS.

## Benchmark Categories

### 1. Core Tensor Operations (`tensor_ops_bench.rs`)
- Tensor creation (zeros, ones, randn)
- Arithmetic operations (add, mul, matmul)
- Reduction operations (sum, mean, softmax)
- Manipulation operations (transpose, reshape, slice)
- GPU operations (when available)

### 2. Model Inference (`model_inference_bench.rs`)
- BERT inference (base, large)
- GPT-2 inference (base, medium)
- LLaMA inference (7B, 13B configurations)
- Text generation benchmarks
- Batch inference scaling

### 3. Optimizers (`optimizer_bench.rs`)
- Adam/AdamW optimizer performance
- SGD with various configurations
- Learning rate schedulers
- Distributed optimizers (ZeRO stages)
- Memory efficiency comparisons

### 4. Tokenizers (`tokenizer_bench.rs`)
- BPE tokenizer (GPT-2 style)
- WordPiece tokenizer (BERT style)
- SentencePiece tokenizer
- Batch tokenization
- Text normalization
- Special token handling

### 5. Quantization (`quantization_bench.rs`)
- Tensor quantization (INT8, INT4, FP16, BF16)
- Quantized operations (matmul, linear layers)
- Model quantization
- Dynamic quantization
- Mixed precision operations

### 6. Memory Usage (`memory_bench.rs`)
- Tensor memory allocation
- Model memory footprint
- Cache memory usage
- Memory pool efficiency
- Gradient accumulation memory
- Activation checkpointing

### 7. Mobile & WebAssembly (`mobile_wasm_bench.rs`)
- Mobile optimization levels
- Mobile quantization
- Device adaptation
- Thermal throttling simulation
- WASM tensor operations
- WASM backends (CPU, SIMD, WebGPU)
- WASM memory patterns

## Running Benchmarks

### Run all benchmarks:
```bash
cargo bench
```

### Run specific benchmark suite:
```bash
cargo bench --bench tensor_ops_bench
cargo bench --bench model_inference_bench
cargo bench --bench optimizer_bench
cargo bench --bench tokenizer_bench
cargo bench --bench quantization_bench
cargo bench --bench memory_bench
cargo bench --bench mobile_wasm_bench
```

### Run with specific features:
```bash
# GPU benchmarks
cargo bench --features gpu

# Distributed benchmarks
cargo bench --features distributed

# Mobile benchmarks
cargo bench --features mobile

# WebAssembly benchmarks
cargo bench --features wasm

# All features
cargo bench --all-features
```

### Run specific benchmark by name:
```bash
# Run only BERT inference benchmarks
cargo bench bert_inference

# Run only quantization benchmarks
cargo bench quantization

# Use regex to match benchmark names
cargo bench "tensor.*matmul"
```

## Benchmark Configuration

### Customize benchmark runs:
```bash
# Increase sample size for more accurate results
cargo bench -- --sample-size 100

# Save benchmark results
cargo bench -- --save-baseline my_baseline

# Compare against baseline
cargo bench -- --baseline my_baseline

# Output format options
cargo bench -- --output-format bencher
cargo bench -- --output-format json
```

### Performance profiling:
```bash
# Run with CPU profiler
cargo bench --bench model_inference_bench -- --profile-time 10

# Generate flamegraph
cargo flamegraph --bench model_inference_bench -- --bench
```

## Interpreting Results

### Throughput Metrics
- **Elements/sec**: Number of tensor elements processed per second
- **Bytes/sec**: Data throughput for memory operations
- **Items/sec**: For tokenization and batch operations

### Time Metrics
- **Time per iteration**: Average time for single operation
- **Min/Max times**: Best and worst case performance
- **Standard deviation**: Consistency of performance

### Memory Metrics
- **Bytes allocated**: Total memory used
- **Peak memory**: Maximum memory usage
- **Allocation count**: Number of allocations

## Continuous Benchmarking

### GitHub Actions Integration
Benchmarks run automatically on:
- Every push to main branch
- Pull requests (with comparison)
- Nightly scheduled runs

### Performance Regression Detection
- Automatic alerts for >5% performance regression
- Detailed comparison reports in PR comments
- Historical performance tracking

## Writing New Benchmarks

### Template for new benchmark:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};

fn my_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_benchmark_group");
    
    // Set throughput if applicable
    group.throughput(Throughput::Elements(1000));
    
    // Benchmark with different inputs
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("operation", size),
            size,
            |b, size| {
                b.iter(|| {
                    // Your code here
                    let result = expensive_operation(*size);
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, my_benchmark);
criterion_main!(benches);
```

### Best Practices
1. Use `black_box` to prevent compiler optimizations
2. Group related benchmarks together
3. Set appropriate throughput metrics
4. Use meaningful benchmark IDs
5. Test multiple input sizes/configurations
6. Warm up caches before benchmarking
7. Account for setup/teardown costs

## Benchmark Results

Results are saved in `target/criterion/` with:
- HTML reports: `target/criterion/report/index.html`
- Raw data: `target/criterion/*/base/`
- Comparison data: `target/criterion/*/change/`

## Platform-Specific Notes

### Linux
- Disable CPU frequency scaling for consistent results
- Use `taskset` to pin benchmarks to specific CPUs
- Consider disabling hyperthreading

### macOS
- Disable Power Nap and App Nap
- Use Energy Saver settings for consistent performance
- Close unnecessary applications

### Windows
- Set power plan to High Performance
- Disable Windows Defender real-time scanning
- Close background applications

## Troubleshooting

### Inconsistent results
- Increase sample size: `--sample-size 200`
- Increase measurement time: `--measurement-time 10`
- Check for background processes
- Verify thermal throttling isn't occurring

### Out of memory
- Reduce batch sizes in benchmarks
- Run memory-intensive benchmarks separately
- Use `--features small_bench` for reduced memory usage

### Compilation errors
- Ensure all feature flags are properly configured
- Check that optional dependencies are installed
- Verify CUDA/ROCm setup for GPU benchmarks