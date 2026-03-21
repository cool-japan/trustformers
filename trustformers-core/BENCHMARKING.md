# TrustformeRS Core Benchmarking Infrastructure

This document describes the performance benchmarking infrastructure for TrustformeRS Core.

## Overview

The benchmarking infrastructure provides comprehensive performance testing for:
- **Tensor Operations**: Core mathematical operations (add, multiply, matmul, etc.)
- **Hardware Acceleration**: CUDA, ROCm, Intel OneAPI, Vulkan backends
- **Memory Performance**: Memory allocation and usage patterns
- **Regression Detection**: Automated performance regression detection

## Quick Start

### Running All Benchmarks
```bash
# Run complete benchmark suite
./run_benchmarks.sh

# Results stored in benchmark_results/
```

### Running Individual Benchmarks
```bash
# Tensor operations only
cargo bench --bench tensor_operations

# Hardware acceleration only  
cargo bench --bench hardware_acceleration
```

## Benchmark Infrastructure

### Core Components

1. **Benchmark Suites** (`/benches/`)
   - `tensor_operations.rs` - Core tensor operation benchmarks
   - `hardware_acceleration.rs` - Hardware backend performance tests

2. **Performance Module** (`/src/performance/`)
   - `benchmark.rs` - Benchmark configuration and execution
   - `continuous.rs` - Continuous benchmarking infrastructure
   - `regression_detector.rs` - Performance regression detection
   - `memory_profiler.rs` - Memory usage profiling
   - `custom_benchmarks/` - Custom benchmark builder framework

3. **Analysis Tools**
   - `run_benchmarks.sh` - Automated benchmark execution
   - `analyze_performance_trends.sh` - Performance trend analysis
   - `benchmark_results/` - Stored benchmark results and reports

### Features

- **Automated Execution**: One-command benchmark runs with result storage
- **Performance Tracking**: Historical performance data with trend analysis
- **Regression Detection**: Statistical analysis to detect performance regressions
- **Hardware Comparison**: Compare performance across different acceleration backends
- **Memory Profiling**: Track memory usage patterns and detect leaks
- **Custom Benchmarks**: Framework for adding domain-specific benchmarks

## Usage Examples

### Basic Benchmarking
```bash
# Run all benchmarks and generate report
./run_benchmarks.sh

# Analyze performance trends  
./analyze_performance_trends.sh benchmark_results
```

### Advanced Usage
```bash
# Run specific benchmark with custom parameters
cargo bench --bench tensor_operations -- --sample-size 1000

# Compare hardware backends
cargo bench --bench hardware_acceleration --features cuda,rocm

# Profile memory usage during benchmarks
cargo bench --bench tensor_operations --features memory-profiling
```

### Performance Analysis
```bash
# View latest benchmark results
ls -la benchmark_results/

# Compare with previous results
./analyze_performance_trends.sh

# Generate performance report
./run_benchmarks.sh && cat benchmark_results/benchmark_report_*.md
```

## Performance Targets

### Tensor Operations
- **Matrix Multiplication**: >100 GFLOPS for 1024×1024 matrices
- **Element-wise Operations**: >10 GB/s memory bandwidth utilization
- **Reductions**: >5 GB/s for large tensor reductions

### Hardware Acceleration
- **CUDA Performance**: >80% of theoretical peak performance
- **Memory Transfer**: >90% of PCIe bandwidth utilization
- **Multi-GPU Scaling**: >80% efficiency for 2-4 GPUs

### Memory Performance
- **Allocation Overhead**: <1% for large tensors (>1MB)
- **Memory Fragmentation**: <5% after extended usage
- **Cache Efficiency**: >90% L1 cache hit rate for tensor operations

## Continuous Integration

### Automated Benchmarking
```bash
# CI pipeline integration
name: Performance Benchmarks
on: [push, pull_request]
jobs:
  benchmarks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Benchmarks
        run: ./run_benchmarks.sh
      - name: Analyze Performance
        run: ./analyze_performance_trends.sh
```

### Performance Regression Alerts
- Automatic detection of >10% performance regressions
- Slack/email notifications for significant performance changes
- Performance comparison in pull request comments

## Development Guidelines

### Adding New Benchmarks
1. Add benchmark functions to appropriate suite in `/benches/`
2. Use Criterion framework for statistical rigor
3. Include multiple data sizes for scalability testing
4. Document expected performance characteristics

### Performance Optimization
1. Run benchmarks before and after optimization attempts
2. Focus on bottlenecks identified by profiling
3. Verify improvements are statistically significant
4. Update performance targets if appropriate

### Best Practices
- **Reproducible Results**: Use fixed seeds for random data
- **Statistical Rigor**: Run sufficient iterations for confidence
- **System Load**: Run benchmarks on idle systems when possible
- **Version Control**: Track benchmark results alongside code changes

## Troubleshooting

### Common Issues

**Benchmark Compilation Errors**
```bash
# Missing dependencies
cargo install criterion

# Feature flag issues
cargo bench --features cuda,rocm,intel
```

**Performance Inconsistencies**
- Ensure system is idle during benchmarking
- Check for thermal throttling on CPU/GPU
- Verify consistent power management settings
- Run multiple iterations to account for variance

**Hardware Acceleration Issues**
- Verify hardware drivers are properly installed
- Check feature flags match available hardware
- Review hardware-specific logs in benchmark output

## Contributing

When contributing performance improvements:
1. Run benchmarks before your changes (baseline)
2. Implement your optimization
3. Run benchmarks again and compare results
4. Include benchmark results in your pull request
5. Update this documentation if adding new benchmarks

## Support

For benchmarking issues:
- Check existing benchmark results in `benchmark_results/`
- Review performance targets and guidelines above
- File issues with benchmark output and system information
- Include steps to reproduce performance issues