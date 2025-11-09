# Contributing to TrustformeRS

Thank you for your interest in contributing to TrustformeRS! This guide will help you get started.

## 🚀 Getting Started

### Prerequisites

- Rust 1.75 or later
- Git
- (Optional) CUDA toolkit for GPU development
- (Optional) Python 3.8+ for comparison testing

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/trustformers.git
   cd trustformers
   ```

2. **Install development dependencies**
   ```bash
   cargo install cargo-nextest  # Fast test runner
   cargo install cargo-flamegraph  # Performance profiling
   cargo install cargo-criterion  # Benchmarking
   ```

3. **Build the project**
   ```bash
   cargo build --all-features
   ```

4. **Run tests**
   ```bash
   cargo nextest run --all-features --no-fail-fast
   ```
   
   **Important**: We maintain a strict **no-warnings policy**. Run tests continuously until all warnings are resolved:
   ```bash
   cargo nextest run --no-fail-fast && cargo check --all-features
   ```

## 📁 Project Structure

```
trustformers/
├── trustformers-core/       # Core traits and abstractions
├── trustformers-models/     # Model implementations
├── trustformers-tokenizers/ # Tokenizer implementations
├── trustformers-optim/      # Optimizers
└── trustformers/            # Main integration crate
```

## 🏗️ Adding a New Model

### 1. Create Model Structure

Create a new module in `trustformers-models/src/`:

```
trustformers-models/src/
└── your_model/
    ├── mod.rs        # Module exports
    ├── config.rs     # Configuration struct
    ├── model.rs      # Model implementation
    └── layers.rs     # Model-specific layers (if any)
```

### 2. Implement Configuration

```rust
// config.rs
use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YourModelConfig {
    // Model hyperparameters
}

impl Config for YourModelConfig {
    fn validate(&self) -> Result<()> {
        // Validation logic
    }
    
    fn architecture(&self) -> &'static str {
        "YourModel"
    }
}
```

### 3. Implement Model

```rust
// model.rs
use trustformers_core::{Model, Result};

pub struct YourModel {
    config: YourModelConfig,
    // Model components
}

impl Model for YourModel {
    type Config = YourModelConfig;
    type Input = TokenizedInput;
    type Output = YourModelOutput;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Forward pass implementation
    }
}
```

### 4. Add Tests

Create parity tests comparing with Hugging Face:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_forward_pass_parity() {
        // Load same weights
        // Run forward pass
        // Compare outputs with HF
    }
}
```

### 5. Update Feature Flags

Add your model to `trustformers-models/Cargo.toml`:

```toml
[features]
your_model = []
all = ["bert", "gpt2", "t5", "your_model"]
```

## 🧪 Testing Guidelines

### Unit Tests
- Test individual components (layers, attention, etc.)
- Use `approx` for floating-point comparisons
- Test edge cases (empty inputs, max length, etc.)

### Integration Tests
- Test full model forward pass
- Compare with reference implementations
- Test serialization/deserialization

### Performance Tests
- Add benchmarks for critical operations
- Compare with baseline implementations
- Profile memory usage

Example benchmark:

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_attention(c: &mut Criterion) {
    c.bench_function("self_attention", |b| {
        b.iter(|| {
            // Benchmark code
        });
    });
}

criterion_group!(benches, bench_attention);
criterion_main!(benches);
```

## 🏎️ Performance Optimization

### Profiling

1. **CPU Profiling**
   ```bash
   cargo flamegraph --bin your_benchmark
   ```

2. **Memory Profiling**
   ```bash
   valgrind --tool=massif target/release/your_benchmark
   ```

### Optimization Checklist

- [ ] Profile first, optimize second
- [ ] Use SIMD where applicable
- [ ] Minimize allocations in hot paths
- [ ] Consider cache-friendly data layouts
- [ ] Parallelize independent operations
- [ ] Benchmark before and after

### SIMD Example

```rust
// Use SciRS2 for SIMD operations
use scirs2_core::simd_ops::SimdUnifiedOps;

fn gelu_optimized(tensor: &Tensor) -> Result<Tensor> {
    // SciRS2 automatically handles SIMD optimization
    tensor.simd_gelu()
}
```

## 🧹 Code Quality Standards

### Continuous Testing

We use `cargo nextest` for faster test execution. Always run with `--no-fail-fast`:

```bash
# Run all tests continuously until no failures
cargo nextest run --no-fail-fast

# Run specific module tests
cargo nextest run -p trustformers-core --no-fail-fast
```

### File Organization

- Keep files under 2000 lines - refactor if larger
- Use module hierarchy for organization
- Group related functionality together
- Separate tests into dedicated test modules

## 📝 Code Style

### Rust Guidelines

- Follow standard Rust naming conventions
- Use `clippy` and `rustfmt`
- Document public APIs with comprehensive rustdoc
- Prefer explicit types for clarity
- Use `Result` for fallible operations
- **No Warnings Policy**: Code must compile without any warnings
- **Refactoring Policy**: Keep single files under 2000 lines
- **Latest Crates Policy**: Always use the latest versions available on crates.io

### Documentation

- Add doc comments to all public items
- Include examples in doc comments
- Update README for significant changes
- Add inline comments for complex logic

Example:

```rust
/// Applies layer normalization to the input tensor.
///
/// # Arguments
/// * `input` - Input tensor of shape [batch, seq_len, hidden]
/// * `eps` - Small value to prevent division by zero
///
/// # Example
/// ```
/// let normalized = layer_norm(&input, 1e-5)?;
/// ```
pub fn layer_norm(input: &Tensor, eps: f32) -> Result<Tensor> {
    // Implementation
}
```

## 🔄 Pull Request Process

1. **Before submitting:**
   - Run `cargo fmt`
   - Run `cargo clippy -- -D warnings`
   - Run `cargo nextest run --all-features --no-fail-fast`
   - Ensure **zero warnings** in compilation
   - Update documentation (including rustdoc)
   - Add tests for new features
   - Update CHANGELOG.md
   - Update TODO.md if applicable

2. **PR Description:**
   - Describe what changes you made
   - Link related issues
   - Include benchmark results if relevant
   - List any breaking changes

3. **Review process:**
   - CI must pass
   - At least one maintainer review
   - Address feedback promptly
   - Squash commits if requested

## 🐛 Reporting Issues

### Bug Reports

Include:
- Rust version (`rustc --version`)
- Minimal reproduction code
- Expected vs actual behavior
- Error messages/stack traces

### Feature Requests

Include:
- Use case description
- Proposed API
- Alternative solutions considered
- Impact on existing code

## 💡 Design Principles

1. **Performance First**: Optimize for speed without sacrificing safety
2. **Explicit Control**: Users decide resource usage
3. **Compatibility**: Maintain HF format compatibility
4. **Modularity**: Keep components loosely coupled
5. **Safety**: Leverage Rust's type system
6. **No Warnings**: Maintain clean, warning-free code
7. **SciRS2 Integration**: Use SciRS2 for SIMD operations and parallelism

## 🔧 SciRS2 Integration Guidelines

When working with tensor operations:

1. **Use SciRS2 SIMD ops**: Replace basic operations with `simd_add`, `simd_mul`, etc.
2. **Parallel operations**: Use `scirs2_core::parallel_ops` instead of direct rayon
3. **BLAS operations**: Integrate through SciRS2's BLAS abstractions
4. **GPU support**: Use SciRS2's GPU context management when available

Example:
```rust
// Instead of:
let result = a + b;

// Use:
let result = a.simd_add(&b)?;
```

## 🤝 Community

- Join discussions in GitHub Issues
- Propose RFCs for major changes
- Help review pull requests
- Improve documentation
- Share benchmarks and use cases

## 📄 License

By contributing, you agree that your contributions will be licensed under the same terms as the project (MIT OR Apache-2.0).

## 🙏 Thank You!

Your contributions make TrustformeRS better for everyone. Whether it's fixing a typo, adding a feature, or implementing a new model, every contribution is valued!

---

For questions not covered here, please open an issue or reach out to the maintainers.