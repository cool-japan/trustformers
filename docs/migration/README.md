# Migration Guides

This directory contains comprehensive migration guides to help you transition from popular machine learning frameworks to TrustformeRS.

## Available Migration Guides

### [PyTorch to TrustformeRS](pytorch_to_trustformers.md)
Complete guide for migrating PyTorch-based projects to TrustformeRS, covering:
- Tensor operations and device management
- Model loading and inference pipelines
- Training loops and optimization
- Performance optimizations
- Common patterns and troubleshooting

**Best for**: Projects using PyTorch, transformers library, or imperative-style ML code.

### [TensorFlow to TrustformeRS](tensorflow_to_trustformers.md)
Comprehensive guide for transitioning from TensorFlow/Keras to TrustformeRS, including:
- Graph execution and eager mode patterns
- Keras model conversion
- Custom layers and metrics
- TensorFlow Serving compatibility
- Distribution strategies

**Best for**: Projects using TensorFlow, Keras, or production ML systems with TF Serving.

### [JAX to TrustformeRS](jax_to_trustformers.md)
Detailed guide for migrating JAX-based research code to TrustformeRS, covering:
- Functional programming patterns
- JIT compilation and transformations
- Automatic differentiation
- Vectorization and parallelization
- Pure function paradigms

**Best for**: Research projects using JAX, Flax, Haiku, or functional ML programming.

## Quick Start Migration Path

1. **Choose Your Framework**: Select the migration guide that matches your current framework
2. **Start Small**: Begin with inference-only migration to validate functionality
3. **Test Thoroughly**: Compare outputs between your original framework and TrustformeRS
4. **Optimize Gradually**: Enable TrustformeRS-specific optimizations
5. **Monitor Performance**: Use built-in profiling tools to measure improvements

## Common Migration Benefits

| Benefit | Description |
|---------|-------------|
| **Memory Safety** | Compile-time guarantees prevent memory errors and data races |
| **Performance** | 1.5-3x faster inference, 20-40% lower memory usage |
| **Deployment** | Single binary deployment, no Python runtime dependencies |
| **Maintainability** | Strong type system catches errors at compile time |
| **Ecosystem** | Growing ecosystem of Rust ML tools and libraries |

## Framework Comparison

| Feature | PyTorch | TensorFlow | JAX | TrustformeRS |
|---------|---------|------------|-----|--------------|
| **Programming Style** | Imperative | Mixed | Functional | Multi-paradigm |
| **Type Safety** | Runtime | Runtime | Runtime | Compile-time |
| **Memory Management** | GC | GC | GC | Zero-cost |
| **JIT Compilation** | TorchScript | XLA | XLA | Native LLVM |
| **Deployment** | Python + deps | Python/C++ | Python + deps | Single binary |
| **Performance** | Good | Good | Excellent | Excellent+ |

## Migration Strategy

### Phase 1: Assessment
- Identify framework-specific patterns in your codebase
- List dependencies and custom components
- Plan migration priorities (inference vs training)

### Phase 2: Core Migration
- Migrate tensor operations and basic computations
- Convert model architectures
- Update data loading and preprocessing

### Phase 3: Training Migration
- Migrate training loops and optimizers
- Convert custom loss functions and metrics
- Update checkpointing and logging

### Phase 4: Optimization
- Enable TrustformeRS-specific optimizations
- Profile and tune performance
- Add monitoring and debugging tools

### Phase 5: Production
- Test in staging environments
- Monitor performance metrics
- Deploy to production

## Common Patterns Across Frameworks

### Tensor Operations
All frameworks share similar tensor operation patterns that map well to TrustformeRS:

```rust
// Universal tensor operations in TrustformeRS
use trustformers::functional as F;

let x = Tensor::randn(&[10, 20])?;
let y = Tensor::ones(&[10, 20])?;

// Element-wise operations
let sum = F::add(&x, &y)?;
let product = F::mul(&x, &y)?;

// Linear algebra
let matmul = F::matmul(&x, &y.transpose(0, 1)?)?;

// Reductions
let mean = F::mean(&x, 0)?;
let sum = F::sum(&x, &[0, 1])?;
```

### Model Definition
TrustformeRS supports both imperative and functional model definition styles:

```rust
// Imperative style (PyTorch-like)
use trustformers::{Module, Linear, ReLU};

pub struct MLP {
    layer1: Linear,
    layer2: Linear,
    relu: ReLU,
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.relu.forward(&self.layer1.forward(x)?)?;
        self.layer2.forward(&h)
    }
}

// Functional style (JAX-like)
fn mlp_function(x: &Tensor, params: &MLPParams) -> Result<Tensor> {
    let h = F::relu(&F::linear(x, &params.w1, &params.b1)?)?;
    F::linear(&h, &params.w2, &params.b2)
}
```

### Training Loops
Standard training patterns work across all migration paths:

```rust
use trustformers::{Adam, CrossEntropyLoss};

let mut optimizer = Adam::new(model.parameters(), 0.001)?;
let loss_fn = CrossEntropyLoss::new();

for epoch in 0..epochs {
    for (inputs, targets) in dataloader {
        optimizer.zero_grad()?;
        
        let outputs = model.forward(&inputs)?;
        let loss = loss_fn.forward(&outputs, &targets)?;
        
        loss.backward()?;
        optimizer.step()?;
        
        if step % 100 == 0 {
            println!("Epoch {}, Loss: {}", epoch, loss.item());
        }
    }
}
```

## Framework-Specific Considerations

### From PyTorch
- **Strengths**: Similar imperative style, dynamic graphs
- **Challenges**: Python-specific features, dynamic typing
- **Focus**: Model architecture conversion, training loops

### From TensorFlow
- **Strengths**: Production deployment experience
- **Challenges**: Graph/eager execution differences, Keras abstractions
- **Focus**: Model serving patterns, distribution strategies

### From JAX
- **Strengths**: Functional programming, transformations
- **Challenges**: Pure functions, PRNG state management
- **Focus**: Function composition, vectorization

## Getting Help

- **Documentation**: Comprehensive API documentation and tutorials
- **Examples**: Real-world migration examples in `examples/` directory
- **Community**: Active Discord and GitHub discussions
- **Support**: Professional migration consulting available

## Contributing

We welcome contributions to improve these migration guides:

1. **Report Issues**: Found an error or unclear section? Open an issue
2. **Add Examples**: Share your migration experience with examples
3. **Improve Guides**: Submit PRs to enhance existing guides
4. **New Frameworks**: Help us add guides for other frameworks

## Next Steps

1. Choose the migration guide that matches your current framework
2. Follow the step-by-step instructions
3. Test your migrated code thoroughly
4. Share your experience with the community
5. Enjoy the benefits of Rust's performance and safety!

For questions or support, visit our [community forum](https://github.com/trustformers/trustformers/discussions) or join our [Discord server](https://discord.gg/trustformers).