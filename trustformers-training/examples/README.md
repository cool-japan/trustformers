# TrustformeRS Training Examples

This directory contains comprehensive examples demonstrating how to use the advanced training infrastructure provided by TrustformeRS. The examples are organized by complexity and use case to help you get started quickly and learn about advanced features.

## 📁 Example Categories

### 🚀 Basic Training (`basic_training/`)
- **Simple Classification**: Basic model training with standard features
- **Text Generation**: Language model training with generation capabilities
- **Multi-task Learning**: Training models on multiple tasks simultaneously
- **Custom Loss Functions**: Implementing and using custom loss functions

### 🔬 Advanced Features (`advanced_features/`)
- **Adaptive Gradient Scaling**: Using automatic gradient scaling for stability
- **Advanced Learning Rate Scheduling**: Multi-strategy adaptive learning rate management
- **Ring Attention**: Ultra-long sequence training with distributed attention
- **Mixed Precision Training**: Advanced mixed precision with dynamic scaling
- **Quantization-Aware Training**: Training with quantization for deployment efficiency

### 🌐 Distributed Training (`distributed_training/`)
- **Data Parallelism**: Simple data parallel training across multiple devices
- **Model Parallelism**: Large model training with model sharding
- **Pipeline Parallelism**: Efficient pipeline parallelism for throughput
- **3D Parallelism**: Combined data, model, and pipeline parallelism
- **Multi-Cloud Training**: Training across multiple cloud providers
- **Elastic Training**: Dynamic scaling based on resource availability

### 🎛️ Hyperparameter Optimization (`hyperparameter_optimization/`)
- **Bayesian Optimization**: Gaussian Process-based hyperparameter search
- **Population-Based Training**: Dynamic hyperparameter adjustment during training
- **Neural Architecture Search**: Automated architecture optimization
- **Multi-Objective Optimization**: Optimizing for multiple criteria simultaneously
- **Transfer Learning HPO**: Using historical data to accelerate optimization

### 🎯 RLHF Training (`rlhf_training/`)
- **Supervised Fine-Tuning**: Initial model fine-tuning with supervised data
- **Reward Model Training**: Training preference-based reward models
- **PPO Training**: Proximal Policy Optimization for RLHF
- **DPO Training**: Direct Preference Optimization without reward models
- **Constitutional AI**: Principle-based training for safe AI systems

### 📊 Benchmarks (`benchmarks/`)
- **Performance Comparison**: Comparing different training strategies
- **Scaling Analysis**: Understanding performance at different scales
- **Memory Efficiency**: Measuring memory usage of different techniques
- **Convergence Analysis**: Analyzing convergence properties of optimizers
- **Hardware Utilization**: GPU/CPU utilization analysis

## 🏃 Quick Start

### Prerequisites
```bash
# Install TrustformeRS training infrastructure
cargo add trustformers-training

# Install additional dependencies for examples
cargo add tokio --features full
cargo add anyhow
cargo add serde --features derive
```

### Running Examples

Each example includes a README with specific instructions. Generally:

```bash
# Navigate to an example directory
cd examples/basic_training/simple_classification

# Run the example
cargo run --example simple_classification

# Or with specific configuration
cargo run --example simple_classification -- --config config.json
```

### Configuration Files

Most examples include configuration files (`config.json` or `config.toml`) that demonstrate different parameter settings. You can modify these to experiment with different configurations.

## 📚 Example Descriptions

### Basic Training Examples

#### `simple_classification/`
Demonstrates basic supervised learning with:
- Standard SGD and Adam optimizers
- Early stopping and checkpointing
- Basic metrics tracking
- Simple evaluation loops

#### `text_generation/`
Shows language model training with:
- Causal language modeling loss
- Text generation during evaluation
- Perplexity calculation
- Generation quality metrics

#### `multi_task_learning/`
Multi-task learning setup with:
- Shared encoder, task-specific heads
- Task balancing strategies
- Cross-task evaluation
- Task interference analysis

#### `custom_loss_functions/`
Custom loss implementation example:
- Focal loss for imbalanced classification
- Contrastive loss for representation learning
- Custom regularization terms
- Loss combination strategies

### Advanced Features Examples

#### `adaptive_gradient_scaling/`
Advanced gradient management with:
- Automatic gradient norm scaling
- Per-layer gradient statistics
- Outlier detection and filtering
- Stability monitoring and recovery

#### `ring_attention/`
Ultra-long sequence training with:
- Ring attention for sequences up to 100M tokens
- Distributed attention computation
- Memory pool optimization
- Communication pattern analysis

#### `mixed_precision_training/`
Advanced mixed precision with:
- Dynamic precision switching
- Per-layer scaling strategies
- Loss scaling optimization
- Numerical stability monitoring

### Distributed Training Examples

#### `data_parallelism/`
Standard distributed training with:
- Multi-GPU data parallelism
- Gradient synchronization
- Load balancing
- Fault tolerance

#### `3d_parallelism/`
Advanced parallelism combining:
- Data parallelism across nodes
- Model parallelism within nodes
- Pipeline parallelism for throughput
- Communication optimization

#### `elastic_training/`
Dynamic resource management with:
- Worker scaling based on availability
- Automatic fault detection and recovery
- Checkpoint-restart optimization
- Resource allocation strategies

### RLHF Examples

#### `supervised_fine_tuning/`
Initial fine-tuning phase with:
- Instruction-response pair training
- Quality-based data filtering
- Learning rate scheduling
- Evaluation on held-out data

#### `ppo_training/`
Reinforcement learning with:
- Advantage calculation (GAE)
- Policy and value loss computation
- KL divergence regularization
- Training stability monitoring

#### `constitutional_ai/`
Principle-based training with:
- Multi-principle violation detection
- Weighted loss based on principles
- Content analysis for safety
- Iterative refinement

## 🛠️ Customization Guide

### Adding Custom Models
```rust
use trustformers_training::trainer::Trainer;
use trustformers_core::Model;

// Your custom model
struct MyModel { /* ... */ }

impl Model for MyModel {
    // Implement required methods
}

// Use with trainer
let trainer = Trainer::new(model, config)?;
```

### Custom Optimizers
```rust
use trustformers_training::optimization::Optimizer;

struct MyOptimizer { /* ... */ }

impl Optimizer for MyOptimizer {
    // Implement optimization logic
}
```

### Custom Metrics
```rust
use trustformers_training::metrics::{Metric, MetricResult};

struct MyMetric;

impl Metric for MyMetric {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> MetricResult {
        // Custom metric computation
    }
}
```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory**: Try reducing batch size, enabling gradient checkpointing, or using CPU offloading
2. **Slow Convergence**: Check learning rate, gradient scaling, or try different optimizers
3. **Numerical Instability**: Enable mixed precision, gradient clipping, or anomaly detection
4. **Distributed Training Issues**: Verify network configuration, check for hanging processes

### Performance Optimization

1. **Enable Compilation Optimizations**:
   ```bash
   cargo build --release
   ```

2. **Use Appropriate Batch Sizes**: Experiment with batch size for your hardware
3. **Enable Hardware Features**: Use SIMD, CUDA, or other acceleration when available
4. **Monitor Resource Usage**: Use built-in profiling tools to identify bottlenecks

## 📖 Additional Resources

- **Training Guide**: `../HYPERPARAMETER_TUNING.md` - Comprehensive hyperparameter tuning guide
- **API Documentation**: Run `cargo doc --open` for detailed API documentation
- **Performance Tips**: See individual example READMEs for performance optimization tips
- **Troubleshooting**: Check the main README for common issues and solutions

## 🤝 Contributing

Have ideas for new examples or improvements? Please contribute!

1. Fork the repository
2. Create a new example in the appropriate category
3. Include comprehensive documentation
4. Add tests and configuration files
5. Submit a pull request

## 📄 License

These examples are part of the TrustformeRS project and are provided under the same license terms.