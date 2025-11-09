# TrustformeRS Interactive Examples

Welcome to the TrustformeRS interactive examples! These examples demonstrate the comprehensive capabilities of the TrustformeRS machine learning library, showcasing everything from basic pipeline usage to advanced ensemble learning and optimization techniques.

## 📚 Overview

TrustformeRS is a high-performance machine learning library written in Rust, providing state-of-the-art transformer models with excellent performance and memory efficiency. These examples will help you understand and utilize the library's full potential.

## 🚀 Getting Started

### Prerequisites

- Rust 1.70 or later
- Cargo package manager
- At least 4GB of available memory for running examples

### Running Examples

To run any example, use the following command:

```bash
cargo run --example <example_name>
```

For example:
```bash
cargo run --example basic_pipeline
```

## 📖 Example Descriptions

### 1. Basic Pipeline (`basic_pipeline.rs`)

**Purpose**: Introduction to fundamental TrustformeRS concepts
**Demonstrates**: 
- Text classification pipeline
- Text generation pipeline
- Question answering pipeline
- Batch processing
- Performance comparison

**Key Features**:
- Simple pipeline creation and usage
- Multiple input processing
- Basic performance benchmarking
- Error handling patterns

**Run with**: `cargo run --example basic_pipeline`

### 2. Advanced Composition (`advanced_composition.rs`)

**Purpose**: Advanced pipeline composition and chaining
**Demonstrates**:
- Pipeline chaining for complex workflows
- Multi-modal pipeline processing
- Ensemble pipeline creation
- Conversational pipeline usage
- Custom pipeline composition

**Key Features**:
- Complex workflow orchestration
- Multi-step processing chains
- Custom analysis pipelines
- Content analysis workflows

**Run with**: `cargo run --example advanced_composition`

### 3. JIT Optimization (`jit_optimization.rs`)

**Purpose**: Just-in-time compilation and optimization
**Demonstrates**:
- JIT compilation configuration
- Kernel fusion techniques
- Performance optimization strategies
- Adaptive compilation
- Hardware-specific optimizations

**Key Features**:
- Automatic kernel fusion
- Performance monitoring
- Optimization recommendations
- Hardware acceleration
- Real-time performance tracking

**Run with**: `cargo run --example jit_optimization`

### 4. Dynamic Batching (`dynamic_batching.rs`)

**Purpose**: Dynamic batching and performance optimization
**Demonstrates**:
- Dynamic batch sizing
- Adaptive batching strategies
- Advanced caching mechanisms
- Memory pool optimization
- Load testing capabilities

**Key Features**:
- Intelligent batch optimization
- Memory management
- Performance profiling
- Concurrent processing
- Resource utilization optimization

**Run with**: `cargo run --example dynamic_batching`

### 5. Conversational AI (`conversational_ai.rs`)

**Purpose**: Advanced conversational AI capabilities
**Demonstrates**:
- Multi-turn dialogue systems
- Memory and context management
- Persona-based conversations
- Safety and content filtering
- Interactive chat capabilities

**Key Features**:
- Context-aware responses
- Personality customization
- Safety filtering
- Memory persistence
- Conversation analytics

**Run with**: `cargo run --example conversational_ai`

### 6. Ensemble Models (`ensemble_models.rs`)

**Purpose**: Ensemble learning and model combination
**Demonstrates**:
- Multiple voting strategies
- Dynamic model selection
- Cascade ensemble processing
- Performance comparison
- Adaptive ensemble learning

**Key Features**:
- Model combination strategies
- Performance optimization
- Intelligent model routing
- Adaptive weight adjustment
- Resource-efficient processing

**Run with**: `cargo run --example ensemble_models`

### 7. Interactive CLI Demo (`interactive_cli.rs`) 🆕

**Purpose**: Interactive command-line interface for exploring TrustformeRS
**Demonstrates**:
- Real-time model interaction
- Task and model selection
- Configuration management
- Performance testing
- Session management

**Key Features**:
- User-friendly CLI interface
- Multiple task support
- Model comparison tools
- Batch processing demos
- Streaming demonstrations
- Performance benchmarking
- Session save/load

**Run with**: `cargo run --example interactive_cli`

### 8. Web-based Interactive Demo (`web_demo.rs`) 🆕

**Purpose**: Browser-based interactive demonstration
**Demonstrates**:
- Web interface for TrustformeRS
- Real-time inference through browser
- Model selection and configuration
- Interactive examples
- Performance visualization

**Key Features**:
- Modern web interface
- Real-time processing
- Interactive model selection
- Visual performance metrics
- Example templates
- Session management
- Mobile-responsive design

**Run with**: `cargo run --example web_demo` then visit `http://localhost:3000`

### 9. Real-time Streaming Demo (`realtime_streaming.rs`) 🆕

**Purpose**: Real-time stream processing capabilities
**Demonstrates**:
- Continuous text stream processing
- Backpressure handling
- Batch optimization
- Real-time analytics
- Concurrent processing

**Key Features**:
- Stream message processing
- Dynamic batching
- Performance monitoring
- Interactive and automated modes
- Real-time analytics dashboard
- Configurable processing rates
- Memory management

**Run with**: `cargo run --example realtime_streaming -- --rate 50 --duration 30`
**Interactive mode**: `cargo run --example realtime_streaming -- --interactive`

### 10. Jupyter Notebook Tutorial (`trustformers_tutorial.ipynb`) 🆕

**Purpose**: Comprehensive interactive tutorial for data scientists
**Demonstrates**:
- Step-by-step TrustformeRS learning
- Jupyter notebook integration
- Data science workflows
- Visualization examples
- Best practices

**Key Features**:
- Interactive code examples
- Visual explanations
- Performance comparisons
- Real-world applications
- Best practices guide
- Comprehensive documentation

**Run with**: Open in Jupyter Lab/Notebook or Google Colab

## 🎯 Learning Path

We recommend following this learning path for the best experience:

### For Beginners 🌱
1. **Interactive CLI Demo** - Get familiar with TrustformeRS through an interactive interface
2. **Basic Pipeline** - Understand fundamental concepts
3. **Jupyter Notebook Tutorial** - Follow the comprehensive tutorial step-by-step

### For Intermediate Users 🚀
4. **Advanced Composition** - Learn complex workflow patterns
5. **Dynamic Batching** - Master batch processing and caching
6. **JIT Optimization** - Discover performance optimization
7. **Web-based Demo** - Explore browser-based interfaces

### For Advanced Users ⚡
8. **Real-time Streaming** - Master stream processing and analytics
9. **Conversational AI** - Experience advanced dialogue systems
10. **Ensemble Models** - Learn advanced model combination

### Alternative Learning Paths 🎲

**🖥️ Command Line Focused**:
Interactive CLI → Basic Pipeline → Advanced Composition → Real-time Streaming

**🌐 Web Development Focused**:
Web Demo → Jupyter Tutorial → Advanced Composition → JIT Optimization

**📊 Data Science Focused**:
Jupyter Tutorial → Basic Pipeline → Dynamic Batching → Ensemble Models

**🏭 Production Focused**:
Basic Pipeline → JIT Optimization → Real-time Streaming → Advanced Composition

## 🔧 Configuration Options

### Environment Variables

Set these environment variables to customize example behavior:

```bash
# Enable debug logging
export TRUSTFORMERS_LOG_LEVEL=debug

# Set model cache directory
export TRUSTFORMERS_CACHE_DIR=/path/to/cache

# Configure GPU usage
export TRUSTFORMERS_DEVICE=gpu  # or 'cpu'

# Set batch size limits
export TRUSTFORMERS_MAX_BATCH_SIZE=32
```

### Example Customization

Most examples support command-line arguments:

```bash
# Run with specific model
cargo run --example basic_pipeline -- --model distilbert-base-uncased

# Enable profiling
cargo run --example jit_optimization -- --profile

# Set batch size
cargo run --example dynamic_batching -- --batch-size 16

# Interactive mode
cargo run --example conversational_ai -- --interactive
```

## 📊 Performance Expectations

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB | 16GB+ |
| CPU | 4 cores | 8+ cores |
| Storage | 2GB | 10GB+ |
| GPU | Optional | NVIDIA RTX/Tesla |

### Expected Performance

| Example | Typical Runtime | Memory Usage | Notes |
|---------|----------------|--------------|-------|
| Basic Pipeline | 30-60 seconds | 1-2GB | Model download time |
| Advanced Composition | 1-2 minutes | 2-4GB | Multiple models |
| JIT Optimization | 1-3 minutes | 2-3GB | Compilation overhead |
| Dynamic Batching | 2-4 minutes | 3-5GB | Stress testing |
| Conversational AI | 1-2 minutes | 2-3GB | Interactive features |
| Ensemble Models | 2-5 minutes | 4-8GB | Multiple models |

## 🛠️ Troubleshooting

### Common Issues

**Out of Memory Error**
```bash
# Solution: Reduce batch size or use smaller models
export TRUSTFORMERS_MAX_BATCH_SIZE=8
```

**Model Download Fails**
```bash
# Solution: Check internet connection and cache directory
export TRUSTFORMERS_CACHE_DIR=/tmp/trustformers
```

**GPU Not Detected**
```bash
# Solution: Ensure CUDA drivers are installed
nvidia-smi  # Check GPU status
```

**Slow Performance**
```bash
# Solution: Enable optimizations
export TRUSTFORMERS_OPTIMIZE=true
export TRUSTFORMERS_JIT=true
```

### Debug Mode

Enable debug logging for detailed information:

```bash
export RUST_LOG=trustformers=debug
cargo run --example basic_pipeline
```

## 🧪 Advanced Usage

### Custom Model Integration

```rust
use trustformers::{pipeline, AutoModel, AutoTokenizer};

// Load custom model
let model = AutoModel::from_pretrained("path/to/your/model")?;
let tokenizer = AutoTokenizer::from_pretrained("path/to/your/tokenizer")?;

// Create pipeline with custom components
let pipeline = pipeline!(model, tokenizer)?;
```

### Performance Optimization

```rust
use trustformers::{PipelineJitConfig, CompilationStrategy};

// Configure JIT optimization
let jit_config = PipelineJitConfig {
    enabled: true,
    compilation_strategy: CompilationStrategy::Adaptive,
    enable_kernel_fusion: true,
    optimization_level: 3,
    ..Default::default()
};
```

### Batch Processing

```rust
use trustformers::DynamicBatchingConfig;

// Configure dynamic batching
let batch_config = DynamicBatchingConfig {
    max_batch_size: 32,
    max_wait_time: Duration::from_millis(100),
    enable_priority_scheduling: true,
    ..Default::default()
};
```

## 📈 Benchmarking

Each example includes benchmarking capabilities. To run comprehensive benchmarks:

```bash
# Run all benchmarks
cargo run --example basic_pipeline -- --benchmark
cargo run --example dynamic_batching -- --stress-test
cargo run --example jit_optimization -- --profile --benchmark

# Compare different configurations
cargo run --example ensemble_models -- --compare-strategies
```

## 🤝 Contributing

We welcome contributions to improve these examples! Areas for enhancement:

- **New Examples**: Additional use cases and scenarios
- **Performance Improvements**: Optimization techniques and patterns
- **Documentation**: Better explanations and tutorials
- **Testing**: More comprehensive test coverage
- **Platform Support**: Windows, macOS, and Linux optimizations

### Guidelines

1. **Code Style**: Follow Rust conventions and use `rustfmt`
2. **Documentation**: Include comprehensive doc comments
3. **Testing**: Add unit tests for new functionality
4. **Performance**: Consider memory usage and execution time
5. **Safety**: Ensure memory safety and error handling

## 📝 Additional Resources

### Documentation

- [TrustformeRS User Guide](../docs/user_guide.md)
- [API Reference](../docs/api_reference.md)
- [Performance Tuning Guide](../docs/performance_tuning.md)
- [Architecture Overview](../docs/architecture.md)

### Community

- [GitHub Discussions](https://github.com/trustformers/trustformers/discussions)
- [Discord Server](https://discord.gg/trustformers)
- [Stack Overflow Tag: trustformers](https://stackoverflow.com/questions/tagged/trustformers)

### Research Papers

- [TrustformeRS: High-Performance Rust ML](https://arxiv.org/paper/trustformers)
- [Kernel Fusion in Modern ML Libraries](https://arxiv.org/paper/kernel-fusion)
- [Dynamic Batching for Transformer Models](https://arxiv.org/paper/dynamic-batching)

## 🏆 Acknowledgments

These examples build upon the excellent work of:

- **Hugging Face**: For pioneering accessible transformer models
- **PyTorch Team**: For ML framework design inspiration
- **Rust Community**: For the amazing ecosystem and tools
- **Research Community**: For advancing the state of ML

## 📄 License

These examples are provided under the same license as TrustformeRS:

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

**Happy Learning with TrustformeRS! 🚀**

For questions, issues, or suggestions, please visit our [GitHub repository](https://github.com/trustformers/trustformers) or join our [Discord community](https://discord.gg/trustformers).