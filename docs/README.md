# TrustformeRS Documentation

Welcome to the TrustformeRS documentation! This directory contains comprehensive guides and references for using TrustformeRS effectively.

## 📚 Documentation Index

### Getting Started
- [Quick Start Guide](../README.md) - Basic installation and usage
- [API Documentation](https://docs.rs/trustformers) - Rust API reference

### Core Concepts
- [Architecture Overview](./architecture.md) - System design and components
- [Tensor Operations](./tensors.md) - Working with tensors
- [Model Implementation](./model_implementation.md) - Creating new models

### Training Guides
- [Basic Training](./basic_training.md) - Single GPU training
- [Distributed Training](./distributed_training.md) - Data parallel training
- [Multi-Node Training](./multi_node_training_guide.md) - MPI and ZeRO optimization
- [Pipeline Parallelism](./pipeline_parallelism.md) - Training huge models
- [Mixed Precision Training](./mixed_precision.md) - FP16/BF16 training

### Performance
- [Performance Tuning](./performance_tuning.md) - Optimization techniques
- [Optimization Advisor](./optimization_advisor_guide.md) - Automated performance analysis
- [Benchmarking](./benchmarking.md) - Performance measurement
- [Memory Management](./memory_management.md) - Efficient memory usage

### Deployment
- [Model Export](./model_export.md) - ONNX, GGUF formats
- [Inference Optimization](./inference.md) - Production deployment
- [Mobile Deployment](./mobile_deployment.md) - iOS/Android integration
- [Server Deployment](./server_deployment.md) - REST/gRPC APIs

### Migration
- [Migration from PyTorch](./migration_pytorch.md) - PyTorch to TrustformeRS
- [Migration from HuggingFace](./migration_huggingface.md) - HF Transformers compatibility

### Advanced Topics
- [Custom Layers](./custom_layers.md) - Implementing new layers
- [Quantization](./quantization.md) - Model compression
- [Advanced Quantization](./advanced_quantization_guide.md) - SmoothQuant and GGML Q5/Q6
- [Fine-tuning Methods](./finetuning.md) - LoRA, QLoRA, PEFT
- [WebAssembly Support](./wasm.md) - Browser deployment

### Developer Resources
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- [Testing Guide](./testing.md) - Writing and running tests
- [Debugging Guide](./debugging.md) - Troubleshooting tips
- [Error Handling Guide](./error_codes.md) - Enhanced error messages and recovery
- [Model Visualization Guide](./model_visualization_guide.md) - Architecture graphs and analysis

## 🚀 Quick Links

- **GitHub Repository**: https://github.com/trustformers/trustformers
- **Crates.io**: https://crates.io/crates/trustformers
- **Discord Community**: Coming soon
- **Examples**: See the `examples/` directory

## 📖 Documentation Conventions

Throughout the documentation:
- 🔧 indicates configuration options
- ⚡ indicates performance tips
- ⚠️ indicates important warnings
- 💡 indicates helpful tips
- 📝 indicates code examples

## 🤝 Contributing to Docs

We welcome documentation improvements! To contribute:

1. Fork the repository
2. Create a new branch for your changes
3. Write clear, concise documentation
4. Include code examples where appropriate
5. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

## 📞 Getting Help

If you can't find what you need:

1. Check the [FAQ](./faq.md)
2. Search existing [GitHub Issues](https://github.com/trustformers/trustformers/issues)
3. Ask in the community Discord
4. Open a new issue with your question

Happy transforming! 🤖