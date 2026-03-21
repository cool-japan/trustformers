# Changelog

All notable changes to TrustformeRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Copyright 2025-2026 COOLJAPAN OU (Team KitaSan)

## [0.1.0] - 2026-03-20

### Added

#### Transformer Architectures
- 21+ transformer model implementations including BERT, GPT-2, T5, LLaMA, Mistral, Falcon, MPT, BLOOM, OPT, Phi, Gemma, Qwen, StableLM, RWKV, Mamba, Flamingo, CLIP, and more
- Configurable model architectures with builder-pattern APIs
- CLIP text and vision encoder weight loading from HuggingFace format
- Conv2D forward pass with full im2col + matmul implementation (groups, dilation, stride, padding)

#### Performance & Hardware Acceleration
- 17x CPU BLAS acceleration via direct cblas_sgemm for matrix operations
- Metal GPU support (macOS) with MPS integration and 2.88x overall improvement
- GPU-resident tensor operations eliminating CPU roundtrips
- Flash Attention support with optimized batched matrix multiplication
- CUDA and ROCm backend support with automatic CPU fallback
- WebGPU compute shader backend for browser-based inference
- SIMD-optimized tensor operations
- NUMA-aware topology detection (Linux, macOS) for optimal thread placement

#### Quantization
- GGML and GGUF quantization format support
- AWQ (Activation-aware Weight Quantization)
- GPTQ (Generative Pre-trained Transformer Quantization)
- Quantization-aware training infrastructure

#### Tokenizers
- BPE (Byte Pair Encoding) tokenizer
- WordPiece tokenizer
- SentencePiece tokenizer
- Configurable vocabulary and special token handling

#### Training
- Distributed training infrastructure with model and data parallelism
- DPO (Direct Preference Optimization) and KTO loss functions
- 20+ optimization algorithms
- Hyperparameter tuning and auto-tuning support
- Gradient checkpointing and mixed-precision training

#### Multi-Platform Deployment
- **WASM**: Browser-based inference with WebGPU acceleration
- **Python bindings**: PEP 440-compliant Python package (`trustformers-py`)
- **C FFI**: C-compatible API (`trustformers-c`) with code generation
- **Mobile**: Optimized inference for mobile targets (`trustformers-mobile`)
- **Server**: gRPC and REST serving infrastructure (`trustformers-serve`)

#### Safety & Reliability
- Content safety filters with toxicity scoring and harm pattern detection
- Model versioning and A/B testing infrastructure
- Inference caching with configurable eviction policies (LRU and custom)
- Memory profiling and leak detection tooling
- Comprehensive error codes and structured error handling

#### Compression
- OxiARC-based compression (Pure Rust): deflate, zstd, lz4

#### Code Quality
- 5,010+ tests passing across the entire workspace
- Zero clippy warnings (`-D warnings` enforced)
- 100% Pure Rust — no C/Fortran dependencies in default features (COOLJAPAN Policy)
- Workspace-consolidated dependencies (110+ shared)
- MSRV 1.75

#### Documentation
- Architecture guide, deployment guide, and performance tuning documentation
- Quantization guide with advanced techniques
- Tokenizer selection guide, training best practices, and troubleshooting
- Model implementation guide and style guide
- Migration guides for PyTorch and HuggingFace users
- Interactive demos: tensor playground, WebGPU demo, benchmark dashboard

---

[0.1.0]: https://github.com/cool-japan/trustformers/releases/tag/0.1.0
