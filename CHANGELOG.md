# Changelog

All notable changes to TrustformeRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Copyright 2025-2026 COOLJAPAN OU (Team KitaSan)

## [0.1.2] - 2026-06-20

### Added
- `Tensor::softmax_entropy_normalized()` method for normalized softmax entropy in `[0, 1]`
- `ActivationType` enum with `apply()` and `from_config_str_or()` helpers in `trustformers-models::common`
- `RotaryEmbedding::half_dim()` accessor in Phi-3 model
- Azure Container Instances (ACI) artifact generation (`generate_aci_artifacts`) with ARM template and CLI deployment script
- OpenShift deployment artifacts (`generate_openshift_artifacts`): BuildConfig, DeploymentConfig, Service, Route, and `oc` deploy script
- Full Kubernetes manifest generation for `DeploymentManifest`, `IngressManifest`, `ServiceManifest`, and `NetworkPolicyManifest` — includes user-supplied labels, annotations, and configurable selectors (previously stubs)
- Hub UI repository CRUD: `update_repository`, `delete_repository`, `update_version`, `delete_version` methods and corresponding HTTP handlers
- `EnhancedProfiler` export formats: Flamegraph (folded-stacks), OpenTelemetry (OTLP JSON spans), and Jaeger trace JSON
- Scaled dot-product attention chain detection in `KernelFusionEngine::find_attention_patterns` (MatMul → element-wise → Softmax → MatMul pattern with configurable flags)
- Disconnected node detection in `GraphDebugger::find_disconnected_nodes` with edge cross-validation
- Non-Maximum Suppression (NMS) in `ObjectDetectionPipeline`
- Token classification pipeline
- Q2_K and Q3_K GGUF block quantization methods with round-trip dequantize tests
- PNG heatmap visualization for sampled layers in `LargeModelVisualizer`
- Android backup module with NNAPI bindings, OpenGL ES and Vulkan GPU backends
- Federated learning v2 module: differential privacy, aggregation, secure communication, and crypto modules
- Regression tests for model integration and error handling in `MultiCloudOrchestrator`

### Changed
- `DynamicArchitectureManager::compute_entropy`, `compute_variance`, and `compute_sparsity` now have real tensor-based implementations (previously returned hardcoded constants 0.5, 0.3, 0.2)
- scirs2-core and scirs2-linalg updated from 0.4.2 → 0.5.0
- oxiarc-zstd, oxiarc-deflate, oxiarc-lz4, oxiarc-archive updated from 0.2.7 → 0.3.3
- oxicode updated from 0.2 → 0.2.4
- `MultiCloudOrchestrator` instance selection logic improved for better cloud instance matching
- Hardware acceleration benchmark: `criterion_main!` moved to crate root to fix E0601 (missing binary entry point) under `#[cfg(not(feature = "cuda"))]`

### Removed
- `tpu` feature flag and `tpu_impl.rs` module — the TPU backend was stub-only (all FFI binding bodies were unimplemented); removed to avoid misleading capability claims

### Fixed
- `GraphDebugger::find_disconnected_nodes` previously always returned an empty vec; now correctly identifies graph nodes with no live edges
- `KernelFusionEngine::find_attention_patterns` previously always returned an empty vec; now detects scaled dot-product attention chains
- Hub UI HTTP handlers `update_repository`, `delete_repository`, `update_version`, `delete_version` previously returned `NOT_IMPLEMENTED`; now delegate to working state methods
- Kubernetes manifest generators (Deployment, Ingress, Service, NetworkPolicy) previously generated minimal/incorrect YAML stubs; now produce correct, configurable manifests

## [0.1.1] - 2026-04-25

### Added
- 49+ transformer architectures (22 new architectures: Falcon2, Gemma2, Granite, Hyena, InternLM2, Jamba, Jamba2, Linformer, LLaMA3.2, Mamba2, Nemotron, Performer, Phi4, Qwen2.5, RetNet, S4, SD3, StableLM, StarCoder2, Whisper, xLSTM, Yi)
- Natural typing simulator for human-like response delivery
- Ensemble model types and strategies
- Resource analysis and monitoring structures

### Changed
- Upgraded SciRS2 dependencies to version 0.4.2
- Replaced ONNX Runtime with oxionnx (Pure Rust policy compliance)
- `tar` crate replaced with `oxiarc-archive` (COOLJAPAN policy)
- `rdkafka` Kafka backend feature-gated (`--features kafka`)
- 7 oversized files split using splitrs (COOLJAPAN 2000-line policy)
- Dependency upgrades: oxiarc-deflate/lz4 0.2.7, scirs2-core/linalg 0.4.2, wasm-bindgen 0.2.118, web-sys 0.3.95, lapin 4.5, redis 1.2

### Fixed
- Version consistency across all workspace crates
- Example crates missing `publish = false`
- cargo fmt formatting across 4 files
- 88 clippy unused-import warnings eliminated

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

[0.1.2]: https://github.com/cool-japan/trustformers/releases/tag/v0.1.2
[0.1.1]: https://github.com/cool-japan/trustformers/releases/tag/v0.1.1
[0.1.0]: https://github.com/cool-japan/trustformers/releases/tag/0.1.0
