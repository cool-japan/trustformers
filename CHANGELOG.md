# Changelog

All notable changes to TrustformeRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Copyright 2025-2026 COOLJAPAN OU (Team KitaSan)

## [0.2.1] - Unreleased

## [0.2.0] - 2026-07-09

### Added
- **CUDA backend gains a GPU-resident attention pipeline**: a new `gpu_ops::cuda::oxicuda::attention` module (`gather_heads_gpu_to_gpu`, `rope_neox_gpu_to_gpu`, `softmax_causal_gpu_to_gpu`, `attention_prefill_gpu_to_gpu`, `attention_decode_gpu_to_gpu`, `concat_v_cache_gpu_to_gpu`, `add_gpu_to_gpu`) gives oxicuda the same fully device-resident chain (QKV split → RoPE → causal softmax/attention → KV-cache append → head merge) the Metal backend already had. GPT-2 and GPT-NeoX (`Gpt2Attention::cuda_resident_attention`, `GPTNeoXAttention::cuda_resident_forward`) now use it for zero-host-round-trip prefill and, for GPT-2, incremental KV-cached decode; `Tensor::add` gains a matching zero-copy path for same-device residual adds.
- Batched and broadcasting CUDA matmul: a new `gpu_ops::cuda::oxicuda::batched` module with `BatchedMatmulPlan` (a NumPy-style broadcasting-batch shape planner) and `dispatch_oxicuda_matmul`/`dispatch_oxicuda_matmul_resident` replaces the previous 2D-only host-round-trip dispatcher; `Tensor::matmul` now routes N-D batched/broadcast F32 matmuls, and any operand pair involving a resident `Tensor::CUDA`, through the CUDA backend.
- `gpu_ops::cuda::BufferHandle` (`OxiCudaBufferHandle`): a reference-counted RAII handle for GPU-resident buffers. `CudaTensorData` and `Linear`'s cached CUDA weight buffer hold this instead of a raw buffer id, so the device allocation frees automatically when the last clone drops instead of leaking until an explicit `clear_buffer_cache()`. `gpu_ops::cuda::default_cuda_device_id()` reads a new `TRUSTFORMERS_CUDA_DEVICE` environment variable to pick the ordinal for host-dispatched matmuls on multi-GPU machines.
- Swin Transformer and DeiT are buildable for the first time: `trustformers-models` gains `swin`/`deit` Cargo features mounting `SwinModel`/`SwinForImageClassification` and `DeiTModel`/`DeiTForImageClassification` (with distillation-token support) — the implementations already existed in the tree but were never wired into `lib.rs`.
- Multi-objective hyperparameter optimization: `trustformers-training::hpo` is now mounted, exposing `MultiObjectiveHpo`/`ParetoFront`/`compute_pareto_front`/`hypervolume_indicator`/`non_domination_sort` for Pareto-front search, plus `AutoLrSelector`/`LrRangeTest` for automatic learning-rate range tests.
- **`trustformers` mounts three previously-dormant subsystems and ten task pipelines** that already existed in the tree but were never `pub mod`-declared: `cache` (`VersionedCache`, TTL/LRU/LFU/size eviction), `finetuning` (`LoraConfig`/`LoraLinear`, `AdapterConfig`/`BottleneckAdapter`), `loading` (`ParallelWeightLoader`/`load_model_parallel`), and the `audio_generation`, `document_classification`, `feature_extraction`, `image_segmentation`, `speech_recognition`, `table_question_answering`, `text_to_image`, `video_classification`, `visual_grounding`, and `zero_shot_audio_classification` pipelines under `trustformers::pipeline`.
- `AutoModelForObjectDetection` and `AutoModelForImageSegmentation` added to `automodel_tasks`, matching the existing `AutoModelForImageClassification`/`AutoModelForAudioClassification` pattern (deterministic mock pipelines pending real model backends).
- `trustformers-optim` re-exports 21 `fsdp`/`optimizer_surgery`/`per_layer_quant` types (`FsdpConfig`, `FsdpState`, `OptimizerSurgeon`, `SurgeryConfig`, `PerLayerQuantSelector`, `BitWidthStrategy`, etc.) directly at the crate root instead of only via their submodule path, locked in by a new `tests/crate_root_reexports.rs`.
- `trustformers-tokenizers` gains `NFKCNormalizer`/`NFKDNormalizer` (Unicode compatibility normalization) alongside the existing `NFCNormalizer`/`NFDNormalizer`.

### Changed
- **The `torch` backend is removed workspace-wide**: the `torch` Cargo feature, `tch` dependency, and `Tensor::Torch` variant are gone from `trustformers-core`, `trustformers-training`, `trustformers`, and `trustformers-c` (`tch`/`candle-nn` also dropped from `[workspace.dependencies]`); `candle` remains the optional non-CPU tensor-framework integration. Advances the Pure-Rust default feature tree (`tch` links libtorch, a C++ library).
- **`CudaTensorData` restructured (breaking)**: the public `buffer_id: BufferId` field is replaced by a `buffer: BufferHandle` field plus `buffer_id()`/`device_id()` accessors and a `CudaTensorData::new(buffer_id, device_id, shape, dtype)` constructor, to support the refcounted lifecycle and per-tensor device tracking above.
- **Model modules `mamba`, `rwkv`, `s4`, `falcon`, `stablelm`, and `linformer` are now properly feature-gated (breaking)** in `trustformers-models`: their Cargo features already existed, but the `pub mod` declarations carried no matching `#[cfg(feature = ...)]`, so they were unconditionally compiled regardless of the flag. Builds that relied on them without enabling the feature (or `all`) now need to add it.
- `OxicudaCudaBackend::new` validates the requested `device_id` against the real enumerated CUDA device count and returns a precise range error instead of an opaque downstream driver failure.
- `AutoTokenizer::from_pretrained_with_revision` now downloads `tokenizer.json` through `hub::download_file_from_hub` (revision-aware; a cache hit short-circuits before any network access), falling back to the previous local-cache-only lookup only if that download fails.
- Metal's scaled-GEMM output buffer switches from `StorageModePrivate` to `StorageModeShared` so oxicuda's resident GEMM can import it via `register_external` (which requires a CPU-accessible buffer); on Apple Silicon's unified memory, `Shared` is still GPU-resident, so there is no readback penalty.
- `oxicuda-{blas,dnn,memory,driver,metal,backend}` updated 0.4.0 → 0.4.1, fixing a GEMM transpose-flag bug and a batched-GEMM launch-tuple corruption bug that made `gemm_strided_batched` produce incorrect results in 0.4.0 (both verified against upstream sources; part of the motivation for the new resident-attention and batched-matmul modules above). `oxiarc-{zstd,deflate,lz4,archive}` updated 0.3.3 → 0.3.5.
- Dependency cleanup: unused `scirs2-linalg` dropped workspace-wide (root and `trustformers-mobile`); the workspace `scirs2-core` feature list drops `gpu` (GPU acceleration now runs entirely through the oxicuda/Metal backends); `trustformers-tokenizers` drops the unused `hangul` dependency; `trustformers-wasm` drops the optional `scirs2-core`/`scirs2` integration.
- `trustformers-mobile`: default features are now empty (previously `["mobile-optimized"]`); the `mobile-optimized`, `ios`, and `android` flags — never referenced by any `#[cfg(feature = ...)]` in the crate — are removed.

### Removed
- **`trustformers-mobile::android_renderscript`** (~750 lines): the legacy RenderScript compute backend for pre-Vulkan Android (API 24-30) devices; RenderScript itself was deprecated by Google in API 31. `AndroidRenderScriptEngine` and its config/stat types are no longer available — current Android builds use the Vulkan/NNAPI backends instead.
- Orphaned duplicate source files, deleted as dead-code cleanup; none were reachable from any crate's `lib.rs` even before this release, so none of this removes a shipped capability:
  - `trustformers-models::qwen2` (config/mod/model/tasks/tests, ~2,090 lines) — Qwen support continues via the mounted `qwen` (Qwen 1) and `qwen2_5` (Qwen 2.5) modules.
  - `trustformers-optim::{adafactor, adafisher, advanced_benchmarking, second_order_new}` (~2,390 lines) — `AdaFactor`/`AdaFisher` continue via the mounted `adafactor_new`/`adafisher_simple` modules, second-order optimizers via the mounted `second_order` module.
  - `trustformers-mobile::{federated_core, federated_learning, federated_learning_v2::*}` (~4,280 lines) — federated learning continues via the mounted, feature-gated `federated` module.
  - `trustformers-training/src/mod.rs`, an orphaned root-level file duplicating declarations already made in `lib.rs`.

### Fixed
- **CUDA GPU-to-host reads could silently return zero or garbage data**: `OxicudaCudaBackend`'s `download_buffer`, `matmul_f32`, `matmul_with_cached_weight`, `gelu_f32`, `layernorm_f32`, `softmax_causal_f32`, and `rope_f32` now synchronize the backend's stream before their device→host copy. oxicuda kernels launch asynchronously on a non-blocking stream, but the synchronous memcpy runs on the legacy default stream, which does not implicitly wait for it — without this, a fast host thread could read back a buffer before the kernel filling it had completed.
- **`Device::cuda_if_available()`/`Device::best_available()` always returned `Device::CPU`, even on CUDA/Metal-capable hardware**: both now probe `gpu_ops::cuda::oxicuda_cuda_available()`/`metal::Device::system_default()` directly instead of `scirs2_core::simd_ops::PlatformCapabilities`, whose `cuda_available` is hardcoded `false` in scirs2-core 0.6.0 (CUDA support retired upstream) and whose `metal_available` needs a scirs2 `metal` feature trustformers never enables.
- Multi-GPU tensors could address the wrong device: `CudaTensorData` now carries its own `device_id` instead of callers inferring it from a `Layer`'s configured `Device` or hardcoding `0`; `LayerNorm`, `Linear`, `gelu`, and `Tensor::add`/`matmul` now operate on the buffer's actual device and fall back to the host path on a device mismatch instead of silently touching the wrong GPU. `Tensor::to_device_enum` between two different CUDA devices previously just cloned the tensor in place (leaving it on the original device); it now transfers via the host.
- `SentencePieceTokenizer::from_pretrained` ignored its `model_name_or_path` argument and always returned a fabricated T5-like vocabulary; it now probes `{path}/spiece.model`, `{path}.model`, and the bare path for a real SentencePiece model file and loads it, falling back to the fabricated vocabulary only when none resolve.
- `OfflineModelPackManager::get_model_info` returned hardcoded mock metadata (a fixed `"text-generation"` tag, 1000 downloads, 50 likes) for every model; it now queries the real Hugging Face Hub `/api/models/{id}` endpoint (behind the `hub` feature) through a new, independently unit-tested `model_info_from_hub_json` mapper.
- `trustformers-wasm` quantization (`apply_dynamic_quantization`/`apply_static_quantization`/`apply_post_training_quantization`) previously just multiplied input values by an arbitrary constant (0.5/0.75/0.8); they now perform real affine (min/max, scale/zero-point) quantize-dequantize rounding sized to the configured `QuantizationPrecision`, and `WebQuantizer::get_stats` computes compression ratios from `QuantizationPrecision::bytes_per_element()` instead of always assuming 4 bytes/element.
- `trustformers-wasm` device-capability probes returned hardcoded results regardless of the real browser/device: `detect_webgl_support` and `get_screen_orientation` now query the actual WebGL context and `window.screen().orientation()`, and `compute::webgpu::get_device_capabilities()` runs the real `DeviceSelector::analyze_device_capabilities()` instead of returning `DeviceCapabilities::default()`; the error-recovery system's `RecoveryAction::ClearCache` now actually clears the browser's Cache Storage instead of being a no-op.
- `trustformers-serve`'s `create_compliance_focused_service()`/`create_resource_efficient_service()` previously discarded their intended configuration because the target fields didn't exist on `TestPerformanceMonitoringConfig`; the config gained `analytics_config`/`event_config`/`historical_data_config`/`alert_config`/`dashboard_config`/`subscription_config` sub-manager fields (with `compliance_reporting`/`compliance_logging`/`audit_trail_enabled`/`rate_limiting_enabled` flags) so these constructors now actually apply the settings they advertise.
- `trustformers-models`' `all` feature aggregate was missing the already-implemented `llama3_2` (LLaMA 3.2) and `mistral_v3` (Mistral v0.3) model features; `--features all` now builds every supported architecture.
- `trustformers-c` (legacy, standalone-build crate): fixed a mismatched-type compile error in `trustformers_batch_decoded_texts_free`'s allocation-failure path; `ContainerDeploymentManager::generate_deployment_artifacts` now routes through the crate's own `DockerImageBuilder` (gaining a `BaseImage::Custom` variant) instead of hand-rolled `format!()` templates that silently dropped `build_args`/`env_vars`/`exposed_ports`/`volumes`.

## [0.1.4] - 2026-07-02

### Added
- Real `PyRwkvModel` / `PyMambaModel` Python classes; `AutoModel` now loads RWKV/Mamba checkpoints correctly instead of silently falling back to BERT. (Re-enabled and modernized the Python bindings to PyO3 0.28.)
- WebGPU device/queue initialization in the WebAssembly compute backend (falls back to CPU when no adapter).
- 12 CPU↔CUDA golden-parity tests (GEMM, GELU, LayerNorm, causal softmax, RoPE — both host and GPU-resident paths, plus cached-weight GEMM) proving the oxicuda CUDA backend is numerically correct against the CPU reference; runtime-verified 12/12 passing on real NVIDIA hardware (RTX A4000, CUDA 12.0).

### Changed
- **CUDA backend migrated from `cudarc` to the Pure-Rust `oxicuda`** (COOLJAPAN Pure-Rust policy): the `cuda` feature now pulls in `oxicuda-blas`/`-dnn`/`-memory`/`-driver` instead of `cudarc`; `cuda-oxicuda` is kept as a deprecated alias for `cuda`. Runtime-verified end-to-end on real NVIDIA hardware. GPU-resident `matmul_gpu_to_gpu` confirmed genuinely zero-copy (cached `DeviceBuffer`, no host round-trip). The `cuda` feature is now propagated through `trustformers-models` and the `trustformers` umbrella crate.
- The CUDA transformer-layer forward pass now runs as a real GPU-resident pre-norm causal self-attention layer (LayerNorm→QKV→bias→RoPE→causal-softmax attention→proj→residual, chained via cached device buffers with no host round-trips), replacing the previous CPU-fallback placeholder.
- Metal GPU compute (matmul + resident attention) migrated from scirs2 MPS to oxicuda-metal (Pure Rust); dropped the `scirs2-core/"gpu"` dependency; GPU-resident matmul is now zero-copy. Verified on Apple Silicon.
- Eliminated production-code `unwrap()`/`expect()` across the entire workspace (replaced with proper error propagation, lock-poison recovery, and documented infallible invariants); reduced `#[allow]` suppressions by fixing the underlying lints. No public API changes; all workspace tests pass unchanged.
- Default feature trees are now Pure-Rust (C/C++-free) for all crates except `trustformers-serve` (HTTP server; keeps rustls/aws-lc-rs TLS). Networking (HuggingFace hub downloads, remote leaderboard), debug visualization, and serve's AWS-Lambda/Swagger-UI adapters are now opt-in behind features (`hub`, `remote-leaderboard`, `visual`, `lambda`, `swagger-ui`). Switched the tokenizer regex backend to pure-Rust `fancy-regex` and removed an unused `jieba-rs`/`zstd` dependency. No default public API removed; all default tests pass.
- GPT-2 feed-forward uses a single fused matmul+bias+GELU Metal kernel on Apple Silicon (one GPU dispatch instead of three).
- Bumped workspace dependency versions across the board, including `sha2` 0.10.9→0.11.0, `nalgebra` 0.34.2→0.35.0, `tokenizers` 0.22→0.23, `candle-core`/`candle-nn` 0.9.2→0.11.0, `tch` 0.23→0.24, `safetensors` 0.7→0.8, `tower-http` 0.6→0.7, `axum-test` 19.1→21.0, and `azure_core`/`azure_identity` 0.33→1.0, plus patch-level updates to `anyhow`, `tokio`, `reqwest`, `wasm-bindgen`/`web-sys`, the AWS SDK crates, and others. The `sha2` 0.11 bump changes `Sha256::finalize()`'s output type (now backed by `hybrid-array` instead of `generic-array`), so hash-to-hex formatting switched from `format!("{:x}", hasher.finalize())` to explicit `hex::encode(hasher.finalize())` (identical lowercase-hex output) in `trustformers-core::versioning::storage`, `trustformers-serve::auth::functions`/`migration::model_migration`, `trustformers-training::model_versioning`, and `trustformers::hub_local_mirror`/`hub_offline_packs`; `trustformers-core` and `trustformers-training` gained an explicit `hex` workspace dependency for this.
- `wgpu` upgraded from 29.0 to 30.0 (now a `[workspace.dependencies]` entry, consumed via `workspace = true` in `trustformers-core`), behind the `wgpu_backend` feature (part of `full`): `WebGpuBackend`'s adapter request now sets the new required `apply_limit_buckets: false` (this is a native, trusted compute backend rather than untrusted web content, so real adapter limits are wanted over fingerprinting-resistant buckets), and the 4 call sites in `gpu_ops::webgpu` reading `BufferSlice::get_mapped_range()` were migrated to handle its new `Result<BufferView, MapRangeError>` return via `.map_err(...)` into `TrustformersError::hardware_error` (no `unwrap()` introduced). Verified clean (`check`/`build`/`test --no-run`/`clippy`) with `--features wgpu_backend`.

### Removed
- The `cudarc` dependency and the entire legacy cudarc-based CUDA backend in `trustformers-core` (`gpu_ops/cuda/cuda_split/`, duplicate `gpu_ops/cuda/{backend,types,buffer_ops}.rs`, `gpu_ops/advanced_kernels.rs`, `kernels/cuda_impl.rs`, and the stubbed-out `kernels/cuda_kernels.rs`) — superseded by the oxicuda backend above. `cudarc` remains only in the out-of-workspace legacy `trustformers-c` FFI crate.
- The orphaned `rope/mod.rs` module (~1,693 lines; never mounted in the module tree, and used a RoPE convention inconsistent with the live kernel) — the compiled CPU RoPE reference is `kernels/rope.rs` (GPT-NeoX half-split convention), which now has a CPU↔CUDA parity test.

### Fixed
- Restored gRPC proto compilation and serving (migrated build to the tonic 0.14 split `tonic-build`/`tonic-prost-build` API); re-enabled the gRPC service module.

## [0.1.3] - 2026-06-24

### Added
- Gorilla-style time-series compression in `trustformers-serve` historical data (`CompressionEngine::compress_series`): delta-of-delta timestamp encoding plus XOR float encoding with leading/trailing zero-bit counts, replacing a `TODO` stub; adds `optimize_compression`
- Historical-data lifecycle, archival, and query engine in `trustformers-serve`: real `cleanup_expired_data`, `evaluate_lifecycle`, and `check_deletion_allowed` (LifecycleManager); `archive_data`/`retrieve_data` (ArchivalSystem, previously `TODO: Implement actual archival logic`); and `execute_query`/`check_cache`/`cache_result` (query engine) — all converted from parameter-ignoring stubs to working implementations
- Concurrency-detector analytics in `trustformers-serve` (`performance_optimizer::test_characterization::concurrency_detector`): real cycle/deadlock, thread, lock, and conflict detection plus pattern, sharing, and risk-assessment analytics across the eight detector modules (lock-cycle extraction, detection-confidence and risk scoring); clears dozens of type-mismatch `TODO` stubs in `detector.rs`
- Interpretability tools wired in (`trustformers-debug`): real SHAP, LIME, and Integrated-Gradients feature attribution via a new `interpretability` module, exposed as `InterpretabilityAnalyzer`, `InterpretabilityConfig`, and `InterpretabilityReport` (replaces previous placeholder unit-struct stubs)
- `ZeroCopyTensorView<'a>` in `trustformers`: a bounds-checked, sub-viewable borrowed `f32` tensor view (`from_slice`, `subview`, stride computation), re-exported from the crate root
- `GlobalMemoryPool` in `trustformers`: a thread-safe aligned allocator (`allocate`, `allocate_aligned`, `unsafe deallocate`) with layout-tracked, leak-free deallocation, re-exported from the crate root
- `GlobalProfiler` operation API in `trustformers`: `Profiler::instance()` plus per-session `start_operation`/`end_operation`, with `GlobalProfiler` re-exported from the crate root
- Real `.xlsx` export in `trustformers-debug` data export: emits a valid Office Open XML (OOXML) workbook package (`[Content_Types].xml`, relationships, workbook, worksheet) via `oxiarc-archive` (Pure Rust), replacing the previous CSV-with-`.xlsx`-extension placeholder

### Changed
- `PerformanceModelingEngine` now stores each trained model in `active_models` (pushing an `Arc<dyn PerformancePredictor>` on every `train`) instead of discarding it after training
- Resource statistics now compute real active-resource and peak-usage figures from per-subsystem snapshot stats (port, directory, GPU, database) instead of hardcoded `0`/snapshot-count placeholders
- Consolidated several crate dependencies to `workspace = true` for single-source version management: `async-trait`, `log`, `tower`, and `tower-http` (`trustformers`); `rmp-serde` (`trustformers-serve`); `libc` (`trustformers-mobile`); and `memmap2` (`trustformers-tokenizers`)

### Fixed
- HuggingFace upload now computes a real SHA-256 digest (via `sha2`) for content addressing; the previous `sha256_stub` was a non-cryptographic 128-hex XOR-fold, not SHA-256 (output is now 64 hex chars, covered by known-answer test vectors)
- Heap mis-layout deallocation (undefined behavior) and a memory leak in the zero-copy / memory pools: reused blocks now record and free their *actual* allocation layout instead of the smaller requested size (freeing with a mismatched layout is UB), and each `MemoryBlock` releases its backing allocation exactly once through its own `Drop` (regression test added for larger-block reuse)
- clap `-c` short-flag collisions that panicked the `load_test` and `message_queue_cli` binaries on startup: the colliding arguments now pin an explicit `short = 'n'`
- Replaced a flaky wall-clock assertion in the async-operations benchmark (`per_operation < 100µs`, which failed under parallel test execution due to scheduler contention) with a deterministic liveness check (every yielded task resumes; elapsed time is positive)

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

[0.2.0]: https://github.com/cool-japan/trustformers/releases/tag/v0.2.0
[0.1.4]: https://github.com/cool-japan/trustformers/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/cool-japan/trustformers/releases/tag/v0.1.3
[0.1.2]: https://github.com/cool-japan/trustformers/releases/tag/v0.1.2
[0.1.1]: https://github.com/cool-japan/trustformers/releases/tag/v0.1.1
[0.1.0]: https://github.com/cool-japan/trustformers/releases/tag/0.1.0
