# Changelog

All notable changes to TrustformeRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Copyright 2025-2026 COOLJAPAN OU (Team KitaSan)

## [0.2.1] - Unreleased

A production-grade honesty and correctness pass: replacing fabricated/placeholder logic with real implementations or structured errors, deleting unsound and orphaned code, fixing several deadlocks, and tightening dependency hygiene. First verified against the tree on 2026-08-18 (`cargo check --workspace --all-targets` green; workspace test count grown from 14,887 to 20,629+ passing across the underlying waves), then a resumed pass on 2026-08-24 closed most of what the 2026-08-18 checkpoint still listed as open (see below) — `cargo check --workspace --all-targets` green and `cargo nextest run --workspace` at 20,629 passed / 43 skipped / 0 failed as of that date too. Not everything targeted this cycle landed cleanly — see `TODO.md` for the specific items still open, including 7 open `cargo deny check advisories` findings and a `--features metal` compile break in `trustformers-models` newly surfaced by this cycle's own in-progress Metal buffer-lifetime work.

### Security
- **Deleted an RS256 JWT auth bypass**: `trustformers-serve`'s orphaned auth module cluster accepted RS256-signed tokens without verifying the signature against a real key, alongside other unsound, never-mounted code. The cluster was never reachable from any router, so no shipped endpoint was exposed — but the fabricated auth surface has been removed rather than fixed-in-place, since nothing depended on keeping its shape.
- `cargo deny check licenses` and `check bans` are green (`deny.toml`'s `[bans]` deny-list, previously an empty vestigial comment behind a schema cargo-deny 0.19 couldn't fully parse, is now populated with the full COOLJAPAN banned-crate list on a current schema).
- Five deadlock-class bugs fixed: an unconditional re-entrant deadlock in the Metal GPU `attention_gpu_to_gpu_optimized` path; a `tokio::sync::Mutex` deadlock in `DistributedDebugger::coordinate_operation`; three `RwLock` read-read reentrancy hazards (`hub_local_mirror::download_model`/`cleanup_cache`, `quality_analyzer`) where a held read guard was re-acquired on the same lock during a nested call.
- `AlertManager::send_notification` now returns `Err` on a misconfigured `file_logging`/`webhook` target, or a webhook request made without the `http-integrations` feature enabled, instead of silently returning `Ok` while doing nothing. These are opt-in notification paths only.

### Added
- `trustformers-serve::openai_compat` is mounted on both the production and test routers (`create_router`/`create_test_router`), backed by a real `OpenAiInferenceBackend` adapter over the batching/model-executor infrastructure (`Gpt2BatchModel`/`HuggingFaceTokenizer`) — previously unmounted and unreachable from any endpoint.
- 27 previously-orphaned `*_tests.rs` files in `trustformers-serve` are now compiled and running (+770 tests); a source-tree orphan guard test was added so a file landing in `src/` without being reachable from `lib.rs`/`mod.rs` fails CI instead of silently never compiling.
- Three honest orphan modules mounted in `trustformers-serve`: `caching::lru`, `semantic_cache`, `batching::variable_length`.
- `trustformers-training::distributed_zero` (ZeRO optimizer-state/gradient/parameter partitioning) is implemented and tested.
- `trustformers-mobile::advanced_security` gained real post-quantum primitives: `KyberKem` (FIPS 203 ML-KEM via the `ml_kem` crate), `DilithiumSigner` (FIPS 204 ML-DSA via `ml_dsa`), `SphincsSigner` (FIPS 205 SLH-DSA via `slh_dsa`) — replacing placeholder byte-tricks that previously stood in for Kyber/Dilithium/SPHINCS+/homomorphic-encryption/MPC. Related pure-Rust additions: `num-bigint`, `chacha20poly1305`, `hkdf`/`hmac`, `subtle`, `zeroize`, `getrandom` 0.4, `rand_core` 0.10.
- `trustformers-py::models::losses` (new): real classification and language-modeling cross-entropy loss, closed-form verified against a reference computation, with 11 tests.
- `trustformers` umbrella crate gained a `metal = ["trustformers-models/metal"]` feature — Metal GPU acceleration was previously unreachable from the published entry-point crate (only `cuda` forwarded).
- **`trustformers-core::gpu_ops::metal::types::MetalBufferHandle`** (new): a refcounted RAII handle that releases its Metal buffer's cache entry when the last handle drops. `Tensor::Metal` and every GPU-to-GPU result site `trustformers-core` itself owns now use it (`gelu`, `Tensor::add`, `layernorm_gpu_to_gpu`, `Linear::forward`, `tensor_to_f32`'s download read) — Metal buffers release on last-reference-drop instead of staying live until an explicit cache eviction. (`trustformers-models`'s `gpt2` module has not yet been converted to the new handle type; see `TODO.md` P0.)
- **`trustformers-serve/clients/rust`** implements the OAuth2 client-credentials grant natively (no `oauth2` dependency), with hermetic regression tests, on top of 11 pre-existing compile errors fixed and a manifest that now resolves standalone (its own `[workspace]` table, literal dependency versions, `reqwest`'s renamed `rustls` feature).
- **`examples/grpc-server`** migrated from `tonic 0.12`/`prost 0.13` to `tonic 0.14.6`/`tonic-prost 0.14.6`/`prost 0.14` (tonic 0.14 moved codec generation into a separate `tonic-prost-build` crate); the service was rewritten against the real proto and the real `AutoModel` API, with a model registry that measures instead of estimating, real env-based configuration, and gRPC reflection v1/v1alpha.
- **`trustformers-models`**: `FNetLMPredictionHead` is now a real implementation (previously `fnet`'s task wrappers silently dropped their head entirely); `bind_head_linear`/`bind_head_layer_norm` were hoisted out of `bert::tasks` into a shared binding module so `roberta` and `albert`'s task wrappers — which previously bound the encoder but dropped the head, same as `fnet` — can reuse them.
- **`trustformers-py`**: a pure `TokenizedInput` builder (replacing an error-swallowing open-coded conversion) and a pure decoding core are now shared between `GPT2LMHeadModel.generate()` and the text-generation pipeline; pure classification scoring, including a real BERT forward pass, backs the text-classification pipeline.
- **`trustformers` umbrella**: several previously-dead fields/types gained real accessors instead of staying unused — `ConfigurationManager::unrecognized_env_vars()`, `Mamba2Model::models_config()`/`layers()`, `OpenVINOModel::wrapper()` plus 5 wrapper accessors, `DdimScheduler::beta_range()`, `ObjectDetectionPipeline::labels()`/`resolve_label()`, `base_model()`/`cached_layer_count()`, `uptime()`/`advisor()`/`metrics_tracker()`; `GroundingProcessor::encode_phrase` now uses its own `djb2_hash` helper instead of leaving it dead, and `GroundingResult::requested_phrases()` was added; `document_cache` is now wired for real cross-hop dedup instead of sitting unused.

### Changed
- **`trustformers-tokenizers::jax::JaxCompiledTokenizer::is_compiled()`** used to hardcode `true` regardless of the config passed to `JaxTokenizer::jit_compile`, so a tokenizer built with `use_xla: false` still (falsely) reported itself compiled. It now honestly reflects the config.
- **`SentencePieceTokenizer::from_pretrained`'s fabricated-vocabulary fallback is gone**: 0.2.0 made it probe for a real model file first but still fell back to a fabricated T5-like vocabulary if none resolved; it now returns a hard `Err` naming every path it probed instead. `WordPieceTokenizer::from_pretrained` does the same (probes a fixed list of local vocab-file paths, hard-errors naming them all if none exist) — neither tokenizer will silently hand back a made-up vocabulary. Migration docs describing the old fallback behavior are corrected in `trustformers-tokenizers/docs/migration/`.
- **`trustformers-core::hardware_acceleration`**: removed a file-level `#![allow(unused_variables)]` that had been hiding a real bug — `cpu_flash_attention` computed but never used `batch_size`/`seq_len`, so mismatched Q/K/V head-dimension shapes reached `matmul` unchecked instead of erroring. Now rejected with a proper shape-mismatch error.
- **`trustformers-models` checkpoint loading**: `roberta`, `albert`, `fnet`, `nemotron`, `phi4`, `mistral_v3`, `phi2`, `yi`, `starcoder2`, `llama3`, and `command_r` now bind real weights via `Checkpoint::from_reader` + `load_checkpoint`, replacing a silent `Ok(())` that left the model at its randomly-initialized weights with no indication anything was wrong. The recurrent/exotic architectures this checkpoint machinery can't yet serve faithfully (`rwkv`, `mamba`, `hyena`, `performer`, `retnet`, plus `llama3_2`'s Mllama tiling and `deepseek`'s fused MLA projections) now return a `not_implemented` error naming the specific architectural reason, instead of the same silent `Ok(())`. (`s4` and `deepseek_v2` are deliberately **not** in either list above — both had their own fake-success loaders removed this same cycle; see the dedicated `s4`/`deepseek_v2` bullet below, which supersedes any earlier impression that either was "complete".)
- **`trustformers-models::generation_utils::should_stop`** now takes the prompt length and counts `max_new_tokens` against the completion only; previously it compared against the whole sequence (prompt + completion), so `max_new_tokens` silently behaved as a whole-sequence cap.
- **GPT-2 contrastive search** is implemented; the `ContrastiveSearch` generation-mode variant was publicly constructible and documented but calling it returned "not yet implemented for GPT-2".
- **`Swin` and `DeiT`** now implement the `Model` trait (`impl Model for SwinModel`/`DeiTModel`), so they load through the standard interface like every other supported architecture.
- **`trustformers-training::distributed`**: NCCL/Gloo/MPI backends no longer simulate collectives — a `broadcast` implementation that corrupted its input via an unconditional `scalar_mul(0.99)` is gone. Requesting one of those three backends now returns a structured "unavailable" error naming the real, working TCP-based process-group backend, instead of silently no-op'ing or corrupting data. `dpo::get_batch_logps` now does a real masked per-token log-probability gather with bounds-checked label validation, replacing an approximation that used `logits.mean()` instead of gathering the label-indexed logit at each position.
- **`trustformers-serve` model/cloud honesty fixes**: `cloud_providers.rs` no longer returns a canned `"Mock response"` / `https://example.com/endpoint` for any of its 6 provider integrations; `model_management::manager::ModelInstance::infer` no longer synthesizes `"Generated response for: {}"` instead of running the model; notification channels now deliver for real or return a structured error instead of three fabricated synchronous stand-ins; GPU statistics report their actual absence instead of inventing VRAM/temperature/power figures; resource-cleanup handlers report only memory a component actually released; model validation scores the model's real output instead of scoring the labels it was given; a benchmark that published its own inputs back as its "results" is deleted; an untrained regressor now refuses to predict instead of returning a number; a real bug where computed normalization parameters were silently dropped before use is fixed; system metrics are read from the host instead of derived by hashing the clock; cache-node health checks now open a real TCP connection; cache warming predicts only queries it has actually observed.
- **`trustformers-mobile::react_native_fabric::FabricRenderer`** gained a real `inference_engine` field; `execute_standard_inference` previously built and discarded a request object and returned a hardcoded `output_data: vec![1.0, 2.0, 3.0]` regardless of the request, and now returns the real inference tensor's data.
- **`trustformers::pipeline` multimodal vision path** (`auto/feature_extractors/vision.rs`) now does real image decoding (binary Netpbm always; every format the `image` crate handles behind the `vision` feature), real bilinear-interpolation resize, real centre crop, and real per-channel `(pixel - mean) / std` normalization, replacing zero vectors.
- `trustformers-serve/Cargo.toml` drops four zero-reference cloud SDKs (`azure_core`, `azure_identity`, `azure_mgmt_machinelearningservices`, `azure_mgmt_web`) plus `google-cloud-gax`, `google-cloud-functions-v2`, and `lambda_runtime` — none had a call site anywhere in `src/`; the Azure/GCP serverless providers already returned `ServerlessError::NotImplemented` rather than calling any SDK, so no capability is lost. `serve/clients/rust` gained its own `[workspace]` table and literal dependency versions (it previously used `*.workspace = true` keys that only resolve inside the repository workspace it is deliberately excluded from, making it unbuildable in place).
- `trustformers-debug` drops the `ffmpeg-next` dependency and `video` feature (zero call sites anywhere in the crate; was breaking the crate's own `--all-features` build). `trustformers-core` drops the dead `candle` feature and `candle-core` dependency (`Tensor::Candle` was never constructed; the feature was pulling the policy-banned `zip` 8.6.0 into every `--all-features` build for no reason). `trustformers-tokenizers`, `trustformers`, and `trustformers-models` all migrate `serde_yaml` → `serde_yaml_ng` (the original is archived upstream, RUSTSEC-2024-0320); the deprecated, workspace-excluded `trustformers-c` is the one remaining consumer of the old crate.
- **`trustformers-serve`'s `resource_manager/` placeholder tree is deleted** (11 files, 5,972 lines) rather than fixed in place — it fabricated network ports (`vec![8080]` always), temp-directory paths (never actually created), database connection ids (synthesized, no real connection), GPU device stats (echoed indices plus a constant 8192MB), and monitoring efficiency (a constant `0.75`). The unprefixed `ResourceManagementSystem` public name now resolves directly to the real, tested `resource_management` module (`ModularResourceManagementSystem` kept as a compatibility type alias), with 4 new regression tests guarding the unprefixed surface.
- **`trustformers-models::deepseek_v2`**'s checkpoint loader now returns a documented structured error instead of reporting fake success on any non-empty input buffer. **`trustformers-models::s4`**'s loader — which validated and skipped every tensor while still returning `Ok(())` — is deleted outright rather than patched.
- **`trustformers-py`** pipelines: the `text-generation` pipeline's `format!("{text} [Generated continuation]")` / hardcoded `score: 0.95` fabrication is gone, replaced by real inference through the shared decoding core above; `token-classification` and `question-answering` pipelines now refuse construction with a structured `NotImplementedError` (naming the real cause: this crate's tokenizers don't yet produce a character-level offset mapping) instead of inventing a `B-PER`/`"John"` entity or an `"Example answer"`/`score: 0.85` response. `AutoTokenizer::from_pretrained`/`save_pretrained` now round-trip for real instead of fabricating a successful load.
- **`trustformers-mobile`**: real `sysinfo`-backed metric collection replaces three sets of invented per-platform constants, and `MobileMetricsSnapshot`'s metric families are now `Option` rather than always-populated; bottleneck detection, alerting, and health scoring — previously structurally incapable of reporting anything real — now work; `battery.rs`'s `power_consumption_mw` is read from a real `power_supply` sysfs value (`power_now`) or `None`, replacing three hardcoded constants (`2500.0`/`2200.0`/`1800.0` mW); cache-hit-rate suggestions now report a measured rate instead of a predicted gain; `DeviceInfo::detect_current_device` measures the host instead of returning a fixed spec; three functions were renamed to the algorithms they actually implement; a LoRA-A initialization fix (scaled by `1/sqrt(fan_in)`) resolved two genuinely flaky loss tests; `react-native-plugin/` is renamed `react-native-example/` (it ships a usage example only, no installable package — the old name read as a publishable package).
- **`trustformers-serve` performance/analytics honesty, continued**: real system telemetry backs several paths that previously had none (structured errors where the controller genuinely can't measure something); 15 fabricated hardware measurements now report `MeasurementUnavailable` instead of a made-up number; a coordination shim now delegates to the real detectors or reports nothing, instead of fabricating; workflow steps record actual failure instead of every step being stamped `Completed` regardless of outcome; alert conditions that could never fire are fixed (plus one that was firing on the wrong condition); dashboard HTTP endpoints stop serving empty data as if it were real; an async-result endpoint stopped telling every caller their work was "in progress" regardless of actual state; silent strategy fallbacks now log instead of failing silently; `MockTokenRequest` is renamed `TokenRequest` (it was never a mock). A file-level constant-folding bug that had been mistaken for a placeholder was a real miscompilation, not fabricated logic, and is fixed as such.

### Removed
- Orphaned root `src/` tree (899 lines) with no `[package]` section — compiled by nothing, tracked in git for no reason.
- Two tracked compiled Mach-O binaries and their orphan test artifacts that would otherwise have shipped inside a published crate.
- `message_queue.rs`'s `impl_placeholder_backend!`-generated RabbitMQ/Redis/NATS/SQS backends.
- ~4,800 lines of dead and duplicate scaffolding in `trustformers-mobile`, including a random-number fabrication factory in the mobile testing framework and a device farm that invented device catalogues and cross-device reports rather than executing anything.
- 24 no-op mesh methods in `trustformers-serve` that logged integrations which never actually happened.
- Dead scaffolding in the `trustformers` umbrella crate: `DocumentRegion`/`RegionType`, a `StateManager` struct, and a `performance_tracker` field — deleted rather than kept unused, and `generate_with_beam_search`/`format_cache_cleanup_start_message` are now correctly feature-gated (`t5`, `hub`) instead of always compiled regardless of feature selection.
- The last `#![allow(dead_code)]` in `trustformers-serve`'s performance_optimizer path, and the fake GPU inventory it was hiding.

### Fixed
- `deny.toml` previously failed to deserialize at all under current cargo-deny, and its `[bans]` deny-list was an empty comment even when it did parse — meaning `cargo deny check bans` was either erroring outright or passing vacuously with no COOLJAPAN-banned crate actually enforced. Both are fixed; the ban list now covers openblas/bincode/rustfft/rusqlite/z3/zip/flate2/zstd/bzip2/lz4-family crates with their oxi\* replacements, and `cargo deny check bans` passes for real.
- `sysinfo` version unified at 0.39.6 workspace-wide (previously split across 0.39.5 and other pins, an avoidable source of duplicate-version warnings).
- ~50 member-crate dependency version pins hoisted to `[workspace.dependencies]` with `workspace = true`; `rust-version.workspace = true` is now set in 10 of 12 member manifests (the 2 exceptions — `trustformers-py` and the deprecated `trustformers-c` — use a literal `rust-version = "1.89"` matching `clippy.toml`'s `msrv`, so there is no longer an MSRV mismatch between them).
- `trustformers-js` is listed in root `Cargo.toml`'s `exclude`, so `cargo` commands no longer silently ignore its presence.
- Root `Cargo.toml`'s 7 orphaned `[workspace.dependencies]` entries left behind by the Azure/GCP/Lambda SDK removal above (`azure_core`, `azure_identity`, `azure_mgmt_machinelearningservices`, `azure_mgmt_web`, `google-cloud-functions-v2`, `google-cloud-gax`, `lambda_runtime`) are deleted, closing the `workspace_dependency_table_has_no_unused_entries` hygiene-gate failure this same cleanup introduced.
- 6 files that exceeded the workspace's 2,000-line policy limit are split into directory modules, each file now under 2,000 lines: `trustformers-mobile/src/inference/`, `trustformers-training/src/data_pipeline/`, `trustformers-mobile/src/profiler/`, `trustformers-training/src/auto_parallelism/`, `trustformers-mobile/src/crash_reporter/`, `trustformers/src/hub_offline_packs/`.
- A sibling-package-induced build break in `trustformers-mobile` (a one-line path fix) and a pre-existing broken doctest in the same crate are both fixed.

### Known issues introduced or exposed this cycle (tracked in `TODO.md`)
- `cargo deny check advisories` fails with 7 findings (an unmaintained `opentelemetry-jaeger` and `paste`, an `h2` vulnerability, an RSA timing side-channel, two `rustls` name-constraints vulnerabilities, and a `rustls-webpki` CRL-parsing panic), all reached through `trustformers-serve`'s `aws-sdk-{sqs,sagemaker,sagemakerruntime,lambda,cloudwatch}` dependencies' `rustls 0.21`/`rustls-webpki 0.101.7` stack. Not yet resolved.
- `cargo check -p trustformers-models --features metal,gpt2` fails with 15 compiler errors: `trustformers-core`'s new `MetalBufferHandle` (see Added, above) is not yet wired into `trustformers-models/src/gpt2/model/{model_blocks.rs,model_core.rs,model_ops.rs}`, which still constructs/reads `MetalTensorData` against the old bare-`BufferId` shape. Not part of the default-feature workspace build, so not caught by the green baseline above.
- `examples/server/Cargo.toml` has the same defect `examples/grpc-server/Cargo.toml` had before this cycle's fix: it declares `version.workspace = true`/`edition.workspace = true` while being excluded from the repository workspace and defining no `[workspace]` table of its own, so those keys can't resolve when built standalone.
- Three files not previously flagged now exceed the 2,000-line policy limit, all in the `trustformers` umbrella crate: `src/hub_ui.rs` (2026), `src/hub.rs` (2013), `src/pipeline/conversational/summarization.rs` (2001).
- `trustformers-core/src/kernels/xla_impl.rs` and `oneapi_impl.rs` are honest at the method level (every call returns a structured "no runtime linked" error, never a fabricated result) but their module-level doc comments, unlike `riscv_impl.rs`'s, don't disclose up front that nothing is actually linked.
- `trustformers-py::training::PyTrainer::train()` is still fabricated: its own source comment reads "In a real implementation, we'd run the training loop here / For now, return a mock training result", and it unconditionally returns `train_loss: 0.5`/`total_steps: 1000` regardless of input. `PyTrainer`/`PyTrainingArguments` are real, registered Python classes, so this is reachable from Python today. Outside this cycle's `py-pipelines` scope (pipelines, not training); not yet fixed.
- `trustformers-tokenizers/docs/migration/tiktoken-migration.md` described a large fictional API surface (batching, caching, chat-completion templating, cost estimation, production monitoring, shadow-testing helpers — none of which exist on the real `TiktokenTokenizer`); rewritten this cycle to document only the real API. The other five migration guides plus the migration `README.md` carry smaller amounts of the same problem and now carry a dated accuracy banner rather than a full rewrite.

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
