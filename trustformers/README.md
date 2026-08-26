# trustformers

**Version:** 0.2.1 | **Status:** Alpha | **Updated:** 2026-08-24

Main integration crate providing high-level APIs, pipelines, and Hugging Face Hub integration for the TrustformeRS ecosystem.

## Current State

This crate serves as the **primary entry point** for users, offering HuggingFace-compatible APIs for common NLP tasks. It includes comprehensive pipeline implementations, auto model classes, and integration points with the Hugging Face Model Hub.

- **SLoC:** 123,252 (`tokei`, verified 2026-08-24 — up from ~109,369 on 2026-07-01)
- **Tests:** ~2,261 as of 2026-07-01, not independently re-run this pass — see the workspace root `README.md`/`TODO.md` for the current baseline (21,370 passed / 41 skipped / 0 failed workspace-wide, default features, 2026-08-26)
- **File-size policy:** `src/hub_offline_packs.rs` (previously 2,009 lines) is now split into a 6-file `src/hub_offline_packs/` directory module. Three files not previously flagged now exceed the 2,000-line policy: `src/hub_ui.rs` (2,026), `src/hub.rs` (2,013), `src/pipeline/conversational/summarization.rs` (2,001) — see `TODO.md`.
- **Doctests:** 5 passed, 164 ignored by design (see [Testing](#testing))
- **Public API (prelude):** 76 exports under default features (`bert` + `async`); 83 with `hub` also enabled
- **Pipeline modules:** 38 task-specific pipelines, plus 6 execution-backend integrations and 10 execution-optimization/composition modules (54 `pub mod` declarations total under `src/pipeline/`) — 10 of the 38 were mounted in 0.2.0 and are mock pipelines pending real backends (see [Pipeline API](#pipeline-api))
- **Public API surface:** ~3,177 `pub` items (fn/struct/enum/trait, including impl-block methods) across `src/`
- **Stubs remaining:** 0 reachable — a static grep finds 12 occurrences of `todo!()`/`unimplemented!()` in `src/`, but every one is either a string literal emitted by the code-generation pipeline (sample "generated code" text) or a hidden setup line inside an `ignore`d doctest example; none are on a reachable production code path

## Features

### Pipeline API

Pipelines are organized into task-specific implementations, backend integrations, and cross-cutting execution infrastructure:

**Text / NLP pipelines**
- **Text Generation**, **Text Classification**, **Token Classification** (NER/POS), **Question Answering**, **Fill-Mask**, **Summarization** (+ **Multi-Doc Summarization**), **Translation** (+ **Enhanced Translation** with language/script detection), **Code Generation**
- *Mounted in 0.2.0 (mock — see note below):* **Document Classification** (long documents via overlapping-chunk aggregation), **Feature Extraction** (dense embeddings; CLS/mean/max/weighted-mean pooling), **Table Question Answering** (TAPAS/TaPEx-style, answers questions over CSV-like tables)

**Retrieval & long-context**
- **RAG** (TF-IDF and BM25 retrievers) and **Advanced RAG**
- **Mamba-2** state-space pipeline for very long sequences

**Vision / multimodal / audio** (feature-gated: `vision`, `audio` — tagged individually)
- Image Classification, Object Detection, Depth Estimation, Optical Flow, Pose Estimation, Mask Generation (SAM-style point/box prompts), Image-to-Text (`vision`), Visual Question Answering (`vision`), MultiModal (CLIP-style), Document Understanding, Audio Classification, Speech-to-Text (`audio`), Text-to-Speech (`audio`)
- *Mounted in 0.2.0 (mock — see note below; not gated behind `vision`/`audio`, compiled unconditionally):* Image Segmentation (SegFormer/Mask2Former-style), Video Classification (VideoMAE/TimeSformer-style), Visual Grounding (GroundingDINO-style free-text phrase grounding), Speech Recognition (Whisper-compatible; adds translation-to-English and word/sentence timestamps — distinct from the existing Speech-to-Text pipeline above), Zero-Shot Audio Classification (CLAP-style; classifies audio against arbitrary natural-language candidate labels)

**Generative audio / image** (mounted in 0.2.0, mock — see note below; not gated behind `vision`/`audio`, compiled unconditionally)
- **Audio Generation**: AudioLDM/MusicGen-style text-to-audio
- **Text-to-Image**: Stable Diffusion XL / DALL-E-style text-to-image

**Conversational** (feature `async`)
- **ConversationalPipeline** with dedicated streaming, memory, safety, and reasoning submodules

**Meta / composition pipelines**
- **ComposedPipeline**: Sequential multi-stage pipelines
- **EnsemblePipeline**: Aggregated predictions from multiple models
- **PipelineChain**: Chained pipeline execution (`add_stage(...)` builder)
- **PipelineComposer**: Dynamic pipeline construction
- **AdaptiveInferenceEngine**: Runtime-adaptive inference wrapper

**Execution backends**
- ONNX Runtime, TensorRT, OpenVINO, CoreML, Metal, and a pluggable custom-backend registry

**Execution optimization**
- Adaptive/dynamic batching, JIT compilation, early-exit, mixture-of-depths, speculative decoding, and real-time/backpressure-aware streaming

> **Mock pipelines pending real backends (mounted in 0.2.0):** `audio_generation`, `document_classification`, `feature_extraction`, `image_segmentation`, `speech_recognition`, `table_question_answering`, `text_to_image`, `video_classification`, `visual_grounding`, and `zero_shot_audio_classification` existed as unwired source files before 0.2.0. They now compile into the crate, but — confirmed by reading each implementation — every one currently returns **deterministic, hash- or heuristic-derived mock output** (e.g. `generate_mock_waveform`, `mock_embed`, `mock_score`, djb2-hash-seeded pixel/embedding synthesis) rather than running real model inference, the same pattern already used by the mock `AutoModelForImageClassification`/`AutoModelForAudioClassification` wrappers. These ten are plain structs with their own methods (`generate`, `classify`, `segment`, `ground`, `answer`, `transcribe`, `extract`, ...) — they do **not** implement the common `Pipeline` trait and are **not** reachable through the string-based `pipeline(task, ...)` factory below; construct them directly via their module path, e.g. `trustformers::pipeline::text_to_image::TextToImagePipeline::new(TextToImageConfig::default())`.

All other pipelines implement a common `Pipeline` trait (`__call__`, `batch`, `adaptive_batch`) plus an `AsyncPipeline` trait under the `async` feature. Batched and async execution, and CPU/GPU device placement are supported throughout.

### Safety Filtering

- **SafetyFilter** with `ExtendedSafetyConfig` (boxed to prevent stack overflow)
- **EnhancedSafetyFilter** with multi-risk assessment:
  - Toxicity detection
  - Hate speech classification
  - Personal information detection
  - Violence content filtering
  - Adult content filtering
  - Harassment detection
  - Bias assessment

### Auto Classes

Automatic model/tokenizer/config selection, mirroring HuggingFace `transformers` conventions:

- **AutoModel** / **AutoConfig**: Base model and config auto-detection
- **AutoModelForSequenceClassification**, **AutoModelForTokenClassification**, **AutoModelForQuestionAnswering**, **AutoModelForCausalLM**, **AutoModelForMaskedLM**, **AutoModelForSeq2SeqLM**: Task-specific model wrappers
- **AutoTokenizer** (type alias for `TokenizerWrapper`): Automatic tokenizer selection
- **AutoProcessor**: Modality-aware input processing
- **AutoFeatureExtractor** / **AutoDataCollator** / **AutoMetric** / **AutoOptimizer**: Automatic feature extraction, batching/collation, evaluation metrics, and optimizer selection (`src/auto/`)
- **AutoModelForImageClassification** / **AutoModelForAudioClassification** / **AutoModelForObjectDetection** / **AutoModelForImageSegmentation** (the latter two added in 0.2.0): thin wrappers around the corresponding pipeline (`from_pretrained`, `from_local`, plus a task method — `detect()`/`segment()`/etc.). Not re-exported at the crate root like the six wrappers above — reach them via `trustformers::automodel_tasks::{AutoModelForObjectDetection, AutoModelForImageSegmentation, ...}`. **Mock notice:** all four wrap deterministic placeholder pipelines pending real detection/segmentation/classification heads; their doc comments say so explicitly — do not treat their output as real inference.

### Fine-tuning

`trustformers::finetuning` — mounted in 0.2.0 (previously an unwired source file); a real, tested implementation, **not** a mock:

- **LoRA** (`LoraConfig`, `LoraConfigBuilder`, `LoraLinear`, `LoraBias`): low-rank adaptation with a fluent builder (`rank`, `alpha`, `dropout`, `target_modules`, `merge_weights`, `bias`), plus `LoraLinear::merge_weights`/`unmerge_weights` to fold the adapter into the base weights and `trainable_parameters()` for optimizer wiring.
- **Bottleneck adapters** (`AdapterConfig`, `AdapterActivation`, `BottleneckAdapter`): Houlsby et al. (2019)-style down-project → activation → up-project + residual adapters.

```rust
use trustformers::finetuning::{AdapterConfig, BottleneckAdapter, LoraConfig, LoraLinear};

// LoRA: 8-rank adapter for a 768 → 768 attention projection
let lora_cfg = LoraConfig::builder().rank(8).alpha(16.0).build()?;
let lora_layer = LoraLinear::new(768, 768, 8, &lora_cfg)?;

// Bottleneck adapter for a BERT-base hidden layer
let adapter_cfg = AdapterConfig { hidden_size: 768, bottleneck_size: 64, ..Default::default() };
let adapter = BottleneckAdapter::new(adapter_cfg)?;
```

Prefix-tuning, prompt-tuning, and p-tuning v2 are not implemented (see `TODO.md`).

### Infrastructure

- **MemoryPool**: Efficient tensor memory management
- **ConfigurationManager**: Centralized configuration handling, diffing, and migration
- **EnhancedProfiler**: Performance profiling and tracing
- **ValidationManager**: Input/output validation
- **BenchmarkSuite**: Built-in benchmarking utilities (re-exported from `trustformers-core`)
- **ModelDiagnostics**: Rich diagnostics — weight-norm checks, activation stats, gradient-flow checks, attention-entropy checks, dead-neuron/weight-collapse detection, and summary reporting (`src/diagnostics/`)
- **Evaluation bridge**: BLEU, ROUGE-N/L, token-F1, exact-match, and perplexity metrics adapted to the `Metric` trait (`src/evaluation/`)
- **VersionedCache** (`trustformers::cache`, mounted in 0.2.0): a generic `VersionedCache<K, V>` with TTL expiry, version-tagged entries (`is_valid_version`), configurable eviction (`CacheEvictionPolicy::{Lru, Lfu, Ttl, Size}`), and hit/miss statistics (`VersionedCacheStats`) — a real, tested implementation, not a mock.
- **ParallelWeightLoader** (`trustformers::loading`, mounted in 0.2.0): sharded/parallel model-weight loading (`load_sharded_directory`) with a progress-callback hook (`with_progress`, `LoadingProgress::pct_complete`) and throughput statistics (`LoadingStats`), plus a `load_model_parallel(path, config)` convenience function — a real, tested implementation, not a mock.

### Hugging Face Hub Integration

Core Hub utilities (offline packs, model cards, differential updates, P2P) are compiled unconditionally. Actual **remote downloads** require the optional **`hub`** feature (pulls in `reqwest`; not enabled by default):

- **Model downloading** (`hub::download_model`) with progress tracking and resumable/parallel chunked downloads
- **Caching** (`hub::get_cache_dir`, `hub::is_cached`) for offline use
- **Authentication** via `HubOptions::token` for private models
- **Revision/branch** selection via `HubOptions::revision`
- **Model card** parsing (`ModelCard`, `hub_model_card`)
- **Hub mirror** support (`HubMirror`, feature `hub`) and a Hub browser UI (`HubUiServer`, feature `async`)

## Usage Examples

### Pipeline Usage

```rust
use trustformers::{pipeline, Result};

fn main() -> Result<()> {
    // Create a pipeline: task, optional model name, optional PipelineOptions
    let classifier = pipeline("sentiment-analysis", None, None)?;
    let result = classifier.__call__("I love using Rust for ML!".to_string())?;
    println!("{:?}", result);

    // Batched inference
    let texts = vec!["Great!".to_string(), "Not great.".to_string()];
    let batch_results = classifier.batch(texts)?;
    println!("{:?}", batch_results);

    // Text generation
    let generator = pipeline("text-generation", Some("gpt2"), None)?;
    let output = generator.__call__("Once upon a time".to_string())?;
    println!("{:?}", output);

    Ok(())
}
```

*(See `examples/basic_pipeline.rs` for the full, compiling version of this example.)*

### Auto Classes Usage

```rust
use trustformers::{AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Tokenizer};

let model_name = "bert-base-uncased";

// Tokenizer: single-argument `encode`, no `add_special_tokens` flag
let tokenizer = AutoTokenizer::from_pretrained(model_name)?;
let encoding = tokenizer.encode("Hello, world!")?;
println!("Token IDs: {:?}", encoding.input_ids);

// Model: task-specific Auto* wrappers take the label count explicitly
let config = AutoConfig::from_pretrained(model_name)?;
let model = AutoModelForSequenceClassification::from_pretrained(model_name, /* num_labels */ 2)?;
```

### Pipeline Composition

```rust
use trustformers::PipelineChain;

// Chain pipelines sequentially via the builder, then invoke like any other Pipeline
let chain = PipelineChain::new()
    .add_stage(summarization_pipeline)
    .add_stage(classification_pipeline);

let result = chain.__call__("Very long document text...".to_string())?;
```

### Hub Integration

Requires the `hub` feature (`trustformers = { version = "0.2.1", features = ["hub"] }`):

```rust
use trustformers::hub::{download_model, HubOptions};

// Download with default options (latest "main" revision, on-disk cache)
let model_path = download_model("gpt2", None)?;

// Or with explicit options
let options = HubOptions {
    revision: Some("main".to_string()),
    token: Some("hf_...".to_string()),
    ..Default::default()
};
let model_path = download_model("meta-llama/Llama-2-7b-hf", Some(options))?;
```

## Architecture

```
trustformers/
├── src/
│   ├── lib.rs                  # Crate root: feature-gated re-exports + prelude
│   ├── automodel.rs            # AutoModel, AutoConfig
│   ├── automodel_tasks.rs      # AutoModelFor{CausalLM,MaskedLM,SequenceClassification,...}
│   ├── auto/                   # AutoFeatureExtractor, AutoDataCollator, AutoMetric, AutoOptimizer
│   │   ├── data_collators/
│   │   ├── feature_extractors/
│   │   ├── metrics/
│   │   └── optimizers/
│   ├── pipeline/                # 54 pub modules: task pipelines, composition, backends, optimization
│   │   ├── conversational/      # ConversationalPipeline (feature = "async")
│   │   ├── ensemble/            # EnsemblePipeline
│   │   ├── onnx_backend.rs, tensorrt_backend.rs, openvino_backend.rs,
│   │   │   coreml_backend.rs, metal_backend.rs, custom_backend.rs
│   │   ├── audio_generation.rs, document_classification.rs, feature_extraction.rs,
│   │   │   image_segmentation.rs, speech_recognition.rs, table_question_answering.rs,
│   │   │   text_to_image.rs, video_classification.rs, visual_grounding.rs,
│   │   │   zero_shot_audio_classification.rs   # mounted in 0.2.0; mock pending real backends
│   │   └── ... (text/vision/audio/RAG pipelines, adaptive/dynamic batching,
│   │            jit_compilation, early_exit, mixture_of_depths,
│   │            speculative_decoding, streaming)
│   ├── hub.rs, hub_upload.rs, hub_model_card.rs, hub_offline_packs.rs,
│   │   hub_p2p.rs, hub_differential.rs   # unconditional Hub utilities
│   ├── hub_local_mirror.rs      # feature = "hub" (networking)
│   ├── hub_ui.rs                # feature = "async"
│   ├── diagnostics/             # ModelDiagnostics
│   ├── evaluation/              # BLEU / ROUGE / F1 / perplexity bridge
│   ├── config_management.rs     # ConfigurationManager
│   ├── enhanced_profiler.rs     # EnhancedProfiler
│   ├── memory_pool.rs           # MemoryPool
│   ├── processor.rs, profiler.rs, training_utils.rs, validation.rs, zero_copy.rs
│   ├── cache/                   # VersionedCache (TTL/LRU/LFU/size eviction) — mounted in 0.2.0
│   ├── finetuning/              # LoraConfig/LoraLinear, AdapterConfig/BottleneckAdapter — mounted in 0.2.0
│   ├── loading/                 # ParallelWeightLoader, load_model_parallel — mounted in 0.2.0
```

## Pipeline Features

### Advanced Generation
- **Sampling strategies**: Top-k, top-p, temperature
- **Beam search**: With length penalty and early stopping
- **Streaming generation**: Token-by-token async output (`pipeline::streaming`)
- **Speculative decoding**: Draft-and-verify acceleration (`pipeline::speculative_decoding`)
- **Batch generation**: Efficient multi-prompt processing

### Pipeline Options
```rust
use trustformers::pipeline::{PipelineOptions, Device};

let options = PipelineOptions {
    device: Some(Device::Gpu(0)),
    batch_size: Some(32),
    max_length: Some(512),
    ..Default::default()
};

let text_gen = trustformers::pipeline::pipeline("text-generation", None, Some(options))?;
```

## Performance

### Optimization Features
- **Dynamic / adaptive batching**: Automatic batch-size optimization (`AdaptiveBatchOptimizer`, `DynamicBatcher`)
- **MemoryPool** / **AdvancedLRUCache**: Efficient tensor and pipeline-output caching and reuse
- **JIT pipeline compilation**: Hardware-aware compilation with anomaly detection (`PipelineJitCompiler`)
- **Early-exit** and **mixture-of-depths**: Conditional compute for latency-sensitive inference

> Note: the previous README included a fixed throughput benchmark table (e.g. "850 samples/s on RTX 4090"). That table was not re-verified against current hardware/code and has been removed rather than repeated unverified; see `tests/performance_benchmarks.rs` and `BenchmarkSuite` for reproducible, up-to-date numbers.

## Supported Models

This crate's own `Cargo.toml` feature-gates direct re-exports for:
- **BERT**, **RoBERTa**, **ALBERT** (via `AutoModel`/`AutoConfig`; ALBERT has no direct top-level type re-export, only Auto* access), **GPT-2**, **GPT-Neo**, **GPT-J**, **T5**

The underlying `trustformers-models` crate (re-exported as `trustformers::models`) implements a much larger set of architectures (LLaMA family, Mistral, Gemma/Gemma2, Qwen/Qwen2.5, Falcon, Mamba, RWKV, CLIP, BLIP-2, LLaVA, and more) behind its own feature flags; those are reachable via `trustformers::models::*` once the corresponding `trustformers-models` feature is enabled in the workspace, or by depending on `trustformers-models` directly. This crate does not yet expose its own convenience feature flags or top-level re-exports for that wider set — see `TODO.md`.

## Testing

- ~2,261 tests covering pipeline correctness and edge cases (part of a workspace-wide 18,102 passed / 0 failed / 119 skipped, 0 clippy warnings, 0 rustdoc warnings)
- Auto class functionality tests
- Hub integration tests
- Generation strategy tests
- Safety filter tests
- Performance benchmarks via `BenchmarkSuite` (`tests/performance_benchmarks.rs`)
- **Doctests:** 5 passed, 164 ignored. Almost all `///`/`//!` examples are intentionally marked `rust,ignore` because they demonstrate pipeline/model-loading flows that require downloading real weights from the Hugging Face Hub; they are illustrative only and are not compiled or executed by `cargo test --doc`.

## Known Limitations (Alpha)

- `src/finetuning/` (LoRA + bottleneck adapters), `src/cache/` (versioned cache), and `src/loading/` (parallel model loader) were mounted into `lib.rs` in 0.2.0 and are now part of the compiled crate and public API (`trustformers::{finetuning, cache, loading}`) — real, tested implementations, not mocks. Remaining refinement: default LoRA rank/target-layer choices, and benchmarking the parallel loader/cache against concrete performance targets (see `TODO.md`).
- Ten pipeline source files under `src/pipeline/` (`audio_generation.rs`, `document_classification.rs`, `feature_extraction.rs`, `image_segmentation.rs`, `speech_recognition.rs`, `table_question_answering.rs`, `text_to_image.rs`, `video_classification.rs`, `visual_grounding.rs`, `zero_shot_audio_classification.rs`) were also mounted in 0.2.0 and now compile in, but every one is a **mock**: it returns deterministic, hash- or heuristic-derived output rather than running real model inference, pending real model backends (see the [Pipeline API](#pipeline-api) note above).
- `AutoModelForObjectDetection` and `AutoModelForImageSegmentation` (added in 0.2.0), like the existing `AutoModelForImageClassification`/`AutoModelForAudioClassification`, wrap mock pipelines for the same reason.
- `examples/conversational_ai.rs.disabled` is currently disabled pending an API rework (the two previously-disabled `#[cfg(test_disabled)]` test modules referenced in earlier revisions of this document have since been re-enabled as ordinary `#[cfg(test)]` modules).
- Hub **downloads** require the optional `hub` feature (not enabled by default) plus an internet connection; without it, only local/cached model loading works.
- Large models require significant disk space; caching may use substantial disk space.

## License

Apache-2.0
