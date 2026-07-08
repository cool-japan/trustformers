# trustformers

**Version:** 0.2.0 | **Status:** Alpha | **Updated:** 2026-07-02

Main integration crate providing high-level APIs, pipelines, and Hugging Face Hub integration for the TrustformeRS ecosystem.

## Current State

This crate serves as the **primary entry point** for users, offering HuggingFace-compatible APIs for common NLP tasks. It includes comprehensive pipeline implementations, auto model classes, and integration points with the Hugging Face Model Hub.

- **SLoC:** ~109,369 (Rust code lines, via `tokei`)
- **Tests:** ~2,261 passing (part of a workspace-wide 18,102 passed / 0 failed / 119 skipped, 0 clippy warnings, 0 rustdoc warnings — verified 2026-07-01)
- **Doctests:** 5 passed, 164 ignored by design (see [Testing](#testing))
- **Public API (prelude):** 76 exports under default features (`bert` + `async`); 83 with `hub` also enabled
- **Pipeline modules:** 28 task-specific pipelines, plus 6 execution-backend integrations and 10 execution-optimization/composition modules (44 `pub mod` declarations total under `src/pipeline/`)
- **Public API surface:** ~3,177 `pub` items (fn/struct/enum/trait, including impl-block methods) across `src/`
- **Stubs remaining:** 0 reachable — a static grep finds 12 occurrences of `todo!()`/`unimplemented!()` in `src/`, but every one is either a string literal emitted by the code-generation pipeline (sample "generated code" text) or a hidden setup line inside an `ignore`d doctest example; none are on a reachable production code path

## Features

### Pipeline API

Pipelines are organized into task-specific implementations, backend integrations, and cross-cutting execution infrastructure:

**Text / NLP pipelines**
- **Text Generation**, **Text Classification**, **Token Classification** (NER/POS), **Question Answering**, **Fill-Mask**, **Summarization** (+ **Multi-Doc Summarization**), **Translation** (+ **Enhanced Translation** with language/script detection), **Code Generation**

**Retrieval & long-context**
- **RAG** (TF-IDF and BM25 retrievers) and **Advanced RAG**
- **Mamba-2** state-space pipeline for very long sequences

**Vision / multimodal / audio** (feature-gated: `vision`, `audio`)
- Image Classification, Object Detection, Depth Estimation, Optical Flow, Pose Estimation, Mask Generation (SAM-style point/box prompts), Image-to-Text (`vision`), Visual Question Answering (`vision`), MultiModal (CLIP-style), Document Understanding, Audio Classification, Speech-to-Text (`audio`), Text-to-Speech (`audio`)

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

All pipelines implement a common `Pipeline` trait (`__call__`, `batch`, `adaptive_batch`) plus an `AsyncPipeline` trait under the `async` feature. Batched and async execution, and CPU/GPU device placement are supported throughout.

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

### Infrastructure

- **MemoryPool**: Efficient tensor memory management
- **ConfigurationManager**: Centralized configuration handling, diffing, and migration
- **EnhancedProfiler**: Performance profiling and tracing
- **ValidationManager**: Input/output validation
- **BenchmarkSuite**: Built-in benchmarking utilities (re-exported from `trustformers-core`)
- **ModelDiagnostics**: Rich diagnostics — weight-norm checks, activation stats, gradient-flow checks, attention-entropy checks, dead-neuron/weight-collapse detection, and summary reporting (`src/diagnostics/`)
- **Evaluation bridge**: BLEU, ROUGE-N/L, token-F1, exact-match, and perplexity metrics adapted to the `Metric` trait (`src/evaluation/`)

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

Requires the `hub` feature (`trustformers = { version = "0.2.0", features = ["hub"] }`):

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
│   ├── pipeline/                # 44 pub modules: task pipelines, composition, backends, optimization
│   │   ├── conversational/      # ConversationalPipeline (feature = "async")
│   │   ├── ensemble/            # EnsemblePipeline
│   │   ├── onnx_backend.rs, tensorrt_backend.rs, openvino_backend.rs,
│   │   │   coreml_backend.rs, metal_backend.rs, custom_backend.rs
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
│   ├── cache/, finetuning/, loading/   # implemented but NOT wired into lib.rs yet — see TODO.md
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

- `src/finetuning/` (LoRA + bottleneck adapters, ~1,180 lines) and `src/cache/`, `src/loading/` (versioned cache, parallel model loader, ~1,650 lines) are fully implemented but **not yet wired into `lib.rs`** — they are not part of the compiled crate or public API today (see `TODO.md`).
- A handful of pipeline source files under `src/pipeline/` (e.g. `audio_generation.rs`, `image_segmentation.rs`, `video_classification.rs`, `text_to_image.rs`) exist as drafts but are not yet declared in `pipeline/mod.rs`, so they are not compiled in.
- Two test modules are disabled (`#[cfg(test_disabled)]`) pending a rewrite after an internal API rename (conversational config presets; feature-extractor error variants).
- `examples/conversational_ai.rs.disabled` is currently disabled pending the same API rework.
- Hub **downloads** require the optional `hub` feature (not enabled by default) plus an internet connection; without it, only local/cached model loading works.
- Large models require significant disk space; caching may use substantial disk space.

## License

Apache-2.0
