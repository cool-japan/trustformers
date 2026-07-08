# trustformers TODO List

**Version:** 0.2.0 | **Status:** Alpha | **Updated:** 2026-07-06

## Overview

The `trustformers` crate is the main integration crate providing high-level APIs, pipelines, and HuggingFace Hub integration. It re-exports functionality from all specialized crates and provides a unified user-facing API.

**Key Responsibilities:**
- High-level API (AutoModel, AutoTokenizer, pipeline)
- HuggingFace Hub integration
- Pre-built pipeline functions
- Model discovery and download
- Unified documentation
- Example applications

---

## Current Status

### Implementation Status (originally 2026-03-21, re-verified against source 2026-07-01)

- [x] **HIGH-LEVEL API** - Complete (AutoModel, AutoTokenizer, AutoConfig, AutoModelFor*)
- [x] **HUB INTEGRATED** - Hub utilities (offline packs, model cards, differential updates, P2P) compile unconditionally; actual remote *downloads* require the optional `hub` feature (not a default feature)
- [x] **PIPELINE COMPLETE** - 28 task-specific pipeline modules implemented (incl. RAG + Advanced RAG + Enhanced Translation), plus 6 execution-backend integrations and 10 composition/execution-optimization modules — see `src/pipeline/mod.rs`
- [x] **AUTO CLASSES** - Auto* classes for model/tokenizer/config loading, plus `AutoProcessor`, `AutoFeatureExtractor`, `AutoDataCollator`, `AutoMetric`, `AutoOptimizer`
- [x] **PIPELINE COMPOSITION** - ComposedPipeline, EnsemblePipeline, PipelineChain, PipelineComposer
- [x] **SAFETY FILTERING** - SafetyFilter, ExtendedSafetyConfig, EnhancedSafetyFilter (multi-risk)
- [x] **ASYNC STREAMING** - Token-by-token async streaming for generation pipelines (feature `async`)
- [x] **INFRASTRUCTURE** - MemoryPool, ConfigurationManager, EnhancedProfiler, HubMirror (feature `hub`), ValidationManager, BenchmarkSuite
- [x] **PRELUDE EXPORTS** - 76 public API exports in `prelude` under default features (83 with `hub` also enabled) — see Metrics below
- [x] **STUB CLEANUP** - No reachable stubs in production code paths; see "Minor Stubs" below for the full accounting of the 12 static `todo!()`/`unimplemented!()` grep hits
- [x] **HUB UPLOAD EXTENDED** - HubUploadConfig, HubUploader extensions, HubUploadProgress, HubError, upload_model, upload_tokenizer (confirmed present in `hub_upload.rs`)
- [x] **MODEL CARDS EXTENDED** - ModelCardBuilder, ModelCardTemplate, ModelCardError, `to_yaml_frontmatter()`, `from_markdown()` (confirmed present in `hub_model_card.rs`)
- [x] **MODEL DIAGNOSTICS** - ModelDiagnostics, DiagnosticResult, DiagStatus, DiagnosticSummary, `check_weight_norms`, `check_activation_stats`, `check_gradient_flow`, `check_attention_entropy`, `detect_dead_neurons`, `detect_weight_collapse`, `report_summary` (confirmed present in `src/diagnostics/mod.rs`)
- [x] Wire finetuning/ + cache/ + loading/ into lib.rs (planned 2026-07-05)
  - Goal: expose ~2900 lines of complete, already-tested code (LoRA/adapter fine-tuning, versioned cache, parallel weight loader) currently invisible to the compiler.
  - Design: add exactly 3 lines to lib.rs: `pub mod cache; pub mod finetuning; pub mod loading;`. Each submodule's own mod.rs already wires its internals correctly.
  - Files: trustformers/src/lib.rs only.
  - Tests: cargo build --all-features; the pre-written #[cfg(test)] modules in versioned_cache.rs/parallel_loader.rs/adapter.rs/lora.rs run for the first time.
  - Risk: low — crate already blankets #![allow(dead_code, unused_variables, unused_imports, unused_assignments)].
- [ ] **MODEL SEARCH** - No `search_models`/Hub-search implementation exists anywhere in `src/`. A prior version of this document listed Hub model search as complete; that was incorrect and has been corrected here.

### Metrics (re-verified 2026-07-01)

- **SLoC:** ~109,369 (Rust code lines, via `tokei`; up from the previously recorded ~62,500 — reflects substantial growth in `src/pipeline/` and `src/auto/`)
- **Tests:** ~2,261 (part of a workspace-wide 18,102 passed / 0 failed / 119 skipped run; 0 clippy warnings, 0 rustdoc warnings)
- **Doctests:** 5 passed, 164 ignored (intentionally `rust,ignore` — see README.md Testing section)
- **Pipeline modules:** 44 `pub mod` declarations under `src/pipeline/` (28 task pipelines, 6 backends, 10 composition/optimization modules)
- **Public API exports (prelude):** 76 under default features (`bert` + `async`); 83 with `hub` also enabled
- **Total public API surface:** ~3,177 `pub` fn/struct/enum/trait items across `src/` (including impl-block methods; a narrower module-level-only count is 1,317)
- **Stubs remaining:** 0 reachable in production code. Static grep for `todo!()`/`unimplemented!()` finds exactly 12 hits in `src/`, all confirmed benign:
  - 1 in `pipeline/code_generation.rs` — a string literal the code-generation pipeline emits as *sample generated Rust code text* (not executed by this crate)
  - 1 in `auto/data_collators/mod.rs` — inside a `` ```rust,ignore `` doc example showing a user-implementable trait method body
  - 10 in `pipeline/conversational/streaming/pipeline.rs` — hidden (`# `-prefixed) setup lines inside `` ```rust,ignore `` doctest examples, e.g. `# let model = todo!();`

### Feature Coverage

- **API:** AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoProcessor
- **Pipelines:** TextGeneration, TextClassification, QuestionAnswering, TokenClassification, Summarization, MultiDocSummarization, Translation, EnhancedTranslation, FillMask, ConversationalPipeline (async), MultiModal, DocumentUnderstanding, RAG (TF-IDF + BM25), AdvancedRAG, CodeGeneration, Mamba2, MaskGeneration, OpticalFlow, PoseEstimation, AudioClassification, ImageClassification, ObjectDetection, DepthEstimation, ImageToText (vision), VisualQuestionAnswering (vision), SpeechToText (audio), TextToSpeech (audio) — 28 task pipelines total
- **Pipeline Composition:** ComposedPipeline, EnsemblePipeline, PipelineChain, PipelineComposer, AdaptiveInferenceEngine
- **Execution backends:** ONNX Runtime, TensorRT, OpenVINO, CoreML, Metal, custom backend registry
- **Execution optimization:** Adaptive/dynamic batching, JIT compilation, early-exit, mixture-of-depths, speculative decoding, streaming
- **Safety:** SafetyFilter (ExtendedSafetyConfig), EnhancedSafetyFilter (toxicity, hate speech, personal info, violence, adult content, harassment, bias)
- **Hub:** Model download/cache/auth (feature `hub`), mirror support (feature `hub`), Hub browser UI (feature `async`), model cards, offline packs, differential updates, P2P
- **Infrastructure:** MemoryPool, ConfigurationManager, EnhancedProfiler, HubMirror, ValidationManager, BenchmarkSuite, ModelDiagnostics, evaluation bridge (BLEU/ROUGE/F1/perplexity)

---

## 0.2.0 Release Scope

Two workspace-wide tracks touch this umbrella crate in 0.2.0: the **OxiCUDA GPU migration** (scirs2-core `gpu` → OxiCUDA backends, already integrated in trustformers-core behind `cuda`/`metal`) and the **PyTorch (tch) dependency removal**. Decision: delete the `tch` dependency and the `torch` feature entirely in 0.2.0 (workspace Cargo.toml:82, trustformers-core torch feature + ~40 lines of cfg arms, and the forwarder features in trustformers, trustformers-training, trustformers-c); do **not** adopt ToRSh as a replacement now — a P2 task (tracked in the root TODO.md) evaluates an optional `torsh-interop` feature in 0.3.x once torsh 0.2.0 ships on crates.io. Candle sub-decision: drop the unused candle-nn workspace dep now, keep the `candle` feature/variant through 0.2.0 (it is in every `full` set), and decide implement-vs-remove in 0.3.x.

### PyTorch (tch) dependency removal

- [ ] **[P0]** Remove the `torch` forwarder feature from the umbrella crate
  - Delete `torch = ["trustformers-core/torch", "trustformers-training/torch"]` (Cargo.toml:81-82) and update the `full` comment about excluding torch (:92-93). Zero code in `trustformers/src` is gated on the feature (verified: no `cfg(feature = "torch")` hits), so this is manifest-only. Must land in the same change as the trustformers-core torch removal — a forwarder referencing a deleted core feature fails Cargo feature resolution. Verify: `cargo check -p trustformers --features full` green.
  - Evidence: trustformers/Cargo.toml:81-82,92-93

### OxiCUDA GPU migration (scirs2-core gpu → OxiCUDA)

- [ ] **[P1]** Add a `metal` forwarding feature to the umbrella crate
  - The umbrella crate forwards cuda (`cuda = ["trustformers-models/cuda"]`, Cargo.toml:84) but has no metal forwarder, so users of the top-level trustformers crate cannot enable the oxicuda-metal path without depending on sub-crates directly. Add `metal = ["trustformers-models/metal"]` (models already forwards to trustformers-core/metal at trustformers-models/Cargo.toml:101, which pulls oxicuda-metal + oxicuda-backend). Verify: `cargo check -p trustformers --features metal` on macOS resolves and compiles the Metal backend.
  - Evidence: trustformers/Cargo.toml:84; trustformers-models/Cargo.toml:101; trustformers-core/Cargo.toml:121,149-161

### Post-0.2.0 (0.3.x)

- No P2 items are scoped to this crate. The `torsh-interop` evaluation (once torsh 0.2.0 ships on crates.io) and the candle implement-vs-remove decision are tracked in the root TODO.md; if either lands, revisit the umbrella crate's feature forwarders here.

---

## Completed Features

### Auto Classes

#### AutoModel

**Automatic model class selection**

- [x] **Model Types**
  - AutoModel - Base model
  - AutoModelForCausalLM - Causal LM (GPT-2, LLaMA)
  - AutoModelForMaskedLM - Masked LM (BERT, RoBERTa)
  - AutoModelForSequenceClassification - Classification
  - AutoModelForTokenClassification - NER, POS tagging
  - AutoModelForQuestionAnswering - Extractive QA
  - AutoModelForSeq2SeqLM - Translation, summarization

**Example** (verified against `src/automodel.rs` / `src/automodel_tasks.rs`):
```rust
use trustformers::AutoModel;

// Load model automatically based on config
let model = AutoModel::from_pretrained("bert-base-uncased")?;

// Or a task-specific wrapper (num_labels is a required argument)
use trustformers::AutoModelForSequenceClassification;
let model = AutoModelForSequenceClassification::from_pretrained("bert-base-uncased", 2)?;
```

---

#### AutoTokenizer

**Automatic tokenizer selection**

- [x] **Tokenizer Types**
  - BPE (GPT-2, GPT-J, LLaMA)
  - WordPiece (BERT, DistilBERT)
  - SentencePiece (T5, ALBERT, XLNet)
  - Unigram (mBART, XLM-RoBERTa)

**Example** (the `Tokenizer` trait's `encode` takes only the text — no boolean/`Option` second argument):
```rust
use trustformers::{AutoTokenizer, Tokenizer};

// Load tokenizer automatically
let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;

// Encode text
let encoding = tokenizer.encode("Hello, world!")?;
println!("Token IDs: {:?}", encoding.input_ids);
```

---

#### AutoConfig

**Automatic configuration loading**

- [x] **Features**
  - Load config from Hub
  - Load from local path
  - Auto-detect model type
  - Validation

**Example:**
```rust
use trustformers::AutoConfig;

let config = AutoConfig::from_pretrained("gpt2")?;
println!("Model type: {}", config.model_type());
println!("Hidden size: {}", config.hidden_size());
```

---

### Pipeline Functions

> All pipeline examples below use the real `pipeline(task: &str, model: Option<&str>, options: Option<PipelineOptions>)` factory signature and the `Pipeline` trait's `__call__`/`batch` methods, matching `src/pipeline/mod.rs` and `examples/basic_pipeline.rs`. Earlier revisions of this document showed a simplified/incorrect calling convention (single-argument `pipeline(task)` plus direct `classifier(text)` invocation) that does not compile against the current API; this has been corrected throughout.

#### Text Generation Pipeline

**Generate text from prompts**

- [x] **Features**
  - Greedy decoding
  - Beam search
  - Nucleus sampling (top-p)
  - Top-k sampling
  - Temperature control
  - Async streaming output

**Example:**
```rust
use trustformers::pipeline;

let generator = pipeline("text-generation", Some("gpt2"), None)?;
let result = generator.__call__("Once upon a time".to_string())?;
println!("Generated: {:?}", result);
```

---

#### Text Classification Pipeline

**Classify text into categories**

- [x] **Features**
  - Sentiment analysis
  - Multi-label classification
  - Zero-shot classification
  - Multi-class classification

**Example:**
```rust
use trustformers::pipeline;

// Sentiment analysis
let classifier = pipeline("sentiment-analysis", Some("distilbert-base-uncased-finetuned-sst-2"), None)?;
let result = classifier.__call__("I love Rust!".to_string())?;
println!("Sentiment: {:?}", result);
```

---

#### Question Answering Pipeline

**Extract answers from context**

- [x] **Features**
  - Extractive QA
  - Span prediction
  - Confidence scores

**Example:**
```rust
use trustformers::pipeline;

let qa = pipeline("question-answering", Some("distilbert-base-cased-distilled-squad"), None)?;
let qa_input = "Question: What is Rust? Context: Rust is a systems programming language...".to_string();
let result = qa.__call__(qa_input)?;
println!("{:?}", result);
```

---

#### Token Classification Pipeline

**Classify individual tokens**

- [x] **Use Cases**
  - Named Entity Recognition (NER)
  - Part-of-Speech tagging
  - Chunking

**Example:**
```rust
use trustformers::pipeline;

let ner = pipeline("ner", Some("dslim/bert-base-NER"), None)?;
let result = ner.__call__("My name is Wolfgang and I live in Berlin".to_string())?;
println!("{:?}", result);
```

---

#### Summarization Pipeline

**Generate text summaries**

- [x] **Features**
  - Abstractive summarization
  - Length control
  - Beam search

**Example:**
```rust
use trustformers::pipeline;

let summarizer = pipeline("summarization", Some("facebook/bart-large-cnn"), None)?;
let result = summarizer.__call__("Very long article text...".to_string())?;
println!("{:?}", result);
```

---

#### Translation Pipeline

**Translate between languages**

- [x] **Features**
  - Multi-language support
  - Language pair detection (via `EnhancedTranslationPipeline` / `LanguageDetector`)
  - Beam search

**Example:**
```rust
use trustformers::pipeline;

let translator = pipeline("translation", Some("Helsinki-NLP/opus-mt-en-fr"), None)?;
let result = translator.__call__("Hello, how are you?".to_string())?;
println!("{:?}", result);
```

---

#### ConversationalPipeline

**Multi-turn dialogue management** (feature `async`)

- [x] **Features**
  - Conversation history tracking
  - Context window management
  - Multi-turn state
  - Streaming responses (`src/pipeline/conversational/streaming/`)

---

#### MultiModal Pipeline

**Vision-language tasks**

- [x] **Features**
  - Image captioning
  - Visual question answering
  - Image-text matching

---

#### DocumentUnderstanding Pipeline

**Document analysis**

- [x] **Features**
  - Document layout analysis
  - Information extraction
  - Form understanding

---

### Pipeline Composition

- [x] **ComposedPipeline** - Sequential multi-stage pipeline execution
- [x] **EnsemblePipeline** - Aggregated predictions from multiple models
- [x] **PipelineChain** - Chained pipeline execution via `.add_stage(pipeline)` builder, invoked like any other `Pipeline` (`.__call__(input)`)
- [x] **PipelineComposer** - Dynamic pipeline construction and management
- [x] **Stub cleanup** - `ComposedPipeline`/`EnsemblePipeline`/`PipelineComposer` internals are fully implemented (historically completed 2026-04-24); no stubs remain in this code today

---

### Safety Filtering

- [x] **SafetyFilter** with `ExtendedSafetyConfig` (boxed to prevent stack overflow)
- [x] **EnhancedSafetyFilter** with multi-risk assessment:
  - [x] Toxicity detection
  - [x] Hate speech classification
  - [x] Personal information detection
  - [x] Violence content filtering
  - [x] Adult content filtering
  - [x] Harassment detection
  - [x] Bias assessment

---

### Infrastructure

- [x] **MemoryPool** - Efficient tensor memory management
- [x] **ConfigurationManager** - Centralized configuration handling
- [x] **EnhancedProfiler** - Performance profiling and tracing
- [x] **HubMirror** - Mirror support for model hub access (feature `hub`)
- [x] **ValidationManager** - Input/output validation
- [x] **BenchmarkSuite** - Built-in benchmarking utilities
- [x] **ModelDiagnostics** - Weight-norm/activation/gradient-flow/attention-entropy checks, dead-neuron and weight-collapse detection (`src/diagnostics/`)
- [x] **Evaluation bridge** - BLEU, ROUGE-N/L, token-F1, exact-match, perplexity adapters to the `Metric` trait (`src/evaluation/bridge.rs`)

---

### HuggingFace Hub Integration

#### Model Download

**Download models from Hub** (feature `hub`)

- [x] **Features**
  - Automatic model download
  - Resumable, chunked, parallel downloads
  - Model caching
  - Revision support (via `HubOptions::revision`)
  - Authentication for private models (via `HubOptions::token`)
  - Mirror support via HubMirror

**Example** (verified against `src/hub.rs`; the free function takes an `Option<HubOptions>`, there is no `Hub`/`HubConfig` type):
```rust
use trustformers::hub::{download_model, HubOptions};

// Download model with default options
let model_path = download_model("gpt2", None)?;

// Specific revision + auth token
let options = HubOptions {
    revision: Some("main".to_string()),
    token: Some("hf_...".to_string()),
    ..Default::default()
};
let model_path = download_model("private-org/private-model", Some(options))?;
```

---

#### Model Search

**Search for models on Hub**

- [ ] **Not implemented.** A previous revision of this document listed Hub model search (search by task/language, sort by downloads, filter by library) as complete with a `search_models()` example. No such function exists anywhere in `src/` — this was incorrect and has been corrected. Tracked as a genuine future enhancement below.

---

### Utilities

#### Device Management

**Simplified device selection**

- [x] **Features** (via `pipeline::Device` / `pipeline::PipelineOptions`, not a standalone top-level `Device::auto()`/`Device::cuda()` API)
  - `Device::Cpu` / `Device::Gpu(usize)` variants, set through `PipelineOptions { device: Some(Device::Gpu(0)), .. }`
  - Passed into `pipeline(task, model, Some(options))`

---

#### Caching

**Efficient model caching**

- [x] **Features**
  - `AdvancedLRUCache` (in-pipeline output cache, `pipeline::advanced_caching`)
  - Disk-based Hub cache (`hub::get_cache_dir`, `hub::is_cached`)
  - Cache invalidation / TTL / tag-based eviction
  - Size limits
- [x] Wire cache/versioned_cache.rs into lib.rs (planned 2026-07-05)
  - Goal: same underlying fix as the "wire finetuning/+cache/+loading/" item above — cache/mod.rs already internally wires versioned_cache.rs; the only missing piece for both items is the identical single `pub mod cache;` line in lib.rs. Implemented once as part of that item, not twice.
  - Files: trustformers/src/lib.rs (same edit as the finetuning/cache/loading item).
  - Risk: none — this is a duplicate of the item above, not independent work.

---

## Remaining Work

### Minor Stubs (12 static grep hits, 0 reachable — re-verified 2026-07-01)

A grep for `todo!()`/`unimplemented!()` across `src/` returns exactly 12 hits. All are confirmed benign; none sit on a reachable production code path:

- [x] `pipeline/code_generation.rs` — 1 hit: a string literal emitted as *sample generated Rust code* by the code-generation pipeline's stub-body generator
- [x] `auto/data_collators/mod.rs` — 1 hit: inside a `` ```rust,ignore `` doc example illustrating a user's own `DataCollator` implementation
- [x] `pipeline/conversational/streaming/pipeline.rs` — 10 hits: hidden `# let x = todo!();` setup lines inside `` ```rust,ignore `` doctest examples
- [x] `ComposedPipeline`/`EnsemblePipeline`/`PipelineComposer` internals (implemented 2026-04-24) — no remaining stubs

### Future Enhancements

#### High Priority
- [x] **RAG Pipeline** - TF-IDF and BM25 based retrieval-augmented generation
- [x] **Enhanced Translation Pipeline** - Language detection + batch translation
- [x] Resolve all stubs in pipeline composition code
- [x] Enhanced Hub features (upload, model cards, diagnostics — see new modules)
- [x] Better error messages and diagnostics (ModelDiagnostics, HubError, ModelCardError)
- [x] More pipeline types (audio, vision-only) — `audio_classification`, `image_classification`, `object_detection`, `depth_estimation`, `speech_to_text` (audio), `text_to_speech` (audio), `image_to_text` (vision), `visual_question_answering` (vision) are now implemented and wired into `pipeline/mod.rs`
- [x] Wire 10 orphaned pipeline drafts into pipeline/mod.rs (planned 2026-07-05)
  - Goal: expose 10 never-compiled pipeline modules (~9900 lines total).
  - Design: add 10 `pub mod X;` lines to pipeline/mod.rs (alphabetized). The generic pipeline() string-dispatch factory does NOT need new cases — this is pub-mod-only wiring, by direct analogy with object_detection/depth_estimation.
  - Files: trustformers/src/pipeline/mod.rs only.
  - Tests: cargo build/cargo test --all-features — this is the real test, since these files have never been compiled and may have drifted against sibling types.
  - Risk: explicit escape hatch — if the build surfaces non-trivial API drift in any of the 10 files, fix what's cheap; for anything that would balloon into a real redesign, `git checkout -- <that one file>` to revert just that file's wiring and leave it un-mounted for a follow-up, rather than let this one item consume the whole batch's budget. Report which (if any) files were reverted.
- [ ] **Hub model search** — no `search_models` implementation exists; needs designing and implementing from scratch (see corrected "Model Search" section above)
- [x] Re-enable 2 disabled test modules (planned 2026-07-05)
  - Goal: flip both #[cfg(test_disabled)] blocks back on.
  - Design: Module A (pipeline/conversational/config/utils.rs) — its "removed types" (PersonaConfigBuilder, ConfigurationPresets) already exist again with matching signatures; likely just flip the cfg and delete the stale comment, then fix whatever the compiler actually flags. Module B (auto/feature_extractors/mod.rs) — one-line wrong-import fix: change `use trustformers_core::errors::TrustformersError;` (a struct with no InvalidInput variant) to `use crate::error::TrustformersError;` (the local enum that has one).
  - Files: pipeline/conversational/config/utils.rs, auto/feature_extractors/mod.rs.
  - Tests: cargo test after flipping each cfg — this is the actual verification.
  - Risk: low, but must be confirmed by a real test run, not just re-reading the code.
- [ ] Re-enable `examples/conversational_ai.rs.disabled` once its API dependencies stabilize
- [x] Clean up stale TODO comments in auto/mod.rs (planned 2026-07-05)
  - Goal/Design: delete 3 comments claiming AutoDataCollator and "remaining auto submodules" are future work — both are already implemented, wired, and re-exported today.
  - Files: trustformers/src/auto/mod.rs.
  - Tests: none needed (comment-only).
  - Risk: none.

#### Performance
- [ ] Faster model loading
  - **Update:** `src/loading/parallel_loader.rs` (815 lines) implements a parallel model loader but is **not wired into `lib.rs`** — wiring it in and benchmarking it is the concrete next step, rather than starting from scratch
  - **Refinement still needed:** target metric (e.g., 20% throughput improvement, <100ms load latency for 7B models?)
- [ ] Better caching strategies
  - **Update:** `src/cache/versioned_cache.rs` (838 lines) implements a versioned cache but is **not wired into `lib.rs`**
  - **Refinement still needed:** which caching layer is targeted — weights, KV cache, tokenizer outputs?
- [ ] Reduced memory usage for large models
  - **Refinement needed:** What is the target metric? (e.g., peak RSS reduction %? 70B model fits in 40GB?)

#### Features
- [ ] Fine-tuning: LoRA adapter implementation
  - **Update:** `src/finetuning/lora.rs` (680 lines: `LoraConfig`, `LoraConfigBuilder`, `LoraLinear`, `LoraBias`) is fully implemented but **not wired into `lib.rs`** (no `pub mod finetuning;` in `lib.rs`) — the remaining work is integration + public-API tests, not a from-scratch implementation
  - **Refinement needed:** default adapter ranks and which layers should be adapted by default once wired in
- [ ] Fine-tuning: PEFT/prefix-tuning
  - **Update:** `src/finetuning/adapter.rs` (503 lines: `BottleneckAdapter`, `AdapterConfig`, `AdapterActivation` — Houlsby-style bottleneck adapters) is implemented but likewise not wired in. Prefix-tuning, prompt-tuning, and p-tuning v2 remain fully unimplemented.
  - **Refinement needed:** which PEFT variants beyond LoRA/adapters to prioritize
- [ ] Fine-tuning: training loop helpers
  - **Refinement needed:** Should helpers wrap trustformers-training or be standalone?
- [x] Evaluation metrics integration (BLEU, ROUGE, F1/exact-match, perplexity — `src/evaluation/bridge.rs`)
- [x] Add AutoModelForObjectDetection (planned 2026-07-05)
  - Goal: mirror the existing AutoModelForImageClassification/AutoModelForAudioClassification (Pattern B: plain struct wrapping ObjectDetectionPipeline, from_pretrained/from_local/detect()/accessors — no AutoConfig dispatch, no weight loading).
  - MANDATORY documentation requirement: ObjectDetectionPipeline produces deterministic MOCK detections today (no real detection head exists anywhere in the ecosystem — consistent with the already-shipped ImageClassification/AudioClassification siblings, which are equally mock). The doc comment on the new struct MUST say so explicitly — do not let this read as real inference to a caller.
  - Files: trustformers/src/automodel_tasks.rs only (no crate-root re-export, matching sibling precedent).
  - Tests: mirror automodel_tasks.rs's existing test style for the closest siblings.
  - Risk: the documentation requirement above is the one thing that must not be skipped.
- [x] Add AutoModelForImageSegmentation (planned 2026-07-05)
  - Goal: mirror the existing AutoModelForImageClassification/AutoModelForAudioClassification (Pattern B: plain struct wrapping ImageSegmentationPipeline, from_pretrained/from_local/segment()/accessors — no AutoConfig dispatch, no weight loading).
  - MANDATORY documentation requirement: ImageSegmentationPipeline produces deterministic MOCK segmentations today (no real segmentation head exists anywhere in the ecosystem — consistent with the already-shipped ImageClassification/AudioClassification siblings, which are equally mock). The doc comment on the new struct MUST say so explicitly — do not let this read as real inference to a caller.
  - Files: trustformers/src/automodel_tasks.rs only (no crate-root re-export, matching sibling precedent).
  - Tests: mirror automodel_tasks.rs's existing test style for the closest siblings.
  - Risk: the documentation requirement above is the one thing that must not be skipped.

---

## Known Limitations (Alpha)

- `src/finetuning/`, `src/cache/`, and `src/loading/` contain real, substantial implementations (~2,900 lines) that are not yet part of the compiled crate (see Remaining Work).
- Ten pipeline source files exist as unwired drafts (see "Wire up orphaned pipeline drafts" above).
- Hub model search is unimplemented (previously mis-documented as complete).
- Two test modules are disabled pending a rewrite after internal API changes.
- One example (`conversational_ai.rs.disabled`) is disabled pending the same rework.
- Some pipelines require specific model types.
- Hub download requires the optional `hub` feature plus an internet connection.
- Large models require significant disk space.
- Caching may use substantial disk space.

---

## Development Guidelines

### Code Standards
- **API Design:** Simple, HuggingFace-compatible
- **Documentation:** Comprehensive examples
- **Testing:** Integration tests with actual models (~2,261 tests)
- **Naming:** Follow HuggingFace conventions

### Build & Test Commands

```bash
# Build
cargo build --release

# Run tests
cargo test --all-features

# Run examples
cargo run --example basic_pipeline
cargo run --example advanced_composition
cargo run --example ensemble_models

# Build documentation
cargo doc --open --all-features
```

---

## Examples

### Basic Usage

```rust
use trustformers::{pipeline, AutoModel, AutoTokenizer, Tokenizer};

// Using pipeline (easiest) — pipeline(task, model, options), invoked via __call__
let generator = pipeline("text-generation", Some("gpt2"), None)?;
let result = generator.__call__("Hello, world!".to_string())?;

// Using Auto classes (more control)
let tokenizer = AutoTokenizer::from_pretrained("gpt2")?;
let model = AutoModel::from_pretrained("gpt2")?;
let encoding = tokenizer.encode("Hello, world!")?;
```

### Multi-Task Example

```rust
use trustformers::pipeline;

// Load multiple pipelines
let generator = pipeline("text-generation", Some("gpt2"), None)?;
let classifier = pipeline("sentiment-analysis", Some("distilbert-base-uncased-finetuned-sst-2"), None)?;

// Generate text, then classify the result
let generated = generator.__call__("Once upon a time".to_string())?;
if let trustformers::pipeline::PipelineOutput::Generation(gen) = generated {
    let sentiment = classifier.__call__(gen.generated_text.clone())?;
    println!("Text: {}", gen.generated_text);
    println!("Sentiment: {:?}", sentiment);
}
```

### Custom Model Configuration

```rust
use trustformers::{AutoModel, AutoConfig};

// Load and inspect config
let config = AutoConfig::from_pretrained("gpt2")?;
println!("Hidden size: {}", config.hidden_size());

// Load model from a (possibly modified) config
let model = AutoModel::from_config(&config)?;
```

---

**Last Updated:** 2026-07-06
**Version:** 0.2.0
**Status:** Alpha
**API:** HuggingFace-compatible high-level API
**Hub:** Core Hub utilities unconditional; remote downloads require the optional `hub` feature
