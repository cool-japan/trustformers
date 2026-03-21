# trustformers TODO List

**Version:** 0.1.0 | **Status:** Alpha | **Updated:** 2026-03-21

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

### Implementation Status (as of 2026-03-21)

- [x] **HIGH-LEVEL API** - Complete (AutoModel, AutoTokenizer, AutoConfig, AutoModelFor*)
- [x] **HUB INTEGRATED** - Full HuggingFace Hub support (download, cache, mirror, auth)
- [x] **PIPELINE COMPLETE** - 23+ pipeline types implemented
- [x] **AUTO CLASSES** - Auto* classes for model/tokenizer/config loading
- [x] **PIPELINE COMPOSITION** - ComposedPipeline, EnsemblePipeline, PipelineChain, PipelineComposer
- [x] **SAFETY FILTERING** - SafetyFilter, ExtendedSafetyConfig, EnhancedSafetyFilter (multi-risk)
- [x] **ASYNC STREAMING** - Token-by-token async streaming for generation pipelines
- [x] **INFRASTRUCTURE** - MemoryPool, ConfigurationManager, EnhancedProfiler, HubMirror, ValidationManager, BenchmarkSuite
- [x] **PRELUDE EXPORTS** - 50+ public API exports in prelude
- [ ] **STUB CLEANUP** - 11 stubs remaining in complex pipeline composition code (minor)

### Metrics (2026-03-21)

- **SLoC:** ~59,862
- **Tests:** ~1,740
- **Pipeline types:** 23+
- **Public API exports:** 50+
- **Stubs remaining:** 11 (minor, in complex pipeline code)

### Feature Coverage

- **API:** AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
- **Pipelines:** TextGeneration, TextClassification, QuestionAnswering, TokenClassification, Summarization, Translation, FillMask, ZeroShotClassification, ConversationalPipeline, MultiModal, DocumentUnderstanding, and 12+ more
- **Pipeline Composition:** ComposedPipeline, EnsemblePipeline, PipelineChain, PipelineComposer
- **Safety:** SafetyFilter (ExtendedSafetyConfig), EnhancedSafetyFilter (toxicity, hate speech, personal info, violence, adult content, harassment, bias)
- **Hub:** Model download, caching, authentication, mirror support
- **Infrastructure:** MemoryPool, ConfigurationManager, EnhancedProfiler, HubMirror, ValidationManager, BenchmarkSuite

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

**Example:**
```rust
use trustformers::AutoModel;

// Load model automatically based on config
let model = AutoModel::from_pretrained("bert-base-uncased")?;

// Or specific task
use trustformers::AutoModelForCausalLM;
let model = AutoModelForCausalLM::from_pretrained("gpt2")?;
```

---

#### AutoTokenizer

**Automatic tokenizer selection**

- [x] **Tokenizer Types**
  - BPE (GPT-2, GPT-J, LLaMA)
  - WordPiece (BERT, DistilBERT)
  - SentencePiece (T5, ALBERT, XLNet)
  - Unigram (mBART, XLM-RoBERTa)

**Example:**
```rust
use trustformers::AutoTokenizer;

// Load tokenizer automatically
let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;

// Encode text
let encoding = tokenizer.encode("Hello, world!", true)?;
println!("Token IDs: {:?}", encoding.input_ids);

// Decode
let text = tokenizer.decode(&encoding.input_ids, true)?;
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

let generator = pipeline("text-generation", "gpt2")?;

let result = generator("Once upon a time", &json!({
    "max_length": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "num_return_sequences": 3,
}))?;

for seq in result {
    println!("Generated: {}", seq["generated_text"]);
}
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
let classifier = pipeline("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2")?;
let result = classifier("I love Rust!")?;
println!("Sentiment: {:?}", result);  // [{"label": "POSITIVE", "score": 0.9998}]

// Zero-shot classification
let classifier = pipeline("zero-shot-classification", "facebook/bart-large-mnli")?;
let result = classifier("This is about technology", &json!({
    "candidate_labels": ["technology", "sports", "politics"]
}))?;
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

let qa = pipeline("question-answering", "distilbert-base-cased-distilled-squad")?;

let result = qa(&json!({
    "question": "What is Rust?",
    "context": "Rust is a systems programming language that runs blazingly fast..."
}))?;

println!("Answer: {}", result["answer"]);
println!("Score: {}", result["score"]);
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

let ner = pipeline("ner", "dslim/bert-base-NER")?;
let result = ner("My name is Wolfgang and I live in Berlin")?;

for entity in result {
    println!("{}: {} ({})", entity["word"], entity["entity"], entity["score"]);
}
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

let summarizer = pipeline("summarization", "facebook/bart-large-cnn")?;

let article = "Very long article text...";
let result = summarizer(article, &json!({
    "max_length": 130,
    "min_length": 30,
}))?;

println!("Summary: {}", result[0]["summary_text"]);
```

---

#### Translation Pipeline

**Translate between languages**

- [x] **Features**
  - Multi-language support
  - Language pair detection
  - Beam search

**Example:**
```rust
use trustformers::pipeline;

let translator = pipeline("translation_en_to_fr", "Helsinki-NLP/opus-mt-en-fr")?;
let result = translator("Hello, how are you?")?;

println!("Translation: {}", result[0]["translation_text"]);
```

---

#### ConversationalPipeline

**Multi-turn dialogue management**

- [x] **Features**
  - Conversation history tracking
  - Context window management
  - Multi-turn state

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
- [x] **PipelineChain** - Chained pipeline execution with data flow
- [x] **PipelineComposer** - Dynamic pipeline construction and management
- [ ] **Stub cleanup** - 11 minor stubs remain in composition internals

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
- [x] **HubMirror** - Mirror support for model hub access
- [x] **ValidationManager** - Input/output validation
- [x] **BenchmarkSuite** - Built-in benchmarking utilities

---

### HuggingFace Hub Integration

#### Model Download

**Download models from Hub**

- [x] **Features**
  - Automatic model download
  - Resume interrupted downloads
  - Model caching
  - Version/revision support
  - Authentication for private models
  - Mirror support via HubMirror

**Example:**
```rust
use trustformers::hub::download_model;

// Download model
let model_path = download_model("gpt2")?;

// Specific revision
let model_path = download_model_revision("gpt2", "main")?;

// With authentication
use trustformers::hub::set_token;
set_token("hf_...")?;
let model_path = download_model("private-org/private-model")?;
```

---

#### Model Search

**Search for models on Hub**

- [x] **Features**
  - Search by task
  - Filter by language
  - Sort by downloads/likes
  - Filter by library

**Example:**
```rust
use trustformers::hub::search_models;

let models = search_models(&json!({
    "task": "text-generation",
    "language": "en",
    "sort": "downloads",
    "limit": 10
}))?;

for model in models {
    println!("{}: {}", model.model_id, model.downloads);
}
```

---

### Utilities

#### Device Management

**Simplified device selection**

- [x] **Features**
  - Auto-detect best device
  - Multi-GPU support
  - Device mapping

**Example:**
```rust
use trustformers::Device;

// Auto-detect (CUDA > ROCm > Metal > CPU)
let device = Device::auto()?;

// Explicit device
let device = Device::cuda(0)?;

// Multi-GPU
let model = AutoModel::from_pretrained("llama-2-70b")?
    .device_map("auto")?;
```

---

#### Caching

**Efficient model caching**

- [x] **Features**
  - LRU cache
  - Disk caching
  - Cache invalidation
  - Size limits

---

## Remaining Work

### Minor Stubs (11 total, low priority)

These are located in complex pipeline composition code and do not block core functionality:

- [ ] Stub implementations in `ComposedPipeline` internals
- [ ] Stub implementations in `EnsemblePipeline` aggregation logic
- [ ] Stub implementations in `PipelineComposer` dynamic routing

### Future Enhancements

#### High Priority
- [ ] Resolve all 11 remaining stubs in pipeline composition code
- [ ] More pipeline types (audio, vision-only)
- [ ] Enhanced Hub features (upload, model cards creation)
- [ ] Better error messages and diagnostics

#### Performance
- [ ] Faster model loading
- [ ] Better caching strategies
- [ ] Reduced memory usage for large models

#### Features
- [ ] Fine-tuning helpers
- [ ] Evaluation metrics integration
- [ ] More auto classes (AutoModelForAudioClassification, etc.)

---

## Known Limitations (Alpha)

- 11 stubs remaining in complex pipeline composition code (minor)
- Some pipelines require specific model types
- Hub download requires internet connection
- Large models require significant disk space
- Caching may use substantial disk space

---

## Development Guidelines

### Code Standards
- **API Design:** Simple, HuggingFace-compatible
- **Documentation:** Comprehensive examples
- **Testing:** Integration tests with actual models (~1,740 tests)
- **Naming:** Follow HuggingFace conventions

### Build & Test Commands

```bash
# Build
cargo build --release

# Run tests
cargo test --all-features

# Run examples
cargo run --example text_generation
cargo run --example question_answering
cargo run --example ner

# Build documentation
cargo doc --open --all-features
```

---

## Examples

### Basic Usage

```rust
use trustformers::{pipeline, AutoModel, AutoTokenizer};

// Using pipeline (easiest)
let generator = pipeline("text-generation", "gpt2")?;
let result = generator("Hello, world!")?;

// Using Auto classes (more control)
let tokenizer = AutoTokenizer::from_pretrained("gpt2")?;
let model = AutoModel::from_pretrained("gpt2")?;

let inputs = tokenizer.encode("Hello, world!", true)?;
let outputs = model.forward(inputs.input_ids)?;
```

### Multi-Task Example

```rust
use trustformers::pipeline;

// Load multiple pipelines
let generator = pipeline("text-generation", "gpt2")?;
let classifier = pipeline("sentiment-analysis", "distilbert")?;

// Generate text
let generated = generator("Once upon a time")?;

// Classify generated text
for seq in generated {
    let sentiment = classifier(&seq["generated_text"])?;
    println!("Text: {}", seq["generated_text"]);
    println!("Sentiment: {:?}", sentiment);
}
```

### Custom Model Configuration

```rust
use trustformers::{AutoModel, AutoConfig};

// Load and modify config
let mut config = AutoConfig::from_pretrained("gpt2")?;
config.set_num_hidden_layers(6)?;  // Smaller model

// Load model with custom config
let model = AutoModel::from_config(&config)?;
```

---

**Last Updated:** 2026-03-21
**Version:** 0.1.0
**Status:** Alpha
**API:** HuggingFace-compatible high-level API
**Hub:** Full integration with HuggingFace Hub
