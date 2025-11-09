# trustformers TODO List

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

### Implementation Status
✅ **PRODUCTION-READY** - Complete high-level API
✅ **HUB INTEGRATED** - Full HuggingFace Hub support
✅ **PIPELINE COMPLETE** - All major pipelines implemented
✅ **AUTO CLASSES** - Auto* classes for model/tokenizer loading
✅ **EXAMPLES READY** - Comprehensive example applications

### Feature Coverage
- **API:** AutoModel, AutoTokenizer, AutoConfig
- **Pipelines:** text-generation, classification, QA, NER, summarization, translation
- **Hub:** Model download, caching, authentication
- **Examples:** 20+ example applications

---

## Completed Features

### Auto Classes

#### AutoModel

**Automatic model class selection**

- ✅ **Model Types**
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

- ✅ **Tokenizer Types**
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

- ✅ **Features**
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

- ✅ **Features**
  - Greedy decoding
  - Beam search
  - Nucleus sampling (top-p)
  - Top-k sampling
  - Temperature control

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

- ✅ **Features**
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

- ✅ **Features**
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

- ✅ **Use Cases**
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

- ✅ **Features**
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

- ✅ **Features**
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

### HuggingFace Hub Integration

#### Model Download

**Download models from Hub**

- ✅ **Features**
  - Automatic model download
  - Resume interrupted downloads
  - Model caching
  - Version/revision support
  - Authentication for private models

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

- ✅ **Features**
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

- ✅ **Features**
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

- ✅ **Features**
  - LRU cache
  - Disk caching
  - Cache invalidation
  - Size limits

---

## Known Limitations

- Some pipelines require specific model types
- Hub download requires internet connection
- Large models require significant disk space
- Caching may use substantial disk space

---

## Future Enhancements

### High Priority
- More pipeline types (audio, vision, multimodal)
- Enhanced Hub features (upload, model cards)
- Better error messages
- Streaming inference for all pipelines

### Performance
- Faster model loading
- Better caching strategies
- Reduced memory usage

### Features
- More auto classes
- Model composition utilities
- Fine-tuning helpers
- Evaluation metrics

---

## Development Guidelines

### Code Standards
- **API Design:** Simple, HuggingFace-compatible
- **Documentation:** Comprehensive examples
- **Testing:** Integration tests with actual models
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

**Last Updated:** Refactored for alpha.1 release
**Status:** Production-ready main integration crate
**API:** HuggingFace-compatible high-level API
**Hub:** Full integration with HuggingFace Hub
