# trustformers

**Version:** 0.1.0 | **Status:** Alpha | **Updated:** 2026-03-21

Main integration crate providing high-level APIs, pipelines, and Hugging Face Hub integration for the TrustformeRS ecosystem.

## Current State

This crate serves as the **primary entry point** for users, offering HuggingFace-compatible APIs for common NLP tasks. It includes comprehensive pipeline implementations, auto model classes, and seamless integration with the Hugging Face Model Hub.

- **SLoC:** ~59,862
- **Tests:** ~1,740
- **Public API exports (prelude):** 50+
- **Pipeline types:** 23+
- **Stubs remaining:** 11 (minor, in complex pipeline composition code)

## Features

### Pipeline API

Complete implementations of 23+ NLP pipeline types:

- **Text Generation**: Language modeling, text completion, causal LM
- **Text Classification**: Sentiment analysis, topic classification
- **Token Classification**: Named Entity Recognition (NER), POS tagging
- **Question Answering**: Extractive QA from context
- **Fill-Mask**: Masked language modeling
- **Summarization**: Abstractive text summarization
- **Translation**: Language translation (seq2seq)
- **Zero-Shot Classification**: Classification without training examples
- **ConversationalPipeline**: Multi-turn dialogue
- **MultiModal**: Vision-language pipelines
- **DocumentUnderstanding**: Document analysis and extraction

### Pipeline Composition

Advanced pipeline orchestration:

- **ComposedPipeline**: Sequential multi-stage pipelines
- **EnsemblePipeline**: Aggregated predictions from multiple models
- **PipelineChain**: Chained pipeline execution
- **PipelineComposer**: Dynamic pipeline construction

All pipelines support:
- Batched inference for efficiency
- Async execution for concurrent requests
- Async streaming for real-time applications
- Device placement (CPU/GPU)

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

Automatic model selection based on task:

- **AutoModel**: Base model auto-selection
- **AutoModelForSequenceClassification**: Text classification models
- **AutoModelForTokenClassification**: Token-level classification
- **AutoModelForQuestionAnswering**: QA models
- **AutoModelForCausalLM**: Text generation models
- **AutoModelForMaskedLM**: Masked language models
- **AutoModelForSeq2SeqLM**: Translation and summarization models
- **AutoTokenizer**: Automatic tokenizer selection
- **AutoConfig**: Configuration auto-detection

### Infrastructure

- **MemoryPool**: Efficient tensor memory management
- **ConfigurationManager**: Centralized configuration handling
- **EnhancedProfiler**: Performance profiling and tracing
- **HubMirror**: Mirror support for model hub access
- **ValidationManager**: Input/output validation
- **BenchmarkSuite**: Built-in benchmarking utilities

### Hugging Face Hub Integration
- **Model downloading** with progress tracking
- **Caching system** for offline use
- **Authentication** for private models
- **Revision/branch** selection
- **Model card** parsing
- **SafeTensors** format support

## Usage Examples

### Pipeline Usage
```rust
use trustformers::pipeline;

// Text classification
let classifier = pipeline("sentiment-analysis")?;
let results = classifier("I love using Rust for ML!")?;
println!("Label: {}, Score: {}", results[0].label, results[0].score);

// Text generation
let generator = pipeline("text-generation")?;
let output = generator("Once upon a time")?;
println!("Generated: {}", output[0].generated_text);

// Question answering
let qa = pipeline("question-answering")?;
let answer = qa(
    "What is Rust?",
    "Rust is a systems programming language focused on safety."
)?;
println!("Answer: {}", answer.answer);
```

### Auto Classes Usage
```rust
use trustformers::{
    AutoModel, AutoTokenizer,
    AutoModelForSequenceClassification,
};

// Load model and tokenizer automatically
let model_name = "bert-base-uncased";
let tokenizer = AutoTokenizer::from_pretrained(model_name)?;
let model = AutoModelForSequenceClassification::from_pretrained(model_name)?;

// Use for inference
let inputs = tokenizer.encode("Hello, world!", None)?;
let outputs = model.forward(&inputs)?;
```

### Pipeline Composition
```rust
use trustformers::pipelines::{PipelineChain, EnsemblePipeline};

// Chain pipelines sequentially
let chain = PipelineChain::new()
    .add(summarization_pipeline)
    .add(classification_pipeline)
    .build()?;

let result = chain.run("Very long document text...")?;
```

### Hub Integration
```rust
use trustformers::hub::{Hub, HubConfig};

// Configure hub access
let config = HubConfig {
    token: Some("your_token".to_string()),
    cache_dir: Some("/path/to/cache".to_string()),
    ..Default::default()
};

let hub = Hub::new(config)?;

// Download model with progress
let model_path = hub.download_model(
    "meta-llama/Llama-2-7b-hf",
    Some("main"), // revision
)?;
```

## Architecture

```
trustformers/
├── src/
│   ├── pipelines/          # 23+ pipeline implementations
│   │   ├── text_classification.rs
│   │   ├── text_generation.rs
│   │   ├── token_classification.rs
│   │   ├── conversational.rs
│   │   ├── multimodal.rs
│   │   ├── document_understanding.rs
│   │   ├── composed.rs
│   │   ├── ensemble.rs
│   │   └── ...
│   ├── auto/              # Auto classes
│   │   ├── model.rs
│   │   ├── tokenizer.rs
│   │   └── config.rs
│   ├── hub/               # Hub integration
│   │   ├── download.rs
│   │   ├── cache.rs
│   │   ├── mirror.rs
│   │   └── auth.rs
│   ├── safety/            # Safety filtering
│   │   ├── filter.rs
│   │   └── enhanced.rs
│   ├── generation/        # Generation strategies
│   │   ├── sampling.rs
│   │   ├── beam_search.rs
│   │   └── streaming.rs
│   └── utils/            # Infrastructure utilities
│       ├── memory_pool.rs
│       ├── profiler.rs
│       ├── benchmark.rs
│       └── validation.rs
```

## Pipeline Features

### Advanced Generation
- **Sampling strategies**: Top-k, top-p, temperature
- **Beam search**: With length penalty and early stopping
- **Streaming generation**: Token-by-token async output
- **Constrained generation**: With logit processors
- **Batch generation**: Efficient multi-prompt processing

### Pipeline Options
```rust
use trustformers::{pipeline, PipelineConfig};

let config = PipelineConfig {
    device: "cuda:0".to_string(),
    batch_size: 32,
    max_length: 512,
    num_threads: 4,
    ..Default::default()
};

let pipeline = pipeline_with_config("text-generation", config)?;
```

## Performance

### Benchmarks
| Pipeline | Model | Batch Size | Throughput |
|----------|-------|------------|------------|
| Text Classification | BERT-base | 32 | 850 samples/s |
| Text Generation | GPT-2 | 1 | 45 tokens/s |
| Question Answering | BERT-base | 16 | 320 QA pairs/s |
| Token Classification | BERT-base | 32 | 750 samples/s |

*Benchmarks on NVIDIA RTX 4090*

### Optimization Features
- **Dynamic batching**: Automatic batch optimization
- **MemoryPool**: Efficient tensor allocation and reuse
- **Lazy loading**: On-demand weight loading
- **Memory mapping**: Efficient large model loading

## Supported Models

The library supports all models implemented in `trustformers-models`:
- BERT, RoBERTa, ALBERT, DistilBERT
- GPT-2, GPT-Neo, GPT-J
- T5 (encoder-decoder)
- LLaMA, Mistral, Gemma, Qwen
- CLIP (multimodal)
- And more...

## Testing

- ~1,740 tests covering pipeline correctness and edge cases
- Auto class functionality tests
- Hub integration tests
- Generation strategy tests
- Safety filter tests
- Performance benchmarks via BenchmarkSuite

## Known Limitations (Alpha)

- 11 stub implementations remain in complex pipeline composition code
- Some pipelines require specific model types
- Hub download requires internet connection
- Large models require significant disk space

## License

Apache-2.0
