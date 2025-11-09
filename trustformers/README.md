# trustformers

Main integration crate providing high-level APIs, pipelines, and Hugging Face Hub integration for the TrustformeRS ecosystem.

## Current State

This crate serves as the **primary entry point** for users, offering HuggingFace-compatible APIs for common NLP tasks. It includes comprehensive pipeline implementations, auto model classes, and seamless integration with the Hugging Face Model Hub.

## Features

### Pipeline API
Complete implementations of all major NLP pipelines:

- **Text Classification**: Sentiment analysis, topic classification
- **Token Classification**: Named Entity Recognition (NER), POS tagging
- **Text Generation**: Language modeling, text completion
- **Question Answering**: Extractive QA from context
- **Fill-Mask**: Masked language modeling
- **Summarization**: Text summarization
- **Translation**: Language translation
- **Zero-Shot Classification**: Classification without training examples

All pipelines support:
- Batched inference for efficiency
- Async execution for concurrent requests
- Streaming for real-time applications
- Device placement (CPU/GPU)

### Auto Classes
Automatic model selection based on task:

- **AutoModel**: Base model auto-selection
- **AutoModelForSequenceClassification**: Text classification models
- **AutoModelForTokenClassification**: Token-level classification
- **AutoModelForQuestionAnswering**: QA models
- **AutoModelForCausalLM**: Text generation models
- **AutoModelForMaskedLM**: Masked language models
- **AutoTokenizer**: Automatic tokenizer selection
- **AutoConfig**: Configuration auto-detection

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
│   ├── pipelines/          # Pipeline implementations
│   │   ├── text_classification.rs
│   │   ├── text_generation.rs
│   │   ├── token_classification.rs
│   │   └── ...
│   ├── auto/              # Auto classes
│   │   ├── model.rs
│   │   ├── tokenizer.rs
│   │   └── config.rs
│   ├── hub/               # Hub integration
│   │   ├── download.rs
│   │   ├── cache.rs
│   │   └── auth.rs
│   ├── generation/        # Generation strategies
│   │   ├── sampling.rs
│   │   ├── beam_search.rs
│   │   └── streaming.rs
│   └── utils/            # Utilities
```

## Pipeline Features

### Advanced Generation
- **Sampling strategies**: Top-k, top-p, temperature
- **Beam search**: With length penalty and early stopping
- **Streaming generation**: Token-by-token output
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
- **Caching**: Model and tokenizer caching
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

- Comprehensive pipeline tests
- Auto class functionality tests
- Hub integration tests
- Generation strategy tests
- Performance benchmarks

## Future Enhancements

- More pipeline types (image-to-text, speech)
- Enhanced streaming support
- Pipeline composition
- Better error messages
- Performance optimizations

## License

MIT OR Apache-2.0