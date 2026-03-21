# Migration from Hugging Face Transformers to TrustformeRS

This guide helps you migrate from Hugging Face Transformers to TrustformeRS, maintaining compatibility while gaining Rust's performance benefits.

## Overview

TrustformeRS provides a familiar API for Hugging Face users while offering:
- 🚀 2-5x faster inference
- 💾 50% less memory usage
- 🔒 Memory safety guarantees
- 🦀 Native Rust performance

## Quick Start

### Installation

#### Hugging Face (Python)
```bash
pip install transformers torch
```

#### TrustformeRS (Rust)
```toml
[dependencies]
trustformers = "0.1"
tokio = { version = "1", features = ["full"] }
```

### Basic Usage Comparison

#### Hugging Face
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
```

#### TrustformeRS
```rust
use trustformers::{AutoModel, AutoTokenizer};

let model = AutoModel::from_pretrained("bert-base-uncased")?;
let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;

let inputs = tokenizer.encode("Hello world!", None)?;
let outputs = model.forward(&inputs)?;
```

## Model Loading and Configuration

### Auto Classes

#### Hugging Face
```python
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)

# Load configuration
config = AutoConfig.from_pretrained("bert-base-uncased")

# Load different model types
base_model = AutoModel.from_pretrained("bert-base-uncased")
classifier = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2
)
```

#### TrustformeRS
```rust
use trustformers::{
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
};

// Load configuration
let config = AutoConfig::from_pretrained("bert-base-uncased")?;

// Load different model types
let base_model = AutoModel::from_pretrained("bert-base-uncased")?;
let classifier = AutoModelForSequenceClassification::from_pretrained(
    "bert-base-uncased",
    Some(2), // num_labels
)?;
```

### Custom Configuration

#### Hugging Face
```python
from transformers import BertConfig, BertModel

config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)

model = BertModel(config)
```

#### TrustformeRS
```rust
use trustformers_models::bert::{BertConfig, BertModel};

let config = BertConfig {
    vocab_size: 30522,
    hidden_size: 768,
    num_hidden_layers: 12,
    num_attention_heads: 12,
    intermediate_size: 3072,
    hidden_dropout_prob: 0.1,
    attention_probs_dropout_prob: 0.1,
    ..Default::default()
};

let model = BertModel::new(config)?;
```

## Tokenization

### Basic Tokenization

#### Hugging Face
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Single text
tokens = tokenizer("Hello world!")
input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]

# Batch processing
batch = tokenizer(
    ["Hello world!", "How are you?"],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```

#### TrustformeRS
```rust
use trustformers::AutoTokenizer;

let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;

// Single text
let encoding = tokenizer.encode("Hello world!", None)?;
let input_ids = encoding.input_ids();
let attention_mask = encoding.attention_mask();

// Batch processing
let batch = tokenizer.encode_batch(
    &["Hello world!", "How are you?"],
    Some(512), // max_length
    true,      // padding
    true,      // truncation
)?;
```

### Advanced Tokenization

#### Hugging Face
```python
# Special tokens
tokenizer.add_special_tokens({
    'additional_special_tokens': ['[CUSTOM1]', '[CUSTOM2]']
})

# Token type IDs for sentence pairs
tokens = tokenizer(
    "First sentence.",
    "Second sentence.",
    return_tensors="pt"
)

# Decoding
text = tokenizer.decode(input_ids, skip_special_tokens=True)
```

#### TrustformeRS
```rust
// Special tokens
tokenizer.add_special_tokens(&[
    "[CUSTOM1]".to_string(),
    "[CUSTOM2]".to_string(),
])?;

// Token type IDs for sentence pairs
let encoding = tokenizer.encode_pair(
    "First sentence.",
    "Second sentence.",
    None,
)?;

// Decoding
let text = tokenizer.decode(&input_ids, true)?; // skip_special_tokens
```

## Pipelines

### Text Classification

#### Hugging Face
```python
from transformers import pipeline

# Create pipeline
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Single prediction
result = classifier("I love this library!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Batch prediction
results = classifier([
    "I love this library!",
    "I hate bugs.",
])
```

#### TrustformeRS
```rust
use trustformers::pipeline::{Pipeline, TextClassificationPipeline};

// Create pipeline
let classifier = TextClassificationPipeline::from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)?;

// Single prediction
let result = classifier.predict("I love this library!")?;
// ClassificationOutput { label: "POSITIVE", score: 0.9998 }

// Batch prediction
let results = classifier.predict_batch(&[
    "I love this library!",
    "I hate bugs.",
])?;
```

### Text Generation

#### Hugging Face
```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

# Basic generation
result = generator(
    "Once upon a time",
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True,
)

# Advanced generation
result = generator(
    "The future of AI is",
    max_new_tokens=100,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    no_repeat_ngram_size=2,
)
```

#### TrustformeRS
```rust
use trustformers::pipeline::{TextGenerationPipeline, GenerationConfig};

let generator = TextGenerationPipeline::from_pretrained("gpt2")?;

// Basic generation
let config = GenerationConfig {
    max_length: 50,
    num_return_sequences: 1,
    temperature: 0.7,
    do_sample: true,
    ..Default::default()
};

let result = generator.generate("Once upon a time", &config)?;

// Advanced generation
let config = GenerationConfig {
    max_new_tokens: 100,
    top_k: 50,
    top_p: 0.95,
    repetition_penalty: 1.2,
    no_repeat_ngram_size: 2,
    ..Default::default()
};

let result = generator.generate("The future of AI is", &config)?;
```

### Question Answering

#### Hugging Face
```python
qa_pipeline = pipeline("question-answering")

result = qa_pipeline(
    question="What is the capital of France?",
    context="Paris is the capital of France. It is known for the Eiffel Tower."
)
# {'answer': 'Paris', 'score': 0.998, 'start': 0, 'end': 5}
```

#### TrustformeRS
```rust
use trustformers::pipeline::QuestionAnsweringPipeline;

let qa_pipeline = QuestionAnsweringPipeline::from_pretrained(
    "distilbert-base-cased-distilled-squad"
)?;

let result = qa_pipeline.answer(
    "What is the capital of France?",
    "Paris is the capital of France. It is known for the Eiffel Tower.",
)?;
// QAOutput { answer: "Paris", score: 0.998, start: 0, end: 5 }
```

## Model Training

### Fine-tuning with Trainer

#### Hugging Face
```python
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()
```

#### TrustformeRS
```rust
use trustformers_training::{Trainer, TrainingArguments};
use trustformers_data::datasets::load_dataset;

// Load dataset
let dataset = load_dataset("imdb")?;

// Training arguments
let training_args = TrainingArguments::builder()
    .output_dir("./results")
    .evaluation_strategy("epoch")
    .learning_rate(2e-5)
    .per_device_train_batch_size(16)
    .per_device_eval_batch_size(64)
    .num_train_epochs(3)
    .weight_decay(0.01)
    .logging_dir("./logs")
    .logging_steps(10)
    .save_strategy("epoch")
    .load_best_model_at_end(true)
    .metric_for_best_model("eval_loss")
    .greater_is_better(false)
    .build()?;

// Create trainer
let trainer = Trainer::new(
    model,
    training_args,
    train_dataset,
    Some(eval_dataset),
    tokenizer,
    Some(compute_metrics),
)?;

// Train
trainer.train()?;
```

### Custom Training Loop

#### Hugging Face
```python
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Scheduler
num_training_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

#### TrustformeRS
```rust
use trustformers_optim::{AdamW, LinearScheduler};

// Optimizer
let mut optimizer = AdamW::new(5e-5, (0.9, 0.999), 1e-8, 0.01);

// Scheduler
let num_training_steps = train_dataloader.len() * num_epochs;
let mut scheduler = LinearScheduler::new(
    5e-5,
    0, // num_warmup_steps
    num_training_steps,
);

// Training loop
model.train();
for epoch in 0..num_epochs {
    for batch in train_dataloader.iter() {
        let outputs = model.forward(&batch)?;
        let loss = outputs.loss;
        
        loss.backward()?;
        
        for (param, grad) in model.parameters_and_gradients()? {
            optimizer.update(param, grad)?;
        }
        
        optimizer.step();
        let lr = scheduler.get_lr(optimizer.get_step());
        optimizer.set_lr(lr);
        scheduler.step();
        optimizer.zero_grad();
    }
}
```

## Model Hub Integration

### Downloading Models

#### Hugging Face
```python
from huggingface_hub import hf_hub_download

# Download specific file
file_path = hf_hub_download(
    repo_id="bert-base-uncased",
    filename="pytorch_model.bin",
    cache_dir="./models"
)

# Download with authentication
from huggingface_hub import login
login(token="your_token")
model = AutoModel.from_pretrained("private/model")
```

#### TrustformeRS
```rust
use trustformers::hub::{download_model, HubConfig};

// Download specific file
let file_path = download_model(
    "bert-base-uncased",
    Some("safetensors"), // preferred format
    Some("./models"),    // cache_dir
)?;

// Download with authentication
let hub_config = HubConfig::new()
    .with_token("your_token")
    .with_cache_dir("./models");

let model = AutoModel::from_pretrained_with_config(
    "private/model",
    &hub_config,
)?;
```

### Uploading Models

#### Hugging Face
```python
from huggingface_hub import HfApi

api = HfApi()

# Upload model
model.push_to_hub("username/my-model")

# Upload with commit message
api.upload_folder(
    folder_path="./my-model",
    repo_id="username/my-model",
    commit_message="Update model weights",
)
```

#### TrustformeRS
```rust
use trustformers::hub::HubApi;

let api = HubApi::new()?;

// Upload model
model.push_to_hub("username/my-model")?;

// Upload with commit message
api.upload_folder(
    "./my-model",
    "username/my-model",
    Some("Update model weights"),
)?;
```

## Advanced Features

### Quantization

#### Hugging Face
```python
from transformers import BitsAndBytesConfig

# 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    quantization_config=quantization_config,
)
```

#### TrustformeRS
```rust
use trustformers::quantization::{QuantizationConfig, QuantizationType};

// 8-bit quantization
let quantization_config = QuantizationConfig {
    quant_type: QuantizationType::Int8,
    threshold: 6.0,
    ..Default::default()
};

let model = AutoModelForCausalLM::from_pretrained_quantized(
    "facebook/opt-6.7b",
    &quantization_config,
)?;
```

### PEFT (Parameter-Efficient Fine-Tuning)

#### Hugging Face
```python
from peft import LoraConfig, get_peft_model

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

# Apply LoRA
model = get_peft_model(model, peft_config)
```

#### TrustformeRS
```rust
use trustformers::peft::{LoraConfig, apply_lora};

// LoRA configuration
let peft_config = LoraConfig {
    r: 16,
    lora_alpha: 32,
    target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
    lora_dropout: 0.1,
    ..Default::default()
};

// Apply LoRA
let model = apply_lora(model, &peft_config)?;
```

## Performance Optimization

### Batch Processing

```rust
// Efficient batch processing
let batch_size = 32;
let mut results = Vec::with_capacity(texts.len());

for chunk in texts.chunks(batch_size) {
    let batch_encoding = tokenizer.encode_batch(chunk, Some(512), true, true)?;
    let batch_output = model.forward(&batch_encoding)?;
    results.extend(process_outputs(batch_output)?);
}
```

### Caching

```rust
use trustformers::cache::ModelCache;

// Enable KV-cache for generation
let mut cache = ModelCache::new();
let output = model.generate_with_cache(
    &input_ids,
    &generation_config,
    &mut cache,
)?;

// Reuse cache for continued generation
let continued = model.generate_with_cache(
    &new_input,
    &generation_config,
    &mut cache,
)?;
```

### Async Processing

```rust
use tokio;

// Async model inference
#[tokio::main]
async fn main() -> Result<()> {
    let model = AutoModel::from_pretrained("bert-base-uncased").await?;
    
    // Process multiple requests concurrently
    let tasks: Vec<_> = texts.iter()
        .map(|text| {
            let model = model.clone();
            async move {
                let encoding = tokenizer.encode(text, None)?;
                model.forward_async(&encoding).await
            }
        })
        .collect();
    
    let results = futures::future::join_all(tasks).await;
    Ok(())
}
```

## Migration Checklist

- [ ] **Dependencies**: Add TrustformeRS to `Cargo.toml`
- [ ] **Model Weights**: Convert `.bin` files to `.safetensors` format
- [ ] **Tokenizers**: Verify tokenizer compatibility
- [ ] **Configuration**: Map Python configs to Rust structs
- [ ] **Data Pipeline**: Implement data loading in Rust
- [ ] **Training Code**: Convert training loops
- [ ] **Evaluation**: Port metrics computation
- [ ] **Deployment**: Update inference endpoints

## Common Issues

### Issue: Missing Model Architecture
```rust
// Use custom model definition if not in AutoModel registry
use trustformers::models::custom::CustomModel;

let model = CustomModel::from_config(config)?;
model.load_weights("path/to/weights.safetensors")?;
```

### Issue: Tokenizer Compatibility
```rust
// Use specific tokenizer implementation
use trustformers_tokenizers::BertTokenizer;

let tokenizer = BertTokenizer::from_file("tokenizer.json")?;
```

### Issue: GPU Memory
```rust
// Enable gradient checkpointing
model.enable_gradient_checkpointing();

// Use mixed precision
let model = model.half()?; // Convert to FP16
```

## Resources

- [Hugging Face to TrustformeRS Model Mapping](./model_mapping.md)
- [Complete Examples](../examples/huggingface_migration/)
- [Performance Benchmarks](./benchmarks.md#huggingface-comparison)
- [Community Forum](https://github.com/trustformers/trustformers/discussions)

For the latest updates and model support, check our [compatibility matrix](./compatibility.md).