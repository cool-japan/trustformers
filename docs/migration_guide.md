# Migration Guide: From Python to TrustformeRS

This comprehensive guide helps you migrate from PyTorch and Hugging Face Transformers to TrustformeRS, covering API differences, common patterns, and best practices.

## Table of Contents

1. [Overview](#overview)
2. [PyTorch to TrustformeRS](#pytorch-to-trustformers)
3. [Hugging Face to TrustformeRS](#hugging-face-to-trustformers)
4. [Common Patterns](#common-patterns)
5. [Performance Considerations](#performance-considerations)
6. [Migration Checklist](#migration-checklist)

## Overview

TrustformeRS provides a Rust-native implementation of transformer models with significant performance improvements and memory safety guarantees. While the core concepts remain similar, the API design follows Rust idioms and patterns.

### Key Differences

| Aspect | Python/PyTorch | TrustformeRS |
|--------|----------------|--------------|
| Memory Management | Garbage collected | RAII, explicit lifetimes |
| Error Handling | Exceptions | Result<T, E> types |
| Tensor Operations | Dynamic typing | Strong static typing |
| GPU Support | CUDA-centric | Multi-backend (CUDA/Metal/Vulkan) |
| Package Management | pip/conda | Cargo |

## PyTorch to TrustformeRS

### Basic Tensor Operations

#### PyTorch
```python
import torch

# Creating tensors
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.zeros(3, 4)
z = torch.randn(2, 3, 4)

# Operations
result = x + y[0]
matrix_mult = torch.matmul(a, b)
```

#### TrustformeRS
```rust
use trustformers_core::tensor::Tensor;
use scirs::array;

// Creating tensors
let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
let y = Tensor::zeros(&[3, 4])?;
let z = Tensor::randn(&[2, 3, 4])?;

// Operations
let result = x.add(&y.index(&[0])?)?;
let matrix_mult = a.matmul(&b)?;
```

### Model Definition

#### PyTorch
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

#### TrustformeRS
```rust
use trustformers_core::{Layer, Linear, ReLU, Sequential};

pub struct MyModel {
    layers: Sequential,
}

impl MyModel {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Result<Self> {
        let layers = Sequential::new(vec![
            Box::new(Linear::new(input_dim, hidden_dim)?),
            Box::new(ReLU::new()),
            Box::new(Linear::new(hidden_dim, output_dim)?),
        ]);
        
        Ok(Self { layers })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.layers.forward(x)
    }
}
```

### Training Loop

#### PyTorch
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

#### TrustformeRS
```rust
use trustformers_optim::{Adam, Optimizer};
use trustformers_core::loss::CrossEntropyLoss;

let mut optimizer = Adam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
let criterion = CrossEntropyLoss::new();

for epoch in 0..num_epochs {
    for (batch_idx, batch) in dataloader.enumerate() {
        let (data, target) = batch?;
        
        // Forward pass
        let output = model.forward(&data)?;
        let loss = criterion.forward(&output, &target)?;
        
        // Backward pass
        loss.backward()?;
        
        // Update parameters
        for (param, grad) in model.parameters_and_gradients()? {
            optimizer.update(param, grad)?;
        }
        optimizer.step();
        optimizer.zero_grad();
    }
}
```

### Data Loading

#### PyTorch
```python
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

#### TrustformeRS
```rust
use trustformers_data::{Dataset, DataLoader};

pub struct MyDataset {
    data: Vec<Tensor>,
    labels: Vec<Tensor>,
}

impl Dataset for MyDataset {
    type Item = (Tensor, Tensor);
    
    fn len(&self) -> usize {
        self.data.len()
    }
    
    fn get(&self, idx: usize) -> Result<Self::Item> {
        Ok((self.data[idx].clone(), self.labels[idx].clone()))
    }
}

let dataloader = DataLoader::new(dataset)
    .batch_size(32)
    .shuffle(true)
    .build()?;
```

## Hugging Face to TrustformeRS

### Model Loading

#### Hugging Face
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Inference
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
```

#### TrustformeRS
```rust
use trustformers::{AutoModel, AutoTokenizer};

let model = AutoModel::from_pretrained("bert-base-uncased")?;
let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;

// Inference
let inputs = tokenizer.encode("Hello world!", None)?;
let outputs = model.forward(&inputs)?;
```

### Text Classification Pipeline

#### Hugging Face
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this library!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

#### TrustformeRS
```rust
use trustformers::pipeline::{Pipeline, TextClassificationPipeline};

let classifier = TextClassificationPipeline::new("sentiment-analysis")?;
let result = classifier.predict("I love this library!")?;
// ClassificationOutput { label: "POSITIVE", score: 0.9998 }
```

### Text Generation

#### Hugging Face
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")
output = model.generate(
    input_ids,
    max_length=50,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)
```

#### TrustformeRS
```rust
use trustformers_models::gpt2::GPT2LMHeadModel;
use trustformers_tokenizers::GPT2Tokenizer;
use trustformers::generation::{GenerationConfig, generate};

let model = GPT2LMHeadModel::from_pretrained("gpt2")?;
let tokenizer = GPT2Tokenizer::from_pretrained("gpt2")?;

let input_ids = tokenizer.encode("Once upon a time", None)?;
let config = GenerationConfig {
    max_length: 50,
    temperature: 0.7,
    do_sample: true,
    top_p: 0.9,
    ..Default::default()
};

let output = generate(&model, &input_ids, &config)?;
```

### Fine-tuning BERT

#### Hugging Face
```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

#### TrustformeRS
```rust
use trustformers_models::bert::BertForSequenceClassification;
use trustformers_training::{Trainer, TrainingArguments};

let mut model = BertForSequenceClassification::from_pretrained(
    "bert-base-uncased",
    Some(2), // num_labels
)?;

let training_args = TrainingArguments::builder()
    .output_dir("./results")
    .num_train_epochs(3)
    .per_device_train_batch_size(16)
    .warmup_steps(500)
    .weight_decay(0.01)
    .logging_dir("./logs")
    .build()?;

let trainer = Trainer::new(
    model,
    training_args,
    train_dataset,
    Some(eval_dataset),
)?;

trainer.train()?;
```

## Common Patterns

### Error Handling

Python uses exceptions, while Rust uses `Result<T, E>`:

#### Python
```python
try:
    model = load_model("path/to/model")
    output = model(input_data)
except Exception as e:
    print(f"Error: {e}")
```

#### Rust
```rust
match load_model("path/to/model") {
    Ok(model) => {
        match model.forward(&input_data) {
            Ok(output) => process_output(output),
            Err(e) => eprintln!("Forward error: {}", e),
        }
    }
    Err(e) => eprintln!("Loading error: {}", e),
}

// Or using the ? operator
let model = load_model("path/to/model")?;
let output = model.forward(&input_data)?;
```

### Batch Processing

#### Python
```python
# Process in batches
for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]
    outputs = model(batch)
    process_batch(outputs)
```

#### Rust
```rust
// Process in batches
for chunk in data.chunks(batch_size) {
    let batch = Tensor::stack(chunk)?;
    let outputs = model.forward(&batch)?;
    process_batch(&outputs)?;
}
```

### GPU Management

#### Python
```python
# Move to GPU
model = model.cuda()
data = data.to("cuda")

# Multi-GPU
model = nn.DataParallel(model)
```

#### Rust
```rust
// Move to GPU
let model = model.to_device(Device::cuda(0)?)?;
let data = data.to_device(Device::cuda(0)?)?;

// Multi-GPU
let model = DataParallel::new(model, vec![0, 1, 2, 3])?;
```

### Model Serialization

#### Python
```python
# Save
torch.save(model.state_dict(), "model.pth")

# Load
model.load_state_dict(torch.load("model.pth"))
```

#### Rust
```rust
// Save
model.save_state_dict("model.safetensors")?;

// Load
model.load_state_dict("model.safetensors")?;
```

## Performance Considerations

### Memory Efficiency

TrustformeRS provides better memory control:

1. **Explicit Memory Management**: No garbage collector overhead
2. **Zero-Copy Operations**: Many operations avoid unnecessary copies
3. **Efficient Tensor Storage**: Contiguous memory layout by default

### Parallelism

```rust
// Automatic parallelism for operations
let result = tensor1.add(&tensor2)?; // Uses BLAS/parallel ops

// Explicit parallel iteration
use rayon::prelude::*;
let results: Vec<_> = data.par_iter()
    .map(|x| model.forward(x))
    .collect::<Result<Vec<_>>>()?;
```

### Compilation Optimizations

Enable optimizations in `Cargo.toml`:

```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

### Benchmarking

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_forward_pass(c: &mut Criterion) {
    let model = load_model().unwrap();
    let input = create_test_input().unwrap();
    
    c.bench_function("forward_pass", |b| {
        b.iter(|| {
            black_box(model.forward(&input).unwrap())
        });
    });
}
```

## Migration Checklist

### Pre-Migration

- [ ] Inventory your Python dependencies
- [ ] Identify custom layers/operations
- [ ] Document model architectures
- [ ] Export model weights to SafeTensors format
- [ ] Prepare test datasets

### Migration Steps

1. **Set up Rust environment**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   cargo new my_trustformers_project
   cd my_trustformers_project
   ```

2. **Add dependencies**
   ```toml
   [dependencies]
   trustformers = "0.1"
   tokio = { version = "1", features = ["full"] }
   anyhow = "1.0"
   ```

3. **Convert model architecture**
   - Map PyTorch layers to TrustformeRS equivalents
   - Implement custom layers if needed
   - Verify tensor shapes match

4. **Load pretrained weights**
   ```rust
   // Convert PyTorch checkpoint to SafeTensors
   let state_dict = StateDict::from_safetensors("model.safetensors")?;
   model.load_state_dict(&state_dict)?;
   ```

5. **Validate outputs**
   - Compare outputs with Python implementation
   - Check numerical precision
   - Benchmark performance

### Post-Migration

- [ ] Run comprehensive tests
- [ ] Profile performance
- [ ] Document API changes
- [ ] Update deployment pipelines
- [ ] Train team on Rust/TrustformeRS

## Common Issues and Solutions

### Issue: Tensor Shape Mismatches
```rust
// Python: implicit broadcasting
// Rust: explicit broadcasting
let broadcasted = tensor.broadcast_to(&[batch_size, seq_len, hidden_dim])?;
```

### Issue: Dynamic Shapes
```rust
// Use dynamic tensors when shape is unknown at compile time
let dynamic_tensor = Tensor::from_vec_dynamic(data, shape)?;
```

### Issue: Missing Operations
```rust
// Implement custom operations
impl Tensor {
    pub fn custom_op(&self) -> Result<Tensor> {
        // Implementation
    }
}
```

### Issue: Async/Await Patterns
```rust
// TrustformeRS supports async operations
let output = model.forward_async(&input).await?;
```

## Getting Help

- **Documentation**: [docs.rs/trustformers](https://docs.rs/trustformers)
- **Examples**: Check the `examples/` directory
- **Community**: Discord (coming soon)
- **Issues**: [GitHub Issues](https://github.com/trustformers/trustformers/issues)

## Next Steps

After migrating your models:

1. Explore [Performance Tuning Guide](./performance_tuning.md)
2. Learn about [Deployment Options](./deployment.md)
3. Try [Advanced Features](./advanced_features.md)
4. Contribute back to the community!

Happy migrating! 🚀