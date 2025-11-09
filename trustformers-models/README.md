# trustformers-models

Comprehensive transformer model implementations for various NLP and vision tasks.

## Current State

This crate provides **extensive model coverage** with 15+ transformer architectures implemented, including state-of-the-art models like LLaMA, Mistral, and CLIP. All models are designed for production use with efficient inference and training support.

## Implemented Models

### Encoder Models
- **BERT**: Bidirectional Encoder Representations from Transformers
  - BertModel, BertForMaskedLM, BertForSequenceClassification, etc.
- **RoBERTa**: Robustly Optimized BERT Pretraining Approach
- **ALBERT**: A Lite BERT with parameter sharing
- **DistilBERT**: Distilled version of BERT (6 layers)
- **ELECTRA**: Efficiently Learning an Encoder that Classifies Token Replacements
- **DeBERTa**: Decoding-enhanced BERT with Disentangled Attention

### Decoder Models
- **GPT-2**: Generative Pre-trained Transformer 2
  - Sizes: Small (124M), Medium (355M), Large (774M), XL (1.5B)
- **GPT-Neo**: Open-source GPT-3 alternative (1.3B, 2.7B)
- **GPT-J**: 6B parameter GPT-3 style model
- **LLaMA**: Large Language Model Meta AI
  - LLaMA 1: 7B, 13B, 30B, 65B
  - LLaMA 2: 7B, 13B, 70B with grouped-query attention
  - Code Llama variants with extended context
- **Mistral**: Efficient transformer with sliding window attention
  - Mistral 7B and Instruct variants
  - Mixtral 8x7B (Mixture of Experts)
- **Gemma**: Google's efficient models (2B, 7B)
- **Qwen**: Alibaba's models (0.5B to 72B)

### Encoder-Decoder Models
- **T5**: Text-to-Text Transfer Transformer
  - Sizes: Small, Base, Large, XL, XXL

### Vision Models
- **ViT**: Vision Transformer for image classification
- **CLIP**: Contrastive Language-Image Pre-training (multimodal)

## Features

### Model Capabilities
- **Pre-trained weight loading** from Hugging Face Hub
- **Task-specific heads** for classification, generation, etc.
- **Generation strategies**: Greedy, sampling, beam search, top-k/top-p
- **Attention optimizations**: FlashAttention support where applicable
- **Quantization support**: Load quantized models for inference

### Architecture Features
- **Modern attention patterns**: Multi-query, grouped-query, sliding window
- **Positional encodings**: Absolute, relative, RoPE, ALiBi
- **Normalization**: LayerNorm, RMSNorm
- **Activation functions**: GELU, SwiGLU, GeGLU, SiLU
- **Parameter sharing**: ALBERT-style factorization

### Performance Optimizations
- **Memory-efficient attention** for long sequences
- **Optimized kernels** for common operations
- **Mixed precision** support (FP16/BF16)
- **Quantization-aware** implementations

## Usage Example

```rust
use trustformers_models::{
    bert::{BertModel, BertConfig},
    gpt2::{GPT2Model, GPT2Config},
    llama::{LlamaModel, LlamaConfig},
    AutoModel,
};

// Load a pre-trained BERT model
let bert = AutoModel::from_pretrained("bert-base-uncased")?;

// Create a GPT-2 model from config
let config = GPT2Config::gpt2_medium();
let gpt2 = GPT2Model::new(&config)?;

// Load LLaMA with custom config
let llama_config = LlamaConfig::llama_7b();
let llama = LlamaModel::new(&llama_config)?;
```

## Model Variants

### BERT Family
- `bert-base-uncased`: 110M parameters
- `bert-large-uncased`: 340M parameters
- `roberta-base`: 125M parameters
- `albert-base-v2`: 11M parameters (shared)
- `distilbert-base-uncased`: 66M parameters

### GPT Family
- `gpt2`: 124M parameters
- `gpt2-medium`: 355M parameters
- `gpt2-large`: 774M parameters
- `gpt2-xl`: 1.5B parameters

### Modern LLMs
- `llama-7b`: 7B parameters
- `llama-13b`: 13B parameters
- `mistral-7b`: 7B parameters
- `gemma-2b`: 2B parameters
- `qwen-0.5b`: 0.5B parameters

## Architecture Highlights

```
trustformers-models/
├── src/
│   ├── bert/            # BERT and variants
│   ├── gpt2/            # GPT-2 family
│   ├── t5/              # T5 models
│   ├── llama/           # LLaMA architectures
│   ├── mistral/         # Mistral models
│   ├── clip/            # Multimodal models
│   ├── auto/            # Auto model classes
│   └── utils/           # Shared utilities
```

## Performance Benchmarks

| Model | Parameters | Inference (ms) | Memory (GB) |
|-------|------------|----------------|-------------|
| BERT-base | 110M | 5.2 | 0.4 |
| GPT-2 | 124M | 8.1 | 0.5 |
| LLaMA-7B | 7B | 42.3 | 13.5 |
| Mistral-7B | 7B | 38.7 | 13.0 |

*Benchmarks on NVIDIA A100, batch size 1, sequence length 512*

## Testing

- Comprehensive unit tests for each model
- Numerical parity tests against reference implementations
- Integration tests with real tokenizers
- Memory leak detection
- Performance regression tests

## Future Models

Planned additions include:
- Mamba (state-space models)
- RWKV (linear attention)
- Phi-3 series
- Falcon models
- More multimodal architectures

## License

MIT OR Apache-2.0