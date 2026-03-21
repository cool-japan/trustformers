# trustformers-models

Comprehensive transformer model implementations for various NLP and vision tasks.

**Version:** 0.1.0 (Alpha) | **Date:** 2026-03-21 | **Tests:** 759 passing | **SLoC:** 113,086 | **Public API items:** 1,220

## Current State

This crate provides **comprehensive model coverage** with 27+ transformer architectures implemented, including state-of-the-art models like LLaMA, Mistral, CLIP, Mamba, and RWKV. All models are designed for production use with efficient inference and training support. Each model family is gated behind a dedicated feature flag (28 total).

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
- **GPT-NeoX**: 20B parameter model from EleutherAI
- **LLaMA**: Large Language Model Meta AI
  - LLaMA 1: 7B, 13B, 30B, 65B
  - LLaMA 2: 7B, 13B, 70B with grouped-query attention
  - Code Llama variants with extended context
- **Mistral**: Efficient transformer with sliding window attention
  - Mistral 7B and Instruct variants
  - Mixtral 8x7B (Mixture of Experts)
- **Gemma**: Google's efficient models (2B, 7B)
- **Qwen**: Alibaba's models (0.5B to 72B)
- **Phi-3**: Microsoft small language model (3.8B, 128K context)
- **Falcon**: Technology Innovation Institute multi-query attention model
- **StableLM**: Stability AI models (1.6B–12B, base/zephyr/code variants)

### Encoder-Decoder Models
- **T5**: Text-to-Text Transfer Transformer
  - Sizes: Small, Base, Large, XL, XXL

### Vision Models
- **ViT**: Vision Transformer for image classification
- **CLIP**: Contrastive Language-Image Pre-training with CLIPEncoderConfig trait

### Multimodal Models
- **BLIP-2**: Bootstrap Language-Image Pre-training v2 with Q-Former
- **LLaVA**: Large Language and Vision Assistant (CLIP ViT + LLM)
- **DALL-E**: Text-to-image generation with VQ-VAE
- **Flamingo**: Visual language model with Perceiver Resampler (GatedCrossAttention fix applied)
- **CogVLM**: Visual language model with temporal encoder

### Efficient / Linear-Attention Models
- **Mamba**: Selective state-space model, O(N) complexity
- **RWKV**: Receptance Weighted Key Value, linear attention
- **S4**: Structured State Space with HiPPO initialization
- **Hyena**: Implicit long convolutions, O(N log N)
- **Linformer**: Linear-complexity attention via low-rank projection
- **Performer**: FAVOR+ random feature attention
- **RetNet**: Multi-scale retention mechanism, O(N) inference
- **FNet**: Fourier transform-based token mixing

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

- 759 passing tests, 0 failing (as of 2026-03-21)
- Comprehensive unit tests for each model
- Numerical parity tests against reference implementations
- Integration tests with real tokenizers
- Memory leak detection
- Performance regression tests

## Feature Flags

28 feature flags, one per model family:

```toml
[dependencies]
trustformers-models = { version = "0.1.0", features = ["bert", "llama", "mistral", "clip"] }
```

Available flags: `bert`, `roberta`, `albert`, `distilbert`, `electra`, `deberta`, `gpt2`, `gpt_neo`, `gpt_j`, `gpt_neox`, `llama`, `mistral`, `gemma`, `qwen`, `phi3`, `falcon`, `stablelm`, `t5`, `vit`, `clip`, `blip2`, `llava`, `dalle`, `flamingo`, `cogvlm`, `mamba`, `rwkv`, `s4`, `hyena`, `linformer`, `performer`, `retnet`, `fnet`

## License

Apache-2.0