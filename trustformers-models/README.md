# trustformers-models

Comprehensive transformer model implementations for NLP, vision, speech, and multimodal tasks — plus a large surrounding toolkit (quantization, distillation, NAS, continual/meta learning, serving, dev tools).

**Version:** 0.2.0 (Alpha) | **Date:** 2026-07-02 | **Tests:** ~4,479 passing | **SLoC:** 151,766 | **Public API items:** ~5,165

## Current State

This is the **largest model-coverage crate in the TrustformeRS workspace**: 53 architecture-specific Cargo feature flags plus a handful of always-on bonus architectures, all built on `trustformers-core` abstractions. There are **0 genuine stub/placeholder implementations** (verified by source scan — no `todo!()`/`unimplemented!()`/`FIXME` in production code paths) and no file exceeds the workspace's 2,000-line refactor threshold.

The crate is comprehensive but not yet 100% uniform in maturity: weight loading from real HuggingFace checkpoints is complete for the large majority of architectures, but 7 of the newer ones currently return a clean "not yet implemented" error instead of loading real weights (see [Weight Loading](#weight-loading) and [Known Limitations](#known-limitations) below). Default feature is just `bert`, which is fully complete (config, model, all four task heads, weight loading, doctested examples).

## Implemented Architectures

### Encoder Models
- **BERT** — bidirectional self-attention, absolute position embeddings, default feature
- **RoBERTa** — BERT training recipe without NSP, dynamic masking
- **ALBERT** — factorized embeddings + cross-layer parameter sharing
- **DistilBERT** — 6-layer distilled BERT
- **ELECTRA** — replaced-token-detection pretraining (generator/discriminator)
- **DeBERTa** — disentangled content/position attention

### Decoder Models / Modern LLMs
- **GPT-2** (`gpt2`), **GPT-Neo** (`gpt_neo`), **GPT-J** (`gpt_j`), **GPT-NeoX** (`gpt_neox`, reuses LLaMA's `RotaryEmbedding`)
- **LLaMA family**: `llama` (1), `llama2` (GQA, RLHF chat variants), `llama3` (Tiktoken vocab 128,256, RoPE θ=500,000), `codellama` (FIM + RoPE scaling)
- **Mistral family**: `mistral` (sliding-window attention + GQA), `mistral_v3` (function calling), `mixtral` (Sparse MoE, depends on `llama`)
- **Gemma family**: `gemma` (multi-query attention, GeGLU), `gemma2` (local/global attention alternation, logit soft-capping)
- **Qwen family**: `qwen`, `qwen2_5`
- **Phi family**: `phi3` (sliding-window + LongRoPE), `phi2` (parallel transformer), `phi4` (14B, GQA, RoPE θ=250,000)
- **Falcon family**: `falcon`, `falcon2`
- **StableLM**: base/zephyr/code variants, partial rotary factor
- **DeepSeek family**: `deepseek`, `deepseek_v2` (Multi-Head Latent Attention + DeepSeekMoE)
- **InternLM2**, **OPT** (Meta), **Granite** (IBM), **Aya** (Cohere, multilingual)
- **Jamba family**: `jamba`, `jamba2` — hybrid Mamba/Transformer
- **Nemotron** (NVIDIA, squared-ReLU + partial rotary + GQA), **Yi** (01.AI, bilingual GQA), **StarCoder2** (BigCode, FIM + near-MQA)
- **Command-R** (Cohere) and **Claude** — both always-on, no feature flag needed. The `claude` module is a best-effort, constitutional-AI-*inspired* implementation; Anthropic has not published Claude's actual model architecture, so this is not a verified reproduction of the real model.

### Encoder-Decoder / Speech / Diffusion-Adjacent
- **T5** — relative position bias, shared enc/dec embeddings, text-to-text framing (`t5`)
- **Whisper** — encoder-decoder speech recognition (`whisper`)
- **SD3** — Stable Diffusion 3 **text-encoder** pipeline only (conditioning encoders, not an image-generation pipeline) (`sd3`)

### Vision Models
- **ViT** (`vit`), **CLIP** (`clip`, with a `CLIPEncoderConfig` trait)
- **Swin Transformer** and **DeiT** are fully implemented in source (`src/swin/`, `src/deit/` — config + model + classification head, distillation token for DeiT) but are **not yet wired into `lib.rs`/`Cargo.toml`** — see [Known Limitations](#known-limitations)

### Multimodal Models
- **BLIP-2** (Q-Former), **LLaVA** (CLIP ViT + LLM), **DALL-E** (VQ-VAE + autoregressive image tokens), **Flamingo** (Perceiver Resampler + gated cross-attention), **CogVLM** (visual expert + CogVideo variant, always-on), **Llama-3.2** (`llama3_2`, vision-language)

### State-Space, Linear- and Efficient-Attention Models
- **Mamba** / **Mamba-2** (`mamba`, `mamba2`) — selective state-space models, O(N)
- **RWKV** — linear attention, O(N) train / O(1) inference step
- **S4** — HiPPO-initialized structured state space, O(N log N) via FFT
- **RetNet** — multi-scale retention, O(N) inference
- **Hyena** — implicit long convolutions, O(N log N)
- **FNet** — Fourier-transform token mixing (no learned attention)
- **Linformer** — low-rank projected linear-complexity attention
- **Performer** — FAVOR+ random-feature attention
- **xLSTM** — extended LSTM with matrix memory
- **Recursive Transformers** — hierarchical/recursive processing for long sequences

All of the above in this section ship unconditionally (no Cargo feature required) except `mamba`, `rwkv`, `s4`, and `linformer`, which have Cargo feature flags declared in `Cargo.toml` but — see the note in [Feature Flags](#feature-flags) — those flags currently don't gate compilation either, so the practical effect is the same: always compiled in. (`mamba2` is the one exception in this section that *is* properly feature-gated — it requires the `mamba2` feature.)

### Domain-Specialized Model Families
Built on top of the above as higher-level, prompt/generation-oriented wrappers: **scientific** (`scientific_specialized`), **legal & medical** (`legal_medical_specialized`), **creative writing** (`creative_writing_specialized`), **code** and **math** (`code_specialized`, `math_specialized`, both gated behind `llama`).

## Beyond the Model Zoo: Supporting Toolkit

Roughly half of this crate's ~5,165 public API items are not model architectures at all — they're a substantial always-on toolkit that explains why the API surface is so much larger than "53 models" would suggest:

- **Quantization**: `advanced_quantization` (NF4/FP4, block-wise, outlier handling), `mixed_bit_quantization`
- **Compression & distillation**: `model_compression`, `knowledge_distillation`, `dynamic_pruning`
- **Architecture search & design**: `neural_architecture_search`, `automated_model_design`, `hybrid_architectures`
- **Training paradigms**: `continual_learning`, `curriculum_learning`, `multi_task_learning`, `progressive_training`, `meta_learning`
- **Attention libraries**: `sparse_attention`, `cross_attention`, `ring_attention` (near-infinite context via ring topology), `hierarchical` (hierarchical transformers), plus the shared `moe` (Mixture-of-Experts) infrastructure reused by Mixtral and friends
- **Serving & operations**: `model_serving`, `batch_inference`, `generation_utils`, `error_recovery`, `memory_profiling`, `performance_optimization`, `benchmarking`, `numerical_parity_tests`, `developer_tools`, `model_cards`, `comprehensive_testing`
- **Exploratory / research**: `biologically_inspired` (Hopfield networks, capsule networks, neural Turing machine, dendritic computation, liquid-time-constant networks) and `quantum_classical_hybrids` (quantum attention/CNN/GNN/RNN/embedding/optimizer components) — these compile and are tested, but should be treated as research-grade rather than production-hardened

## Feature Flags

Default feature is `bert`. 53 architecture-specific flags plus `all`, `metal`, `cuda`:

```toml
[dependencies]
trustformers-models = { version = "0.2.0", features = ["bert", "llama", "mistral", "clip"] }
```

`bert`, `roberta`, `distilbert`, `gpt2`, `gpt_neo`, `gpt_j`, `t5`, `albert`, `electra`, `deberta`, `vit`, `llama`, `llama2`, `llama3`, `codellama`, `deepseek`, `gpt_neox`, `mistral`, `clip`, `gemma`, `qwen`, `phi3`, `gemma2`, `mamba`, `rwkv`, `s4`, `stablelm`, `falcon`, `blip2`, `llava`, `dalle`, `flamingo`, `linformer`, `internlm2`, `falcon2`, `deepseek_v2`, `qwen2_5`, `opt`, `granite`, `aya`, `jamba`, `jamba2`, `sd3`, `llama3_2`, `mistral_v3`, `mixtral`, `phi2`, `mamba2`, `phi4`, `nemotron`, `whisper`, `yi`, `starcoder2`

Plus `metal` / `cuda` (forward to `trustformers-core`'s GPU backends) and `all` (enables every architecture flag **except** `cuda`, `metal`, `llama3_2`, and `mistral_v3` — the last two are a currently-unintentional-looking gap in `Cargo.toml`, not a documented design choice; enable them explicitly if you need them alongside `all`).

**Feature-gating note (verified against `src/lib.rs`)**: `mamba`, `rwkv`, `s4`, `stablelm`, `falcon`, and `linformer` are declared as Cargo features, but their `pub mod` declarations in `lib.rs` have no matching `#[cfg(feature = ...)]` guard — these five architectures compile unconditionally regardless of which features you enable. Toggling those specific flags currently has no effect on the build. Everything else in the flag list above is a real, working `#[cfg(feature = "...")]` gate.

## Quick Start

```rust,no_run
use trustformers_models::bert::{BertModel, BertConfig};
use trustformers_core::traits::{Model, TokenizedInput};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = BertConfig::default();
    let model = BertModel::new(config)?;

    let input_ids: Vec<u32> = vec![101, 2054, 2003, 102];
    let attention_mask: Vec<u8> = vec![1, 1, 1, 1];
    let outputs = model.forward(TokenizedInput::new(input_ids, attention_mask))?;

    let pooled_output = outputs.pooler_output;      // [CLS] token representation
    let sequence_output = outputs.last_hidden_state; // all token representations
    # let _ = (pooled_output, sequence_output);
    Ok(())
}
```

This mirrors the crate's own doctested example (`src/bert/mod.rs`). Task heads follow the same pattern: `BertForSequenceClassification::new(config, num_labels)`, `BertForMaskedLM::new(config)`, `BertForTokenClassification`, `BertForQuestionAnswering`. There is currently no `AutoModel`/`from_pretrained`-style dispatcher in this crate — construct the concrete model type for the architecture you need.

## Weight Loading

HuggingFace-format loading (SafeTensors, PyTorch, JSON configs) plus GGUF, memory-mapped, streaming, and distributed loaders live in `weight_loading/` (8 focused modules, largest 981 lines, well under the 2,000-line policy limit).

**Status**: complete for the large majority of the 53 feature-gated architectures. Verified exceptions that currently return a descriptive `Err(...)` ("weight loading not yet implemented for ...") instead of loading real weights:

- `llama3` (LLaMA-3)
- `llama3_2` (Llama-3.2)
- `mistral_v3` (Mistral v0.3)
- `phi2` (Phi-2)
- `deepseek` (DeepSeek v1 — `deepseek_v2` is unaffected)
- `yi` (Yi)
- `starcoder2` (StarCoder2)

These are handled errors, not panics or `todo!()`/`unimplemented!()` — hence they don't count against the "0 stubs" figure — but functionally, pretrained-checkpoint loading isn't yet available for these seven. Random-initialization / from-scratch construction and forward passes work normally for all of them.

GPT-2 has one narrower, unrelated gap: contrastive-search generation returns "not yet implemented" (greedy, sampling, top-k/top-p, and beam search all work).

## Architecture Highlights

```
trustformers-models/
├── src/
│   ├── bert/, roberta/, albert/, distilbert/, electra/, deberta/   # Encoder models
│   ├── gpt2/, gpt_neo/, gpt_j/, gpt_neox/                          # GPT family
│   ├── llama/, llama2/, llama3/, llama3_2/, codellama/             # LLaMA family
│   ├── mistral/, mistral_v3/, mixtral/                             # Mistral family
│   ├── gemma/, gemma2/, qwen/, qwen2_5/, phi3/, phi2/, phi4/       # Modern LLMs
│   ├── falcon/, falcon2/, stablelm/, deepseek/, deepseek_v2/       # Modern LLMs (cont.)
│   ├── internlm2/, opt/, granite/, aya/, jamba/, jamba2/           # Modern LLMs (cont.)
│   ├── nemotron/, yi/, starcoder2/, command_r/, claude/            # Modern LLMs (cont.)
│   ├── t5/, whisper/, sd3/                                         # Encoder-decoder / speech
│   ├── vit/, clip/, swin/, deit/                                   # Vision (swin/deit orphaned, see Known Limitations)
│   ├── blip2/, llava/, dalle/, flamingo/, cogvlm/                  # Multimodal
│   ├── mamba/, mamba2/, rwkv/, s4/, retnet/, hyena/                # State-space / linear attention
│   ├── fnet/, linformer/, performer/, xlstm/, recursive/           # Efficient attention
│   ├── scientific_specialized.rs, legal_medical_specialized.rs     # Domain-specialized wrappers
│   ├── creative_writing_specialized.rs, code_specialized.rs, math_specialized.rs
│   ├── weight_loading/                                             # HF/GGUF/streaming/distributed loading
│   ├── common/                                                     # Shared ActivationType etc. (always compiled)
│   └── lib.rs                                                      # Module exports
│   └── ... 40+ additional infrastructure modules (quantization, distillation, NAS,
│           continual/meta learning, serving, benchmarking, dev tools — see above)
├── tests/            # Property-based tests (proptest)
├── examples/         # Flash-attention benchmark, GPT-2 generation/Metal examples
└── templates/        # Scaffolding + generator script + tutorial for adding new models
```

## Testing

- ~4,479 tests passing in this crate (0 failing) as part of today's full-workspace run: 18,102 passed / 0 failed / 119 skipped workspace-wide, 0 clippy warnings, 0 rustdoc warnings
- Property-based tests (`proptest`, `tests/models_property_tests.rs`) with recorded regression seeds
- Numerical parity tests (`numerical_parity_tests.rs`) and a dedicated `comprehensive_testing/` validation framework
- **Doctest note**: the 6 core architecture reference doc examples (`bert`, `gpt2`, `llama`, `llava`, `phi3`, `t5`) and the 5 domain-specialized/xLSTM doc examples (`scientific_specialized`, `legal_medical_specialized`, `creative_writing_specialized`, `math_specialized`, `xlstm`) all use `no_run` — they construct real models and are compile-checked but not executed, which is intentional (some construct multi-billion-parameter configs) and keeps `cargo test --doc` fast. This was verified/fixed as of today's release.

## Known Limitations

- **Vestigial feature flags**: `mamba`, `rwkv`, `s4`, `stablelm`, `falcon`, `linformer` are declared in `Cargo.toml` but don't gate their module's compilation (see [Feature Flags](#feature-flags)).
- **Orphaned implementations**: `src/swin/` (2,502 lines) and `src/deit/` (1,692 lines) exist with real, documented APIs but have no `pub mod` declaration anywhere in `lib.rs` — they're unreachable dead code today. Both look ready to wire up behind new feature flags. (A third, legacy `src/qwen2/`, had the same problem — superseded by `qwen2_5` — and has been deleted after confirming it had zero references anywhere in the workspace.)
- **`all` meta-feature gap**: excludes `llama3_2` and `mistral_v3` in addition to the intentional `cuda`/`metal` exclusion.
- **Weight-loading gaps**: 7 of 53 feature-gated architectures (see [Weight Loading](#weight-loading)).
- **GPU coverage**: `cuda`/`metal` features exist and forward to `trustformers-core`, but within this crate real GPU-resident `#[cfg(feature = "cuda"/"metal")]` code paths currently exist only for `gpt2` and `gpt_neox`; the rest run on CPU (`f32`) regardless of GPU features being enabled, consistent with the workspace-wide GPU maturity notes in the top-level README.
- **No `AutoModel`/`from_pretrained` dispatcher** in this crate (see [Quick Start](#quick-start)).

## License

Apache-2.0
