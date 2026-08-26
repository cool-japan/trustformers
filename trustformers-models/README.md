# trustformers-models

Comprehensive transformer model implementations for NLP, vision, speech, and multimodal tasks — plus a large surrounding toolkit (quantization, distillation, NAS, continual/meta learning, serving, dev tools).

**Version:** 0.2.1 (Alpha) | **Date:** 2026-08-24 | **Tests:** 1,681 passed, 0 failed, 28 skipped as of 2026-08-18, not independently re-run this pass — see root `TODO.md` for the current workspace-wide baseline (21,370 passed / 41 skipped / 0 failed, 2026-08-26) | **SLoC:** 188,417 (`tokei`, verified 2026-08-24) | **Public API items:** ~5,165 (not re-counted this pass)

## Current State

This is the **largest model-coverage crate in the TrustformeRS workspace**: 55 architecture-specific Cargo feature flags plus a handful of always-on bonus architectures, all built on `trustformers-core` abstractions. There are **0 genuine stub/placeholder implementations** (verified by source scan — no `todo!()`/`unimplemented!()`/`FIXME` in production code paths) and no file exceeds the workspace's 2,000-line refactor threshold.

The crate is comprehensive and close to uniform in maturity: weight loading from real HuggingFace checkpoints is complete for 51 of 55 architectures (corrected 2026-08-24 from a previously-claimed 53 — `deepseek_v2` and `s4` were double-counted as complete when they weren't, see below); the remaining 4 return a clean, specific error instead of loading real weights, three for a documented architectural reason and one (`deepseek_v2`) as an honesty fix for what used to be fake success (see [Weight Loading](#weight-loading) and [Known Limitations](#known-limitations) below). Default feature is just `bert`, which is fully complete (config, model, all four task heads, weight loading, doctested examples).

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
- **Swin Transformer** (`swin`) and **DeiT** (`deit`) — hierarchical shifted-window attention and data-efficient distillation-token training respectively (`src/swin/`, `src/deit/` — config + model + classification head, distillation token for DeiT). Newly wired into `lib.rs`/`Cargo.toml` this release (previously present in source but unreachable). Random-initialized construction and forward passes work; neither has a HuggingFace checkpoint-loading path yet — see [Weight Loading](#weight-loading)

### Multimodal Models
- **BLIP-2** (Q-Former), **LLaVA** (CLIP ViT + LLM), **DALL-E** (VQ-VAE + autoregressive image tokens), **Flamingo** (Perceiver Resampler + gated cross-attention), **CogVLM** (visual expert + CogVideo variant, always-on), **Llama-3.2** (`llama3_2`, vision-language)

### State-Space, Linear- and Efficient-Attention Models
- **Mamba** / **Mamba-2** (`mamba`, `mamba2`) — selective state-space models, O(N)
- **RWKV** — linear attention, O(N) train / O(1) inference step
- **S4** — HiPPO-initialized structured state space, O(N log N) via FFT. **No checkpoint-loading path as of 2026-08-24** (see [Weight Loading](#weight-loading)): a fake loader that skipped every tensor while returning `Ok(())` was deleted this wave rather than fixed. Random-init/from-scratch use is unaffected.
- **RetNet** — multi-scale retention, O(N) inference
- **Hyena** — implicit long convolutions, O(N log N)
- **FNet** — Fourier-transform token mixing (no learned attention)
- **Linformer** — low-rank projected linear-complexity attention
- **Performer** — FAVOR+ random-feature attention
- **xLSTM** — extended LSTM with matrix memory
- **Recursive Transformers** — hierarchical/recursive processing for long sequences

`mamba`, `mamba2`, `rwkv`, `s4`, and `linformer` each require their matching Cargo feature to compile (fixed this release — previously declared in `Cargo.toml` but not actually gating anything, see [Feature Flags](#feature-flags)); `retnet`, `hyena`, `fnet`, and `performer` ship unconditionally (no Cargo feature required).

### Domain-Specialized Model Families
Built on top of the above as higher-level, prompt/generation-oriented wrappers: **scientific** (`scientific_specialized`), **legal & medical** (`legal_medical_specialized`), **creative writing** (`creative_writing_specialized`), **code** and **math** (`code_specialized`, `math_specialized`, both gated behind `llama`).

## Beyond the Model Zoo: Supporting Toolkit

Roughly half of this crate's ~5,165 public API items are not model architectures at all — they're a substantial always-on toolkit that explains why the API surface is so much larger than "55 models" would suggest:

- **Quantization**: `advanced_quantization` (NF4/FP4, block-wise, outlier handling), `mixed_bit_quantization`
- **Compression & distillation**: `model_compression`, `knowledge_distillation`, `dynamic_pruning`
- **Architecture search & design**: `neural_architecture_search`, `automated_model_design`, `hybrid_architectures`
- **Training paradigms**: `continual_learning`, `curriculum_learning`, `multi_task_learning`, `progressive_training`, `meta_learning`
- **Attention libraries**: `sparse_attention`, `cross_attention`, `ring_attention` (near-infinite context via ring topology), `hierarchical` (hierarchical transformers), plus the shared `moe` (Mixture-of-Experts) infrastructure reused by Mixtral and friends
- **Serving & operations**: `model_serving`, `batch_inference`, `generation_utils`, `error_recovery`, `memory_profiling`, `performance_optimization`, `benchmarking`, `numerical_parity_tests`, `developer_tools`, `model_cards`, `comprehensive_testing`
- **Exploratory / research**: `biologically_inspired` (Hopfield networks, capsule networks, neural Turing machine, dendritic computation, liquid-time-constant networks) and `quantum_classical_hybrids` (quantum attention/CNN/GNN/RNN/embedding/optimizer components) — these compile and are tested, but should be treated as research-grade rather than production-hardened

## Feature Flags

Default feature is `bert`. 55 architecture-specific flags plus `all`, `metal`, `cuda`:

```toml
[dependencies]
trustformers-models = { version = "0.2.2", features = ["bert", "llama", "mistral", "clip"] }
```

`bert`, `roberta`, `distilbert`, `gpt2`, `gpt_neo`, `gpt_j`, `t5`, `albert`, `electra`, `deberta`, `vit`, `swin`, `deit`, `llama`, `llama2`, `llama3`, `codellama`, `deepseek`, `gpt_neox`, `mistral`, `clip`, `gemma`, `qwen`, `phi3`, `gemma2`, `mamba`, `rwkv`, `s4`, `stablelm`, `falcon`, `blip2`, `llava`, `dalle`, `flamingo`, `linformer`, `internlm2`, `falcon2`, `deepseek_v2`, `qwen2_5`, `opt`, `granite`, `aya`, `jamba`, `jamba2`, `sd3`, `llama3_2`, `mistral_v3`, `mixtral`, `phi2`, `mamba2`, `phi4`, `nemotron`, `whisper`, `yi`, `starcoder2`

Plus `metal` / `cuda` (forward to `trustformers-core`'s GPU backends) and `all` (enables every one of the 55 architecture flags above, including `swin`/`deit`/`llama3_2`/`mistral_v3` — the `llama3_2`/`mistral_v3` gap is fixed this release; `all` still intentionally excludes only `cuda`/`metal`).

**Feature-gating note (verified against `src/lib.rs`)**: every flag in the list above is a real, working `#[cfg(feature = "...")]` gate on both its `pub mod` declaration and its `pub use` re-export. `mamba`, `rwkv`, `s4`, `stablelm`, `falcon`, and `linformer` are fixed this release — previously declared as Cargo features but with no matching `#[cfg(feature = ...)]` guard, so they compiled unconditionally regardless of which features were enabled.

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

**Status** (updated 2026-08-24, correcting the 2026-08-18 figure below): complete for 51 of the 55 feature-gated architectures, including `swin` and `deit` (each has a real `load_pretrained_report` binding a `Checkpoint` via architecture-specific tensor-naming logic, with dedicated tests) and `llama3`/`mistral_v3`/`phi2`/`yi`/`starcoder2` (all previously listed here as unimplemented; all now call the same real `Checkpoint`-based loading as every other architecture). Four verified exceptions remain:

- `llama3_2` (Llama-3.2 vision): Mllama tile/aspect-ratio embeddings and gated cross-attention aren't modelled, so a checkpoint can't be bound faithfully. Documented architectural gap, returns `Err(...)`.
- `deepseek` (DeepSeek-V1's fused MLA projections and `kv_a_layernorm` aren't modelled). Documented architectural gap, returns `Err(...)`.
- **`deepseek_v2`** (corrected 2026-08-24 — previously, wrongly, described here as "unaffected" by the v1 gap): its loader used to report fake success on any non-empty input buffer without binding real weights; this wave replaced that with a documented `Err(...)` instead. Still no real MLA+MoE checkpoint binder — an honesty fix, not a completion.
- **`s4`** (corrected 2026-08-24, newly added to this list): its ~290-line loader — which validated and skipped every tensor while still returning `Ok(())` — was deleted rather than fixed this wave. There is currently no loading path for this architecture at all, so it doesn't even reach the "returns a descriptive `Err(...)`" bar the other three meet.

The three `Err(...)`-returning architectures give handled errors, not panics or `todo!()`/`unimplemented!()` — hence they don't count against the "0 stubs" figure — but functionally, pretrained-checkpoint loading isn't yet available for any of these four. Random-initialization / from-scratch construction and forward passes work normally for all four.

GPT-2's contrastive-search generation is implemented as of this update (greedy, sampling, top-k/top-p, beam search, and contrastive search all work).

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
│   ├── vit/, clip/, swin/, deit/                                   # Vision (swin/deit each require their own Cargo feature)
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

- **Weight-loading gaps** (corrected 2026-08-24, was wrongly "2 of 55" as of 2026-08-18): 4 of 55 feature-gated architectures have no working checkpoint loader today — `llama3_2` and `deepseek` (v1) each for a documented architectural reason; `deepseek_v2` because its loader's fake-success bug was fixed into an honest error rather than a real binder; `s4` because its fake-`Ok(())` loader was deleted rather than fixed, leaving no loading path at all (see [Weight Loading](#weight-loading)). `swin`/`deit` do have a full loading path, same as every other architecture.
- **GPU coverage**: `cuda`/`metal` features exist and forward to `trustformers-core`, but within this crate real GPU-resident `#[cfg(feature = "cuda"/"metal")]` code paths currently exist only for `gpt2` and `gpt_neox`; the rest run on CPU (`f32`) regardless of GPU features being enabled, consistent with the workspace-wide GPU maturity notes in the top-level README.
- **No `AutoModel`/`from_pretrained` dispatcher** in this crate (see [Quick Start](#quick-start)).

Resolved this release (previously listed here): `swin`/`deit` are now wired into `lib.rs`/`Cargo.toml` behind their own Cargo features (previously orphaned dead code); the legacy, unreferenced `src/qwen2/` has been deleted (`qwen2_5` is the supported successor); `mamba`, `rwkv`, `s4`, `stablelm`, `falcon`, and `linformer` now properly gate their module's compilation (previously declared in `Cargo.toml` but vestigial); the `all` meta-feature now includes `llama3_2` and `mistral_v3`.

## License

Apache-2.0
