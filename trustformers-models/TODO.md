# trustformers-models TODO List

**Version:** 0.2.1 (Alpha, unreleased) | **Date:** 2026-08-24 | **Tests:** ~4,479 as of 2026-07-09, not independently re-run this pass (root `TODO.md` recorded 1,681/1,681 for this crate on 2026-08-18; see root `TODO.md` for the current workspace-wide baseline) | **SLoC:** 188,417 (`tokei`, verified 2026-08-24 — up from 151,766 on 2026-07-09, largely real checkpoint-binding work landed since, see "Per-Model Weight Loading Status" below) | **Stubs:** 0 as of 2026-07-09, not re-verified | **Public API items:** ~5,165 as of 2026-07-09, not re-verified

## Overview

The `trustformers-models` crate provides implementations of **55 feature-gated transformer architectures**
plus a handful of always-on bonus architectures (CogVLM, Command-R, Claude-inspired, Recursive Transformers,
Hyena, RetNet, FNet, Performer), covering encoder-only, decoder-only, encoder-decoder, vision, speech,
multimodal, and state-space models. It also ships a large surrounding toolkit — quantization, distillation,
model compression, neural architecture search, continual/curriculum/multi-task/meta learning, model serving,
benchmarking, and developer tooling — which is why the public API surface (~5,165 items) is much larger than
the architecture count alone would suggest. All models are built on top of `trustformers-core` abstractions
and follow consistent patterns for configuration, weight loading, and forward passes.

**Key Responsibilities:**
- Model architecture implementations (BERT, GPT-2, LLaMA, Mistral, T5, ViT, CLIP, Mamba, etc.)
- Task-specific heads (CausalLM, SequenceClassification, QuestionAnswering, etc.)
- Weight loading from HuggingFace format (SafeTensors, PyTorch, JSON, GGUF)
- Model configuration management
- Generation methods (greedy, sampling, beam search)
- Integration with trustformers-core layers
- Supporting ML-ops toolkit (quantization, distillation, NAS, serving, benchmarking)

---

## Current Status

### Implementation Status (reality-checked against source 2026-07-09)
- ALPHA RELEASE — all 55 feature-gated architectures implemented, 0 genuine stubs (`todo!()`/`unimplemented!()`/`FIXME` scan returned only false positives in regex-pattern comments)
- COMPREHENSIVE MODEL ZOO — 55 feature-gated architectures + 8 always-on bonus modules (cogvlm, recursive, command_r, claude, hyena, retnet, fnet, performer)
- ZERO COMPILATION ERRORS, 0 clippy warnings, 0 rustdoc warnings — clean across the workspace as of today's full `cargo nextest run --workspace --all-features`
- **1,681 TESTS PASSING** in this crate, 0 failing, 28 skipped (`cargo nextest run -p trustformers-models`, default features, verified 2026-08-18 — supersedes the ~4,479/18,102 figures below, which mixed an old per-crate count with an old workspace-wide count)
- NO FILE EXCEEDS 2,000 LINES — not re-verified crate-wide this pass; see root `TODO.md` for the current cross-workspace list of files that do
- **WEIGHT LOADING, updated 2026-08-18 — this section's "46 of 55" / "2 with no path" figures are both stale**: `swin` and `deit` (`src/{swin,deit}/model.rs`) now have full, real, well-tested weight loading — `load_pretrained_report` calls `Checkpoint::from_reader` + `load_from_checkpoint`, with tensor-naming logic specific to each architecture (DeiT's two prepended special tokens; Swin's per-stage channel-doubling) and 6+ dedicated tests each (`load_pretrained_binds_every_parameter`, `load_pretrained_binds_the_distillation_token`, `load_pretrained_rejects_a_foreign_tensor`, ...). Separately, 5 of the 7 architectures this section used to list as returning a handled "not yet implemented" error now also load real checkpoints — `llama3`, `mistral_v3`, `phi2`, `yi`, and `starcoder2` all call `Checkpoint::from_reader` + `load_checkpoint`. Only `llama3_2` (Mllama tile/aspect-ratio embeddings, gated cross-attention not modelled) and `deepseek` (fused MLA projections + `kv_a_layernorm` not modelled) still honestly return `not_implemented`, each naming the specific architectural reason a checkpoint can't be bound faithfully. Corrected count as of 2026-08-18: 53 of 55 feature-gated architectures have complete weight loading; 2 honestly refuse; 0 have "no path at all". **Further corrected 2026-08-24 — the "53 of 55" figure above is itself now stale, see "Per-Model Weight Loading Status" below for the live figure**: `deepseek_v2` and `s4` were double-counted inside that 53 (they were never actually complete — `deepseek_v2`'s loader silently faked success, and `s4`'s loader silently skipped every tensor while returning `Ok(())`); both were fixed into honest non-complete states this wave (`deepseek_v2` now errors, `s4`'s fake loader was deleted outright). Current: **51 of 55** complete, 3 honestly refuse (`llama3_2`, `deepseek`, `deepseek_v2`), 1 has no loading path at all (`s4`).
- **Correction**: removed a "BART" entry from this file — BART does not exist anywhere in this crate's source, Cargo.toml, or `lib.rs` (no `bart` feature, no `src/bart*`). It was documented here previously but was never actually implemented.
- **Resolved this release** (verified against source 2026-07-09, weight-loading gap since closed 2026-08-18 per the note above): `swin`/`deit` are now wired into `lib.rs`/`Cargo.toml` behind their own Cargo features (previously fully-implemented-but-orphaned, unreachable from the public API); the legacy, zero-referenced `qwen2/` has been deleted; the 6 previously-vestigial Cargo feature flags (`mamba`, `rwkv`, `s4`, `stablelm`, `falcon`, `linformer`) now properly gate compilation; the `all` meta-feature now includes `llama3_2`/`mistral_v3`.

### Model Categories (reality-checked counts)
- **Encoder Models:** 6 (BERT, RoBERTa, ALBERT, DistilBERT, ELECTRA, DeBERTa)
- **Decoder Models / Modern LLMs:** 33 feature flags across GPT-2/GPT-Neo/GPT-J/GPT-NeoX, LLaMA family (llama/llama2/llama3/llama3_2/codellama), Mistral family (mistral/mistral_v3/mixtral), Gemma family (gemma/gemma2), Qwen family (qwen/qwen2_5), Phi family (phi3/phi2/phi4), Falcon family (falcon/falcon2), StableLM, DeepSeek family (deepseek/deepseek_v2), InternLM2, OPT, Granite, Aya, Jamba family (jamba/jamba2), Nemotron, Yi, StarCoder2 — plus always-on Command-R and Claude-inspired
- **Encoder-Decoder / Speech / Diffusion-adjacent:** T5, Whisper, SD3 (text-encoder pipeline only)
- **Vision Models:** 4 shipped (ViT, CLIP, Swin, DeiT — Swin/DeiT newly wired into `lib.rs` this release; neither has a weight-loading path yet)
- **Multimodal Models:** 6 (BLIP-2, LLaVA, DALL-E, Flamingo, CogVLM, Llama-3.2)
- **State-Space / Linear / Efficient-Attention:** 9 (Mamba, Mamba-2, RWKV, S4, RetNet, Hyena, FNet, Linformer, Performer) + xLSTM + Recursive Transformers
- **Domain-Specialized wrappers:** 5 (scientific, legal & medical, creative writing, code, math)

---

## Completed Model Implementations

### Encoder Models (BERT Family)

#### BERT (Bidirectional Encoder Representations from Transformers)
- **Architecture**
  - Bidirectional self-attention
  - Absolute position embeddings (learned, max 512 tokens)
  - Segment embeddings for sentence pairs
  - [CLS] token for classification, [SEP] for sentence separation
- **Variants**
  - BERT-base: 12 layers, 768 hidden, 12 heads (110M params)
  - BERT-large: 24 layers, 1024 hidden, 16 heads (340M params)
- **Weight Loading**
  - Complete weight loading from HuggingFace (SafeTensors, PyTorch, JSON)
  - Automatic model variant detection
- **Task Heads**
  - Sequence classification, token classification (NER/POS), question answering, masked LM

#### RoBERTa (Robustly Optimized BERT)
- Same architecture as BERT, no Next Sentence Prediction loss, dynamic masking, byte-level BPE

#### ALBERT (A Lite BERT)
- Factorized embedding parameterization (V×H → V×E + E×H), cross-layer parameter sharing, sentence-order prediction

#### DeBERTa (Decoding-enhanced BERT with disentangled attention)
- Disentangled content/position attention, enhanced mask decoder, relative position encodings

#### DistilBERT (Distilled BERT)
- 6-layer student network, same hidden size as BERT-base, knowledge distillation from BERT-base teacher

#### ELECTRA (Efficiently Learning an Encoder)
- Replaced-token-detection pretraining, generator/discriminator setup, more compute-efficient than MLM

---

### Decoder Models (GPT Family & Modern LLMs)

#### GPT-2 (Generative Pre-trained Transformer 2)
- **Architecture**: causal self-attention, learned absolute position embeddings (max 1024), pre-LN, byte-level BPE (50,257 vocab)
- **Variants**: Small (124M), Medium (355M), Large (774M), XL (1.5B)
- **Generation**: greedy, top-k, top-p (nucleus), temperature scaling, beam search. **Gap**: contrastive search returns "not yet implemented" (`src/gpt2/generation.rs`)
- **GPU**: only architecture in this crate (besides GPT-NeoX) with real `#[cfg(feature = "cuda"/"metal")]` code paths

#### GPT-Neo (EleutherAI)
- Alternating local (window=256) / global attention layers, optional RoPE. Variants: 125M, 1.3B, 2.7B

#### GPT-J (EleutherAI, 6B)
- RoPE, parallel attention+FFN, dense attention across full sequence

#### GPT-NeoX
- Reuses LLaMA's `RotaryEmbedding` (Cargo feature `gpt_neox = ["llama"]`); has real CUDA/Metal code paths alongside GPT-2

#### LLaMA family
- **LLaMA 1** (`llama`): RoPE, RMSNorm, SwiGLU, pre-normalization. Variants: 7B/13B/30B/65B. Weight loading complete (`src/llama/model.rs`)
- **LLaMA 2** (`llama2`): grouped-query attention, RLHF chat variants
- **LLaMA 3** (`llama3`): extended Tiktoken vocab (128,256), RoPE θ=500,000, GQA on all sizes. **Weight loading NOT yet implemented** — returns a handled error
- **Llama-3.2** (`llama3_2`): multimodal vision-language variant. **Weight loading NOT yet implemented**
- **CodeLlama** (`codellama`): code-specialized LLaMA-2 fine-tune with FIM + RoPE scaling; weight loading complete

#### Mistral family
- **Mistral** (`mistral`): sliding-window attention (4096), GQA, RoPE θ=10000. Weight loading complete (`src/mistral/model.rs`)
- **Mistral v0.3** (`mistral_v3`): function calling. **Weight loading NOT yet implemented**
- **Mixtral** (`mixtral`, depends on `llama`): Sparse Mixture-of-Experts (8x7B), built on the shared `moe` module

#### Gemma family
- **Gemma** (`gemma`): multi-query attention, GeGLU, RMSNorm, RoPE. Weight loading complete (`src/gemma/model.rs`). Variants: 2B, 7B
- **Gemma-2** (`gemma2`): alternating local/global attention, logit soft-capping (`tanh(x/cap)*cap`, cap=50 attn / 30 final), GQA (8 KV heads @ 9B), fixed head_dim=256, vocab 256,000. `tasks.rs` includes `Gemma2ForCausalLM` with soft-capped generation, 18 tests

#### Qwen family
- **Qwen** (`qwen`): multilingual pretraining, multi-format weight loading (`src/qwen/model.rs`)
- **Qwen2.5** (`qwen2_5`): current-generation Qwen (an older, unwired legacy `qwen2` module was removed — see [Future Enhancements](#future-enhancements))

#### Phi family
- **Phi-3** (`phi3`): sliding-window attention (2048), LongRoPE scaling to 128K context, SwiGLU, GQA. `tasks.rs` includes `Phi3Error`, sliding-window masking, chat-prompt formatting, RoPE with LongRoPE scale factors, GQA+SWA+causal attention, greedy generation — 17 tests
- **Phi-2** (`phi2`): parallel transformer. **Weight loading NOT yet implemented**
- **Phi-4** (`phi4`): 14B dense decoder, GQA, RoPE θ=250,000, tied embeddings

#### Falcon family
- **Falcon** (`falcon`): multi-query attention, RoPE, parallel attention+FFN, QKV weight splitting on load (`src/falcon/model.rs`)
- **Falcon2** (`falcon2`)

#### StableLM (Stability AI)
- Base/zephyr/code variants, 1.6B–12B, GQA, partial rotary factor; full weight loading (`src/stablelm/model.rs`)

#### DeepSeek family
- **DeepSeek** (`deepseek`): Multi-Head Latent Attention + DeepSeekMoE. **Weight loading NOT yet implemented**
- **DeepSeek-V2** (`deepseek_v2`): **Corrected 2026-08-24** — the "weight loading complete" claim this line previously carried was wrong: the loader reported fake success on any non-empty input buffer without actually binding weights. As of this wave it returns a documented structured error instead of the fake success (an honesty fix, not a completion) — a real MLA+MoE checkpoint binder for `deepseek_v2` is still open work, same category as the `deepseek` v1 gap above, not "unaffected by" it.

#### InternLM2, OPT, Granite, Aya
- `internlm2`, `opt` (Meta), `granite` (IBM), `aya` (Cohere, multilingual) — each an independent feature-gated decoder

#### Jamba family
- `jamba`, `jamba2` (AI21) — hybrid Mamba/Transformer architectures

#### Nemotron, Yi, StarCoder2
- **Nemotron** (NVIDIA): squared-ReLU, partial rotary embeddings, GQA
- **Yi** (01.AI): bilingual (EN/中文) GQA decoder family. **Weight loading NOT yet implemented**
- **StarCoder2** (BigCode): code generation, FIM, near-MQA. **Weight loading NOT yet implemented**

#### Command-R and Claude (always-on, no feature flag)
- **Command-R** (Cohere-style): `src/command_r/{config,model,tasks}.rs`
- **Claude** (`src/claude/`): a constitutional-AI-*inspired* implementation. Anthropic has not published Claude's real architecture, so treat this as a best-effort community reconstruction, not a verified reproduction

---

### Encoder-Decoder / Speech Models

#### T5 (Text-to-Text Transfer Transformer)
- Encoder-decoder, relative position bias, shared embeddings, SentencePiece. Variants: Small/Base/Large/3B/11B(XXL). Unified text-to-text task framing

#### Whisper (`whisper`)
- Encoder-decoder speech recognition

#### SD3 (`sd3`)
- Stable Diffusion 3 **text-encoder** pipeline (conditioning encoders only — not an image-generation/diffusion pipeline)

---

### Vision & Multimodal Models

#### Vision Transformer (ViT)
- Patch embeddings (16×16/32×32), [CLS] token, standard transformer encoder. Variants: Tiny→Huge (86M–632M params)

#### CLIP (Contrastive Language-Image Pre-training)
- Dual encoder (text + vision), contrastive objective, zero-shot classification. `CLIPEncoderConfig` trait + chunked weight loading (completed 0.1.0 stable)

#### CogVLM (always-on, no feature flag)
- Visual-expert architecture with a CogVideo variant for video understanding

#### BLIP-2
- Querying Transformer (Q-Former) bridging frozen vision/language models

#### LLaVA
- CLIP ViT + LLM (LLaMA/Vicuna-style) visual instruction tuning

#### DALL-E
- VQ-VAE image tokenization (8192-code codebook) + autoregressive transformer decoder

#### Flamingo
- Perceiver Resampler + gated cross-attention for few-shot, interleaved image-text inputs

#### Swin Transformer and DeiT — newly wired this release
- `src/swin/` (2,502 lines, feature `swin`): `SwinConfig` (tiny/small/base/base-384 presets), `SwinModel`, `SwinForImageClassification`
- `src/deit/` (1,692 lines, feature `deit`): `DeiTConfig`, `DeiTModel`, `DeiTForImageClassification` (with distillation token)
- Both now have a `pub mod` declaration and a Cargo feature in `lib.rs`/`Cargo.toml` (previously unreachable dead code despite being fully written). Neither implements `trustformers_core::traits::Model` yet, so there's no `load_pretrained`/checkpoint-loading path — random-initialized construction and forward passes work normally. See [Weight Loading Infrastructure](#weight-loading-infrastructure).

---

### State-Space, Linear- and Efficient-Attention Models

#### S4 (Structured State Space)
- HiPPO initialization (LEGS/LEGT/LAGT/Fourier), O(N log N) via FFT convolution, diagonal-plus-low-rank structure. **Checkpoint loading: none, as of 2026-08-24** — a ~290-line loader that validated and skipped every tensor while still returning `Ok(())` (a fabricated-success bug) was deleted rather than fixed; see "Per-Model Weight Loading Status" below and root `TODO.md`.

#### Mamba / Mamba-2
- Selective scan mechanism (data-dependent parameters), O(N), hardware-aware design. Mamba-2 adds SSD (structured state-space duality)

#### RWKV
- Linear attention, recurrent + parallelizable modes, O(N) train / O(1) inference step, time-mixing + channel-mixing blocks

#### RetNet
- Multi-scale retention (recurrent, parallel, and chunkwise-recurrent modes), O(N) inference, exponential decay + group norm

#### Hyena
- Implicit long convolutions (subquadratic, O(N log N)), data-controlled filters parameterized by a position MLP

#### FNet
- Replaces learned attention with Fourier-transform token mixing

#### Linformer
- Low-rank projection of keys/values for linear-complexity self-attention

#### Performer
- FAVOR+ positive-orthogonal-random-feature attention approximation

#### xLSTM
- Extended LSTM with matrix memory / exponential gating (`src/xlstm/`)

#### Recursive Transformers
- Hierarchical/recursive processing for long sequences, maintaining memory across recursive calls (`src/recursive/`)

---

### Domain-Specialized Model Families
Higher-level wrappers built on the architectures above:
- `scientific_specialized.rs` — scientific/physics-oriented generation (includes an "experimental" content flag)
- `legal_medical_specialized.rs` — legal & medical domain wrappers (includes PII-pattern regexes, e.g. SSN/phone formats)
- `creative_writing_specialized.rs` — creative writing, including an explicit "Experimental" mode
- `code_specialized.rs`, `math_specialized.rs` — both gated behind the `llama` feature

---

## Weight Loading Infrastructure

### HuggingFace Format Support
- **SafeTensors**, **PyTorch** (pickle-based), **JSON** (config.json), **GGUF** (GGML quantized), automatic format detection

### Weight Loading Modules (`src/weight_loading/`, 8 files, 3,932 lines total, largest file 981 lines)
- `config.rs` (198), `utils.rs` (77), `memory_mapped.rs` (138, zero-copy), `streaming.rs` (366, chunk-based), `distributed.rs` (843, multi-node), `huggingface.rs` (948), `gguf.rs` (981), `tests.rs` (309)

### Per-Model Weight Loading Status (updated 2026-08-24; supersedes the 2026-08-18 breakdown, which is stale on 2 architectures — DeepSeek-V2 and S4 — inherited unchanged from the 2026-07-09 list below without being re-checked)
- **Complete (51/55 feature-gated architectures)**: the 2026-08-18 pass's "53/55" figure double-counted `deepseek_v2` and `s4` as complete — both moved out this pass, see below. Otherwise unchanged: the 46 architectures from the superseded 2026-07-09 list (minus `deepseek_v2` and `s4`, now 44) plus `llama3`, `mistral_v3`, `phi2`, `yi`, `starcoder2`, `swin`, `deit` (7) = 51. Plus, separately, the 8 always-on bonus architectures load/construct correctly.
- **NOT yet implemented (3/55)**: `llama3_2` (Mllama tile/aspect-ratio embeddings and gated cross-attention not modelled), `deepseek` (fused MLA projections + `kv_a_layernorm` not modelled), and **`deepseek_v2`, newly moved here 2026-08-24**: its loader previously reported fake success on any non-empty input buffer without binding real weights (a bug, not a deliberate design choice like the other two); this wave replaced that with a documented structured error. All three return a `not_implemented`-class error naming the specific reason.
- **No weight-loading path at all (1/55)**: **`s4`, newly moved here 2026-08-24**: its ~290-line loader — which validated and skipped every tensor while still returning `Ok(())` — is deleted outright rather than fixed, so there is currently no loading path for this architecture at all, honest but a regression in coverage from the (fake) "complete" status it previously carried. See root `TODO.md` for "a real S4 checkpoint loader" as open work.
- <details><summary>Superseded 2026-07-09 breakdown (kept for the "46 already-complete" list only; the 7-item "not yet implemented" and 2-item "no path" lists below are wrong as of 2026-08-18 — see above)</summary>

  - **Complete (46/55 feature-gated architectures)**: BERT, RoBERTa, ALBERT, DeBERTa, DistilBERT, ELECTRA, GPT-2, GPT-Neo, GPT-J, GPT-NeoX, LLaMA, LLaMA-2, CodeLlama, Mistral, Mixtral, Gemma, Gemma-2, Qwen, Qwen2.5, Phi-3, Phi-4, Falcon, Falcon2, StableLM, DeepSeek-V2, InternLM2, T5, Whisper, SD3, ViT, CLIP, BLIP-2, LLaVA, DALL-E, Flamingo, Linformer, Mamba, Mamba-2, RWKV, S4, Opt, Granite, Aya, Jamba, Jamba2, Nemotron — plus, separately, the 8 always-on bonus architectures (CogVLM, Command-R, Claude, Recursive Transformers, Hyena, RetNet, FNet, Performer).
  - ~~NOT yet implemented (7/55): `llama3`, `llama3_2`, `mistral_v3`, `phi2`, `deepseek`, `yi`, `starcoder2`~~
  - ~~No weight-loading path at all (2/55): `swin`, `deit`~~
  </details>

---

## Model Features

### Common Components
- Multi-head attention (MHA, GQA, MQA), RoPE/absolute/relative position embeddings, LayerNorm and RMSNorm, dropout (train/inference modes), FFN variants (SwiGLU, GeGLU), residual connections, causal + padding attention masks

### Task-Specific Heads
- **CausalLM** (GPT-2, LLaMA, etc.), **MaskedLM** (BERT), **SequenceClassification**, **TokenClassification** (NER/POS), **QuestionAnswering** (SQuAD-style), **ImageClassification** (vision models)

### Generation Support
- Greedy decoding, beam search, temperature/top-k/top-p/min-p sampling, constrained/guided (CFG) generation, speculative decoding, streaming token-by-token output
- ~~**Known gap**: contrastive search is not yet implemented for GPT-2~~ — **fixed, verified 2026-08-18**: `src/gpt2/generation.rs` now routes `GenerationMode::ContrastiveSearch` to a real `generate_contrastive_internal`; the doc comment there confirms it "used to be an error arm ... calling it returned 'Contrastive search not yet implemented for GPT-2'".

---

## Code Organization

### Module Structure (abridged — see README.md for the full annotated tree)
```
trustformers-models/src/
├── bert/, roberta/, albert/, distilbert/, electra/, deberta/    # Encoder models
├── gpt2/, gpt_neo/, gpt_j/, gpt_neox/                           # GPT family
├── llama/, llama2/, llama3/, llama3_2/, codellama/              # LLaMA family
├── mistral/, mistral_v3/, mixtral/, gemma/, gemma2/             # Modern LLMs
├── qwen/, qwen2_5/, phi3/, phi2/, phi4/, falcon/, falcon2/      # Modern LLMs (cont.)
├── stablelm/, deepseek/, deepseek_v2/, internlm2/, opt/         # Modern LLMs (cont.)
├── granite/, aya/, jamba/, jamba2/, nemotron/, yi/, starcoder2/ # Modern LLMs (cont.)
├── command_r/, claude/                                          # Always-on decoder models
├── t5/, whisper/, sd3/                                          # Encoder-decoder / speech
├── vit/, clip/, swin/, deit/                                    # Vision (swin/deit orphaned)
├── blip2/, llava/, dalle/, flamingo/, cogvlm/                   # Multimodal
├── mamba/, mamba2/, rwkv/, s4/, retnet/, hyena/                 # State-space / linear attention
├── fnet/, linformer/, performer/, xlstm/, recursive/            # Efficient attention
├── moe.rs                                                       # Shared Mixture-of-Experts infra
├── scientific_specialized.rs, legal_medical_specialized.rs      # Domain-specialized wrappers
├── creative_writing_specialized.rs, code_specialized.rs, math_specialized.rs
├── weight_loading/                                              # HF/GGUF/streaming/distributed loading
├── common/                                                      # Shared ActivationType (always compiled)
├── advanced_quantization.rs, mixed_bit_quantization.rs          # Quantization
├── model_compression.rs, knowledge_distillation.rs, dynamic_pruning.rs
├── neural_architecture_search.rs, automated_model_design.rs, hybrid_architectures.rs
├── continual_learning.rs, curriculum_learning.rs, multi_task_learning.rs
├── progressive_training.rs, meta_learning.rs
├── sparse_attention.rs, cross_attention/, ring_attention.rs, hierarchical/
├── model_serving.rs, batch_inference.rs, generation_utils.rs, error_recovery.rs
├── memory_profiling/, performance_optimization.rs, benchmarking.rs
├── numerical_parity_tests.rs, developer_tools/, model_cards.rs, comprehensive_testing/
├── biologically_inspired/, quantum_classical_hybrids/            # Research-grade / exploratory
└── lib.rs                                                        # Module exports
```

---

## Testing & Validation

- ~4,479 tests passing in this crate, 0 failing (workspace-wide: 18,102 passed / 0 failed / 119 skipped, 0 clippy warnings, 0 rustdoc warnings, as of today's full run)
- Property-based tests (`proptest`) in `tests/models_property_tests.rs`, with a committed `.proptest-regressions` seed file
- Numerical parity tests (`numerical_parity_tests.rs`) and `comprehensive_testing/` validation framework
- Integration examples in `examples/` (flash-attention benchmark, GPT-2 generation, GPT-2 Metal)
- Doctests: 6 core architecture references + 5 domain/xLSTM references use `no_run` (compile-checked, not executed by design) — fixed/verified as of this release; all other doctests execute normally

---

## Known Limitations

- **Weight-loading gaps, corrected 2026-08-24 (was wrongly "only 2 of 55" as of 2026-08-18)**: 4 of 55 feature-gated architectures return a handled error instead of loading real checkpoints — `llama3_2` and `deepseek` each for a specific, documented architectural reason, plus `deepseek_v2` (its fake-success loader was fixed into an honest error this wave); and 1 (`s4`) has no loading path at all (its fake `Ok(())` loader was deleted this wave). See "Per-Model Weight Loading Status" above for the full breakdown. The 5 architectures this line used to also list as gaps (`llama3`, `mistral_v3`, `phi2`, `yi`, `starcoder2`) do load real checkpoints, and `swin`/`deit` do have a full loading path — those five remain correctly excluded from the gap count.
- ~~**GPT-2 generation gap**: contrastive search not yet implemented.~~ Fixed — see "Generation Support" above.
- **GPU coverage within this crate**: real `#[cfg(feature = "cuda"/"metal")]` code paths verified only in `gpt2` and `gpt_neox`; all other architectures run CPU/`f32` regardless of GPU features, matching the workspace-wide GPU maturity notes. (Closing this gap is now tracked under [0.2.0 Release Scope](#020-release-scope-oxicuda-gpu-migration--tch-removal).)
- **No `AutoModel`/`from_pretrained` dispatcher** — callers construct concrete model types directly.
- Some multimodal models (Flamingo, CogVLM) have complex architectures; weight mapping covers all documented components, but coverage of undocumented/edge-case checkpoint layouts is unverified.
- Alpha status: API surface may still evolve before a Stable designation.

Resolved this release (previously listed here, verified against source 2026-07-09): the 6 previously-vestigial Cargo feature flags (`mamba`, `rwkv`, `s4`, `stablelm`, `falcon`, `linformer`) now properly gate their modules; `src/swin/` and `src/deit/` are wired into `lib.rs`/`Cargo.toml` behind their own features (no longer orphaned — though see the weight-loading gap above); the legacy, zero-referenced `src/qwen2/` has been deleted; the `all` meta-feature now includes `llama3_2` and `mistral_v3`.

---

## 0.2.0 Release Scope (OxiCUDA GPU migration & tch removal)

Two workspace-wide tracks land in 0.2.0: **(1) OxiCUDA GPU migration** — GPU execution moves from the `scirs2-core` gpu abstractions to OxiCUDA (~/work/oxicuda, 0.4.x); trustformers-core already integrates oxicuda behind the `cuda`/`metal` features, and this crate's job is routing model forwards through those resident paths. **(2) PyTorch (tch) dependency removal** — the `tch` dependency and the `torch` feature are deleted entirely in 0.2.0 (workspace Cargo.toml:82, trustformers-core torch feature + ~40 lines of cfg arms, and the forwarder features in trustformers, trustformers-training, trustformers-c); `Tensor::Torch` was never constructed anywhere in the workspace, so nothing in this crate is affected — all real PyTorch interop used here (SafeTensors/pickle checkpoint loading via `weight_loading/`) is pure Rust and is kept. ToRSh is **not** adopted as a replacement now (crates.io torsh 0.1.3 pins a type-incompatible scirs2 0.5.1; torsh 0.2.0 is unpublished); a P2 workspace task tracks evaluating an optional `torsh-interop` feature in 0.3.x once torsh 0.2.0 ships on crates.io. Candle sub-decision: the unused `candle-nn` workspace dep is dropped now, the `candle` feature/variant is kept through 0.2.0 (it is in every `full` set), and implement-vs-remove is decided in 0.3.x.

### OxiCUDA GPU migration (scirs2-core gpu → OxiCUDA)
- [x] **[P1]** Wire GPT-2 and GPT-NeoX forwards through the CUDA-resident attention path (done 2026-07-06)
  - Depended on trustformers-core's CUDA-resident attention chain (CUDA-7), which landed this session (`gpu_ops/cuda/oxicuda/attention.rs`: gather/RoPE/softmax/attention prefill+decode/KV-cache concat, all resident `gpu_to_gpu` methods).
  - GPT-2 (`gpt2/model/model_blocks.rs`): added `cuda_resident_attention`, mirroring the existing Metal fast path — bulk causal prefill on an empty cache, exact per-token decode against a resident `Tensor::CUDA` cache (key transposed `[1,H,d,kv]`, value `[1,H,kv,d]`); host-format caches and non-batch-1 inputs still take the existing host path.
  - GPT-NeoX (`gpt_neox/model.rs`): added `cuda_resident_forward` (NeoX-packing QKV gather → device RoPE → K^T → per-head causal prefill → head merge → resident dense projection), replacing the previous per-sub-block CPU round-trip; prefill-only, since the NeoX `Layer` trait has no KV-cache plumbing (pre-existing, unaffected).
  - Verified: `cargo check`/`clippy --all-targets` clean (zero warnings) on `trustformers-core` and `trustformers-models` with `cuda` and default features; `nextest` core cuda-tagged tests 163/163, wider matmul/gemm/buffer/cuda/gather/concat/pitched set 205/205, `trustformers-models` full suite with `cuda` feature 1089/1089 (hardware execution itself not verifiable — no CUDA device on the build host; correctness rests on source-verified kernel signatures + CPU-parity availability-gated tests that skip cleanly without a GPU).
  - Deferred/out of scope, tracked separately: GPT-2 batch>1 and non-F32 resident dtypes still fall back to host; `attention_mask` is ignored on the resident path (matches the Metal precedent); multi-token continuation behind a cache runs a per-token decode loop rather than one fused rectangular-causal kernel (no such upstream kernel exists yet).
  - Evidence: `trustformers-models/src/gpt2/model/model_blocks.rs`; `trustformers-models/src/gpt_neox/model.rs`; `trustformers-core/src/gpu_ops/cuda/oxicuda/attention.rs`

#### Post-0.2.0 (0.3.x)
- [ ] **[P2]** Extend device-aware GPU forward beyond GPT-2/RetNet (record-only follow-up for 0.3.x)
  - GPU forward is wired end-to-end only for GPT-2 and RetNet today. Factor a device-aware attention/linear helper (already sketched as task 5 in the root TODO.md, lines 154-164) so llama, gpt_neox, and the other model families pick up the resident CUDA/Metal path without per-model duplication.
  - Evidence: `trustformers-models/README.md:11,251`; root `TODO.md:154-164`

### PyTorch (tch) dependency removal
No tasks in this crate — the `torch` feature never reached trustformers-models (no `cfg(feature = "torch")` here), and the PyTorch checkpoint loaders in `src/weight_loading/` are pure Rust and unaffected. The deletion itself (workspace Cargo.toml, trustformers-core, trustformers, trustformers-training, trustformers-c) and the P2 `torsh-interop` evaluation for 0.3.x are tracked in the root TODO.md and trustformers-core/TODO.md.

---

## Future Enhancements

### High Priority
- [x] Wire up Swin Transformer (planned 2026-07-05)
  - Goal: expose the already-complete (2502 lines, 75 tests) Swin implementation.
  - Design: add `swin = []` to Cargo.toml features; #[cfg(feature = "swin")] pub mod swin; + gated pub use, mirroring vit's exact wiring. Add "swin" to `all`.
  - Files: Cargo.toml, lib.rs.
  - Tests: cargo nextest run --features swin (75 pre-written tests run for the first time); cargo build --no-default-features (gate must be inert when off).
  - Risk: none — zero cross-feature coupling confirmed.
- [x] Wire up DeiT (planned 2026-07-05)
  - Goal: expose the already-complete (1692 lines, 54 tests) DeiT implementation.
  - Design: identical shape to Swin — `deit = []`, gated pub mod/pub use (include layer-level types for parity with vit's re-export), add to `all`.
  - Files: Cargo.toml, lib.rs.
  - Tests: cargo nextest run --features deit; spot-check the distillation-token path is covered by the existing 54 tests.
  - Risk: none.
- [x] Delete legacy qwen2/ (planned 2026-07-05)
  - Goal: remove the superseded, unreferenced legacy Qwen-2 implementation.
  - Design: delete src/qwen2/ (5 files, 2090 lines, 112 tests) — confirmed zero references anywhere in the workspace outside itself. qwen2_5 is the architectural successor (note: its weight loading is also a no-op today, just silently rather than loudly — not a blocker to this deletion).
  - Files: delete trustformers-models/src/qwen2/; edit TODO.md, README.md references.
  - Tests: cargo build --all-features (no-op diff expected); grep for dangling qwen2:: references post-deletion.
  - Risk: none.
- [x] Fix 6 vestigial Cargo features (mamba, rwkv, s4, stablelm, falcon, linformer) (planned 2026-07-05)
  - Goal: these 6 features actually gate their modules (currently zero #[cfg(feature=...)] occurrences for any of them).
  - Design: add #[cfg(feature = "X")] to each of the 6 pub mod lines and their corresponding pub use lines in lib.rs.
  - Files: trustformers-models/src/lib.rs (12 line edits).
  - Tests: CRITICAL — cargo build -p trustformers-models --no-default-features must still succeed, then --no-default-features --features mamba etc, then full --all-features regression.
  - Risk: the no-default-features build is the only way to catch a hidden cross-reference from always-on code; don't skip it.
- [x] Add llama3_2/mistral_v3 to `all` meta-feature (planned 2026-07-05)
  - Goal: --features all actually includes these two.
  - Design: confirmed via git blame this was an accidental git-history omission, not intentional (5 of 7 "incomplete weight loading" architectures are already in `all`, so that was never the exclusion criterion). Add "llama3_2", "mistral_v3" to the all=[...] list in Cargo.toml.
  - Files: trustformers-models/Cargo.toml.
  - Tests: cargo build --features all + cargo nextest run --features all.
  - Risk: none.
- [x] Complete weight loading for 5 of the 7 architectures listed under [Known Limitations](#known-limitations) — **done, verified 2026-08-18**: `llama3`, `mistral_v3`, `phi2`, `yi`, `starcoder2` all load real checkpoints now. `llama3_2` and `deepseek` remain, each deliberately: see the Known Limitations entry for the specific architectural reason.
- [x] Add weight loading for `swin`/`deit` — **done, verified 2026-08-18**: both implement `trustformers_core::traits::Model` with a real `load_pretrained_report` (`Checkpoint::from_reader` + `load_from_checkpoint`) and dedicated tests for each architecture's tensor-naming quirks.
- [x] Implement contrastive search generation for GPT-2 — **done, verified 2026-08-18**: `GenerationMode::ContrastiveSearch` routes to a real `generate_contrastive_internal`; the design notes below (hidden-state field, scoring formulas, per-candidate lookahead) describe the plan that was followed — kept for anyone reading the implementation later, not because the work is still open.
  - Prerequisites: hidden-state access does not currently exist at the point generation strategies run — Gpt2LMOutput only carries logits/past_key_values. Must add a hidden_states: Tensor field and update both construction sites (Model::forward and forward_with_cache in model_core.rs) to clone-before-consume.
  - Design: add a hidden-state-aware sibling to get_next_token_logits in generation.rs; implement generate_contrastive_search_internal following the existing generate_greedy_internal/generate_beam_search_internal dispatch shape. Port the scoring formulas (cosine_similarity, contrastive_score) from the orphaned trustformers-training/src/contrastive_search/mod.rs (pure math — reimplement, do not add a dependency edge between the sibling crates). Scope explicitly to a correct-but-not-cache-optimized first version: real per-candidate lookahead forward pass (k+1 forward passes per step), not a shortcut reusing the context's hidden state for all k candidates.
  - Files: trustformers-models/src/gpt2/model/model_core.rs, gpt2/generation.rs, generation_utils.rs.
  - Tests: unit tests for scoring helpers in isolation; integration test with ContrastiveSearch{top_k:4, alpha:0.6} on a tiny model asserting no panics and fewer immediate token repeats than greedy on a repetitive tiny model; regression test locking in Gpt2LMOutput.hidden_states.shape().
- [ ] Add BEiT (BERT pre-training for image transformers)
- [ ] Add DINOv2 (self-supervised ViT with DINO pretraining)
- [ ] Add SAM (Segment Anything Model)
- [ ] Add EVA (Exploring the Limits of Masked Visual Pre-training)

### New Models
- [ ] Add BioMedLM / BioGPT (biomedical language model)
- [ ] Add FinBERT (financial sentiment analysis)
- [ ] More efficient architectures — **refinement needed**: which efficiency axis (memory, latency, throughput)? Candidates: MEGALODON, GLA, RecurrentGemma
- [ ] Latest research architectures as they emerge — **refinement needed**: this item requires specific architecture selection; list candidates and add separate items rather than tracking this generically

### Optimizations
- [ ] Kernel opt: LLaMA fused RoPE+projection (reduces memory bandwidth)
- [ ] Kernel opt: Mistral sliding-window attention fused kernel
- [ ] Kernel opt: BERT fused attention+layer_norm kernel
- [ ] Kernel opt: T5 cross-attention cache optimization
- [ ] Quantization: FP8 (E4M3/E5M2) for inference on H100/A100-class GPUs
- [ ] Quantization: MX microscaling (MXFP4/MXFP6) for next-gen hardware
- [ ] Quantization: AWQ (Activation-aware Weight Quantization) — currently only a conceptual per-element error-bound unit test exists (`advanced_quantization.rs`), no real activation-statistics calibration pipeline
- [ ] Quantization: GPTQ full support — currently only a conceptual per-group error-bound unit test exists, no real calibration/Hessian-based quantization pipeline
- [ ] Memory usage improvements — **refinement needed**: target peak RSS reduction %, which model family?
- [ ] Faster weight loading

---

## Development Guidelines

### Adding a New Model

A `templates/` directory at the crate root already provides scaffolding for this: `transformer_model_template.rs`, `cnn_model_template.rs`, `custom_model_template.rs`, `generate_model.py`, plus `STEP_BY_STEP_TUTORIAL.md`, `ARCHITECTURE_PATTERNS.md`, `BEST_PRACTICES.md`, and `COMMON_PITFALLS.md`. Start there before following the manual checklist below.

**Step-by-step checklist:**

1. **Create Module Structure**
   ```
   src/your_model/
   ├── config.rs    # Configuration struct
   ├── model.rs     # Base model implementation
   ├── tasks.rs     # Task-specific heads
   └── mod.rs       # Module exports
   ```

2. **Implement Configuration**
   - Create `YourModelConfig` struct
   - Implement `Config` trait from trustformers-core
   - Add `validate()` method for config validation
   - Support `from_pretrained` for HuggingFace compatibility

3. **Implement Base Model**
   - Create `YourModel` struct with layers
   - Implement `Model` trait from trustformers-core
   - Add `forward()` method for inference
   - Use only trustformers-core abstractions (no external deps)

4. **Implement Task Heads**
   - `YourModelForCausalLM` for text generation
   - `YourModelForSequenceClassification` for classification
   - `YourModelForTokenClassification` for NER
   - Other task-specific variants as needed

5. **Add Weight Loading**
   - Implement `load_from_path()` method
   - Use weight loading infrastructure from `weight_loading/`
   - Handle SafeTensors, PyTorch, JSON, GGUF formats
   - Add proper error handling and logging — if not landing immediately, return a clear `Err(...)` (as the 7 architectures under Known Limitations do) rather than a `todo!()`/`unimplemented!()` panic

6. **Add Feature Gate**
   - Add to `Cargo.toml`: `your_model = []`
   - Use `#[cfg(feature = "your_model")]` guards on **both** the `pub mod` declaration and the `pub use` re-export in `lib.rs` — several existing modules skip the first of these (see Known Limitations); don't repeat that pattern

7. **Export Types**
   - Export in `src/lib.rs`
   - Add to relevant feature gates, and to the `all` meta-feature

8. **Write Tests**
   - Compare outputs with HuggingFace implementation
   - Test weight loading
   - Test forward pass correctness
   - Test generation (if applicable)

9. **Document**
   - Add rustdoc with architecture description
   - Include usage examples (guard with `no_run` if the example constructs a large/expensive model)
   - Document any limitations or special requirements

### Code Standards
- **Use only trustformers-core abstractions** (no external dependencies directly)
- **File size limit:** <2000 lines per file (currently satisfied crate-wide — largest file is under this threshold)
- **Error handling:** Use `Result<T, TrustformersError>`
- **Testing:** Compare with HuggingFace for numerical validation
- **Naming:** snake_case for all identifiers

### Build & Test Commands

```bash
# Build specific model
cargo build -p trustformers-models --features llama

# Test specific model
cargo nextest run -p trustformers-models --features llama

# Test all models
cargo nextest run -p trustformers-models --all-features

# Check compilation
cargo check -p trustformers-models --all-features
```

---

**Last Updated:** 2026-08-18 — weight-loading and generation gaps re-verified against source; 5 architectures (`llama3`, `mistral_v3`, `phi2`, `yi`, `starcoder2`) and `swin`/`deit` moved from "gap" to "complete" since the 2026-07-09 entry below, and GPT-2 contrastive search is implemented. Test count re-measured: 1,681 passed / 0 failed / 28 skipped (`cargo nextest run -p trustformers-models`, default features), superseding the ~4,479/18,102 figures elsewhere in this file. Previous: 2026-07-09 — version bumped to 0.2.1; documentation corrected for internal consistency (Swin/DeiT wiring, the 6-feature vestigial-flag fix, legacy-qwen2 deletion, and the `all`-meta-feature fix were already implemented in source but several sections here still described them as open issues — now corrected against source). Previous: 2026-07-06 0.2.0 OxiCUDA GPU migration task (GPT-2/GPT-NeoX CUDA-resident attention) completed and verified; tch/torch removal decision recorded previously; previous baseline: 0.1.4 Alpha Release (53 feature-gated architectures + 8 always-on bonus architectures, ~4,479 tests passing, 0 stubs, ~5,165 public API items)
**Status:** Alpha
**Model Count:** 55 feature-gated architectures + 8 always-on architectures, 51/55 with complete weight loading as of 2026-08-24 (`llama3_2`, `deepseek`, and `deepseek_v2` honestly refuse with a named reason; `s4` has no loading path at all — corrected from an earlier "53/55" that double-counted `deepseek_v2`/`s4` as complete, see "Per-Model Weight Loading Status" above); legacy Qwen2 deleted
