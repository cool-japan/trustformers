# trustformers-models TODO List

**Version:** 0.1.4 (Alpha) | **Date:** 2026-07-02 | **Tests:** ~4,479 passing | **SLoC:** 151,766 | **Stubs:** 0 | **Public API items:** ~5,165

## Overview

The `trustformers-models` crate provides implementations of **53 feature-gated transformer architectures**
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

### Implementation Status (reality-checked against source 2026-07-01)
- ALPHA RELEASE — all 53 feature-gated architectures implemented, 0 genuine stubs (`todo!()`/`unimplemented!()`/`FIXME` scan returned only false positives in regex-pattern comments)
- COMPREHENSIVE MODEL ZOO — 53 feature-gated architectures + 8 always-on bonus modules (cogvlm, recursive, command_r, claude, hyena, retnet, fnet, performer)
- ZERO COMPILATION ERRORS, 0 clippy warnings, 0 rustdoc warnings — clean across the workspace as of today's full `cargo nextest run --workspace --all-features`
- ~4,479 TESTS PASSING in this crate (0 failing); workspace-wide 18,102 passed / 0 failed / 119 skipped
- NO FILE EXCEEDS 2,000 LINES — refactor policy satisfied crate-wide
- WEIGHT LOADING: complete for 46 of 53 feature-gated architectures; 7 return a handled "not yet implemented" error instead of loading real checkpoints (see Weight Loading section below) — **this is a correction from the previous "Complete (27/27)" claim, which was accurate for the original 27 but did not account for architectures added since**
- **Correction**: removed a "BART" entry from this file — BART does not exist anywhere in this crate's source, Cargo.toml, or `lib.rs` (no `bart` feature, no `src/bart*`). It was documented here previously but was never actually implemented.
- **New findings**: 3 fully-implemented-but-orphaned modules (`swin/`, `deit/`, legacy `qwen2/` — none reachable from the public API), 6 vestigial Cargo feature flags that don't gate compilation, and an `all` meta-feature that omits `llama3_2`/`mistral_v3`. See [Known Limitations](#known-limitations).

### Model Categories (reality-checked counts)
- **Encoder Models:** 6 (BERT, RoBERTa, ALBERT, DistilBERT, ELECTRA, DeBERTa)
- **Decoder Models / Modern LLMs:** 33 feature flags across GPT-2/GPT-Neo/GPT-J/GPT-NeoX, LLaMA family (llama/llama2/llama3/llama3_2/codellama), Mistral family (mistral/mistral_v3/mixtral), Gemma family (gemma/gemma2), Qwen family (qwen/qwen2_5), Phi family (phi3/phi2/phi4), Falcon family (falcon/falcon2), StableLM, DeepSeek family (deepseek/deepseek_v2), InternLM2, OPT, Granite, Aya, Jamba family (jamba/jamba2), Nemotron, Yi, StarCoder2 — plus always-on Command-R and Claude-inspired
- **Encoder-Decoder / Speech / Diffusion-adjacent:** T5, Whisper, SD3 (text-encoder pipeline only)
- **Vision Models:** 2 shipped (ViT, CLIP) + 2 fully implemented but orphaned (Swin, DeiT — not wired into `lib.rs`)
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
- **Qwen2.5** (`qwen2_5`): current-generation Qwen (note: an older, unwired `qwen2` also exists in source — see Known Limitations)

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
- **DeepSeek-V2** (`deepseek_v2`): weight loading complete, unaffected by the v1 gap

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

#### Swin Transformer and DeiT — implemented but not wired up
- `src/swin/` (2,502 lines): `SwinConfig` (tiny/small/base/base-384 presets), `SwinModel`, `SwinForImageClassification`
- `src/deit/` (1,692 lines): `DeiTConfig`, `DeiTModel`, `DeiTForImageClassification` (with distillation token)
- Neither has a `pub mod` declaration in `lib.rs` nor a Cargo feature — currently unreachable dead code despite being fully written. See [Future Enhancements](#future-enhancements).

---

### State-Space, Linear- and Efficient-Attention Models

#### S4 (Structured State Space)
- HiPPO initialization (LEGS/LEGT/LAGT/Fourier), O(N log N) via FFT convolution, diagonal-plus-low-rank structure

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

### Per-Model Weight Loading Status (reality-checked 2026-07-01)
- **Complete (46/53 feature-gated architectures)**: BERT, RoBERTa, ALBERT, DeBERTa, DistilBERT, ELECTRA, GPT-2, GPT-Neo, GPT-J, GPT-NeoX, LLaMA, LLaMA-2, CodeLlama, Mistral, Mixtral, Gemma, Gemma-2, Qwen, Qwen2.5, Phi-3, Phi-4, Falcon, Falcon2, StableLM, DeepSeek-V2, InternLM2, T5, Whisper, SD3, ViT, CLIP, BLIP-2, LLaVA, DALL-E, Flamingo, Linformer, Mamba, Mamba-2, RWKV, S4, Opt, Granite, Aya, Jamba, Jamba2, Nemotron (all 46 have a real, working Cargo feature flag — note Linformer's flag is one of the vestigial ones, see below, but its weight loading itself is complete) — plus, separately, the 8 always-on bonus architectures that have no Cargo feature at all (CogVLM, Command-R, Claude, Recursive Transformers, Hyena, RetNet, FNet, Performer) also load/construct correctly. (Note: Llama-3.2 is explicitly **not** in either list — see the gap entry directly below.)
- **NOT yet implemented (7/53)** — verified via source scan for `"not yet implemented"` error strings, one file each: `llama3` (LLaMA-3), `llama3_2` (Llama-3.2), `mistral_v3` (Mistral v0.3), `phi2` (Phi-2), `deepseek` (DeepSeek v1), `yi` (Yi), `starcoder2` (StarCoder2)
- **Orphaned, same gap**: the unwired legacy `qwen2/` module also has an incomplete weight loader (moot until/unless it's wired up or removed)

---

## Model Features

### Common Components
- Multi-head attention (MHA, GQA, MQA), RoPE/absolute/relative position embeddings, LayerNorm and RMSNorm, dropout (train/inference modes), FFN variants (SwiGLU, GeGLU), residual connections, causal + padding attention masks

### Task-Specific Heads
- **CausalLM** (GPT-2, LLaMA, etc.), **MaskedLM** (BERT), **SequenceClassification**, **TokenClassification** (NER/POS), **QuestionAnswering** (SQuAD-style), **ImageClassification** (vision models)

### Generation Support
- Greedy decoding, beam search, temperature/top-k/top-p/min-p sampling, constrained/guided (CFG) generation, speculative decoding, streaming token-by-token output
- **Known gap**: contrastive search is not yet implemented for GPT-2 (`src/gpt2/generation.rs`)

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

- **6 vestigial Cargo feature flags** (`mamba`, `rwkv`, `s4`, `stablelm`, `falcon`, `linformer`): declared in `Cargo.toml` but their `pub mod` in `lib.rs` has no matching `#[cfg(feature = ...)]` — these always compile in regardless of flag state. Either add the missing `#[cfg(...)]` guards or remove the now-decorative Cargo.toml entries.
- **3 orphaned, fully-written modules**: `src/swin/` (2,502 lines), `src/deit/` (1,692 lines), legacy `src/qwen2/` (2,090 lines) have no `pub mod` anywhere in `lib.rs` — unreachable from the public API today.
- **`all` meta-feature gap**: does not include `llama3_2` or `mistral_v3` (in addition to the intentional `cuda`/`metal` exclusion).
- **Weight-loading gaps**: 7 of 53 feature-gated architectures return a handled error instead of loading real checkpoints (`llama3`, `llama3_2`, `mistral_v3`, `phi2`, `deepseek`, `yi`, `starcoder2`).
- **GPT-2 generation gap**: contrastive search not yet implemented.
- **GPU coverage within this crate**: real `#[cfg(feature = "cuda"/"metal")]` code paths verified only in `gpt2` and `gpt_neox`; all other architectures run CPU/`f32` regardless of GPU features, matching the workspace-wide GPU maturity notes.
- **No `AutoModel`/`from_pretrained` dispatcher** — callers construct concrete model types directly.
- Some multimodal models (Flamingo, CogVLM) have complex architectures; weight mapping covers all documented components, but coverage of undocumented/edge-case checkpoint layouts is unverified.
- Alpha status: API surface may still evolve before a Stable designation.

---

## Future Enhancements

### High Priority
- [ ] Wire up Swin Transformer: add `#[cfg(feature = "swin")] pub mod swin;` to `lib.rs`, add a `swin` feature to `Cargo.toml`, re-export `SwinConfig`/`SwinModel`/`SwinForImageClassification` — **implementation already exists** (2,502 lines), this is integration work, not new development
- [ ] Wire up DeiT similarly (`deit` feature) — **implementation already exists** (1,692 lines, includes distillation token)
- [ ] Resolve legacy `qwen2/`: either remove it (superseded by `qwen2_5`) or finish its weight loader and wire it up as a distinct feature — currently dead code with an unresolved gap either way
- [ ] Fix the 6 vestigial feature flags (add real `#[cfg]` gates or drop the unused Cargo.toml entries)
- [ ] Add `llama3_2` and `mistral_v3` to the `all` meta-feature (or document why they're intentionally excluded)
- [ ] Complete weight loading for the 7 architectures listed under [Known Limitations](#known-limitations): `llama3`, `llama3_2`, `mistral_v3`, `phi2`, `deepseek`, `yi`, `starcoder2`
- [ ] Implement contrastive search generation for GPT-2
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

**Last Updated:** 2026-07-02 — 0.1.4 Alpha Release (53 feature-gated architectures + 8 always-on bonus architectures, ~4,479 tests passing, 0 stubs, ~5,165 public API items)
**Status:** Alpha
**Model Count:** 53 feature-gated architectures + 8 always-on architectures, 46/53 with complete weight loading (7 pending), 3 additional architectures implemented but not yet wired into the public API (Swin, DeiT, legacy Qwen2)
