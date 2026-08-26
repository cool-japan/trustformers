# TrustformeRS 🦀

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/version-0.2.2-blue.svg)](https://github.com/cool-japan/trustformers)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

A high-performance, memory-safe Rust implementation of Hugging Face Transformers. TrustformeRS brings the power of transformer models to the Rust ecosystem with zero-cost abstractions, fearless concurrency, and deployment flexibility from edge to cloud.

> **Project Status (alpha)**: TrustformeRS 0.2.1 (last verified 2026-08-26) is a large Pure-Rust transformer stack — 3,117 Rust files, ~1.57M lines (~1.31M lines of code, via `tokei`) across 10 crates and 49+ transformer architectures — together with multi-platform packaging (WebAssembly, server REST/gRPC/GraphQL, mobile iOS/Android, and RLHF/DPO training scaffolding).
>
> **Honest maturity note**: today's compute path is primarily **CPU and `f32`**. F16/BF16 are supported as a storage/serialization format but are upcast to `f32` for arithmetic (native low-precision kernels are on the roadmap). GPU acceleration is **real** (CUDA via the Pure-Rust `oxicuda` backend, Metal via `objc2`/`oxicuda-metal`, WebGPU via `wgpu`) but is currently wired end-to-end **only for GPT-2 and RetNet** (0.2.0 added a GPU-resident CUDA attention path for GPT-NeoX too, but it is **prefill-only** — no KV-cached decode, since its `Layer` trait carries no cache); the remaining backends (ROCm, Vulkan, OpenCL) are feature-gated and experimental, and **TPU is a placeholder, not implemented**. Several newer architectures are still being completed. See [Development Status](#-development-status) for the precise maturity of each area.

## 🚀 Why TrustformeRS?

- **🏎️ Performance**: Leverages Rust's zero-cost abstractions, SIMD optimizations, and efficient memory management
- **🔒 Safety**: Memory-safe by design with Rust's ownership model - no more segfaults or memory leaks
- **📦 Portability**: Deploy anywhere from WebAssembly to embedded devices to GPU clusters
- **🔧 Control**: Explicit resource management following SciRS2's Core Usage Policy
- **🤝 Compatibility**: Loads Hugging Face model formats directly

## 📊 Performance

No benchmark harness in this repository has produced a TrustformeRS-vs-HF-Transformers comparison number — an earlier version of this section carried a table of specific per-model millisecond figures and speedup multipliers that no automated gate in this repository ever measured. `trustformers-tokenizers`' own migration guides (`trustformers-tokenizers/docs/migration/`) removed an equivalent fabricated table for the same reason: numbers presented as measurements that no harness here produced. Benchmark sources do exist (`benches/`, plus per-crate `benches/` directories) — run `cargo bench --all-features` yourself and compare against a real HF Transformers install on your own hardware if you need a number; do not trust a number quoted here that isn't accompanied by the command that produced it.

## 🏗️ Architecture

TrustformeRS follows a modular workspace structure inspired by Hugging Face Transformers:

```
trustformers/
├── trustformers-core/      # Core traits and tensor abstractions  (178,532 SLoC, Stable)
├── trustformers-models/    # 49+ model implementations           (188,417 SLoC, Alpha)
├── trustformers-tokenizers/# BPE, WordPiece, SentencePiece       ( 45,324 SLoC, Stable)
├── trustformers-optim/     # 20+ optimizers and LR schedulers    ( 65,983 SLoC, Stable)
├── trustformers-training/  # Distributed training, RLHF/DPO      ( 83,319 SLoC, Stable)
├── trustformers-serve/     # REST/gRPC/GraphQL serving           (273,756 SLoC, Stable)
├── trustformers-wasm/      # WebAssembly + WebGPU deployment     ( 48,359 SLoC, Stable)
├── trustformers-mobile/    # iOS/Android deployment              (108,021 SLoC, Alpha)
├── trustformers-debug/     # Profilers, visualizers, TensorBoard ( 88,454 SLoC, Alpha)
└── trustformers/           # High-level integration crate        (123,252 SLoC, Alpha)
```

**Total**: ~1.20M SLoC across these 10 crates; **3,046 Rust files, ~1.53M lines total (~1.28M lines of code)** across the full repository including bindings/examples/tooling (via `tokei`, 2026-08-24). Both workspace-wide figures use the same `tokei` invocation shape as the 2026-08-18 count they're compared against, so that comparison is apples-to-apples: `trustformers-serve` and `trustformers-mobile` shrank the most — a placeholder resource-manager tree (5,972 lines) deleted outright in `trustformers-serve`, and roughly 4,800 lines of dead/duplicate scaffolding deleted from `trustformers-mobile` — while `trustformers-core`, `trustformers-models`, and `trustformers` (umbrella) grew from real implementation work (a Metal buffer-lifetime RAII type, model checkpoint-loading fixes, and dead-code-turned-real-accessors, respectively). The **per-crate** figures in the tree above are less certain: three crates no wave touched this cycle (`trustformers-wasm`, `trustformers-debug`, `trustformers-tokenizers`) all show a double-digit-percent *decrease* from their last-recorded per-crate figure despite no known code deletion — most likely a difference in how the per-crate number was computed this time (e.g. `tokei <crate-dir>` scope) rather than an undocumented removal, but this was not individually confirmed for any of the three. 100% Pure Rust source (COOLJAPAN Policy) — default-feature builds are C/C++-free for every crate except `trustformers-serve` (accepted exception: rustls/aws-lc-rs TLS for the HTTP server).

### Design Principles

1. **Trait-based abstractions**: Models, layers, and tokenizers implement common traits for composability
2. **Feature-gated backends**: Choose between CPU, GPU, or WebAssembly targets
3. **Zero-copy model loading**: Memory-mapped weights with SafeTensors format
4. **Explicit parallelism**: You control thread and GPU usage, not the library

## 🚦 Quick Start

### Installation

```toml
[dependencies]
trustformers = "0.2.2"
```

### Basic Usage

```rust
use trustformers::prelude::*;
use trustformers::{AutoModel, AutoTokenizer, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model and tokenizer
    let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;
    let model = AutoModel::from_pretrained("bert-base-uncased")?;

    // Tokenize input
    let tokenized = tokenizer.encode("Hello, Rust world!")?;

    // AutoModel's `Model` impl is Tensor-in/Tensor-out (uniform across every
    // architecture it wraps), so wrap the token IDs as a Tensor before running
    // inference.
    let ids: Vec<f32> = tokenized.input_ids.iter().map(|&id| id as f32).collect();
    let len = ids.len();
    let inputs = Tensor::from_vec(ids, &[len])?;

    // Run inference
    let outputs = model.forward(inputs)?;

    println!("Output shape: {:?}", outputs.shape());
    Ok(())
}
```

*(For architecture-specific outputs with named fields — e.g. `BertModelOutput { last_hidden_state, pooler_output }` — construct the concrete model type directly, such as `trustformers::BertModel`, whose `forward` takes the `TokenizedInput` from `tokenizer.encode(..)` straight through.)*

### Pipeline API

```rust
use trustformers::pipeline;

// `pipeline(task, model, options)` — pass `None, None` for the defaults.
let classifier = pipeline("sentiment-analysis", None, None)?;
let result = classifier.__call__("I love writing Rust code!".to_string())?;
// Output: PipelineOutput::Classification([ClassificationOutput { label: "POSITIVE", score: 0.999 }])

// This crate ships 38 task-specific pipeline modules; they are not uniformly real. The
// 7 below (spot-checked 2026-08-24: each calls a real AutoModelType::*.forward(...), not
// a canned response) are the commonly-used NLP set and are real:
// - text-generation
// - token-classification (NER)
// - question-answering
// - fill-mask
// - summarization
// - translation
//
// 10 of the 38 — audio_generation, document_classification, feature_extraction,
// image_segmentation, speech_recognition, table_question_answering, text_to_image,
// video_classification, visual_grounding, zero_shot_audio_classification — are mock
// pipelines pending real model backends (see trustformers/TODO.md's "Mock Pipelines"
// note). The remaining ~21 were not individually re-verified this pass.
```

## 🏛️ Model Zoo

### Currently Supported (49+ architectures!)

#### Encoder Models
| Model | Variants | Tasks |
|-------|----------|-------|
| BERT | base, large | Masked LM, Classification, Token Classification, QA |
| RoBERTa | base, large | Same as BERT |
| DistilBERT | base | Same as BERT (faster) |
| ALBERT | base, large | Same as BERT (parameter sharing) |
| ELECTRA | base, large | Discriminative pretraining |
| DeBERTa | base, large | Disentangled attention |

#### Decoder Models
| Model | Variants | Tasks |
|-------|----------|-------|
| GPT-2 | small, medium, large, xl | Text Generation |
| GPT-Neo | 125M, 1.3B, 2.7B | Text Generation |
| GPT-J | 6B | Text Generation |
| GPT-NeoX | various | Text Generation |
| LLaMA | 7B, 13B, 30B, 65B, 70B | Text Generation |
| Mistral | 7B | Text Generation |
| Gemma | 2B, 7B | Text Generation |
| Qwen | 1.8B, 7B, 14B | Text Generation |
| Phi-3 | mini, small, medium | Text Generation |
| Falcon | 7B, 40B | Text Generation |
| StableLM | 1.6B–12B | Text Generation |

#### Encoder-Decoder Models
| Model | Variants | Tasks |
|-------|----------|-------|
| T5 | small, base, large, 3B, 11B | Text-to-Text Generation |

#### Vision & Multimodal Models
| Model | Variants | Tasks |
|-------|----------|-------|
| ViT | tiny, small, base, large | Image Classification |
| Swin Transformer | tiny, small, base, large | Hierarchical Image Classification (shifted windows) |
| DeiT | tiny, small, base, large | Image Classification (w/ distillation token) |
| CLIP | base, large | Text-Image Matching |
| BLIP-2 | various | Vision-Language |
| LLaVA | various | Visual Instruction Tuning |
| DALL-E | various | Text-to-Image Generation |
| Flamingo | various | Visual Language Model |

#### State-Space & Linear Attention Models
| Model | Complexity | Tasks |
|-------|------------|-------|
| Mamba | O(N) | Long-context Generation |
| RWKV | O(N) | Recurrent Language Modeling |
| S4 | O(N log N) | Long-range Sequence Modeling |

## ⚡ Performance Features

TrustformeRS includes state-of-the-art optimizations not mentioned in typical documentation:

- **FlashAttention & FlashAttention-2**: O(N) memory complexity for attention
- **PagedAttention**: Efficient KV cache management for long sequences
- **INT8/INT4 Quantization**: GPTQ and AWQ quantization methods
- **Mixed Precision (partial)**: FP16/BF16 weight storage + casting/loss-scaling utilities; core arithmetic currently upcasts to FP32 (native FP16/BF16 compute kernels are on the roadmap)
- **ZeRO Optimization**: All 3 stages for distributed training
- **SIMD Operations**: Leveraging SciRS2 for vectorized computations
- **Tensor Parallelism**: Split large models across multiple GPUs
- **Gradient Checkpointing**: Trade compute for memory efficiency

## 🚀 Deployment Options

TrustformeRS supports multiple deployment targets:

- **WebAssembly**: Browser deployment (trustformers-wasm, Stable)
  - WebGPU acceleration support
  - JavaScript/TypeScript bindings
  - React/Vue component-ready

- **Server**: Production-ready API serving (trustformers-serve, Stable)
  - REST, gRPC, and GraphQL endpoints
  - Dynamic batching with Kubernetes deployment manifests
  - Docker containers and auto-scaling support

- **Training**: Full training infrastructure (trustformers-training, Stable)
  - RLHF and DPO training support
  - Distributed training with ZeRO optimization
  - Mixed precision (FP16/BF16)

- **Mobile**: Native mobile deployment (trustformers-mobile, Alpha)
  - iOS framework with Core ML and Metal acceleration
  - Android library with NNAPI and Vulkan support
  - React Native, Flutter, and Unity integrations

- **Edge**: Export to optimized formats
  - ONNX export/import
  - GGUF format support
  - Quantized models (INT8/INT4, GPTQ, AWQ) for embedded devices

## 🛠️ Advanced Usage

### Custom Model Implementation

```rust
use trustformers_core::{Model, Layer, Config};
use trustformers_core::layers::Embedding; // real building block
use trustformers_core::traits::TokenizedInput;

// `TransformerEncoder`, `Pooler`, `MyConfig`, and `ModelOutput` are illustrative
// types you define yourself, typically composed from real trustformers_core
// building blocks such as `MultiHeadAttention`, `LayerNorm`, and `FeedForward`
// (all in `trustformers_core::layers`).
struct MyTransformer {
    embeddings: Embedding,
    encoder: TransformerEncoder,
    pooler: Pooler,
}

impl Model for MyTransformer {
    type Config = MyConfig;
    type Input = TokenizedInput;
    type Output = ModelOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden_states = self.embeddings.forward(input.input_ids)?;
        let encoded = self.encoder.forward(hidden_states)?;
        let pooled = self.pooler.forward(encoded.clone())?;

        Ok(ModelOutput { hidden_states: encoded, pooled_output: pooled })
    }
}
```

### GPU Acceleration

GPU backends are real (CUDA via the Pure-Rust `oxicuda` backend, Metal via `objc2`/`oxicuda-metal`, WebGPU via `wgpu`) but are **currently wired end-to-end only for GPT-2 and RetNet** (with persistent KV-cached decode). Enable the matching feature flag (`metal` on macOS, `cuda` on Linux/Windows) and build a supported model on a GPU `Device`; its `forward` then runs the linear/attention path on-device with a persistent KV cache. GPT-NeoX also gained a GPU-resident CUDA attention path in 0.2.0, but it is **prefill-only** (no KV-cached decode yet — its `Layer` trait carries no cache), so it falls back to the CPU-download path for incremental decoding.

> **CUDA runtime-verified (2026-07-01):** the CUDA backend was migrated from `cudarc` to the Pure-Rust `oxicuda` (`oxicuda-blas`/`-dnn`/`-memory`/`-driver`) — the `cuda` feature now pulls in `oxicuda` instead of `cudarc` (`cuda-oxicuda` is kept only as a deprecated alias for `cuda`). 12 CPU↔CUDA golden-parity tests (GEMM, GELU, LayerNorm, causal softmax, RoPE — both host and GPU-resident paths, plus cached-weight GEMM) prove the backend is numerically correct against the CPU reference, runtime-verified **12/12 passing on a real NVIDIA RTX A4000 (CUDA 12.0)**. The GPU-resident CUDA transformer layer (LayerNorm→QKV→bias→RoPE→causal-softmax attention→proj→residual, chained via cached device buffers with no host round-trips) replaces the previous CPU-fallback placeholder. Metal compute (matmul + resident attention) similarly migrated to `oxicuda-metal`, dropping the earlier `scirs2-core` MPS dependency.

```rust
// Cargo.toml: trustformers-core = { version = "0.1", features = ["metal"] }  // or "cuda"
use trustformers_core::Device;
use trustformers_models::gpt2::{Gpt2Config, Gpt2Model};

// Construct a supported model (e.g. GPT-2) and move its weights to a GPU device.
let device = Device::Metal(0); // or Device::CUDA(0)
let mut model = Gpt2Model::new_with_device(Gpt2Config::default(), device)?;
model.weights_to_gpu(&device)?;        // Metal (use `weights_to_gpu_cuda` on CUDA)
let outputs = model.forward(inputs)?;  // attention + linear run on-device
```

> A generic `model.to_gpu()` covering **all** 49+ architectures is **not yet available** — broader coverage is tracked under [Development Status](#-development-status). For unsupported models, inference currently runs on CPU (`f32`).

### WebAssembly Deployment

```bash
# Build for WASM
cargo build --target wasm32-unknown-unknown --features wasm

# Use in JavaScript (real wasm-bindgen exports: WasmTokenizer/BertModelWasm,
# constructed from a config rather than a HuggingFace Hub name)
import init, { WasmTokenizer, TokenizerType, BertModelWasm, BertConfig } from './trustformers_wasm.js';

await init();
const tokenizer = new WasmTokenizer(TokenizerType.WordPiece);
const model = new BertModelWasm(new BertConfig());

const ids = tokenizer.encode("Hello, Rust world!", true); // -> token ID array
const hiddenStates = model.forward(ids);
```

## 🔄 Migration from Python

TrustformeRS maintains API similarity with Hugging Face Transformers for easy migration:

<table>
<tr>
<td>Python (Transformers)</td>
<td>Rust (TrustformeRS)</td>
</tr>
<tr>
<td>

```python
from transformers import (
    AutoModel, 
    AutoTokenizer
)

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)
model = AutoModel.from_pretrained(
    "bert-base-uncased"
)

inputs = tokenizer(
    "Hello world!", 
    return_tensors="pt"
)
outputs = model(**inputs)
```

</td>
<td>

```rust
use trustformers::{
    AutoModel, 
    AutoTokenizer,
    Tensor,
};

let tokenizer = AutoTokenizer::from_pretrained(
    "bert-base-uncased"
)?;
let model = AutoModel::from_pretrained(
    "bert-base-uncased"
)?;

let tokenized = tokenizer.encode(
    "Hello world!"
)?;
// AutoModel is Tensor-in/Tensor-out; wrap token IDs first.
let ids: Vec<f32> = tokenized.input_ids
    .iter().map(|&i| i as f32).collect();
let len = ids.len();
let inputs = Tensor::from_vec(ids, &[len])?;
let outputs = model.forward(inputs)?;
```

</td>
</tr>
</table>

## 🎯 Development Status

### Completed Features (v0.2.1 - 2026-08-26)
- [x] **Deleted an RS256 JWT auth bypass**: `trustformers-serve`'s orphaned auth module cluster accepted RS256-signed tokens without verifying the signature against a real key. It was never reachable from any router, so no shipped endpoint was exposed — removed rather than fixed-in-place.
- [x] **`cargo deny check advisories`/`bans`/`licenses` all pass**: the one real, currently-unpatched vulnerability (RUSTSEC-2023-0071, RSA "Marvin Attack" timing side-channel via `jsonwebtoken`) is `ignore`-listed with a documented rationale and removal path, not silently suppressed; the `[bans]` deny-list — previously an empty comment behind a schema cargo-deny 0.19 couldn't fully parse — is populated with the full COOLJAPAN banned-crate list on a current schema.
- [x] **Five deadlock-class bugs fixed**: an unconditional re-entrant deadlock in the Metal GPU `attention_gpu_to_gpu_optimized` path, a `tokio::sync::Mutex` deadlock in `DistributedDebugger::coordinate_operation`, and three `RwLock` read-read reentrancy hazards.
- [x] **`trustformers-core::gpu_ops::metal::types::MetalBufferHandle`** (new): a refcounted RAII handle that releases its Metal buffer's cache entry when the last handle drops, replacing stay-live-until-eviction semantics across every GPU-to-GPU result site `trustformers-core` owns.
- [x] **Checkpoint loading now binds real weights** for `roberta`, `albert`, `fnet`, `nemotron`, `phi4`, `mistral_v3`, `phi2`, `yi`, `starcoder2`, `llama3`, and `command_r` (previously a silent `Ok(())` left them at random-initialized weights); architectures this machinery can't yet serve faithfully (`rwkv`, `mamba`, `hyena`, `performer`, `retnet`, and others) now return a structured `not_implemented` error naming the reason instead.
- [x] **`trustformers-mobile::advanced_security` gained real post-quantum primitives**: `KyberKem` (FIPS 203 ML-KEM), `DilithiumSigner` (FIPS 204 ML-DSA), `SphincsSigner` (FIPS 205 SLH-DSA), replacing placeholder byte-tricks.
- [x] **`trustformers-serve::openai_compat` is mounted** on both production and test routers, backed by a real `OpenAiInferenceBackend` adapter — previously unmounted and unreachable from any endpoint. 27 previously-orphaned test files (+770 tests) are now compiled and running, guarded by a new source-tree orphan-detection test.
- [x] **`trustformers-serve`'s placeholder `resource_manager/` tree deleted** (11 files, 5,972 lines) — it fabricated network ports, temp-directory paths, database connection ids, GPU device stats, and monitoring efficiency. The unprefixed `ResourceManagementSystem` name now resolves directly to the real, tested `resource_management` module.
- [x] **`trustformers-wasm::performance_profiler` stops fabricating telemetry**: 22 `estimate_*`/`get_*`/`check_*`/`calculate_*` helpers that invented CPU/GPU usage, memory/FLOPs/bandwidth, battery level, power/temperature, and an "ML-powered" improvement estimate are deleted; what remains is real wall-clock duration, real WASM memory growth, and a real Battery Status API read. `multi_model_manager::warmup_model` now runs one real forward pass instead of a no-op status transition.
- [x] **`trustformers-py::training::PyTrainer::train()` now honestly refuses** instead of fabricating `train_loss`/`total_steps` unconditionally — no backward/gradient path from a loss back to model parameters exists yet anywhere in the training stack, so it names that reason rather than reporting a loss that never changes.
- [x] **Real checkpoint/vocabulary loading, no fabricated fallback**: `SentencePieceTokenizer`/`WordPieceTokenizer::from_pretrained` now hard-error naming every path probed instead of silently handing back a fabricated vocabulary; `trustformers-models::deepseek_v2`/`s4` checkpoint loaders no longer report fake success.
- [x] **~4,800 lines of dead/duplicate scaffolding removed** from `trustformers-mobile` (including a random-number fabrication factory and a device farm that invented device catalogues); 24 no-op mesh methods removed from `trustformers-serve`; an 899-line orphaned root `src/` tree with no `[package]` section removed.
- [x] **9 files split into directory modules** to stay under the workspace's 2,000-line policy limit, each confirmed under 2,000 lines afterward — see [`CHANGELOG.md`](CHANGELOG.md) for the full path list.
- [x] **`AdvancedRAGPipeline`'s self-reflection bug fixed**: reflection results were hardcoded to an empty `Vec` at the return site despite being populated during the hop loop, and an unconditional `break` made multi-hop retrieval driven by self-reflection dead code; both fixed.
- [x] **Full workspace verification (2026-08-26)**: `cargo nextest run --workspace` = **21,370 passed, 0 failed** (41 skipped) · `--all-features` = **25,883 passed, 0 failed** (113 skipped) · `cargo clippy --workspace --all-features --all-targets -- -D warnings` = 0 warnings · `cargo test --doc --workspace --all-features` = 0 failures · `cargo doc --workspace --all-features --no-deps` (`RUSTDOCFLAGS="-D warnings"`) = 0 warnings · `cargo fmt --all -- --check` = clean.

See [`CHANGELOG.md`](CHANGELOG.md) (the "[0.2.1]" section) for the full list and [`TODO.md`](TODO.md) for the wave-by-wave audit trail.

### Completed Features (v0.2.0 - 2026-07-09)
- [x] **CUDA gains a GPU-resident attention pipeline**: a new `gpu_ops::cuda::oxicuda::attention` module (QKV head-gather, RoPE, causal softmax, prefill/decode attention, KV-cache concat, residual add) gives CUDA the same fully device-resident chain the Metal backend already had. GPT-2 (`Gpt2Attention::cuda_resident_attention`) uses it for zero-host-round-trip prefill **and** incremental KV-cached decode; GPT-NeoX (`GPTNeoXAttention::cuda_resident_forward`) uses it for prefill only, since its `Layer` trait carries no KV cache.
- [x] **Batched/broadcasting CUDA matmul**: a new `gpu_ops::cuda::oxicuda::batched` module (`BatchedMatmulPlan`, a NumPy-style broadcasting-batch shape planner) replaces the previous 2D-only host-round-trip dispatcher; `Tensor::matmul` now routes N-D batched/broadcast F32 matmuls, and any operand pair involving a resident `Tensor::CUDA`, through the CUDA backend.
- [x] **Reference-counted GPU buffer lifecycle**: `gpu_ops::cuda::BufferHandle` (`OxiCudaBufferHandle`) frees a device allocation automatically when the last clone drops, replacing the previous leak-until-`clear_buffer_cache()` behavior.
- [x] **Swin Transformer and DeiT are buildable for the first time**: `trustformers-models` gained `swin`/`deit` Cargo features mounting `SwinModel`/`SwinForImageClassification` and `DeiTModel`/`DeiTForImageClassification` (with distillation-token support) — the implementations already existed in the tree but were never wired into `lib.rs`.
- [x] **`trustformers` mounts three previously-dormant subsystems and ten task pipelines** that already existed in the tree but were never `pub mod`-declared: `cache` (`VersionedCache`, TTL/LRU/LFU/size eviction), `finetuning` (`LoraConfig`/`LoraLinear`, `AdapterConfig`/`BottleneckAdapter`), `loading` (`ParallelWeightLoader`/`load_model_parallel`), and the `audio_generation`, `document_classification`, `feature_extraction`, `image_segmentation`, `speech_recognition`, `table_question_answering`, `text_to_image`, `video_classification`, `visual_grounding`, and `zero_shot_audio_classification` pipelines under `trustformers::pipeline`.
- [x] **Multi-objective hyperparameter optimization mounted**: `trustformers-training::hpo` exposes `MultiObjectiveHpo`/`ParetoFront`/`compute_pareto_front`/`hypervolume_indicator`/`non_domination_sort` plus `AutoLrSelector`/`LrRangeTest` for automatic learning-rate range tests.
- [x] **Fixed a silent CUDA correctness bug**: GPU-to-host reads (`download_buffer`, matmul/GELU/LayerNorm/softmax/RoPE) now synchronize the CUDA stream before their device→host copy — oxicuda kernels launch asynchronously on a non-blocking stream, and without this fix a fast host thread could read back a buffer before the kernel filling it had finished, risking silent zero/garbage data.
- [x] **Fixed `Device::cuda_if_available()` / `Device::best_available()`**: both always returned `Device::CPU` even on CUDA/Metal-capable hardware, because they read `scirs2_core`'s `PlatformCapabilities` (whose `cuda_available` is hardcoded `false` upstream and whose `metal_available` needs a scirs2 `metal` feature trustformers never enables); both now probe trustformers' own `oxicuda`/Metal backends directly.
- [x] **Multi-GPU device-addressing fix**: `CudaTensorData` now carries its own `device_id` instead of callers inferring it from a `Layer`'s configured `Device` or hardcoding `0`; `LayerNorm`, `Linear`, `gelu`, and `Tensor::add`/`matmul` now operate on the buffer's actual device.
- [x] **The `torch` (`tch`/libtorch) backend is removed workspace-wide** (the `torch` feature, `tch` dependency, and `Tensor::Torch` variant are gone from every crate) — advances the Pure-Rust default feature tree; `candle` remains the optional non-CPU tensor-framework integration.
- [x] Several real-vs-mock bug fixes: `SentencePieceTokenizer::from_pretrained` now loads a real SentencePiece model file instead of always returning a fabricated T5-like vocabulary; `trustformers-wasm` quantization now performs real affine (min/max, scale/zero-point) quantize-dequantize instead of a constant-multiply placeholder; `trustformers-wasm` device-capability probes (`detect_webgl_support`, `get_screen_orientation`, WebGPU capabilities) now query the real browser/device instead of returning hardcoded results; `OfflineModelPackManager::get_model_info` now queries the real HuggingFace Hub API instead of hardcoded mock metadata.
- [x] **`trustformers-models`' `all` feature aggregate fixed**: it was missing the already-implemented `llama3_2` (LLaMA 3.2) and `mistral_v3` (Mistral v0.3) model features; `--features all` now builds every supported architecture.

### Completed Features (v0.1.4 - 2026-07-01)
- [x] **CUDA backend migrated to the Pure-Rust `oxicuda`** (COOLJAPAN Pure-Rust policy), entirely replacing `cudarc` (`cuda` now pulls `oxicuda-blas`/`-dnn`/`-memory`/`-driver`; `cuda-oxicuda` kept as a deprecated alias). GPU-resident `matmul_gpu_to_gpu` confirmed genuinely zero-copy (cached `DeviceBuffer`, no host round-trip); the `cuda` feature now propagates through `trustformers-models` and the `trustformers` umbrella crate.
- [x] **12 CPU↔CUDA golden-parity tests** (GEMM, GELU, LayerNorm, causal softmax, RoPE — host and GPU-resident paths, plus cached-weight GEMM), runtime-verified 12/12 passing on a real NVIDIA RTX A4000 (CUDA 12.0).
- [x] **Real GPU-resident CUDA transformer layer**: pre-norm causal self-attention (LayerNorm→QKV→bias→RoPE→causal-softmax attention→proj→residual) chained via cached device buffers with no host round-trips, replacing the previous CPU-fallback placeholder.
- [x] **Metal GPU compute migrated to `oxicuda-metal`** (Pure Rust), dropping the `scirs2-core` MPS dependency entirely; GPU-resident matmul is zero-copy; GPT-2's feed-forward now uses a single fused matmul+bias+GELU Metal kernel (one GPU dispatch instead of three). Verified on Apple Silicon.
- [x] **Real `PyRwkvModel` / `PyMambaModel` Python classes**: `AutoModel` now loads RWKV/Mamba checkpoints correctly instead of silently falling back to BERT (the Python binding layer was also re-enabled and modernized to PyO3 0.28).
- [x] **WebGPU device/queue initialization** in the WebAssembly compute backend (`navigator.gpu` → adapter → device), falling back to CPU when no adapter is available.
- [x] **Eliminated production-code `unwrap()`/`expect()`** across the entire workspace (lock-poison recovery, proper `Result` propagation, documented infallible invariants only where truly load-bearing); no public API changes, all tests pass unchanged.
- [x] **Default feature trees are Pure-Rust** (C/C++-free) for every crate except `trustformers-serve` (HTTP server; keeps rustls/aws-lc-rs TLS, accepted exception). Networking (HuggingFace Hub downloads, remote leaderboard), debug visualization, and serve's AWS-Lambda/Swagger-UI adapters are now opt-in behind features (`hub`, `remote-leaderboard`, `visual`, `lambda`, `swagger-ui`); switched the tokenizer regex backend to pure-Rust `fancy-regex`.
- [x] **Restored gRPC proto compilation and serving** (migrated `build.rs` to the tonic 0.14 split `tonic-build`/`tonic-prost-build` API).
- [x] **Removed the legacy `cudarc`-based CUDA backend** and an orphaned, never-mounted `rope/mod.rs` module (~1,693 lines whose RoPE convention was inconsistent with the live, now parity-tested kernel).
- [x] **Full workspace verification (2026-07-01)**: `cargo nextest run --workspace --all-features` = **18,102 passed, 0 failed** (119 skipped, ~565s) · `cargo clippy --workspace --all-features --all-targets -- -D warnings` = 0 warnings/errors · `cargo doc --workspace --all-features --no-deps` (`RUSTDOCFLAGS="-D warnings"`) = 0 warnings · `cargo fmt --all -- --check` = clean.

### Completed Features (v0.1.3 - 2026-06-24)
- [x] **Gorilla time-series compression + historical-data engine** (trustformers-serve): delta-of-delta timestamp encoding plus XOR float encoding for metric series, with a real lifecycle/archival/query engine (expiry cleanup, archive/retrieve, cached query execution) replacing prior stubs
- [x] **Real concurrency-detector analytics** (trustformers-serve): working cycle/deadlock, thread, lock, and conflict detection with pattern, sharing, and risk-assessment scoring across all eight detector modules
- [x] **Interpretability tools wired in** (trustformers-debug): real SHAP, LIME, and Integrated-Gradients feature attribution exposed as `InterpretabilityAnalyzer` / `InterpretabilityConfig` / `InterpretabilityReport`
- [x] **`GlobalMemoryPool` / `ZeroCopyTensorView` / `GlobalProfiler`** (trustformers): a thread-safe aligned allocator, a bounds-checked borrowed `f32` tensor view, and a per-session profiler operation API, all re-exported from the crate root
- [x] **Real `.xlsx` export** (trustformers-debug): emits a valid OOXML workbook package via `oxiarc-archive` (Pure Rust), replacing the previous CSV-with-`.xlsx`-extension placeholder
- [x] **Real SHA-256 on the HuggingFace upload path** (via `sha2`), replacing a non-cryptographic XOR-fold stub
- [x] **Reliability fixes**: clap `-c` short-flag collisions that panicked the `load_test` / `message_queue_cli` binaries are fixed; Miri-verified memory-pool fixes for heap mis-layout deallocation (undefined behavior) and a block-reuse leak

### Completed Features (v0.1.2 - 2026-06-20)
- [x] **49+ transformer architectures** for CPU inference (BERT, RoBERTa, ALBERT, DistilBERT, ELECTRA, DeBERTa, GPT-2, GPT-Neo, GPT-J, GPT-NeoX, LLaMA, Mistral, Gemma, Qwen, Phi-3, Falcon, StableLM, T5, ViT, CLIP, BLIP-2, LLaVA, DALL-E, Flamingo, Mamba, RWKV, S4, Falcon2, Gemma2, Granite, Hyena, InternLM2, Jamba, Jamba2, Linformer, LLaMA3.2, Mamba2, Nemotron, Performer, Phi4, Qwen2.5, RetNet, SD3, StarCoder2, Whisper, xLSTM, Yi). Maturity varies — the BERT/GPT-2/LLaMA/T5/ViT families are the most exercised; several newer or experimental architectures are alpha-quality and still being completed for full numerical parity with the reference implementations.
- [x] **All major NLP pipelines** fully implemented (text-generation, classification, QA, NER, fill-mask, summarization, translation)
- [x] **Complete training infrastructure** with distributed training, ZeRO optimization, mixed precision, RLHF and DPO support
- [x] **Mobile deployment** with iOS (Core ML, Metal) and Android (NNAPI, Vulkan) support
- [x] **WebAssembly deployment** with WebGPU acceleration
- [x] **REST/gRPC/GraphQL APIs** with dynamic batching, Kubernetes deployment, and autoscaling
- [x] **Safety filtering pipeline** with configurable content moderation
- [x] **Advanced optimizations**: FlashAttention, PagedAttention, quantization (INT8/INT4/GPTQ/AWQ)
- [x] **GPU backends**: CUDA (`cudarc`) and Metal (MPS) wired into GPT-2/RetNet forward paths; WebGPU/Vulkan/OpenCL/ROCm present as feature-gated backends (experimental, not yet wired into model `forward`)
- [x] **AutoModel/AutoTokenizer** system with HuggingFace Hub integration
- [x] **Large test suite**: **18,008 tests pass** via `cargo nextest run --all-features` (119 skipped) across the full workspace, verified locally on 2026-06-24
- [x] **Debugging tools**: Profilers, visualizers, interactive debugging, TensorBoard integration
- [x] **100% Pure Rust** (COOLJAPAN Policy) - ~1,408,134 SLoC across 10 crates

### Future Enhancements

#### High Priority
- [ ] **Broader GPU model coverage**: extend the `oxicuda`/`oxicuda-metal` device-resident forward path beyond GPT-2/RetNet/GPT-NeoX to more architectures, and extend GPT-NeoX's CUDA path from prefill-only to full KV-cached decode (superseded the earlier scirs2-core MPSGraph plan — Metal now runs on `oxicuda-metal` directly, no longer blocked on scirs2-core)
- [ ] **More quantization methods**: Enhanced GGUF format, AutoGPTQ improvements
- [ ] **Additional vision transformer variants**: ViT-Huge (DeiT and Swin shipped in 0.2.0 — see the Model Zoo table)

#### Performance
- [ ] **Fused CUDA megakernel**: LayerNorm+QKV+RoPE+Attention+Proj+Residual as a single `oxicuda-dnn` kernel (today's oxicuda path executes ops individually — correct, but not fused)
- [ ] **Streaming inference**: Real-time token streaming for all generation pipelines

#### Documentation
- [ ] **Comprehensive guides**: Model implementation, deployment, optimization tuning
- [ ] **Cookbook**: Common patterns and best practices

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Adding a New Model

1. Create a new module in `trustformers-models/src/`
2. Implement the `Config`, `Model`, and task-specific heads
3. Add tests comparing outputs with Hugging Face
4. Submit a PR with benchmarks

### Performance Contributions

- Profile with `cargo-flamegraph`
- Benchmark with `criterion`
- Consider SIMD optimizations for hot paths
- Ensure thread-safety for concurrent use

## 📈 Benchmarks

Run benchmarks with:

```bash
cargo bench --all-features
```

View detailed results in `target/criterion/report/index.html`

## 🛡️ Safety and Security

- No unsafe code in public APIs (only in carefully reviewed hot paths)
- All models are `Send + Sync` for safe concurrent use
- Fuzzing tests for tokenizers
- Memory usage bounds for OOM prevention

## 📚 Documentation

- [API Documentation](https://docs.rs/trustformers)
- [Architecture Guide](docs/architecture.md)
- [Performance Tuning](docs/performance.md)
- [Model Implementation Guide](docs/implementing-models.md)

## 🙏 Acknowledgments

- Inspired by [Hugging Face Transformers](https://github.com/huggingface/transformers)
- Built on [SciRS2](https://github.com/scirs) for scientific computing
- Tokenizers from [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)
- Community contributions and feedback

## Sponsorship

TrustFormers is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

If you find TrustFormers useful, please consider sponsoring the project to support continued development of the Pure Rust ecosystem.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and improve the COOLJAPAN ecosystem
- Keep the entire ecosystem (OxiBLAS, OxiFFT, SciRS2, etc.) 100% Pure Rust
- Provide long-term support and security updates

## 📄 License

Licensed under Apache License, Version 2.0 ([LICENSE](LICENSE)).

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=trustformers/trustformers&type=Date)](https://star-history.com/#trustformers/trustformers&Date)

---

<p align="center">
  Built with 🦀 and ❤️ by COOLJAPAN OU (Team KitaSan)
</p>