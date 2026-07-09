# trustformers-tokenizers

Tokenization library for transformer models, providing Byte-Pair Encoding (BPE), WordPiece, SentencePiece (Unigram), TikToken, and Fairseq-dictionary tokenizers, plus language-specific (Arabic, Chinese, Japanese, Korean, Thai) and domain-specific (Chemical, Music, Math, Code, BIO, Multimodal) tokenizers for the TrustformeRS ecosystem. Version 0.2.1 — Development.

**Version:** 0.2.1 | **Status:** Stable | **Tests:** ~500 | **SLoC:** 51,372 | **Last Updated:** 2026-07-09

## Current State

This crate provides **24 concrete tokenizer implementations** of the shared `trustformers_core::traits::Tokenizer` trait, a standalone multimodal tokenizer, and optional ML-framework batch adapters (GPU/JAX/TensorFlow/PyTorch/ONNX, all off by default). It wraps the real upstream Hugging Face `tokenizers` crate (re-exported through `trustformers-core`) for `.json`-format compatibility, and is otherwise Pure Rust by default — the C/C++ `onig`, `esaxx`, `jieba-rs`, and `zstd` dependencies that historically shipped transitively have all been removed from the default build.

## Features

### Implemented Tokenizers (24 core + multimodal)

#### General-Purpose
- **BPE** (`BPETokenizer`) — byte-level BPE (GPT-2 style), merge-table lookup, `from_files`/`from_roberta_files` loaders, offset tracking
- **WordPiece** (`WordPieceTokenizer`) — greedy longest-match-first, `##` continuation prefix; `from_pretrained` checks local vocab-file paths first, then falls back to a handful of built-in BERT/DistilBERT vocabularies
- **SentencePiece / Unigram** (`SentencePieceTokenizer`) — Viterbi-decoded unigram LM segmentation; `from_model_file` loads real `.model` files
- **Unigram** (`UnigramTokenizer`) — standalone Viterbi-decoded unigram scorer
- **TikToken** (`TiktokenTokenizer`) — `cl100k_base()` and `r50k_base()` ship as built-in presets; other encodings (`p50k_base`, `o200k_base`, custom) can be loaded via `from_tiktoken_file`
- **Fairseq** (`FairseqTokenizer`) — loads fairseq `dict.txt`-style token/frequency dictionaries with fairseq's special-token IDs (`<pad>`=0, `</s>`=1, `<unk>`=2, `<s>`=3)
- **Character-level** (`CharTokenizer`) — one token per character
- **CANINE** (`CanineTokenizer`) — vocabulary-free character-hash tokenizer
- **Regex-based** (`RegexTokenizer`) — configurable `Regex`/`RegexSet` pattern splitting
- **Custom vocabulary / format** (`CustomVocabTokenizer` + `CustomVocabTokenizerBuilder`, `CustomFormatTokenizer`) — user-supplied vocabularies and JSON tokenizer definitions
- **Zero-copy** (`ZeroCopyTokenizer`) — memory-mapped (`memmap2`) vocabulary tokenizer
- **HuggingFace-format** (`TokenizerImpl`, `TokenizerWrapper`) — `TokenizerImpl` is a thin wrapper around the real upstream `tokenizers` crate for `.json` tokenizer files; `TokenizerWrapper` is an enum that dispatches across the WordPiece/BPE/Unigram/Char/HuggingFace variants

#### Language-Specific
- **Arabic** (`ArabicTokenizer`) — diacritic (tashkeel) removal, letter-form normalization, RTL-aware word segmentation, root/pattern morphological analysis (`analyze_morphology`)
- **Chinese** (`ChineseTokenizer`) — in-crate pure-Rust dictionary + character-frequency segmentation (the `jieba-rs` dependency was removed as unused dead weight in a prior Pure-Rust hygiene pass)
- **Japanese** (`JapaneseTokenizer`) — word/morpheme/character modes, katakana↔hiragana normalization, hiragana/katakana/kanji classification; morpheme mode uses real MeCab when the `mecab` feature is enabled, otherwise falls back to word-mode segmentation
- **Korean** (`KoreanTokenizer`) — syllable/jamo/word modes, Hangul syllable↔jamo decomposition via direct Unicode code-point arithmetic, Hanja detection
- **Thai** (`ThaiTokenizer`) — word/syllable/character modes, Thai-numeral normalization, tone-mark handling

#### Domain-Specific
- **Chemical** (`ChemicalTokenizer`) — SMILES notation, molecular formulae, IUPAC name tokens
- **Music** (`MusicTokenizer`) — ABC notation, MusicXML, chord/tempo symbols
- **Math** (`MathTokenizer`) — LaTeX, MathML, expression-tree tokenization
- **Code** (`CodeTokenizer`) — language-aware tokenization for 15+ languages via the `Language` enum (Rust, Python, JavaScript, TypeScript, Java, C#, C++, C, Go, Ruby, PHP, Swift, Kotlin, Scala, Haskell, ...)
- **BIO** (`BioTokenizer`) — FASTA/FASTQ, amino acids, gene-ontology terms
- **Multimodal** (`MultimodalTokenizer`) — image patches, audio frames, video/table/graph token interleaving (standalone API; does not implement the shared `Tokenizer` trait)

### Core Features
- **Zero-copy vocabulary access** — memory-mapped vocabularies for large-scale use
- **SIMD acceleration** — hand-written AVX2 intrinsics (`std::arch::x86_64`) for character classification/scanning; x86_64-only today, scalar fallback elsewhere (including Apple Silicon)
- **Parallel batch processing** — CPU-parallel batch encode/decode via `scirs2-core`'s `parallel` feature
- **Async tokenization** — non-blocking encode/decode via `tokio` tasks, channels, and timeouts
- **Vocabulary intelligence** — `VocabIntelligenceAnalyzer::analyze()` combines semantic clustering, compression-efficiency, cross-lingual coverage, domain-fit, and vocabulary-evolution analysis into one scored report with actionable recommendations
- **Training infrastructure** — `training::{BPETrainer, WordPieceTrainer, UnigramTrainer}`, each trained from `&[String]` corpora
- **Batch processing** — `ParallelTokenizer`/`BatchTokenizer` for multi-text encode/decode with padding
- **Offset mapping** — character-position tracking for tokens
- **Special tokens** — `SpecialTokenManager` with placeholder/template support
- **Padding/Truncation** — sequence-length management utilities
- **Thread-safe** — the `Tokenizer` trait requires `Send + Sync`

### Pre/Post Processing
- **Normalization** — composable `Normalizer` trait with `NFCNormalizer`/`NFDNormalizer`/`NFKCNormalizer`/`NFKDNormalizer` (via the `unicode-normalization` crate), plus whitespace, accent-removal, punctuation-removal, digit, and case normalizers, combinable via `ChainedNormalizer`
- **Alignment** — `AlignmentEngine` for token ↔ word span alignment
- **Decoding** — token-to-text reconstruction, including byte-level BPE decoding
- **Debugging/visualization** — `TokenizationDebugger` and `TokenVisualizer` for inspecting tokenization output

### Feature Flags
- `default = []` — Pure Rust, no optional bindings enabled
- `python` — PyO3 Python bindings (pulls in `pyo3`)
- `pyo3` — low-level PyO3 extension-module support
- `mecab` — real MeCab morphological analysis for Japanese (adds the `mecab` FFI crate; not Pure Rust)
- `gpu` — GPU backend *abstraction*: CUDA/ROCm/oneAPI/OpenCL/Vulkan device detection and simulated device-property reporting; batch tokenization itself still executes on CPU today (no CUDA/ROCm/OpenCL kernels are dispatched)
- `jax` / `tensorflow` / `pytorch` — pure-Rust data structures shaped like each framework's arrays/batches/dtype/padding conventions, for downstream layers to consume; these do **not** link against libjax/libtensorflow/libtorch
- `onnx` — ONNX model/graph metadata export types (no ONNX Runtime execution)

`gpu`, `jax`, `tensorflow`, `pytorch`, and `onnx` add **zero** extra crate dependencies — they are logic/data-structure layers, not real accelerator or runtime bindings yet.

## Quick Start

```rust
use trustformers_core::traits::Tokenizer;
use trustformers_tokenizers::BPETokenizer;
use std::collections::HashMap;

let mut vocab = HashMap::new();
vocab.insert("hello".to_string(), 0);
vocab.insert("world".to_string(), 1);
let merges = vec![];

let tokenizer = BPETokenizer::new(vocab, merges);
let encoding = tokenizer.encode("hello world")?;
println!("IDs: {:?}", encoding.input_ids);

let text = tokenizer.decode(&encoding.input_ids)?;
println!("Decoded: {}", text);
```

### Loading HuggingFace-format tokenizers

```rust
use trustformers_tokenizers::tokenizer::TokenizerImpl;
use std::path::Path;

// Loads a tokenizer.json file directly (backed by the real upstream `tokenizers` crate)
let tokenizer = TokenizerImpl::from_file(Path::new("tokenizer.json"))?;

// `from_pretrained` only resolves a local HF cache directory
// ($HF_HOME / $TRANSFORMERS_CACHE / ~/.cache/huggingface/transformers) —
// it does not download anything from the Hugging Face Hub.
let tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;
```

### WordPiece from a vocab file

```rust
use trustformers_tokenizers::WordPieceTokenizer;

let tokenizer = WordPieceTokenizer::from_vocab_file("vocab.txt", true)?;
let encoding = tokenizer.encode("Hello, world!")?;
```

### Batch Tokenization

```rust
use trustformers_tokenizers::ParallelTokenizer;

let texts = vec!["First sentence.", "Second sentence is longer.", "Third one."];
let parallel = ParallelTokenizer::new(tokenizer);
let encodings = parallel.encode_batch(&texts)?;
let padded = parallel.encode_batch_padded(&texts)?;
```

### Language-Specific Tokenization

```rust
use trustformers_tokenizers::{JapaneseTokenizer, JapaneseTokenizerConfig};
use trustformers_tokenizers::vocab::Vocab;

// Word-mode segmentation by default; Morpheme mode additionally requires the `mecab` feature.
let tokenizer = JapaneseTokenizer::new(JapaneseTokenizerConfig::default(), Vocab::default())?;
let encoding = tokenizer.encode("こんにちは世界")?;
```

### Domain-Specific Tokenization

```rust
use trustformers_tokenizers::ChemicalTokenizer;
use trustformers_tokenizers::{CodeTokenizer, Language};

// Chemical SMILES tokenizer
let tokenizer = ChemicalTokenizer::new();
let encoding = tokenizer.encode("CC(=O)Oc1ccccc1C(=O)O")?; // Aspirin

// Code-aware tokenizer
let tokenizer = CodeTokenizer::for_language(Language::Rust);
let encoding = tokenizer.encode("fn main() { println!(\"Hello\"); }")?;
```

## Architecture

```
trustformers-tokenizers/
├── src/
│   ├── tokenizer.rs        # TokenizerImpl / TokenizerWrapper (HF-compatible)
│   ├── bpe.rs, wordpiece.rs, sentencepiece.rs, unigram.rs, tiktoken.rs, fairseq.rs
│   ├── char.rs, canine.rs, regex_tokenizer.rs, custom.rs, custom_format.rs, zero_copy.rs
│   ├── arabic.rs, chinese.rs, japanese.rs, korean.rs, thai.rs      # language-specific
│   ├── chemical.rs, music.rs, math_tokenizer.rs, code_tokenizer.rs,
│   │   bio.rs, multimodal.rs                                      # domain-specific
│   ├── normalizer.rs, alignment.rs                                # pre/post-processing
│   ├── special_tokens.rs, sequence_packing.rs                     # special tokens & packing
│   ├── vocab/                # Vocab, FlexibleVocab, LazyVocab
│   ├── training/             # BPETrainer, WordPieceTrainer, UnigramTrainer, corpus, distributed
│   ├── advanced_vocab_intelligence/  # VocabIntelligenceAnalyzer
│   ├── vocab_analyzer.rs, coverage.rs, benchmark_utils.rs, performance_profiler.rs
│   ├── simd.rs, parallel.rs, async_tokenizer.rs, streaming.rs, shared_vocab_pool.rs
│   ├── mmap_vocab.rs, compressed_vocab.rs, minimal_perfect_hash.rs, binary_format.rs,
│   │   messagepack_serialization.rs, protobuf_serialization.rs
│   ├── tokenization_debugger.rs, visualization.rs, test_infrastructure.rs
│   ├── gpu_tokenization.rs    # `gpu` feature
│   ├── jax.rs                 # `jax` feature
│   ├── tensorflow.rs          # `tensorflow` feature
│   ├── pytorch.rs             # `pytorch` feature
│   ├── onnx.rs                 # `onnx` feature
│   └── python.rs               # `python`/`pyo3` feature (PyO3 bindings)
├── python/                  # Python package (AutoTokenizer, TokenizerTrainer, ...)
├── docs/                    # api-reference, examples, tutorial, migration guides
└── benches/                 # criterion benchmarks
```

The module layout is intentionally flat (no nested `models/`, `languages/`, or `domains/` sub-directories) — every tokenizer lives directly under `src/` and is re-exported at the crate root by `lib.rs`.

## Performance

Real, verified performance characteristics:
- **SIMD**: AVX2 intrinsics (`#[target_feature(enable = "avx2")]`) accelerate character classification on x86_64; there is currently no NEON/ARM path, so this is inactive on Apple Silicon
- **Zero-copy**: `ZeroCopyTokenizer`/`MmapVocab` memory-map vocabulary files instead of heap-loading them, keeping resident memory low for large vocabularies
- **Parallel batches**: `ParallelTokenizer`/`BatchTokenizer` parallelize `encode_batch`/`decode_batch` across CPU cores via `scirs2-core`
- **Async**: `AsyncTokenizer` offloads encode/decode onto `tokio` tasks with cooperative timeouts, for non-blocking use in async services

Run `cargo bench -p trustformers-tokenizers` (see `benches/tokenizer_performance.rs`) to measure throughput on your own hardware — no fixed benchmark numbers are published here since they vary significantly by CPU and vocabulary size.

## Training Tokenizers

```rust
use trustformers_tokenizers::training::{BPETrainer, TrainingConfig};

let config = TrainingConfig {
    vocab_size: 30_000,
    min_frequency: 2,
    special_tokens: vec![
        "[PAD]".to_string(), "[UNK]".to_string(), "[CLS]".to_string(),
        "[SEP]".to_string(), "[MASK]".to_string(),
    ],
    ..Default::default()
};

let trainer = BPETrainer::new(config);
let texts: Vec<String> = std::fs::read_to_string("data/corpus.txt")?
    .lines()
    .map(String::from)
    .collect();
let tokenizer = trainer.train(&texts)?; // -> BPETokenizer
```

`WordPieceTrainer` and `UnigramTrainer` follow the same `new(TrainingConfig) -> train(&[String]) -> Result<XxxTokenizer>` shape.

## Vocabulary Intelligence

```rust
use trustformers_tokenizers::{VocabIntelligenceAnalyzer, VocabIntelligenceConfig};
use trustformers_tokenizers::vocab_analyzer::VocabAnalyzer;

let basic_analysis = VocabAnalyzer::new(Default::default()).analyze_tokenizer(&tokenizer)?;

let mut analyzer = VocabIntelligenceAnalyzer::new(VocabIntelligenceConfig::default());
let result = analyzer.analyze(&tokenizer, basic_analysis)?;

println!("Intelligence score: {:.1}/100", result.intelligence_score);
for rec in &result.actionable_recommendations {
    println!("- {:?}", rec);
}
```

## Compatibility

### Supported Formats
- **Hugging Face**: `TokenizerImpl` embeds the real upstream `tokenizers` crate (re-exported via `trustformers-core`) — `.json` tokenizer files load with full fidelity
- **SentencePiece**: `SentencePieceTokenizer::from_model_file` loads real `.model` files directly; `from_pretrained(model_name_or_path)` now also probes `{path}/spiece.model`, `{path}.model`, and the bare path for a real model file (delegating to `from_model_file` on a hit) before falling back to a simplified built-in vocabulary — see Known Limitations
- **TikToken**: `cl100k_base`/`r50k_base` presets built in; other `.tiktoken` files loadable via `from_tiktoken_file`
- **Fairseq**: `dict.txt` token/frequency dictionary format
- **Custom**: JSON-based tokenizer configuration (`CustomFormatTokenizer`)

### Integration
- Direct use in TrustformeRS models
- Python bindings source via the `python` feature (PyO3) — see `src/python.rs`, `python/`, and `README_PYTHON.md`. Note: actual pip-installable extension building now happens in the `trustformers-py` crate (see Known Limitations)
- WASM support via `trustformers-wasm`

## Known Limitations

- `TokenizerImpl::from_pretrained` and `WordPieceTokenizer::from_pretrained` resolve a **local** cache/vocab path (or a small set of built-in vocabularies); neither downloads from the Hugging Face Hub
- `SentencePieceTokenizer::from_pretrained` probes `{path}/spiece.model`, `{path}.model`, and the bare path for a real SentencePiece model file and loads it via `from_model_file` on a hit; it falls back to a simplified built-in vocabulary only when none of those candidates resolve
- TikToken ships two named presets (`cl100k_base`, `r50k_base`); `p50k_base`/`o200k_base` are not built-in convenience constructors (load them manually via `from_tiktoken_file`)
- SIMD acceleration is AVX2/x86_64-only; no ARM/NEON path yet
- The `gpu`, `jax`, `tensorflow`, `pytorch`, and `onnx` features provide detection/data-structure/metadata layers, not real CUDA/ROCm/OpenCL/JAX/TensorFlow/PyTorch/ONNX-Runtime execution
- `AutoTokenizer` exists only in the Python package (`python/trustformers_tokenizers`) — there is no Rust-level `AutoTokenizer` type; use `TokenizerWrapper` for enum-based dispatch across the built-in Rust tokenizer families
- This crate's own `pyproject.toml` is configured for a `maturin` extension-module build, but `Cargo.toml`'s `[lib]` only declares `crate-type = ["rlib"]` (no `cdylib`) — that responsibility was moved to `trustformers-py`. Building the Python package straight from this crate directory will not currently produce a loadable native module; `python/trustformers_tokenizers/tokenizers.py` falls back to `unittest.mock.MagicMock` when the native import fails

## Testing

- **~500 unit tests** for this crate (workspace-wide: 18,102 passed / 0 failed / 119 skipped, 0 clippy warnings, 0 rustdoc warnings — verified 2026-07-01)
- Encoding/decoding round-trip correctness, special-token handling, edge cases (empty input, long sequences, Unicode)
- Language-specific and domain-specific correctness tests
- Criterion benchmarks in `benches/tokenizer_performance.rs`

```bash
cargo nextest run -p trustformers-tokenizers --all-features
cargo bench -p trustformers-tokenizers
```

## License

Apache-2.0
