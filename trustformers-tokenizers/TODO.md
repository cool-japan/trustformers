# trustformers-tokenizers TODO List

## Overview

The `trustformers-tokenizers` crate provides text tokenization for the TrustformeRS ecosystem.
It implements 24 concrete tokenizer types (plus a standalone multimodal tokenizer) covering
general-purpose subword algorithms, language-specific tokenizers, and domain-specific tokenizers,
along with vocabulary training, batch processing, vocabulary intelligence tooling, and optional
Python bindings.

**Key Responsibilities:**
- Tokenizer implementations (BPE, WordPiece, SentencePiece/Unigram, TikToken, Fairseq, Character-level, CANINE, Regex, Custom)
- Encoding (text → token IDs) and decoding (token IDs → text)
- Vocabulary management and training (`training::{BPETrainer, WordPieceTrainer, UnigramTrainer}`)
- Special token handling ([CLS], [SEP], [PAD], [MASK], [UNK])
- Batch processing with padding via `ParallelTokenizer`/`BatchTokenizer`
- Python bindings (PyO3, `python` feature)
- Migration reference docs from other tokenizer libraries (`docs/migration/`)

---

## Current Status

**Version:** 0.2.1 (unreleased) | **Date:** 2026-08-24 (SLoC and the migration-docs correction above refreshed; remainder largely last reviewed 2026-07-09) | **Status:** Stable

### Implementation Status
✅ **STABLE** — 24 tokenizer types implemented and tested, 0 genuine stub/placeholder implementations
✅ **COMPREHENSIVE TEST COVERAGE** — ~500 tests in this crate, 100% pass rate
⚠️ **PYTHON BINDINGS** — PyO3 source exists behind the `python` feature (`src/python.rs`) and a Python package exists (`python/trustformers_tokenizers/`), but `Cargo.toml` no longer builds a `cdylib` here (moved to `trustformers-py`), while this crate's own `pyproject.toml` still targets a `maturin` extension build — see Python Bindings below
✅ **ZERO COMPILATION ERRORS** — clean compilation; workspace-wide 0 clippy warnings, 0 rustdoc warnings (2026-07-01)
✅ **HUGGINGFACE `.json` COMPATIBLE** — `TokenizerImpl` wraps the real upstream `tokenizers` crate (re-exported via `trustformers-core`), not a reimplementation
✅ **24 TOKENIZER TYPES + MULTIMODAL** — general-purpose, language-specific (Arabic/Chinese/Japanese/Korean/Thai), domain-specific (Chemical/Music/Math/Code/BIO/Multimodal)
✅ **ZERO-COPY VOCAB** — memory-mapped vocabulary access (`ZeroCopyTokenizer`, `MmapVocab`, `memmap2`)
✅ **SIMD ACCELERATION** — AVX2 (x86_64) and NEON (aarch64) intrinsics for character classification, with a scalar fallback elsewhere
✅ **ASYNC TOKENIZATION** — non-blocking encode/decode via `tokio` (`AsyncTokenizer`); CPU-parallel batches separately via `scirs2-core`
✅ **VOCABULARY INTELLIGENCE** — `VocabIntelligenceAnalyzer` (semantic/compression/cross-lingual/domain/evolution analysis + scoring)
✅ **HANGUL DEPENDENCY REMOVED** (verified stale 2026-08-18) — `Cargo.toml` no longer declares `hangul`; Korean Hangul decomposition uses inline Unicode code-point arithmetic (`korean.rs`). This line previously described it as a still-present "candidate for removal"; the removal already landed (see CHANGELOG 0.2.0).

### Test Metrics
- **Test Count:** `cargo nextest run -p trustformers-tokenizers` (2026-08-18, default features) — 620 tests, 619 passed, **1 FAILED**, 0 skipped. Workspace-wide baseline reported to this documentation pass: 20,629 passed / 43 skipped / 0 failed (`cargo nextest run --workspace`, default features) — that baseline predates the one failure below, which was found while writing this file; see root `TODO.md` P0 #1.
- **Pass Rate:** 99.8% in this crate right now, not 100% — `workspace_hygiene::workspace_dependency_table_has_no_unused_entries` fails: `trustformers-serve` dropped 7 cloud-SDK dependencies with no remaining consumer, and root `Cargo.toml`'s `[workspace.dependencies]` still declares all 7. This crate's test is correctly catching a real, current problem in a manifest this crate does not own; see root `TODO.md` for the fix (delete the 7 lines from root `Cargo.toml`).
- **Public API Surface:** ~1,341 `pub fn`/`struct`/`enum`/`trait` items (+2 for `NFKCNormalizer`/`NFKDNormalizer`, added 2026-07-09)
- **SLoC:** 45,324 (`tokei`, whole crate, verified 2026-08-24; the 2026-07-01 figure was `src/` only, 68 files, not directly comparable)
- **Coverage:** Encoding/decoding, special tokens, edge cases, language-specific, domain-specific

---

## Completed Tokenizer Implementations

### TikToken

**Used by:** GPT-4, ChatGPT, Codex, GPT-2/GPT-3 (byte-level BPE)

- ✅ **Built-in presets:** `TiktokenTokenizer::cl100k_base()` (GPT-3.5/GPT-4), `TiktokenTokenizer::r50k_base()` (GPT-2)
- ✅ **Custom encodings:** `TiktokenTokenizer::from_tiktoken_file(...)` loads any `.tiktoken` merge-rank file — this is how `p50k_base`/`o200k_base`/other encodings can be used; they are **not** shipped as named convenience constructors today
- ✅ **`encode_with_special_tokens`**, `special_tokens()`, `is_special_token(id)`

---

### Fairseq

**Used by:** fairseq NMT/NLG models

- ✅ **Dictionary loading:** `FairseqTokenizer::from_file` parses fairseq's plain-text `dict.txt` format (`token frequency` per line)
- ✅ **Special tokens:** fairseq's fixed IDs — `<pad>`=0, `</s>`=1, `<unk>`=2, `<s>`=3
- Note: this is a dictionary/vocabulary loader, not a full fairseq preprocessing pipeline — there is no Moses tokenizer or subword-nmt BPE-merge step bundled; upstream text must already be pre-segmented the way the dictionary expects

---

### Language-Specific Tokenizers

- ✅ **Arabic** (`ArabicTokenizer`) — diacritic (tashkeel) removal, Arabic letter-form normalization, RTL-aware word segmentation, root/pattern morphological analysis (`analyze_morphology`); no external Farasa dependency
- ✅ **Chinese** (`ChineseTokenizer`) — in-crate pure-Rust dictionary + character-frequency segmentation; the `jieba-rs` dependency was removed as unused dead weight (COOLJAPAN Pure-Rust hygiene pass)
- ✅ **Japanese** (`JapaneseTokenizer`) — word/morpheme/character modes, katakana↔hiragana normalization, hiragana/katakana/kanji classification; morpheme mode uses real MeCab under the `mecab` feature, otherwise falls back to word-mode segmentation (no SudachiPy integration)
- ✅ **Korean** (`KoreanTokenizer`) — syllable/jamo/word modes, Hangul syllable↔jamo decomposition via direct Unicode arithmetic (`0xAC00` base), Hanja detection (no Mecab/Komoran morphological analyzer integration)
- ✅ **Thai** (`ThaiTokenizer`) — word/syllable/character modes, Thai-numeral normalization, tone-mark handling

---

### Domain-Specific Tokenizers

- ✅ **Chemical** (`ChemicalTokenizer`) — SMILES notation, molecular formulae, IUPAC name tokens
- ✅ **Music** (`MusicTokenizer`) — ABC notation, MusicXML, chord/tempo symbols
- ✅ **Math** (`MathTokenizer`) — LaTeX, MathML, expression-tree tokenization
- ✅ **Code** (`CodeTokenizer`) — language-aware tokenization via the `Language` enum (Rust, Python, JavaScript, TypeScript, Java, C#, C++, C, Go, Ruby, PHP, Swift, Kotlin, Scala, Haskell, ...)
- ✅ **BIO** (`BioTokenizer`) — FASTA/FASTQ, amino acids, gene ontology terms
- ✅ **Multimodal** (`MultimodalTokenizer`) — image patches, audio frames, video/table/graph token interleaving (standalone API — does not implement the shared `Tokenizer` trait like the other types above)

---

### BPE (Byte-Pair Encoding)

**Used by:** GPT-2, GPT-Neo, GPT-J, RoBERTa, BART, LLaMA (variants)

- ✅ Byte-level encoding, merge-table lookup (`HashMap`), regex-free byte fallback for unknown characters
- ✅ `from_files(vocab_path, merges_path)` and `from_roberta_files(vocab_path, merges_path)` loaders
- ✅ `tokenize_with_offsets` for character-offset tracking
- ✅ Training via `training::BPETrainer`

**Example:**
```rust
use trustformers_core::traits::Tokenizer;
use trustformers_tokenizers::BPETokenizer;

let tokenizer = BPETokenizer::from_files("vocab.json", "merges.txt")?;
let encoding = tokenizer.encode("Hello, world!")?;
let text = tokenizer.decode(&encoding.input_ids)?;
```
*(there is no `BPETokenizer::from_pretrained` — load vocab/merges files directly, or use `TokenizerImpl::from_file`/`from_pretrained` for a full `tokenizer.json`)*

---

### WordPiece

**Used by:** BERT, DistilBERT, ELECTRA, ALBERT

- ✅ Greedy longest-match-first tokenization, `##` continuation prefix, `[UNK]` fallback
- ✅ `encode()` automatically wraps input as `[CLS] ... [SEP]` with `token_type_ids = [0, 0, ...]`
- ✅ `encode_pair()` produces `[CLS] A [SEP] B [SEP]` with `token_type_ids = [0,...,0,1,...,1]`
- ✅ `from_pretrained(model_name)` checks local vocab-file path conventions first (`{model}/vocab.txt`, `{model}-vocab.txt`, ...), then falls back to small built-in vocabularies for `bert-base-uncased`/`bert-base-cased`/`distilbert-base-uncased` — it does not download from the Hugging Face Hub
- ✅ `from_vocab_file(path, do_lower_case)` for a real local `vocab.txt`

**Example:**
```rust
use trustformers_core::traits::Tokenizer;
use trustformers_tokenizers::WordPieceTokenizer;

let tokenizer = WordPieceTokenizer::from_vocab_file("vocab.txt", true)?;
let encoding = tokenizer.encode_pair("First sentence.", "Second sentence.")?;
// encoding.input_ids:      [CLS] First sentence . [SEP] Second sentence . [SEP]
// encoding.token_type_ids: Some([0, 0, 0, 0, 0, 1, 1, 1, 1])
```

---

### SentencePiece (Unigram)

**Used by:** T5, ALBERT, mBART, mT5, XLM-RoBERTa

- ✅ Unigram language-model segmentation via Viterbi decoding
- ✅ `from_model_file(path)` loads a real SentencePiece `.model` file
- ✅ `from_pretrained(model_name_or_path)` probes `{path}/spiece.model`, `{path}.model`, and the bare path for a real model file and loads it via `from_model_file` on a hit; only falls back to a simplified built-in vocabulary when none of those candidates resolve

**Example:**
```rust
use trustformers_core::traits::Tokenizer;
use trustformers_tokenizers::SentencePieceTokenizer;

let tokenizer = SentencePieceTokenizer::from_model_file("t5.model")?;
let encoding = tokenizer.encode("▁Hello,▁world!")?;
```

---

### Character-Level Tokenizer

**Used by:** Character-aware models, baseline experiments

- ✅ Simple character-by-character tokenization, direct char→ID mapping
- ✅ `CharTokenizer` — small vocabulary, O(n) encoding

### CANINE, Regex, Custom, and Zero-Copy Tokenizers

- ✅ **CANINE** (`CanineTokenizer`) — vocabulary-free character-hash tokenizer (CANINE architecture: hashing + downsampling config)
- ✅ **Regex** (`RegexTokenizer`) — configurable `Regex`/`RegexSet` splitting with custom patterns and priorities
- ✅ **Custom vocab** (`CustomVocabTokenizer` via `CustomVocabTokenizerBuilder`) — `.vocab_from_map(...)`/`.vocab_from_file(...)`, `.unk_token(...)`, `.special_token(...)`, `.max_length(...)`, `.build()`
- ✅ **Custom format** (`CustomFormatTokenizer`) — JSON-based tokenizer definitions with configurable normalization/pre-tokenization/post-processing rules
- ✅ **Zero-copy** (`ZeroCopyTokenizer`, `MmapVocab`) — `memmap2`-backed vocabularies for large-scale, low-memory deployments

### HuggingFace-Format Wrapper

- ✅ **`TokenizerImpl`** — thin wrapper around the real upstream `tokenizers` crate (`trustformers_core::tokenizer_backend`); `from_file(path)` loads any real `tokenizer.json`; `from_pretrained(name)` only resolves a local HF cache directory (`$HF_HOME`/`$TRANSFORMERS_CACHE`/`~/.cache/huggingface/transformers`) — no Hub download
- ✅ **`TokenizerWrapper`** — enum dispatching across `WordPiece`/`BPE`/`Unigram`/`Char`/`HuggingFace` variants behind the shared `Tokenizer` trait

---

## Core Functionality

### Encoding (Text → Token IDs)

- ✅ **Single Text Encoding**
  ```rust
  let encoding = tokenizer.encode("Hello, world!")?;
  // encoding.input_ids: Vec<u32>
  // encoding.attention_mask: Vec<u8>
  // encoding.token_type_ids: Option<Vec<u32>>
  ```
  (the `Tokenizer` trait's `encode` takes only the text — there is no separate `add_special_tokens` boolean parameter; special-token insertion, if any, is baked into each tokenizer's own `encode` implementation)

- ✅ **Text Pair Encoding**
  ```rust
  let encoding = tokenizer.encode_pair(text_a, text_b)?;
  ```

- ✅ **Batch Encoding**
  ```rust
  use trustformers_tokenizers::ParallelTokenizer;
  let parallel = ParallelTokenizer::new(tokenizer);
  let encodings = parallel.encode_batch(&texts)?;          // Vec<TokenizedInput>
  let padded = parallel.encode_batch_padded(&texts)?;      // BatchedTokenizedInput
  ```

### Decoding (Token IDs → Text)

- ✅ **Single Sequence Decoding**
  ```rust
  let text = tokenizer.decode(&token_ids)?;
  ```
  (single-argument — there is no `skip_special_tokens` flag on the shared trait; behavior is implementation-defined per tokenizer)

- ✅ **Batch Decoding**
  ```rust
  let texts = parallel.decode_batch(&ids_batch)?;
  ```

### Padding and Truncation

- ✅ **`BatchedTokenizedInput`** (`input_ids: Vec<Vec<u32>>`, `attention_mask: Vec<Vec<u8>>`, `token_type_ids: Option<Vec<Vec<u32>>>`) — produced by `ParallelTokenizer::encode_batch_padded`
- ✅ **Framework-specific strategies** — `PaddingStrategy`/`TruncationStrategy` enums exist under the `pytorch` feature (`PyTorchPaddingStrategy`/`TruncationStrategy`), with parallel `GpuPaddingStrategy` (`gpu`), `TfPaddingStrategy`/`TfTruncationStrategy` (`tensorflow`), and `JaxPaddingStrategy`/`JaxTruncationStrategy` (`jax`) — there is **no** default-feature, framework-agnostic `.pad()`/`.truncate()` builder chain on `TokenizedInput` itself

### Special Token Handling

- ✅ **Standard Special Tokens** — `[CLS]`/`<s>`, `[SEP]`/`</s>`, `[PAD]`/`<pad>`, `[MASK]`/`<mask>`, `[UNK]`/`<unk>`. Language-specific tokenizers (Japanese/Arabic/Chinese/Korean/Thai/Code/...) expose a dedicated `XxxTokenizerConfig` struct for this; `WordPieceTokenizer`/`BPETokenizer` currently hardcode their special-token strings inside `new()` (not yet exposed as a configurable struct)
- ✅ **`SpecialTokenManager`** (`special_tokens.rs`) — advanced template rendering with typed placeholders (`PlaceholderToken`/`PlaceholderType`), validation, and transformation hooks
- Note: the shared `Tokenizer` trait itself has no generic `add_special_tokens`/`add_tokens` methods — extending a vocabulary with new tokens is per-implementation (e.g. `ZeroCopyBuilder::add_tokens_from_map`, or constructing a new vocab/tokenizer)

### Attention Mask Generation

- ✅ **Automatic Generation** — every built-in `encode`/`encode_pair` implementation fills `attention_mask` with `1`s for the tokens it produces (there is no padding-aware mask generation until a batch is padded via `ParallelTokenizer`/`BatchedTokenizedInput`)

### Token Type IDs (Segment Embeddings)

- ✅ **Sentence Pair Support** — `0` for first sequence, `1` for second sequence (verified real in `WordPieceTokenizer::encode_pair`)
- ✅ **Automatic Generation** — generated during `encode`/`encode_pair` for WordPiece; other tokenizers may leave `token_type_ids: None` (e.g. `BPETokenizer::encode_pair` just concatenates the two texts with a space and re-encodes as a single sequence — check the specific tokenizer's implementation before relying on segment IDs)

---

## Vocabulary Management

### Vocabulary Construction

- ✅ **From Training** — `training::{BPETrainer, WordPieceTrainer, UnigramTrainer}` build a vocabulary from a `&[String]` corpus
- ✅ **From Files** — `BPETokenizer::from_files`, `WordPieceTokenizer::from_vocab_file`, `SentencePieceTokenizer::from_model_file`, `FairseqTokenizer::from_file`

### Vocabulary Operations (via the `Vocab` type and the `Tokenizer` trait)

- ✅ **Lookup** — `token_to_id(token)` / `id_to_token(id)` on any `Tokenizer` impl
- ✅ **Statistics** — `vocab_size()`, `get_vocab()` on any `Tokenizer` impl
- ✅ **Flexible/Lazy vocabularies** — `FlexibleVocab`, `LazyVocab` (`vocab/vocab_extended.rs`) for deferred/streaming vocabulary loading

---

## Tokenizer Training

### Training API

```rust
use trustformers_tokenizers::training::{BPETrainer, TrainingConfig};

let config = TrainingConfig {
    vocab_size: 30_000,
    min_frequency: 2,
    special_tokens: vec![
        "[PAD]".into(), "[UNK]".into(), "[CLS]".into(), "[SEP]".into(), "[MASK]".into(),
    ],
    ..Default::default()
};

let trainer = BPETrainer::new(config);
let texts: Vec<String> = /* corpus lines */ vec![];
let tokenizer = trainer.train(&texts)?; // -> BPETokenizer
```

`WordPieceTrainer` and `UnigramTrainer` mirror this `new(TrainingConfig) -> train(&[String]) -> Result<XxxTokenizer>` shape — there is no `TrainerBuilder`/fluent `.vocab_size(...)`/`.build()` API; configuration goes through the `TrainingConfig` struct (`vocab_size`, `min_frequency`, `special_tokens`, `end_of_word_suffix`, `max_input_chars_per_word`).

### Training Configuration (`TrainingConfig`)

- ✅ **Vocabulary Size** (`vocab_size`, default 30,000)
- ✅ **Minimum Frequency** (`min_frequency`, default 2)
- ✅ **Special Tokens** (`special_tokens`, default `[PAD]/[UNK]/[CLS]/[SEP]/[MASK]`)
- ✅ **End-of-word suffix** (`end_of_word_suffix`, default `"##"`) and **max input chars per word** (`max_input_chars_per_word`, default 100)

---

## Advanced Features

### Vocabulary Intelligence

- ✅ **Single entry point:** `VocabIntelligenceAnalyzer::new(VocabIntelligenceConfig).analyze(&tokenizer, basic_analysis)` returns a `VocabIntelligenceResult` bundling:
  - `semantic_analysis: Option<SemanticAnalysis>` — clustering, redundant-token groups
  - `compression_analysis: Option<CompressionAnalysis>` — tokens-per-word, compression opportunities
  - `cross_lingual_analysis: Option<CrossLingualAnalysis>` — per-language coverage
  - `domain_analysis: Option<DomainAnalysis>` — domain distribution/fit
  - `evolution_analysis: Option<EvolutionAnalysis>` — trending/declining tokens over training history
  - `intelligence_score: f32` (0–100) and `actionable_recommendations: Vec<ActionableRecommendation>`
  - each analysis is individually toggleable via `VocabIntelligenceConfig` flags

### Analysis Tools

- ✅ **`VocabAnalyzer`** (`vocab_analyzer.rs`) — `analyze_tokenizer`/`analyze_vocabulary`/`analyze_coverage`, character-pattern and subword-pattern detection, issue severity classification
- ✅ **`CoverageAnalyzer`** (`coverage.rs`) — unknown-token rate, vocabulary coverage, quality metrics, report export
- ✅ **`TokenizerBenchmark`** (`benchmark_utils.rs`) and **`PerformanceProfiler`** (`performance_profiler.rs`) — throughput/latency/memory measurement helpers
- ✅ **`TokenizationDebugger`** / **`TokenVisualizer`** — inspect and visualize tokenization output

### Performance Optimization

- ✅ **Parallel Tokenization** — `ParallelTokenizer`/`BatchTokenizer` parallelize batch encode/decode via `scirs2-core`'s `parallel` feature
- ✅ **SIMD Acceleration** — AVX2 (`std::arch::x86_64`) and NEON (`std::arch::aarch64`) intrinsics for character scanning, both `#[target_feature]`-gated with runtime detection; scalar fallback on other architectures
- ✅ **Zero-Copy Vocabulary Access** — `memmap2`-backed `MmapVocab`/`ZeroCopyTokenizer`
- ✅ **Async Tokenization** — `AsyncTokenizer` via `tokio` tasks/channels/timeouts (not `scirs2-core` — that crate powers the *parallel* CPU batch path instead)
- ✅ **Vocabulary Intelligence** — see above
- ✅ **Efficient Vocabulary Lookups** — `HashMap`-backed token↔ID lookup; `MinimalPerfectHash`/`MinimalPerfectHashVocab` for compile-time-known vocabularies
- ✅ **Compressed vocab** — `CompressedVocab`/`PrefixTrie` (`compressed_vocab.rs`)

---

## Python Bindings

### PyO3 Integration

- ✅ **Native Python Extension source** — `src/python.rs` under the `python`/`pyo3` features; wraps `BPETokenizer`, `CharTokenizer`, `UnigramTokenizer`, `WordPieceTokenizer`, `TokenizerImpl`
- ✅ **Python package** — `python/trustformers_tokenizers/` (`tokenizers.py`, `training.py`, `utils.py`, `__init__.py`) provides the Python-facing `AutoTokenizer` convenience API (Python-only — there is no Rust-level `AutoTokenizer` type; see `README_PYTHON.md` for the Python API surface)
- ⚠️ **`pyproject.toml` / `Cargo.toml` mismatch** — this crate's `pyproject.toml` is configured for a `maturin` extension-module build (`module-name = "trustformers_tokenizers"`), but `Cargo.toml`'s `[lib]` only declares `crate-type = ["rlib"]` — the `cdylib` target was intentionally removed (per its own comment: "Python extension building handled by trustformers-py crate"). Running `maturin build` from this crate directory will not currently produce a loadable native module; `python/trustformers_tokenizers/tokenizers.py` already anticipates this and falls back to `unittest.mock.MagicMock` stand-ins when `from .trustformers_tokenizers import (...)` raises `ImportError`
- ⚠️ **Python test suite** — no `python/tests/` directory exists yet; the previously documented `pytest tests/` command has nothing to run today

---

## Migration Guides

Reference docs live under `docs/migration/`:
- `docs/migration/huggingface-migration.md`
- `docs/migration/tiktoken-migration.md`
- `docs/migration/sentencepiece-migration.md`
- `docs/migration/spacy-migration.md`
- `docs/migration/nltk-migration.md`
- `docs/migration/fairseq-migration.md`

---

## Documentation

- ✅ `docs/api-reference.md`
- ✅ `docs/examples.md`
- ✅ `docs/custom-tokenizer-tutorial.md`
- ✅ `docs/ml-framework-integration.md`
- ⚠️ `docs/migration/` (6 guides, see above) — **corrected 2026-08-24**: this checkmark previously implied the whole set was done/accurate. It isn't: `tiktoken-migration.md` was found to describe ~30 methods/types that don't exist on the real `TiktokenTokenizer` (batching, caching config, chat templating, cost estimation, a whole Python-bindings section for a tokenizer with no Python binding, and more) and has been rewritten from the real source this pass. The other five guides plus `migration/README.md` carry smaller amounts of the same problem (a same-pattern grep found roughly 12/9/2/2/0/1 hits respectively) and now each carry a dated accuracy banner rather than a full rewrite — see the item below, which already flagged the performance-table half of this but not the API-fabrication half.
- ✅ Rustdoc for public APIs (0 rustdoc warnings workspace-wide, verified 2026-07-01)
- [~] Write tokenizer-selection, performance-tuning, and troubleshooting guides (planned 2026-07-05)
  - Goal: one combined deliverable (confirmed not 3 separate asks — TODO.md names all 3 in one line at two locations).
  - Design: 3 new files under docs/. Follow docs/migration/README.md's STRUCTURE (tables, checklists, troubleshooting section) but NOT its content practice — that file was found to contain fabricated benchmark numbers and references to APIs that don't exist anywhere in src/. Every code sample in the new docs must be grep-verified against a real `pub fn` signature before inclusion. Use README.md's honest style as the tone template instead.
  - Files: new docs/tokenizer-selection-guide.md, docs/performance-tuning-guide.md, docs/troubleshooting-guide.md.
  - Tests: grep every method name used in the new docs against `grep -rn "pub fn <name>" src/` before finalizing.
  - Risk: repeating docs/migration/README.md's fabrication pattern — explicitly guard against it. (2026-08-18: the invented-performance-table half of that pattern was found and fixed. 2026-08-24: a second, larger half of the same pattern was found — fabricated API methods/types, not just numbers — and `tiktoken-migration.md` was rewritten to fix it; the other five guides got dated accuracy banners instead of a full rewrite, see "Documentation" above. The 3 new guide files below still don't exist; this remains genuinely open.)

---

## Testing

### Test Coverage

- **620 tests** in this crate as of 2026-08-18 (`cargo nextest run -p trustformers-tokenizers`), 619 passing — **not** 100%: see "Test Metrics" above for the 1 currently-failing hygiene test, which is a real, current finding, not flaky.
- ✅ **Encoding/Decoding Correctness** — round-trip verification
- ✅ **Special Token Handling** — insertion/preservation checks
- ✅ **Edge Cases** — empty strings, very long texts, Unicode
- ✅ **Language-Specific Tests** — Arabic, Chinese, Japanese, Korean, Thai
- ✅ **Domain-Specific Tests** — Chemical, Music, Math, Code, BIO, Multimodal
- ✅ **Performance Benchmarks** — `benches/tokenizer_performance.rs` (criterion)
- ✅ **`test_infrastructure.rs`** — in-crate cross-validation/fuzzing/regression test harness (`TestRunner`, `FuzzingResults`, `CrossValidationRunner`)

---

## Known Limitations

- `TokenizerImpl::from_pretrained` and `WordPieceTokenizer::from_pretrained` only resolve local cache paths / a small built-in vocabulary set — neither downloads from the Hugging Face Hub
- `SentencePieceTokenizer::from_pretrained` probes `{path}/spiece.model`, `{path}.model`, and the bare path for a real model file before falling back to a simplified built-in vocabulary — see the SentencePiece section above
- TikToken ships only `cl100k_base`/`r50k_base` as named presets; other encodings need `from_tiktoken_file`
- SIMD acceleration covers AVX2 (x86_64) and NEON (aarch64); other architectures use the scalar fallback
- `jax`, `tensorflow`, `pytorch` features are pure-Rust detection/data-structure/metadata layers — not real JAX/TensorFlow/PyTorch execution (each adds zero extra crate dependencies)
- `gpu` feature: no real CUDA/ROCm/OpenCL/Vulkan kernel dispatch is compiled in (this crate is pure Rust with no unsafe FFI GPU driver bindings) — `GpuTokenizer::tokenize_batch` always executes via the real wrapped `Tokenizer`, sequentially or chunked across CPU cores in parallel (`scirs2_core::parallel_ops`) depending on `GpuTokenizerConfig::enable_gpu`; set `require_real_gpu: true` to get a hard `BackendUnavailable` error instead of the CPU fallback
- `onnx` feature: no real ONNX protobuf format or ONNX Runtime session — `OnnxTokenizerExporter` writes (and `OnnxTokenizerRuntime` reads back) this crate's own JSON interchange format, structured to mirror ONNX's graph/tensor model; `OnnxTokenizerRuntime::tokenize` performs real greedy longest-match tokenization against the real vocabulary recovered from that file (not hash-derived), and rejects a genuine binary `.onnx` protobuf file with a structured error rather than fabricating a vocabulary for it
- `AutoTokenizer` is Python-only; Rust callers use `TokenizerWrapper` (enum dispatch) or a concrete tokenizer type directly
- This crate's `pyproject.toml` still targets a `maturin` extension-module build, but `Cargo.toml` no longer declares a `cdylib` target (moved to `trustformers-py`) — `maturin build` here will not currently produce a working native module
- No `python/tests/` directory exists yet

---

## Future Enhancements

### High Priority
- [~] Implement Unigram sampling / BPE-dropout (planned 2026-07-05)
  - Goal: real stochastic tokenization — the existing subword_regularization.rs module sounds like it already does this but implements a different technique (character-level noise, not merge-dropping).
  - Design: inside bpe.rs's merge-selection loop, roll each candidate pair against a dropout_p probability and exclude dropped pairs from that iteration's min-rank search (Provilkov et al. 2020's actual BPE-dropout). Inside unigram.rs's existing Viterbi DP table, add forward-filter/backward-sample instead of always taking the single best segmentation. Reuse the RNG pattern already established in subword_regularization.rs (scirs2_core::random::*).
  - Files: trustformers-tokenizers/src/bpe.rs, src/unigram.rs.
  - Tests: dropout_p=0.0 must reproduce today's exact deterministic output; seeded-RNG determinism; >=2 distinct segmentations across N samples at dropout_p=0.5; decode still round-trips.
  - Risk: BPETokenizer caches bpe() results in a RwLock<HashMap> — sampling must explicitly bypass this cache, or the "random" result freezes after the first call per unique input.
- [x] Real HF Hub download for from_pretrained — NARROWED, fix lives in trustformers crate (planned 2026-07-05) — **DONE (2026-07-05):** `AutoTokenizer::from_pretrained_with_revision` in `trustformers/src/automodel.rs` now calls `crate::hub::download_file_from_hub` for `tokenizer.json` (mirroring `AutoConfig::from_pretrained_with_revision`'s already-working pattern) before falling back to the local-cache-only lookup; added a cache-hit test proving no network call is needed when the file is already present.
  - Goal: narrowed to fixing AutoTokenizer::from_pretrained_with_revision in trustformers/src/automodel.rs (NOT this crate) to call the hub download code that AutoConfig, 130 lines above it in the same file, already calls.
  - Design: trustformers-tokenizers cannot depend on trustformers (would be a Cargo dependency cycle — trustformers already depends on trustformers-tokenizers). This fix lives entirely in the trustformers crate, at the call site next to its already-working sibling. Deferring "give trustformers-tokenizers its own standalone hub feature" as a separate, deeper follow-up.
  - Files: trustformers/src/automodel.rs (NOT a file in this crate).
  - Tests: cache-hit test asserting no network call when file already exists; existing offline tests must keep passing.
  - Risk: none new — copying an already-proven 130-line-away pattern in the same file.
- [x] Fix SentencePieceTokenizer::from_pretrained ignoring its argument (planned 2026-07-05) — **DONE (2026-07-09):** `from_pretrained` now probes `{path}/spiece.model`, `{path}.model`, and the bare path (delegating to `from_model_file` on a hit) before falling back to the fabricated built-in vocabulary; verified via `test_from_pretrained_loads_real_fixture_not_hardcoded_fallback`, which writes a real fixture `.model` file under `std::env::temp_dir()` and asserts the fixture's tokens load correctly while the fallback's T5 sentinel tokens do not appear.
  - Goal: the argument is no longer ignored (currently always returns the same hardcoded fake T5 vocab regardless of input).
  - Design: probe candidate paths built from the argument, delegate to the existing, already-correct from_model_file() on a hit; keep today's fabricated-vocab body only as the final fallback arm, now actually gated on the argument.
  - Files: trustformers-tokenizers/src/sentencepiece.rs only.
  - Tests: the existing tests cannot detect this bug (both call from_pretrained with the same argument) — add a new test with a distinct fixture file via std::env::temp_dir().
  - Risk: interacts with the Hub-download item above — out of scope to design against it now.
- [ ] Enhanced multilingual support (better handling of non-Latin scripts)
- [ ] Real binary ONNX protobuf export/import for tokenizers (export tokenizer to a file an actual ONNX runtime can load, for cross-framework compatibility)
  - **Note:** Use the `oxionnx` crate per COOLJAPAN policy. `OnnxTokenizerExporter`/`OnnxTokenizerRuntime` (src/onnx.rs) currently read and write this crate's own JSON interchange format instead of real ONNX protobuf bytes -- `OnnxTokenizerRuntime::tokenize` does perform real greedy longest-match tokenization against the real vocabulary recovered from that JSON file (not a stub), it just is not talking to an actual ONNX Runtime session or a real `.onnx` file.

### Performance
- [x] Port 4 AVX2 SIMD functions to ARM/NEON (planned 2026-07-05) — **DONE (2026-08-17):** `classify_ascii_chars`, `find_whitespace_boundaries`, `validate_utf8_fast`, `to_lowercase_ascii` all got real `#[cfg(target_arch = "aarch64")]` NEON siblings in `src/simd.rs`, dispatched 3-way (x86_64 AVX2 / aarch64 NEON / scalar) with `std::arch::is_aarch64_feature_detected!("neon")` mirroring the existing `is_x86_feature_detected!("avx2")` check. `find_whitespace_boundaries_neon` avoids needing a `_mm256_movemask_epi8`-style emulation entirely by storing the NEON comparison result to a 16-byte array and scanning it directly for transitions (the comparison itself, the expensive part, is still real NEON). Also fixed in the same pass, both found by exhaustive `*_parity_with_scalar_for_every_byte` tests (all 256 byte values) added for classify/whitespace/lowercase: (1) `classify_ascii_chars_avx2` was dead code dressed as SIMD — it loaded a chunk into an AVX2 register via `_mm256_loadu_si256` and then never read it, doing a plain scalar per-byte lookup-table loop instead; replaced with real vectorized ASCII range comparisons (falling back to the scalar table only for chunks containing a non-ASCII byte). (2) `find_whitespace_boundaries_avx2` only checked space/tab/LF/CR, silently disagreeing with the scalar reference (built from real `char::is_whitespace()`) for VT (0x0B), FF (0x0C), NEL (0x85), and NBSP (0xA0) — both bugs affected the pre-existing AVX2 path too, not just the new NEON one. Verified for real on this Apple Silicon dev machine (native NEON execution, all parity tests passing) and cross-compile-checked for `x86_64-apple-darwin`; the corrected AVX2 intrinic patterns were additionally spot-verified by extracting the exact range-comparison sequence into a standalone program and running it under Rosetta 2 with `-C target-feature=+avx2` forced (`is_x86_feature_detected!` reports `false` under Rosetta, so the crate's own runtime-dispatched tests exercise the scalar fallback there, not AVX2).
- [ ] Real GPU kernel dispatch for the `gpu` feature (currently: best-effort driver-presence detection only, informational; `GpuTokenizer::tokenize_batch` always executes via the real wrapped `Tokenizer`, parallelized across CPU cores or sequential per `enable_gpu` — see `src/gpu_tokenization.rs` module docs. An earlier version of this module simulated GPU kernel dispatch entirely — fake memory pointers, fake kernel function pointers, fake per-vendor "compute capability" numbers — and its batch tokenization path returned the hardcoded sequence `[1, 2, .., 10]` for every input regardless of content; that whole fake pipeline has been removed.)
  - **Refinement needed:** which ops to GPU-accelerate (vocab lookup? regex? both)? Target throughput (tokens/sec)? Given the pure-Rust-without-unsafe-FFI constraint, a real implementation likely means a `wgpu` compute-shader backend (feature-gated) rather than direct CUDA/ROCm driver bindings.
- [~] Implement real incremental/streaming tokenization (planned 2026-07-05)
  - Goal: tokenize arbitrary raw byte chunks incrementally — the existing streaming.rs doesn't solve this (buffers whole lines/whole text, not arbitrary byte boundaries).
  - Design: add IncrementalTokenizer<T: Tokenizer> with a pending: Vec<u8> field and push_bytes() using std::str::from_utf8's error_len()/valid_up_to() to distinguish "genuinely invalid" from "incomplete multi-byte tail"; plus finish() for real stream end.
  - Files: trustformers-tokenizers/src/streaming.rs.
  - Tests: feed a multi-byte-character string one byte at a time through push_bytes, assert final tokenization equals whole-string tokenization.
  - Risk: MUST document explicitly that tokenization is not generally composable across arbitrary split points for BPE/WordPiece (only UTF-8 byte-safety is guaranteed) — do not let this silently imply full tokenization equivalence.
- [ ] Further optimization of vocabulary lookups

### Features
- [ ] Custom normalizers / pre-tokenizers plugin API for user-supplied normalizers
- [x] Add NFKC/NFKD normalizers (planned 2026-07-05) — **DONE (2026-07-09):** `NFKCNormalizer`/`NFKDNormalizer` added to `src/normalizer.rs` via `.nfkc()`/`.nfkd()` on the already-imported `unicode_normalization::UnicodeNormalization` trait; tests cover U+00B2 (SUPERSCRIPT TWO) and U+FB01 (LATIN SMALL LIGATURE FI), both of which fold under NFKC/NFKD but are left untouched by NFC/NFD, proving the K-variants do genuinely different work rather than aliasing the plain variants.
  - Goal/Design: add NFKCNormalizer/NFKDNormalizer via .nfkc()/.nfkd() — sibling methods on the exact trait already imported for NFC/NFD. Zero new dependency; sentencepiece.rs already calls .nfkc() internally, proving it works here.
  - Files: trustformers-tokenizers/src/normalizer.rs only.
  - Tests: assert on a real compatibility-decomposition case that changes under NFKC/NFKD but is a no-op under plain NFC/NFD.
  - Risk: none — lowest-risk item in the whole batch.
- [~] Build tokenizer alignment visualizer (planned 2026-07-05)
  - Goal: visualize token<->source-text alignment using real byte offsets (existing TokenVisualizer hardcodes None because it's generic against a trait with no offset method, even though real offsets already exist elsewhere).
  - Design: build a new visualizer consuming the concrete offset-bearing types directly (BPETokenizer::tokenize_with_offsets, TokenizerImpl::encode_with_offsets) rather than widening the generic Tokenizer trait (too high blast radius). Copy (do not depend on) trustformers-debug's ~40-line HTML/JSON/ASCII three-format pattern.
  - Files: trustformers-tokenizers/src/visualization.rs, src/alignment.rs.
  - Tests: invariant test that computed byte ranges exactly tile 0..text.len() with no gaps/overlaps.
  - Risk: low — confirmed no dependency cycle risk; just don't add trustformers-debug as a dependency, copy the small pattern.
- [x] Delete unused hangul dependency — **done, verified 2026-08-18**: `rg hangul trustformers-tokenizers/Cargo.toml` finds nothing; `cargo tree -i hangul` would report not-found. This item was still open in this file despite the removal already having landed (CHANGELOG 0.2.0).
- [ ] Automatic tokenizer repair/optimization

### Housekeeping
- [ ] Add a `python/tests/` suite (referenced in older docs but never created)
- This guide-writing item was previously also listed here as a third copy; collapsed into the single entry above (search "tokenizer-selection" in this file) rather than tracked three times, per the 2026-08-18 documentation pass.

---

## Development Guidelines

### Code Standards
- **Use trustformers-core/scirs2-core abstractions only** (no external deps directly — enforced via `trustformers-core::tokenizer_backend` for the upstream `tokenizers` crate)
- **File size limit:** <2000 lines per file — currently satisfied; largest file is `src/advanced_vocab_intelligence/avi_analyzer.rs` at 1,539 lines (`wc -l`, verified 2026-08-18; `gpu_tokenization.rs`, previously cited as largest at 1,616 lines, is now 865 lines)
- **Error handling:** Use `Result<T, TrustformersError>`
- **Testing:** Comprehensive test coverage required
- **Naming:** snake_case for all identifiers

### Build & Test Commands

```bash
# Run all tests
cargo nextest run -p trustformers-tokenizers --all-features

# Benchmark
cargo bench -p trustformers-tokenizers

# Check compilation
cargo check -p trustformers-tokenizers --all-features

# Format and clippy
cargo fmt --all
cargo clippy -p trustformers-tokenizers --all-features -- -D warnings
```

---

**Last Updated:** 2026-07-09
**Version:** 0.2.1
**Status:** Stable
**Test Coverage:** 620 tests in this crate as of 2026-08-18, 619 passing (see "Test Metrics" above for the 1 current, real failure — a hygiene gate catching an orphaned root-manifest entry, not a bug in this crate)
**Public API:** ~1,341 items (not re-counted this pass)
**SLoC:** 45,325 (`tokei`, 2026-08-18)
