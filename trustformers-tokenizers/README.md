# trustformers-tokenizers

High-performance tokenization library for transformer models with support for 50+ tokenization algorithms. Version 0.1.0 — Stable.

**Version:** 0.1.0 | **Status:** Stable | **Tests:** 500 | **SLoC:** 51,211 | **Last Updated:** 2026-03-21

## Current State

This crate provides **production-ready tokenizer implementations** covering BPE (Byte-Pair Encoding), WordPiece, SentencePiece, TikToken, Fairseq, language-specific tokenizers (Arabic, Chinese, Japanese, Korean), domain-specific tokenizers (Chemical, Music, Math, Code, BIO, Multimodal), and many more. It is designed to be fast, memory-efficient, and compatible with popular tokenizer formats.

## Features

### Implemented Tokenizers (50+)

#### General-Purpose
- **BPE (Byte-Pair Encoding)**: Used by GPT models
  - Byte-level BPE for better unicode handling
  - Efficient merge operations
  - Pre-tokenization with regex patterns
- **WordPiece**: Used by BERT models
  - Greedy longest-match-first algorithm
  - Unknown token handling
  - Case and accent normalization options
- **SentencePiece**: Unsupervised text tokenizer
  - Unigram and BPE modes
  - Direct training from raw text
  - Language-agnostic design
- **TikToken**: OpenAI tokenizer (cl100k_base, p50k_base, r50k_base)
  - Compatible with GPT-4, ChatGPT, Codex
  - Fast BPE implementation
- **Fairseq**: Dictionary format support
  - Moses-style tokenization
  - Subword NMT integration

#### Language-Specific
- **Arabic**: Morphological segmentation, right-to-left handling, Farasa integration
- **Chinese**: Character-based, jieba-based word segmentation, radical decomposition
- **Japanese**: MeCab/SudachiPy integration, kanji/kana normalization, reading variants
- **Korean**: Morpheme-based with Mecab/Komoran, Hangul decomposition

#### Domain-Specific
- **Chemical**: SMILES notation, molecular formula, IUPAC names
- **Music**: ABC notation, MusicXML, chord/tempo symbols
- **Math**: LaTeX, MathML, expression tree tokenization
- **Code**: Language-aware (Python, Rust, JavaScript, C/C++, SQL)
- **BIO**: FASTA/FASTQ, amino acids, gene ontology terms
- **Multimodal**: Image patches, audio frames, video token interleaving

### Core Features
- **Zero-copy vocabulary access**: Memory-mapped vocabularies for large-scale use
- **SIMD acceleration**: Vectorized encoding operations for high throughput
- **Async batch processing**: Non-blocking tokenization via scirs2-core
- **Vocabulary intelligence**: Semantic analysis, compression efficiency, cross-lingual coverage
- **Training infrastructure**: BPE, WordPiece, SentencePiece trainers from corpus
- **Batch processing**: Efficient handling of multiple texts
- **Offset mapping**: Track original text positions
- **Special tokens**: Configurable special token handling
- **Padding/Truncation**: Automatic sequence length management
- **Thread-safe**: Safe concurrent tokenization

### Pre/Post Processing
- **Normalization**: Unicode normalization (NFC, NFD, NFKC, NFKD)
- **Pre-tokenization**: Whitespace, punctuation, regex-based splitting
- **Post-processing**: Template-based token type IDs and attention masks
- **Decoding**: Convert tokens back to text with proper formatting

### Feature Flags
- `python` — PyO3 Python bindings (pip-installable package)
- `mecab` — Japanese/CJK tokenization via MeCab
- `gpu` — GPU-accelerated tokenization for large batches
- `jax` — JAX integration for JAX/Flax workflows
- `onnx` — ONNX export for tokenizer graphs
- `pytorch` — PyTorch DataLoader integration
- `tensorflow` — TensorFlow tf.data pipeline integration

## Usage Example

### Basic Tokenization
```rust
use trustformers_tokenizers::{
    tokenizer::Tokenizer,
    models::bpe::BPE,
    pre_tokenizers::whitespace::Whitespace,
    processors::template::TemplateProcessing,
};

// Create a tokenizer
let mut tokenizer = Tokenizer::new(BPE::default());

// Add pre-tokenizer
tokenizer.with_pre_tokenizer(Whitespace::default());

// Add post-processor for BERT-style tokens
tokenizer.with_post_processor(
    TemplateProcessing::builder()
        .single("[CLS] $A [SEP]")
        .pair("[CLS] $A [SEP] $B [SEP]")
        .build()
);

// Tokenize text
let encoding = tokenizer.encode("Hello, world!", true)?;
println!("Tokens: {:?}", encoding.get_tokens());
println!("IDs: {:?}", encoding.get_ids());
```

### Loading Pre-trained Tokenizers
```rust
use trustformers_tokenizers::tokenizer::Tokenizer;

// Load from file
let tokenizer = Tokenizer::from_file("path/to/tokenizer.json")?;

// Load from Hugging Face format
let tokenizer = Tokenizer::from_pretrained("bert-base-uncased")?;

// Tokenize with offsets
let encoding = tokenizer.encode_with_offsets("Hello world!", true)?;
for (token, (start, end)) in encoding.get_tokens().iter()
    .zip(encoding.get_offsets()) {
    println!("{}: {}-{}", token, start, end);
}
```

### Batch Tokenization
```rust
let texts = vec![
    "First sentence.",
    "Second sentence is longer.",
    "Third one.",
];

let encodings = tokenizer.encode_batch(&texts, true)?;

// Pad to same length
let padded = tokenizer.pad_batch(&mut encodings, None)?;
```

### Language-Specific Tokenization
```rust
use trustformers_tokenizers::languages::japanese::MeCabTokenizer;

// Japanese tokenizer with MeCab (requires `mecab` feature)
let tokenizer = MeCabTokenizer::new()?;
let tokens = tokenizer.encode("こんにちは世界", true)?;

use trustformers_tokenizers::languages::chinese::JiebaTokenizer;

// Chinese word segmentation
let tokenizer = JiebaTokenizer::new()?;
let tokens = tokenizer.encode("你好世界", true)?;
```

### Domain-Specific Tokenization
```rust
use trustformers_tokenizers::domains::chemical::SmilesTokenizer;

// Chemical SMILES tokenizer
let tokenizer = SmilesTokenizer::new()?;
let tokens = tokenizer.encode("CC(=O)Oc1ccccc1C(=O)O", true)?; // Aspirin

use trustformers_tokenizers::domains::code::CodeTokenizer;

// Code-aware tokenizer
let tokenizer = CodeTokenizer::for_language("rust")?;
let tokens = tokenizer.encode("fn main() { println!(\"Hello\"); }", true)?;
```

## Architecture

```
trustformers-tokenizers/
├── src/
│   ├── tokenizer/        # Main tokenizer interface
│   ├── models/           # Tokenization algorithms
│   │   ├── bpe/         # BPE implementation
│   │   ├── wordpiece/   # WordPiece implementation
│   │   ├── unigram/     # SentencePiece unigram
│   │   ├── tiktoken/    # TikToken implementation
│   │   └── fairseq/     # Fairseq dictionary
│   ├── languages/        # Language-specific tokenizers
│   │   ├── arabic/      # Arabic morphological
│   │   ├── chinese/     # Chinese segmentation
│   │   ├── japanese/    # Japanese MeCab/SudachiPy
│   │   └── korean/      # Korean morpheme
│   ├── domains/          # Domain-specific tokenizers
│   │   ├── chemical/    # SMILES/IUPAC
│   │   ├── music/       # ABC/MusicXML
│   │   ├── math/        # LaTeX/MathML
│   │   ├── code/        # Programming languages
│   │   ├── bio/         # FASTA/amino acids
│   │   └── multimodal/  # Vision/audio tokens
│   ├── pre_tokenizers/   # Pre-processing steps
│   ├── normalizers/      # Text normalization
│   ├── processors/       # Post-processing
│   ├── decoders/        # Token-to-text decoding
│   ├── training/         # Tokenizer trainers
│   └── intelligence/     # Vocabulary analysis tools
```

## Performance

### Benchmarks
| Tokenizer | Text Size | Time (ms) | Throughput (MB/s) |
|-----------|-----------|-----------|-------------------|
| BPE | 1KB | 0.12 | 8.3 |
| BPE | 1MB | 45 | 22.2 |
| WordPiece | 1KB | 0.15 | 6.7 |
| WordPiece | 1MB | 52 | 19.2 |
| SentencePiece | 1KB | 0.18 | 5.6 |
| SentencePiece | 1MB | 61 | 16.4 |
| BPE (SIMD) | 1MB | 28 | 35.7 |
| BPE (Batch/Async) | 16x1KB | 0.85 | 18.8 |

*Benchmarks on Apple M1, single-threaded unless noted*

### Memory Usage
- BPE with 50k vocabulary: ~12MB
- WordPiece with 30k vocabulary: ~8MB
- SentencePiece with 32k vocabulary: ~10MB
- Zero-copy memory-mapped vocab (100k): ~2MB resident

## Training Tokenizers

```rust
use trustformers_tokenizers::{
    models::bpe::{BPE, BpeTrainer},
    tokenizer::Tokenizer,
};

// Configure trainer
let mut trainer = BpeTrainer::builder()
    .vocab_size(30000)
    .min_frequency(2)
    .special_tokens(vec![
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
    ])
    .build();

// Train from files
let files = vec!["data/corpus.txt"];
tokenizer.train(&files, trainer)?;

// Save trained tokenizer
tokenizer.save("my_tokenizer.json", false)?;
```

## Vocabulary Intelligence

```rust
use trustformers_tokenizers::intelligence::VocabAnalyzer;

let analyzer = VocabAnalyzer::new(&tokenizer);

// Semantic clustering
let clusters = analyzer.cluster_semantic_tokens()?;

// Compression efficiency
let stats = analyzer.compression_stats(&corpus)?;
println!("Avg tokens/word: {:.2}", stats.avg_tokens_per_word);

// Cross-lingual coverage
let cov = analyzer.cross_lingual_coverage(&["en", "ja", "zh", "ar"])?;
```

## Compatibility

### Supported Formats
- **Hugging Face**: Full compatibility with `tokenizers` library
- **SentencePiece**: Load `.model` files directly
- **TikToken**: Load `.tiktoken` encoding files
- **Fairseq**: Dictionary format support
- **Custom**: JSON-based configuration

### Integration
- Direct use in TrustformeRS models
- Python bindings via `trustformers-py` (PyO3, `python` feature)
- WASM support via `trustformers-wasm`
- C API for other language bindings

## Advanced Features

### Custom Pre-tokenizers
```rust
use trustformers_tokenizers::pre_tokenizers::{
    PreTokenizer, PreTokenizedString,
};

struct CustomPreTokenizer;

impl PreTokenizer for CustomPreTokenizer {
    fn pre_tokenize(&self, pretok: &mut PreTokenizedString) -> Result<()> {
        // Custom splitting logic
        pretok.split(|c| c.is_whitespace(), SplitDelimiterBehavior::Remove)?;
        Ok(())
    }
}
```

### Performance Tips
1. **Reuse tokenizers**: Create once, use many times
2. **Batch processing**: Tokenize multiple texts together
3. **Pre-compile regex**: For custom pre-tokenizers
4. **Zero-copy vocab**: Memory-map vocabularies for 100k+ tokens
5. **Use appropriate tokenizer**: BPE for generation, WordPiece for understanding
6. **Enable SIMD**: Compile with `RUSTFLAGS="-C target-feature=+simd128"`

## Testing

- **500 unit tests** with 100% pass rate
- Cross-validation with Python tokenizers (HuggingFace, tiktoken, SentencePiece)
- Fuzzing tests for edge cases
- Performance benchmarks (throughput regression detection)
- Memory leak detection

## License

Apache-2.0
