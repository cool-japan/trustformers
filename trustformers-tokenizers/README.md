# trustformers-tokenizers

High-performance tokenization library for transformer models with support for multiple tokenization algorithms.

## Current State

This crate provides **production-ready tokenizer implementations** including BPE (Byte-Pair Encoding), WordPiece, and SentencePiece tokenizers. It's designed to be fast, memory-efficient, and compatible with popular tokenizer formats.

## Features

### Implemented Tokenizers
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

### Core Features
- **Fast tokenization**: Optimized Rust implementation
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

## Architecture

```
trustformers-tokenizers/
├── src/
│   ├── tokenizer/        # Main tokenizer interface
│   ├── models/           # Tokenization algorithms
│   │   ├── bpe/         # BPE implementation
│   │   ├── wordpiece/   # WordPiece implementation
│   │   └── unigram/     # SentencePiece unigram
│   ├── pre_tokenizers/   # Pre-processing steps
│   ├── normalizers/      # Text normalization
│   ├── processors/       # Post-processing
│   └── decoders/        # Token-to-text decoding
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

*Benchmarks on Apple M1, single-threaded*

### Memory Usage
- BPE with 50k vocabulary: ~12MB
- WordPiece with 30k vocabulary: ~8MB
- SentencePiece with 32k vocabulary: ~10MB

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

## Compatibility

### Supported Formats
- **Hugging Face**: Full compatibility with `tokenizers` library
- **SentencePiece**: Load `.model` files directly
- **Fairseq**: Dictionary format support
- **Custom**: JSON-based configuration

### Integration
- Direct use in TrustformeRS models
- Python bindings via `trustformers-py`
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
4. **Memory-map large vocabularies**: For 100k+ tokens
5. **Use appropriate tokenizer**: BPE for generation, WordPiece for understanding

## Testing

- Unit tests for each tokenizer type
- Cross-validation with Python tokenizers
- Fuzzing tests for edge cases
- Performance benchmarks
- Memory leak detection

## License

MIT OR Apache-2.0