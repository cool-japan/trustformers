# TrustformeRS Tokenizers - Python Bindings

High-performance tokenizers for transformer models, implemented in Rust with Python bindings.

## Features

- **Fast**: Rust implementation with memory-efficient algorithms
- **Comprehensive**: Support for BPE, WordPiece, Unigram, and character-level tokenization
- **Compatible**: Drop-in replacement for HuggingFace tokenizers in many cases
- **Extensible**: Easy to create custom tokenizers
- **Production-ready**: Memory-efficient, thread-safe implementations

## Installation

### From PyPI (when available)

```bash
pip install trustformers-tokenizers
```

### Development Installation

1. Make sure you have Rust installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Install maturin for building Python extensions:
```bash
pip install maturin
```

3. Build and install the package:
```bash
cd trustformers-tokenizers
maturin develop --features python
```

## Quick Start

```python
import trustformers_tokenizers as tt

# Create a character-level tokenizer
tokenizer = tt.CharTokenizer(lowercase=True, max_length=512)

# Encode text
text = "Hello, world! How are you today?"
encoded = tokenizer.encode(text)
print(f"Tokens: {encoded.input_ids}")
print(f"Length: {len(encoded)}")

# Decode back to text
decoded = tokenizer.decode(encoded.input_ids)
print(f"Decoded: {decoded}")

# Load a pre-trained tokenizer
tokenizer = tt.AutoTokenizer.from_pretrained("bert-base-uncased")
```

## Tokenizer Types

### Character-Level Tokenizer

```python
from trustformers_tokenizers import CharTokenizer

# Basic character tokenizer
tokenizer = CharTokenizer(lowercase=False, max_length=512)

# With lowercasing
tokenizer = CharTokenizer(lowercase=True, max_length=1024)
```

### BPE Tokenizer

```python
from trustformers_tokenizers import BPETokenizer

# Create from vocabulary and merges
vocab = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}
merges = [("h", "e"), ("l", "l")]
tokenizer = BPETokenizer(vocab, merges)

# Load from files
tokenizer = BPETokenizer.from_files("vocab.txt", "merges.txt")
```

### WordPiece Tokenizer

```python
from trustformers_tokenizers import WordPieceTokenizer

# Create from vocabulary
vocab = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "##world": 3}
tokenizer = WordPieceTokenizer(vocab, unk_token="[UNK]")

# Load from file
tokenizer = WordPieceTokenizer.from_file("vocab.txt")
```

### Auto Tokenizer

```python
from trustformers_tokenizers import AutoTokenizer

# Load from pre-trained model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load from file
tokenizer = AutoTokenizer.from_file("tokenizer.json")
```

## Training Custom Tokenizers

```python
from trustformers_tokenizers import TokenizerTrainer

# Prepare training data
training_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is fascinating!",
    # ... more texts
]

# Create trainer
trainer = TokenizerTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    min_frequency=2,
)

# Train BPE tokenizer
bpe_tokenizer = trainer.train_bpe(texts=training_texts)

# Train WordPiece tokenizer
wp_tokenizer = trainer.train_wordpiece(texts=training_texts)

# Train from files
bpe_tokenizer = trainer.train_bpe(files=["corpus1.txt", "corpus2.txt"])
```

## Analysis and Benchmarking

### Coverage Analysis

```python
from trustformers_tokenizers import analyze_coverage

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
test_texts = ["Your test texts here..."]

coverage = analyze_coverage(tokenizer, test_texts)
print(f"Coverage ratio: {coverage['coverage_ratio']:.3f}")
print(f"Average tokens per text: {coverage['avg_tokens_per_text']:.1f}")
print(f"OOV rate: {coverage['oov_rate']:.3f}")
```

### Benchmarking

```python
from trustformers_tokenizers import benchmark_tokenizers

tokenizers = {
    "bert": AutoTokenizer.from_pretrained("bert-base-uncased"),
    "char": CharTokenizer(lowercase=True),
}

results = benchmark_tokenizers(tokenizers, test_texts, iterations=3)
for name, result in results.items():
    print(f"{name}: {result['encoding']['texts_per_second']:.1f} texts/sec")
```

### Tokenization Comparison

```python
from trustformers_tokenizers import compare_tokenizations

comparison = compare_tokenizations(tokenizers, ["Hello world!"])
for result in comparison:
    print(f"Text: {result['text']}")
    for name, tokenization in result['tokenizations'].items():
        print(f"  {name}: {tokenization['tokens']}")
```

## Batch Processing

```python
# Batch encoding
texts = ["First text", "Second text", "Third text"]
encoded_batch = tokenizer.encode_batch(texts)

# Batch decoding  
token_ids_batch = [encoded.input_ids for encoded in encoded_batch]
decoded_batch = tokenizer.decode_batch(token_ids_batch)
```

## Memory Profiling

```python
from trustformers_tokenizers.utils import profile_memory_usage

profile = profile_memory_usage(tokenizer, test_texts)
print(f"Memory per text: {profile['memory_per_text_kb']:.1f} KB")
print(f"Peak memory: {profile['peak_memory_mb']:.1f} MB")
```

## Saving and Loading

```python
from trustformers_tokenizers import save_tokenizer, load_tokenizer

# Save tokenizer
save_tokenizer(tokenizer, "my_tokenizer/")

# Load tokenizer
loaded_tokenizer = load_tokenizer("my_tokenizer/")
```

## Advanced Features

### Subword Regularization

```python
# Enable dropout for subword regularization during training
trainer = TokenizerTrainer(vocab_size=30000, dropout=0.1)
```

### Custom Special Tokens

```python
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[CUSTOM]"]
trainer = TokenizerTrainer(vocab_size=30000, special_tokens=special_tokens)
```

### Memory-Mapped Vocabularies

```python
# For very large vocabularies, use memory mapping for efficiency
from trustformers_tokenizers import MmapVocab
vocab = MmapVocab.from_file("large_vocab.bin")
```

## Performance Tips

1. **Use character-level tokenizers** for maximum speed on short texts
2. **Pre-compile vocabularies** to binary format for faster loading
3. **Use batch processing** for better throughput
4. **Enable memory mapping** for large vocabularies
5. **Profile memory usage** to optimize for your use case

## API Reference

### Core Classes

- `TokenizedInput`: Container for tokenized text with IDs, attention mask, etc.
- `Tokenizer`: Base tokenizer interface
- `AutoTokenizer`: Automatic tokenizer loading
- `CharTokenizer`: Character-level tokenization
- `BPETokenizer`: Byte-Pair Encoding tokenization
- `WordPieceTokenizer`: WordPiece tokenization
- `UnigramTokenizer`: Unigram tokenization

### Training

- `TokenizerTrainer`: High-level training interface
- `TrainingConfig`: Training configuration options

### Utilities

- `analyze_coverage()`: Analyze tokenizer coverage on texts
- `benchmark_tokenizers()`: Benchmark multiple tokenizers
- `compare_tokenizations()`: Compare how different tokenizers process text
- `profile_memory_usage()`: Profile memory usage
- `validate_tokenizer_roundtrip()`: Validate encode/decode consistency

## Examples

See `python/example.py` for comprehensive usage examples.

## Contributing

1. Make sure tests pass: `pytest tests/`
2. Run benchmarks: `python python/example.py`
3. Format code: `black . && isort .`
4. Type check: `mypy trustformers_tokenizers/`

## License

This project is licensed under either of

- Apache License, Version 2.0
- MIT License

at your option.

## Performance Comparison

| Tokenizer | Speed (texts/sec) | Memory (MB) | Use Case |
|-----------|-------------------|-------------|----------|
| Character | 50,000+ | Low | Fast processing, simple texts |
| BPE | 10,000+ | Medium | Balanced, good for most tasks |
| WordPiece | 8,000+ | Medium | BERT-style models |
| Unigram | 7,000+ | Medium | SentencePiece-style models |

*Benchmarks run on typical modern hardware with 1000-character texts.*