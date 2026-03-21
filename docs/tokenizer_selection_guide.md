# TrustformeRS Tokenizer Selection Guide

## Overview

Choosing the right tokenizer is crucial for optimal model performance. TrustformeRS provides a comprehensive suite of tokenizers, each optimized for different use cases, languages, and domains. This guide helps you select the most appropriate tokenizer for your specific needs.

## Quick Selection Matrix

| Use Case | Recommended Tokenizer | Alternative Options |
|----------|----------------------|-------------------|
| General English Text | WordPiece (BERT-style) | BPE, SentencePiece |
| Multilingual Tasks | SentencePiece | Unigram, BPE |
| Code Generation | BPE (GPT-style) | CodeTokenizer |
| Mathematical Text | MathTokenizer | BPE with math vocab |
| Scientific Papers | MathTokenizer + Custom | BPE, WordPiece |
| Chemical Formulas | ChemicalTokenizer | Custom vocabulary |
| DNA/Protein Sequences | BioTokenizer | Character-level |
| Music Notation | MusicTokenizer | Custom format |
| Arabic Text | ArabicTokenizer | SentencePiece |
| Chinese Text | ChineseTokenizer | SentencePiece |
| Japanese Text | JapaneseTokenizer (MeCab) | SentencePiece |
| Korean Text | KoreanTokenizer | SentencePiece |
| Thai Text | ThaiTokenizer | SentencePiece |
| Memory-Constrained | CompressedVocab + BPE | Minimal Perfect Hash |
| High-Speed Inference | SIMD-optimized tokenizers | Memory-mapped vocab |
| Large-Scale Training | Streaming tokenizers | Async tokenizers |

## Tokenizer Categories

### 1. Subword Tokenizers

#### Byte-Pair Encoding (BPE)
- **Best for**: Code generation, English text, general NLP
- **Advantages**: Simple, effective, widely adopted
- **Use when**: Training GPT-style models, code tasks
- **Implementation**: `BPETokenizer`

```rust
use trustformers_tokenizers::BPETokenizer;

let tokenizer = BPETokenizer::new()
    .with_vocab_size(32000)
    .with_unicode_normalization(true)
    .build()?;
```

#### WordPiece
- **Best for**: BERT-style models, classification tasks
- **Advantages**: Balanced subword splitting, good for downstream tasks
- **Use when**: Fine-tuning BERT family models
- **Implementation**: `WordPieceTokenizer`

```rust
use trustformers_tokenizers::WordPieceTokenizer;

let tokenizer = WordPieceTokenizer::new()
    .with_vocab_size(30000)
    .with_unk_token("[UNK]")
    .build()?;
```

#### SentencePiece
- **Best for**: Multilingual tasks, language-agnostic processing
- **Advantages**: No pre-tokenization required, handles any language
- **Use when**: Training multilingual models, unknown languages
- **Implementation**: `SentencePieceTokenizer`

```rust
use trustformers_tokenizers::SentencePieceTokenizer;

let tokenizer = SentencePieceTokenizer::from_file("model.spm")?;
```

#### Unigram
- **Best for**: High-quality subword segmentation
- **Advantages**: Probabilistic segmentation, multiple tokenizations
- **Use when**: Quality is more important than speed
- **Implementation**: `UnigramTokenizer`

```rust
use trustformers_tokenizers::UnigramTokenizer;

let tokenizer = UnigramTokenizer::new()
    .with_vocab_size(32000)
    .with_alpha(0.1)
    .build()?;
```

### 2. Character-Level Tokenizers

#### Character Tokenizer
- **Best for**: Unknown languages, very noisy text
- **Advantages**: No OOV issues, simple
- **Use when**: Dealing with corrupted text, character-level tasks
- **Implementation**: `CharTokenizer`

```rust
use trustformers_tokenizers::CharTokenizer;

let tokenizer = CharTokenizer::new()
    .with_lowercase(true)
    .with_chinese_support(true)
    .build()?;
```

#### CANINE Tokenizer
- **Best for**: Multilingual without fixed vocabulary
- **Advantages**: No vocabulary constraints, hash-based
- **Use when**: Extreme multilingual scenarios
- **Implementation**: `CanineTokenizer`

```rust
use trustformers_tokenizers::CanineTokenizer;

let tokenizer = CanineTokenizer::new()
    .with_hash_size(131072)
    .with_downsample_rate(4)
    .build()?;
```

### 3. Language-Specific Tokenizers

#### Japanese Tokenizer
- **Best for**: Japanese text processing
- **Advantages**: Proper morphological analysis
- **Requirements**: MeCab integration
- **Implementation**: `JapaneseTokenizer`

```rust
use trustformers_tokenizers::JapaneseTokenizer;

let tokenizer = JapaneseTokenizer::new()
    .with_mode(TokenizationMode::Word)
    .with_pos_tagging(true)
    .build()?;
```

#### Chinese Tokenizer
- **Best for**: Chinese text with word segmentation
- **Advantages**: Proper word boundary detection
- **Implementation**: `ChineseTokenizer`

```rust
use trustformers_tokenizers::ChineseTokenizer;

let tokenizer = ChineseTokenizer::new()
    .with_dictionary_path("dict.txt")
    .with_segmentation_algorithm(SegmentationAlgorithm::Viterbi)
    .build()?;
```

#### Arabic Tokenizer
- **Best for**: Arabic script languages
- **Advantages**: Handles RTL text, diacritics
- **Implementation**: `ArabicTokenizer`

```rust
use trustformers_tokenizers::ArabicTokenizer;

let tokenizer = ArabicTokenizer::new()
    .with_mode(ArabicTokenizationMode::Morphological)
    .with_dialect_support(true)
    .build()?;
```

### 4. Domain-Specific Tokenizers

#### Code Tokenizer
- **Best for**: Programming languages, code analysis
- **Advantages**: Syntax-aware, 25+ language support
- **Implementation**: `CodeTokenizer`

```rust
use trustformers_tokenizers::{CodeTokenizer, ProgrammingLanguage};

let tokenizer = CodeTokenizer::new()
    .with_language(ProgrammingLanguage::Rust)
    .with_comment_parsing(true)
    .with_string_literal_handling(true)
    .build()?;
```

#### Mathematical Tokenizer
- **Best for**: Mathematical expressions, scientific text
- **Advantages**: LaTeX support, mathematical symbols
- **Implementation**: `MathTokenizer`

```rust
use trustformers_tokenizers::MathTokenizer;

let tokenizer = MathTokenizer::new()
    .with_latex_support(true)
    .with_scientific_notation(true)
    .with_greek_letters(true)
    .build()?;
```

#### Chemical Tokenizer
- **Best for**: Chemical formulas, molecular structures
- **Advantages**: SMILES/InChI support, molecular analysis
- **Implementation**: `ChemicalTokenizer`

```rust
use trustformers_tokenizers::{ChemicalTokenizer, ChemicalFormat};

let tokenizer = ChemicalTokenizer::new()
    .with_format(ChemicalFormat::SMILES)
    .with_molecular_analysis(true)
    .build()?;
```

#### Biological Sequence Tokenizer
- **Best for**: DNA, RNA, protein sequences
- **Advantages**: Bioinformatics support, k-mer analysis
- **Implementation**: `BioTokenizer`

```rust
use trustformers_tokenizers::{BioTokenizer, SequenceType};

let tokenizer = BioTokenizer::new()
    .with_sequence_type(SequenceType::DNA)
    .with_kmer_size(3)
    .with_translation_support(true)
    .build()?;
```

## Performance Considerations

### Memory Usage

| Tokenizer Type | Memory Usage | Best For |
|---------------|--------------|----------|
| CompressedVocab | Low | Memory-constrained devices |
| MemoryMapped | Very Low | Large vocabularies |
| MinimalPerfectHash | Minimal | Lookup-heavy applications |
| SharedVocabPool | Optimized | Multiple tokenizers |

### Speed Optimization

| Feature | Speed Boost | Use Case |
|---------|-------------|----------|
| SIMD Operations | 2-4x | CPU-intensive processing |
| GPU Acceleration | 10-50x | Large-scale processing |
| Parallel Processing | N-cores | Batch processing |
| Async Tokenization | Variable | Non-blocking operations |

### Vocabulary Size Guidelines

```rust
// Small models (mobile/edge)
let small_vocab = 8000..16000;

// Standard models
let standard_vocab = 30000..50000;

// Large multilingual models
let large_vocab = 100000..250000;

// Code/domain-specific
let specialized_vocab = 50000..100000;
```

## Decision Tree

```
1. What type of text are you processing?
   ├── Code → CodeTokenizer
   ├── Mathematical → MathTokenizer
   ├── Chemical → ChemicalTokenizer
   ├── Biological → BioTokenizer
   ├── Music → MusicTokenizer
   └── Natural Language → Continue to 2

2. What languages do you need to support?
   ├── Single Language → Continue to 3
   ├── Multilingual → SentencePiece or Unigram
   └── Unknown/Many → CANINE

3. Which single language?
   ├── English → Continue to 4
   ├── Japanese → JapaneseTokenizer
   ├── Chinese → ChineseTokenizer
   ├── Arabic → ArabicTokenizer
   ├── Korean → KoreanTokenizer
   ├── Thai → ThaiTokenizer
   └── Other → SentencePiece

4. What is your model architecture?
   ├── BERT-family → WordPiece
   ├── GPT-family → BPE
   ├── T5-family → SentencePiece
   └── Custom → BPE or WordPiece

5. What are your performance requirements?
   ├── Highest Quality → Unigram
   ├── Balanced → BPE or WordPiece
   ├── Fastest → CharTokenizer with SIMD
   └── Memory Constrained → CompressedVocab
```

## Migration Considerations

### From HuggingFace Transformers

```rust
// HuggingFace AutoTokenizer equivalent
use trustformers_tokenizers::AutoTokenizer;

let tokenizer = AutoTokenizer::from_pretrained("bert-base-uncased")?;
```

### From OpenAI tiktoken

```rust
// tiktoken equivalent
use trustformers_tokenizers::TiktokenTokenizer;

let tokenizer = TiktokenTokenizer::from_tiktoken_file("cl100k_base.tiktoken")?;
```

### Custom Format Conversion

```rust
use trustformers_tokenizers::CustomFormatTokenizer;

// Convert from custom format
let tokenizer = CustomFormatTokenizer::from_custom_format(
    vocab_file,
    normalization_rules,
    special_tokens
)?;
```

## Best Practices

### 1. Vocabulary Size Selection

- **Small models (< 100M params)**: 8K-16K tokens
- **Medium models (100M-1B params)**: 30K-50K tokens  
- **Large models (> 1B params)**: 50K-100K tokens
- **Multilingual models**: 100K-250K tokens

### 2. Special Token Management

```rust
let tokenizer = BPETokenizer::new()
    .with_special_tokens(vec![
        ("[PAD]", 0),
        ("[UNK]", 1),
        ("[CLS]", 2),
        ("[SEP]", 3),
        ("[MASK]", 4),
    ])
    .build()?;
```

### 3. Normalization Strategy

```rust
let tokenizer = BPETokenizer::new()
    .with_unicode_normalization(true)
    .with_case_normalization(false) // Preserve case for code
    .with_accent_removal(false)     // Preserve for multilingual
    .build()?;
```

### 4. Performance Optimization

```rust
// For high-throughput applications
let fast_tokenizer = BPETokenizer::new()
    .with_simd_optimization(true)
    .with_parallel_processing(true)
    .with_memory_mapping(true)
    .build()?;
```

## Common Pitfalls

### 1. Wrong Tokenizer for Task
- **Issue**: Using CharTokenizer for English text
- **Solution**: Use BPE or WordPiece for better subword modeling

### 2. Incompatible Vocabulary Size
- **Issue**: Too small vocabulary for complex domain
- **Solution**: Increase vocabulary size or use domain-specific tokenizer

### 3. Missing Language Support
- **Issue**: Using English-only tokenizer for multilingual text
- **Solution**: Use SentencePiece or language-specific tokenizers

### 4. Performance Bottlenecks
- **Issue**: Slow tokenization in production
- **Solution**: Enable SIMD, use parallel processing, or GPU acceleration

## Advanced Usage

### Streaming Tokenization

```rust
use trustformers_tokenizers::StreamingTokenizer;

let streaming = StreamingTokenizer::new(base_tokenizer)
    .with_chunk_size(1024)
    .with_overlap(64)
    .build()?;

for chunk in streaming.process_stream(large_text_stream) {
    // Process chunk
}
```

### Batch Processing

```rust
use trustformers_tokenizers::BatchTokenizer;

let batch_tokenizer = BatchTokenizer::new(base_tokenizer)
    .with_padding(true)
    .with_truncation(512)
    .build()?;

let results = batch_tokenizer.encode_batch(texts)?;
```

### Async Tokenization

```rust
use trustformers_tokenizers::AsyncTokenizer;

let async_tokenizer = AsyncTokenizer::new(base_tokenizer)
    .with_max_concurrent_tasks(8)
    .build()?;

let result = async_tokenizer.encode_async(text).await?;
```

## Testing and Validation

### Coverage Analysis

```rust
use trustformers_tokenizers::CoverageAnalyzer;

let analyzer = CoverageAnalyzer::new();
let report = analyzer.analyze_coverage(&tokenizer, &test_corpus)?;

println!("Character coverage: {:.2}%", report.character_coverage * 100.0);
println!("OOV rate: {:.2}%", report.oov_rate * 100.0);
```

### Performance Profiling

```rust
use trustformers_tokenizers::PerformanceProfiler;

let profiler = PerformanceProfiler::new();
let metrics = profiler.profile_tokenizer(&tokenizer, &test_data)?;

println!("Tokens/second: {}", metrics.throughput.tokens_per_second);
println!("Memory usage: {} MB", metrics.memory.peak_usage_mb);
```

## Conclusion

Selecting the right tokenizer depends on your specific use case, performance requirements, and target languages. Start with the recommended tokenizer for your domain, then optimize based on performance profiling and coverage analysis. TrustformeRS provides the flexibility to fine-tune tokenization for optimal results.

For additional guidance, consult the [Training Best Practices Guide](tokenizer_training_best_practices.md) and [Performance Tuning Guide](tokenizer_performance_tuning.md).