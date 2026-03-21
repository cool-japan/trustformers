# TrustformeRS Tokenizers API Reference

Complete API reference for the TrustformeRS Tokenizers library.

## Table of Contents

- [Core Tokenizers](#core-tokenizers)
- [Language-Specific Tokenizers](#language-specific-tokenizers)
- [Specialized Tokenizers](#specialized-tokenizers)
- [Framework Integration](#framework-integration)
- [Training and Analysis](#training-and-analysis)
- [Performance and Optimization](#performance-and-optimization)
- [Serialization and Storage](#serialization-and-storage)
- [Utilities and Helpers](#utilities-and-helpers)

## Core Tokenizers

### `TokenizerImpl`
Main tokenizer implementation with HuggingFace compatibility.

```rust
use trustformers_tokenizers::TokenizerImpl;

// Load from HuggingFace tokenizer.json
let tokenizer = TokenizerImpl::from_file("tokenizer.json")?;
let tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;

// Basic tokenization
let encoded = tokenizer.encode("Hello world")?;
let decoded = tokenizer.decode(&encoded.ids())?;

// With options
let encoded = tokenizer.encode_with_offsets("Hello world")?;
let tokens_with_alignment = tokenizer.encode_with_alignment("Hello world")?;
```

**Methods:**
- `from_file(path: &str) -> Result<Self>`
- `from_pretrained(model_name: &str) -> Result<Self>`
- `encode(text: &str) -> Result<TokenizedInput>`
- `encode_with_offsets(text: &str) -> Result<TokenizedInputWithOffsets>`
- `encode_with_alignment(text: &str) -> Result<TokenizedInputWithAlignment>`
- `decode(ids: &[u32]) -> Result<String>`
- `get_vocab() -> &Vocab`
- `save_to_file(path: &str) -> Result<()>`

### `BPETokenizer`
Byte Pair Encoding tokenizer with advanced features.

```rust
use trustformers_tokenizers::BPETokenizer;

// Create from vocabulary and merges
let tokenizer = BPETokenizer::from_files("vocab.json", "merges.txt")?;

// With options
let tokenizer = BPETokenizer::new()
    .with_normalizer(normalizer)
    .with_pre_tokenizer(pre_tokenizer)
    .with_post_processor(post_processor);

// Encoding with byte-level processing
let encoded = tokenizer.encode_as_bytes("Hello world")?;
```

**Methods:**
- `new() -> Self`
- `from_files(vocab: &str, merges: &str) -> Result<Self>`
- `with_normalizer(normalizer: impl Normalizer) -> Self`
- `encode_as_bytes(text: &str) -> Result<Vec<u8>>`
- `tokenize_with_offsets(text: &str) -> Result<Vec<(String, (usize, usize))>>`

### `WordPieceTokenizer`
WordPiece tokenizer for BERT-style models.

```rust
use trustformers_tokenizers::WordPieceTokenizer;

// Create with vocabulary
let tokenizer = WordPieceTokenizer::from_vocab(vocab, unk_token)?;

// With configuration
let tokenizer = WordPieceTokenizer::new()
    .with_vocab(vocab)
    .with_unk_token("[UNK]")
    .with_max_input_chars_per_word(200);
```

**Methods:**
- `new() -> Self`
- `from_vocab(vocab: HashMap<String, u32>, unk_token: &str) -> Result<Self>`
- `with_unk_token(token: &str) -> Self`
- `with_max_input_chars_per_word(max: usize) -> Self`

### `UnigramTokenizer`
Unigram tokenizer with Viterbi algorithm.

```rust
use trustformers_tokenizers::UnigramTokenizer;

// Create with vocabulary and scores
let tokenizer = UnigramTokenizer::from_vocab_and_scores(vocab, scores)?;

// Sample tokenization with regularization
let samples = tokenizer.sample_tokenize("Hello world", alpha=0.1, n_samples=5)?;
```

**Methods:**
- `from_vocab_and_scores(vocab: Vec<String>, scores: Vec<f32>) -> Result<Self>`
- `sample_tokenize(text: &str, alpha: f32, n_samples: usize) -> Result<Vec<Vec<String>>>`
- `get_score(piece: &str) -> Option<f32>`

### `CharTokenizer`
Character-level tokenizer.

```rust
use trustformers_tokenizers::CharTokenizer;

// Basic character tokenization
let tokenizer = CharTokenizer::new();
let encoded = tokenizer.encode("Hello")?; // ["H", "e", "l", "l", "o"]

// With options
let tokenizer = CharTokenizer::new()
    .with_lowercase(true)
    .with_special_tokens(vec!["[UNK]", "[PAD]"]);
```

**Methods:**
- `new() -> Self`
- `with_lowercase(lowercase: bool) -> Self`
- `with_special_tokens(tokens: Vec<&str>) -> Self`
- `with_max_length(max_len: usize) -> Self`

### `CanineTokenizer`
CANINE-style character hashing tokenizer.

```rust
use trustformers_tokenizers::CanineTokenizer;

// Character hashing without fixed vocabulary
let tokenizer = CanineTokenizer::new()
    .with_hash_size(262144)
    .with_downsample_rate(4);

let encoded = tokenizer.encode("Hello 世界")?;
```

**Methods:**
- `new() -> Self`
- `with_hash_size(size: usize) -> Self`
- `with_downsample_rate(rate: usize) -> Self`
- `hash_char(c: char) -> u32`

## Language-Specific Tokenizers

### `ChineseTokenizer`
Chinese text tokenization with word segmentation.

```rust
use trustformers_tokenizers::{ChineseTokenizer, ChineseTokenizerConfig};

let config = ChineseTokenizerConfig {
    normalize_traditional: true,
    handle_punctuation: true,
    ..Default::default()
};

let tokenizer = ChineseTokenizer::new(config);
let tokens = tokenizer.tokenize("你好世界")?;
```

**Configuration Options:**
- `normalize_traditional: bool` - Convert traditional to simplified
- `handle_punctuation: bool` - Special punctuation handling
- `max_word_length: usize` - Maximum word length for segmentation

### `JapaneseTokenizer`
Japanese tokenization with MeCab integration.

```rust
use trustformers_tokenizers::{JapaneseTokenizer, JapaneseMode};

// Different tokenization modes
let tokenizer = JapaneseTokenizer::new(JapaneseMode::Character);
let tokenizer = JapaneseTokenizer::new(JapaneseMode::Word);
let tokenizer = JapaneseTokenizer::new(JapaneseMode::Morpheme);

let tokens = tokenizer.tokenize("こんにちは世界")?;
```

**Modes:**
- `Character` - Character-level tokenization
- `Word` - Word-level tokenization
- `Morpheme` - Morphological analysis with MeCab

### `KoreanTokenizer`
Korean tokenization with Hangul processing.

```rust
use trustformers_tokenizers::{KoreanTokenizer, KoreanMode};

// Hangul syllable decomposition
let tokenizer = KoreanTokenizer::new(KoreanMode::Syllable);
let tokenizer = KoreanTokenizer::new(KoreanMode::Jamo);
let tokenizer = KoreanTokenizer::new(KoreanMode::Word);

let tokens = tokenizer.tokenize("안녕하세요")?;
```

**Modes:**
- `Syllable` - Syllable-level tokenization
- `Jamo` - Individual jamo (letter) tokenization
- `Word` - Word-level tokenization

### `ArabicTokenizer`
Arabic text tokenization with RTL support.

```rust
use trustformers_tokenizers::{ArabicTokenizer, ArabicMode};

let tokenizer = ArabicTokenizer::new(ArabicMode::Character);
let analysis = tokenizer.tokenize_with_analysis("مرحبا بك")?;
```

**Features:**
- RTL text processing
- Diacritical marks handling
- Morphological analysis
- Arabic-Indic numeral conversion

### `ThaiTokenizer`
Thai tokenization with word segmentation.

```rust
use trustformers_tokenizers::{ThaiTokenizer, ThaiMode};

let tokenizer = ThaiTokenizer::new(ThaiMode::Word);
let tokens = tokenizer.tokenize("สวัสดีครับ")?;
```

**Modes:**
- `Word` - Dictionary-based word segmentation
- `Syllable` - Syllable-level tokenization
- `Character` - Character-level tokenization

## Specialized Tokenizers

### `CodeTokenizer`
Programming language tokenization.

```rust
use trustformers_tokenizers::{CodeTokenizer, Language};

// Language-specific tokenization
let tokenizer = CodeTokenizer::new(Language::Rust);
let tokens = tokenizer.tokenize(r#"fn main() { println!("Hello"); }"#)?;

// Get detailed token information
let detailed = tokenizer.tokenize_detailed(code)?;
for token in detailed {
    println!("{:?}: {}", token.token_type, token.text);
}
```

**Supported Languages:**
- Rust, Python, JavaScript, TypeScript, Java, C/C++, Go, Swift, Kotlin, etc.

**Token Types:**
- `Keyword`, `Identifier`, `Literal`, `Operator`, `Punctuation`, `Comment`, `Whitespace`

### `MathTokenizer`
Mathematical notation tokenization.

```rust
use trustformers_tokenizers::MathTokenizer;

let tokenizer = MathTokenizer::new();
let tokens = tokenizer.tokenize(r"\frac{d}{dx} \sin(x) = \cos(x)")?;

// Mathematical analysis
let analysis = tokenizer.analyze(equation)?;
println!("Complexity score: {}", analysis.complexity_score);
```

**Features:**
- LaTeX command recognition
- Mathematical symbols and operators
- Greek letters and functions
- Scientific notation support

### `ChemicalTokenizer`
Chemical notation tokenization.

```rust
use trustformers_tokenizers::ChemicalTokenizer;

let tokenizer = ChemicalTokenizer::new();

// SMILES notation
let smiles_tokens = tokenizer.tokenize_smiles("CCO")?; // Ethanol

// Chemical formulas
let formula_tokens = tokenizer.tokenize_formula("H2SO4")?;

// Molecular analysis
let analysis = tokenizer.analyze_molecule("CCO")?;
println!("Molecular weight: {}", analysis.molecular_weight);
```

### `MusicTokenizer`
Musical notation tokenization.

```rust
use trustformers_tokenizers::MusicTokenizer;

let tokenizer = MusicTokenizer::new();

// ABC notation
let tokens = tokenizer.tokenize_abc("C D E F | G A B c |")?;

// Musical analysis
let analysis = tokenizer.analyze_music(notation)?;
println!("Key: {:?}", analysis.detected_key);
```

### `BioTokenizer`
Biological sequence tokenization.

```rust
use trustformers_tokenizers::BioTokenizer;

let tokenizer = BioTokenizer::new();

// DNA sequence
let dna_tokens = tokenizer.tokenize_dna("ATCGATCG")?;

// Protein sequence
let protein_tokens = tokenizer.tokenize_protein("MVLSPADKTNVKAAW")?;

// K-mer analysis
let kmers = tokenizer.generate_kmers("ATCGATCG", k=3)?;

// Sequence analysis
let analysis = tokenizer.analyze_sequence("ATCGATCG")?;
println!("GC content: {:.2}%", analysis.gc_content * 100.0);
```

### `MultimodalTokenizer`
Cross-modal tokenization for vision+language models.

```rust
use trustformers_tokenizers::{MultimodalTokenizer, MultimodalInput, ModalityType};

let tokenizer = MultimodalTokenizer::new();

// Combine text, image, and audio
let input = MultimodalInput::new()
    .add_text("A cat sitting on a chair")
    .add_image_patches(image_patches)
    .add_audio_frames(audio_frames);

let tokens = tokenizer.tokenize_multimodal(input)?;
```

**Supported Modalities:**
- Text, Images (patches), Audio (frames), Video (sequences), Tables, Graphs

## Framework Integration

### PyTorch Integration

```rust
use trustformers_tokenizers::{PyTorchTokenizer, PyTorchConfig, TensorDType};

let config = PyTorchConfig {
    device: "cuda:0".to_string(),
    dtype: TensorDType::Long,
    return_attention_mask: true,
    return_token_type_ids: true,
};

let tokenizer = PyTorchTokenizer::from_pretrained("bert-base-uncased")?
    .with_config(config);

// Direct tensor output
let batch = tokenizer.encode_batch_to_tensors(&texts)?;
```

### TensorFlow Integration

```rust
use trustformers_tokenizers::{TensorFlowTokenizer, TensorFlowConfig};

let tokenizer = TensorFlowTokenizer::from_pretrained("bert-base-uncased")?;

// TensorFlow tensor output
let tf_tensors = tokenizer.encode_to_tf_tensors("Hello world")?;

// Ragged tensor support
let ragged = tokenizer.encode_to_ragged_tensor(&texts)?;
```

### JAX Integration

```rust
use trustformers_tokenizers::{JaxTokenizer, JaxConfig};

let tokenizer = JaxTokenizer::from_pretrained("bert-base-uncased")?;

// JAX array output
let jax_arrays = tokenizer.encode_to_jax_arrays("Hello world")?;

// Compiled tokenization for performance
let compiled_tokenizer = tokenizer.compile()?;
```

### ONNX Export

```rust
use trustformers_tokenizers::{OnnxTokenizerExporter, OnnxExportConfig};

let exporter = OnnxTokenizerExporter::new();
let config = OnnxExportConfig {
    opset_version: 11,
    optimize_graph: true,
    ..Default::default()
};

// Export tokenizer to ONNX format
exporter.export_tokenizer(&tokenizer, "tokenizer.onnx", config)?;
```

## Training and Analysis

### Training Infrastructure

```rust
use trustformers_tokenizers::{StreamingTrainer, TrainingCheckpoint};

// Streaming training for large corpora
let trainer = StreamingTrainer::new();
let checkpoint = trainer.train_from_files(
    &["corpus1.txt", "corpus2.txt"],
    TrainingConfig {
        vocab_size: 32000,
        min_frequency: 2,
        algorithm: TrainingAlgorithm::BPE,
        streaming: true,
        checkpoint_every: 1000,
    }
)?;

// Resume from checkpoint
let resumed_trainer = StreamingTrainer::from_checkpoint(checkpoint)?;
```

### Coverage Analysis

```rust
use trustformers_tokenizers::{CoverageAnalyzer, CoverageConfig};

let analyzer = CoverageAnalyzer::new(CoverageConfig {
    min_frequency: 1,
    oov_threshold: 0.05,
    quality_thresholds: QualityThresholds::default(),
});

let report = analyzer.analyze_corpus(&tokenizer, &["test_corpus.txt"])?;
println!("Vocabulary coverage: {:.2}%", report.vocabulary_coverage.overall * 100.0);
```

### Vocabulary Analysis

```rust
use trustformers_tokenizers::VocabAnalyzer;

let analyzer = VocabAnalyzer::new();
let analysis = analyzer.analyze_vocabulary(&tokenizer.get_vocab())?;

// Detect issues
for issue in &analysis.issues {
    println!("{:?}: {}", issue.issue_type, issue.description);
}

// Get statistics
println!("Vocabulary size: {}", analysis.basic_stats.total_tokens);
println!("Average token length: {:.2}", analysis.basic_stats.avg_token_length);
```

### Performance Profiling

```rust
use trustformers_tokenizers::PerformanceProfiler;

let profiler = PerformanceProfiler::new();
let report = profiler.profile_tokenizer(&tokenizer, &test_texts)?;

println!("Tokens per second: {:.0}", report.throughput.tokens_per_second);
println!("Memory usage: {:.1} MB", report.memory.peak_mb);
```

## Performance and Optimization

### Parallel Processing

```rust
use trustformers_tokenizers::ParallelTokenizer;

// Parallel batch tokenization
let parallel_tokenizer = ParallelTokenizer::new(tokenizer)
    .with_thread_count(8)
    .with_chunk_size(1000);

let results = parallel_tokenizer.encode_batch(&large_text_batch)?;
```

### Async Tokenization

```rust
use trustformers_tokenizers::AsyncTokenizer;

// Async tokenization for non-blocking operations
let async_tokenizer = AsyncTokenizer::from_tokenizer(tokenizer);

// Async batch processing
let futures: Vec<_> = texts.iter()
    .map(|text| async_tokenizer.encode_async(text))
    .collect();

let results = futures_util::future::join_all(futures).await;
```

### Streaming Processing

```rust
use trustformers_tokenizers::StreamingTokenizer;

// Process large files without loading into memory
let streaming_tokenizer = StreamingTokenizer::new(tokenizer);

for batch in streaming_tokenizer.process_file_streaming("large_file.txt")? {
    let tokens = batch?;
    process_token_batch(tokens);
}
```

### Memory Optimization

```rust
use trustformers_tokenizers::{CompressedVocab, SharedVocabPool};

// Compressed vocabulary storage
let compressed_vocab = CompressedVocab::from_token_map(vocab_map)?;
let tokenizer = tokenizer.with_compressed_vocab(compressed_vocab);

// Shared vocabulary pool for multiple tokenizers
let vocab_pool = SharedVocabPool::new(VocabPoolConfig {
    max_size: 10,
    cleanup_interval: Duration::from_secs(300),
});

let tokenizer1 = tokenizer1.with_shared_vocab_pool(vocab_pool.clone());
let tokenizer2 = tokenizer2.with_shared_vocab_pool(vocab_pool.clone());
```

### SIMD Optimization

```rust
use trustformers_tokenizers::SimdTokenizer;

// SIMD-accelerated tokenization
let simd_tokenizer = SimdTokenizer::new();
let processed = simd_tokenizer.preprocess_batch(&texts)?;
```

### Sequence Packing

```rust
use trustformers_tokenizers::{SequencePacker, PackingStrategy};

// Efficient sequence packing for training
let packer = SequencePacker::new(PackingConfig {
    max_length: 512,
    strategy: PackingStrategy::BestFit,
    add_separators: true,
});

let packed = packer.pack_sequences(&token_sequences)?;
```

## Serialization and Storage

### Binary Format

```rust
use trustformers_tokenizers::BinarySerializer;

// Fast binary serialization
let serializer = BinarySerializer::new();
serializer.save_tokenizer(&tokenizer, "tokenizer.bin")?;

let loaded_tokenizer = serializer.load_tokenizer("tokenizer.bin")?;
```

### MessagePack

```rust
use trustformers_tokenizers::MessagePackSerializer;

// Cross-platform MessagePack format
let serializer = MessagePackSerializer::new();
let data = serializer.serialize_tokenizer(&tokenizer)?;
let loaded = serializer.deserialize_tokenizer(&data)?;
```

### Protocol Buffers

```rust
use trustformers_tokenizers::ProtobufSerializer;

// Industry-standard protobuf format
let serializer = ProtobufSerializer::new();
serializer.export_tokenizer(&tokenizer, "tokenizer.pb", ProtobufFormat::Binary)?;
```

### Zero-Copy Loading

```rust
use trustformers_tokenizers::ZeroCopyTokenizer;

// Memory-mapped zero-copy loading
let builder = ZeroCopyBuilder::new();
builder.build_from_tokenizer(&tokenizer, "tokenizer.zc")?;

let zero_copy_tokenizer = ZeroCopyTokenizer::load("tokenizer.zc")?;
```

## Utilities and Helpers

### Text Alignment

```rust
use trustformers_tokenizers::AlignmentEngine;

// Token-to-word alignment
let aligner = AlignmentEngine::new(AlignmentConfig::default());
let alignment = aligner.align_tokens("Hello world", &tokens)?;

for aligned_token in alignment.tokens {
    println!("Token '{}' aligns to word {}", aligned_token.text, aligned_token.word_index);
}
```

### Special Token Management

```rust
use trustformers_tokenizers::SpecialTokenManager;

// Advanced special token handling
let manager = SpecialTokenManager::new()
    .add_token("[MASK]", SpecialTokenType::Mask)
    .add_template("user", "<|user|>{content}<|endoftext|>")
    .add_placeholder("name", PlaceholderType::String);

let formatted = manager.format_conversation(&messages)?;
```

### Visualization

```rust
use trustformers_tokenizers::TokenVisualizer;

// Generate token visualizations
let visualizer = TokenVisualizer::new();
let html = visualizer.visualize_tokens("Hello world", &tokens, VisualizationConfig {
    show_ids: true,
    color_by_frequency: true,
    include_positions: true,
})?;

// Save visualization
std::fs::write("tokenization.html", html)?;
```

### Testing Infrastructure

```rust
use trustformers_tokenizers::TestRunner;

// Comprehensive tokenizer testing
let test_runner = TestRunner::new();
let results = test_runner.run_comprehensive_tests(&tokenizer, TestConfig {
    include_fuzzing: true,
    include_edge_cases: true,
    include_unicode_tests: true,
    benchmark_performance: true,
})?;

println!("Test results: {}", results.summary());
```

### Debugging Tools

```rust
use trustformers_tokenizers::TokenizationDebugger;

// Debug tokenization behavior
let debugger = TokenizationDebugger::new();
let debug_result = debugger.debug_tokenization(&tokenizer, "problematic text")?;

println!("Tokenization steps: {:#?}", debug_result.steps);
println!("Issues detected: {:?}", debug_result.issues);
```

## Error Handling

All functions that can fail return `Result<T, E>` where `E` implements the standard error traits. Common error types include:

- `TokenizerError` - General tokenization errors
- `VocabularyError` - Vocabulary-related errors  
- `SerializationError` - Serialization/deserialization errors
- `IoError` - File I/O errors
- `ConfigError` - Configuration errors

```rust
use trustformers_tokenizers::{TokenizerError, Result};

fn example() -> Result<()> {
    let tokenizer = TokenizerImpl::from_file("tokenizer.json")
        .map_err(|e| TokenizerError::LoadError(e.to_string()))?;
    
    let encoded = tokenizer.encode("Hello world")?;
    
    Ok(())
}
```

## Feature Flags

The library uses feature flags to enable optional functionality:

```toml
[dependencies]
trustformers-tokenizers = { version = "0.1.0", features = [
    "python",      # Python bindings
    "pytorch",     # PyTorch integration  
    "tensorflow",  # TensorFlow integration
    "jax",         # JAX integration
    "onnx",        # ONNX export
    "gpu",         # GPU acceleration
] }
```

## Thread Safety

Most tokenizers are `Send + Sync` and can be safely shared across threads. For concurrent access, consider using `Arc<Tokenizer>`:

```rust
use std::sync::Arc;

let tokenizer = Arc::new(TokenizerImpl::from_pretrained("bert-base-uncased")?);

// Share across threads
let tokenizer_clone = tokenizer.clone();
let handle = std::thread::spawn(move || {
    tokenizer_clone.encode("Hello from thread")
});
```

For high-concurrency scenarios, use dedicated parallel processing utilities like `ParallelTokenizer` or `AsyncTokenizer`.

---

For more examples and detailed usage patterns, see the [examples directory](../examples/) and [migration guides](./migration/).