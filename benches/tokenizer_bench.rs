//! Benchmarks for tokenizer performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use trustformers_tokenizers::{
    Tokenizer, TokenizerConfig,
    BPETokenizer, WordPieceTokenizer, SentencePieceTokenizer,
    Normalizer, NormalizerType,
};

fn prepare_test_texts() -> Vec<(&'static str, &'static str)> {
    vec![
        ("short", "Hello, world!"),
        ("medium", "The quick brown fox jumps over the lazy dog. This is a medium length sentence for tokenization benchmarking."),
        ("long", "Artificial intelligence and machine learning have revolutionized the way we process and understand data. These technologies enable computers to learn from experience, adapt to new inputs, and perform human-like tasks. Deep learning, a subset of machine learning, uses neural networks with multiple layers to progressively extract higher-level features from raw input. This approach has led to breakthroughs in computer vision, natural language processing, and many other fields."),
        ("code", "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\n# Example usage\nfor i in range(10):\n    print(f\"fibonacci({i}) = {fibonacci(i)}\")"),
        ("multilingual", "Hello world! Bonjour le monde! Hola mundo! Привет мир! こんにちは世界! 你好世界！ مرحبا بالعالم!"),
    ]
}

fn bpe_tokenizer_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("bpe_tokenizer");

    // Load or create BPE tokenizer
    let vocab_size = 50000;
    let tokenizer = BPETokenizer::new(
        "path/to/vocab.json",
        "path/to/merges.txt",
        vocab_size,
    ).unwrap_or_else(|_| {
        // Fallback: create a simple tokenizer for benchmarking
        BPETokenizer::from_pretrained("gpt2").expect("Failed to load GPT2 tokenizer")
    });

    for (name, text) in prepare_test_texts() {
        let text_len = text.len();

        group.throughput(Throughput::Bytes(text_len as u64));

        group.bench_with_input(
            BenchmarkId::new("encode", name),
            text,
            |b, text| {
                b.iter(|| {
                    let tokens = tokenizer.encode(text, true, false);
                    black_box(tokens)
                })
            },
        );

        // Benchmark decoding
        let tokens = tokenizer.encode(text, true, false).expect("Failed to encode");

        group.bench_with_input(
            BenchmarkId::new("decode", name),
            &tokens,
            |b, tokens| {
                b.iter(|| {
                    let decoded = tokenizer.decode(tokens, true);
                    black_box(decoded)
                })
            },
        );
    }

    group.finish();
}

fn wordpiece_tokenizer_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("wordpiece_tokenizer");

    let tokenizer = WordPieceTokenizer::new(
        "path/to/vocab.txt",
        true,  // do_lower_case
        "[UNK]",
        100,   // max_input_chars_per_word
    ).unwrap_or_else(|_| {
        WordPieceTokenizer::from_pretrained("bert-base-uncased").expect("Failed to load BERT tokenizer")
    });

    for (name, text) in prepare_test_texts() {
        let text_len = text.len();

        group.throughput(Throughput::Bytes(text_len as u64));

        group.bench_with_input(
            BenchmarkId::new("encode", name),
            text,
            |b, text| {
                b.iter(|| {
                    let tokens = tokenizer.encode(text, true, false);
                    black_box(tokens)
                })
            },
        );
    }

    group.finish();
}

fn sentencepiece_tokenizer_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("sentencepiece_tokenizer");

    let tokenizer = SentencePieceTokenizer::new("path/to/spiece.model")
        .unwrap_or_else(|_| {
            SentencePieceTokenizer::from_pretrained("t5-base").expect("Failed to load T5 tokenizer")
        });

    for (name, text) in prepare_test_texts() {
        let text_len = text.len();

        group.throughput(Throughput::Bytes(text_len as u64));

        group.bench_with_input(
            BenchmarkId::new("encode", name),
            text,
            |b, text| {
                b.iter(|| {
                    let tokens = tokenizer.encode(text, true, false);
                    black_box(tokens)
                })
            },
        );
    }

    group.finish();
}

fn batch_tokenization_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_tokenization");

    let tokenizer = BPETokenizer::from_pretrained("gpt2").expect("Failed to load tokenizer");

    // Prepare batches of different sizes
    let base_texts = prepare_test_texts();
    let batch_sizes = vec![1, 8, 16, 32, 64];

    for batch_size in batch_sizes {
        let texts: Vec<String> = base_texts
            .iter()
            .cycle()
            .take(batch_size)
            .map(|(_, text)| text.to_string())
            .collect();

        let total_bytes: usize = texts.iter().map(|t| t.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_encode", batch_size),
            &texts,
            |b, texts| {
                b.iter(|| {
                    let batch_tokens = tokenizer.encode_batch(texts, true, false);
                    black_box(batch_tokens)
                })
            },
        );
    }

    group.finish();
}

fn normalization_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_normalization");

    let normalizers = vec![
        ("nfc", NormalizerType::NFC),
        ("nfkc", NormalizerType::NFKC),
        ("lowercase", NormalizerType::Lowercase),
        ("strip_accents", NormalizerType::StripAccents),
    ];

    let test_texts = vec![
        ("ascii", "Hello World! This is a TEST."),
        ("unicode", "Héllo Wörld! Thîs ís à TËST."),
        ("mixed", "Hello 世界! これは テスト です。"),
    ];

    for (norm_name, norm_type) in normalizers {
        let normalizer = Normalizer::new(norm_type);

        for (text_name, text) in &test_texts {
            let text_len = text.len();

            group.throughput(Throughput::Bytes(text_len as u64));

            group.bench_with_input(
                BenchmarkId::new(norm_name, text_name),
                text,
                |b, text| {
                    b.iter(|| {
                        let normalized = normalizer.normalize(text);
                        black_box(normalized)
                    })
                },
            );
        }
    }

    group.finish();
}

fn special_tokens_handling_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_tokens");

    let tokenizer = BPETokenizer::from_pretrained("gpt2").expect("Failed to load tokenizer");

    let texts_with_special = vec![
        ("no_special", "This is a normal text without any special tokens."),
        ("with_mask", "This is a text with [MASK] tokens that need special handling."),
        ("with_sep", "First sentence. [SEP] Second sentence. [SEP] Third sentence."),
        ("mixed", "[CLS] This is a [MASK] example with multiple [PAD] special tokens [SEP]"),
    ];

    for (name, text) in texts_with_special {
        let text_len = text.len();

        group.throughput(Throughput::Bytes(text_len as u64));

        // With special tokens
        group.bench_with_input(
            BenchmarkId::new("with_special", name),
            text,
            |b, text| {
                b.iter(|| {
                    let tokens = tokenizer.encode(text, true, true);
                    black_box(tokens)
                })
            },
        );

        // Without special tokens
        group.bench_with_input(
            BenchmarkId::new("no_special", name),
            text,
            |b, text| {
                b.iter(|| {
                    let tokens = tokenizer.encode(text, true, false);
                    black_box(tokens)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bpe_tokenizer_benchmarks,
    wordpiece_tokenizer_benchmarks,
    sentencepiece_tokenizer_benchmarks,
    batch_tokenization_benchmarks,
    normalization_benchmarks,
    special_tokens_handling_benchmarks
);
criterion_main!(benches);