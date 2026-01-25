//! Performance benchmarks for trustformers-tokenizers
//!
//! This benchmark suite tests the performance of different tokenizers
//! to identify potential bottlenecks and optimization opportunities.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;
use std::hint::black_box;
use trustformers_core::traits::Tokenizer;
use trustformers_tokenizers::{
    bpe::BPETokenizer, char::CharTokenizer, wordpiece::WordPieceTokenizer,
};

fn create_sample_texts() -> Vec<String> {
    vec![
        "Hello world!".to_string(),
        "This is a test sentence with multiple words.".to_string(),
        "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet.".to_string(),
        "Artificial intelligence and machine learning are transforming the way we process and understand natural language.".to_string(),
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.".to_string(),
    ]
}

fn benchmark_tokenizers(c: &mut Criterion) {
    let texts = create_sample_texts();
    let short_text = "Hello world!";
    let medium_text = "This is a medium-length text with multiple sentences. It should give us a good baseline for performance testing.";
    let long_text = "This is a much longer text that contains many sentences and words, designed to test tokenization performance on longer inputs. It includes various punctuation marks, numbers, and different types of content that tokenizers encounter in real-world scenarios. The goal is to identify performance characteristics across different text lengths and content types.";

    // Create tokenizers for benchmarking
    let mut char_vocab = HashMap::new();
    char_vocab.insert("a".to_string(), 1);
    char_vocab.insert("b".to_string(), 2);
    char_vocab.insert("c".to_string(), 3);
    char_vocab.insert("d".to_string(), 4);
    char_vocab.insert("e".to_string(), 5);
    char_vocab.insert("f".to_string(), 6);
    char_vocab.insert("g".to_string(), 7);
    char_vocab.insert("h".to_string(), 8);
    char_vocab.insert("i".to_string(), 9);
    char_vocab.insert("j".to_string(), 10);
    char_vocab.insert("k".to_string(), 11);
    char_vocab.insert("l".to_string(), 12);
    char_vocab.insert("m".to_string(), 13);
    char_vocab.insert("n".to_string(), 14);
    char_vocab.insert("o".to_string(), 15);
    char_vocab.insert("p".to_string(), 16);
    char_vocab.insert("q".to_string(), 17);
    char_vocab.insert("r".to_string(), 18);
    char_vocab.insert("s".to_string(), 19);
    char_vocab.insert("t".to_string(), 20);
    char_vocab.insert("u".to_string(), 21);
    char_vocab.insert("v".to_string(), 22);
    char_vocab.insert("w".to_string(), 23);
    char_vocab.insert("x".to_string(), 24);
    char_vocab.insert("y".to_string(), 25);
    char_vocab.insert("z".to_string(), 26);
    char_vocab.insert(" ".to_string(), 27);
    char_vocab.insert(".".to_string(), 28);
    char_vocab.insert("!".to_string(), 29);
    char_vocab.insert("?".to_string(), 30);
    let char_tokenizer = CharTokenizer::new(char_vocab);

    let mut bpe_vocab = HashMap::new();
    bpe_vocab.insert("hello".to_string(), 1);
    bpe_vocab.insert("world".to_string(), 2);
    bpe_vocab.insert("test".to_string(), 3);
    bpe_vocab.insert("sentence".to_string(), 4);
    bpe_vocab.insert("the".to_string(), 5);
    bpe_vocab.insert("quick".to_string(), 6);
    bpe_vocab.insert("brown".to_string(), 7);
    bpe_vocab.insert("fox".to_string(), 8);

    let bpe_tokenizer = BPETokenizer::new(bpe_vocab, Vec::new());

    let mut wp_vocab = HashMap::new();
    wp_vocab.insert("[UNK]".to_string(), 0);
    wp_vocab.insert("hello".to_string(), 1);
    wp_vocab.insert("world".to_string(), 2);
    wp_vocab.insert("test".to_string(), 3);
    wp_vocab.insert("##ing".to_string(), 4);
    wp_vocab.insert("the".to_string(), 5);

    let wp_tokenizer = WordPieceTokenizer::new(wp_vocab, false);

    // Benchmark different tokenizers on short text
    let mut group = c.benchmark_group("tokenizer_comparison_short");

    group.bench_with_input(BenchmarkId::new("char", "short"), &short_text, |b, text| {
        b.iter(|| char_tokenizer.encode(black_box(text)))
    });

    group.bench_with_input(BenchmarkId::new("bpe", "short"), &short_text, |b, text| {
        b.iter(|| bpe_tokenizer.encode(black_box(text)))
    });

    group.bench_with_input(
        BenchmarkId::new("wordpiece", "short"),
        &short_text,
        |b, text| b.iter(|| wp_tokenizer.encode(black_box(text))),
    );

    group.finish();

    // Benchmark different tokenizers on medium text
    let mut group = c.benchmark_group("tokenizer_comparison_medium");

    group.bench_with_input(
        BenchmarkId::new("char", "medium"),
        &medium_text,
        |b, text| b.iter(|| char_tokenizer.encode(black_box(text))),
    );

    group.bench_with_input(
        BenchmarkId::new("bpe", "medium"),
        &medium_text,
        |b, text| b.iter(|| bpe_tokenizer.encode(black_box(text))),
    );

    group.bench_with_input(
        BenchmarkId::new("wordpiece", "medium"),
        &medium_text,
        |b, text| b.iter(|| wp_tokenizer.encode(black_box(text))),
    );

    group.finish();

    // Benchmark different tokenizers on long text
    let mut group = c.benchmark_group("tokenizer_comparison_long");

    group.bench_with_input(BenchmarkId::new("char", "long"), &long_text, |b, text| {
        b.iter(|| char_tokenizer.encode(black_box(text)))
    });

    group.bench_with_input(BenchmarkId::new("bpe", "long"), &long_text, |b, text| {
        b.iter(|| bpe_tokenizer.encode(black_box(text)))
    });

    group.bench_with_input(
        BenchmarkId::new("wordpiece", "long"),
        &long_text,
        |b, text| b.iter(|| wp_tokenizer.encode(black_box(text))),
    );

    group.finish();

    // Batch tokenization benchmarks
    let mut group = c.benchmark_group("batch_tokenization");

    group.bench_with_input(BenchmarkId::new("char", "batch"), &texts, |b, texts| {
        b.iter(|| {
            for text in texts {
                char_tokenizer.encode(black_box(text)).unwrap();
            }
        })
    });

    group.bench_with_input(BenchmarkId::new("bpe", "batch"), &texts, |b, texts| {
        b.iter(|| {
            for text in texts {
                bpe_tokenizer.encode(black_box(text)).unwrap();
            }
        })
    });

    group.finish();
}

fn benchmark_text_lengths(c: &mut Criterion) {
    let mut char_vocab = HashMap::new();
    char_vocab.insert("a".to_string(), 1);
    char_vocab.insert("b".to_string(), 2);
    char_vocab.insert("c".to_string(), 3);
    char_vocab.insert("d".to_string(), 4);
    char_vocab.insert("e".to_string(), 5);
    char_vocab.insert("f".to_string(), 6);
    char_vocab.insert("g".to_string(), 7);
    char_vocab.insert("h".to_string(), 8);
    char_vocab.insert("i".to_string(), 9);
    char_vocab.insert("j".to_string(), 10);
    char_vocab.insert("k".to_string(), 11);
    char_vocab.insert("l".to_string(), 12);
    char_vocab.insert("m".to_string(), 13);
    char_vocab.insert("n".to_string(), 14);
    char_vocab.insert("o".to_string(), 15);
    char_vocab.insert("p".to_string(), 16);
    char_vocab.insert("q".to_string(), 17);
    char_vocab.insert("r".to_string(), 18);
    char_vocab.insert("s".to_string(), 19);
    char_vocab.insert("t".to_string(), 20);
    char_vocab.insert("u".to_string(), 21);
    char_vocab.insert("v".to_string(), 22);
    char_vocab.insert("w".to_string(), 23);
    char_vocab.insert("x".to_string(), 24);
    char_vocab.insert("y".to_string(), 25);
    char_vocab.insert("z".to_string(), 26);
    char_vocab.insert(" ".to_string(), 27);
    char_vocab.insert(".".to_string(), 28);
    char_vocab.insert("!".to_string(), 29);
    char_vocab.insert("?".to_string(), 30);
    let char_tokenizer = CharTokenizer::new(char_vocab);

    let text_lengths = [10, 50, 100, 500, 1000, 5000];
    let base_text = "The quick brown fox jumps over the lazy dog. ";

    let mut group = c.benchmark_group("text_length_scaling");

    for &length in &text_lengths {
        let text = base_text.repeat(length / base_text.len() + 1);
        let text = &text[..std::cmp::min(text.len(), length)];

        group.bench_with_input(
            BenchmarkId::new("char_tokenizer", length),
            &text,
            |b, text| b.iter(|| char_tokenizer.encode(black_box(text))),
        );
    }

    group.finish();
}

fn benchmark_memory_usage(c: &mut Criterion) {
    let mut char_vocab = HashMap::new();
    char_vocab.insert("a".to_string(), 1);
    char_vocab.insert("b".to_string(), 2);
    char_vocab.insert("c".to_string(), 3);
    char_vocab.insert("d".to_string(), 4);
    char_vocab.insert("e".to_string(), 5);
    char_vocab.insert("f".to_string(), 6);
    char_vocab.insert("g".to_string(), 7);
    char_vocab.insert("h".to_string(), 8);
    char_vocab.insert("i".to_string(), 9);
    char_vocab.insert("j".to_string(), 10);
    char_vocab.insert("k".to_string(), 11);
    char_vocab.insert("l".to_string(), 12);
    char_vocab.insert("m".to_string(), 13);
    char_vocab.insert("n".to_string(), 14);
    char_vocab.insert("o".to_string(), 15);
    char_vocab.insert("p".to_string(), 16);
    char_vocab.insert("q".to_string(), 17);
    char_vocab.insert("r".to_string(), 18);
    char_vocab.insert("s".to_string(), 19);
    char_vocab.insert("t".to_string(), 20);
    char_vocab.insert("u".to_string(), 21);
    char_vocab.insert("v".to_string(), 22);
    char_vocab.insert("w".to_string(), 23);
    char_vocab.insert("x".to_string(), 24);
    char_vocab.insert("y".to_string(), 25);
    char_vocab.insert("z".to_string(), 26);
    char_vocab.insert(" ".to_string(), 27);
    char_vocab.insert(".".to_string(), 28);
    char_vocab.insert("!".to_string(), 29);
    char_vocab.insert("?".to_string(), 30);
    let char_tokenizer = CharTokenizer::new(char_vocab);
    let long_text = "word ".repeat(10000);

    c.bench_function("memory_intensive_tokenization", |b| {
        b.iter(|| char_tokenizer.encode(black_box(&long_text)))
    });
}

criterion_group!(
    benches,
    benchmark_tokenizers,
    benchmark_text_lengths,
    benchmark_memory_usage
);
criterion_main!(benches);
