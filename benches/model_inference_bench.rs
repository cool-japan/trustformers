//! Benchmarks for model inference performance

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::hint::black_box;
use trustformers::prelude::*;
use trustformers_models::{
    bert::{BertConfig, BertModel},
    gpt2::{GPT2Config, GPT2Model},
    t5::{T5Config, T5Model},
    llama::{LlamaConfig, LlamaModel},
};
use trustformers_core::tensor::Tensor;

fn bert_inference_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("bert_inference");

    // Different model sizes
    let configs = vec![
        ("bert-base", BertConfig {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            ..Default::default()
        }),
        ("bert-large", BertConfig {
            vocab_size: 30522,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            ..Default::default()
        }),
    ];

    for (name, config) in configs {
        let model = BertModel::new(config.clone()).expect("Failed to create model");

        // Different sequence lengths
        for seq_len in [128, 256, 512].iter() {
            let input_ids = vec![101; *seq_len]; // [CLS] token repeated
            let attention_mask = vec![1; *seq_len];

            group.throughput(Throughput::Elements(*seq_len as u64));

            group.bench_with_input(
                BenchmarkId::new(name, seq_len),
                &(input_ids, attention_mask),
                |b, (input_ids, attention_mask)| {
                    b.iter(|| {
                        let output = model.forward(
                            &Tensor::from_vec(input_ids.clone(), &[1, *seq_len]),
                            Some(&Tensor::from_vec(attention_mask.clone(), &[1, *seq_len])),
                            None,
                        );
                        black_box(output)
                    })
                },
            );
        }
    }

    group.finish();
}

fn gpt2_inference_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpt2_inference");

    let configs = vec![
        ("gpt2", GPT2Config {
            vocab_size: 50257,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            ..Default::default()
        }),
        ("gpt2-medium", GPT2Config {
            vocab_size: 50257,
            n_embd: 1024,
            n_layer: 24,
            n_head: 16,
            ..Default::default()
        }),
    ];

    for (name, config) in configs {
        let model = GPT2Model::new(config.clone()).expect("Failed to create model");

        for seq_len in [128, 256, 512].iter() {
            let input_ids = vec![50256; *seq_len]; // GPT2 token

            group.throughput(Throughput::Elements(*seq_len as u64));

            group.bench_with_input(
                BenchmarkId::new(name, seq_len),
                &input_ids,
                |b, input_ids| {
                    b.iter(|| {
                        let output = model.forward(
                            &Tensor::from_vec(input_ids.clone(), &[1, *seq_len]),
                            None,
                        );
                        black_box(output)
                    })
                },
            );
        }
    }

    group.finish();
}

fn llama_inference_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("llama_inference");

    let configs = vec![
        ("llama-7b", LlamaConfig {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            ..Default::default()
        }),
        ("llama-13b", LlamaConfig {
            vocab_size: 32000,
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: 40,
            ..Default::default()
        }),
    ];

    for (name, config) in configs {
        let model = LlamaModel::new(config.clone()).expect("Failed to create model");

        // Shorter sequences for large models
        for seq_len in [64, 128, 256].iter() {
            let input_ids = vec![1; *seq_len];

            group.throughput(Throughput::Elements(*seq_len as u64));

            group.bench_with_input(
                BenchmarkId::new(name, seq_len),
                &input_ids,
                |b, input_ids| {
                    b.iter(|| {
                        let output = model.forward(
                            &Tensor::from_vec(input_ids.clone(), &[1, *seq_len]),
                            None,
                        );
                        black_box(output)
                    })
                },
            );
        }
    }

    group.finish();
}

fn generation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("generation");

    let config = GPT2Config {
        vocab_size: 50257,
        n_embd: 768,
        n_layer: 12,
        n_head: 12,
        ..Default::default()
    };

    let model = GPT2Model::new(config).expect("Failed to create model");

    for max_length in [20, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("gpt2_generate", max_length),
            max_length,
            |b, &max_length| {
                b.iter(|| {
                    let input_ids = vec![50256]; // Start token
                    let generated = model.generate(
                        &Tensor::from_vec(input_ids, &[1, 1]),
                        max_length,
                        1.0, // temperature
                        0.9, // top_p
                        None, // top_k
                    );
                    black_box(generated)
                })
            },
        );
    }

    group.finish();
}

fn batch_inference_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_inference");

    let config = BertConfig {
        vocab_size: 30522,
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        ..Default::default()
    };

    let model = BertModel::new(config).expect("Failed to create model");
    let seq_len = 128;

    for batch_size in [1, 8, 16, 32, 64].iter() {
        let input_ids = vec![101; seq_len * batch_size];
        let attention_mask = vec![1; seq_len * batch_size];

        group.throughput(Throughput::Elements((seq_len * batch_size) as u64));

        group.bench_with_input(
            BenchmarkId::new("bert_batch", batch_size),
            &(input_ids, attention_mask, *batch_size),
            |b, (input_ids, attention_mask, batch_size)| {
                b.iter(|| {
                    let output = model.forward(
                        &Tensor::from_vec(input_ids.clone(), &[*batch_size, seq_len]),
                        Some(&Tensor::from_vec(attention_mask.clone(), &[*batch_size, seq_len])),
                        None,
                    );
                    black_box(output)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bert_inference_benchmarks,
    gpt2_inference_benchmarks,
    llama_inference_benchmarks,
    generation_benchmarks,
    batch_inference_benchmarks
);
criterion_main!(benches);