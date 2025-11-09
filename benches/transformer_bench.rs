use criterion::{black_box, criterion_group, criterion_main, Criterion};
use trustformers::{BertConfig, BertModel, TokenizedInput};
use trustformers::prelude::*;

fn bert_forward_pass(c: &mut Criterion) {
    let config = BertConfig {
        vocab_size: 30522,
        hidden_size: 768,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        intermediate_size: 3072,
        ..Default::default()
    };

    let model = BertModel::new(config).expect("Failed to create model");

    c.bench_function("bert_forward_pass", |b| {
        b.iter(|| {
            let input = TokenizedInput {
                input_ids: vec![101, 7592, 1010, 2088, 999, 102],
                attention_mask: vec![1, 1, 1, 1, 1, 1],
                token_type_ids: Some(vec![0, 0, 0, 0, 0, 0]),
            };

            let output = model.forward(black_box(input)).expect("Forward pass failed");
            black_box(output)
        })
    });
}

fn attention_layer_benchmark(c: &mut Criterion) {
    use trustformers::core::layers::{MultiHeadAttention, AttentionInput};
    use trustformers::core::tensor::Tensor;

    let attention = MultiHeadAttention::new(768, 12, 0.1, true)
        .expect("Failed to create attention layer");

    c.bench_function("multi_head_attention", |b| {
        b.iter(|| {
            let hidden_states = Tensor::randn(&[1, 128, 768])
                .expect("Failed to create tensor");

            let input = AttentionInput {
                hidden_states,
                attention_mask: None,
            };

            let output = attention.forward(black_box(input))
                .expect("Attention forward failed");
            black_box(output)
        })
    });
}

criterion_group!(benches, bert_forward_pass, attention_layer_benchmark);
criterion_main!(benches);