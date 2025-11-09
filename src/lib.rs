pub mod attention;
pub mod layers;
pub mod models;
pub mod utils;
pub mod bert;
pub mod gpt;
pub mod t5;
pub mod visualization;
pub mod benchmarks;
pub mod codegen;

// Re-export commonly used items
pub use attention::{MultiHeadAttention, SelfAttention, CrossAttention};
pub use layers::{LayerNorm, RMSNorm, FeedForward, ActivationType, GLU, SwiGLU};
pub use models::{
    Transformer, TransformerConfig, TransformerEncoder, TransformerDecoder,
    TransformerLayer, LayerNormTrait, FeedForwardTrait,
};
pub use utils::{
    PositionalEncoding, PositionalEncodingType, ALiBi,
    create_padding_mask, create_causal_mask, create_attention_mask,
    scaled_dot_product_attention_with_alibi,
};

// BERT exports
pub use bert::{
    BertConfig, BertModel, BertEmbeddings, BertPooler, BertOutput,
    BertForMaskedLM, BertForNextSentencePrediction, BertForPreTraining,
    BertLMPredictionHead, BertOnlyNSPHead, BertPreTrainingHeads,
    MaskedLMOutput, NextSentencePredictionOutput, PreTrainingOutput,
    create_masked_lm_labels,
};

// GPT exports
pub use gpt::{
    GPTConfig, GPTModel, GPTBlock, GPTOutput,
    GPTForCausalLM, CausalLMOutput, GPTTrainer, LRScheduler,
    gpt2_config, GPT2ModelSize,
};

// T5 exports
pub use t5::{
    T5Config, T5Model, T5Stack, T5Block, T5LayerWrapper, T5Layer,
    T5Attention, T5DenseReluDense, T5DenseGatedGeluDense,
    T5ModelOutput, T5ForConditionalGeneration, T5ConditionalGenerationOutput,
    t5_small_config, t5_base_config, t5_large_config,
};

// Visualization exports
pub use visualization::{
    AttentionVisualizer, AttentionStatistics, LayerStatistics, HeadStatistics,
    AttentionPattern, PatternType, TokenAttentionAnalyzer, AttentionHooks,
    create_attention_hooks,
};

// Benchmark exports
pub use benchmarks::{
    BenchmarkSuite, BenchmarkResult, BenchmarkConfig,
    run_quick_benchmark, run_comprehensive_benchmark,
};

// Code generation exports
pub use codegen::{
    CodeGenerator, ModelTemplate, LayerConfig, ModelConfig,
    create_model_from_config,
};

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor, nn};

    #[test]
    fn test_multi_head_attention() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        let d_model = 512i64;
        let num_heads = 8i64;
        let batch_size = 2i64;
        let seq_len = 10i64;

        let attention = MultiHeadAttention::new(&vs.root(), d_model, num_heads, 0.1);
        let input = Tensor::randn(&[batch_size, seq_len, d_model]).unwrap();

        let output = attention.forward(&input, &input, &input, None);

        assert_eq!(output.size(), vec![batch_size, seq_len, d_model]);
    }

    #[test]
    fn test_layer_norm() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        let d_model = 512i64;
        let batch_size = 2i64;
        let seq_len = 10i64;

        let layer_norm = LayerNorm::new(&vs.root(), vec![d_model], 1e-5, true);
        let input = Tensor::randn(&[batch_size, seq_len, d_model]).unwrap();

        let output = layer_norm.forward(&input);

        assert_eq!(output.size(), input.size());
    }

    #[test]
    fn test_feed_forward() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        let d_model = 512i64;
        let d_ff = 2048i64;
        let batch_size = 2i64;
        let seq_len = 10i64;

        let feed_forward = FeedForward::new(&vs.root(), d_model, d_ff, 0.1, ActivationType::GELU);
        let input = Tensor::randn(&[batch_size, seq_len, d_model]).unwrap();

        let output = feed_forward.forward(&input);

        assert_eq!(output.size(), input.size());
    }

    #[test]
    fn test_positional_encoding() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        let d_model = 512i64;
        let max_seq_len = 1024i64;
        let batch_size = 2i64;
        let seq_len = 10i64;

        let pos_enc = PositionalEncoding::new(
            &vs.root(),
            PositionalEncodingType::Sinusoidal,
            d_model,
            max_seq_len,
            0.1,
        );

        let input = Tensor::randn(&[batch_size, seq_len, d_model]).unwrap();
        let output = pos_enc.forward(&input);

        assert_eq!(output.size(), input.size());
    }

    #[test]
    fn test_transformer_model() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        let config = TransformerConfig {
            vocab_size: 1000,
            d_model: 256,
            num_heads: 4,
            num_layers: 2,
            d_ff: 1024,
            max_seq_len: 128,
            dropout: 0.1,
            ..Default::default()
        };

        let transformer = Transformer::new(&vs.root(), config.clone());
        let batch_size = 2i64;
        let seq_len = 10i64;

        let input_ids = Tensor::randint(config.vocab_size, &[batch_size, seq_len], (Kind::Int64, device));
        let output = transformer.forward(&input_ids, None);

        assert_eq!(output.size(), vec![batch_size, seq_len, config.vocab_size]);
    }

    #[test]
    fn test_bert_model() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        let config = BertConfig {
            vocab_size: 1000,
            d_model: 256,
            num_heads: 4,
            num_layers: 2,
            d_ff: 1024,
            max_seq_len: 128,
            ..Default::default()
        };

        let bert = BertModel::new(&vs.root(), config.clone(), false);
        let batch_size = 2i64;
        let seq_len = 10i64;

        let input_ids = Tensor::randint(config.vocab_size, &[batch_size, seq_len], (Kind::Int64, device));
        let output = bert.forward(&input_ids, None, None);

        assert_eq!(output.last_hidden_state.size(), vec![batch_size, seq_len, config.d_model]);
    }

    #[test]
    fn test_gpt_model() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        let config = GPTConfig {
            vocab_size: 1000,
            d_model: 256,
            num_heads: 4,
            num_layers: 2,
            d_ff: 1024,
            max_seq_len: 128,
            ..Default::default()
        };

        let gpt = GPTModel::new(&vs.root(), config.clone());
        let batch_size = 2i64;
        let seq_len = 10i64;

        let input_ids = Tensor::randint(config.vocab_size, &[batch_size, seq_len], (Kind::Int64, device));
        let output = gpt.forward(&input_ids, None, None);

        assert_eq!(output.last_hidden_state.size(), vec![batch_size, seq_len, config.d_model]);
    }

    #[test]
    fn test_t5_model() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        let config = T5Config {
            vocab_size: 1000,
            d_model: 256,
            num_heads: 4,
            num_layers: 2,
            num_decoder_layers: 2,
            d_ff: 1024,
            ..Default::default()
        };

        let t5 = T5Model::new(&vs.root(), config.clone());
        let batch_size = 2i64;
        let seq_len = 10i64;

        let input_ids = Tensor::randint(config.vocab_size, &[batch_size, seq_len], (Kind::Int64, device));
        let decoder_input_ids = Tensor::randint(config.vocab_size, &[batch_size, seq_len], (Kind::Int64, device));

        let output = t5.forward(Some(&input_ids), None, Some(&decoder_input_ids), None, None);

        assert_eq!(output.last_hidden_state.size(), vec![batch_size, seq_len, config.d_model]);
    }

    #[test]
    fn test_attention_visualizer() {
        let mut visualizer = AttentionVisualizer::new();

        visualizer.add_layer("layer_0".to_string(), 4);

        let attention_weights = Tensor::rand(&[2, 10, 10], (Kind::Float, Device::Cpu));
        visualizer.record_attention(0, 0, &attention_weights);

        let stats = visualizer.compute_attention_statistics();
        assert_eq!(stats.layer_statistics.len(), 1);
        assert_eq!(stats.layer_statistics[0].head_statistics.len(), 1);
    }

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();

        assert!(!config.batch_sizes.is_empty());
        assert!(!config.sequence_lengths.is_empty());
        assert!(config.num_iterations > 0);
        assert!(config.warmup_iterations > 0);
    }
}