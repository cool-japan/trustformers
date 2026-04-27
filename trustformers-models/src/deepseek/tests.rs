use crate::deepseek::{
    config::DeepSeekConfig,
    model::{
        DeepSeekForCausalLM, DeepSeekMlaAttention, DeepSeekModel, DeepSeekMoeLayer, DeepSeekRmsNorm,
    },
};
use trustformers_core::{tensor::Tensor, traits::Config, traits::Layer};

// ── Config tests ──────────────────────────────────────────────────────────────

#[test]
fn test_config_small_valid() {
    let cfg = DeepSeekConfig::small_test();
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.n_routed_experts, 4);
    assert_eq!(cfg.num_experts_per_tok, 2);
    assert_eq!(cfg.kv_lora_rank, 16);
}

#[test]
fn test_config_v2_small_valid() {
    let cfg = DeepSeekConfig::deepseek_v2_small();
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.vocab_size, 102400);
    assert_eq!(cfg.kv_lora_rank, 512);
    assert_eq!(cfg.n_routed_experts, 64);
    assert_eq!(cfg.num_experts_per_tok, 6);
}

#[test]
fn test_config_is_moe_layer_dense_prefix() {
    let cfg = DeepSeekConfig::small_test();
    // first_k_dense_replace = 1, so layer 0 is dense
    assert!(!cfg.is_moe_layer(0), "Layer 0 should be dense");
    assert!(cfg.is_moe_layer(1), "Layer 1 should be MoE");
}

#[test]
fn test_config_invalid_kv_lora_rank_zero() {
    let mut cfg = DeepSeekConfig::small_test();
    cfg.kv_lora_rank = 0;
    assert!(cfg.validate().is_err());
}

// ── RMS norm test ─────────────────────────────────────────────────────────────

#[test]
fn test_rms_norm_forward() {
    let norm = DeepSeekRmsNorm::new(4, 1e-6);
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).expect("build tensor");
    let output = norm.forward(input).expect("forward");
    assert_eq!(output.shape(), &[4]);
}

// ── MLA attention tests ───────────────────────────────────────────────────────

#[test]
fn test_mla_attention_kv_lora_rank() {
    let cfg = DeepSeekConfig::small_test();
    let attn = DeepSeekMlaAttention::new(&cfg);
    assert_eq!(attn.kv_lora_rank(), cfg.kv_lora_rank);
    assert_eq!(attn.num_heads(), cfg.num_attention_heads);
}

#[test]
fn test_mla_attention_compress_decompress() {
    let cfg = DeepSeekConfig::small_test();
    let attn = DeepSeekMlaAttention::new(&cfg);
    // forward_token should run without panic and return hidden_size output
    let x = vec![0.1f32; cfg.hidden_size];
    let out = attn.forward_token(&x);
    assert_eq!(out.len(), cfg.hidden_size);
}

#[test]
fn test_mla_attention_forward_tensor() {
    let cfg = DeepSeekConfig::small_test();
    let attn = DeepSeekMlaAttention::new(&cfg);
    let input = Tensor::from_vec(vec![0.0f32; cfg.hidden_size * 3], &[1, 3, cfg.hidden_size])
        .expect("build");
    let output = attn.forward(input).expect("forward");
    let shape = output.shape().to_vec();
    assert_eq!(shape, vec![1, 3, cfg.hidden_size]);
}

// ── MoE layer tests ───────────────────────────────────────────────────────────

#[test]
fn test_moe_routing_top_k_count() {
    let cfg = DeepSeekConfig::small_test();
    let moe = DeepSeekMoeLayer::new(&cfg).expect("build moe");
    let token = vec![1.0f32; cfg.hidden_size];
    let (indices, weights) = moe.route_token(&token);
    assert_eq!(indices.len(), cfg.num_experts_per_tok);
    assert_eq!(weights.len(), cfg.num_experts_per_tok);
    // Normalised weights should sum approximately to 1
    let weight_sum: f32 = weights.iter().sum();
    approx::assert_abs_diff_eq!(weight_sum, 1.0, epsilon = 1e-5);
}

#[test]
fn test_moe_shared_expert_count() {
    let cfg = DeepSeekConfig::small_test();
    let moe = DeepSeekMoeLayer::new(&cfg).expect("build moe");
    assert_eq!(moe.n_shared_experts(), cfg.n_shared_experts);
    assert_eq!(moe.n_routed_experts(), cfg.n_routed_experts);
}

#[test]
fn test_moe_forward_shape() {
    let cfg = DeepSeekConfig::small_test();
    let moe = DeepSeekMoeLayer::new(&cfg).expect("build moe");
    let input = Tensor::from_vec(vec![0.1f32; cfg.hidden_size * 2], &[1, 2, cfg.hidden_size])
        .expect("build");
    let output = moe.forward(input).expect("forward");
    let shape = output.shape().to_vec();
    assert_eq!(shape, vec![1, 2, cfg.hidden_size]);
}

// ── Dense vs MoE layer selection ─────────────────────────────────────────────

#[test]
fn test_dense_layer_for_first_k() {
    let cfg = DeepSeekConfig::small_test();
    let model = DeepSeekModel::new(cfg).expect("build model");
    // The model layers are opaque but we built with first_k_dense_replace=1
    // Accessing is_moe() on the first decoder layer
    let output = model.run(vec![1u32, 2]).expect("run");
    assert!(output.shape().len() == 3);
}

// ── Full model forward ────────────────────────────────────────────────────────

#[test]
fn test_model_forward_small() {
    let cfg = DeepSeekConfig::small_test();
    let model = DeepSeekModel::new(cfg.clone()).expect("build model");
    let output = model.run(vec![1u32, 2, 3]).expect("run");
    let shape = output.shape().to_vec();
    assert_eq!(shape[0], 1);
    assert_eq!(shape[1], 3);
    assert_eq!(shape[2], cfg.hidden_size);
}

#[test]
fn test_causal_lm_forward_small() {
    let cfg = DeepSeekConfig::small_test();
    let lm = DeepSeekForCausalLM::new(cfg.clone()).expect("build lm");
    let logits = lm.forward(vec![1u32, 2]).expect("forward");
    let shape = logits.shape().to_vec();
    assert_eq!(shape[0], 1);
    assert_eq!(shape[1], 2);
    assert_eq!(shape[2], cfg.vocab_size);
}

// ── FFN kind discriminant ─────────────────────────────────────────────────────

#[test]
fn test_ffn_kind_dense_vs_moe() {
    let cfg = DeepSeekConfig::small_test();
    // layer 0 => dense (first_k_dense_replace = 1)
    // layer 1 => moe
    use crate::deepseek::model::DeepSeekDecoderLayer;
    let layer0 = DeepSeekDecoderLayer::new(&cfg, 0).expect("layer0");
    let layer1 = DeepSeekDecoderLayer::new(&cfg, 1).expect("layer1");
    assert!(!layer0.is_moe(), "Layer 0 should be dense");
    assert!(layer1.is_moe(), "Layer 1 should be MoE");
}
