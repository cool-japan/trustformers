use crate::llama3::{
    config::LLaMA3Config,
    model::{
        LLaMA3Attention, LLaMA3ForCausalLM, LLaMA3MLP, LLaMA3Model, LLaMA3RmsNorm,
        LLaMA3RotaryEmbedding,
    },
    tasks::{format_llama3_chat, LLaMA3ChatModel},
};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use trustformers_core::{tensor::Tensor, traits::Config, traits::Layer};

// ─────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_llama3_config_small_test_valid() {
    let config = LLaMA3Config::small_test();
    assert!(config.validate().is_ok(), "small_test config must be valid");
}

#[test]
fn test_llama3_config_8b_preset() {
    let config = LLaMA3Config::llama3_8b();
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.vocab_size, 128256);
    assert_eq!(config.max_position_embeddings, 8192);
    assert!((config.rope_theta - 500000.0).abs() < 1.0);
    assert!(config.uses_gqa(), "8B must use GQA");
    assert!(config.validate().is_ok());
}

#[test]
fn test_llama3_config_70b_preset() {
    let config = LLaMA3Config::llama3_70b();
    assert_eq!(config.hidden_size, 8192);
    assert_eq!(config.num_attention_heads, 64);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.num_hidden_layers, 80);
    assert_eq!(config.num_query_groups(), 8);
    assert_eq!(config.head_dim(), 128); // 8192 / 64
    assert!(config.validate().is_ok());
}

#[test]
fn test_llama3_config_invalid_head_division() {
    let config = LLaMA3Config {
        hidden_size: 65, // not divisible by 4
        ..LLaMA3Config::small_test()
    };
    assert!(
        config.validate().is_err(),
        "hidden_size not divisible by num_attention_heads must fail"
    );
}

#[test]
fn test_llama3_config_architecture_label() {
    assert_eq!(LLaMA3Config::small_test().architecture(), "LLaMA-3");
}

// ─────────────────────────────────────────────────────────────────────────
// RMSNorm
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_llama3_rmsnorm_construction() {
    let norm = LLaMA3RmsNorm::new(64, 1e-5);
    assert!(norm.is_ok(), "RMSNorm construction must succeed");
    let norm = norm.expect("checked");
    assert_eq!(norm.parameter_count(), 64);
}

#[test]
fn test_llama3_rmsnorm_forward_shape() {
    let norm = LLaMA3RmsNorm::new(8, 1e-5).expect("construction failed");
    let input = Tensor::ones(&[8]).expect("tensor");
    let output = norm.forward(input.clone()).expect("forward failed");
    assert_eq!(output.shape(), input.shape(), "RMSNorm must preserve shape");
}

// ─────────────────────────────────────────────────────────────────────────
// RoPE
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_llama3_rope_half_dim() {
    let config = LLaMA3Config::small_test();
    let rope = LLaMA3RotaryEmbedding::new(
        config.head_dim(),
        config.max_position_embeddings,
        config.rope_theta,
    );
    assert_eq!(rope.half_dim(), config.head_dim() / 2);
}

#[test]
fn test_llama3_rope_apply_preserves_shape() {
    let rope = LLaMA3RotaryEmbedding::new(16, 64, 500000.0);
    let q = Tensor::ones(&[4, 16]).expect("tensor");
    let k = Tensor::ones(&[4, 16]).expect("tensor");
    let positions: Vec<usize> = (0..4).collect();
    let (q_out, k_out) = rope.apply_rotary_emb(&q, &k, &positions).expect("RoPE failed");
    assert_eq!(q_out.shape(), q.shape(), "q shape must be preserved");
    assert_eq!(k_out.shape(), k.shape(), "k shape must be preserved");
}

// ─────────────────────────────────────────────────────────────────────────
// GQA repeat_kv
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_llama3_repeat_kv_no_op_when_no_gqa() {
    // Build a config with num_kv == num_q → no expansion
    let config = LLaMA3Config {
        num_attention_heads: 4,
        num_key_value_heads: 4,
        ..LLaMA3Config::small_test()
    };
    let attn = LLaMA3Attention::new(&config).expect("construction failed");
    let kv = Tensor::ones(&[8]).expect("tensor");
    let expanded = attn.repeat_kv(&kv).expect("repeat_kv failed");
    assert_eq!(
        expanded.shape(),
        kv.shape(),
        "no expansion when groups == 1"
    );
}

#[test]
fn test_llama3_repeat_kv_expands_correctly() {
    let config = LLaMA3Config::small_test(); // 4 Q, 2 KV → 2× sharing
    let attn = LLaMA3Attention::new(&config).expect("construction failed");
    let head_dim = config.head_dim(); // 64/4 = 16
    let kv = Tensor::ones(&[config.num_key_value_heads * head_dim]).expect("tensor");

    let expanded = attn.repeat_kv(&kv).expect("repeat_kv failed");
    let expanded_len: usize = expanded.shape().iter().product();
    let original_len: usize = kv.shape().iter().product();
    assert_eq!(
        expanded_len,
        original_len * config.num_query_groups(),
        "expanded KV must be num_query_groups × larger"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Attention forward
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_llama3_attention_forward_shape() {
    let config = LLaMA3Config::small_test(); // hidden=64
    let attn = LLaMA3Attention::new(&config).expect("construction failed");
    let seq_len = 3_usize;
    let arr = ArrayD::from_shape_vec(
        IxDyn(&[seq_len, config.hidden_size]),
        vec![0.01f32; seq_len * config.hidden_size],
    )
    .expect("reshape");
    let input = Tensor::F32(arr);
    let output = attn.forward(input).expect("attention forward failed");
    let out_len: usize = output.shape().iter().product();
    assert_eq!(out_len, seq_len * config.hidden_size);
}

// ─────────────────────────────────────────────────────────────────────────
// SwiGLU MLP
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_llama3_mlp_construction() {
    let config = LLaMA3Config::small_test();
    let mlp = LLaMA3MLP::new(&config);
    assert!(mlp.is_ok(), "MLP construction must succeed");
    assert!(mlp.expect("checked").parameter_count() > 0);
}

#[test]
fn test_llama3_mlp_swiglu_output_shape() {
    let config = LLaMA3Config::small_test(); // hidden=64, intermediate=128
    let mlp = LLaMA3MLP::new(&config).expect("construction failed");
    // Linear requires at least 2D: shape [seq_len=1, hidden=64]
    let arr = ArrayD::from_shape_vec(IxDyn(&[1, 64]), vec![0.01f32; 64]).expect("reshape");
    let input = Tensor::F32(arr);
    let output = mlp.forward(input).expect("MLP forward failed");
    // Output shape: [1, hidden_size] → total = 64
    let total: usize = output.shape().iter().product();
    assert_eq!(total, 64, "MLP output must equal hidden_size");
}

// ─────────────────────────────────────────────────────────────────────────
// Full model
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_llama3_model_construction() {
    let model = LLaMA3Model::new(LLaMA3Config::small_test());
    assert!(model.is_ok(), "LLaMA3Model construction must succeed");
    let model = model.expect("checked");
    assert!(model.parameter_count() > 0);
}

#[test]
fn test_llama3_for_causal_lm_forward() {
    let config = LLaMA3Config::small_test();
    let model = LLaMA3ForCausalLM::new(config.clone()).expect("construction failed");
    let input_ids = vec![1u32, 2, 3];
    let logits = model.forward(input_ids).expect("forward failed");
    let total: usize = logits.shape().iter().product();
    assert_eq!(
        total,
        3 * config.vocab_size,
        "logits = seq_len × vocab_size"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Chat template
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_llama3_chat_format_contains_tokens() {
    let messages = vec![
        ("user".to_string(), "Hello!".to_string()),
        ("assistant".to_string(), "Hi there!".to_string()),
    ];
    let prompt = format_llama3_chat("You are helpful.", &messages);
    assert!(
        prompt.contains("<|begin_of_text|>"),
        "must start with begin_of_text"
    );
    assert!(prompt.contains("<|start_header_id|>system<|end_header_id|>"));
    assert!(prompt.contains("<|start_header_id|>user<|end_header_id|>"));
    assert!(prompt.contains("<|start_header_id|>assistant<|end_header_id|>"));
    assert!(prompt.contains("<|eot_id|>"));
    assert!(prompt.contains("You are helpful."));
    assert!(prompt.contains("Hello!"));
    assert!(prompt.contains("Hi there!"));
}

#[test]
fn test_llama3_chat_model_build_prompt() {
    let model = LLaMA3ChatModel::new(LLaMA3Config::small_test()).expect("construction failed");
    let msgs = vec![("user".to_string(), "Write a poem.".to_string())];
    let prompt = model.build_prompt("You are a poet.", &msgs);
    assert!(prompt.contains("Write a poem."));
    assert!(prompt.contains("You are a poet."));
}
