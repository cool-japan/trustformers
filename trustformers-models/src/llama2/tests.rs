use crate::llama2::config::LLaMA2Config;
use crate::llama2::model::{
    LLaMA2Attention, LLaMA2ForCausalLM, LLaMA2MLP, LLaMA2Model, LLaMA2RMSNorm,
    LLaMA2RotaryEmbedding,
};
use trustformers_core::{tensor::Tensor, traits::Config};

/// Minimal config for fast tests — avoids large heap allocations
fn mini_config() -> LLaMA2Config {
    LLaMA2Config {
        hidden_size: 64,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 8,
        num_key_value_heads: 8, // MHA (no GQA)
        vocab_size: 512,
        max_position_embeddings: 128,
        ..LLaMA2Config::default()
    }
}

/// Minimal config with GQA (2 KV heads for 8 Q heads → 4x sharing)
fn mini_gqa_config() -> LLaMA2Config {
    LLaMA2Config {
        hidden_size: 64,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 8,
        num_key_value_heads: 2, // GQA: 4x sharing
        vocab_size: 512,
        max_position_embeddings: 128,
        ..LLaMA2Config::default()
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Config validation
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_llama2_config_default_is_valid() {
    let config = mini_config();
    assert!(
        config.validate().is_ok(),
        "default mini config must be valid"
    );
}

#[test]
fn test_llama2_config_invalid_head_division() {
    let config = LLaMA2Config {
        hidden_size: 65, // Not divisible by 8 heads
        ..mini_config()
    };
    assert!(
        config.validate().is_err(),
        "hidden_size not divisible by num_attention_heads should be invalid"
    );
}

#[test]
fn test_llama2_config_invalid_kv_head_division() {
    let config = LLaMA2Config {
        num_attention_heads: 8,
        num_key_value_heads: 3, // 8 % 3 != 0
        ..mini_config()
    };
    assert!(
        config.validate().is_err(),
        "num_attention_heads not divisible by num_key_value_heads should be invalid"
    );
}

#[test]
fn test_llama2_7b_config_shape() {
    let cfg = LLaMA2Config::llama2_7b();
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 32);
    assert_eq!(cfg.max_position_embeddings, 4096);
    assert!(!cfg.uses_gqa(), "7B uses full MHA");
    assert_eq!(cfg.num_query_groups(), 1);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_llama2_70b_config_gqa() {
    let cfg = LLaMA2Config::llama2_70b();
    assert_eq!(cfg.num_attention_heads, 64);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert!(cfg.uses_gqa(), "70B must use GQA");
    assert_eq!(cfg.num_query_groups(), 8); // 64/8
    assert_eq!(cfg.head_dim(), 128); // 8192/64
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_llama2_13b_config() {
    let cfg = LLaMA2Config::llama2_13b();
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.num_attention_heads, 40);
    assert_eq!(cfg.num_key_value_heads, 40);
    assert!(!cfg.uses_gqa());
    assert!(cfg.validate().is_ok());
}

// ─────────────────────────────────────────────────────────────────────────
// GQA repeat_kv logic
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_gqa_repeat_kv_no_op_when_groups_equal_one() {
    let config = mini_config(); // 8 Q, 8 KV → num_query_groups = 1
    let attn = LLaMA2Attention::new(&config).expect("attention construction failed");

    let kv = Tensor::ones(&[8]).expect("tensor creation failed");
    let expanded = attn.repeat_kv(&kv).expect("repeat_kv failed");
    assert_eq!(
        expanded.shape(),
        kv.shape(),
        "no expansion expected when num_query_groups == 1"
    );
}

#[test]
fn test_gqa_repeat_kv_expands_correctly() {
    let config = mini_gqa_config(); // 8 Q, 2 KV → 4x sharing
    let attn = LLaMA2Attention::new(&config).expect("attention construction failed");

    // Simulate 2 KV heads each with head_dim=8 features = 16 values
    let head_dim = config.head_dim(); // 64/8 = 8
    let kv_features = config.num_key_value_heads * head_dim; // 2 * 8 = 16
    let kv = Tensor::ones(&[kv_features]).expect("tensor creation failed");

    let expanded = attn.repeat_kv(&kv).expect("repeat_kv failed");
    let expected_len: usize = expanded.shape().iter().product();
    let kv_len: usize = kv.shape().iter().product();

    assert_eq!(
        expected_len,
        kv_len * config.num_query_groups(),
        "expanded KV should be num_query_groups times larger"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// RMSNorm
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_rmsnorm_construction() {
    let norm = LLaMA2RMSNorm::new(64, 1e-5);
    assert!(norm.is_ok(), "RMSNorm construction must succeed");
    let norm = norm.expect("already checked");
    assert_eq!(norm.parameter_count(), 64);
}

// ─────────────────────────────────────────────────────────────────────────
// RotaryEmbedding
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_rope_inv_freq_length() {
    let rope = LLaMA2RotaryEmbedding::new(64, 128, 10000.0);
    let inv_freq = rope.compute_inv_freq();
    assert_eq!(inv_freq.len(), 32, "inv_freq length must be dim/2");
}

#[test]
fn test_rope_apply_returns_same_shape() {
    let rope = LLaMA2RotaryEmbedding::new(8, 16, 10000.0);
    let q = Tensor::ones(&[4, 8]).expect("tensor");
    let k = Tensor::ones(&[4, 8]).expect("tensor");
    let positions: Vec<usize> = (0..4).collect();
    let (q_out, k_out) = rope.apply_rotary_emb(&q, &k, &positions).expect("rope");
    assert_eq!(q_out.shape(), q.shape());
    assert_eq!(k_out.shape(), k.shape());
}

// ─────────────────────────────────────────────────────────────────────────
// Model construction & forward pass
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn test_llama2_model_construction() {
    let model = LLaMA2Model::new(mini_config());
    assert!(model.is_ok(), "model construction must succeed");
    let model = model.expect("already checked");
    assert!(model.parameter_count() > 0);
}

#[test]
fn test_llama2_for_causal_lm_weight_map() {
    let model = LLaMA2ForCausalLM::new(mini_config()).expect("model construction failed");
    let map = model.weight_map();
    assert!(map.contains_key("model.embed_tokens.weight"));
    assert!(map.contains_key("model.norm.weight"));
    assert!(map.contains_key("lm_head.weight"));
    // Each layer should contribute 9 weight entries
    let config = mini_config();
    for i in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{i}");
        assert!(
            map.contains_key(&format!("{prefix}.self_attn.q_proj.weight")),
            "missing q_proj for layer {i}"
        );
        assert!(
            map.contains_key(&format!("{prefix}.mlp.gate_proj.weight")),
            "missing gate_proj for layer {i}"
        );
    }
}

#[test]
fn test_llama2_mlp_construction() {
    let mlp = LLaMA2MLP::new(&mini_config());
    assert!(mlp.is_ok(), "MLP construction must succeed");
    assert!(mlp.expect("checked").parameter_count() > 0);
}

#[test]
fn test_llama2_architecture_label() {
    let config = mini_config();
    assert_eq!(config.architecture(), "LLaMA-2");
}
