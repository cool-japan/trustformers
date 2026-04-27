use crate::codellama::config::{CodeLlamaConfig, RopeScalingConfig, RopeScalingType};
use crate::codellama::model::{
    CodeLlamaAttention, CodeLlamaForCausalLM, CodeLlamaMLP, CodeLlamaModel, CodeLlamaRMSNorm,
    CodeLlamaRotaryEmbedding,
};
use trustformers_core::{tensor::Tensor, traits::Config};

fn mini_config() -> CodeLlamaConfig {
    CodeLlamaConfig {
        hidden_size: 64,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 8,
        num_key_value_heads: 8,
        vocab_size: 512,
        max_position_embeddings: 128,
        rope_scaling: None,
        infilling: false,
        programming_languages: vec!["rust".to_string()],
        ..CodeLlamaConfig::default()
    }
}

fn mini_gqa_config() -> CodeLlamaConfig {
    CodeLlamaConfig {
        hidden_size: 64,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 8,
        num_key_value_heads: 2,
        vocab_size: 512,
        max_position_embeddings: 128,
        rope_scaling: None,
        infilling: true,
        programming_languages: vec![],
        ..CodeLlamaConfig::default()
    }
}

// ── Config validation ─────────────────────────────────────────────────────

#[test]
fn test_codellama_config_valid() {
    assert!(mini_config().validate().is_ok());
}

#[test]
fn test_codellama_config_invalid_hidden_size() {
    let config = CodeLlamaConfig {
        hidden_size: 65, // not divisible by 8
        ..mini_config()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_codellama_config_invalid_kv_heads() {
    let config = CodeLlamaConfig {
        num_key_value_heads: 3, // 8 % 3 != 0
        ..mini_config()
    };
    assert!(config.validate().is_err());
}

#[test]
fn test_codellama_7b_preset() {
    let cfg = CodeLlamaConfig::codellama_7b();
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 32);
    assert!(!cfg.uses_gqa());
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_codellama_34b_rope_scaling() {
    let cfg = CodeLlamaConfig::codellama_34b();
    assert!(cfg.rope_scaling.is_some(), "34B must have rope_scaling");
    let scaling = cfg.rope_scaling.as_ref().expect("checked above");
    assert_eq!(scaling.scaling_type, RopeScalingType::Linear);
    assert!(scaling.factor > 1.0, "scale factor must be > 1");
    assert!(cfg.uses_gqa(), "34B uses GQA");
    assert_eq!(
        cfg.num_query_groups(),
        cfg.num_attention_heads / cfg.num_key_value_heads
    );
    // Effective context should exceed native context
    assert!(cfg.effective_max_context() > cfg.max_position_embeddings);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_codellama_70b_gqa() {
    let cfg = CodeLlamaConfig::codellama_70b();
    assert_eq!(cfg.num_attention_heads, 64);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.num_query_groups(), 8);
    assert!(cfg.uses_gqa());
    assert!(cfg.validate().is_ok());
}

// ── RoPE scaling logic ────────────────────────────────────────────────────

#[test]
fn test_linear_rope_scaling_reduces_freq() {
    let config_base = mini_config();
    let config_scaled = CodeLlamaConfig {
        rope_scaling: Some(RopeScalingConfig {
            scaling_type: RopeScalingType::Linear,
            factor: 4.0,
        }),
        ..mini_config()
    };

    let rope_base = CodeLlamaRotaryEmbedding::new(&config_base);
    let rope_scaled = CodeLlamaRotaryEmbedding::new(&config_scaled);

    let freq_base = rope_base.compute_inv_freq();
    let freq_scaled = rope_scaled.compute_inv_freq();

    for (b, s) in freq_base.iter().zip(freq_scaled.iter()) {
        assert!(
            s < b,
            "linear scaled freq {s} should be less than base freq {b}"
        );
    }
}

#[test]
fn test_dynamic_ntk_rope_scaling() {
    let config = CodeLlamaConfig {
        rope_scaling: Some(RopeScalingConfig {
            scaling_type: RopeScalingType::Dynamic,
            factor: 2.0,
        }),
        ..mini_config()
    };
    let rope = CodeLlamaRotaryEmbedding::new(&config);
    let inv_freq = rope.compute_inv_freq();
    assert_eq!(inv_freq.len(), config.head_dim() / 2);
    // All freqs should be finite and positive
    for f in &inv_freq {
        assert!(f.is_finite() && *f > 0.0, "freq should be finite positive");
    }
}

// ── RMSNorm ───────────────────────────────────────────────────────────────

#[test]
fn test_codellama_rmsnorm() {
    let norm = CodeLlamaRMSNorm::new(64, 1e-5);
    assert!(norm.is_ok());
    assert_eq!(norm.expect("checked").parameter_count(), 64);
}

// ── Attention & repeat_kv ─────────────────────────────────────────────────

#[test]
fn test_repeat_kv_no_expansion_for_mha() {
    let config = mini_config();
    let attn = CodeLlamaAttention::new(&config).expect("attention");
    let kv = Tensor::ones(&[16]).expect("tensor");
    let expanded = attn.repeat_kv(&kv).expect("repeat_kv");
    assert_eq!(expanded.shape(), kv.shape());
}

#[test]
fn test_repeat_kv_expansion_for_gqa() {
    let config = mini_gqa_config();
    let attn = CodeLlamaAttention::new(&config).expect("attention");
    let head_dim = config.head_dim();
    let kv_features = config.num_key_value_heads * head_dim;
    let kv = Tensor::ones(&[kv_features]).expect("tensor");
    let expanded = attn.repeat_kv(&kv).expect("repeat_kv");
    let expanded_len: usize = expanded.shape().iter().product();
    let kv_len: usize = kv.shape().iter().product();
    assert_eq!(expanded_len, kv_len * config.num_query_groups());
}

// ── Model construction ────────────────────────────────────────────────────

#[test]
fn test_codellama_model_construction() {
    let model = CodeLlamaModel::new(mini_config());
    assert!(model.is_ok());
    assert!(model.expect("checked").parameter_count() > 0);
}

#[test]
fn test_codellama_causal_lm_weight_map() {
    let model = CodeLlamaForCausalLM::new(mini_config()).expect("model");
    let map = model.weight_map();
    assert!(map.contains_key("model.embed_tokens.weight"));
    assert!(map.contains_key("lm_head.weight"));
    let config = mini_config();
    for i in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{i}");
        assert!(map.contains_key(&format!("{prefix}.self_attn.q_proj.weight")));
        assert!(map.contains_key(&format!("{prefix}.mlp.gate_proj.weight")));
    }
}

#[test]
fn test_codellama_architecture_label() {
    assert_eq!(mini_config().architecture(), "CodeLlama");
}

#[test]
fn test_codellama_mlp_construction() {
    let mlp = CodeLlamaMLP::new(&mini_config());
    assert!(mlp.is_ok());
    assert!(mlp.expect("checked").parameter_count() > 0);
}
