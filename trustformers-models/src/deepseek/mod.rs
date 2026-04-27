//! # DeepSeek-V2
//!
//! DeepSeek-V2 (DeepSeek-AI, 2024) introduces two key innovations:
//!
//! - **Multi-Head Latent Attention (MLA)**: KV tensors are compressed into a
//!   low-rank latent space before being stored in the KV cache, enabling
//!   large batch sizes at reduced memory cost.
//! - **DeepSeekMoE**: Fine-grained MoE with shared always-active experts and
//!   top-k routed experts.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use trustformers_models::deepseek::{DeepSeekConfig, DeepSeekForCausalLM};
//!
//! let config = DeepSeekConfig::small_test();
//! let model = DeepSeekForCausalLM::new(config)?;
//! let logits = model.forward(vec![1u32, 2, 3])?;
//! # Ok::<(), trustformers_core::errors::TrustformersError>(())
//! ```

pub mod config;
pub mod model;
pub mod tasks;

#[cfg(test)]
mod tests;

pub use config::DeepSeekConfig;
pub use model::{
    DeepSeekDecoderLayer, DeepSeekFfnKind, DeepSeekForCausalLM, DeepSeekMlaAttention, DeepSeekMlp,
    DeepSeekModel, DeepSeekMoeLayer, DeepSeekRmsNorm,
};

// ── Additional tests beyond tests.rs ─────────────────────────────────────────

#[cfg(test)]
mod extra_tests {
    use super::*;
    use trustformers_core::traits::Config;

    // ── Config field accessors ────────────────────────────────────────────────

    #[test]
    fn test_head_dim_small_test() {
        let cfg = DeepSeekConfig::small_test();
        // hidden_size=64, num_attention_heads=4 → head_dim=16
        assert_eq!(cfg.head_dim(), cfg.hidden_size / cfg.num_attention_heads);
        assert_eq!(cfg.head_dim(), 16);
    }

    #[test]
    fn test_head_dim_v2_small() {
        let cfg = DeepSeekConfig::deepseek_v2_small();
        // hidden_size=2048, num_attention_heads=16 → head_dim=128
        assert_eq!(cfg.head_dim(), 128);
    }

    #[test]
    fn test_architecture_label() {
        let cfg = DeepSeekConfig::small_test();
        assert_eq!(cfg.architecture(), "DeepSeek-V2");
    }

    #[test]
    fn test_v_head_dim_accessor() {
        let cfg = DeepSeekConfig::small_test();
        assert_eq!(cfg.v_head_dim, 16);
        let cfg2 = DeepSeekConfig::deepseek_v2_small();
        assert_eq!(cfg2.v_head_dim, 128);
    }

    #[test]
    fn test_rope_head_dim_accessor() {
        let cfg = DeepSeekConfig::small_test();
        assert_eq!(cfg.rope_head_dim, 8);
        let cfg2 = DeepSeekConfig::deepseek_v2_small();
        assert_eq!(cfg2.rope_head_dim, 64);
    }

    #[test]
    fn test_rope_theta_accessor() {
        let cfg = DeepSeekConfig::small_test();
        assert!((cfg.rope_theta - 10000.0).abs() < 1e-6);
    }

    #[test]
    fn test_rms_norm_eps_accessor() {
        let cfg = DeepSeekConfig::small_test();
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_q_lora_rank_none_by_default() {
        let cfg = DeepSeekConfig::small_test();
        assert!(cfg.q_lora_rank.is_none());
        let cfg2 = DeepSeekConfig::deepseek_v2_small();
        assert!(cfg2.q_lora_rank.is_none());
    }

    #[test]
    fn test_q_lora_rank_some_variant() {
        let mut cfg = DeepSeekConfig::small_test();
        cfg.q_lora_rank = Some(32);
        assert_eq!(cfg.q_lora_rank, Some(32));
        assert!(cfg.validate().is_ok());
    }

    // ── MoE layer frequency stride ────────────────────────────────────────────

    #[test]
    fn test_moe_layer_stride_pattern_freq_2() {
        let mut cfg = DeepSeekConfig::small_test();
        // first_k_dense_replace=1, moe_layer_freq=2
        cfg.moe_layer_freq = 2;
        assert!(
            !cfg.is_moe_layer(0),
            "layer 0 must be dense (in dense prefix)"
        );
        assert!(cfg.is_moe_layer(1), "layer 1: offset=0, 0%2==0 → MoE");
        assert!(!cfg.is_moe_layer(2), "layer 2: offset=1, 1%2!=0 → dense");
        assert!(cfg.is_moe_layer(3), "layer 3: offset=2, 2%2==0 → MoE");
    }

    #[test]
    fn test_moe_layer_all_dense_when_first_k_covers_all() {
        let mut cfg = DeepSeekConfig::small_test();
        cfg.num_hidden_layers = 4;
        cfg.first_k_dense_replace = 4; // all layers are in the dense prefix
        for i in 0..4 {
            assert!(!cfg.is_moe_layer(i), "all layers should be dense");
        }
    }

    // ── Validation error paths ────────────────────────────────────────────────

    #[test]
    fn test_validate_zero_hidden_size() {
        let mut cfg = DeepSeekConfig::small_test();
        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_vocab_size() {
        let mut cfg = DeepSeekConfig::small_test();
        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_hidden_layers() {
        let mut cfg = DeepSeekConfig::small_test();
        cfg.num_hidden_layers = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_experts_per_tok_exceeds_routed() {
        let mut cfg = DeepSeekConfig::small_test();
        cfg.num_experts_per_tok = cfg.n_routed_experts + 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_hidden_size_not_divisible_by_heads() {
        let mut cfg = DeepSeekConfig::small_test();
        cfg.hidden_size = 65; // 65 % 4 != 0
        assert!(cfg.validate().is_err());
    }

    // ── Config clone and default ──────────────────────────────────────────────

    #[test]
    fn test_config_clone_is_equal() {
        let cfg = DeepSeekConfig::small_test();
        let cloned = cfg.clone();
        assert_eq!(cfg.hidden_size, cloned.hidden_size);
        assert_eq!(cfg.kv_lora_rank, cloned.kv_lora_rank);
        assert_eq!(cfg.n_routed_experts, cloned.n_routed_experts);
    }

    #[test]
    fn test_default_equals_v2_small() {
        let default = DeepSeekConfig::default();
        let v2_small = DeepSeekConfig::deepseek_v2_small();
        assert_eq!(default.hidden_size, v2_small.hidden_size);
        assert_eq!(default.vocab_size, v2_small.vocab_size);
        assert_eq!(default.kv_lora_rank, v2_small.kv_lora_rank);
    }

    // ── Serialization round-trip ──────────────────────────────────────────────

    #[test]
    fn test_config_json_round_trip() {
        let cfg = DeepSeekConfig::small_test();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let restored: DeepSeekConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg.hidden_size, restored.hidden_size);
        assert_eq!(cfg.kv_lora_rank, restored.kv_lora_rank);
        assert_eq!(cfg.n_routed_experts, restored.n_routed_experts);
        assert_eq!(cfg.q_lora_rank, restored.q_lora_rank);
    }

    // ── MoE shared vs routed expert counts ───────────────────────────────────

    #[test]
    fn test_v2_small_expert_counts() {
        let cfg = DeepSeekConfig::deepseek_v2_small();
        assert_eq!(cfg.n_routed_experts, 64);
        assert_eq!(cfg.n_shared_experts, 2);
        assert_eq!(cfg.num_experts_per_tok, 6);
        // experts_per_tok must not exceed routed experts
        assert!(cfg.num_experts_per_tok <= cfg.n_routed_experts);
    }

    #[test]
    fn test_small_test_kv_lora_rank_smaller_than_hidden() {
        let cfg = DeepSeekConfig::small_test();
        assert!(
            cfg.kv_lora_rank < cfg.hidden_size,
            "MLA compression rank should be smaller than hidden_size for memory savings"
        );
    }

    // ── FfnKind enum ─────────────────────────────────────────────────────────

    #[test]
    fn test_ffn_kind_discriminants_via_decoder_layer() {
        // Verify that a dense-prefix layer contains a Dense FFN and that a MoE
        // layer contains a Moe FFN, using the public `is_moe()` accessor.
        let cfg = DeepSeekConfig::small_test();
        // layer 0 → dense (first_k_dense_replace = 1)
        let layer0 = DeepSeekDecoderLayer::new(&cfg, 0).expect("layer0");
        assert!(!layer0.is_moe(), "layer0 FFN must be Dense");
        // layer 1 → MoE
        let layer1 = DeepSeekDecoderLayer::new(&cfg, 1).expect("layer1");
        assert!(layer1.is_moe(), "layer1 FFN must be Moe");
    }
}
