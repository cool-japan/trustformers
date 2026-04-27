//! # DeepSeek-V2 (DeepSeek-AI, 2024)
//!
//! DeepSeek-V2 is a 236B Mixture-of-Experts language model with two major architectural
//! novelties:
//!
//! ## Multi-head Latent Attention (MLA)
//!
//! Standard MHA caches `num_heads × (k_dim + v_dim)` floats per token per layer.
//! MLA instead compresses the joint KV representation into a low-rank latent vector
//! `c_KV ∈ ℝ^{kv_lora_rank}` (plus a small RoPE key slice of size `qk_rope_head_dim`).
//! K and V are re-expanded from this latent at every forward pass.
//!
//! For the default 236B config, the KV cache per token is reduced by ~5.75× compared to
//! standard MHA with the same number of heads.
//!
//! ## Mixture of Experts (MoE)
//!
//! All layers except the first `first_k_dense_replace` use sparse MoE FFN:
//! - `n_shared_experts` always-active experts provide a stable base computation.
//! - `n_routed_experts` experts are routed with GroupLimitedGreedy top-k selection:
//!   experts are first ranked within each of `n_group` groups (keeping `topk_group` per group),
//!   then the global top-`num_experts_per_tok` are selected.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use trustformers_models::deepseek_v2::{DeepSeekV2Config, DeepSeekV2ForCausalLM};
//!
//! // Tiny config for tests / prototyping
//! let config = DeepSeekV2Config {
//!     vocab_size: 64,
//!     hidden_size: 32,
//!     intermediate_size: 64,
//!     num_hidden_layers: 2,
//!     num_attention_heads: 4,
//!     kv_lora_rank: 16,
//!     q_lora_rank: 32,
//!     qk_rope_head_dim: 8,
//!     qk_nope_head_dim: 8,
//!     v_head_dim: 8,
//!     n_routed_experts: 8,
//!     num_experts_per_tok: 2,
//!     n_group: 2,
//!     topk_group: 1,
//!     ..DeepSeekV2Config::default()
//! };
//! let model = DeepSeekV2ForCausalLM::new(config).expect("model creation");
//! ```

pub mod config;
pub mod model;
pub mod tasks;

pub use config::{ActivationType, DeepSeekV2Config, TopKMethod};
pub use model::{
    apply_activation, gelu, silu, DeepSeekV2DecoderLayer, DeepSeekV2MLP, DeepSeekV2MoELayer,
    DeepSeekV2Model, DeepSeekV2RmsNorm, DeepSeekV2RotaryEmbedding, ExpertRouter, MlaAttention,
};
pub use tasks::{DeepSeekV2Error, DeepSeekV2ForCausalLM};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    /// Build a minimal DeepSeek-V2 config suitable for unit tests.
    fn tiny_config() -> DeepSeekV2Config {
        DeepSeekV2Config {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            kv_lora_rank: 16,
            q_lora_rank: 32,
            qk_rope_head_dim: 8,
            qk_nope_head_dim: 8,
            v_head_dim: 8,
            n_routed_experts: 8,
            num_experts_per_tok: 2,
            n_shared_experts: 1,
            n_group: 2,
            topk_group: 1,
            first_k_dense_replace: 1,
            moe_layer_freq: 1,
            ..DeepSeekV2Config::default()
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: Default config field values
    // -----------------------------------------------------------------------
    #[test]
    fn test_config_defaults() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.vocab_size, 102400);
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.intermediate_size, 12288);
        assert_eq!(cfg.num_hidden_layers, 60);
        assert_eq!(cfg.num_attention_heads, 128);
        assert_eq!(cfg.kv_lora_rank, 512);
        assert_eq!(cfg.q_lora_rank, 1536);
        assert_eq!(cfg.qk_rope_head_dim, 64);
        assert_eq!(cfg.qk_nope_head_dim, 128);
        assert_eq!(cfg.v_head_dim, 128);
        assert_eq!(cfg.n_routed_experts, 160);
        assert_eq!(cfg.num_experts_per_tok, 6);
        assert_eq!(cfg.n_shared_experts, 2);
        assert_eq!(cfg.n_group, 8);
        assert_eq!(cfg.topk_group, 3);
        assert!((cfg.aux_loss_alpha - 0.001).abs() < 1e-7);
        assert_eq!(cfg.first_k_dense_replace, 1);
        assert_eq!(cfg.moe_layer_freq, 1);
    }

    // -----------------------------------------------------------------------
    // Test 2: MLA projection dimensions are consistent
    // -----------------------------------------------------------------------
    #[test]
    fn test_mla_projection_dimensions() {
        let cfg = tiny_config();
        // The full query head dim is rope + nope
        assert_eq!(
            cfg.qk_head_dim(),
            cfg.qk_rope_head_dim + cfg.qk_nope_head_dim
        );
        // q_b_proj output: num_heads * (rope + nope)
        let q_b_out = cfg.num_attention_heads * cfg.qk_head_dim();
        assert_eq!(q_b_out, 4 * 16);
        // o_proj input: num_heads * v_head_dim
        let o_in = cfg.num_attention_heads * cfg.v_head_dim;
        assert_eq!(o_in, 4 * 8);
    }

    // -----------------------------------------------------------------------
    // Test 3: Dense vs MoE layer selection logic
    // -----------------------------------------------------------------------
    #[test]
    fn test_dense_vs_moe_layer_selection() {
        let cfg = tiny_config();
        // first_k_dense_replace = 1, so layer 0 is dense
        assert!(cfg.is_dense_layer(0), "layer 0 should be dense");
        // moe_layer_freq = 1, so all layers >= first_k_dense_replace are MoE
        assert!(!cfg.is_dense_layer(1), "layer 1 should be MoE");
        assert!(!cfg.is_dense_layer(2), "layer 2 should be MoE");

        // With first_k_dense_replace=3, layers 0,1,2 are dense
        let cfg2 = DeepSeekV2Config {
            first_k_dense_replace: 3,
            moe_layer_freq: 2,
            ..tiny_config()
        };
        assert!(cfg2.is_dense_layer(0));
        assert!(cfg2.is_dense_layer(1));
        assert!(cfg2.is_dense_layer(2));
        // Layer 3: (3-3) % 2 = 0 → MoE
        assert!(!cfg2.is_dense_layer(3));
        // Layer 4: (4-3) % 2 = 1 → dense
        assert!(cfg2.is_dense_layer(4));
    }

    // -----------------------------------------------------------------------
    // Test 4: Config validation catches bad inputs
    // -----------------------------------------------------------------------
    #[test]
    fn test_config_validation() {
        let mut cfg = tiny_config();
        assert!(cfg.validate().is_ok(), "valid config should pass");

        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err(), "vocab_size=0 should fail");
        cfg.vocab_size = 64;

        cfg.kv_lora_rank = 0;
        assert!(cfg.validate().is_err(), "kv_lora_rank=0 should fail");
        cfg.kv_lora_rank = 16;

        cfg.num_experts_per_tok = cfg.n_routed_experts + 1;
        assert!(
            cfg.validate().is_err(),
            "num_experts_per_tok > n_routed_experts should fail"
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: Causal LM forward mock (small model)
    // -----------------------------------------------------------------------
    #[test]
    fn test_causal_lm_forward_mock() {
        use trustformers_core::tensor::Tensor;
        use trustformers_core::traits::Model;

        let cfg = tiny_config();
        let model = DeepSeekV2ForCausalLM::new(cfg).expect("model creation");
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).expect("tensor");
        let result = model.forward(input);
        assert!(result.is_ok(), "forward failed: {:?}", result.err());
    }

    // -----------------------------------------------------------------------
    // Test 6: KV cache size — MLA saves vs standard MHA
    // -----------------------------------------------------------------------
    #[test]
    fn test_kv_cache_compression() {
        let cfg = DeepSeekV2Config::default();
        let mla_size = cfg.mla_kv_cache_per_token_per_layer();
        let mha_size = cfg.mha_kv_cache_per_token_per_layer();
        // MLA must be strictly smaller than MHA
        assert!(
            mla_size < mha_size,
            "MLA KV cache ({mla_size}) should be smaller than MHA ({mha_size})"
        );
        let ratio = cfg.kv_cache_compression_ratio();
        // For default config the ratio should be well below 0.5 (≈ 0.174)
        assert!(
            ratio < 0.5,
            "KV cache compression ratio {ratio} should be < 0.5"
        );
        assert!(ratio > 0.0, "compression ratio must be positive");

        // Tiny config sanity check
        let tiny = tiny_config();
        let tiny_mla = tiny.mla_kv_cache_per_token_per_layer();
        let tiny_mha = tiny.mha_kv_cache_per_token_per_layer();
        assert!(
            tiny_mla < tiny_mha,
            "tiny config: MLA ({tiny_mla}) must be < MHA ({tiny_mha})"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: TopKMethod and ActivationType Display impls
    // -----------------------------------------------------------------------
    #[test]
    fn test_topk_method_display() {
        assert_eq!(
            TopKMethod::GroupLimitedGreedy.to_string(),
            "GroupLimitedGreedy"
        );
        assert_eq!(TopKMethod::Noaux.to_string(), "Noaux");

        assert_eq!(ActivationType::SiLU.to_string(), "silu");
        assert_eq!(ActivationType::GeLU.to_string(), "gelu");

        // Round-trip serialisation
        let json = serde_json::to_string(&TopKMethod::GroupLimitedGreedy).expect("serialize");
        let back: TopKMethod = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, TopKMethod::GroupLimitedGreedy);
    }

    // -----------------------------------------------------------------------
    // Test 8: MoE layer has correct expert counts
    // -----------------------------------------------------------------------
    #[test]
    fn test_moe_layer_expert_counts() {
        use trustformers_core::device::Device;

        let cfg = tiny_config();
        let moe = DeepSeekV2MoELayer::new(&cfg, Device::CPU).expect("moe creation");
        assert_eq!(moe.num_routed_experts(), cfg.n_routed_experts);
        assert_eq!(moe.num_shared_experts(), cfg.n_shared_experts);
    }

    // -----------------------------------------------------------------------
    // Test 9: Decoder layer dense/MoE identity is consistent with config
    // -----------------------------------------------------------------------
    #[test]
    fn test_decoder_layer_type() {
        use trustformers_core::device::Device;

        let cfg = tiny_config();
        // Layer 0 → dense (first_k_dense_replace = 1)
        let layer0 = DeepSeekV2DecoderLayer::new(&cfg, 0, Device::CPU).expect("layer0 creation");
        assert!(layer0.is_dense(), "layer 0 must be dense");

        // Layer 1 → MoE
        let layer1 = DeepSeekV2DecoderLayer::new(&cfg, 1, Device::CPU).expect("layer1 creation");
        assert!(!layer1.is_dense(), "layer 1 must be MoE");
    }

    // -----------------------------------------------------------------------
    // Test 10: kv_lora_rank default and typical value
    // -----------------------------------------------------------------------
    #[test]
    fn test_kv_lora_rank_default() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.kv_lora_rank, 512, "default kv_lora_rank must be 512");
    }

    // -----------------------------------------------------------------------
    // Test 11: q_lora_rank default and typical value
    // -----------------------------------------------------------------------
    #[test]
    fn test_q_lora_rank_default() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.q_lora_rank, 1536, "default q_lora_rank must be 1536");
    }

    // -----------------------------------------------------------------------
    // Test 12: qk_rope_head_dim vs qk_nope_head_dim distinction
    // -----------------------------------------------------------------------
    #[test]
    fn test_qk_rope_vs_nope_head_dim() {
        let cfg = DeepSeekV2Config::default();
        assert_ne!(
            cfg.qk_rope_head_dim, cfg.qk_nope_head_dim,
            "rope and nope head dims must differ for default 236B config"
        );
        // qk_rope=64, qk_nope=128 in default config
        assert_eq!(cfg.qk_rope_head_dim, 64);
        assert_eq!(cfg.qk_nope_head_dim, 128);
    }

    // -----------------------------------------------------------------------
    // Test 13: qk_head_dim = rope + nope
    // -----------------------------------------------------------------------
    #[test]
    fn test_qk_head_dim_sum() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(
            cfg.qk_head_dim(),
            cfg.qk_rope_head_dim + cfg.qk_nope_head_dim,
            "qk_head_dim must equal qk_rope + qk_nope"
        );
        // Same for tiny config
        let tiny = tiny_config();
        assert_eq!(
            tiny.qk_head_dim(),
            tiny.qk_rope_head_dim + tiny.qk_nope_head_dim
        );
    }

    // -----------------------------------------------------------------------
    // Test 14: MLA architecture — kv_lora_rank < num_heads * v_head_dim
    // -----------------------------------------------------------------------
    #[test]
    fn test_mla_kv_latent_smaller_than_expanded_kv() {
        let cfg = DeepSeekV2Config::default();
        let expanded_v = cfg.num_attention_heads * cfg.v_head_dim;
        assert!(
            cfg.kv_lora_rank < expanded_v,
            "kv_lora_rank ({}) must be smaller than num_heads*v_head_dim ({}) — compression",
            cfg.kv_lora_rank,
            expanded_v
        );
    }

    // -----------------------------------------------------------------------
    // Test 15: num_experts for MoE variant
    // -----------------------------------------------------------------------
    #[test]
    fn test_moe_num_routed_experts_default() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.n_routed_experts, 160, "default has 160 routed experts");
    }

    // -----------------------------------------------------------------------
    // Test 16: num_experts_per_tok for routing
    // -----------------------------------------------------------------------
    #[test]
    fn test_moe_num_experts_per_tok_default() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(
            cfg.num_experts_per_tok, 6,
            "default routes to 6 experts per token"
        );
        assert!(
            cfg.num_experts_per_tok < cfg.n_routed_experts,
            "experts_per_tok must be less than total routed experts"
        );
    }

    // -----------------------------------------------------------------------
    // Test 17: Dense model variant (all layers dense)
    // -----------------------------------------------------------------------
    #[test]
    fn test_dense_model_all_layers_dense() {
        // A config with first_k_dense_replace >= num_hidden_layers is all-dense
        let cfg = DeepSeekV2Config {
            first_k_dense_replace: 4, // all layers dense for 2-layer model
            moe_layer_freq: 1,
            num_hidden_layers: 2,
            ..tiny_config()
        };
        for i in 0..cfg.num_hidden_layers {
            assert!(
                cfg.is_dense_layer(i),
                "layer {i} should be dense in all-dense variant"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 18: Shared experts vs routed experts distinction
    // -----------------------------------------------------------------------
    #[test]
    fn test_shared_vs_routed_experts() {
        let cfg = DeepSeekV2Config::default();
        // Shared experts are always active; routed experts use top-k selection
        assert!(
            cfg.n_shared_experts > 0,
            "must have at least one shared expert"
        );
        assert!(
            cfg.n_routed_experts > cfg.n_shared_experts,
            "more routed than shared"
        );
    }

    // -----------------------------------------------------------------------
    // Test 19: ExpertRouter — routing output shape (via route method)
    // -----------------------------------------------------------------------
    #[test]
    fn test_expert_router_output_shape() {
        use trustformers_core::{device::Device, tensor::Tensor};
        let cfg = tiny_config();
        let router = ExpertRouter::new(&cfg, Device::CPU);
        // Linear requires at least 2D input: [1, hidden_size]
        let hidden =
            Tensor::from_vec(vec![0.1f32; cfg.hidden_size], &[1, cfg.hidden_size]).expect("tensor");
        let result = router.route(&hidden).expect("router route");
        let (expert_indices, weights) = result;
        // top-k experts selected per token
        assert_eq!(
            expert_indices.len(),
            cfg.num_experts_per_tok,
            "router must select num_experts_per_tok experts"
        );
        assert_eq!(
            weights.len(),
            cfg.num_experts_per_tok,
            "routing weights count must match num_experts_per_tok"
        );
    }

    // -----------------------------------------------------------------------
    // Test 20: MLA attention — compressed KV dims smaller than full KV
    // -----------------------------------------------------------------------
    #[test]
    fn test_mla_kv_cache_per_token_ratio() {
        let cfg = tiny_config();
        let mla = cfg.mla_kv_cache_per_token_per_layer();
        let mha = cfg.mha_kv_cache_per_token_per_layer();
        assert!(
            mla < mha,
            "MLA ({mla}) must be smaller than MHA ({mha}) for tiny config"
        );
    }

    // -----------------------------------------------------------------------
    // Test 21: CausalLM forward_ids returns seq_len * vocab_size logits
    // -----------------------------------------------------------------------
    #[test]
    fn test_causal_lm_forward_ids_output_length() {
        let cfg = tiny_config();
        let model = DeepSeekV2ForCausalLM::new(cfg.clone()).expect("model");
        let logits = model.forward_ids(&[1u32, 2, 3]).expect("forward_ids");
        assert_eq!(
            logits.len(),
            3 * cfg.vocab_size,
            "forward_ids output must be seq*vocab"
        );
    }

    // -----------------------------------------------------------------------
    // Test 22: CausalLM empty input returns EmptyInput error
    // -----------------------------------------------------------------------
    #[test]
    fn test_causal_lm_empty_input_error() {
        let cfg = tiny_config();
        let model = DeepSeekV2ForCausalLM::new(cfg).expect("model");
        let result = model.forward_ids(&[]);
        assert!(result.is_err(), "empty input must return an error");
        matches!(result.unwrap_err(), DeepSeekV2Error::EmptyInput);
    }

    // -----------------------------------------------------------------------
    // Test 23: Greedy generation returns correct token count
    // -----------------------------------------------------------------------
    #[test]
    fn test_causal_lm_generate_token_count() {
        let cfg = tiny_config();
        let model = DeepSeekV2ForCausalLM::new(cfg).expect("model");
        let result = model.generate(&[1u32, 2], 3);
        assert!(result.is_ok(), "generate failed: {:?}", result.err());
        assert_eq!(
            result.expect("generated").len(),
            3,
            "must generate exactly 3 tokens"
        );
    }

    // -----------------------------------------------------------------------
    // Test 24: Error display impls
    // -----------------------------------------------------------------------
    #[test]
    fn test_error_display_invalid_config() {
        let s = format!(
            "{}",
            DeepSeekV2Error::InvalidConfig("bad param".to_string())
        );
        assert!(
            s.contains("bad param"),
            "InvalidConfig display must include message"
        );
    }

    #[test]
    fn test_error_display_empty_input() {
        let s = format!("{}", DeepSeekV2Error::EmptyInput);
        assert!(
            s.contains("empty") || s.contains("Empty"),
            "EmptyInput must mention empty"
        );
    }

    #[test]
    fn test_error_display_shape_mismatch() {
        let err = DeepSeekV2Error::ShapeMismatch {
            expected: vec![3, 4],
            got: vec![5, 6],
        };
        let s = format!("{err}");
        assert!(
            s.contains("3") && s.contains("5"),
            "ShapeMismatch must include both shapes"
        );
    }

    // -----------------------------------------------------------------------
    // Test 25: aux_loss_alpha is small (per paper recommendation)
    // -----------------------------------------------------------------------
    #[test]
    fn test_aux_loss_alpha_small() {
        let cfg = DeepSeekV2Config::default();
        assert!(
            cfg.aux_loss_alpha < 0.01,
            "aux_loss_alpha ({}) should be small (< 0.01)",
            cfg.aux_loss_alpha
        );
        assert!(cfg.aux_loss_alpha > 0.0, "aux_loss_alpha must be positive");
    }
}
