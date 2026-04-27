mod config;
mod model;
mod tasks;

pub use config::{Jamba2Config, Jamba2ConfigError, LayerType};
pub use model::{Jamba2Attention, Jamba2DecoderLayer, Jamba2Error, Jamba2Model, MambaBlock};
pub use tasks::{CausalLmOutput, Jamba2ForCausalLM, Jamba2TaskError};

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: default config values
    #[test]
    fn test_default_config_values() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.vocab_size, 65536);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 14336);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.mamba_d_state, 16);
        assert_eq!(cfg.mamba_d_conv, 4);
        assert_eq!(cfg.mamba_expand, 2);
        assert_eq!(cfg.mamba_dt_rank, 256);
        assert_eq!(cfg.attn_layer_offset, 4);
        assert_eq!(cfg.attn_layer_period, 8);
        assert_eq!(cfg.expert_layer_offset, 1);
        assert_eq!(cfg.expert_layer_period, 2);
        assert_eq!(cfg.num_experts, 16);
        assert_eq!(cfg.num_experts_per_tok, 2);
        assert_eq!(cfg.max_position_embeddings, 262144);
        assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-10);
        assert!((cfg.rope_theta - 10000.0).abs() < 1e-5);
        assert_eq!(cfg.hidden_act, "silu");
        assert!(!cfg.tie_word_embeddings);
    }

    // Test 2: is_attention_layer at various indices
    #[test]
    fn test_is_attention_layer() {
        let cfg = Jamba2Config::default();
        // offset=4, period=8 → attention at 4, 12, 20, 28
        assert!(!cfg.is_attention_layer(0));
        assert!(!cfg.is_attention_layer(1));
        assert!(!cfg.is_attention_layer(3));
        assert!(cfg.is_attention_layer(4));
        assert!(!cfg.is_attention_layer(5));
        assert!(!cfg.is_attention_layer(11));
        assert!(cfg.is_attention_layer(12));
        assert!(cfg.is_attention_layer(20));
        assert!(cfg.is_attention_layer(28));
        assert!(!cfg.is_attention_layer(29));
    }

    // Test 3: is_moe_layer at various indices
    #[test]
    fn test_is_moe_layer() {
        let cfg = Jamba2Config::default();
        // offset=1, period=2 → MoE at 1, 3, 5, 7, 9, ...
        assert!(!cfg.is_moe_layer(0));
        assert!(cfg.is_moe_layer(1));
        assert!(!cfg.is_moe_layer(2));
        assert!(cfg.is_moe_layer(3));
        assert!(!cfg.is_moe_layer(4));
        assert!(cfg.is_moe_layer(5));
        assert!(cfg.is_moe_layer(11));
    }

    // Test 4: layer_type enum
    #[test]
    fn test_layer_type() {
        let cfg = Jamba2Config::default();
        // Layer 0: not attn (0 < 4), not moe (0 < 1) → Mamba
        assert_eq!(cfg.layer_type(0), LayerType::Mamba);
        // Layer 1: not attn, moe → MambaMoE
        assert_eq!(cfg.layer_type(1), LayerType::MambaMoE);
        // Layer 2: not attn, not moe → Mamba
        assert_eq!(cfg.layer_type(2), LayerType::Mamba);
        // Layer 3: not attn, moe → MambaMoE
        assert_eq!(cfg.layer_type(3), LayerType::MambaMoE);
        // Layer 4: attn, not moe (4 is even, offset=1 period=2: (4-1)%2=1≠0) → Attention
        assert_eq!(cfg.layer_type(4), LayerType::Attention);
        // Layer 5: not attn, moe → MambaMoE
        assert_eq!(cfg.layer_type(5), LayerType::MambaMoE);
        // Layer 12: attn, (12-1)%2=1≠0 → Attention
        assert_eq!(cfg.layer_type(12), LayerType::Attention);
    }

    // Test 5: mamba_inner_dim
    #[test]
    fn test_mamba_inner_dim() {
        let cfg = Jamba2Config::default();
        // expand=2, hidden=4096 → inner=8192
        assert_eq!(cfg.mamba_inner_dim(), 8192);
        let cfg_small = Jamba2Config {
            hidden_size: 512,
            mamba_expand: 4,
            ..Jamba2Config::default()
        };
        assert_eq!(cfg_small.mamba_inner_dim(), 2048);
    }

    // Test 6: jamba2_1_5b preset
    #[test]
    fn test_jamba2_1_5b_preset() {
        let cfg = Jamba2Config::jamba2_1_5b();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.mamba_dt_rank, 128);
        assert_eq!(cfg.mamba_inner_dim(), 4096);
    }

    // Test 7: validate catches invalid configs
    #[test]
    fn test_validate() {
        let valid = Jamba2Config::default();
        assert!(valid.validate().is_ok());

        let bad_vocab = Jamba2Config {
            vocab_size: 0,
            ..Jamba2Config::default()
        };
        assert!(bad_vocab.validate().is_err());

        let bad_heads = Jamba2Config {
            num_attention_heads: 7,
            num_key_value_heads: 3, // 7 % 3 != 0
            ..Jamba2Config::default()
        };
        assert!(bad_heads.validate().is_err());

        let bad_experts = Jamba2Config {
            num_experts_per_tok: 0,
            ..Jamba2Config::default()
        };
        assert!(bad_experts.validate().is_err());

        let bad_period = Jamba2Config {
            attn_layer_period: 0,
            ..Jamba2Config::default()
        };
        assert!(bad_period.validate().is_err());
    }

    // Test 8: layer combination coverage — some layers Mamba, some Attn
    #[test]
    fn test_layer_combination_coverage() {
        let cfg = Jamba2Config::default();
        let mut mamba_count = 0usize;
        let mut attn_count = 0usize;
        let mut mamba_moe_count = 0usize;
        let mut attn_moe_count = 0usize;

        for i in 0..cfg.num_hidden_layers {
            match cfg.layer_type(i) {
                LayerType::Mamba => mamba_count += 1,
                LayerType::Attention => attn_count += 1,
                LayerType::MambaMoE => mamba_moe_count += 1,
                LayerType::AttentionMoE => attn_moe_count += 1,
            }
        }

        // With 32 layers: offset=4, period=8 → attn at 4,12,20,28 (4 layers)
        assert_eq!(
            attn_count + attn_moe_count,
            4,
            "Should have 4 attention layers total"
        );
        // The remaining 28 layers are Mamba or MambaMoE
        assert_eq!(
            mamba_count + mamba_moe_count,
            28,
            "Should have 28 Mamba layers total"
        );
        // Total must be 32
        assert_eq!(
            mamba_count + attn_count + mamba_moe_count + attn_moe_count,
            32
        );
    }

    // ── Helper: minimal config for fast forward/generate tests ────────────────
    fn small_jamba2_config() -> Jamba2Config {
        Jamba2Config {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            mamba_d_state: 4,
            mamba_d_conv: 2,
            mamba_expand: 2,
            mamba_dt_rank: 4,
            attn_layer_offset: 1,
            attn_layer_period: 2,
            expert_layer_offset: 1,
            expert_layer_period: 2,
            num_experts: 4,
            num_experts_per_tok: 2,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            hidden_act: "silu".to_string(),
            attention_dropout: 0.0,
            tie_word_embeddings: false,
        }
    }

    // Test 9: jamba2_1_5b validate passes
    #[test]
    fn test_jamba2_1_5b_validate_ok() {
        let cfg = Jamba2Config::jamba2_1_5b();
        assert!(cfg.validate().is_ok(), "jamba2_1_5b should pass validation");
    }

    // Test 10: jamba2_1_5b layer_type at index 0 is Mamba
    #[test]
    fn test_jamba2_1_5b_layer_types() {
        let cfg = Jamba2Config::jamba2_1_5b();
        // offset=4, period=8: attention at 4 (only one in 12 layers)
        // moe offset=1, period=2: moe at 1,3,5,7,9,11
        assert_eq!(cfg.layer_type(0), LayerType::Mamba);
        assert_eq!(cfg.layer_type(1), LayerType::MambaMoE);
        assert_eq!(cfg.layer_type(2), LayerType::Mamba);
        assert_eq!(cfg.layer_type(4), LayerType::Attention);
        // Layer 5: is_moe(5): (5-1)%2=0 → moe; is_attn(5): (5-4)%8=1≠0 → not attn → MambaMoE
        assert_eq!(cfg.layer_type(5), LayerType::MambaMoE);
    }

    // Test 11: effective_dt_rank when mamba_dt_rank=0 → ceil(hidden_size/16)
    #[test]
    fn test_effective_dt_rank_auto_compute() {
        let cfg = Jamba2Config {
            hidden_size: 512,
            mamba_dt_rank: 0,
            ..Jamba2Config::default()
        };
        // ceil(512/16) = 32
        assert_eq!(cfg.effective_dt_rank(), 32);
    }

    // Test 12: effective_dt_rank when explicit value provided
    #[test]
    fn test_effective_dt_rank_explicit() {
        let cfg = Jamba2Config::default();
        // mamba_dt_rank=256 explicitly set
        assert_eq!(cfg.effective_dt_rank(), 256);
    }

    // Test 13: mamba_inner_dim for 1_5b preset
    #[test]
    fn test_jamba2_1_5b_mamba_inner_dim() {
        let cfg = Jamba2Config::jamba2_1_5b();
        // expand=2, hidden=2048 → inner=4096
        assert_eq!(cfg.mamba_inner_dim(), 4096);
    }

    // Test 14: AttentionMoE layer type exists somewhere (find a layer that is both attn and moe)
    #[test]
    fn test_attn_moe_layer_exists_in_custom_config() {
        // Create a config where attn and moe coincide at layer 1
        // attn_layer_offset=1, attn_layer_period=4: attn at 1,5,9,...
        // expert_layer_offset=1, expert_layer_period=2: moe at 1,3,5,7,...
        // Layer 1: attn=true (offset=1, period=4: (1-1)%4=0), moe=true → AttentionMoE
        let cfg = Jamba2Config {
            attn_layer_offset: 1,
            attn_layer_period: 4,
            expert_layer_offset: 1,
            expert_layer_period: 2,
            ..Jamba2Config::default()
        };
        assert_eq!(cfg.layer_type(1), LayerType::AttentionMoE);
    }

    // Test 15: num_experts default = 16
    #[test]
    fn test_num_experts_default() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.num_experts, 16);
    }

    // Test 16: num_experts_per_tok default = 2
    #[test]
    fn test_num_experts_per_tok_default() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.num_experts_per_tok, 2);
    }

    // Test 17: mamba_d_state default = 16
    #[test]
    fn test_mamba_d_state_default() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.mamba_d_state, 16);
    }

    // Test 18: mamba_d_conv default = 4
    #[test]
    fn test_mamba_d_conv_default() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.mamba_d_conv, 4);
    }

    // Test 19: max_position_embeddings = 262144
    #[test]
    fn test_max_position_embeddings_default() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.max_position_embeddings, 262144);
    }

    // Test 20: config clone preserves all key fields
    #[test]
    fn test_config_clone() {
        let cfg = Jamba2Config::default();
        let cloned = cfg.clone();
        assert_eq!(cloned.vocab_size, cfg.vocab_size);
        assert_eq!(cloned.hidden_size, cfg.hidden_size);
        assert_eq!(cloned.num_experts, cfg.num_experts);
        assert_eq!(cloned.mamba_d_state, cfg.mamba_d_state);
        assert_eq!(cloned.attn_layer_period, cfg.attn_layer_period);
        assert_eq!(cloned.tie_word_embeddings, cfg.tie_word_embeddings);
    }

    // Test 21: config debug format contains key field names
    #[test]
    fn test_config_debug() {
        let cfg = Jamba2Config::default();
        let s = format!("{:?}", cfg);
        assert!(
            s.contains("Jamba2Config"),
            "debug must contain type name, got: {s}"
        );
        assert!(
            s.contains("vocab_size"),
            "debug must contain vocab_size, got: {s}"
        );
        assert!(
            s.contains("num_experts"),
            "debug must contain num_experts, got: {s}"
        );
    }

    // Test 22: validate fails when hidden_size = 0
    #[test]
    fn test_validate_fails_hidden_size_zero() {
        let cfg = Jamba2Config {
            hidden_size: 0,
            ..Jamba2Config::default()
        };
        assert!(cfg.validate().is_err(), "hidden_size=0 should fail");
    }

    // Test 23: validate fails when num_hidden_layers = 0
    #[test]
    fn test_validate_fails_num_hidden_layers_zero() {
        let cfg = Jamba2Config {
            num_hidden_layers: 0,
            ..Jamba2Config::default()
        };
        assert!(cfg.validate().is_err(), "num_hidden_layers=0 should fail");
    }

    // Test 24: validate fails when mamba_expand = 0
    #[test]
    fn test_validate_fails_mamba_expand_zero() {
        let cfg = Jamba2Config {
            mamba_expand: 0,
            ..Jamba2Config::default()
        };
        assert!(cfg.validate().is_err(), "mamba_expand=0 should fail");
    }

    // Test 25: validate fails when num_experts_per_tok > num_experts
    #[test]
    fn test_validate_fails_experts_per_tok_exceeds_num_experts() {
        let cfg = Jamba2Config {
            num_experts: 4,
            num_experts_per_tok: 8,
            ..Jamba2Config::default()
        };
        assert!(
            cfg.validate().is_err(),
            "experts_per_tok > num_experts should fail"
        );
    }

    // Test 26: validate fails when expert_layer_period = 0
    #[test]
    fn test_validate_fails_expert_layer_period_zero() {
        let cfg = Jamba2Config {
            expert_layer_period: 0,
            ..Jamba2Config::default()
        };
        assert!(cfg.validate().is_err(), "expert_layer_period=0 should fail");
    }

    // Test 27: Jamba2Error display — EmptyInput
    #[test]
    fn test_jamba2_error_display_empty_input() {
        let err = Jamba2Error::EmptyInput;
        let s = err.to_string();
        assert!(
            s.to_lowercase().contains("empty"),
            "EmptyInput display should mention 'empty', got: {s}"
        );
    }

    // Test 28: Jamba2Error display — DimensionMismatch
    #[test]
    fn test_jamba2_error_display_dimension_mismatch() {
        let err = Jamba2Error::DimensionMismatch {
            expected: 64,
            got: 32,
            context: "test_context".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("64"), "should contain expected value, got: {s}");
        assert!(s.contains("32"), "should contain got value, got: {s}");
        assert!(
            s.contains("test_context"),
            "should contain context, got: {s}"
        );
    }

    // Test 29: Jamba2TaskError display — Config variant
    #[test]
    fn test_jamba2_task_error_display_config_variant() {
        let config_err = Jamba2ConfigError::InvalidField("bad field".to_string());
        let task_err: Jamba2TaskError = config_err.into();
        let s = task_err.to_string();
        assert!(
            s.contains("bad field") || s.to_lowercase().contains("config"),
            "Config task error should mention config or field, got: {s}"
        );
    }

    // Test 30: CausalLM forward on small config returns correct seq_len in logits
    #[test]
    fn test_causal_lm_forward_logits_shape() {
        let cfg = small_jamba2_config();
        let lm = Jamba2ForCausalLM::new(cfg.clone()).expect("Jamba2ForCausalLM::new");
        let input = &[1u32, 2, 3];
        let output = lm.forward(input).expect("forward");
        // logits must have seq_len entries, each of vocab_size
        assert_eq!(
            output.logits.len(),
            3,
            "logits seq_len must match input length"
        );
        assert_eq!(
            output.logits[0].len(),
            cfg.vocab_size,
            "each logit row must have vocab_size entries"
        );
    }

    // Test 31: CausalLM forward hidden_states shape
    #[test]
    fn test_causal_lm_forward_hidden_states_shape() {
        let cfg = small_jamba2_config();
        let lm = Jamba2ForCausalLM::new(cfg.clone()).expect("Jamba2ForCausalLM::new");
        let output = lm.forward(&[5u32, 6]).expect("forward");
        assert_eq!(
            output.hidden_states.len(),
            2,
            "hidden_states must match seq_len"
        );
        assert_eq!(
            output.hidden_states[0].len(),
            cfg.hidden_size,
            "hidden state width must equal hidden_size"
        );
    }

    // Test 32: CausalLM generate returns correct count of new tokens
    #[test]
    fn test_causal_lm_generate_token_count() {
        let cfg = small_jamba2_config();
        let lm = Jamba2ForCausalLM::new(cfg).expect("Jamba2ForCausalLM::new");
        let generated = lm.generate(&[1u32, 2], 5).expect("generate");
        assert_eq!(
            generated.len(),
            5,
            "generate must return exactly max_new_tokens tokens"
        );
    }

    // Test 33: CausalLM generate empty input returns error
    #[test]
    fn test_causal_lm_generate_empty_input_error() {
        let cfg = small_jamba2_config();
        let lm = Jamba2ForCausalLM::new(cfg).expect("Jamba2ForCausalLM::new");
        let result = lm.generate(&[], 3);
        assert!(result.is_err(), "generate with empty input must return Err");
    }

    // Test 34: SSM layer count in default config (32 - 4 attn = 28 total mamba layers)
    #[test]
    fn test_ssm_layer_count_default() {
        let cfg = Jamba2Config::default();
        let ssm_count = (0..cfg.num_hidden_layers).filter(|&i| !cfg.is_attention_layer(i)).count();
        assert_eq!(
            ssm_count, 28,
            "default config must have 28 SSM (Mamba) layers"
        );
    }

    // Test 35: attention layer count in 1_5b config
    #[test]
    fn test_attention_layer_count_1_5b() {
        let cfg = Jamba2Config::jamba2_1_5b();
        // offset=4, period=8, num_layers=12: attn at index 4 only
        let attn_count = (0..cfg.num_hidden_layers).filter(|&i| cfg.is_attention_layer(i)).count();
        assert_eq!(attn_count, 1, "jamba2_1_5b must have 1 attention layer");
    }

    // Test 36: vocab_size default = 65536
    #[test]
    fn test_vocab_size_default() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.vocab_size, 65536);
    }

    // Test 37: vocab_size 1_5b = 65536
    #[test]
    fn test_vocab_size_1_5b() {
        let cfg = Jamba2Config::jamba2_1_5b();
        assert_eq!(cfg.vocab_size, 65536);
    }

    // Test 38: GQA ratio default = num_attention_heads / num_key_value_heads = 32/8 = 4
    #[test]
    fn test_gqa_ratio_default() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.num_attention_heads / cfg.num_key_value_heads, 4);
    }

    // Test 39: mamba_dt_rank 1_5b = ceil(2048/16) = 128
    #[test]
    fn test_mamba_dt_rank_1_5b_matches_auto() {
        let cfg = Jamba2Config::jamba2_1_5b();
        let auto = cfg.hidden_size.div_ceil(16);
        assert_eq!(
            cfg.mamba_dt_rank, auto,
            "jamba2_1_5b dt_rank should equal ceil(hidden/16)"
        );
    }

    // Test 40: attention_dropout default = 0.0
    #[test]
    fn test_attention_dropout_default() {
        let cfg = Jamba2Config::default();
        assert!((cfg.attention_dropout - 0.0_f32).abs() < 1e-7);
    }

    // Test 41: LayerType variants are PartialEq-comparable
    #[test]
    fn test_layer_type_partial_eq() {
        assert_eq!(LayerType::Mamba, LayerType::Mamba);
        assert_eq!(LayerType::Attention, LayerType::Attention);
        assert_eq!(LayerType::MambaMoE, LayerType::MambaMoE);
        assert_eq!(LayerType::AttentionMoE, LayerType::AttentionMoE);
        assert_ne!(LayerType::Mamba, LayerType::Attention);
    }

    // Test 42: Jamba2Error display — LayerError
    #[test]
    fn test_jamba2_error_display_layer_error() {
        let err = Jamba2Error::LayerError {
            layer: 3,
            msg: "oops".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains('3'), "should mention layer index, got: {s}");
        assert!(s.contains("oops"), "should mention message, got: {s}");
    }

    // Test 43: CausalLM generate generates token ids within vocab range
    #[test]
    fn test_causal_lm_generate_tokens_within_vocab() {
        let cfg = small_jamba2_config();
        let vocab = cfg.vocab_size;
        let lm = Jamba2ForCausalLM::new(cfg).expect("Jamba2ForCausalLM::new");
        let generated = lm.generate(&[0u32], 4).expect("generate");
        for &tok in &generated {
            assert!(
                (tok as usize) < vocab,
                "generated token {tok} out of vocab range {vocab}"
            );
        }
    }

    // Test 44: Validate fails when num_attention_heads=0
    #[test]
    fn test_validate_fails_num_attention_heads_zero() {
        let cfg = Jamba2Config {
            num_attention_heads: 0,
            ..Jamba2Config::default()
        };
        assert!(
            cfg.validate().is_err(),
            "num_attention_heads=0 should fail validation"
        );
    }

    // Test 45: Validate fails when num_key_value_heads=0
    #[test]
    fn test_validate_fails_num_key_value_heads_zero() {
        let cfg = Jamba2Config {
            num_key_value_heads: 0,
            ..Jamba2Config::default()
        };
        assert!(
            cfg.validate().is_err(),
            "num_key_value_heads=0 should fail validation"
        );
    }
}
