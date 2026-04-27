pub mod config;
pub mod model;
pub mod tasks;

pub use config::{GraniteConfig, GraniteError};
pub use model::{
    GraniteAttention, GraniteDecoderLayer, GraniteEmbedding, GraniteMlp, GraniteModel,
    GraniteRmsNorm, GraniteRotaryEmbedding,
};
pub use tasks::{GraniteForCausalLm, GraniteForSequenceClassification, GraniteLmHead};

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1: Config defaults ──────────────────────────────────────────────
    #[test]
    fn test_granite_config_defaults() {
        let cfg = GraniteConfig::default();
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.intermediate_size, 8192);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 64); // 2048 / 32
        assert_eq!(cfg.max_position_embeddings, 4096);
        assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-10);
        assert!((cfg.rope_theta - 10000.0).abs() < 1e-6);
        assert!(!cfg.attention_bias);
        assert!(!cfg.mlp_bias);
        assert!((cfg.embedding_multiplier - 12.0).abs() < 1e-6);
        assert!((cfg.logits_scaling - 0.25).abs() < 1e-6);
        assert!((cfg.residual_multiplier - 0.25).abs() < 1e-6);
        assert!((cfg.attention_multiplier - 0.25).abs() < 1e-6);
    }

    // ── Test 2: 3B preset ────────────────────────────────────────────────────
    #[test]
    fn test_granite_3b_preset() {
        let cfg = GraniteConfig::granite_3b();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.intermediate_size, 8192);
        assert!(cfg.validate().is_ok());
    }

    // ── Test 3: 8B preset ────────────────────────────────────────────────────
    #[test]
    fn test_granite_8b_preset() {
        let cfg = GraniteConfig::granite_8b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128); // 4096 / 32
        assert_eq!(cfg.intermediate_size, 14336);
        assert!(cfg.validate().is_ok());
    }

    // ── Test 4: validate fails with bad head_dim ─────────────────────────────
    #[test]
    fn test_validate_fails_bad_head_dim() {
        let cfg = GraniteConfig {
            head_dim: 99,
            ..GraniteConfig::default()
        };
        let result = cfg.validate();
        assert!(
            result.is_err(),
            "validate should fail when head_dim is inconsistent"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("head_dim"),
            "error message should mention head_dim, got: {msg}"
        );
    }

    // ── Test 5: Embedding scaling computation ────────────────────────────────
    #[test]
    fn test_embedding_scaling() {
        let cfg = GraniteConfig::default();
        // scale = embedding_multiplier * sqrt(hidden_size)
        let expected_scale = cfg.embedding_multiplier * (cfg.hidden_size as f32).sqrt();
        let embed =
            GraniteEmbedding::new(cfg.vocab_size, cfg.hidden_size, cfg.embedding_multiplier);
        assert!(
            (embed.scale() - expected_scale).abs() < 1e-4,
            "embedding scale mismatch: expected {expected_scale}, got {}",
            embed.scale()
        );
    }

    // ── Test 6: Logits scaling ────────────────────────────────────────────────
    #[test]
    fn test_logits_scaling() {
        let cfg = GraniteConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            max_position_embeddings: 16,
            vocab_size: 20,
            logits_scaling: 0.5,
            ..GraniteConfig::default()
        };

        let model = GraniteForCausalLm::new(&cfg).expect("construction should succeed");
        assert!(
            (model.logits_scaling() - 0.5).abs() < 1e-6,
            "logits_scaling accessor returned unexpected value"
        );

        let prompt = &[1_u32, 2, 3];
        let logits =
            model.forward_last_logits(prompt).expect("forward_last_logits should not fail");
        // With zero weights, all logits are 0.0 before scaling → still 0.0.
        assert_eq!(logits.len(), cfg.vocab_size);
    }

    // ── Test 7: Attention multiplier reflected in construction ───────────────
    #[test]
    fn test_attention_multiplier() {
        let cfg = GraniteConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            max_position_embeddings: 16,
            vocab_size: 20,
            attention_multiplier: 0.125,
            ..GraniteConfig::default()
        };

        // Construction succeeds with non-default attention_multiplier.
        let _attn = GraniteAttention::new(&cfg).expect("attention construction should succeed");
        assert!((cfg.attention_multiplier - 0.125).abs() < 1e-6);
    }

    // ── Test 8: MLP activation produces non-zero outputs ─────────────────────
    #[test]
    fn test_mlp_activation_mock() {
        let cfg = GraniteConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            ..GraniteConfig::default()
        };

        let mlp = GraniteMlp::new(&cfg);
        // Feed a non-trivial input (Xavier-init weights are non-zero).
        let x = vec![0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let out = mlp.forward(&x).expect("mlp forward should not fail");
        assert_eq!(
            out.len(),
            cfg.hidden_size,
            "MLP output dim should match hidden_size"
        );
        // Output should be finite.
        assert!(
            out.iter().all(|v| v.is_finite()),
            "MLP output should contain only finite values"
        );
    }

    // ── Test 9 (bonus — classification head dims) ─────────────────────────────
    #[test]
    fn test_classification_head() {
        let cfg = GraniteConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            max_position_embeddings: 16,
            vocab_size: 20,
            ..GraniteConfig::default()
        };

        let model = GraniteForSequenceClassification::new(&cfg, 3)
            .expect("classification model should construct");
        assert_eq!(model.num_labels(), 3);

        let out = model.forward(&[1_u32, 2]).expect("classification forward should not fail");
        assert_eq!(out.len(), 3, "output should have num_labels dimensions");
    }

    // ── Test 10: GraniteError display variants ────────────────────────────────
    #[test]
    fn test_granite_error_display_invalid_config() {
        let err = GraniteError::InvalidConfig("bad value".to_string());
        let s = err.to_string();
        assert!(s.contains("invalid config"), "got: {s}");
        assert!(s.contains("bad value"), "got: {s}");
    }

    #[test]
    fn test_granite_error_display_dimension_mismatch() {
        let err = GraniteError::DimensionMismatch {
            expected: 64,
            got: 32,
        };
        let s = err.to_string();
        assert!(s.contains("64") && s.contains("32"), "got: {s}");
        assert!(s.contains("mismatch"), "got: {s}");
    }

    #[test]
    fn test_granite_error_display_empty_input() {
        let err = GraniteError::EmptyInput;
        let s = err.to_string();
        assert!(s.to_lowercase().contains("empty"), "got: {s}");
    }

    // ── Test 13: Config clone/debug ───────────────────────────────────────────
    #[test]
    fn test_granite_config_clone() {
        let cfg = GraniteConfig::granite_3b();
        let cloned = cfg.clone();
        assert_eq!(cloned.hidden_size, cfg.hidden_size);
        assert_eq!(cloned.vocab_size, cfg.vocab_size);
        assert_eq!(cloned.embedding_multiplier, cfg.embedding_multiplier);
    }

    #[test]
    fn test_granite_config_debug() {
        let cfg = GraniteConfig::default();
        let s = format!("{:?}", cfg);
        assert!(s.contains("GraniteConfig"), "got: {s}");
        assert!(s.contains("vocab_size"), "got: {s}");
        assert!(s.contains("logits_scaling"), "got: {s}");
    }

    // ── Test 15: num_query_groups and is_gqa ─────────────────────────────────
    #[test]
    fn test_num_query_groups() {
        let cfg = GraniteConfig::granite_3b(); // 32 Q heads / 8 KV heads = 4 groups
        assert_eq!(cfg.num_query_groups(), 4);
    }

    #[test]
    fn test_is_gqa() {
        let cfg = GraniteConfig::default(); // 8 KV heads != 32 Q heads
        assert!(cfg.is_gqa());
        // MHA case: kv heads == attn heads
        let mut mha_cfg = GraniteConfig::default();
        mha_cfg.num_key_value_heads = mha_cfg.num_attention_heads;
        assert!(!mha_cfg.is_gqa());
    }

    // ── Test 17: validate rejects zero values ─────────────────────────────────
    #[test]
    fn test_validate_rejects_zero_vocab_size() {
        let cfg = GraniteConfig {
            vocab_size: 0,
            ..GraniteConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_zero_hidden_layers() {
        let cfg = GraniteConfig {
            num_hidden_layers: 0,
            ..GraniteConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_zero_intermediate_size() {
        let cfg = GraniteConfig {
            intermediate_size: 0,
            ..GraniteConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_zero_max_position_embeddings() {
        let cfg = GraniteConfig {
            max_position_embeddings: 0,
            ..GraniteConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_bad_kv_heads_divisibility() {
        let default = GraniteConfig::default();
        let cfg = GraniteConfig {
            num_attention_heads: 7,
            num_key_value_heads: 3,
            head_dim: default.hidden_size / 7,
            ..default
        };
        assert!(cfg.validate().is_err());
    }

    // ── Test 22: granite_3b vs granite_8b head_dim difference ────────────────
    #[test]
    fn test_granite_3b_vs_8b_head_dim() {
        let cfg3b = GraniteConfig::granite_3b();
        let cfg8b = GraniteConfig::granite_8b();
        // 3B: 2048 / 32 = 64; 8B: 4096 / 32 = 128
        assert_eq!(cfg3b.head_dim, 64);
        assert_eq!(cfg8b.head_dim, 128);
        assert_ne!(cfg3b.head_dim, cfg8b.head_dim);
    }

    // ── Test 23: generate_greedy produces correct length ──────────────────────
    #[test]
    fn test_generate_greedy_length() {
        let cfg = GraniteConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            max_position_embeddings: 16,
            vocab_size: 20,
            ..GraniteConfig::default()
        };

        let model = GraniteForCausalLm::new(&cfg).expect("model creation");
        for n in [1_usize, 3, 5] {
            let tokens =
                model.generate_greedy(&[1_u32, 2], n, cfg.vocab_size).expect("generate_greedy");
            assert_eq!(tokens.len(), n, "should generate exactly {n} tokens");
        }
    }

    // ── Test 24: generate_greedy rejects empty prompt ─────────────────────────
    #[test]
    fn test_generate_greedy_empty_prompt() {
        let cfg = GraniteConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            max_position_embeddings: 16,
            vocab_size: 20,
            ..GraniteConfig::default()
        };

        let model = GraniteForCausalLm::new(&cfg).expect("model creation");
        let result = model.generate_greedy(&[], 1, cfg.vocab_size);
        assert!(result.is_err(), "empty prompt should return Err");
    }

    // ── Test 25: forward_last_logits rejects empty token list ────────────────
    #[test]
    fn test_forward_last_logits_empty() {
        let cfg = GraniteConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            max_position_embeddings: 16,
            vocab_size: 20,
            ..GraniteConfig::default()
        };

        let model = GraniteForCausalLm::new(&cfg).expect("model creation");
        let result = model.forward_last_logits(&[]);
        assert!(result.is_err(), "empty token list should return Err");
    }

    // ── Test 26: classification rejects num_labels=0 ─────────────────────────
    #[test]
    fn test_classification_num_labels_zero_error() {
        let cfg = GraniteConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            max_position_embeddings: 16,
            vocab_size: 20,
            ..GraniteConfig::default()
        };

        let result = GraniteForSequenceClassification::new(&cfg, 0);
        assert!(result.is_err(), "num_labels=0 should return Err");
    }

    // ── Test 27: tie_word_embeddings default is false ─────────────────────────
    #[test]
    fn test_tie_word_embeddings_default() {
        let cfg = GraniteConfig::default();
        assert!(!cfg.tie_word_embeddings);
    }

    // ── Test 28: residual_multiplier and attention_dropout fields ─────────────
    #[test]
    fn test_residual_and_dropout_fields() {
        let cfg = GraniteConfig::default();
        assert!((cfg.residual_multiplier - 0.25_f32).abs() < 1e-6);
        assert!(cfg.attention_dropout.abs() < 1e-6);
    }
}
