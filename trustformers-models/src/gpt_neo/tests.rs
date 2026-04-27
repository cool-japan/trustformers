#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::gpt_neo::config::GptNeoConfig;
    use crate::gpt_neo::model::{GptNeoLMHeadModel, GptNeoModel};
    use trustformers_core::traits::Config;

    // ── LCG ───────────────────────────────────────────────────────────────────
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }

        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005_u64)
                .wrapping_add(1_442_695_040_888_963_407_u64);
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1_u64 << 53) as f32
        }
    }

    // ── Minimal config ────────────────────────────────────────────────────────

    fn minimal_gpt_neo_config() -> GptNeoConfig {
        GptNeoConfig {
            vocab_size: 512,
            hidden_size: 64,
            num_layers: 2,
            attention_types: vec!["global".to_string(), "local".to_string()],
            num_heads: 8,
            intermediate_size: 256,
            window_size: 32,
            activation_function: "gelu_new".to_string(),
            resid_dropout: 0.0,
            embed_dropout: 0.0,
            attention_dropout: 0.0,
            max_position_embeddings: 128,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            use_cache: true,
            bos_token_id: 50256,
            eos_token_id: 50256,
            model_type: "gpt_neo".to_string(),
        }
    }

    // ── Default config tests ──────────────────────────────────────────────────

    #[test]
    fn test_gpt_neo_default_config_is_valid() {
        let config = GptNeoConfig::default();
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_default_config_params() {
        let config = GptNeoConfig::default();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.window_size, 256);
        assert_eq!(config.model_type, "gpt_neo");
        drop(config);
        std::hint::black_box(());
    }

    // ── Preset configs ────────────────────────────────────────────────────────

    #[test]
    fn test_gpt_neo_125m_config() {
        let config = GptNeoConfig::gpt_neo_125m();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.attention_types.len(), 12);
        // Alternating global/local
        assert_eq!(config.attention_types[0], "global");
        assert_eq!(config.attention_types[1], "local");
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_1_3b_config() {
        let config = GptNeoConfig::gpt_neo_1_3b();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.intermediate_size, 8192);
        assert_eq!(config.attention_types.len(), 24);
        // Even indices are global
        assert_eq!(config.attention_types[0], "global");
        assert_eq!(config.attention_types[1], "local");
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_2_7b_config() {
        let config = GptNeoConfig::gpt_neo_2_7b();
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 20);
        assert_eq!(config.intermediate_size, 10240);
        assert_eq!(config.attention_types.len(), 32);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    // ── Architecture string ───────────────────────────────────────────────────

    #[test]
    fn test_gpt_neo_architecture_string() {
        let config = GptNeoConfig::default();
        assert_eq!(config.architecture(), "GPT-Neo");
    }

    // ── Validation failure tests ──────────────────────────────────────────────

    #[test]
    fn test_gpt_neo_invalid_hidden_size_not_divisible_by_heads() {
        let mut config = minimal_gpt_neo_config();
        config.hidden_size = 65; // not divisible by 8
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gpt_neo_invalid_empty_attention_types() {
        let mut config = minimal_gpt_neo_config();
        config.attention_types = vec![];
        assert!(config.validate().is_err());
    }

    // ── from_pretrained_name tests ────────────────────────────────────────────

    #[test]
    fn test_gpt_neo_from_pretrained_name_125m() {
        let config = GptNeoConfig::from_pretrained_name("gpt-neo-125M");
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_heads, 12);
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_from_pretrained_name_1_3b() {
        let config = GptNeoConfig::from_pretrained_name("gpt-neo-1.3b");
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_heads, 16);
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_from_pretrained_name_1_3b_underscore() {
        let config = GptNeoConfig::from_pretrained_name("gpt-neo-1_3b");
        assert_eq!(config.hidden_size, 2048);
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_from_pretrained_name_2_7b() {
        let config = GptNeoConfig::from_pretrained_name("gpt-neo-2.7b");
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_heads, 20);
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_from_pretrained_name_unknown_defaults_to_125m() {
        let config = GptNeoConfig::from_pretrained_name("unknown-variant");
        assert_eq!(config.hidden_size, 768);
        drop(config);
        std::hint::black_box(());
    }

    // ── Attention types alternation ───────────────────────────────────────────

    #[test]
    fn test_gpt_neo_125m_attention_type_pattern() {
        let config = GptNeoConfig::gpt_neo_125m();
        for (i, attn_type) in config.attention_types.iter().enumerate() {
            let expected = if i % 2 == 0 { "global" } else { "local" };
            assert_eq!(attn_type.as_str(), expected, "layer {} mismatch", i);
        }
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_1_3b_attention_type_pattern() {
        let config = GptNeoConfig::gpt_neo_1_3b();
        for (i, attn_type) in config.attention_types.iter().enumerate() {
            let expected = if i % 2 == 0 { "global" } else { "local" };
            assert_eq!(attn_type.as_str(), expected, "layer {} mismatch", i);
        }
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_attention_types_count_matches_layers_125m() {
        let config = GptNeoConfig::gpt_neo_125m();
        assert_eq!(config.attention_types.len(), config.num_layers);
    }

    #[test]
    fn test_gpt_neo_attention_types_count_matches_layers_1_3b() {
        let config = GptNeoConfig::gpt_neo_1_3b();
        assert_eq!(config.attention_types.len(), config.num_layers);
    }

    #[test]
    fn test_gpt_neo_attention_types_count_matches_layers_2_7b() {
        let config = GptNeoConfig::gpt_neo_2_7b();
        assert_eq!(config.attention_types.len(), config.num_layers);
    }

    // ── Model creation tests ──────────────────────────────────────────────────

    #[test]
    fn test_gpt_neo_model_creation_minimal() {
        let config = minimal_gpt_neo_config();
        let model = GptNeoModel::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_gpt_neo_lm_head_model_creation_minimal() {
        let config = minimal_gpt_neo_config();
        let model = GptNeoLMHeadModel::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    // ── Token IDs ─────────────────────────────────────────────────────────────

    #[test]
    fn test_gpt_neo_125m_token_ids() {
        let config = GptNeoConfig::gpt_neo_125m();
        assert_eq!(config.bos_token_id, 50256);
        assert_eq!(config.eos_token_id, 50256);
    }

    #[test]
    fn test_gpt_neo_dropout_values_in_pretrained() {
        let config = GptNeoConfig::gpt_neo_125m();
        // Pre-trained configs use 0.0 dropout
        assert_eq!(config.resid_dropout, 0.0);
        assert_eq!(config.embed_dropout, 0.0);
        assert_eq!(config.attention_dropout, 0.0);
    }

    // ── Config cloning ────────────────────────────────────────────────────────

    #[test]
    fn test_gpt_neo_config_clone() {
        let config = minimal_gpt_neo_config();
        let cloned = config.clone();
        assert_eq!(config.vocab_size, cloned.vocab_size);
        assert_eq!(config.hidden_size, cloned.hidden_size);
        assert_eq!(config.attention_types.len(), cloned.attention_types.len());
        drop(config);
        drop(cloned);
        std::hint::black_box(());
    }

    // ── LCG reproducibility ───────────────────────────────────────────────────

    #[test]
    fn test_lcg_reproducibility() {
        let mut rng1 = Lcg::new(1111);
        let mut rng2 = Lcg::new(1111);
        for _ in 0..30 {
            assert_eq!(rng1.next_f32(), rng2.next_f32());
        }
    }

    #[test]
    fn test_lcg_range() {
        let mut rng = Lcg::new(7777);
        for _ in 0..100 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
