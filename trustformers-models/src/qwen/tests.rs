#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::qwen::config::QwenConfig;
    use crate::qwen::model::{QwenForCausalLM, QwenModel, QwenRMSNorm};
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

    fn minimal_qwen_config() -> QwenConfig {
        QwenConfig {
            vocab_size: 512,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: Some(2),
            hidden_act: "silu".to_string(),
            max_position_embeddings: 256,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
            pad_token_id: None,
            bos_token_id: 1,
            eos_token_id: 2,
            rope_theta: 1_000_000.0,
            rope_scaling: None,
            attention_dropout: 0.0,
            use_sliding_window: false,
            sliding_window: None,
            max_window_layers: None,
            use_logn_attn: false,
            logn_list: None,
            model_type: "qwen2".to_string(),
        }
    }

    // ── Default config tests ──────────────────────────────────────────────────

    #[test]
    fn test_qwen_default_config_is_valid() {
        let config = QwenConfig::default();
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen_default_config_params() {
        let config = QwenConfig::default();
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.model_type, "qwen2");
        drop(config);
        std::hint::black_box(());
    }

    // ── Preset configs ────────────────────────────────────────────────────────

    #[test]
    fn test_qwen2_0_5b_config() {
        let config = QwenConfig::qwen2_0_5b();
        assert_eq!(config.hidden_size, 896);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 14);
        assert_eq!(config.num_key_value_heads, Some(2));
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen2_1_5b_config() {
        let config = QwenConfig::qwen2_1_5b();
        assert_eq!(config.hidden_size, 1536);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.num_key_value_heads, Some(2));
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen2_7b_config() {
        let config = QwenConfig::qwen2_7b();
        assert_eq!(config.hidden_size, 3584);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 28);
        assert_eq!(config.num_key_value_heads, Some(4));
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen2_72b_config() {
        let config = QwenConfig::qwen2_72b();
        assert_eq!(config.hidden_size, 8192);
        assert_eq!(config.num_hidden_layers, 80);
        assert_eq!(config.num_attention_heads, 64);
        assert_eq!(config.num_key_value_heads, Some(8));
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen2_5_7b_config() {
        let config = QwenConfig::qwen2_5_7b();
        assert_eq!(config.max_position_embeddings, 131072);
        assert_eq!(config.model_type, "qwen2.5");
        assert!(config.is_qwen2_5());
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen2_5_14b_config() {
        let config = QwenConfig::qwen2_5_14b();
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.num_hidden_layers, 48);
        assert_eq!(config.model_type, "qwen2.5");
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen2_5_32b_config() {
        let config = QwenConfig::qwen2_5_32b();
        assert_eq!(config.num_hidden_layers, 64);
        assert_eq!(config.max_position_embeddings, 131072);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen2_5_72b_config() {
        let config = QwenConfig::qwen2_5_72b();
        assert_eq!(config.hidden_size, 8192);
        assert_eq!(config.num_hidden_layers, 80);
        assert_eq!(config.model_type, "qwen2.5");
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen2_5_coder_7b_config() {
        let config = QwenConfig::qwen2_5_coder_7b();
        assert_eq!(config.model_type, "qwen2.5-coder");
        assert_eq!(config.max_position_embeddings, 131072);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    // ── Architecture string ───────────────────────────────────────────────────

    #[test]
    fn test_qwen_architecture_string() {
        let config = QwenConfig::default();
        assert_eq!(config.architecture(), "Qwen");
    }

    // ── Validation failure tests ──────────────────────────────────────────────

    #[test]
    fn test_qwen_invalid_hidden_size_not_divisible_by_heads() {
        let mut config = minimal_qwen_config();
        config.hidden_size = 129; // 129 not divisible by 8
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_qwen_invalid_heads_not_divisible_by_kv_heads() {
        let mut config = minimal_qwen_config();
        config.num_key_value_heads = Some(3); // 8 not divisible by 3
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_qwen_invalid_vocab_size_zero() {
        let mut config = minimal_qwen_config();
        config.vocab_size = 0;
        assert!(config.validate().is_err());
    }

    // ── Helper method tests ───────────────────────────────────────────────────

    #[test]
    fn test_qwen_head_dim() {
        let config = QwenConfig::qwen2_7b();
        // head_dim = 3584 / 28 = 128
        assert_eq!(config.head_dim(), 128);
    }

    #[test]
    fn test_qwen_num_kv_heads_explicit() {
        let config = QwenConfig::qwen2_7b();
        assert_eq!(config.num_kv_heads(), 4);
    }

    #[test]
    fn test_qwen_num_kv_heads_fallback() {
        let mut config = minimal_qwen_config();
        config.num_key_value_heads = None;
        // Should fall back to num_attention_heads
        assert_eq!(config.num_kv_heads(), config.num_attention_heads);
    }

    #[test]
    fn test_qwen_num_query_groups() {
        let config = QwenConfig::qwen2_7b();
        // 28 heads / 4 kv heads = 7 groups
        assert_eq!(config.num_query_groups(), 7);
    }

    #[test]
    fn test_qwen_uses_grouped_query_attention_true() {
        let config = QwenConfig::qwen2_7b();
        assert!(config.uses_grouped_query_attention());
    }

    #[test]
    fn test_qwen_uses_grouped_query_attention_false_when_equal() {
        let config = QwenConfig::default(); // num_kv_heads = Some(32) == num_attention_heads = 32
        assert!(!config.uses_grouped_query_attention());
    }

    #[test]
    fn test_qwen_uses_grouped_query_attention_false_when_none() {
        let mut config = minimal_qwen_config();
        config.num_key_value_heads = None;
        assert!(!config.uses_grouped_query_attention());
    }

    #[test]
    fn test_qwen_sliding_window_false_by_default() {
        let config = QwenConfig::default();
        assert!(!config.uses_sliding_window());
    }

    #[test]
    fn test_qwen_sliding_window_true_when_both_set() {
        let mut config = minimal_qwen_config();
        config.use_sliding_window = true;
        config.sliding_window = Some(256);
        assert!(config.uses_sliding_window());
        assert_eq!(config.sliding_window_size(), 256);
    }

    #[test]
    fn test_qwen_sliding_window_false_when_flag_false() {
        let mut config = minimal_qwen_config();
        config.use_sliding_window = false;
        config.sliding_window = Some(256); // window set but flag false
        assert!(!config.uses_sliding_window());
    }

    #[test]
    fn test_qwen_sliding_window_size_falls_back_to_max_position() {
        let config = minimal_qwen_config(); // sliding_window = None
        assert_eq!(config.sliding_window_size(), config.max_position_embeddings);
    }

    #[test]
    fn test_qwen_is_qwen2_5_true() {
        let config = QwenConfig::qwen2_5_7b();
        assert!(config.is_qwen2_5());
    }

    #[test]
    fn test_qwen_is_qwen2_5_false_for_qwen2() {
        let config = QwenConfig::qwen2_7b();
        assert!(!config.is_qwen2_5());
    }

    // ── RMSNorm creation ──────────────────────────────────────────────────────

    #[test]
    fn test_qwen_rmsnorm_creation() {
        let norm = QwenRMSNorm::new(128, 1e-6);
        assert!(norm.is_ok());
        if let Ok(n) = norm {
            assert_eq!(n.parameter_count(), 128);
        }
        std::hint::black_box(());
    }

    // ── Model creation tests ──────────────────────────────────────────────────

    #[test]
    fn test_qwen_model_creation_minimal() {
        let config = minimal_qwen_config();
        let model = QwenModel::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_qwen_for_causal_lm_creation_minimal() {
        let config = minimal_qwen_config();
        let model = QwenForCausalLM::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    // ── LCG reproducibility ───────────────────────────────────────────────────

    #[test]
    fn test_lcg_reproducibility() {
        let mut rng1 = Lcg::new(0xDEAD_BEEF);
        let mut rng2 = Lcg::new(0xDEAD_BEEF);
        for _ in 0..50 {
            assert_eq!(rng1.next_f32(), rng2.next_f32());
        }
    }

    #[test]
    fn test_lcg_range() {
        let mut rng = Lcg::new(42);
        for _ in 0..200 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
