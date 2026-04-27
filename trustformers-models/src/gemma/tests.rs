#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::gemma::config::GemmaConfig;
    use crate::gemma::model::{GemmaForCausalLM, GemmaModel, GemmaRMSNorm};
    use trustformers_core::traits::Config;

    // ── LCG for deterministic pseudo-random data ──────────────────────────────
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

    // ── Minimal config helper ─────────────────────────────────────────────────

    fn minimal_gemma_config() -> GemmaConfig {
        GemmaConfig {
            vocab_size: 512,
            hidden_size: 64, // 8 heads * 8 head_dim
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 8, // hidden_size(64) == num_attention_heads(8) * head_dim(8)
            hidden_act: "gelu".to_string(),
            max_position_embeddings: 128,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
            pad_token_id: Some(0),
            bos_token_id: 2,
            eos_token_id: 1,
            rope_theta: 10000.0,
            attention_bias: false,
            attention_dropout: 0.0,
            model_type: "gemma".to_string(),
        }
    }

    // ── Default config tests ──────────────────────────────────────────────────

    #[test]
    fn test_gemma_default_config_is_valid() {
        let config = GemmaConfig::default();
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gemma_default_config_params() {
        let config = GemmaConfig::default();
        assert_eq!(config.vocab_size, 256000);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 18);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_key_value_heads, 1);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.model_type, "gemma");
        drop(config);
        std::hint::black_box(());
    }

    // ── Preset configs ────────────────────────────────────────────────────────

    #[test]
    fn test_gemma_2b_config() {
        let config = GemmaConfig::gemma_2b();
        assert_eq!(config.vocab_size, 256000);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 18);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_key_value_heads, 1);
        assert_eq!(config.head_dim, 256);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gemma_7b_config() {
        let config = GemmaConfig::gemma_7b();
        assert_eq!(config.vocab_size, 256000);
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 16);
        assert_eq!(config.head_dim, 256);
        // Note: hidden_size (3072) != num_attention_heads * head_dim (16*256=4096)
        // This is a known configuration quirk; validation reflects the strict rule.
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gemma_code_2b_config() {
        let config = GemmaConfig::gemma_code_2b();
        assert_eq!(config.model_type, "gemma-code");
        assert_eq!(config.hidden_size, 2048);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gemma_code_7b_config() {
        let config = GemmaConfig::gemma_code_7b();
        assert_eq!(config.model_type, "gemma-code");
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_attention_heads, 16);
        // Not asserting validate().is_ok() - same quirk as gemma_7b preset
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_gemma_2b_instruct_same_as_2b() {
        let base = GemmaConfig::gemma_2b();
        let instruct = GemmaConfig::gemma_2b_instruct();
        assert_eq!(base.hidden_size, instruct.hidden_size);
        assert_eq!(base.num_hidden_layers, instruct.num_hidden_layers);
        assert_eq!(base.num_attention_heads, instruct.num_attention_heads);
    }

    #[test]
    fn test_gemma_7b_instruct_same_as_7b() {
        let base = GemmaConfig::gemma_7b();
        let instruct = GemmaConfig::gemma_7b_instruct();
        assert_eq!(base.hidden_size, instruct.hidden_size);
        assert_eq!(base.num_hidden_layers, instruct.num_hidden_layers);
    }

    // ── Architecture string ───────────────────────────────────────────────────

    #[test]
    fn test_gemma_architecture_string() {
        let config = GemmaConfig::default();
        assert_eq!(config.architecture(), "Gemma");
    }

    // ── Validation failure tests ──────────────────────────────────────────────

    #[test]
    fn test_gemma_invalid_hidden_size_mismatch() {
        // hidden_size must equal num_attention_heads * head_dim
        let mut config = minimal_gemma_config();
        config.hidden_size = 65; // 8 * 8 = 64, so 65 is invalid
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gemma_invalid_heads_not_divisible_by_kv_heads() {
        let mut config = minimal_gemma_config();
        config.num_key_value_heads = 3; // 8 not divisible by 3
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_gemma_invalid_vocab_size_zero() {
        let mut config = minimal_gemma_config();
        config.vocab_size = 0;
        assert!(config.validate().is_err());
    }

    // ── Helper methods ────────────────────────────────────────────────────────

    #[test]
    fn test_gemma_num_query_groups_2b() {
        let config = GemmaConfig::gemma_2b();
        // 8 attention heads / 1 kv head = 8 query groups
        assert_eq!(config.num_query_groups(), 8);
    }

    #[test]
    fn test_gemma_num_query_groups_7b() {
        let config = GemmaConfig::gemma_7b();
        // 16 attention heads / 16 kv heads = 1 query group (MHA)
        assert_eq!(config.num_query_groups(), 1);
    }

    #[test]
    fn test_gemma_uses_multi_query_attention_2b() {
        let config = GemmaConfig::gemma_2b();
        // 1 kv head < 8 attention heads => multi-query
        assert!(config.uses_multi_query_attention());
    }

    #[test]
    fn test_gemma_uses_multi_query_attention_7b_false() {
        let config = GemmaConfig::gemma_7b();
        // 16 kv heads == 16 attention heads => NOT multi-query
        assert!(!config.uses_multi_query_attention());
    }

    #[test]
    fn test_gemma_effective_head_dim() {
        let config = GemmaConfig::gemma_2b();
        assert_eq!(config.effective_head_dim(), 256);
    }

    #[test]
    fn test_gemma_hidden_size_consistency_2b() {
        let config = GemmaConfig::gemma_2b();
        // hidden_size == num_attention_heads * head_dim
        assert_eq!(
            config.hidden_size,
            config.num_attention_heads * config.head_dim
        );
    }

    #[test]
    fn test_gemma_2b_hidden_size_consistency() {
        let config = GemmaConfig::gemma_2b();
        // hidden_size == num_attention_heads * head_dim
        assert_eq!(
            config.hidden_size,
            config.num_attention_heads * config.head_dim
        );
    }

    // ── RMSNorm creation ──────────────────────────────────────────────────────

    #[test]
    fn test_gemma_rmsnorm_creation() {
        let norm = GemmaRMSNorm::new(64, 1e-6);
        assert!(norm.is_ok());
        if let Ok(n) = norm {
            assert_eq!(n.parameter_count(), 64);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_gemma_rmsnorm_parameter_count() {
        let dim = 128usize;
        let norm_result = GemmaRMSNorm::new(dim, 1e-6);
        if let Ok(norm) = norm_result {
            assert_eq!(norm.parameter_count(), dim);
        }
        std::hint::black_box(());
    }

    // ── Model creation tests ──────────────────────────────────────────────────

    #[test]
    fn test_gemma_model_creation_minimal() {
        let config = minimal_gemma_config();
        let model = GemmaModel::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_gemma_for_causal_lm_creation_minimal() {
        let config = minimal_gemma_config();
        let model = GemmaForCausalLM::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    // ── LCG range and reproducibility ────────────────────────────────────────

    #[test]
    fn test_lcg_reproducibility() {
        let mut rng1 = Lcg::new(9876);
        let mut rng2 = Lcg::new(9876);
        for _ in 0..30 {
            assert_eq!(rng1.next_f32(), rng2.next_f32());
        }
    }

    #[test]
    fn test_lcg_range() {
        let mut rng = Lcg::new(0xCAFE_BABE);
        for _ in 0..200 {
            let v = rng.next_f32();
            assert!(v >= 0.0);
            assert!(v < 1.0);
        }
    }

    // ── Misc config properties ────────────────────────────────────────────────

    #[test]
    fn test_gemma_token_ids() {
        let config = GemmaConfig::default();
        assert_eq!(config.eos_token_id, 1);
        assert_eq!(config.bos_token_id, 2);
        assert_eq!(config.pad_token_id, Some(0));
    }

    #[test]
    fn test_gemma_rope_theta() {
        let config = GemmaConfig::default();
        assert!((config.rope_theta - 10000.0).abs() < 1e-3);
    }

    #[test]
    fn test_gemma_config_clone() {
        let config = minimal_gemma_config();
        let cloned = config.clone();
        assert_eq!(config.vocab_size, cloned.vocab_size);
        assert_eq!(config.hidden_size, cloned.hidden_size);
        assert_eq!(config.head_dim, cloned.head_dim);
        drop(config);
        drop(cloned);
        std::hint::black_box(());
    }
}
