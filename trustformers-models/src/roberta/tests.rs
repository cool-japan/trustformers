#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::roberta::config::RobertaConfig;
    use crate::roberta::model::RobertaModel;
    use crate::roberta::tasks::{
        RobertaForMaskedLM, RobertaForQuestionAnswering, RobertaForSequenceClassification,
        RobertaForTokenClassification,
    };
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
            self.state = self.state
                .wrapping_mul(6_364_136_223_846_793_005_u64)
                .wrapping_add(1_442_695_040_888_963_407_u64);
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1_u64 << 53) as f32
        }
    }

    // ── Minimal config ────────────────────────────────────────────────────────

    fn minimal_roberta_config() -> RobertaConfig {
        RobertaConfig {
            vocab_size: 512,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            intermediate_size: 256,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 128,
            type_vocab_size: 1,
            initializer_range: 0.02,
            layer_norm_eps: 1e-5,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            position_embedding_type: Some("absolute".to_string()),
            use_cache: Some(true),
            classifier_dropout: None,
        }
    }

    // ── Default config tests ──────────────────────────────────────────────────

    #[test]
    fn test_roberta_default_config_is_valid() {
        let config = RobertaConfig::default();
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_roberta_default_config_params() {
        let config = RobertaConfig::default();
        assert_eq!(config.vocab_size, 50265);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.hidden_act, "gelu");
        drop(config);
        std::hint::black_box(());
    }

    // ── Preset configs ────────────────────────────────────────────────────────

    #[test]
    fn test_roberta_base_config() {
        let config = RobertaConfig::roberta_base();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.num_attention_heads, 12);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_roberta_large_config() {
        let config = RobertaConfig::roberta_large();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.intermediate_size, 4096);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    // ── Architecture string ───────────────────────────────────────────────────

    #[test]
    fn test_roberta_architecture_string() {
        let config = RobertaConfig::default();
        assert_eq!(config.architecture(), "RoBERTa");
    }

    // ── Validation failure tests ──────────────────────────────────────────────

    #[test]
    fn test_roberta_invalid_hidden_size_not_divisible_by_heads() {
        let mut config = minimal_roberta_config();
        config.hidden_size = 65; // not divisible by 8
        assert!(config.validate().is_err());
    }

    // ── Token IDs ─────────────────────────────────────────────────────────────

    #[test]
    fn test_roberta_token_ids() {
        let config = RobertaConfig::default();
        assert_eq!(config.pad_token_id, 1);
        assert_eq!(config.bos_token_id, 0);
        assert_eq!(config.eos_token_id, 2);
    }

    #[test]
    fn test_roberta_max_position_embeddings() {
        let config = RobertaConfig::default();
        assert_eq!(config.max_position_embeddings, 514);
    }

    // ── Optional fields ───────────────────────────────────────────────────────

    #[test]
    fn test_roberta_position_embedding_type() {
        let config = RobertaConfig::default();
        assert_eq!(config.position_embedding_type, Some("absolute".to_string()));
    }

    #[test]
    fn test_roberta_use_cache_default() {
        let config = RobertaConfig::default();
        assert_eq!(config.use_cache, Some(true));
    }

    #[test]
    fn test_roberta_classifier_dropout_fallback() {
        let config = minimal_roberta_config();
        let resolved = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);
        assert_eq!(resolved, config.hidden_dropout_prob);
    }

    #[test]
    fn test_roberta_classifier_dropout_explicit() {
        let mut config = minimal_roberta_config();
        config.classifier_dropout = Some(0.3);
        let resolved = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);
        assert!((resolved - 0.3).abs() < 1e-6);
    }

    // ── Model creation tests ──────────────────────────────────────────────────

    #[test]
    fn test_roberta_model_creation_minimal() {
        let config = minimal_roberta_config();
        let model = RobertaModel::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_roberta_for_sequence_classification_creation() {
        let config = minimal_roberta_config();
        let num_labels = 2usize;
        let model = RobertaForSequenceClassification::new(config, num_labels);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_roberta_for_token_classification_creation() {
        let config = minimal_roberta_config();
        let num_labels = 5usize;
        let model = RobertaForTokenClassification::new(config, num_labels);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_roberta_for_question_answering_creation() {
        let config = minimal_roberta_config();
        let model = RobertaForQuestionAnswering::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_roberta_for_masked_lm_creation() {
        let config = minimal_roberta_config();
        let model = RobertaForMaskedLM::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_roberta_for_sequence_classification_device() {
        let config = minimal_roberta_config();
        let num_labels = 3usize;
        let model = RobertaForSequenceClassification::new(config, num_labels);
        if let Ok(m) = model {
            // Device should default to CPU
            let _ = m.device();
            drop(m);
        }
        std::hint::black_box(());
    }

    // ── Config cloning ────────────────────────────────────────────────────────

    #[test]
    fn test_roberta_config_clone() {
        let config = minimal_roberta_config();
        let cloned = config.clone();
        assert_eq!(config.vocab_size, cloned.vocab_size);
        assert_eq!(config.hidden_size, cloned.hidden_size);
        assert_eq!(config.num_attention_heads, cloned.num_attention_heads);
        drop(config);
        drop(cloned);
        std::hint::black_box(());
    }

    // ── Multiple labels ───────────────────────────────────────────────────────

    #[test]
    fn test_roberta_sequence_classification_multi_class() {
        let config = minimal_roberta_config();
        let model = RobertaForSequenceClassification::new(config, 10);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_roberta_token_classification_bio_tags() {
        let config = minimal_roberta_config();
        // B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O = 9 tags
        let model = RobertaForTokenClassification::new(config, 9);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    // ── LCG reproducibility ───────────────────────────────────────────────────

    #[test]
    fn test_lcg_reproducibility() {
        let mut rng1 = Lcg::new(12345);
        let mut rng2 = Lcg::new(12345);
        for _ in 0..40 {
            assert_eq!(rng1.next_f32(), rng2.next_f32());
        }
    }

    #[test]
    fn test_lcg_range() {
        let mut rng = Lcg::new(99999);
        for _ in 0..100 {
            let v = rng.next_f32();
            assert!(v >= 0.0 && v < 1.0);
        }
    }
}
