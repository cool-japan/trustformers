#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::distilbert::config::DistilBertConfig;
    use crate::distilbert::model::DistilBertModel;
    use crate::distilbert::tasks::{
        DistilBertForMaskedLM, DistilBertForQuestionAnswering, DistilBertForSequenceClassification,
        DistilBertForTokenClassification,
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

    fn minimal_distilbert_config() -> DistilBertConfig {
        DistilBertConfig {
            vocab_size: 512,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            intermediate_size: 256,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 128,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: Some("absolute".to_string()),
            use_cache: Some(true),
            classifier_dropout: None,
            sinusoidal_pos_embds: false,
            tie_weights: Some(true),
        }
    }

    // ── Default config tests ──────────────────────────────────────────────────

    #[test]
    fn test_distilbert_default_config_is_valid() {
        let config = DistilBertConfig::default();
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_distilbert_default_config_params() {
        let config = DistilBertConfig::default();
        assert_eq!(config.vocab_size, 30522);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 6); // Half of BERT-base
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.hidden_act, "gelu");
        assert_eq!(config.max_position_embeddings, 512);
        drop(config);
        std::hint::black_box(());
    }

    // ── Preset configs ────────────────────────────────────────────────────────

    #[test]
    fn test_distilbert_base_config() {
        let config = DistilBertConfig::distilbert_base();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 768);
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_distilbert_base_cased_config() {
        let config = DistilBertConfig::distilbert_base_cased();
        assert!(config.validate().is_ok());
        assert_eq!(config.hidden_size, 768);
        drop(config);
        std::hint::black_box(());
    }

    // ── Architecture string ───────────────────────────────────────────────────

    #[test]
    fn test_distilbert_architecture_string() {
        let config = DistilBertConfig::default();
        assert_eq!(config.architecture(), "DistilBERT");
    }

    // ── Validation failure tests ──────────────────────────────────────────────

    #[test]
    fn test_distilbert_invalid_hidden_not_divisible_by_heads() {
        let mut config = minimal_distilbert_config();
        config.hidden_size = 65; // not divisible by 8
        assert!(config.validate().is_err());
    }

    // ── Optional fields ───────────────────────────────────────────────────────

    #[test]
    fn test_distilbert_sinusoidal_pos_embds_default_false() {
        let config = DistilBertConfig::default();
        assert!(!config.sinusoidal_pos_embds);
    }

    #[test]
    fn test_distilbert_tie_weights_default() {
        let config = DistilBertConfig::default();
        assert_eq!(config.tie_weights, Some(true));
    }

    #[test]
    fn test_distilbert_use_cache_default() {
        let config = DistilBertConfig::default();
        assert_eq!(config.use_cache, Some(true));
    }

    #[test]
    fn test_distilbert_classifier_dropout_fallback() {
        let config = minimal_distilbert_config();
        let resolved = config.classifier_dropout.unwrap_or(config.hidden_dropout_prob);
        assert_eq!(resolved, config.hidden_dropout_prob);
    }

    // ── Model creation tests ──────────────────────────────────────────────────

    #[test]
    fn test_distilbert_model_creation_minimal() {
        let config = minimal_distilbert_config();
        let model = DistilBertModel::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_distilbert_for_sequence_classification_creation() {
        let config = minimal_distilbert_config();
        let model = DistilBertForSequenceClassification::new(config, 2);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_distilbert_for_token_classification_creation() {
        let config = minimal_distilbert_config();
        let model = DistilBertForTokenClassification::new(config, 9);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_distilbert_for_question_answering_creation() {
        let config = minimal_distilbert_config();
        let model = DistilBertForQuestionAnswering::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_distilbert_for_masked_lm_creation() {
        let config = minimal_distilbert_config();
        let model = DistilBertForMaskedLM::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_distilbert_device_accessor_for_sequence_classification() {
        let config = minimal_distilbert_config();
        let model = DistilBertForSequenceClassification::new(config, 3);
        if let Ok(m) = model {
            let _ = m.device();
            drop(m);
        }
        std::hint::black_box(());
    }

    // ── Numeric properties ────────────────────────────────────────────────────

    #[test]
    fn test_distilbert_layer_norm_eps() {
        let config = DistilBertConfig::default();
        assert!((config.layer_norm_eps - 1e-12).abs() < 1e-15);
    }

    #[test]
    fn test_distilbert_pad_token_id() {
        let config = DistilBertConfig::default();
        assert_eq!(config.pad_token_id, 0);
    }

    #[test]
    fn test_distilbert_half_layers_of_bert() {
        // DistilBERT key property: half the layers of BERT-base
        let config = DistilBertConfig::default();
        assert_eq!(config.num_hidden_layers, 6);
    }

    // ── Config cloning ────────────────────────────────────────────────────────

    #[test]
    fn test_distilbert_config_clone() {
        let config = minimal_distilbert_config();
        let cloned = config.clone();
        assert_eq!(config.vocab_size, cloned.vocab_size);
        assert_eq!(config.hidden_size, cloned.hidden_size);
        assert_eq!(config.num_attention_heads, cloned.num_attention_heads);
        drop(config);
        drop(cloned);
        std::hint::black_box(());
    }

    // ── LCG ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_lcg_reproducibility() {
        let mut rng1 = Lcg::new(2468);
        let mut rng2 = Lcg::new(2468);
        for _ in 0..30 {
            assert_eq!(rng1.next_f32(), rng2.next_f32());
        }
    }

    #[test]
    fn test_lcg_range() {
        let mut rng = Lcg::new(13579);
        for _ in 0..200 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
