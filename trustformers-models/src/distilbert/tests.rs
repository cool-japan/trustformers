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
    // ── Real weight loading ───────────────────────────────────────────────────

    use crate::bert::layers::BertLayerNames;
    use crate::weight_loading::test_support::{build_safetensors, BertFixtureSpec, F32Tensor};
    use trustformers_core::tensor::Tensor;
    use trustformers_core::traits::Model;

    fn loading_config() -> DistilBertConfig {
        DistilBertConfig {
            vocab_size: 16,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 16,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 8,
            ..DistilBertConfig::default()
        }
    }

    fn fixture_spec(config: &DistilBertConfig, prefix: &str) -> BertFixtureSpec {
        BertFixtureSpec {
            prefix: prefix.to_string(),
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_layers: config.num_hidden_layers,
            intermediate_size: config.intermediate_size,
            max_position_embeddings: config.max_position_embeddings,
            type_vocab_size: None,
            include_pooler: false,
            names: BertLayerNames::distilbert(),
        }
    }

    fn hidden_values(model: &DistilBertModel) -> Vec<f32> {
        let output = model
            .forward_with_embeddings(vec![1, 2, 3], Some(vec![1, 1, 1]))
            .expect("forward must succeed after loading");
        match output.last_hidden_state {
            Tensor::F32(arr) => arr.iter().copied().collect(),
            other => panic!("expected an F32 hidden state, got {other:?}"),
        }
    }

    #[test]
    fn distilbert_load_pretrained_makes_the_model_determined_by_the_checkpoint() {
        // `load_pretrained` used to ignore the reader and return Ok(()), leaving
        // the model randomly initialised. Two differently-initialised models must
        // now agree exactly once they have loaded the same checkpoint.
        let config = loading_config();
        let bytes = fixture_spec(&config, "").safetensors();

        let mut first = DistilBertModel::new(config.clone()).expect("model must build");
        let mut second = DistilBertModel::new(config).expect("model must build");
        assert_ne!(
            hidden_values(&first),
            hidden_values(&second),
            "two random initialisations must differ"
        );

        first.load_pretrained(&mut bytes.as_slice()).expect("checkpoint must load");
        second.load_pretrained(&mut bytes.as_slice()).expect("checkpoint must load");

        assert_eq!(
            hidden_values(&first),
            hidden_values(&second),
            "after loading, both models must be the checkpoint's model"
        );
    }

    #[test]
    fn distilbert_load_pretrained_reports_missing_tensors() {
        let config = loading_config();
        let mut tensors = fixture_spec(&config, "").tensors();
        tensors.retain(|t| !t.name.contains("transformer.layer.0.ffn.lin1"));
        let bytes = build_safetensors(&tensors);

        let mut model = DistilBertModel::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an incomplete checkpoint must not load silently");
        assert!(
            err.to_string().contains("transformer.layer.0.ffn.lin1.weight"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn distilbert_load_pretrained_accepts_the_task_prefix_and_the_mlm_head() {
        let config = loading_config();
        let mut tensors = fixture_spec(&config, "distilbert.").tensors();
        tensors.push(F32Tensor::ramp("vocab_projector.bias", &[16], 3.0));
        let bytes = build_safetensors(&tensors);

        let mut model = DistilBertModel::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a prefixed checkpoint must load");
        assert!(report.is_complete());
        assert_eq!(report.ignored, vec!["vocab_projector.bias".to_string()]);
    }

    #[test]
    fn distilbert_num_parameters_tracks_the_configuration() {
        // Regression: this used to return a hardcoded 1_000_000 for every config.
        let small = DistilBertModel::new(loading_config()).expect("model must build");
        let bigger = DistilBertModel::new(DistilBertConfig {
            hidden_size: 16,
            intermediate_size: 32,
            ..loading_config()
        })
        .expect("model must build");

        assert_ne!(small.num_parameters(), 1_000_000);
        assert!(
            bigger.num_parameters() > small.num_parameters(),
            "a wider model must report more parameters ({} vs {})",
            bigger.num_parameters(),
            small.num_parameters()
        );

        let config = loading_config();
        let expected_embeddings = config.vocab_size * config.hidden_size
            + config.max_position_embeddings * config.hidden_size
            + 2 * config.hidden_size;
        assert!(
            small.num_parameters() > expected_embeddings,
            "the count must include the transformer stack as well as the embeddings"
        );
    }
    #[test]
    fn distilbert_task_head_weights_present_in_the_checkpoint_reach_the_model() {
        use crate::distilbert::tasks::DistilBertForSequenceClassification;

        let config = loading_config();
        let num_labels = 3usize;
        let hidden = config.hidden_size;
        let mut tensors = fixture_spec(&config, "distilbert.").tensors();
        let pre_weight = F32Tensor::ramp("pre_classifier.weight", &[hidden, hidden], 60.0);
        let pre_bias = F32Tensor::ramp("pre_classifier.bias", &[hidden], 70.0);
        let classifier_weight = F32Tensor::ramp("classifier.weight", &[num_labels, hidden], 80.0);
        let classifier_bias = F32Tensor::ramp("classifier.bias", &[num_labels], 90.0);
        tensors.extend([
            pre_weight,
            pre_bias,
            classifier_weight.clone(),
            classifier_bias,
        ]);
        let bytes = build_safetensors(&tensors);

        let mut model = DistilBertForSequenceClassification::new(config, num_labels)
            .expect("task model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a fine-tuned checkpoint must load");
        assert!(
            report.is_complete(),
            "a checkpoint carrying the head must produce a complete load, missing: {:?}",
            report.missing
        );
        assert!(
            report.loaded.contains(&"classifier.weight".to_string()),
            "the classifier weight must be reported as loaded"
        );
        match model.classifier_weight() {
            Tensor::F32(arr) => {
                assert_eq!(
                    arr.iter().copied().collect::<Vec<f32>>(),
                    classifier_weight.values
                );
            },
            other => panic!("expected an F32 classifier weight, got {other:?}"),
        }
    }

    #[test]
    fn distilbert_backbone_only_checkpoint_reports_the_absent_head() {
        use crate::distilbert::tasks::DistilBertForSequenceClassification;

        let config = loading_config();
        let bytes = fixture_spec(&config, "distilbert.").safetensors();
        let mut model =
            DistilBertForSequenceClassification::new(config, 3).expect("task model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a backbone-only checkpoint must still load");
        assert!(!report.is_complete());
        assert!(report.missing.contains(&"classifier.weight".to_string()));
    }

    #[test]
    fn distilbert_task_num_parameters_includes_the_head() {
        use crate::distilbert::tasks::DistilBertForSequenceClassification;

        let config = loading_config();
        let backbone = DistilBertModel::new(config.clone()).expect("model must build");
        let task =
            DistilBertForSequenceClassification::new(config, 3).expect("task model must build");
        assert!(
            task.num_parameters() > backbone.num_parameters(),
            "the head's parameters must be counted"
        );
    }
}
