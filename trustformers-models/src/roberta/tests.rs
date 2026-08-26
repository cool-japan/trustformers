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
            assert!((0.0..1.0).contains(&v));
        }
    }

    // ── Real checkpoint loading (regression for the silent `Ok(())`) ────────

    use crate::bert::layers::BertLayerNames;
    use crate::weight_loading::test_support::{build_safetensors, BertFixtureSpec, F32Tensor};
    use trustformers_core::traits::Model;

    fn loading_config() -> RobertaConfig {
        RobertaConfig {
            vocab_size: 16,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 8,
            type_vocab_size: 1,
            ..RobertaConfig::default()
        }
    }

    fn fixture(config: &RobertaConfig, prefix: &str) -> BertFixtureSpec {
        BertFixtureSpec {
            prefix: prefix.to_string(),
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_layers: config.num_hidden_layers,
            intermediate_size: config.intermediate_size,
            max_position_embeddings: config.max_position_embeddings,
            type_vocab_size: Some(config.type_vocab_size),
            include_pooler: true,
            names: BertLayerNames::bert(),
        }
    }

    /// Regression: `load_pretrained` was `Ok(())`, so the reader was never read
    /// and the model kept its random initialisation while reporting success.
    #[test]
    fn load_pretrained_binds_the_checkpoint_instead_of_returning_ok() {
        let config = loading_config();
        let tensors = fixture(&config, "roberta.").tensors();
        let bytes = build_safetensors(&tensors);

        let mut model = RobertaModel::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a matching checkpoint must load");
        assert!(
            report.is_complete(),
            "every parameter must be bound, missing: {:?}",
            report.missing
        );
        assert!(
            !report.loaded.is_empty(),
            "a real load must report the tensors it consumed"
        );
        assert!(
            report.loaded.contains(&"roberta.embeddings.word_embeddings.weight".to_string()),
            "the embedding matrix must be among the loaded tensors: {:?}",
            report.loaded
        );
    }

    #[test]
    fn load_pretrained_reports_a_missing_parameter_instead_of_inventing_it() {
        let config = loading_config();
        let mut tensors = fixture(&config, "roberta.").tensors();
        tensors.retain(|t| !t.name.contains("encoder.layer.1.attention.self.key"));
        let bytes = build_safetensors(&tensors);

        let mut model = RobertaModel::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an incomplete checkpoint must not load silently");
        assert!(
            err.to_string().contains("encoder.layer.1.attention.self.key.weight"),
            "the error must name the gap: {err}"
        );
    }

    #[test]
    fn load_pretrained_rejects_a_foreign_tensor() {
        let config = loading_config();
        let mut tensors = fixture(&config, "roberta.").tensors();
        tensors.push(F32Tensor::ramp(
            "roberta.encoder.layer.9.mystery.weight",
            &[8, 8],
            42.0,
        ));
        let bytes = build_safetensors(&tensors);

        let mut model = RobertaModel::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an unrecognised weight must fail the load");
        assert!(
            err.to_string().contains("mystery.weight"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn load_pretrained_rejects_bytes_that_are_not_a_checkpoint() {
        let mut model = RobertaModel::new(loading_config()).expect("model must build");
        let garbage = vec![0xABu8; 4096];
        let err = model
            .load_pretrained(&mut garbage.as_slice())
            .expect_err("garbage must not be accepted as weights");
        assert!(
            err.to_string().contains("unrecognised checkpoint container"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn load_pretrained_drops_the_pooler_when_the_checkpoint_has_none() {
        let config = loading_config();
        let mut spec = fixture(&config, "roberta.");
        spec.include_pooler = false;
        let bytes = spec.safetensors();

        let mut model = RobertaModel::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a pooler-less checkpoint must load");
        assert!(report.is_complete());
    }

    // ── Task heads (regression for the head-dropping delegation) ────────────

    /// The tensors a fine-tuned `RobertaForSequenceClassification` adds on top
    /// of the encoder.
    fn sequence_head_tensors(config: &RobertaConfig, num_labels: usize) -> Vec<F32Tensor> {
        let hidden = config.hidden_size;
        vec![
            F32Tensor::ramp("classifier.dense.weight", &[hidden, hidden], 200.0),
            F32Tensor::ramp("classifier.dense.bias", &[hidden], 210.0),
            F32Tensor::ramp("classifier.out_proj.weight", &[num_labels, hidden], 220.0),
            F32Tensor::ramp("classifier.out_proj.bias", &[num_labels], 230.0),
        ]
    }

    /// The tensors a `RobertaForMaskedLM` adds on top of the encoder.
    fn masked_lm_head_tensors(config: &RobertaConfig) -> Vec<F32Tensor> {
        let hidden = config.hidden_size;
        vec![
            F32Tensor::ramp("lm_head.dense.weight", &[hidden, hidden], 300.0),
            F32Tensor::ramp("lm_head.dense.bias", &[hidden], 310.0),
            F32Tensor::ramp("lm_head.layer_norm.weight", &[hidden], 320.0),
            F32Tensor::ramp("lm_head.layer_norm.bias", &[hidden], 330.0),
            F32Tensor::ramp(
                "lm_head.decoder.weight",
                &[config.vocab_size, hidden],
                340.0,
            ),
            F32Tensor::ramp("lm_head.bias", &[config.vocab_size], 350.0),
        ]
    }

    /// Regression: every task wrapper delegated to `RobertaModel::load_pretrained`,
    /// whose unused-tensor policy tolerates `classifier.` / `lm_head.` /
    /// `qa_outputs.`. The head was therefore dropped while the load reported
    /// success.
    #[test]
    fn sequence_classification_load_pretrained_binds_the_classification_head() {
        let config = loading_config();
        let num_labels = 3;
        let mut tensors = fixture(&config, "roberta.").tensors();
        tensors.extend(sequence_head_tensors(&config, num_labels));
        let bytes = build_safetensors(&tensors);

        let mut model =
            RobertaForSequenceClassification::new(config, num_labels).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a matching checkpoint must load");

        assert!(
            report.is_complete(),
            "every parameter must be bound, missing: {:?}",
            report.missing
        );
        for name in [
            "classifier.dense.weight",
            "classifier.out_proj.weight",
            "classifier.out_proj.bias",
        ] {
            assert!(
                report.loaded.contains(&name.to_string()),
                "{name} must be among the loaded tensors: {:?}",
                report.loaded
            );
        }
    }

    #[test]
    fn sequence_classification_load_pretrained_records_an_absent_head() {
        let config = loading_config();
        let bytes = fixture(&config, "roberta.").safetensors();

        let mut model = RobertaForSequenceClassification::new(config, 3).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a head-less encoder checkpoint must still load");
        assert!(
            report.missing.contains(&"classifier.out_proj.weight".to_string()),
            "the absent head must be named: {:?}",
            report.missing
        );
    }

    #[test]
    fn sequence_classification_load_pretrained_rejects_a_head_of_the_wrong_width() {
        let config = loading_config();
        let mut tensors = fixture(&config, "roberta.").tensors();
        tensors.extend(sequence_head_tensors(&config, 9));
        let bytes = build_safetensors(&tensors);

        let mut model = RobertaForSequenceClassification::new(config, 3).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("a 9-label head must not be reshaped into a 3-label model");
        assert!(
            err.to_string().contains("classifier.out_proj.weight"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn masked_lm_load_pretrained_binds_the_prediction_head() {
        let config = loading_config();
        let mut tensors = fixture(&config, "roberta.").tensors();
        tensors.extend(masked_lm_head_tensors(&config));
        let bytes = build_safetensors(&tensors);

        let mut model = RobertaForMaskedLM::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a matching checkpoint must load");

        assert!(
            report.is_complete(),
            "every parameter must be bound, missing: {:?}",
            report.missing
        );
        for name in [
            "lm_head.dense.weight",
            "lm_head.layer_norm.weight",
            "lm_head.decoder.weight",
            "lm_head.bias",
        ] {
            assert!(
                report.loaded.contains(&name.to_string()),
                "{name} must be among the loaded tensors: {:?}",
                report.loaded
            );
        }
    }

    #[test]
    fn masked_lm_load_pretrained_accepts_the_aliased_decoder_bias() {
        let config = loading_config();
        let mut tensors = fixture(&config, "roberta.").tensors();
        let mut head = masked_lm_head_tensors(&config);
        for tensor in &mut head {
            if tensor.name == "lm_head.bias" {
                tensor.name = "lm_head.decoder.bias".to_string();
            }
        }
        tensors.extend(head);
        let bytes = build_safetensors(&tensors);

        let mut model = RobertaForMaskedLM::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("the aliased bias spelling must load");
        assert!(
            report.loaded.contains(&"lm_head.decoder.bias".to_string()),
            "the aliased bias must be consumed: {:?}",
            report.loaded
        );
    }

    #[test]
    fn token_classification_load_pretrained_binds_the_classifier() {
        let config = loading_config();
        let num_labels = 5;
        let hidden = config.hidden_size;
        let mut tensors = fixture(&config, "roberta.").tensors();
        tensors.push(F32Tensor::ramp(
            "classifier.weight",
            &[num_labels, hidden],
            400.0,
        ));
        tensors.push(F32Tensor::ramp("classifier.bias", &[num_labels], 410.0));
        let bytes = build_safetensors(&tensors);

        let mut model =
            RobertaForTokenClassification::new(config, num_labels).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a matching checkpoint must load");
        assert!(
            report.is_complete(),
            "every parameter must be bound, missing: {:?}",
            report.missing
        );
        assert!(report.loaded.contains(&"classifier.weight".to_string()));
    }

    #[test]
    fn question_answering_load_pretrained_binds_the_span_head() {
        let config = loading_config();
        let hidden = config.hidden_size;
        let mut tensors = fixture(&config, "roberta.").tensors();
        tensors.push(F32Tensor::ramp("qa_outputs.weight", &[2, hidden], 500.0));
        tensors.push(F32Tensor::ramp("qa_outputs.bias", &[2], 510.0));
        let bytes = build_safetensors(&tensors);

        let mut model = RobertaForQuestionAnswering::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a matching checkpoint must load");
        assert!(
            report.is_complete(),
            "every parameter must be bound, missing: {:?}",
            report.missing
        );
        assert!(
            report.loaded.contains(&"qa_outputs.weight".to_string())
                && report.loaded.contains(&"qa_outputs.bias".to_string()),
            "the span head must be among the loaded tensors: {:?}",
            report.loaded
        );
    }

    #[test]
    fn question_answering_load_pretrained_rejects_a_span_head_of_the_wrong_width() {
        let config = loading_config();
        let hidden = config.hidden_size;
        let mut tensors = fixture(&config, "roberta.").tensors();
        tensors.push(F32Tensor::ramp("qa_outputs.weight", &[3, hidden], 500.0));
        let bytes = build_safetensors(&tensors);

        let mut model = RobertaForQuestionAnswering::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("a 3-logit span head is not a start/end head");
        assert!(
            err.to_string().contains("qa_outputs.weight"),
            "unexpected: {err}"
        );
    }

    // ── Contextual strictness: wrapper path vs bare-encoder path ────────────

    /// A task wrapper must refuse a checkpoint entry it does not recognise
    /// inside a namespace it binds itself.
    ///
    /// RoBERTa's bare encoder reuses `BertModel::unused_tensor_policy()`, which
    /// tolerates `classifier.`, `lm_head.` and `qa_outputs.` so that an encoder
    /// can be lifted out of a fine-tuned checkpoint. The task wrappers used to
    /// inherit that tolerance even though they bind those namespaces, so a
    /// misspelling landed in `ignored`, the load returned `Ok`, and the layer
    /// the typo was meant to fill kept its random initialisation. See
    /// [`crate::weight_loading::binding::BoundNamespaces`].
    #[test]
    fn a_wrapper_rejects_an_unknown_tensor_inside_a_namespace_it_binds() {
        // Sequence classification: a misspelt `classifier.out_proj.weight`.
        let config = loading_config();
        let hidden = config.hidden_size;
        let mut tensors = fixture(&config, "roberta.").tensors();
        tensors.extend(sequence_head_tensors(&config, 3));
        tensors.push(F32Tensor::ramp(
            "classifier.out_prj.weight",
            &[3, hidden],
            240.0,
        ));
        let bytes = build_safetensors(&tensors);
        let mut model = RobertaForSequenceClassification::new(config, 3).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("a misspelt classifier tensor must not be tolerated by its own binder");
        let message = err.to_string();
        assert!(
            message.contains("classifier.out_prj.weight"),
            "the offending name must be reported: {message}"
        );
        // The refusal must come from the wrapper's own namespace check, not
        // from a shape or missing-parameter error that happens to mention the
        // name: only `BoundNamespaces::verify` phrases it this way.
        assert!(
            message.contains("does not recognise inside the head namespace"),
            "the refusal must be the bound-namespace check: {message}"
        );

        // Token classification binds the same namespace with flat names.
        let config = loading_config();
        let hidden = config.hidden_size;
        let mut tensors = fixture(&config, "roberta.").tensors();
        tensors.push(F32Tensor::ramp("classifier.weight", &[5, hidden], 400.0));
        tensors.push(F32Tensor::ramp("classifier.bias", &[5], 410.0));
        tensors.push(F32Tensor::ramp(
            "classifier.extra_head.weight",
            &[5, hidden],
            420.0,
        ));
        let bytes = build_safetensors(&tensors);
        let mut model = RobertaForTokenClassification::new(config, 5).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an unknown tensor under the bound classifier namespace must be refused");
        assert!(
            err.to_string().contains("classifier.extra_head.weight"),
            "unexpected: {err}"
        );

        // Masked LM: a misspelling inside `lm_head.`.
        let config = loading_config();
        let hidden = config.hidden_size;
        let mut tensors = fixture(&config, "roberta.").tensors();
        tensors.extend(masked_lm_head_tensors(&config));
        tensors.push(F32Tensor::ramp(
            "lm_head.layer_norm.weigth",
            &[hidden],
            360.0,
        ));
        let bytes = build_safetensors(&tensors);
        let mut model = RobertaForMaskedLM::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("a misspelt prediction-head tensor must be refused");
        assert!(
            err.to_string().contains("lm_head.layer_norm.weigth"),
            "unexpected: {err}"
        );

        // Question answering: a misspelling under `qa_outputs.`.
        let config = loading_config();
        let hidden = config.hidden_size;
        let mut tensors = fixture(&config, "roberta.").tensors();
        tensors.push(F32Tensor::ramp("qa_outputs.weight", &[2, hidden], 500.0));
        tensors.push(F32Tensor::ramp("qa_outputs.bias", &[2], 510.0));
        tensors.push(F32Tensor::ramp("qa_outputs.baias", &[2], 520.0));
        let bytes = build_safetensors(&tensors);
        let mut model = RobertaForQuestionAnswering::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("a misspelt span-head tensor must be refused");
        assert!(
            err.to_string().contains("qa_outputs.baias"),
            "unexpected: {err}"
        );
    }

    /// The strictness above must not turn into "every checkpoint entry must be
    /// consumed": a checkpoint legitimately carries heads a particular model
    /// does not bind, and the bare encoder binds none of them at all.
    #[test]
    fn namespaces_a_model_does_not_bind_stay_tolerated() {
        let config = loading_config();
        let mut tensors = fixture(&config, "roberta.").tensors();
        tensors.extend(sequence_head_tensors(&config, 3));
        // A masked-LM head the classification wrapper does not bind at all.
        tensors.extend(masked_lm_head_tensors(&config));
        let bytes = build_safetensors(&tensors);

        let mut model = RobertaForSequenceClassification::new(config, 3).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a masked-LM head this model does not bind must stay tolerated");
        assert!(
            report.ignored.iter().any(|name| name == "lm_head.dense.weight"),
            "the unbound head must be reported as ignored: {:?}",
            report.ignored
        );

        // The same checkpoint through the bare encoder: it binds neither head,
        // so both namespaces stay tolerated exactly as before.
        let mut encoder = RobertaModel::new(loading_config()).expect("model must build");
        let encoder_report = encoder
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("the bare encoder must keep tolerating head namespaces it never binds");
        for name in ["classifier.out_proj.weight", "lm_head.dense.weight"] {
            assert!(
                encoder_report.ignored.iter().any(|ignored| ignored == name),
                "{name} must stay tolerated on the bare-encoder path: {:?}",
                encoder_report.ignored
            );
        }
    }
}
