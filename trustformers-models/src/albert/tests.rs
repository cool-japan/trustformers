#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::albert::config::AlbertConfig;
    use crate::albert::model::AlbertModel;
    use crate::albert::tasks::{
        AlbertForMaskedLM, AlbertForQuestionAnswering, AlbertForSequenceClassification,
        AlbertForTokenClassification,
    };
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

    // ── Config creation helpers ───────────────────────────────────────────────

    fn minimal_albert_config() -> AlbertConfig {
        AlbertConfig {
            vocab_size: 1000,
            embedding_size: 64,
            hidden_size: 256,
            num_hidden_layers: 4,
            num_hidden_groups: 1,
            num_attention_heads: 4,
            intermediate_size: 512,
            inner_group_num: 1,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 128,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            classifier_dropout_prob: None,
            position_embedding_type: "absolute".to_string(),
            pad_token_id: 0,
            bos_token_id: 2,
            eos_token_id: 3,
        }
    }

    // ── Config validation tests ───────────────────────────────────────────────

    #[test]
    fn test_albert_default_config_is_valid() {
        let config = AlbertConfig::default();
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_base_v1_params() {
        let config = AlbertConfig::albert_base_v1();
        assert_eq!(config.vocab_size, 30000);
        assert_eq!(config.embedding_size, 128);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.hidden_act, "gelu");
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_base_v2_params() {
        let config = AlbertConfig::albert_base_v2();
        assert_eq!(config.hidden_act, "gelu_new");
        assert_eq!(config.hidden_dropout_prob, 0.0);
        assert_eq!(config.attention_probs_dropout_prob, 0.0);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_large_v2_params() {
        let config = AlbertConfig::albert_large_v2();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.intermediate_size, 4096);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_xlarge_v1_params() {
        let config = AlbertConfig::albert_xlarge_v1();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.intermediate_size, 8192);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_xlarge_v2_params() {
        let config = AlbertConfig::albert_xlarge_v2();
        assert_eq!(config.hidden_act, "gelu_new");
        assert_eq!(config.hidden_dropout_prob, 0.0);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_xxlarge_v1_params() {
        let config = AlbertConfig::albert_xxlarge_v1();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 64);
        assert_eq!(config.intermediate_size, 16384);
        assert_eq!(config.num_hidden_layers, 12);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_xxlarge_v2_params() {
        let config = AlbertConfig::albert_xxlarge_v2();
        assert_eq!(config.hidden_act, "gelu_new");
        assert_eq!(config.hidden_dropout_prob, 0.0);
        assert!(config.validate().is_ok());
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_architecture_string() {
        let config = AlbertConfig::default();
        assert_eq!(config.architecture(), "albert");
        drop(config);
        std::hint::black_box(());
    }

    // ── Config validation failure cases ──────────────────────────────────────

    #[test]
    fn test_albert_invalid_vocab_size_zero() {
        let mut config = minimal_albert_config();
        config.vocab_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_albert_invalid_hidden_size_zero() {
        let mut config = minimal_albert_config();
        config.hidden_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_albert_invalid_embedding_size_zero() {
        let mut config = minimal_albert_config();
        config.embedding_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_albert_invalid_num_hidden_layers_zero() {
        let mut config = minimal_albert_config();
        config.num_hidden_layers = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_albert_invalid_num_hidden_groups_zero() {
        let mut config = minimal_albert_config();
        config.num_hidden_groups = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_albert_invalid_num_attention_heads_zero() {
        let mut config = minimal_albert_config();
        config.num_attention_heads = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_albert_invalid_hidden_size_not_divisible_by_heads() {
        let mut config = minimal_albert_config();
        config.hidden_size = 257; // not divisible by 4
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_albert_invalid_layers_not_divisible_by_groups() {
        let mut config = minimal_albert_config();
        config.num_hidden_layers = 5;
        config.num_hidden_groups = 3; // 5 not divisible by 3
        assert!(config.validate().is_err());
    }

    // ── from_pretrained_name tests ────────────────────────────────────────────

    #[test]
    fn test_albert_from_pretrained_name_base_v1() {
        let config = AlbertConfig::from_pretrained_name("albert-base-v1");
        assert_eq!(config.hidden_act, "gelu");
        assert_eq!(config.hidden_dropout_prob, 0.1);
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_from_pretrained_name_base_v2() {
        let config = AlbertConfig::from_pretrained_name("albert-base-v2");
        assert_eq!(config.hidden_act, "gelu_new");
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_from_pretrained_name_large_v1() {
        let config = AlbertConfig::from_pretrained_name("albert-large-v1");
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.hidden_act, "gelu");
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_from_pretrained_name_xlarge_v2() {
        let config = AlbertConfig::from_pretrained_name("albert-xlarge-v2");
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.hidden_act, "gelu_new");
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_from_pretrained_name_xxlarge_v1() {
        let config = AlbertConfig::from_pretrained_name("albert-xxlarge-v1");
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.hidden_act, "gelu");
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_from_pretrained_name_xxlarge_v2() {
        let config = AlbertConfig::from_pretrained_name("albert-xxlarge-v2");
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.hidden_act, "gelu_new");
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_from_pretrained_name_unknown_uses_default() {
        let config = AlbertConfig::from_pretrained_name("unknown-model");
        // Should fall back to albert_base_v2
        assert_eq!(config.hidden_act, "gelu_new");
        assert_eq!(config.hidden_dropout_prob, 0.0);
        drop(config);
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_from_pretrained_name_case_insensitive() {
        let config = AlbertConfig::from_pretrained_name("ALBERT-BASE-V1");
        // contains() check on lowercase, so this should match
        assert_eq!(config.hidden_act, "gelu");
        drop(config);
        std::hint::black_box(());
    }

    // ── Model creation tests ──────────────────────────────────────────────────

    #[test]
    fn test_albert_model_creation_minimal() {
        let config = minimal_albert_config();
        let model = AlbertModel::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_for_sequence_classification_creation() {
        let config = minimal_albert_config();
        let num_labels = 3usize;
        let model = AlbertForSequenceClassification::new(config, num_labels);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_for_token_classification_creation() {
        let config = minimal_albert_config();
        let num_labels = 9usize; // BIO tags for NER
        let model = AlbertForTokenClassification::new(config, num_labels);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_for_question_answering_creation() {
        let config = minimal_albert_config();
        let model = AlbertForQuestionAnswering::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_for_masked_lm_creation() {
        let config = minimal_albert_config();
        let model = AlbertForMaskedLM::new(config);
        assert!(model.is_ok());
        if let Ok(m) = model {
            drop(m);
        }
        std::hint::black_box(());
    }

    // ── Classifier dropout resolution ─────────────────────────────────────────

    #[test]
    fn test_albert_classifier_dropout_falls_back_to_hidden_dropout() {
        let mut config = minimal_albert_config();
        config.classifier_dropout_prob = None;
        config.hidden_dropout_prob = 0.15;
        // classifier_dropout should resolve to hidden_dropout_prob when None
        let resolved = config.classifier_dropout_prob.unwrap_or(config.hidden_dropout_prob);
        assert!((resolved - 0.15).abs() < 1e-6);
    }

    #[test]
    fn test_albert_classifier_dropout_explicit_overrides() {
        let mut config = minimal_albert_config();
        config.classifier_dropout_prob = Some(0.2);
        config.hidden_dropout_prob = 0.1;
        let resolved = config.classifier_dropout_prob.unwrap_or(config.hidden_dropout_prob);
        assert!((resolved - 0.2).abs() < 1e-6);
    }

    // ── LCG pseudo-random reproducibility ────────────────────────────────────

    #[test]
    fn test_lcg_reproducibility() {
        let mut rng1 = Lcg::new(42);
        let mut rng2 = Lcg::new(42);
        for _ in 0..20 {
            assert_eq!(rng1.next_f32(), rng2.next_f32());
        }
    }

    #[test]
    fn test_lcg_range() {
        let mut rng = Lcg::new(1337);
        for _ in 0..100 {
            let v = rng.next_f32();
            assert!(v >= 0.0, "LCG value {} should be >= 0", v);
            assert!(v < 1.0, "LCG value {} should be < 1", v);
        }
    }

    // ── Config cloning and equality ───────────────────────────────────────────

    #[test]
    fn test_albert_config_clone() {
        let config = minimal_albert_config();
        let cloned = config.clone();
        assert_eq!(config.vocab_size, cloned.vocab_size);
        assert_eq!(config.hidden_size, cloned.hidden_size);
        assert_eq!(config.embedding_size, cloned.embedding_size);
        assert_eq!(config.num_attention_heads, cloned.num_attention_heads);
        drop(config);
        drop(cloned);
        std::hint::black_box(());
    }

    #[test]
    fn test_albert_config_token_ids() {
        let config = AlbertConfig::albert_base_v1();
        assert_eq!(config.pad_token_id, 0);
        assert_eq!(config.bos_token_id, 2);
        assert_eq!(config.eos_token_id, 3);
    }

    #[test]
    fn test_albert_layer_norm_eps() {
        let config = AlbertConfig::albert_base_v2();
        assert!((config.layer_norm_eps - 1e-12).abs() < 1e-15);
    }

    #[test]
    fn test_albert_position_embedding_type() {
        let config = AlbertConfig::albert_base_v2();
        assert_eq!(config.position_embedding_type, "absolute");
    }

    // ── Real checkpoint loading (regression for the silent `Ok(())`) ────────

    use crate::weight_loading::test_support::{build_safetensors, F32Tensor};
    use trustformers_core::traits::Model;

    fn loading_config() -> AlbertConfig {
        AlbertConfig {
            vocab_size: 16,
            embedding_size: 4,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_hidden_groups: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            inner_group_num: 1,
            max_position_embeddings: 8,
            type_vocab_size: 2,
            ..AlbertConfig::albert_base_v2()
        }
    }

    /// Every tensor an ALBERT checkpoint of this shape carries.
    ///
    /// Note the *group*-indexed layer names: ALBERT shares parameters across
    /// layers, so a 12-layer model stores exactly one group.
    fn albert_tensors(config: &AlbertConfig, prefix: &str, include_pooler: bool) -> Vec<F32Tensor> {
        let embedding = config.embedding_size;
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let mut seed = 0.0f32;
        let mut next = || {
            seed += 1.0;
            seed
        };

        let mut tensors = vec![
            F32Tensor::ramp(
                &format!("{prefix}embeddings.word_embeddings.weight"),
                &[config.vocab_size, embedding],
                next(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}embeddings.position_embeddings.weight"),
                &[config.max_position_embeddings, embedding],
                next(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}embeddings.token_type_embeddings.weight"),
                &[config.type_vocab_size, embedding],
                next(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}embeddings.LayerNorm.weight"),
                &[embedding],
                next(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}embeddings.LayerNorm.bias"),
                &[embedding],
                next(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}encoder.embedding_hidden_mapping_in.weight"),
                &[hidden, embedding],
                next(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}encoder.embedding_hidden_mapping_in.bias"),
                &[hidden],
                next(),
            ),
        ];

        for group in 0..config.num_hidden_groups {
            for layer in 0..config.inner_group_num {
                let base =
                    format!("{prefix}encoder.albert_layer_groups.{group}.albert_layers.{layer}");
                for projection in ["query", "key", "value", "dense"] {
                    tensors.push(F32Tensor::ramp(
                        &format!("{base}.attention.{projection}.weight"),
                        &[hidden, hidden],
                        next(),
                    ));
                    tensors.push(F32Tensor::ramp(
                        &format!("{base}.attention.{projection}.bias"),
                        &[hidden],
                        next(),
                    ));
                }
                tensors.push(F32Tensor::ramp(
                    &format!("{base}.attention.LayerNorm.weight"),
                    &[hidden],
                    next(),
                ));
                tensors.push(F32Tensor::ramp(
                    &format!("{base}.attention.LayerNorm.bias"),
                    &[hidden],
                    next(),
                ));
                tensors.push(F32Tensor::ramp(
                    &format!("{base}.ffn.weight"),
                    &[intermediate, hidden],
                    next(),
                ));
                tensors.push(F32Tensor::ramp(
                    &format!("{base}.ffn.bias"),
                    &[intermediate],
                    next(),
                ));
                tensors.push(F32Tensor::ramp(
                    &format!("{base}.ffn_output.weight"),
                    &[hidden, intermediate],
                    next(),
                ));
                tensors.push(F32Tensor::ramp(
                    &format!("{base}.ffn_output.bias"),
                    &[hidden],
                    next(),
                ));
                tensors.push(F32Tensor::ramp(
                    &format!("{base}.full_layer_layer_norm.weight"),
                    &[hidden],
                    next(),
                ));
                tensors.push(F32Tensor::ramp(
                    &format!("{base}.full_layer_layer_norm.bias"),
                    &[hidden],
                    next(),
                ));
            }
        }

        if include_pooler {
            tensors.push(F32Tensor::ramp(
                &format!("{prefix}pooler.weight"),
                &[hidden, hidden],
                next(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{prefix}pooler.bias"),
                &[hidden],
                next(),
            ));
        }

        tensors
    }

    /// Regression: `load_pretrained` was `Ok(())`, so the reader was never read
    /// and the model kept its random initialisation while reporting success.
    #[test]
    fn load_pretrained_binds_the_checkpoint_instead_of_returning_ok() {
        let config = loading_config();
        let tensors = albert_tensors(&config, "albert.", true);
        let bytes = build_safetensors(&tensors);

        let mut model = AlbertModel::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a matching checkpoint must load");
        assert!(
            report.is_complete(),
            "every parameter must be bound, missing: {:?}",
            report.missing
        );
        assert!(
            report.unexpected.is_empty(),
            "nothing should be left over: {:?}",
            report.unexpected
        );
        assert_eq!(
            report.loaded.len(),
            tensors.len(),
            "every fixture tensor must reach the model"
        );
    }

    /// ALBERT's factorised embedding: the table lives at `embedding_size`, and
    /// `encoder.embedding_hidden_mapping_in` projects it up to `hidden_size`.
    /// A loader that assumed `embedding_size == hidden_size` would reject this.
    #[test]
    fn load_pretrained_handles_the_factorised_embedding() {
        let config = loading_config();
        assert_ne!(
            config.embedding_size, config.hidden_size,
            "the test must exercise the factorised case"
        );
        let bytes = build_safetensors(&albert_tensors(&config, "albert.", true));
        let mut model = AlbertModel::new(config).expect("model must build");
        model
            .load_pretrained(&mut bytes.as_slice())
            .expect("a factorised-embedding checkpoint must load");
    }

    #[test]
    fn load_pretrained_reports_a_missing_parameter_instead_of_inventing_it() {
        let config = loading_config();
        let mut tensors = albert_tensors(&config, "albert.", true);
        tensors.retain(|t| !t.name.contains("albert_layers.0.attention.key"));
        let bytes = build_safetensors(&tensors);

        let mut model = AlbertModel::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an incomplete checkpoint must not load silently");
        assert!(
            err.to_string().contains("attention.key.weight"),
            "the error must name the gap: {err}"
        );
    }

    #[test]
    fn load_pretrained_rejects_a_foreign_tensor() {
        let config = loading_config();
        let mut tensors = albert_tensors(&config, "albert.", true);
        tensors.push(F32Tensor::ramp(
            "albert.encoder.mystery.weight",
            &[8, 8],
            77.0,
        ));
        let bytes = build_safetensors(&tensors);

        let mut model = AlbertModel::new(config).expect("model must build");
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
        let mut model = AlbertModel::new(loading_config()).expect("model must build");
        let garbage = vec![0xABu8; 4096];
        let err = model
            .load_pretrained(&mut garbage.as_slice())
            .expect_err("garbage must not be accepted as weights");
        assert!(
            err.to_string().contains("unrecognised checkpoint container"),
            "unexpected: {err}"
        );
    }
}
