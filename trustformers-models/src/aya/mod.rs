pub mod config;
pub mod model;
pub mod tasks;

pub use config::{AyaConfig, AyaError, AYA_23_LANGUAGE_CODES};
pub use model::{
    AyaAttention, AyaDecoderLayer, AyaDenseLayer, AyaEmbedding, AyaLayerNorm, AyaMlp, AyaModel,
    AyaRotaryEmbedding,
};
pub use tasks::{
    AyaForCausalLm, AyaForMultilingualGeneration, AyaForSequenceClassification, AyaLmHead,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal AyaConfig that constructs quickly in tests.
    fn tiny_config() -> AyaConfig {
        AyaConfig {
            vocab_size: 64,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4, // 16 / 4
            max_position_embeddings: 32,
            layer_norm_eps: 1e-5,
            rope_theta: 10000.0,
            logit_scale: 0.0625,
            use_qk_norm: false,
            tie_word_embeddings: false,
            attention_dropout: 0.0,
            supported_languages: AYA_23_LANGUAGE_CODES.iter().map(|s| s.to_string()).collect(),
            tokenizer_class: "PreTrainedTokenizer".to_string(),
        }
    }

    // ── Test 1: Config defaults — 23 languages ───────────────────────────────
    #[test]
    fn test_aya_config_defaults_languages() {
        let cfg = AyaConfig::default();
        assert_eq!(cfg.supported_language_count(), 23);
        assert_eq!(cfg.vocab_size, 256000);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.max_position_embeddings, 131072);
        assert!((cfg.logit_scale - 0.0625).abs() < 1e-6);
        assert!(!cfg.use_qk_norm);
    }

    // ── Test 2: supports_language ─────────────────────────────────────────────
    #[test]
    fn test_supports_language() {
        let cfg = AyaConfig::default();
        assert!(cfg.supports_language("en"));
        assert!(cfg.supports_language("zh"));
        assert!(cfg.supports_language("ar"));
        assert!(cfg.supports_language("ja"));
        assert!(cfg.supports_language("de"));
    }

    // ── Test 3: unsupported language ─────────────────────────────────────────
    #[test]
    fn test_unsupported_language() {
        let cfg = AyaConfig::default();
        assert!(!cfg.supports_language("xx")); // not a real code
        assert!(!cfg.supports_language("kw")); // Cornish — not in Aya-23
    }

    // ── Test 4: LayerNorm forward (mean ≈ 0) ─────────────────────────────────
    #[test]
    fn test_layer_norm_mean_zero() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0_f32; 8];
        let bias = vec![0.0_f32; 8];
        let out = AyaLayerNorm::forward(&x, &weight, &bias, 1e-5)
            .expect("LayerNorm forward should not fail");
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(
            mean.abs() < 1e-5,
            "LayerNorm output mean should be ≈0, got {mean}"
        );
        assert_eq!(out.len(), 8);
    }

    // ── Test 5: Logit scale is applied ───────────────────────────────────────
    #[test]
    fn test_logit_scale_applied() {
        let cfg = tiny_config();
        let model = AyaForCausalLm::new(&cfg).expect("construction should succeed");
        assert!((model.logit_scale() - 0.0625).abs() < 1e-6);

        // With zero weights the logits are 0 before and after scaling.
        let logits = model.forward_last_logits(&[1_u32, 2]).expect("forward should not fail");
        assert_eq!(logits.len(), cfg.vocab_size);
        // All should be exactly 0.0 with zero-init weights.
        assert!(logits.iter().all(|v| *v == 0.0));
    }

    // ── Test 6: Multilingual generation succeeds for supported language ───────
    #[test]
    fn test_multilingual_generation_supported() {
        let cfg = tiny_config();
        let model = AyaForMultilingualGeneration::new(&cfg).expect("construction should succeed");
        let result = model.generate_in_language(&[1_u32, 2, 3], "en", 2, cfg.vocab_size);
        assert!(
            result.is_ok(),
            "generation for 'en' should succeed, got: {:?}",
            result.err()
        );
        let tokens = result.unwrap();
        assert_eq!(tokens.len(), 2, "should generate exactly 2 new tokens");
    }

    // ── Test 7: validate ─────────────────────────────────────────────────────
    #[test]
    fn test_aya_validate() {
        let cfg = AyaConfig::default();
        assert!(cfg.validate().is_ok());

        let mut bad_cfg = cfg.clone();
        bad_cfg.head_dim = 99; // wrong
        assert!(
            bad_cfg.validate().is_err(),
            "validate should fail with wrong head_dim"
        );

        let mut bad2 = cfg.clone();
        bad2.supported_languages.clear();
        assert!(
            bad2.validate().is_err(),
            "validate should fail with empty supported_languages"
        );
    }

    // ── Test 8: Vocab size for multilingual (256K) ───────────────────────────
    #[test]
    fn test_vocab_size_multilingual() {
        let cfg = AyaConfig::default();
        assert_eq!(
            cfg.vocab_size, 256000,
            "Aya-23 vocab_size must be 256000 to cover 23 languages"
        );
        // Verify count one more time.
        assert_eq!(
            cfg.supported_language_count(),
            23,
            "Aya-23 should support exactly 23 languages"
        );
    }

    // ── Test 9: Multilingual generation fails for unsupported language ────────
    #[test]
    fn test_multilingual_generation_unsupported() {
        let cfg = tiny_config();
        let model = AyaForMultilingualGeneration::new(&cfg).expect("construction should succeed");
        let result = model.generate_in_language(&[1_u32, 2], "xx", 2, cfg.vocab_size);
        assert!(
            result.is_err(),
            "generation for unsupported language 'xx' should fail"
        );
    }

    // ── Test 10: AyaError display variants ───────────────────────────────────
    #[test]
    fn test_aya_error_display_invalid_config() {
        let err = AyaError::InvalidConfig("bad field".to_string());
        let s = err.to_string();
        assert!(s.contains("invalid config"), "got: {s}");
        assert!(s.contains("bad field"), "got: {s}");
    }

    #[test]
    fn test_aya_error_display_dimension_mismatch() {
        let err = AyaError::DimensionMismatch {
            expected: 128,
            got: 64,
        };
        let s = err.to_string();
        assert!(s.contains("128") && s.contains("64"), "got: {s}");
        assert!(s.contains("mismatch"), "got: {s}");
    }

    #[test]
    fn test_aya_error_display_empty_input() {
        let err = AyaError::EmptyInput;
        let s = err.to_string();
        assert!(s.to_lowercase().contains("empty"), "got: {s}");
    }

    #[test]
    fn test_aya_error_display_unsupported_language() {
        let err = AyaError::UnsupportedLanguage("zz".to_string());
        let s = err.to_string();
        assert!(s.contains("zz"), "got: {s}");
        assert!(
            s.to_lowercase().contains("not supported") || s.contains("language"),
            "got: {s}"
        );
    }

    // ── Test 14: Config clone/debug ───────────────────────────────────────────
    #[test]
    fn test_aya_config_clone() {
        let cfg = AyaConfig::default();
        let cloned = cfg.clone();
        assert_eq!(cloned.vocab_size, cfg.vocab_size);
        assert_eq!(cloned.hidden_size, cfg.hidden_size);
        assert_eq!(
            cloned.supported_languages.len(),
            cfg.supported_languages.len()
        );
    }

    #[test]
    fn test_aya_config_debug() {
        let cfg = tiny_config();
        let s = format!("{:?}", cfg);
        assert!(s.contains("AyaConfig"), "got: {s}");
        assert!(s.contains("vocab_size"), "got: {s}");
        assert!(s.contains("logit_scale"), "got: {s}");
    }

    // ── Test 16: num_query_groups and is_gqa ─────────────────────────────────
    #[test]
    fn test_aya_num_query_groups() {
        let cfg = AyaConfig::default(); // 32 Q / 8 KV = 4 groups
        assert_eq!(cfg.num_query_groups(), 4);
    }

    #[test]
    fn test_aya_is_gqa() {
        let cfg = AyaConfig::default(); // 8 KV != 32 Q
        assert!(cfg.is_gqa());
        let mut mha = tiny_config();
        mha.num_key_value_heads = mha.num_attention_heads; // MHA
        assert!(!mha.is_gqa());
    }

    // ── Test 18: all 23 language codes in AYA_23_LANGUAGE_CODES constant ──────
    #[test]
    fn test_aya_23_language_codes_count() {
        assert_eq!(
            AYA_23_LANGUAGE_CODES.len(),
            23,
            "AYA_23_LANGUAGE_CODES must contain exactly 23 entries"
        );
    }

    #[test]
    fn test_aya_23_language_codes_contains_key_languages() {
        let codes = AYA_23_LANGUAGE_CODES;
        for lang in ["en", "zh", "ar", "fr", "de", "ja", "ko", "es", "pt", "ru"] {
            assert!(
                codes.contains(&lang),
                "AYA_23_LANGUAGE_CODES must contain '{lang}'"
            );
        }
    }

    // ── Test 20: forward_last_logits empty input error ────────────────────────
    #[test]
    fn test_forward_last_logits_empty_input() {
        let cfg = tiny_config();
        let model = AyaForCausalLm::new(&cfg).expect("construction should succeed");
        let result = model.forward_last_logits(&[]);
        assert!(result.is_err(), "empty input should return Err");
        let s = result.unwrap_err().to_string();
        assert!(s.to_lowercase().contains("empty"), "got: {s}");
    }

    // ── Test 21: generate_in_language empty prompt error ─────────────────────
    #[test]
    fn test_generate_in_language_empty_prompt() {
        let cfg = tiny_config();
        let model = AyaForMultilingualGeneration::new(&cfg).expect("construction");
        let result = model.generate_in_language(&[], "en", 2, cfg.vocab_size);
        assert!(result.is_err(), "empty prompt should return Err");
    }

    // ── Test 22: set_target_language mutates state ───────────────────────────
    #[test]
    fn test_set_target_language() {
        let cfg = tiny_config();
        let mut model = AyaForMultilingualGeneration::new(&cfg).expect("construction");
        assert!(
            model.target_language.is_none(),
            "target_language should start as None"
        );
        model.set_target_language("fr");
        assert_eq!(model.target_language.as_deref(), Some("fr"));
    }

    // ── Test 23: sequence classification forward output length ────────────────
    #[test]
    fn test_aya_classification_output_length() {
        let cfg = tiny_config();
        let model =
            AyaForSequenceClassification::new(&cfg, 4).expect("classification model creation");
        assert_eq!(model.num_labels(), 4);
        let out = model.forward(&[1_u32, 2, 3]).expect("classification forward");
        assert_eq!(out.len(), 4, "output should have num_labels=4 logits");
    }

    // ── Test 24: classification num_labels=0 should fail ─────────────────────
    #[test]
    fn test_aya_classification_num_labels_zero_error() {
        let cfg = tiny_config();
        let result = AyaForSequenceClassification::new(&cfg, 0);
        assert!(result.is_err(), "num_labels=0 should return Err");
    }

    // ── Test 25: forward_last_logits single token ─────────────────────────────
    #[test]
    fn test_forward_last_logits_single_token() {
        let cfg = tiny_config();
        let model = AyaForCausalLm::new(&cfg).expect("construction");
        let result = model.forward_last_logits(&[0_u32]);
        assert!(
            result.is_ok(),
            "single token should succeed: {:?}",
            result.err()
        );
        let logits = result.unwrap();
        assert_eq!(logits.len(), cfg.vocab_size);
    }

    // ── Test 26: rope_theta field value ──────────────────────────────────────
    #[test]
    fn test_aya_rope_theta() {
        let cfg = AyaConfig::default();
        assert!(
            (cfg.rope_theta - 10000.0_f64).abs() < 1e-3,
            "Aya-23 uses rope_theta=10000 (Command-R base), got {}",
            cfg.rope_theta
        );
    }

    // ── Test 27: tie_word_embeddings default false ────────────────────────────
    #[test]
    fn test_aya_tie_word_embeddings_default() {
        let cfg = AyaConfig::default();
        assert!(
            !cfg.tie_word_embeddings,
            "tie_word_embeddings should default to false"
        );
    }

    // ── Test 28: tokenizer_class field ───────────────────────────────────────
    #[test]
    fn test_aya_tokenizer_class_field() {
        let cfg = AyaConfig::default();
        assert!(
            !cfg.tokenizer_class.is_empty(),
            "tokenizer_class must not be empty"
        );
        assert!(
            cfg.tokenizer_class.contains("Tokenizer"),
            "tokenizer_class should contain 'Tokenizer', got: {}",
            cfg.tokenizer_class
        );
    }

    // ── Test 29: LayerNorm variance normalisation ─────────────────────────────
    #[test]
    fn test_layer_norm_unit_variance() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0_f32; 4];
        let bias = vec![0.0_f32; 4];
        let out = AyaLayerNorm::forward(&x, &weight, &bias, 1e-5).expect("LayerNorm forward");
        // Variance of the output should be close to 1.0
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        let var: f32 = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / out.len() as f32;
        assert!(
            (var - 1.0_f32).abs() < 0.1,
            "LayerNorm output should have variance ≈ 1, got {var}"
        );
    }
}
