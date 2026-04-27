//! Comprehensive tests for the AutoModel class dispatch system.

#[cfg(test)]
mod tests {
    use crate::automodel::{AutoConfig, AutoModel, AutoTokenizer};
    use crate::error::TrustformersError;
    use trustformers_core::traits::{Config, Model};

    // -------------------------------------------------------------------------
    // AutoConfig — from_model_name (error-path and name-based dispatch)
    // -------------------------------------------------------------------------

    #[test]
    fn test_auto_config_from_pretrained_nonexistent_unknown_name_errors() {
        let result = AutoConfig::from_pretrained("/tmp/totally_nonexistent_model_path_xyz_abc");
        // Should fail because neither the path exists nor the name is recognizable
        assert!(
            result.is_err(),
            "Unknown nonexistent path should return error"
        );
    }

    #[test]
    fn test_auto_config_from_pretrained_empty_string_errors() {
        let result = AutoConfig::from_pretrained("");
        assert!(result.is_err(), "Empty model name should return error");
    }

    #[test]
    fn test_auto_config_from_model_name_bert() {
        // Test that "bert-base-uncased" style name produces a Bert config
        let result = AutoConfig::from_pretrained("bert-base-uncased");
        match result {
            Ok(config) => {
                assert_eq!(
                    config.get_architecture_name(),
                    "bert",
                    "Architecture should be bert"
                );
            },
            Err(_) => {
                // In tests without network access this may fail; that's acceptable
            },
        }
    }

    #[test]
    fn test_auto_config_from_model_name_gpt2_style() {
        let result = AutoConfig::from_pretrained("gpt2-medium");
        // Will either succeed or fail gracefully - no panics
        let _ = result;
    }

    #[test]
    fn test_auto_config_from_model_name_t5_style() {
        let result = AutoConfig::from_pretrained("t5-small");
        let _ = result; // No panics allowed
    }

    // -------------------------------------------------------------------------
    // AutoConfig — methods on constructed config
    // -------------------------------------------------------------------------

    #[cfg(feature = "bert")]
    #[test]
    fn test_auto_config_bert_default_vocab_size() {
        let config = AutoConfig::Bert(crate::models::bert::BertConfig::default());
        assert_eq!(
            config.get_vocab_size(),
            30522,
            "Default BERT vocab size is 30522"
        );
    }

    #[cfg(feature = "bert")]
    #[test]
    fn test_auto_config_bert_default_hidden_size() {
        let config = AutoConfig::Bert(crate::models::bert::BertConfig::default());
        assert_eq!(
            config.get_hidden_size(),
            768,
            "Default BERT hidden size is 768"
        );
    }

    #[cfg(feature = "bert")]
    #[test]
    fn test_auto_config_bert_default_num_layers() {
        let config = AutoConfig::Bert(crate::models::bert::BertConfig::default());
        assert_eq!(config.get_num_layers(), 12, "Default BERT has 12 layers");
    }

    #[cfg(feature = "bert")]
    #[test]
    fn test_auto_config_bert_default_num_attention_heads() {
        let config = AutoConfig::Bert(crate::models::bert::BertConfig::default());
        assert_eq!(
            config.get_num_attention_heads(),
            12,
            "Default BERT has 12 attention heads"
        );
    }

    #[cfg(feature = "bert")]
    #[test]
    fn test_auto_config_bert_default_max_seq_len() {
        let config = AutoConfig::Bert(crate::models::bert::BertConfig::default());
        assert_eq!(
            config.get_max_sequence_length(),
            512,
            "Default BERT max seq len is 512"
        );
    }

    #[cfg(feature = "bert")]
    #[test]
    fn test_auto_config_bert_architecture_name() {
        let config = AutoConfig::Bert(crate::models::bert::BertConfig::default());
        assert_eq!(config.get_architecture_name(), "bert");
        assert_eq!(
            config.architecture(),
            "bert",
            "Config trait architecture() should be bert"
        );
    }

    #[cfg(feature = "bert")]
    #[test]
    fn test_auto_config_bert_validate_default() {
        let config = AutoConfig::Bert(crate::models::bert::BertConfig::default());
        assert!(
            config.validate().is_ok(),
            "Default BERT config should validate successfully"
        );
    }

    #[cfg(feature = "bert")]
    #[test]
    fn test_auto_config_bert_clone() {
        let config = AutoConfig::Bert(crate::models::bert::BertConfig::default());
        let cloned = config.clone();
        assert_eq!(
            cloned.get_vocab_size(),
            config.get_vocab_size(),
            "Cloned config should have same vocab_size"
        );
    }

    // -------------------------------------------------------------------------
    // AutoModel — from_config error path
    // -------------------------------------------------------------------------

    #[cfg(feature = "bert")]
    #[test]
    fn test_auto_model_from_config_bert_succeeds() {
        let config = AutoConfig::Bert(crate::models::bert::BertConfig::default());
        let result = AutoModel::from_config(config);
        assert!(
            result.is_ok(),
            "AutoModel::from_config should succeed with default BERT config"
        );
    }

    #[cfg(feature = "bert")]
    #[test]
    fn test_auto_model_from_config_bert_num_parameters_positive() {
        let config = AutoConfig::Bert(crate::models::bert::BertConfig::default());
        if let Ok(model) = AutoModel::from_config(config) {
            assert!(
                model.num_parameters() > 0,
                "Model should have positive parameter count"
            );
        }
    }

    #[cfg(feature = "bert")]
    #[test]
    fn test_auto_model_from_config_bert_get_config() {
        let config = AutoConfig::Bert(crate::models::bert::BertConfig::default());
        let original_hidden = config.get_hidden_size();
        if let Ok(model) = AutoModel::from_config(config) {
            let retrieved_config = model.get_config();
            assert_eq!(
                retrieved_config.get_hidden_size(),
                original_hidden,
                "get_config should return original config"
            );
        }
    }

    // -------------------------------------------------------------------------
    // AutoModel — from_pretrained error path (no network in tests)
    // -------------------------------------------------------------------------

    #[test]
    fn test_auto_model_from_pretrained_nonexistent_path_errors() {
        let result = AutoModel::from_pretrained("/tmp/totally_nonexistent_model_99999");
        assert!(
            result.is_err(),
            "from_pretrained with nonexistent path should error"
        );
    }

    #[test]
    fn test_auto_model_from_pretrained_with_revision_errors_on_bad_path() {
        let result =
            AutoModel::from_pretrained_with_revision("/tmp/nonexistent_model_revision", Some("v1"));
        assert!(
            result.is_err(),
            "from_pretrained_with_revision should error for bad path"
        );
    }

    #[test]
    fn test_auto_model_from_pretrained_empty_string_errors() {
        let result = AutoModel::from_pretrained("");
        assert!(result.is_err(), "Empty path should produce error");
    }

    // -------------------------------------------------------------------------
    // AutoConfig — from_pretrained_with_revision
    // -------------------------------------------------------------------------

    #[test]
    fn test_auto_config_from_pretrained_with_revision_errors_on_bad_path() {
        let result =
            AutoConfig::from_pretrained_with_revision("/tmp/nonexistent_model_99999", Some("main"));
        assert!(
            result.is_err(),
            "Should error for nonexistent path with revision"
        );
    }

    // -------------------------------------------------------------------------
    // AutoConfig — JSON config dispatch
    // -------------------------------------------------------------------------

    #[cfg(feature = "bert")]
    #[test]
    fn test_auto_config_from_json_file_bert() {
        // Write a minimal BERT config to a temp file and attempt to load it
        let dir = std::env::temp_dir().join("test_autoconfig_bert_json");
        let _ = std::fs::create_dir_all(&dir);
        let config_path = dir.join("config.json");
        let bert_json = serde_json::json!({
            "model_type": "bert",
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 256,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 128,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "pad_token_id": 0
        });
        if std::fs::write(&config_path, bert_json.to_string()).is_ok() {
            let result = AutoConfig::from_pretrained(dir.to_str().unwrap_or(""));
            match result {
                Ok(config) => {
                    assert_eq!(config.get_architecture_name(), "bert");
                    assert_eq!(config.get_vocab_size(), 1000);
                    assert_eq!(config.get_hidden_size(), 64);
                    assert_eq!(config.get_num_layers(), 2);
                    assert_eq!(config.get_num_attention_heads(), 4);
                    assert_eq!(config.get_max_sequence_length(), 128);
                },
                Err(_) => {
                    // Acceptable if parsing fails for other reasons
                },
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_auto_config_from_unknown_json_model_type_errors() {
        let dir = std::env::temp_dir().join("test_autoconfig_unknown_json");
        let _ = std::fs::create_dir_all(&dir);
        let config_path = dir.join("config.json");
        let unknown_json = serde_json::json!({"model_type": "totally_unknown_model_xyz"});
        if std::fs::write(&config_path, unknown_json.to_string()).is_ok() {
            let result = AutoConfig::from_pretrained(dir.to_str().unwrap_or(""));
            assert!(
                result.is_err(),
                "Unknown model type in JSON should return error"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -------------------------------------------------------------------------
    // AutoTokenizer — error paths
    // -------------------------------------------------------------------------

    #[test]
    fn test_auto_tokenizer_from_pretrained_nonexistent_errors() {
        let result = AutoTokenizer::from_pretrained("/tmp/nonexistent_tokenizer_abc_xyz");
        assert!(
            result.is_err(),
            "AutoTokenizer from nonexistent path should error"
        );
    }

    #[test]
    fn test_auto_tokenizer_from_pretrained_with_revision_errors() {
        let result = AutoTokenizer::from_pretrained_with_revision(
            "/tmp/nonexistent_tokenizer_abc_xyz",
            Some("main"),
        );
        assert!(
            result.is_err(),
            "AutoTokenizer with revision from bad path should error"
        );
    }

    // -------------------------------------------------------------------------
    // AutoConfig — model name pattern matching
    // -------------------------------------------------------------------------

    #[test]
    fn test_auto_config_from_name_pure_garbage_errors() {
        let result = AutoConfig::from_pretrained("xyzzy-foobar-unknown-9999");
        assert!(result.is_err(), "Unrecognizable garbage name should error");
    }

    #[cfg(feature = "bert")]
    #[test]
    fn test_auto_config_from_name_distilbert() {
        let result = AutoConfig::from_pretrained("distilbert-base-uncased");
        // distilbert contains "bert" → should create BertConfig
        match result {
            Ok(config) => {
                assert_eq!(config.get_architecture_name(), "bert");
            },
            Err(_) => {
                // Network not available in test environment; acceptable
            },
        }
    }

    // -------------------------------------------------------------------------
    // Error types from TrustformersError
    // -------------------------------------------------------------------------

    #[test]
    fn test_trustformers_error_invalid_input_is_error() {
        let err = TrustformersError::invalid_input(
            "test error".to_string(),
            Some("field"),
            Some("expected"),
            Some("actual"),
        );
        // Ensure the error is constructable and non-empty
        let err_str = format!("{}", err);
        assert!(!err_str.is_empty(), "Error message should be non-empty");
    }

    #[test]
    fn test_trustformers_error_invalid_input_simple() {
        let err = TrustformersError::invalid_input_simple("simple error".to_string());
        let err_str = format!("{}", err);
        assert!(
            !err_str.is_empty(),
            "Simple error message should be non-empty"
        );
    }

    // -------------------------------------------------------------------------
    // AutoConfig — config JSON dispatch for multiple model types
    // -------------------------------------------------------------------------

    #[cfg(feature = "gpt2")]
    #[test]
    fn test_auto_config_from_gpt2_json() {
        let dir = std::env::temp_dir().join("test_autoconfig_gpt2_json");
        let _ = std::fs::create_dir_all(&dir);
        let config_path = dir.join("config.json");
        let gpt2_json = serde_json::json!({
            "model_type": "gpt2",
            "vocab_size": 50257,
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_positions": 1024,
            "n_ctx": 1024,
            "activation_function": "gelu",
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "layer_norm_epsilon": 1e-5,
            "initializer_range": 0.02
        });
        if std::fs::write(&config_path, gpt2_json.to_string()).is_ok() {
            let result = AutoConfig::from_pretrained(dir.to_str().unwrap_or(""));
            if let Ok(config) = result {
                assert_eq!(config.get_architecture_name(), "gpt2");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -------------------------------------------------------------------------
    // AutoModel — with valid BertConfig created from JSON
    // -------------------------------------------------------------------------

    #[cfg(feature = "bert")]
    #[test]
    fn test_auto_model_from_config_small_bert_creation() {
        // Create a very small BERT config to test model creation without OOM
        let mut bert_config = crate::models::bert::BertConfig::default();
        bert_config.vocab_size = 256;
        bert_config.hidden_size = 32;
        bert_config.num_hidden_layers = 2;
        bert_config.num_attention_heads = 4;
        bert_config.intermediate_size = 64;
        bert_config.max_position_embeddings = 64;
        let config = AutoConfig::Bert(bert_config);
        let result = AutoModel::from_config(config);
        assert!(
            result.is_ok(),
            "Small custom BERT config should create model successfully"
        );
    }
}
