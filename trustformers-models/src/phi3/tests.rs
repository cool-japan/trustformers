use super::*;
use trustformers_core::Config;

#[test]
fn test_phi3_config_validation() {
    let config = Phi3Config::phi3_mini_4k_instruct();
    assert!(config.validate().is_ok());

    // Test invalid config
    let mut invalid_config = config.clone();
    invalid_config.hidden_size = 3071; // Not divisible by num_attention_heads (32)
    assert!(invalid_config.validate().is_err());
}

#[test]
fn test_phi3_config_presets() {
    // Test Mini configurations
    let mini_4k = Phi3Config::phi3_mini_4k_instruct();
    assert_eq!(mini_4k.hidden_size, 3072);
    assert_eq!(mini_4k.num_hidden_layers, 32);
    assert_eq!(mini_4k.max_position_embeddings, 4096);
    assert!(mini_4k.is_instruct_model());
    assert!(!mini_4k.is_long_context());

    let mini_128k = Phi3Config::phi3_mini_128k_instruct();
    assert_eq!(mini_128k.max_position_embeddings, 131072);
    assert!(mini_128k.is_long_context());
    assert!(mini_128k.rope_scaling.is_some());

    // Test Small configurations
    let small_8k = Phi3Config::phi3_small_8k_instruct();
    assert_eq!(small_8k.hidden_size, 4096);
    assert_eq!(small_8k.vocab_size, 100352);
    assert_eq!(small_8k.max_position_embeddings, 8192);

    // Test Medium configurations
    let medium_4k = Phi3Config::phi3_medium_4k_instruct();
    assert_eq!(medium_4k.hidden_size, 5120);
    assert_eq!(medium_4k.num_hidden_layers, 40);
    assert_eq!(medium_4k.num_key_value_heads, Some(10)); // GQA
}

#[test]
fn test_phi3_config_from_pretrained() {
    let config = Phi3Config::from_pretrained_name("microsoft/Phi-3-mini-4k-instruct");
    assert!(config.is_some());
    let config = config.unwrap();
    assert_eq!(config.model_type, "phi3-mini-instruct");

    let config = Phi3Config::from_pretrained_name("microsoft/Phi-3-small-128k-instruct");
    assert!(config.is_some());
    let config = config.unwrap();
    assert!(config.is_long_context());

    let config = Phi3Config::from_pretrained_name("unknown-model");
    assert!(config.is_none());
}

#[test]
fn test_phi3_config_helpers() {
    let config = Phi3Config::phi3_mini_4k_instruct();
    assert_eq!(config.head_dim(), 96); // 3072 / 32
    assert_eq!(config.num_kv_heads(), 32); // No GQA in mini
    assert_eq!(config.num_query_groups(), 1);

    let medium_config = Phi3Config::phi3_medium_4k_instruct();
    assert_eq!(medium_config.head_dim(), 128); // 5120 / 40
    assert_eq!(medium_config.num_kv_heads(), 10); // GQA with 10 KV heads
    assert_eq!(medium_config.num_query_groups(), 4); // 40 / 10
}

#[test]
fn test_phi3_config_effective_context() {
    let config = Phi3Config::phi3_mini_4k_instruct();
    assert_eq!(config.effective_context_length(), 4096);

    let long_config = Phi3Config::phi3_mini_128k_instruct();
    assert_eq!(long_config.effective_context_length(), 131072);
}

#[test]
fn test_rms_norm_creation() {
    let norm = RMSNorm::new(768, 1e-5);
    assert!(norm.is_ok());
}

#[test]
fn test_phi3_mlp_creation() {
    let config = Phi3Config::phi3_mini_4k_instruct();
    let mlp = Phi3MLP::new(&config);
    assert!(mlp.is_ok());
}

#[test]
fn test_phi3_attention_creation() {
    let config = Phi3Config::phi3_mini_4k_instruct();
    let attention = Phi3Attention::new(&config);
    assert!(attention.is_ok());
}

#[test]
fn test_phi3_decoder_layer_creation() {
    let config = Phi3Config::phi3_mini_4k_instruct();
    let layer = Phi3DecoderLayer::new(&config);
    assert!(layer.is_ok());
}

#[test]
fn test_phi3_model_creation() {
    let config = Phi3Config::phi3_mini_4k_instruct();
    let model = Phi3Model::new(config.clone());
    assert!(model.is_ok());

    let model = model.unwrap();
    assert_eq!(model.config().hidden_size, config.hidden_size);
}

#[test]
fn test_phi3_causal_lm_creation() {
    let config = Phi3Config::phi3_mini_4k_instruct();
    let model = Phi3ForCausalLM::new(config.clone());
    assert!(model.is_ok());

    let model = model.unwrap();
    assert_eq!(model.config().vocab_size, config.vocab_size);
}

#[test]
fn test_phi3_forward_shape() {
    let config = Phi3Config::phi3_mini_4k_instruct();
    let model = Phi3Model::new(config);
    assert!(model.is_ok());

    // Test would require actual tensor operations
    // Placeholder for integration tests
}

#[test]
fn test_rope_scaling_types() {
    let mini_config = Phi3Config::phi3_mini_4k_instruct();
    assert!(mini_config.rope_scaling.is_none());

    let long_config = Phi3Config::phi3_mini_128k_instruct();
    assert!(long_config.rope_scaling.is_some());

    let scaling = long_config.rope_scaling.unwrap();
    assert_eq!(scaling.scaling_type, "longrope");
    assert!(scaling.long_factor.is_some());
    assert!(scaling.short_factor.is_some());
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_all_phi3_variants() {
        // Test that all preset configurations are valid
        let configs = vec![
            Phi3Config::phi3_mini_3_8b(),
            Phi3Config::phi3_mini_4k_instruct(),
            Phi3Config::phi3_mini_128k_instruct(),
            Phi3Config::phi3_small_7b(),
            Phi3Config::phi3_small_8k_instruct(),
            Phi3Config::phi3_small_128k_instruct(),
            Phi3Config::phi3_medium_14b(),
            Phi3Config::phi3_medium_4k_instruct(),
            Phi3Config::phi3_medium_128k_instruct(),
        ];

        for config in configs {
            assert!(
                config.validate().is_ok(),
                "Config validation failed for: {}",
                config.model_type
            );

            // Test model creation
            let model = Phi3Model::new(config.clone());
            assert!(
                model.is_ok(),
                "Model creation failed for: {}",
                config.model_type
            );

            // Test causal LM creation
            let causal_lm = Phi3ForCausalLM::new(config.clone());
            assert!(
                causal_lm.is_ok(),
                "CausalLM creation failed for: {}",
                config.model_type
            );
        }
    }
}
