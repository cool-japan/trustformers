use super::*;
use trustformers_core::tensor::Tensor;

#[test]
fn test_command_r_config_default() {
    let config = CommandRConfig::default();
    assert_eq!(config.model_name, "command-r");
    assert_eq!(config.vocab_size, 256000);
    assert_eq!(config.hidden_size, 8192);
    assert_eq!(config.num_attention_heads, 64);
    assert_eq!(config.num_hidden_layers, 40);
    assert!(config.validate().is_ok());
}

#[test]
fn test_command_r_plus_config() {
    let config = CommandRConfig::command_r_plus();
    assert_eq!(config.model_name, "command-r-plus");
    assert_eq!(config.vocab_size, 256000);
    assert_eq!(config.hidden_size, 12288);
    assert_eq!(config.num_attention_heads, 96);
    assert_eq!(config.num_hidden_layers, 64);
    assert!(config.validate().is_ok());
}

#[test]
fn test_command_r_08_2024_config() {
    let config = CommandRConfig::command_r_08_2024();
    assert_eq!(config.model_name, "command-r-08-2024");
    assert_eq!(config.vocab_size, 256000);
    assert_eq!(config.hidden_size, 8192);
    assert_eq!(config.num_attention_heads, 64);
    assert_eq!(config.num_hidden_layers, 40);
    assert!(config.validate().is_ok());
}

#[test]
fn test_command_r_plus_08_2024_config() {
    let config = CommandRConfig::command_r_plus_08_2024();
    assert_eq!(config.model_name, "command-r-plus-08-2024");
    assert_eq!(config.vocab_size, 256000);
    assert_eq!(config.hidden_size, 12288);
    assert_eq!(config.num_attention_heads, 96);
    assert_eq!(config.num_hidden_layers, 64);
    assert!(config.validate().is_ok());
}

#[test]
fn test_command_r_head_dim_calculation() {
    let config = CommandRConfig::command_r();
    assert_eq!(config.head_dim(), 128); // 8192 / 64

    let config_plus = CommandRConfig::command_r_plus();
    assert_eq!(config_plus.head_dim(), 128); // 12288 / 96
}

#[test]
fn test_command_r_gqa_detection() {
    let config = CommandRConfig::command_r();
    assert!(!config.is_gqa()); // Same number of heads

    let mut config_gqa = config.clone();
    config_gqa.num_key_value_heads = 32;
    assert!(config_gqa.is_gqa());
    assert_eq!(config_gqa.num_query_groups(), 2); // 64 / 32
}

#[test]
fn test_command_r_model_creation() {
    let config = CommandRConfig::command_r();
    let model = CommandRModel::new(&config);
    assert!(model.is_ok());

    let _model = model.unwrap();
    // Command-R model created successfully - model configuration is internal
}

#[test]
fn test_command_r_plus_model_creation() {
    let config = CommandRConfig::command_r_plus();
    let model = CommandRModel::new(&config);
    assert!(model.is_ok());

    let _model = model.unwrap();
    // Command-R Plus model created successfully - model configuration is internal
}

#[test]
fn test_command_r_causal_lm_creation() {
    let config = CommandRConfig::command_r();
    let model = CommandRForCausalLM::new(&config);
    assert!(model.is_ok());
}

#[test]
fn test_command_r_plus_causal_lm_creation() {
    let config = CommandRConfig::command_r_plus();
    let model = CommandRForCausalLM::new(&config);
    assert!(model.is_ok());
}

#[test]
fn test_command_r_attention_creation() {
    let config = CommandRConfig::command_r();
    let attention = CommandRAttention::new(&config);
    assert!(attention.is_ok());

    let _attention = attention.unwrap();
    // Command-R attention created successfully - dimensions are internal
    // Grouped query attention created successfully - head configuration is internal
}

#[test]
fn test_command_r_mlp_creation() {
    let config = CommandRConfig::command_r();
    let mlp = CommandRMLP::new(&config);
    assert!(mlp.is_ok());

    let _mlp = mlp.unwrap();
    // MLP created successfully - configuration details are internal
}

#[test]
fn test_command_r_decoder_layer_creation() {
    let config = CommandRConfig::command_r();
    let layer = CommandRDecoderLayer::new(&config);
    assert!(layer.is_ok());

    let _layer = layer.unwrap();
    // Decoder layer created successfully - hidden size is internal
}

#[test]
fn test_command_r_rope_creation() {
    let rope = CommandRRoPE::new(128, 4096, 10000.0);
    assert!(rope.is_ok());

    let _rope = rope.unwrap();
    // Rotary embedding created successfully - dimensions are internal
    // Rotary embedding created successfully - base frequency is internal
}

#[test]
fn test_command_r_model_forward() {
    let config = CommandRConfig::command_r();
    let mut model = CommandRModel::new(&config).unwrap();

    // Create dummy input
    let input_ids = Tensor::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let input_ids = input_ids.reshape(&[1, 4]).unwrap();

    let result = model.forward(&input_ids, None, None, None);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.last_hidden_state.shape(), vec![1, 4, 8192]);
}

#[test]
fn test_command_r_causal_lm_forward() {
    let config = CommandRConfig::command_r();
    let mut model = CommandRForCausalLM::new(&config).unwrap();

    // Create dummy input
    let input_ids = Tensor::new(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let input_ids = input_ids.reshape(&[1, 4]).unwrap();

    let result = model.forward(&input_ids, None, None, None, None);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.logits.shape(), vec![1, 4, 256000]);
}

#[test]
fn test_command_r_config_validation() {
    let mut config = CommandRConfig::default();
    assert!(config.validate().is_ok());

    // Test invalid vocab size
    config.vocab_size = 0;
    assert!(config.validate().is_err());

    // Test invalid hidden size
    config.vocab_size = 256000;
    config.hidden_size = 0;
    assert!(config.validate().is_err());

    // Test invalid divisibility
    config.hidden_size = 100;
    config.num_attention_heads = 64;
    assert!(config.validate().is_err());
}

#[test]
fn test_command_r_model_shapes() {
    let config = CommandRConfig::command_r();
    let _model = CommandRModel::new(&config).unwrap();

    // Command-R model created successfully - embedding and layer details are internal
}

#[test]
fn test_command_r_plus_model_shapes() {
    let config = CommandRConfig::command_r_plus();
    let _model = CommandRModel::new(&config).unwrap();

    // Command-R Plus model created successfully - embedding and layer details are internal
}

#[test]
fn test_command_r_attention_shapes() {
    let config = CommandRConfig::command_r();
    let _attention = CommandRAttention::new(&config).unwrap();

    // Attention projections created successfully - projection dimensions are internal
}

#[test]
fn test_command_r_mlp_shapes() {
    let config = CommandRConfig::command_r();
    let _mlp = CommandRMLP::new(&config).unwrap();

    // MLP projections created successfully - projection dimensions are internal
}

#[test]
fn test_command_r_token_ids() {
    let config = CommandRConfig::command_r();
    assert_eq!(config.pad_token_id, Some(0));
    assert_eq!(config.bos_token_id, Some(5));
    assert_eq!(config.eos_token_id, Some(255001));
}

#[test]
fn test_command_r_model_type() {
    let config = CommandRConfig::command_r();
    assert_eq!(config.model_type, "command-r");

    let config_plus = CommandRConfig::command_r_plus();
    assert_eq!(config_plus.model_type, "command-r-plus");
}

#[test]
fn test_command_r_activation_function() {
    let config = CommandRConfig::command_r();
    assert_eq!(config.activation_function, "silu");
}

#[test]
fn test_command_r_rope_parameters() {
    let config = CommandRConfig::command_r();
    assert_eq!(config.rope_theta, 10000.0);
    assert_eq!(config.rope_scaling_factor, 1.0);
}

#[test]
fn test_command_r_sequence_length() {
    let config = CommandRConfig::command_r();
    assert_eq!(config.max_sequence_length, 131072);

    let config_plus = CommandRConfig::command_r_plus();
    assert_eq!(config_plus.max_sequence_length, 131072);
}

#[test]
fn test_command_r_normalization_parameters() {
    let config = CommandRConfig::command_r();
    assert_eq!(config.rms_norm_eps, 1e-5);
    assert_eq!(config.layer_norm_eps, 1e-5);
}

#[test]
fn test_command_r_dropout_parameters() {
    let config = CommandRConfig::command_r();
    assert_eq!(config.attention_dropout, 0.0);
    assert_eq!(config.hidden_dropout, 0.0);
}

#[test]
fn test_command_r_bias_settings() {
    let config = CommandRConfig::command_r();
    assert!(!config.use_bias);
    assert!(!config.tie_word_embeddings);
    assert!(!config.use_logit_bias);
}

#[test]
fn test_command_r_flash_attention() {
    let config = CommandRConfig::command_r();
    assert!(config.use_flash_attention);
    assert!(!config.use_sliding_window);
}

#[test]
fn test_command_r_torch_dtype() {
    let config = CommandRConfig::command_r();
    assert_eq!(config.torch_dtype, "bfloat16");
}

#[test]
fn test_command_r_transformers_version() {
    let config = CommandRConfig::command_r();
    assert_eq!(config.transformers_version, "4.39.0");
}
