use super::*;
use trustformers_core::traits::Config;

#[test]
fn test_claude_config_creation() {
    let config = ClaudeConfig::default();
    assert_eq!(config.vocab_size, 100352);
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_hidden_layers, 32);
    assert_eq!(config.num_attention_heads, 32);
    assert!(config.constitutional_ai);
}

#[test]
fn test_claude_3_variants() {
    let haiku = ClaudeConfig::claude_3_haiku();
    assert_eq!(haiku.hidden_size, 3072);
    assert_eq!(haiku.num_hidden_layers, 24);
    assert_eq!(haiku.model_type, "claude-3-haiku");

    let sonnet = ClaudeConfig::claude_3_sonnet();
    assert_eq!(sonnet.hidden_size, 4096);
    assert_eq!(sonnet.num_hidden_layers, 32);
    assert_eq!(sonnet.model_type, "claude-3-sonnet");

    let opus = ClaudeConfig::claude_3_opus();
    assert_eq!(opus.hidden_size, 8192);
    assert_eq!(opus.num_hidden_layers, 64);
    assert_eq!(opus.model_type, "claude-3-opus");
}

#[test]
fn test_constitutional_ai_weights() {
    let mut config = ClaudeConfig::default();
    config.with_constitutional_weights(1.5, 1.0, 1.2);

    assert_eq!(config.harmlessness_weight, 1.5);
    assert_eq!(config.helpfulness_weight, 1.0);
    assert_eq!(config.honesty_weight, 1.2);
}

#[test]
fn test_config_validation() {
    let config = ClaudeConfig::default();
    assert!(config.validate().is_ok());

    // Test invalid configuration
    let mut invalid_config = ClaudeConfig::default();
    invalid_config.hidden_size = 4095; // Not divisible by num_attention_heads
    assert!(invalid_config.validate().is_err());
}

#[test]
fn test_from_pretrained_name() {
    let config = ClaudeConfig::from_pretrained_name("claude-3-sonnet");
    assert!(config.is_some());
    let config = config.expect("operation failed");
    assert_eq!(config.model_type, "claude-3-sonnet");

    let config = ClaudeConfig::from_pretrained_name("invalid-model");
    assert!(config.is_none());
}

#[test]
fn test_head_calculations() {
    let config = ClaudeConfig::claude_3_sonnet();
    assert_eq!(config.head_dim(), 128); // 4096 / 32
    assert_eq!(config.num_kv_heads(), 8); // Uses grouped-query attention
    assert_eq!(config.num_query_groups(), 4); // 32 / 8
}

#[test]
fn test_claude_model_creation() {
    let config = ClaudeConfig::small_test_config();
    let model = ClaudeModel::new(config);
    assert!(model.is_ok());

    let _model = model.expect("operation failed");
    // Claude model created successfully - layer count and constitutional weights are internal
}

#[test]
fn test_claude_for_causal_lm_creation() {
    let config = ClaudeConfig::small_test_config();
    let model = ClaudeForCausalLM::new(config);
    assert!(model.is_ok());
}

#[test]
fn test_constitutional_ai_disabled() {
    let mut config = ClaudeConfig::small_test_config();
    config.with_constitutional_ai(false);

    let _model = ClaudeModel::new(config).expect("operation failed");
    // Claude model created successfully with constitutional AI disabled - weights are internal
}

#[test]
fn test_rope_scaling() {
    let mut config = ClaudeConfig::default();
    config.rope_scaling = Some(RopeScaling {
        scaling_type: "linear".to_string(),
        scaling_factor: 2.0,
    });

    assert!(config.validate().is_ok());
}

#[test]
fn test_grouped_query_attention() {
    let config = ClaudeConfig::claude_3_opus();
    assert_eq!(config.num_key_value_heads, Some(8));
    assert_eq!(config.num_kv_heads(), 8);
    assert_eq!(config.num_query_groups(), 8); // 64 / 8
}

#[test]
fn test_context_length_variants() {
    let claude_1 = ClaudeConfig::claude_1();
    assert_eq!(claude_1.max_position_embeddings, 8192);

    let claude_2 = ClaudeConfig::claude_2();
    assert_eq!(claude_2.max_position_embeddings, 100000);

    let claude_2_1 = ClaudeConfig::claude_2_1();
    assert_eq!(claude_2_1.max_position_embeddings, 200000);

    let claude_3_sonnet = ClaudeConfig::claude_3_sonnet();
    assert_eq!(claude_3_sonnet.max_position_embeddings, 200000);
}

#[test]
fn test_constitutional_ai_validation() {
    let mut config = ClaudeConfig::default();
    config.with_constitutional_weights(-1.0, 1.0, 1.0);

    assert!(config.validate().is_err()); // Negative weights should fail

    config.with_constitutional_weights(1.0, 1.0, 1.0);
    assert!(config.validate().is_ok());
}

#[test]
fn test_rotary_embedding_creation() {
    let _rope = RotaryEmbedding::new(128, 8192, 10000.0);
    // RoPE created successfully - implementation details are private
}

#[test]
fn test_claude_attention_creation() {
    // Use small_test_config for fast testing
    let config = ClaudeConfig::small_test_config();
    let attention = ClaudeAttention::new(config);
    assert!(attention.is_ok());
}

#[test]
fn test_claude_mlp_creation() {
    // Use small_test_config for fast testing
    let config = ClaudeConfig::small_test_config();
    let mlp = ClaudeMLP::new(config);
    assert!(mlp.is_ok());
}

#[test]
fn test_claude_decoder_layer_creation() {
    // Use small_test_config for fast testing
    let config = ClaudeConfig::small_test_config();
    let layer = ClaudeDecoderLayer::new(config);
    assert!(layer.is_ok());
}

// Full model size tests - ignored by default due to memory/time requirements
#[test]
#[ignore = "Full model size test - requires significant memory and time"]
fn test_claude_attention_creation_haiku() {
    let config = ClaudeConfig::claude_3_haiku();
    let attention = ClaudeAttention::new(config);
    assert!(attention.is_ok());
}

#[test]
#[ignore = "Full model size test - requires significant memory and time"]
fn test_claude_mlp_creation_sonnet() {
    let config = ClaudeConfig::claude_3_sonnet();
    let mlp = ClaudeMLP::new(config);
    assert!(mlp.is_ok());
}

#[test]
#[ignore = "Full model size test - requires significant memory and time"]
fn test_claude_decoder_layer_creation_opus() {
    let config = ClaudeConfig::claude_3_opus();
    let layer = ClaudeDecoderLayer::new(config);
    assert!(layer.is_ok());
}

#[test]
fn test_enhanced_rope_theta() {
    let config = ClaudeConfig::claude_3_5_sonnet();
    assert_eq!(config.rope_theta, 500000.0); // Enhanced RoPE
}

#[test]
fn test_model_type_strings() {
    assert_eq!(ClaudeConfig::claude_1().model_type, "claude-1");
    assert_eq!(ClaudeConfig::claude_2().model_type, "claude-2");
    assert_eq!(ClaudeConfig::claude_2_1().model_type, "claude-2.1");
    assert_eq!(ClaudeConfig::claude_3_haiku().model_type, "claude-3-haiku");
    assert_eq!(
        ClaudeConfig::claude_3_sonnet().model_type,
        "claude-3-sonnet"
    );
    assert_eq!(ClaudeConfig::claude_3_opus().model_type, "claude-3-opus");
    assert_eq!(
        ClaudeConfig::claude_3_5_sonnet().model_type,
        "claude-3.5-sonnet"
    );
}
