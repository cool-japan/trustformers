use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// Falcon model configuration
/// Reference: "The Falcon Series of Open Language Models" (Almazrouei et al., 2023)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalconConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: Option<usize>, // For multi-query attention
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub initializer_range: f32,
    pub layer_norm_epsilon: f32,
    pub use_cache: bool,
    pub pad_token_id: Option<u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub apply_residual_connection_post_layernorm: bool,
    pub hidden_dropout: f32,
    pub attention_dropout: f32,
    pub model_type: String,
    pub parallel_attn: bool, // Parallel attention and MLP
    pub bias: bool,
    pub multi_query: bool,                 // Use multi-query attention
    pub alibi: bool,                       // Use ALiBi positional encoding
    pub new_decoder_architecture: bool,    // Use new decoder architecture
    pub use_flash_attention: Option<bool>, // Enable FlashAttention
}

impl Default for FalconConfig {
    fn default() -> Self {
        Self {
            vocab_size: 65024,
            hidden_size: 4544,
            num_hidden_layers: 32,
            num_attention_heads: 71,
            num_kv_heads: Some(1), // Multi-query attention by default
            hidden_act: "gelu".to_string(),
            max_position_embeddings: 2048,
            initializer_range: 0.02,
            layer_norm_epsilon: 1e-5,
            use_cache: true,
            pad_token_id: Some(0),
            bos_token_id: 1,
            eos_token_id: 2,
            apply_residual_connection_post_layernorm: false,
            hidden_dropout: 0.0,
            attention_dropout: 0.0,
            model_type: "falcon".to_string(),
            parallel_attn: true,
            bias: false,
            multi_query: true,
            alibi: false,
            new_decoder_architecture: false,
            use_flash_attention: None,
        }
    }
}

impl Config for FalconConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(trustformers_core::errors::TrustformersError::config_error(
                "hidden_size must be divisible by num_attention_heads",
                "FalconConfig::validate",
            ));
        }

        if let Some(num_kv_heads) = self.num_kv_heads {
            if self.num_attention_heads % num_kv_heads != 0 {
                return Err(trustformers_core::errors::TrustformersError::config_error(
                    "num_attention_heads must be divisible by num_kv_heads",
                    "FalconConfig::validate",
                ));
            }
        }

        if self.vocab_size == 0 {
            return Err(trustformers_core::errors::TrustformersError::config_error(
                "vocab_size must be greater than 0",
                "FalconConfig::validate",
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Falcon"
    }
}

impl FalconConfig {
    /// Falcon-7B configuration
    /// Compact model suitable for many applications
    pub fn falcon_7b() -> Self {
        Self {
            vocab_size: 65024,
            hidden_size: 4544,
            num_hidden_layers: 32,
            num_attention_heads: 71,
            num_kv_heads: Some(1), // Multi-query attention
            max_position_embeddings: 2048,
            model_type: "falcon-7b".to_string(),
            new_decoder_architecture: false,
            alibi: true, // Falcon-7B uses ALiBi
            ..Self::default()
        }
    }

    /// Falcon-7B Instruct configuration
    /// Instruction-tuned version of Falcon-7B
    pub fn falcon_7b_instruct() -> Self {
        Self {
            model_type: "falcon-7b-instruct".to_string(),
            ..Self::falcon_7b()
        }
    }

    /// Falcon-40B configuration
    /// Large model with high performance
    pub fn falcon_40b() -> Self {
        Self {
            vocab_size: 65024,
            hidden_size: 8192,
            num_hidden_layers: 60,
            num_attention_heads: 128,
            num_kv_heads: Some(8), // Multi-query with more KV heads
            max_position_embeddings: 2048,
            model_type: "falcon-40b".to_string(),
            new_decoder_architecture: false,
            alibi: true,
            ..Self::default()
        }
    }

    /// Falcon-40B Instruct configuration
    /// Instruction-tuned version of Falcon-40B
    pub fn falcon_40b_instruct() -> Self {
        Self {
            model_type: "falcon-40b-instruct".to_string(),
            ..Self::falcon_40b()
        }
    }

    /// Falcon-180B configuration
    /// Largest model in the Falcon family
    pub fn falcon_180b() -> Self {
        Self {
            vocab_size: 65024,
            hidden_size: 14848,
            num_hidden_layers: 80,
            num_attention_heads: 232,
            num_kv_heads: Some(8), // Multi-query with 8 KV heads
            max_position_embeddings: 2048,
            model_type: "falcon-180b".to_string(),
            new_decoder_architecture: true, // Uses new architecture
            alibi: false,                   // Falcon-180B uses learned positional embeddings
            ..Self::default()
        }
    }

    /// Falcon-180B Chat configuration
    /// Chat-optimized version of Falcon-180B
    pub fn falcon_180b_chat() -> Self {
        Self {
            model_type: "falcon-180b-chat".to_string(),
            ..Self::falcon_180b()
        }
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the number of key-value heads (for multi-query attention)
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the number of query groups per key-value head
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads()
    }

    /// Create configuration from pretrained model name
    pub fn from_pretrained_name(name: &str) -> Option<Self> {
        match name {
            // Falcon-7B models
            "tiiuae/falcon-7b" | "falcon-7b" => Some(Self::falcon_7b()),
            "tiiuae/falcon-7b-instruct" | "falcon-7b-instruct" => Some(Self::falcon_7b_instruct()),

            // Falcon-40B models
            "tiiuae/falcon-40b" | "falcon-40b" => Some(Self::falcon_40b()),
            "tiiuae/falcon-40b-instruct" | "falcon-40b-instruct" => {
                Some(Self::falcon_40b_instruct())
            },

            // Falcon-180B models
            "tiiuae/falcon-180b" | "falcon-180b" => Some(Self::falcon_180b()),
            "tiiuae/falcon-180b-chat" | "falcon-180b-chat" => Some(Self::falcon_180b_chat()),

            _ => None,
        }
    }

    /// Check if this is an instruct/chat model
    pub fn is_instruct_model(&self) -> bool {
        self.model_type.contains("instruct") || self.model_type.contains("chat")
    }

    /// Check if this model uses ALiBi positional encoding
    pub fn uses_alibi(&self) -> bool {
        self.alibi
    }

    /// Check if this model uses the new decoder architecture
    pub fn uses_new_architecture(&self) -> bool {
        self.new_decoder_architecture
    }

    /// Get the number of parameters (approximate)
    pub fn num_parameters(&self) -> usize {
        // Rough estimation: embedding + transformer layers + head
        let embedding_params = self.vocab_size * self.hidden_size;
        let transformer_params = self.num_hidden_layers
            * (
                // Attention
                self.hidden_size * (self.hidden_size + 2 * self.num_kv_heads() * self.head_dim()) +
            // MLP
            self.hidden_size * self.hidden_size * 2 * 8 / 3
                // Approximate for GLU
            );
        let head_params = self.hidden_size * self.vocab_size;

        embedding_params + transformer_params + head_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_falcon_config_validation() {
        let config = FalconConfig::falcon_7b();
        assert!(config.validate().is_ok());

        // Test invalid config
        let mut invalid_config = config.clone();
        invalid_config.hidden_size = 4543; // Not divisible by num_attention_heads (71)
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_falcon_config_presets() {
        // Test 7B configuration
        let falcon_7b = FalconConfig::falcon_7b();
        assert_eq!(falcon_7b.hidden_size, 4544);
        assert_eq!(falcon_7b.num_hidden_layers, 32);
        assert_eq!(falcon_7b.num_attention_heads, 71);
        assert_eq!(falcon_7b.num_kv_heads(), 1);
        assert!(falcon_7b.uses_alibi());
        assert!(!falcon_7b.uses_new_architecture());

        // Test 40B configuration
        let falcon_40b = FalconConfig::falcon_40b();
        assert_eq!(falcon_40b.hidden_size, 8192);
        assert_eq!(falcon_40b.num_hidden_layers, 60);
        assert_eq!(falcon_40b.num_attention_heads, 128);
        assert_eq!(falcon_40b.num_kv_heads(), 8);

        // Test 180B configuration
        let falcon_180b = FalconConfig::falcon_180b();
        assert_eq!(falcon_180b.hidden_size, 14848);
        assert_eq!(falcon_180b.num_hidden_layers, 80);
        assert_eq!(falcon_180b.num_attention_heads, 232);
        assert_eq!(falcon_180b.num_kv_heads(), 8);
        assert!(!falcon_180b.uses_alibi());
        assert!(falcon_180b.uses_new_architecture());
    }

    #[test]
    fn test_falcon_config_from_pretrained() {
        let config = FalconConfig::from_pretrained_name("tiiuae/falcon-7b");
        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.model_type, "falcon-7b");

        let config = FalconConfig::from_pretrained_name("tiiuae/falcon-180b-chat");
        assert!(config.is_some());
        let config = config.unwrap();
        assert!(config.is_instruct_model());

        let config = FalconConfig::from_pretrained_name("unknown-model");
        assert!(config.is_none());
    }

    #[test]
    fn test_falcon_config_helpers() {
        let config = FalconConfig::falcon_7b();
        assert_eq!(config.head_dim(), 64); // 4544 / 71
        assert_eq!(config.num_kv_heads(), 1);
        assert_eq!(config.num_query_groups(), 71); // 71 / 1

        let config_40b = FalconConfig::falcon_40b();
        assert_eq!(config_40b.head_dim(), 64); // 8192 / 128
        assert_eq!(config_40b.num_kv_heads(), 8);
        assert_eq!(config_40b.num_query_groups(), 16); // 128 / 8
    }

    #[test]
    fn test_config_trait() {
        let config = FalconConfig::falcon_7b();
        assert_eq!(config.architecture(), "Falcon");
    }

    #[test]
    fn test_parameter_estimation() {
        let config = FalconConfig::falcon_7b();
        let params = config.num_parameters();
        // Should be approximately 7B parameters
        assert!(params > 6_000_000_000 && params < 8_000_000_000);

        let config_40b = FalconConfig::falcon_40b();
        let params_40b = config_40b.num_parameters();
        // Should be approximately 40B parameters
        assert!(params_40b > 35_000_000_000 && params_40b < 45_000_000_000);
    }
}
