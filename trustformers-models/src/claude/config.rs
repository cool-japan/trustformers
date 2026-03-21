use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// Claude model configuration
/// Implements Anthropic's Claude architecture with Constitutional AI principles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub use_cache: bool,
    pub pad_token_id: Option<u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub attention_dropout: f32,
    pub ffn_dropout: f32,
    pub constitutional_ai: bool,
    pub harmlessness_weight: f32,
    pub helpfulness_weight: f32,
    pub honesty_weight: f32,
    pub model_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub scaling_type: String,
    pub scaling_factor: f32,
}

impl Default for ClaudeConfig {
    fn default() -> Self {
        Self {
            vocab_size: 100352,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None,
            hidden_act: "swiglu".to_string(),
            max_position_embeddings: 8192,
            initializer_range: 0.02,
            layer_norm_eps: 1e-5,
            use_cache: true,
            pad_token_id: None,
            bos_token_id: 1,
            eos_token_id: 2,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_dropout: 0.0,
            ffn_dropout: 0.0,
            constitutional_ai: true,
            harmlessness_weight: 1.0,
            helpfulness_weight: 1.0,
            honesty_weight: 1.0,
            model_type: "claude".to_string(),
        }
    }
}

impl Config for ClaudeConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(trustformers_core::errors::TrustformersError::config_error(
                "hidden_size must be divisible by num_attention_heads",
                "ClaudeConfig::validate",
            ));
        }

        if let Some(num_kv_heads) = self.num_key_value_heads {
            if self.num_attention_heads % num_kv_heads != 0 {
                return Err(trustformers_core::errors::TrustformersError::config_error(
                    "num_attention_heads must be divisible by num_key_value_heads",
                    "ClaudeConfig::validate",
                ));
            }
        }

        if self.constitutional_ai
            && (self.harmlessness_weight < 0.0
                || self.helpfulness_weight < 0.0
                || self.honesty_weight < 0.0)
        {
            return Err(trustformers_core::errors::TrustformersError::config_error(
                "Constitutional AI weights must be non-negative",
                "ClaudeConfig::validate",
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Claude"
    }
}

impl ClaudeConfig {
    /// Claude-1 configuration (approximate)
    pub fn claude_1() -> Self {
        Self {
            vocab_size: 100352,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            max_position_embeddings: 8192,
            model_type: "claude-1".to_string(),
            ..Self::default()
        }
    }

    /// Claude-2 configuration with improved capabilities
    pub fn claude_2() -> Self {
        Self {
            vocab_size: 100352,
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            max_position_embeddings: 100000, // Extended context
            model_type: "claude-2".to_string(),
            ..Self::default()
        }
    }

    /// Claude-2.1 configuration with enhanced context
    pub fn claude_2_1() -> Self {
        Self {
            vocab_size: 100352,
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            max_position_embeddings: 200000, // Very long context
            model_type: "claude-2.1".to_string(),
            ..Self::default()
        }
    }

    /// Claude-3 Haiku (smallest, fastest)
    pub fn claude_3_haiku() -> Self {
        Self {
            vocab_size: 100352,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 24,
            num_attention_heads: 24,
            num_key_value_heads: Some(8), // Grouped-query attention
            max_position_embeddings: 200000,
            model_type: "claude-3-haiku".to_string(),
            ..Self::default()
        }
    }

    /// Claude-3 Sonnet (balanced performance)
    pub fn claude_3_sonnet() -> Self {
        Self {
            vocab_size: 100352,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 200000,
            model_type: "claude-3-sonnet".to_string(),
            ..Self::default()
        }
    }

    /// Claude-3 Opus (largest, most capable)
    pub fn claude_3_opus() -> Self {
        Self {
            vocab_size: 100352,
            hidden_size: 8192,
            intermediate_size: 22016,
            num_hidden_layers: 64,
            num_attention_heads: 64,
            num_key_value_heads: Some(8),
            max_position_embeddings: 200000,
            model_type: "claude-3-opus".to_string(),
            ..Self::default()
        }
    }

    /// Claude-3.5 Sonnet (improved Sonnet)
    pub fn claude_3_5_sonnet() -> Self {
        Self {
            vocab_size: 100352,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            max_position_embeddings: 200000,
            rope_theta: 500000.0, // Enhanced RoPE
            model_type: "claude-3.5-sonnet".to_string(),
            ..Self::default()
        }
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the number of key-value heads
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the number of query groups per key-value head
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads()
    }

    /// Create configuration from pretrained model name
    pub fn from_pretrained_name(name: &str) -> Option<Self> {
        match name {
            "claude-1" => Some(Self::claude_1()),
            "claude-2" => Some(Self::claude_2()),
            "claude-2.1" => Some(Self::claude_2_1()),
            "claude-3-haiku" | "claude-3-haiku-20240307" => Some(Self::claude_3_haiku()),
            "claude-3-sonnet" | "claude-3-sonnet-20240229" => Some(Self::claude_3_sonnet()),
            "claude-3-opus" | "claude-3-opus-20240229" => Some(Self::claude_3_opus()),
            "claude-3-5-sonnet" | "claude-3-5-sonnet-20240620" => Some(Self::claude_3_5_sonnet()),
            _ => None,
        }
    }

    /// Enable Constitutional AI features
    pub fn with_constitutional_ai(&mut self, enabled: bool) -> &mut Self {
        self.constitutional_ai = enabled;
        self
    }

    /// Set Constitutional AI weights
    pub fn with_constitutional_weights(
        &mut self,
        harmlessness: f32,
        helpfulness: f32,
        honesty: f32,
    ) -> &mut Self {
        self.harmlessness_weight = harmlessness;
        self.helpfulness_weight = helpfulness;
        self.honesty_weight = honesty;
        self
    }

    /// Small configuration for testing (fast model creation)
    pub fn small_test_config() -> Self {
        Self {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            max_position_embeddings: 512,
            model_type: "claude-test".to_string(),
            ..Self::default()
        }
    }
}
