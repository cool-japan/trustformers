use serde::{Deserialize, Serialize};
use trustformers_core::errors::invalid_config;
use trustformers_core::traits::Config;

/// Gemma model configuration
/// Reference: "Gemma: Open Models Based on Gemini Research and Technology" (Google, 2024)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize, // Multi-query attention
    pub head_dim: usize,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub initializer_range: f32,
    pub rms_norm_eps: f32,
    pub use_cache: bool,
    pub pad_token_id: Option<u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub rope_theta: f32,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub model_type: String,
}

impl Default for GemmaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 2048,
            intermediate_size: 16384,
            num_hidden_layers: 18,
            num_attention_heads: 8,
            num_key_value_heads: 1, // Multi-query attention
            head_dim: 256,
            hidden_act: "gelu".to_string(),
            max_position_embeddings: 8192,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
            pad_token_id: Some(0),
            bos_token_id: 2,
            eos_token_id: 1,
            rope_theta: 10000.0,
            attention_bias: false,
            attention_dropout: 0.0,
            model_type: "gemma".to_string(),
        }
    }
}

impl Config for GemmaConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.hidden_size != self.num_attention_heads * self.head_dim {
            return Err(invalid_config(
                "config_field",
                "hidden_size must equal num_attention_heads * head_dim".to_string(),
            ));
        }

        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(invalid_config(
                "config_field",
                "num_attention_heads must be divisible by num_key_value_heads".to_string(),
            ));
        }

        if self.vocab_size == 0 {
            return Err(invalid_config(
                "config_field",
                "vocab_size must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Gemma"
    }
}

impl GemmaConfig {
    /// Gemma 2B configuration
    pub fn gemma_2b() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 2048,
            intermediate_size: 16384,
            num_hidden_layers: 18,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            head_dim: 256,
            max_position_embeddings: 8192,
            ..Self::default()
        }
    }

    /// Gemma 7B configuration
    pub fn gemma_7b() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 3072,
            intermediate_size: 24576,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 16, // Full multi-head attention for 7B
            head_dim: 256,
            max_position_embeddings: 8192,
            ..Self::default()
        }
    }

    /// Gemma 2B Instruct configuration
    pub fn gemma_2b_instruct() -> Self {
        Self::gemma_2b()
    }

    /// Gemma 7B Instruct configuration
    pub fn gemma_7b_instruct() -> Self {
        Self::gemma_7b()
    }

    /// Gemma Code 2B configuration
    pub fn gemma_code_2b() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 2048,
            intermediate_size: 16384,
            num_hidden_layers: 18,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            head_dim: 256,
            max_position_embeddings: 8192,
            model_type: "gemma-code".to_string(),
            ..Self::default()
        }
    }

    /// Gemma Code 7B configuration
    pub fn gemma_code_7b() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 3072,
            intermediate_size: 24576,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            head_dim: 256,
            max_position_embeddings: 8192,
            model_type: "gemma-code".to_string(),
            ..Self::default()
        }
    }

    /// Get the number of query groups per key-value head
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Check if using multi-query attention
    pub fn uses_multi_query_attention(&self) -> bool {
        self.num_key_value_heads < self.num_attention_heads
    }

    /// Get effective head dimension
    pub fn effective_head_dim(&self) -> usize {
        self.head_dim
    }
}
