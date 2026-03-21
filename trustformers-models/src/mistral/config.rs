use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// Mistral model configuration
/// Reference: "Mistral 7B" technical report (Jiang et al., 2023)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize, // Grouped-query attention
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub initializer_range: f32,
    pub rms_norm_eps: f32,
    pub use_cache: bool,
    pub pad_token_id: Option<u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub rope_theta: f32,
    pub sliding_window: Option<usize>, // For sliding window attention
    pub attention_dropout: f32,
    pub model_type: String,
}

impl Default for MistralConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8, // Grouped-query attention
            hidden_act: "silu".to_string(),
            max_position_embeddings: 32768, // Long context support
            initializer_range: 0.02,
            rms_norm_eps: 1e-5,
            use_cache: true,
            pad_token_id: None,
            bos_token_id: 1,
            eos_token_id: 2,
            rope_theta: 10000.0,
            sliding_window: Some(4096), // Sliding window attention
            attention_dropout: 0.0,
            model_type: "mistral".to_string(),
        }
    }
}

impl Config for MistralConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(trustformers_core::errors::invalid_config(
                "hidden_size",
                "hidden_size must be divisible by num_attention_heads",
            ));
        }

        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(trustformers_core::errors::invalid_config(
                "num_attention_heads",
                "num_attention_heads must be divisible by num_key_value_heads",
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Mistral"
    }
}

impl MistralConfig {
    /// Mistral 7B configuration
    pub fn mistral_7b() -> Self {
        Self::default()
    }

    /// Mistral 7B Instruct configuration
    pub fn mistral_7b_instruct() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            max_position_embeddings: 32768,
            sliding_window: Some(4096),
            ..Self::default()
        }
    }

    /// Mistral 8x7B (Mixtral) base configuration
    /// Note: This is a simplified config for the mixture of experts model
    pub fn mixtral_8x7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            max_position_embeddings: 32768,
            sliding_window: None, // Mixtral doesn't use sliding window
            model_type: "mixtral".to_string(),
            ..Self::default()
        }
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the number of query groups per key-value head
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Check if using sliding window attention
    pub fn uses_sliding_window(&self) -> bool {
        self.sliding_window.is_some()
    }

    /// Get the sliding window size
    pub fn sliding_window_size(&self) -> usize {
        self.sliding_window.unwrap_or(self.max_position_embeddings)
    }
}
