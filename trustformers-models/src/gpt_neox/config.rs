use serde::{Deserialize, Serialize};
use trustformers_core::{errors::Result, traits::Config};

/// Configuration for GPT-NeoX models
///
/// GPT-NeoX uses Rotary Position Embeddings and can optionally use
/// parallel attention + MLP computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPTNeoXConfig {
    /// Vocabulary size
    pub vocab_size: usize,

    /// Hidden size (embedding dimension)
    pub hidden_size: usize,

    /// Number of hidden layers
    pub num_hidden_layers: usize,

    /// Number of attention heads
    pub num_attention_heads: usize,

    /// Intermediate size for MLP (typically 4x hidden_size)
    pub intermediate_size: usize,

    /// Maximum sequence length
    pub max_position_embeddings: usize,

    /// Layer normalization epsilon
    pub layer_norm_eps: f32,

    /// Activation function ("gelu", "relu", etc.)
    pub hidden_act: String,

    /// Rotary embedding base (theta)
    pub rotary_emb_base: f32,

    /// Percentage of hidden dimensions to apply rotary embedding to
    pub rotary_pct: f32,

    /// Use parallel residual connections (attention + MLP in parallel)
    pub use_parallel_residual: bool,

    /// Tie word embeddings with output layer
    pub tie_word_embeddings: bool,

    /// Initializer range for weights
    pub initializer_range: f32,

    /// Beginning of sequence token ID
    pub bos_token_id: Option<u32>,

    /// End of sequence token ID
    pub eos_token_id: Option<u32>,
}

impl Default for GPTNeoXConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50432,
            hidden_size: 2048,
            num_hidden_layers: 16,
            num_attention_heads: 16,
            intermediate_size: 8192,
            max_position_embeddings: 2048,
            layer_norm_eps: 1e-5,
            hidden_act: "gelu".to_string(),
            rotary_emb_base: 10000.0,
            rotary_pct: 1.0,
            use_parallel_residual: false,
            tie_word_embeddings: false,
            initializer_range: 0.02,
            bos_token_id: Some(0),
            eos_token_id: Some(2),
        }
    }
}

impl Config for GPTNeoXConfig {
    fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(trustformers_core::errors::TrustformersError::invalid_config(
                "hidden_size must be greater than 0".to_string(),
            ));
        }

        if self.num_hidden_layers == 0 {
            return Err(trustformers_core::errors::TrustformersError::invalid_config(
                "num_hidden_layers must be greater than 0".to_string(),
            ));
        }

        if self.num_attention_heads == 0 {
            return Err(trustformers_core::errors::TrustformersError::invalid_config(
                "num_attention_heads must be greater than 0".to_string(),
            ));
        }

        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(trustformers_core::errors::TrustformersError::invalid_config(
                format!(
                    "hidden_size must be divisible by num_attention_heads ({})",
                    self.num_attention_heads
                ),
            ));
        }

        if self.rotary_pct < 0.0 || self.rotary_pct > 1.0 {
            return Err(trustformers_core::errors::TrustformersError::invalid_config(
                "rotary_pct must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "gpt_neox"
    }
}

impl GPTNeoXConfig {
    /// Create configuration for rinna/japanese-gpt-neox-3.6b
    pub fn rinna_japanese_3_6b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 2816,
            num_hidden_layers: 36,
            num_attention_heads: 22,
            intermediate_size: 11264,
            max_position_embeddings: 2048,
            layer_norm_eps: 1e-5,
            hidden_act: "gelu".to_string(),
            rotary_emb_base: 10000.0,
            rotary_pct: 1.0,
            use_parallel_residual: false,
            tie_word_embeddings: false,
            initializer_range: 0.02,
            bos_token_id: Some(2),
            eos_token_id: Some(3),
        }
    }
}
