use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// Linformer model configuration
/// Reference: "Linformer: Self-Attention with Linear Complexity" (Wang et al., 2020)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinformerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub pad_token_id: u32,
    pub position_embedding_type: String,

    // Linformer-specific parameters
    pub projected_attention_size: usize, // k dimension for linear projection
    pub share_projection: bool,          // Share projection matrices across heads
    pub share_layers: bool,              // Share projection matrices across layers
    pub use_efficient_attention: bool,   // Use efficient attention computation
}

impl Default for LinformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: "absolute".to_string(),

            // Linformer defaults
            projected_attention_size: 256, // Much smaller than seq_len for efficiency
            share_projection: true,
            share_layers: false,
            use_efficient_attention: true,
        }
    }
}

impl Config for LinformerConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(trustformers_core::errors::TrustformersError::config_error(
                "hidden_size must be divisible by num_attention_heads",
                "LinformerConfig::validate",
            ));
        }

        if self.projected_attention_size > self.max_position_embeddings {
            return Err(trustformers_core::errors::TrustformersError::config_error(
                "projected_attention_size should be <= max_position_embeddings for efficiency",
                "LinformerConfig::validate",
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Linformer"
    }
}

impl LinformerConfig {
    /// Linformer-Base configuration (similar to BERT-Base but with linear attention)
    pub fn linformer_base() -> Self {
        Self::default()
    }

    /// Linformer-Large configuration
    pub fn linformer_large() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            projected_attention_size: 512, // Still much smaller than typical seq_len
            ..Self::default()
        }
    }

    /// Long-sequence Linformer (for sequences up to 8K tokens)
    pub fn linformer_long() -> Self {
        Self {
            max_position_embeddings: 8192,
            projected_attention_size: 512, // Fixed small size for O(n) complexity
            ..Self::default()
        }
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the attention efficiency ratio (compression factor)
    pub fn compression_ratio(&self) -> f32 {
        self.projected_attention_size as f32 / self.max_position_embeddings as f32
    }
}
