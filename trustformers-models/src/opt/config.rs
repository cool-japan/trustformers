//! Configuration for OPT (Open Pre-trained Transformer) language models.
//!
//! Reference: "OPT: Open Pre-trained Transformer Language Models" (Zhang et al., 2022)
//! <https://arxiv.org/abs/2205.01068>

use serde::{Deserialize, Serialize};

/// Configuration struct for Meta's OPT model family.
///
/// OPT models are decoder-only transformers using:
/// - **Learned positional embeddings** (not RoPE) with a +2 offset convention.
/// - **Pre-norm** architecture (LayerNorm before each sub-layer).
/// - **ReLU** activation in the FFN (no gating, no SwiGLU).
/// - **Bias** in all attention and FFN projections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptConfig {
    /// Vocabulary size (50272 for all OPT variants).
    pub vocab_size: usize,
    /// Dimensionality of the token embeddings and hidden states.
    pub hidden_size: usize,
    /// Number of transformer decoder layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Dimensionality of the intermediate FFN layer (typically 4× `hidden_size`).
    pub ffn_dim: usize,
    /// Maximum number of positions the model can handle.
    pub max_position_embeddings: usize,
    /// Projection dimension for the word embedding (may differ from `hidden_size`
    /// in some OPT variants).  For OPT-125M this equals `hidden_size`.
    pub word_embed_proj_dim: usize,
    /// Epsilon for layer-normalisation layers.
    pub layer_norm_eps: f64,
    /// Dropout probability (set to 0.0 for inference).
    pub dropout: f64,
    /// Whether to apply the LayerNorm *before* the sub-layer (pre-norm).
    pub do_layer_norm_before: bool,
    /// Activation function name (always `"relu"` for OPT).
    pub activation_function: String,
    /// Whether to use the KV cache during generation.
    pub use_cache: bool,
    /// BOS token id.
    pub bos_token_id: u32,
    /// EOS token id.
    pub eos_token_id: u32,
    /// Pad token id.
    pub pad_token_id: Option<u32>,
}

impl Default for OptConfig {
    fn default() -> Self {
        // OPT-125M defaults
        Self {
            vocab_size: 50272,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            ffn_dim: 3072,
            max_position_embeddings: 2048,
            word_embed_proj_dim: 768,
            layer_norm_eps: 1e-5,
            dropout: 0.0,
            do_layer_norm_before: true,
            activation_function: "relu".to_string(),
            use_cache: true,
            bos_token_id: 2,
            eos_token_id: 2,
            pad_token_id: Some(1),
        }
    }
}

impl OptConfig {
    /// OPT-125M: 125 million parameters.
    ///
    /// 12 layers, 768 hidden, 12 heads, 3072 FFN dim.
    pub fn opt_125m() -> Self {
        Self::default()
    }

    /// OPT-350M: 350 million parameters.
    ///
    /// 24 layers, 1024 hidden, 16 heads, 4096 FFN dim.
    pub fn opt_350m() -> Self {
        Self {
            vocab_size: 50272,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            ffn_dim: 4096,
            max_position_embeddings: 2048,
            word_embed_proj_dim: 512, // OPT-350M uses a smaller embed proj
            layer_norm_eps: 1e-5,
            dropout: 0.0,
            do_layer_norm_before: false, // OPT-350M uses post-norm
            activation_function: "relu".to_string(),
            use_cache: true,
            bos_token_id: 2,
            eos_token_id: 2,
            pad_token_id: Some(1),
        }
    }

    /// OPT-6.7B: 6.7 billion parameters.
    ///
    /// 32 layers, 4096 hidden, 32 heads, 16384 FFN dim.
    pub fn opt_6_7b() -> Self {
        Self {
            vocab_size: 50272,
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            ffn_dim: 16384,
            max_position_embeddings: 2048,
            word_embed_proj_dim: 4096,
            layer_norm_eps: 1e-5,
            dropout: 0.0,
            do_layer_norm_before: true,
            activation_function: "relu".to_string(),
            use_cache: true,
            bos_token_id: 2,
            eos_token_id: 2,
            pad_token_id: Some(1),
        }
    }

    /// Head dimension: `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size.checked_div(self.num_attention_heads).unwrap_or(0)
    }

    /// Validate the configuration and return a descriptive error on failure.
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".to_string());
        }
        if self.hidden_size == 0 {
            return Err("hidden_size must be > 0".to_string());
        }
        if self.num_hidden_layers == 0 {
            return Err("num_hidden_layers must be > 0".to_string());
        }
        if self.num_attention_heads == 0 {
            return Err("num_attention_heads must be > 0".to_string());
        }
        if self.ffn_dim == 0 {
            return Err("ffn_dim must be > 0".to_string());
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err("hidden_size must be divisible by num_attention_heads".to_string());
        }
        if self.word_embed_proj_dim == 0 {
            return Err("word_embed_proj_dim must be > 0".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opt_config_default() {
        let cfg = OptConfig::default();
        assert_eq!(cfg.vocab_size, 50272);
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.num_attention_heads, 12);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_opt_config_125m() {
        let cfg = OptConfig::opt_125m();
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.ffn_dim, 3072); // 4 × 768
        assert_eq!(cfg.word_embed_proj_dim, 768);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_opt_config_350m() {
        let cfg = OptConfig::opt_350m();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.ffn_dim, 4096);
        // 350M has a smaller word_embed_proj_dim
        assert_eq!(cfg.word_embed_proj_dim, 512);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_opt_config_6_7b() {
        let cfg = OptConfig::opt_6_7b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.ffn_dim, 16384);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_opt_head_dim() {
        let cfg = OptConfig::opt_125m();
        assert_eq!(cfg.head_dim(), 64); // 768 / 12

        let cfg6b = OptConfig::opt_6_7b();
        assert_eq!(cfg6b.head_dim(), 128); // 4096 / 32
    }

    #[test]
    fn test_opt_embed_proj_dim() {
        // 350M separates embed_proj from hidden_size
        let cfg = OptConfig::opt_350m();
        assert_ne!(cfg.word_embed_proj_dim, cfg.hidden_size);

        // 125M and 6.7B keep them equal
        let cfg_125m = OptConfig::opt_125m();
        assert_eq!(cfg_125m.word_embed_proj_dim, cfg_125m.hidden_size);
    }

    #[test]
    fn test_opt_config_validation_invalid() {
        let cfg = OptConfig {
            num_attention_heads: 7,
            ..OptConfig::default()
        }; // 768 not divisible by 7
        assert!(cfg.validate().is_err());

        let cfg2 = OptConfig {
            num_attention_heads: 12,
            hidden_size: 0,
            ..OptConfig::default()
        };
        assert!(cfg2.validate().is_err());
    }
}
