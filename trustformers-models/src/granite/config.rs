use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors for Granite model configuration.
#[derive(Debug, Error)]
pub enum GraniteError {
    /// Configuration validation failed.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    /// Dimension mismatch during computation.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    /// Empty input provided.
    #[error("empty input provided")]
    EmptyInput,
}

/// Configuration for IBM Granite models.
///
/// Granite is a family of enterprise-focused language models with RoPE positional
/// embeddings, optional GQA, SiLU MLP, and RMSNorm pre-normalization. The
/// architecture introduces four scaling multipliers that control embedding
/// magnitude, residual stream magnitude, attention output, and final logits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraniteConfig {
    /// Vocabulary size (default 32000).
    pub vocab_size: usize,
    /// Hidden dimension size (default 2048).
    pub hidden_size: usize,
    /// Intermediate (FFN) size (default 8192).
    pub intermediate_size: usize,
    /// Number of transformer decoder layers (default 32).
    pub num_hidden_layers: usize,
    /// Number of query attention heads (default 32).
    pub num_attention_heads: usize,
    /// Number of key/value heads for GQA (default 8).
    pub num_key_value_heads: usize,
    /// Dimension of each attention head (default 64 = hidden_size / num_attention_heads).
    pub head_dim: usize,
    /// Maximum sequence length (default 4096).
    pub max_position_embeddings: usize,
    /// Epsilon for RMSNorm (default 1e-5).
    pub rms_norm_eps: f64,
    /// RoPE base frequency (default 10000.0).
    pub rope_theta: f64,
    /// Whether to add bias to attention projection layers (default false).
    pub attention_bias: bool,
    /// Whether to add bias to MLP projection layers (default false).
    pub mlp_bias: bool,
    /// Whether to tie input/output embedding weights (default false).
    pub tie_word_embeddings: bool,
    /// Activation function name (default "silu").
    pub hidden_act: String,
    /// Dropout probability for attention weights (default 0.0).
    pub attention_dropout: f32,
    /// Standard deviation for weight initialisation (default 0.02).
    pub initializer_range: f32,
    /// Multiplier applied to token embeddings before the first layer:
    /// `embedding_multiplier * sqrt(hidden_size)` (default 12.0).
    pub embedding_multiplier: f32,
    /// Scale factor applied to final logits before computing loss / argmax
    /// (default 0.25).
    pub logits_scaling: f32,
    /// Scale factor applied to each residual connection output (default 0.25).
    pub residual_multiplier: f32,
    /// Scale factor applied to attention output projection (default 0.25).
    pub attention_multiplier: f32,
}

impl Default for GraniteConfig {
    fn default() -> Self {
        let hidden_size: usize = 2048;
        let num_attention_heads: usize = 32;
        Self {
            vocab_size: 32000,
            hidden_size,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads,
            num_key_value_heads: 8,
            head_dim: hidden_size / num_attention_heads, // 64
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_bias: false,
            mlp_bias: false,
            tie_word_embeddings: false,
            hidden_act: "silu".to_string(),
            attention_dropout: 0.0,
            initializer_range: 0.02,
            embedding_multiplier: 12.0,
            logits_scaling: 0.25,
            residual_multiplier: 0.25,
            attention_multiplier: 0.25,
        }
    }
}

impl GraniteConfig {
    /// Validate the configuration, returning a [`GraniteError`] when any
    /// constraint is violated.
    pub fn validate(&self) -> Result<(), GraniteError> {
        if self.vocab_size == 0 {
            return Err(GraniteError::InvalidConfig(
                "vocab_size must be greater than 0".to_string(),
            ));
        }
        if self.hidden_size == 0 {
            return Err(GraniteError::InvalidConfig(
                "hidden_size must be greater than 0".to_string(),
            ));
        }
        if self.num_attention_heads == 0 {
            return Err(GraniteError::InvalidConfig(
                "num_attention_heads must be greater than 0".to_string(),
            ));
        }
        if self.num_key_value_heads == 0 {
            return Err(GraniteError::InvalidConfig(
                "num_key_value_heads must be greater than 0".to_string(),
            ));
        }
        let expected_head_dim = self.hidden_size / self.num_attention_heads;
        if self.head_dim != expected_head_dim {
            return Err(GraniteError::InvalidConfig(format!(
                "head_dim ({}) must equal hidden_size ({}) / num_attention_heads ({}), \
                 i.e. {}",
                self.head_dim, self.hidden_size, self.num_attention_heads, expected_head_dim,
            )));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(GraniteError::InvalidConfig(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads,
            )));
        }
        if self.num_hidden_layers == 0 {
            return Err(GraniteError::InvalidConfig(
                "num_hidden_layers must be greater than 0".to_string(),
            ));
        }
        if self.intermediate_size == 0 {
            return Err(GraniteError::InvalidConfig(
                "intermediate_size must be greater than 0".to_string(),
            ));
        }
        if self.max_position_embeddings == 0 {
            return Err(GraniteError::InvalidConfig(
                "max_position_embeddings must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Pre-set for the Granite 3B model.
    ///
    /// hidden=2048, layers=32, heads=32, kv_heads=8, intermediate=8192.
    pub fn granite_3b() -> Self {
        let hidden_size: usize = 2048;
        let num_attention_heads: usize = 32;
        Self {
            vocab_size: 32000,
            hidden_size,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads,
            num_key_value_heads: 8,
            head_dim: hidden_size / num_attention_heads, // 64
            max_position_embeddings: 4096,
            ..Default::default()
        }
    }

    /// Pre-set for the Granite 8B model.
    ///
    /// hidden=4096, layers=32, heads=32, kv_heads=8, intermediate=14336.
    pub fn granite_8b() -> Self {
        let hidden_size: usize = 4096;
        let num_attention_heads: usize = 32;
        Self {
            vocab_size: 32000,
            hidden_size,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads,
            num_key_value_heads: 8,
            head_dim: hidden_size / num_attention_heads, // 128
            max_position_embeddings: 4096,
            ..Default::default()
        }
    }

    /// Number of query groups (num_attention_heads / num_key_value_heads).
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Whether grouped-query attention is in use.
    pub fn is_gqa(&self) -> bool {
        self.num_key_value_heads != self.num_attention_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_granite_default_vocab_size() {
        let cfg = GraniteConfig::default();
        assert_eq!(cfg.vocab_size, 32000);
    }

    #[test]
    fn test_granite_default_hidden_size() {
        let cfg = GraniteConfig::default();
        assert_eq!(cfg.hidden_size, 2048);
    }

    #[test]
    fn test_granite_default_intermediate_size() {
        let cfg = GraniteConfig::default();
        assert_eq!(cfg.intermediate_size, 8192);
    }

    #[test]
    fn test_granite_default_num_hidden_layers() {
        let cfg = GraniteConfig::default();
        assert_eq!(cfg.num_hidden_layers, 32);
    }

    #[test]
    fn test_granite_default_num_attention_heads() {
        let cfg = GraniteConfig::default();
        assert_eq!(cfg.num_attention_heads, 32);
    }

    #[test]
    fn test_granite_default_num_key_value_heads() {
        let cfg = GraniteConfig::default();
        assert_eq!(cfg.num_key_value_heads, 8);
    }

    #[test]
    fn test_granite_default_head_dim() {
        let cfg = GraniteConfig::default();
        assert_eq!(cfg.head_dim, 64); // 2048 / 32
    }

    #[test]
    fn test_granite_default_scalings() {
        let cfg = GraniteConfig::default();
        assert!((cfg.embedding_multiplier - 12.0).abs() < 1e-6);
        assert!((cfg.logits_scaling - 0.25).abs() < 1e-6);
        assert!((cfg.residual_multiplier - 0.25).abs() < 1e-6);
        assert!((cfg.attention_multiplier - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_granite_validate_passes_default() {
        let cfg = GraniteConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_granite_validate_fails_zero_vocab_size() {
        let cfg = GraniteConfig {
            vocab_size: 0,
            ..GraniteConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_granite_validate_fails_zero_hidden_size() {
        let cfg = GraniteConfig {
            hidden_size: 0,
            ..GraniteConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_granite_validate_fails_wrong_head_dim() {
        let cfg = GraniteConfig {
            head_dim: 100,
            ..GraniteConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_granite_validate_fails_heads_not_divisible_by_kv_heads() {
        let cfg = GraniteConfig {
            num_attention_heads: 32,
            num_key_value_heads: 7,
            head_dim: 64,
            ..GraniteConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_granite_3b_preset() {
        let cfg = GraniteConfig::granite_3b();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.intermediate_size, 8192);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_granite_8b_preset() {
        let cfg = GraniteConfig::granite_8b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.head_dim, 128);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_granite_num_query_groups() {
        let cfg = GraniteConfig::default();
        assert_eq!(cfg.num_query_groups(), 32 / 8);
    }

    #[test]
    fn test_granite_is_gqa_true() {
        let cfg = GraniteConfig::default();
        assert!(cfg.is_gqa());
    }

    #[test]
    fn test_granite_is_gqa_false_when_equal_heads() {
        let hidden_size: usize = 2048;
        let num_attention_heads: usize = 32;
        let cfg = GraniteConfig {
            num_key_value_heads: 32,
            head_dim: hidden_size / num_attention_heads,
            ..GraniteConfig::default()
        };
        assert!(!cfg.is_gqa());
    }

    #[test]
    fn test_granite_hidden_act_default() {
        let cfg = GraniteConfig::default();
        assert_eq!(cfg.hidden_act, "silu");
    }

    #[test]
    fn test_granite_attention_bias_default_false() {
        let cfg = GraniteConfig::default();
        assert!(!cfg.attention_bias);
        assert!(!cfg.mlp_bias);
    }

    #[test]
    fn test_granite_validate_fails_zero_num_hidden_layers() {
        let cfg = GraniteConfig {
            num_hidden_layers: 0,
            ..GraniteConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_granite_validate_fails_zero_max_position_embeddings() {
        let cfg = GraniteConfig {
            max_position_embeddings: 0,
            ..GraniteConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_granite_lcg_values_in_range() {
        let mut s = 42u64;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (s % 1000) as f32 / 1000.0;
        assert!((0.0..1.0).contains(&v));
    }
}
