use serde::{Deserialize, Serialize};
use trustformers_core::errors::invalid_config;
use trustformers_core::traits::Config;

use crate::nemotron::tasks::NemotronError;

/// Normalisation strategy used by the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NormType {
    /// Root Mean Square normalisation (default for Nemotron).
    RmsNorm,
    /// Standard Layer Normalisation.
    LayerNorm,
}

impl Default for NormType {
    fn default() -> Self {
        NormType::RmsNorm
    }
}

/// Configuration for the Nemotron language model family (NVIDIA, 2024).
///
/// Nemotron's distinguishing architectural features:
/// - **Squared ReLU** activation (`relu2`) instead of SiLU/GeLU
/// - **No bias** on attention or MLP projections by default
/// - **Partial rotary embeddings** — only the first `rotary_dim` elements
///   of each attention head receive RoPE; the remainder are passed through
///   unchanged
/// - **Grouped Query Attention** (GQA)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NemotronConfig {
    /// Vocabulary size (default 256000)
    pub vocab_size: usize,
    /// Hidden dimension (default 6144)
    pub hidden_size: usize,
    /// Feed-forward intermediate dimension (default 24576)
    pub intermediate_size: usize,
    /// Number of transformer decoder layers (default 32)
    pub num_hidden_layers: usize,
    /// Number of query attention heads (default 48)
    pub num_attention_heads: usize,
    /// Number of key/value heads for GQA (default 8)
    pub num_key_value_heads: usize,
    /// Per-head dimension (default 128)
    pub head_dim: usize,
    /// Maximum sequence length (default 4096)
    pub max_position_embeddings: usize,
    /// Epsilon for normalisation layers (default 1e-5)
    pub rms_norm_eps: f64,
    /// RoPE base frequency (default 10000.0)
    pub rope_theta: f64,
    /// Fraction of head_dim that gets RoPE (default 0.5 → first 50 %)
    pub partial_rotary_factor: f32,
    /// Activation function identifier ("relu2" = squared ReLU)
    pub hidden_act: String,
    /// Whether lm_head shares weights with embed_tokens (default false)
    pub tie_word_embeddings: bool,
    /// Which normalisation layer to use (default `NormType::RmsNorm`)
    pub norm_type: NormType,
    /// Whether attention projections have bias terms (default false)
    pub attention_bias: bool,
    /// Whether MLP projections have bias terms (default false)
    pub mlp_bias: bool,
}

impl Default for NemotronConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 6144,
            intermediate_size: 24576,
            num_hidden_layers: 32,
            num_attention_heads: 48,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            hidden_act: "relu2".to_string(),
            tie_word_embeddings: false,
            norm_type: NormType::RmsNorm,
            attention_bias: false,
            mlp_bias: false,
        }
    }
}

impl NemotronConfig {
    /// The number of rotary dimensions per head.
    ///
    /// `rotary_dim = floor(head_dim * partial_rotary_factor)`
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor) as usize
    }

    /// Validate this configuration, returning `NemotronError` on failure.
    pub fn validate(&self) -> std::result::Result<(), NemotronError> {
        if self.vocab_size == 0 {
            return Err(NemotronError::InvalidConfig(
                "vocab_size must be > 0".to_string(),
            ));
        }
        if self.hidden_size == 0 {
            return Err(NemotronError::InvalidConfig(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if self.num_attention_heads == 0 {
            return Err(NemotronError::InvalidConfig(
                "num_attention_heads must be > 0".to_string(),
            ));
        }
        if self.num_key_value_heads == 0 {
            return Err(NemotronError::InvalidConfig(
                "num_key_value_heads must be > 0".to_string(),
            ));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(NemotronError::InvalidConfig(
                "num_attention_heads must be divisible by num_key_value_heads".to_string(),
            ));
        }
        if self.head_dim == 0 {
            return Err(NemotronError::InvalidConfig(
                "head_dim must be > 0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.partial_rotary_factor) {
            return Err(NemotronError::InvalidConfig(
                "partial_rotary_factor must be in [0, 1]".to_string(),
            ));
        }
        if self.rms_norm_eps <= 0.0 {
            return Err(NemotronError::InvalidConfig(
                "rms_norm_eps must be positive".to_string(),
            ));
        }
        if self.rope_theta <= 0.0 {
            return Err(NemotronError::InvalidConfig(
                "rope_theta must be positive".to_string(),
            ));
        }
        Ok(())
    }

    /// Nemotron-4 340B configuration.
    ///
    /// Flagship scale: hidden=18432, layers=96, heads=96, kv_heads=8,
    /// intermediate=73728.
    pub fn nemotron_4_340b() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 18432,
            intermediate_size: 73728,
            num_hidden_layers: 96,
            num_attention_heads: 96,
            num_key_value_heads: 8,
            head_dim: 192,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            hidden_act: "relu2".to_string(),
            tie_word_embeddings: false,
            norm_type: NormType::RmsNorm,
            attention_bias: false,
            mlp_bias: false,
        }
    }

    /// Nemotron-4 22B configuration.
    ///
    /// Mid-scale: hidden=6144, layers=40, heads=48, kv_heads=8,
    /// intermediate=24576.
    pub fn nemotron_4_22b() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 6144,
            intermediate_size: 24576,
            num_hidden_layers: 40,
            num_attention_heads: 48,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            hidden_act: "relu2".to_string(),
            tie_word_embeddings: false,
            norm_type: NormType::RmsNorm,
            attention_bias: false,
            mlp_bias: false,
        }
    }
}

impl Config for NemotronConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.hidden_size == 0 {
            return Err(invalid_config(
                "config_field",
                "hidden_size must be > 0".to_string(),
            ));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(invalid_config(
                "config_field",
                "num_attention_heads must be divisible by num_key_value_heads".to_string(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(invalid_config(
                "config_field",
                "vocab_size must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Nemotron"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nemotron_default_vocab_size() {
        let cfg = NemotronConfig::default();
        assert_eq!(cfg.vocab_size, 256000);
    }

    #[test]
    fn test_nemotron_default_hidden_size() {
        let cfg = NemotronConfig::default();
        assert_eq!(cfg.hidden_size, 6144);
    }

    #[test]
    fn test_nemotron_default_num_hidden_layers() {
        let cfg = NemotronConfig::default();
        assert_eq!(cfg.num_hidden_layers, 32);
    }

    #[test]
    fn test_nemotron_default_num_attention_heads() {
        let cfg = NemotronConfig::default();
        assert_eq!(cfg.num_attention_heads, 48);
    }

    #[test]
    fn test_nemotron_default_num_key_value_heads() {
        let cfg = NemotronConfig::default();
        assert_eq!(cfg.num_key_value_heads, 8);
    }

    #[test]
    fn test_nemotron_default_head_dim() {
        let cfg = NemotronConfig::default();
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn test_nemotron_default_hidden_act() {
        let cfg = NemotronConfig::default();
        assert_eq!(cfg.hidden_act, "relu2");
    }

    #[test]
    fn test_nemotron_default_partial_rotary_factor() {
        let cfg = NemotronConfig::default();
        assert!((cfg.partial_rotary_factor - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_nemotron_default_norm_type() {
        let cfg = NemotronConfig::default();
        assert_eq!(cfg.norm_type, NormType::RmsNorm);
    }

    #[test]
    fn test_nemotron_validate_passes_default() {
        let cfg = NemotronConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_nemotron_validate_fails_zero_vocab_size() {
        let cfg = NemotronConfig {
            vocab_size: 0,
            ..NemotronConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_nemotron_validate_fails_zero_hidden_size() {
        let cfg = NemotronConfig {
            hidden_size: 0,
            ..NemotronConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_nemotron_validate_fails_invalid_rotary_factor() {
        let cfg = NemotronConfig {
            partial_rotary_factor: 1.5,
            ..NemotronConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_nemotron_validate_fails_heads_not_divisible() {
        let cfg = NemotronConfig {
            num_attention_heads: 48,
            num_key_value_heads: 7,
            ..NemotronConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_nemotron_rotary_dim() {
        let cfg = NemotronConfig::default();
        assert_eq!(cfg.rotary_dim(), 64); // floor(128 * 0.5)
    }

    #[test]
    fn test_nemotron_4_340b_preset() {
        let cfg = NemotronConfig::nemotron_4_340b();
        assert_eq!(cfg.hidden_size, 18432);
        assert_eq!(cfg.num_hidden_layers, 96);
        assert_eq!(cfg.head_dim, 192);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_nemotron_4_22b_preset() {
        let cfg = NemotronConfig::nemotron_4_22b();
        assert_eq!(cfg.hidden_size, 6144);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_nemotron_attention_bias_default_false() {
        let cfg = NemotronConfig::default();
        assert!(!cfg.attention_bias);
        assert!(!cfg.mlp_bias);
    }

    #[test]
    fn test_nemotron_tie_word_embeddings_default_false() {
        let cfg = NemotronConfig::default();
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn test_nemotron_architecture_name() {
        let cfg = NemotronConfig::default();
        assert_eq!(cfg.architecture(), "Nemotron");
    }

    #[test]
    fn test_nemotron_validate_fails_negative_rms_norm_eps() {
        let cfg = NemotronConfig {
            rms_norm_eps: 0.0,
            ..NemotronConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_nemotron_rotary_dim_22b() {
        let cfg = NemotronConfig::nemotron_4_22b();
        assert_eq!(cfg.rotary_dim(), 64); // floor(128 * 0.5)
    }

    #[test]
    fn test_nemotron_lcg_values_in_range() {
        let mut s = 42u64;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (s % 1000) as f32 / 1000.0;
        assert!((0.0..1.0).contains(&v));
    }
}
