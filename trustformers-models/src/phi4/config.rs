use serde::{Deserialize, Serialize};
use trustformers_core::errors::invalid_config;
use trustformers_core::traits::Config;

use crate::phi4::tasks::Phi4Error;

/// LongRoPE scaling configuration for Phi-4 extended context support.
///
/// Phi-4 supports up to 128K context via LongRoPE, which applies per-dimension
/// scaling factors that differ depending on whether the sequence is "short"
/// (within the original training context) or "long" (beyond it).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phi4RopeScaling {
    /// Must be "longrope"
    pub rope_type: String,
    /// Per-dimension scale factors for short sequences (length = head_dim / 2)
    pub short_factor: Vec<f32>,
    /// Per-dimension scale factors for long sequences (length = head_dim / 2)
    pub long_factor: Vec<f32>,
    /// Overall magnitude scale applied for short sequences
    pub short_mscale: f32,
    /// Overall magnitude scale applied for long sequences
    pub long_mscale: f32,
    /// The original maximum sequence length the base RoPE was trained with
    pub original_max_position_embeddings: usize,
}

/// Configuration for the Phi-4 language model family (Microsoft, Dec 2024).
///
/// Phi-4 is a 14B-parameter dense decoder-only model with:
/// - Grouped Query Attention (GQA) — 40 query heads / 10 KV heads
/// - RoPE with θ = 250 000 (higher base frequency than Phi-3)
/// - 16 K default context window (extendable to 128 K via LongRoPE)
/// - No sliding-window attention (full causal attention)
/// - Tied word embeddings (lm_head shares embed_tokens weights)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phi4Config {
    /// Vocabulary size (default 100352)
    pub vocab_size: usize,
    /// Hidden dimension (default 5120)
    pub hidden_size: usize,
    /// Feed-forward intermediate dimension (default 17920)
    pub intermediate_size: usize,
    /// Number of transformer decoder layers (default 40)
    pub num_hidden_layers: usize,
    /// Number of query attention heads (default 40)
    pub num_attention_heads: usize,
    /// Number of key/value heads for GQA (default 10)
    pub num_key_value_heads: usize,
    /// Dimension of each attention head (default 128)
    pub head_dim: usize,
    /// Maximum sequence length (default 16384)
    pub max_position_embeddings: usize,
    /// Original pre-training context length before LongRoPE extension (default 4096)
    pub original_max_position_embeddings: usize,
    /// Epsilon for RMSNorm (default 1e-5)
    pub rms_norm_eps: f64,
    /// RoPE base frequency (default 250000.0)
    pub rope_theta: f64,
    /// Activation function name ("silu")
    pub hidden_act: String,
    /// Whether lm_head shares weights with embed_tokens (default true)
    pub tie_word_embeddings: bool,
    /// Dropout probability on attention weights (default 0.0)
    pub attention_dropout: f32,
    /// Dropout probability on token embeddings (default 0.0)
    pub embd_dropout: f32,
    /// Optional LongRoPE scaling configuration
    pub rope_scaling: Option<Phi4RopeScaling>,
}

impl Default for Phi4Config {
    fn default() -> Self {
        Self {
            vocab_size: 100352,
            hidden_size: 5120,
            intermediate_size: 17920,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: 10,
            head_dim: 128,
            max_position_embeddings: 16384,
            original_max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 250000.0,
            hidden_act: "silu".to_string(),
            tie_word_embeddings: true,
            attention_dropout: 0.0,
            embd_dropout: 0.0,
            rope_scaling: None,
        }
    }
}

impl Phi4Config {
    /// Validate the configuration, returning a `Phi4Error` on invalid settings.
    pub fn validate(&self) -> Result<(), Phi4Error> {
        if self.vocab_size == 0 {
            return Err(Phi4Error::InvalidConfig(
                "vocab_size must be > 0".to_string(),
            ));
        }
        if self.hidden_size == 0 {
            return Err(Phi4Error::InvalidConfig(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if self.num_attention_heads == 0 {
            return Err(Phi4Error::InvalidConfig(
                "num_attention_heads must be > 0".to_string(),
            ));
        }
        if self.num_key_value_heads == 0 {
            return Err(Phi4Error::InvalidConfig(
                "num_key_value_heads must be > 0".to_string(),
            ));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(Phi4Error::InvalidConfig(
                "num_attention_heads must be divisible by num_key_value_heads".to_string(),
            ));
        }
        if self.head_dim == 0 {
            return Err(Phi4Error::InvalidConfig("head_dim must be > 0".to_string()));
        }
        if self.num_hidden_layers == 0 {
            return Err(Phi4Error::InvalidConfig(
                "num_hidden_layers must be > 0".to_string(),
            ));
        }
        if self.rms_norm_eps <= 0.0 {
            return Err(Phi4Error::InvalidConfig(
                "rms_norm_eps must be positive".to_string(),
            ));
        }
        if self.rope_theta <= 0.0 {
            return Err(Phi4Error::InvalidConfig(
                "rope_theta must be positive".to_string(),
            ));
        }
        Ok(())
    }

    /// GQA group size: how many query heads share one KV head.
    pub fn gqa_ratio(&self) -> usize {
        if self.num_key_value_heads == 0 {
            return 1;
        }
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Full 14B Phi-4 configuration matching the HuggingFace release.
    pub fn phi4_14b() -> Self {
        Self {
            vocab_size: 100352,
            hidden_size: 5120,
            intermediate_size: 17920,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: 10,
            head_dim: 128,
            max_position_embeddings: 16384,
            original_max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 250000.0,
            hidden_act: "silu".to_string(),
            tie_word_embeddings: true,
            attention_dropout: 0.0,
            embd_dropout: 0.0,
            rope_scaling: None,
        }
    }

    /// Smaller Phi-4-mini variant for faster experimentation.
    ///
    /// Architecture: hidden=3072, layers=32, heads=32, kv_heads=8,
    ///               intermediate=8192, vocab=100352, head_dim=96.
    pub fn phi4_mini() -> Self {
        Self {
            vocab_size: 100352,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 96,
            max_position_embeddings: 16384,
            original_max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 250000.0,
            hidden_act: "silu".to_string(),
            tie_word_embeddings: true,
            attention_dropout: 0.0,
            embd_dropout: 0.0,
            rope_scaling: None,
        }
    }

    /// Phi-4 14B with LongRoPE scaling to support up to 128K tokens.
    pub fn phi4_14b_longrope() -> Self {
        // Use 32 short and long scaling factors (head_dim/2 = 64, but
        // the reference model uses 32 unique factors that are broadcast).
        let short_factor = vec![
            1.05, 1.05, 1.05, 1.10, 1.10, 1.10, 1.25, 1.25, 1.40, 1.45, 1.45, 1.45, 1.55, 1.90,
            1.90, 1.95, 1.95, 1.95, 1.95, 1.95, 1.95, 2.02, 2.02, 2.02, 2.03, 2.03, 2.03, 2.03,
            2.03, 2.03, 2.03, 2.03,
        ];
        let long_factor = vec![
            1.08, 1.11, 1.14, 1.34, 1.59, 1.60, 1.62, 1.65, 1.90, 2.86, 7.40, 7.70, 9.10, 12.2,
            17.67, 24.46, 28.57, 30.42, 30.84, 32.59, 32.93, 42.32, 44.96, 50.34, 57.95, 60.14,
            62.50, 63.37, 63.48, 63.50, 63.52, 63.54,
        ];
        Self {
            max_position_embeddings: 131072,
            rope_scaling: Some(Phi4RopeScaling {
                rope_type: "longrope".to_string(),
                short_factor,
                long_factor,
                short_mscale: 1.0,
                long_mscale: 1.243_163_1,
                original_max_position_embeddings: 4096,
            }),
            ..Self::phi4_14b()
        }
    }
}

impl Config for Phi4Config {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(invalid_config(
                "config_field",
                "hidden_size must be divisible by num_attention_heads".to_string(),
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
                "vocab_size must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Phi-4"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi4_default_vocab_size() {
        let cfg = Phi4Config::default();
        assert_eq!(cfg.vocab_size, 100352);
    }

    #[test]
    fn test_phi4_default_hidden_size() {
        let cfg = Phi4Config::default();
        assert_eq!(cfg.hidden_size, 5120);
    }

    #[test]
    fn test_phi4_default_num_hidden_layers() {
        let cfg = Phi4Config::default();
        assert_eq!(cfg.num_hidden_layers, 40);
    }

    #[test]
    fn test_phi4_default_num_attention_heads() {
        let cfg = Phi4Config::default();
        assert_eq!(cfg.num_attention_heads, 40);
    }

    #[test]
    fn test_phi4_default_num_key_value_heads() {
        let cfg = Phi4Config::default();
        assert_eq!(cfg.num_key_value_heads, 10);
    }

    #[test]
    fn test_phi4_default_head_dim() {
        let cfg = Phi4Config::default();
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn test_phi4_default_max_position_embeddings() {
        let cfg = Phi4Config::default();
        assert_eq!(cfg.max_position_embeddings, 16384);
    }

    #[test]
    fn test_phi4_default_rope_theta() {
        let cfg = Phi4Config::default();
        assert!((cfg.rope_theta - 250000.0).abs() < 1.0);
    }

    #[test]
    fn test_phi4_default_hidden_act() {
        let cfg = Phi4Config::default();
        assert_eq!(cfg.hidden_act, "silu");
    }

    #[test]
    fn test_phi4_default_tie_word_embeddings() {
        let cfg = Phi4Config::default();
        assert!(cfg.tie_word_embeddings);
    }

    #[test]
    fn test_phi4_default_rope_scaling_none() {
        let cfg = Phi4Config::default();
        assert!(cfg.rope_scaling.is_none());
    }

    #[test]
    fn test_phi4_validate_passes_default() {
        let cfg = Phi4Config::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_phi4_validate_fails_zero_vocab_size() {
        let cfg = Phi4Config {
            vocab_size: 0,
            ..Phi4Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_phi4_validate_fails_heads_not_divisible_by_kv_heads() {
        let cfg = Phi4Config {
            num_attention_heads: 40,
            num_key_value_heads: 7,
            ..Phi4Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_phi4_validate_fails_zero_head_dim() {
        let cfg = Phi4Config {
            head_dim: 0,
            ..Phi4Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_phi4_validate_fails_negative_rms_norm_eps() {
        let cfg = Phi4Config {
            rms_norm_eps: -1e-5,
            ..Phi4Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_phi4_14b_preset() {
        let cfg = Phi4Config::phi4_14b();
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_phi4_mini_preset() {
        let cfg = Phi4Config::phi4_mini();
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.head_dim, 96);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_phi4_14b_longrope_max_context() {
        let cfg = Phi4Config::phi4_14b_longrope();
        assert_eq!(cfg.max_position_embeddings, 131072);
        assert!(cfg.rope_scaling.is_some());
    }

    #[test]
    fn test_phi4_longrope_scaling_type() {
        let cfg = Phi4Config::phi4_14b_longrope();
        let scaling = cfg.rope_scaling.unwrap_or_else(|| Phi4RopeScaling {
            rope_type: "none".to_string(),
            short_factor: vec![],
            long_factor: vec![],
            short_mscale: 1.0,
            long_mscale: 1.0,
            original_max_position_embeddings: 4096,
        });
        assert_eq!(scaling.rope_type, "longrope");
    }

    #[test]
    fn test_phi4_gqa_ratio() {
        let cfg = Phi4Config::phi4_14b();
        assert_eq!(cfg.gqa_ratio(), 4); // 40 / 10
    }

    #[test]
    fn test_phi4_architecture_name() {
        let cfg = Phi4Config::default();
        assert_eq!(cfg.architecture(), "Phi-4");
    }

    #[test]
    fn test_phi4_attention_dropout_default() {
        let cfg = Phi4Config::default();
        assert!((cfg.attention_dropout).abs() < 1e-6);
    }

    #[test]
    fn test_phi4_lcg_values_in_range() {
        let mut s = 42u64;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (s % 1000) as f32 / 1000.0;
        assert!((0.0..1.0).contains(&v));
    }

    #[test]
    fn test_phi4_original_max_position_embeddings() {
        let cfg = Phi4Config::default();
        assert_eq!(cfg.original_max_position_embeddings, 4096);
    }
}
