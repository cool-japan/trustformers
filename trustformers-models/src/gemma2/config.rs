//! # Gemma-2 Configuration
//!
//! Configuration for Google's Gemma-2 model family, which features:
//! - Alternating local (sliding window) and global attention layers
//! - Logit soft-capping on attention scores and final logits
//! - Grouped Query Attention (GQA)
//! - Post-normalization (RMSNorm after residual add)
//! - GEGLU activation in the MLP
//! - Fixed 256-dim head size across all variants

use serde::{Deserialize, Serialize};
use trustformers_core::errors::invalid_config;
use trustformers_core::traits::Config;

/// Gemma-2 model configuration.
///
/// Reference: "Gemma 2: Improving Open Language Models at a Practical Size"
/// (Google DeepMind, 2024)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gemma2Config {
    /// Vocabulary size (Gemma tokenizer: 256000)
    pub vocab_size: usize,
    /// Hidden dimension size (3584 for 9B, 2304 for 2B)
    pub hidden_size: usize,
    /// Number of transformer layers (42 for 9B, 26 for 2B)
    pub num_hidden_layers: usize,
    /// Number of query attention heads (16 for 9B, 8 for 2B)
    pub num_attention_heads: usize,
    /// Number of key/value attention heads — GQA (8 for 9B, 4 for 2B)
    pub num_key_value_heads: usize,
    /// MLP intermediate dimension
    pub intermediate_size: usize,
    /// Per-head dimension (fixed at 256 for all Gemma-2 variants)
    pub head_dim: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// RoPE base frequency
    pub rope_theta: f64,
    /// RMSNorm epsilon
    pub rms_norm_eps: f64,
    /// Sliding window size for local attention layers (even layer indices)
    pub sliding_window: usize,
    /// Soft-capping applied to attention logits before softmax: `tanh(x/cap)*cap`
    pub attention_logit_softcapping: f64,
    /// Soft-capping applied to final LM logits: `tanh(x/cap)*cap`
    pub final_logit_softcapping: f64,
    /// Query pre-attention scalar — typically `head_dim^{-0.5}`
    pub query_pre_attn_scalar: f64,
    /// Model type identifier
    pub model_type: String,
}

impl Default for Gemma2Config {
    /// Default configuration matches Gemma-2 9B.
    fn default() -> Self {
        let head_dim = 256usize;
        Self {
            vocab_size: 256000,
            hidden_size: 3584,
            num_hidden_layers: 42,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            intermediate_size: 14336,
            head_dim,
            max_position_embeddings: 8192,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            sliding_window: 4096,
            attention_logit_softcapping: 50.0,
            final_logit_softcapping: 30.0,
            query_pre_attn_scalar: 1.0 / (head_dim as f64).sqrt(),
            model_type: "gemma2".to_string(),
        }
    }
}

impl Config for Gemma2Config {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.num_attention_heads == 0 {
            return Err(invalid_config(
                "config_field",
                "num_attention_heads must be > 0".to_string(),
            ));
        }
        if self.num_key_value_heads == 0 {
            return Err(invalid_config(
                "config_field",
                "num_key_value_heads must be > 0".to_string(),
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
        if self.head_dim == 0 {
            return Err(invalid_config(
                "config_field",
                "head_dim must be > 0".to_string(),
            ));
        }
        if self.sliding_window == 0 {
            return Err(invalid_config(
                "config_field",
                "sliding_window must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Gemma-2"
    }
}

impl Gemma2Config {
    /// Gemma-2 9B configuration.
    pub fn gemma2_9b() -> Self {
        Self::default()
    }

    /// Gemma-2 2B configuration.
    ///
    /// `hidden_size=2304, num_layers=26, num_heads=8, head_dim=256, kv_heads=4`
    pub fn gemma2_2b() -> Self {
        let head_dim = 256usize;
        Self {
            vocab_size: 256000,
            hidden_size: 2304,
            num_hidden_layers: 26,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            intermediate_size: 9216,
            head_dim,
            max_position_embeddings: 8192,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            sliding_window: 4096,
            attention_logit_softcapping: 50.0,
            final_logit_softcapping: 30.0,
            query_pre_attn_scalar: 1.0 / (head_dim as f64).sqrt(),
            model_type: "gemma2-2b".to_string(),
        }
    }

    /// Return the number of query heads per KV head (GQA group size).
    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Returns `true` when the layer at `layer_idx` uses local (sliding window) attention.
    ///
    /// Gemma-2 alternates: even layers → local, odd layers → global.
    pub fn is_local_layer(layer_idx: usize) -> bool {
        layer_idx.is_multiple_of(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    #[test]
    fn test_default_is_9b() {
        let cfg = Gemma2Config::default();
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.num_hidden_layers, 42);
        assert_eq!(cfg.vocab_size, 256000);
    }

    #[test]
    fn test_9b_preset_fields() {
        let cfg = Gemma2Config::gemma2_9b();
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.sliding_window, 4096);
        assert_eq!(cfg.model_type, "gemma2");
    }

    #[test]
    fn test_2b_preset_fields() {
        let cfg = Gemma2Config::gemma2_2b();
        assert_eq!(cfg.hidden_size, 2304);
        assert_eq!(cfg.num_hidden_layers, 26);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 256);
    }

    #[test]
    fn test_kv_group_size_9b() {
        // 16 / 8 = 2
        assert_eq!(Gemma2Config::gemma2_9b().kv_group_size(), 2);
    }

    #[test]
    fn test_kv_group_size_2b() {
        // 8 / 4 = 2
        assert_eq!(Gemma2Config::gemma2_2b().kv_group_size(), 2);
    }

    #[test]
    fn test_even_layers_are_local() {
        for i in [0usize, 2, 4, 6, 8] {
            assert!(Gemma2Config::is_local_layer(i), "Layer {i} should be local");
        }
    }

    #[test]
    fn test_odd_layers_are_global() {
        for i in [1usize, 3, 5, 7, 9] {
            assert!(
                !Gemma2Config::is_local_layer(i),
                "Layer {i} should be global"
            );
        }
    }

    #[test]
    fn test_attention_softcapping_default() {
        assert!((Gemma2Config::default().attention_logit_softcapping - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_final_logit_softcapping_default() {
        assert!((Gemma2Config::default().final_logit_softcapping - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_query_pre_attn_scalar_9b() {
        let cfg = Gemma2Config::gemma2_9b();
        let expected = 1.0 / (256.0f64).sqrt();
        assert!((cfg.query_pre_attn_scalar - expected).abs() < 1e-9);
    }

    #[test]
    fn test_architecture_label() {
        assert_eq!(Gemma2Config::default().architecture(), "Gemma-2");
    }

    #[test]
    fn test_validate_9b_ok() {
        assert!(Gemma2Config::gemma2_9b().validate().is_ok());
    }

    #[test]
    fn test_validate_2b_ok() {
        assert!(Gemma2Config::gemma2_2b().validate().is_ok());
    }

    #[test]
    fn test_validate_zero_num_attention_heads() {
        let mut cfg = Gemma2Config::gemma2_2b();
        cfg.num_attention_heads = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_num_key_value_heads() {
        let mut cfg = Gemma2Config::gemma2_2b();
        cfg.num_key_value_heads = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_heads_not_divisible_by_kv_heads() {
        let mut cfg = Gemma2Config::gemma2_2b();
        cfg.num_key_value_heads = 3;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_vocab_size() {
        let mut cfg = Gemma2Config::gemma2_2b();
        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_head_dim() {
        let mut cfg = Gemma2Config::gemma2_2b();
        cfg.head_dim = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_sliding_window() {
        let mut cfg = Gemma2Config::gemma2_2b();
        cfg.sliding_window = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_clone_preserves_fields() {
        let cfg = Gemma2Config::gemma2_9b();
        let cloned = cfg.clone();
        assert_eq!(cfg.hidden_size, cloned.hidden_size);
        assert_eq!(cfg.head_dim, cloned.head_dim);
        assert_eq!(cfg.model_type, cloned.model_type);
    }

    #[test]
    fn test_lcg_varied_sliding_windows() {
        let mut s = 99u64;
        for _ in 0..4 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let win = ((s % 4096) + 128) as usize;
            let mut cfg = Gemma2Config::gemma2_2b();
            cfg.sliding_window = win;
            assert!(cfg.validate().is_ok(), "window={win} failed");
        }
    }
}
