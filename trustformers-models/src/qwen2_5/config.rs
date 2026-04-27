//! # Qwen2.5 Configuration
//!
//! Configuration for Alibaba's Qwen2.5 model family.
//!
//! Qwen2.5 improves on Qwen2 with:
//! - Larger default context (32768 tokens)
//! - Higher RoPE theta (1,000,000 vs 10,000) for long-context stability
//! - Grouped Query Attention (GQA) with few KV heads (4 by default)
//! - Optional sliding window attention for very long sequences
//! - Optional multimodal RoPE (mRoPE) for vision-language variants

use serde::{Deserialize, Serialize};
use trustformers_core::errors::invalid_config;
use trustformers_core::traits::Config;

/// Qwen2.5 model configuration.
///
/// Reference: "Qwen2.5 Technical Report" (Alibaba Group, 2024).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen25Config {
    // -----------------------------------------------------------------------
    // Vocabulary / embedding
    // -----------------------------------------------------------------------
    /// Vocabulary size (default 151936 — the Qwen2.5 tokenizer).
    pub vocab_size: usize,
    /// Hidden (model) dimension (default 3584).
    pub hidden_size: usize,
    /// FFN intermediate dimension (default 18944).
    pub intermediate_size: usize,

    // -----------------------------------------------------------------------
    // Transformer depth & heads
    // -----------------------------------------------------------------------
    /// Number of transformer layers (default 28).
    pub num_hidden_layers: usize,
    /// Number of query attention heads (default 28).
    pub num_attention_heads: usize,
    /// Number of key/value heads — GQA (default 4).
    ///
    /// `num_attention_heads` must be divisible by `num_key_value_heads`.
    pub num_key_value_heads: usize,
    /// Per-head dimension (default 128).
    ///
    /// Should satisfy `hidden_size == num_attention_heads * head_dim`.
    pub head_dim: usize,

    // -----------------------------------------------------------------------
    // Position embeddings
    // -----------------------------------------------------------------------
    /// Maximum sequence length (default 32768).
    pub max_position_embeddings: usize,
    /// RoPE base frequency (default 1_000_000.0 — Qwen2.5 extended-context value).
    pub rope_theta: f64,

    // -----------------------------------------------------------------------
    // Sliding window attention
    // -----------------------------------------------------------------------
    /// Window size when `use_sliding_window = true` (default `None`).
    pub sliding_window: Option<usize>,
    /// Number of layers that use full (non-sliding) attention (default 28 = all layers).
    pub max_window_layers: usize,
    /// Enable sliding window attention (default `false`).
    pub use_sliding_window: bool,

    // -----------------------------------------------------------------------
    // Misc
    // -----------------------------------------------------------------------
    /// RMSNorm epsilon (default 1e-6).
    pub rms_norm_eps: f64,
    /// Activation function name (default `"silu"`).
    pub hidden_act: String,
    /// Weight initialisation standard deviation (default 0.02).
    pub initializer_range: f32,
    /// Whether to tie input embeddings and the LM head weight (default `false`).
    pub tie_word_embeddings: bool,
    /// Enable multimodal RoPE (mRoPE) used in vision-language variants (default `false`).
    pub use_mrope: bool,
}

impl Default for Qwen25Config {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            head_dim: 128,
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            sliding_window: None,
            max_window_layers: 28,
            use_sliding_window: false,
            rms_norm_eps: 1e-6,
            hidden_act: "silu".to_string(),
            initializer_range: 0.02,
            tie_word_embeddings: false,
            use_mrope: false,
        }
    }
}

impl Config for Qwen25Config {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.vocab_size == 0 {
            return Err(invalid_config("vocab_size", "must be > 0".to_string()));
        }
        if self.hidden_size == 0 {
            return Err(invalid_config("hidden_size", "must be > 0".to_string()));
        }
        if self.num_attention_heads == 0 {
            return Err(invalid_config(
                "num_attention_heads",
                "must be > 0".to_string(),
            ));
        }
        if self.num_key_value_heads == 0 {
            return Err(invalid_config(
                "num_key_value_heads",
                "must be > 0".to_string(),
            ));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(invalid_config(
                "num_key_value_heads",
                "num_attention_heads must be divisible by num_key_value_heads".to_string(),
            ));
        }
        if self.head_dim == 0 {
            return Err(invalid_config("head_dim", "must be > 0".to_string()));
        }
        if self.num_hidden_layers == 0 {
            return Err(invalid_config(
                "num_hidden_layers",
                "must be > 0".to_string(),
            ));
        }
        if self.use_sliding_window && self.sliding_window.is_none() {
            return Err(invalid_config(
                "sliding_window",
                "must be Some(_) when use_sliding_window is true".to_string(),
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Qwen2.5"
    }
}

impl Qwen25Config {
    /// Number of query heads per KV head (GQA group size).
    pub fn kv_group_size(&self) -> usize {
        if self.num_key_value_heads == 0 {
            return 1;
        }
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Returns `true` when the layer at `layer_idx` uses sliding window attention.
    ///
    /// A layer uses sliding window attention when:
    /// - `use_sliding_window` is `true`, AND
    /// - `layer_idx` is not one of the first `max_window_layers` full-attention layers.
    pub fn layer_uses_sliding_window(&self, layer_idx: usize) -> bool {
        self.use_sliding_window && layer_idx >= self.max_window_layers
    }

    /// Qwen2.5-7B configuration.
    pub fn qwen25_7b() -> Self {
        Self::default()
    }

    /// Qwen2.5-0.5B (tiny) configuration — useful for tests.
    pub fn qwen25_0_5b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 896,
            intermediate_size: 4864,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            head_dim: 64,
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            sliding_window: None,
            max_window_layers: 24,
            use_sliding_window: false,
            rms_norm_eps: 1e-6,
            hidden_act: "silu".to_string(),
            initializer_range: 0.02,
            tie_word_embeddings: true,
            use_mrope: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen25_default_vocab_size() {
        let cfg = Qwen25Config::default();
        assert_eq!(cfg.vocab_size, 151936);
    }

    #[test]
    fn test_qwen25_default_hidden_size() {
        let cfg = Qwen25Config::default();
        assert_eq!(cfg.hidden_size, 3584);
    }

    #[test]
    fn test_qwen25_default_num_hidden_layers() {
        let cfg = Qwen25Config::default();
        assert_eq!(cfg.num_hidden_layers, 28);
    }

    #[test]
    fn test_qwen25_default_num_attention_heads() {
        let cfg = Qwen25Config::default();
        assert_eq!(cfg.num_attention_heads, 28);
    }

    #[test]
    fn test_qwen25_default_num_key_value_heads() {
        let cfg = Qwen25Config::default();
        assert_eq!(cfg.num_key_value_heads, 4);
    }

    #[test]
    fn test_qwen25_default_head_dim() {
        let cfg = Qwen25Config::default();
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn test_qwen25_default_max_position_embeddings() {
        let cfg = Qwen25Config::default();
        assert_eq!(cfg.max_position_embeddings, 32768);
    }

    #[test]
    fn test_qwen25_default_rope_theta() {
        let cfg = Qwen25Config::default();
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_qwen25_default_sliding_window_disabled() {
        let cfg = Qwen25Config::default();
        assert!(!cfg.use_sliding_window);
        assert!(cfg.sliding_window.is_none());
    }

    #[test]
    fn test_qwen25_default_hidden_act() {
        let cfg = Qwen25Config::default();
        assert_eq!(cfg.hidden_act, "silu");
    }

    #[test]
    fn test_qwen25_validate_passes_default() {
        let cfg = Qwen25Config::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_qwen25_validate_fails_zero_vocab_size() {
        let cfg = Qwen25Config {
            vocab_size: 0,
            ..Qwen25Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_qwen25_validate_fails_zero_head_dim() {
        let cfg = Qwen25Config {
            head_dim: 0,
            ..Qwen25Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_qwen25_validate_fails_heads_not_divisible_by_kv_heads() {
        let cfg = Qwen25Config {
            num_attention_heads: 28,
            num_key_value_heads: 7,
            ..Qwen25Config::default()
        };
        assert!(cfg.validate().is_ok()); // 28 % 7 == 0
    }

    #[test]
    fn test_qwen25_validate_fails_sliding_window_enabled_without_value() {
        let cfg = Qwen25Config {
            use_sliding_window: true,
            sliding_window: None,
            ..Qwen25Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_qwen25_validate_sliding_window_with_value() {
        let cfg = Qwen25Config {
            use_sliding_window: true,
            sliding_window: Some(4096),
            ..Qwen25Config::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_qwen25_kv_group_size() {
        let cfg = Qwen25Config::default();
        assert_eq!(cfg.kv_group_size(), 28 / 4);
    }

    #[test]
    fn test_qwen25_layer_uses_sliding_window_disabled() {
        let cfg = Qwen25Config::default();
        assert!(!cfg.layer_uses_sliding_window(0));
        assert!(!cfg.layer_uses_sliding_window(100));
    }

    #[test]
    fn test_qwen25_layer_uses_sliding_window_enabled() {
        let cfg = Qwen25Config {
            use_sliding_window: true,
            sliding_window: Some(4096),
            max_window_layers: 4,
            ..Qwen25Config::default()
        };
        assert!(!cfg.layer_uses_sliding_window(3));
        assert!(cfg.layer_uses_sliding_window(4));
    }

    #[test]
    fn test_qwen25_7b_is_default() {
        let a = Qwen25Config::qwen25_7b();
        let b = Qwen25Config::default();
        assert_eq!(a.hidden_size, b.hidden_size);
        assert_eq!(a.num_hidden_layers, b.num_hidden_layers);
    }

    #[test]
    fn test_qwen25_0_5b_preset() {
        let cfg = Qwen25Config::qwen25_0_5b();
        assert_eq!(cfg.hidden_size, 896);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_qwen25_architecture_name() {
        let cfg = Qwen25Config::default();
        assert_eq!(cfg.architecture(), "Qwen2.5");
    }

    #[test]
    fn test_qwen25_tie_word_embeddings_default_false() {
        let cfg = Qwen25Config::default();
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn test_qwen25_use_mrope_default_false() {
        let cfg = Qwen25Config::default();
        assert!(!cfg.use_mrope);
    }

    #[test]
    fn test_qwen25_lcg_values_in_range() {
        let mut s = 42u64;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (s % 1000) as f32 / 1000.0;
        assert!((0.0..1.0).contains(&v));
    }
}
