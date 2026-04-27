use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// Qwen-2 model configuration
///
/// Alibaba Qwen-2 family (0.5B to 72B) with Grouped Query Attention, SwiGLU
/// activation, RoPE position embeddings with a high base frequency of 1,000,000
/// for extended context support, and optional sliding window attention.
///
/// Notable quirk: q_proj and k_proj carry a bias term (`qkv_bias = true`),
/// while v_proj and o_proj do not.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen2Config {
    /// Vocabulary size (151 936 for the official Qwen-2 tokeniser)
    pub vocab_size: usize,
    /// Hidden (model) dimension
    pub hidden_size: usize,
    /// Intermediate SwiGLU FFN dimension
    pub intermediate_size: usize,
    /// Number of transformer decoder layers
    pub num_hidden_layers: usize,
    /// Number of query attention heads
    pub num_attention_heads: usize,
    /// Number of key-value attention heads (GQA parameter)
    pub num_key_value_heads: usize,
    /// Maximum sequence length supported
    pub max_position_embeddings: usize,
    /// RoPE base frequency θ — 1 000 000 for Qwen-2's extra-long context
    pub rope_theta: f64,
    /// Epsilon for RMSNorm
    pub rms_norm_eps: f64,
    /// Whether to use sliding window attention in self-attention layers
    pub use_sliding_window: bool,
    /// Sliding window size; None for small models (0.5B/1.5B)
    pub sliding_window: Option<usize>,
    /// Whether q_proj and k_proj carry a bias term (Qwen-2 quirk)
    pub qkv_bias: bool,
}

impl Default for Qwen2Config {
    fn default() -> Self {
        Self::qwen2_7b()
    }
}

impl Config for Qwen2Config {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.hidden_size == 0 {
            return Err(trustformers_core::errors::TrustformersError::invalid_config(
                "hidden_size must be greater than 0".to_string(),
            ));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(trustformers_core::errors::TrustformersError::invalid_config(
                "hidden_size must be divisible by num_attention_heads".to_string(),
            ));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(trustformers_core::errors::TrustformersError::invalid_config(
                "num_attention_heads must be divisible by num_key_value_heads".to_string(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(trustformers_core::errors::TrustformersError::invalid_config(
                "vocab_size must be greater than 0".to_string(),
            ));
        }
        if self.num_hidden_layers == 0 {
            return Err(trustformers_core::errors::TrustformersError::invalid_config(
                "num_hidden_layers must be greater than 0".to_string(),
            ));
        }
        if self.intermediate_size == 0 {
            return Err(trustformers_core::errors::TrustformersError::invalid_config(
                "intermediate_size must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Qwen2"
    }
}

impl Qwen2Config {
    /// Per-head dimension: `hidden_size / num_attention_heads`
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// GQA expansion factor: how many Q heads share each KV head
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Whether this config uses Grouped Query Attention
    pub fn uses_gqa(&self) -> bool {
        self.num_key_value_heads < self.num_attention_heads
    }

    /// Qwen-2 0.5B configuration
    ///
    /// 24 layers, 14 Q heads, 2 KV heads (GQA 7×), 896 hidden.
    /// No sliding window for this small model.
    pub fn qwen2_0_5b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 896,
            intermediate_size: 4864,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            sliding_window: None,
            qkv_bias: true,
        }
    }

    /// Qwen-2 7B configuration
    ///
    /// 28 layers, 28 Q heads, 4 KV heads (GQA 7×), 3584 hidden.
    pub fn qwen2_7b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: true,
            sliding_window: Some(32768),
            qkv_bias: true,
        }
    }

    /// Qwen-2 72B configuration
    ///
    /// 80 layers, 64 Q heads, 8 KV heads (GQA 8×), 8192 hidden.
    pub fn qwen2_72b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 8192,
            intermediate_size: 29568,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: true,
            sliding_window: Some(32768),
            qkv_bias: true,
        }
    }

    /// Small configuration for unit tests — fast, minimal memory
    pub fn small_test() -> Self {
        Self {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            sliding_window: None,
            qkv_bias: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    #[test]
    fn test_default_is_7b() {
        let cfg = Qwen2Config::default();
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.num_hidden_layers, 28);
    }

    #[test]
    fn test_0_5b_preset() {
        let cfg = Qwen2Config::qwen2_0_5b();
        assert_eq!(cfg.hidden_size, 896);
        assert_eq!(cfg.num_attention_heads, 14);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert!(!cfg.use_sliding_window);
        assert!(cfg.sliding_window.is_none());
    }

    #[test]
    fn test_7b_preset() {
        let cfg = Qwen2Config::qwen2_7b();
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.num_attention_heads, 28);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert!(cfg.use_sliding_window);
        assert_eq!(cfg.sliding_window, Some(32768));
        assert!(cfg.qkv_bias);
    }

    #[test]
    fn test_72b_preset() {
        let cfg = Qwen2Config::qwen2_72b();
        assert_eq!(cfg.hidden_size, 8192);
        assert_eq!(cfg.num_hidden_layers, 80);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.num_key_value_heads, 8);
    }

    #[test]
    fn test_head_dim_7b() {
        // 3584 / 28 = 128
        assert_eq!(Qwen2Config::qwen2_7b().head_dim(), 128);
    }

    #[test]
    fn test_num_query_groups_7b() {
        // 28 / 4 = 7
        assert_eq!(Qwen2Config::qwen2_7b().num_query_groups(), 7);
    }

    #[test]
    fn test_uses_gqa_7b() {
        assert!(Qwen2Config::qwen2_7b().uses_gqa());
    }

    #[test]
    fn test_rope_theta_is_1m() {
        assert!((Qwen2Config::qwen2_7b().rope_theta - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_qkv_bias_enabled() {
        assert!(Qwen2Config::qwen2_7b().qkv_bias);
    }

    #[test]
    fn test_architecture_label() {
        assert_eq!(Qwen2Config::default().architecture(), "Qwen2");
    }

    #[test]
    fn test_validate_7b_ok() {
        assert!(Qwen2Config::qwen2_7b().validate().is_ok());
    }

    #[test]
    fn test_validate_72b_ok() {
        assert!(Qwen2Config::qwen2_72b().validate().is_ok());
    }

    #[test]
    fn test_validate_zero_hidden_size() {
        let mut cfg = Qwen2Config::small_test();
        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_heads_not_divisible_by_kv_heads() {
        let mut cfg = Qwen2Config::small_test();
        cfg.num_key_value_heads = 3;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_vocab_size() {
        let mut cfg = Qwen2Config::small_test();
        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_layers() {
        let mut cfg = Qwen2Config::small_test();
        cfg.num_hidden_layers = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_clone_preserves_sliding_window() {
        let cfg = Qwen2Config::qwen2_7b();
        let cloned = cfg.clone();
        assert_eq!(cfg.sliding_window, cloned.sliding_window);
        assert_eq!(cfg.use_sliding_window, cloned.use_sliding_window);
    }

    #[test]
    fn test_lcg_varied_vocab_sizes() {
        let mut s = 53u64;
        for _ in 0..4 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let vocab = ((s % 1024) + 256) as usize;
            let mut cfg = Qwen2Config::small_test();
            cfg.vocab_size = vocab;
            assert!(cfg.validate().is_ok(), "vocab={vocab} failed");
        }
    }
}
