use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// StarCoder2 model configuration.
///
/// BigCode StarCoder2 (Lozhkov et al., 2024) is a family of code-generation
/// models (3B / 7B / 15B) with the following architectural choices:
///
/// - **Multi-Query Attention** (near-MQA): `num_key_value_heads = 2` for all
///   sizes, giving extreme KV-cache compression.
/// - **SwiGLU FFN** with biases on all linear layers.
/// - **RoPE** positional embeddings (θ = 10 000).
/// - **Fill-In-the-Middle (FIM)** special tokens for in-filling tasks.
/// - Optional sliding-window attention (unused in released checkpoints).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarCoder2Config {
    /// Vocabulary size (49 152)
    pub vocab_size: usize,
    /// Hidden dimension (3072 / 4608 / 6144)
    pub hidden_size: usize,
    /// SwiGLU intermediate size (12 288 / 18 432 / 24 576)
    pub intermediate_size: usize,
    /// Number of decoder layers (30 / 32 / 40)
    pub num_hidden_layers: usize,
    /// Number of query attention heads (24 / 36 / 48)
    pub num_attention_heads: usize,
    /// Number of key-value heads — 2 for all released sizes (near-MQA)
    pub num_key_value_heads: usize,
    /// Optional sliding-window size (None for all released checkpoints)
    pub sliding_window: Option<usize>,
    /// RoPE base frequency θ
    pub rope_theta: f64,
    /// Epsilon for RMSNorm stability
    pub rms_norm_eps: f64,
    /// Maximum sequence length supported
    pub max_position_embeddings: usize,
    /// Whether to add bias terms to all linear projections.
    ///
    /// StarCoder2 differs from LLaMA in using bias in q/k/v/o projections.
    pub use_bias: bool,
}

impl Default for StarCoder2Config {
    fn default() -> Self {
        Self::starcoder2_3b()
    }
}

impl Config for StarCoder2Config {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        use trustformers_core::errors::TrustformersError;
        if self.hidden_size == 0 {
            return Err(TrustformersError::invalid_config(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(TrustformersError::invalid_config(
                "hidden_size must be divisible by num_attention_heads".to_string(),
            ));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(TrustformersError::invalid_config(
                "num_attention_heads must be divisible by num_key_value_heads".to_string(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(TrustformersError::invalid_config(
                "vocab_size must be > 0".to_string(),
            ));
        }
        if self.num_hidden_layers == 0 {
            return Err(TrustformersError::invalid_config(
                "num_hidden_layers must be > 0".to_string(),
            ));
        }
        if self.intermediate_size == 0 {
            return Err(TrustformersError::invalid_config(
                "intermediate_size must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "StarCoder2"
    }
}

impl StarCoder2Config {
    /// Computed head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// GQA expansion factor (num_attention_heads / num_key_value_heads).
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// StarCoder2-3B configuration.
    pub fn starcoder2_3b() -> Self {
        Self {
            vocab_size: 49152,
            hidden_size: 3072,
            intermediate_size: 12288,
            num_hidden_layers: 30,
            num_attention_heads: 24,
            num_key_value_heads: 2,
            sliding_window: None,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 16384,
            use_bias: true,
        }
    }

    /// StarCoder2-7B configuration.
    pub fn starcoder2_7b() -> Self {
        Self {
            vocab_size: 49152,
            hidden_size: 4608,
            intermediate_size: 18432,
            num_hidden_layers: 32,
            num_attention_heads: 36,
            num_key_value_heads: 2,
            sliding_window: None,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 16384,
            use_bias: true,
        }
    }

    /// StarCoder2-15B configuration.
    pub fn starcoder2_15b() -> Self {
        Self {
            vocab_size: 49152,
            hidden_size: 6144,
            intermediate_size: 24576,
            num_hidden_layers: 40,
            num_attention_heads: 48,
            num_key_value_heads: 2,
            sliding_window: None,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 16384,
            use_bias: true,
        }
    }

    /// Minimal configuration for fast unit tests.
    pub fn small_test() -> Self {
        Self {
            vocab_size: 512,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            sliding_window: None,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 64,
            use_bias: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    #[test]
    fn test_default_is_3b() {
        let cfg = StarCoder2Config::default();
        assert_eq!(cfg.vocab_size, 49152);
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_hidden_layers, 30);
    }

    #[test]
    fn test_3b_preset() {
        let cfg = StarCoder2Config::starcoder2_3b();
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.intermediate_size, 12288);
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.max_position_embeddings, 16384);
        assert!(cfg.use_bias);
        assert!(cfg.sliding_window.is_none());
    }

    #[test]
    fn test_7b_preset() {
        let cfg = StarCoder2Config::starcoder2_7b();
        assert_eq!(cfg.hidden_size, 4608);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 36);
        assert_eq!(cfg.num_key_value_heads, 2);
    }

    #[test]
    fn test_15b_preset() {
        let cfg = StarCoder2Config::starcoder2_15b();
        assert_eq!(cfg.hidden_size, 6144);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.num_attention_heads, 48);
    }

    #[test]
    fn test_head_dim_3b() {
        // 3072 / 24 = 128
        assert_eq!(StarCoder2Config::starcoder2_3b().head_dim(), 128);
    }

    #[test]
    fn test_head_dim_15b() {
        // 6144 / 48 = 128
        assert_eq!(StarCoder2Config::starcoder2_15b().head_dim(), 128);
    }

    #[test]
    fn test_num_query_groups_3b() {
        // 24 / 2 = 12
        assert_eq!(StarCoder2Config::starcoder2_3b().num_query_groups(), 12);
    }

    #[test]
    fn test_architecture_label() {
        assert_eq!(StarCoder2Config::default().architecture(), "StarCoder2");
    }

    #[test]
    fn test_sliding_window_none_all_presets() {
        assert!(StarCoder2Config::starcoder2_3b().sliding_window.is_none());
        assert!(StarCoder2Config::starcoder2_7b().sliding_window.is_none());
        assert!(StarCoder2Config::starcoder2_15b().sliding_window.is_none());
    }

    #[test]
    fn test_validate_3b_ok() {
        assert!(StarCoder2Config::starcoder2_3b().validate().is_ok());
    }

    #[test]
    fn test_validate_15b_ok() {
        assert!(StarCoder2Config::starcoder2_15b().validate().is_ok());
    }

    #[test]
    fn test_validate_small_test_ok() {
        assert!(StarCoder2Config::small_test().validate().is_ok());
    }

    #[test]
    fn test_validate_zero_hidden_size() {
        let mut cfg = StarCoder2Config::small_test();
        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_hidden_not_divisible_by_heads() {
        let mut cfg = StarCoder2Config::small_test();
        cfg.hidden_size = 65;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_heads_not_divisible_by_kv_heads() {
        let mut cfg = StarCoder2Config::small_test();
        cfg.num_key_value_heads = 3;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_vocab_size() {
        let mut cfg = StarCoder2Config::small_test();
        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_layers() {
        let mut cfg = StarCoder2Config::small_test();
        cfg.num_hidden_layers = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_clone_preserves_use_bias() {
        let cfg = StarCoder2Config::starcoder2_3b();
        let cloned = cfg.clone();
        assert_eq!(cfg.use_bias, cloned.use_bias);
        assert_eq!(cfg.hidden_size, cloned.hidden_size);
    }

    #[test]
    fn test_lcg_varied_layer_counts() {
        let mut s = 41u64;
        for _ in 0..4 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let layers = ((s % 8) + 2) as usize;
            let mut cfg = StarCoder2Config::small_test();
            cfg.num_hidden_layers = layers;
            assert!(cfg.validate().is_ok(), "layers={layers} failed");
        }
    }
}
