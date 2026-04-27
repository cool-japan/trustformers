use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// Mistral 7B v0.3 model configuration
///
/// Mistral v0.3 is built on the same architecture as Mistral 7B v0.1
/// (GQA + sliding window attention + SwiGLU) but expands the vocabulary
/// to 32 768 tokens and adds function / tool calling capability via the
/// `[TOOL_CALLS]` special token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralV3Config {
    /// Vocabulary size — 32 768 (expanded from v0.1's 32 000)
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate SwiGLU FFN dimension
    pub intermediate_size: usize,
    /// Number of transformer decoder layers
    pub num_hidden_layers: usize,
    /// Number of query attention heads
    pub num_attention_heads: usize,
    /// Number of key-value attention heads (GQA parameter)
    pub num_key_value_heads: usize,
    /// Sliding window size for local attention
    pub sliding_window: usize,
    /// RoPE base frequency θ
    pub rope_theta: f64,
    /// Epsilon for RMSNorm
    pub rms_norm_eps: f64,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
}

impl Default for MistralV3Config {
    fn default() -> Self {
        Self::mistral_7b_v0_3()
    }
}

impl Config for MistralV3Config {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.hidden_size == 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "hidden_size must be greater than 0".to_string(),
                ),
            );
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "hidden_size must be divisible by num_attention_heads".to_string(),
                ),
            );
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "num_attention_heads must be divisible by num_key_value_heads".to_string(),
                ),
            );
        }
        if self.vocab_size == 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "vocab_size must be greater than 0".to_string(),
                ),
            );
        }
        if self.num_hidden_layers == 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "num_hidden_layers must be greater than 0".to_string(),
                ),
            );
        }
        if self.intermediate_size == 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "intermediate_size must be greater than 0".to_string(),
                ),
            );
        }
        if self.sliding_window == 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "sliding_window must be greater than 0".to_string(),
                ),
            );
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "MistralV3"
    }
}

impl MistralV3Config {
    /// Per-head dimension: `hidden_size / num_attention_heads`
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// GQA expansion factor
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Mistral 7B v0.3 production configuration
    pub fn mistral_7b_v0_3() -> Self {
        Self {
            vocab_size: 32768,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            sliding_window: 4096,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 32768,
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
            sliding_window: 8,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    #[test]
    fn test_default_is_mistral_7b_v0_3() {
        let cfg = MistralV3Config::default();
        assert_eq!(cfg.vocab_size, 32768);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
    }

    #[test]
    fn test_7b_preset_fields() {
        let cfg = MistralV3Config::mistral_7b_v0_3();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 14336);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.sliding_window, 4096);
        assert_eq!(cfg.max_position_embeddings, 32768);
    }

    #[test]
    fn test_small_test_fields() {
        let cfg = MistralV3Config::small_test();
        assert_eq!(cfg.vocab_size, 256);
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.sliding_window, 8);
    }

    #[test]
    fn test_head_dim_7b() {
        // 4096 / 32 = 128
        assert_eq!(MistralV3Config::mistral_7b_v0_3().head_dim(), 128);
    }

    #[test]
    fn test_head_dim_small() {
        // 64 / 4 = 16
        assert_eq!(MistralV3Config::small_test().head_dim(), 16);
    }

    #[test]
    fn test_num_query_groups_7b() {
        // 32 / 8 = 4
        assert_eq!(MistralV3Config::mistral_7b_v0_3().num_query_groups(), 4);
    }

    #[test]
    fn test_rope_theta_is_1m() {
        assert!((MistralV3Config::mistral_7b_v0_3().rope_theta - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_architecture_label() {
        assert_eq!(MistralV3Config::default().architecture(), "MistralV3");
    }

    #[test]
    fn test_validate_7b_ok() {
        assert!(MistralV3Config::mistral_7b_v0_3().validate().is_ok());
    }

    #[test]
    fn test_validate_small_test_ok() {
        assert!(MistralV3Config::small_test().validate().is_ok());
    }

    #[test]
    fn test_validate_zero_hidden_size() {
        let mut cfg = MistralV3Config::small_test();
        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_hidden_not_divisible_by_heads() {
        let mut cfg = MistralV3Config::small_test();
        cfg.hidden_size = 63;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_heads_not_divisible_by_kv_heads() {
        let mut cfg = MistralV3Config::small_test();
        cfg.num_key_value_heads = 3;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_vocab_size() {
        let mut cfg = MistralV3Config::small_test();
        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_layers() {
        let mut cfg = MistralV3Config::small_test();
        cfg.num_hidden_layers = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_intermediate_size() {
        let mut cfg = MistralV3Config::small_test();
        cfg.intermediate_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_sliding_window() {
        let mut cfg = MistralV3Config::small_test();
        cfg.sliding_window = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_clone_preserves_fields() {
        let cfg = MistralV3Config::mistral_7b_v0_3();
        let cloned = cfg.clone();
        assert_eq!(cfg.vocab_size, cloned.vocab_size);
        assert_eq!(cfg.sliding_window, cloned.sliding_window);
        assert_eq!(cfg.rope_theta, cloned.rope_theta);
    }

    #[test]
    fn test_lcg_varied_sliding_windows() {
        let mut s = 13u64;
        for _ in 0..4 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let window = ((s % 512) + 8) as usize;
            let mut cfg = MistralV3Config::small_test();
            cfg.sliding_window = window;
            assert!(cfg.validate().is_ok(), "window={window} failed");
        }
    }
}
