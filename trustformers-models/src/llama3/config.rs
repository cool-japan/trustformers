use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// LLaMA-3 model configuration
///
/// Meta LLaMA-3 (Dubey et al., 2024) introduces several improvements over
/// LLaMA-2:
/// - Extended Tiktoken vocabulary (128 256 tokens)
/// - Higher RoPE base frequency (500 000) for improved long-context handling
/// - Grouped Query Attention (GQA) on all model sizes
/// - Increased default context length (8192 tokens)
/// - Instruction-tuned variants fine-tuned with RLHF and DPO
///
/// Reference: "The Llama 3 Herd of Models" (Meta AI, 2024)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLaMA3Config {
    /// Vocabulary size (128 256 for the Tiktoken-based tokeniser)
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate (SwiGLU FFN) dimension
    pub intermediate_size: usize,
    /// Number of transformer decoder layers
    pub num_hidden_layers: usize,
    /// Number of query attention heads
    pub num_attention_heads: usize,
    /// Number of key-value attention heads (GQA parameter)
    ///
    /// For the 8B model: equal to `num_attention_heads` (full MHA).
    /// For the 70B model: 8, giving 8× KV sharing.
    pub num_key_value_heads: usize,
    /// RoPE base frequency θ — 500 000 for extended context
    pub rope_theta: f64,
    /// Epsilon for RMSNorm stability
    pub rms_norm_eps: f64,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
}

impl Default for LLaMA3Config {
    fn default() -> Self {
        Self::llama3_8b()
    }
}

impl Config for LLaMA3Config {
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
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "LLaMA-3"
    }
}

impl LLaMA3Config {
    /// Head dimension: `hidden_size / num_attention_heads`
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// GQA expansion factor: `num_attention_heads / num_key_value_heads`
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Whether this config uses Grouped Query Attention
    pub fn uses_gqa(&self) -> bool {
        self.num_key_value_heads < self.num_attention_heads
    }

    /// LLaMA-3 8B configuration
    ///
    /// 32 layers, 32 Q heads, 8 KV heads (GQA 4×), 4096 hidden.
    pub fn llama3_8b() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 8192,
        }
    }

    /// LLaMA-3 70B configuration
    ///
    /// 80 layers, 64 Q heads, 8 KV heads (GQA 8×), 8192 hidden.
    pub fn llama3_70b() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 8192,
            intermediate_size: 28672,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 8192,
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
            rope_theta: 500000.0,
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
    fn test_default_is_8b() {
        let cfg = LLaMA3Config::default();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.vocab_size, 128256);
    }

    #[test]
    fn test_8b_preset_fields() {
        let cfg = LLaMA3Config::llama3_8b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 14336);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.vocab_size, 128256);
        assert_eq!(cfg.max_position_embeddings, 8192);
    }

    #[test]
    fn test_70b_preset_fields() {
        let cfg = LLaMA3Config::llama3_70b();
        assert_eq!(cfg.hidden_size, 8192);
        assert_eq!(cfg.intermediate_size, 28672);
        assert_eq!(cfg.num_hidden_layers, 80);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.num_key_value_heads, 8);
    }

    #[test]
    fn test_small_test_preset_fields() {
        let cfg = LLaMA3Config::small_test();
        assert_eq!(cfg.vocab_size, 256);
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.intermediate_size, 128);
        assert_eq!(cfg.num_hidden_layers, 2);
        assert_eq!(cfg.num_attention_heads, 4);
        assert_eq!(cfg.num_key_value_heads, 2);
    }

    #[test]
    fn test_head_dim_8b() {
        // 4096 / 32 = 128
        assert_eq!(LLaMA3Config::llama3_8b().head_dim(), 128);
    }

    #[test]
    fn test_head_dim_70b() {
        // 8192 / 64 = 128
        assert_eq!(LLaMA3Config::llama3_70b().head_dim(), 128);
    }

    #[test]
    fn test_num_query_groups_8b() {
        // 32 / 8 = 4
        assert_eq!(LLaMA3Config::llama3_8b().num_query_groups(), 4);
    }

    #[test]
    fn test_num_query_groups_70b() {
        // 64 / 8 = 8
        assert_eq!(LLaMA3Config::llama3_70b().num_query_groups(), 8);
    }

    #[test]
    fn test_uses_gqa_8b() {
        assert!(LLaMA3Config::llama3_8b().uses_gqa());
    }

    #[test]
    fn test_no_gqa_when_heads_equal() {
        let mut cfg = LLaMA3Config::small_test();
        cfg.num_key_value_heads = cfg.num_attention_heads;
        assert!(!cfg.uses_gqa());
    }

    #[test]
    fn test_rope_theta_is_500k() {
        assert!((LLaMA3Config::llama3_8b().rope_theta - 500000.0).abs() < 1.0);
    }

    #[test]
    fn test_rms_norm_eps_positive() {
        assert!(LLaMA3Config::llama3_8b().rms_norm_eps > 0.0);
    }

    #[test]
    fn test_architecture_label() {
        assert_eq!(LLaMA3Config::default().architecture(), "LLaMA-3");
    }

    #[test]
    fn test_validate_8b_ok() {
        assert!(LLaMA3Config::llama3_8b().validate().is_ok());
    }

    #[test]
    fn test_validate_70b_ok() {
        assert!(LLaMA3Config::llama3_70b().validate().is_ok());
    }

    #[test]
    fn test_validate_small_test_ok() {
        assert!(LLaMA3Config::small_test().validate().is_ok());
    }

    #[test]
    fn test_validate_zero_hidden_size() {
        let mut cfg = LLaMA3Config::small_test();
        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_hidden_not_divisible_by_heads() {
        let mut cfg = LLaMA3Config::small_test();
        cfg.hidden_size = 63;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_heads_not_divisible_by_kv_heads() {
        let mut cfg = LLaMA3Config::small_test();
        cfg.num_key_value_heads = 3;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_vocab_size() {
        let mut cfg = LLaMA3Config::small_test();
        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_hidden_layers() {
        let mut cfg = LLaMA3Config::small_test();
        cfg.num_hidden_layers = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_intermediate_size() {
        let mut cfg = LLaMA3Config::small_test();
        cfg.intermediate_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_clone_preserves_fields() {
        let cfg = LLaMA3Config::llama3_8b();
        let cloned = cfg.clone();
        assert_eq!(cfg.hidden_size, cloned.hidden_size);
        assert_eq!(cfg.vocab_size, cloned.vocab_size);
        assert_eq!(cfg.rope_theta, cloned.rope_theta);
    }

    #[test]
    fn test_lcg_random_hidden_sizes_valid() {
        let mut s = 42u64;
        for _ in 0..5 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let heads = 4usize;
            let multiplier = ((s % 8) + 1) as usize;
            let hidden_size = heads * multiplier * 4;
            let mut cfg = LLaMA3Config::small_test();
            cfg.hidden_size = hidden_size;
            cfg.num_attention_heads = heads;
            cfg.num_key_value_heads = 2;
            cfg.intermediate_size = hidden_size * 2;
            assert!(cfg.validate().is_ok(), "hidden_size={hidden_size} failed");
        }
    }
}
