use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// Yi-1.5 model configuration.
///
/// 01.AI Yi-1.5 (Young et al., 2024) is a family of bilingual (English/Chinese)
/// language models with the following architecture:
///
/// - Based on LLaMA-2 decoder-only transformer.
/// - Extended vocabulary: 64 000 tokens (vs 32 000 for LLaMA-2).
/// - Grouped Query Attention (GQA): 4 KV heads for 6B/9B, 8 for 34B.
/// - Tied embeddings: `lm_head.weight == embed_tokens.weight`.
/// - High RoPE base (θ = 5 000 000) for long-context variants (200K tokens).
/// - SwiGLU FFN, no bias in any linear.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YiConfig {
    /// Vocabulary size (64 000)
    pub vocab_size: usize,
    /// Hidden dimension (4096 for 6B/9B, 7168 for 34B)
    pub hidden_size: usize,
    /// SwiGLU intermediate size (11008 / 14336 / 20480)
    pub intermediate_size: usize,
    /// Number of decoder layers (32 / 48 / 60)
    pub num_hidden_layers: usize,
    /// Number of query attention heads (32 / 32 / 56)
    pub num_attention_heads: usize,
    /// Number of key-value heads (4 for 6B/9B, 8 for 34B)
    pub num_key_value_heads: usize,
    /// RoPE base frequency θ (5 000 000 for long-context variants)
    pub rope_theta: f64,
    /// Epsilon for RMSNorm stability
    pub rms_norm_eps: f64,
    /// Maximum sequence length (4096 base, 200000 long-context)
    pub max_position_embeddings: usize,
    /// Whether lm_head shares weights with embed_tokens
    pub tie_word_embeddings: bool,
}

impl Default for YiConfig {
    fn default() -> Self {
        Self::yi_6b()
    }
}

impl Config for YiConfig {
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
        "Yi-1.5"
    }
}

impl YiConfig {
    /// Head dimension: `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// GQA expansion factor.
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Yi-1.5-6B configuration.
    pub fn yi_6b() -> Self {
        Self {
            vocab_size: 64000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            rope_theta: 5_000_000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 4096,
            tie_word_embeddings: true,
        }
    }

    /// Yi-1.5-9B configuration.
    pub fn yi_9b() -> Self {
        Self {
            vocab_size: 64000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 48,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            rope_theta: 5_000_000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 4096,
            tie_word_embeddings: true,
        }
    }

    /// Yi-1.5-34B configuration.
    pub fn yi_34b() -> Self {
        Self {
            vocab_size: 64000,
            hidden_size: 7168,
            intermediate_size: 20480,
            num_hidden_layers: 60,
            num_attention_heads: 56,
            num_key_value_heads: 8,
            rope_theta: 5_000_000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 4096,
            tie_word_embeddings: true,
        }
    }

    /// Yi-1.5-6B long-context variant (200K context, rope_theta = 5 000 000).
    pub fn yi_6b_200k() -> Self {
        Self {
            max_position_embeddings: 200000,
            ..Self::yi_6b()
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
            rope_theta: 5_000_000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 64,
            tie_word_embeddings: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    #[test]
    fn test_default_is_6b() {
        let cfg = YiConfig::default();
        assert_eq!(cfg.vocab_size, 64000);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
    }

    #[test]
    fn test_6b_preset() {
        let cfg = YiConfig::yi_6b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 11008);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.max_position_embeddings, 4096);
        assert!(cfg.tie_word_embeddings);
    }

    #[test]
    fn test_9b_preset() {
        let cfg = YiConfig::yi_9b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 48);
        assert_eq!(cfg.num_key_value_heads, 4);
    }

    #[test]
    fn test_34b_preset() {
        let cfg = YiConfig::yi_34b();
        assert_eq!(cfg.hidden_size, 7168);
        assert_eq!(cfg.num_hidden_layers, 60);
        assert_eq!(cfg.num_attention_heads, 56);
        assert_eq!(cfg.num_key_value_heads, 8);
    }

    #[test]
    fn test_6b_200k_long_context() {
        let cfg = YiConfig::yi_6b_200k();
        assert_eq!(cfg.max_position_embeddings, 200000);
        assert_eq!(cfg.hidden_size, 4096);
    }

    #[test]
    fn test_head_dim_6b() {
        // 4096 / 32 = 128
        assert_eq!(YiConfig::yi_6b().head_dim(), 128);
    }

    #[test]
    fn test_head_dim_34b() {
        // 7168 / 56 = 128
        assert_eq!(YiConfig::yi_34b().head_dim(), 128);
    }

    #[test]
    fn test_num_query_groups_6b() {
        // 32 / 4 = 8
        assert_eq!(YiConfig::yi_6b().num_query_groups(), 8);
    }

    #[test]
    fn test_rope_theta_is_5m() {
        assert!((YiConfig::yi_6b().rope_theta - 5_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_tie_word_embeddings_true() {
        assert!(YiConfig::yi_6b().tie_word_embeddings);
        assert!(YiConfig::yi_34b().tie_word_embeddings);
    }

    #[test]
    fn test_architecture_label() {
        assert_eq!(YiConfig::default().architecture(), "Yi-1.5");
    }

    #[test]
    fn test_validate_6b_ok() {
        assert!(YiConfig::yi_6b().validate().is_ok());
    }

    #[test]
    fn test_validate_34b_ok() {
        assert!(YiConfig::yi_34b().validate().is_ok());
    }

    #[test]
    fn test_validate_small_test_ok() {
        assert!(YiConfig::small_test().validate().is_ok());
    }

    #[test]
    fn test_validate_zero_hidden_size() {
        let mut cfg = YiConfig::small_test();
        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_hidden_not_divisible_by_heads() {
        let mut cfg = YiConfig::small_test();
        cfg.hidden_size = 65;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_heads_not_divisible_by_kv_heads() {
        let mut cfg = YiConfig::small_test();
        cfg.num_key_value_heads = 3;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_vocab_size() {
        let mut cfg = YiConfig::small_test();
        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_layers() {
        let mut cfg = YiConfig::small_test();
        cfg.num_hidden_layers = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_clone_preserves_tie_word_embeddings() {
        let cfg = YiConfig::yi_6b();
        let cloned = cfg.clone();
        assert_eq!(cfg.tie_word_embeddings, cloned.tie_word_embeddings);
        assert_eq!(cfg.vocab_size, cloned.vocab_size);
    }

    #[test]
    fn test_lcg_varied_context_lengths() {
        let mut s = 37u64;
        for _ in 0..4 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let ctx = ((s % 8192) + 64) as usize;
            let mut cfg = YiConfig::small_test();
            cfg.max_position_embeddings = ctx;
            assert!(cfg.validate().is_ok(), "ctx={ctx} failed");
        }
    }
}
