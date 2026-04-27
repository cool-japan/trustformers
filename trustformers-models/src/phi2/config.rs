use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// Microsoft Phi-2 model configuration
///
/// Phi-2 is a 2.7B parameter language model from Microsoft Research (Li et al.,
/// 2023).  Key architectural features:
/// - Parallel attention + MLP blocks (both branches operate on the same layernorm output)
/// - Rotary Position Embeddings (RoPE) on Q and K
/// - Multi-Head Attention (MHA) — no GQA
/// - GELU-activated MLP: `fc1 (hidden→4*hidden) → GELU → fc2 (4*hidden→hidden)`
///
/// Reference: "Textbooks Are All You Need II: phi-1.5 technical report"
///            (Li et al., Microsoft Research, 2023)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phi2Config {
    /// Vocabulary size (51200 for the released model)
    pub vocab_size: usize,
    /// Hidden dimension (2560 for phi-2)
    pub hidden_size: usize,
    /// Intermediate (FFN) hidden dimension — defaults to 4 × hidden_size
    pub intermediate_size: usize,
    /// Number of transformer decoder layers
    pub num_hidden_layers: usize,
    /// Number of attention heads (head_dim = hidden_size / num_attention_heads)
    pub num_attention_heads: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// RoPE base frequency θ
    pub rope_theta: f64,
    /// Layer-norm epsilon
    pub layer_norm_eps: f64,
    /// Weight-initialisation range
    pub initializer_range: f64,
}

impl Default for Phi2Config {
    fn default() -> Self {
        Self {
            vocab_size: 51200,
            hidden_size: 2560,
            intermediate_size: 10240,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            max_position_embeddings: 2048,
            rope_theta: 10000.0,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }
}

impl Config for Phi2Config {
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
        "Phi-2"
    }
}

impl Phi2Config {
    /// Head dimension: `hidden_size / num_attention_heads`
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Official Phi-2 2.7B configuration as released by Microsoft Research
    pub fn phi2_2_7b() -> Self {
        Self {
            vocab_size: 51200,
            hidden_size: 2560,
            intermediate_size: 10240,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            max_position_embeddings: 2048,
            rope_theta: 10000.0,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }

    /// Small configuration for unit tests — fast construction, minimal memory
    pub fn small_test() -> Self {
        Self {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            max_position_embeddings: 64,
            rope_theta: 10000.0,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    #[test]
    fn test_default_is_phi2_2_7b() {
        let cfg = Phi2Config::default();
        assert_eq!(cfg.vocab_size, 51200);
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_hidden_layers, 32);
    }

    #[test]
    fn test_phi2_2_7b_preset() {
        let cfg = Phi2Config::phi2_2_7b();
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.intermediate_size, 10240);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.max_position_embeddings, 2048);
    }

    #[test]
    fn test_small_test_preset() {
        let cfg = Phi2Config::small_test();
        assert_eq!(cfg.vocab_size, 256);
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.num_hidden_layers, 2);
        assert_eq!(cfg.num_attention_heads, 4);
    }

    #[test]
    fn test_head_dim_2_7b() {
        // 2560 / 32 = 80
        assert_eq!(Phi2Config::phi2_2_7b().head_dim(), 80);
    }

    #[test]
    fn test_head_dim_small() {
        // 64 / 4 = 16
        assert_eq!(Phi2Config::small_test().head_dim(), 16);
    }

    #[test]
    fn test_rope_theta_is_10000() {
        assert!((Phi2Config::default().rope_theta - 10000.0).abs() < 1e-3);
    }

    #[test]
    fn test_initializer_range_positive() {
        assert!(Phi2Config::default().initializer_range > 0.0);
    }

    #[test]
    fn test_architecture_label() {
        assert_eq!(Phi2Config::default().architecture(), "Phi-2");
    }

    #[test]
    fn test_intermediate_is_4x_hidden() {
        let cfg = Phi2Config::phi2_2_7b();
        assert_eq!(cfg.intermediate_size, 4 * cfg.hidden_size);
    }

    #[test]
    fn test_validate_default_ok() {
        assert!(Phi2Config::default().validate().is_ok());
    }

    #[test]
    fn test_validate_small_test_ok() {
        assert!(Phi2Config::small_test().validate().is_ok());
    }

    #[test]
    fn test_validate_zero_hidden_size() {
        let mut cfg = Phi2Config::small_test();
        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_hidden_not_divisible_by_heads() {
        let mut cfg = Phi2Config::small_test();
        cfg.hidden_size = 65;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_vocab_size() {
        let mut cfg = Phi2Config::small_test();
        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_layers() {
        let mut cfg = Phi2Config::small_test();
        cfg.num_hidden_layers = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_intermediate_size() {
        let mut cfg = Phi2Config::small_test();
        cfg.intermediate_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_clone_preserves_all_fields() {
        let cfg = Phi2Config::phi2_2_7b();
        let cloned = cfg.clone();
        assert_eq!(cfg.vocab_size, cloned.vocab_size);
        assert_eq!(cfg.rope_theta, cloned.rope_theta);
    }

    #[test]
    fn test_lcg_varied_position_embeddings() {
        let mut s = 31u64;
        for _ in 0..5 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let max_pos = ((s % 1024) + 64) as usize;
            let mut cfg = Phi2Config::small_test();
            cfg.max_position_embeddings = max_pos;
            assert!(cfg.validate().is_ok(), "max_pos={max_pos} failed");
        }
    }
}
