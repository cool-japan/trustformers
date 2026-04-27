use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// DeepSeek-V2 model configuration.
///
/// DeepSeek-V2 (DeepSeek-AI, 2024) introduces two key innovations:
///
/// - **Multi-Head Latent Attention (MLA)**: KV tensors are first compressed
///   into a low-rank latent space (`kv_lora_rank`) then decompressed per head,
///   dramatically reducing the KV cache memory footprint.
/// - **DeepSeekMoE**: Fine-grained Mixture-of-Experts with shared always-active
///   experts plus top-k routed experts.  The routing uses a softmax gate with
///   top-k selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekConfig {
    /// Vocabulary size (102 400)
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate size for non-MoE (dense) FFN layers
    pub intermediate_size: usize,
    /// Number of decoder layers
    pub num_hidden_layers: usize,
    /// Number of query attention heads
    pub num_attention_heads: usize,
    /// Number of key-value attention heads (standard GQA axis; MLA replaces
    /// this with the lora ranks but we keep it for initialisation convenience)
    pub num_key_value_heads: usize,

    // MLA-specific fields
    /// Rank of KV compression (reduces KV cache from `nheads*head_dim` to
    /// `kv_lora_rank`)
    pub kv_lora_rank: usize,
    /// Optional rank of Q compression.  `None` means Q is projected directly.
    pub q_lora_rank: Option<usize>,
    /// Dimension of the decoupled RoPE applied to a subset of head dimensions
    pub rope_head_dim: usize,
    /// Per-head value dimension
    pub v_head_dim: usize,
    /// RoPE base frequency θ
    pub rope_theta: f64,
    /// Epsilon for RMSNorm stability
    pub rms_norm_eps: f64,

    // MoE fields
    /// Total number of routed experts
    pub n_routed_experts: usize,
    /// Number of shared (always-active) experts
    pub n_shared_experts: usize,
    /// Top-k experts selected per token
    pub num_experts_per_tok: usize,
    /// Number of leading layers that use a dense FFN instead of MoE
    pub first_k_dense_replace: usize,
    /// Stride between MoE layers (1 = every layer after the dense prefix)
    pub moe_layer_freq: usize,
}

impl Default for DeepSeekConfig {
    fn default() -> Self {
        Self::deepseek_v2_small()
    }
}

impl Config for DeepSeekConfig {
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
        if self.kv_lora_rank == 0 {
            return Err(TrustformersError::invalid_config(
                "kv_lora_rank must be > 0".to_string(),
            ));
        }
        if self.n_routed_experts == 0 {
            return Err(TrustformersError::invalid_config(
                "n_routed_experts must be > 0".to_string(),
            ));
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.n_routed_experts {
            return Err(TrustformersError::invalid_config(
                "num_experts_per_tok must be in (0, n_routed_experts]".to_string(),
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "DeepSeek-V2"
    }
}

impl DeepSeekConfig {
    /// Query head dimension (hidden_size / num_attention_heads).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Whether layer `idx` uses a MoE FFN.
    ///
    /// The first `first_k_dense_replace` layers use a dense FFN; subsequent
    /// layers follow the `moe_layer_freq` stride.
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        if layer_idx < self.first_k_dense_replace {
            return false;
        }
        let offset = layer_idx - self.first_k_dense_replace;
        offset.is_multiple_of(self.moe_layer_freq)
    }

    /// DeepSeek-V2-Small (simplified) — good for integration testing.
    pub fn deepseek_v2_small() -> Self {
        Self {
            vocab_size: 102400,
            hidden_size: 2048,
            intermediate_size: 1408,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            kv_lora_rank: 512,
            q_lora_rank: None,
            rope_head_dim: 64,
            v_head_dim: 128,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            n_routed_experts: 64,
            n_shared_experts: 2,
            num_experts_per_tok: 6,
            first_k_dense_replace: 1,
            moe_layer_freq: 1,
        }
    }

    /// Minimal configuration for fast unit tests.
    pub fn small_test() -> Self {
        Self {
            vocab_size: 1024,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            kv_lora_rank: 16,
            q_lora_rank: None,
            rope_head_dim: 8,
            v_head_dim: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            n_routed_experts: 4,
            n_shared_experts: 1,
            num_experts_per_tok: 2,
            first_k_dense_replace: 1,
            moe_layer_freq: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    #[test]
    fn test_default_vocab_size() {
        assert_eq!(DeepSeekConfig::default().vocab_size, 102400);
    }

    #[test]
    fn test_small_test_fields() {
        let cfg = DeepSeekConfig::small_test();
        assert_eq!(cfg.vocab_size, 1024);
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.n_routed_experts, 4);
        assert_eq!(cfg.n_shared_experts, 1);
        assert_eq!(cfg.num_experts_per_tok, 2);
        assert_eq!(cfg.kv_lora_rank, 16);
        assert_eq!(cfg.first_k_dense_replace, 1);
    }

    #[test]
    fn test_head_dim_small_test() {
        // 64 / 4 = 16
        assert_eq!(DeepSeekConfig::small_test().head_dim(), 16);
    }

    #[test]
    fn test_head_dim_v2_small() {
        // 2048 / 16 = 128
        assert_eq!(DeepSeekConfig::deepseek_v2_small().head_dim(), 128);
    }

    #[test]
    fn test_dense_layer_before_first_k() {
        let cfg = DeepSeekConfig::small_test();
        assert!(!cfg.is_moe_layer(0));
    }

    #[test]
    fn test_moe_layer_after_first_k() {
        let cfg = DeepSeekConfig::small_test();
        assert!(cfg.is_moe_layer(1));
        assert!(cfg.is_moe_layer(2));
    }

    #[test]
    fn test_moe_layer_freq_two() {
        let mut cfg = DeepSeekConfig::small_test();
        cfg.moe_layer_freq = 2;
        cfg.first_k_dense_replace = 0;
        assert!(cfg.is_moe_layer(0));
        assert!(!cfg.is_moe_layer(1));
    }

    #[test]
    fn test_q_lora_rank_none_by_default() {
        assert!(DeepSeekConfig::small_test().q_lora_rank.is_none());
    }

    #[test]
    fn test_rope_theta_positive() {
        assert!(DeepSeekConfig::small_test().rope_theta > 0.0);
    }

    #[test]
    fn test_architecture_label() {
        assert_eq!(DeepSeekConfig::default().architecture(), "DeepSeek-V2");
    }

    #[test]
    fn test_validate_small_test_ok() {
        assert!(DeepSeekConfig::small_test().validate().is_ok());
    }

    #[test]
    fn test_validate_v2_small_ok() {
        assert!(DeepSeekConfig::deepseek_v2_small().validate().is_ok());
    }

    #[test]
    fn test_validate_zero_hidden_size() {
        let mut cfg = DeepSeekConfig::small_test();
        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_hidden_not_divisible_by_heads() {
        let mut cfg = DeepSeekConfig::small_test();
        cfg.hidden_size = 65;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_vocab_size() {
        let mut cfg = DeepSeekConfig::small_test();
        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_kv_lora_rank() {
        let mut cfg = DeepSeekConfig::small_test();
        cfg.kv_lora_rank = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_routed_experts() {
        let mut cfg = DeepSeekConfig::small_test();
        cfg.n_routed_experts = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_experts_per_tok_exceeds_routed() {
        let mut cfg = DeepSeekConfig::small_test();
        cfg.num_experts_per_tok = cfg.n_routed_experts + 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_experts_per_tok() {
        let mut cfg = DeepSeekConfig::small_test();
        cfg.num_experts_per_tok = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_clone_preserves_fields() {
        let cfg = DeepSeekConfig::small_test();
        let cloned = cfg.clone();
        assert_eq!(cfg.vocab_size, cloned.vocab_size);
        assert_eq!(cfg.kv_lora_rank, cloned.kv_lora_rank);
        assert_eq!(cfg.n_routed_experts, cloned.n_routed_experts);
    }

    #[test]
    fn test_lcg_experts_per_tok_valid_range() {
        let mut s = 7u64;
        let n_experts = 8usize;
        for _ in 0..5 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let topk = ((s % n_experts as u64) + 1) as usize;
            let mut cfg = DeepSeekConfig::small_test();
            cfg.n_routed_experts = n_experts;
            cfg.num_experts_per_tok = topk;
            assert!(cfg.validate().is_ok(), "topk={topk} failed");
        }
    }
}
