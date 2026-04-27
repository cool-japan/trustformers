use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// Mixtral model configuration
/// Reference: "Mixtral of Experts" (Jiang et al., 2024)
/// Mixtral is a Sparse Mixture of Experts variant of Mistral.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixtralConfig {
    pub hidden_size: usize,             // 4096
    pub intermediate_size: usize,       // 14336
    pub num_hidden_layers: usize,       // 32
    pub num_attention_heads: usize,     // 32
    pub num_key_value_heads: usize,     // 8 (GQA)
    pub num_local_experts: usize,       // 8
    pub num_experts_per_tok: usize,     // 2 (top-k routing)
    pub sliding_window: Option<usize>,  // None for Mixtral (full attention)
    pub vocab_size: usize,              // 32000
    pub max_position_embeddings: usize, // 32768
    pub rope_theta: f32,                // 1000000.0
    pub rms_norm_eps: f64,              // 1e-5
    pub hidden_act: String,             // "silu"
    pub router_aux_loss_coef: f32,      // 0.02 (load balancing loss coefficient)
    pub model_type: String,
}

impl Default for MixtralConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            num_local_experts: 8,
            num_experts_per_tok: 2,
            sliding_window: None,
            vocab_size: 32000,
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-5,
            hidden_act: "silu".to_string(),
            router_aux_loss_coef: 0.02,
            model_type: "mixtral".to_string(),
        }
    }
}

impl Config for MixtralConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(trustformers_core::errors::invalid_config(
                "hidden_size",
                "hidden_size must be divisible by num_attention_heads",
            ));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(trustformers_core::errors::invalid_config(
                "num_attention_heads",
                "num_attention_heads must be divisible by num_key_value_heads",
            ));
        }
        if self.num_local_experts == 0 {
            return Err(trustformers_core::errors::invalid_config(
                "num_local_experts",
                "num_local_experts must be greater than 0",
            ));
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_local_experts {
            return Err(trustformers_core::errors::invalid_config(
                "num_experts_per_tok",
                "num_experts_per_tok must be in [1, num_local_experts]",
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Mixtral"
    }
}

impl MixtralConfig {
    /// Mixtral 8x7B configuration
    pub fn mixtral_8x7b() -> Self {
        Self::default()
    }

    /// Mixtral 8x22B configuration
    pub fn mixtral_8x22b() -> Self {
        Self {
            hidden_size: 6144,
            intermediate_size: 16384,
            num_hidden_layers: 56,
            num_attention_heads: 48,
            num_key_value_heads: 8,
            num_local_experts: 8,
            num_experts_per_tok: 2,
            vocab_size: 32768,
            max_position_embeddings: 65536,
            ..Self::default()
        }
    }

    /// Head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Number of query groups per KV head
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    #[test]
    fn test_default_is_8x7b() {
        let cfg = MixtralConfig::default();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_local_experts, 8);
        assert_eq!(cfg.num_experts_per_tok, 2);
    }

    #[test]
    fn test_8x7b_preset() {
        let cfg = MixtralConfig::mixtral_8x7b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 14336);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.vocab_size, 32000);
        assert!(cfg.sliding_window.is_none());
    }

    #[test]
    fn test_8x22b_preset() {
        let cfg = MixtralConfig::mixtral_8x22b();
        assert_eq!(cfg.hidden_size, 6144);
        assert_eq!(cfg.num_hidden_layers, 56);
        assert_eq!(cfg.num_attention_heads, 48);
        assert_eq!(cfg.vocab_size, 32768);
        assert_eq!(cfg.max_position_embeddings, 65536);
    }

    #[test]
    fn test_head_dim_8x7b() {
        // 4096 / 32 = 128
        assert_eq!(MixtralConfig::mixtral_8x7b().head_dim(), 128);
    }

    #[test]
    fn test_head_dim_8x22b() {
        // 6144 / 48 = 128
        assert_eq!(MixtralConfig::mixtral_8x22b().head_dim(), 128);
    }

    #[test]
    fn test_num_query_groups_8x7b() {
        // 32 / 8 = 4
        assert_eq!(MixtralConfig::mixtral_8x7b().num_query_groups(), 4);
    }

    #[test]
    fn test_model_type_is_mixtral() {
        assert_eq!(MixtralConfig::default().model_type, "mixtral");
    }

    #[test]
    fn test_architecture_label() {
        assert_eq!(MixtralConfig::default().architecture(), "Mixtral");
    }

    #[test]
    fn test_validate_8x7b_ok() {
        assert!(MixtralConfig::mixtral_8x7b().validate().is_ok());
    }

    #[test]
    fn test_validate_8x22b_ok() {
        assert!(MixtralConfig::mixtral_8x22b().validate().is_ok());
    }

    #[test]
    fn test_validate_hidden_not_divisible_by_heads() {
        let cfg = MixtralConfig {
            hidden_size: 4097,
            ..MixtralConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_heads_not_divisible_by_kv_heads() {
        let cfg = MixtralConfig {
            num_key_value_heads: 7,
            ..MixtralConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_local_experts() {
        let cfg = MixtralConfig {
            num_local_experts: 0,
            ..MixtralConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_experts_per_tok_exceeds_local() {
        let default = MixtralConfig::default();
        let cfg = MixtralConfig {
            num_experts_per_tok: default.num_local_experts + 1,
            ..default
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_experts_per_tok() {
        let cfg = MixtralConfig {
            num_experts_per_tok: 0,
            ..MixtralConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_sliding_window_none_by_default() {
        assert!(MixtralConfig::default().sliding_window.is_none());
    }

    #[test]
    fn test_clone_preserves_fields() {
        let cfg = MixtralConfig::mixtral_8x7b();
        let cloned = cfg.clone();
        assert_eq!(cfg.hidden_size, cloned.hidden_size);
        assert_eq!(cfg.num_local_experts, cloned.num_local_experts);
        assert_eq!(cfg.model_type, cloned.model_type);
    }

    #[test]
    fn test_lcg_varied_expert_topk() {
        let mut s = 23u64;
        let n_experts = 8usize;
        for _ in 0..5 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let topk = ((s % n_experts as u64) + 1) as usize;
            let cfg = MixtralConfig {
                num_local_experts: n_experts,
                num_experts_per_tok: topk,
                ..MixtralConfig::default()
            };
            assert!(cfg.validate().is_ok(), "topk={topk} failed");
        }
    }
}
