//! # DeepSeek-V2 Configuration
//!
//! Configuration for DeepSeek-V2, which introduces:
//! - **Multi-head Latent Attention (MLA)**: Compresses KV cache via low-rank projections,
//!   dramatically reducing memory bandwidth vs. standard MHA.
//! - **Mixture of Experts (MoE)**: Most layers use sparse MoE FFN with shared + routed experts.
//! - **GroupLimitedGreedy routing**: Experts are selected within groups to encourage diversity.

use serde::{Deserialize, Serialize};
use trustformers_core::errors::invalid_config;
use trustformers_core::traits::Config;

/// Method used to select top-k experts from the routing logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopKMethod {
    /// Route to top-k experts while respecting per-group limits — the default DeepSeek-V2 method.
    GroupLimitedGreedy,
    /// Pure greedy top-k without auxiliary load-balancing loss.
    Noaux,
}

impl std::fmt::Display for TopKMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TopKMethod::GroupLimitedGreedy => write!(f, "GroupLimitedGreedy"),
            TopKMethod::Noaux => write!(f, "Noaux"),
        }
    }
}

/// Activation function used in the MLP / expert FFN blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    /// Sigmoid-weighted Linear Unit — default for DeepSeek-V2.
    SiLU,
    /// Gaussian Error Linear Unit.
    GeLU,
}

impl std::fmt::Display for ActivationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivationType::SiLU => write!(f, "silu"),
            ActivationType::GeLU => write!(f, "gelu"),
        }
    }
}

/// DeepSeek-V2 model configuration.
///
/// Reference: "DeepSeek-V2: A Strong, Economical, and Efficient
/// Mixture-of-Experts Language Model" (DeepSeek-AI, 2024).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekV2Config {
    // -----------------------------------------------------------------------
    // Vocabulary / embedding
    // -----------------------------------------------------------------------
    /// Vocabulary size (default 102400).
    pub vocab_size: usize,
    /// Hidden (model) dimension (default 5120).
    pub hidden_size: usize,
    /// FFN / expert intermediate dimension (default 12288).
    pub intermediate_size: usize,

    // -----------------------------------------------------------------------
    // Transformer depth & attention
    // -----------------------------------------------------------------------
    /// Total number of transformer layers (default 60).
    pub num_hidden_layers: usize,
    /// Number of query heads (default 128).
    pub num_attention_heads: usize,

    // -----------------------------------------------------------------------
    // MLA-specific fields
    // -----------------------------------------------------------------------
    /// Low-rank KV compression dimension: hidden → kv_lora_rank (default 512).
    ///
    /// The combined KV latent vector `c_KV` has this size; from it both K and V
    /// are expanded, giving an O(kv_lora_rank) KV cache instead of O(num_heads × head_dim).
    pub kv_lora_rank: usize,
    /// Query down-projection rank (default 1536). Set to 0 to disable query compression.
    pub q_lora_rank: usize,
    /// Per-head dimension for RoPE-applied portion of Q and K (default 64).
    pub qk_rope_head_dim: usize,
    /// Per-head dimension for non-RoPE portion of Q and K (default 128).
    pub qk_nope_head_dim: usize,
    /// Per-head dimension for values (default 128).
    pub v_head_dim: usize,

    // -----------------------------------------------------------------------
    // MoE fields
    // -----------------------------------------------------------------------
    /// Number of routed experts activated per token (default 6).
    pub num_experts_per_tok: usize,
    /// Total number of routed experts in the expert pool (default 160).
    pub n_routed_experts: usize,
    /// Number of shared (always-active) experts per MoE layer (default 2).
    pub n_shared_experts: usize,
    /// Scaling factor applied to routed-expert outputs (default 1.0).
    pub routed_scaling_factor: f32,
    /// Expert selection strategy.
    pub topk_method: TopKMethod,
    /// Number of expert groups for GroupLimitedGreedy routing (default 8).
    pub n_group: usize,
    /// Maximum experts per group selected by GroupLimitedGreedy (default 3).
    pub topk_group: usize,
    /// Weight of the auxiliary load-balancing loss (default 0.001).
    pub aux_loss_alpha: f32,

    // -----------------------------------------------------------------------
    // Standard fields
    // -----------------------------------------------------------------------
    /// Maximum context length supported (default 163840).
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon (default 1e-6).
    pub rms_norm_eps: f64,
    /// RoPE base frequency (default 10000.0).
    pub rope_theta: f64,
    /// Activation function for expert / MLP FFN (default SiLU).
    pub hidden_act: ActivationType,
    /// Weight initialisation standard deviation (default 0.02).
    pub initializer_range: f32,
    /// The first `first_k_dense_replace` layers use a plain dense FFN rather than MoE (default 1).
    pub first_k_dense_replace: usize,
    /// MoE layers appear every `moe_layer_freq` layers after `first_k_dense_replace` (default 1).
    pub moe_layer_freq: usize,
}

impl Default for DeepSeekV2Config {
    fn default() -> Self {
        Self {
            vocab_size: 102400,
            hidden_size: 5120,
            intermediate_size: 12288,
            num_hidden_layers: 60,
            num_attention_heads: 128,
            kv_lora_rank: 512,
            q_lora_rank: 1536,
            qk_rope_head_dim: 64,
            qk_nope_head_dim: 128,
            v_head_dim: 128,
            num_experts_per_tok: 6,
            n_routed_experts: 160,
            n_shared_experts: 2,
            routed_scaling_factor: 1.0,
            topk_method: TopKMethod::GroupLimitedGreedy,
            n_group: 8,
            topk_group: 3,
            aux_loss_alpha: 0.001,
            max_position_embeddings: 163840,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            hidden_act: ActivationType::SiLU,
            initializer_range: 0.02,
            first_k_dense_replace: 1,
            moe_layer_freq: 1,
        }
    }
}

impl Config for DeepSeekV2Config {
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
        if self.kv_lora_rank == 0 {
            return Err(invalid_config("kv_lora_rank", "must be > 0".to_string()));
        }
        if self.qk_rope_head_dim == 0 {
            return Err(invalid_config(
                "qk_rope_head_dim",
                "must be > 0".to_string(),
            ));
        }
        if self.qk_nope_head_dim == 0 {
            return Err(invalid_config(
                "qk_nope_head_dim",
                "must be > 0".to_string(),
            ));
        }
        if self.v_head_dim == 0 {
            return Err(invalid_config("v_head_dim", "must be > 0".to_string()));
        }
        if self.n_routed_experts == 0 {
            return Err(invalid_config(
                "n_routed_experts",
                "must be > 0".to_string(),
            ));
        }
        if self.num_experts_per_tok == 0 {
            return Err(invalid_config(
                "num_experts_per_tok",
                "must be > 0".to_string(),
            ));
        }
        if self.num_experts_per_tok > self.n_routed_experts {
            return Err(invalid_config(
                "num_experts_per_tok",
                "must be <= n_routed_experts".to_string(),
            ));
        }
        if self.n_group == 0 {
            return Err(invalid_config("n_group", "must be > 0".to_string()));
        }
        if self.num_hidden_layers == 0 {
            return Err(invalid_config(
                "num_hidden_layers",
                "must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "DeepSeek-V2"
    }
}

impl DeepSeekV2Config {
    /// Returns the full per-head query dimension (RoPE + non-RoPE portions).
    pub fn qk_head_dim(&self) -> usize {
        self.qk_rope_head_dim + self.qk_nope_head_dim
    }

    /// Returns whether the layer at `layer_idx` is a dense (non-MoE) FFN layer.
    ///
    /// The first `first_k_dense_replace` layers always use dense FFN regardless of
    /// `moe_layer_freq`.
    pub fn is_dense_layer(&self, layer_idx: usize) -> bool {
        if layer_idx < self.first_k_dense_replace {
            return true;
        }
        // After the initial dense layers, every `moe_layer_freq`-th layer is MoE.
        // moe_layer_freq == 1 → all remaining layers are MoE.
        !(layer_idx - self.first_k_dense_replace).is_multiple_of(self.moe_layer_freq)
    }

    /// KV cache size per token for **standard MHA** (for comparison with MLA).
    ///
    /// MHA stores `num_heads * (qk_nope_head_dim + qk_rope_head_dim + v_head_dim)` f32 values
    /// per layer per token.
    pub fn mha_kv_cache_per_token_per_layer(&self) -> usize {
        self.num_attention_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim + self.v_head_dim)
    }

    /// KV cache size per token for **MLA**.
    ///
    /// MLA only needs to store the compressed latent `c_KV` (size `kv_lora_rank`) plus the
    /// RoPE key portion `k_pe` (size `qk_rope_head_dim`) — everything else is re-expanded on
    /// the fly.
    pub fn mla_kv_cache_per_token_per_layer(&self) -> usize {
        self.kv_lora_rank + self.qk_rope_head_dim
    }

    /// Ratio of MLA KV cache size to standard MHA KV cache size (smaller = better).
    ///
    /// For the 236B DeepSeek-V2 default config this is approximately 1/5.75.
    pub fn kv_cache_compression_ratio(&self) -> f64 {
        let mla = self.mla_kv_cache_per_token_per_layer() as f64;
        let mha = self.mha_kv_cache_per_token_per_layer() as f64;
        if mha == 0.0 {
            return 1.0;
        }
        mla / mha
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepseekv2_default_vocab_size() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.vocab_size, 102400);
    }

    #[test]
    fn test_deepseekv2_default_hidden_size() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.hidden_size, 5120);
    }

    #[test]
    fn test_deepseekv2_default_num_hidden_layers() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.num_hidden_layers, 60);
    }

    #[test]
    fn test_deepseekv2_default_num_attention_heads() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.num_attention_heads, 128);
    }

    #[test]
    fn test_deepseekv2_default_kv_lora_rank() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.kv_lora_rank, 512);
    }

    #[test]
    fn test_deepseekv2_default_q_lora_rank() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.q_lora_rank, 1536);
    }

    #[test]
    fn test_deepseekv2_default_qk_rope_head_dim() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.qk_rope_head_dim, 64);
    }

    #[test]
    fn test_deepseekv2_default_qk_nope_head_dim() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.qk_nope_head_dim, 128);
    }

    #[test]
    fn test_deepseekv2_default_v_head_dim() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.v_head_dim, 128);
    }

    #[test]
    fn test_deepseekv2_default_n_routed_experts() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.n_routed_experts, 160);
    }

    #[test]
    fn test_deepseekv2_default_num_experts_per_tok() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.num_experts_per_tok, 6);
    }

    #[test]
    fn test_deepseekv2_default_n_shared_experts() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.n_shared_experts, 2);
    }

    #[test]
    fn test_deepseekv2_validate_passes_default() {
        let cfg = DeepSeekV2Config::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_deepseekv2_validate_fails_zero_vocab_size() {
        let cfg = DeepSeekV2Config {
            vocab_size: 0,
            ..DeepSeekV2Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_deepseekv2_validate_fails_zero_hidden_size() {
        let cfg = DeepSeekV2Config {
            hidden_size: 0,
            ..DeepSeekV2Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_deepseekv2_validate_fails_zero_kv_lora_rank() {
        let cfg = DeepSeekV2Config {
            kv_lora_rank: 0,
            ..DeepSeekV2Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_deepseekv2_validate_fails_experts_per_tok_exceeds_total() {
        let cfg = DeepSeekV2Config {
            num_experts_per_tok: 200,
            ..DeepSeekV2Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_deepseekv2_qk_head_dim() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(
            cfg.qk_head_dim(),
            cfg.qk_rope_head_dim + cfg.qk_nope_head_dim
        );
        assert_eq!(cfg.qk_head_dim(), 192);
    }

    #[test]
    fn test_deepseekv2_is_dense_layer_first() {
        let cfg = DeepSeekV2Config::default();
        assert!(cfg.is_dense_layer(0));
    }

    #[test]
    fn test_deepseekv2_mla_kv_cache_smaller_than_mha() {
        let cfg = DeepSeekV2Config::default();
        let ratio = cfg.kv_cache_compression_ratio();
        assert!(ratio < 1.0, "MLA should compress KV cache relative to MHA");
    }

    #[test]
    fn test_deepseekv2_topk_method_default() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.topk_method, TopKMethod::GroupLimitedGreedy);
    }

    #[test]
    fn test_deepseekv2_hidden_act_default() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.hidden_act, ActivationType::SiLU);
    }

    #[test]
    fn test_deepseekv2_mha_kv_cache_size() {
        let cfg = DeepSeekV2Config::default();
        let expected = cfg.num_attention_heads
            * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim);
        assert_eq!(cfg.mha_kv_cache_per_token_per_layer(), expected);
    }

    #[test]
    fn test_deepseekv2_mla_kv_cache_size() {
        let cfg = DeepSeekV2Config::default();
        let expected = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
        assert_eq!(cfg.mla_kv_cache_per_token_per_layer(), expected);
    }

    #[test]
    fn test_deepseekv2_lcg_values_in_range() {
        let mut s = 42u64;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (s % 1000) as f32 / 1000.0;
        assert!((0.0..1.0).contains(&v));
    }

    #[test]
    fn test_deepseekv2_architecture_name() {
        let cfg = DeepSeekV2Config::default();
        assert_eq!(cfg.architecture(), "DeepSeek-V2");
    }
}
