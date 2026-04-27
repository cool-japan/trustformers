/// Configuration for AI21 Jamba — the first production Mamba+Transformer hybrid.
///
/// Jamba interleaves Mamba SSM blocks with Transformer attention blocks,
/// optionally using Mixture of Experts (MoE) in the attention layers.
///
/// Layer pattern (default):
///   Layers 0,1,2 → Mamba
///   Layer 3      → Attention + MoE   (attn_layer_offset=3, period=8)
///   Layers 4,5,6 → Mamba
///   Layer 7      → Attention + Dense
///   ...
#[derive(Debug, Clone)]
pub struct JambaConfig {
    /// Vocabulary size (default 65536)
    pub vocab_size: usize,
    /// Hidden dimension (default 4096)
    pub hidden_size: usize,
    /// MLP intermediate dimension for experts (default 14336)
    pub intermediate_size: usize,
    /// Total number of decoder layers (default 32)
    pub num_hidden_layers: usize,
    /// Number of attention heads (default 32)
    pub num_attention_heads: usize,
    /// Number of key-value heads for GQA (default 8)
    pub num_key_value_heads: usize,
    /// Index of the first attention layer (default 3)
    pub attn_layer_offset: usize,
    /// Attention repeats every N layers (default 8: layers 3,11,19,27)
    pub attn_layer_period: usize,
    /// Index of the first MoE layer (default 1)
    pub expert_layer_offset: usize,
    /// MoE repeats every N layers (default 2)
    pub expert_layer_period: usize,
    /// Total number of MoE experts (default 16)
    pub num_experts: usize,
    /// Experts activated per token (default 2)
    pub num_experts_per_tok: usize,
    /// Mamba SSM state dimension (default 16)
    pub mamba_d_state: usize,
    /// Mamba local conv width (default 4)
    pub mamba_d_conv: usize,
    /// Mamba expansion factor (default 2)
    pub mamba_expand: usize,
    /// Epsilon for RMSNorm
    pub rms_norm_eps: f64,
    /// Base frequency for RoPE
    pub rope_theta: f64,
}

impl JambaConfig {
    /// Jamba 1.5B simplified configuration (single-expert MoE for open weights).
    pub fn jamba_1_5b() -> Self {
        Self {
            vocab_size: 65536,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            attn_layer_offset: 3,
            attn_layer_period: 8,
            expert_layer_offset: 1,
            expert_layer_period: 2,
            num_experts: 16,
            num_experts_per_tok: 2,
            mamba_d_state: 16,
            mamba_d_conv: 4,
            mamba_expand: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }

    /// Small test configuration for unit testing (hidden_size=64, num_layers=8).
    pub fn small_test() -> Self {
        Self {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 8,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            attn_layer_offset: 3,
            attn_layer_period: 8,
            expert_layer_offset: 1,
            expert_layer_period: 2,
            num_experts: 4,
            num_experts_per_tok: 2,
            mamba_d_state: 8,
            mamba_d_conv: 4,
            mamba_expand: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }

    /// Whether layer `layer_idx` is an attention (transformer) layer.
    ///
    /// Attention layers occur at: layer_idx ≡ attn_layer_offset (mod attn_layer_period)
    /// e.g. with offset=3, period=8: layers 3, 11, 19, 27
    pub fn is_attention_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.attn_layer_offset
            && (layer_idx - self.attn_layer_offset).is_multiple_of(self.attn_layer_period)
    }

    /// Whether layer `layer_idx` uses Mixture of Experts.
    ///
    /// MoE layers occur at: layer_idx ≡ expert_layer_offset (mod expert_layer_period)
    /// and only among attention layers.
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.is_attention_layer(layer_idx)
            && layer_idx >= self.expert_layer_offset
            && (layer_idx - self.expert_layer_offset).is_multiple_of(self.expert_layer_period)
    }

    /// Head dimension for attention: hidden_size / num_attention_heads
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Mamba inner dimension: hidden_size * mamba_expand
    pub fn mamba_inner_dim(&self) -> usize {
        self.hidden_size * self.mamba_expand
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jamba_1_5b_vocab_size() {
        let cfg = JambaConfig::jamba_1_5b();
        assert_eq!(cfg.vocab_size, 65536);
    }

    #[test]
    fn test_jamba_1_5b_hidden_size() {
        let cfg = JambaConfig::jamba_1_5b();
        assert_eq!(cfg.hidden_size, 4096);
    }

    #[test]
    fn test_jamba_1_5b_num_hidden_layers() {
        let cfg = JambaConfig::jamba_1_5b();
        assert_eq!(cfg.num_hidden_layers, 32);
    }

    #[test]
    fn test_jamba_1_5b_num_attention_heads() {
        let cfg = JambaConfig::jamba_1_5b();
        assert_eq!(cfg.num_attention_heads, 32);
    }

    #[test]
    fn test_jamba_1_5b_num_key_value_heads() {
        let cfg = JambaConfig::jamba_1_5b();
        assert_eq!(cfg.num_key_value_heads, 8);
    }

    #[test]
    fn test_jamba_1_5b_moe_params() {
        let cfg = JambaConfig::jamba_1_5b();
        assert_eq!(cfg.num_experts, 16);
        assert_eq!(cfg.num_experts_per_tok, 2);
    }

    #[test]
    fn test_jamba_1_5b_attn_layer_pattern() {
        let cfg = JambaConfig::jamba_1_5b();
        assert_eq!(cfg.attn_layer_offset, 3);
        assert_eq!(cfg.attn_layer_period, 8);
    }

    #[test]
    fn test_jamba_1_5b_expert_layer_pattern() {
        let cfg = JambaConfig::jamba_1_5b();
        assert_eq!(cfg.expert_layer_offset, 1);
        assert_eq!(cfg.expert_layer_period, 2);
    }

    #[test]
    fn test_jamba_small_test_config() {
        let cfg = JambaConfig::small_test();
        assert_eq!(cfg.vocab_size, 256);
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.num_hidden_layers, 8);
    }

    #[test]
    fn test_jamba_is_attention_layer_true() {
        let cfg = JambaConfig::jamba_1_5b();
        assert!(cfg.is_attention_layer(3));
        assert!(cfg.is_attention_layer(11));
        assert!(cfg.is_attention_layer(19));
        assert!(cfg.is_attention_layer(27));
    }

    #[test]
    fn test_jamba_is_attention_layer_false() {
        let cfg = JambaConfig::jamba_1_5b();
        assert!(!cfg.is_attention_layer(0));
        assert!(!cfg.is_attention_layer(1));
        assert!(!cfg.is_attention_layer(4));
    }

    #[test]
    fn test_jamba_is_moe_layer() {
        let cfg = JambaConfig::jamba_1_5b();
        // Layer 3 is attn; expert offset=1 period=2 → layers 1,3,5... so 3 is MoE if attn
        // Layer 3: (3 - 1) % 2 == 0 → yes MoE, and it's attn
        assert!(cfg.is_moe_layer(3));
    }

    #[test]
    fn test_jamba_is_moe_layer_false_non_attn() {
        let cfg = JambaConfig::jamba_1_5b();
        // Layer 5 is not attn (3+8=11 is next attn), so not MoE
        assert!(!cfg.is_moe_layer(5));
    }

    #[test]
    fn test_jamba_head_dim() {
        let cfg = JambaConfig::jamba_1_5b();
        assert_eq!(cfg.head_dim(), 4096 / 32);
    }

    #[test]
    fn test_jamba_mamba_inner_dim() {
        let cfg = JambaConfig::jamba_1_5b();
        assert_eq!(cfg.mamba_inner_dim(), cfg.hidden_size * cfg.mamba_expand);
        assert_eq!(cfg.mamba_inner_dim(), 8192);
    }

    #[test]
    fn test_jamba_mamba_d_state() {
        let cfg = JambaConfig::jamba_1_5b();
        assert_eq!(cfg.mamba_d_state, 16);
    }

    #[test]
    fn test_jamba_mamba_d_conv() {
        let cfg = JambaConfig::jamba_1_5b();
        assert_eq!(cfg.mamba_d_conv, 4);
    }

    #[test]
    fn test_jamba_mamba_expand() {
        let cfg = JambaConfig::jamba_1_5b();
        assert_eq!(cfg.mamba_expand, 2);
    }

    #[test]
    fn test_jamba_rms_norm_eps() {
        let cfg = JambaConfig::jamba_1_5b();
        assert!(cfg.rms_norm_eps > 0.0);
    }

    #[test]
    fn test_jamba_rope_theta() {
        let cfg = JambaConfig::jamba_1_5b();
        assert!((cfg.rope_theta - 10000.0).abs() < 1.0);
    }

    #[test]
    fn test_jamba_lcg_values_in_range() {
        let mut s = 42u64;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (s % 1000) as f32 / 1000.0;
        assert!((0.0..1.0).contains(&v));
    }

    #[test]
    fn test_jamba_small_head_dim() {
        let cfg = JambaConfig::small_test();
        assert_eq!(cfg.head_dim(), 64 / 4);
    }
}
