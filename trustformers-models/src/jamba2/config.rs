/// Configuration for AI21 Jamba-2 — the successor hybrid Mamba-Transformer model.
///
/// Jamba-2 interleaves Mamba SSM layers with Transformer attention layers,
/// and uses Mixture-of-Experts (MoE) in some layers.
///
/// Layer pattern (default):
///   attn_layer_offset=4, attn_layer_period=8  → attn layers at 4,12,20,28,...
///   expert_layer_offset=1, expert_layer_period=2 → MoE layers at 1,3,5,...
#[derive(Debug, Clone)]
pub struct Jamba2Config {
    /// Vocabulary size (default 65536)
    pub vocab_size: usize,
    /// Hidden dimension (default 4096)
    pub hidden_size: usize,
    /// MLP intermediate dimension (default 14336)
    pub intermediate_size: usize,
    /// Total number of decoder layers (default 32)
    pub num_hidden_layers: usize,
    /// Number of attention heads (default 32)
    pub num_attention_heads: usize,
    /// Number of key-value heads for GQA (default 8)
    pub num_key_value_heads: usize,
    /// Per-head dimension (default 128)
    pub head_dim: usize,
    /// Mamba SSM state dimension (default 16)
    pub mamba_d_state: usize,
    /// Mamba depthwise conv kernel size (default 4)
    pub mamba_d_conv: usize,
    /// Mamba expansion factor: inner_dim = expand * hidden_size (default 2)
    pub mamba_expand: usize,
    /// Mamba delta (dt) rank (default 256; "auto" = ceil(hidden_size/16))
    pub mamba_dt_rank: usize,
    /// Index of the first attention layer (default 4)
    pub attn_layer_offset: usize,
    /// Attention layer repeats every N layers (default 8)
    pub attn_layer_period: usize,
    /// Index of the first MoE layer (default 1)
    pub expert_layer_offset: usize,
    /// MoE repeats every N layers (default 2)
    pub expert_layer_period: usize,
    /// Total number of MoE experts (default 16)
    pub num_experts: usize,
    /// Experts activated per token (default 2)
    pub num_experts_per_tok: usize,
    /// Maximum sequence length (default 262144)
    pub max_position_embeddings: usize,
    /// Epsilon for RMSNorm (default 1e-5)
    pub rms_norm_eps: f64,
    /// Base frequency for RoPE (default 10000.0)
    pub rope_theta: f64,
    /// Activation function (default "silu")
    pub hidden_act: String,
    /// Attention dropout probability (default 0.0)
    pub attention_dropout: f32,
    /// Whether to tie word embeddings (default false)
    pub tie_word_embeddings: bool,
}

/// Layer type classification for a Jamba-2 decoder layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerType {
    /// Pure Mamba SSM layer (no attention, no MoE)
    Mamba,
    /// Attention layer with standard dense FFN
    Attention,
    /// Mamba SSM layer with MoE FFN
    MambaMoE,
    /// Attention layer with MoE FFN
    AttentionMoE,
}

impl Default for Jamba2Config {
    fn default() -> Self {
        Self {
            vocab_size: 65536,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            mamba_d_state: 16,
            mamba_d_conv: 4,
            mamba_expand: 2,
            mamba_dt_rank: 256,
            attn_layer_offset: 4,
            attn_layer_period: 8,
            expert_layer_offset: 1,
            expert_layer_period: 2,
            num_experts: 16,
            num_experts_per_tok: 2,
            max_position_embeddings: 262144,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            hidden_act: "silu".to_string(),
            attention_dropout: 0.0,
            tie_word_embeddings: false,
        }
    }
}

impl Jamba2Config {
    /// Validate the configuration for consistency.
    pub fn validate(&self) -> Result<(), Jamba2ConfigError> {
        if self.vocab_size == 0 {
            return Err(Jamba2ConfigError::InvalidField(
                "vocab_size must be > 0".to_string(),
            ));
        }
        if self.hidden_size == 0 {
            return Err(Jamba2ConfigError::InvalidField(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if self.num_hidden_layers == 0 {
            return Err(Jamba2ConfigError::InvalidField(
                "num_hidden_layers must be > 0".to_string(),
            ));
        }
        if self.num_attention_heads == 0 {
            return Err(Jamba2ConfigError::InvalidField(
                "num_attention_heads must be > 0".to_string(),
            ));
        }
        if self.num_key_value_heads == 0 {
            return Err(Jamba2ConfigError::InvalidField(
                "num_key_value_heads must be > 0".to_string(),
            ));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(Jamba2ConfigError::InvalidField(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.mamba_expand == 0 {
            return Err(Jamba2ConfigError::InvalidField(
                "mamba_expand must be > 0".to_string(),
            ));
        }
        if self.num_experts == 0 {
            return Err(Jamba2ConfigError::InvalidField(
                "num_experts must be > 0".to_string(),
            ));
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
            return Err(Jamba2ConfigError::InvalidField(format!(
                "num_experts_per_tok ({}) must be in [1, num_experts ({})]",
                self.num_experts_per_tok, self.num_experts
            )));
        }
        if self.attn_layer_period == 0 {
            return Err(Jamba2ConfigError::InvalidField(
                "attn_layer_period must be > 0".to_string(),
            ));
        }
        if self.expert_layer_period == 0 {
            return Err(Jamba2ConfigError::InvalidField(
                "expert_layer_period must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Returns true if `layer_idx` is an attention (Transformer) layer.
    ///
    /// Condition: layer_idx >= attn_layer_offset AND
    ///            (layer_idx - attn_layer_offset) % attn_layer_period == 0
    pub fn is_attention_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.attn_layer_offset
            && (layer_idx - self.attn_layer_offset).is_multiple_of(self.attn_layer_period)
    }

    /// Returns true if `layer_idx` uses Mixture of Experts.
    ///
    /// Condition: layer_idx >= expert_layer_offset AND
    ///            (layer_idx - expert_layer_offset) % expert_layer_period == 0
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.expert_layer_offset
            && (layer_idx - self.expert_layer_offset).is_multiple_of(self.expert_layer_period)
    }

    /// Classify the layer at `layer_idx` into its [`LayerType`].
    pub fn layer_type(&self, layer_idx: usize) -> LayerType {
        let is_attn = self.is_attention_layer(layer_idx);
        let is_moe = self.is_moe_layer(layer_idx);
        match (is_attn, is_moe) {
            (true, true) => LayerType::AttentionMoE,
            (true, false) => LayerType::Attention,
            (false, true) => LayerType::MambaMoE,
            (false, false) => LayerType::Mamba,
        }
    }

    /// Mamba inner dimension: expand * hidden_size.
    pub fn mamba_inner_dim(&self) -> usize {
        self.mamba_expand * self.hidden_size
    }

    /// Auto-compute dt_rank as ceil(hidden_size / 16) if not set explicitly.
    pub fn effective_dt_rank(&self) -> usize {
        if self.mamba_dt_rank == 0 {
            self.hidden_size.div_ceil(16)
        } else {
            self.mamba_dt_rank
        }
    }

    /// Jamba-2 1.5B preset configuration.
    pub fn jamba2_1_5b() -> Self {
        Self {
            vocab_size: 65536,
            hidden_size: 2048,
            intermediate_size: 7168,
            num_hidden_layers: 12,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 128,
            mamba_d_state: 16,
            mamba_d_conv: 4,
            mamba_expand: 2,
            mamba_dt_rank: 128, // ceil(2048/16) = 128
            attn_layer_offset: 4,
            attn_layer_period: 8,
            expert_layer_offset: 1,
            expert_layer_period: 2,
            num_experts: 16,
            num_experts_per_tok: 2,
            max_position_embeddings: 262144,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            hidden_act: "silu".to_string(),
            attention_dropout: 0.0,
            tie_word_embeddings: false,
        }
    }
}

/// Errors arising from Jamba-2 configuration validation.
#[derive(Debug, thiserror::Error)]
pub enum Jamba2ConfigError {
    #[error("Invalid configuration field: {0}")]
    InvalidField(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jamba2_default_vocab_size() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.vocab_size, 65536);
    }

    #[test]
    fn test_jamba2_default_hidden_size() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.hidden_size, 4096);
    }

    #[test]
    fn test_jamba2_default_num_hidden_layers() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.num_hidden_layers, 32);
    }

    #[test]
    fn test_jamba2_default_num_experts() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.num_experts, 16);
        assert_eq!(cfg.num_experts_per_tok, 2);
    }

    #[test]
    fn test_jamba2_default_attn_pattern() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.attn_layer_offset, 4);
        assert_eq!(cfg.attn_layer_period, 8);
    }

    #[test]
    fn test_jamba2_default_mamba_params() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.mamba_d_state, 16);
        assert_eq!(cfg.mamba_d_conv, 4);
        assert_eq!(cfg.mamba_expand, 2);
        assert_eq!(cfg.mamba_dt_rank, 256);
    }

    #[test]
    fn test_jamba2_validate_passes_default() {
        let cfg = Jamba2Config::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_jamba2_validate_fails_zero_vocab_size() {
        let cfg = Jamba2Config {
            vocab_size: 0,
            ..Jamba2Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_jamba2_validate_fails_heads_not_divisible_by_kv_heads() {
        let cfg = Jamba2Config {
            num_attention_heads: 32,
            num_key_value_heads: 7,
            ..Jamba2Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_jamba2_validate_fails_experts_per_tok_exceeds_total() {
        let cfg = Jamba2Config {
            num_experts_per_tok: 20,
            num_experts: 16,
            ..Jamba2Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_jamba2_is_attention_layer() {
        let cfg = Jamba2Config::default();
        // offset=4, period=8 → attn at 4, 12, 20, 28
        assert!(cfg.is_attention_layer(4));
        assert!(cfg.is_attention_layer(12));
        assert!(!cfg.is_attention_layer(0));
        assert!(!cfg.is_attention_layer(5));
    }

    #[test]
    fn test_jamba2_is_moe_layer() {
        let cfg = Jamba2Config::default();
        // offset=1, period=2 → MoE at 1, 3, 5, 7, ...
        assert!(cfg.is_moe_layer(1));
        assert!(cfg.is_moe_layer(3));
        assert!(!cfg.is_moe_layer(2));
        assert!(!cfg.is_moe_layer(0));
    }

    #[test]
    fn test_jamba2_layer_type_mamba() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.layer_type(0), LayerType::Mamba);
        assert_eq!(cfg.layer_type(2), LayerType::Mamba);
    }

    #[test]
    fn test_jamba2_layer_type_mamba_moe() {
        let cfg = Jamba2Config::default();
        // Layer 1: not attn but is MoE
        assert_eq!(cfg.layer_type(1), LayerType::MambaMoE);
    }

    #[test]
    fn test_jamba2_layer_type_attention() {
        let cfg = Jamba2Config::default();
        // Layer 8: attn (4+4=8? no, 4+period=12), actually 4 and 12
        // Layer 4: attn, not MoE (4-1=3, 3%2!=0)
        assert_eq!(cfg.layer_type(4), LayerType::Attention);
    }

    #[test]
    fn test_jamba2_mamba_inner_dim() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.mamba_inner_dim(), 4096 * 2);
    }

    #[test]
    fn test_jamba2_effective_dt_rank() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.effective_dt_rank(), 256);
    }

    #[test]
    fn test_jamba2_effective_dt_rank_auto_when_zero() {
        let cfg = Jamba2Config {
            mamba_dt_rank: 0,
            ..Jamba2Config::default()
        };
        let expected = 4096_usize.div_ceil(16);
        assert_eq!(cfg.effective_dt_rank(), expected);
    }

    #[test]
    fn test_jamba2_1_5b_preset() {
        let cfg = Jamba2Config::jamba2_1_5b();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.mamba_dt_rank, 128);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_jamba2_max_position_embeddings() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.max_position_embeddings, 262144);
    }

    #[test]
    fn test_jamba2_lcg_values_in_range() {
        let mut s = 42u64;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (s % 1000) as f32 / 1000.0;
        assert!((0.0..1.0).contains(&v));
    }

    #[test]
    fn test_jamba2_validate_fails_zero_mamba_expand() {
        let cfg = Jamba2Config {
            mamba_expand: 0,
            ..Jamba2Config::default()
        };
        assert!(cfg.validate().is_err());
    }
}
