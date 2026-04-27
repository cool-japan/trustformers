//! InternLM-2 configuration.
//!
//! Reference: <https://huggingface.co/internlm/internlm2-7b>

use serde::{Deserialize, Serialize};

/// Configuration for InternLM-2 models.
///
/// InternLM-2 uses Grouped Query Attention (GQA) with RoPE positional embeddings
/// and optional NTK-aware dynamic scaling for extended context lengths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternLm2Config {
    /// Vocabulary size (default: 92544)
    pub vocab_size: usize,
    /// Hidden dimension size (default: 4096)
    pub hidden_size: usize,
    /// Number of transformer layers (default: 32)
    pub num_hidden_layers: usize,
    /// Number of query attention heads (default: 32)
    pub num_attention_heads: usize,
    /// Number of key-value attention heads for GQA (default: 8)
    pub num_key_value_heads: usize,
    /// Intermediate MLP size for SwiGLU (default: 14336)
    pub intermediate_size: usize,
    /// Maximum sequence length (default: 32768)
    pub max_position_embeddings: usize,
    /// RoPE base theta (default: 1_000_000.0)
    pub rope_theta: f64,
    /// Optional NTK dynamic scaling factor for extended context
    pub rope_scaling: Option<f64>,
    /// Activation function name (default: "silu")
    pub hidden_act: String,
    /// RMS normalization epsilon (default: 1e-5)
    pub rms_norm_eps: f64,
    /// Whether to tie word embedding weights (default: false)
    pub tie_word_embeddings: bool,
    /// Whether to use KV cache (default: true)
    pub use_cache: bool,
}

impl Default for InternLm2Config {
    fn default() -> Self {
        Self {
            vocab_size: 92544,
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 14336,
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            rope_scaling: None,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
            use_cache: true,
        }
    }
}

impl InternLm2Config {
    /// Create a new InternLM-2 config with the given parameters.
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        intermediate_size: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        rope_scaling: Option<f64>,
        hidden_act: &str,
        rms_norm_eps: f64,
        tie_word_embeddings: bool,
        use_cache: bool,
    ) -> Self {
        Self {
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            max_position_embeddings,
            rope_theta,
            rope_scaling,
            hidden_act: hidden_act.to_string(),
            rms_norm_eps,
            tie_word_embeddings,
            use_cache,
        }
    }

    /// InternLM-2 7B configuration
    pub fn internlm2_7b() -> Self {
        Self::default()
    }

    /// InternLM-2 20B configuration
    pub fn internlm2_20b() -> Self {
        Self {
            hidden_size: 6144,
            num_hidden_layers: 48,
            num_attention_heads: 48,
            num_key_value_heads: 8,
            intermediate_size: 16384,
            ..Self::default()
        }
    }

    /// Head dimension for this config
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// GQA ratio: how many Q heads share one KV head
    pub fn gqa_ratio(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internlm2_default_vocab_size() {
        let cfg = InternLm2Config::default();
        assert_eq!(cfg.vocab_size, 92544);
    }

    #[test]
    fn test_internlm2_default_hidden_size() {
        let cfg = InternLm2Config::default();
        assert_eq!(cfg.hidden_size, 4096);
    }

    #[test]
    fn test_internlm2_default_num_hidden_layers() {
        let cfg = InternLm2Config::default();
        assert_eq!(cfg.num_hidden_layers, 32);
    }

    #[test]
    fn test_internlm2_default_num_attention_heads() {
        let cfg = InternLm2Config::default();
        assert_eq!(cfg.num_attention_heads, 32);
    }

    #[test]
    fn test_internlm2_default_num_key_value_heads() {
        let cfg = InternLm2Config::default();
        assert_eq!(cfg.num_key_value_heads, 8);
    }

    #[test]
    fn test_internlm2_default_intermediate_size() {
        let cfg = InternLm2Config::default();
        assert_eq!(cfg.intermediate_size, 14336);
    }

    #[test]
    fn test_internlm2_default_max_position_embeddings() {
        let cfg = InternLm2Config::default();
        assert_eq!(cfg.max_position_embeddings, 32768);
    }

    #[test]
    fn test_internlm2_default_rope_theta() {
        let cfg = InternLm2Config::default();
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_internlm2_default_rope_scaling_none() {
        let cfg = InternLm2Config::default();
        assert!(cfg.rope_scaling.is_none());
    }

    #[test]
    fn test_internlm2_default_hidden_act() {
        let cfg = InternLm2Config::default();
        assert_eq!(cfg.hidden_act, "silu");
    }

    #[test]
    fn test_internlm2_default_use_cache() {
        let cfg = InternLm2Config::default();
        assert!(cfg.use_cache);
    }

    #[test]
    fn test_internlm2_7b_preset() {
        let cfg = InternLm2Config::internlm2_7b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
    }

    #[test]
    fn test_internlm2_20b_preset() {
        let cfg = InternLm2Config::internlm2_20b();
        assert_eq!(cfg.hidden_size, 6144);
        assert_eq!(cfg.num_hidden_layers, 48);
        assert_eq!(cfg.num_attention_heads, 48);
    }

    #[test]
    fn test_internlm2_head_dim() {
        let cfg = InternLm2Config::default();
        assert_eq!(cfg.head_dim(), 4096 / 32);
    }

    #[test]
    fn test_internlm2_gqa_ratio() {
        let cfg = InternLm2Config::default();
        assert_eq!(cfg.gqa_ratio(), 32 / 8);
    }

    #[test]
    fn test_internlm2_new_constructor() {
        let cfg = InternLm2Config::new(
            92544,
            4096,
            32,
            32,
            8,
            14336,
            32768,
            1_000_000.0,
            None,
            "silu",
            1e-5,
            false,
            true,
        );
        assert_eq!(cfg.vocab_size, 92544);
        assert_eq!(cfg.num_attention_heads, 32);
    }

    #[test]
    fn test_internlm2_rope_scaling_with_value() {
        let cfg = InternLm2Config {
            rope_scaling: Some(4.0),
            ..InternLm2Config::default()
        };
        assert!(cfg.rope_scaling.is_some());
        let v = cfg.rope_scaling.unwrap_or(1.0);
        assert!((v - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_internlm2_tie_word_embeddings_default_false() {
        let cfg = InternLm2Config::default();
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn test_internlm2_20b_gqa_ratio() {
        let cfg = InternLm2Config::internlm2_20b();
        assert_eq!(cfg.gqa_ratio(), 48 / 8);
    }

    #[test]
    fn test_internlm2_head_dim_20b() {
        let cfg = InternLm2Config::internlm2_20b();
        assert_eq!(cfg.head_dim(), 6144 / 48);
    }

    #[test]
    fn test_internlm2_rms_norm_eps_default() {
        let cfg = InternLm2Config::default();
        assert!(cfg.rms_norm_eps > 0.0);
        assert!(cfg.rms_norm_eps < 1e-3);
    }

    #[test]
    fn test_internlm2_lcg_values_in_range() {
        let mut s = 42u64;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (s % 1000) as f32 / 1000.0;
        assert!((0.0..1.0).contains(&v));
    }
}
