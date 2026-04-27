//! Falcon-2 configuration.
//!
//! Reference: <https://huggingface.co/tiiuae/falcon-11B>

use serde::{Deserialize, Serialize};

/// Configuration for Falcon-2 models (e.g., Falcon-11B).
///
/// Falcon-2 uses Multi-Query Attention (MQA), ALiBi positional biases,
/// parallel attention+MLP computation, and GELU activation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Falcon2Config {
    /// Hidden dimension size (default: 4096)
    pub hidden_size: usize,
    /// Number of transformer layers (default: 60)
    pub num_hidden_layers: usize,
    /// Number of query attention heads (default: 64)
    pub num_attention_heads: usize,
    /// Number of key-value heads — MQA uses 1 (default: 1)
    pub num_kv_heads: usize,
    /// MLP intermediate size (default: 16384 = 4 × hidden_size)
    pub intermediate_size: usize,
    /// Maximum sequence length (default: 8192)
    pub max_position_embeddings: usize,
    /// Vocabulary size (default: 65024)
    pub vocab_size: usize,
    /// Layer-norm epsilon (default: 1e-5)
    pub layer_norm_epsilon: f64,
    /// Whether to use ALiBi positional biases (default: true)
    pub use_alibi: bool,
    /// Whether to use parallel attention+MLP (default: true)
    pub parallel_attn: bool,
    /// Whether linear layers use bias terms (default: false)
    pub bias: bool,
    /// Activation function name (default: "gelu")
    pub hidden_act: String,
}

impl Default for Falcon2Config {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            num_hidden_layers: 60,
            num_attention_heads: 64,
            num_kv_heads: 1,
            intermediate_size: 16384,
            max_position_embeddings: 8192,
            vocab_size: 65024,
            layer_norm_epsilon: 1e-5,
            use_alibi: true,
            parallel_attn: true,
            bias: false,
            hidden_act: "gelu".to_string(),
        }
    }
}

impl Falcon2Config {
    /// Per-head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_hidden_size() {
        assert_eq!(Falcon2Config::default().hidden_size, 4096);
    }

    #[test]
    fn test_default_num_layers() {
        assert_eq!(Falcon2Config::default().num_hidden_layers, 60);
    }

    #[test]
    fn test_default_num_kv_heads_is_mqa() {
        assert_eq!(Falcon2Config::default().num_kv_heads, 1);
    }

    #[test]
    fn test_default_use_alibi_true() {
        assert!(Falcon2Config::default().use_alibi);
    }

    #[test]
    fn test_default_parallel_attn_true() {
        assert!(Falcon2Config::default().parallel_attn);
    }

    #[test]
    fn test_default_bias_false() {
        assert!(!Falcon2Config::default().bias);
    }

    #[test]
    fn test_default_hidden_act_is_gelu() {
        assert_eq!(Falcon2Config::default().hidden_act, "gelu");
    }

    #[test]
    fn test_layer_norm_epsilon_positive() {
        assert!(Falcon2Config::default().layer_norm_epsilon > 0.0);
    }

    #[test]
    fn test_head_dim_default() {
        // 4096 / 64 = 64
        assert_eq!(Falcon2Config::default().head_dim(), 64);
    }

    #[test]
    fn test_mqa_expansion_factor() {
        let cfg = Falcon2Config::default();
        // 64 / 1 = 64
        assert_eq!(cfg.num_attention_heads / cfg.num_kv_heads, 64);
    }

    #[test]
    fn test_intermediate_is_4x_hidden() {
        let cfg = Falcon2Config::default();
        assert_eq!(cfg.intermediate_size, 4 * cfg.hidden_size);
    }

    #[test]
    fn test_vocab_size_default() {
        assert_eq!(Falcon2Config::default().vocab_size, 65024);
    }

    #[test]
    fn test_max_position_embeddings_default() {
        assert_eq!(Falcon2Config::default().max_position_embeddings, 8192);
    }

    #[test]
    fn test_toggle_alibi() {
        let cfg = Falcon2Config {
            use_alibi: false,
            ..Falcon2Config::default()
        };
        assert!(!cfg.use_alibi);
    }

    #[test]
    fn test_clone_preserves_fields() {
        let cfg = Falcon2Config::default();
        let cloned = cfg.clone();
        assert_eq!(cfg.hidden_size, cloned.hidden_size);
        assert_eq!(cfg.vocab_size, cloned.vocab_size);
        assert_eq!(cfg.use_alibi, cloned.use_alibi);
    }

    #[test]
    fn test_lcg_varied_hidden_sizes() {
        let mut s = 17u64;
        for _ in 0..5 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let factor = ((s % 8) + 1) as usize;
            let hidden = 64 * factor;
            let heads = 8usize;
            let cfg = Falcon2Config {
                hidden_size: hidden,
                num_attention_heads: heads,
                intermediate_size: hidden * 4,
                ..Falcon2Config::default()
            };
            assert_eq!(cfg.head_dim(), hidden / heads);
        }
    }
}
