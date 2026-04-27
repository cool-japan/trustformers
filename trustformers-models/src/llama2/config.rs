use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// LLaMA-2 model configuration
///
/// LLaMA-2 is the successor to LLaMA-1, featuring:
/// - Grouped Query Attention (GQA) for efficient inference
/// - Extended context window of 4096 tokens
/// - RLHF-fine-tuned chat variants
///
/// Reference: "Llama 2: Open Foundation and Fine-Tuned Chat Models"
///            (Touvron et al., Meta AI, 2023)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLaMA2Config {
    /// Vocabulary size (32000 for LLaMA-2)
    pub hidden_size: usize,
    /// Intermediate (FFN) hidden dimension
    pub intermediate_size: usize,
    /// Number of transformer decoder layers
    pub num_hidden_layers: usize,
    /// Number of query attention heads
    pub num_attention_heads: usize,
    /// Number of key/value attention heads — the GQA parameter
    ///
    /// When `num_key_value_heads < num_attention_heads`, the model uses Grouped
    /// Query Attention where each KV head is shared among
    /// `num_attention_heads / num_key_value_heads` query heads.
    pub num_key_value_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length (4096 for LLaMA-2, vs 2048 for LLaMA-1)
    pub max_position_embeddings: usize,
    /// Base frequency for Rotary Position Embeddings
    pub rope_theta: f32,
    /// Epsilon for RMSNorm stability
    pub rms_norm_eps: f64,
    /// Hidden activation function (always "silu" for LLaMA-2)
    pub hidden_act: String,
    /// Tensor parallelism degree (1 = no sharding, for metadata only)
    pub pretraining_tp: usize,
    /// Whether to use attention bias
    pub attention_bias: bool,
    /// Whether to use MLP bias
    pub mlp_bias: bool,
    /// Beginning-of-sequence token ID
    pub bos_token_id: u32,
    /// End-of-sequence token ID
    pub eos_token_id: u32,
    /// Whether to use KV cache during generation
    pub use_cache: bool,
    /// Optional pad token ID
    pub pad_token_id: Option<u32>,
}

impl Default for LLaMA2Config {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32, // Full MHA by default
            vocab_size: 32000,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            hidden_act: "silu".to_string(),
            pretraining_tp: 1,
            attention_bias: false,
            mlp_bias: false,
            bos_token_id: 1,
            eos_token_id: 2,
            use_cache: true,
            pad_token_id: None,
        }
    }
}

impl Config for LLaMA2Config {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
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
        if self.pretraining_tp == 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "pretraining_tp must be at least 1".to_string(),
                ),
            );
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "LLaMA-2"
    }
}

impl LLaMA2Config {
    /// Compute the head dimension: `hidden_size / num_attention_heads`
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Number of query heads per KV head group
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Whether this config uses Grouped Query Attention
    pub fn uses_gqa(&self) -> bool {
        self.num_key_value_heads < self.num_attention_heads
    }

    /// LLaMA-2 7B — 32 Q heads, 32 KV heads (standard MHA, no GQA sharing)
    ///
    /// Paper: Table 1, 7B model
    pub fn llama2_7b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32, // No KV sharing
            vocab_size: 32000,
            max_position_embeddings: 4096,
            ..Self::default()
        }
    }

    /// LLaMA-2 13B — 40 Q heads, 40 KV heads (standard MHA)
    ///
    /// Paper: Table 1, 13B model
    pub fn llama2_13b() -> Self {
        Self {
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: 40, // No KV sharing
            vocab_size: 32000,
            max_position_embeddings: 4096,
            ..Self::default()
        }
    }

    /// LLaMA-2 70B — 64 Q heads, 8 KV heads (GQA with 8x sharing)
    ///
    /// Paper: Table 1, 70B model — the first LLaMA variant to use GQA in production
    pub fn llama2_70b() -> Self {
        Self {
            hidden_size: 8192,
            intermediate_size: 28672,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: 8, // GQA: each KV head serves 8 Q heads
            vocab_size: 32000,
            max_position_embeddings: 4096,
            ..Self::default()
        }
    }

    /// LLaMA-2-chat 7B — instruction-tuned via RLHF
    pub fn llama2_7b_chat() -> Self {
        Self::llama2_7b()
    }

    /// LLaMA-2-chat 13B — instruction-tuned via RLHF
    pub fn llama2_13b_chat() -> Self {
        Self::llama2_13b()
    }

    /// LLaMA-2-chat 70B — instruction-tuned via RLHF
    pub fn llama2_70b_chat() -> Self {
        Self::llama2_70b()
    }

    /// Create configuration by HuggingFace model name
    pub fn from_pretrained_name(name: &str) -> Option<Self> {
        match name {
            "meta-llama/Llama-2-7b-hf" | "llama2-7b" => Some(Self::llama2_7b()),
            "meta-llama/Llama-2-13b-hf" | "llama2-13b" => Some(Self::llama2_13b()),
            "meta-llama/Llama-2-70b-hf" | "llama2-70b" => Some(Self::llama2_70b()),
            "meta-llama/Llama-2-7b-chat-hf" | "llama2-7b-chat" => Some(Self::llama2_7b_chat()),
            "meta-llama/Llama-2-13b-chat-hf" | "llama2-13b-chat" => Some(Self::llama2_13b_chat()),
            "meta-llama/Llama-2-70b-chat-hf" | "llama2-70b-chat" => Some(Self::llama2_70b_chat()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama2_default_hidden_size() {
        let cfg = LLaMA2Config::default();
        assert_eq!(cfg.hidden_size, 4096);
    }

    #[test]
    fn test_llama2_default_intermediate_size() {
        let cfg = LLaMA2Config::default();
        assert_eq!(cfg.intermediate_size, 11008);
    }

    #[test]
    fn test_llama2_default_num_hidden_layers() {
        let cfg = LLaMA2Config::default();
        assert_eq!(cfg.num_hidden_layers, 32);
    }

    #[test]
    fn test_llama2_default_num_attention_heads() {
        let cfg = LLaMA2Config::default();
        assert_eq!(cfg.num_attention_heads, 32);
    }

    #[test]
    fn test_llama2_default_num_key_value_heads() {
        let cfg = LLaMA2Config::default();
        assert_eq!(cfg.num_key_value_heads, 32);
    }

    #[test]
    fn test_llama2_default_vocab_size() {
        let cfg = LLaMA2Config::default();
        assert_eq!(cfg.vocab_size, 32000);
    }

    #[test]
    fn test_llama2_default_max_position_embeddings() {
        let cfg = LLaMA2Config::default();
        assert_eq!(cfg.max_position_embeddings, 4096);
    }

    #[test]
    fn test_llama2_default_hidden_act() {
        let cfg = LLaMA2Config::default();
        assert_eq!(cfg.hidden_act, "silu");
    }

    #[test]
    fn test_llama2_default_use_cache() {
        let cfg = LLaMA2Config::default();
        assert!(cfg.use_cache);
    }

    #[test]
    fn test_llama2_default_pad_token_id_none() {
        let cfg = LLaMA2Config::default();
        assert!(cfg.pad_token_id.is_none());
    }

    #[test]
    fn test_llama2_validate_passes_default() {
        let cfg = LLaMA2Config::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_llama2_validate_fails_zero_vocab_size() {
        let cfg = LLaMA2Config {
            vocab_size: 0,
            ..LLaMA2Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_llama2_validate_fails_zero_num_hidden_layers() {
        let cfg = LLaMA2Config {
            num_hidden_layers: 0,
            ..LLaMA2Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_llama2_validate_fails_hidden_not_divisible_by_heads() {
        let cfg = LLaMA2Config {
            hidden_size: 4096,
            num_attention_heads: 7,
            num_key_value_heads: 7,
            ..LLaMA2Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_llama2_validate_fails_heads_not_divisible_by_kv_heads() {
        let cfg = LLaMA2Config {
            num_attention_heads: 32,
            num_key_value_heads: 7,
            ..LLaMA2Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_llama2_validate_fails_zero_pretraining_tp() {
        let cfg = LLaMA2Config {
            pretraining_tp: 0,
            ..LLaMA2Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_llama2_7b_preset() {
        let cfg = LLaMA2Config::llama2_7b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 32);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_llama2_13b_preset() {
        let cfg = LLaMA2Config::llama2_13b();
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.num_attention_heads, 40);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_llama2_70b_preset_gqa() {
        let cfg = LLaMA2Config::llama2_70b();
        assert_eq!(cfg.hidden_size, 8192);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert!(cfg.uses_gqa());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_llama2_head_dim() {
        let cfg = LLaMA2Config::llama2_7b();
        assert_eq!(cfg.head_dim(), 4096 / 32);
    }

    #[test]
    fn test_llama2_num_query_groups_no_gqa() {
        let cfg = LLaMA2Config::llama2_7b();
        assert_eq!(cfg.num_query_groups(), 1);
    }

    #[test]
    fn test_llama2_num_query_groups_with_gqa() {
        let cfg = LLaMA2Config::llama2_70b();
        assert_eq!(cfg.num_query_groups(), 64 / 8);
    }

    #[test]
    fn test_llama2_uses_gqa_false_for_7b() {
        let cfg = LLaMA2Config::llama2_7b();
        assert!(!cfg.uses_gqa());
    }

    #[test]
    fn test_llama2_from_pretrained_name_7b() {
        let cfg = LLaMA2Config::from_pretrained_name("llama2-7b");
        assert!(cfg.is_some());
        let c = cfg.unwrap_or_default();
        assert_eq!(c.hidden_size, 4096);
    }

    #[test]
    fn test_llama2_from_pretrained_name_unknown() {
        let cfg = LLaMA2Config::from_pretrained_name("unknown-model");
        assert!(cfg.is_none());
    }

    #[test]
    fn test_llama2_chat_variant_same_arch_as_base() {
        let base = LLaMA2Config::llama2_7b();
        let chat = LLaMA2Config::llama2_7b_chat();
        assert_eq!(base.hidden_size, chat.hidden_size);
        assert_eq!(base.num_hidden_layers, chat.num_hidden_layers);
    }

    #[test]
    fn test_llama2_bos_eos_token_ids() {
        let cfg = LLaMA2Config::default();
        assert_eq!(cfg.bos_token_id, 1);
        assert_eq!(cfg.eos_token_id, 2);
    }

    #[test]
    fn test_llama2_architecture_name() {
        let cfg = LLaMA2Config::default();
        assert_eq!(cfg.architecture(), "LLaMA-2");
    }

    #[test]
    fn test_llama2_lcg_derived_values_in_range() {
        let mut s = 42u64;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (s % 1000) as f32 / 1000.0;
        assert!((0.0..1.0).contains(&v));
    }

    #[test]
    fn test_llama2_pretraining_tp_default() {
        let cfg = LLaMA2Config::default();
        assert_eq!(cfg.pretraining_tp, 1);
    }

    #[test]
    fn test_llama2_attention_and_mlp_bias_default_false() {
        let cfg = LLaMA2Config::default();
        assert!(!cfg.attention_bias);
        assert!(!cfg.mlp_bias);
    }
}
