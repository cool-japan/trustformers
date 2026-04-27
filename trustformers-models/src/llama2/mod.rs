//! # LLaMA-2
//!
//! LLaMA-2 is Meta AI's second generation of the LLaMA foundation model family
//! (Touvron et al., 2023).  Compared to LLaMA-1 it introduces:
//!
//! * **Grouped Query Attention (GQA)**: The 70B model uses 8 KV heads shared
//!   among 64 query heads, reducing memory bandwidth by up to 8×.
//! * **Extended context**: 4096 tokens (doubled from LLaMA-1's 2048).
//! * **RLHF-trained chat variants**: LLaMA-2-chat models are instruction-tuned
//!   via Reinforcement Learning from Human Feedback.
//!
//! ## Model variants
//!
//! | Variant     | Layers | Hidden | Q-heads | KV-heads | GQA |
//! |-------------|--------|--------|---------|----------|-----|
//! | 7B          | 32     | 4096   | 32      | 32       | No  |
//! | 13B         | 40     | 5120   | 40      | 40       | No  |
//! | 70B         | 80     | 8192   | 64      | 8        | Yes |
//!
//! ## GQA repeat-interleave
//!
//! ```text
//! K_expanded = K.repeat_interleave(num_heads / num_kv_heads, dim=kv_head_dim)
//! V_expanded = V.repeat_interleave(num_heads / num_kv_heads, dim=kv_head_dim)
//! attn = softmax(Q * K_expanded^T / sqrt(head_dim)) * V_expanded
//! ```
//!
//! ## Usage
//!
//! ```rust,no_run
//! use trustformers_models::llama2::{LLaMA2Config, LLaMA2ForCausalLM};
//!
//! let config = LLaMA2Config::llama2_7b();
//! let model = LLaMA2ForCausalLM::new(config)?;
//! # Ok::<(), trustformers_core::errors::TrustformersError>(())
//! ```

pub mod config;
pub mod model;
pub mod tasks;

#[cfg(test)]
mod tests;

pub use config::LLaMA2Config;
pub use model::{
    LLaMA2Attention, LLaMA2DecoderLayer, LLaMA2ForCausalLM, LLaMA2MLP, LLaMA2Model, LLaMA2RMSNorm,
    LLaMA2RotaryEmbedding,
};
pub use tasks::{CausalLMOutput, LLaMA2ChatModel, LLaMA2TextGeneration};

// ── Additional tests beyond tests.rs ─────────────────────────────────────────

#[cfg(test)]
mod extra_tests {
    use super::*;
    use trustformers_core::traits::Config;

    fn mini_config() -> LLaMA2Config {
        LLaMA2Config {
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 8,
            vocab_size: 512,
            max_position_embeddings: 128,
            ..LLaMA2Config::default()
        }
    }

    // ── Architecture label ────────────────────────────────────────────────────

    #[test]
    fn test_llama2_architecture_label() {
        assert_eq!(mini_config().architecture(), "LLaMA-2");
    }

    // ── Preset head_dim values ────────────────────────────────────────────────

    #[test]
    fn test_head_dim_7b() {
        let cfg = LLaMA2Config::llama2_7b();
        // hidden_size=4096, num_attention_heads=32 → head_dim=128
        assert_eq!(cfg.head_dim(), 128);
    }

    #[test]
    fn test_head_dim_13b() {
        let cfg = LLaMA2Config::llama2_13b();
        // hidden_size=5120, num_attention_heads=40 → head_dim=128
        assert_eq!(cfg.head_dim(), 128);
    }

    #[test]
    fn test_head_dim_70b() {
        let cfg = LLaMA2Config::llama2_70b();
        // hidden_size=8192, num_attention_heads=64 → head_dim=128
        assert_eq!(cfg.head_dim(), 128);
    }

    // ── GQA head counts ───────────────────────────────────────────────────────

    #[test]
    fn test_gqa_num_query_groups_70b() {
        let cfg = LLaMA2Config::llama2_70b();
        // 64 Q heads / 8 KV heads = 8 groups
        assert_eq!(cfg.num_query_groups(), 8);
    }

    #[test]
    fn test_gqa_uses_gqa_7b_false() {
        let cfg = LLaMA2Config::llama2_7b();
        assert!(!cfg.uses_gqa(), "7B must not use GQA");
        assert_eq!(cfg.num_query_groups(), 1);
    }

    #[test]
    fn test_gqa_uses_gqa_70b_true() {
        let cfg = LLaMA2Config::llama2_70b();
        assert!(cfg.uses_gqa(), "70B must use GQA");
    }

    // ── Config field accessors ────────────────────────────────────────────────

    #[test]
    fn test_rope_theta_default() {
        let cfg = LLaMA2Config::default();
        assert!((cfg.rope_theta - 10000.0).abs() < 1e-3);
    }

    #[test]
    fn test_hidden_act_is_silu() {
        let cfg = LLaMA2Config::default();
        assert_eq!(cfg.hidden_act, "silu");
    }

    #[test]
    fn test_pretraining_tp_default_is_one() {
        let cfg = LLaMA2Config::default();
        assert_eq!(cfg.pretraining_tp, 1);
    }

    #[test]
    fn test_pretraining_tp_zero_invalid() {
        let mut cfg = mini_config();
        cfg.pretraining_tp = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_bos_eos_token_ids() {
        let cfg = LLaMA2Config::default();
        assert_eq!(cfg.bos_token_id, 1);
        assert_eq!(cfg.eos_token_id, 2);
    }

    #[test]
    fn test_pad_token_id_none_by_default() {
        let cfg = LLaMA2Config::default();
        assert!(cfg.pad_token_id.is_none());
    }

    #[test]
    fn test_use_cache_enabled_by_default() {
        let cfg = LLaMA2Config::default();
        assert!(cfg.use_cache);
    }

    // ── Chat and text generation task wrappers ────────────────────────────────

    #[test]
    fn test_llama2_text_generation_construction() {
        let task = LLaMA2TextGeneration::new(mini_config());
        assert!(
            task.is_ok(),
            "LLaMA2TextGeneration must construct successfully"
        );
    }

    #[test]
    fn test_llama2_chat_model_construction() {
        let task = LLaMA2ChatModel::new(mini_config());
        assert!(task.is_ok(), "LLaMA2ChatModel must construct successfully");
    }

    #[test]
    fn test_llama2_chat_model_uses_chat_template() {
        let model = LLaMA2ChatModel::new(mini_config()).expect("construction");
        assert!(
            model.uses_chat_template,
            "LLaMA2ChatModel must have uses_chat_template=true"
        );
    }

    #[test]
    fn test_llama2_chat_model_config_accessor() {
        let model = LLaMA2ChatModel::new(mini_config()).expect("construction");
        assert_eq!(model.config().hidden_size, 64);
        assert_eq!(model.config().vocab_size, 512);
    }

    #[test]
    fn test_llama2_text_generation_config_accessor() {
        let model = LLaMA2TextGeneration::new(mini_config()).expect("construction");
        assert_eq!(model.config().hidden_size, 64);
    }

    // ── from_pretrained_name lookups ──────────────────────────────────────────

    #[test]
    fn test_from_pretrained_name_7b() {
        let cfg = LLaMA2Config::from_pretrained_name("meta-llama/Llama-2-7b-hf");
        assert!(cfg.is_some());
        assert_eq!(cfg.expect("checked").hidden_size, 4096);
    }

    #[test]
    fn test_from_pretrained_name_70b_chat() {
        let cfg = LLaMA2Config::from_pretrained_name("llama2-70b-chat");
        assert!(cfg.is_some());
        assert!(cfg.expect("checked").uses_gqa());
    }

    #[test]
    fn test_from_pretrained_name_unknown() {
        let cfg = LLaMA2Config::from_pretrained_name("some-unknown-model");
        assert!(cfg.is_none());
    }

    // ── Chat variant presets ──────────────────────────────────────────────────

    #[test]
    fn test_llama2_7b_chat_same_arch_as_base() {
        let base = LLaMA2Config::llama2_7b();
        let chat = LLaMA2Config::llama2_7b_chat();
        assert_eq!(base.hidden_size, chat.hidden_size);
        assert_eq!(base.num_attention_heads, chat.num_attention_heads);
    }

    #[test]
    fn test_llama2_13b_chat_preset() {
        let cfg = LLaMA2Config::llama2_13b_chat();
        assert_eq!(cfg.hidden_size, 5120);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_llama2_70b_chat_preset() {
        let cfg = LLaMA2Config::llama2_70b_chat();
        assert!(cfg.uses_gqa());
        assert_eq!(cfg.num_key_value_heads, 8);
        assert!(cfg.validate().is_ok());
    }

    // ── Config clone and serialization ────────────────────────────────────────

    #[test]
    fn test_config_clone() {
        let cfg = mini_config();
        let cloned = cfg.clone();
        assert_eq!(cfg.hidden_size, cloned.hidden_size);
        assert_eq!(cfg.vocab_size, cloned.vocab_size);
        assert_eq!(cfg.hidden_act, cloned.hidden_act);
    }

    #[test]
    fn test_config_json_round_trip() {
        let cfg = LLaMA2Config::llama2_70b();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let restored: LLaMA2Config = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg.hidden_size, restored.hidden_size);
        assert_eq!(cfg.num_key_value_heads, restored.num_key_value_heads);
        assert_eq!(cfg.hidden_act, restored.hidden_act);
    }

    // ── Validation paths ──────────────────────────────────────────────────────

    #[test]
    fn test_validate_zero_vocab_size() {
        let mut cfg = mini_config();
        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_hidden_layers() {
        let mut cfg = mini_config();
        cfg.num_hidden_layers = 0;
        assert!(cfg.validate().is_err());
    }
}
