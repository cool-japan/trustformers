//! # LLaMA-3
//!
//! Meta LLaMA-3 (Dubey et al., 2024) is the third generation of Meta's
//! open-weight foundation model series.  Architectural highlights:
//!
//! * **Extended Tiktoken vocabulary**: 128 256 tokens (vs. 32 000 for LLaMA-2).
//! * **Higher RoPE base** (θ = 500 000): enables longer effective context.
//! * **GQA on all sizes**: 8B (4× sharing) and 70B (8× sharing).
//! * **SwiGLU FFN**: identical gating to LLaMA-2.
//! * **No bias** in any linear layer.
//! * **8192-token** default context window.
//!
//! ## Model variants
//!
//! | Variant | Layers | Hidden | Q-heads | KV-heads | GQA factor |
//! |---------|--------|--------|---------|----------|------------|
//! | 8B      | 32     | 4096   | 32      | 8        | 4×         |
//! | 70B     | 80     | 8192   | 64      | 8        | 8×         |
//!
//! ## Usage
//!
//! ```rust,no_run
//! use trustformers_models::llama3::{LLaMA3Config, LLaMA3ForCausalLM};
//!
//! let config = LLaMA3Config::llama3_8b();
//! let model = LLaMA3ForCausalLM::new(config)?;
//! let logits = model.forward(vec![1u32, 2, 3])?;
//! # Ok::<(), trustformers_core::errors::TrustformersError>(())
//! ```

pub mod config;
pub mod model;
pub mod tasks;

#[cfg(test)]
mod tests;

pub use config::LLaMA3Config;
pub use model::{
    LLaMA3Attention, LLaMA3DecoderLayer, LLaMA3ForCausalLM, LLaMA3MLP, LLaMA3Model, LLaMA3RmsNorm,
    LLaMA3RotaryEmbedding,
};
pub use tasks::{format_llama3_chat, LLaMA3CausalLMOutput, LLaMA3ChatModel};

// ── Additional tests beyond tests.rs ─────────────────────────────────────────

#[cfg(test)]
mod extra_tests {
    use super::*;
    use trustformers_core::traits::Config;

    // ── Config preset field verification ─────────────────────────────────────

    #[test]
    fn test_default_equals_llama3_8b() {
        let default = LLaMA3Config::default();
        let preset = LLaMA3Config::llama3_8b();
        assert_eq!(default.hidden_size, preset.hidden_size);
        assert_eq!(default.vocab_size, preset.vocab_size);
        assert_eq!(default.rope_theta, preset.rope_theta);
    }

    #[test]
    fn test_vocab_size_128256_for_8b() {
        let cfg = LLaMA3Config::llama3_8b();
        assert_eq!(
            cfg.vocab_size, 128256,
            "LLaMA-3 Tiktoken vocabulary must be 128256"
        );
    }

    #[test]
    fn test_vocab_size_128256_for_70b() {
        let cfg = LLaMA3Config::llama3_70b();
        assert_eq!(
            cfg.vocab_size, 128256,
            "LLaMA-3 70B must share the same Tiktoken vocabulary"
        );
    }

    #[test]
    fn test_rope_theta_500000_for_8b() {
        let cfg = LLaMA3Config::llama3_8b();
        assert!(
            (cfg.rope_theta - 500000.0).abs() < 1.0,
            "LLaMA-3 rope_theta must be 500000 for extended context"
        );
    }

    #[test]
    fn test_rope_theta_500000_for_70b() {
        let cfg = LLaMA3Config::llama3_70b();
        assert!((cfg.rope_theta - 500000.0).abs() < 1.0);
    }

    #[test]
    fn test_head_dim_8b() {
        let cfg = LLaMA3Config::llama3_8b();
        // hidden_size=4096, num_attention_heads=32 → head_dim=128
        assert_eq!(cfg.head_dim(), 128);
    }

    #[test]
    fn test_head_dim_70b() {
        let cfg = LLaMA3Config::llama3_70b();
        // hidden_size=8192, num_attention_heads=64 → head_dim=128
        assert_eq!(cfg.head_dim(), 128);
    }

    #[test]
    fn test_gqa_factor_8b() {
        let cfg = LLaMA3Config::llama3_8b();
        // 32 Q heads / 8 KV heads = 4× sharing
        assert_eq!(cfg.num_query_groups(), 4);
        assert!(cfg.uses_gqa());
    }

    #[test]
    fn test_gqa_factor_70b() {
        let cfg = LLaMA3Config::llama3_70b();
        // 64 Q heads / 8 KV heads = 8× sharing
        assert_eq!(cfg.num_query_groups(), 8);
        assert!(cfg.uses_gqa());
    }

    #[test]
    fn test_intermediate_size_8b() {
        let cfg = LLaMA3Config::llama3_8b();
        assert_eq!(cfg.intermediate_size, 14336);
    }

    #[test]
    fn test_intermediate_size_70b() {
        let cfg = LLaMA3Config::llama3_70b();
        assert_eq!(cfg.intermediate_size, 28672);
    }

    #[test]
    fn test_max_position_embeddings_8b() {
        let cfg = LLaMA3Config::llama3_8b();
        assert_eq!(cfg.max_position_embeddings, 8192);
    }

    #[test]
    fn test_uses_gqa_for_small_test() {
        let cfg = LLaMA3Config::small_test();
        // 4 Q heads, 2 KV heads → GQA
        assert!(cfg.uses_gqa());
        assert_eq!(cfg.num_query_groups(), 2);
    }

    // ── Validation error paths ────────────────────────────────────────────────

    #[test]
    fn test_validate_zero_hidden_layers_fails() {
        let mut cfg = LLaMA3Config::small_test();
        cfg.num_hidden_layers = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_intermediate_size_fails() {
        let mut cfg = LLaMA3Config::small_test();
        cfg.intermediate_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_vocab_size_fails() {
        let mut cfg = LLaMA3Config::small_test();
        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_kv_heads_not_divisor_of_q_heads() {
        let mut cfg = LLaMA3Config::small_test();
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 3; // 4 % 3 != 0
        assert!(cfg.validate().is_err());
    }

    // ── rms_norm_eps ──────────────────────────────────────────────────────────

    #[test]
    fn test_rms_norm_eps_value() {
        let cfg = LLaMA3Config::llama3_8b();
        assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-10);
    }

    // ── Architecture label ────────────────────────────────────────────────────

    #[test]
    fn test_architecture_label() {
        assert_eq!(LLaMA3Config::llama3_8b().architecture(), "LLaMA-3");
        assert_eq!(LLaMA3Config::llama3_70b().architecture(), "LLaMA-3");
    }

    // ── Config clone and serialization ────────────────────────────────────────

    #[test]
    fn test_config_clone() {
        let cfg = LLaMA3Config::llama3_8b();
        let cloned = cfg.clone();
        assert_eq!(cfg.vocab_size, cloned.vocab_size);
        assert_eq!(cfg.rope_theta, cloned.rope_theta);
    }

    #[test]
    fn test_config_json_round_trip() {
        let cfg = LLaMA3Config::llama3_70b();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let restored: LLaMA3Config = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg.hidden_size, restored.hidden_size);
        assert_eq!(cfg.vocab_size, restored.vocab_size);
        assert!((cfg.rope_theta - restored.rope_theta).abs() < 1e-6);
    }

    // ── LLaMA3ChatModel task wrapper ──────────────────────────────────────────

    #[test]
    fn test_llama3_chat_model_construction() {
        let model = LLaMA3ChatModel::new(LLaMA3Config::small_test());
        assert!(model.is_ok(), "LLaMA3ChatModel must construct successfully");
    }

    #[test]
    fn test_llama3_chat_model_parameter_count() {
        let model = LLaMA3ChatModel::new(LLaMA3Config::small_test()).expect("construction");
        assert!(
            model.parameter_count() > 0,
            "chat model must have parameters"
        );
    }

    #[test]
    fn test_llama3_chat_model_config_accessor() {
        let model = LLaMA3ChatModel::new(LLaMA3Config::small_test()).expect("construction");
        assert_eq!(
            model.config().vocab_size,
            LLaMA3Config::small_test().vocab_size
        );
    }

    #[test]
    fn test_llama3_chat_model_greedy_next_token() {
        use trustformers_core::tensor::Tensor;
        let model = LLaMA3ChatModel::new(LLaMA3Config::small_test()).expect("construction");
        // Construct a simple logits tensor — position 2 has the max value
        let data = vec![0.1f32, 0.2f32, 0.9f32, 0.3f32];
        let logits = Tensor::from_vec(data, &[4]).expect("tensor");
        let next = model.greedy_next_token(&logits).expect("greedy");
        assert_eq!(next, 2, "greedy must pick argmax index=2");
    }

    // ── format_llama3_chat edge cases ─────────────────────────────────────────

    #[test]
    fn test_format_llama3_chat_empty_system_skips_system_block() {
        let messages = vec![("user".to_string(), "Hello!".to_string())];
        let prompt = format_llama3_chat("", &messages);
        // No system block should appear
        assert!(
            !prompt.contains("<|start_header_id|>system<|end_header_id|>"),
            "empty system string must not emit a system block"
        );
        assert!(prompt.contains("<|begin_of_text|>"));
        assert!(prompt.contains("Hello!"));
    }

    #[test]
    fn test_format_llama3_chat_multi_turn_ordering() {
        let messages = vec![
            ("user".to_string(), "First".to_string()),
            ("assistant".to_string(), "Second".to_string()),
            ("user".to_string(), "Third".to_string()),
        ];
        let prompt = format_llama3_chat("sys", &messages);
        // All three messages must appear in document order
        let pos_first = prompt.find("First").expect("First missing");
        let pos_second = prompt.find("Second").expect("Second missing");
        let pos_third = prompt.find("Third").expect("Third missing");
        assert!(
            pos_first < pos_second && pos_second < pos_third,
            "messages must appear in order"
        );
    }

    #[test]
    fn test_format_llama3_chat_ends_with_open_assistant_turn() {
        let messages: Vec<(String, String)> = vec![];
        let prompt = format_llama3_chat("You are helpful.", &messages);
        // The prompt must end with an open assistant header for the model to continue
        assert!(
            prompt.contains("<|start_header_id|>assistant<|end_header_id|>"),
            "prompt must end with an open assistant turn"
        );
    }
}
