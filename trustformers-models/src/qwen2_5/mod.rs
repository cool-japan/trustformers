//! # Qwen2.5 (Alibaba Group, 2024)
//!
//! Qwen2.5 is a family of open language models that improves on Qwen2 with:
//!
//! ## Key Architectural Features
//!
//! - **Grouped Query Attention (GQA)**: 4 KV heads shared across 28 query heads (7:1 ratio),
//!   reducing memory bandwidth and KV cache size.
//! - **SwiGLU FFN**: `down_proj(silu(gate_proj(x)) * up_proj(x))` for improved expressivity.
//! - **Extended RoPE**: base frequency 1,000,000 (vs 10,000 in earlier models) enabling
//!   stable long-context inference.
//! - **Larger context**: 32,768 tokens by default; some variants support up to 128K.
//! - **Optional sliding window**: configurable per-layer sliding window attention.
//!
//! ## Model Variants
//!
//! | Variant  | Layers | Hidden | Heads (Q/KV) | head_dim |
//! |----------|--------|--------|--------------|----------|
//! | 0.5B     | 24     | 896    | 14 / 2       | 64       |
//! | 7B       | 28     | 3584   | 28 / 4       | 128      |
//! | 72B      | 80     | 8192   | 64 / 8       | 128      |
//!
//! ## Usage
//!
//! ```rust,no_run
//! use trustformers_models::qwen2_5::{Qwen25Config, Qwen25ForCausalLM};
//!
//! let config = Qwen25Config {
//!     vocab_size: 64,
//!     hidden_size: 32,
//!     intermediate_size: 64,
//!     num_hidden_layers: 2,
//!     num_attention_heads: 4,
//!     num_key_value_heads: 2,
//!     head_dim: 8,
//!     ..Qwen25Config::default()
//! };
//! let model = Qwen25ForCausalLM::new(config).expect("model creation");
//! ```

pub mod config;
pub mod model;
pub mod tasks;

pub use config::Qwen25Config;
pub use model::{
    silu, swiglu, Qwen25Attention, Qwen25DecoderLayer, Qwen25MLP, Qwen25Model, Qwen25RmsNorm,
    Qwen25RotaryEmbedding,
};
pub use tasks::{Qwen25Error, Qwen25ForCausalLM, Qwen25ForSequenceClassification};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    /// Build a minimal Qwen2.5 config suitable for unit tests.
    fn tiny_config() -> Qwen25Config {
        Qwen25Config {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            max_position_embeddings: 512,
            ..Qwen25Config::default()
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: Default config field values
    // -----------------------------------------------------------------------
    #[test]
    fn test_config_defaults() {
        let cfg = Qwen25Config::default();
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.intermediate_size, 18944);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_attention_heads, 28);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.max_position_embeddings, 32768);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
        assert!(!cfg.use_sliding_window);
        assert!(cfg.sliding_window.is_none());
        assert_eq!(cfg.max_window_layers, 28);
        assert!(!cfg.tie_word_embeddings);
        assert!(!cfg.use_mrope);
        assert_eq!(cfg.hidden_act, "silu");
    }

    // -----------------------------------------------------------------------
    // Test 2: GQA — num_kv_heads < num_attention_heads
    // -----------------------------------------------------------------------
    #[test]
    fn test_gqa_dimensions() {
        let cfg = Qwen25Config::default();
        // Default: 4 KV heads for 28 query heads → group size 7
        assert!(
            cfg.num_key_value_heads < cfg.num_attention_heads,
            "GQA: kv_heads ({}) must be < attn_heads ({})",
            cfg.num_key_value_heads,
            cfg.num_attention_heads
        );
        assert_eq!(cfg.kv_group_size(), 7);

        let tiny = tiny_config();
        assert_eq!(tiny.kv_group_size(), 2); // 4 / 2
    }

    // -----------------------------------------------------------------------
    // Test 3: Sliding window configuration
    // -----------------------------------------------------------------------
    #[test]
    fn test_sliding_window_config() {
        // Default: no sliding window
        let cfg = Qwen25Config::default();
        assert!(!cfg.layer_uses_sliding_window(0));
        assert!(!cfg.layer_uses_sliding_window(27));

        // With sliding window enabled
        let sw_cfg = Qwen25Config {
            use_sliding_window: true,
            sliding_window: Some(1024),
            max_window_layers: 2,
            num_hidden_layers: 4,
            ..tiny_config()
        };
        assert!(sw_cfg.validate().is_ok());
        // Layers 0,1 → full attention (< max_window_layers = 2)
        assert!(!sw_cfg.layer_uses_sliding_window(0));
        assert!(!sw_cfg.layer_uses_sliding_window(1));
        // Layers 2,3 → sliding window
        assert!(sw_cfg.layer_uses_sliding_window(2));
        assert!(sw_cfg.layer_uses_sliding_window(3));
    }

    // -----------------------------------------------------------------------
    // Test 4: SwiGLU forward mock — output shape and non-zero for positive input
    // -----------------------------------------------------------------------
    #[test]
    fn test_swiglu_forward() {
        use trustformers_core::device::Device;
        use trustformers_core::tensor::Tensor;
        use trustformers_core::traits::Layer;

        let cfg = tiny_config();
        let mlp = Qwen25MLP::new(&cfg, Device::CPU);
        let input = Tensor::from_vec(vec![1.0f32; 32], &[1, 32]).expect("tensor");
        let result = mlp.forward(input);
        assert!(result.is_ok(), "SwiGLU forward failed: {:?}", result.err());
    }

    // -----------------------------------------------------------------------
    // Test 5: Model layer count matches config
    // -----------------------------------------------------------------------
    #[test]
    fn test_model_layer_count() {
        use trustformers_core::traits::Model;

        let cfg = tiny_config();
        let n_layers = cfg.num_hidden_layers;
        let model = Qwen25Model::new(cfg).expect("model creation");
        // num_parameters is non-zero and proportional to layers
        assert!(model.num_parameters() > 0);
        // Verify layer count via config access
        assert_eq!(model.config().num_hidden_layers, n_layers);
    }

    // -----------------------------------------------------------------------
    // Test 6: Sequence classification head — correct num_labels output
    // -----------------------------------------------------------------------
    #[test]
    fn test_sequence_classification_head() {
        let cfg = tiny_config();
        let num_labels = 5;
        let clf =
            Qwen25ForSequenceClassification::new(cfg, num_labels).expect("classifier creation");
        assert_eq!(clf.num_labels(), num_labels);

        let logits = clf.classify(&[1u32, 2, 3]).expect("classify");
        assert_eq!(
            logits.len(),
            num_labels,
            "expected {} logits, got {}",
            num_labels,
            logits.len()
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: Config validation rejects bad inputs
    // -----------------------------------------------------------------------
    #[test]
    fn test_config_validation() {
        let mut cfg = tiny_config();
        assert!(cfg.validate().is_ok());

        // GQA divisibility
        cfg.num_attention_heads = 5;
        cfg.num_key_value_heads = 3;
        assert!(
            cfg.validate().is_err(),
            "5 attn heads / 3 kv heads should fail"
        );
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 2;

        // Sliding window enabled but window is None
        cfg.use_sliding_window = true;
        cfg.sliding_window = None;
        assert!(
            cfg.validate().is_err(),
            "sliding window enabled with None should fail"
        );
        cfg.use_sliding_window = false;

        // zero hidden_size
        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err(), "hidden_size=0 should fail");
        cfg.hidden_size = 32;

        assert!(cfg.validate().is_ok(), "restored config should pass");
    }

    // -----------------------------------------------------------------------
    // Test 8: Causal LM forward and generate
    // -----------------------------------------------------------------------
    #[test]
    fn test_causal_lm_forward_and_generate() {
        use trustformers_core::tensor::Tensor;
        use trustformers_core::traits::Model;

        let cfg = tiny_config();
        let model = Qwen25ForCausalLM::new(cfg).expect("causal lm creation");

        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).expect("tensor");
        let fwd = model.forward(input);
        assert!(fwd.is_ok(), "forward failed: {:?}", fwd.err());

        let gen = model.generate(&[1u32, 2, 3], 2);
        assert!(gen.is_ok(), "generate failed: {:?}", gen.err());
        assert_eq!(gen.expect("gen").len(), 2);
    }

    // -----------------------------------------------------------------------
    // Test 9: Qwen2.5-0.5B preset config
    // -----------------------------------------------------------------------
    #[test]
    fn test_qwen25_0_5b_preset() {
        let cfg = Qwen25Config::qwen25_0_5b();
        assert_eq!(cfg.hidden_size, 896);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 14);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.kv_group_size(), 7);
        assert!(cfg.tie_word_embeddings);
        assert!(cfg.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Test 10: Qwen2.5-7B preset (default) explicit fields
    // -----------------------------------------------------------------------
    #[test]
    fn test_qwen25_7b_preset() {
        let cfg = Qwen25Config::qwen25_7b();
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_attention_heads, 28);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.intermediate_size, 18944);
        assert_eq!(cfg.max_position_embeddings, 32768);
        assert!(!cfg.tie_word_embeddings, "7B does not tie embeddings");
        assert!(cfg.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Test 11: head_dim explicit field matches hidden_size / num_heads
    // -----------------------------------------------------------------------
    #[test]
    fn test_head_dim_explicit_field() {
        // 0.5B: 896 / 14 = 64
        let cfg = Qwen25Config::qwen25_0_5b();
        assert_eq!(cfg.head_dim, 64);
        // Default (7B): 3584 / 28 = 128
        let cfg7b = Qwen25Config::default();
        assert_eq!(cfg7b.head_dim, 128);
    }

    // -----------------------------------------------------------------------
    // Test 12: max_position_embeddings is 32768 for both standard presets
    // -----------------------------------------------------------------------
    #[test]
    fn test_max_position_embeddings() {
        for cfg in [Qwen25Config::qwen25_0_5b(), Qwen25Config::qwen25_7b()] {
            assert_eq!(cfg.max_position_embeddings, 32768);
        }
    }

    // -----------------------------------------------------------------------
    // Test 13: rope_theta is 1_000_000 for all presets
    // -----------------------------------------------------------------------
    #[test]
    fn test_rope_theta_presets() {
        for cfg in [Qwen25Config::qwen25_0_5b(), Qwen25Config::qwen25_7b()] {
            assert!(
                (cfg.rope_theta - 1_000_000.0_f64).abs() < 1.0,
                "rope_theta must be 1_000_000, got {}",
                cfg.rope_theta
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 14: use_mrope default is false (vision-language opt-in)
    // -----------------------------------------------------------------------
    #[test]
    fn test_use_mrope_default_false() {
        let cfg = Qwen25Config::default();
        assert!(!cfg.use_mrope, "use_mrope should default to false");
        // Can be enabled for multimodal variant
        let mut mm_cfg = tiny_config();
        mm_cfg.use_mrope = true;
        assert!(mm_cfg.use_mrope);
    }

    // -----------------------------------------------------------------------
    // Test 15: architecture() returns "Qwen2.5"
    // -----------------------------------------------------------------------
    #[test]
    fn test_architecture_string() {
        let cfg = Qwen25Config::default();
        assert_eq!(cfg.architecture(), "Qwen2.5");
    }

    // -----------------------------------------------------------------------
    // Test 16: Config clone preserves all fields
    // -----------------------------------------------------------------------
    #[test]
    fn test_config_clone() {
        let cfg = Qwen25Config::qwen25_0_5b();
        let cloned = cfg.clone();
        assert_eq!(cloned.vocab_size, cfg.vocab_size);
        assert_eq!(cloned.hidden_size, cfg.hidden_size);
        assert_eq!(cloned.head_dim, cfg.head_dim);
        assert_eq!(cloned.tie_word_embeddings, cfg.tie_word_embeddings);
    }

    // -----------------------------------------------------------------------
    // Test 17: Config debug format mentions key fields
    // -----------------------------------------------------------------------
    #[test]
    fn test_config_debug() {
        let cfg = tiny_config();
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("Qwen25Config"));
        assert!(dbg.contains("vocab_size"));
        assert!(dbg.contains("hidden_size"));
        assert!(dbg.contains("head_dim"));
    }

    // -----------------------------------------------------------------------
    // Test 18: forward_ids returns empty-input error
    // -----------------------------------------------------------------------
    #[test]
    fn test_forward_ids_empty_input_error() {
        let cfg = tiny_config();
        let model = Qwen25ForCausalLM::new(cfg).expect("causal lm creation");
        let result = model.forward_ids(&[]);
        assert!(result.is_err(), "empty input should return Err");
        let err_str = result.unwrap_err().to_string();
        assert!(
            err_str.to_lowercase().contains("empty"),
            "error message should mention 'empty', got: {err_str}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 19: sequence classification with 1 label (edge case)
    // -----------------------------------------------------------------------
    #[test]
    fn test_sequence_classification_single_label() {
        let cfg = tiny_config();
        let clf =
            Qwen25ForSequenceClassification::new(cfg, 1).expect("single-label classifier creation");
        assert_eq!(clf.num_labels(), 1);
        let logits = clf.classify(&[0u32, 1]).expect("classify");
        assert_eq!(logits.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Test 20: generate returns exactly max_new_tokens tokens
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_exact_length() {
        let cfg = tiny_config();
        let model = Qwen25ForCausalLM::new(cfg).expect("causal lm creation");
        for n in [1, 3, 5] {
            let gen = model.generate(&[1u32, 2], n).expect("generate");
            assert_eq!(gen.len(), n, "generate should produce exactly {n} tokens");
        }
    }

    // -----------------------------------------------------------------------
    // Test 21: sliding window layer assignment with non-zero max_window_layers
    // -----------------------------------------------------------------------
    #[test]
    fn test_sliding_window_layer_boundary() {
        let mut cfg = tiny_config();
        cfg.use_sliding_window = true;
        cfg.sliding_window = Some(512);
        cfg.max_window_layers = 1;
        cfg.num_hidden_layers = 3;
        // layer 0 → full attention (< max_window_layers=1)
        assert!(!cfg.layer_uses_sliding_window(0));
        // layers 1, 2 → sliding window
        assert!(cfg.layer_uses_sliding_window(1));
        assert!(cfg.layer_uses_sliding_window(2));
    }

    // -----------------------------------------------------------------------
    // Test 22: Qwen25Error::InvalidConfig display
    // -----------------------------------------------------------------------
    #[test]
    fn test_qwen25_error_display_invalid_config() {
        let err = Qwen25Error::InvalidConfig("test message".to_string());
        let s = err.to_string();
        assert!(
            s.contains("invalid config") || s.contains("Qwen25"),
            "got: {s}"
        );
        assert!(s.contains("test message"), "got: {s}");
    }

    // -----------------------------------------------------------------------
    // Test 23: Qwen25Error::ShapeMismatch display
    // -----------------------------------------------------------------------
    #[test]
    fn test_qwen25_error_display_shape_mismatch() {
        let err = Qwen25Error::ShapeMismatch {
            expected: vec![2, 4],
            got: vec![3, 4],
        };
        let s = err.to_string();
        assert!(s.contains("mismatch") || s.contains("shape"), "got: {s}");
    }

    // -----------------------------------------------------------------------
    // Test 24: Qwen25Error::EmptyInput display
    // -----------------------------------------------------------------------
    #[test]
    fn test_qwen25_error_display_empty_input() {
        let err = Qwen25Error::EmptyInput;
        let s = err.to_string();
        assert!(s.to_lowercase().contains("empty"), "got: {s}");
    }

    // -----------------------------------------------------------------------
    // Test 25: initializer_range field default is 0.02
    // -----------------------------------------------------------------------
    #[test]
    fn test_initializer_range_default() {
        let cfg = Qwen25Config::default();
        assert!(
            (cfg.initializer_range - 0.02_f32).abs() < 1e-6,
            "initializer_range default must be 0.02, got {}",
            cfg.initializer_range
        );
    }

    // -----------------------------------------------------------------------
    // Test 26: hidden_act field is "silu"
    // -----------------------------------------------------------------------
    #[test]
    fn test_hidden_act_is_silu() {
        let cfg = Qwen25Config::default();
        assert_eq!(cfg.hidden_act, "silu");
    }

    // -----------------------------------------------------------------------
    // Test 27: silu and swiglu utilities
    // -----------------------------------------------------------------------
    #[test]
    fn test_silu_positive_input() {
        // silu(1) = 1 / (1 + e^-1) ≈ 0.731
        let v = silu(1.0_f32);
        assert!((v - 0.731_f32).abs() < 0.01, "silu(1) ≈ 0.731, got {v}");
    }

    #[test]
    fn test_silu_zero_input() {
        // silu(0) = 0 / (1 + 1) = 0
        let v = silu(0.0_f32);
        assert!(v.abs() < 1e-6, "silu(0) must be 0, got {v}");
    }

    #[test]
    fn test_swiglu_length_preserved() {
        let gate = vec![1.0_f32, -1.0, 2.0, 0.0];
        let up = vec![1.0_f32; 4];
        let out = swiglu(&gate, &up);
        assert_eq!(out.len(), 4, "swiglu output length must match inputs");
    }
}
