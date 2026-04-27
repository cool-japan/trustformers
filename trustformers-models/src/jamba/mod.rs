//! AI21 Jamba — hybrid Mamba-SSM + Transformer language model.
//!
//! Jamba is the first production-grade hybrid SSM/Attention architecture.
//! It interleaves Mamba state-space model (SSM) blocks with Transformer
//! attention blocks (using GQA), and uses Mixture of Experts (MoE) in
//! the attention-layer FFNs.
//!
//! ## Architecture highlights
//! - SSM (Mamba) layers handle the majority of positions efficiently
//! - Sparse Attention layers capture long-range dependencies
//! - MoE FFN layers expand model capacity at low per-token cost
//!
//! Reference: "Jamba: A Hybrid Transformer-Mamba Language Model" (AI21, 2024)

pub mod config;
pub mod model;
pub mod tasks;

pub use config::*;
pub use model::*;
pub use tasks::*;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn small() -> JambaConfig {
        JambaConfig::small_test()
    }

    fn full_1_5b() -> JambaConfig {
        JambaConfig::jamba_1_5b()
    }

    // ── 1. Config defaults (small_test) ─────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = small();
        assert_eq!(cfg.vocab_size, 256);
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.intermediate_size, 128);
        assert_eq!(cfg.num_hidden_layers, 8);
        assert_eq!(cfg.num_attention_heads, 4);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.attn_layer_offset, 3);
        assert_eq!(cfg.attn_layer_period, 8);
        assert_eq!(cfg.expert_layer_offset, 1);
        assert_eq!(cfg.expert_layer_period, 2);
        assert_eq!(cfg.num_experts, 4);
        assert_eq!(cfg.num_experts_per_tok, 2);
        assert_eq!(cfg.mamba_d_state, 8);
        assert_eq!(cfg.mamba_d_conv, 4);
        assert_eq!(cfg.mamba_expand, 2);
    }

    // ── 2. Jamba 1.5B preset ─────────────────────────────────────────────────

    #[test]
    fn test_jamba_1_5b() {
        let cfg = full_1_5b();
        assert_eq!(cfg.vocab_size, 65536);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.intermediate_size, 14336);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.num_experts, 16);
        assert_eq!(cfg.num_experts_per_tok, 2);
        assert_eq!(cfg.attn_layer_offset, 3);
        assert_eq!(cfg.attn_layer_period, 8);
        assert_eq!(cfg.rope_theta, 10000.0);
    }

    // ── 3. Hybrid layer pattern — SSM vs Attention alternation ───────────────

    #[test]
    fn test_hybrid_layer_pattern() {
        let cfg = full_1_5b();
        // attn_layer_offset=3, period=8 → attention at 3, 11, 19, 27
        assert!(!cfg.is_attention_layer(0), "layer 0 must be Mamba");
        assert!(!cfg.is_attention_layer(1), "layer 1 must be Mamba");
        assert!(!cfg.is_attention_layer(2), "layer 2 must be Mamba");
        assert!(cfg.is_attention_layer(3), "layer 3 must be Attention");
        assert!(!cfg.is_attention_layer(4), "layer 4 must be Mamba");
        assert!(!cfg.is_attention_layer(10), "layer 10 must be Mamba");
        assert!(cfg.is_attention_layer(11), "layer 11 must be Attention");
        assert!(cfg.is_attention_layer(19), "layer 19 must be Attention");
        assert!(cfg.is_attention_layer(27), "layer 27 must be Attention");
        assert!(!cfg.is_attention_layer(31), "layer 31 must be Mamba");
    }

    // ── 4. MoE expert count ───────────────────────────────────────────────────

    #[test]
    fn test_moe_expert_count() {
        let cfg = full_1_5b();
        assert_eq!(cfg.num_experts, 16, "Jamba 1.5B has 16 experts");
        assert_eq!(
            cfg.num_experts_per_tok, 2,
            "Jamba activates 2 experts per token"
        );
        assert!(
            cfg.num_experts_per_tok <= cfg.num_experts,
            "experts_per_tok must not exceed num_experts"
        );
    }

    // ── 5. Number of SSM layers ───────────────────────────────────────────────

    #[test]
    fn test_num_ssm_layers() {
        let cfg = full_1_5b();
        let ssm_count = (0..cfg.num_hidden_layers).filter(|&i| !cfg.is_attention_layer(i)).count();
        // 32 total - 4 attention (3,11,19,27) = 28 SSM layers
        assert_eq!(ssm_count, 28, "Jamba 1.5B should have 28 SSM layers");
    }

    // ── 6. SSM state dimension ────────────────────────────────────────────────

    #[test]
    fn test_ssm_state_dim() {
        let cfg_small = small();
        let cfg_full = full_1_5b();
        assert_eq!(cfg_small.mamba_d_state, 8);
        assert_eq!(cfg_full.mamba_d_state, 16);
        // State dim should always be > 0
        assert!(cfg_small.mamba_d_state > 0);
        assert!(cfg_full.mamba_d_state > 0);
    }

    // ── 7. Attention layer count ──────────────────────────────────────────────

    #[test]
    fn test_attention_layer_count() {
        let cfg = full_1_5b();
        let attn_count = (0..cfg.num_hidden_layers).filter(|&i| cfg.is_attention_layer(i)).count();
        // Attention at layers 3, 11, 19, 27 → 4 attention layers
        assert_eq!(attn_count, 4, "Jamba 1.5B should have 4 attention layers");
    }

    // ── 8. Head dimension ─────────────────────────────────────────────────────

    #[test]
    fn test_head_dim() {
        let cfg = full_1_5b();
        // head_dim = hidden_size / num_attention_heads = 4096 / 32 = 128
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.head_dim() * cfg.num_attention_heads, cfg.hidden_size);
    }

    // ── 9. Vocab size ─────────────────────────────────────────────────────────

    #[test]
    fn test_vocab_size() {
        let cfg_full = full_1_5b();
        assert_eq!(cfg_full.vocab_size, 65536);
        let cfg_small = small();
        assert_eq!(cfg_small.vocab_size, 256);
    }

    // ── 10. Hidden size ───────────────────────────────────────────────────────

    #[test]
    fn test_hidden_size() {
        let cfg = full_1_5b();
        assert_eq!(cfg.hidden_size, 4096);
        // Mamba inner dim should be hidden_size * mamba_expand
        assert_eq!(cfg.mamba_inner_dim(), 4096 * 2);
    }

    // ── 11. Forward output shape ──────────────────────────────────────────────

    #[test]
    fn test_forward_output_shape() {
        let cfg = small();
        let head = JambaCausalLMHead::new(cfg.clone()).expect("construct model");
        let input_ids = vec![0usize, 1, 2, 3];
        let output = head.forward(&input_ids).expect("forward should succeed");
        assert_eq!(
            output.logits.len(),
            4,
            "logits must have one row per input token"
        );
        assert_eq!(
            output.logits[0].len(),
            cfg.vocab_size,
            "each logit row must span the full vocab"
        );
    }

    // ── 12. LM head vocabulary dimension ─────────────────────────────────────

    #[test]
    fn test_lm_head_vocab_dim() {
        let cfg = small();
        let head = JambaCausalLMHead::new(cfg.clone()).expect("construct model");
        // Run forward on a single token
        let output = head.forward(&[0usize]).expect("forward ok");
        assert_eq!(
            output.logits[0].len(),
            cfg.vocab_size,
            "LM head output dim must equal vocab_size"
        );
    }

    // ── 13. Error display ─────────────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e1 = JambaError::EmptyInput;
        assert!(
            e1.to_string().contains("Empty"),
            "EmptyInput message should mention 'Empty'"
        );

        let e2 = JambaError::LayerError {
            layer: 5,
            msg: "dimension mismatch".to_string(),
        };
        let s2 = e2.to_string();
        assert!(s2.contains('5'), "LayerError should contain layer index");
        assert!(
            s2.contains("dimension"),
            "LayerError should contain message text"
        );

        // Verify std::error::Error trait is implemented
        let _boxed: Box<dyn std::error::Error> = Box::new(JambaError::EmptyInput);
    }

    // ── 14. GQA: num_key_value_heads < num_attention_heads ───────────────────

    #[test]
    fn test_gqa_kv_head_ratio() {
        let cfg = full_1_5b();
        assert!(
            cfg.num_key_value_heads < cfg.num_attention_heads,
            "GQA requires fewer KV heads than Q heads"
        );
        // Groups per KV head must divide evenly
        assert_eq!(
            cfg.num_attention_heads % cfg.num_key_value_heads,
            0,
            "num_attention_heads must be divisible by num_key_value_heads"
        );
    }

    // ── 15. MoE layers are a strict subset of attention layers ───────────────

    #[test]
    fn test_moe_layers_subset_of_attention() {
        let cfg = full_1_5b();
        for i in 0..cfg.num_hidden_layers {
            if cfg.is_moe_layer(i) {
                assert!(
                    cfg.is_attention_layer(i),
                    "layer {} is MoE but not attention — impossible",
                    i
                );
            }
        }
    }

    // ── 16. Greedy generation returns correct token count ────────────────────

    #[test]
    fn test_greedy_generation_token_count() {
        let cfg = small();
        let head = JambaCausalLMHead::new(cfg).expect("construct model");
        let input = vec![0usize, 1];
        let generated = head.generate_greedy(&input, 4).expect("generation should succeed");
        assert_eq!(generated.len(), 4, "should produce exactly 4 new tokens");
    }

    // ── 17. Mamba inner dim consistency ──────────────────────────────────────

    #[test]
    fn test_mamba_inner_dim() {
        let cfg_small = small();
        let cfg_full = full_1_5b();
        assert_eq!(
            cfg_small.mamba_inner_dim(),
            cfg_small.hidden_size * cfg_small.mamba_expand
        );
        assert_eq!(
            cfg_full.mamba_inner_dim(),
            cfg_full.hidden_size * cfg_full.mamba_expand
        );
    }

    // ── 18. Layer type assignment in constructed model ────────────────────────

    #[test]
    fn test_model_layer_type_assignment() {
        let cfg = small();
        let head = JambaCausalLMHead::new(cfg.clone()).expect("construct");
        let layers = head.model().layers();
        assert_eq!(layers.len(), cfg.num_hidden_layers);
        for (i, layer) in layers.iter().enumerate() {
            if cfg.is_attention_layer(i) {
                assert!(layer.is_attention(), "layer {} should be attention", i);
            } else {
                assert!(layer.is_mamba(), "layer {} should be mamba", i);
            }
        }
    }

    // ── 19. MoE layer detection in constructed model ──────────────────────────

    #[test]
    fn test_model_moe_layer_detection() {
        let cfg = small();
        let head = JambaCausalLMHead::new(cfg.clone()).expect("construct");
        let layers = head.model().layers();
        for (i, layer) in layers.iter().enumerate() {
            if cfg.is_moe_layer(i) {
                assert!(layer.is_moe(), "layer {} should be MoE", i);
            }
        }
    }

    // ── 20. SSM + Attention layer counts sum to total ─────────────────────────

    #[test]
    fn test_layer_counts_sum_to_total() {
        let cfg = small();
        let ssm = count_ssm_layers(&cfg);
        let attn = count_attention_layers(&cfg);
        assert_eq!(
            ssm + attn,
            cfg.num_hidden_layers,
            "SSM + attention must equal total layers"
        );
    }
}
