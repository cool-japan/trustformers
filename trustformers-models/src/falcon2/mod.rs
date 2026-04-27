//! Falcon-2 model implementation.
//!
//! Falcon-2 (11B) from TII (Technology Innovation Institute).
//! Key features:
//! - Multi-Query Attention (MQA): single KV head shared across all Q heads
//! - ALiBi positional bias (no RoPE)
//! - Parallel attention + MLP (computed from same normed input, outputs summed)
//! - New GELU activation (tanh approximation)
//! - Large 60-layer architecture (default 11B parameters)

pub mod config;
pub mod model;
pub mod tasks;

pub use config::*;
pub use model::*;
pub use tasks::*;

// ─────────────────────────────────────────────────────────────────────────────
// Module-level tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn tiny() -> Falcon2Config {
        Falcon2Config {
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 1,
            intermediate_size: 128,
            max_position_embeddings: 512,
            vocab_size: 128,
            layer_norm_epsilon: 1e-5,
            use_alibi: true,
            parallel_attn: true,
            bias: false,
            hidden_act: "gelu".to_string(),
        }
    }

    // ── 1. Falcon-2 11B config ────────────────────────────────────────────────

    #[test]
    fn test_falcon2_11b_config() {
        let cfg = Falcon2Config::default();
        // Falcon-2 11B is a 60-layer, 4096-hidden model
        assert_eq!(cfg.hidden_size, 4096, "11B hidden size must be 4096");
        assert_eq!(cfg.num_hidden_layers, 60, "11B must have 60 layers");
        assert_eq!(
            cfg.num_attention_heads, 64,
            "11B must have 64 attention heads"
        );
        assert_eq!(
            cfg.intermediate_size, 16384,
            "11B intermediate size must be 16384"
        );
        assert_eq!(cfg.vocab_size, 65024, "11B vocab size must be 65024");
        assert_eq!(cfg.max_position_embeddings, 8192);
    }

    // ── 2. GQA head ratio — num_kv_heads < num_attention_heads ───────────────

    #[test]
    fn test_gqa_head_ratio() {
        let cfg = Falcon2Config::default();
        assert!(
            cfg.num_kv_heads < cfg.num_attention_heads,
            "Falcon-2 MQA: num_kv_heads={} must be < num_attention_heads={}",
            cfg.num_kv_heads,
            cfg.num_attention_heads
        );
        // Tiny config also holds
        let small = tiny();
        assert!(small.num_kv_heads < small.num_attention_heads);
    }

    // ── 3. Parallel attention flag ────────────────────────────────────────────

    #[test]
    fn test_parallel_attention() {
        let cfg = Falcon2Config::default();
        assert!(
            cfg.parallel_attn,
            "Falcon-2 uses parallel attn+MLP by default"
        );
        // Tiny config also uses parallel attn
        let small = tiny();
        assert!(small.parallel_attn);
    }

    // ── 4. ALiBi is enabled, not RoPE ─────────────────────────────────────────

    #[test]
    fn test_alibi_enabled() {
        let cfg = Falcon2Config::default();
        assert!(
            cfg.use_alibi,
            "Falcon-2 default should use ALiBi positional bias"
        );
    }

    // ── 5. Hidden size ────────────────────────────────────────────────────────

    #[test]
    fn test_hidden_size_falcon2() {
        let cfg = Falcon2Config::default();
        assert_eq!(cfg.hidden_size, 4096);
        // head_dim = hidden_size / num_heads = 4096 / 64 = 64
        assert_eq!(cfg.head_dim(), 64);
    }

    // ── 6. Num layers ─────────────────────────────────────────────────────────

    #[test]
    fn test_num_layers_falcon2() {
        let cfg = Falcon2Config::default();
        assert_eq!(cfg.num_hidden_layers, 60);
    }

    // ── 7. Forward basic — output shape ───────────────────────────────────────

    #[test]
    fn test_forward_basic() {
        let cfg = tiny();
        let model = Falcon2Model::new(cfg.clone());
        let input_ids = vec![1_u32, 2, 3];
        let output = model.forward(&input_ids).expect("forward should not fail");
        assert_eq!(
            output.len(),
            input_ids.len() * cfg.hidden_size,
            "output must be [seq_len * hidden_size]"
        );
    }

    // ── 8. CausalLM creation ──────────────────────────────────────────────────

    #[test]
    fn test_causal_lm_creation() {
        let cfg = tiny();
        let lm = Falcon2ForCausalLM::new(cfg.clone());
        // The lm head weight should have vocab_size * hidden_size elements
        assert_eq!(
            lm.lm_head_weight.len(),
            cfg.vocab_size * cfg.hidden_size,
            "lm_head_weight must span vocab × hidden"
        );
    }

    // ── 9. MQA: exactly 1 KV head by default ─────────────────────────────────

    #[test]
    fn test_mqa_exactly_one_kv_head() {
        let cfg = Falcon2Config::default();
        assert_eq!(
            cfg.num_kv_heads, 1,
            "Falcon-2 default uses a single KV head (MQA)"
        );
    }

    // ── 10. Bias defaults to false ────────────────────────────────────────────

    #[test]
    fn test_no_linear_bias_default() {
        let cfg = Falcon2Config::default();
        assert!(!cfg.bias, "Falcon-2 default should have no linear bias");
    }

    // ── 11. Hidden act is GELU ─────────────────────────────────────────────────

    #[test]
    fn test_hidden_act_gelu() {
        let cfg = Falcon2Config::default();
        assert_eq!(cfg.hidden_act, "gelu", "Falcon-2 uses GELU activation");
    }

    // ── 12. Head dimension consistency ────────────────────────────────────────

    #[test]
    fn test_head_dim_consistency() {
        let cfg = Falcon2Config::default();
        assert_eq!(
            cfg.head_dim() * cfg.num_attention_heads,
            cfg.hidden_size,
            "head_dim * num_heads must equal hidden_size"
        );
    }

    // ── 13. ALiBi bias shape for default config ───────────────────────────────

    #[test]
    fn test_alibi_bias_shape_default() {
        let num_heads = 4_usize;
        let seq_len = 8_usize;
        let slopes = Falcon2AlibiPositionalBias::compute_slopes(num_heads);
        let bias = Falcon2AlibiPositionalBias::compute_bias(seq_len, &slopes);
        assert_eq!(
            bias.len(),
            num_heads * seq_len * seq_len,
            "ALiBi bias tensor must be [heads, seq, seq]"
        );
    }

    // ── 14. ALiBi diagonal entries are zero ───────────────────────────────────

    #[test]
    fn test_alibi_diagonal_zero() {
        let num_heads = 2;
        let seq_len = 5;
        let slopes = Falcon2AlibiPositionalBias::compute_slopes(num_heads);
        let bias = Falcon2AlibiPositionalBias::compute_bias(seq_len, &slopes);
        for h in 0..num_heads {
            for i in 0..seq_len {
                let idx = h * seq_len * seq_len + i * seq_len + i;
                assert!(
                    bias[idx].abs() < 1e-7,
                    "diagonal bias[h={h},i={i}] should be 0, got {}",
                    bias[idx]
                );
            }
        }
    }

    // ── 15. Causal LM logits shape ────────────────────────────────────────────

    #[test]
    fn test_causal_lm_logits_shape() {
        let cfg = tiny();
        let lm = Falcon2ForCausalLM::new(cfg.clone());
        let input_ids = vec![0_u32, 1, 2, 3, 4];
        let logits = lm.forward(&input_ids).expect("forward should succeed");
        assert_eq!(
            logits.len(),
            input_ids.len() * cfg.vocab_size,
            "logits must be [seq_len * vocab_size]"
        );
    }

    // ── 16. Generate output length ────────────────────────────────────────────

    #[test]
    fn test_generate_output_length() {
        let cfg = tiny();
        let lm = Falcon2ForCausalLM::new(cfg.clone());
        let input_ids = vec![1_u32, 2];
        let new_toks = 4_usize;
        let generated = lm.generate(&input_ids, new_toks).expect("generate ok");
        assert_eq!(
            generated.len(),
            new_toks,
            "generate() must return exactly max_new_tokens tokens"
        );
    }

    // ── 17. Generated tokens are within vocab ─────────────────────────────────

    #[test]
    fn test_generated_tokens_in_vocab() {
        let cfg = tiny();
        let lm = Falcon2ForCausalLM::new(cfg.clone());
        let input_ids = vec![1_u32, 2, 3];
        let generated = lm.generate(&input_ids, 6).expect("generate ok");
        for &tok in &generated {
            assert!(
                (tok as usize) < cfg.vocab_size,
                "generated token {} must be within vocab_size {}",
                tok,
                cfg.vocab_size
            );
        }
    }

    // ── 18. Error on out-of-vocab token ───────────────────────────────────────

    #[test]
    fn test_out_of_vocab_token_error() {
        let cfg = tiny(); // vocab_size = 128
        let model = Falcon2Model::new(cfg);
        let result = model.forward(&[200_u32]); // OOV
        assert!(result.is_err(), "OOV token must yield an error");
    }

    // ── 19. Layer norm output has zero mean ───────────────────────────────────

    #[test]
    fn test_layer_norm_zero_mean() {
        let dim = 8;
        let x: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let weight = vec![1.0_f32; dim];
        let bias_v = vec![0.0_f32; dim];
        let out = Falcon2LayerNorm::forward(&x, &weight, &bias_v, 1e-5);
        let mean = out.iter().sum::<f32>() / dim as f32;
        assert!(
            mean.abs() < 1e-5,
            "LayerNorm output must have zero mean, got {mean}"
        );
    }

    // ── 20. Intermediate size is 4× hidden size ───────────────────────────────

    #[test]
    fn test_intermediate_size_ratio() {
        let cfg = Falcon2Config::default();
        assert_eq!(
            cfg.intermediate_size,
            4 * cfg.hidden_size,
            "Falcon-2 MLP intermediate size must be 4× hidden_size"
        );
    }
}
