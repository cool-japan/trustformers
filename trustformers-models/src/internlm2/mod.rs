//! InternLM-2 model implementation.
//!
//! InternLM-2 is an open-source large language model developed by Shanghai AI Lab.
//! Key features:
//! - Grouped Query Attention (GQA) for efficient inference
//! - RoPE with optional NTK dynamic scaling for extended context
//! - SwiGLU activation function
//! - Large vocabulary (92544 tokens)

pub mod config;
pub mod model;
pub mod tasks;

pub use config::*;
pub use model::*;
pub use tasks::*;

#[cfg(test)]
mod tests {
    use crate::internlm2::{
        config::InternLm2Config,
        model::{
            InternLm2Attention, InternLm2DecoderLayer, InternLm2Error, InternLm2MLP,
            InternLm2Model, InternLm2RmsNorm, InternLm2RotaryEmbedding,
        },
        tasks::InternLm2ForCausalLM,
    };

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn tiny_config() -> InternLm2Config {
        InternLm2Config::new(
            128,     // vocab_size
            64,      // hidden_size
            2,       // num_hidden_layers
            4,       // num_attention_heads
            2,       // num_key_value_heads  (GQA ratio = 2)
            128,     // intermediate_size
            512,     // max_position_embeddings
            10000.0, // rope_theta
            None,    // rope_scaling
            "silu",  // hidden_act
            1e-5,    // rms_norm_eps
            false,   // tie_word_embeddings
            true,    // use_cache
        )
    }

    // ── 1. Config default values ──────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = InternLm2Config::default();
        assert_eq!(cfg.vocab_size, 92544);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.intermediate_size, 14336);
        assert_eq!(cfg.max_position_embeddings, 32768);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
        assert!(cfg.rope_scaling.is_none());
        assert_eq!(cfg.hidden_act, "silu");
        assert!(!cfg.tie_word_embeddings);
        assert!(cfg.use_cache);
    }

    // ── 2. internlm2_7b preset ───────────────────────────────────────────────

    #[test]
    fn test_internlm2_7b_preset() {
        let cfg = InternLm2Config::internlm2_7b();
        // 7B uses the same parameters as default
        assert_eq!(cfg.vocab_size, 92544);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.gqa_ratio(), 4);
        assert_eq!(cfg.head_dim(), 128);
    }

    // ── 3. internlm2_20b preset ──────────────────────────────────────────────

    #[test]
    fn test_internlm2_20b_preset() {
        let cfg = InternLm2Config::internlm2_20b();
        assert_eq!(cfg.hidden_size, 6144);
        assert_eq!(cfg.num_hidden_layers, 48);
        assert_eq!(cfg.num_attention_heads, 48);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.intermediate_size, 16384);
        // GQA ratio should be 6 (48 / 8)
        assert_eq!(cfg.gqa_ratio(), 6);
        // head_dim = 6144 / 48 = 128
        assert_eq!(cfg.head_dim(), 128);
        // Vocab should be the same as 7B
        assert_eq!(cfg.vocab_size, 92544);
    }

    // ── 4. Config computed properties ────────────────────────────────────────

    #[test]
    fn test_config_computed_properties() {
        let cfg = tiny_config();
        // head_dim = 64 / 4 = 16
        assert_eq!(cfg.head_dim(), 16);
        // gqa_ratio = 4 / 2 = 2
        assert_eq!(cfg.gqa_ratio(), 2);
    }

    // ── 5. RopE basic rotation ───────────────────────────────────────────────

    #[test]
    fn test_rope_rotation_changes_values() {
        let rope = InternLm2RotaryEmbedding::new(10000.0, None);
        let head_dim = 8;
        let seq_len = 4;
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.1).collect();
        let k = q.clone();
        let (q_rot, k_rot) = rope.apply(&q, &k, seq_len, head_dim);
        assert_eq!(q_rot.len(), q.len());
        assert_eq!(k_rot.len(), k.len());
        let changed = q.iter().zip(q_rot.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "RoPE should modify values for non-zero positions");
    }

    // ── 6. RMSNorm normalisation ─────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_unit_rms_output() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32; 4];
        let out = InternLm2RmsNorm::forward(&x, &w, 1e-5);
        assert_eq!(out.len(), 4);
        let rms: f32 = (out.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!(
            (rms - 1.0).abs() < 0.01,
            "RMS of normed output ≈ 1, got {rms}"
        );
    }

    // ── 7. Attention output shape ────────────────────────────────────────────

    #[test]
    fn test_attention_output_shape() {
        let cfg = tiny_config();
        let h = cfg.hidden_size;
        let seq_len = 5;
        let attn = InternLm2Attention::new(cfg, 0);
        let input = vec![0.5f32; seq_len * h];
        let out = attn.forward(&input, seq_len);
        assert_eq!(
            out.len(),
            seq_len * h,
            "attention output must have shape [seq_len * hidden_size]"
        );
    }

    // ── 8. GQA head mapping ──────────────────────────────────────────────────

    #[test]
    fn test_gqa_head_mapping() {
        let cfg = tiny_config(); // 4 Q heads, 2 KV heads
        let attn = InternLm2Attention::new(cfg, 0);
        // heads 0,1 → kv head 0; heads 2,3 → kv head 1
        assert_eq!(attn.kv_head_for_q(0), 0);
        assert_eq!(attn.kv_head_for_q(1), 0);
        assert_eq!(attn.kv_head_for_q(2), 1);
        assert_eq!(attn.kv_head_for_q(3), 1);
    }

    // ── 9. MLP SwiGLU shape ──────────────────────────────────────────────────

    #[test]
    fn test_mlp_swiglu_output_shape() {
        let cfg = tiny_config();
        let h = cfg.hidden_size;
        let mlp = InternLm2MLP::new(&cfg);
        let input = vec![1.0f32; h];
        let out = mlp.forward(&input);
        assert_eq!(out.len(), h, "MLP output length must equal hidden_size");
    }

    // ── 10. Decoder layer shape ──────────────────────────────────────────────

    #[test]
    fn test_decoder_layer_output_shape() {
        let cfg = tiny_config();
        let h = cfg.hidden_size;
        let seq_len = 3;
        let layer = InternLm2DecoderLayer::new(cfg, 0);
        let input = vec![0.2f32; seq_len * h];
        let out = layer.forward(&input, seq_len);
        assert_eq!(out.len(), seq_len * h);
    }

    // ── 11. Model creation and layer count ──────────────────────────────────

    #[test]
    fn test_model_creation_layer_count() {
        let cfg = tiny_config();
        let num_layers = cfg.num_hidden_layers;
        let model = InternLm2Model::new(cfg);
        assert_eq!(model.layers.len(), num_layers);
    }

    // ── 12. Model forward pass ───────────────────────────────────────────────

    #[test]
    fn test_model_forward_output_shape() {
        let cfg = tiny_config();
        let h = cfg.hidden_size;
        let model = InternLm2Model::new(cfg);
        let input_ids = vec![1u32, 2, 3];
        let out = model.forward(&input_ids).expect("forward should succeed");
        assert_eq!(out.len(), 3 * h, "output must be [seq_len * hidden_size]");
    }

    // ── 13. CausalLM logits shape ────────────────────────────────────────────

    #[test]
    fn test_causal_lm_logits_shape() {
        let cfg = tiny_config();
        let v = cfg.vocab_size;
        let lm = InternLm2ForCausalLM::new(cfg);
        let input_ids = vec![1u32, 2, 3, 4];
        let logits = lm.forward(&input_ids).expect("forward should succeed");
        assert_eq!(logits.len(), 4 * v, "logits must be [seq_len * vocab_size]");
    }

    // ── 14. CausalLM greedy generation ──────────────────────────────────────

    #[test]
    fn test_causal_lm_greedy_generation() {
        let cfg = tiny_config();
        let lm = InternLm2ForCausalLM::new(cfg);
        let input_ids = vec![1u32, 2];
        let generated = lm.generate(&input_ids, 4).expect("generate should succeed");
        assert_eq!(
            generated.len(),
            4,
            "should return exactly max_new_tokens tokens"
        );
        for tok in &generated {
            assert!(
                (*tok as usize) < 128,
                "all tokens must be within vocab range"
            );
        }
    }

    // ── 15. Chat prompt formatting ───────────────────────────────────────────

    #[test]
    fn test_chatml_prompt_format() {
        let prompt = InternLm2ForCausalLM::format_chat_prompt("Be helpful.", "What is 2+2?");
        assert!(prompt.contains("<|im_start|>system\nBe helpful.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nWhat is 2+2?<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    // ── 16. Error type display ───────────────────────────────────────────────

    #[test]
    fn test_error_display_variants() {
        let inv = InternLm2Error::InvalidInput("bad token".to_string());
        let fwd = InternLm2Error::ForwardError("NaN".to_string());
        assert!(inv.to_string().contains("bad token"));
        assert!(fwd.to_string().contains("NaN"));
        let _boxed: Box<dyn std::error::Error> = Box::new(InternLm2Error::InvalidInput("x".into()));
    }

    // ── 17. Empty input returns error ────────────────────────────────────────

    #[test]
    fn test_empty_input_returns_error() {
        let cfg = tiny_config();
        let model = InternLm2Model::new(cfg);
        let err = model.forward(&[]);
        assert!(err.is_err(), "empty input must return an error");
    }

    // ── 18. Out-of-vocabulary token returns error ────────────────────────────

    #[test]
    fn test_oov_token_returns_error() {
        let cfg = tiny_config();
        let model = InternLm2Model::new(cfg);
        // Token 128 is out of range for vocab_size=128
        let err = model.forward(&[128u32]);
        assert!(err.is_err(), "OOV token should return an error");
        // Token 127 is the boundary token and must succeed
        assert!(
            model.forward(&[127u32]).is_ok(),
            "last valid token should succeed"
        );
    }

    // ── 19. NTK RoPE scaling produces different rotations ──────────────────

    #[test]
    fn test_ntk_rope_scaling_differs_from_base() {
        let base = InternLm2RotaryEmbedding::new(10000.0, None);
        let ntk = InternLm2RotaryEmbedding::new(10000.0, Some(2.0));
        let head_dim = 16;
        let seq_len = 4;
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.1).collect();
        let k = q.clone();
        let (q_base, _) = base.apply(&q, &k, seq_len, head_dim);
        let (q_ntk, _) = ntk.apply(&q, &k, seq_len, head_dim);
        let differs = q_base.iter().zip(q_ntk.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs, "NTK scaling must produce different rotations");
    }

    // ── 20. RMSNorm sign preservation ────────────────────────────────────────

    #[test]
    fn test_rmsnorm_preserves_sign() {
        let x = vec![3.0f32, -3.0, 0.0, 6.0];
        let w = vec![1.0f32; 4];
        let out = InternLm2RmsNorm::forward(&x, &w, 1e-8);
        assert!(
            out[0] > 0.0,
            "positive inputs should remain positive after RMSNorm"
        );
        assert!(
            out[1] < 0.0,
            "negative inputs should remain negative after RMSNorm"
        );
    }
}
