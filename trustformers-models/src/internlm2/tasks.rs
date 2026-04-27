//! InternLM-2 task-specific wrappers.
//!
//! Provides `InternLm2ForCausalLM` for text generation with greedy decoding,
//! and chat-format helpers compatible with the InternLM ChatML template.

use crate::internlm2::config::InternLm2Config;
use crate::internlm2::model::{InternLm2Error, InternLm2Model};

// ─────────────────────────────────────────────────────────────────────────────
// CausalLM head
// ─────────────────────────────────────────────────────────────────────────────

/// InternLM-2 causal language model (base model + LM head).
pub struct InternLm2ForCausalLM {
    /// Underlying InternLM-2 base model.
    pub model: InternLm2Model,
    /// LM-head projection weight: `[vocab_size × hidden_size]` (zero-init placeholder).
    pub lm_head_weight: Vec<f32>,
}

impl InternLm2ForCausalLM {
    /// Create a new causal-LM model with zero-initialised weights.
    pub fn new(config: InternLm2Config) -> Self {
        let v = config.vocab_size;
        let h = config.hidden_size;
        let model = InternLm2Model::new(config);
        let lm_head_weight = vec![0.0_f32; v * h];
        Self {
            model,
            lm_head_weight,
        }
    }

    /// Compute logits over the vocabulary for each position.
    ///
    /// Returns a flat `[seq_len × vocab_size]` array.
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>, InternLm2Error> {
        let hidden = self.model.forward(input_ids)?;
        let seq_len = input_ids.len();
        let h = self.model.config.hidden_size;
        let v = self.model.config.vocab_size;

        // Linear projection: logits[pos][vocab] = Σ_i hidden[pos][i] * lm_head[vocab][i]
        // With zero lm_head_weight the logits will be zero; tests can verify the shape.
        let mut logits = vec![0.0_f32; seq_len * v];
        for pos in 0..seq_len {
            let h_slice = &hidden[pos * h..(pos + 1) * h];
            for vocab_idx in 0..v {
                let weight_slice = &self.lm_head_weight[vocab_idx * h..(vocab_idx + 1) * h];
                let dot: f32 = h_slice.iter().zip(weight_slice.iter()).map(|(x, w)| x * w).sum();
                logits[pos * v + vocab_idx] = dot;
            }
        }
        Ok(logits)
    }

    /// Greedy decoding: generate up to `max_new_tokens` new tokens.
    ///
    /// At each step the model performs a full forward pass and selects the token
    /// with the highest logit for the last position.
    pub fn generate(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, InternLm2Error> {
        if input_ids.is_empty() {
            return Err(InternLm2Error::InvalidInput(
                "input_ids must not be empty for generation".to_string(),
            ));
        }

        let v = self.model.config.vocab_size;
        let mut ids: Vec<u32> = input_ids.to_vec();

        for _ in 0..max_new_tokens {
            let logits = self.forward(&ids)?;
            let seq_len = ids.len();
            // Slice logits for the last position.
            let last_logits = &logits[(seq_len - 1) * v..seq_len * v];
            // Argmax.
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .ok_or_else(|| InternLm2Error::ForwardError("empty logits".to_string()))?;
            ids.push(next_token);
        }

        Ok(ids[input_ids.len()..].to_vec())
    }

    /// Format a system + user prompt into InternLM-2's ChatML template.
    ///
    /// ```text
    /// <|im_start|>system
    /// {system}<|im_end|>
    /// <|im_start|>user
    /// {user}<|im_end|>
    /// <|im_start|>assistant
    /// ```
    pub fn format_chat_prompt(system: &str, user: &str) -> String {
        format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internlm2::{
        config::InternLm2Config,
        model::{
            InternLm2Attention, InternLm2Error, InternLm2MLP, InternLm2RmsNorm,
            InternLm2RotaryEmbedding,
        },
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Tiny config suitable for fast unit tests.
    fn tiny_config() -> InternLm2Config {
        InternLm2Config::new(
            128, // vocab_size
            64,  // hidden_size
            2,   // num_hidden_layers
            4,   // num_attention_heads
            2,   // num_key_value_heads (GQA ratio = 2)
            128, // intermediate_size
            512, // max_position_embeddings
            10000.0, None, "silu", 1e-5, false, true,
        )
    }

    // ── 1. Config default ────────────────────────────────────────────────────

    #[test]
    fn test_internlm2_config_default() {
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
        assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-10);
        assert!(!cfg.tie_word_embeddings);
        assert!(cfg.use_cache);
    }

    // ── 2. Config custom ─────────────────────────────────────────────────────

    #[test]
    fn test_internlm2_config_custom() {
        let cfg = tiny_config();
        assert_eq!(cfg.vocab_size, 128);
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.gqa_ratio(), 2);
        assert_eq!(cfg.head_dim(), 16);
    }

    // ── 3. RoPE basic ────────────────────────────────────────────────────────

    #[test]
    fn test_internlm2_rope_basic() {
        let rope = InternLm2RotaryEmbedding::new(10000.0, None);
        let seq_len = 4;
        let head_dim = 8;
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.1).collect();
        let k = q.clone();
        let (q_rot, k_rot) = rope.apply(&q, &k, seq_len, head_dim);
        assert_eq!(q_rot.len(), q.len());
        assert_eq!(k_rot.len(), k.len());
        // Rotation should change at least some values.
        let changed = q.iter().zip(q_rot.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "RoPE should modify at least some values");
    }

    // ── 4. RoPE NTK scaling ──────────────────────────────────────────────────

    #[test]
    fn test_internlm2_rope_ntk_scaling() {
        let rope_base = InternLm2RotaryEmbedding::new(10000.0, None);
        let rope_ntk = InternLm2RotaryEmbedding::new(10000.0, Some(2.0));
        let seq_len = 4;
        let head_dim = 8;
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.1).collect();
        let k = q.clone();
        let (q_base, _) = rope_base.apply(&q, &k, seq_len, head_dim);
        let (q_ntk, _) = rope_ntk.apply(&q, &k, seq_len, head_dim);
        // NTK scaling should produce different rotations than the base.
        let differs = q_base.iter().zip(q_ntk.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differs, "NTK scaling should change the rotations");
    }

    // ── 5. RMSNorm ───────────────────────────────────────────────────────────

    #[test]
    fn test_internlm2_rms_norm() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0_f32; 4];
        let out = InternLm2RmsNorm::forward(&x, &weight, 1e-5);
        assert_eq!(out.len(), 4);
        // RMS norm output should have unit RMS (approximately).
        let rms_out: f32 = (out.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!(
            (rms_out - 1.0).abs() < 0.01,
            "RMS of normed output should be ~1, got {rms_out}"
        );
    }

    // ── 6. Attention output shape ────────────────────────────────────────────

    #[test]
    fn test_internlm2_attention_shape() {
        let cfg = tiny_config();
        let h = cfg.hidden_size;
        let seq_len = 4;
        let attn = InternLm2Attention::new(cfg, 0);
        let input = vec![0.5_f32; seq_len * h];
        let out = attn.forward(&input, seq_len);
        assert_eq!(
            out.len(),
            seq_len * h,
            "Attention output must have shape [seq_len * hidden_size]"
        );
    }

    // ── 7. GQA head mapping ──────────────────────────────────────────────────

    #[test]
    fn test_internlm2_gqa_head_mapping() {
        let cfg = tiny_config(); // 4 Q heads, 2 KV heads → ratio 2
        let attn = InternLm2Attention::new(cfg, 0);
        assert_eq!(attn.kv_head_for_q(0), 0);
        assert_eq!(attn.kv_head_for_q(1), 0);
        assert_eq!(attn.kv_head_for_q(2), 1);
        assert_eq!(attn.kv_head_for_q(3), 1);
    }

    // ── 8. MLP SwiGLU output shape ───────────────────────────────────────────

    #[test]
    fn test_internlm2_mlp_swiglu() {
        let cfg = tiny_config();
        let h = cfg.hidden_size;
        let mlp = InternLm2MLP::new(&cfg);
        let input = vec![1.0_f32; h];
        let out = mlp.forward(&input);
        assert_eq!(out.len(), h, "MLP output must match hidden_size");
    }

    // ── 9. Decoder layer shape ───────────────────────────────────────────────

    #[test]
    fn test_internlm2_decoder_layer() {
        let cfg = tiny_config();
        let h = cfg.hidden_size;
        let seq_len = 4;
        use crate::internlm2::model::InternLm2DecoderLayer;
        let layer = InternLm2DecoderLayer::new(cfg, 0);
        let input = vec![0.1_f32; seq_len * h];
        let out = layer.forward(&input, seq_len);
        assert_eq!(out.len(), seq_len * h);
    }

    // ── 10. Model forward ────────────────────────────────────────────────────

    #[test]
    fn test_internlm2_model_forward() {
        let cfg = tiny_config();
        let h = cfg.hidden_size;
        let model = InternLm2Model::new(cfg);
        let input_ids = vec![1_u32, 2, 3, 4];
        let out = model.forward(&input_ids).expect("forward should succeed");
        assert_eq!(out.len(), 4 * h);
    }

    // ── 11. CausalLM forward ─────────────────────────────────────────────────

    #[test]
    fn test_internlm2_causal_lm_forward() {
        let cfg = tiny_config();
        let v = cfg.vocab_size;
        let lm = InternLm2ForCausalLM::new(cfg);
        let input_ids = vec![1_u32, 2, 3];
        let logits = lm.forward(&input_ids).expect("forward should succeed");
        assert_eq!(
            logits.len(),
            3 * v,
            "logits shape must be [seq_len * vocab_size]"
        );
    }

    // ── 12. Generate greedy ──────────────────────────────────────────────────

    #[test]
    fn test_internlm2_generate_greedy() {
        let cfg = tiny_config();
        let lm = InternLm2ForCausalLM::new(cfg);
        let input_ids = vec![1_u32, 2];
        let generated = lm.generate(&input_ids, 3).expect("generate should succeed");
        assert_eq!(
            generated.len(),
            3,
            "generate should return max_new_tokens new tokens"
        );
        // Greedy decoding with all-zero logits will always return token 0.
        for tok in &generated {
            assert!(*tok < 128, "generated tokens must be within vocab");
        }
    }

    // ── 13. Chat format ──────────────────────────────────────────────────────

    #[test]
    fn test_internlm2_chat_format() {
        let prompt = InternLm2ForCausalLM::format_chat_prompt("You are helpful.", "Hello!");
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.contains("<|im_start|>assistant"));
    }

    // ── 14. Model new ────────────────────────────────────────────────────────

    #[test]
    fn test_internlm2_model_new() {
        let cfg = tiny_config();
        let num_layers = cfg.num_hidden_layers;
        let model = InternLm2Model::new(cfg);
        assert_eq!(model.layers.len(), num_layers);
    }

    // ── 15. Error display ────────────────────────────────────────────────────

    #[test]
    fn test_internlm2_error_display() {
        let err1 = InternLm2Error::InvalidInput("bad token".to_string());
        let err2 = InternLm2Error::ForwardError("NaN detected".to_string());
        assert!(err1.to_string().contains("bad token"));
        assert!(err2.to_string().contains("NaN detected"));
        // Verify std::error::Error is implemented.
        let _boxed: Box<dyn std::error::Error> = Box::new(InternLm2Error::InvalidInput("x".into()));
    }

    // ── 16. Large vocabulary ─────────────────────────────────────────────────

    #[test]
    fn test_internlm2_large_vocab() {
        // Ensure tokens near the vocab boundary are handled correctly.
        let cfg = tiny_config();
        let model = InternLm2Model::new(cfg);
        // Token 127 is within vocab (vocab_size = 128).
        let ok = model.forward(&[127_u32]);
        assert!(ok.is_ok(), "token at vocab boundary should succeed");
        // Token 128 is out of range.
        let err = model.forward(&[128_u32]);
        assert!(
            err.is_err(),
            "token out of vocabulary should return an error"
        );
    }

    // ── 17. Empty input error ────────────────────────────────────────────────

    #[test]
    fn test_internlm2_empty_input_error() {
        let cfg = tiny_config();
        let model = InternLm2Model::new(cfg);
        let err = model.forward(&[]);
        assert!(err.is_err(), "empty input should return an error");
    }

    // ── 18. RMSNorm all-ones weight ──────────────────────────────────────────

    #[test]
    fn test_internlm2_rms_norm_identity_weight() {
        let x = vec![2.0_f32, 0.0, -2.0, 0.0];
        let weight = vec![1.0_f32; 4];
        let out = InternLm2RmsNorm::forward(&x, &weight, 1e-8);
        // With all-ones weight the norm should scale but preserve sign.
        assert!(out[0] > 0.0);
        assert!(out[2] < 0.0);
    }
}
