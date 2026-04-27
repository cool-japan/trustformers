//! Falcon-2 task-specific wrappers.
//!
//! Provides `Falcon2ForCausalLM` for text generation with greedy decoding,
//! Falcon Instruct chat-format helpers, and ALiBi slope utilities.

use crate::falcon2::config::Falcon2Config;
use crate::falcon2::model::{Falcon2AlibiPositionalBias, Falcon2Error, Falcon2Model};

// ─────────────────────────────────────────────────────────────────────────────
// CausalLM head
// ─────────────────────────────────────────────────────────────────────────────

/// Falcon-2 causal language model (base model + LM head).
pub struct Falcon2ForCausalLM {
    /// Underlying Falcon-2 base model.
    pub model: Falcon2Model,
    /// LM-head projection weight: `[vocab_size × hidden_size]` (zero-init placeholder).
    pub lm_head_weight: Vec<f32>,
}

impl Falcon2ForCausalLM {
    /// Create a new causal-LM model with zero-initialised weights.
    pub fn new(config: Falcon2Config) -> Self {
        let v = config.vocab_size;
        let h = config.hidden_size;
        let model = Falcon2Model::new(config);
        let lm_head_weight = vec![0.0_f32; v * h];
        Self {
            model,
            lm_head_weight,
        }
    }

    /// Compute logits over the vocabulary for each position.
    ///
    /// Returns a flat `[seq_len × vocab_size]` array.
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>, Falcon2Error> {
        let hidden = self.model.forward(input_ids)?;
        let seq_len = input_ids.len();
        let h = self.model.config.hidden_size;
        let v = self.model.config.vocab_size;

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
    pub fn generate(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, Falcon2Error> {
        if input_ids.is_empty() {
            return Err(Falcon2Error::InvalidInput(
                "input_ids must not be empty for generation".to_string(),
            ));
        }

        let v = self.model.config.vocab_size;
        let mut ids: Vec<u32> = input_ids.to_vec();

        for _ in 0..max_new_tokens {
            let logits = self.forward(&ids)?;
            let seq_len = ids.len();
            let last_logits = &logits[(seq_len - 1) * v..seq_len * v];
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .ok_or_else(|| Falcon2Error::ForwardError("empty logits".to_string()))?;
            ids.push(next_token);
        }

        Ok(ids[input_ids.len()..].to_vec())
    }

    /// Format a user prompt into the Falcon Instruct template.
    ///
    /// ```text
    /// User: {user}
    /// Falcon:
    /// ```
    pub fn format_chat_prompt(user: &str) -> String {
        format!("User: {user}\nFalcon: ")
    }

    /// Compute ALiBi slopes for `n` heads (convenience re-export).
    ///
    /// `slopes[h] = 2^(-8*(h+1)/n)` — slopes are strictly decreasing.
    pub fn compute_alibi_slopes(n: usize) -> Vec<f32> {
        Falcon2AlibiPositionalBias::compute_slopes(n)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::falcon2::{
        config::Falcon2Config,
        model::{
            Falcon2AlibiPositionalBias, Falcon2DecoderLayer, Falcon2Error, Falcon2LayerNorm,
            Falcon2MLP,
        },
    };

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Tiny config for fast unit tests.
    fn tiny_config() -> Falcon2Config {
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

    // ── 1. Config default ────────────────────────────────────────────────────

    #[test]
    fn test_falcon2_config_default() {
        let cfg = Falcon2Config::default();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 60);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.num_kv_heads, 1);
        assert_eq!(cfg.intermediate_size, 16384);
        assert_eq!(cfg.max_position_embeddings, 8192);
        assert_eq!(cfg.vocab_size, 65024);
        assert!(cfg.use_alibi);
        assert!(cfg.parallel_attn);
        assert!(!cfg.bias);
        assert_eq!(cfg.hidden_act, "gelu");
    }

    // ── 2. Config custom ─────────────────────────────────────────────────────

    #[test]
    fn test_falcon2_config_custom() {
        let cfg = tiny_config();
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.num_kv_heads, 1);
        assert_eq!(cfg.head_dim(), 64 / 4);
    }

    // ── 3. ALiBi slopes computation ──────────────────────────────────────────

    #[test]
    fn test_falcon2_alibi_slopes_compute() {
        let slopes = Falcon2AlibiPositionalBias::compute_slopes(4);
        assert_eq!(slopes.len(), 4);
        // slopes[0] = 2^(-8*1/4) = 2^(-2) = 0.25
        assert!(
            (slopes[0] - 0.25).abs() < 1e-5,
            "slope[0] should be 0.25, got {}",
            slopes[0]
        );
        // slopes[1] = 2^(-8*2/4) = 2^(-4) = 0.0625
        assert!(
            (slopes[1] - 0.0625).abs() < 1e-5,
            "slope[1] should be 0.0625, got {}",
            slopes[1]
        );
    }

    // ── 4. ALiBi slopes are strictly decreasing ──────────────────────────────

    #[test]
    fn test_falcon2_alibi_slopes_decreasing() {
        let slopes = Falcon2AlibiPositionalBias::compute_slopes(8);
        for w in slopes.windows(2) {
            assert!(
                w[0] > w[1],
                "slopes must be strictly decreasing: {} !> {}",
                w[0],
                w[1]
            );
        }
    }

    // ── 5. ALiBi bias shape ──────────────────────────────────────────────────

    #[test]
    fn test_falcon2_alibi_bias_shape() {
        let num_heads = 4;
        let seq_len = 6;
        let slopes = Falcon2AlibiPositionalBias::compute_slopes(num_heads);
        let bias = Falcon2AlibiPositionalBias::compute_bias(seq_len, &slopes);
        assert_eq!(bias.len(), num_heads * seq_len * seq_len);
    }

    // ── 6. ALiBi causal mask: diagonal is always zero ────────────────────────

    #[test]
    fn test_falcon2_alibi_causal_mask() {
        let num_heads = 2;
        let seq_len = 4;
        let slopes = Falcon2AlibiPositionalBias::compute_slopes(num_heads);
        let bias = Falcon2AlibiPositionalBias::compute_bias(seq_len, &slopes);
        // Diagonal elements (i == j) must be 0 (distance = 0).
        for h in 0..num_heads {
            for i in 0..seq_len {
                let val = bias[h * seq_len * seq_len + i * seq_len + i];
                assert!((val).abs() < 1e-7, "diagonal bias must be 0, got {val}");
            }
        }
        // Off-diagonal elements must be negative.
        for h in 0..num_heads {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if i != j {
                        let val = bias[h * seq_len * seq_len + i * seq_len + j];
                        assert!(val < 0.0, "off-diagonal bias must be negative, got {val}");
                    }
                }
            }
        }
    }

    // ── 7. LayerNorm ─────────────────────────────────────────────────────────

    #[test]
    fn test_falcon2_layer_norm() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0_f32; 4];
        let bias = vec![0.0_f32; 4];
        let out = Falcon2LayerNorm::forward(&x, &weight, &bias, 1e-5);
        assert_eq!(out.len(), 4);
        // Mean of output should be ~0, std ~1.
        let mean = out.iter().sum::<f32>() / 4.0;
        assert!(
            mean.abs() < 1e-5,
            "LayerNorm output mean should be ~0, got {mean}"
        );
    }

    // ── 8. GELU activation ───────────────────────────────────────────────────

    #[test]
    fn test_falcon2_gelu_activation() {
        // GELU(1.0) ≈ 0.841
        let y = Falcon2MLP::gelu(1.0);
        assert!(
            (y - 0.841).abs() < 0.005,
            "gelu(1.0) should be ~0.841, got {y}"
        );
        // GELU is approximately linear for large positive x.
        let large = Falcon2MLP::gelu(10.0);
        assert!(
            (large - 10.0).abs() < 0.1,
            "gelu(10.0) should be ~10.0, got {large}"
        );
    }

    // ── 9. GELU at zero ──────────────────────────────────────────────────────

    #[test]
    fn test_falcon2_gelu_zero() {
        let y = Falcon2MLP::gelu(0.0);
        assert!(y.abs() < 1e-6, "gelu(0) must be 0, got {y}");
    }

    // ── 10. MQA: single KV head ──────────────────────────────────────────────

    #[test]
    fn test_falcon2_mqa_single_kv_head() {
        let cfg = tiny_config();
        assert_eq!(cfg.num_kv_heads, 1, "Falcon-2 uses single KV head (MQA)");
        // The MQA property: all Q heads share the same K/V.
        // We verify the config enforces this.
        assert!(cfg.num_attention_heads > cfg.num_kv_heads);
    }

    // ── 11. Parallel attn+MLP ────────────────────────────────────────────────

    #[test]
    fn test_falcon2_parallel_attn_mlp() {
        let cfg = tiny_config();
        let h = cfg.hidden_size;
        let seq_len = 4;
        let layer = Falcon2DecoderLayer::new(cfg);
        let input = vec![0.5_f32; seq_len * h];
        let out = layer.forward(&input, seq_len);
        assert_eq!(out.len(), seq_len * h);
    }

    // ── 12. Model forward ────────────────────────────────────────────────────

    #[test]
    fn test_falcon2_model_forward() {
        let cfg = tiny_config();
        let h = cfg.hidden_size;
        use crate::falcon2::model::Falcon2Model;
        let model = Falcon2Model::new(cfg);
        let input_ids = vec![1_u32, 2, 3, 4];
        let out = model.forward(&input_ids).expect("forward should succeed");
        assert_eq!(out.len(), 4 * h);
    }

    // ── 13. CausalLM forward ─────────────────────────────────────────────────

    #[test]
    fn test_falcon2_causal_lm_forward() {
        let cfg = tiny_config();
        let v = cfg.vocab_size;
        let lm = Falcon2ForCausalLM::new(cfg);
        let input_ids = vec![1_u32, 2, 3];
        let logits = lm.forward(&input_ids).expect("forward should succeed");
        assert_eq!(
            logits.len(),
            3 * v,
            "logits shape must be [seq_len * vocab_size]"
        );
    }

    // ── 14. Generate ─────────────────────────────────────────────────────────

    #[test]
    fn test_falcon2_generate() {
        let cfg = tiny_config();
        let lm = Falcon2ForCausalLM::new(cfg);
        let input_ids = vec![1_u32, 2];
        let generated = lm.generate(&input_ids, 3).expect("generate should succeed");
        assert_eq!(generated.len(), 3);
        for tok in &generated {
            assert!(*tok < 128, "generated tokens must be within vocab");
        }
    }

    // ── 15. Chat format ──────────────────────────────────────────────────────

    #[test]
    fn test_falcon2_chat_format() {
        let prompt = Falcon2ForCausalLM::format_chat_prompt("What is Rust?");
        assert!(prompt.starts_with("User: "));
        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.contains("Falcon: "));
    }

    // ── 16. Error types ──────────────────────────────────────────────────────

    #[test]
    fn test_falcon2_error_types() {
        let e1 = Falcon2Error::InvalidInput("bad".to_string());
        let e2 = Falcon2Error::ForwardError("nan".to_string());
        assert!(e1.to_string().contains("bad"));
        assert!(e2.to_string().contains("nan"));
        // Verify std::error::Error impl.
        let _boxed: Box<dyn std::error::Error> = Box::new(Falcon2Error::ForwardError("x".into()));
    }

    // ── 17. Empty input error ────────────────────────────────────────────────

    #[test]
    fn test_falcon2_empty_input_error() {
        let cfg = tiny_config();
        use crate::falcon2::model::Falcon2Model;
        let model = Falcon2Model::new(cfg);
        let err = model.forward(&[]);
        assert!(err.is_err(), "empty input should return an error");
    }

    // ── 18. Out-of-vocab token error ─────────────────────────────────────────

    #[test]
    fn test_falcon2_oov_token_error() {
        let cfg = tiny_config();
        use crate::falcon2::model::Falcon2Model;
        let model = Falcon2Model::new(cfg);
        let err = model.forward(&[128_u32]); // vocab_size = 128
        assert!(err.is_err(), "out-of-vocab token should return an error");
    }
}
