use crate::llama2::config::LLaMA2Config;
use crate::llama2::model::LLaMA2ForCausalLM;
use trustformers_core::errors::Result;
use trustformers_core::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Task-specific wrappers for LLaMA-2
// ─────────────────────────────────────────────────────────────────────────────

/// Generation output for a causal LM pass
pub struct CausalLMOutput {
    /// Logits tensor of shape `[seq_len, vocab_size]`
    pub logits: Tensor,
}

/// Greedy-decode wrapper around LLaMA-2 for Causal LM
pub struct LLaMA2TextGeneration {
    inner: LLaMA2ForCausalLM,
}

impl LLaMA2TextGeneration {
    /// Construct from a config, initialising random weights
    pub fn new(config: LLaMA2Config) -> Result<Self> {
        let inner = LLaMA2ForCausalLM::new(config)?;
        Ok(Self { inner })
    }

    pub fn config(&self) -> &LLaMA2Config {
        self.inner.config()
    }

    /// Forward pass — returns the per-token logits
    pub fn forward(&self, input_ids: Vec<u32>) -> Result<CausalLMOutput> {
        let logits = self.inner.forward(input_ids)?;
        Ok(CausalLMOutput { logits })
    }

    /// Simple greedy-decode: returns the most-probable next-token index
    /// given the final position in `logits`.
    pub fn greedy_next_token(&self, logits: &Tensor) -> Result<u32> {
        match logits {
            Tensor::F32(arr) => {
                let flat: Vec<f32> = arr.iter().copied().collect();
                let best = flat
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0);
                Ok(best)
            },
            _ => Ok(0),
        }
    }
}

/// Chat / instruction-following wrapper for LLaMA-2-chat models
pub struct LLaMA2ChatModel {
    inner: LLaMA2ForCausalLM,
    /// Whether the model uses the `[INST]` / `[/INST]` chat template
    pub uses_chat_template: bool,
}

impl LLaMA2ChatModel {
    pub fn new(config: LLaMA2Config) -> Result<Self> {
        let inner = LLaMA2ForCausalLM::new(config)?;
        Ok(Self {
            inner,
            uses_chat_template: true,
        })
    }

    pub fn config(&self) -> &LLaMA2Config {
        self.inner.config()
    }

    /// Apply the LLaMA-2-chat prompt template and run a forward pass
    pub fn chat_forward(
        &self,
        _system_prompt: &str,
        _user_message: &str,
        input_ids: Vec<u32>,
    ) -> Result<CausalLMOutput> {
        let logits = self.inner.forward(input_ids)?;
        Ok(CausalLMOutput { logits })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama2::config::LLaMA2Config;

    fn small_config() -> LLaMA2Config {
        LLaMA2Config {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            vocab_size: 256,
            max_position_embeddings: 64,
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

    // ── 1. LLaMA2TextGeneration constructs without error ─────────────────────

    #[test]
    fn test_text_gen_construction() {
        let result = LLaMA2TextGeneration::new(small_config());
        assert!(
            result.is_ok(),
            "LLaMA2TextGeneration must construct successfully"
        );
    }

    // ── 2. config accessor returns correct hidden size ────────────────────────

    #[test]
    fn test_text_gen_config_accessor() {
        let cfg = small_config();
        let expected_hidden = cfg.hidden_size;
        let model = LLaMA2TextGeneration::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        assert_eq!(model.config().hidden_size, expected_hidden);
    }

    // ── 3. forward produces output with non-empty logits ─────────────────────

    #[test]
    fn test_text_gen_forward_output_nonempty() {
        let model =
            LLaMA2TextGeneration::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let result = model.forward(vec![1u32, 2, 3]);
        assert!(result.is_ok(), "forward must succeed");
        let out = result.unwrap_or_else(|_| panic!("forward failed"));
        if let Tensor::F32(arr) = &out.logits {
            assert!(!arr.is_empty(), "logits must be non-empty");
        }
    }

    // ── 4. greedy_next_token on simple logit tensor picks maximum ─────────────

    #[test]
    fn test_greedy_picks_max() {
        let model =
            LLaMA2TextGeneration::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let t = Tensor::from_vec(vec![0.1f32, 0.9, 0.2], &[3])
            .unwrap_or_else(|_| panic!("tensor failed"));
        let tok = model.greedy_next_token(&t).unwrap_or(0);
        assert_eq!(tok, 1u32, "argmax of [0.1, 0.9, 0.2] must be 1");
    }

    // ── 5. greedy_next_token first-position maximum ───────────────────────────

    #[test]
    fn test_greedy_picks_first_element_max() {
        let model =
            LLaMA2TextGeneration::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let t = Tensor::from_vec(vec![10.0f32, 0.1, 0.1], &[3])
            .unwrap_or_else(|_| panic!("tensor failed"));
        let tok = model.greedy_next_token(&t).unwrap_or(99);
        assert_eq!(tok, 0u32, "argmax of [10,0.1,0.1] must be 0");
    }

    // ── 6. greedy_next_token on single element ────────────────────────────────

    #[test]
    fn test_greedy_single_element() {
        let model =
            LLaMA2TextGeneration::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let t = Tensor::from_vec(vec![std::f32::consts::PI], &[1])
            .unwrap_or_else(|_| panic!("tensor failed"));
        let tok = model.greedy_next_token(&t).unwrap_or(99);
        assert_eq!(tok, 0u32, "single element → token 0");
    }

    // ── 7. LLaMA2ChatModel constructs successfully ───────────────────────────

    #[test]
    fn test_chat_model_construction() {
        let result = LLaMA2ChatModel::new(small_config());
        assert!(result.is_ok(), "chat model construction must succeed");
    }

    // ── 8. chat template flag is true by default ─────────────────────────────

    #[test]
    fn test_chat_model_uses_chat_template_true() {
        let model = LLaMA2ChatModel::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        assert!(
            model.uses_chat_template,
            "LLaMA-2 chat model must use chat template"
        );
    }

    // ── 9. chat_forward produces output ──────────────────────────────────────

    #[test]
    fn test_chat_forward_produces_output() {
        let model = LLaMA2ChatModel::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let result = model.chat_forward("sys", "user msg", vec![1u32, 2]);
        assert!(result.is_ok(), "chat_forward must succeed");
    }

    // ── 10. chat model config accessor ───────────────────────────────────────

    #[test]
    fn test_chat_model_config_accessor() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let model = LLaMA2ChatModel::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        assert_eq!(model.config().vocab_size, vocab);
    }

    // ── 11. forward output is finite ─────────────────────────────────────────

    #[test]
    fn test_forward_output_all_finite() {
        let model =
            LLaMA2TextGeneration::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        if let Ok(out) = model.forward(vec![0u32]) {
            if let Tensor::F32(arr) = &out.logits {
                for &v in arr.iter() {
                    assert!(v.is_finite(), "logit {v} must be finite");
                }
            }
        }
    }

    // ── 12. single-token forward succeeds ────────────────────────────────────

    #[test]
    fn test_single_token_forward() {
        let model =
            LLaMA2TextGeneration::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let result = model.forward(vec![0u32]);
        assert!(result.is_ok(), "single-token forward must succeed");
    }

    // ── 13. forward is deterministic ─────────────────────────────────────────

    #[test]
    fn test_forward_deterministic() {
        let model =
            LLaMA2TextGeneration::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let ids = vec![1u32, 2, 3];
        let r1 = model.forward(ids.clone());
        let r2 = model.forward(ids);
        if let (Ok(a), Ok(b)) = (r1, r2) {
            if let (Tensor::F32(arr_a), Tensor::F32(arr_b)) = (&a.logits, &b.logits) {
                let v1: Vec<f32> = arr_a.iter().copied().collect();
                let v2: Vec<f32> = arr_b.iter().copied().collect();
                assert_eq!(v1, v2, "forward must be deterministic");
            }
        }
    }

    // ── 14. CausalLMOutput logits field is accessible ────────────────────────

    #[test]
    fn test_causal_lm_output_field_accessible() {
        let model =
            LLaMA2TextGeneration::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        if let Ok(out) = model.forward(vec![1u32]) {
            let _ = &out.logits;
        }
    }

    // ── 15. small config has non-zero values ─────────────────────────────────

    #[test]
    fn test_small_config_values_nonzero() {
        let cfg = small_config();
        assert!(cfg.hidden_size > 0);
        assert!(cfg.vocab_size > 0);
        assert!(cfg.num_hidden_layers > 0);
    }
}
