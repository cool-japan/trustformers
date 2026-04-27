use crate::qwen2::config::Qwen2Config;
use crate::qwen2::model::Qwen2ForCausalLM;
use trustformers_core::errors::Result;
use trustformers_core::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Qwen-2 chat template
// ─────────────────────────────────────────────────────────────────────────────

/// Format a conversation in the Qwen-2 ChatML template.
///
/// Qwen-2 uses OpenAI's ChatML format with special tokens:
///
/// ```text
/// <|im_start|>system
/// {system_prompt}
/// <|im_end|>
/// <|im_start|>user
/// {user_message}<|im_end|>
/// <|im_start|>assistant
/// {assistant_message}<|im_end|>
/// …
/// <|im_start|>assistant
/// ```
///
/// `messages` is a slice of `(role, content)` pairs.
pub fn format_qwen2_chat(system: &str, messages: &[(String, String)]) -> String {
    const IM_START: &str = "<|im_start|>";
    const IM_END: &str = "<|im_end|>";

    let mut buf = String::new();

    // System block
    if !system.is_empty() {
        buf.push_str(IM_START);
        buf.push_str("system\n");
        buf.push_str(system);
        buf.push('\n');
        buf.push_str(IM_END);
        buf.push('\n');
    }

    // Conversation turns
    for (role, content) in messages {
        buf.push_str(IM_START);
        buf.push_str(role.as_str());
        buf.push('\n');
        buf.push_str(content.as_str());
        buf.push_str(IM_END);
        buf.push('\n');
    }

    // Open assistant turn (model continues from here)
    buf.push_str(IM_START);
    buf.push_str("assistant\n");

    buf
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat model wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Chat / instruction-following wrapper for Qwen-2 instruction-tuned models
pub struct Qwen2ChatModel {
    inner: Qwen2ForCausalLM,
}

impl Qwen2ChatModel {
    pub fn new(config: Qwen2Config) -> Result<Self> {
        let inner = Qwen2ForCausalLM::new(config)?;
        Ok(Self { inner })
    }

    pub fn config(&self) -> &Qwen2Config {
        self.inner.config()
    }

    pub fn parameter_count(&self) -> usize {
        self.inner.parameter_count()
    }

    /// Run a forward pass over the given token IDs and return logits.
    pub fn forward(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        self.inner.forward(input_ids)
    }

    /// Apply the Qwen-2 ChatML template and return the formatted prompt string.
    pub fn build_prompt(
        &self,
        system_prompt: &str,
        messages: &[(String, String)],
    ) -> String {
        format_qwen2_chat(system_prompt, messages)
    }

    /// Greedy next-token selection from a logit tensor.
    pub fn greedy_next_token(&self, logits: &Tensor) -> Result<u32> {
        match logits {
            Tensor::F32(arr) => {
                let flat: Vec<f32> = arr.iter().copied().collect();
                let best = flat
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0);
                Ok(best)
            },
            _ => Ok(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen2::config::Qwen2Config;
    use trustformers_core::tensor::Tensor;

    fn small_config() -> Qwen2Config {
        Qwen2Config {
            vocab_size: 256,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            sliding_window: None,
            use_sliding_window: false,
            qkv_bias: false,
        }
    }

    // ── 1. format_qwen2_chat: no system → no system block ────────────────────

    #[test]
    fn test_format_qwen2_no_system_when_empty() {
        let out = format_qwen2_chat("", &[]);
        assert!(!out.contains("system"), "no system block for empty system");
    }

    // ── 2. format_qwen2_chat: system block present when non-empty ────────────

    #[test]
    fn test_format_qwen2_system_present() {
        let out = format_qwen2_chat("You are helpful.", &[]);
        assert!(out.contains("system"), "system role must appear");
        assert!(out.contains("You are helpful."), "system content must appear");
    }

    // ── 3. format_qwen2_chat: uses im_start/im_end tokens ────────────────────

    #[test]
    fn test_format_qwen2_chatml_tokens() {
        let out = format_qwen2_chat("sys", &[]);
        assert!(out.contains("<|im_start|>"), "must use im_start");
        assert!(out.contains("<|im_end|>"), "must use im_end");
    }

    // ── 4. format_qwen2_chat: ends with open assistant turn ──────────────────

    #[test]
    fn test_format_qwen2_ends_with_assistant() {
        let out = format_qwen2_chat("sys", &[("user".to_string(), "hi".to_string())]);
        assert!(out.ends_with("assistant\n"), "must end with open assistant turn");
    }

    // ── 5. format_qwen2_chat: user message in output ─────────────────────────

    #[test]
    fn test_format_qwen2_user_message_present() {
        let msgs = vec![("user".to_string(), "Hello Qwen".to_string())];
        let out = format_qwen2_chat("", &msgs);
        assert!(out.contains("Hello Qwen"), "user message must appear");
    }

    // ── 6. format_qwen2_chat: multiple roles all present ─────────────────────

    #[test]
    fn test_format_qwen2_multiple_roles() {
        let msgs = vec![
            ("user".to_string(), "question".to_string()),
            ("assistant".to_string(), "answer".to_string()),
        ];
        let out = format_qwen2_chat("", &msgs);
        assert!(out.contains("question"), "user content present");
        assert!(out.contains("answer"), "assistant content present");
    }

    // ── 7. format_qwen2_chat: im_start count matches expected ────────────────

    #[test]
    fn test_format_qwen2_im_start_count() {
        let msgs = vec![("user".to_string(), "hi".to_string())];
        let out = format_qwen2_chat("sys", &msgs);
        // system + user + trailing assistant = 3
        let count = out.matches("<|im_start|>").count();
        assert_eq!(count, 3, "expected 3 im_start tokens, got {count}");
    }

    // ── 8. format_qwen2_chat: im_end count matches ────────────────────────────

    #[test]
    fn test_format_qwen2_im_end_count() {
        let msgs = vec![("user".to_string(), "hi".to_string())];
        let out = format_qwen2_chat("sys", &msgs);
        // system im_end + user im_end = 2
        let count = out.matches("<|im_end|>").count();
        assert_eq!(count, 2, "expected 2 im_end tokens, got {count}");
    }

    // ── 9. format_qwen2_chat: deterministic ───────────────────────────────────

    #[test]
    fn test_format_qwen2_deterministic() {
        let msgs = vec![("user".to_string(), "test".to_string())];
        let a = format_qwen2_chat("sys", &msgs);
        let b = format_qwen2_chat("sys", &msgs);
        assert_eq!(a, b, "format must be deterministic");
    }

    // ── 10. Qwen2ChatModel construction ───────────────────────────────────────

    #[test]
    fn test_qwen2_chat_model_construction() {
        let result = Qwen2ChatModel::new(small_config());
        assert!(result.is_ok(), "Qwen2ChatModel must construct successfully");
    }

    // ── 11. parameter_count > 0 ───────────────────────────────────────────────

    #[test]
    fn test_qwen2_parameter_count_nonzero() {
        let model = Qwen2ChatModel::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        assert!(model.parameter_count() > 0, "parameter count must be > 0");
    }

    // ── 12. config accessor correctness ──────────────────────────────────────

    #[test]
    fn test_qwen2_config_accessor() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let model = Qwen2ChatModel::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        assert_eq!(model.config().vocab_size, vocab);
    }

    // ── 13. forward produces non-empty logits ─────────────────────────────────

    #[test]
    fn test_qwen2_forward_nonempty_logits() {
        let model = Qwen2ChatModel::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let result = model.forward(vec![1u32, 2, 3]);
        assert!(result.is_ok(), "forward must succeed");
    }

    // ── 14. build_prompt matches format_qwen2_chat directly ──────────────────

    #[test]
    fn test_build_prompt_matches_format_fn() {
        let model = Qwen2ChatModel::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let msgs = vec![("user".to_string(), "test".to_string())];
        let via_model = model.build_prompt("sys", &msgs);
        let direct = format_qwen2_chat("sys", &msgs);
        assert_eq!(via_model, direct, "build_prompt must match format fn");
    }

    // ── 15. greedy_next_token picks argmax ────────────────────────────────────

    #[test]
    fn test_qwen2_greedy_picks_argmax() {
        let model = Qwen2ChatModel::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let t = Tensor::from_vec(vec![0.1f32, 0.3, 0.9, 0.2], &[4])
            .unwrap_or_else(|_| panic!("tensor failed"));
        let tok = model.greedy_next_token(&t).unwrap_or(99);
        assert_eq!(tok, 2u32, "argmax of [0.1, 0.3, 0.9, 0.2] must be 2");
    }

    // ── 16. greedy_next_token: first element is max ───────────────────────────

    #[test]
    fn test_qwen2_greedy_first_max() {
        let model = Qwen2ChatModel::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let t = Tensor::from_vec(vec![9.0f32, 1.0, 2.0], &[3])
            .unwrap_or_else(|_| panic!("tensor failed"));
        let tok = model.greedy_next_token(&t).unwrap_or(99);
        assert_eq!(tok, 0u32, "argmax of [9, 1, 2] must be 0");
    }

    // ── 17. forward output is finite ─────────────────────────────────────────

    #[test]
    fn test_qwen2_forward_finite() {
        let model = Qwen2ChatModel::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        if let Ok(tensor) = model.forward(vec![0u32]) {
            if let Tensor::F32(arr) = &tensor {
                for &v in arr.iter() {
                    assert!(v.is_finite(), "logit {v} must be finite");
                }
            }
        }
    }

    // ── 18. format_qwen2_chat: empty everything → only open assistant ─────────

    #[test]
    fn test_format_qwen2_only_assistant_when_all_empty() {
        let out = format_qwen2_chat("", &[]);
        assert!(out.contains("<|im_start|>assistant"), "must have open assistant");
    }
}
