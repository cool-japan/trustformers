use crate::llama3::config::LLaMA3Config;
use crate::llama3::model::LLaMA3ForCausalLM;
use trustformers_core::errors::Result;
use trustformers_core::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Task-specific wrappers for LLaMA-3
// ─────────────────────────────────────────────────────────────────────────────

/// Output of a LLaMA-3 causal LM forward pass
pub struct LLaMA3CausalLMOutput {
    /// Token logits, shape `[seq_len, vocab_size]`
    pub logits: Tensor,
}

// ─────────────────────────────────────────────────────────────────────────────
// LLaMA-3 chat template
// ─────────────────────────────────────────────────────────────────────────────

/// Format a conversation in the LLaMA-3 chat template.
///
/// LLaMA-3 uses special header and EOS tokens defined in the extended Tiktoken
/// vocabulary:
///
/// ```text
/// <|begin_of_text|>
/// <|start_header_id|>system<|end_header_id|>
/// {system_prompt}<|eot_id|>
/// <|start_header_id|>user<|end_header_id|>
/// {user_message}<|eot_id|>
/// <|start_header_id|>assistant<|end_header_id|>
/// {assistant_message}<|eot_id|>
/// …
/// <|start_header_id|>assistant<|end_header_id|>
/// ```
///
/// `messages` is a slice of `(role, content)` pairs.  If `role` is
/// `"user"` or `"assistant"` the appropriate header is emitted.
pub fn format_llama3_chat(system: &str, messages: &[(String, String)]) -> String {
    const BEGIN: &str = "<|begin_of_text|>";
    const START_HDR: &str = "<|start_header_id|>";
    const END_HDR: &str = "<|end_header_id|>";
    const EOT: &str = "<|eot_id|>";

    let mut buf = String::new();

    // Begin-of-text
    buf.push_str(BEGIN);

    // System block
    if !system.is_empty() {
        buf.push_str(START_HDR);
        buf.push_str("system");
        buf.push_str(END_HDR);
        buf.push('\n');
        buf.push_str(system);
        buf.push_str(EOT);
        buf.push('\n');
    }

    // Conversation turns
    for (role, content) in messages {
        buf.push_str(START_HDR);
        buf.push_str(role.as_str());
        buf.push_str(END_HDR);
        buf.push('\n');
        buf.push_str(content.as_str());
        buf.push_str(EOT);
        buf.push('\n');
    }

    // Open assistant turn (the model is expected to continue from here)
    buf.push_str(START_HDR);
    buf.push_str("assistant");
    buf.push_str(END_HDR);
    buf.push('\n');

    buf
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat model wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Chat / instruction-following wrapper for LLaMA-3 instruction-tuned models
pub struct LLaMA3ChatModel {
    inner: LLaMA3ForCausalLM,
}

impl LLaMA3ChatModel {
    pub fn new(config: LLaMA3Config) -> Result<Self> {
        let inner = LLaMA3ForCausalLM::new(config)?;
        Ok(Self { inner })
    }

    pub fn config(&self) -> &LLaMA3Config {
        self.inner.config()
    }

    pub fn parameter_count(&self) -> usize {
        self.inner.parameter_count()
    }

    /// Run a forward pass over the given token IDs and return causal LM output.
    pub fn forward(&self, input_ids: Vec<u32>) -> Result<LLaMA3CausalLMOutput> {
        let logits = self.inner.forward(input_ids)?;
        Ok(LLaMA3CausalLMOutput { logits })
    }

    /// Apply the LLaMA-3 chat template and return the formatted prompt string.
    ///
    /// In production this would be tokenised and fed to `forward`; the string
    /// is returned here to allow inspection and testing of template formatting.
    pub fn build_prompt(&self, system_prompt: &str, messages: &[(String, String)]) -> String {
        format_llama3_chat(system_prompt, messages)
    }

    /// Greedy next-token selection from a logit tensor.
    pub fn greedy_next_token(&self, logits: &Tensor) -> Result<u32> {
        match logits {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                let vocab_size = if shape.len() >= 2 {
                    *shape.last().unwrap_or(&arr.len())
                } else {
                    arr.len()
                };
                let flat: Vec<f32> = arr.iter().copied().collect();
                let last_row = if flat.len() > vocab_size {
                    &flat[flat.len() - vocab_size..]
                } else {
                    &flat[..]
                };
                let best = last_row
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. format_llama3_chat: begin_of_text always present ──────────────────

    #[test]
    fn test_format_begins_with_begin_token() {
        let out = format_llama3_chat("", &[]);
        assert!(
            out.starts_with("<|begin_of_text|>"),
            "must start with begin token"
        );
    }

    // ── 2. format_llama3_chat: no system → no system block ───────────────────

    #[test]
    fn test_format_no_system_block_when_empty() {
        let out = format_llama3_chat("", &[]);
        assert!(!out.contains("system"), "no system block when empty");
    }

    // ── 3. format_llama3_chat: system block included when non-empty ──────────

    #[test]
    fn test_format_includes_system_block() {
        let out = format_llama3_chat("You are helpful.", &[]);
        assert!(out.contains("system"), "system role must appear");
        assert!(
            out.contains("You are helpful."),
            "system content must appear"
        );
        assert!(out.contains("<|eot_id|>"), "system block must end with eot");
    }

    // ── 4. format_llama3_chat: user message included ─────────────────────────

    #[test]
    fn test_format_user_message_present() {
        let msgs = vec![("user".to_string(), "Hello world".to_string())];
        let out = format_llama3_chat("", &msgs);
        assert!(out.contains("user"), "user role present");
        assert!(out.contains("Hello world"), "user content present");
    }

    // ── 5. format_llama3_chat: ends with open assistant turn ─────────────────

    #[test]
    fn test_format_ends_with_open_assistant_turn() {
        let msgs = vec![("user".to_string(), "Hi".to_string())];
        let out = format_llama3_chat("sys", &msgs);
        assert!(
            out.ends_with("<|end_header_id|>\n"),
            "must end with open assistant header"
        );
    }

    // ── 6. format_llama3_chat: assistant message included ────────────────────

    #[test]
    fn test_format_assistant_message() {
        let msgs = vec![
            ("user".to_string(), "question".to_string()),
            ("assistant".to_string(), "answer".to_string()),
        ];
        let out = format_llama3_chat("", &msgs);
        assert!(out.contains("question"), "user message in output");
        assert!(out.contains("answer"), "assistant message in output");
    }

    // ── 7. format_llama3_chat: multiple rounds ───────────────────────────────

    #[test]
    fn test_format_multiple_rounds() {
        let msgs = vec![
            ("user".to_string(), "turn 1".to_string()),
            ("assistant".to_string(), "reply 1".to_string()),
            ("user".to_string(), "turn 2".to_string()),
        ];
        let out = format_llama3_chat("sys", &msgs);
        let count = out.matches("<|start_header_id|>").count();
        // system + user + assistant + user + trailing assistant = 5
        assert_eq!(count, 5, "expected 5 header openings, got {count}");
    }

    // ── 8. format_llama3_chat: special tokens present ────────────────────────

    #[test]
    fn test_format_eot_tokens_count() {
        let msgs = vec![
            ("user".to_string(), "hello".to_string()),
            ("assistant".to_string(), "hi".to_string()),
        ];
        let out = format_llama3_chat("sys", &msgs);
        let eot_count = out.matches("<|eot_id|>").count();
        // system eot + user eot + assistant eot = 3
        assert_eq!(eot_count, 3, "expected 3 eot tokens, got {eot_count}");
    }

    // ── 9. format_llama3_chat: empty messages list ───────────────────────────

    #[test]
    fn test_format_empty_messages_only_system_and_assistant() {
        let out = format_llama3_chat("Sys", &[]);
        // system block + trailing assistant header
        assert!(out.contains("Sys"), "system content present");
        assert!(
            out.ends_with("<|end_header_id|>\n"),
            "trailing assistant header"
        );
    }

    // ── 10. format_llama3_chat: deterministic output ─────────────────────────

    #[test]
    fn test_format_deterministic() {
        let msgs = vec![("user".to_string(), "deterministic".to_string())];
        let a = format_llama3_chat("sys", &msgs);
        let b = format_llama3_chat("sys", &msgs);
        assert_eq!(a, b, "format must be deterministic");
    }

    // ── 11. LLaMA3Config::small_test creates valid config ────────────────────

    #[test]
    fn test_small_test_config_valid() {
        use trustformers_core::traits::Config;
        let cfg = LLaMA3Config::small_test();
        assert!(cfg.validate().is_ok(), "small_test config must be valid");
    }

    // ── 12. LLaMA3Config::small_test fields ──────────────────────────────────

    #[test]
    fn test_small_test_config_fields() {
        let cfg = LLaMA3Config::small_test();
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.num_hidden_layers, 2);
        assert!(cfg.vocab_size > 0);
    }

    // ── 13. LLaMA3ChatModel can be constructed with small config ─────────────

    #[test]
    fn test_chat_model_creation_small_config() {
        let cfg = LLaMA3Config::small_test();
        let model = LLaMA3ChatModel::new(cfg);
        assert!(model.is_ok(), "model construction must succeed");
    }

    // ── 14. parameter_count > 0 ──────────────────────────────────────────────

    #[test]
    fn test_chat_model_parameter_count_nonzero() {
        let cfg = LLaMA3Config::small_test();
        let model = LLaMA3ChatModel::new(cfg).unwrap_or_else(|_| panic!("model init failed"));
        assert!(model.parameter_count() > 0, "parameter count must be > 0");
    }

    // ── 15. config() returns the config used during construction ─────────────

    #[test]
    fn test_chat_model_config_accessor() {
        let cfg = LLaMA3Config::small_test();
        let hidden = cfg.hidden_size;
        let model = LLaMA3ChatModel::new(cfg).unwrap_or_else(|_| panic!("model init failed"));
        assert_eq!(model.config().hidden_size, hidden);
    }

    // ── 16. build_prompt returns same as format_llama3_chat ──────────────────

    #[test]
    fn test_build_prompt_matches_format_fn() {
        let cfg = LLaMA3Config::small_test();
        let model = LLaMA3ChatModel::new(cfg).unwrap_or_else(|_| panic!("model init failed"));
        let msgs = vec![("user".to_string(), "test".to_string())];
        let via_model = model.build_prompt("system", &msgs);
        let direct = format_llama3_chat("system", &msgs);
        assert_eq!(
            via_model, direct,
            "build_prompt must match format_llama3_chat"
        );
    }

    // ── 17. LLaMA3Config::llama3_8b head_dim computation ────────────────────

    #[test]
    fn test_llama3_8b_head_dim() {
        let cfg = LLaMA3Config::llama3_8b();
        let expected = cfg.hidden_size / cfg.num_attention_heads;
        assert_eq!(cfg.head_dim(), expected);
    }

    // ── 18. LLaMA3Config::llama3_8b uses_gqa ────────────────────────────────

    #[test]
    fn test_llama3_8b_uses_gqa() {
        let cfg = LLaMA3Config::llama3_8b();
        assert!(
            cfg.uses_gqa(),
            "8B model uses GQA (8 KV heads < 32 Q heads)"
        );
    }

    // ── 19. LLaMA3Config::llama3_70b num_query_groups ────────────────────────

    #[test]
    fn test_llama3_70b_query_groups() {
        let cfg = LLaMA3Config::llama3_70b();
        let expected = cfg.num_attention_heads / cfg.num_key_value_heads;
        assert_eq!(cfg.num_query_groups(), expected);
    }

    // ── 20. forward produces a tensor of correct shape ───────────────────────

    #[test]
    fn test_chat_model_forward_output_shape() {
        let cfg = LLaMA3Config::small_test();
        let model = LLaMA3ChatModel::new(cfg.clone()).unwrap_or_else(|_| panic!("init failed"));
        let input_ids = vec![1u32, 2, 3];
        let output = model.forward(input_ids);
        assert!(output.is_ok(), "forward must succeed");
        let out = output.unwrap_or_else(|_| panic!("forward failed"));
        // Tensor should be non-empty
        if let Tensor::F32(arr) = &out.logits {
            assert!(!arr.is_empty(), "logits must be non-empty");
        }
    }

    // ── 21. greedy_next_token returns index within vocab ─────────────────────

    #[test]
    fn test_greedy_next_token_within_vocab() {
        let cfg = LLaMA3Config::small_test();
        let model = LLaMA3ChatModel::new(cfg.clone()).unwrap_or_else(|_| panic!("init failed"));
        let input_ids = vec![1u32, 2];
        if let Ok(out) = model.forward(input_ids) {
            if let Ok(tok) = model.greedy_next_token(&out.logits) {
                assert!(
                    (tok as usize) < cfg.vocab_size,
                    "token must be within vocab"
                );
            }
        }
    }

    // ── 22. greedy_next_token on manual logits picks maximum ─────────────────

    #[test]
    fn test_greedy_next_token_picks_max() {
        let cfg = LLaMA3Config::small_test();
        let model = LLaMA3ChatModel::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        // Create a tensor where position 3 is largest
        let logits_vec = vec![0.1f32, 0.2, 0.1, 5.0, 0.1];
        let tensor =
            Tensor::from_vec(logits_vec, &[5]).unwrap_or_else(|_| panic!("tensor creation failed"));
        let tok = model.greedy_next_token(&tensor).unwrap_or(0);
        assert_eq!(tok, 3u32, "greedy must pick index 3 (highest logit)");
    }

    // ── 23. format_llama3_chat: system with special chars ────────────────────

    #[test]
    fn test_format_system_special_chars() {
        let sys = "Role: AI <&> entity\nLine2";
        let out = format_llama3_chat(sys, &[]);
        assert!(out.contains(sys), "system content must be verbatim");
    }

    // ── 24. forward output is finite ─────────────────────────────────────────

    #[test]
    fn test_forward_output_finite() {
        let cfg = LLaMA3Config::small_test();
        let model = LLaMA3ChatModel::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        let input_ids = vec![5u32];
        if let Ok(out) = model.forward(input_ids) {
            if let Tensor::F32(arr) = &out.logits {
                for &v in arr.iter() {
                    assert!(v.is_finite(), "logit {v} is not finite");
                }
            }
        }
    }

    // ── 25. LLaMA3CausalLMOutput struct field accessible ─────────────────────

    #[test]
    fn test_causal_lm_output_logits_accessible() {
        let cfg = LLaMA3Config::small_test();
        let model = LLaMA3ChatModel::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        if let Ok(out) = model.forward(vec![1u32]) {
            // Just accessing the field should not panic
            let _ = &out.logits;
        }
    }
}
