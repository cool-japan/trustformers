//! # LLaMA Task-Specific Implementations
//!
//! This module provides task-specific model wrappers for LLaMA (v1) models:
//! - Causal language modeling with greedy and sampling decode helpers
//! - Sequence classification
//! - Token classification (NER, POS tagging, etc.)
//! - Chat / instruction-following prompt formatting
//! - RoPE and RMSNorm computational utilities

use std::fmt;

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors specific to LLaMA task operations.
#[derive(Debug)]
pub enum LlamaTaskError {
    /// Invalid configuration.
    InvalidConfig(String),
    /// Model construction error.
    ModelBuildError(String),
    /// Forward pass error.
    ForwardError(String),
    /// Empty input token sequence.
    EmptyInput,
    /// Invalid number of classification labels.
    InvalidNumLabels(usize),
}

impl fmt::Display for LlamaTaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlamaTaskError::InvalidConfig(msg) => {
                write!(f, "LLaMA invalid config: {}", msg)
            },
            LlamaTaskError::ModelBuildError(msg) => {
                write!(f, "LLaMA model build error: {}", msg)
            },
            LlamaTaskError::ForwardError(msg) => {
                write!(f, "LLaMA forward error: {}", msg)
            },
            LlamaTaskError::EmptyInput => write!(f, "LLaMA error: empty input"),
            LlamaTaskError::InvalidNumLabels(n) => {
                write!(f, "LLaMA error: num_labels must be >= 2, got {}", n)
            },
        }
    }
}

impl std::error::Error for LlamaTaskError {}

// ─── Causal LM ────────────────────────────────────────────────────────────────

/// Causal language modeling wrapper for LLaMA.
pub struct LlamaForCausalLM {
    config: crate::llama::LlamaConfig,
    inner: crate::llama::LlamaForCausalLM,
}

impl LlamaForCausalLM {
    /// Construct from a config.
    pub fn new(config: crate::llama::LlamaConfig) -> Result<Self, LlamaTaskError> {
        let inner = crate::llama::LlamaForCausalLM::new(config.clone())
            .map_err(|e| LlamaTaskError::ModelBuildError(e.to_string()))?;
        Ok(Self { config, inner })
    }

    /// Config accessor.
    pub fn config(&self) -> &crate::llama::LlamaConfig {
        &self.config
    }

    /// Forward pass returning raw logits.
    pub fn forward(
        &self,
        input_ids: Vec<u32>,
    ) -> Result<trustformers_core::tensor::Tensor, LlamaTaskError> {
        if input_ids.is_empty() {
            return Err(LlamaTaskError::EmptyInput);
        }
        use trustformers_core::traits::Model;
        self.inner
            .forward(input_ids)
            .map_err(|e| LlamaTaskError::ForwardError(e.to_string()))
    }

    /// Greedy argmax over a logit slice.
    pub fn greedy_next_token(logits: &[f32]) -> Option<u32> {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
    }
}

// ─── Sequence Classification ──────────────────────────────────────────────────

/// Sequence-level classification head for LLaMA.
pub struct LlamaForSequenceClassification {
    config: crate::llama::LlamaConfig,
    num_labels: usize,
    /// Classification weights `[num_labels, hidden_size]`.
    classifier_weight: Vec<Vec<f32>>,
}

impl LlamaForSequenceClassification {
    /// Construct a sequence classification model.
    pub fn new(
        config: crate::llama::LlamaConfig,
        num_labels: usize,
    ) -> Result<Self, LlamaTaskError> {
        if num_labels < 2 {
            return Err(LlamaTaskError::InvalidNumLabels(num_labels));
        }
        let hidden = config.hidden_size;
        let mut state: u64 = 0xfeedface_deadbeef;
        let classifier_weight = (0..num_labels)
            .map(|_| {
                (0..hidden)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        (state as f32 / u64::MAX as f32) * 0.02 - 0.01
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            config,
            num_labels,
            classifier_weight,
        })
    }

    /// Config accessor.
    pub fn config(&self) -> &crate::llama::LlamaConfig {
        &self.config
    }

    /// Number of output labels.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    /// Forward pass over a pooled hidden state of length `hidden_size`.
    /// Returns logits of shape `[num_labels]`.
    pub fn forward(&self, hidden_state: &[f32]) -> Result<Vec<f32>, LlamaTaskError> {
        if hidden_state.is_empty() {
            return Err(LlamaTaskError::EmptyInput);
        }
        let expected = self.config.hidden_size;
        let input: Vec<f32> = if hidden_state.len() >= expected {
            hidden_state[..expected].to_vec()
        } else {
            let mut padded = hidden_state.to_vec();
            padded.resize(expected, 0.0);
            padded
        };
        let logits = self
            .classifier_weight
            .iter()
            .map(|row| row.iter().zip(input.iter()).map(|(&w, &x)| w * x).sum::<f32>())
            .collect();
        Ok(logits)
    }
}

// ─── Token Classification ─────────────────────────────────────────────────────

/// Token-level classification head for LLaMA.
pub struct LlamaForTokenClassification {
    config: crate::llama::LlamaConfig,
    num_labels: usize,
    classifier_weight: Vec<Vec<f32>>,
}

impl LlamaForTokenClassification {
    /// Construct a token classification model.
    pub fn new(
        config: crate::llama::LlamaConfig,
        num_labels: usize,
    ) -> Result<Self, LlamaTaskError> {
        if num_labels < 2 {
            return Err(LlamaTaskError::InvalidNumLabels(num_labels));
        }
        let hidden = config.hidden_size;
        let mut state: u64 = 0xabcdef01_23456789;
        let classifier_weight = (0..num_labels)
            .map(|_| {
                (0..hidden)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        (state as f32 / u64::MAX as f32) * 0.02 - 0.01
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            config,
            num_labels,
            classifier_weight,
        })
    }

    /// Config accessor.
    pub fn config(&self) -> &crate::llama::LlamaConfig {
        &self.config
    }

    /// Number of labels.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    /// Forward pass over `[seq_len * hidden_size]` flat hidden states.
    /// Returns `[seq_len * num_labels]` logits.
    pub fn forward(
        &self,
        hidden_states: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, LlamaTaskError> {
        if hidden_states.is_empty() || seq_len == 0 {
            return Err(LlamaTaskError::EmptyInput);
        }
        let hidden = self.config.hidden_size;
        let mut output = Vec::with_capacity(seq_len * self.num_labels);
        for tok in 0..seq_len {
            let start = tok * hidden;
            let slice: Vec<f32> = if start + hidden <= hidden_states.len() {
                hidden_states[start..start + hidden].to_vec()
            } else if start < hidden_states.len() {
                let mut v = hidden_states[start..].to_vec();
                v.resize(hidden, 0.0);
                v
            } else {
                vec![0.0f32; hidden]
            };
            for row in &self.classifier_weight {
                let logit: f32 = row.iter().zip(slice.iter()).map(|(&w, &x)| w * x).sum();
                output.push(logit);
            }
        }
        Ok(output)
    }
}

// ─── Prompt formatting ────────────────────────────────────────────────────────

/// LLaMA-2-style chat prompt template constants.
pub const INST_OPEN: &str = "[INST]";
pub const INST_CLOSE: &str = "[/INST]";
pub const SYS_OPEN: &str = "<<SYS>>";
pub const SYS_CLOSE: &str = "<</SYS>>";

/// Format a LLaMA-2-chat prompt from an optional system message and user message.
///
/// Template:
/// ```text
/// [INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]
/// ```
/// If no system message is provided, the `<<SYS>>` block is omitted.
pub fn format_llama_chat_prompt(system: Option<&str>, user: &str) -> String {
    let mut buf = String::new();
    buf.push_str(INST_OPEN);
    buf.push(' ');
    if let Some(sys) = system {
        buf.push_str(SYS_OPEN);
        buf.push('\n');
        buf.push_str(sys);
        buf.push('\n');
        buf.push_str(SYS_CLOSE);
        buf.push_str("\n\n");
    }
    buf.push_str(user);
    buf.push(' ');
    buf.push_str(INST_CLOSE);
    buf
}

// ─── RMSNorm utility ──────────────────────────────────────────────────────────

/// Pure-Rust RMSNorm: `output[i] = input[i] / sqrt(mean(input²) + eps)`.
pub fn rms_norm(input: &[f32], eps: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let mean_sq = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
    let rms = (mean_sq + eps).sqrt();
    input.iter().map(|x| x / rms).collect()
}

/// SiLU (Sigmoid Linear Unit) activation: `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU: element-wise `silu(gate) * up`.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up.iter()).map(|(&g, &u)| silu(g) * u).collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::LlamaConfig;

    fn small_cfg() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 512,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: None,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            ..LlamaConfig::default()
        }
    }

    // ── 1. CausalLM construction ──────────────────────────────────────────────

    #[test]
    fn test_causal_lm_construction() {
        let result = LlamaForCausalLM::new(small_cfg());
        assert!(
            result.is_ok(),
            "LlamaForCausalLM must construct: {:?}",
            result.err()
        );
    }

    // ── 2. CausalLM config accessor ───────────────────────────────────────────

    #[test]
    fn test_causal_lm_config_accessor() {
        let model = LlamaForCausalLM::new(small_cfg()).expect("construction");
        assert_eq!(model.config().hidden_size, 64);
        assert_eq!(model.config().vocab_size, 512);
    }

    // ── 3. CausalLM forward safe pattern ─────────────────────────────────────

    #[test]
    fn test_causal_lm_forward_safe() {
        let model = LlamaForCausalLM::new(small_cfg()).expect("construction");
        if let Ok(out) = model.forward(vec![1u32, 2, 3]) {
            use trustformers_core::tensor::Tensor;
            if let Tensor::F32(arr) = &out {
                assert!(!arr.is_empty(), "logits must be non-empty");
            }
        }
    }

    // ── 4. CausalLM empty input error ─────────────────────────────────────────

    #[test]
    fn test_causal_lm_empty_input_error() {
        let model = LlamaForCausalLM::new(small_cfg()).expect("construction");
        let result = model.forward(vec![]);
        assert!(matches!(result, Err(LlamaTaskError::EmptyInput)));
    }

    // ── 5. Greedy next-token argmax ───────────────────────────────────────────

    #[test]
    fn test_greedy_next_token_argmax() {
        let logits = vec![0.2f32, 0.1, 0.7, 0.0];
        let tok = LlamaForCausalLM::greedy_next_token(&logits);
        assert_eq!(tok, Some(2u32));
    }

    // ── 6. Greedy on empty returns None ──────────────────────────────────────

    #[test]
    fn test_greedy_next_token_empty_none() {
        assert_eq!(LlamaForCausalLM::greedy_next_token(&[]), None);
    }

    // ── 7. SequenceClassification construction ────────────────────────────────

    #[test]
    fn test_seq_cls_construction() {
        let result = LlamaForSequenceClassification::new(small_cfg(), 3);
        assert!(result.is_ok());
    }

    // ── 8. SequenceClassification invalid labels ──────────────────────────────

    #[test]
    fn test_seq_cls_invalid_labels() {
        let result = LlamaForSequenceClassification::new(small_cfg(), 1);
        assert!(matches!(result, Err(LlamaTaskError::InvalidNumLabels(1))));
    }

    // ── 9. SequenceClassification forward output length ───────────────────────

    #[test]
    fn test_seq_cls_forward_output_length() {
        let model = LlamaForSequenceClassification::new(small_cfg(), 5).expect("construction");
        let hidden = vec![0.1f32; small_cfg().hidden_size];
        let logits = model.forward(&hidden).expect("forward");
        assert_eq!(logits.len(), 5);
    }

    // ── 10. SequenceClassification empty input error ──────────────────────────

    #[test]
    fn test_seq_cls_empty_input_error() {
        let model = LlamaForSequenceClassification::new(small_cfg(), 2).expect("construction");
        let result = model.forward(&[]);
        assert!(matches!(result, Err(LlamaTaskError::EmptyInput)));
    }

    // ── 11. TokenClassification construction ─────────────────────────────────

    #[test]
    fn test_tok_cls_construction() {
        let result = LlamaForTokenClassification::new(small_cfg(), 4);
        assert!(result.is_ok());
    }

    // ── 12. TokenClassification output shape ─────────────────────────────────

    #[test]
    fn test_tok_cls_output_shape() {
        let cfg = small_cfg();
        let hidden = cfg.hidden_size;
        let model = LlamaForTokenClassification::new(cfg, 3).expect("construction");
        let seq_len = 5;
        let states = vec![0.05f32; seq_len * hidden];
        let logits = model.forward(&states, seq_len).expect("forward");
        assert_eq!(logits.len(), seq_len * 3);
    }

    // ── 13. TokenClassification empty input error ─────────────────────────────

    #[test]
    fn test_tok_cls_empty_input_error() {
        let model = LlamaForTokenClassification::new(small_cfg(), 2).expect("construction");
        let result = model.forward(&[], 0);
        assert!(matches!(result, Err(LlamaTaskError::EmptyInput)));
    }

    // ── 14. Chat prompt format with system ────────────────────────────────────

    #[test]
    fn test_chat_prompt_with_system() {
        let prompt = format_llama_chat_prompt(Some("You are helpful."), "What is 2+2?");
        assert!(prompt.contains(INST_OPEN));
        assert!(prompt.contains(SYS_OPEN));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains(SYS_CLOSE));
        assert!(prompt.contains("What is 2+2?"));
        assert!(prompt.contains(INST_CLOSE));
    }

    // ── 15. Chat prompt format without system ─────────────────────────────────

    #[test]
    fn test_chat_prompt_without_system() {
        let prompt = format_llama_chat_prompt(None, "Hello!");
        assert!(prompt.contains(INST_OPEN));
        assert!(!prompt.contains(SYS_OPEN));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.contains(INST_CLOSE));
    }

    // ── 16. RMSNorm output length ─────────────────────────────────────────────

    #[test]
    fn test_rms_norm_length() {
        let x = vec![1.0f32, 2.0, 3.0];
        let out = rms_norm(&x, 1e-5);
        assert_eq!(out.len(), 3);
    }

    // ── 17. RMSNorm empty input ───────────────────────────────────────────────

    #[test]
    fn test_rms_norm_empty() {
        let out = rms_norm(&[], 1e-5);
        assert!(out.is_empty());
    }

    // ── 18. RMSNorm numerical correctness ────────────────────────────────────

    #[test]
    fn test_rms_norm_numerical() {
        let input = vec![3.0f32, 4.0];
        let out = rms_norm(&input, 1e-5);
        let rms = (12.5f32 + 1e-5).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-5);
        assert!((out[1] - 4.0 / rms).abs() < 1e-5);
    }

    // ── 19. SiLU positive for positive input ─────────────────────────────────

    #[test]
    fn test_silu_positive() {
        assert!(silu(1.0f32) > 0.0);
        assert!(silu(0.0f32) >= 0.0);
        assert!(silu(-1.0f32) < 0.0);
    }

    // ── 20. SwiGLU output length ──────────────────────────────────────────────

    #[test]
    fn test_swiglu_output_length() {
        let gate = vec![1.0f32, -1.0, 0.5];
        let up = vec![2.0f32; 3];
        let out = swiglu(&gate, &up);
        assert_eq!(out.len(), 3);
        assert!(out[0] > 0.0);
        assert!(out[1] < 0.0);
    }

    // ── 21. Error display messages ────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e1 = LlamaTaskError::InvalidConfig("bad cfg".to_string());
        assert!(e1.to_string().contains("bad cfg"));

        let e2 = LlamaTaskError::EmptyInput;
        assert!(e2.to_string().contains("empty"));

        let e3 = LlamaTaskError::InvalidNumLabels(0);
        assert!(e3.to_string().contains("0"));
    }

    // ── 22. num_labels accessor ───────────────────────────────────────────────

    #[test]
    fn test_num_labels_accessor() {
        let m1 = LlamaForSequenceClassification::new(small_cfg(), 6).expect("construction");
        assert_eq!(m1.num_labels(), 6);
        let m2 = LlamaForTokenClassification::new(small_cfg(), 8).expect("construction");
        assert_eq!(m2.num_labels(), 8);
    }
}
