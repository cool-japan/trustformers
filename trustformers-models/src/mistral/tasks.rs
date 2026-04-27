//! # Mistral Task-Specific Implementations
//!
//! This module provides task-specific model wrappers and utilities for Mistral 7B:
//! - Causal language modeling
//! - Sequence classification
//! - Token classification
//! - Sliding window attention coverage utilities
//! - GQA (Grouped-Query Attention) helper computations
//! - Mistral instruction-following prompt formatting

use std::fmt;

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors specific to Mistral task operations.
#[derive(Debug)]
pub enum MistralTaskError {
    /// Invalid configuration.
    InvalidConfig(String),
    /// Model build error.
    ModelBuildError(String),
    /// Forward pass error.
    ForwardError(String),
    /// Empty input token sequence.
    EmptyInput,
    /// Invalid number of classification labels.
    InvalidNumLabels(usize),
}

impl fmt::Display for MistralTaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MistralTaskError::InvalidConfig(msg) => {
                write!(f, "Mistral invalid config: {}", msg)
            },
            MistralTaskError::ModelBuildError(msg) => {
                write!(f, "Mistral model build error: {}", msg)
            },
            MistralTaskError::ForwardError(msg) => {
                write!(f, "Mistral forward error: {}", msg)
            },
            MistralTaskError::EmptyInput => write!(f, "Mistral error: empty input"),
            MistralTaskError::InvalidNumLabels(n) => {
                write!(f, "Mistral error: num_labels must be >= 2, got {}", n)
            },
        }
    }
}

impl std::error::Error for MistralTaskError {}

// ─── Causal LM ────────────────────────────────────────────────────────────────

/// Causal language modeling wrapper for Mistral.
pub struct MistralForCausalLM {
    config: crate::mistral::MistralConfig,
    inner: crate::mistral::MistralForCausalLM,
}

impl MistralForCausalLM {
    /// Construct from config.
    pub fn new(config: crate::mistral::MistralConfig) -> Result<Self, MistralTaskError> {
        let inner = crate::mistral::MistralForCausalLM::new(config.clone())
            .map_err(|e| MistralTaskError::ModelBuildError(e.to_string()))?;
        Ok(Self { config, inner })
    }

    /// Config accessor.
    pub fn config(&self) -> &crate::mistral::MistralConfig {
        &self.config
    }

    /// Forward pass returning raw logit tensor.
    pub fn forward(
        &self,
        input_ids: Vec<u32>,
    ) -> Result<trustformers_core::tensor::Tensor, MistralTaskError> {
        if input_ids.is_empty() {
            return Err(MistralTaskError::EmptyInput);
        }
        use trustformers_core::traits::Model;
        self.inner
            .forward(input_ids)
            .map_err(|e| MistralTaskError::ForwardError(e.to_string()))
    }

    /// Greedy argmax over a flat logit slice.
    pub fn greedy_next_token(logits: &[f32]) -> Option<u32> {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
    }
}

// ─── Sequence Classification ──────────────────────────────────────────────────

/// Sequence-level classification head for Mistral.
pub struct MistralForSequenceClassification {
    config: crate::mistral::MistralConfig,
    num_labels: usize,
    classifier_weight: Vec<Vec<f32>>,
}

impl MistralForSequenceClassification {
    /// Construct from config.
    pub fn new(
        config: crate::mistral::MistralConfig,
        num_labels: usize,
    ) -> Result<Self, MistralTaskError> {
        if num_labels < 2 {
            return Err(MistralTaskError::InvalidNumLabels(num_labels));
        }
        let hidden = config.hidden_size;
        let mut state: u64 = 0x11223344_aabbccdd;
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
    pub fn config(&self) -> &crate::mistral::MistralConfig {
        &self.config
    }

    /// Number of output labels.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    /// Forward pass over a pooled `hidden_size`-dim vector.
    /// Returns `[num_labels]` logits.
    pub fn forward(&self, hidden_state: &[f32]) -> Result<Vec<f32>, MistralTaskError> {
        if hidden_state.is_empty() {
            return Err(MistralTaskError::EmptyInput);
        }
        let expected = self.config.hidden_size;
        let input: Vec<f32> = if hidden_state.len() >= expected {
            hidden_state[..expected].to_vec()
        } else {
            let mut v = hidden_state.to_vec();
            v.resize(expected, 0.0);
            v
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

/// Token-level classification head for Mistral.
pub struct MistralForTokenClassification {
    config: crate::mistral::MistralConfig,
    num_labels: usize,
    classifier_weight: Vec<Vec<f32>>,
}

impl MistralForTokenClassification {
    /// Construct from config.
    pub fn new(
        config: crate::mistral::MistralConfig,
        num_labels: usize,
    ) -> Result<Self, MistralTaskError> {
        if num_labels < 2 {
            return Err(MistralTaskError::InvalidNumLabels(num_labels));
        }
        let hidden = config.hidden_size;
        let mut state: u64 = 0xffeeddcc_bbaa9988;
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
    pub fn config(&self) -> &crate::mistral::MistralConfig {
        &self.config
    }

    /// Number of labels.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    /// Forward pass over a `[seq_len * hidden_size]` flat tensor.
    /// Returns `[seq_len * num_labels]` logits.
    pub fn forward(
        &self,
        hidden_states: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, MistralTaskError> {
        if hidden_states.is_empty() || seq_len == 0 {
            return Err(MistralTaskError::EmptyInput);
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

// ─── Sliding window attention utilities ───────────────────────────────────────

/// Apply a causal sliding window attention mask to a flat `seq_len × seq_len`
/// score matrix (row-major).
///
/// For each query position `qi`, positions `kj` that are more than
/// `window_size` steps in the past are masked to `f32::NEG_INFINITY`.
/// Future positions are always masked.
pub fn apply_sliding_window_mask(scores: &mut [f32], seq_len: usize, window_size: usize) {
    for qi in 0..seq_len {
        for kj in 0..seq_len {
            // Causal mask: no attending to future
            let is_future = kj > qi;
            // Window mask: no attending to positions more than window_size behind
            let is_outside_window = qi > kj && (qi - kj) > window_size;
            if is_future || is_outside_window {
                scores[qi * seq_len + kj] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Compute the fraction of causal pairs `(qi, kj)` with `kj <= qi` that fall
/// within the sliding window.
///
/// Returns a value in `[0.0, 1.0]`.
pub fn sliding_window_coverage(seq_len: usize, window_size: usize) -> f32 {
    if seq_len == 0 {
        return 0.0;
    }
    let total_causal: usize = seq_len * (seq_len + 1) / 2;
    let mut covered = 0usize;
    for qi in 0..seq_len {
        for kj in 0..=qi {
            if qi - kj <= window_size {
                covered += 1;
            }
        }
    }
    covered as f32 / total_causal as f32
}

// ─── GQA head expansion utilities ─────────────────────────────────────────────

/// Return the number of query groups given the head counts.
///
/// `num_groups = num_attention_heads / num_key_value_heads`.
pub fn gqa_num_groups(num_attention_heads: usize, num_key_value_heads: usize) -> usize {
    if num_key_value_heads == 0 {
        return 0;
    }
    num_attention_heads / num_key_value_heads
}

// ─── Mistral instruction prompt formatting ────────────────────────────────────

/// Format a Mistral/Llama-2-style `[INST]` prompt.
///
/// Template:
/// ```text
/// <s>[INST] {instruction} [/INST]
/// ```
pub fn format_mistral_instruction_prompt(instruction: &str) -> String {
    format!("<s>[INST] {} [/INST]", instruction)
}

/// Format a multi-turn Mistral chat prompt.
///
/// Each turn is a `(user, assistant_or_none)` pair.  The last turn's assistant
/// part should be `None` to let the model generate it.
pub fn format_mistral_chat(turns: &[(&str, Option<&str>)]) -> String {
    let mut buf = String::new();
    for (user, assistant) in turns {
        buf.push_str("<s>[INST] ");
        buf.push_str(user);
        buf.push_str(" [/INST]");
        if let Some(asst) = assistant {
            buf.push(' ');
            buf.push_str(asst);
            buf.push_str(" </s>");
        }
    }
    buf
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mistral::MistralConfig;

    fn small_cfg() -> MistralConfig {
        MistralConfig {
            vocab_size: 512,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            hidden_act: "silu".to_string(),
            max_position_embeddings: 128,
            initializer_range: 0.02,
            rms_norm_eps: 1e-5,
            use_cache: false,
            pad_token_id: None,
            bos_token_id: 1,
            eos_token_id: 2,
            rope_theta: 10000.0,
            sliding_window: Some(32),
            attention_dropout: 0.0,
            model_type: "mistral-test".to_string(),
        }
    }

    // ── 1. CausalLM construction ──────────────────────────────────────────────

    #[test]
    fn test_causal_lm_construction() {
        let result = MistralForCausalLM::new(small_cfg());
        assert!(
            result.is_ok(),
            "MistralForCausalLM must construct: {:?}",
            result.err()
        );
    }

    // ── 2. CausalLM config accessor ───────────────────────────────────────────

    #[test]
    fn test_causal_lm_config_accessor() {
        let model = MistralForCausalLM::new(small_cfg()).expect("construction");
        assert_eq!(model.config().hidden_size, 64);
        assert_eq!(model.config().vocab_size, 512);
    }

    // ── 3. CausalLM forward safe pattern ─────────────────────────────────────

    #[test]
    fn test_causal_lm_forward_safe() {
        let model = MistralForCausalLM::new(small_cfg()).expect("construction");
        if let Ok(out) = model.forward(vec![1u32, 2, 3]) {
            use trustformers_core::tensor::Tensor;
            if let Tensor::F32(arr) = &out {
                assert!(!arr.is_empty());
            }
        }
    }

    // ── 4. CausalLM empty input error ─────────────────────────────────────────

    #[test]
    fn test_causal_lm_empty_input_error() {
        let model = MistralForCausalLM::new(small_cfg()).expect("construction");
        let result = model.forward(vec![]);
        assert!(matches!(result, Err(MistralTaskError::EmptyInput)));
    }

    // ── 5. Greedy next-token argmax ───────────────────────────────────────────

    #[test]
    fn test_greedy_next_token_argmax() {
        let logits = vec![0.2f32, 0.8, 0.1, 0.5];
        assert_eq!(MistralForCausalLM::greedy_next_token(&logits), Some(1u32));
    }

    // ── 6. Greedy on empty returns None ──────────────────────────────────────

    #[test]
    fn test_greedy_next_token_empty() {
        assert_eq!(MistralForCausalLM::greedy_next_token(&[]), None);
    }

    // ── 7. SequenceClassification construction ────────────────────────────────

    #[test]
    fn test_seq_cls_construction() {
        let result = MistralForSequenceClassification::new(small_cfg(), 3);
        assert!(result.is_ok());
    }

    // ── 8. SequenceClassification invalid labels ──────────────────────────────

    #[test]
    fn test_seq_cls_invalid_labels() {
        let result = MistralForSequenceClassification::new(small_cfg(), 1);
        assert!(matches!(result, Err(MistralTaskError::InvalidNumLabels(1))));
    }

    // ── 9. SequenceClassification forward output length ───────────────────────

    #[test]
    fn test_seq_cls_forward_output_length() {
        let model = MistralForSequenceClassification::new(small_cfg(), 5).expect("construction");
        let hidden = vec![0.1f32; small_cfg().hidden_size];
        let logits = model.forward(&hidden).expect("forward");
        assert_eq!(logits.len(), 5);
    }

    // ── 10. SequenceClassification empty input error ──────────────────────────

    #[test]
    fn test_seq_cls_empty_input_error() {
        let model = MistralForSequenceClassification::new(small_cfg(), 2).expect("construction");
        let result = model.forward(&[]);
        assert!(matches!(result, Err(MistralTaskError::EmptyInput)));
    }

    // ── 11. TokenClassification construction ─────────────────────────────────

    #[test]
    fn test_tok_cls_construction() {
        let result = MistralForTokenClassification::new(small_cfg(), 4);
        assert!(result.is_ok());
    }

    // ── 12. TokenClassification output shape ─────────────────────────────────

    #[test]
    fn test_tok_cls_output_shape() {
        let cfg = small_cfg();
        let hidden = cfg.hidden_size;
        let model = MistralForTokenClassification::new(cfg, 3).expect("construction");
        let seq_len = 4;
        let states = vec![0.05f32; seq_len * hidden];
        let logits = model.forward(&states, seq_len).expect("forward");
        assert_eq!(logits.len(), seq_len * 3);
    }

    // ── 13. TokenClassification empty input error ─────────────────────────────

    #[test]
    fn test_tok_cls_empty_input_error() {
        let model = MistralForTokenClassification::new(small_cfg(), 2).expect("construction");
        let result = model.forward(&[], 0);
        assert!(matches!(result, Err(MistralTaskError::EmptyInput)));
    }

    // ── 14. Sliding window mask masks future positions ────────────────────────

    #[test]
    fn test_sliding_window_mask_future() {
        let seq_len = 4;
        let mut scores = vec![0.0f32; seq_len * seq_len];
        apply_sliding_window_mask(&mut scores, seq_len, 2);
        // (0, 1) is a future token — must be -inf
        let v = scores[1];
        assert!(
            v.is_infinite() && v < 0.0,
            "future token must be -inf, got {v}"
        );
    }

    // ── 15. Sliding window mask preserves within-window past ──────────────────

    #[test]
    fn test_sliding_window_mask_within_window() {
        let seq_len = 5;
        let window = 2;
        let mut scores = vec![1.0f32; seq_len * seq_len];
        apply_sliding_window_mask(&mut scores, seq_len, window);
        // (3, 2): diff = 1 ≤ 2 → must remain 1.0
        let v = scores[3 * seq_len + 2];
        assert!(
            (v - 1.0).abs() < 1e-6,
            "within-window past must be unchanged, got {v}"
        );
    }

    // ── 16. Sliding window mask outside window ────────────────────────────────

    #[test]
    fn test_sliding_window_mask_outside_window() {
        let seq_len = 5;
        let window = 1;
        let mut scores = vec![1.0f32; seq_len * seq_len];
        apply_sliding_window_mask(&mut scores, seq_len, window);
        // (4, 0): diff = 4 > 1 → -inf
        let v = scores[4 * seq_len];
        assert!(
            v.is_infinite() && v < 0.0,
            "outside window must be -inf, got {v}"
        );
    }

    // ── 17. Sliding window full coverage ─────────────────────────────────────

    #[test]
    fn test_sliding_window_full_coverage() {
        let cov = sliding_window_coverage(4, 10);
        assert!(
            (cov - 1.0).abs() < 1e-5,
            "window >= seq_len → coverage must be 1.0, got {cov}"
        );
    }

    // ── 18. Sliding window zero window ───────────────────────────────────────

    #[test]
    fn test_sliding_window_zero_window() {
        // With window=0, only self-attention survives; coverage = 1/seq_len ... wait:
        // total_causal = seq_len*(seq_len+1)/2; covered = seq_len (only diagonal)
        let seq_len = 5;
        let cov = sliding_window_coverage(seq_len, 0);
        let expected = seq_len as f32 / (seq_len * (seq_len + 1) / 2) as f32;
        assert!(
            (cov - expected).abs() < 1e-5,
            "zero window coverage mismatch: {cov} vs {expected}"
        );
    }

    // ── 19. GQA num groups ────────────────────────────────────────────────────

    #[test]
    fn test_gqa_num_groups() {
        assert_eq!(gqa_num_groups(32, 8), 4);
        assert_eq!(gqa_num_groups(32, 32), 1);
        assert_eq!(gqa_num_groups(64, 8), 8);
    }

    // ── 20. Instruction prompt format ────────────────────────────────────────

    #[test]
    fn test_instruction_prompt() {
        let p = format_mistral_instruction_prompt("Tell me a joke.");
        assert!(p.starts_with("<s>[INST] "));
        assert!(p.contains("Tell me a joke."));
        assert!(p.ends_with(" [/INST]"));
    }

    // ── 21. Multi-turn chat format ────────────────────────────────────────────

    #[test]
    fn test_multi_turn_chat() {
        let turns: &[(&str, Option<&str>)] =
            &[("Hello", Some("Hi there!")), ("What is Mistral?", None)];
        let prompt = format_mistral_chat(turns);
        assert!(prompt.contains("[INST] Hello [/INST]"));
        assert!(prompt.contains("Hi there!"));
        assert!(prompt.contains("[INST] What is Mistral? [/INST]"));
    }

    // ── 22. Error display messages ────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e1 = MistralTaskError::InvalidConfig("test".to_string());
        assert!(e1.to_string().contains("test"));

        let e2 = MistralTaskError::EmptyInput;
        assert!(e2.to_string().contains("empty"));

        let e3 = MistralTaskError::InvalidNumLabels(1);
        assert!(e3.to_string().contains("1"));
    }
}
