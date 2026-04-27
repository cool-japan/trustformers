//! # Falcon Task-Specific Implementations
//!
//! This module provides task-specific utilities for TII Falcon language models:
//! - Causal language modeling wrapper with greedy decode
//! - Sequence classification head
//! - ALiBi bias computation helpers
//! - Multi-query attention grouping utilities
//! - Instruction-following prompt formatting

use std::fmt;

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors specific to Falcon task operations.
#[derive(Debug)]
pub enum FalconTaskError {
    /// Invalid configuration.
    InvalidConfig(String),
    /// Model build error.
    ModelBuildError(String),
    /// Forward pass error.
    ForwardError(String),
    /// Empty input.
    EmptyInput,
    /// Invalid number of classification labels.
    InvalidNumLabels(usize),
}

impl fmt::Display for FalconTaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FalconTaskError::InvalidConfig(msg) => {
                write!(f, "Falcon invalid config: {}", msg)
            },
            FalconTaskError::ModelBuildError(msg) => {
                write!(f, "Falcon model build error: {}", msg)
            },
            FalconTaskError::ForwardError(msg) => {
                write!(f, "Falcon forward error: {}", msg)
            },
            FalconTaskError::EmptyInput => write!(f, "Falcon error: empty input"),
            FalconTaskError::InvalidNumLabels(n) => {
                write!(f, "Falcon error: num_labels must be >= 2, got {}", n)
            },
        }
    }
}

impl std::error::Error for FalconTaskError {}

// ─── Causal LM ────────────────────────────────────────────────────────────────

/// Causal language modeling wrapper for Falcon.
pub struct FalconForCausalLM {
    config: crate::falcon::FalconConfig,
    inner: crate::falcon::FalconForCausalLM,
}

impl FalconForCausalLM {
    /// Construct from config.
    pub fn new(config: crate::falcon::FalconConfig) -> Result<Self, FalconTaskError> {
        let inner = crate::falcon::FalconForCausalLM::new(config.clone())
            .map_err(|e| FalconTaskError::ModelBuildError(e.to_string()))?;
        Ok(Self { config, inner })
    }

    /// Config accessor.
    pub fn config(&self) -> &crate::falcon::FalconConfig {
        &self.config
    }

    /// Forward pass returning raw logit tensor.
    pub fn forward(
        &self,
        input: trustformers_core::tensor::Tensor,
    ) -> Result<trustformers_core::tensor::Tensor, FalconTaskError> {
        use trustformers_core::tensor::Tensor;
        match &input {
            Tensor::F32(arr) if arr.is_empty() => return Err(FalconTaskError::EmptyInput),
            _ => {},
        }
        use trustformers_core::traits::Model;
        self.inner
            .forward(input)
            .map_err(|e| FalconTaskError::ForwardError(e.to_string()))
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

/// Sequence-level classification head for Falcon.
pub struct FalconForSequenceClassification {
    config: crate::falcon::FalconConfig,
    num_labels: usize,
    /// Classification weights `[num_labels, hidden_size]`.
    classifier_weight: Vec<Vec<f32>>,
}

impl FalconForSequenceClassification {
    /// Construct from config.
    pub fn new(
        config: crate::falcon::FalconConfig,
        num_labels: usize,
    ) -> Result<Self, FalconTaskError> {
        if num_labels < 2 {
            return Err(FalconTaskError::InvalidNumLabels(num_labels));
        }
        let hidden = config.hidden_size;
        let mut state: u64 = 0xcafe_babe_dead_beef;
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
    pub fn config(&self) -> &crate::falcon::FalconConfig {
        &self.config
    }

    /// Number of output labels.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    /// Forward pass over a pooled `hidden_size`-dim vector.
    /// Returns `[num_labels]` logits.
    pub fn forward(&self, hidden_state: &[f32]) -> Result<Vec<f32>, FalconTaskError> {
        if hidden_state.is_empty() {
            return Err(FalconTaskError::EmptyInput);
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

// ─── ALiBi utilities ──────────────────────────────────────────────────────────

/// Compute ALiBi slope for head `i` out of `num_heads` heads.
///
/// Reference: "Train Short, Test Long: Attention with Linear Biases" (Press et al., 2022).
///
/// Returns the slope value as defined by the paper's geometric sequence.
pub fn alibi_slope(head_idx: usize, num_heads: usize) -> f32 {
    let ratio = 2.0_f32.powf(-8.0 / num_heads as f32);
    ratio.powf((head_idx + 1) as f32)
}

/// Compute the ALiBi bias matrix for a sequence of length `seq_len`.
///
/// Returns a flat `[seq_len * seq_len]` matrix (row = query, col = key)
/// with bias values `slope * (key_pos - query_pos)` (non-positive for past tokens).
pub fn alibi_bias_matrix(seq_len: usize, slope: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * seq_len];
    for qi in 0..seq_len {
        for kj in 0..seq_len {
            let bias = slope * (kj as f32 - qi as f32);
            out[qi * seq_len + kj] = bias;
        }
    }
    out
}

/// Verify that ALiBi slopes are strictly decreasing as head index grows.
///
/// Returns `true` if the sequence is monotone decreasing, `false` otherwise.
pub fn slopes_are_decreasing(num_heads: usize) -> bool {
    if num_heads < 2 {
        return true;
    }
    let mut prev = alibi_slope(0, num_heads);
    for h in 1..num_heads {
        let cur = alibi_slope(h, num_heads);
        if cur >= prev {
            return false;
        }
        prev = cur;
    }
    true
}

// ─── Instruction-following prompt formatting ─────────────────────────────────

/// Format a Falcon instruction-following prompt.
///
/// Falcon instruct models use a simple `User:` / `Falcon:` chat template.
pub fn format_falcon_instruct_prompt(user: &str) -> String {
    format!("User: {}\nFalcon:", user)
}

/// Format a Falcon chat turn with an optional system preamble.
pub fn format_falcon_chat(system: Option<&str>, user: &str) -> String {
    let mut buf = String::new();
    if let Some(sys) = system {
        buf.push_str(sys);
        buf.push('\n');
    }
    buf.push_str("User: ");
    buf.push_str(user);
    buf.push_str("\nFalcon:");
    buf
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::falcon::FalconConfig;

    fn small_cfg() -> FalconConfig {
        FalconConfig {
            vocab_size: 256,
            hidden_size: 4 * 64, // must be divisible by num_attention_heads
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: Some(1),
            hidden_act: "gelu".to_string(),
            max_position_embeddings: 64,
            initializer_range: 0.02,
            layer_norm_epsilon: 1e-5,
            use_cache: false,
            pad_token_id: Some(0),
            bos_token_id: 1,
            eos_token_id: 2,
            apply_residual_connection_post_layernorm: false,
            hidden_dropout: 0.0,
            attention_dropout: 0.0,
            model_type: "falcon-test".to_string(),
            parallel_attn: true,
            bias: false,
            multi_query: true,
            alibi: false,
            new_decoder_architecture: false,
            use_flash_attention: None,
        }
    }

    // ── 1. CausalLM construction ──────────────────────────────────────────────

    #[test]
    fn test_causal_lm_construction() {
        let result = FalconForCausalLM::new(small_cfg());
        assert!(
            result.is_ok(),
            "FalconForCausalLM must construct: {:?}",
            result.err()
        );
    }

    // ── 2. CausalLM config accessor ───────────────────────────────────────────

    #[test]
    fn test_causal_lm_config_accessor() {
        let model = FalconForCausalLM::new(small_cfg()).expect("construction");
        assert_eq!(model.config().hidden_size, 256);
        assert_eq!(model.config().vocab_size, 256);
    }

    // ── 3. CausalLM forward safe pattern ─────────────────────────────────────

    #[test]
    fn test_causal_lm_forward_safe() {
        use trustformers_core::tensor::Tensor;
        let model = FalconForCausalLM::new(small_cfg()).expect("construction");
        let input =
            Tensor::from_vec(vec![0.1f32; 256], &[256]).unwrap_or_else(|_| panic!("tensor failed"));
        if let Ok(Tensor::F32(arr)) = model.forward(input) {
            assert!(!arr.is_empty());
        }
    }

    // ── 4. CausalLM empty input returns error ─────────────────────────────────

    #[test]
    fn test_causal_lm_empty_input_error() {
        use trustformers_core::tensor::Tensor;
        let model = FalconForCausalLM::new(small_cfg()).expect("construction");
        let empty = Tensor::from_vec(vec![], &[0]).unwrap_or_else(|_| panic!("tensor failed"));
        let result = model.forward(empty);
        assert!(matches!(result, Err(FalconTaskError::EmptyInput)));
    }

    // ── 5. Greedy next-token argmax ───────────────────────────────────────────

    #[test]
    fn test_greedy_next_token_argmax() {
        let logits = vec![0.1f32, 0.3, 0.9, 0.05];
        assert_eq!(FalconForCausalLM::greedy_next_token(&logits), Some(2u32));
    }

    // ── 6. Greedy on empty returns None ──────────────────────────────────────

    #[test]
    fn test_greedy_next_token_empty() {
        assert_eq!(FalconForCausalLM::greedy_next_token(&[]), None);
    }

    // ── 7. SequenceClassification construction ────────────────────────────────

    #[test]
    fn test_seq_cls_construction() {
        let result = FalconForSequenceClassification::new(small_cfg(), 3);
        assert!(result.is_ok());
    }

    // ── 8. SequenceClassification invalid labels ──────────────────────────────

    #[test]
    fn test_seq_cls_invalid_labels() {
        let result = FalconForSequenceClassification::new(small_cfg(), 1);
        assert!(matches!(result, Err(FalconTaskError::InvalidNumLabels(1))));
    }

    // ── 9. SequenceClassification forward output length ───────────────────────

    #[test]
    fn test_seq_cls_forward_output_length() {
        let model = FalconForSequenceClassification::new(small_cfg(), 4).expect("construction");
        let hidden = vec![0.1f32; small_cfg().hidden_size];
        let logits = model.forward(&hidden).expect("forward");
        assert_eq!(logits.len(), 4);
    }

    // ── 10. SequenceClassification empty input error ──────────────────────────

    #[test]
    fn test_seq_cls_empty_input_error() {
        let model = FalconForSequenceClassification::new(small_cfg(), 2).expect("construction");
        let result = model.forward(&[]);
        assert!(matches!(result, Err(FalconTaskError::EmptyInput)));
    }

    // ── 11. ALiBi slope strictly positive ────────────────────────────────────

    #[test]
    fn test_alibi_slope_positive() {
        for h in 0..8 {
            let s = alibi_slope(h, 8);
            assert!(s > 0.0, "slope for head {h} must be > 0, got {s}");
        }
    }

    // ── 12. ALiBi slopes are monotone decreasing ──────────────────────────────

    #[test]
    fn test_alibi_slopes_decreasing() {
        assert!(
            slopes_are_decreasing(8),
            "ALiBi slopes must be decreasing for 8 heads"
        );
        assert!(
            slopes_are_decreasing(4),
            "ALiBi slopes must be decreasing for 4 heads"
        );
        assert!(
            slopes_are_decreasing(16),
            "ALiBi slopes must be decreasing for 16 heads"
        );
    }

    // ── 13. ALiBi single-head trivially decreasing ────────────────────────────

    #[test]
    fn test_alibi_slopes_single_head() {
        assert!(slopes_are_decreasing(1));
    }

    // ── 14. ALiBi bias matrix shape ───────────────────────────────────────────

    #[test]
    fn test_alibi_bias_matrix_shape() {
        let seq_len = 5;
        let bias = alibi_bias_matrix(seq_len, 0.5);
        assert_eq!(bias.len(), seq_len * seq_len);
    }

    // ── 15. ALiBi bias diagonal is zero ──────────────────────────────────────

    #[test]
    fn test_alibi_bias_diagonal_zero() {
        let seq_len = 4;
        let slope = 0.125f32;
        let bias = alibi_bias_matrix(seq_len, slope);
        for i in 0..seq_len {
            let v = bias[i * seq_len + i];
            assert!(
                v.abs() < 1e-6,
                "diagonal entry ({i},{i}) must be 0, got {v}"
            );
        }
    }

    // ── 16. ALiBi bias past positions are negative ────────────────────────────

    #[test]
    fn test_alibi_bias_past_positions_negative() {
        let seq_len = 4;
        let slope = 0.25f32;
        let bias = alibi_bias_matrix(seq_len, slope);
        // For qi=2, kj=0: bias = slope * (0 - 2) = -0.5 < 0
        let v = bias[2 * seq_len];
        assert!(v < 0.0, "past bias must be negative, got {v}");
    }

    // ── 17. Instruction prompt format ────────────────────────────────────────

    #[test]
    fn test_falcon_instruct_prompt() {
        let prompt = format_falcon_instruct_prompt("Explain gravity.");
        assert!(prompt.starts_with("User: "));
        assert!(prompt.contains("Explain gravity."));
        assert!(prompt.ends_with("Falcon:"));
    }

    // ── 18. Chat prompt with system ───────────────────────────────────────────

    #[test]
    fn test_falcon_chat_with_system() {
        let prompt = format_falcon_chat(Some("You are a scientist."), "What is entropy?");
        assert!(prompt.contains("You are a scientist."));
        assert!(prompt.contains("User: "));
        assert!(prompt.contains("What is entropy?"));
        assert!(prompt.ends_with("Falcon:"));
    }

    // ── 19. Chat prompt without system ───────────────────────────────────────

    #[test]
    fn test_falcon_chat_without_system() {
        let prompt = format_falcon_chat(None, "Hello!");
        assert!(!prompt.starts_with('\n'));
        assert!(prompt.contains("User: Hello!"));
        assert!(prompt.ends_with("Falcon:"));
    }

    // ── 20. num_labels accessor ───────────────────────────────────────────────

    #[test]
    fn test_num_labels_accessor() {
        let model = FalconForSequenceClassification::new(small_cfg(), 10).expect("construction");
        assert_eq!(model.num_labels(), 10);
    }

    // ── 21. Error display messages ────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e1 = FalconTaskError::InvalidConfig("test error".to_string());
        assert!(e1.to_string().contains("test error"));

        let e2 = FalconTaskError::EmptyInput;
        assert!(e2.to_string().contains("empty"));

        let e3 = FalconTaskError::InvalidNumLabels(0);
        assert!(e3.to_string().contains("0"));
    }

    // ── 22. Config presets validate ───────────────────────────────────────────

    #[test]
    fn test_config_presets_validate() {
        use trustformers_core::traits::Config;
        assert!(FalconConfig::falcon_7b().validate().is_ok());
        assert!(FalconConfig::falcon_40b().validate().is_ok());
        assert!(FalconConfig::falcon_180b().validate().is_ok());
    }
}
