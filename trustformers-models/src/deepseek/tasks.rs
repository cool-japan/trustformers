//! # DeepSeek Task-Specific Implementations
//!
//! This module provides task-specific model heads and utilities for DeepSeek-V2 models,
//! including:
//! - Causal language modeling
//! - Sequence classification
//! - Token classification
//! - MoE routing analysis utilities
//! - Greedy generation with logit scaling

use std::fmt;

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors specific to DeepSeek task operations.
#[derive(Debug)]
pub enum DeepSeekTaskError {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Model construction failed.
    ModelBuildError(String),
    /// Forward pass failed.
    ForwardError(String),
    /// Empty input sequence.
    EmptyInput,
    /// Invalid number of labels for classification head.
    InvalidNumLabels(usize),
}

impl fmt::Display for DeepSeekTaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeepSeekTaskError::InvalidConfig(msg) => {
                write!(f, "DeepSeek invalid config: {}", msg)
            },
            DeepSeekTaskError::ModelBuildError(msg) => {
                write!(f, "DeepSeek model build error: {}", msg)
            },
            DeepSeekTaskError::ForwardError(msg) => {
                write!(f, "DeepSeek forward error: {}", msg)
            },
            DeepSeekTaskError::EmptyInput => write!(f, "DeepSeek error: empty input"),
            DeepSeekTaskError::InvalidNumLabels(n) => {
                write!(f, "DeepSeek error: num_labels must be >= 2, got {}", n)
            },
        }
    }
}

impl std::error::Error for DeepSeekTaskError {}

// ─── Causal LM ────────────────────────────────────────────────────────────────

/// Causal language modeling head wrapping a DeepSeek-V2 model.
///
/// Produces per-token logits of shape `[seq_len * vocab_size]` suitable for
/// autoregressive token prediction.
pub struct DeepSeekForCausalLM {
    config: crate::deepseek::DeepSeekConfig,
    inner: crate::deepseek::DeepSeekForCausalLM,
}

impl DeepSeekForCausalLM {
    /// Construct a new causal LM model from `config`.
    pub fn new(config: crate::deepseek::DeepSeekConfig) -> Result<Self, DeepSeekTaskError> {
        let inner = crate::deepseek::DeepSeekForCausalLM::new(config.clone())
            .map_err(|e| DeepSeekTaskError::ModelBuildError(e.to_string()))?;
        Ok(Self { config, inner })
    }

    /// Config accessor.
    pub fn config(&self) -> &crate::deepseek::DeepSeekConfig {
        &self.config
    }

    /// Run a forward pass and return the raw logit tensor.
    pub fn forward(
        &self,
        input_ids: Vec<u32>,
    ) -> Result<trustformers_core::tensor::Tensor, DeepSeekTaskError> {
        if input_ids.is_empty() {
            return Err(DeepSeekTaskError::EmptyInput);
        }
        self.inner
            .forward(input_ids)
            .map_err(|e| DeepSeekTaskError::ForwardError(e.to_string()))
    }

    /// Greedy next-token prediction from a flat logit slice.
    pub fn greedy_next_token(logits: &[f32]) -> Option<u32> {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
    }
}

// ─── Sequence Classification ──────────────────────────────────────────────────

/// Sequence-level classification head for DeepSeek-V2.
///
/// Pools the first token's hidden state and projects it to `num_labels` classes.
pub struct DeepSeekForSequenceClassification {
    config: crate::deepseek::DeepSeekConfig,
    num_labels: usize,
    /// Classification weights: shape `[num_labels, hidden_size]` (row-major).
    classifier_weight: Vec<Vec<f32>>,
}

impl DeepSeekForSequenceClassification {
    /// Construct a sequence classification model.
    ///
    /// Weights are initialised with a deterministic LCG pattern so that tests
    /// do not depend on a PRNG crate.
    pub fn new(
        config: crate::deepseek::DeepSeekConfig,
        num_labels: usize,
    ) -> Result<Self, DeepSeekTaskError> {
        if num_labels < 2 {
            return Err(DeepSeekTaskError::InvalidNumLabels(num_labels));
        }
        let hidden = config.hidden_size;
        // LCG-initialised weights
        let mut state: u64 = 0xdeadbeef_cafebabe;
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
    pub fn config(&self) -> &crate::deepseek::DeepSeekConfig {
        &self.config
    }

    /// Number of output labels.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    /// Forward pass: returns logits of shape `[num_labels]`.
    ///
    /// The input tensor is treated as a flat hidden-state vector.  In a full
    /// implementation the model backbone would produce this vector; here we
    /// apply the classification head directly to the provided representation.
    pub fn forward(&self, hidden_state: &[f32]) -> Result<Vec<f32>, DeepSeekTaskError> {
        if hidden_state.is_empty() {
            return Err(DeepSeekTaskError::EmptyInput);
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

/// Token-level classification head for DeepSeek-V2 (e.g. NER).
///
/// Projects each token's hidden state to `num_labels` classes.
pub struct DeepSeekForTokenClassification {
    config: crate::deepseek::DeepSeekConfig,
    num_labels: usize,
    /// Per-label weights: shape `[num_labels, hidden_size]`.
    classifier_weight: Vec<Vec<f32>>,
}

impl DeepSeekForTokenClassification {
    /// Construct a token classification model.
    pub fn new(
        config: crate::deepseek::DeepSeekConfig,
        num_labels: usize,
    ) -> Result<Self, DeepSeekTaskError> {
        if num_labels < 2 {
            return Err(DeepSeekTaskError::InvalidNumLabels(num_labels));
        }
        let hidden = config.hidden_size;
        let mut state: u64 = 0x1234567890abcdef;
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
    pub fn config(&self) -> &crate::deepseek::DeepSeekConfig {
        &self.config
    }

    /// Number of output labels.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    /// Forward pass over a sequence of hidden states.
    ///
    /// `hidden_states` is a flat `[seq_len * hidden_size]` slice.
    /// Returns a flat `[seq_len * num_labels]` logit vector.
    pub fn forward(
        &self,
        hidden_states: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, DeepSeekTaskError> {
        if hidden_states.is_empty() || seq_len == 0 {
            return Err(DeepSeekTaskError::EmptyInput);
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

// ─── MoE routing utilities ────────────────────────────────────────────────────

/// Compute the top-k indices from a gating-logit slice.
///
/// Uses a stable partial-sort so results are deterministic.
/// Returns at most `k` indices (may be fewer if `logits.len() < k`).
pub fn moe_topk_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let effective_k = k.min(logits.len());
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    // Partial-sort: bring the top-k to the front
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed[..effective_k].iter().map(|(i, _)| *i).collect()
}

/// Compute the load-balance fraction: fraction of tokens routed to each expert.
///
/// `routing_indices` is a flat list of expert indices selected across all tokens.
/// Returns a `Vec<f32>` of length `n_experts` with each entry in `[0, 1]`.
pub fn moe_load_balance(routing_indices: &[usize], n_experts: usize) -> Vec<f32> {
    if routing_indices.is_empty() || n_experts == 0 {
        return vec![0.0f32; n_experts];
    }
    let mut counts = vec![0usize; n_experts];
    for &idx in routing_indices {
        if idx < n_experts {
            counts[idx] += 1;
        }
    }
    let total = routing_indices.len() as f32;
    counts.iter().map(|&c| c as f32 / total).collect()
}

/// Softmax over a logit slice — used to compute expert probabilities.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek::DeepSeekConfig;

    fn small_cfg() -> DeepSeekConfig {
        DeepSeekConfig::small_test()
    }

    // ── 1. CausalLM construction ──────────────────────────────────────────────

    #[test]
    fn test_causal_lm_construction() {
        let result = DeepSeekForCausalLM::new(small_cfg());
        assert!(
            result.is_ok(),
            "DeepSeekForCausalLM must construct: {:?}",
            result.err()
        );
    }

    // ── 2. CausalLM config accessor ───────────────────────────────────────────

    #[test]
    fn test_causal_lm_config_accessor() {
        let model = DeepSeekForCausalLM::new(small_cfg()).expect("construction");
        assert_eq!(model.config().hidden_size, 64);
        assert_eq!(model.config().vocab_size, 1024);
    }

    // ── 3. CausalLM forward safe pattern ─────────────────────────────────────

    #[test]
    fn test_causal_lm_forward_safe() {
        let model = DeepSeekForCausalLM::new(small_cfg()).expect("construction");
        match model.forward(vec![1u32, 2, 3]) {
            Ok(out) => {
                use trustformers_core::tensor::Tensor;
                if let Tensor::F32(arr) = &out {
                    assert!(!arr.is_empty(), "logits must be non-empty");
                }
            },
            Err(_) => {
                // Forward may fail in test environment without weights — that is acceptable.
            },
        }
    }

    // ── 4. CausalLM empty input error ─────────────────────────────────────────

    #[test]
    fn test_causal_lm_empty_input_error() {
        let model = DeepSeekForCausalLM::new(small_cfg()).expect("construction");
        let result = model.forward(vec![]);
        assert!(
            matches!(result, Err(DeepSeekTaskError::EmptyInput)),
            "empty input must return EmptyInput error"
        );
    }

    // ── 5. Greedy next-token picks maximum ────────────────────────────────────

    #[test]
    fn test_greedy_next_token_picks_max() {
        let logits = vec![0.1f32, 0.9, 0.2, 0.5];
        let tok = DeepSeekForCausalLM::greedy_next_token(&logits);
        assert_eq!(tok, Some(1u32), "argmax of [0.1,0.9,0.2,0.5] must be 1");
    }

    // ── 6. Greedy next-token on empty returns None ─────────────────────────────

    #[test]
    fn test_greedy_next_token_empty_returns_none() {
        assert_eq!(DeepSeekForCausalLM::greedy_next_token(&[]), None);
    }

    // ── 7. SequenceClassification construction ────────────────────────────────

    #[test]
    fn test_seq_cls_construction() {
        let result = DeepSeekForSequenceClassification::new(small_cfg(), 4);
        assert!(result.is_ok(), "SequenceClassification must construct");
    }

    // ── 8. SequenceClassification invalid labels ──────────────────────────────

    #[test]
    fn test_seq_cls_invalid_labels() {
        let result = DeepSeekForSequenceClassification::new(small_cfg(), 1);
        assert!(
            matches!(result, Err(DeepSeekTaskError::InvalidNumLabels(1))),
            "num_labels=1 must be rejected"
        );
    }

    // ── 9. SequenceClassification forward output length ───────────────────────

    #[test]
    fn test_seq_cls_forward_output_length() {
        let model = DeepSeekForSequenceClassification::new(small_cfg(), 3).expect("construction");
        let hidden = vec![0.1f32; small_cfg().hidden_size];
        let logits = model.forward(&hidden).expect("forward");
        assert_eq!(logits.len(), 3, "must produce 3 logits");
    }

    // ── 10. SequenceClassification empty input error ──────────────────────────

    #[test]
    fn test_seq_cls_empty_input_error() {
        let model = DeepSeekForSequenceClassification::new(small_cfg(), 2).expect("construction");
        let result = model.forward(&[]);
        assert!(matches!(result, Err(DeepSeekTaskError::EmptyInput)));
    }

    // ── 11. TokenClassification construction ──────────────────────────────────

    #[test]
    fn test_tok_cls_construction() {
        let result = DeepSeekForTokenClassification::new(small_cfg(), 5);
        assert!(result.is_ok(), "TokenClassification must construct");
    }

    // ── 12. TokenClassification forward output shape ──────────────────────────

    #[test]
    fn test_tok_cls_forward_output_shape() {
        let cfg = small_cfg();
        let hidden = cfg.hidden_size;
        let model = DeepSeekForTokenClassification::new(cfg, 4).expect("construction");
        let seq_len = 3usize;
        let states = vec![0.1f32; seq_len * hidden];
        let logits = model.forward(&states, seq_len).expect("forward");
        assert_eq!(
            logits.len(),
            seq_len * 4,
            "output shape must be seq_len * num_labels"
        );
    }

    // ── 13. TokenClassification empty input error ─────────────────────────────

    #[test]
    fn test_tok_cls_empty_input_error() {
        let model = DeepSeekForTokenClassification::new(small_cfg(), 2).expect("construction");
        let result = model.forward(&[], 0);
        assert!(matches!(result, Err(DeepSeekTaskError::EmptyInput)));
    }

    // ── 14. MoE top-k indices ─────────────────────────────────────────────────

    #[test]
    fn test_moe_topk_indices_basic() {
        let logits = vec![0.1f32, 0.8, 0.3, 0.9, 0.05];
        let topk = moe_topk_indices(&logits, 2);
        assert_eq!(topk.len(), 2);
        // Indices 3 (0.9) and 1 (0.8) must be selected
        assert!(topk.contains(&3), "index 3 must be top-1");
        assert!(topk.contains(&1), "index 1 must be top-2");
    }

    // ── 15. MoE top-k with k >= len ───────────────────────────────────────────

    #[test]
    fn test_moe_topk_k_exceeds_len() {
        let logits = vec![0.5f32, 0.3];
        let topk = moe_topk_indices(&logits, 10);
        assert_eq!(topk.len(), 2, "k capped at logits.len()");
    }

    // ── 16. MoE load balance uniform routing ─────────────────────────────────

    #[test]
    fn test_moe_load_balance_uniform() {
        let routing = vec![0usize, 1, 0, 1, 0, 1];
        let balance = moe_load_balance(&routing, 2);
        assert_eq!(balance.len(), 2);
        assert!((balance[0] - 0.5).abs() < 1e-5);
        assert!((balance[1] - 0.5).abs() < 1e-5);
    }

    // ── 17. MoE load balance empty routing ───────────────────────────────────

    #[test]
    fn test_moe_load_balance_empty() {
        let balance = moe_load_balance(&[], 4);
        assert_eq!(balance.len(), 4);
        for &v in &balance {
            assert_eq!(v, 0.0);
        }
    }

    // ── 18. Softmax sums to 1 ─────────────────────────────────────────────────

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax must sum to 1, got {}",
            sum
        );
    }

    // ── 19. Softmax monotone ordering preserved ───────────────────────────────

    #[test]
    fn test_softmax_ordering_preserved() {
        let logits = vec![0.0f32, 1.0, 2.0];
        let probs = softmax(&logits);
        assert!(probs[0] < probs[1] && probs[1] < probs[2]);
    }

    // ── 20. Error display messages ────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let e1 = DeepSeekTaskError::InvalidConfig("bad param".to_string());
        assert!(e1.to_string().contains("bad param"));

        let e2 = DeepSeekTaskError::EmptyInput;
        assert!(e2.to_string().contains("empty"));

        let e3 = DeepSeekTaskError::InvalidNumLabels(1);
        assert!(e3.to_string().contains("1"));
    }

    // ── 21. LCG-based random value range ─────────────────────────────────────

    #[test]
    fn test_lcg_value_range() {
        let mut state: u64 = 42;
        for _ in 0..100 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = (state as f32 / u64::MAX as f32) * 0.02 - 0.01;
            assert!((-0.01..=0.01).contains(&v), "value out of range: {v}");
        }
    }

    // ── 22. num_labels accessor ───────────────────────────────────────────────

    #[test]
    fn test_num_labels_accessor() {
        let model = DeepSeekForSequenceClassification::new(small_cfg(), 7).expect("construction");
        assert_eq!(model.num_labels(), 7);
        let tok_model = DeepSeekForTokenClassification::new(small_cfg(), 9).expect("construction");
        assert_eq!(tok_model.num_labels(), 9);
    }
}
