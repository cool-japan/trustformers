//! # Mamba-2 Task-Specific Implementations
//!
//! This module provides task-specific utilities for Mamba-2 State Space Models,
//! including:
//! - Causal LM head utilities (greedy decode, top-k, top-p sampling)
//! - Sequence classification head
//! - Softplus and selective scan helpers
//! - SSD (State Space Duality) parallel scan coverage metric
//! - Generation utilities tailored to the recurrent SSM architecture

use std::fmt;

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors produced by Mamba-2 task-specific operations.
#[derive(Debug)]
pub enum Mamba2TaskError {
    /// Empty input sequence.
    EmptyInput,
    /// Dimension mismatch between tensors.
    DimMismatch { expected: usize, got: usize },
    /// Top-k value exceeds vocabulary size.
    TopKTooLarge { k: usize, vocab_size: usize },
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Forward pass failed.
    ForwardError(String),
    /// Invalid nucleus probability.
    InvalidNucleus(f32),
}

impl fmt::Display for Mamba2TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mamba2TaskError::EmptyInput => write!(f, "Mamba-2 task error: empty input"),
            Mamba2TaskError::DimMismatch { expected, got } => write!(
                f,
                "Mamba-2 task error: dim mismatch — expected {expected}, got {got}"
            ),
            Mamba2TaskError::TopKTooLarge { k, vocab_size } => write!(
                f,
                "Mamba-2 task error: top_k={k} exceeds vocab_size={vocab_size}"
            ),
            Mamba2TaskError::InvalidConfig(msg) => {
                write!(f, "Mamba-2 task error: invalid config: {msg}")
            },
            Mamba2TaskError::ForwardError(msg) => {
                write!(f, "Mamba-2 task error: forward error: {msg}")
            },
            Mamba2TaskError::InvalidNucleus(p) => {
                write!(
                    f,
                    "Mamba-2 task error: nucleus probability {p} out of (0, 1]"
                )
            },
        }
    }
}

impl std::error::Error for Mamba2TaskError {}

// ─── Softplus ─────────────────────────────────────────────────────────────────

/// Softplus activation used for the `dt` parameter in Mamba-2.
///
/// `softplus(x) = log(1 + exp(x))`
///
/// Numerically stable for large positive and large negative values.
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

// ─── SiLU ─────────────────────────────────────────────────────────────────────

/// SiLU activation: `x * sigmoid(x)`.
pub fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

// ─── RMSNorm ──────────────────────────────────────────────────────────────────

/// RMSNorm (unit weight, no bias) for f64 slices.
pub fn rms_norm_f64(input: &[f64], eps: f64) -> Vec<f64> {
    if input.is_empty() {
        return Vec::new();
    }
    let mean_sq = input.iter().map(|x| x * x).sum::<f64>() / input.len() as f64;
    let rms = (mean_sq + eps).sqrt();
    input.iter().map(|x| x / rms).collect()
}

// ─── Softmax ──────────────────────────────────────────────────────────────────

/// Numerically stable softmax (f32).
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return exps;
    }
    exps.iter().map(|&e| e / sum).collect()
}

// ─── Greedy decode ────────────────────────────────────────────────────────────

/// Greedy decode: return the index of the maximum logit.
pub fn greedy_decode(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
}

// ─── Top-k filter ─────────────────────────────────────────────────────────────

/// Mask all but top-k logits with `f32::NEG_INFINITY`.
pub fn top_k_filter(logits: &[f32], k: usize) -> Result<Vec<f32>, Mamba2TaskError> {
    let vocab = logits.len();
    if k > vocab {
        return Err(Mamba2TaskError::TopKTooLarge {
            k,
            vocab_size: vocab,
        });
    }
    if k == 0 {
        return Ok(vec![f32::NEG_INFINITY; vocab]);
    }
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = indexed[k - 1].1;
    Ok(logits
        .iter()
        .map(|&l| if l >= threshold { l } else { f32::NEG_INFINITY })
        .collect())
}

// ─── Top-p (nucleus) filter ───────────────────────────────────────────────────

/// Mask logits outside the nucleus (smallest set summing to ≥ p probability).
pub fn top_p_filter(logits: &[f32], p: f32) -> Result<Vec<f32>, Mamba2TaskError> {
    if p <= 0.0 || p > 1.0 {
        return Err(Mamba2TaskError::InvalidNucleus(p));
    }
    let probs = softmax(logits);
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumsum = 0.0f32;
    let mut included = vec![false; logits.len()];
    for (orig_idx, prob) in &indexed {
        included[*orig_idx] = true;
        cumsum += prob;
        if cumsum >= p {
            break;
        }
    }
    Ok(logits
        .iter()
        .enumerate()
        .map(|(i, &l)| if included[i] { l } else { f32::NEG_INFINITY })
        .collect())
}

// ─── SSD scan coverage ────────────────────────────────────────────────────────

/// Fraction of the sequence that falls within the SSD chunk window.
///
/// In Mamba-2 the parallel scan is decomposed into chunks of `chunk_size`.
/// Within each chunk, every token can attend to all prior tokens in the same
/// chunk.  Across chunks, information is passed only via the SSM state.
///
/// This function returns the ratio of within-chunk pairs to total causal pairs.
pub fn ssd_chunk_coverage(seq_len: usize, chunk_size: usize) -> f64 {
    if seq_len == 0 || chunk_size == 0 {
        return 0.0;
    }
    let total_causal: usize = (1..=seq_len).sum(); // seq_len*(seq_len+1)/2
    let mut within_chunk = 0usize;
    let mut pos = 0usize;
    while pos < seq_len {
        let end = (pos + chunk_size).min(seq_len);
        let chunk_len = end - pos;
        within_chunk += (1..=chunk_len).sum::<usize>();
        pos = end;
    }
    within_chunk as f64 / total_causal as f64
}

// ─── Selective scan step ──────────────────────────────────────────────────────

/// Perform a single Mamba-2 SSM state-update step.
///
/// Updates the hidden state:
/// ```text
/// h_new = a_bar * h + b_bar * x
/// y     = C * h_new + d * x
/// ```
///
/// All inputs are 1-D slices of length `d_state`.
///
/// # Errors
///
/// Returns [`Mamba2TaskError::DimMismatch`] if the slice lengths are inconsistent.
pub fn ssm_step(
    h: &[f64],
    x: f64,
    a_bar: &[f64],
    b_bar: &[f64],
    c: &[f64],
    d: f64,
) -> Result<(Vec<f64>, f64), Mamba2TaskError> {
    let d_state = h.len();
    if a_bar.len() != d_state {
        return Err(Mamba2TaskError::DimMismatch {
            expected: d_state,
            got: a_bar.len(),
        });
    }
    if b_bar.len() != d_state {
        return Err(Mamba2TaskError::DimMismatch {
            expected: d_state,
            got: b_bar.len(),
        });
    }
    if c.len() != d_state {
        return Err(Mamba2TaskError::DimMismatch {
            expected: d_state,
            got: c.len(),
        });
    }

    let h_new: Vec<f64> = (0..d_state).map(|i| a_bar[i] * h[i] + b_bar[i] * x).collect();
    let y: f64 = h_new.iter().zip(c.iter()).map(|(hi, ci)| ci * hi).sum::<f64>() + d * x;
    Ok((h_new, y))
}

// ─── Causal LM head ───────────────────────────────────────────────────────────

/// Mamba-2 causal LM head.
///
/// Maps the model's `d_model`-dimensional hidden state to vocabulary logits.
/// Weights are zero-initialised for test use.
pub struct Mamba2ForCausalLMHead {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Model hidden dimension (`d_model`).
    pub d_model: usize,
    /// Whether input/output embeddings are tied.
    pub tie_embeddings: bool,
    /// LM head weight `[vocab_size × d_model]`.
    lm_weight: Vec<Vec<f32>>,
}

impl Mamba2ForCausalLMHead {
    /// Create a new causal LM head.
    pub fn new(
        d_model: usize,
        vocab_size: usize,
        tie_embeddings: bool,
    ) -> Result<Self, Mamba2TaskError> {
        if d_model == 0 {
            return Err(Mamba2TaskError::InvalidConfig(
                "d_model must be > 0".to_string(),
            ));
        }
        if vocab_size == 0 {
            return Err(Mamba2TaskError::InvalidConfig(
                "vocab_size must be > 0".to_string(),
            ));
        }
        Ok(Self {
            vocab_size,
            d_model,
            tie_embeddings,
            lm_weight: vec![vec![0.0f32; d_model]; vocab_size],
        })
    }

    /// Compute vocabulary logits from the last hidden state.
    pub fn compute_logits(&self, last_hidden: &[f32]) -> Result<Vec<f32>, Mamba2TaskError> {
        if last_hidden.len() != self.d_model {
            return Err(Mamba2TaskError::DimMismatch {
                expected: self.d_model,
                got: last_hidden.len(),
            });
        }
        let logits: Vec<f32> = self
            .lm_weight
            .iter()
            .map(|row| row.iter().zip(last_hidden.iter()).map(|(w, x)| w * x).sum())
            .collect();
        Ok(logits)
    }

    /// Greedy forward: argmax token from the last hidden state.
    pub fn forward_greedy(&self, last_hidden: &[f32]) -> Result<u32, Mamba2TaskError> {
        let logits = self.compute_logits(last_hidden)?;
        greedy_decode(&logits).ok_or_else(|| Mamba2TaskError::ForwardError("argmax failed".into()))
    }
}

// ─── Sequence classification head ────────────────────────────────────────────

/// Mamba-2 sequence classification head.
///
/// Pools the last-token hidden state and projects to `[num_labels]` logits.
pub struct Mamba2ForSequenceClassification {
    /// Number of output classes.
    pub num_labels: usize,
    /// Model hidden dimension.
    pub d_model: usize,
    /// Weight matrix `[num_labels × d_model]`.
    weight: Vec<Vec<f32>>,
    /// Bias `[num_labels]`.
    bias: Vec<f32>,
}

impl Mamba2ForSequenceClassification {
    /// Create a new sequence classification head.
    pub fn new(d_model: usize, num_labels: usize) -> Result<Self, Mamba2TaskError> {
        if d_model == 0 {
            return Err(Mamba2TaskError::InvalidConfig(
                "d_model must be > 0".to_string(),
            ));
        }
        if num_labels == 0 {
            return Err(Mamba2TaskError::InvalidConfig(
                "num_labels must be > 0".to_string(),
            ));
        }
        Ok(Self {
            num_labels,
            d_model,
            weight: vec![vec![0.0f32; d_model]; num_labels],
            bias: vec![0.0f32; num_labels],
        })
    }

    /// Pool last-token hidden state → label logits.
    ///
    /// `hidden_states` is flat `[seq_len * d_model]`.
    pub fn forward(&self, hidden_states: &[f32]) -> Result<Vec<f32>, Mamba2TaskError> {
        if hidden_states.is_empty() {
            return Err(Mamba2TaskError::EmptyInput);
        }
        let seq_len = hidden_states.len() / self.d_model;
        if seq_len == 0 {
            return Err(Mamba2TaskError::EmptyInput);
        }
        let start = (seq_len - 1) * self.d_model;
        let last = &hidden_states[start..start + self.d_model];
        let logits: Vec<f32> = self
            .weight
            .iter()
            .zip(self.bias.iter())
            .map(|(row, &b)| row.iter().zip(last.iter()).map(|(w, x)| w * x).sum::<f32>() + b)
            .collect();
        Ok(logits)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── test_mamba2_config_small_test ─────────────────────────────────────

    #[test]
    fn test_mamba2_config_small_test() {
        use crate::mamba2::config::Mamba2Config;
        let cfg = Mamba2Config::small_test();
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.n_layer, 2);
        assert!(cfg.validate());
    }

    // ── test_mamba2_config_2_7b ───────────────────────────────────────────

    #[test]
    fn test_mamba2_config_2_7b() {
        use crate::mamba2::config::Mamba2Config;
        let cfg = Mamba2Config::mamba2_2_7b();
        assert_eq!(cfg.d_model, 2560);
        assert_eq!(cfg.nheads, 80);
        assert_eq!(cfg.headdim, 64);
        assert!(cfg.validate());
    }

    // ── test_mamba2_softplus_positive ─────────────────────────────────────

    #[test]
    fn test_mamba2_softplus_positive() {
        for &x in &[-100.0_f64, -10.0, 0.0, 10.0, 100.0] {
            assert!(softplus(x) > 0.0, "softplus({x}) must be positive");
        }
    }

    // ── test_mamba2_softplus_large_identity ───────────────────────────────

    #[test]
    fn test_mamba2_softplus_large_identity() {
        let v = softplus(50.0);
        assert!((v - 50.0).abs() < 0.01, "softplus(50) ≈ 50, got {v}");
    }

    // ── test_mamba2_silu_zero ──────────────────────────────────────────────

    #[test]
    fn test_mamba2_silu_zero() {
        assert!(silu(0.0).abs() < 1e-10);
    }

    // ── test_mamba2_silu_large_positive ───────────────────────────────────

    #[test]
    fn test_mamba2_silu_large_positive() {
        // silu(x) ≈ x for large x
        assert!((silu(100.0) - 100.0).abs() < 0.01);
    }

    // ── test_mamba2_rms_norm_constant ─────────────────────────────────────

    #[test]
    fn test_mamba2_rms_norm_constant() {
        let x = vec![3.0f64; 8];
        let out = rms_norm_f64(&x, 1e-8);
        for &v in &out {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "constant must normalize to 1, got {v}"
            );
        }
    }

    // ── test_mamba2_rms_norm_pair ─────────────────────────────────────────

    #[test]
    fn test_mamba2_rms_norm_pair() {
        let x = vec![3.0f64, 4.0];
        let out = rms_norm_f64(&x, 1e-8);
        let rms = (12.5_f64 + 1e-8).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-8);
        assert!((out[1] - 4.0 / rms).abs() < 1e-8);
    }

    // ── test_mamba2_softmax ───────────────────────────────────────────────

    #[test]
    fn test_mamba2_softmax() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    // ── test_mamba2_greedy_decode ─────────────────────────────────────────

    #[test]
    fn test_mamba2_greedy_decode() {
        let logits = vec![0.1f32, 0.9, 0.5];
        assert_eq!(greedy_decode(&logits), Some(1u32));
    }

    // ── test_mamba2_top_k_filter ──────────────────────────────────────────

    #[test]
    fn test_mamba2_top_k_filter() {
        let logits = vec![1.0f32, 5.0, 3.0, 2.0];
        let f = top_k_filter(&logits, 2).expect("top_k");
        assert!((f[1] - 5.0).abs() < 1e-6);
        assert!((f[2] - 3.0).abs() < 1e-6);
        assert!(f[0].is_infinite() && f[0] < 0.0);
    }

    // ── test_mamba2_top_k_too_large ───────────────────────────────────────

    #[test]
    fn test_mamba2_top_k_too_large() {
        let logits = vec![1.0f32; 3];
        assert!(matches!(
            top_k_filter(&logits, 10),
            Err(Mamba2TaskError::TopKTooLarge {
                k: 10,
                vocab_size: 3
            })
        ));
    }

    // ── test_mamba2_top_p_filter ──────────────────────────────────────────

    #[test]
    fn test_mamba2_top_p_filter() {
        let logits = vec![0.0f32, 10.0, 0.0]; // dominated by index 1
        let f = top_p_filter(&logits, 0.99).expect("top_p");
        assert!((f[1] - 10.0).abs() < 1e-6);
    }

    // ── test_mamba2_top_p_invalid_nucleus ─────────────────────────────────

    #[test]
    fn test_mamba2_top_p_invalid_nucleus() {
        let logits = vec![1.0f32; 4];
        assert!(matches!(
            top_p_filter(&logits, 0.0),
            Err(Mamba2TaskError::InvalidNucleus(_))
        ));
        assert!(matches!(
            top_p_filter(&logits, 1.1),
            Err(Mamba2TaskError::InvalidNucleus(_))
        ));
    }

    // ── test_mamba2_ssd_chunk_coverage_full ───────────────────────────────

    #[test]
    fn test_mamba2_ssd_chunk_coverage_full() {
        // If chunk_size >= seq_len, all causal pairs are within-chunk → 1.0
        let cov = ssd_chunk_coverage(8, 8);
        assert!(
            (cov - 1.0).abs() < 1e-10,
            "full chunk should give coverage 1.0, got {cov}"
        );
    }

    // ── test_mamba2_ssd_chunk_coverage_one ────────────────────────────────

    #[test]
    fn test_mamba2_ssd_chunk_coverage_one() {
        // chunk_size=1: only self-pairs → 1 / total_causal
        let seq_len = 4usize;
        let cov = ssd_chunk_coverage(seq_len, 1);
        let total: usize = (1..=seq_len).sum();
        let expected = seq_len as f64 / total as f64; // 4/10 = 0.4
        assert!(
            (cov - expected).abs() < 1e-10,
            "expected {expected}, got {cov}"
        );
    }

    // ── test_mamba2_ssm_step ──────────────────────────────────────────────

    #[test]
    fn test_mamba2_ssm_step() {
        let d_state = 4;
        let h = vec![1.0f64; d_state];
        let x = 0.5f64;
        let a_bar = vec![0.9f64; d_state];
        let b_bar = vec![0.1f64; d_state];
        let c = vec![1.0f64; d_state];
        let d = 0.0f64;
        let (h_new, y) = ssm_step(&h, x, &a_bar, &b_bar, &c, d).expect("ssm_step");
        assert_eq!(h_new.len(), d_state);
        // h_new[i] = 0.9*1.0 + 0.1*0.5 = 0.95
        assert!(
            (h_new[0] - 0.95).abs() < 1e-10,
            "h_new[0] should be 0.95, got {}",
            h_new[0]
        );
        // y = sum(c * h_new) + d*x = 4 * 0.95 = 3.8
        assert!((y - 3.8).abs() < 1e-10, "y should be 3.8, got {y}");
    }

    // ── test_mamba2_ssm_step_dim_mismatch ─────────────────────────────────

    #[test]
    fn test_mamba2_ssm_step_dim_mismatch() {
        let h = vec![1.0f64; 4];
        let a_bar = vec![0.9f64; 3]; // wrong dimension
        let b_bar = vec![0.1f64; 4];
        let c = vec![1.0f64; 4];
        let result = ssm_step(&h, 0.0, &a_bar, &b_bar, &c, 0.0);
        assert!(matches!(result, Err(Mamba2TaskError::DimMismatch { .. })));
    }

    // ── test_mamba2_causal_lm_construction ───────────────────────────────

    #[test]
    fn test_mamba2_causal_lm_construction() {
        let head = Mamba2ForCausalLMHead::new(64, 256, true);
        assert!(head.is_ok());
        let h = head.expect("causal lm head");
        assert_eq!(h.vocab_size, 256);
        assert_eq!(h.d_model, 64);
        assert!(h.tie_embeddings);
    }

    // ── test_mamba2_causal_lm_greedy ──────────────────────────────────────

    #[test]
    fn test_mamba2_causal_lm_greedy() {
        let head = Mamba2ForCausalLMHead::new(4, 10, false).expect("causal lm");
        let token = head.forward_greedy(&[0.0f32; 4]).expect("greedy");
        // zero weights → all 0 → max_by returns last equal-max index
        assert!(token < 10u32, "token {token} must be within vocab_size=10");
    }

    // ── test_mamba2_causal_lm_dim_mismatch ───────────────────────────────

    #[test]
    fn test_mamba2_causal_lm_dim_mismatch() {
        let head = Mamba2ForCausalLMHead::new(8, 10, false).expect("causal lm");
        assert!(matches!(
            head.compute_logits(&[0.0f32; 4]),
            Err(Mamba2TaskError::DimMismatch { .. })
        ));
    }

    // ── test_mamba2_seq_cls_forward ───────────────────────────────────────

    #[test]
    fn test_mamba2_seq_cls_forward() {
        let head = Mamba2ForSequenceClassification::new(8, 3).expect("seq cls");
        let hidden = vec![0.5f32; 24]; // seq_len=3
        let logits = head.forward(&hidden).expect("forward");
        assert_eq!(logits.len(), 3);
    }

    // ── test_mamba2_seq_cls_empty ─────────────────────────────────────────

    #[test]
    fn test_mamba2_seq_cls_empty() {
        let head = Mamba2ForSequenceClassification::new(4, 2).expect("seq cls");
        assert!(matches!(
            head.forward(&[]),
            Err(Mamba2TaskError::EmptyInput)
        ));
    }

    // ── test_mamba2_error_display ─────────────────────────────────────────

    #[test]
    fn test_mamba2_error_display() {
        let e1 = Mamba2TaskError::EmptyInput;
        assert!(e1.to_string().contains("empty"));

        let e2 = Mamba2TaskError::DimMismatch {
            expected: 8,
            got: 4,
        };
        assert!(e2.to_string().contains("8") && e2.to_string().contains("4"));

        let e3 = Mamba2TaskError::TopKTooLarge {
            k: 10,
            vocab_size: 5,
        };
        assert!(e3.to_string().contains("10"));

        let e4 = Mamba2TaskError::InvalidConfig("bad".to_string());
        assert!(e4.to_string().contains("bad"));
    }

    // ── test_mamba2_lcg_varied_seq_cls ────────────────────────────────────

    #[test]
    fn test_mamba2_lcg_varied_seq_cls() {
        let mut state = 53u64;
        for _ in 0..6 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let d_model = ((state % 4) + 2) as usize * 8;
            let num_labels = ((state >> 4) % 4 + 2) as usize;
            let head =
                Mamba2ForSequenceClassification::new(d_model, num_labels).expect("seq cls head");
            let hs: Vec<f32> = (0..d_model * 2).map(|i| i as f32 * 0.01).collect();
            let out = head.forward(&hs).expect("forward");
            assert_eq!(out.len(), num_labels);
        }
    }
}
