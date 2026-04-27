//! # GPT-2 Task-Specific Implementations
//!
//! This module provides task-specific utility functions and data structures for
//! GPT-2 models, including:
//! - Causal language modelling helpers (GELU, layer norm, softmax, greedy decode)
//! - Sequence classification head logic
//! - Token classification head logic
//! - Top-k and top-p sampling utilities
//! - Positional embedding helpers

use std::fmt;

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors produced by GPT-2 task-specific operations.
#[derive(Debug)]
pub enum Gpt2TaskError {
    /// Empty input sequence.
    EmptyInput,
    /// The requested top-k value exceeds the vocabulary size.
    TopKTooLarge { k: usize, vocab_size: usize },
    /// Probability mass is zero (degenerate distribution).
    ZeroProbMass,
    /// Nucleus probability must be in (0, 1].
    InvalidNucleus(f32),
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Forward pass failed.
    ForwardError(String),
}

impl fmt::Display for Gpt2TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Gpt2TaskError::EmptyInput => write!(f, "GPT-2 task error: empty input"),
            Gpt2TaskError::TopKTooLarge { k, vocab_size } => write!(
                f,
                "GPT-2 task error: top_k={k} exceeds vocab_size={vocab_size}"
            ),
            Gpt2TaskError::ZeroProbMass => {
                write!(f, "GPT-2 task error: probability mass is zero")
            },
            Gpt2TaskError::InvalidNucleus(p) => {
                write!(
                    f,
                    "GPT-2 task error: nucleus probability {p} out of range (0, 1]"
                )
            },
            Gpt2TaskError::InvalidConfig(msg) => {
                write!(f, "GPT-2 task error: invalid config: {msg}")
            },
            Gpt2TaskError::ForwardError(msg) => {
                write!(f, "GPT-2 task error: forward pass failed: {msg}")
            },
        }
    }
}

impl std::error::Error for Gpt2TaskError {}

// ─── GELU activation ─────────────────────────────────────────────────────────

/// GELU activation (Gaussian Error Linear Unit) used throughout GPT-2.
///
/// Uses the `tanh` approximation:
/// `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2 / π)
    let inner = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Apply GELU element-wise to a slice.
pub fn gelu_vec(xs: &[f32]) -> Vec<f32> {
    xs.iter().map(|&x| gelu(x)).collect()
}

// ─── Layer normalisation ──────────────────────────────────────────────────────

/// Apply Layer Normalisation to a flat slice using unit weight and zero bias.
///
/// `output[i] = (input[i] - mean) / sqrt(var + eps)`
pub fn layer_norm(input: &[f32], eps: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let n = input.len() as f32;
    let mean = input.iter().sum::<f32>() / n;
    let var = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let denom = (var + eps).sqrt();
    input.iter().map(|&x| (x - mean) / denom).collect()
}

// ─── Softmax ──────────────────────────────────────────────────────────────────

/// Numerically stable softmax over a logit slice.
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
        .map(|(idx, _)| idx as u32)
}

// ─── Top-k filtering ─────────────────────────────────────────────────────────

/// Mask all but the top-k logits with `f32::NEG_INFINITY`.
///
/// Returns the filtered logit vector, leaving the top-k values untouched.
///
/// # Errors
///
/// Returns [`Gpt2TaskError::TopKTooLarge`] if `k > logits.len()`.
pub fn top_k_filter(logits: &[f32], k: usize) -> Result<Vec<f32>, Gpt2TaskError> {
    let vocab = logits.len();
    if k > vocab {
        return Err(Gpt2TaskError::TopKTooLarge {
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

// ─── Top-p (nucleus) filtering ────────────────────────────────────────────────

/// Mask logits outside the nucleus: the smallest set of tokens whose
/// cumulative probability is at least `p`.
///
/// # Errors
///
/// Returns [`Gpt2TaskError::InvalidNucleus`] when `p <= 0` or `p > 1`.
pub fn top_p_filter(logits: &[f32], p: f32) -> Result<Vec<f32>, Gpt2TaskError> {
    if p <= 0.0 || p > 1.0 {
        return Err(Gpt2TaskError::InvalidNucleus(p));
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

// ─── Temperature scaling ─────────────────────────────────────────────────────

/// Apply temperature to logits: `logit / temperature`.
///
/// Temperatures < 1 sharpen the distribution; temperatures > 1 flatten it.
/// Returns a new `Vec<f32>` (the original slice is not modified).
pub fn apply_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    if temperature <= 0.0 {
        return logits.to_vec();
    }
    logits.iter().map(|&l| l / temperature).collect()
}

// ─── Learned positional embedding helper ─────────────────────────────────────

/// Compute sinusoidal positional embeddings for the given positions.
///
/// Even dimensions use cosine, odd dimensions use sine (GPT-style learned
/// embeddings are replaced by sinusoidal here for testing).
///
/// # Arguments
/// * `positions` - Token positions (0-indexed)
/// * `d_model`   - Embedding dimension
pub fn sinusoidal_pos_embed(positions: &[usize], d_model: usize) -> Vec<Vec<f32>> {
    positions
        .iter()
        .map(|&pos| {
            (0..d_model)
                .map(|i| {
                    let angle =
                        pos as f32 / (10000.0_f32).powf(2.0 * (i / 2) as f32 / d_model as f32);
                    if i % 2 == 0 {
                        angle.cos()
                    } else {
                        angle.sin()
                    }
                })
                .collect()
        })
        .collect()
}

// ─── Sequence classification head ────────────────────────────────────────────

/// A lightweight sequence-classification head on top of GPT-2.
///
/// Given the last hidden state `[seq_len × hidden_size]` (flat), the head
/// applies a single linear layer on the pooled representation (last token)
/// to produce `[num_labels]` class logits.
///
/// All weights are initialised to zero and biases to zero, matching the
/// "dummy" forward-pass contract used in tests.
pub struct Gpt2ForSequenceClassification {
    /// Number of output classes.
    pub num_labels: usize,
    /// GPT-2 hidden size.
    pub hidden_size: usize,
    /// Linear weight matrix `[num_labels × hidden_size]`.
    weight: Vec<Vec<f32>>,
    /// Bias vector `[num_labels]`.
    bias: Vec<f32>,
}

impl Gpt2ForSequenceClassification {
    /// Create a new sequence classification head.
    pub fn new(hidden_size: usize, num_labels: usize) -> Result<Self, Gpt2TaskError> {
        if hidden_size == 0 {
            return Err(Gpt2TaskError::InvalidConfig(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if num_labels == 0 {
            return Err(Gpt2TaskError::InvalidConfig(
                "num_labels must be > 0".to_string(),
            ));
        }
        Ok(Self {
            num_labels,
            hidden_size,
            weight: vec![vec![0.0f32; hidden_size]; num_labels],
            bias: vec![0.0f32; num_labels],
        })
    }

    /// Forward pass: pool last-token hidden state and project to class logits.
    ///
    /// # Arguments
    /// * `hidden_states` - Flat `[seq_len * hidden_size]` last-layer activations
    ///
    /// Returns `[num_labels]` logits.
    pub fn forward(&self, hidden_states: &[f32]) -> Result<Vec<f32>, Gpt2TaskError> {
        if hidden_states.is_empty() {
            return Err(Gpt2TaskError::EmptyInput);
        }
        let seq_len = hidden_states.len() / self.hidden_size;
        if seq_len == 0 {
            return Err(Gpt2TaskError::EmptyInput);
        }
        // Pool: use the last token's hidden state
        let start = (seq_len - 1) * self.hidden_size;
        let last_token = &hidden_states[start..start + self.hidden_size];

        // Linear projection
        let logits: Vec<f32> = self
            .weight
            .iter()
            .zip(self.bias.iter())
            .map(|(row, &b)| {
                let dot: f32 = row.iter().zip(last_token.iter()).map(|(w, x)| w * x).sum();
                dot + b
            })
            .collect();

        Ok(logits)
    }
}

// ─── Token classification head ────────────────────────────────────────────────

/// A token-level classification head on top of GPT-2.
///
/// Projects each token's hidden state to `[num_labels]` class logits.
pub struct Gpt2ForTokenClassification {
    /// Number of token-level output classes.
    pub num_labels: usize,
    /// GPT-2 hidden size.
    pub hidden_size: usize,
    /// Weight matrix `[num_labels × hidden_size]`.
    weight: Vec<Vec<f32>>,
    /// Bias vector `[num_labels]`.
    bias: Vec<f32>,
}

impl Gpt2ForTokenClassification {
    /// Create a new token classification head.
    pub fn new(hidden_size: usize, num_labels: usize) -> Result<Self, Gpt2TaskError> {
        if hidden_size == 0 {
            return Err(Gpt2TaskError::InvalidConfig(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if num_labels == 0 {
            return Err(Gpt2TaskError::InvalidConfig(
                "num_labels must be > 0".to_string(),
            ));
        }
        Ok(Self {
            num_labels,
            hidden_size,
            weight: vec![vec![0.0f32; hidden_size]; num_labels],
            bias: vec![0.0f32; num_labels],
        })
    }

    /// Forward pass: project every token's hidden state to class logits.
    ///
    /// # Arguments
    /// * `hidden_states` - Flat `[seq_len * hidden_size]` activations
    ///
    /// Returns `seq_len × num_labels` logits (outer Vec is per-token).
    pub fn forward(&self, hidden_states: &[f32]) -> Result<Vec<Vec<f32>>, Gpt2TaskError> {
        if hidden_states.is_empty() {
            return Err(Gpt2TaskError::EmptyInput);
        }
        let seq_len = hidden_states.len() / self.hidden_size;
        if seq_len == 0 {
            return Err(Gpt2TaskError::EmptyInput);
        }
        let result = (0..seq_len)
            .map(|t| {
                let start = t * self.hidden_size;
                let token_hidden = &hidden_states[start..start + self.hidden_size];
                self.weight
                    .iter()
                    .zip(self.bias.iter())
                    .map(|(row, &b)| {
                        let dot: f32 =
                            row.iter().zip(token_hidden.iter()).map(|(w, x)| w * x).sum();
                        dot + b
                    })
                    .collect::<Vec<f32>>()
            })
            .collect();
        Ok(result)
    }
}

// ─── CausalLM head ────────────────────────────────────────────────────────────

/// Causal LM head: wraps greedy and nucleus generation utilities for GPT-2.
///
/// All weight tensors are zero-initialised for test use; in production the
/// weights would be loaded from a checkpoint.
pub struct Gpt2ForCausalLM {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// LM head weight `[vocab_size × hidden_size]`.
    lm_weight: Vec<Vec<f32>>,
}

impl Gpt2ForCausalLM {
    /// Create a new causal LM head.
    pub fn new(hidden_size: usize, vocab_size: usize) -> Result<Self, Gpt2TaskError> {
        if hidden_size == 0 {
            return Err(Gpt2TaskError::InvalidConfig(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if vocab_size == 0 {
            return Err(Gpt2TaskError::InvalidConfig(
                "vocab_size must be > 0".to_string(),
            ));
        }
        Ok(Self {
            vocab_size,
            hidden_size,
            lm_weight: vec![vec![0.0f32; hidden_size]; vocab_size],
        })
    }

    /// Compute next-token logits from the last hidden state.
    ///
    /// # Arguments
    /// * `last_hidden` - `[hidden_size]` slice for the last token position
    pub fn compute_logits(&self, last_hidden: &[f32]) -> Result<Vec<f32>, Gpt2TaskError> {
        if last_hidden.len() != self.hidden_size {
            return Err(Gpt2TaskError::ForwardError(format!(
                "expected hidden_size={}, got {}",
                self.hidden_size,
                last_hidden.len()
            )));
        }
        let logits: Vec<f32> = self
            .lm_weight
            .iter()
            .map(|row| row.iter().zip(last_hidden.iter()).map(|(w, x)| w * x).sum())
            .collect();
        Ok(logits)
    }

    /// Greedy forward: return the argmax token id given the last hidden state.
    pub fn forward_greedy(&self, last_hidden: &[f32]) -> Result<u32, Gpt2TaskError> {
        let logits = self.compute_logits(last_hidden)?;
        greedy_decode(&logits).ok_or_else(|| Gpt2TaskError::ForwardError("argmax failed".into()))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── test_gpt2_config_small ────────────────────────────────────────────

    #[test]
    fn test_gpt2_config_small() {
        use crate::gpt2::config::Gpt2Config;
        use trustformers_core::traits::Config;
        let cfg = Gpt2Config::small();
        assert_eq!(cfg.vocab_size, 50257);
        assert_eq!(cfg.n_embd, 768);
        assert_eq!(cfg.n_layer, 12);
        assert_eq!(cfg.n_head, 12);
        assert!(cfg.validate().is_ok());
    }

    // ── test_gpt2_config_medium ───────────────────────────────────────────

    #[test]
    fn test_gpt2_config_medium() {
        use crate::gpt2::config::Gpt2Config;
        let cfg = Gpt2Config::medium();
        assert_eq!(cfg.n_embd, 1024);
        assert_eq!(cfg.n_head, 16);
        assert_eq!(cfg.n_layer, 24);
    }

    // ── test_gpt2_config_xl ───────────────────────────────────────────────

    #[test]
    fn test_gpt2_config_xl() {
        use crate::gpt2::config::Gpt2Config;
        let cfg = Gpt2Config::xl();
        assert_eq!(cfg.n_embd, 1600);
        assert_eq!(cfg.n_head, 25);
    }

    // ── test_gpt2_gelu_zero ───────────────────────────────────────────────

    #[test]
    fn test_gpt2_gelu_zero() {
        // gelu(0) = 0.5 * 0 * (...) = 0
        assert!(gelu(0.0).abs() < 1e-6);
    }

    // ── test_gpt2_gelu_positive ───────────────────────────────────────────

    #[test]
    fn test_gpt2_gelu_positive() {
        // For large positive x, gelu(x) ≈ x
        let v = gelu(10.0);
        assert!((v - 10.0).abs() < 1e-3, "gelu(10) ≈ 10, got {v}");
    }

    // ── test_gpt2_gelu_negative ───────────────────────────────────────────

    #[test]
    fn test_gpt2_gelu_negative() {
        // For large negative x, gelu(x) ≈ 0
        let v = gelu(-10.0);
        assert!(v.abs() < 1e-3, "gelu(-10) ≈ 0, got {v}");
    }

    // ── test_gpt2_gelu_vec_length ─────────────────────────────────────────

    #[test]
    fn test_gpt2_gelu_vec_length() {
        let xs = vec![1.0f32, -1.0, 0.0, 2.0, -2.0];
        let out = gelu_vec(&xs);
        assert_eq!(out.len(), xs.len());
    }

    // ── test_gpt2_layer_norm_zero_mean ────────────────────────────────────

    #[test]
    fn test_gpt2_layer_norm_zero_mean() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = layer_norm(&x, 1e-5);
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(
            mean.abs() < 1e-4,
            "layer_norm output should have ~0 mean, got {mean}"
        );
    }

    // ── test_gpt2_layer_norm_unit_variance ────────────────────────────────

    #[test]
    fn test_gpt2_layer_norm_unit_variance() {
        let x = vec![1.0f32, 3.0, 5.0, 7.0];
        let out = layer_norm(&x, 1e-5);
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        let var: f32 = out.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / out.len() as f32;
        assert!(
            (var - 1.0).abs() < 1e-4,
            "layer_norm output variance should be ~1, got {var}"
        );
    }

    // ── test_gpt2_softmax_sums_to_one ─────────────────────────────────────

    #[test]
    fn test_gpt2_softmax_sums_to_one() {
        let logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1, got {sum}");
    }

    // ── test_gpt2_softmax_max_is_last ─────────────────────────────────────

    #[test]
    fn test_gpt2_softmax_max_is_last() {
        let logits = vec![0.0f32, 1.0, 2.0, 5.0];
        let probs = softmax(&logits);
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);
        assert_eq!(max_idx, Some(3));
    }

    // ── test_gpt2_greedy_decode ───────────────────────────────────────────

    #[test]
    fn test_gpt2_greedy_decode() {
        let logits = vec![0.1f32, 0.9, 0.3, 0.7];
        let token = greedy_decode(&logits);
        assert_eq!(token, Some(1u32));
    }

    // ── test_gpt2_top_k_filter ────────────────────────────────────────────

    #[test]
    fn test_gpt2_top_k_filter() {
        let logits = vec![1.0f32, 5.0, 3.0, 2.0, 4.0];
        let filtered = top_k_filter(&logits, 2).expect("top_k_filter should succeed");
        // Top 2 are indices 1 (5.0) and 4 (4.0)
        assert!(filtered[0].is_infinite() && filtered[0] < 0.0); // 1.0 masked
        assert!((filtered[1] - 5.0).abs() < 1e-6); // 5.0 kept
        assert!(filtered[2].is_infinite() && filtered[2] < 0.0); // 3.0 masked
        assert!(filtered[3].is_infinite() && filtered[3] < 0.0); // 2.0 masked
        assert!((filtered[4] - 4.0).abs() < 1e-6); // 4.0 kept
    }

    // ── test_gpt2_top_k_too_large ─────────────────────────────────────────

    #[test]
    fn test_gpt2_top_k_too_large() {
        let logits = vec![1.0f32, 2.0];
        let err = top_k_filter(&logits, 5);
        assert!(matches!(
            err,
            Err(Gpt2TaskError::TopKTooLarge {
                k: 5,
                vocab_size: 2
            })
        ));
    }

    // ── test_gpt2_top_p_filter ────────────────────────────────────────────

    #[test]
    fn test_gpt2_top_p_filter() {
        let logits = vec![0.0f32, 10.0, 0.0]; // dominated by index 1
        let filtered = top_p_filter(&logits, 0.99).expect("top_p_filter should succeed");
        // After softmax the mass is almost entirely on index 1
        assert!((filtered[1] - 10.0).abs() < 1e-6);
    }

    // ── test_gpt2_top_p_invalid_nucleus ───────────────────────────────────

    #[test]
    fn test_gpt2_top_p_invalid_nucleus() {
        let logits = vec![1.0f32, 2.0];
        assert!(matches!(
            top_p_filter(&logits, 0.0),
            Err(Gpt2TaskError::InvalidNucleus(_))
        ));
        assert!(matches!(
            top_p_filter(&logits, 1.1),
            Err(Gpt2TaskError::InvalidNucleus(_))
        ));
    }

    // ── test_gpt2_temperature_sharpens ────────────────────────────────────

    #[test]
    fn test_gpt2_temperature_sharpens() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let hot = apply_temperature(&logits, 2.0);
        let cold = apply_temperature(&logits, 0.5);
        // Cold temperature: values are scaled up (more extreme)
        assert!(
            (cold[2] - 6.0).abs() < 1e-5,
            "cold[2] should be 6, got {}",
            cold[2]
        );
        // Hot temperature: values are scaled down (more uniform)
        assert!(
            (hot[2] - 1.5).abs() < 1e-5,
            "hot[2] should be 1.5, got {}",
            hot[2]
        );
    }

    // ── test_gpt2_sinusoidal_pos_embed_shape ──────────────────────────────

    #[test]
    fn test_gpt2_sinusoidal_pos_embed_shape() {
        let positions = vec![0usize, 1, 2, 3];
        let d_model = 16;
        let embeds = sinusoidal_pos_embed(&positions, d_model);
        assert_eq!(embeds.len(), 4);
        for row in &embeds {
            assert_eq!(row.len(), d_model);
        }
    }

    // ── test_gpt2_sinusoidal_pos_zero ─────────────────────────────────────

    #[test]
    fn test_gpt2_sinusoidal_pos_zero() {
        let embeds = sinusoidal_pos_embed(&[0], 8);
        // Position 0: angle=0, cos(0)=1.0 for even dims
        assert!(
            (embeds[0][0] - 1.0).abs() < 1e-5,
            "pos=0 even dim should be cos(0)=1"
        );
    }

    // ── test_gpt2_seq_cls_construction ───────────────────────────────────

    #[test]
    fn test_gpt2_seq_cls_construction() {
        let head = Gpt2ForSequenceClassification::new(64, 3);
        assert!(head.is_ok());
        let head = head.expect("seq cls head");
        assert_eq!(head.num_labels, 3);
        assert_eq!(head.hidden_size, 64);
    }

    // ── test_gpt2_seq_cls_invalid_config ─────────────────────────────────

    #[test]
    fn test_gpt2_seq_cls_invalid_config() {
        assert!(matches!(
            Gpt2ForSequenceClassification::new(0, 3),
            Err(Gpt2TaskError::InvalidConfig(_))
        ));
        assert!(matches!(
            Gpt2ForSequenceClassification::new(64, 0),
            Err(Gpt2TaskError::InvalidConfig(_))
        ));
    }

    // ── test_gpt2_seq_cls_forward_output_shape ────────────────────────────

    #[test]
    fn test_gpt2_seq_cls_forward_output_shape() {
        let head = Gpt2ForSequenceClassification::new(8, 4).expect("head");
        let hidden = vec![0.5f32; 24]; // seq_len=3, hidden=8
        let logits = head.forward(&hidden).expect("seq cls forward");
        assert_eq!(logits.len(), 4);
    }

    // ── test_gpt2_seq_cls_empty_input ─────────────────────────────────────

    #[test]
    fn test_gpt2_seq_cls_empty_input() {
        let head = Gpt2ForSequenceClassification::new(8, 2).expect("head");
        assert!(matches!(head.forward(&[]), Err(Gpt2TaskError::EmptyInput)));
    }

    // ── test_gpt2_token_cls_forward_shape ─────────────────────────────────

    #[test]
    fn test_gpt2_token_cls_forward_shape() {
        let head = Gpt2ForTokenClassification::new(16, 5).expect("head");
        // seq_len=4, hidden=16
        let hidden = vec![0.1f32; 64];
        let logits = head.forward(&hidden).expect("token cls forward");
        assert_eq!(logits.len(), 4, "one row per token");
        for row in &logits {
            assert_eq!(row.len(), 5, "each row has num_labels logits");
        }
    }

    // ── test_gpt2_token_cls_single_token ─────────────────────────────────

    #[test]
    fn test_gpt2_token_cls_single_token() {
        let head = Gpt2ForTokenClassification::new(4, 2).expect("head");
        let hidden = vec![1.0f32, 2.0, 3.0, 4.0]; // seq_len=1
        let logits = head.forward(&hidden).expect("forward");
        assert_eq!(logits.len(), 1);
        assert_eq!(logits[0].len(), 2);
    }

    // ── test_gpt2_causal_lm_construction ──────────────────────────────────

    #[test]
    fn test_gpt2_causal_lm_construction() {
        let head = Gpt2ForCausalLM::new(32, 100);
        assert!(head.is_ok());
        let h = head.expect("causal lm head");
        assert_eq!(h.vocab_size, 100);
        assert_eq!(h.hidden_size, 32);
    }

    // ── test_gpt2_causal_lm_forward_greedy ────────────────────────────────

    #[test]
    fn test_gpt2_causal_lm_forward_greedy() {
        // Zero weights → all logits = 0 → greedy returns last max-index (vocab_size - 1)
        let head = Gpt2ForCausalLM::new(4, 10).expect("causal lm head");
        let last_hidden = vec![1.0f32, 2.0, 3.0, 4.0];
        let token = head.forward_greedy(&last_hidden).expect("forward_greedy");
        // With zero weights every logit = 0; max_by returns last equal-max, i.e. index 9
        assert!(token < 10u32, "token {token} must be within vocab_size=10");
    }

    // ── test_gpt2_causal_lm_dim_mismatch ─────────────────────────────────

    #[test]
    fn test_gpt2_causal_lm_dim_mismatch() {
        let head = Gpt2ForCausalLM::new(8, 10).expect("causal lm head");
        // Wrong hidden size
        let err = head.compute_logits(&[1.0f32; 4]);
        assert!(matches!(err, Err(Gpt2TaskError::ForwardError(_))));
    }

    // ── test_gpt2_error_display ───────────────────────────────────────────

    #[test]
    fn test_gpt2_error_display() {
        let e = Gpt2TaskError::EmptyInput;
        assert!(e.to_string().contains("empty"));

        let e2 = Gpt2TaskError::TopKTooLarge {
            k: 10,
            vocab_size: 5,
        };
        let s = e2.to_string();
        assert!(s.contains("10") && s.contains("5"));

        let e3 = Gpt2TaskError::InvalidConfig("bad".to_string());
        assert!(e3.to_string().contains("bad"));

        let e4 = Gpt2TaskError::InvalidNucleus(1.5);
        assert!(e4.to_string().contains("1.5"));
    }

    // ── test_gpt2_lcg_diverse_inputs ──────────────────────────────────────

    #[test]
    fn test_gpt2_lcg_diverse_inputs() {
        let mut state = 17u64;
        for _ in 0..8 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let hidden_size = ((state % 8) + 2) as usize * 4;
            let num_labels = ((state >> 8) % 5 + 2) as usize;
            let head =
                Gpt2ForSequenceClassification::new(hidden_size, num_labels).expect("seq cls head");
            let hidden: Vec<f32> = (0..hidden_size * 3).map(|i| (i as f32) * 0.01).collect();
            let out = head.forward(&hidden).expect("forward");
            assert_eq!(out.len(), num_labels);
        }
    }
}
