//! # StarCoder2 Task-Specific Implementations
//!
//! This module provides task-specific utilities for StarCoder2 code-generation
//! models, including:
//! - Fill-In-the-Middle (FIM) prompt helpers (PSM and SPM formats)
//! - Code prefix/suffix completion formatting
//! - Causal LM head utilities (RMSNorm, greedy decode, top-k/top-p)
//! - Sequence classification head
//! - Sliding-window attention mask utilities specific to StarCoder2

use std::fmt;

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors produced by StarCoder2 task-specific operations.
#[derive(Debug)]
pub enum StarCoder2TaskError {
    /// Empty input was provided.
    EmptyInput,
    /// A required field (prefix, suffix, …) was empty.
    EmptyField(&'static str),
    /// Top-k value exceeds vocabulary size.
    TopKTooLarge { k: usize, vocab_size: usize },
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Forward pass failed.
    ForwardError(String),
    /// Nucleus probability out of range.
    InvalidNucleus(f32),
}

impl fmt::Display for StarCoder2TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StarCoder2TaskError::EmptyInput => {
                write!(f, "StarCoder2 task error: empty input")
            },
            StarCoder2TaskError::EmptyField(field) => {
                write!(f, "StarCoder2 task error: empty field '{field}'")
            },
            StarCoder2TaskError::TopKTooLarge { k, vocab_size } => write!(
                f,
                "StarCoder2 task error: top_k={k} exceeds vocab_size={vocab_size}"
            ),
            StarCoder2TaskError::InvalidConfig(msg) => {
                write!(f, "StarCoder2 task error: invalid config: {msg}")
            },
            StarCoder2TaskError::ForwardError(msg) => {
                write!(f, "StarCoder2 task error: forward error: {msg}")
            },
            StarCoder2TaskError::InvalidNucleus(p) => {
                write!(
                    f,
                    "StarCoder2 task error: nucleus probability {p} out of (0,1]"
                )
            },
        }
    }
}

impl std::error::Error for StarCoder2TaskError {}

// ─── FIM format constants ────────────────────────────────────────────────────

/// `<fim_prefix>` special token string.
pub const FIM_PREFIX: &str = "<fim_prefix>";
/// `<fim_suffix>` special token string.
pub const FIM_SUFFIX: &str = "<fim_suffix>";
/// `<fim_middle>` special token string.
pub const FIM_MIDDLE: &str = "<fim_middle>";
/// `<fim_pad>` special token string.
pub const FIM_PAD: &str = "<fim_pad>";
/// End-of-sequence token for StarCoder2.
pub const END_OF_TEXT: &str = "<|endoftext|>";

// ─── FIM prompt builders ──────────────────────────────────────────────────────

/// Format a PSM (Prefix–Suffix–Middle) FIM prompt.
///
/// ```text
/// <fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>
/// ```
///
/// The model generates the *middle* from here.
///
/// # Errors
///
/// Returns [`StarCoder2TaskError::EmptyField`] if either `prefix` or `suffix`
/// is empty after trimming.
pub fn format_psm_prompt(prefix: &str, suffix: &str) -> Result<String, StarCoder2TaskError> {
    if prefix.trim().is_empty() {
        return Err(StarCoder2TaskError::EmptyField("prefix"));
    }
    if suffix.trim().is_empty() {
        return Err(StarCoder2TaskError::EmptyField("suffix"));
    }
    Ok(format!(
        "{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    ))
}

/// Format an SPM (Suffix–Prefix–Middle) FIM prompt.
///
/// ```text
/// <fim_suffix>{suffix}<fim_prefix>{prefix}<fim_middle>
/// ```
///
/// # Errors
///
/// Returns [`StarCoder2TaskError::EmptyField`] if either `prefix` or `suffix`
/// is empty after trimming.
pub fn format_spm_prompt(prefix: &str, suffix: &str) -> Result<String, StarCoder2TaskError> {
    if prefix.trim().is_empty() {
        return Err(StarCoder2TaskError::EmptyField("prefix"));
    }
    if suffix.trim().is_empty() {
        return Err(StarCoder2TaskError::EmptyField("suffix"));
    }
    Ok(format!(
        "{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}"
    ))
}

/// Extract the generated middle section from a raw FIM completion.
///
/// Looks for `<fim_middle>` and returns the text up to `<|endoftext|>` (or the
/// end of the string if no EOS is found).  Returns `None` if the marker is
/// absent.
pub fn parse_fim_middle(output: &str) -> Option<String> {
    let start = output.find(FIM_MIDDLE)?;
    let after = &output[start + FIM_MIDDLE.len()..];
    let middle = match after.find(END_OF_TEXT) {
        Some(eot) => &after[..eot],
        None => after,
    };
    Some(middle.to_string())
}

// ─── RMSNorm ──────────────────────────────────────────────────────────────────

/// Apply RMSNorm to a flat slice (unit weights, no bias).
///
/// `output[i] = input[i] / sqrt(mean(input^2) + eps)`
pub fn rms_norm(input: &[f32], eps: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let mean_sq = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
    let rms = (mean_sq + eps).sqrt();
    input.iter().map(|x| x / rms).collect()
}

// ─── Softmax ──────────────────────────────────────────────────────────────────

/// Numerically stable softmax.
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

/// Keep only the top-k logits; mask the rest with `f32::NEG_INFINITY`.
pub fn top_k_filter(logits: &[f32], k: usize) -> Result<Vec<f32>, StarCoder2TaskError> {
    let vocab = logits.len();
    if k > vocab {
        return Err(StarCoder2TaskError::TopKTooLarge {
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

// ─── Sliding-window mask ──────────────────────────────────────────────────────

/// Apply a sliding-window causal mask to a flat `seq_len × seq_len` attention
/// score matrix (row-major).
///
/// Any pair `(query, key)` with `|query - key| > window_size` or `key > query`
/// (future token) is set to `f32::NEG_INFINITY`.
pub fn apply_sliding_window_causal_mask(scores: &mut [f32], seq_len: usize, window_size: usize) {
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                // future token → causal mask
                scores[i * seq_len + j] = f32::NEG_INFINITY;
            } else {
                let diff = i - j;
                if diff > window_size {
                    scores[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
}

/// Compute the fraction of token pairs visible under the sliding-window causal
/// mask.
pub fn sliding_window_causal_coverage(seq_len: usize, window_size: usize) -> f32 {
    if seq_len == 0 {
        return 0.0;
    }
    let mut visible = 0usize;
    for i in 0..seq_len {
        for j in 0..=i {
            if i - j <= window_size {
                visible += 1;
            }
        }
    }
    visible as f32 / (seq_len * seq_len) as f32
}

// ─── SwiGLU ──────────────────────────────────────────────────────────────────

/// SiLU (Swish) activation: `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply SwiGLU: `gate[i] * silu(gate[i]) * up[i]` element-wise.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up.iter()).map(|(&g, &u)| silu(g) * u).collect()
}

// ─── Causal LM head ───────────────────────────────────────────────────────────

/// StarCoder2 causal LM head.
///
/// Maps the final hidden state `[hidden_size]` to vocabulary logits
/// `[vocab_size]`.  All weights are initialised to zero for test use.
pub struct StarCoder2ForCausalLM {
    /// Model vocabulary size.
    pub vocab_size: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// LM head weight `[vocab_size × hidden_size]`.
    lm_weight: Vec<Vec<f32>>,
}

impl StarCoder2ForCausalLM {
    /// Create a new causal LM head.
    pub fn new(hidden_size: usize, vocab_size: usize) -> Result<Self, StarCoder2TaskError> {
        if hidden_size == 0 {
            return Err(StarCoder2TaskError::InvalidConfig(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if vocab_size == 0 {
            return Err(StarCoder2TaskError::InvalidConfig(
                "vocab_size must be > 0".to_string(),
            ));
        }
        Ok(Self {
            vocab_size,
            hidden_size,
            lm_weight: vec![vec![0.0f32; hidden_size]; vocab_size],
        })
    }

    /// Compute vocabulary logits from the last hidden state.
    pub fn compute_logits(&self, last_hidden: &[f32]) -> Result<Vec<f32>, StarCoder2TaskError> {
        if last_hidden.len() != self.hidden_size {
            return Err(StarCoder2TaskError::ForwardError(format!(
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

    /// Greedy forward: argmax token from the last hidden state.
    pub fn forward_greedy(&self, last_hidden: &[f32]) -> Result<u32, StarCoder2TaskError> {
        let logits = self.compute_logits(last_hidden)?;
        greedy_decode(&logits)
            .ok_or_else(|| StarCoder2TaskError::ForwardError("argmax failed".into()))
    }
}

// ─── Sequence classification head ────────────────────────────────────────────

/// Sequence classification head for StarCoder2.
///
/// Pools the last-token hidden state and projects to `[num_labels]` logits.
pub struct StarCoder2ForSequenceClassification {
    /// Number of output labels.
    pub num_labels: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Weight matrix `[num_labels × hidden_size]`.
    weight: Vec<Vec<f32>>,
    /// Bias vector `[num_labels]`.
    bias: Vec<f32>,
}

impl StarCoder2ForSequenceClassification {
    /// Create a new sequence classification head.
    pub fn new(hidden_size: usize, num_labels: usize) -> Result<Self, StarCoder2TaskError> {
        if hidden_size == 0 {
            return Err(StarCoder2TaskError::InvalidConfig(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if num_labels == 0 {
            return Err(StarCoder2TaskError::InvalidConfig(
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

    /// Pool last-token hidden state and project to label logits.
    pub fn forward(&self, hidden_states: &[f32]) -> Result<Vec<f32>, StarCoder2TaskError> {
        if hidden_states.is_empty() {
            return Err(StarCoder2TaskError::EmptyInput);
        }
        let seq_len = hidden_states.len() / self.hidden_size;
        if seq_len == 0 {
            return Err(StarCoder2TaskError::EmptyInput);
        }
        let start = (seq_len - 1) * self.hidden_size;
        let last_token = &hidden_states[start..start + self.hidden_size];
        let logits: Vec<f32> = self
            .weight
            .iter()
            .zip(self.bias.iter())
            .map(|(row, &b)| row.iter().zip(last_token.iter()).map(|(w, x)| w * x).sum::<f32>() + b)
            .collect();
        Ok(logits)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── test_starcoder2_config_default ────────────────────────────────────

    #[test]
    fn test_starcoder2_config_default() {
        use crate::starcoder2::config::StarCoder2Config;
        use trustformers_core::traits::Config;
        let cfg = StarCoder2Config::default();
        assert_eq!(cfg.vocab_size, 49152);
        assert_eq!(cfg.hidden_size, 3072);
        assert!(cfg.validate().is_ok());
    }

    // ── test_starcoder2_config_3b ─────────────────────────────────────────

    #[test]
    fn test_starcoder2_config_3b() {
        use crate::starcoder2::config::StarCoder2Config;
        let cfg = StarCoder2Config::starcoder2_3b();
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.num_query_groups(), 12);
        assert!(cfg.use_bias);
    }

    // ── test_starcoder2_config_15b ────────────────────────────────────────

    #[test]
    fn test_starcoder2_config_15b() {
        use crate::starcoder2::config::StarCoder2Config;
        let cfg = StarCoder2Config::starcoder2_15b();
        assert_eq!(cfg.hidden_size, 6144);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.head_dim(), 128);
    }

    // ── test_starcoder2_psm_prompt ────────────────────────────────────────

    #[test]
    fn test_starcoder2_psm_prompt() {
        let p = format_psm_prompt("def foo():\n    ", "\n    return x").expect("psm");
        assert!(p.starts_with(FIM_PREFIX));
        assert!(p.contains(FIM_SUFFIX));
        assert!(p.ends_with(FIM_MIDDLE));
        assert!(p.contains("def foo()"));
    }

    // ── test_starcoder2_spm_prompt ────────────────────────────────────────

    #[test]
    fn test_starcoder2_spm_prompt() {
        let p = format_spm_prompt("def bar():\n    ", "\n    pass").expect("spm");
        assert!(p.starts_with(FIM_SUFFIX));
        assert!(p.contains(FIM_PREFIX));
        assert!(p.ends_with(FIM_MIDDLE));
    }

    // ── test_starcoder2_psm_empty_prefix ──────────────────────────────────

    #[test]
    fn test_starcoder2_psm_empty_prefix() {
        let err = format_psm_prompt("  ", "suffix");
        assert!(matches!(
            err,
            Err(StarCoder2TaskError::EmptyField("prefix"))
        ));
    }

    // ── test_starcoder2_psm_empty_suffix ──────────────────────────────────

    #[test]
    fn test_starcoder2_psm_empty_suffix() {
        let err = format_psm_prompt("prefix", "  ");
        assert!(matches!(
            err,
            Err(StarCoder2TaskError::EmptyField("suffix"))
        ));
    }

    // ── test_starcoder2_parse_fim_middle ──────────────────────────────────

    #[test]
    fn test_starcoder2_parse_fim_middle() {
        let raw = "<fim_prefix>prefix<fim_suffix>suffix<fim_middle>GENERATED<|endoftext|>";
        let middle = parse_fim_middle(raw).expect("parse_fim_middle");
        assert_eq!(middle, "GENERATED");
    }

    // ── test_starcoder2_parse_fim_middle_no_eos ───────────────────────────

    #[test]
    fn test_starcoder2_parse_fim_middle_no_eos() {
        let raw = "<fim_middle>no_eos_here";
        let middle = parse_fim_middle(raw).expect("parse_fim_middle");
        assert_eq!(middle, "no_eos_here");
    }

    // ── test_starcoder2_parse_fim_missing_marker ──────────────────────────

    #[test]
    fn test_starcoder2_parse_fim_missing_marker() {
        assert!(parse_fim_middle("no marker here").is_none());
    }

    // ── test_starcoder2_rms_norm ──────────────────────────────────────────

    #[test]
    fn test_starcoder2_rms_norm() {
        let x = vec![3.0f32, 4.0];
        let out = rms_norm(&x, 1e-5);
        let rms = (12.5f32 + 1e-5).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-5);
        assert!((out[1] - 4.0 / rms).abs() < 1e-5);
    }

    // ── test_starcoder2_rms_norm_constant ────────────────────────────────

    #[test]
    fn test_starcoder2_rms_norm_constant() {
        let x = vec![2.0f32; 4];
        let out = rms_norm(&x, 1e-8);
        for &v in &out {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "constant input must normalize to 1, got {v}"
            );
        }
    }

    // ── test_starcoder2_sliding_window_causal_mask ────────────────────────

    #[test]
    fn test_starcoder2_sliding_window_causal_mask() {
        let seq_len = 4;
        let mut scores = vec![1.0f32; seq_len * seq_len];
        apply_sliding_window_causal_mask(&mut scores, seq_len, 1);
        // (0,0): diff=0 ≤ 1, not future → kept
        assert!((scores[0] - 1.0).abs() < 1e-6);
        // (1,0): diff=1 ≤ 1, not future → kept
        assert!((scores[seq_len] - 1.0).abs() < 1e-6);
        // (3,0): diff=3 > 1 → masked
        assert!(scores[3 * seq_len].is_infinite() && scores[3 * seq_len] < 0.0);
        // (0,1): future token → masked
        assert!(scores[1].is_infinite() && scores[1] < 0.0);
    }

    // ── test_starcoder2_sliding_window_coverage ───────────────────────────

    #[test]
    fn test_starcoder2_sliding_window_coverage() {
        // Full causal attention: window = seq_len - 1
        let full = sliding_window_causal_coverage(4, 3);
        // All causal pairs visible: 1+2+3+4=10 out of 16 total
        let expected = 10.0 / 16.0;
        assert!(
            (full - expected).abs() < 1e-5,
            "expected {expected}, got {full}"
        );
    }

    // ── test_starcoder2_swiglu ────────────────────────────────────────────

    #[test]
    fn test_starcoder2_swiglu() {
        let gate = vec![1.0f32, -1.0, 0.5];
        let up = vec![2.0f32, 2.0, 2.0];
        let out = swiglu(&gate, &up);
        assert_eq!(out.len(), 3);
        assert!(out[0] > 0.0, "positive gate → positive output");
        assert!(out[1] < 0.0, "negative gate → negative output");
    }

    // ── test_starcoder2_causal_lm_construction ────────────────────────────

    #[test]
    fn test_starcoder2_causal_lm_construction() {
        let head = StarCoder2ForCausalLM::new(64, 1024);
        assert!(head.is_ok());
        let h = head.expect("causal lm");
        assert_eq!(h.vocab_size, 1024);
        assert_eq!(h.hidden_size, 64);
    }

    // ── test_starcoder2_causal_lm_greedy ──────────────────────────────────

    #[test]
    fn test_starcoder2_causal_lm_greedy() {
        let head = StarCoder2ForCausalLM::new(4, 10).expect("causal lm");
        let token = head.forward_greedy(&[0.0f32; 4]).expect("greedy");
        // zero weights → all logits 0 → max_by returns last equal-max index
        assert!(token < 10u32, "token {token} must be within vocab_size=10");
    }

    // ── test_starcoder2_causal_lm_dim_mismatch ────────────────────────────

    #[test]
    fn test_starcoder2_causal_lm_dim_mismatch() {
        let head = StarCoder2ForCausalLM::new(8, 10).expect("causal lm");
        assert!(matches!(
            head.compute_logits(&[0.0f32; 4]),
            Err(StarCoder2TaskError::ForwardError(_))
        ));
    }

    // ── test_starcoder2_seq_cls_forward ───────────────────────────────────

    #[test]
    fn test_starcoder2_seq_cls_forward() {
        let head = StarCoder2ForSequenceClassification::new(8, 3).expect("seq cls");
        let hidden = vec![0.5f32; 24]; // seq_len=3, hidden=8
        let logits = head.forward(&hidden).expect("forward");
        assert_eq!(logits.len(), 3);
    }

    // ── test_starcoder2_seq_cls_empty ─────────────────────────────────────

    #[test]
    fn test_starcoder2_seq_cls_empty() {
        let head = StarCoder2ForSequenceClassification::new(4, 2).expect("seq cls");
        assert!(matches!(
            head.forward(&[]),
            Err(StarCoder2TaskError::EmptyInput)
        ));
    }

    // ── test_starcoder2_top_k_filter ─────────────────────────────────────

    #[test]
    fn test_starcoder2_top_k_filter() {
        let logits = vec![1.0f32, 5.0, 3.0, 2.0];
        let f = top_k_filter(&logits, 2).expect("top_k");
        assert!((f[1] - 5.0).abs() < 1e-6);
        assert!((f[2] - 3.0).abs() < 1e-6);
        assert!(f[0].is_infinite() && f[0] < 0.0);
    }

    // ── test_starcoder2_softmax ───────────────────────────────────────────

    #[test]
    fn test_starcoder2_softmax() {
        let logits = vec![0.0f32, 1.0, 2.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    // ── test_starcoder2_error_display ─────────────────────────────────────

    #[test]
    fn test_starcoder2_error_display() {
        let e1 = StarCoder2TaskError::EmptyInput;
        assert!(e1.to_string().contains("empty"));

        let e2 = StarCoder2TaskError::EmptyField("prefix");
        assert!(e2.to_string().contains("prefix"));

        let e3 = StarCoder2TaskError::TopKTooLarge {
            k: 10,
            vocab_size: 3,
        };
        assert!(e3.to_string().contains("10") && e3.to_string().contains("3"));

        let e4 = StarCoder2TaskError::InvalidConfig("bad".to_string());
        assert!(e4.to_string().contains("bad"));
    }

    // ── test_starcoder2_lcg_varied_configs ───────────────────────────────

    #[test]
    fn test_starcoder2_lcg_varied_configs() {
        let mut state = 31u64;
        for _ in 0..6 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let hidden = ((state % 4) + 2) as usize * 8;
            let labels = ((state >> 4) % 4 + 2) as usize;
            let head = StarCoder2ForSequenceClassification::new(hidden, labels).expect("head");
            let hs: Vec<f32> = (0..hidden * 2).map(|i| i as f32 * 0.01).collect();
            let out = head.forward(&hs).expect("forward");
            assert_eq!(out.len(), labels);
        }
    }
}
