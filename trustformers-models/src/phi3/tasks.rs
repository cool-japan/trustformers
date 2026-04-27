//! # Phi-3 Task-Specific Implementations
//!
//! This module provides task-specific functionality for Phi-3 models, including:
//! - Causal language modeling
//! - Chat prompt formatting (Phi-3 instruction template)
//! - Sliding window attention coverage utilities
//! - Greedy text generation

use std::fmt;

/// Errors specific to Phi-3 operations
#[derive(Debug)]
pub enum Phi3Error {
    /// Invalid configuration parameter
    InvalidConfig(String),
    /// Tensor shape mismatch
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    /// Sequence too long for the configured window
    SequenceTooLong { max: usize, got: usize },
    /// Forward pass computation error
    ForwardError(String),
    /// Generation error
    GenerationError(String),
    /// Empty input
    EmptyInput,
}

impl fmt::Display for Phi3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Phi3Error::InvalidConfig(msg) => write!(f, "Phi3 invalid config: {}", msg),
            Phi3Error::ShapeMismatch { expected, got } => {
                write!(
                    f,
                    "Phi3 shape mismatch: expected {:?}, got {:?}",
                    expected, got
                )
            },
            Phi3Error::SequenceTooLong { max, got } => {
                write!(
                    f,
                    "Phi3 sequence too long: max {}, got {}",
                    max, got
                )
            },
            Phi3Error::ForwardError(msg) => write!(f, "Phi3 forward error: {}", msg),
            Phi3Error::GenerationError(msg) => write!(f, "Phi3 generation error: {}", msg),
            Phi3Error::EmptyInput => write!(f, "Phi3 error: empty input"),
        }
    }
}

impl std::error::Error for Phi3Error {}

/// Apply a sliding window mask to attention scores in-place.
///
/// For any (query_pos, key_pos) pair where `|query_pos - key_pos| > window_size`,
/// the corresponding score is set to `f32::NEG_INFINITY` so that softmax
/// produces zero weight for out-of-window tokens.
///
/// # Arguments
/// * `scores`      - Flat `seq_len × seq_len` attention score matrix (row-major)
/// * `seq_len`     - Length of the sequence
/// * `window_size` - Sliding window radius (tokens ≤ `window_size` steps away are visible)
pub fn apply_sliding_window_mask(scores: &mut [f32], seq_len: usize, window_size: usize) {
    for i in 0..seq_len {
        for j in 0..seq_len {
            let diff = if i >= j { i - j } else { j - i };
            if diff > window_size {
                scores[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Compute the fraction of token pairs covered by the sliding window.
///
/// A pair (i, j) is covered when `|i - j| <= window_size`.
/// For a sequence of length `seq_len`, the total number of pairs is
/// `seq_len^2` and the number covered by the window is approximately
/// `window_size * seq_len - window_size^2 / 2`.
///
/// Returns a value in `[0.0, 1.0]`.
pub fn sliding_window_coverage(seq_len: usize, window_size: usize) -> f32 {
    if seq_len == 0 {
        return 0.0;
    }
    // Count covered pairs exactly
    let mut covered: usize = 0;
    for i in 0..seq_len {
        for j in 0..seq_len {
            let diff = if i >= j { i - j } else { j - i };
            if diff <= window_size {
                covered += 1;
            }
        }
    }
    covered as f32 / (seq_len * seq_len) as f32
}

/// Format a Phi-3 chat prompt using the standard instruction template.
///
/// Template:
/// ```text
/// <|system|>
/// {system}<|end|>
/// <|user|>
/// {user}<|end|>
/// <|assistant|>
/// ```
///
/// If no system message is provided the system turn is omitted.
pub fn format_chat_prompt(system: Option<&str>, user: &str) -> String {
    let mut prompt = String::new();
    if let Some(sys) = system {
        prompt.push_str("<|system|>\n");
        prompt.push_str(sys);
        prompt.push_str("<|end|>\n");
    }
    prompt.push_str("<|user|>\n");
    prompt.push_str(user);
    prompt.push_str("<|end|>\n<|assistant|>\n");
    prompt
}

/// Pure-Rust RoPE embedding application (used in standalone forward computation).
///
/// Rotates pairs of elements in `q` and `k` using cosine / sine frequencies
/// derived from `rope_theta` and an optional per-dimension scale factor.
///
/// # Arguments
/// * `q`          - Query vector slice (length = `seq_len * head_dim`)
/// * `k`          - Key vector slice   (length = `seq_len * head_dim`)
/// * `seq_len`    - Number of tokens
/// * `head_dim`   - Dimension of each attention head
/// * `rope_theta` - Base frequency for RoPE
/// * `scale_factors` - Optional per-pair scale (length = `head_dim / 2`)
///
/// Returns `(rotated_q, rotated_k)`.
pub fn apply_rope(
    q: &[f32],
    k: &[f32],
    seq_len: usize,
    head_dim: usize,
    rope_theta: f64,
    scale_factors: Option<&[f64]>,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = head_dim / 2;
    let mut q_out = q.to_vec();
    let mut k_out = k.to_vec();

    for pos in 0..seq_len {
        for i in 0..half_dim {
            let scale = scale_factors.map(|sf| sf.get(i).copied().unwrap_or(1.0)).unwrap_or(1.0);
            let freq = 1.0 / (rope_theta.powf(2.0 * i as f64 / head_dim as f64) * scale);
            let angle = (pos as f64 * freq) as f32;
            let cos_v = angle.cos();
            let sin_v = angle.sin();

            let base = pos * head_dim;
            let q0 = q_out[base + i];
            let q1 = q_out[base + i + half_dim];
            q_out[base + i] = q0 * cos_v - q1 * sin_v;
            q_out[base + i + half_dim] = q0 * sin_v + q1 * cos_v;

            let k0 = k_out[base + i];
            let k1 = k_out[base + i + half_dim];
            k_out[base + i] = k0 * cos_v - k1 * sin_v;
            k_out[base + i + half_dim] = k0 * sin_v + k1 * cos_v;
        }
    }
    (q_out, k_out)
}

/// Compute GQA attention scores with optional sliding window masking.
///
/// This is a lightweight pure-float implementation used for testing the
/// GQA key/value grouping logic without depending on the `Tensor` abstraction.
///
/// # Arguments
/// * `q`            - Query  [num_heads  * seq_len * head_dim]
/// * `k`            - Key    [num_kv_heads * seq_len * head_dim]
/// * `v`            - Value  [num_kv_heads * seq_len * head_dim]
/// * `seq_len`      - Sequence length
/// * `num_heads`    - Number of query heads
/// * `num_kv_heads` - Number of key/value heads
/// * `head_dim`     - Per-head dimension
/// * `window_size`  - Optional sliding window size (`None` = full attention)
///
/// Returns a flat `[num_heads * seq_len * head_dim]` context vector.
pub fn gqa_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    window_size: Option<usize>,
) -> Vec<f32> {
    let kv_group = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let total_out = num_heads * seq_len * head_dim;
    let mut output = vec![0.0f32; total_out];

    for h in 0..num_heads {
        let kv_h = h / kv_group;
        for qi in 0..seq_len {
            // Compute attention scores for this query position
            let mut scores = vec![0.0f32; seq_len];
            for kj in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let q_idx = h * seq_len * head_dim + qi * head_dim + d;
                    let k_idx = kv_h * seq_len * head_dim + kj * head_dim + d;
                    dot += q.get(q_idx).copied().unwrap_or(0.0)
                        * k.get(k_idx).copied().unwrap_or(0.0);
                }
                scores[kj] = dot * scale;
            }

            // Apply causal mask: future positions get -inf
            for kj in (qi + 1)..seq_len {
                scores[kj] = f32::NEG_INFINITY;
            }

            // Apply sliding window mask if requested
            if let Some(ws) = window_size {
                for kj in 0..seq_len {
                    let diff = if qi >= kj { qi - kj } else { kj - qi };
                    if diff > ws {
                        scores[kj] = f32::NEG_INFINITY;
                    }
                }
            }

            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            let mut exp_scores: Vec<f32> = scores
                .iter()
                .map(|&s| {
                    let e = (s - max_s).exp();
                    exp_sum += e;
                    e
                })
                .collect();
            if exp_sum > 0.0 {
                for es in &mut exp_scores {
                    *es /= exp_sum;
                }
            }

            // Weighted sum over values
            for d in 0..head_dim {
                let mut weighted = 0.0f32;
                for kj in 0..seq_len {
                    let v_idx = kv_h * seq_len * head_dim + kj * head_dim + d;
                    weighted += exp_scores[kj] * v.get(v_idx).copied().unwrap_or(0.0);
                }
                output[h * seq_len * head_dim + qi * head_dim + d] = weighted;
            }
        }
    }
    output
}

/// Greedy token generation using a logit table.
///
/// This implementation does NOT load weights; it is intended for testing
/// the generation loop logic. The model's `forward` is stubbed by a
/// simple embedding lookup that returns the input token ids as logits.
///
/// # Arguments
/// * `input_ids`      - Initial token sequence
/// * `max_new_tokens` - Number of tokens to generate
/// * `vocab_size`     - Size of the vocabulary
/// * `logit_fn`       - Closure: `(token_id: u32) -> Vec<f32>` returning vocabulary logits
///
/// Returns the generated continuation (without the prompt).
pub fn greedy_generate<F>(
    input_ids: &[u32],
    max_new_tokens: usize,
    _vocab_size: usize,
    logit_fn: F,
) -> Result<Vec<u32>, Phi3Error>
where
    F: Fn(u32) -> Result<Vec<f32>, Phi3Error>,
{
    if input_ids.is_empty() {
        return Err(Phi3Error::EmptyInput);
    }
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut last_token = *input_ids.last().ok_or(Phi3Error::EmptyInput)?;

    for _ in 0..max_new_tokens {
        let logits = logit_fn(last_token)?;
        if logits.is_empty() {
            return Err(Phi3Error::GenerationError("logit function returned empty logits".to_string()));
        }
        // Greedy argmax
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| Phi3Error::GenerationError("argmax failed".to_string()))?;
        generated.push(next_token);
        last_token = next_token;
    }
    Ok(generated)
}

/// Compute RMSNorm for a flat slice.
///
/// `output[i] = input[i] / sqrt(mean(input^2) + eps)`
///
/// The weight is assumed to be all-ones (as in a freshly initialised layer).
pub fn rms_norm(input: &[f32], eps: f32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let mean_sq = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
    let rms = (mean_sq + eps).sqrt();
    input.iter().map(|x| x / rms).collect()
}

/// Apply SiLU activation: `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply SwiGLU to gate and up projections.
///
/// `output[i] = silu(gate[i]) * up[i]`
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up.iter()).map(|(&g, &u)| silu(g) * u).collect()
}

#[cfg(test)]
mod task_tests {
    use super::*;

    // -----------------------------------------------------------------------
    // test_phi3_config_default
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_config_default() {
        use crate::phi3::config::Phi3Config;
        let cfg = Phi3Config::default();
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.vocab_size, 32064);
        assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // test_phi3_config_mini
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_config_mini() {
        use crate::phi3::config::Phi3Config;
        let cfg = Phi3Config::phi3_mini_4k_instruct();
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.intermediate_size, 8192);
        assert_eq!(cfg.max_position_embeddings, 4096);
        // mini uses MHA (no GQA)
        assert!(cfg.num_key_value_heads.is_none());
    }

    // -----------------------------------------------------------------------
    // test_phi3_rope_no_scaling
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_rope_no_scaling() {
        let seq_len = 4;
        let head_dim = 8;
        let q: Vec<f32> = (0..(seq_len * head_dim)).map(|i| i as f32 * 0.1).collect();
        let k = q.clone();
        let (q_out, k_out) = apply_rope(&q, &k, seq_len, head_dim, 10000.0, None);
        // Result should have same length
        assert_eq!(q_out.len(), q.len());
        assert_eq!(k_out.len(), k.len());
        // Position 0 angle = 0, so cos=1, sin=0 → unchanged
        for d in 0..head_dim {
            assert!(
                (q_out[d] - q[d]).abs() < 1e-5,
                "pos=0 should be unchanged, d={}", d
            );
        }
    }

    // -----------------------------------------------------------------------
    // test_phi3_rope_longrope_short_seq
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_rope_longrope_short_seq() {
        let seq_len = 2;
        let head_dim = 4;
        let q = vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let k = q.clone();
        // Simulate short-seq scale factors (head_dim/2 = 2 factors)
        let short_factors = vec![1.0_f64, 1.0];
        let (q_out, _) = apply_rope(&q, &k, seq_len, head_dim, 10000.0, Some(&short_factors));
        assert_eq!(q_out.len(), q.len());
    }

    // -----------------------------------------------------------------------
    // test_phi3_rope_longrope_long_seq
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_rope_longrope_long_seq() {
        let seq_len = 3;
        let head_dim = 4;
        let q: Vec<f32> = (0..(seq_len * head_dim)).map(|i| (i + 1) as f32 * 0.5).collect();
        let k = q.clone();
        // Simulate long-seq scale factors
        let long_factors = vec![2.0_f64, 4.0];
        let (q_out, k_out) = apply_rope(&q, &k, seq_len, head_dim, 10000.0, Some(&long_factors));
        assert_eq!(q_out.len(), q.len());
        assert_eq!(k_out.len(), k.len());
    }

    // -----------------------------------------------------------------------
    // test_phi3_rms_norm
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_rms_norm() {
        let input = vec![3.0f32, 4.0];
        // rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        let out = rms_norm(&input, 1e-5);
        let rms = (12.5f32 + 1e-5).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-5);
        assert!((out[1] - 4.0 / rms).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // test_phi3_sliding_window_mask_within_window
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_sliding_window_mask_within_window() {
        let seq_len = 4;
        let window_size = 2;
        let mut scores = vec![1.0f32; seq_len * seq_len];
        apply_sliding_window_mask(&mut scores, seq_len, window_size);
        // Position (0,0): diff=0 ≤ 2 → should remain 1.0
        assert!((scores[0 * seq_len + 0] - 1.0).abs() < 1e-6);
        // Position (0,2): diff=2 ≤ 2 → should remain 1.0
        assert!((scores[0 * seq_len + 2] - 1.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // test_phi3_sliding_window_mask_outside_window
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_sliding_window_mask_outside_window() {
        let seq_len = 5;
        let window_size = 1;
        let mut scores = vec![0.5f32; seq_len * seq_len];
        apply_sliding_window_mask(&mut scores, seq_len, window_size);
        // Position (0,3): diff=3 > 1 → should be -inf
        assert!(scores[0 * seq_len + 3].is_infinite() && scores[0 * seq_len + 3] < 0.0);
        // Position (4,4): diff=0 ≤ 1 → should remain 0.5
        assert!((scores[4 * seq_len + 4] - 0.5).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // test_phi3_gqa_kv_grouping
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_gqa_kv_grouping() {
        let seq_len = 2;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;

        let q = vec![0.1f32; num_heads * seq_len * head_dim];
        let k = vec![0.1f32; num_kv_heads * seq_len * head_dim];
        let v = vec![1.0f32; num_kv_heads * seq_len * head_dim];

        let out = gqa_attention(&q, &k, &v, seq_len, num_heads, num_kv_heads, head_dim, None);
        assert_eq!(out.len(), num_heads * seq_len * head_dim);
    }

    // -----------------------------------------------------------------------
    // test_phi3_mlp_swiglu
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_mlp_swiglu() {
        let gate = vec![1.0f32, 0.0, -1.0];
        let up = vec![2.0f32, 2.0, 2.0];
        let out = swiglu(&gate, &up);
        assert_eq!(out.len(), 3);
        // silu(0.0) = 0.0 * sigmoid(0.0) = 0.0 * 0.5 = 0.0; out[1] = 0.0 * 2.0 = 0.0
        assert!(out[1].abs() < 1e-5, "silu(0) * 2 should be 0, got {}", out[1]);
        // silu(1.0) > 0
        assert!(out[0] > 0.0);
        // silu(-1.0) < 0
        assert!(out[2] < 0.0);
    }

    // -----------------------------------------------------------------------
    // test_phi3_model_forward
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_model_forward() {
        use crate::phi3::config::Phi3Config;
        use crate::phi3::model::Phi3Model;
        use trustformers_core::traits::Model;
        use trustformers_core::tensor::Tensor;

        let mut cfg = Phi3Config::default();
        cfg.hidden_size = 32;
        cfg.intermediate_size = 64;
        cfg.num_hidden_layers = 1;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = None;
        cfg.vocab_size = 100;

        let model = Phi3Model::new(cfg);
        assert!(model.is_ok(), "Phi3Model::new failed: {:?}", model.err());
        let model = model.expect("model creation");

        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).expect("tensor");
        let result = model.forward(input);
        assert!(result.is_ok(), "forward failed: {:?}", result.err());
    }

    // -----------------------------------------------------------------------
    // test_phi3_causal_lm_forward
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_causal_lm_forward() {
        use crate::phi3::config::Phi3Config;
        use crate::phi3::model::Phi3ForCausalLM;
        use trustformers_core::traits::Model;
        use trustformers_core::tensor::Tensor;

        let mut cfg = Phi3Config::default();
        cfg.hidden_size = 32;
        cfg.intermediate_size = 64;
        cfg.num_hidden_layers = 1;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = None;
        cfg.vocab_size = 100;

        let model = Phi3ForCausalLM::new(cfg);
        assert!(model.is_ok(), "Phi3ForCausalLM::new failed");
        let model = model.expect("causal lm creation");

        let input = Tensor::from_vec(vec![0.0f32, 1.0, 2.0], &[3]).expect("tensor");
        let result = model.forward(input);
        assert!(result.is_ok(), "forward failed: {:?}", result.err());
    }

    // -----------------------------------------------------------------------
    // test_phi3_generate
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_generate() {
        // Use the greedy_generate helper with a simple stub logit function
        let input_ids = vec![1u32, 2, 3];
        let vocab_size = 10usize;
        let result = greedy_generate(&input_ids, 3, vocab_size, |token| {
            // Deterministic: always predict token+1 (mod vocab_size)
            let mut logits = vec![0.0f32; vocab_size];
            let next = ((token as usize) + 1) % vocab_size;
            logits[next] = 1.0;
            Ok(logits)
        });
        assert!(result.is_ok(), "generate failed: {:?}", result.err());
        let generated = result.expect("generated");
        assert_eq!(generated.len(), 3);
        // First generated token = token(3)+1 = 4
        assert_eq!(generated[0], 4);
        assert_eq!(generated[1], 5);
        assert_eq!(generated[2], 6);
    }

    // -----------------------------------------------------------------------
    // test_phi3_chat_format_no_system
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_chat_format_no_system() {
        let prompt = format_chat_prompt(None, "Hello!");
        assert!(prompt.contains("<|user|>"));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.contains("<|end|>"));
        assert!(prompt.contains("<|assistant|>"));
        assert!(!prompt.contains("<|system|>"));
    }

    // -----------------------------------------------------------------------
    // test_phi3_chat_format_with_system
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_chat_format_with_system() {
        let prompt = format_chat_prompt(Some("You are helpful."), "What is 2+2?");
        assert!(prompt.contains("<|system|>"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("<|user|>"));
        assert!(prompt.contains("What is 2+2?"));
        assert!(prompt.contains("<|assistant|>"));
    }

    // -----------------------------------------------------------------------
    // test_phi3_sliding_window_coverage
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_sliding_window_coverage() {
        // Full attention: window = seq_len - 1 → coverage = 1.0
        let cov = sliding_window_coverage(5, 4);
        assert!((cov - 1.0).abs() < 1e-5, "full coverage should be 1.0, got {}", cov);

        // Zero window: each token only attends to itself → coverage = 1/seq_len
        let cov0 = sliding_window_coverage(5, 0);
        assert!((cov0 - 0.2).abs() < 1e-5, "zero window coverage should be 0.2, got {}", cov0);

        // Empty sequence
        let cov_empty = sliding_window_coverage(0, 10);
        assert_eq!(cov_empty, 0.0);
    }

    // -----------------------------------------------------------------------
    // test_phi3_error_display
    // -----------------------------------------------------------------------
    #[test]
    fn test_phi3_error_display() {
        let err = Phi3Error::InvalidConfig("bad value".to_string());
        let s = format!("{}", err);
        assert!(s.contains("invalid config"));
        assert!(s.contains("bad value"));

        let err2 = Phi3Error::SequenceTooLong { max: 512, got: 1024 };
        let s2 = format!("{}", err2);
        assert!(s2.contains("512"));
        assert!(s2.contains("1024"));

        let err3 = Phi3Error::EmptyInput;
        let s3 = format!("{}", err3);
        assert!(s3.contains("empty"));
    }
}
