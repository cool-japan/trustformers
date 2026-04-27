//! Fused operations — combine multiple ops for cache efficiency.
//!
//! Fused operations reduce memory bandwidth by combining multiple sequential
//! tensor operations into a single computation kernel. This avoids intermediate
//! allocations and reduces the number of memory passes.

use std::fmt;

// ── Error type ────────────────────────────────────────────────────────────────

/// Error type returned by fused operations.
#[derive(Debug, Clone)]
pub enum FusedOpError {
    /// Input/weight dimensions do not match expectations.
    DimensionMismatch {
        /// Name of the operation that failed.
        op: String,
        /// Expected size.
        expected: usize,
        /// Actual size received.
        got: usize,
    },
    /// An input slice was empty when it should contain data.
    EmptyInput(String),
    /// Configuration value is invalid (e.g. zero hidden_size).
    InvalidConfig(String),
}

impl fmt::Display for FusedOpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusedOpError::DimensionMismatch { op, expected, got } => write!(
                f,
                "dimension mismatch in {op}: expected {expected}, got {got}"
            ),
            FusedOpError::EmptyInput(msg) => write!(f, "empty input: {msg}"),
            FusedOpError::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for FusedOpError {}

// ── Result type ───────────────────────────────────────────────────────────────

/// Result of a fused operation.
#[derive(Debug, Clone)]
pub struct FusedOpResult {
    /// The computed output values.
    pub output: Vec<f32>,
    /// Names of the individual operations that were fused.
    pub ops_fused: Vec<String>,
    /// Rough estimate of floating-point operations performed.
    pub estimated_flops: u64,
}

// ── Helper: LayerNorm (in-place on a slice) ────────────────────────────────

/// Compute LayerNorm for a single token vector `x` of length `hidden_size`.
/// Returns the normalized vector; `x` itself is not modified.
///
/// `norm = (x - mean) / sqrt(var + eps) * weight + bias`
fn layer_norm_slice(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .zip(bias.iter())
        .map(|((&xi, &wi), &bi)| (xi - mean) * inv_std * wi + bi)
        .collect()
}

/// Compute RMSNorm for a single token vector `x` of length `hidden_size`.
///
/// `norm = x / sqrt(mean(x^2) + eps) * weight`
fn rms_norm_slice(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / n + eps).sqrt();
    let inv_rms = 1.0 / rms;
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| xi * inv_rms * wi)
        .collect()
}

/// Dense linear projection: `output[i] = sum_j(input[j] * weight[i * in + j]) + bias[i]`.
fn linear_projection(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_features];
    for i in 0..out_features {
        let row_start = i * in_features;
        let mut acc = 0.0f32;
        for j in 0..in_features {
            acc += input[j] * weight[row_start + j];
        }
        if let Some(b) = bias {
            acc += b[i];
        }
        out[i] = acc;
    }
    out
}

/// SiLU activation: `x * sigmoid(x) = x / (1 + exp(-x))`.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Tanh-based GELU approximation used by most transformers.
///
/// `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`
#[inline]
fn gelu(x: f32) -> f32 {
    // sqrt(2/π) ≈ 0.797_884_5
    const SQRT_2_OVER_PI: f32 = 0.797_884_5;
    const COEFF: f32 = 0.044715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + COEFF * x * x * x)).tanh())
}

/// Numerically stable softmax over a mutable slice.
fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for vi in v.iter_mut() {
        *vi = (*vi - max).exp();
        sum += *vi;
    }
    if sum > 0.0 {
        for vi in v.iter_mut() {
            *vi /= sum;
        }
    }
}

// ── Public fused ops ──────────────────────────────────────────────────────────

/// Fused LayerNorm + Linear.
///
/// Computes `Linear(LayerNorm(x))` in a single memory pass over `x`, avoiding
/// a separate allocation for the normalized activations.
///
/// # Parameters
/// - `x`: input, shape `[hidden_size]` (single token vector)
/// - `ln_weight` / `ln_bias`: LayerNorm affine parameters, each `hidden_size`
/// - `linear_weight`: row-major `[out_features, hidden_size]`
/// - `linear_bias`: optional `[out_features]`
/// - `hidden_size`, `out_features`: dimensions
/// - `eps`: LayerNorm epsilon
pub fn fused_layer_norm_linear(
    x: &[f32],
    ln_weight: &[f32],
    ln_bias: &[f32],
    linear_weight: &[f32],
    linear_bias: Option<&[f32]>,
    hidden_size: usize,
    out_features: usize,
    eps: f32,
) -> Result<FusedOpResult, FusedOpError> {
    if x.is_empty() {
        return Err(FusedOpError::EmptyInput("x".to_string()));
    }
    if hidden_size == 0 {
        return Err(FusedOpError::InvalidConfig(
            "hidden_size must be > 0".to_string(),
        ));
    }
    if out_features == 0 {
        return Err(FusedOpError::InvalidConfig(
            "out_features must be > 0".to_string(),
        ));
    }
    if x.len() != hidden_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_layer_norm_linear/x".to_string(),
            expected: hidden_size,
            got: x.len(),
        });
    }
    if ln_weight.len() != hidden_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_layer_norm_linear/ln_weight".to_string(),
            expected: hidden_size,
            got: ln_weight.len(),
        });
    }
    if ln_bias.len() != hidden_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_layer_norm_linear/ln_bias".to_string(),
            expected: hidden_size,
            got: ln_bias.len(),
        });
    }
    if linear_weight.len() != out_features * hidden_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_layer_norm_linear/linear_weight".to_string(),
            expected: out_features * hidden_size,
            got: linear_weight.len(),
        });
    }
    if let Some(b) = linear_bias {
        if b.len() != out_features {
            return Err(FusedOpError::DimensionMismatch {
                op: "fused_layer_norm_linear/linear_bias".to_string(),
                expected: out_features,
                got: b.len(),
            });
        }
    }

    // Step 1: LayerNorm
    let normed = layer_norm_slice(x, ln_weight, ln_bias, eps);

    // Step 2: Linear projection using normalized values
    let output = linear_projection(&normed, linear_weight, linear_bias, hidden_size, out_features);

    // FLOPs: LayerNorm ≈ 5*H, Linear ≈ 2*H*O
    let estimated_flops = 5 * hidden_size as u64 + 2 * hidden_size as u64 * out_features as u64;

    Ok(FusedOpResult {
        output,
        ops_fused: vec!["LayerNorm".to_string(), "Linear".to_string()],
        estimated_flops,
    })
}

/// Fused RMSNorm + Linear.
///
/// Computes `Linear(RMSNorm(x))` — the LLaMA/Mistral style pre-norm + projection.
///
/// RMSNorm: `norm = x / sqrt(mean(x²) + eps) * weight` (no mean subtraction, no bias).
pub fn fused_rms_norm_linear(
    x: &[f32],
    rms_weight: &[f32],
    linear_weight: &[f32],
    linear_bias: Option<&[f32]>,
    hidden_size: usize,
    out_features: usize,
    eps: f32,
) -> Result<FusedOpResult, FusedOpError> {
    if x.is_empty() {
        return Err(FusedOpError::EmptyInput("x".to_string()));
    }
    if hidden_size == 0 {
        return Err(FusedOpError::InvalidConfig(
            "hidden_size must be > 0".to_string(),
        ));
    }
    if out_features == 0 {
        return Err(FusedOpError::InvalidConfig(
            "out_features must be > 0".to_string(),
        ));
    }
    if x.len() != hidden_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_rms_norm_linear/x".to_string(),
            expected: hidden_size,
            got: x.len(),
        });
    }
    if rms_weight.len() != hidden_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_rms_norm_linear/rms_weight".to_string(),
            expected: hidden_size,
            got: rms_weight.len(),
        });
    }
    if linear_weight.len() != out_features * hidden_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_rms_norm_linear/linear_weight".to_string(),
            expected: out_features * hidden_size,
            got: linear_weight.len(),
        });
    }
    if let Some(b) = linear_bias {
        if b.len() != out_features {
            return Err(FusedOpError::DimensionMismatch {
                op: "fused_rms_norm_linear/linear_bias".to_string(),
                expected: out_features,
                got: b.len(),
            });
        }
    }

    let normed = rms_norm_slice(x, rms_weight, eps);
    let output = linear_projection(&normed, linear_weight, linear_bias, hidden_size, out_features);

    // FLOPs: RMSNorm ≈ 4*H, Linear ≈ 2*H*O
    let estimated_flops = 4 * hidden_size as u64 + 2 * hidden_size as u64 * out_features as u64;

    Ok(FusedOpResult {
        output,
        ops_fused: vec!["RMSNorm".to_string(), "Linear".to_string()],
        estimated_flops,
    })
}

/// Fused Attention Scores: QKᵀ / √d + causal mask + softmax.
///
/// Computes scaled dot-product attention weights in a single pass.
///
/// # Layout
/// - `q`: `[seq_len, num_heads, head_dim]` (row-major, head-minor)
/// - `k`: `[seq_len, num_kv_heads, head_dim]`
///
/// # GQA
/// `kv_group = num_heads / num_kv_heads`; head `h` uses KV head `h / kv_group`.
///
/// # Output
/// Attention weights, shape `[num_heads, seq_len, seq_len]`, as a flat `Vec<f32>`.
pub fn fused_attention_scores(
    q: &[f32],
    k: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    causal_mask: bool,
) -> Result<FusedOpResult, FusedOpError> {
    if q.is_empty() {
        return Err(FusedOpError::EmptyInput("q".to_string()));
    }
    if k.is_empty() {
        return Err(FusedOpError::EmptyInput("k".to_string()));
    }
    if seq_len == 0 {
        return Err(FusedOpError::InvalidConfig(
            "seq_len must be > 0".to_string(),
        ));
    }
    if num_heads == 0 {
        return Err(FusedOpError::InvalidConfig(
            "num_heads must be > 0".to_string(),
        ));
    }
    if num_kv_heads == 0 {
        return Err(FusedOpError::InvalidConfig(
            "num_kv_heads must be > 0".to_string(),
        ));
    }
    if head_dim == 0 {
        return Err(FusedOpError::InvalidConfig(
            "head_dim must be > 0".to_string(),
        ));
    }
    if num_heads % num_kv_heads != 0 {
        return Err(FusedOpError::InvalidConfig(format!(
            "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )));
    }

    let expected_q = seq_len * num_heads * head_dim;
    if q.len() != expected_q {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_attention_scores/q".to_string(),
            expected: expected_q,
            got: q.len(),
        });
    }
    let expected_k = seq_len * num_kv_heads * head_dim;
    if k.len() != expected_k {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_attention_scores/k".to_string(),
            expected: expected_k,
            got: k.len(),
        });
    }

    let kv_group = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    // output: [num_heads, seq_len, seq_len]
    let total_out = num_heads * seq_len * seq_len;
    let mut output = vec![0.0f32; total_out];

    for h in 0..num_heads {
        let kv_h = h / kv_group;
        for qi in 0..seq_len {
            // q[qi, h, :] base offset
            let q_base = qi * num_heads * head_dim + h * head_dim;
            for ki in 0..seq_len {
                // k[ki, kv_h, :] base offset
                let k_base = ki * num_kv_heads * head_dim + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_base + d] * k[k_base + d];
                }
                dot *= scale;
                // causal mask: future positions → -∞
                if causal_mask && ki > qi {
                    dot = f32::NEG_INFINITY;
                }
                output[h * seq_len * seq_len + qi * seq_len + ki] = dot;
            }
            // softmax over the ki dimension for this (h, qi) row
            let row_start = h * seq_len * seq_len + qi * seq_len;
            let row_end = row_start + seq_len;
            softmax_inplace(&mut output[row_start..row_end]);
        }
    }

    // FLOPs: num_heads * seq_len * seq_len * (2*head_dim) for QKT
    let estimated_flops =
        num_heads as u64 * seq_len as u64 * seq_len as u64 * 2 * head_dim as u64;

    Ok(FusedOpResult {
        output,
        ops_fused: vec![
            "QK_matmul".to_string(),
            "scale".to_string(),
            "causal_mask".to_string(),
            "softmax".to_string(),
        ],
        estimated_flops,
    })
}

/// Fused SwiGLU feed-forward block.
///
/// Implements the standard LLaMA/Mistral FFN:
///
/// ```text
/// gate  = gate_weight @ x
/// up    = up_weight @ x
/// activated[i] = gate[i] * silu(up[i])
/// output = down_weight @ activated
/// ```
///
/// Both projections are computed in the same loop, avoiding materialising `gate`
/// and `up` in separate memory buffers before combining them.
pub fn fused_swiglu(
    x: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<FusedOpResult, FusedOpError> {
    if x.is_empty() {
        return Err(FusedOpError::EmptyInput("x".to_string()));
    }
    if hidden_size == 0 {
        return Err(FusedOpError::InvalidConfig(
            "hidden_size must be > 0".to_string(),
        ));
    }
    if intermediate_size == 0 {
        return Err(FusedOpError::InvalidConfig(
            "intermediate_size must be > 0".to_string(),
        ));
    }
    if x.len() != hidden_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_swiglu/x".to_string(),
            expected: hidden_size,
            got: x.len(),
        });
    }
    if gate_weight.len() != intermediate_size * hidden_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_swiglu/gate_weight".to_string(),
            expected: intermediate_size * hidden_size,
            got: gate_weight.len(),
        });
    }
    if up_weight.len() != intermediate_size * hidden_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_swiglu/up_weight".to_string(),
            expected: intermediate_size * hidden_size,
            got: up_weight.len(),
        });
    }
    if down_weight.len() != hidden_size * intermediate_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_swiglu/down_weight".to_string(),
            expected: hidden_size * intermediate_size,
            got: down_weight.len(),
        });
    }

    // Compute gate and up projections, apply SwiGLU in one fused loop
    let mut activated = vec![0.0f32; intermediate_size];
    for i in 0..intermediate_size {
        let base = i * hidden_size;
        let mut gate_val = 0.0f32;
        let mut up_val = 0.0f32;
        for j in 0..hidden_size {
            gate_val += x[j] * gate_weight[base + j];
            up_val += x[j] * up_weight[base + j];
        }
        // SwiGLU: gate * silu(up)
        activated[i] = gate_val * silu(up_val);
    }

    // Down projection
    let output = linear_projection(&activated, down_weight, None, intermediate_size, hidden_size);

    // FLOPs: gate+up = 2 * 2*H*I, silu = I, down = 2*I*H
    let estimated_flops = 6 * hidden_size as u64 * intermediate_size as u64 + intermediate_size as u64;

    Ok(FusedOpResult {
        output,
        ops_fused: vec![
            "gate_proj".to_string(),
            "up_proj".to_string(),
            "silu".to_string(),
            "mul".to_string(),
            "down_proj".to_string(),
        ],
        estimated_flops,
    })
}

/// Fused GeGLU feed-forward block (Gemma-2 / PaLM style).
///
/// ```text
/// gate  = gate_weight @ x
/// up    = up_weight @ x
/// activated[i] = gelu(gate[i]) * up[i]
/// output = down_weight @ activated
/// ```
///
/// Uses the tanh approximation of GELU.
pub fn fused_geglu(
    x: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<FusedOpResult, FusedOpError> {
    if x.is_empty() {
        return Err(FusedOpError::EmptyInput("x".to_string()));
    }
    if hidden_size == 0 {
        return Err(FusedOpError::InvalidConfig(
            "hidden_size must be > 0".to_string(),
        ));
    }
    if intermediate_size == 0 {
        return Err(FusedOpError::InvalidConfig(
            "intermediate_size must be > 0".to_string(),
        ));
    }
    if x.len() != hidden_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_geglu/x".to_string(),
            expected: hidden_size,
            got: x.len(),
        });
    }
    if gate_weight.len() != intermediate_size * hidden_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_geglu/gate_weight".to_string(),
            expected: intermediate_size * hidden_size,
            got: gate_weight.len(),
        });
    }
    if up_weight.len() != intermediate_size * hidden_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_geglu/up_weight".to_string(),
            expected: intermediate_size * hidden_size,
            got: up_weight.len(),
        });
    }
    if down_weight.len() != hidden_size * intermediate_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_geglu/down_weight".to_string(),
            expected: hidden_size * intermediate_size,
            got: down_weight.len(),
        });
    }

    let mut activated = vec![0.0f32; intermediate_size];
    for i in 0..intermediate_size {
        let base = i * hidden_size;
        let mut gate_val = 0.0f32;
        let mut up_val = 0.0f32;
        for j in 0..hidden_size {
            gate_val += x[j] * gate_weight[base + j];
            up_val += x[j] * up_weight[base + j];
        }
        // GeGLU: gelu(gate) * up
        activated[i] = gelu(gate_val) * up_val;
    }

    let output = linear_projection(&activated, down_weight, None, intermediate_size, hidden_size);

    let estimated_flops = 6 * hidden_size as u64 * intermediate_size as u64 + intermediate_size as u64;

    Ok(FusedOpResult {
        output,
        ops_fused: vec![
            "gate_proj".to_string(),
            "up_proj".to_string(),
            "gelu".to_string(),
            "mul".to_string(),
            "down_proj".to_string(),
        ],
        estimated_flops,
    })
}

/// Fused residual add + normalization.
///
/// Computes:
/// 1. `x = residual + hidden_states`
/// 2. Either RMSNorm or LayerNorm of `x`, controlled by `use_rms_norm`
///
/// When `use_rms_norm = true`, `norm_bias` is ignored.
pub fn fused_residual_add_norm(
    residual: &[f32],
    hidden_states: &[f32],
    norm_weight: &[f32],
    norm_bias: Option<&[f32]>,
    hidden_size: usize,
    eps: f32,
    use_rms_norm: bool,
) -> Result<FusedOpResult, FusedOpError> {
    if residual.is_empty() {
        return Err(FusedOpError::EmptyInput("residual".to_string()));
    }
    if hidden_states.is_empty() {
        return Err(FusedOpError::EmptyInput("hidden_states".to_string()));
    }
    if hidden_size == 0 {
        return Err(FusedOpError::InvalidConfig(
            "hidden_size must be > 0".to_string(),
        ));
    }
    if residual.len() != hidden_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_residual_add_norm/residual".to_string(),
            expected: hidden_size,
            got: residual.len(),
        });
    }
    if hidden_states.len() != hidden_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_residual_add_norm/hidden_states".to_string(),
            expected: hidden_size,
            got: hidden_states.len(),
        });
    }
    if norm_weight.len() != hidden_size {
        return Err(FusedOpError::DimensionMismatch {
            op: "fused_residual_add_norm/norm_weight".to_string(),
            expected: hidden_size,
            got: norm_weight.len(),
        });
    }
    if let Some(b) = norm_bias {
        if b.len() != hidden_size {
            return Err(FusedOpError::DimensionMismatch {
                op: "fused_residual_add_norm/norm_bias".to_string(),
                expected: hidden_size,
                got: b.len(),
            });
        }
    }

    // Step 1: residual addition
    let added: Vec<f32> = residual
        .iter()
        .zip(hidden_states.iter())
        .map(|(r, h)| r + h)
        .collect();

    // Step 2: normalization
    let output = if use_rms_norm {
        rms_norm_slice(&added, norm_weight, eps)
    } else {
        let bias = norm_bias.unwrap_or(&[]);
        // If no bias supplied, use zero bias
        let zero_bias: Vec<f32>;
        let effective_bias = if bias.is_empty() {
            zero_bias = vec![0.0f32; hidden_size];
            &zero_bias[..]
        } else {
            bias
        };
        layer_norm_slice(&added, norm_weight, effective_bias, eps)
    };

    let norm_name = if use_rms_norm { "RMSNorm" } else { "LayerNorm" };
    let estimated_flops = 2 * hidden_size as u64 + 5 * hidden_size as u64;

    Ok(FusedOpResult {
        output,
        ops_fused: vec!["residual_add".to_string(), norm_name.to_string()],
        estimated_flops,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    /// Approximate equality for f32 vectors.
    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(
            a.len(),
            b.len(),
            "{label}: length mismatch {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (ai - bi).abs() <= tol,
                "{label}[{i}]: |{ai} - {bi}| = {} > {tol}",
                (ai - bi).abs()
            );
        }
    }

    // ── layer_norm + linear ──────────────────────────────────────────────────

    #[test]
    fn test_fused_layer_norm_linear_basic() {
        let hidden = 4;
        let out = 2;
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let ln_w = vec![1.0f32; hidden];
        let ln_b = vec![0.0f32; hidden];
        let lw = vec![1.0f32; out * hidden];
        let result = fused_layer_norm_linear(&x, &ln_w, &ln_b, &lw, None, hidden, out, EPS)
            .expect("should succeed");
        assert_eq!(result.output.len(), out);
        assert_eq!(result.ops_fused, vec!["LayerNorm", "Linear"]);
        assert!(result.estimated_flops > 0);
    }

    #[test]
    fn test_fused_layer_norm_linear_vs_sequential() {
        let hidden = 8;
        let out = 4;
        let x: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.5 - 1.0).collect();
        let ln_w: Vec<f32> = (0..hidden).map(|i| 1.0 + i as f32 * 0.1).collect();
        let ln_b: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.05).collect();
        let lw: Vec<f32> = (0..out * hidden).map(|i| (i as f32) * 0.01 - 0.2).collect();
        let lb: Vec<f32> = (0..out).map(|i| i as f32 * 0.1).collect();

        // Fused
        let fused_result = fused_layer_norm_linear(
            &x,
            &ln_w,
            &ln_b,
            &lw,
            Some(&lb),
            hidden,
            out,
            EPS,
        )
        .expect("fused ok");

        // Sequential (manual)
        let normed = layer_norm_slice(&x, &ln_w, &ln_b, EPS);
        let seq_out = linear_projection(&normed, &lw, Some(&lb), hidden, out);

        assert_approx_eq(&fused_result.output, &seq_out, 1e-5, "layer_norm_linear_vs_seq");
    }

    // ── rms_norm + linear ────────────────────────────────────────────────────

    #[test]
    fn test_fused_rms_norm_linear_basic() {
        let hidden = 4;
        let out = 3;
        let x = vec![0.5f32, -0.5, 1.0, -1.0];
        let rw = vec![1.0f32; hidden];
        let lw = vec![0.5f32; out * hidden];
        let result = fused_rms_norm_linear(&x, &rw, &lw, None, hidden, out, EPS)
            .expect("should succeed");
        assert_eq!(result.output.len(), out);
        assert_eq!(result.ops_fused, vec!["RMSNorm", "Linear"]);
        assert!(result.estimated_flops > 0);
    }

    #[test]
    fn test_fused_rms_norm_linear_vs_sequential() {
        let hidden = 6;
        let out = 3;
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.3).collect();
        let rw: Vec<f32> = (0..hidden).map(|i| 1.0 + i as f32 * 0.05).collect();
        let lw: Vec<f32> = (0..out * hidden).map(|i| (i as f32) * 0.02 - 0.1).collect();
        let lb: Vec<f32> = vec![0.1, -0.1, 0.2];

        let fused_result =
            fused_rms_norm_linear(&x, &rw, &lw, Some(&lb), hidden, out, EPS)
                .expect("fused ok");

        let normed = rms_norm_slice(&x, &rw, EPS);
        let seq_out = linear_projection(&normed, &lw, Some(&lb), hidden, out);

        assert_approx_eq(&fused_result.output, &seq_out, 1e-5, "rms_norm_linear_vs_seq");
    }

    // ── attention scores ─────────────────────────────────────────────────────

    #[test]
    fn test_fused_attention_scores_shape() {
        let seq = 3;
        let nh = 2;
        let nkv = 2;
        let hd = 4;
        let q = vec![0.1f32; seq * nh * hd];
        let k = vec![0.1f32; seq * nkv * hd];
        let result = fused_attention_scores(&q, &k, seq, nh, nkv, hd, false)
            .expect("ok");
        assert_eq!(result.output.len(), nh * seq * seq);
    }

    #[test]
    fn test_fused_attention_scores_causal_mask() {
        let seq = 3;
        let nh = 1;
        let nkv = 1;
        let hd = 2;
        let q = vec![1.0f32; seq * nh * hd];
        let k = vec![1.0f32; seq * nkv * hd];
        let result = fused_attention_scores(&q, &k, seq, nh, nkv, hd, true)
            .expect("ok");
        // For query position 0, k positions 1 and 2 should be masked → weight ~0
        let attn = &result.output;
        // head 0, query 0, key 1 and key 2 should be near-zero after softmax
        assert!(
            attn[0 * seq * seq + 0 * seq + 1] < 1e-10,
            "future key should be masked"
        );
        assert!(
            attn[0 * seq * seq + 0 * seq + 2] < 1e-10,
            "future key should be masked"
        );
    }

    #[test]
    fn test_fused_attention_scores_softmax_sums_to_one() {
        let seq = 4;
        let nh = 2;
        let nkv = 2;
        let hd = 8;
        // Use varied values to exercise softmax
        let q: Vec<f32> = (0..seq * nh * hd).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..seq * nkv * hd).map(|i| (i as f32) * 0.02 - 0.5).collect();
        let result = fused_attention_scores(&q, &k, seq, nh, nkv, hd, false)
            .expect("ok");
        // Each (head, query) row must sum to 1.0
        for h in 0..nh {
            for qi in 0..seq {
                let row_start = h * seq * seq + qi * seq;
                let sum: f32 = result.output[row_start..row_start + seq].iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "softmax row h={h} qi={qi} sums to {sum}"
                );
            }
        }
    }

    #[test]
    fn test_fused_attention_gqa() {
        // GQA: 4 query heads, 2 KV heads
        let seq = 2;
        let nh = 4;
        let nkv = 2;
        let hd = 4;
        let q = vec![0.5f32; seq * nh * hd];
        let k = vec![0.5f32; seq * nkv * hd];
        let result = fused_attention_scores(&q, &k, seq, nh, nkv, hd, false)
            .expect("GQA ok");
        assert_eq!(result.output.len(), nh * seq * seq);
        // Softmax rows must sum to 1
        for h in 0..nh {
            for qi in 0..seq {
                let rs = h * seq * seq + qi * seq;
                let sum: f32 = result.output[rs..rs + seq].iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "GQA softmax row h={h} qi={qi}");
            }
        }
    }

    // ── SwiGLU ───────────────────────────────────────────────────────────────

    #[test]
    fn test_fused_swiglu_basic() {
        let h = 4;
        let inter = 8;
        let x = vec![0.5f32; h];
        let gw = vec![0.1f32; inter * h];
        let uw = vec![0.1f32; inter * h];
        let dw = vec![0.1f32; h * inter];
        let result = fused_swiglu(&x, &gw, &uw, &dw, h, inter).expect("ok");
        assert_eq!(result.output.len(), h);
        assert!(result.ops_fused.contains(&"silu".to_string()));
        assert!(result.estimated_flops > 0);
    }

    #[test]
    fn test_fused_swiglu_vs_sequential() {
        let h = 4;
        let inter = 6;
        let x: Vec<f32> = (0..h).map(|i| i as f32 * 0.3 - 0.5).collect();
        let gw: Vec<f32> = (0..inter * h).map(|i| (i as f32) * 0.05 - 0.1).collect();
        let uw: Vec<f32> = (0..inter * h).map(|i| (i as f32) * 0.03 + 0.01).collect();
        let dw: Vec<f32> = (0..h * inter).map(|i| (i as f32) * 0.02 - 0.05).collect();

        let fused_result = fused_swiglu(&x, &gw, &uw, &dw, h, inter).expect("fused ok");

        // Sequential: gate proj, up proj, activate, down proj
        let gate_out = linear_projection(&x, &gw, None, h, inter);
        let up_out = linear_projection(&x, &uw, None, h, inter);
        let activated: Vec<f32> = gate_out
            .iter()
            .zip(up_out.iter())
            .map(|(&g, &u)| g * silu(u))
            .collect();
        let seq_out = linear_projection(&activated, &dw, None, inter, h);

        assert_approx_eq(&fused_result.output, &seq_out, 1e-5, "swiglu_vs_seq");
    }

    // ── GeGLU ────────────────────────────────────────────────────────────────

    #[test]
    fn test_fused_geglu_basic() {
        let h = 4;
        let inter = 8;
        let x = vec![0.5f32; h];
        let gw = vec![0.1f32; inter * h];
        let uw = vec![0.1f32; inter * h];
        let dw = vec![0.1f32; h * inter];
        let result = fused_geglu(&x, &gw, &uw, &dw, h, inter).expect("ok");
        assert_eq!(result.output.len(), h);
        assert!(result.ops_fused.contains(&"gelu".to_string()));
    }

    #[test]
    fn test_fused_geglu_vs_swiglu_differ() {
        // GeGLU and SwiGLU should produce different outputs (different activations)
        let h = 4;
        let inter = 6;
        let x: Vec<f32> = (0..h).map(|i| i as f32 * 0.3 + 0.1).collect();
        let gw: Vec<f32> = (0..inter * h).map(|i| (i as f32) * 0.05 + 0.01).collect();
        let uw: Vec<f32> = (0..inter * h).map(|i| (i as f32) * 0.03 + 0.01).collect();
        let dw: Vec<f32> = (0..h * inter).map(|i| (i as f32) * 0.02 + 0.01).collect();

        let swiglu_out = fused_swiglu(&x, &gw, &uw, &dw, h, inter)
            .expect("swiglu ok")
            .output;
        let geglu_out = fused_geglu(&x, &gw, &uw, &dw, h, inter)
            .expect("geglu ok")
            .output;

        // They should not be equal (different activations)
        let all_same = swiglu_out
            .iter()
            .zip(geglu_out.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(!all_same, "SwiGLU and GeGLU should produce different outputs");
    }

    // ── residual add + norm ──────────────────────────────────────────────────

    #[test]
    fn test_fused_residual_add_norm_rms() {
        let h = 4;
        let residual = vec![1.0f32, 0.0, -1.0, 0.5];
        let hidden = vec![0.5f32, 0.5, 0.5, 0.5];
        let nw = vec![1.0f32; h];
        let result = fused_residual_add_norm(&residual, &hidden, &nw, None, h, EPS, true)
            .expect("ok");
        assert_eq!(result.output.len(), h);
        assert!(result.ops_fused.contains(&"RMSNorm".to_string()));

        // Verify: x = residual + hidden = [1.5, 0.5, -0.5, 1.0]
        let x_sum = vec![1.5f32, 0.5, -0.5, 1.0];
        let expected = rms_norm_slice(&x_sum, &nw, EPS);
        assert_approx_eq(&result.output, &expected, 1e-5, "residual_rms");
    }

    #[test]
    fn test_fused_residual_add_norm_layer() {
        let h = 4;
        let residual = vec![1.0f32, 2.0, 3.0, 4.0];
        let hidden = vec![0.1f32, 0.1, 0.1, 0.1];
        let nw = vec![1.0f32; h];
        let nb = vec![0.0f32; h];
        let result =
            fused_residual_add_norm(&residual, &hidden, &nw, Some(&nb), h, EPS, false)
                .expect("ok");
        assert_eq!(result.output.len(), h);
        assert!(result.ops_fused.contains(&"LayerNorm".to_string()));

        let x_sum: Vec<f32> = residual.iter().zip(hidden.iter()).map(|(r, h)| r + h).collect();
        let expected = layer_norm_slice(&x_sum, &nw, &nb, EPS);
        assert_approx_eq(&result.output, &expected, 1e-5, "residual_layernorm");
    }

    // ── FusedOpResult fields ─────────────────────────────────────────────────

    #[test]
    fn test_fused_op_result_fields() {
        let h = 4;
        let out = 2;
        let x = vec![1.0f32, -1.0, 2.0, -2.0];
        let lw = vec![1.0f32; h];
        let lb = vec![0.0f32; h];
        let pw = vec![0.5f32; out * h];
        let result = fused_layer_norm_linear(&x, &lw, &lb, &pw, None, h, out, EPS)
            .expect("ok");
        assert_eq!(result.ops_fused.len(), 2);
        assert!(result.estimated_flops > 0);
        assert!(!result.output.is_empty());
    }

    // ── Error display ────────────────────────────────────────────────────────

    #[test]
    fn test_fused_error_display() {
        let e1 = FusedOpError::DimensionMismatch {
            op: "test_op".to_string(),
            expected: 10,
            got: 5,
        };
        let s1 = e1.to_string();
        assert!(s1.contains("test_op"));
        assert!(s1.contains("10"));
        assert!(s1.contains("5"));

        let e2 = FusedOpError::EmptyInput("my_tensor".to_string());
        assert!(e2.to_string().contains("my_tensor"));

        let e3 = FusedOpError::InvalidConfig("bad value".to_string());
        assert!(e3.to_string().contains("bad value"));
    }

    // ── Edge cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_fused_layer_norm_zero_eps_guard() {
        // With very small eps the function should still not panic (no unwrap)
        let h = 4;
        let out = 2;
        let x = vec![0.0f32; h];  // all-zero → mean=0, var=0 → uses eps for stability
        let lw = vec![1.0f32; h];
        let lb = vec![0.0f32; h];
        let pw = vec![1.0f32; out * h];
        // eps=0 is unusual but must not panic
        let result = fused_layer_norm_linear(&x, &lw, &lb, &pw, None, h, out, 0.0);
        // Result is allowed to be NaN/Inf but must not panic
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_fused_attention_single_token() {
        // seq_len = 1: typical decode step
        let seq = 1;
        let nh = 2;
        let nkv = 2;
        let hd = 4;
        let q = vec![1.0f32; seq * nh * hd];
        let k = vec![1.0f32; seq * nkv * hd];
        let result = fused_attention_scores(&q, &k, seq, nh, nkv, hd, true)
            .expect("single token ok");
        assert_eq!(result.output.len(), nh * seq * seq);
        // With 1 token, softmax of single element = 1.0
        for h in 0..nh {
            let val = result.output[h * seq * seq];
            assert!((val - 1.0).abs() < 1e-6, "single-token attn weight should be 1.0, got {val}");
        }
    }

    // ── Additional tests: numerical accuracy and coverage ─────────────────────

    /// fused_layer_norm_linear numerical accuracy: output must match sequential reference.
    #[test]
    fn test_fused_layer_norm_linear_numerical_accuracy() {
        let hidden = 8;
        let out = 4;
        let x: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.7 - 2.5).collect();
        let ln_w: Vec<f32> = (0..hidden).map(|i| 1.0 + i as f32 * 0.1).collect();
        let ln_b: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.05 - 0.2).collect();
        let lw: Vec<f32> = (0..out * hidden).map(|i| (i as f32) * 0.03 - 0.3).collect();

        let fused = fused_layer_norm_linear(&x, &ln_w, &ln_b, &lw, None, hidden, out, EPS)
            .expect("fused ok");

        // Sequential reference
        let normed = layer_norm_slice(&x, &ln_w, &ln_b, EPS);
        let ref_out = linear_projection(&normed, &lw, None, hidden, out);

        assert_approx_eq(&fused.output, &ref_out, 1e-4, "layer_norm_linear accuracy");
    }

    /// fused_rms_norm_linear numerical accuracy vs manual rms_norm_slice + linear_projection.
    #[test]
    fn test_fused_rms_norm_linear_numerical_accuracy() {
        let hidden = 6;
        let out = 3;
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 + 0.5) * 0.4).collect();
        let rw: Vec<f32> = (0..hidden).map(|i| 1.0 + i as f32 * 0.08).collect();
        let lw: Vec<f32> = (0..out * hidden).map(|i| (i as f32) * 0.04 - 0.1).collect();

        let fused = fused_rms_norm_linear(&x, &rw, &lw, None, hidden, out, EPS)
            .expect("fused rms ok");

        let normed = rms_norm_slice(&x, &rw, EPS);
        let ref_out = linear_projection(&normed, &lw, None, hidden, out);

        assert_approx_eq(&fused.output, &ref_out, 1e-4, "rms_norm_linear accuracy");
    }

    /// fused_geglu vs sequential: gelu(gate)*up then down_proj.
    #[test]
    fn test_fused_geglu_vs_sequential() {
        let h = 4;
        let inter = 6;
        let x: Vec<f32> = (0..h).map(|i| i as f32 * 0.25 - 0.5).collect();
        let gw: Vec<f32> = (0..inter * h).map(|i| (i as f32) * 0.04 - 0.1).collect();
        let uw: Vec<f32> = (0..inter * h).map(|i| (i as f32) * 0.03 + 0.02).collect();
        let dw: Vec<f32> = (0..h * inter).map(|i| (i as f32) * 0.02 - 0.04).collect();

        let fused = fused_geglu(&x, &gw, &uw, &dw, h, inter).expect("geglu ok");

        // Sequential
        let gate_out = linear_projection(&x, &gw, None, h, inter);
        let up_out = linear_projection(&x, &uw, None, h, inter);
        let activated: Vec<f32> = gate_out.iter().zip(up_out.iter())
            .map(|(&g, &u)| gelu(g) * u)
            .collect();
        let seq_out = linear_projection(&activated, &dw, None, inter, h);

        assert_approx_eq(&fused.output, &seq_out, 1e-4, "geglu vs sequential");
    }

    /// fused_swiglu with zero gate produces near-zero output (SiLU(0)=0).
    #[test]
    fn test_fused_swiglu_zero_gate_produces_near_zero_output() {
        let h = 4;
        let inter = 4;
        let x = vec![0.0f32; h];
        // Zero input → zero gate, zero up → activated = 0 * silu(0) = 0 → down = 0
        let gw = vec![1.0f32; inter * h];
        let uw = vec![1.0f32; inter * h];
        let dw = vec![1.0f32; h * inter];
        let result = fused_swiglu(&x, &gw, &uw, &dw, h, inter).expect("swiglu ok");
        assert!(result.output.iter().all(|&v| v.abs() < 1e-6),
            "zero input should produce near-zero output");
    }

    /// Dimension mismatch error for fused_layer_norm_linear.
    #[test]
    fn test_fused_layer_norm_linear_dim_mismatch() {
        let h = 4;
        let out = 2;
        let x = vec![1.0f32; h];
        let bad_ln_w = vec![1.0f32; h + 1]; // wrong size
        let ln_b = vec![0.0f32; h];
        let lw = vec![1.0f32; out * h];
        let r = fused_layer_norm_linear(&x, &bad_ln_w, &ln_b, &lw, None, h, out, EPS);
        assert!(r.is_err(), "dimension mismatch should return error");
    }

    /// Dimension mismatch error for fused_rms_norm_linear.
    #[test]
    fn test_fused_rms_norm_linear_dim_mismatch() {
        let h = 4;
        let out = 2;
        let x = vec![1.0f32; h];
        let rw = vec![1.0f32; h];
        let bad_lw = vec![1.0f32; out * h + 1]; // wrong size
        let r = fused_rms_norm_linear(&x, &rw, &bad_lw, None, h, out, EPS);
        assert!(r.is_err(), "bad linear_weight size should error");
    }

    /// Dimension mismatch for fused_swiglu gate weight.
    #[test]
    fn test_fused_swiglu_dim_mismatch() {
        let h = 4;
        let inter = 4;
        let x = vec![1.0f32; h];
        let bad_gw = vec![1.0f32; inter * h + 1];
        let uw = vec![1.0f32; inter * h];
        let dw = vec![1.0f32; h * inter];
        let r = fused_swiglu(&x, &bad_gw, &uw, &dw, h, inter);
        assert!(r.is_err(), "bad gate_weight size should error");
    }

    /// estimated_flops is positive for all 6 fused ops.
    #[test]
    fn test_all_six_ops_have_positive_estimated_flops() {
        let h = 4;
        let out = 2;
        let inter = 4;
        let seq = 2;
        let nh = 2;
        let hd = 4;

        let x = vec![0.5f32; h];
        let lnw = vec![1.0f32; h];
        let lnb = vec![0.0f32; h];
        let lw = vec![0.1f32; out * h];

        let r1 = fused_layer_norm_linear(&x, &lnw, &lnb, &lw, None, h, out, EPS).expect("ok1");
        assert!(r1.estimated_flops > 0, "layer_norm_linear flops");

        let r2 = fused_rms_norm_linear(&x, &lnw, &lw, None, h, out, EPS).expect("ok2");
        assert!(r2.estimated_flops > 0, "rms_norm_linear flops");

        let q = vec![0.5f32; seq * nh * hd];
        let k = vec![0.5f32; seq * nh * hd];
        let r3 = fused_attention_scores(&q, &k, seq, nh, nh, hd, false).expect("ok3");
        assert!(r3.estimated_flops > 0, "attention flops");

        let gw = vec![0.1f32; inter * h];
        let uw = vec![0.1f32; inter * h];
        let dw = vec![0.1f32; h * inter];
        let r4 = fused_swiglu(&x, &gw, &uw, &dw, h, inter).expect("ok4");
        assert!(r4.estimated_flops > 0, "swiglu flops");

        let r5 = fused_geglu(&x, &gw, &uw, &dw, h, inter).expect("ok5");
        assert!(r5.estimated_flops > 0, "geglu flops");

        let residual = vec![0.5f32; h];
        let hidden_s = vec![0.5f32; h];
        let r6 = fused_residual_add_norm(&residual, &hidden_s, &lnw, None, h, EPS, true).expect("ok6");
        assert!(r6.estimated_flops > 0, "residual_add_norm flops");
    }

    /// fused_residual_add_norm LayerNorm path with explicit bias.
    #[test]
    fn test_fused_residual_add_norm_layernorm_with_bias() {
        let h = 4;
        let residual: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let hidden_s: Vec<f32> = vec![-0.5, 0.5, -0.5, 0.5];
        let nw = vec![1.5f32; h];
        let nb = vec![0.1f32; h];

        let result = fused_residual_add_norm(&residual, &hidden_s, &nw, Some(&nb), h, EPS, false)
            .expect("layernorm path ok");
        assert_eq!(result.output.len(), h);

        let x_sum: Vec<f32> = residual.iter().zip(hidden_s.iter()).map(|(a, b)| a + b).collect();
        let expected = layer_norm_slice(&x_sum, &nw, &nb, EPS);
        assert_approx_eq(&result.output, &expected, 1e-5, "residual+layernorm vs ref");
    }

    /// fused_attention_scores with GQA: num_kv_heads=1, num_heads=4.
    #[test]
    fn test_fused_attention_gqa_one_kv_head() {
        let seq = 3;
        let nh = 4;
        let nkv = 1;
        let hd = 8;
        let q: Vec<f32> = (0..seq * nh * hd).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..seq * nkv * hd).map(|i| (i as f32) * 0.01).collect();
        let result = fused_attention_scores(&q, &k, seq, nh, nkv, hd, false)
            .expect("gqa nkv=1 ok");
        assert_eq!(result.output.len(), nh * seq * seq);
        // Each row must sum to 1.
        for h in 0..nh {
            for qi in 0..seq {
                let rs = h * seq * seq + qi * seq;
                let sum: f32 = result.output[rs..rs + seq].iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "GQA h={h} qi={qi} sum={sum}");
            }
        }
    }

    /// fused_attention_scores: all values in [0,1] after softmax.
    #[test]
    fn test_fused_attention_scores_all_positive_after_softmax() {
        let seq = 4;
        let nh = 2;
        let hd = 8;
        let q: Vec<f32> = (0..seq * nh * hd).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let k: Vec<f32> = (0..seq * nh * hd).map(|i| (i as f32) * 0.05).collect();
        let result = fused_attention_scores(&q, &k, seq, nh, nh, hd, false)
            .expect("attention ok");
        // After softmax, all weights in [0, 1].
        for &v in &result.output {
            assert!(v >= 0.0 && v <= 1.0 + 1e-6, "attention weight {v} out of [0,1]");
        }
    }

    /// ops_fused list contains expected kernel names for SwiGLU.
    #[test]
    fn test_fused_swiglu_ops_fused_names() {
        let h = 4;
        let inter = 4;
        let x = vec![0.1f32; h];
        let gw = vec![0.1f32; inter * h];
        let uw = vec![0.1f32; inter * h];
        let dw = vec![0.1f32; h * inter];
        let result = fused_swiglu(&x, &gw, &uw, &dw, h, inter).expect("swiglu ok");
        assert!(result.ops_fused.contains(&"gate_proj".to_string()));
        assert!(result.ops_fused.contains(&"down_proj".to_string()));
    }

    /// fused_rms_norm_linear ops_fused list correctness.
    #[test]
    fn test_fused_rms_norm_linear_ops_fused_list() {
        let h = 4;
        let out = 2;
        let x = vec![1.0f32; h];
        let rw = vec![1.0f32; h];
        let lw = vec![1.0f32; out * h];
        let r = fused_rms_norm_linear(&x, &rw, &lw, None, h, out, EPS).expect("ok");
        assert!(r.ops_fused.contains(&"RMSNorm".to_string()));
        assert!(r.ops_fused.contains(&"Linear".to_string()));
    }

    /// fused_layer_norm_linear: empty x returns error.
    #[test]
    fn test_fused_layer_norm_linear_empty_input() {
        let r = fused_layer_norm_linear(&[], &[], &[], &[], None, 0, 2, EPS);
        assert!(r.is_err());
    }
}
