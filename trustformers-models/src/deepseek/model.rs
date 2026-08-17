use crate::deepseek::config::DeepSeekConfig;
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result, TrustformersError},
    layers::{Embedding, Linear},
    ops::activations::silu,
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

// ─────────────────────────────────────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────────────────────────────────────

fn make_contiguous(t: Tensor) -> Result<Tensor> {
    let shape = t.shape().to_vec();
    t.reshape(&shape)
}

/// Simple matrix-vector multiply: out[i] = sum_j(mat[i][j] * vec[j]).
///
/// `mat` has shape `[out_dim, in_dim]`, `input` has shape `[in_dim]`.
/// Returns a flat `Vec<f32>` of length `out_dim`.
fn matmul_vec(mat: &[Vec<f64>], input: &[f32]) -> Vec<f32> {
    mat.iter()
        .map(|row| row.iter().zip(input.iter()).map(|(&w, &x)| (w as f32) * x).sum::<f32>())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm
// ─────────────────────────────────────────────────────────────────────────────

/// Root Mean Square Layer Normalisation for DeepSeek.
pub struct DeepSeekRmsNorm {
    weight: Vec<f64>,
    eps: f64,
}

impl DeepSeekRmsNorm {
    pub fn new(size: usize, eps: f64) -> Self {
        Self {
            weight: vec![1.0f64; size],
            eps,
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.weight.len()
    }

    /// Forward on a flat tensor interpreted as `[..., size]`.
    ///
    /// Each trailing `size`-element vector is normalised **independently** —
    /// RMSNorm is a per-token operation, so pooling the mean square over the
    /// whole batch would leak information between tokens.
    pub fn forward_tensor(&self, input: &Tensor) -> Result<Tensor> {
        match input {
            Tensor::F32(arr) => {
                let eps_f32 = self.eps as f32;
                let size = self.weight.len();
                if size == 0 || !arr.len().is_multiple_of(size) {
                    return Err(tensor_op_error(
                        "DeepSeekRmsNorm::forward_tensor",
                        format!(
                            "tensor of {} elements is not a multiple of the norm size {size}",
                            arr.len()
                        ),
                    ));
                }
                let w_f32: Vec<f32> = self.weight.iter().map(|&w| w as f32).collect();
                let values: Vec<f32> = arr.iter().copied().collect();
                let mut data = Vec::with_capacity(values.len());
                for chunk in values.chunks(size) {
                    let mean_sq = chunk.iter().map(|x| x * x).sum::<f32>() / size as f32;
                    let inv_rms = 1.0 / (mean_sq + eps_f32).sqrt();
                    for (value, weight) in chunk.iter().zip(w_f32.iter()) {
                        data.push(value * inv_rms * weight);
                    }
                }
                use scirs2_core::ndarray::{ArrayD, IxDyn};
                let out = ArrayD::from_shape_vec(IxDyn(arr.shape()), data).map_err(|e| {
                    tensor_op_error(
                        "DeepSeekRmsNorm::forward_tensor",
                        format!("shape error: {e}"),
                    )
                })?;
                Ok(Tensor::F32(out))
            },
            _ => Err(tensor_op_error(
                "DeepSeekRmsNorm::forward_tensor",
                "unsupported dtype",
            )),
        }
    }
}

impl Layer for DeepSeekRmsNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_tensor(&input)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MLP (SwiGLU, used for both dense FFN and individual experts)
// ─────────────────────────────────────────────────────────────────────────────

/// Single SwiGLU MLP (used as a dense FFN layer or as one MoE expert).
pub struct DeepSeekMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl DeepSeekMlp {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        Self::new_with_device(hidden_size, intermediate_size, Device::CPU)
    }

    pub fn new_with_device(
        hidden_size: usize,
        intermediate_size: usize,
        device: Device,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: Linear::new_with_device(hidden_size, intermediate_size, false, device),
            up_proj: Linear::new_with_device(hidden_size, intermediate_size, false, device),
            down_proj: Linear::new_with_device(intermediate_size, hidden_size, false, device),
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.gate_proj.parameter_count()
            + self.up_proj.parameter_count()
            + self.down_proj.parameter_count()
    }
}

impl Layer for DeepSeekMlp {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let gate = self.gate_proj.forward(input.clone())?;
        let up = self.up_proj.forward(input)?;
        let gate_act = silu(&gate)?;
        let combined = match (&gate_act, &up) {
            (Tensor::F32(g), Tensor::F32(u)) => Ok(Tensor::F32(g * u)),
            _ => Err(tensor_op_error(
                "DeepSeekMlp::forward",
                "dtype mismatch in SwiGLU",
            )),
        }?;
        self.down_proj.forward(combined)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MoE Layer
// ─────────────────────────────────────────────────────────────────────────────

/// DeepSeek-V2 MoE layer.
///
/// Combines:
/// - `n_shared_experts` always-active shared experts.
/// - `n_routed_experts` candidates with top-`k` routing per token.
///
/// The routing gate is a simple linear projection `hidden -> n_routed_experts`
/// followed by softmax + top-k selection.
pub struct DeepSeekMoeLayer {
    /// Always-active shared experts (contribute to every token).
    shared_experts: Vec<DeepSeekMlp>,
    /// Candidate routed experts.
    routed_experts: Vec<DeepSeekMlp>,
    /// Routing gate matrix: `[n_routed_experts, hidden_size]` (row-major).
    gate: Vec<Vec<f64>>,
    /// Config snapshot for routing parameters.
    n_routed_experts: usize,
    n_shared_experts: usize,
    num_experts_per_tok: usize,
    hidden_size: usize,
}

impl DeepSeekMoeLayer {
    pub fn new(config: &DeepSeekConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &DeepSeekConfig, device: Device) -> Result<Self> {
        let mut shared_experts = Vec::new();
        for _ in 0..config.n_shared_experts {
            shared_experts.push(DeepSeekMlp::new_with_device(
                config.hidden_size,
                config.intermediate_size,
                device,
            )?);
        }
        let mut routed_experts = Vec::new();
        for _ in 0..config.n_routed_experts {
            routed_experts.push(DeepSeekMlp::new_with_device(
                config.hidden_size,
                config.intermediate_size,
                device,
            )?);
        }
        // Initialise gate matrix with zeros
        let gate = vec![vec![0.0f64; config.hidden_size]; config.n_routed_experts];

        Ok(Self {
            shared_experts,
            routed_experts,
            gate,
            n_routed_experts: config.n_routed_experts,
            n_shared_experts: config.n_shared_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            hidden_size: config.hidden_size,
        })
    }

    /// Compute routing scores for a single token vector.
    ///
    /// Returns `(top_k_indices, normalised_weights)` sorted by descending score.
    pub fn route_token(&self, hidden: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let scores: Vec<f32> = matmul_vec(&self.gate, hidden);
        let mut softmax_scores = scores.clone();
        let max_val = softmax_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in softmax_scores.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        if sum > 1e-9 {
            for v in softmax_scores.iter_mut() {
                *v /= sum;
            }
        }

        // Partial sort to select top-k
        let k = self.num_experts_per_tok;
        let ne = self.n_routed_experts;
        let mut indexed: Vec<(usize, f32)> = softmax_scores.into_iter().enumerate().collect();
        for i in 0..k {
            for j in (i + 1)..ne {
                if indexed[j].1 > indexed[i].1 {
                    indexed.swap(i, j);
                }
            }
        }
        let top_k = &indexed[..k];
        let weight_sum: f32 = top_k.iter().map(|(_, w)| w).sum();
        let norm_w: Vec<f32> = if weight_sum > 1e-9 {
            top_k.iter().map(|(_, w)| w / weight_sum).collect()
        } else {
            vec![1.0 / k as f32; k]
        };
        let indices: Vec<usize> = top_k.iter().map(|(i, _)| *i).collect();
        (indices, norm_w)
    }

    pub fn parameter_count(&self) -> usize {
        let shared: usize = self.shared_experts.iter().map(|e| e.parameter_count()).sum();
        let routed: usize = self.routed_experts.iter().map(|e| e.parameter_count()).sum();
        let gate = self.n_routed_experts * self.hidden_size;
        shared + routed + gate
    }

    pub fn n_shared_experts(&self) -> usize {
        self.n_shared_experts
    }

    pub fn n_routed_experts(&self) -> usize {
        self.n_routed_experts
    }
}

impl Layer for DeepSeekMoeLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let shape = input.shape().to_vec();
        let (batch, seq_len, hidden_size) = match shape.len() {
            3 => (shape[0], shape[1], shape[2]),
            2 => (1, shape[0], shape[1]),
            _ => {
                return Err(TrustformersError::shape_error(
                    "DeepSeekMoeLayer expects 2D or 3D input".to_string(),
                ))
            },
        };
        let num_tokens = batch * seq_len;
        let flat = input.reshape(&[num_tokens, hidden_size])?;

        let flat_data: Vec<f32> = match &flat {
            Tensor::F32(arr) => arr.iter().cloned().collect(),
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 input".to_string(),
                ))
            },
        };

        let mut output_data = vec![0.0f32; num_tokens * hidden_size];

        for tok in 0..num_tokens {
            let tok_slice = &flat_data[tok * hidden_size..(tok + 1) * hidden_size];
            let tok_tensor = Tensor::from_vec(tok_slice.to_vec(), &[1, 1, hidden_size])?;

            // Shared experts contribute unconditionally
            for expert in &self.shared_experts {
                let out = expert.forward(tok_tensor.clone())?;
                let out_data: Vec<f32> = match &out {
                    Tensor::F32(arr) => arr.iter().cloned().collect(),
                    _ => {
                        return Err(TrustformersError::invalid_input_simple(
                            "Expected F32 expert output".to_string(),
                        ))
                    },
                };
                for h in 0..hidden_size {
                    output_data[tok * hidden_size + h] += out_data[h];
                }
            }

            // Routed experts: top-k selection
            let (top_indices, norm_weights) = self.route_token(tok_slice);
            for (rank, &expert_idx) in top_indices.iter().enumerate() {
                let out = self.routed_experts[expert_idx].forward(tok_tensor.clone())?;
                let out_data: Vec<f32> = match &out {
                    Tensor::F32(arr) => arr.iter().cloned().collect(),
                    _ => {
                        return Err(TrustformersError::invalid_input_simple(
                            "Expected F32 routed expert output".to_string(),
                        ))
                    },
                };
                let w = norm_weights[rank];
                for h in 0..hidden_size {
                    output_data[tok * hidden_size + h] += w * out_data[h];
                }
            }
        }

        Tensor::from_vec(output_data, &[batch, seq_len, hidden_size])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-Head Latent Attention (MLA)
// ─────────────────────────────────────────────────────────────────────────────

/// DeepSeek-V2 Multi-Head Latent Attention.
///
/// The MLA mechanism compresses K/V into a low-rank latent vector:
/// ```text
/// c_kv = W_DKV @ x              (compress: hidden -> kv_lora_rank)
/// k    = W_UK  @ c_kv           (decompress: kv_lora_rank -> nheads*head_dim)
/// v    = W_UV  @ c_kv           (decompress: kv_lora_rank -> nheads*v_head_dim)
/// q    = W_Q   @ x              (query projection, or W_DQ then W_UQ if q_lora_rank)
/// out  = attention(q, k, v)
/// y    = W_O   @ out
/// ```
///
/// This design reduces the KV cache from `O(L * nheads * head_dim)` to
/// `O(L * kv_lora_rank)` where `kv_lora_rank << nheads * head_dim`.
///
/// Positional information travels on a **decoupled RoPE path** (DeepSeek-V2,
/// §2.1.2): the compressed content path carries no position, so an extra
/// `W_QR`/`W_KR` pair produces `rope_head_dim` channels that are rotated and
/// concatenated to the content scores. The key half `k_r` is shared by every
/// head, which is what keeps the cache small.
pub struct DeepSeekMlaAttention {
    /// Downproject hidden -> kv_lora_rank
    w_dkv: Vec<Vec<f64>>,
    /// Upproject kv_lora_rank -> nheads * head_dim (keys)
    w_uk: Vec<Vec<f64>>,
    /// Upproject kv_lora_rank -> nheads * v_head_dim (values)
    w_uv: Vec<Vec<f64>>,
    /// Query projection: hidden -> nheads * head_dim
    w_q: Vec<Vec<f64>>,
    /// Decoupled RoPE query projection: hidden -> nheads * rope_head_dim
    w_qr: Vec<Vec<f64>>,
    /// Decoupled RoPE key projection (shared across heads): hidden -> rope_head_dim
    w_kr: Vec<Vec<f64>>,
    /// Output projection: nheads * v_head_dim -> hidden
    w_o: Vec<Vec<f64>>,
    num_heads: usize,
    head_dim: usize,
    v_head_dim: usize,
    rope_head_dim: usize,
    rope_theta: f64,
    kv_lora_rank: usize,
    hidden_size: usize,
}

/// Deterministic weight initialisation.
///
/// A zero-initialised projection makes every attention score identical and hides
/// bugs behind a constant output, so the matrices are filled with a reproducible
/// pseudo-random draw scaled by `1/sqrt(fan_in)` (the usual Xavier-style scale).
/// Loading real checkpoint weights replaces them wholesale via the setters.
fn init_matrix(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let scale = 1.0 / (cols.max(1) as f64).sqrt();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    state =
                        state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let unit = ((state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
                    unit * scale
                })
                .collect()
        })
        .collect()
}

impl DeepSeekMlaAttention {
    pub fn new(config: &DeepSeekConfig) -> Self {
        let num_heads = config.num_attention_heads;
        let head_dim = config.head_dim();
        let v_head_dim = config.v_head_dim;
        let kv_lora_rank = config.kv_lora_rank;
        let hidden_size = config.hidden_size;
        let rope_head_dim = config.rope_head_dim;

        let w_dkv = init_matrix(kv_lora_rank, hidden_size, 0x0D5E);
        let w_uk = init_matrix(num_heads * head_dim, kv_lora_rank, 0x0D5F);
        let w_uv = init_matrix(num_heads * v_head_dim, kv_lora_rank, 0x0D60);
        let w_q = init_matrix(num_heads * head_dim, hidden_size, 0x0D61);
        let w_qr = init_matrix(num_heads * rope_head_dim, hidden_size, 0x0D62);
        let w_kr = init_matrix(rope_head_dim, hidden_size, 0x0D63);
        let w_o = init_matrix(hidden_size, num_heads * v_head_dim, 0x0D64);

        Self {
            w_dkv,
            w_uk,
            w_uv,
            w_q,
            w_qr,
            w_kr,
            w_o,
            num_heads,
            head_dim,
            v_head_dim,
            rope_head_dim,
            rope_theta: config.rope_theta,
            kv_lora_rank,
            hidden_size,
        }
    }

    fn check_shape(name: &str, matrix: &[Vec<f64>], rows: usize, cols: usize) -> Result<()> {
        if matrix.len() != rows || matrix.iter().any(|row| row.len() != cols) {
            return Err(tensor_op_error(
                name,
                format!(
                    "expected a [{rows}, {cols}] matrix, got [{}, {}]",
                    matrix.len(),
                    matrix.first().map(|row| row.len()).unwrap_or(0)
                ),
            ));
        }
        Ok(())
    }

    /// Replace `W_DKV` (`[kv_lora_rank, hidden_size]`).
    pub fn set_kv_down_proj(&mut self, weight: Vec<Vec<f64>>) -> Result<()> {
        Self::check_shape(
            "DeepSeekMlaAttention::set_kv_down_proj",
            &weight,
            self.kv_lora_rank,
            self.hidden_size,
        )?;
        self.w_dkv = weight;
        Ok(())
    }

    /// Replace `W_UK` (`[num_heads * head_dim, kv_lora_rank]`).
    pub fn set_key_up_proj(&mut self, weight: Vec<Vec<f64>>) -> Result<()> {
        Self::check_shape(
            "DeepSeekMlaAttention::set_key_up_proj",
            &weight,
            self.num_heads * self.head_dim,
            self.kv_lora_rank,
        )?;
        self.w_uk = weight;
        Ok(())
    }

    /// Replace `W_UV` (`[num_heads * v_head_dim, kv_lora_rank]`).
    pub fn set_value_up_proj(&mut self, weight: Vec<Vec<f64>>) -> Result<()> {
        Self::check_shape(
            "DeepSeekMlaAttention::set_value_up_proj",
            &weight,
            self.num_heads * self.v_head_dim,
            self.kv_lora_rank,
        )?;
        self.w_uv = weight;
        Ok(())
    }

    /// Replace `W_Q` (`[num_heads * head_dim, hidden_size]`).
    pub fn set_query_proj(&mut self, weight: Vec<Vec<f64>>) -> Result<()> {
        Self::check_shape(
            "DeepSeekMlaAttention::set_query_proj",
            &weight,
            self.num_heads * self.head_dim,
            self.hidden_size,
        )?;
        self.w_q = weight;
        Ok(())
    }

    /// Replace the decoupled-RoPE query projection
    /// (`[num_heads * rope_head_dim, hidden_size]`).
    pub fn set_query_rope_proj(&mut self, weight: Vec<Vec<f64>>) -> Result<()> {
        Self::check_shape(
            "DeepSeekMlaAttention::set_query_rope_proj",
            &weight,
            self.num_heads * self.rope_head_dim,
            self.hidden_size,
        )?;
        self.w_qr = weight;
        Ok(())
    }

    /// Replace the shared decoupled-RoPE key projection
    /// (`[rope_head_dim, hidden_size]`).
    pub fn set_key_rope_proj(&mut self, weight: Vec<Vec<f64>>) -> Result<()> {
        Self::check_shape(
            "DeepSeekMlaAttention::set_key_rope_proj",
            &weight,
            self.rope_head_dim,
            self.hidden_size,
        )?;
        self.w_kr = weight;
        Ok(())
    }

    /// Replace `W_O` (`[hidden_size, num_heads * v_head_dim]`).
    pub fn set_output_proj(&mut self, weight: Vec<Vec<f64>>) -> Result<()> {
        Self::check_shape(
            "DeepSeekMlaAttention::set_output_proj",
            &weight,
            self.hidden_size,
            self.num_heads * self.v_head_dim,
        )?;
        self.w_o = weight;
        Ok(())
    }

    /// Rotate `vector` (length `rope_head_dim`) for absolute position `pos`.
    ///
    /// Uses the `rotate_half` convention: channel `i` pairs with `i + rd/2`.
    fn apply_rope(&self, vector: &mut [f32], pos: usize) {
        let half = self.rope_head_dim / 2;
        for i in 0..half {
            let inv_freq = 1.0 / self.rope_theta.powf(2.0 * i as f64 / self.rope_head_dim as f64);
            let angle = (pos as f64 * inv_freq) as f32;
            let (sin_val, cos_val) = angle.sin_cos();
            let x0 = vector[i];
            let x1 = vector[i + half];
            vector[i] = x0 * cos_val - x1 * sin_val;
            vector[i + half] = x0 * sin_val + x1 * cos_val;
        }
    }

    /// Full Multi-head Latent Attention over one sequence.
    ///
    /// `x` is a flat `[seq_len * hidden_size]` buffer; the result has the same
    /// shape. The computation follows DeepSeek-V2:
    ///
    /// 1. `c_kv = W_DKV x` compresses the KV state to `kv_lora_rank` channels,
    /// 2. `k_c = W_UK c_kv` / `v = W_UV c_kv` decompress per head,
    /// 3. `q_c = W_Q x` and the decoupled RoPE halves `q_r`, `k_r` add position,
    /// 4. scores are `(q_c·k_c + q_r·k_r) / sqrt(head_dim + rope_head_dim)`,
    ///    causally masked and softmaxed,
    /// 5. the value-weighted sum is projected back with `W_O`.
    pub fn forward_sequence(&self, x: &[f32], seq_len: usize) -> Result<Vec<f32>> {
        let hidden = self.hidden_size;
        if seq_len == 0 {
            return Ok(Vec::new());
        }
        if x.len() != seq_len * hidden {
            return Err(tensor_op_error(
                "DeepSeekMlaAttention::forward_sequence",
                format!("expected {} elements, got {}", seq_len * hidden, x.len()),
            ));
        }

        let head_dim = self.head_dim;
        let v_head_dim = self.v_head_dim;
        let rope_dim = self.rope_head_dim;
        let num_heads = self.num_heads;

        // Per-token projections.
        let mut keys = Vec::with_capacity(seq_len); // [nheads * head_dim]
        let mut values = Vec::with_capacity(seq_len); // [nheads * v_head_dim]
        let mut queries = Vec::with_capacity(seq_len); // [nheads * head_dim]
        let mut query_rope = Vec::with_capacity(seq_len); // [nheads * rope_dim]
        let mut key_rope = Vec::with_capacity(seq_len); // [rope_dim] (shared)

        for (pos, token) in x.chunks(hidden).enumerate().take(seq_len) {
            let c_kv = matmul_vec(&self.w_dkv, token);
            keys.push(matmul_vec(&self.w_uk, &c_kv));
            values.push(matmul_vec(&self.w_uv, &c_kv));
            queries.push(matmul_vec(&self.w_q, token));

            if rope_dim >= 2 {
                let mut q_r = matmul_vec(&self.w_qr, token);
                for head in 0..num_heads {
                    self.apply_rope(&mut q_r[head * rope_dim..(head + 1) * rope_dim], pos);
                }
                query_rope.push(q_r);

                let mut k_r = matmul_vec(&self.w_kr, token);
                self.apply_rope(&mut k_r, pos);
                key_rope.push(k_r);
            }
        }

        let use_rope = rope_dim >= 2;
        let scale = 1.0 / ((head_dim + if use_rope { rope_dim } else { 0 }) as f32).sqrt();

        let mut context = vec![0.0f32; seq_len * num_heads * v_head_dim];
        let mut scores = vec![0.0f32; seq_len];
        for head in 0..num_heads {
            for query_pos in 0..seq_len {
                let mut max_score = f32::NEG_INFINITY;
                for key_pos in 0..=query_pos {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += queries[query_pos][head * head_dim + d]
                            * keys[key_pos][head * head_dim + d];
                    }
                    if use_rope {
                        for d in 0..rope_dim {
                            dot +=
                                query_rope[query_pos][head * rope_dim + d] * key_rope[key_pos][d];
                        }
                    }
                    dot *= scale;
                    scores[key_pos] = dot;
                    if dot > max_score {
                        max_score = dot;
                    }
                }

                let mut sum = 0.0f32;
                for score in scores.iter_mut().take(query_pos + 1) {
                    *score = (*score - max_score).exp();
                    sum += *score;
                }
                let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };

                let out_base = (query_pos * num_heads + head) * v_head_dim;
                for key_pos in 0..=query_pos {
                    let weight = scores[key_pos] * inv_sum;
                    for d in 0..v_head_dim {
                        context[out_base + d] += weight * values[key_pos][head * v_head_dim + d];
                    }
                }
            }
        }

        // Output projection per token.
        let attn_size = num_heads * v_head_dim;
        let mut output = Vec::with_capacity(seq_len * hidden);
        for token_context in context.chunks(attn_size) {
            output.extend_from_slice(&matmul_vec(&self.w_o, token_context));
        }
        Ok(output)
    }

    /// MLA forward for a single token vector.
    ///
    /// A one-token sequence attends only to itself, so the softmax is trivially
    /// `1` and the output is `W_O · v` — the real mechanism, not an approximation.
    /// Inputs shorter than `hidden_size` are zero-padded and longer ones are
    /// truncated, which makes the length precondition of
    /// [`forward_sequence`](Self::forward_sequence) hold by construction.
    /// Returns the output vector `[hidden_size]`.
    pub fn forward_token(&self, x: &[f32]) -> Vec<f32> {
        let mut token = vec![0.0f32; self.hidden_size];
        let copied = x.len().min(self.hidden_size);
        token[..copied].copy_from_slice(&x[..copied]);
        // The length now matches exactly, so `forward_sequence` cannot fail; the
        // fallback keeps the infallible signature without hiding a real error.
        self.forward_sequence(&token, 1)
            .unwrap_or_else(|_| vec![0.0f32; self.hidden_size])
    }

    pub fn kv_lora_rank(&self) -> usize {
        self.kv_lora_rank
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Per-head dimension of the decoupled RoPE path.
    pub fn rope_head_dim(&self) -> usize {
        self.rope_head_dim
    }

    pub fn parameter_count(&self) -> usize {
        let w_dkv = self.kv_lora_rank * self.hidden_size;
        let w_uk = self.num_heads * self.head_dim * self.kv_lora_rank;
        let w_uv = self.num_heads * self.v_head_dim * self.kv_lora_rank;
        let w_q = self.num_heads * self.head_dim * self.hidden_size;
        let w_qr = self.num_heads * self.rope_head_dim * self.hidden_size;
        let w_kr = self.rope_head_dim * self.hidden_size;
        let w_o = self.hidden_size * self.num_heads * self.v_head_dim;
        w_dkv + w_uk + w_uv + w_q + w_qr + w_kr + w_o
    }
}

impl Layer for DeepSeekMlaAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let shape = input.shape().to_vec();
        let (batch, seq_len, _hidden) = match shape.len() {
            3 => (shape[0], shape[1], shape[2]),
            2 => (1, shape[0], shape[1]),
            _ => {
                return Err(TrustformersError::shape_error(
                    "DeepSeekMlaAttention expects 2D or 3D input".to_string(),
                ))
            },
        };

        let data: Vec<f32> = match &input {
            Tensor::F32(arr) => arr.iter().cloned().collect(),
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 input for MLA".to_string(),
                ))
            },
        };

        let hs = self.hidden_size;
        if data.len() != batch * seq_len * hs {
            return Err(tensor_op_error(
                "DeepSeekMlaAttention::forward",
                format!(
                    "expected {} elements, got {}",
                    batch * seq_len * hs,
                    data.len()
                ),
            ));
        }

        // Attention runs over each sequence in the batch independently.
        let mut out_data = Vec::with_capacity(batch * seq_len * hs);
        for sequence in data.chunks(seq_len * hs) {
            out_data.extend_from_slice(&self.forward_sequence(sequence, seq_len)?);
        }

        Tensor::from_vec(out_data, &[batch, seq_len, hs])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FFN kind enum
// ─────────────────────────────────────────────────────────────────────────────

/// Either a dense MLP or a MoE block, depending on layer index.
pub enum DeepSeekFfnKind {
    Dense(DeepSeekMlp),
    Moe(DeepSeekMoeLayer),
}

impl DeepSeekFfnKind {
    pub fn parameter_count(&self) -> usize {
        match self {
            Self::Dense(m) => m.parameter_count(),
            Self::Moe(m) => m.parameter_count(),
        }
    }
}

impl Layer for DeepSeekFfnKind {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match self {
            Self::Dense(m) => m.forward(input),
            Self::Moe(m) => m.forward(input),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Decoder Layer
// ─────────────────────────────────────────────────────────────────────────────

/// Single DeepSeek-V2 decoder layer.
pub struct DeepSeekDecoderLayer {
    self_attn: DeepSeekMlaAttention,
    ffn: DeepSeekFfnKind,
    input_norm: DeepSeekRmsNorm,
    post_attn_norm: DeepSeekRmsNorm,
}

impl DeepSeekDecoderLayer {
    pub fn new(config: &DeepSeekConfig, layer_idx: usize) -> Result<Self> {
        Self::new_with_device(config, layer_idx, Device::CPU)
    }

    pub fn new_with_device(
        config: &DeepSeekConfig,
        layer_idx: usize,
        device: Device,
    ) -> Result<Self> {
        let self_attn = DeepSeekMlaAttention::new(config);
        let ffn = if config.is_moe_layer(layer_idx) {
            DeepSeekFfnKind::Moe(DeepSeekMoeLayer::new_with_device(config, device)?)
        } else {
            DeepSeekFfnKind::Dense(DeepSeekMlp::new_with_device(
                config.hidden_size,
                config.intermediate_size,
                device,
            )?)
        };
        let input_norm = DeepSeekRmsNorm::new(config.hidden_size, config.rms_norm_eps);
        let post_attn_norm = DeepSeekRmsNorm::new(config.hidden_size, config.rms_norm_eps);
        Ok(Self {
            self_attn,
            ffn,
            input_norm,
            post_attn_norm,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.self_attn.parameter_count()
            + self.ffn.parameter_count()
            + self.input_norm.parameter_count()
            + self.post_attn_norm.parameter_count()
    }

    pub fn is_moe(&self) -> bool {
        matches!(self.ffn, DeepSeekFfnKind::Moe(_))
    }
}

impl Layer for DeepSeekDecoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let normed = make_contiguous(self.input_norm.forward(input.clone())?)?;
        let attn_out = self.self_attn.forward(normed)?;
        let input_c = make_contiguous(input)?;
        let after_attn = input_c.add(&make_contiguous(attn_out)?)?;

        let normed2 = make_contiguous(self.post_attn_norm.forward(after_attn.clone())?)?;
        let ffn_out = self.ffn.forward(normed2)?;
        make_contiguous(after_attn)?.add(&make_contiguous(ffn_out)?)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DeepSeek Base Model
// ─────────────────────────────────────────────────────────────────────────────

/// DeepSeek-V2 transformer model (without LM head).
pub struct DeepSeekModel {
    config: DeepSeekConfig,
    embed_tokens: Embedding,
    layers: Vec<DeepSeekDecoderLayer>,
    norm: DeepSeekRmsNorm,
}

impl DeepSeekModel {
    pub fn new(config: DeepSeekConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: DeepSeekConfig, device: Device) -> Result<Self> {
        config.validate()?;
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            layers.push(DeepSeekDecoderLayer::new_with_device(&config, idx, device)?);
        }
        let norm = DeepSeekRmsNorm::new(config.hidden_size, config.rms_norm_eps);
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn config(&self) -> &DeepSeekConfig {
        &self.config
    }

    pub fn parameter_count(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| l.parameter_count()).sum();
        self.embed_tokens.parameter_count() + layer_params + self.norm.parameter_count()
    }

    pub fn run(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        let seq_len = input_ids.len();
        let embeddings = self.embed_tokens.forward(input_ids)?;
        let mut hidden = embeddings.reshape(&[1, seq_len, self.config.hidden_size])?;
        for layer in &self.layers {
            hidden = layer.forward(hidden)?;
        }
        make_contiguous(self.norm.forward(hidden)?)
    }
}

impl Model for DeepSeekModel {
    type Config = DeepSeekConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        self.run(input_ids)
    }

    /// Not supported yet — returns a structured error rather than pretending.
    ///
    /// A DeepSeek-V2 checkpoint stores MLA as fused tensors
    /// (`self_attn.q_proj` = `[num_heads * (qk_nope_head_dim + qk_rope_head_dim), hidden]`,
    /// `self_attn.kv_a_proj_with_mqa` = `[kv_lora_rank + qk_rope_head_dim, hidden]`,
    /// `self_attn.kv_b_proj` = `[num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]`)
    /// plus a `kv_a_layernorm` that this implementation does not model, so the
    /// checkpoint cannot be bound without inventing a correspondence. Build the
    /// weights explicitly through the `DeepSeekMlaAttention::set_*` setters until
    /// the fused split is implemented.
    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(TrustformersError::not_implemented(
            "DeepSeek-V2 checkpoint loading is not implemented: the MLA weights are fused \
             (q_proj = nope+rope rows, kv_a_proj_with_mqa = c_kv+rope rows, kv_b_proj = k+v rows) \
             and kv_a_layernorm is not modelled. Use the DeepSeekMlaAttention::set_* setters to \
             install weights explicitly."
                .to_string(),
        ))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.parameter_count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DeepSeek Causal LM
// ─────────────────────────────────────────────────────────────────────────────

/// DeepSeek-V2 with causal language-modelling head.
pub struct DeepSeekForCausalLM {
    model: DeepSeekModel,
    lm_head: Linear,
}

impl DeepSeekForCausalLM {
    pub fn new(config: DeepSeekConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: DeepSeekConfig, device: Device) -> Result<Self> {
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);
        let model = DeepSeekModel::new_with_device(config, device)?;
        Ok(Self { model, lm_head })
    }

    pub fn config(&self) -> &DeepSeekConfig {
        self.model.config()
    }

    pub fn parameter_count(&self) -> usize {
        self.model.parameter_count() + self.lm_head.parameter_count()
    }

    pub fn forward(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        let hidden = self.model.run(input_ids)?;
        self.lm_head.forward(hidden)
    }
}

impl Model for DeepSeekForCausalLM {
    type Config = DeepSeekConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        DeepSeekForCausalLM::forward(self, input_ids)
    }

    /// Not supported yet — returns a structured error rather than pretending.
    ///
    /// A DeepSeek-V2 checkpoint stores MLA as fused tensors
    /// (`self_attn.q_proj` = `[num_heads * (qk_nope_head_dim + qk_rope_head_dim), hidden]`,
    /// `self_attn.kv_a_proj_with_mqa` = `[kv_lora_rank + qk_rope_head_dim, hidden]`,
    /// `self_attn.kv_b_proj` = `[num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]`)
    /// plus a `kv_a_layernorm` that this implementation does not model, so the
    /// checkpoint cannot be bound without inventing a correspondence. Build the
    /// weights explicitly through the `DeepSeekMlaAttention::set_*` setters until
    /// the fused split is implemented.
    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(TrustformersError::not_implemented(
            "DeepSeek-V2 checkpoint loading is not implemented: the MLA weights are fused \
             (q_proj = nope+rope rows, kv_a_proj_with_mqa = c_kv+rope rows, kv_b_proj = k+v rows) \
             and kv_a_layernorm is not modelled. Use the DeepSeekMlaAttention::set_* setters to \
             install weights explicitly."
                .to_string(),
        ))
    }

    fn get_config(&self) -> &Self::Config {
        self.model.config()
    }

    fn num_parameters(&self) -> usize {
        self.parameter_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek::config::DeepSeekConfig;
    use trustformers_core::traits::Config;

    /// LCG pseudo-random: a=6364136223846793005, c=1442695040888963407
    fn lcg_next(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
        (*state as f32) / (u64::MAX as f32)
    }

    fn lcg_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| lcg_next(&mut s) * 2.0 - 1.0).collect()
    }

    /// Minimal config suitable for fast unit tests.
    fn test_cfg() -> DeepSeekConfig {
        DeepSeekConfig::small_test()
    }

    // ── config tests ──────────────────────────────────────────────────────

    #[test]
    fn test_config_validate_passes_for_small_test() {
        test_cfg().validate().expect("small_test config should pass validation");
    }

    #[test]
    fn test_config_head_dim() {
        let cfg = test_cfg();
        assert_eq!(
            cfg.head_dim(),
            cfg.hidden_size / cfg.num_attention_heads,
            "head_dim should be hidden_size / num_attention_heads"
        );
    }

    #[test]
    fn test_config_is_moe_layer_dense_prefix() {
        let cfg = test_cfg();
        // first_k_dense_replace = 1; so layer 0 should NOT be MoE
        assert!(
            !cfg.is_moe_layer(0),
            "layer 0 should be dense (in dense prefix)"
        );
    }

    #[test]
    fn test_config_is_moe_layer_after_dense_prefix() {
        let cfg = test_cfg();
        // moe_layer_freq=1 => every layer after dense prefix is MoE
        assert!(
            cfg.is_moe_layer(1),
            "layer 1 should be MoE (after dense prefix with freq=1)"
        );
    }

    #[test]
    fn test_config_architecture_string() {
        let cfg = test_cfg();
        assert_eq!(cfg.architecture(), "DeepSeek-V2");
    }

    // ── DeepSeekRmsNorm tests ─────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_parameter_count() {
        let size = 64;
        let norm = DeepSeekRmsNorm::new(size, 1e-6);
        assert_eq!(
            norm.parameter_count(),
            size,
            "RMSNorm param count = hidden_size"
        );
    }

    #[test]
    fn test_rmsnorm_forward_shape_preserved() {
        let size = 64;
        let norm = DeepSeekRmsNorm::new(size, 1e-6);
        let data = lcg_vec(size, 1);
        let t = Tensor::from_vec(data, &[1, 1, size]).expect("build tensor");
        let out = norm.forward_tensor(&t).expect("RMSNorm forward should succeed");
        assert_eq!(
            out.shape(),
            t.shape(),
            "RMSNorm should preserve tensor shape"
        );
    }

    #[test]
    fn test_rmsnorm_zero_input_produces_near_zero_output() {
        let size = 8;
        let norm = DeepSeekRmsNorm::new(size, 1e-6);
        let data = vec![0.0f32; size];
        let t = Tensor::from_vec(data, &[1, 1, size]).expect("build tensor");
        let out = norm.forward_tensor(&t).expect("RMSNorm forward on zeros");
        if let Tensor::F32(arr) = &out {
            for &v in arr.iter() {
                assert!(v.abs() < 1e-4, "RMSNorm of all-zeros should be near-zero");
            }
        }
    }

    // ── DeepSeekMlp tests ─────────────────────────────────────────────────

    #[test]
    fn test_mlp_parameter_count() {
        let cfg = test_cfg();
        let mlp = DeepSeekMlp::new(cfg.hidden_size, cfg.intermediate_size)
            .expect("should create DeepSeekMlp");
        let expected = cfg.hidden_size * cfg.intermediate_size * 3;
        assert_eq!(mlp.parameter_count(), expected, "MLP param count mismatch");
    }

    #[test]
    fn test_mlp_forward_output_shape() {
        let cfg = test_cfg();
        let mlp = DeepSeekMlp::new(cfg.hidden_size, cfg.intermediate_size)
            .expect("should create DeepSeekMlp");
        let data = lcg_vec(cfg.hidden_size, 3);
        let input = Tensor::from_vec(data, &[1, 1, cfg.hidden_size]).expect("build tensor");
        let output = mlp.forward(input).expect("MLP forward should succeed");
        let shape = output.shape();
        assert_eq!(
            shape[shape.len() - 1],
            cfg.hidden_size,
            "MLP output last dim should match hidden_size"
        );
    }

    // ── DeepSeekMoeLayer tests ────────────────────────────────────────────

    #[test]
    fn test_moe_layer_creation() {
        let cfg = test_cfg();
        let moe = DeepSeekMoeLayer::new(&cfg).expect("should create DeepSeekMoeLayer");
        assert_eq!(
            moe.n_routed_experts(),
            cfg.n_routed_experts,
            "routed expert count should match config"
        );
        assert_eq!(
            moe.n_shared_experts(),
            cfg.n_shared_experts,
            "shared expert count should match config"
        );
    }

    #[test]
    fn test_moe_layer_route_token_returns_k_indices() {
        let cfg = test_cfg();
        let moe = DeepSeekMoeLayer::new(&cfg).expect("should create DeepSeekMoeLayer");
        let hidden = lcg_vec(cfg.hidden_size, 42);
        let (indices, weights) = moe.route_token(&hidden);
        assert_eq!(
            indices.len(),
            cfg.num_experts_per_tok,
            "should return exactly num_experts_per_tok indices"
        );
        assert_eq!(
            weights.len(),
            cfg.num_experts_per_tok,
            "should return exactly num_experts_per_tok weights"
        );
    }

    #[test]
    fn test_moe_layer_route_token_weights_sum_to_one() {
        let cfg = test_cfg();
        let moe = DeepSeekMoeLayer::new(&cfg).expect("should create DeepSeekMoeLayer");
        let hidden = lcg_vec(cfg.hidden_size, 99);
        let (_, weights) = moe.route_token(&hidden);
        let total: f32 = weights.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "routing weights should sum to 1.0, got {total}"
        );
    }

    #[test]
    fn test_moe_layer_route_token_indices_in_range() {
        let cfg = test_cfg();
        let moe = DeepSeekMoeLayer::new(&cfg).expect("should create DeepSeekMoeLayer");
        let hidden = lcg_vec(cfg.hidden_size, 7);
        let (indices, _) = moe.route_token(&hidden);
        for &idx in &indices {
            assert!(
                idx < cfg.n_routed_experts,
                "expert index {idx} out of range"
            );
        }
    }

    #[test]
    fn test_moe_layer_forward_output_shape() {
        let cfg = test_cfg();
        let moe = DeepSeekMoeLayer::new(&cfg).expect("should create DeepSeekMoeLayer");
        let data = lcg_vec(cfg.hidden_size, 11);
        let input = Tensor::from_vec(data, &[1, 1, cfg.hidden_size]).expect("build tensor");
        let output = moe.forward(input).expect("MoE forward should succeed");
        let shape = output.shape();
        assert_eq!(
            shape[shape.len() - 1],
            cfg.hidden_size,
            "MoE output last dim should equal hidden_size"
        );
    }

    // ── DeepSeekMlaAttention tests ────────────────────────────────────────

    #[test]
    fn test_mla_attention_kv_lora_rank() {
        let cfg = test_cfg();
        let mla = DeepSeekMlaAttention::new(&cfg);
        assert_eq!(
            mla.kv_lora_rank(),
            cfg.kv_lora_rank,
            "kv_lora_rank should match config"
        );
    }

    #[test]
    fn test_mla_attention_num_heads() {
        let cfg = test_cfg();
        let mla = DeepSeekMlaAttention::new(&cfg);
        assert_eq!(
            mla.num_heads(),
            cfg.num_attention_heads,
            "num_heads should match config"
        );
    }

    #[test]
    fn test_mla_attention_forward_token_output_size() {
        let cfg = test_cfg();
        let mla = DeepSeekMlaAttention::new(&cfg);
        let x = lcg_vec(cfg.hidden_size, 5);
        let out = mla.forward_token(&x);
        assert_eq!(
            out.len(),
            cfg.hidden_size,
            "MLA forward_token output should have hidden_size elements"
        );
    }

    #[test]
    fn test_mla_attention_forward_tensor_output_shape() {
        let cfg = test_cfg();
        let mla = DeepSeekMlaAttention::new(&cfg);
        let data = lcg_vec(2 * 4 * cfg.hidden_size, 33);
        let input = Tensor::from_vec(data, &[2, 4, cfg.hidden_size]).expect("build tensor");
        let output = Layer::forward(&mla, input).expect("MLA forward should succeed");
        let shape = output.shape();
        assert_eq!(shape[0], 2, "batch dim preserved");
        assert_eq!(shape[1], 4, "seq dim preserved");
        assert_eq!(shape[2], cfg.hidden_size, "hidden dim preserved");
    }

    /// MLA must read the decompressed keys: perturbing an earlier token has to
    /// change the last token's output. The old code discarded `k` and `q` and
    /// returned a scaled `v`, so earlier tokens had no effect at all.
    #[test]
    fn test_mla_attends_to_previous_tokens() {
        let cfg = test_cfg();
        let mla = DeepSeekMlaAttention::new(&cfg);
        let seq_len = 3;
        let hidden = cfg.hidden_size;
        let base = lcg_vec(seq_len * hidden, 101);

        let out_a = mla.forward_sequence(&base, seq_len).expect("sequence forward");

        let mut perturbed = base.clone();
        for value in perturbed.iter_mut().take(hidden) {
            *value += 1.0;
        }
        let out_b = mla.forward_sequence(&perturbed, seq_len).expect("sequence forward");

        let last = (seq_len - 1) * hidden;
        let diff = out_a[last..]
            .iter()
            .zip(out_b[last..].iter())
            .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
        assert!(
            diff > 1e-6,
            "the final token must attend to earlier tokens (diff {diff})"
        );
    }

    /// Causality: a later token must not influence an earlier output.
    #[test]
    fn test_mla_is_causal() {
        let cfg = test_cfg();
        let mla = DeepSeekMlaAttention::new(&cfg);
        let seq_len = 3;
        let hidden = cfg.hidden_size;
        let base = lcg_vec(seq_len * hidden, 202);

        let out_a = mla.forward_sequence(&base, seq_len).expect("forward");
        let mut perturbed = base.clone();
        for value in perturbed.iter_mut().skip((seq_len - 1) * hidden) {
            *value += 2.0;
        }
        let out_b = mla.forward_sequence(&perturbed, seq_len).expect("forward");

        for i in 0..(seq_len - 1) * hidden {
            assert!(
                (out_a[i] - out_b[i]).abs() < 1e-5,
                "position {} must not see the future",
                i / hidden
            );
        }
    }

    /// The scores must combine the content path *and* the decoupled RoPE path:
    /// two identical tokens at different positions produce different attention.
    #[test]
    fn test_mla_decoupled_rope_makes_positions_distinguishable() {
        let cfg = test_cfg();
        let mla = DeepSeekMlaAttention::new(&cfg);
        assert!(mla.rope_head_dim() >= 2, "test config must exercise RoPE");

        let hidden = cfg.hidden_size;
        let token_a = lcg_vec(hidden, 303);
        let token_b = lcg_vec(hidden, 404);

        // Sequence [a, b] versus [b, a]: with a position-blind score the second
        // output would be the same weighted mixture in both orders.
        let mut ab = token_a.clone();
        ab.extend_from_slice(&token_b);
        let mut ba = token_b.clone();
        ba.extend_from_slice(&token_a);

        let out_ab = mla.forward_sequence(&ab, 2).expect("forward ab");
        let out_ba = mla.forward_sequence(&ba, 2).expect("forward ba");
        let diff = out_ab[hidden..]
            .iter()
            .zip(out_ba[hidden..].iter())
            .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
        assert!(diff > 1e-6, "positional information must reach the scores");
    }

    /// Reference math: a two-token sequence computed by hand from the module's
    /// own projections must match `forward_sequence`.
    #[test]
    fn test_mla_matches_hand_computed_two_token_attention() {
        let mut cfg = test_cfg();
        cfg.rope_head_dim = 0; // isolate the content path for an exact reference
        let mla = DeepSeekMlaAttention::new(&cfg);

        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let v_head_dim = cfg.v_head_dim;
        let num_heads = cfg.num_attention_heads;
        let x = lcg_vec(2 * hidden, 505);
        let got = mla.forward_sequence(&x, 2).expect("forward");

        // Reference: project both tokens, then attend for position 1.
        let project = |token: &[f32]| {
            let c_kv = matmul_vec(&mla.w_dkv, token);
            (
                matmul_vec(&mla.w_uk, &c_kv),
                matmul_vec(&mla.w_uv, &c_kv),
                matmul_vec(&mla.w_q, token),
            )
        };
        let (k0, v0, _q0) = project(&x[..hidden]);
        let (k1, v1, q1) = project(&x[hidden..]);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut context = vec![0.0f32; num_heads * v_head_dim];
        for head in 0..num_heads {
            let dot = |k: &[f32]| {
                (0..head_dim)
                    .map(|d| q1[head * head_dim + d] * k[head * head_dim + d])
                    .sum::<f32>()
                    * scale
            };
            let s0 = dot(&k0);
            let s1 = dot(&k1);
            let max = s0.max(s1);
            let e0 = (s0 - max).exp();
            let e1 = (s1 - max).exp();
            let sum = e0 + e1;
            for d in 0..v_head_dim {
                context[head * v_head_dim + d] =
                    (e0 / sum) * v0[head * v_head_dim + d] + (e1 / sum) * v1[head * v_head_dim + d];
            }
        }
        let expected = matmul_vec(&mla.w_o, &context);

        for (i, (a, b)) in got[hidden..].iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "element {i}: got {a}, expected {b}");
        }
    }

    #[test]
    fn test_mla_output_is_not_constant() {
        let cfg = test_cfg();
        let mla = DeepSeekMlaAttention::new(&cfg);
        let out_a = mla.forward_token(&lcg_vec(cfg.hidden_size, 606));
        let out_b = mla.forward_token(&lcg_vec(cfg.hidden_size, 707));
        let diff = out_a
            .iter()
            .zip(out_b.iter())
            .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
        assert!(
            diff > 1e-6,
            "zero-initialised projections would make every output identical"
        );
    }

    #[test]
    fn test_mla_weight_setters_validate_shapes() {
        let cfg = test_cfg();
        let mut mla = DeepSeekMlaAttention::new(&cfg);
        let good = vec![vec![0.5f64; cfg.hidden_size]; cfg.kv_lora_rank];
        assert!(mla.set_kv_down_proj(good).is_ok());
        let bad = vec![vec![0.5f64; cfg.hidden_size]; cfg.kv_lora_rank + 1];
        assert!(
            mla.set_kv_down_proj(bad).is_err(),
            "a mis-shaped weight must be rejected"
        );
    }

    #[test]
    fn test_mla_forward_sequence_rejects_bad_length() {
        let cfg = test_cfg();
        let mla = DeepSeekMlaAttention::new(&cfg);
        assert!(mla.forward_sequence(&[0.0; 3], 2).is_err());
    }

    #[test]
    fn test_rms_norm_normalises_each_row() {
        let norm = DeepSeekRmsNorm::new(4, 1e-6);
        let input = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0, 50.0, 50.0, 50.0, 50.0], &[2, 4])
            .expect("tensor");
        let out = norm.forward(input).expect("forward").data().expect("data");
        for value in out {
            assert!(
                (value - 1.0).abs() < 1e-3,
                "each row must be normalised independently, got {value}"
            );
        }
    }

    #[test]
    fn test_mla_attention_parameter_count_positive() {
        let cfg = test_cfg();
        let mla = DeepSeekMlaAttention::new(&cfg);
        assert!(
            mla.parameter_count() > 0,
            "MLA parameter count should be positive"
        );
    }

    // ── DeepSeekDecoderLayer tests ────────────────────────────────────────

    #[test]
    fn test_decoder_layer_dense_is_not_moe() {
        let cfg = test_cfg();
        let layer = DeepSeekDecoderLayer::new(&cfg, 0).expect("should create decoder layer 0");
        assert!(!layer.is_moe(), "layer 0 should be dense (in dense prefix)");
    }

    #[test]
    fn test_decoder_layer_moe_layer() {
        let cfg = test_cfg();
        let layer = DeepSeekDecoderLayer::new(&cfg, 1).expect("should create decoder layer 1");
        assert!(
            layer.is_moe(),
            "layer 1 should be MoE with moe_layer_freq=1"
        );
    }

    #[test]
    fn test_decoder_layer_forward_output_shape() {
        let cfg = test_cfg();
        let layer = DeepSeekDecoderLayer::new(&cfg, 0).expect("should create decoder layer");
        let data = lcg_vec(cfg.hidden_size, 17);
        let input = Tensor::from_vec(data, &[1, 1, cfg.hidden_size]).expect("build tensor");
        let output = layer.forward(input).expect("decoder layer forward should succeed");
        let shape = output.shape();
        assert_eq!(
            shape[2], cfg.hidden_size,
            "decoder layer preserves hidden_size"
        );
    }

    // ── DeepSeekModel tests ───────────────────────────────────────────────

    #[test]
    fn test_deepseek_model_creation_and_parameter_count() {
        let cfg = test_cfg();
        let model = DeepSeekModel::new(cfg).expect("should create DeepSeekModel");
        assert!(
            model.parameter_count() > 0,
            "DeepSeekModel should have positive parameter count"
        );
    }

    #[test]
    fn test_deepseek_model_run_output_shape() {
        let cfg = test_cfg();
        let seq_len = 3usize;
        let model = DeepSeekModel::new(cfg.clone()).expect("should create DeepSeekModel");
        let input_ids: Vec<u32> = (0..seq_len as u32).collect();
        let output = model.run(input_ids).expect("DeepSeekModel run should succeed");
        let shape = output.shape();
        assert_eq!(
            shape[1], seq_len,
            "output seq dim should match input length"
        );
        assert_eq!(
            shape[2], cfg.hidden_size,
            "output hidden dim should match config"
        );
    }

    // ── DeepSeekForCausalLM tests ─────────────────────────────────────────

    #[test]
    fn test_causal_lm_forward_output_shape() {
        let cfg = test_cfg();
        let seq_len = 2usize;
        let lm = DeepSeekForCausalLM::new(cfg.clone()).expect("should create DeepSeekForCausalLM");
        let input_ids: Vec<u32> = (0..seq_len as u32).collect();
        let output = lm.forward(input_ids).expect("CausalLM forward should succeed");
        let shape = output.shape();
        assert_eq!(
            shape[shape.len() - 1],
            cfg.vocab_size,
            "logits last dim should equal vocab_size"
        );
    }

    #[test]
    fn test_causal_lm_parameter_count_exceeds_model() {
        let cfg = test_cfg();
        let lm = DeepSeekForCausalLM::new(cfg.clone()).expect("should create DeepSeekForCausalLM");
        // lm_head adds hidden_size * vocab_size params
        let lm_head_extra = cfg.hidden_size * cfg.vocab_size;
        assert!(
            lm.parameter_count() > lm_head_extra,
            "CausalLM total params should exceed lm_head alone"
        );
    }
}
