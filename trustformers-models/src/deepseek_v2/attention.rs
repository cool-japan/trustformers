//! Multi-head Latent Attention (MLA) and its two normalisation building blocks.
//!
//! The structure here follows DeepSeek-V2's reference implementation
//! (`modeling_deepseek.py`, class `DeepseekV2Attention`) rather than a
//! convenience decomposition, because a checkpoint's tensors have to land
//! somewhere:
//!
//! * the query path is `q_proj` when `q_lora_rank == 0`, and
//!   `q_a_proj → q_a_layernorm → q_b_proj` otherwise;
//! * the key/value path is a single fused `kv_a_proj_with_mqa` producing
//!   `kv_lora_rank + qk_rope_head_dim` values per token, split into the
//!   compressed latent `c_KV` and a head-shared RoPE key slice `k_pe`; the
//!   latent is normalised by `kv_a_layernorm` and expanded by `kv_b_proj` into
//!   `num_heads × (qk_nope_head_dim + v_head_dim)`.
//!
//! A previous revision stored `c_kv` / `k_pe` / `k_nope` / `v_proj` as four
//! independent projections and had no latent norms at all, so no DeepSeek-V2
//! export could be bound without dropping weights — and its `forward` discarded
//! the keys and values outright, resizing the query buffer into the attended
//! shape. Both are gone: the projections match the checkpoint and the attention
//! is a real causal scaled-dot-product attention.

use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result},
    layers::Linear,
    tensor::Tensor,
    traits::Layer,
};

use super::config::DeepSeekV2Config;

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// DeepSeek-V2 RMSNorm layer.
///
/// `output[t, i] = weight[i] * input[t, i] / sqrt(mean_i(input[t, i]²) + eps)`
///
/// The statistic is taken over the **last** axis, independently per row, which
/// is what `DeepseekV2RMSNorm` does and what makes the norm meaningful on a
/// compressed latent: `q_a_layernorm` normalises over `q_lora_rank` and
/// `kv_a_layernorm` over `kv_lora_rank`, not over the whole activation buffer.
/// A previous revision averaged over every element of the tensor at once, so the
/// result depended on the sequence length.
pub struct DeepSeekV2RmsNorm {
    weight: Tensor,
    eps: f32,
    device: Device,
}

impl DeepSeekV2RmsNorm {
    /// Build a norm over `size` trailing elements, initialised to unit weight.
    ///
    /// # Errors
    ///
    /// Fails when the weight tensor cannot be allocated.
    pub fn new(size: usize, eps: f64, device: Device) -> Result<Self> {
        let weight = Tensor::ones(&[size])?;
        Ok(Self {
            weight,
            eps: eps as f32,
            device,
        })
    }

    /// The device this norm reports.
    pub fn device(&self) -> Device {
        self.device
    }

    /// The learnable scale.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Install a checkpoint's scale.
    ///
    /// # Errors
    ///
    /// Fails when the tensor's element count differs from the width this norm
    /// was built for — silently accepting a different width would make the
    /// normalisation quietly wrong instead of loudly refused.
    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        let expected = self.weight.len();
        if weight.len() != expected {
            return Err(tensor_op_error(
                "deepseek_v2_rmsnorm",
                format!(
                    "weight has {} element(s) but this norm is {expected} wide",
                    weight.len()
                ),
            ));
        }
        self.weight = weight;
        Ok(())
    }

    /// Number of learnable parameters.
    pub fn parameter_count(&self) -> usize {
        self.weight.len()
    }
}

impl Layer for DeepSeekV2RmsNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let (data, shape) = tensor_parts("deepseek_v2_rmsnorm", &input)?;
        let width = shape.last().copied().unwrap_or(0);
        let weight = self.weight.to_vec_f32()?;
        if width == 0 || weight.len() != width {
            return Err(tensor_op_error(
                "deepseek_v2_rmsnorm",
                format!(
                    "input's last dimension is {width} but the norm weight has {} element(s)",
                    weight.len()
                ),
            ));
        }
        let mut out = vec![0.0_f32; data.len()];
        for (row_in, row_out) in data.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
            let mean_sq = row_in.iter().map(|x| x * x).sum::<f32>() / width as f32;
            let inv_rms = 1.0 / (mean_sq + self.eps).sqrt();
            for ((slot, value), scale) in row_out.iter_mut().zip(row_in.iter()).zip(weight.iter()) {
                *slot = value * inv_rms * scale;
            }
        }
        Tensor::from_vec(out, &shape)
    }
}

// ---------------------------------------------------------------------------
// Rotary Position Embedding (applied to the rope portion only)
// ---------------------------------------------------------------------------

/// RoPE applied to the `qk_rope_head_dim`-dimensional slice of Q and K.
///
/// The half-split convention is used throughout this crate: element `i` pairs
/// with element `i + rope_head_dim / 2`.
pub struct DeepSeekV2RotaryEmbedding {
    /// Dimension of the RoPE slice (= `qk_rope_head_dim`).
    rope_head_dim: usize,
    rope_theta: f64,
    #[allow(dead_code)]
    device: Device,
}

impl DeepSeekV2RotaryEmbedding {
    /// Build the rotary table description for a config.
    pub fn new(config: &DeepSeekV2Config, device: Device) -> Self {
        Self {
            rope_head_dim: config.qk_rope_head_dim,
            rope_theta: config.rope_theta,
            device,
        }
    }

    /// The RoPE base frequency.
    pub fn rope_theta(&self) -> f64 {
        self.rope_theta
    }

    /// Apply RoPE in-place to a flat slice of length `seq_len * rope_head_dim`.
    pub fn apply(&self, data: &mut [f32], seq_len: usize) {
        self.apply_from(data, seq_len, 0);
    }

    /// Apply RoPE in-place, treating the first row as absolute position
    /// `start_position`.
    pub fn apply_from(&self, data: &mut [f32], seq_len: usize, start_position: usize) {
        let half = self.rope_head_dim / 2;
        if half == 0 {
            return;
        }
        for pos in 0..seq_len {
            let base = pos * self.rope_head_dim;
            if base + self.rope_head_dim > data.len() {
                break;
            }
            let absolute = (start_position + pos) as f64;
            for i in 0..half {
                let freq = 1.0 / self.rope_theta.powf(2.0 * i as f64 / self.rope_head_dim as f64);
                let angle = (absolute * freq) as f32;
                let (sin_v, cos_v) = angle.sin_cos();
                let x0 = data[base + i];
                let x1 = data[base + i + half];
                data[base + i] = x0 * cos_v - x1 * sin_v;
                data[base + i + half] = x0 * sin_v + x1 * cos_v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-head Latent Attention (MLA)
// ---------------------------------------------------------------------------

/// Multi-head Latent Attention as introduced in DeepSeek-V2.
///
/// ## Key idea
///
/// Instead of projecting hidden states separately into full K and V tensors
/// (which become large for many heads), MLA first compresses them jointly into a
/// low-rank *latent* vector `c_KV` of dimension `kv_lora_rank`. K and V are then
/// expanded from this latent on the fly. At inference only `c_KV` (plus a small
/// head-shared RoPE key slice) needs to be cached.
///
/// ### Projections
///
/// | Weight | Shape | Present when |
/// |--------|-------|--------------|
/// | `q_proj` | `[num_heads * qk_head_dim, hidden_size]` | `q_lora_rank == 0` |
/// | `q_a_proj` | `[q_lora_rank, hidden_size]` | `q_lora_rank > 0` |
/// | `q_a_layernorm` | `[q_lora_rank]` | `q_lora_rank > 0` |
/// | `q_b_proj` | `[num_heads * qk_head_dim, q_lora_rank]` | `q_lora_rank > 0` |
/// | `kv_a_proj_with_mqa` | `[kv_lora_rank + qk_rope_head_dim, hidden_size]` | always |
/// | `kv_a_layernorm` | `[kv_lora_rank]` | always |
/// | `kv_b_proj` | `[num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]` | always |
/// | `o_proj` | `[hidden_size, num_heads * v_head_dim]` | always |
///
/// where `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim`.
pub struct MlaAttention {
    /// Single-step query projection, used when `q_lora_rank == 0`.
    q_proj: Option<Linear>,
    /// Query down-projection: `hidden_size → q_lora_rank`.
    q_a_proj: Option<Linear>,
    /// RMS norm on the compressed query latent.
    q_a_layernorm: Option<DeepSeekV2RmsNorm>,
    /// Query up-projection: `q_lora_rank → num_heads * qk_head_dim`.
    q_b_proj: Option<Linear>,
    /// Fused KV compression: `hidden_size → kv_lora_rank + qk_rope_head_dim`.
    kv_a_proj_with_mqa: Linear,
    /// RMS norm on the compressed KV latent.
    kv_a_layernorm: DeepSeekV2RmsNorm,
    /// KV expansion: `kv_lora_rank → num_heads * (qk_nope_head_dim + v_head_dim)`.
    kv_b_proj: Linear,
    /// Output projection: `num_heads * v_head_dim → hidden_size`.
    o_proj: Linear,
    rotary_emb: DeepSeekV2RotaryEmbedding,
    hidden_size: usize,
    num_heads: usize,
    qk_rope_head_dim: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    device: Device,
}

impl MlaAttention {
    /// Build the attention block for a config.
    ///
    /// # Errors
    ///
    /// Fails when a norm's weight tensor cannot be allocated.
    pub fn new(config: &DeepSeekV2Config, device: Device) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let kv_lora_rank = config.kv_lora_rank;
        let q_lora_rank = config.q_lora_rank;
        let rope_dim = config.qk_rope_head_dim;
        let nope_dim = config.qk_nope_head_dim;
        let v_head_dim = config.v_head_dim;
        let qk_head_dim = rope_dim + nope_dim;

        let (q_proj, q_a_proj, q_a_layernorm, q_b_proj) = if q_lora_rank == 0 {
            (
                Some(Linear::new_with_device(
                    hidden_size,
                    num_heads * qk_head_dim,
                    false,
                    device,
                )),
                None,
                None,
                None,
            )
        } else {
            (
                None,
                Some(Linear::new_with_device(
                    hidden_size,
                    q_lora_rank,
                    false,
                    device,
                )),
                Some(DeepSeekV2RmsNorm::new(
                    q_lora_rank,
                    config.rms_norm_eps,
                    device,
                )?),
                Some(Linear::new_with_device(
                    q_lora_rank,
                    num_heads * qk_head_dim,
                    false,
                    device,
                )),
            )
        };

        Ok(Self {
            q_proj,
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj_with_mqa: Linear::new_with_device(
                hidden_size,
                kv_lora_rank + rope_dim,
                false,
                device,
            ),
            kv_a_layernorm: DeepSeekV2RmsNorm::new(kv_lora_rank, config.rms_norm_eps, device)?,
            kv_b_proj: Linear::new_with_device(
                kv_lora_rank,
                num_heads * (nope_dim + v_head_dim),
                false,
                device,
            ),
            o_proj: Linear::new_with_device(num_heads * v_head_dim, hidden_size, false, device),
            rotary_emb: DeepSeekV2RotaryEmbedding::new(config, device),
            hidden_size,
            num_heads,
            qk_rope_head_dim: rope_dim,
            qk_nope_head_dim: nope_dim,
            v_head_dim,
            kv_lora_rank,
            device,
        })
    }

    /// The device this block reports.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// KV lora rank — dimension of the compressed latent KV vector.
    pub fn kv_lora_rank(&self) -> usize {
        self.kv_lora_rank
    }

    /// Total per-head query/key width, `qk_nope_head_dim + qk_rope_head_dim`.
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Whether the query path is compressed through `q_a_proj`/`q_b_proj`.
    pub fn uses_query_lora(&self) -> bool {
        self.q_a_proj.is_some()
    }

    /// Total learnable parameters in this block.
    pub fn parameter_count(&self) -> usize {
        let mut total = self.kv_a_proj_with_mqa.parameter_count()
            + self.kv_a_layernorm.parameter_count()
            + self.kv_b_proj.parameter_count()
            + self.o_proj.parameter_count();
        if let Some(layer) = &self.q_proj {
            total += layer.parameter_count();
        }
        if let Some(layer) = &self.q_a_proj {
            total += layer.parameter_count();
        }
        if let Some(norm) = &self.q_a_layernorm {
            total += norm.parameter_count();
        }
        if let Some(layer) = &self.q_b_proj {
            total += layer.parameter_count();
        }
        total
    }

    // --- mutable access for the checkpoint binder -------------------------

    pub(super) fn q_proj_mut(&mut self) -> Option<&mut Linear> {
        self.q_proj.as_mut()
    }

    pub(super) fn q_a_proj_mut(&mut self) -> Option<&mut Linear> {
        self.q_a_proj.as_mut()
    }

    pub(super) fn q_a_layernorm_mut(&mut self) -> Option<&mut DeepSeekV2RmsNorm> {
        self.q_a_layernorm.as_mut()
    }

    pub(super) fn q_b_proj_mut(&mut self) -> Option<&mut Linear> {
        self.q_b_proj.as_mut()
    }

    pub(super) fn kv_a_proj_with_mqa_mut(&mut self) -> &mut Linear {
        &mut self.kv_a_proj_with_mqa
    }

    pub(super) fn kv_a_layernorm_mut(&mut self) -> &mut DeepSeekV2RmsNorm {
        &mut self.kv_a_layernorm
    }

    pub(super) fn kv_b_proj_mut(&mut self) -> &mut Linear {
        &mut self.kv_b_proj
    }

    pub(super) fn o_proj_mut(&mut self) -> &mut Linear {
        &mut self.o_proj
    }

    /// Query projection for a `[tokens, hidden_size]` input, as
    /// `[tokens, num_heads * qk_head_dim]`.
    fn project_queries(&self, input: &Tensor) -> Result<Tensor> {
        match (
            &self.q_proj,
            &self.q_a_proj,
            &self.q_a_layernorm,
            &self.q_b_proj,
        ) {
            (Some(q_proj), ..) => q_proj.forward(input.clone()),
            (None, Some(down), Some(norm), Some(up)) => {
                let latent = down.forward(input.clone())?;
                let normed = norm.forward(latent)?;
                up.forward(normed)
            },
            _ => Err(tensor_op_error(
                "deepseek_v2_mla",
                "the query path is neither a plain projection nor a complete LoRA pair",
            )),
        }
    }
}

/// Flatten a tensor into contiguous `f32` values plus its shape.
fn tensor_parts(op: &'static str, tensor: &Tensor) -> Result<(Vec<f32>, Vec<usize>)> {
    match tensor {
        Tensor::F32(arr) => {
            let contiguous = arr.as_standard_layout().to_owned();
            let shape = contiguous.shape().to_vec();
            let values = match contiguous.as_slice() {
                Some(slice) => slice.to_vec(),
                None => contiguous.iter().copied().collect(),
            };
            Ok((values, shape))
        },
        _ => Err(tensor_op_error(op, "tensor must be F32")),
    }
}

impl Layer for MlaAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let (_, input_shape) = tensor_parts("deepseek_v2_mla", &input)?;
        let width = input_shape.last().copied().unwrap_or(0);
        if width != self.hidden_size {
            return Err(tensor_op_error(
                "deepseek_v2_mla",
                format!(
                    "input's last dimension is {width}, expected hidden_size {}",
                    self.hidden_size
                ),
            ));
        }
        let seq_len: usize = input_shape.iter().take(input_shape.len() - 1).product();
        if seq_len == 0 {
            return Ok(input);
        }

        let heads = self.num_heads;
        let nope = self.qk_nope_head_dim;
        let rope = self.qk_rope_head_dim;
        let qk = nope + rope;
        let v_dim = self.v_head_dim;

        // --- Query path ---------------------------------------------------
        let (q_flat, _) = tensor_parts("deepseek_v2_mla", &self.project_queries(&input)?)?;
        if q_flat.len() != seq_len * heads * qk {
            return Err(tensor_op_error(
                "deepseek_v2_mla",
                format!(
                    "query projection produced {} value(s), expected {}",
                    q_flat.len(),
                    seq_len * heads * qk
                ),
            ));
        }
        // Per head: [seq_len, qk_head_dim], RoPE applied to the trailing slice.
        let mut queries = vec![0.0_f32; seq_len * heads * qk];
        for head in 0..heads {
            let mut rope_slice = vec![0.0_f32; seq_len * rope];
            for token in 0..seq_len {
                let src = token * heads * qk + head * qk;
                let dst = head * seq_len * qk + token * qk;
                queries[dst..dst + nope].copy_from_slice(&q_flat[src..src + nope]);
                rope_slice[token * rope..(token + 1) * rope]
                    .copy_from_slice(&q_flat[src + nope..src + qk]);
            }
            self.rotary_emb.apply(&mut rope_slice, seq_len);
            for token in 0..seq_len {
                let dst = head * seq_len * qk + token * qk + nope;
                queries[dst..dst + rope]
                    .copy_from_slice(&rope_slice[token * rope..(token + 1) * rope]);
            }
        }

        // --- Key/value path ------------------------------------------------
        let compressed = self.kv_a_proj_with_mqa.forward(input)?;
        let (compressed_flat, _) = tensor_parts("deepseek_v2_mla", &compressed)?;
        let fused_width = self.kv_lora_rank + rope;
        if compressed_flat.len() != seq_len * fused_width {
            return Err(tensor_op_error(
                "deepseek_v2_mla",
                format!(
                    "kv_a_proj_with_mqa produced {} value(s), expected {}",
                    compressed_flat.len(),
                    seq_len * fused_width
                ),
            ));
        }
        let mut latent = vec![0.0_f32; seq_len * self.kv_lora_rank];
        let mut key_rope = vec![0.0_f32; seq_len * rope];
        for token in 0..seq_len {
            let src = token * fused_width;
            latent[token * self.kv_lora_rank..(token + 1) * self.kv_lora_rank]
                .copy_from_slice(&compressed_flat[src..src + self.kv_lora_rank]);
            key_rope[token * rope..(token + 1) * rope]
                .copy_from_slice(&compressed_flat[src + self.kv_lora_rank..src + fused_width]);
        }
        self.rotary_emb.apply(&mut key_rope, seq_len);

        let latent_tensor = Tensor::from_vec(latent, &[seq_len, self.kv_lora_rank])?;
        let normed_latent = self.kv_a_layernorm.forward(latent_tensor)?;
        let expanded = self.kv_b_proj.forward(normed_latent)?;
        let (kv_flat, _) = tensor_parts("deepseek_v2_mla", &expanded)?;
        let kv_head_width = nope + v_dim;
        if kv_flat.len() != seq_len * heads * kv_head_width {
            return Err(tensor_op_error(
                "deepseek_v2_mla",
                format!(
                    "kv_b_proj produced {} value(s), expected {}",
                    kv_flat.len(),
                    seq_len * heads * kv_head_width
                ),
            ));
        }

        // --- Causal scaled dot-product attention ----------------------------
        let scale = 1.0 / (qk as f32).sqrt();
        let mut attended = vec![0.0_f32; seq_len * heads * v_dim];
        let mut scores = vec![0.0_f32; seq_len];
        for head in 0..heads {
            for query_pos in 0..seq_len {
                let q_base = head * seq_len * qk + query_pos * qk;
                let mut max_score = f32::NEG_INFINITY;
                for key_pos in 0..=query_pos {
                    let kv_base = key_pos * heads * kv_head_width + head * kv_head_width;
                    let mut dot = 0.0_f32;
                    for i in 0..nope {
                        dot += queries[q_base + i] * kv_flat[kv_base + i];
                    }
                    for i in 0..rope {
                        dot += queries[q_base + nope + i] * key_rope[key_pos * rope + i];
                    }
                    let score = dot * scale;
                    scores[key_pos] = score;
                    if score > max_score {
                        max_score = score;
                    }
                }
                let mut denominator = 0.0_f32;
                for score in scores.iter_mut().take(query_pos + 1) {
                    *score = (*score - max_score).exp();
                    denominator += *score;
                }
                let inv = if denominator > 0.0 { 1.0 / denominator } else { 0.0 };
                let out_base = query_pos * heads * v_dim + head * v_dim;
                for key_pos in 0..=query_pos {
                    let weight = scores[key_pos] * inv;
                    let value_base = key_pos * heads * kv_head_width + head * kv_head_width + nope;
                    for i in 0..v_dim {
                        attended[out_base + i] += weight * kv_flat[value_base + i];
                    }
                }
            }
        }

        let attended_tensor = Tensor::from_vec(attended, &[seq_len, heads * v_dim])?;
        let projected = self.o_proj.forward(attended_tensor)?;
        if input_shape.len() == 2 {
            return Ok(projected);
        }
        let mut output_shape = input_shape;
        if let Some(last) = output_shape.last_mut() {
            *last = self.hidden_size;
        }
        projected.reshape(&output_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek_v2::config::{ActivationType, TopKMethod};

    fn tiny_config() -> DeepSeekV2Config {
        DeepSeekV2Config {
            vocab_size: 24,
            hidden_size: 16,
            intermediate_size: 12,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            kv_lora_rank: 6,
            q_lora_rank: 8,
            qk_rope_head_dim: 4,
            qk_nope_head_dim: 4,
            v_head_dim: 4,
            num_experts_per_tok: 2,
            n_routed_experts: 3,
            n_shared_experts: 2,
            routed_scaling_factor: 1.0,
            topk_method: TopKMethod::Noaux,
            n_group: 1,
            topk_group: 3,
            aux_loss_alpha: 0.001,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            hidden_act: ActivationType::SiLU,
            initializer_range: 0.02,
            first_k_dense_replace: 1,
            moe_layer_freq: 1,
        }
    }

    /// Deterministic small weights, so a forward pass is reproducible and tame.
    fn fill(layer: &mut Linear, seed: f32) {
        let shape = layer.weight().shape();
        let count: usize = shape.iter().product();
        let values: Vec<f32> =
            (0..count).map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.25).collect();
        layer
            .set_weight(Tensor::from_vec(values, &shape).expect("weight tensor must build"))
            .expect("weight must install");
    }

    fn deterministic_attention(config: &DeepSeekV2Config) -> MlaAttention {
        let mut attention = MlaAttention::new(config, Device::CPU).expect("attention must build");
        let mut seed = 0.0_f32;
        let mut next = || {
            seed += 1.31;
            seed
        };
        if let Some(layer) = attention.q_proj_mut() {
            fill(layer, next());
        }
        if let Some(layer) = attention.q_a_proj_mut() {
            fill(layer, next());
        }
        if let Some(layer) = attention.q_b_proj_mut() {
            fill(layer, next());
        }
        let value = next();
        fill(attention.kv_a_proj_with_mqa_mut(), value);
        let value = next();
        fill(attention.kv_b_proj_mut(), value);
        let value = next();
        fill(attention.o_proj_mut(), value);
        attention
    }

    fn hidden_states(config: &DeepSeekV2Config, seq_len: usize) -> Tensor {
        let values: Vec<f32> = (0..seq_len * config.hidden_size)
            .map(|i| ((i as f32 * 0.19).cos()) * 0.5)
            .collect();
        Tensor::from_vec(values, &[seq_len, config.hidden_size]).expect("input must build")
    }

    /// RMSNorm normalises over the trailing axis, per row.
    ///
    /// The previous implementation averaged `x²` over *every* element of the
    /// tensor at once, so a norm on a compressed latent depended on the sequence
    /// length and two rows could not be normalised independently.
    #[test]
    fn rms_norm_normalises_each_row_over_the_last_dimension() {
        let norm = DeepSeekV2RmsNorm::new(4, 1e-12, Device::CPU).expect("norm must build");
        // Row 0 is 100× the scale of row 1; after RMS normalisation both rows
        // must have unit root-mean-square.
        let input = Tensor::from_vec(
            vec![100.0, 200.0, 300.0, 400.0, 1.0, 2.0, 3.0, 4.0],
            &[2, 4],
        )
        .expect("input must build");
        let output = norm.forward(input).expect("forward must succeed");
        let values = output.to_vec_f32().expect("output must be F32");
        assert_eq!(output.shape(), vec![2, 4]);
        for row in values.chunks_exact(4) {
            let rms = (row.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
            assert!(
                (rms - 1.0).abs() < 1e-4,
                "each row must be normalised on its own, got rms {rms} for {row:?}"
            );
        }
        // Both rows are proportional to (1,2,3,4), so after per-row normalisation
        // they must be equal — which a whole-tensor mean could never produce.
        assert!(
            values[..4].iter().zip(&values[4..]).all(|(a, b)| (a - b).abs() < 1e-4),
            "proportional rows must normalise to the same vector: {values:?}"
        );
    }

    /// The learnable scale is applied element-wise along the normalised axis.
    #[test]
    fn rms_norm_applies_its_weight_along_the_normalised_axis() {
        let mut norm = DeepSeekV2RmsNorm::new(4, 1e-12, Device::CPU).expect("norm must build");
        norm.set_weight(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("weight must build"),
        )
        .expect("weight must install");
        let input = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], &[1, 4]).expect("input must build");
        let values = norm
            .forward(input)
            .expect("forward must succeed")
            .to_vec_f32()
            .expect("output must be F32");
        // Uniform input normalises to 1.0, so the output is the weight itself.
        for (got, want) in values.iter().zip([1.0_f32, 2.0, 3.0, 4.0]) {
            assert!((got - want).abs() < 1e-4, "got {values:?}");
        }
    }

    /// A weight of the wrong width is refused instead of silently broadcast.
    #[test]
    fn rms_norm_refuses_a_weight_of_the_wrong_width() {
        let mut norm = DeepSeekV2RmsNorm::new(4, 1e-6, Device::CPU).expect("norm must build");
        norm.set_weight(Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("tensor must build"))
            .expect_err("a 2-wide weight is not a 4-wide norm");
    }

    /// A small deterministic configuration produces finite, shape-correct output.
    #[test]
    fn mla_forward_is_finite_shape_correct_and_reproducible() {
        let config = tiny_config();
        let attention = deterministic_attention(&config);
        let input = hidden_states(&config, 5);

        let first = attention.forward(input.clone()).expect("forward must succeed");
        let second = attention.forward(input).expect("forward must succeed");

        assert_eq!(
            first.shape(),
            vec![5, config.hidden_size],
            "MLA must project back to hidden_size"
        );
        let values = first.to_vec_f32().expect("output must be F32");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "every output value must be finite"
        );
        assert!(
            values.iter().any(|value| value.abs() > 1e-6),
            "an all-zero attention output would mean nothing was computed"
        );
        assert_eq!(
            values,
            second.to_vec_f32().expect("output must be F32"),
            "the same input must give the same output"
        );
    }

    /// The latent norms are part of the computation, not decoration.
    ///
    /// This is the property that made a real checkpoint binder possible: an
    /// earlier revision had nowhere to put `q_a_layernorm` / `kv_a_layernorm`,
    /// so a checkpoint's latent norms would have been dropped. Scaling either
    /// one must move the output.
    #[test]
    fn the_latent_norms_participate_in_the_forward() {
        let config = tiny_config();
        let input = hidden_states(&config, 4);

        for scale_kv in [false, true] {
            let mut attention = deterministic_attention(&config);
            let before = attention
                .forward(input.clone())
                .expect("forward must succeed")
                .to_vec_f32()
                .expect("output must be F32");

            let (width, norm) = if scale_kv {
                (config.kv_lora_rank, attention.kv_a_layernorm_mut())
            } else {
                (
                    config.q_lora_rank,
                    attention.q_a_layernorm_mut().expect("this config compresses the query path"),
                )
            };
            norm.set_weight(
                Tensor::from_vec(vec![2.5_f32; width], &[width]).expect("weight must build"),
            )
            .expect("weight must install");

            let after = attention
                .forward(input.clone())
                .expect("forward must succeed")
                .to_vec_f32()
                .expect("output must be F32");
            assert_ne!(
                before,
                after,
                "scaling the {} latent norm must change the output",
                if scale_kv { "kv" } else { "q" }
            );
        }
    }

    /// Attention is causal: a later token cannot change an earlier position.
    ///
    /// A previous revision discarded the keys and values entirely and resized the
    /// query buffer into the attended shape, so no position ever attended to
    /// another and this property was vacuous.
    #[test]
    fn mla_attention_is_causal_and_reads_the_keys_and_values() {
        let config = tiny_config();
        let attention = deterministic_attention(&config);
        let base = hidden_states(&config, 4);

        let mut altered = base.to_vec_f32().expect("input must be F32");
        // Perturb the *last* token only.
        for slot in altered.iter_mut().skip(3 * config.hidden_size) {
            *slot += 1.0;
        }
        let altered =
            Tensor::from_vec(altered, &[4, config.hidden_size]).expect("input must build");

        let first = attention
            .forward(base)
            .expect("forward must succeed")
            .to_vec_f32()
            .expect("output must be F32");
        let second = attention
            .forward(altered)
            .expect("forward must succeed")
            .to_vec_f32()
            .expect("output must be F32");

        let prefix = 3 * config.hidden_size;
        assert_eq!(
            first[..prefix],
            second[..prefix],
            "changing the last token must not move earlier positions"
        );
        assert_ne!(
            first[prefix..],
            second[prefix..],
            "the last position must react to its own input"
        );
    }

    /// Earlier tokens do reach later positions — the keys and values are read.
    #[test]
    fn a_later_position_depends_on_earlier_tokens() {
        let config = tiny_config();
        let attention = deterministic_attention(&config);
        let base = hidden_states(&config, 4);

        let mut altered = base.to_vec_f32().expect("input must be F32");
        for slot in altered.iter_mut().take(config.hidden_size) {
            *slot += 1.0;
        }
        let altered =
            Tensor::from_vec(altered, &[4, config.hidden_size]).expect("input must build");

        let first = attention
            .forward(base)
            .expect("forward must succeed")
            .to_vec_f32()
            .expect("output must be F32");
        let second = attention
            .forward(altered)
            .expect("forward must succeed")
            .to_vec_f32()
            .expect("output must be F32");
        let tail = 3 * config.hidden_size;
        assert_ne!(
            first[tail..],
            second[tail..],
            "the last position must see the first token through the KV path"
        );
    }

    /// `q_lora_rank == 0` selects the single-step query projection and drops the
    /// query latent norm from the structure entirely.
    #[test]
    fn a_zero_query_lora_rank_uses_a_plain_query_projection() {
        let mut config = tiny_config();
        config.q_lora_rank = 0;
        let mut attention = MlaAttention::new(&config, Device::CPU).expect("must build");
        assert!(!attention.uses_query_lora());
        assert!(attention.q_a_layernorm_mut().is_none());
        assert!(attention.q_proj_mut().is_some());

        let attention = deterministic_attention(&config);
        let output = attention.forward(hidden_states(&config, 3)).expect("forward must succeed");
        assert_eq!(output.shape(), vec![3, config.hidden_size]);
        assert!(output
            .to_vec_f32()
            .expect("output must be F32")
            .iter()
            .all(|value| value.is_finite()));
    }

    /// An input whose width is not `hidden_size` is refused rather than reshaped.
    #[test]
    fn mla_refuses_an_input_of_the_wrong_width() {
        let config = tiny_config();
        let attention = deterministic_attention(&config);
        let input = Tensor::from_vec(vec![0.5_f32; 8], &[2, 4]).expect("input must build");
        attention
            .forward(input)
            .expect_err("a 4-wide input is not a hidden state of this model");
    }
}
