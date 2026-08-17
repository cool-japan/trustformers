use crate::llama3_2::config::Llama32Config;
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result},
    layers::{Embedding, Linear},
    ops::activations::{gelu, silu},
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm (shared by both text and vision components)
// ─────────────────────────────────────────────────────────────────────────────

/// Root Mean Square Layer Normalisation
///
/// `RMSNorm(x) = x / RMS(x) * weight`
pub struct Llama32RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Llama32RmsNorm {
    pub fn new(normalized_shape: usize, eps: f64) -> Result<Self> {
        let weight = Tensor::ones(&[normalized_shape])?;
        Ok(Self { weight, eps })
    }

    pub fn parameter_count(&self) -> usize {
        self.weight.len()
    }
}

impl Layer for Llama32RmsNorm {
    type Input = Tensor;
    type Output = Tensor;

    /// Normalise every trailing `normalized_shape`-sized vector independently.
    ///
    /// Pooling the mean square over the whole tensor would make one token's
    /// scale depend on its neighbours, which is not RMSNorm.
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match (&input, &self.weight) {
            (Tensor::F32(arr), Tensor::F32(w)) => {
                let eps_f32 = self.eps as f32;
                let size = w.len();
                if size == 0 || !arr.len().is_multiple_of(size) {
                    return Err(tensor_op_error(
                        "Llama32RmsNorm::forward",
                        format!(
                            "tensor of {} elements is not a multiple of the norm size {size}",
                            arr.len()
                        ),
                    ));
                }
                let weight: Vec<f32> = w.iter().copied().collect();
                let values: Vec<f32> = arr.iter().copied().collect();
                let mut data = Vec::with_capacity(values.len());
                for chunk in values.chunks(size) {
                    let mean_sq = chunk.iter().map(|x| x * x).sum::<f32>() / size as f32;
                    let inv_rms = 1.0 / (mean_sq + eps_f32).sqrt();
                    for (value, scale) in chunk.iter().zip(weight.iter()) {
                        data.push(value * inv_rms * scale);
                    }
                }
                let out = ArrayD::from_shape_vec(IxDyn(arr.shape()), data).map_err(|e| {
                    tensor_op_error("Llama32RmsNorm::forward", format!("shape error: {e}"))
                })?;
                Ok(Tensor::F32(out))
            },
            _ => Err(tensor_op_error(
                "Llama32RmsNorm::forward",
                "unsupported input tensor dtype",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared scaled dot-product attention
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-head scaled dot-product attention over flat `[seq, num_heads * head_dim]`
/// buffers.
///
/// `queries` has `q_len` rows, `keys`/`values` have `kv_len` rows (they may come
/// from a different modality, which is exactly what cross-attention needs). When
/// `causal` is set, query `i` only sees keys `0..=i`.
///
/// Returns `[q_len, num_heads * head_dim]`.
pub(crate) fn multi_head_sdpa(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    q_len: usize,
    kv_len: usize,
    num_heads: usize,
    head_dim: usize,
    causal: bool,
) -> Result<Vec<f32>> {
    let width = num_heads * head_dim;
    if queries.len() != q_len * width
        || keys.len() != kv_len * width
        || values.len() != kv_len * width
    {
        return Err(tensor_op_error(
            "multi_head_sdpa",
            format!(
                "shape mismatch: q {} (expected {}), k {} / v {} (expected {})",
                queries.len(),
                q_len * width,
                keys.len(),
                values.len(),
                kv_len * width
            ),
        ));
    }
    if q_len == 0 {
        // Attention over an empty query sequence is vacuously empty (a tiny image
        // can yield zero patches).
        return Ok(Vec::new());
    }
    if kv_len == 0 {
        return Err(tensor_op_error(
            "multi_head_sdpa",
            "attention needs at least one key/value position".to_string(),
        ));
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; q_len * width];
    let mut scores = vec![0.0f32; kv_len];

    for head in 0..num_heads {
        for query_pos in 0..q_len {
            let visible = if causal { (query_pos + 1).min(kv_len) } else { kv_len };
            let q_base = query_pos * width + head * head_dim;

            let mut max_score = f32::NEG_INFINITY;
            for (key_pos, score) in scores.iter_mut().take(visible).enumerate() {
                let k_base = key_pos * width + head * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += queries[q_base + d] * keys[k_base + d];
                }
                dot *= scale;
                *score = dot;
                if dot > max_score {
                    max_score = dot;
                }
            }

            let mut sum = 0.0f32;
            for score in scores.iter_mut().take(visible) {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };

            for (key_pos, score) in scores.iter().take(visible).enumerate() {
                let weight = score * inv_sum;
                let v_base = key_pos * width + head * head_dim;
                for d in 0..head_dim {
                    output[q_base + d] += weight * values[v_base + d];
                }
            }
        }
    }

    Ok(output)
}

/// Split a 2-D `[seq, features]` or 3-D `[batch, seq, features]` shape.
fn split_sequence_shape(shape: &[usize], context: &str) -> Result<(usize, usize, usize)> {
    match shape.len() {
        2 => Ok((1, shape[0], shape[1])),
        3 => Ok((shape[0], shape[1], shape[2])),
        _ => Err(tensor_op_error(
            context,
            format!("expected [seq, features] or [batch, seq, features], got {shape:?}"),
        )),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LayerNorm (used in vision encoder, follows CLIP ViT convention)
// ─────────────────────────────────────────────────────────────────────────────

/// Standard Layer Normalisation for the vision encoder
pub struct VisionLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl VisionLayerNorm {
    pub fn new(normalized_shape: usize, eps: f64) -> Result<Self> {
        let weight = Tensor::ones(&[normalized_shape])?;
        let bias = Tensor::zeros(&[normalized_shape])?;
        Ok(Self { weight, bias, eps })
    }

    pub fn parameter_count(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
}

impl Layer for VisionLayerNorm {
    type Input = Tensor;
    type Output = Tensor;

    /// Normalise every trailing `normalized_shape`-sized vector independently.
    ///
    /// LayerNorm is defined per token: pooling the mean and variance across the
    /// whole tensor would make one patch's activation depend on the other patches
    /// in the batch, which is not LayerNorm and leaks information across the
    /// sequence.
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match (&input, &self.weight, &self.bias) {
            (Tensor::F32(arr), Tensor::F32(w), Tensor::F32(b)) => {
                let size = w.len();
                if size == 0 || !arr.len().is_multiple_of(size) {
                    return Err(tensor_op_error(
                        "VisionLayerNorm::forward",
                        format!(
                            "tensor of {} elements is not a multiple of the norm size {size}",
                            arr.len()
                        ),
                    ));
                }
                let weight: Vec<f32> = w.iter().copied().collect();
                let bias: Vec<f32> = b.iter().copied().collect();
                let values: Vec<f32> = arr.iter().copied().collect();
                let mut data = Vec::with_capacity(values.len());
                let n = size as f32;
                for chunk in values.chunks(size) {
                    let mean = chunk.iter().sum::<f32>() / n;
                    let var = chunk.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
                    let inv_std = 1.0 / (var + self.eps as f32).sqrt();
                    for ((value, scale), shift) in chunk.iter().zip(weight.iter()).zip(bias.iter())
                    {
                        data.push((value - mean) * inv_std * scale + shift);
                    }
                }
                let out = ArrayD::from_shape_vec(IxDyn(arr.shape()), data).map_err(|e| {
                    tensor_op_error("VisionLayerNorm::forward", format!("shape error: {e}"))
                })?;
                Ok(Tensor::F32(out))
            },
            _ => Err(tensor_op_error(
                "VisionLayerNorm::forward",
                "unsupported tensor dtype",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotary Position Embedding with LongRoPE scaling
// ─────────────────────────────────────────────────────────────────────────────

/// Rotary Position Embedding for Llama-3.2 with optional position interpolation
pub struct Llama32RotaryEmbedding {
    inv_freq: Vec<f64>,
    max_seq_len: usize,
    head_dim: usize,
    scaling_factor: f32,
    use_scaled: bool,
}

impl Llama32RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        theta: f64,
        scaling_factor: f32,
        use_scaled: bool,
    ) -> Self {
        let half = head_dim / 2;
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| {
                let exponent = 2.0 * i as f64 / head_dim as f64;
                1.0 / theta.powf(exponent)
            })
            .collect();
        Self {
            inv_freq,
            max_seq_len,
            head_dim,
            scaling_factor,
            use_scaled,
        }
    }

    /// Number of inv-freq components (`head_dim / 2`)
    pub fn half_dim(&self) -> usize {
        self.inv_freq.len()
    }

    /// Width of one rotated head block.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Longest position this table was built for.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Effective position divisor: `scaling_factor` when scaled RoPE is on.
    fn position_scale(&self) -> f64 {
        if self.use_scaled {
            self.scaling_factor as f64
        } else {
            1.0
        }
    }

    /// Apply RoPE to query and key tensors (shape-preserving).
    ///
    /// `q` and `k` are `[seq_len, heads * head_dim]` or
    /// `[batch, seq_len, heads * head_dim]` and may carry different head counts
    /// (GQA keeps fewer KV heads). Every `head_dim`-wide head block is rotated
    /// independently with the LLaMA "rotate-half" pairing: component `i` pairs
    /// with component `i + head_dim/2` and the pair is rotated by
    /// `angle = pos * inv_freq[i]`.
    ///
    /// With `use_scaled` the position is divided by `scaling_factor` — linear
    /// position interpolation (Chen et al., 2023), which is what a single scalar
    /// factor can express. The piecewise low/high-frequency schedule shipped with
    /// Llama-3.1/3.2 needs `low_freq_factor`, `high_freq_factor` and the original
    /// context length; this config carries none of them, so it is not claimed
    /// here.
    pub fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let scale = self.position_scale();
        if !scale.is_finite() || scale <= 0.0 {
            return Err(tensor_op_error(
                "Llama32RotaryEmbedding::apply_rotary_emb",
                format!("rope scaling factor must be finite and positive, got {scale}"),
            ));
        }
        // One (sin, cos) table shared by both tensors and every head.
        let mut table = Vec::with_capacity(position_ids.len() * self.inv_freq.len());
        for &pos in position_ids {
            for &freq in &self.inv_freq {
                let angle = (pos as f64 / scale) * freq;
                table.push((angle.sin() as f32, angle.cos() as f32));
            }
        }

        let q_rotated = self.rotate(q, position_ids.len(), &table, "query")?;
        let k_rotated = self.rotate(k, position_ids.len(), &table, "key")?;
        Ok((q_rotated, k_rotated))
    }

    /// Rotate every head block of one tensor with the precomputed table.
    fn rotate(
        &self,
        tensor: &Tensor,
        positions: usize,
        table: &[(f32, f32)],
        role: &str,
    ) -> Result<Tensor> {
        let half = self.inv_freq.len();
        match tensor {
            Tensor::F32(arr) => {
                let shape = arr.shape().to_vec();
                let (batch, seq_len, width) =
                    split_sequence_shape(&shape, "Llama32RotaryEmbedding::apply_rotary_emb")?;
                if self.head_dim == 0 || !width.is_multiple_of(self.head_dim) {
                    return Err(tensor_op_error(
                        "Llama32RotaryEmbedding::apply_rotary_emb",
                        format!(
                            "{role} width {width} is not a multiple of head_dim {}",
                            self.head_dim
                        ),
                    ));
                }
                if seq_len != positions {
                    return Err(tensor_op_error(
                        "Llama32RotaryEmbedding::apply_rotary_emb",
                        format!(
                            "{role} has {seq_len} positions but {positions} position ids were given"
                        ),
                    ));
                }

                let heads = width / self.head_dim;
                let mut data: Vec<f32> = arr.iter().copied().collect();
                for b in 0..batch {
                    for t in 0..seq_len {
                        let row = (b * seq_len + t) * width;
                        for head in 0..heads {
                            let base = row + head * self.head_dim;
                            for i in 0..half {
                                let (sin, cos) = table[t * half + i];
                                let x = data[base + i];
                                let y = data[base + i + half];
                                data[base + i] = x * cos - y * sin;
                                data[base + i + half] = x * sin + y * cos;
                            }
                        }
                    }
                }

                let rotated = ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|e| {
                    tensor_op_error(
                        "Llama32RotaryEmbedding::apply_rotary_emb",
                        format!("shape error while rebuilding the {role} tensor: {e}"),
                    )
                })?;
                Ok(Tensor::F32(rotated))
            },
            _ => Err(tensor_op_error(
                "Llama32RotaryEmbedding::apply_rotary_emb",
                "unsupported tensor dtype for RoPE",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vision Patch Embedding
// ─────────────────────────────────────────────────────────────────────────────

/// Splits an image into non-overlapping patches and projects each patch to the
/// vision hidden dimension.
///
/// Input shape:  `[H, W, 3]` (HxW pixels, 3 channels)
/// Output shape: `[num_patches, vision_hidden_size]`
pub struct VisionPatchEmbedding {
    /// Linear projection from `(patch_size² * channels)` → `vision_hidden_size`
    patch_proj: Linear,
    patch_size: usize,
    num_channels: usize,
    vision_hidden_size: usize,
    /// Position embedding: `[num_patches, vision_hidden_size]`
    position_embedding: Tensor,
    num_patches: usize,
}

impl VisionPatchEmbedding {
    pub fn new(config: &Llama32Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &Llama32Config, device: Device) -> Result<Self> {
        let num_channels = 3_usize;
        let patch_dim = config.patch_size * config.patch_size * num_channels;
        let patch_proj =
            Linear::new_with_device(patch_dim, config.vision_hidden_size, false, device);
        // Learned position embedding for all patches
        let pos_emb_size = config.num_patches * config.vision_hidden_size;
        let position_embedding = Tensor::zeros(&[pos_emb_size])?;

        Ok(Self {
            patch_proj,
            patch_size: config.patch_size,
            num_channels,
            vision_hidden_size: config.vision_hidden_size,
            position_embedding,
            num_patches: config.num_patches,
        })
    }

    /// Embed pixel values into patch tokens.
    ///
    /// `pixel_values` must have length `height * width * num_channels`.
    /// Returns a tensor of shape `[num_patches, vision_hidden_size]`.
    pub fn embed_patches(
        &self,
        pixel_values: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let expected = height * width * self.num_channels;
        if pixel_values.len() != expected {
            return Err(tensor_op_error(
                "VisionPatchEmbedding::embed_patches",
                format!(
                    "pixel_values length mismatch: expected {expected}, got {}",
                    pixel_values.len()
                ),
            ));
        }
        let patches_h = height / self.patch_size;
        let patches_w = width / self.patch_size;
        let total_patches = patches_h * patches_w;
        let patch_dim = self.patch_size * self.patch_size * self.num_channels;

        // Extract patches row by row
        let mut patch_buffer = Vec::with_capacity(total_patches * patch_dim);
        for ph in 0..patches_h {
            for pw in 0..patches_w {
                for pi in 0..self.patch_size {
                    for pj in 0..self.patch_size {
                        let row = ph * self.patch_size + pi;
                        let col = pw * self.patch_size + pj;
                        for c in 0..self.num_channels {
                            let idx = (row * width + col) * self.num_channels + c;
                            patch_buffer.push(pixel_values[idx]);
                        }
                    }
                }
            }
        }

        let patches_tensor = Tensor::from_vec(patch_buffer, &[total_patches, patch_dim])?;
        let projected = self.patch_proj.forward(patches_tensor)?;
        Ok(projected)
    }

    pub fn parameter_count(&self) -> usize {
        self.patch_proj.parameter_count() + self.position_embedding.len()
    }

    pub fn num_patches(&self) -> usize {
        self.num_patches
    }

    pub fn vision_hidden_size(&self) -> usize {
        self.vision_hidden_size
    }
}

impl Layer for VisionPatchEmbedding {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // input shape: [total_patches, patch_dim]
        let projected = self.patch_proj.forward(input)?;
        // Add position embeddings (broadcast-compatible with projected shape)
        match (&projected, &self.position_embedding) {
            (Tensor::F32(p), Tensor::F32(pe)) => {
                let p_shape = p.shape();
                let total_elems: usize = p_shape.iter().product();
                if pe.len() >= total_elems {
                    let pe_slice: Vec<f32> = pe.iter().copied().take(total_elems).collect();
                    let pe_arr = ArrayD::from_shape_vec(IxDyn(p_shape), pe_slice).map_err(|e| {
                        tensor_op_error(
                            "VisionPatchEmbedding::forward",
                            format!("position embedding shape error: {e}"),
                        )
                    })?;
                    Ok(Tensor::F32(p + &pe_arr))
                } else {
                    Ok(projected)
                }
            },
            _ => Err(tensor_op_error(
                "VisionPatchEmbedding::forward",
                "unsupported tensor dtype",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vision MLP (GELU feed-forward)
// ─────────────────────────────────────────────────────────────────────────────

/// GELU feed-forward network used in the ViT-style vision encoder
pub struct VisionMLP {
    fc1: Linear,
    fc2: Linear,
}

impl VisionMLP {
    pub fn new(config: &Llama32Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &Llama32Config, device: Device) -> Result<Self> {
        let fc1 = Linear::new_with_device(
            config.vision_hidden_size,
            config.vision_intermediate_size,
            true,
            device,
        );
        let fc2 = Linear::new_with_device(
            config.vision_intermediate_size,
            config.vision_hidden_size,
            true,
            device,
        );
        Ok(Self { fc1, fc2 })
    }

    pub fn parameter_count(&self) -> usize {
        self.fc1.parameter_count() + self.fc2.parameter_count()
    }
}

impl Layer for VisionMLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden = self.fc1.forward(input)?;
        let activated = gelu(&hidden)?;
        self.fc2.forward(activated)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vision Attention (standard multi-head self-attention for vision encoder)
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-head self-attention for the ViT-style vision encoder
pub struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    pub fn new(config: &Llama32Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &Llama32Config, device: Device) -> Result<Self> {
        let head_dim = config.vision_hidden_size / config.vision_num_attention_heads;
        let q_proj = Linear::new_with_device(
            config.vision_hidden_size,
            config.vision_hidden_size,
            true,
            device,
        );
        let k_proj = Linear::new_with_device(
            config.vision_hidden_size,
            config.vision_hidden_size,
            true,
            device,
        );
        let v_proj = Linear::new_with_device(
            config.vision_hidden_size,
            config.vision_hidden_size,
            true,
            device,
        );
        let out_proj = Linear::new_with_device(
            config.vision_hidden_size,
            config.vision_hidden_size,
            true,
            device,
        );
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: config.vision_num_attention_heads,
            head_dim,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.q_proj.parameter_count()
            + self.k_proj.parameter_count()
            + self.v_proj.parameter_count()
            + self.out_proj.parameter_count()
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl Layer for VisionAttention {
    type Input = Tensor;
    type Output = Tensor;

    /// Bidirectional multi-head self-attention over the patch tokens.
    ///
    /// `softmax(Q Kᵀ / sqrt(head_dim)) V` — every patch may attend to every
    /// other patch (a ViT encoder is not causal).
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let shape = input.shape().to_vec();
        let (batch, seq_len, features) = split_sequence_shape(&shape, "VisionAttention::forward")?;
        let width = self.num_heads * self.head_dim;
        if features != width {
            return Err(tensor_op_error(
                "VisionAttention::forward",
                format!(
                    "input feature size {features} != num_heads {} * head_dim {}",
                    self.num_heads, self.head_dim
                ),
            ));
        }

        let q = self.q_proj.forward(input.clone())?.data()?;
        let k = self.k_proj.forward(input.clone())?.data()?;
        let v = self.v_proj.forward(input)?.data()?;

        let stride = seq_len * width;
        let mut context = Vec::with_capacity(batch * stride);
        for b in 0..batch {
            let range = b * stride..(b + 1) * stride;
            context.extend_from_slice(&multi_head_sdpa(
                &q[range.clone()],
                &k[range.clone()],
                &v[range],
                seq_len,
                seq_len,
                self.num_heads,
                self.head_dim,
                false,
            )?);
        }

        let attn_output = Tensor::from_vec(context, &shape)?;
        self.out_proj.forward(attn_output)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vision Encoder Layer
// ─────────────────────────────────────────────────────────────────────────────

/// Single ViT-style encoder layer: self-attention + MLP with pre-norm
pub struct VisionEncoderLayer {
    self_attn: VisionAttention,
    mlp: VisionMLP,
    layer_norm1: VisionLayerNorm,
    layer_norm2: VisionLayerNorm,
}

impl VisionEncoderLayer {
    pub fn new(config: &Llama32Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &Llama32Config, device: Device) -> Result<Self> {
        let self_attn = VisionAttention::new_with_device(config, device)?;
        let mlp = VisionMLP::new_with_device(config, device)?;
        let layer_norm1 = VisionLayerNorm::new(config.vision_hidden_size, 1e-6)?;
        let layer_norm2 = VisionLayerNorm::new(config.vision_hidden_size, 1e-6)?;
        Ok(Self {
            self_attn,
            mlp,
            layer_norm1,
            layer_norm2,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.self_attn.parameter_count()
            + self.mlp.parameter_count()
            + self.layer_norm1.parameter_count()
            + self.layer_norm2.parameter_count()
    }
}

impl Layer for VisionEncoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Pre-norm self-attention with residual
        let normed1 = self.layer_norm1.forward(input.clone())?;
        let attn_out = self.self_attn.forward(normed1)?;
        let after_attn = input.add(&attn_out)?;

        // Pre-norm MLP with residual
        let normed2 = self.layer_norm2.forward(after_attn.clone())?;
        let mlp_out = self.mlp.forward(normed2)?;
        after_attn.add(&mlp_out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vision Encoder (ViT-style stack)
// ─────────────────────────────────────────────────────────────────────────────

/// Full ViT-style vision encoder consisting of:
///   1. Patch embedding (splits image into patches, projects to hidden dim)
///   2. Stack of `VisionEncoderLayer` blocks
///   3. Final layer norm
///
/// Output shape: `[num_patches, vision_hidden_size]`
pub struct VisionEncoder {
    patch_embedding: VisionPatchEmbedding,
    layers: Vec<VisionEncoderLayer>,
    final_norm: VisionLayerNorm,
    vision_hidden_size: usize,
}

impl VisionEncoder {
    pub fn new(config: &Llama32Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &Llama32Config, device: Device) -> Result<Self> {
        let patch_embedding = VisionPatchEmbedding::new_with_device(config, device)?;
        let mut layers = Vec::with_capacity(config.vision_num_hidden_layers);
        for _ in 0..config.vision_num_hidden_layers {
            layers.push(VisionEncoderLayer::new_with_device(config, device)?);
        }
        let final_norm = VisionLayerNorm::new(config.vision_hidden_size, 1e-6)?;
        Ok(Self {
            patch_embedding,
            layers,
            final_norm,
            vision_hidden_size: config.vision_hidden_size,
        })
    }

    /// Encode pixel values into vision token features.
    ///
    /// Returns a tensor of shape `[num_patches, vision_hidden_size]`.
    pub fn encode(&self, pixel_values: &[f32], height: usize, width: usize) -> Result<Tensor> {
        let patch_tokens = self.patch_embedding.embed_patches(pixel_values, height, width)?;
        let mut hidden = patch_tokens;
        for layer in &self.layers {
            hidden = layer.forward(hidden)?;
        }
        self.final_norm.forward(hidden)
    }

    pub fn parameter_count(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| l.parameter_count()).sum();
        self.patch_embedding.parameter_count() + layer_params + self.final_norm.parameter_count()
    }

    pub fn vision_hidden_size(&self) -> usize {
        self.vision_hidden_size
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-Attention Layer (text queries, vision keys/values)
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-attention layer that lets text tokens attend to vision encoder output.
///
/// Text queries come from the text decoder hidden states.
/// Keys and values come from the vision encoder output.
pub struct CrossAttentionLayer {
    /// Query projection (text hidden → head_dim * num_heads)
    q_proj: Linear,
    /// Key projection (vision_output → head_dim * num_heads)
    k_proj: Linear,
    /// Value projection (vision_output → head_dim * num_heads)
    v_proj: Linear,
    /// Output projection
    o_proj: Linear,
    /// Query norm
    q_norm: Llama32RmsNorm,
    /// Key norm
    k_norm: Llama32RmsNorm,
    num_heads: usize,
    head_dim: usize,
}

impl CrossAttentionLayer {
    pub fn new(config: &Llama32Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &Llama32Config, device: Device) -> Result<Self> {
        let head_dim = config.head_dim;
        let num_heads = config.num_attention_heads;
        let total_head_dim = head_dim * num_heads;

        let q_proj = Linear::new_with_device(config.hidden_size, total_head_dim, false, device);
        let k_proj =
            Linear::new_with_device(config.vision_hidden_size, total_head_dim, false, device);
        let v_proj =
            Linear::new_with_device(config.vision_hidden_size, total_head_dim, false, device);
        let o_proj = Linear::new_with_device(total_head_dim, config.hidden_size, false, device);
        let q_norm = Llama32RmsNorm::new(head_dim, config.rms_norm_eps)?;
        let k_norm = Llama32RmsNorm::new(head_dim, config.rms_norm_eps)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            head_dim,
        })
    }

    /// Cross-attend: text queries attend to vision key/value pairs.
    ///
    /// * `text_hidden` — shape `[seq_len, hidden_size]`
    /// * `vision_features` — shape `[num_patches, vision_hidden_size]`
    ///
    /// Returns a tensor of shape `[seq_len, hidden_size]`.
    pub fn cross_attend(&self, text_hidden: Tensor, vision_features: &Tensor) -> Result<Tensor> {
        let text_shape = text_hidden.shape().to_vec();
        let (text_batch, seq_len, _) =
            split_sequence_shape(&text_shape, "CrossAttentionLayer::cross_attend")?;
        let vision_shape = vision_features.shape().to_vec();
        let (vision_batch, num_patches, _) =
            split_sequence_shape(&vision_shape, "CrossAttentionLayer::cross_attend")?;
        if text_batch != vision_batch {
            return Err(tensor_op_error(
                "CrossAttentionLayer::cross_attend",
                format!("text batch {text_batch} does not match vision batch {vision_batch}"),
            ));
        }

        let q = self.q_proj.forward(text_hidden)?;
        let k = self.k_proj.forward(vision_features.clone())?;
        let v = self.v_proj.forward(vision_features.clone())?;

        // Per-head RMS norm on queries and keys (the norms are head_dim wide).
        let q_normed = self.q_norm.forward(q)?.data()?;
        let k_normed = self.k_norm.forward(k)?.data()?;
        let v_data = v.data()?;

        // Text queries attend over every vision patch (no causal mask).
        let width = self.num_heads * self.head_dim;
        let q_stride = seq_len * width;
        let kv_stride = num_patches * width;
        let mut context = Vec::with_capacity(text_batch * q_stride);
        for b in 0..text_batch {
            let kv_range = b * kv_stride..(b + 1) * kv_stride;
            context.extend_from_slice(&multi_head_sdpa(
                &q_normed[b * q_stride..(b + 1) * q_stride],
                &k_normed[kv_range.clone()],
                &v_data[kv_range],
                seq_len,
                num_patches,
                self.num_heads,
                self.head_dim,
                false,
            )?);
        }

        let attn_shape: Vec<usize> = if text_shape.len() == 2 {
            vec![seq_len, width]
        } else {
            vec![text_batch, seq_len, width]
        };
        let attn_output = Tensor::from_vec(context, &attn_shape)?;
        self.o_proj.forward(attn_output)
    }

    pub fn parameter_count(&self) -> usize {
        self.q_proj.parameter_count()
            + self.k_proj.parameter_count()
            + self.v_proj.parameter_count()
            + self.o_proj.parameter_count()
            + self.q_norm.parameter_count()
            + self.k_norm.parameter_count()
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Text Self-Attention (GQA with LongRoPE)
// ─────────────────────────────────────────────────────────────────────────────

/// Text decoder self-attention with GQA and LongRoPE
pub struct Llama32SelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: Llama32RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_query_groups: usize,
}

impl Llama32SelfAttention {
    pub fn new(config: &Llama32Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &Llama32Config, device: Device) -> Result<Self> {
        let head_dim = config.head_dim;
        let num_query_groups = config.num_attention_heads / config.num_key_value_heads;

        let q_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            false,
            device,
        );
        let k_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            false,
            device,
        );
        let v_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            false,
            device,
        );
        let o_proj = Linear::new_with_device(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            false,
            device,
        );
        let rotary_emb = Llama32RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.rope_scaling_factor,
            config.use_scaled_rope,
        );

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            num_query_groups,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.q_proj.parameter_count()
            + self.k_proj.parameter_count()
            + self.v_proj.parameter_count()
            + self.o_proj.parameter_count()
    }

    /// Number of query heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Number of key/value heads (`num_heads / num_query_groups`).
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Width of one attention head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl Layer for Llama32SelfAttention {
    type Input = Tensor;
    type Output = Tensor;

    /// Causal grouped-query self-attention.
    ///
    /// Accepts `[seq_len, hidden]` or `[batch, seq_len, hidden]` and returns the
    /// same rank. The computation is `softmax(mask(Q Kᵀ) / sqrt(head_dim)) V`
    /// with RoPE applied to Q and K and each KV head shared by
    /// `num_query_groups` query heads — the keys and values are read, not
    /// discarded.
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let shape = input.shape().to_vec();
        let (batch, seq_len, hidden) =
            split_sequence_shape(&shape, "Llama32SelfAttention::forward")?;
        let width = self.num_heads * self.head_dim;
        if hidden != width {
            return Err(tensor_op_error(
                "Llama32SelfAttention::forward",
                format!(
                    "input hidden size {hidden} != num_heads {} * head_dim {}",
                    self.num_heads, self.head_dim
                ),
            ));
        }

        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input)?;

        let position_ids: Vec<usize> = (0..seq_len).collect();
        let (q_rope, k_rope) = self.rotary_emb.apply_rotary_emb(&q, &k, &position_ids)?;

        // GQA: expand KV heads so query head `h` reads KV head `h / groups`.
        let k_expanded = self.expand_kv(&k_rope)?.data()?;
        let v_expanded = self.expand_kv(&v)?.data()?;
        let queries = q_rope.data()?;

        let stride = seq_len * width;
        let mut context = Vec::with_capacity(batch * stride);
        for b in 0..batch {
            let range = b * stride..(b + 1) * stride;
            context.extend_from_slice(&multi_head_sdpa(
                &queries[range.clone()],
                &k_expanded[range.clone()],
                &v_expanded[range],
                seq_len,
                seq_len,
                self.num_heads,
                self.head_dim,
                true,
            )?);
        }

        let attn_output = Tensor::from_vec(context, &shape)?;
        self.o_proj.forward(attn_output)
    }
}

impl Llama32SelfAttention {
    fn expand_kv(&self, kv: &Tensor) -> Result<Tensor> {
        if self.num_query_groups == 1 {
            return Ok(kv.clone());
        }
        match kv {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                let total = shape.iter().product::<usize>();
                let chunk_size = self.head_dim;
                let num_chunks = total / chunk_size;

                let flat: Vec<f32> = arr.iter().copied().collect();
                let mut expanded = Vec::with_capacity(total * self.num_query_groups);
                for chunk in 0..num_chunks {
                    let start = chunk * chunk_size;
                    let slice = &flat[start..start + chunk_size];
                    for _ in 0..self.num_query_groups {
                        expanded.extend_from_slice(slice);
                    }
                }
                let mut new_shape = shape.to_vec();
                if let Some(last) = new_shape.last_mut() {
                    *last *= self.num_query_groups;
                }
                let expanded_arr =
                    ArrayD::from_shape_vec(IxDyn(&new_shape), expanded).map_err(|e| {
                        tensor_op_error(
                            "Llama32SelfAttention::expand_kv",
                            format!("shape error: {e}"),
                        )
                    })?;
                Ok(Tensor::F32(expanded_arr))
            },
            _ => Err(tensor_op_error(
                "Llama32SelfAttention::expand_kv",
                "unsupported tensor dtype",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Text Decoder MLP (SwiGLU)
// ─────────────────────────────────────────────────────────────────────────────

/// SwiGLU FFN for the text decoder
pub struct Llama32MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Llama32MLP {
    pub fn new(config: &Llama32Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &Llama32Config, device: Device) -> Result<Self> {
        let gate_proj =
            Linear::new_with_device(config.hidden_size, config.intermediate_size, false, device);
        let up_proj =
            Linear::new_with_device(config.hidden_size, config.intermediate_size, false, device);
        let down_proj =
            Linear::new_with_device(config.intermediate_size, config.hidden_size, false, device);
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.gate_proj.parameter_count()
            + self.up_proj.parameter_count()
            + self.down_proj.parameter_count()
    }
}

impl Layer for Llama32MLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let gate_out = self.gate_proj.forward(input.clone())?;
        let up_out = self.up_proj.forward(input)?;
        let gate_activated = silu(&gate_out)?;
        let combined = match (&gate_activated, &up_out) {
            (Tensor::F32(g), Tensor::F32(u)) => Ok(Tensor::F32(g * u)),
            _ => Err(tensor_op_error(
                "Llama32MLP::forward",
                "tensor dtype mismatch in SwiGLU gate multiply",
            )),
        }?;
        self.down_proj.forward(combined)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Text Decoder Layer (self-attention + optional cross-attention + MLP)
// ─────────────────────────────────────────────────────────────────────────────

/// Llama-3.2 decoder layer.
///
/// When `has_cross_attention` is true, a `CrossAttentionLayer` is interleaved
/// between the self-attention block and the MLP block.
pub struct Llama32DecoderLayer {
    self_attn: Llama32SelfAttention,
    cross_attn: Option<CrossAttentionLayer>,
    mlp: Llama32MLP,
    input_layernorm: Llama32RmsNorm,
    post_attention_layernorm: Llama32RmsNorm,
    cross_attn_layernorm: Option<Llama32RmsNorm>,
}

impl Llama32DecoderLayer {
    pub fn new(config: &Llama32Config, has_cross_attention: bool) -> Result<Self> {
        Self::new_with_device(config, has_cross_attention, Device::CPU)
    }

    pub fn new_with_device(
        config: &Llama32Config,
        has_cross_attention: bool,
        device: Device,
    ) -> Result<Self> {
        let self_attn = Llama32SelfAttention::new_with_device(config, device)?;
        let mlp = Llama32MLP::new_with_device(config, device)?;
        let input_layernorm = Llama32RmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm =
            Llama32RmsNorm::new(config.hidden_size, config.rms_norm_eps)?;

        let (cross_attn, cross_attn_layernorm) = if has_cross_attention {
            (
                Some(CrossAttentionLayer::new_with_device(config, device)?),
                Some(Llama32RmsNorm::new(
                    config.hidden_size,
                    config.rms_norm_eps,
                )?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            self_attn,
            cross_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            cross_attn_layernorm,
        })
    }

    /// Forward pass with optional vision features for cross-attention.
    pub fn forward_with_vision(
        &self,
        input: Tensor,
        vision_features: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self-attention block
        let normed = self.input_layernorm.forward(input.clone())?;
        let sa_out = self.self_attn.forward(normed)?;
        let mut hidden = input.add(&sa_out)?;

        // Cross-attention block (only on designated layers with vision features)
        if let (Some(cross_attn), Some(norm), Some(vis)) = (
            &self.cross_attn,
            &self.cross_attn_layernorm,
            vision_features,
        ) {
            let normed_for_ca = norm.forward(hidden.clone())?;
            let ca_out = cross_attn.cross_attend(normed_for_ca, vis)?;
            hidden = hidden.add(&ca_out)?;
        }

        // MLP block
        let normed_mlp = self.post_attention_layernorm.forward(hidden.clone())?;
        let mlp_out = self.mlp.forward(normed_mlp)?;
        hidden.add(&mlp_out)
    }

    pub fn has_cross_attention(&self) -> bool {
        self.cross_attn.is_some()
    }

    pub fn parameter_count(&self) -> usize {
        let cross_params = self.cross_attn.as_ref().map(|c| c.parameter_count()).unwrap_or(0)
            + self.cross_attn_layernorm.as_ref().map(|n| n.parameter_count()).unwrap_or(0);
        self.self_attn.parameter_count()
            + self.mlp.parameter_count()
            + self.input_layernorm.parameter_count()
            + self.post_attention_layernorm.parameter_count()
            + cross_params
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Llama32CrossAttentionDecoder
// ─────────────────────────────────────────────────────────────────────────────

/// Text decoder that interleaves cross-attention and self-attention layers.
///
/// Cross-attention is injected at the layer indices specified by
/// `config.cross_attention_layers`.
pub struct Llama32CrossAttentionDecoder {
    config: Llama32Config,
    embed_tokens: Embedding,
    layers: Vec<Llama32DecoderLayer>,
    norm: Llama32RmsNorm,
}

impl Llama32CrossAttentionDecoder {
    pub fn new(config: Llama32Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: Llama32Config, device: Device) -> Result<Self> {
        config.validate()?;
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let has_cross_attention = config.cross_attention_layers.contains(&layer_idx);
            layers.push(Llama32DecoderLayer::new_with_device(
                &config,
                has_cross_attention,
                device,
            )?);
        }
        let norm = Llama32RmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn config(&self) -> &Llama32Config {
        &self.config
    }

    pub fn parameter_count(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| l.parameter_count()).sum();
        self.embed_tokens.parameter_count() + layer_params + self.norm.parameter_count()
    }

    /// Run the decoder: embed → layers → final norm.
    ///
    /// `vision_features` is `None` for text-only inference.
    pub fn run(&self, input_ids: Vec<u32>, vision_features: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            let vis = if layer.has_cross_attention() { vision_features } else { None };
            hidden = layer.forward_with_vision(hidden, vis)?;
        }
        self.norm.forward(hidden)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Llama32VisionModel (vision encoder + cross-attention text decoder)
// ─────────────────────────────────────────────────────────────────────────────

/// Full Llama-3.2 vision-language model.
///
/// Consists of:
/// 1. A ViT-style `VisionEncoder`
/// 2. A `Llama32CrossAttentionDecoder` (text backbone with cross-attn layers)
pub struct Llama32VisionModel {
    config: Llama32Config,
    vision_encoder: VisionEncoder,
    text_decoder: Llama32CrossAttentionDecoder,
}

impl Llama32VisionModel {
    pub fn new(config: Llama32Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: Llama32Config, device: Device) -> Result<Self> {
        let vision_encoder = VisionEncoder::new_with_device(&config, device)?;
        let text_decoder = Llama32CrossAttentionDecoder::new_with_device(config.clone(), device)?;
        Ok(Self {
            config,
            vision_encoder,
            text_decoder,
        })
    }

    pub fn config(&self) -> &Llama32Config {
        &self.config
    }

    pub fn parameter_count(&self) -> usize {
        self.vision_encoder.parameter_count() + self.text_decoder.parameter_count()
    }

    /// Encode pixel values through the vision encoder.
    ///
    /// Returns vision token features of shape `[num_patches, vision_hidden_size]`.
    pub fn encode_image(
        &self,
        pixel_values: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        self.vision_encoder.encode(pixel_values, height, width)
    }

    /// Run the full vision-language forward pass.
    ///
    /// * `input_ids` — text token IDs
    /// * `pixel_values` — flat pixel buffer `[height * width * 3]` for the image
    /// * `height`, `width` — image dimensions
    ///
    /// Returns hidden states of shape `[seq_len, hidden_size]`.
    pub fn forward_multimodal(
        &self,
        input_ids: Vec<u32>,
        pixel_values: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let vision_features = self.encode_image(pixel_values, height, width)?;
        self.text_decoder.run(input_ids, Some(&vision_features))
    }

    /// Text-only forward pass (no image).
    pub fn forward_text_only(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        self.text_decoder.run(input_ids, None)
    }
}

impl Model for Llama32VisionModel {
    type Config = Llama32Config;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        self.forward_text_only(input_ids)
    }

    /// Not supported yet — returns a structured error rather than pretending.
    ///
    /// A `MllamaForConditionalGeneration` checkpoint carries a vision tower with
    /// tile/aspect-ratio embeddings and gated cross-attention that this
    /// simplified architecture does not model, so its tensors have no faithful
    /// destination here. Binding only the text decoder would leave the vision
    /// path randomly initialised while reporting success, which would be worse
    /// than refusing.
    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(
            trustformers_core::errors::TrustformersError::not_implemented(
                "Llama-3.2 vision checkpoint loading is not implemented: this architecture omits \
                 the Mllama tile/aspect-ratio embeddings and cross-attention gates, so a \
                 checkpoint cannot be bound faithfully. Install weights explicitly through the \
                 layer setters instead."
                    .to_string(),
            ),
        )
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.parameter_count()
    }
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
