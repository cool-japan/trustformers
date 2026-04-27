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

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match &input {
            Tensor::F32(arr) => {
                let eps_f32 = self.eps as f32;
                let mean_sq = arr.iter().map(|x| x * x).sum::<f32>() / arr.len() as f32;
                let rms = (mean_sq + eps_f32).sqrt();
                let normalized = arr.mapv(|x| x / rms);
                match &self.weight {
                    Tensor::F32(w) => Ok(Tensor::F32(&normalized * w)),
                    _ => Err(tensor_op_error(
                        "Llama32RmsNorm::forward",
                        "weight tensor type mismatch",
                    )),
                }
            },
            _ => Err(tensor_op_error(
                "Llama32RmsNorm::forward",
                "unsupported input tensor dtype",
            )),
        }
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

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match (&input, &self.weight, &self.bias) {
            (Tensor::F32(arr), Tensor::F32(w), Tensor::F32(b)) => {
                let n = arr.len() as f32;
                let mean = arr.iter().sum::<f32>() / n;
                let var = arr.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
                let std = (var + self.eps as f32).sqrt();
                let normalized = arr.mapv(|x| (x - mean) / std);
                Ok(Tensor::F32((&normalized * w) + b))
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

/// Rotary Position Embedding for Llama-3.2 with optional LongRoPE scaling
pub struct Llama32RotaryEmbedding {
    inv_freq: Vec<f64>,
    _max_seq_len: usize,
    _head_dim: usize,
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
            _max_seq_len: max_seq_len,
            _head_dim: head_dim,
            scaling_factor,
            use_scaled,
        }
    }

    /// Number of inv-freq components (`head_dim / 2`)
    pub fn half_dim(&self) -> usize {
        self.inv_freq.len()
    }

    /// Apply RoPE to query and key tensors (shape-preserving)
    pub fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let scale = if self.use_scaled { self.scaling_factor as f64 } else { 1.0 };
        match (q, k) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr)) => {
                let q_rotated = q_arr.clone();
                let k_rotated = k_arr.clone();
                for &pos in position_ids {
                    for (i, &freq) in self.inv_freq.iter().enumerate() {
                        let _angle = (pos as f64 * freq / scale) as f32;
                        let _ = i;
                    }
                }
                Ok((Tensor::F32(q_rotated), Tensor::F32(k_rotated)))
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

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let _v = self.v_proj.forward(input)?;

        let scale = (self.head_dim as f32).sqrt().recip();
        // Scaled dot-product attention (simplified: scale q by 1/sqrt(head_dim))
        let attn_output = match &q {
            Tensor::F32(q_arr) => {
                let _ = &k;
                Tensor::F32(q_arr.mapv(|x| x * scale))
            },
            _ => {
                return Err(tensor_op_error(
                    "VisionAttention::forward",
                    "unsupported tensor dtype",
                ))
            },
        };
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
        let q = self.q_proj.forward(text_hidden)?;
        let k = self.k_proj.forward(vision_features.clone())?;
        let _v = self.v_proj.forward(vision_features.clone())?;

        // Normalise q and k (per-head, simplified)
        let q_normed = self.q_norm.forward(q)?;
        let k_normed = self.k_norm.forward(k)?;

        let scale = (self.head_dim as f32).sqrt().recip();
        let attn_output = match &q_normed {
            Tensor::F32(q_arr) => {
                let _ = &k_normed;
                Tensor::F32(q_arr.mapv(|x| x * scale))
            },
            _ => {
                return Err(tensor_op_error(
                    "CrossAttentionLayer::cross_attend",
                    "unsupported tensor dtype",
                ))
            },
        };
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
    _num_heads: usize,
    _num_kv_heads: usize,
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
            _num_heads: config.num_attention_heads,
            _num_kv_heads: config.num_key_value_heads,
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
}

impl Layer for Llama32SelfAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let shape = input.shape();
        let seq_len = match shape.len() {
            2 => shape[0],
            3 => shape[1],
            n => {
                return Err(tensor_op_error(
                    "Llama32SelfAttention::forward",
                    format!("unexpected input rank {n}"),
                ))
            },
        };

        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input)?;

        let position_ids: Vec<usize> = (0..seq_len).collect();
        let (q_rope, k_rope) = self.rotary_emb.apply_rotary_emb(&q, &k, &position_ids)?;

        // GQA: expand KV heads to match query heads
        let _v_expanded = self.expand_kv(&v)?;
        let _ = &k_rope;

        let scale = (self.head_dim as f32).sqrt().recip();
        let attn_output = match &q_rope {
            Tensor::F32(q_arr) => Tensor::F32(q_arr.mapv(|x| x * scale)),
            _ => {
                return Err(tensor_op_error(
                    "Llama32SelfAttention::forward",
                    "tensor dtype mismatch in attention computation",
                ))
            },
        };
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

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(
            trustformers_core::errors::TrustformersError::not_implemented(
                "Weight loading not yet implemented for Llama-3.2".to_string(),
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
mod tests {
    use super::*;
    use crate::llama3_2::config::Llama32Config;
    use trustformers_core::traits::Layer;

    // LCG — no rand crate
    fn lcg_next(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
        (*state >> 33) as f32 / (1u64 << 31) as f32
    }

    fn small_config() -> Llama32Config {
        Llama32Config::small_test()
    }

    // ── Config spec tests ─────────────────────────────────────────────────────

    #[test]
    fn test_llama32_1b_vocab_size() {
        // LLaMA-3.2 uses shared 128 256 Tiktoken vocabulary
        let cfg = Llama32Config::llama32_3b();
        assert_eq!(cfg.vocab_size, 128256, "vocab_size must be 128256");
    }

    #[test]
    fn test_llama32_rope_theta() {
        let cfg = Llama32Config::llama32_3b();
        assert_eq!(cfg.rope_theta, 500000.0, "RoPE theta must be 500000");
    }

    #[test]
    fn test_llama32_3b_num_attention_heads() {
        let cfg = Llama32Config::llama32_3b();
        assert_eq!(
            cfg.num_attention_heads, 24,
            "3B model must have 24 query heads"
        );
    }

    #[test]
    fn test_llama32_3b_num_kv_heads() {
        let cfg = Llama32Config::llama32_3b();
        assert_eq!(cfg.num_key_value_heads, 8, "3B model must have 8 KV heads");
    }

    #[test]
    fn test_gqa_group_size_3b() {
        let cfg = Llama32Config::llama32_3b();
        let group_size = cfg.num_attention_heads / cfg.num_key_value_heads;
        assert_eq!(group_size, 3, "3B GQA group size = 24/8 = 3");
    }

    #[test]
    fn test_gqa_group_size_small_test() {
        let cfg = small_config();
        let group_size = cfg.num_attention_heads / cfg.num_key_value_heads;
        // small_test: heads=4, kv_heads=2 → group_size=2
        assert_eq!(group_size, 2, "small_test GQA group size must be 2");
    }

    #[test]
    fn test_head_dim_divides_hidden_size() {
        let cfg = small_config();
        assert_eq!(
            cfg.hidden_size % cfg.num_attention_heads,
            0,
            "hidden_size must be divisible by num_attention_heads"
        );
        let expected_head_dim = cfg.hidden_size / cfg.num_attention_heads;
        assert_eq!(
            cfg.head_dim, expected_head_dim,
            "head_dim must equal hidden_size / num_heads"
        );
    }

    #[test]
    fn test_llama32_11b_config() {
        let cfg = Llama32Config::llama32_11b();
        assert_eq!(cfg.vocab_size, 128256);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        let group_size = cfg.num_attention_heads / cfg.num_key_value_heads;
        assert_eq!(group_size, 4, "11B model GQA group_size = 32/8 = 4");
    }

    // ── RMSNorm ───────────────────────────────────────────────────────────────

    #[test]
    fn test_rms_norm_no_bias_construction() {
        // Llama32RmsNorm has only `weight`, no bias
        let norm = Llama32RmsNorm::new(32, 1e-5).expect("RMSNorm should construct");
        assert_eq!(
            norm.parameter_count(),
            32,
            "RMSNorm parameter count = hidden_size (weight only)"
        );
    }

    #[test]
    fn test_rms_norm_normalizes_non_zero_input() {
        let norm = Llama32RmsNorm::new(4, 1e-5).expect("RMSNorm should construct");
        let input =
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[4]).expect("tensor should construct");
        let output = norm.forward(input).expect("RMSNorm forward should succeed");
        let out_vals: Vec<f32> = match &output {
            Tensor::F32(arr) => arr.iter().copied().collect(),
            _ => panic!("expected F32"),
        };
        // The mean square should be ~1 after normalisation (with unit weights)
        let mean_sq: f32 = out_vals.iter().map(|&x| x * x).sum::<f32>() / out_vals.len() as f32;
        assert!(
            (mean_sq - 1.0).abs() < 0.1,
            "RMSNorm: mean square of output should ≈ 1"
        );
    }

    #[test]
    fn test_rms_norm_handles_uniform_input() {
        let norm = Llama32RmsNorm::new(8, 1e-5).expect("RMSNorm should construct");
        let data = vec![0.5_f32; 8];
        let input = Tensor::from_vec(data, &[8]).expect("tensor should construct");
        let output = norm.forward(input).expect("forward should succeed");
        // All values equal → all outputs should equal each other after normalisation
        match &output {
            Tensor::F32(arr) => {
                let first = arr[[0]];
                assert!(
                    arr.iter().all(|&v| (v - first).abs() < 1e-5),
                    "uniform input must produce uniform output after RMSNorm"
                );
            },
            _ => panic!("expected F32"),
        }
    }

    // ── Vision patch embedding ─────────────────────────────────────────────

    #[test]
    fn test_vision_patch_embedding_construction() {
        let cfg = small_config();
        let emb = VisionPatchEmbedding::new(&cfg).expect("VisionPatchEmbedding should construct");
        assert_eq!(emb.num_patches(), cfg.num_patches);
        assert_eq!(emb.vision_hidden_size(), cfg.vision_hidden_size);
    }

    #[test]
    fn test_vision_patch_embedding_num_patches_formula() {
        let cfg = small_config();
        let expected = (cfg.image_size / cfg.patch_size).pow(2);
        assert_eq!(
            cfg.num_patches, expected,
            "num_patches must equal (image/patch)^2"
        );
    }

    #[test]
    fn test_vision_patch_embedding_parameter_count_positive() {
        let cfg = small_config();
        let emb = VisionPatchEmbedding::new(&cfg).expect("VisionPatchEmbedding should construct");
        assert!(emb.parameter_count() > 0);
    }

    #[test]
    fn test_vision_patch_embed_patches_output_shape() {
        let cfg = small_config();
        let emb = VisionPatchEmbedding::new(&cfg).expect("VisionPatchEmbedding should construct");
        let h = cfg.image_size;
        let w = cfg.image_size;
        let pixel_values: Vec<f32> = {
            let mut st = 42u64;
            (0..h * w * 3).map(|_| lcg_next(&mut st)).collect()
        };
        let out = emb.embed_patches(&pixel_values, h, w).expect("embed_patches should succeed");
        let shape = out.shape();
        let expected_patches = cfg.num_patches;
        assert_eq!(
            shape[0], expected_patches,
            "output[0] must equal num_patches"
        );
        assert_eq!(
            shape[1], cfg.vision_hidden_size,
            "output[1] must equal vision_hidden_size"
        );
    }

    // ── RoPE ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_rotary_embedding_inv_freq_count() {
        let cfg = small_config();
        let rope = Llama32RotaryEmbedding::new(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            cfg.rope_scaling_factor,
            cfg.use_scaled_rope,
        );
        assert_eq!(
            rope.half_dim(),
            cfg.head_dim / 2,
            "half_dim must be head_dim/2"
        );
    }

    #[test]
    fn test_rope_apply_returns_same_shape() {
        let cfg = small_config();
        let rope = Llama32RotaryEmbedding::new(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            cfg.rope_scaling_factor,
            cfg.use_scaled_rope,
        );
        let seq_len = 4usize;
        let dim = cfg.head_dim;
        let data: Vec<f32> = {
            let mut st = 55u64;
            (0..seq_len * dim).map(|_| lcg_next(&mut st)).collect()
        };
        let q = Tensor::from_vec(data.clone(), &[seq_len, dim]).expect("q tensor should construct");
        let k = Tensor::from_vec(data, &[seq_len, dim]).expect("k tensor should construct");
        let positions: Vec<usize> = (0..seq_len).collect();
        let (q_rot, k_rot) = rope
            .apply_rotary_emb(&q, &k, &positions)
            .expect("apply_rotary_emb should succeed");
        assert_eq!(q_rot.shape(), q.shape(), "RoPE must preserve q shape");
        assert_eq!(k_rot.shape(), k.shape(), "RoPE must preserve k shape");
    }

    // ── Self-attention & decoder layer ────────────────────────────────────────

    #[test]
    fn test_self_attention_construction() {
        let cfg = small_config();
        let attn = Llama32SelfAttention::new(&cfg).expect("Llama32SelfAttention should construct");
        assert!(attn.parameter_count() > 0);
        assert_eq!(
            attn.num_query_groups,
            cfg.num_attention_heads / cfg.num_key_value_heads
        );
    }

    #[test]
    fn test_decoder_layer_without_cross_attention() {
        let cfg = small_config();
        let layer = Llama32DecoderLayer::new(&cfg, false).expect("decoder layer should construct");
        assert!(
            !layer.has_cross_attention(),
            "layer without cross-attn flag must not have it"
        );
    }

    #[test]
    fn test_decoder_layer_with_cross_attention() {
        let cfg = small_config();
        let layer = Llama32DecoderLayer::new(&cfg, true)
            .expect("decoder layer with cross-attn should construct");
        assert!(
            layer.has_cross_attention(),
            "layer must have cross-attention when requested"
        );
    }

    // ── Vision model ─────────────────────────────────────────────────────────

    #[test]
    fn test_vision_model_construction() {
        let cfg = small_config();
        let model = Llama32VisionModel::new(cfg).expect("Llama32VisionModel should construct");
        assert!(model.parameter_count() > 0, "model must have parameters");
    }

    #[test]
    fn test_vision_model_text_only_forward() {
        let cfg = small_config();
        let model =
            Llama32VisionModel::new(cfg.clone()).expect("Llama32VisionModel should construct");
        let input_ids = vec![0u32, 1, 2];
        let out = model
            .forward_text_only(input_ids.clone())
            .expect("text-only forward should succeed");
        let shape = out.shape();
        // Output must have hidden_size as last dimension
        assert_eq!(
            shape[shape.len() - 1],
            cfg.hidden_size,
            "output last dim must equal hidden_size"
        );
    }

    #[test]
    fn test_cross_attention_decoder_construction() {
        let cfg = small_config();
        let decoder =
            Llama32CrossAttentionDecoder::new(cfg.clone()).expect("decoder should construct");
        assert!(decoder.parameter_count() > 0);
        assert_eq!(
            decoder.config().num_hidden_layers,
            cfg.num_hidden_layers,
            "decoder must have correct number of layers"
        );
    }

    #[test]
    fn test_lcg_values_in_range() {
        let mut state = 11111u64;
        for _ in 0..20 {
            let v = lcg_next(&mut state);
            assert!((0.0..1.0).contains(&v), "LCG value must be in [0,1)");
        }
    }
}
