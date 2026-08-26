use crate::common::ActivationType;
use crate::phi3::config::Phi3Config;
use crate::weight_loading::checkpoint::{Checkpoint, LoadReport, UnusedTensors, WeightBinder};
use scirs2_core::ndarray::{Array2, ArrayD, Ix2, IxDyn};
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result, TrustformersError},
    layers::{Embedding, Linear},
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

/// RMSNorm layer (Root Mean Square Layer Normalization)
/// Used in Phi-3 for efficient normalization
pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
    device: Device,
}

impl RMSNorm {
    pub fn new(normalized_shape: usize, eps: f32) -> Result<Self> {
        Self::new_with_device(normalized_shape, eps, Device::CPU)
    }

    pub fn new_with_device(normalized_shape: usize, eps: f32, device: Device) -> Result<Self> {
        let weight = Tensor::ones(&[normalized_shape])?;
        Ok(Self {
            weight,
            eps,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Replace the learnable scale (used when loading pretrained weights).
    ///
    /// # Errors
    ///
    /// Never fails; the shape has already been checked by the caller's binder.
    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        self.weight = weight;
        Ok(())
    }

    /// Number of learnable parameters (the scale vector).
    pub fn parameter_count(&self) -> usize {
        self.weight.len()
    }
}

impl Layer for RMSNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // RMSNorm: x * weight / sqrt(mean(x^2) + eps)
        match &input {
            Tensor::F32(arr) => {
                let mean_sq = arr.iter().map(|x| x * x).sum::<f32>() / arr.len() as f32;
                let rms = (mean_sq + self.eps).sqrt();
                let normalized = arr.mapv(|x| x / rms);

                // Apply learnable weight
                match &self.weight {
                    Tensor::F32(weight_arr) => {
                        let result = &normalized * weight_arr;
                        Ok(Tensor::F32(result))
                    },
                    _ => Err(tensor_op_error(
                        "tensor_operation",
                        "Unsupported weight tensor type for RMSNorm",
                    )),
                }
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported input tensor type for RMSNorm",
            )),
        }
    }
}

/// Rotary Position Embedding (RoPE) for Phi-3
/// Enhanced implementation with LongRope support for extended context
pub struct RotaryEmbedding {
    pub dim: usize,
    pub max_seq_len: usize,
    pub base: f32,
    /// Base inverse frequencies `1 / base^(2i / dim)` for `i` in `0..dim/2`.
    pub inv_freq: Vec<f64>,
    /// Threshold (the model's *original* context length) beyond which the
    /// LongRope `long_factor` is used instead of `short_factor`.
    pub original_max_seq_len: usize,
    pub scaling_factor: Option<f32>,
    pub long_factor: Option<Vec<f32>>,
    pub short_factor: Option<Vec<f32>>,
    device: Device,
}

impl RotaryEmbedding {
    pub fn new(config: &Phi3Config) -> Self {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &Phi3Config, device: Device) -> Self {
        let dim = config.head_dim();

        let (scaling_factor, long_factor, short_factor) =
            if let Some(scaling) = &config.rope_scaling {
                (
                    Some(scaling.scaling_factor),
                    scaling.long_factor.clone(),
                    scaling.short_factor.clone(),
                )
            } else {
                (None, None, None)
            };

        // Standard RoPE base inverse frequencies: 1 / base^(2i / dim).
        let half = dim / 2;
        let base = config.rope_theta as f64;
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| {
                let exponent = 2.0 * i as f64 / dim as f64;
                1.0 / base.powf(exponent)
            })
            .collect();

        Self {
            dim,
            max_seq_len: config.max_position_embeddings,
            base: config.rope_theta,
            inv_freq,
            original_max_seq_len: config.original_max_position_embeddings,
            scaling_factor,
            long_factor,
            short_factor,
            device,
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Half the rotary dimension (number of rotation pairs).
    pub fn half_dim(&self) -> usize {
        self.inv_freq.len()
    }

    /// Effective per-pair inverse frequencies for a given sequence length.
    ///
    /// Phi-3 uses *LongRope*: each base inverse frequency is divided by a
    /// per-dimension rescaling factor. When the context exceeds the model's
    /// original training length the `long_factor` table is applied, otherwise
    /// the `short_factor` table is used. With no `rope_scaling` configured the
    /// base inverse frequencies are returned unchanged (vanilla RoPE).
    fn effective_inv_freq(&self, seq_len: usize) -> Vec<f64> {
        let factors = if seq_len > self.original_max_seq_len {
            self.long_factor.as_ref()
        } else {
            self.short_factor.as_ref()
        };

        match factors {
            Some(factor) if factor.len() == self.inv_freq.len() => self
                .inv_freq
                .iter()
                .zip(factor.iter())
                .map(|(freq, scale)| freq / (*scale as f64))
                .collect(),
            // No (or mismatched) rescaling table: fall back to vanilla RoPE.
            _ => self.inv_freq.clone(),
        }
    }

    /// Apply rotary position embeddings to `q` and `k` (shape-preserving) using
    /// the `rotate_half` convention, with Phi-3 LongRope frequency rescaling.
    ///
    /// Each input is `[seq, n_heads * head_dim]`; the number of heads is inferred
    /// from the projection width so the same routine serves the query and the
    /// (narrower) key projection. Within every head, dimension `i` and
    /// `i + head_dim/2` form a rotation pair driven by `position · inv_freq[i]`:
    ///
    /// ```text
    ///   out[i]        = x[i]·cos − x[i+half]·sin
    ///   out[i + half] = x[i+half]·cos + x[i]·sin
    /// ```
    pub fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = position_ids.len();
        let inv_freq = self.effective_inv_freq(seq_len);
        match (q, k) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr)) => Ok((
                Tensor::F32(self.rotate(q_arr, position_ids, &inv_freq)?),
                Tensor::F32(self.rotate(k_arr, position_ids, &inv_freq)?),
            )),
            _ => Err(tensor_op_error(
                "RotaryEmbedding::apply_rotary_emb",
                "Unsupported tensor types for RoPE",
            )),
        }
    }

    /// Rotate a single `[seq, n_heads * head_dim]` projection.
    fn rotate(
        &self,
        arr: &ArrayD<f32>,
        position_ids: &[usize],
        inv_freq: &[f64],
    ) -> Result<ArrayD<f32>> {
        let view = arr.view().into_dimensionality::<Ix2>().map_err(|_| {
            tensor_op_error(
                "RotaryEmbedding::rotate",
                "RoPE input must be a 2D [seq, n_heads * head_dim] tensor",
            )
        })?;
        let seq = view.shape()[0];
        let width = view.shape()[1];
        let head_dim = self.dim;
        let half = head_dim / 2;
        if head_dim == 0 || width % head_dim != 0 {
            return Err(tensor_op_error(
                "RotaryEmbedding::rotate",
                "projection width is not a multiple of head_dim",
            ));
        }
        let n_heads = width / head_dim;

        let mut out = Array2::<f32>::zeros((seq, width));
        for t in 0..seq {
            let pos = position_ids.get(t).copied().unwrap_or(t);
            for h in 0..n_heads {
                let base = h * head_dim;
                for i in 0..half {
                    let angle = pos as f64 * inv_freq[i];
                    let cos = angle.cos() as f32;
                    let sin = angle.sin() as f32;
                    let x1 = view[[t, base + i]];
                    let x2 = view[[t, base + i + half]];
                    out[[t, base + i]] = x1 * cos - x2 * sin;
                    out[[t, base + i + half]] = x2 * cos + x1 * sin;
                }
            }
        }
        Ok(out.into_dyn())
    }
}

/// Phi-3 Multi-Layer Perceptron with SwiGLU activation
/// Uses gated linear units for improved performance
pub struct Phi3MLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    hidden_act: ActivationType,
    device: Device,
}

impl Phi3MLP {
    pub fn new(config: &Phi3Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &Phi3Config, device: Device) -> Result<Self> {
        // Combined gate and up projection for efficiency
        let gate_up_proj = Linear::new_with_device(
            config.hidden_size,
            2 * config.intermediate_size, // Gate and up projections combined
            config.mlp_bias,
            device,
        );

        let down_proj = Linear::new_with_device(
            config.intermediate_size,
            config.hidden_size,
            config.mlp_bias,
            device,
        );

        Ok(Self {
            gate_up_proj,
            down_proj,
            // Parse the activation string once at construction. Unsupported
            // identifiers are rejected here (previously this error was raised on
            // every forward pass).
            hidden_act: ActivationType::try_from(config.hidden_act.as_str())?,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Number of learnable parameters in the two projections.
    pub fn parameter_count(&self) -> usize {
        self.gate_up_proj.parameter_count() + self.down_proj.parameter_count()
    }

    /// Copy the SwiGLU projections out of a checkpoint.
    ///
    /// Phi-3 stores the gate and up projections fused as `mlp.gate_up_proj`
    /// (`[2 * intermediate, hidden]`, gate first), which is exactly this layer's
    /// layout. Checkpoints that keep them apart as `mlp.gate_proj` / `mlp.up_proj`
    /// are accepted too and concatenated in the same order.
    ///
    /// # Errors
    ///
    /// Fails on a shape mismatch; absent tensors are recorded on the binder.
    pub fn load_weights(
        &mut self,
        binder: &mut WeightBinder<'_>,
        prefix: &str,
        config: &Phi3Config,
    ) -> Result<()> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;

        let fused = format!("{prefix}gate_up_proj.weight");
        if binder.has(&fused) || !binder.has(&format!("{prefix}gate_proj.weight")) {
            if let Some(weight) = binder.take_shaped(&fused, &[2 * intermediate, hidden])? {
                self.gate_up_proj.set_weight(weight)?;
            }
        } else {
            let gate = binder.take_shaped(
                &format!("{prefix}gate_proj.weight"),
                &[intermediate, hidden],
            )?;
            let up =
                binder.take_shaped(&format!("{prefix}up_proj.weight"), &[intermediate, hidden])?;
            if let (Some(gate), Some(up)) = (gate, up) {
                self.gate_up_proj.set_weight(concat_rows(&[&gate, &up])?)?;
            }
        }

        if config.mlp_bias {
            if let Some(bias) =
                binder.take_shaped(&format!("{prefix}gate_up_proj.bias"), &[2 * intermediate])?
            {
                self.gate_up_proj.set_bias(bias)?;
            }
        }

        if let Some(weight) = binder.take_shaped(
            &format!("{prefix}down_proj.weight"),
            &[hidden, intermediate],
        )? {
            self.down_proj.set_weight(weight)?;
        }
        if config.mlp_bias {
            if let Some(bias) = binder.take_shaped(&format!("{prefix}down_proj.bias"), &[hidden])? {
                self.down_proj.set_bias(bias)?;
            }
        }
        Ok(())
    }
}

/// Concatenate 2-D f32 tensors along their first (row) axis.
///
/// Used to fuse separately-stored `q/k/v` or `gate/up` projections into the
/// single matrix this implementation keeps, and to split them apart again.
fn concat_rows(parts: &[&Tensor]) -> Result<Tensor> {
    let mut columns = None;
    let mut rows = 0usize;
    let mut values: Vec<f32> = Vec::new();
    for part in parts {
        match part {
            Tensor::F32(arr) => {
                if arr.ndim() != 2 {
                    return Err(TrustformersError::shape_error(format!(
                        "expected a 2-D projection, got {} dimensions",
                        arr.ndim()
                    )));
                }
                let shape = arr.shape();
                match columns {
                    None => columns = Some(shape[1]),
                    Some(expected) if expected == shape[1] => {},
                    Some(expected) => {
                        return Err(TrustformersError::shape_error(format!(
                            "cannot concatenate projections with {expected} and {} columns",
                            shape[1]
                        )))
                    },
                }
                rows += shape[0];
                values.extend(arr.iter().copied());
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "projection concatenation requires F32 tensors",
                ))
            },
        }
    }
    let columns = columns.ok_or_else(|| {
        TrustformersError::shape_error("cannot concatenate an empty projection list".to_string())
    })?;
    Tensor::from_vec(values, &[rows, columns])
}

/// Take `count` consecutive elements of a 1-D f32 tensor, starting at `start`.
fn slice_vector(tensor: &Tensor, start: usize, count: usize) -> Result<Tensor> {
    match tensor {
        Tensor::F32(arr) => {
            let values: Vec<f32> = arr.iter().copied().collect();
            if start + count > values.len() {
                return Err(TrustformersError::shape_error(format!(
                    "cannot take elements {start}..{} from a vector of length {}",
                    start + count,
                    values.len()
                )));
            }
            Tensor::from_vec(values[start..start + count].to_vec(), &[count])
        },
        _ => Err(tensor_op_error(
            "tensor_operation",
            "bias slicing requires F32 tensors",
        )),
    }
}

/// Take `count` consecutive rows of a 2-D f32 tensor, starting at `start`.
fn slice_rows(tensor: &Tensor, start: usize, count: usize) -> Result<Tensor> {
    match tensor {
        Tensor::F32(arr) => {
            if arr.ndim() != 2 {
                return Err(TrustformersError::shape_error(format!(
                    "expected a 2-D projection, got {} dimensions",
                    arr.ndim()
                )));
            }
            let columns = arr.shape()[1];
            let rows = arr.shape()[0];
            if start + count > rows {
                return Err(TrustformersError::shape_error(format!(
                    "cannot take rows {start}..{} from a projection with {rows} rows",
                    start + count
                )));
            }
            let view = arr.clone().into_dimensionality::<Ix2>().map_err(|e| {
                TrustformersError::shape_error(format!("projection is not 2-D: {e}"))
            })?;
            let slice: Array2<f32> =
                view.slice(scirs2_core::ndarray::s![start..start + count, ..]).to_owned();
            Tensor::from_vec(slice.into_raw_vec_and_offset().0, &[count, columns])
        },
        _ => Err(tensor_op_error(
            "tensor_operation",
            "projection slicing requires F32 tensors",
        )),
    }
}

impl Layer for Phi3MLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Combined gate and up projection
        let gate_up = self.gate_up_proj.forward(input)?;

        // Split into gate and up parts
        let (gate, up) = match &gate_up {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                let intermediate_size = shape[shape.len() - 1] / 2;

                // Split the tensor along the last dimension
                let total_elements = arr.len();
                let batch_size = total_elements / (intermediate_size * 2);

                let arr_slice = arr.as_slice().unwrap_or_default();
                let mut gate_data = Vec::with_capacity(batch_size * intermediate_size);
                let mut up_data = Vec::with_capacity(batch_size * intermediate_size);

                // Split each batch's data
                for batch in 0..batch_size {
                    let batch_offset = batch * intermediate_size * 2;

                    // Gate projection (first half)
                    for i in 0..intermediate_size {
                        gate_data.push(arr_slice[batch_offset + i]);
                    }

                    // Up projection (second half)
                    for i in intermediate_size..(2 * intermediate_size) {
                        up_data.push(arr_slice[batch_offset + i]);
                    }
                }

                // Create output tensors with proper shapes
                let mut output_shape = shape.to_vec();
                let last_dim = output_shape.len() - 1;
                output_shape[last_dim] = intermediate_size;

                let gate_tensor = Tensor::from_vec(gate_data, &output_shape)?;
                let up_tensor = Tensor::from_vec(up_data, &output_shape)?;
                (gate_tensor, up_tensor)
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor type for MLP",
                ))
            },
        };

        // Apply activation to gate
        let activated_gate = self.hidden_act.apply(&gate)?;

        // Gated activation: gate * up
        let gated = match (&activated_gate, &up) {
            (Tensor::F32(gate_arr), Tensor::F32(up_arr)) => Tensor::F32(gate_arr * up_arr),
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Tensor type mismatch in gated activation",
                ))
            },
        };

        // Down projection
        self.down_proj.forward(gated)
    }
}

/// Phi-3 Attention layer with optional sliding window and grouped-query attention
#[allow(dead_code)]
pub struct Phi3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// `num_heads / num_kv_heads`: how many query heads share each KV head (GQA).
    num_query_groups: usize,
    sliding_window: Option<usize>,
    attention_dropout: f32,
    device: Device,
}

impl Phi3Attention {
    pub fn new(config: &Phi3Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &Phi3Config, device: Device) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_kv_heads = config.num_kv_heads();

        let q_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            config.attention_bias,
            device,
        );

        let k_proj = Linear::new_with_device(
            config.hidden_size,
            num_kv_heads * head_dim,
            config.attention_bias,
            device,
        );

        let v_proj = Linear::new_with_device(
            config.hidden_size,
            num_kv_heads * head_dim,
            config.attention_bias,
            device,
        );

        let o_proj = Linear::new_with_device(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            config.attention_bias,
            device,
        );

        let rotary_emb = RotaryEmbedding::new_with_device(config, device);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads: config.num_attention_heads,
            num_kv_heads,
            head_dim,
            num_query_groups: config.num_query_groups(),
            sliding_window: config.sliding_window,
            attention_dropout: config.attention_dropout,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Number of learnable parameters in the four projections.
    pub fn parameter_count(&self) -> usize {
        self.q_proj.parameter_count()
            + self.k_proj.parameter_count()
            + self.v_proj.parameter_count()
            + self.o_proj.parameter_count()
    }

    /// Copy the attention projections out of a checkpoint.
    ///
    /// Phi-3 fuses the three input projections into `self_attn.qkv_proj`
    /// (`[q_out + k_out + v_out, hidden]`, query rows first, then key, then
    /// value), which this implementation keeps as three separate matrices, so the
    /// fused tensor is split by row. Checkpoints that store `q_proj` / `k_proj` /
    /// `v_proj` separately are bound directly.
    ///
    /// # Errors
    ///
    /// Fails on a shape mismatch; absent tensors are recorded on the binder.
    pub fn load_weights(
        &mut self,
        binder: &mut WeightBinder<'_>,
        prefix: &str,
        config: &Phi3Config,
    ) -> Result<()> {
        let hidden = config.hidden_size;
        let head_dim = config.head_dim();
        let q_out = config.num_attention_heads * head_dim;
        let kv_out = config.num_kv_heads() * head_dim;

        let fused = format!("{prefix}qkv_proj.weight");
        if binder.has(&fused) {
            if let Some(weight) = binder.take_shaped(&fused, &[q_out + 2 * kv_out, hidden])? {
                self.q_proj.set_weight(slice_rows(&weight, 0, q_out)?)?;
                self.k_proj.set_weight(slice_rows(&weight, q_out, kv_out)?)?;
                self.v_proj.set_weight(slice_rows(&weight, q_out + kv_out, kv_out)?)?;
            }
            if config.attention_bias {
                if let Some(bias) =
                    binder.take_shaped(&format!("{prefix}qkv_proj.bias"), &[q_out + 2 * kv_out])?
                {
                    self.q_proj.set_bias(slice_vector(&bias, 0, q_out)?)?;
                    self.k_proj.set_bias(slice_vector(&bias, q_out, kv_out)?)?;
                    self.v_proj.set_bias(slice_vector(&bias, q_out + kv_out, kv_out)?)?;
                }
            }
        } else {
            if let Some(weight) =
                binder.take_shaped(&format!("{prefix}q_proj.weight"), &[q_out, hidden])?
            {
                self.q_proj.set_weight(weight)?;
            }
            if let Some(weight) =
                binder.take_shaped(&format!("{prefix}k_proj.weight"), &[kv_out, hidden])?
            {
                self.k_proj.set_weight(weight)?;
            }
            if let Some(weight) =
                binder.take_shaped(&format!("{prefix}v_proj.weight"), &[kv_out, hidden])?
            {
                self.v_proj.set_weight(weight)?;
            }
            if config.attention_bias {
                if let Some(bias) = binder.take_shaped(&format!("{prefix}q_proj.bias"), &[q_out])? {
                    self.q_proj.set_bias(bias)?;
                }
                if let Some(bias) =
                    binder.take_shaped(&format!("{prefix}k_proj.bias"), &[kv_out])?
                {
                    self.k_proj.set_bias(bias)?;
                }
                if let Some(bias) =
                    binder.take_shaped(&format!("{prefix}v_proj.bias"), &[kv_out])?
                {
                    self.v_proj.set_bias(bias)?;
                }
            }
        }

        if let Some(weight) =
            binder.take_shaped(&format!("{prefix}o_proj.weight"), &[hidden, q_out])?
        {
            self.o_proj.set_weight(weight)?;
        }
        if config.attention_bias {
            if let Some(bias) = binder.take_shaped(&format!("{prefix}o_proj.bias"), &[hidden])? {
                self.o_proj.set_bias(bias)?;
            }
        }
        Ok(())
    }

    /// Expand grouped key/value heads to match the query heads for GQA.
    ///
    /// Input is `[seq, num_kv_heads * head_dim]`; each KV head is repeated
    /// `num_query_groups` times contiguously, producing
    /// `[seq, num_heads * head_dim]`. For multi-head attention
    /// (`num_query_groups == 1`) the tensor is returned unchanged.
    fn repeat_kv(&self, kv: &Tensor) -> Result<Tensor> {
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
                            "Phi3Attention::repeat_kv",
                            format!("shape error during KV expansion: {e}"),
                        )
                    })?;
                Ok(Tensor::F32(expanded_arr))
            },
            _ => Err(tensor_op_error(
                "Phi3Attention::repeat_kv",
                "unsupported tensor dtype for KV expansion",
            )),
        }
    }
}

impl Layer for Phi3Attention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Determine the real sequence length from the input rank.
        // The Phi-3 model feeds a 2D `[seq, hidden]` tensor; a leading batch
        // dimension (`[1, seq, hidden]`) is also accepted.
        let shape = input.shape().to_vec();
        let seq_len = match shape.len() {
            2 => shape[0],
            3 => shape[1],
            n => {
                return Err(tensor_op_error(
                    "Phi3Attention::forward",
                    format!("unexpected input rank {n}"),
                ))
            },
        };

        // Project to Q, K, V.
        let q = self.q_proj.forward(input.clone())?; // [seq, num_heads    * head_dim]
        let k = self.k_proj.forward(input.clone())?; // [seq, num_kv_heads * head_dim]
        let v = self.v_proj.forward(input)?; // [seq, num_kv_heads * head_dim]

        // Flatten any leading batch dimension so RoPE / GQA operate on a
        // canonical 2D `[seq, width]` layout.
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let q = q.reshape(&[seq_len, num_heads * head_dim])?;
        let k = k.reshape(&[seq_len, self.num_kv_heads * head_dim])?;
        let v = v.reshape(&[seq_len, self.num_kv_heads * head_dim])?;

        // Rotary position embeddings (real positions 0..seq_len).
        let position_ids: Vec<usize> = (0..seq_len).collect();
        let (q_rope, k_rope) = self.rotary_emb.apply_rotary_emb(&q, &k, &position_ids)?;

        // Expand grouped key/value heads to match the query heads (GQA).
        let k_expanded = self.repeat_kv(&k_rope)?;
        let v_expanded = self.repeat_kv(&v)?;

        // [seq, num_heads * head_dim] -> [1, num_heads, seq, head_dim].
        let to_heads = |t: &Tensor| -> Result<Tensor> {
            t.reshape(&[1, seq_len, num_heads, head_dim])?.transpose(1, 2)
        };
        let q_h = to_heads(&q_rope)?;
        let k_h = to_heads(&k_expanded)?;
        let v_h = to_heads(&v_expanded)?;

        // Scaled dot-product scores: [1, num_heads, seq, seq].
        let scale = (head_dim as f32).sqrt().recip();
        let scores = q_h.matmul(&k_h.transpose(2, 3)?)?.mul_scalar(scale)?;

        // Additive causal mask (optionally narrowed to a sliding window),
        // softmax over the key axis, then weight the values.
        let scores = scores.add(&self.attention_mask(seq_len)?)?;
        let weights = scores.softmax(-1)?;
        let context = weights.matmul(&v_h)?; // [1, num_heads, seq, head_dim]

        // [1, num_heads, seq, head_dim] -> [seq, num_heads * head_dim].
        let context = context.transpose(1, 2)?.reshape(&[seq_len, num_heads * head_dim])?;

        self.o_proj.forward(context)
    }
}

impl Phi3Attention {
    /// Build the additive attention mask of shape `[1, 1, seq, seq]`.
    ///
    /// Positions strictly above the diagonal (the future) are set to a large
    /// negative value so they vanish under softmax. When `sliding_window` is
    /// configured, positions farther than the window into the past are masked
    /// out as well, matching Phi-3's local-attention variants.
    fn attention_mask(&self, seq_len: usize) -> Result<Tensor> {
        let mut mask = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let masked = j > i
                    || self.sliding_window.is_some_and(|window| i.saturating_sub(j) >= window);
                if masked {
                    mask[i * seq_len + j] = -1.0e9;
                }
            }
        }
        Tensor::from_vec(mask, &[seq_len, seq_len])?.reshape(&[1, 1, seq_len, seq_len])
    }
}

/// Phi-3 Decoder Layer
pub struct Phi3DecoderLayer {
    self_attn: Phi3Attention,
    mlp: Phi3MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    device: Device,
}

impl Phi3DecoderLayer {
    pub fn new(config: &Phi3Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &Phi3Config, device: Device) -> Result<Self> {
        let self_attn = Phi3Attention::new_with_device(config, device)?;
        let mlp = Phi3MLP::new_with_device(config, device)?;
        let input_layernorm =
            RMSNorm::new_with_device(config.hidden_size, config.rms_norm_eps, device)?;
        let post_attention_layernorm =
            RMSNorm::new_with_device(config.hidden_size, config.rms_norm_eps, device)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Number of learnable parameters in this decoder layer.
    pub fn parameter_count(&self) -> usize {
        self.self_attn.parameter_count()
            + self.mlp.parameter_count()
            + self.input_layernorm.parameter_count()
            + self.post_attention_layernorm.parameter_count()
    }

    /// Copy one decoder layer's parameters out of a checkpoint.
    ///
    /// # Errors
    ///
    /// Fails on a shape mismatch; absent tensors are recorded on the binder.
    pub fn load_weights(
        &mut self,
        binder: &mut WeightBinder<'_>,
        layer_prefix: &str,
        config: &Phi3Config,
    ) -> Result<()> {
        self.self_attn
            .load_weights(binder, &format!("{layer_prefix}self_attn."), config)?;
        self.mlp.load_weights(binder, &format!("{layer_prefix}mlp."), config)?;

        if let Some(weight) = binder.take_shaped(
            &format!("{layer_prefix}input_layernorm.weight"),
            &[config.hidden_size],
        )? {
            self.input_layernorm.set_weight(weight)?;
        }
        if let Some(weight) = binder.take_shaped(
            &format!("{layer_prefix}post_attention_layernorm.weight"),
            &[config.hidden_size],
        )? {
            self.post_attention_layernorm.set_weight(weight)?;
        }
        Ok(())
    }
}

impl Layer for Phi3DecoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Pre-attention normalization
        let normed_input = self.input_layernorm.forward(input.clone())?;

        // Self-attention with residual connection
        let attn_output = self.self_attn.forward(normed_input)?;
        let hidden_states = input.add(&attn_output)?;

        // Pre-MLP normalization
        let normed_hidden = self.post_attention_layernorm.forward(hidden_states.clone())?;

        // MLP with residual connection
        let mlp_output = self.mlp.forward(normed_hidden)?;
        hidden_states.add(&mlp_output)
    }
}

/// Phi-3 Model (base model without task-specific head)
pub struct Phi3Model {
    config: Phi3Config,
    embed_tokens: Embedding,
    layers: Vec<Phi3DecoderLayer>,
    norm: RMSNorm,
    device: Device,
}

impl Phi3Model {
    pub fn new(config: Phi3Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: Phi3Config, device: Device) -> Result<Self> {
        config.validate()?;

        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(Phi3DecoderLayer::new_with_device(&config, device)?);
        }

        let norm = RMSNorm::new_with_device(config.hidden_size, config.rms_norm_eps, device)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            device,
        })
    }

    pub fn config(&self) -> &Phi3Config {
        &self.config
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for Phi3Model {
    type Config = Phi3Config;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        // Convert tensor to token IDs
        let token_ids = match &input_ids {
            Tensor::I64(arr) => arr.as_slice().unwrap_or(&[]).iter().map(|&x| x as u32).collect(),
            Tensor::F32(arr) => {
                // Convert f32 to token IDs by rounding
                arr.as_slice().unwrap_or(&[]).iter().map(|&x| x.round() as u32).collect()
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor type for input_ids",
                ))
            },
        };

        // Token embeddings
        let mut hidden_states = self.embed_tokens.forward(token_ids)?;

        // Apply transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        // Final normalization
        self.norm.forward(hidden_states)
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.parameter_count()
    }
}

impl Phi3Model {
    /// Checkpoint namespaces a base Phi-3 model legitimately leaves unused.
    ///
    /// A `Phi3ForCausalLM` export carries the untied LM head next to the
    /// backbone; loading only the backbone leaves it unconsumed.
    const ALLOWED_UNUSED_PREFIXES: &'static [&'static str] = &["lm_head.", "score.", "classifier."];

    /// Per-layer buffers HuggingFace stores alongside Phi-3's weights.
    ///
    /// The rotary tables are derived from the configuration, not trained, and
    /// they repeat under every layer prefix — so they are matched by suffix
    /// rather than by namespace.
    const ALLOWED_UNUSED_SUFFIXES: &'static [&'static str] = &[
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
        "attn.bias",
        "attn.masked_bias",
    ];

    /// The unused-tensor policy for a bare Phi-3 backbone load.
    fn unused_tensor_policy() -> UnusedTensors<'static> {
        UnusedTensors::new(Self::ALLOWED_UNUSED_PREFIXES, Self::ALLOWED_UNUSED_SUFFIXES)
    }

    /// Total parameter count, summed from the live layers.
    pub fn parameter_count(&self) -> usize {
        self.embed_tokens.parameter_count()
            + self.layers.iter().map(|layer| layer.parameter_count()).sum::<usize>()
            + self.norm.parameter_count()
    }

    /// Load pretrained weights and report exactly what was bound.
    ///
    /// The stream may hold a safetensors file or a PyTorch archive; both are
    /// parsed for real by [`Checkpoint::from_reader`].
    ///
    /// A previous revision printed `"Assigning mock tensors for demonstration..."`,
    /// walked the tensor names printing each one, assigned nothing, and returned
    /// `Ok(())`. Even the safetensors branch never touched the tensor payload.
    ///
    /// # Errors
    ///
    /// Fails when the container cannot be parsed, when the checkpoint is not a
    /// Phi-3 checkpoint, when a tensor has the wrong shape, or when any parameter
    /// is missing.
    pub fn load_pretrained_report(&mut self, reader: &mut dyn Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.load_from_checkpoint(&checkpoint)
    }

    /// Bind an already-parsed checkpoint into this backbone.
    ///
    /// # Errors
    ///
    /// See [`Phi3Model::load_pretrained_report`].
    pub fn load_from_checkpoint(&mut self, checkpoint: &Checkpoint) -> Result<LoadReport> {
        // `Phi3ForCausalLM` nests the backbone under `model.`; a bare backbone
        // export does not.
        let prefix = checkpoint.detect_prefix(&["", "model."], "embed_tokens.weight")?;
        let mut binder = checkpoint.binder(&prefix);
        self.bind_weights(&mut binder, "")?;
        binder.finish(Self::unused_tensor_policy())
    }

    /// Bind the backbone's parameters through an existing binder.
    ///
    /// `inner_prefix` is prepended to every logical name, so a causal-LM loader
    /// can bind `model.` while keeping `lm_head.*` on the same binder.
    ///
    /// # Errors
    ///
    /// Fails on a shape mismatch; absent tensors are recorded on the binder.
    pub fn bind_weights(
        &mut self,
        binder: &mut WeightBinder<'_>,
        inner_prefix: &str,
    ) -> Result<()> {
        let config = self.config.clone();

        if let Some(weight) = binder.take_shaped(
            &format!("{inner_prefix}embed_tokens.weight"),
            &[config.vocab_size, config.hidden_size],
        )? {
            self.embed_tokens.set_weight(weight)?;
        }

        for (index, layer) in self.layers.iter_mut().enumerate() {
            let layer_prefix = format!("{inner_prefix}layers.{index}.");
            layer.load_weights(binder, &layer_prefix, &config)?;
        }

        if let Some(weight) =
            binder.take_shaped(&format!("{inner_prefix}norm.weight"), &[config.hidden_size])?
        {
            self.norm.set_weight(weight)?;
        }
        Ok(())
    }

    /// Number of decoder layers this model was built with.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// Phi-3 Model for Causal Language Modeling
pub struct Phi3ForCausalLM {
    model: Phi3Model,
    lm_head: Linear,
    device: Device,
}

impl Phi3ForCausalLM {
    pub fn new(config: Phi3Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: Phi3Config, device: Device) -> Result<Self> {
        let model = Phi3Model::new_with_device(config.clone(), device)?;
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);

        Ok(Self {
            model,
            lm_head,
            device,
        })
    }

    pub fn config(&self) -> &Phi3Config {
        self.model.config()
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Phi3ForCausalLM {
    /// The backbone this head sits on.
    pub fn model(&self) -> &Phi3Model {
        &self.model
    }

    /// Mutable access to the backbone.
    pub fn model_mut(&mut self) -> &mut Phi3Model {
        &mut self.model
    }

    /// Number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.model.num_layers()
    }
}

impl Model for Phi3ForCausalLM {
    type Config = Phi3Config;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        let hidden_states = self.model.forward(input_ids)?;
        self.lm_head.forward(hidden_states)
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        &self.model.config
    }

    fn num_parameters(&self) -> usize {
        self.model.parameter_count() + self.lm_head.parameter_count()
    }
}

impl Phi3ForCausalLM {
    /// Checkpoint namespaces a causal-LM export legitimately leaves unused.
    const ALLOWED_UNUSED_PREFIXES: &'static [&'static str] = &["score.", "classifier."];

    /// The unused-tensor policy for a Phi-3 causal-LM load.
    fn unused_tensor_policy() -> UnusedTensors<'static> {
        UnusedTensors::new(
            Self::ALLOWED_UNUSED_PREFIXES,
            Phi3Model::ALLOWED_UNUSED_SUFFIXES,
        )
    }

    /// Load pretrained weights and report exactly what was bound.
    ///
    /// # Errors
    ///
    /// Fails when the container cannot be parsed, when the checkpoint is not a
    /// Phi-3 checkpoint, when a tensor has the wrong shape, or when any parameter
    /// is missing.
    pub fn load_pretrained_report(&mut self, reader: &mut dyn Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.load_from_checkpoint(&checkpoint)
    }

    /// Bind an already-parsed checkpoint into the backbone and the LM head.
    ///
    /// # Errors
    ///
    /// See [`Phi3ForCausalLM::load_pretrained_report`].
    pub fn load_from_checkpoint(&mut self, checkpoint: &Checkpoint) -> Result<LoadReport> {
        // The backbone sits under `model.` in a HuggingFace causal-LM export and
        // at the root in a bare backbone export; the LM head is always at the root.
        let inner_prefix = if checkpoint.contains("model.embed_tokens.weight") {
            "model."
        } else if checkpoint.contains("embed_tokens.weight") {
            ""
        } else {
            return Err(TrustformersError::weight_load_error(
                "checkpoint has neither model.embed_tokens.weight nor embed_tokens.weight, so it \
                 is not a Phi-3 checkpoint"
                    .to_string(),
            ));
        };

        let config = self.model.config.clone();
        let mut binder = checkpoint.binder("");
        self.model.bind_weights(&mut binder, inner_prefix)?;

        // `tie_word_embeddings` models ship no separate head; the embedding table
        // is reused instead of leaving a randomly-initialised projection in place.
        if binder.has("lm_head.weight") {
            if let Some(weight) =
                binder.take_shaped("lm_head.weight", &[config.vocab_size, config.hidden_size])?
            {
                self.lm_head.set_weight(weight)?;
            }
        } else {
            let tied = format!("{inner_prefix}embed_tokens.weight");
            let embeddings = checkpoint.get(&tied).ok_or_else(|| {
                TrustformersError::weight_load_error(format!(
                    "checkpoint has no lm_head.weight and no {tied} to tie it to"
                ))
            })?;
            self.lm_head.set_weight(embeddings.clone())?;
            binder.mark_consumed(&tied);
        }

        binder.finish(Self::unused_tensor_policy())
    }
}

// Helper for tensor slicing (would normally be imported)
// SciRS2 Integration Policy

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phi3::config::Phi3Config;

    /// Tiny grouped-query-attention config used to exercise the real attention
    /// path without allocating a full-size model.
    ///
    /// `hidden_size = 16`, `num_attention_heads = 4` (so `head_dim = 4`) and
    /// `num_key_value_heads = 2` give two query groups, forcing the GQA
    /// `repeat_kv` expansion to run.
    fn tiny_config() -> Phi3Config {
        Phi3Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            max_position_embeddings: 64,
            original_max_position_embeddings: 64,
            ..Phi3Config::default()
        }
    }

    fn all_finite(tensor: &Tensor) -> bool {
        match tensor {
            Tensor::F32(arr) => arr.iter().all(|x| x.is_finite()),
            Tensor::F64(arr) => arr.iter().all(|x| x.is_finite()),
            _ => false,
        }
    }

    #[test]
    fn test_rotary_emb_actually_rotates() {
        // A non-zero, position-dependent input must come back *changed* — proving
        // the rotary embedding is a real rotation and not an identity pass-through.
        let cfg = tiny_config();
        let rope = RotaryEmbedding::new(&cfg);
        let head_dim = cfg.head_dim();
        let seq = 3usize;
        // Single head per row so the width equals head_dim.
        let data: Vec<f32> = (0..seq * head_dim).map(|i| 0.1 + i as f32 * 0.05).collect();
        let q = Tensor::from_vec(data.clone(), &[seq, head_dim]).expect("q tensor");
        let k = q.clone();
        let pos: Vec<usize> = (0..seq).collect();
        let (q_rot, _k_rot) = rope.apply_rotary_emb(&q, &k, &pos).expect("rope must run");

        assert_eq!(q_rot.shape(), &[seq, head_dim], "RoPE must preserve shape");

        // Row 0 has position 0 (angle 0 => identity); later rows must differ.
        if let (Tensor::F32(input), Tensor::F32(output)) = (&q, &q_rot) {
            let changed = input.iter().zip(output.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
            assert!(changed, "RoPE must actually rotate (not an identity op)");
        } else {
            panic!("expected F32 tensors");
        }
    }

    #[test]
    fn test_attention_forward_shape_and_finite() {
        let cfg = tiny_config();
        let attn = Phi3Attention::new(&cfg).expect("attention must construct");
        let seq = 5usize;
        let hidden = cfg.hidden_size;
        let input_data: Vec<f32> =
            (0..seq * hidden).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let input = Tensor::from_vec(input_data, &[seq, hidden]).expect("input tensor");

        let out = attn.forward(input).expect("attention forward must succeed");
        assert_eq!(
            out.shape(),
            &[seq, hidden],
            "attention output must be [seq, hidden]"
        );
        assert!(
            all_finite(&out),
            "attention output must be finite (no NaN/Inf)"
        );
    }

    #[test]
    fn test_causal_lm_forward_shape_and_finite() {
        // Run the full model forward through the real attention path and confirm
        // the logits have the right shape and contain no NaN/Inf.
        let cfg = tiny_config();
        let vocab = cfg.vocab_size;
        let model = Phi3ForCausalLM::new(cfg).expect("causal LM must construct");

        let token_ids: Vec<i64> = vec![1, 5, 9, 2];
        let seq = token_ids.len();
        let input = Tensor::from_vec_i64(token_ids, &[seq]).expect("token tensor");

        let logits = model.forward(input).expect("forward must succeed");
        assert_eq!(
            *logits.shape().last().expect("logits have a shape"),
            vocab,
            "causal LM output last dim must be vocab_size"
        );
        assert!(all_finite(&logits), "logits must be finite (no NaN/Inf)");
    }
    // --- Real weight loading (regression for the mock-tensor "loader") ---

    use crate::weight_loading::test_support::{build_safetensors, F32Tensor};

    /// Every tensor a Phi-3 checkpoint of `config`'s shape must supply.
    ///
    /// Built from the same names the binder resolves, with a distinct
    /// deterministic ramp per tensor.
    fn phi3_fixture(config: &Phi3Config, prefix: &str, fused: bool) -> Vec<F32Tensor> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let head_dim = config.head_dim();
        let q_out = config.num_attention_heads * head_dim;
        let kv_out = config.num_kv_heads() * head_dim;

        let mut seed = 0.0f32;
        let mut next_seed = || {
            seed += 1.0;
            seed
        };

        let mut tensors = vec![F32Tensor::ramp(
            &format!("{prefix}embed_tokens.weight"),
            &[config.vocab_size, hidden],
            next_seed(),
        )];

        for layer in 0..config.num_hidden_layers {
            let p = format!("{prefix}layers.{layer}.");
            if fused {
                tensors.push(F32Tensor::ramp(
                    &format!("{p}self_attn.qkv_proj.weight"),
                    &[q_out + 2 * kv_out, hidden],
                    next_seed(),
                ));
            } else {
                tensors.push(F32Tensor::ramp(
                    &format!("{p}self_attn.q_proj.weight"),
                    &[q_out, hidden],
                    next_seed(),
                ));
                tensors.push(F32Tensor::ramp(
                    &format!("{p}self_attn.k_proj.weight"),
                    &[kv_out, hidden],
                    next_seed(),
                ));
                tensors.push(F32Tensor::ramp(
                    &format!("{p}self_attn.v_proj.weight"),
                    &[kv_out, hidden],
                    next_seed(),
                ));
            }
            tensors.push(F32Tensor::ramp(
                &format!("{p}self_attn.o_proj.weight"),
                &[hidden, q_out],
                next_seed(),
            ));
            if fused {
                tensors.push(F32Tensor::ramp(
                    &format!("{p}mlp.gate_up_proj.weight"),
                    &[2 * intermediate, hidden],
                    next_seed(),
                ));
            } else {
                tensors.push(F32Tensor::ramp(
                    &format!("{p}mlp.gate_proj.weight"),
                    &[intermediate, hidden],
                    next_seed(),
                ));
                tensors.push(F32Tensor::ramp(
                    &format!("{p}mlp.up_proj.weight"),
                    &[intermediate, hidden],
                    next_seed(),
                ));
            }
            tensors.push(F32Tensor::ramp(
                &format!("{p}mlp.down_proj.weight"),
                &[hidden, intermediate],
                next_seed(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{p}input_layernorm.weight"),
                &[hidden],
                next_seed(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{p}post_attention_layernorm.weight"),
                &[hidden],
                next_seed(),
            ));
        }

        tensors.push(F32Tensor::ramp(
            &format!("{prefix}norm.weight"),
            &[hidden],
            next_seed(),
        ));
        tensors
    }

    fn phi3_hidden(model: &Phi3Model) -> Vec<f32> {
        let ids = Tensor::from_vec_i64(vec![1, 2, 3], &[3]).expect("input ids");
        match model.forward(ids).expect("forward must succeed after loading") {
            Tensor::F32(arr) => arr.iter().copied().collect(),
            other => panic!("expected an F32 hidden state, got {other:?}"),
        }
    }

    #[test]
    fn phi3_load_pretrained_makes_the_model_determined_by_the_checkpoint() {
        // The old loader printed tensor names, assigned nothing and reported
        // success, so the model stayed randomly initialised. Two independently
        // initialised models must now agree exactly after loading.
        let config = tiny_config();
        let bytes = build_safetensors(&phi3_fixture(&config, "", true));

        let mut first = Phi3Model::new(config.clone()).expect("model must build");
        let mut second = Phi3Model::new(config).expect("model must build");
        assert_ne!(
            phi3_hidden(&first),
            phi3_hidden(&second),
            "two random initialisations must differ"
        );

        first.load_pretrained(&mut bytes.as_slice()).expect("checkpoint must load");
        second.load_pretrained(&mut bytes.as_slice()).expect("checkpoint must load");

        assert_eq!(
            phi3_hidden(&first),
            phi3_hidden(&second),
            "after loading, both models must be the checkpoint's model"
        );
    }

    #[test]
    fn phi3_fused_qkv_is_split_into_the_three_projections() {
        // The fused Phi-3 `qkv_proj` holds the query rows first, then key, then
        // value. Loading it and re-reading each projection must reproduce exactly
        // those row ranges.
        let config = tiny_config();
        let tensors = phi3_fixture(&config, "", true);
        let fused = tensors
            .iter()
            .find(|t| t.name == "layers.0.self_attn.qkv_proj.weight")
            .expect("fixture must hold the fused projection")
            .clone();
        let bytes = build_safetensors(&tensors);

        let mut model = Phi3Model::new(config.clone()).expect("model must build");
        model.load_pretrained(&mut bytes.as_slice()).expect("checkpoint must load");

        let hidden = config.hidden_size;
        let head_dim = config.head_dim();
        let q_out = config.num_attention_heads * head_dim;
        let kv_out = config.num_kv_heads() * head_dim;

        let layer = &model.layers[0];
        let expect_rows = |tensor: &Tensor, start: usize, count: usize, label: &str| {
            let expected = &fused.values[start * hidden..(start + count) * hidden];
            match tensor {
                Tensor::F32(arr) => {
                    assert_eq!(arr.shape(), &[count, hidden], "{label} has the wrong shape");
                    assert_eq!(
                        arr.iter().copied().collect::<Vec<f32>>(),
                        expected.to_vec(),
                        "{label} did not receive its rows of the fused projection"
                    );
                },
                other => panic!("expected an F32 tensor for {label}, got {other:?}"),
            }
        };
        expect_rows(layer.self_attn.q_proj.weight(), 0, q_out, "q_proj");
        expect_rows(layer.self_attn.k_proj.weight(), q_out, kv_out, "k_proj");
        expect_rows(
            layer.self_attn.v_proj.weight(),
            q_out + kv_out,
            kv_out,
            "v_proj",
        );
    }

    #[test]
    fn phi3_accepts_unfused_q_k_v_and_gate_up_projections() {
        let config = tiny_config();
        let bytes = build_safetensors(&phi3_fixture(&config, "", false));
        let mut model = Phi3Model::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("an unfused checkpoint must load");
        assert!(report.is_complete());
        assert!(report.unexpected.is_empty());
    }

    #[test]
    fn phi3_load_pretrained_reports_missing_tensors() {
        let config = tiny_config();
        let mut tensors = phi3_fixture(&config, "", true);
        tensors.retain(|t| t.name != "layers.1.mlp.down_proj.weight");
        let bytes = build_safetensors(&tensors);

        let mut model = Phi3Model::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an incomplete checkpoint must not load silently");
        assert!(
            err.to_string().contains("layers.1.mlp.down_proj.weight"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn phi3_load_pretrained_rejects_unknown_tensors() {
        let config = tiny_config();
        let mut tensors = phi3_fixture(&config, "", true);
        tensors.push(F32Tensor::ramp(
            "layers.0.self_attn.rotary_emb.mystery",
            &[4],
            1.0,
        ));
        let bytes = build_safetensors(&tensors);

        let mut model = Phi3Model::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an unrecognised tensor must fail the load");
        assert!(
            err.to_string().contains("rotary_emb.mystery"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn phi3_causal_lm_loads_the_backbone_and_the_head() {
        let config = tiny_config();
        let mut tensors = phi3_fixture(&config, "model.", true);
        let head = F32Tensor::ramp(
            "lm_head.weight",
            &[config.vocab_size, config.hidden_size],
            99.0,
        );
        tensors.push(head.clone());
        let bytes = build_safetensors(&tensors);

        let mut model = Phi3ForCausalLM::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a causal-LM checkpoint must load");
        assert!(report.is_complete());
        assert!(report.unexpected.is_empty());

        match model.lm_head.weight() {
            Tensor::F32(arr) => {
                assert_eq!(arr.iter().copied().collect::<Vec<f32>>(), head.values);
            },
            other => panic!("expected an F32 LM head, got {other:?}"),
        }
    }

    #[test]
    fn phi3_causal_lm_ties_the_head_to_the_embeddings_when_absent() {
        let config = tiny_config();
        let tensors = phi3_fixture(&config, "model.", true);
        let embeddings = tensors
            .iter()
            .find(|t| t.name == "model.embed_tokens.weight")
            .expect("fixture must hold the embedding table")
            .clone();
        let bytes = build_safetensors(&tensors);

        let mut model = Phi3ForCausalLM::new(config).expect("model must build");
        model
            .load_pretrained(&mut bytes.as_slice())
            .expect("a tied-embedding checkpoint must load");

        match model.lm_head.weight() {
            Tensor::F32(arr) => {
                assert_eq!(arr.iter().copied().collect::<Vec<f32>>(), embeddings.values);
            },
            other => panic!("expected an F32 LM head, got {other:?}"),
        }
    }

    #[test]
    fn phi3_load_pretrained_rejects_bytes_that_are_not_a_checkpoint() {
        let config = tiny_config();
        let mut model = Phi3Model::new(config).expect("model must build");
        let garbage = vec![0x5A_u8; 8192];
        let err = model
            .load_pretrained(&mut garbage.as_slice())
            .expect_err("garbage must not be accepted as weights");
        assert!(
            err.to_string().contains("unrecognised checkpoint container"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn phi3_num_parameters_is_summed_from_the_live_layers() {
        let config = tiny_config();
        let model = Phi3Model::new(config.clone()).expect("model must build");
        let head_dim = config.head_dim();
        let q_out = config.num_attention_heads * head_dim;
        let kv_out = config.num_kv_heads() * head_dim;
        let per_layer = q_out * config.hidden_size
            + 2 * kv_out * config.hidden_size
            + config.hidden_size * q_out
            + 2 * config.intermediate_size * config.hidden_size
            + config.hidden_size * config.intermediate_size
            + 2 * config.hidden_size;
        let expected = config.vocab_size * config.hidden_size
            + config.num_hidden_layers * per_layer
            + config.hidden_size;
        assert_eq!(model.num_parameters(), expected);
    }
}
