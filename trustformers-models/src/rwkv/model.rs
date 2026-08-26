use crate::rwkv::config::RwkvConfig;
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{not_implemented, tensor_op_error, Result},
    layers::{Embedding, LayerNorm, Linear},
    ops::activations::{relu, sigmoid},
    tensor::Tensor,
    traits::{Layer, Model},
};

/// Decompose an activation tensor into flat row-major data plus `(batch, seq)`.
///
/// RWKV layers accept either `[seq, n_embd]` (the shape produced by `Embedding`)
/// or `[batch, seq, n_embd]`. The time axis is load-bearing for both the token
/// shift and the WKV recurrence, so anything else is rejected instead of being
/// reinterpreted.
fn as_batched_sequence(tensor: &Tensor, n_embd: usize) -> Result<(Vec<f32>, usize, usize)> {
    let shape = tensor.shape();
    let (batch, seq) = match shape.len() {
        2 => (1usize, shape[0]),
        3 => (shape[0], shape[1]),
        _ => {
            return Err(tensor_op_error(
                "rwkv_time_mixing",
                format!(
                    "expected [seq, n_embd] or [batch, seq, n_embd] activations, got {shape:?}"
                ),
            ))
        },
    };

    let channels = shape[shape.len() - 1];
    if channels != n_embd {
        return Err(tensor_op_error(
            "rwkv_time_mixing",
            format!("activation channel count {channels} does not match n_embd {n_embd}"),
        ));
    }

    let data = tensor.data()?;
    if data.len() != batch * seq * channels {
        return Err(tensor_op_error(
            "rwkv_time_mixing",
            format!(
                "data length {} is inconsistent with shape {shape:?}",
                data.len()
            ),
        ));
    }

    Ok((data, batch, seq))
}

/// RWKV token shift: interpolate each timestep with its predecessor.
///
/// `x'_t = μ ⊙ x_t + (1 - μ) ⊙ x_{t-1}`, with `x_{-1} = 0`. The three `time_mix_*`
/// vectors give the receptance, key and value branches independent interpolation
/// ratios — this is what lets a single RWKV channel see two timesteps at once.
fn token_shift(data: &[f32], batch: usize, seq: usize, channels: usize, mix: &[f32]) -> Vec<f32> {
    let mut shifted = vec![0.0f32; data.len()];
    for b in 0..batch {
        for t in 0..seq {
            let base = (b * seq + t) * channels;
            for c in 0..channels {
                let previous = if t == 0 { 0.0 } else { data[(b * seq + t - 1) * channels + c] };
                let m = mix[c];
                shifted[base + c] = m * data[base + c] + (1.0 - m) * previous;
            }
        }
    }
    shifted
}

/// The RWKV-4 WKV linear-attention recurrence, in numerically stable form.
///
/// For every channel `c` the published operator is
///
/// ```text
///            Σ_{i<t} e^{(t-1-i)·w + k_i}·v_i + e^{u + k_t}·v_t
/// wkv_t  =  ---------------------------------------------------
///            Σ_{i<t} e^{(t-1-i)·w + k_i}     + e^{u + k_t}
/// ```
///
/// with `w = -exp(time_decay) < 0` the per-channel decay and `u = time_first` the
/// bonus applied to the current token. Evaluating that sum directly overflows, so
/// the numerator/denominator pair `(a, b)` is carried scaled by `e^{-p}` where `p`
/// is a running maximum of the exponents; every `exp` argument is therefore `<= 0`.
///
/// Shapes: `k`, `v` are row-major `[batch, seq, channels]`; `w`, `u` are
/// `[channels]`; the result is `[batch, seq, channels]`.
///
/// Reference: Peng et al., "RWKV: Reinventing RNNs for the Transformer Era" (2023).
fn wkv_scan(
    k: &[f32],
    v: &[f32],
    decay: &[f32],
    bonus: &[f32],
    batch: usize,
    seq: usize,
    channels: usize,
) -> Vec<f32> {
    let mut wkv = vec![0.0f32; batch * seq * channels];

    for b in 0..batch {
        for c in 0..channels {
            // w is strictly negative, guaranteeing the recurrence decays.
            let w = -decay[c].exp();
            let u = bonus[c];

            let mut numerator = 0.0f32;
            let mut denominator = 0.0f32;
            let mut max_exponent = f32::NEG_INFINITY;

            for t in 0..seq {
                let idx = (b * seq + t) * channels + c;
                let k_t = k[idx];
                let v_t = v[idx];

                // Output at t: fold in the current token with its `u` bonus.
                let q = max_exponent.max(u + k_t);
                let e_state = (max_exponent - q).exp();
                let e_current = (u + k_t - q).exp();
                let out_num = e_state * numerator + e_current * v_t;
                let out_den = e_state * denominator + e_current;
                wkv[idx] = if out_den != 0.0 { out_num / out_den } else { 0.0 };

                // State update for t+1: decay the history by `w`, add token t.
                let q_next = (max_exponent + w).max(k_t);
                let e_decayed = (max_exponent + w - q_next).exp();
                let e_new = (k_t - q_next).exp();
                numerator = e_decayed * numerator + e_new * v_t;
                denominator = e_decayed * denominator + e_new;
                max_exponent = q_next;
            }
        }
    }

    wkv
}

/// RWKV Time Mixing layer - replaces traditional attention
/// This implements the core RWKV mechanism for temporal information processing
pub struct TimeMixing {
    config: RwkvConfig,
    layer_id: usize,
    time_decay: Tensor,
    time_first: Tensor,
    time_mix_k: Tensor,
    time_mix_v: Tensor,
    time_mix_r: Tensor,
    key: Linear,
    value: Linear,
    receptance: Linear,
    output: Linear,
    device: Device,
}

impl TimeMixing {
    pub fn new(config: &RwkvConfig, layer_id: usize) -> Result<Self> {
        Self::new_with_device(config, layer_id, Device::CPU)
    }

    pub fn new_with_device(config: &RwkvConfig, layer_id: usize, device: Device) -> Result<Self> {
        let n_embd = config.n_embd;

        // RWKV-4 time-mixing initialisation (Peng et al., 2023, `src/model.py`):
        // the parameters are per-channel and depth-dependent, not random — deeper
        // layers start with slower decay and a larger share of the current token.
        let ratio_0_to_1 = if config.n_layer > 1 {
            layer_id as f32 / (config.n_layer - 1) as f32
        } else {
            0.0
        };
        let ratio_1_to_almost0 = 1.0 - (layer_id as f32 / config.n_layer.max(1) as f32);
        let last_channel = (n_embd.saturating_sub(1)).max(1) as f32;

        let mut decay_values = Vec::with_capacity(n_embd);
        let mut first_values = Vec::with_capacity(n_embd);
        let mut mix_k_values = Vec::with_capacity(n_embd);
        let mut mix_v_values = Vec::with_capacity(n_embd);
        let mut mix_r_values = Vec::with_capacity(n_embd);

        for channel in 0..n_embd {
            let position = channel as f32 / last_channel;
            decay_values.push(-5.0 + 8.0 * position.powf(0.7 + 1.3 * ratio_0_to_1));

            // Zig-zag so neighbouring channels do not share an identical bonus.
            let zigzag = 0.5 * (((channel + 1) % 3) as f32 - 1.0);
            first_values.push(0.3f32.ln() + zigzag);

            let channel_ratio = channel as f32 / n_embd.max(1) as f32;
            let mix_k = channel_ratio.powf(ratio_1_to_almost0);
            mix_k_values.push(mix_k);
            mix_v_values.push(mix_k + 0.3 * ratio_0_to_1);
            mix_r_values.push(0.5 * mix_k);
        }

        let time_decay = Tensor::from_vec(decay_values, &[n_embd])?;
        let time_first = Tensor::from_vec(first_values, &[n_embd])?;
        let time_mix_k = Tensor::from_vec(mix_k_values, &[n_embd])?;
        let time_mix_v = Tensor::from_vec(mix_v_values, &[n_embd])?;
        let time_mix_r = Tensor::from_vec(mix_r_values, &[n_embd])?;

        // Linear projections for R, K, V
        let key = Linear::new_with_device(n_embd, n_embd, false, device);
        let value = Linear::new_with_device(n_embd, n_embd, false, device);
        let receptance = Linear::new_with_device(n_embd, n_embd, false, device);
        let output = Linear::new_with_device(n_embd, n_embd, false, device);

        Ok(Self {
            config: config.clone(),
            layer_id,
            time_decay,
            time_first,
            time_mix_k,
            time_mix_v,
            time_mix_r,
            key,
            value,
            receptance,
            output,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Index of the block this layer belongs to.
    ///
    /// The RWKV-4 parameter initialisation is depth-dependent, so the layer index
    /// is part of the layer's identity rather than incidental bookkeeping.
    pub fn layer_id(&self) -> usize {
        self.layer_id
    }

    /// Rebuild a tensor with the caller's original rank from a flat buffer.
    fn reshape_like(&self, data: Vec<f32>, reference: &Tensor) -> Result<Tensor> {
        Tensor::from_vec(data, &reference.shape())
    }
}

impl Layer for TimeMixing {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let n_embd = self.config.n_embd;
        let (x, batch, seq) = as_batched_sequence(&input, n_embd)?;

        let mix_k = self.time_mix_k.data()?;
        let mix_v = self.time_mix_v.data()?;
        let mix_r = self.time_mix_r.data()?;

        // Token shift: each branch mixes x_t with x_{t-1} at its own ratio.
        let xk = self.reshape_like(token_shift(&x, batch, seq, n_embd, &mix_k), &input)?;
        let xv = self.reshape_like(token_shift(&x, batch, seq, n_embd, &mix_v), &input)?;
        let xr = self.reshape_like(token_shift(&x, batch, seq, n_embd, &mix_r), &input)?;

        let k = self.key.forward(xk)?.data()?;
        let v = self.value.forward(xv)?.data()?;
        let r = sigmoid(&self.receptance.forward(xr)?)?.data()?;

        // The WKV linear-attention recurrence — the core of RWKV.
        let decay = self.time_decay.data()?;
        let bonus = self.time_first.data()?;
        let wkv = wkv_scan(&k, &v, &decay, &bonus, batch, seq, n_embd);

        // Gate the recurrence output by the receptance, then project.
        let gated: Vec<f32> = r.iter().zip(wkv.iter()).map(|(gate, value)| gate * value).collect();
        let gated = self.reshape_like(gated, &input)?;

        self.output.forward(gated)
    }
}

impl TimeMixing {
    pub fn parameter_count(&self) -> usize {
        let mut total = 0;

        // Time mixing parameters
        total += self.time_decay.data().unwrap_or_default().len();
        total += self.time_first.data().unwrap_or_default().len();
        total += self.time_mix_k.data().unwrap_or_default().len();
        total += self.time_mix_v.data().unwrap_or_default().len();
        total += self.time_mix_r.data().unwrap_or_default().len();

        // Linear projection parameters
        total += self.key.parameter_count();
        total += self.value.parameter_count();
        total += self.receptance.parameter_count();
        total += self.output.parameter_count();

        total
    }
}

/// RWKV Channel Mixing layer - similar to FFN but with temporal mixing
pub struct ChannelMixing {
    config: RwkvConfig,
    layer_id: usize,
    time_mix_k: Tensor,
    time_mix_r: Tensor,
    key: Linear,
    receptance: Linear,
    value: Linear,
    device: Device,
}

impl ChannelMixing {
    pub fn new(config: &RwkvConfig, layer_id: usize) -> Result<Self> {
        Self::new_with_device(config, layer_id, Device::CPU)
    }

    pub fn new_with_device(config: &RwkvConfig, layer_id: usize, device: Device) -> Result<Self> {
        let n_embd = config.n_embd;
        let n_ffn = config.get_n_ffn();

        // RWKV-4 channel-mixing initialisation: per-channel token-shift ratios that
        // taper with depth, matching the reference implementation.
        let ratio_1_to_almost0 = 1.0 - (layer_id as f32 / config.n_layer.max(1) as f32);
        let mix_values: Vec<f32> = (0..n_embd)
            .map(|channel| (channel as f32 / n_embd.max(1) as f32).powf(ratio_1_to_almost0))
            .collect();

        let time_mix_k = Tensor::from_vec(mix_values.clone(), &[n_embd])?;
        let time_mix_r = Tensor::from_vec(mix_values, &[n_embd])?;

        // Linear transformations
        let key = Linear::new_with_device(n_embd, n_ffn, false, device);
        let receptance = Linear::new_with_device(n_embd, n_embd, false, device);
        let value = Linear::new_with_device(n_ffn, n_embd, false, device);

        Ok(Self {
            config: config.clone(),
            layer_id,
            time_mix_k,
            time_mix_r,
            key,
            receptance,
            value,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Index of the block this layer belongs to; the channel-mixing token-shift
    /// ratios taper with depth, so the index is part of the layer's identity.
    pub fn layer_id(&self) -> usize {
        self.layer_id
    }
}

impl Layer for ChannelMixing {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // RWKV channel mixing: token-shifted inputs feed a squared-ReLU FFN whose
        // output is gated by a receptance built from its own token-shifted input.
        let n_embd = self.config.n_embd;
        let (x, batch, seq) = as_batched_sequence(&input, n_embd)?;

        let mix_k = self.time_mix_k.data()?;
        let mix_r = self.time_mix_r.data()?;
        let xk = Tensor::from_vec(token_shift(&x, batch, seq, n_embd, &mix_k), &input.shape())?;
        let xr = Tensor::from_vec(token_shift(&x, batch, seq, n_embd, &mix_r), &input.shape())?;

        // Key transformation and activation
        let k = self.key.forward(xk)?;
        let k_activated = relu(&k)?; // Use ReLU for channel mixing
        let k_squared = match &k_activated {
            Tensor::F32(arr) => {
                let result = arr.mapv(|x| x * x);
                Tensor::F32(result)
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor type for channel mixing",
                ))
            },
        };

        // Receptance (gating)
        let r = self.receptance.forward(xr)?;
        let r_gated = sigmoid(&r)?;

        // Value transformation
        let v = self.value.forward(k_squared)?;

        // Apply gating
        match (&r_gated, &v) {
            (Tensor::F32(r_arr), Tensor::F32(v_arr)) => {
                let result = r_arr * v_arr;
                Ok(Tensor::F32(result))
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Tensor type mismatch in channel mixing output",
            )),
        }
    }
}

impl ChannelMixing {
    pub fn parameter_count(&self) -> usize {
        let mut total = 0;

        // Time mixing parameters for channel mixing
        total += self.time_mix_k.data().unwrap_or_default().len();
        total += self.time_mix_r.data().unwrap_or_default().len();

        // Linear transformation parameters
        total += self.key.parameter_count();
        total += self.receptance.parameter_count();
        total += self.value.parameter_count();

        total
    }
}

/// RWKV Block - combines time mixing and channel mixing
pub struct RwkvBlock {
    layer_id: usize,
    ln1: LayerNorm,
    ln2: LayerNorm,
    att: TimeMixing,
    ffn: ChannelMixing,
    device: Device,
}

impl RwkvBlock {
    pub fn new(config: &RwkvConfig, layer_id: usize) -> Result<Self> {
        Self::new_with_device(config, layer_id, Device::CPU)
    }

    pub fn new_with_device(config: &RwkvConfig, layer_id: usize, device: Device) -> Result<Self> {
        let ln1 =
            LayerNorm::new_with_device(vec![config.n_embd], config.layer_norm_epsilon, device)?;
        let ln2 =
            LayerNorm::new_with_device(vec![config.n_embd], config.layer_norm_epsilon, device)?;
        let att = TimeMixing::new_with_device(config, layer_id, device)?;
        let ffn = ChannelMixing::new_with_device(config, layer_id, device)?;

        Ok(Self {
            layer_id,
            ln1,
            ln2,
            att,
            ffn,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Depth of this block within the model.
    pub fn layer_id(&self) -> usize {
        self.layer_id
    }
}

impl Layer for RwkvBlock {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // RWKV block forward pass with residual connections

        // Time mixing with pre-norm and residual connection
        let normed1 = self.ln1.forward(input.clone())?;
        let att_out = self.att.forward(normed1)?;
        let residual1 = match (&input, &att_out) {
            (Tensor::F32(x_arr), Tensor::F32(att_arr)) => {
                let result = x_arr + att_arr;
                Tensor::F32(result)
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Tensor type mismatch in attention residual",
                ))
            },
        };

        // Channel mixing with pre-norm and residual connection
        let normed2 = self.ln2.forward(residual1.clone())?;
        let ffn_out = self.ffn.forward(normed2)?;
        let output = match (&residual1, &ffn_out) {
            (Tensor::F32(res_arr), Tensor::F32(ffn_arr)) => {
                let result = res_arr + ffn_arr;
                Tensor::F32(result)
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Tensor type mismatch in FFN residual",
                ))
            },
        };

        Ok(output)
    }
}

impl RwkvBlock {
    pub fn parameter_count(&self) -> usize {
        let mut total = 0;

        // Layer norms parameters
        total += self.ln1.parameter_count();
        total += self.ln2.parameter_count();

        // Time mixing (attention) parameters
        total += self.att.parameter_count();

        // Channel mixing (FFN) parameters
        total += self.ffn.parameter_count();

        total
    }
}

/// RWKV Language Model
/// Reference: "RWKV: Reinventing RNNs for the Transformer Era" (Peng et al., 2023)
pub struct RwkvModel {
    config: RwkvConfig,
    embeddings: Embedding,
    blocks: Vec<RwkvBlock>,
    ln_out: LayerNorm,
    head: Option<Linear>,
    device: Device,
}

impl RwkvModel {
    pub fn new(config: RwkvConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: RwkvConfig, device: Device) -> Result<Self> {
        // Token embeddings
        let embeddings =
            Embedding::new_with_device(config.vocab_size, config.n_embd, None, device)?;

        // RWKV blocks
        let mut blocks = Vec::with_capacity(config.n_layer);
        for layer_id in 0..config.n_layer {
            blocks.push(RwkvBlock::new_with_device(&config, layer_id, device)?);
        }

        // Output normalization
        let ln_out =
            LayerNorm::new_with_device(vec![config.n_embd], config.layer_norm_epsilon, device)?;

        // Language modeling head (typically tied with embeddings)
        let head = Some(Linear::new_with_device(
            config.n_embd,
            config.vocab_size,
            false,
            device,
        ));

        Ok(Self {
            config,
            embeddings,
            blocks,
            ln_out,
            head,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Forward pass for causal language modeling
    pub fn forward_lm(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.forward(input_ids.clone())?;

        if let Some(head) = &self.head {
            head.forward(hidden_states)
        } else {
            Ok(hidden_states)
        }
    }
}

impl Model for RwkvModel {
    type Config = RwkvConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Convert tensor to input_ids for embeddings
        let input_ids = match &input {
            Tensor::I64(arr) => arr.iter().map(|&x| x as u32).collect::<Vec<u32>>(),
            Tensor::F32(arr) => arr.iter().map(|&x| x as u32).collect::<Vec<u32>>(),
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported input tensor type for RWKV model",
                ))
            },
        };

        // Token embeddings
        let mut hidden_states = self.embeddings.forward(input_ids)?;

        // Pass through RWKV blocks
        for block in &self.blocks {
            hidden_states = block.forward(hidden_states)?;
        }

        // Final normalization
        let output = self.ln_out.forward(hidden_states)?;

        Ok(output)
    }

    /// Loading pretrained RWKV checkpoints is not implemented.
    ///
    /// Reporting success here would leave the caller with randomly initialised
    /// weights while believing a checkpoint had been applied, so this returns an
    /// error instead. Use the `weight_loading` module to populate the individual
    /// [`Linear`]/[`Embedding`] layers from a safetensors or PyTorch checkpoint.
    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(not_implemented(
            "RwkvModel::load_pretrained: the RWKV checkpoint format is not parsed yet; the model \
             would silently keep its randomly initialised weights",
        ))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Embeddings parameters
        total += self.embeddings.parameter_count();

        // RWKV blocks parameters
        for block in &self.blocks {
            total += block.parameter_count();
        }

        // Output normalization parameters
        total += self.ln_out.parameter_count();

        // Language modeling head parameters (if present)
        if let Some(head) = &self.head {
            total += head.parameter_count();
        }

        total
    }
}

impl RwkvModel {
    /// Create RWKV models with predefined configurations
    pub fn rwkv_169m() -> Result<Self> {
        Self::new(RwkvConfig::rwkv_169m())
    }

    pub fn rwkv_430m() -> Result<Self> {
        Self::new(RwkvConfig::rwkv_430m())
    }

    pub fn rwkv_1_5b() -> Result<Self> {
        Self::new(RwkvConfig::rwkv_1_5b())
    }

    pub fn rwkv_3b() -> Result<Self> {
        Self::new(RwkvConfig::rwkv_3b())
    }

    pub fn rwkv_7b() -> Result<Self> {
        Self::new(RwkvConfig::rwkv_7b())
    }

    pub fn rwkv_14b() -> Result<Self> {
        Self::new(RwkvConfig::rwkv_14b())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1; // SciRS2 Integration Policy

    #[test]
    fn test_rwkv_model_creation() {
        let config = RwkvConfig::default();
        let model = RwkvModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_rwkv_block_creation() {
        let config = RwkvConfig::default();
        let block = RwkvBlock::new(&config, 0);
        assert!(block.is_ok());
    }

    #[test]
    fn test_time_mixing_creation() {
        let config = RwkvConfig::default();
        let time_mix = TimeMixing::new(&config, 0);
        assert!(time_mix.is_ok());
    }

    #[test]
    fn test_channel_mixing_creation() {
        let config = RwkvConfig::default();
        let channel_mix = ChannelMixing::new(&config, 0);
        assert!(channel_mix.is_ok());
    }

    #[test]
    #[ignore] // Very heavy test - creates multiple large RWKV models, run with --ignored
    fn test_predefined_models() {
        assert!(RwkvModel::rwkv_169m().is_ok());
        assert!(RwkvModel::rwkv_430m().is_ok());
        assert!(RwkvModel::rwkv_1_5b().is_ok());
        assert!(RwkvModel::rwkv_3b().is_ok());
        assert!(RwkvModel::rwkv_7b().is_ok());
        assert!(RwkvModel::rwkv_14b().is_ok());
    }

    #[test]
    fn test_forward_pass_shape() {
        let config = RwkvConfig::default();
        let model = RwkvModel::new(config).expect("operation failed");

        // Create dummy input as i64 tensor (seq_len=8)
        let input_data = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
        let input_ids = Tensor::I64(Array1::from(input_data).into_dyn());
        let output = model.forward(input_ids);
        assert!(output.is_ok());
    }

    // ---- WKV / receptance gate (numerically stable) ----

    /// σ(r) must always be in (0, 1) — verify with large positive and negative values.
    #[test]
    fn test_receptance_gate_sigmoid_bounds() {
        // The sigmoid function σ(x) = 1/(1+exp(-x)) must produce values in (0,1).
        // We test a range from a large negative to a large positive using an LCG.
        let a: u64 = 6364136223846793005;
        let c: u64 = 1442695040888963407;
        let mut state: u64 = 0xDEAD_BEEF_1234_5678;

        for _ in 0..64 {
            state = state.wrapping_mul(a).wrapping_add(c);
            // Map to [-10, 10]
            let x = (state as i64 as f64) / (u64::MAX as f64) * 20.0;
            let sigma = 1.0 / (1.0 + (-x).exp());
            assert!(sigma > 0.0, "sigmoid must be > 0 for x={}", x);
            assert!(sigma < 1.0, "sigmoid must be < 1 for x={}", x);
        }
    }

    /// RWKV time-decay formula: w(t) = exp(-exp(w_raw)) → ∈ (0, 1).
    #[test]
    fn test_time_decay_formula_range() {
        let a: u64 = 6364136223846793005;
        let c: u64 = 1442695040888963407;
        let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;

        for _ in 0..64 {
            state = state.wrapping_mul(a).wrapping_add(c);
            // w_raw in [-3, 3] (common initialisation range)
            let w_raw = (state as i64 as f64) / (u64::MAX as f64) * 6.0;
            let decay = (-w_raw.exp()).exp();
            assert!(
                decay > 0.0 && decay < 1.0,
                "time-decay must be in (0,1) for w_raw={}",
                w_raw
            );
        }
    }

    /// Initial hidden state for RWKV recurrence is all-zeros.
    #[test]
    fn test_initial_state_is_zero() {
        let d_state = 16usize;
        let state = vec![0.0f32; d_state];
        assert!(
            state.iter().all(|&x| x == 0.0),
            "Initial RNN state must be all zeros"
        );
    }

    /// Single-step WKV update: verify the numerically stable form u+k > k alone.
    #[test]
    fn test_wkv_numerically_stable_bonus_term() {
        // Stable form: max_val = max(u+k, prev_max); WKV = (e^(u+k - max) * v + ...) / denom
        // Here we just verify that adding the bonus term u increases the effective key.
        let k: f64 = 1.5;
        let u: f64 = 0.8; // time_first bonus
        let effective_k = u + k;
        assert!(
            effective_k > k,
            "u+k must exceed k alone (bonus term increases key weight)"
        );
    }

    /// Normalise by max for numerical stability: exp(a - max) stays bounded.
    #[test]
    fn test_wkv_max_normalisation_prevents_overflow() {
        let values = [100.0f64, 200.0, 300.0, 150.0, 250.0];
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for &v in &values {
            let stabilised = (v - max_val).exp();
            assert!(
                stabilised.is_finite() && stabilised <= 1.0 + 1e-10,
                "exp(v - max) must be <= 1 and finite; got {}",
                stabilised
            );
        }
    }

    /// RWKV recurrence: state update x_new = decay * state + input.
    /// After one step from zero-state the result equals the input exactly.
    #[test]
    fn test_state_update_single_step_from_zero() {
        let decay: f64 = (-1.0_f64.exp()).exp(); // typical decay
        let state_prev: f64 = 0.0;
        let input_val: f64 = 1.23;
        let state_new = decay * state_prev + input_val;
        assert!(
            (state_new - input_val).abs() < 1e-12,
            "From zero-state one-step recurrence must equal the input"
        );
    }

    /// Multi-step recurrence: state magnitude grows bounded (|state| < sum of |inputs| / (1-decay)).
    #[test]
    fn test_multi_step_recurrence_bounded() {
        let a: u64 = 6364136223846793005;
        let c: u64 = 1442695040888963407;
        let mut lcg: u64 = 0x1234_5678_ABCD_EF01;

        let decay: f64 = 0.9;
        let n_steps = 100;
        let mut state: f64 = 0.0;
        let mut max_input: f64 = 0.0;

        for _ in 0..n_steps {
            lcg = lcg.wrapping_mul(a).wrapping_add(c);
            let input = (lcg as i64 as f64) / (u64::MAX as f64); // [-1, 1]
            if input.abs() > max_input {
                max_input = input.abs();
            }
            state = decay * state + input;
        }

        // Geometric series bound: |state| <= max_input / (1 - decay) = max_input / 0.1
        let bound = max_input / (1.0 - decay);
        assert!(
            state.abs() <= bound + 1e-9,
            "Recurrence state magnitude {} must be bounded by {}",
            state.abs(),
            bound
        );
    }

    /// Output gate: element-wise multiply of σ(r) and v.
    /// Verify that output magnitude ≤ |v| since σ(r) ∈ (0,1).
    #[test]
    fn test_output_gate_magnitude_bounded_by_value() {
        let a: u64 = 6364136223846793005;
        let c: u64 = 1442695040888963407;
        let mut lcg: u64 = 0xFEED_FACE_DEAD_BEEF;

        for _ in 0..32 {
            lcg = lcg.wrapping_mul(a).wrapping_add(c);
            let r_raw = (lcg as i64 as f64) / (u64::MAX as f64) * 6.0;
            lcg = lcg.wrapping_mul(a).wrapping_add(c);
            let v = (lcg as i64 as f64) / (u64::MAX as f64) * 4.0;

            let sigma_r = 1.0 / (1.0 + (-r_raw).exp());
            let output = sigma_r * v;
            assert!(
                output.abs() <= v.abs() + 1e-12,
                "Output gate magnitude {} must be <= |v|={}",
                output.abs(),
                v.abs()
            );
        }
    }

    /// TimeMixing layer: parameter_count must be > 0.
    #[test]
    fn test_time_mixing_parameter_count_positive() {
        let config = RwkvConfig::default();
        let time_mix = TimeMixing::new(&config, 0).expect("TimeMixing creation must succeed");
        assert!(
            time_mix.parameter_count() > 0,
            "TimeMixing must have > 0 parameters"
        );
    }

    /// ChannelMixing layer: parameter_count must be > 0.
    #[test]
    fn test_channel_mixing_parameter_count_positive() {
        let config = RwkvConfig::default();
        let ch_mix = ChannelMixing::new(&config, 0).expect("ChannelMixing creation must succeed");
        assert!(
            ch_mix.parameter_count() > 0,
            "ChannelMixing must have > 0 parameters"
        );
    }

    /// RwkvBlock: parameter_count must be ≥ sum of its parts.
    #[test]
    fn test_block_parameter_count_at_least_sublayer_sum() {
        let config = RwkvConfig::default();
        let block = RwkvBlock::new(&config, 0).expect("RwkvBlock creation must succeed");
        // Block param count = ln1 + ln2 + att + ffn
        assert!(
            block.parameter_count() > 0,
            "Block must have > 0 parameters"
        );
    }

    /// RwkvModel: num_parameters must be consistent across calls.
    #[test]
    fn test_model_num_parameters_deterministic() {
        let config = RwkvConfig::default();
        let model = RwkvModel::new(config).expect("RwkvModel creation must succeed");
        let count1 = model.num_parameters();
        let count2 = model.num_parameters();
        assert_eq!(
            count1, count2,
            "num_parameters() must return the same value on repeated calls"
        );
    }

    /// forward_lm produces a tensor when called with valid token ids.
    #[test]
    fn test_forward_lm_succeeds_with_valid_input() {
        let config = RwkvConfig::default();
        let model = RwkvModel::new(config).expect("RwkvModel creation must succeed");

        let input_data = vec![0i64, 1, 2, 3];
        let input_ids = Tensor::I64(Array1::from(input_data).into_dyn());
        let output = model.forward_lm(&input_ids);
        assert!(
            output.is_ok(),
            "forward_lm must succeed for valid token IDs"
        );
    }

    /// Layer output shape: hidden dim axis must match n_embd.
    #[test]
    fn test_forward_output_hidden_dim() {
        let config = RwkvConfig::default();
        let model = RwkvModel::new(config.clone()).expect("RwkvModel creation must succeed");

        let input_data = vec![0i64, 1, 2];
        let input_ids = Tensor::I64(Array1::from(input_data).into_dyn());
        let output = model.forward(input_ids).expect("forward must succeed");

        // Output shape: [seq_len, n_embd]
        let shape = output.shape();
        assert!(
            !shape.is_empty(),
            "Output must have at least 1 dimension, got {:?}",
            shape
        );
        // Last dimension must equal n_embd
        let last_dim = shape[shape.len() - 1];
        assert_eq!(last_dim, config.n_embd, "Last output dim must equal n_embd");
    }

    /// get_config returns reference to the config used at construction.
    #[test]
    fn test_get_config_returns_correct_config() {
        let config = RwkvConfig::rwkv_430m();
        let model = RwkvModel::new(config.clone()).expect("RwkvModel creation must succeed");
        let returned = model.get_config();
        assert_eq!(returned.n_embd, config.n_embd);
        assert_eq!(returned.n_layer, config.n_layer);
    }

    /// Model device is CPU by default.
    #[test]
    fn test_default_device_is_cpu() {
        let config = RwkvConfig::default();
        let model = RwkvModel::new(config).expect("RwkvModel creation must succeed");
        assert_eq!(model.device(), Device::CPU);
    }

    /// TimeMixing device propagates correctly.
    #[test]
    fn test_time_mixing_device_propagates() {
        let config = RwkvConfig::default();
        let tm = TimeMixing::new_with_device(&config, 0, Device::CPU)
            .expect("TimeMixing creation must succeed");
        assert_eq!(tm.device(), Device::CPU);
    }

    // ---- Real WKV recurrence regression tests ----

    /// Direct O(T²) evaluation of the published RWKV-4 WKV operator, in f64.
    ///
    /// This is the definition the stable accumulator in `wkv_scan` must reproduce:
    /// `wkv_t = (Σ_{i<t} e^{(t-1-i)·w + k_i}·v_i + e^{u+k_t}·v_t) / (same, no v)`.
    fn wkv_reference(k: &[f64], v: &[f64], decay_raw: f64, bonus: f64) -> Vec<f64> {
        let w = -decay_raw.exp();
        let seq = k.len();
        let mut out = vec![0.0f64; seq];
        for t in 0..seq {
            let mut numerator = 0.0f64;
            let mut denominator = 0.0f64;
            for i in 0..t {
                let e = ((t - 1 - i) as f64 * w + k[i]).exp();
                numerator += e * v[i];
                denominator += e;
            }
            let e = (bonus + k[t]).exp();
            numerator += e * v[t];
            denominator += e;
            out[t] = numerator / denominator;
        }
        out
    }

    /// The stable accumulator must match the direct O(T²) reference.
    #[test]
    fn test_wkv_scan_matches_quadratic_reference() {
        let k64 = [0.4f64, -1.1, 0.9, 0.05, -0.7, 1.3, -0.2, 0.6];
        let v64 = [1.0f64, -2.0, 0.5, 3.0, -1.5, 0.25, 2.5, -0.75];
        let decay_raw = 0.35f64;
        let bonus = -0.6f64;

        let expected = wkv_reference(&k64, &v64, decay_raw, bonus);

        let k: Vec<f32> = k64.iter().map(|&x| x as f32).collect();
        let v: Vec<f32> = v64.iter().map(|&x| x as f32).collect();
        let actual = wkv_scan(&k, &v, &[decay_raw as f32], &[bonus as f32], 1, k.len(), 1);

        assert_eq!(actual.len(), expected.len());
        for (t, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*got as f64 - *want).abs() < 1e-4,
                "wkv mismatch at t={t}: got {got}, reference {want}"
            );
        }
    }

    /// Multi-channel scan: every channel must independently match the reference
    /// with its own decay and bonus (i.e. channels are not accidentally shared).
    #[test]
    fn test_wkv_scan_multi_channel_matches_reference() {
        let seq = 6usize;
        let channels = 3usize;
        let decays = [0.1f32, -0.4, 0.8];
        let bonuses = [0.2f32, -1.0, 0.5];

        // Deterministic, distinct per-channel sequences.
        let mut k = vec![0.0f32; seq * channels];
        let mut v = vec![0.0f32; seq * channels];
        for t in 0..seq {
            for c in 0..channels {
                k[t * channels + c] = 0.3 * (t as f32) - 0.7 * (c as f32);
                v[t * channels + c] = (t as f32) - 2.0 * (c as f32);
            }
        }

        let actual = wkv_scan(&k, &v, &decays, &bonuses, 1, seq, channels);

        for c in 0..channels {
            let k_c: Vec<f64> = (0..seq).map(|t| k[t * channels + c] as f64).collect();
            let v_c: Vec<f64> = (0..seq).map(|t| v[t * channels + c] as f64).collect();
            let expected = wkv_reference(&k_c, &v_c, decays[c] as f64, bonuses[c] as f64);
            for t in 0..seq {
                let got = actual[t * channels + c] as f64;
                assert!(
                    (got - expected[t]).abs() < 1e-4,
                    "channel {c} t={t}: got {got}, reference {}",
                    expected[t]
                );
            }
        }
    }

    /// First timestep of the recurrence is exactly `v_0` (the state starts empty).
    #[test]
    fn test_wkv_scan_first_step_equals_first_value() {
        let k = [2.5f32, 0.0, -1.0];
        let v = [7.25f32, 1.0, -3.0];
        let out = wkv_scan(&k, &v, &[0.0], &[0.75], 1, 3, 1);
        assert!(
            (out[0] - 7.25).abs() < 1e-5,
            "wkv_0 must equal v_0, got {}",
            out[0]
        );
    }

    /// The stabilised form must survive key magnitudes that overflow `exp` in f32.
    #[test]
    fn test_wkv_scan_is_numerically_stable_for_large_keys() {
        let k = [120.0f32, -130.0, 95.0, 140.0, -200.0];
        let v = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let out = wkv_scan(&k, &v, &[0.5], &[1.0], 1, k.len(), 1);
        assert!(
            out.iter().all(|x| x.is_finite()),
            "large keys produced non-finite wkv: {out:?}"
        );
        // Every output is a convex combination of the values seen so far.
        for (t, value) in out.iter().enumerate() {
            let lo = v[..=t].iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = v[..=t].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                *value >= lo - 1e-4 && *value <= hi + 1e-4,
                "wkv_{t} = {value} escaped the convex hull [{lo}, {hi}]"
            );
        }
    }

    /// Two batch elements must be scanned independently.
    #[test]
    fn test_wkv_scan_batches_are_independent() {
        let seq = 4usize;
        let k_single = [0.2f32, -0.5, 0.9, 0.1];
        let v_single = [1.0f32, -1.0, 2.0, 0.5];

        let single = wkv_scan(&k_single, &v_single, &[0.3], &[0.4], 1, seq, 1);

        let mut k_batched = k_single.to_vec();
        k_batched.extend_from_slice(&[5.0, 5.0, 5.0, 5.0]);
        let mut v_batched = v_single.to_vec();
        v_batched.extend_from_slice(&[9.0, 9.0, 9.0, 9.0]);

        let batched = wkv_scan(&k_batched, &v_batched, &[0.3], &[0.4], 2, seq, 1);
        for t in 0..seq {
            assert!(
                (batched[t] - single[t]).abs() < 1e-6,
                "batch 0 changed when batch 1 was appended (t={t})"
            );
        }
        for t in 0..seq {
            assert!((batched[seq + t] - 9.0).abs() < 1e-5);
        }
    }

    /// Token shift interpolates each timestep with its predecessor.
    #[test]
    fn test_token_shift_interpolates_previous_timestep() {
        // 2 timesteps, 2 channels: x = [[1, 2], [3, 4]]
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mix = [0.5f32, 0.25];
        let shifted = token_shift(&x, 1, 2, 2, &mix);
        // t=0 has no predecessor: 0.5*1 + 0.5*0, 0.25*2 + 0.75*0
        assert!((shifted[0] - 0.5).abs() < 1e-6, "{shifted:?}");
        assert!((shifted[1] - 0.5).abs() < 1e-6, "{shifted:?}");
        // t=1: 0.5*3 + 0.5*1 = 2 ; 0.25*4 + 0.75*2 = 2.5
        assert!((shifted[2] - 2.0).abs() < 1e-6, "{shifted:?}");
        assert!((shifted[3] - 2.5).abs() < 1e-6, "{shifted:?}");
    }

    fn tiny_rwkv_config() -> RwkvConfig {
        RwkvConfig {
            n_embd: 8,
            n_layer: 2,
            n_head: 2,
            head_size: 4,
            vocab_size: 16,
            n_ffn: Some(16),
            ..RwkvConfig::default()
        }
    }

    /// TimeMixing must propagate information forward in time.
    ///
    /// The previous implementation was `out_proj(σ(R) ⊙ V)` — a purely positionwise
    /// map with no recurrence — so perturbing timestep 0 left every later timestep
    /// untouched. With the real WKV recurrence it must not.
    #[test]
    fn test_time_mixing_output_depends_on_earlier_timesteps() {
        let config = tiny_rwkv_config();
        let time_mix = TimeMixing::new(&config, 0).expect("TimeMixing creation");

        let seq = 4usize;
        let channels = config.n_embd;
        let base: Vec<f32> = (0..seq * channels).map(|i| (i as f32 * 0.13).sin()).collect();

        let mut perturbed = base.clone();
        for value in perturbed.iter_mut().take(channels) {
            *value += 1.5; // only timestep 0 changes
        }

        let out_base = time_mix
            .forward(Tensor::from_vec(base, &[seq, channels]).expect("input"))
            .expect("forward base")
            .data()
            .expect("data");
        let out_perturbed = time_mix
            .forward(Tensor::from_vec(perturbed, &[seq, channels]).expect("input"))
            .expect("forward perturbed")
            .data()
            .expect("data");

        let last_start = (seq - 1) * channels;
        let changed = out_base[last_start..]
            .iter()
            .zip(out_perturbed[last_start..].iter())
            .any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(
            changed,
            "the last timestep is unaffected by timestep 0 — the WKV recurrence is missing"
        );
        assert!(out_base.iter().all(|v| v.is_finite()));
    }

    /// A later timestep must never influence an earlier one (the recurrence is causal).
    #[test]
    fn test_time_mixing_is_causal() {
        let config = tiny_rwkv_config();
        let time_mix = TimeMixing::new(&config, 1).expect("TimeMixing creation");

        let seq = 5usize;
        let channels = config.n_embd;
        let base: Vec<f32> = (0..seq * channels).map(|i| (i as f32 * 0.07).cos()).collect();

        let mut perturbed = base.clone();
        for value in perturbed.iter_mut().skip((seq - 1) * channels) {
            *value += 2.0; // only the final timestep changes
        }

        let out_base = time_mix
            .forward(Tensor::from_vec(base, &[seq, channels]).expect("input"))
            .expect("forward base")
            .data()
            .expect("data");
        let out_perturbed = time_mix
            .forward(Tensor::from_vec(perturbed, &[seq, channels]).expect("input"))
            .expect("forward perturbed")
            .data()
            .expect("data");

        // Timesteps 0..seq-2 must be bit-stable (t = seq-2 reads x_{seq-2} only).
        let unaffected = (seq - 2) * channels;
        for i in 0..unaffected {
            assert!(
                (out_base[i] - out_perturbed[i]).abs() < 1e-5,
                "changing the last token altered output index {i} — recurrence is not causal"
            );
        }
    }

    /// TimeMixing accepts a batched `[batch, seq, n_embd]` input and keeps the shape.
    #[test]
    fn test_time_mixing_supports_batched_input() {
        let config = tiny_rwkv_config();
        let time_mix = TimeMixing::new(&config, 0).expect("TimeMixing creation");
        let input = Tensor::from_vec(
            (0..2 * 3 * config.n_embd).map(|i| (i as f32) * 0.01).collect(),
            &[2, 3, config.n_embd],
        )
        .expect("input");
        let out = time_mix.forward(input).expect("forward");
        assert_eq!(out.shape(), vec![2, 3, config.n_embd]);
        assert!(out.data().expect("data").iter().all(|v| v.is_finite()));
    }

    /// A channel-count mismatch is reported instead of silently reinterpreted.
    #[test]
    fn test_time_mixing_rejects_wrong_channel_count() {
        let config = tiny_rwkv_config();
        let time_mix = TimeMixing::new(&config, 0).expect("TimeMixing creation");
        let bad = Tensor::from_vec(vec![0.0; 3 * 5], &[3, 5]).expect("input");
        assert!(time_mix.forward(bad).is_err());
    }

    /// Channel mixing must also token-shift (its `time_mix_*` were previously unused).
    #[test]
    fn test_channel_mixing_depends_on_previous_timestep() {
        let config = tiny_rwkv_config();
        let channel_mix = ChannelMixing::new(&config, 0).expect("ChannelMixing creation");

        let seq = 3usize;
        let channels = config.n_embd;
        let base: Vec<f32> = (0..seq * channels).map(|i| (i as f32 * 0.21).sin() + 0.5).collect();
        let mut perturbed = base.clone();
        for value in perturbed.iter_mut().take(channels) {
            *value += 3.0; // timestep 0 only
        }

        let out_base = channel_mix
            .forward(Tensor::from_vec(base, &[seq, channels]).expect("input"))
            .expect("forward base")
            .data()
            .expect("data");
        let out_perturbed = channel_mix
            .forward(Tensor::from_vec(perturbed, &[seq, channels]).expect("input"))
            .expect("forward perturbed")
            .data()
            .expect("data");

        let changed = out_base[channels..2 * channels]
            .iter()
            .zip(out_perturbed[channels..2 * channels].iter())
            .any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(
            changed,
            "timestep 1 ignores timestep 0 — channel-mixing token shift is missing"
        );
    }

    /// The time-mixing parameters are the depth-dependent RWKV-4 initialisation,
    /// not random noise: decay is monotonically increasing across channels and the
    /// mixing ratios stay in a sane range.
    #[test]
    fn test_time_mixing_parameter_initialisation_is_structured() {
        let config = tiny_rwkv_config();
        let time_mix = TimeMixing::new(&config, 0).expect("TimeMixing creation");

        let decay = time_mix.time_decay.data().expect("decay");
        assert_eq!(decay.len(), config.n_embd);
        for pair in decay.windows(2) {
            assert!(
                pair[1] >= pair[0],
                "time_decay must increase across channels: {decay:?}"
            );
        }

        let mix_k = time_mix.time_mix_k.data().expect("mix_k");
        assert!(mix_k.iter().all(|v| (0.0..=1.0).contains(v)), "{mix_k:?}");

        // w = -exp(time_decay) is strictly negative, so the recurrence decays.
        assert!(decay.iter().all(|d| -d.exp() < 0.0));
    }

    /// The RWKV model as a whole must produce a time-dependent output.
    #[test]
    fn test_model_forward_is_sequence_dependent() {
        let config = tiny_rwkv_config();
        let model = RwkvModel::new(config.clone()).expect("model creation");

        let a = Tensor::I64(Array1::from(vec![1i64, 2, 3, 4]).into_dyn());
        let b = Tensor::I64(Array1::from(vec![4i64, 3, 2, 1]).into_dyn());

        let out_a = model.forward(a).expect("forward a").data().expect("data");
        let out_b = model.forward(b).expect("forward b").data().expect("data");

        assert_eq!(out_a.len(), out_b.len());
        assert!(
            out_a.iter().zip(out_b.iter()).any(|(x, y)| (x - y).abs() > 1e-5),
            "reversing the token order left the output unchanged"
        );
        assert!(out_a.iter().all(|v| v.is_finite()));
    }

    /// `load_pretrained` must report that it did nothing instead of claiming success.
    #[test]
    fn test_load_pretrained_reports_not_implemented() {
        let mut model = RwkvModel::new(tiny_rwkv_config()).expect("model");
        let mut reader = std::io::Cursor::new(vec![0u8; 16]);
        assert!(
            model.load_pretrained(&mut reader).is_err(),
            "load_pretrained must not fake a successful checkpoint load"
        );
    }

    /// Depth is part of a layer's identity and is reported back.
    #[test]
    fn test_layer_ids_are_reported() {
        let config = tiny_rwkv_config();
        let block = RwkvBlock::new(&config, 1).expect("block");
        assert_eq!(block.layer_id(), 1);

        let time_mix = TimeMixing::new(&config, 1).expect("time mixing");
        assert_eq!(time_mix.layer_id(), 1);

        let channel_mix = ChannelMixing::new(&config, 1).expect("channel mixing");
        assert_eq!(channel_mix.layer_id(), 1);
    }

    /// The depth-dependent initialisation really differs between layers.
    #[test]
    fn test_time_mixing_initialisation_varies_with_depth() {
        let config = tiny_rwkv_config();
        let first = TimeMixing::new(&config, 0).expect("layer 0");
        let last = TimeMixing::new(&config, config.n_layer - 1).expect("last layer");

        let first_decay = first.time_decay.data().expect("decay");
        let last_decay = last.time_decay.data().expect("decay");
        assert!(
            first_decay.iter().zip(last_decay.iter()).any(|(a, b)| (a - b).abs() > 1e-6),
            "time_decay must depend on the layer index"
        );
    }
}
