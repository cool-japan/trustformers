use crate::mamba::config::MambaConfig;
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result},
    layers::{Embedding, Linear},
    ops::activations::silu,
    tensor::Tensor,
    traits::{Layer, Model},
};

use scirs2_core::ndarray::{s, Array2, Ix1, Ix2}; // SciRS2 Integration Policy

/// Numerically stable softplus: `ln(1 + e^x)`.
///
/// Computed as `max(x, 0) + ln(1 + e^{-|x|})` so it never overflows for large
/// magnitudes. Used to obtain the strictly-positive Mamba timestep Δ.
#[inline]
fn softplus(x: f32) -> f32 {
    x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()
}

/// RMSNorm layer (Root Mean Square Layer Normalization)
/// Used in Mamba for normalization
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
}

impl Layer for RMSNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match &input {
            Tensor::F32(arr) => {
                let mean_sq = arr.iter().map(|x| x * x).sum::<f32>() / arr.len() as f32;
                let rms = (mean_sq + self.eps).sqrt();
                let normalized = arr.mapv(|x| x / rms);

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

impl RMSNorm {
    pub fn parameter_count(&self) -> usize {
        self.weight.data().unwrap_or_default().len()
    }
}

/// 1D Causal Convolution layer for local dependencies
pub struct CausalConv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    #[allow(dead_code)]
    kernel_size: usize,
    #[allow(dead_code)]
    padding: usize,
    device: Device,
}

impl CausalConv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        use_bias: bool,
    ) -> Result<Self> {
        Self::new_with_device(
            in_channels,
            out_channels,
            kernel_size,
            use_bias,
            Device::CPU,
        )
    }

    pub fn new_with_device(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        use_bias: bool,
        device: Device,
    ) -> Result<Self> {
        // Mamba uses a *depthwise* causal convolution (each channel is convolved
        // independently), so the weight is [channels, kernel_size] rather than a
        // dense [out, in, kernel] kernel. `in_channels` is kept for API symmetry
        // and must equal `out_channels` for the depthwise operation.
        debug_assert_eq!(
            in_channels, out_channels,
            "CausalConv1d is depthwise; in_channels must equal out_channels"
        );
        let weight = Tensor::randn(&[out_channels, kernel_size])?;
        let bias = if use_bias { Some(Tensor::zeros(&[out_channels])?) } else { None };
        let padding = kernel_size - 1;

        Ok(Self {
            weight,
            bias,
            kernel_size,
            padding,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for CausalConv1d {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Causal depthwise 1D convolution over the time axis.
        //
        // Input is [seq, channels]; each output position only sees current and
        // past timesteps (left zero-padding of `kernel_size - 1`), so the layer
        // is autoregressive-safe:
        //   out[t, c] = bias[c] + Σ_{j<K} weight[c, j] · x[t - (K-1) + j, c]
        let x = match &input {
            Tensor::F32(arr) => arr.view().into_dimensionality::<Ix2>().map_err(|_| {
                tensor_op_error(
                    "tensor_operation",
                    "CausalConv1d expects a 2D [seq, channels] input",
                )
            })?,
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported input tensor type for CausalConv1d",
                ))
            },
        };
        let weight = match &self.weight {
            Tensor::F32(w) => w.view().into_dimensionality::<Ix2>().map_err(|_| {
                tensor_op_error(
                    "tensor_operation",
                    "CausalConv1d weight must be [channels, kernel_size]",
                )
            })?,
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported weight tensor type for CausalConv1d",
                ))
            },
        };

        let seq = x.shape()[0];
        let channels = x.shape()[1];
        let k = self.kernel_size;
        let bias = match &self.bias {
            Some(Tensor::F32(b)) => Some(b.view().into_dimensionality::<Ix1>().map_err(|_| {
                tensor_op_error("tensor_operation", "CausalConv1d bias must be 1D")
            })?),
            _ => None,
        };

        let mut out = Array2::<f32>::zeros((seq, channels));
        for t in 0..seq {
            for c in 0..channels {
                let mut acc = bias.as_ref().map(|b| b[c]).unwrap_or(0.0);
                for j in 0..k {
                    let src = t as isize - (k as isize - 1) + j as isize;
                    if src >= 0 {
                        acc += weight[[c, j]] * x[[src as usize, c]];
                    }
                }
                out[[t, c]] = acc;
            }
        }

        Ok(Tensor::F32(out.into_dyn()))
    }
}

impl CausalConv1d {
    pub fn parameter_count(&self) -> usize {
        let mut total = self.weight.data().unwrap_or_default().len();
        if let Some(bias) = &self.bias {
            total += bias.data().unwrap_or_default().len();
        }
        total
    }
}

/// Selective State Space Model (S6) Layer
/// Core component of Mamba architecture implementing selective SSMs
pub struct MambaBlock {
    config: MambaConfig,
    in_proj: Linear,
    conv1d: CausalConv1d,
    x_proj: Linear,
    dt_proj: Linear,
    a_log: Tensor,
    d: Tensor,
    out_proj: Linear,
    norm: RMSNorm,
    device: Device,
}

impl MambaBlock {
    pub fn new(config: &MambaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &MambaConfig, device: Device) -> Result<Self> {
        let d_inner = config.get_d_inner();
        let dt_rank = config.get_dt_rank();

        // Input projection: maps d_model to 2 * d_inner
        let in_proj = Linear::new_with_device(config.d_model, 2 * d_inner, config.use_bias, device);

        // 1D convolution for local dependencies
        let conv1d = CausalConv1d::new_with_device(
            d_inner,
            d_inner,
            config.d_conv,
            config.use_conv_bias,
            device,
        )?;

        // State space projections
        let x_proj = Linear::new_with_device(d_inner, dt_rank + config.d_state * 2, false, device);
        let dt_proj = Linear::new_with_device(dt_rank, d_inner, true, device);

        // State space matrices
        let a_log = Tensor::randn(&[d_inner, config.d_state])?;
        let d = Tensor::ones(&[d_inner])?;

        // Output projection
        let out_proj = Linear::new_with_device(d_inner, config.d_model, config.use_bias, device);

        // Normalization
        let norm = RMSNorm::new_with_device(config.d_model, config.rms_norm_eps, device)?;

        Ok(Self {
            config: config.clone(),
            in_proj,
            conv1d,
            x_proj,
            dt_proj,
            a_log,
            d,
            out_proj,
            norm,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Split the `x_proj` output into the selective parameters (Δ, B, C).
    ///
    /// `x_proj` produces `dt_rank + 2 * d_state` channels. The first `dt_rank`
    /// columns are projected through `dt_proj` and passed through softplus to
    /// obtain the strictly-positive, per-channel timestep Δ ∈ [seq, d_inner].
    /// The remaining two `d_state`-wide blocks are the input-dependent B and C
    /// matrices ∈ [seq, d_state].
    fn compute_ssm_parameters(&self, ssm_out: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let dt_rank = self.config.get_dt_rank();
        let d_state = self.config.d_state;

        let arr = match ssm_out {
            Tensor::F32(a) => a,
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor type in SSM parameter projection",
                ))
            },
        };
        let shape = arr.shape();
        if shape.len() != 2 || shape[1] != dt_rank + 2 * d_state {
            return Err(tensor_op_error(
                "tensor_operation",
                "Invalid x_proj output shape for (Δ, B, C) split",
            ));
        }

        let dt_unproj = Tensor::F32(arr.slice(s![.., ..dt_rank]).to_owned().into_dyn());
        let b = Tensor::F32(arr.slice(s![.., dt_rank..dt_rank + d_state]).to_owned().into_dyn());
        let c = Tensor::F32(arr.slice(s![.., dt_rank + d_state..]).to_owned().into_dyn());

        // Δ = softplus(dt_proj(dt_unproj)) — the data-dependent discretisation step.
        let delta = match self.dt_proj.forward(dt_unproj)? {
            Tensor::F32(d) => Tensor::F32(d.mapv(softplus)),
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor type for Δ projection",
                ))
            },
        };

        Ok((delta, b, c))
    }

    /// Real selective scan — the "S6" recurrence from Gu & Dao (2023).
    ///
    /// For each timestep `t`, the continuous-time SSM `(A, B, C)` is discretised
    /// with the per-channel, input-dependent step Δ using a zero-order hold:
    ///   Ā = exp(Δ · A),   B̄ = Δ · B
    /// and the hidden state is advanced recurrently:
    ///   h_t = Ā ⊙ h_{t-1} + B̄ · x_t
    ///   y_t = C_t · h_t + D ⊙ x_t
    /// with `A = -exp(a_log)` guaranteeing a stable (decaying) recurrence.
    ///
    /// Shapes: `x`, `Δ` ∈ [seq, d_inner]; `B`, `C` ∈ [seq, d_state];
    /// `a_log` ∈ [d_inner, d_state]; `D` ∈ [d_inner]; output ∈ [seq, d_inner].
    fn selective_scan(&self, x: &Tensor, delta: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        let to_2d = |t: &Tensor, msg: &'static str| -> Result<Array2<f32>> {
            match t {
                Tensor::F32(a) => a
                    .view()
                    .into_dimensionality::<Ix2>()
                    .map(|v| v.to_owned())
                    .map_err(|_| tensor_op_error("tensor_operation", msg)),
                _ => Err(tensor_op_error("tensor_operation", msg)),
            }
        };

        let x2 = to_2d(x, "selective_scan: x must be 2D f32")?;
        let delta2 = to_2d(delta, "selective_scan: Δ must be 2D f32")?;
        let b2 = to_2d(b, "selective_scan: B must be 2D f32")?;
        let c2 = to_2d(c, "selective_scan: C must be 2D f32")?;

        let a_log2 = match &self.a_log {
            Tensor::F32(a) => {
                a.view().into_dimensionality::<Ix2>().map(|v| v.to_owned()).map_err(|_| {
                    tensor_op_error(
                        "tensor_operation",
                        "selective_scan: a_log must be [d_inner, d_state]",
                    )
                })?
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "selective_scan: a_log must be f32",
                ))
            },
        };
        let d1 = match &self.d {
            Tensor::F32(a) => {
                a.view().into_dimensionality::<Ix1>().map(|v| v.to_owned()).map_err(|_| {
                    tensor_op_error("tensor_operation", "selective_scan: D must be 1D [d_inner]")
                })?
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "selective_scan: D must be f32",
                ))
            },
        };

        let seq = x2.shape()[0];
        let d_inner = x2.shape()[1];
        let d_state = a_log2.shape()[1];

        if delta2.shape() != [seq, d_inner]
            || b2.shape() != [seq, d_state]
            || c2.shape() != [seq, d_state]
            || a_log2.shape()[0] != d_inner
            || d1.len() != d_inner
        {
            return Err(tensor_op_error(
                "tensor_operation",
                "selective_scan: inconsistent operand shapes",
            ));
        }

        // A = -exp(a_log), precomputed once for the whole sequence.
        let a_neg = a_log2.mapv(|v| -v.exp());

        let mut h = Array2::<f32>::zeros((d_inner, d_state));
        let mut y = Array2::<f32>::zeros((seq, d_inner));

        for t in 0..seq {
            for i in 0..d_inner {
                let delta_ti = delta2[[t, i]];
                let x_ti = x2[[t, i]];
                let mut acc = 0.0f32;
                for n in 0..d_state {
                    // Zero-order-hold discretisation of (A, B) for this channel/state.
                    let a_bar = (delta_ti * a_neg[[i, n]]).exp();
                    let b_bar = delta_ti * b2[[t, n]];
                    let h_in = a_bar * h[[i, n]] + b_bar * x_ti;
                    h[[i, n]] = h_in;
                    acc += c2[[t, n]] * h_in;
                }
                y[[t, i]] = acc + d1[i] * x_ti;
            }
        }

        Ok(Tensor::F32(y.into_dyn()))
    }

    fn parameter_count(&self) -> usize {
        let mut total = 0;

        // Input projection parameters
        total += self.in_proj.parameter_count();

        // 1D convolution parameters
        total += self.conv1d.parameter_count();

        // State space projections
        total += self.x_proj.parameter_count();
        total += self.dt_proj.parameter_count();

        // State space matrices
        total += self.a_log.data().unwrap_or_default().len();
        total += self.d.data().unwrap_or_default().len();

        // Output projection parameters
        total += self.out_proj.parameter_count();

        // Normalization parameters
        total += self.norm.parameter_count();

        total
    }
}

impl Layer for MambaBlock {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Mamba block forward pass
        let residual = input.clone();

        // Pre-norm
        let normed = self.norm.forward(input)?;

        // Input projection: split into two paths
        let projected = self.in_proj.forward(normed)?;

        // Split projected into x and z paths (each of size d_inner)
        let d_inner = self.config.get_d_inner();
        let (x, z) = match &projected {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                if shape.len() != 2 || shape[1] != 2 * d_inner {
                    return Err(tensor_op_error(
                        "tensor_operation",
                        "Invalid projected tensor shape for splitting",
                    ));
                }
                let x_slice = arr.slice(s![.., ..d_inner]).to_owned().into_dyn();
                let z_slice = arr.slice(s![.., d_inner..]).to_owned().into_dyn();
                (Tensor::F32(x_slice), Tensor::F32(z_slice))
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor type for splitting",
                ))
            },
        };

        // Convolution for local dependencies
        let conv_out = self.conv1d.forward(x)?;

        // Apply SiLU activation
        let activated = silu(&conv_out)?;

        // State space projection: x_proj maps d_inner -> dt_rank + 2 * d_state.
        let ssm_out = self.x_proj.forward(activated.clone())?;

        // Recover the input-dependent (Δ, B, C) parameters that make the Mamba
        // SSM *selective*, then run the real S6 selective scan.
        let (delta, b, c) = self.compute_ssm_parameters(&ssm_out)?;
        let ssm_result = self.selective_scan(&activated, &delta, &b, &c)?;

        // Apply gating with z (element-wise multiplication after SiLU activation)
        let z_activated = silu(&z)?;
        let gated = match (&ssm_result, &z_activated) {
            (Tensor::F32(ssm_arr), Tensor::F32(z_arr)) => {
                let result = ssm_arr * z_arr;
                Tensor::F32(result)
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Tensor type mismatch in gating",
                ))
            },
        };

        // Output projection
        let output = self.out_proj.forward(gated)?;

        // Residual connection
        match (&residual, &output) {
            (Tensor::F32(res_arr), Tensor::F32(out_arr)) => {
                let result = res_arr + out_arr;
                Ok(Tensor::F32(result))
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Tensor type mismatch in residual connection",
            )),
        }
    }
}

/// Mamba Language Model
/// Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
pub struct MambaModel {
    config: MambaConfig,
    embeddings: Embedding,
    layers: Vec<MambaBlock>,
    norm_f: RMSNorm,
    lm_head: Option<Linear>,
    device: Device,
}

impl MambaModel {
    pub fn new(config: MambaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: MambaConfig, device: Device) -> Result<Self> {
        // Word embeddings
        let embeddings =
            Embedding::new_with_device(config.vocab_size, config.d_model, None, device)?;

        // Mamba layers
        let mut layers = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            layers.push(MambaBlock::new_with_device(&config, device)?);
        }

        // Final normalization
        let norm_f = RMSNorm::new_with_device(config.d_model, config.rms_norm_eps, device)?;

        // Language modeling head (optional, can be tied with embeddings)
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(Linear::new_with_device(
                config.d_model,
                config.vocab_size,
                false,
                device,
            ))
        };

        Ok(Self {
            config,
            embeddings,
            layers,
            norm_f,
            lm_head,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Forward pass for causal language modeling
    pub fn forward_lm(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.forward(input_ids.clone())?;

        if let Some(lm_head) = &self.lm_head {
            lm_head.forward(hidden_states)
        } else {
            // Use tied embeddings for output projection
            // This would require access to embedding weights
            Ok(hidden_states)
        }
    }
}

impl Model for MambaModel {
    type Config = MambaConfig;
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
                    "Unsupported input tensor type for Mamba model",
                ))
            },
        };

        // Token embeddings
        let mut hidden_states = self.embeddings.forward(input_ids)?;

        // Pass through Mamba layers
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        // Final normalization
        let output = self.norm_f.forward(hidden_states)?;

        Ok(output)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        // Placeholder for loading pretrained weights
        // In practice, this would load weights from safetensors or PyTorch format
        Ok(())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Embeddings parameters
        total += self.embeddings.parameter_count();

        // Mamba layers parameters
        for layer in &self.layers {
            total += layer.parameter_count();
        }

        // Final normalization parameters
        total += self.norm_f.parameter_count();

        // Language modeling head parameters (if present)
        if let Some(lm_head) = &self.lm_head {
            total += lm_head.parameter_count();
        }

        total
    }
}

impl MambaModel {
    /// Create a Mamba model with specified size
    pub fn mamba_130m() -> Result<Self> {
        Self::new(MambaConfig::mamba_130m())
    }

    pub fn mamba_130m_with_device(device: Device) -> Result<Self> {
        Self::new_with_device(MambaConfig::mamba_130m(), device)
    }

    pub fn mamba_370m() -> Result<Self> {
        Self::new(MambaConfig::mamba_370m())
    }

    pub fn mamba_370m_with_device(device: Device) -> Result<Self> {
        Self::new_with_device(MambaConfig::mamba_370m(), device)
    }

    pub fn mamba_790m() -> Result<Self> {
        Self::new(MambaConfig::mamba_790m())
    }

    pub fn mamba_790m_with_device(device: Device) -> Result<Self> {
        Self::new_with_device(MambaConfig::mamba_790m(), device)
    }

    pub fn mamba_1_4b() -> Result<Self> {
        Self::new(MambaConfig::mamba_1_4b())
    }

    pub fn mamba_1_4b_with_device(device: Device) -> Result<Self> {
        Self::new_with_device(MambaConfig::mamba_1_4b(), device)
    }

    pub fn mamba_2_8b() -> Result<Self> {
        Self::new(MambaConfig::mamba_2_8b())
    }

    pub fn mamba_2_8b_with_device(device: Device) -> Result<Self> {
        Self::new_with_device(MambaConfig::mamba_2_8b(), device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1; // SciRS2 Integration Policy

    #[test]
    fn test_mamba_model_creation() {
        let config = MambaConfig::default();
        let model = MambaModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_mamba_block_creation() {
        let config = MambaConfig::default();
        let block = MambaBlock::new(&config);
        assert!(block.is_ok());
    }

    #[test]
    fn test_rms_norm_creation() {
        let norm = RMSNorm::new(768, 1e-5);
        assert!(norm.is_ok());
    }

    #[test]
    fn test_causal_conv1d_creation() {
        let conv = CausalConv1d::new(768, 768, 4, true);
        assert!(conv.is_ok());
    }

    #[test]
    fn test_mamba_block_forward_runs_real_ssm() {
        // Tiny config so the O(seq · d_inner · d_state) selective scan is cheap.
        let config = MambaConfig {
            d_model: 16,
            d_state: 4,
            d_conv: 4,
            expand: 2,
            n_layer: 1,
            vocab_size: 32,
            ..MambaConfig::default()
        };
        let block = MambaBlock::new(&config).expect("block construction");
        let seq = 5;
        let input = Tensor::randn(&[seq, config.d_model]).expect("input");
        let out = block.forward(input).expect("forward");
        // The real selective scan preserves shape and yields finite values
        // (the old placeholder returned silu(x); this exercises the S6 path).
        assert_eq!(out.shape(), vec![seq, config.d_model]);
        let data = out.data().expect("data");
        assert!(
            data.iter().all(|v| v.is_finite()),
            "selective scan produced non-finite values"
        );
    }

    #[test]
    fn test_causal_conv1d_is_causal_and_not_identity() {
        // A depthwise causal conv must preserve shape AND actually mix timesteps,
        // i.e. it must NOT return its input unchanged (the old fake behaviour).
        let channels = 3;
        let conv = CausalConv1d::new(channels, channels, 3, false).expect("conv");
        let seq = 6;
        let input = Tensor::randn(&[seq, channels]).expect("input");
        let out = conv.forward(input.clone()).expect("forward");
        assert_eq!(out.shape(), vec![seq, channels]);
        let before = input.data().expect("in data");
        let after = out.data().expect("out data");
        // Random weights make an identity pass-through astronomically unlikely.
        assert!(
            before.iter().zip(after.iter()).any(|(a, b)| (a - b).abs() > 1e-6),
            "causal conv returned its input unchanged (identity) — fake implementation"
        );
        assert!(after.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[ignore] // Very heavy test - creates multiple large models, run with --ignored
    fn test_predefined_models() {
        assert!(MambaModel::mamba_130m().is_ok());
        assert!(MambaModel::mamba_370m().is_ok());
        assert!(MambaModel::mamba_790m().is_ok());
        assert!(MambaModel::mamba_1_4b().is_ok());
        assert!(MambaModel::mamba_2_8b().is_ok());
    }

    #[test]
    fn test_forward_pass_shape() {
        let config = MambaConfig::default();
        let model = MambaModel::new(config).expect("operation failed");

        // Create dummy input as i64 tensor (batch_size=1, seq_len=10)
        let input_data = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let input_ids = Tensor::I64(Array1::from(input_data).into_dyn());
        let output = model.forward(input_ids);
        assert!(output.is_ok());
    }

    #[test]
    fn test_device_support() {
        // Test CPU device
        let config = MambaConfig::default();
        let model_cpu =
            MambaModel::new_with_device(config.clone(), Device::CPU).expect("operation failed");
        assert_eq!(model_cpu.device(), Device::CPU);

        // Test predefined models with device
        let model_130m = MambaModel::mamba_130m_with_device(Device::CPU).expect("operation failed");
        assert_eq!(model_130m.device(), Device::CPU);

        // Test all components have device support
        let block = MambaBlock::new_with_device(&config, Device::CPU).expect("operation failed");
        assert_eq!(block.device(), Device::CPU);

        let norm = RMSNorm::new_with_device(768, 1e-5, Device::CPU).expect("operation failed");
        assert_eq!(norm.device(), Device::CPU);

        let conv = CausalConv1d::new_with_device(768, 768, 4, true, Device::CPU)
            .expect("operation failed");
        assert_eq!(conv.device(), Device::CPU);
    }

    #[test]
    fn test_metal_device_creation() {
        // Test Metal device creation (will use Metal or fall back to CPU)
        let device = Device::Metal(0);
        let config = MambaConfig::default();
        let model = MambaModel::new_with_device(config, device).expect("operation failed");
        // Device should be set (either Metal or CPU depending on availability)
        assert!(model.device() == Device::Metal(0) || model.device() == Device::CPU);
    }

    #[test]
    #[ignore] // Very heavy test - creates multiple large models with device, run with --ignored
    fn test_all_predefined_models_with_device() {
        let device = Device::CPU;
        assert!(MambaModel::mamba_130m_with_device(device).is_ok());
        assert!(MambaModel::mamba_370m_with_device(device).is_ok());
        assert!(MambaModel::mamba_790m_with_device(device).is_ok());
        assert!(MambaModel::mamba_1_4b_with_device(device).is_ok());
        assert!(MambaModel::mamba_2_8b_with_device(device).is_ok());
    }
}
