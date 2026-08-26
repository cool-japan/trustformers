use crate::mamba::config::MambaConfig;
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{not_implemented, tensor_op_error, Result},
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
///
/// Normalises **each token independently** over the last axis:
///
/// ```text
/// y_i = x_i / sqrt(mean_j(x_j²) + eps) · g_i
/// ```
///
/// where the mean runs over the `normalized_shape` channels of that one token.
/// A previous revision took the mean over every element of the tensor, which made
/// each token's output depend on every other token in the batch and sequence —
/// the value of a position then changed when unrelated positions were appended,
/// breaking both the architecture and incremental decoding.
///
/// Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019).
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
        let Tensor::F32(arr) = &input else {
            return Err(tensor_op_error(
                "tensor_operation",
                "Unsupported input tensor type for RMSNorm",
            ));
        };

        let shape = arr.shape().to_vec();
        let Some(&channels) = shape.last() else {
            return Err(tensor_op_error(
                "tensor_operation",
                "RMSNorm expects a tensor with at least one axis",
            ));
        };

        let gain = self.weight.data()?;
        if gain.len() != channels {
            return Err(tensor_op_error(
                "tensor_operation",
                format!(
                    "RMSNorm weight width {} does not match the input's last axis {channels}",
                    gain.len()
                ),
            ));
        }
        if channels == 0 {
            return Err(tensor_op_error(
                "tensor_operation",
                "RMSNorm requires a non-empty last axis",
            ));
        }

        // Each token is normalised by its *own* root mean square.
        let mut data: Vec<f32> = arr.iter().copied().collect();
        for token in data.chunks_mut(channels) {
            let mean_square = token.iter().map(|v| v * v).sum::<f32>() / channels as f32;
            let inverse_rms = 1.0 / (mean_square + self.eps).sqrt();
            for (value, weight) in token.iter_mut().zip(gain.iter()) {
                *value = *value * inverse_rms * weight;
            }
        }

        Tensor::from_vec(data, &shape)
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
    kernel_size: usize,
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
    /// Width of the convolution kernel.
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Amount of implicit left padding (`kernel_size - 1`) that makes the
    /// convolution causal: output `t` only ever reads inputs `<= t`.
    pub fn padding(&self) -> usize {
        self.padding
    }

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
    /// The embedding matrix, kept when `tie_word_embeddings` is set so the LM head
    /// can genuinely share it (`logits = h · Eᵀ`) instead of silently returning
    /// hidden states in its place.
    tied_embedding_weight: Option<Tensor>,
    device: Device,
}

impl MambaModel {
    pub fn new(config: MambaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: MambaConfig, device: Device) -> Result<Self> {
        // Word embeddings
        let mut embeddings =
            Embedding::new_with_device(config.vocab_size, config.d_model, None, device)?;

        // Mamba layers
        let mut layers = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            layers.push(MambaBlock::new_with_device(&config, device)?);
        }

        // Final normalization
        let norm_f = RMSNorm::new_with_device(config.d_model, config.rms_norm_eps, device)?;

        // Language modeling head: either its own projection, or genuinely tied to
        // the embedding matrix. Tying means holding the *same* matrix, so the
        // weight is installed on the embedding table and retained here.
        let (lm_head, tied_embedding_weight) = if config.tie_word_embeddings {
            let weight = Tensor::randn(&[config.vocab_size, config.d_model])?;
            embeddings.set_weight(weight.clone())?;
            (None, Some(weight))
        } else {
            (
                Some(Linear::new_with_device(
                    config.d_model,
                    config.vocab_size,
                    false,
                    device,
                )),
                None,
            )
        };

        Ok(Self {
            config,
            embeddings,
            layers,
            norm_f,
            lm_head,
            tied_embedding_weight,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Forward pass for causal language modeling.
    ///
    /// Returns `[seq_len, vocab_size]` logits. With `tie_word_embeddings` the
    /// output projection is the transposed embedding matrix (`logits = h · Eᵀ`),
    /// exactly as in the reference implementation.
    pub fn forward_lm(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.forward(input_ids.clone())?;

        match (&self.lm_head, &self.tied_embedding_weight) {
            (Some(lm_head), _) => lm_head.forward(hidden_states),
            (None, Some(embedding_weight)) => {
                // E is [vocab, d_model]; h is [seq, d_model]; h · Eᵀ -> [seq, vocab].
                hidden_states.matmul(&embedding_weight.t()?)
            },
            (None, None) => Err(tensor_op_error(
                "mamba_forward_lm",
                "no language-modelling head and no tied embedding weight is available",
            )),
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

    /// Loading pretrained Mamba checkpoints is not implemented.
    ///
    /// Reporting success here would leave the caller with randomly initialised
    /// weights while believing a checkpoint had been applied, so this returns an
    /// error instead. Use the `weight_loading` module to populate the individual
    /// layers from a safetensors or PyTorch checkpoint.
    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(not_implemented(
            "MambaModel::load_pretrained: the Mamba checkpoint format is not parsed yet; the \
             model would silently keep its randomly initialised weights",
        ))
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

    /// RMSNorm must divide each token by its own root mean square.
    ///
    /// Hand-computed: for the token `[3, 4]`, `mean(x²) = (9 + 16) / 2 = 12.5`,
    /// so with `eps = 0` the output is `[3, 4] / sqrt(12.5) ≈ [0.8485, 1.1314]`.
    /// The token `[1, 1]` has `rms = 1`, so it must come back unchanged. A
    /// whole-tensor mean (the previous behaviour) gives `sqrt(13.5/2)` for *both*
    /// rows and reproduces neither.
    #[test]
    fn test_rms_norm_normalises_each_token_independently() {
        let norm = RMSNorm::new(2, 0.0).expect("RMSNorm creation");
        let input = Tensor::from_vec(vec![3.0, 4.0, 1.0, 1.0], &[2, 2]).expect("input");

        let out = norm.forward(input).expect("forward").data().expect("data");
        let scale = (12.5f32).sqrt();

        assert!((out[0] - 3.0 / scale).abs() < 1e-6, "got {}", out[0]);
        assert!((out[1] - 4.0 / scale).abs() < 1e-6, "got {}", out[1]);
        assert!((out[2] - 1.0).abs() < 1e-6, "got {}", out[2]);
        assert!((out[3] - 1.0).abs() < 1e-6, "got {}", out[3]);
    }

    /// A token's normalised value must not change when other tokens are appended.
    ///
    /// This is what the whole-tensor mean broke: the same token produced a
    /// different output depending on the rest of the sequence.
    #[test]
    fn test_rms_norm_is_independent_of_sequence_length() {
        let norm = RMSNorm::new(3, 1e-5).expect("RMSNorm creation");

        let short = Tensor::from_vec(vec![0.5, -1.5, 2.0], &[1, 3]).expect("short");
        let long = Tensor::from_vec(
            vec![0.5, -1.5, 2.0, 40.0, -40.0, 40.0, 0.25, 0.25, 0.25],
            &[3, 3],
        )
        .expect("long");

        let short_out = norm.forward(short).expect("short forward").data().expect("data");
        let long_out = norm.forward(long).expect("long forward").data().expect("data");

        for i in 0..3 {
            assert!(
                (short_out[i] - long_out[i]).abs() < 1e-6,
                "token 0 changed when later tokens were appended: {} vs {}",
                short_out[i],
                long_out[i]
            );
        }
    }

    /// A weight whose width disagrees with the input is reported, not broadcast.
    #[test]
    fn test_rms_norm_rejects_mismatched_width() {
        let norm = RMSNorm::new(4, 1e-5).expect("RMSNorm creation");
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).expect("input");
        assert!(norm.forward(input).is_err());
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

    /// The selective scan must reproduce the S6 recurrence exactly.
    ///
    /// Reference (Gu & Dao, 2023), evaluated in f64 with a naive triple loop:
    ///   Ā = exp(Δ · A),  B̄ = Δ · B,  h_t = Ā ⊙ h_{t-1} + B̄ · x_t,
    ///   y_t = C_t · h_t + D ⊙ x_t,  with A = -exp(a_log).
    #[test]
    fn test_selective_scan_matches_naive_reference() {
        let d_model = 4usize;
        let d_state = 3usize;
        let config = MambaConfig {
            d_model,
            d_state,
            d_conv: 2,
            expand: 2,
            n_layer: 1,
            vocab_size: 8,
            ..MambaConfig::default()
        };
        let mut block = MambaBlock::new(&config).expect("block");
        let d_inner = config.get_d_inner();

        // Pin A and D so the comparison is fully deterministic.
        let a_log_values: Vec<f32> =
            (0..d_inner * d_state).map(|i| -0.5 + 0.1 * (i % 7) as f32).collect();
        let d_values: Vec<f32> = (0..d_inner).map(|i| 0.25 * (i as f32) - 0.5).collect();
        block.a_log = Tensor::from_vec(a_log_values.clone(), &[d_inner, d_state]).expect("a_log");
        block.d = Tensor::from_vec(d_values.clone(), &[d_inner]).expect("d");

        let seq = 5usize;
        let x_values: Vec<f32> = (0..seq * d_inner).map(|i| (i as f32 * 0.37).sin()).collect();
        let delta_values: Vec<f32> =
            (0..seq * d_inner).map(|i| 0.1 + 0.05 * ((i % 5) as f32)).collect();
        let b_values: Vec<f32> = (0..seq * d_state).map(|i| (i as f32 * 0.21).cos()).collect();
        let c_values: Vec<f32> = (0..seq * d_state).map(|i| 0.5 - 0.1 * ((i % 4) as f32)).collect();

        let x = Tensor::from_vec(x_values.clone(), &[seq, d_inner]).expect("x");
        let delta = Tensor::from_vec(delta_values.clone(), &[seq, d_inner]).expect("delta");
        let b = Tensor::from_vec(b_values.clone(), &[seq, d_state]).expect("b");
        let c = Tensor::from_vec(c_values.clone(), &[seq, d_state]).expect("c");

        let actual = block
            .selective_scan(&x, &delta, &b, &c)
            .expect("selective scan")
            .data()
            .expect("data");

        // Naive f64 reference.
        let mut h = vec![0.0f64; d_inner * d_state];
        let mut expected = vec![0.0f64; seq * d_inner];
        for t in 0..seq {
            for i in 0..d_inner {
                let delta_ti = delta_values[t * d_inner + i] as f64;
                let x_ti = x_values[t * d_inner + i] as f64;
                let mut acc = 0.0f64;
                for n in 0..d_state {
                    let a = -(a_log_values[i * d_state + n] as f64).exp();
                    let a_bar = (delta_ti * a).exp();
                    let b_bar = delta_ti * b_values[t * d_state + n] as f64;
                    let h_new = a_bar * h[i * d_state + n] + b_bar * x_ti;
                    h[i * d_state + n] = h_new;
                    acc += c_values[t * d_state + n] as f64 * h_new;
                }
                expected[t * d_inner + i] = acc + d_values[i] as f64 * x_ti;
            }
        }

        assert_eq!(actual.len(), expected.len());
        for (idx, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*got as f64 - *want).abs() < 1e-4,
                "selective scan element {idx}: got {got}, reference {want}"
            );
        }
    }

    /// The scan is a real recurrence: an earlier timestep must influence a later
    /// output (a purely position-wise map would not).
    #[test]
    fn test_selective_scan_carries_state_forward() {
        let config = MambaConfig {
            d_model: 4,
            d_state: 2,
            d_conv: 2,
            expand: 2,
            n_layer: 1,
            vocab_size: 8,
            ..MambaConfig::default()
        };
        let block = MambaBlock::new(&config).expect("block");
        let d_inner = config.get_d_inner();
        let seq = 4usize;

        let base_x: Vec<f32> = (0..seq * d_inner).map(|i| (i as f32 * 0.29).sin()).collect();
        let mut perturbed_x = base_x.clone();
        for value in perturbed_x.iter_mut().take(d_inner) {
            *value += 2.0; // timestep 0 only
        }

        let delta = Tensor::from_vec(vec![0.5f32; seq * d_inner], &[seq, d_inner]).expect("delta");
        let b = Tensor::from_vec(vec![1.0f32; seq * config.d_state], &[seq, config.d_state])
            .expect("b");
        let c = Tensor::from_vec(vec![1.0f32; seq * config.d_state], &[seq, config.d_state])
            .expect("c");

        let y_base = block
            .selective_scan(
                &Tensor::from_vec(base_x, &[seq, d_inner]).expect("x"),
                &delta,
                &b,
                &c,
            )
            .expect("scan")
            .data()
            .expect("data");
        let y_perturbed = block
            .selective_scan(
                &Tensor::from_vec(perturbed_x, &[seq, d_inner]).expect("x"),
                &delta,
                &b,
                &c,
            )
            .expect("scan")
            .data()
            .expect("data");

        let last = (seq - 1) * d_inner;
        assert!(
            y_base[last..]
                .iter()
                .zip(y_perturbed[last..].iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "the last timestep ignores timestep 0 — the scan carries no state"
        );
    }

    /// Shape mismatches in the scan operands are reported, not silently ignored.
    #[test]
    fn test_selective_scan_rejects_inconsistent_shapes() {
        let config = MambaConfig {
            d_model: 4,
            d_state: 2,
            d_conv: 2,
            expand: 2,
            n_layer: 1,
            vocab_size: 8,
            ..MambaConfig::default()
        };
        let block = MambaBlock::new(&config).expect("block");
        let d_inner = config.get_d_inner();

        let x = Tensor::from_vec(vec![0.0f32; 3 * d_inner], &[3, d_inner]).expect("x");
        let delta = Tensor::from_vec(vec![0.0f32; 2 * d_inner], &[2, d_inner]).expect("delta");
        let b =
            Tensor::from_vec(vec![0.0f32; 3 * config.d_state], &[3, config.d_state]).expect("b");
        let c =
            Tensor::from_vec(vec![0.0f32; 3 * config.d_state], &[3, config.d_state]).expect("c");

        assert!(block.selective_scan(&x, &delta, &b, &c).is_err());
    }

    /// A tied LM head must return real vocabulary logits, not the hidden states.
    #[test]
    fn test_forward_lm_with_tied_embeddings_returns_vocab_logits() {
        let config = MambaConfig {
            d_model: 8,
            d_state: 4,
            d_conv: 2,
            expand: 2,
            n_layer: 1,
            vocab_size: 12,
            tie_word_embeddings: true,
            ..MambaConfig::default()
        };
        let model = MambaModel::new(config.clone()).expect("model");

        let input = Tensor::I64(Array1::from(vec![1i64, 2, 3]).into_dyn());
        let logits = model.forward_lm(&input).expect("forward_lm");

        assert_eq!(
            logits.shape(),
            vec![3, config.vocab_size],
            "a tied head must still project to the vocabulary"
        );
        let data = logits.data().expect("data");
        assert!(data.iter().all(|v| v.is_finite()));
        assert!(data.iter().any(|v| v.abs() > 1e-6));
    }

    /// An untied LM head also produces vocabulary logits.
    #[test]
    fn test_forward_lm_with_untied_head_returns_vocab_logits() {
        let config = MambaConfig {
            d_model: 8,
            d_state: 4,
            d_conv: 2,
            expand: 2,
            n_layer: 1,
            vocab_size: 12,
            tie_word_embeddings: false,
            ..MambaConfig::default()
        };
        let model = MambaModel::new(config.clone()).expect("model");

        let input = Tensor::I64(Array1::from(vec![0i64, 5]).into_dyn());
        let logits = model.forward_lm(&input).expect("forward_lm");
        assert_eq!(logits.shape(), vec![2, config.vocab_size]);
    }

    /// `load_pretrained` must report that it did nothing instead of claiming success.
    #[test]
    fn test_load_pretrained_reports_not_implemented() {
        let config = MambaConfig {
            d_model: 8,
            d_state: 4,
            d_conv: 2,
            expand: 2,
            n_layer: 1,
            vocab_size: 12,
            ..MambaConfig::default()
        };
        let mut model = MambaModel::new(config).expect("model");
        let mut reader = std::io::Cursor::new(vec![0u8; 16]);
        assert!(
            model.load_pretrained(&mut reader).is_err(),
            "load_pretrained must not fake a successful checkpoint load"
        );
    }

    /// The causal convolution reports the padding that makes it causal.
    #[test]
    fn test_causal_conv1d_reports_its_padding() {
        let conv = CausalConv1d::new(4, 4, 3, false).expect("conv");
        assert_eq!(conv.kernel_size(), 3);
        assert_eq!(conv.padding(), 2);
    }
}
