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

use scirs2_core::ndarray::s; // SciRS2 Integration Policy

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
        let weight = Tensor::randn(&[out_channels, in_channels, kernel_size])?;
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
        // Simplified 1D convolution implementation
        // In practice, this would use optimized convolution operations
        match &input {
            Tensor::F32(_input_arr) => {
                // For now, return input as-is - full implementation would require
                // proper convolution operations with causal padding
                Ok(input.clone())
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported input tensor type for CausalConv1d",
            )),
        }
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

    fn selective_ssm(
        &self,
        x: &Tensor,
        delta: &Tensor,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
    ) -> Result<Tensor> {
        // Simplified selective state space model computation
        // Implements a basic version of the S6 (selective scan) algorithm
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let d_model = x.shape()[2];

        // Discretize the continuous parameters using zero-order hold (ZOH)
        // delta_a = delta * a (element-wise)
        let delta_a = delta.mul(a)?;

        // exp(delta_a) for discretization
        let a_discrete = delta_a.exp()?;

        // delta_b = delta * b
        let delta_b = delta.mul(b)?;

        // Initialize state and output
        let mut y = Tensor::zeros(&[batch_size, seq_len, d_model])?;
        let mut h = Tensor::zeros(&[batch_size, d_model])?; // Hidden state

        // Sequential processing for each time step
        for t in 0..seq_len {
            // Get current input
            let x_t = x.slice(1, t, t + 1)?.squeeze(1)?;
            let a_t = a_discrete.slice(1, t, t + 1)?.squeeze(1)?;
            let b_t = delta_b.slice(1, t, t + 1)?.squeeze(1)?;
            let c_t = c.slice(1, t, t + 1)?.squeeze(1)?;

            // State update: h = A * h + B * x
            h = a_t.mul(&h)?.add(&b_t.mul(&x_t)?)?;

            // Output: y = C * h
            let y_t = c_t.mul(&h)?;

            // Store output (simplified assignment)
            // In a real implementation, this would properly assign to the slice
            let y_expanded = y_t.unsqueeze(1)?;
            if t == 0 {
                y = y_expanded;
            } else {
                y = Tensor::concat(&[y, y_expanded], 1)?;
            }
        }

        Ok(y)
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

        // State space projection
        let ssm_out = self.x_proj.forward(activated.clone())?;

        // Apply selective SSM (simplified implementation)
        let ssm_result =
            self.selective_ssm(&activated, &ssm_out, &self.a_log, &ssm_out, &ssm_out)?;

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
        let model = MambaModel::new(config).unwrap();

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
        let model_cpu = MambaModel::new_with_device(config.clone(), Device::CPU).unwrap();
        assert_eq!(model_cpu.device(), Device::CPU);

        // Test predefined models with device
        let model_130m = MambaModel::mamba_130m_with_device(Device::CPU).unwrap();
        assert_eq!(model_130m.device(), Device::CPU);

        // Test all components have device support
        let block = MambaBlock::new_with_device(&config, Device::CPU).unwrap();
        assert_eq!(block.device(), Device::CPU);

        let norm = RMSNorm::new_with_device(768, 1e-5, Device::CPU).unwrap();
        assert_eq!(norm.device(), Device::CPU);

        let conv = CausalConv1d::new_with_device(768, 768, 4, true, Device::CPU).unwrap();
        assert_eq!(conv.device(), Device::CPU);
    }

    #[test]
    fn test_metal_device_creation() {
        // Test Metal device creation (will use Metal or fall back to CPU)
        let device = Device::Metal(0);
        let config = MambaConfig::default();
        let model = MambaModel::new_with_device(config, device).unwrap();
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
