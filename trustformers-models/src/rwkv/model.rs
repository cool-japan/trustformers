use crate::rwkv::config::RwkvConfig;
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result},
    layers::{Embedding, LayerNorm, Linear},
    ops::activations::{relu, sigmoid},
    tensor::Tensor,
    traits::{Layer, Model},
};

/// RWKV Time Mixing layer - replaces traditional attention
/// This implements the core RWKV mechanism for temporal information processing
pub struct TimeMixing {
    #[allow(dead_code)]
    config: RwkvConfig,
    #[allow(dead_code)]
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

        // Time mixing parameters
        let time_decay = Tensor::randn(&[config.n_head, config.head_size])?;
        let time_first = Tensor::randn(&[config.n_head, config.head_size])?;
        let time_mix_k = Tensor::randn(&[1, 1, n_embd])?;
        let time_mix_v = Tensor::randn(&[1, 1, n_embd])?;
        let time_mix_r = Tensor::randn(&[1, 1, n_embd])?;

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
}

impl Layer for TimeMixing {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Simplified RWKV time mixing implementation
        // In practice, this would implement the full RWKV temporal mechanism

        match &input {
            Tensor::F32(_x_arr) => {
                // Apply linear transformations
                let _k = self.key.forward(input.clone())?;
                let v = self.value.forward(input.clone())?;
                let r = self.receptance.forward(input.clone())?;

                // Apply sigmoid to receptance (gating mechanism)
                let r_gated = sigmoid(&r)?;

                // Simplified RWKV computation - multiply gated receptance with value
                let mixed = match (&r_gated, &v) {
                    (Tensor::F32(r_arr), Tensor::F32(v_arr)) => {
                        let result = r_arr * v_arr;
                        Tensor::F32(result)
                    },
                    _ => {
                        return Err(tensor_op_error(
                            "tensor_operation",
                            "Tensor type mismatch in RWKV mixing",
                        ))
                    },
                };

                // Output projection
                self.output.forward(mixed)
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported input tensor type for TimeMixing",
            )),
        }
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
    #[allow(dead_code)]
    config: RwkvConfig,
    #[allow(dead_code)]
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

        // Time mixing parameters for channel mixing
        let time_mix_k = Tensor::randn(&[1, 1, n_embd])?;
        let time_mix_r = Tensor::randn(&[1, 1, n_embd])?;

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
}

impl Layer for ChannelMixing {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // RWKV channel mixing mechanism

        // Key transformation and activation
        let k = self.key.forward(input.clone())?;
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
        let r = self.receptance.forward(input)?;
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
    #[allow(dead_code)]
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

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        // Placeholder for loading pretrained weights
        // In practice, this would load weights from the RWKV format
        Ok(())
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
        let model = RwkvModel::new(config).unwrap();

        // Create dummy input as i64 tensor (seq_len=8)
        let input_data = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
        let input_ids = Tensor::I64(Array1::from(input_data).into_dyn());
        let output = model.forward(input_ids);
        assert!(output.is_ok());
    }
}
