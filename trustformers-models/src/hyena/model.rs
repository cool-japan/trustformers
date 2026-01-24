use crate::hyena::config::HyenaConfig;
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result},
    layers::{Embedding, LayerNorm, Linear},
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

/// Hyena implicit filter for long convolutions
pub struct HyenaFilter {
    filter_order: usize,
    #[allow(dead_code)]
    hidden_size: usize,
    #[allow(dead_code)]
    seq_len: usize,

    // Learned parameters for implicit filter
    filter_fn: Linear,          // Generates filter coefficients
    modulation: Option<Linear>, // Optional modulation

    // Filter state
    use_fft: bool,
    w: f32,
    wd: f32,
    device: Device,
}

impl HyenaFilter {
    pub fn new(config: &HyenaConfig, seq_len: usize) -> Result<Self> {
        Self::new_with_device(config, seq_len, Device::CPU)
    }

    pub fn new_with_device(config: &HyenaConfig, seq_len: usize, device: Device) -> Result<Self> {
        let filter_fn =
            Linear::new_with_device(config.filter_order, config.hidden_size, config.bias, device);

        let modulation = if config.modulate {
            Some(Linear::new_with_device(
                config.hidden_size,
                config.hidden_size,
                config.bias,
                device,
            ))
        } else {
            None
        };

        Ok(Self {
            filter_order: config.filter_order,
            hidden_size: config.hidden_size,
            seq_len,
            filter_fn,
            modulation,
            use_fft: config.use_flashfft,
            w: config.w,
            wd: config.wd,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Generate implicit filter coefficients
    fn generate_filter(&self, length: usize) -> Result<Tensor> {
        // Create position indices
        let positions: Vec<f32> = (0..length).map(|i| i as f32).collect();
        let _position_tensor = Tensor::from_vec(positions, &[length])?;

        // Generate base frequencies using exponential spacing
        let mut frequencies = Vec::new();
        for i in 0..self.filter_order {
            let freq = self.w * (-self.wd * i as f32).exp();
            frequencies.push(freq);
        }
        let _freq_tensor = Tensor::from_vec(frequencies, &[self.filter_order])?;

        // Generate filter using learned function
        // In practice, this would compute: filter_fn(position_encoding(positions))
        // For simplicity, we'll create a basic exponentially decaying filter
        let mut filter_coeffs = Vec::new();
        for i in 0..length {
            let decay = (-0.01 * i as f32).exp();
            filter_coeffs.push(decay);
        }

        Tensor::from_vec(filter_coeffs, &[length])
    }

    /// Apply convolution using FFT for efficiency
    fn fft_conv(&self, x: &Tensor, filter: &Tensor) -> Result<Tensor> {
        // This would use FFT libraries like FFTW or cuFFT
        // For now, implement a simplified convolution
        self.simple_conv(x, filter)
    }

    /// Simple convolution fallback
    fn simple_conv(&self, x: &Tensor, filter: &Tensor) -> Result<Tensor> {
        let filter_len = filter.shape()[0];

        if x.shape().len() == 2 {
            // Handle 2D tensor [seq_len, hidden_size]
            let seq_len = x.shape()[0];
            let hidden_size = x.shape()[1];

            // Initialize output
            let mut output = Tensor::zeros(&[seq_len, hidden_size])?;

            // Apply convolution (simplified implementation)
            for i in 0..seq_len {
                for j in 0..hidden_size {
                    let mut sum = 0.0;
                    for k in 0..std::cmp::min(filter_len, i + 1) {
                        let x_val = x.get_scalar(&[i - k, j])?;
                        let f_val = filter.get_scalar(&[k])?;
                        sum += x_val * f_val;
                    }
                    output = output.set_scalar(&[i, j], sum)?;
                }
            }

            Ok(output)
        } else if x.shape().len() == 3 {
            // Handle 3D tensor [batch_size, seq_len, hidden_size]
            let batch_size = x.shape()[0];
            let seq_len = x.shape()[1];
            let hidden_size = x.shape()[2];

            // Initialize output
            let mut output = Tensor::zeros(&[batch_size, seq_len, hidden_size])?;

            // Apply convolution (simplified implementation)
            for b in 0..batch_size {
                for i in 0..seq_len {
                    for j in 0..hidden_size {
                        let mut sum = 0.0;
                        for k in 0..std::cmp::min(filter_len, i + 1) {
                            let x_val = x.get_scalar(&[b, i - k, j])?;
                            let f_val = filter.get_scalar(&[k])?;
                            sum += x_val * f_val;
                        }
                        output = output.set_scalar(&[b, i, j], sum)?;
                    }
                }
            }

            Ok(output)
        } else {
            Err(tensor_op_error(
                "tensor_operation",
                format!("Unsupported tensor shape for convolution: {:?}", x.shape()),
            ))
        }
    }

    pub fn parameter_count(&self) -> usize {
        let filter_params = self.filter_fn.parameter_count();
        let modulation_params = self.modulation.as_ref().map(|m| m.parameter_count()).unwrap_or(0);
        filter_params + modulation_params
    }
}

impl Layer for HyenaFilter {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Handle both 2D [seq_len, hidden_size] and 3D [batch_size, seq_len, hidden_size]
        let seq_len = if input.shape().len() == 2 {
            input.shape()[0] // 2D: [seq_len, hidden_size]
        } else {
            input.shape()[1] // 3D: [batch_size, seq_len, hidden_size]
        };

        // Generate filter for current sequence length
        let filter = self.generate_filter(seq_len)?;

        // Apply modulation if enabled
        let modulated_input = if let Some(ref mod_layer) = self.modulation {
            let modulation = mod_layer.forward(input.clone())?;
            input.mul(&modulation)?
        } else {
            input
        };

        // Apply convolution
        if self.use_fft && seq_len > 1024 {
            self.fft_conv(&modulated_input, &filter)
        } else {
            self.simple_conv(&modulated_input, &filter)
        }
    }
}

/// Hyena operator combining multiple projections and implicit convolutions
pub struct HyenaOperator {
    order: usize,
    hidden_size: usize,

    // Projections for each order
    projections: Vec<Linear>,

    // Implicit filters
    filters: Vec<HyenaFilter>,

    // Output projection
    output_proj: Linear,

    // Local convolution for short-range dependencies
    local_conv: Option<LocalConvolution>,

    device: Device,
}

impl HyenaOperator {
    pub fn new(config: &HyenaConfig, seq_len: usize) -> Result<Self> {
        Self::new_with_device(config, seq_len, Device::CPU)
    }

    pub fn new_with_device(config: &HyenaConfig, seq_len: usize, device: Device) -> Result<Self> {
        let mut projections = Vec::new();
        let mut filters = Vec::new();

        // Create projections and filters for each order
        for _ in 0..config.order {
            projections.push(Linear::new_with_device(
                config.hidden_size,
                config.hidden_size,
                config.bias,
                device,
            ));
            filters.push(HyenaFilter::new_with_device(config, seq_len, device)?);
        }

        let output_proj = Linear::new_with_device(
            config.hidden_size * config.order,
            config.hidden_size,
            config.bias,
            device,
        );

        let local_conv = if config.local_order > 0 {
            Some(LocalConvolution::new_with_device(config, device)?)
        } else {
            None
        };

        Ok(Self {
            order: config.order,
            hidden_size: config.hidden_size,
            projections,
            filters,
            output_proj,
            local_conv,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for HyenaOperator {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let mut outputs = Vec::new();

        // Apply local convolution first for short-range dependencies
        let processed_input = if let Some(ref local_conv) = self.local_conv {
            local_conv.forward(input)?
        } else {
            input
        };

        // Apply each order of the Hyena operator
        for i in 0..self.order {
            // Project input
            let projected = self.projections[i].forward(processed_input.clone())?;

            // Apply implicit filter
            let filtered = self.filters[i].forward(projected)?;

            outputs.push(filtered);
        }

        // Concatenate outputs from all orders
        let concatenated = self.concatenate_tensors(outputs)?;

        // Final output projection
        self.output_proj.forward(concatenated)
    }
}

impl HyenaOperator {
    pub fn parameter_count(&self) -> usize {
        let projections_params: usize = self.projections.iter().map(|p| p.parameter_count()).sum();
        let filters_params: usize = self.filters.iter().map(|f| f.parameter_count()).sum();
        let output_params = self.output_proj.parameter_count();
        let local_conv_params =
            self.local_conv.as_ref().map(|lc| lc.parameter_count()).unwrap_or(0);

        projections_params + filters_params + output_params + local_conv_params
    }

    fn concatenate_tensors(&self, tensors: Vec<Tensor>) -> Result<Tensor> {
        // Concatenate along the hidden dimension
        // This is a simplified implementation
        if tensors.is_empty() {
            return Err(tensor_op_error(
                "tensor_operation",
                "No tensors to concatenate".to_string(),
            ));
        }

        let total_hidden = self.hidden_size * self.order;

        if tensors[0].shape().len() == 2 {
            // Handle 2D tensors: [seq_len, hidden_size]
            let seq_len = tensors[0].shape()[0];

            let mut result = Tensor::zeros(&[seq_len, total_hidden])?;

            for (i, tensor) in tensors.iter().enumerate() {
                let start_idx = i * self.hidden_size;

                // Copy tensor data to the appropriate slice
                for s in 0..seq_len {
                    for h in 0..self.hidden_size {
                        let val = tensor.get_scalar(&[s, h])?;
                        result = result.set_scalar(&[s, start_idx + h], val)?;
                    }
                }
            }

            Ok(result)
        } else {
            // Handle 3D tensors: [batch_size, seq_len, hidden_size]
            let batch_size = tensors[0].shape()[0];
            let seq_len = tensors[0].shape()[1];

            let mut result = Tensor::zeros(&[batch_size, seq_len, total_hidden])?;

            for (i, tensor) in tensors.iter().enumerate() {
                let start_idx = i * self.hidden_size;

                // Copy tensor data to the appropriate slice
                for b in 0..batch_size {
                    for s in 0..seq_len {
                        for h in 0..self.hidden_size {
                            let val = tensor.get_scalar(&[b, s, h])?;
                            result = result.set_scalar(&[b, s, start_idx + h], val)?;
                        }
                    }
                }
            }

            Ok(result)
        }
    }
}

/// Local convolution for capturing short-range dependencies
pub struct LocalConvolution {
    kernel_size: usize,
    conv: Linear,
    padding: usize,
    device: Device,
}

impl LocalConvolution {
    pub fn new(config: &HyenaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &HyenaConfig, device: Device) -> Result<Self> {
        let kernel_size = config.conv_kernel_size;
        let padding = kernel_size / 2;

        // This would be a 1D convolution in practice
        let conv = Linear::new_with_device(
            config.hidden_size * kernel_size,
            config.hidden_size,
            config.bias,
            device,
        );

        Ok(Self {
            kernel_size,
            conv,
            padding,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.conv.parameter_count()
    }
}

impl Layer for LocalConvolution {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Simplified 1D convolution implementation
        // In practice, this would use optimized conv1d operations

        if input.shape().len() == 2 {
            // Handle 2D tensor [seq_len, hidden_size]
            let seq_len = input.shape()[0];
            let hidden_size = input.shape()[1];

            let mut output = Tensor::zeros(&[seq_len, hidden_size])?;

            for i in 0..seq_len {
                // Extract local window
                let mut window_data = Vec::new();
                for k in 0..self.kernel_size {
                    let pos = i as i32 - self.padding as i32 + k as i32;
                    if pos >= 0 && pos < seq_len as i32 {
                        for h in 0..hidden_size {
                            let val = input.get_scalar(&[pos as usize, h])?;
                            window_data.push(val);
                        }
                    } else {
                        // Padding
                        for _ in 0..hidden_size {
                            window_data.push(0.0);
                        }
                    }
                }

                // Apply convolution
                let window_tensor =
                    Tensor::from_vec(window_data, &[1, 1, self.kernel_size * hidden_size])?;
                let conv_output = self.conv.forward(window_tensor)?;

                // Set output
                for h in 0..hidden_size {
                    let val = conv_output.get_scalar(&[0, 0, h])?;
                    output = output.set_scalar(&[i, h], val)?;
                }
            }

            Ok(output)
        } else if input.shape().len() == 3 {
            // Handle 3D tensor [batch_size, seq_len, hidden_size]
            let batch_size = input.shape()[0];
            let seq_len = input.shape()[1];
            let hidden_size = input.shape()[2];

            let mut output = Tensor::zeros(&[batch_size, seq_len, hidden_size])?;

            for b in 0..batch_size {
                for i in 0..seq_len {
                    // Extract local window
                    let mut window_data = Vec::new();
                    for k in 0..self.kernel_size {
                        let pos = i as i32 - self.padding as i32 + k as i32;
                        if pos >= 0 && pos < seq_len as i32 {
                            for h in 0..hidden_size {
                                let val = input.get_scalar(&[b, pos as usize, h])?;
                                window_data.push(val);
                            }
                        } else {
                            // Padding
                            for _ in 0..hidden_size {
                                window_data.push(0.0);
                            }
                        }
                    }

                    // Apply convolution
                    let window_tensor =
                        Tensor::from_vec(window_data, &[1, 1, self.kernel_size * hidden_size])?;
                    let conv_output = self.conv.forward(window_tensor)?;

                    // Set output
                    for h in 0..hidden_size {
                        let val = conv_output.get_scalar(&[0, 0, h])?;
                        output = output.set_scalar(&[b, i, h], val)?;
                    }
                }
            }

            Ok(output)
        } else {
            Err(tensor_op_error(
                "tensor_operation",
                format!(
                    "Unsupported tensor shape for local convolution: {:?}",
                    input.shape()
                ),
            ))
        }
    }
}

/// Hyena block (replaces transformer block)
pub struct HyenaBlock {
    hyena_op: HyenaOperator,
    mlp: HyenaMLp,
    norm1: LayerNorm,
    norm2: LayerNorm,
    #[allow(dead_code)]
    dropout: f32,
    device: Device,
}

impl HyenaBlock {
    pub fn new(config: &HyenaConfig, seq_len: usize) -> Result<Self> {
        Self::new_with_device(config, seq_len, Device::CPU)
    }

    pub fn new_with_device(config: &HyenaConfig, seq_len: usize, device: Device) -> Result<Self> {
        let hyena_op = HyenaOperator::new_with_device(config, seq_len, device)?;
        let mlp = HyenaMLp::new_with_device(config, device)?;
        let norm1 =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;
        let norm2 =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;

        Ok(Self {
            hyena_op,
            mlp,
            norm1,
            norm2,
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.hyena_op.parameter_count()
            + self.mlp.parameter_count()
            + self.norm1.parameter_count()
            + self.norm2.parameter_count()
    }
}

impl Layer for HyenaBlock {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Pre-norm: norm -> hyena -> residual
        let normed = self.norm1.forward(input.clone())?;
        let hyena_out = self.hyena_op.forward(normed)?;
        let residual1 = input.add(&hyena_out)?;

        // Pre-norm: norm -> mlp -> residual
        let normed2 = self.norm2.forward(residual1.clone())?;
        let mlp_out = self.mlp.forward(normed2)?;
        let residual2 = residual1.add(&mlp_out)?;

        Ok(residual2)
    }
}

/// MLP for Hyena block
pub struct HyenaMLp {
    up_proj: Linear,
    down_proj: Linear,
    activation: String,
    #[allow(dead_code)]
    dropout: f32,
    device: Device,
}

impl HyenaMLp {
    pub fn new(config: &HyenaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &HyenaConfig, device: Device) -> Result<Self> {
        let up_proj = Linear::new_with_device(
            config.hidden_size,
            config.intermediate_size,
            config.bias,
            device,
        );
        let down_proj = Linear::new_with_device(
            config.intermediate_size,
            config.hidden_size,
            config.bias,
            device,
        );

        Ok(Self {
            up_proj,
            down_proj,
            activation: config.hidden_act.clone(),
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.up_proj.parameter_count() + self.down_proj.parameter_count()
    }

    fn apply_activation(&self, x: &Tensor) -> Result<Tensor> {
        match self.activation.as_str() {
            "gelu" => x.gelu(),
            "relu" => x.relu(),
            "silu" | "swish" => x.silu(),
            _ => Ok(x.clone()),
        }
    }
}

impl Layer for HyenaMLp {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let up = self.up_proj.forward(input)?;
        let activated = self.apply_activation(&up)?;
        self.down_proj.forward(activated)
    }
}

/// Hyena embeddings
pub struct HyenaEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    layer_norm: LayerNorm,
    #[allow(dead_code)]
    dropout: f32,
    device: Device,
}

impl HyenaEmbeddings {
    pub fn new(config: &HyenaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &HyenaConfig, device: Device) -> Result<Self> {
        let word_embeddings = Embedding::new_with_device(
            config.vocab_size,
            config.hidden_size,
            Some(config.pad_token_id as usize),
            device,
        )?;

        let position_embeddings = if config.use_positional_embeddings {
            Some(Embedding::new_with_device(
                config.max_position_embeddings,
                config.hidden_size,
                None,
                device,
            )?)
        } else {
            None
        };

        let layer_norm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            layer_norm,
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        let mut count = self.word_embeddings.parameter_count();
        if let Some(pos_emb) = &self.position_embeddings {
            count += pos_emb.parameter_count();
        }
        count += self.layer_norm.parameter_count();
        count
    }
}

impl Layer for HyenaEmbeddings {
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let seq_len = input.len();

        // Word embeddings
        let mut embeddings = self.word_embeddings.forward(input)?;

        // Add positional embeddings if enabled
        if let Some(ref pos_emb) = self.position_embeddings {
            let position_ids: Vec<u32> = (0..seq_len as u32).collect();
            let pos_embeddings = pos_emb.forward(position_ids)?;
            embeddings = embeddings.add(&pos_embeddings)?;
        }

        // Apply layer norm
        embeddings = self.layer_norm.forward(embeddings)?;

        // Apply dropout (in training mode)
        Ok(embeddings)
    }
}

/// Main Hyena model
pub struct HyenaModel {
    config: HyenaConfig,
    embeddings: HyenaEmbeddings,
    layers: Vec<HyenaBlock>,
    final_norm: LayerNorm,
    device: Device,
}

impl HyenaModel {
    pub fn new(config: HyenaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: HyenaConfig, device: Device) -> Result<Self> {
        config.validate()?;

        let embeddings = HyenaEmbeddings::new_with_device(&config, device)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(HyenaBlock::new_with_device(
                &config,
                config.max_position_embeddings,
                device,
            )?);
        }

        let final_norm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;

        Ok(Self {
            config,
            embeddings,
            layers,
            final_norm,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for HyenaModel {
    type Config = HyenaConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let seq_len = input.len();
        let mut hidden_states = self.embeddings.forward(input)?;

        // Embeddings return [seq_len, hidden_size], but layers expect [batch_size, seq_len, hidden_size]
        // Add batch dimension
        hidden_states = hidden_states.reshape(&[1, seq_len, self.config.hidden_size])?;

        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        // Apply final norm
        hidden_states = self.final_norm.forward(hidden_states)?;

        // Remove batch dimension to return [seq_len, hidden_size]
        hidden_states.reshape(&[seq_len, self.config.hidden_size])
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Ok(())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Embedding parameters
        total += self.embeddings.parameter_count();

        // Layer parameters
        for layer in &self.layers {
            total += layer.parameter_count();
        }

        // Final norm parameters
        total += self.final_norm.parameter_count();

        total
    }
}

/// Hyena for language modeling
pub struct HyenaForLanguageModeling {
    hyena: HyenaModel,
    lm_head: Linear,
    device: Device,
}

impl HyenaForLanguageModeling {
    pub fn new(config: HyenaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: HyenaConfig, device: Device) -> Result<Self> {
        let hyena = HyenaModel::new_with_device(config.clone(), device)?;
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);

        Ok(Self {
            hyena,
            lm_head,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for HyenaForLanguageModeling {
    type Config = HyenaConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden_states = self.hyena.forward(input)?;
        self.lm_head.forward(hidden_states)
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.hyena.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.hyena.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.hyena.num_parameters() + self.lm_head.parameter_count()
    }
}

/// Hyena for sequence classification
pub struct HyenaForSequenceClassification {
    hyena: HyenaModel,
    classifier: Linear,
    #[allow(dead_code)]
    num_labels: usize,
    device: Device,
}

impl HyenaForSequenceClassification {
    pub fn new(config: HyenaConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    pub fn new_with_device(config: HyenaConfig, num_labels: usize, device: Device) -> Result<Self> {
        let hyena = HyenaModel::new_with_device(config.clone(), device)?;
        let classifier = Linear::new_with_device(config.hidden_size, num_labels, true, device);

        Ok(Self {
            hyena,
            classifier,
            num_labels,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for HyenaForSequenceClassification {
    type Config = HyenaConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let sequence_output = self.hyena.forward(input)?;

        // Use global average pooling for sequence classification
        let pooled = self.global_average_pool(&sequence_output)?;

        // Handle 1D pooled output - reshape to 2D for classifier
        let classifier_input = if pooled.shape().len() == 1 {
            let hidden_size = pooled.shape()[0];
            pooled.reshape(&[1, hidden_size])?
        } else {
            pooled
        };

        let logits = self.classifier.forward(classifier_input)?;

        // If output is 2D [1, num_labels], squeeze to 1D [num_labels]
        if logits.shape().len() == 2 && logits.shape()[0] == 1 {
            logits.reshape(&[logits.shape()[1]])
        } else {
            Ok(logits)
        }
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.hyena.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.hyena.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.hyena.num_parameters() + self.classifier.parameter_count()
    }
}

impl HyenaForSequenceClassification {
    fn global_average_pool(&self, x: &Tensor) -> Result<Tensor> {
        // Average pool along sequence dimension
        if x.shape().len() == 2 {
            // Handle 2D tensor [seq_len, hidden_size]
            let seq_len = x.shape()[0];
            let hidden_size = x.shape()[1];

            let mut pooled = Tensor::zeros(&[hidden_size])?;

            for h in 0..hidden_size {
                let mut sum = 0.0;
                for s in 0..seq_len {
                    sum += x.get_scalar(&[s, h])?;
                }
                let avg = sum / seq_len as f32;
                pooled = pooled.set_scalar(&[h], avg)?;
            }

            Ok(pooled)
        } else if x.shape().len() == 3 {
            // Handle 3D tensor [batch_size, seq_len, hidden_size]
            let batch_size = x.shape()[0];
            let seq_len = x.shape()[1];
            let hidden_size = x.shape()[2];

            let mut pooled = Tensor::zeros(&[batch_size, hidden_size])?;

            for b in 0..batch_size {
                for h in 0..hidden_size {
                    let mut sum = 0.0;
                    for s in 0..seq_len {
                        sum += x.get_scalar(&[b, s, h])?;
                    }
                    let avg = sum / seq_len as f32;
                    pooled = pooled.set_scalar(&[b, h], avg)?;
                }
            }

            Ok(pooled)
        } else {
            Err(tensor_op_error(
                "tensor_operation",
                format!("Unsupported tensor shape for pooling: {:?}", x.shape()),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> HyenaConfig {
        HyenaConfig {
            vocab_size: 1000,
            hidden_size: 256,
            num_hidden_layers: 4,
            intermediate_size: 1024,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 2048,
            initializer_range: 0.02,
            layer_norm_eps: 1e-5,
            pad_token_id: 0,

            // Hyena-specific parameters
            order: 2,
            filter_order: 64,
            local_order: 3,
            outer_mixing: true,
            conv_kernel_size: 3,
            use_positional_embeddings: false,
            short_filter_order: 3,
            modulate: true,
            w: 1.0,
            wd: 0.1,
            bias: true,
            num_inner_mlps: 2,
            normalized: false,
            use_flashfft: true,
        }
    }

    #[test]
    fn test_hyena_config_validation() {
        let mut config = create_test_config();
        assert!(config.validate().is_ok());

        // Test invalid order
        config.order = 1;
        assert!(config.validate().is_err());

        config.order = 2; // Reset

        // Test invalid filter order
        config.filter_order = 0;
        assert!(config.validate().is_err());

        config.filter_order = 64; // Reset

        // Test even kernel size
        config.conv_kernel_size = 4;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_hyena_config_presets() {
        let small = HyenaConfig::hyena_small();
        assert_eq!(small.hidden_size, 768);
        assert_eq!(small.num_hidden_layers, 12);
        assert!(small.validate().is_ok());

        let medium = HyenaConfig::hyena_medium();
        assert_eq!(medium.hidden_size, 1024);
        assert_eq!(medium.num_hidden_layers, 24);
        assert!(medium.validate().is_ok());

        let large = HyenaConfig::hyena_large();
        assert_eq!(large.hidden_size, 1280);
        assert_eq!(large.num_hidden_layers, 36);
        assert!(large.validate().is_ok());

        let dna = HyenaConfig::hyena_dna();
        assert_eq!(dna.vocab_size, 12);
        assert_eq!(dna.max_position_embeddings, 1048576);
        assert!(dna.validate().is_ok());
    }

    #[test]
    fn test_hyena_config_methods() {
        let config = create_test_config();

        // Test receptive field calculation
        let rf = config.receptive_field();
        assert_eq!(rf, config.filter_order * config.num_hidden_layers);

        // Test memory advantage calculation
        let advantage = config.memory_advantage();
        assert!(advantage > 1.0); // Should be better than attention

        // Test long context optimization check
        let mut long_config = config.clone();
        long_config.max_position_embeddings = 65536;
        long_config.use_flashfft = true;
        assert!(long_config.is_long_context_optimized());

        long_config.use_flashfft = false;
        assert!(!long_config.is_long_context_optimized());
    }

    #[test]
    fn test_hyena_filter_creation() {
        let config = create_test_config();
        let seq_len = 128;

        let filter = HyenaFilter::new(&config, seq_len);
        assert!(filter.is_ok());

        let filter = filter.unwrap();
        assert_eq!(filter.filter_order, config.filter_order);
        assert_eq!(filter.hidden_size, config.hidden_size);
        assert_eq!(filter.seq_len, seq_len);

        // Test parameter count
        let param_count = filter.parameter_count();
        assert!(param_count > 0);
    }

    #[test]
    fn test_hyena_filter_forward() {
        let config = create_test_config();
        let seq_len = 32;
        let batch_size = 2;

        let filter = HyenaFilter::new(&config, seq_len).unwrap();

        // Create test input
        let input_data: Vec<f32> = (0..batch_size * seq_len * config.hidden_size)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let input =
            Tensor::from_vec(input_data, &[batch_size, seq_len, config.hidden_size]).unwrap();

        let output = filter.forward(input);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, config.hidden_size]);
    }

    #[test]
    fn test_local_convolution() {
        let config = create_test_config();
        let local_conv = LocalConvolution::new(&config).unwrap();

        let batch_size = 2;
        let seq_len = 16;
        let input_data: Vec<f32> = (0..batch_size * seq_len * config.hidden_size)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let input =
            Tensor::from_vec(input_data, &[batch_size, seq_len, config.hidden_size]).unwrap();

        let output = local_conv.forward(input);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, config.hidden_size]);
    }

    #[test]
    fn test_hyena_operator() {
        let config = create_test_config();
        let seq_len = 16;
        let batch_size = 1;

        let hyena_op = HyenaOperator::new(&config, seq_len).unwrap();

        let input_data: Vec<f32> = (0..batch_size * seq_len * config.hidden_size)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let input =
            Tensor::from_vec(input_data, &[batch_size, seq_len, config.hidden_size]).unwrap();

        let output = hyena_op.forward(input);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, config.hidden_size]);

        // Test parameter count
        let param_count = hyena_op.parameter_count();
        assert!(param_count > 0);
    }

    #[test]
    fn test_hyena_mlp() {
        let config = create_test_config();
        let mlp = HyenaMLp::new(&config).unwrap();

        let batch_size = 2;
        let seq_len = 8;
        let input_data: Vec<f32> = (0..batch_size * seq_len * config.hidden_size)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let input =
            Tensor::from_vec(input_data, &[batch_size, seq_len, config.hidden_size]).unwrap();

        let output = mlp.forward(input);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, config.hidden_size]);

        // Test parameter count
        let param_count = mlp.parameter_count();
        assert!(param_count > 0);
    }

    #[test]
    fn test_hyena_embeddings() {
        let config = create_test_config();
        let embeddings = HyenaEmbeddings::new(&config).unwrap();

        let input_tokens = vec![1, 5, 10, 25, 50, 100];
        let output = embeddings.forward(input_tokens.clone());
        assert!(output.is_ok());

        let output = output.unwrap();
        // The embedding output shape should be [seq_len, hidden_size]
        assert_eq!(output.shape(), &[input_tokens.len(), config.hidden_size]);

        // Test parameter count
        let param_count = embeddings.parameter_count();
        assert!(param_count > 0);
    }

    #[test]
    fn test_hyena_block() {
        let config = create_test_config();
        let seq_len = 16;
        let block = HyenaBlock::new(&config, seq_len).unwrap();

        let batch_size = 1;
        let input_data: Vec<f32> = (0..batch_size * seq_len * config.hidden_size)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let input =
            Tensor::from_vec(input_data, &[batch_size, seq_len, config.hidden_size]).unwrap();

        let output = block.forward(input);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, config.hidden_size]);

        // Test parameter count
        let param_count = block.parameter_count();
        assert!(param_count > 0);
    }

    #[test]
    fn test_hyena_model() {
        let config = create_test_config();
        let model = HyenaModel::new(config.clone()).unwrap();

        let input_tokens = vec![1, 5, 10, 25, 50];
        let output = model.forward(input_tokens.clone());

        match &output {
            Ok(_) => {},
            Err(e) => panic!("Model forward failed: {}", e),
        }

        let output = output.unwrap();
        // Model output should be [seq_len, hidden_size] after processing
        assert_eq!(output.shape(), &[input_tokens.len(), config.hidden_size]);

        // Test configuration access
        assert_eq!(model.get_config().vocab_size, config.vocab_size);

        // Test parameter count
        let param_count = model.num_parameters();
        assert!(param_count > 0);

        // Verify it's substantial but not excessive
        assert!(param_count > 10000); // Should have significant parameters
        assert!(param_count < 10000000); // But not too many for test config
    }

    #[test]
    fn test_hyena_for_language_modeling() {
        let config = create_test_config();
        let model = HyenaForLanguageModeling::new(config.clone()).unwrap();

        let input_tokens = vec![1, 5, 10, 25];
        let output = model.forward(input_tokens.clone());
        assert!(output.is_ok());

        let output = output.unwrap();
        // Output should have vocab_size as last dimension for logits [seq_len, vocab_size]
        assert_eq!(output.shape(), &[input_tokens.len(), config.vocab_size]);

        // Test parameter count includes language modeling head
        let param_count = model.num_parameters();
        assert!(param_count > 0);
    }

    #[test]
    fn test_hyena_for_sequence_classification() {
        let config = create_test_config();
        let num_labels = 10;
        let model = HyenaForSequenceClassification::new(config.clone(), num_labels).unwrap();

        let input_tokens = vec![1, 5, 10, 25, 50, 100];
        let output = model.forward(input_tokens.clone());
        assert!(output.is_ok());

        let output = output.unwrap();
        // Output should be [num_labels] for classification after pooling
        assert_eq!(output.shape(), &[num_labels]);

        // Test parameter count includes classification head
        let param_count = model.num_parameters();
        assert!(param_count > 0);
    }

    #[test]
    fn test_hyena_memory_efficiency() {
        let attention_config = HyenaConfig {
            max_position_embeddings: 2048,
            filter_order: 64,
            ..create_test_config()
        };

        let advantage = attention_config.memory_advantage();

        // Hyena should have significant memory advantage over attention
        // For 2048 sequence length: attention needs 2048^2, Hyena needs 2048*64
        assert!(advantage > 30.0); // Should be much more efficient
    }

    #[test]
    fn test_hyena_long_sequence_optimization() {
        let short_config = HyenaConfig {
            max_position_embeddings: 1024,
            use_flashfft: false,
            ..create_test_config()
        };
        assert!(!short_config.is_long_context_optimized());

        let long_config = HyenaConfig {
            max_position_embeddings: 65536,
            use_flashfft: true,
            ..create_test_config()
        };
        assert!(long_config.is_long_context_optimized());
    }
}
