use trustformers_core::{
    errors::{Result, TrustformersError},
    layers::{LayerNorm, Linear},
    quantum::{QuantumAnsatz, QuantumNeuralLayer},
    tensor::Tensor,
    traits::Layer,
};

use super::{config::QuantumClassicalConfig, model::QuantumClassicalModelOutput};

/// Receptive field of the 1-D convolutions, in sequence positions.
pub const CONV_KERNEL_SIZE: usize = 3;

/// Build the `im2col` window tensor of a 1-D convolution with "same" padding.
///
/// `input` is `[batch, seq_len, channels]`; the result is
/// `[batch, seq_len, channels * kernel_size]`, where window position `t`
/// concatenates the channel vectors at `t - k/2 ..= t + k/2`, zero-padded at
/// the sequence boundaries. Applying a `Linear(channels * kernel_size,
/// out_channels)` to that tensor is exactly a 1-D convolution.
pub fn conv1d_windows(input: &Tensor, kernel_size: usize) -> Result<Tensor> {
    if kernel_size == 0 || kernel_size.is_multiple_of(2) {
        return Err(TrustformersError::invalid_input(
            "conv1d kernel size must be odd and non-zero".to_string(),
        ));
    }

    let shape = input.shape();
    if shape.len() != 3 {
        return Err(TrustformersError::shape_error(format!(
            "conv1d_windows expects [batch, seq_len, channels], got {:?}",
            shape
        )));
    }
    let (batch, seq_len, channels) = (shape[0], shape[1], shape[2]);
    let radius = (kernel_size / 2) as isize;

    let data = input.to_vec_f32()?;
    let mut windows = vec![0.0f32; batch * seq_len * channels * kernel_size];

    for b in 0..batch {
        for t in 0..seq_len {
            for (tap, offset) in (-radius..=radius).enumerate() {
                let source = t as isize + offset;
                if source < 0 || source >= seq_len as isize {
                    continue; // zero padding
                }
                let source = source as usize;
                let src = (b * seq_len + source) * channels;
                let dst = ((b * seq_len + t) * kernel_size + tap) * channels;
                windows[dst..dst + channels].copy_from_slice(&data[src..src + channels]);
            }
        }
    }

    Tensor::from_vec(windows, &[batch, seq_len, channels * kernel_size])
}

/// Quantum convolutional neural network
#[derive(Debug)]
pub struct QuantumConvolutionalNN {
    /// Configuration
    pub config: QuantumClassicalConfig,
    /// Convolutional layers
    pub conv_layers: Vec<Linear>,
    /// Quantum pooling layers
    pub quantum_pooling_layers: Vec<QuantumNeuralLayer>,
    /// Fully connected layers
    pub fc_layers: Vec<Linear>,
    /// Layer normalization
    pub layer_norms: Vec<LayerNorm>,
    /// Output projection
    pub output_projection: Linear,
}

impl QuantumConvolutionalNN {
    /// Create a new quantum CNN
    pub fn new(config: &QuantumClassicalConfig) -> Result<Self> {
        let mut conv_layers = Vec::new();
        let mut quantum_pooling_layers = Vec::new();
        let mut fc_layers = Vec::new();
        let mut layer_norms = Vec::new();

        for _ in 0..config.n_classical_layers {
            // A genuine 1-D convolution: the kernel spans CONV_KERNEL_SIZE
            // sequence positions of every input channel.
            conv_layers.push(Linear::new(
                config.d_model * CONV_KERNEL_SIZE,
                config.d_model,
                config.use_bias,
            ));
            fc_layers.push(Linear::new(config.d_model, config.d_model, config.use_bias));
            layer_norms.push(LayerNorm::new(vec![config.d_model], 1e-12)?);
        }

        for _ in 0..config.n_quantum_layers {
            let ansatz = QuantumAnsatz::from(config.quantum_ansatz.clone());
            let parameters = vec![0.1; config.get_quantum_parameters_count()];
            quantum_pooling_layers.push(QuantumNeuralLayer::new(
                config.num_qubits,
                ansatz,
                &parameters,
            )?);
        }

        let output_projection = Linear::new(config.d_model, config.d_model, config.use_bias);

        Ok(Self {
            config: config.clone(),
            conv_layers,
            quantum_pooling_layers,
            fc_layers,
            layer_norms,
            output_projection,
        })
    }

    /// Forward pass through quantum CNN
    pub fn forward(&mut self, input: &Tensor) -> Result<QuantumClassicalModelOutput> {
        let mut hidden_states = input.clone();
        let mut quantum_measurements = Vec::new();

        // Convolutional layers with quantum pooling
        for (i, (conv_layer, layer_norm)) in
            self.conv_layers.iter().zip(self.layer_norms.iter()).enumerate()
        {
            // 1-D convolution over the sequence axis (im2col + GEMM).
            let windows = conv1d_windows(&hidden_states, CONV_KERNEL_SIZE)?;
            let conv_output = conv_layer.forward(windows)?;
            hidden_states = layer_norm.forward(conv_output)?;

            // Quantum pooling
            if i < self.quantum_pooling_layers.len() {
                let quantum_output = self.quantum_pooling_layers[i].forward(&hidden_states)?;
                quantum_measurements.push(quantum_output.clone());
                hidden_states = quantum_output;
            }
        }

        // Fully connected layers
        for fc_layer in &self.fc_layers {
            hidden_states = fc_layer.forward(hidden_states)?;
            hidden_states = hidden_states.relu()?;
        }

        let output = self.output_projection.forward(hidden_states.clone())?;

        let combined_measurements = if !quantum_measurements.is_empty() {
            let mut combined = quantum_measurements[0].clone();
            for measurement in quantum_measurements.iter().skip(1) {
                combined = combined.add(measurement)?;
            }
            Some(combined)
        } else {
            None
        };

        Ok(QuantumClassicalModelOutput {
            hidden_states: output,
            quantum_measurements: combined_measurements,
            classical_activations: Some(hidden_states),
            quantum_attention_weights: None,
            quantum_fidelity_scores: None,
            quantum_entanglement_measures: None,
            quantum_error_mitigation: None,
        })
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.conv_layers.iter().map(|l| l.parameter_count()).sum::<usize>()
            + self.fc_layers.iter().map(|l| l.parameter_count()).sum::<usize>()
            + self.layer_norms.iter().map(|l| l.parameter_count()).sum::<usize>()
            + self.output_projection.parameter_count()
            + self.quantum_pooling_layers.len() * self.config.get_quantum_parameters_count()
    }

    /// Get memory usage in MB
    pub fn memory_usage(&self) -> f32 {
        self.parameter_count() as f32 * 4.0 / 1_000_000.0
    }
}
