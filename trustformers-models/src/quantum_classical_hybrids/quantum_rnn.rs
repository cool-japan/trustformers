use trustformers_core::{
    errors::Result,
    layers::{LayerNorm, Linear},
    quantum::{QuantumAnsatz, QuantumNeuralLayer},
    tensor::Tensor,
    traits::Layer,
};

use super::{config::QuantumClassicalConfig, model::QuantumClassicalModelOutput};

/// Quantum recurrent neural network
#[derive(Debug)]
pub struct QuantumRecurrentNN {
    /// Configuration
    pub config: QuantumClassicalConfig,
    /// RNN layers
    pub rnn_layers: Vec<Linear>,
    /// Quantum memory layers
    pub quantum_memory_layers: Vec<QuantumNeuralLayer>,
    /// Hidden state
    pub hidden_state: Option<Tensor>,
    /// Layer normalization
    pub layer_norm: LayerNorm,
    /// Output projection
    pub output_projection: Linear,
}

impl QuantumRecurrentNN {
    /// Create a new quantum RNN
    pub fn new(config: &QuantumClassicalConfig) -> Result<Self> {
        let mut rnn_layers = Vec::new();
        let mut quantum_memory_layers = Vec::new();

        for _ in 0..config.n_classical_layers {
            rnn_layers.push(Linear::new(
                config.d_model * 2,
                config.d_model,
                config.use_bias,
            ));
        }

        for _ in 0..config.n_quantum_layers {
            let ansatz = QuantumAnsatz::from(config.quantum_ansatz.clone());
            let parameters = vec![0.1; config.get_quantum_parameters_count()];
            quantum_memory_layers.push(QuantumNeuralLayer::new(
                config.num_qubits,
                ansatz,
                &parameters,
            )?);
        }

        let layer_norm = LayerNorm::new(vec![config.d_model], 1e-12)?;
        let output_projection = Linear::new(config.d_model, config.d_model, config.use_bias);

        Ok(Self {
            config: config.clone(),
            rnn_layers,
            quantum_memory_layers,
            hidden_state: None,
            layer_norm,
            output_projection,
        })
    }

    /// Forward pass through quantum RNN
    pub fn forward(&mut self, input: &Tensor) -> Result<QuantumClassicalModelOutput> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let d_model = self.config.d_model;

        // Initialize hidden state if needed
        if self.hidden_state.is_none() {
            self.hidden_state = Some(Tensor::zeros(&[batch_size, d_model])?);
        }

        let mut outputs = Vec::new();
        let mut quantum_measurements = Vec::new();

        // Process sequence step by step
        for t in 0..seq_len {
            let input_t = input.slice(1, t, t + 1)?.squeeze(1)?;

            // Combine input with hidden state
            let combined =
                Tensor::concat(&[input_t, self.hidden_state.as_ref().unwrap().clone()], 1)?;

            // Process through RNN layers
            let mut hidden = combined;
            for rnn_layer in &self.rnn_layers {
                hidden = rnn_layer.forward(hidden)?;
                hidden = hidden.tanh()?;
            }

            // Update hidden state
            self.hidden_state = Some(hidden.clone());

            // Process through quantum memory
            for quantum_layer in &mut self.quantum_memory_layers {
                let quantum_output = quantum_layer.forward(&hidden)?;
                quantum_measurements.push(quantum_output.clone());
                hidden = quantum_output;
            }

            outputs.push(hidden);
        }

        // Stack outputs
        let output = Tensor::concat(&outputs, 1)?;

        let normalized = self.layer_norm.forward(output.clone())?;
        let final_output = self.output_projection.forward(normalized)?;

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
            hidden_states: final_output,
            quantum_measurements: combined_measurements,
            classical_activations: Some(output),
            quantum_attention_weights: None,
            quantum_fidelity_scores: None,
            quantum_entanglement_measures: None,
            quantum_error_mitigation: None,
        })
    }

    /// Reset hidden state
    pub fn reset_hidden_state(&mut self) {
        self.hidden_state = None;
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.rnn_layers.iter().map(|l| l.parameter_count()).sum::<usize>()
            + self.layer_norm.parameter_count()
            + self.output_projection.parameter_count()
            + self.quantum_memory_layers.len() * self.config.get_quantum_parameters_count()
    }

    /// Get memory usage in MB
    pub fn memory_usage(&self) -> f32 {
        self.parameter_count() as f32 * 4.0 / 1_000_000.0
    }
}
