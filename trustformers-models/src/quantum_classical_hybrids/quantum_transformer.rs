use trustformers_core::{
    errors::Result,
    layers::{LayerNorm, Linear},
    quantum::QuantumNeuralLayer,
    tensor::Tensor,
    traits::Layer,
};

use super::{config::QuantumClassicalConfig, model::QuantumClassicalModelOutput};

/// Quantum transformer model
#[derive(Debug)]
pub struct QuantumTransformer {
    /// Configuration
    pub config: QuantumClassicalConfig,
    /// Classical attention layers
    pub classical_attention_layers: Vec<Linear>,
    /// Quantum attention layers
    pub quantum_attention_layers: Vec<QuantumNeuralLayer>,
    /// Feed-forward layers
    pub feed_forward_layers: Vec<Linear>,
    /// Layer normalization
    pub layer_norms: Vec<LayerNorm>,
    /// Output projection
    pub output_projection: Linear,
}

impl QuantumTransformer {
    /// Create a new quantum transformer
    pub fn new(config: &QuantumClassicalConfig) -> Result<Self> {
        let mut classical_attention_layers = Vec::new();
        let mut quantum_attention_layers = Vec::new();
        let mut feed_forward_layers = Vec::new();
        let mut layer_norms = Vec::new();

        // Create classical attention layers
        for _ in 0..config.n_classical_layers {
            classical_attention_layers.push(Linear::new(
                config.d_model,
                config.d_model,
                config.use_bias,
            ));
            layer_norms.push(LayerNorm::new(vec![config.d_model], 1e-12)?);
        }

        // Create quantum attention layers
        for _ in 0..config.n_quantum_layers {
            let ansatz = config.quantum_ansatz.clone().into();
            let parameters = vec![0.1; config.get_quantum_parameters_count()];
            quantum_attention_layers.push(QuantumNeuralLayer::new(
                config.num_qubits,
                ansatz,
                &parameters,
            )?);
        }

        // Create feed-forward layers
        for _ in 0..config.n_classical_layers {
            feed_forward_layers.push(Linear::new(
                config.d_model,
                config.d_model * 4,
                config.use_bias,
            ));
        }

        let output_projection = Linear::new(config.d_model, config.d_model, config.use_bias);

        Ok(Self {
            config: config.clone(),
            classical_attention_layers,
            quantum_attention_layers,
            feed_forward_layers,
            layer_norms,
            output_projection,
        })
    }

    /// Forward pass through quantum transformer
    pub fn forward(&mut self, input: &Tensor) -> Result<QuantumClassicalModelOutput> {
        let mut hidden_states = input.clone();
        let mut quantum_measurements = Vec::new();

        // Process through classical attention layers
        for (i, (attention_layer, layer_norm)) in
            self.classical_attention_layers.iter().zip(self.layer_norms.iter()).enumerate()
        {
            // Self-attention (simplified)
            let attention_output = attention_layer.forward(hidden_states.clone())?;
            hidden_states = layer_norm.forward(attention_output)?;

            // Feed-forward
            if i < self.feed_forward_layers.len() {
                let ff_output = self.feed_forward_layers[i].forward(hidden_states.clone())?;
                hidden_states = ff_output.gelu()?;
            }
        }

        // Process through quantum attention layers
        for quantum_layer in &mut self.quantum_attention_layers {
            let quantum_output = quantum_layer.forward(&hidden_states.clone())?;
            quantum_measurements.push(quantum_output.clone());
            hidden_states = quantum_output;
        }

        // Final projection (clone before move to preserve for classical_activations)
        let output = self.output_projection.forward(hidden_states.clone())?;

        // Combine quantum measurements
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
        self.classical_attention_layers
            .iter()
            .map(|l| l.parameter_count())
            .sum::<usize>()
            + self.feed_forward_layers.iter().map(|l| l.parameter_count()).sum::<usize>()
            + self.layer_norms.iter().map(|l| l.parameter_count()).sum::<usize>()
            + self.output_projection.parameter_count()
            + self.quantum_attention_layers.len() * self.config.get_quantum_parameters_count()
    }

    /// Get memory usage in MB
    pub fn memory_usage(&self) -> f32 {
        self.parameter_count() as f32 * 4.0 / 1_000_000.0
    }
}
