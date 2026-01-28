use trustformers_core::{
    errors::Result,
    layers::{LayerNorm, Linear},
    quantum::QuantumAttention,
    tensor::Tensor,
    traits::Layer,
};

use super::{config::QuantumClassicalConfig, model::QuantumClassicalModelOutput};

/// Quantum attention layer
#[derive(Debug)]
pub struct QuantumAttentionLayer {
    /// Configuration
    pub config: QuantumClassicalConfig,
    /// Quantum attention mechanism
    pub quantum_attention: QuantumAttention,
    /// Classical query projection
    pub query_projection: Linear,
    /// Classical key projection
    pub key_projection: Linear,
    /// Classical value projection
    pub value_projection: Linear,
    /// Output projection
    pub output_projection: Linear,
    /// Layer normalization
    pub layer_norm: LayerNorm,
}

impl QuantumAttentionLayer {
    /// Create a new quantum attention layer
    pub fn new(config: &QuantumClassicalConfig) -> Result<Self> {
        let quantum_attention = QuantumAttention::new(config.d_model, config.num_qubits)?;
        let query_projection = Linear::new(config.d_model, config.d_model, config.use_bias);
        let key_projection = Linear::new(config.d_model, config.d_model, config.use_bias);
        let value_projection = Linear::new(config.d_model, config.d_model, config.use_bias);
        let output_projection = Linear::new(config.d_model, config.d_model, config.use_bias);
        let layer_norm = LayerNorm::new(vec![config.d_model], 1e-12)?;

        Ok(Self {
            config: config.clone(),
            quantum_attention,
            query_projection,
            key_projection,
            value_projection,
            output_projection,
            layer_norm,
        })
    }

    /// Forward pass through quantum attention
    pub fn forward(&mut self, input: &Tensor) -> Result<QuantumClassicalModelOutput> {
        let _batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        // Process each sequence position
        let mut outputs = Vec::new();
        let mut quantum_measurements = Vec::new();

        for t in 0..seq_len {
            let input_t = input.slice(1, t, t + 1)?.squeeze(1)?;

            // Classical projections
            let query = self.query_projection.forward(input_t.clone())?;
            let key = self.key_projection.forward(input_t.clone())?;
            let value = self.value_projection.forward(input_t)?;

            // Quantum attention
            let quantum_output = self.quantum_attention.forward(&query, &key, &value)?;
            quantum_measurements.push(quantum_output.clone());

            outputs.push(quantum_output);
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
            hidden_states: final_output.clone(),
            quantum_measurements: combined_measurements.clone(),
            classical_activations: Some(output),
            quantum_attention_weights: Some(combined_measurements.unwrap_or(final_output)),
            quantum_fidelity_scores: None,
            quantum_entanglement_measures: None,
            quantum_error_mitigation: None,
        })
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.query_projection.parameter_count()
            + self.key_projection.parameter_count()
            + self.value_projection.parameter_count()
            + self.output_projection.parameter_count()
            + self.layer_norm.parameter_count()
    }

    /// Get memory usage in MB
    pub fn memory_usage(&self) -> f32 {
        self.parameter_count() as f32 * 4.0 / 1_000_000.0
    }
}
