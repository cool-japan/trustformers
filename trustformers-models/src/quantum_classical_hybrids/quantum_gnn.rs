use trustformers_core::{
    errors::Result,
    layers::{LayerNorm, Linear},
    quantum::QuantumNeuralLayer,
    tensor::Tensor,
    Layer,
};

use super::{config::QuantumClassicalConfig, model::QuantumClassicalModelOutput};

/// Quantum graph neural network
#[derive(Debug)]
pub struct QuantumGraphNeuralNetwork {
    /// Configuration
    pub config: QuantumClassicalConfig,
    /// Message passing layers
    pub message_passing_layers: Vec<Linear>,
    /// Quantum aggregation layers
    pub quantum_aggregation_layers: Vec<QuantumNeuralLayer>,
    /// Node update layers
    pub node_update_layers: Vec<Linear>,
    /// Layer normalization
    pub layer_norms: Vec<LayerNorm>,
    /// Output projection
    pub output_projection: Linear,
}

impl QuantumGraphNeuralNetwork {
    /// Create a new quantum GNN
    pub fn new(config: &QuantumClassicalConfig) -> Result<Self> {
        let mut message_passing_layers = Vec::new();
        let mut quantum_aggregation_layers = Vec::new();
        let mut node_update_layers = Vec::new();
        let mut layer_norms = Vec::new();

        for _ in 0..config.n_classical_layers {
            message_passing_layers.push(Linear::new(
                config.d_model,
                config.d_model,
                config.use_bias,
            ));
            node_update_layers.push(Linear::new(config.d_model, config.d_model, config.use_bias));
            layer_norms.push(LayerNorm::new(vec![config.d_model], 1e-12)?);
        }

        for _ in 0..config.n_quantum_layers {
            let ansatz = config.quantum_ansatz.clone().into();
            let parameters = vec![0.1; config.get_quantum_parameters_count()];
            quantum_aggregation_layers.push(QuantumNeuralLayer::new(
                config.num_qubits,
                ansatz,
                &parameters,
            )?);
        }

        let output_projection = Linear::new(config.d_model, config.d_model, config.use_bias);

        Ok(Self {
            config: config.clone(),
            message_passing_layers,
            quantum_aggregation_layers,
            node_update_layers,
            layer_norms,
            output_projection,
        })
    }

    /// Forward pass through quantum GNN
    pub fn forward(&mut self, input: &Tensor) -> Result<QuantumClassicalModelOutput> {
        let mut hidden_states = input.clone();
        let mut quantum_measurements = Vec::new();

        // Message passing and node updates
        for (i, ((mp_layer, update_layer), layer_norm)) in self
            .message_passing_layers
            .iter()
            .zip(self.node_update_layers.iter())
            .zip(self.layer_norms.iter())
            .enumerate()
        {
            // Message passing
            let messages = mp_layer.forward(hidden_states.clone())?;

            // Node updates
            let updated_nodes = update_layer.forward(messages)?;
            hidden_states = layer_norm.forward(updated_nodes)?;

            // Quantum aggregation
            if i < self.quantum_aggregation_layers.len() {
                let quantum_output = self.quantum_aggregation_layers[i].forward(&hidden_states)?;
                quantum_measurements.push(quantum_output.clone());
                hidden_states = quantum_output;
            }
        }

        let output = self.output_projection.forward(hidden_states.clone())?;

        let combined_measurements = if !quantum_measurements.is_empty() {
            let mut combined = quantum_measurements[0].clone();
            for i in 1..quantum_measurements.len() {
                combined = combined.add(&quantum_measurements[i])?;
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
        self.message_passing_layers.iter().map(|l| l.parameter_count()).sum::<usize>()
            + self.node_update_layers.iter().map(|l| l.parameter_count()).sum::<usize>()
            + self.layer_norms.iter().map(|l| l.parameter_count()).sum::<usize>()
            + self.output_projection.parameter_count()
            + self.quantum_aggregation_layers.len() * self.config.get_quantum_parameters_count()
    }

    /// Get memory usage in MB
    pub fn memory_usage(&self) -> f32 {
        self.parameter_count() as f32 * 4.0 / 1_000_000.0
    }
}
