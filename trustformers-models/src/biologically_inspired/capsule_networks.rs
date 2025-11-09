use trustformers_core::{
    errors::Result,
    layers::{LayerNorm, Linear},
    tensor::Tensor,
    traits::Layer,
};

use super::{config::BiologicalConfig, model::BiologicalModelOutput};

/// Capsule layer
#[derive(Debug)]
pub struct CapsuleLayer {
    /// Configuration
    pub config: BiologicalConfig,
    /// Capsule transformation matrices
    pub capsule_transform: Vec<Linear>,
    /// Routing coefficients
    pub routing_coefficients: Option<Tensor>,
    /// Layer normalization
    pub layer_norm: LayerNorm,
}

impl CapsuleLayer {
    /// Create a new capsule layer
    pub fn new(config: &BiologicalConfig) -> Result<Self> {
        let num_capsules = config.num_capsules;
        let capsule_dim = config.capsule_dim;
        let d_model = config.d_model;

        let mut capsule_transform = Vec::new();
        for _ in 0..num_capsules {
            capsule_transform.push(Linear::new(d_model, capsule_dim, config.use_bias));
        }

        let layer_norm = LayerNorm::new(vec![num_capsules * capsule_dim], 1e-12)?;

        Ok(Self {
            config: config.clone(),
            capsule_transform,
            routing_coefficients: None,
            layer_norm,
        })
    }

    /// Forward pass with dynamic routing
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let num_capsules = self.config.num_capsules;
        let capsule_dim = self.config.capsule_dim;

        // Initialize routing coefficients
        if self.routing_coefficients.is_none() {
            self.routing_coefficients = Some(Tensor::zeros(&[batch_size, seq_len, num_capsules])?);
        }

        let mut capsule_outputs = Vec::new();

        // Transform input for each capsule
        for transform in self.capsule_transform.iter() {
            let transformed = transform.forward(input.clone())?;
            capsule_outputs.push(transformed);
        }

        // Dynamic routing algorithm
        let mut routing_logits = Tensor::zeros(&[batch_size, seq_len, num_capsules])?;

        for iteration in 0..self.config.routing_iterations {
            // Compute routing probabilities
            let routing_probs = routing_logits.softmax(2)?;

            // Compute weighted sum of capsule outputs
            let mut weighted_sum = Tensor::zeros(&[batch_size, seq_len, capsule_dim])?;

            for (i, capsule_output) in capsule_outputs.iter().enumerate() {
                let weight = routing_probs.slice(2, i, i + 1)?;
                let weighted = capsule_output.mul(&weight)?;
                weighted_sum = weighted_sum.add(&weighted)?;
            }

            // Apply squashing function
            let squashed = self.squash(&weighted_sum)?;

            // Update routing logits (except for last iteration)
            if iteration < self.config.routing_iterations - 1 {
                for (i, capsule_output) in capsule_outputs.iter().enumerate() {
                    let agreement = capsule_output.mul(&squashed)?.sum(Some(vec![2]), false)?;
                    let logit_update = routing_logits.slice(2, i, i + 1)?;
                    // Update routing logits (simplified assignment)
                    routing_logits = logit_update.add(&agreement.unsqueeze(2)?)?;
                }
            }

            if iteration == self.config.routing_iterations - 1 {
                // Final output - concatenate all capsule outputs
                let mut final_output = capsule_outputs[0].clone();
                for i in 1..capsule_outputs.len() {
                    let tensors = vec![final_output, capsule_outputs[i].clone()];
                    final_output = Tensor::concat(&tensors, 2)?;
                }

                // Apply layer normalization
                let normalized = self.layer_norm.forward(final_output)?;
                return Ok(normalized);
            }
        }

        // Fallback (should not reach here)
        let mut final_output = capsule_outputs[0].clone();
        for i in 1..capsule_outputs.len() {
            let tensors = vec![final_output, capsule_outputs[i].clone()];
            final_output = Tensor::concat(&tensors, 2)?;
        }

        let normalized = self.layer_norm.forward(final_output)?;
        Ok(normalized)
    }

    /// Squashing function for capsules
    fn squash(&self, input: &Tensor) -> Result<Tensor> {
        let squared_norm = input.pow_scalar(2.0)?.sum(Some(vec![2]), false)?.unsqueeze(2)?;
        let norm = squared_norm.sqrt()?;
        let scale = squared_norm.div(&norm.add_scalar(1e-8)?.mul(&norm.add_scalar(1.0)?)?)?;
        let squashed = input.mul(&scale)?;
        Ok(squashed)
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.capsule_transform.iter().map(|t| t.parameter_count()).sum::<usize>()
            + self.layer_norm.parameter_count()
    }

    /// Get memory usage in MB
    pub fn memory_usage(&self) -> f32 {
        self.parameter_count() as f32 * 4.0 / 1_000_000.0
    }
}

/// Capsule Network model
#[derive(Debug)]
pub struct CapsuleNetwork {
    /// Configuration
    pub config: BiologicalConfig,
    /// Capsule layers
    pub layers: Vec<CapsuleLayer>,
    /// Output projection
    pub output_projection: Linear,
}

impl CapsuleNetwork {
    /// Create a new Capsule Network
    pub fn new(config: &BiologicalConfig) -> Result<Self> {
        let mut layers = Vec::new();
        let output_dim = config.num_capsules * config.capsule_dim;

        for _ in 0..config.n_layer {
            layers.push(CapsuleLayer::new(config)?);
        }

        let output_projection = Linear::new(output_dim, config.d_model, config.use_bias);

        Ok(Self {
            config: config.clone(),
            layers,
            output_projection,
        })
    }

    /// Forward pass through the network
    pub fn forward(&mut self, input: &Tensor) -> Result<BiologicalModelOutput> {
        let mut hidden_states = input.clone();
        let mut all_capsule_outputs = Vec::new();

        // Pass through all layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states)?;
            all_capsule_outputs.push(hidden_states.clone());
        }

        // Project to output dimension
        let output = self.output_projection.forward(hidden_states)?;

        // Stack capsule outputs
        let capsule_outputs = if !all_capsule_outputs.is_empty() {
            let mut stacked = all_capsule_outputs[0].clone();
            for i in 1..all_capsule_outputs.len() {
                let tensors = vec![stacked, all_capsule_outputs[i].clone()];
                stacked = Tensor::concat(&tensors, 2)?;
            }
            Some(stacked)
        } else {
            None
        };

        Ok(BiologicalModelOutput {
            hidden_states: output,
            spike_trains: None,
            memory_states: None,
            attention_weights: None,
            capsule_outputs,
            dendritic_activations: None,
            plasticity_traces: None,
        })
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|l| l.parameter_count()).sum::<usize>()
            + self.output_projection.parameter_count()
    }

    /// Get memory usage in MB
    pub fn memory_usage(&self) -> f32 {
        self.layers.iter().map(|l| l.memory_usage()).sum::<f32>()
            + (self.output_projection.parameter_count() as f32 * 4.0 / 1_000_000.0)
    }
}
