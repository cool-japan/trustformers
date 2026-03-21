use trustformers_core::{
    errors::Result,
    layers::{LayerNorm, Linear},
    tensor::Tensor,
    Layer,
};

use super::{config::BiologicalConfig, model::BiologicalModelOutput};

/// Reservoir state for Echo State Networks
#[derive(Debug, Clone)]
pub struct ReservoirState {
    /// Internal reservoir activations
    pub activations: Tensor,
    /// Reservoir weight matrix
    pub reservoir_weights: Tensor,
    /// Input weight matrix
    pub input_weights: Tensor,
}

/// Reservoir Computing layer (Echo State Network)
#[derive(Debug)]
pub struct ReservoirLayer {
    /// Configuration
    pub config: BiologicalConfig,
    /// Reservoir state
    pub reservoir_state: Option<ReservoirState>,
    /// Output projection (readout)
    pub readout: Linear,
    /// Layer normalization
    pub layer_norm: LayerNorm,
}

impl ReservoirLayer {
    /// Create a new reservoir layer
    pub fn new(config: &BiologicalConfig) -> Result<Self> {
        let reservoir_size = config.reservoir_size;
        let d_model = config.d_model;

        let readout = Linear::new(reservoir_size, d_model, config.use_bias);
        let layer_norm = LayerNorm::new(vec![d_model], 1e-12)?;

        Ok(Self {
            config: config.clone(),
            reservoir_state: None,
            readout,
            layer_norm,
        })
    }

    /// Initialize reservoir state
    pub fn init_reservoir(&mut self, batch_size: usize) -> Result<()> {
        let reservoir_size = self.config.reservoir_size;
        let d_model = self.config.d_model;

        // Initialize reservoir activations
        let activations = Tensor::zeros(&[batch_size, reservoir_size])?;

        // Initialize reservoir weights (sparse, random)
        let mut reservoir_weights = Tensor::randn(&[reservoir_size, reservoir_size])?;

        // Make reservoir weights sparse (simplified - reduce magnitude for sparsity effect)
        reservoir_weights = reservoir_weights.mul_scalar(0.1)?;

        // Scale to desired spectral radius
        let spectral_radius = self.config.spectral_radius;
        reservoir_weights = reservoir_weights.mul_scalar(spectral_radius)?;

        // Initialize input weights
        let input_weights =
            Tensor::randn(&[d_model, reservoir_size])?.scalar_mul(self.config.input_scaling)?;

        self.reservoir_state = Some(ReservoirState {
            activations,
            reservoir_weights,
            input_weights,
        });

        Ok(())
    }

    /// Forward pass through reservoir layer
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        // Initialize reservoir if not present
        if self.reservoir_state.is_none() {
            self.init_reservoir(batch_size)?;
        }

        let mut outputs = Vec::new();

        // Process each time step
        for t in 0..seq_len {
            let input_t = input.slice(1, t, t + 1)?.squeeze(1)?;
            let output_t = self.forward_timestep(&input_t)?;
            outputs.push(output_t);
        }

        // Stack outputs
        let mut output = outputs[0].clone();
        for i in 1..outputs.len() {
            let tensors = vec![output, outputs[i].clone()];
            output = Tensor::concat(&tensors, 1)?;
        }

        Ok(output)
    }

    /// Forward pass for a single timestep
    fn forward_timestep(&mut self, input: &Tensor) -> Result<Tensor> {
        let reservoir_state = self.reservoir_state.as_mut().expect("operation failed");
        let leak_rate = self.config.leak_rate;

        // Compute input to reservoir
        let input_to_reservoir = input.matmul(&reservoir_state.input_weights)?;

        // Compute reservoir recurrence
        let recurrent_activation =
            reservoir_state.activations.matmul(&reservoir_state.reservoir_weights)?;

        // Total activation
        let total_activation = input_to_reservoir.add(&recurrent_activation)?;

        // Apply nonlinearity (tanh)
        let new_activations = total_activation.tanh()?;

        // Apply leak rate (leaky integration)
        reservoir_state.activations = reservoir_state
            .activations
            .mul_scalar(1.0 - leak_rate)?
            .add(&new_activations.mul_scalar(leak_rate)?)?;

        // Compute output through readout
        let output = self.readout.forward(reservoir_state.activations.clone())?;

        // Apply layer normalization
        let normalized_output = self.layer_norm.forward(output)?;

        Ok(normalized_output)
    }

    /// Reset reservoir state
    pub fn reset_state(&mut self) -> Result<()> {
        if let Some(state) = &mut self.reservoir_state {
            state.activations = Tensor::zeros_like(&state.activations)?;
        }
        Ok(())
    }

    /// Get reservoir activations
    pub fn get_reservoir_activations(&self) -> Option<&Tensor> {
        self.reservoir_state.as_ref().map(|s| &s.activations)
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.readout.parameter_count() + self.layer_norm.parameter_count()
    }

    /// Get memory usage in MB
    pub fn memory_usage(&self) -> f32 {
        let param_memory = self.parameter_count() as f32 * 4.0 / 1_000_000.0;
        let reservoir_memory = if self.reservoir_state.is_some() {
            let reservoir_size = self.config.reservoir_size;
            (reservoir_size * reservoir_size + reservoir_size * self.config.d_model) as f32 * 4.0
                / 1_000_000.0
        } else {
            0.0
        };
        param_memory + reservoir_memory
    }
}

/// Reservoir Computing model
#[derive(Debug)]
pub struct ReservoirComputing {
    /// Configuration
    pub config: BiologicalConfig,
    /// Reservoir layers
    pub layers: Vec<ReservoirLayer>,
    /// Output projection
    pub output_projection: Linear,
}

impl ReservoirComputing {
    /// Create a new Reservoir Computing model
    pub fn new(config: &BiologicalConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.n_layer {
            layers.push(ReservoirLayer::new(config)?);
        }

        let output_projection = Linear::new(config.d_model, config.d_model, config.use_bias);

        Ok(Self {
            config: config.clone(),
            layers,
            output_projection,
        })
    }

    /// Forward pass through the network
    pub fn forward(&mut self, input: &Tensor) -> Result<BiologicalModelOutput> {
        let mut hidden_states = input.clone();
        let mut all_reservoir_states = Vec::new();

        // Pass through all layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states)?;

            // Collect reservoir states
            if let Some(activations) = layer.get_reservoir_activations() {
                all_reservoir_states.push(activations.clone());
            }
        }

        // Project to output dimension
        let output = self.output_projection.forward(hidden_states)?;

        // Stack reservoir states
        let memory_states = if !all_reservoir_states.is_empty() {
            let mut stacked = all_reservoir_states[0].clone();
            for i in 1..all_reservoir_states.len() {
                let tensors = vec![stacked, all_reservoir_states[i].clone()];
                stacked = Tensor::concat(&tensors, 2)?;
            }
            Some(stacked)
        } else {
            None
        };

        Ok(BiologicalModelOutput {
            hidden_states: output,
            spike_trains: None,
            memory_states,
            attention_weights: None,
            capsule_outputs: None,
            dendritic_activations: None,
            plasticity_traces: None,
        })
    }

    /// Reset states for all layers
    pub fn reset_states(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.reset_state()?;
        }
        Ok(())
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
