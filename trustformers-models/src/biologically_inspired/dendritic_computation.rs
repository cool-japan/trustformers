use trustformers_core::{
    errors::Result,
    layers::{LayerNorm, Linear},
    tensor::Tensor,
    Layer,
};

use super::{config::BiologicalConfig, model::BiologicalModelOutput};

/// Dendritic compartment
#[derive(Debug, Clone)]
pub struct DendriticCompartment {
    /// Compartment activation
    pub activation: Tensor,
    /// Compartment weights
    pub weights: Tensor,
    /// Delay buffer
    pub delay_buffer: Vec<Tensor>,
    /// Buffer index
    pub buffer_index: usize,
}

/// Dendritic computation layer
#[derive(Debug)]
pub struct DendriticLayer {
    /// Configuration
    pub config: BiologicalConfig,
    /// Dendritic compartments
    pub compartments: Vec<DendriticCompartment>,
    /// Integration weights
    pub integration_weights: Linear,
    /// Output projection
    pub output_projection: Linear,
    /// Layer normalization
    pub layer_norm: LayerNorm,
}

impl DendriticLayer {
    /// Create a new dendritic layer
    pub fn new(config: &BiologicalConfig) -> Result<Self> {
        let num_compartments = config.num_compartments;
        let d_model = config.d_model;
        let neurons_per_layer = config.neurons_per_layer;

        let integration_weights = Linear::new(
            num_compartments * neurons_per_layer,
            d_model,
            config.use_bias,
        );
        let output_projection = Linear::new(d_model, d_model, config.use_bias);
        let layer_norm = LayerNorm::new(vec![d_model], 1e-12)?;

        Ok(Self {
            config: config.clone(),
            compartments: Vec::new(),
            integration_weights,
            output_projection,
            layer_norm,
        })
    }

    /// Initialize compartments
    pub fn init_compartments(&mut self, batch_size: usize) -> Result<()> {
        let num_compartments = self.config.num_compartments;
        let neurons_per_layer = self.config.neurons_per_layer;
        let delay_steps = (self.config.dendritic_delay / self.config.dt) as usize;

        self.compartments.clear();

        for _ in 0..num_compartments {
            let activation = Tensor::zeros(&[batch_size, neurons_per_layer])?;
            let weights = Tensor::randn(&[neurons_per_layer, self.config.d_model])?
                .mul_scalar(self.config.initializer_range)?;
            let delay_buffer =
                vec![Tensor::zeros(&[batch_size, neurons_per_layer])?; delay_steps.max(1)];

            self.compartments.push(DendriticCompartment {
                activation,
                weights,
                delay_buffer,
                buffer_index: 0,
            });
        }

        Ok(())
    }

    /// Forward pass through dendritic layer
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        // Initialize compartments if not present
        if self.compartments.is_empty() {
            self.init_compartments(batch_size)?;
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
        let mut compartment_outputs = Vec::new();

        // Process each compartment
        for compartment in &mut self.compartments {
            // Compute input to compartment
            let compartment_input = input.matmul(&compartment.weights)?;

            // Update delay buffer
            let buffer_len = compartment.delay_buffer.len();
            compartment.delay_buffer[compartment.buffer_index] = compartment_input;

            // Get delayed input
            let delayed_input = &compartment.delay_buffer[compartment.buffer_index];

            // Update compartment activation with leaky integration
            let leak_rate = self.config.leak_rate;
            compartment.activation = compartment
                .activation
                .mul_scalar(1.0 - leak_rate)?
                .add(&delayed_input.mul_scalar(leak_rate)?)?;

            // Apply nonlinearity
            compartment.activation = compartment.activation.tanh()?;

            // Update buffer index
            compartment.buffer_index = (compartment.buffer_index + 1) % buffer_len;

            compartment_outputs.push(compartment.activation.clone());
        }

        // Integrate compartment outputs
        let mut integrated = compartment_outputs[0].clone();
        for i in 1..compartment_outputs.len() {
            integrated = Tensor::concat(&[integrated, compartment_outputs[i].clone()], 1)?;
        }

        // Apply integration weights
        let output = self.integration_weights.forward(integrated)?;

        // Apply layer normalization
        let normalized = self.layer_norm.forward(output)?;

        // Final projection
        let final_output = self.output_projection.forward(normalized)?;

        Ok(final_output)
    }

    /// Get compartment activations
    pub fn get_compartment_activations(&self) -> Vec<Tensor> {
        self.compartments.iter().map(|c| c.activation.clone()).collect()
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        let compartment_params = self
            .compartments
            .iter()
            .map(|c| c.weights.shape().iter().product::<usize>())
            .sum::<usize>();

        compartment_params
            + self.integration_weights.parameter_count()
            + self.output_projection.parameter_count()
            + self.layer_norm.parameter_count()
    }

    /// Get memory usage in MB
    pub fn memory_usage(&self) -> f32 {
        self.parameter_count() as f32 * 4.0 / 1_000_000.0
    }
}

/// Dendritic computation model
#[derive(Debug)]
pub struct DendriticComputation {
    /// Configuration
    pub config: BiologicalConfig,
    /// Dendritic layers
    pub layers: Vec<DendriticLayer>,
    /// Output projection
    pub output_projection: Linear,
}

impl DendriticComputation {
    /// Create a new Dendritic Computation model
    pub fn new(config: &BiologicalConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.n_layer {
            layers.push(DendriticLayer::new(config)?);
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
        let mut all_dendritic_activations = Vec::new();

        // Pass through all layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states)?;

            // Collect dendritic activations
            let activations = layer.get_compartment_activations();
            if !activations.is_empty() {
                let mut concatenated = activations[0].clone();
                for i in 1..activations.len() {
                    concatenated = Tensor::concat(&[concatenated, activations[i].clone()], 1)?;
                }
                all_dendritic_activations.push(concatenated);
            }
        }

        // Project to output dimension
        let output = self.output_projection.forward(hidden_states)?;

        // Stack dendritic activations
        let dendritic_activations = if !all_dendritic_activations.is_empty() {
            let mut stacked = all_dendritic_activations[0].clone();
            for i in 1..all_dendritic_activations.len() {
                stacked = Tensor::concat(&[stacked, all_dendritic_activations[i].clone()], 2)?;
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
            capsule_outputs: None,
            dendritic_activations,
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
