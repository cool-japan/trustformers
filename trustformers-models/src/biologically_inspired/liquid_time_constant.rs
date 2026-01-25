use trustformers_core::{
    errors::Result,
    layers::{LayerNorm, Linear},
    tensor::Tensor,
    traits::Layer,
};

use super::{config::BiologicalConfig, model::BiologicalModelOutput};

/// Liquid Time-Constant (LTC) neuron state
#[derive(Debug, Clone)]
pub struct LTCNeuronState {
    /// Neuron activations
    pub activations: Tensor,
    /// Time constants
    pub time_constants: Tensor,
    /// Sensory inputs
    pub sensory_inputs: Tensor,
    /// Inter-neuron connections
    pub inter_connections: Tensor,
}

/// Liquid Time-Constant Network layer
#[derive(Debug)]
pub struct LTCLayer {
    /// Configuration
    pub config: BiologicalConfig,
    /// Input projection
    pub input_projection: Linear,
    /// Sensory weight matrix
    pub sensory_weights: Linear,
    /// Inter-neuron weight matrix
    pub inter_weights: Linear,
    /// Output projection
    pub output_projection: Linear,
    /// Time constant parameters
    pub time_constant_params: Linear,
    /// Neuron state
    pub neuron_state: Option<LTCNeuronState>,
    /// Layer normalization
    pub layer_norm: LayerNorm,
}

impl LTCLayer {
    /// Create a new LTC layer
    pub fn new(config: &BiologicalConfig) -> Result<Self> {
        let d_model = config.d_model;
        let neurons_per_layer = config.neurons_per_layer;

        let input_projection = Linear::new(d_model, neurons_per_layer, config.use_bias);
        let sensory_weights = Linear::new(neurons_per_layer, neurons_per_layer, false);
        let inter_weights = Linear::new(neurons_per_layer, neurons_per_layer, false);
        let output_projection = Linear::new(neurons_per_layer, d_model, config.use_bias);
        let time_constant_params =
            Linear::new(neurons_per_layer, neurons_per_layer, config.use_bias);
        let layer_norm = LayerNorm::new(vec![d_model], 1e-12)?;

        Ok(Self {
            config: config.clone(),
            input_projection,
            sensory_weights,
            inter_weights,
            output_projection,
            time_constant_params,
            neuron_state: None,
            layer_norm,
        })
    }

    /// Initialize neuron state
    pub fn init_state(&mut self, batch_size: usize) -> Result<()> {
        let neurons_per_layer = self.config.neurons_per_layer;

        let activations = Tensor::zeros(&[batch_size, neurons_per_layer])?;
        let time_constants = Tensor::full(1.0, vec![batch_size, neurons_per_layer])?;
        let sensory_inputs = Tensor::zeros(&[batch_size, neurons_per_layer])?;
        let inter_connections = Tensor::zeros(&[batch_size, neurons_per_layer])?;

        self.neuron_state = Some(LTCNeuronState {
            activations,
            time_constants,
            sensory_inputs,
            inter_connections,
        });

        Ok(())
    }

    /// Forward pass through LTC layer
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        // Initialize state if not present
        if self.neuron_state.is_none() {
            self.init_state(batch_size)?;
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
            output = Tensor::concat(&[output, outputs[i].clone()], 1)?;
        }

        Ok(output)
    }

    /// Forward pass for a single timestep
    fn forward_timestep(&mut self, input: &Tensor) -> Result<Tensor> {
        let neuron_state = self.neuron_state.as_mut().expect("operation failed");
        let dt = self.config.dt;

        // Project input
        let projected_input = self.input_projection.forward(input.clone())?;

        // Compute sensory input
        let sensory_input = self.sensory_weights.forward(projected_input)?;

        // Compute inter-neuron connections
        let inter_input = self.inter_weights.forward(neuron_state.activations.clone())?;

        // Update time constants based on current state
        let time_constant_update =
            self.time_constant_params.forward(neuron_state.activations.clone())?;
        neuron_state.time_constants =
            time_constant_update.sigmoid()?.mul_scalar(10.0)?.add_scalar(0.1)?;

        // LTC dynamics: dx/dt = -x/tau + f(Ax + Bu + noise)
        let total_input = sensory_input.add(&inter_input)?;
        let activation_input = total_input.tanh()?;

        // Add noise
        let noise = Tensor::randn_like(&neuron_state.activations)?
            .mul_scalar(self.config.noise_variance)?;
        let noisy_input = activation_input.add(&noise)?;

        // Compute derivative
        let decay_term = neuron_state.activations.div(&neuron_state.time_constants)?;
        let dx_dt = decay_term.mul_scalar(-1.0)?.add(&noisy_input)?;

        // Update activations using forward Euler
        neuron_state.activations = neuron_state.activations.add(&dx_dt.mul_scalar(dt)?)?;

        // Apply leak rate
        neuron_state.activations =
            neuron_state.activations.mul_scalar(1.0 - self.config.leak_rate)?;

        // Update stored inputs
        neuron_state.sensory_inputs = sensory_input;
        neuron_state.inter_connections = inter_input;

        // Project to output dimension
        let output = self.output_projection.forward(neuron_state.activations.clone())?;

        // Apply layer normalization
        let normalized_output = self.layer_norm.forward(output)?;

        Ok(normalized_output)
    }

    /// Reset neuron state
    pub fn reset_state(&mut self) -> Result<()> {
        if let Some(state) = &mut self.neuron_state {
            state.activations = Tensor::zeros_like(&state.activations)?;
            state.time_constants = Tensor::ones_like(&state.time_constants)?;
            state.sensory_inputs = Tensor::zeros_like(&state.sensory_inputs)?;
            state.inter_connections = Tensor::zeros_like(&state.inter_connections)?;
        }
        Ok(())
    }

    /// Get current neuron activations
    pub fn get_activations(&self) -> Option<&Tensor> {
        self.neuron_state.as_ref().map(|s| &s.activations)
    }

    /// Get current time constants
    pub fn get_time_constants(&self) -> Option<&Tensor> {
        self.neuron_state.as_ref().map(|s| &s.time_constants)
    }

    /// Compute liquid state vector (for reservoir computing)
    pub fn compute_liquid_state(&self) -> Result<Tensor> {
        if let Some(state) = &self.neuron_state {
            // Combine activations with time constants for richer representation
            let weighted_activations = state.activations.mul(&state.time_constants)?;
            let liquid_state =
                Tensor::concat(&[state.activations.clone(), weighted_activations], 1)?;
            Ok(liquid_state)
        } else {
            Err(trustformers_core::errors::TrustformersError::model_error(
                "Neuron state not initialized".to_string(),
            ))
        }
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.input_projection.parameter_count()
            + self.sensory_weights.parameter_count()
            + self.inter_weights.parameter_count()
            + self.output_projection.parameter_count()
            + self.time_constant_params.parameter_count()
            + self.layer_norm.parameter_count()
    }

    /// Get memory usage in MB
    pub fn memory_usage(&self) -> f32 {
        let param_memory = self.parameter_count() as f32 * 4.0 / 1_000_000.0;
        let state_memory = if self.neuron_state.is_some() {
            self.config.neurons_per_layer as f32 * 4.0 * 4.0 / 1_000_000.0 // 4 tensors per layer
        } else {
            0.0
        };
        param_memory + state_memory
    }
}

/// Liquid Time-Constant Network model
#[derive(Debug)]
pub struct LiquidTimeConstantNetwork {
    /// Configuration
    pub config: BiologicalConfig,
    /// LTC layers
    pub layers: Vec<LTCLayer>,
    /// Readout layer
    pub readout: Linear,
    /// Output projection
    pub output_projection: Linear,
}

impl LiquidTimeConstantNetwork {
    /// Create a new LTC network
    pub fn new(config: &BiologicalConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.n_layer {
            layers.push(LTCLayer::new(config)?);
        }

        let readout = Linear::new(
            config.neurons_per_layer * config.n_layer,
            config.d_model,
            config.use_bias,
        );
        let output_projection = Linear::new(config.d_model, config.d_model, config.use_bias);

        Ok(Self {
            config: config.clone(),
            layers,
            readout,
            output_projection,
        })
    }

    /// Forward pass through the network
    pub fn forward(&mut self, input: &Tensor) -> Result<BiologicalModelOutput> {
        let mut hidden_states = input.clone();
        let mut all_liquid_states = Vec::new();

        // Pass through all layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states)?;

            // Collect liquid states
            if let Ok(liquid_state) = layer.compute_liquid_state() {
                all_liquid_states.push(liquid_state);
            }
        }

        // Compute readout from all liquid states
        let readout_input = if !all_liquid_states.is_empty() {
            let concatenated = Tensor::concat(&all_liquid_states, 1)?;
            Some(concatenated)
        } else {
            None
        };

        // Apply readout if available
        let output = if let Some(ref readout_input) = readout_input {
            let readout_output = self.readout.forward(readout_input.clone())?;
            self.output_projection.forward(readout_output)?
        } else {
            self.output_projection.forward(hidden_states)?
        };

        Ok(BiologicalModelOutput {
            hidden_states: output,
            spike_trains: None,
            memory_states: readout_input,
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

    /// Get all neuron activations
    pub fn get_all_activations(&self) -> Vec<Option<&Tensor>> {
        self.layers.iter().map(|l| l.get_activations()).collect()
    }

    /// Get all time constants
    pub fn get_all_time_constants(&self) -> Vec<Option<&Tensor>> {
        self.layers.iter().map(|l| l.get_time_constants()).collect()
    }

    /// Compute network stability measure
    pub fn compute_stability(&self) -> Result<f32> {
        let mut stability_sum = 0.0;
        let mut count = 0;

        for layer in &self.layers {
            if let Some(time_constants) = layer.get_time_constants() {
                let mean_tau = time_constants.mean()?.to_scalar()?;
                let std_tau = time_constants.std()?.to_scalar()?;
                let stability = mean_tau / (std_tau + 1e-8);
                stability_sum += stability;
                count += 1;
            }
        }

        if count > 0 {
            Ok(stability_sum / count as f32)
        } else {
            Ok(0.0)
        }
    }

    /// Adapt time constants based on input statistics
    pub fn adapt_time_constants(&mut self, input_variance: f32) -> Result<()> {
        for layer in &mut self.layers {
            if let Some(state) = &mut layer.neuron_state {
                // Adapt time constants based on input variance
                let adaptation_factor = (input_variance + 1e-8).sqrt();
                state.time_constants = state.time_constants.mul_scalar(adaptation_factor)?;

                // Clamp to reasonable range
                state.time_constants = state.time_constants.clamp(0.1, 10.0)?;
            }
        }
        Ok(())
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|l| l.parameter_count()).sum::<usize>()
            + self.readout.parameter_count()
            + self.output_projection.parameter_count()
    }

    /// Get memory usage in MB
    pub fn memory_usage(&self) -> f32 {
        self.layers.iter().map(|l| l.memory_usage()).sum::<f32>()
            + ((self.readout.parameter_count() + self.output_projection.parameter_count()) as f32
                * 4.0
                / 1_000_000.0)
    }
}
