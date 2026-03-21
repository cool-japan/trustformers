use trustformers_core::{
    errors::Result,
    layers::{LayerNorm, Linear},
    tensor::Tensor,
    Layer,
};

use super::{
    config::{BiologicalConfig, MemoryType, PlasticityType},
    model::BiologicalModelOutput,
};

/// Biological memory trace
#[derive(Debug, Clone)]
pub struct MemoryTrace {
    /// Memory content
    pub content: Tensor,
    /// Memory strength
    pub strength: f32,
    /// Memory age
    pub age: f32,
    /// Memory type
    pub memory_type: MemoryType,
}

/// Synaptic connection with plasticity
#[derive(Debug, Clone)]
pub struct SynapticConnection {
    /// Synaptic weight
    pub weight: f32,
    /// Plasticity trace
    pub plasticity_trace: f32,
    /// Metaplasticity state
    pub metaplasticity_state: f32,
}

/// Biological memory layer
#[derive(Debug)]
pub struct BiologicalMemoryLayer {
    /// Configuration
    pub config: BiologicalConfig,
    /// Memory traces
    pub memory_traces: Vec<MemoryTrace>,
    /// Synaptic connections
    pub synaptic_connections: Vec<Vec<SynapticConnection>>,
    /// Input projection
    pub input_projection: Linear,
    /// Memory encoding
    pub memory_encoding: Linear,
    /// Memory retrieval
    pub memory_retrieval: Linear,
    /// Output projection
    pub output_projection: Linear,
    /// Layer normalization
    pub layer_norm: LayerNorm,
}

impl BiologicalMemoryLayer {
    /// Create a new biological memory layer
    pub fn new(config: &BiologicalConfig) -> Result<Self> {
        let d_model = config.d_model;
        let memory_capacity = config.memory_capacity;

        let input_projection = Linear::new(d_model, d_model, config.use_bias);
        let memory_encoding = Linear::new(d_model, d_model, config.use_bias);
        let memory_retrieval = Linear::new(d_model, d_model, config.use_bias);
        let output_projection = Linear::new(d_model, d_model, config.use_bias);
        let layer_norm = LayerNorm::new(vec![d_model], 1e-12)?;

        // Initialize synaptic connections
        let mut synaptic_connections = Vec::new();
        for _ in 0..memory_capacity {
            let mut connections = Vec::new();
            for _ in 0..memory_capacity {
                connections.push(SynapticConnection {
                    weight: 0.0,
                    plasticity_trace: 0.0,
                    metaplasticity_state: 0.0,
                });
            }
            synaptic_connections.push(connections);
        }

        Ok(Self {
            config: config.clone(),
            memory_traces: Vec::new(),
            synaptic_connections,
            input_projection,
            memory_encoding,
            memory_retrieval,
            output_projection,
            layer_norm,
        })
    }

    /// Forward pass through biological memory layer
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let _batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

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
        // Project input
        let projected_input = self.input_projection.forward(input.clone())?;

        // Encode current input as memory
        let encoded_memory = self.memory_encoding.forward(projected_input.clone())?;

        // Store in memory traces
        self.store_memory_trace(&encoded_memory)?;

        // Retrieve relevant memories
        let retrieved_memories = self.retrieve_memories(&projected_input)?;

        // Combine input with retrieved memories
        let combined_input = if !retrieved_memories.is_empty() {
            let mut combined = projected_input.clone();
            for memory in &retrieved_memories {
                combined = combined.add(memory)?;
            }
            combined.div_scalar(1.0 + retrieved_memories.len() as f32)?
        } else {
            projected_input
        };

        // Apply layer normalization
        let normalized = self.layer_norm.forward(combined_input)?;

        // Final projection
        let output = self.output_projection.forward(normalized)?;

        Ok(output)
    }

    /// Store memory trace
    fn store_memory_trace(&mut self, memory: &Tensor) -> Result<()> {
        let memory_capacity = self.config.memory_capacity;

        // Create new memory trace
        let trace = MemoryTrace {
            content: memory.clone(),
            strength: 1.0,
            age: 0.0,
            memory_type: self.config.memory_type.clone(),
        };

        // Add to memory traces
        self.memory_traces.push(trace);

        // Enforce memory capacity
        if self.memory_traces.len() > memory_capacity {
            // Remove oldest/weakest memory
            let mut min_idx = 0;
            let mut min_strength = f32::INFINITY;

            for (i, trace) in self.memory_traces.iter().enumerate() {
                let effective_strength = trace.strength / (trace.age + 1.0);
                if effective_strength < min_strength {
                    min_strength = effective_strength;
                    min_idx = i;
                }
            }

            self.memory_traces.remove(min_idx);
        }

        Ok(())
    }

    /// Retrieve relevant memories
    fn retrieve_memories(&mut self, query: &Tensor) -> Result<Vec<Tensor>> {
        let mut retrieved = Vec::new();
        let threshold = 0.5; // Similarity threshold

        // Collect similarities first to avoid borrowing conflicts
        let mut similarities = Vec::new();
        for trace in &self.memory_traces {
            let similarity = self.compute_similarity(query, &trace.content)?;
            similarities.push(similarity);
        }

        // Now update the traces with the computed similarities
        for (i, trace) in self.memory_traces.iter_mut().enumerate() {
            let similarity = similarities[i];

            if similarity > threshold {
                // Strengthen memory (consolidation)
                trace.strength += 0.1 * similarity;
                trace.strength = trace.strength.min(2.0); // Cap strength

                // Retrieve memory
                retrieved.push(trace.content.clone());
            }

            // Age memory
            trace.age += 1.0;
        }

        Ok(retrieved)
    }

    /// Compute similarity between tensors
    fn compute_similarity(&self, a: &Tensor, b: &Tensor) -> Result<f32> {
        let dot_product = a.mul(b)?.sum_axes(&[1])?.mean()?;
        let norm_a = a.pow_scalar(2.0)?.sum_axes(&[1])?.sqrt()?.mean()?;
        let norm_b = b.pow_scalar(2.0)?.sum_axes(&[1])?.sqrt()?.mean()?;

        let similarity = dot_product.div(&norm_a.mul(&norm_b)?.add_scalar(1e-8)?)?;
        similarity.to_scalar()
    }

    /// Update plasticity based on memory type
    pub fn update_plasticity(&mut self, targets: &Tensor) -> Result<()> {
        match self.config.plasticity_type {
            PlasticityType::STDP => self.update_stdp_plasticity(targets)?,
            PlasticityType::Hebbian => self.update_hebbian_plasticity(targets)?,
            PlasticityType::Metaplasticity => self.update_metaplasticity(targets)?,
            _ => {}, // Other types not implemented for memory
        }
        Ok(())
    }

    /// Update STDP plasticity
    fn update_stdp_plasticity(&mut self, _targets: &Tensor) -> Result<()> {
        let learning_rate = self.config.learning_rate;
        let _stdp_window = self.config.stdp_window;

        for i in 0..self.synaptic_connections.len() {
            for j in 0..self.synaptic_connections[i].len() {
                let connection = &mut self.synaptic_connections[i][j];

                // Simple STDP update
                let weight_change = learning_rate * connection.plasticity_trace;
                connection.weight += weight_change;

                // Decay plasticity trace
                connection.plasticity_trace *= 0.95;
            }
        }

        Ok(())
    }

    /// Update Hebbian plasticity
    fn update_hebbian_plasticity(&mut self, _targets: &Tensor) -> Result<()> {
        let learning_rate = self.config.learning_rate;

        for i in 0..self.synaptic_connections.len() {
            for j in 0..self.synaptic_connections[i].len() {
                let connection = &mut self.synaptic_connections[i][j];

                // Hebbian update
                let weight_change = learning_rate * connection.plasticity_trace;
                connection.weight += weight_change;
            }
        }

        Ok(())
    }

    /// Update metaplasticity
    fn update_metaplasticity(&mut self, _targets: &Tensor) -> Result<()> {
        let learning_rate = self.config.learning_rate;

        for i in 0..self.synaptic_connections.len() {
            for j in 0..self.synaptic_connections[i].len() {
                let connection = &mut self.synaptic_connections[i][j];

                // Metaplasticity update
                let meta_factor = 1.0 / (1.0 + connection.metaplasticity_state);
                let weight_change = learning_rate * meta_factor * connection.plasticity_trace;
                connection.weight += weight_change;

                // Update metaplasticity state
                connection.metaplasticity_state += 0.01 * connection.plasticity_trace.abs();
                connection.metaplasticity_state *= 0.99; // Decay
            }
        }

        Ok(())
    }

    /// Reset memory
    pub fn reset_memory(&mut self) -> Result<()> {
        self.memory_traces.clear();

        // Reset synaptic connections
        for connections in &mut self.synaptic_connections {
            for connection in connections {
                connection.weight = 0.0;
                connection.plasticity_trace = 0.0;
                connection.metaplasticity_state = 0.0;
            }
        }

        Ok(())
    }

    /// Get memory traces
    pub fn get_memory_traces(&self) -> &Vec<MemoryTrace> {
        &self.memory_traces
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.input_projection.parameter_count()
            + self.memory_encoding.parameter_count()
            + self.memory_retrieval.parameter_count()
            + self.output_projection.parameter_count()
            + self.layer_norm.parameter_count()
    }

    /// Get memory usage in MB
    pub fn memory_usage(&self) -> f32 {
        let param_memory = self.parameter_count() as f32 * 4.0 / 1_000_000.0;
        let memory_traces_size =
            self.memory_traces.len() as f32 * self.config.d_model as f32 * 4.0 / 1_000_000.0;
        let synaptic_memory =
            (self.synaptic_connections.len() * self.synaptic_connections[0].len()) as f32 * 12.0
                / 1_000_000.0; // 3 f32 per connection
        param_memory + memory_traces_size + synaptic_memory
    }
}

/// Biological memory model
#[derive(Debug)]
pub struct BiologicalMemory {
    /// Configuration
    pub config: BiologicalConfig,
    /// Memory layers
    pub layers: Vec<BiologicalMemoryLayer>,
    /// Output projection
    pub output_projection: Linear,
}

impl BiologicalMemory {
    /// Create a new Biological Memory model
    pub fn new(config: &BiologicalConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.n_layer {
            layers.push(BiologicalMemoryLayer::new(config)?);
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
        let mut all_plasticity_traces = Vec::new();

        // Pass through all layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states)?;

            // Collect plasticity traces
            let traces = layer.get_memory_traces();
            if !traces.is_empty() {
                let trace_content = traces[0].content.clone();
                all_plasticity_traces.push(trace_content);
            }
        }

        // Project to output dimension
        let output = self.output_projection.forward(hidden_states)?;

        // Stack plasticity traces
        let plasticity_traces = if !all_plasticity_traces.is_empty() {
            let mut stacked = all_plasticity_traces[0].clone();
            for i in 1..all_plasticity_traces.len() {
                stacked = Tensor::concat(&[stacked, all_plasticity_traces[i].clone()], 1)?;
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
            dendritic_activations: None,
            plasticity_traces,
        })
    }

    /// Update plasticity for all layers
    pub fn update_plasticity(&mut self, targets: &Tensor) -> Result<()> {
        for layer in &mut self.layers {
            layer.update_plasticity(targets)?;
        }
        Ok(())
    }

    /// Reset states for all layers
    pub fn reset_states(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.reset_memory()?;
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
