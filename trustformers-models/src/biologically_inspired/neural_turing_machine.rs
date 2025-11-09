use trustformers_core::{
    errors::Result,
    layers::{LayerNorm, Linear},
    tensor::Tensor,
    Layer,
};

use super::{config::BiologicalConfig, model::BiologicalModelOutput};

/// Neural Turing Machine memory bank
#[derive(Debug, Clone)]
pub struct NTMMemoryBank {
    /// Memory matrix (N x M)
    pub memory: Tensor,
    /// Read heads
    pub read_heads: Vec<NTMHead>,
    /// Write heads
    pub write_heads: Vec<NTMHead>,
    /// Memory size
    pub memory_size: (usize, usize), // (N, M)
}

/// Neural Turing Machine head
#[derive(Debug, Clone)]
pub struct NTMHead {
    /// Attention weights
    pub attention_weights: Tensor,
    /// Previous attention weights
    pub prev_attention_weights: Tensor,
    /// Key vector
    pub key: Tensor,
    /// Key strength
    pub key_strength: f32,
    /// Interpolation gate
    pub interpolation_gate: f32,
    /// Shift weights
    pub shift_weights: Tensor,
    /// Sharpening factor
    pub sharpening_factor: f32,
}

/// Neural Turing Machine layer
#[derive(Debug)]
pub struct NTMLayer {
    /// Configuration
    pub config: BiologicalConfig,
    /// Controller network
    pub controller: Linear,
    /// Memory bank
    pub memory_bank: Option<NTMMemoryBank>,
    /// Read head controllers
    pub read_head_controllers: Vec<Linear>,
    /// Write head controllers
    pub write_head_controllers: Vec<Linear>,
    /// Erase head controllers
    pub erase_head_controllers: Vec<Linear>,
    /// Add head controllers
    pub add_head_controllers: Vec<Linear>,
    /// Output projection
    pub output_projection: Linear,
    /// Layer normalization
    pub layer_norm: LayerNorm,
    /// Number of read heads
    pub num_read_heads: usize,
    /// Number of write heads
    pub num_write_heads: usize,
    /// Memory width
    pub memory_width: usize,
}

impl NTMLayer {
    /// Create a new NTM layer
    pub fn new(config: &BiologicalConfig) -> Result<Self> {
        let d_model = config.d_model;
        let _memory_capacity = config.memory_capacity;
        let memory_width = d_model; // Use d_model as memory width
        let num_read_heads = 1;
        let num_write_heads = 1;

        let controller = Linear::new(
            d_model + num_read_heads * memory_width,
            d_model,
            config.use_bias,
        );
        let output_projection = Linear::new(d_model, d_model, config.use_bias);
        let layer_norm = LayerNorm::new(vec![d_model], 1e-12)?;

        // Create head controllers
        let head_control_size = memory_width + 1 + 1 + 3 + 1; // key + strength + gate + shift + sharpening
        let mut read_head_controllers = Vec::new();
        let mut write_head_controllers = Vec::new();
        let mut erase_head_controllers = Vec::new();
        let mut add_head_controllers = Vec::new();

        for _ in 0..num_read_heads {
            read_head_controllers.push(Linear::new(d_model, head_control_size, config.use_bias));
        }

        for _ in 0..num_write_heads {
            write_head_controllers.push(Linear::new(d_model, head_control_size, config.use_bias));
            erase_head_controllers.push(Linear::new(d_model, memory_width, config.use_bias));
            add_head_controllers.push(Linear::new(d_model, memory_width, config.use_bias));
        }

        Ok(Self {
            config: config.clone(),
            controller,
            memory_bank: None,
            read_head_controllers,
            write_head_controllers,
            erase_head_controllers,
            add_head_controllers,
            output_projection,
            layer_norm,
            num_read_heads,
            num_write_heads,
            memory_width,
        })
    }

    /// Initialize memory bank
    pub fn init_memory(&mut self, batch_size: usize) -> Result<()> {
        let memory_capacity = self.config.memory_capacity;
        let memory_width = self.memory_width;

        // Initialize memory matrix
        let memory = Tensor::zeros(&[batch_size, memory_capacity, memory_width])?;

        // Initialize read heads
        let mut read_heads = Vec::new();
        for _ in 0..self.num_read_heads {
            let attention_weights = Tensor::zeros(&[batch_size, memory_capacity])?;
            let prev_attention_weights = Tensor::zeros(&[batch_size, memory_capacity])?;
            let key = Tensor::zeros(&[batch_size, memory_width])?;
            let shift_weights = Tensor::zeros(&[batch_size, 3])?; // Left, center, right

            read_heads.push(NTMHead {
                attention_weights,
                prev_attention_weights,
                key,
                key_strength: 1.0,
                interpolation_gate: 0.0,
                shift_weights,
                sharpening_factor: 1.0,
            });
        }

        // Initialize write heads
        let mut write_heads = Vec::new();
        for _ in 0..self.num_write_heads {
            let attention_weights = Tensor::zeros(&[batch_size, memory_capacity])?;
            let prev_attention_weights = Tensor::zeros(&[batch_size, memory_capacity])?;
            let key = Tensor::zeros(&[batch_size, memory_width])?;
            let shift_weights = Tensor::zeros(&[batch_size, 3])?;

            write_heads.push(NTMHead {
                attention_weights,
                prev_attention_weights,
                key,
                key_strength: 1.0,
                interpolation_gate: 0.0,
                shift_weights,
                sharpening_factor: 1.0,
            });
        }

        self.memory_bank = Some(NTMMemoryBank {
            memory,
            read_heads,
            write_heads,
            memory_size: (memory_capacity, memory_width),
        });

        Ok(())
    }

    /// Forward pass through NTM layer
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        // Initialize memory if not present
        if self.memory_bank.is_none() {
            self.init_memory(batch_size)?;
        }

        let mut outputs = Vec::new();

        // Process each time step
        for t in 0..seq_len {
            let input_t = input.slice(1, t, t + 1)?.squeeze(1)?;
            let output_t = self.forward_timestep(&input_t)?;
            outputs.push(output_t);
        }

        // Stack outputs
        let output = Tensor::concat(&outputs, 1)?;

        Ok(output)
    }

    /// Forward pass for a single timestep
    fn forward_timestep(&mut self, input: &Tensor) -> Result<Tensor> {
        // Read from memory (simplified to avoid borrowing conflicts)
        let read_vectors = {
            let memory_bank = self.memory_bank.as_ref().unwrap();
            // Simple read operation - average of memory rows weighted by attention
            let mut vectors = Vec::new();
            for head in &memory_bank.read_heads {
                let weights = &head.attention_weights;
                let read_vector = weights.matmul(&memory_bank.memory)?;
                vectors.push(read_vector);
            }
            vectors
        };

        // Combine input with read vectors
        let controller_input = if !read_vectors.is_empty() {
            let mut vectors = vec![input.clone()];
            vectors.extend(read_vectors.iter().cloned());
            Tensor::concat(&vectors, 1)?
        } else {
            input.clone()
        };

        // Controller forward pass
        let controller_output = self.controller.forward(controller_input)?;

        // Generate head control signals and write to memory (simplified)
        {
            let memory_bank = self.memory_bank.as_mut().unwrap();

            // Simple uniform attention weights for heads
            let memory_size = memory_bank.memory_size.0;
            let uniform_weights =
                Tensor::ones(&[1, memory_size])?.div_scalar(memory_size as f32)?;

            // Update read head attention weights
            for head in memory_bank.read_heads.iter_mut() {
                head.prev_attention_weights = head.attention_weights.clone();
                head.attention_weights = uniform_weights.clone();
            }

            // Update write head attention weights
            for head in memory_bank.write_heads.iter_mut() {
                head.prev_attention_weights = head.attention_weights.clone();
                head.attention_weights = uniform_weights.clone();
            }

            // Simple write to memory - just update with controller output
            let write_vector = self.output_projection.forward(controller_output.clone())?;
            if write_vector.shape()[1] == memory_bank.memory.shape()[1] {
                // Add a small portion of the write vector to all memory locations
                let update = write_vector.mul_scalar(0.01)?; // Small learning rate
                memory_bank.memory = memory_bank.memory.add(&update)?;
            }
        }

        // Project output
        let output = self.output_projection.forward(controller_output.clone())?;

        // Apply layer normalization
        let normalized_output = self.layer_norm.forward(output)?;

        Ok(normalized_output)
    }

    /// Read from memory using read heads
    #[allow(dead_code)]
    fn read_from_memory(&mut self, memory_bank: &mut NTMMemoryBank) -> Result<Vec<Tensor>> {
        let mut read_vectors = Vec::new();

        for head in &memory_bank.read_heads {
            // Compute read vector: sum over memory weighted by attention
            let read_vector = head
                .attention_weights
                .unsqueeze(2)?
                .matmul(&memory_bank.memory.unsqueeze(1)?)?
                .squeeze(1)?;
            read_vectors.push(read_vector);
        }

        Ok(read_vectors)
    }

    /// Update head control signals
    #[allow(dead_code)]
    fn update_head_controls(
        &mut self,
        controller_output: &Tensor,
        memory_bank: &mut NTMMemoryBank,
    ) -> Result<()> {
        // Collect control parameters for read heads first
        let mut read_control_params = Vec::new();
        for i in 0..memory_bank.read_heads.len() {
            let control_params =
                self.read_head_controllers[i].forward(controller_output.clone())?;
            read_control_params.push(control_params);
        }

        // Get memory capacity for head updates
        let memory_capacity = memory_bank.memory_size.0;

        // Update read heads with pre-computed control parameters
        for (i, head) in memory_bank.read_heads.iter_mut().enumerate() {
            self.update_head_from_params(head, &read_control_params[i], memory_capacity)?;
        }

        // Collect control parameters for write heads first
        let mut write_control_params = Vec::new();
        for i in 0..memory_bank.write_heads.len() {
            let control_params =
                self.write_head_controllers[i].forward(controller_output.clone())?;
            write_control_params.push(control_params);
        }

        // Update write heads with pre-computed control parameters
        for (i, head) in memory_bank.write_heads.iter_mut().enumerate() {
            self.update_head_from_params(head, &write_control_params[i], memory_capacity)?;
        }

        // Update all head attention weights (both read and write)
        for head in memory_bank.read_heads.iter_mut() {
            head.prev_attention_weights = head.attention_weights.clone();
            // For now, create simple uniform attention weights
            let memory_size = memory_bank.memory_size.0;
            head.attention_weights =
                Tensor::ones(&[1, memory_size])?.div_scalar(memory_size as f32)?;
        }

        for head in memory_bank.write_heads.iter_mut() {
            head.prev_attention_weights = head.attention_weights.clone();
            // For now, create simple uniform attention weights
            let memory_size = memory_bank.memory_size.0;
            head.attention_weights =
                Tensor::ones(&[1, memory_size])?.div_scalar(memory_size as f32)?;
        }

        Ok(())
    }

    /// Update head from control parameters
    fn update_head_from_params(
        &self,
        head: &mut NTMHead,
        params: &Tensor,
        _memory_capacity: usize,
    ) -> Result<()> {
        let memory_width = self.memory_width;

        // Parse control parameters
        let key = params.slice(1, 0, memory_width)?;
        let key_strength =
            params.slice(1, memory_width, memory_width + 1)?.sigmoid()?.mul_scalar(10.0)?; // 0-10 range
        let interpolation_gate = params.slice(1, memory_width + 1, memory_width + 2)?.sigmoid()?;
        let shift_weights = params.slice(1, memory_width + 2, memory_width + 5)?.softmax(1)?;
        let sharpening_factor = params
            .slice(1, memory_width + 5, memory_width + 6)?
            .sigmoid()?
            .mul_scalar(10.0)?
            .add_scalar(1.0)?; // 1-11 range

        // Update head parameters
        head.key = key;
        head.key_strength = key_strength.mean()?.to_scalar()?;
        head.interpolation_gate = interpolation_gate.mean()?.to_scalar()?;
        head.shift_weights = shift_weights;
        head.sharpening_factor = sharpening_factor.mean()?.to_scalar()?;

        // Note: Attention weights will be computed separately in the calling context

        Ok(())
    }

    /// Compute attention weights for a head
    #[allow(dead_code)]
    fn compute_attention_weights(
        &self,
        head: &NTMHead,
        memory_bank: &NTMMemoryBank,
    ) -> Result<Tensor> {
        let memory = &memory_bank.memory;
        let _memory_capacity = memory_bank.memory_size.0;

        // Content-based addressing
        let key_expanded = head.key.unsqueeze(1)?; // [batch, 1, memory_width]
        let similarities = key_expanded.matmul(&memory.transpose(1, 2)?)?; // [batch, 1, memory_capacity]
        let similarities = similarities.squeeze(1)?; // [batch, memory_capacity]

        // Apply key strength
        let content_weights = similarities.mul_scalar(head.key_strength)?.softmax(1)?;

        // Interpolation with previous weights
        let interpolated_weights = content_weights
            .mul_scalar(head.interpolation_gate)?
            .add(&head.prev_attention_weights.mul_scalar(1.0 - head.interpolation_gate)?)?;

        // Convolutional shift
        let shifted_weights =
            self.convolutional_shift(&interpolated_weights, &head.shift_weights)?;

        // Sharpening
        let sharpened_weights = shifted_weights.pow_scalar(head.sharpening_factor.into())?;
        let normalized_weights =
            sharpened_weights.div(&sharpened_weights.sum(Some(vec![1]), false)?.unsqueeze(1)?)?;

        Ok(normalized_weights)
    }

    /// Apply convolutional shift to attention weights
    fn convolutional_shift(&self, weights: &Tensor, shift_weights: &Tensor) -> Result<Tensor> {
        let _batch_size = weights.shape()[0];
        let memory_capacity = weights.shape()[1];

        let mut shifted = Tensor::zeros_like(weights)?;

        // Apply shifts: left (-1), center (0), right (+1)
        for i in 0..memory_capacity {
            let left_idx = if i == 0 { memory_capacity - 1 } else { i - 1 };
            let right_idx = if i == memory_capacity - 1 { 0 } else { i + 1 };

            let left_weight = shift_weights.slice(1, 0, 1)?;
            let center_weight = shift_weights.slice(1, 1, 2)?;
            let right_weight = shift_weights.slice(1, 2, 3)?;

            let left_contrib = weights.slice(1, left_idx, left_idx + 1)?.mul(&left_weight)?;
            let center_contrib = weights.slice(1, i, i + 1)?.mul(&center_weight)?;
            let right_contrib = weights.slice(1, right_idx, right_idx + 1)?.mul(&right_weight)?;

            let total_contrib = left_contrib.add(&center_contrib)?.add(&right_contrib)?;
            // Set this slice to the total contribution (simple assignment for this implementation)
            shifted = total_contrib;
        }

        Ok(shifted)
    }

    /// Write to memory using write heads
    #[allow(dead_code)]
    fn write_to_memory(
        &mut self,
        controller_output: &Tensor,
        memory_bank: &mut NTMMemoryBank,
    ) -> Result<()> {
        for (i, head) in memory_bank.write_heads.iter().enumerate() {
            // Generate erase and add vectors
            let erase_vector =
                self.erase_head_controllers[i].forward(controller_output.clone())?.sigmoid()?;
            let add_vector = self.add_head_controllers[i].forward(controller_output.clone())?;

            // Erase operation
            let erase_weights = head.attention_weights.unsqueeze(2)?; // [batch, memory_capacity, 1]
            let erase_matrix = erase_weights.matmul(&erase_vector.unsqueeze(1)?)?; // [batch, memory_capacity, memory_width]
            let one_minus_erase = Tensor::ones_like(&erase_matrix)?.sub(&erase_matrix)?;
            memory_bank.memory = memory_bank.memory.mul(&one_minus_erase)?;

            // Add operation
            let add_weights = head.attention_weights.unsqueeze(2)?; // [batch, memory_capacity, 1]
            let add_matrix = add_weights.matmul(&add_vector.unsqueeze(1)?)?; // [batch, memory_capacity, memory_width]
            memory_bank.memory = memory_bank.memory.add(&add_matrix)?;
        }

        Ok(())
    }

    /// Reset memory state
    pub fn reset_memory(&mut self) -> Result<()> {
        self.memory_bank = None;
        Ok(())
    }

    /// Get memory contents
    pub fn get_memory(&self) -> Option<&Tensor> {
        self.memory_bank.as_ref().map(|mb| &mb.memory)
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        let mut count = self.controller.parameter_count()
            + self.output_projection.parameter_count()
            + self.layer_norm.parameter_count();

        for controller in &self.read_head_controllers {
            count += controller.parameter_count();
        }

        for controller in &self.write_head_controllers {
            count += controller.parameter_count();
        }

        for controller in &self.erase_head_controllers {
            count += controller.parameter_count();
        }

        for controller in &self.add_head_controllers {
            count += controller.parameter_count();
        }

        count
    }

    /// Get memory usage in MB
    pub fn memory_usage(&self) -> f32 {
        let param_memory = self.parameter_count() as f32 * 4.0 / 1_000_000.0;
        let memory_bank_size = if self.memory_bank.is_some() {
            self.config.memory_capacity as f32 * self.memory_width as f32 * 4.0 / 1_000_000.0
        } else {
            0.0
        };
        param_memory + memory_bank_size
    }
}

/// Neural Turing Machine model
#[derive(Debug)]
pub struct NeuralTuringMachine {
    /// Configuration
    pub config: BiologicalConfig,
    /// NTM layers
    pub layers: Vec<NTMLayer>,
    /// Output projection
    pub output_projection: Linear,
}

impl NeuralTuringMachine {
    /// Create a new Neural Turing Machine
    pub fn new(config: &BiologicalConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.n_layer {
            layers.push(NTMLayer::new(config)?);
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
        let mut all_memory_states = Vec::new();

        // Pass through all layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states)?;

            // Collect memory states
            if let Some(memory) = layer.get_memory() {
                all_memory_states.push(memory.clone());
            }
        }

        // Project to output dimension
        let output = self.output_projection.forward(hidden_states)?;

        // Stack memory states
        let memory_states = if !all_memory_states.is_empty() {
            let stacked = Tensor::concat(&all_memory_states, 2)?;
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
            layer.reset_memory()?;
        }
        Ok(())
    }

    /// Get all memory contents
    pub fn get_all_memories(&self) -> Vec<Option<&Tensor>> {
        self.layers.iter().map(|l| l.get_memory()).collect()
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
