//! Hopfield associative memory networks.
//!
//! This module implements two complementary flavours of Hopfield dynamics that
//! share one memory state:
//!
//! * **Classical (binary) Hopfield networks** — patterns are stored with the
//!   Hebbian outer-product rule `W += x xᵀ` (zero diagonal) and recalled by
//!   iterating the synchronous update `s ← sign(W s)` until a fixpoint is
//!   reached. Stored bipolar patterns are fixpoints of that map, and probes
//!   within the basin of attraction converge onto them.
//! * **Modern (continuous) Hopfield networks** — recall is the softmax update
//!   `retrieve(q) = softmax(β · q Xᵀ) X` over the stored pattern matrix `X`,
//!   which is the update rule underlying "Hopfield Networks is All You Need".
//!
//! Both paths operate on the *same* stored patterns, so a pattern written with
//! [`HopfieldLayer::store_pattern`] can be recalled either way.

use trustformers_core::{
    errors::{Result, TrustformersError},
    layers::{LayerNorm, Linear},
    tensor::Tensor,
    traits::Layer,
};

use super::{config::BiologicalConfig, model::BiologicalModelOutput};

/// Replace row `row` of a 2-D tensor with `values`, returning a new tensor.
fn set_row(matrix: &Tensor, row: usize, values: &[f32]) -> Result<Tensor> {
    let shape = matrix.shape();
    if shape.len() != 2 {
        return Err(TrustformersError::shape_error(format!(
            "set_row expects a 2-D tensor, got shape {:?}",
            shape
        )));
    }
    if row >= shape[0] {
        return Err(TrustformersError::shape_error(format!(
            "row index {} out of bounds for {} rows",
            row, shape[0]
        )));
    }
    if values.len() != shape[1] {
        return Err(TrustformersError::shape_error(format!(
            "row length {} does not match tensor width {}",
            values.len(),
            shape[1]
        )));
    }

    let mut data = matrix.to_vec_f32()?;
    let offset = row * shape[1];
    data[offset..offset + shape[1]].copy_from_slice(values);
    Tensor::from_vec(data, &shape)
}

/// Zero the diagonal of a square matrix (self-connections are forbidden in a
/// classical Hopfield network — they would make every state a fixpoint).
fn zero_diagonal(matrix: &Tensor) -> Result<Tensor> {
    let shape = matrix.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(TrustformersError::shape_error(format!(
            "zero_diagonal expects a square 2-D tensor, got shape {:?}",
            shape
        )));
    }
    let n = shape[0];
    let mut data = matrix.to_vec_f32()?;
    for i in 0..n {
        data[i * n + i] = 0.0;
    }
    Tensor::from_vec(data, &shape)
}

/// Bipolar sign: strictly negative values map to `-1.0`, everything else
/// (including exact zero) maps to `+1.0`. This is the standard convention for
/// binary Hopfield networks, where zero-field neurons keep the positive state.
fn bipolar_sign(tensor: &Tensor) -> Result<Tensor> {
    let shape = tensor.shape();
    let data: Vec<f32> = tensor
        .to_vec_f32()?
        .into_iter()
        .map(|x| if x < 0.0 { -1.0 } else { 1.0 })
        .collect();
    Tensor::from_vec(data, &shape)
}

/// Normalise a pattern argument to a `[rows, d_model]` matrix.
fn as_pattern_matrix(pattern: &Tensor, d_model: usize) -> Result<Tensor> {
    let shape = pattern.shape();
    match shape.len() {
        1 if shape[0] == d_model => pattern.reshape(&[1, d_model]),
        2 if shape[1] == d_model => Ok(pattern.clone()),
        _ => Err(TrustformersError::shape_error(format!(
            "Hopfield pattern must be [d_model] or [rows, d_model] with d_model = {}, got {:?}",
            d_model, shape
        ))),
    }
}

/// Hopfield network memory state
#[derive(Debug, Clone)]
pub struct HopfieldMemoryState {
    /// Stored patterns, shape `[memory_capacity, d_model]`
    pub patterns: Tensor,
    /// Most recent per-slot attention/activation, shape `[batch, memory_capacity]`
    pub activations: Tensor,
    /// Hebbian association matrix, shape `[d_model, d_model]`, zero diagonal
    pub weights: Tensor,
    /// Current state, shape `[batch, d_model]`
    pub current_state: Tensor,
    /// Number of slots that hold an explicitly stored pattern
    pub stored_count: usize,
    /// Per-slot usage counter, used by the replacement policy when memory is full
    pub slot_usage: Vec<f32>,
}

/// Modern Hopfield network layer
#[derive(Debug)]
pub struct HopfieldLayer {
    /// Configuration
    pub config: BiologicalConfig,
    /// Query projection
    pub query_projection: Linear,
    /// Key projection
    pub key_projection: Linear,
    /// Value projection
    pub value_projection: Linear,
    /// Output projection
    pub output_projection: Linear,
    /// Memory state
    pub memory_state: Option<HopfieldMemoryState>,
    /// Layer normalization
    pub layer_norm: LayerNorm,
    /// Beta parameter for sharpness
    pub beta: f32,
}

impl HopfieldLayer {
    /// Create a new Hopfield layer
    pub fn new(config: &BiologicalConfig) -> Result<Self> {
        let d_model = config.d_model;

        let query_projection = Linear::new(d_model, d_model, config.use_bias);
        let key_projection = Linear::new(d_model, d_model, config.use_bias);
        let value_projection = Linear::new(d_model, d_model, config.use_bias);
        let output_projection = Linear::new(d_model, d_model, config.use_bias);
        let layer_norm = LayerNorm::new(vec![d_model], 1e-12)?;

        Ok(Self {
            config: config.clone(),
            query_projection,
            key_projection,
            value_projection,
            output_projection,
            memory_state: None,
            layer_norm,
            beta: 1.0,
        })
    }

    /// Initialize memory state
    pub fn init_memory(&mut self, batch_size: usize) -> Result<()> {
        let d_model = self.config.d_model;
        let memory_capacity = self.config.memory_capacity;

        // Stored patterns start from a small random initialisation so that the
        // modern-Hopfield softmax is well defined before anything is stored.
        let patterns = Tensor::randn(&[memory_capacity, d_model])?
            .scalar_mul(self.config.initializer_range)?;
        let activations = Tensor::zeros(&[batch_size, memory_capacity])?;
        // The Hebbian association matrix lives in pattern space, not slot space.
        let weights = Tensor::zeros(&[d_model, d_model])?;
        let current_state = Tensor::zeros(&[batch_size, d_model])?;

        self.memory_state = Some(HopfieldMemoryState {
            patterns,
            activations,
            weights,
            current_state,
            stored_count: 0,
            slot_usage: vec![0.0; memory_capacity],
        });

        Ok(())
    }

    fn memory(&self) -> Result<&HopfieldMemoryState> {
        self.memory_state.as_ref().ok_or_else(|| {
            TrustformersError::model_error("Hopfield memory state not initialized".to_string())
        })
    }

    fn memory_mut(&mut self) -> Result<&mut HopfieldMemoryState> {
        self.memory_state.as_mut().ok_or_else(|| {
            TrustformersError::model_error("Hopfield memory state not initialized".to_string())
        })
    }

    /// Forward pass through Hopfield layer.
    ///
    /// `input` is `[batch, seq_len, d_model]`; the output keeps that shape.
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(TrustformersError::shape_error(format!(
                "HopfieldLayer::forward expects [batch, seq_len, d_model], got {:?}",
                shape
            )));
        }
        let batch_size = shape[0];
        let seq_len = shape[1];

        if self.memory_state.is_none() {
            self.init_memory(batch_size)?;
        }

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let input_t = input.slice(1, t, t + 1)?.squeeze(1)?;
            let output_t = self.forward_timestep(&input_t)?;
            outputs.push(output_t.unsqueeze(1)?);
        }

        Tensor::concat(&outputs, 1)?.contiguous()
    }

    /// Forward pass for a single timestep (`input` is `[batch, d_model]`).
    fn forward_timestep(&mut self, input: &Tensor) -> Result<Tensor> {
        // Project input to query, and the stored patterns to keys/values.
        let query = self.query_projection.forward(input.clone())?;
        let (key, value) = {
            let memory_state = self.memory()?;
            let key = self.key_projection.forward(memory_state.patterns.clone())?;
            let value = self.value_projection.forward(memory_state.patterns.clone())?;
            (key, value)
        };

        // Modern Hopfield retrieval: softmax(beta * q Kᵀ) V.
        let attention_scores = query.matmul(&key.transpose(0, 1)?)?.mul_scalar(self.beta)?;
        let attention_weights = attention_scores.softmax(1)?;
        let output = attention_weights.matmul(&value)?;

        // Real memory update: Hebbian plasticity plus activation bookkeeping.
        self.update_memory_state(input, &output, &attention_weights)?;

        let normalized_output = self.layer_norm.forward(output)?;
        self.output_projection.forward(normalized_output)
    }

    /// Update the memory state after a retrieval step.
    ///
    /// * `activations` tracks the attention mass each memory slot received.
    /// * `weights` accumulates the Hebbian outer product of the presented
    ///   pattern, scaled by the configured plasticity learning rate.
    fn update_memory_state(
        &mut self,
        input: &Tensor,
        output: &Tensor,
        attention_weights: &Tensor,
    ) -> Result<()> {
        let learning_rate = self.config.learning_rate;
        // Hebbian outer product over the batch: Σ_b x_b x_bᵀ  -> [d_model, d_model].
        let outer_product = input.transpose(0, 1)?.matmul(input)?.mul_scalar(learning_rate)?;
        let outer_product = zero_diagonal(&outer_product)?;

        // Track which slots were used so the replacement policy is meaningful.
        let per_slot: Vec<f32> = {
            let shape = attention_weights.shape();
            let data = attention_weights.to_vec_f32()?;
            let (rows, cols) = (shape[0], shape[1]);
            (0..cols).map(|c| (0..rows).map(|r| data[r * cols + c]).sum::<f32>()).collect()
        };

        let memory_state = self.memory_mut()?;
        memory_state.current_state = output.clone();
        memory_state.activations = attention_weights.clone();
        memory_state.weights = memory_state.weights.add(&outer_product)?;
        for (slot, mass) in memory_state.slot_usage.iter_mut().zip(per_slot.iter()) {
            *slot += *mass;
        }

        Ok(())
    }

    /// Store one or more patterns into memory.
    ///
    /// Each row of `pattern` is written into a memory slot (the next free slot,
    /// or the least-used slot once memory is full) and folded into the Hebbian
    /// association matrix with `W += x xᵀ` (diagonal zeroed).
    pub fn store_pattern(&mut self, pattern: &Tensor) -> Result<()> {
        let d_model = self.config.d_model;
        let matrix = as_pattern_matrix(pattern, d_model)?;
        let rows = matrix.shape()[0];

        if self.memory_state.is_none() {
            self.init_memory(rows.max(1))?;
        }

        let capacity = self.config.memory_capacity;
        if capacity == 0 {
            return Err(TrustformersError::invalid_config(
                "memory_capacity must be greater than 0 to store Hopfield patterns".to_string(),
            ));
        }

        let flat = matrix.to_vec_f32()?;

        for row in 0..rows {
            let values = &flat[row * d_model..(row + 1) * d_model];
            let vector = Tensor::from_vec(values.to_vec(), &[1, d_model])?;
            // Hebbian storage: W += x xᵀ, no self-connections.
            let outer = zero_diagonal(&vector.transpose(0, 1)?.matmul(&vector)?)?;

            let memory_state = self.memory_mut()?;
            let slot = if memory_state.stored_count < capacity {
                let slot = memory_state.stored_count;
                memory_state.stored_count += 1;
                slot
            } else {
                // Least-frequently-used replacement over real usage statistics.
                let mut best = 0usize;
                let mut best_usage = f32::INFINITY;
                for (idx, usage) in memory_state.slot_usage.iter().enumerate() {
                    if *usage < best_usage {
                        best_usage = *usage;
                        best = idx;
                    }
                }
                best
            };

            memory_state.patterns = set_row(&memory_state.patterns, slot, values)?;
            memory_state.slot_usage[slot] = 0.0;
            memory_state.weights = memory_state.weights.add(&outer)?;
        }

        Ok(())
    }

    /// Number of patterns explicitly stored via [`Self::store_pattern`].
    pub fn stored_count(&self) -> usize {
        self.memory_state.as_ref().map(|m| m.stored_count).unwrap_or(0)
    }

    /// Index of the memory slot that best matches each query row.
    ///
    /// Returns one index per row of `query` (`query` is `[batch, d_model]` or
    /// `[d_model]`). Only slots that hold an explicitly stored pattern are
    /// considered; if nothing has been stored yet all slots are candidates.
    pub fn best_match_indices(&self, query: &Tensor) -> Result<Vec<usize>> {
        let d_model = self.config.d_model;
        let query_matrix = as_pattern_matrix(query, d_model)?;
        let memory_state = self.memory()?;

        let capacity = memory_state.patterns.shape()[0];
        let searchable =
            if memory_state.stored_count > 0 { memory_state.stored_count } else { capacity };

        let patterns = memory_state.patterns.slice(0, 0, searchable)?;
        let similarities = query_matrix.matmul(&patterns.transpose(0, 1)?)?;
        let indices = similarities.argmax(1)?.to_vec_f32()?;

        Ok(indices.into_iter().map(|i| (i.max(0.0) as usize).min(searchable - 1)).collect())
    }

    /// Retrieve the stored pattern that best matches each query row.
    ///
    /// The returned tensor has shape `[rows, d_model]`, one recalled pattern per
    /// query row, selected by the *actual* similarity argmax.
    pub fn retrieve_pattern(&mut self, query: &Tensor) -> Result<Tensor> {
        let d_model = self.config.d_model;
        let indices = self.best_match_indices(query)?;

        let memory_state = self.memory_mut()?;
        let mut rows = Vec::with_capacity(indices.len());
        for &index in &indices {
            memory_state.slot_usage[index] += 1.0;
            rows.push(memory_state.patterns.select(0, index as i64)?.reshape(&[1, d_model])?);
        }

        Tensor::concat(&rows, 0)
    }

    /// Modern (continuous) Hopfield retrieval: `softmax(β · q Xᵀ) X`.
    ///
    /// `query` is `[rows, d_model]` (or `[d_model]`); the result has shape
    /// `[rows, d_model]`. Large `beta` makes the retrieval converge to the
    /// nearest stored pattern; small `beta` produces a metastable mixture.
    pub fn modern_retrieval(&self, query: &Tensor, beta: f32) -> Result<Tensor> {
        let d_model = self.config.d_model;
        let query_matrix = as_pattern_matrix(query, d_model)?;
        let memory_state = self.memory()?;

        let capacity = memory_state.patterns.shape()[0];
        let searchable =
            if memory_state.stored_count > 0 { memory_state.stored_count } else { capacity };
        let patterns = memory_state.patterns.slice(0, 0, searchable)?;

        let scores = query_matrix.matmul(&patterns.transpose(0, 1)?)?.mul_scalar(beta)?;
        let weights = scores.softmax(1)?;
        weights.matmul(&patterns)
    }

    /// One synchronous classical Hopfield update: `s ← sign(W s)`.
    ///
    /// `state` is `[rows, d_model]` (or `[d_model]`).
    pub fn classical_step(&self, state: &Tensor) -> Result<Tensor> {
        let d_model = self.config.d_model;
        let state_matrix = as_pattern_matrix(state, d_model)?;
        let memory_state = self.memory()?;
        let field = state_matrix.matmul(&memory_state.weights)?;
        bipolar_sign(&field)
    }

    /// Iterate the classical Hopfield update until a fixpoint is reached.
    ///
    /// Units are updated *asynchronously* (one unit at a time, in index order,
    /// immediately visible to the units that follow). For a symmetric weight
    /// matrix with zero diagonal this update is guaranteed to decrease the
    /// Hopfield energy `E = -½ sᵀ W s` monotonically and therefore to converge
    /// to a fixpoint; the synchronous map of [`Self::classical_step`] can
    /// instead settle into a two-cycle.
    ///
    /// `max_sweeps` bounds the number of full passes over the units.
    pub fn run_classical_dynamics(&self, probe: &Tensor, max_sweeps: usize) -> Result<Tensor> {
        let d_model = self.config.d_model;
        let state_matrix = as_pattern_matrix(probe, d_model)?;
        let mut state = bipolar_sign(&state_matrix)?.to_vec_f32()?;
        let rows = state.len() / d_model;
        let weights = self.memory()?.weights.to_vec_f32()?;

        for _ in 0..max_sweeps {
            let mut changed = false;
            for row in 0..rows {
                for j in 0..d_model {
                    let mut field = 0.0f32;
                    for i in 0..d_model {
                        field += state[row * d_model + i] * weights[i * d_model + j];
                    }
                    let updated = if field < 0.0 { -1.0 } else { 1.0 };
                    if (updated - state[row * d_model + j]).abs() > f32::EPSILON {
                        state[row * d_model + j] = updated;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        Tensor::from_vec(state, &[rows, d_model])
    }

    /// Run modern Hopfield dynamics to convergence.
    ///
    /// Repeatedly applies `s ← softmax(β · s Xᵀ) X` until the update size falls
    /// below `1e-6` or `max_iterations` is exhausted.
    pub fn run_dynamics(&mut self, input: &Tensor, max_iterations: usize) -> Result<Tensor> {
        let d_model = self.config.d_model;
        let mut state = as_pattern_matrix(input, d_model)?;

        if self.memory_state.is_none() {
            self.init_memory(state.shape()[0])?;
        }

        let tolerance = 1e-6;
        for _ in 0..max_iterations {
            let new_state = self.modern_retrieval(&state, self.beta)?;
            let diff = new_state.sub(&state)?.pow(2.0)?.mean()?.sqrt()?.to_scalar()?;
            state = new_state;
            if diff < tolerance {
                break;
            }
        }

        Ok(state)
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
        let param_memory = self.parameter_count() as f32 * 4.0 / 1_000_000.0;
        let memory_state_size = if self.memory_state.is_some() {
            self.config.memory_capacity as f32 * self.config.d_model as f32 * 4.0 / 1_000_000.0
        } else {
            0.0
        };
        param_memory + memory_state_size
    }
}

/// Hopfield network model
#[derive(Debug)]
pub struct HopfieldNetwork {
    /// Configuration
    pub config: BiologicalConfig,
    /// Hopfield layers
    pub layers: Vec<HopfieldLayer>,
    /// Output projection
    pub output_projection: Linear,
}

impl HopfieldNetwork {
    /// Create a new Hopfield network
    pub fn new(config: &BiologicalConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.n_layer {
            layers.push(HopfieldLayer::new(config)?);
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

        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states)?;

            if let Some(memory_state) = &layer.memory_state {
                all_memory_states.push(memory_state.current_state.clone().unsqueeze(1)?);
            }
        }

        let output = self.output_projection.forward(hidden_states)?;

        let memory_states = if all_memory_states.is_empty() {
            None
        } else {
            Some(Tensor::concat(&all_memory_states, 1)?.contiguous()?)
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

    /// Update plasticity for all layers by storing the targets as patterns
    pub fn update_plasticity(&mut self, targets: &Tensor) -> Result<()> {
        for layer in &mut self.layers {
            layer.store_pattern(targets)?;
        }
        Ok(())
    }

    /// Reset states for all layers
    pub fn reset_states(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.memory_state = None;
        }
        Ok(())
    }

    /// Store patterns in all layers
    pub fn store_patterns(&mut self, patterns: &[Tensor]) -> Result<()> {
        for pattern in patterns {
            for layer in &mut self.layers {
                layer.store_pattern(pattern)?;
            }
        }
        Ok(())
    }

    /// Retrieve patterns from network
    pub fn retrieve_patterns(&mut self, queries: &[Tensor]) -> Result<Vec<Tensor>> {
        let mut retrieved = Vec::new();

        for query in queries {
            let layer = self.layers.first_mut().ok_or_else(|| {
                TrustformersError::model_error(
                    "Hopfield network has no layers to retrieve from".to_string(),
                )
            })?;
            retrieved.push(layer.retrieve_pattern(query)?);
        }

        Ok(retrieved)
    }

    /// Run associative memory retrieval through modern Hopfield dynamics
    pub fn associative_retrieval(&mut self, partial_input: &Tensor) -> Result<Tensor> {
        let mut current_state = partial_input.clone();

        for layer in &mut self.layers {
            current_state = layer.run_dynamics(&current_state, 50)?;
        }

        Ok(current_state)
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
