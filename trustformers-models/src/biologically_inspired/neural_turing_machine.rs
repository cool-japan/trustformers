//! Neural Turing Machine with a full content + location addressing chain.
//!
//! The addressing pipeline follows Graves et al. (2014) exactly:
//!
//! 1. **Content addressing** — cosine similarity between the head's key vector
//!    and every memory row, sharpened by the key strength `β` and normalised
//!    with a softmax: `w_c = softmax(β · cos(k, M))`.
//! 2. **Interpolation** — blend with the previous weighting through the gate
//!    `g`: `w_g = g · w_c + (1 − g) · w_prev`.
//! 3. **Convolutional shift** — circular convolution with the 3-tap shift
//!    distribution `s` over offsets `{−1, 0, +1}`.
//! 4. **Sharpening** — `w = w_sᵞ / Σ w_sᵞ`.
//!
//! Reads are `r = Σ_i w_i M_i`; writes apply the erase/add pair
//! `M_i ← M_i (1 − w_i e) + w_i a`.

use trustformers_core::{
    errors::{Result, TrustformersError},
    layers::{LayerNorm, Linear},
    tensor::Tensor,
    Layer,
};

use super::{config::BiologicalConfig, model::BiologicalModelOutput};

/// Numerical floor used when normalising cosine similarities.
const COSINE_EPSILON: f32 = 1e-8;

/// Magnitude of the deterministic symmetry-breaking memory initialisation.
const MEMORY_INIT_SCALE: f32 = 1e-2;

/// Content-based addressing: `softmax(strength · cos(key, memory_row))`.
///
/// `memory` is one batch element laid out row-major as `n × m`.
pub(crate) fn content_addressing(
    key: &[f32],
    strength: f32,
    memory: &[f32],
    n: usize,
    m: usize,
) -> Vec<f32> {
    let key_norm = key.iter().map(|x| x * x).sum::<f32>().sqrt();

    let mut scores = Vec::with_capacity(n);
    for row in 0..n {
        let slice = &memory[row * m..(row + 1) * m];
        let dot: f32 = slice.iter().zip(key.iter()).map(|(a, b)| a * b).sum();
        let row_norm = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = dot / (row_norm * key_norm + COSINE_EPSILON);
        scores.push(strength * cosine);
    }

    softmax(&scores)
}

/// Numerically stable softmax over a slice.
pub(crate) fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        let uniform = 1.0 / scores.len().max(1) as f32;
        return vec![uniform; scores.len()];
    }
    let exponentials: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let total: f32 = exponentials.iter().sum();
    if total <= 0.0 {
        let uniform = 1.0 / scores.len().max(1) as f32;
        return vec![uniform; scores.len()];
    }
    exponentials.into_iter().map(|e| e / total).collect()
}

/// Circular convolution of the weighting with a 3-tap shift distribution.
///
/// `shift[0]` is the `−1` (rotate left) tap, `shift[1]` the identity tap and
/// `shift[2]` the `+1` (rotate right) tap.
pub(crate) fn circular_shift(weights: &[f32], shift: &[f32]) -> Vec<f32> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }
    let mut shifted = vec![0.0f32; n];
    for (i, slot) in shifted.iter_mut().enumerate() {
        // offset k in {-1, 0, +1}: w~[i] = Σ_k w[(i - k) mod n] · s[k + 1]
        for (tap, &weight) in shift.iter().enumerate().take(3) {
            let k = tap as isize - 1;
            let source = ((i as isize - k).rem_euclid(n as isize)) as usize;
            *slot += weights[source] * weight;
        }
    }
    shifted
}

/// Sharpen a weighting: `w^γ / Σ w^γ`.
pub(crate) fn sharpen(weights: &[f32], gamma: f32) -> Vec<f32> {
    let powered: Vec<f32> = weights.iter().map(|w| w.max(0.0).powf(gamma)).collect();
    let total: f32 = powered.iter().sum();
    if total <= 0.0 {
        let uniform = 1.0 / weights.len().max(1) as f32;
        return vec![uniform; weights.len()];
    }
    powered.into_iter().map(|p| p / total).collect()
}

/// Neural Turing Machine memory bank
#[derive(Debug, Clone)]
pub struct NTMMemoryBank {
    /// Memory matrix, shape `[batch, N, M]`
    pub memory: Tensor,
    /// Read heads
    pub read_heads: Vec<NTMHead>,
    /// Write heads
    pub write_heads: Vec<NTMHead>,
    /// Memory size `(N, M)`
    pub memory_size: (usize, usize),
}

/// Neural Turing Machine head.
///
/// All control signals are kept per batch element rather than collapsed to a
/// scalar, so each sequence in a batch addresses memory independently.
#[derive(Debug, Clone)]
pub struct NTMHead {
    /// Attention weights, shape `[batch, N]`
    pub attention_weights: Tensor,
    /// Previous attention weights, shape `[batch, N]`
    pub prev_attention_weights: Tensor,
    /// Key vector, shape `[batch, M]`
    pub key: Tensor,
    /// Key strength `β`, shape `[batch, 1]`
    pub key_strength: Tensor,
    /// Interpolation gate `g`, shape `[batch, 1]`
    pub interpolation_gate: Tensor,
    /// Shift distribution, shape `[batch, 3]`
    pub shift_weights: Tensor,
    /// Sharpening factor `γ`, shape `[batch, 1]`
    pub sharpening_factor: Tensor,
}

impl NTMHead {
    /// Create a head with a one-hot weighting focused on memory slot 0.
    fn new(batch_size: usize, memory_capacity: usize, memory_width: usize) -> Result<Self> {
        let mut focus = vec![0.0f32; batch_size * memory_capacity];
        for row in 0..batch_size {
            focus[row * memory_capacity] = 1.0;
        }
        let attention_weights = Tensor::from_vec(focus, &[batch_size, memory_capacity])?;

        Ok(Self {
            prev_attention_weights: attention_weights.clone(),
            attention_weights,
            key: Tensor::zeros(&[batch_size, memory_width])?,
            key_strength: Tensor::ones(&[batch_size, 1])?,
            interpolation_gate: Tensor::zeros(&[batch_size, 1])?,
            shift_weights: Tensor::full(1.0 / 3.0, vec![batch_size, 3])?,
            sharpening_factor: Tensor::ones(&[batch_size, 1])?,
        })
    }
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

        // key + strength + gate + shift(3) + sharpening
        let head_control_size = memory_width + 1 + 1 + 3 + 1;
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

        if memory_capacity == 0 {
            return Err(TrustformersError::invalid_config(
                "NTM requires memory_capacity > 0".to_string(),
            ));
        }

        // The initial memory must be non-zero (so cosine similarity is defined
        // on the first timestep) *and* must have linearly independent rows.
        // A constant memory makes every row point the same way, so content
        // addressing is exactly uniform for every key — an unstable fixed point
        // the write head can never escape, because writing a uniform weighting
        // into identical rows keeps them identical. The deterministic
        // low-discrepancy pattern below breaks that symmetry without
        // introducing run-to-run randomness.
        let mut initial = Vec::with_capacity(batch_size * memory_capacity * memory_width);
        for index in 0..batch_size * memory_capacity * memory_width {
            let phase = ((index + 1) as f32) * 0.754_877_7;
            initial.push(MEMORY_INIT_SCALE * (phase.fract() - 0.5));
        }
        let memory = Tensor::from_vec(initial, &[batch_size, memory_capacity, memory_width])?;

        let mut read_heads = Vec::new();
        for _ in 0..self.num_read_heads {
            read_heads.push(NTMHead::new(batch_size, memory_capacity, memory_width)?);
        }

        let mut write_heads = Vec::new();
        for _ in 0..self.num_write_heads {
            write_heads.push(NTMHead::new(batch_size, memory_capacity, memory_width)?);
        }

        self.memory_bank = Some(NTMMemoryBank {
            memory,
            read_heads,
            write_heads,
            memory_size: (memory_capacity, memory_width),
        });

        Ok(())
    }

    fn bank(&self) -> Result<&NTMMemoryBank> {
        self.memory_bank.as_ref().ok_or_else(|| {
            TrustformersError::model_error(
                "Neural Turing Machine memory bank not initialized".to_string(),
            )
        })
    }

    fn bank_mut(&mut self) -> Result<&mut NTMMemoryBank> {
        self.memory_bank.as_mut().ok_or_else(|| {
            TrustformersError::model_error(
                "Neural Turing Machine memory bank not initialized".to_string(),
            )
        })
    }

    /// Forward pass through NTM layer.
    ///
    /// `input` is `[batch, seq_len, d_model]`; the output keeps that shape.
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(TrustformersError::shape_error(format!(
                "NTMLayer::forward expects [batch, seq_len, d_model], got {:?}",
                shape
            )));
        }
        let batch_size = shape[0];
        let seq_len = shape[1];

        if self.memory_bank.is_none() {
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
        // 1. Read from memory with the current read-head weightings.
        let read_vectors = self.read_from_memory()?;

        // 2. Feed input + read vectors to the controller.
        let controller_input = if read_vectors.is_empty() {
            input.clone()
        } else {
            let mut parts = vec![input.clone()];
            parts.extend(read_vectors.iter().cloned());
            Tensor::concat(&parts, 1)?
        };
        let controller_output = self.controller.forward(controller_input)?;

        // 3. Update every head through the full addressing chain.
        self.update_head_controls(&controller_output)?;

        // 4. Write to memory with the freshly computed write weightings.
        self.write_to_memory(&controller_output)?;

        // 5. Project the controller output.
        let output = self.output_projection.forward(controller_output)?;
        self.layer_norm.forward(output)
    }

    /// Read from memory using the current read-head weightings.
    ///
    /// Returns one `[batch, memory_width]` vector per read head.
    pub fn read_from_memory(&self) -> Result<Vec<Tensor>> {
        let memory_bank = self.bank()?;
        let mut read_vectors = Vec::with_capacity(memory_bank.read_heads.len());

        for head in &memory_bank.read_heads {
            // [batch, 1, N] x [batch, N, M] -> [batch, 1, M]
            let weights = head.attention_weights.unsqueeze(1)?;
            let read_vector = weights.matmul(&memory_bank.memory)?.squeeze(1)?;
            read_vectors.push(read_vector);
        }

        Ok(read_vectors)
    }

    /// Recompute every head's addressing from the controller output.
    fn update_head_controls(&mut self, controller_output: &Tensor) -> Result<()> {
        let mut read_params = Vec::with_capacity(self.read_head_controllers.len());
        for controller in &self.read_head_controllers {
            read_params.push(controller.forward(controller_output.clone())?);
        }

        let mut write_params = Vec::with_capacity(self.write_head_controllers.len());
        for controller in &self.write_head_controllers {
            write_params.push(controller.forward(controller_output.clone())?);
        }

        let memory_width = self.memory_width;
        let memory = self.bank()?.memory.clone();

        {
            let memory_bank = self.bank_mut()?;
            for (head, params) in memory_bank.read_heads.iter_mut().zip(read_params.iter()) {
                Self::apply_head_params(head, params, memory_width)?;
                Self::recompute_attention(head, &memory)?;
            }
            for (head, params) in memory_bank.write_heads.iter_mut().zip(write_params.iter()) {
                Self::apply_head_params(head, params, memory_width)?;
                Self::recompute_attention(head, &memory)?;
            }
        }

        Ok(())
    }

    /// Parse a controller parameter vector into head control signals.
    fn apply_head_params(head: &mut NTMHead, params: &Tensor, memory_width: usize) -> Result<()> {
        let expected = memory_width + 6;
        let shape = params.shape();
        if shape.len() != 2 || shape[1] != expected {
            return Err(TrustformersError::shape_error(format!(
                "NTM head parameters must be [batch, {}], got {:?}",
                expected, shape
            )));
        }

        head.key = params.slice(1, 0, memory_width)?;
        // β in [1, 11]. Graves et al. require β > 0; a sigmoid alone can drive
        // it to ~1e-8, at which point `softmax(β · cos)` is numerically exactly
        // uniform and content addressing silently stops working. Offsetting by
        // 1 keeps the head at least as sharp as the raw cosine similarity,
        // which is the standard `softplus`-style parameterisation.
        head.key_strength = params
            .slice(1, memory_width, memory_width + 1)?
            .sigmoid()?
            .mul_scalar(10.0)?
            .add_scalar(1.0)?;
        head.interpolation_gate = params.slice(1, memory_width + 1, memory_width + 2)?.sigmoid()?;
        head.shift_weights = params.slice(1, memory_width + 2, memory_width + 5)?.softmax(1)?;
        // γ in [1, 11]: sharpening must not blur the weighting.
        head.sharpening_factor = params
            .slice(1, memory_width + 5, memory_width + 6)?
            .sigmoid()?
            .mul_scalar(10.0)?
            .add_scalar(1.0)?;

        Ok(())
    }

    /// Run the content → interpolation → shift → sharpen chain for one head.
    fn recompute_attention(head: &mut NTMHead, memory: &Tensor) -> Result<()> {
        let memory_shape = memory.shape();
        if memory_shape.len() != 3 {
            return Err(TrustformersError::shape_error(format!(
                "NTM memory must be [batch, N, M], got {:?}",
                memory_shape
            )));
        }
        let (batch, n, m) = (memory_shape[0], memory_shape[1], memory_shape[2]);

        let memory_data = memory.to_vec_f32()?;
        let key_data = head.key.to_vec_f32()?;
        let strength_data = head.key_strength.to_vec_f32()?;
        let gate_data = head.interpolation_gate.to_vec_f32()?;
        let shift_data = head.shift_weights.to_vec_f32()?;
        let gamma_data = head.sharpening_factor.to_vec_f32()?;
        let prev_data = head.attention_weights.to_vec_f32()?;

        let mut updated = Vec::with_capacity(batch * n);
        for b in 0..batch {
            let memory_b = &memory_data[b * n * m..(b + 1) * n * m];
            let key_b = &key_data[b * m..(b + 1) * m];
            let prev_b = &prev_data[b * n..(b + 1) * n];

            // 1. Content addressing.
            let content = content_addressing(key_b, strength_data[b], memory_b, n, m);

            // 2. Interpolation with the previous weighting.
            let gate = gate_data[b];
            let interpolated: Vec<f32> = content
                .iter()
                .zip(prev_b.iter())
                .map(|(c, p)| gate * c + (1.0 - gate) * p)
                .collect();

            // 3. Circular shift.
            let shifted = circular_shift(&interpolated, &shift_data[b * 3..(b + 1) * 3]);

            // 4. Sharpening.
            updated.extend(sharpen(&shifted, gamma_data[b]));
        }

        head.prev_attention_weights = head.attention_weights.clone();
        head.attention_weights = Tensor::from_vec(updated, &[batch, n])?;

        Ok(())
    }

    /// Apply the erase/add write for every write head.
    fn write_to_memory(&mut self, controller_output: &Tensor) -> Result<()> {
        let mut erase_vectors = Vec::with_capacity(self.erase_head_controllers.len());
        let mut add_vectors = Vec::with_capacity(self.add_head_controllers.len());
        for controller in &self.erase_head_controllers {
            erase_vectors.push(controller.forward(controller_output.clone())?.sigmoid()?);
        }
        for controller in &self.add_head_controllers {
            add_vectors.push(controller.forward(controller_output.clone())?);
        }

        let memory_bank = self.bank_mut()?;
        for (index, head) in memory_bank.write_heads.iter().enumerate() {
            let Some(erase_vector) = erase_vectors.get(index) else {
                continue;
            };
            let Some(add_vector) = add_vectors.get(index) else {
                continue;
            };

            // [batch, N, 1] x [batch, 1, M] -> [batch, N, M]
            let weights = head.attention_weights.unsqueeze(2)?;
            let erase_matrix = weights.matmul(&erase_vector.unsqueeze(1)?)?;
            let keep = Tensor::ones_like(&erase_matrix)?.sub(&erase_matrix)?;
            let add_matrix = weights.matmul(&add_vector.unsqueeze(1)?)?;

            memory_bank.memory = memory_bank.memory.mul(&keep)?.add(&add_matrix)?;
        }

        Ok(())
    }

    /// Write `content` into memory at the location addressed by `weights`.
    ///
    /// This is the erase/add write of the NTM specialised to a full erase, and
    /// is the primitive a copy task exercises: writing with a one-hot weighting
    /// stores `content` verbatim in that memory row.
    pub fn write_with_weights(&mut self, weights: &Tensor, content: &Tensor) -> Result<()> {
        let memory_bank = self.bank_mut()?;
        let expanded = weights.unsqueeze(2)?;
        let erase_matrix = expanded.matmul(&Tensor::ones_like(content)?.unsqueeze(1)?)?;
        let keep = Tensor::ones_like(&erase_matrix)?.sub(&erase_matrix)?;
        let add_matrix = expanded.matmul(&content.unsqueeze(1)?)?;
        memory_bank.memory = memory_bank.memory.mul(&keep)?.add(&add_matrix)?;
        Ok(())
    }

    /// Set the attention weighting of read head `index` (used to address reads).
    pub fn set_read_weights(&mut self, index: usize, weights: Tensor) -> Result<()> {
        let memory_bank = self.bank_mut()?;
        let head = memory_bank.read_heads.get_mut(index).ok_or_else(|| {
            TrustformersError::invalid_input(format!("read head {} does not exist", index))
        })?;
        head.prev_attention_weights = head.attention_weights.clone();
        head.attention_weights = weights;
        Ok(())
    }

    /// Content-addressed weighting for a key, without mutating any head state.
    ///
    /// `key` is `[batch, memory_width]`; the result is `[batch, N]`.
    pub fn content_weights(&self, key: &Tensor, strength: f32) -> Result<Tensor> {
        let memory_bank = self.bank()?;
        let shape = memory_bank.memory.shape();
        let (batch, n, m) = (shape[0], shape[1], shape[2]);

        let memory_data = memory_bank.memory.to_vec_f32()?;
        let key_data = key.to_vec_f32()?;
        if key_data.len() != batch * m {
            return Err(TrustformersError::shape_error(format!(
                "content key must be [{}, {}], got {:?}",
                batch,
                m,
                key.shape()
            )));
        }

        let mut weights = Vec::with_capacity(batch * n);
        for b in 0..batch {
            weights.extend(content_addressing(
                &key_data[b * m..(b + 1) * m],
                strength,
                &memory_data[b * n * m..(b + 1) * n * m],
                n,
                m,
            ));
        }

        Tensor::from_vec(weights, &[batch, n])
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

        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states)?;
            if let Some(memory) = layer.get_memory() {
                all_memory_states.push(memory.clone());
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
