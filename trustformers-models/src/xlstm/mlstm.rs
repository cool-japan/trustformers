//! mLSTM: the matrix-memory xLSTM block.
//!
//! mLSTM drops the recurrent connections entirely — every gate depends only on
//! `x_t` — and promotes the scalar cell state to a `d × d` matrix memory updated
//! with an outer product. That makes it a covariance-style associative memory with
//! a retrieval step that looks like single-query attention, while still running in
//! `O(seq)` time and `O(d²)` memory.
//!
//! Per head, per timestep, with `d = head_dim`:
//!
//! ```text
//! q_t = W_q x_t + b_q
//! k_t = (W_k x_t + b_k) / sqrt(d)
//! v_t = W_v x_t + b_v
//! ĩ_t = w_i · x_t + b_i           (scalar)      f̃_t = w_f · x_t + b_f   (scalar)
//! o_t = σ(W_o x_t + b_o)
//!
//! m_t = max(log σ(f̃_t) + m_{t-1}, ĩ_t)          (stabilizer)
//! C_t = f'_t C_{t-1} + i'_t v_t k_tᵀ            (matrix memory)
//! n_t = f'_t n_{t-1} + i'_t k_t                 (normaliser)
//! h_t = o_t ⊙ (C_t q_t) / max(|n_tᵀ q_t|, 1)
//! ```
//!
//! Reference: Beck et al., "xLSTM: Extended Long Short-Term Memory" (2024).

use crate::xlstm::gating::{as_sequence, log_sigmoid, sigmoid, stabilized_gates};
use trustformers_core::{
    device::Device,
    errors::{invalid_input, Result},
    layers::Linear,
    tensor::Tensor,
    traits::Layer,
};

/// Recurrent state of an [`MLstmBlock`].
///
/// The matrix memory is stored head-major: head `h` occupies
/// `memory[h · head_dim² .. (h+1) · head_dim²]` in row-major `[head_dim, head_dim]`
/// order, matching `C[i][j] = v_i k_j`.
#[derive(Debug, Clone)]
pub struct MLstmState {
    /// Total width of the block.
    pub hidden_size: usize,
    /// Number of heads sharing the block width.
    pub num_heads: usize,
    /// Matrix memory `C_t`, `num_heads · head_dim²` entries.
    pub memory: Vec<f32>,
    /// Normaliser `n_t`, `num_heads · head_dim` entries.
    pub normalizer: Vec<f32>,
    /// Per-head stabilizer `m_t`.
    pub stabilizer: Vec<f32>,
}

impl MLstmState {
    /// Create a zero-initialised state.
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        let heads = num_heads.max(1);
        let head_dim = hidden_size / heads;
        Self {
            hidden_size,
            num_heads: heads,
            memory: vec![0.0; heads * head_dim * head_dim],
            normalizer: vec![0.0; heads * head_dim],
            stabilizer: vec![0.0; heads],
        }
    }

    /// Reset every component back to zero.
    pub fn reset(&mut self) {
        self.memory.iter_mut().for_each(|v| *v = 0.0);
        self.normalizer.iter_mut().for_each(|v| *v = 0.0);
        self.stabilizer.iter_mut().for_each(|v| *v = 0.0);
    }
}

/// mLSTM block: matrix memory with exponential gating.
#[derive(Debug, Clone)]
pub struct MLstmBlock {
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    /// `W_q`, `b_q`.
    query: Linear,
    /// `W_k`, `b_k`.
    key: Linear,
    /// `W_v`, `b_v`.
    value: Linear,
    /// `W_o`, `b_o` — the element-wise output gate.
    output_gate: Linear,
    /// `w_i`, `b_i` — one scalar input-gate pre-activation per head.
    input_gate: Linear,
    /// `w_f`, `b_f` — one scalar forget-gate pre-activation per head.
    forget_gate: Linear,
    device: Device,
}

impl MLstmBlock {
    /// Create a new mLSTM block on CPU (backward compatibility)
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        Self::new_with_device(hidden_size, num_heads, Device::CPU)
    }

    /// Create a new mLSTM block with specified device.
    ///
    /// `num_heads` is clamped to at least one; when it does not divide
    /// `hidden_size` the remainder columns are left outside the heads, which
    /// [`MLstmBlock::forward_sequence`] reports as an error rather than silently
    /// dropping.
    pub fn new_with_device(hidden_size: usize, num_heads: usize, device: Device) -> Self {
        let heads = num_heads.max(1);
        let head_dim = hidden_size / heads;
        Self {
            hidden_size,
            num_heads: heads,
            head_dim,
            query: Linear::new_with_device(hidden_size, hidden_size, true, device),
            key: Linear::new_with_device(hidden_size, hidden_size, true, device),
            value: Linear::new_with_device(hidden_size, hidden_size, true, device),
            output_gate: Linear::new_with_device(hidden_size, hidden_size, true, device),
            input_gate: Linear::new_with_device(hidden_size, heads, true, device),
            forget_gate: Linear::new_with_device(hidden_size, heads, true, device),
            device,
        }
    }

    /// Bias the forget gate towards remembering (see [`crate::xlstm::SLstmBlock`]).
    pub fn with_forget_gate_bias(mut self, bias: f32) -> Result<Self> {
        let bias_tensor = Tensor::from_vec(vec![bias; self.num_heads], &[self.num_heads])?;
        self.forget_gate.set_bias(bias_tensor)?;
        Ok(self)
    }

    /// Get the device this block is on
    pub fn device(&self) -> Device {
        self.device
    }

    /// Width of the block's input and output.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Number of heads, each with its own matrix memory.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Width of a single head's memory (`hidden_size / num_heads`).
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Number of learned parameters: four `d × d` projections with biases plus the
    /// two per-head scalar gate projections.
    pub fn parameter_count(&self) -> usize {
        self.query.parameter_count()
            + self.key.parameter_count()
            + self.value.parameter_count()
            + self.output_gate.parameter_count()
            + self.input_gate.parameter_count()
            + self.forget_gate.parameter_count()
    }

    /// Advance one head by one timestep, returning that head's hidden slice.
    fn step_head(
        &self,
        head: usize,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        output_gate: &[f32],
        input_raw: f32,
        forget_raw: f32,
        state: &mut MLstmState,
    ) -> Vec<f32> {
        let d = self.head_dim;
        let (m, input_gate, forget_gate) =
            stabilized_gates(input_raw, log_sigmoid(forget_raw), state.stabilizer[head]);
        state.stabilizer[head] = m;

        let memory_base = head * d * d;
        let normalizer_base = head * d;

        // C_t = f' C_{t-1} + i' v kᵀ  and  n_t = f' n_{t-1} + i' k
        for i in 0..d {
            let row = memory_base + i * d;
            let vi = input_gate * value[i];
            for j in 0..d {
                state.memory[row + j] = forget_gate * state.memory[row + j] + vi * key[j];
            }
        }
        for j in 0..d {
            state.normalizer[normalizer_base + j] =
                forget_gate * state.normalizer[normalizer_base + j] + input_gate * key[j];
        }

        // Retrieval: h = o ⊙ (C q) / max(|nᵀ q|, 1)
        let mut denominator = 0.0f32;
        for j in 0..d {
            denominator += state.normalizer[normalizer_base + j] * query[j];
        }
        let denominator = denominator.abs().max(1.0);

        let mut hidden = vec![0.0f32; d];
        for (i, slot) in hidden.iter_mut().enumerate() {
            let row = memory_base + i * d;
            let mut acc = 0.0f32;
            for j in 0..d {
                acc += state.memory[row + j] * query[j];
            }
            *slot = output_gate[i] * acc / denominator;
        }

        hidden
    }

    /// Run the block over a whole sequence, starting from a zero state.
    ///
    /// Accepts `[seq, hidden]` or `[batch, seq, hidden]` and returns the same shape.
    pub fn forward_sequence(&self, input: &Tensor) -> Result<Tensor> {
        if self.hidden_size == 0 || self.head_dim == 0 {
            return Err(invalid_input(
                "mLSTM hidden_size must be greater than zero and divisible by num_heads",
            ));
        }
        if self.num_heads * self.head_dim != self.hidden_size {
            return Err(invalid_input(format!(
                "mLSTM hidden_size {} is not divisible by num_heads {}",
                self.hidden_size, self.num_heads
            )));
        }

        let (_, batch, seq) = as_sequence(input, self.hidden_size)?;

        // Every mLSTM gate depends on x_t alone, so the whole sequence can be
        // projected in one pass; only the memory update is sequential.
        let queries = self.query.forward(input.clone())?.data()?;
        let keys = self.key.forward(input.clone())?.data()?;
        let values = self.value.forward(input.clone())?.data()?;
        let output_gates = self.output_gate.forward(input.clone())?.data()?;
        let input_raw = self.input_gate.forward(input.clone())?.data()?;
        let forget_raw = self.forget_gate.forward(input.clone())?.data()?;

        let d = self.head_dim;
        let key_scale = 1.0 / (d as f32).sqrt();
        let mut out = vec![0.0f32; batch * seq * self.hidden_size];

        for b in 0..batch {
            let mut state = MLstmState::new(self.hidden_size, self.num_heads);
            for t in 0..seq {
                let base = (b * seq + t) * self.hidden_size;
                let gate_base = (b * seq + t) * self.num_heads;
                for head in 0..self.num_heads {
                    let head_base = base + head * d;
                    let scaled_key: Vec<f32> =
                        keys[head_base..head_base + d].iter().map(|k| k * key_scale).collect();
                    let gated: Vec<f32> = output_gates[head_base..head_base + d]
                        .iter()
                        .map(|v| sigmoid(*v))
                        .collect();

                    let hidden = self.step_head(
                        head,
                        &queries[head_base..head_base + d],
                        &scaled_key,
                        &values[head_base..head_base + d],
                        &gated,
                        input_raw[gate_base + head],
                        forget_raw[gate_base + head],
                        &mut state,
                    );
                    out[head_base..head_base + d].copy_from_slice(&hidden);
                }
            }
        }

        Tensor::from_vec(out, &input.shape())
    }

    /// Advance the recurrence by one timestep for streaming inference.
    pub fn step_vector(&self, x: &[f32], state: &mut MLstmState) -> Result<Vec<f32>> {
        if x.len() != self.hidden_size || state.hidden_size != self.hidden_size {
            return Err(invalid_input(format!(
                "mLSTM step expects a {}-wide input and state, got input {} / state {}",
                self.hidden_size,
                x.len(),
                state.hidden_size
            )));
        }
        if state.num_heads != self.num_heads {
            return Err(invalid_input(format!(
                "mLSTM state has {} heads but the block has {}",
                state.num_heads, self.num_heads
            )));
        }

        let single = Tensor::from_vec(x.to_vec(), &[1, self.hidden_size])?;
        let queries = self.query.forward(single.clone())?.data()?;
        let keys = self.key.forward(single.clone())?.data()?;
        let values = self.value.forward(single.clone())?.data()?;
        let output_gates = self.output_gate.forward(single.clone())?.data()?;
        let input_raw = self.input_gate.forward(single.clone())?.data()?;
        let forget_raw = self.forget_gate.forward(single)?.data()?;

        let d = self.head_dim;
        let key_scale = 1.0 / (d as f32).sqrt();
        let mut out = vec![0.0f32; self.hidden_size];

        for head in 0..self.num_heads {
            let head_base = head * d;
            let scaled_key: Vec<f32> =
                keys[head_base..head_base + d].iter().map(|k| k * key_scale).collect();
            let gated: Vec<f32> =
                output_gates[head_base..head_base + d].iter().map(|v| sigmoid(*v)).collect();
            let hidden = self.step_head(
                head,
                &queries[head_base..head_base + d],
                &scaled_key,
                &values[head_base..head_base + d],
                &gated,
                input_raw[head],
                forget_raw[head],
                state,
            );
            out[head_base..head_base + d].copy_from_slice(&hidden);
        }

        Ok(out)
    }
}

impl Layer for MLstmBlock {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_sequence(&input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wave_input(seq: usize, hidden: usize) -> Tensor {
        Tensor::from_vec(
            (0..seq * hidden).map(|i| (i as f32 * 0.23).sin()).collect(),
            &[seq, hidden],
        )
        .expect("input tensor")
    }

    /// Real weights, counted from the real matrices.
    #[test]
    fn test_parameter_count_matches_the_declared_matrices() {
        let hidden = 8usize;
        let heads = 2usize;
        let block = MLstmBlock::new(hidden, heads);
        // 4 × (d² + d) for q/k/v/o, plus 2 × (d·heads + heads) for the scalar gates.
        let expected = 4 * (hidden * hidden + hidden) + 2 * (hidden * heads + heads);
        assert_eq!(block.parameter_count(), expected);
    }

    /// The forward pass must produce a real, finite, input-dependent signal.
    #[test]
    fn test_forward_is_nonzero_and_input_dependent() {
        let hidden = 8usize;
        let block = MLstmBlock::new(hidden, 2);

        let a = block.forward_sequence(&wave_input(5, hidden)).expect("a");
        let b_input = Tensor::from_vec(
            (0..5 * hidden).map(|i| (i as f32 * 0.41).cos()).collect(),
            &[5, hidden],
        )
        .expect("input b");
        let b = block.forward_sequence(&b_input).expect("b");

        let a_data = a.data().expect("a data");
        let b_data = b.data().expect("b data");
        assert_eq!(a.shape(), vec![5, hidden]);
        assert!(a_data.iter().all(|v| v.is_finite()));
        assert!(
            a_data.iter().any(|v| v.abs() > 1e-6),
            "mLSTM returned zeros"
        );
        assert!(a_data.iter().zip(b_data.iter()).any(|(x, y)| (x - y).abs() > 1e-6));
    }

    /// Hand-computed single step against the paper's equations.
    ///
    /// With `C_0 = 0`, `n_0 = 0`, `m_0 = 0`, one step gives `C_1 = i' v kᵀ`,
    /// `n_1 = i' k`, so `h = o ⊙ i' v (k·q) / max(|i' k·q|, 1)`.
    #[test]
    fn test_single_step_matches_hand_computation() {
        let hidden = 2usize;
        let mut block = MLstmBlock::new(hidden, 1);

        // q = x, k = x, v = x, o gate weight = 0 (so σ(0) = 0.5), gates = raw x sums.
        let identity = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], &[2, 2]).expect("w");
        let zeros2 = Tensor::from_vec(vec![0.0f32; 4], &[2, 2]).expect("w");
        block.query.set_weight(identity.clone()).expect("w_q");
        block.key.set_weight(identity.clone()).expect("w_k");
        block.value.set_weight(identity).expect("w_v");
        block.output_gate.set_weight(zeros2).expect("w_o");
        // Scalar gate projections: i = 0, f = 0 -> log σ(0) = -ln 2.
        block
            .input_gate
            .set_weight(Tensor::from_vec(vec![0.0f32, 0.0], &[1, 2]).expect("w"))
            .expect("w_i");
        block
            .forget_gate
            .set_weight(Tensor::from_vec(vec![0.0f32, 0.0], &[1, 2]).expect("w"))
            .expect("w_f");

        let x = [0.6f32, -0.8f32];
        let input = Tensor::from_vec(x.to_vec(), &[1, hidden]).expect("input");
        let out = block.forward_sequence(&input).expect("forward").data().expect("data");

        // Reference in f64.
        let d = hidden as f64;
        let q = [x[0] as f64, x[1] as f64];
        let k = [q[0] / d.sqrt(), q[1] / d.sqrt()];
        let v = q;
        let i_raw = 0.0f64;
        let log_f = (1.0f64 / 2.0).ln();
        let m = (log_f + 0.0f64).max(i_raw);
        let i_stab = (i_raw - m).exp();
        // C = i' v kᵀ ; n = i' k
        let dot_nq = i_stab * (k[0] * q[0] + k[1] * q[1]);
        let denominator = dot_nq.abs().max(1.0);
        let gate = 0.5f64; // σ(0)
        for (i, value) in out.iter().enumerate() {
            let cq = i_stab * v[i] * (k[0] * q[0] + k[1] * q[1]);
            let expected = gate * cq / denominator;
            assert!(
                (*value as f64 - expected).abs() < 1e-5,
                "channel {i}: got {value}, expected {expected}"
            );
        }
    }

    /// Streaming `step_vector` must reproduce the sequence pass exactly.
    #[test]
    fn test_streaming_steps_match_the_sequence_pass() {
        let hidden = 6usize;
        let heads = 3usize;
        let block = MLstmBlock::new(hidden, heads);
        let seq = 5usize;
        let input = wave_input(seq, hidden);
        let batched = block.forward_sequence(&input).expect("forward").data().expect("data");

        let data = input.data().expect("input data");
        let mut state = MLstmState::new(hidden, heads);
        for t in 0..seq {
            let step = block
                .step_vector(&data[t * hidden..(t + 1) * hidden], &mut state)
                .expect("step");
            for j in 0..hidden {
                assert!(
                    (step[j] - batched[t * hidden + j]).abs() < 1e-5,
                    "t={t} channel {j}: streaming {} vs batched {}",
                    step[j],
                    batched[t * hidden + j]
                );
            }
        }
    }

    /// The matrix memory must actually accumulate: a second timestep sees the first.
    #[test]
    fn test_matrix_memory_carries_information_forward() {
        let hidden = 4usize;
        let block = MLstmBlock::new(hidden, 1);
        let seq = 3usize;

        let base = wave_input(seq, hidden);
        let mut perturbed_data = base.data().expect("data");
        for value in perturbed_data.iter_mut().take(hidden) {
            *value += 2.0; // timestep 0 only
        }
        let perturbed = Tensor::from_vec(perturbed_data, &[seq, hidden]).expect("input");

        let a = block.forward_sequence(&base).expect("a").data().expect("data");
        let b = block.forward_sequence(&perturbed).expect("b").data().expect("data");

        let changed = a[(seq - 1) * hidden..]
            .iter()
            .zip(b[(seq - 1) * hidden..].iter())
            .any(|(x, y)| (x - y).abs() > 1e-6);
        assert!(
            changed,
            "the matrix memory does not carry the first timestep forward"
        );
    }

    /// The recurrence must be causal.
    #[test]
    fn test_recurrence_is_causal() {
        let hidden = 4usize;
        let block = MLstmBlock::new(hidden, 2);
        let seq = 5usize;

        let base = wave_input(seq, hidden);
        let mut perturbed_data = base.data().expect("data");
        for value in perturbed_data.iter_mut().skip((seq - 1) * hidden) {
            *value += 3.0;
        }
        let perturbed = Tensor::from_vec(perturbed_data, &[seq, hidden]).expect("input");

        let a = block.forward_sequence(&base).expect("a").data().expect("data");
        let b = block.forward_sequence(&perturbed).expect("b").data().expect("data");

        for i in 0..(seq - 1) * hidden {
            assert!((a[i] - b[i]).abs() < 1e-6, "index {i} is not causal");
        }
    }

    /// Heads must be independent: each carries its own matrix memory.
    #[test]
    fn test_heads_have_independent_memories() {
        let hidden = 4usize;
        let heads = 2usize;
        let block = MLstmBlock::new(hidden, heads);
        let mut state = MLstmState::new(hidden, heads);
        assert_eq!(
            state.memory.len(),
            heads * (hidden / heads) * (hidden / heads)
        );

        block.step_vector(&[1.0, 0.0, 0.0, 1.0], &mut state).expect("step");
        let head_dim = hidden / heads;
        let head0: f32 = state.memory[..head_dim * head_dim].iter().map(|v| v.abs()).sum();
        let head1: f32 = state.memory[head_dim * head_dim..].iter().map(|v| v.abs()).sum();
        assert!(head0 > 0.0 && head1 > 0.0, "a head memory stayed empty");
    }

    /// Exponential gating must not overflow.
    #[test]
    fn test_large_inputs_stay_finite() {
        let hidden = 4usize;
        let block = MLstmBlock::new(hidden, 2);
        let input = Tensor::from_vec(
            vec![150.0f32, -200.0, 90.0, 300.0, -120.0, 60.0, 45.0, -80.0],
            &[2, hidden],
        )
        .expect("input");
        let out = block.forward_sequence(&input).expect("forward").data().expect("data");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "exponential gating overflowed: {out:?}"
        );
    }

    /// An indivisible head count is reported instead of dropping channels.
    #[test]
    fn test_indivisible_head_count_is_rejected() {
        let block = MLstmBlock::new(7, 2);
        let input = Tensor::from_vec(vec![0.0; 7], &[1, 7]).expect("input");
        assert!(block.forward_sequence(&input).is_err());
    }

    /// State bookkeeping.
    #[test]
    fn test_state_reset() {
        let mut state = MLstmState::new(4, 2);
        state.memory[0] = 5.0;
        state.stabilizer[1] = -3.0;
        state.reset();
        assert!(state.memory.iter().all(|v| *v == 0.0));
        assert!(state.stabilizer.iter().all(|v| *v == 0.0));
    }
}
