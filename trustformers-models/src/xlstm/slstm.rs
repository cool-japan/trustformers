//! sLSTM: the scalar-memory xLSTM block.
//!
//! sLSTM keeps the classic LSTM scalar cell state but replaces the sigmoid input
//! gate with an exponential one and adds a normaliser state `n_t` plus the
//! stabilizer `m_t`. It retains the recurrent (memory-mixing) connections `R`, so
//! it is a genuine RNN and is evaluated one timestep at a time.
//!
//! Per timestep, with `x_t ∈ R^d` and `h_{t-1} ∈ R^d`:
//!
//! ```text
//! z̃_t = W_z x_t + R_z h_{t-1} + b_z      z_t = tanh(z̃_t)      cell input
//! ĩ_t = W_i x_t + R_i h_{t-1} + b_i       i_t = exp(ĩ_t)        input gate
//! f̃_t = W_f x_t + R_f h_{t-1} + b_f       f_t = σ(f̃_t)          forget gate
//! õ_t = W_o x_t + R_o h_{t-1} + b_o       o_t = σ(õ_t)          output gate
//!
//! m_t = max(log f_t + m_{t-1}, ĩ_t)      (stabilizer)
//! c_t = f'_t c_{t-1} + i'_t z_t          (cell state)
//! n_t = f'_t n_{t-1} + i'_t              (normaliser)
//! h_t = o_t ⊙ c_t / n_t
//! ```
//!
//! Reference: Beck et al., "xLSTM: Extended Long Short-Term Memory" (2024).

use crate::xlstm::gating::{as_sequence, log_sigmoid, matvec, sigmoid, stabilized_gates};
use trustformers_core::{
    device::Device,
    errors::{invalid_input, Result},
    layers::Linear,
    tensor::Tensor,
    traits::Layer,
};

/// Recurrent state of an [`SLstmBlock`].
///
/// `c` is the cell state, `n` the normaliser, `m` the stabilizer and `h` the last
/// emitted hidden state (fed back through the recurrent matrices). All four start
/// at zero, which makes the first step reduce to `h_1 = o_1 ⊙ z_1`.
#[derive(Debug, Clone)]
pub struct SLstmState {
    /// Width of the state vectors.
    pub hidden_size: usize,
    /// Cell state `c_t`.
    pub cell: Vec<f32>,
    /// Normaliser state `n_t`.
    pub normalizer: Vec<f32>,
    /// Stabilizer state `m_t`.
    pub stabilizer: Vec<f32>,
    /// Previous hidden state `h_{t-1}`.
    pub hidden: Vec<f32>,
}

impl SLstmState {
    /// Create a zero-initialised state for a block of width `hidden_size`.
    pub fn new(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            cell: vec![0.0; hidden_size],
            normalizer: vec![0.0; hidden_size],
            stabilizer: vec![0.0; hidden_size],
            hidden: vec![0.0; hidden_size],
        }
    }

    /// Reset every component back to zero.
    pub fn reset(&mut self) {
        self.cell.iter_mut().for_each(|v| *v = 0.0);
        self.normalizer.iter_mut().for_each(|v| *v = 0.0);
        self.stabilizer.iter_mut().for_each(|v| *v = 0.0);
        self.hidden.iter_mut().for_each(|v| *v = 0.0);
    }
}

/// sLSTM block: scalar memory with exponential gating and memory mixing.
#[derive(Debug, Clone)]
pub struct SLstmBlock {
    hidden_size: usize,
    /// `W_z` and `b_z` — cell input.
    cell_input: Linear,
    /// `W_i` and `b_i` — exponential input gate.
    input_gate: Linear,
    /// `W_f` and `b_f` — forget gate.
    forget_gate: Linear,
    /// `W_o` and `b_o` — output gate.
    output_gate: Linear,
    /// `R_z` — recurrent cell input (memory mixing).
    recurrent_cell: Linear,
    /// `R_i` — recurrent input gate.
    recurrent_input: Linear,
    /// `R_f` — recurrent forget gate.
    recurrent_forget: Linear,
    /// `R_o` — recurrent output gate.
    recurrent_output: Linear,
    device: Device,
}

/// The recurrent matrices, extracted once per forward pass.
struct RecurrentWeights {
    cell: Vec<f32>,
    input: Vec<f32>,
    forget: Vec<f32>,
    output: Vec<f32>,
}

impl SLstmBlock {
    /// Create a new sLSTM block on CPU (backward compatibility)
    pub fn new(hidden_size: usize) -> Self {
        Self::new_with_device(hidden_size, Device::CPU)
    }

    /// Create a new sLSTM block with specified device
    pub fn new_with_device(hidden_size: usize, device: Device) -> Self {
        Self {
            hidden_size,
            cell_input: Linear::new_with_device(hidden_size, hidden_size, true, device),
            input_gate: Linear::new_with_device(hidden_size, hidden_size, true, device),
            forget_gate: Linear::new_with_device(hidden_size, hidden_size, true, device),
            output_gate: Linear::new_with_device(hidden_size, hidden_size, true, device),
            recurrent_cell: Linear::new_with_device(hidden_size, hidden_size, false, device),
            recurrent_input: Linear::new_with_device(hidden_size, hidden_size, false, device),
            recurrent_forget: Linear::new_with_device(hidden_size, hidden_size, false, device),
            recurrent_output: Linear::new_with_device(hidden_size, hidden_size, false, device),
            device,
        }
    }

    /// Bias the forget gate towards remembering.
    ///
    /// A large positive forget-gate bias is the standard LSTM initialisation trick
    /// (`initial_forget_gate_bias` in [`crate::xlstm::XLSTMConfig`]): it keeps the
    /// cell state alive for many steps before training shortens it.
    pub fn with_forget_gate_bias(mut self, bias: f32) -> Result<Self> {
        let bias_tensor = Tensor::from_vec(vec![bias; self.hidden_size], &[self.hidden_size])?;
        self.forget_gate.set_bias(bias_tensor)?;
        Ok(self)
    }

    /// Get the device this block is on
    pub fn device(&self) -> Device {
        self.device
    }

    /// Width of the block's input, output and state vectors.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Number of learned parameters: four gates with `W` (d×d) and `b` (d), plus
    /// four recurrent matrices `R` (d×d).
    pub fn parameter_count(&self) -> usize {
        self.cell_input.parameter_count()
            + self.input_gate.parameter_count()
            + self.forget_gate.parameter_count()
            + self.output_gate.parameter_count()
            + self.recurrent_cell.parameter_count()
            + self.recurrent_input.parameter_count()
            + self.recurrent_forget.parameter_count()
            + self.recurrent_output.parameter_count()
    }

    fn recurrent_weights(&self) -> Result<RecurrentWeights> {
        Ok(RecurrentWeights {
            cell: self.recurrent_cell.weight().data()?,
            input: self.recurrent_input.weight().data()?,
            forget: self.recurrent_forget.weight().data()?,
            output: self.recurrent_output.weight().data()?,
        })
    }

    /// Advance the recurrence by one timestep.
    ///
    /// `projected_*` are the already-computed input contributions `W· x_t + b` for
    /// the four gates at this timestep; `recurrent` supplies the `R· h_{t-1}` term.
    /// Returns the new hidden state `h_t` and updates `state` in place.
    fn step(
        &self,
        projected: &StepProjections<'_>,
        recurrent: &RecurrentWeights,
        state: &mut SLstmState,
    ) -> Vec<f32> {
        let d = self.hidden_size;
        let r_cell = matvec(&recurrent.cell, &state.hidden, d, d);
        let r_input = matvec(&recurrent.input, &state.hidden, d, d);
        let r_forget = matvec(&recurrent.forget, &state.hidden, d, d);
        let r_output = matvec(&recurrent.output, &state.hidden, d, d);

        let mut hidden = vec![0.0f32; d];
        for j in 0..d {
            let cell_input = (projected.cell[j] + r_cell[j]).tanh();
            let input_raw = projected.input[j] + r_input[j];
            let forget_raw = projected.forget[j] + r_forget[j];
            let output_gate = sigmoid(projected.output[j] + r_output[j]);

            // Exponential input gate + sigmoid forget gate, carried in log space.
            let (m, input_gate, forget_gate) =
                stabilized_gates(input_raw, log_sigmoid(forget_raw), state.stabilizer[j]);

            let cell = forget_gate * state.cell[j] + input_gate * cell_input;
            let normalizer = forget_gate * state.normalizer[j] + input_gate;

            state.stabilizer[j] = m;
            state.cell[j] = cell;
            state.normalizer[j] = normalizer;

            // `n_t` is a sum of strictly positive stabilised gates, so it is > 0
            // from the first step onwards; the guard only covers a degenerate
            // all-zero-gate configuration.
            hidden[j] = if normalizer > 0.0 { output_gate * cell / normalizer } else { 0.0 };
        }

        state.hidden.copy_from_slice(&hidden);
        hidden
    }

    /// Run the block over a whole sequence, starting from a zero state.
    ///
    /// Accepts `[seq, hidden]` or `[batch, seq, hidden]` and returns the same shape.
    pub fn forward_sequence(&self, input: &Tensor) -> Result<Tensor> {
        if self.hidden_size == 0 {
            return Err(invalid_input("sLSTM hidden_size must be greater than zero"));
        }

        let (_, batch, seq) = as_sequence(input, self.hidden_size)?;

        // Input contributions for every timestep at once: this is the part of the
        // recurrence that does not depend on h_{t-1}.
        let cell_proj = self.cell_input.forward(input.clone())?.data()?;
        let input_proj = self.input_gate.forward(input.clone())?.data()?;
        let forget_proj = self.forget_gate.forward(input.clone())?.data()?;
        let output_proj = self.output_gate.forward(input.clone())?.data()?;

        let recurrent = self.recurrent_weights()?;
        let d = self.hidden_size;
        let mut out = vec![0.0f32; batch * seq * d];

        for b in 0..batch {
            let mut state = SLstmState::new(d);
            for t in 0..seq {
                let base = (b * seq + t) * d;
                let projections = StepProjections {
                    cell: &cell_proj[base..base + d],
                    input: &input_proj[base..base + d],
                    forget: &forget_proj[base..base + d],
                    output: &output_proj[base..base + d],
                };
                let hidden = self.step(&projections, &recurrent, &mut state);
                out[base..base + d].copy_from_slice(&hidden);
            }
        }

        Tensor::from_vec(out, &input.shape())
    }

    /// Advance the recurrence by one timestep for streaming inference.
    ///
    /// `x` is a single `[hidden_size]` input vector; `state` is updated in place
    /// and the emitted hidden state is returned.
    pub fn step_vector(&self, x: &[f32], state: &mut SLstmState) -> Result<Vec<f32>> {
        if x.len() != self.hidden_size || state.hidden_size != self.hidden_size {
            return Err(invalid_input(format!(
                "sLSTM step expects a {}-wide input and state, got input {} / state {}",
                self.hidden_size,
                x.len(),
                state.hidden_size
            )));
        }

        let single = Tensor::from_vec(x.to_vec(), &[1, self.hidden_size])?;
        let cell_proj = self.cell_input.forward(single.clone())?.data()?;
        let input_proj = self.input_gate.forward(single.clone())?.data()?;
        let forget_proj = self.forget_gate.forward(single.clone())?.data()?;
        let output_proj = self.output_gate.forward(single)?.data()?;

        let recurrent = self.recurrent_weights()?;
        let projections = StepProjections {
            cell: &cell_proj,
            input: &input_proj,
            forget: &forget_proj,
            output: &output_proj,
        };
        Ok(self.step(&projections, &recurrent, state))
    }
}

impl Layer for SLstmBlock {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_sequence(&input)
    }
}

/// The four per-timestep input projections `W·x_t + b`.
struct StepProjections<'a> {
    cell: &'a [f32],
    input: &'a [f32],
    forget: &'a [f32],
    output: &'a [f32],
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_input(seq: usize, hidden: usize) -> Tensor {
        Tensor::from_vec(
            (0..seq * hidden).map(|i| (i as f32 * 0.17).sin()).collect(),
            &[seq, hidden],
        )
        .expect("input tensor")
    }

    /// The block owns real weights.
    #[test]
    fn test_parameter_count_matches_the_declared_matrices() {
        let hidden = 6usize;
        let block = SLstmBlock::new(hidden);
        // 4 gates × (W: d² + b: d) + 4 recurrent matrices × d²
        let expected = 4 * (hidden * hidden + hidden) + 4 * (hidden * hidden);
        assert_eq!(block.parameter_count(), expected);
        assert!(block.parameter_count() > 0);
    }

    /// The forward pass must produce a real, finite, input-dependent signal.
    #[test]
    fn test_forward_is_nonzero_and_input_dependent() {
        let hidden = 4usize;
        let block = SLstmBlock::new(hidden);

        let a = block.forward_sequence(&ramp_input(5, hidden)).expect("forward a");
        let b_input = Tensor::from_vec(
            (0..5 * hidden).map(|i| (i as f32 * 0.31).cos()).collect(),
            &[5, hidden],
        )
        .expect("input b");
        let b = block.forward_sequence(&b_input).expect("forward b");

        let a_data = a.data().expect("a data");
        let b_data = b.data().expect("b data");
        assert_eq!(a.shape(), vec![5, hidden]);
        assert!(a_data.iter().all(|v| v.is_finite()));
        assert!(
            a_data.iter().any(|v| v.abs() > 1e-6),
            "sLSTM returned zeros"
        );
        assert!(a_data.iter().zip(b_data.iter()).any(|(x, y)| (x - y).abs() > 1e-6));
    }

    /// Hand-computed single step from the zero state.
    ///
    /// With `c_0 = n_0 = m_0 = h_0 = 0` the recurrence collapses to
    /// `m_1 = max(log σ(f̃), ĩ)`, `c_1 = i' z`, `n_1 = i'`, so `h_1 = o · z`
    /// exactly — independent of the input gate.
    #[test]
    fn test_single_step_from_zero_state_equals_output_gate_times_cell_input() {
        let hidden = 3usize;
        let mut block = SLstmBlock::new(hidden);

        // Pin the weights so the step is fully determined.
        // W_z = I, W_i = 0, W_f = 0, W_o = 0 (so õ = b_o), R_* = 0.
        let identity: Vec<f32> = (0..hidden * hidden)
            .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
            .collect();
        let zeros = vec![0.0f32; hidden * hidden];

        block
            .cell_input
            .set_weight(Tensor::from_vec(identity, &[hidden, hidden]).expect("w"))
            .expect("set w_z");
        for linear in [
            &mut block.input_gate,
            &mut block.forget_gate,
            &mut block.output_gate,
        ] {
            linear
                .set_weight(Tensor::from_vec(zeros.clone(), &[hidden, hidden]).expect("w"))
                .expect("set gate weight");
        }
        for linear in [
            &mut block.recurrent_cell,
            &mut block.recurrent_input,
            &mut block.recurrent_forget,
            &mut block.recurrent_output,
        ] {
            linear
                .set_weight(Tensor::from_vec(zeros.clone(), &[hidden, hidden]).expect("w"))
                .expect("set recurrent weight");
        }
        // b_o = 0.5 for every channel, all other biases stay zero.
        block
            .output_gate
            .set_bias(Tensor::from_vec(vec![0.5f32; hidden], &[hidden]).expect("b"))
            .expect("set b_o");

        let x = [0.4f32, -1.2, 2.0];
        let input = Tensor::from_vec(x.to_vec(), &[1, hidden]).expect("input");
        let out = block.forward_sequence(&input).expect("forward").data().expect("data");

        let expected_gate = 1.0f64 / (1.0 + (-0.5f64).exp());
        for (j, value) in out.iter().enumerate() {
            let expected = expected_gate * (x[j] as f64).tanh();
            assert!(
                (*value as f64 - expected).abs() < 1e-5,
                "channel {j}: got {value}, expected {expected}"
            );
        }
    }

    /// A second hand-computed step, this time exercising the exponential input
    /// gate and the sigmoid forget gate against a scalar reference.
    #[test]
    fn test_two_step_recurrence_matches_scalar_reference() {
        let hidden = 1usize;
        let mut block = SLstmBlock::new(hidden);

        // W_z = 1, W_i = 1, W_f = 1, W_o = 0 with b_o = 0; all R = 0.
        let one = Tensor::from_vec(vec![1.0f32], &[1, 1]).expect("w");
        let zero = Tensor::from_vec(vec![0.0f32], &[1, 1]).expect("w");
        block.cell_input.set_weight(one.clone()).expect("w_z");
        block.input_gate.set_weight(one.clone()).expect("w_i");
        block.forget_gate.set_weight(one).expect("w_f");
        block.output_gate.set_weight(zero.clone()).expect("w_o");
        for linear in [
            &mut block.recurrent_cell,
            &mut block.recurrent_input,
            &mut block.recurrent_forget,
            &mut block.recurrent_output,
        ] {
            linear.set_weight(zero.clone()).expect("recurrent");
        }

        let x = [0.7f32, -0.4f32];
        let input = Tensor::from_vec(x.to_vec(), &[2, 1]).expect("input");
        let out = block.forward_sequence(&input).expect("forward").data().expect("data");

        // Scalar reference in f64, straight from the paper's equations.
        let sigmoid64 = |v: f64| 1.0 / (1.0 + (-v).exp());
        let mut c = 0.0f64;
        let mut n = 0.0f64;
        let mut m = 0.0f64;
        let mut expected = [0.0f64; 2];
        for (t, &xt) in x.iter().enumerate() {
            let xt = xt as f64;
            let z = xt.tanh();
            let i_raw = xt;
            let log_f = sigmoid64(xt).ln();
            let m_new = (log_f + m).max(i_raw);
            let i_stab = (i_raw - m_new).exp();
            let f_stab = (log_f + m - m_new).exp();
            c = f_stab * c + i_stab * z;
            n = f_stab * n + i_stab;
            m = m_new;
            expected[t] = sigmoid64(0.0) * c / n;
        }

        for (t, value) in out.iter().enumerate() {
            assert!(
                (*value as f64 - expected[t]).abs() < 1e-5,
                "t={t}: got {value}, reference {}",
                expected[t]
            );
        }
    }

    /// Streaming `step_vector` must reproduce the batched sequence pass exactly.
    #[test]
    fn test_streaming_steps_match_the_sequence_pass() {
        let hidden = 4usize;
        let block = SLstmBlock::new(hidden);
        let seq = 6usize;
        let input = ramp_input(seq, hidden);
        let batched = block.forward_sequence(&input).expect("forward").data().expect("data");

        let data = input.data().expect("input data");
        let mut state = SLstmState::new(hidden);
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

    /// The recurrence must be causal: a later input cannot change an earlier output.
    #[test]
    fn test_recurrence_is_causal() {
        let hidden = 3usize;
        let block = SLstmBlock::new(hidden);
        let seq = 5usize;

        let base = ramp_input(seq, hidden);
        let mut perturbed_data = base.data().expect("data");
        for value in perturbed_data.iter_mut().skip((seq - 1) * hidden) {
            *value += 4.0;
        }
        let perturbed = Tensor::from_vec(perturbed_data, &[seq, hidden]).expect("input");

        let a = block.forward_sequence(&base).expect("a").data().expect("data");
        let b = block.forward_sequence(&perturbed).expect("b").data().expect("data");

        for i in 0..(seq - 1) * hidden {
            assert!((a[i] - b[i]).abs() < 1e-6, "index {i} is not causal");
        }
        assert!(a[(seq - 1) * hidden..]
            .iter()
            .zip(b[(seq - 1) * hidden..].iter())
            .any(|(x, y)| (x - y).abs() > 1e-6));
    }

    /// Exponential gating must not overflow even for huge activations.
    #[test]
    fn test_large_inputs_stay_finite() {
        let hidden = 3usize;
        let block = SLstmBlock::new(hidden);
        let input = Tensor::from_vec(
            vec![120.0f32, -95.0, 200.0, -300.0, 80.0, 40.0],
            &[2, hidden],
        )
        .expect("input");
        let out = block.forward_sequence(&input).expect("forward").data().expect("data");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "exponential gating overflowed: {out:?}"
        );
    }

    /// A forget-gate bias shifts the forget gate towards 1.
    #[test]
    fn test_forget_gate_bias_is_applied() {
        let hidden = 2usize;
        let block = SLstmBlock::new(hidden).with_forget_gate_bias(5.0).expect("bias");
        let bias = block.forget_gate.bias().expect("bias present").data().expect("data");
        assert!(bias.iter().all(|v| (*v - 5.0).abs() < 1e-6), "{bias:?}");
    }

    /// State bookkeeping.
    #[test]
    fn test_state_starts_and_resets_to_zero() {
        let mut state = SLstmState::new(4);
        assert_eq!(state.hidden_size, 4);
        assert!(state.cell.iter().all(|v| *v == 0.0));

        state.cell[0] = 3.0;
        state.stabilizer[1] = -2.0;
        state.reset();
        assert!(state.cell.iter().all(|v| *v == 0.0));
        assert!(state.stabilizer.iter().all(|v| *v == 0.0));
    }

    /// A malformed streaming call is reported rather than silently truncated.
    #[test]
    fn test_step_vector_rejects_mismatched_width() {
        let block = SLstmBlock::new(4);
        let mut state = SLstmState::new(4);
        assert!(block.step_vector(&[0.0, 1.0], &mut state).is_err());
    }
}
