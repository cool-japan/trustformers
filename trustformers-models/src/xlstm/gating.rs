//! Exponential gating primitives shared by the sLSTM and mLSTM blocks.
//!
//! The defining change xLSTM makes to the LSTM cell is replacing the sigmoid
//! input gate with an *exponential* one. `exp` is unbounded, so the cell state
//! must be carried together with a stabilizer state `m_t` that tracks the running
//! maximum of the gate exponents; every gate value the recurrence actually uses is
//! divided by `e^{m_t}` and therefore lies in `(0, 1]`.
//!
//! Reference: Beck et al., "xLSTM: Extended Long Short-Term Memory" (2024),
//! equations (15)-(23) for sLSTM and (24)-(32) for mLSTM.

use trustformers_core::{
    errors::{tensor_op_error, Result},
    tensor::Tensor,
};

/// Logistic sigmoid, evaluated so neither tail overflows.
pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `ln σ(x)`, computed without ever forming `σ(x)`.
///
/// `σ(x)` underflows to zero for `x ≪ 0`, which would make `ln σ(x)` `-inf` and
/// destroy the stabilizer. Using `ln σ(x) = -softplus(-x)` keeps full precision.
pub(crate) fn log_sigmoid(x: f32) -> f32 {
    -((-x).max(0.0) + (1.0 + (-x.abs()).exp()).ln())
}

/// One step of the xLSTM stabilizer.
///
/// Given the raw input-gate pre-activation `i_raw`, the log forget gate `log_f`
/// and the previous stabilizer state `m_prev`, returns `(m_t, i'_t, f'_t)`:
///
/// ```text
/// m_t  = max(log f_t + m_{t-1},  ĩ_t)
/// i'_t = exp(ĩ_t - m_t)                  ∈ (0, 1]
/// f'_t = exp(log f_t + m_{t-1} - m_t)    ∈ (0, 1]
/// ```
///
/// Both returned gates are bounded, so the recurrence `c_t = f'_t c_{t-1} + i'_t z_t`
/// can never overflow no matter how large the raw gate pre-activations grow.
pub(crate) fn stabilized_gates(i_raw: f32, log_f: f32, m_prev: f32) -> (f32, f32, f32) {
    let decayed = log_f + m_prev;
    let m = if decayed > i_raw { decayed } else { i_raw };
    let input_gate = (i_raw - m).exp();
    let forget_gate = (decayed - m).exp();
    (m, input_gate, forget_gate)
}

/// `out[o] = Σ_i weight[o · in_dim + i] · x[i]`.
///
/// Matches the `[out_features, in_features]` weight layout used by
/// [`trustformers_core::layers::Linear`], so a matrix extracted from a `Linear`
/// can be applied directly to a single timestep without rebuilding a tensor.
pub(crate) fn matvec(weight: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for (o, slot) in out.iter_mut().enumerate() {
        let row = &weight[o * in_dim..(o + 1) * in_dim];
        let mut acc = 0.0f32;
        for (w, v) in row.iter().zip(x.iter()) {
            acc += w * v;
        }
        *slot = acc;
    }
    out
}

/// Decompose a `[seq, hidden]` or `[batch, seq, hidden]` tensor.
///
/// Returns the flat row-major data together with `(batch, seq)`. The time axis
/// drives the recurrence, so any other rank is an error rather than a guess.
pub(crate) fn as_sequence(tensor: &Tensor, hidden_size: usize) -> Result<(Vec<f32>, usize, usize)> {
    let shape = tensor.shape();
    let (batch, seq) = match shape.len() {
        2 => (1usize, shape[0]),
        3 => (shape[0], shape[1]),
        _ => {
            return Err(tensor_op_error(
                "xlstm_block",
                format!("expected [seq, hidden] or [batch, seq, hidden] input, got {shape:?}"),
            ))
        },
    };

    let channels = shape[shape.len() - 1];
    if channels != hidden_size {
        return Err(tensor_op_error(
            "xlstm_block",
            format!("input width {channels} does not match hidden_size {hidden_size}"),
        ));
    }

    let data = tensor.data()?;
    if data.len() != batch * seq * channels {
        return Err(tensor_op_error(
            "xlstm_block",
            format!(
                "data length {} is inconsistent with shape {shape:?}",
                data.len()
            ),
        ));
    }

    Ok((data, batch, seq))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_matches_definition() {
        for x in [-30.0f32, -3.0, -0.5, 0.0, 0.5, 3.0, 30.0] {
            let expected = 1.0 / (1.0 + (-x as f64).exp());
            assert!(
                (sigmoid(x) as f64 - expected).abs() < 1e-6,
                "sigmoid({x}) = {} vs {expected}",
                sigmoid(x)
            );
        }
    }

    #[test]
    fn test_log_sigmoid_is_stable_in_the_left_tail() {
        // σ(-120) underflows to exactly zero in f32, so the naive ln(σ(x)) is -inf.
        let naive = sigmoid(-120.0).ln();
        assert!(
            naive.is_infinite(),
            "precondition: naive log-sigmoid underflows"
        );
        let stable = log_sigmoid(-120.0);
        assert!(stable.is_finite());
        assert!(
            (stable - (-120.0)).abs() < 1e-3,
            "ln σ(x) ≈ x for x ≪ 0, got {stable}"
        );
    }

    #[test]
    fn test_stabilized_gates_are_bounded() {
        // Huge input-gate pre-activation: the raw exp would overflow f32. After
        // stabilisation the input gate saturates at 1 and the forget gate is
        // driven to (an entirely correct) zero — the old state is negligible next
        // to a contribution e^200 larger.
        let (m, input_gate, forget_gate) = stabilized_gates(200.0, log_sigmoid(1.0), 0.0);
        assert!((m - 200.0).abs() < 1e-3);
        assert!((input_gate - 1.0).abs() < 1e-6);
        assert!(
            (0.0..=1.0).contains(&forget_gate),
            "forget gate {forget_gate} escaped [0, 1]"
        );
        assert!(input_gate.is_finite() && forget_gate.is_finite());

        // A moderate configuration keeps both gates strictly positive.
        let (_, input_gate, forget_gate) = stabilized_gates(0.5, log_sigmoid(0.25), 0.1);
        assert!(input_gate > 0.0 && input_gate <= 1.0);
        assert!(forget_gate > 0.0 && forget_gate <= 1.0);
    }

    #[test]
    fn test_stabilized_gates_preserve_the_gate_ratio() {
        // The stabilizer divides both gates by the same e^m, so their ratio is
        // exactly the unstabilised ratio e^{i} / (f · e^{m_prev}).
        let i_raw = 1.25f32;
        let log_f = log_sigmoid(0.4);
        let m_prev = 0.75f32;
        let (_, i_stab, f_stab) = stabilized_gates(i_raw, log_f, m_prev);
        let expected_ratio = (i_raw - (log_f + m_prev)).exp();
        assert!(
            ((i_stab / f_stab) - expected_ratio).abs() < 1e-4,
            "ratio {} vs {expected_ratio}",
            i_stab / f_stab
        );
    }

    #[test]
    fn test_matvec_matches_hand_computation() {
        // W = [[1, 2, 3], [4, 5, 6]], x = [1, 0, -1] -> [1-3, 4-6] = [-2, -2]
        let weight = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0f32, 0.0, -1.0];
        let out = matvec(&weight, &x, 2, 3);
        assert_eq!(out, vec![-2.0, -2.0]);
    }

    #[test]
    fn test_as_sequence_accepts_both_ranks() -> Result<()> {
        let two_d = Tensor::from_vec(vec![0.0; 6], &[3, 2])?;
        let (_, batch, seq) = as_sequence(&two_d, 2)?;
        assert_eq!((batch, seq), (1, 3));

        let three_d = Tensor::from_vec(vec![0.0; 12], &[2, 3, 2])?;
        let (_, batch, seq) = as_sequence(&three_d, 2)?;
        assert_eq!((batch, seq), (2, 3));

        let wrong_width = Tensor::from_vec(vec![0.0; 6], &[3, 2])?;
        assert!(as_sequence(&wrong_width, 4).is_err());

        let wrong_rank = Tensor::from_vec(vec![0.0; 6], &[6])?;
        assert!(as_sequence(&wrong_rank, 6).is_err());

        Ok(())
    }
}
