//! Variational quantum optimizer backed by a real statevector simulation.
//!
//! The forward pass is a genuine parameterised quantum circuit:
//!
//! 1. **Angle encoding.** Each `d_model`-wide feature vector is partitioned
//!    into `num_qubits` contiguous groups; the mean of group `q` is squashed
//!    with `tanh` and scaled by `π` to give the encoding angle of qubit `q`.
//! 2. **Variational circuit.** [`VariationalCircuit`] applies `RY`/`RZ`
//!    rotations parameterised by [`QuantumOptimizer::parameters`], followed by
//!    a CNOT entangling ring, repeated for the configured number of layers.
//! 3. **Measurement.** The Pauli-`Z` expectation value of each qubit is read
//!    out of the exact statevector and broadcast back over the feature group
//!    it encodes.
//!
//! Gradients use the parameter-shift rule (`±π/2`, denominator 2), which is
//! exact for these rotation generators, combined with the analytic chain rule
//! through the loss. That makes each parameter's gradient genuinely distinct,
//! unlike a finite difference of a function that is linear in the parameter
//! mean.

use trustformers_core::{
    errors::{Result, TrustformersError},
    tensor::Tensor,
};

use super::{config::QuantumClassicalConfig, statevector::VariationalCircuit};

/// Quantum optimizer for hybrid models
#[derive(Debug, Clone)]
pub struct QuantumOptimizer {
    /// Configuration
    pub config: QuantumClassicalConfig,
    /// Parameterised circuit executed by the forward pass
    pub circuit: VariationalCircuit,
    /// Circuit rotation angles
    pub parameters: Vec<f64>,
    /// Gradient history
    pub gradient_history: Vec<Vec<f64>>,
    /// Learning rate schedule
    pub learning_rate_schedule: Vec<f64>,
    /// Current iteration
    pub current_iteration: usize,
}

impl QuantumOptimizer {
    /// Create a new quantum optimizer.
    ///
    /// Returns an error when the configured qubit count exceeds what the dense
    /// simulator can represent.
    pub fn new(config: &QuantumClassicalConfig) -> Result<Self> {
        let circuit = VariationalCircuit::new(config.num_qubits, config.ansatz_layers())?;
        // A uniform non-zero initialisation avoids the all-|0> barren point
        // while keeping construction deterministic.
        let parameters = vec![0.1; circuit.parameter_count()];
        let learning_rate_schedule =
            vec![config.quantum_learning_rate; config.max_quantum_iterations.max(1)];

        Ok(Self {
            config: config.clone(),
            circuit,
            parameters,
            gradient_history: Vec::new(),
            learning_rate_schedule,
            current_iteration: 0,
        })
    }

    /// Split an input tensor into `[vector_count, d_model]` feature rows.
    fn feature_rows(&self, input: &Tensor) -> Result<(Vec<f32>, usize, Vec<usize>)> {
        let shape = input.shape();
        let d_model = *shape.last().ok_or_else(|| {
            TrustformersError::shape_error("quantum optimizer input has no dimensions".to_string())
        })?;
        if d_model == 0 {
            return Err(TrustformersError::shape_error(
                "quantum optimizer input has a zero-width feature dimension".to_string(),
            ));
        }
        if d_model < self.circuit.num_qubits() {
            return Err(TrustformersError::shape_error(format!(
                "feature width {} is smaller than the {} qubits it must encode",
                d_model,
                self.circuit.num_qubits()
            )));
        }
        Ok((input.to_vec_f32()?, d_model, shape))
    }

    /// Boundaries of the `num_qubits` contiguous feature groups.
    fn group_bounds(&self, d_model: usize) -> Vec<(usize, usize)> {
        let qubits = self.circuit.num_qubits();
        (0..qubits)
            .map(|q| {
                let start = q * d_model / qubits;
                let end = ((q + 1) * d_model / qubits).max(start + 1).min(d_model);
                (start, end)
            })
            .collect()
    }

    /// Angle-encode one feature row.
    fn encode(&self, row: &[f32], bounds: &[(usize, usize)]) -> Vec<f64> {
        bounds
            .iter()
            .map(|(start, end)| {
                let slice = &row[*start..*end];
                let mean = slice.iter().map(|x| *x as f64).sum::<f64>() / slice.len() as f64;
                std::f64::consts::PI * mean.tanh()
            })
            .collect()
    }

    /// Optimize the circuit parameters against `input` and return the output of
    /// the optimized circuit.
    pub fn optimize(&mut self, input: &Tensor) -> Result<Tensor> {
        let mut best_parameters = self.parameters.clone();
        let mut best_loss = self.compute_loss(input)?;

        for iteration in 0..self.config.max_quantum_iterations {
            let gradients = self.compute_gradients(input)?;
            self.update_parameters(&gradients);

            let loss = self.compute_loss(input)?;
            if loss < best_loss {
                best_loss = loss;
                best_parameters = self.parameters.clone();
            }

            self.gradient_history.push(gradients);
            self.current_iteration = iteration;

            if loss < self.config.quantum_optimization_tolerance {
                break;
            }
        }

        self.parameters = best_parameters;
        self.forward(input)
    }

    /// Forward pass: encode, run the circuit, measure `⟨Z⟩` per qubit.
    ///
    /// The output has the same shape as the input; every feature inherits the
    /// expectation value of the qubit that encodes its group.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (data, d_model, shape) = self.feature_rows(input)?;
        let bounds = self.group_bounds(d_model);
        let rows = data.len() / d_model;

        let mut output = vec![0.0f32; data.len()];
        for row in 0..rows {
            let slice = &data[row * d_model..(row + 1) * d_model];
            let encoding = self.encode(slice, &bounds);
            let expectations = self.circuit.expectations(&encoding, &self.parameters)?;
            for (qubit, (start, end)) in bounds.iter().enumerate() {
                for column in *start..*end {
                    output[row * d_model + column] = expectations[qubit] as f32;
                }
            }
        }

        Tensor::from_vec(output, &shape)
    }

    /// Measured `⟨Z⟩` expectation values, one row per input feature vector.
    pub fn measure(&self, input: &Tensor) -> Result<Vec<Vec<f64>>> {
        let (data, d_model, _) = self.feature_rows(input)?;
        let bounds = self.group_bounds(d_model);
        let rows = data.len() / d_model;

        (0..rows)
            .map(|row| {
                let slice = &data[row * d_model..(row + 1) * d_model];
                let encoding = self.encode(slice, &bounds);
                self.circuit.expectations(&encoding, &self.parameters)
            })
            .collect()
    }

    /// Gradient of [`Self::compute_loss`] via the parameter-shift rule.
    ///
    /// `L = mean_v Σ_q ⟨Z_q⟩(v)²`, so
    /// `∂L/∂θ_k = mean_v Σ_q 2 ⟨Z_q⟩ · ∂⟨Z_q⟩/∂θ_k`, where the inner
    /// derivative is exact under the parameter-shift rule.
    pub fn compute_gradients(&self, input: &Tensor) -> Result<Vec<f64>> {
        let (data, d_model, _) = self.feature_rows(input)?;
        let bounds = self.group_bounds(d_model);
        let rows = data.len() / d_model;
        if rows == 0 {
            return Ok(vec![0.0; self.parameters.len()]);
        }

        let mut gradients = vec![0.0f64; self.parameters.len()];
        for row in 0..rows {
            let slice = &data[row * d_model..(row + 1) * d_model];
            let encoding = self.encode(slice, &bounds);
            let expectations = self.circuit.expectations(&encoding, &self.parameters)?;
            let jacobian = self.circuit.expectation_jacobian(&encoding, &self.parameters)?;

            for (qubit, expectation) in expectations.iter().enumerate() {
                for (k, gradient) in gradients.iter_mut().enumerate() {
                    *gradient += 2.0 * expectation * jacobian[qubit][k];
                }
            }
        }

        for gradient in gradients.iter_mut() {
            *gradient /= rows as f64;
        }

        Ok(gradients)
    }

    /// Gradient descent step over the circuit parameters.
    fn update_parameters(&mut self, gradients: &[f64]) {
        let index = self.current_iteration.min(self.learning_rate_schedule.len().saturating_sub(1));
        let learning_rate = self
            .learning_rate_schedule
            .get(index)
            .copied()
            .unwrap_or(self.config.quantum_learning_rate);

        for (parameter, gradient) in self.parameters.iter_mut().zip(gradients) {
            *parameter -= learning_rate * gradient;
        }
    }

    /// Loss `mean_v Σ_q ⟨Z_q⟩(v)²`, minimised when every measured qubit is
    /// maximally mixed in the computational basis.
    pub fn compute_loss(&self, input: &Tensor) -> Result<f64> {
        let measurements = self.measure(input)?;
        if measurements.is_empty() {
            return Ok(0.0);
        }
        let total: f64 =
            measurements.iter().map(|row| row.iter().map(|z| z * z).sum::<f64>()).sum();
        Ok(total / measurements.len() as f64)
    }

    /// State fidelity `|⟨0…0|ψ(θ)⟩|²` of the circuit applied to a zero input.
    ///
    /// This is an exactly computed quantity, not a configuration constant.
    pub fn reference_fidelity(&self) -> Result<f64> {
        let encoding = vec![0.0; self.circuit.num_qubits()];
        let state = self.circuit.run(&encoding, &self.parameters)?;
        Ok(state.probability(0))
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    /// Get memory usage in MB
    pub fn memory_usage(&self) -> f32 {
        (self.parameters.len() * 8
            + self.gradient_history.len() * self.parameters.len() * 8
            + self.learning_rate_schedule.len() * 8) as f32
            / 1_000_000.0
    }
}
