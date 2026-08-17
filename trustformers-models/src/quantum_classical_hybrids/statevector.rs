//! Dense statevector simulator for the quantum-classical hybrid models.
//!
//! The simulator keeps the full `2^n` complex amplitude vector and applies
//! gates by direct amplitude updates, so every expectation value it reports is
//! the exact noiseless result of the circuit rather than an approximation.
//!
//! Conventions:
//!
//! * Qubit `q` corresponds to bit `q` of the basis-state index (qubit 0 is the
//!   least significant bit).
//! * `RX(θ) = exp(−i θ X / 2)`, `RY(θ) = exp(−i θ Y / 2)` and
//!   `RZ(θ) = exp(−i θ Z / 2)`, i.e. the standard textbook convention.
//! * Because every rotation generator `P` satisfies `P² = I`, the derivative of
//!   an expectation value with respect to a rotation angle is exactly
//!   `[f(θ + π/2) − f(θ − π/2)] / 2` — the parameter-shift rule, which
//!   [`VariationalCircuit::expectation_jacobian`] implements.

use std::f64::consts::FRAC_1_SQRT_2;

use trustformers_core::errors::{Result, TrustformersError};

/// Largest number of qubits the dense simulator will allocate.
///
/// `2^20` complex amplitudes is 16 MiB, which is the practical ceiling for a
/// CPU statevector simulator inside a model forward pass.
pub const MAX_SIMULATED_QUBITS: usize = 20;

/// Shift used by the parameter-shift rule.
pub const PARAMETER_SHIFT: f64 = std::f64::consts::FRAC_PI_2;

/// A complex amplitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Amplitude {
    /// Real part
    pub re: f64,
    /// Imaginary part
    pub im: f64,
}

impl Amplitude {
    /// Create a complex amplitude.
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// The zero amplitude.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// Complex multiplication.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }

    /// Complex addition.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }

    /// Complex conjugate.
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Squared magnitude `|z|²`.
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Magnitude `|z|`.
    pub fn magnitude(self) -> f64 {
        self.norm_sqr().sqrt()
    }
}

/// A single gate in a circuit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantumGate {
    /// Hadamard
    Hadamard(usize),
    /// Pauli-X
    PauliX(usize),
    /// Pauli-Y
    PauliY(usize),
    /// Pauli-Z
    PauliZ(usize),
    /// `exp(-i θ X / 2)`
    RotationX {
        /// Target qubit
        qubit: usize,
        /// Rotation angle
        angle: f64,
    },
    /// `exp(-i θ Y / 2)`
    RotationY {
        /// Target qubit
        qubit: usize,
        /// Rotation angle
        angle: f64,
    },
    /// `exp(-i θ Z / 2)`
    RotationZ {
        /// Target qubit
        qubit: usize,
        /// Rotation angle
        angle: f64,
    },
    /// Controlled NOT
    ControlledNot {
        /// Control qubit
        control: usize,
        /// Target qubit
        target: usize,
    },
    /// Controlled Z
    ControlledZ {
        /// Control qubit
        control: usize,
        /// Target qubit
        target: usize,
    },
}

/// A dense statevector.
#[derive(Debug, Clone, PartialEq)]
pub struct StateVector {
    amplitudes: Vec<Amplitude>,
    num_qubits: usize,
}

impl StateVector {
    /// Allocate the `|0…0⟩` state on `num_qubits` qubits.
    pub fn zero_state(num_qubits: usize) -> Result<Self> {
        if num_qubits == 0 {
            return Err(TrustformersError::invalid_input(
                "a statevector needs at least one qubit".to_string(),
            ));
        }
        if num_qubits > MAX_SIMULATED_QUBITS {
            return Err(TrustformersError::invalid_input(format!(
                "dense statevector simulation is limited to {} qubits, got {}",
                MAX_SIMULATED_QUBITS, num_qubits
            )));
        }

        let mut amplitudes = vec![Amplitude::ZERO; 1usize << num_qubits];
        amplitudes[0] = Amplitude::new(1.0, 0.0);
        Ok(Self {
            amplitudes,
            num_qubits,
        })
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// The raw amplitude vector.
    pub fn amplitudes(&self) -> &[Amplitude] {
        &self.amplitudes
    }

    /// Probability of the computational basis state `index`.
    pub fn probability(&self, index: usize) -> f64 {
        self.amplitudes.get(index).map(|a| a.norm_sqr()).unwrap_or(0.0)
    }

    /// The `L2` norm of the state (1 for a valid state).
    pub fn norm(&self) -> f64 {
        self.amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt()
    }

    /// State fidelity `|⟨ψ|φ⟩|²` with another state of the same width.
    pub fn fidelity(&self, other: &StateVector) -> Result<f64> {
        if self.num_qubits != other.num_qubits {
            return Err(TrustformersError::invalid_input(format!(
                "fidelity needs matching qubit counts, got {} and {}",
                self.num_qubits, other.num_qubits
            )));
        }
        let overlap = self
            .amplitudes
            .iter()
            .zip(other.amplitudes.iter())
            .fold(Amplitude::ZERO, |acc, (a, b)| acc.add(a.conj().mul(*b)));
        Ok(overlap.norm_sqr())
    }

    fn check_qubit(&self, qubit: usize) -> Result<()> {
        if qubit >= self.num_qubits {
            return Err(TrustformersError::invalid_input(format!(
                "qubit {} is out of range for a {}-qubit state",
                qubit, self.num_qubits
            )));
        }
        Ok(())
    }

    /// Apply an arbitrary single-qubit unitary `[[a, b], [c, d]]`.
    fn apply_single(&mut self, qubit: usize, matrix: [Amplitude; 4]) -> Result<()> {
        self.check_qubit(qubit)?;
        let mask = 1usize << qubit;
        let [a, b, c, d] = matrix;

        for index in 0..self.amplitudes.len() {
            if index & mask != 0 {
                continue;
            }
            let partner = index | mask;
            let amp0 = self.amplitudes[index];
            let amp1 = self.amplitudes[partner];
            self.amplitudes[index] = a.mul(amp0).add(b.mul(amp1));
            self.amplitudes[partner] = c.mul(amp0).add(d.mul(amp1));
        }

        Ok(())
    }

    /// Apply one gate in place.
    pub fn apply(&mut self, gate: &QuantumGate) -> Result<()> {
        match *gate {
            QuantumGate::Hadamard(qubit) => {
                let s = FRAC_1_SQRT_2;
                self.apply_single(
                    qubit,
                    [
                        Amplitude::new(s, 0.0),
                        Amplitude::new(s, 0.0),
                        Amplitude::new(s, 0.0),
                        Amplitude::new(-s, 0.0),
                    ],
                )
            },
            QuantumGate::PauliX(qubit) => self.apply_single(
                qubit,
                [
                    Amplitude::ZERO,
                    Amplitude::new(1.0, 0.0),
                    Amplitude::new(1.0, 0.0),
                    Amplitude::ZERO,
                ],
            ),
            QuantumGate::PauliY(qubit) => self.apply_single(
                qubit,
                [
                    Amplitude::ZERO,
                    Amplitude::new(0.0, -1.0),
                    Amplitude::new(0.0, 1.0),
                    Amplitude::ZERO,
                ],
            ),
            QuantumGate::PauliZ(qubit) => self.apply_single(
                qubit,
                [
                    Amplitude::new(1.0, 0.0),
                    Amplitude::ZERO,
                    Amplitude::ZERO,
                    Amplitude::new(-1.0, 0.0),
                ],
            ),
            QuantumGate::RotationX { qubit, angle } => {
                let (sin, cos) = (angle / 2.0).sin_cos();
                self.apply_single(
                    qubit,
                    [
                        Amplitude::new(cos, 0.0),
                        Amplitude::new(0.0, -sin),
                        Amplitude::new(0.0, -sin),
                        Amplitude::new(cos, 0.0),
                    ],
                )
            },
            QuantumGate::RotationY { qubit, angle } => {
                let (sin, cos) = (angle / 2.0).sin_cos();
                self.apply_single(
                    qubit,
                    [
                        Amplitude::new(cos, 0.0),
                        Amplitude::new(-sin, 0.0),
                        Amplitude::new(sin, 0.0),
                        Amplitude::new(cos, 0.0),
                    ],
                )
            },
            QuantumGate::RotationZ { qubit, angle } => {
                let (sin, cos) = (angle / 2.0).sin_cos();
                self.apply_single(
                    qubit,
                    [
                        Amplitude::new(cos, -sin),
                        Amplitude::ZERO,
                        Amplitude::ZERO,
                        Amplitude::new(cos, sin),
                    ],
                )
            },
            QuantumGate::ControlledNot { control, target } => {
                self.check_qubit(control)?;
                self.check_qubit(target)?;
                if control == target {
                    return Err(TrustformersError::invalid_input(
                        "CNOT control and target must differ".to_string(),
                    ));
                }
                let control_mask = 1usize << control;
                let target_mask = 1usize << target;
                for index in 0..self.amplitudes.len() {
                    if index & control_mask != 0 && index & target_mask == 0 {
                        self.amplitudes.swap(index, index | target_mask);
                    }
                }
                Ok(())
            },
            QuantumGate::ControlledZ { control, target } => {
                self.check_qubit(control)?;
                self.check_qubit(target)?;
                if control == target {
                    return Err(TrustformersError::invalid_input(
                        "CZ control and target must differ".to_string(),
                    ));
                }
                let mask = (1usize << control) | (1usize << target);
                for (index, amplitude) in self.amplitudes.iter_mut().enumerate() {
                    if index & mask == mask {
                        amplitude.re = -amplitude.re;
                        amplitude.im = -amplitude.im;
                    }
                }
                Ok(())
            },
        }
    }

    /// Apply a whole gate sequence in order.
    pub fn apply_all(&mut self, gates: &[QuantumGate]) -> Result<()> {
        for gate in gates {
            self.apply(gate)?;
        }
        Ok(())
    }

    /// Expectation value `⟨Z_q⟩` in `[-1, 1]`.
    pub fn expectation_z(&self, qubit: usize) -> Result<f64> {
        self.check_qubit(qubit)?;
        let mask = 1usize << qubit;
        Ok(self
            .amplitudes
            .iter()
            .enumerate()
            .map(|(index, amplitude)| {
                let sign = if index & mask == 0 { 1.0 } else { -1.0 };
                sign * amplitude.norm_sqr()
            })
            .sum())
    }

    /// `⟨Z_q⟩` for every qubit.
    pub fn expectation_z_all(&self) -> Result<Vec<f64>> {
        (0..self.num_qubits).map(|q| self.expectation_z(q)).collect()
    }
}

/// A hardware-efficient variational circuit: angle encoding, then `layers`
/// repetitions of `RY`/`RZ` rotations followed by a CNOT entangling ring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariationalCircuit {
    num_qubits: usize,
    layers: usize,
}

impl VariationalCircuit {
    /// Build a circuit over `num_qubits` qubits with `layers` variational
    /// layers.
    pub fn new(num_qubits: usize, layers: usize) -> Result<Self> {
        if num_qubits == 0 {
            return Err(TrustformersError::invalid_input(
                "a variational circuit needs at least one qubit".to_string(),
            ));
        }
        if num_qubits > MAX_SIMULATED_QUBITS {
            return Err(TrustformersError::invalid_input(format!(
                "dense statevector simulation is limited to {} qubits, got {}",
                MAX_SIMULATED_QUBITS, num_qubits
            )));
        }
        if layers == 0 {
            return Err(TrustformersError::invalid_input(
                "a variational circuit needs at least one layer".to_string(),
            ));
        }
        Ok(Self { num_qubits, layers })
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of variational layers.
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// Number of free rotation parameters (`2` per qubit per layer).
    pub fn parameter_count(&self) -> usize {
        self.num_qubits * self.layers * 2
    }

    /// Build the gate list for the given encoding angles and parameters.
    pub fn gates(&self, encoding: &[f64], parameters: &[f64]) -> Result<Vec<QuantumGate>> {
        if encoding.len() != self.num_qubits {
            return Err(TrustformersError::invalid_input(format!(
                "expected {} encoding angles, got {}",
                self.num_qubits,
                encoding.len()
            )));
        }
        if parameters.len() != self.parameter_count() {
            return Err(TrustformersError::invalid_input(format!(
                "expected {} circuit parameters, got {}",
                self.parameter_count(),
                parameters.len()
            )));
        }

        let mut gates = Vec::with_capacity(self.parameter_count() + self.num_qubits * 2);

        // Angle encoding of the classical input.
        for (qubit, angle) in encoding.iter().enumerate() {
            gates.push(QuantumGate::RotationY {
                qubit,
                angle: *angle,
            });
        }

        // Variational layers.
        let mut index = 0;
        for _ in 0..self.layers {
            for qubit in 0..self.num_qubits {
                gates.push(QuantumGate::RotationY {
                    qubit,
                    angle: parameters[index],
                });
                gates.push(QuantumGate::RotationZ {
                    qubit,
                    angle: parameters[index + 1],
                });
                index += 2;
            }
            if self.num_qubits > 1 {
                for qubit in 0..self.num_qubits {
                    gates.push(QuantumGate::ControlledNot {
                        control: qubit,
                        target: (qubit + 1) % self.num_qubits,
                    });
                }
            }
        }

        Ok(gates)
    }

    /// Run the circuit and return the resulting statevector.
    pub fn run(&self, encoding: &[f64], parameters: &[f64]) -> Result<StateVector> {
        let gates = self.gates(encoding, parameters)?;
        let mut state = StateVector::zero_state(self.num_qubits)?;
        state.apply_all(&gates)?;
        Ok(state)
    }

    /// `⟨Z_q⟩` for every qubit after running the circuit.
    pub fn expectations(&self, encoding: &[f64], parameters: &[f64]) -> Result<Vec<f64>> {
        self.run(encoding, parameters)?.expectation_z_all()
    }

    /// Parameter-shift Jacobian `∂⟨Z_j⟩ / ∂θ_k`.
    ///
    /// Every parameter drives a single Pauli rotation, so the exact derivative
    /// is `[f(θ_k + π/2) − f(θ_k − π/2)] / 2`. The returned matrix is indexed
    /// `[qubit][parameter]`.
    pub fn expectation_jacobian(
        &self,
        encoding: &[f64],
        parameters: &[f64],
    ) -> Result<Vec<Vec<f64>>> {
        let count = self.parameter_count();
        if parameters.len() != count {
            return Err(TrustformersError::invalid_input(format!(
                "expected {} circuit parameters, got {}",
                count,
                parameters.len()
            )));
        }

        let mut jacobian = vec![vec![0.0f64; count]; self.num_qubits];

        for k in 0..count {
            let mut plus = parameters.to_vec();
            plus[k] += PARAMETER_SHIFT;
            let mut minus = parameters.to_vec();
            minus[k] -= PARAMETER_SHIFT;

            let z_plus = self.expectations(encoding, &plus)?;
            let z_minus = self.expectations(encoding, &minus)?;

            for qubit in 0..self.num_qubits {
                jacobian[qubit][k] = (z_plus[qubit] - z_minus[qubit]) / 2.0;
            }
        }

        Ok(jacobian)
    }
}
