//! Quantum operations and primitives
//!
//! This module defines fundamental quantum operations that can be used
//! in quantum neural networks and hybrid quantum-classical models.

#![allow(unused_variables)] // Quantum operations implementation

use anyhow::Result;

/// Quantum state representation
#[derive(Debug, Clone)]
pub struct QuantumState {
    pub amplitudes: Vec<Complex>,
    pub num_qubits: usize,
}

/// Complex number for quantum amplitudes
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

/// Quantum operation trait
pub trait QuantumOperation: std::fmt::Debug {
    fn apply(&self, state: &QuantumState) -> Result<QuantumState>;
    fn dagger(&self) -> Box<dyn QuantumOperation>;
    fn num_qubits(&self) -> usize;
    fn operation_name(&self) -> String;
    fn target_qubits(&self) -> Option<Vec<usize>>;
}

/// Parameterized quantum operation
pub trait ParameterizedQuantumOperation: QuantumOperation {
    fn parameters(&self) -> &[f64];
    fn set_parameters(&mut self, params: &[f64]) -> Result<()>;
    fn gradient(&self, state: &QuantumState, param_index: usize) -> Result<QuantumState>;
}

/// Single-qubit rotation gate
#[derive(Debug, Clone)]
pub struct RotationGate {
    pub qubit: usize,
    pub axis: RotationAxis,
    pub angle: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RotationAxis {
    X,
    Y,
    Z,
}

/// Two-qubit entangling gate
#[derive(Debug, Clone)]
pub struct EntanglingGate {
    pub control: usize,
    pub target: usize,
    pub gate_type: EntanglingType,
    pub parameters: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntanglingType {
    CNOT,
    CZ,
    RZZ,
    RXX,
    RYY,
}

/// Quantum Fourier Transform
#[derive(Debug, Clone)]
pub struct QuantumFourierTransform {
    pub qubits: Vec<usize>,
    pub inverse: bool,
}

/// Variational quantum ansatz
#[derive(Debug, Clone)]
pub struct VariationalAnsatz {
    pub qubits: Vec<usize>,
    pub layers: Vec<AnsatzLayer>,
    pub parameters: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct AnsatzLayer {
    pub single_qubit_rotations: Vec<RotationGate>,
    pub entangling_gates: Vec<EntanglingGate>,
}

/// Quantum measurement operation
#[derive(Debug, Clone)]
pub struct QuantumMeasurement {
    pub qubits: Vec<usize>,
    pub basis: MeasurementBasis,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum MeasurementBasis {
    Computational,
    Pauli(RotationAxis),
    Custom,
}

impl Complex {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn magnitude(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    pub fn phase(&self) -> f64 {
        self.imag.atan2(self.real)
    }

    pub fn conjugate(&self) -> Self {
        Self::new(self.real, -self.imag)
    }
}

impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.real + other.real, self.imag + other.imag)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )
    }
}

impl QuantumState {
    /// Create a new quantum state in |0⟩ state
    pub fn zero_state(num_qubits: usize) -> Self {
        let size = 2_usize.pow(num_qubits as u32);
        let mut amplitudes = vec![Complex::new(0.0, 0.0); size];
        amplitudes[0] = Complex::new(1.0, 0.0); // |000...0⟩

        Self {
            amplitudes,
            num_qubits,
        }
    }

    /// Create a superposition state
    pub fn superposition_state(num_qubits: usize) -> Self {
        let size = 2_usize.pow(num_qubits as u32);
        let amplitude = Complex::new(1.0 / (size as f64).sqrt(), 0.0);
        let amplitudes = vec![amplitude; size];

        Self {
            amplitudes,
            num_qubits,
        }
    }

    /// Get probability of measuring a specific computational basis state
    pub fn probability(&self, state_index: usize) -> f64 {
        if state_index >= self.amplitudes.len() {
            0.0
        } else {
            self.amplitudes[state_index].magnitude().powi(2)
        }
    }

    /// Normalize the quantum state
    pub fn normalize(&mut self) {
        let norm: f64 =
            self.amplitudes.iter().map(|amp| amp.magnitude().powi(2)).sum::<f64>().sqrt();

        if norm > 1e-10 {
            for amp in &mut self.amplitudes {
                amp.real /= norm;
                amp.imag /= norm;
            }
        }
    }

    /// Calculate fidelity with another state
    pub fn fidelity(&self, other: &QuantumState) -> f64 {
        if self.num_qubits != other.num_qubits {
            return 0.0;
        }

        let overlap: Complex = self
            .amplitudes
            .iter()
            .zip(&other.amplitudes)
            .map(|(a, b)| a.conjugate() * *b)
            .fold(Complex::new(0.0, 0.0), |acc, x| acc + x);

        overlap.magnitude().powi(2)
    }
}

impl QuantumOperation for RotationGate {
    fn apply(&self, state: &QuantumState) -> Result<QuantumState> {
        let mut new_state = state.clone();

        // Apply single-qubit rotation
        let cos_half = (self.angle / 2.0).cos();
        let sin_half = (self.angle / 2.0).sin();

        let size = state.amplitudes.len();
        let qubit_mask = 1 << self.qubit;

        for i in 0..size {
            if i & qubit_mask == 0 {
                let j = i | qubit_mask;
                let amp0 = state.amplitudes[i];
                let amp1 = state.amplitudes[j];

                match self.axis {
                    RotationAxis::X => {
                        new_state.amplitudes[i] = Complex::new(
                            cos_half * amp0.real - sin_half * amp1.imag,
                            cos_half * amp0.imag + sin_half * amp1.real,
                        );
                        new_state.amplitudes[j] = Complex::new(
                            cos_half * amp1.real - sin_half * amp0.imag,
                            cos_half * amp1.imag + sin_half * amp0.real,
                        );
                    },
                    RotationAxis::Y => {
                        new_state.amplitudes[i] = Complex::new(
                            cos_half * amp0.real + sin_half * amp1.real,
                            cos_half * amp0.imag + sin_half * amp1.imag,
                        );
                        new_state.amplitudes[j] = Complex::new(
                            cos_half * amp1.real - sin_half * amp0.real,
                            cos_half * amp1.imag - sin_half * amp0.imag,
                        );
                    },
                    RotationAxis::Z => {
                        new_state.amplitudes[i] = Complex::new(
                            cos_half * amp0.real - sin_half * amp0.imag,
                            cos_half * amp0.imag + sin_half * amp0.real,
                        );
                        new_state.amplitudes[j] = Complex::new(
                            cos_half * amp1.real + sin_half * amp1.imag,
                            cos_half * amp1.imag - sin_half * amp1.real,
                        );
                    },
                }
            }
        }

        Ok(new_state)
    }

    fn dagger(&self) -> Box<dyn QuantumOperation> {
        Box::new(RotationGate {
            qubit: self.qubit,
            axis: self.axis,
            angle: -self.angle,
        })
    }

    fn num_qubits(&self) -> usize {
        self.qubit + 1
    }

    fn operation_name(&self) -> String {
        format!("R{:?}({:.3})_{}", self.axis, self.angle, self.qubit)
    }

    fn target_qubits(&self) -> Option<Vec<usize>> {
        Some(vec![self.qubit])
    }
}

impl ParameterizedQuantumOperation for RotationGate {
    fn parameters(&self) -> &[f64] {
        std::slice::from_ref(&self.angle)
    }

    fn set_parameters(&mut self, params: &[f64]) -> Result<()> {
        if params.len() != 1 {
            return Err(anyhow::anyhow!("RotationGate requires exactly 1 parameter"));
        }
        self.angle = params[0];
        Ok(())
    }

    fn gradient(&self, state: &QuantumState, param_index: usize) -> Result<QuantumState> {
        if param_index != 0 {
            return Err(anyhow::anyhow!("Invalid parameter index for RotationGate"));
        }

        // Gradient of rotation gate using parameter shift rule
        let shifted_gate_plus = RotationGate {
            qubit: self.qubit,
            axis: self.axis,
            angle: self.angle + std::f64::consts::PI / 2.0,
        };

        let shifted_gate_minus = RotationGate {
            qubit: self.qubit,
            axis: self.axis,
            angle: self.angle - std::f64::consts::PI / 2.0,
        };

        let state_plus = shifted_gate_plus.apply(state)?;
        let state_minus = shifted_gate_minus.apply(state)?;

        // Compute gradient: (f(θ + π/2) - f(θ - π/2)) / 2
        let mut gradient_state = state_plus.clone();
        for (i, (plus_amp, minus_amp)) in
            state_plus.amplitudes.iter().zip(&state_minus.amplitudes).enumerate()
        {
            gradient_state.amplitudes[i] = Complex::new(
                (plus_amp.real - minus_amp.real) / 2.0,
                (plus_amp.imag - minus_amp.imag) / 2.0,
            );
        }

        Ok(gradient_state)
    }
}

impl QuantumOperation for EntanglingGate {
    fn apply(&self, state: &QuantumState) -> Result<QuantumState> {
        let mut new_state = state.clone();

        match self.gate_type {
            EntanglingType::CNOT => {
                self.apply_cnot(&mut new_state)?;
            },
            EntanglingType::CZ => {
                self.apply_cz(&mut new_state)?;
            },
            EntanglingType::RZZ => {
                if self.parameters.is_empty() {
                    return Err(anyhow::anyhow!("RZZ gate requires angle parameter"));
                }
                self.apply_rzz(&mut new_state, self.parameters[0])?;
            },
            EntanglingType::RXX => {
                if self.parameters.is_empty() {
                    return Err(anyhow::anyhow!("RXX gate requires angle parameter"));
                }
                self.apply_rxx(&mut new_state, self.parameters[0])?;
            },
            EntanglingType::RYY => {
                if self.parameters.is_empty() {
                    return Err(anyhow::anyhow!("RYY gate requires angle parameter"));
                }
                self.apply_ryy(&mut new_state, self.parameters[0])?;
            },
        }

        Ok(new_state)
    }

    fn dagger(&self) -> Box<dyn QuantumOperation> {
        match self.gate_type {
            EntanglingType::CNOT | EntanglingType::CZ => {
                // Self-inverse gates
                Box::new(self.clone())
            },
            EntanglingType::RZZ | EntanglingType::RXX | EntanglingType::RYY => {
                let mut inverted = self.clone();
                if !inverted.parameters.is_empty() {
                    inverted.parameters[0] = -inverted.parameters[0];
                }
                Box::new(inverted)
            },
        }
    }

    fn num_qubits(&self) -> usize {
        self.control.max(self.target) + 1
    }

    fn operation_name(&self) -> String {
        format!("{:?}_{}_{}", self.gate_type, self.control, self.target)
    }

    fn target_qubits(&self) -> Option<Vec<usize>> {
        Some(vec![self.control, self.target])
    }
}

impl EntanglingGate {
    fn apply_cnot(&self, state: &mut QuantumState) -> Result<()> {
        let size = state.amplitudes.len();
        let control_mask = 1 << self.control;
        let target_mask = 1 << self.target;

        for i in 0..size {
            if i & control_mask != 0 {
                // Control qubit is |1⟩
                let j = i ^ target_mask; // Flip target qubit
                if i < j {
                    state.amplitudes.swap(i, j);
                }
            }
        }

        Ok(())
    }

    fn apply_cz(&self, state: &mut QuantumState) -> Result<()> {
        let size = state.amplitudes.len();
        let control_mask = 1 << self.control;
        let target_mask = 1 << self.target;

        for i in 0..size {
            if (i & control_mask != 0) && (i & target_mask != 0) {
                // Both qubits are |1⟩, apply phase flip
                state.amplitudes[i].real = -state.amplitudes[i].real;
                state.amplitudes[i].imag = -state.amplitudes[i].imag;
            }
        }

        Ok(())
    }

    fn apply_rzz(&self, state: &mut QuantumState, angle: f64) -> Result<()> {
        let size = state.amplitudes.len();
        let control_mask = 1 << self.control;
        let target_mask = 1 << self.target;
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        for i in 0..size {
            let parity = ((i & control_mask != 0) as i32) ^ ((i & target_mask != 0) as i32);
            let phase_factor = if parity == 0 {
                Complex::new(cos_half, -sin_half)
            } else {
                Complex::new(cos_half, sin_half)
            };

            state.amplitudes[i] = state.amplitudes[i] * phase_factor;
        }

        Ok(())
    }

    fn apply_rxx(&self, state: &mut QuantumState, angle: f64) -> Result<()> {
        let size = state.amplitudes.len();
        let control_mask = 1 << self.control;
        let target_mask = 1 << self.target;
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        for i in 0..size {
            let control_bit = (i & control_mask) != 0;
            let target_bit = (i & target_mask) != 0;

            // RXX gate mixes amplitudes between states that differ by both qubits
            let j = i ^ control_mask ^ target_mask;

            if i < j {
                let amp_i = state.amplitudes[i];
                let amp_j = state.amplitudes[j];

                // Apply RXX rotation matrix
                // RXX = exp(-i * angle/2 * X ⊗ X)
                state.amplitudes[i] = Complex::new(
                    cos_half * amp_i.real - sin_half * amp_j.imag,
                    cos_half * amp_i.imag + sin_half * amp_j.real,
                );
                state.amplitudes[j] = Complex::new(
                    cos_half * amp_j.real - sin_half * amp_i.imag,
                    cos_half * amp_j.imag + sin_half * amp_i.real,
                );
            }
        }

        Ok(())
    }

    fn apply_ryy(&self, state: &mut QuantumState, angle: f64) -> Result<()> {
        let size = state.amplitudes.len();
        let control_mask = 1 << self.control;
        let target_mask = 1 << self.target;
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        for i in 0..size {
            let control_bit = (i & control_mask) != 0;
            let target_bit = (i & target_mask) != 0;

            // RYY gate mixes amplitudes between states that differ by both qubits
            let j = i ^ control_mask ^ target_mask;

            if i < j {
                let amp_i = state.amplitudes[i];
                let amp_j = state.amplitudes[j];

                // Apply RYY rotation matrix
                // RYY = exp(-i * angle/2 * Y ⊗ Y)
                // The Y ⊗ Y operator has a different sign pattern than X ⊗ X
                let parity = (control_bit as i32) ^ (target_bit as i32);
                let sign = if parity == 0 { 1.0 } else { -1.0 };

                state.amplitudes[i] = Complex::new(
                    cos_half * amp_i.real + sign * sin_half * amp_j.real,
                    cos_half * amp_i.imag + sign * sin_half * amp_j.imag,
                );
                state.amplitudes[j] = Complex::new(
                    cos_half * amp_j.real - sign * sin_half * amp_i.real,
                    cos_half * amp_j.imag - sign * sin_half * amp_i.imag,
                );
            }
        }

        Ok(())
    }
}

impl VariationalAnsatz {
    /// Create a new variational ansatz
    pub fn new(qubits: Vec<usize>, depth: usize) -> Self {
        let mut layers = Vec::new();
        let mut parameters = Vec::new();

        for layer_idx in 0..depth {
            let mut single_qubit_rotations = Vec::new();
            let mut entangling_gates = Vec::new();

            // Add rotation gates for each qubit
            for &qubit in &qubits {
                for axis in [RotationAxis::X, RotationAxis::Y, RotationAxis::Z] {
                    single_qubit_rotations.push(RotationGate {
                        qubit,
                        axis,
                        angle: 0.0, // Will be set from parameters
                    });
                    parameters.push(0.1 * (layer_idx as f64 + 1.0)); // Initialize with small values
                }
            }

            // Add entangling gates
            for i in 0..qubits.len() {
                let next_i = (i + 1) % qubits.len();
                entangling_gates.push(EntanglingGate {
                    control: qubits[i],
                    target: qubits[next_i],
                    gate_type: EntanglingType::CNOT,
                    parameters: vec![],
                });
            }

            layers.push(AnsatzLayer {
                single_qubit_rotations,
                entangling_gates,
            });
        }

        Self {
            qubits,
            layers,
            parameters,
        }
    }

    /// Apply the ansatz to a quantum state
    pub fn apply(&self, state: &QuantumState) -> Result<QuantumState> {
        let mut current_state = state.clone();
        let mut param_idx = 0;

        for layer in &self.layers {
            // Apply single-qubit rotations
            for rotation in &layer.single_qubit_rotations {
                let mut parameterized_rotation = rotation.clone();
                parameterized_rotation.angle = self.parameters[param_idx];
                param_idx += 1;

                current_state = parameterized_rotation.apply(&current_state)?;
            }

            // Apply entangling gates
            for entangling in &layer.entangling_gates {
                current_state = entangling.apply(&current_state)?;
            }
        }

        Ok(current_state)
    }

    /// Update parameters
    pub fn set_parameters(&mut self, new_params: Vec<f64>) -> Result<()> {
        if new_params.len() != self.parameters.len() {
            return Err(anyhow::anyhow!(
                "Parameter count mismatch: expected {}, got {}",
                self.parameters.len(),
                new_params.len()
            ));
        }
        self.parameters = new_params;
        Ok(())
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);

        let sum = a + b;
        assert!((sum.real - 4.0).abs() < 1e-10);
        assert!((sum.imag - 6.0).abs() < 1e-10);

        let product = a * b;
        assert!((product.real + 5.0).abs() < 1e-10); // 1*3 - 2*4 = -5
        assert!((product.imag - 10.0).abs() < 1e-10); // 1*4 + 2*3 = 10

        assert!((a.magnitude() - (5.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::zero_state(2);
        assert_eq!(state.num_qubits, 2);
        assert_eq!(state.amplitudes.len(), 4);
        assert!((state.amplitudes[0].real - 1.0).abs() < 1e-10);
        assert!(state.amplitudes[1].magnitude() < 1e-10);

        let superposition = QuantumState::superposition_state(2);
        assert!((superposition.amplitudes[0].real - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_gate() {
        let state = QuantumState::zero_state(1);
        let x_gate = RotationGate {
            qubit: 0,
            axis: RotationAxis::X,
            angle: std::f64::consts::PI,
        };

        let result = x_gate.apply(&state).unwrap();
        assert!(result.amplitudes[0].magnitude() < 1e-10);
        assert!((result.amplitudes[1].magnitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cnot_gate() {
        let mut state = QuantumState::zero_state(2);
        // Prepare |10⟩ state
        state.amplitudes[0] = Complex::new(0.0, 0.0);
        state.amplitudes[2] = Complex::new(1.0, 0.0);

        let cnot = EntanglingGate {
            control: 1,
            target: 0,
            gate_type: EntanglingType::CNOT,
            parameters: vec![],
        };

        let result = cnot.apply(&state).unwrap();
        assert!(result.amplitudes[2].magnitude() < 1e-10);
        assert!((result.amplitudes[3].magnitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_variational_ansatz() {
        let ansatz = VariationalAnsatz::new(vec![0, 1], 2);
        assert_eq!(ansatz.qubits, vec![0, 1]);
        assert_eq!(ansatz.layers.len(), 2);
        assert!(ansatz.num_parameters() > 0);

        let state = QuantumState::zero_state(2);
        let result = ansatz.apply(&state);
        assert!(result.is_ok());
    }

    #[test]
    fn test_state_fidelity() {
        let state1 = QuantumState::zero_state(2);
        let state2 = QuantumState::zero_state(2);
        let state3 = QuantumState::superposition_state(2);

        assert!((state1.fidelity(&state2) - 1.0).abs() < 1e-10);
        assert!(state1.fidelity(&state3) < 1.0);
        assert!(state1.fidelity(&state3) > 0.0);
    }

    #[test]
    fn test_parameter_gradient() {
        let state = QuantumState::zero_state(1);
        let rotation = RotationGate {
            qubit: 0,
            axis: RotationAxis::X,
            angle: 0.5,
        };

        let gradient = rotation.gradient(&state, 0);
        assert!(gradient.is_ok());
    }
}
