//! Quantum circuit representation and operations

use crate::quantum::quantum_ops::*;
use anyhow::Result;
use std::collections::HashMap;

/// Quantum circuit representation
#[derive(Debug)]
pub struct QuantumCircuit {
    pub num_qubits: usize,
    pub gates: Vec<Box<dyn QuantumOperation>>,
    pub measurements: Vec<QuantumMeasurement>,
    pub classical_registers: HashMap<String, usize>,
}

impl Clone for QuantumCircuit {
    fn clone(&self) -> Self {
        Self {
            num_qubits: self.num_qubits,
            gates: Vec::new(), // Gates cannot be cloned due to trait object
            measurements: self.measurements.clone(),
            classical_registers: self.classical_registers.clone(),
        }
    }
}

/// Quantum ansatz types for neural networks
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum QuantumAnsatz {
    Hardware,
    Efficient {
        layers: usize,
    },
    RealAmplitudes {
        layers: usize,
    },
    TwoLocal {
        rotation_blocks: Vec<RotationAxis>,
        entanglement: String,
    },
}

/// Quantum encoding schemes
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum QuantumEncoding {
    Amplitude,
    Angle,
    Basis,
    IQP { depth: usize },
}

impl QuantumCircuit {
    /// Create a new quantum circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
            measurements: Vec::new(),
            classical_registers: HashMap::new(),
        }
    }

    /// Add a quantum gate to the circuit
    pub fn add_gate(&mut self, gate: Box<dyn QuantumOperation>) {
        self.gates.push(gate);
    }

    /// Add a rotation gate
    pub fn add_rotation(&mut self, qubit: usize, axis: RotationAxis, angle: f64) {
        let gate = RotationGate { qubit, axis, angle };
        self.gates.push(Box::new(gate));
    }

    /// Add a CNOT gate
    pub fn add_cnot(&mut self, control: usize, target: usize) {
        let gate = EntanglingGate {
            control,
            target,
            gate_type: EntanglingType::CNOT,
            parameters: vec![],
        };
        self.gates.push(Box::new(gate));
    }

    /// Add Hadamard gate
    pub fn add_hadamard(&mut self, qubit: usize) {
        self.add_rotation(qubit, RotationAxis::Y, std::f64::consts::PI / 2.0);
        self.add_rotation(qubit, RotationAxis::X, std::f64::consts::PI);
    }

    /// Add measurement
    pub fn add_measurement(&mut self, qubits: Vec<usize>, basis: MeasurementBasis) {
        self.measurements.push(QuantumMeasurement { qubits, basis });
    }

    /// Execute the circuit on a quantum state
    pub fn execute(&self, initial_state: &QuantumState) -> Result<QuantumState> {
        let mut state = initial_state.clone();

        for gate in &self.gates {
            state = gate.apply(&state)?;
        }

        Ok(state)
    }

    /// Get circuit depth
    pub fn depth(&self) -> usize {
        // Simplified depth calculation
        self.gates.len()
    }

    /// Get gates
    pub fn gates(&self) -> &[Box<dyn QuantumOperation>] {
        &self.gates
    }

    /// Count gate types
    pub fn gate_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for gate in &self.gates {
            let name = gate.operation_name();
            *counts.entry(name).or_insert(0) += 1;
        }
        counts
    }
}

impl QuantumAnsatz {
    /// Build circuit for the ansatz
    pub fn build_circuit(&self, qubits: &[usize], parameters: &[f64]) -> Result<QuantumCircuit> {
        let num_qubits = qubits.iter().max().unwrap_or(&0) + 1;
        let mut circuit = QuantumCircuit::new(num_qubits);

        match self {
            QuantumAnsatz::Hardware => {
                self.build_hardware_ansatz(&mut circuit, qubits, parameters)?;
            },
            QuantumAnsatz::Efficient { layers } => {
                self.build_efficient_ansatz(&mut circuit, qubits, parameters, *layers)?;
            },
            QuantumAnsatz::RealAmplitudes { layers } => {
                self.build_real_amplitudes(&mut circuit, qubits, parameters, *layers)?;
            },
            QuantumAnsatz::TwoLocal {
                rotation_blocks,
                entanglement,
            } => {
                self.build_two_local(
                    &mut circuit,
                    qubits,
                    parameters,
                    rotation_blocks,
                    entanglement,
                )?;
            },
        }

        Ok(circuit)
    }

    fn build_hardware_ansatz(
        &self,
        circuit: &mut QuantumCircuit,
        qubits: &[usize],
        parameters: &[f64],
    ) -> Result<()> {
        let mut param_idx = 0;

        // Layer of Y rotations
        for &qubit in qubits {
            if param_idx < parameters.len() {
                circuit.add_rotation(qubit, RotationAxis::Y, parameters[param_idx]);
                param_idx += 1;
            }
        }

        // Entangling layer
        for i in 0..qubits.len() {
            let next_i = (i + 1) % qubits.len();
            circuit.add_cnot(qubits[i], qubits[next_i]);
        }

        Ok(())
    }

    fn build_efficient_ansatz(
        &self,
        circuit: &mut QuantumCircuit,
        qubits: &[usize],
        parameters: &[f64],
        layers: usize,
    ) -> Result<()> {
        let mut param_idx = 0;

        for _layer in 0..layers {
            // Rotation layer
            for &qubit in qubits {
                for axis in [RotationAxis::Y, RotationAxis::Z] {
                    if param_idx < parameters.len() {
                        circuit.add_rotation(qubit, axis, parameters[param_idx]);
                        param_idx += 1;
                    }
                }
            }

            // Entangling layer with linear connectivity
            for i in 0..qubits.len().saturating_sub(1) {
                circuit.add_cnot(qubits[i], qubits[i + 1]);
            }
        }

        Ok(())
    }

    fn build_real_amplitudes(
        &self,
        circuit: &mut QuantumCircuit,
        qubits: &[usize],
        parameters: &[f64],
        layers: usize,
    ) -> Result<()> {
        let mut param_idx = 0;

        for _layer in 0..layers {
            // Only Y rotations for real amplitudes
            for &qubit in qubits {
                if param_idx < parameters.len() {
                    circuit.add_rotation(qubit, RotationAxis::Y, parameters[param_idx]);
                    param_idx += 1;
                }
            }

            // Entangling layer
            for i in 0..qubits.len() {
                let next_i = (i + 1) % qubits.len();
                circuit.add_cnot(qubits[i], qubits[next_i]);
            }
        }

        Ok(())
    }

    fn build_two_local(
        &self,
        circuit: &mut QuantumCircuit,
        qubits: &[usize],
        parameters: &[f64],
        rotation_blocks: &[RotationAxis],
        entanglement: &str,
    ) -> Result<()> {
        let mut param_idx = 0;

        // Rotation blocks
        for &qubit in qubits {
            for &axis in rotation_blocks {
                if param_idx < parameters.len() {
                    circuit.add_rotation(qubit, axis, parameters[param_idx]);
                    param_idx += 1;
                }
            }
        }

        // Entanglement pattern
        match entanglement {
            "linear" => {
                for i in 0..qubits.len().saturating_sub(1) {
                    circuit.add_cnot(qubits[i], qubits[i + 1]);
                }
            },
            "circular" => {
                for i in 0..qubits.len() {
                    let next_i = (i + 1) % qubits.len();
                    circuit.add_cnot(qubits[i], qubits[next_i]);
                }
            },
            "full" => {
                for i in 0..qubits.len() {
                    for j in (i + 1)..qubits.len() {
                        circuit.add_cnot(qubits[i], qubits[j]);
                    }
                }
            },
            _ => {
                return Err(anyhow::anyhow!(
                    "Unknown entanglement pattern: {}",
                    entanglement
                ));
            },
        }

        Ok(())
    }

    /// Get the number of parameters for this ansatz
    pub fn num_parameters(&self, num_qubits: usize) -> usize {
        match self {
            QuantumAnsatz::Hardware => num_qubits,
            QuantumAnsatz::Efficient { layers } => layers * num_qubits * 2,
            QuantumAnsatz::RealAmplitudes { layers } => layers * num_qubits,
            QuantumAnsatz::TwoLocal {
                rotation_blocks, ..
            } => rotation_blocks.len() * num_qubits,
        }
    }
}

impl QuantumEncoding {
    /// Encode classical data into quantum state
    pub fn encode(&self, data: &[f64], num_qubits: usize) -> Result<QuantumCircuit> {
        let mut circuit = QuantumCircuit::new(num_qubits);

        match self {
            QuantumEncoding::Amplitude => {
                self.amplitude_encoding(&mut circuit, data, num_qubits)?;
            },
            QuantumEncoding::Angle => {
                self.angle_encoding(&mut circuit, data, num_qubits)?;
            },
            QuantumEncoding::Basis => {
                self.basis_encoding(&mut circuit, data, num_qubits)?;
            },
            QuantumEncoding::IQP { depth } => {
                self.iqp_encoding(&mut circuit, data, num_qubits, *depth)?;
            },
        }

        Ok(circuit)
    }

    fn amplitude_encoding(
        &self,
        circuit: &mut QuantumCircuit,
        data: &[f64],
        num_qubits: usize,
    ) -> Result<()> {
        // Amplitude encoding requires state preparation
        // This is a simplified version - real implementation would need
        // sophisticated state preparation algorithms

        let expected_size = 2_usize.pow(num_qubits as u32);
        if data.len() > expected_size {
            return Err(anyhow::anyhow!(
                "Data size {} exceeds quantum state dimension {}",
                data.len(),
                expected_size
            ));
        }

        // Normalize data
        let norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return Err(anyhow::anyhow!("Data has zero norm"));
        }

        // For now, use a simplified approach with rotation gates
        for (i, &value) in data.iter().enumerate().take(num_qubits) {
            let angle = 2.0 * (value / norm).asin();
            circuit.add_rotation(i, RotationAxis::Y, angle);
        }

        Ok(())
    }

    fn angle_encoding(
        &self,
        circuit: &mut QuantumCircuit,
        data: &[f64],
        num_qubits: usize,
    ) -> Result<()> {
        for (i, &value) in data.iter().enumerate().take(num_qubits) {
            circuit.add_rotation(i, RotationAxis::Y, value);
        }
        Ok(())
    }

    fn basis_encoding(
        &self,
        circuit: &mut QuantumCircuit,
        data: &[f64],
        num_qubits: usize,
    ) -> Result<()> {
        for (i, &value) in data.iter().enumerate().take(num_qubits) {
            if value > 0.5 {
                circuit.add_rotation(i, RotationAxis::X, std::f64::consts::PI);
            }
        }
        Ok(())
    }

    fn iqp_encoding(
        &self,
        circuit: &mut QuantumCircuit,
        data: &[f64],
        num_qubits: usize,
        depth: usize,
    ) -> Result<()> {
        // IQP (Instantaneous Quantum Polynomial) encoding
        for _layer in 0..depth {
            // Hadamard layer
            for qubit in 0..num_qubits {
                circuit.add_hadamard(qubit);
            }

            // Diagonal gates with data encoding
            for (i, &value) in data.iter().enumerate().take(num_qubits) {
                circuit.add_rotation(i, RotationAxis::Z, value);
            }

            // Entangling layer
            for i in 0..num_qubits.saturating_sub(1) {
                circuit.add_cnot(i, i + 1);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_circuit_creation() {
        let circuit = QuantumCircuit::new(3);
        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.gates.len(), 0);
        assert_eq!(circuit.measurements.len(), 0);
    }

    #[test]
    fn test_add_gates() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_rotation(0, RotationAxis::X, std::f64::consts::PI);
        circuit.add_cnot(0, 1);
        circuit.add_hadamard(1);

        assert_eq!(circuit.gates.len(), 4); // H = RY + RX
        assert_eq!(circuit.depth(), 4);
    }

    #[test]
    fn test_circuit_execution() {
        let mut circuit = QuantumCircuit::new(1);
        circuit.add_rotation(0, RotationAxis::X, std::f64::consts::PI);

        let initial_state = QuantumState::zero_state(1);
        let result = circuit.execute(&initial_state).unwrap();

        assert!(result.amplitudes[0].magnitude() < 1e-10);
        assert!((result.amplitudes[1].magnitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hardware_ansatz() {
        let ansatz = QuantumAnsatz::Hardware;
        let qubits = vec![0, 1, 2];
        let parameters = vec![0.1, 0.2, 0.3];

        let circuit = ansatz.build_circuit(&qubits, &parameters).unwrap();
        assert_eq!(circuit.num_qubits, 3);
        assert!(!circuit.gates.is_empty());
    }

    #[test]
    fn test_efficient_ansatz() {
        let ansatz = QuantumAnsatz::Efficient { layers: 2 };
        let qubits = vec![0, 1];
        let parameters = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let circuit = ansatz.build_circuit(&qubits, &parameters).unwrap();
        assert_eq!(circuit.num_qubits, 2);

        let num_params = ansatz.num_parameters(2);
        assert_eq!(num_params, 8); // 2 layers * 2 qubits * 2 rotations
    }

    #[test]
    fn test_angle_encoding() {
        let encoding = QuantumEncoding::Angle;
        let data = vec![0.5, 1.0, 1.5];

        let circuit = encoding.encode(&data, 3).unwrap();
        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.gates.len(), 3);
    }

    #[test]
    fn test_basis_encoding() {
        let encoding = QuantumEncoding::Basis;
        let data = vec![0.3, 0.7, 0.1]; // Only middle value > 0.5

        let circuit = encoding.encode(&data, 3).unwrap();
        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.gates.len(), 1); // Only one X gate for value > 0.5
    }

    #[test]
    fn test_iqp_encoding() {
        let encoding = QuantumEncoding::IQP { depth: 2 };
        let data = vec![0.1, 0.2];

        let circuit = encoding.encode(&data, 2).unwrap();
        assert_eq!(circuit.num_qubits, 2);
        assert!(!circuit.gates.is_empty());
    }

    #[test]
    fn test_gate_counts() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_rotation(0, RotationAxis::X, 0.5);
        circuit.add_rotation(1, RotationAxis::Y, 0.3);
        circuit.add_cnot(0, 1);

        let counts = circuit.gate_counts();
        assert!(!counts.is_empty());
    }

    #[test]
    fn test_measurements() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_measurement(vec![0, 1], MeasurementBasis::Computational);

        assert_eq!(circuit.measurements.len(), 1);
        assert_eq!(circuit.measurements[0].qubits, vec![0, 1]);
    }
}
