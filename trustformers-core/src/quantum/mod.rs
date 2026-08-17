//! Quantum Computing Exploration Framework
//!
//! This module provides experimental quantum computing support for future
//! integration with quantum accelerators and hybrid quantum-classical workflows.

pub mod hybrid_layers;
pub mod quantum_attention;
pub mod quantum_circuit;
pub mod quantum_embeddings;
pub mod quantum_gates;
pub mod quantum_ops;

pub use hybrid_layers::*;
pub use quantum_circuit::*;
pub use quantum_gates::*;
pub use quantum_ops::*;

use anyhow::Result;
use scirs2_core::random::*; // SciRS2 Integration Policy
use std::collections::HashMap;

/// Quantum computing backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QuantumBackend {
    Simulator,
    Qiskit,
    Cirq,
    PennyLane,
    Braket,
    IonQ,
    Rigetti,
}

/// Measurement basis
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MeasurementBasis {
    Computational,
    Pauli,
    Bell,
    Custom,
}

/// Quantum device configuration
#[derive(Debug, Clone)]
pub struct QuantumDevice {
    pub backend: QuantumBackend,
    pub num_qubits: usize,
    pub connectivity: QuantumConnectivity,
    pub noise_model: Option<NoiseModel>,
    pub calibration: Option<DeviceCalibration>,
}

/// Quantum connectivity graph
#[derive(Debug, Clone)]
pub enum QuantumConnectivity {
    FullyConnected,
    Linear,
    Grid { rows: usize, cols: usize },
    Custom { edges: Vec<(usize, usize)> },
}

/// Noise model for quantum simulations
#[derive(Debug, Clone)]
pub struct NoiseModel {
    pub gate_error_rates: HashMap<String, f64>,
    pub readout_error: f64,
    pub decoherence_time: Option<f64>,
    pub thermal_noise: bool,
}

/// Device calibration data
#[derive(Debug, Clone)]
pub struct DeviceCalibration {
    pub gate_fidelities: HashMap<String, f64>,
    pub qubit_frequencies: Vec<f64>,
    pub coupling_strengths: Vec<f64>,
    pub timestamp: u64,
}

/// Quantum measurement result
#[derive(Debug, Clone)]
pub struct QuantumMeasurement {
    pub counts: HashMap<String, usize>,
    pub probabilities: HashMap<String, f64>,
    pub shots: usize,
}

/// Main quantum computing manager
#[derive(Debug)]
pub struct QuantumManager {
    device: QuantumDevice,
    circuit_cache: HashMap<String, QuantumCircuit>,
    optimization_enabled: bool,
    /// Measurement shots taken per circuit execution.
    shots: usize,
}

impl QuantumManager {
    /// Widest circuit the state-vector simulator will attempt.
    ///
    /// 20 qubits is 2^20 complex amplitudes (~16 MiB at f64 pairs); beyond
    /// that the simulation is refused rather than approximated.
    pub const MAX_SIMULATED_QUBITS: usize = 20;

    /// Default number of measurement shots.
    pub const DEFAULT_SHOTS: usize = 1024;

    /// Create a new quantum manager
    pub fn new(device: QuantumDevice) -> Self {
        Self {
            device,
            circuit_cache: HashMap::new(),
            optimization_enabled: true,
            shots: Self::DEFAULT_SHOTS,
        }
    }

    /// Set the number of measurement shots taken per execution.
    pub fn set_shots(&mut self, shots: usize) -> Result<()> {
        if shots == 0 {
            return Err(anyhow::anyhow!("shot count must be at least 1"));
        }
        self.shots = shots;
        Ok(())
    }

    /// The number of measurement shots taken per execution.
    pub fn shots(&self) -> usize {
        self.shots
    }

    /// Create a quantum manager with simulator backend
    pub fn simulator(num_qubits: usize) -> Self {
        let device = QuantumDevice {
            backend: QuantumBackend::Simulator,
            num_qubits,
            connectivity: QuantumConnectivity::FullyConnected,
            noise_model: None,
            calibration: None,
        };
        Self::new(device)
    }

    /// Execute a quantum circuit
    pub fn execute_circuit(&mut self, circuit: &QuantumCircuit) -> Result<QuantumMeasurement> {
        // Validate circuit compatibility
        self.validate_circuit(circuit)?;

        // Optimize circuit if optimization is enabled
        let optimized_circuit = if self.optimization_enabled {
            self.optimize_circuit(circuit)?
        } else {
            circuit.clone()
        };

        // Execute on the specified backend
        match self.device.backend {
            QuantumBackend::Simulator => self.simulate_circuit(&optimized_circuit),
            _ => self.execute_on_real_device(&optimized_circuit),
        }
    }

    /// Create a quantum neural network layer
    pub fn create_qnn_layer(
        &self,
        input_qubits: usize,
        ansatz: QuantumAnsatz,
        parameters: &[f64],
    ) -> Result<QuantumNeuralLayer> {
        QuantumNeuralLayer::new(input_qubits, ansatz, parameters)
    }

    /// Create a quantum embedding layer
    pub fn create_embedding_layer(
        &self,
        classical_dim: usize,
        quantum_dim: usize,
        encoding: QuantumEncoding,
    ) -> Result<QuantumEmbeddingLayer> {
        QuantumEmbeddingLayer::new(classical_dim, quantum_dim, encoding)
    }

    /// Validate circuit compatibility with device
    fn validate_circuit(&self, circuit: &QuantumCircuit) -> Result<()> {
        if circuit.num_qubits > self.device.num_qubits {
            return Err(anyhow::anyhow!(
                "Circuit requires {} qubits, but device only has {}",
                circuit.num_qubits,
                self.device.num_qubits
            ));
        }

        // Check connectivity constraints
        match &self.device.connectivity {
            QuantumConnectivity::Linear => {
                // Validate linear connectivity
                for gate in &circuit.gates {
                    if let Some(qubits) = gate.target_qubits() {
                        if qubits.len() == 2 {
                            let diff = (qubits[0] as i32 - qubits[1] as i32).abs();
                            if diff != 1 {
                                return Err(anyhow::anyhow!(
                                    "Two-qubit gate between non-adjacent qubits: {} and {}",
                                    qubits[0],
                                    qubits[1]
                                ));
                            }
                        }
                    }
                }
            },
            QuantumConnectivity::Custom { edges } => {
                // Validate custom connectivity
                for gate in &circuit.gates {
                    if let Some(qubits) = gate.target_qubits() {
                        if qubits.len() == 2 {
                            let edge = (qubits[0].min(qubits[1]), qubits[0].max(qubits[1]));
                            if !edges.contains(&edge) {
                                return Err(anyhow::anyhow!(
                                    "Two-qubit gate on disconnected qubits: {} and {}",
                                    qubits[0],
                                    qubits[1]
                                ));
                            }
                        }
                    }
                }
            },
            _ => {}, // Fully connected or grid - assume valid
        }

        Ok(())
    }

    /// Optimize quantum circuit in-place
    fn optimize_circuit(&self, circuit: &QuantumCircuit) -> Result<QuantumCircuit> {
        if !self.optimization_enabled {
            return Ok(circuit.clone());
        }

        // Create a working copy that we'll optimize in-place
        let mut optimized_circuit = circuit.clone();

        // Apply in-place optimizations
        self.merge_single_qubit_gates_inplace(&mut optimized_circuit)?;
        self.cancel_inverse_gates_inplace(&mut optimized_circuit)?;
        self.decompose_multi_qubit_gates_inplace(&mut optimized_circuit)?;

        Ok(optimized_circuit)
    }

    /// Merge consecutive single-qubit gates in-place
    fn merge_single_qubit_gates_inplace(&self, circuit: &mut QuantumCircuit) -> Result<()> {
        use crate::quantum::quantum_ops::RotationGate;

        let mut i = 0;
        while i + 1 < circuit.gates.len() {
            // Check if consecutive gates are rotation gates on the same qubit
            if let (Some(gate1), Some(gate2)) = (
                self.try_extract_rotation_gate(circuit.gates[i].as_ref()),
                self.try_extract_rotation_gate(circuit.gates[i + 1].as_ref()),
            ) {
                if gate1.qubit == gate2.qubit && gate1.axis == gate2.axis {
                    // Merge the two rotation gates
                    let merged_angle = gate1.angle + gate2.angle;
                    let merged_gate = RotationGate {
                        qubit: gate1.qubit,
                        axis: gate1.axis,
                        angle: merged_angle,
                    };

                    // Replace first gate with merged gate, remove second gate
                    circuit.gates[i] = Box::new(merged_gate);
                    circuit.gates.remove(i + 1);
                    continue; // Don't increment i, check this position again
                }
            }
            i += 1;
        }
        Ok(())
    }

    /// Cancel inverse gate pairs in-place
    fn cancel_inverse_gates_inplace(&self, circuit: &mut QuantumCircuit) -> Result<()> {
        use crate::quantum::quantum_ops::EntanglingType;

        let mut i = 0;
        while i + 1 < circuit.gates.len() {
            let mut should_remove_pair = false;

            // Check if consecutive gates are inverses
            if let (Some(rot1), Some(rot2)) = (
                self.try_extract_rotation_gate(circuit.gates[i].as_ref()),
                self.try_extract_rotation_gate(circuit.gates[i + 1].as_ref()),
            ) {
                // Check if they're inverse rotations (same qubit, axis, opposite angles)
                if rot1.qubit == rot2.qubit
                    && rot1.axis == rot2.axis
                    && (rot1.angle + rot2.angle).abs() < 1e-10
                {
                    should_remove_pair = true;
                }
            } else if let (Some(ent1), Some(ent2)) = (
                self.try_extract_entangling_gate(circuit.gates[i].as_ref()),
                self.try_extract_entangling_gate(circuit.gates[i + 1].as_ref()),
            ) {
                // Check if they're the same self-inverse gate (CNOT, CZ)
                if ent1.control == ent2.control
                    && ent1.target == ent2.target
                    && matches!(ent1.gate_type, EntanglingType::CNOT | EntanglingType::CZ)
                    && ent1.gate_type == ent2.gate_type
                {
                    should_remove_pair = true;
                }
            }

            if should_remove_pair {
                // Remove both gates
                circuit.gates.remove(i + 1);
                circuit.gates.remove(i);
                continue; // Don't increment i, check this position again
            }

            i += 1;
        }
        Ok(())
    }

    /// Decompose multi-qubit gates for device constraints in-place
    fn decompose_multi_qubit_gates_inplace(&self, circuit: &mut QuantumCircuit) -> Result<()> {
        use crate::quantum::quantum_ops::{EntanglingGate, EntanglingType};

        match &self.device.connectivity {
            QuantumConnectivity::Linear => {
                let mut i = 0;
                while i < circuit.gates.len() {
                    if let Some(ent_gate) =
                        self.try_extract_entangling_gate(circuit.gates[i].as_ref())
                    {
                        // Check if this is a non-adjacent two-qubit gate
                        let qubit_diff = (ent_gate.control as i32 - ent_gate.target as i32).abs();
                        if qubit_diff > 1 && matches!(ent_gate.gate_type, EntanglingType::CNOT) {
                            // Decompose into adjacent CNOTs with SWAP gates
                            let start = ent_gate.control.min(ent_gate.target);
                            let end = ent_gate.control.max(ent_gate.target);
                            let is_control_first = ent_gate.control < ent_gate.target;

                            // Remove the original gate
                            circuit.gates.remove(i);

                            // Insert decomposed gates
                            let mut insert_pos = i;

                            // SWAP qubits to make them adjacent
                            for qubit in start..end {
                                let next_qubit = qubit + 1;
                                // SWAP gate decomposition: 3 CNOTs
                                circuit.gates.insert(
                                    insert_pos,
                                    Box::new(EntanglingGate {
                                        control: qubit,
                                        target: next_qubit,
                                        gate_type: EntanglingType::CNOT,
                                        parameters: vec![],
                                    }),
                                );
                                insert_pos += 1;

                                circuit.gates.insert(
                                    insert_pos,
                                    Box::new(EntanglingGate {
                                        control: next_qubit,
                                        target: qubit,
                                        gate_type: EntanglingType::CNOT,
                                        parameters: vec![],
                                    }),
                                );
                                insert_pos += 1;

                                circuit.gates.insert(
                                    insert_pos,
                                    Box::new(EntanglingGate {
                                        control: qubit,
                                        target: next_qubit,
                                        gate_type: EntanglingType::CNOT,
                                        parameters: vec![],
                                    }),
                                );
                                insert_pos += 1;
                            }

                            // Now add the actual CNOT (qubits are now adjacent)
                            let (actual_control, actual_target) =
                                if is_control_first { (end - 1, end) } else { (end, end - 1) };

                            circuit.gates.insert(
                                insert_pos,
                                Box::new(EntanglingGate {
                                    control: actual_control,
                                    target: actual_target,
                                    gate_type: EntanglingType::CNOT,
                                    parameters: vec![],
                                }),
                            );
                            insert_pos += 1;

                            // SWAP back to original positions
                            for qubit in (start..end).rev() {
                                let next_qubit = qubit + 1;
                                circuit.gates.insert(
                                    insert_pos,
                                    Box::new(EntanglingGate {
                                        control: qubit,
                                        target: next_qubit,
                                        gate_type: EntanglingType::CNOT,
                                        parameters: vec![],
                                    }),
                                );
                                insert_pos += 1;

                                circuit.gates.insert(
                                    insert_pos,
                                    Box::new(EntanglingGate {
                                        control: next_qubit,
                                        target: qubit,
                                        gate_type: EntanglingType::CNOT,
                                        parameters: vec![],
                                    }),
                                );
                                insert_pos += 1;

                                circuit.gates.insert(
                                    insert_pos,
                                    Box::new(EntanglingGate {
                                        control: qubit,
                                        target: next_qubit,
                                        gate_type: EntanglingType::CNOT,
                                        parameters: vec![],
                                    }),
                                );
                                insert_pos += 1;
                            }

                            // Continue from the new position
                            i = insert_pos;
                            continue;
                        }
                    }
                    i += 1;
                }
            },
            QuantumConnectivity::Custom { edges } => {
                // For custom connectivity, check each two-qubit gate
                let mut i = 0;
                while i < circuit.gates.len() {
                    if let Some(ent_gate) =
                        self.try_extract_entangling_gate(circuit.gates[i].as_ref())
                    {
                        let edge = (
                            ent_gate.control.min(ent_gate.target),
                            ent_gate.control.max(ent_gate.target),
                        );
                        if !edges.contains(&edge) {
                            // This gate operates on disconnected qubits, needs routing
                            // For now, we'll just skip optimization for such gates
                            // A full implementation would find a path and insert SWAPs
                        }
                    }
                    i += 1;
                }
            },
            _ => {
                // Fully connected or grid - no decomposition needed
            },
        }

        Ok(())
    }

    /// Helper to extract rotation gate information
    fn try_extract_rotation_gate(&self, gate: &dyn QuantumOperation) -> Option<RotationGate> {
        use crate::quantum::quantum_ops::{RotationAxis, RotationGate};
        // This is a simplified approach - in a real implementation, we'd need
        // a way to downcast or pattern match on the concrete gate type
        let name = gate.operation_name();
        if name.starts_with("RX") || name.starts_with("RY") || name.starts_with("RZ") {
            // Parse the rotation gate from its string representation
            // This is a workaround since we can't directly downcast trait objects
            if let Some(qubit_targets) = gate.target_qubits() {
                if qubit_targets.len() == 1 {
                    let qubit = qubit_targets[0];
                    // Extract axis and angle from name (format: "R{axis}({angle})_{qubit}")
                    if let Some(axis_char) = name.chars().nth(1) {
                        let axis = match axis_char {
                            'X' => RotationAxis::X,
                            'Y' => RotationAxis::Y,
                            'Z' => RotationAxis::Z,
                            _ => return None,
                        };

                        // Extract angle from parentheses
                        if let (Some(start), Some(end)) = (name.find('('), name.find(')')) {
                            if let Ok(angle) = name[start + 1..end].parse::<f64>() {
                                return Some(RotationGate { qubit, axis, angle });
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Helper to extract entangling gate information
    fn try_extract_entangling_gate(&self, gate: &dyn QuantumOperation) -> Option<EntanglingGate> {
        use crate::quantum::quantum_ops::{EntanglingGate, EntanglingType};
        let name = gate.operation_name();
        if let Some(qubit_targets) = gate.target_qubits() {
            if qubit_targets.len() == 2 {
                let control = qubit_targets[0];
                let target = qubit_targets[1];

                let gate_type = if name.starts_with("CNOT") {
                    EntanglingType::CNOT
                } else if name.starts_with("CZ") {
                    EntanglingType::CZ
                } else if name.starts_with("RZZ") {
                    EntanglingType::RZZ
                } else {
                    return None;
                };

                return Some(EntanglingGate {
                    control,
                    target,
                    gate_type,
                    parameters: vec![],
                });
            }
        }
        None
    }

    /// Merge consecutive single-qubit gates (deprecated - use in-place version)
    #[allow(dead_code)]
    fn merge_single_qubit_gates(&self, circuit: QuantumCircuit) -> Result<QuantumCircuit> {
        let mut optimized = circuit;
        self.merge_single_qubit_gates_inplace(&mut optimized)?;
        Ok(optimized)
    }

    /// Cancel inverse gate pairs (deprecated - use in-place version)
    #[allow(dead_code)]
    fn cancel_inverse_gates(&self, circuit: QuantumCircuit) -> Result<QuantumCircuit> {
        let mut optimized = circuit;
        self.cancel_inverse_gates_inplace(&mut optimized)?;
        Ok(optimized)
    }

    /// Decompose multi-qubit gates for device constraints (deprecated - use in-place version)
    #[allow(dead_code)]
    fn decompose_multi_qubit_gates(&self, circuit: QuantumCircuit) -> Result<QuantumCircuit> {
        let mut optimized = circuit;
        self.decompose_multi_qubit_gates_inplace(&mut optimized)?;
        Ok(optimized)
    }

    /// Simulate the circuit with a real state-vector simulator.
    ///
    /// Every gate is applied to the state vector through
    /// [`QuantumOperation::apply`], and the measurement counts are sampled from
    /// the resulting Born-rule probabilities. This replaces a heuristic that
    /// counted H/CNOT/rotation gates and synthesised counts with
    /// `rng.random_range(-0.1..0.1)` fudges — it never applied a unitary and
    /// its output bore no relation to the circuit's actual state.
    ///
    /// State-vector simulation is exponential in qubit count; circuits wider
    /// than [`Self::MAX_SIMULATED_QUBITS`] are rejected rather than
    /// approximated.
    fn simulate_circuit(&self, circuit: &QuantumCircuit) -> Result<QuantumMeasurement> {
        if circuit.num_qubits > Self::MAX_SIMULATED_QUBITS {
            return Err(anyhow::anyhow!(
                "state-vector simulation of {} qubits needs {} amplitudes; the limit is {} qubits",
                circuit.num_qubits,
                1u64 << circuit.num_qubits.min(63),
                Self::MAX_SIMULATED_QUBITS
            ));
        }

        // Apply every gate to the state vector.
        let mut state = QuantumState::zero_state(circuit.num_qubits);
        for gate in &circuit.gates {
            state = gate.apply(&state)?;
        }

        // Born rule: P(i) = |amplitude_i|^2.
        let mut probabilities_by_index: Vec<f64> =
            (0..state.amplitudes.len()).map(|index| state.probability(index)).collect();

        let total: f64 = probabilities_by_index.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return Err(anyhow::anyhow!(
                "circuit produced a zero-norm state; the gate set is not unitary"
            ));
        }
        // Renormalise against accumulated floating-point drift.
        for probability in &mut probabilities_by_index {
            *probability /= total;
        }

        // Sample `shots` measurements from that distribution.
        let shots = self.shots;
        let mut rng = thread_rng();
        let mut counts: HashMap<String, usize> = HashMap::new();

        for _ in 0..shots {
            let sample: f64 = rng.random_range(0.0..1.0);
            let mut cumulative = 0.0;
            let mut chosen = probabilities_by_index.len() - 1;
            for (index, probability) in probabilities_by_index.iter().enumerate() {
                cumulative += probability;
                if sample < cumulative {
                    chosen = index;
                    break;
                }
            }
            let bitstring = format!("{:0width$b}", chosen, width = circuit.num_qubits);
            *counts.entry(bitstring).or_insert(0) += 1;
        }

        // Report the exact probabilities alongside the sampled counts.
        let probabilities: HashMap<String, f64> = probabilities_by_index
            .iter()
            .enumerate()
            .filter(|(_, probability)| **probability > 0.0)
            .map(|(index, probability)| {
                (
                    format!("{:0width$b}", index, width = circuit.num_qubits),
                    *probability,
                )
            })
            .collect();

        Ok(QuantumMeasurement {
            counts,
            probabilities,
            shots,
        })
    }

    /// Execute the circuit on real quantum hardware.
    ///
    /// Not implemented: no quantum cloud client is linked. This previously
    /// printed "Executing on real quantum device: <backend>" and then ran the
    /// local simulator, so the caller was told hardware had run their circuit
    /// when it had not.
    fn execute_on_real_device(&self, _circuit: &QuantumCircuit) -> Result<QuantumMeasurement> {
        Err(anyhow::anyhow!(
            "execution on the {:?} backend: no quantum cloud client is linked into \
             trustformers-core. Use the local state-vector simulator instead.",
            self.device.backend
        ))
    }

    /// Get device information
    pub fn device_info(&self) -> &QuantumDevice {
        &self.device
    }

    /// Enable or disable circuit optimization
    pub fn set_optimization(&mut self, enabled: bool) {
        self.optimization_enabled = enabled;
    }

    /// Clear circuit cache
    pub fn clear_cache(&mut self) {
        self.circuit_cache.clear();
    }
}

impl Default for QuantumDevice {
    fn default() -> Self {
        Self {
            backend: QuantumBackend::Simulator,
            num_qubits: 4,
            connectivity: QuantumConnectivity::FullyConnected,
            noise_model: None,
            calibration: None,
        }
    }
}

impl Default for NoiseModel {
    fn default() -> Self {
        let mut gate_error_rates = HashMap::new();
        gate_error_rates.insert("X".to_string(), 0.001);
        gate_error_rates.insert("Y".to_string(), 0.001);
        gate_error_rates.insert("Z".to_string(), 0.001);
        gate_error_rates.insert("H".to_string(), 0.002);
        gate_error_rates.insert("CNOT".to_string(), 0.01);

        Self {
            gate_error_rates,
            readout_error: 0.02,
            decoherence_time: Some(100.0), // microseconds
            thermal_noise: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: `simulate_circuit` counted H/CNOT/rotation gates and
    /// synthesised counts with `rng.random_range(-0.1..0.1)` fudges, never
    /// applying a unitary. A real simulator must reproduce textbook results.
    #[test]
    fn test_simulation_applies_real_unitaries() {
        use crate::quantum::quantum_ops::{RotationAxis, RotationGate};

        let manager = QuantumManager::simulator(1);

        // RX(pi) takes |0> to |1> (up to global phase), so every shot is "1".
        let mut circuit = QuantumCircuit::new(1);
        circuit.gates.push(Box::new(RotationGate {
            qubit: 0,
            axis: RotationAxis::X,
            angle: std::f64::consts::PI,
        }));

        let measurement = manager.simulate_circuit(&circuit).expect("simulation failed");
        assert_eq!(measurement.shots, QuantumManager::DEFAULT_SHOTS);
        assert_eq!(
            measurement.counts.get("1").copied().unwrap_or(0),
            QuantumManager::DEFAULT_SHOTS,
            "RX(pi)|0> = |1>, so every shot must read 1: {:?}",
            measurement.counts
        );
        let probability_one = measurement.probabilities.get("1").copied().unwrap_or(0.0);
        assert!(
            (probability_one - 1.0).abs() < 1e-9,
            "P(|1>) must be 1, got {probability_one}"
        );

        // The identity circuit leaves |0>, so every shot is "0".
        let identity = QuantumCircuit::new(1);
        let measurement = manager.simulate_circuit(&identity).expect("simulation failed");
        assert_eq!(
            measurement.counts.get("0").copied().unwrap_or(0),
            QuantumManager::DEFAULT_SHOTS
        );
    }

    /// RX(pi/2) puts the qubit in an equal superposition, so the sampled counts
    /// must straddle 50/50 — a distribution the old heuristic could not produce
    /// from the circuit itself.
    #[test]
    fn test_simulation_samples_from_the_born_rule() {
        use crate::quantum::quantum_ops::{RotationAxis, RotationGate};

        let mut manager = QuantumManager::simulator(1);
        manager.set_shots(4096).expect("positive shot count");

        let mut circuit = QuantumCircuit::new(1);
        circuit.gates.push(Box::new(RotationGate {
            qubit: 0,
            axis: RotationAxis::X,
            angle: std::f64::consts::FRAC_PI_2,
        }));

        let measurement = manager.simulate_circuit(&circuit).expect("simulation failed");
        let zeros = measurement.counts.get("0").copied().unwrap_or(0) as f64;
        let total = measurement.shots as f64;
        assert!(
            (zeros / total - 0.5).abs() < 0.06,
            "an equal superposition must sample near 50/50, got {}",
            zeros / total
        );

        // The reported probabilities are exact, not sampled.
        let probability_zero = measurement.probabilities.get("0").copied().unwrap_or(0.0);
        assert!(
            (probability_zero - 0.5).abs() < 1e-9,
            "got {probability_zero}"
        );
    }

    /// A circuit too wide to simulate is refused, not approximated.
    #[test]
    fn test_oversized_circuit_is_refused() {
        let manager = QuantumManager::simulator(64);
        let circuit = QuantumCircuit::new(QuantumManager::MAX_SIMULATED_QUBITS + 1);
        let error = manager
            .simulate_circuit(&circuit)
            .expect_err("a 21-qubit state vector must not be attempted");
        assert!(error.to_string().contains("limit"), "unexpected: {error}");
    }

    /// Regression test: `execute_on_real_device` printed that it was using real
    /// hardware and then ran the simulator.
    #[test]
    fn test_real_device_execution_is_refused() {
        let manager = QuantumManager::simulator(2);
        let circuit = QuantumCircuit::new(2);
        let error = manager
            .execute_on_real_device(&circuit)
            .expect_err("no quantum cloud client is linked");
        assert!(
            error.to_string().contains("no quantum cloud client"),
            "unexpected: {error}"
        );
    }

    #[test]
    fn test_shot_count_must_be_positive() {
        let mut manager = QuantumManager::simulator(1);
        assert!(manager.set_shots(0).is_err());
        assert_eq!(manager.shots(), QuantumManager::DEFAULT_SHOTS);
        manager.set_shots(10).expect("positive");
        assert_eq!(manager.shots(), 10);
    }

    #[test]
    fn test_quantum_manager_creation() {
        let manager = QuantumManager::simulator(4);
        assert_eq!(manager.device.num_qubits, 4);
        assert_eq!(manager.device.backend, QuantumBackend::Simulator);
        assert!(manager.optimization_enabled);
    }

    #[test]
    fn test_quantum_device_default() {
        let device = QuantumDevice::default();
        assert_eq!(device.num_qubits, 4);
        assert_eq!(device.backend, QuantumBackend::Simulator);
        assert!(matches!(
            device.connectivity,
            QuantumConnectivity::FullyConnected
        ));
    }

    #[test]
    fn test_noise_model_default() {
        let noise = NoiseModel::default();
        assert_eq!(noise.readout_error, 0.02);
        assert!(noise.gate_error_rates.contains_key("CNOT"));
        assert_eq!(noise.gate_error_rates["CNOT"], 0.01);
        assert!(!noise.thermal_noise);
    }

    #[test]
    fn test_quantum_connectivity() {
        let linear = QuantumConnectivity::Linear;
        let grid = QuantumConnectivity::Grid { rows: 2, cols: 2 };
        let custom = QuantumConnectivity::Custom {
            edges: vec![(0, 1), (1, 2), (2, 3)],
        };

        // Test that different connectivity types can be created
        assert!(matches!(linear, QuantumConnectivity::Linear));
        assert!(matches!(grid, QuantumConnectivity::Grid { .. }));
        assert!(matches!(custom, QuantumConnectivity::Custom { .. }));
    }

    #[test]
    fn test_quantum_backends() {
        let backends = [
            QuantumBackend::Simulator,
            QuantumBackend::Qiskit,
            QuantumBackend::Cirq,
            QuantumBackend::PennyLane,
            QuantumBackend::Braket,
            QuantumBackend::IonQ,
            QuantumBackend::Rigetti,
        ];

        assert_eq!(backends.len(), 7);
        assert!(backends.contains(&QuantumBackend::Simulator));
        assert!(backends.contains(&QuantumBackend::IonQ));
    }

    #[test]
    fn test_device_calibration() {
        let mut gate_fidelities = HashMap::new();
        gate_fidelities.insert("X".to_string(), 0.999);
        gate_fidelities.insert("CNOT".to_string(), 0.995);

        let calibration = DeviceCalibration {
            gate_fidelities,
            qubit_frequencies: vec![5.0e9, 5.1e9, 4.9e9, 5.05e9],
            coupling_strengths: vec![0.02, 0.018, 0.022],
            timestamp: 1640995200, // Example timestamp
        };

        assert_eq!(calibration.qubit_frequencies.len(), 4);
        assert_eq!(calibration.coupling_strengths.len(), 3);
        assert_eq!(calibration.gate_fidelities["X"], 0.999);
    }
}
