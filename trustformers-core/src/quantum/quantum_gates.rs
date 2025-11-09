//! Standard quantum gates and gate libraries

use crate::quantum::quantum_ops::*;

/// Standard single-qubit gates
pub struct StandardGates;

impl StandardGates {
    /// Pauli-X gate (bit flip)
    pub fn x(qubit: usize) -> RotationGate {
        RotationGate {
            qubit,
            axis: RotationAxis::X,
            angle: std::f64::consts::PI,
        }
    }

    /// Pauli-Y gate
    pub fn y(qubit: usize) -> RotationGate {
        RotationGate {
            qubit,
            axis: RotationAxis::Y,
            angle: std::f64::consts::PI,
        }
    }

    /// Pauli-Z gate (phase flip)
    pub fn z(qubit: usize) -> RotationGate {
        RotationGate {
            qubit,
            axis: RotationAxis::Z,
            angle: std::f64::consts::PI,
        }
    }

    /// Hadamard gate
    pub fn h(qubit: usize) -> Vec<RotationGate> {
        vec![
            RotationGate {
                qubit,
                axis: RotationAxis::Y,
                angle: std::f64::consts::PI / 2.0,
            },
            RotationGate {
                qubit,
                axis: RotationAxis::X,
                angle: std::f64::consts::PI,
            },
        ]
    }

    /// S gate (phase gate)
    pub fn s(qubit: usize) -> RotationGate {
        RotationGate {
            qubit,
            axis: RotationAxis::Z,
            angle: std::f64::consts::PI / 2.0,
        }
    }

    /// T gate
    pub fn t(qubit: usize) -> RotationGate {
        RotationGate {
            qubit,
            axis: RotationAxis::Z,
            angle: std::f64::consts::PI / 4.0,
        }
    }
}

/// Two-qubit gate library
pub struct TwoQubitGates;

impl TwoQubitGates {
    /// CNOT gate
    pub fn cnot(control: usize, target: usize) -> EntanglingGate {
        EntanglingGate {
            control,
            target,
            gate_type: EntanglingType::CNOT,
            parameters: vec![],
        }
    }

    /// CZ gate
    pub fn cz(control: usize, target: usize) -> EntanglingGate {
        EntanglingGate {
            control,
            target,
            gate_type: EntanglingType::CZ,
            parameters: vec![],
        }
    }

    /// RZZ gate (Ising coupling)
    pub fn rzz(control: usize, target: usize, angle: f64) -> EntanglingGate {
        EntanglingGate {
            control,
            target,
            gate_type: EntanglingType::RZZ,
            parameters: vec![angle],
        }
    }

    /// SWAP gate
    pub fn swap(qubit1: usize, qubit2: usize) -> Vec<EntanglingGate> {
        vec![
            Self::cnot(qubit1, qubit2),
            Self::cnot(qubit2, qubit1),
            Self::cnot(qubit1, qubit2),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_gates() {
        let x = StandardGates::x(0);
        assert_eq!(x.qubit, 0);
        assert!(matches!(x.axis, RotationAxis::X));
        assert!((x.angle - std::f64::consts::PI).abs() < 1e-10);

        let y = StandardGates::y(1);
        assert_eq!(y.qubit, 1);
        assert!(matches!(y.axis, RotationAxis::Y));

        let z = StandardGates::z(2);
        assert_eq!(z.qubit, 2);
        assert!(matches!(z.axis, RotationAxis::Z));
    }

    #[test]
    fn test_hadamard_gate() {
        let h_gates = StandardGates::h(0);
        assert_eq!(h_gates.len(), 2);
        assert_eq!(h_gates[0].qubit, 0);
        assert_eq!(h_gates[1].qubit, 0);
    }

    #[test]
    fn test_two_qubit_gates() {
        let cnot = TwoQubitGates::cnot(0, 1);
        assert_eq!(cnot.control, 0);
        assert_eq!(cnot.target, 1);
        assert!(matches!(cnot.gate_type, EntanglingType::CNOT));

        let cz = TwoQubitGates::cz(1, 2);
        assert_eq!(cz.control, 1);
        assert_eq!(cz.target, 2);
        assert!(matches!(cz.gate_type, EntanglingType::CZ));
    }

    #[test]
    fn test_parameterized_gates() {
        let rzz = TwoQubitGates::rzz(0, 1, 0.5);
        assert_eq!(rzz.parameters.len(), 1);
        assert!((rzz.parameters[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_swap_gate() {
        let swap_gates = TwoQubitGates::swap(0, 1);
        assert_eq!(swap_gates.len(), 3);
        // SWAP = CNOT(0,1) CNOT(1,0) CNOT(0,1)
        assert_eq!(swap_gates[0].control, 0);
        assert_eq!(swap_gates[0].target, 1);
        assert_eq!(swap_gates[1].control, 1);
        assert_eq!(swap_gates[1].target, 0);
        assert_eq!(swap_gates[2].control, 0);
        assert_eq!(swap_gates[2].target, 1);
    }
}
