#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_3, PI};

    use crate::quantum_classical_hybrids::config::{
        QuantumAnsatzConfig, QuantumClassicalConfig, QuantumHybridArchitecture,
    };
    use crate::quantum_classical_hybrids::quantum_cnn::{conv1d_windows, CONV_KERNEL_SIZE};
    use crate::quantum_classical_hybrids::quantum_optimizer::QuantumOptimizer;
    use crate::quantum_classical_hybrids::quantum_training::QuantumTrainingManager;
    use crate::quantum_classical_hybrids::statevector::{
        QuantumGate, StateVector, VariationalCircuit, MAX_SIMULATED_QUBITS,
    };
    use trustformers_core::tensor::Tensor;

    const TOL: f64 = 1e-9;

    fn tiny_config(num_qubits: usize, layers: usize, d_model: usize) -> QuantumClassicalConfig {
        QuantumClassicalConfig {
            architecture: QuantumHybridArchitecture::VariationalQuantumCircuit,
            d_model,
            n_classical_layers: 1,
            n_quantum_layers: 1,
            num_qubits,
            quantum_ansatz: QuantumAnsatzConfig::HardwareEfficient { layers },
            max_quantum_iterations: 5,
            quantum_learning_rate: 0.1,
            ..QuantumClassicalConfig::default()
        }
    }

    // --- analytic single- and two-qubit states ---

    #[test]
    fn test_zero_state_is_normalised_and_deterministic() {
        let state = StateVector::zero_state(3).expect("state");
        assert_eq!(state.amplitudes().len(), 8);
        assert!((state.probability(0) - 1.0).abs() < TOL);
        assert!((state.norm() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_hadamard_on_zero_gives_equal_amplitudes() {
        let mut state = StateVector::zero_state(1).expect("state");
        state.apply(&QuantumGate::Hadamard(0)).expect("H");

        let amps = state.amplitudes();
        assert!((amps[0].re - FRAC_1_SQRT_2).abs() < TOL, "{:?}", amps[0]);
        assert!((amps[1].re - FRAC_1_SQRT_2).abs() < TOL, "{:?}", amps[1]);
        assert!(amps[0].im.abs() < TOL);
        assert!(amps[1].im.abs() < TOL);
        assert!(
            state.expectation_z(0).expect("z").abs() < TOL,
            "H|0> has <Z> = 0"
        );
    }

    #[test]
    fn test_hadamard_is_its_own_inverse() {
        let mut state = StateVector::zero_state(2).expect("state");
        state.apply(&QuantumGate::Hadamard(1)).expect("H");
        state.apply(&QuantumGate::Hadamard(1)).expect("H");
        assert!((state.probability(0) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_bell_state_amplitudes_and_correlation() {
        // H on qubit 0 then CNOT(0 -> 1) prepares (|00> + |11>)/sqrt(2).
        let mut state = StateVector::zero_state(2).expect("state");
        state.apply(&QuantumGate::Hadamard(0)).expect("H");
        state
            .apply(&QuantumGate::ControlledNot {
                control: 0,
                target: 1,
            })
            .expect("CNOT");

        assert!((state.probability(0b00) - 0.5).abs() < TOL);
        assert!((state.probability(0b11) - 0.5).abs() < TOL);
        assert!(state.probability(0b01).abs() < TOL);
        assert!(state.probability(0b10).abs() < TOL);
        assert!((state.norm() - 1.0).abs() < TOL);

        // Both marginals are maximally mixed.
        assert!(state.expectation_z(0).expect("z0").abs() < TOL);
        assert!(state.expectation_z(1).expect("z1").abs() < TOL);
    }

    #[test]
    fn test_ry_expectation_is_cos_theta() {
        for &theta in &[0.0, FRAC_PI_3, FRAC_PI_2, PI, 2.0] {
            let mut state = StateVector::zero_state(1).expect("state");
            state
                .apply(&QuantumGate::RotationY {
                    qubit: 0,
                    angle: theta,
                })
                .expect("RY");
            let z = state.expectation_z(0).expect("z");
            assert!(
                (z - theta.cos()).abs() < 1e-9,
                "<Z> after RY({theta}) must be cos(theta): got {z}"
            );
        }
    }

    #[test]
    fn test_rx_expectation_is_cos_theta() {
        for &theta in &[0.0, FRAC_PI_3, FRAC_PI_2, PI] {
            let mut state = StateVector::zero_state(1).expect("state");
            state
                .apply(&QuantumGate::RotationX {
                    qubit: 0,
                    angle: theta,
                })
                .expect("RX");
            let z = state.expectation_z(0).expect("z");
            assert!(
                (z - theta.cos()).abs() < 1e-9,
                "<Z> after RX({theta}) must be cos(theta): got {z}"
            );
        }
    }

    #[test]
    fn test_rz_leaves_computational_basis_probabilities_unchanged() {
        let mut state = StateVector::zero_state(1).expect("state");
        state.apply(&QuantumGate::Hadamard(0)).expect("H");
        state
            .apply(&QuantumGate::RotationZ {
                qubit: 0,
                angle: FRAC_PI_3,
            })
            .expect("RZ");
        assert!((state.probability(0) - 0.5).abs() < TOL);
        assert!((state.probability(1) - 0.5).abs() < TOL);
        // But the relative phase moved.
        let amps = state.amplitudes();
        assert!(amps[0].im.abs() > 1e-6 || amps[1].im.abs() > 1e-6);
    }

    #[test]
    fn test_pauli_x_flips_the_qubit() {
        let mut state = StateVector::zero_state(2).expect("state");
        state.apply(&QuantumGate::PauliX(1)).expect("X");
        assert!((state.probability(0b10) - 1.0).abs() < TOL);
        assert!((state.expectation_z(1).expect("z") + 1.0).abs() < TOL);
        assert!((state.expectation_z(0).expect("z") - 1.0).abs() < TOL);
    }

    #[test]
    fn test_controlled_z_adds_a_sign_only_on_the_eleven_branch() {
        let mut state = StateVector::zero_state(2).expect("state");
        state.apply(&QuantumGate::Hadamard(0)).expect("H");
        state.apply(&QuantumGate::Hadamard(1)).expect("H");
        state
            .apply(&QuantumGate::ControlledZ {
                control: 0,
                target: 1,
            })
            .expect("CZ");
        let amps = state.amplitudes();
        assert!((amps[0b11].re + 0.5).abs() < TOL, "{:?}", amps[0b11]);
        assert!((amps[0b01].re - 0.5).abs() < TOL);
    }

    #[test]
    fn test_fidelity_of_identical_and_orthogonal_states() {
        let zero = StateVector::zero_state(1).expect("state");
        assert!((zero.fidelity(&zero).expect("f") - 1.0).abs() < TOL);

        let mut one = StateVector::zero_state(1).expect("state");
        one.apply(&QuantumGate::PauliX(0)).expect("X");
        assert!(zero.fidelity(&one).expect("f").abs() < TOL);
    }

    #[test]
    fn test_simulator_rejects_impossible_sizes() {
        assert!(StateVector::zero_state(0).is_err());
        assert!(StateVector::zero_state(MAX_SIMULATED_QUBITS + 1).is_err());
        let mut state = StateVector::zero_state(2).expect("state");
        assert!(state.apply(&QuantumGate::Hadamard(5)).is_err());
        assert!(state
            .apply(&QuantumGate::ControlledNot {
                control: 0,
                target: 0
            })
            .is_err());
    }

    // --- variational circuit + parameter shift ---

    #[test]
    fn test_circuit_parameter_count_and_shape() {
        let circuit = VariationalCircuit::new(3, 2).expect("circuit");
        assert_eq!(circuit.parameter_count(), 3 * 2 * 2);
        let state = circuit
            .run(&[0.1, 0.2, 0.3], &vec![0.4; circuit.parameter_count()])
            .expect("run");
        assert_eq!(state.num_qubits(), 3);
        assert!((state.norm() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_parameter_shift_matches_central_finite_difference() {
        // The parameter-shift rule uses +-pi/2 with denominator 2 and is exact
        // for Pauli rotations; a small central difference must agree with it.
        let circuit = VariationalCircuit::new(2, 2).expect("circuit");
        let encoding = [0.37, -0.62];
        let parameters: Vec<f64> =
            (0..circuit.parameter_count()).map(|i| 0.17 * (i as f64 + 1.0)).collect();

        let jacobian = circuit.expectation_jacobian(&encoding, &parameters).expect("jacobian");

        let epsilon = 1e-5;
        for k in 0..parameters.len() {
            let mut plus = parameters.clone();
            plus[k] += epsilon;
            let mut minus = parameters.clone();
            minus[k] -= epsilon;
            let z_plus = circuit.expectations(&encoding, &plus).expect("z+");
            let z_minus = circuit.expectations(&encoding, &minus).expect("z-");

            for qubit in 0..circuit.num_qubits() {
                let finite = (z_plus[qubit] - z_minus[qubit]) / (2.0 * epsilon);
                assert!(
                    (jacobian[qubit][k] - finite).abs() < 1e-5,
                    "parameter-shift {} vs finite difference {} (qubit {qubit}, param {k})",
                    jacobian[qubit][k],
                    finite
                );
            }
        }
    }

    #[test]
    fn test_parameter_shift_is_analytically_exact_for_a_single_rotation() {
        // One qubit, one layer: <Z> = cos(encoding + theta_0), so the exact
        // derivative with respect to theta_0 is -sin(encoding + theta_0).
        let circuit = VariationalCircuit::new(1, 1).expect("circuit");
        let encoding = [0.3];
        let parameters = [0.8, 0.0];
        let jacobian = circuit.expectation_jacobian(&encoding, &parameters).expect("jacobian");
        let expected = -(encoding[0] + parameters[0]).sin();
        assert!(
            (jacobian[0][0] - expected).abs() < 1e-9,
            "expected {expected}, got {}",
            jacobian[0][0]
        );
        // RZ does not change <Z>, so its derivative is exactly zero.
        assert!(jacobian[0][1].abs() < 1e-12);
    }

    #[test]
    fn test_circuit_rejects_wrong_argument_lengths() {
        let circuit = VariationalCircuit::new(2, 1).expect("circuit");
        assert!(circuit.run(&[0.1], &vec![0.0; circuit.parameter_count()]).is_err());
        assert!(circuit.run(&[0.1, 0.2], &[0.0]).is_err());
        assert!(VariationalCircuit::new(0, 1).is_err());
        assert!(VariationalCircuit::new(2, 0).is_err());
        assert!(VariationalCircuit::new(MAX_SIMULATED_QUBITS + 1, 1).is_err());
    }

    // --- QuantumOptimizer ---

    #[test]
    fn test_optimizer_forward_is_not_a_scalar_multiple_of_its_input() {
        // Regression: forward used to be `input * mean(parameters)`.
        let config = tiny_config(2, 1, 4);
        let optimizer = QuantumOptimizer::new(&config).expect("optimizer");
        let input = Tensor::from_vec(vec![0.5, -0.5, 1.0, -1.0], &[1, 1, 4]).expect("input");

        let output = optimizer.forward(&input).expect("forward");
        assert_eq!(output.shape(), vec![1, 1, 4]);

        let input_data = input.to_vec_f32().expect("in");
        let output_data = output.to_vec_f32().expect("out");
        let ratios: Vec<f32> = input_data
            .iter()
            .zip(output_data.iter())
            .filter(|(x, _)| x.abs() > 1e-6)
            .map(|(x, y)| y / x)
            .collect();
        assert!(
            ratios.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-4),
            "a real circuit output cannot be a single scalar multiple: {ratios:?}"
        );

        // Expectation values live in [-1, 1].
        assert!(output_data.iter().all(|z| (-1.0..=1.0).contains(z)));
    }

    #[test]
    fn test_optimizer_output_depends_on_every_parameter() {
        // Regression: parameter-shift gradients used to be identical for all
        // parameters because forward was linear in their mean.
        let config = tiny_config(2, 1, 4);
        let optimizer = QuantumOptimizer::new(&config).expect("optimizer");
        let input = Tensor::from_vec(vec![0.5, -0.5, 1.0, -1.0], &[1, 4]).expect("input");

        let gradients = optimizer.compute_gradients(&input).expect("gradients");
        assert_eq!(gradients.len(), optimizer.parameter_count());
        let first = gradients[0];
        assert!(
            gradients.iter().any(|g| (g - first).abs() > 1e-6),
            "gradients must differ per parameter: {gradients:?}"
        );
    }

    #[test]
    fn test_optimizer_gradients_match_finite_difference_of_the_loss() {
        let config = tiny_config(2, 1, 4);
        let mut optimizer = QuantumOptimizer::new(&config).expect("optimizer");
        for (index, parameter) in optimizer.parameters.iter_mut().enumerate() {
            *parameter = 0.23 * (index as f64 + 1.0);
        }
        let input = Tensor::from_vec(vec![0.5, -0.5, 1.0, -1.0], &[1, 4]).expect("input");

        let gradients = optimizer.compute_gradients(&input).expect("gradients");

        let epsilon = 1e-5;
        for k in 0..optimizer.parameters.len() {
            let mut plus = optimizer.clone();
            plus.parameters[k] += epsilon;
            let mut minus = optimizer.clone();
            minus.parameters[k] -= epsilon;
            let finite = (plus.compute_loss(&input).expect("loss+")
                - minus.compute_loss(&input).expect("loss-"))
                / (2.0 * epsilon);
            assert!(
                (gradients[k] - finite).abs() < 1e-4,
                "gradient {} vs finite difference {} for parameter {k}",
                gradients[k],
                finite
            );
        }
    }

    #[test]
    fn test_optimize_reduces_the_loss_and_changes_parameters() {
        let config = tiny_config(2, 1, 4);
        let mut optimizer = QuantumOptimizer::new(&config).expect("optimizer");
        let input = Tensor::from_vec(vec![0.9, 0.8, 0.7, 0.6], &[1, 4]).expect("input");

        let before_parameters = optimizer.parameters.clone();
        let before_loss = optimizer.compute_loss(&input).expect("loss");
        optimizer.optimize(&input).expect("optimize");
        let after_loss = optimizer.compute_loss(&input).expect("loss");

        assert!(
            optimizer
                .parameters
                .iter()
                .zip(before_parameters.iter())
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "optimization must move the parameters"
        );
        assert!(
            after_loss <= before_loss + 1e-9,
            "optimization must not increase the loss ({before_loss} -> {after_loss})"
        );
    }

    #[test]
    fn test_optimizer_rejects_too_narrow_features() {
        let config = tiny_config(4, 1, 2);
        let optimizer = QuantumOptimizer::new(&config).expect("optimizer");
        let input = Tensor::zeros(&[1, 2]).expect("input");
        assert!(optimizer.forward(&input).is_err());
    }

    #[test]
    fn test_optimizer_rejects_unsimulatable_qubit_counts() {
        let config = tiny_config(MAX_SIMULATED_QUBITS + 1, 1, 64);
        assert!(QuantumOptimizer::new(&config).is_err());
    }

    #[test]
    fn test_reference_fidelity_is_a_probability() {
        let config = tiny_config(2, 2, 4);
        let optimizer = QuantumOptimizer::new(&config).expect("optimizer");
        let fidelity = optimizer.reference_fidelity().expect("fidelity");
        assert!((0.0..=1.0).contains(&fidelity));

        // A circuit with all-zero angles leaves |0..0> untouched.
        let mut identity = QuantumOptimizer::new(&config).expect("optimizer");
        for parameter in identity.parameters.iter_mut() {
            *parameter = 0.0;
        }
        assert!((identity.reference_fidelity().expect("fidelity") - 1.0).abs() < 1e-9);
    }

    // --- training manager ---

    #[test]
    fn test_training_actually_updates_parameters() {
        // Regression: the trainer used to only record loss metrics.
        let mut config = tiny_config(2, 1, 3);
        config.hybrid_training_strategy =
            crate::quantum_classical_hybrids::config::HybridTrainingStrategy::Joint;
        let mut manager = QuantumTrainingManager::new(&config).expect("manager");

        let classical_before = manager.classical_parameters.clone();
        let quantum_before = manager.quantum_parameters.clone();

        let classical_gradients = vec![1.0f32, -2.0, 0.5];
        let quantum_gradients: Vec<f64> =
            (0..manager.quantum_parameters.len()).map(|i| 0.5 * (i as f64 + 1.0)).collect();

        let metrics = manager
            .train_epoch(&classical_gradients, &quantum_gradients)
            .expect("train epoch");

        assert!(
            manager
                .classical_parameters
                .iter()
                .zip(classical_before.iter())
                .any(|(a, b)| (a - b).abs() > 1e-12),
            "classical parameters must change"
        );
        assert!(
            manager
                .quantum_parameters
                .iter()
                .zip(quantum_before.iter())
                .any(|(a, b)| (a - b).abs() > 1e-12),
            "quantum parameters must change"
        );

        // Fidelity is a measured overlap, not `1 - noise_variance`.
        assert!((0.0..=1.0).contains(&metrics.quantum_fidelity));
        assert!(
            (metrics.quantum_fidelity - (1.0 - config.quantum_noise_variance)).abs() > 1e-9,
            "fidelity must not be the configured noise constant"
        );
        assert_eq!(manager.current_epoch, 1);
    }

    #[test]
    fn test_training_rejects_mismatched_gradient_lengths() {
        let config = tiny_config(2, 1, 3);
        let mut manager = QuantumTrainingManager::new(&config).expect("manager");
        assert!(manager.update_classical_parameters(&[1.0]).is_err());
        assert!(manager.update_quantum_parameters(&[1.0]).is_err());
    }

    #[test]
    fn test_zero_quantum_gradient_keeps_fidelity_at_one() {
        let config = tiny_config(2, 1, 3);
        let mut manager = QuantumTrainingManager::new(&config).expect("manager");
        let zeros = vec![0.0f64; manager.quantum_parameters.len()];
        manager.update_quantum_parameters(&zeros).expect("update");
        assert!((manager.training_metrics.quantum_fidelity - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_training_stats_reflect_history() {
        let config = tiny_config(2, 1, 3);
        let mut manager = QuantumTrainingManager::new(&config).expect("manager");
        let classical = vec![0.1f32; 3];
        let quantum = vec![0.1f64; manager.quantum_parameters.len()];
        for _ in 0..4 {
            manager.train_epoch(&classical, &quantum).expect("epoch");
        }
        let stats = manager.get_training_stats();
        assert_eq!(stats.total_epochs, 4);
        assert!(stats.avg_quantum_fidelity > 0.0);
    }

    // --- QuantumCNN convolution ---

    #[test]
    fn test_conv1d_windows_are_a_real_sliding_window() {
        // Regression: QuantumCNN used a plain Linear and called it convolution.
        let input =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, 2]).expect("input");
        let windows = conv1d_windows(&input, CONV_KERNEL_SIZE).expect("windows");
        assert_eq!(windows.shape(), vec![1, 3, 6]);

        let data = windows.to_vec_f32().expect("data");
        // t = 0: zero padding, then positions 0 and 1.
        assert_eq!(&data[0..6], &[0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
        // t = 1: positions 0, 1, 2.
        assert_eq!(&data[6..12], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // t = 2: positions 1, 2, then zero padding.
        assert_eq!(&data[12..18], &[3.0, 4.0, 5.0, 6.0, 0.0, 0.0]);
    }

    #[test]
    fn test_conv1d_windows_rejects_bad_arguments() {
        let input = Tensor::zeros(&[1, 3, 2]).expect("input");
        assert!(conv1d_windows(&input, 2).is_err());
        assert!(conv1d_windows(&input, 0).is_err());
        let flat = Tensor::zeros(&[3, 2]).expect("flat");
        assert!(conv1d_windows(&flat, 3).is_err());
    }
}
