#[cfg(test)]
mod tests {
    use crate::biologically_inspired::config::{
        BiologicalArchitecture, BiologicalConfig, PlasticityType,
    };
    use crate::biologically_inspired::hopfield_networks::*;
    use trustformers_core::tensor::Tensor;

    /// Three mutually orthogonal Walsh patterns over `DIM` units.
    ///
    /// Orthogonality keeps the three patterns comfortably inside the classical
    /// Hopfield capacity (~0.14 N), so each one is a genuine attractor.
    const DIM: usize = 16;

    fn hopfield_config(d_model: usize, capacity: usize) -> BiologicalConfig {
        BiologicalConfig {
            architecture: BiologicalArchitecture::HopfieldNetwork,
            plasticity_type: PlasticityType::Hebbian,
            d_model,
            n_layer: 1,
            memory_capacity: capacity,
            neurons_per_layer: d_model,
            use_bias: false,
            ..BiologicalConfig::default()
        }
    }

    fn bipolar(values: &[f32]) -> Tensor {
        let len = values.len();
        Tensor::from_vec(values.to_vec(), &[1, len]).expect("failed to build bipolar pattern")
    }

    fn walsh(bit: usize) -> Tensor {
        let values: Vec<f32> =
            (0..DIM).map(|i| if (i >> bit) & 1 == 0 { 1.0 } else { -1.0 }).collect();
        bipolar(&values)
    }

    fn patterns() -> Vec<Tensor> {
        vec![walsh(3), walsh(2), walsh(1)]
    }

    /// Copy `pattern` and flip the sign of the units listed in `positions`.
    fn corrupt(pattern: &Tensor, positions: &[usize]) -> Tensor {
        let mut values = pattern.to_vec_f32().expect("pattern data");
        for &position in positions {
            values[position] = -values[position];
        }
        bipolar(&values)
    }

    fn stored_layer() -> HopfieldLayer {
        let config = hopfield_config(DIM, 4);
        let mut layer = HopfieldLayer::new(&config).expect("layer");
        layer.init_memory(1).expect("init");
        for pattern in &patterns() {
            layer.store_pattern(pattern).expect("store");
        }
        layer
    }

    #[test]
    fn test_store_pattern_actually_writes_the_slot() {
        // Regression: `store_pattern` used to compute everything into `_`-bound
        // locals and return Ok(()) without touching memory at all.
        let config = hopfield_config(DIM, 4);
        let mut layer = HopfieldLayer::new(&config).expect("layer");
        layer.init_memory(1).expect("init");

        let pattern = patterns()[0].clone();
        layer.store_pattern(&pattern).expect("store");

        assert_eq!(layer.stored_count(), 1);
        let state = layer.memory_state.as_ref().expect("memory");
        let row = state.patterns.select(0, 0).expect("row").to_vec_f32().expect("data");
        let expected = pattern.to_vec_f32().expect("expected");
        assert_eq!(row, expected, "stored pattern must be written into slot 0");
    }

    #[test]
    fn test_retrieve_pattern_returns_the_argmax_slot_not_slot_zero() {
        // Regression: retrieval used to discard the argmax and always return
        // memory row 0, so associative recall was constant.
        let mut layer = stored_layer();

        for (index, pattern) in patterns().iter().enumerate() {
            let matched = layer.best_match_indices(pattern).expect("argmax");
            assert_eq!(
                matched,
                vec![index],
                "query {index} must recall slot {index}"
            );

            let retrieved = layer.retrieve_pattern(pattern).expect("retrieve");
            assert_eq!(
                retrieved.to_vec_f32().expect("data"),
                pattern.to_vec_f32().expect("data"),
                "retrieval of pattern {index} must return that pattern"
            );
        }
    }

    #[test]
    fn test_retrieve_pattern_recovers_from_a_noisy_probe() {
        let mut layer = stored_layer();
        let probe = corrupt(&patterns()[2], &[0, 5]);
        assert_eq!(layer.best_match_indices(&probe).expect("argmax"), vec![2]);
        assert_eq!(
            layer.retrieve_pattern(&probe).expect("retrieve").to_vec_f32().expect("data"),
            patterns()[2].to_vec_f32().expect("data")
        );
    }

    #[test]
    fn test_retrieve_pattern_batches_independently() {
        let mut layer = stored_layer();
        let all = patterns();
        let batch = Tensor::concat(&[all[1].clone(), all[0].clone()], 0).expect("batch");
        let retrieved = layer.retrieve_pattern(&batch).expect("retrieve");
        assert_eq!(retrieved.shape(), vec![2, DIM]);

        let data = retrieved.to_vec_f32().expect("data");
        assert_eq!(&data[..DIM], &all[1].to_vec_f32().expect("data")[..]);
        assert_eq!(&data[DIM..], &all[0].to_vec_f32().expect("data")[..]);
    }

    #[test]
    fn test_stored_patterns_are_fixpoints_of_classical_dynamics() {
        let layer = stored_layer();

        for (index, pattern) in patterns().iter().enumerate() {
            let stepped = layer.classical_step(pattern).expect("step");
            assert_eq!(
                stepped.to_vec_f32().expect("data"),
                pattern.to_vec_f32().expect("data"),
                "stored pattern {index} must be a fixpoint of sign(W s)"
            );
        }
    }

    #[test]
    fn test_classical_dynamics_converge_from_a_noisy_probe() {
        let layer = stored_layer();
        let target = patterns()[0].clone();
        let probe = corrupt(&target, &[2, 11]);

        let converged = layer.run_classical_dynamics(&probe, 32).expect("dynamics");
        assert_eq!(
            converged.to_vec_f32().expect("data"),
            target.to_vec_f32().expect("data"),
            "a two-bit-corrupted probe must fall back into the stored attractor"
        );

        // And the converged state is genuinely a fixpoint of the update map.
        let again = layer.classical_step(&converged).expect("step");
        assert_eq!(
            again.to_vec_f32().expect("data"),
            converged.to_vec_f32().expect("data")
        );
    }

    #[test]
    fn test_hebbian_weights_have_zero_diagonal() {
        let config = hopfield_config(DIM, 4);
        let mut layer = HopfieldLayer::new(&config).expect("layer");
        layer.init_memory(1).expect("init");
        let pattern = patterns()[0].clone();
        layer.store_pattern(&pattern).expect("store");

        let weights = layer.memory_state.as_ref().expect("memory").weights.clone();
        assert_eq!(weights.shape(), vec![DIM, DIM]);
        let data = weights.to_vec_f32().expect("data");
        let values = pattern.to_vec_f32().expect("data");
        for i in 0..DIM {
            assert_eq!(data[i * DIM + i], 0.0, "diagonal element {i} must be zero");
            for j in 0..DIM {
                if i != j {
                    assert_eq!(
                        data[i * DIM + j],
                        values[i] * values[j],
                        "W[{i}][{j}] must be the Hebbian outer product"
                    );
                }
            }
        }
    }

    #[test]
    fn test_modern_hopfield_retrieval_converges_to_nearest_pattern() {
        let layer = stored_layer();
        let all = patterns();
        let probe = corrupt(&all[0], &[3]);

        // A high inverse temperature makes softmax(beta q Xᵀ) X ~ nearest pattern.
        let retrieved = layer.modern_retrieval(&probe, 8.0).expect("modern");
        let retrieved = retrieved.to_vec_f32().expect("data");
        let expected = all[0].to_vec_f32().expect("data");

        for (got, want) in retrieved.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-3,
                "modern retrieval {got} should approach {want}"
            );
        }
    }

    #[test]
    fn test_modern_retrieval_of_a_stored_pattern_is_a_fixpoint() {
        let layer = stored_layer();
        let all = patterns();

        let once = layer.modern_retrieval(&all[1], 16.0).expect("first");
        let twice = layer.modern_retrieval(&once, 16.0).expect("second");
        for (a, b) in once
            .to_vec_f32()
            .expect("data")
            .iter()
            .zip(twice.to_vec_f32().expect("data").iter())
        {
            assert!((a - b).abs() < 1e-4, "modern retrieval must be stable");
        }
    }

    #[test]
    fn test_replacement_policy_reuses_least_used_slot_when_full() {
        let config = hopfield_config(4, 2);
        let mut layer = HopfieldLayer::new(&config).expect("layer");
        layer.init_memory(1).expect("init");

        let a = bipolar(&[1.0, 1.0, -1.0, -1.0]);
        let b = bipolar(&[-1.0, 1.0, 1.0, -1.0]);
        let c = bipolar(&[1.0, -1.0, -1.0, 1.0]);

        layer.store_pattern(&a).expect("store a");
        layer.store_pattern(&b).expect("store b");
        assert_eq!(layer.stored_count(), 2);

        // Use slot 0 so slot 1 becomes the least-used slot.
        layer.retrieve_pattern(&a).expect("retrieve a");
        layer.store_pattern(&c).expect("store c");

        let state = layer.memory_state.as_ref().expect("memory");
        let slot0 = state.patterns.select(0, 0).expect("slot0").to_vec_f32().expect("d");
        let slot1 = state.patterns.select(0, 1).expect("slot1").to_vec_f32().expect("d");
        assert_eq!(slot0, a.to_vec_f32().expect("d"));
        assert_eq!(slot1, c.to_vec_f32().expect("d"));
    }

    #[test]
    fn test_forward_preserves_sequence_shape() {
        // Regression: `forward` used to concat timestep outputs along dim 1,
        // collapsing [batch, seq, d] into [batch, seq*d].
        let config = hopfield_config(DIM, 4);
        let mut layer = HopfieldLayer::new(&config).expect("layer");
        let input = Tensor::randn(&[2, 3, DIM]).expect("input");
        let output = layer.forward(&input).expect("forward");
        assert_eq!(output.shape(), vec![2, 3, DIM]);
    }

    #[test]
    fn test_forward_updates_memory_activations() {
        let config = hopfield_config(DIM, 4);
        let mut layer = HopfieldLayer::new(&config).expect("layer");
        let input = Tensor::randn(&[2, 1, DIM]).expect("input");
        layer.forward(&input).expect("forward");

        let state = layer.memory_state.as_ref().expect("memory");
        assert_eq!(state.activations.shape(), vec![2, 4]);
        let attention: f32 = state.activations.to_vec_f32().expect("data").iter().sum();
        // Two rows of softmax weights sum to 2.
        assert!((attention - 2.0).abs() < 1e-4);
        // The Hebbian trace must have picked up the presented pattern.
        let weight_energy: f32 =
            state.weights.to_vec_f32().expect("data").iter().map(|w| w.abs()).sum();
        assert!(weight_energy > 0.0, "forward must update the Hebbian trace");
    }

    #[test]
    fn test_network_forward_and_store_roundtrip() {
        let config = hopfield_config(DIM, 4);
        let mut network = HopfieldNetwork::new(&config).expect("network");
        let input = Tensor::randn(&[1, 2, DIM]).expect("input");
        let output = network.forward(&input).expect("forward");
        assert_eq!(output.hidden_states.shape(), vec![1, 2, DIM]);

        let all = patterns();
        network.store_patterns(&all).expect("store");
        let retrieved = network.retrieve_patterns(&[all[2].clone()]).expect("retrieve");
        assert_eq!(retrieved.len(), 1);
        assert_eq!(
            retrieved[0].to_vec_f32().expect("data"),
            all[2].to_vec_f32().expect("data")
        );
    }

    #[test]
    fn test_store_pattern_rejects_wrong_width() {
        let config = hopfield_config(DIM, 4);
        let mut layer = HopfieldLayer::new(&config).expect("layer");
        layer.init_memory(1).expect("init");
        let bad = Tensor::zeros(&[1, 5]).expect("bad");
        assert!(layer.store_pattern(&bad).is_err());
    }

    #[test]
    fn test_retrieve_without_memory_errors() {
        let config = hopfield_config(DIM, 4);
        let mut layer = HopfieldLayer::new(&config).expect("layer");
        let query = Tensor::zeros(&[1, DIM]).expect("query");
        assert!(layer.retrieve_pattern(&query).is_err());
    }
}
