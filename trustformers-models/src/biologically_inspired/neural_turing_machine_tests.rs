#[cfg(test)]
mod tests {
    use crate::biologically_inspired::config::{
        BiologicalArchitecture, BiologicalConfig, MemoryType,
    };
    use crate::biologically_inspired::neural_turing_machine::{
        circular_shift, content_addressing, sharpen, NTMLayer, NTMMemoryBank, NeuralTuringMachine,
    };
    use trustformers_core::tensor::Tensor;

    fn small_ntm_config() -> BiologicalConfig {
        BiologicalConfig {
            architecture: BiologicalArchitecture::NeuralTuringMachine,
            d_model: 8,
            n_layer: 2,
            vocab_size: 100,
            max_position_embeddings: 64,
            memory_capacity: 6,
            memory_type: MemoryType::Working,
            use_bias: true,
            ..BiologicalConfig::default()
        }
    }

    fn one_hot(batch: usize, capacity: usize, slot: usize) -> Tensor {
        let mut data = vec![0.0f32; batch * capacity];
        for row in 0..batch {
            data[row * capacity + slot] = 1.0;
        }
        Tensor::from_vec(data, &[batch, capacity]).expect("one-hot")
    }

    // --- addressing primitives ---

    #[test]
    fn test_content_addressing_peaks_at_matching_row() {
        // memory rows: 3 x 4
        let memory = vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
        ];
        let key = [0.0, 1.0, 0.0, 0.0];
        let weights = content_addressing(&key, 20.0, &memory, 3, 4);

        assert_eq!(weights.len(), 3);
        let total: f32 = weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
        assert!(
            weights[1] > weights[0] && weights[1] > weights[2],
            "cosine similarity must peak at the matching row: {weights:?}"
        );
        assert!(weights[1] > 0.9, "strength 20 must sharpen the peak");
    }

    #[test]
    fn test_content_addressing_is_scale_invariant() {
        let memory = vec![2.0, 0.0, 0.0, 4.0];
        let a = content_addressing(&[1.0, 0.0], 5.0, &memory, 2, 2);
        let b = content_addressing(&[7.0, 0.0], 5.0, &memory, 2, 2);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!(
                (x - y).abs() < 1e-5,
                "cosine addressing must ignore key norm"
            );
        }
    }

    #[test]
    fn test_circular_shift_rotates_left_and_right() {
        let weights = [1.0, 0.0, 0.0, 0.0];

        // s = [1, 0, 0] is the -1 tap: rotate left.
        let left = circular_shift(&weights, &[1.0, 0.0, 0.0]);
        assert_eq!(left, vec![0.0, 0.0, 0.0, 1.0]);

        // s = [0, 1, 0] is the identity tap.
        let identity = circular_shift(&weights, &[0.0, 1.0, 0.0]);
        assert_eq!(identity, vec![1.0, 0.0, 0.0, 0.0]);

        // s = [0, 0, 1] is the +1 tap: rotate right.
        let right = circular_shift(&weights, &[0.0, 0.0, 1.0]);
        assert_eq!(right, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_circular_shift_preserves_mass() {
        let weights = [0.1, 0.2, 0.3, 0.4];
        let shifted = circular_shift(&weights, &[0.25, 0.5, 0.25]);
        let total: f32 = shifted.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sharpening_concentrates_the_weighting() {
        let weights = [0.4, 0.3, 0.2, 0.1];
        let sharp = sharpen(&weights, 4.0);
        let total: f32 = sharp.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
        assert!(
            sharp[0] > weights[0],
            "gamma > 1 must increase the dominant weight"
        );
        assert!(sharp[3] < weights[3]);
    }

    // --- layer construction ---

    #[test]
    fn test_ntm_layer_creation() {
        let config = small_ntm_config();
        assert!(NTMLayer::new(&config).is_ok());
    }

    #[test]
    fn test_ntm_layer_config() {
        let config = small_ntm_config();
        let layer = NTMLayer::new(&config).expect("layer");
        assert_eq!(layer.config.d_model, 8);
        assert_eq!(layer.num_read_heads, 1);
        assert_eq!(layer.num_write_heads, 1);
        assert_eq!(layer.memory_width, 8);
    }

    #[test]
    fn test_ntm_layer_no_initial_memory() {
        let config = small_ntm_config();
        let layer = NTMLayer::new(&config).expect("layer");
        assert!(layer.memory_bank.is_none());
    }

    #[test]
    fn test_ntm_layer_controller_size() {
        let config = small_ntm_config();
        let layer = NTMLayer::new(&config).expect("layer");
        assert_eq!(layer.read_head_controllers.len(), 1);
        assert_eq!(layer.write_head_controllers.len(), 1);
        assert_eq!(layer.erase_head_controllers.len(), 1);
        assert_eq!(layer.add_head_controllers.len(), 1);
    }

    #[test]
    fn test_ntm_memory_bank_shapes() {
        let config = small_ntm_config();
        let mut layer = NTMLayer::new(&config).expect("layer");
        layer.init_memory(2).expect("init");
        let bank = layer.memory_bank.as_ref().expect("bank");
        assert_eq!(bank.memory.shape(), vec![2, 6, 8]);
        assert_eq!(bank.memory_size, (6, 8));
        assert_eq!(bank.read_heads[0].attention_weights.shape(), vec![2, 6]);
        assert_eq!(bank.read_heads[0].shift_weights.shape(), vec![2, 3]);
    }

    #[test]
    fn test_ntm_memory_bank_clone() {
        let config = small_ntm_config();
        let mut layer = NTMLayer::new(&config).expect("layer");
        layer.init_memory(1).expect("init");
        let bank: NTMMemoryBank = layer.memory_bank.as_ref().expect("bank").clone();
        assert_eq!(bank.memory_size, (6, 8));
        assert_eq!(bank.read_heads.len(), 1);
    }

    // --- read / write roundtrip (copy-task primitive) ---

    #[test]
    fn test_write_then_read_roundtrip_at_addressed_slot() {
        let config = small_ntm_config();
        let mut layer = NTMLayer::new(&config).expect("layer");
        layer.init_memory(1).expect("init");

        let content = Tensor::from_vec(vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.0, 3.0], &[1, 8])
            .expect("content");
        let slot = 4;
        layer.write_with_weights(&one_hot(1, 6, slot), &content).expect("write");
        layer.set_read_weights(0, one_hot(1, 6, slot)).expect("read weights");

        let read = layer.read_from_memory().expect("read");
        assert_eq!(read.len(), 1);
        let got = read[0].to_vec_f32().expect("data");
        let want = content.to_vec_f32().expect("data");
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "read {g} must match written {w}");
        }
    }

    #[test]
    fn test_reading_an_unwritten_slot_does_not_return_the_written_content() {
        let config = small_ntm_config();
        let mut layer = NTMLayer::new(&config).expect("layer");
        layer.init_memory(1).expect("init");

        let content = Tensor::full(9.0, vec![1, 8]).expect("content");
        layer.write_with_weights(&one_hot(1, 6, 2), &content).expect("write");
        layer.set_read_weights(0, one_hot(1, 6, 5)).expect("read weights");

        let read = layer.read_from_memory().expect("read");
        let got = read[0].to_vec_f32().expect("data");
        assert!(
            got.iter().all(|v| v.abs() < 1.0),
            "an unaddressed slot must not return the written content: {got:?}"
        );
    }

    #[test]
    fn test_content_weights_locate_a_written_row() {
        let config = small_ntm_config();
        let mut layer = NTMLayer::new(&config).expect("layer");
        layer.init_memory(1).expect("init");

        let content = Tensor::from_vec(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], &[1, 8])
            .expect("content");
        layer.write_with_weights(&one_hot(1, 6, 3), &content).expect("write");

        let weights = layer.content_weights(&content, 20.0).expect("content weights");
        let data = weights.to_vec_f32().expect("data");
        let best = data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(usize::MAX);
        assert_eq!(
            best, 3,
            "content addressing must find the written row: {data:?}"
        );
    }

    // --- the regression the audit reported ---

    #[test]
    fn test_forward_produces_non_uniform_attention() {
        // Regression: the addressing chain used to be overwritten with
        // `Tensor::ones(&[1, N]) / N`, so every head was exactly uniform.
        let config = small_ntm_config();
        let mut layer = NTMLayer::new(&config).expect("layer");
        let input = Tensor::randn(&[1, 3, 8]).expect("input");
        layer.forward(&input).expect("forward");

        let bank = layer.memory_bank.as_ref().expect("bank");
        let uniform = 1.0 / 6.0;
        for head in bank.read_heads.iter().chain(bank.write_heads.iter()) {
            let weights = head.attention_weights.to_vec_f32().expect("data");
            let total: f32 = weights.iter().sum();
            assert!((total - 1.0).abs() < 1e-4, "weights must be normalised");
            assert!(
                weights.iter().any(|w| (w - uniform).abs() > 1e-3),
                "attention must not collapse to uniform: {weights:?}"
            );
        }
    }

    #[test]
    fn test_different_inputs_produce_different_attention() {
        let config = small_ntm_config();

        let run = |seed: f32| -> Vec<f32> {
            let mut layer = NTMLayer::new(&config).expect("layer");
            layer.init_memory(1).expect("init");
            let input = Tensor::full(seed, vec![1, 2, 8]).expect("input");
            layer.forward(&input).expect("forward");
            layer.memory_bank.as_ref().expect("bank").write_heads[0]
                .attention_weights
                .to_vec_f32()
                .expect("data")
        };

        let a = run(1.0);
        let b = run(-1.0);
        assert!(
            a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-4),
            "attention must depend on the input: {a:?} vs {b:?}"
        );
    }

    #[test]
    fn test_forward_writes_change_memory() {
        let config = small_ntm_config();
        let mut layer = NTMLayer::new(&config).expect("layer");
        layer.init_memory(1).expect("init");
        let before = layer.get_memory().expect("memory").to_vec_f32().expect("data");

        let input = Tensor::randn(&[1, 2, 8]).expect("input");
        layer.forward(&input).expect("forward");
        let after = layer.get_memory().expect("memory").to_vec_f32().expect("data");

        assert!(
            before.iter().zip(after.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
            "the write head must modify memory"
        );
    }

    #[test]
    fn test_forward_preserves_sequence_shape() {
        let config = small_ntm_config();
        let mut layer = NTMLayer::new(&config).expect("layer");
        let input = Tensor::randn(&[2, 4, 8]).expect("input");
        let output = layer.forward(&input).expect("forward");
        assert_eq!(output.shape(), vec![2, 4, 8]);
    }

    #[test]
    fn test_forward_rejects_two_dimensional_input() {
        let config = small_ntm_config();
        let mut layer = NTMLayer::new(&config).expect("layer");
        let input = Tensor::randn(&[2, 8]).expect("input");
        assert!(layer.forward(&input).is_err());
    }

    // --- NeuralTuringMachine ---

    #[test]
    fn test_neural_turing_machine_creation() {
        let config = small_ntm_config();
        assert!(NeuralTuringMachine::new(&config).is_ok());
    }

    #[test]
    fn test_neural_turing_machine_forward() {
        let config = small_ntm_config();
        let mut ntm = NeuralTuringMachine::new(&config).expect("ntm");
        let input = Tensor::randn(&[1, 3, 8]).expect("input");
        let output = ntm.forward(&input).expect("forward");
        assert_eq!(output.hidden_states.shape(), vec![1, 3, 8]);
        assert!(output.memory_states.is_some());
    }

    // --- Config tests for NTM ---

    #[test]
    fn test_ntm_config_builder() {
        let config = BiologicalConfig::neural_turing_machine();
        assert!(matches!(
            config.architecture,
            BiologicalArchitecture::NeuralTuringMachine
        ));
        assert_eq!(config.memory_capacity, 128);
        assert!(matches!(config.memory_type, MemoryType::Working));
    }

    #[test]
    fn test_ntm_config_default_d_model() {
        let config = BiologicalConfig::neural_turing_machine();
        assert_eq!(config.d_model, 768);
    }

    #[test]
    fn test_ntm_config_custom_memory_capacity() {
        let config = BiologicalConfig {
            memory_capacity: 256,
            ..BiologicalConfig::neural_turing_machine()
        };
        assert_eq!(config.memory_capacity, 256);
    }

    #[test]
    fn test_ntm_layer_different_configs() {
        let configs = vec![
            BiologicalConfig {
                d_model: 16,
                memory_capacity: 8,
                use_bias: true,
                ..BiologicalConfig::neural_turing_machine()
            },
            BiologicalConfig {
                d_model: 64,
                memory_capacity: 32,
                use_bias: false,
                ..BiologicalConfig::neural_turing_machine()
            },
        ];
        for config in configs {
            assert!(NTMLayer::new(&config).is_ok());
        }
    }

    #[test]
    fn test_zero_memory_capacity_is_rejected() {
        let config = BiologicalConfig {
            memory_capacity: 0,
            ..small_ntm_config()
        };
        let mut layer = NTMLayer::new(&config).expect("layer");
        assert!(layer.init_memory(1).is_err());
    }
}
