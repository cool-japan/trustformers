#[cfg(test)]
mod tests {
    use crate::biologically_inspired::config::{
        BiologicalArchitecture, BiologicalConfig, NeuronModel, PlasticityType,
    };
    use crate::biologically_inspired::spiking_networks::*;
    use trustformers_core::tensor::Tensor;

    fn small_spiking_config() -> BiologicalConfig {
        BiologicalConfig {
            architecture: BiologicalArchitecture::SpikingNeuralNetwork,
            d_model: 32,
            n_layer: 2,
            vocab_size: 100,
            max_position_embeddings: 64,
            neuron_model: NeuronModel::LeakyIntegrateAndFire,
            neurons_per_layer: 16,
            use_bias: true,
            ..BiologicalConfig::default()
        }
    }

    // --- SpikingLayer tests ---

    #[test]
    fn test_spiking_layer_creation() {
        let config = small_spiking_config();
        let layer = SpikingLayer::new(&config);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_spiking_layer_init_states() {
        let config = small_spiking_config();
        if let Ok(mut layer) = SpikingLayer::new(&config) {
            let result = layer.init_states(2);
            assert!(result.is_ok());
            assert!(layer.neuron_states.is_some());
        }
    }

    #[test]
    fn test_spiking_layer_neuron_state_shapes() {
        let config = small_spiking_config();
        if let Ok(mut layer) = SpikingLayer::new(&config) {
            if layer.init_states(4).is_ok() {
                if let Some(ref states) = layer.neuron_states {
                    assert_eq!(states.v_mem.shape(), &[4, 16]);
                    assert_eq!(states.spikes.shape(), &[4, 16]);
                    assert_eq!(states.refractory_time.shape(), &[4, 16]);
                }
            }
        }
    }

    #[test]
    fn test_spiking_layer_lif_no_recovery() {
        let config = small_spiking_config();
        if let Ok(mut layer) = SpikingLayer::new(&config) {
            if layer.init_states(1).is_ok() {
                if let Some(ref states) = layer.neuron_states {
                    assert!(states.u_recovery.is_none());
                }
            }
        }
    }

    #[test]
    fn test_spiking_layer_izhikevich_has_recovery() {
        let mut config = small_spiking_config();
        config.neuron_model = NeuronModel::Izhikevich;
        if let Ok(mut layer) = SpikingLayer::new(&config) {
            if layer.init_states(1).is_ok() {
                if let Some(ref states) = layer.neuron_states {
                    assert!(states.u_recovery.is_some());
                }
            }
        }
    }

    #[test]
    fn test_spiking_layer_adexp_has_adaptation() {
        let mut config = small_spiking_config();
        config.neuron_model = NeuronModel::AdaptiveExponentialIF;
        if let Ok(mut layer) = SpikingLayer::new(&config) {
            if layer.init_states(1).is_ok() {
                if let Some(ref states) = layer.neuron_states {
                    assert!(states.adaptation.is_some());
                }
            }
        }
    }

    #[test]
    fn test_spiking_layer_config_preserved() {
        let config = small_spiking_config();
        if let Ok(layer) = SpikingLayer::new(&config) {
            assert_eq!(layer.config.d_model, 32);
            assert_eq!(layer.config.neurons_per_layer, 16);
        }
    }

    // --- NeuronState tests ---

    #[test]
    fn test_neuron_state_clone() {
        if let Ok(v_mem) = Tensor::zeros(&[2, 8]) {
            if let Ok(refractory) = Tensor::zeros(&[2, 8]) {
                if let Ok(spikes) = Tensor::zeros(&[2, 8]) {
                    let state = NeuronState {
                        v_mem: v_mem.clone(),
                        u_recovery: None,
                        adaptation: None,
                        refractory_time: refractory,
                        spikes,
                    };
                    let cloned = state.clone();
                    assert_eq!(cloned.v_mem.shape(), &[2, 8]);
                    assert!(cloned.u_recovery.is_none());
                }
            }
        }
    }

    // --- SynapticState tests ---

    #[test]
    fn test_synaptic_state_creation() {
        if let Ok(weights) = Tensor::zeros(&[8, 8]) {
            if let Ok(pre_traces) = Tensor::zeros(&[8, 8]) {
                if let Ok(post_traces) = Tensor::zeros(&[8, 8]) {
                    if let Ok(eligibility) = Tensor::zeros(&[8, 8]) {
                        let state = SynapticState {
                            weights,
                            pre_traces,
                            post_traces,
                            eligibility,
                        };
                        assert_eq!(state.weights.shape(), &[8, 8]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_synaptic_state_clone() {
        if let Ok(weights) = Tensor::zeros(&[4, 4]) {
            if let Ok(pre_traces) = Tensor::zeros(&[4, 4]) {
                if let Ok(post_traces) = Tensor::zeros(&[4, 4]) {
                    if let Ok(eligibility) = Tensor::zeros(&[4, 4]) {
                        let state = SynapticState {
                            weights,
                            pre_traces,
                            post_traces,
                            eligibility,
                        };
                        let cloned = state.clone();
                        assert_eq!(cloned.weights.shape(), &[4, 4]);
                    }
                }
            }
        }
    }

    // --- SpikingNeuralNetwork tests ---

    #[test]
    fn test_spiking_neural_network_creation() {
        let config = small_spiking_config();
        let snn = SpikingNeuralNetwork::new(&config);
        assert!(snn.is_ok());
    }

    // --- BiologicalConfig builder tests ---

    #[test]
    fn test_config_spiking_neural_network() {
        let config = BiologicalConfig::spiking_neural_network();
        assert!(matches!(
            config.architecture,
            BiologicalArchitecture::SpikingNeuralNetwork
        ));
        assert!(matches!(
            config.neuron_model,
            NeuronModel::LeakyIntegrateAndFire
        ));
    }

    #[test]
    fn test_config_hopfield_network() {
        let config = BiologicalConfig::hopfield_network();
        assert!(matches!(
            config.architecture,
            BiologicalArchitecture::HopfieldNetwork
        ));
        assert!(matches!(config.plasticity_type, PlasticityType::Hebbian));
    }

    #[test]
    fn test_config_liquid_time_constant() {
        let config = BiologicalConfig::liquid_time_constant();
        assert!(matches!(
            config.architecture,
            BiologicalArchitecture::LiquidTimeConstant
        ));
    }

    #[test]
    fn test_config_reservoir_computing() {
        let config = BiologicalConfig::reservoir_computing();
        assert!(matches!(
            config.architecture,
            BiologicalArchitecture::ReservoirComputing
        ));
        assert_eq!(config.reservoir_size, 1000);
    }

    #[test]
    fn test_config_capsule_network() {
        let config = BiologicalConfig::capsule_network();
        assert!(matches!(
            config.architecture,
            BiologicalArchitecture::CapsuleNetwork
        ));
        assert_eq!(config.num_capsules, 10);
        assert_eq!(config.capsule_dim, 16);
    }

    #[test]
    fn test_config_default_values() {
        let config = BiologicalConfig::default();
        assert_eq!(config.d_model, 768);
        assert_eq!(config.n_layer, 12);
        assert_eq!(config.vocab_size, 50000);
        assert!((config.dt - 0.001).abs() < f32::EPSILON);
        assert!((config.v_threshold - 1.0).abs() < f32::EPSILON);
        assert!((config.v_reset - 0.0).abs() < f32::EPSILON);
    }
}
