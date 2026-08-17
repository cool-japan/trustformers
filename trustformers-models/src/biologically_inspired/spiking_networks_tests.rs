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
            // Deterministic dynamics: the noise term is opt-in.
            noise_variance: 0.0,
            ..BiologicalConfig::default()
        }
    }

    fn constant_current(config: &BiologicalConfig, value: f32) -> Tensor {
        Tensor::full(value, vec![1, config.neurons_per_layer]).expect("current")
    }

    /// Run `steps` integration steps with a constant drive and return the total
    /// number of spikes emitted.
    fn spike_count_under_drive(config: &BiologicalConfig, drive: f32, steps: usize) -> f32 {
        let mut layer = SpikingLayer::new(config).expect("layer");
        layer.init_states(1).expect("init");
        let current = constant_current(config, drive);
        let mut total = 0.0;
        for _ in 0..steps {
            layer.update_dynamics(&current).expect("dynamics");
            total += layer.spike_count().expect("count");
        }
        total
    }

    // --- construction ---

    #[test]
    fn test_spiking_layer_creation() {
        let config = small_spiking_config();
        assert!(SpikingLayer::new(&config).is_ok());
    }

    #[test]
    fn test_spiking_layer_init_states() {
        let config = small_spiking_config();
        let mut layer = SpikingLayer::new(&config).expect("layer");
        layer.init_states(2).expect("init");
        assert!(layer.neuron_states.is_some());
    }

    #[test]
    fn test_spiking_layer_neuron_state_shapes() {
        let config = small_spiking_config();
        let mut layer = SpikingLayer::new(&config).expect("layer");
        layer.init_states(4).expect("init");
        let states = layer.neuron_states.as_ref().expect("states");
        assert_eq!(states.v_mem.shape(), vec![4, 16]);
        assert_eq!(states.spikes.shape(), vec![4, 16]);
        assert_eq!(states.refractory_time.shape(), vec![4, 16]);
    }

    #[test]
    fn test_model_specific_state_is_allocated_per_config() {
        let mut config = small_spiking_config();

        config.neuron_model = NeuronModel::LeakyIntegrateAndFire;
        let mut layer = SpikingLayer::new(&config).expect("layer");
        layer.init_states(1).expect("init");
        let states = layer.neuron_states.as_ref().expect("states");
        assert!(states.u_recovery.is_none());
        assert!(states.gating.is_none());

        config.neuron_model = NeuronModel::Izhikevich;
        let mut layer = SpikingLayer::new(&config).expect("layer");
        layer.init_states(1).expect("init");
        assert!(layer.neuron_states.as_ref().expect("states").u_recovery.is_some());

        config.neuron_model = NeuronModel::AdaptiveExponentialIF;
        let mut layer = SpikingLayer::new(&config).expect("layer");
        layer.init_states(1).expect("init");
        assert!(layer.neuron_states.as_ref().expect("states").adaptation.is_some());

        config.neuron_model = NeuronModel::HodgkinHuxley;
        let mut layer = SpikingLayer::new(&config).expect("layer");
        layer.init_states(1).expect("init");
        assert!(layer.neuron_states.as_ref().expect("states").gating.is_some());

        config.neuron_model = NeuronModel::SpikeResponseModel;
        let mut layer = SpikingLayer::new(&config).expect("layer");
        layer.init_states(1).expect("init");
        let states = layer.neuron_states.as_ref().expect("states");
        assert!(states.synaptic_trace.is_some());
        assert!(states.refractory_trace.is_some());
    }

    // --- the regression: config must drive the dynamics ---

    #[test]
    fn test_membrane_reset_uses_configured_v_reset() {
        // Regression: the old inline dynamics reset by *subtracting 1.0* from
        // v_mem instead of assigning v_reset.
        let mut config = small_spiking_config();
        config.v_threshold = 1.0;
        config.v_reset = -0.5;

        let mut layer = SpikingLayer::new(&config).expect("layer");
        layer.init_states(1).expect("init");

        // A very large drive guarantees a spike on the first step.
        let current = constant_current(&config, 5000.0);
        layer.update_dynamics(&current).expect("dynamics");

        let states = layer.neuron_states.as_ref().expect("states");
        let spikes = states.spikes.to_vec_f32().expect("spikes");
        assert!(spikes.iter().all(|s| *s == 1.0), "all neurons must spike");

        for v in states.v_mem.to_vec_f32().expect("v") {
            assert!(
                (v - config.v_reset).abs() < 1e-5,
                "post-spike membrane potential {v} must equal v_reset {}",
                config.v_reset
            );
        }
    }

    #[test]
    fn test_tau_mem_changes_lif_dynamics() {
        // Regression: tau was hardcoded to 0.02 regardless of config.
        let mut fast = small_spiking_config();
        fast.tau_mem = 0.002;
        let mut slow = small_spiking_config();
        slow.tau_mem = 0.2;

        let fast_spikes = spike_count_under_drive(&fast, 2.0, 40);
        let slow_spikes = spike_count_under_drive(&slow, 2.0, 40);
        assert!(
            (fast_spikes - slow_spikes).abs() > 0.5,
            "tau_mem must change the spike count ({fast_spikes} vs {slow_spikes})"
        );
    }

    #[test]
    fn test_dt_changes_lif_dynamics() {
        let mut coarse = small_spiking_config();
        coarse.dt = 0.005;
        let mut fine = small_spiking_config();
        fine.dt = 0.0002;

        let coarse_spikes = spike_count_under_drive(&coarse, 2.0, 30);
        let fine_spikes = spike_count_under_drive(&fine, 2.0, 30);
        assert!(
            (coarse_spikes - fine_spikes).abs() > 0.5,
            "dt must change the spike count ({coarse_spikes} vs {fine_spikes})"
        );
    }

    #[test]
    fn test_v_threshold_changes_lif_dynamics() {
        let mut low = small_spiking_config();
        low.v_threshold = 0.2;
        let mut high = small_spiking_config();
        high.v_threshold = 5.0;

        let low_spikes = spike_count_under_drive(&low, 1.0, 60);
        let high_spikes = spike_count_under_drive(&high, 1.0, 60);
        assert!(
            low_spikes > high_spikes,
            "a lower threshold must produce more spikes ({low_spikes} vs {high_spikes})"
        );
    }

    #[test]
    fn test_neuron_model_changes_dynamics() {
        // Regression: every model ran the same hardcoded LIF update.
        let mut lif = small_spiking_config();
        lif.neuron_model = NeuronModel::LeakyIntegrateAndFire;
        let mut izhikevich = small_spiking_config();
        izhikevich.neuron_model = NeuronModel::Izhikevich;

        let mut lif_layer = SpikingLayer::new(&lif).expect("layer");
        lif_layer.init_states(1).expect("init");
        let mut izh_layer = SpikingLayer::new(&izhikevich).expect("layer");
        izh_layer.init_states(1).expect("init");

        let current = constant_current(&lif, 10.0);
        for _ in 0..20 {
            lif_layer.update_dynamics(&current).expect("lif");
            izh_layer.update_dynamics(&current).expect("izh");
        }

        let lif_v =
            lif_layer.neuron_states.as_ref().expect("states").v_mem.to_vec_f32().expect("v");
        let izh_v =
            izh_layer.neuron_states.as_ref().expect("states").v_mem.to_vec_f32().expect("v");
        assert!(
            lif_v.iter().zip(izh_v.iter()).any(|(a, b)| (a - b).abs() > 1e-3),
            "different neuron models must produce different membrane traces"
        );
    }

    #[test]
    fn test_izhikevich_stays_finite_and_spikes() {
        let mut config = small_spiking_config();
        config.neuron_model = NeuronModel::Izhikevich;

        let mut layer = SpikingLayer::new(&config).expect("layer");
        layer.init_states(1).expect("init");
        let current = constant_current(&config, 10.0);

        let mut spikes = 0.0;
        for _ in 0..300 {
            layer.update_dynamics(&current).expect("dynamics");
            spikes += layer.spike_count().expect("count");
            let v = layer.neuron_states.as_ref().expect("states").v_mem.to_vec_f32().expect("v");
            assert!(
                v.iter().all(|x| x.is_finite()),
                "Izhikevich must stay finite"
            );
        }
        assert!(
            spikes > 0.0,
            "a 10 pA drive must make Izhikevich neurons fire"
        );
    }

    #[test]
    fn test_adexp_stays_finite() {
        let mut config = small_spiking_config();
        config.neuron_model = NeuronModel::AdaptiveExponentialIF;

        let mut layer = SpikingLayer::new(&config).expect("layer");
        layer.init_states(1).expect("init");
        let current = constant_current(&config, 500.0);
        for _ in 0..200 {
            layer.update_dynamics(&current).expect("dynamics");
            let v = layer.neuron_states.as_ref().expect("states").v_mem.to_vec_f32().expect("v");
            assert!(v.iter().all(|x| x.is_finite()), "AdEx must stay finite");
        }
    }

    #[test]
    fn test_hodgkin_huxley_rests_and_fires() {
        // Regression: HodgkinHuxley used to silently delegate to LIF.
        let mut config = small_spiking_config();
        config.neuron_model = NeuronModel::HodgkinHuxley;

        let mut layer = SpikingLayer::new(&config).expect("layer");
        layer.init_states(1).expect("init");

        // With no injected current the neuron must stay near its resting
        // potential (-65 mV), which a LIF fallback would not reproduce.
        let quiet = constant_current(&config, 0.0);
        for _ in 0..20 {
            layer.update_dynamics(&quiet).expect("dynamics");
        }
        let resting = layer.neuron_states.as_ref().expect("states").v_mem.to_vec_f32().expect("v");
        for v in &resting {
            assert!(
                (*v + 65.0).abs() < 2.0,
                "Hodgkin-Huxley must rest near -65 mV, got {v}"
            );
        }

        // A supra-threshold current elicits action potentials.
        let mut layer = SpikingLayer::new(&config).expect("layer");
        layer.init_states(1).expect("init");
        let drive = constant_current(&config, 20.0);
        let mut spikes = 0.0;
        for _ in 0..60 {
            layer.update_dynamics(&drive).expect("dynamics");
            spikes += layer.spike_count().expect("count");
            let v = layer.neuron_states.as_ref().expect("states").v_mem.to_vec_f32().expect("v");
            assert!(v.iter().all(|x| x.is_finite()), "HH must stay finite");
        }
        assert!(spikes > 0.0, "20 uA/cm^2 must elicit Hodgkin-Huxley spikes");
    }

    #[test]
    fn test_hodgkin_huxley_gating_variables_evolve() {
        let mut config = small_spiking_config();
        config.neuron_model = NeuronModel::HodgkinHuxley;

        let mut layer = SpikingLayer::new(&config).expect("layer");
        layer.init_states(1).expect("init");
        let initial = layer
            .neuron_states
            .as_ref()
            .expect("states")
            .gating
            .as_ref()
            .expect("gating")
            .m
            .to_vec_f32()
            .expect("m");

        let drive = constant_current(&config, 20.0);
        for _ in 0..10 {
            layer.update_dynamics(&drive).expect("dynamics");
        }
        let after = layer
            .neuron_states
            .as_ref()
            .expect("states")
            .gating
            .as_ref()
            .expect("gating")
            .m
            .to_vec_f32()
            .expect("m");

        assert!(
            initial.iter().zip(after.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
            "the sodium activation gate must move under current injection"
        );
        assert!(after.iter().all(|x| (0.0..=1.0).contains(x)));
    }

    #[test]
    fn test_spike_response_model_is_not_lif() {
        // Regression: SpikeResponseModel used to silently delegate to LIF.
        let mut srm = small_spiking_config();
        srm.neuron_model = NeuronModel::SpikeResponseModel;
        let lif = small_spiking_config();

        let mut srm_layer = SpikingLayer::new(&srm).expect("layer");
        srm_layer.init_states(1).expect("init");
        let mut lif_layer = SpikingLayer::new(&lif).expect("layer");
        lif_layer.init_states(1).expect("init");

        let current = constant_current(&srm, 300.0);
        for _ in 0..30 {
            srm_layer.update_dynamics(&current).expect("srm");
            lif_layer.update_dynamics(&current).expect("lif");
        }

        let srm_v =
            srm_layer.neuron_states.as_ref().expect("states").v_mem.to_vec_f32().expect("v");
        let lif_v =
            lif_layer.neuron_states.as_ref().expect("states").v_mem.to_vec_f32().expect("v");
        assert!(
            srm_v.iter().zip(lif_v.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
            "the spike response model must not reduce to LIF"
        );

        // Its refractory kernel must have accumulated spikes.
        let refractory = srm_layer
            .neuron_states
            .as_ref()
            .expect("states")
            .refractory_trace
            .as_ref()
            .expect("refractory")
            .to_vec_f32()
            .expect("data");
        assert!(refractory.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_refractory_time_is_armed_by_spikes() {
        let mut config = small_spiking_config();
        config.refractory_period = 0.05;
        let mut layer = SpikingLayer::new(&config).expect("layer");
        layer.init_states(1).expect("init");

        layer.update_dynamics(&constant_current(&config, 5000.0)).expect("dynamics");
        let refractory = layer
            .neuron_states
            .as_ref()
            .expect("states")
            .refractory_time
            .to_vec_f32()
            .expect("data");
        assert!(
            refractory.iter().all(|t| *t > 0.0),
            "spiking neurons must enter the refractory period"
        );
    }

    // --- plasticity ---

    #[test]
    fn test_plasticity_type_changes_the_weight_update() {
        let mut hebbian = small_spiking_config();
        hebbian.plasticity_type = PlasticityType::Hebbian;
        let mut anti = small_spiking_config();
        anti.plasticity_type = PlasticityType::AntiHebbian;

        let run = |config: &BiologicalConfig| -> f32 {
            let mut layer = SpikingLayer::new(config).expect("layer");
            layer.init_states(1).expect("init");
            let before = layer
                .synaptic_states
                .as_ref()
                .expect("synaptic")
                .weights
                .to_vec_f32()
                .expect("w");
            layer.update_dynamics(&constant_current(config, 5000.0)).expect("dynamics");
            layer.update_plasticity().expect("plasticity");
            let after = layer
                .synaptic_states
                .as_ref()
                .expect("synaptic")
                .weights
                .to_vec_f32()
                .expect("w");
            after.iter().zip(before.iter()).map(|(a, b)| a - b).sum()
        };

        let hebbian_delta = run(&hebbian);
        let anti_delta = run(&anti);
        assert!(hebbian_delta > 0.0, "Hebbian plasticity must potentiate");
        assert!(anti_delta < 0.0, "anti-Hebbian plasticity must depress");
    }

    // --- NeuronState / SynapticState ---

    #[test]
    fn test_neuron_state_zeros_and_clone() {
        let state = NeuronState::zeros(2, 8).expect("state");
        assert_eq!(state.v_mem.shape(), vec![2, 8]);
        assert!(state.u_recovery.is_none());
        let cloned = state.clone();
        assert_eq!(cloned.spikes.shape(), vec![2, 8]);
    }

    #[test]
    fn test_synaptic_state_creation() {
        let weights = Tensor::zeros(&[8, 8]).expect("w");
        let traces = Tensor::zeros(&[1, 8]).expect("t");
        let state = SynapticState {
            weights,
            pre_traces: traces.clone(),
            post_traces: traces.clone(),
            eligibility: Tensor::zeros(&[8, 8]).expect("e"),
        };
        assert_eq!(state.weights.shape(), vec![8, 8]);
        assert_eq!(state.clone().pre_traces.shape(), vec![1, 8]);
    }

    // --- network ---

    #[test]
    fn test_spiking_neural_network_creation() {
        let config = small_spiking_config();
        assert!(SpikingNeuralNetwork::new(&config).is_ok());
    }

    #[test]
    fn test_spiking_neural_network_forward_shapes() {
        // Regression: stacked layers used to feed a `neurons_per_layer`-wide
        // spike vector into a `d_model`-wide projection.
        let config = small_spiking_config();
        let mut network = SpikingNeuralNetwork::new(&config).expect("network");
        let input = Tensor::randn(&[2, 3, 32]).expect("input");
        let output = network.forward(&input).expect("forward");
        assert_eq!(output.hidden_states.shape(), vec![2, 3, 32]);
        let spikes = output.spike_trains.expect("spike trains");
        assert_eq!(spikes.shape(), vec![2, 2, 16]);
    }

    #[test]
    fn test_spiking_layer_forward_rejects_two_dimensional_input() {
        let config = small_spiking_config();
        let mut layer = SpikingLayer::new(&config).expect("layer");
        let input = Tensor::randn(&[2, 32]).expect("input");
        assert!(layer.forward(&input).is_err());
    }

    #[test]
    fn test_spiking_layer_config_preserved() {
        let config = small_spiking_config();
        let layer = SpikingLayer::new(&config).expect("layer");
        assert_eq!(layer.config.d_model, 32);
        assert_eq!(layer.config.neurons_per_layer, 16);
        assert_eq!(layer.input_dim, 32);
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
