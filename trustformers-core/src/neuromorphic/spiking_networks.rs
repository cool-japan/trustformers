//! Spiking Neural Network implementation

#![allow(unused_variables)] // Neuromorphic implementation with reserved parameters

use crate::neuromorphic::SpikeEvent;
use anyhow::Result;
use scirs2_core::random::*; // SciRS2 Policy compliant

/// Spiking neural network
#[derive(Debug, Clone)]
pub struct SpikingNeuralNetwork {
    pub neurons: Vec<SpikingNeuron>,
    pub synapses: Vec<Synapse>,
    pub topology: NetworkTopology,
    pub learning_rules: Vec<PlasticityRule>,
    pub power_gated: bool,
    /// Per-neuron BCM sliding thresholds, keyed by neuron id.
    ///
    /// Tracks the time-averaged squared postsynaptic activity, which is what
    /// makes BCM plasticity stable.
    pub bcm_thresholds: std::collections::HashMap<usize, f32>,
}

/// Individual spiking neuron
#[derive(Debug, Clone)]
pub struct SpikingNeuron {
    pub id: usize,
    pub neuron_type: NeuronType,
    pub membrane_potential: f32,
    pub threshold: f32,
    pub leak_rate: f32,
    pub refractory_period: f32,
    pub last_spike_time: f64,
    pub input_current: f32,
    /// Slow adaptation variable.
    ///
    /// The `w` current of the AdEx model and the `u` recovery variable of the
    /// Izhikevich model. Unused by LIF and Hodgkin-Huxley.
    pub adaptation: f32,
    /// Hodgkin-Huxley gating variables `(m, h, n)`.
    ///
    /// Only meaningful for [`NeuronType::HodgkinHuxley`].
    pub gating: (f32, f32, f32),
    /// Simulation time accumulated by [`SpikingNeuron::update`], in the same
    /// units as its `dt`. Used to enforce the refractory period.
    pub time: f64,
    /// Presynaptic spikes not yet due, as `(arrival_time, weight)`.
    ///
    /// Populated by [`SpikingNeuron::receive_spike`] with a non-zero delay.
    pub pending_spikes: Vec<(f64, f32)>,
}

/// Types of neurons
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuronType {
    LeakyIntegrateAndFire,
    AdaptiveExponential,
    Izhikevich,
    HodgkinHuxley,
}

/// Synaptic connection between neurons
#[derive(Debug, Clone)]
pub struct Synapse {
    pub pre_neuron: usize,
    pub post_neuron: usize,
    pub weight: f32,
    pub delay: f32,
    pub synapse_type: SynapseType,
    pub plasticity_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynapseType {
    Excitatory,
    Inhibitory,
    Modulatory,
}

/// Network topology patterns
#[derive(Debug, Clone)]
pub enum NetworkTopology {
    FullyConnected,
    Layered { layers: Vec<usize> },
    SmallWorld { rewiring_prob: f32 },
    ScaleFree { gamma: f32 },
    Custom { adjacency_matrix: Vec<Vec<bool>> },
}

/// Plasticity learning rules
#[derive(Debug, Clone)]
pub enum PlasticityRule {
    STDP {
        tau_plus: f32,
        tau_minus: f32,
        a_plus: f32,
        a_minus: f32,
    },
    BCM {
        theta: f32,
        tau: f32,
    },
    Homeostatic {
        target_rate: f32,
        alpha: f32,
    },
}

impl SpikingNeuralNetwork {
    /// Create a new spiking neural network
    pub fn new(num_neurons: usize) -> Self {
        let neurons = (0..num_neurons)
            .map(|id| SpikingNeuron::new(id, NeuronType::LeakyIntegrateAndFire))
            .collect();

        Self {
            neurons,
            synapses: Vec::new(),
            topology: NetworkTopology::FullyConnected,
            learning_rules: Vec::new(),
            power_gated: false,
            bcm_thresholds: std::collections::HashMap::new(),
        }
    }

    /// Add a synapse between neurons
    pub fn add_synapse(&mut self, pre: usize, post: usize, weight: f32, delay: f32) -> Result<()> {
        if pre >= self.neurons.len() || post >= self.neurons.len() {
            return Err(anyhow::anyhow!("Invalid neuron indices"));
        }

        let synapse = Synapse {
            pre_neuron: pre,
            post_neuron: post,
            weight,
            delay,
            synapse_type: if weight >= 0.0 {
                SynapseType::Excitatory
            } else {
                SynapseType::Inhibitory
            },
            plasticity_enabled: true,
        };

        self.synapses.push(synapse);
        Ok(())
    }

    /// Process one time step
    pub fn process_time_step(
        &mut self,
        input_spikes: &[SpikeEvent],
        dt: f64,
    ) -> Result<Vec<SpikeEvent>> {
        let mut output_spikes = Vec::new();

        // Apply input currents from spikes
        for spike in input_spikes {
            if spike.neuron_id < self.neurons.len() {
                self.neurons[spike.neuron_id].input_current += spike.weight;
            }
        }

        // Update each neuron and collect spikes
        let mut spiked_neurons = Vec::new();
        for neuron in &mut self.neurons {
            if neuron.update(dt) {
                // Neuron spiked
                output_spikes.push(SpikeEvent::new(neuron.id, 0.0, 1.0));
                spiked_neurons.push(neuron.id);
            }
        }

        // Propagate spikes through synapses
        for neuron_id in spiked_neurons {
            for synapse in &self.synapses {
                if synapse.pre_neuron == neuron_id {
                    if let Some(post_neuron) = self.neurons.get_mut(synapse.post_neuron) {
                        post_neuron.receive_spike(synapse.weight, synapse.delay);
                    }
                }
            }
        }

        // Apply plasticity rules
        self.apply_plasticity(&output_spikes, dt);

        // Reset input currents
        for neuron in &mut self.neurons {
            neuron.input_current = 0.0;
        }

        Ok(output_spikes)
    }

    /// Apply plasticity learning rules
    fn apply_plasticity(&mut self, spikes: &[SpikeEvent], dt: f64) {
        if self.learning_rules.is_empty() {
            return;
        }

        // Collect rules to avoid borrowing conflicts
        let rules = self.learning_rules.clone();
        for rule in &rules {
            match rule {
                PlasticityRule::STDP {
                    tau_plus,
                    tau_minus,
                    a_plus,
                    a_minus,
                } => {
                    self.apply_stdp(*tau_plus, *tau_minus, *a_plus, *a_minus, spikes, dt);
                },
                PlasticityRule::BCM { theta, tau } => {
                    self.apply_bcm(*theta, *tau, spikes, dt);
                },
                PlasticityRule::Homeostatic { target_rate, alpha } => {
                    self.apply_homeostatic(*target_rate, *alpha, spikes, dt);
                },
            }
        }
    }

    fn apply_stdp(
        &mut self,
        tau_plus: f32,
        tau_minus: f32,
        a_plus: f32,
        a_minus: f32,
        spikes: &[SpikeEvent],
        dt: f64,
    ) {
        // Simplified STDP implementation
        for spike in spikes {
            let pre_neuron_id = spike.neuron_id;

            for synapse in &mut self.synapses {
                if synapse.plasticity_enabled && synapse.pre_neuron == pre_neuron_id {
                    let post_neuron = &self.neurons[synapse.post_neuron];
                    let delta_t = spike.timestamp - post_neuron.last_spike_time;

                    if delta_t > 0.0 {
                        // Pre before post - potentiation
                        synapse.weight += a_plus * (-delta_t as f32 / tau_plus).exp();
                    } else {
                        // Post before pre - depression
                        synapse.weight -= a_minus * (delta_t as f32 / tau_minus).exp();
                    }

                    // Clip weights
                    synapse.weight = synapse.weight.clamp(-1.0, 1.0);
                }
            }
        }
    }

    /// Bienenstock-Cooper-Munro plasticity.
    ///
    /// `dw/dt = phi(y, theta) * x / tau` with `phi(y, theta) = y (y - theta)`,
    /// and the sliding threshold `theta` tracking the time-averaged squared
    /// postsynaptic activity: `d theta/dt = (y^2 - theta) / tau_theta`. The
    /// sliding threshold is what makes BCM stable — omitting it (as this used
    /// to) leaves plain Hebbian growth under a BCM label.
    fn apply_bcm(&mut self, theta: f32, tau: f32, spikes: &[SpikeEvent], dt: f64) {
        const THETA_TAU: f32 = 100.0;

        // Update the sliding threshold from the observed activity first.
        for spike in spikes {
            let activity = self.neurons[spike.neuron_id].membrane_potential;
            let threshold = self.bcm_thresholds.entry(spike.neuron_id).or_insert(theta);
            *threshold += dt as f32 * (activity * activity - *threshold) / THETA_TAU;
        }

        for spike in spikes {
            // BCM plasticity based on postsynaptic activity
            let neuron = &self.neurons[spike.neuron_id];
            let activity = neuron.membrane_potential;
            // Use this neuron's own sliding threshold, not the fixed initial one.
            let threshold = self.bcm_thresholds.get(&spike.neuron_id).copied().unwrap_or(theta);

            for synapse in &mut self.synapses {
                if synapse.plasticity_enabled && synapse.post_neuron == spike.neuron_id {
                    let delta_w =
                        activity * (activity - threshold) * synapse.weight * dt as f32 / tau;
                    synapse.weight += delta_w;
                    synapse.weight = synapse.weight.clamp(-1.0, 1.0);
                }
            }
        }
    }

    fn apply_homeostatic(&mut self, target_rate: f32, alpha: f32, spikes: &[SpikeEvent], dt: f64) {
        // Homeostatic plasticity to maintain target firing rate
        let current_rate = spikes.len() as f32 / (self.neurons.len() as f32 * dt as f32);
        let rate_error = target_rate - current_rate;

        for synapse in &mut self.synapses {
            if synapse.plasticity_enabled {
                synapse.weight += alpha * rate_error * dt as f32;
                synapse.weight = synapse.weight.clamp(-1.0, 1.0);
            }
        }
    }

    /// Set network topology
    pub fn set_topology(&mut self, topology: NetworkTopology) {
        // Clone the topology data we need before assigning
        match &topology {
            NetworkTopology::Layered { layers } => {
                let layers_clone = layers.clone();
                self.topology = topology;
                self.create_layered_connections(&layers_clone);
            },
            NetworkTopology::SmallWorld { rewiring_prob } => {
                let rewiring_prob_val = *rewiring_prob;
                self.topology = topology;
                self.create_small_world_connections(rewiring_prob_val);
            },
            _ => {
                self.topology = topology;
            },
        }
    }

    fn create_layered_connections(&mut self, layers: &[usize]) {
        self.synapses.clear();
        let mut neuron_idx = 0;
        let mut rng = thread_rng();

        for i in 0..layers.len() - 1 {
            let current_layer_size = layers[i];
            let next_layer_size = layers[i + 1];

            for current in 0..current_layer_size {
                for next in 0..next_layer_size {
                    let pre = neuron_idx + current;
                    let post = neuron_idx + current_layer_size + next;
                    let weight = (rng.random::<f32>() - 0.5) * 2.0; // Random weight [-1, 1]
                    let _ = self.add_synapse(pre, post, weight, 1.0);
                }
            }
            neuron_idx += current_layer_size;
        }
    }

    fn create_small_world_connections(&mut self, rewiring_prob: f32) {
        // Simplified small-world network creation
        self.synapses.clear();
        let n = self.neurons.len();
        let mut rng = thread_rng();

        // Create ring lattice
        for i in 0..n {
            let next = (i + 1) % n;
            let weight = (rng.random::<f32>() - 0.5) * 2.0;
            let _ = self.add_synapse(i, next, weight, 1.0);
        }

        // Rewire some connections
        let mut synapses_to_rewire = Vec::new();
        for (idx, synapse) in self.synapses.iter().enumerate() {
            if rng.random::<f32>() < rewiring_prob {
                synapses_to_rewire.push(idx);
            }
        }

        for idx in synapses_to_rewire {
            let new_target = rng.random_range(0..n);
            self.synapses[idx].post_neuron = new_target;
        }
    }

    /// Add plasticity rule
    pub fn add_plasticity_rule(&mut self, rule: PlasticityRule) {
        self.learning_rules.push(rule);
    }

    /// Enable power gating
    pub fn enable_power_gating(&mut self) {
        self.power_gated = true;
    }

    /// Adjust firing thresholds
    pub fn adjust_firing_thresholds(&mut self, factor: f32) {
        for neuron in &mut self.neurons {
            neuron.threshold *= factor;
        }
    }

    /// Get network statistics
    pub fn get_statistics(&self) -> NetworkStatistics {
        let total_synapses = self.synapses.len();
        let excitatory_synapses = self
            .synapses
            .iter()
            .filter(|s| matches!(s.synapse_type, SynapseType::Excitatory))
            .count();
        let inhibitory_synapses = total_synapses - excitatory_synapses;

        let average_weight = if total_synapses > 0 {
            self.synapses.iter().map(|s| s.weight).sum::<f32>() / total_synapses as f32
        } else {
            0.0
        };

        NetworkStatistics {
            num_neurons: self.neurons.len(),
            num_synapses: total_synapses,
            excitatory_synapses,
            inhibitory_synapses,
            average_weight,
            plasticity_enabled: !self.learning_rules.is_empty(),
        }
    }
}

impl SpikingNeuron {
    /// Create a new spiking neuron
    pub fn new(id: usize, neuron_type: NeuronType) -> Self {
        Self {
            id,
            neuron_type,
            membrane_potential: 0.0,
            threshold: 1.0,
            leak_rate: 0.1,
            refractory_period: 2.0,
            last_spike_time: -100.0,
            input_current: 0.0,
            adaptation: 0.0,
            // Resting-state HH gating values at V = -65 mV. `n` is written
            // with extra digits so it is not mistaken for 1/pi.
            gating: (0.053, 0.596, 0.3177),
            time: 0.0,
            pending_spikes: Vec::new(),
        }
    }

    /// Update neuron state and return true if spiked
    pub fn update(&mut self, dt: f64) -> bool {
        // Advance the clock once, here, so every model shares one timeline and
        // delayed spikes are delivered on schedule regardless of model.
        self.time += dt;
        self.deliver_due_spikes();
        match self.neuron_type {
            NeuronType::LeakyIntegrateAndFire => self.update_lif(dt),
            NeuronType::AdaptiveExponential => self.update_aeif(dt),
            NeuronType::Izhikevich => self.update_izhikevich(dt),
            NeuronType::HodgkinHuxley => self.update_hh(dt),
        }
    }

    fn update_lif(&mut self, dt: f64) -> bool {
        // Leaky Integrate-and-Fire model
        let dt_f32 = dt as f32;

        // Update membrane potential
        self.membrane_potential +=
            dt_f32 * (-self.leak_rate * self.membrane_potential + self.input_current);

        // Consume the accumulated input so it does not drive the next step too.
        self.input_current = 0.0;

        // Check for spike
        if self.membrane_potential >= self.threshold {
            self.membrane_potential = 0.0; // Reset
            self.last_spike_time = self.time;
            return true;
        }

        false
    }

    /// Adaptive Exponential Integrate-and-Fire (Brette & Gerstner, 2005).
    ///
    /// `C dV/dt = -g_L (V - E_L) + g_L * Δ_T * exp((V - V_T)/Δ_T) - w + I`
    /// `τ_w dw/dt = a (V - E_L) - w`
    ///
    /// Working in the module's normalised units (rest at 0, threshold at
    /// `self.threshold`), with `Δ_T` a fixed sharpness and `g_L` taken from
    /// `leak_rate`. The exponential term is what makes this different from LIF:
    /// the upswing is self-accelerating rather than linear.
    fn update_aeif(&mut self, dt: f64) -> bool {
        const SHARPNESS: f32 = 0.2; // Δ_T
        const ADAPT_COUPLING: f32 = 0.05; // a
        const ADAPT_TAU: f32 = 20.0; // τ_w
        const ADAPT_JUMP: f32 = 0.15; // b

        let dt = dt as f32;

        if self.in_refractory() {
            self.input_current = 0.0;
            return false;
        }

        // Soft threshold for the exponential term, below the hard threshold.
        let soft_threshold = self.threshold * 0.7;
        let exponential =
            SHARPNESS * ((self.membrane_potential - soft_threshold) / SHARPNESS).exp();
        // Cap the exponential so a large potential cannot produce a non-finite step.
        let exponential = exponential.min(self.threshold * 10.0);

        let dv = -self.leak_rate * self.membrane_potential + exponential - self.adaptation
            + self.input_current;
        self.membrane_potential += dt * dv;

        let dw = (ADAPT_COUPLING * self.membrane_potential - self.adaptation) / ADAPT_TAU;
        self.adaptation += dt * dw;

        self.input_current = 0.0;

        if self.membrane_potential >= self.threshold {
            self.membrane_potential = 0.0;
            // Spike-triggered adaptation: this is what makes AdEx adapt.
            self.adaptation += ADAPT_JUMP;
            self.last_spike_time = self.time;
            return true;
        }

        false
    }

    /// Izhikevich model (2003), regular-spiking parameterisation.
    ///
    /// `dv/dt = 0.04 v^2 + 5 v + 140 - u + I`
    /// `du/dt = a (b v - u)`, with reset `v <- c`, `u <- u + d` on spike.
    ///
    /// Uses the model's own millivolt scale, so `membrane_potential` is in mV
    /// here and the spike threshold is the model's fixed +30 mV cutoff.
    fn update_izhikevich(&mut self, dt: f64) -> bool {
        const A: f32 = 0.02;
        const B: f32 = 0.2;
        const C: f32 = -65.0;
        const D: f32 = 8.0;
        const PEAK_MV: f32 = 30.0;

        let dt = dt as f32;

        let v = self.membrane_potential;
        let u = self.adaptation;

        // Sub-stepping keeps the quadratic term stable for larger dt.
        let steps = ((dt / 0.5).ceil() as usize).max(1);
        let step_dt = dt / steps as f32;
        let mut v = v;
        let mut u = u;
        for _ in 0..steps {
            v += step_dt * (0.04 * v * v + 5.0 * v + 140.0 - u + self.input_current);
            u += step_dt * (A * (B * v - u));
            if v >= PEAK_MV {
                break;
            }
        }

        self.input_current = 0.0;

        if v >= PEAK_MV {
            self.membrane_potential = C;
            self.adaptation = u + D;
            self.last_spike_time = self.time;
            return true;
        }

        self.membrane_potential = v;
        self.adaptation = u;
        false
    }

    /// Hodgkin-Huxley model (1952), standard squid-axon parameters.
    ///
    /// Integrates the membrane equation together with the `m`, `h` and `n`
    /// gating variables. `membrane_potential` is in mV on this path; a spike is
    /// reported on the upward crossing of 0 mV.
    fn update_hh(&mut self, dt: f64) -> bool {
        // Conductances (mS/cm^2) and reversal potentials (mV).
        const G_NA: f32 = 120.0;
        const G_K: f32 = 36.0;
        const G_L: f32 = 0.3;
        const E_NA: f32 = 50.0;
        const E_K: f32 = -77.0;
        const E_L: f32 = -54.387;
        const CAPACITANCE: f32 = 1.0; // µF/cm^2

        let dt = dt as f32;

        let mut v = self.membrane_potential;
        let (mut m, mut h, mut n) = self.gating;
        let was_below = v < 0.0;

        // HH is stiff; sub-step at 10 µs or finer.
        let steps = ((dt / 0.01).ceil() as usize).max(1);
        let step_dt = dt / steps as f32;

        for _ in 0..steps {
            let alpha_m = 0.1 * (v + 40.0) / (1.0 - (-(v + 40.0) / 10.0).exp()).max(1e-6);
            let beta_m = 4.0 * (-(v + 65.0) / 18.0).exp();
            let alpha_h = 0.07 * (-(v + 65.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 35.0) / 10.0).exp());
            let alpha_n = 0.01 * (v + 55.0) / (1.0 - (-(v + 55.0) / 10.0).exp()).max(1e-6);
            let beta_n = 0.125 * (-(v + 65.0) / 80.0).exp();

            m += step_dt * (alpha_m * (1.0 - m) - beta_m * m);
            h += step_dt * (alpha_h * (1.0 - h) - beta_h * h);
            n += step_dt * (alpha_n * (1.0 - n) - beta_n * n);

            m = m.clamp(0.0, 1.0);
            h = h.clamp(0.0, 1.0);
            n = n.clamp(0.0, 1.0);

            let i_na = G_NA * m * m * m * h * (v - E_NA);
            let i_k = G_K * n * n * n * n * (v - E_K);
            let i_l = G_L * (v - E_L);

            v += step_dt * (self.input_current - i_na - i_k - i_l) / CAPACITANCE;
        }

        self.membrane_potential = v;
        self.gating = (m, h, n);
        self.input_current = 0.0;

        // A spike is the upward crossing of 0 mV.
        if was_below && v >= 0.0 {
            self.last_spike_time = self.time;
            return true;
        }
        false
    }

    /// Whether the neuron is still inside its refractory period.
    fn in_refractory(&self) -> bool {
        (self.time - self.last_spike_time) < self.refractory_period as f64
    }

    /// Deliver a presynaptic spike after `delay`.
    ///
    /// The spike is queued and applied when simulation time reaches
    /// `now + delay`; adding it to the input current immediately (as this used
    /// to) discarded the synaptic delay entirely, collapsing every conduction
    /// delay in the network to zero.
    pub fn receive_spike(&mut self, weight: f32, delay: f32) {
        if delay <= 0.0 {
            self.input_current += weight;
        } else {
            self.pending_spikes.push((self.time + delay as f64, weight));
        }
    }

    /// Move any spikes whose delay has elapsed into the input current.
    ///
    /// Called at the start of each update step.
    fn deliver_due_spikes(&mut self) {
        let now = self.time;
        let mut remaining = Vec::with_capacity(self.pending_spikes.len());
        for (arrival, weight) in self.pending_spikes.drain(..) {
            if arrival <= now {
                self.input_current += weight;
            } else {
                remaining.push((arrival, weight));
            }
        }
        self.pending_spikes = remaining;
    }
}

/// Network statistics
#[derive(Debug, Clone)]
pub struct NetworkStatistics {
    pub num_neurons: usize,
    pub num_synapses: usize,
    pub excitatory_synapses: usize,
    pub inhibitory_synapses: usize,
    pub average_weight: f32,
    pub plasticity_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: `update_aeif`, `update_izhikevich` and `update_hh` all
    /// called `update_lif`, so selecting a neuron model changed nothing about
    /// the dynamics. Each model must now evolve its own state.
    #[test]
    fn test_neuron_models_have_distinct_dynamics() {
        // A drive well below threshold, so each model's own dynamics — not
        // instant saturation — shape the trace.
        let drive = 0.05f32;
        let dt = 0.5;

        let mut traces: Vec<(NeuronType, Vec<f32>)> = Vec::new();
        for neuron_type in [
            NeuronType::LeakyIntegrateAndFire,
            NeuronType::AdaptiveExponential,
            NeuronType::Izhikevich,
        ] {
            let mut neuron = SpikingNeuron::new(0, neuron_type);
            let mut trace = Vec::new();
            for _ in 0..40 {
                neuron.receive_spike(drive, 0.0);
                neuron.update(dt);
                trace.push(neuron.membrane_potential);
            }
            traces.push((neuron_type, trace));
        }

        // No two models may produce an identical membrane-potential trace.
        for i in 0..traces.len() {
            for j in (i + 1)..traces.len() {
                assert_ne!(
                    traces[i].1, traces[j].1,
                    "{:?} and {:?} produced identical traces; the models are not distinct",
                    traces[i].0, traces[j].0
                );
            }
        }
    }

    /// AdEx must adapt: its spike-triggered `w` current makes successive
    /// inter-spike intervals lengthen under constant drive. Plain LIF does not.
    #[test]
    fn test_adex_adaptation_lengthens_intervals() {
        let mut neuron = SpikingNeuron::new(0, NeuronType::AdaptiveExponential);
        neuron.refractory_period = 0.0;

        let mut spike_times = Vec::new();
        for step in 0..2000 {
            neuron.receive_spike(1.2, 0.0);
            if neuron.update(0.1) {
                spike_times.push(step as f64 * 0.1);
            }
        }

        assert!(
            spike_times.len() >= 3,
            "constant drive should produce several spikes, got {}",
            spike_times.len()
        );

        let first_interval = spike_times[1] - spike_times[0];
        let last_interval = spike_times[spike_times.len() - 1] - spike_times[spike_times.len() - 2];
        assert!(
            last_interval > first_interval,
            "adaptation must lengthen intervals: first {first_interval}, last {last_interval}"
        );

        assert!(
            neuron.adaptation > 0.0,
            "the adaptation current must have accumulated"
        );
    }

    /// The Izhikevich model resets to its own `c` parameter (-65 mV), not to 0.
    #[test]
    fn test_izhikevich_uses_its_own_reset() {
        let mut neuron = SpikingNeuron::new(0, NeuronType::Izhikevich);
        neuron.membrane_potential = -65.0;
        neuron.adaptation = -13.0;

        let mut spiked = false;
        for _ in 0..400 {
            neuron.receive_spike(10.0, 0.0);
            if neuron.update(0.5) {
                spiked = true;
                break;
            }
        }

        assert!(spiked, "strong drive must elicit a spike");
        assert!(
            (neuron.membrane_potential - (-65.0)).abs() < 1e-4,
            "Izhikevich resets to c = -65 mV, got {}",
            neuron.membrane_potential
        );
        assert!(
            neuron.adaptation > -13.0,
            "the recovery variable must jump by d on spike"
        );
    }

    /// Hodgkin-Huxley must move its gating variables, which LIF has no notion of.
    #[test]
    fn test_hodgkin_huxley_evolves_gating_variables() {
        let mut neuron = SpikingNeuron::new(0, NeuronType::HodgkinHuxley);
        neuron.membrane_potential = -65.0;
        let initial_gating = neuron.gating;

        for _ in 0..200 {
            neuron.receive_spike(10.0, 0.0);
            neuron.update(0.05);
        }

        assert_ne!(
            neuron.gating, initial_gating,
            "the m/h/n gates must evolve under drive"
        );
        for gate in [neuron.gating.0, neuron.gating.1, neuron.gating.2] {
            assert!((0.0..=1.0).contains(&gate), "gate {gate} left [0,1]");
        }
        assert!(
            neuron.membrane_potential.is_finite(),
            "the HH integration must stay finite"
        );
    }

    /// Regression test: `receive_spike` discarded the synaptic delay, so every
    /// conduction delay in the network collapsed to zero.
    #[test]
    fn test_synaptic_delay_is_honoured() {
        let mut neuron = SpikingNeuron::new(0, NeuronType::LeakyIntegrateAndFire);
        neuron.threshold = 0.5;

        // A spike delayed by 5 time units must not arrive on the first step.
        neuron.receive_spike(10.0, 5.0);
        assert!(
            !neuron.update(1.0),
            "a delayed spike must not arrive immediately"
        );
        assert_eq!(
            neuron.pending_spikes.len(),
            1,
            "the spike is still in flight"
        );

        // After enough steps it arrives and drives the neuron over threshold.
        let mut fired = false;
        for _ in 0..10 {
            if neuron.update(1.0) {
                fired = true;
                break;
            }
        }
        assert!(fired, "the delayed spike must eventually arrive and fire");
        assert!(neuron.pending_spikes.is_empty());
    }

    /// A zero delay still delivers immediately.
    #[test]
    fn test_zero_delay_delivers_immediately() {
        let mut neuron = SpikingNeuron::new(0, NeuronType::LeakyIntegrateAndFire);
        neuron.threshold = 0.5;
        neuron.receive_spike(10.0, 0.0);
        assert!(
            neuron.update(1.0),
            "an undelayed spike arrives on this step"
        );
    }

    #[test]
    fn test_spiking_neuron_creation() {
        let neuron = SpikingNeuron::new(0, NeuronType::LeakyIntegrateAndFire);
        assert_eq!(neuron.id, 0);
        assert_eq!(neuron.membrane_potential, 0.0);
        assert_eq!(neuron.threshold, 1.0);
    }

    #[test]
    fn test_neuron_update() {
        let mut neuron = SpikingNeuron::new(0, NeuronType::LeakyIntegrateAndFire);
        neuron.input_current = 2.0; // Strong input

        let spiked = neuron.update(1.0);
        assert!(spiked); // Should spike with strong input
        assert_eq!(neuron.membrane_potential, 0.0); // Should reset
    }

    #[test]
    fn test_spiking_network_creation() {
        let network = SpikingNeuralNetwork::new(5);
        assert_eq!(network.neurons.len(), 5);
        assert_eq!(network.synapses.len(), 0);
    }

    #[test]
    fn test_add_synapse() {
        let mut network = SpikingNeuralNetwork::new(3);
        let result = network.add_synapse(0, 1, 0.5, 1.0);
        assert!(result.is_ok());
        assert_eq!(network.synapses.len(), 1);

        let synapse = &network.synapses[0];
        assert_eq!(synapse.pre_neuron, 0);
        assert_eq!(synapse.post_neuron, 1);
        assert_eq!(synapse.weight, 0.5);
    }

    #[test]
    fn test_invalid_synapse() {
        let mut network = SpikingNeuralNetwork::new(2);
        let result = network.add_synapse(0, 5, 0.5, 1.0); // Invalid post neuron
        assert!(result.is_err());
    }

    #[test]
    fn test_plasticity_rules() {
        let mut network = SpikingNeuralNetwork::new(3);
        let stdp_rule = PlasticityRule::STDP {
            tau_plus: 20.0,
            tau_minus: 20.0,
            a_plus: 0.01,
            a_minus: 0.012,
        };

        network.add_plasticity_rule(stdp_rule);
        assert_eq!(network.learning_rules.len(), 1);
    }

    #[test]
    fn test_network_statistics() {
        let mut network = SpikingNeuralNetwork::new(4);
        let _ = network.add_synapse(0, 1, 0.5, 1.0);
        let _ = network.add_synapse(1, 2, -0.3, 1.0);
        let _ = network.add_synapse(2, 3, 0.8, 1.0);

        let stats = network.get_statistics();
        assert_eq!(stats.num_neurons, 4);
        assert_eq!(stats.num_synapses, 3);
        assert_eq!(stats.excitatory_synapses, 2);
        assert_eq!(stats.inhibitory_synapses, 1);
    }

    #[test]
    fn test_layered_topology() {
        let mut network = SpikingNeuralNetwork::new(6);
        let topology = NetworkTopology::Layered {
            layers: vec![2, 2, 2],
        };
        network.set_topology(topology);

        // Should create connections between layers
        assert!(!network.synapses.is_empty());
    }

    #[test]
    fn test_synapse_types() {
        let excitatory = Synapse {
            pre_neuron: 0,
            post_neuron: 1,
            weight: 0.5,
            delay: 1.0,
            synapse_type: SynapseType::Excitatory,
            plasticity_enabled: true,
        };

        let inhibitory = Synapse {
            pre_neuron: 1,
            post_neuron: 2,
            weight: -0.3,
            delay: 1.0,
            synapse_type: SynapseType::Inhibitory,
            plasticity_enabled: true,
        };

        assert_eq!(excitatory.synapse_type, SynapseType::Excitatory);
        assert_eq!(inhibitory.synapse_type, SynapseType::Inhibitory);
    }

    #[test]
    fn test_power_gating() {
        let mut network = SpikingNeuralNetwork::new(3);
        assert!(!network.power_gated);

        network.enable_power_gating();
        assert!(network.power_gated);
    }

    #[test]
    fn test_threshold_adjustment() {
        let mut network = SpikingNeuralNetwork::new(3);
        let original_threshold = network.neurons[0].threshold;

        network.adjust_firing_thresholds(1.5);
        assert_eq!(network.neurons[0].threshold, original_threshold * 1.5);
    }
}
