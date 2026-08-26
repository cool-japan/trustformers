//! Spiking neural networks with configurable neuron dynamics.
//!
//! [`SpikingLayer`] dispatches on [`NeuronModel`] and integrates the selected
//! membrane equation with the timing constants taken from
//! [`BiologicalConfig`] (`dt`, `tau_mem`, `tau_syn`, `v_threshold`, `v_reset`,
//! `refractory_period`, `noise_variance`). Five models are implemented:
//!
//! | Model | State | Units |
//! |---|---|---|
//! | Leaky integrate-and-fire | `v` | configuration units |
//! | Izhikevich | `v`, `u` | mV / ms |
//! | Adaptive exponential IF | `v`, `w` | mV / ms |
//! | Hodgkin–Huxley | `v`, `m`, `h`, `n` | mV / ms |
//! | Spike response model | synaptic + refractory kernels | configuration units |
//!
//! Models defined in the neuroscience literature in millivolts and
//! milliseconds (Izhikevich, AdEx, Hodgkin–Huxley) convert `config.dt` from
//! seconds to milliseconds; the configured `v_threshold` / `v_reset` do not
//! apply to them because their thresholds and reset potentials are intrinsic
//! to the model. That is documented per model below.

use trustformers_core::{
    errors::{Result, TrustformersError},
    layers::{LayerNorm, Linear},
    tensor::Tensor,
    traits::Layer,
};

use super::{
    config::{BiologicalConfig, NeuronModel, PlasticityType},
    model::BiologicalModelOutput,
};

/// Hodgkin–Huxley gating variables.
#[derive(Debug, Clone)]
pub struct HodgkinHuxleyGating {
    /// Sodium activation
    pub m: Tensor,
    /// Sodium inactivation
    pub h: Tensor,
    /// Potassium activation
    pub n: Tensor,
}

/// Neuron state for spiking neural networks
#[derive(Debug, Clone)]
pub struct NeuronState {
    /// Membrane potential
    pub v_mem: Tensor,
    /// Recovery variable (Izhikevich)
    pub u_recovery: Option<Tensor>,
    /// Adaptation current (adaptive exponential IF)
    pub adaptation: Option<Tensor>,
    /// Gating variables (Hodgkin–Huxley)
    pub gating: Option<HodgkinHuxleyGating>,
    /// Synaptic kernel state (spike response model)
    pub synaptic_trace: Option<Tensor>,
    /// Refractory kernel state (spike response model)
    pub refractory_trace: Option<Tensor>,
    /// Refractory time remaining
    pub refractory_time: Tensor,
    /// Spike output
    pub spikes: Tensor,
}

impl NeuronState {
    /// Create an all-zero state with no model-specific variables.
    pub fn zeros(batch_size: usize, neurons: usize) -> Result<Self> {
        Ok(Self {
            v_mem: Tensor::zeros(&[batch_size, neurons])?,
            u_recovery: None,
            adaptation: None,
            gating: None,
            synaptic_trace: None,
            refractory_trace: None,
            refractory_time: Tensor::zeros(&[batch_size, neurons])?,
            spikes: Tensor::zeros(&[batch_size, neurons])?,
        })
    }
}

/// Synaptic state for plasticity
#[derive(Debug, Clone)]
pub struct SynapticState {
    /// Synaptic weights, shape `[neurons, neurons]`
    pub weights: Tensor,
    /// Presynaptic traces, shape `[batch, neurons]`
    pub pre_traces: Tensor,
    /// Postsynaptic traces, shape `[batch, neurons]`
    pub post_traces: Tensor,
    /// Eligibility traces, shape `[neurons, neurons]`
    pub eligibility: Tensor,
}

// --- Izhikevich constants (regular-spiking cortical neuron, mV / ms) ---
const IZH_A: f32 = 0.02;
const IZH_B: f32 = 0.2;
const IZH_C: f32 = -65.0;
const IZH_D: f32 = 8.0;
/// Izhikevich spikes are detected at the intrinsic +30 mV peak.
const IZH_PEAK: f32 = 30.0;

// --- Adaptive exponential integrate-and-fire constants (mV / ms) ---
const ADEX_E_L: f32 = -70.6;
const ADEX_V_T: f32 = -50.4;
const ADEX_DELTA_T: f32 = 2.0;
const ADEX_TAU_W_MS: f32 = 144.0;
const ADEX_A: f32 = 4.0;
const ADEX_B: f32 = 80.5;
const ADEX_PEAK: f32 = 0.0;

// --- Hodgkin–Huxley constants (Hodgkin & Huxley 1952, mV / ms) ---
const HH_C_M: f32 = 1.0;
const HH_G_NA: f32 = 120.0;
const HH_G_K: f32 = 36.0;
const HH_G_L: f32 = 0.3;
const HH_E_NA: f32 = 50.0;
const HH_E_K: f32 = -77.0;
const HH_E_L: f32 = -54.387;
const HH_V_REST: f32 = -65.0;
/// Upward crossing of this potential counts as a spike.
const HH_SPIKE_THRESHOLD: f32 = 0.0;
/// Largest stable explicit-Euler step for the Hodgkin–Huxley equations.
const HH_MAX_SUBSTEP_MS: f32 = 0.01;

/// `coefficient · (v + offset) / (1 − exp(−(v + offset)/scale))`.
///
/// The expression has a removable singularity at `v = −offset`; the limit
/// there is `coefficient · scale`, which is used directly to keep the rate
/// finite.
fn hh_exp_rate(coefficient: f32, offset: f32, scale: f32, v: f32) -> f32 {
    let shifted = v + offset;
    let x = shifted / scale;
    if x.abs() < 1e-5 {
        coefficient * scale
    } else {
        coefficient * shifted / (1.0 - (-x).exp())
    }
}

fn hh_alpha_m(v: f32) -> f32 {
    hh_exp_rate(0.1, 40.0, 10.0, v)
}
fn hh_beta_m(v: f32) -> f32 {
    4.0 * (-(v + 65.0) / 18.0).exp()
}
fn hh_alpha_h(v: f32) -> f32 {
    0.07 * (-(v + 65.0) / 20.0).exp()
}
fn hh_beta_h(v: f32) -> f32 {
    1.0 / (1.0 + (-(v + 35.0) / 10.0).exp())
}
fn hh_alpha_n(v: f32) -> f32 {
    hh_exp_rate(0.01, 55.0, 10.0, v)
}
fn hh_beta_n(v: f32) -> f32 {
    0.125 * (-(v + 65.0) / 80.0).exp()
}

/// Steady-state gating values at the Hodgkin–Huxley resting potential.
fn hh_steady_state(v: f32) -> (f32, f32, f32) {
    let m = hh_alpha_m(v) / (hh_alpha_m(v) + hh_beta_m(v));
    let h = hh_alpha_h(v) / (hh_alpha_h(v) + hh_beta_h(v));
    let n = hh_alpha_n(v) / (hh_alpha_n(v) + hh_beta_n(v));
    (m, h, n)
}

/// Spiking neural network layer
#[derive(Debug)]
pub struct SpikingLayer {
    /// Configuration
    pub config: BiologicalConfig,
    /// Input projection
    pub input_projection: Linear,
    /// Recurrent projection
    pub recurrent_projection: Linear,
    /// Neuron states
    pub neuron_states: Option<NeuronState>,
    /// Synaptic states
    pub synaptic_states: Option<SynapticState>,
    /// Layer normalization
    pub layer_norm: LayerNorm,
    /// Width of the input this layer consumes
    pub input_dim: usize,
}

impl SpikingLayer {
    /// Create a new spiking layer that consumes `config.d_model` inputs.
    pub fn new(config: &BiologicalConfig) -> Result<Self> {
        Self::with_input_dim(config, config.d_model)
    }

    /// Create a spiking layer with an explicit input width.
    ///
    /// Stacked layers consume the spike vector of the layer below, which is
    /// `neurons_per_layer` wide rather than `d_model` wide.
    pub fn with_input_dim(config: &BiologicalConfig, input_dim: usize) -> Result<Self> {
        let input_projection = Linear::new(input_dim, config.neurons_per_layer, config.use_bias);
        let recurrent_projection =
            Linear::new(config.neurons_per_layer, config.neurons_per_layer, false);
        let layer_norm = LayerNorm::new(vec![config.neurons_per_layer], 1e-12)?;

        Ok(Self {
            config: config.clone(),
            input_projection,
            recurrent_projection,
            neuron_states: None,
            synaptic_states: None,
            layer_norm,
            input_dim,
        })
    }

    /// Initialize neuron states for the configured neuron model.
    pub fn init_states(&mut self, batch_size: usize) -> Result<()> {
        let neurons = self.config.neurons_per_layer;
        let mut state = NeuronState::zeros(batch_size, neurons)?;

        match self.config.neuron_model {
            NeuronModel::LeakyIntegrateAndFire => {
                state.v_mem = Tensor::full(self.config.v_reset, vec![batch_size, neurons])?;
            },
            NeuronModel::Izhikevich => {
                state.v_mem = Tensor::full(IZH_C, vec![batch_size, neurons])?;
                state.u_recovery = Some(Tensor::full(IZH_B * IZH_C, vec![batch_size, neurons])?);
            },
            NeuronModel::AdaptiveExponentialIF => {
                state.v_mem = Tensor::full(ADEX_E_L, vec![batch_size, neurons])?;
                state.adaptation = Some(Tensor::zeros(&[batch_size, neurons])?);
            },
            NeuronModel::HodgkinHuxley => {
                let (m, h, n) = hh_steady_state(HH_V_REST);
                state.v_mem = Tensor::full(HH_V_REST, vec![batch_size, neurons])?;
                state.gating = Some(HodgkinHuxleyGating {
                    m: Tensor::full(m, vec![batch_size, neurons])?,
                    h: Tensor::full(h, vec![batch_size, neurons])?,
                    n: Tensor::full(n, vec![batch_size, neurons])?,
                });
            },
            NeuronModel::SpikeResponseModel => {
                state.v_mem = Tensor::full(self.config.v_reset, vec![batch_size, neurons])?;
                state.synaptic_trace = Some(Tensor::zeros(&[batch_size, neurons])?);
                state.refractory_trace = Some(Tensor::zeros(&[batch_size, neurons])?);
            },
        }

        self.neuron_states = Some(state);

        let weights =
            Tensor::randn(&[neurons, neurons])?.scalar_mul(self.config.initializer_range)?;
        self.synaptic_states = Some(SynapticState {
            weights,
            pre_traces: Tensor::zeros(&[batch_size, neurons])?,
            post_traces: Tensor::zeros(&[batch_size, neurons])?,
            eligibility: Tensor::zeros(&[neurons, neurons])?,
        });

        Ok(())
    }

    /// Forward pass through the spiking layer.
    ///
    /// `input` is `[batch, seq_len, input_dim]`; the output is
    /// `[batch, seq_len, neurons_per_layer]`.
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(TrustformersError::shape_error(format!(
                "SpikingLayer::forward expects [batch, seq_len, input_dim], got {:?}",
                shape
            )));
        }
        let batch_size = shape[0];
        let seq_len = shape[1];

        if self.neuron_states.is_none() {
            self.init_states(batch_size)?;
        }

        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let input_t = input.slice(1, t, t + 1)?.squeeze(1)?;
            let output_t = self.forward_timestep(&input_t)?;
            outputs.push(output_t.unsqueeze(1)?);
        }

        Tensor::concat(&outputs, 1)?.contiguous()
    }

    /// Advance the layer by one timestep (`input` is `[batch, input_dim]`).
    fn forward_timestep(&mut self, input: &Tensor) -> Result<Tensor> {
        let input_current = self.input_projection.forward(input.clone())?;

        let recurrent_current = {
            let states = self.states()?;
            self.recurrent_projection.forward(states.spikes.clone())?
        };

        let total_current = input_current.add(&recurrent_current)?;

        // Dispatch on the configured neuron model — no silent LIF fallback.
        self.update_dynamics(&total_current)?;

        // Synaptic plasticity as configured.
        self.update_plasticity()?;

        let spikes = self.states()?.spikes.clone();
        self.layer_norm.forward(spikes)
    }

    fn states(&self) -> Result<&NeuronState> {
        self.neuron_states.as_ref().ok_or_else(|| {
            TrustformersError::runtime_error("Neuron states not initialized".to_string())
        })
    }

    /// Integrate the configured membrane equation for one timestep.
    pub fn update_dynamics(&mut self, current: &Tensor) -> Result<()> {
        let mut states = self.neuron_states.take().ok_or_else(|| {
            TrustformersError::runtime_error("Neuron states not initialized".to_string())
        })?;

        let result = match self.config.neuron_model {
            NeuronModel::LeakyIntegrateAndFire => {
                update_lif_dynamics(&self.config, current, &mut states)
            },
            NeuronModel::Izhikevich => {
                update_izhikevich_dynamics(&self.config, current, &mut states)
            },
            NeuronModel::AdaptiveExponentialIF => {
                update_adexp_dynamics(&self.config, current, &mut states)
            },
            NeuronModel::HodgkinHuxley => update_hh_dynamics(&self.config, current, &mut states),
            NeuronModel::SpikeResponseModel => {
                update_srm_dynamics(&self.config, current, &mut states)
            },
        };

        self.neuron_states = Some(states);
        result
    }

    /// Update synaptic traces and weights according to `config.plasticity_type`.
    pub fn update_plasticity(&mut self) -> Result<()> {
        let dt = self.config.dt;
        let learning_rate = self.config.learning_rate;
        let tau_trace = self.config.tau_syn;
        let plasticity = self.config.plasticity_type.clone();
        let target_rate = self.config.target_rate;

        let spikes = self.states()?.spikes.clone();
        let synaptic_states = self.synaptic_states.as_mut().ok_or_else(|| {
            TrustformersError::runtime_error("Synaptic states not initialized".to_string())
        })?;

        // Exponentially decaying eligibility traces.
        let trace_decay = (-dt / tau_trace).exp();
        synaptic_states.pre_traces =
            synaptic_states.pre_traces.mul_scalar(trace_decay)?.add(&spikes)?;
        synaptic_states.post_traces =
            synaptic_states.post_traces.mul_scalar(trace_decay)?.add(&spikes)?;

        match plasticity {
            PlasticityType::STDP => {
                // LTP: pre trace paired with a post spike. LTD: post trace
                // paired with a pre spike. Both are [neurons, neurons].
                let ltp = synaptic_states.pre_traces.transpose(0, 1)?.matmul(&spikes)?;
                let ltd = spikes.transpose(0, 1)?.matmul(&synaptic_states.post_traces)?;
                let update = ltp.sub(&ltd)?.mul_scalar(learning_rate)?;
                synaptic_states.weights = synaptic_states.weights.add(&update)?;
            },
            PlasticityType::Hebbian => {
                let update = spikes.transpose(0, 1)?.matmul(&spikes)?.mul_scalar(learning_rate)?;
                synaptic_states.weights = synaptic_states.weights.add(&update)?;
            },
            PlasticityType::AntiHebbian => {
                let update = spikes.transpose(0, 1)?.matmul(&spikes)?.mul_scalar(-learning_rate)?;
                synaptic_states.weights = synaptic_states.weights.add(&update)?;
            },
            PlasticityType::Homeostatic => {
                let update = homeostatic_delta(&spikes, dt, target_rate, learning_rate)?;
                synaptic_states.weights = synaptic_states.weights.add_scalar(update)?;
            },
            PlasticityType::Metaplasticity => {
                // STDP under homeostatic regulation.
                let ltp = synaptic_states.pre_traces.transpose(0, 1)?.matmul(&spikes)?;
                let ltd = spikes.transpose(0, 1)?.matmul(&synaptic_states.post_traces)?;
                let update = ltp.sub(&ltd)?.mul_scalar(learning_rate * 0.8)?;
                synaptic_states.weights = synaptic_states.weights.add(&update)?;

                let homeostatic = homeostatic_delta(&spikes, dt, target_rate, learning_rate * 0.2)?;
                synaptic_states.weights = synaptic_states.weights.add_scalar(homeostatic)?;
            },
        }

        Ok(())
    }

    /// Reset neuron and synaptic state to their model-specific rest values.
    pub fn reset_states(&mut self) -> Result<()> {
        let batch_size = match &self.neuron_states {
            Some(states) => states.v_mem.shape()[0],
            None => return Ok(()),
        };
        let weights = self.synaptic_states.as_ref().map(|s| s.weights.clone());
        self.init_states(batch_size)?;
        if let (Some(weights), Some(synaptic)) = (weights, self.synaptic_states.as_mut()) {
            synaptic.weights = weights;
        }
        Ok(())
    }

    /// Total spike count across the current spike vector.
    pub fn spike_count(&self) -> Result<f32> {
        Ok(self.states()?.spikes.to_vec_f32()?.iter().sum())
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.input_projection.parameter_count()
            + self.recurrent_projection.parameter_count()
            + self.layer_norm.parameter_count()
    }

    /// Get memory usage in MB
    pub fn memory_usage(&self) -> f32 {
        let param_memory = self.parameter_count() as f32 * 4.0 / 1_000_000.0;
        let state_memory = if self.neuron_states.is_some() {
            self.config.neurons_per_layer as f32 * 4.0 * 4.0 / 1_000_000.0
        } else {
            0.0
        };
        param_memory + state_memory
    }
}

/// Homeostatic weight adjustment driven by the deviation of the instantaneous
/// firing rate (in Hz) from `target_rate`.
fn homeostatic_delta(spikes: &Tensor, dt: f32, target_rate: f32, lr: f32) -> Result<f32> {
    let mean_spike = spikes.mean()?.to_scalar()?;
    let rate_hz = mean_spike / dt.max(f32::EPSILON);
    Ok(-lr * (rate_hz - target_rate))
}

/// Add configured membrane noise, if any. A zero `noise_variance` keeps the
/// dynamics deterministic.
fn add_noise(config: &BiologicalConfig, tensor: &Tensor) -> Result<Tensor> {
    if config.noise_variance <= 0.0 {
        return Ok(tensor.clone());
    }
    let noise = Tensor::randn_like(tensor)?.scalar_mul(config.noise_variance)?;
    tensor.add(&noise)
}

/// Threshold, emit spikes, and reset to `reset_value` where a spike occurred.
fn threshold_and_reset(
    v_mem: &Tensor,
    threshold: f32,
    reset_value: f32,
) -> Result<(Tensor, Tensor)> {
    let threshold_tensor = Tensor::full(threshold, v_mem.shape())?;
    let spikes = v_mem.greater(&threshold_tensor)?;
    let keep = Tensor::ones_like(&spikes)?.sub(&spikes)?;
    let reset_tensor = Tensor::full(reset_value, v_mem.shape())?;
    let reset = v_mem.mul(&keep)?.add(&reset_tensor.mul(&spikes)?)?;
    Ok((spikes, reset))
}

/// Advance the refractory countdown and arm it for freshly spiking neurons.
fn update_refractory(config: &BiologicalConfig, states: &mut NeuronState) -> Result<()> {
    states.refractory_time = states
        .refractory_time
        .sub_scalar(config.dt)?
        .clamp(0.0, f32::INFINITY)?
        .add(&states.spikes.mul_scalar(config.refractory_period)?)?;
    Ok(())
}

/// Leaky integrate-and-fire: `τ dv/dt = -(v - v_reset) + I`.
///
/// Uses `config.dt`, `config.tau_mem`, `config.v_threshold`, `config.v_reset`
/// and `config.refractory_period` directly.
pub fn update_lif_dynamics(
    config: &BiologicalConfig,
    current: &Tensor,
    states: &mut NeuronState,
) -> Result<()> {
    let decay = (-config.dt / config.tau_mem).exp();
    let leak = states
        .v_mem
        .sub_scalar(config.v_reset)?
        .mul_scalar(decay)?
        .add_scalar(config.v_reset)?;
    let driven = leak.add(&current.mul_scalar(config.dt / config.tau_mem)?)?;
    states.v_mem = add_noise(config, &driven)?;

    let (spikes, reset) = threshold_and_reset(&states.v_mem, config.v_threshold, config.v_reset)?;
    states.spikes = spikes;
    states.v_mem = reset;

    update_refractory(config, states)
}

/// Izhikevich dynamics (mV / ms).
///
/// `config.dt` is interpreted in seconds and converted to milliseconds. The
/// threshold (+30 mV) and reset (`c`, `d`) are intrinsic to the model, so
/// `config.v_threshold` / `config.v_reset` do not apply here.
pub fn update_izhikevich_dynamics(
    config: &BiologicalConfig,
    current: &Tensor,
    states: &mut NeuronState,
) -> Result<()> {
    let dt_ms = config.dt * 1000.0;
    let u_recovery = states.u_recovery.clone().ok_or_else(|| {
        TrustformersError::runtime_error(
            "Recovery variable not initialized for Izhikevich model".to_string(),
        )
    })?;

    let dv = states
        .v_mem
        .pow_scalar(2.0)?
        .mul_scalar(0.04)?
        .add(&states.v_mem.mul_scalar(5.0)?)?
        .add_scalar(140.0)?
        .sub(&u_recovery)?
        .add(current)?;
    let v_next = states.v_mem.add(&dv.mul_scalar(dt_ms)?)?;
    let v_next = add_noise(config, &v_next)?;

    let du = v_next.mul_scalar(IZH_B)?.sub(&u_recovery)?.mul_scalar(IZH_A)?;
    let u_next = u_recovery.add(&du.mul_scalar(dt_ms)?)?;

    let (spikes, reset) = threshold_and_reset(&v_next, IZH_PEAK, IZH_C)?;
    let keep = Tensor::ones_like(&spikes)?.sub(&spikes)?;
    let u_after = u_next.mul(&keep)?.add(&u_next.add_scalar(IZH_D)?.mul(&spikes)?)?;

    states.v_mem = reset;
    states.u_recovery = Some(u_after);
    states.spikes = spikes;

    update_refractory(config, states)
}

/// Adaptive exponential integrate-and-fire (Brette & Gerstner 2005, mV / ms).
///
/// `config.tau_mem` sets the membrane time constant (converted to ms); the
/// threshold, reset and adaptation parameters are intrinsic to the model.
pub fn update_adexp_dynamics(
    config: &BiologicalConfig,
    current: &Tensor,
    states: &mut NeuronState,
) -> Result<()> {
    let dt_ms = config.dt * 1000.0;
    let tau_mem_ms = (config.tau_mem * 1000.0).max(f32::EPSILON);
    let adaptation = states.adaptation.clone().ok_or_else(|| {
        TrustformersError::runtime_error(
            "Adaptation current not initialized for AdExp model".to_string(),
        )
    })?;

    // Clamp the exponential argument so the upstroke cannot overflow to inf.
    let exponential = states
        .v_mem
        .sub_scalar(ADEX_V_T)?
        .div_scalar(ADEX_DELTA_T)?
        .clamp(f32::NEG_INFINITY, 20.0)?
        .exp()?
        .mul_scalar(ADEX_DELTA_T)?;

    let dv = states
        .v_mem
        .sub_scalar(ADEX_E_L)?
        .mul_scalar(-1.0)?
        .add(&exponential)?
        .sub(&adaptation)?
        .add(current)?
        .div_scalar(tau_mem_ms)?;
    let v_next = states.v_mem.add(&dv.mul_scalar(dt_ms)?)?;
    let v_next = add_noise(config, &v_next)?;

    let dw = v_next
        .sub_scalar(ADEX_E_L)?
        .mul_scalar(ADEX_A)?
        .sub(&adaptation)?
        .div_scalar(ADEX_TAU_W_MS)?;
    let w_next = adaptation.add(&dw.mul_scalar(dt_ms)?)?;

    let (spikes, reset) = threshold_and_reset(&v_next, ADEX_PEAK, ADEX_E_L)?;
    let keep = Tensor::ones_like(&spikes)?.sub(&spikes)?;
    let w_after = w_next.mul(&keep)?.add(&w_next.add_scalar(ADEX_B)?.mul(&spikes)?)?;

    states.v_mem = reset;
    states.adaptation = Some(w_after);
    states.spikes = spikes;

    update_refractory(config, states)
}

/// Hodgkin–Huxley dynamics with the full `m`/`h`/`n` gating ODEs.
///
/// The equations are integrated with explicit Euler sub-steps of at most
/// `HH_MAX_SUBSTEP_MS` milliseconds, which keeps the stiff sodium current
/// stable for the millisecond-scale `config.dt` used elsewhere. A spike is an
/// upward crossing of 0 mV, so a single action potential is reported once.
pub fn update_hh_dynamics(
    config: &BiologicalConfig,
    current: &Tensor,
    states: &mut NeuronState,
) -> Result<()> {
    let gating = states.gating.clone().ok_or_else(|| {
        TrustformersError::runtime_error(
            "Gating variables not initialized for Hodgkin-Huxley model".to_string(),
        )
    })?;

    let shape = states.v_mem.shape();
    let mut v = states.v_mem.to_vec_f32()?;
    let mut m = gating.m.to_vec_f32()?;
    let mut h = gating.h.to_vec_f32()?;
    let mut n = gating.n.to_vec_f32()?;
    let injected = current.to_vec_f32()?;

    let dt_ms = config.dt * 1000.0;
    let substeps = (dt_ms / HH_MAX_SUBSTEP_MS).ceil().max(1.0) as usize;
    let step = dt_ms / substeps as f32;

    let mut spikes = vec![0.0f32; v.len()];

    for index in 0..v.len() {
        let drive = injected.get(index).copied().unwrap_or(0.0);
        let mut crossed = false;
        for _ in 0..substeps {
            let v_i = v[index];
            let was_below = v_i <= HH_SPIKE_THRESHOLD;

            let i_na = HH_G_NA * m[index].powi(3) * h[index] * (v_i - HH_E_NA);
            let i_k = HH_G_K * n[index].powi(4) * (v_i - HH_E_K);
            let i_l = HH_G_L * (v_i - HH_E_L);
            let dv = (drive - i_na - i_k - i_l) / HH_C_M;

            let dm = hh_alpha_m(v_i) * (1.0 - m[index]) - hh_beta_m(v_i) * m[index];
            let dh = hh_alpha_h(v_i) * (1.0 - h[index]) - hh_beta_h(v_i) * h[index];
            let dn = hh_alpha_n(v_i) * (1.0 - n[index]) - hh_beta_n(v_i) * n[index];

            v[index] = v_i + step * dv;
            m[index] = (m[index] + step * dm).clamp(0.0, 1.0);
            h[index] = (h[index] + step * dh).clamp(0.0, 1.0);
            n[index] = (n[index] + step * dn).clamp(0.0, 1.0);

            if was_below && v[index] > HH_SPIKE_THRESHOLD {
                crossed = true;
            }
        }
        if crossed {
            spikes[index] = 1.0;
        }
    }

    states.v_mem = add_noise(config, &Tensor::from_vec(v, &shape)?)?;
    states.gating = Some(HodgkinHuxleyGating {
        m: Tensor::from_vec(m, &shape)?,
        h: Tensor::from_vec(h, &shape)?,
        n: Tensor::from_vec(n, &shape)?,
    });
    states.spikes = Tensor::from_vec(spikes, &shape)?;

    update_refractory(config, states)
}

/// Spike response model (SRM₀) with exponential synaptic and refractory
/// kernels.
///
/// The membrane potential is `u = (κ * I)(t) − η(t)`, where the synaptic
/// kernel `κ` decays with `config.tau_syn`, the refractory kernel `η` decays
/// with `config.tau_mem`, and each emitted spike adds a unit refractory kick
/// scaled by `v_threshold − v_reset`. Spikes are emitted when `u` crosses
/// `config.v_threshold`.
pub fn update_srm_dynamics(
    config: &BiologicalConfig,
    current: &Tensor,
    states: &mut NeuronState,
) -> Result<()> {
    let synaptic = states.synaptic_trace.clone().ok_or_else(|| {
        TrustformersError::runtime_error(
            "Synaptic kernel not initialized for spike response model".to_string(),
        )
    })?;
    let refractory = states.refractory_trace.clone().ok_or_else(|| {
        TrustformersError::runtime_error(
            "Refractory kernel not initialized for spike response model".to_string(),
        )
    })?;

    let synaptic_decay = (-config.dt / config.tau_syn).exp();
    let refractory_decay = (-config.dt / config.tau_mem).exp();
    let kick = (config.v_threshold - config.v_reset).abs().max(f32::EPSILON);

    let synaptic_next =
        synaptic.mul_scalar(synaptic_decay)?.add(&current.mul_scalar(config.dt)?)?;
    let refractory_next = refractory.mul_scalar(refractory_decay)?;

    let potential = synaptic_next
        .sub(&refractory_next.mul_scalar(kick)?)?
        .add_scalar(config.v_reset)?;
    let potential = add_noise(config, &potential)?;

    let threshold_tensor = Tensor::full(config.v_threshold, potential.shape())?;
    let spikes = potential.greater(&threshold_tensor)?;

    states.v_mem = potential;
    states.synaptic_trace = Some(synaptic_next);
    states.refractory_trace = Some(refractory_next.add(&spikes)?);
    states.spikes = spikes;

    update_refractory(config, states)
}

/// Spiking neural network model
#[derive(Debug)]
pub struct SpikingNeuralNetwork {
    /// Configuration
    pub config: BiologicalConfig,
    /// Spiking layers
    pub layers: Vec<SpikingLayer>,
    /// Output projection
    pub output_projection: Linear,
}

impl SpikingNeuralNetwork {
    /// Create a new spiking neural network
    pub fn new(config: &BiologicalConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for index in 0..config.n_layer {
            // The first layer consumes d_model inputs; deeper layers consume
            // the spike vector produced by the layer below.
            let input_dim = if index == 0 { config.d_model } else { config.neurons_per_layer };
            layers.push(SpikingLayer::with_input_dim(config, input_dim)?);
        }

        let output_projection =
            Linear::new(config.neurons_per_layer, config.d_model, config.use_bias);

        Ok(Self {
            config: config.clone(),
            layers,
            output_projection,
        })
    }

    /// Forward pass through the network.
    ///
    /// `input` is `[batch, seq_len, d_model]`; the output is the same shape.
    pub fn forward(&mut self, input: &Tensor) -> Result<BiologicalModelOutput> {
        let mut hidden_states = input.clone();
        let mut all_spike_trains = Vec::new();

        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states)?;
            if let Some(states) = &layer.neuron_states {
                all_spike_trains.push(states.spikes.clone().unsqueeze(1)?);
            }
        }

        let output = self.output_projection.forward(hidden_states)?;

        let spike_trains = if all_spike_trains.is_empty() {
            None
        } else {
            Some(Tensor::concat(&all_spike_trains, 1)?.contiguous()?)
        };

        Ok(BiologicalModelOutput {
            hidden_states: output,
            spike_trains,
            memory_states: None,
            attention_weights: None,
            capsule_outputs: None,
            dendritic_activations: None,
            plasticity_traces: None,
        })
    }

    /// Apply one plasticity update across all layers.
    pub fn update_plasticity(&mut self, _targets: &Tensor) -> Result<()> {
        for layer in &mut self.layers {
            if layer.neuron_states.is_some() {
                layer.update_plasticity()?;
            }
        }
        Ok(())
    }

    /// Reset states for all layers
    pub fn reset_states(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.reset_states()?;
        }
        Ok(())
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|l| l.parameter_count()).sum::<usize>()
            + self.output_projection.parameter_count()
    }

    /// Get memory usage in MB
    pub fn memory_usage(&self) -> f32 {
        self.layers.iter().map(|l| l.memory_usage()).sum::<f32>()
            + (self.output_projection.parameter_count() as f32 * 4.0 / 1_000_000.0)
    }
}
