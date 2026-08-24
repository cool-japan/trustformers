//! The S4 state-space layer: parameters, discretisation cache and recurrence.
//!
//! # Cache invalidation
//!
//! The layer holds the continuous-time parameters `A`, `B`, `C`, `D`, `Δ` and a
//! *cache* of the discrete-time pair `Ā`, `B̄` derived from `A`, `B` and `Δ`.
//! Every setter here drops that cache and rebuilds it, because installing a
//! checkpoint's parameters while leaving a stale `Ā`/`B̄` in place would make the
//! model keep computing with the *pre-load* discretisation while reporting a
//! successful load — the same silent-wrong-answer class this crate is being
//! cleaned of, in a new disguise. [`S4Layer::is_discretized`] and
//! [`S4Layer::discretization_generation`] exist so a test can prove it.
//!
//! # One discretisation, several channels
//!
//! The reference implementation (state-spaces/s4) keeps a *diagonal* `A`, so it
//! can afford a separate `Ā` per SSM channel — `O(N)` each. This implementation
//! keeps a dense `[N, N]` `A`, where a per-channel `Ā` would cost
//! `n_ssm × N × N` (50 MB per layer at the default `d_model = 768`,
//! `d_state = 64`). It therefore discretises **once**, at the mean of `Δ`, and
//! shares `Ā`/`B̄` across channels; the per-channel `Δ` enters through that mean
//! and the per-channel `D` is applied in full. That is a documented modelling
//! difference from the reference, not an approximation hidden behind a comment.

use scirs2_core::ndarray::{Array1, Array2}; // SciRS2 Integration Policy
use scirs2_core::Complex64; // SciRS2 Integration Policy
use trustformers_core::{
    device::Device,
    errors::{runtime_error, Result},
};

use super::config::S4Config;
use super::discretization::{Discretization, HiPPOMatrix};

/// S4 Layer implementing the diagonal plus low-rank structure
pub struct S4Layer {
    config: S4Config,
    // State space parameters
    a_real: Array2<f32>, // Real part of A matrix
    a_imag: Array2<f32>, // Imaginary part of A matrix
    b_real: Array1<f32>, // Real part of B vector
    b_imag: Array1<f32>, // Imaginary part of B vector
    c_real: Array1<f32>, // Real part of C vector
    c_imag: Array1<f32>, // Imaginary part of C vector
    d: Array1<f32>,      // D vector (skip connection)
    dt: Array1<f32>,     // Discretization timestep
    // Cached discrete parameters, rebuilt whenever a parameter above changes.
    a_bar: Option<Array2<Complex64>>,
    b_bar: Option<Array1<Complex64>>,
    /// Bumped on every rebuild, so a caller can observe that invalidation
    /// actually happened rather than trusting that it did.
    discretization_generation: u64,
    // Device for computation
    device: Device,
}

impl S4Layer {
    /// Build a layer and discretise it, so it is ready to run.
    ///
    /// # Errors
    ///
    /// Fails when the configured discretisation has no solution for these
    /// parameters (a singular implicit factor, a non-finite entry).
    pub fn new_with_device(config: &S4Config, device: Device) -> Result<Self> {
        let n = config.d_state;
        let h = config.get_n_ssm();

        let hippo = match config.hippo_matrix.as_str() {
            "legs" => HiPPOMatrix::LEGS,
            "legt" => HiPPOMatrix::LEGT,
            "lagt" => HiPPOMatrix::LAGT,
            "fourier" => HiPPOMatrix::Fourier,
            "random" => HiPPOMatrix::Random,
            _ => HiPPOMatrix::LEGS,
        };

        let a_base = hippo.initialize(n);

        // Initialize as diagonal plus low-rank for efficiency
        // A = Λ - pq^T where Λ is diagonal
        let a_real = a_base.clone();
        let a_imag = Array2::<f32>::zeros((n, n));

        // Initialize B, C, D
        let b_real = Array1::<f32>::ones(n) / (n as f32).sqrt();
        let b_imag = Array1::<f32>::zeros(n);
        let c_real = Array1::<f32>::ones(n) / (n as f32).sqrt();
        let c_imag = Array1::<f32>::zeros(n);
        let d = Array1::<f32>::ones(h);

        // Initialize timestep
        let dt = Array1::<f32>::from_elem(h, config.dt);

        let mut layer = Self {
            config: config.clone(),
            a_real,
            a_imag,
            b_real,
            b_imag,
            c_real,
            c_imag,
            d,
            dt,
            a_bar: None,
            b_bar: None,
            discretization_generation: 0,
            device,
        };
        layer.discretize()?;
        Ok(layer)
    }

    /// Build a layer on the CPU.
    ///
    /// # Errors
    ///
    /// See [`S4Layer::new_with_device`].
    pub fn new(config: &S4Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    /// The device this layer reports.
    pub fn device(&self) -> Device {
        self.device
    }

    /// The state dimension `N`.
    pub fn state_size(&self) -> usize {
        self.config.d_state
    }

    /// The number of independent SSM channels `H`.
    pub fn channels(&self) -> usize {
        self.config.get_n_ssm()
    }

    /// Whether a usable discrete-time pair is cached.
    pub fn is_discretized(&self) -> bool {
        self.a_bar.is_some() && self.b_bar.is_some()
    }

    /// How many times the discretisation has been rebuilt.
    ///
    /// A caller installing parameters can assert this moved, which is what
    /// distinguishes a real invalidation from a setter that quietly left the
    /// previous `Ā`/`B̄` in place.
    pub fn discretization_generation(&self) -> u64 {
        self.discretization_generation
    }

    // --- read access -------------------------------------------------------

    /// Real part of the state matrix `A`, shape `[N, N]`.
    pub fn a_real(&self) -> &Array2<f32> {
        &self.a_real
    }

    /// Imaginary part of the state matrix `A`, shape `[N, N]`.
    pub fn a_imag(&self) -> &Array2<f32> {
        &self.a_imag
    }

    /// Real part of the input vector `B`, shape `[N]`.
    pub fn b_real(&self) -> &Array1<f32> {
        &self.b_real
    }

    /// Imaginary part of the input vector `B`, shape `[N]`.
    pub fn b_imag(&self) -> &Array1<f32> {
        &self.b_imag
    }

    /// Real part of the output vector `C`, shape `[N]`.
    pub fn c_real(&self) -> &Array1<f32> {
        &self.c_real
    }

    /// Imaginary part of the output vector `C`, shape `[N]`.
    pub fn c_imag(&self) -> &Array1<f32> {
        &self.c_imag
    }

    /// The skip-connection vector `D`, shape `[H]`.
    pub fn d(&self) -> &Array1<f32> {
        &self.d
    }

    /// The per-channel timestep `Δ`, shape `[H]`.
    pub fn dt(&self) -> &Array1<f32> {
        &self.dt
    }

    // --- write access ------------------------------------------------------
    //
    // Every setter checks the shape, installs the values and re-discretises.

    /// Install the real part of `A`.
    ///
    /// # Errors
    ///
    /// Fails when the shape is not `[N, N]`, or when the new parameters have no
    /// discrete-time equivalent.
    pub fn set_a_real(&mut self, values: Array2<f32>) -> Result<()> {
        self.check_matrix("a_real", &values)?;
        self.a_real = values;
        self.discretize()
    }

    /// Install the imaginary part of `A`.
    ///
    /// # Errors
    ///
    /// See [`S4Layer::set_a_real`].
    pub fn set_a_imag(&mut self, values: Array2<f32>) -> Result<()> {
        self.check_matrix("a_imag", &values)?;
        self.a_imag = values;
        self.discretize()
    }

    /// Install the real part of `B`.
    ///
    /// # Errors
    ///
    /// Fails when the length is not `N`, or when the new parameters have no
    /// discrete-time equivalent.
    pub fn set_b_real(&mut self, values: Array1<f32>) -> Result<()> {
        self.check_state_vector("b_real", &values)?;
        self.b_real = values;
        self.discretize()
    }

    /// Install the imaginary part of `B`.
    ///
    /// # Errors
    ///
    /// See [`S4Layer::set_b_real`].
    pub fn set_b_imag(&mut self, values: Array1<f32>) -> Result<()> {
        self.check_state_vector("b_imag", &values)?;
        self.b_imag = values;
        self.discretize()
    }

    /// Install the real part of `C`.
    ///
    /// `C` is an output map and does not enter the discretisation, but the cache
    /// is still rebuilt so that every setter has one behaviour rather than two.
    ///
    /// # Errors
    ///
    /// Fails when the length is not `N`.
    pub fn set_c_real(&mut self, values: Array1<f32>) -> Result<()> {
        self.check_state_vector("c_real", &values)?;
        self.c_real = values;
        self.discretize()
    }

    /// Install the imaginary part of `C`.
    ///
    /// # Errors
    ///
    /// See [`S4Layer::set_c_real`].
    pub fn set_c_imag(&mut self, values: Array1<f32>) -> Result<()> {
        self.check_state_vector("c_imag", &values)?;
        self.c_imag = values;
        self.discretize()
    }

    /// Install the skip-connection vector `D`.
    ///
    /// # Errors
    ///
    /// Fails when the length is not `H`.
    pub fn set_d(&mut self, values: Array1<f32>) -> Result<()> {
        self.check_channel_vector("d", &values)?;
        self.d = values;
        self.discretize()
    }

    /// Install the per-channel timestep `Δ`.
    ///
    /// # Errors
    ///
    /// Fails when the length is not `H`, or when the new timestep has no
    /// discrete-time equivalent.
    pub fn set_dt(&mut self, values: Array1<f32>) -> Result<()> {
        self.check_channel_vector("dt", &values)?;
        self.dt = values;
        self.discretize()
    }

    fn check_matrix(&self, name: &str, values: &Array2<f32>) -> Result<()> {
        let n = self.config.d_state;
        if values.shape() != [n, n] {
            return Err(runtime_error(format!(
                "S4Layer::set_{name}: expected shape [{n}, {n}], got {:?}",
                values.shape()
            )));
        }
        Ok(())
    }

    fn check_state_vector(&self, name: &str, values: &Array1<f32>) -> Result<()> {
        let n = self.config.d_state;
        if values.len() != n {
            return Err(runtime_error(format!(
                "S4Layer::set_{name}: expected {n} value(s), got {}",
                values.len()
            )));
        }
        Ok(())
    }

    fn check_channel_vector(&self, name: &str, values: &Array1<f32>) -> Result<()> {
        let h = self.config.get_n_ssm();
        if values.len() != h {
            return Err(runtime_error(format!(
                "S4Layer::set_{name}: expected {h} value(s), got {}",
                values.len()
            )));
        }
        Ok(())
    }

    /// Rebuild the discrete-time pair from the current continuous parameters.
    ///
    /// # Errors
    ///
    /// Fails when the configured method has no solution for these parameters.
    fn discretize(&mut self) -> Result<()> {
        // Drop the stale pair *first*: if the rebuild below fails, the layer is
        // left un-discretised and refuses to run, rather than silently keeping a
        // discretisation that belongs to the previous parameters.
        self.a_bar = None;
        self.b_bar = None;

        let method = Discretization::from_name(&self.config.discretization);
        let dt_avg = self.dt.mean().unwrap_or(self.config.dt);
        let n = self.config.d_state;

        // `A` and `B` are complex, and the discretisation routines are real. The
        // complex linear map `z ↦ (P + iQ)z` is exactly the real map
        // `[x; y] ↦ [[P, −Q], [Q, P]] [x; y]`, and that embedding is a ring
        // homomorphism — so the real matrix exponential, the real inverse and the
        // real augmented-exponential trick applied to the 2N×2N block form give
        // the *complex* results, exactly. Discretising the real and imaginary
        // parts independently (what a previous revision effectively did, by
        // discretising `a_real` and then scaling `a_imag` by Δ) is only correct
        // when `a_imag` is zero, which stops being true the moment a checkpoint
        // installs one.
        let mut block = Array2::<f32>::zeros((2 * n, 2 * n));
        let mut stacked = Array1::<f32>::zeros(2 * n);
        for i in 0..n {
            for j in 0..n {
                block[[i, j]] = self.a_real[[i, j]];
                block[[i, n + j]] = -self.a_imag[[i, j]];
                block[[n + i, j]] = self.a_imag[[i, j]];
                block[[n + i, n + j]] = self.a_real[[i, j]];
            }
            stacked[i] = self.b_real[i];
            stacked[n + i] = self.b_imag[i];
        }

        let (block_bar, stacked_bar) = method.discretize(&block, &stacked, dt_avg)?;

        let mut a_bar = Array2::<Complex64>::zeros((n, n));
        let mut b_bar = Array1::<Complex64>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                a_bar[[i, j]] = Complex64::new(
                    f64::from(block_bar[[i, j]]),
                    f64::from(block_bar[[n + i, j]]),
                );
            }
            b_bar[i] = Complex64::new(f64::from(stacked_bar[i]), f64::from(stacked_bar[n + i]));
        }

        self.a_bar = Some(a_bar);
        self.b_bar = Some(b_bar);
        self.discretization_generation = self.discretization_generation.wrapping_add(1);
        Ok(())
    }

    /// Run the state-space recurrence over `[channels, seq_len]` input.
    ///
    /// Each of the `H` channels carries its own state `x ∈ ℂ^N`:
    ///
    /// ```text
    /// x_h[t] = Ā x_h[t-1] + B̄ u_h[t]
    /// y_h[t] = Re(Cᴴ x_h[t]) + D[h] u_h[t]
    /// ```
    ///
    /// A previous revision collapsed every channel into `u_t.mean()`, used
    /// `D[0]` for all of them and broadcast one scalar `y_t` across every row of
    /// the output — so the layer could not represent a per-channel response at
    /// all. It was also unreachable: the block that called it returned its input
    /// untouched because `Ā` was never built.
    ///
    /// # Errors
    ///
    /// Fails when the layer is not discretised, or when the input's channel
    /// count does not match `H`.
    pub fn apply(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let (channels, seq_len) = (input.nrows(), input.ncols());
        let expected = self.config.get_n_ssm();
        if channels != expected {
            return Err(runtime_error(format!(
                "S4Layer::apply: expected {expected} channel(s), got {channels}"
            )));
        }
        let a_bar = self.a_bar.as_ref().ok_or_else(|| runtime_error("S4 layer not discretized"))?;
        let b_bar = self.b_bar.as_ref().ok_or_else(|| runtime_error("S4 layer not discretized"))?;

        let n = self.config.d_state;
        let mut output = Array2::<f32>::zeros((channels, seq_len));
        let mut state = vec![Complex64::new(0.0, 0.0); n];
        let mut next = vec![Complex64::new(0.0, 0.0); n];

        for channel in 0..channels {
            for value in state.iter_mut() {
                *value = Complex64::new(0.0, 0.0);
            }
            for t in 0..seq_len {
                let u = f64::from(input[[channel, t]]);
                for i in 0..n {
                    let mut accumulated = b_bar[i] * u;
                    for (j, previous) in state.iter().enumerate() {
                        accumulated += a_bar[[i, j]] * previous;
                    }
                    next[i] = accumulated;
                }
                state.copy_from_slice(&next);

                // y = Re(Cᴴ x) + D[h] u
                let mut y = 0.0_f64;
                for i in 0..n {
                    y += f64::from(self.c_real[i]) * state[i].re
                        - f64::from(self.c_imag[i]) * state[i].im;
                }
                y += f64::from(self.d[channel]) * u;
                output[[channel, t]] = y as f32;
            }
        }

        Ok(output)
    }

    /// Total learnable parameters in this layer.
    pub fn parameter_count(&self) -> usize {
        self.a_real.len()
            + self.a_imag.len()
            + self.b_real.len()
            + self.b_imag.len()
            + self.c_real.len()
            + self.c_imag.len()
            + self.d.len()
            + self.dt.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> S4Config {
        S4Config {
            d_model: 4,
            d_state: 3,
            n_layer: 1,
            vocab_size: 16,
            n_ssm: Some(4),
            dt: 0.05,
            ..S4Config::default()
        }
    }

    fn ramp_input(channels: usize, seq_len: usize) -> Array2<f32> {
        let mut input = Array2::<f32>::zeros((channels, seq_len));
        for channel in 0..channels {
            for t in 0..seq_len {
                input[[channel, t]] = ((channel * seq_len + t) as f32 * 0.31).sin();
            }
        }
        input
    }

    #[test]
    fn a_new_layer_is_already_discretized() {
        let layer = S4Layer::new(&tiny_config()).expect("layer must build");
        assert!(
            layer.is_discretized(),
            "a layer that is not discretised cannot run, and `forward` takes &self"
        );
        assert_eq!(layer.discretization_generation(), 1);
    }

    /// Every setter rebuilds the discretisation cache.
    ///
    /// Regression test for the silent-wrong-answer class the S4 loader was
    /// blocked on: installing parameters while leaving a stale `Ā`/`B̄` in place
    /// would leave the model computing with the *pre-load* discretisation while
    /// reporting a successful load.
    #[test]
    fn every_setter_rebuilds_the_discretization_cache() {
        let config = tiny_config();
        let n = config.d_state;
        let h = config.get_n_ssm();
        let mut layer = S4Layer::new(&config).expect("layer must build");

        let mut generation = layer.discretization_generation();
        let bump = |layer: &S4Layer, label: &str, generation: &mut u64| {
            assert!(
                layer.discretization_generation() > *generation,
                "set_{label} must invalidate and rebuild the discretisation"
            );
            *generation = layer.discretization_generation();
        };

        layer.set_a_real(Array2::<f32>::eye(n) * -1.5).expect("a_real must install");
        bump(&layer, "a_real", &mut generation);
        layer.set_a_imag(Array2::<f32>::eye(n) * 0.25).expect("a_imag must install");
        bump(&layer, "a_imag", &mut generation);
        layer.set_b_real(Array1::<f32>::from_elem(n, 0.5)).expect("b_real must install");
        bump(&layer, "b_real", &mut generation);
        layer.set_b_imag(Array1::<f32>::from_elem(n, 0.1)).expect("b_imag must install");
        bump(&layer, "b_imag", &mut generation);
        layer.set_c_real(Array1::<f32>::from_elem(n, 0.7)).expect("c_real must install");
        bump(&layer, "c_real", &mut generation);
        layer.set_c_imag(Array1::<f32>::from_elem(n, 0.2)).expect("c_imag must install");
        bump(&layer, "c_imag", &mut generation);
        layer.set_d(Array1::<f32>::from_elem(h, 2.0)).expect("d must install");
        bump(&layer, "d", &mut generation);
        layer.set_dt(Array1::<f32>::from_elem(h, 0.2)).expect("dt must install");
        bump(&layer, "dt", &mut generation);
    }

    /// The rebuilt discretisation is the one the recurrence actually uses.
    ///
    /// `A` and `Δ` only reach the output through `Ā`/`B̄`, so if the cache were
    /// not rebuilt the two outputs below would be identical.
    #[test]
    fn changing_a_discretization_input_changes_the_output() {
        let config = tiny_config();
        let n = config.d_state;
        let h = config.get_n_ssm();
        let input = ramp_input(h, 6);

        let mut layer = S4Layer::new(&config).expect("layer must build");
        let before = layer.apply(&input).expect("recurrence must run");

        layer.set_a_real(Array2::<f32>::eye(n) * -3.0).expect("a_real must install");
        let after_a = layer.apply(&input).expect("recurrence must run");
        assert_ne!(
            before.iter().copied().collect::<Vec<f32>>(),
            after_a.iter().copied().collect::<Vec<f32>>(),
            "changing A must change the output through the rebuilt discretisation"
        );

        layer.set_dt(Array1::<f32>::from_elem(h, 0.5)).expect("dt must install");
        let after_dt = layer.apply(&input).expect("recurrence must run");
        assert_ne!(
            after_a.iter().copied().collect::<Vec<f32>>(),
            after_dt.iter().copied().collect::<Vec<f32>>(),
            "changing the timestep must change the output"
        );
    }

    /// Each channel has its own state and its own `D`.
    ///
    /// The previous recurrence averaged all channels into one scalar and
    /// broadcast one output row, so this could not hold.
    #[test]
    fn channels_are_independent_and_use_their_own_skip_weight() {
        let config = tiny_config();
        let h = config.get_n_ssm();
        let mut layer = S4Layer::new(&config).expect("layer must build");
        let mut skip = Array1::<f32>::zeros(h);
        skip[0] = 1.0;
        layer.set_d(skip).expect("d must install");

        let mut input = Array2::<f32>::zeros((h, 4));
        // Drive channel 1 only.
        for t in 0..4 {
            input[[1, t]] = 1.0;
        }
        let output = layer.apply(&input).expect("recurrence must run");

        for t in 0..4 {
            assert_eq!(
                output[[0, t]],
                0.0,
                "an unexcited channel must stay at zero, not pick up its neighbour's input"
            );
        }
        assert!(
            (0..4).any(|t| output[[1, t]].abs() > 1e-6),
            "the driven channel must respond"
        );
        assert!(
            (0..4).any(|t| output[[2, t]] == 0.0),
            "channels above the driven one must also stay quiet"
        );
    }

    /// The recurrence is causal: a later input cannot change an earlier output.
    #[test]
    fn the_recurrence_is_causal() {
        let config = tiny_config();
        let h = config.get_n_ssm();
        let layer = S4Layer::new(&config).expect("layer must build");
        let base = ramp_input(h, 5);
        let mut altered = base.clone();
        for channel in 0..h {
            altered[[channel, 4]] += 3.0;
        }

        let first = layer.apply(&base).expect("recurrence must run");
        let second = layer.apply(&altered).expect("recurrence must run");
        for channel in 0..h {
            for t in 0..4 {
                assert!(
                    (first[[channel, t]] - second[[channel, t]]).abs() < 1e-6,
                    "changing the last input must not move output ({channel}, {t})"
                );
            }
        }
    }

    /// The imaginary part of `A` reaches the recurrence through the complex
    /// discretisation, not through a scaled copy of itself.
    #[test]
    fn the_imaginary_state_matrix_participates() {
        let config = tiny_config();
        let n = config.d_state;
        let h = config.get_n_ssm();
        let input = ramp_input(h, 5);

        let mut layer = S4Layer::new(&config).expect("layer must build");
        layer.set_c_imag(Array1::<f32>::from_elem(n, 0.4)).expect("c_imag must install");
        let before = layer.apply(&input).expect("recurrence must run");

        layer.set_a_imag(Array2::<f32>::eye(n) * 1.75).expect("a_imag must install");
        let after = layer.apply(&input).expect("recurrence must run");
        assert_ne!(
            before.iter().copied().collect::<Vec<f32>>(),
            after.iter().copied().collect::<Vec<f32>>(),
            "a non-zero imaginary A must change the output"
        );
    }

    /// The complex block embedding is exact: with a purely imaginary diagonal
    /// `A = i·ω` and zero-order hold, `Ā` must be `diag(exp(i·ω·Δ))`.
    #[test]
    fn the_complex_discretization_matches_the_closed_form() {
        let mut config = tiny_config();
        config.discretization = "zoh".to_string();
        config.d_state = 2;
        config.dt = 0.3;
        let mut layer = S4Layer::new(&config).expect("layer must build");

        let omega = 1.25_f32;
        layer.set_a_real(Array2::<f32>::zeros((2, 2))).expect("a_real must install");
        layer.set_a_imag(Array2::<f32>::eye(2) * omega).expect("a_imag must install");

        let a_bar = layer.a_bar.as_ref().expect("the layer is discretised");
        let angle = f64::from(omega) * f64::from(config.dt);
        for index in 0..2 {
            assert!(
                (a_bar[[index, index]].re - angle.cos()).abs() < 1e-4,
                "Ā[{index},{index}].re = {} vs cos = {}",
                a_bar[[index, index]].re,
                angle.cos()
            );
            assert!(
                (a_bar[[index, index]].im - angle.sin()).abs() < 1e-4,
                "Ā[{index},{index}].im = {} vs sin = {}",
                a_bar[[index, index]].im,
                angle.sin()
            );
        }
    }

    /// A setter with the wrong shape is refused rather than silently reshaped.
    #[test]
    fn a_setter_refuses_the_wrong_shape() {
        let config = tiny_config();
        let mut layer = S4Layer::new(&config).expect("layer must build");
        layer
            .set_b_real(Array1::<f32>::zeros(config.d_state + 1))
            .expect_err("a B of the wrong length is not this layer's B");
        layer
            .set_d(Array1::<f32>::zeros(config.get_n_ssm() + 1))
            .expect_err("a D of the wrong length is not this layer's D");
        layer
            .set_a_real(Array2::<f32>::zeros((2, 3)))
            .expect_err("a non-square A is not this layer's A");
    }

    /// An input with the wrong channel count is refused.
    #[test]
    fn apply_refuses_the_wrong_channel_count() {
        let config = tiny_config();
        let layer = S4Layer::new(&config).expect("layer must build");
        layer
            .apply(&Array2::<f32>::zeros((config.get_n_ssm() + 1, 3)))
            .expect_err("an input with too many channels is not this layer's input");
    }
}
