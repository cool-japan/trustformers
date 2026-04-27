//! # Sophia Optimizer Module
//!
//! Contains both the original Sophia implementation (legacy) and the new advanced
//! Sophia optimizer with diagonal Hessian estimation via Hutchinson estimator.

pub mod legacy;

pub use legacy::Sophia;

// Re-export legacy SophiaConfig with disambiguation alias so it doesn't
// clash with the new SophiaConfig defined below.
use legacy::SophiaConfig as LegacySophiaConfig;
pub use legacy::SophiaConfig as SophiaLegacyConfig;

// Ensure LegacySophiaConfig is used (it is re-exported, but suppress dead_code)
#[allow(dead_code)]
fn _use_legacy(_: &LegacySophiaConfig) {}

use trustformers_core::errors::TrustformersError;

/// Error type for the advanced Sophia optimizer.
#[derive(Debug, thiserror::Error)]
pub enum SophiaError {
    /// Parameter and gradient vectors have different lengths.
    #[error("param/grad length mismatch: param={param} grad={grad}")]
    LengthMismatch { param: usize, grad: usize },
    /// No state has been initialised for the given index.
    #[error("no state for parameter index {0}")]
    StateNotInitialised(usize),
    /// Unexpected numerical issue.
    #[error("numerical error: {0}")]
    NumericalError(String),
}

impl From<SophiaError> for TrustformersError {
    fn from(e: SophiaError) -> Self {
        TrustformersError::invalid_operation(e.to_string())
    }
}

/// Configuration for the advanced Sophia optimizer (Liu et al., 2023).
///
/// Reference: "Sophia: A Scalable Stochastic Second-order Optimizer for
/// Language Model Pre-training"
#[derive(Debug, Clone)]
pub struct SophiaConfig {
    /// Learning rate (default 2e-4).
    pub lr: f64,
    /// (β1, β2) — gradient and Hessian EMA coefficients.
    /// Default: (0.965, 0.99).
    pub betas: (f64, f64),
    /// Small constant for numerical stability (default 1e-12).
    pub eps: f64,
    /// Decoupled weight decay (default 0.1).
    pub weight_decay: f64,
    /// Clipping threshold ρ (default 0.04).
    pub rho: f64,
    /// Hessian update interval k (default 10).
    pub hessian_update_interval: usize,
}

impl Default for SophiaConfig {
    fn default() -> Self {
        Self {
            lr: 2e-4,
            betas: (0.965, 0.99),
            eps: 1e-12,
            weight_decay: 0.1,
            rho: 0.04,
            hessian_update_interval: 10,
        }
    }
}

/// Per-parameter state for the advanced Sophia optimizer.
#[derive(Debug, Clone)]
pub struct SophiaParamState {
    /// Number of update steps.
    pub step: u64,
    /// First moment (gradient EMA).
    pub m: Vec<f32>,
    /// Diagonal Hessian estimate.
    pub h: Vec<f32>,
    /// Gradient buffer accumulated for the next Hessian update.
    pub grad_buffer: Vec<f32>,
}

impl SophiaParamState {
    /// Create a zeroed state for a parameter of `size` elements.
    pub fn new(size: usize) -> Self {
        Self {
            step: 0,
            m: vec![0.0_f32; size],
            h: vec![0.0_f32; size],
            grad_buffer: vec![0.0_f32; size],
        }
    }
}

/// Diagonal Hessian estimate via (simplified) Hutchinson estimator.
///
/// For a squared-loss model the exact diagonal is `grad²`.  For general
/// models this is the Hutchinson approximation `(grad * u)²` where `u` is a
/// Rademacher vector (here we use `u = 1` everywhere for simplicity, which
/// reduces to `grad²`).
///
/// # Arguments
///
/// * `grad` – gradient vector.
/// * `u`    – Rademacher random vector (each element ∈ {−1, +1}).
///   When `u` is all-ones this returns the exact squared gradient.
///
/// # Returns
///
/// A `Vec<f32>` with `h_i = (grad_i * u_i)²`.
pub fn hutchinson_hessian_estimate(grad: &[f32], u: &[f32]) -> Vec<f32> {
    grad.iter()
        .zip(u.iter())
        .map(|(&g, &ui)| {
            let val = g * ui;
            val * val
        })
        .collect()
}

/// Perform one Sophia update for a single parameter.
///
/// # Arguments
///
/// * `param`           – mutable parameter slice.
/// * `grad`            – gradient slice (same length).
/// * `state`           – mutable per-parameter state.
/// * `config`          – optimizer configuration.
/// * `update_hessian`  – if `true`, refresh the Hessian estimate this step.
///
/// # Errors
///
/// Returns [`SophiaError`] on length mismatches.
pub fn sophia_update(
    param: &mut [f32],
    grad: &[f32],
    state: &mut SophiaParamState,
    config: &SophiaConfig,
    update_hessian: bool,
) -> Result<(), SophiaError> {
    let size = param.len();
    if grad.len() != size {
        return Err(SophiaError::LengthMismatch {
            param: size,
            grad: grad.len(),
        });
    }

    state.step += 1;

    let beta1 = config.betas.0 as f32;
    let beta2 = config.betas.1 as f32;
    let eps = config.eps as f32;
    let rho = config.rho as f32;
    let lr = config.lr as f32;

    // --- Momentum update: m = β1 * m + (1 − β1) * grad -------------------
    for (m, &g) in state.m.iter_mut().zip(grad.iter()) {
        *m = beta1 * *m + (1.0 - beta1) * g;
    }

    // --- Hessian update (every k steps) ------------------------------------
    if update_hessian {
        // Simplified Hutchinson: h_new_i = grad_i²
        // Store current gradient into buffer for accumulation
        for (buf, &g) in state.grad_buffer.iter_mut().zip(grad.iter()) {
            *buf = g;
        }
        // Update Hessian EMA: h = β2 * h + (1 − β2) * grad²
        for (h, &g) in state.h.iter_mut().zip(state.grad_buffer.iter()) {
            let h_new = g * g;
            *h = beta2 * *h + (1.0 - beta2) * h_new;
        }
    }

    // --- Compute clipped update and apply ----------------------------------
    for ((p, &m), &h) in param.iter_mut().zip(state.m.iter()).zip(state.h.iter()) {
        // Weight decay (decoupled)
        if config.weight_decay != 0.0 {
            *p -= lr * config.weight_decay as f32 * *p;
        }

        // Sophia update: clip( m / max(ρ * h, ε), ρ ) — we clip to ±ρ
        let denom = (rho * h).max(eps);
        let raw_update = m / denom;
        let clipped_update = raw_update.clamp(-rho, rho);

        *p -= lr * clipped_update;
    }

    Ok(())
}

/// Advanced Sophia optimizer with diagonal Hessian estimate.
///
/// Reference: "Sophia: A Scalable Stochastic Second-order Optimizer for
/// Language Model Pre-training" (Liu et al., 2023)
#[derive(Debug)]
pub struct SophiaOptimizer {
    /// Hyperparameter configuration.
    pub config: SophiaConfig,
    /// Per-parameter states.
    pub states: Vec<SophiaParamState>,
}

impl SophiaOptimizer {
    /// Create a new optimizer with the given configuration.
    pub fn new(config: SophiaConfig) -> Self {
        Self {
            config,
            states: Vec::new(),
        }
    }

    /// Register a new parameter of `size` elements.
    pub fn add_param(&mut self, size: usize) {
        self.states.push(SophiaParamState::new(size));
    }

    /// Perform one update step across all registered parameters.
    ///
    /// The Hessian is refreshed automatically every
    /// `config.hessian_update_interval` steps (counting from step 1).
    ///
    /// # Errors
    ///
    /// Returns [`SophiaError`] on any dimension mismatch.
    pub fn step(&mut self, params: &mut [Vec<f32>], grads: &[Vec<f32>]) -> Result<(), SophiaError> {
        let num_states = self.states.len();
        let hessian_interval = self.config.hessian_update_interval as u64;
        let lr = self.config.lr;
        let weight_decay = self.config.weight_decay;
        let betas = self.config.betas;
        let eps = self.config.eps;
        let rho = self.config.rho;

        for (idx, ((param, grad), state)) in
            params.iter_mut().zip(grads.iter()).zip(self.states.iter_mut()).enumerate()
        {
            if idx >= num_states {
                return Err(SophiaError::StateNotInitialised(idx));
            }
            // Determine if Hessian should be updated this step
            let next_step = state.step + 1;
            let update_hessian = next_step % hessian_interval == 0;

            // Inline config copy to avoid borrow conflict
            let local_config = SophiaConfig {
                lr,
                betas,
                eps,
                weight_decay,
                rho,
                hessian_update_interval: hessian_interval as usize,
            };
            sophia_update(param, grad, state, &local_config, update_hessian)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // -----------------------------------------------------------------------
    // 1. Config defaults
    // -----------------------------------------------------------------------
    #[test]
    fn test_sophia_config_defaults() {
        let cfg = SophiaConfig::default();
        assert_relative_eq!(cfg.lr, 2e-4);
        assert_relative_eq!(cfg.betas.0, 0.965);
        assert_relative_eq!(cfg.betas.1, 0.99);
        assert_relative_eq!(cfg.eps, 1e-12);
        assert_relative_eq!(cfg.weight_decay, 0.1);
        assert_relative_eq!(cfg.rho, 0.04);
        assert_eq!(cfg.hessian_update_interval, 10);
    }

    // -----------------------------------------------------------------------
    // 2. Hessian estimate: simplified grad²
    // -----------------------------------------------------------------------
    #[test]
    fn test_hutchinson_hessian_grad_squared() {
        let grad = vec![2.0_f32, -3.0, 0.5];
        let u = vec![1.0_f32; 3]; // Rademacher = all +1 → exact grad²
        let h = hutchinson_hessian_estimate(&grad, &u);
        assert_relative_eq!(h[0], 4.0, epsilon = 1e-6);
        assert_relative_eq!(h[1], 9.0, epsilon = 1e-6);
        assert_relative_eq!(h[2], 0.25, epsilon = 1e-6);
    }

    // -----------------------------------------------------------------------
    // 3. Hessian estimate with Rademacher ±1
    // -----------------------------------------------------------------------
    #[test]
    fn test_hutchinson_hessian_rademacher() {
        let grad = vec![1.0_f32, -1.0];
        let u_pos = vec![1.0_f32; 2];
        let u_neg = vec![-1.0_f32; 2];
        let h_pos = hutchinson_hessian_estimate(&grad, &u_pos);
        let h_neg = hutchinson_hessian_estimate(&grad, &u_neg);
        // (grad * u)² is the same for +1 and -1
        assert_relative_eq!(h_pos[0], h_neg[0], epsilon = 1e-6);
        assert_relative_eq!(h_pos[1], h_neg[1], epsilon = 1e-6);
    }

    // -----------------------------------------------------------------------
    // 4. Momentum update
    // -----------------------------------------------------------------------
    #[test]
    fn test_momentum_update() {
        let cfg = SophiaConfig {
            lr: 0.0,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut state = SophiaParamState::new(2);
        let mut param = vec![0.0_f32; 2];
        let grad = vec![1.0_f32; 2];

        sophia_update(&mut param, &grad, &mut state, &cfg, false).expect("update failed");
        // m = (1 - β1) * 1.0 = (1 - 0.965) * 1.0 = 0.035
        let expected_m = (1.0 - 0.965_f32) * 1.0;
        assert_relative_eq!(state.m[0], expected_m, epsilon = 1e-5);
        assert_relative_eq!(state.m[1], expected_m, epsilon = 1e-5);
    }

    // -----------------------------------------------------------------------
    // 5. Hessian EMA update
    // -----------------------------------------------------------------------
    #[test]
    fn test_hessian_ema_update() {
        let cfg = SophiaConfig {
            lr: 0.0,
            weight_decay: 0.0,
            ..SophiaConfig::default()
        };
        let mut state = SophiaParamState::new(1);
        let mut param = vec![0.0_f32];
        let grad = vec![2.0_f32]; // h_new = 4.0

        sophia_update(&mut param, &grad, &mut state, &cfg, true).expect("update failed");
        // h = 0.99 * 0 + 0.01 * 4 = 0.04
        let expected_h = (1.0 - 0.99_f32) * 4.0;
        assert_relative_eq!(state.h[0], expected_h, epsilon = 1e-5);
    }

    // -----------------------------------------------------------------------
    // 6. Conditional Hessian update — only every k steps
    // -----------------------------------------------------------------------
    #[test]
    fn test_conditional_hessian_update() {
        let cfg = SophiaConfig {
            hessian_update_interval: 3,
            lr: 0.0,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut optimizer = SophiaOptimizer::new(cfg);
        optimizer.add_param(2);

        let grad = vec![1.0_f32; 2];
        let mut params = vec![vec![0.0_f32; 2]];

        // Steps 1, 2 — Hessian should NOT be updated (step 3 would trigger)
        optimizer.step(&mut params, std::slice::from_ref(&grad)).expect("step 1 failed");
        let h_after_step1 = optimizer.states[0].h.clone();
        optimizer.step(&mut params, std::slice::from_ref(&grad)).expect("step 2 failed");
        let h_after_step2 = optimizer.states[0].h.clone();

        // At step 3, Hessian IS updated
        optimizer.step(&mut params, std::slice::from_ref(&grad)).expect("step 3 failed");
        let h_after_step3 = optimizer.states[0].h.clone();

        // H unchanged between step 1 and 2 (no update scheduled)
        assert_eq!(
            h_after_step1, h_after_step2,
            "H changed on step 2 unexpectedly"
        );

        // H changes on step 3
        assert_ne!(h_after_step2, h_after_step3, "H did not change on step 3");
    }

    // -----------------------------------------------------------------------
    // 7. Clipping threshold applied
    // -----------------------------------------------------------------------
    #[test]
    fn test_clipping_threshold() {
        let cfg = SophiaConfig {
            lr: 1.0,
            weight_decay: 0.0,
            ..SophiaConfig::default()
        };
        // Set h to be very small so that m/denom is very large without clipping
        let mut state = SophiaParamState::new(1);
        state.h[0] = 0.0; // denom = eps → update will be huge without clip
        state.m[0] = 1.0; // large momentum

        let mut param = vec![0.0_f32];
        let grad = vec![0.0_f32]; // no grad, only use existing momentum

        sophia_update(&mut param, &grad, &mut state, &cfg, false).expect("update failed");

        // Clip to ±ρ = ±0.04 then multiply by lr=1.0
        // param change should be at most ±0.04
        let change = (param[0]).abs();
        assert!(
            change <= cfg.rho as f32 + 1e-5,
            "update not clipped: change={change} rho={}",
            cfg.rho
        );
    }

    // -----------------------------------------------------------------------
    // 8. Weight decay applied
    // -----------------------------------------------------------------------
    #[test]
    fn test_weight_decay_sophia() {
        let cfg = SophiaConfig {
            weight_decay: 0.1,
            lr: 0.1,
            ..SophiaConfig::default()
        };
        let mut state = SophiaParamState::new(2);
        let initial_param = vec![1.0_f32; 2];
        let mut param = initial_param.clone();
        let grad = vec![0.0_f32; 2];

        sophia_update(&mut param, &grad, &mut state, &cfg, false).expect("update failed");

        for (p_new, p_old) in param.iter().zip(initial_param.iter()) {
            assert!(
                p_new.abs() < p_old.abs(),
                "weight decay did not reduce param"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 9. Single-step gradient descent direction
    // -----------------------------------------------------------------------
    #[test]
    fn test_single_step_direction() {
        let cfg = SophiaConfig {
            lr: 1e-2,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut state = SophiaParamState::new(3);
        let mut param = vec![0.5_f32; 3];
        let grad = vec![0.1_f32; 3]; // positive gradient

        let param_before = param.clone();
        sophia_update(&mut param, &grad, &mut state, &cfg, false).expect("update failed");

        for (p_new, p_old) in param.iter().zip(param_before.iter()) {
            assert!(
                p_new < p_old,
                "param did not decrease with positive gradient"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 10. Param/grad length mismatch returns error
    // -----------------------------------------------------------------------
    #[test]
    fn test_param_grad_length_mismatch() {
        let cfg = SophiaConfig::default();
        let mut state = SophiaParamState::new(3);
        let mut param = vec![0.0_f32; 3];
        let wrong_grad = vec![0.0_f32; 5];

        let result = sophia_update(&mut param, &wrong_grad, &mut state, &cfg, false);
        assert!(result.is_err());
        matches!(result.unwrap_err(), SophiaError::LengthMismatch { .. });
    }

    // -----------------------------------------------------------------------
    // 11. Convergence test on quadratic
    // -----------------------------------------------------------------------
    #[test]
    fn test_convergence_quadratic() {
        // Minimise f(x) = x² / 2, gradient = x.
        //
        // Sophia's update: param -= lr * clip(m / max(rho * h, eps), rho).
        // Maximum step size per iteration = lr * rho.
        // From param=1.0 to < 0.05 we need at least 1.0/(lr*rho) steps at the clip limit.
        // With lr=0.1, rho=1.0 the max step is 0.1 ⇒ ~10 steps minimum; 2000 is ample.
        let cfg = SophiaConfig {
            lr: 0.1,
            betas: (0.965, 0.99),
            eps: 1e-12,
            weight_decay: 0.0,
            rho: 1.0, // large rho → smaller effective clipping → larger steps
            hessian_update_interval: 1, // update every step for the test
        };
        let mut state = SophiaParamState::new(1);
        let mut param = vec![1.0_f32];

        for _ in 0..2000 {
            let grad = param.clone();
            sophia_update(&mut param, &grad, &mut state, &cfg, true).expect("update failed");
        }

        assert!(
            param[0].abs() < 0.05,
            "Sophia did not converge on quadratic: final param = {}",
            param[0]
        );
    }

    // -----------------------------------------------------------------------
    // 12. SophiaOptimizer step count
    // -----------------------------------------------------------------------
    #[test]
    fn test_sophia_optimizer_step_count() {
        let cfg = SophiaConfig::default();
        let mut optimizer = SophiaOptimizer::new(cfg);
        optimizer.add_param(4);
        optimizer.add_param(2);

        let mut params = vec![vec![0.0_f32; 4], vec![0.0_f32; 2]];
        let grads = vec![vec![0.01_f32; 4], vec![0.01_f32; 2]];

        optimizer.step(&mut params, &grads).expect("step 1 failed");
        optimizer.step(&mut params, &grads).expect("step 2 failed");

        assert_eq!(optimizer.states[0].step, 2);
        assert_eq!(optimizer.states[1].step, 2);
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sophia_g_hessian_all_ones_u() {
        let grad = vec![3.0_f32, -2.0_f32, 0.5_f32];
        let u = vec![1.0_f32; 3];
        let h = hutchinson_hessian_estimate(&grad, &u);
        assert_relative_eq!(h[0], 9.0, epsilon = 1e-6);
        assert_relative_eq!(h[1], 4.0, epsilon = 1e-6);
        assert_relative_eq!(h[2], 0.25, epsilon = 1e-6);
    }

    #[test]
    fn test_sophia_g_hessian_negative_u_same_as_positive() {
        let grad = vec![1.5_f32, -0.8_f32];
        let u_pos = vec![1.0_f32; 2];
        let u_neg = vec![-1.0_f32; 2];
        let h_pos = hutchinson_hessian_estimate(&grad, &u_pos);
        let h_neg = hutchinson_hessian_estimate(&grad, &u_neg);
        for (p, n) in h_pos.iter().zip(h_neg.iter()) {
            assert_relative_eq!(p, n, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sophia_h_initial_zero() {
        let state = SophiaParamState::new(5);
        assert!(
            state.h.iter().all(|&x| x == 0.0),
            "h should be all zeros initially"
        );
    }

    #[test]
    fn test_sophia_m_initial_zero() {
        let state = SophiaParamState::new(5);
        assert!(
            state.m.iter().all(|&x| x == 0.0),
            "m should be all zeros initially"
        );
    }

    #[test]
    fn test_sophia_step_increments_count() {
        let cfg = SophiaConfig::default();
        let mut optimizer = SophiaOptimizer::new(cfg);
        optimizer.add_param(3);
        let mut params = vec![vec![0.5_f32; 3]];
        let grads = vec![vec![0.01_f32; 3]];
        optimizer.step(&mut params, &grads).expect("step 1 failed");
        assert_eq!(optimizer.states[0].step, 1);
        optimizer.step(&mut params, &grads).expect("step 2 failed");
        assert_eq!(optimizer.states[0].step, 2);
    }

    #[test]
    fn test_sophia_hessian_update_only_on_interval() {
        let cfg = SophiaConfig {
            hessian_update_interval: 5,
            lr: 0.0,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut optimizer = SophiaOptimizer::new(cfg);
        optimizer.add_param(2);
        let grad = vec![1.0_f32; 2];
        let mut params = vec![vec![0.0_f32; 2]];
        // Steps 1-4: h should remain zero (hessian not updated)
        for _ in 0..4 {
            optimizer.step(&mut params, std::slice::from_ref(&grad)).expect("step failed");
        }
        assert!(
            optimizer.states[0].h.iter().all(|&x| x == 0.0),
            "h should remain zero through steps 1-4"
        );
        // Step 5: h should be updated (next_step=5, 5%5==0)
        optimizer.step(&mut params, std::slice::from_ref(&grad)).expect("step 5 failed");
        assert!(
            optimizer.states[0].h.iter().all(|&x| x > 0.0),
            "h should be updated at step 5"
        );
    }

    #[test]
    fn test_sophia_momentum_decays_to_zero() {
        let cfg = SophiaConfig {
            lr: 0.0,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut state = SophiaParamState::new(1);
        state.m[0] = 1.0;
        let mut param = vec![0.0_f32];
        let grad = vec![0.0_f32];
        for _ in 0..200 {
            sophia_update(&mut param, &grad, &mut state, &cfg, false).expect("update failed");
        }
        assert!(
            state.m[0].abs() < 0.01,
            "momentum should decay toward 0 with zero grad: {}",
            state.m[0]
        );
    }

    #[test]
    fn test_sophia_weight_decay_reduces_magnitude() {
        let cfg = SophiaConfig {
            weight_decay: 0.1,
            lr: 0.01,
            ..Default::default()
        };
        let mut state = SophiaParamState::new(2);
        let mut param = vec![1.0_f32; 2];
        let grad = vec![0.0_f32; 2];
        for _ in 0..10 {
            sophia_update(&mut param, &grad, &mut state, &cfg, false).expect("update failed");
        }
        assert!(
            param[0] < 1.0,
            "weight decay should reduce positive param: {}",
            param[0]
        );
    }

    #[test]
    fn test_sophia_rho_clipping_small_h() {
        let cfg = SophiaConfig {
            lr: 1.0,
            weight_decay: 0.0,
            rho: 0.04,
            ..Default::default()
        };
        let mut state = SophiaParamState::new(1);
        state.h[0] = 0.0;
        state.m[0] = 1.0;
        let mut param = vec![0.0_f32];
        let grad = vec![0.0_f32];
        sophia_update(&mut param, &grad, &mut state, &cfg, false).expect("update failed");
        assert!(
            param[0].abs() <= 0.04 + 1e-5,
            "update should be clipped to rho=0.04: change={}",
            param[0].abs()
        );
    }

    #[test]
    fn test_sophia_large_h_reduces_update() {
        let cfg = SophiaConfig {
            lr: 1.0,
            weight_decay: 0.0,
            rho: 1.0,
            ..Default::default()
        };

        let mut state_small_h = SophiaParamState::new(1);
        state_small_h.m[0] = 0.01;
        state_small_h.h[0] = 0.01;
        let mut p_small = vec![0.0_f32];
        sophia_update(&mut p_small, &[0.0_f32], &mut state_small_h, &cfg, false)
            .expect("small h update failed");

        let mut state_large_h = SophiaParamState::new(1);
        state_large_h.m[0] = 0.01;
        state_large_h.h[0] = 100.0;
        let mut p_large = vec![0.0_f32];
        sophia_update(&mut p_large, &[0.0_f32], &mut state_large_h, &cfg, false)
            .expect("large h update failed");

        assert!(
            p_large[0].abs() < p_small[0].abs(),
            "large h should give smaller update: small_h_change={}, large_h_change={}",
            p_small[0].abs(),
            p_large[0].abs()
        );
    }

    #[test]
    fn test_sophia_update_direction_correct() {
        let cfg = SophiaConfig {
            lr: 0.01,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut state = SophiaParamState::new(3);
        let mut param = vec![1.0_f32; 3];
        let grad = vec![0.1_f32; 3];
        let before = param.clone();
        sophia_update(&mut param, &grad, &mut state, &cfg, false).expect("update failed");
        for (p_new, p_old) in param.iter().zip(before.iter()) {
            assert!(p_new < p_old, "positive grad should decrease param");
        }
    }

    #[test]
    fn test_sophia_multi_param_independent() {
        let cfg = SophiaConfig::default();
        let mut optimizer = SophiaOptimizer::new(cfg);
        optimizer.add_param(3);
        optimizer.add_param(5);
        let mut params = vec![vec![1.0_f32; 3], vec![2.0_f32; 5]];
        let grads = vec![vec![0.1_f32; 3], vec![0.2_f32; 5]];
        optimizer.step(&mut params, &grads).expect("step failed");
        assert_eq!(optimizer.states[0].m.len(), 3);
        assert_eq!(optimizer.states[1].m.len(), 5);
        assert_eq!(optimizer.states[0].step, 1);
        assert_eq!(optimizer.states[1].step, 1);
    }

    #[test]
    fn test_sophia_length_mismatch_error() {
        let cfg = SophiaConfig::default();
        let mut state = SophiaParamState::new(4);
        let mut param = vec![0.0_f32; 4];
        let short_grad = vec![0.0_f32; 2];
        let result = sophia_update(&mut param, &short_grad, &mut state, &cfg, false);
        assert!(result.is_err());
        match result {
            Err(SophiaError::LengthMismatch { .. }) => {},
            other => panic!("Expected LengthMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_sophia_zero_lr_no_update() {
        let cfg = SophiaConfig {
            lr: 0.0,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut state = SophiaParamState::new(3);
        let mut param = vec![1.5_f32; 3];
        let original = param.clone();
        let grad = vec![1.0_f32; 3];
        sophia_update(&mut param, &grad, &mut state, &cfg, false).expect("update failed");
        for (p_new, p_old) in param.iter().zip(original.iter()) {
            assert_relative_eq!(*p_new, *p_old, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sophia_higher_beta1_smaller_momentum_from_zero() {
        // m = (1 - beta1) * grad from zero initial state
        // higher beta1 → smaller (1-beta1) → smaller m
        let grad = vec![1.0_f32; 2];

        let cfg_high = SophiaConfig {
            betas: (0.99, 0.99),
            lr: 0.0,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut state_high = SophiaParamState::new(2);
        let mut p_high = vec![0.0_f32; 2];
        sophia_update(&mut p_high, &grad, &mut state_high, &cfg_high, false)
            .expect("update failed");

        let cfg_low = SophiaConfig {
            betas: (0.5, 0.99),
            lr: 0.0,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut state_low = SophiaParamState::new(2);
        let mut p_low = vec![0.0_f32; 2];
        sophia_update(&mut p_low, &grad, &mut state_low, &cfg_low, false).expect("update failed");

        assert!(
            state_high.m[0] < state_low.m[0],
            "higher β1 gives smaller m from zero: high={}, low={}",
            state_high.m[0],
            state_low.m[0]
        );
    }

    #[test]
    fn test_sophia_hessian_buffer_stored() {
        let cfg = SophiaConfig::default();
        let mut state = SophiaParamState::new(3);
        let mut param = vec![0.5_f32; 3];
        let grad = vec![0.2_f32, -0.3_f32, 0.7_f32];
        sophia_update(&mut param, &grad, &mut state, &cfg, true).expect("update failed");
        for (buf, g) in state.grad_buffer.iter().zip(grad.iter()) {
            assert_relative_eq!(*buf, *g, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sophia_optimizer_step_count_three_params() {
        let cfg = SophiaConfig::default();
        let mut optimizer = SophiaOptimizer::new(cfg);
        optimizer.add_param(2);
        optimizer.add_param(4);
        optimizer.add_param(3);
        let mut params = vec![vec![0.0_f32; 2], vec![0.0_f32; 4], vec![0.0_f32; 3]];
        let grads = vec![vec![0.01_f32; 2], vec![0.01_f32; 4], vec![0.01_f32; 3]];
        for _ in 0..5 {
            optimizer.step(&mut params, &grads).expect("step failed");
        }
        assert_eq!(optimizer.states[0].step, 5);
        assert_eq!(optimizer.states[1].step, 5);
        assert_eq!(optimizer.states[2].step, 5);
    }
}
