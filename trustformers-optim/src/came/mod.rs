//! # CAME Optimizer Module
//!
//! Contains both the original CAME implementation and the new advanced
//! CAME optimizer with factored second-moment estimation and confidence guidance.

pub mod legacy;

pub use legacy::{CAMEConfig, CAME};

// New advanced CAME implementation as specified in Wave 15 Workstream BB

use trustformers_core::errors::TrustformersError;

/// Error type for the advanced optimizer implementations.
#[derive(Debug, thiserror::Error)]
pub enum OptimError {
    /// Parameter and gradient length mismatch.
    #[error("length mismatch: param length {param} != grad length {grad}")]
    LengthMismatch { param: usize, grad: usize },
    /// Row/col dimensions inconsistent with total size.
    #[error("dimension mismatch: rows * cols ({rows} * {cols} = {product}) != size {size}")]
    DimensionMismatch {
        rows: usize,
        cols: usize,
        product: usize,
        size: usize,
    },
    /// State not initialised for a parameter group index.
    #[error("no state initialised for parameter group index {0}")]
    StateNotInitialised(usize),
    /// Unexpected numerical issue (NaN/Inf).
    #[error("numerical error: {0}")]
    NumericalError(String),
}

impl From<OptimError> for TrustformersError {
    fn from(e: OptimError) -> Self {
        TrustformersError::invalid_operation(e.to_string())
    }
}

/// Configuration for the advanced CAME optimizer (Luo et al., 2023).
///
/// Reference: "CAME: Confidence-guided Adaptive Memory Efficient Optimization"
#[derive(Debug, Clone)]
pub struct CameConfig {
    /// Learning rate (default 2e-4).
    pub lr: f64,
    /// (β1, β2, β3) — momentum, RMS, confidence decay rates.
    /// Default: (0.9, 0.999, 0.9999).
    pub betas: (f64, f64, f64),
    /// (ε1, ε2) — numerical stability constants.
    /// Default: (1e-30, 1e-16).
    pub eps: (f64, f64),
    /// Decoupled weight decay (default 0.0).
    pub weight_decay: f64,
    /// RMS gradient clipping threshold (default 1.0).
    pub clip_threshold: f64,
    /// Exponent for second-moment decay schedule: β2_t = min(1 − t^decay_rate, β2).
    /// Default: -0.8.
    pub decay_rate: f64,
}

impl Default for CameConfig {
    fn default() -> Self {
        Self {
            lr: 2e-4,
            betas: (0.9, 0.999, 0.9999),
            eps: (1e-30, 1e-16),
            weight_decay: 0.0,
            clip_threshold: 1.0,
            decay_rate: -0.8,
        }
    }
}

/// Per-parameter optimizer state for the advanced CAME optimizer.
#[derive(Debug, Clone)]
pub struct CameParamState {
    /// Number of update steps taken.
    pub step: u64,
    /// Exponential moving average of gradients (first moment).
    pub exp_avg: Vec<f32>,
    /// Factored second moment — row factor `[rows]`.
    pub exp_avg_sq_row: Vec<f32>,
    /// Factored second moment — column factor `[cols]`.
    pub exp_avg_sq_col: Vec<f32>,
    /// Full second moment for 1-D parameters (`None` for 2-D params).
    pub exp_avg_sq: Option<Vec<f32>>,
    /// Instantaneous second-moment row factor (for confidence estimation).
    pub exp_avg_insta_sq_row: Vec<f32>,
    /// Instantaneous second-moment column factor (for confidence estimation).
    pub exp_avg_insta_sq_col: Vec<f32>,
}

impl CameParamState {
    /// Create a zeroed state for a 2-D parameter with the given dimensions.
    pub fn new_2d(size: usize, rows: usize, cols: usize) -> Self {
        Self {
            step: 0,
            exp_avg: vec![0.0_f32; size],
            exp_avg_sq_row: vec![0.0_f32; rows],
            exp_avg_sq_col: vec![0.0_f32; cols],
            exp_avg_sq: None,
            exp_avg_insta_sq_row: vec![0.0_f32; rows],
            exp_avg_insta_sq_col: vec![0.0_f32; cols],
        }
    }

    /// Create a zeroed state for a 1-D parameter.
    pub fn new_1d(size: usize) -> Self {
        Self {
            step: 0,
            exp_avg: vec![0.0_f32; size],
            exp_avg_sq_row: Vec::new(),
            exp_avg_sq_col: Vec::new(),
            exp_avg_sq: Some(vec![0.0_f32; size]),
            exp_avg_insta_sq_row: Vec::new(),
            exp_avg_insta_sq_col: Vec::new(),
        }
    }
}

/// Compute the Root-Mean-Square of `v`.
#[inline]
fn rms(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let sq_sum: f32 = v.iter().map(|x| x * x).sum();
    (sq_sum / v.len() as f32).sqrt()
}

/// Perform one CAME update step for a single parameter group.
///
/// # Arguments
///
/// * `param`  – mutable slice of parameter values (length = `rows * cols`).
/// * `grad`   – gradient slice (same length).
/// * `state`  – mutable per-parameter state.
/// * `config` – optimizer configuration.
/// * `rows`   – matrix row count (set to 1 for 1-D parameters).
/// * `cols`   – matrix column count (= `param.len()` for 1-D parameters).
///
/// # Errors
///
/// Returns [`OptimError`] on dimension mismatches or numerical issues.
pub fn came_update(
    param: &mut [f32],
    grad: &[f32],
    state: &mut CameParamState,
    config: &CameConfig,
    rows: usize,
    cols: usize,
) -> Result<(), OptimError> {
    // --- Validate dimensions ------------------------------------------------
    let size = param.len();
    if grad.len() != size {
        return Err(OptimError::LengthMismatch {
            param: size,
            grad: grad.len(),
        });
    }
    let expected = rows * cols;
    if expected != size {
        return Err(OptimError::DimensionMismatch {
            rows,
            cols,
            product: expected,
            size,
        });
    }

    // --- Step counter -------------------------------------------------------
    state.step += 1;
    let step = state.step as f64;

    // --- Dynamic β2_t -------------------------------------------------------
    // β2_t = min(1 - step^decay_rate, β2)
    let beta2_t = (1.0 - step.powf(config.decay_rate)).min(config.betas.1) as f32;

    let beta1 = config.betas.0 as f32;
    let beta3 = config.betas.2 as f32;
    let eps1 = config.eps.0 as f32;
    let eps2 = config.eps.1 as f32;

    // --- RMS gradient clip --------------------------------------------------
    let grad_rms = rms(grad);
    let clip_scale = if grad_rms > config.clip_threshold as f32 {
        config.clip_threshold as f32 / (grad_rms + eps1)
    } else {
        1.0
    };

    // Lazily clipped gradient (we avoid a heap allocation by applying the
    // scale inline in the loops below).

    // --- First moment update ------------------------------------------------
    for (m, &g) in state.exp_avg.iter_mut().zip(grad.iter()) {
        let g_clipped = g * clip_scale;
        *m = beta1 * *m + (1.0 - beta1) * g_clipped;
    }

    // --- Second-moment and confidence update --------------------------------
    if rows == 1 {
        // ---- 1-D path: full second moment -----------------------------------
        let sq = state
            .exp_avg_sq
            .as_mut()
            .ok_or_else(|| OptimError::NumericalError("1-D state missing exp_avg_sq".into()))?;
        for (s, &g) in sq.iter_mut().zip(grad.iter()) {
            let g_clipped = g * clip_scale;
            *s = beta2_t * *s + (1.0 - beta2_t) * (g_clipped * g_clipped + eps1);
        }

        // Parameter update
        for ((p, &m), &s) in param.iter_mut().zip(state.exp_avg.iter()).zip(sq.iter()) {
            let denom = s.sqrt() + eps2;
            let update = m / denom;
            if config.weight_decay != 0.0 {
                *p -= config.lr as f32 * config.weight_decay as f32 * *p;
            }
            *p -= config.lr as f32 * update;
        }
    } else {
        // ---- 2-D path: factored second moment + confidence ------------------
        // grad² row-means and col-means
        let mut row_mean = vec![0.0_f32; rows];
        let mut col_mean = vec![0.0_f32; cols];

        for i in 0..rows {
            let mut s = 0.0_f32;
            for j in 0..cols {
                let g = grad[i * cols + j] * clip_scale;
                s += g * g;
            }
            row_mean[i] = s / cols as f32 + eps1;
        }
        for j in 0..cols {
            let mut s = 0.0_f32;
            for i in 0..rows {
                let g = grad[i * cols + j] * clip_scale;
                s += g * g;
            }
            col_mean[j] = s / rows as f32 + eps1;
        }

        // Smoothed second-moment factors
        for (r, &rm) in state.exp_avg_sq_row.iter_mut().zip(row_mean.iter()) {
            *r = beta2_t * *r + (1.0 - beta2_t) * rm;
        }
        for (c, &cm) in state.exp_avg_sq_col.iter_mut().zip(col_mean.iter()) {
            *c = beta2_t * *c + (1.0 - beta2_t) * cm;
        }

        // Instantaneous second-moment factors (for confidence), use β3
        for (r, &rm) in state.exp_avg_insta_sq_row.iter_mut().zip(row_mean.iter()) {
            *r = beta3 * *r + (1.0 - beta3) * rm;
        }
        for (c, &cm) in state.exp_avg_insta_sq_col.iter_mut().zip(col_mean.iter()) {
            *c = beta3 * *c + (1.0 - beta3) * cm;
        }

        // Compute R = mean of smoothed row factors (used to normalize outer-product)
        let row_sum: f32 = state.exp_avg_sq_row.iter().sum();
        let row_normaliser = (row_sum / rows as f32).max(eps1);

        // Parameter update with confidence weighting
        for i in 0..rows {
            let smoothed_row = state.exp_avg_sq_row[i];
            let insta_row = state.exp_avg_insta_sq_row[i];

            for j in 0..cols {
                let smoothed_col = state.exp_avg_sq_col[j];
                let insta_col = state.exp_avg_insta_sq_col[j];

                // RMS estimate from factored moments
                let v_approx = (smoothed_row * smoothed_col / row_normaliser).sqrt();

                // Confidence weight: ratio of smoothed vs instantaneous
                let smoothed_insta_row = (insta_row * insta_col / row_normaliser).sqrt();
                let confidence = if smoothed_insta_row > eps1 {
                    (v_approx / (smoothed_insta_row + eps2)).min(1.0_f32)
                } else {
                    1.0_f32
                };

                let denom = v_approx + eps2;
                let idx = i * cols + j;
                let m = state.exp_avg[idx];
                let update = confidence * m / denom;

                let p = &mut param[idx];
                if config.weight_decay != 0.0 {
                    *p -= config.lr as f32 * config.weight_decay as f32 * *p;
                }
                *p -= config.lr as f32 * update;
            }
        }
    }

    Ok(())
}

/// Per-parameter group descriptor stored alongside the state.
#[derive(Debug, Clone)]
struct ParamGroupMeta {
    #[allow(dead_code)]
    size: usize,
    rows: usize,
    cols: usize,
}

/// Advanced CAME optimizer (factored second-moment + confidence guidance).
///
/// Reference: "CAME: Confidence-guided Adaptive Memory Efficient Optimization"
/// (Luo et al., 2023)
#[derive(Debug)]
pub struct CameOptimizer {
    /// Hyperparameter configuration.
    pub config: CameConfig,
    /// Per-parameter states.
    pub states: Vec<CameParamState>,
    /// Metadata (size/rows/cols) for each parameter group.
    meta: Vec<ParamGroupMeta>,
}

impl CameOptimizer {
    /// Create a new optimizer with the given configuration.
    pub fn new(config: CameConfig) -> Self {
        Self {
            config,
            states: Vec::new(),
            meta: Vec::new(),
        }
    }

    /// Register a parameter group and initialise its state.
    ///
    /// For 2-D matrices set `rows` and `cols` appropriately.
    /// For 1-D tensors use `rows = 1` and `cols = param_size`.
    pub fn add_param_group(&mut self, param_size: usize, rows: usize, cols: usize) {
        let state = if rows == 1 {
            CameParamState::new_1d(param_size)
        } else {
            CameParamState::new_2d(param_size, rows, cols)
        };
        self.states.push(state);
        self.meta.push(ParamGroupMeta {
            size: param_size,
            rows,
            cols,
        });
    }

    /// Perform one update step across all parameter groups.
    ///
    /// # Arguments
    ///
    /// * `params` – mutable reference to all parameter vectors (one per group).
    /// * `grads`  – gradient vectors (same order as `params`).
    ///
    /// # Errors
    ///
    /// Returns [`OptimError`] on any dimension mismatch.
    pub fn step(&mut self, params: &mut [Vec<f32>], grads: &[Vec<f32>]) -> Result<(), OptimError> {
        for (idx, ((param, grad), state)) in
            params.iter_mut().zip(grads.iter()).zip(self.states.iter_mut()).enumerate()
        {
            let meta = self.meta.get(idx).ok_or(OptimError::StateNotInitialised(idx))?;
            came_update(param, grad, state, &self.config, meta.rows, meta.cols)?;
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
    fn test_came_config_defaults() {
        let cfg = CameConfig::default();
        assert_relative_eq!(cfg.lr, 2e-4);
        assert_relative_eq!(cfg.betas.0, 0.9);
        assert_relative_eq!(cfg.betas.1, 0.999);
        assert_relative_eq!(cfg.betas.2, 0.9999);
        assert_relative_eq!(cfg.eps.0, 1e-30);
        assert_relative_eq!(cfg.eps.1, 1e-16);
        assert_relative_eq!(cfg.weight_decay, 0.0);
        assert_relative_eq!(cfg.clip_threshold, 1.0);
        assert_relative_eq!(cfg.decay_rate, -0.8);
    }

    // -----------------------------------------------------------------------
    // 2. State initialisation — 2-D
    // -----------------------------------------------------------------------
    #[test]
    fn test_state_init_2d() {
        let state = CameParamState::new_2d(6, 2, 3);
        assert_eq!(state.step, 0);
        assert_eq!(state.exp_avg.len(), 6);
        assert_eq!(state.exp_avg_sq_row.len(), 2);
        assert_eq!(state.exp_avg_sq_col.len(), 3);
        assert!(state.exp_avg_sq.is_none());
        assert_eq!(state.exp_avg_insta_sq_row.len(), 2);
        assert_eq!(state.exp_avg_insta_sq_col.len(), 3);
        assert!(state.exp_avg.iter().all(|&x| x == 0.0));
    }

    // -----------------------------------------------------------------------
    // 3. State initialisation — 1-D
    // -----------------------------------------------------------------------
    #[test]
    fn test_state_init_1d() {
        let state = CameParamState::new_1d(5);
        assert_eq!(state.step, 0);
        assert_eq!(state.exp_avg.len(), 5);
        assert!(state.exp_avg_sq_row.is_empty());
        assert!(state.exp_avg_sq_col.is_empty());
        assert!(state.exp_avg_sq.is_some());
        assert_eq!(state.exp_avg_sq.as_ref().map(|v| v.len()), Some(5));
    }

    // -----------------------------------------------------------------------
    // 4. Step counter increments
    // -----------------------------------------------------------------------
    #[test]
    fn test_step_counter() {
        let cfg = CameConfig::default();
        let mut state = CameParamState::new_1d(2);
        let mut param = vec![1.0_f32; 2];
        let grad = vec![0.1_f32; 2];

        came_update(&mut param, &grad, &mut state, &cfg, 1, 2).expect("update failed");
        assert_eq!(state.step, 1);
        came_update(&mut param, &grad, &mut state, &cfg, 1, 2).expect("update failed");
        assert_eq!(state.step, 2);
    }

    // -----------------------------------------------------------------------
    // 5. Factored second moment update (2-D)
    // -----------------------------------------------------------------------
    #[test]
    fn test_factored_second_moment_update() {
        let cfg = CameConfig {
            lr: 0.0,
            ..CameConfig::default()
        };
        let rows = 2_usize;
        let cols = 3_usize;
        let size = rows * cols;
        let mut state = CameParamState::new_2d(size, rows, cols);
        let mut param = vec![0.0_f32; size];
        let grad = vec![1.0_f32; size];

        // After step 1 all row/col factors must be positive
        came_update(&mut param, &grad, &mut state, &cfg, rows, cols).expect("update failed");
        assert!(state.exp_avg_sq_row.iter().all(|&x| x > 0.0));
        assert!(state.exp_avg_sq_col.iter().all(|&x| x > 0.0));
    }

    // -----------------------------------------------------------------------
    // 6. Dynamic β2 schedule
    // -----------------------------------------------------------------------
    #[test]
    fn test_dynamic_beta2_schedule() {
        let cfg = CameConfig::default();
        // At step 1: beta2_t = min(1 - 1^(-0.8), 0.999) = min(0.0, 0.999) = 0.0
        let step = 1_f64;
        let beta2_t = (1.0 - step.powf(cfg.decay_rate)).min(cfg.betas.1);
        assert_relative_eq!(beta2_t, 0.0, epsilon = 1e-9);

        // At step 100: 1 - 100^(-0.8) ≈ 1 - 0.025 = 0.975 < 0.999, so not capped
        let step100 = 100_f64;
        let beta2_100 = (1.0 - step100.powf(cfg.decay_rate)).min(cfg.betas.1);
        assert!(beta2_100 > 0.9 && beta2_100 < 1.0);
    }

    // -----------------------------------------------------------------------
    // 7. Confidence adaptation (insta rows updated with β3)
    // -----------------------------------------------------------------------
    #[test]
    fn test_confidence_adaptation() {
        let cfg = CameConfig::default();
        let rows = 2_usize;
        let cols = 2_usize;
        let size = rows * cols;
        let mut state = CameParamState::new_2d(size, rows, cols);
        let mut param = vec![0.0_f32; size];
        let grad = vec![1.0_f32; size];

        came_update(&mut param, &grad, &mut state, &cfg, rows, cols).expect("update failed");

        // Instantaneous factors are updated with β3 = 0.9999 — they should be non-zero
        assert!(state.exp_avg_insta_sq_row.iter().all(|&x| x > 0.0));
        assert!(state.exp_avg_insta_sq_col.iter().all(|&x| x > 0.0));
    }

    // -----------------------------------------------------------------------
    // 8. Weight decay applied
    // -----------------------------------------------------------------------
    #[test]
    fn test_weight_decay() {
        let cfg = CameConfig {
            lr: 1e-1,
            weight_decay: 0.1,
            ..CameConfig::default()
        };
        let mut state = CameParamState::new_1d(2);
        let initial_param = vec![1.0_f32; 2];
        let mut param = initial_param.clone();
        let grad = vec![0.0_f32; 2]; // zero grad — only weight decay effect

        came_update(&mut param, &grad, &mut state, &cfg, 1, 2).expect("update failed");

        // Parameters must be strictly smaller in absolute value
        for (p_new, p_old) in param.iter().zip(initial_param.iter()) {
            assert!(
                p_new.abs() < p_old.abs(),
                "weight decay did not reduce param"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 9. Single-step update moves in the right direction
    // -----------------------------------------------------------------------
    #[test]
    fn test_single_step_direction() {
        let cfg = CameConfig::default();
        let mut state = CameParamState::new_1d(3);
        let mut param = vec![0.5_f32; 3];
        let grad = vec![0.1_f32; 3]; // positive gradient

        let param_before = param.clone();
        came_update(&mut param, &grad, &mut state, &cfg, 1, 3).expect("update failed");

        // With positive gradient, parameters should decrease
        for (p_new, p_old) in param.iter().zip(param_before.iter()) {
            assert!(
                p_new < p_old,
                "param did not decrease with positive gradient"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 10. Gradient clipping — first moment is smaller under aggressive clip
    // -----------------------------------------------------------------------
    #[test]
    fn test_gradient_clipping() {
        // The clip_scale = clip_threshold / (rms(grad) + eps1) when rms > threshold.
        // With a large gradient the clipped first moment should be smaller than the
        // unclipped first moment.
        let cfg_tight = CameConfig {
            clip_threshold: 0.1,
            ..CameConfig::default()
        };
        let cfg_loose = CameConfig {
            clip_threshold: 1000.0,
            ..CameConfig::default()
        };

        let large_grad = vec![5.0_f32; 4];

        let mut state_tight = CameParamState::new_1d(4);
        let mut param_tight = vec![0.0_f32; 4];
        came_update(
            &mut param_tight,
            &large_grad,
            &mut state_tight,
            &cfg_tight,
            1,
            4,
        )
        .expect("tight update failed");

        let mut state_loose = CameParamState::new_1d(4);
        let mut param_loose = vec![0.0_f32; 4];
        came_update(
            &mut param_loose,
            &large_grad,
            &mut state_loose,
            &cfg_loose,
            1,
            4,
        )
        .expect("loose update failed");

        // Under tight clipping the first moment exp_avg values must be smaller in
        // absolute value because the effective gradient fed into the EMA was scaled down.
        let m_tight: f32 = state_tight.exp_avg.iter().map(|x| x.abs()).sum();
        let m_loose: f32 = state_loose.exp_avg.iter().map(|x| x.abs()).sum();
        assert!(
            m_tight < m_loose,
            "tight clipping did not reduce first moment: m_tight={m_tight} m_loose={m_loose}"
        );
    }

    // -----------------------------------------------------------------------
    // 11. Multi-step convergence on a quadratic (1-D)
    // -----------------------------------------------------------------------
    #[test]
    fn test_convergence_quadratic() {
        // Minimise f(x) = x^2 / 2, gradient = x
        let cfg = CameConfig {
            lr: 1e-2,
            ..CameConfig::default()
        };
        let mut state = CameParamState::new_1d(1);
        let mut param = vec![5.0_f32];

        for _ in 0..2000 {
            let grad = param.clone(); // gradient of x^2/2 is x
            came_update(&mut param, &grad, &mut state, &cfg, 1, 1).expect("update failed");
        }

        assert!(
            param[0].abs() < 0.1,
            "CAME did not converge on quadratic: final param = {}",
            param[0]
        );
    }

    // -----------------------------------------------------------------------
    // 12. Dimension mismatch error returned (not panicked)
    // -----------------------------------------------------------------------
    #[test]
    fn test_dimension_mismatch_error() {
        let cfg = CameConfig::default();
        let mut state = CameParamState::new_1d(4);
        let mut param = vec![0.0_f32; 4];
        let grad = vec![0.0_f32; 5]; // wrong size

        let result = came_update(&mut param, &grad, &mut state, &cfg, 1, 4);
        assert!(result.is_err());
        matches!(result.unwrap_err(), OptimError::LengthMismatch { .. });
    }

    // -----------------------------------------------------------------------
    // 13. CameOptimizer multi-param step
    // -----------------------------------------------------------------------
    #[test]
    fn test_came_optimizer_multi_param() {
        let cfg = CameConfig::default();
        let mut optimizer = CameOptimizer::new(cfg);
        optimizer.add_param_group(4, 2, 2);
        optimizer.add_param_group(3, 1, 3);

        let mut params = vec![vec![1.0_f32; 4], vec![1.0_f32; 3]];
        let grads = vec![vec![0.1_f32; 4], vec![0.1_f32; 3]];

        optimizer.step(&mut params, &grads).expect("step failed");
        assert_eq!(optimizer.states[0].step, 1);
        assert_eq!(optimizer.states[1].step, 1);
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_came_state_step_zero_at_init() {
        let state = CameParamState::new_2d(6, 2, 3);
        assert_eq!(state.step, 0);
        let state1d = CameParamState::new_1d(4);
        assert_eq!(state1d.step, 0);
    }

    #[test]
    fn test_came_confidence_factors_nonzero_after_step() {
        let cfg = CameConfig::default();
        let mut state = CameParamState::new_2d(6, 2, 3);
        let mut param = vec![0.5_f32; 6];
        let grad = vec![0.1_f32; 6];
        came_update(&mut param, &grad, &mut state, &cfg, 2, 3).expect("update failed");
        assert!(
            state.exp_avg_insta_sq_row.iter().all(|&x| x > 0.0),
            "insta_sq_row should be nonzero after update"
        );
        assert!(
            state.exp_avg_insta_sq_col.iter().all(|&x| x > 0.0),
            "insta_sq_col should be nonzero after update"
        );
    }

    #[test]
    fn test_came_positive_grad_decreases_params() {
        let cfg = CameConfig::default();
        let mut state = CameParamState::new_1d(4);
        let mut param = vec![1.0_f32; 4];
        let grad = vec![0.5_f32; 4];
        let before = param.clone();
        came_update(&mut param, &grad, &mut state, &cfg, 1, 4).expect("update failed");
        for (p_new, p_old) in param.iter().zip(before.iter()) {
            assert!(
                p_new < p_old,
                "param should decrease with positive gradient"
            );
        }
    }

    #[test]
    fn test_came_1d_vs_2d_single_element_both_decrease() {
        let cfg = CameConfig::default();
        let grad = vec![0.2_f32];

        // 1D path: new_1d, rows=1, cols=1
        let mut state_1d = CameParamState::new_1d(1);
        let mut param_1d = vec![1.0_f32];
        came_update(&mut param_1d, &grad, &mut state_1d, &cfg, 1, 1).expect("1d update failed");
        assert!(param_1d[0] < 1.0, "1D param should decrease");

        // True 2D path: 2 rows x 2 cols (rows != 1 to take the factored path)
        let grad_2d = vec![0.2_f32; 4];
        let mut state_2d = CameParamState::new_2d(4, 2, 2);
        let mut param_2d = vec![1.0_f32; 4];
        came_update(&mut param_2d, &grad_2d, &mut state_2d, &cfg, 2, 2).expect("2d update failed");
        for &p in &param_2d {
            assert!(p < 1.0, "2D param should decrease");
        }
    }

    #[test]
    fn test_came_weight_decay_larger_shrinks_more() {
        let grad = vec![0.0_f32; 3];

        let cfg_small = CameConfig {
            lr: 0.1,
            weight_decay: 0.01,
            ..CameConfig::default()
        };
        let mut state_small = CameParamState::new_1d(3);
        let mut param_small = vec![1.0_f32; 3];
        came_update(&mut param_small, &grad, &mut state_small, &cfg_small, 1, 3)
            .expect("small wd update failed");

        let cfg_large = CameConfig {
            lr: 0.1,
            weight_decay: 0.1,
            ..CameConfig::default()
        };
        let mut state_large = CameParamState::new_1d(3);
        let mut param_large = vec![1.0_f32; 3];
        came_update(&mut param_large, &grad, &mut state_large, &cfg_large, 1, 3)
            .expect("large wd update failed");

        for (ps, pl) in param_small.iter().zip(param_large.iter()) {
            assert!(
                ps.abs() > pl.abs(),
                "larger weight_decay should shrink more: small={ps}, large={pl}"
            );
        }
    }

    #[test]
    fn test_came_zero_grad_zero_wd_params_unchanged() {
        let cfg = CameConfig {
            lr: 0.1,
            weight_decay: 0.0,
            ..CameConfig::default()
        };
        let mut state = CameParamState::new_1d(3);
        let mut param = vec![2.0_f32; 3];
        let original = param.clone();
        let grad = vec![0.0_f32; 3];
        came_update(&mut param, &grad, &mut state, &cfg, 1, 3).expect("update failed");
        for (p_new, p_old) in param.iter().zip(original.iter()) {
            assert_relative_eq!(*p_new, *p_old, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_came_multiple_steps_move_toward_zero() {
        let cfg = CameConfig {
            lr: 1e-2,
            weight_decay: 0.0,
            ..CameConfig::default()
        };
        let mut state = CameParamState::new_1d(1);
        let mut param = vec![3.0_f32];
        for _ in 0..500 {
            let grad = param.clone();
            came_update(&mut param, &grad, &mut state, &cfg, 1, 1).expect("update failed");
        }
        assert!(
            param[0].abs() < 3.0,
            "param should move toward 0 over many steps"
        );
    }

    #[test]
    fn test_came_state_not_initialised_no_panic() {
        let cfg = CameConfig::default();
        let mut optimizer = CameOptimizer::new(cfg);
        // No add_param_group calls — zip with 0 states = 0 iterations, no panic
        let mut params = vec![vec![1.0_f32; 3]];
        let grads = vec![vec![0.1_f32; 3]];
        let result = optimizer.step(&mut params, &grads);
        // Should not panic; either Ok or Err is acceptable
        let _ = result;
    }

    #[test]
    fn test_came_batch_2d_params_step_count() {
        let cfg = CameConfig::default();
        let mut optimizer = CameOptimizer::new(cfg);
        optimizer.add_param_group(6, 2, 3);
        optimizer.add_param_group(9, 3, 3);
        let mut params = vec![vec![0.5_f32; 6], vec![0.5_f32; 9]];
        let grads = vec![vec![0.1_f32; 6], vec![0.1_f32; 9]];
        optimizer.step(&mut params, &grads).expect("step failed");
        assert_eq!(optimizer.states[0].step, 1);
        assert_eq!(optimizer.states[1].step, 1);
    }

    #[test]
    fn test_came_clipping_bounds_param_change() {
        // Clipping affects the first moment (exp_avg). After step 1:
        // exp_avg_tight[i] = (1-beta1) * clip_scale * grad[i]  (small clip_scale for tight)
        // exp_avg_loose[i] = (1-beta1) * 1.0 * grad[i]         (no clipping needed)
        // We verify by checking that the first moments differ.
        let large_grad = vec![100.0_f32; 4];

        let cfg_tight = CameConfig {
            lr: 1.0,
            clip_threshold: 0.001,
            weight_decay: 0.0,
            ..CameConfig::default()
        };
        let mut s_tight = CameParamState::new_1d(4);
        let mut p_tight = vec![0.0_f32; 4];
        came_update(&mut p_tight, &large_grad, &mut s_tight, &cfg_tight, 1, 4)
            .expect("tight failed");

        let cfg_loose = CameConfig {
            lr: 1.0,
            clip_threshold: 1000.0,
            weight_decay: 0.0,
            ..CameConfig::default()
        };
        let mut s_loose = CameParamState::new_1d(4);
        let mut p_loose = vec![0.0_f32; 4];
        came_update(&mut p_loose, &large_grad, &mut s_loose, &cfg_loose, 1, 4)
            .expect("loose failed");

        // The tight-clipped first moment should be much smaller in magnitude
        let m_tight: f32 = s_tight.exp_avg.iter().map(|x| x.abs()).sum();
        let m_loose: f32 = s_loose.exp_avg.iter().map(|x| x.abs()).sum();
        assert!(
            m_tight < m_loose,
            "tight clipping should reduce first moment: tight={m_tight}, loose={m_loose}"
        );
    }

    #[test]
    fn test_came_2d_factored_memory_efficiency() {
        let rows = 100_usize;
        let cols = 200_usize;
        let size = rows * cols;
        let state = CameParamState::new_2d(size, rows, cols);
        let factored_size = state.exp_avg_sq_row.len() + state.exp_avg_sq_col.len();
        assert!(
            factored_size < size,
            "factored memory ({factored_size}) should be less than full size ({size})"
        );
    }

    #[test]
    fn test_came_beta3_effect_on_insta_sq() {
        let rows = 2_usize;
        let cols = 2_usize;
        let grad = vec![1.0_f32; 4];

        let cfg_high = CameConfig {
            betas: (0.9, 0.999, 0.9999),
            ..CameConfig::default()
        };
        let mut state_high = CameParamState::new_2d(4, rows, cols);
        let mut param_high = vec![0.5_f32; 4];
        came_update(
            &mut param_high,
            &grad,
            &mut state_high,
            &cfg_high,
            rows,
            cols,
        )
        .expect("high beta3 update failed");

        let cfg_low = CameConfig {
            betas: (0.9, 0.999, 0.5),
            ..CameConfig::default()
        };
        let mut state_low = CameParamState::new_2d(4, rows, cols);
        let mut param_low = vec![0.5_f32; 4];
        came_update(&mut param_low, &grad, &mut state_low, &cfg_low, rows, cols)
            .expect("low beta3 update failed");

        let sum_high: f32 = state_high.exp_avg_insta_sq_row.iter().sum();
        let sum_low: f32 = state_low.exp_avg_insta_sq_row.iter().sum();
        assert!(
            sum_high < sum_low,
            "higher β3 should give smaller insta_sq update: high={sum_high}, low={sum_low}"
        );
    }

    #[test]
    fn test_came_three_groups_distinct_states() {
        let cfg = CameConfig::default();
        let mut optimizer = CameOptimizer::new(cfg);
        optimizer.add_param_group(2, 1, 2);
        optimizer.add_param_group(4, 2, 2);
        optimizer.add_param_group(6, 2, 3);

        let mut params = vec![vec![1.0_f32; 2], vec![1.0_f32; 4], vec![1.0_f32; 6]];
        let grads = vec![vec![0.1_f32; 2], vec![0.1_f32; 4], vec![0.1_f32; 6]];
        optimizer.step(&mut params, &grads).expect("step failed");
        assert_eq!(optimizer.states[0].step, 1);
        assert_eq!(optimizer.states[1].step, 1);
        assert_eq!(optimizer.states[2].step, 1);
        assert_eq!(optimizer.states[0].exp_avg.len(), 2);
        assert_eq!(optimizer.states[1].exp_avg.len(), 4);
        assert_eq!(optimizer.states[2].exp_avg.len(), 6);
    }

    #[test]
    fn test_came_lr_scaling_effect() {
        let grad = vec![0.1_f32; 3];

        let cfg_small_lr = CameConfig {
            lr: 1e-4,
            weight_decay: 0.0,
            ..CameConfig::default()
        };
        let mut s_small = CameParamState::new_1d(3);
        let mut p_small = vec![2.0_f32; 3];
        came_update(&mut p_small, &grad, &mut s_small, &cfg_small_lr, 1, 3)
            .expect("small lr failed");

        let cfg_large_lr = CameConfig {
            lr: 1e-1,
            weight_decay: 0.0,
            ..CameConfig::default()
        };
        let mut s_large = CameParamState::new_1d(3);
        let mut p_large = vec![2.0_f32; 3];
        came_update(&mut p_large, &grad, &mut s_large, &cfg_large_lr, 1, 3)
            .expect("large lr failed");

        let change_small: f32 = (2.0 - p_small[0]).abs();
        let change_large: f32 = (2.0 - p_large[0]).abs();
        assert!(
            change_large > change_small,
            "larger lr should produce larger change: small={change_small}, large={change_large}"
        );
    }

    #[test]
    fn test_came_dimension_mismatch_rows_cols_wrong() {
        let cfg = CameConfig::default();
        let mut state = CameParamState::new_2d(9, 3, 3);
        // param has 8 elements but rows*cols=9
        let mut param = vec![0.0_f32; 8];
        let grad = vec![0.0_f32; 8];
        let result = came_update(&mut param, &grad, &mut state, &cfg, 3, 3);
        assert!(result.is_err(), "should return error on dimension mismatch");
    }

    #[test]
    fn test_came_exp_avg_direction_matches_grad() {
        let cfg = CameConfig::default();
        let mut state = CameParamState::new_1d(3);
        let mut param = vec![0.0_f32; 3];
        let grad = vec![0.5_f32, -0.5_f32, 0.3_f32];
        came_update(&mut param, &grad, &mut state, &cfg, 1, 3).expect("update failed");
        assert!(
            state.exp_avg[0] > 0.0,
            "positive grad → positive exp_avg[0]"
        );
        assert!(
            state.exp_avg[1] < 0.0,
            "negative grad → negative exp_avg[1]"
        );
        assert!(
            state.exp_avg[2] > 0.0,
            "positive grad → positive exp_avg[2]"
        );
    }
}
