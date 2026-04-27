//! Cyclical Learning Rate scheduler with decay.
//!
//! Combines cyclic learning rates (Smith 2015) with amplitude decay over cycles.
//! The LR oscillates between `base_lr` and `max_lr`, with the peak amplitude of
//! each successive cycle decaying by a configurable factor.
//!
//! ## Schedule formula
//! ```text
//! cycle      = floor(1 + step / (step_size_up + step_size_down))
//! x          = |step / step_size_up - 2 * cycle + 1|
//! scale      = max(0, 1 - x) * mode_scale(cycle)
//! lr         = base_lr + (max_lr - base_lr) * scale
//! ```
//!
//! For `Triangular` mode:   `mode_scale(cycle) = 1.0`
//! For `Triangular2` mode:  `mode_scale(cycle) = 1 / 2^(cycle-1)`
//! For `ExpRange` mode:     `mode_scale(cycle) = gamma^(step)`  (cycle-step granularity)
//!
//! ## References
//! - Smith (2015) "Cyclical Learning Rates for Training Neural Networks"
//! - Smith & Touvron (2019) "Super-Convergence: Very Fast Training of Neural Networks"

use trustformers_core::errors::{Result, TrustformersError};

// ─────────────────────────────────────────── CyclicLrMode ───────────────────

/// Mode that controls how the cycle amplitude decays.
#[derive(Debug, Clone, PartialEq)]
pub enum CyclicLrMode {
    /// Constant amplitude — classic triangular wave.
    Triangular,
    /// Amplitude halved every cycle: `scale = 1 / 2^(cycle−1)`.
    Triangular2,
    /// Amplitude decays exponentially by `gamma` at every **step**:
    /// `scale = gamma^step`.
    ExpRange {
        /// Per-step decay factor (typically 0.99994 or similar).
        gamma: f64,
    },
}

// ─────────────────────────────────────────── CyclicLrConfig ─────────────────

/// Configuration for [`CyclicLrScheduler`].
#[derive(Debug, Clone)]
pub struct CyclicLrConfig {
    /// Minimum (base) learning rate — the LR never drops below this.
    pub base_lr: f64,
    /// Maximum learning rate reached at the cycle peak.
    pub max_lr: f64,
    /// Number of steps in the increasing (warm-up) half of each cycle.
    pub step_size_up: usize,
    /// Number of steps in the decreasing half.  Defaults to `step_size_up`.
    pub step_size_down: Option<usize>,
    /// Which cycling mode to use.
    pub mode: CyclicLrMode,
    /// If `true`, momentum is cycled inversely to the learning rate.
    pub cycle_momentum: bool,
    /// Base momentum (used when LR is at its maximum).
    pub base_momentum: f64,
    /// Maximum momentum (used when LR is at its minimum).
    pub max_momentum: f64,
}

impl Default for CyclicLrConfig {
    fn default() -> Self {
        Self {
            base_lr: 1e-4,
            max_lr: 1e-3,
            step_size_up: 2000,
            step_size_down: None,
            mode: CyclicLrMode::Triangular,
            cycle_momentum: false,
            base_momentum: 0.8,
            max_momentum: 0.9,
        }
    }
}

impl CyclicLrConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.base_lr <= 0.0 {
            return Err(TrustformersError::config_error(
                "base_lr must be positive",
                "CyclicLrConfig::validate",
            ));
        }
        if self.max_lr <= self.base_lr {
            return Err(TrustformersError::config_error(
                "max_lr must be greater than base_lr",
                "CyclicLrConfig::validate",
            ));
        }
        if self.step_size_up == 0 {
            return Err(TrustformersError::config_error(
                "step_size_up must be > 0",
                "CyclicLrConfig::validate",
            ));
        }
        if let Some(sd) = self.step_size_down {
            if sd == 0 {
                return Err(TrustformersError::config_error(
                    "step_size_down must be > 0",
                    "CyclicLrConfig::validate",
                ));
            }
        }
        if self.cycle_momentum && self.base_momentum >= self.max_momentum {
            return Err(TrustformersError::config_error(
                "max_momentum must be greater than base_momentum when cycle_momentum is true",
                "CyclicLrConfig::validate",
            ));
        }
        if let CyclicLrMode::ExpRange { gamma } = self.mode {
            if gamma <= 0.0 || gamma > 1.0 {
                return Err(TrustformersError::config_error(
                    "gamma for ExpRange mode must be in (0, 1]",
                    "CyclicLrConfig::validate",
                ));
            }
        }
        Ok(())
    }

    /// Effective step_size_down (falls back to `step_size_up` if `None`).
    fn effective_step_size_down(&self) -> usize {
        self.step_size_down.unwrap_or(self.step_size_up)
    }
}

// ─────────────────────────────────────────── CyclicLrScheduler ──────────────

/// Cyclical LR scheduler that oscillates between `base_lr` and `max_lr`.
///
/// The scheduler is *stateful* — call [`step`](CyclicLrScheduler::step) to
/// advance it by one training step.
pub struct CyclicLrScheduler {
    config: CyclicLrConfig,
    step_count: usize,
}

impl CyclicLrScheduler {
    /// Create a new scheduler, validating the configuration.
    pub fn new(config: CyclicLrConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            step_count: 0,
        })
    }

    /// Compute the learning rate at an arbitrary step (without advancing state).
    pub fn get_lr_at(&self, step: usize) -> f64 {
        let step_size_up = self.config.step_size_up as f64;
        let step_size_down = self.config.effective_step_size_down() as f64;
        let cycle_len = step_size_up + step_size_down;
        let step_f = step as f64;

        // Compute which cycle we are in (1-indexed)
        let cycle = (1.0 + step_f / cycle_len).floor();
        let cycle_step = step_f - (cycle - 1.0) * cycle_len;

        // Compute x: triangular wave [0, 1]
        let x = if cycle_step < step_size_up {
            cycle_step / step_size_up
        } else {
            1.0 - (cycle_step - step_size_up) / step_size_down
        };
        let x = x.clamp(0.0, 1.0);

        // Mode-specific amplitude scale
        let scale = match &self.config.mode {
            CyclicLrMode::Triangular => x,
            CyclicLrMode::Triangular2 => x / 2_f64.powf(cycle - 1.0),
            CyclicLrMode::ExpRange { gamma } => x * gamma.powf(step_f),
        };

        self.config.base_lr + (self.config.max_lr - self.config.base_lr) * scale
    }

    /// Get the learning rate at the *current* step count.
    pub fn get_lr(&self) -> f64 {
        self.get_lr_at(self.step_count)
    }

    /// Get the momentum at the current step (only meaningful when `cycle_momentum = true`).
    ///
    /// Momentum cycles inversely: it is highest when LR is lowest and vice-versa.
    pub fn get_momentum(&self) -> f64 {
        if !self.config.cycle_momentum {
            return self.config.max_momentum;
        }
        let lr = self.get_lr();
        let lr_range = self.config.max_lr - self.config.base_lr;
        if lr_range.abs() < f64::EPSILON {
            return self.config.max_momentum;
        }
        // Inverse scale: when lr == base_lr (scale=0), momentum == max_momentum
        //                when lr == max_lr  (scale=1), momentum == base_momentum
        let lr_scale = (lr - self.config.base_lr) / lr_range;
        self.config.max_momentum - (self.config.max_momentum - self.config.base_momentum) * lr_scale
    }

    /// Advance by one training step and return the new learning rate.
    pub fn step(&mut self) -> f64 {
        let lr = self.get_lr();
        self.step_count += 1;
        lr
    }

    /// Current cycle number (1-indexed).
    pub fn current_cycle(&self) -> usize {
        let cycle_len = (self.config.step_size_up + self.config.effective_step_size_down()) as f64;
        (1.0 + self.step_count as f64 / cycle_len).floor() as usize
    }

    /// Step index within the current cycle (0-indexed).
    pub fn cycle_step(&self) -> usize {
        let cycle_len = self.config.step_size_up + self.config.effective_step_size_down();
        self.step_count % cycle_len
    }

    /// Reset to the initial state (step 0).
    pub fn reset(&mut self) {
        self.step_count = 0;
    }

    /// Generate the complete LR schedule for `n_steps` steps (starting from current state).
    pub fn schedule(&self, n_steps: usize) -> Vec<f64> {
        let start = self.step_count;
        (start..start + n_steps).map(|s| self.get_lr_at(s)).collect()
    }
}

// ─────────────────────────────────────────── AnnealStrategy ─────────────────

/// Annealing strategy used by [`OneCycleLrScheduler`].
#[derive(Debug, Clone, PartialEq)]
pub enum AnnealStrategy {
    /// Cosine annealing between the phase's start and end LR.
    Cos,
    /// Linear interpolation between the phase's start and end LR.
    Linear,
}

// ─────────────────────────────────────────── OneCycleLrScheduler ─────────────

/// One-Cycle LR scheduler (Smith & Touvron 2019).
///
/// Uses a **single** cycle that consists of three phases:
///
/// 1. **Warm-up** (`pct_start * total_steps` steps): LR rises from
///    `initial_lr = max_lr / div_factor` to `max_lr`.
/// 2. **Annealing** (`(1 − pct_start) * total_steps` steps): LR falls from
///    `max_lr` to `min_lr = initial_lr / final_div_factor`.
/// 3. The cycle ends at the last step.
///
/// This schedule (popularised by *fastai*) is empirically very effective for
/// achieving fast convergence ("super-convergence").
pub struct OneCycleLrScheduler {
    max_lr: f64,
    total_steps: usize,
    pct_start: f64,
    div_factor: f64,
    final_div_factor: f64,
    anneal_strategy: AnnealStrategy,
    step_count: usize,
}

impl OneCycleLrScheduler {
    /// Create a new `OneCycleLrScheduler` with default auxiliary parameters.
    ///
    /// Defaults:
    /// - `pct_start = 0.3`
    /// - `div_factor = 25.0`  → initial_lr = max_lr / 25
    /// - `final_div_factor = 1e4` → min_lr = initial_lr / 1e4
    /// - `anneal_strategy = AnnealStrategy::Cos`
    pub fn new(max_lr: f64, total_steps: usize) -> Result<Self> {
        if max_lr <= 0.0 {
            return Err(TrustformersError::config_error(
                "max_lr must be positive",
                "OneCycleLrScheduler::new",
            ));
        }
        if total_steps == 0 {
            return Err(TrustformersError::config_error(
                "total_steps must be > 0",
                "OneCycleLrScheduler::new",
            ));
        }
        Ok(Self {
            max_lr,
            total_steps,
            pct_start: 0.3,
            div_factor: 25.0,
            final_div_factor: 1e4,
            anneal_strategy: AnnealStrategy::Cos,
            step_count: 0,
        })
    }

    /// Set the fraction of total steps used for the warm-up phase.
    pub fn with_pct_start(mut self, pct: f64) -> Self {
        self.pct_start = pct.clamp(0.0, 1.0);
        self
    }

    /// Set the initial LR divisor: `initial_lr = max_lr / div_factor`.
    pub fn with_div_factor(mut self, div: f64) -> Self {
        self.div_factor = div;
        self
    }

    /// Set the final LR divisor: `min_lr = initial_lr / final_div_factor`.
    pub fn with_final_div_factor(mut self, div: f64) -> Self {
        self.final_div_factor = div;
        self
    }

    /// Set the annealing strategy for both phases.
    pub fn with_anneal_strategy(mut self, strategy: AnnealStrategy) -> Self {
        self.anneal_strategy = strategy;
        self
    }

    /// `initial_lr = max_lr / div_factor`
    fn initial_lr(&self) -> f64 {
        self.max_lr / self.div_factor
    }

    /// `min_lr = initial_lr / final_div_factor`
    fn min_lr(&self) -> f64 {
        self.initial_lr() / self.final_div_factor
    }

    fn anneal(&self, start: f64, end: f64, pct: f64) -> f64 {
        match self.anneal_strategy {
            AnnealStrategy::Linear => start + (end - start) * pct,
            AnnealStrategy::Cos => {
                use std::f64::consts::PI;
                // pct=0 → start, pct=1 → end (cosine interpolation)
                // cos(0)=1 → start; cos(π)=−1 → end
                start + (end - start) * (1.0 - (PI * pct).cos()) / 2.0
            },
        }
    }

    /// Compute LR at an arbitrary step (without advancing state).
    pub fn get_lr_at(&self, step: usize) -> f64 {
        let step_f = step.min(self.total_steps) as f64;
        let total_f = self.total_steps as f64;
        let warmup_steps = self.pct_start * total_f;

        if step_f < warmup_steps {
            // Phase 1: warm-up
            let pct = if warmup_steps > 0.0 { step_f / warmup_steps } else { 1.0 };
            self.anneal(self.initial_lr(), self.max_lr, pct)
        } else {
            // Phase 2: annealing
            let decay_steps = total_f - warmup_steps;
            let pct = if decay_steps > 0.0 { (step_f - warmup_steps) / decay_steps } else { 1.0 };
            self.anneal(self.max_lr, self.min_lr(), pct)
        }
    }

    /// Get the learning rate at the current step.
    pub fn get_lr(&self) -> f64 {
        self.get_lr_at(self.step_count)
    }

    /// Advance by one step and return the LR for that step.
    pub fn step(&mut self) -> f64 {
        let lr = self.get_lr();
        if self.step_count < self.total_steps {
            self.step_count += 1;
        }
        lr
    }

    /// Reset to step 0.
    pub fn reset(&mut self) {
        self.step_count = 0;
    }

    /// Generate the complete LR schedule for `n_steps` steps.
    pub fn schedule(&self, n_steps: usize) -> Vec<f64> {
        let start = self.step_count;
        (start..start + n_steps).map(|s| self.get_lr_at(s)).collect()
    }
}

// ─────────────────────────────────────────── tests ───────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CyclicLrConfig validation ──

    #[test]
    fn test_cyclic_config_valid() {
        let cfg = CyclicLrConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_cyclic_config_invalid_base_lr() {
        let cfg = CyclicLrConfig {
            base_lr: 0.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_cyclic_config_invalid_max_lr_lte_base() {
        let cfg = CyclicLrConfig {
            base_lr: 1e-3,
            max_lr: 1e-4,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_cyclic_config_invalid_step_size_up_zero() {
        let cfg = CyclicLrConfig {
            step_size_up: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_cyclic_config_invalid_exp_range_gamma() {
        let cfg = CyclicLrConfig {
            mode: CyclicLrMode::ExpRange { gamma: 1.5 },
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ── CyclicLrScheduler ──

    #[test]
    fn test_cyclic_lr_starts_at_base_lr() {
        let cfg = CyclicLrConfig {
            base_lr: 1e-4,
            max_lr: 1e-3,
            step_size_up: 4,
            ..Default::default()
        };
        let sched = CyclicLrScheduler::new(cfg).expect("construct");
        let lr0 = sched.get_lr();
        assert!(
            (lr0 - 1e-4).abs() < 1e-12,
            "Expected base_lr at step 0, got {}",
            lr0
        );
    }

    #[test]
    fn test_cyclic_lr_peaks_at_max_lr() {
        let cfg = CyclicLrConfig {
            base_lr: 1e-4,
            max_lr: 1e-3,
            step_size_up: 4,
            step_size_down: Some(4),
            ..Default::default()
        };
        let sched = CyclicLrScheduler::new(cfg).expect("construct");
        // Peak is at step == step_size_up
        let peak_lr = sched.get_lr_at(4);
        assert!(
            (peak_lr - 1e-3).abs() < 1e-12,
            "Expected max_lr at peak, got {}",
            peak_lr
        );
    }

    #[test]
    fn test_cyclic_lr_triangular2_halves_each_cycle() {
        // base_lr must be > 0 per validation; we set it small to isolate the amplitude effect.
        let cfg = CyclicLrConfig {
            base_lr: 1e-9, // near-zero base, so the peak is dominated by (max_lr - base_lr)
            max_lr: 1.0,
            step_size_up: 2,
            step_size_down: Some(2),
            mode: CyclicLrMode::Triangular2,
            ..Default::default()
        };
        let sched = CyclicLrScheduler::new(cfg).expect("construct");
        let base = sched.config.base_lr;
        let peak_cycle1 = sched.get_lr_at(2) - base; // amplitude above base, cycle 1
        let peak_cycle2 = sched.get_lr_at(6) - base; // amplitude above base, cycle 2
                                                     // Cycle 2 amplitude should be half of cycle 1
        let ratio = peak_cycle2 / peak_cycle1;
        assert!(
            (ratio - 0.5).abs() < 1e-6,
            "Expected ratio ≈ 0.5, got {}",
            ratio
        );
    }

    #[test]
    fn test_cyclic_lr_exprange_decays() {
        let cfg = CyclicLrConfig {
            base_lr: 1e-9,
            max_lr: 1.0,
            step_size_up: 5,
            step_size_down: Some(5),
            mode: CyclicLrMode::ExpRange { gamma: 0.99 },
            ..Default::default()
        };
        let sched = CyclicLrScheduler::new(cfg).expect("construct");
        let peak1 = sched.get_lr_at(5);
        let peak2 = sched.get_lr_at(15);
        assert!(
            peak2 < peak1,
            "ExpRange peak should decay: {} < {}",
            peak2,
            peak1
        );
    }

    #[test]
    fn test_cyclic_lr_step_advances() {
        let cfg = CyclicLrConfig {
            base_lr: 1e-4,
            max_lr: 1e-3,
            step_size_up: 4,
            ..Default::default()
        };
        let mut sched = CyclicLrScheduler::new(cfg).expect("construct");
        let lr0 = sched.step();
        let lr1 = sched.step();
        assert_eq!(sched.step_count, 2);
        assert!(lr0 < lr1, "LR should increase during warm-up phase");
    }

    #[test]
    fn test_cyclic_lr_reset() {
        let cfg = CyclicLrConfig {
            step_size_up: 4,
            ..Default::default()
        };
        let mut sched = CyclicLrScheduler::new(cfg).expect("construct");
        for _ in 0..10 {
            sched.step();
        }
        let lr_before_reset = sched.get_lr();
        sched.reset();
        assert_eq!(sched.step_count, 0);
        let lr_after_reset = sched.get_lr();
        assert!(
            (lr_after_reset - sched.config.base_lr).abs() < 1e-12,
            "After reset LR should be base_lr, got {}",
            lr_after_reset
        );
        let _ = lr_before_reset; // suppress unused warning
    }

    #[test]
    fn test_cyclic_lr_schedule_length() {
        let cfg = CyclicLrConfig {
            step_size_up: 4,
            ..Default::default()
        };
        let sched = CyclicLrScheduler::new(cfg).expect("construct");
        let sch = sched.schedule(20);
        assert_eq!(sch.len(), 20);
    }

    #[test]
    fn test_cyclic_lr_momentum_inverse() {
        let cfg = CyclicLrConfig {
            base_lr: 1e-4,
            max_lr: 1e-3,
            step_size_up: 10,
            cycle_momentum: true,
            base_momentum: 0.8,
            max_momentum: 0.9,
            ..Default::default()
        };
        let sched = CyclicLrScheduler::new(cfg).expect("construct");
        // At step 0 (LR is at base_lr, scale=0), momentum should be max_momentum
        let mom0 = sched.get_momentum();
        assert!(
            (mom0 - 0.9).abs() < 1e-6,
            "Expected max_momentum at step 0, got {}",
            mom0
        );
    }

    // ── OneCycleLrScheduler ──

    #[test]
    fn test_one_cycle_starts_below_max_lr() {
        let sched = OneCycleLrScheduler::new(1e-3, 100).expect("construct");
        let lr0 = sched.get_lr();
        assert!(lr0 < 1e-3, "Initial LR should be below max_lr");
    }

    #[test]
    fn test_one_cycle_peaks_at_max_lr() {
        let total = 100;
        let pct = 0.3;
        let sched = OneCycleLrScheduler::new(1e-3, total).expect("construct").with_pct_start(pct);
        let warmup_end = (total as f64 * pct) as usize;
        let peak_lr = sched.get_lr_at(warmup_end);
        assert!(
            (peak_lr - 1e-3).abs() < 1e-10,
            "Expected max_lr at warmup end, got {}",
            peak_lr
        );
    }

    #[test]
    fn test_one_cycle_ends_below_initial_lr() {
        let sched = OneCycleLrScheduler::new(1e-2, 100)
            .expect("construct")
            .with_div_factor(25.0)
            .with_final_div_factor(1e4);
        let lr_final = sched.get_lr_at(100);
        let initial_lr = 1e-2 / 25.0;
        let min_lr = initial_lr / 1e4;
        assert!(
            (lr_final - min_lr).abs() < 1e-15,
            "Expected min_lr={} at final step, got {}",
            min_lr,
            lr_final
        );
    }

    #[test]
    fn test_one_cycle_linear_annealing() {
        let sched = OneCycleLrScheduler::new(1.0, 10)
            .expect("construct")
            .with_pct_start(0.5)
            .with_anneal_strategy(AnnealStrategy::Linear);
        // Warm-up phase is linear, so LR at step 2 and 3 should differ by a constant delta
        let lr2 = sched.get_lr_at(2);
        let lr3 = sched.get_lr_at(3);
        let lr4 = sched.get_lr_at(4);
        let delta1 = lr3 - lr2;
        let delta2 = lr4 - lr3;
        assert!(
            (delta1 - delta2).abs() < 1e-10,
            "Linear annealing should have constant steps: {} vs {}",
            delta1,
            delta2
        );
    }

    #[test]
    fn test_one_cycle_reset() {
        let mut sched = OneCycleLrScheduler::new(1e-3, 50).expect("construct");
        for _ in 0..20 {
            sched.step();
        }
        sched.reset();
        assert_eq!(sched.step_count, 0);
    }

    #[test]
    fn test_one_cycle_schedule_length() {
        let sched = OneCycleLrScheduler::new(1e-3, 100).expect("construct");
        let sch = sched.schedule(50);
        assert_eq!(sch.len(), 50);
    }

    #[test]
    fn test_one_cycle_invalid_max_lr() {
        assert!(OneCycleLrScheduler::new(0.0, 100).is_err());
        assert!(OneCycleLrScheduler::new(-1.0, 100).is_err());
    }

    #[test]
    fn test_one_cycle_invalid_total_steps() {
        assert!(OneCycleLrScheduler::new(1e-3, 0).is_err());
    }
}
