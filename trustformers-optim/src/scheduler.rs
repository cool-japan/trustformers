//! # Learning Rate Schedulers
//!
//! This module provides various learning rate scheduling strategies for optimizers.
//! Learning rate scheduling is crucial for achieving good convergence in deep learning.
//!
//! ## Available Schedulers
//!
//! - **LinearScheduler**: Linear warmup followed by linear decay
//! - **CosineScheduler**: Linear warmup followed by cosine annealing
//! - **PolynomialScheduler**: Polynomial decay with configurable power
//! - **ConstantWithWarmupScheduler**: Constant LR after warmup
//! - **ExponentialScheduler**: Exponential decay
//! - **StepScheduler**: Step-wise decay at specified milestones
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_optim::{AdamW, CosineScheduler, LRScheduler};
//! use trustformers_core::traits::Optimizer;
//!
//! let base_lr = 5e-4;
//! let mut optimizer = AdamW::new(base_lr, (0.9, 0.999), 1e-8, 0.01);
//!
//! let mut scheduler = CosineScheduler::new(
//!     base_lr,
//!     1000,   // Linear warmup for 1000 steps
//!     10000,  // Total training steps
//!     1e-5,   // Minimum learning rate
//! );
//!
//! // Training loop
//! for step in 0..10000 {
//!     // Get current learning rate
//!     let lr = scheduler.get_lr(step);
//!     optimizer.set_lr(lr);
//!
//!     // Training step...
//!
//!     scheduler.step();
//! }
//! ```
//!
//! ## Choosing a Scheduler
//!
//! ### For Transformer Pre-training
//! - **CosineScheduler**: Most common, smooth decay
//! - **LinearScheduler**: Simple and effective
//!
//! ### For Fine-tuning
//! - **ConstantWithWarmupScheduler**: Stable for small datasets
//! - **LinearScheduler**: With small decay rate
//!
//! ### For Computer Vision
//! - **StepScheduler**: Traditional for CNNs
//! - **CosineScheduler**: Modern alternative
//!
//! ## Warmup Importance
//!
//! Warmup is crucial for:
//! - Stabilizing training with large learning rates
//! - Preventing early divergence
//! - Allowing adaptive optimizers to estimate statistics
//!
//! Typical warmup steps:
//! - 2-10% of total training steps
//! - 500-2000 steps for most tasks

/// Trait for learning rate schedulers.
pub trait LRScheduler: Send + Sync {
    /// Get the learning rate for a given step.
    fn get_lr(&self, step: usize) -> f32;
    /// Advance the scheduler by one step.
    fn step(&mut self);
}

/// Linear learning rate scheduler with warmup.
///
/// Implements linear warmup from 0 to base_lr, followed by linear decay to 0.
/// This is commonly used for transformer pre-training.
#[derive(Debug)]
pub struct LinearScheduler {
    base_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl LinearScheduler {
    pub fn new(base_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }
}

impl LRScheduler for LinearScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.base_lr * (step as f32) / (self.warmup_steps as f32);
        }

        // `total_steps <= warmup_steps` would underflow the usize subtraction.
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps);
        if decay_steps == 0 {
            return 0.0;
        }

        let progress = (step - self.warmup_steps) as f32 / decay_steps as f32;
        self.base_lr * (1.0 - progress).max(0.0)
    }

    fn step(&mut self) {
        self.current_step += 1;
    }
}

/// Cosine annealing learning rate scheduler with warmup.
///
/// Implements linear warmup followed by cosine decay to min_lr.
/// This provides a smoother decay than linear scheduling and often
/// leads to better final performance.
#[derive(Debug)]
pub struct CosineScheduler {
    base_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
    min_lr: f32,
}

impl CosineScheduler {
    pub fn new(base_lr: f32, warmup_steps: usize, total_steps: usize, min_lr: f32) -> Self {
        Self {
            base_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
            min_lr,
        }
    }
}

impl LRScheduler for CosineScheduler {
    /// Linear warmup, then cosine decay to `min_lr`.
    ///
    /// `progress` is clamped to `[0, 1]`: past `total_steps` the raw cosine turns back
    /// upward and the learning rate would climb above `min_lr` again.
    fn get_lr(&self, step: usize) -> f32 {
        use std::f32::consts::PI;

        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.base_lr * (step as f32) / (self.warmup_steps as f32);
        }

        // `total_steps <= warmup_steps` would underflow the usize subtraction.
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps);
        if decay_steps == 0 {
            return self.min_lr;
        }

        let progress = ((step - self.warmup_steps) as f32 / decay_steps as f32).clamp(0.0, 1.0);
        let cosine_decay = 0.5 * (1.0 + (PI * progress).cos());
        self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
    }

    fn step(&mut self) {
        self.current_step += 1;
    }
}

/// Polynomial decay scheduler with configurable power.
///
/// Decays learning rate according to: lr = (base_lr - min_lr) * (1 - t)^power + min_lr
/// where t is the progress ratio. Common powers:
/// - power = 1.0: Linear decay
/// - power = 0.5: Square root decay
/// - power = 2.0: Quadratic decay
#[derive(Debug)]
pub struct PolynomialScheduler {
    base_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
    min_lr: f32,
    power: f32,
}

impl PolynomialScheduler {
    pub fn new(
        base_lr: f32,
        warmup_steps: usize,
        total_steps: usize,
        min_lr: f32,
        power: f32,
    ) -> Self {
        Self {
            base_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
            min_lr,
            power,
        }
    }
}

impl LRScheduler for PolynomialScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.base_lr * (step as f32) / (self.warmup_steps as f32)
        } else {
            let progress =
                (step - self.warmup_steps) as f32 / (self.total_steps - self.warmup_steps) as f32;
            let decay_factor = (1.0 - progress.min(1.0)).powf(self.power);
            self.min_lr + (self.base_lr - self.min_lr) * decay_factor
        }
    }

    fn step(&mut self) {
        self.current_step += 1;
    }
}

/// Constant learning rate with warmup
#[derive(Debug)]
pub struct ConstantWithWarmupScheduler {
    base_lr: f32,
    warmup_steps: usize,
    current_step: usize,
}

impl ConstantWithWarmupScheduler {
    pub fn new(base_lr: f32, warmup_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
            current_step: 0,
        }
    }
}

impl LRScheduler for ConstantWithWarmupScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.base_lr * (step as f32) / (self.warmup_steps as f32)
        } else {
            self.base_lr
        }
    }

    fn step(&mut self) {
        self.current_step += 1;
    }
}

/// Exponential decay scheduler
#[derive(Debug)]
pub struct ExponentialScheduler {
    base_lr: f32,
    warmup_steps: usize,
    current_step: usize,
    decay_rate: f32,
    decay_steps: usize,
}

impl ExponentialScheduler {
    pub fn new(base_lr: f32, warmup_steps: usize, decay_rate: f32, decay_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
            current_step: 0,
            decay_rate,
            decay_steps,
        }
    }
}

impl LRScheduler for ExponentialScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.base_lr * (step as f32) / (self.warmup_steps as f32)
        } else {
            let decay_step = (step - self.warmup_steps) / self.decay_steps;
            self.base_lr * self.decay_rate.powf(decay_step as f32)
        }
    }

    fn step(&mut self) {
        self.current_step += 1;
    }
}

/// Step decay scheduler (reduce LR at specific steps)
#[derive(Debug)]
pub struct StepScheduler {
    base_lr: f32,
    warmup_steps: usize,
    current_step: usize,
    step_size: usize,
    gamma: f32,
}

impl StepScheduler {
    pub fn new(base_lr: f32, warmup_steps: usize, step_size: usize, gamma: f32) -> Self {
        Self {
            base_lr,
            warmup_steps,
            current_step: 0,
            step_size,
            gamma,
        }
    }
}

impl LRScheduler for StepScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.base_lr * (step as f32) / (self.warmup_steps as f32)
        } else {
            let decay_step = (step - self.warmup_steps) / self.step_size;
            self.base_lr * self.gamma.powf(decay_step as f32)
        }
    }

    fn step(&mut self) {
        self.current_step += 1;
    }
}

/// OneCycle learning rate scheduler.
///
/// Implements the OneCycle policy: ramp up LR to max_lr over pct_start of training,
/// then decay to final_lr for the remainder. This scheduler often enables training
/// with much higher learning rates.
#[derive(Debug)]
pub struct OneCycleScheduler {
    max_lr: f32,
    final_lr: f32,
    total_steps: usize,
    pct_start: f32,
    current_step: usize,
}

impl OneCycleScheduler {
    /// Creates a one-cycle schedule.
    ///
    /// `pct_start` is clamped to the *open* interval `(0, 1)`: the phase formulas
    /// divide by `pct_start` and `1 − pct_start`, so the closed interval yields
    /// `inf`/`NaN` at the endpoints.
    pub fn new(max_lr: f32, total_steps: usize, pct_start: f32, final_lr: f32) -> Self {
        const MIN_PCT: f32 = 1e-3;
        Self {
            max_lr,
            final_lr,
            total_steps,
            pct_start: if pct_start.is_finite() {
                pct_start.clamp(MIN_PCT, 1.0 - MIN_PCT)
            } else {
                0.3
            },
            current_step: 0,
        }
    }
}

impl LRScheduler for OneCycleScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        use std::f32::consts::PI;

        if self.total_steps == 0 {
            return self.final_lr;
        }
        let step = step.min(self.total_steps);
        let pct = step as f32 / self.total_steps as f32;

        if pct <= self.pct_start {
            // Ramp up phase
            let phase_pct = pct / self.pct_start;
            let cosine_term = 0.5 * (1.0 - (PI * phase_pct).cos());
            self.final_lr + (self.max_lr - self.final_lr) * cosine_term
        } else {
            // Decay phase
            let remaining_pct = (pct - self.pct_start) / (1.0 - self.pct_start);
            let cosine_term = 0.5 * (1.0 + (PI * remaining_pct).cos());
            self.final_lr + (self.max_lr - self.final_lr) * cosine_term
        }
    }

    fn step(&mut self) {
        self.current_step += 1;
    }
}

/// Cosine annealing with warm restarts (SGDR).
///
/// Periodically restarts the learning rate schedule. This can help escape
/// local minima and often improves final performance.
#[derive(Debug)]
pub struct CosineWithRestartsScheduler {
    base_lr: f32,
    min_lr: f32,
    t_0: usize,
    t_mult: f32,
    current_step: usize,
    next_restart: usize,
    current_t: usize,
}

impl CosineWithRestartsScheduler {
    /// Creates an SGDR schedule.
    ///
    /// `t_0` is forced to at least one step and `t_mult` to at least `1.0`. A cycle
    /// length of zero — which `t_0 == 0` or `t_mult < 1` produce — makes the
    /// cycle-search loop in [`LRScheduler::get_lr`] subtract zero forever, hanging the
    /// process inside a scheduler call, and then divides by zero.
    ///
    /// Use [`CosineWithRestartsScheduler::try_new`] to be told about an invalid
    /// argument instead of having it silently corrected.
    pub fn new(base_lr: f32, min_lr: f32, t_0: usize, t_mult: f32) -> Self {
        let t_0 = t_0.max(1);
        let t_mult = if t_mult.is_finite() { t_mult.max(1.0) } else { 1.0 };
        Self {
            base_lr,
            min_lr,
            t_0,
            t_mult,
            current_step: 0,
            next_restart: t_0,
            current_t: t_0,
        }
    }

    /// Fallible constructor that rejects a degenerate cycle configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when `t_0 == 0` or `t_mult < 1.0` (or is not finite).
    pub fn try_new(
        base_lr: f32,
        min_lr: f32,
        t_0: usize,
        t_mult: f32,
    ) -> trustformers_core::errors::Result<Self> {
        if t_0 == 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "CosineWithRestartsScheduler requires t_0 >= 1".to_string(),
                ),
            );
        }
        if !t_mult.is_finite() || t_mult < 1.0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(format!(
                    "CosineWithRestartsScheduler requires a finite t_mult >= 1.0, got {t_mult}"
                )),
            );
        }
        Ok(Self::new(base_lr, min_lr, t_0, t_mult))
    }
}

impl LRScheduler for CosineWithRestartsScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        use std::f32::consts::PI;

        let mut step_in_cycle = step;
        // `new`/`try_new` guarantee `t_0 >= 1` and `t_mult >= 1`, so the cycle length
        // can never reach zero and this loop always terminates.
        let mut cycle_length = self.t_0.max(1);

        // Find which cycle we're in
        while step_in_cycle >= cycle_length {
            step_in_cycle -= cycle_length;
            cycle_length = ((cycle_length as f32 * self.t_mult) as usize).max(1);
        }

        let progress = step_in_cycle as f32 / cycle_length as f32;
        let cosine_decay = 0.5 * (1.0 + (PI * progress).cos());

        self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
    }

    fn step(&mut self) {
        self.current_step += 1;

        if self.current_step >= self.next_restart {
            self.current_t = (self.current_t as f32 * self.t_mult) as usize;
            self.next_restart += self.current_t;
        }
    }
}

/// Cyclical learning rate scheduler.
///
/// Cycles the learning rate between base_lr and max_lr over step_size_up + step_size_down steps.
/// This can help find better learning rates and escape local minima.
#[derive(Debug)]
pub struct CyclicalScheduler {
    base_lr: f32,
    max_lr: f32,
    step_size_up: usize,
    step_size_down: usize,
    current_step: usize,
    mode: CyclicalMode,
}

#[derive(Debug, Clone)]
pub enum CyclicalMode {
    Triangular,
    Triangular2,
    ExpRange(f32), // gamma parameter
}

impl CyclicalScheduler {
    pub fn new(
        base_lr: f32,
        max_lr: f32,
        step_size_up: usize,
        step_size_down: usize,
        mode: CyclicalMode,
    ) -> Self {
        Self {
            base_lr,
            max_lr,
            step_size_up,
            step_size_down,
            current_step: 0,
            mode,
        }
    }
}

impl LRScheduler for CyclicalScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        let cycle_length = self.step_size_up + self.step_size_down;
        let cycle = (step / cycle_length) + 1;
        let x = (step % cycle_length) as f32;

        let (amplitude, _phase) = if x <= self.step_size_up as f32 {
            // Ascending phase
            (x / self.step_size_up as f32, 1.0)
        } else {
            // Descending phase
            (
                (self.step_size_down as f32 - (x - self.step_size_up as f32))
                    / self.step_size_down as f32,
                1.0,
            )
        };

        let scale_factor = match &self.mode {
            CyclicalMode::Triangular => 1.0,
            CyclicalMode::Triangular2 => 1.0 / (2.0_f32.powi((cycle - 1) as i32)),
            CyclicalMode::ExpRange(gamma) => gamma.powi(step as i32),
        };

        self.base_lr + (self.max_lr - self.base_lr) * amplitude * scale_factor
    }

    fn step(&mut self) {
        self.current_step += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_scheduler() {
        let scheduler = LinearScheduler::new(1e-3, 100, 1000);

        // Test warmup
        assert_eq!(scheduler.get_lr(0), 0.0);
        assert_eq!(scheduler.get_lr(50), 5e-4);
        assert_eq!(scheduler.get_lr(100), 1e-3);

        // Test decay
        assert_eq!(scheduler.get_lr(550), 5e-4);
        assert_eq!(scheduler.get_lr(1000), 0.0);
    }

    #[test]
    fn test_cosine_scheduler() {
        let scheduler = CosineScheduler::new(1e-3, 100, 1000, 1e-5);

        // Test warmup
        assert_eq!(scheduler.get_lr(0), 0.0);
        assert_eq!(scheduler.get_lr(50), 5e-4);
        assert_eq!(scheduler.get_lr(100), 1e-3);

        // Test cosine decay - should be smooth
        let mid_lr = scheduler.get_lr(550);
        assert!(mid_lr > 1e-5 && mid_lr < 1e-3);

        // Should approach min_lr at the end
        let end_lr = scheduler.get_lr(1000);
        assert!((end_lr - 1e-5).abs() < 1e-6);
    }

    #[test]
    fn test_polynomial_scheduler() {
        let scheduler = PolynomialScheduler::new(1e-3, 100, 1000, 1e-5, 2.0);

        // Test warmup
        assert_eq!(scheduler.get_lr(0), 0.0);
        assert_eq!(scheduler.get_lr(100), 1e-3);

        // Test polynomial decay
        let mid_lr = scheduler.get_lr(550);
        assert!(mid_lr > 1e-5 && mid_lr < 1e-3);
    }

    #[test]
    fn test_constant_with_warmup_scheduler() {
        let scheduler = ConstantWithWarmupScheduler::new(1e-3, 100);

        // Test warmup
        assert_eq!(scheduler.get_lr(0), 0.0);
        assert_eq!(scheduler.get_lr(50), 5e-4);
        assert_eq!(scheduler.get_lr(100), 1e-3);

        // Test constant after warmup
        assert_eq!(scheduler.get_lr(200), 1e-3);
        assert_eq!(scheduler.get_lr(1000), 1e-3);
    }

    #[test]
    fn test_exponential_scheduler() {
        let scheduler = ExponentialScheduler::new(1e-3, 100, 0.9, 100);

        // Test warmup
        assert_eq!(scheduler.get_lr(0), 0.0);
        assert_eq!(scheduler.get_lr(100), 1e-3);

        // Test exponential decay
        assert_eq!(scheduler.get_lr(200), 1e-3 * 0.9);
        assert_eq!(scheduler.get_lr(300), 1e-3 * 0.9 * 0.9);
    }

    #[test]
    fn test_step_scheduler() {
        let scheduler = StepScheduler::new(1e-3, 100, 200, 0.5);

        // Test warmup
        assert_eq!(scheduler.get_lr(0), 0.0);
        assert_eq!(scheduler.get_lr(100), 1e-3);

        // Test step decay
        assert_eq!(scheduler.get_lr(250), 1e-3); // Still first step
        assert_eq!(scheduler.get_lr(300), 1e-3 * 0.5); // Second step
        assert_eq!(scheduler.get_lr(500), 1e-3 * 0.5 * 0.5); // Third step
    }

    #[test]
    fn test_onecycle_scheduler() {
        let scheduler = OneCycleScheduler::new(1e-2, 1000, 0.3, 1e-5);

        // Test start
        assert_eq!(scheduler.get_lr(0), 1e-5);

        // Test peak (around 30% of training)
        let peak_lr = scheduler.get_lr(150);
        assert!(peak_lr > 5e-3);

        // Test end
        let end_lr = scheduler.get_lr(1000);
        assert!((end_lr - 1e-5).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_with_restarts_scheduler() {
        let scheduler = CosineWithRestartsScheduler::new(1e-3, 1e-5, 100, 2.0);

        // Test initial learning rate
        assert!((scheduler.get_lr(0) - 1e-3).abs() < 1e-6);

        // Test mid-cycle (should be between min and max)
        let mid_lr = scheduler.get_lr(50);
        assert!(mid_lr > 1e-5 && mid_lr < 1e-3);

        // Test near end of first cycle (should be close to minimum)
        let near_end_lr = scheduler.get_lr(99);
        assert!(near_end_lr < 2e-4);

        // Test restart (should be back to max)
        let restart_lr = scheduler.get_lr(100);
        assert!(restart_lr > 5e-4);
    }

    #[test]
    fn test_cyclical_scheduler() {
        let scheduler = CyclicalScheduler::new(1e-4, 1e-3, 50, 50, CyclicalMode::Triangular);

        // Test base learning rate
        assert!((scheduler.get_lr(0) - 1e-4).abs() < 1e-6);

        // Test peak learning rate
        assert!((scheduler.get_lr(50) - 1e-3).abs() < 1e-6);

        // Test return to base
        assert!((scheduler.get_lr(100) - 1e-4).abs() < 1e-6);

        // Test second cycle
        assert!((scheduler.get_lr(150) - 1e-3).abs() < 1e-6);
    }
}

/// Adaptive learning rate scheduler that reduces LR when a metric has stopped improving.
///
/// This scheduler monitors a metric (typically validation loss) and reduces the learning rate
/// when the metric plateaus for a certain number of epochs. Similar to ReduceLROnPlateau
/// in PyTorch, this provides adaptive learning rate scheduling based on actual training progress.
#[derive(Debug, Clone)]
pub struct AdaptiveScheduler {
    /// Current learning rate
    current_lr: f32,
    /// Factor by which to reduce learning rate (new_lr = lr * factor)
    factor: f32,
    /// Number of epochs with no improvement after which LR will be reduced
    patience: usize,
    /// Threshold for measuring the new optimum (relative improvement)
    threshold: f32,
    /// Minimum learning rate (will not go below this)
    min_lr: f32,
    /// Mode: "min" for minimizing (loss), "max" for maximizing (accuracy)
    mode: String,
    /// Number of epochs since last improvement
    epochs_since_improvement: usize,
    /// Best metric value seen so far
    best_metric: Option<f32>,
    /// Step counter
    current_step: usize,
}

impl AdaptiveScheduler {
    /// Creates a new adaptive scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `factor` - Factor by which to reduce LR (typical: 0.1 to 0.5)
    /// * `patience` - Number of epochs to wait before reducing LR (typical: 5-10)
    /// * `threshold` - Minimum improvement threshold (typical: 1e-4)
    /// * `min_lr` - Minimum learning rate (typical: 1e-8)
    /// * `mode` - "min" for loss, "max" for accuracy
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_optim::AdaptiveScheduler;
    ///
    /// let scheduler = AdaptiveScheduler::new(1e-3, 0.1, 5, 1e-4, 1e-8, "min");
    /// ```
    /// # Panics
    ///
    /// Panics when any argument is out of range (`factor` outside `(0, 1)`,
    /// `patience == 0`, negative `threshold`/`min_lr`, or a `mode` other than `"min"`
    /// or `"max"`). Prefer [`AdaptiveScheduler::try_new`], which returns an error, for
    /// values that come from a configuration file.
    pub fn new(
        initial_lr: f32,
        factor: f32,
        patience: usize,
        threshold: f32,
        min_lr: f32,
        mode: &str,
    ) -> Self {
        match Self::try_new(initial_lr, factor, patience, threshold, min_lr, mode) {
            Ok(scheduler) => scheduler,
            Err(error) => panic!("invalid AdaptiveScheduler configuration: {error}"),
        }
    }

    /// Fallible constructor: every guard reports an error instead of aborting.
    ///
    /// The `mode` string is the most dangerous of these — a typo like `"Min"` in a
    /// config file used to kill the training process.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidConfiguration`](trustformers_core::errors::ErrorKind)
    /// when `factor` is not in `(0, 1)`, `patience` is zero, `threshold` or `min_lr`
    /// is negative, or `mode` is not `"min"`/`"max"`.
    pub fn try_new(
        initial_lr: f32,
        factor: f32,
        patience: usize,
        threshold: f32,
        min_lr: f32,
        mode: &str,
    ) -> trustformers_core::errors::Result<Self> {
        use trustformers_core::errors::TrustformersError;

        if !(factor > 0.0 && factor < 1.0) {
            return Err(TrustformersError::invalid_config(format!(
                "AdaptiveScheduler factor must be in (0, 1), got {factor}"
            )));
        }
        if patience == 0 {
            return Err(TrustformersError::invalid_config(
                "AdaptiveScheduler patience must be positive".to_string(),
            ));
        }
        if threshold < 0.0 {
            return Err(TrustformersError::invalid_config(format!(
                "AdaptiveScheduler threshold must be non-negative, got {threshold}"
            )));
        }
        if min_lr < 0.0 {
            return Err(TrustformersError::invalid_config(format!(
                "AdaptiveScheduler min_lr must be non-negative, got {min_lr}"
            )));
        }
        if mode != "min" && mode != "max" {
            return Err(TrustformersError::invalid_config(format!(
                "AdaptiveScheduler mode must be \"min\" or \"max\", got \"{mode}\""
            )));
        }

        Ok(Self {
            current_lr: initial_lr,
            factor,
            patience,
            threshold,
            min_lr,
            mode: mode.to_string(),
            epochs_since_improvement: 0,
            best_metric: None,
            current_step: 0,
        })
    }

    /// Update the scheduler with a new metric value.
    /// Returns the new learning rate and whether it was reduced.
    pub fn step_with_metric(&mut self, metric: f32) -> (f32, bool) {
        self.current_step += 1;
        let mut lr_reduced = false;

        let is_improvement = match self.best_metric {
            None => {
                // First metric, set as best
                self.best_metric = Some(metric);
                true
            },
            Some(best) => {
                let improvement = if self.mode == "min" {
                    // For minimizing (loss), improvement is when metric decreases
                    (best - metric) / best.abs().max(1e-8) > self.threshold
                } else {
                    // For maximizing (accuracy), improvement is when metric increases
                    (metric - best) / best.abs().max(1e-8) > self.threshold
                };

                if improvement {
                    self.best_metric = Some(metric);
                }

                improvement
            },
        };

        if is_improvement {
            self.epochs_since_improvement = 0;
        } else {
            self.epochs_since_improvement += 1;

            if self.epochs_since_improvement >= self.patience {
                // Reduce learning rate
                let new_lr = (self.current_lr * self.factor).max(self.min_lr);
                if new_lr < self.current_lr {
                    self.current_lr = new_lr;
                    lr_reduced = true;
                    self.epochs_since_improvement = 0; // Reset patience counter
                }
            }
        }

        (self.current_lr, lr_reduced)
    }

    /// Get current learning rate without updating.
    pub fn get_current_lr(&self) -> f32 {
        self.current_lr
    }

    /// Get the best metric seen so far.
    pub fn get_best_metric(&self) -> Option<f32> {
        self.best_metric
    }

    /// Get epochs since last improvement.
    pub fn get_epochs_since_improvement(&self) -> usize {
        self.epochs_since_improvement
    }

    /// Reset the scheduler state.
    pub fn reset(&mut self) {
        self.epochs_since_improvement = 0;
        self.best_metric = None;
        self.current_step = 0;
    }

    /// Set the learning rate manually.
    pub fn set_lr(&mut self, lr: f32) {
        self.current_lr = lr;
    }
}

impl LRScheduler for AdaptiveScheduler {
    fn get_lr(&self, _step: usize) -> f32 {
        self.current_lr
    }

    fn step(&mut self) {
        // For adaptive scheduler, stepping is done via step_with_metric
        // This method is kept for compatibility with the LRScheduler trait
    }
}

/// A composite scheduler that chains multiple schedulers together.
///
/// This allows combining different scheduling strategies, e.g., warmup + cosine + linear decay.
/// Each scheduler is active for a specified number of steps.
pub struct CompositeScheduler {
    schedulers: Vec<Box<dyn LRScheduler>>,
    step_boundaries: Vec<usize>,
    current_step: usize,
    // reason: reserved for global-step offsetting across composed schedulers;
    // retained intentionally for an in-progress feature.
    #[allow(dead_code)]
    global_step_offset: usize,
}

impl CompositeScheduler {
    /// Creates a new composite scheduler.
    ///
    /// # Arguments
    /// * `schedulers` - Vector of schedulers to chain
    /// * `step_boundaries` - Steps at which to switch to the next scheduler
    ///
    /// # Example
    /// ```rust,no_run
    /// use trustformers_optim::{LinearScheduler, CosineScheduler, CompositeScheduler, LRScheduler};
    ///
    /// let warmup = Box::new(LinearScheduler::new(1e-4, 1000, 1000));
    /// let main = Box::new(CosineScheduler::new(1e-4, 0, 9000, 1e-6));
    /// let composite = CompositeScheduler::new(
    ///     vec![warmup, main],
    ///     vec![1000, 10000]
    /// );
    /// ```
    /// # Panics
    ///
    /// Panics when `schedulers` is empty or its length differs from
    /// `step_boundaries`. Use [`CompositeScheduler::try_new`] for values that come
    /// from a configuration file.
    pub fn new(schedulers: Vec<Box<dyn LRScheduler>>, step_boundaries: Vec<usize>) -> Self {
        match Self::try_new(schedulers, step_boundaries) {
            Ok(scheduler) => scheduler,
            Err(error) => panic!("invalid CompositeScheduler configuration: {error}"),
        }
    }

    /// Fallible constructor.
    ///
    /// # Errors
    ///
    /// Returns an error when `schedulers` is empty or its length differs from
    /// `step_boundaries`.
    pub fn try_new(
        schedulers: Vec<Box<dyn LRScheduler>>,
        step_boundaries: Vec<usize>,
    ) -> trustformers_core::errors::Result<Self> {
        use trustformers_core::errors::TrustformersError;

        if schedulers.is_empty() {
            return Err(TrustformersError::invalid_config(
                "CompositeScheduler needs at least one scheduler".to_string(),
            ));
        }
        if schedulers.len() != step_boundaries.len() {
            return Err(TrustformersError::invalid_config(format!(
                "CompositeScheduler has {} schedulers but {} boundaries",
                schedulers.len(),
                step_boundaries.len()
            )));
        }

        Ok(Self {
            schedulers,
            step_boundaries,
            current_step: 0,
            global_step_offset: 0,
        })
    }

    fn get_active_scheduler_index(&self, step: usize) -> usize {
        for (i, &boundary) in self.step_boundaries.iter().enumerate() {
            if step < boundary {
                return i;
            }
        }
        self.schedulers.len() - 1
    }

    fn get_local_step(&self, global_step: usize, scheduler_index: usize) -> usize {
        if scheduler_index == 0 {
            global_step
        } else {
            global_step - self.step_boundaries[scheduler_index - 1]
        }
    }
}

impl LRScheduler for CompositeScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        let scheduler_idx = self.get_active_scheduler_index(step);
        let local_step = self.get_local_step(step, scheduler_idx);
        self.schedulers[scheduler_idx].get_lr(local_step)
    }

    fn step(&mut self) {
        self.current_step += 1;
        let _scheduler_idx = self.get_active_scheduler_index(self.current_step);
        // Note: Individual schedulers manage their own state
    }
}

/// A phase-based scheduler that applies different scheduling strategies during training phases.
///
/// This is useful for complex training regimes like pre-training -> fine-tuning -> evaluation.
pub struct PhaseBasedScheduler {
    phases: Vec<Phase>,
    current_phase: usize,
    current_step: usize,
    phase_start_step: usize,
}

pub struct Phase {
    pub name: String,
    pub scheduler: Box<dyn LRScheduler>,
    pub duration_steps: usize,
    pub lr_multiplier: f32,
}

impl PhaseBasedScheduler {
    /// Creates a new phase-based scheduler.
    ///
    /// # Example
    /// ```rust,no_run
    /// use trustformers_optim::{Phase, LinearScheduler, CosineScheduler, ConstantWithWarmupScheduler, PhaseBasedScheduler};
    ///
    /// let phases = vec![
    ///     Phase {
    ///         name: "warmup".to_string(),
    ///         scheduler: Box::new(LinearScheduler::new(1e-4, 1000, 1000)),
    ///         duration_steps: 1000,
    ///         lr_multiplier: 1.0,
    ///     },
    ///     Phase {
    ///         name: "main_training".to_string(),
    ///         scheduler: Box::new(CosineScheduler::new(1e-4, 0, 9000, 1e-6)),
    ///         duration_steps: 9000,
    ///         lr_multiplier: 1.0,
    ///     },
    ///     Phase {
    ///         name: "fine_tuning".to_string(),
    ///         scheduler: Box::new(ConstantWithWarmupScheduler::new(1e-5, 0)),
    ///         duration_steps: 1000,
    ///         lr_multiplier: 0.1,
    ///     },
    /// ];
    /// let scheduler = PhaseBasedScheduler::new(phases);
    /// ```
    /// # Panics
    ///
    /// Panics when `phases` is empty. Use [`PhaseBasedScheduler::try_new`] for values
    /// that come from a configuration file.
    pub fn new(phases: Vec<Phase>) -> Self {
        match Self::try_new(phases) {
            Ok(scheduler) => scheduler,
            Err(error) => panic!("invalid PhaseBasedScheduler configuration: {error}"),
        }
    }

    /// Fallible constructor.
    ///
    /// # Errors
    ///
    /// Returns an error when `phases` is empty.
    pub fn try_new(phases: Vec<Phase>) -> trustformers_core::errors::Result<Self> {
        if phases.is_empty() {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "PhaseBasedScheduler needs at least one phase".to_string(),
                ),
            );
        }

        Ok(Self {
            phases,
            current_phase: 0,
            current_step: 0,
            phase_start_step: 0,
        })
    }

    /// Get the current phase name.
    pub fn get_current_phase(&self) -> &str {
        &self.phases[self.current_phase].name
    }

    /// Get the current phase index.
    pub fn get_current_phase_index(&self) -> usize {
        self.current_phase
    }

    /// Check if training is complete (all phases finished).
    pub fn is_complete(&self) -> bool {
        self.current_phase >= self.phases.len()
    }

    fn update_phase(&mut self, step: usize) {
        while self.current_phase < self.phases.len() {
            let phase_end = self.phase_start_step + self.phases[self.current_phase].duration_steps;

            if step < phase_end {
                break; // Still in current phase
            }

            // Move to next phase
            self.current_phase += 1;
            self.phase_start_step = phase_end;
        }
    }
}

impl LRScheduler for PhaseBasedScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        if self.current_phase >= self.phases.len() {
            return 0.0; // Training complete
        }

        let phase = &self.phases[self.current_phase];
        let phase_step = step - self.phase_start_step;
        let base_lr = phase.scheduler.get_lr(phase_step);

        base_lr * phase.lr_multiplier
    }

    fn step(&mut self) {
        self.current_step += 1;
        self.update_phase(self.current_step);
    }
}

/// A dynamic scheduler that adjusts its behavior based on training metrics.
///
/// This scheduler can dynamically switch between different scheduling strategies
/// based on training progress, loss trends, or other metrics.
pub struct DynamicScheduler {
    primary_scheduler: Box<dyn LRScheduler>,
    fallback_scheduler: Box<dyn LRScheduler>,
    current_scheduler: usize, // 0 = primary, 1 = fallback
    switch_condition: SwitchCondition,
    metrics_window: Vec<f32>,
    window_size: usize,
    current_step: usize,
}

#[derive(Debug)]
pub enum SwitchCondition {
    /// Switch when loss stops improving for N steps
    LossPlateauSteps(usize),
    /// Switch when gradient norm exceeds threshold
    GradientNormThreshold(f32),
    /// Switch at specific step
    StepThreshold(usize),
    /// Switch when loss increases by factor
    LossIncreaseFactor(f32),
}

impl DynamicScheduler {
    /// Creates a new dynamic scheduler.
    pub fn new(
        primary_scheduler: Box<dyn LRScheduler>,
        fallback_scheduler: Box<dyn LRScheduler>,
        switch_condition: SwitchCondition,
        window_size: usize,
    ) -> Self {
        Self {
            primary_scheduler,
            fallback_scheduler,
            current_scheduler: 0,
            switch_condition,
            metrics_window: Vec::with_capacity(window_size),
            window_size,
            current_step: 0,
        }
    }

    /// Update with a new metric (e.g., loss value).
    pub fn update_metric(&mut self, metric: f32) {
        self.metrics_window.push(metric);
        if self.metrics_window.len() > self.window_size {
            self.metrics_window.remove(0);
        }

        // Check switch condition
        if self.current_scheduler == 0 && self.should_switch() {
            self.current_scheduler = 1;
        }
    }

    fn should_switch(&self) -> bool {
        match &self.switch_condition {
            SwitchCondition::LossPlateauSteps(steps) => {
                if self.metrics_window.len() < *steps {
                    return false;
                }

                let recent_avg =
                    self.metrics_window.iter().rev().take(*steps).sum::<f32>() / *steps as f32;
                let older_avg =
                    self.metrics_window.iter().take(self.metrics_window.len() - steps).sum::<f32>()
                        / (self.metrics_window.len() - steps) as f32;

                recent_avg >= older_avg * 0.995 // Less than 0.5% improvement
            },
            SwitchCondition::StepThreshold(step) => self.current_step >= *step,
            SwitchCondition::LossIncreaseFactor(factor) => {
                if self.metrics_window.len() < 2 {
                    return false;
                }
                let latest = self.metrics_window[self.metrics_window.len() - 1];
                let previous = self.metrics_window[self.metrics_window.len() - 2];
                latest > previous * factor
            },
            SwitchCondition::GradientNormThreshold(_) => false, // Requires external gradient norm input
        }
    }

    /// Get which scheduler is currently active.
    pub fn get_active_scheduler(&self) -> &str {
        if self.current_scheduler == 0 {
            "primary"
        } else {
            "fallback"
        }
    }
}

impl LRScheduler for DynamicScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        if self.current_scheduler == 0 {
            self.primary_scheduler.get_lr(step)
        } else {
            self.fallback_scheduler.get_lr(step)
        }
    }

    fn step(&mut self) {
        self.current_step += 1;
        if self.current_scheduler == 0 {
            self.primary_scheduler.step();
        } else {
            self.fallback_scheduler.step();
        }
    }
}

/// A task-specific scheduler optimized for different ML tasks.
pub struct TaskSpecificScheduler {
    scheduler: Box<dyn LRScheduler>,
    task_type: TaskType,
    current_step: usize,
}

#[derive(Debug)]
pub enum TaskType {
    /// Language model pre-training (warmup + cosine decay)
    LanguageModelPretraining,
    /// Fine-tuning (low LR, minimal decay)
    FineTuning,
    /// Computer vision (step decay)
    ComputerVision,
    /// Reinforcement learning (adaptive)
    ReinforcementLearning,
    /// GAN training (alternating or constant)
    GANTraining,
}

impl TaskSpecificScheduler {
    /// Creates a task-specific scheduler with optimal defaults.
    pub fn new(task_type: TaskType, base_lr: f32, total_steps: usize) -> Self {
        let scheduler: Box<dyn LRScheduler> = match task_type {
            TaskType::LanguageModelPretraining => {
                Box::new(CosineScheduler::new(
                    base_lr,
                    (total_steps as f32 * 0.06) as usize, // 6% warmup
                    total_steps,
                    base_lr * 0.1, // Decay to 10% of base LR
                ))
            },
            TaskType::FineTuning => {
                Box::new(LinearScheduler::new(
                    base_lr * 0.1,                       // Lower LR for fine-tuning
                    (total_steps as f32 * 0.1) as usize, // 10% warmup
                    total_steps,
                ))
            },
            TaskType::ComputerVision => {
                Box::new(StepScheduler::new(
                    base_lr,
                    (total_steps as f32 * 0.05) as usize, // 5% warmup
                    total_steps / 3,                      // Step every 1/3 of training
                    0.1,                                  // Decay by factor of 10
                ))
            },
            TaskType::ReinforcementLearning => {
                Box::new(AdaptiveScheduler::new(
                    base_lr,
                    0.5,            // Moderate reduction factor
                    10,             // Patience
                    1e-4,           // Threshold
                    base_lr * 1e-3, // Min LR
                    "max",          // Maximize reward
                ))
            },
            TaskType::GANTraining => {
                Box::new(ConstantWithWarmupScheduler::new(
                    base_lr,
                    (total_steps as f32 * 0.02) as usize, // 2% warmup
                ))
            },
        };

        Self {
            scheduler,
            task_type,
            current_step: 0,
        }
    }

    /// Get the task type.
    pub fn get_task_type(&self) -> &TaskType {
        &self.task_type
    }
}

impl LRScheduler for TaskSpecificScheduler {
    fn get_lr(&self, step: usize) -> f32 {
        self.scheduler.get_lr(step)
    }

    fn step(&mut self) {
        self.current_step += 1;
        self.scheduler.step();
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;

    /// Regression: `t_mult < 1` (or `t_0 == 0`) drove the cycle length to zero, after
    /// which `step_in_cycle -= 0` looped forever inside a scheduler call.
    #[test]
    fn cosine_with_restarts_never_hangs() {
        for (t_0, t_mult) in [(0_usize, 2.0_f32), (10, 0.5), (0, 0.0), (5, f32::NAN)] {
            let scheduler = CosineWithRestartsScheduler::new(1e-3, 1e-5, t_0, t_mult);
            // If the guard were missing this call would never return.
            let lr = scheduler.get_lr(1_000);
            assert!(lr.is_finite(), "t_0={t_0}, t_mult={t_mult} produced {lr}");
        }
    }

    /// The fallible constructor reports the degenerate cases instead of correcting them.
    #[test]
    fn cosine_with_restarts_try_new_validates() {
        assert!(CosineWithRestartsScheduler::try_new(1e-3, 1e-5, 10, 2.0).is_ok());
        assert!(CosineWithRestartsScheduler::try_new(1e-3, 1e-5, 0, 2.0).is_err());
        assert!(CosineWithRestartsScheduler::try_new(1e-3, 1e-5, 10, 0.5).is_err());
        assert!(CosineWithRestartsScheduler::try_new(1e-3, 1e-5, 10, f32::NAN).is_err());
    }

    /// Regression: past `total_steps` the unclamped cosine turned back upward and the
    /// learning rate climbed above `min_lr` again.
    #[test]
    fn cosine_lr_never_rises_after_the_schedule_ends() {
        let scheduler = CosineScheduler::new(1e-3, 100, 1000, 1e-5);

        assert!((scheduler.get_lr(0) - 0.0).abs() < 1e-9);
        assert!((scheduler.get_lr(100) - 1e-3).abs() < 1e-9);
        let at_end = scheduler.get_lr(1000);
        assert!((at_end - 1e-5).abs() < 1e-7, "at total_steps: {at_end}");

        for step in [1001_usize, 1500, 2000, 10_000] {
            let lr = scheduler.get_lr(step);
            assert!(
                (lr - 1e-5).abs() < 1e-7,
                "step {step} must stay at min_lr, got {lr}"
            );
        }
    }

    /// `total_steps <= warmup_steps` used to underflow the usize subtraction.
    #[test]
    fn degenerate_step_counts_do_not_underflow() {
        let cosine = CosineScheduler::new(1e-3, 1000, 100, 1e-5);
        assert!(cosine.get_lr(2000).is_finite());

        let linear = LinearScheduler::new(1e-3, 1000, 100);
        assert!(linear.get_lr(2000).is_finite());
    }

    /// `pct_start` of exactly 0 or 1 used to divide by zero.
    #[test]
    fn one_cycle_endpoints_stay_finite() {
        for pct_start in [0.0_f32, 1.0, -1.0, 2.0, f32::NAN] {
            let scheduler = OneCycleScheduler::new(1e-2, 1000, pct_start, 1e-5);
            for step in [0_usize, 1, 500, 999, 1000, 5000] {
                let lr = scheduler.get_lr(step);
                assert!(lr.is_finite(), "pct_start={pct_start}, step={step} -> {lr}");
            }
        }

        // A zero-length schedule must not divide by zero either.
        let empty = OneCycleScheduler::new(1e-2, 0, 0.3, 1e-5);
        assert!(empty.get_lr(0).is_finite());
    }

    /// Regression: eight public constructors aborted the process on a bad argument.
    #[test]
    fn constructors_report_invalid_configuration() {
        assert!(AdaptiveScheduler::try_new(1e-3, 0.1, 5, 1e-4, 1e-8, "min").is_ok());
        assert!(AdaptiveScheduler::try_new(1e-3, 1.5, 5, 1e-4, 1e-8, "min").is_err());
        assert!(AdaptiveScheduler::try_new(1e-3, 0.1, 0, 1e-4, 1e-8, "min").is_err());
        assert!(AdaptiveScheduler::try_new(1e-3, 0.1, 5, -1.0, 1e-8, "min").is_err());
        assert!(AdaptiveScheduler::try_new(1e-3, 0.1, 5, 1e-4, -1.0, "min").is_err());
        // The stringly-typed mode is the one most likely to come from a config file.
        assert!(AdaptiveScheduler::try_new(1e-3, 0.1, 5, 1e-4, 1e-8, "Min").is_err());

        assert!(CompositeScheduler::try_new(Vec::new(), Vec::new()).is_err());
        assert!(CompositeScheduler::try_new(
            vec![Box::new(LinearScheduler::new(1e-3, 10, 100)) as Box<dyn LRScheduler>],
            vec![10, 20],
        )
        .is_err());

        assert!(PhaseBasedScheduler::try_new(Vec::new()).is_err());
    }
}
