//! Adaptive curriculum learning strategies.
//!
//! Curriculum learning trains models by ordering examples from easy to hard.
//! This module implements adaptive strategies that adjust the curriculum
//! dynamically based on the model's learning progress.
//!
//! # Key abstractions
//!
//! - [`AdaptiveCurriculumScheduler`] — tracks training progress and decides when
//!   to advance the curriculum to harder examples.
//! - [`SelfPacedLearning`] — lets the model "choose" which examples to include by
//!   excluding samples whose loss exceeds an adaptive threshold `lambda`.
//!
//! # Example
//!
//! ```rust
//! use trustformers_training::curriculum::adaptive::{
//!     AdaptiveCurriculumScheduler, CurriculumAdvanceStrategy, CurriculumAction,
//! };
//!
//! let mut scheduler = AdaptiveCurriculumScheduler::new(
//!     CurriculumAdvanceStrategy::Linear { steps_per_advance: 500 },
//!     5,
//! );
//!
//! for step in 0..2000 {
//!     let action = scheduler.step(2.0 / (1.0 + step as f64 * 0.01), None);
//!     match action {
//!         CurriculumAction::Advance { new_phase, .. } => {
//!             println!("Advanced to phase {}", new_phase);
//!         }
//!         CurriculumAction::Complete => break,
//!         CurriculumAction::Continue => {}
//!     }
//! }
//! ```

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Advance strategy
// ---------------------------------------------------------------------------

/// Determines when the curriculum should advance to harder examples.
#[derive(Debug, Clone, PartialEq)]
pub enum CurriculumAdvanceStrategy {
    /// Advance unconditionally after a fixed number of training steps.
    Linear {
        /// Steps between consecutive curriculum advances.
        steps_per_advance: usize,
    },
    /// Advance when the recent-average loss drops below `threshold` and stays
    /// there for `patience` consecutive steps.
    LossTriggered {
        /// Loss value below which advancement is considered.
        threshold: f64,
        /// Number of consecutive qualifying steps required before advancing.
        patience: usize,
    },
    /// Advance when the recent accuracy exceeds `threshold`.
    AccuracyTriggered {
        /// Accuracy value above which advancement occurs (in [0, 1]).
        threshold: f64,
    },
    /// Advance when the running competence score exceeds `target_competence`.
    CompetenceBased {
        /// Target competence score (in [0, 1]).
        target_competence: f64,
    },
    /// Advance at pre-specified training steps.
    FixedSchedule {
        /// Sorted list of global step counts at which to advance.
        advance_at_steps: Vec<usize>,
    },
}

// ---------------------------------------------------------------------------
// Difficulty bins
// ---------------------------------------------------------------------------

/// A bucket of training examples sharing a difficulty range.
#[derive(Debug, Clone)]
pub struct CurriculumBin {
    /// Zero-based bin identifier.
    pub bin_id: usize,
    /// Inclusive lower and exclusive upper difficulty bounds `[min, max)`.
    pub difficulty_range: (f64, f64),
    /// Number of examples assigned to this bin.
    pub num_examples: usize,
    /// Current sampling weight for this bin.
    pub weight: f64,
}

// ---------------------------------------------------------------------------
// Curriculum state
// ---------------------------------------------------------------------------

/// Internal state of the curriculum scheduler.
#[derive(Debug, Clone)]
pub struct CurriculumState {
    /// Current curriculum phase (0-indexed).
    pub current_phase: usize,
    /// Maximum difficulty allowed in the current phase.
    pub current_max_difficulty: f64,
    /// Total training steps elapsed.
    pub step: usize,
    /// Sliding window of recent loss values for trend analysis.
    pub loss_history: VecDeque<f64>,
    /// Model competence score in `[0, 1]`.
    pub competence: f64,
}

impl Default for CurriculumState {
    fn default() -> Self {
        Self::new()
    }
}

impl CurriculumState {
    /// Create a fresh curriculum state at phase 0.
    pub fn new() -> Self {
        Self {
            current_phase: 0,
            current_max_difficulty: 0.0,
            step: 0,
            loss_history: VecDeque::with_capacity(64),
            competence: 0.0,
        }
    }

    /// Record a new training loss value and update the competence score.
    pub fn update_loss(&mut self, loss: f64) {
        const HISTORY_WINDOW: usize = 32;
        if self.loss_history.len() >= HISTORY_WINDOW {
            self.loss_history.pop_front();
        }
        self.loss_history.push_back(loss);
        // Competence rises as loss falls toward 0.
        self.competence = self.competence * 0.95 + (1.0 / (1.0 + loss)) * 0.05;
    }

    /// Update competence directly from an accuracy measurement.
    pub fn update_accuracy(&mut self, accuracy: f64) {
        self.competence = self.competence * 0.9 + accuracy.clamp(0.0, 1.0) * 0.1;
    }

    /// Compute the recent loss trend.
    ///
    /// Returns a negative value when loss is improving (decreasing) and a
    /// positive value when it is worsening.
    pub fn recent_loss_trend(&self) -> f64 {
        let n = self.loss_history.len();
        if n < 2 {
            return 0.0;
        }
        // Linear regression slope over the loss history.
        let half = n / 2;
        let first_half_mean: f64 =
            self.loss_history.iter().take(half).sum::<f64>() / half as f64;
        let second_half_mean: f64 =
            self.loss_history.iter().skip(half).sum::<f64>() / (n - half) as f64;
        second_half_mean - first_half_mean
    }
}

// ---------------------------------------------------------------------------
// Action type
// ---------------------------------------------------------------------------

/// The action the scheduler recommends after processing a training step.
#[derive(Debug, Clone, PartialEq)]
pub enum CurriculumAction {
    /// Continue at the current curriculum phase.
    Continue,
    /// Advance to a harder phase.
    Advance {
        /// The new phase index.
        new_phase: usize,
        /// The new maximum difficulty threshold.
        new_max_difficulty: f64,
    },
    /// All phases have been completed.
    Complete,
}

// ---------------------------------------------------------------------------
// Progress report
// ---------------------------------------------------------------------------

/// A snapshot of curriculum progress for logging and monitoring.
#[derive(Debug, Clone)]
pub struct CurriculumProgress {
    /// Current curriculum phase (0-indexed).
    pub phase: usize,
    /// Total number of phases.
    pub total_phases: usize,
    /// Progress as a percentage in `[0, 100]`.
    pub progress_pct: f64,
    /// Maximum difficulty currently included.
    pub current_max_difficulty: f64,
    /// Recent loss trend (negative = improving).
    pub recent_loss_trend: f64,
    /// Model competence score.
    pub competence_score: f64,
}

// ---------------------------------------------------------------------------
// Adaptive curriculum scheduler
// ---------------------------------------------------------------------------

/// Adaptive curriculum scheduler that tracks learning progress and decides
/// when to expose the model to harder training examples.
pub struct AdaptiveCurriculumScheduler {
    strategy: CurriculumAdvanceStrategy,
    bins: Vec<CurriculumBin>,
    state: CurriculumState,
    min_difficulty: f64,
    max_difficulty: f64,
    total_phases: usize,
    /// Steps-since-last-advance counter (used by `LossTriggered` patience).
    patience_counter: usize,
    /// Next schedule index (used by `FixedSchedule`).
    schedule_idx: usize,
}

impl AdaptiveCurriculumScheduler {
    /// Create a new scheduler.
    ///
    /// - `strategy`: determines when to advance the curriculum.
    /// - `num_phases`: total number of curriculum phases (must be ≥ 1).
    ///
    /// Difficulty bins are evenly spaced over `[0.0, 1.0]` with `num_phases`
    /// bins.  Sampling weights start at `1.0` for the first bin and `0.0` for
    /// all harder bins; they are updated as the curriculum advances.
    pub fn new(strategy: CurriculumAdvanceStrategy, num_phases: usize) -> Self {
        let num_phases = num_phases.max(1);
        let step_size = 1.0 / num_phases as f64;

        let bins: Vec<CurriculumBin> = (0..num_phases)
            .map(|i| {
                let lo = i as f64 * step_size;
                let hi = (i + 1) as f64 * step_size;
                CurriculumBin {
                    bin_id: i,
                    difficulty_range: (lo, hi),
                    num_examples: 0,
                    weight: if i == 0 { 1.0 } else { 0.0 },
                }
            })
            .collect();

        let mut state = CurriculumState::new();
        state.current_max_difficulty = step_size;

        Self {
            strategy,
            bins,
            state,
            min_difficulty: 0.0,
            max_difficulty: 1.0,
            total_phases: num_phases,
            patience_counter: 0,
            schedule_idx: 0,
        }
    }

    /// Process one training step with the given loss (and optionally accuracy).
    ///
    /// Returns a [`CurriculumAction`] indicating whether the curriculum should
    /// stay, advance, or is complete.
    pub fn step(&mut self, loss: f64, accuracy: Option<f64>) -> CurriculumAction {
        self.state.step += 1;
        self.state.update_loss(loss);
        if let Some(acc) = accuracy {
            self.state.update_accuracy(acc);
        }

        if self.should_advance() {
            self.advance();
            if self.state.current_phase >= self.total_phases {
                return CurriculumAction::Complete;
            }
            return CurriculumAction::Advance {
                new_phase: self.state.current_phase,
                new_max_difficulty: self.state.current_max_difficulty,
            };
        }

        CurriculumAction::Continue
    }

    /// Current maximum difficulty threshold.
    pub fn current_max_difficulty(&self) -> f64 {
        self.state.current_max_difficulty
    }

    /// Current phase index.
    pub fn phase(&self) -> usize {
        self.state.current_phase
    }

    /// Whether the curriculum should advance right now, based on the strategy.
    pub fn should_advance(&mut self) -> bool {
        if self.state.current_phase >= self.total_phases {
            return false;
        }

        match &self.strategy.clone() {
            CurriculumAdvanceStrategy::Linear { steps_per_advance } => {
                self.state.step > 0 && self.state.step % steps_per_advance == 0
            }
            CurriculumAdvanceStrategy::LossTriggered { threshold, patience } => {
                let recent_avg = if self.state.loss_history.is_empty() {
                    f64::INFINITY
                } else {
                    self.state.loss_history.iter().sum::<f64>()
                        / self.state.loss_history.len() as f64
                };
                if recent_avg < *threshold {
                    self.patience_counter += 1;
                    self.patience_counter >= *patience
                } else {
                    self.patience_counter = 0;
                    false
                }
            }
            CurriculumAdvanceStrategy::AccuracyTriggered { threshold } => {
                self.state.competence >= *threshold
            }
            CurriculumAdvanceStrategy::CompetenceBased { target_competence } => {
                self.state.competence >= *target_competence
            }
            CurriculumAdvanceStrategy::FixedSchedule { advance_at_steps } => {
                if self.schedule_idx < advance_at_steps.len() {
                    let target = advance_at_steps[self.schedule_idx];
                    self.state.step >= target
                } else {
                    false
                }
            }
        }
    }

    /// Advance the curriculum to the next phase.
    ///
    /// Updates the difficulty threshold and sampling weights.
    pub fn advance(&mut self) {
        if let CurriculumAdvanceStrategy::FixedSchedule { .. } = &self.strategy {
            self.schedule_idx += 1;
        }
        if let CurriculumAdvanceStrategy::LossTriggered { .. } = &self.strategy {
            self.patience_counter = 0;
        }

        self.state.current_phase += 1;
        if self.state.current_phase >= self.total_phases {
            self.state.current_max_difficulty = self.max_difficulty;
            return;
        }

        let step_size = (self.max_difficulty - self.min_difficulty) / self.total_phases as f64;
        self.state.current_max_difficulty =
            self.min_difficulty + (self.state.current_phase + 1) as f64 * step_size;

        // Update bin weights: use a soft linear ramp up to current phase.
        for bin in &mut self.bins {
            bin.weight = if bin.bin_id <= self.state.current_phase {
                let decay = (self.state.current_phase - bin.bin_id) as f64;
                (1.0 - 0.15 * decay).max(0.1)
            } else {
                0.0
            };
        }
    }

    /// Sampling weights for each difficulty bin at the current state.
    ///
    /// Bins with `weight = 0.0` should not be sampled.
    pub fn sampling_weights(&self) -> Vec<f64> {
        self.bins.iter().map(|b| b.weight).collect()
    }

    /// Build a progress snapshot for logging or reporting.
    pub fn progress_report(&self) -> CurriculumProgress {
        let progress_pct = if self.total_phases == 0 {
            100.0
        } else {
            (self.state.current_phase as f64 / self.total_phases as f64 * 100.0).min(100.0)
        };

        CurriculumProgress {
            phase: self.state.current_phase,
            total_phases: self.total_phases,
            progress_pct,
            current_max_difficulty: self.state.current_max_difficulty,
            recent_loss_trend: self.state.recent_loss_trend(),
            competence_score: self.state.competence,
        }
    }
}

// ---------------------------------------------------------------------------
// Self-paced learning
// ---------------------------------------------------------------------------

/// Self-paced learning (SPL) excludes training examples that are "too hard" for
/// the model at its current stage.
///
/// An example with `difficulty` and `loss` values is included if both:
///
/// - `difficulty ≤ lambda` (difficulty gate), and
/// - `loss ≤ lambda` (loss gate).
///
/// `lambda` starts at `initial_lambda` and increases linearly to `max_lambda`
/// over `total_steps`.
pub struct SelfPacedLearning {
    lambda: f64,
    step: usize,
    lambda_schedule: Vec<f64>,
}

impl SelfPacedLearning {
    /// Create a new SPL instance.
    ///
    /// - `initial_lambda`: starting threshold (low → only easy examples included).
    /// - `max_lambda`: final threshold at `total_steps` (high → all examples included).
    /// - `total_steps`: total number of training steps.
    pub fn new(initial_lambda: f64, max_lambda: f64, total_steps: usize) -> Self {
        let n = total_steps.max(1);
        let schedule: Vec<f64> = (0..=n)
            .map(|i| initial_lambda + (max_lambda - initial_lambda) * (i as f64 / n as f64))
            .collect();

        Self { lambda: initial_lambda, step: 0, lambda_schedule: schedule }
    }

    /// Returns `true` if an example with the given difficulty and loss values
    /// should be included in training at the current lambda.
    pub fn include_example(&self, difficulty: f64, loss: f64) -> bool {
        difficulty <= self.lambda && loss <= self.lambda
    }

    /// Advance the SPL schedule by one step.
    pub fn step(&mut self) {
        self.step += 1;
        if let Some(&next_lambda) = self.lambda_schedule.get(self.step) {
            self.lambda = next_lambda;
        }
    }

    /// Current lambda threshold.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- CurriculumState ---------------------------------------------------

    #[test]
    fn test_state_loss_trend_improving() {
        let mut state = CurriculumState::new();
        // Feed decreasing losses.
        for i in 0..20 {
            state.update_loss(3.0 - i as f64 * 0.1);
        }
        let trend = state.recent_loss_trend();
        assert!(trend < 0.0, "Expected negative trend, got {}", trend);
    }

    #[test]
    fn test_state_loss_trend_worsening() {
        let mut state = CurriculumState::new();
        // Feed increasing losses.
        for i in 0..20 {
            state.update_loss(1.0 + i as f64 * 0.2);
        }
        let trend = state.recent_loss_trend();
        assert!(trend > 0.0, "Expected positive trend, got {}", trend);
    }

    #[test]
    fn test_state_competence_increases_with_accuracy() {
        let mut state = CurriculumState::new();
        for _ in 0..50 {
            state.update_accuracy(0.95);
        }
        assert!(state.competence > 0.3, "Competence should rise with high accuracy");
    }

    // ---- AdaptiveCurriculumScheduler — Linear ------------------------------

    #[test]
    fn test_linear_advance_at_correct_steps() {
        let mut sched = AdaptiveCurriculumScheduler::new(
            CurriculumAdvanceStrategy::Linear { steps_per_advance: 10 },
            5,
        );

        let mut advance_or_complete_count = 0;
        for _ in 1..=50 {
            let action = sched.step(1.0, None);
            match action {
                CurriculumAction::Advance { .. } | CurriculumAction::Complete => {
                    advance_or_complete_count += 1;
                }
                CurriculumAction::Continue => {}
            }
        }
        // 50 steps / 10 steps_per_advance = 5 curriculum transitions
        // (4 Advance + 1 Complete, or all Complete depending on scheduling).
        assert_eq!(advance_or_complete_count, 5);
    }

    #[test]
    fn test_linear_completes_after_all_phases() {
        let mut sched = AdaptiveCurriculumScheduler::new(
            CurriculumAdvanceStrategy::Linear { steps_per_advance: 5 },
            3,
        );

        let mut last_action = CurriculumAction::Continue;
        for _ in 1..=20 {
            let a = sched.step(1.0, None);
            if a == CurriculumAction::Complete {
                last_action = a;
                break;
            }
        }
        assert_eq!(last_action, CurriculumAction::Complete);
    }

    // ---- AdaptiveCurriculumScheduler — LossTriggered -----------------------

    #[test]
    fn test_loss_triggered_advances_when_loss_low() {
        let mut sched = AdaptiveCurriculumScheduler::new(
            CurriculumAdvanceStrategy::LossTriggered { threshold: 0.5, patience: 3 },
            4,
        );

        // Drive loss below threshold for 3 consecutive steps.
        sched.step(0.3, None);
        sched.step(0.3, None);
        let action = sched.step(0.3, None);

        // Should have advanced.
        assert!(
            matches!(action, CurriculumAction::Advance { .. }) || sched.phase() > 0,
            "Should have advanced"
        );
    }

    #[test]
    fn test_loss_triggered_does_not_advance_when_loss_high() {
        let mut sched = AdaptiveCurriculumScheduler::new(
            CurriculumAdvanceStrategy::LossTriggered { threshold: 0.5, patience: 5 },
            4,
        );
        for _ in 0..10 {
            let action = sched.step(2.0, None);
            assert_eq!(action, CurriculumAction::Continue);
        }
        assert_eq!(sched.phase(), 0);
    }

    // ---- AdaptiveCurriculumScheduler — FixedSchedule -----------------------

    #[test]
    fn test_fixed_schedule_advances_at_specified_steps() {
        let mut sched = AdaptiveCurriculumScheduler::new(
            CurriculumAdvanceStrategy::FixedSchedule {
                advance_at_steps: vec![5, 10, 15],
            },
            3,
        );

        let mut transitions = Vec::new();
        for step in 1..=20 {
            let action = sched.step(1.0, None);
            match &action {
                CurriculumAction::Advance { new_phase, .. } => {
                    transitions.push((step, *new_phase));
                }
                CurriculumAction::Complete => {
                    transitions.push((step, usize::MAX));
                    break;
                }
                CurriculumAction::Continue => {}
            }
        }
        // 3 schedule points → 3 transitions total (the last one may be Complete).
        assert_eq!(transitions.len(), 3, "Expected 3 curriculum transitions, got {:?}", transitions);
    }

    // ---- Sampling weights --------------------------------------------------

    #[test]
    fn test_sampling_weights_update_on_advance() {
        let mut sched = AdaptiveCurriculumScheduler::new(
            CurriculumAdvanceStrategy::Linear { steps_per_advance: 1 },
            4,
        );

        // Initially only the first bin has non-zero weight.
        let w0 = sched.sampling_weights();
        assert!(w0[0] > 0.0);
        assert_eq!(w0[1], 0.0);

        // After one step (advance), bin 1 should also get weight.
        sched.step(1.0, None);
        let w1 = sched.sampling_weights();
        assert!(w1[0] > 0.0);
        assert!(w1[1] > 0.0);
    }

    // ---- Progress report ---------------------------------------------------

    #[test]
    fn test_progress_report_structure() {
        let sched = AdaptiveCurriculumScheduler::new(
            CurriculumAdvanceStrategy::Linear { steps_per_advance: 100 },
            5,
        );
        let report = sched.progress_report();
        assert_eq!(report.total_phases, 5);
        assert_eq!(report.phase, 0);
        assert!(report.progress_pct >= 0.0 && report.progress_pct <= 100.0);
    }

    // ---- SelfPacedLearning -------------------------------------------------

    #[test]
    fn test_spl_includes_easy_examples() {
        let spl = SelfPacedLearning::new(0.5, 2.0, 1000);
        // Easy example: low difficulty and low loss.
        assert!(spl.include_example(0.2, 0.3));
    }

    #[test]
    fn test_spl_excludes_hard_examples() {
        let spl = SelfPacedLearning::new(0.5, 2.0, 1000);
        // Hard example: high difficulty.
        assert!(!spl.include_example(0.8, 0.3));
    }

    #[test]
    fn test_spl_lambda_increases_with_steps() {
        let mut spl = SelfPacedLearning::new(0.1, 1.0, 100);
        let initial = spl.lambda();
        for _ in 0..50 {
            spl.step();
        }
        let mid = spl.lambda();
        assert!(mid > initial, "lambda should increase over time");
    }

    #[test]
    fn test_spl_eventually_includes_all() {
        let mut spl = SelfPacedLearning::new(0.1, 10.0, 100);
        for _ in 0..100 {
            spl.step();
        }
        // After max_lambda is reached, virtually all examples should be included.
        assert!(spl.include_example(0.9, 0.9));
    }
}
