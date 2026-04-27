//! Curriculum learning modules.
//!
//! This module contains adaptive curriculum learning strategies that complement
//! the curriculum-learning utilities already present in [`crate::data_pipeline`].
//!
//! # Modules
//!
//! - [`adaptive`]: Adaptive and self-paced curriculum scheduling.
//! - [`scheduler`]: Core curriculum scheduler with multiple strategies.

pub mod adaptive;
pub mod scheduler;

pub use adaptive::{
    AdaptiveCurriculumScheduler, CurriculumAction, CurriculumAdvanceStrategy, CurriculumBin,
    CurriculumProgress, CurriculumState, SelfPacedLearning,
};
pub use scheduler::{
    CurriculumError, CurriculumErrorKind, CurriculumScheduler, CurriculumStrategy,
    CurriculumWindow, anti_curriculum_indices, sort_by_length, sort_by_perplexity,
};

#[cfg(test)]
mod tests {
    use super::adaptive::{
        AdaptiveCurriculumScheduler, CurriculumAction, CurriculumAdvanceStrategy,
        CurriculumState, SelfPacedLearning,
    };
    use super::scheduler::{CurriculumScheduler, CurriculumStrategy};

    // -----------------------------------------------------------------------
    // Re-exported symbol availability
    // -----------------------------------------------------------------------

    #[test]
    fn test_curriculum_advance_strategy_linear_is_accessible() {
        let strategy = CurriculumAdvanceStrategy::Linear { steps_per_advance: 100 };
        let _sched = AdaptiveCurriculumScheduler::new(strategy, 4);
    }

    #[test]
    fn test_curriculum_state_accessible_from_mod() {
        let state = CurriculumState::new();
        assert_eq!(state.current_phase, 0, "initial phase must be 0");
        assert_eq!(state.step, 0, "initial step must be 0");
    }

    #[test]
    fn test_curriculum_scheduler_from_mod() {
        let sched = CurriculumScheduler::new(
            CurriculumStrategy::ByLength {
                initial_max_len: 64,
                final_max_len: 512,
                ramp_steps: 1000,
            },
            100,
        ).expect("scheduler creation must succeed");
        assert!(sched.total_samples() > 0, "scheduler must report positive total samples");
    }

    // -----------------------------------------------------------------------
    // AdaptiveCurriculumScheduler integration via re-exports
    // -----------------------------------------------------------------------

    #[test]
    fn test_linear_scheduler_advances_through_all_phases() {
        let mut sched = AdaptiveCurriculumScheduler::new(
            CurriculumAdvanceStrategy::Linear { steps_per_advance: 5 },
            4,
        );

        let mut completed = false;
        for _ in 0..100 {
            let action = sched.step(1.0, None);
            if action == CurriculumAction::Complete {
                completed = true;
                break;
            }
        }
        assert!(completed, "linear scheduler should complete all phases within 100 steps");
    }

    #[test]
    fn test_difficulty_increases_as_phases_advance() {
        let mut sched = AdaptiveCurriculumScheduler::new(
            CurriculumAdvanceStrategy::Linear { steps_per_advance: 3 },
            3,
        );
        let initial_difficulty = sched.current_max_difficulty();

        // Drive past the first advance boundary.
        for _ in 0..4 {
            sched.step(1.0, None);
        }
        let after_advance_difficulty = sched.current_max_difficulty();
        assert!(
            after_advance_difficulty >= initial_difficulty,
            "max difficulty should be non-decreasing as curriculum advances"
        );
    }

    #[test]
    fn test_sampling_weights_len_matches_phases() {
        let num_phases = 5_usize;
        let sched = AdaptiveCurriculumScheduler::new(
            CurriculumAdvanceStrategy::Linear { steps_per_advance: 10 },
            num_phases,
        );
        assert_eq!(
            sched.sampling_weights().len(),
            num_phases,
            "sampling_weights must have one entry per phase"
        );
    }

    #[test]
    fn test_first_bin_weight_starts_at_one() {
        let sched = AdaptiveCurriculumScheduler::new(
            CurriculumAdvanceStrategy::Linear { steps_per_advance: 100 },
            5,
        );
        let weights = sched.sampling_weights();
        assert!(
            (weights[0] - 1.0).abs() < 1e-9,
            "first bin weight should start at 1.0, got {}",
            weights[0]
        );
    }

    #[test]
    fn test_higher_bins_start_at_zero_weight() {
        let sched = AdaptiveCurriculumScheduler::new(
            CurriculumAdvanceStrategy::Linear { steps_per_advance: 100 },
            5,
        );
        let weights = sched.sampling_weights();
        for (i, &w) in weights.iter().enumerate().skip(1) {
            assert!(
                w < 1e-9,
                "bin {} should start with zero weight, got {}",
                i,
                w
            );
        }
    }

    #[test]
    fn test_progress_report_initial_state() {
        let sched = AdaptiveCurriculumScheduler::new(
            CurriculumAdvanceStrategy::Linear { steps_per_advance: 100 },
            4,
        );
        let progress = sched.progress_report();
        assert_eq!(progress.phase, 0, "initial phase should be 0");
        assert_eq!(progress.total_phases, 4, "total phases should be 4");
        assert!(
            progress.progress_pct >= 0.0 && progress.progress_pct <= 100.0,
            "progress_pct should be in [0, 100]"
        );
    }

    #[test]
    fn test_loss_triggered_advances_when_loss_low() {
        let mut sched = AdaptiveCurriculumScheduler::new(
            CurriculumAdvanceStrategy::LossTriggered { threshold: 0.5, patience: 3 },
            4,
        );

        let mut advanced = false;
        for _ in 0..50 {
            let action = sched.step(0.1, None); // loss well below threshold
            if matches!(action, CurriculumAction::Advance { .. } | CurriculumAction::Complete) {
                advanced = true;
                break;
            }
        }
        assert!(advanced, "loss-triggered scheduler should advance when loss stays below threshold");
    }

    #[test]
    fn test_loss_triggered_does_not_advance_with_high_loss() {
        let mut sched = AdaptiveCurriculumScheduler::new(
            CurriculumAdvanceStrategy::LossTriggered { threshold: 0.3, patience: 5 },
            4,
        );

        let mut advanced = false;
        for _ in 0..20 {
            let action = sched.step(2.0, None); // loss well above threshold
            if matches!(action, CurriculumAction::Advance { .. }) {
                advanced = true;
            }
        }
        assert!(!advanced, "scheduler should not advance when loss stays above threshold");
    }

    #[test]
    fn test_fixed_schedule_advances_at_specified_steps() {
        let mut sched = AdaptiveCurriculumScheduler::new(
            CurriculumAdvanceStrategy::FixedSchedule { advance_at_steps: vec![5, 10, 15] },
            4,
        );

        let mut advance_steps = Vec::new();
        for step in 1_usize..=20 {
            let action = sched.step(1.0, None);
            if matches!(action, CurriculumAction::Advance { .. }) {
                advance_steps.push(step);
            }
        }
        assert!(!advance_steps.is_empty(), "should have at least one advance with fixed schedule");
    }

    // -----------------------------------------------------------------------
    // SelfPacedLearning
    // -----------------------------------------------------------------------

    #[test]
    fn test_spl_excludes_hard_examples_initially() {
        let spl = SelfPacedLearning::new(0.3, 1.0, 100);
        // Hard example: both difficulty and loss above initial lambda
        assert!(
            !spl.include_example(0.8, 0.9),
            "SPL should exclude examples harder than initial lambda"
        );
    }

    #[test]
    fn test_spl_includes_easy_examples() {
        let spl = SelfPacedLearning::new(0.5, 1.0, 100);
        // Easy example: both values below lambda
        assert!(
            spl.include_example(0.2, 0.1),
            "SPL should include easy examples within lambda"
        );
    }

    #[test]
    fn test_spl_lambda_increases_with_steps() {
        let mut spl = SelfPacedLearning::new(0.1, 1.0, 100);
        let initial_lambda = spl.lambda();
        for _ in 0..50 {
            spl.step();
        }
        assert!(
            spl.lambda() > initial_lambda,
            "lambda should increase as SPL steps advance"
        );
    }

    #[test]
    fn test_spl_lambda_reaches_max() {
        let mut spl = SelfPacedLearning::new(0.1, 0.9, 10);
        for _ in 0..20 {
            spl.step();
        }
        // After more steps than total_steps, lambda should be at or near max
        assert!(
            spl.lambda() >= 0.9 - 1e-9,
            "lambda should reach max value after total_steps"
        );
    }

    #[test]
    fn test_spl_anti_curriculum_schedule() {
        // Anti-curriculum: start with high lambda (include hard examples first)
        let spl = SelfPacedLearning::new(0.9, 0.1, 100);
        assert!(
            spl.include_example(0.7, 0.8),
            "anti-curriculum SPL should include hard examples at start"
        );
    }
}
