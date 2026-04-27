//! Hyperparameter optimisation — new multi-objective and auto-LR modules.
//!
//! This `hpo` module complements the existing [`crate::hyperopt`] module with
//! additional algorithms:
//!
//! - [`multi_objective`]: NSGA-II-inspired Pareto front search.
//! - [`auto_lr`]: Automatic learning-rate selection via heuristics and LR-range-test analysis.

pub mod auto_lr;
pub mod multi_objective;

pub use auto_lr::{
    AutoLrConfig, AutoLrModelType, AutoLrResult, AutoLrSelector, AutoLrStrategy,
    LrRangeTest, RecommendedSchedule, TrainingLrConfig,
};
pub use multi_objective::{
    compute_pareto_front, hypervolume_indicator, non_domination_sort,
    HpConfig, HpSearchSpace, HpValue, MultiObjectiveHpo, MultiObjectiveHpoConfig,
    MultiObjectiveResult, ObjectiveDirection, ParetoFront,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ── helper: build a minimal MultiObjectiveHpoConfig ──────────────────────

    fn two_obj_config() -> MultiObjectiveHpoConfig {
        let mut search_space = HashMap::new();
        search_space.insert(
            "lr".to_string(),
            HpSearchSpace::Float { min: 1e-5, max: 1e-2, log_scale: true },
        );
        search_space.insert(
            "batch".to_string(),
            HpSearchSpace::Int { min: 16, max: 128 },
        );
        MultiObjectiveHpoConfig {
            search_space,
            objectives: vec![
                ("accuracy".to_string(), ObjectiveDirection::Maximize),
                ("latency_ms".to_string(), ObjectiveDirection::Minimize),
            ],
            n_trials: 10,
            seed: 42,
            use_pareto_front: true,
        }
    }

    fn make_result(trial_id: usize, obj: Vec<f64>) -> MultiObjectiveResult {
        MultiObjectiveResult {
            config: HpConfig { params: HashMap::new(), trial_id },
            objectives: obj,
            metadata: HashMap::new(),
        }
    }

    // ── LrRangeTest: step 0 returns min_lr ─────────────────────────────────

    #[test]
    fn test_lr_range_test_step_zero_returns_min_lr() {
        let t = LrRangeTest::new(1e-6, 1e-1, 50);
        let lr = t.lr_at_step(0);
        assert!((lr - 1e-6_f32).abs() < 1e-12_f32, "expected min_lr at step 0, got {lr}");
    }

    // ── LrRangeTest: last step returns max_lr ──────────────────────────────

    #[test]
    fn test_lr_range_test_last_step_returns_max_lr() {
        let t = LrRangeTest::new(1e-6, 1e-1, 50);
        let lr = t.lr_at_step(49);
        assert!((lr - 1e-1_f32).abs() < 1e-6_f32, "expected max_lr at last step, got {lr}");
    }

    // ── LrRangeTest: midpoint is geometric mean ────────────────────────────

    #[test]
    fn test_lr_range_test_midpoint_is_geometric_mean() {
        let min_lr = 1e-6_f32;
        let max_lr = 1e-2_f32;
        let num_iters = 101; // odd → midpoint is index 50
        let t = LrRangeTest::new(min_lr, max_lr, num_iters);
        let mid_lr = t.lr_at_step(50);
        let expected = (min_lr * max_lr).sqrt();
        let rel_err = (mid_lr - expected).abs() / expected;
        assert!(rel_err < 1e-4_f32, "midpoint lr={mid_lr} expected≈{expected}, rel_err={rel_err}");
    }

    // ── smooth_losses: smoothing=0.0 reproduces input exactly ─────────────

    #[test]
    fn test_smooth_losses_zero_smoothing_passthrough() {
        let t = LrRangeTest { min_lr: 1e-5, max_lr: 1e-1, num_iters: 10, smoothing: 0.0 };
        let losses = vec![1.0_f32, 2.0, 3.0, 4.0];
        let smoothed = t.smooth_losses(&losses);
        assert_eq!(smoothed.len(), losses.len());
        // With beta=0 each smoothed[i] = loss[i] (bias-corrected ema = loss directly).
        for (s, l) in smoothed.iter().zip(losses.iter()) {
            assert!((s - l).abs() < 1e-5_f32, "expected passthrough, got {s} vs {l}");
        }
    }

    // ── smooth_losses: constant input converges to that constant ───────────

    #[test]
    fn test_smooth_losses_constant_converges() {
        let t = LrRangeTest { min_lr: 1e-5, max_lr: 1e-1, num_iters: 200, smoothing: 0.9 };
        let losses = vec![3.0_f32; 200];
        let smoothed = t.smooth_losses(&losses);
        let last = *smoothed.last().expect("non-empty");
        assert!((last - 3.0_f32).abs() < 0.01_f32, "expected ~3.0, got {last}");
    }

    // ── find_optimal_lr: None for fewer than 4 points ─────────────────────

    #[test]
    fn test_find_optimal_lr_returns_none_for_short_curve() {
        let short = vec![(1e-5_f32, 3.0_f32), (1e-4, 2.8), (1e-3, 2.5)];
        assert!(LrRangeTest::find_optimal_lr(&short).is_none());
    }

    // ── find_optimal_lr: returns Some for valid curve ─────────────────────

    #[test]
    fn test_find_optimal_lr_valid_descending_then_ascending() {
        // loss drops for first 15 steps then rises → steepest descent in first half
        let curve: Vec<(f32, f32)> = (0..30)
            .map(|i| {
                let lr = 1e-6_f32 * 10f32.powf(i as f32 * 0.15);
                let loss = if i < 15 { 3.0 - 0.1 * i as f32 } else { 1.5 + (i - 15) as f32 * 0.5 };
                (lr, loss)
            })
            .collect();
        let opt = LrRangeTest::find_optimal_lr(&curve);
        assert!(opt.is_some(), "should return Some");
        assert!(opt.expect("some") > 0.0, "optimal LR must be positive");
    }

    // ── compute_pareto_front: dominated point is excluded ─────────────────

    #[test]
    fn test_compute_pareto_front_dominates() {
        // Point [0.1, 0.2] dominates [0.5, 0.8] under minimisation.
        let points = vec![
            (vec![], vec![0.5_f32, 0.8_f32]),  // dominated
            (vec![], vec![0.1_f32, 0.2_f32]),  // Pareto-optimal
        ];
        let front = compute_pareto_front(&points);
        // Only index 1 should be on the front.
        assert_eq!(front.len(), 1, "expected 1 non-dominated point");
        assert!(front.contains(&1), "index 1 should be on the front");
    }

    // ── compute_pareto_front: incomparable points both on front ──────────

    #[test]
    fn test_compute_pareto_front_incomparable_both_survive() {
        // [0.1, 0.9] and [0.9, 0.1] are incomparable under minimisation.
        let points = vec![
            (vec![], vec![0.1_f32, 0.9_f32]),
            (vec![], vec![0.9_f32, 0.1_f32]),
        ];
        let front = compute_pareto_front(&points);
        assert_eq!(front.len(), 2, "both incomparable points should be on front");
    }

    // ── compute_pareto_front: single point is always on front ────────────

    #[test]
    fn test_compute_pareto_front_single_point() {
        let points = vec![(vec![], vec![0.5_f32, 0.5_f32])];
        let front = compute_pareto_front(&points);
        assert_eq!(front, vec![0_usize]);
    }

    // ── compute_pareto_front: empty input returns empty ───────────────────

    #[test]
    fn test_compute_pareto_front_empty() {
        let front = compute_pareto_front(&[]);
        assert!(front.is_empty());
    }

    // ── hypervolume_indicator: empty front = 0.0 ─────────────────────────

    #[test]
    fn test_hypervolume_indicator_empty_front() {
        let hv = hypervolume_indicator(&[], &[1.0_f32, 1.0_f32]);
        assert_eq!(hv, 0.0_f32);
    }

    // ── hypervolume_indicator: single point = area of box ────────────────

    #[test]
    fn test_hypervolume_indicator_single_point_2d() {
        // Point [0.0, 0.0] with reference [1.0, 1.0] → area = 1.0
        let front = vec![vec![0.0_f32, 0.0_f32]];
        let reference = vec![1.0_f32, 1.0_f32];
        let hv = hypervolume_indicator(&front, &reference);
        assert!((hv - 1.0_f32).abs() < 0.01_f32, "expected hypervolume ~1.0, got {hv}");
    }

    // ── hypervolume_indicator: positive area for valid 2D front ──────────

    #[test]
    fn test_hypervolume_indicator_two_points_2d() {
        let front = vec![
            vec![0.2_f32, 0.8_f32],
            vec![0.8_f32, 0.2_f32],
        ];
        let reference = vec![1.0_f32, 1.0_f32];
        let hv = hypervolume_indicator(&front, &reference);
        assert!(hv > 0.0_f32, "hypervolume should be positive, got {hv}");
        assert!(hv <= 1.0_f32, "hypervolume cannot exceed bounding box area");
    }

    // ── non_domination_sort: rank-1 contains all Pareto-optimal ──────────

    #[test]
    fn test_non_domination_sort_first_rank_is_pareto_optimal() {
        // Three incomparable objectives → all should appear first.
        let objectives = vec![
            vec![0.1_f32, 0.9_f32],
            vec![0.5_f32, 0.5_f32],
            vec![0.9_f32, 0.1_f32],
        ];
        let sorted = non_domination_sort(&objectives);
        assert_eq!(sorted.len(), 3, "all 3 points should be in result");
    }

    // ── non_domination_sort: dominated point comes after dominator ────────

    #[test]
    fn test_non_domination_sort_dominated_after_dominator() {
        let objectives = vec![
            vec![0.1_f32, 0.1_f32],  // index 0: dominates everything
            vec![0.9_f32, 0.9_f32],  // index 1: dominated
        ];
        let sorted = non_domination_sort(&objectives);
        assert_eq!(sorted.len(), 2);
        // Index 0 should appear first (rank 1).
        assert_eq!(sorted[0], 0, "dominator should be first");
        assert_eq!(sorted[1], 1, "dominated should be second");
    }

    // ── non_domination_sort: empty input returns empty ────────────────────

    #[test]
    fn test_non_domination_sort_empty() {
        let sorted = non_domination_sort(&[]);
        assert!(sorted.is_empty());
    }

    // ── MultiObjectiveHpo: new() succeeds with valid config ───────────────

    #[test]
    fn test_multi_objective_hpo_new_valid() {
        let config = two_obj_config();
        let hpo = MultiObjectiveHpo::new(config);
        assert!(hpo.is_ok(), "new() should succeed with valid config");
    }

    // ── MultiObjectiveHpo: new() errors with no objectives ────────────────

    #[test]
    fn test_multi_objective_hpo_new_no_objectives_errors() {
        let mut config = two_obj_config();
        config.objectives.clear();
        assert!(MultiObjectiveHpo::new(config).is_err());
    }

    // ── MultiObjectiveHpo: new() errors with no search space ─────────────

    #[test]
    fn test_multi_objective_hpo_new_no_search_space_errors() {
        let mut config = two_obj_config();
        config.search_space.clear();
        assert!(MultiObjectiveHpo::new(config).is_err());
    }

    // ── MultiObjectiveHpo: new() errors with n_trials=0 ──────────────────

    #[test]
    fn test_multi_objective_hpo_new_zero_trials_errors() {
        let mut config = two_obj_config();
        config.n_trials = 0;
        assert!(MultiObjectiveHpo::new(config).is_err());
    }

    // ── MultiObjectiveHpo: suggest() returns config with expected params ──

    #[test]
    fn test_multi_objective_hpo_suggest_has_params() {
        let config = two_obj_config();
        let mut hpo = MultiObjectiveHpo::new(config).expect("valid config");
        let hp_cfg = hpo.suggest();
        assert!(hp_cfg.params.contains_key("lr"), "should contain 'lr' param");
        assert!(hp_cfg.params.contains_key("batch"), "should contain 'batch' param");
    }

    // ── MultiObjectiveHpo: trial_id increments on each suggest ───────────

    #[test]
    fn test_multi_objective_hpo_suggest_trial_id_increments() {
        let config = two_obj_config();
        let mut hpo = MultiObjectiveHpo::new(config).expect("valid config");
        let cfg0 = hpo.suggest();
        let cfg1 = hpo.suggest();
        assert_eq!(cfg0.trial_id, 0);
        assert_eq!(cfg1.trial_id, 1);
    }

    // ── MultiObjectiveHpo: record() stores result ─────────────────────────

    #[test]
    fn test_multi_objective_hpo_record_stores_result() {
        let config = two_obj_config();
        let mut hpo = MultiObjectiveHpo::new(config).expect("valid config");
        let hp_cfg = hpo.suggest();
        let result = MultiObjectiveResult {
            config: hp_cfg,
            objectives: vec![0.85, 120.0],
            metadata: HashMap::new(),
        };
        hpo.record(result);
        assert_eq!(hpo.results().len(), 1);
    }

    // ── MultiObjectiveHpo: pareto_front updates after record ─────────────

    #[test]
    fn test_multi_objective_hpo_pareto_front_updates() {
        let config = two_obj_config();
        let mut hpo = MultiObjectiveHpo::new(config).expect("valid config");
        for i in 0..5 {
            let hp_cfg = hpo.suggest();
            let result = MultiObjectiveResult {
                config: hp_cfg,
                objectives: vec![0.7 + i as f64 * 0.02, 100.0 - i as f64 * 5.0],
                metadata: HashMap::new(),
            };
            hpo.record(result);
        }
        // Pareto front should be non-empty.
        assert!(!hpo.pareto_front().is_empty(), "Pareto front should be populated");
    }

    // ── ParetoFront: empty front has 0 solutions ──────────────────────────

    #[test]
    fn test_pareto_front_empty() {
        let front = ParetoFront::new();
        assert!(front.is_empty());
        assert_eq!(front.len(), 0);
    }

    // ── AutoLrConfig: default has Bert model and use_warmup=true ─────────

    #[test]
    fn test_auto_lr_config_default() {
        let cfg = AutoLrConfig::default();
        assert!(cfg.use_warmup);
        assert_eq!(cfg.model_type, AutoLrModelType::Bert);
        assert_eq!(cfg.strategy, AutoLrStrategy::Heuristic);
    }

    // ── AutoLrModelType::Custom geometric mean sweet_spot ─────────────────

    #[test]
    fn test_custom_model_type_sweet_spot_geometric_mean() {
        let model_type = AutoLrModelType::Custom { suggested_range: (1e-4, 1e-2) };
        let sweet = model_type.sweet_spot();
        let expected = (1e-4_f64 * 1e-2_f64).sqrt();
        assert!((sweet - expected).abs() < 1e-10, "expected geometric mean {expected}, got {sweet}");
    }

    // ── AutoLrStrategy variants exist ────────────────────────────────────

    #[test]
    fn test_auto_lr_strategy_variants() {
        let _h = AutoLrStrategy::Heuristic;
        let _r = AutoLrStrategy::LrRangeTest;
        let _c = AutoLrStrategy::Combined;
    }

    // ── RecommendedSchedule variants exist ────────────────────────────────

    #[test]
    fn test_recommended_schedule_variants() {
        let _l = RecommendedSchedule::Linear;
        let _c = RecommendedSchedule::Cosine;
        let _cr = RecommendedSchedule::CosineWithRestarts;
        let _oc = RecommendedSchedule::OneCycle;
    }

    // ── AutoLrSelector: heuristic for Bert in expected range ─────────────

    #[test]
    fn test_auto_lr_selector_bert_range() {
        let selector = AutoLrSelector::new(AutoLrConfig {
            model_type: AutoLrModelType::Bert,
            use_warmup: true,
            warmup_fraction: 0.06,
            strategy: AutoLrStrategy::Heuristic,
        });
        let result = selector.select_from_heuristics(110_000_000);
        let (lo, hi) = AutoLrModelType::Bert.suggested_range();
        assert!(result.suggested_lr >= lo * 0.4 && result.suggested_lr <= hi * 2.5,
            "Bert LR {:.2e} out of expected range [{:.2e}, {:.2e}]",
            result.suggested_lr, lo * 0.4, hi * 2.5);
    }

    // ── AutoLrSelector: recommend_training_config returns valid config ────

    #[test]
    fn test_auto_lr_selector_recommend_training_config() {
        let selector = AutoLrSelector::new(AutoLrConfig::default());
        let cfg = selector.recommend_training_config(100_000_000, 10_000);
        assert!(cfg.initial_lr > 0.0, "initial_lr must be positive");
        assert!(cfg.max_lr >= cfg.min_lr, "max_lr must be >= min_lr");
        assert!(cfg.warmup_steps > 0, "warmup_steps should be > 0 with use_warmup=true");
        assert!(cfg.weight_decay >= 0.0, "weight_decay must be non-negative");
    }
}
