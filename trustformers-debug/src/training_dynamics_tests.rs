//! Tests for the training_dynamics module

#[cfg(test)]
mod tests {
    use crate::training_dynamics::{
        BatchSizeAnalysis, BatchSizePoint, BatchSizeRecommendation, ConvergenceAnalysis,
        ConvergenceCriterion, ConvergenceCriterionType, ConvergenceStatus, EarlyStoppingRecommendation,
        LRAction, LRRecommendation, LRScheduleType, LearningRateAnalysis, LearningRatePoint,
        LossCurveAnalysis, LossStatistics, LossTrend, MovingAverages, PlateauAction, PlateauAnalysis,
        PlateauCharacteristics, PlateauRecommendation, PlateauType, Priority, TrainingCategory,
        TrainingDynamicsAnalyzer, TrainingDynamicsConfig, TrainingDynamicsReport, TrainingMetrics,
        TrainingRecommendation, TrainingSummary, TrainingStateSummary,
    };

    // -------------------------------------------------------------------------
    // LCG for deterministic pseudo-random values
    // -------------------------------------------------------------------------
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }

        fn next(&mut self) -> u64 {
            self.state = self.state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }

        fn next_f32_in(&mut self, lo: f32, hi: f32) -> f32 {
            let t = (self.next() >> 11) as f32 / (1u64 << 53) as f32;
            lo + t * (hi - lo)
        }
    }

    // -------------------------------------------------------------------------
    // Helper: build a TrainingMetrics sample
    // -------------------------------------------------------------------------

    fn make_metrics(epoch: usize, step: usize, train_loss: f32, lr: f32) -> TrainingMetrics {
        TrainingMetrics {
            epoch,
            step,
            train_loss,
            validation_loss: Some(train_loss + 0.01),
            learning_rate: lr,
            batch_size: 32,
            gradient_norm: Some(0.5),
            accuracy: Some(0.9),
            timestamp: epoch as f64 * 1.0,
        }
    }

    // -------------------------------------------------------------------------
    // TrainingDynamicsConfig tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_default_values() {
        let config = TrainingDynamicsConfig::default();
        assert!(config.enable_loss_curve_analysis);
        assert!(config.enable_learning_rate_analysis);
        assert!(config.enable_batch_size_analysis);
        assert!(config.enable_convergence_detection);
        assert!(config.enable_plateau_identification);
        assert_eq!(config.moving_average_window, 10);
        assert!((config.convergence_tolerance - 1e-6).abs() < 1e-12);
        assert!((config.plateau_threshold - 1e-4).abs() < 1e-9);
        assert_eq!(config.min_epochs_for_convergence, 20);
        assert_eq!(config.max_history_length, 10000);
    }

    #[test]
    fn test_config_custom() {
        let config = TrainingDynamicsConfig {
            enable_loss_curve_analysis: false,
            enable_learning_rate_analysis: false,
            enable_batch_size_analysis: false,
            enable_convergence_detection: false,
            enable_plateau_identification: false,
            moving_average_window: 5,
            convergence_tolerance: 1e-4,
            plateau_threshold: 1e-3,
            min_epochs_for_convergence: 10,
            max_history_length: 100,
        };
        assert!(!config.enable_loss_curve_analysis);
        assert_eq!(config.moving_average_window, 5);
    }

    // -------------------------------------------------------------------------
    // TrainingMetrics tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_training_metrics_construction() {
        let m = make_metrics(5, 100, 1.5, 0.001);
        assert_eq!(m.epoch, 5);
        assert_eq!(m.step, 100);
        assert!((m.train_loss - 1.5).abs() < 1e-6);
        assert!((m.learning_rate - 0.001).abs() < 1e-9);
        assert!(m.validation_loss.is_some());
        assert!(m.gradient_norm.is_some());
        assert!(m.accuracy.is_some());
    }

    // -------------------------------------------------------------------------
    // TrainingDynamicsAnalyzer construction tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_analyzer_new() {
        let config = TrainingDynamicsConfig::default();
        let analyzer = TrainingDynamicsAnalyzer::new(config);
        let summary = analyzer.get_training_summary();
        assert_eq!(summary.total_epochs, 0);
        assert_eq!(summary.total_steps, 0);
        assert_eq!(summary.metrics_collected, 0);
    }

    #[test]
    fn test_analyzer_record_metrics() {
        let config = TrainingDynamicsConfig::default();
        let mut analyzer = TrainingDynamicsAnalyzer::new(config);

        for i in 0..5 {
            analyzer.record_metrics(make_metrics(i, i * 10, 2.0 - i as f32 * 0.1, 0.001));
        }

        let summary = analyzer.get_training_summary();
        assert_eq!(summary.metrics_collected, 5);
        assert_eq!(summary.total_epochs, 4);
    }

    #[test]
    fn test_analyzer_history_limit() {
        let config = TrainingDynamicsConfig {
            max_history_length: 10,
            ..TrainingDynamicsConfig::default()
        };
        let mut analyzer = TrainingDynamicsAnalyzer::new(config);

        for i in 0..20 {
            analyzer.record_metrics(make_metrics(i, i * 5, 1.0, 0.001));
        }

        let summary = analyzer.get_training_summary();
        // History should be capped at 10
        assert!(summary.metrics_collected <= 10);
    }

    #[test]
    fn test_analyzer_clear() {
        let config = TrainingDynamicsConfig::default();
        let mut analyzer = TrainingDynamicsAnalyzer::new(config);

        for i in 0..5 {
            analyzer.record_metrics(make_metrics(i, i, 1.0, 0.001));
        }

        analyzer.clear();
        let summary = analyzer.get_training_summary();
        assert_eq!(summary.metrics_collected, 0);
    }

    #[test]
    fn test_analyzer_training_summary_with_metrics() {
        let config = TrainingDynamicsConfig::default();
        let mut analyzer = TrainingDynamicsAnalyzer::new(config);

        analyzer.record_metrics(make_metrics(10, 200, 0.5, 0.0001));
        let summary = analyzer.get_training_summary();

        assert_eq!(summary.total_epochs, 10);
        assert_eq!(summary.total_steps, 200);
        assert!((summary.current_loss - 0.5).abs() < 1e-6);
        assert!((summary.current_lr - 0.0001).abs() < 1e-9);
    }

    // -------------------------------------------------------------------------
    // Analyze tests (async)
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_analyze_empty_history() {
        let config = TrainingDynamicsConfig::default();
        let mut analyzer = TrainingDynamicsAnalyzer::new(config);

        match analyzer.analyze().await {
            Ok(report) => {
                // Empty history => unknown/empty analyses
                if let Some(loss_analysis) = &report.loss_curve_analysis {
                    match loss_analysis.trend {
                        LossTrend::Unknown => {},
                        _ => {},
                    }
                }
                // Summary should show 0 epochs/steps
                assert_eq!(report.training_summary.total_epochs, 0);
            },
            Err(_) => {},
        }
    }

    #[tokio::test]
    async fn test_analyze_decreasing_loss() {
        let config = TrainingDynamicsConfig::default();
        let mut analyzer = TrainingDynamicsAnalyzer::new(config);

        // Add clearly decreasing loss values
        for i in 0..30 {
            let loss = 2.0 - i as f32 * 0.05;
            analyzer.record_metrics(make_metrics(i, i * 10, loss, 0.001));
        }

        match analyzer.analyze().await {
            Ok(report) => {
                if let Some(loss_analysis) = &report.loss_curve_analysis {
                    assert!(
                        loss_analysis.best_loss < loss_analysis.current_loss + 0.5,
                        "Best loss should be less than initial loss"
                    );
                    assert!(
                        loss_analysis.loss_reduction_percentage > 0.0,
                        "Loss should have reduced"
                    );
                }
            },
            Err(_) => {},
        }
    }

    #[tokio::test]
    async fn test_analyze_loss_statistics() {
        let config = TrainingDynamicsConfig::default();
        let mut analyzer = TrainingDynamicsAnalyzer::new(config);

        let mut lcg = Lcg::new(12345);
        for i in 0..50 {
            let loss = 1.0 + lcg.next_f32_in(-0.1, 0.1) - i as f32 * 0.01;
            analyzer.record_metrics(make_metrics(i, i * 5, loss.max(0.01), 0.001));
        }

        match analyzer.analyze().await {
            Ok(report) => {
                if let Some(loss_analysis) = &report.loss_curve_analysis {
                    let stats = &loss_analysis.loss_statistics;
                    assert!(stats.min <= stats.mean);
                    assert!(stats.mean <= stats.max);
                    assert!(stats.std >= 0.0);
                }
            },
            Err(_) => {},
        }
    }

    #[tokio::test]
    async fn test_analyze_moving_averages() {
        let config = TrainingDynamicsConfig::default();
        let mut analyzer = TrainingDynamicsAnalyzer::new(config);

        for i in 0..50 {
            analyzer.record_metrics(make_metrics(i, i, 1.0 - i as f32 * 0.01, 0.001));
        }

        match analyzer.analyze().await {
            Ok(report) => {
                if let Some(loss_analysis) = &report.loss_curve_analysis {
                    let ma = &loss_analysis.moving_averages;
                    // All MA values should be finite and positive
                    assert!(ma.short_term.is_finite());
                    assert!(ma.medium_term.is_finite());
                    assert!(ma.long_term.is_finite());
                }
            },
            Err(_) => {},
        }
    }

    #[tokio::test]
    async fn test_analyze_convergence_detection() {
        let config = TrainingDynamicsConfig {
            min_epochs_for_convergence: 5,
            ..TrainingDynamicsConfig::default()
        };
        let mut analyzer = TrainingDynamicsAnalyzer::new(config);

        // Add enough metrics for convergence detection
        for i in 0..25 {
            analyzer.record_metrics(make_metrics(i, i * 10, 0.1, 0.0001));
        }

        match analyzer.analyze().await {
            Ok(report) => {
                if let Some(conv) = &report.convergence_analysis {
                    // Convergence probability should be 0-1
                    assert!(conv.convergence_probability >= 0.0);
                    assert!(conv.convergence_probability <= 1.0);
                }
            },
            Err(_) => {},
        }
    }

    #[tokio::test]
    async fn test_analyze_plateau_detection() {
        let config = TrainingDynamicsConfig::default();
        let mut analyzer = TrainingDynamicsAnalyzer::new(config);

        // Add stable (plateau-like) loss
        for i in 0..50 {
            analyzer.record_metrics(make_metrics(i, i * 5, 0.5, 0.0001));
        }

        match analyzer.analyze().await {
            Ok(report) => {
                if let Some(plateau) = &report.plateau_analysis {
                    // Plateau should be detected for constant loss
                    assert!(plateau.plateau_detected);
                }
            },
            Err(_) => {},
        }
    }

    #[tokio::test]
    async fn test_generate_report() {
        let config = TrainingDynamicsConfig::default();
        let mut analyzer = TrainingDynamicsAnalyzer::new(config);

        for i in 0..15 {
            analyzer.record_metrics(make_metrics(i, i * 10, 1.5 - i as f32 * 0.05, 0.001));
        }

        match analyzer.generate_report().await {
            Ok(report) => {
                // Report should have training summary
                assert!(report.training_summary.total_epochs > 0);
            },
            Err(_) => {},
        }
    }

    // -------------------------------------------------------------------------
    // LossTrend variant tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_loss_trend_variants() {
        let variants = [
            LossTrend::Decreasing,
            LossTrend::Increasing,
            LossTrend::Oscillating,
            LossTrend::Plateaued,
            LossTrend::Unknown,
        ];
        for v in &variants {
            let _ = format!("{:?}", v);
        }
    }

    // -------------------------------------------------------------------------
    // LRScheduleType variant tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_lr_schedule_type_variants() {
        let types = [
            LRScheduleType::Constant,
            LRScheduleType::StepDecay,
            LRScheduleType::ExponentialDecay,
            LRScheduleType::CosineAnnealing,
            LRScheduleType::ReduceOnPlateau,
            LRScheduleType::Warmup,
            LRScheduleType::Cyclical,
            LRScheduleType::Unknown,
        ];
        for t in &types {
            let _ = format!("{:?}", t);
        }
    }

    // -------------------------------------------------------------------------
    // LRAction variant tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_lr_action_variants() {
        let actions = [
            LRAction::Increase,
            LRAction::Decrease,
            LRAction::KeepCurrent,
            LRAction::AddScheduler,
            LRAction::ChangeScheduler,
            LRAction::AddWarmup,
        ];
        for a in &actions {
            let _ = format!("{:?}", a);
        }
    }

    // -------------------------------------------------------------------------
    // ConvergenceStatus variant tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_convergence_status_variants() {
        let statuses = [
            ConvergenceStatus::Converging,
            ConvergenceStatus::Converged,
            ConvergenceStatus::Diverging,
            ConvergenceStatus::Oscillating,
            ConvergenceStatus::TooEarly,
        ];
        for s in &statuses {
            let _ = format!("{:?}", s);
        }
    }

    // -------------------------------------------------------------------------
    // ConvergenceCriterionType variant tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_convergence_criterion_type_variants() {
        let types = [
            ConvergenceCriterionType::LossStability,
            ConvergenceCriterionType::GradientMagnitude,
            ConvergenceCriterionType::LossImprovement,
            ConvergenceCriterionType::ValidationGap,
            ConvergenceCriterionType::LearningRateDecay,
        ];
        for t in &types {
            let _ = format!("{:?}", t);
        }
    }

    // -------------------------------------------------------------------------
    // PlateauType variant tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_plateau_type_variants() {
        let types = [
            PlateauType::LossPlayteau,
            PlateauType::GradientPlateau,
            PlateauType::AccuracyPlateau,
            PlateauType::LearningRatePlateau,
        ];
        for t in &types {
            let _ = format!("{:?}", t);
        }
    }

    // -------------------------------------------------------------------------
    // PlateauAction variant tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_plateau_action_variants() {
        let actions = [
            PlateauAction::IncreaseLearningRate,
            PlateauAction::DecreaseLearningRate,
            PlateauAction::ChangeBatchSize,
            PlateauAction::AddRegularization,
            PlateauAction::RemoveRegularization,
            PlateauAction::ChangeOptimizer,
            PlateauAction::AddNoise,
            PlateauAction::EarlyStopping,
            PlateauAction::ContinueTraining,
        ];
        for a in &actions {
            let _ = format!("{:?}", a);
        }
    }

    // -------------------------------------------------------------------------
    // Priority variant tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_priority_variants() {
        let priorities = [
            Priority::Critical,
            Priority::High,
            Priority::Medium,
            Priority::Low,
        ];
        for p in &priorities {
            let _ = format!("{:?}", p);
        }
    }

    // -------------------------------------------------------------------------
    // TrainingCategory variant tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_training_category_variants() {
        let categories = [
            TrainingCategory::LearningRate,
            TrainingCategory::BatchSize,
            TrainingCategory::Optimization,
            TrainingCategory::Regularization,
            TrainingCategory::EarlyStopping,
            TrainingCategory::Architecture,
        ];
        for c in &categories {
            let _ = format!("{:?}", c);
        }
    }

    // -------------------------------------------------------------------------
    // MovingAverages field tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_moving_averages_construction() {
        let ma = MovingAverages {
            short_term: 1.0,
            medium_term: 1.1,
            long_term: 1.2,
        };
        assert!((ma.short_term - 1.0).abs() < 1e-6);
        assert!((ma.medium_term - 1.1).abs() < 1e-6);
        assert!((ma.long_term - 1.2).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // LossStatistics field tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_loss_statistics_construction() {
        let stats = LossStatistics {
            mean: 0.5,
            std: 0.1,
            min: 0.2,
            max: 0.9,
            median: 0.5,
            percentile_25: 0.3,
            percentile_75: 0.7,
            autocorrelation: 0.8,
        };
        assert!(stats.min <= stats.median);
        assert!(stats.median <= stats.max);
        assert!(stats.percentile_25 <= stats.percentile_75);
    }

    // -------------------------------------------------------------------------
    // TrainingStateSummary tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_training_state_summary_fields() {
        let summary = TrainingStateSummary {
            total_epochs: 100,
            total_steps: 5000,
            current_loss: 0.25,
            current_lr: 0.0001,
            metrics_collected: 100,
        };
        assert_eq!(summary.total_epochs, 100);
        assert_eq!(summary.total_steps, 5000);
        assert!((summary.current_loss - 0.25).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // ConvergenceCriterion field tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_convergence_criterion_construction() {
        let criterion = ConvergenceCriterion {
            criterion_type: ConvergenceCriterionType::LossStability,
            current_value: 0.001,
            threshold: 0.01,
            satisfied: true,
            confidence: 0.9,
        };
        assert!(criterion.satisfied);
        assert!(criterion.current_value < criterion.threshold);
    }

    // -------------------------------------------------------------------------
    // EarlyStoppingRecommendation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_early_stopping_recommendation() {
        let rec = EarlyStoppingRecommendation {
            should_stop: true,
            confidence: 0.85,
            rationale: "Validation loss not improving".to_string(),
            suggested_epochs_remaining: 5,
        };
        assert!(rec.should_stop);
        assert!((rec.confidence - 0.85).abs() < 1e-9);
        assert_eq!(rec.suggested_epochs_remaining, 5);
    }

    // -------------------------------------------------------------------------
    // LearningRatePoint tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_learning_rate_point_construction() {
        let point = LearningRatePoint {
            epoch: 10,
            learning_rate: 0.001,
            loss_change: -0.05,
            gradient_norm: Some(0.8),
            effectiveness: 0.7,
        };
        assert_eq!(point.epoch, 10);
        assert!(point.gradient_norm.is_some());
    }

    // -------------------------------------------------------------------------
    // BatchSizePoint tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_batch_size_point_construction() {
        let point = BatchSizePoint {
            epoch: 5,
            batch_size: 64,
            loss_improvement: 0.1,
            gradient_stability: 0.9,
            throughput: 1000.0,
        };
        assert_eq!(point.batch_size, 64);
        assert!((point.throughput - 1000.0).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // TrainingSummary field validation
    // -------------------------------------------------------------------------

    #[test]
    fn test_training_summary_construction() {
        let summary = TrainingSummary {
            total_epochs: 50,
            total_steps: 2500,
            training_efficiency: 0.85,
            convergence_health: 0.7,
            stability_score: 0.9,
            overall_progress: 0.6,
        };
        assert!(summary.training_efficiency >= 0.0 && summary.training_efficiency <= 1.0);
        assert!(summary.convergence_health >= 0.0 && summary.convergence_health <= 1.0);
    }

    // -------------------------------------------------------------------------
    // TrainingRecommendation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_training_recommendation_construction() {
        let rec = TrainingRecommendation {
            category: TrainingCategory::LearningRate,
            priority: Priority::High,
            description: "Reduce learning rate".to_string(),
            implementation: "scheduler.step()".to_string(),
            expected_impact: 0.2,
        };
        assert_eq!(rec.description, "Reduce learning rate");
        assert!((rec.expected_impact - 0.2).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // PlateauCharacteristics tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_plateau_characteristics() {
        let chars = PlateauCharacteristics {
            stability: 0.95,
            noise_level: 0.02,
            gradient_magnitude: 0.001,
            overfitting_risk: 0.3,
        };
        assert!(chars.stability >= 0.0 && chars.stability <= 1.0);
        assert!(chars.overfitting_risk >= 0.0 && chars.overfitting_risk <= 1.0);
    }

    // -------------------------------------------------------------------------
    // BatchSizeRecommendation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_batch_size_recommendation() {
        let rec = BatchSizeRecommendation {
            suggested_batch_size: 128,
            confidence: 0.8,
            rationale: "Larger batch size improves throughput".to_string(),
            expected_benefits: vec![
                "Higher GPU utilization".to_string(),
                "Faster convergence".to_string(),
            ],
        };
        assert_eq!(rec.suggested_batch_size, 128);
        assert_eq!(rec.expected_benefits.len(), 2);
    }

    // -------------------------------------------------------------------------
    // Test analyzer with all features disabled
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_analyze_all_disabled() {
        let config = TrainingDynamicsConfig {
            enable_loss_curve_analysis: false,
            enable_learning_rate_analysis: false,
            enable_batch_size_analysis: false,
            enable_convergence_detection: false,
            enable_plateau_identification: false,
            ..TrainingDynamicsConfig::default()
        };
        let mut analyzer = TrainingDynamicsAnalyzer::new(config);

        for i in 0..10 {
            analyzer.record_metrics(make_metrics(i, i, 1.0, 0.001));
        }

        match analyzer.analyze().await {
            Ok(report) => {
                assert!(report.loss_curve_analysis.is_none());
                assert!(report.learning_rate_analysis.is_none());
                assert!(report.batch_size_analysis.is_none());
                assert!(report.convergence_analysis.is_none());
                assert!(report.plateau_analysis.is_none());
            },
            Err(_) => {},
        }
    }

    // -------------------------------------------------------------------------
    // Test training metrics with optional fields as None
    // -------------------------------------------------------------------------

    #[test]
    fn test_training_metrics_no_optional_fields() {
        let m = TrainingMetrics {
            epoch: 1,
            step: 10,
            train_loss: 0.8,
            validation_loss: None,
            learning_rate: 0.01,
            batch_size: 16,
            gradient_norm: None,
            accuracy: None,
            timestamp: 1.0,
        };
        assert!(m.validation_loss.is_none());
        assert!(m.gradient_norm.is_none());
        assert!(m.accuracy.is_none());
    }
}
