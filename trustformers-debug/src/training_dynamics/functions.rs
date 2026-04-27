//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use super::super::*;
    fn make_metrics(epoch: usize, step: usize, loss: f32, lr: f32) -> TrainingMetrics {
        TrainingMetrics {
            epoch,
            step,
            train_loss: loss,
            validation_loss: None,
            learning_rate: lr,
            batch_size: 32,
            gradient_norm: Some(1.0),
            accuracy: Some(0.8),
            timestamp: epoch as f64,
        }
    }
    fn make_analyzer() -> TrainingDynamicsAnalyzer {
        TrainingDynamicsAnalyzer::new(TrainingDynamicsConfig::default())
    }
    fn make_analyzer_with_decreasing_loss(n: usize) -> TrainingDynamicsAnalyzer {
        let mut analyzer = make_analyzer();
        for i in 0..n {
            let loss = 2.0 - (i as f32 * 0.05);
            analyzer.record_metrics(make_metrics(i, i * 100, loss, 0.001));
        }
        analyzer
    }
    #[test]
    fn test_config_default() {
        let config = TrainingDynamicsConfig::default();
        assert!(config.enable_loss_curve_analysis);
        assert!(config.enable_convergence_detection);
        assert_eq!(config.moving_average_window, 10);
        assert_eq!(config.max_history_length, 10000);
    }
    #[test]
    fn test_analyzer_new() {
        let analyzer = make_analyzer();
        assert_eq!(analyzer.metrics_history.len(), 0);
    }
    #[test]
    fn test_analyzer_record_metrics() {
        let mut analyzer = make_analyzer();
        analyzer.record_metrics(make_metrics(0, 0, 1.5, 0.001));
        assert_eq!(analyzer.metrics_history.len(), 1);
    }
    #[test]
    fn test_analyzer_record_metrics_limit() {
        let mut analyzer = TrainingDynamicsAnalyzer::new(TrainingDynamicsConfig {
            max_history_length: 5,
            ..TrainingDynamicsConfig::default()
        });
        for i in 0..10 {
            analyzer.record_metrics(make_metrics(i, i, 1.0, 0.001));
        }
        assert_eq!(analyzer.metrics_history.len(), 5);
    }
    #[test]
    fn test_analyzer_clear() {
        let mut analyzer = make_analyzer();
        analyzer.record_metrics(make_metrics(0, 0, 1.0, 0.001));
        analyzer.clear();
        assert!(analyzer.metrics_history.is_empty());
        assert!(analyzer.analysis_cache.is_empty());
    }
    #[test]
    fn test_analyzer_get_training_summary_empty() {
        let analyzer = make_analyzer();
        let summary = analyzer.get_training_summary();
        assert_eq!(summary.total_epochs, 0);
        assert_eq!(summary.metrics_collected, 0);
    }
    #[test]
    fn test_analyzer_get_training_summary_with_data() {
        let mut analyzer = make_analyzer();
        analyzer.record_metrics(make_metrics(5, 500, 0.5, 0.001));
        let summary = analyzer.get_training_summary();
        assert_eq!(summary.total_epochs, 5);
        assert_eq!(summary.total_steps, 500);
        assert!((summary.current_loss - 0.5).abs() < 1e-6);
        assert_eq!(summary.metrics_collected, 1);
    }
    #[test]
    fn test_calculate_std_empty() {
        let analyzer = make_analyzer();
        assert!((analyzer.calculate_std(&[]) - 0.0).abs() < 1e-9);
    }
    #[test]
    fn test_calculate_std_single() {
        let analyzer = make_analyzer();
        assert!((analyzer.calculate_std(&[5.0]) - 0.0).abs() < 1e-9);
    }
    #[test]
    fn test_calculate_std_uniform() {
        let analyzer = make_analyzer();
        assert!((analyzer.calculate_std(&[3.0, 3.0, 3.0]) - 0.0).abs() < 1e-6);
    }
    #[test]
    fn test_calculate_std_known_values() {
        let analyzer = make_analyzer();
        let std = analyzer.calculate_std(&[1.0, 2.0, 3.0]);
        let expected = (2.0_f32 / 3.0).sqrt();
        assert!((std - expected).abs() < 1e-5);
    }
    #[test]
    fn test_detect_loss_trend_insufficient() {
        let analyzer = make_analyzer();
        let trend = analyzer.detect_loss_trend(&[1.0, 0.5]);
        assert!(matches!(trend, LossTrend::Unknown));
    }
    #[test]
    fn test_detect_loss_trend_decreasing() {
        let analyzer = make_analyzer();
        let losses: Vec<f32> = (0..30).map(|i| 2.0 - i as f32 * 0.05).collect();
        let trend = analyzer.detect_loss_trend(&losses);
        assert!(matches!(trend, LossTrend::Decreasing));
    }
    #[test]
    fn test_detect_loss_trend_increasing() {
        let analyzer = make_analyzer();
        let losses: Vec<f32> = (0..30).map(|i| 0.5 + i as f32 * 0.05).collect();
        let trend = analyzer.detect_loss_trend(&losses);
        assert!(matches!(trend, LossTrend::Increasing));
    }
    #[test]
    fn test_smoothness_single_value() {
        let analyzer = make_analyzer();
        let s = analyzer.calculate_smoothness(&[1.0]);
        assert!((s - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_smoothness_constant() {
        let analyzer = make_analyzer();
        let s = analyzer.calculate_smoothness(&[5.0, 5.0, 5.0, 5.0]);
        assert!((s - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_volatility_single() {
        let analyzer = make_analyzer();
        assert!((analyzer.calculate_volatility(&[1.0]) - 0.0).abs() < 1e-6);
    }
    #[test]
    fn test_volatility_constant() {
        let analyzer = make_analyzer();
        assert!((analyzer.calculate_volatility(&[2.0, 2.0, 2.0]) - 0.0).abs() < 1e-6);
    }
    #[test]
    fn test_improvement_rate_empty() {
        let analyzer = make_analyzer();
        assert!((analyzer.calculate_improvement_rate(&[]) - 0.0).abs() < 1e-6);
    }
    #[test]
    fn test_improvement_rate_decreasing() {
        let analyzer = make_analyzer();
        let rate = analyzer.calculate_improvement_rate(&[2.0, 1.5, 1.0]);
        assert!((rate - 1.0 / 3.0).abs() < 1e-5);
    }
    #[test]
    fn test_epochs_since_improvement_best_at_end() {
        let analyzer = make_analyzer();
        let losses = vec![2.0, 1.5, 1.0];
        assert_eq!(analyzer.calculate_epochs_since_improvement(&losses, 1.0), 0);
    }
    #[test]
    fn test_epochs_since_improvement_best_at_start() {
        let analyzer = make_analyzer();
        let losses = vec![0.5, 1.0, 1.5];
        assert_eq!(analyzer.calculate_epochs_since_improvement(&losses, 0.5), 2);
    }
    #[test]
    fn test_moving_averages_short_sequence() {
        let analyzer = make_analyzer();
        let avgs = analyzer.calculate_moving_averages(&[1.0, 2.0, 3.0]);
        assert!(avgs.short_term > 0.0);
        assert!(avgs.medium_term > 0.0);
    }
    #[test]
    fn test_detect_oscillation_short() {
        let analyzer = make_analyzer();
        assert!((analyzer.detect_oscillation(&[1.0, 2.0]) - 0.0).abs() < 1e-6);
    }
    #[test]
    fn test_detect_oscillation_zigzag() {
        let analyzer = make_analyzer();
        let score = analyzer.detect_oscillation(&[1.0, 2.0, 1.0, 2.0, 1.0]);
        assert!(score > 0.5);
    }
    #[test]
    fn test_detect_oscillation_monotonic() {
        let analyzer = make_analyzer();
        let score = analyzer.detect_oscillation(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((score - 0.0).abs() < 1e-6);
    }
    #[test]
    fn test_autocorrelation_short() {
        let analyzer = make_analyzer();
        assert!((analyzer.calculate_autocorrelation(&[1.0], 1) - 0.0).abs() < 1e-6);
    }
    #[test]
    fn test_autocorrelation_constant() {
        let analyzer = make_analyzer();
        assert!((analyzer.calculate_autocorrelation(&[5.0, 5.0, 5.0, 5.0], 1) - 0.0).abs() < 1e-6);
    }
    #[test]
    fn test_linear_trend_short() {
        let analyzer = make_analyzer();
        assert!((analyzer.calculate_linear_trend(&[1.0]) - 0.0).abs() < 1e-6);
    }
    #[test]
    fn test_linear_trend_positive() {
        let analyzer = make_analyzer();
        let trend = analyzer.calculate_linear_trend(&[1.0, 2.0, 3.0, 4.0]);
        assert!(trend > 0.0);
    }
    #[test]
    fn test_linear_trend_negative() {
        let analyzer = make_analyzer();
        let trend = analyzer.calculate_linear_trend(&[4.0, 3.0, 2.0, 1.0]);
        assert!(trend < 0.0);
    }
    #[test]
    fn test_cyclical_pattern_constant() {
        let analyzer = make_analyzer();
        let score = analyzer.detect_cyclical_pattern(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert!((score - 0.0).abs() < 1e-6);
    }
    #[test]
    fn test_lr_schedule_insufficient() {
        let analyzer = make_analyzer();
        assert!(matches!(
            analyzer.detect_lr_schedule_type(),
            LRScheduleType::Unknown
        ));
    }
    #[test]
    fn test_lr_schedule_constant() {
        let mut analyzer = make_analyzer();
        for i in 0..20 {
            analyzer.record_metrics(make_metrics(i, i, 1.0, 0.001));
        }
        assert!(matches!(
            analyzer.detect_lr_schedule_type(),
            LRScheduleType::Constant
        ));
    }
    #[test]
    fn test_loss_statistics_empty() {
        let analyzer = make_analyzer();
        let stats = analyzer.calculate_loss_statistics(&[]);
        assert!((stats.mean - 0.0).abs() < 1e-6);
    }
    #[test]
    fn test_loss_statistics_known() {
        let analyzer = make_analyzer();
        let stats = analyzer.calculate_loss_statistics(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((stats.mean - 3.0).abs() < 1e-5);
        assert!((stats.min - 1.0).abs() < 1e-5);
        assert!((stats.max - 5.0).abs() < 1e-5);
    }
    #[test]
    fn test_memory_utilization_small_batch() {
        let analyzer = make_analyzer();
        let util = analyzer.estimate_memory_utilization(32);
        assert!(util > 0.0 && util < 1.0);
    }
    #[test]
    fn test_memory_utilization_large_batch() {
        let analyzer = make_analyzer();
        let util = analyzer.estimate_memory_utilization(2048);
        assert!((util - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_convergence_speed_empty() {
        let analyzer = make_analyzer();
        assert!((analyzer.estimate_convergence_speed() - 0.0).abs() < 1e-6);
    }
    #[test]
    fn test_convergence_speed_decreasing() {
        let analyzer = make_analyzer_with_decreasing_loss(20);
        let speed = analyzer.estimate_convergence_speed();
        assert!(speed > 0.0);
    }
    #[test]
    fn test_gradient_noise_empty() {
        let analyzer = make_analyzer();
        assert!((analyzer.estimate_gradient_noise_level() - 0.5).abs() < 1e-6);
    }
    #[test]
    fn test_batch_size_recommendations_small() {
        let analyzer = make_analyzer();
        let recs = analyzer.generate_batch_size_recommendations(8, &[]);
        assert!(!recs.is_empty());
        assert_eq!(recs[0].suggested_batch_size, 32);
    }
    #[test]
    fn test_batch_size_recommendations_large() {
        let analyzer = make_analyzer();
        let recs = analyzer.generate_batch_size_recommendations(1024, &[]);
        assert!(!recs.is_empty());
        assert_eq!(recs[0].suggested_batch_size, 256);
    }
    #[test]
    fn test_batch_size_recommendations_normal() {
        let analyzer = make_analyzer();
        let recs = analyzer.generate_batch_size_recommendations(64, &[]);
        assert!(recs.is_empty());
    }
    #[test]
    fn test_plateau_detection_short() {
        let analyzer = make_analyzer();
        assert!(!analyzer.detect_plateau_in_window(&[1.0, 2.0]));
    }
    #[test]
    fn test_plateau_detection_stable() {
        let analyzer = make_analyzer();
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        assert!(analyzer.detect_plateau_in_window(&values));
    }
    #[test]
    fn test_optimal_batch_size_empty() {
        let analyzer = make_analyzer();
        assert_eq!(analyzer.estimate_optimal_batch_size(&[]), 32);
    }
    #[test]
    fn test_plateau_escape_no_plateau() {
        let analyzer = make_analyzer();
        assert!((analyzer.estimate_plateau_escape_probability(&[1.0], 0) - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_plateau_recommendations_no_plateau() {
        let analyzer = make_analyzer();
        let recs = analyzer.generate_plateau_recommendations(false, 0);
        assert!(recs.is_empty());
    }
    #[test]
    fn test_plateau_recommendations_long_plateau() {
        let analyzer = make_analyzer();
        let recs = analyzer.generate_plateau_recommendations(true, 25);
        assert!(!recs.is_empty());
    }
    #[test]
    fn test_plateau_recommendations_very_long() {
        let analyzer = make_analyzer();
        let recs = analyzer.generate_plateau_recommendations(true, 35);
        assert!(recs.len() >= 2);
    }
    #[test]
    fn test_convergence_probability_empty_criteria() {
        let analyzer = make_analyzer();
        assert!((analyzer.calculate_convergence_probability(&[]) - 0.0).abs() < 1e-6);
    }
    #[test]
    fn test_convergence_probability_all_satisfied() {
        let analyzer = make_analyzer();
        let criteria = vec![
            ConvergenceCriterion {
                criterion_type: ConvergenceCriterionType::LossStability,
                current_value: 0.001,
                threshold: 0.01,
                satisfied: true,
                confidence: 0.8,
            },
            ConvergenceCriterion {
                criterion_type: ConvergenceCriterionType::GradientMagnitude,
                current_value: 0.0001,
                threshold: 0.001,
                satisfied: true,
                confidence: 0.7,
            },
        ];
        let prob = analyzer.calculate_convergence_probability(&criteria);
        assert!((prob - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_convergence_status_empty() {
        let analyzer = make_analyzer();
        let status = analyzer.determine_convergence_status(&[]);
        assert!(matches!(status, ConvergenceStatus::TooEarly));
    }
    #[test]
    fn test_epochs_to_convergence_insufficient() {
        let analyzer = make_analyzer();
        assert!(analyzer.estimate_epochs_to_convergence().is_none());
    }
    #[test]
    fn test_epochs_to_convergence_no_improvement() {
        let mut analyzer = make_analyzer();
        for i in 0..10 {
            analyzer.record_metrics(make_metrics(i, i, 1.0, 0.001));
        }
        assert!(analyzer.estimate_epochs_to_convergence().is_none());
    }
    #[test]
    fn test_lr_impact_score_empty() {
        let analyzer = make_analyzer();
        let score = analyzer.calculate_lr_impact_score(&[]);
        assert!((score - 0.0).abs() < 1e-6);
    }
    #[test]
    fn test_estimate_optimal_lr_empty() {
        let analyzer = make_analyzer();
        let lr = analyzer.estimate_optimal_lr(&[]);
        assert!((lr - 0.001).abs() < 1e-6);
    }
    #[test]
    fn test_lr_sensitivity_short() {
        let analyzer = make_analyzer();
        assert!((analyzer.calculate_lr_sensitivity(&[]) - 0.0).abs() < 1e-6);
    }
    #[test]
    fn test_batch_size_efficiency_empty() {
        let analyzer = make_analyzer();
        assert!((analyzer.calculate_batch_size_efficiency(&[]) - 0.0).abs() < 1e-6);
    }
}
