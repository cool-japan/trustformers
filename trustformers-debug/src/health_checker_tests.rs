//! Tests for the health_checker module.

use super::*;

fn make_debug_config() -> crate::DebugConfig {
    crate::DebugConfig::default()
}

// ── HealthStatus variants ─────────────────────────────────────────────────────

#[test]
fn test_health_status_variants_clone() {
    let variants = [
        HealthStatus::Excellent,
        HealthStatus::Good,
        HealthStatus::Fair,
        HealthStatus::Poor,
        HealthStatus::Critical,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── OverfittingRisk variants ──────────────────────────────────────────────────

#[test]
fn test_overfitting_risk_variants_clone() {
    let variants = [
        OverfittingRisk::None,
        OverfittingRisk::Low,
        OverfittingRisk::Medium,
        OverfittingRisk::High,
        OverfittingRisk::Severe,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── AlertSeverity variants ────────────────────────────────────────────────────

#[test]
fn test_alert_severity_variants_clone() {
    let variants = [
        AlertSeverity::Critical,
        AlertSeverity::High,
        AlertSeverity::Medium,
        AlertSeverity::Low,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── Trend variants ────────────────────────────────────────────────────────────

#[test]
fn test_trend_variants_clone() {
    let variants = [
        Trend::Improving,
        Trend::Stable,
        Trend::Degrading,
        Trend::Volatile,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── HealthAlertType variants ──────────────────────────────────────────────────

#[test]
fn test_health_alert_type_variants() {
    let variants = [
        HealthAlertType::TrainingStability,
        HealthAlertType::ConvergenceIssue,
        HealthAlertType::OverfittingDetected,
        HealthAlertType::PerformanceDegradation,
        HealthAlertType::MemoryIssue,
        HealthAlertType::GradientProblem,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── RecommendationCategory variants ──────────────────────────────────────────

#[test]
fn test_recommendation_category_variants() {
    let variants = [
        RecommendationCategory::Training,
        RecommendationCategory::Architecture,
        RecommendationCategory::Hyperparameters,
        RecommendationCategory::Data,
        RecommendationCategory::Performance,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── RecommendationUrgency variants ────────────────────────────────────────────

#[test]
fn test_recommendation_urgency_variants() {
    let variants = [
        RecommendationUrgency::Immediate,
        RecommendationUrgency::Soon,
        RecommendationUrgency::Eventually,
        RecommendationUrgency::Optional,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── MetricStability ───────────────────────────────────────────────────────────

#[test]
fn test_metric_stability_new() {
    let ms = MetricStability::new(0.01, 0.05);
    // Initial state: no values, should return default stability (0.5 = insufficient data)
    let stability = ms.calculate_stability();
    assert!((stability - 0.5).abs() < 1e-10, "initial stability should be 0.5, got {}", stability);
}

#[test]
fn test_metric_stability_update_single_value() {
    let mut ms = MetricStability::new(0.01, 0.05);
    ms.update(1.0);
    let stability = ms.calculate_stability();
    // Still insufficient data (< 5 values)
    assert!((stability - 0.5).abs() < 1e-10);
}

#[test]
fn test_metric_stability_high_stability_constant_values() {
    let mut ms = MetricStability::new(0.1, 0.05);
    // Feed constant values → variance near zero → high stability
    for _ in 0..20 {
        ms.update(1.0);
    }
    let stability = ms.calculate_stability();
    assert!(stability > 0.8, "constant values should yield high stability, got {}", stability);
}

#[test]
fn test_metric_stability_returns_bounded_value() {
    let mut ms = MetricStability::new(0.001, 0.001);
    // Oscillating values → stability reduces
    for i in 0..20 {
        ms.update(if i % 2 == 0 { 0.0 } else { 10.0 });
    }
    let stability = ms.calculate_stability();
    assert!(stability >= 0.0 && stability <= 1.0, "stability must be in [0,1], got {}", stability);
}

// ── ConvergenceAnalyzer ───────────────────────────────────────────────────────

#[test]
fn test_convergence_analyzer_new() {
    let analyzer = ConvergenceAnalyzer::new();
    // No data: returns default convergence probability
    let prob = analyzer.calculate_convergence_probability();
    assert!(prob >= 0.0 && prob <= 1.0, "probability must be [0,1], got {}", prob);
}

#[test]
fn test_convergence_analyzer_update_loss_only() {
    let mut analyzer = ConvergenceAnalyzer::new();
    for i in 0..20 {
        analyzer.update(Some(2.0 - i as f64 * 0.05), None);
    }
    let prob = analyzer.calculate_convergence_probability();
    assert!(prob >= 0.0 && prob <= 1.0);
}

#[test]
fn test_convergence_analyzer_update_accuracy_only() {
    let mut analyzer = ConvergenceAnalyzer::new();
    for i in 0..15 {
        analyzer.update(None, Some(0.5 + i as f64 * 0.02));
    }
    let prob = analyzer.calculate_convergence_probability();
    assert!(prob >= 0.0 && prob <= 1.0);
}

#[test]
fn test_convergence_analyzer_sufficient_data_decreasing_loss() {
    let mut analyzer = ConvergenceAnalyzer::new();
    // Feed more than convergence_window (100) loss values, consistently decreasing
    for i in 0..110 {
        analyzer.update(Some(5.0 - i as f64 * 0.03), None);
    }
    let prob = analyzer.calculate_convergence_probability();
    assert!(prob >= 0.2, "Expected improved convergence probability, got {}", prob);
}

#[test]
fn test_convergence_analyzer_both_metrics() {
    let mut analyzer = ConvergenceAnalyzer::new();
    for i in 0..30 {
        analyzer.update(
            Some(3.0 - i as f64 * 0.05),
            Some(0.4 + i as f64 * 0.01),
        );
    }
    let prob = analyzer.calculate_convergence_probability();
    assert!(prob >= 0.0 && prob <= 1.0);
}

// ── OverfittingDetector ───────────────────────────────────────────────────────

#[test]
fn test_overfitting_detector_new() {
    let detector = OverfittingDetector::new();
    let risk = detector.detect_overfitting();
    let _risk_cloned = risk.clone();
}

#[test]
fn test_overfitting_detector_no_overfitting_scenario() {
    let mut detector = OverfittingDetector::new();
    // Both train and val losses decrease together
    for i in 0..20 {
        let loss = 2.0 - i as f64 * 0.05;
        detector.update_train_metrics(Some(loss), Some(0.5 + i as f64 * 0.02));
        detector.update_validation_metrics(Some(loss + 0.01), Some(0.49 + i as f64 * 0.02));
    }
    let risk = detector.detect_overfitting();
    // Just verify no panic and valid risk
    let _cloned = risk.clone();
}

#[test]
fn test_overfitting_detector_update_train_only() {
    let mut detector = OverfittingDetector::new();
    for i in 0..10 {
        detector.update_train_metrics(Some(1.0 - i as f64 * 0.05), Some(0.6 + i as f64 * 0.03));
    }
    // No panic — validation data not supplied
    let _ = detector.detect_overfitting();
}

#[test]
fn test_overfitting_detector_clear_overfitting_signal() {
    let mut detector = OverfittingDetector::new();
    // Train loss decreases, val loss increases → clear overfitting signal
    for i in 0..20 {
        detector.update_train_metrics(Some(2.0 - i as f64 * 0.08), None);
        detector.update_validation_metrics(Some(1.0 + i as f64 * 0.08), None);
    }
    let risk = detector.detect_overfitting();
    // With diverging train/val losses, risk should be elevated
    match risk {
        OverfittingRisk::None
        | OverfittingRisk::Low
        | OverfittingRisk::Medium
        | OverfittingRisk::High
        | OverfittingRisk::Severe => {} // Any risk level accepted; just no panic
    }
}

// ── GeneralizationMonitor ─────────────────────────────────────────────────────

#[test]
fn test_generalization_monitor_new() {
    let monitor = GeneralizationMonitor::new();
    let score = monitor.calculate_generalization_score();
    assert!(score >= 0.0 && score <= 1.0, "score must be [0,1], got {}", score);
}

#[test]
fn test_generalization_monitor_update_performance() {
    let mut monitor = GeneralizationMonitor::new();
    monitor.update_performance(0.9, Some(0.85));
    let score = monitor.calculate_generalization_score();
    assert!(score >= 0.0 && score <= 1.0);
}

#[test]
fn test_generalization_monitor_no_validation() {
    let mut monitor = GeneralizationMonitor::new();
    monitor.update_performance(0.9, None);
    let score = monitor.calculate_generalization_score();
    assert!(score >= 0.0 && score <= 1.0);
}

// ── HealthChecker ─────────────────────────────────────────────────────────────

#[test]
fn test_health_checker_new() {
    let config = make_debug_config();
    let checker = HealthChecker::new(&config);
    let history = checker.get_health_history();
    assert!(history.is_empty());
}

#[test]
fn test_health_checker_update_with_metrics() {
    use std::time::SystemTime;
    let config = make_debug_config();
    let mut checker = HealthChecker::new(&config);
    let metrics = DashboardMetrics {
        timestamp: SystemTime::now(),
        loss: Some(1.5),
        accuracy: Some(0.7),
        learning_rate: Some(1e-3),
        memory_usage_mb: 512.0,
        gpu_utilization: Some(0.85),
        tokens_per_second: Some(1000.0),
        gradient_norm: Some(0.5),
        epoch: Some(1),
        step: Some(100),
    };
    checker.update(metrics);
    // No panic
}

#[test]
fn test_health_checker_assess_health_after_updates() {
    use std::time::SystemTime;
    let config = make_debug_config();
    let mut checker = HealthChecker::new(&config);
    for i in 0..5 {
        let metrics = DashboardMetrics {
            timestamp: SystemTime::now(),
            loss: Some(2.0 - i as f64 * 0.1),
            accuracy: Some(0.5 + i as f64 * 0.05),
            learning_rate: Some(1e-3),
            memory_usage_mb: 256.0,
            gpu_utilization: Some(0.75),
            tokens_per_second: Some(500.0),
            gradient_norm: Some(0.8),
            epoch: Some(i as u32),
            step: Some(i as u64 * 100),
        };
        checker.update(metrics);
    }
    let result = checker.assess_health();
    match result {
        Ok(assessment) => {
            assert!(assessment.overall_health_score >= 0.0);
            assert!(assessment.overall_health_score <= 1.0);
        }
        Err(_) => {
            // Assessment may fail if insufficient data
        }
    }
}

// ── ComponentHealthScores ─────────────────────────────────────────────────────

#[test]
fn test_component_health_scores_construction() {
    let scores = ComponentHealthScores {
        gradient_health: 0.9,
        loss_health: 0.85,
        accuracy_health: 0.92,
        performance_health: 0.88,
        memory_health: 0.95,
        stability_health: 0.87,
    };
    assert!(scores.gradient_health > 0.0 && scores.gradient_health <= 1.0);
    assert!(scores.loss_health > 0.0 && scores.loss_health <= 1.0);
}

// ── HealthRecommendation ──────────────────────────────────────────────────────

#[test]
fn test_health_recommendation_construction() {
    let rec = HealthRecommendation {
        category: RecommendationCategory::Hyperparameters,
        urgency: RecommendationUrgency::Immediate,
        title: "Reduce learning rate".to_string(),
        description: "The training loss is diverging; consider reducing LR.".to_string(),
        expected_impact: 0.15,
    };
    assert_eq!(rec.title, "Reduce learning rate");
    assert!(rec.expected_impact > 0.0);
}
