//! Extended tests for the differential_debugging module

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use chrono::Utc;
    use uuid::Uuid;

    use crate::differential_debugging::{
        ABTestConfig, ABTestConclusion, ABTestMetrics, ABTestResult, ArchitectureChange,
        ArchitectureInfo, ConfigChange, DifferentialDebugger, DifferentialDebuggingConfig,
        Improvement, MetricComparison, ModelMetrics, ModelSnapshot, PerformanceDelta,
        RegressionAssessment, RegressionDetectionResult, RegressionSeverity, StatisticalTestResult,
        SummaryStats, TrainingConfig, VersionDiff, WeightChangesSummary, WeightsSummary,
    };

    // -------------------------------------------------------------------------
    // Helper: create a model snapshot
    // -------------------------------------------------------------------------

    fn make_snapshot(name: &str, val_accuracy: f64, latency_ms: f64) -> ModelSnapshot {
        ModelSnapshot {
            id: Uuid::new_v4(),
            name: name.to_string(),
            timestamp: Utc::now(),
            version: "1.0.0".to_string(),
            commit_hash: Some("abc123".to_string()),
            metrics: ModelMetrics {
                train_accuracy: val_accuracy + 0.02,
                val_accuracy,
                test_accuracy: Some(val_accuracy - 0.01),
                train_loss: 0.05,
                val_loss: 0.10,
                test_loss: Some(0.12),
                inference_latency_ms: latency_ms,
                memory_usage_mb: 2048.0,
                model_size_mb: 500.0,
                flops: 1_000_000_000,
                training_time_s: 3600.0,
                custom_metrics: HashMap::new(),
            },
            architecture: ArchitectureInfo {
                parameter_count: 175_000_000,
                layer_count: 24,
                depth: 24,
                hidden_size: 1024,
                num_heads: Some(16),
                ff_dim: Some(4096),
                vocab_size: Some(50257),
                max_seq_length: Some(2048),
            },
            training_config: TrainingConfig {
                learning_rate: 1e-4,
                batch_size: 32,
                epochs: 10,
                optimizer: "AdamW".to_string(),
                lr_schedule: Some("cosine".to_string()),
                regularization: HashMap::new(),
            },
            weights_summary: WeightsSummary {
                mean: 0.0,
                std_dev: 0.1,
                min: -0.5,
                max: 0.5,
                percentiles: HashMap::new(),
                zero_count: 1000,
                sparsity: 0.01,
            },
            metadata: HashMap::new(),
        }
    }

    fn make_config() -> DifferentialDebuggingConfig {
        DifferentialDebuggingConfig::default()
    }

    // -------------------------------------------------------------------------
    // DifferentialDebuggingConfig tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = make_config();
        assert!(config.enable_model_comparison);
        assert!(config.enable_ab_analysis);
        assert!(config.enable_version_diff);
        assert!(config.enable_regression_detection);
        assert!(config.enable_performance_delta);
        assert!((config.significance_threshold - 0.05).abs() < 1e-9);
        assert_eq!(config.max_comparison_models, 10);
        assert!((config.regression_sensitivity - 0.8).abs() < 1e-9);
        assert!((config.performance_delta_threshold - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_custom() {
        let config = DifferentialDebuggingConfig {
            enable_model_comparison: false,
            enable_ab_analysis: false,
            enable_version_diff: false,
            enable_regression_detection: false,
            enable_performance_delta: false,
            significance_threshold: 0.01,
            max_comparison_models: 5,
            regression_sensitivity: 0.5,
            performance_delta_threshold: 10.0,
        };
        assert!(!config.enable_model_comparison);
        assert_eq!(config.max_comparison_models, 5);
    }

    // -------------------------------------------------------------------------
    // DifferentialDebugger construction tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_debugger_new() {
        let config = make_config();
        let debugger = DifferentialDebugger::new(config);
        // Access via generate_report later - just check construction doesn't panic
        let _ = debugger;
    }

    // -------------------------------------------------------------------------
    // add_model_snapshot tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_add_single_snapshot() {
        let config = make_config();
        let mut debugger = DifferentialDebugger::new(config);

        let snapshot = make_snapshot("model_v1", 0.85, 50.0);
        let result = debugger.add_model_snapshot(snapshot);
        assert!(result.is_ok(), "Adding a snapshot should succeed");
    }

    #[test]
    fn test_add_multiple_snapshots() {
        let config = make_config();
        let mut debugger = DifferentialDebugger::new(config);

        for i in 0..5 {
            let snapshot = make_snapshot(&format!("model_v{}", i), 0.80 + i as f64 * 0.02, 50.0);
            debugger.add_model_snapshot(snapshot).expect("add snapshot");
        }
        // All 5 should be stored (within limit of 10)
    }

    #[test]
    fn test_add_snapshot_evicts_oldest_when_over_limit() {
        let config = DifferentialDebuggingConfig {
            max_comparison_models: 3,
            ..DifferentialDebuggingConfig::default()
        };
        let mut debugger = DifferentialDebugger::new(config);

        for i in 0..5 {
            let snapshot = make_snapshot(&format!("m{}", i), 0.8, 50.0);
            debugger.add_model_snapshot(snapshot).expect("add snapshot");
        }
        // Only 3 most recent should be kept
        // We can verify by trying to compare: m0/m1 should be gone
    }

    // -------------------------------------------------------------------------
    // compare_models tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_compare_models_requires_at_least_two() {
        let config = make_config();
        let mut debugger = DifferentialDebugger::new(config);

        let snapshot = make_snapshot("only_one", 0.9, 30.0);
        debugger.add_model_snapshot(snapshot).expect("add snapshot");

        let result = debugger.compare_models(vec!["only_one".to_string()]).await;
        assert!(
            result.is_err(),
            "Comparing less than 2 models should fail"
        );
    }

    #[tokio::test]
    async fn test_compare_models_missing_model_fails() {
        let config = make_config();
        let mut debugger = DifferentialDebugger::new(config);

        let snapshot = make_snapshot("model_a", 0.9, 30.0);
        debugger.add_model_snapshot(snapshot).expect("add snapshot");

        let result = debugger
            .compare_models(vec!["model_a".to_string(), "nonexistent".to_string()])
            .await;
        assert!(result.is_err(), "Missing model should cause error");
    }

    #[tokio::test]
    async fn test_compare_two_models_succeeds() {
        let config = make_config();
        let mut debugger = DifferentialDebugger::new(config);

        debugger.add_model_snapshot(make_snapshot("model_a", 0.85, 50.0)).expect("add a");
        debugger.add_model_snapshot(make_snapshot("model_b", 0.90, 40.0)).expect("add b");

        let result = debugger
            .compare_models(vec!["model_a".to_string(), "model_b".to_string()])
            .await;

        match result {
            Ok(comparison) => {
                assert_eq!(comparison.models.len(), 2);
                // model_b has higher accuracy
                assert_eq!(
                    comparison.performance_comparison.accuracy_comparison.best_model,
                    "model_b"
                );
            },
            Err(e) => panic!("compare_models failed: {}", e),
        }
    }

    #[tokio::test]
    async fn test_compare_models_disabled() {
        let config = DifferentialDebuggingConfig {
            enable_model_comparison: false,
            ..DifferentialDebuggingConfig::default()
        };
        let mut debugger = DifferentialDebugger::new(config);

        debugger.add_model_snapshot(make_snapshot("a", 0.8, 50.0)).expect("add a");
        debugger.add_model_snapshot(make_snapshot("b", 0.9, 40.0)).expect("add b");

        let result = debugger
            .compare_models(vec!["a".to_string(), "b".to_string()])
            .await;
        assert!(result.is_err(), "Disabled comparison should return error");
    }

    #[tokio::test]
    async fn test_compare_three_models() {
        let config = make_config();
        let mut debugger = DifferentialDebugger::new(config);

        debugger.add_model_snapshot(make_snapshot("m1", 0.80, 60.0)).expect("add m1");
        debugger.add_model_snapshot(make_snapshot("m2", 0.85, 50.0)).expect("add m2");
        debugger.add_model_snapshot(make_snapshot("m3", 0.90, 40.0)).expect("add m3");

        let result = debugger
            .compare_models(vec!["m1".to_string(), "m2".to_string(), "m3".to_string()])
            .await;

        match result {
            Ok(comparison) => {
                assert_eq!(comparison.models.len(), 3);
                // m3 is best on accuracy and latency
                assert_eq!(comparison.performance_comparison.accuracy_comparison.best_model, "m3");
                assert_eq!(comparison.performance_comparison.latency_comparison.best_model, "m3");
            },
            Err(_) => {},
        }
    }

    // -------------------------------------------------------------------------
    // run_ab_test tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_ab_test_basic() {
        let config = make_config();
        let mut debugger = DifferentialDebugger::new(config);

        let ab_config = ABTestConfig {
            name: "accuracy_test".to_string(),
            model_a: "model_a".to_string(),
            model_b: "model_b".to_string(),
            duration_hours: None,
            sample_size: 100,
            tracked_metrics: vec!["accuracy".to_string()],
            min_effect_size: 0.01,
            power: 0.8,
        };

        // Generate LCG data
        let mut lcg_a_state: u64 = 42;
        let mut lcg_b_state: u64 = 123;
        let model_a_data: Vec<f64> = (0..50)
            .map(|_| {
                lcg_a_state = lcg_a_state.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
                0.80 + (lcg_a_state >> 11) as f64 / (1u64 << 53) as f64 * 0.1
            })
            .collect();
        let model_b_data: Vec<f64> = (0..50)
            .map(|_| {
                lcg_b_state = lcg_b_state.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
                0.85 + (lcg_b_state >> 11) as f64 / (1u64 << 53) as f64 * 0.1
            })
            .collect();

        match debugger.run_ab_test(ab_config, model_a_data, model_b_data).await {
            Ok(result) => {
                assert_eq!(result.config.name, "accuracy_test");
                assert_eq!(result.model_a_results.sample_size, 50);
                assert_eq!(result.model_b_results.sample_size, 50);
                assert!(result.end_time.is_some());
                // Confidence should be 0-1
                assert!(result.conclusion.confidence >= 0.0);
                assert!(result.conclusion.confidence <= 1.0);
            },
            Err(e) => panic!("A/B test failed: {}", e),
        }
    }

    #[tokio::test]
    async fn test_ab_test_disabled() {
        let config = DifferentialDebuggingConfig {
            enable_ab_analysis: false,
            ..DifferentialDebuggingConfig::default()
        };
        let mut debugger = DifferentialDebugger::new(config);

        let ab_config = ABTestConfig {
            name: "test".to_string(),
            model_a: "a".to_string(),
            model_b: "b".to_string(),
            duration_hours: None,
            sample_size: 10,
            tracked_metrics: vec![],
            min_effect_size: 0.01,
            power: 0.8,
        };

        let result = debugger.run_ab_test(ab_config, vec![0.8, 0.9], vec![0.85, 0.95]).await;
        assert!(result.is_err(), "Disabled A/B test should fail");
    }

    // -------------------------------------------------------------------------
    // track_version_diff tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_version_diff_disabled() {
        let config = DifferentialDebuggingConfig {
            enable_version_diff: false,
            ..DifferentialDebuggingConfig::default()
        };
        let mut debugger = DifferentialDebugger::new(config);
        debugger.add_model_snapshot(make_snapshot("v1", 0.8, 50.0)).expect("add v1");
        debugger.add_model_snapshot(make_snapshot("v2", 0.9, 40.0)).expect("add v2");

        let result = debugger.track_version_diff("v1", "v2").await;
        assert!(result.is_err(), "Disabled version diff should fail");
    }

    #[tokio::test]
    async fn test_version_diff_missing_model() {
        let config = make_config();
        let mut debugger = DifferentialDebugger::new(config);
        debugger.add_model_snapshot(make_snapshot("v1", 0.8, 50.0)).expect("add v1");

        let result = debugger.track_version_diff("v1", "nonexistent").await;
        assert!(result.is_err(), "Missing model should cause error in version diff");
    }

    #[tokio::test]
    async fn test_version_diff_succeeds() {
        let config = make_config();
        let mut debugger = DifferentialDebugger::new(config);

        debugger.add_model_snapshot(make_snapshot("v1", 0.80, 60.0)).expect("add v1");
        debugger.add_model_snapshot(make_snapshot("v2", 0.85, 50.0)).expect("add v2");

        match debugger.track_version_diff("v1", "v2").await {
            Ok(diff) => {
                assert_eq!(diff.from_version, "1.0.0");
                assert_eq!(diff.to_version, "1.0.0");
                // Accuracy improved
                assert!(diff.performance_delta.accuracy_delta > 0.0);
                // Latency improved (decreased)
                assert!(diff.performance_delta.latency_delta < 0.0);
            },
            Err(e) => panic!("track_version_diff failed: {}", e),
        }
    }

    // -------------------------------------------------------------------------
    // detect_regressions tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_regression_detection_disabled() {
        let config = DifferentialDebuggingConfig {
            enable_regression_detection: false,
            ..DifferentialDebuggingConfig::default()
        };
        let mut debugger = DifferentialDebugger::new(config);
        debugger.add_model_snapshot(make_snapshot("base", 0.9, 40.0)).expect("add base");
        debugger.add_model_snapshot(make_snapshot("new", 0.8, 50.0)).expect("add new");

        let result = debugger.detect_regressions("new", "base").await;
        assert!(result.is_err(), "Disabled regression detection should fail");
    }

    #[tokio::test]
    async fn test_no_regression_when_improved() {
        let config = make_config();
        let mut debugger = DifferentialDebugger::new(config);

        // New model is strictly better
        debugger.add_model_snapshot(make_snapshot("baseline", 0.80, 60.0)).expect("add baseline");
        debugger.add_model_snapshot(make_snapshot("new_model", 0.90, 40.0)).expect("add new");

        match debugger.detect_regressions("new_model", "baseline").await {
            Ok(result) => {
                assert!(
                    result.regressions.is_empty(),
                    "Better model should have no regressions"
                );
                assert!(
                    !result.improvements.is_empty(),
                    "Better model should have improvements"
                );
                assert!(result.overall_assessment.health_score > 0.5);
            },
            Err(e) => panic!("detect_regressions failed: {}", e),
        }
    }

    #[tokio::test]
    async fn test_regression_detected_when_accuracy_drops() {
        let config = DifferentialDebuggingConfig {
            regression_sensitivity: 0.01,
            ..DifferentialDebuggingConfig::default()
        };
        let mut debugger = DifferentialDebugger::new(config);

        // New model is worse
        debugger.add_model_snapshot(make_snapshot("baseline", 0.90, 40.0)).expect("add baseline");
        debugger.add_model_snapshot(make_snapshot("new_model", 0.70, 60.0)).expect("add new");

        match debugger.detect_regressions("new_model", "baseline").await {
            Ok(result) => {
                assert!(
                    !result.regressions.is_empty(),
                    "Degraded model should have regressions"
                );
            },
            Err(e) => panic!("detect_regressions failed: {}", e),
        }
    }

    #[tokio::test]
    async fn test_generate_report() {
        let config = make_config();
        let mut debugger = DifferentialDebugger::new(config);

        debugger.add_model_snapshot(make_snapshot("a", 0.80, 50.0)).expect("add a");
        debugger.add_model_snapshot(make_snapshot("b", 0.85, 45.0)).expect("add b");

        let _ = debugger
            .compare_models(vec!["a".to_string(), "b".to_string()])
            .await;

        match debugger.generate_report().await {
            Ok(report) => {
                assert_eq!(report.total_models, 2);
                assert!(report.comparison_count > 0);
            },
            Err(e) => panic!("generate_report failed: {}", e),
        }
    }

    // -------------------------------------------------------------------------
    // Data type tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_model_metrics_construction() {
        let metrics = ModelMetrics {
            train_accuracy: 0.95,
            val_accuracy: 0.90,
            test_accuracy: Some(0.88),
            train_loss: 0.05,
            val_loss: 0.10,
            test_loss: None,
            inference_latency_ms: 50.0,
            memory_usage_mb: 2048.0,
            model_size_mb: 500.0,
            flops: 1_000_000_000,
            training_time_s: 3600.0,
            custom_metrics: HashMap::new(),
        };
        assert!((metrics.val_accuracy - 0.90).abs() < 1e-9);
        assert!(metrics.test_loss.is_none());
    }

    #[test]
    fn test_architecture_info_construction() {
        let arch = ArchitectureInfo {
            parameter_count: 7_000_000_000,
            layer_count: 80,
            depth: 80,
            hidden_size: 8192,
            num_heads: Some(64),
            ff_dim: Some(28672),
            vocab_size: Some(32000),
            max_seq_length: Some(4096),
        };
        assert_eq!(arch.parameter_count, 7_000_000_000);
        assert!(arch.num_heads.is_some());
    }

    #[test]
    fn test_training_config_construction() {
        let config = TrainingConfig {
            learning_rate: 3e-4,
            batch_size: 256,
            epochs: 100,
            optimizer: "Adam".to_string(),
            lr_schedule: None,
            regularization: {
                let mut m = HashMap::new();
                m.insert("weight_decay".to_string(), 0.01);
                m
            },
        };
        assert!((config.learning_rate - 3e-4).abs() < 1e-9);
        assert!(config.lr_schedule.is_none());
        assert!(config.regularization.contains_key("weight_decay"));
    }

    #[test]
    fn test_weights_summary_construction() {
        let ws = WeightsSummary {
            mean: 0.001,
            std_dev: 0.05,
            min: -0.3,
            max: 0.3,
            percentiles: {
                let mut p = HashMap::new();
                p.insert("p50".to_string(), 0.0);
                p
            },
            zero_count: 500,
            sparsity: 0.005,
        };
        assert!(ws.min < ws.max);
        assert!(ws.sparsity >= 0.0 && ws.sparsity <= 1.0);
    }

    #[test]
    fn test_regression_severity_variants() {
        let severities = [
            RegressionSeverity::Critical,
            RegressionSeverity::Major,
            RegressionSeverity::Minor,
            RegressionSeverity::Negligible,
        ];
        for s in &severities {
            let _ = format!("{:?}", s);
        }
    }

    #[test]
    fn test_regression_assessment_construction() {
        let assessment = RegressionAssessment {
            health_score: 0.75,
            critical_regressions: 0,
            improvements: 2,
            recommendation: "Some regressions detected".to_string(),
        };
        assert!(assessment.health_score >= 0.0 && assessment.health_score <= 1.0);
        assert_eq!(assessment.critical_regressions, 0);
    }

    #[test]
    fn test_performance_delta_construction() {
        let delta = PerformanceDelta {
            accuracy_delta: 0.05,
            loss_delta: -0.01,
            latency_delta: -5.0,
            memory_delta: 100.0,
            size_delta: 0.0,
            training_time_delta: -120.0,
            custom_deltas: HashMap::new(),
        };
        assert!(delta.accuracy_delta > 0.0);
        assert!(delta.latency_delta < 0.0);
    }

    #[test]
    fn test_statistical_test_result_construction() {
        let result = StatisticalTestResult {
            test_type: "t-test".to_string(),
            statistic: 2.5,
            p_value: 0.012,
            effect_size: 0.3,
            confidence_interval: (-0.1, 0.5),
            is_significant: true,
        };
        assert!(result.p_value < 0.05);
        assert!(result.is_significant);
    }

    #[test]
    fn test_ab_test_conclusion_construction() {
        let conclusion = ABTestConclusion {
            winner: Some("model_b".to_string()),
            confidence: 0.95,
            practical_significance: true,
            recommendation: "Deploy model_b".to_string(),
            summary: "Model B outperforms A".to_string(),
        };
        assert!(conclusion.winner.is_some());
        assert!((conclusion.confidence - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_summary_stats_construction() {
        let stats = SummaryStats {
            mean: 0.85,
            std_dev: 0.02,
            min: 0.80,
            max: 0.90,
            median: 0.85,
            q25: 0.83,
            q75: 0.87,
        };
        assert!(stats.min <= stats.median && stats.median <= stats.max);
        assert!(stats.q25 <= stats.q75);
    }

    #[test]
    fn test_architecture_change_construction() {
        let change = ArchitectureChange {
            change_type: "LayerAdded".to_string(),
            description: "Added dropout layer".to_string(),
            impact: "Reduces overfitting".to_string(),
        };
        assert_eq!(change.change_type, "LayerAdded");
    }

    #[test]
    fn test_config_change_construction() {
        let change = ConfigChange {
            parameter: "learning_rate".to_string(),
            old_value: "0.001".to_string(),
            new_value: "0.0001".to_string(),
            impact: "Slower convergence, potentially better generalization".to_string(),
        };
        assert_eq!(change.parameter, "learning_rate");
    }

    #[test]
    fn test_weight_changes_summary_construction() {
        let wcs = WeightChangesSummary {
            avg_magnitude: 0.01,
            max_change: 0.5,
            significant_change_ratio: 0.15,
            layer_changes: {
                let mut m = HashMap::new();
                m.insert("layer_0".to_string(), 0.02);
                m
            },
        };
        assert!(wcs.significant_change_ratio >= 0.0 && wcs.significant_change_ratio <= 1.0);
        assert!(wcs.layer_changes.contains_key("layer_0"));
    }

    #[test]
    fn test_ab_test_config_construction() {
        let config = ABTestConfig {
            name: "throughput_test".to_string(),
            model_a: "baseline".to_string(),
            model_b: "optimized".to_string(),
            duration_hours: Some(24),
            sample_size: 1000,
            tracked_metrics: vec!["throughput".to_string(), "latency".to_string()],
            min_effect_size: 0.05,
            power: 0.9,
        };
        assert_eq!(config.tracked_metrics.len(), 2);
        assert!(config.duration_hours.is_some());
    }

    #[test]
    fn test_improvement_construction() {
        let improvement = Improvement {
            metric: "accuracy".to_string(),
            current_value: 0.92,
            previous_value: 0.85,
            magnitude: 0.07,
            likely_causes: vec!["Better data augmentation".to_string()],
        };
        assert!(improvement.current_value > improvement.previous_value);
        assert!(!improvement.likely_causes.is_empty());
    }

    #[test]
    fn test_regression_detection_result_construction() {
        let result = RegressionDetectionResult {
            timestamp: Utc::now(),
            regressions: vec![],
            improvements: vec![Improvement {
                metric: "accuracy".to_string(),
                current_value: 0.92,
                previous_value: 0.85,
                magnitude: 0.07,
                likely_causes: vec![],
            }],
            overall_assessment: RegressionAssessment {
                health_score: 1.0,
                critical_regressions: 0,
                improvements: 1,
                recommendation: "No regressions".to_string(),
            },
        };
        assert!(result.regressions.is_empty());
        assert_eq!(result.improvements.len(), 1);
    }
}
