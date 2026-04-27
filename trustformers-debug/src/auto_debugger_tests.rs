//! Tests for auto_debugger module

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use crate::auto_debugger::{
        ArchitectureSuggestion, AutoDebugReport, AutoDebugger, CodeExample, DataStrategy,
        DebugContext, DetectedIssue, EstimatedEffort, Evidence, ExpectedImpact, FixPriority,
        FixSuggestion, FixType, HyperparameterRecommendation, IssuePattern, IssueType,
        IssueSeverity, KnowledgeBase, ModelInfo, OptimizationAttempt, TrainingRecipeOptimization,
        TrainingSchedule,
    };
    use crate::core::session::DebugConfig;
    use crate::dashboard::DashboardMetrics;

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    fn make_debug_config() -> DebugConfig {
        DebugConfig::default()
    }

    fn make_dashboard_metrics(loss: Option<f64>, gpu_util: Option<f64>) -> DashboardMetrics {
        DashboardMetrics {
            timestamp: std::time::SystemTime::now(),
            loss,
            accuracy: None,
            learning_rate: None,
            memory_usage_mb: 512.0,
            gpu_utilization: gpu_util,
            tokens_per_second: None,
            gradient_norm: None,
            epoch: None,
            step: None,
        }
    }

    fn empty_context<'a>(metrics: &'a [DashboardMetrics]) -> DebugContext<'a> {
        DebugContext {
            profiler_report: None,
            gradient_report: None,
            anomaly_report: None,
            recent_metrics: metrics,
            training_duration: Duration::from_secs(60),
            model_info: None,
        }
    }

    // -------------------------------------------------------------------------
    // AutoDebugger construction
    // -------------------------------------------------------------------------

    #[test]
    fn test_auto_debugger_new() {
        let config = make_debug_config();
        let debugger = AutoDebugger::new(&config);
        // Verify the debugger was created – we can inspect optimization history as a proxy
        assert!(debugger.get_optimization_history().is_empty());
    }

    #[test]
    fn test_auto_debugger_analyze_empty_context() {
        let config = make_debug_config();
        let debugger = AutoDebugger::new(&config);
        let metrics: Vec<DashboardMetrics> = vec![];
        let context = empty_context(&metrics);
        let report = debugger.analyze_issues(&context);
        assert!(report.is_ok(), "analyze_issues should succeed with empty context");
        let report = report.expect("report should be Ok");
        assert!(report.detected_issues.is_empty());
    }

    // -------------------------------------------------------------------------
    // IssueType variants
    // -------------------------------------------------------------------------

    #[test]
    fn test_issue_type_equality() {
        assert_eq!(IssueType::VanishingGradients, IssueType::VanishingGradients);
        assert_ne!(IssueType::VanishingGradients, IssueType::ExplodingGradients);
    }

    #[test]
    fn test_all_issue_type_variants_hashable() {
        let mut map: HashMap<IssueType, usize> = HashMap::new();
        let variants = [
            IssueType::VanishingGradients,
            IssueType::ExplodingGradients,
            IssueType::LearningRateProblems,
            IssueType::OverfittingDetected,
            IssueType::UnderfittingDetected,
            IssueType::TrainingStalled,
            IssueType::LossNotDecreasing,
            IssueType::UnstableTraining,
            IssueType::MemoryIssues,
            IssueType::ModelTooLarge,
            IssueType::ModelTooSmall,
            IssueType::InappropriateArchitecture,
            IssueType::LayerMismatch,
            IssueType::ActivationProblems,
            IssueType::DataImbalance,
            IssueType::DataLeakage,
            IssueType::InsufficientData,
            IssueType::DataQualityIssues,
            IssueType::BatchSizeProblems,
            IssueType::SlowTraining,
            IssueType::LowGpuUtilization,
            IssueType::MemoryBottleneck,
            IssueType::IoBottleneck,
            IssueType::ComputeBottleneck,
            IssueType::LearningRateTooHigh,
            IssueType::LearningRateTooLow,
            IssueType::BatchSizeTooLarge,
            IssueType::BatchSizeTooSmall,
            IssueType::RegularizationIssues,
        ];
        for (i, variant) in variants.iter().enumerate() {
            map.insert(variant.clone(), i);
        }
        assert_eq!(map.len(), variants.len());
    }

    // -------------------------------------------------------------------------
    // IssueSeverity
    // -------------------------------------------------------------------------

    #[test]
    fn test_issue_severity_variants_debug() {
        let severities = [
            IssueSeverity::Critical,
            IssueSeverity::High,
            IssueSeverity::Medium,
            IssueSeverity::Low,
            IssueSeverity::Info,
        ];
        for s in &severities {
            let dbg = format!("{:?}", s);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // FixType / FixPriority / EstimatedEffort
    // -------------------------------------------------------------------------

    #[test]
    fn test_fix_type_variants_debug() {
        let types = [
            FixType::HyperparameterAdjustment,
            FixType::ArchitectureChange,
            FixType::TrainingProcedure,
            FixType::DataProcessing,
            FixType::OptimizationTechnique,
            FixType::EnvironmentConfig,
        ];
        for t in &types {
            let dbg = format!("{:?}", t);
            assert!(!dbg.is_empty());
        }
    }

    #[test]
    fn test_fix_priority_variants_debug() {
        let priorities = [
            FixPriority::Critical,
            FixPriority::High,
            FixPriority::Medium,
            FixPriority::Low,
        ];
        for p in &priorities {
            let dbg = format!("{:?}", p);
            assert!(!dbg.is_empty());
        }
    }

    #[test]
    fn test_estimated_effort_variants_debug() {
        let efforts = [
            EstimatedEffort::Trivial,
            EstimatedEffort::Easy,
            EstimatedEffort::Medium,
            EstimatedEffort::Hard,
            EstimatedEffort::Complex,
        ];
        for e in &efforts {
            let dbg = format!("{:?}", e);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // DetectedIssue construction
    // -------------------------------------------------------------------------

    #[test]
    fn test_detected_issue_construction() {
        let issue = DetectedIssue {
            issue_type: IssueType::VanishingGradients,
            severity: IssueSeverity::High,
            confidence: 0.9,
            description: "Vanishing gradients".to_string(),
            evidence: vec![Evidence {
                metric_name: "gradient_norm".to_string(),
                observed_value: 0.001,
                expected_range: (0.01, 1.0),
                explanation: "Too small".to_string(),
            }],
            metrics: HashMap::new(),
            detected_at: chrono::Utc::now(),
        };
        assert_eq!(issue.confidence, 0.9);
        assert_eq!(issue.evidence.len(), 1);
    }

    // -------------------------------------------------------------------------
    // Evidence
    // -------------------------------------------------------------------------

    #[test]
    fn test_evidence_range() {
        let evidence = Evidence {
            metric_name: "test_metric".to_string(),
            observed_value: 5.0,
            expected_range: (1.0, 10.0),
            explanation: "Within range".to_string(),
        };
        assert!(evidence.observed_value >= evidence.expected_range.0);
        assert!(evidence.observed_value <= evidence.expected_range.1);
    }

    // -------------------------------------------------------------------------
    // ModelInfo
    // -------------------------------------------------------------------------

    #[test]
    fn test_model_info_construction() {
        let model_info = ModelInfo {
            model_type: "transformer".to_string(),
            parameter_count: 125_000_000,
            layer_count: 12,
            architecture_details: HashMap::new(),
        };
        assert_eq!(model_info.layer_count, 12);
    }

    // -------------------------------------------------------------------------
    // AutoDebugger with model info
    // -------------------------------------------------------------------------

    #[test]
    fn test_analyze_with_large_model_detects_model_too_large() {
        let config = make_debug_config();
        let debugger = AutoDebugger::new(&config);
        let model_info = ModelInfo {
            model_type: "gpt".to_string(),
            parameter_count: 2_000_000_000,
            layer_count: 96,
            architecture_details: HashMap::new(),
        };
        let metrics: Vec<DashboardMetrics> = vec![];
        let context = DebugContext {
            profiler_report: None,
            gradient_report: None,
            anomaly_report: None,
            recent_metrics: &metrics,
            training_duration: Duration::from_secs(60),
            model_info: Some(&model_info),
        };
        let report = debugger.analyze_issues(&context).expect("analyze should succeed");
        let has_model_too_large = report
            .detected_issues
            .iter()
            .any(|i| matches!(i.issue_type, IssueType::ModelTooLarge));
        assert!(has_model_too_large, "Should detect ModelTooLarge for 2B params");
    }

    #[test]
    fn test_analyze_with_very_deep_model() {
        let config = make_debug_config();
        let debugger = AutoDebugger::new(&config);
        let model_info = ModelInfo {
            model_type: "deep".to_string(),
            parameter_count: 50_000_000,
            layer_count: 150,
            architecture_details: HashMap::new(),
        };
        let metrics: Vec<DashboardMetrics> = vec![];
        let context = DebugContext {
            profiler_report: None,
            gradient_report: None,
            anomaly_report: None,
            recent_metrics: &metrics,
            training_duration: Duration::from_secs(60),
            model_info: Some(&model_info),
        };
        let report = debugger.analyze_issues(&context).expect("analyze should succeed");
        let has_arch_issue = report
            .detected_issues
            .iter()
            .any(|i| matches!(i.issue_type, IssueType::InappropriateArchitecture));
        assert!(has_arch_issue, "Should detect InappropriateArchitecture for 150 layers");
    }

    // -------------------------------------------------------------------------
    // GPU utilization detection
    // -------------------------------------------------------------------------

    #[test]
    fn test_analyze_low_gpu_utilization() {
        let config = make_debug_config();
        let debugger = AutoDebugger::new(&config);
        let metrics = vec![make_dashboard_metrics(None, Some(0.3))];
        let context = empty_context(&metrics);
        let report = debugger.analyze_issues(&context).expect("analyze should succeed");
        let has_low_gpu = report
            .detected_issues
            .iter()
            .any(|i| matches!(i.issue_type, IssueType::LowGpuUtilization));
        assert!(has_low_gpu, "Should detect LowGpuUtilization when GPU at 30%");
    }

    #[test]
    fn test_analyze_high_gpu_utilization_no_issue() {
        let config = make_debug_config();
        let debugger = AutoDebugger::new(&config);
        let metrics = vec![make_dashboard_metrics(None, Some(0.9))];
        let context = empty_context(&metrics);
        let report = debugger.analyze_issues(&context).expect("analyze should succeed");
        let has_low_gpu = report
            .detected_issues
            .iter()
            .any(|i| matches!(i.issue_type, IssueType::LowGpuUtilization));
        assert!(!has_low_gpu, "Should not detect LowGpuUtilization when GPU at 90%");
    }

    // -------------------------------------------------------------------------
    // Stalled training detection
    // -------------------------------------------------------------------------

    #[test]
    fn test_analyze_stalled_training() {
        let config = make_debug_config();
        let debugger = AutoDebugger::new(&config);
        // Create 10 metrics with nearly identical loss to simulate stalled training
        let metrics: Vec<DashboardMetrics> = (0..10)
            .map(|_| make_dashboard_metrics(Some(1.0), None))
            .collect();
        let context = empty_context(&metrics);
        let report = debugger.analyze_issues(&context).expect("analyze should succeed");
        let has_stalled = report
            .detected_issues
            .iter()
            .any(|i| matches!(i.issue_type, IssueType::TrainingStalled));
        assert!(has_stalled, "Should detect stalled training with constant loss");
    }

    // -------------------------------------------------------------------------
    // Hyperparameter recommendations
    // -------------------------------------------------------------------------

    #[test]
    fn test_hyperparameter_recommendation_high_loss() {
        let config = make_debug_config();
        let debugger = AutoDebugger::new(&config);
        let metrics = vec![make_dashboard_metrics(Some(2.5), None)];
        let context = empty_context(&metrics);
        let report = debugger.analyze_issues(&context).expect("analyze should succeed");
        let has_lr_recommendation = report
            .hyperparameter_recommendations
            .iter()
            .any(|r| r.parameter == "learning_rate");
        assert!(has_lr_recommendation, "Should recommend learning rate when loss > 1.0");
    }

    // -------------------------------------------------------------------------
    // OptimizationAttempt recording
    // -------------------------------------------------------------------------

    #[test]
    fn test_record_optimization_attempt() {
        let config = make_debug_config();
        let mut debugger = AutoDebugger::new(&config);

        let attempt = OptimizationAttempt {
            attempt_id: "test_001".to_string(),
            issue_addressed: IssueType::LearningRateTooHigh,
            fix_applied: "Reduced LR from 0.01 to 0.001".to_string(),
            before_metrics: {
                let mut m = HashMap::new();
                m.insert("loss".to_string(), 2.5);
                m
            },
            after_metrics: None,
            success: None,
            notes: "Test attempt".to_string(),
            timestamp: chrono::Utc::now(),
        };

        debugger.record_optimization_attempt(attempt);
        assert_eq!(debugger.get_optimization_history().len(), 1);
    }

    #[test]
    fn test_record_multiple_optimization_attempts() {
        let config = make_debug_config();
        let mut debugger = AutoDebugger::new(&config);

        for i in 0..5 {
            let attempt = OptimizationAttempt {
                attempt_id: format!("attempt_{}", i),
                issue_addressed: IssueType::VanishingGradients,
                fix_applied: "Added residual connections".to_string(),
                before_metrics: HashMap::new(),
                after_metrics: None,
                success: Some(true),
                notes: String::new(),
                timestamp: chrono::Utc::now(),
            };
            debugger.record_optimization_attempt(attempt);
        }

        assert_eq!(debugger.get_optimization_history().len(), 5);
    }

    #[test]
    fn test_optimization_attempt_with_success() {
        let config = make_debug_config();
        let mut debugger = AutoDebugger::new(&config);

        let mut before = HashMap::new();
        before.insert("loss".to_string(), 3.0);
        let mut after = HashMap::new();
        after.insert("loss".to_string(), 1.5);

        let attempt = OptimizationAttempt {
            attempt_id: "success_001".to_string(),
            issue_addressed: IssueType::ExplodingGradients,
            fix_applied: "Gradient clipping".to_string(),
            before_metrics: before,
            after_metrics: Some(after),
            success: Some(true),
            notes: "Gradient clipping applied".to_string(),
            timestamp: chrono::Utc::now(),
        };

        debugger.record_optimization_attempt(attempt);
        let history = debugger.get_optimization_history();
        if let Some(recorded) = history.first() {
            assert_eq!(recorded.success, Some(true));
        }
    }

    // -------------------------------------------------------------------------
    // KnowledgeBase
    // -------------------------------------------------------------------------

    #[test]
    fn test_knowledge_base_new() {
        let kb = KnowledgeBase::new();
        let dbg = format!("{:?}", kb);
        assert!(!dbg.is_empty());
    }

    #[test]
    fn test_knowledge_base_default() {
        let kb = KnowledgeBase::default();
        let dbg = format!("{:?}", kb);
        assert!(!dbg.is_empty());
    }

    // -------------------------------------------------------------------------
    // IssuePattern
    // -------------------------------------------------------------------------

    #[test]
    fn test_issue_pattern_construction() {
        let pattern = IssuePattern {
            symptoms: vec!["loss not decreasing".to_string()],
            common_causes: vec!["lr too high".to_string()],
            diagnostic_metrics: vec!["loss".to_string()],
            typical_solutions: vec!["reduce lr".to_string()],
        };
        assert_eq!(pattern.symptoms.len(), 1);
    }

    // -------------------------------------------------------------------------
    // FixSuggestion
    // -------------------------------------------------------------------------

    #[test]
    fn test_fix_suggestion_construction() {
        let suggestion = FixSuggestion {
            fix_id: "fs_001".to_string(),
            fix_type: FixType::HyperparameterAdjustment,
            title: "Reduce LR".to_string(),
            description: "Lower learning rate".to_string(),
            implementation_steps: vec!["Step 1".to_string()],
            expected_impact: ExpectedImpact {
                performance_improvement: 0.1,
                training_speed_improvement: 0.0,
                stability_improvement: 0.2,
                memory_usage_change: 0.0,
            },
            priority: FixPriority::High,
            estimated_effort: EstimatedEffort::Trivial,
            prerequisites: vec![],
            code_examples: vec![CodeExample {
                language: "python".to_string(),
                code: "lr = 0.001".to_string(),
                explanation: "Set lr".to_string(),
            }],
        };
        assert_eq!(suggestion.fix_id, "fs_001");
        assert_eq!(suggestion.code_examples.len(), 1);
    }

    // -------------------------------------------------------------------------
    // AutoDebugReport
    // -------------------------------------------------------------------------

    #[test]
    fn test_auto_debug_report_fields() {
        let config = make_debug_config();
        let debugger = AutoDebugger::new(&config);
        let metrics: Vec<DashboardMetrics> = vec![];
        let context = empty_context(&metrics);
        let report = debugger.analyze_issues(&context).expect("report ok");
        // confidence_score should be in [0, 1]
        assert!(
            report.confidence_score >= 0.0 && report.confidence_score <= 1.0,
            "confidence_score should be in [0,1], got {}",
            report.confidence_score
        );
        assert!(!report.analysis_summary.is_empty());
    }

    // -------------------------------------------------------------------------
    // TrainingRecipeOptimization
    // -------------------------------------------------------------------------

    #[test]
    fn test_training_recipe_long_duration() {
        let config = make_debug_config();
        let debugger = AutoDebugger::new(&config);
        let metrics: Vec<DashboardMetrics> = vec![];
        let context = DebugContext {
            profiler_report: None,
            gradient_report: None,
            anomaly_report: None,
            recent_metrics: &metrics,
            training_duration: Duration::from_secs(7200), // 2 hours
            model_info: None,
        };
        let report = debugger.analyze_issues(&context).expect("report ok");
        // Should have optimizations recommended for long training
        assert!(
            !report.training_recipe.training_schedule.learning_rate_schedule.is_empty()
        );
    }

    #[test]
    fn test_training_recipe_schedule_fields() {
        let schedule = TrainingSchedule {
            warmup_steps: 500,
            learning_rate_schedule: "linear".to_string(),
            batch_size_schedule: "constant".to_string(),
            early_stopping: true,
            checkpoint_frequency: 500,
        };
        assert!(schedule.early_stopping);
        assert_eq!(schedule.warmup_steps, 500);
    }

    // -------------------------------------------------------------------------
    // Architecture suggestions
    // -------------------------------------------------------------------------

    #[test]
    fn test_architecture_suggestion_construction() {
        let suggestion = ArchitectureSuggestion {
            suggestion_type: "compression".to_string(),
            title: "Compress model".to_string(),
            description: "Use quantization".to_string(),
            impact_assessment: "50% size reduction".to_string(),
            implementation_difficulty: "Medium".to_string(),
        };
        assert_eq!(suggestion.suggestion_type, "compression");
    }

    #[test]
    fn test_architecture_suggestions_for_large_model() {
        let config = make_debug_config();
        let debugger = AutoDebugger::new(&config);
        let model_info = ModelInfo {
            model_type: "gpt".to_string(),
            parameter_count: 200_000_000,
            layer_count: 60,
            architecture_details: HashMap::new(),
        };
        let metrics: Vec<DashboardMetrics> = vec![];
        let context = DebugContext {
            profiler_report: None,
            gradient_report: None,
            anomaly_report: None,
            recent_metrics: &metrics,
            training_duration: Duration::from_secs(60),
            model_info: Some(&model_info),
        };
        let report = debugger.analyze_issues(&context).expect("report ok");
        // Large model + many layers should trigger architecture suggestions
        assert!(
            !report.architecture_suggestions.is_empty(),
            "Should have architecture suggestions for large model with many layers"
        );
    }

    // -------------------------------------------------------------------------
    // Hyperparameter recommendation
    // -------------------------------------------------------------------------

    #[test]
    fn test_hyperparameter_recommendation_fields() {
        let recommendation = HyperparameterRecommendation {
            parameter: "learning_rate".to_string(),
            current_value: Some(0.01),
            recommended_value: 0.001,
            reason: "Too high".to_string(),
            confidence: 0.8,
        };
        assert_eq!(recommendation.parameter, "learning_rate");
        assert!(recommendation.current_value.is_some());
    }

    // -------------------------------------------------------------------------
    // DataStrategy
    // -------------------------------------------------------------------------

    #[test]
    fn test_data_strategy_construction() {
        let strategy = DataStrategy {
            data_augmentation: vec!["flip".to_string(), "crop".to_string()],
            sampling_strategy: "balanced".to_string(),
            preprocessing_optimizations: vec!["normalize".to_string()],
        };
        assert_eq!(strategy.data_augmentation.len(), 2);
    }

    // -------------------------------------------------------------------------
    // No-issue scenario
    // -------------------------------------------------------------------------

    #[test]
    fn test_no_issues_summary() {
        let config = make_debug_config();
        let debugger = AutoDebugger::new(&config);
        let metrics: Vec<DashboardMetrics> = vec![];
        let context = empty_context(&metrics);
        let report = debugger.analyze_issues(&context).expect("report ok");
        assert!(
            report.detected_issues.is_empty(),
            "Should have no detected issues for empty context"
        );
        assert!(
            report.analysis_summary.contains("No significant") || !report.analysis_summary.is_empty()
        );
    }
}
