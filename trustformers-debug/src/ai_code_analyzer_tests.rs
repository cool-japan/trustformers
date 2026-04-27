#[cfg(test)]
mod extended_tests {
    use crate::ai_code_analyzer::*;
    use std::collections::HashMap;

    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }
        fn next_f64(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    // Test 1: AIAnalysisConfig default
    #[test]
    fn test_ai_analysis_config_default() {
        let config = AIAnalysisConfig::default();
        assert!(config.enable_deep_analysis);
        assert!(config.enable_pattern_recognition);
        assert!(config.enable_optimization_suggestions);
        assert!(config.enable_vulnerability_detection);
        assert!(config.enable_performance_prediction);
        assert_eq!(config.max_analysis_time_secs, 30);
        assert!((config.confidence_threshold - 0.75).abs() < f64::EPSILON);
        assert!(config.enable_caching);
        assert_eq!(config.cache_expiration_hours, 24);
    }

    // Test 2: AICodeAnalyzer creation
    #[test]
    fn test_ai_code_analyzer_creation() {
        let config = AIAnalysisConfig::default();
        let analyzer = AICodeAnalyzer::new(config);
        let metrics = analyzer.get_performance_metrics();
        assert_eq!(metrics.total_analyses, 0);
        assert_eq!(metrics.cached_results, 0);
        assert!((metrics.cache_hit_rate - 0.0).abs() < f64::EPSILON);
    }

    // Test 3: CodeAnalysisResult construction
    #[test]
    fn test_code_analysis_result_construction() {
        let result = CodeAnalysisResult {
            quality_score: 85.0,
            detected_patterns: Vec::new(),
            identified_issues: Vec::new(),
            optimization_suggestions: Vec::new(),
            security_issues: Vec::new(),
            performance_predictions: PerformancePredictions::new(),
            analysis_metadata: AnalysisMetadata::default(),
        };
        assert!((result.quality_score - 85.0).abs() < f64::EPSILON);
        assert!(result.detected_patterns.is_empty());
    }

    // Test 4: DetectedPattern construction
    #[test]
    fn test_detected_pattern_construction() {
        let pattern = DetectedPattern {
            pattern_type: PatternType::GoodPattern,
            name: "Gradient Clipping".to_string(),
            description: "Proper gradient clipping detected".to_string(),
            severity: Severity::Info,
            confidence: 0.9,
            recommendations: vec!["Consider adaptive clipping".to_string()],
        };
        assert_eq!(pattern.pattern_type, PatternType::GoodPattern);
        assert!((pattern.confidence - 0.9).abs() < f64::EPSILON);
        assert_eq!(pattern.severity, Severity::Info);
    }

    // Test 5: PatternType variants
    #[test]
    fn test_pattern_type_variants() {
        assert_eq!(PatternType::GoodPattern, PatternType::GoodPattern);
        assert_ne!(PatternType::GoodPattern, PatternType::AntiPattern);
        assert_ne!(
            PatternType::OptimizationOpportunity,
            PatternType::SecurityConcern
        );
    }

    // Test 6: Severity variants
    #[test]
    fn test_severity_variants() {
        let severities = [
            Severity::Critical,
            Severity::High,
            Severity::Medium,
            Severity::Low,
            Severity::Info,
        ];
        assert_eq!(severities.len(), 5);
        assert_eq!(Severity::Critical, Severity::Critical);
        assert_ne!(Severity::Critical, Severity::Info);
    }

    // Test 7: IdentifiedIssue construction
    #[test]
    fn test_identified_issue_construction() {
        let issue = IdentifiedIssue {
            issue_type: IssueType::NumericalStability,
            title: "Log-Softmax Instability".to_string(),
            description: "Using log(softmax(x)) can be unstable".to_string(),
            severity: Severity::High,
            confidence: 0.88,
            suggested_fix: "Use log_softmax directly".to_string(),
            code_location: Some(CodeLocation {
                file: "model.rs".to_string(),
                line: 42,
                column: 10,
            }),
        };
        assert!(issue.code_location.is_some());
        if let Some(ref loc) = issue.code_location {
            assert_eq!(loc.line, 42);
        }
    }

    // Test 8: IssueType variants
    #[test]
    fn test_issue_type_variants() {
        let types: Vec<IssueType> = vec![
            IssueType::NumericalStability,
            IssueType::Performance,
            IssueType::MemoryLeak,
            IssueType::LogicError,
            IssueType::TypeMismatch,
            IssueType::ResourceLeak,
        ];
        assert_eq!(types.len(), 6);
    }

    // Test 9: OptimizationSuggestion construction
    #[test]
    fn test_optimization_suggestion_construction() {
        let suggestion = OptimizationSuggestion {
            optimization_type: OptimizationType::MixedPrecision,
            title: "Enable Mixed Precision".to_string(),
            description: "Mixed precision training speeds up training".to_string(),
            potential_speedup: 1.5,
            memory_savings: 0.4,
            implementation_effort: ImplementationEffort::Low,
            confidence: 0.9,
            code_example: Some("with torch.autocast(...)".to_string()),
        };
        assert!((suggestion.potential_speedup - 1.5).abs() < f64::EPSILON);
        assert!(suggestion.code_example.is_some());
    }

    // Test 10: OptimizationType variants
    #[test]
    fn test_optimization_type_variants() {
        let types = [
            OptimizationType::MixedPrecision,
            OptimizationType::ModelCompilation,
            OptimizationType::MemoryOptimization,
            OptimizationType::ComputationOptimization,
            OptimizationType::IOOptimization,
            OptimizationType::ParallelizationOptimization,
        ];
        assert_eq!(types.len(), 6);
    }

    // Test 11: SecurityIssue construction
    #[test]
    fn test_security_issue_construction() {
        let issue = SecurityIssue {
            vulnerability_type: VulnerabilityType::CodeExecution,
            title: "Unsafe Pickle Loading".to_string(),
            description: "pickle.load can execute arbitrary code".to_string(),
            severity: Severity::Critical,
            confidence: 0.95,
            mitigation: "Use safe alternatives".to_string(),
            cve_references: vec!["CWE-502".to_string()],
        };
        assert_eq!(issue.vulnerability_type, VulnerabilityType::CodeExecution);
        assert_eq!(issue.severity, Severity::Critical);
        assert_eq!(issue.cve_references.len(), 1);
    }

    // Test 12: VulnerabilityType variants
    #[test]
    fn test_vulnerability_type_variants() {
        let types = [
            VulnerabilityType::CodeExecution,
            VulnerabilityType::DataExposure,
            VulnerabilityType::InputValidation,
            VulnerabilityType::AuthenticationBypass,
            VulnerabilityType::PrivilegeEscalation,
        ];
        assert_eq!(types.len(), 5);
        assert_eq!(
            VulnerabilityType::CodeExecution,
            VulnerabilityType::CodeExecution
        );
    }

    // Test 13: PerformancePredictions creation
    #[test]
    fn test_performance_predictions_creation() {
        let predictions = PerformancePredictions::new();
        assert!((predictions.estimated_memory_usage - 0.0).abs() < f64::EPSILON);
        assert!((predictions.estimated_training_time - 0.0).abs() < f64::EPSILON);
        assert!((predictions.estimated_inference_latency - 0.0).abs() < f64::EPSILON);
        assert!(predictions.predicted_bottlenecks.is_empty());
        assert!((predictions.confidence_score - 0.0).abs() < f64::EPSILON);
    }

    // Test 14: ScalingCharacteristics default
    #[test]
    fn test_scaling_characteristics_default() {
        let scaling = ScalingCharacteristics::default();
        let debug_str = format!("{:?}", scaling.batch_size_scaling);
        assert!(debug_str.contains("Linear"));
    }

    // Test 15: ScalingBehavior variants
    #[test]
    fn test_scaling_behavior_variants() {
        let behaviors = [
            ScalingBehavior::Constant,
            ScalingBehavior::Linear,
            ScalingBehavior::Quadratic,
            ScalingBehavior::Exponential,
            ScalingBehavior::Sublinear,
        ];
        assert_eq!(behaviors.len(), 5);
    }

    // Test 16: AnalysisMetadata default
    #[test]
    fn test_analysis_metadata_default() {
        let metadata = AnalysisMetadata::default();
        assert!((metadata.confidence_score - 0.0).abs() < f64::EPSILON);
        assert_eq!(metadata.analyzer_version, "1.0.0");
    }

    // Test 17: TensorOperation default
    #[test]
    fn test_tensor_operation_default() {
        let op = TensorOperation::default();
        assert!(op.name.is_empty());
        assert!(op.inputs.is_empty());
        assert!(op.outputs.is_empty());
        assert!(!op.is_inplace);
        assert_eq!(op.output_size_bytes, 0);
    }

    // Test 18: ModelContext construction
    #[test]
    fn test_model_context_construction() {
        let context = ModelContext {
            model_type: ModelType::Production,
            model_size: 7_000_000_000,
            framework: "trustformers".to_string(),
            target_hardware: "gpu".to_string(),
            training_stage: TrainingStage::Inference,
        };
        assert_eq!(context.model_type, ModelType::Production);
        assert!(context.model_size > 1_000_000_000);
    }

    // Test 19: ModelType equality
    #[test]
    fn test_model_type_equality() {
        assert_eq!(ModelType::Training, ModelType::Training);
        assert_ne!(ModelType::Training, ModelType::Production);
        assert_ne!(ModelType::Inference, ModelType::Development);
    }

    // Test 20: ImplementationEffort variants
    #[test]
    fn test_implementation_effort_variants() {
        let efforts = [
            ImplementationEffort::Low,
            ImplementationEffort::Medium,
            ImplementationEffort::High,
        ];
        assert_eq!(efforts.len(), 3);
    }

    // Test 21: Quality score calculation logic
    #[test]
    fn test_quality_score_calculation() {
        let config = AIAnalysisConfig::default();
        let analyzer = AICodeAnalyzer::new(config);

        let mut result = CodeAnalysisResult {
            quality_score: 0.0,
            detected_patterns: vec![DetectedPattern {
                pattern_type: PatternType::GoodPattern,
                name: "Test".to_string(),
                description: "Test".to_string(),
                severity: Severity::Info,
                confidence: 0.9,
                recommendations: vec![],
            }],
            identified_issues: vec![IdentifiedIssue {
                issue_type: IssueType::Performance,
                title: "Slow".to_string(),
                description: "Slow op".to_string(),
                severity: Severity::Medium,
                confidence: 0.8,
                suggested_fix: "Fix it".to_string(),
                code_location: None,
            }],
            optimization_suggestions: Vec::new(),
            security_issues: Vec::new(),
            performance_predictions: PerformancePredictions::new(),
            analysis_metadata: AnalysisMetadata::default(),
        };
        result.quality_score = analyzer.calculate_quality_score(&result);
        // Should be 100 - 5 (medium issue) + 2 (good pattern) = 97
        assert!((result.quality_score - 97.0).abs() < 0.001);
    }

    // Test 22: Confidence score calculation
    #[test]
    fn test_confidence_score_calculation() {
        let config = AIAnalysisConfig::default();
        let analyzer = AICodeAnalyzer::new(config);

        let result = CodeAnalysisResult {
            quality_score: 80.0,
            detected_patterns: vec![DetectedPattern {
                pattern_type: PatternType::GoodPattern,
                name: "Test".to_string(),
                description: "Test".to_string(),
                severity: Severity::Info,
                confidence: 0.8,
                recommendations: vec![],
            }],
            identified_issues: vec![IdentifiedIssue {
                issue_type: IssueType::Performance,
                title: "Slow".to_string(),
                description: "Slow op".to_string(),
                severity: Severity::Medium,
                confidence: 0.6,
                suggested_fix: "Fix it".to_string(),
                code_location: None,
            }],
            optimization_suggestions: Vec::new(),
            security_issues: Vec::new(),
            performance_predictions: PerformancePredictions::new(),
            analysis_metadata: AnalysisMetadata::default(),
        };
        let confidence = analyzer.calculate_confidence_score(&result);
        // (0.8 + 0.6) / 2 = 0.7
        assert!((confidence - 0.7).abs() < 0.001);
    }

    // Test 23: Performance monitor metrics
    #[test]
    fn test_performance_monitor_metrics() {
        let config = AIAnalysisConfig::default();
        let analyzer = AICodeAnalyzer::new(config);
        let metrics = analyzer.get_performance_metrics();
        assert_eq!(metrics.total_analyses, 0);
        assert_eq!(metrics.cached_results, 0);
    }

    // Test 24: TensorOperation construction
    #[test]
    fn test_tensor_operation_construction() {
        let op = TensorOperation {
            name: "matmul".to_string(),
            op_type: OperationType::MatMul,
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["C".to_string()],
            parameters: HashMap::new(),
            output_size_bytes: 1024 * 1024,
            is_inplace: false,
        };
        assert_eq!(op.name, "matmul");
        assert_eq!(op.inputs.len(), 2);
        assert_eq!(op.outputs.len(), 1);
    }

    // Test 25: Multiple DetectedPatterns with LCG
    #[test]
    fn test_detected_patterns_with_lcg() {
        let mut lcg = Lcg::new(42);
        let pattern_types = [
            PatternType::GoodPattern,
            PatternType::AntiPattern,
            PatternType::OptimizationOpportunity,
        ];
        let severities = [
            Severity::Info,
            Severity::Low,
            Severity::Medium,
            Severity::High,
        ];

        let patterns: Vec<DetectedPattern> = (0..20)
            .map(|i| {
                let pt_idx = (lcg.next() % pattern_types.len() as u64) as usize;
                let sev_idx = (lcg.next() % severities.len() as u64) as usize;
                DetectedPattern {
                    pattern_type: pattern_types[pt_idx].clone(),
                    name: format!("Pattern_{}", i),
                    description: format!("Description for pattern {}", i),
                    severity: severities[sev_idx].clone(),
                    confidence: lcg.next_f64(),
                    recommendations: vec![],
                }
            })
            .collect();
        assert_eq!(patterns.len(), 20);
        for pattern in &patterns {
            assert!(pattern.confidence >= 0.0 && pattern.confidence <= 1.0);
        }
    }
}
