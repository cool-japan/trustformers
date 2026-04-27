#[cfg(test)]
mod tests {
    use crate::comprehensive_testing::fairness::*;
    use std::collections::HashMap;

    // --- FairnessConfig tests ---

    #[test]
    fn test_fairness_config_default() {
        let config = FairnessConfig::default();
        assert!(!config.protected_attributes.is_empty());
        assert!(!config.fairness_metrics.is_empty());
        assert!(!config.mitigation_strategies.is_empty());
        assert!((config.bias_threshold - 0.05).abs() < f32::EPSILON);
        assert!(config.test_intersectional);
    }

    #[test]
    fn test_fairness_config_default_attributes() {
        let config = FairnessConfig::default();
        assert!(config.protected_attributes.contains(&"gender".to_string()));
        assert!(config.protected_attributes.contains(&"race".to_string()));
        assert!(config.protected_attributes.contains(&"age".to_string()));
    }

    #[test]
    fn test_fairness_config_default_sample_size() {
        let config = FairnessConfig::default();
        assert_eq!(config.sample_size, 10000);
    }

    #[test]
    fn test_fairness_config_confidence_level() {
        let config = FairnessConfig::default();
        assert!((config.confidence_level - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fairness_config_custom() {
        let config = FairnessConfig {
            protected_attributes: vec!["gender".to_string()],
            fairness_metrics: vec![FairnessMetricType::DemographicParity],
            mitigation_strategies: vec![BiasmitigationStrategy::Preprocessing],
            bias_threshold: 0.1,
            test_intersectional: false,
            sample_size: 5000,
            confidence_level: 0.99,
        };
        assert_eq!(config.protected_attributes.len(), 1);
        assert!(!config.test_intersectional);
        assert_eq!(config.sample_size, 5000);
    }

    // --- FairnessAssessment tests ---

    #[test]
    fn test_fairness_assessment_new() {
        let assessment = FairnessAssessment::new();
        assert!(assessment.bias_metrics.is_empty());
        assert!(assessment.results.is_empty());
    }

    #[test]
    fn test_fairness_assessment_with_config() {
        let config = FairnessConfig {
            bias_threshold: 0.2,
            ..FairnessConfig::default()
        };
        let assessment = FairnessAssessment::with_config(config);
        assert!((assessment.config.bias_threshold - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fairness_assessment_default_config() {
        let assessment = FairnessAssessment::new();
        assert!(!assessment.config.protected_attributes.is_empty());
    }

    // --- FairnessMetricType tests ---

    #[test]
    fn test_fairness_metric_types() {
        let types = vec![
            FairnessMetricType::DemographicParity,
            FairnessMetricType::EqualOpportunity,
            FairnessMetricType::EqualizeDOdds,
            FairnessMetricType::CalibrationMetrics,
            FairnessMetricType::IndividualFairness,
            FairnessMetricType::CounterfactualFairness,
            FairnessMetricType::TreatmentEquality,
            FairnessMetricType::ConditionalUseAccuracyEquality,
        ];
        assert_eq!(types.len(), 8);
    }

    // --- BiasmitigationStrategy tests ---

    #[test]
    fn test_bias_mitigation_strategies() {
        let strategies = vec![
            BiasmitigationStrategy::Preprocessing,
            BiasmitigationStrategy::InProcessing,
            BiasmitigationStrategy::Postprocessing,
            BiasmitigationStrategy::AdversarialDebiasing,
            BiasmitigationStrategy::FairRepresentation,
        ];
        assert_eq!(strategies.len(), 5);
    }

    // --- BiasMetric tests ---

    #[test]
    fn test_bias_metric_below_threshold() {
        let metric = BiasMetric {
            name: "demographic_parity".to_string(),
            metric_type: FairnessMetricType::DemographicParity,
            protected_attribute: "gender".to_string(),
            bias_value: 0.02,
            p_value: Some(0.03),
            confidence_interval: Some((0.01, 0.04)),
            exceeds_threshold: false,
        };
        assert!(!metric.exceeds_threshold);
        assert!(metric.p_value.is_some());
    }

    #[test]
    fn test_bias_metric_above_threshold() {
        let metric = BiasMetric {
            name: "equal_opportunity".to_string(),
            metric_type: FairnessMetricType::EqualOpportunity,
            protected_attribute: "race".to_string(),
            bias_value: 0.15,
            p_value: Some(0.001),
            confidence_interval: Some((0.1, 0.2)),
            exceeds_threshold: true,
        };
        assert!(metric.exceeds_threshold);
    }

    #[test]
    fn test_bias_metric_no_p_value() {
        let metric = BiasMetric {
            name: "test".to_string(),
            metric_type: FairnessMetricType::IndividualFairness,
            protected_attribute: "age".to_string(),
            bias_value: 0.05,
            p_value: None,
            confidence_interval: None,
            exceeds_threshold: false,
        };
        assert!(metric.p_value.is_none());
        assert!(metric.confidence_interval.is_none());
    }

    // --- FairnessResult tests ---

    #[test]
    fn test_fairness_result_creation() {
        let result = FairnessResult {
            overall_fairness_score: 0.85,
            bias_metrics: HashMap::new(),
            intersectional_bias: None,
            mitigation_recommendations: vec!["reduce bias".to_string()],
            statistical_tests: Vec::new(),
            violations: Vec::new(),
        };
        assert!((result.overall_fairness_score - 0.85).abs() < f32::EPSILON);
        assert!(result.intersectional_bias.is_none());
    }

    #[test]
    fn test_fairness_result_with_violations() {
        let violations = vec![FairnessViolation {
            violation_type: "DemographicParity".to_string(),
            severity: "high".to_string(),
            description: "Significant bias".to_string(),
            affected_groups: vec!["group_a".to_string(), "group_b".to_string()],
            recommendations: vec!["mitigation needed".to_string()],
        }];
        let result = FairnessResult {
            overall_fairness_score: 0.3,
            bias_metrics: HashMap::new(),
            intersectional_bias: None,
            mitigation_recommendations: Vec::new(),
            statistical_tests: Vec::new(),
            violations,
        };
        assert_eq!(result.violations.len(), 1);
    }

    // --- StatisticalTest tests ---

    #[test]
    fn test_statistical_test_significant() {
        let test = StatisticalTest {
            test_name: "chi_square".to_string(),
            statistic: 15.0,
            p_value: 0.001,
            critical_value: 3.84,
            is_significant: true,
            degrees_of_freedom: Some(1),
        };
        assert!(test.is_significant);
        assert!(test.degrees_of_freedom.is_some());
    }

    #[test]
    fn test_statistical_test_not_significant() {
        let test = StatisticalTest {
            test_name: "t_test".to_string(),
            statistic: 0.5,
            p_value: 0.6,
            critical_value: 1.96,
            is_significant: false,
            degrees_of_freedom: Some(50),
        };
        assert!(!test.is_significant);
    }

    // --- FairnessViolation tests ---

    #[test]
    fn test_fairness_violation_creation() {
        let violation = FairnessViolation {
            violation_type: "EqualOpportunity".to_string(),
            severity: "medium".to_string(),
            description: "Disparity in TPR".to_string(),
            affected_groups: vec!["male".to_string(), "female".to_string()],
            recommendations: vec!["apply reweighting".to_string()],
        };
        assert_eq!(violation.severity, "medium");
        assert_eq!(violation.affected_groups.len(), 2);
    }

    // --- GroupData tests ---

    #[test]
    fn test_group_data_creation() {
        let group = GroupData {
            inputs: Vec::new(),
            labels: vec![0, 1, 1, 0],
            metadata: HashMap::new(),
        };
        assert_eq!(group.labels.len(), 4);
    }

    #[test]
    fn test_group_data_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "dataset_a".to_string());
        let group = GroupData {
            inputs: Vec::new(),
            labels: vec![1],
            metadata,
        };
        assert!(group.metadata.contains_key("source"));
    }

    // --- FairnessTestData tests ---

    #[test]
    fn test_fairness_test_data_creation() {
        let data = FairnessTestData {
            grouped_data: HashMap::new(),
            intersectional_data: HashMap::new(),
        };
        assert!(data.grouped_data.is_empty());
        assert!(data.intersectional_data.is_empty());
    }
}
