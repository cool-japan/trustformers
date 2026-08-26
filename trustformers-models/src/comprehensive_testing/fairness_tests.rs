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

    // ------------------------------------------------------------------
    // Regression tests: the audit must measure the model, not return 0.02
    // ------------------------------------------------------------------

    use serde::{Deserialize, Serialize};
    use std::io::Read;
    use trustformers_core::tensor::Tensor;
    use trustformers_core::traits::{Config, Model};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct ScoreConfig;

    impl Config for ScoreConfig {
        fn architecture(&self) -> &'static str {
            "score"
        }
    }

    /// Deterministic classifier: the input's first value *is* the probability of
    /// the positive class, so a test can dictate exactly what the model predicts.
    struct ScoreModel {
        config: ScoreConfig,
    }

    impl ScoreModel {
        fn new() -> Self {
            Self {
                config: ScoreConfig,
            }
        }
    }

    impl Model for ScoreModel {
        type Config = ScoreConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, input: Tensor) -> trustformers_core::Result<Tensor> {
            let values = input.data()?;
            let probability = values.first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
            Tensor::from_slice(&[1.0 - probability, probability], &[2])
        }

        fn load_pretrained(&mut self, _reader: &mut dyn Read) -> trustformers_core::Result<()> {
            Ok(())
        }

        fn get_config(&self) -> &ScoreConfig {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            0
        }
    }

    fn scored_group(scores: &[f32], labels: &[i32]) -> GroupData {
        GroupData {
            inputs: scores
                .iter()
                .map(|&score| Tensor::from_slice(&[score], &[1]).expect("input tensor"))
                .collect(),
            labels: labels.to_vec(),
            metadata: HashMap::new(),
        }
    }

    fn two_group_data(
        attribute: &str,
        a: (&[f32], &[i32]),
        b: (&[f32], &[i32]),
    ) -> FairnessTestData {
        let mut groups = HashMap::new();
        groups.insert("a".to_string(), scored_group(a.0, a.1));
        groups.insert("b".to_string(), scored_group(b.0, b.1));

        let mut grouped_data = HashMap::new();
        grouped_data.insert(attribute.to_string(), groups);

        FairnessTestData {
            grouped_data,
            intersectional_data: HashMap::new(),
        }
    }

    fn single_attribute_config(metrics: Vec<FairnessMetricType>) -> FairnessConfig {
        FairnessConfig {
            protected_attributes: vec!["gender".to_string()],
            fairness_metrics: metrics,
            test_intersectional: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_demographic_parity_measures_the_real_gap() {
        // Group "a" is always accepted, group "b" always rejected.
        let data = two_group_data("gender", (&[0.9; 20], &[1; 20]), (&[0.1; 20], &[1; 20]));
        let mut assessment = FairnessAssessment::with_config(single_attribute_config(vec![
            FairnessMetricType::DemographicParity,
        ]));

        let result = assessment
            .evaluate_fairness(&ScoreModel::new(), &data)
            .expect("fairness evaluation");

        let metric = &result.bias_metrics["gender"][0];
        assert!(
            (metric.bias_value - 1.0).abs() < 1e-6,
            "a total accept/reject split is a parity gap of 1.0, got {}",
            metric.bias_value
        );
        assert!(
            metric.exceeds_threshold,
            "a gap of 1.0 must exceed the 5% threshold"
        );
        let p_value = metric.p_value.expect("a real p-value");
        assert!(
            p_value < 0.001,
            "a total split must be significant, got {p_value}"
        );
        assert!(
            !result.violations.is_empty(),
            "the violation must be reported"
        );
        assert!(
            result.overall_fairness_score < 0.5,
            "the fairness score must collapse, got {}",
            result.overall_fairness_score
        );
    }

    #[test]
    fn test_demographic_parity_reports_no_gap_for_identical_groups() {
        let data = two_group_data(
            "gender",
            (&[0.9, 0.1, 0.9, 0.1], &[1, 0, 1, 0]),
            (&[0.9, 0.1, 0.9, 0.1], &[1, 0, 1, 0]),
        );
        let mut assessment = FairnessAssessment::with_config(single_attribute_config(vec![
            FairnessMetricType::DemographicParity,
        ]));

        let result = assessment
            .evaluate_fairness(&ScoreModel::new(), &data)
            .expect("fairness evaluation");
        let metric = &result.bias_metrics["gender"][0];

        assert!(metric.bias_value.abs() < 1e-6, "got {}", metric.bias_value);
        assert!(!metric.exceeds_threshold);
        // Identical groups: the two-proportion test cannot reject anything.
        assert!(metric.p_value.expect("p-value") > 0.99);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_equal_opportunity_conditions_on_the_labels() {
        // Both groups get the same *predictions*, but only group "b" has its
        // positives among the rejected examples -> the TPR gap is real.
        let data = two_group_data(
            "gender",
            (&[0.9, 0.9, 0.1, 0.1], &[1, 1, 0, 0]),
            (&[0.9, 0.9, 0.1, 0.1], &[0, 0, 1, 1]),
        );
        let mut assessment = FairnessAssessment::with_config(single_attribute_config(vec![
            FairnessMetricType::DemographicParity,
            FairnessMetricType::EqualOpportunity,
        ]));

        let result = assessment
            .evaluate_fairness(&ScoreModel::new(), &data)
            .expect("fairness evaluation");

        let parity = &result.bias_metrics["gender"][0];
        let opportunity = &result.bias_metrics["gender"][1];

        assert!(
            parity.bias_value.abs() < 1e-6,
            "the positive rates are identical, so parity sees no gap"
        );
        assert!(
            (opportunity.bias_value - 1.0).abs() < 1e-6,
            "TPR is 1.0 for group a and 0.0 for group b, got {}",
            opportunity.bias_value
        );
        assert!(opportunity.exceeds_threshold);
    }

    #[test]
    fn test_equalized_odds_takes_the_larger_of_tpr_and_fpr_gaps() {
        // Identical TPR (1.0 in both groups) but very different FPR.
        let data = two_group_data(
            "gender",
            (&[0.9, 0.9, 0.1, 0.1], &[1, 1, 0, 0]),
            (&[0.9, 0.9, 0.9, 0.9], &[1, 1, 0, 0]),
        );
        let mut assessment = FairnessAssessment::with_config(single_attribute_config(vec![
            FairnessMetricType::EqualizeDOdds,
        ]));

        let result = assessment
            .evaluate_fairness(&ScoreModel::new(), &data)
            .expect("fairness evaluation");
        let metric = &result.bias_metrics["gender"][0];

        // TPR gap = 0, FPR gap = 1.0 -> equalized odds reports 1.0.
        assert!(
            (metric.bias_value - 1.0).abs() < 1e-6,
            "got {}",
            metric.bias_value
        );
        assert!(metric.exceeds_threshold);
    }

    #[test]
    fn test_calibration_gap_is_measured_not_assumed() {
        // Group "a" is perfectly calibrated, group "b" is confidently wrong.
        let data = two_group_data(
            "gender",
            (&[1.0, 1.0, 0.0, 0.0], &[1, 1, 0, 0]),
            (&[1.0, 1.0, 0.0, 0.0], &[0, 0, 1, 1]),
        );
        let mut assessment = FairnessAssessment::with_config(single_attribute_config(vec![
            FairnessMetricType::CalibrationMetrics,
        ]));

        let result = assessment
            .evaluate_fairness(&ScoreModel::new(), &data)
            .expect("fairness evaluation");
        let metric = &result.bias_metrics["gender"][0];

        assert!(
            (metric.bias_value - 1.0).abs() < 1e-6,
            "ECE is 0.0 for group a and 1.0 for group b, got {}",
            metric.bias_value
        );
        // There is no closed-form test for a difference of calibration errors.
        assert!(metric.p_value.is_none());
        assert!(metric.confidence_interval.is_none());
    }

    #[test]
    fn test_statistical_tests_are_computed_from_the_predictions() {
        let data = two_group_data("gender", (&[0.9; 30], &[1; 30]), (&[0.1; 30], &[1; 30]));
        let mut assessment = FairnessAssessment::with_config(single_attribute_config(vec![
            FairnessMetricType::DemographicParity,
        ]));

        let result = assessment
            .evaluate_fairness(&ScoreModel::new(), &data)
            .expect("fairness evaluation");

        let test = result
            .statistical_tests
            .iter()
            .find(|test| test.test_name.contains("gender"))
            .expect("a chi-square test for the gender attribute");

        // 30/0 vs 0/30 with equal margins: X^2 = n = 60, df = 1.
        assert_eq!(test.degrees_of_freedom, Some(1));
        assert!(
            (test.statistic - 60.0).abs() < 1e-3,
            "expected the real chi-square statistic, got {}",
            test.statistic
        );
        assert!(test.p_value < 1e-6, "got {}", test.p_value);
        assert!(test.is_significant);
        assert!(
            (test.critical_value - 3.841).abs() < 1e-2,
            "the 5% critical value with df=1 is 3.841, got {}",
            test.critical_value
        );
    }

    #[test]
    fn test_statistical_tests_are_not_significant_without_a_gap() {
        let data = two_group_data(
            "gender",
            (&[0.9, 0.1, 0.9, 0.1], &[1, 0, 1, 0]),
            (&[0.9, 0.1, 0.9, 0.1], &[1, 0, 1, 0]),
        );
        let mut assessment = FairnessAssessment::with_config(single_attribute_config(vec![
            FairnessMetricType::DemographicParity,
        ]));

        let result = assessment
            .evaluate_fairness(&ScoreModel::new(), &data)
            .expect("fairness evaluation");
        let test = result
            .statistical_tests
            .first()
            .expect("a chi-square test for the gender attribute");

        assert!(test.statistic.abs() < 1e-6, "got {}", test.statistic);
        assert!((test.p_value - 1.0).abs() < 1e-6, "got {}", test.p_value);
        assert!(!test.is_significant);
    }

    #[test]
    fn test_unimplemented_metric_is_an_error_not_a_clean_bill_of_health() {
        let data = two_group_data("gender", (&[0.9; 4], &[1; 4]), (&[0.1; 4], &[1; 4]));
        let mut assessment = FairnessAssessment::with_config(single_attribute_config(vec![
            FairnessMetricType::IndividualFairness,
        ]));

        let error = assessment
            .evaluate_fairness(&ScoreModel::new(), &data)
            .expect_err("an unimplemented metric must not report 'no bias'");
        assert!(error.to_string().contains("not implemented"), "{error}");
    }

    #[test]
    fn test_missing_groups_are_an_error() {
        let mut groups = HashMap::new();
        groups.insert("only".to_string(), scored_group(&[0.9, 0.1], &[1, 0]));
        let mut grouped_data = HashMap::new();
        grouped_data.insert("gender".to_string(), groups);
        let data = FairnessTestData {
            grouped_data,
            intersectional_data: HashMap::new(),
        };

        let mut assessment = FairnessAssessment::with_config(single_attribute_config(vec![
            FairnessMetricType::DemographicParity,
        ]));
        assert!(assessment.evaluate_fairness(&ScoreModel::new(), &data).is_err());
    }

    #[test]
    fn test_intersectional_bias_uses_the_supplied_cells() {
        let mut data = two_group_data("gender", (&[0.9, 0.9], &[1, 1]), (&[0.1, 0.1], &[1, 1]));
        data.intersectional_data.insert(
            "gender:a+race:x".to_string(),
            scored_group(&[0.9, 0.9], &[1, 1]),
        );
        data.intersectional_data.insert(
            "gender:b+race:x".to_string(),
            scored_group(&[0.1, 0.1], &[1, 1]),
        );

        let mut config = single_attribute_config(vec![FairnessMetricType::DemographicParity]);
        config.test_intersectional = true;
        let mut assessment = FairnessAssessment::with_config(config);

        let result = assessment
            .evaluate_fairness(&ScoreModel::new(), &data)
            .expect("fairness evaluation");
        let intersectional = result.intersectional_bias.expect("intersectional analysis");

        assert_eq!(intersectional.len(), 1);
        let gap = intersectional["gender+race"];
        assert!((gap - 1.0).abs() < 1e-6, "got {gap}");
    }
}
