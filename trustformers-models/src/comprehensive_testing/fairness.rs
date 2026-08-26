//! Fairness assessment framework for model evaluation

use super::stats::{
    chi_square_quantile, chi_square_test, expected_calibration_error, two_proportion_z_test,
};
use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Model;

/// Fairness assessment framework for model evaluation
pub struct FairnessAssessment {
    /// Configuration for fairness tests
    pub config: FairnessConfig,
    /// Bias detection metrics
    pub bias_metrics: Vec<BiasMetric>,
    /// Fairness evaluation results
    pub results: Vec<FairnessResult>,
}

/// Configuration for fairness assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessConfig {
    /// Protected attributes to test for bias
    pub protected_attributes: Vec<String>,
    /// Fairness metrics to compute
    pub fairness_metrics: Vec<FairnessMetricType>,
    /// Bias mitigation strategies to test
    pub mitigation_strategies: Vec<BiasmitigationStrategy>,
    /// Threshold for acceptable bias levels
    pub bias_threshold: f32,
    /// Whether to test intersectional bias
    pub test_intersectional: bool,
    /// Sample size for statistical tests
    pub sample_size: usize,
    /// Confidence level for statistical tests
    pub confidence_level: f32,
}

impl Default for FairnessConfig {
    fn default() -> Self {
        Self {
            protected_attributes: vec![
                "gender".to_string(),
                "race".to_string(),
                "age".to_string(),
                "religion".to_string(),
                "nationality".to_string(),
            ],
            fairness_metrics: vec![
                FairnessMetricType::DemographicParity,
                FairnessMetricType::EqualOpportunity,
                FairnessMetricType::EqualizeDOdds,
                FairnessMetricType::CalibrationMetrics,
            ],
            mitigation_strategies: vec![
                BiasmitigationStrategy::Preprocessing,
                BiasmitigationStrategy::InProcessing,
                BiasmitigationStrategy::Postprocessing,
            ],
            bias_threshold: 0.05, // 5% threshold
            test_intersectional: true,
            sample_size: 10000,
            confidence_level: 0.95,
        }
    }
}

/// Types of fairness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FairnessMetricType {
    /// Demographic parity (equal positive prediction rates)
    DemographicParity,
    /// Equal opportunity (equal true positive rates)
    EqualOpportunity,
    /// Equalized odds (equal TPR and FPR)
    EqualizeDOdds,
    /// Calibration metrics (equal positive predictive value)
    CalibrationMetrics,
    /// Individual fairness (similar individuals treated similarly)
    IndividualFairness,
    /// Counterfactual fairness
    CounterfactualFairness,
    /// Treatment equality
    TreatmentEquality,
    /// Conditional use accuracy equality
    ConditionalUseAccuracyEquality,
}

/// Bias mitigation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BiasmitigationStrategy {
    /// Data preprocessing techniques
    Preprocessing,
    /// In-processing constraints during training
    InProcessing,
    /// Post-processing output adjustments
    Postprocessing,
    /// Adversarial debiasing
    AdversarialDebiasing,
    /// Fair representation learning
    FairRepresentation,
}

/// Individual bias metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasMetric {
    /// Name of the metric
    pub name: String,
    /// Metric type
    pub metric_type: FairnessMetricType,
    /// Protected attribute being tested
    pub protected_attribute: String,
    /// Computed bias value
    pub bias_value: f32,
    /// Statistical significance
    pub p_value: Option<f32>,
    /// Confidence interval
    pub confidence_interval: Option<(f32, f32)>,
    /// Whether bias exceeds threshold
    pub exceeds_threshold: bool,
}

/// Fairness evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessResult {
    /// Overall fairness score (0-1, higher is more fair)
    pub overall_fairness_score: f32,
    /// Bias metrics by protected attribute
    pub bias_metrics: HashMap<String, Vec<BiasMetric>>,
    /// Intersectional bias analysis
    pub intersectional_bias: Option<HashMap<String, f32>>,
    /// Recommendations for bias mitigation
    pub mitigation_recommendations: Vec<String>,
    /// Statistical test results
    pub statistical_tests: Vec<StatisticalTest>,
    /// Fairness violations detected
    pub violations: Vec<FairnessViolation>,
}

/// Statistical test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    /// Test name
    pub test_name: String,
    /// Test statistic value
    pub statistic: f32,
    /// P-value
    pub p_value: f32,
    /// Critical value
    pub critical_value: f32,
    /// Whether null hypothesis is rejected
    pub is_significant: bool,
    /// Degrees of freedom
    pub degrees_of_freedom: Option<i32>,
}

/// Fairness violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessViolation {
    /// Type of violation
    pub violation_type: String,
    /// Severity level (low, medium, high)
    pub severity: String,
    /// Description of the violation
    pub description: String,
    /// Affected groups
    pub affected_groups: Vec<String>,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Test data structure for fairness evaluation
#[derive(Debug, Clone)]
pub struct FairnessTestData {
    /// Data grouped by protected attributes
    pub grouped_data: HashMap<String, HashMap<String, GroupData>>,
    /// Intersectional data for combinations of attributes
    pub intersectional_data: HashMap<String, GroupData>,
}

/// Data for a specific group
#[derive(Debug, Clone)]
pub struct GroupData {
    /// Input tensors
    pub inputs: Vec<Tensor>,
    /// Ground truth labels
    pub labels: Vec<i32>,
    /// Group metadata
    pub metadata: HashMap<String, String>,
}

impl FairnessAssessment {
    /// Create a new fairness assessment
    pub fn new() -> Self {
        Self {
            config: FairnessConfig::default(),
            bias_metrics: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Create fairness assessment with custom configuration
    pub fn with_config(config: FairnessConfig) -> Self {
        Self {
            config,
            bias_metrics: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Run comprehensive fairness evaluation
    pub fn evaluate_fairness<M: Model<Input = Tensor, Output = Tensor>>(
        &mut self,
        model: &M,
        test_data: &FairnessTestData,
    ) -> Result<FairnessResult> {
        let mut bias_metrics = HashMap::new();
        let mut violations = Vec::new();
        let mut statistical_tests = Vec::new();

        // Evaluate each protected attribute
        for attribute in &self.config.protected_attributes {
            let mut attribute_metrics = Vec::new();

            // Compute each fairness metric
            for metric_type in &self.config.fairness_metrics {
                let metric = self.compute_bias_metric(model, test_data, attribute, metric_type)?;

                if metric.exceeds_threshold {
                    violations.push(FairnessViolation {
                        violation_type: format!("{:?}", metric_type),
                        severity: self.determine_violation_severity(metric.bias_value),
                        description: format!("Bias detected for {} in {}", attribute, metric.name),
                        affected_groups: test_data.get_groups_for_attribute(attribute),
                        recommendations: self.generate_recommendations(metric_type, &metric),
                    });
                }

                attribute_metrics.push(metric);
            }

            bias_metrics.insert(attribute.clone(), attribute_metrics);
        }

        // Perform statistical tests
        statistical_tests.extend(self.perform_statistical_tests(model, test_data)?);

        // Compute intersectional bias if enabled
        let intersectional_bias = if self.config.test_intersectional {
            Some(self.analyze_intersectional_bias(model, test_data)?)
        } else {
            None
        };

        // Compute overall fairness score
        let overall_fairness_score = self.compute_overall_fairness_score(&bias_metrics);

        // Generate mitigation recommendations
        let mitigation_recommendations = self.generate_mitigation_recommendations(&violations);

        let result = FairnessResult {
            overall_fairness_score,
            bias_metrics,
            intersectional_bias,
            mitigation_recommendations,
            statistical_tests,
            violations,
        };

        self.results.push(result.clone());
        Ok(result)
    }

    /// Compute a single bias metric for one protected attribute.
    fn compute_bias_metric<M: Model<Input = Tensor, Output = Tensor>>(
        &self,
        model: &M,
        test_data: &FairnessTestData,
        attribute: &str,
        metric_type: &FairnessMetricType,
    ) -> Result<BiasMetric> {
        let groups = test_data.get_groups_for_attribute(attribute);

        match metric_type {
            FairnessMetricType::DemographicParity => {
                self.compute_demographic_parity(model, test_data, attribute, &groups)
            },
            FairnessMetricType::EqualOpportunity => {
                self.compute_equal_opportunity(model, test_data, attribute, &groups)
            },
            FairnessMetricType::EqualizeDOdds => {
                self.compute_equalized_odds(model, test_data, attribute, &groups)
            },
            FairnessMetricType::CalibrationMetrics => {
                self.compute_calibration_metrics(model, test_data, attribute, &groups)
            },
            // Metrics without an implementation must fail loudly: a fairness
            // audit that quietly reports "no bias" is worse than no audit.
            other => Err(Error::msg(format!(
                "fairness metric {other:?} is not implemented; remove it from \
                 FairnessConfig::fairness_metrics or implement it"
            ))),
        }
    }

    /// Demographic parity: the gap in positive-decision rate across groups.
    ///
    /// Significance comes from a two-proportion z-test between the two extreme
    /// groups — the same pair the reported gap is measured on.
    fn compute_demographic_parity<M: Model<Input = Tensor, Output = Tensor>>(
        &self,
        model: &M,
        test_data: &FairnessTestData,
        attribute: &str,
        groups: &[String],
    ) -> Result<BiasMetric> {
        let outcomes = self.collect_group_outcomes(model, test_data, attribute, groups)?;

        let rates: Vec<MeasuredRate> = outcomes
            .iter()
            .map(|outcome| {
                let (successes, total) = outcome.positive_rate();
                MeasuredRate {
                    group: outcome.group.clone(),
                    rate: successes / total,
                    successes,
                    total,
                }
            })
            .collect();

        self.metric_from_rates(
            "Demographic Parity",
            FairnessMetricType::DemographicParity,
            attribute,
            &rates,
        )
    }
}

/// Decision threshold: a predicted positive-class probability above this counts
/// as a positive decision.
const POSITIVE_DECISION_THRESHOLD: f32 = 0.5;

/// Number of equal-width bins used when measuring calibration error.
const CALIBRATION_BINS: usize = 10;

/// Predictions and ground truth for one protected group.
#[derive(Debug, Clone)]
struct GroupOutcome {
    group: String,
    /// Probability of the positive class for every example in the group.
    predictions: Vec<f32>,
    /// Ground-truth labels (`> 0` means positive).
    labels: Vec<i32>,
}

impl GroupOutcome {
    /// (positive predictions, examples).
    fn positive_rate(&self) -> (f64, f64) {
        let positives =
            self.predictions.iter().filter(|&&p| p > POSITIVE_DECISION_THRESHOLD).count();
        (positives as f64, self.predictions.len() as f64)
    }

    /// (true positives, positive examples), or `None` when the group has no
    /// positive ground-truth examples.
    fn true_positive_rate(&self) -> Option<(f64, f64)> {
        self.conditional_rate(true)
    }

    /// (false positives, negative examples), or `None` when the group has no
    /// negative ground-truth examples.
    fn false_positive_rate(&self) -> Option<(f64, f64)> {
        self.conditional_rate(false)
    }

    fn conditional_rate(&self, positive_label: bool) -> Option<(f64, f64)> {
        let mut hits = 0.0;
        let mut total = 0.0;
        for (&prediction, &label) in self.predictions.iter().zip(self.labels.iter()) {
            if (label > 0) == positive_label {
                total += 1.0;
                if prediction > POSITIVE_DECISION_THRESHOLD {
                    hits += 1.0;
                }
            }
        }
        if total == 0.0 {
            None
        } else {
            Some((hits, total))
        }
    }
}

/// A rate measured on one group, together with the counts behind it.
struct MeasuredRate {
    /// Group the rate was measured on (kept for diagnostics).
    #[allow(dead_code)]
    group: String,
    rate: f64,
    successes: f64,
    total: f64,
}

impl FairnessAssessment {
    /// Run every group of `attribute` through the model and collect the real
    /// predictions and labels.
    fn collect_group_outcomes<M: Model<Input = Tensor, Output = Tensor>>(
        &self,
        model: &M,
        test_data: &FairnessTestData,
        attribute: &str,
        groups: &[String],
    ) -> Result<Vec<GroupOutcome>> {
        if groups.len() < 2 {
            return Err(Error::msg(format!(
                "fairness metrics for `{attribute}` need at least two groups, got {}",
                groups.len()
            )));
        }

        let mut outcomes = Vec::with_capacity(groups.len());
        for group in groups {
            let group_data = test_data.get_group_data(attribute, group)?;
            if group_data.inputs.is_empty() {
                return Err(Error::msg(format!(
                    "group `{attribute}:{group}` carries no examples"
                )));
            }
            if !group_data.labels.is_empty() && group_data.labels.len() != group_data.inputs.len() {
                return Err(Error::msg(format!(
                    "group `{attribute}:{group}` has {} inputs but {} labels",
                    group_data.inputs.len(),
                    group_data.labels.len()
                )));
            }

            outcomes.push(GroupOutcome {
                group: group.clone(),
                predictions: self.get_model_predictions(model, &group_data.inputs)?,
                labels: group_data.labels.clone(),
            });
        }

        Ok(outcomes)
    }

    /// Build a bias metric out of per-group rates: the bias value is the gap
    /// between the extreme groups, and the significance comes from a real
    /// two-proportion z-test between exactly those groups.
    fn metric_from_rates(
        &self,
        name: &str,
        metric_type: FairnessMetricType,
        attribute: &str,
        rates: &[MeasuredRate],
    ) -> Result<BiasMetric> {
        if rates.len() < 2 {
            return Err(Error::msg(format!(
                "`{name}` for `{attribute}` needs at least two comparable groups, got {}",
                rates.len()
            )));
        }

        let highest = rates
            .iter()
            .max_by(|a, b| a.rate.partial_cmp(&b.rate).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| Error::msg("no group rates to compare"))?;
        let lowest = rates
            .iter()
            .min_by(|a, b| a.rate.partial_cmp(&b.rate).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| Error::msg("no group rates to compare"))?;

        let bias_value = (highest.rate - lowest.rate) as f32;

        let test = two_proportion_z_test(
            highest.successes,
            highest.total,
            lowest.successes,
            lowest.total,
            f64::from(self.config.confidence_level),
        )
        .map_err(|e| Error::msg(e.to_string()))?;

        Ok(BiasMetric {
            name: name.to_string(),
            metric_type,
            protected_attribute: attribute.to_string(),
            bias_value,
            p_value: Some(test.p_value as f32),
            confidence_interval: Some((
                test.confidence_interval.0 as f32,
                test.confidence_interval.1 as f32,
            )),
            exceeds_threshold: bias_value > self.config.bias_threshold,
        })
    }

    /// Equal opportunity: the gap in true-positive rate across groups.
    fn compute_equal_opportunity<M: Model<Input = Tensor, Output = Tensor>>(
        &self,
        model: &M,
        test_data: &FairnessTestData,
        attribute: &str,
        groups: &[String],
    ) -> Result<BiasMetric> {
        let outcomes = self.collect_group_outcomes(model, test_data, attribute, groups)?;

        let rates: Vec<MeasuredRate> = outcomes
            .iter()
            .filter_map(|outcome| {
                outcome.true_positive_rate().map(|(successes, total)| MeasuredRate {
                    group: outcome.group.clone(),
                    rate: successes / total,
                    successes,
                    total,
                })
            })
            .collect();

        if rates.len() < 2 {
            return Err(Error::msg(format!(
                "equal opportunity for `{attribute}` needs at least two groups with \
                     positive ground-truth examples; only {} qualify",
                rates.len()
            )));
        }

        self.metric_from_rates(
            "Equal Opportunity",
            FairnessMetricType::EqualOpportunity,
            attribute,
            &rates,
        )
    }

    /// Equalized odds: the larger of the true-positive-rate gap and the
    /// false-positive-rate gap across groups.
    fn compute_equalized_odds<M: Model<Input = Tensor, Output = Tensor>>(
        &self,
        model: &M,
        test_data: &FairnessTestData,
        attribute: &str,
        groups: &[String],
    ) -> Result<BiasMetric> {
        let outcomes = self.collect_group_outcomes(model, test_data, attribute, groups)?;

        let collect = |positive_label: bool| -> Vec<MeasuredRate> {
            outcomes
                .iter()
                .filter_map(|outcome| {
                    let counts = if positive_label {
                        outcome.true_positive_rate()
                    } else {
                        outcome.false_positive_rate()
                    };
                    counts.map(|(successes, total)| MeasuredRate {
                        group: outcome.group.clone(),
                        rate: successes / total,
                        successes,
                        total,
                    })
                })
                .collect()
        };

        let tpr = collect(true);
        let fpr = collect(false);

        if tpr.len() < 2 && fpr.len() < 2 {
            return Err(Error::msg(format!(
                "equalized odds for `{attribute}` needs at least two groups with both \
                     positive and negative ground-truth examples"
            )));
        }

        let gap = |rates: &[MeasuredRate]| -> f32 {
            if rates.len() < 2 {
                return 0.0;
            }
            let max = rates.iter().map(|r| r.rate).fold(f64::NEG_INFINITY, f64::max);
            let min = rates.iter().map(|r| r.rate).fold(f64::INFINITY, f64::min);
            (max - min) as f32
        };

        // Report the dominating gap, with the significance of that same gap.
        let dominant = if gap(&tpr) >= gap(&fpr) { &tpr } else { &fpr };
        let mut metric = self.metric_from_rates(
            "Equalized Odds",
            FairnessMetricType::EqualizeDOdds,
            attribute,
            dominant,
        )?;
        metric.bias_value = gap(&tpr).max(gap(&fpr));
        metric.exceeds_threshold = metric.bias_value > self.config.bias_threshold;
        Ok(metric)
    }

    /// Calibration: the gap in expected calibration error across groups.
    ///
    /// There is no closed-form significance test for a difference of ECEs, so
    /// `p_value` and `confidence_interval` are `None` rather than invented.
    fn compute_calibration_metrics<M: Model<Input = Tensor, Output = Tensor>>(
        &self,
        model: &M,
        test_data: &FairnessTestData,
        attribute: &str,
        groups: &[String],
    ) -> Result<BiasMetric> {
        let outcomes = self.collect_group_outcomes(model, test_data, attribute, groups)?;

        let mut errors = Vec::with_capacity(outcomes.len());
        for outcome in &outcomes {
            if outcome.labels.len() != outcome.predictions.len() {
                return Err(Error::msg(format!(
                    "calibration for `{attribute}:{}` needs one label per example",
                    outcome.group
                )));
            }
            let ece =
                expected_calibration_error(&outcome.predictions, &outcome.labels, CALIBRATION_BINS)
                    .map_err(|e| Error::msg(e.to_string()))?;
            errors.push(ece);
        }

        if errors.len() < 2 {
            return Err(Error::msg(format!(
                "calibration for `{attribute}` needs at least two labelled groups"
            )));
        }

        let max = errors.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = errors.iter().copied().fold(f64::INFINITY, f64::min);
        let bias_value = (max - min) as f32;

        Ok(BiasMetric {
            name: "Calibration".to_string(),
            metric_type: FairnessMetricType::CalibrationMetrics,
            protected_attribute: attribute.to_string(),
            bias_value,
            p_value: None,
            confidence_interval: None,
            exceeds_threshold: bias_value > self.config.bias_threshold,
        })
    }

    /// Run the model over a group's inputs and read out the positive-class
    /// probability for each example.
    fn get_model_predictions<M: Model<Input = Tensor, Output = Tensor>>(
        &self,
        model: &M,
        inputs: &[Tensor],
    ) -> Result<Vec<f32>> {
        let mut predictions = Vec::with_capacity(inputs.len());
        for input in inputs {
            let output = model.forward(input.clone())?;
            predictions.push(Self::extract_probability(&output)?);
        }
        Ok(predictions)
    }

    /// Interpret a model output as the probability of the positive class.
    ///
    /// * A single value is taken as an already-normalised probability and must
    ///   lie in `[0, 1]`.
    /// * Several values are taken as class scores for one example: if they are
    ///   non-negative and sum to 1 they are used as-is, otherwise they are
    ///   softmaxed. The last class is the positive one.
    ///
    /// Anything else (an empty output, a batched output, a non-float tensor) is
    /// an error — there is no defensible probability to report.
    fn extract_probability(output: &Tensor) -> Result<f32> {
        let values = output
            .data()
            .map_err(|e| Error::msg(format!("failed to read the model output: {e}")))?;

        match values.len() {
            0 => Err(Error::msg("the model returned an empty output")),
            1 => {
                let probability = values[0];
                if !(0.0..=1.0).contains(&probability) {
                    return Err(Error::msg(format!(
                        "a single-valued model output is interpreted as a probability but \
                             {probability} is outside [0, 1]"
                    )));
                }
                Ok(probability)
            },
            _ => {
                let sum: f32 = values.iter().sum();
                let already_normalized =
                    values.iter().all(|&v| (0.0..=1.0).contains(&v)) && (sum - 1.0).abs() < 1e-3;

                if already_normalized {
                    Ok(values[values.len() - 1])
                } else {
                    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    if !max.is_finite() {
                        return Err(Error::msg(
                            "the model output contains no finite values".to_string(),
                        ));
                    }
                    let exps: Vec<f32> = values.iter().map(|&v| (v - max).exp()).collect();
                    let total: f32 = exps.iter().sum();
                    if total <= 0.0 {
                        return Err(Error::msg(
                            "the softmax of the model output is degenerate".to_string(),
                        ));
                    }
                    Ok(exps[exps.len() - 1] / total)
                }
            },
        }
    }

    /// Share of examples the model decides positively.
    fn compute_positive_rate(&self, predictions: &[f32]) -> f32 {
        if predictions.is_empty() {
            return 0.0;
        }
        let positive_count =
            predictions.iter().filter(|&&p| p > POSITIVE_DECISION_THRESHOLD).count();
        positive_count as f32 / predictions.len() as f32
    }

    /// Demographic-parity gap for every pair of protected attributes for which
    /// intersectional data was supplied.
    ///
    /// The returned map is keyed by `"attr1:group1+attr2:group2"`-style pairs
    /// collapsed to the attribute pair, and is empty when the caller provided
    /// no intersectional data (there is nothing to measure, and inventing a
    /// number would be worse than saying nothing).
    fn analyze_intersectional_bias<M: Model<Input = Tensor, Output = Tensor>>(
        &self,
        model: &M,
        test_data: &FairnessTestData,
    ) -> Result<HashMap<String, f32>> {
        let mut rates_by_pair: HashMap<String, Vec<f32>> = HashMap::new();

        for (key, group_data) in &test_data.intersectional_data {
            if group_data.inputs.is_empty() {
                return Err(Error::msg(format!(
                    "intersectional cell `{key}` carries no examples"
                )));
            }
            let predictions = self.get_model_predictions(model, &group_data.inputs)?;
            let rate = self.compute_positive_rate(&predictions);

            // `attr1:group1+attr2:group2` -> `attr1+attr2`
            let pair = key
                .split('+')
                .map(|part| part.split(':').next().unwrap_or(part).to_string())
                .collect::<Vec<_>>()
                .join("+");
            rates_by_pair.entry(pair).or_default().push(rate);
        }

        let mut bias = HashMap::new();
        for (pair, rates) in rates_by_pair {
            if rates.len() < 2 {
                continue;
            }
            let max = rates.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min = rates.iter().copied().fold(f32::INFINITY, f32::min);
            bias.insert(pair, max - min);
        }

        Ok(bias)
    }

    /// Chi-square test of independence between group membership and the
    /// model's decision, one test per protected attribute.
    ///
    /// The contingency table is `groups x {positive, negative}`, so the test
    /// has `(groups - 1)` degrees of freedom. Attributes whose table is
    /// degenerate (a single group, or a decision column nobody lands in) are
    /// skipped rather than reported with a fabricated statistic.
    fn perform_statistical_tests<M: Model<Input = Tensor, Output = Tensor>>(
        &self,
        model: &M,
        test_data: &FairnessTestData,
    ) -> Result<Vec<StatisticalTest>> {
        let mut tests = Vec::new();

        for attribute in &self.config.protected_attributes {
            let groups = test_data.get_groups_for_attribute(attribute);
            if groups.len() < 2 {
                continue;
            }

            let outcomes = self.collect_group_outcomes(model, test_data, attribute, &groups)?;
            let table: Vec<Vec<f64>> = outcomes
                .iter()
                .map(|outcome| {
                    let (positives, total) = outcome.positive_rate();
                    vec![positives, total - positives]
                })
                .collect();

            let result = match chi_square_test(&table) {
                Ok(result) => result,
                // A degenerate table means the test does not apply here.
                Err(_) => continue,
            };

            let significance = 1.0 - f64::from(self.config.confidence_level);
            tests.push(StatisticalTest {
                test_name: format!("Chi-square test for independence ({attribute})"),
                statistic: result.statistic as f32,
                p_value: result.p_value as f32,
                critical_value: chi_square_quantile(result.degrees_of_freedom, significance) as f32,
                is_significant: result.p_value < significance,
                degrees_of_freedom: Some(result.degrees_of_freedom as i32),
            });
        }

        Ok(tests)
    }
}

impl FairnessAssessment {
    fn compute_overall_fairness_score(
        &self,
        bias_metrics: &HashMap<String, Vec<BiasMetric>>,
    ) -> f32 {
        let mut total_bias = 0.0;
        let mut metric_count = 0;
        for metrics in bias_metrics.values() {
            for metric in metrics {
                total_bias += metric.bias_value;
                metric_count += 1;
            }
        }
        if metric_count == 0 {
            1.0
        } else {
            (1.0 - total_bias / metric_count as f32).clamp(0.0, 1.0)
        }
    }

    fn determine_violation_severity(&self, bias_value: f32) -> String {
        if bias_value > 0.2 {
            "high".to_string()
        } else if bias_value > 0.1 {
            "medium".to_string()
        } else {
            "low".to_string()
        }
    }

    fn generate_recommendations(
        &self,
        _metric_type: &FairnessMetricType,
        _metric: &BiasMetric,
    ) -> Vec<String> {
        vec!["Consider bias mitigation strategies".to_string()]
    }

    fn generate_mitigation_recommendations(&self, violations: &[FairnessViolation]) -> Vec<String> {
        if violations.is_empty() {
            vec!["No significant bias violations detected. Continue monitoring.".to_string()]
        } else {
            vec!["Implement bias mitigation strategies".to_string()]
        }
    }

    /// Generate fairness assessment report
    pub fn generate_report(&self, result: &FairnessResult) -> String {
        format!(
            "# Fairness Assessment Report\n\n**Overall Fairness Score:** {:.3}\n",
            result.overall_fairness_score
        )
    }
}

impl Default for FairnessAssessment {
    fn default() -> Self {
        Self::new()
    }
}

impl FairnessTestData {
    pub fn new() -> Self {
        Self {
            grouped_data: HashMap::new(),
            intersectional_data: HashMap::new(),
        }
    }

    pub fn get_groups_for_attribute(&self, attribute: &str) -> Vec<String> {
        self.grouped_data
            .get(attribute)
            .map(|groups| groups.keys().cloned().collect())
            .unwrap_or_default()
    }

    pub fn get_group_data(&self, attribute: &str, group: &str) -> Result<&GroupData> {
        self.grouped_data
            .get(attribute)
            .and_then(|groups| groups.get(group))
            .ok_or_else(|| Error::msg(format!("Group data not found for {}:{}", attribute, group)))
    }

    pub fn get_intersectional_data(
        &self,
        attr1: &str,
        group1: &str,
        attr2: &str,
        group2: &str,
    ) -> Result<&GroupData> {
        let key = format!("{}:{}+{}:{}", attr1, group1, attr2, group2);
        self.intersectional_data
            .get(&key)
            .ok_or_else(|| Error::msg(format!("Intersectional data not found for {}", key)))
    }
}

impl Default for FairnessTestData {
    fn default() -> Self {
        Self::new()
    }
}
