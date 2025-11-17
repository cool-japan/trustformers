//! Statistical analysis for A/B test results

#![allow(unused_variables)] // A/B testing analysis

use super::{MetricDataPoint, MetricType, Variant};
use anyhow::Result;
use std::collections::HashMap;

/// Statistical test results
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Control variant metrics
    pub control_stats: VariantStatistics,
    /// Treatment variant metrics
    pub treatment_stats: Vec<VariantStatistics>,
    /// Statistical test results
    pub test_stats: TestStatistics,
    /// Overall recommendation
    pub recommendation: TestRecommendation,
}

/// Statistics for a single variant
#[derive(Debug, Clone)]
pub struct VariantStatistics {
    /// The variant
    pub variant: Variant,
    /// Sample size
    pub sample_size: usize,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Standard error
    pub std_error: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
}

/// Results of statistical tests
#[derive(Debug, Clone)]
pub struct TestStatistics {
    /// P-value from hypothesis test
    pub p_value: f64,
    /// Test statistic (t-stat or z-stat)
    pub test_statistic: f64,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
    /// Statistical power
    pub power: f64,
    /// Minimum detectable effect
    pub min_detectable_effect: f64,
    /// Confidence level used
    pub confidence_level: ConfidenceLevel,
}

/// Confidence levels for testing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConfidenceLevel {
    /// 90% confidence
    Low,
    /// 95% confidence
    Medium,
    /// 99% confidence
    High,
}

impl ConfidenceLevel {
    /// Get alpha value
    pub fn alpha(&self) -> f64 {
        match self {
            ConfidenceLevel::Low => 0.10,
            ConfidenceLevel::Medium => 0.05,
            ConfidenceLevel::High => 0.01,
        }
    }

    /// Get z-score for confidence interval
    pub fn z_score(&self) -> f64 {
        match self {
            ConfidenceLevel::Low => 1.645,
            ConfidenceLevel::Medium => 1.96,
            ConfidenceLevel::High => 2.576,
        }
    }
}

/// Test recommendation
#[derive(Debug, Clone, PartialEq)]
pub enum TestRecommendation {
    /// Treatment is significantly better
    AdoptTreatment { variant: String, improvement: f64 },
    /// Control is significantly better
    KeepControl { degradation: f64 },
    /// No significant difference
    NoSignificantDifference,
    /// Need more data
    InsufficientData { required_sample_size: usize },
}

/// Statistical analyzer for A/B tests
pub struct StatisticalAnalyzer {
    /// Default confidence level
    default_confidence: ConfidenceLevel,
    /// Minimum sample size per variant
    min_sample_size: usize,
}

impl Default for StatisticalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self {
            default_confidence: ConfidenceLevel::Medium,
            min_sample_size: 30,
        }
    }

    /// Create with custom settings
    pub fn with_settings(confidence: ConfidenceLevel, min_sample_size: usize) -> Self {
        Self {
            default_confidence: confidence,
            min_sample_size,
        }
    }

    /// Analyze experiment results
    pub fn analyze(
        &self,
        metrics: HashMap<(Variant, MetricType), Vec<MetricDataPoint>>,
    ) -> Result<TestResult> {
        // Separate control and treatment metrics, preserving metric type
        let mut control_data = None;
        let mut treatment_data = Vec::new();
        let mut metric_type = None;

        for ((variant, m_type), data_points) in metrics {
            let values: Vec<f64> = data_points.iter().map(|dp| dp.value.as_f64()).collect();

            // Store the metric type (assume all entries have the same metric type)
            if metric_type.is_none() {
                metric_type = Some(m_type.clone());
            }

            if variant.name() == "control" {
                control_data = Some((variant, values));
            } else {
                treatment_data.push((variant, values));
            }
        }

        let primary_metric_type =
            metric_type.ok_or_else(|| anyhow::anyhow!("No metric type found"))?;

        let (control_variant, control_values) =
            control_data.ok_or_else(|| anyhow::anyhow!("No control variant data found"))?;

        if treatment_data.is_empty() {
            anyhow::bail!("No treatment variant data found");
        }

        // Calculate statistics for control
        let control_stats = self.calculate_variant_stats(control_variant, &control_values)?;

        // Calculate statistics for treatments
        let mut treatment_stats = Vec::new();
        let mut best_treatment = None;
        let mut best_p_value = f64::INFINITY; // Start with infinity so any p-value will be better

        for (variant, values) in treatment_data {
            let stats = self.calculate_variant_stats(variant.clone(), &values)?;

            // Perform hypothesis test
            let test_result = self.perform_test(&control_values, &values)?;

            if test_result.p_value < best_p_value {
                best_p_value = test_result.p_value;
                best_treatment = Some((variant, stats.clone(), test_result));
            }

            treatment_stats.push(stats);
        }

        // Generate recommendation
        let recommendation = if let Some((variant, stats, test_result)) = &best_treatment {
            self.generate_recommendation(
                &control_stats,
                stats,
                test_result,
                variant,
                &primary_metric_type,
            )
        } else {
            TestRecommendation::NoSignificantDifference
        };

        // Use the best treatment's test statistics
        let test_stats = if let Some((_, _, test_result)) = best_treatment {
            test_result
        } else {
            TestStatistics {
                p_value: 1.0,
                test_statistic: 0.0,
                effect_size: 0.0,
                power: 0.0,
                min_detectable_effect: 0.0,
                confidence_level: self.default_confidence,
            }
        };

        Ok(TestResult {
            control_stats,
            treatment_stats,
            test_stats,
            recommendation,
        })
    }

    /// Calculate statistics for a variant
    fn calculate_variant_stats(
        &self,
        variant: Variant,
        values: &[f64],
    ) -> Result<VariantStatistics> {
        let sample_size = values.len();
        if sample_size == 0 {
            anyhow::bail!("No data points for variant");
        }

        let mean = values.iter().sum::<f64>() / sample_size as f64;
        let variance =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (sample_size - 1) as f64;
        let std_dev = variance.sqrt();
        let std_error = std_dev / (sample_size as f64).sqrt();

        let z_score = self.default_confidence.z_score();
        let margin_of_error = z_score * std_error;
        let confidence_interval = (mean - margin_of_error, mean + margin_of_error);

        Ok(VariantStatistics {
            variant,
            sample_size,
            mean,
            std_dev,
            std_error,
            confidence_interval,
        })
    }

    /// Perform two-sample t-test
    fn perform_test(&self, control: &[f64], treatment: &[f64]) -> Result<TestStatistics> {
        let n1 = control.len() as f64;
        let n2 = treatment.len() as f64;

        if n1 < self.min_sample_size as f64 || n2 < self.min_sample_size as f64 {
            return Ok(TestStatistics {
                p_value: 1.0,
                test_statistic: 0.0,
                effect_size: 0.0,
                power: 0.0,
                min_detectable_effect: 0.0,
                confidence_level: self.default_confidence,
            });
        }

        // Calculate means
        let mean1 = control.iter().sum::<f64>() / n1;
        let mean2 = treatment.iter().sum::<f64>() / n2;

        // Calculate variances
        let var1 = control.iter().map(|v| (v - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
        let var2 = treatment.iter().map(|v| (v - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

        // Pooled standard deviation
        let pooled_std = (((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0)).sqrt();

        // Test statistic
        let test_statistic = (mean2 - mean1) / (pooled_std * (1.0 / n1 + 1.0 / n2).sqrt());

        // Degrees of freedom
        let df = n1 + n2 - 2.0;

        // P-value (simplified - in practice use a proper t-distribution)
        let p_value = self.calculate_p_value(test_statistic.abs(), df);

        // Effect size (Cohen's d)
        let effect_size = (mean2 - mean1).abs() / pooled_std;

        // Statistical power (simplified calculation)
        let power = self.calculate_power(effect_size, n1, n2);

        // Minimum detectable effect
        let min_detectable_effect = self.calculate_mde(n1, n2, pooled_std);

        Ok(TestStatistics {
            p_value,
            test_statistic,
            effect_size,
            power,
            min_detectable_effect,
            confidence_level: self.default_confidence,
        })
    }

    /// Calculate p-value (simplified)
    fn calculate_p_value(&self, t_stat: f64, _df: f64) -> f64 {
        // Simplified normal approximation
        // In practice, use proper t-distribution
        let z = t_stat;
        2.0 * (1.0 - self.normal_cdf(z))
    }

    /// Normal CDF approximation
    fn normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + self.erf(x / std::f64::consts::SQRT_2))
    }

    /// Error function approximation
    fn erf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Calculate statistical power
    fn calculate_power(&self, effect_size: f64, n1: f64, n2: f64) -> f64 {
        // Simplified power calculation
        let n_harmonic = (n1 * n2) / (n1 + n2);
        let noncentrality = effect_size * (n_harmonic / 2.0).sqrt();
        let critical_value = self.default_confidence.z_score();

        // Power = P(Z > critical_value - noncentrality)
        1.0 - self.normal_cdf(critical_value - noncentrality)
    }

    /// Calculate minimum detectable effect
    fn calculate_mde(&self, n1: f64, n2: f64, pooled_std: f64) -> f64 {
        let alpha = self.default_confidence.alpha();
        let beta = 0.2; // 80% power
        let z_alpha = self.default_confidence.z_score();
        let z_beta = 0.84; // z-score for 80% power

        let n_harmonic = (n1 * n2) / (n1 + n2);
        (z_alpha + z_beta) * pooled_std * (2.0 / n_harmonic).sqrt()
    }

    /// Generate recommendation
    fn generate_recommendation(
        &self,
        control: &VariantStatistics,
        treatment: &VariantStatistics,
        test_stats: &TestStatistics,
        variant: &Variant,
        metric_type: &MetricType,
    ) -> TestRecommendation {
        // Check sample size
        if control.sample_size < self.min_sample_size
            || treatment.sample_size < self.min_sample_size
        {
            let required =
                self.min_sample_size.max(control.sample_size).max(treatment.sample_size) * 2;
            return TestRecommendation::InsufficientData {
                required_sample_size: required,
            };
        }

        // Check statistical significance
        if test_stats.p_value >= self.default_confidence.alpha() {
            return TestRecommendation::NoSignificantDifference;
        }

        // Calculate improvement based on metric type directionality
        let improvement = ((treatment.mean - control.mean) / control.mean) * 100.0;

        // Determine if treatment is better based on metric type
        let treatment_is_better = if metric_type.lower_is_better() {
            treatment.mean < control.mean
        } else {
            treatment.mean > control.mean
        };

        if treatment_is_better {
            TestRecommendation::AdoptTreatment {
                variant: variant.name().to_string(),
                improvement, // Keep the raw improvement (negative for latency improvement)
            }
        } else {
            TestRecommendation::KeepControl {
                degradation: improvement.abs(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ab_testing::MetricValue;

    fn create_test_data(mean: f64, std_dev: f64, size: usize) -> Vec<MetricDataPoint> {
        use scirs2_core::random::*;
        let normal = Normal::new(mean, std_dev).unwrap();
        let mut rng = thread_rng();

        (0..size)
            .map(|_| MetricDataPoint {
                timestamp: chrono::Utc::now(),
                value: MetricValue::Numeric(normal.sample(&mut rng)),
                metadata: None,
            })
            .collect()
    }

    #[test]
    fn test_significant_difference() {
        let analyzer = StatisticalAnalyzer::new();

        let control = Variant::new("control", "v1");
        let treatment = Variant::new("treatment", "v2");

        let mut metrics = HashMap::new();

        // Control: mean=100, std=10
        metrics.insert(
            (control.clone(), MetricType::Latency),
            create_test_data(100.0, 10.0, 100),
        );

        // Treatment: mean=90, std=10 (10% improvement)
        metrics.insert(
            (treatment.clone(), MetricType::Latency),
            create_test_data(90.0, 10.0, 100),
        );

        let result = analyzer.analyze(metrics).unwrap();

        // Should detect significant improvement
        match result.recommendation {
            TestRecommendation::AdoptTreatment {
                variant,
                improvement,
            } => {
                assert_eq!(variant, "treatment");
                assert!(improvement < 0.0); // Negative because lower latency is better
            },
            other => panic!("Expected to recommend treatment adoption, got: {:?}", other),
        }
    }

    #[test]
    fn test_no_significant_difference() {
        let analyzer = StatisticalAnalyzer::new();

        let control = Variant::new("control", "v1");
        let treatment = Variant::new("treatment", "v2");

        let mut metrics = HashMap::new();

        // Create deterministic data with identical means to ensure no statistical significance
        let create_identical_data = |mean: f64, size: usize| -> Vec<MetricDataPoint> {
            (0..size)
                .map(|_| MetricDataPoint {
                    timestamp: chrono::Utc::now(),
                    value: MetricValue::Numeric(mean),
                    metadata: None,
                })
                .collect()
        };

        // Both variants have identical performance to ensure no significance
        metrics.insert(
            (control.clone(), MetricType::Accuracy),
            create_identical_data(0.95, 100),
        );

        metrics.insert(
            (treatment.clone(), MetricType::Accuracy),
            create_identical_data(0.95, 100),
        );

        let result = analyzer.analyze(metrics).unwrap();

        assert_eq!(
            result.recommendation,
            TestRecommendation::NoSignificantDifference
        );
    }

    #[test]
    fn test_insufficient_data() {
        let analyzer = StatisticalAnalyzer::new();

        let control = Variant::new("control", "v1");
        let treatment = Variant::new("treatment", "v2");

        let mut metrics = HashMap::new();

        // Too few samples
        metrics.insert(
            (control.clone(), MetricType::Throughput),
            create_test_data(1000.0, 50.0, 10),
        );

        metrics.insert(
            (treatment.clone(), MetricType::Throughput),
            create_test_data(1100.0, 50.0, 10),
        );

        let result = analyzer.analyze(metrics).unwrap();

        match result.recommendation {
            TestRecommendation::InsufficientData {
                required_sample_size,
            } => {
                assert!(required_sample_size > 20);
            },
            ref other => panic!(
                "Expected insufficient data recommendation, got: {:?}",
                other
            ),
        }
    }
}
