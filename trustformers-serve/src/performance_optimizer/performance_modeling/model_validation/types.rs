//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::{collections::HashMap, sync::Arc, time::Duration};

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use scirs2_core::random::prelude::*;
use tracing;

// use super::types::*; // Circular import - commented out
use crate::performance_optimizer::performance_modeling::types::{
    DistributionInfo, DistributionType, PerformancePredictor, ResidualAnalysis, TestDataStatistics,
    ValidationConfig, ValidationDetails, ValidationMetric, ValidationResult,
    ValidationStrategyType,
};
use crate::performance_optimizer::types::PerformanceDataPoint;

use super::functions::{MetricCalculator, ValidationStrategy};

/// Leave-one-out validation strategy
#[derive(Debug)]
pub struct LeaveOneOutValidation;
impl LeaveOneOutValidation {
    pub fn new() -> Self {
        Self
    }
}
/// Validation statistics
#[derive(Debug, Clone)]
pub struct ValidationStatistics {
    /// Total number of validations performed
    pub total_validations: usize,
    /// Average confidence across all validations
    pub average_confidence: f32,
    /// Best confidence achieved
    pub best_confidence: f32,
    /// Validation frequency (validations per hour)
    pub validation_frequency: f32,
    /// Last validation timestamp
    pub last_validation: Option<DateTime<Utc>>,
}
/// Cross-validation strategy
#[derive(Debug)]
pub struct CrossValidation {
    pub(super) folds: usize,
}
impl CrossValidation {
    pub fn new(folds: usize) -> Self {
        Self { folds }
    }
    pub(crate) fn create_folds(
        &self,
        data: &[PerformanceDataPoint],
    ) -> Vec<(Vec<PerformanceDataPoint>, Vec<PerformanceDataPoint>)> {
        let fold_size = data.len() / self.folds;
        let mut folds = Vec::new();
        for i in 0..self.folds {
            let start = i * fold_size;
            let end = if i == self.folds - 1 { data.len() } else { start + fold_size };
            let mut train_data = Vec::new();
            let mut test_data = Vec::new();
            for (idx, item) in data.iter().enumerate() {
                if idx >= start && idx < end {
                    test_data.push(item.clone());
                } else {
                    train_data.push(item.clone());
                }
            }
            folds.push((train_data, test_data));
        }
        folds
    }
}
/// Comprehensive validation result combining multiple strategies
#[derive(Debug, Clone)]
pub struct ComprehensiveValidationResult {
    /// Results from different validation strategies
    pub strategy_results: HashMap<ValidationStrategyType, ValidationResult>,
    /// Comprehensive metrics across all strategies
    pub comprehensive_metrics: HashMap<ValidationMetric, f32>,
    /// Statistical significance tests
    pub significance_tests: HashMap<String, SignificanceTestResult>,
    /// Model name
    pub model_name: String,
    /// Validation timestamp
    pub validation_timestamp: DateTime<Utc>,
    /// Total validation duration
    pub validation_duration: Duration,
    /// Data characteristics
    pub data_characteristics: TestDataStatistics,
    pub overall_score: f64,
}
impl ComprehensiveValidationResult {
    /// Get cross-validation scores
    pub fn get_cv_scores(&self) -> Vec<f32> {
        self.strategy_results
            .values()
            .flat_map(|result| result.cv_scores.clone())
            .collect()
    }
    /// Calculate overall confidence
    pub fn calculate_overall_confidence(&self) -> f32 {
        if self.strategy_results.is_empty() {
            return 0.0;
        }
        let total_confidence: f32 =
            self.strategy_results.values().map(|result| result.confidence).sum();
        total_confidence / self.strategy_results.len() as f32
    }
    /// Get prediction errors
    pub fn get_prediction_errors(&self) -> Vec<f32> {
        self.strategy_results
            .values()
            .flat_map(|result| result.details.prediction_errors.clone())
            .collect()
    }
    /// Calculate residual analysis
    pub fn calculate_residual_analysis(&self) -> ResidualAnalysis {
        let errors = self.get_prediction_errors();
        if errors.is_empty() {
            return ResidualAnalysis {
                autocorrelation: 0.0,
                heteroscedasticity_p_value: 1.0,
                normality_p_value: 1.0,
                outliers: Vec::new(),
            };
        }
        let autocorr = if errors.len() > 1 {
            let mut sum = 0.0f32;
            for i in 1..errors.len() {
                sum += errors[i] * errors[i - 1];
            }
            sum / (errors.len() - 1) as f32
        } else {
            0.0
        };
        let mean_error = errors.iter().sum::<f32>() / errors.len() as f32;
        let std_error = {
            let variance =
                errors.iter().map(|e| (e - mean_error).powi(2)).sum::<f32>() / errors.len() as f32;
            variance.sqrt()
        };
        let outliers: Vec<usize> =
            errors
                .iter()
                .enumerate()
                .filter_map(|(i, &error)| {
                    if (error - mean_error).abs() > 3.0 * std_error {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();
        ResidualAnalysis {
            autocorrelation: autocorr,
            heteroscedasticity_p_value: 0.5,
            normality_p_value: 0.5,
            outliers,
        }
    }
}
/// Statistical significance test result
#[derive(Debug, Clone)]
pub struct SignificanceTestResult {
    /// Test statistic value
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Whether result is statistically significant
    pub significant: bool,
    /// Type of test performed
    pub test_type: String,
    /// Effect size
    pub effect_size: f64,
}
/// Bootstrap validation strategy
#[derive(Debug)]
pub struct BootstrapValidation {
    pub(super) n_bootstrap: usize,
}
impl BootstrapValidation {
    pub fn new(n_bootstrap: usize) -> Self {
        Self { n_bootstrap }
    }
}
impl BootstrapValidation {
    pub(crate) fn create_bootstrap_sample(
        &self,
        data: &[PerformanceDataPoint],
    ) -> Vec<PerformanceDataPoint> {
        let mut rng = thread_rng();
        let mut bootstrap_sample = Vec::with_capacity(data.len());
        for _ in 0..data.len() {
            let idx = rng.gen_range(0..data.len());
            bootstrap_sample.push(data[idx].clone());
        }
        bootstrap_sample
    }
}
/// Hold-out validation strategy
#[derive(Debug)]
pub struct HoldOutValidation {
    pub(super) test_size: f32,
}
impl HoldOutValidation {
    pub fn new(test_size: f32) -> Self {
        Self { test_size }
    }
}
impl HoldOutValidation {
    pub(crate) fn calculate_confidence(&self, metrics: &HashMap<ValidationMetric, f32>) -> f32 {
        if let Some(&r_squared) = metrics.get(&ValidationMetric::RSquared) {
            r_squared.clamp(0.0, 1.0)
        } else if let Some(&mae) = metrics.get(&ValidationMetric::MeanAbsoluteError) {
            (1.0 - mae / 100.0).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }
}
/// Mean Absolute Error calculator
#[derive(Debug, Clone, Copy)]
pub struct MAECalculator;
/// R-squared calculator
#[derive(Debug, Clone, Copy)]
pub struct RSquaredCalculator;
/// Mean Squared Logarithmic Error calculator
#[derive(Debug, Clone, Copy)]
pub struct MSLECalculator;
/// Explained Variance Score calculator
#[derive(Debug, Clone, Copy)]
pub struct ExplainedVarianceCalculator;
/// Root Mean Squared Error calculator
#[derive(Debug, Clone, Copy)]
pub struct RMSECalculator;
/// Comprehensive model validation orchestrator
#[derive(Debug)]
pub struct ModelValidationOrchestrator {
    /// Validation configuration
    pub(super) config: Arc<RwLock<ValidationConfig>>,
    /// Available validation strategies
    pub(super) strategies: HashMap<ValidationStrategyType, Box<dyn ValidationStrategy>>,
    /// Validation results history
    validation_history: Arc<RwLock<Vec<ValidationResult>>>,
    /// Metrics calculators
    metrics_calculators: HashMap<ValidationMetric, Box<dyn MetricCalculator>>,
}
impl ModelValidationOrchestrator {
    /// Create new validation orchestrator
    pub fn new(config: ValidationConfig) -> Self {
        let mut orchestrator = Self {
            config: Arc::new(RwLock::new(config.clone())),
            strategies: HashMap::new(),
            validation_history: Arc::new(RwLock::new(Vec::new())),
            metrics_calculators: HashMap::new(),
        };
        orchestrator.register_strategy(
            ValidationStrategyType::HoldOut,
            Box::new(HoldOutValidation::new(config.test_size)),
        );
        orchestrator.register_strategy(
            ValidationStrategyType::CrossValidation,
            Box::new(CrossValidation::new(config.cv_folds)),
        );
        orchestrator.register_strategy(
            ValidationStrategyType::TimeSeries,
            Box::new(TimeSeriesValidation::new()),
        );
        orchestrator.register_strategy(
            ValidationStrategyType::Bootstrap,
            Box::new(BootstrapValidation::new(1000)),
        );
        orchestrator.register_strategy(
            ValidationStrategyType::LeaveOneOut,
            Box::new(LeaveOneOutValidation::new()),
        );
        orchestrator.register_metric_calculator(
            ValidationMetric::MeanAbsoluteError,
            Box::new(MAECalculator),
        );
        orchestrator.register_metric_calculator(
            ValidationMetric::RootMeanSquaredError,
            Box::new(RMSECalculator),
        );
        orchestrator
            .register_metric_calculator(ValidationMetric::RSquared, Box::new(RSquaredCalculator));
        orchestrator.register_metric_calculator(
            ValidationMetric::MeanAbsolutePercentageError,
            Box::new(MAPECalculator),
        );
        orchestrator.register_metric_calculator(
            ValidationMetric::MeanSquaredLogarithmicError,
            Box::new(MSLECalculator),
        );
        orchestrator.register_metric_calculator(
            ValidationMetric::ExplainedVarianceScore,
            Box::new(ExplainedVarianceCalculator),
        );
        orchestrator
    }
    /// Register a validation strategy
    pub fn register_strategy(
        &mut self,
        strategy_type: ValidationStrategyType,
        strategy: Box<dyn ValidationStrategy>,
    ) {
        self.strategies.insert(strategy_type, strategy);
    }
    /// Register a metric calculator
    pub fn register_metric_calculator(
        &mut self,
        metric: ValidationMetric,
        calculator: Box<dyn MetricCalculator>,
    ) {
        self.metrics_calculators.insert(metric, calculator);
    }
    /// Validate a model using configured strategies
    pub async fn validate_model(
        &self,
        model: &dyn PerformancePredictor,
        test_data: &[PerformanceDataPoint],
    ) -> Result<ComprehensiveValidationResult> {
        if test_data.len() < self.config.read().min_validation_samples {
            return Err(anyhow!(
                "Insufficient validation samples: {} < {}",
                test_data.len(),
                self.config.read().min_validation_samples
            ));
        }
        let mut strategy_results = HashMap::new();
        let start_time = std::time::Instant::now();
        for (strategy_type, strategy) in &self.strategies {
            let config = self.config.read();
            if config.strategy == *strategy_type
                || matches!(config.strategy, ValidationStrategyType::CrossValidation)
            {
                match strategy.validate(model, test_data, &*config).await {
                    Ok(result) => {
                        strategy_results.insert(*strategy_type, result);
                        tracing::info!("Completed validation with strategy: {:?}", strategy_type);
                    },
                    Err(e) => {
                        tracing::warn!("Validation failed for strategy {:?}: {}", strategy_type, e);
                    },
                }
            }
        }
        if strategy_results.is_empty() {
            return Err(anyhow!("No validation strategies succeeded"));
        }
        let comprehensive_metrics =
            self.calculate_comprehensive_metrics(&strategy_results, test_data)?;
        let significance_tests = self.perform_significance_tests(&strategy_results)?;
        let mut result = ComprehensiveValidationResult {
            strategy_results,
            comprehensive_metrics,
            significance_tests,
            model_name: model.name().to_string(),
            validation_timestamp: Utc::now(),
            validation_duration: start_time.elapsed(),
            data_characteristics: self.analyze_data_characteristics(test_data)?,
            overall_score: 0.0,
        };
        result.overall_score = result.calculate_overall_confidence() as f64;
        {
            let mut history = self.validation_history.write();
            history.push(ValidationResult {
                metrics: result.comprehensive_metrics.clone(),
                cv_scores: result.get_cv_scores(),
                confidence: result.calculate_overall_confidence(),
                details: ValidationDetails {
                    test_samples: test_data.len(),
                    test_statistics: result.data_characteristics.clone(),
                    prediction_errors: result.get_prediction_errors(),
                    residual_analysis: result.calculate_residual_analysis(),
                },
                validated_at: result.validation_timestamp,
            });
            if history.len() > 100 {
                history.remove(0);
            }
        }
        Ok(result)
    }
    /// Calculate comprehensive metrics across strategies
    fn calculate_comprehensive_metrics(
        &self,
        strategy_results: &HashMap<ValidationStrategyType, ValidationResult>,
        test_data: &[PerformanceDataPoint],
    ) -> Result<HashMap<ValidationMetric, f32>> {
        let mut comprehensive_metrics = HashMap::new();
        let config = self.config.read();
        for metric in &config.metrics {
            if let Some(calculator) = self.metrics_calculators.get(metric) {
                let mut all_predictions = Vec::new();
                let mut all_actuals = Vec::new();
                for result in strategy_results.values() {
                    if let Some(predictions) = self.extract_predictions_from_result(result) {
                        let predictions_len = predictions.len();
                        all_predictions.extend(predictions);
                        all_actuals
                            .extend(test_data.iter().take(predictions_len).map(|d| d.throughput));
                    }
                }
                if !all_predictions.is_empty() && all_predictions.len() == all_actuals.len() {
                    let metric_value = calculator.calculate(&all_predictions, &all_actuals)?;
                    comprehensive_metrics.insert(*metric, metric_value);
                }
            }
        }
        Ok(comprehensive_metrics)
    }
    /// Extract predictions from validation result
    fn extract_predictions_from_result(&self, _result: &ValidationResult) -> Option<Vec<f64>> {
        None
    }
    /// Perform statistical significance tests
    fn perform_significance_tests(
        &self,
        strategy_results: &HashMap<ValidationStrategyType, ValidationResult>,
    ) -> Result<HashMap<String, SignificanceTestResult>> {
        let mut tests = HashMap::new();
        let strategies: Vec<_> = strategy_results.keys().collect();
        for i in 0..strategies.len() {
            for j in i + 1..strategies.len() {
                let strategy1 = strategies[i];
                let strategy2 = strategies[j];
                if let (Some(result1), Some(result2)) = (
                    strategy_results.get(strategy1),
                    strategy_results.get(strategy2),
                ) {
                    let test_name = format!("{:?}_vs_{:?}", strategy1, strategy2);
                    let test_result = self.compare_validation_results(result1, result2)?;
                    tests.insert(test_name, test_result);
                }
            }
        }
        Ok(tests)
    }
    /// Compare two validation results statistically
    fn compare_validation_results(
        &self,
        result1: &ValidationResult,
        result2: &ValidationResult,
    ) -> Result<SignificanceTestResult> {
        let scores1 = &result1.cv_scores;
        let scores2 = &result2.cv_scores;
        if scores1.len() != scores2.len() || scores1.is_empty() {
            return Ok(SignificanceTestResult {
                test_statistic: 0.0,
                p_value: 1.0,
                significant: false,
                test_type: "insufficient_data".to_string(),
                effect_size: 0.0,
            });
        }
        let differences: Vec<f32> =
            scores1.iter().zip(scores2.iter()).map(|(s1, s2)| s1 - s2).collect();
        let mean_diff = differences.iter().sum::<f32>() / differences.len() as f32;
        let var_diff = differences.iter().map(|d| (d - mean_diff).powi(2)).sum::<f32>()
            / (differences.len() - 1).max(1) as f32;
        let std_err = (var_diff / differences.len() as f32).sqrt();
        let t_statistic = if std_err > 1e-8 { mean_diff / std_err } else { 0.0 };
        let p_value = if t_statistic.abs() > 2.0 {
            0.01
        } else if t_statistic.abs() > 1.5 {
            0.05
        } else {
            0.2
        };
        Ok(SignificanceTestResult {
            test_statistic: t_statistic as f64,
            p_value,
            significant: p_value < 0.05,
            test_type: "paired_t_test".to_string(),
            effect_size: mean_diff.abs() as f64,
        })
    }
    /// Analyze data characteristics
    fn analyze_data_characteristics(
        &self,
        test_data: &[PerformanceDataPoint],
    ) -> Result<TestDataStatistics> {
        if test_data.is_empty() {
            return Err(anyhow!("No test data provided"));
        }
        let throughputs: Vec<f64> = test_data.iter().map(|d| d.throughput).collect();
        let mean_target = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let variance = throughputs.iter().map(|t| (t - mean_target).powi(2)).sum::<f64>()
            / throughputs.len() as f64;
        let target_std = variance.sqrt();
        let mut feature_correlations = HashMap::new();
        feature_correlations.insert(
            "parallelism".to_string(),
            self.calculate_correlation_with_throughput(test_data, |d| d.parallelism as f64),
        );
        feature_correlations.insert(
            "cpu_utilization".to_string(),
            self.calculate_correlation_with_throughput(test_data, |d| d.cpu_utilization as f64),
        );
        feature_correlations.insert(
            "memory_utilization".to_string(),
            self.calculate_correlation_with_throughput(test_data, |d| d.memory_utilization as f64),
        );
        let distribution_info = self.analyze_distribution(&throughputs)?;
        Ok(TestDataStatistics {
            mean_target: mean_target as f32,
            target_std: target_std as f32,
            feature_correlations,
            distribution_info,
        })
    }
    /// Calculate correlation with throughput
    fn calculate_correlation_with_throughput<F>(
        &self,
        data: &[PerformanceDataPoint],
        feature_extractor: F,
    ) -> f32
    where
        F: Fn(&PerformanceDataPoint) -> f64,
    {
        let features: Vec<f64> = data.iter().map(feature_extractor).collect();
        let throughputs: Vec<f64> = data.iter().map(|d| d.throughput).collect();
        self.calculate_correlation(&features, &throughputs)
    }
    /// Calculate correlation between two vectors
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f32 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        let numerator: f64 =
            x.iter().zip(y.iter()).map(|(xi, yi)| (xi - mean_x) * (yi - mean_y)).sum();
        let sum_sq_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator > 1e-12 {
            (numerator / denominator) as f32
        } else {
            0.0
        }
    }
    /// Analyze data distribution
    fn analyze_distribution(&self, data: &[f64]) -> Result<DistributionInfo> {
        if data.is_empty() {
            return Err(anyhow!("No data for distribution analysis"));
        }
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();
        let normality_p_value = self.approximate_normality_test(&sorted_data, mean, std_dev)?;
        let distribution_type = if normality_p_value > 0.05 {
            DistributionType::Normal
        } else if data.iter().all(|&x| x > 0.0) && variance > mean * mean {
            DistributionType::LogNormal
        } else {
            DistributionType::Custom("Unknown".to_string())
        };
        let mut parameters = HashMap::new();
        parameters.insert("mean".to_string(), mean as f32);
        parameters.insert("std".to_string(), std_dev as f32);
        parameters.insert("min".to_string(), sorted_data[0] as f32);
        parameters.insert("max".to_string(), sorted_data[sorted_data.len() - 1] as f32);
        Ok(DistributionInfo {
            distribution_type,
            parameters,
            normality_p_value,
        })
    }
    /// Approximate normality test
    fn approximate_normality_test(
        &self,
        sorted_data: &[f64],
        mean: f64,
        std_dev: f64,
    ) -> Result<f32> {
        if sorted_data.len() < 3 {
            return Ok(1.0);
        }
        let n = sorted_data.len() as f64;
        let skewness = sorted_data.iter().map(|x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / n;
        let kurtosis =
            sorted_data.iter().map(|x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / n - 3.0;
        let skewness_test = skewness.abs() < 1.0;
        let kurtosis_test = kurtosis.abs() < 2.0;
        if skewness_test && kurtosis_test {
            Ok(0.8)
        } else if skewness_test || kurtosis_test {
            Ok(0.3)
        } else {
            Ok(0.05)
        }
    }
    /// Get validation history
    pub fn get_validation_history(&self) -> Vec<ValidationResult> {
        let history = self.validation_history.read();
        history.clone()
    }
    /// Get validation statistics
    pub fn get_validation_statistics(&self) -> ValidationStatistics {
        let history = self.validation_history.read();
        if history.is_empty() {
            return ValidationStatistics::default();
        }
        let total_validations = history.len();
        let average_confidence =
            history.iter().map(|r| r.confidence).sum::<f32>() / total_validations as f32;
        let best_result = history.iter().max_by(|a, b| {
            a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });
        let validation_frequency = if total_validations > 1 {
            let time_span = match (history.last(), history.first()) {
                (Some(last), Some(first)) => last.validated_at - first.validated_at,
                _ => chrono::Duration::zero(),
            };
            total_validations as f32 / time_span.num_hours().max(1) as f32
        } else {
            0.0
        };
        ValidationStatistics {
            total_validations,
            average_confidence,
            best_confidence: best_result.map(|r| r.confidence).unwrap_or(0.0),
            validation_frequency,
            last_validation: history.last().map(|r| r.validated_at),
        }
    }
}
/// Mean Absolute Percentage Error calculator
#[derive(Debug, Clone, Copy)]
pub struct MAPECalculator;
/// Time series validation strategy
#[derive(Debug)]
pub struct TimeSeriesValidation;
impl TimeSeriesValidation {
    pub fn new() -> Self {
        Self
    }
}
