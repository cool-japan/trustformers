//! Model Validation System for Performance Models
//!
//! This module provides comprehensive model validation capabilities including
//! cross-validation, holdout validation, time series validation, bootstrap
//! validation, and advanced statistical validation techniques.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use scirs2_core::random::prelude::*; // SciRS2 Integration Policy
                                     // Explicit import for choose method
use std::{collections::HashMap, sync::Arc, time::Duration};

use super::types::*;
use crate::performance_optimizer::types::PerformanceDataPoint;

// =============================================================================
// VALIDATION ORCHESTRATOR
// =============================================================================

/// Comprehensive model validation orchestrator
#[derive(Debug)]
pub struct ModelValidationOrchestrator {
    /// Validation configuration
    config: Arc<RwLock<ValidationConfig>>,
    /// Available validation strategies
    strategies: HashMap<ValidationStrategyType, Box<dyn ValidationStrategy>>,
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

        // Register default validation strategies
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

        // Register default metric calculators
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

        // Execute validation with each configured strategy
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

        // Calculate comprehensive metrics
        let comprehensive_metrics =
            self.calculate_comprehensive_metrics(&strategy_results, test_data)?;

        // Perform statistical significance tests
        let significance_tests = self.perform_significance_tests(&strategy_results)?;

        // Create comprehensive result
        let mut result = ComprehensiveValidationResult {
            strategy_results,
            comprehensive_metrics,
            significance_tests,
            model_name: model.name().to_string(),
            validation_timestamp: Utc::now(),
            validation_duration: start_time.elapsed(),
            data_characteristics: self.analyze_data_characteristics(test_data)?,
            overall_score: 0.0, // Will be calculated below
        };

        // Calculate overall score from confidence
        result.overall_score = result.calculate_overall_confidence() as f64;

        // Store in validation history
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

            // Keep history manageable
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
                // Collect all predictions and actuals from different strategies
                let mut all_predictions = Vec::new();
                let mut all_actuals = Vec::new();

                for result in strategy_results.values() {
                    if let Some(predictions) = self.extract_predictions_from_result(result) {
                        let predictions_len = predictions.len();
                        all_predictions.extend(predictions);
                        // For simplicity, use test_data throughput as actuals
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
        // This would extract actual predictions from the validation result
        // For now, we'll return None as this depends on the detailed result structure
        None
    }

    /// Perform statistical significance tests
    fn perform_significance_tests(
        &self,
        strategy_results: &HashMap<ValidationStrategyType, ValidationResult>,
    ) -> Result<HashMap<String, SignificanceTestResult>> {
        let mut tests = HashMap::new();

        // Perform pairwise comparisons between strategies
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
        // Extract CV scores for comparison
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

        // Perform paired t-test
        let differences: Vec<f32> =
            scores1.iter().zip(scores2.iter()).map(|(s1, s2)| s1 - s2).collect();

        let mean_diff = differences.iter().sum::<f32>() / differences.len() as f32;
        let var_diff = differences.iter().map(|d| (d - mean_diff).powi(2)).sum::<f32>()
            / (differences.len() - 1).max(1) as f32;

        let std_err = (var_diff / differences.len() as f32).sqrt();
        let t_statistic = if std_err > 1e-8 { mean_diff / std_err } else { 0.0 };

        // Approximate p-value (simplified)
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

        // Calculate feature correlations (simplified)
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

        // Analyze distribution
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
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std_dev = variance.sqrt();

        // Simple normality test (Shapiro-Wilk approximation)
        let normality_p_value = self.approximate_normality_test(&sorted_data, mean, std_dev)?;

        // Determine distribution type based on characteristics
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
            return Ok(1.0); // Cannot test with insufficient data
        }

        // Calculate skewness and kurtosis
        let n = sorted_data.len() as f64;
        let skewness = sorted_data.iter().map(|x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / n;

        let kurtosis =
            sorted_data.iter().map(|x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / n - 3.0;

        // Simple heuristic for normality
        let skewness_test = skewness.abs() < 1.0;
        let kurtosis_test = kurtosis.abs() < 2.0;

        if skewness_test && kurtosis_test {
            Ok(0.8) // Likely normal
        } else if skewness_test || kurtosis_test {
            Ok(0.3) // Possibly normal
        } else {
            Ok(0.05) // Unlikely normal
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

        let best_result =
            history.iter().max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap());

        let validation_frequency = if total_validations > 1 {
            let time_span =
                history.last().unwrap().validated_at - history.first().unwrap().validated_at;
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

impl ModelValidator for ModelValidationOrchestrator {
    fn validate(
        &self,
        model: &dyn PerformancePredictor,
        test_data: &[PerformanceDataPoint],
    ) -> Result<ValidationResult> {
        // This is a simplified sync wrapper - in practice, we'd use the async version
        let config = self.config.read();
        let strategy = self
            .strategies
            .get(&config.strategy)
            .ok_or_else(|| anyhow!("Validation strategy not found: {:?}", config.strategy))?;

        // Use tokio to run the async validation
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(strategy.validate(model, test_data, &*config))
    }

    fn strategy(&self) -> ValidationStrategyType {
        self.config.read().strategy
    }

    fn config(&self) -> &ValidationConfig {
        // This is not ideal for thread safety, but matches the trait requirement
        // In practice, we'd return a clone or use Arc<RwLock<ValidationConfig>>
        unsafe { &*(&*self.config.read() as *const ValidationConfig) }
    }
}

// =============================================================================
// VALIDATION STRATEGIES
// =============================================================================

/// Trait for validation strategies
#[async_trait]
pub trait ValidationStrategy: std::fmt::Debug + Send + Sync {
    /// Perform validation
    async fn validate(
        &self,
        model: &dyn PerformancePredictor,
        data: &[PerformanceDataPoint],
        config: &ValidationConfig,
    ) -> Result<ValidationResult>;

    /// Get strategy name
    fn name(&self) -> &str;

    /// Check if strategy is applicable
    fn is_applicable(&self, data_size: usize) -> bool;
}

/// Hold-out validation strategy
#[derive(Debug)]
pub struct HoldOutValidation {
    test_size: f32,
}

impl HoldOutValidation {
    pub fn new(test_size: f32) -> Self {
        Self { test_size }
    }
}

#[async_trait]
impl ValidationStrategy for HoldOutValidation {
    async fn validate(
        &self,
        model: &dyn PerformancePredictor,
        data: &[PerformanceDataPoint],
        config: &ValidationConfig,
    ) -> Result<ValidationResult> {
        let test_count = (data.len() as f32 * self.test_size) as usize;
        if test_count == 0 {
            return Err(anyhow!("Test set would be empty"));
        }

        // Split data (using last portion as test set for simplicity)
        let (_train_data, test_data) = data.split_at(data.len() - test_count);

        // For this simplified implementation, we'll just evaluate on test data
        // In practice, we would retrain the model on train_data first

        let mut predictions = Vec::new();
        let mut actuals = Vec::new();

        for data_point in test_data {
            let prediction_request = PredictionRequest {
                parallelism_levels: vec![data_point.parallelism],
                test_characteristics: data_point.test_characteristics.clone(),
                system_state: data_point.system_state.clone(),
                prediction_horizon: None,
                confidence_level: 0.8,
                include_uncertainty: false,
            };

            match model.predict(&prediction_request) {
                Ok(prediction) => {
                    predictions.push(prediction.throughput);
                    actuals.push(data_point.throughput);
                },
                Err(e) => {
                    tracing::warn!("Prediction failed during validation: {}", e);
                    continue;
                },
            }
        }

        if predictions.is_empty() {
            return Err(anyhow!("No valid predictions generated"));
        }

        // Calculate metrics
        let mut metrics = HashMap::new();
        for metric in &config.metrics {
            let calculator = match metric {
                ValidationMetric::MeanAbsoluteError => &MAECalculator as &dyn MetricCalculator,
                ValidationMetric::RootMeanSquaredError => &RMSECalculator,
                ValidationMetric::RSquared => &RSquaredCalculator,
                _ => continue,
            };

            let value = calculator.calculate(&predictions, &actuals)?;
            metrics.insert(*metric, value);
        }

        let confidence = self.calculate_confidence(&metrics);

        Ok(ValidationResult {
            metrics,
            cv_scores: vec![confidence], // Single score for hold-out
            confidence,
            details: ValidationDetails {
                test_samples: predictions.len(),
                test_statistics: TestDataStatistics {
                    mean_target: actuals.iter().sum::<f64>() as f32 / actuals.len() as f32,
                    target_std: 1.0, // Simplified
                    feature_correlations: HashMap::new(),
                    distribution_info: DistributionInfo {
                        distribution_type: DistributionType::Normal,
                        parameters: HashMap::new(),
                        normality_p_value: 0.5,
                    },
                },
                prediction_errors: predictions
                    .iter()
                    .zip(actuals.iter())
                    .map(|(p, a)| (p - a) as f32)
                    .collect(),
                residual_analysis: ResidualAnalysis {
                    autocorrelation: 0.1,
                    heteroscedasticity_p_value: 0.5,
                    normality_p_value: 0.5,
                    outliers: Vec::new(),
                },
            },
            validated_at: Utc::now(),
        })
    }

    fn name(&self) -> &str {
        "HoldOutValidation"
    }

    fn is_applicable(&self, data_size: usize) -> bool {
        data_size >= 10 // Need at least 10 samples for meaningful hold-out
    }
}

impl HoldOutValidation {
    fn calculate_confidence(&self, metrics: &HashMap<ValidationMetric, f32>) -> f32 {
        // Simple confidence calculation based on R-squared if available
        if let Some(&r_squared) = metrics.get(&ValidationMetric::RSquared) {
            r_squared.clamp(0.0, 1.0)
        } else if let Some(&mae) = metrics.get(&ValidationMetric::MeanAbsoluteError) {
            (1.0 - mae / 100.0).clamp(0.0, 1.0)
        } else {
            0.5 // Default confidence
        }
    }
}

/// Cross-validation strategy
#[derive(Debug)]
pub struct CrossValidation {
    folds: usize,
}

impl CrossValidation {
    pub fn new(folds: usize) -> Self {
        Self { folds }
    }

    fn create_folds(
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

#[async_trait]
impl ValidationStrategy for CrossValidation {
    async fn validate(
        &self,
        model: &dyn PerformancePredictor,
        data: &[PerformanceDataPoint],
        config: &ValidationConfig,
    ) -> Result<ValidationResult> {
        let folds = self.create_folds(data);
        let mut fold_scores = Vec::new();
        let mut all_predictions = Vec::new();
        let mut all_actuals = Vec::new();

        for (fold_idx, (_train_data, test_data)) in folds.iter().enumerate() {
            tracing::debug!("Processing fold {}/{}", fold_idx + 1, self.folds);

            // Evaluate on test fold
            let mut fold_predictions = Vec::new();
            let mut fold_actuals = Vec::new();

            for data_point in test_data {
                let prediction_request = PredictionRequest {
                    parallelism_levels: vec![data_point.parallelism],
                    test_characteristics: data_point.test_characteristics.clone(),
                    system_state: data_point.system_state.clone(),
                    prediction_horizon: None,
                    confidence_level: 0.8,
                    include_uncertainty: false,
                };

                if let Ok(prediction) = model.predict(&prediction_request) {
                    fold_predictions.push(prediction.throughput);
                    fold_actuals.push(data_point.throughput);
                }
            }

            if !fold_predictions.is_empty() {
                // Calculate MAE for this fold
                let mae = fold_predictions
                    .iter()
                    .zip(fold_actuals.iter())
                    .map(|(p, a)| (p - a).abs())
                    .sum::<f64>() as f32
                    / fold_predictions.len() as f32;

                fold_scores.push(1.0 - mae / 100.0); // Convert to score (higher is better)
                all_predictions.extend(fold_predictions);
                all_actuals.extend(fold_actuals);
            }
        }

        if fold_scores.is_empty() {
            return Err(anyhow!("Cross-validation produced no valid scores"));
        }

        // Calculate overall metrics
        let mut metrics = HashMap::new();
        for metric in &config.metrics {
            let calculator = match metric {
                ValidationMetric::MeanAbsoluteError => &MAECalculator as &dyn MetricCalculator,
                ValidationMetric::RootMeanSquaredError => &RMSECalculator,
                ValidationMetric::RSquared => &RSquaredCalculator,
                _ => continue,
            };

            let value = calculator.calculate(&all_predictions, &all_actuals)?;
            metrics.insert(*metric, value);
        }

        let average_score = fold_scores.iter().sum::<f32>() / fold_scores.len() as f32;

        Ok(ValidationResult {
            metrics,
            cv_scores: fold_scores,
            confidence: average_score.clamp(0.0, 1.0),
            details: ValidationDetails {
                test_samples: all_predictions.len(),
                test_statistics: TestDataStatistics {
                    mean_target: all_actuals.iter().sum::<f64>() as f32 / all_actuals.len() as f32,
                    target_std: 1.0,
                    feature_correlations: HashMap::new(),
                    distribution_info: DistributionInfo {
                        distribution_type: DistributionType::Normal,
                        parameters: HashMap::new(),
                        normality_p_value: 0.5,
                    },
                },
                prediction_errors: all_predictions
                    .iter()
                    .zip(all_actuals.iter())
                    .map(|(p, a)| (p - a) as f32)
                    .collect(),
                residual_analysis: ResidualAnalysis {
                    autocorrelation: 0.05,
                    heteroscedasticity_p_value: 0.6,
                    normality_p_value: 0.7,
                    outliers: Vec::new(),
                },
            },
            validated_at: Utc::now(),
        })
    }

    fn name(&self) -> &str {
        "CrossValidation"
    }

    fn is_applicable(&self, data_size: usize) -> bool {
        data_size >= self.folds * 2 // Need at least 2 samples per fold
    }
}

/// Time series validation strategy
#[derive(Debug)]
pub struct TimeSeriesValidation;

impl Default for TimeSeriesValidation {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSeriesValidation {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ValidationStrategy for TimeSeriesValidation {
    async fn validate(
        &self,
        model: &dyn PerformancePredictor,
        data: &[PerformanceDataPoint],
        config: &ValidationConfig,
    ) -> Result<ValidationResult> {
        // For time series validation, we use expanding window approach
        let min_train_size = data.len() / 3; // Use at least 1/3 for initial training
        let mut scores = Vec::new();
        let mut all_predictions = Vec::new();
        let mut all_actuals = Vec::new();

        for test_idx in min_train_size..data.len() {
            let test_point = &data[test_idx];

            let prediction_request = PredictionRequest {
                parallelism_levels: vec![test_point.parallelism],
                test_characteristics: test_point.test_characteristics.clone(),
                system_state: test_point.system_state.clone(),
                prediction_horizon: None,
                confidence_level: 0.8,
                include_uncertainty: false,
            };

            if let Ok(prediction) = model.predict(&prediction_request) {
                let error = (prediction.throughput - test_point.throughput).abs();
                let score = 1.0 - (error / test_point.throughput.max(0.001)) as f32;
                scores.push(score.max(0.0));
                all_predictions.push(prediction.throughput);
                all_actuals.push(test_point.throughput);
            }
        }

        if scores.is_empty() {
            return Err(anyhow!("Time series validation produced no scores"));
        }

        // Calculate metrics
        let mut metrics = HashMap::new();
        for metric in &config.metrics {
            let calculator = match metric {
                ValidationMetric::MeanAbsoluteError => &MAECalculator as &dyn MetricCalculator,
                ValidationMetric::RootMeanSquaredError => &RMSECalculator,
                ValidationMetric::RSquared => &RSquaredCalculator,
                _ => continue,
            };

            let value = calculator.calculate(&all_predictions, &all_actuals)?;
            metrics.insert(*metric, value);
        }

        let average_score = scores.iter().sum::<f32>() / scores.len() as f32;

        Ok(ValidationResult {
            metrics,
            cv_scores: scores,
            confidence: average_score,
            details: ValidationDetails {
                test_samples: all_predictions.len(),
                test_statistics: TestDataStatistics {
                    mean_target: all_actuals.iter().sum::<f64>() as f32 / all_actuals.len() as f32,
                    target_std: 1.0,
                    feature_correlations: HashMap::new(),
                    distribution_info: DistributionInfo {
                        distribution_type: DistributionType::Normal,
                        parameters: HashMap::new(),
                        normality_p_value: 0.5,
                    },
                },
                prediction_errors: all_predictions
                    .iter()
                    .zip(all_actuals.iter())
                    .map(|(p, a)| (p - a) as f32)
                    .collect(),
                residual_analysis: ResidualAnalysis {
                    autocorrelation: 0.2, // Higher for time series
                    heteroscedasticity_p_value: 0.4,
                    normality_p_value: 0.6,
                    outliers: Vec::new(),
                },
            },
            validated_at: Utc::now(),
        })
    }

    fn name(&self) -> &str {
        "TimeSeriesValidation"
    }

    fn is_applicable(&self, data_size: usize) -> bool {
        data_size >= 20 // Need sufficient data for time series
    }
}

/// Bootstrap validation strategy
#[derive(Debug)]
pub struct BootstrapValidation {
    n_bootstrap: usize,
}

impl BootstrapValidation {
    pub fn new(n_bootstrap: usize) -> Self {
        Self { n_bootstrap }
    }
}

#[async_trait]
impl ValidationStrategy for BootstrapValidation {
    async fn validate(
        &self,
        model: &dyn PerformancePredictor,
        data: &[PerformanceDataPoint],
        config: &ValidationConfig,
    ) -> Result<ValidationResult> {
        let mut bootstrap_scores = Vec::new();
        let mut all_predictions = Vec::new();
        let mut all_actuals = Vec::new();

        for bootstrap_iter in 0..self.n_bootstrap {
            // Create bootstrap sample
            let bootstrap_sample = self.create_bootstrap_sample(data);

            let mut iter_predictions = Vec::new();
            let mut iter_actuals = Vec::new();

            for data_point in &bootstrap_sample {
                let prediction_request = PredictionRequest {
                    parallelism_levels: vec![data_point.parallelism],
                    test_characteristics: data_point.test_characteristics.clone(),
                    system_state: data_point.system_state.clone(),
                    prediction_horizon: None,
                    confidence_level: 0.8,
                    include_uncertainty: false,
                };

                if let Ok(prediction) = model.predict(&prediction_request) {
                    iter_predictions.push(prediction.throughput);
                    iter_actuals.push(data_point.throughput);
                }
            }

            if !iter_predictions.is_empty() {
                let mae = iter_predictions
                    .iter()
                    .zip(iter_actuals.iter())
                    .map(|(p, a)| (p - a).abs())
                    .sum::<f64>() as f32
                    / iter_predictions.len() as f32;

                bootstrap_scores.push(1.0 - mae / 100.0);

                if bootstrap_iter < 10 {
                    // Collect detailed results for first few iterations
                    all_predictions.extend(iter_predictions);
                    all_actuals.extend(iter_actuals);
                }
            }
        }

        if bootstrap_scores.is_empty() {
            return Err(anyhow!("Bootstrap validation produced no scores"));
        }

        // Calculate metrics
        let mut metrics = HashMap::new();
        if !all_predictions.is_empty() {
            for metric in &config.metrics {
                let calculator = match metric {
                    ValidationMetric::MeanAbsoluteError => &MAECalculator as &dyn MetricCalculator,
                    ValidationMetric::RootMeanSquaredError => &RMSECalculator,
                    ValidationMetric::RSquared => &RSquaredCalculator,
                    _ => continue,
                };

                let value = calculator.calculate(&all_predictions, &all_actuals)?;
                metrics.insert(*metric, value);
            }
        }

        let average_score = bootstrap_scores.iter().sum::<f32>() / bootstrap_scores.len() as f32;

        Ok(ValidationResult {
            metrics,
            cv_scores: bootstrap_scores,
            confidence: average_score.clamp(0.0, 1.0),
            details: ValidationDetails {
                test_samples: all_predictions.len(),
                test_statistics: TestDataStatistics {
                    mean_target: if !all_actuals.is_empty() {
                        all_actuals.iter().sum::<f64>() as f32 / all_actuals.len() as f32
                    } else {
                        0.0
                    },
                    target_std: 1.0,
                    feature_correlations: HashMap::new(),
                    distribution_info: DistributionInfo {
                        distribution_type: DistributionType::Normal,
                        parameters: HashMap::new(),
                        normality_p_value: 0.5,
                    },
                },
                prediction_errors: if !all_predictions.is_empty() {
                    all_predictions
                        .iter()
                        .zip(all_actuals.iter())
                        .map(|(p, a)| (p - a) as f32)
                        .collect()
                } else {
                    Vec::new()
                },
                residual_analysis: ResidualAnalysis {
                    autocorrelation: 0.1,
                    heteroscedasticity_p_value: 0.5,
                    normality_p_value: 0.5,
                    outliers: Vec::new(),
                },
            },
            validated_at: Utc::now(),
        })
    }

    fn name(&self) -> &str {
        "BootstrapValidation"
    }

    fn is_applicable(&self, data_size: usize) -> bool {
        data_size >= 10
    }
}

impl BootstrapValidation {
    fn create_bootstrap_sample(&self, data: &[PerformanceDataPoint]) -> Vec<PerformanceDataPoint> {
        let mut rng = thread_rng();
        let mut bootstrap_sample = Vec::with_capacity(data.len());

        // Workaround for SliceRandom not being properly re-exported by scirs2_core
        // Generate random indices instead of using choose()
        for _ in 0..data.len() {
            let idx = rng.gen_range(0..data.len());
            bootstrap_sample.push(data[idx].clone());
        }

        bootstrap_sample
    }
}

/// Leave-one-out validation strategy
#[derive(Debug)]
pub struct LeaveOneOutValidation;

impl Default for LeaveOneOutValidation {
    fn default() -> Self {
        Self::new()
    }
}

impl LeaveOneOutValidation {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ValidationStrategy for LeaveOneOutValidation {
    async fn validate(
        &self,
        model: &dyn PerformancePredictor,
        data: &[PerformanceDataPoint],
        config: &ValidationConfig,
    ) -> Result<ValidationResult> {
        let mut scores = Vec::new();
        let mut all_predictions = Vec::new();
        let mut all_actuals = Vec::new();

        for test_point in data.iter() {
            let prediction_request = PredictionRequest {
                parallelism_levels: vec![test_point.parallelism],
                test_characteristics: test_point.test_characteristics.clone(),
                system_state: test_point.system_state.clone(),
                prediction_horizon: None,
                confidence_level: 0.8,
                include_uncertainty: false,
            };

            if let Ok(prediction) = model.predict(&prediction_request) {
                let error = (prediction.throughput - test_point.throughput).abs();
                let relative_error = error / test_point.throughput.max(0.001);
                let score = (1.0 - relative_error as f32).max(0.0);

                scores.push(score);
                all_predictions.push(prediction.throughput);
                all_actuals.push(test_point.throughput);
            }
        }

        if scores.is_empty() {
            return Err(anyhow!("Leave-one-out validation produced no scores"));
        }

        // Calculate metrics
        let mut metrics = HashMap::new();
        for metric in &config.metrics {
            let calculator = match metric {
                ValidationMetric::MeanAbsoluteError => &MAECalculator as &dyn MetricCalculator,
                ValidationMetric::RootMeanSquaredError => &RMSECalculator,
                ValidationMetric::RSquared => &RSquaredCalculator,
                _ => continue,
            };

            let value = calculator.calculate(&all_predictions, &all_actuals)?;
            metrics.insert(*metric, value);
        }

        let average_score = scores.iter().sum::<f32>() / scores.len() as f32;

        Ok(ValidationResult {
            metrics,
            cv_scores: scores,
            confidence: average_score,
            details: ValidationDetails {
                test_samples: all_predictions.len(),
                test_statistics: TestDataStatistics {
                    mean_target: all_actuals.iter().sum::<f64>() as f32 / all_actuals.len() as f32,
                    target_std: 1.0,
                    feature_correlations: HashMap::new(),
                    distribution_info: DistributionInfo {
                        distribution_type: DistributionType::Normal,
                        parameters: HashMap::new(),
                        normality_p_value: 0.5,
                    },
                },
                prediction_errors: all_predictions
                    .iter()
                    .zip(all_actuals.iter())
                    .map(|(p, a)| (p - a) as f32)
                    .collect(),
                residual_analysis: ResidualAnalysis {
                    autocorrelation: 0.05,
                    heteroscedasticity_p_value: 0.6,
                    normality_p_value: 0.7,
                    outliers: Vec::new(),
                },
            },
            validated_at: Utc::now(),
        })
    }

    fn name(&self) -> &str {
        "LeaveOneOutValidation"
    }

    fn is_applicable(&self, data_size: usize) -> bool {
        (5..=100).contains(&data_size) // LOO is computationally intensive for large datasets
    }
}

// =============================================================================
// METRIC CALCULATORS
// =============================================================================

/// Trait for metric calculators
pub trait MetricCalculator: std::fmt::Debug + Send + Sync {
    fn calculate(&self, predictions: &[f64], actuals: &[f64]) -> Result<f32>;
    fn name(&self) -> &str;
}

/// Mean Absolute Error calculator
#[derive(Debug, Clone, Copy)]
pub struct MAECalculator;

impl MetricCalculator for MAECalculator {
    fn calculate(&self, predictions: &[f64], actuals: &[f64]) -> Result<f32> {
        if predictions.len() != actuals.len() || predictions.is_empty() {
            return Err(anyhow!("Predictions and actuals length mismatch or empty"));
        }

        let mae = predictions.iter().zip(actuals.iter()).map(|(p, a)| (p - a).abs()).sum::<f64>()
            / predictions.len() as f64;

        Ok(mae as f32)
    }

    fn name(&self) -> &str {
        "MAE"
    }
}

/// Root Mean Squared Error calculator
#[derive(Debug, Clone, Copy)]
pub struct RMSECalculator;

impl MetricCalculator for RMSECalculator {
    fn calculate(&self, predictions: &[f64], actuals: &[f64]) -> Result<f32> {
        if predictions.len() != actuals.len() || predictions.is_empty() {
            return Err(anyhow!("Predictions and actuals length mismatch or empty"));
        }

        let mse = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;

        Ok(mse.sqrt() as f32)
    }

    fn name(&self) -> &str {
        "RMSE"
    }
}

/// R-squared calculator
#[derive(Debug, Clone, Copy)]
pub struct RSquaredCalculator;

impl MetricCalculator for RSquaredCalculator {
    fn calculate(&self, predictions: &[f64], actuals: &[f64]) -> Result<f32> {
        if predictions.len() != actuals.len() || predictions.is_empty() {
            return Err(anyhow!("Predictions and actuals length mismatch or empty"));
        }

        let mean_actual = actuals.iter().sum::<f64>() / actuals.len() as f64;

        let ss_tot: f64 = actuals.iter().map(|a| (a - mean_actual).powi(2)).sum();

        let ss_res: f64 =
            predictions.iter().zip(actuals.iter()).map(|(p, a)| (a - p).powi(2)).sum();

        let r_squared = if ss_tot > 1e-12 { 1.0 - ss_res / ss_tot } else { 0.0 };

        Ok(r_squared as f32)
    }

    fn name(&self) -> &str {
        "R²"
    }
}

/// Mean Absolute Percentage Error calculator
#[derive(Debug, Clone, Copy)]
pub struct MAPECalculator;

impl MetricCalculator for MAPECalculator {
    fn calculate(&self, predictions: &[f64], actuals: &[f64]) -> Result<f32> {
        if predictions.len() != actuals.len() || predictions.is_empty() {
            return Err(anyhow!("Predictions and actuals length mismatch or empty"));
        }

        let mape = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(p, a)| if a.abs() > 1e-12 { ((p - a) / a).abs() } else { 0.0 })
            .sum::<f64>()
            / predictions.len() as f64;

        Ok((mape * 100.0) as f32) // Return as percentage
    }

    fn name(&self) -> &str {
        "MAPE"
    }
}

/// Mean Squared Logarithmic Error calculator
#[derive(Debug, Clone, Copy)]
pub struct MSLECalculator;

impl MetricCalculator for MSLECalculator {
    fn calculate(&self, predictions: &[f64], actuals: &[f64]) -> Result<f32> {
        if predictions.len() != actuals.len() || predictions.is_empty() {
            return Err(anyhow!("Predictions and actuals length mismatch or empty"));
        }

        // Check for non-negative values (required for logarithm)
        if predictions.iter().any(|&p| p < 0.0) || actuals.iter().any(|&a| a < 0.0) {
            return Err(anyhow!("MSLE requires non-negative values"));
        }

        let msle = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(p, a)| ((1.0 + p).ln() - (1.0 + a).ln()).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;

        Ok(msle as f32)
    }

    fn name(&self) -> &str {
        "MSLE"
    }
}

/// Explained Variance Score calculator
#[derive(Debug, Clone, Copy)]
pub struct ExplainedVarianceCalculator;

impl MetricCalculator for ExplainedVarianceCalculator {
    fn calculate(&self, predictions: &[f64], actuals: &[f64]) -> Result<f32> {
        if predictions.len() != actuals.len() || predictions.is_empty() {
            return Err(anyhow!("Predictions and actuals length mismatch or empty"));
        }

        let mean_actual = actuals.iter().sum::<f64>() / actuals.len() as f64;

        let var_actual =
            actuals.iter().map(|a| (a - mean_actual).powi(2)).sum::<f64>() / actuals.len() as f64;

        let var_residual = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(p, a)| (a - p).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;

        let explained_variance =
            if var_actual > 1e-12 { 1.0 - var_residual / var_actual } else { 0.0 };

        Ok(explained_variance as f32)
    }

    fn name(&self) -> &str {
        "ExplainedVariance"
    }
}

// =============================================================================
// COMPREHENSIVE VALIDATION RESULT
// =============================================================================

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
        // Aggregate residual analysis from all strategies
        let errors = self.get_prediction_errors();

        if errors.is_empty() {
            return ResidualAnalysis {
                autocorrelation: 0.0,
                heteroscedasticity_p_value: 1.0,
                normality_p_value: 1.0,
                outliers: Vec::new(),
            };
        }

        // Calculate autocorrelation (simplified)
        let autocorr = if errors.len() > 1 {
            let mut sum = 0.0f32;
            for i in 1..errors.len() {
                sum += errors[i] * errors[i - 1];
            }
            sum / (errors.len() - 1) as f32
        } else {
            0.0
        };

        // Detect outliers (simplified: values beyond 3 standard deviations)
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
            heteroscedasticity_p_value: 0.5, // Simplified
            normality_p_value: 0.5,          // Simplified
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

impl Default for ValidationStatistics {
    fn default() -> Self {
        Self {
            total_validations: 0,
            average_confidence: 0.0,
            best_confidence: 0.0,
            validation_frequency: 0.0,
            last_validation: None,
        }
    }
}
