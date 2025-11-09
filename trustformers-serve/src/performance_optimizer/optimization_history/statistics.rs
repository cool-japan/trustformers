//! Advanced Statistics Computer
//!
//! This module provides comprehensive statistical analysis capabilities for optimization
//! history data, including descriptive statistics, distribution analysis, correlation
//! analysis, time series analysis, and statistical significance testing. It enables
//! data-driven insights and sophisticated statistical modeling for optimization decisions.

use anyhow::Result;
use std::collections::HashMap;

use super::types::*;
use crate::performance_optimizer::types::PerformanceDataPoint;

// =============================================================================
// ADVANCED STATISTICS COMPUTER
// =============================================================================

/// Advanced statistics computer with comprehensive analysis capabilities
///
/// Provides sophisticated statistical analysis including descriptive statistics,
/// distribution analysis, correlation analysis, time series analysis, and
/// hypothesis testing for optimization history data.
pub struct AdvancedStatisticsComputer {
    /// Configuration for statistical analysis
    config: StatisticsConfig,
}

impl AdvancedStatisticsComputer {
    /// Create new advanced statistics computer
    pub fn new() -> Self {
        Self {
            config: StatisticsConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: StatisticsConfig) -> Self {
        Self { config }
    }

    /// Compute comprehensive statistics
    pub fn compute_comprehensive_statistics(
        &self,
        data_points: &[PerformanceDataPoint],
    ) -> Result<ComprehensiveOptimizationStatistics> {
        if data_points.is_empty() {
            return Err(anyhow::anyhow!(
                "No data points provided for statistical analysis"
            ));
        }

        let throughputs: Vec<f64> = data_points.iter().map(|p| p.throughput).collect();
        let latencies: Vec<f64> =
            data_points.iter().map(|p| p.latency.as_millis() as f64).collect();

        // Basic statistics
        let basic_stats = self.compute_basic_statistics(&throughputs)?;

        // Distribution analysis
        let distribution_analysis = if self.config.enable_distribution_analysis {
            self.analyze_distribution(&throughputs)?
        } else {
            DistributionAnalysis {
                distribution_type: DistributionType::Normal,
                parameters: HashMap::new(),
                goodness_of_fit: 0.0,
                confidence_level: 0.95,
            }
        };

        // Correlation analysis
        let correlation_analysis = if self.config.enable_correlation_analysis {
            self.analyze_correlations(data_points)?
        } else {
            CorrelationAnalysis {
                correlations: HashMap::new(),
                significance: HashMap::new(),
                correlation_matrix: Vec::new(),
            }
        };

        // Time series analysis
        let time_series_analysis = self.analyze_time_series(&throughputs)?;

        // Statistical tests
        let statistical_tests = if self.config.enable_advanced_metrics {
            self.perform_statistical_tests(&throughputs, &latencies)?
        } else {
            Vec::new()
        };

        Ok(ComprehensiveOptimizationStatistics {
            basic_stats,
            distribution_analysis,
            correlation_analysis,
            time_series_analysis,
            statistical_tests,
            analyzed_at: chrono::Utc::now(),
        })
    }

    /// Update configuration
    pub fn update_config(&mut self, new_config: StatisticsConfig) {
        self.config = new_config;
    }

    /// Compute basic descriptive statistics
    pub fn compute_basic_statistics(&self, values: &[f64]) -> Result<BasicStatistics> {
        if values.is_empty() {
            return Err(anyhow::anyhow!(
                "Cannot compute statistics for empty dataset"
            ));
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let median = compute_median(&sorted_values);
        let variance = compute_variance(values, mean);
        let std_dev = variance.sqrt();
        let min = sorted_values[0];
        let max = sorted_values[sorted_values.len() - 1];
        let range = max - min;
        let skewness = compute_skewness(values, mean, std_dev);
        let kurtosis = compute_kurtosis(values, mean, std_dev);

        Ok(BasicStatistics {
            mean,
            median,
            std_dev,
            variance,
            min,
            max,
            range,
            skewness,
            kurtosis,
        })
    }

    /// Analyze data distribution
    pub fn analyze_distribution(&self, values: &[f64]) -> Result<DistributionAnalysis> {
        if values.len() < 10 {
            return Err(anyhow::anyhow!(
                "Insufficient data for distribution analysis"
            ));
        }

        // Test for different distributions
        let normal_test = self.test_normal_distribution(values)?;
        let exponential_test = self.test_exponential_distribution(values)?;
        let uniform_test = self.test_uniform_distribution(values)?;

        // Select best fitting distribution
        let (best_distribution, best_fit, parameters) =
            if normal_test.0 > exponential_test.0 && normal_test.0 > uniform_test.0 {
                (DistributionType::Normal, normal_test.0, normal_test.1)
            } else if exponential_test.0 > uniform_test.0 {
                (
                    DistributionType::Exponential,
                    exponential_test.0,
                    exponential_test.1,
                )
            } else {
                (DistributionType::Uniform, uniform_test.0, uniform_test.1)
            };

        Ok(DistributionAnalysis {
            distribution_type: best_distribution,
            parameters,
            goodness_of_fit: best_fit,
            confidence_level: 0.95,
        })
    }

    /// Analyze correlations between metrics
    pub fn analyze_correlations(
        &self,
        data_points: &[PerformanceDataPoint],
    ) -> Result<CorrelationAnalysis> {
        if data_points.len() < 3 {
            return Err(anyhow::anyhow!(
                "Insufficient data for correlation analysis"
            ));
        }

        let throughputs: Vec<f64> = data_points.iter().map(|p| p.throughput).collect();
        let latencies: Vec<f64> =
            data_points.iter().map(|p| p.latency.as_millis() as f64).collect();
        let timestamps: Vec<f64> = data_points.iter().enumerate().map(|(i, _)| i as f64).collect();

        let mut correlations = HashMap::new();
        let mut significance = HashMap::new();

        // Throughput vs Latency
        let throughput_latency_corr = compute_correlation(&throughputs, &latencies)?;
        let corr_significance =
            compute_correlation_significance(throughput_latency_corr, data_points.len());
        correlations.insert("throughput_latency".to_string(), throughput_latency_corr);
        significance.insert("throughput_latency".to_string(), corr_significance);

        // Throughput vs Time
        let throughput_time_corr = compute_correlation(&throughputs, &timestamps)?;
        let time_significance =
            compute_correlation_significance(throughput_time_corr, data_points.len());
        correlations.insert("throughput_time".to_string(), throughput_time_corr);
        significance.insert("throughput_time".to_string(), time_significance);

        // Latency vs Time
        let latency_time_corr = compute_correlation(&latencies, &timestamps)?;
        let latency_time_significance =
            compute_correlation_significance(latency_time_corr, data_points.len());
        correlations.insert("latency_time".to_string(), latency_time_corr);
        significance.insert("latency_time".to_string(), latency_time_significance);

        // Correlation matrix
        let correlation_matrix = vec![
            vec![1.0, throughput_latency_corr, throughput_time_corr],
            vec![throughput_latency_corr, 1.0, latency_time_corr],
            vec![throughput_time_corr, latency_time_corr, 1.0],
        ];

        Ok(CorrelationAnalysis {
            correlations,
            significance,
            correlation_matrix,
        })
    }

    /// Analyze time series properties
    pub fn analyze_time_series(&self, values: &[f64]) -> Result<TimeSeriesAnalysis> {
        if values.len() < 4 {
            return Err(anyhow::anyhow!(
                "Insufficient data for time series analysis"
            ));
        }

        // Decompose time series into trend, seasonal, and residual components
        let trend = self.extract_trend_component(values);
        let seasonal = self.extract_seasonal_component(values, &trend);
        let residual = self.compute_residuals(values, &trend, &seasonal);

        // Compute autocorrelation
        let autocorrelation = self.compute_autocorrelation(values, 10.min(values.len() / 4));

        // Test for stationarity
        let stationarity_test = self.test_stationarity(values)?;

        Ok(TimeSeriesAnalysis {
            trend,
            seasonal,
            residual,
            autocorrelation,
            stationarity_test,
        })
    }

    /// Perform various statistical tests
    pub fn perform_statistical_tests(
        &self,
        throughputs: &[f64],
        latencies: &[f64],
    ) -> Result<Vec<StatisticalTest>> {
        let mut tests = Vec::new();

        // Normality test for throughput
        if let Ok((statistic, p_value)) = self.shapiro_wilk_test(throughputs) {
            tests.push(StatisticalTest {
                test_name: "Shapiro-Wilk Normality Test (Throughput)".to_string(),
                test_statistic: statistic,
                p_value,
                critical_value: 0.05,
                is_significant: p_value < 0.05,
            });
        }

        // Normality test for latency
        if let Ok((statistic, p_value)) = self.shapiro_wilk_test(latencies) {
            tests.push(StatisticalTest {
                test_name: "Shapiro-Wilk Normality Test (Latency)".to_string(),
                test_statistic: statistic,
                p_value,
                critical_value: 0.05,
                is_significant: p_value < 0.05,
            });
        }

        // Two-sample t-test comparing first and second half
        if throughputs.len() >= 4 {
            let mid_point = throughputs.len() / 2;
            let first_half = &throughputs[..mid_point];
            let second_half = &throughputs[mid_point..];

            if let Ok((statistic, p_value)) = self.two_sample_t_test(first_half, second_half) {
                tests.push(StatisticalTest {
                    test_name: "Two-Sample T-Test (Performance Change)".to_string(),
                    test_statistic: statistic,
                    p_value,
                    critical_value: 1.96, // For 95% confidence
                    is_significant: statistic.abs() > 1.96,
                });
            }
        }

        // F-test for variance comparison
        if let Ok((statistic, p_value)) = self.f_test_variance(throughputs, latencies) {
            tests.push(StatisticalTest {
                test_name: "F-Test Variance Comparison (Throughput vs Latency)".to_string(),
                test_statistic: statistic,
                p_value,
                critical_value: 2.0, // Approximate
                is_significant: p_value < 0.05,
            });
        }

        Ok(tests)
    }

    /// Test for normal distribution
    fn test_normal_distribution(&self, values: &[f64]) -> Result<(f64, HashMap<String, f64>)> {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = compute_variance(values, mean);
        let std_dev = variance.sqrt();

        // Calculate goodness of fit using chi-square test approximation
        let mut histogram = [0; 10];
        let min_val = values.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let bin_width = (max_val - min_val) / 10.0;

        for &value in values {
            let bin_index = ((value - min_val) / bin_width).min(9.0) as usize;
            histogram[bin_index] += 1;
        }

        // Calculate expected frequencies for normal distribution
        let mut chi_square = 0.0;
        for (i, &observed) in histogram.iter().enumerate() {
            let bin_start = min_val + i as f64 * bin_width;
            let bin_end = bin_start + bin_width;

            let expected = values.len() as f64
                * (normal_cdf(bin_end, mean, std_dev) - normal_cdf(bin_start, mean, std_dev));

            if expected > 0.0 {
                chi_square += (observed as f64 - expected).powi(2) / expected;
            }
        }

        let goodness_of_fit = (-chi_square / 10.0).exp(); // Normalized goodness of fit

        let mut parameters = HashMap::new();
        parameters.insert("mean".to_string(), mean);
        parameters.insert("std_dev".to_string(), std_dev);
        parameters.insert("variance".to_string(), variance);

        Ok((goodness_of_fit, parameters))
    }

    /// Test for exponential distribution
    fn test_exponential_distribution(&self, values: &[f64]) -> Result<(f64, HashMap<String, f64>)> {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let lambda = 1.0 / mean;

        // Kolmogorov-Smirnov test approximation
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut max_diff = 0.0_f64;
        for (i, &value) in sorted_values.iter().enumerate() {
            let empirical_cdf = (i + 1) as f64 / sorted_values.len() as f64;
            let theoretical_cdf = 1.0 - (-lambda * value).exp();
            // TODO: Added f64 type annotation to fix E0689 ambiguous numeric type
            max_diff = max_diff.max((empirical_cdf - theoretical_cdf).abs());
        }

        // TODO: Added f64 type annotation to fix E0689 ambiguous numeric type
        let goodness_of_fit = (-max_diff * 5.0_f64).exp(); // Approximate transformation

        let mut parameters = HashMap::new();
        parameters.insert("lambda".to_string(), lambda);
        parameters.insert("mean".to_string(), mean);

        Ok((goodness_of_fit, parameters))
    }

    /// Test for uniform distribution
    fn test_uniform_distribution(&self, values: &[f64]) -> Result<(f64, HashMap<String, f64>)> {
        let min_val = values.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let range = max_val - min_val;

        // Chi-square test for uniformity
        let mut histogram = [0; 10];
        let bin_width = range / 10.0;

        for &value in values {
            let bin_index = ((value - min_val) / bin_width).min(9.0) as usize;
            histogram[bin_index] += 1;
        }

        let expected_per_bin = values.len() as f64 / 10.0;
        let mut chi_square = 0.0;

        for &observed in &histogram {
            chi_square += (observed as f64 - expected_per_bin).powi(2) / expected_per_bin;
        }

        let goodness_of_fit = (-chi_square / 10.0).exp();

        let mut parameters = HashMap::new();
        parameters.insert("min".to_string(), min_val);
        parameters.insert("max".to_string(), max_val);
        parameters.insert("range".to_string(), range);

        Ok((goodness_of_fit, parameters))
    }

    /// Extract trend component using simple moving average
    fn extract_trend_component(&self, values: &[f64]) -> Vec<f64> {
        let window_size = (values.len() / 4).clamp(3, 10);
        let mut trend = Vec::new();

        for i in 0..values.len() {
            let start = if i >= window_size / 2 { i - window_size / 2 } else { 0 };
            let end = (i + window_size / 2 + 1).min(values.len());
            let window_mean = values[start..end].iter().sum::<f64>() / (end - start) as f64;
            trend.push(window_mean);
        }

        trend
    }

    /// Extract seasonal component (simplified)
    fn extract_seasonal_component(&self, values: &[f64], trend: &[f64]) -> Vec<f64> {
        let detrended: Vec<f64> = values.iter().zip(trend.iter()).map(|(v, t)| v - t).collect();

        // Simple seasonal extraction using periodic averaging
        let period = (values.len() / 4).max(2);
        let mut seasonal = vec![0.0; values.len()];

        for i in 0..values.len() {
            let seasonal_index = i % period;
            let mut seasonal_sum = 0.0;
            let mut seasonal_count = 0;

            for j in (seasonal_index..detrended.len()).step_by(period) {
                seasonal_sum += detrended[j];
                seasonal_count += 1;
            }

            seasonal[i] =
                if seasonal_count > 0 { seasonal_sum / seasonal_count as f64 } else { 0.0 };
        }

        seasonal
    }

    /// Compute residual component
    fn compute_residuals(&self, values: &[f64], trend: &[f64], seasonal: &[f64]) -> Vec<f64> {
        values
            .iter()
            .zip(trend.iter())
            .zip(seasonal.iter())
            .map(|((v, t), s)| v - t - s)
            .collect()
    }

    /// Compute autocorrelation function
    fn compute_autocorrelation(&self, values: &[f64], max_lag: usize) -> Vec<f64> {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = compute_variance(values, mean);

        let mut autocorr = Vec::new();

        for lag in 0..=max_lag {
            if lag >= values.len() {
                autocorr.push(0.0);
                continue;
            }

            let mut covariance = 0.0;
            let count = values.len() - lag;

            for i in 0..count {
                covariance += (values[i] - mean) * (values[i + lag] - mean);
            }

            covariance /= count as f64;
            let correlation = if variance > 0.0 { covariance / variance } else { 0.0 };
            autocorr.push(correlation);
        }

        autocorr
    }

    /// Test stationarity using augmented Dickey-Fuller test (simplified)
    fn test_stationarity(&self, values: &[f64]) -> Result<StationarityTest> {
        if values.len() < 4 {
            return Ok(StationarityTest {
                test_name: "Augmented Dickey-Fuller".to_string(),
                is_stationary: false,
                test_statistic: 0.0,
                p_value: 1.0,
            });
        }

        // Simple stationarity check using variance of differences
        let differences: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();

        let diff_mean = differences.iter().sum::<f64>() / differences.len() as f64;
        let diff_variance = compute_variance(&differences, diff_mean);

        // Simple heuristic: if the variance of differences is much smaller than the variance of values
        let value_mean = values.iter().sum::<f64>() / values.len() as f64;
        let value_variance = compute_variance(values, value_mean);

        let test_statistic =
            if value_variance > 0.0 { diff_variance / value_variance } else { 0.0 };

        let is_stationary = test_statistic < 0.1; // Simple threshold
        let p_value = if is_stationary { 0.01 } else { 0.5 };

        Ok(StationarityTest {
            test_name: "Simplified Stationarity Test".to_string(),
            is_stationary,
            test_statistic,
            p_value,
        })
    }

    /// Shapiro-Wilk test for normality (simplified approximation)
    fn shapiro_wilk_test(&self, values: &[f64]) -> Result<(f64, f64)> {
        if values.len() < 3 || values.len() > 50 {
            return Err(anyhow::anyhow!(
                "Sample size not suitable for Shapiro-Wilk test"
            ));
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let mut sum_squared_deviations = 0.0;

        for &value in values {
            sum_squared_deviations += (value - mean).powi(2);
        }

        // Simplified W statistic calculation
        let n = values.len();
        let range = sorted_values[n - 1] - sorted_values[0];
        let w_statistic = if sum_squared_deviations > 0.0 {
            (range / sum_squared_deviations.sqrt()).min(1.0)
        } else {
            1.0
        };

        // Approximate p-value transformation
        let p_value = (1.0 - w_statistic).clamp(0.001, 0.999);

        Ok((w_statistic, p_value))
    }

    /// Two-sample t-test
    fn two_sample_t_test(&self, sample1: &[f64], sample2: &[f64]) -> Result<(f64, f64)> {
        if sample1.len() < 2 || sample2.len() < 2 {
            return Err(anyhow::anyhow!("Insufficient sample sizes for t-test"));
        }

        let mean1 = sample1.iter().sum::<f64>() / sample1.len() as f64;
        let mean2 = sample2.iter().sum::<f64>() / sample2.len() as f64;

        let var1 = compute_variance(sample1, mean1);
        let var2 = compute_variance(sample2, mean2);

        let n1 = sample1.len() as f64;
        let n2 = sample2.len() as f64;

        // Pooled standard error
        let pooled_se = ((var1 / n1) + (var2 / n2)).sqrt();

        let t_statistic = if pooled_se > 0.0 { (mean1 - mean2) / pooled_se } else { 0.0 };

        // Degrees of freedom (Welch's approximation)
        let df = if var1 > 0.0 && var2 > 0.0 {
            let numerator = ((var1 / n1) + (var2 / n2)).powi(2);
            let denominator =
                ((var1 / n1).powi(2) / (n1 - 1.0)) + ((var2 / n2).powi(2) / (n2 - 1.0));
            numerator / denominator
        } else {
            n1 + n2 - 2.0
        };

        // Approximate p-value using t-distribution
        let p_value = 2.0 * (1.0 - t_cdf(t_statistic.abs(), df));

        Ok((t_statistic, p_value))
    }

    /// F-test for variance comparison
    fn f_test_variance(&self, sample1: &[f64], sample2: &[f64]) -> Result<(f64, f64)> {
        if sample1.len() < 2 || sample2.len() < 2 {
            return Err(anyhow::anyhow!("Insufficient sample sizes for F-test"));
        }

        let mean1 = sample1.iter().sum::<f64>() / sample1.len() as f64;
        let mean2 = sample2.iter().sum::<f64>() / sample2.len() as f64;

        let var1 = compute_variance(sample1, mean1);
        let var2 = compute_variance(sample2, mean2);

        let f_statistic = if var2 > 0.0 { var1 / var2 } else { f64::INFINITY };

        // Approximate p-value (simplified)
        let p_value = if f_statistic > 2.0 || f_statistic < 0.5 { 0.05 } else { 0.5 };

        Ok((f_statistic, p_value))
    }
}

impl Default for AdvancedStatisticsComputer {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Compute median of sorted values
fn compute_median(sorted_values: &[f64]) -> f64 {
    let len = sorted_values.len();
    if len % 2 == 0 {
        (sorted_values[len / 2 - 1] + sorted_values[len / 2]) / 2.0
    } else {
        sorted_values[len / 2]
    }
}

/// Compute variance
fn compute_variance(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// Compute skewness
fn compute_skewness(values: &[f64], mean: f64, std_dev: f64) -> f64 {
    if std_dev == 0.0 || values.is_empty() {
        return 0.0;
    }

    let n = values.len() as f64;
    let skewness_sum: f64 = values.iter().map(|x| ((x - mean) / std_dev).powi(3)).sum();

    skewness_sum / n
}

/// Compute kurtosis
fn compute_kurtosis(values: &[f64], mean: f64, std_dev: f64) -> f64 {
    if std_dev == 0.0 || values.is_empty() {
        return 0.0;
    }

    let n = values.len() as f64;
    let kurtosis_sum: f64 = values.iter().map(|x| ((x - mean) / std_dev).powi(4)).sum();

    (kurtosis_sum / n) - 3.0 // Excess kurtosis
}

/// Compute Pearson correlation coefficient
fn compute_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() || x.is_empty() {
        return Err(anyhow::anyhow!("Invalid input for correlation calculation"));
    }

    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_y2: f64 = y.iter().map(|yi| yi * yi).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

    if denominator == 0.0 {
        Ok(0.0)
    } else {
        Ok(numerator / denominator)
    }
}

/// Compute correlation significance
fn compute_correlation_significance(correlation: f64, sample_size: usize) -> f64 {
    if sample_size < 3 {
        return 1.0;
    }

    let df = sample_size as f64 - 2.0;
    let t_statistic = correlation * (df / (1.0 - correlation * correlation)).sqrt();

    // Approximate p-value using t-distribution
    2.0 * (1.0 - t_cdf(t_statistic.abs(), df))
}

/// Normal cumulative distribution function (CDF)
fn normal_cdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    if std_dev <= 0.0 {
        return if x >= mean { 1.0 } else { 0.0 };
    }

    let z = (x - mean) / std_dev;
    0.5 * (1.0 + erf(z / 2.0_f64.sqrt()))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// T-distribution CDF approximation
fn t_cdf(t: f64, df: f64) -> f64 {
    // Simple approximation for t-distribution CDF
    if df > 30.0 {
        // For large df, t-distribution approaches normal
        normal_cdf(t, 0.0, 1.0)
    } else {
        // Simplified approximation
        let x = t / (df + t * t).sqrt();
        0.5 + 0.5 * erf(x * (df / 2.0).sqrt())
    }
}

// =============================================================================
// STATISTICAL ANALYSIS FUNCTIONS
// =============================================================================

/// Perform descriptive statistics analysis
pub fn perform_descriptive_analysis(values: &[f64]) -> Result<HashMap<String, f64>> {
    let computer = AdvancedStatisticsComputer::new();
    let basic_stats = computer.compute_basic_statistics(values)?;

    let mut results = HashMap::new();
    results.insert("mean".to_string(), basic_stats.mean);
    results.insert("median".to_string(), basic_stats.median);
    results.insert("std_dev".to_string(), basic_stats.std_dev);
    results.insert("variance".to_string(), basic_stats.variance);
    results.insert("min".to_string(), basic_stats.min);
    results.insert("max".to_string(), basic_stats.max);
    results.insert("range".to_string(), basic_stats.range);
    results.insert("skewness".to_string(), basic_stats.skewness);
    results.insert("kurtosis".to_string(), basic_stats.kurtosis);

    Ok(results)
}

/// Calculate percentiles
pub fn calculate_percentiles(
    values: &[f64],
    percentiles: &[f64],
) -> Result<std::collections::BTreeMap<ordered_float::OrderedFloat<f64>, f64>> {
    if values.is_empty() {
        return Err(anyhow::anyhow!(
            "Cannot calculate percentiles for empty dataset"
        ));
    }

    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut results = std::collections::BTreeMap::new();

    for &percentile in percentiles {
        if percentile < 0.0 || percentile > 100.0 {
            return Err(anyhow::anyhow!("Percentile must be between 0 and 100"));
        }

        let index = (percentile / 100.0) * (sorted_values.len() - 1) as f64;
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;

        let value = if lower_index == upper_index {
            sorted_values[lower_index]
        } else {
            let weight = index - lower_index as f64;
            sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight
        };

        results.insert(ordered_float::OrderedFloat(percentile), value);
    }

    Ok(results)
}

/// Calculate confidence intervals
pub fn calculate_confidence_interval(values: &[f64], confidence_level: f64) -> Result<(f64, f64)> {
    if values.len() < 2 {
        return Err(anyhow::anyhow!(
            "Need at least 2 values for confidence interval"
        ));
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = compute_variance(values, mean);
    let std_error = (variance / values.len() as f64).sqrt();

    // Use normal distribution approximation for large samples or t-distribution for small samples
    let critical_value = if values.len() >= 30 {
        // Normal distribution
        match confidence_level {
            0.90 => 1.645,
            0.95 => 1.96,
            0.99 => 2.576,
            _ => 1.96, // Default to 95%
        }
    } else {
        // T-distribution (simplified approximation)
        match confidence_level {
            0.90 => 2.0,
            0.95 => 2.5,
            0.99 => 3.0,
            _ => 2.5, // Default to 95%
        }
    };

    let margin_of_error = critical_value * std_error;
    let lower_bound = mean - margin_of_error;
    let upper_bound = mean + margin_of_error;

    Ok((lower_bound, upper_bound))
}

/// Perform outlier detection using IQR method
pub fn detect_outliers_iqr(values: &[f64], multiplier: f64) -> Result<Vec<(usize, f64)>> {
    if values.len() < 4 {
        return Ok(Vec::new());
    }

    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q1_idx = sorted_values.len() / 4;
    let q3_idx = (3 * sorted_values.len()) / 4;
    let q1 = sorted_values[q1_idx];
    let q3 = sorted_values[q3_idx];
    let iqr = q3 - q1;

    let lower_bound = q1 - multiplier * iqr;
    let upper_bound = q3 + multiplier * iqr;

    let mut outliers = Vec::new();
    for (i, &value) in values.iter().enumerate() {
        if value < lower_bound || value > upper_bound {
            outliers.push((i, value));
        }
    }

    Ok(outliers)
}

/// Perform outlier detection using Z-score method
pub fn detect_outliers_zscore(values: &[f64], threshold: f64) -> Result<Vec<(usize, f64)>> {
    if values.len() < 2 {
        return Ok(Vec::new());
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let std_dev = compute_variance(values, mean).sqrt();

    if std_dev == 0.0 {
        return Ok(Vec::new());
    }

    let mut outliers = Vec::new();
    for (i, &value) in values.iter().enumerate() {
        let z_score = (value - mean).abs() / std_dev;
        if z_score > threshold {
            outliers.push((i, value));
        }
    }

    Ok(outliers)
}
