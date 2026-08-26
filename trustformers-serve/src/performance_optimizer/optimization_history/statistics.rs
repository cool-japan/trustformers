//! Advanced Statistics Computer
//!
//! This module provides comprehensive statistical analysis capabilities for optimization
//! history data, including descriptive statistics, distribution analysis, correlation
//! analysis, time series analysis, and statistical significance testing. It enables
//! data-driven insights and sophisticated statistical modeling for optimization decisions.

use anyhow::Result;
use std::collections::HashMap;

use super::types::*;
use crate::performance_optimizer::real_time_metrics::analytics::analyzers::series::{
    beta_inc, chi_square_sf, ks_p_value, ks_statistic, normal_cdf, pearson_p_value, sorted_finite,
    student_t_two_sided,
};
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
        if let Ok((statistic, p_value)) = self.ks_normality_test(throughputs) {
            tests.push(StatisticalTest {
                test_name: "Kolmogorov-Smirnov Normality Test (Throughput)".to_string(),
                test_statistic: statistic,
                p_value,
                critical_value: 0.05,
                is_significant: p_value < 0.05,
            });
        }

        // Normality test for latency
        if let Ok((statistic, p_value)) = self.ks_normality_test(latencies) {
            tests.push(StatisticalTest {
                test_name: "Kolmogorov-Smirnov Normality Test (Latency)".to_string(),
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

        // F-test comparing the throughput variance of the two halves.
        //
        // 0.2.1: this compared the variance of `throughputs` (tests/second)
        // against the variance of `latencies` (milliseconds) and published the
        // ratio as "F-Test Variance Comparison (Throughput vs Latency)". The
        // two series carry different units, so their variance ratio has no
        // interpretation whatever value it takes. The comparison now mirrors
        // the t-test above: the same quantity, measured over two halves of the
        // same run.
        if throughputs.len() >= 4 {
            let mid_point = throughputs.len() / 2;
            let first_half = &throughputs[..mid_point];
            let second_half = &throughputs[mid_point..];

            if let Ok((statistic, p_value)) = self.f_test_variance(first_half, second_half) {
                tests.push(StatisticalTest {
                    test_name: "F-Test Variance Change (Throughput, First vs Second Half)"
                        .to_string(),
                    test_statistic: statistic,
                    p_value,
                    critical_value: 0.05,
                    is_significant: p_value < 0.05,
                });
            }
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
                * (normal_cdf_at(bin_end, mean, std_dev) - normal_cdf_at(bin_start, mean, std_dev));

            if expected > 0.0 {
                chi_square += (observed as f64 - expected).powi(2) / expected;
            }
        }

        // Goodness of fit is the chi-square test's p-value: the probability of
        // seeing a discrepancy at least this large if the data really were
        // normal. Ten bins less one constraint less the two estimated
        // parameters leaves seven degrees of freedom.
        //
        // 0.2.1: this was `exp(-chi2 / 10)`, a monotone rescaling of the
        // statistic with no distributional meaning; `analyze_distribution`
        // compares the three fits by this number, so the comparison was
        // between three differently-scaled quantities.
        let goodness_of_fit = chi_square_sf(chi_square, 7.0);

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

        // Kolmogorov-Smirnov statistic against the fitted exponential CDF,
        // taken over both edges of each step as the definition requires.
        let max_diff = ks_statistic(&sorted_values, |x| 1.0 - (-lambda * x).exp()).unwrap_or(1.0);

        // Goodness of fit is the Kolmogorov p-value of that statistic, so it is
        // on the same scale as the other two candidate distributions.
        // 0.2.1: this was `exp(-5 * D)`, an arbitrary rescaling.
        let goodness_of_fit = ks_p_value(max_diff, sorted_values.len());

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

        // Ten bins less one constraint less the two estimated endpoints leaves
        // seven degrees of freedom. 0.2.1: `exp(-chi2 / 10)`.
        let goodness_of_fit = chi_square_sf(chi_square, 7.0);

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
            let start = i.saturating_sub(window_size / 2);
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

    /// Augmented Dickey-Fuller test for a unit root, with a constant term.
    ///
    /// Fits `Δy_t = α + γ·y_{t-1} + Σ δ_i·Δy_{t-i} + ε_t` by ordinary least
    /// squares and reports `τ = γ̂ / se(γ̂)`. The null hypothesis is `γ = 0`
    /// (a unit root, so a non-stationary series); a sufficiently negative τ
    /// rejects it.
    ///
    /// The p-value is a Monte-Carlo p-value: the same regression is run on
    /// [`DICKEY_FULLER_REPLICATES`] driftless random walks generated under the
    /// null and the p-value is the rank of the observed τ among them. Under the
    /// unit-root null, τ is **not** t-distributed -- that is the whole point of
    /// the Dickey-Fuller test -- so referring it to a t or normal distribution
    /// would produce a wrong number that looks rigorous. Simulating the null
    /// needs no distribution table and is exact up to the Monte-Carlo error of
    /// roughly `sqrt(p(1-p)/B)` (about 0.01 near p = 0.05). The generator is
    /// seeded with a constant, so the same series always yields the same
    /// p-value.
    ///
    /// ## Changed in 0.2.1
    ///
    /// This computed `var(Δy) / var(y)`, called a series "stationary" when that
    /// ratio fell below 0.1, and then reported `p_value = if is_stationary
    /// { 0.01 } else { 0.5 }` -- a p-value with two possible values, neither of
    /// them derived from a distribution, published under the name "Augmented
    /// Dickey-Fuller".
    ///
    /// # Errors
    ///
    /// Returns an error when the series is too short to fit the regression at
    /// all, or when the regressors are collinear (a perfectly constant series).
    fn test_stationarity(&self, values: &[f64]) -> Result<StationarityTest> {
        // Shorter lag orders are tried in turn: a series whose differences are
        // exactly constant (a noiseless straight line) makes the lagged
        // difference collinear with the intercept, and a shorter regression is
        // the right answer there rather than a refusal.
        let mut fitted = None;
        for lags in (0..=adf_lag_order(values.len())).rev() {
            if let Some(statistic) = augmented_dickey_fuller_tau(values, lags) {
                fitted = Some((lags, statistic));
                break;
            }
        }

        let Some((lags, statistic)) = fitted else {
            // A perfectly deterministic series leaves no residual variance, so
            // `se(gamma)` is zero and tau is genuinely undefined. NaN says
            // that: no comparison passes on it, so an untestable series cannot
            // read as a tested one. `is_stationary` stays false because a unit
            // root that was never tested was never rejected.
            return Ok(StationarityTest {
                test_name: "Augmented Dickey-Fuller (not computable: the series leaves no                             residual variance to estimate a standard error from)"
                    .to_string(),
                is_stationary: false,
                test_statistic: f64::NAN,
                p_value: f64::NAN,
            });
        };

        let p_value = dickey_fuller_p_value(statistic, values.len(), lags);

        Ok(StationarityTest {
            test_name: format!("Augmented Dickey-Fuller (constant, {lags} lags)"),
            // A unit root is rejected -- so the series is called stationary --
            // only when the observed tau is in the lower tail of the simulated
            // null at the conventional 5% level.
            is_stationary: p_value < 0.05,
            test_statistic: statistic,
            p_value,
        })
    }

    /// One-sample Kolmogorov-Smirnov test of `values` against the normal
    /// distribution fitted to them, returning `(D, p)`.
    ///
    /// ## Changed in 0.2.1
    ///
    /// This was a second `shapiro_wilk_test`, published as "Shapiro-Wilk
    /// Normality Test". It computed neither a Shapiro-Wilk W (which needs the
    /// tabulated `a_i` coefficients this crate does not carry) nor a p-value:
    /// the statistic was `range / sqrt(SS)` clamped to 1, and the "p-value" was
    /// `1 - W`. Both numbers grew and shrank with the sample's *scale*, not
    /// with its normality. The identical defect was fixed in
    /// `real_time_metrics::analytics` earlier in 0.2.1; this reuses the
    /// Kolmogorov-Smirnov machinery that fix introduced.
    ///
    /// Because the mean and standard deviation are estimated from the same
    /// sample, the asymptotic Kolmogorov p-value is conservative -- an exact
    /// level would need a Lilliefors correction, which this crate does not
    /// carry.
    ///
    /// # Errors
    ///
    /// Returns an error for fewer than three finite samples, or for a sample
    /// with no spread: neither supports a distributional test.
    fn ks_normality_test(&self, values: &[f64]) -> Result<(f64, f64)> {
        let sorted = sorted_finite(values);
        if sorted.len() < 3 {
            return Err(anyhow::anyhow!(
                "Kolmogorov-Smirnov normality test needs at least 3 finite samples, got {}",
                sorted.len()
            ));
        }

        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let std_dev = compute_variance(&sorted, mean).sqrt();
        if std_dev.is_nan() || std_dev <= 0.0 {
            return Err(anyhow::anyhow!(
                "Kolmogorov-Smirnov normality test needs a sample with non-zero spread"
            ));
        }

        let statistic =
            ks_statistic(&sorted, |x| normal_cdf((x - mean) / std_dev)).ok_or_else(|| {
                anyhow::anyhow!("Kolmogorov-Smirnov statistic is undefined for this sample")
            })?;

        Ok((statistic, ks_p_value(statistic, sorted.len())))
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

        // Two-sided p-value from the t distribution with Welch's degrees of
        // freedom. 0.2.1: this used a local `t_cdf` whose small-sample branch
        // was `0.5 + 0.5*erf(x*sqrt(df/2))`, an expression that is not the t
        // CDF for any df.
        let p_value = student_t_two_sided(t_statistic, df);

        Ok((t_statistic, p_value))
    }

    /// Two-sided F-test comparing the variances of two samples.
    ///
    /// Returns `(F, p)` where `F = s1² / s2²` and `p` is
    /// `2·min(P(F ≤ f), P(F ≥ f))` under `F(n1-1, n2-1)`, computed from the
    /// regularized incomplete beta function.
    ///
    /// ## Changed in 0.2.1
    ///
    /// The p-value was `if !(0.5..=2.0).contains(&f) { 0.05 } else { 0.5 }`:
    /// two constants selected by a threshold on the statistic, published as a
    /// p-value. Every caller testing `p < 0.05` therefore got `false` for every
    /// input, whatever the variances did.
    ///
    /// # Errors
    ///
    /// Returns an error when either sample has fewer than two observations, or
    /// when the second sample has no variance (the ratio is undefined).
    fn f_test_variance(&self, sample1: &[f64], sample2: &[f64]) -> Result<(f64, f64)> {
        if sample1.len() < 2 || sample2.len() < 2 {
            return Err(anyhow::anyhow!("Insufficient sample sizes for F-test"));
        }

        let mean1 = sample1.iter().sum::<f64>() / sample1.len() as f64;
        let mean2 = sample2.iter().sum::<f64>() / sample2.len() as f64;

        // Unbiased (n-1) variances: the F ratio is defined on sample variances.
        let n1 = sample1.len() as f64;
        let n2 = sample2.len() as f64;
        let var1 = compute_variance(sample1, mean1) * n1 / (n1 - 1.0);
        let var2 = compute_variance(sample2, mean2) * n2 / (n2 - 1.0);

        if var2.is_nan() || var2 <= 0.0 {
            return Err(anyhow::anyhow!(
                "F-test is undefined: the second sample has no variance"
            ));
        }

        let f_statistic = var1 / var2;
        let p_value = f_two_sided_p_value(f_statistic, n1 - 1.0, n2 - 1.0);

        Ok((f_statistic, p_value))
    }
}

// =============================================================================
// DISTRIBUTIONAL MACHINERY
// =============================================================================

/// Number of simulated null series behind the Dickey-Fuller p-value.
///
/// 499 keeps the Monte-Carlo standard error near 0.01 at p = 0.05 while
/// costing a few milliseconds per call.
const DICKEY_FULLER_REPLICATES: usize = 499;

/// Seed of the generator that draws the simulated null series.
///
/// Constant, so the same input series always produces the same p-value.
const DICKEY_FULLER_SEED: u64 = 0x005D_FADF_5EED;

/// Longest series simulated when calibrating the Dickey-Fuller null.
///
/// The finite-sample τ distribution is within Monte-Carlo error of its
/// asymptotic limit well before this length, so calibrating a very long series
/// at its own length would cost time without moving the answer.
const DICKEY_FULLER_MAX_SIMULATION_LENGTH: usize = 512;

/// Lag order for the augmented Dickey-Fuller regression.
///
/// `floor(cbrt(n - 1))` is the usual rule of thumb; it is capped so that the
/// regression keeps at least twice as many observations as parameters.
fn adf_lag_order(len: usize) -> usize {
    if len < 4 {
        return 0;
    }
    let rule_of_thumb = ((len - 1) as f64).cbrt().floor() as usize;
    // observations = len - lags - 1, parameters = lags + 2.
    let mut lags = rule_of_thumb;
    while lags > 0 && len < 3 * lags + 5 {
        lags -= 1;
    }
    lags
}

/// τ statistic of the augmented Dickey-Fuller regression with a constant.
///
/// Returns `None` when the series is too short for the requested lag order or
/// the normal equations are singular (a perfectly constant series).
fn augmented_dickey_fuller_tau(values: &[f64], lags: usize) -> Option<f64> {
    let n = values.len();
    if n < lags + 3 {
        return None;
    }

    let differences: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();

    // Row t uses Δy_t, y_{t-1} and Δy_{t-1}..Δy_{t-lags}; the first usable t is
    // `lags + 1` in the level index, i.e. `lags` in the difference index.
    let parameters = lags + 2;
    let observations = differences.len().checked_sub(lags)?;
    if observations <= parameters {
        return None;
    }

    let mut design = Vec::with_capacity(observations);
    let mut response = Vec::with_capacity(observations);
    for row in lags..differences.len() {
        let mut regressors = Vec::with_capacity(parameters);
        regressors.push(1.0);
        regressors.push(*values.get(row)?);
        for lag in 1..=lags {
            regressors.push(*differences.get(row - lag)?);
        }
        design.push(regressors);
        response.push(*differences.get(row)?);
    }

    // Normal equations X'X b = X'y, solved together with X'X's inverse so the
    // standard error of the gamma coefficient is available.
    let mut normal = vec![vec![0.0f64; parameters]; parameters];
    let mut moment = vec![0.0f64; parameters];
    for (row, y) in design.iter().zip(response.iter()) {
        for i in 0..parameters {
            moment[i] += row[i] * y;
            for j in 0..parameters {
                normal[i][j] += row[i] * row[j];
            }
        }
    }

    let (coefficients, inverse) = solve_with_inverse(&normal, &moment)?;

    let mut residual_sum_of_squares = 0.0;
    for (row, y) in design.iter().zip(response.iter()) {
        let fitted: f64 = row.iter().zip(coefficients.iter()).map(|(x, b)| x * b).sum();
        residual_sum_of_squares += (y - fitted).powi(2);
    }
    let degrees_of_freedom = (observations - parameters) as f64;
    let residual_variance = residual_sum_of_squares / degrees_of_freedom;

    // Index 1 is the coefficient on the lagged level, which is gamma.
    let gamma = *coefficients.get(1)?;
    let gamma_variance = residual_variance * *inverse.get(1)?.get(1)?;
    if gamma_variance.is_nan() || gamma_variance <= 0.0 {
        return None;
    }

    Some(gamma / gamma_variance.sqrt())
}

/// Monte-Carlo p-value of `tau` against the unit-root null.
///
/// Driftless random walks of the same length are simulated, the same regression
/// is run on each, and the p-value is `(1 + #{τ_sim ≤ τ}) / (B + 1)`. That form
/// is never zero, which is correct: a finite simulation cannot establish a
/// p-value below `1 / (B + 1)`.
fn dickey_fuller_p_value(tau: f64, len: usize, lags: usize) -> f64 {
    let simulation_length = len.min(DICKEY_FULLER_MAX_SIMULATION_LENGTH);
    let mut rng = fastrand::Rng::with_seed(DICKEY_FULLER_SEED);
    let mut at_least_as_extreme = 0usize;

    for _ in 0..DICKEY_FULLER_REPLICATES {
        let mut walk = Vec::with_capacity(simulation_length);
        let mut level = 0.0f64;
        for _ in 0..simulation_length {
            level += standard_normal(&mut rng);
            walk.push(level);
        }
        if let Some(simulated) = augmented_dickey_fuller_tau(&walk, lags) {
            if simulated <= tau {
                at_least_as_extreme += 1;
            }
        }
    }

    (at_least_as_extreme as f64 + 1.0) / (DICKEY_FULLER_REPLICATES as f64 + 1.0)
}

/// One standard normal draw by the Box-Muller transform.
fn standard_normal(rng: &mut fastrand::Rng) -> f64 {
    // `f64()` yields [0, 1); the log needs a strictly positive argument.
    let u1 = rng.f64().max(f64::MIN_POSITIVE);
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Solve `a·x = b` and return both the solution and `a`'s inverse.
///
/// Gauss-Jordan elimination with partial pivoting on `[a | b | I]`. Returns
/// `None` when `a` is singular to working precision.
fn solve_with_inverse(a: &[Vec<f64>], b: &[f64]) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = a.len();
    if n == 0 || b.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }

    // Augmented rows: n coefficients, then the right-hand side, then identity.
    let mut work = vec![vec![0.0f64; 2 * n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            work[i][j] = a[i][j];
        }
        work[i][n] = b[i];
        work[i][n + 1 + i] = 1.0;
    }

    for column in 0..n {
        let mut pivot_row = column;
        for row in column + 1..n {
            if work[row][column].abs() > work[pivot_row][column].abs() {
                pivot_row = row;
            }
        }
        if work[pivot_row][column].abs() < 1e-12 {
            return None;
        }
        work.swap(column, pivot_row);

        let pivot = work[column][column];
        for value in work[column].iter_mut() {
            *value /= pivot;
        }
        for row in 0..n {
            if row == column {
                continue;
            }
            let factor = work[row][column];
            if factor == 0.0 {
                continue;
            }
            for index in 0..2 * n + 1 {
                work[row][index] -= factor * work[column][index];
            }
        }
    }

    let solution: Vec<f64> = (0..n).map(|i| work[i][n]).collect();
    let inverse: Vec<Vec<f64>> =
        (0..n).map(|i| (0..n).map(|j| work[i][n + 1 + j]).collect()).collect();
    Some((solution, inverse))
}

/// Cumulative distribution function of the F distribution.
///
/// `P(F ≤ f) = I_{d1·f / (d1·f + d2)}(d1/2, d2/2)`, the standard relation
/// between the F distribution and the regularized incomplete beta function.
fn f_cdf(f: f64, df1: f64, df2: f64) -> f64 {
    if !f.is_finite() || f <= 0.0 || df1 <= 0.0 || df2 <= 0.0 {
        return 0.0;
    }
    let x = df1 * f / (df1 * f + df2);
    beta_inc(df1 / 2.0, df2 / 2.0, x).clamp(0.0, 1.0)
}

/// Two-sided Student-t critical value for `confidence_level` at `df` degrees of
/// freedom.
///
/// Found by bisecting [`student_t_two_sided`], which is strictly decreasing in
/// `t`; the bracket is widened until it contains the root, so no quantile table
/// is needed and any level in `(0, 1)` is supported.
///
/// # Errors
///
/// Returns an error for a confidence level outside `(0, 1)` or for
/// non-positive degrees of freedom.
fn student_t_critical_value(confidence_level: f64, df: f64) -> Result<f64> {
    if !(0.0..1.0).contains(&confidence_level) || confidence_level <= 0.0 {
        return Err(anyhow::anyhow!(
            "confidence level must lie strictly between 0 and 1, got {confidence_level}"
        ));
    }
    if df.is_nan() || df <= 0.0 {
        return Err(anyhow::anyhow!(
            "a confidence interval needs at least one degree of freedom"
        ));
    }

    let target = 1.0 - confidence_level;
    let mut low = 0.0f64;
    let mut high = 1.0f64;
    // The tail probability falls as t grows; widen until it is below target.
    while student_t_two_sided(high, df) > target && high < 1e6 {
        high *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if student_t_two_sided(mid, df) > target {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(0.5 * (low + high))
}

/// Two-sided p-value of an F statistic: `2·min(P(F ≤ f), P(F ≥ f))`.
fn f_two_sided_p_value(f: f64, df1: f64, df2: f64) -> f64 {
    let lower = f_cdf(f, df1, df2);
    (2.0 * lower.min(1.0 - lower)).clamp(0.0, 1.0)
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
    if len.is_multiple_of(2) {
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

    // Two-sided p-value of the Pearson coefficient over `sample_size` pairs.
    // 0.2.1: this went through a local `t_cdf` whose small-sample branch was
    // not the t CDF for any degrees of freedom.
    pearson_p_value(correlation, sample_size)
}

/// Normal CDF of `x` under `N(mean, std_dev²)`.
///
/// A degenerate distribution (no spread) puts all its mass at the mean, so the
/// CDF is a step there.
fn normal_cdf_at(x: f64, mean: f64, std_dev: f64) -> f64 {
    if std_dev <= 0.0 {
        return if x >= mean { 1.0 } else { 0.0 };
    }
    normal_cdf((x - mean) / std_dev)
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
        if !(0.0..=100.0).contains(&percentile) {
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

    // Student-t critical value for the requested level, at n-1 degrees of
    // freedom. This is exact for every sample size and every level, and
    // converges to the normal quantile as n grows.
    //
    // 0.2.1: two hardcoded lookup tables selected by an equality match on the
    // confidence level, so 0.975 silently became 0.95, and the small-sample
    // branch used 2.0/2.5/3.0 -- numbers that match no t quantile at any
    // degrees of freedom.
    let critical_value = student_t_critical_value(confidence_level, values.len() as f64 - 1.0)?;

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

#[cfg(test)]
#[path = "statistics_tests.rs"]
mod statistics_tests;

#[cfg(test)]
mod honesty_tests {
    use super::*;

    fn seeded_noise(seed: u64, count: usize) -> Vec<f64> {
        let mut rng = fastrand::Rng::with_seed(seed);
        (0..count).map(|_| standard_normal(&mut rng)).collect()
    }

    fn seeded_random_walk(seed: u64, count: usize) -> Vec<f64> {
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut level = 0.0;
        (0..count)
            .map(|_| {
                level += standard_normal(&mut rng);
                level
            })
            .collect()
    }

    /// Regression: the "Shapiro-Wilk" statistic was `range / sqrt(SS)`, which
    /// is a function of the sample's *scale*, and the "p-value" was `1 - W`.
    /// A real Kolmogorov-Smirnov statistic is scale- and shift-invariant.
    #[test]
    fn the_normality_statistic_is_scale_invariant() {
        let computer = AdvancedStatisticsComputer::new();
        let sample = seeded_noise(11, 60);
        let scaled: Vec<f64> = sample.iter().map(|v| v * 1000.0 + 7.0).collect();

        let (statistic, p_value) =
            computer.ks_normality_test(&sample).expect("60 samples support the test");
        let (scaled_statistic, scaled_p) =
            computer.ks_normality_test(&scaled).expect("60 samples support the test");

        assert!(
            (statistic - scaled_statistic).abs() < 1e-9,
            "rescaling the sample must not move the statistic: {statistic} vs {scaled_statistic}"
        );
        assert!(
            (p_value - scaled_p).abs() < 1e-9,
            "rescaling must not move the p-value: {p_value} vs {scaled_p}"
        );
        assert!(
            (p_value - (1.0 - statistic)).abs() > 1e-6,
            "the p-value is no longer `1 - statistic`: p = {p_value}, D = {statistic}"
        );
    }

    /// The test tells a normal sample from one that is plainly not normal.
    #[test]
    fn the_normality_test_discriminates() {
        let computer = AdvancedStatisticsComputer::new();

        let normal = seeded_noise(23, 200);
        let (_, normal_p) = computer.ks_normality_test(&normal).expect("test runs");

        // A hard two-point mixture.
        let bimodal: Vec<f64> = (0..200).map(|i| if i % 2 == 0 { -10.0 } else { 10.0 }).collect();
        let (_, bimodal_p) = computer.ks_normality_test(&bimodal).expect("test runs");

        assert!(
            normal_p > 0.05,
            "a normal sample must not be rejected: p = {normal_p}"
        );
        assert!(
            bimodal_p < 0.01,
            "a two-point mixture must be rejected: p = {bimodal_p}"
        );

        // The old implementation refused every sample above 50 observations.
        assert!(computer.ks_normality_test(&seeded_noise(5, 400)).is_ok());
        // Guards that remain: too few samples, and no spread at all.
        assert!(computer.ks_normality_test(&[1.0, 2.0]).is_err());
        assert!(computer.ks_normality_test(&[3.0; 20]).is_err());
    }

    /// Regression: the F-test p-value was `if !(0.5..=2.0).contains(&f) { 0.05 }
    /// else { 0.5 }`, so `p < 0.05` was false for every input ever.
    #[test]
    fn the_f_test_p_value_comes_from_the_f_distribution() {
        // The median of F(1, 1) is 1, so the CDF there is exactly one half.
        assert!(
            (f_cdf(1.0, 1.0, 1.0) - 0.5).abs() < 1e-6,
            "{}",
            f_cdf(1.0, 1.0, 1.0)
        );
        assert!(
            (f_two_sided_p_value(1.0, 1.0, 1.0) - 1.0).abs() < 1e-6,
            "{}",
            f_two_sided_p_value(1.0, 1.0, 1.0)
        );

        let computer = AdvancedStatisticsComputer::new();

        // Same spread in both halves: the variance ratio is not significant.
        let same_a = seeded_noise(31, 80);
        let same_b = seeded_noise(32, 80);
        let (_, same_p) = computer.f_test_variance(&same_a, &same_b).expect("F-test runs");
        assert!(
            same_p > 0.05,
            "equal variances must not be flagged: p = {same_p}"
        );

        // One half twenty times as volatile as the other.
        let wide: Vec<f64> = seeded_noise(33, 80).iter().map(|v| v * 20.0).collect();
        let (statistic, different_p) =
            computer.f_test_variance(&wide, &same_b).expect("F-test runs");
        assert!(
            different_p < 0.001,
            "a twenty-fold variance ratio must be flagged: F = {statistic}, p = {different_p}"
        );
        assert!(
            (different_p - 0.05).abs() > 1e-9 && (same_p - 0.5).abs() > 1e-9,
            "no longer one of the two hardcoded constants: {different_p}, {same_p}"
        );
    }

    /// Regression: stationarity was `var(diff)/var(level) < 0.1` with
    /// `p_value = if is_stationary { 0.01 } else { 0.5 }`.
    #[test]
    fn stationarity_separates_a_random_walk_from_white_noise() {
        let computer = AdvancedStatisticsComputer::new();

        let noise = seeded_noise(101, 200);
        let noise_test = computer.test_stationarity(&noise).expect("the regression fits");
        assert!(
            noise_test.is_stationary,
            "white noise has no unit root: {noise_test:?}"
        );
        assert!(
            noise_test.test_statistic < -3.0,
            "tau must be deep in the lower tail: {}",
            noise_test.test_statistic
        );

        let walk = seeded_random_walk(202, 200);
        let walk_test = computer.test_stationarity(&walk).expect("the regression fits");
        assert!(
            !walk_test.is_stationary,
            "a random walk has a unit root: {walk_test:?}"
        );
        assert!(
            walk_test.p_value > noise_test.p_value,
            "the walk must be less significant than the noise: {} vs {}",
            walk_test.p_value,
            noise_test.p_value
        );

        for test in [&noise_test, &walk_test] {
            assert!(
                (test.p_value - 0.01).abs() > 1e-9 && (test.p_value - 0.5).abs() > 1e-9,
                "no longer one of the two hardcoded constants: {test:?}"
            );
            assert!(
                test.test_name.starts_with("Augmented Dickey-Fuller"),
                "the published name must be the test that ran: {}",
                test.test_name
            );
        }
    }

    /// The same series always yields the same p-value: the Monte-Carlo null is
    /// drawn from a constant seed.
    #[test]
    fn the_stationarity_p_value_is_reproducible() {
        let computer = AdvancedStatisticsComputer::new();
        let series = seeded_noise(7, 120);
        let first = computer.test_stationarity(&series).expect("the regression fits");
        let second = computer.test_stationarity(&series).expect("the regression fits");
        assert!(
            (first.p_value - second.p_value).abs() < f64::EPSILON,
            "{first:?}"
        );
    }

    /// Regression: the critical value came from two lookup tables matched on
    /// the confidence level by equality, so 0.975 silently became 0.95 and
    /// small samples used 2.0/2.5/3.0 -- no t quantile at any df.
    #[test]
    fn the_confidence_interval_uses_a_real_t_quantile() {
        // t(0.975, 4) = 2.776445, t(0.975, 9) = 2.262157, z(0.975) = 1.959964.
        let four = student_t_critical_value(0.95, 4.0).expect("valid level");
        let nine = student_t_critical_value(0.95, 9.0).expect("valid level");
        let huge = student_t_critical_value(0.95, 100_000.0).expect("valid level");
        assert!((four - 2.7764).abs() < 1e-3, "{four}");
        assert!((nine - 2.2622).abs() < 1e-3, "{nine}");
        assert!((huge - 1.9600).abs() < 1e-3, "{huge}");

        // A level the old lookup tables did not carry is now honoured rather
        // than silently collapsed onto 95%.
        let ninety_seven_five = student_t_critical_value(0.975, 9.0).expect("valid level");
        assert!(
            ninety_seven_five > nine,
            "a tighter level needs a wider interval: {ninety_seven_five} vs {nine}"
        );

        assert!(student_t_critical_value(1.0, 9.0).is_err());
        assert!(student_t_critical_value(0.95, 0.0).is_err());
    }

    /// The published test names describe the tests that actually ran.
    #[test]
    fn the_published_test_names_match_the_tests_that_ran() {
        let computer = AdvancedStatisticsComputer::new();
        let throughputs = seeded_noise(55, 60);
        let latencies: Vec<f64> = seeded_noise(56, 60).iter().map(|v| v.abs() * 10.0).collect();
        let tests = computer
            .perform_statistical_tests(&throughputs, &latencies)
            .expect("statistical tests run");

        let names: Vec<&str> = tests.iter().map(|t| t.test_name.as_str()).collect();
        assert!(
            !names.iter().any(|name| name.contains("Shapiro-Wilk")),
            "no Shapiro-Wilk W is computed anywhere in this crate: {names:?}"
        );
        assert!(
            names
                .iter()
                .any(|name| name.contains("Kolmogorov-Smirnov Normality Test (Throughput)")),
            "{names:?}"
        );
        assert!(
            !names.iter().any(|name| name.contains("Throughput vs Latency")),
            "tests/second and milliseconds are not commensurable: {names:?}"
        );
    }
}
