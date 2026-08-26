//! Measured test-set statistics and residual diagnostics.
//!
//! ## History
//!
//! All five [`ValidationStrategy`] implementations
//! (`HoldOutValidation`, `CrossValidation`, `TimeSeriesValidation`,
//! `BootstrapValidation`, `LeaveOneOutValidation`) built their
//! [`ValidationDetails`] from literals: `target_std: 1.0`,
//! `DistributionType::Normal` with an empty parameter map,
//! `normality_p_value: 0.5`, `autocorrelation: 0.05`/`0.1`,
//! `heteroscedasticity_p_value: 0.4`/`0.5`/`0.6`, `normality_p_value:
//! 0.6`/`0.7` and `outliers: Vec::new()`. The five differed from each other
//! only in which literals had been typed, so a model whose residuals were
//! wildly heteroscedastic and one whose residuals were textbook-clean produced
//! byte-identical diagnostics, and `ValidationResult` is what
//! `ModelValidationOrchestrator` reports.
//!
//! Every field produced here is computed from the predictions and observations
//! the strategy actually made. Where a diagnostic is undefined for the sample
//! it is reported as `NaN` rather than as a plausible number: no comparison
//! passes on `NaN`, so an unmeasurable residual cannot read as a well-behaved
//! one.
//!
//! [`ValidationStrategy`]: super::functions::ValidationStrategy

use std::collections::HashMap;

use crate::performance_optimizer::performance_modeling::types::{
    DistributionInfo, DistributionType, ResidualAnalysis, TestDataStatistics, ValidationDetails,
};
use crate::performance_optimizer::real_time_metrics::analytics::analyzers::series::{
    autocorrelation, chi_square_sf, ks_p_value, ks_statistic, mean, normal_cdf, pearson,
    sample_std_dev, sorted_finite,
};

/// Standardised-residual magnitude beyond which a residual is reported as an
/// outlier.
///
/// Three standard deviations is the conventional rule of thumb; it is a
/// reporting boundary, not a measurement, and the residual it is applied to is
/// measured.
const OUTLIER_Z_SCORE: f64 = 3.0;

/// Significance level at which the normal fit of the target is rejected.
const NORMALITY_ALPHA: f64 = 0.05;

/// Build the measured [`ValidationDetails`] for one validation run.
///
/// `predictions[i]` and `actuals[i]` must describe the same test observation;
/// the residual reported for it is `prediction - actual`, matching the sign
/// convention every strategy already used for `prediction_errors`.
pub fn measured_details(predictions: &[f64], actuals: &[f64]) -> ValidationDetails {
    let residuals: Vec<f64> = predictions.iter().zip(actuals.iter()).map(|(p, a)| p - a).collect();

    ValidationDetails {
        test_samples: residuals.len(),
        test_statistics: test_data_statistics(actuals),
        prediction_errors: residuals.iter().map(|residual| *residual as f32).collect(),
        residual_analysis: residual_analysis(predictions, &residuals),
    }
}

/// Descriptive statistics of the observed targets.
///
/// `feature_correlations` stays empty: the strategies hand this module the
/// target series alone, and an empty map is the honest report of "no feature
/// correlation was computed" rather than a set of invented coefficients.
pub fn test_data_statistics(actuals: &[f64]) -> TestDataStatistics {
    let target_mean = mean(actuals);
    // Unbiased (n-1) spread: the test set is a sample of the target, not the
    // population. Undefined below two observations.
    let target_std = sample_std_dev(actuals);

    let mut parameters = HashMap::new();
    if let (Some(mu), Some(sigma)) = (target_mean, target_std) {
        parameters.insert("mean".to_string(), mu as f32);
        parameters.insert("std_dev".to_string(), sigma as f32);
    }

    let normality_p_value = normality_p_value(actuals);
    // The only distribution this module fits is the normal one. Saying
    // "Normal" once its own goodness-of-fit test has rejected it would be a
    // claim the data contradicts, and saying it when the test could not run at
    // all would be a claim nothing supports.
    let distribution_type = match normality_p_value {
        p if p.is_nan() => DistributionType::Custom(
            "undetermined: too few finite observations to test a distribution".to_string(),
        ),
        p if (p as f64) < NORMALITY_ALPHA => DistributionType::Custom(format!(
            "not normal: Kolmogorov-Smirnov rejects the normal fit at {NORMALITY_ALPHA} (p = {p:.4})"
        )),
        _ => DistributionType::Normal,
    };

    TestDataStatistics {
        mean_target: target_mean.map(|m| m as f32).unwrap_or(f32::NAN),
        target_std: target_std.map(|s| s as f32).unwrap_or(f32::NAN),
        feature_correlations: HashMap::new(),
        distribution_info: DistributionInfo {
            distribution_type,
            parameters,
            normality_p_value,
        },
    }
}

/// Diagnostics of the residuals `fitted - observed`.
///
/// * `autocorrelation` is the lag-1 autocorrelation of the residual series: a
///   value far from zero says consecutive residuals carry the same sign, which
///   is what an unmodelled trend looks like.
/// * `heteroscedasticity_p_value` is the Breusch-Pagan LM test -- the squared
///   residuals regressed on the fitted values, `n * R^2` referred to a
///   chi-square with one degree of freedom. A small p-value says the residual
///   spread grows or shrinks with the prediction.
/// * `normality_p_value` is a one-sample Kolmogorov-Smirnov test of the
///   residuals against the normal distribution fitted to them. Because the mean
///   and standard deviation are estimated from the same sample the asymptotic
///   Kolmogorov p-value is conservative (a Lilliefors correction, which this
///   crate does not carry, would be needed for an exact level).
/// * `outliers` are the indices whose standardised residual exceeds
///   `OUTLIER_Z_SCORE`.
///
/// Each statistic is `NaN` (or, for `outliers`, empty) when the sample cannot
/// support it.
pub fn residual_analysis(fitted: &[f64], residuals: &[f64]) -> ResidualAnalysis {
    ResidualAnalysis {
        autocorrelation: autocorrelation(residuals, 1).map(|a| a as f32).unwrap_or(f32::NAN),
        heteroscedasticity_p_value: breusch_pagan_p_value(fitted, residuals)
            .map(|p| p as f32)
            .unwrap_or(f32::NAN),
        normality_p_value: normality_p_value(residuals),
        outliers: outlier_indices(residuals),
    }
}

/// Breusch-Pagan p-value for residual heteroscedasticity, or `None` when the
/// sample cannot support the test (fewer than three points, or no variation in
/// either the fitted values or the squared residuals).
fn breusch_pagan_p_value(fitted: &[f64], residuals: &[f64]) -> Option<f64> {
    if fitted.len() != residuals.len() {
        return None;
    }
    let squared: Vec<f64> = residuals.iter().map(|r| r * r).collect();
    let correlation = pearson(fitted, &squared)?;
    let lagrange_multiplier = residuals.len() as f64 * correlation * correlation;
    Some(chi_square_sf(lagrange_multiplier, 1.0).clamp(0.0, 1.0))
}

/// Kolmogorov-Smirnov p-value for the normal fit of `values`, or `NaN` when the
/// sample cannot support the test.
fn normality_p_value(values: &[f64]) -> f32 {
    let sorted = sorted_finite(values);
    if sorted.len() < 3 {
        return f32::NAN;
    }
    let (Some(mu), Some(sigma)) = (mean(&sorted), sample_std_dev(&sorted)) else {
        return f32::NAN;
    };
    if sigma <= 0.0 {
        // Every observation is the same value: there is no distribution to test.
        return f32::NAN;
    }
    let Some(statistic) = ks_statistic(&sorted, |x| normal_cdf((x - mu) / sigma)) else {
        return f32::NAN;
    };
    ks_p_value(statistic, sorted.len()) as f32
}

/// Indices whose standardised residual exceeds [`OUTLIER_Z_SCORE`].
///
/// Empty when the residuals have no measurable spread, which is an absence of
/// evidence rather than an absence of outliers.
fn outlier_indices(residuals: &[f64]) -> Vec<usize> {
    let (Some(mu), Some(sigma)) = (mean(residuals), sample_std_dev(residuals)) else {
        return Vec::new();
    };
    if sigma.is_nan() || sigma <= 0.0 {
        return Vec::new();
    }
    residuals
        .iter()
        .enumerate()
        .filter(|(_, residual)| ((*residual - mu) / sigma).abs() > OUTLIER_Z_SCORE)
        .map(|(index, _)| index)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: `target_std` was the literal `1.0` in all five strategies.
    #[test]
    fn target_spread_is_measured() {
        let tight = test_data_statistics(&[100.0, 100.5, 99.5, 100.2, 99.8]);
        let wide = test_data_statistics(&[10.0, 500.0, 90.0, 900.0, 3.0]);
        assert!(
            tight.target_std < wide.target_std,
            "spread must follow the data: {} vs {}",
            tight.target_std,
            wide.target_std
        );
        assert!(
            (tight.target_std - 1.0).abs() > 1e-6,
            "no longer the hardcoded 1.0: {}",
            tight.target_std
        );
        assert!(
            tight.mean_target > 99.0 && tight.mean_target < 101.0,
            "{}",
            tight.mean_target
        );
    }

    /// A single observation has no spread; that is reported as unmeasurable,
    /// not as a tight one.
    #[test]
    fn a_single_observation_has_no_measurable_spread() {
        let stats = test_data_statistics(&[42.0]);
        assert!(stats.target_std.is_nan(), "{}", stats.target_std);
        assert!(
            stats.distribution_info.normality_p_value.is_nan(),
            "{}",
            stats.distribution_info.normality_p_value
        );
        assert!(
            matches!(
                stats.distribution_info.distribution_type,
                DistributionType::Custom(_)
            ),
            "{:?}",
            stats.distribution_info.distribution_type
        );
    }

    /// Regression: `distribution_type` was `Normal` and `normality_p_value` was
    /// `0.5` whatever the observations looked like.
    #[test]
    fn a_rejected_normal_fit_is_not_reported_as_normal() {
        // A hard two-point mixture: far from any normal distribution.
        let bimodal: Vec<f64> = (0..40).map(|i| if i % 2 == 0 { 0.0 } else { 1000.0 }).collect();
        let stats = test_data_statistics(&bimodal);
        assert!(
            (stats.distribution_info.normality_p_value as f64) < NORMALITY_ALPHA,
            "a two-point mixture is not normal: {}",
            stats.distribution_info.normality_p_value
        );
        assert!(
            matches!(
                stats.distribution_info.distribution_type,
                DistributionType::Custom(_)
            ),
            "{:?}",
            stats.distribution_info.distribution_type
        );

        // A near-uniform spread over a symmetric range passes the test.
        let symmetric: Vec<f64> = (0..40).map(|i| i as f64 - 19.5).collect();
        let stats = test_data_statistics(&symmetric);
        assert!(
            (stats.distribution_info.normality_p_value - 0.5).abs() > 1e-6,
            "no longer the hardcoded 0.5: {}",
            stats.distribution_info.normality_p_value
        );
    }

    /// Regression: `heteroscedasticity_p_value` was one of `0.4`/`0.5`/`0.6`
    /// and could never distinguish a fan-shaped residual cloud from a flat one.
    #[test]
    fn heteroscedasticity_is_detected_from_the_residuals() {
        let fitted: Vec<f64> = (1..=60).map(|i| i as f64).collect();

        // Residual magnitude grows with the fitted value: heteroscedastic.
        let fanning: Vec<f64> = fitted
            .iter()
            .enumerate()
            .map(|(i, f)| if i % 2 == 0 { *f } else { -*f })
            .collect();
        let fanning_p =
            breusch_pagan_p_value(&fitted, &fanning).expect("60 points support the test");

        // Constant-magnitude residuals: homoscedastic.
        let flat: Vec<f64> = (0..60).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let flat_p = breusch_pagan_p_value(&fitted, &flat);

        assert!(
            fanning_p < 0.05,
            "a fanning residual cloud must be flagged: p = {fanning_p}"
        );
        // Perfectly constant squared residuals have no variance, so the test
        // is undefined rather than "passing"; either outcome must differ from
        // the fanning case.
        match flat_p {
            Some(p) => assert!(p > fanning_p, "flat = {p}, fanning = {fanning_p}"),
            None => {},
        }
    }

    /// Regression: `autocorrelation` was `0.05`/`0.1` and `outliers` was always
    /// empty.
    #[test]
    fn residual_structure_is_measured() {
        // Alternating residuals: strongly negatively autocorrelated at lag 1.
        let fitted: Vec<f64> = (1..=40).map(|i| i as f64).collect();
        let alternating: Vec<f64> = (0..40).map(|i| if i % 2 == 0 { 5.0 } else { -5.0 }).collect();
        let analysis = residual_analysis(&fitted, &alternating);
        assert!(
            analysis.autocorrelation < -0.5,
            "alternating residuals are negatively autocorrelated: {}",
            analysis.autocorrelation
        );
        assert!(
            (analysis.autocorrelation - 0.05).abs() > 1e-6
                && (analysis.autocorrelation - 0.1).abs() > 1e-6,
            "no longer one of the hardcoded pair: {}",
            analysis.autocorrelation
        );

        // One residual far outside the rest.
        let mut spiky = vec![0.1_f64; 40];
        spiky[7] = 50.0;
        let analysis = residual_analysis(&fitted, &spiky);
        assert_eq!(
            analysis.outliers,
            vec![7],
            "the one residual outside three standard deviations must be reported"
        );
    }

    /// `measured_details` keeps the sign convention the strategies published
    /// before and counts the residuals it was given.
    #[test]
    fn details_carry_the_measured_residuals() {
        let predictions = vec![10.0, 20.0, 30.0, 40.0];
        let actuals = vec![9.0, 22.0, 30.0, 37.0];
        let details = measured_details(&predictions, &actuals);
        assert_eq!(details.test_samples, 4);
        assert_eq!(details.prediction_errors, vec![1.0, -2.0, 0.0, 3.0]);
    }
}
