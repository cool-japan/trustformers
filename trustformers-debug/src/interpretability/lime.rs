//! LIME (Local Interpretable Model-agnostic Explanations) analysis
//!
//! This module implements LIME analysis for local model interpretability,
//! providing local explanations of model predictions through perturbation analysis.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LIME (Local Interpretable Model-agnostic Explanations) analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimeAnalysisResult {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// Local model coefficients
    pub local_coefficients: HashMap<String, f64>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Coefficient of determination of the published local surrogate against
    /// the perturbation predictions it is supposed to explain -- i.e. how
    /// faithful the explanation actually is. `None` when the perturbation
    /// predictions are all identical, leaving nothing to explain.
    ///
    /// This used to be the constant `0.75`, alongside a separate
    /// `local_fidelity: 0.85` that named the same idea with a second made-up
    /// number. Computed by `fit_local_surrogate`.
    pub local_r_squared: Option<f64>,
    /// Local model intercept
    pub intercept: f64,
    /// Feature importance scores
    pub feature_importance: Vec<FeatureImportance>,
    /// Perturbation analysis
    pub perturbation_analysis: PerturbationAnalysis,
    /// Local neighborhood statistics
    pub neighborhood_stats: NeighborhoodStats,
}

/// Feature importance from LIME
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// Feature name
    pub feature_name: String,
    /// Importance score
    pub importance_score: f64,
    /// Residual standard error of the feature's local slope estimate, or
    /// `None` when it is not estimable (see [`Self::confidence_interval`]).
    pub standard_error: Option<f64>,
    /// 95% confidence interval on the feature's local slope, from the
    /// residual standard error of its univariate fit. `None` when the
    /// perturbations left too little variation in this feature (or too few
    /// samples) to estimate one. Previously the literal `(coeff - 0.1,
    /// coeff + 0.1)`.
    pub confidence_interval: Option<(f64, f64)>,
    /// Two-sided p-value for `H0: slope = 0` on the same univariate fit,
    /// `None` under the same conditions. Previously the constant `0.05` for
    /// every feature of every instance.
    pub p_value: Option<f64>,
    /// Always `None`: measuring how stable a feature's attribution is across
    /// perturbation *rounds* needs repeated independent LIME runs, which this
    /// analyzer does not perform. Previously the constant `0.8`.
    pub stability: Option<f64>,
}

/// Perturbation analysis details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationAnalysis {
    /// Number of perturbations generated
    pub num_perturbations: usize,
    /// Perturbation strategy used
    pub strategy: String,
    /// Average prediction variance
    pub prediction_variance: f64,
    /// Fraction of the generated perturbations that actually changed at least
    /// one feature -- the real coverage of the sampling, not the old constant
    /// `0.8`.
    pub neighborhood_coverage: f64,
    /// Most influential perturbations
    pub influential_perturbations: Vec<PerturbationResult>,
}

/// Individual perturbation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationResult {
    /// Perturbation ID
    pub id: String,
    /// Features that were perturbed
    pub perturbed_features: Vec<String>,
    /// Original prediction
    pub original_prediction: f64,
    /// Perturbed prediction
    pub perturbed_prediction: f64,
    /// Prediction change
    pub prediction_change: f64,
    /// Distance from original instance
    pub distance: f64,
}

/// Local neighborhood statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborhoodStats {
    /// Mean prediction in neighborhood
    pub mean_prediction: f64,
    /// Population standard deviation of the perturbation predictions.
    /// Previously the constant `0.1`.
    pub std_prediction: f64,
    /// Always `None`: "neighborhood density" has no definition this analyzer
    /// can evaluate -- the perturbation sampler draws from an unnormalised
    /// noise distribution with no reference volume. Previously the constant
    /// `0.5`.
    pub density: Option<f64>,
    /// Feature correlation matrix in neighborhood
    pub correlation_matrix: HashMap<(String, String), f64>,
}

/// Goodness-of-fit statistics for the local surrogate a LIME run publishes.
///
/// Every field is measured from the perturbation sample the run actually
/// generated; the values these replace (`local_r_squared: 0.75`,
/// `local_fidelity: 0.85`, `p_value: 0.05`, `stability: 0.8`,
/// `confidence_interval: (coeff +/- 0.1)`, `std_prediction: 0.1`) were
/// constants that never varied with the model, the instance or the data.
#[derive(Debug, Clone)]
pub struct LocalSurrogateFit {
    /// See [`LimeAnalysisResult::local_r_squared`].
    pub r_squared: Option<f64>,
    /// Mean of the perturbation predictions.
    pub mean_prediction: f64,
    /// Population standard deviation of the perturbation predictions.
    pub std_prediction: f64,
    /// Per-feature inference statistics, keyed by feature name.
    pub coefficient_stats: HashMap<String, CoefficientStats>,
}

/// Inference statistics for one feature's local slope.
#[derive(Debug, Clone, Copy, Default)]
pub struct CoefficientStats {
    /// Residual standard error of the slope estimate.
    pub standard_error: Option<f64>,
    /// Two-sided p-value for `H0: slope = 0`.
    pub p_value: Option<f64>,
    /// 95% confidence interval on the slope.
    pub confidence_interval: Option<(f64, f64)>,
}

/// Fit statistics for the additive local surrogate
/// `y_hat(x) = mean(y) + sum_j coeff_j * (x_j - mean(x_j))`, which is exactly
/// the model a [`LimeAnalysisResult`]'s `local_coefficients` describe.
///
/// `local_data` and `predictions` must be the same length; a shorter-than-3
/// sample yields `None` everywhere rather than an invented number, because
/// the residual degrees of freedom (`n - 2`) would not be positive.
pub fn fit_local_surrogate(
    feature_names: &[String],
    local_data: &[HashMap<String, f64>],
    predictions: &[f64],
    coefficients: &HashMap<String, f64>,
) -> LocalSurrogateFit {
    let n = predictions.len().min(local_data.len());
    if n == 0 {
        return LocalSurrogateFit {
            r_squared: None,
            mean_prediction: 0.0,
            std_prediction: 0.0,
            coefficient_stats: HashMap::new(),
        };
    }
    let predictions = &predictions[..n];
    let local_data = &local_data[..n];

    let mean_prediction = predictions.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = predictions.iter().map(|y| (y - mean_prediction).powi(2)).sum();
    let std_prediction = (ss_tot / n as f64).sqrt();

    let feature_means: HashMap<&str, f64> = feature_names
        .iter()
        .map(|name| {
            let sum: f64 = local_data.iter().map(|row| row.get(name).copied().unwrap_or(0.0)).sum();
            (name.as_str(), sum / n as f64)
        })
        .collect();

    // R^2 of the additive surrogate against the real predictions.
    let ss_res: f64 = local_data
        .iter()
        .zip(predictions.iter())
        .map(|(row, y)| {
            let fitted = mean_prediction
                + feature_names
                    .iter()
                    .map(|name| {
                        let beta = coefficients.get(name).copied().unwrap_or(0.0);
                        let mean = feature_means.get(name.as_str()).copied().unwrap_or(0.0);
                        beta * (row.get(name).copied().unwrap_or(0.0) - mean)
                    })
                    .sum::<f64>();
            (y - fitted).powi(2)
        })
        .sum();
    let r_squared = if ss_tot > 0.0 { Some(1.0 - ss_res / ss_tot) } else { None };

    let degrees_of_freedom = n as f64 - 2.0;
    let t_critical = if degrees_of_freedom > 0.0 {
        statrs::distribution::StudentsT::new(0.0, 1.0, degrees_of_freedom)
            .ok()
            .map(|dist| {
                use statrs::distribution::ContinuousCDF;
                dist.inverse_cdf(0.975)
            })
    } else {
        None
    };

    let coefficient_stats = feature_names
        .iter()
        .map(|name| {
            let beta = coefficients.get(name).copied().unwrap_or(0.0);
            let mean = feature_means.get(name.as_str()).copied().unwrap_or(0.0);
            let sxx: f64 = local_data
                .iter()
                .map(|row| (row.get(name).copied().unwrap_or(0.0) - mean).powi(2))
                .sum();
            if degrees_of_freedom <= 0.0 || sxx <= 0.0 {
                return (name.clone(), CoefficientStats::default());
            }
            // Residuals of this feature's own univariate fit -- the model the
            // slope was estimated from.
            let sse: f64 = local_data
                .iter()
                .zip(predictions.iter())
                .map(|(row, y)| {
                    let fitted =
                        mean_prediction + beta * (row.get(name).copied().unwrap_or(0.0) - mean);
                    (y - fitted).powi(2)
                })
                .sum();
            let standard_error = (sse / degrees_of_freedom / sxx).sqrt();
            if !standard_error.is_finite() {
                return (name.clone(), CoefficientStats::default());
            }
            // A zero residual standard error is the degenerate exact-fit case:
            // the surrogate reproduces every prediction, so the slope is
            // pinned. With a non-zero slope the null is rejected at any alpha
            // (p = 0 exactly, not an underflow); with a zero slope there is
            // nothing to test at all.
            let (p_value, confidence_interval) = if standard_error == 0.0 {
                if beta == 0.0 {
                    (None, None)
                } else {
                    (Some(0.0), Some((beta, beta)))
                }
            } else {
                let t_statistic = beta / standard_error;
                (
                    trustformers_core::statistics::student_t_two_sided_p_value(
                        t_statistic,
                        degrees_of_freedom,
                    ),
                    t_critical.map(|t| (beta - t * standard_error, beta + t * standard_error)),
                )
            };
            (
                name.clone(),
                CoefficientStats {
                    standard_error: Some(standard_error),
                    p_value,
                    confidence_interval,
                },
            )
        })
        .collect();

    LocalSurrogateFit {
        r_squared,
        mean_prediction,
        std_prediction,
        coefficient_stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(xs: &[f64]) -> Vec<HashMap<String, f64>> {
        xs.iter()
            .map(|&x| {
                let mut row = HashMap::new();
                row.insert("x".to_string(), x);
                row
            })
            .collect()
    }

    /// An exactly linear neighbourhood must report a perfect fit and a
    /// vanishing p-value -- not the old constants 0.75 / 0.85 / 0.05.
    #[test]
    fn fit_local_surrogate_recovers_an_exact_linear_relationship() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let local_data = sample(&xs);
        let predictions: Vec<f64> = xs.iter().map(|x| 3.0 * x + 1.0).collect();
        let names = vec!["x".to_string()];
        let mut coefficients = HashMap::new();
        coefficients.insert("x".to_string(), 3.0);

        let fit = fit_local_surrogate(&names, &local_data, &predictions, &coefficients);

        let r2 = fit.r_squared.expect("the predictions vary, so R^2 is defined");
        assert!(
            (r2 - 1.0).abs() < 1e-12,
            "an exact fit has R^2 = 1, got {r2}"
        );
        assert!(
            (fit.mean_prediction - 8.5).abs() < 1e-12,
            "got {}",
            fit.mean_prediction
        );
        assert!(fit.std_prediction > 0.0);

        let stats = fit.coefficient_stats.get("x").expect("stats for the only feature");
        assert_eq!(
            stats.standard_error,
            Some(0.0),
            "an exact fit leaves no residual"
        );
        assert_eq!(stats.p_value, Some(0.0));
        assert_eq!(stats.confidence_interval, Some((3.0, 3.0)));
    }

    /// The ordinary (noisy) case: a real p-value strictly between 0 and 1, and
    /// a confidence interval of real width that brackets the slope.
    #[test]
    fn fit_local_surrogate_reports_real_inference_statistics_under_noise() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let noise = [0.30, -0.25, 0.10, 0.40, -0.35, 0.20, -0.15, 0.05];
        let local_data = sample(&xs);
        let predictions: Vec<f64> =
            xs.iter().zip(noise.iter()).map(|(x, e)| 3.0 * x + 1.0 + e).collect();
        let names = vec!["x".to_string()];
        let mut coefficients = HashMap::new();
        let beta = {
            let mean_x = xs.iter().sum::<f64>() / xs.len() as f64;
            let mean_y = predictions.iter().sum::<f64>() / predictions.len() as f64;
            let num: f64 = xs
                .iter()
                .zip(predictions.iter())
                .map(|(x, y)| (x - mean_x) * (y - mean_y))
                .sum();
            let den: f64 = xs.iter().map(|x| (x - mean_x).powi(2)).sum();
            num / den
        };
        coefficients.insert("x".to_string(), beta);

        let fit = fit_local_surrogate(&names, &local_data, &predictions, &coefficients);
        let r2 = fit.r_squared.expect("defined");
        assert!(
            r2 > 0.99 && r2 < 1.0,
            "a nearly-linear neighbourhood, got {r2}"
        );

        let stats = fit.coefficient_stats.get("x").expect("present");
        let se = stats.standard_error.expect("estimable");
        assert!(se > 0.0 && se.is_finite(), "got {se}");
        let p = stats.p_value.expect("estimable");
        assert!(p > 0.0 && p < 1e-6, "a strong but noisy slope, got {p}");
        let (lo, hi) = stats.confidence_interval.expect("estimable");
        assert!(hi - lo > 0.0);
        assert!(
            lo < beta && beta < hi,
            "the CI must bracket the estimate: ({lo}, {hi})"
        );
    }

    /// A neighbourhood the surrogate cannot explain must say so, rather than
    /// reporting the old flattering constants.
    #[test]
    fn fit_local_surrogate_reports_absence_when_nothing_is_estimable() {
        let names = vec!["x".to_string()];

        // Constant predictions: no variance to explain.
        let flat = fit_local_surrogate(
            &names,
            &sample(&[0.0, 1.0, 2.0, 3.0]),
            &[5.0, 5.0, 5.0, 5.0],
            &HashMap::from([("x".to_string(), 0.0)]),
        );
        assert_eq!(flat.r_squared, None);
        assert_eq!(flat.std_prediction, 0.0);

        // Two samples: no residual degrees of freedom for inference.
        let tiny = fit_local_surrogate(
            &names,
            &sample(&[0.0, 1.0]),
            &[1.0, 4.0],
            &HashMap::from([("x".to_string(), 3.0)]),
        );
        let stats = tiny.coefficient_stats.get("x").expect("present");
        assert_eq!(stats.p_value, None);
        assert_eq!(stats.confidence_interval, None);
        assert_eq!(stats.standard_error, None);

        // No samples at all.
        let empty = fit_local_surrogate(&names, &[], &[], &HashMap::new());
        assert_eq!(empty.r_squared, None);
        assert!(empty.coefficient_stats.is_empty());
    }

    /// A poor surrogate must report a poor (possibly negative) R^2 rather
    /// than the constant 0.75 the old code published for every run.
    #[test]
    fn fit_local_surrogate_reports_a_poor_fit_as_a_poor_fit() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        // Predictions unrelated to x, and a coefficient that claims otherwise.
        let predictions = vec![10.0, -4.0, 7.0, -9.0, 2.0, 12.0];
        let fit = fit_local_surrogate(
            &["x".to_string()],
            &sample(&xs),
            &predictions,
            &HashMap::from([("x".to_string(), 5.0)]),
        );
        let r2 = fit.r_squared.expect("the predictions vary");
        assert!(
            r2 < 0.5,
            "a surrogate this wrong must not report a good fit, got {r2}"
        );
    }
}
