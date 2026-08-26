//! Forecasting over an observed metric series.
//!
//! Replaces the `create_analyzer_placeholder!`-generated `ForecastingEngine`,
//! which ignored its input and returned `mae: 0.1 / mape: 0.05 /
//! directional_accuracy: 0.9 / confidence: 0.85` with no models at all. Every
//! model below is fitted to the samples the caller supplied and scored on a
//! held-out tail of that same window.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};

use super::super::super::types::data_structures::TimestampedMetrics;
use super::super::types::ForecastingModel;
use super::super::types::{
    EnsembleForecast, ForecastAccuracyMetrics, ForecastPoint, ForecastingModelType,
    ForecastingResult, ModelPerformanceMetrics,
};
use super::series::{extract_series, mean, sample_std_dev};

/// The series this engine forecasts.
const FORECAST_SERIES: &str = "throughput";

/// Minimum samples before a holdout split leaves enough to fit on.
const MIN_SAMPLES: usize = 12;

/// Number of future points produced.
const HORIZON_STEPS: usize = 10;

/// Fits and scores short-horizon forecasting models.
#[derive(Clone, Debug)]
pub struct ForecastingEngine {
    shutdown: Arc<AtomicBool>,
}

impl ForecastingEngine {
    /// Create a new forecasting engine.
    pub async fn new() -> Result<Self> {
        Ok(Self {
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Fit every candidate model to `data` and score them on a holdout tail.
    pub async fn analyze(&self, data: &[TimestampedMetrics]) -> Result<ForecastingResult> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(anyhow!("Forecasting engine is shut down"));
        }
        if data.len() < MIN_SAMPLES {
            return Err(anyhow!(
                "Forecasting needs at least {} samples to hold out a validation tail, got {}",
                MIN_SAMPLES,
                data.len()
            ));
        }
        let values = extract_series(data)
            .into_iter()
            .find(|s| s.name == FORECAST_SERIES)
            .map(|s| s.values)
            .filter(|v| v.len() == data.len())
            .ok_or_else(|| {
                anyhow!(
                    "Metric series `{}` carried no finite samples",
                    FORECAST_SERIES
                )
            })?;

        let step = sampling_step(data);
        let last_timestamp = data.last().map(|s| s.timestamp).unwrap_or_else(Utc::now);
        let split = values.len() * 4 / 5;
        let (train, holdout) = values.split_at(split);
        if train.len() < 3 || holdout.is_empty() {
            return Err(anyhow!(
                "Forecasting split left {} training and {} holdout points",
                train.len(),
                holdout.len()
            ));
        }

        let mut models = Vec::new();
        for candidate in [
            Candidate::Linear,
            Candidate::MovingAverage,
            Candidate::ExponentialSmoothing,
        ] {
            if let Some(model) = candidate.fit(&values, train, holdout, step, last_timestamp) {
                models.push(model);
            }
        }
        if models.is_empty() {
            return Err(anyhow!(
                "No forecasting model could be fitted to `{}`",
                FORECAST_SERIES
            ));
        }

        let best = models
            .iter()
            .max_by(|a, b| {
                a.performance
                    .validation_accuracy
                    .partial_cmp(&b.performance.validation_accuracy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();
        let best_name = best.as_ref().map(|m| m.name.clone());
        let accuracy_metrics = best
            .as_ref()
            .and_then(|m| score_model(m, &values, train, holdout))
            .ok_or_else(|| anyhow!("Selected forecasting model produced no scoreable holdout"))?;
        let ensemble = build_ensemble(&models, &values, train, holdout);
        let series_spread = sample_std_dev(&values).unwrap_or(0.0);
        // Confidence is one minus the normalised holdout RMSE: a forecast whose
        // error matches the series' own spread carries no information (0.0).
        let confidence = if series_spread > 0.0 {
            (1.0 - accuracy_metrics.rmse / series_spread).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Ok(ForecastingResult {
            models,
            best_model: best_name,
            ensemble_forecast: ensemble,
            accuracy_metrics,
            confidence,
            horizon: step * HORIZON_STEPS as u32,
        })
    }

    /// Stop accepting analyses.
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// True once `shutdown` has been called.
    pub fn is_shut_down(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }
}

/// Median inter-sample gap of the window; one second when it cannot be measured.
fn sampling_step(data: &[TimestampedMetrics]) -> Duration {
    let mut gaps: Vec<i64> = Vec::new();
    for pair in data.windows(2) {
        let (Some(first), Some(second)) = (pair.first(), pair.get(1)) else {
            continue;
        };
        let delta = second.timestamp.signed_duration_since(first.timestamp).num_milliseconds();
        if delta > 0 {
            gaps.push(delta);
        }
    }
    if gaps.is_empty() {
        return Duration::from_secs(1);
    }
    gaps.sort_unstable();
    let median = gaps.get(gaps.len() / 2).copied().unwrap_or(1000);
    Duration::from_millis(median.max(1) as u64)
}

#[derive(Clone, Copy)]
enum Candidate {
    Linear,
    MovingAverage,
    ExponentialSmoothing,
}

impl Candidate {
    fn name(self) -> &'static str {
        match self {
            Candidate::Linear => "ordinary_least_squares",
            Candidate::MovingAverage => "moving_average",
            Candidate::ExponentialSmoothing => "simple_exponential_smoothing",
        }
    }

    fn model_type(self) -> ForecastingModelType {
        match self {
            Candidate::Linear => ForecastingModelType::LinearRegression,
            Candidate::MovingAverage => ForecastingModelType::MovingAverage,
            Candidate::ExponentialSmoothing => ForecastingModelType::ExponentialSmoothing,
        }
    }

    fn parameter_count(self) -> usize {
        match self {
            Candidate::Linear => 2,
            Candidate::MovingAverage => 1,
            Candidate::ExponentialSmoothing => 1,
        }
    }

    /// One-step-ahead predictions for `series[from..]` given the fitted rule.
    fn predict_series(self, series: &[f64], parameters: &HashMap<String, f64>) -> Vec<f64> {
        match self {
            Candidate::Linear => {
                let slope = parameters.get("slope").copied().unwrap_or(0.0);
                let intercept = parameters.get("intercept").copied().unwrap_or(0.0);
                (0..series.len()).map(|i| intercept + slope * i as f64).collect()
            },
            Candidate::MovingAverage => {
                let window = parameters.get("window").copied().unwrap_or(3.0).max(1.0) as usize;
                let mut out = Vec::with_capacity(series.len());
                for i in 0..series.len() {
                    let start = i.saturating_sub(window);
                    let slice = series.get(start..i).unwrap_or(&[]);
                    out.push(mean(slice).unwrap_or_else(|| series.first().copied().unwrap_or(0.0)));
                }
                out
            },
            Candidate::ExponentialSmoothing => {
                let alpha = parameters.get("alpha").copied().unwrap_or(0.3);
                let mut level = series.first().copied().unwrap_or(0.0);
                let mut out = Vec::with_capacity(series.len());
                for value in series {
                    out.push(level);
                    level = alpha * value + (1.0 - alpha) * level;
                }
                out
            },
        }
    }

    fn fit_parameters(self, train: &[f64]) -> Option<HashMap<String, f64>> {
        let mut parameters = HashMap::new();
        match self {
            Candidate::Linear => {
                let fit = super::series::linear_fit(train)?;
                parameters.insert("slope".to_string(), fit.slope);
                parameters.insert("intercept".to_string(), fit.intercept);
                parameters.insert("r_squared".to_string(), fit.r_squared);
            },
            Candidate::MovingAverage => {
                // Grid-search the window that minimises in-sample SSE.
                let mut best = (f64::INFINITY, 2usize);
                for window in 2..=(train.len() / 2).max(2).min(12) {
                    let mut params = HashMap::new();
                    params.insert("window".to_string(), window as f64);
                    let predictions = self.predict_series(train, &params);
                    let sse = sum_squared_error(train, &predictions);
                    if sse < best.0 {
                        best = (sse, window);
                    }
                }
                parameters.insert("window".to_string(), best.1 as f64);
            },
            Candidate::ExponentialSmoothing => {
                let mut best = (f64::INFINITY, 0.3f64);
                for step in 1..=19 {
                    let alpha = step as f64 / 20.0;
                    let mut params = HashMap::new();
                    params.insert("alpha".to_string(), alpha);
                    let predictions = self.predict_series(train, &params);
                    let sse = sum_squared_error(train, &predictions);
                    if sse < best.0 {
                        best = (sse, alpha);
                    }
                }
                parameters.insert("alpha".to_string(), best.1);
            },
        }
        Some(parameters)
    }

    /// Project `steps` points past the end of `series`.
    fn extrapolate(
        self,
        series: &[f64],
        parameters: &HashMap<String, f64>,
        steps: usize,
    ) -> Vec<f64> {
        match self {
            Candidate::Linear => {
                let slope = parameters.get("slope").copied().unwrap_or(0.0);
                let intercept = parameters.get("intercept").copied().unwrap_or(0.0);
                (0..steps).map(|k| intercept + slope * (series.len() + k) as f64).collect()
            },
            Candidate::MovingAverage => {
                let window = parameters.get("window").copied().unwrap_or(3.0).max(1.0) as usize;
                let start = series.len().saturating_sub(window);
                let level = mean(series.get(start..).unwrap_or(&[])).unwrap_or(0.0);
                vec![level; steps]
            },
            Candidate::ExponentialSmoothing => {
                let alpha = parameters.get("alpha").copied().unwrap_or(0.3);
                let mut level = series.first().copied().unwrap_or(0.0);
                for value in series {
                    level = alpha * value + (1.0 - alpha) * level;
                }
                vec![level; steps]
            },
        }
    }

    fn fit(
        self,
        full: &[f64],
        train: &[f64],
        holdout: &[f64],
        step: Duration,
        last_timestamp: DateTime<Utc>,
    ) -> Option<ForecastingModel> {
        let train_start = Instant::now();
        let parameters = self.fit_parameters(train)?;
        let training_time = train_start.elapsed();

        let in_sample = self.predict_series(train, &parameters);
        let training_accuracy = r_squared(train, &in_sample);
        let validation_accuracy = {
            let full_predictions = self.predict_series(full, &parameters);
            let tail = full_predictions.get(full.len() - holdout.len()..).unwrap_or(&[]);
            r_squared(holdout, tail)
        };
        let cv_score = rolling_origin_score(self, full, &parameters);

        let predict_start = Instant::now();
        let projected = self.extrapolate(full, &parameters, HORIZON_STEPS);
        let prediction_time = predict_start.elapsed();

        let residual_sd = residual_std_dev(train, &in_sample);
        let forecast_points = projected
            .iter()
            .enumerate()
            .map(|(k, value)| {
                // Prediction interval widens with the square root of the step
                // index, the random-walk error accumulation for a level model.
                let spread = 1.96 * residual_sd * ((k + 1) as f64).sqrt();
                ForecastPoint {
                    timestamp: last_timestamp
                        + chrono::Duration::from_std(step * (k as u32 + 1))
                            .unwrap_or_else(|_| chrono::Duration::zero()),
                    value: *value,
                    lower_bound: value - spread,
                    upper_bound: value + spread,
                    confidence: validation_accuracy.clamp(0.0, 1.0),
                }
            })
            .collect();

        Some(ForecastingModel {
            name: self.name().to_string(),
            model_type: self.model_type(),
            parameters,
            forecast_points,
            performance: ModelPerformanceMetrics {
                // Accuracy is reported as the coefficient of determination on
                // the fit window; this is a regression model, not a classifier.
                training_accuracy,
                validation_accuracy,
                cv_score,
                complexity: self.parameter_count() as f64,
                training_time,
                prediction_time,
            },
            confidence: validation_accuracy.clamp(0.0, 1.0),
        })
    }
}

fn sum_squared_error(actual: &[f64], predicted: &[f64]) -> f64 {
    actual.iter().zip(predicted.iter()).map(|(a, p)| (a - p).powi(2)).sum()
}

fn r_squared(actual: &[f64], predicted: &[f64]) -> f64 {
    let Some(m) = mean(actual) else {
        return 0.0;
    };
    let ss_tot: f64 = actual.iter().map(|a| (a - m).powi(2)).sum();
    if ss_tot <= 0.0 {
        return 0.0;
    }
    let ss_res = sum_squared_error(actual, predicted);
    (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
}

fn residual_std_dev(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() < 2 {
        return 0.0;
    }
    (sum_squared_error(actual, predicted) / (actual.len() as f64 - 1.0)).sqrt()
}

/// Mean holdout R-squared over three expanding-window folds.
fn rolling_origin_score(
    candidate: Candidate,
    series: &[f64],
    parameters: &HashMap<String, f64>,
) -> f64 {
    let folds = 3usize;
    let mut scores = Vec::new();
    for fold in 1..=folds {
        let split = series.len() * (folds + fold) / (2 * folds + 1);
        let (train, test) = series.split_at(split.min(series.len()));
        if train.len() < 3 || test.is_empty() {
            continue;
        }
        let predictions = candidate.predict_series(series, parameters);
        let tail = predictions.get(train.len()..).unwrap_or(&[]);
        scores.push(r_squared(test, tail));
    }
    mean(&scores).unwrap_or(0.0)
}

fn score_model(
    model: &ForecastingModel,
    full: &[f64],
    train: &[f64],
    holdout: &[f64],
) -> Option<ForecastAccuracyMetrics> {
    let candidate = match model.model_type {
        ForecastingModelType::LinearRegression => Candidate::Linear,
        ForecastingModelType::MovingAverage => Candidate::MovingAverage,
        ForecastingModelType::ExponentialSmoothing => Candidate::ExponentialSmoothing,
        _ => return None,
    };
    let predictions = candidate.predict_series(full, &model.parameters);
    let tail = predictions.get(full.len() - holdout.len()..)?;
    if tail.len() != holdout.len() || holdout.is_empty() {
        return None;
    }
    let n = holdout.len() as f64;
    let mae = holdout.iter().zip(tail).map(|(a, p)| (a - p).abs()).sum::<f64>() / n;
    let mse = sum_squared_error(holdout, tail) / n;
    let rmse = mse.sqrt();
    let mape = {
        let mut total = 0.0;
        let mut count = 0.0;
        for (a, p) in holdout.iter().zip(tail) {
            if a.abs() > f64::EPSILON {
                total += ((a - p) / a).abs();
                count += 1.0;
            }
        }
        if count > 0.0 {
            total / count
        } else {
            f64::NAN
        }
    };
    let smape = {
        let mut total = 0.0;
        let mut count = 0.0;
        for (a, p) in holdout.iter().zip(tail) {
            let denominator = a.abs() + p.abs();
            if denominator > f64::EPSILON {
                total += 2.0 * (a - p).abs() / denominator;
                count += 1.0;
            }
        }
        if count > 0.0 {
            total / count
        } else {
            f64::NAN
        }
    };
    // MASE scales the holdout MAE by the in-sample naive random-walk MAE.
    let naive_mae = {
        let diffs: Vec<f64> = train
            .windows(2)
            .filter_map(|w| {
                let (Some(a), Some(b)) = (w.first(), w.get(1)) else {
                    return None;
                };
                Some((b - a).abs())
            })
            .collect();
        mean(&diffs).unwrap_or(0.0)
    };
    let mase = if naive_mae > 0.0 { mae / naive_mae } else { f64::NAN };
    let directional_accuracy = {
        let mut hits = 0.0;
        let mut total = 0.0;
        let mut previous_actual = train.last().copied();
        for (a, p) in holdout.iter().zip(tail) {
            if let Some(previous) = previous_actual {
                let actual_direction = (a - previous).signum();
                let predicted_direction = (p - previous).signum();
                if actual_direction == predicted_direction {
                    hits += 1.0;
                }
                total += 1.0;
            }
            previous_actual = Some(*a);
        }
        if total > 0.0 {
            hits / total
        } else {
            f64::NAN
        }
    };
    let residual_sd = residual_std_dev(train, &candidate.predict_series(train, &model.parameters));
    let coverage_probability = if residual_sd > 0.0 {
        let inside = holdout
            .iter()
            .zip(tail)
            .filter(|(a, p)| (*a - *p).abs() <= 1.96 * residual_sd)
            .count() as f64;
        inside / n
    } else {
        f64::NAN
    };
    Some(ForecastAccuracyMetrics {
        mae,
        mse,
        rmse,
        mape,
        smape,
        mase,
        directional_accuracy,
        coverage_probability,
    })
}

/// Inverse-MSE weighted combination of the fitted models.
fn build_ensemble(
    models: &[ForecastingModel],
    full: &[f64],
    train: &[f64],
    holdout: &[f64],
) -> Option<EnsembleForecast> {
    if models.len() < 2 {
        return None;
    }
    let mut weights = HashMap::new();
    let mut total_weight = 0.0;
    for model in models {
        let Some(metrics) = score_model(model, full, train, holdout) else {
            continue;
        };
        if !metrics.mse.is_finite() {
            continue;
        }
        let weight = 1.0 / (metrics.mse + 1e-9);
        weights.insert(model.name.clone(), weight);
        total_weight += weight;
    }
    if total_weight <= 0.0 || weights.len() < 2 {
        return None;
    }
    for weight in weights.values_mut() {
        *weight /= total_weight;
    }
    let steps = models.iter().map(|m| m.forecast_points.len()).min().unwrap_or(0);
    if steps == 0 {
        return None;
    }
    let mut forecast_points = Vec::with_capacity(steps);
    let mut uncertainty = Vec::with_capacity(steps);
    for k in 0..steps {
        let mut value = 0.0;
        let mut lower = 0.0;
        let mut upper = 0.0;
        let mut timestamp = None;
        let mut spread_samples = Vec::new();
        for model in models {
            let Some(weight) = weights.get(&model.name) else {
                continue;
            };
            let Some(point) = model.forecast_points.get(k) else {
                continue;
            };
            value += weight * point.value;
            lower += weight * point.lower_bound;
            upper += weight * point.upper_bound;
            spread_samples.push(point.value);
            timestamp.get_or_insert(point.timestamp);
        }
        let Some(timestamp) = timestamp else {
            continue;
        };
        // Disagreement between the members is itself an uncertainty signal.
        let disagreement = sample_std_dev(&spread_samples).unwrap_or(0.0);
        uncertainty.push(disagreement);
        forecast_points.push(ForecastPoint {
            timestamp,
            value,
            lower_bound: lower,
            upper_bound: upper,
            confidence: models
                .iter()
                .map(|m| m.confidence)
                .fold(f64::INFINITY, f64::min)
                .clamp(0.0, 1.0),
        });
    }
    let confidence = models.iter().map(|m| m.confidence).fold(0.0f64, f64::max).clamp(0.0, 1.0);
    Some(EnsembleForecast {
        method: "inverse_mse_weighted".to_string(),
        model_weights: weights,
        forecast_points,
        confidence,
        uncertainty,
    })
}
