//! Machine Learning Model Implementations for Performance Prediction
//!
//! This module provides comprehensive implementations of various machine learning
//! models for performance prediction, including linear regression, polynomial
//! regression, neural networks, ensemble methods, and custom model types.

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use std::{collections::HashMap, sync::Arc, time::Duration};

use super::types::*;
use crate::performance_optimizer::types::{SystemState, TestCharacteristics};

// Re-export traits needed by other modules
pub use super::types::ModelFactory;

// =============================================================================
// FIT-DERIVED REPORTING
// =============================================================================

/// Standard-normal multiplier for a two-sided 95% interval.
const NORMAL_95: f32 = 1.96;

/// One standard deviation of a fit's residuals: `sqrt(MSE)`.
fn residual_sigma(stats: &TrainingStatistics) -> f32 {
    stats.mean_squared_error.sqrt()
}

/// A 95% predictive interval around `point`, assuming residuals of scale
/// `sigma`.
///
/// A model that has never been trained carries an infinite residual scale, and
/// the interval it produces is correspondingly infinite: that is the honest
/// statement of "this model knows nothing yet", and it is what the caller sees
/// instead of the point estimate plus and minus a tenth of an infinite error.
fn prediction_interval(point: f64, sigma: f32) -> (f64, f64) {
    let half_width = (NORMAL_95 * sigma) as f64;
    (point - half_width, point + half_width)
}

/// Confidence attached to a prediction: the share of target variance the fit
/// explains, clamped to `[0, 1]`.
///
/// 0.2.1: this was `1 - mean_absolute_error / 10`, clamped to `[0.1, 1]`, which
/// mixed an error in throughput units into a `[0, 1]` confidence and floored an
/// untrained model at `0.1`. An untrained fit explains no variance and reports
/// `0.0` now.
fn fit_confidence(stats: &TrainingStatistics) -> f32 {
    if stats.sample_count == 0 || !stats.r_squared.is_finite() {
        return 0.0;
    }
    stats.r_squared.clamp(0.0, 1.0)
}

/// Accuracy relative to the target scale: `1 - MAE / mean|y|`, clamped to
/// `[0, 1]`.
///
/// `None` when the model has not been trained, or when the targets it was
/// trained on average to zero magnitude and there is no scale to be relative
/// to. 0.2.1: this was `1 - MAE / 100`, which assumed the targets lived on a
/// 0-100 scale and reported `-inf` for an untrained model.
fn relative_accuracy(stats: &TrainingStatistics) -> Option<f32> {
    if stats.sample_count == 0
        || stats.mean_absolute_target.is_nan()
        || stats.mean_absolute_target <= 0.0
    {
        return None;
    }
    if !stats.mean_absolute_error.is_finite() {
        return None;
    }
    Some((1.0 - stats.mean_absolute_error / stats.mean_absolute_target).clamp(0.0, 1.0))
}

/// Normal-approximation 95% interval for the mean absolute error measured on
/// the training sample.
///
/// The half-width is `1.96·s/sqrt(n)` for `s` the standard deviation of the
/// absolute residuals; the lower end is clamped at zero because an absolute
/// error cannot be negative. `None` for fewer than two points, where the
/// standard error is undefined. 0.2.1: the field carrying this was the constant
/// `(0.8, 0.95)` for linear and polynomial models and `(0.75, 0.90)` for the
/// exponential one, in neither case an interval around anything.
fn mean_absolute_error_interval(stats: &TrainingStatistics) -> Option<(f32, f32)> {
    if stats.sample_count < 2 || !stats.absolute_error_std.is_finite() {
        return None;
    }
    let standard_error = stats.absolute_error_std / (stats.sample_count as f32).sqrt();
    let half_width = NORMAL_95 * standard_error;
    Some((
        (stats.mean_absolute_error - half_width).max(0.0),
        stats.mean_absolute_error + half_width,
    ))
}

/// Build the published accuracy record from a measured fit.
fn accuracy_from_training(
    stats: &TrainingStatistics,
    trained_at: DateTime<Utc>,
) -> ModelAccuracyMetrics {
    ModelAccuracyMetrics {
        overall_accuracy: relative_accuracy(stats),
        r_squared: stats.r_squared,
        mean_absolute_error: stats.mean_absolute_error,
        root_mean_squared_error: residual_sigma(stats),
        // Nothing here cross-validates: the training pipeline's validators do,
        // and they build their own record. An empty list says so.
        cross_validation_scores: Vec::new(),
        mean_absolute_error_interval: mean_absolute_error_interval(stats),
        // No trainer in this module measures prediction stability; it would
        // take repeated fits on resampled data, and a trained model keeps its
        // coefficients but not its training set.
        prediction_stability: None,
        last_validated: trained_at,
    }
}

// =============================================================================
// LINEAR REGRESSION MODEL
// =============================================================================

/// Linear regression model for performance prediction
#[derive(Debug, Clone)]
pub struct LinearRegressionModel {
    /// Model coefficients
    coefficients: Vec<f64>,
    /// Intercept term
    intercept: f64,
    /// Feature names
    feature_names: Vec<String>,
    /// Model metadata
    metadata: ModelMetadata,
    /// Training statistics
    training_stats: TrainingStatistics,
    /// Feature normalisation captured during training, applied again at
    /// prediction time. `None` when the model was trained on raw features.
    normalization: Option<NormalizationParams>,
}

#[derive(Debug, Clone)]
struct ModelMetadata {
    name: String,
    trained_at: DateTime<Utc>,
    training_samples: usize,
    feature_count: usize,
}

/// Fit quality measured on the sample the model was trained on.
///
/// `sample_count`, `absolute_error_std` and `mean_absolute_target` were added
/// in 0.2.1 so that [`ModelAccuracyMetrics`] can report an interval around the
/// mean absolute error and an accuracy relative to the target scale, instead of
/// the constants `(0.8, 0.95)` / `(0.75, 0.90)` and the `1 - MAE/100` formula
/// that assumed targets lived on a 0-100 scale.
#[derive(Debug, Clone)]
struct TrainingStatistics {
    r_squared: f32,
    mean_squared_error: f32,
    mean_absolute_error: f32,
    /// Points the fit was measured on; zero before the model is trained.
    sample_count: usize,
    /// Standard deviation of the absolute residuals, the spread the mean
    /// absolute error's standard error is computed from.
    absolute_error_std: f32,
    /// Mean magnitude of the targets, the scale the relative accuracy divides
    /// by.
    mean_absolute_target: f32,
}

impl LinearRegressionModel {
    /// The fit quality measured when this model was last trained.
    fn training_statistics(&self) -> &TrainingStatistics {
        &self.training_stats
    }

    /// Create new linear regression model
    pub fn new(feature_names: Vec<String>) -> Self {
        Self {
            coefficients: vec![0.0; feature_names.len()],
            intercept: 0.0,
            feature_names,
            metadata: ModelMetadata {
                name: "LinearRegression".to_string(),
                trained_at: Utc::now(),
                training_samples: 0,
                feature_count: 0,
            },
            training_stats: TrainingStatistics {
                r_squared: 0.0,
                mean_squared_error: f32::INFINITY,
                mean_absolute_error: f32::INFINITY,
                sample_count: 0,
                absolute_error_std: f32::INFINITY,
                mean_absolute_target: 0.0,
            },
            normalization: None,
        }
    }

    /// Train the linear regression model
    pub fn train(
        &mut self,
        features: &[Vec<f64>],
        targets: &[f64],
        config: &ModelTrainingConfig,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        if features.len() != targets.len() {
            return Err(anyhow!("Feature matrix and target vector size mismatch"));
        }

        if features.is_empty() {
            return Err(anyhow!("No training data provided"));
        }

        // Normalize features if requested, keeping the parameters so the same
        // transform can be applied to every later prediction.
        let (normalized_features, normalization_params) = if config.normalize_features {
            self.normalize_features(features)?
        } else {
            (features.to_vec(), None)
        };
        self.normalization = normalization_params;

        // Perform ordinary least squares regression
        let (coefficients, intercept) =
            self.ordinary_least_squares(&normalized_features, targets)?;

        self.coefficients = coefficients;
        self.intercept = intercept;

        // Calculate training statistics
        let predictions: Vec<f64> = normalized_features
            .iter()
            .map(|feature_vec| self.predict_raw(feature_vec))
            .collect();

        self.training_stats =
            self.calculate_training_statistics(targets, &predictions, start_time.elapsed());

        self.metadata.trained_at = Utc::now();
        self.metadata.training_samples = features.len();
        self.metadata.feature_count = features[0].len();

        Ok(())
    }

    /// Ordinary least squares implementation
    fn ordinary_least_squares(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
    ) -> Result<(Vec<f64>, f64)> {
        let n_samples = features.len();
        let n_features = features[0].len();

        // Build design matrix X with intercept column
        let mut x_matrix = vec![vec![0.0; n_features + 1]; n_samples];
        for (i, feature_vec) in features.iter().enumerate() {
            x_matrix[i][0] = 1.0; // Intercept term
            for (j, &feature) in feature_vec.iter().enumerate() {
                x_matrix[i][j + 1] = feature;
            }
        }

        // Solve normal equations: (X^T * X)^-1 * X^T * y
        let xt_x = self.matrix_multiply_transpose(&x_matrix, &x_matrix)?;
        let xt_y = self.matrix_vector_multiply_transpose(&x_matrix, targets)?;
        let xt_x_inv = self.matrix_inverse(&xt_x)?;
        let coefficients_with_intercept = self.matrix_vector_multiply(&xt_x_inv, &xt_y)?;

        let intercept = coefficients_with_intercept[0];
        let coefficients = coefficients_with_intercept[1..].to_vec();

        Ok((coefficients, intercept))
    }

    /// Predict from features expressed in the model's own coordinate system.
    ///
    /// Callers inside `train` pass already-normalised rows; external callers go
    /// through [`LinearRegressionModel::predict_features`], which applies the
    /// training-time normalisation first.
    fn predict_raw(&self, features: &[f64]) -> f64 {
        let mut prediction = self.intercept;
        for (coef, &feature) in self.coefficients.iter().zip(features.iter()) {
            prediction += coef * feature;
        }
        prediction
    }

    /// Predict from features in their original units.
    ///
    /// Applies the normalisation captured during training, so a model trained
    /// with `normalize_features` predicts on the same scale it learned on.
    fn predict_features(&self, features: &[f64]) -> f64 {
        match &self.normalization {
            Some(params) => self.predict_raw(&params.apply(features)),
            None => self.predict_raw(features),
        }
    }

    /// Normalize features using z-score normalization
    fn normalize_features(
        &self,
        features: &[Vec<f64>],
    ) -> Result<(Vec<Vec<f64>>, Option<NormalizationParams>)> {
        if features.is_empty() {
            return Ok((Vec::new(), None));
        }

        let n_features = features[0].len();
        let mut means = vec![0.0; n_features];
        let mut stds = vec![0.0; n_features];

        // Calculate means
        for feature_vec in features {
            for (j, &value) in feature_vec.iter().enumerate() {
                means[j] += value;
            }
        }
        for mean in &mut means {
            *mean /= features.len() as f64;
        }

        // Calculate standard deviations
        for feature_vec in features {
            for (j, &value) in feature_vec.iter().enumerate() {
                stds[j] += (value - means[j]).powi(2);
            }
        }
        for std in &mut stds {
            *std = (*std / features.len() as f64).sqrt();
            if *std == 0.0 {
                *std = 1.0; // Prevent division by zero
            }
        }

        // Normalize features
        let normalized_features: Vec<Vec<f64>> = features
            .iter()
            .map(|feature_vec| {
                feature_vec
                    .iter()
                    .enumerate()
                    .map(|(j, &value)| (value - means[j]) / stds[j])
                    .collect()
            })
            .collect();

        Ok((
            normalized_features,
            Some(NormalizationParams { means, stds }),
        ))
    }

    /// Matrix operations
    fn matrix_multiply_transpose(&self, a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let rows_a = a.len();
        let cols_a = a[0].len();
        let rows_b = b.len();
        let cols_b = b[0].len();

        if rows_a != rows_b {
            return Err(anyhow!("Matrix dimension mismatch for A^T * B"));
        }

        let mut result = vec![vec![0.0; cols_b]; cols_a];
        for i in 0..cols_a {
            for j in 0..cols_b {
                for k in 0..rows_a {
                    result[i][j] += a[k][i] * b[k][j];
                }
            }
        }
        Ok(result)
    }

    fn matrix_vector_multiply_transpose(
        &self,
        matrix: &[Vec<f64>],
        vector: &[f64],
    ) -> Result<Vec<f64>> {
        let rows = matrix.len();
        let cols = matrix[0].len();

        if rows != vector.len() {
            return Err(anyhow!("Matrix-vector dimension mismatch"));
        }

        let mut result = vec![0.0; cols];
        for j in 0..cols {
            for i in 0..rows {
                result[j] += matrix[i][j] * vector[i];
            }
        }
        Ok(result)
    }

    fn matrix_vector_multiply(&self, matrix: &[Vec<f64>], vector: &[f64]) -> Result<Vec<f64>> {
        let rows = matrix.len();
        let cols = matrix[0].len();

        if cols != vector.len() {
            return Err(anyhow!("Matrix-vector dimension mismatch"));
        }

        let mut result = vec![0.0; rows];
        for i in 0..rows {
            for j in 0..cols {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        Ok(result)
    }

    fn matrix_inverse(&self, matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = matrix.len();
        if matrix.iter().any(|row| row.len() != n) {
            return Err(anyhow!("Matrix must be square for inversion"));
        }

        // Gauss-Jordan elimination with partial pivoting
        let mut aug_matrix = vec![vec![0.0; 2 * n]; n];

        // Initialize augmented matrix [A|I]
        for i in 0..n {
            for j in 0..n {
                aug_matrix[i][j] = matrix[i][j];
                aug_matrix[i][j + n] = if i == j { 1.0 } else { 0.0 };
            }
        }

        // Forward elimination
        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            for i in k + 1..n {
                if aug_matrix[i][k].abs() > aug_matrix[max_row][k].abs() {
                    max_row = i;
                }
            }

            // Swap rows if needed
            if max_row != k {
                aug_matrix.swap(k, max_row);
            }

            // Check for singularity
            if aug_matrix[k][k].abs() < 1e-12 {
                return Err(anyhow!("Matrix is singular and cannot be inverted"));
            }

            // Scale pivot row
            let pivot = aug_matrix[k][k];
            for j in 0..2 * n {
                aug_matrix[k][j] /= pivot;
            }

            // Eliminate column
            for i in 0..n {
                if i != k {
                    let factor = aug_matrix[i][k];
                    for j in 0..2 * n {
                        aug_matrix[i][j] -= factor * aug_matrix[k][j];
                    }
                }
            }
        }

        // Extract inverse matrix
        let mut inverse = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                inverse[i][j] = aug_matrix[i][j + n];
            }
        }

        Ok(inverse)
    }

    fn calculate_training_statistics(
        &self,
        targets: &[f64],
        predictions: &[f64],
        _training_time: Duration,
    ) -> TrainingStatistics {
        let n = targets.len() as f64;

        // Calculate MSE and MAE
        let mut mse = 0.0;
        let mut mae = 0.0;
        for (actual, predicted) in targets.iter().zip(predictions.iter()) {
            let error = actual - predicted;
            mse += error * error;
            mae += error.abs();
        }
        mse /= n;
        mae /= n;

        // Calculate R-squared
        let mean_target: f64 = targets.iter().sum::<f64>() / n;
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;
        for (actual, predicted) in targets.iter().zip(predictions.iter()) {
            ss_tot += (actual - mean_target).powi(2);
            ss_res += (actual - predicted).powi(2);
        }
        let r_squared = 1.0 - (ss_res / ss_tot.max(1e-12));

        // Spread of the absolute residuals, for the interval around `mae`, and
        // the mean target magnitude, the scale a relative accuracy needs.
        let absolute_error_variance = targets
            .iter()
            .zip(predictions.iter())
            .map(|(actual, predicted)| ((actual - predicted).abs() - mae).powi(2))
            .sum::<f64>()
            / n;
        let mean_absolute_target = targets.iter().map(|target| target.abs()).sum::<f64>() / n;

        TrainingStatistics {
            r_squared: r_squared as f32,
            mean_squared_error: mse as f32,
            mean_absolute_error: mae as f32,
            sample_count: targets.len(),
            absolute_error_std: absolute_error_variance.sqrt() as f32,
            mean_absolute_target: mean_absolute_target as f32,
        }
    }
}

/// Per-feature z-score parameters captured during training.
///
/// These are applied again at prediction time; before 0.2.1 they were computed,
/// returned, bound to `_normalization_params` and dropped, so a model trained
/// with `normalize_features` enabled was queried with raw, unnormalised
/// features and its predictions were systematically wrong.
#[derive(Debug, Clone)]
struct NormalizationParams {
    means: Vec<f64>,
    stds: Vec<f64>,
}

impl NormalizationParams {
    /// Apply the training-time z-score transform to one feature vector.
    ///
    /// Features beyond the ones seen during training are passed through
    /// unchanged rather than dropped, so a dimension mismatch cannot silently
    /// truncate the input.
    fn apply(&self, features: &[f64]) -> Vec<f64> {
        features
            .iter()
            .enumerate()
            .map(|(j, &value)| match (self.means.get(j), self.stds.get(j)) {
                (Some(mean), Some(std)) if *std != 0.0 => (value - mean) / std,
                _ => value,
            })
            .collect()
    }
}

impl PerformancePredictor for LinearRegressionModel {
    fn predict(&self, request: &PredictionRequest) -> Result<PerformancePrediction> {
        if request.parallelism_levels.is_empty() {
            return Err(anyhow!("No parallelism levels specified"));
        }

        // For simplicity, predict for the first parallelism level
        let parallelism = request.parallelism_levels[0];

        // Extract features from the request
        let features = self.extract_prediction_features(
            parallelism,
            &request.test_characteristics,
            &request.system_state,
        )?;

        // Make prediction
        let throughput = self.predict_features(&features);

        // A 95% predictive interval from the fit's own residual spread, and a
        // confidence that reports how much of the target variance the fit
        // explains. 0.2.1: the interval was the point estimate plus and minus
        // `mean_absolute_error / 10`, and the confidence was one minus that
        // same number.
        let sigma = self.residual_sigma();
        let confidence = fit_confidence(&self.training_stats);

        Ok(PerformancePrediction {
            throughput: throughput.max(0.0),
            latency: Duration::from_millis((1000.0 / throughput.max(0.001)) as u64),
            confidence,
            uncertainty_bounds: prediction_interval(throughput, sigma),
            model_name: self.metadata.name.clone(),
            feature_importance: self.get_feature_importance(),
            predicted_at: Utc::now(),
        })
    }

    fn get_accuracy(&self) -> ModelAccuracyMetrics {
        accuracy_from_training(&self.training_stats, self.metadata.trained_at)
    }

    fn name(&self) -> &str {
        &self.metadata.name
    }

    fn supports_online_learning(&self) -> bool {
        true // Linear regression supports incremental updates
    }
}

impl LinearRegressionModel {
    fn extract_prediction_features(
        &self,
        parallelism: usize,
        test_characteristics: &TestCharacteristics,
        system_state: &SystemState,
    ) -> Result<Vec<f64>> {
        let mut features = Vec::new();

        // Basic parallelism features
        features.push(parallelism as f64);
        features.push((parallelism as f64).sqrt());
        features.push((parallelism as f64).ln());

        // System state features
        features.push(system_state.available_cores as f64);
        features.push(system_state.available_memory_mb as f64);
        features.push(system_state.load_average as f64);
        features.push(system_state.active_processes as f64);

        // Test characteristics features
        features.push(test_characteristics.average_duration.as_secs_f64());
        features.push(test_characteristics.resource_intensity.cpu_intensity as f64);
        features.push(test_characteristics.resource_intensity.memory_intensity as f64);
        features.push(test_characteristics.resource_intensity.io_intensity as f64);
        features.push(test_characteristics.dependency_complexity as f64);

        // Concurrency features
        let max_concurrency =
            test_characteristics.concurrency_requirements.max_safe_concurrency as f64;
        features.push(max_concurrency);
        features.push(
            if test_characteristics.concurrency_requirements.parallel_capable {
                1.0
            } else {
                0.0
            },
        );

        Ok(features)
    }

    /// One standard deviation of this model's residuals on its training sample.
    ///
    /// 0.2.1: this was `mean_absolute_error / 10.0` (and `/ 5.0` in
    /// [`ExponentialModel`]) -- a divisor with no derivation, ignoring the
    /// features it was handed. The residual standard deviation is the scale a
    /// prediction interval is actually built from, and it is the square root of
    /// the mean squared error the fit already measured.
    ///
    /// The value does not vary per point: this model keeps its coefficients but
    /// not the design matrix, so a per-point predictive variance cannot be
    /// recovered from it.
    fn residual_sigma(&self) -> f32 {
        residual_sigma(&self.training_stats)
    }

    fn get_feature_importance(&self) -> HashMap<String, f32> {
        let mut importance = HashMap::new();

        // Calculate feature importance based on coefficient magnitudes
        let total_magnitude: f64 = self.coefficients.iter().map(|c| c.abs()).sum();

        for (i, &coef) in self.coefficients.iter().enumerate() {
            let normalized_importance =
                if total_magnitude > 0.0 { (coef.abs() / total_magnitude) as f32 } else { 0.0 };

            let feature_name =
                self.feature_names.get(i).unwrap_or(&format!("feature_{}", i)).clone();

            importance.insert(feature_name, normalized_importance);
        }

        importance
    }
}

// =============================================================================
// POLYNOMIAL REGRESSION MODEL
// =============================================================================

/// Polynomial regression model for performance prediction
#[derive(Debug, Clone)]
pub struct PolynomialRegressionModel {
    /// Underlying linear model with polynomial features
    linear_model: LinearRegressionModel,
    /// Polynomial degree
    degree: usize,
}

impl PolynomialRegressionModel {
    /// Create new polynomial regression model
    pub fn new(feature_names: Vec<String>, degree: usize) -> Self {
        let poly_feature_names = Self::generate_polynomial_feature_names(&feature_names, degree);

        Self {
            linear_model: LinearRegressionModel::new(poly_feature_names),
            degree,
        }
    }

    /// Train the polynomial regression model
    pub fn train(
        &mut self,
        features: &[Vec<f64>],
        targets: &[f64],
        config: &ModelTrainingConfig,
    ) -> Result<()> {
        // Transform features to polynomial space
        let poly_features = self.transform_to_polynomial_features(features)?;

        // Train the underlying linear model
        self.linear_model.train(&poly_features, targets, config)
    }

    /// Transform features to polynomial space
    fn transform_to_polynomial_features(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let mut poly_features = Vec::new();

        for feature_vec in features {
            let poly_vec = self.generate_polynomial_features(feature_vec)?;
            poly_features.push(poly_vec);
        }

        Ok(poly_features)
    }

    /// Generate polynomial features for a single feature vector
    fn generate_polynomial_features(&self, features: &[f64]) -> Result<Vec<f64>> {
        let mut poly_features = Vec::new();

        // Include original features (degree 1)
        poly_features.extend_from_slice(features);

        // Add polynomial terms for degrees 2 to self.degree
        for degree in 2..=self.degree {
            for indices in self.generate_polynomial_indices(features.len(), degree) {
                let mut term = 1.0;
                for &idx in &indices {
                    term *= features[idx];
                }
                poly_features.push(term);
            }
        }

        Ok(poly_features)
    }

    /// Generate polynomial feature names
    fn generate_polynomial_feature_names(original_names: &[String], degree: usize) -> Vec<String> {
        let mut names = Vec::new();

        // Original features (degree 1)
        names.extend(original_names.iter().cloned());

        // Polynomial terms
        for deg in 2..=degree {
            for indices in Self::generate_polynomial_indices_static(original_names.len(), deg) {
                let term_name = indices
                    .iter()
                    .map(|&idx| original_names[idx].clone())
                    .collect::<Vec<_>>()
                    .join("*");
                names.push(format!("{}^{}", term_name, deg));
            }
        }

        names
    }

    /// Generate polynomial term indices
    fn generate_polynomial_indices(&self, n_features: usize, degree: usize) -> Vec<Vec<usize>> {
        Self::generate_polynomial_indices_static(n_features, degree)
    }

    fn generate_polynomial_indices_static(n_features: usize, degree: usize) -> Vec<Vec<usize>> {
        let mut indices = Vec::new();

        // Generate all combinations with replacement
        fn generate_combinations(
            n: usize,
            k: usize,
            start: usize,
            current: &mut Vec<usize>,
            result: &mut Vec<Vec<usize>>,
        ) {
            if current.len() == k {
                result.push(current.clone());
                return;
            }

            for i in start..n {
                current.push(i);
                generate_combinations(n, k, i, current, result);
                current.pop();
            }
        }

        let mut current = Vec::new();
        generate_combinations(n_features, degree, 0, &mut current, &mut indices);

        indices
    }
}

impl PerformancePredictor for PolynomialRegressionModel {
    fn predict(&self, request: &PredictionRequest) -> Result<PerformancePrediction> {
        // Transform the request features to polynomial space
        let parallelism = request.parallelism_levels[0];
        let features = self.linear_model.extract_prediction_features(
            parallelism,
            &request.test_characteristics,
            &request.system_state,
        )?;

        let poly_features = self.generate_polynomial_features(&features)?;

        // Create a modified request with polynomial features
        let throughput = self.linear_model.predict_features(&poly_features);
        let sigma = self.linear_model.residual_sigma();
        let confidence = fit_confidence(self.linear_model.training_statistics());

        Ok(PerformancePrediction {
            throughput: throughput.max(0.0),
            latency: Duration::from_millis((1000.0 / throughput.max(0.001)) as u64),
            confidence,
            uncertainty_bounds: prediction_interval(throughput, sigma),
            model_name: format!("PolynomialRegression(degree={})", self.degree),
            feature_importance: self.linear_model.get_feature_importance(),
            predicted_at: Utc::now(),
        })
    }

    /// The polynomial model *is* its expanded-feature linear fit, so it reports
    /// that fit's measured accuracy.
    ///
    /// 0.2.1: the measured accuracy was multiplied by `0.95` here, because
    /// "Polynomial models may overfit slightly" -- a guess applied on top of a
    /// measurement, which made the published figure neither.
    fn get_accuracy(&self) -> ModelAccuracyMetrics {
        self.linear_model.get_accuracy()
    }

    fn name(&self) -> &str {
        "PolynomialRegression"
    }

    fn supports_online_learning(&self) -> bool {
        false // Polynomial regression typically requires batch retraining
    }
}

// =============================================================================
// EXPONENTIAL MODEL
// =============================================================================

/// Exponential model for performance prediction
#[derive(Debug, Clone)]
pub struct ExponentialModel {
    /// Model parameters
    parameters: ExponentialParameters,
    /// Feature names
    feature_names: Vec<String>,
    /// Model metadata
    metadata: ModelMetadata,
    /// Training statistics
    training_stats: TrainingStatistics,
}

#[derive(Debug, Clone)]
struct ExponentialParameters {
    /// Base coefficient
    base_coef: f64,
    /// Exponential coefficients
    exp_coefs: Vec<f64>,
    /// Scaling factor
    scale_factor: f64,
}

impl ExponentialModel {
    /// Create new exponential model
    pub fn new(feature_names: Vec<String>) -> Self {
        Self {
            parameters: ExponentialParameters {
                base_coef: 1.0,
                exp_coefs: vec![0.0; feature_names.len()],
                scale_factor: 1.0,
            },
            feature_names,
            metadata: ModelMetadata {
                name: "ExponentialModel".to_string(),
                trained_at: Utc::now(),
                training_samples: 0,
                feature_count: 0,
            },
            training_stats: TrainingStatistics {
                r_squared: 0.0,
                mean_squared_error: f32::INFINITY,
                mean_absolute_error: f32::INFINITY,
                sample_count: 0,
                absolute_error_std: f32::INFINITY,
                mean_absolute_target: 0.0,
            },
        }
    }

    /// Train the exponential model
    pub fn train(
        &mut self,
        features: &[Vec<f64>],
        targets: &[f64],
        config: &ModelTrainingConfig,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Transform to log space for linear fitting
        let log_targets: Result<Vec<f64>, _> = targets
            .iter()
            .map(|&t| {
                if t > 0.0 {
                    Ok(t.ln())
                } else {
                    Err(anyhow!("Exponential model requires positive targets"))
                }
            })
            .collect();
        let log_targets = log_targets?;

        // Fit linear model in log space
        let mut linear_model = LinearRegressionModel::new(self.feature_names.clone());
        linear_model.train(features, &log_targets, config)?;

        // Extract parameters
        self.parameters.exp_coefs = linear_model.coefficients.clone();
        self.parameters.base_coef = linear_model.intercept.exp();
        self.parameters.scale_factor = 1.0;

        // Calculate training statistics
        let predictions: Vec<f64> =
            features.iter().map(|feature_vec| self.predict_raw(feature_vec)).collect();

        self.training_stats =
            self.calculate_training_statistics(targets, &predictions, start_time.elapsed());

        self.metadata.trained_at = Utc::now();
        self.metadata.training_samples = features.len();
        self.metadata.feature_count = features[0].len();

        Ok(())
    }

    /// Predict using raw features
    fn predict_raw(&self, features: &[f64]) -> f64 {
        let mut exponent = 0.0;
        for (coef, &feature) in self.parameters.exp_coefs.iter().zip(features.iter()) {
            exponent += coef * feature;
        }
        self.parameters.base_coef * exponent.exp() * self.parameters.scale_factor
    }

    fn calculate_training_statistics(
        &self,
        targets: &[f64],
        predictions: &[f64],
        _training_time: Duration,
    ) -> TrainingStatistics {
        let n = targets.len() as f64;

        let mut mse = 0.0;
        let mut mae = 0.0;
        for (actual, predicted) in targets.iter().zip(predictions.iter()) {
            let error = actual - predicted;
            mse += error * error;
            mae += error.abs();
        }
        mse /= n;
        mae /= n;

        let mean_target: f64 = targets.iter().sum::<f64>() / n;
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;
        for (actual, predicted) in targets.iter().zip(predictions.iter()) {
            ss_tot += (actual - mean_target).powi(2);
            ss_res += (actual - predicted).powi(2);
        }
        let r_squared = 1.0 - (ss_res / ss_tot.max(1e-12));

        // Spread of the absolute residuals, for the interval around `mae`, and
        // the mean target magnitude, the scale a relative accuracy needs.
        let absolute_error_variance = targets
            .iter()
            .zip(predictions.iter())
            .map(|(actual, predicted)| ((actual - predicted).abs() - mae).powi(2))
            .sum::<f64>()
            / n;
        let mean_absolute_target = targets.iter().map(|target| target.abs()).sum::<f64>() / n;

        TrainingStatistics {
            r_squared: r_squared as f32,
            mean_squared_error: mse as f32,
            mean_absolute_error: mae as f32,
            sample_count: targets.len(),
            absolute_error_std: absolute_error_variance.sqrt() as f32,
            mean_absolute_target: mean_absolute_target as f32,
        }
    }
}

impl PerformancePredictor for ExponentialModel {
    fn predict(&self, request: &PredictionRequest) -> Result<PerformancePrediction> {
        let parallelism = request.parallelism_levels[0];
        let features = self.extract_prediction_features(
            parallelism,
            &request.test_characteristics,
            &request.system_state,
        )?;

        let throughput = self.predict_raw(&features);
        let sigma = self.residual_sigma();
        let confidence = fit_confidence(&self.training_stats);

        Ok(PerformancePrediction {
            throughput: throughput.max(0.0),
            latency: Duration::from_millis((1000.0 / throughput.max(0.001)) as u64),
            confidence,
            uncertainty_bounds: prediction_interval(throughput, sigma),
            model_name: self.metadata.name.clone(),
            feature_importance: self.get_feature_importance(),
            predicted_at: Utc::now(),
        })
    }

    fn get_accuracy(&self) -> ModelAccuracyMetrics {
        accuracy_from_training(&self.training_stats, self.metadata.trained_at)
    }

    fn name(&self) -> &str {
        &self.metadata.name
    }

    fn supports_online_learning(&self) -> bool {
        false
    }
}

impl ExponentialModel {
    fn extract_prediction_features(
        &self,
        parallelism: usize,
        test_characteristics: &TestCharacteristics,
        system_state: &SystemState,
    ) -> Result<Vec<f64>> {
        let mut features = Vec::new();

        features.push(parallelism as f64);
        features.push(system_state.available_cores as f64);
        features.push(system_state.available_memory_mb as f64);
        features.push(system_state.load_average as f64);
        features.push(test_characteristics.average_duration.as_secs_f64());
        features.push(test_characteristics.resource_intensity.cpu_intensity as f64);
        features.push(test_characteristics.dependency_complexity as f64);

        Ok(features)
    }

    /// One standard deviation of this model's residuals on its training sample;
    /// see [`LinearRegressionModel::residual_sigma`].
    fn residual_sigma(&self) -> f32 {
        residual_sigma(&self.training_stats)
    }

    fn get_feature_importance(&self) -> HashMap<String, f32> {
        let mut importance = HashMap::new();
        let total_magnitude: f64 = self.parameters.exp_coefs.iter().map(|c| c.abs()).sum();

        for (i, &coef) in self.parameters.exp_coefs.iter().enumerate() {
            let normalized_importance =
                if total_magnitude > 0.0 { (coef.abs() / total_magnitude) as f32 } else { 0.0 };

            let feature_name =
                self.feature_names.get(i).unwrap_or(&format!("feature_{}", i)).clone();

            importance.insert(feature_name, normalized_importance);
        }

        importance
    }
}

// =============================================================================
// MODEL REGISTRY
// =============================================================================

/// Registry for managing different model implementations
pub struct ModelRegistry {
    /// Registered model factories
    factories: Arc<RwLock<HashMap<String, Box<dyn ModelImplementationFactory>>>>,
}

/// Trait for model implementation factories
pub trait ModelImplementationFactory: std::fmt::Debug + Send + Sync {
    /// Create a new model instance
    fn create(&self, config: &ModelTypeConfig) -> Result<Box<dyn PerformancePredictor>>;

    /// Get model type name
    fn model_type(&self) -> &str;

    /// Get model capabilities
    fn capabilities(&self) -> ModelCapabilities;
}

/// Model capabilities
#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    /// Supports online learning
    pub online_learning: bool,
    /// Supports batch training
    pub batch_training: bool,
    /// Supports polynomial features
    pub polynomial_features: bool,
    /// Memory requirements
    pub memory_requirements: ResourceRequirements,
}

impl ModelRegistry {
    /// Create new model registry
    pub fn new() -> Self {
        Self {
            factories: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a model factory
    pub fn register_factory(&self, factory: Box<dyn ModelImplementationFactory>) {
        let mut factories = self.factories.write();
        factories.insert(factory.model_type().to_string(), factory);
    }

    /// Create model by type
    pub fn create_model(
        &self,
        model_type: &str,
        config: &ModelTypeConfig,
    ) -> Result<Box<dyn PerformancePredictor>> {
        let factories = self.factories.read();
        let factory = factories
            .get(model_type)
            .ok_or_else(|| anyhow!("Model type '{}' not registered", model_type))?;
        factory.create(config)
    }

    /// Get available model types
    pub fn available_types(&self) -> Vec<String> {
        let factories = self.factories.read();
        factories.keys().cloned().collect()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        let registry = Self::new();

        // Register default model implementations
        registry.register_factory(Box::new(LinearRegressionFactory));
        registry.register_factory(Box::new(PolynomialRegressionFactory));
        registry.register_factory(Box::new(ExponentialModelFactory));

        registry
    }
}

// =============================================================================
// FACTORY IMPLEMENTATIONS
// =============================================================================

/// Linear regression model factory
#[derive(Debug, Clone, Copy)]
struct LinearRegressionFactory;

impl ModelImplementationFactory for LinearRegressionFactory {
    fn create(&self, config: &ModelTypeConfig) -> Result<Box<dyn PerformancePredictor>> {
        let feature_names = config
            .parameters
            .get("feature_names")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_else(|| vec!["parallelism".to_string(), "system_load".to_string()]);

        Ok(Box::new(LinearRegressionModel::new(feature_names)))
    }

    fn model_type(&self) -> &str {
        "LinearRegression"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            online_learning: true,
            batch_training: true,
            polynomial_features: false,
            memory_requirements: ResourceRequirements {
                min_memory_mb: 10,
                cpu_utilization: 0.1,
                gpu_requirement: GpuRequirement::None,
                disk_space_mb: 1,
            },
        }
    }
}

/// Polynomial regression model factory
#[derive(Debug, Clone, Copy)]
struct PolynomialRegressionFactory;

impl ModelImplementationFactory for PolynomialRegressionFactory {
    fn create(&self, config: &ModelTypeConfig) -> Result<Box<dyn PerformancePredictor>> {
        let feature_names = config
            .parameters
            .get("feature_names")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_else(|| vec!["parallelism".to_string(), "system_load".to_string()]);

        let degree = config.parameters.get("degree").and_then(|v| v.as_u64()).unwrap_or(2) as usize;

        Ok(Box::new(PolynomialRegressionModel::new(
            feature_names,
            degree,
        )))
    }

    fn model_type(&self) -> &str {
        "PolynomialRegression"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            online_learning: false,
            batch_training: true,
            polynomial_features: true,
            memory_requirements: ResourceRequirements {
                min_memory_mb: 50,
                cpu_utilization: 0.3,
                gpu_requirement: GpuRequirement::None,
                disk_space_mb: 5,
            },
        }
    }
}

/// Exponential model factory
#[derive(Debug, Clone, Copy)]
struct ExponentialModelFactory;

impl ModelImplementationFactory for ExponentialModelFactory {
    fn create(&self, config: &ModelTypeConfig) -> Result<Box<dyn PerformancePredictor>> {
        let feature_names = config
            .parameters
            .get("feature_names")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_else(|| vec!["parallelism".to_string(), "system_load".to_string()]);

        Ok(Box::new(ExponentialModel::new(feature_names)))
    }

    fn model_type(&self) -> &str {
        "ExponentialModel"
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            online_learning: false,
            batch_training: true,
            polynomial_features: false,
            memory_requirements: ResourceRequirements {
                min_memory_mb: 15,
                cpu_utilization: 0.2,
                gpu_requirement: GpuRequirement::None,
                disk_space_mb: 2,
            },
        }
    }
}

#[cfg(test)]
mod honesty_tests {
    use super::*;
    use crate::performance_optimizer::types::{SystemState, TestCharacteristics};

    fn training_config() -> ModelTrainingConfig {
        ModelTrainingConfig {
            normalize_features: false,
            ..ModelTrainingConfig::default()
        }
    }

    /// `y = 3x + 1` sampled without noise, plus an optional per-point residual.
    fn linear_sample(count: usize, noise: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut features = Vec::with_capacity(count);
        let mut targets = Vec::with_capacity(count);
        for index in 0..count {
            let x = index as f64;
            features.push(vec![x]);
            // Alternating residual: mean absolute error is exactly `noise`.
            let residual = if index % 2 == 0 { noise } else { -noise };
            targets.push(3.0 * x + 1.0 + residual);
        }
        (features, targets)
    }

    fn prediction_request(parallelism: usize) -> PredictionRequest {
        PredictionRequest {
            parallelism_levels: vec![parallelism],
            test_characteristics: TestCharacteristics::default(),
            system_state: SystemState::default(),
            prediction_horizon: None,
            confidence_level: 0.95,
            include_uncertainty: false,
        }
    }

    /// Regression: an untrained model reported `overall_accuracy` as
    /// `1 - INFINITY/100` and a `confidence_interval` of `(0.8, 0.95)`.
    #[test]
    fn an_untrained_model_reports_no_accuracy() {
        let model = LinearRegressionModel::new(vec!["x".to_string()]);
        let accuracy = model.get_accuracy();
        assert_eq!(
            accuracy.overall_accuracy, None,
            "a model trained on nothing has no measured accuracy"
        );
        assert_eq!(
            accuracy.mean_absolute_error_interval, None,
            "an interval needs at least two training points"
        );
        assert_eq!(
            accuracy.prediction_stability, None,
            "no trainer in this module measures stability"
        );
    }

    /// Regression: `confidence_interval` was the constant `(0.8, 0.95)` for
    /// every linear model. The reported interval now brackets the mean absolute
    /// error this particular fit measured.
    #[test]
    fn the_reported_interval_brackets_the_measured_error() {
        let mut model = LinearRegressionModel::new(vec!["x".to_string()]);
        let (features, targets) = linear_sample(40, 2.0);
        model.train(&features, &targets, &training_config()).expect("training succeeds");

        let accuracy = model.get_accuracy();
        let (low, high) =
            accuracy.mean_absolute_error_interval.expect("forty points give an interval");
        assert!(
            low <= accuracy.mean_absolute_error && accuracy.mean_absolute_error <= high,
            "the interval ({low}, {high}) must bracket the error {}",
            accuracy.mean_absolute_error
        );
        assert!(
            low < high,
            "an interval over a spread sample must have width"
        );

        // A cleaner fit reports a smaller error and a tighter interval.
        let mut clean = LinearRegressionModel::new(vec!["x".to_string()]);
        let (clean_features, clean_targets) = linear_sample(40, 0.2);
        clean
            .train(&clean_features, &clean_targets, &training_config())
            .expect("training succeeds");
        let clean_accuracy = clean.get_accuracy();
        assert!(
            clean_accuracy.mean_absolute_error < accuracy.mean_absolute_error,
            "the cleaner sample must fit better: {} vs {}",
            clean_accuracy.mean_absolute_error,
            accuracy.mean_absolute_error
        );
        let (clean_low, clean_high) = clean_accuracy
            .mean_absolute_error_interval
            .expect("forty points give an interval");
        assert!(
            clean_high - clean_low < high - low,
            "a tighter fit must give a tighter interval: {} vs {}",
            clean_high - clean_low,
            high - low
        );
    }

    /// Regression: `overall_accuracy` was `1 - MAE/100`, which assumed the
    /// targets lived on a 0-100 scale. It is relative to the measured target
    /// scale now, so the same relative error reports the same accuracy whatever
    /// the units are.
    #[test]
    fn accuracy_is_relative_to_the_target_scale() {
        let mut small = LinearRegressionModel::new(vec!["x".to_string()]);
        let (features, targets) = linear_sample(40, 1.0);
        small.train(&features, &targets, &training_config()).expect("training succeeds");

        // The same series scaled by 1000, with the residuals scaled too.
        let mut large = LinearRegressionModel::new(vec!["x".to_string()]);
        let scaled_targets: Vec<f64> = targets.iter().map(|t| t * 1000.0).collect();
        large
            .train(&features, &scaled_targets, &training_config())
            .expect("training succeeds");

        let small_accuracy =
            small.get_accuracy().overall_accuracy.expect("trained model reports accuracy");
        let large_accuracy =
            large.get_accuracy().overall_accuracy.expect("trained model reports accuracy");
        assert!(
            (small_accuracy - large_accuracy).abs() < 0.01,
            "a thousand-fold change of units must not change the accuracy: \
             {small_accuracy} vs {large_accuracy}"
        );
    }

    /// Regression: the prediction interval was the point estimate plus and
    /// minus `mean_absolute_error / 10`, and the confidence was one minus that
    /// same quantity clamped to `[0.1, 1]`. Both come from the fit now.
    #[test]
    fn prediction_bounds_widen_with_the_residuals() {
        let mut clean = LinearRegressionModel::new(vec!["x".to_string()]);
        let (features, targets) = linear_sample(40, 0.1);
        clean.train(&features, &targets, &training_config()).expect("training succeeds");

        let mut noisy = LinearRegressionModel::new(vec!["x".to_string()]);
        let (noisy_features, noisy_targets) = linear_sample(40, 20.0);
        noisy
            .train(&noisy_features, &noisy_targets, &training_config())
            .expect("training succeeds");

        let clean_prediction = clean.predict(&prediction_request(4)).expect("prediction succeeds");
        let noisy_prediction = noisy.predict(&prediction_request(4)).expect("prediction succeeds");

        let clean_width =
            clean_prediction.uncertainty_bounds.1 - clean_prediction.uncertainty_bounds.0;
        let noisy_width =
            noisy_prediction.uncertainty_bounds.1 - noisy_prediction.uncertainty_bounds.0;
        assert!(
            noisy_width > clean_width,
            "noisier residuals must give a wider interval: {noisy_width} vs {clean_width}"
        );

        // The width is the 95% normal interval of the fit's residual scale.
        let expected = 2.0 * 1.96 * (noisy.get_accuracy().root_mean_squared_error as f64);
        assert!(
            (noisy_width - expected).abs() < expected * 1e-3,
            "the interval must be ±1.96σ: {noisy_width} vs {expected}"
        );

        assert!(
            clean_prediction.confidence > noisy_prediction.confidence,
            "the better fit must report the higher confidence: {} vs {}",
            clean_prediction.confidence,
            noisy_prediction.confidence
        );
    }

    /// Regression: the polynomial model multiplied its measured accuracy by
    /// `0.95` because polynomials "may overfit slightly".
    #[test]
    fn the_polynomial_model_reports_its_own_fit_unaltered() {
        let mut model = PolynomialRegressionModel::new(vec!["x".to_string()], 2);
        let (features, targets) = linear_sample(40, 1.0);
        model.train(&features, &targets, &training_config()).expect("training succeeds");

        let accuracy = model.get_accuracy();
        let inner = model.linear_model.get_accuracy();
        assert_eq!(
            accuracy.overall_accuracy, inner.overall_accuracy,
            "the polynomial model is its expanded-feature linear fit"
        );
    }
}
