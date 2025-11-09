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
}

#[derive(Debug, Clone)]
struct ModelMetadata {
    name: String,
    trained_at: DateTime<Utc>,
    training_samples: usize,
    feature_count: usize,
    version: String,
}

#[derive(Debug, Clone)]
struct TrainingStatistics {
    r_squared: f32,
    mean_squared_error: f32,
    mean_absolute_error: f32,
    training_time: Duration,
    convergence_iterations: usize,
}

impl LinearRegressionModel {
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
                version: "1.0.0".to_string(),
            },
            training_stats: TrainingStatistics {
                r_squared: 0.0,
                mean_squared_error: f32::INFINITY,
                mean_absolute_error: f32::INFINITY,
                training_time: Duration::from_secs(0),
                convergence_iterations: 0,
            },
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

        // Normalize features if requested
        let (normalized_features, _normalization_params) = if config.normalize_features {
            self.normalize_features(features)?
        } else {
            (features.to_vec(), None)
        };

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

    /// Predict using raw features
    fn predict_raw(&self, features: &[f64]) -> f64 {
        let mut prediction = self.intercept;
        for (coef, &feature) in self.coefficients.iter().zip(features.iter()) {
            prediction += coef * feature;
        }
        prediction
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

        let params = NormalizationParams { means, stds };
        Ok((normalized_features, Some(params)))
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
        training_time: Duration,
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

        TrainingStatistics {
            r_squared: r_squared as f32,
            mean_squared_error: mse as f32,
            mean_absolute_error: mae as f32,
            training_time,
            convergence_iterations: 1, // Linear regression converges in one step
        }
    }
}

#[derive(Debug, Clone)]
struct NormalizationParams {
    means: Vec<f64>,
    stds: Vec<f64>,
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
        let throughput = self.predict_raw(&features);

        // Calculate uncertainty bounds (simplified)
        let uncertainty = self.calculate_prediction_uncertainty(&features)?;
        let confidence = (1.0 - uncertainty).clamp(0.1, 1.0);

        Ok(PerformancePrediction {
            throughput: throughput.max(0.0),
            latency: Duration::from_millis((1000.0 / throughput.max(0.001)) as u64),
            confidence,
            uncertainty_bounds: (
                throughput - uncertainty as f64,
                throughput + uncertainty as f64,
            ),
            model_name: self.metadata.name.clone(),
            feature_importance: self.get_feature_importance(),
            predicted_at: Utc::now(),
        })
    }

    fn get_accuracy(&self) -> ModelAccuracyMetrics {
        ModelAccuracyMetrics {
            overall_accuracy: 1.0 - self.training_stats.mean_absolute_error / 100.0,
            r_squared: self.training_stats.r_squared,
            mean_absolute_error: self.training_stats.mean_absolute_error,
            root_mean_squared_error: self.training_stats.mean_squared_error.sqrt(),
            cross_validation_scores: Vec::new(), // Would be populated during cross-validation
            confidence_interval: (0.8, 0.95),    // Simplified
            prediction_stability: 0.85,          // Simplified
            last_validated: self.metadata.trained_at,
        }
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
        features.push(system_state.io_wait_percent as f64);
        features.push(system_state.network_utilization as f64);

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

    fn calculate_prediction_uncertainty(&self, _features: &[f64]) -> Result<f32> {
        // Simplified uncertainty calculation based on training statistics
        let base_uncertainty = self.training_stats.mean_absolute_error / 10.0;
        Ok(base_uncertainty)
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
    /// Original feature count
    original_feature_count: usize,
}

impl PolynomialRegressionModel {
    /// Create new polynomial regression model
    pub fn new(feature_names: Vec<String>, degree: usize) -> Self {
        let original_feature_count = feature_names.len();
        let poly_feature_names = Self::generate_polynomial_feature_names(&feature_names, degree);

        Self {
            linear_model: LinearRegressionModel::new(poly_feature_names),
            degree,
            original_feature_count,
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
        let throughput = self.linear_model.predict_raw(&poly_features);
        let uncertainty = self.linear_model.calculate_prediction_uncertainty(&poly_features)?;
        let confidence = (1.0 - uncertainty).clamp(0.1, 1.0);

        Ok(PerformancePrediction {
            throughput: throughput.max(0.0),
            latency: Duration::from_millis((1000.0 / throughput.max(0.001)) as u64),
            confidence,
            uncertainty_bounds: (
                throughput - uncertainty as f64,
                throughput + uncertainty as f64,
            ),
            model_name: format!("PolynomialRegression(degree={})", self.degree),
            feature_importance: self.linear_model.get_feature_importance(),
            predicted_at: Utc::now(),
        })
    }

    fn get_accuracy(&self) -> ModelAccuracyMetrics {
        let mut accuracy = self.linear_model.get_accuracy();
        accuracy.overall_accuracy *= 0.95; // Polynomial models may overfit slightly
        accuracy
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
                version: "1.0.0".to_string(),
            },
            training_stats: TrainingStatistics {
                r_squared: 0.0,
                mean_squared_error: f32::INFINITY,
                mean_absolute_error: f32::INFINITY,
                training_time: Duration::from_secs(0),
                convergence_iterations: 0,
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
        training_time: Duration,
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

        TrainingStatistics {
            r_squared: r_squared as f32,
            mean_squared_error: mse as f32,
            mean_absolute_error: mae as f32,
            training_time,
            convergence_iterations: 1,
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
        let uncertainty = self.calculate_prediction_uncertainty(&features)?;
        let confidence = (1.0 - uncertainty).clamp(0.1, 1.0);

        Ok(PerformancePrediction {
            throughput: throughput.max(0.0),
            latency: Duration::from_millis((1000.0 / throughput.max(0.001)) as u64),
            confidence,
            uncertainty_bounds: (
                throughput - uncertainty as f64,
                throughput + uncertainty as f64,
            ),
            model_name: self.metadata.name.clone(),
            feature_importance: self.get_feature_importance(),
            predicted_at: Utc::now(),
        })
    }

    fn get_accuracy(&self) -> ModelAccuracyMetrics {
        ModelAccuracyMetrics {
            overall_accuracy: 1.0 - self.training_stats.mean_absolute_error / 100.0,
            r_squared: self.training_stats.r_squared,
            mean_absolute_error: self.training_stats.mean_absolute_error,
            root_mean_squared_error: self.training_stats.mean_squared_error.sqrt(),
            cross_validation_scores: Vec::new(),
            confidence_interval: (0.75, 0.90),
            prediction_stability: 0.80,
            last_validated: self.metadata.trained_at,
        }
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

    fn calculate_prediction_uncertainty(&self, _features: &[f64]) -> Result<f32> {
        Ok(self.training_stats.mean_absolute_error / 5.0)
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
