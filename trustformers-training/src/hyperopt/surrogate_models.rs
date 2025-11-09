//! Concrete implementations of surrogate models and acquisition functions
//!
//! This module provides practical implementations of surrogate models for Bayesian optimization
//! and acquisition functions for guiding the search process.

use super::{
    AcquisitionFunction, AcquisitionFunctionType, ParameterValue, SearchSpace, SurrogateModel,
};
use anyhow::Result;
use std::collections::HashMap;

/// Simple Gaussian Process surrogate model
#[derive(Debug)]
pub struct SimpleGaussianProcess {
    /// Observed input-output pairs
    observations: Vec<(Vec<f64>, f64)>,
    /// Noise level
    noise_level: f64,
    /// Length scale
    length_scale: f64,
    /// Signal variance
    signal_variance: f64,
}

impl SimpleGaussianProcess {
    pub fn new(noise_level: f64, length_scale: f64, signal_variance: f64) -> Self {
        Self {
            observations: Vec::new(),
            noise_level,
            length_scale,
            signal_variance,
        }
    }

    /// Convert parameter map to feature vector
    fn params_to_features(&self, params: &HashMap<String, ParameterValue>) -> Vec<f64> {
        let mut features = Vec::new();

        // Sort parameter names for consistent ordering
        let mut param_names: Vec<String> = params.keys().cloned().collect();
        param_names.sort();

        for name in param_names {
            if let Some(value) = params.get(&name) {
                let feature = match value {
                    ParameterValue::Float(f) => *f,
                    ParameterValue::Int(i) => *i as f64,
                    ParameterValue::Bool(b) => {
                        if *b {
                            1.0
                        } else {
                            0.0
                        }
                    },
                    ParameterValue::String(s) => {
                        // Simple hash-based encoding for categorical variables
                        let hash = s.chars().map(|c| c as u32).sum::<u32>() as f64;
                        hash / 1000.0 // Normalize
                    },
                };
                features.push(feature);
            }
        }

        features
    }

    /// RBF kernel function
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let squared_distance: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();

        self.signal_variance * (-squared_distance / (2.0 * self.length_scale.powi(2))).exp()
    }

    /// Compute kernel matrix
    fn kernel_matrix(&self, x1_list: &[Vec<f64>], x2_list: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; x2_list.len()]; x1_list.len()];

        for (i, x1) in x1_list.iter().enumerate() {
            for (j, x2) in x2_list.iter().enumerate() {
                matrix[i][j] = self.kernel(x1, x2);
            }
        }

        matrix
    }
}

impl SurrogateModel for SimpleGaussianProcess {
    fn fit(&mut self, observations: &[(HashMap<String, ParameterValue>, f64)]) -> Result<()> {
        self.observations.clear();

        for (params, value) in observations {
            let features = self.params_to_features(params);
            self.observations.push((features, *value));
        }

        Ok(())
    }

    fn predict(&self, parameters: &HashMap<String, ParameterValue>) -> Result<(f64, f64)> {
        if self.observations.is_empty() {
            return Ok((0.0, self.signal_variance));
        }

        let test_features = self.params_to_features(parameters);

        // Extract training data
        let train_features: Vec<Vec<f64>> =
            self.observations.iter().map(|(features, _)| features.clone()).collect();
        let train_targets: Vec<f64> = self.observations.iter().map(|(_, target)| *target).collect();

        // Compute kernel matrices
        let k_train_train = self.kernel_matrix(&train_features, &train_features);
        let k_test_train: Vec<f64> = train_features
            .iter()
            .map(|train_feat| self.kernel(&test_features, train_feat))
            .collect();
        let k_test_test = self.kernel(&test_features, &test_features);

        // Add noise to diagonal of training kernel matrix
        let mut k_train_train_noisy = k_train_train;
        for i in 0..k_train_train_noisy.len() {
            k_train_train_noisy[i][i] += self.noise_level;
        }

        // Solve for alpha = K^{-1} y (simplified using pseudo-inverse)
        let alpha = self.solve_linear_system(&k_train_train_noisy, &train_targets)?;

        // Compute posterior mean
        let mean: f64 = k_test_train.iter().zip(alpha.iter()).map(|(k, a)| k * a).sum();

        // Compute posterior variance (simplified)
        let variance =
            k_test_test - k_test_train.iter().zip(alpha.iter()).map(|(k, a)| k * a).sum::<f64>();

        Ok((mean, variance.max(1e-6))) // Ensure positive variance
    }

    fn update(&mut self, parameters: HashMap<String, ParameterValue>, value: f64) -> Result<()> {
        let features = self.params_to_features(&parameters);
        self.observations.push((features, value));
        Ok(())
    }
}

impl SimpleGaussianProcess {
    /// Simplified linear system solver (Gauss-Seidel method)
    fn solve_linear_system(&self, matrix: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>> {
        let n = matrix.len();
        if n == 0 || matrix[0].len() != n || rhs.len() != n {
            return Err(anyhow::anyhow!("Invalid matrix dimensions"));
        }

        let mut solution = vec![0.0; n];
        let max_iterations = 100;
        let tolerance = 1e-6;

        for _iteration in 0..max_iterations {
            let mut max_change: f64 = 0.0;

            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    if i != j {
                        sum += matrix[i][j] * solution[j];
                    }
                }

                let new_value = (rhs[i] - sum) / matrix[i][i].max(1e-10);
                max_change = max_change.max((new_value - solution[i]).abs());
                solution[i] = new_value;
            }

            if max_change < tolerance {
                break;
            }
        }

        Ok(solution)
    }
}

/// Expected Improvement acquisition function
#[derive(Debug)]
pub struct ExpectedImprovement {
    xi: f64,
}

impl ExpectedImprovement {
    pub fn new(xi: f64) -> Self {
        Self { xi }
    }

    /// Standard normal probability density function
    fn normal_pdf(x: f64) -> f64 {
        (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * x * x).exp()
    }

    /// Standard normal cumulative distribution function (approximation)
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / 2.0_f64.sqrt()))
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

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}

impl AcquisitionFunction for ExpectedImprovement {
    fn compute(
        &self,
        parameters: &HashMap<String, ParameterValue>,
        model: &dyn SurrogateModel,
        best_value: f64,
    ) -> Result<f64> {
        let (mean, variance) = model.predict(parameters)?;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return Ok(0.0);
        }

        let improvement = mean - best_value - self.xi;
        let z = improvement / std_dev;

        let ei = improvement * Self::normal_cdf(z) + std_dev * Self::normal_pdf(z);
        Ok(ei.max(0.0))
    }

    fn optimize(
        &self,
        model: &dyn SurrogateModel,
        search_space: &SearchSpace,
        best_value: f64,
    ) -> Result<HashMap<String, ParameterValue>> {
        // Simple grid search optimization
        let num_candidates = 1000;
        let mut best_acquisition = f64::NEG_INFINITY;
        let mut best_params = search_space.sample_random()?;

        for _ in 0..num_candidates {
            let candidate = search_space.sample_random()?;
            let acquisition_value = self.compute(&candidate, model, best_value)?;

            if acquisition_value > best_acquisition {
                best_acquisition = acquisition_value;
                best_params = candidate;
            }
        }

        Ok(best_params)
    }
}

/// Upper Confidence Bound acquisition function
#[derive(Debug)]
pub struct UpperConfidenceBound {
    beta: f64,
}

impl UpperConfidenceBound {
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }
}

impl AcquisitionFunction for UpperConfidenceBound {
    fn compute(
        &self,
        parameters: &HashMap<String, ParameterValue>,
        model: &dyn SurrogateModel,
        _best_value: f64,
    ) -> Result<f64> {
        let (mean, variance) = model.predict(parameters)?;
        let std_dev = variance.sqrt();

        Ok(mean + self.beta * std_dev)
    }

    fn optimize(
        &self,
        model: &dyn SurrogateModel,
        search_space: &SearchSpace,
        best_value: f64,
    ) -> Result<HashMap<String, ParameterValue>> {
        let num_candidates = 1000;
        let mut best_acquisition = f64::NEG_INFINITY;
        let mut best_params = search_space.sample_random()?;

        for _ in 0..num_candidates {
            let candidate = search_space.sample_random()?;
            let acquisition_value = self.compute(&candidate, model, best_value)?;

            if acquisition_value > best_acquisition {
                best_acquisition = acquisition_value;
                best_params = candidate;
            }
        }

        Ok(best_params)
    }
}

/// Factory function to create acquisition functions
pub fn create_acquisition_function(
    acquisition_type: &AcquisitionFunctionType,
) -> Box<dyn AcquisitionFunction> {
    match acquisition_type {
        AcquisitionFunctionType::ExpectedImprovement { xi } => {
            Box::new(ExpectedImprovement::new(*xi))
        },
        AcquisitionFunctionType::UpperConfidenceBound { beta } => {
            Box::new(UpperConfidenceBound::new(*beta))
        },
        AcquisitionFunctionType::ProbabilityOfImprovement { xi } => {
            // Simplified - use Expected Improvement for now
            Box::new(ExpectedImprovement::new(*xi))
        },
        AcquisitionFunctionType::EntropySearch => {
            // Simplified - use Upper Confidence Bound
            Box::new(UpperConfidenceBound::new(2.0))
        },
        AcquisitionFunctionType::KnowledgeGradient => {
            // Simplified - use Expected Improvement
            Box::new(ExpectedImprovement::new(0.01))
        },
    }
}

/// Factory function to create surrogate models
pub fn create_surrogate_model(model_type: &super::SurrogateModelType) -> Box<dyn SurrogateModel> {
    match model_type {
        super::SurrogateModelType::GaussianProcess {
            noise_level,
            length_scales,
            ..
        } => {
            let length_scale = length_scales.first().copied().unwrap_or(1.0);
            Box::new(SimpleGaussianProcess::new(*noise_level, length_scale, 1.0))
        },
        super::SurrogateModelType::RandomForest { .. } => {
            // For now, fall back to Gaussian Process
            Box::new(SimpleGaussianProcess::new(0.01, 1.0, 1.0))
        },
        super::SurrogateModelType::NeuralNetwork { .. } => {
            // For now, fall back to Gaussian Process
            Box::new(SimpleGaussianProcess::new(0.01, 1.0, 1.0))
        },
        super::SurrogateModelType::TPE { .. } => {
            // For now, fall back to Gaussian Process
            Box::new(SimpleGaussianProcess::new(0.01, 1.0, 1.0))
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::hyperopt::{KernelType, SurrogateModelType};

    #[test]
    fn test_simple_gaussian_process() {
        let mut gp = SimpleGaussianProcess::new(0.01, 1.0, 1.0);

        // Create test data
        let mut params1 = HashMap::new();
        params1.insert("x".to_string(), ParameterValue::Float(0.0));
        let mut params2 = HashMap::new();
        params2.insert("x".to_string(), ParameterValue::Float(1.0));

        let observations = vec![(params1.clone(), 0.5), (params2.clone(), 1.5)];

        gp.fit(&observations).unwrap();

        let (mean, variance) = gp.predict(&params1).unwrap();
        assert!(mean >= 0.0);
        assert!(variance > 0.0);
    }

    #[test]
    fn test_expected_improvement() {
        let ei = ExpectedImprovement::new(0.01);
        let gp = SimpleGaussianProcess::new(0.01, 1.0, 1.0);

        let mut params = HashMap::new();
        params.insert("x".to_string(), ParameterValue::Float(0.5));

        let acquisition_value = ei.compute(&params, &gp, 1.0).unwrap();
        assert!(acquisition_value >= 0.0);
    }

    #[test]
    fn test_upper_confidence_bound() {
        let ucb = UpperConfidenceBound::new(2.0);
        let gp = SimpleGaussianProcess::new(0.01, 1.0, 1.0);

        let mut params = HashMap::new();
        params.insert("x".to_string(), ParameterValue::Float(0.5));

        let acquisition_value = ucb.compute(&params, &gp, 1.0).unwrap();
        assert!(acquisition_value.is_finite());
    }

    #[test]
    fn test_acquisition_function_factory() {
        let ei_type = AcquisitionFunctionType::ExpectedImprovement { xi: 0.01 };
        let _ei = create_acquisition_function(&ei_type);

        let ucb_type = AcquisitionFunctionType::UpperConfidenceBound { beta: 2.0 };
        let _ucb = create_acquisition_function(&ucb_type);
    }

    #[test]
    fn test_surrogate_model_factory() {
        let gp_type = SurrogateModelType::GaussianProcess {
            kernel: KernelType::RBF,
            noise_level: 0.01,
            length_scales: vec![1.0],
        };
        let _gp = create_surrogate_model(&gp_type);
    }
}
