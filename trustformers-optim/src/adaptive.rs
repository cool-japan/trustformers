//! # Advanced Adaptive Optimizers
//!
//! This module implements advanced adaptive optimization algorithms that extend
//! beyond basic Adam/AdamW functionality.

use crate::adam::RAdam;
use crate::lookahead::Lookahead;
use anyhow::Result;
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

/// AdaBound optimizer with adaptive bounds on learning rates.
///
/// AdaBound is an optimizer that transforms from Adam-like behavior at the beginning
/// to SGD-like behavior at the end of training by gradually tightening bounds on
/// the learning rates. This combines the fast convergence of adaptive methods
/// with the generalization of SGD.
///
/// Paper: "Adaptive Gradient Methods with Dynamic Bound of Learning Rate" (ICLR 2019)
#[derive(Debug)]
pub struct AdaBound {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub final_lr: f32,
    pub gamma: f32,

    // Internal state
    pub step: usize,
    pub momentum_states: HashMap<String, Vec<f32>>,
    pub variance_states: HashMap<String, Vec<f32>>,
}

impl Default for AdaBound {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-16,
            weight_decay: 0.0,
            final_lr: 0.1,
            gamma: 1e-3,
            step: 0,
            momentum_states: HashMap::new(),
            variance_states: HashMap::new(),
        }
    }
}

impl AdaBound {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            ..Default::default()
        }
    }

    pub fn with_config(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        final_lr: f32,
        gamma: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            final_lr,
            gamma,
            step: 0,
            momentum_states: HashMap::new(),
            variance_states: HashMap::new(),
        }
    }

    pub fn step(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<()> {
        self.step += 1;

        let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);

        // Compute dynamic bounds
        let step_size = self.learning_rate * (bias_correction2.sqrt() / bias_correction1);
        let base_lr = self.final_lr * self.learning_rate / self.learning_rate;
        let lower_bound = base_lr * (1.0 - 1.0 / (self.gamma * self.step as f32 + 1.0));
        let upper_bound = base_lr * (1.0 + 1.0 / (self.gamma * self.step as f32));

        for (name, param) in parameters.iter_mut() {
            let grad = gradients
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Missing gradient for parameter: {}", name))?;

            let mut param_data = param.data()?;
            let grad_data = grad.data()?;

            if param_data.len() != grad_data.len() {
                return Err(anyhow::anyhow!(
                    "Parameter and gradient size mismatch for: {}",
                    name
                ));
            }

            // Initialize states if needed
            if !self.momentum_states.contains_key(name) {
                let zeros = vec![0.0; param_data.len()];
                self.momentum_states.insert(name.clone(), zeros.clone());
                self.variance_states.insert(name.clone(), zeros);
            }

            let momentum_state = self
                .momentum_states
                .get_mut(name)
                .ok_or_else(|| anyhow::anyhow!("Momentum state not found"))?;
            let variance_state = self
                .variance_states
                .get_mut(name)
                .ok_or_else(|| anyhow::anyhow!("Variance state not found"))?;

            for i in 0..param_data.len() {
                let mut grad_val = grad_data[i];

                // Add weight decay
                if self.weight_decay > 0.0 {
                    grad_val += self.weight_decay * param_data[i];
                }

                // Update biased first moment estimate
                momentum_state[i] = self.beta1 * momentum_state[i] + (1.0 - self.beta1) * grad_val;

                // Update biased second moment estimate
                variance_state[i] =
                    self.beta2 * variance_state[i] + (1.0 - self.beta2) * grad_val * grad_val;

                // Compute bias-corrected moments
                let corrected_momentum = momentum_state[i] / bias_correction1;
                let corrected_variance = variance_state[i] / bias_correction2;

                // Compute adaptive learning rate
                let denominator = corrected_variance.sqrt() + self.epsilon;
                let adaptive_lr = step_size / denominator;

                // Apply bounds to learning rate
                let bounded_lr = adaptive_lr.clamp(lower_bound, upper_bound);

                // Update parameter
                param_data[i] -= bounded_lr * corrected_momentum;
            }

            *param = Tensor::new(param_data)?;
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        self.step = 0;
        self.momentum_states.clear();
        self.variance_states.clear();
    }

    pub fn get_bounds(&self) -> (f32, f32) {
        let base_lr = self.final_lr * self.learning_rate / self.learning_rate;
        let lower_bound = base_lr * (1.0 - 1.0 / (self.gamma * self.step as f32 + 1.0));
        let upper_bound = base_lr * (1.0 + 1.0 / (self.gamma * self.step as f32));
        (lower_bound, upper_bound)
    }
}

/// AMSBound optimizer - AdaBound with AMSGrad-style variance tracking.
///
/// AMSBound combines AdaBound's learning rate bounds with AMSGrad's
/// non-increasing second moment estimates for better convergence.
#[derive(Debug)]
pub struct AMSBound {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub final_lr: f32,
    pub gamma: f32,

    // Internal state
    pub step: usize,
    pub momentum_states: HashMap<String, Vec<f32>>,
    pub variance_states: HashMap<String, Vec<f32>>,
    pub max_variance_states: HashMap<String, Vec<f32>>, // AMSGrad-style max variance
}

impl Default for AMSBound {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-16,
            weight_decay: 0.0,
            final_lr: 0.1,
            gamma: 1e-3,
            step: 0,
            momentum_states: HashMap::new(),
            variance_states: HashMap::new(),
            max_variance_states: HashMap::new(),
        }
    }
}

impl AMSBound {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            ..Default::default()
        }
    }

    pub fn with_config(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        final_lr: f32,
        gamma: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            final_lr,
            gamma,
            step: 0,
            momentum_states: HashMap::new(),
            variance_states: HashMap::new(),
            max_variance_states: HashMap::new(),
        }
    }

    pub fn step(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<()> {
        self.step += 1;

        let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);

        // Compute dynamic bounds
        let step_size = self.learning_rate * (bias_correction2.sqrt() / bias_correction1);
        let base_lr = self.final_lr * self.learning_rate / self.learning_rate;
        let lower_bound = base_lr * (1.0 - 1.0 / (self.gamma * self.step as f32 + 1.0));
        let upper_bound = base_lr * (1.0 + 1.0 / (self.gamma * self.step as f32));

        for (name, param) in parameters.iter_mut() {
            let grad = gradients
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Missing gradient for parameter: {}", name))?;

            let mut param_data = param.data()?;
            let grad_data = grad.data()?;

            if param_data.len() != grad_data.len() {
                return Err(anyhow::anyhow!(
                    "Parameter and gradient size mismatch for: {}",
                    name
                ));
            }

            // Initialize states if needed
            if !self.momentum_states.contains_key(name) {
                let zeros = vec![0.0; param_data.len()];
                self.momentum_states.insert(name.clone(), zeros.clone());
                self.variance_states.insert(name.clone(), zeros.clone());
                self.max_variance_states.insert(name.clone(), zeros);
            }

            let momentum_state = self
                .momentum_states
                .get_mut(name)
                .ok_or_else(|| anyhow::anyhow!("Momentum state not found"))?;
            let variance_state = self
                .variance_states
                .get_mut(name)
                .ok_or_else(|| anyhow::anyhow!("Variance state not found"))?;
            let max_variance_state = self
                .max_variance_states
                .get_mut(name)
                .ok_or_else(|| anyhow::anyhow!("Max variance state not found"))?;

            for i in 0..param_data.len() {
                let mut grad_val = grad_data[i];

                // Add weight decay
                if self.weight_decay > 0.0 {
                    grad_val += self.weight_decay * param_data[i];
                }

                // Update biased first moment estimate
                momentum_state[i] = self.beta1 * momentum_state[i] + (1.0 - self.beta1) * grad_val;

                // Update biased second moment estimate
                variance_state[i] =
                    self.beta2 * variance_state[i] + (1.0 - self.beta2) * grad_val * grad_val;

                // Update maximum variance (AMSGrad component)
                max_variance_state[i] = max_variance_state[i].max(variance_state[i]);

                // Compute bias-corrected moments
                let corrected_momentum = momentum_state[i] / bias_correction1;
                let corrected_max_variance = max_variance_state[i] / bias_correction2;

                // Compute adaptive learning rate using max variance
                let denominator = corrected_max_variance.sqrt() + self.epsilon;
                let adaptive_lr = step_size / denominator;

                // Apply bounds to learning rate
                let bounded_lr = adaptive_lr.clamp(lower_bound, upper_bound);

                // Update parameter
                param_data[i] -= bounded_lr * corrected_momentum;
            }

            *param = Tensor::new(param_data)?;
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        self.step = 0;
        self.momentum_states.clear();
        self.variance_states.clear();
        self.max_variance_states.clear();
    }

    pub fn get_bounds(&self) -> (f32, f32) {
        let base_lr = self.final_lr * self.learning_rate / self.learning_rate;
        let lower_bound = base_lr * (1.0 - 1.0 / (self.gamma * self.step as f32 + 1.0));
        let upper_bound = base_lr * (1.0 + 1.0 / (self.gamma * self.step as f32));
        (lower_bound, upper_bound)
    }
}

/// Ranger optimizer - RAdam with Lookahead meta-optimization.
///
/// Ranger combines the Rectified Adam (RAdam) optimizer with Lookahead meta-optimization
/// to achieve both fast convergence and improved stability. RAdam provides variance
/// rectification while Lookahead reduces variance in the optimization process.
///
/// This is a powerful combination that often outperforms both RAdam and Lookahead
/// individually, especially in challenging optimization landscapes.
pub type Ranger = Lookahead<RAdam>;

/// Creates a new Ranger optimizer with default hyperparameters.
///
/// # Arguments
///
/// * `learning_rate` - Base learning rate for RAdam
/// * `k` - Lookahead update frequency (typical: 5-6)
/// * `alpha` - Lookahead interpolation factor (typical: 0.5)
///
/// # Example
///
/// ```rust,no_run
/// use trustformers_optim::create_ranger;
///
/// let optimizer = create_ranger(1e-3, 5, 0.5);
/// ```
pub fn create_ranger(learning_rate: f32, k: usize, alpha: f32) -> Ranger {
    let radam = RAdam::new(learning_rate, (0.9, 0.999), 1e-8, 0.0);
    Lookahead::new(radam, k, alpha)
}

/// Creates a new Ranger optimizer with custom RAdam configuration.
///
/// # Arguments
///
/// * `learning_rate` - Learning rate for RAdam
/// * `beta1` - RAdam beta1 parameter (momentum)
/// * `beta2` - RAdam beta2 parameter (variance)
/// * `epsilon` - RAdam epsilon for numerical stability
/// * `weight_decay` - RAdam weight decay
/// * `k` - Lookahead update frequency
/// * `alpha` - Lookahead interpolation factor
///
/// # Example
///
/// ```rust,no_run
/// use trustformers_optim::create_ranger_with_config;
///
/// let optimizer = create_ranger_with_config(
///     1e-3,   // learning_rate
///     0.9,    // beta1
///     0.999,  // beta2
///     1e-8,   // epsilon
///     0.01,   // weight_decay
///     5,      // k
///     0.5,    // alpha
/// );
/// ```
pub fn create_ranger_with_config(
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    k: usize,
    alpha: f32,
) -> Ranger {
    let radam = RAdam::new(learning_rate, (beta1, beta2), epsilon, weight_decay);
    Lookahead::new(radam, k, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adabound_creation() {
        let optimizer = AdaBound::new(0.001);
        assert_eq!(optimizer.learning_rate, 0.001);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
        assert_eq!(optimizer.step, 0);
    }

    #[test]
    fn test_adabound_with_config() {
        let optimizer = AdaBound::with_config(0.01, 0.95, 0.9999, 1e-8, 0.01, 0.05, 1e-2);
        assert_eq!(optimizer.learning_rate, 0.01);
        assert_eq!(optimizer.beta1, 0.95);
        assert_eq!(optimizer.final_lr, 0.05);
        assert_eq!(optimizer.gamma, 1e-2);
    }

    #[test]
    fn test_adabound_bounds() {
        let mut optimizer = AdaBound::new(0.001);
        optimizer.step = 100;
        let (lower, upper) = optimizer.get_bounds();
        assert!(lower < upper);
        assert!(lower >= 0.0);
    }

    #[test]
    fn test_adabound_step() {
        let mut optimizer = AdaBound::new(0.001);
        let mut parameters = HashMap::new();
        let mut gradients = HashMap::new();

        let param_data = vec![1.0, 2.0, 3.0, 4.0];
        let grad_data = vec![0.1, 0.2, 0.3, 0.4];

        parameters.insert(
            "layer1".to_string(),
            Tensor::new(param_data.clone()).unwrap(),
        );
        gradients.insert(
            "layer1".to_string(),
            Tensor::new(grad_data.clone()).unwrap(),
        );

        optimizer.step(&mut parameters, &gradients).unwrap();

        assert_eq!(optimizer.step, 1);
        assert!(optimizer.momentum_states.contains_key("layer1"));
        assert!(optimizer.variance_states.contains_key("layer1"));

        let updated_data = parameters.get("layer1").unwrap().data().unwrap();
        // Parameters should have changed
        for i in 0..updated_data.len() {
            assert_ne!(updated_data[i], param_data[i]);
        }
    }

    #[test]
    fn test_amsbound_creation() {
        let optimizer = AMSBound::new(0.001);
        assert_eq!(optimizer.learning_rate, 0.001);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
        assert_eq!(optimizer.step, 0);
    }

    #[test]
    fn test_amsbound_step() {
        let mut optimizer = AMSBound::new(0.001);
        let mut parameters = HashMap::new();
        let mut gradients = HashMap::new();

        let param_data = vec![1.0, 2.0, 3.0, 4.0];
        let grad_data = vec![0.1, 0.2, 0.3, 0.4];

        parameters.insert(
            "layer1".to_string(),
            Tensor::new(param_data.clone()).unwrap(),
        );
        gradients.insert(
            "layer1".to_string(),
            Tensor::new(grad_data.clone()).unwrap(),
        );

        optimizer.step(&mut parameters, &gradients).unwrap();

        assert_eq!(optimizer.step, 1);
        assert!(optimizer.momentum_states.contains_key("layer1"));
        assert!(optimizer.variance_states.contains_key("layer1"));
        assert!(optimizer.max_variance_states.contains_key("layer1"));

        let updated_data = parameters.get("layer1").unwrap().data().unwrap();
        // Parameters should have changed
        for i in 0..updated_data.len() {
            assert_ne!(updated_data[i], param_data[i]);
        }
    }
}
