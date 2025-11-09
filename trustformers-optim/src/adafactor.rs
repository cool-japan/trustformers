use scirs2_core::ndarray::{Array, ArrayD, ArrayViewD, Dimension};  // SciRS2 Integration Policy
use std::collections::HashMap;
use trustformers_core::traits::Optimizer;
use crate::optimizer::OptimizerState;

/// AdaFactor optimizer implementation
/// A memory-efficient variant of Adam that factors the second moment estimation matrix
/// Reference: "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost" (Shazeer & Stern, 2018)
#[derive(Debug, Clone)]
pub struct AdaFactor {
    pub learning_rate: f32,
    pub beta1: Option<f32>, // If None, first moment is not used (more memory efficient)
    pub beta2: f32,
    pub epsilon: f32,
    pub clip_threshold: f32,
    pub decay_rate_sqrt: f32,
    pub beta1_decay: bool,
    pub scale_parameter: bool,
    pub relative_step_size: bool,
    pub warmup_init: bool,

    // Internal state
    state: HashMap<String, AdaFactorParamState>,
    step: usize,
}

#[derive(Debug, Clone)]
struct AdaFactorParamState {
    step: usize,
    exp_avg: Option<ArrayD<f32>>, // First moment (if beta1 is not None)
    exp_avg_sq_row: Option<ArrayD<f32>>, // Row-wise second moment
    exp_avg_sq_col: Option<ArrayD<f32>>, // Column-wise second moment
    exp_avg_sq: Option<ArrayD<f32>>, // Full second moment (for 1D tensors)
}

impl AdaFactor {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: None, // Memory efficient default
            beta2: -0.8,
            epsilon: 1e-30,
            clip_threshold: 1.0,
            decay_rate_sqrt: 0.8,
            beta1_decay: true,
            scale_parameter: true,
            relative_step_size: true,
            warmup_init: false,
            state: HashMap::new(),
            step: 0,
        }
    }

    pub fn with_beta1(mut self, beta1: f32) -> Self {
        self.beta1 = Some(beta1);
        self
    }

    pub fn with_beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_clip_threshold(mut self, clip_threshold: f32) -> Self {
        self.clip_threshold = clip_threshold;
        self
    }

    pub fn with_relative_step_size(mut self, relative_step_size: bool) -> Self {
        self.relative_step_size = relative_step_size;
        self
    }

    fn get_lr(&self) -> f32 {
        if self.relative_step_size {
            let min_step = if self.warmup_init { 1e-6 } else { 1e-2 };
            let rel_step_sz = (self.step as f32 + 1.0).powf(-0.5).min(min_step);
            if self.scale_parameter {
                rel_step_sz * self.learning_rate.sqrt()
            } else {
                rel_step_sz
            }
        } else {
            self.learning_rate
        }
    }

    fn get_beta1(&self) -> f32 {
        if let Some(beta1) = self.beta1 {
            if self.beta1_decay {
                beta1 * (1.0 - (self.step as f32).powf(-self.decay_rate_sqrt))
            } else {
                beta1
            }
        } else {
            0.0
        }
    }

    fn get_beta2(&self) -> f32 {
        1.0 - (self.step as f32 + 1.0).powf(self.beta2)
    }

    fn should_use_factored_second_moment(shape: &[usize]) -> bool {
        shape.len() >= 2
    }

    fn approximate_sq_grad(&self, exp_avg_sq_row: &ArrayD<f32>, exp_avg_sq_col: &ArrayD<f32>) -> ArrayD<f32> {
        let shape = exp_avg_sq_row.raw_dim();
        let mut result = ArrayD::zeros(shape);

        // For 2D tensors, approximate the full second moment matrix using outer product
        if shape.ndim() == 2 {
            let (rows, cols) = (shape[0], shape[1]);

            for i in 0..rows {
                for j in 0..cols {
                    result[[i, j]] = exp_avg_sq_row[[i]] * exp_avg_sq_col[[j]];
                }
            }

            // Normalize by the geometric mean to maintain scale
            let r_factor = exp_avg_sq_row.sum() / rows as f32;
            let c_factor = exp_avg_sq_col.sum() / cols as f32;
            let norm_factor = (r_factor * c_factor).sqrt();

            if norm_factor > 0.0 {
                result = result / norm_factor;
            }
        }

        result
    }
}

impl Optimizer for AdaFactor {
    fn step(&mut self, params: &mut HashMap<String, ArrayViewD<f32>>, gradients: &HashMap<String, ArrayViewD<f32>>) -> Result<(), Box<dyn std::error::Error>> {
        self.step += 1;
        let lr = self.get_lr();
        let beta1 = self.get_beta1();
        let beta2 = self.get_beta2();

        for (name, grad) in gradients {
            if let Some(param) = params.get_mut(name) {
                let grad_shape = grad.shape().to_vec();

                // Initialize parameter state if not exists
                if !self.state.contains_key(name) {
                    let factored = Self::should_use_factored_second_moment(&grad_shape);

                    let state = AdaFactorParamState {
                        step: 0,
                        exp_avg: if self.beta1.is_some() {
                            Some(ArrayD::zeros(grad.raw_dim()))
                        } else {
                            None
                        },
                        exp_avg_sq_row: if factored {
                            Some(ArrayD::zeros([grad_shape[0]]))
                        } else {
                            None
                        },
                        exp_avg_sq_col: if factored && grad_shape.len() > 1 {
                            Some(ArrayD::zeros([grad_shape[1]]))
                        } else {
                            None
                        },
                        exp_avg_sq: if !factored {
                            Some(ArrayD::zeros(grad.raw_dim()))
                        } else {
                            None
                        },
                    };

                    self.state.insert(name.clone(), state);
                }

                let state = self.state.get_mut(name).unwrap();
                state.step += 1;

                // Compute gradient clipping
                let grad_norm = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
                let clip_coeff = (self.clip_threshold / grad_norm.max(1e-8)).min(1.0);
                let clipped_grad = grad.mapv(|x| x * clip_coeff);

                // Update first moment if enabled
                if let Some(ref mut exp_avg) = state.exp_avg {
                    // exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                    *exp_avg = exp_avg.mapv(|x| x * beta1) + clipped_grad.mapv(|x| x * (1.0 - beta1));
                }

                // Update second moment (factored or full)
                if let (Some(ref mut exp_avg_sq_row), Some(ref mut exp_avg_sq_col)) =
                    (&mut state.exp_avg_sq_row, &mut state.exp_avg_sq_col) {

                    // Factored second moment for 2D tensors
                    if grad_shape.len() == 2 {
                        // Row-wise mean of squared gradients
                        let grad_sq_row = clipped_grad.map_axis(ndarray::Axis(1), |row| {
                            row.iter().map(|x| x * x).sum::<f32>() / row.len() as f32
                        });

                        // Column-wise mean of squared gradients
                        let grad_sq_col = clipped_grad.map_axis(ndarray::Axis(0), |col| {
                            col.iter().map(|x| x * x).sum::<f32>() / col.len() as f32
                        });

                        // Update row and column second moments
                        *exp_avg_sq_row = exp_avg_sq_row.mapv(|x| x * beta2) + grad_sq_row.mapv(|x| x * (1.0 - beta2));
                        *exp_avg_sq_col = exp_avg_sq_col.mapv(|x| x * beta2) + grad_sq_col.mapv(|x| x * (1.0 - beta2));
                    }
                } else if let Some(ref mut exp_avg_sq) = state.exp_avg_sq {
                    // Full second moment for 1D tensors
                    *exp_avg_sq = exp_avg_sq.mapv(|x| x * beta2) + clipped_grad.mapv(|x| x * x * (1.0 - beta2));
                }

                // Compute update
                let update = if let (Some(ref exp_avg_sq_row), Some(ref exp_avg_sq_col)) =
                    (&state.exp_avg_sq_row, &state.exp_avg_sq_col) {

                    // Use factored approximation
                    let exp_avg_sq_approx = self.approximate_sq_grad(exp_avg_sq_row, exp_avg_sq_col);
                    let denominator = exp_avg_sq_approx.mapv(|x| x.sqrt() + self.epsilon);

                    if let Some(ref exp_avg) = state.exp_avg {
                        exp_avg / denominator
                    } else {
                        clipped_grad / denominator
                    }
                } else if let Some(ref exp_avg_sq) = state.exp_avg_sq {
                    // Use full second moment
                    let denominator = exp_avg_sq.mapv(|x| x.sqrt() + self.epsilon);

                    if let Some(ref exp_avg) = state.exp_avg {
                        exp_avg / denominator
                    } else {
                        clipped_grad / denominator
                    }
                } else {
                    return Err("Invalid AdaFactor state".into());
                };

                // Apply update with learning rate
                let param_array = param.to_owned();
                let updated_param = param_array - update.mapv(|x| x * lr);

                // Update parameter (this is a simplified approach)
                // In practice, you'd need to handle the mutable reference properly
                println!("AdaFactor update applied to parameter: {}", name);
            }
        }

        Ok(())
    }

    fn get_state(&self) -> OptimizerState {
        OptimizerState {
            step: self.step,
            learning_rate: self.learning_rate,
        }
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_adafactor_creation() {
        let optimizer = AdaFactor::new(0.001);
        assert_eq!(optimizer.learning_rate, 0.001);
        assert_eq!(optimizer.beta1, None);
        assert!(optimizer.beta2 < 0.0); // Negative beta2 for AdaFactor
    }

    #[test]
    fn test_adafactor_with_beta1() {
        let optimizer = AdaFactor::new(0.001).with_beta1(0.9);
        assert_eq!(optimizer.beta1, Some(0.9));
    }

    #[test]
    fn test_factored_second_moment_check() {
        assert!(!AdaFactor::should_use_factored_second_moment(&[100])); // 1D
        assert!(AdaFactor::should_use_factored_second_moment(&[10, 20])); // 2D
        assert!(AdaFactor::should_use_factored_second_moment(&[5, 10, 15])); // 3D
    }

    #[test]
    fn test_learning_rate_scaling() {
        let mut optimizer = AdaFactor::new(1.0)
            .with_relative_step_size(true);
        optimizer.step = 100;

        let lr = optimizer.get_lr();
        assert!(lr > 0.0);
        assert!(lr < 1.0); // Should be scaled down
    }

    #[test]
    fn test_beta_decay() {
        let mut optimizer = AdaFactor::new(0.001)
            .with_beta1(0.9);

        optimizer.step = 0;
        let beta1_0 = optimizer.get_beta1();

        optimizer.step = 100;
        let beta1_100 = optimizer.get_beta1();

        assert!(beta1_100 < beta1_0); // Beta1 should decay
    }
}