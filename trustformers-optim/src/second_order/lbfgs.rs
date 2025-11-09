use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use trustformers_core::tensor::Tensor;

/// Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer.
///
/// L-BFGS is a quasi-Newton method that approximates the second-order derivative
/// information using only first-order gradients. It maintains a limited history
/// of gradient and parameter updates to approximate the inverse Hessian matrix.
#[derive(Debug)]
pub struct LBFGS {
    pub learning_rate: f32,
    pub history_size: usize,
    pub line_search_fn: Option<LineSearchMethod>,
    pub max_iter: usize,
    pub tolerance_grad: f32,
    pub tolerance_change: f32,

    // Internal state
    pub step: usize,
    pub s_history: VecDeque<HashMap<String, Vec<f32>>>, // parameter differences
    pub y_history: VecDeque<HashMap<String, Vec<f32>>>, // gradient differences
    pub rho_history: VecDeque<f32>,                     // 1 / (y^T s)
    pub prev_params: HashMap<String, Vec<f32>>,
    pub prev_grads: HashMap<String, Vec<f32>>,
}

#[derive(Debug, Clone)]
pub enum LineSearchMethod {
    None,
    StrongWolfe,
    Backtracking,
}

impl Default for LBFGS {
    fn default() -> Self {
        Self {
            learning_rate: 1.0,
            history_size: 10,
            line_search_fn: Some(LineSearchMethod::StrongWolfe),
            max_iter: 20,
            tolerance_grad: 1e-7,
            tolerance_change: 1e-9,
            step: 0,
            s_history: VecDeque::new(),
            y_history: VecDeque::new(),
            rho_history: VecDeque::new(),
            prev_params: HashMap::new(),
            prev_grads: HashMap::new(),
        }
    }
}

impl LBFGS {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            ..Default::default()
        }
    }

    pub fn with_config(
        learning_rate: f32,
        history_size: usize,
        line_search_fn: Option<LineSearchMethod>,
        max_iter: usize,
    ) -> Self {
        Self {
            learning_rate,
            history_size,
            line_search_fn,
            max_iter,
            ..Default::default()
        }
    }

    pub fn step(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<()> {
        // First step - store current state
        if self.step == 0 {
            for (name, param) in parameters.iter() {
                self.prev_params.insert(name.clone(), param.data()?);
            }
            for (name, grad) in gradients.iter() {
                self.prev_grads.insert(name.clone(), grad.data()?);
            }

            // Simple gradient descent for first step
            for (name, param) in parameters.iter_mut() {
                let grad = gradients
                    .get(name)
                    .ok_or_else(|| anyhow::anyhow!("Missing gradient for parameter: {}", name))?;
                let mut param_data = param.data()?;
                let grad_data = grad.data()?;

                for i in 0..param_data.len() {
                    param_data[i] -= self.learning_rate * grad_data[i];
                }

                *param = Tensor::new(param_data)?;
            }

            self.step += 1;
            return Ok(());
        }

        // Subsequent steps - use L-BFGS
        let mut s_k = HashMap::new();
        let mut y_k = HashMap::new();

        // Compute parameter and gradient differences
        for (name, param) in parameters.iter() {
            let param_data = param.data()?;
            let prev_param = self.prev_params.get(name).unwrap();

            let s: Vec<f32> =
                param_data.iter().zip(prev_param.iter()).map(|(p, prev_p)| p - prev_p).collect();
            s_k.insert(name.clone(), s);
        }

        for (name, grad) in gradients.iter() {
            let grad_data = grad.data()?;
            let prev_grad = self.prev_grads.get(name).unwrap();

            let y: Vec<f32> =
                grad_data.iter().zip(prev_grad.iter()).map(|(g, prev_g)| g - prev_g).collect();
            y_k.insert(name.clone(), y);
        }

        // Compute rho = 1 / (y^T s)
        let mut rho = 0.0;
        for name in parameters.keys() {
            let s = s_k.get(name).unwrap();
            let y = y_k.get(name).unwrap();

            rho += s.iter().zip(y.iter()).map(|(s_i, y_i)| s_i * y_i).sum::<f32>();
        }

        if rho.abs() < 1e-10 {
            // Skip this update if rho is too small
            self.step += 1;
            return Ok(());
        }

        rho = 1.0 / rho;

        // Store in history
        self.s_history.push_back(s_k);
        self.y_history.push_back(y_k);
        self.rho_history.push_back(rho);

        // Maintain history size
        if self.s_history.len() > self.history_size {
            self.s_history.pop_front();
            self.y_history.pop_front();
            self.rho_history.pop_front();
        }

        // Compute search direction using two-loop recursion
        let search_direction = self.compute_search_direction(gradients)?;

        // Apply update
        for (name, param) in parameters.iter_mut() {
            let direction = search_direction.get(name).unwrap();
            let mut param_data = param.data()?;

            for i in 0..param_data.len() {
                param_data[i] -= self.learning_rate * direction[i];
            }

            *param = Tensor::new(param_data)?;
        }

        // Update stored state
        for (name, param) in parameters.iter() {
            self.prev_params.insert(name.clone(), param.data()?);
        }
        for (name, grad) in gradients.iter() {
            self.prev_grads.insert(name.clone(), grad.data()?);
        }

        self.step += 1;
        Ok(())
    }

    fn compute_search_direction(
        &self,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Vec<f32>>> {
        let mut q: HashMap<String, Vec<f32>> = HashMap::new();

        // Initialize q with current gradients
        for (name, grad) in gradients.iter() {
            q.insert(name.clone(), grad.data()?);
        }

        let history_len = self.s_history.len();
        let mut alpha = vec![0.0; history_len];

        // First loop (backward)
        for i in (0..history_len).rev() {
            let rho_i = self.rho_history[i];
            let s_i = &self.s_history[i];

            let mut alpha_i = 0.0;
            for name in gradients.keys() {
                let s_i_param = s_i.get(name).unwrap();
                let q_param = q.get(name).unwrap();

                alpha_i +=
                    s_i_param.iter().zip(q_param.iter()).map(|(s, q_val)| s * q_val).sum::<f32>();
            }
            alpha_i *= rho_i;
            alpha[i] = alpha_i;

            // Update q
            for name in gradients.keys() {
                let y_i_param = self.y_history[i].get(name).unwrap();
                let q_param = q.get_mut(name).unwrap();

                for j in 0..q_param.len() {
                    q_param[j] -= alpha_i * y_i_param[j];
                }
            }
        }

        // Scale by initial Hessian approximation (H_0 = I / gamma)
        if !self.s_history.is_empty() {
            let recent_idx = self.s_history.len() - 1;
            let recent_s = &self.s_history[recent_idx];
            let recent_y = &self.y_history[recent_idx];

            let mut s_dot_y = 0.0;
            let mut y_dot_y = 0.0;

            for name in gradients.keys() {
                let s_param = recent_s.get(name).unwrap();
                let y_param = recent_y.get(name).unwrap();

                s_dot_y += s_param.iter().zip(y_param.iter()).map(|(s, y)| s * y).sum::<f32>();
                y_dot_y += y_param.iter().map(|y| y * y).sum::<f32>();
            }

            if y_dot_y > 1e-10 {
                let gamma = s_dot_y / y_dot_y;
                for (_, q_param) in q.iter_mut() {
                    for val in q_param.iter_mut() {
                        *val *= gamma;
                    }
                }
            }
        }

        // Second loop (forward)
        for i in 0..history_len {
            let rho_i = self.rho_history[i];
            let y_i = &self.y_history[i];

            let mut beta = 0.0;
            for name in gradients.keys() {
                let y_i_param = y_i.get(name).unwrap();
                let q_param = q.get(name).unwrap();

                beta +=
                    y_i_param.iter().zip(q_param.iter()).map(|(y, q_val)| y * q_val).sum::<f32>();
            }
            beta *= rho_i;

            let correction = alpha[i] - beta;

            // Update q
            for name in gradients.keys() {
                let s_i_param = self.s_history[i].get(name).unwrap();
                let q_param = q.get_mut(name).unwrap();

                for j in 0..q_param.len() {
                    q_param[j] += correction * s_i_param[j];
                }
            }
        }

        Ok(q)
    }

    pub fn reset(&mut self) {
        self.step = 0;
        self.s_history.clear();
        self.y_history.clear();
        self.rho_history.clear();
        self.prev_params.clear();
        self.prev_grads.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbfgs_creation() {
        let optimizer = LBFGS::new(0.01);
        assert_eq!(optimizer.learning_rate, 0.01);
        assert_eq!(optimizer.history_size, 10);
        assert_eq!(optimizer.step, 0);
    }

    #[test]
    fn test_lbfgs_with_config() {
        let optimizer = LBFGS::with_config(0.1, 5, None, 10);
        assert_eq!(optimizer.learning_rate, 0.1);
        assert_eq!(optimizer.history_size, 5);
        assert_eq!(optimizer.max_iter, 10);
    }

    #[test]
    fn test_lbfgs_reset() {
        let mut optimizer = LBFGS::new(0.01);
        optimizer.step = 5;
        optimizer.reset();
        assert_eq!(optimizer.step, 0);
        assert!(optimizer.s_history.is_empty());
        assert!(optimizer.y_history.is_empty());
        assert!(optimizer.rho_history.is_empty());
    }

    #[test]
    fn test_lbfgs_first_step() -> Result<(), Box<dyn std::error::Error>> {
        let mut optimizer = LBFGS::new(0.01);
        let mut parameters = HashMap::new();
        let mut gradients = HashMap::new();

        let param_data = vec![1.0, 2.0, 3.0];
        let grad_data = vec![0.1, 0.2, 0.3];

        parameters.insert(
            "param1".to_string(),
            Tensor::new(param_data.clone()).unwrap(),
        );
        gradients.insert(
            "param1".to_string(),
            Tensor::new(grad_data.clone()).unwrap(),
        );

        optimizer.step(&mut parameters, &gradients).unwrap();

        assert_eq!(optimizer.step, 1);

        let updated_data = parameters.get("param1").unwrap().data()?;
        for i in 0..updated_data.len() {
            let expected = param_data[i] - 0.01 * grad_data[i];
            assert!((updated_data[i] - expected).abs() < 1e-6);
        }
        Ok(())
    }
}
