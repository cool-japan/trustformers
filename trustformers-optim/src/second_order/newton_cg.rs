use super::lbfgs::LineSearchMethod;
use anyhow::Result;
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

/// Newton-CG (Newton-Conjugate Gradient) optimizer.
///
/// Newton-CG uses the Conjugate Gradient method to approximately solve the Newton system
/// H * d = -g, where H is the Hessian matrix, g is the gradient, and d is the search direction.
/// This avoids the need to compute and store the full Hessian matrix or its inverse.
#[derive(Debug)]
pub struct NewtonCG {
    pub learning_rate: f32,
    pub cg_tolerance: f32,
    pub cg_max_iterations: usize,
    pub damping: f32,
    pub weight_decay: f32,
    pub line_search_fn: Option<LineSearchMethod>,
    pub max_iter: usize,
    pub tolerance_grad: f32,
    pub tolerance_change: f32,

    // Internal state
    pub step: usize,
    pub prev_params: HashMap<String, Vec<f32>>,
    pub prev_grads: HashMap<String, Vec<f32>>,
    pub hessian_vector_products: HashMap<String, Vec<f32>>,
}

impl Default for NewtonCG {
    fn default() -> Self {
        Self {
            learning_rate: 1.0,
            cg_tolerance: 1e-4,
            cg_max_iterations: 50,
            damping: 1e-3,
            weight_decay: 0.0,
            line_search_fn: Some(LineSearchMethod::Backtracking),
            max_iter: 20,
            tolerance_grad: 1e-7,
            tolerance_change: 1e-9,
            step: 0,
            prev_params: HashMap::new(),
            prev_grads: HashMap::new(),
            hessian_vector_products: HashMap::new(),
        }
    }
}

impl NewtonCG {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            ..Default::default()
        }
    }

    pub fn with_config(
        learning_rate: f32,
        cg_tolerance: f32,
        cg_max_iterations: usize,
        damping: f32,
        weight_decay: f32,
        line_search_fn: Option<LineSearchMethod>,
        max_iter: usize,
    ) -> Self {
        Self {
            learning_rate,
            cg_tolerance,
            cg_max_iterations,
            damping,
            weight_decay,
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
        self.step += 1;

        // Store previous state for finite difference Hessian approximation
        if self.step > 1 {
            for (name, param) in parameters.iter() {
                self.prev_params.insert(name.clone(), param.data()?);
            }
            for (name, grad) in gradients.iter() {
                self.prev_grads.insert(name.clone(), grad.data()?);
            }
        }

        // Compute Newton-CG search direction
        let search_direction = self.compute_newton_cg_direction(parameters, gradients)?;

        // Apply the update
        for (name, param) in parameters.iter_mut() {
            let empty_vec = vec![];
            let direction = search_direction.get(name).unwrap_or(&empty_vec);
            if direction.is_empty() {
                continue;
            }

            let mut param_data = param.data()?;

            // Apply weight decay
            if self.weight_decay > 0.0 {
                for i in 0..param_data.len() {
                    param_data[i] *= 1.0 - self.weight_decay;
                }
            }

            // Apply search direction
            for i in 0..param_data.len().min(direction.len()) {
                param_data[i] += self.learning_rate * direction[i];
            }

            *param = Tensor::new(param_data)?;
        }

        Ok(())
    }

    fn compute_newton_cg_direction(
        &mut self,
        parameters: &HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Vec<f32>>> {
        let mut direction = HashMap::new();

        for (name, grad) in gradients.iter() {
            let grad_data = grad.data()?;
            let newton_direction = self.conjugate_gradient_solve(name, &grad_data, parameters)?;
            direction.insert(name.clone(), newton_direction);
        }

        Ok(direction)
    }

    fn conjugate_gradient_solve(
        &mut self,
        param_name: &str,
        gradient: &[f32],
        parameters: &HashMap<String, Tensor>,
    ) -> Result<Vec<f32>> {
        let n = gradient.len();
        let mut x = vec![0.0; n]; // Initial guess (zero vector)
        let mut r = gradient.to_vec(); // Residual: r = b - A*x = -g (since A*x = 0 initially)
        let mut p = r.clone(); // Search direction
        let mut rsold = Self::dot_product(&r, &r);

        // Conjugate gradient iterations
        for _ in 0..self.cg_max_iterations {
            // Compute A*p (Hessian-vector product)
            let ap = self.hessian_vector_product(param_name, &p, parameters)?;

            // Compute alpha
            let pap = Self::dot_product(&p, &ap);
            if pap.abs() < 1e-10 {
                break; // Avoid division by zero
            }

            let alpha = rsold / pap;

            // Update x and residual
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            let rsnew = Self::dot_product(&r, &r);

            // Check for convergence
            if rsnew.sqrt() < self.cg_tolerance {
                break;
            }

            // Update search direction
            let beta = rsnew / rsold;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }

            rsold = rsnew;
        }

        // Return negative of the solution (since we solved H*d = -g)
        Ok(x.iter().map(|&xi| -xi).collect())
    }

    fn hessian_vector_product(
        &mut self,
        param_name: &str,
        vector: &[f32],
        parameters: &HashMap<String, Tensor>,
    ) -> Result<Vec<f32>> {
        let n = vector.len();
        let mut hvp = vec![0.0; n];

        // Use finite differences to approximate Hessian-vector product
        // H*v ≈ (∇f(x + εv) - ∇f(x)) / ε
        let epsilon = 1e-4;

        if let Some(param) = parameters.get(param_name) {
            let original_data = param.data()?;

            // Compute perturbed gradient
            let mut perturbed_data = original_data.clone();
            for i in 0..n.min(perturbed_data.len()) {
                perturbed_data[i] += epsilon * vector[i];
            }

            // For simplicity, we'll use a diagonal approximation of the Hessian
            // In practice, you would compute the actual gradient at the perturbed point
            if let Some(prev_grad) = self.prev_grads.get(param_name) {
                for i in 0..n.min(prev_grad.len()) {
                    // Approximate Hessian diagonal element
                    let hessian_diag = if i < original_data.len() {
                        let param_diff = perturbed_data[i] - original_data[i];
                        if param_diff.abs() > 1e-10 {
                            prev_grad[i] / param_diff
                        } else {
                            1.0 // Fallback
                        }
                    } else {
                        1.0
                    };

                    // Add damping for numerical stability
                    let damped_hessian = hessian_diag + self.damping;
                    hvp[i] = damped_hessian * vector[i];
                }
            } else {
                // If no previous gradient, use identity matrix (with damping)
                for i in 0..n {
                    hvp[i] = (1.0 + self.damping) * vector[i];
                }
            }
        } else {
            // Fallback to identity matrix
            for i in 0..n {
                hvp[i] = vector[i];
            }
        }

        Ok(hvp)
    }

    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
    }

    pub fn reset(&mut self) {
        self.step = 0;
        self.prev_params.clear();
        self.prev_grads.clear();
        self.hessian_vector_products.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newton_cg_creation() {
        let optimizer = NewtonCG::new(0.01);
        assert_eq!(optimizer.learning_rate, 0.01);
        assert_eq!(optimizer.cg_tolerance, 1e-4);
        assert_eq!(optimizer.cg_max_iterations, 50);
        assert_eq!(optimizer.damping, 1e-3);
        assert_eq!(optimizer.step, 0);
    }

    #[test]
    fn test_newton_cg_with_config() {
        let optimizer = NewtonCG::with_config(
            0.1,
            1e-5,
            30,
            1e-2,
            0.01,
            Some(LineSearchMethod::Backtracking),
            15,
        );
        assert_eq!(optimizer.learning_rate, 0.1);
        assert_eq!(optimizer.cg_tolerance, 1e-5);
        assert_eq!(optimizer.cg_max_iterations, 30);
        assert_eq!(optimizer.damping, 1e-2);
        assert_eq!(optimizer.weight_decay, 0.01);
        assert_eq!(optimizer.max_iter, 15);
    }

    #[test]
    fn test_newton_cg_reset() {
        let mut optimizer = NewtonCG::new(0.01);
        optimizer.step = 5;
        optimizer.prev_params.insert("test".to_string(), vec![1.0, 2.0]);
        optimizer.prev_grads.insert("test".to_string(), vec![0.1, 0.2]);

        optimizer.reset();

        assert_eq!(optimizer.step, 0);
        assert!(optimizer.prev_params.is_empty());
        assert!(optimizer.prev_grads.is_empty());
        assert!(optimizer.hessian_vector_products.is_empty());
    }

    #[test]
    fn test_newton_cg_step() -> Result<(), Box<dyn std::error::Error>> {
        let mut optimizer = NewtonCG::new(0.01);
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
        // Parameters should be updated (exact values depend on CG solver)
        for i in 0..updated_data.len() {
            // Should be different from original parameters
            assert_ne!(updated_data[i], param_data[i]);
        }
        Ok(())
    }

    #[test]
    fn test_newton_cg_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = NewtonCG::dot_product(&a, &b);
        assert_eq!(result, 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
    }

    #[test]
    fn test_newton_cg_dot_product_empty() {
        let a = vec![];
        let b = vec![];
        let result = NewtonCG::dot_product(&a, &b);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_newton_cg_multiple_steps() {
        let mut optimizer = NewtonCG::new(0.01);
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

        // First step
        optimizer.step(&mut parameters, &gradients).unwrap();
        assert_eq!(optimizer.step, 1);

        // Second step (should use previous state for Hessian approximation)
        optimizer.step(&mut parameters, &gradients).unwrap();
        assert_eq!(optimizer.step, 2);

        // Verify that previous parameters and gradients are stored
        assert!(optimizer.prev_params.contains_key("param1"));
        assert!(optimizer.prev_grads.contains_key("param1"));
    }

    #[test]
    fn test_newton_cg_with_weight_decay() -> Result<(), Box<dyn std::error::Error>> {
        let mut optimizer = NewtonCG::with_config(
            0.01,
            1e-4,
            50,
            1e-3,
            0.01, // weight_decay
            Some(LineSearchMethod::Backtracking),
            20,
        );

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

        let original_params = parameters.get("param1").unwrap().data()?;
        optimizer.step(&mut parameters, &gradients).unwrap();

        let updated_params = parameters.get("param1").unwrap().data()?;

        // With weight decay, parameters should be affected
        for i in 0..updated_params.len() {
            assert_ne!(updated_params[i], original_params[i]);
        }
        Ok(())
    }
}
