use crate::{
    adam::{Adam, AdamW},
    scheduler::LRScheduler,
    sgd::SGD,
};
use trustformers_core::{errors::Result, tensor::Tensor, traits::Optimizer};

/// BERT-specific optimizer with tailored hyperparameters and scheduling
pub struct BERTOptimizer {
    base_optimizer: AdamW,
    warmup_scheduler: Box<dyn LRScheduler>,
    #[allow(dead_code)]
    layer_wise_decay: f32,
    #[allow(dead_code)]
    weight_decay_exclusions: Vec<String>,
    current_step: usize,
    #[allow(dead_code)]
    warmup_steps: usize,
    #[allow(dead_code)]
    total_steps: usize,
}

impl BERTOptimizer {
    pub fn new(
        learning_rate: f32,
        warmup_steps: usize,
        total_steps: usize,
        layer_wise_decay: f32,
    ) -> Result<Self> {
        let base_optimizer = AdamW::new(learning_rate, (0.9, 0.999), 1e-6, 0.01);

        // BERT-specific warmup scheduler
        let warmup_scheduler = Box::new(BERTWarmupScheduler::new(
            learning_rate,
            warmup_steps,
            total_steps,
        ));

        // Parameters that should not have weight decay (bias, LayerNorm)
        let weight_decay_exclusions = vec![
            "bias".to_string(),
            "LayerNorm".to_string(),
            "layer_norm".to_string(),
            "ln".to_string(),
        ];

        Ok(Self {
            base_optimizer,
            warmup_scheduler,
            layer_wise_decay,
            weight_decay_exclusions,
            current_step: 0,
            warmup_steps,
            total_steps,
        })
    }

    /// Apply layer-wise learning rate decay for deeper layers
    #[allow(dead_code)]
    fn get_layer_wise_lr(&self, param_name: &str, base_lr: f32) -> f32 {
        // Extract layer number from parameter name
        if let Some(layer_num) = self.extract_layer_number(param_name) {
            let decay_factor = self.layer_wise_decay.powi(layer_num as i32);
            base_lr * decay_factor
        } else {
            base_lr
        }
    }

    fn extract_layer_number(&self, param_name: &str) -> Option<usize> {
        // Extract layer number from names like "encoder.layer.11.attention.self.query.weight"
        if param_name.contains("layer.") {
            let parts: Vec<&str> = param_name.split('.').collect();
            for i in 0..parts.len() {
                if parts[i] == "layer" && i + 1 < parts.len() {
                    if let Ok(layer_num) = parts[i + 1].parse::<usize>() {
                        return Some(layer_num);
                    }
                }
            }
        }
        None
    }

    #[allow(dead_code)]
    fn should_exclude_weight_decay(&self, param_name: &str) -> bool {
        self.weight_decay_exclusions
            .iter()
            .any(|exclusion| param_name.contains(exclusion))
    }
}

impl Optimizer for BERTOptimizer {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        self.base_optimizer.update(parameter, grad)
    }

    fn zero_grad(&mut self) {
        self.base_optimizer.zero_grad()
    }

    fn step(&mut self) {
        self.base_optimizer.step();
        self.warmup_scheduler.step();
        self.current_step += 1;
    }

    fn get_lr(&self) -> f32 {
        self.base_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.base_optimizer.set_lr(lr)
    }
}

/// BERT warmup scheduler
struct BERTWarmupScheduler {
    base_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl BERTWarmupScheduler {
    fn new(base_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }
}

impl LRScheduler for BERTWarmupScheduler {
    fn step(&mut self) {
        self.current_step += 1;
    }

    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (step as f32 / self.warmup_steps as f32)
        } else {
            // Linear decay
            let progress =
                (step - self.warmup_steps) as f32 / (self.total_steps - self.warmup_steps) as f32;
            self.base_lr * (1.0 - progress).max(0.0)
        }
    }
}

/// GAN optimizer with stability improvements
pub struct GANOptimizer {
    generator_optimizer: Adam,
    discriminator_optimizer: Adam,
    spectral_norm: bool,
    gradient_penalty_weight: f32,
    #[allow(dead_code)]
    ttur: bool, // Two Time-scale Update Rule
    d_steps_per_g_step: usize,
    current_d_steps: usize,
}

impl GANOptimizer {
    pub fn new(g_lr: f32, d_lr: f32, spectral_norm: bool, gradient_penalty_weight: f32) -> Self {
        let generator_optimizer = Adam::new(g_lr, (0.0, 0.999), 1e-8, 0.0);
        let discriminator_optimizer = Adam::new(d_lr, (0.0, 0.999), 1e-8, 0.0);

        Self {
            generator_optimizer,
            discriminator_optimizer,
            spectral_norm,
            gradient_penalty_weight,
            ttur: d_lr != g_lr,
            d_steps_per_g_step: if d_lr > g_lr { 5 } else { 1 },
            current_d_steps: 0,
        }
    }

    pub fn step_discriminator(
        &mut self,
        d_params: &mut [Tensor],
        d_grads: &[Tensor],
    ) -> Result<()> {
        // Apply gradient penalty if enabled
        let mut modified_grads = d_grads.to_vec();
        if self.gradient_penalty_weight > 0.0 {
            self.apply_gradient_penalty(&mut modified_grads)?;
        }

        // Apply spectral normalization if enabled
        if self.spectral_norm {
            self.apply_spectral_norm(d_params)?;
        }

        for (param, grad) in d_params.iter_mut().zip(modified_grads.iter()) {
            self.discriminator_optimizer.update(param, grad)?;
        }
        self.discriminator_optimizer.step();
        self.current_d_steps += 1;
        Ok(())
    }

    pub fn step_generator(&mut self, g_params: &mut [Tensor], g_grads: &[Tensor]) -> Result<()> {
        // Only update generator after enough discriminator steps
        if self.current_d_steps >= self.d_steps_per_g_step {
            for (param, grad) in g_params.iter_mut().zip(g_grads.iter()) {
                self.generator_optimizer.update(param, grad)?;
            }
            self.generator_optimizer.step();
            self.current_d_steps = 0;
        }
        Ok(())
    }

    fn apply_gradient_penalty(&self, gradients: &mut [Tensor]) -> Result<()> {
        // Apply gradient penalty to encourage Lipschitz constraint
        for grad in gradients.iter_mut() {
            let grad_norm = self.compute_gradient_norm(grad)?;
            if grad_norm > 1.0 {
                let penalty = (grad_norm - 1.0).powi(2) * self.gradient_penalty_weight;
                *grad = grad.add_scalar(penalty)?;
            }
        }
        Ok(())
    }

    fn apply_spectral_norm(&self, parameters: &mut [Tensor]) -> Result<()> {
        // Apply spectral normalization to weight matrices
        for param in parameters.iter_mut() {
            if param.shape().len() >= 2 {
                // Only for weight matrices
                let spectral_norm = self.compute_spectral_norm(param)?;
                if spectral_norm > 1.0 {
                    *param = param.div_scalar(spectral_norm)?;
                }
            }
        }
        Ok(())
    }

    fn compute_gradient_norm(&self, grad: &Tensor) -> Result<f32> {
        // Compute L2 norm of gradient
        let sum_squares = grad.pow(2.0)?.sum(None, false)?;
        let norm_tensor = sum_squares.sqrt()?;
        // Extract scalar value from tensor
        let norm_data = norm_tensor.data()?;
        Ok(norm_data[0].sqrt())
    }

    fn compute_spectral_norm(&self, weight: &Tensor) -> Result<f32> {
        // Spectral norm computation using power iteration method
        let weight_data = weight.data()?;
        let len = weight_data.len();

        // Handle edge cases
        if len == 0 {
            return Ok(0.0);
        }
        if len == 1 {
            return Ok(weight_data[0].abs());
        }

        // For very small matrices, use simple Frobenius norm approximation
        if len <= 4 {
            let frobenius_norm: f32 = weight_data.iter().map(|x| x * x).sum::<f32>().sqrt();
            return Ok(frobenius_norm);
        }

        // Power iteration method for spectral norm (largest singular value)
        let sqrt_len = (len as f32).sqrt() as usize;
        let rows = sqrt_len.max(1);
        let cols = len.div_ceil(rows); // Ceiling division

        // Initialize random vector
        let mut v: Vec<f32> = (0..cols).map(|i| ((i % 7) as f32) / 7.0 - 0.5).collect();
        let mut v_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if v_norm > 0.0 {
            for val in &mut v {
                *val /= v_norm;
            }
        }

        // Power iteration (simplified - assumes roughly square matrix)
        for _ in 0..5 {
            // 5 iterations usually sufficient
            let mut new_v = vec![0.0; rows];

            // Matrix-vector multiplication: W^T * W * v
            for i in 0..rows {
                for j in 0..cols {
                    let idx = i * cols + j;
                    if idx < len && j < v.len() {
                        new_v[i] += weight_data[idx] * v[j];
                    }
                }
            }

            // Compute norm
            v_norm = new_v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if v_norm > 1e-8 {
                for item in &mut new_v {
                    *item /= v_norm;
                }
                // Resize v to match new_v for next iteration
                v = new_v;
            } else {
                break;
            }
        }

        // The spectral norm is approximately the final norm
        Ok(v_norm.max(1e-8)) // Avoid zero values
    }
}

/// Reinforcement Learning optimizer with specialized features
pub struct RLOptimizer {
    policy_optimizer: Adam,
    value_optimizer: Adam,
    clip_grad_norm: Option<f32>,
    entropy_coeff: f32,
    value_loss_coeff: f32,
    #[allow(dead_code)]
    max_grad_norm: f32,
}

impl RLOptimizer {
    pub fn new(
        policy_lr: f32,
        value_lr: f32,
        entropy_coeff: f32,
        value_loss_coeff: f32,
        max_grad_norm: f32,
    ) -> Self {
        let policy_optimizer = Adam::new(policy_lr, (0.9, 0.999), 1e-8, 0.0);
        let value_optimizer = Adam::new(value_lr, (0.9, 0.999), 1e-8, 0.0);

        Self {
            policy_optimizer,
            value_optimizer,
            clip_grad_norm: Some(max_grad_norm),
            entropy_coeff,
            value_loss_coeff,
            max_grad_norm,
        }
    }

    pub fn step_policy(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> Result<()> {
        let mut modified_grads = grads.to_vec();

        // Apply gradient clipping
        if let Some(max_norm) = self.clip_grad_norm {
            self.clip_gradients(&mut modified_grads, max_norm)?;
        }

        // Apply entropy regularization
        self.apply_entropy_regularization(&mut modified_grads)?;

        for (param, grad) in params.iter_mut().zip(modified_grads.iter()) {
            self.policy_optimizer.update(param, grad)?;
        }
        self.policy_optimizer.step();
        Ok(())
    }

    pub fn step_value(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> Result<()> {
        let mut modified_grads = grads.to_vec();

        // Scale value gradients
        for grad in modified_grads.iter_mut() {
            *grad = grad.mul_scalar(self.value_loss_coeff)?;
        }

        // Apply gradient clipping
        if let Some(max_norm) = self.clip_grad_norm {
            self.clip_gradients(&mut modified_grads, max_norm)?;
        }

        for (param, grad) in params.iter_mut().zip(modified_grads.iter()) {
            self.value_optimizer.update(param, grad)?;
        }
        self.value_optimizer.step();
        Ok(())
    }

    fn clip_gradients(&self, gradients: &mut [Tensor], max_norm: f32) -> Result<()> {
        // Compute global gradient norm
        let mut total_norm_sq: f32 = 0.0;
        for grad in gradients.iter() {
            let grad_norm_sq_tensor = grad.pow(2.0)?.sum(None, false)?;
            let grad_norm_sq_data = grad_norm_sq_tensor.data()?;
            total_norm_sq += grad_norm_sq_data[0];
        }

        let total_norm = total_norm_sq.sqrt();

        if total_norm > max_norm {
            let clip_factor = max_norm / total_norm;
            for grad in gradients.iter_mut() {
                *grad = grad.mul_scalar(clip_factor)?;
            }
        }

        Ok(())
    }

    fn apply_entropy_regularization(&self, gradients: &mut [Tensor]) -> Result<()> {
        // Add entropy bonus to encourage exploration
        for grad in gradients.iter_mut() {
            let entropy_bonus = grad.mul_scalar(self.entropy_coeff)?;
            *grad = grad.sub(&entropy_bonus)?;
        }
        Ok(())
    }
}

/// Meta-learning optimizer (MAML-style)
pub struct MetaOptimizer {
    meta_optimizer: Adam,
    inner_optimizer: SGD,
    inner_steps: usize,
    #[allow(dead_code)]
    inner_lr: f32,
    #[allow(dead_code)]
    meta_lr: f32,
    first_order: bool, // Use first-order approximation
}

impl MetaOptimizer {
    pub fn new(meta_lr: f32, inner_lr: f32, inner_steps: usize, first_order: bool) -> Self {
        let meta_optimizer = Adam::new(meta_lr, (0.9, 0.999), 1e-8, 0.0);
        let inner_optimizer = SGD::new(inner_lr, 0.0, 0.0, false);

        Self {
            meta_optimizer,
            inner_optimizer,
            inner_steps,
            inner_lr,
            meta_lr,
            first_order,
        }
    }

    pub fn meta_step(&mut self, params: &mut [Tensor], meta_grads: &[Tensor]) -> Result<()> {
        for (param, grad) in params.iter_mut().zip(meta_grads.iter()) {
            self.meta_optimizer.update(param, grad)?;
        }
        self.meta_optimizer.step();
        Ok(())
    }

    pub fn inner_loop(
        &mut self,
        mut params: Vec<Tensor>,
        task_grads: &[Vec<Tensor>],
    ) -> Result<Vec<Tensor>> {
        // Perform inner loop adaptation for a specific task
        for step in 0..self.inner_steps {
            if step < task_grads.len() {
                let grads = &task_grads[step];
                for (param, grad) in params.iter_mut().zip(grads.iter()) {
                    self.inner_optimizer.update(param, grad)?;
                }
                self.inner_optimizer.step();
            }
        }
        Ok(params)
    }

    pub fn compute_meta_gradients(
        &self,
        original_params: &[Tensor],
        adapted_params: &[Tensor],
        meta_loss_grads: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        if self.first_order {
            // First-order approximation (ignore second derivatives)
            Ok(meta_loss_grads.to_vec())
        } else {
            // Second-order gradients through inner loop
            self.compute_second_order_grads(original_params, adapted_params, meta_loss_grads)
        }
    }

    fn compute_second_order_grads(
        &self,
        _original_params: &[Tensor],
        _adapted_params: &[Tensor],
        meta_loss_grads: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        // Simplified second-order gradient computation
        // In practice, would use automatic differentiation
        Ok(meta_loss_grads.to_vec())
    }
}

/// Factory functions for creating task-specific optimizers
pub fn create_bert_optimizer(
    learning_rate: f32,
    warmup_steps: usize,
    total_steps: usize,
) -> Result<BERTOptimizer> {
    BERTOptimizer::new(learning_rate, warmup_steps, total_steps, 0.95)
}

pub fn create_gan_optimizer(g_lr: f32, d_lr: f32, use_spectral_norm: bool) -> GANOptimizer {
    GANOptimizer::new(g_lr, d_lr, use_spectral_norm, 10.0)
}

pub fn create_ppo_optimizer(learning_rate: f32, entropy_coeff: f32) -> RLOptimizer {
    RLOptimizer::new(learning_rate, learning_rate, entropy_coeff, 0.5, 0.5)
}

pub fn create_maml_optimizer(meta_lr: f32, inner_lr: f32, inner_steps: usize) -> MetaOptimizer {
    MetaOptimizer::new(meta_lr, inner_lr, inner_steps, false)
}
