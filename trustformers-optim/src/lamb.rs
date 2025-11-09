use crate::common::OptimizerState;
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// LAMB (Layer-wise Adaptive Moments optimizer for Batch training) optimizer
///
/// LAMB is an optimization algorithm that uses layer-wise adaptation to scale
/// the learning rate based on the ratio of weight norm to update norm for each layer.
/// This helps with training stability and convergence for large batch sizes.
#[derive(Debug)]
pub struct LAMB {
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    state: OptimizerState,
    exp_avg: HashMap<String, Vec<f32>>,
    exp_avg_sq: HashMap<String, Vec<f32>>,
}

impl LAMB {
    pub fn new(lr: f32, betas: (f32, f32), eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            betas,
            eps,
            weight_decay,
            state: OptimizerState::new(),
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }
}

impl Optimizer for LAMB {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        // LAMB optimizer with layer-wise adaptation
        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_arr)) => {
                let param_id = format!("{:p}", param.as_ptr());
                let size = grad_arr.len();

                let exp_avg =
                    self.exp_avg.entry(param_id.clone()).or_insert_with(|| vec![0.0; size]);
                let exp_avg_sq = self.exp_avg_sq.entry(param_id).or_insert_with(|| vec![0.0; size]);

                if exp_avg.len() != size || exp_avg_sq.len() != size {
                    return Err(TrustformersError::tensor_op_error(
                        "LAMB state buffer size mismatch",
                        "buffer size validation",
                    ));
                }

                let step = (self.state.step + 1) as f32;
                let bias_correction1 = 1.0 - self.betas.0.powf(step);
                let bias_correction2 = 1.0 - self.betas.1.powf(step);

                // First, update the moment estimates and compute the raw update
                let mut raw_updates = Vec::with_capacity(size);
                for ((p, g), (m, v)) in param
                    .iter()
                    .zip(grad_arr.iter())
                    .zip(exp_avg.iter_mut().zip(exp_avg_sq.iter_mut()))
                {
                    // Update biased first moment estimate
                    *m = self.betas.0 * *m + (1.0 - self.betas.0) * g;
                    // Update biased second raw moment estimate
                    *v = self.betas.1 * *v + (1.0 - self.betas.1) * g * g;

                    // Compute bias-corrected first moment estimate
                    let m_hat = *m / bias_correction1;
                    // Compute bias-corrected second raw moment estimate
                    let v_hat = *v / bias_correction2;

                    // Apply weight decay to the update (L2 regularization)
                    let decay_term =
                        if self.weight_decay != 0.0 { self.weight_decay * *p } else { 0.0 };

                    // Compute the raw update step (before layer-wise adaptation)
                    let raw_update = m_hat / (v_hat.sqrt() + self.eps) + decay_term;
                    raw_updates.push(raw_update);
                }

                // LAMB layer-wise adaptation: compute norms for adaptation
                let weight_norm: f32 = param.iter().map(|&p| p * p).sum::<f32>().sqrt();
                let update_norm: f32 = raw_updates.iter().map(|&u| u * u).sum::<f32>().sqrt();

                // Compute the layer-wise adaptation rate
                let trust_ratio = if update_norm > 0.0 && weight_norm > 0.0 {
                    weight_norm / update_norm
                } else {
                    1.0
                };

                // Apply the adapted learning rate with layer-wise scaling
                let adapted_lr = self.lr * trust_ratio;

                // Apply the final update with layer-wise adaptation
                for (p, &raw_update) in param.iter_mut().zip(raw_updates.iter()) {
                    *p -= adapted_lr * raw_update;
                }

                Ok(())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for LAMB",
                "tensor type validation",
            )),
        }
    }

    fn zero_grad(&mut self) {}

    fn step(&mut self) {
        self.state.step += 1;
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}
