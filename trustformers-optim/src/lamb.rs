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
    /// Lower and upper bound of the trust ratio (`φ` in You et al.).
    trust_clip: (f32, f32),
    /// Parameter names excluded from layer adaptation and weight decay.
    ///
    /// Reference implementations exclude biases and LayerNorm parameters: their weight
    /// norm is not comparable to a weight matrix's, so the trust ratio is meaningless
    /// for them.
    excluded: std::collections::HashSet<String>,
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
            // You et al. define `φ(z) = min(max(z, γ_l), γ_u)`; reference
            // implementations (NVIDIA, TF) clamp the ratio to a bounded interval so a
            // large-norm layer paired with a tiny update cannot produce an unbounded
            // step.
            trust_clip: (0.0, 10.0),
            excluded: std::collections::HashSet::new(),
            state: OptimizerState::new(),
            exp_avg: HashMap::new(),
            exp_avg_sq: HashMap::new(),
        }
    }

    /// Sets the bounds `(γ_l, γ_u)` of the trust-ratio clipping function `φ`.
    ///
    /// # Errors
    ///
    /// Returns an error when the interval is empty or negative.
    pub fn with_trust_clip(mut self, lower: f32, upper: f32) -> Result<Self> {
        if !(lower >= 0.0 && upper > lower) {
            return Err(TrustformersError::invalid_config(format!(
                "LAMB trust-ratio bounds must satisfy 0 <= lower < upper, got ({lower}, {upper})"
            )));
        }
        self.trust_clip = (lower, upper);
        Ok(self)
    }

    /// Excludes a named parameter from layer adaptation and weight decay.
    ///
    /// Conventionally applied to biases and LayerNorm gains/biases.
    pub fn exclude_from_adaptation(&mut self, name: impl Into<String>) {
        self.excluded.insert(name.into());
    }

    /// Whether `name` was excluded via [`LAMB::exclude_from_adaptation`].
    pub fn is_excluded(&self, name: &str) -> bool {
        self.excluded.contains(name)
    }

    /// Updates one parameter identified by a stable caller-supplied name.
    ///
    /// Names are what makes the bias/LayerNorm exclusion possible, so this is the
    /// preferred entry point. See [`crate::param_id`].
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported tensor dtypes or a state size mismatch.
    pub fn update_named(
        &mut self,
        name: &str,
        parameter: &mut Tensor,
        grad: &Tensor,
    ) -> Result<()> {
        let key = self.state.params.key_for_named_tensor(name, parameter)?;
        let adapt = !self.excluded.contains(name);
        self.update_with_key(key, adapt, parameter, grad)
    }
}

impl LAMB {
    /// Shared update body given an already-resolved stable key.
    fn update_with_key(
        &mut self,
        param_id: String,
        adapt: bool,
        parameter: &mut Tensor,
        grad: &Tensor,
    ) -> Result<()> {
        // LAMB optimizer with layer-wise adaptation
        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_arr)) => {
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

                    // Apply weight decay to the update (L2 regularization). Excluded
                    // parameters (biases, LayerNorm) take neither decay nor adaptation.
                    let decay_term = if self.weight_decay != 0.0 && adapt {
                        self.weight_decay * *p
                    } else {
                        0.0
                    };

                    // Compute the raw update step (before layer-wise adaptation)
                    let raw_update = m_hat / (v_hat.sqrt() + self.eps) + decay_term;
                    raw_updates.push(raw_update);
                }

                // LAMB layer-wise adaptation: compute norms for adaptation
                let weight_norm: f32 = param.iter().map(|&p| p * p).sum::<f32>().sqrt();
                let update_norm: f32 = raw_updates.iter().map(|&u| u * u).sum::<f32>().sqrt();

                // Layer-wise adaptation rate `φ(‖w‖) / ‖r + λw‖`, with `φ` the
                // clipping function from You et al. An unbounded ratio makes a
                // large-norm layer with a tiny update take a divergent step.
                let trust_ratio = if update_norm > 0.0 && weight_norm > 0.0 && adapt {
                    (weight_norm / update_norm).clamp(self.trust_clip.0, self.trust_clip.1)
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
}

impl Optimizer for LAMB {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        let key = self.state.params.key_for_tensor(parameter)?;
        self.update_with_key(key, true, parameter, grad)
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
