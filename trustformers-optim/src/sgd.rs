use crate::common::{OptimizerState, StateMemoryStats};
use crate::traits::StatefulOptimizer;
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Configuration for SGD optimizer.
#[derive(Debug, Clone)]
pub struct SGDConfig {
    /// Learning rate
    pub lr: f32,
    /// Momentum factor
    pub momentum: f32,
    /// Weight decay coefficient
    pub weight_decay: f32,
    /// Dampening for momentum
    pub dampening: f32,
    /// Enable Nesterov momentum
    pub nesterov: bool,
}

impl Default for SGDConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SGD {
    /// Configuration for this optimizer
    config: SGDConfig,
    /// Optimizer state tracking steps
    state: OptimizerState,
}

impl SGD {
    pub fn new(lr: f32, momentum: f32, weight_decay: f32, nesterov: bool) -> Self {
        Self {
            config: SGDConfig {
                lr,
                momentum,
                weight_decay,
                dampening: 0.0,
                nesterov,
            },
            state: OptimizerState::new(),
        }
    }

    /// Creates a new SGD optimizer from configuration.
    pub fn from_config(config: SGDConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
        }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_arr)) => {
                if self.config.weight_decay != 0.0 {
                    let decay = &*param * self.config.weight_decay;
                    *param = &*param - &decay;
                }

                let mut d_p = grad_arr.clone();

                if self.config.momentum != 0.0 {
                    let param_id = format!("{:p}", param.as_ptr());
                    let buf = self
                        .state
                        .momentum
                        .entry(param_id)
                        .or_insert_with(|| vec![0.0; grad_arr.len()]);

                    if buf.len() != grad_arr.len() {
                        return Err(TrustformersError::tensor_op_error(
                            "Momentum buffer size mismatch",
                            "sgd_update",
                        ));
                    }

                    let d_p_vec = d_p.as_slice_mut().expect("cloned array must be contiguous");
                    for (i, (b, g)) in buf.iter_mut().zip(grad_arr.iter()).enumerate() {
                        *b = *b * self.config.momentum + (1.0 - self.config.dampening) * g;
                        if self.config.nesterov {
                            d_p_vec[i] = g + self.config.momentum * *b;
                        } else {
                            d_p_vec[i] = *b;
                        }
                    }
                }

                *param = &*param - &(d_p * self.config.lr);
                Ok(())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for SGD",
                "sgd_update",
            )),
        }
    }

    fn zero_grad(&mut self) {}

    fn step(&mut self) {
        self.state.step += 1;
    }

    fn get_lr(&self) -> f32 {
        self.config.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }
}

impl StatefulOptimizer for SGD {
    type Config = SGDConfig;
    type State = OptimizerState;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state_dict = HashMap::new();

        // Save configuration
        state_dict.insert("lr".to_string(), Tensor::new(vec![self.config.lr])?);
        state_dict.insert(
            "momentum".to_string(),
            Tensor::new(vec![self.config.momentum])?,
        );
        state_dict.insert(
            "weight_decay".to_string(),
            Tensor::new(vec![self.config.weight_decay])?,
        );
        state_dict.insert(
            "dampening".to_string(),
            Tensor::new(vec![self.config.dampening])?,
        );
        state_dict.insert(
            "nesterov".to_string(),
            Tensor::new(vec![if self.config.nesterov { 1.0 } else { 0.0 }])?,
        );
        state_dict.insert(
            "step".to_string(),
            Tensor::new(vec![self.state.step as f32])?,
        );

        // Save momentum buffers
        for (param_id, momentum) in &self.state.momentum {
            state_dict.insert(
                format!("momentum_{}", param_id),
                Tensor::new(momentum.clone())?,
            );
        }

        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        // Load configuration
        if let Some(lr_tensor) = state.get("lr") {
            if let Ok(lr_vec) = lr_tensor.data() {
                if !lr_vec.is_empty() {
                    self.config.lr = lr_vec[0];
                }
            }
        }
        if let Some(momentum_tensor) = state.get("momentum") {
            if let Ok(momentum_vec) = momentum_tensor.data() {
                if !momentum_vec.is_empty() {
                    self.config.momentum = momentum_vec[0];
                }
            }
        }
        if let Some(weight_decay_tensor) = state.get("weight_decay") {
            if let Ok(weight_decay_vec) = weight_decay_tensor.data() {
                if !weight_decay_vec.is_empty() {
                    self.config.weight_decay = weight_decay_vec[0];
                }
            }
        }
        if let Some(dampening_tensor) = state.get("dampening") {
            if let Ok(dampening_vec) = dampening_tensor.data() {
                if !dampening_vec.is_empty() {
                    self.config.dampening = dampening_vec[0];
                }
            }
        }
        if let Some(nesterov_tensor) = state.get("nesterov") {
            if let Ok(nesterov_vec) = nesterov_tensor.data() {
                if !nesterov_vec.is_empty() {
                    self.config.nesterov = nesterov_vec[0] != 0.0;
                }
            }
        }
        if let Some(step_tensor) = state.get("step") {
            if let Ok(step_vec) = step_tensor.data() {
                if !step_vec.is_empty() {
                    self.state.step = step_vec[0] as usize;
                }
            }
        }

        // Load momentum buffers
        for (key, tensor) in state.iter() {
            if key.starts_with("momentum_") {
                let param_id = key.trim_start_matches("momentum_");
                if let Ok(momentum) = tensor.data() {
                    self.state.momentum.insert(param_id.to_string(), momentum.clone());
                }
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let mut momentum_elements = 0;

        for momentum in self.state.momentum.values() {
            momentum_elements += momentum.len();
        }

        let total_bytes = momentum_elements * std::mem::size_of::<f32>();

        StateMemoryStats {
            momentum_elements,
            variance_elements: 0,
            third_moment_elements: 0,
            total_bytes,
            num_parameters: momentum_elements,
        }
    }

    fn reset_state(&mut self) {
        self.state.step = 0;
        self.state.momentum.clear();
    }

    fn num_parameters(&self) -> usize {
        self.state.momentum.values().map(|v| v.len()).sum()
    }
}
