//! TensorFlow Optimizer API Compatibility Layer
//!
//! This module provides TensorFlow-compatible optimizer interfaces for seamless
//! integration with TensorFlow-based training workflows. It wraps our native
//! optimizers to provide the familiar TensorFlow API while maintaining high performance.

use crate::{Adam, AdamW};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::Optimizer;
use trustformers_core::Tensor;

/// TensorFlow-compatible optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorFlowOptimizerConfig {
    pub optimizer_type: String,
    pub learning_rate: f64,
    pub beta_1: Option<f64>,
    pub beta_2: Option<f64>,
    pub epsilon: Option<f64>,
    pub weight_decay: Option<f64>,
    pub clipnorm: Option<f64>,
    pub clipvalue: Option<f64>,
    pub global_clipnorm: Option<f64>,
    pub use_ema: Option<bool>,
    pub ema_momentum: Option<f64>,
    pub ema_overwrite_frequency: Option<i32>,
    pub jit_compile: Option<bool>,
    pub name: Option<String>,
    pub parameters: HashMap<String, serde_json::Value>,
}

impl Default for TensorFlowOptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: "Adam".to_string(),
            learning_rate: 0.001,
            beta_1: Some(0.9),
            beta_2: Some(0.999),
            epsilon: Some(1e-7),
            weight_decay: None,
            clipnorm: None,
            clipvalue: None,
            global_clipnorm: None,
            use_ema: Some(false),
            ema_momentum: Some(0.99),
            ema_overwrite_frequency: None,
            jit_compile: Some(true),
            name: None,
            parameters: HashMap::new(),
        }
    }
}

/// TensorFlow-compatible learning rate schedule
pub trait TensorFlowLearningRateSchedule: Send + Sync {
    /// Get learning rate at current step
    fn get_lr(&self, step: i64) -> f64;

    /// Get configuration
    fn get_config(&self) -> serde_json::Value;
}

/// TensorFlow-compatible exponential decay schedule
#[derive(Debug, Clone)]
pub struct TensorFlowExponentialDecay {
    initial_learning_rate: f64,
    decay_steps: i64,
    decay_rate: f64,
    staircase: bool,
}

impl TensorFlowExponentialDecay {
    pub fn new(
        initial_learning_rate: f64,
        decay_steps: i64,
        decay_rate: f64,
        staircase: bool,
    ) -> Self {
        Self {
            initial_learning_rate,
            decay_steps,
            decay_rate,
            staircase,
        }
    }
}

impl TensorFlowLearningRateSchedule for TensorFlowExponentialDecay {
    fn get_lr(&self, step: i64) -> f64 {
        let decay_factor = if self.staircase {
            (step / self.decay_steps) as f64
        } else {
            step as f64 / self.decay_steps as f64
        };

        self.initial_learning_rate * self.decay_rate.powf(decay_factor)
    }

    fn get_config(&self) -> serde_json::Value {
        serde_json::json!({
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
        })
    }
}

/// TensorFlow-compatible cosine decay schedule
#[derive(Debug, Clone)]
pub struct TensorFlowCosineDecay {
    initial_learning_rate: f64,
    decay_steps: i64,
    alpha: f64,
}

impl TensorFlowCosineDecay {
    pub fn new(initial_learning_rate: f64, decay_steps: i64, alpha: f64) -> Self {
        Self {
            initial_learning_rate,
            decay_steps,
            alpha,
        }
    }
}

impl TensorFlowLearningRateSchedule for TensorFlowCosineDecay {
    fn get_lr(&self, step: i64) -> f64 {
        let completed_fraction = (step.min(self.decay_steps) as f64) / (self.decay_steps as f64);
        let cosine_decayed = 0.5 * (1.0 + (std::f64::consts::PI * completed_fraction).cos());
        let decayed = (1.0 - self.alpha) * cosine_decayed + self.alpha;

        self.initial_learning_rate * decayed
    }

    fn get_config(&self) -> serde_json::Value {
        serde_json::json!({
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
        })
    }
}

/// TensorFlow-compatible optimizer interface
pub trait TensorFlowOptimizer: Send + Sync {
    /// Apply gradients to variables
    fn apply_gradients(
        &mut self,
        grads_and_vars: &[(Tensor, String)],
        global_step: Option<i64>,
    ) -> Result<()>;

    /// Minimize loss function
    fn minimize(
        &mut self,
        loss_fn: Box<dyn Fn() -> Result<Tensor>>,
        var_list: &[String],
        global_step: Option<i64>,
    ) -> Result<Tensor>;

    /// Get optimizer configuration
    fn get_config(&self) -> TensorFlowOptimizerConfig;

    /// Get optimizer variables (state)
    fn variables(&self) -> Vec<String>;

    /// Get optimizer weights
    fn get_weights(&self) -> Vec<Tensor>;

    /// Set optimizer weights
    fn set_weights(&mut self, weights: Vec<Tensor>) -> Result<()>;

    /// Get learning rate
    fn get_learning_rate(&self) -> f64;

    /// Set learning rate
    fn set_learning_rate(&mut self, lr: f64) -> Result<()>;

    /// Get optimizer name
    fn get_name(&self) -> &str;
}

/// TensorFlow-compatible Adam optimizer
pub struct TensorFlowAdam {
    inner: Adam,
    config: TensorFlowOptimizerConfig,
    variables: Arc<Mutex<HashMap<String, Tensor>>>,
    lr_schedule: Option<Box<dyn TensorFlowLearningRateSchedule>>,
    global_step: i64,
}

impl TensorFlowAdam {
    /// Create new TensorFlow-compatible Adam optimizer
    pub fn new(
        learning_rate: f64,
        beta_1: f64,
        beta_2: f64,
        epsilon: f64,
        weight_decay: Option<f64>,
        clipnorm: Option<f64>,
        clipvalue: Option<f64>,
        global_clipnorm: Option<f64>,
        use_ema: bool,
        ema_momentum: f64,
        jit_compile: bool,
        name: Option<String>,
    ) -> Result<Self> {
        let config = TensorFlowOptimizerConfig {
            optimizer_type: "Adam".to_string(),
            learning_rate,
            beta_1: Some(beta_1),
            beta_2: Some(beta_2),
            epsilon: Some(epsilon),
            weight_decay,
            clipnorm,
            clipvalue,
            global_clipnorm,
            use_ema: Some(use_ema),
            ema_momentum: Some(ema_momentum),
            ema_overwrite_frequency: None,
            jit_compile: Some(jit_compile),
            name,
            parameters: HashMap::new(),
        };

        // optimizer_config is redundant - using config above

        let inner = Adam::new(
            learning_rate as f32,
            (beta_1 as f32, beta_2 as f32),
            epsilon as f32,
            weight_decay.unwrap_or(0.0) as f32,
        );

        Ok(Self {
            inner,
            config,
            variables: Arc::new(Mutex::new(HashMap::new())),
            lr_schedule: None,
            global_step: 0,
        })
    }

    /// Create with default parameters
    pub fn with_defaults() -> Result<Self> {
        Self::new(
            0.001,
            0.9,
            0.999,
            1e-7,
            None,
            None,
            None,
            None,
            false,
            0.99,
            true,
            Some("Adam".to_string()),
        )
    }

    /// Create TensorFlow Adam optimizer from configuration
    pub fn from_config(config: TensorFlowOptimizerConfig) -> Result<Self> {
        Self::new(
            config.learning_rate,
            config.beta_1.unwrap_or(0.9),
            config.beta_2.unwrap_or(0.999),
            config.epsilon.unwrap_or(1e-7),
            config.weight_decay,
            config.clipnorm,
            config.clipvalue,
            config.global_clipnorm,
            config.use_ema.unwrap_or(false),
            config.ema_momentum.unwrap_or(0.99),
            config.jit_compile.unwrap_or(true),
            config.name,
        )
    }

    /// Create with learning rate schedule
    pub fn with_schedule(
        schedule: Box<dyn TensorFlowLearningRateSchedule>,
        beta_1: f64,
        beta_2: f64,
        epsilon: f64,
        weight_decay: Option<f64>,
        clipnorm: Option<f64>,
        clipvalue: Option<f64>,
        global_clipnorm: Option<f64>,
        use_ema: bool,
        ema_momentum: f64,
        jit_compile: bool,
        name: Option<String>,
    ) -> Result<Self> {
        let mut optimizer = Self::new(
            schedule.get_lr(0),
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            clipnorm,
            clipvalue,
            global_clipnorm,
            use_ema,
            ema_momentum,
            jit_compile,
            name,
        )?;

        optimizer.lr_schedule = Some(schedule);
        Ok(optimizer)
    }

    /// Add variable to optimizer
    pub fn add_variable(&mut self, name: String, var: Tensor) -> Result<()> {
        let mut variables = self.variables.lock().unwrap();
        variables.insert(name, var);
        Ok(())
    }

    /// Update learning rate based on schedule
    fn update_learning_rate(&mut self) -> Result<()> {
        if let Some(ref schedule) = self.lr_schedule {
            let new_lr = schedule.get_lr(self.global_step);
            self.config.learning_rate = new_lr;

            // Update inner optimizer learning rate
            self.inner.set_lr(new_lr as f32);
        }
        Ok(())
    }

    /// Apply gradient clipping
    fn clip_gradients(&self, gradients: &mut [Tensor]) -> Result<()> {
        if let Some(clipnorm) = self.config.clipnorm {
            // Clip by norm
            for grad in gradients.iter_mut() {
                let norm = grad.norm()?;
                if norm > clipnorm as f32 {
                    grad.mul_scalar((clipnorm as f32) / norm)?;
                }
            }
        }

        if let Some(clipvalue) = self.config.clipvalue {
            // Clip by value
            for grad in gradients.iter_mut() {
                grad.clamp(-clipvalue as f32, clipvalue as f32)?;
            }
        }

        if let Some(global_clipnorm) = self.config.global_clipnorm {
            // Global gradient clipping
            let global_norm: f64 = gradients
                .iter()
                .map(|g| g.norm().unwrap_or(0.0).powi(2) as f64)
                .sum::<f64>()
                .sqrt();

            if global_norm > global_clipnorm {
                let scale = global_clipnorm / global_norm;
                for grad in gradients.iter_mut() {
                    grad.mul_scalar(scale as f32)?;
                }
            }
        }

        Ok(())
    }
}

impl TensorFlowOptimizer for TensorFlowAdam {
    fn apply_gradients(
        &mut self,
        grads_and_vars: &[(Tensor, String)],
        global_step: Option<i64>,
    ) -> Result<()> {
        if let Some(step) = global_step {
            self.global_step = step;
        } else {
            self.global_step += 1;
        }

        // Update learning rate if schedule is set
        self.update_learning_rate()?;

        let mut gradients: Vec<Tensor> = grads_and_vars.iter().map(|(g, _)| g.clone()).collect();

        // Apply gradient clipping
        self.clip_gradients(&mut gradients)?;

        // Apply gradients using inner optimizer
        let mut variables = self.variables.lock().unwrap();
        for (grad, var_name) in grads_and_vars {
            if let Some(var) = variables.get_mut(var_name) {
                self.inner.update(var, grad)?;
            }
        }
        self.inner.step();

        Ok(())
    }

    fn minimize(
        &mut self,
        loss_fn: Box<dyn Fn() -> Result<Tensor>>,
        var_list: &[String],
        global_step: Option<i64>,
    ) -> Result<Tensor> {
        let loss = loss_fn()?;

        // Compute gradients (this would normally be done by automatic differentiation)
        let mut grads_and_vars = Vec::new();
        {
            let mut variables = self.variables.lock().unwrap();

            for var_name in var_list {
                if let Some(var) = variables.get_mut(var_name) {
                    // Compute numerical gradient using finite differences
                    let grad = self.compute_numerical_gradient(loss_fn.as_ref(), var, var_name)?;
                    grads_and_vars.push((grad, var_name.clone()));
                }
            }
        } // variables lock is dropped here

        self.apply_gradients(&grads_and_vars, global_step)?;
        Ok(loss)
    }

    fn get_config(&self) -> TensorFlowOptimizerConfig {
        self.config.clone()
    }

    fn variables(&self) -> Vec<String> {
        let variables = self.variables.lock().unwrap();
        variables.keys().cloned().collect()
    }

    fn get_weights(&self) -> Vec<Tensor> {
        let variables = self.variables.lock().unwrap();
        variables.values().cloned().collect()
    }

    fn set_weights(&mut self, weights: Vec<Tensor>) -> Result<()> {
        let mut variables = self.variables.lock().unwrap();
        let var_names: Vec<String> = variables.keys().cloned().collect();

        if weights.len() != var_names.len() {
            return Err(TrustformersError::invalid_argument(
                "Number of weights must match number of variables".to_string(),
            ));
        }

        for (weight, var_name) in weights.into_iter().zip(var_names) {
            variables.insert(var_name, weight);
        }

        Ok(())
    }

    fn get_learning_rate(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) -> Result<()> {
        self.config.learning_rate = lr;

        // Update inner optimizer
        self.inner.set_lr(lr as f32);

        Ok(())
    }

    fn get_name(&self) -> &str {
        self.config.name.as_deref().unwrap_or("Adam")
    }
}

impl TensorFlowAdam {
    /// Compute numerical gradient using finite differences
    fn compute_numerical_gradient(
        &self,
        loss_fn: &dyn Fn() -> Result<Tensor>,
        var: &mut Tensor,
        _var_name: &str,
    ) -> Result<Tensor> {
        const EPSILON: f32 = 1e-4;

        let original_loss = loss_fn()?;
        #[allow(unused_assignments)]
        let mut grad = Tensor::zeros(&var.shape())?;

        // Compute gradient for each element using finite differences
        let var_data = var.data()?;
        let mut grad_data = vec![0.0; var_data.len()];

        for i in 0..var_data.len() {
            // Forward difference: f(x + h) - f(x) / h
            let mut var_plus = var_data.clone();
            var_plus[i] += EPSILON;
            *var = Tensor::from_vec(var_plus, &var.shape())?;

            let loss_plus = loss_fn()?;
            let loss_plus_scalar = loss_plus.data()?[0];
            let original_loss_scalar = original_loss.data()?[0];

            grad_data[i] = (loss_plus_scalar - original_loss_scalar) / EPSILON;

            // Restore original value
            let var_original = var_data.clone();
            *var = Tensor::from_vec(var_original, &var.shape())?;
        }

        grad = Tensor::from_vec(grad_data, &var.shape())?;
        Ok(grad)
    }
}

/// TensorFlow-compatible AdamW optimizer
pub struct TensorFlowAdamW {
    inner: AdamW,
    config: TensorFlowOptimizerConfig,
    variables: Arc<Mutex<HashMap<String, Tensor>>>,
    lr_schedule: Option<Box<dyn TensorFlowLearningRateSchedule>>,
    global_step: i64,
}

impl TensorFlowAdamW {
    /// Create new TensorFlow-compatible AdamW optimizer
    pub fn new(
        learning_rate: f64,
        beta_1: f64,
        beta_2: f64,
        epsilon: f64,
        weight_decay: f64,
        clipnorm: Option<f64>,
        clipvalue: Option<f64>,
        global_clipnorm: Option<f64>,
        use_ema: bool,
        ema_momentum: f64,
        jit_compile: bool,
        name: Option<String>,
    ) -> Result<Self> {
        let config = TensorFlowOptimizerConfig {
            optimizer_type: "AdamW".to_string(),
            learning_rate,
            beta_1: Some(beta_1),
            beta_2: Some(beta_2),
            epsilon: Some(epsilon),
            weight_decay: Some(weight_decay),
            clipnorm,
            clipvalue,
            global_clipnorm,
            use_ema: Some(use_ema),
            ema_momentum: Some(ema_momentum),
            ema_overwrite_frequency: None,
            jit_compile: Some(jit_compile),
            name,
            parameters: HashMap::new(),
        };

        let _optimizer_config = TensorFlowOptimizerConfig {
            learning_rate,
            beta_1: Some(beta_1),
            beta_2: Some(beta_2),
            epsilon: Some(epsilon),
            weight_decay: Some(weight_decay),
            ..Default::default()
        };

        let inner = AdamW::new(
            learning_rate as f32,
            (beta_1 as f32, beta_2 as f32),
            epsilon as f32,
            weight_decay as f32,
        );

        Ok(Self {
            inner,
            config,
            variables: Arc::new(Mutex::new(HashMap::new())),
            lr_schedule: None,
            global_step: 0,
        })
    }

    /// Create with default parameters
    pub fn with_defaults() -> Result<Self> {
        Self::new(
            0.001,
            0.9,
            0.999,
            1e-7,
            0.01,
            None,
            None,
            None,
            false,
            0.99,
            true,
            Some("AdamW".to_string()),
        )
    }

    /// Create with learning rate schedule
    pub fn with_schedule(
        schedule: Box<dyn TensorFlowLearningRateSchedule>,
        beta_1: f64,
        beta_2: f64,
        epsilon: f64,
        weight_decay: f64,
        clipnorm: Option<f64>,
        clipvalue: Option<f64>,
        global_clipnorm: Option<f64>,
        use_ema: bool,
        ema_momentum: f64,
        jit_compile: bool,
        name: Option<String>,
    ) -> Result<Self> {
        let mut optimizer = Self::new(
            schedule.get_lr(0),
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            clipnorm,
            clipvalue,
            global_clipnorm,
            use_ema,
            ema_momentum,
            jit_compile,
            name,
        )?;

        optimizer.lr_schedule = Some(schedule);
        Ok(optimizer)
    }

    /// Add variable to optimizer
    pub fn add_variable(&mut self, name: String, var: Tensor) -> Result<()> {
        let mut variables = self.variables.lock().unwrap();
        variables.insert(name, var);
        Ok(())
    }

    /// Update learning rate based on schedule
    fn update_learning_rate(&mut self) -> Result<()> {
        if let Some(ref schedule) = self.lr_schedule {
            let new_lr = schedule.get_lr(self.global_step);
            self.config.learning_rate = new_lr;

            // Update inner optimizer learning rate
            self.inner.set_lr(new_lr as f32);
        }
        Ok(())
    }

    /// Apply gradient clipping
    fn clip_gradients(&self, gradients: &mut [Tensor]) -> Result<()> {
        if let Some(clipnorm) = self.config.clipnorm {
            // Clip by norm
            for grad in gradients.iter_mut() {
                let norm = grad.norm()?;
                if norm > clipnorm as f32 {
                    grad.mul_scalar((clipnorm as f32) / norm)?;
                }
            }
        }

        if let Some(clipvalue) = self.config.clipvalue {
            // Clip by value
            for grad in gradients.iter_mut() {
                grad.clamp(-clipvalue as f32, clipvalue as f32)?;
            }
        }

        if let Some(global_clipnorm) = self.config.global_clipnorm {
            // Global gradient clipping
            let global_norm: f64 = gradients
                .iter()
                .map(|g| g.norm().unwrap_or(0.0).powi(2) as f64)
                .sum::<f64>()
                .sqrt();

            if global_norm > global_clipnorm {
                let scale = global_clipnorm / global_norm;
                for grad in gradients.iter_mut() {
                    grad.mul_scalar(scale as f32)?;
                }
            }
        }

        Ok(())
    }
}

impl TensorFlowOptimizer for TensorFlowAdamW {
    fn apply_gradients(
        &mut self,
        grads_and_vars: &[(Tensor, String)],
        global_step: Option<i64>,
    ) -> Result<()> {
        if let Some(step) = global_step {
            self.global_step = step;
        } else {
            self.global_step += 1;
        }

        // Update learning rate if schedule is set
        self.update_learning_rate()?;

        let mut gradients: Vec<Tensor> = grads_and_vars.iter().map(|(g, _)| g.clone()).collect();

        // Apply gradient clipping
        self.clip_gradients(&mut gradients)?;

        // Apply gradients using inner optimizer
        let mut variables = self.variables.lock().unwrap();
        for (grad, var_name) in grads_and_vars {
            if let Some(var) = variables.get_mut(var_name) {
                self.inner.update(var, grad)?;
            }
        }
        self.inner.step();

        Ok(())
    }

    fn minimize(
        &mut self,
        loss_fn: Box<dyn Fn() -> Result<Tensor>>,
        var_list: &[String],
        global_step: Option<i64>,
    ) -> Result<Tensor> {
        let loss = loss_fn()?;

        // Compute gradients (this would normally be done by automatic differentiation)
        let mut grads_and_vars = Vec::new();
        {
            let mut variables = self.variables.lock().unwrap();

            for var_name in var_list {
                if let Some(var) = variables.get_mut(var_name) {
                    // Compute numerical gradient using finite differences
                    let grad = self.compute_numerical_gradient(loss_fn.as_ref(), var, var_name)?;
                    grads_and_vars.push((grad, var_name.clone()));
                }
            }
        } // variables lock is dropped here

        self.apply_gradients(&grads_and_vars, global_step)?;
        Ok(loss)
    }

    fn get_config(&self) -> TensorFlowOptimizerConfig {
        self.config.clone()
    }

    fn variables(&self) -> Vec<String> {
        let variables = self.variables.lock().unwrap();
        variables.keys().cloned().collect()
    }

    fn get_weights(&self) -> Vec<Tensor> {
        let variables = self.variables.lock().unwrap();
        variables.values().cloned().collect()
    }

    fn set_weights(&mut self, weights: Vec<Tensor>) -> Result<()> {
        let mut variables = self.variables.lock().unwrap();
        let var_names: Vec<String> = variables.keys().cloned().collect();

        if weights.len() != var_names.len() {
            return Err(TrustformersError::invalid_argument(
                "Number of weights must match number of variables".to_string(),
            ));
        }

        for (weight, var_name) in weights.into_iter().zip(var_names) {
            variables.insert(var_name, weight);
        }

        Ok(())
    }

    fn get_learning_rate(&self) -> f64 {
        self.config.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) -> Result<()> {
        self.config.learning_rate = lr;

        // Update inner optimizer
        self.inner.set_lr(lr as f32);

        Ok(())
    }

    fn get_name(&self) -> &str {
        self.config.name.as_deref().unwrap_or("AdamW")
    }
}

impl TensorFlowAdamW {
    /// Compute numerical gradient using finite differences
    fn compute_numerical_gradient(
        &self,
        loss_fn: &dyn Fn() -> Result<Tensor>,
        var: &mut Tensor,
        _var_name: &str,
    ) -> Result<Tensor> {
        const EPSILON: f32 = 1e-4;

        let original_loss = loss_fn()?;
        #[allow(unused_assignments)]
        let mut grad = Tensor::zeros(&var.shape())?;

        // Compute gradient for each element using finite differences
        let var_data = var.data()?;
        let mut grad_data = vec![0.0; var_data.len()];

        for i in 0..var_data.len() {
            // Forward difference: f(x + h) - f(x) / h
            let mut var_plus = var_data.clone();
            var_plus[i] += EPSILON;
            *var = Tensor::from_vec(var_plus, &var.shape())?;

            let loss_plus = loss_fn()?;
            let loss_plus_scalar = loss_plus.data()?[0];
            let original_loss_scalar = original_loss.data()?[0];

            grad_data[i] = (loss_plus_scalar - original_loss_scalar) / EPSILON;

            // Restore original value
            let var_original = var_data.clone();
            *var = Tensor::from_vec(var_original, &var.shape())?;
        }

        grad = Tensor::from_vec(grad_data, &var.shape())?;
        Ok(grad)
    }
}

/// TensorFlow optimizer factory
pub struct TensorFlowOptimizerFactory;

impl TensorFlowOptimizerFactory {
    /// Create Adam optimizer
    pub fn adam(
        learning_rate: f64,
        beta_1: f64,
        beta_2: f64,
        epsilon: f64,
        weight_decay: Option<f64>,
        clipnorm: Option<f64>,
        clipvalue: Option<f64>,
        global_clipnorm: Option<f64>,
        use_ema: bool,
        ema_momentum: f64,
        jit_compile: bool,
        name: Option<String>,
    ) -> Result<TensorFlowAdam> {
        TensorFlowAdam::new(
            learning_rate,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            clipnorm,
            clipvalue,
            global_clipnorm,
            use_ema,
            ema_momentum,
            jit_compile,
            name,
        )
    }

    /// Create AdamW optimizer
    pub fn adamw(
        learning_rate: f64,
        beta_1: f64,
        beta_2: f64,
        epsilon: f64,
        weight_decay: f64,
        clipnorm: Option<f64>,
        clipvalue: Option<f64>,
        global_clipnorm: Option<f64>,
        use_ema: bool,
        ema_momentum: f64,
        jit_compile: bool,
        name: Option<String>,
    ) -> Result<TensorFlowAdamW> {
        TensorFlowAdamW::new(
            learning_rate,
            beta_1,
            beta_2,
            epsilon,
            weight_decay,
            clipnorm,
            clipvalue,
            global_clipnorm,
            use_ema,
            ema_momentum,
            jit_compile,
            name,
        )
    }

    /// Create exponential decay schedule
    pub fn exponential_decay(
        initial_learning_rate: f64,
        decay_steps: i64,
        decay_rate: f64,
        staircase: bool,
    ) -> TensorFlowExponentialDecay {
        TensorFlowExponentialDecay::new(initial_learning_rate, decay_steps, decay_rate, staircase)
    }

    /// Create cosine decay schedule
    pub fn cosine_decay(
        initial_learning_rate: f64,
        decay_steps: i64,
        alpha: f64,
    ) -> TensorFlowCosineDecay {
        TensorFlowCosineDecay::new(initial_learning_rate, decay_steps, alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::Tensor;

    #[test]
    fn test_tensorflow_adam_creation() {
        let optimizer = TensorFlowAdam::with_defaults().unwrap();
        assert_eq!(optimizer.get_learning_rate(), 0.001);
        assert_eq!(optimizer.get_name(), "Adam");
    }

    #[test]
    fn test_tensorflow_adamw_creation() {
        let optimizer = TensorFlowAdamW::with_defaults().unwrap();
        assert_eq!(optimizer.get_learning_rate(), 0.001);
        assert_eq!(optimizer.get_name(), "AdamW");
    }

    #[test]
    fn test_tensorflow_exponential_decay() {
        let schedule = TensorFlowExponentialDecay::new(0.1, 100, 0.96, false);
        assert_eq!(schedule.get_lr(0), 0.1);
        assert!(schedule.get_lr(100) < 0.1);
    }

    #[test]
    fn test_tensorflow_cosine_decay() {
        let schedule = TensorFlowCosineDecay::new(0.1, 100, 0.0);
        assert_eq!(schedule.get_lr(0), 0.1);
        assert!(schedule.get_lr(50) < 0.1);
        assert!(schedule.get_lr(100) < 0.1);
    }

    #[test]
    fn test_tensorflow_optimizer_factory() {
        let adam = TensorFlowOptimizerFactory::adam(
            0.001,
            0.9,
            0.999,
            1e-7,
            None,
            None,
            None,
            None,
            false,
            0.99,
            true,
            Some("TestAdam".to_string()),
        )
        .unwrap();
        assert_eq!(adam.get_name(), "TestAdam");

        let adamw = TensorFlowOptimizerFactory::adamw(
            0.001,
            0.9,
            0.999,
            1e-7,
            0.01,
            None,
            None,
            None,
            false,
            0.99,
            true,
            Some("TestAdamW".to_string()),
        )
        .unwrap();
        assert_eq!(adamw.get_name(), "TestAdamW");
    }

    #[test]
    fn test_learning_rate_schedule_with_optimizer() {
        let schedule = Box::new(TensorFlowExponentialDecay::new(0.1, 100, 0.96, false));
        let optimizer = TensorFlowAdam::with_schedule(
            schedule,
            0.9,
            0.999,
            1e-7,
            None,
            None,
            None,
            None,
            false,
            0.99,
            true,
            Some("ScheduledAdam".to_string()),
        )
        .unwrap();

        assert_eq!(optimizer.get_learning_rate(), 0.1);
    }

    #[test]
    fn test_variable_management() {
        let mut optimizer = TensorFlowAdam::default().unwrap();

        let var1 = Tensor::zeros(&[10, 10]).unwrap();
        let var2 = Tensor::zeros(&[5, 5]).unwrap();

        optimizer.add_variable("var1".to_string(), var1).unwrap();
        optimizer.add_variable("var2".to_string(), var2).unwrap();

        let variables = optimizer.variables();
        assert_eq!(variables.len(), 2);
        assert!(variables.contains(&"var1".to_string()));
        assert!(variables.contains(&"var2".to_string()));
    }

    #[test]
    fn test_learning_rate_updates() {
        let mut optimizer = TensorFlowAdam::default().unwrap();
        assert_eq!(optimizer.get_learning_rate(), 0.001);

        optimizer.set_learning_rate(0.01).unwrap();
        assert_eq!(optimizer.get_learning_rate(), 0.01);
    }

    #[test]
    fn test_config_serialization() {
        let optimizer = TensorFlowAdam::with_defaults().unwrap();
        let config = optimizer.get_config();

        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.beta_1, Some(0.9));
        assert_eq!(config.beta_2, Some(0.999));
        assert_eq!(config.epsilon, Some(1e-7));
    }
}
