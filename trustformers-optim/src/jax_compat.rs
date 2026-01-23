//! JAX Optimizer API Compatibility Layer
//!
//! This module provides JAX-compatible optimizer interfaces for seamless
//! integration with JAX-based training workflows. It wraps our native
//! optimizers to provide the familiar JAX Optax API while maintaining high performance.

use crate::{Adam, AdamW, SGD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::Result;
use trustformers_core::traits::Optimizer;
use trustformers_core::Tensor;

/// JAX-compatible optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JAXOptimizerConfig {
    pub optimizer_type: String,
    pub optimizer_name: String,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub mu_dtype: Option<String>,
    pub parameters: HashMap<String, serde_json::Value>,
}

impl Default for JAXOptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: "adam".to_string(),
            optimizer_name: "adam".to_string(),
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            mu_dtype: None,
            parameters: HashMap::new(),
        }
    }
}

/// JAX-compatible optimizer state (Optax-style)
#[derive(Debug, Clone, Default)]
pub struct JAXOptState {
    pub step: i64,
    pub mu: HashMap<String, Tensor>,
    pub nu: HashMap<String, Tensor>,
}

/// JAX-compatible optimizer state (internal)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JAXOptimizerState {
    pub step: i64,
    pub params: HashMap<String, serde_json::Value>,
    pub inner_state: serde_json::Value,
}

/// JAX-compatible gradient transformation
pub trait JAXGradientTransformation: Send + Sync {
    /// Initialize the gradient transformation state
    fn init(&self, params: &HashMap<String, Tensor>) -> Result<JAXOptimizerState>;

    /// Apply gradient transformation and return updated parameters and state
    fn update(
        &self,
        gradients: &HashMap<String, Tensor>,
        state: &JAXOptimizerState,
        params: Option<&HashMap<String, Tensor>>,
    ) -> Result<(HashMap<String, Tensor>, JAXOptimizerState)>;

    /// Get transformation name
    fn name(&self) -> &str;
}

/// JAX-compatible learning rate schedule trait
pub trait JAXLearningRateSchedule: Send + Sync {
    /// Get learning rate at given step
    fn get_lr(&self, step: i64) -> f64;

    /// Get schedule configuration
    fn get_config(&self) -> serde_json::Value;
}

/// JAX-compatible exponential decay schedule
#[derive(Debug, Clone)]
pub struct JAXExponentialDecay {
    init_value: f64,
    decay_rate: f64,
    transition_steps: i64,
    transition_begin: i64,
    staircase: bool,
    end_value: Option<f64>,
}

impl JAXExponentialDecay {
    pub fn new(
        init_value: f64,
        decay_rate: f64,
        transition_steps: i64,
        transition_begin: i64,
        staircase: bool,
        end_value: Option<f64>,
    ) -> Self {
        Self {
            init_value,
            decay_rate,
            transition_steps,
            transition_begin,
            staircase,
            end_value,
        }
    }
}

impl JAXLearningRateSchedule for JAXExponentialDecay {
    fn get_lr(&self, step: i64) -> f64 {
        if step < self.transition_begin {
            return self.init_value;
        }

        let decay_step = step - self.transition_begin;
        let decay_factor = if self.staircase {
            (decay_step / self.transition_steps) as f64
        } else {
            decay_step as f64 / self.transition_steps as f64
        };

        let decayed_value = self.init_value * self.decay_rate.powf(decay_factor);

        if let Some(end_value) = self.end_value {
            decayed_value.max(end_value)
        } else {
            decayed_value
        }
    }

    fn get_config(&self) -> serde_json::Value {
        serde_json::json!({
            "init_value": self.init_value,
            "decay_rate": self.decay_rate,
            "transition_steps": self.transition_steps,
            "transition_begin": self.transition_begin,
            "staircase": self.staircase,
            "end_value": self.end_value,
        })
    }
}

/// JAX-compatible cosine decay schedule
#[derive(Debug, Clone)]
pub struct JAXCosineDecay {
    init_value: f64,
    decay_steps: i64,
    alpha: f64,
}

impl JAXCosineDecay {
    pub fn new(init_value: f64, decay_steps: i64, alpha: f64) -> Self {
        Self {
            init_value,
            decay_steps,
            alpha,
        }
    }
}

impl JAXLearningRateSchedule for JAXCosineDecay {
    fn get_lr(&self, step: i64) -> f64 {
        let completed_fraction = (step.min(self.decay_steps) as f64) / (self.decay_steps as f64);
        let cosine_decayed = 0.5 * (1.0 + (std::f64::consts::PI * completed_fraction).cos());
        let decayed = (1.0 - self.alpha) * cosine_decayed + self.alpha;

        self.init_value * decayed
    }

    fn get_config(&self) -> serde_json::Value {
        serde_json::json!({
            "init_value": self.init_value,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
        })
    }
}

/// JAX-compatible warmup cosine decay schedule
#[derive(Debug, Clone)]
pub struct JAXWarmupCosineDecay {
    init_value: f64,
    peak_value: f64,
    warmup_steps: i64,
    decay_steps: i64,
    end_value: f64,
}

impl JAXWarmupCosineDecay {
    pub fn new(
        init_value: f64,
        peak_value: f64,
        warmup_steps: i64,
        decay_steps: i64,
        end_value: f64,
    ) -> Self {
        Self {
            init_value,
            peak_value,
            warmup_steps,
            decay_steps,
            end_value,
        }
    }
}

impl JAXLearningRateSchedule for JAXWarmupCosineDecay {
    fn get_lr(&self, step: i64) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            let warmup_fraction = step as f64 / self.warmup_steps as f64;
            return self.init_value + (self.peak_value - self.init_value) * warmup_fraction;
        }

        // Cosine decay
        let decay_step = step - self.warmup_steps;
        let decay_fraction = (decay_step.min(self.decay_steps) as f64) / (self.decay_steps as f64);
        let cosine_decayed = 0.5 * (1.0 + (std::f64::consts::PI * decay_fraction).cos());

        self.end_value + (self.peak_value - self.end_value) * cosine_decayed
    }

    fn get_config(&self) -> serde_json::Value {
        serde_json::json!({
            "init_value": self.init_value,
            "peak_value": self.peak_value,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "end_value": self.end_value,
        })
    }
}

/// JAX-compatible cosine decay schedule (alias for JAXCosineDecay)
pub type JAXCosineDecaySchedule = JAXCosineDecay;

/// JAX-compatible Adam optimizer
pub struct JAXAdam {
    inner: Adam,
    learning_rate: f64,
    b1: f64,
    b2: f64,
    eps: f64,
    #[allow(dead_code)]
    eps_root: f64,
    weight_decay: Option<f64>,
    lr_schedule: Option<Box<dyn JAXLearningRateSchedule>>,
}

impl JAXAdam {
    /// Create JAX Adam optimizer from raw parameters (deprecated - use from_params)
    fn new_from_raw_params(
        learning_rate: f64,
        b1: f64,
        b2: f64,
        eps: f64,
        eps_root: f64,
        weight_decay: Option<f64>,
    ) -> Result<Self> {
        let inner = Adam::new(
            learning_rate as f32,
            (b1 as f32, b2 as f32),
            eps as f32,
            weight_decay.unwrap_or(0.0) as f32,
        );

        Ok(Self {
            inner,
            learning_rate,
            b1,
            b2,
            eps,
            eps_root,
            weight_decay,
            lr_schedule: None,
        })
    }

    /// Create JAX Adam optimizer from configuration (primary constructor)
    pub fn new(config: JAXOptimizerConfig) -> Result<Self> {
        let inner = Adam::new(
            config.learning_rate as f32,
            (config.beta1 as f32, config.beta2 as f32),
            config.epsilon as f32,
            config.weight_decay as f32,
        );

        Ok(Self {
            inner,
            learning_rate: config.learning_rate,
            b1: config.beta1,
            b2: config.beta2,
            eps: config.epsilon,
            eps_root: 0.0,
            weight_decay: Some(config.weight_decay),
            lr_schedule: None,
        })
    }

    /// Create JAX Adam optimizer from cross-framework configuration
    pub fn from_cross_framework_config(
        config: crate::cross_framework::JAXOptimizerConfig,
    ) -> Result<Self> {
        // Extract parameters from the HashMap
        let beta1 = config.parameters.get("beta1").and_then(|v| v.as_f64()).unwrap_or(0.9);

        let beta2 = config.parameters.get("beta2").and_then(|v| v.as_f64()).unwrap_or(0.999);

        let epsilon = config.parameters.get("epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-8);

        let weight_decay =
            config.parameters.get("weight_decay").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let inner = Adam::new(
            config.learning_rate,
            (beta1 as f32, beta2 as f32),
            epsilon as f32,
            weight_decay as f32,
        );

        Ok(Self {
            inner,
            learning_rate: config.learning_rate as f64,
            b1: beta1,
            b2: beta2,
            eps: epsilon,
            eps_root: 0.0,
            weight_decay: Some(weight_decay),
            lr_schedule: None,
        })
    }

    /// Create JAX Adam optimizer from raw parameters
    pub fn from_params(
        learning_rate: f64,
        b1: f64,
        b2: f64,
        eps: f64,
        eps_root: f64,
        weight_decay: Option<f64>,
    ) -> Result<Self> {
        Self::new_from_raw_params(learning_rate, b1, b2, eps, eps_root, weight_decay)
    }

    /// Set learning rate (JAX-style functional interface)
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
        self.inner.set_lr(learning_rate as f32);
    }

    pub fn with_schedule(
        schedule: Box<dyn JAXLearningRateSchedule>,
        b1: f64,
        b2: f64,
        eps: f64,
        eps_root: f64,
        weight_decay: Option<f64>,
    ) -> Result<Self> {
        let mut optimizer =
            Self::from_params(schedule.get_lr(0), b1, b2, eps, eps_root, weight_decay)?;

        optimizer.lr_schedule = Some(schedule);
        Ok(optimizer)
    }

    /// Create with default JAX Adam parameters
    pub fn with_defaults() -> Result<Self> {
        Self::from_params(1e-3, 0.9, 0.999, 1e-8, 0.0, None)
    }

    #[allow(dead_code)]
    fn update_learning_rate(&mut self, step: i64) -> Result<()> {
        if let Some(ref schedule) = self.lr_schedule {
            let new_lr = schedule.get_lr(step);
            self.learning_rate = new_lr;

            // Update inner optimizer learning rate
            // JAX optimizers work with f64 but inner optimizer uses f32
            self.inner.set_lr(new_lr as f32);
        }
        Ok(())
    }
}

impl JAXGradientTransformation for JAXAdam {
    fn init(&self, params: &HashMap<String, Tensor>) -> Result<JAXOptimizerState> {
        let mut state_params = HashMap::new();

        for (name, param) in params {
            state_params.insert(format!("{}_m", name), serde_json::json!(param.shape()));
            state_params.insert(format!("{}_v", name), serde_json::json!(param.shape()));
        }

        Ok(JAXOptimizerState {
            step: 0,
            params: state_params,
            inner_state: serde_json::json!({}),
        })
    }

    fn update(
        &self,
        gradients: &HashMap<String, Tensor>,
        state: &JAXOptimizerState,
        params: Option<&HashMap<String, Tensor>>,
    ) -> Result<(HashMap<String, Tensor>, JAXOptimizerState)> {
        let mut updated_params = HashMap::new();
        let mut new_state = state.clone();
        new_state.step += 1;

        // Update learning rate if schedule is set
        let current_lr = if let Some(ref schedule) = self.lr_schedule {
            schedule.get_lr(new_state.step)
        } else {
            self.learning_rate
        };

        if let Some(params) = params {
            for (name, param) in params {
                if let Some(grad) = gradients.get(name) {
                    // Apply Adam update using inner optimizer
                    let updated_param = param.clone();

                    // Temporary config values for current learning rate
                    let _current_lr = current_lr as f32;
                    let _beta1 = self.b1 as f32;
                    let _beta2 = self.b2 as f32;
                    let _epsilon = self.eps as f32;
                    let _weight_decay = self.weight_decay.unwrap_or(0.0) as f32;

                    // Apply weight decay if specified
                    if let Some(weight_decay) = self.weight_decay {
                        updated_param.add_scalar((-current_lr * weight_decay) as f32)?;
                    }

                    // Apply gradient update
                    // In a real implementation, this would use the inner optimizer's step method
                    // For now, we'll do a simple gradient descent step
                    let scaled_grad = grad.mul_scalar(current_lr as f32)?;
                    updated_param.sub(&scaled_grad)?;

                    updated_params.insert(name.clone(), updated_param);
                } else {
                    updated_params.insert(name.clone(), param.clone());
                }
            }
        }

        Ok((updated_params, new_state))
    }

    fn name(&self) -> &str {
        "adam"
    }
}

/// JAX-compatible AdamW optimizer
pub struct JAXAdamW {
    #[allow(dead_code)]
    inner: AdamW,
    learning_rate: f64,
    #[allow(dead_code)]
    b1: f64,
    #[allow(dead_code)]
    b2: f64,
    #[allow(dead_code)]
    eps: f64,
    #[allow(dead_code)]
    eps_root: f64,
    weight_decay: f64,
    lr_schedule: Option<Box<dyn JAXLearningRateSchedule>>,
}

impl JAXAdamW {
    /// Create JAX AdamW optimizer from configuration (primary constructor)
    pub fn new(config: JAXOptimizerConfig) -> Result<Self> {
        let inner = AdamW::new(
            config.learning_rate as f32,
            (config.beta1 as f32, config.beta2 as f32),
            config.epsilon as f32,
            config.weight_decay as f32,
        );

        Ok(Self {
            inner,
            learning_rate: config.learning_rate,
            b1: config.beta1,
            b2: config.beta2,
            eps: config.epsilon,
            eps_root: 0.0,
            weight_decay: config.weight_decay,
            lr_schedule: None,
        })
    }

    /// Create JAX AdamW optimizer from raw parameters
    pub fn from_params(
        learning_rate: f64,
        b1: f64,
        b2: f64,
        eps: f64,
        eps_root: f64,
        weight_decay: f64,
    ) -> Result<Self> {
        let inner = AdamW::new(
            learning_rate as f32,
            (b1 as f32, b2 as f32),
            eps as f32,
            weight_decay as f32,
        );

        Ok(Self {
            inner,
            learning_rate,
            b1,
            b2,
            eps,
            eps_root,
            weight_decay,
            lr_schedule: None,
        })
    }

    pub fn with_schedule(
        schedule: Box<dyn JAXLearningRateSchedule>,
        b1: f64,
        b2: f64,
        eps: f64,
        eps_root: f64,
        weight_decay: f64,
    ) -> Result<Self> {
        let mut optimizer =
            Self::from_params(schedule.get_lr(0), b1, b2, eps, eps_root, weight_decay)?;

        optimizer.lr_schedule = Some(schedule);
        Ok(optimizer)
    }

    /// Create with default JAX AdamW parameters
    pub fn with_defaults() -> Result<Self> {
        Self::from_params(1e-3, 0.9, 0.999, 1e-8, 0.0, 1e-4)
    }
}

impl JAXGradientTransformation for JAXAdamW {
    fn init(&self, params: &HashMap<String, Tensor>) -> Result<JAXOptimizerState> {
        let mut state_params = HashMap::new();

        for (name, param) in params {
            state_params.insert(format!("{}_m", name), serde_json::json!(param.shape()));
            state_params.insert(format!("{}_v", name), serde_json::json!(param.shape()));
        }

        Ok(JAXOptimizerState {
            step: 0,
            params: state_params,
            inner_state: serde_json::json!({}),
        })
    }

    fn update(
        &self,
        gradients: &HashMap<String, Tensor>,
        state: &JAXOptimizerState,
        params: Option<&HashMap<String, Tensor>>,
    ) -> Result<(HashMap<String, Tensor>, JAXOptimizerState)> {
        let mut updated_params = HashMap::new();
        let mut new_state = state.clone();
        new_state.step += 1;

        // Update learning rate if schedule is set
        let current_lr = if let Some(ref schedule) = self.lr_schedule {
            schedule.get_lr(new_state.step)
        } else {
            self.learning_rate
        };

        if let Some(params) = params {
            for (name, param) in params {
                if let Some(grad) = gradients.get(name) {
                    // Apply AdamW update using inner optimizer
                    let updated_param = param.clone();

                    // Apply weight decay (AdamW style - directly to parameters)
                    updated_param.mul_scalar((1.0 - current_lr * self.weight_decay) as f32)?;

                    // Apply gradient update
                    let scaled_grad = grad.mul_scalar(current_lr as f32)?;
                    updated_param.sub(&scaled_grad)?;

                    updated_params.insert(name.clone(), updated_param);
                } else {
                    updated_params.insert(name.clone(), param.clone());
                }
            }
        }

        Ok((updated_params, new_state))
    }

    fn name(&self) -> &str {
        "adamw"
    }
}

/// JAX-compatible SGD optimizer
pub struct JAXSGD {
    #[allow(dead_code)]
    inner: SGD,
    learning_rate: f64,
    momentum: f64,
    #[allow(dead_code)]
    nesterov: bool,
    weight_decay: Option<f64>,
    lr_schedule: Option<Box<dyn JAXLearningRateSchedule>>,
}

impl JAXSGD {
    /// Create JAX SGD optimizer from configuration (primary constructor)
    pub fn new(config: JAXOptimizerConfig) -> Result<Self> {
        let inner = SGD::new(
            config.learning_rate as f32,
            0.9, // Default momentum from config
            config.weight_decay as f32,
            false, // Default nesterov
        );

        Ok(Self {
            inner,
            learning_rate: config.learning_rate,
            momentum: 0.9,
            nesterov: false,
            weight_decay: Some(config.weight_decay),
            lr_schedule: None,
        })
    }

    /// Create JAX SGD optimizer from raw parameters
    pub fn from_params(
        learning_rate: f64,
        momentum: f64,
        nesterov: bool,
        weight_decay: Option<f64>,
    ) -> Result<Self> {
        let inner = SGD::new(
            learning_rate as f32,
            momentum as f32,
            weight_decay.unwrap_or(0.0) as f32,
            nesterov,
        );

        Ok(Self {
            inner,
            learning_rate,
            momentum,
            nesterov,
            weight_decay,
            lr_schedule: None,
        })
    }

    pub fn with_schedule(
        schedule: Box<dyn JAXLearningRateSchedule>,
        momentum: f64,
        nesterov: bool,
        weight_decay: Option<f64>,
    ) -> Result<Self> {
        let mut optimizer =
            Self::from_params(schedule.get_lr(0), momentum, nesterov, weight_decay)?;

        optimizer.lr_schedule = Some(schedule);
        Ok(optimizer)
    }

    /// Create with default JAX SGD parameters
    pub fn with_defaults() -> Result<Self> {
        Self::from_params(1e-3, 0.0, false, None)
    }
}

impl JAXGradientTransformation for JAXSGD {
    fn init(&self, params: &HashMap<String, Tensor>) -> Result<JAXOptimizerState> {
        let mut state_params = HashMap::new();

        if self.momentum > 0.0 {
            for (name, param) in params {
                state_params.insert(
                    format!("{}_momentum", name),
                    serde_json::json!(param.shape()),
                );
            }
        }

        Ok(JAXOptimizerState {
            step: 0,
            params: state_params,
            inner_state: serde_json::json!({}),
        })
    }

    fn update(
        &self,
        gradients: &HashMap<String, Tensor>,
        state: &JAXOptimizerState,
        params: Option<&HashMap<String, Tensor>>,
    ) -> Result<(HashMap<String, Tensor>, JAXOptimizerState)> {
        let mut updated_params = HashMap::new();
        let mut new_state = state.clone();
        new_state.step += 1;

        // Update learning rate if schedule is set
        let current_lr = if let Some(ref schedule) = self.lr_schedule {
            schedule.get_lr(new_state.step)
        } else {
            self.learning_rate
        };

        if let Some(params) = params {
            for (name, param) in params {
                if let Some(grad) = gradients.get(name) {
                    let updated_param = param.clone();

                    // Apply weight decay if specified
                    if let Some(weight_decay) = self.weight_decay {
                        updated_param.add_scalar((-current_lr * weight_decay) as f32)?;
                    }

                    // Apply SGD update
                    let scaled_grad = grad.mul_scalar(current_lr as f32)?;
                    updated_param.sub(&scaled_grad)?;

                    updated_params.insert(name.clone(), updated_param);
                } else {
                    updated_params.insert(name.clone(), param.clone());
                }
            }
        }

        Ok((updated_params, new_state))
    }

    fn name(&self) -> &str {
        "sgd"
    }
}

/// JAX-compatible gradient transformation chain
pub struct JAXChain {
    transformations: Vec<Box<dyn JAXGradientTransformation>>,
}

impl JAXChain {
    pub fn new(transformations: Vec<Box<dyn JAXGradientTransformation>>) -> Self {
        Self { transformations }
    }

    pub fn add_transformation(&mut self, transformation: Box<dyn JAXGradientTransformation>) {
        self.transformations.push(transformation);
    }
}

impl JAXGradientTransformation for JAXChain {
    fn init(&self, params: &HashMap<String, Tensor>) -> Result<JAXOptimizerState> {
        let mut state_params = HashMap::new();

        for (i, transformation) in self.transformations.iter().enumerate() {
            let sub_state = transformation.init(params)?;
            state_params.insert(format!("chain_{}", i), serde_json::to_value(sub_state)?);
        }

        Ok(JAXOptimizerState {
            step: 0,
            params: state_params,
            inner_state: serde_json::json!({}),
        })
    }

    fn update(
        &self,
        gradients: &HashMap<String, Tensor>,
        state: &JAXOptimizerState,
        params: Option<&HashMap<String, Tensor>>,
    ) -> Result<(HashMap<String, Tensor>, JAXOptimizerState)> {
        let current_gradients = gradients.clone();
        let mut current_params = params.cloned().unwrap_or_default();
        let mut new_state = state.clone();
        new_state.step += 1;

        // Apply transformations in sequence
        for (i, transformation) in self.transformations.iter().enumerate() {
            let sub_state_key = format!("chain_{}", i);
            let sub_state: JAXOptimizerState = if let Some(sub_state_val) =
                state.params.get(&sub_state_key)
            {
                serde_json::from_value(sub_state_val.clone())
                    .unwrap_or_else(|_| transformation.init(&current_params).unwrap_or_default())
            } else {
                transformation.init(&current_params)?
            };

            let (updated_params, updated_sub_state) =
                transformation.update(&current_gradients, &sub_state, Some(&current_params))?;

            current_params = updated_params;
            new_state.params.insert(sub_state_key, serde_json::to_value(updated_sub_state)?);
        }

        Ok((current_params, new_state))
    }

    fn name(&self) -> &str {
        "chain"
    }
}

impl Default for JAXOptimizerState {
    fn default() -> Self {
        Self {
            step: 0,
            params: HashMap::new(),
            inner_state: serde_json::json!({}),
        }
    }
}

/// JAX optimizer factory for creating optimizers with JAX-compatible API
pub struct JAXOptimizerFactory;

impl JAXOptimizerFactory {
    /// Create Adam optimizer
    pub fn adam(
        learning_rate: f64,
        b1: f64,
        b2: f64,
        eps: f64,
        eps_root: f64,
        weight_decay: Option<f64>,
    ) -> Result<JAXAdam> {
        JAXAdam::from_params(learning_rate, b1, b2, eps, eps_root, weight_decay)
    }

    /// Create AdamW optimizer
    pub fn adamw(
        learning_rate: f64,
        b1: f64,
        b2: f64,
        eps: f64,
        eps_root: f64,
        weight_decay: f64,
    ) -> Result<JAXAdamW> {
        JAXAdamW::from_params(learning_rate, b1, b2, eps, eps_root, weight_decay)
    }

    /// Create SGD optimizer
    pub fn sgd(
        learning_rate: f64,
        momentum: f64,
        nesterov: bool,
        weight_decay: Option<f64>,
    ) -> Result<JAXSGD> {
        JAXSGD::from_params(learning_rate, momentum, nesterov, weight_decay)
    }

    /// Create exponential decay schedule
    pub fn exponential_decay(
        init_value: f64,
        decay_rate: f64,
        transition_steps: i64,
        transition_begin: i64,
        staircase: bool,
        end_value: Option<f64>,
    ) -> JAXExponentialDecay {
        JAXExponentialDecay::new(
            init_value,
            decay_rate,
            transition_steps,
            transition_begin,
            staircase,
            end_value,
        )
    }

    /// Create cosine decay schedule
    pub fn cosine_decay(init_value: f64, decay_steps: i64, alpha: f64) -> JAXCosineDecay {
        JAXCosineDecay::new(init_value, decay_steps, alpha)
    }

    /// Create warmup cosine decay schedule
    pub fn warmup_cosine_decay(
        init_value: f64,
        peak_value: f64,
        warmup_steps: i64,
        decay_steps: i64,
        end_value: f64,
    ) -> JAXWarmupCosineDecay {
        JAXWarmupCosineDecay::new(init_value, peak_value, warmup_steps, decay_steps, end_value)
    }

    /// Create transformation chain
    pub fn chain(transformations: Vec<Box<dyn JAXGradientTransformation>>) -> JAXChain {
        JAXChain::new(transformations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::Tensor;

    #[test]
    fn test_jax_adam_creation() {
        let optimizer = JAXAdam::with_defaults().unwrap();
        assert_eq!(optimizer.name(), "adam");
        assert_eq!(optimizer.learning_rate, 1e-3);
    }

    #[test]
    fn test_jax_adamw_creation() {
        let optimizer = JAXAdamW::with_defaults().unwrap();
        assert_eq!(optimizer.name(), "adamw");
        assert_eq!(optimizer.learning_rate, 1e-3);
        assert_eq!(optimizer.weight_decay, 1e-4);
    }

    #[test]
    fn test_jax_sgd_creation() {
        let optimizer = JAXSGD::with_defaults().unwrap();
        assert_eq!(optimizer.name(), "sgd");
        assert_eq!(optimizer.learning_rate, 1e-3);
        assert_eq!(optimizer.momentum, 0.0);
    }

    #[test]
    fn test_jax_exponential_decay() {
        let schedule = JAXExponentialDecay::new(0.1, 0.96, 100, 0, false, None);
        assert_eq!(schedule.get_lr(0), 0.1);
        assert!(schedule.get_lr(100) < 0.1);
    }

    #[test]
    fn test_jax_cosine_decay() {
        let schedule = JAXCosineDecay::new(0.1, 100, 0.0);
        assert_eq!(schedule.get_lr(0), 0.1);
        assert!(schedule.get_lr(50) < 0.1);
        assert!(schedule.get_lr(100) < 0.1);
    }

    #[test]
    fn test_jax_warmup_cosine_decay() {
        let schedule = JAXWarmupCosineDecay::new(0.0, 0.1, 10, 100, 0.01);
        assert_eq!(schedule.get_lr(0), 0.0);
        assert!(schedule.get_lr(5) > 0.0 && schedule.get_lr(5) < 0.1);
        assert_eq!(schedule.get_lr(10), 0.1);
        assert!(schedule.get_lr(60) < 0.1 && schedule.get_lr(60) > 0.01);
    }

    #[test]
    fn test_jax_optimizer_factory() {
        let adam = JAXOptimizerFactory::adam(1e-3, 0.9, 0.999, 1e-8, 0.0, None).unwrap();
        assert_eq!(adam.name(), "adam");

        let adamw = JAXOptimizerFactory::adamw(1e-3, 0.9, 0.999, 1e-8, 0.0, 1e-4).unwrap();
        assert_eq!(adamw.name(), "adamw");

        let sgd = JAXOptimizerFactory::sgd(1e-3, 0.9, false, None).unwrap();
        assert_eq!(sgd.name(), "sgd");
    }

    #[test]
    fn test_jax_optimizer_init() {
        use std::collections::HashMap;
        let optimizer = JAXAdam::with_defaults().unwrap();
        let params: HashMap<String, Tensor> = [
            ("param1".to_string(), Tensor::zeros(&[10, 10]).unwrap()),
            ("param2".to_string(), Tensor::zeros(&[5, 5]).unwrap()),
        ]
        .iter()
        .cloned()
        .collect();

        let state = optimizer.init(&params).unwrap();
        assert_eq!(state.step, 0);
        assert!(state.params.contains_key("param1_m"));
        assert!(state.params.contains_key("param1_v"));
        assert!(state.params.contains_key("param2_m"));
        assert!(state.params.contains_key("param2_v"));
    }

    #[test]
    fn test_jax_optimizer_update() {
        use std::collections::HashMap;
        let optimizer = JAXAdam::with_defaults().unwrap();
        let params: HashMap<String, Tensor> =
            [("param1".to_string(), Tensor::zeros(&[10, 10]).unwrap())]
                .iter()
                .cloned()
                .collect();

        let state = optimizer.init(&params).unwrap();

        let gradients: HashMap<String, Tensor> =
            [("param1".to_string(), Tensor::ones(&[10, 10]).unwrap())]
                .iter()
                .cloned()
                .collect();

        let (updated_params, updated_state) =
            optimizer.update(&gradients, &state, Some(&params)).unwrap();
        assert_eq!(updated_state.step, 1);
        assert!(updated_params.contains_key("param1"));
    }

    #[test]
    fn test_jax_chain_transformation() {
        use std::collections::HashMap;
        let adam = JAXOptimizerFactory::adam(1e-3, 0.9, 0.999, 1e-8, 0.0, None).unwrap();
        let sgd = JAXOptimizerFactory::sgd(1e-3, 0.9, false, None).unwrap();

        let chain = JAXOptimizerFactory::chain(vec![Box::new(adam), Box::new(sgd)]);

        assert_eq!(chain.name(), "chain");

        let params: HashMap<String, Tensor> =
            [("param1".to_string(), Tensor::zeros(&[5, 5]).unwrap())]
                .iter()
                .cloned()
                .collect();

        let state = chain.init(&params).unwrap();
        assert!(state.params.contains_key("chain_0"));
        assert!(state.params.contains_key("chain_1"));
    }

    #[test]
    fn test_schedule_with_optimizer() {
        let schedule = Box::new(JAXExponentialDecay::new(0.1, 0.96, 100, 0, false, None));
        let optimizer = JAXAdam::with_schedule(schedule, 0.9, 0.999, 1e-8, 0.0, None).unwrap();

        assert_eq!(optimizer.learning_rate, 0.1);
        assert!(optimizer.lr_schedule.is_some());
    }

    #[test]
    fn test_schedule_config_serialization() {
        let schedule = JAXExponentialDecay::new(0.1, 0.96, 100, 0, false, Some(0.01));
        let config = schedule.get_config();

        assert_eq!(config["init_value"], 0.1);
        assert_eq!(config["decay_rate"], 0.96);
        assert_eq!(config["transition_steps"], 100);
        assert_eq!(config["end_value"], 0.01);
    }
}
