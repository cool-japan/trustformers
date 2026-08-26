//! JAX Optimizer API Compatibility Layer
//!
//! This module provides JAX-compatible optimizer interfaces for seamless
//! integration with JAX-based training workflows. It wraps our native
//! optimizers to provide the familiar JAX Optax API while maintaining high performance.

// reason: research-stage module — reserved API/scaffolding fields and methods
// retained intentionally for in-progress features; not yet on active call paths.
#![allow(dead_code)]

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

/// Reads a per-parameter moment buffer out of the functional Optax-style state.
///
/// Optax keeps optimizer state *outside* the optimizer object (`update` takes `&self`),
/// so the moments live in [`JAXOptimizerState::params`] as JSON arrays. A missing or
/// wrongly-sized entry is treated as a fresh zero buffer, which matches Optax's
/// behaviour when a parameter tree grows between steps.
fn read_moment(state: &JAXOptimizerState, key: &str, len: usize) -> Vec<f32> {
    match state.params.get(key).and_then(|value| value.as_array()) {
        Some(values) if values.len() == len => {
            values.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()
        },
        _ => vec![0.0_f32; len],
    }
}

/// Writes a per-parameter moment buffer back into the functional state.
fn write_moment(state: &mut JAXOptimizerState, key: &str, values: &[f32]) {
    let encoded: Vec<serde_json::Value> = values
        .iter()
        .map(|&v| {
            serde_json::Number::from_f64(v as f64)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        })
        .collect();
    state.params.insert(key.to_string(), serde_json::Value::Array(encoded));
}

/// Allocates zeroed moment buffers for every parameter under the given suffixes.
fn init_moments(params: &HashMap<String, Tensor>, suffixes: &[&str]) -> JAXOptimizerState {
    let mut state = JAXOptimizerState {
        step: 0,
        params: HashMap::new(),
        inner_state: serde_json::json!({}),
    };
    for (name, param) in params {
        let len = param.shape().iter().product::<usize>();
        for suffix in suffixes {
            write_moment(&mut state, &format!("{name}{suffix}"), &vec![0.0_f32; len]);
        }
    }
    state
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
        Ok(init_moments(params, &["_m", "_v"]))
    }

    /// Applies one Optax `adam` (or `adamw`, when `weight_decay` is set) step.
    ///
    /// ```text
    /// m_t = b1 * m_{t-1} + (1 - b1) * g
    /// v_t = b2 * v_{t-1} + (1 - b2) * g^2
    /// update = lr * (m_t / (1 - b1^t)) / (sqrt(v_t / (1 - b2^t) + eps_root) + eps)
    /// p_t = p_{t-1} - update - lr * weight_decay * p_{t-1}
    /// ```
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

        let step = new_state.step.max(1) as i32;
        let bias_correction1 = 1.0 - self.b1.powi(step);
        let bias_correction2 = 1.0 - self.b2.powi(step);
        let weight_decay = self.weight_decay.unwrap_or(0.0);

        if let Some(params) = params {
            for (name, param) in params {
                let Some(grad) = gradients.get(name) else {
                    updated_params.insert(name.clone(), param.clone());
                    continue;
                };

                let param_data = param.data_f32()?;
                let grad_data = grad.data_f32()?;
                if grad_data.len() != param_data.len() {
                    return Err(trustformers_core::errors::TrustformersError::invalid_input(
                        format!(
                            "gradient for '{name}' has {} elements but the parameter has {}",
                            grad_data.len(),
                            param_data.len()
                        ),
                    ));
                }

                let mut mu = read_moment(state, &format!("{name}_m"), param_data.len());
                let mut nu = read_moment(state, &format!("{name}_v"), param_data.len());
                let mut updated = Vec::with_capacity(param_data.len());

                for index in 0..param_data.len() {
                    let g = grad_data[index] as f64;
                    let p = param_data[index] as f64;

                    mu[index] = (self.b1 * mu[index] as f64 + (1.0 - self.b1) * g) as f32;
                    nu[index] = (self.b2 * nu[index] as f64 + (1.0 - self.b2) * g * g) as f32;

                    let m_hat = mu[index] as f64 / bias_correction1;
                    let v_hat = nu[index] as f64 / bias_correction2;

                    let mut delta =
                        current_lr * m_hat / ((v_hat + self.eps_root).sqrt() + self.eps);
                    // Optax applies decoupled weight decay (`adamw`) on top of the
                    // Adam update, not to the gradient.
                    delta += current_lr * weight_decay * p;
                    updated.push((p - delta) as f32);
                }

                write_moment(&mut new_state, &format!("{name}_m"), &mu);
                write_moment(&mut new_state, &format!("{name}_v"), &nu);
                updated_params.insert(name.clone(), Tensor::from_vec(updated, &param.shape())?);
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
    inner: AdamW,
    learning_rate: f64,
    b1: f64,
    b2: f64,
    eps: f64,
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
        Ok(init_moments(params, &["_m", "_v"]))
    }

    /// Applies one Optax `adamw` step (Adam plus decoupled weight decay).
    fn update(
        &self,
        gradients: &HashMap<String, Tensor>,
        state: &JAXOptimizerState,
        params: Option<&HashMap<String, Tensor>>,
    ) -> Result<(HashMap<String, Tensor>, JAXOptimizerState)> {
        // AdamW is exactly Adam with a mandatory decoupled decay term, so route it
        // through one implementation instead of duplicating the math.
        let adam = JAXAdam {
            inner: Adam::new(
                self.learning_rate as f32,
                (self.b1 as f32, self.b2 as f32),
                self.eps as f32,
                0.0,
            ),
            learning_rate: self.learning_rate,
            b1: self.b1,
            b2: self.b2,
            eps: self.eps,
            eps_root: self.eps_root,
            weight_decay: Some(self.weight_decay),
            lr_schedule: None,
        };

        // Resolve the schedule here: the delegate has no schedule of its own.
        let mut delegate = adam;
        if let Some(ref schedule) = self.lr_schedule {
            delegate.learning_rate = schedule.get_lr(state.step + 1);
        }

        delegate.update(gradients, state, params)
    }

    fn name(&self) -> &str {
        "adamw"
    }
}

/// JAX-compatible SGD optimizer
pub struct JAXSGD {
    inner: SGD,
    learning_rate: f64,
    momentum: f64,
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
        if self.momentum > 0.0 {
            Ok(init_moments(params, &["_momentum"]))
        } else {
            Ok(init_moments(params, &[]))
        }
    }

    /// Applies one Optax `sgd` step.
    ///
    /// ```text
    /// g'   = g + weight_decay * p
    /// buf  = momentum * buf + g'                 (only when momentum > 0)
    /// d    = g' + momentum * buf   if nesterov
    ///        buf                   otherwise
    /// p_t  = p_{t-1} - lr * d
    /// ```
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

        let weight_decay = self.weight_decay.unwrap_or(0.0);

        if let Some(params) = params {
            for (name, param) in params {
                let Some(grad) = gradients.get(name) else {
                    updated_params.insert(name.clone(), param.clone());
                    continue;
                };

                let param_data = param.data_f32()?;
                let grad_data = grad.data_f32()?;
                if grad_data.len() != param_data.len() {
                    return Err(trustformers_core::errors::TrustformersError::invalid_input(
                        format!(
                            "gradient for '{name}' has {} elements but the parameter has {}",
                            grad_data.len(),
                            param_data.len()
                        ),
                    ));
                }

                let mut buffer = read_moment(state, &format!("{name}_momentum"), param_data.len());
                let mut updated = Vec::with_capacity(param_data.len());

                for index in 0..param_data.len() {
                    let p = param_data[index] as f64;
                    let g = grad_data[index] as f64 + weight_decay * p;

                    let direction = if self.momentum > 0.0 {
                        let buf = self.momentum * buffer[index] as f64 + g;
                        buffer[index] = buf as f32;
                        if self.nesterov {
                            g + self.momentum * buf
                        } else {
                            buf
                        }
                    } else {
                        g
                    };

                    updated.push((p - current_lr * direction) as f32);
                }

                if self.momentum > 0.0 {
                    write_moment(&mut new_state, &format!("{name}_momentum"), &buffer);
                }
                updated_params.insert(name.clone(), Tensor::from_vec(updated, &param.shape())?);
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
        let optimizer = JAXAdam::with_defaults().expect("Operation failed in test");
        assert_eq!(optimizer.name(), "adam");
        assert_eq!(optimizer.learning_rate, 1e-3);
    }

    #[test]
    fn test_jax_adamw_creation() {
        let optimizer = JAXAdamW::with_defaults().expect("Operation failed in test");
        assert_eq!(optimizer.name(), "adamw");
        assert_eq!(optimizer.learning_rate, 1e-3);
        assert_eq!(optimizer.weight_decay, 1e-4);
    }

    #[test]
    fn test_jax_sgd_creation() {
        let optimizer = JAXSGD::with_defaults().expect("Operation failed in test");
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
        let adam = JAXOptimizerFactory::adam(1e-3, 0.9, 0.999, 1e-8, 0.0, None)
            .expect("Operation failed in test");
        assert_eq!(adam.name(), "adam");

        let adamw = JAXOptimizerFactory::adamw(1e-3, 0.9, 0.999, 1e-8, 0.0, 1e-4)
            .expect("Operation failed in test");
        assert_eq!(adamw.name(), "adamw");

        let sgd =
            JAXOptimizerFactory::sgd(1e-3, 0.9, false, None).expect("Operation failed in test");
        assert_eq!(sgd.name(), "sgd");
    }

    #[test]
    fn test_jax_optimizer_init() {
        use std::collections::HashMap;
        let optimizer = JAXAdam::with_defaults().expect("Operation failed in test");
        let params: HashMap<String, Tensor> = [
            (
                "param1".to_string(),
                Tensor::zeros(&[10, 10]).expect("Failed to create tensor"),
            ),
            (
                "param2".to_string(),
                Tensor::zeros(&[5, 5]).expect("Failed to create tensor"),
            ),
        ]
        .iter()
        .cloned()
        .collect();

        let state = optimizer.init(&params).expect("Init failed");
        assert_eq!(state.step, 0);
        assert!(state.params.contains_key("param1_m"));
        assert!(state.params.contains_key("param1_v"));
        assert!(state.params.contains_key("param2_m"));
        assert!(state.params.contains_key("param2_v"));
    }

    #[test]
    fn test_jax_optimizer_update() {
        use std::collections::HashMap;
        let optimizer = JAXAdam::with_defaults().expect("Operation failed in test");
        let params: HashMap<String, Tensor> = [(
            "param1".to_string(),
            Tensor::zeros(&[10, 10]).expect("Failed to create tensor"),
        )]
        .iter()
        .cloned()
        .collect();

        let state = optimizer.init(&params).expect("Init failed");

        let gradients: HashMap<String, Tensor> = [(
            "param1".to_string(),
            Tensor::ones(&[10, 10]).expect("Failed to create tensor"),
        )]
        .iter()
        .cloned()
        .collect();

        let (updated_params, updated_state) = optimizer
            .update(&gradients, &state, Some(&params))
            .expect("Optimizer update failed");
        assert_eq!(updated_state.step, 1);
        assert!(updated_params.contains_key("param1"));
    }

    #[test]
    fn test_jax_chain_transformation() {
        use std::collections::HashMap;
        let adam = JAXOptimizerFactory::adam(1e-3, 0.9, 0.999, 1e-8, 0.0, None)
            .expect("Operation failed in test");
        let sgd =
            JAXOptimizerFactory::sgd(1e-3, 0.9, false, None).expect("Operation failed in test");

        let chain = JAXOptimizerFactory::chain(vec![Box::new(adam), Box::new(sgd)]);

        assert_eq!(chain.name(), "chain");

        let params: HashMap<String, Tensor> = [(
            "param1".to_string(),
            Tensor::zeros(&[5, 5]).expect("Failed to create tensor"),
        )]
        .iter()
        .cloned()
        .collect();

        let state = chain.init(&params).expect("Init failed");
        assert!(state.params.contains_key("chain_0"));
        assert!(state.params.contains_key("chain_1"));
    }

    #[test]
    fn test_schedule_with_optimizer() {
        let schedule = Box::new(JAXExponentialDecay::new(0.1, 0.96, 100, 0, false, None));
        let optimizer = JAXAdam::with_schedule(schedule, 0.9, 0.999, 1e-8, 0.0, None)
            .expect("Operation failed in test");

        assert_eq!(optimizer.learning_rate, 0.1);
        assert!(optimizer.lr_schedule.is_some());
    }

    fn single(name: &str, values: &[f32]) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(
            name.to_string(),
            Tensor::from_vec(values.to_vec(), &[values.len()]).expect("tensor"),
        );
        map
    }

    /// Regression: `update` used to drop every tensor-op result and return the
    /// parameter completely unchanged.
    #[test]
    fn jax_adam_moves_the_parameter() {
        let optimizer = JAXAdam::from_params(0.1, 0.9, 0.999, 1e-8, 0.0, None).expect("adam");
        let params = single("w", &[1.0, -2.0, 3.0]);
        let grads = single("w", &[2.0, 2.0, 2.0]);
        let state = optimizer.init(&params).expect("init");

        let (updated, _) = optimizer.update(&grads, &state, Some(&params)).expect("update");
        let after = updated["w"].data_f32().expect("data");
        let before = params["w"].data_f32().expect("data");
        for (a, b) in after.iter().zip(before.iter()) {
            assert!((a - b).abs() > 1e-6, "parameter must move: {a} vs {b}");
        }
    }

    /// Exact hand-computed first Adam step.
    ///
    /// `m = 0.1 * 2 = 0.2`, `v = 0.001 * 4 = 0.004`, `m̂ = 2.0`, `v̂ = 4.0`,
    /// `Δ = 0.1 * 2 / (2 + 1e-8) = 0.1`, so `1.0 - 0.1 = 0.9`.
    #[test]
    fn jax_adam_first_step_matches_hand_computation() {
        let optimizer = JAXAdam::from_params(0.1, 0.9, 0.999, 1e-8, 0.0, None).expect("adam");
        let params = single("w", &[1.0]);
        let grads = single("w", &[2.0]);
        let state = optimizer.init(&params).expect("init");

        let (updated, new_state) = optimizer.update(&grads, &state, Some(&params)).expect("update");
        assert_eq!(new_state.step, 1);
        let value = updated["w"].data_f32().expect("data")[0];
        assert!((value - 0.9).abs() < 1e-5, "expected 0.9, got {value}");
    }

    /// The moments must survive from one functional step to the next.
    #[test]
    fn jax_adam_carries_state_between_steps() {
        let optimizer = JAXAdam::from_params(0.1, 0.9, 0.999, 1e-8, 0.0, None).expect("adam");
        let params = single("w", &[1.0]);
        let grads = single("w", &[2.0]);
        let state = optimizer.init(&params).expect("init");

        let (after_one, state_one) =
            optimizer.update(&grads, &state, Some(&params)).expect("step 1");
        let mu = read_moment(&state_one, "w_m", 1)[0];
        assert!((mu - 0.2).abs() < 1e-6, "m after one step: {mu}");

        // A *zero* gradient on the second step can only move the parameter if the
        // first step's momentum survived. With the old no-op update — and with any
        // implementation that re-zeroes the moments — the second step does nothing.
        let zero = single("w", &[0.0]);
        let (after_two, _) = optimizer.update(&zero, &state_one, Some(&after_one)).expect("step 2");
        let delta2 = after_one["w"].data_f32().expect("data")[0]
            - after_two["w"].data_f32().expect("data")[0];
        assert!(
            delta2 > 1e-3,
            "carried momentum must still drive step 2, moved only {delta2}"
        );
    }

    /// Exact hand-computed heavy-ball SGD steps.
    #[test]
    fn jax_sgd_momentum_matches_hand_computation() {
        let optimizer = JAXSGD::from_params(0.1, 0.9, false, None).expect("sgd");
        let params = single("w", &[1.0]);
        let grads = single("w", &[2.0]);
        let state = optimizer.init(&params).expect("init");

        let (after_one, state_one) =
            optimizer.update(&grads, &state, Some(&params)).expect("step 1");
        let value1 = after_one["w"].data_f32().expect("data")[0];
        assert!((value1 - 0.8).abs() < 1e-6, "expected 0.8, got {value1}");

        // buf = 0.9 * 2 + 2 = 3.8 → 0.8 - 0.38 = 0.42
        let (after_two, _) =
            optimizer.update(&grads, &state_one, Some(&after_one)).expect("step 2");
        let value2 = after_two["w"].data_f32().expect("data")[0];
        assert!((value2 - 0.42).abs() < 1e-5, "expected 0.42, got {value2}");
    }

    /// AdamW's decoupled decay must actually shrink a zero-gradient parameter.
    #[test]
    fn jax_adamw_applies_decoupled_weight_decay() {
        let optimizer = JAXAdamW::from_params(0.1, 0.9, 0.999, 1e-8, 0.0, 0.5).expect("adamw");
        let params = single("w", &[1.0]);
        let grads = single("w", &[0.0]);
        let state = optimizer.init(&params).expect("init");

        let (updated, _) = optimizer.update(&grads, &state, Some(&params)).expect("update");
        let value = updated["w"].data_f32().expect("data")[0];
        // Δ = lr * wd * p = 0.1 * 0.5 * 1.0 = 0.05
        assert!((value - 0.95).abs() < 1e-6, "expected 0.95, got {value}");
    }

    /// Convergence smoke test on the quadratic bowl `f(x) = Σ x²` (`∇f = 2x`).
    #[test]
    fn jax_adam_descends_a_quadratic_bowl() {
        let optimizer = JAXAdam::from_params(0.1, 0.9, 0.999, 1e-8, 0.0, None).expect("adam");
        let mut params = single("w", &[3.0, -4.0]);
        let mut state = optimizer.init(&params).expect("init");

        let loss = |p: &HashMap<String, Tensor>| -> f32 {
            p["w"].data_f32().expect("data").iter().map(|v| v * v).sum()
        };
        let initial = loss(&params);

        for _ in 0..200 {
            let values = params["w"].data_f32().expect("data");
            let grad: Vec<f32> = values.iter().map(|v| 2.0 * v).collect();
            let grads = single("w", &grad);
            let (next, next_state) = optimizer.update(&grads, &state, Some(&params)).expect("step");
            params = next;
            state = next_state;
        }

        let final_loss = loss(&params);
        assert!(
            final_loss < initial * 0.01,
            "loss must decrease: {initial} -> {final_loss}"
        );
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
