//! PyTorch Optimizer API Compatibility Layer
//!
//! This module provides PyTorch-compatible optimizer interfaces for seamless
//! integration with PyTorch-based training workflows. It wraps our native
//! optimizers to provide the familiar PyTorch API while maintaining high performance.

use crate::traits::StatefulOptimizer;
use crate::{Adam, AdamW, LRScheduler, OptimizerState, SGD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use trustformers_core::errors::Result;
use trustformers_core::traits::Optimizer;
use trustformers_core::Tensor;

/// PyTorch-compatible optimizer parameter group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyTorchParamGroup {
    pub params: Vec<String>, // Parameter names/IDs
    pub lr: f64,
    pub weight_decay: f64,
    pub momentum: Option<f64>,
    pub dampening: Option<f64>,
    pub eps: Option<f64>,
    pub betas: Option<(f64, f64)>,
    pub alpha: Option<f64>,
    pub amsgrad: Option<bool>,
    pub maximize: Option<bool>,
    pub foreach: Option<bool>,
    pub differentiable: Option<bool>,
}

impl Default for PyTorchParamGroup {
    fn default() -> Self {
        Self {
            params: Vec::new(),
            lr: 0.001,
            weight_decay: 0.0,
            momentum: None,
            dampening: None,
            eps: Some(1e-8),
            betas: Some((0.9, 0.999)),
            alpha: None,
            amsgrad: Some(false),
            maximize: Some(false),
            foreach: None,
            differentiable: Some(false),
        }
    }
}

/// PyTorch-compatible optimizer state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyTorchOptimizerState {
    pub state: HashMap<String, serde_json::Value>,
    pub param_groups: Vec<PyTorchParamGroup>,
}

/// PyTorch-compatible optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyTorchOptimizerConfig {
    pub optimizer_type: String,
    pub learning_rate: f64,
    pub betas: (f64, f64),
    pub epsilon: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
    pub maximize: bool,
    pub parameters: HashMap<String, serde_json::Value>,
}

impl Default for PyTorchOptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: "Adam".to_string(),
            learning_rate: 1e-3,
            betas: (0.9, 0.999),
            epsilon: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            maximize: false,
            parameters: HashMap::new(),
        }
    }
}

/// PyTorch-compatible optimizer interface
pub trait PyTorchOptimizer: Send + Sync {
    /// Get parameter groups
    fn param_groups(&self) -> &[PyTorchParamGroup];

    /// Get mutable parameter groups
    fn param_groups_mut(&mut self) -> &mut [PyTorchParamGroup];

    /// Get optimizer state
    fn state_dict(&self) -> PyTorchOptimizerState;

    /// Load optimizer state
    fn load_state_dict(&mut self, state: PyTorchOptimizerState) -> Result<()>;

    /// Perform optimization step
    fn step(&mut self, closure: Option<Box<dyn Fn() -> f64>>) -> Result<Option<f64>>;

    /// Zero gradients
    fn zero_grad(&mut self, set_to_none: bool) -> Result<()>;

    /// Add parameter group
    fn add_param_group(&mut self, param_group: PyTorchParamGroup) -> Result<()>;

    /// Get defaults
    fn defaults(&self) -> PyTorchParamGroup;
}

/// PyTorch-compatible Adam optimizer
#[derive(Debug)]
pub struct PyTorchAdam {
    inner: Adam,
    param_groups: Vec<PyTorchParamGroup>,
    parameters: Arc<Mutex<HashMap<String, Tensor>>>,
    gradients: Arc<Mutex<HashMap<String, Tensor>>>,
}

impl PyTorchAdam {
    /// Create new PyTorch-compatible Adam optimizer
    pub fn new(
        params: Vec<PyTorchParamGroup>,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
        _amsgrad: bool,
    ) -> Result<Self> {
        let inner = Adam::new(
            lr as f32,
            (betas.0 as f32, betas.1 as f32),
            eps as f32,
            weight_decay as f32,
        );

        Ok(Self {
            inner,
            param_groups: params,
            parameters: Arc::new(Mutex::new(HashMap::new())),
            gradients: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Create with default parameters
    pub fn from_params(params: impl IntoIterator<Item = (String, Tensor)>) -> Result<Self> {
        let param_group = PyTorchParamGroup {
            params: params.into_iter().map(|(name, _)| name).collect(),
            ..Default::default()
        };

        Self::new(vec![param_group], 0.001, (0.9, 0.999), 1e-8, 0.0, false)
    }

    /// Create PyTorch Adam optimizer from configuration
    pub fn from_config(config: PyTorchOptimizerConfig) -> Result<Self> {
        // Create parameter group from config
        let param_group = PyTorchParamGroup {
            params: config.parameters.keys().cloned().collect(),
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            eps: Some(config.epsilon),
            betas: Some(config.betas),
            amsgrad: Some(config.amsgrad),
            maximize: Some(config.maximize),
            ..Default::default()
        };

        Self::new(
            vec![param_group],
            config.learning_rate,
            config.betas,
            config.epsilon,
            config.weight_decay,
            config.amsgrad,
        )
    }

    /// Create PyTorch Adam optimizer from cross-framework configuration
    pub fn from_cross_framework_config(
        config: crate::cross_framework::PyTorchOptimizerConfig,
    ) -> Result<Self> {
        // Extract parameters from the HashMap
        let betas = if let Some(betas_val) = config.parameters.get("betas") {
            if let Some(arr) = betas_val.as_array() {
                (
                    arr[0].as_f64().unwrap_or(0.9),
                    arr[1].as_f64().unwrap_or(0.999),
                )
            } else {
                (0.9, 0.999)
            }
        } else {
            (0.9, 0.999)
        };

        let epsilon = config.parameters.get("epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-8);

        let weight_decay =
            config.parameters.get("weight_decay").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let amsgrad = config.parameters.get("amsgrad").and_then(|v| v.as_bool()).unwrap_or(false);

        // Create parameter group from config
        let param_group = PyTorchParamGroup {
            params: Vec::new(),
            lr: config.learning_rate as f64,
            weight_decay,
            eps: Some(epsilon),
            betas: Some(betas),
            amsgrad: Some(amsgrad),
            maximize: Some(false),
            ..Default::default()
        };

        Self::new(
            vec![param_group],
            config.learning_rate as f64,
            betas,
            epsilon,
            weight_decay,
            amsgrad,
        )
    }

    /// Register parameter
    pub fn register_param(&mut self, name: String, param: Tensor) -> Result<()> {
        let mut params = self.parameters.lock().unwrap();
        params.insert(name, param);
        Ok(())
    }

    /// Set gradient for parameter
    pub fn set_grad(&mut self, name: String, grad: Tensor) -> Result<()> {
        let mut grads = self.gradients.lock().unwrap();
        grads.insert(name, grad);
        Ok(())
    }

    /// Load optimizer state from OptimizerState format
    ///
    /// Converts OptimizerState (with Vec<f32> values) to PyTorch-compatible format
    fn load_optimizer_state(&mut self, optimizer_state: OptimizerState) -> Result<()> {
        // Convert momentum buffers from Vec<f32> to Tensor format
        for (param_name, momentum_data) in optimizer_state.momentum {
            let momentum_tensor = Tensor::new(momentum_data)?;
            // Store in a format that the inner optimizer can use
            // For now, we'll store it directly but in a real implementation,
            // this would integrate with the inner optimizer's state management

            // The inner optimizer would typically have its own state management
            // Here we just ensure the data is available for parameter updates
            let mut params = self.parameters.lock().unwrap();
            if !params.contains_key(&param_name) {
                // Create placeholder parameter if it doesn't exist
                params.insert(param_name.clone(), momentum_tensor.clone());
            }
        }

        // Convert variance buffers from Vec<f32> to Tensor format (for Adam-like optimizers)
        for (param_name, variance_data) in optimizer_state.variance {
            let variance_tensor = Tensor::new(variance_data)?;
            // Similar to momentum, store variance information
            // The inner Adam optimizer would use this for second moment estimation

            // For now, we ensure the parameter exists in our registry
            let mut params = self.parameters.lock().unwrap();
            if !params.contains_key(&param_name) {
                params.insert(param_name.clone(), variance_tensor.clone());
            }
        }

        // Update step counter if available
        // The inner optimizer should sync with this step count for bias correction
        // self.inner would typically have a method to set the step count

        Ok(())
    }
}

impl PyTorchOptimizer for PyTorchAdam {
    fn param_groups(&self) -> &[PyTorchParamGroup] {
        &self.param_groups
    }

    fn param_groups_mut(&mut self) -> &mut [PyTorchParamGroup] {
        &mut self.param_groups
    }

    fn state_dict(&self) -> PyTorchOptimizerState {
        let state = self.inner.state();
        let state_json = serde_json::to_value(state).unwrap_or_default();

        PyTorchOptimizerState {
            state: [(String::from("adam_state"), state_json)].into(),
            param_groups: self.param_groups.clone(),
        }
    }

    fn load_state_dict(&mut self, state: PyTorchOptimizerState) -> Result<()> {
        self.param_groups = state.param_groups;

        if let Some(adam_state) = state.state.get("adam_state") {
            if let Ok(optimizer_state) =
                serde_json::from_value::<OptimizerState>(adam_state.clone())
            {
                // Convert OptimizerState to PyTorch-compatible format
                self.load_optimizer_state(optimizer_state)?;
            }
        }

        Ok(())
    }

    fn step(&mut self, closure: Option<Box<dyn Fn() -> f64>>) -> Result<Option<f64>> {
        let loss = closure.map(|closure_fn| closure_fn());

        // Apply gradients to parameters using the inner optimizer
        for group in &self.param_groups {
            for param_name in &group.params {
                // Get copies of parameter and gradient to avoid borrow conflicts
                let param_copy = {
                    let params = self.parameters.lock().unwrap();
                    params.get(param_name).cloned()
                };
                let grad_copy = {
                    let grads = self.gradients.lock().unwrap();
                    grads.get(param_name).cloned()
                };

                if let (Some(mut param), Some(grad)) = (param_copy, grad_copy) {
                    // Apply update to parameter copy
                    self.inner.update(&mut param, &grad)?;

                    // Store updated parameter back
                    let mut params = self.parameters.lock().unwrap();
                    params.insert(param_name.clone(), param);
                }
            }
        }

        Ok(loss)
    }

    fn zero_grad(&mut self, _set_to_none: bool) -> Result<()> {
        let mut grads = self.gradients.lock().unwrap();
        grads.clear();
        Ok(())
    }

    fn add_param_group(&mut self, param_group: PyTorchParamGroup) -> Result<()> {
        self.param_groups.push(param_group);
        Ok(())
    }

    fn defaults(&self) -> PyTorchParamGroup {
        PyTorchParamGroup {
            lr: 0.001,
            betas: Some((0.9, 0.999)),
            eps: Some(1e-8),
            weight_decay: 0.0,
            amsgrad: Some(false),
            ..Default::default()
        }
    }
}

/// PyTorch-compatible AdamW optimizer
#[derive(Debug)]
pub struct PyTorchAdamW {
    inner: AdamW,
    param_groups: Vec<PyTorchParamGroup>,
    parameters: Arc<Mutex<HashMap<String, Tensor>>>,
    gradients: Arc<Mutex<HashMap<String, Tensor>>>,
}

impl PyTorchAdamW {
    /// Create new PyTorch-compatible AdamW optimizer
    pub fn new(
        params: Vec<PyTorchParamGroup>,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
        _amsgrad: bool,
    ) -> Result<Self> {
        let inner = AdamW::new(
            lr as f32,
            (betas.0 as f32, betas.1 as f32),
            eps as f32,
            weight_decay as f32,
        );

        Ok(Self {
            inner,
            param_groups: params,
            parameters: Arc::new(Mutex::new(HashMap::new())),
            gradients: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Create with default parameters
    pub fn from_params(params: impl IntoIterator<Item = (String, Tensor)>) -> Result<Self> {
        let param_group = PyTorchParamGroup {
            params: params.into_iter().map(|(name, _)| name).collect(),
            ..Default::default()
        };

        Self::new(vec![param_group], 0.001, (0.9, 0.999), 1e-8, 0.01, false)
    }

    /// Register parameter
    pub fn register_param(&mut self, name: String, param: Tensor) -> Result<()> {
        let mut params = self.parameters.lock().unwrap();
        params.insert(name, param);
        Ok(())
    }

    /// Set gradient for parameter
    pub fn set_grad(&mut self, name: String, grad: Tensor) -> Result<()> {
        let mut grads = self.gradients.lock().unwrap();
        grads.insert(name, grad);
        Ok(())
    }

    /// Load optimizer state from OptimizerState format
    ///
    /// Converts OptimizerState (with Vec<f32> values) to PyTorch-compatible format
    fn load_optimizer_state(&mut self, optimizer_state: OptimizerState) -> Result<()> {
        // Convert momentum buffers from Vec<f32> to Tensor format
        for (param_name, momentum_data) in optimizer_state.momentum {
            let momentum_tensor = Tensor::new(momentum_data)?;
            let mut params = self.parameters.lock().unwrap();
            if !params.contains_key(&param_name) {
                params.insert(param_name.clone(), momentum_tensor.clone());
            }
        }

        // Convert variance buffers from Vec<f32> to Tensor format (for AdamW)
        for (param_name, variance_data) in optimizer_state.variance {
            let variance_tensor = Tensor::new(variance_data)?;
            let mut params = self.parameters.lock().unwrap();
            if !params.contains_key(&param_name) {
                params.insert(param_name.clone(), variance_tensor.clone());
            }
        }

        Ok(())
    }
}

impl PyTorchOptimizer for PyTorchAdamW {
    fn param_groups(&self) -> &[PyTorchParamGroup] {
        &self.param_groups
    }

    fn param_groups_mut(&mut self) -> &mut [PyTorchParamGroup] {
        &mut self.param_groups
    }

    fn state_dict(&self) -> PyTorchOptimizerState {
        let state = self.inner.state();
        let state_json = serde_json::to_value(state).unwrap_or_default();

        PyTorchOptimizerState {
            state: [(String::from("adamw_state"), state_json)].into(),
            param_groups: self.param_groups.clone(),
        }
    }

    fn load_state_dict(&mut self, state: PyTorchOptimizerState) -> Result<()> {
        self.param_groups = state.param_groups;

        if let Some(adamw_state) = state.state.get("adamw_state") {
            if let Ok(optimizer_state) =
                serde_json::from_value::<OptimizerState>(adamw_state.clone())
            {
                // Convert OptimizerState to PyTorch-compatible format
                self.load_optimizer_state(optimizer_state)?;
            }
        }

        Ok(())
    }

    fn step(&mut self, closure: Option<Box<dyn Fn() -> f64>>) -> Result<Option<f64>> {
        let loss = closure.map(|closure_fn| closure_fn());

        for group in &self.param_groups {
            for param_name in &group.params {
                // Get copies of parameter and gradient to avoid borrow conflicts
                let param_copy = {
                    let params = self.parameters.lock().unwrap();
                    params.get(param_name).cloned()
                };
                let grad_copy = {
                    let grads = self.gradients.lock().unwrap();
                    grads.get(param_name).cloned()
                };

                if let (Some(mut param), Some(grad)) = (param_copy, grad_copy) {
                    // Apply update to parameter copy
                    self.inner.update(&mut param, &grad)?;

                    // Store updated parameter back
                    let mut params = self.parameters.lock().unwrap();
                    params.insert(param_name.clone(), param);
                }
            }
        }

        Ok(loss)
    }

    fn zero_grad(&mut self, _set_to_none: bool) -> Result<()> {
        let mut grads = self.gradients.lock().unwrap();
        grads.clear();
        Ok(())
    }

    fn add_param_group(&mut self, param_group: PyTorchParamGroup) -> Result<()> {
        self.param_groups.push(param_group);
        Ok(())
    }

    fn defaults(&self) -> PyTorchParamGroup {
        PyTorchParamGroup {
            lr: 0.001,
            betas: Some((0.9, 0.999)),
            eps: Some(1e-8),
            weight_decay: 0.01,
            amsgrad: Some(false),
            ..Default::default()
        }
    }
}

/// PyTorch-compatible SGD optimizer
#[derive(Debug)]
pub struct PyTorchSGD {
    inner: SGD,
    param_groups: Vec<PyTorchParamGroup>,
    parameters: Arc<Mutex<HashMap<String, Tensor>>>,
    gradients: Arc<Mutex<HashMap<String, Tensor>>>,
}

impl PyTorchSGD {
    /// Create new PyTorch-compatible SGD optimizer
    pub fn new(
        params: Vec<PyTorchParamGroup>,
        lr: f64,
        momentum: f64,
        dampening: f64,
        weight_decay: f64,
        nesterov: bool,
    ) -> Result<Self> {
        let config = crate::sgd::SGDConfig {
            lr: lr as f32,
            momentum: momentum as f32,
            dampening: dampening as f32,
            weight_decay: weight_decay as f32,
            nesterov,
        };

        let inner = SGD::from_config(config);

        Ok(Self {
            inner,
            param_groups: params,
            parameters: Arc::new(Mutex::new(HashMap::new())),
            gradients: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Create with default parameters
    pub fn from_params(params: impl IntoIterator<Item = (String, Tensor)>) -> Result<Self> {
        let param_group = PyTorchParamGroup {
            params: params.into_iter().map(|(name, _)| name).collect(),
            lr: 0.01,
            momentum: Some(0.0),
            dampening: Some(0.0),
            weight_decay: 0.0,
            ..Default::default()
        };

        Self::new(vec![param_group], 0.01, 0.0, 0.0, 0.0, false)
    }

    /// Register parameter
    pub fn register_param(&mut self, name: String, param: Tensor) -> Result<()> {
        let mut params = self.parameters.lock().unwrap();
        params.insert(name, param);
        Ok(())
    }

    /// Set gradient for parameter
    pub fn set_grad(&mut self, name: String, grad: Tensor) -> Result<()> {
        let mut grads = self.gradients.lock().unwrap();
        grads.insert(name, grad);
        Ok(())
    }

    /// Load optimizer state from OptimizerState format
    ///
    /// Converts OptimizerState (with Vec<f32> values) to PyTorch-compatible format
    fn load_optimizer_state(&mut self, optimizer_state: OptimizerState) -> Result<()> {
        // Convert momentum buffers from Vec<f32> to Tensor format (SGD momentum)
        for (param_name, momentum_data) in optimizer_state.momentum {
            let momentum_tensor = Tensor::new(momentum_data)?;
            let mut params = self.parameters.lock().unwrap();
            if !params.contains_key(&param_name) {
                params.insert(param_name.clone(), momentum_tensor.clone());
            }
        }

        // SGD typically doesn't use variance buffers, but handle them if present
        for (param_name, variance_data) in optimizer_state.variance {
            let variance_tensor = Tensor::new(variance_data)?;
            let mut params = self.parameters.lock().unwrap();
            if !params.contains_key(&param_name) {
                params.insert(param_name.clone(), variance_tensor.clone());
            }
        }

        Ok(())
    }
}

impl PyTorchOptimizer for PyTorchSGD {
    fn param_groups(&self) -> &[PyTorchParamGroup] {
        &self.param_groups
    }

    fn param_groups_mut(&mut self) -> &mut [PyTorchParamGroup] {
        &mut self.param_groups
    }

    fn state_dict(&self) -> PyTorchOptimizerState {
        let state = self.inner.state();
        let state_json = serde_json::to_value(state).unwrap_or_default();

        PyTorchOptimizerState {
            state: [(String::from("sgd_state"), state_json)].into(),
            param_groups: self.param_groups.clone(),
        }
    }

    fn load_state_dict(&mut self, state: PyTorchOptimizerState) -> Result<()> {
        self.param_groups = state.param_groups;

        if let Some(sgd_state) = state.state.get("sgd_state") {
            if let Ok(optimizer_state) = serde_json::from_value::<OptimizerState>(sgd_state.clone())
            {
                // Convert OptimizerState to PyTorch-compatible format
                self.load_optimizer_state(optimizer_state)?;
            }
        }

        Ok(())
    }

    fn step(&mut self, closure: Option<Box<dyn Fn() -> f64>>) -> Result<Option<f64>> {
        let loss = closure.map(|closure_fn| closure_fn());

        for group in &self.param_groups {
            for param_name in &group.params {
                // Get copies of parameter and gradient to avoid borrow conflicts
                let param_copy = {
                    let params = self.parameters.lock().unwrap();
                    params.get(param_name).cloned()
                };
                let grad_copy = {
                    let grads = self.gradients.lock().unwrap();
                    grads.get(param_name).cloned()
                };

                if let (Some(mut param), Some(grad)) = (param_copy, grad_copy) {
                    // Apply update to parameter copy
                    self.inner.update(&mut param, &grad)?;

                    // Store updated parameter back
                    let mut params = self.parameters.lock().unwrap();
                    params.insert(param_name.clone(), param);
                }
            }
        }

        Ok(loss)
    }

    fn zero_grad(&mut self, _set_to_none: bool) -> Result<()> {
        let mut grads = self.gradients.lock().unwrap();
        grads.clear();
        Ok(())
    }

    fn add_param_group(&mut self, param_group: PyTorchParamGroup) -> Result<()> {
        self.param_groups.push(param_group);
        Ok(())
    }

    fn defaults(&self) -> PyTorchParamGroup {
        PyTorchParamGroup {
            lr: 0.01,
            momentum: Some(0.0),
            dampening: Some(0.0),
            weight_decay: 0.0,
            ..Default::default()
        }
    }
}

/// PyTorch optimizer factory for creating optimizers with PyTorch-compatible API
pub struct PyTorchOptimizerFactory;

impl PyTorchOptimizerFactory {
    /// Create Adam optimizer with PyTorch API
    pub fn adam(
        params: impl IntoIterator<Item = (String, Tensor)>,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
        amsgrad: bool,
    ) -> Result<PyTorchAdam> {
        let param_group = PyTorchParamGroup {
            params: params.into_iter().map(|(name, _)| name).collect(),
            lr,
            betas: Some(betas),
            eps: Some(eps),
            weight_decay,
            amsgrad: Some(amsgrad),
            ..Default::default()
        };

        PyTorchAdam::new(vec![param_group], lr, betas, eps, weight_decay, amsgrad)
    }

    /// Create AdamW optimizer with PyTorch API
    pub fn adamw(
        params: impl IntoIterator<Item = (String, Tensor)>,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
        amsgrad: bool,
    ) -> Result<PyTorchAdamW> {
        let param_group = PyTorchParamGroup {
            params: params.into_iter().map(|(name, _)| name).collect(),
            lr,
            betas: Some(betas),
            eps: Some(eps),
            weight_decay,
            amsgrad: Some(amsgrad),
            ..Default::default()
        };

        PyTorchAdamW::new(vec![param_group], lr, betas, eps, weight_decay, amsgrad)
    }

    /// Create SGD optimizer with PyTorch API
    pub fn sgd(
        params: impl IntoIterator<Item = (String, Tensor)>,
        lr: f64,
        momentum: f64,
        dampening: f64,
        weight_decay: f64,
        nesterov: bool,
    ) -> Result<PyTorchSGD> {
        let param_group = PyTorchParamGroup {
            params: params.into_iter().map(|(name, _)| name).collect(),
            lr,
            momentum: Some(momentum),
            dampening: Some(dampening),
            weight_decay,
            ..Default::default()
        };

        PyTorchSGD::new(
            vec![param_group],
            lr,
            momentum,
            dampening,
            weight_decay,
            nesterov,
        )
    }
}

/// PyTorch-compatible learning rate scheduler wrapper
pub struct PyTorchLRScheduler {
    inner_scheduler: Box<dyn LRScheduler>,
    optimizer: Box<dyn PyTorchOptimizer>,
    last_epoch: i64,
}

impl PyTorchLRScheduler {
    /// Create new scheduler wrapper
    pub fn new(optimizer: Box<dyn PyTorchOptimizer>, scheduler: Box<dyn LRScheduler>) -> Self {
        Self {
            inner_scheduler: scheduler,
            optimizer,
            last_epoch: -1,
        }
    }

    /// Step the scheduler
    pub fn step(&mut self, epoch: Option<i64>) -> Result<()> {
        let current_epoch = epoch.unwrap_or(self.last_epoch + 1);
        self.last_epoch = current_epoch;

        let new_lr = self.inner_scheduler.get_lr(current_epoch as usize);

        // Update all parameter groups
        for group in self.optimizer.param_groups_mut() {
            group.lr = new_lr as f64;
        }

        Ok(())
    }

    /// Get current learning rate
    pub fn get_last_lr(&self) -> f64 {
        self.inner_scheduler.get_lr(self.last_epoch.max(0) as usize) as f64
    }

    /// Get current state dict
    pub fn state_dict(&self) -> serde_json::Value {
        serde_json::json!({
            "last_epoch": self.last_epoch,
            "scheduler_state": "serialized_state" // Would need scheduler serialization
        })
    }

    /// Load state dict
    pub fn load_state_dict(&mut self, state: serde_json::Value) -> Result<()> {
        if let Some(epoch) = state.get("last_epoch").and_then(|e| e.as_i64()) {
            self.last_epoch = epoch;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::Tensor;

    #[test]
    fn test_pytorch_adam_creation() {
        let params = vec![
            ("param1".to_string(), Tensor::zeros(&[10, 10]).unwrap()),
            ("param2".to_string(), Tensor::zeros(&[5, 5]).unwrap()),
        ];

        let optimizer = PyTorchAdam::from_params(params).unwrap();
        assert_eq!(optimizer.param_groups().len(), 1);
        assert_eq!(optimizer.param_groups()[0].params.len(), 2);
    }

    #[test]
    fn test_pytorch_adamw_creation() {
        let params = vec![("param1".to_string(), Tensor::zeros(&[10, 10]).unwrap())];

        let optimizer = PyTorchAdamW::from_params(params).unwrap();
        assert_eq!(optimizer.param_groups().len(), 1);
        assert_eq!(optimizer.defaults().weight_decay, 0.01);
    }

    #[test]
    fn test_pytorch_sgd_creation() {
        let params = vec![("param1".to_string(), Tensor::zeros(&[10, 10]).unwrap())];

        let optimizer = PyTorchSGD::from_params(params).unwrap();
        assert_eq!(optimizer.param_groups().len(), 1);
        assert_eq!(optimizer.defaults().lr, 0.01);
    }

    #[test]
    fn test_pytorch_optimizer_factory() {
        let params = vec![("param1".to_string(), Tensor::zeros(&[10, 10]).unwrap())];

        let adam =
            PyTorchOptimizerFactory::adam(params.clone(), 0.001, (0.9, 0.999), 1e-8, 0.0, false)
                .unwrap();
        assert_eq!(adam.param_groups()[0].lr, 0.001);

        let adamw =
            PyTorchOptimizerFactory::adamw(params.clone(), 0.001, (0.9, 0.999), 1e-8, 0.01, false)
                .unwrap();
        assert_eq!(adamw.param_groups()[0].weight_decay, 0.01);

        let sgd = PyTorchOptimizerFactory::sgd(params, 0.01, 0.9, 0.0, 0.0, false).unwrap();
        assert_eq!(sgd.param_groups()[0].momentum, Some(0.9));
    }

    #[test]
    fn test_param_group_operations() {
        let params = vec![("param1".to_string(), Tensor::zeros(&[10, 10]).unwrap())];

        let mut optimizer = PyTorchAdam::from_params(params).unwrap();

        let new_group = PyTorchParamGroup {
            params: vec!["param2".to_string()],
            lr: 0.002,
            ..Default::default()
        };

        optimizer.add_param_group(new_group).unwrap();
        assert_eq!(optimizer.param_groups().len(), 2);
        assert_eq!(optimizer.param_groups()[1].lr, 0.002);
    }

    #[test]
    fn test_state_dict_operations() {
        let params = vec![("param1".to_string(), Tensor::zeros(&[10, 10]).unwrap())];

        let optimizer = PyTorchAdam::from_params(params).unwrap();
        let state_dict = optimizer.state_dict();

        assert_eq!(state_dict.param_groups.len(), 1);
        assert!(state_dict.state.contains_key("adam_state"));
    }

    #[test]
    fn test_zero_grad() {
        let params = vec![("param1".to_string(), Tensor::zeros(&[10, 10]).unwrap())];

        let mut optimizer = PyTorchAdam::from_params(params).unwrap();
        optimizer
            .set_grad("param1".to_string(), Tensor::ones(&[10, 10]).unwrap())
            .unwrap();

        // Check that gradient is set
        assert_eq!(optimizer.gradients.lock().unwrap().len(), 1);

        // Zero gradients
        optimizer.zero_grad(false).unwrap();
        assert_eq!(optimizer.gradients.lock().unwrap().len(), 0);
    }
}
