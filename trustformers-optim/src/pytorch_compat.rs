//! PyTorch Optimizer API Compatibility Layer
//!
//! This module provides PyTorch-compatible optimizer interfaces for seamless
//! integration with PyTorch-based training workflows. It wraps our native
//! optimizers to provide the familiar PyTorch API while maintaining high performance.

use crate::traits::StatefulOptimizer;
use crate::{Adam, AdamW, LRScheduler, SGD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use trustformers_core::errors::{Result, TrustformersError};
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
    /// Serialises the optimizer state.
    ///
    /// # Errors
    ///
    /// Returns an error when the inner optimizer's state cannot be serialised;
    /// silently emitting an empty state would make a checkpoint look valid.
    fn state_dict(&self) -> Result<PyTorchOptimizerState>;

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

/// JSON key under which a tensor's logical shape is stored.
const TENSOR_SHAPE_KEY: &str = "shape";
/// JSON key under which a tensor's flattened `f32` payload is stored.
const TENSOR_DATA_KEY: &str = "data";

/// Encodes a [`StatefulOptimizer`] tensor state dict as JSON.
///
/// Each entry becomes `{"shape": [...], "data": [...]}` so the round trip is lossless
/// for `f32` payloads and, crucially, carries the *real* optimizer moments rather than
/// a summary of them.
///
/// # Errors
///
/// Returns an error when a tensor cannot be read as `f32` or a value is not
/// representable in JSON (NaN / infinity).
fn encode_tensor_state(state: &HashMap<String, Tensor>) -> Result<serde_json::Value> {
    let mut encoded = serde_json::Map::new();
    for (name, tensor) in state {
        let data = tensor.data_f32()?;
        let mut numbers = Vec::with_capacity(data.len());
        for value in &data {
            let number = serde_json::Number::from_f64(*value as f64).ok_or_else(|| {
                TrustformersError::invalid_input(format!(
                    "optimizer state entry '{name}' contains {value}, which JSON cannot represent"
                ))
            })?;
            numbers.push(serde_json::Value::Number(number));
        }

        let mut entry = serde_json::Map::new();
        entry.insert(
            TENSOR_SHAPE_KEY.to_string(),
            serde_json::json!(tensor.shape()),
        );
        entry.insert(
            TENSOR_DATA_KEY.to_string(),
            serde_json::Value::Array(numbers),
        );
        encoded.insert(name.clone(), serde_json::Value::Object(entry));
    }
    Ok(serde_json::Value::Object(encoded))
}

/// Decodes the JSON produced by [`encode_tensor_state`].
///
/// # Errors
///
/// Returns an error for any structural problem — a missing shape, a payload whose
/// length disagrees with the shape, or a non-object value. A malformed checkpoint must
/// never load "successfully" with nothing restored.
fn decode_tensor_state(value: &serde_json::Value) -> Result<HashMap<String, Tensor>> {
    let object = value.as_object().ok_or_else(|| {
        TrustformersError::invalid_input(
            "optimizer state must be a JSON object of tensor entries".to_string(),
        )
    })?;

    let mut decoded = HashMap::new();
    for (name, entry) in object {
        let entry = entry.as_object().ok_or_else(|| {
            TrustformersError::invalid_input(format!(
                "optimizer state entry '{name}' is not an object"
            ))
        })?;

        let shape: Vec<usize> = entry
            .get(TENSOR_SHAPE_KEY)
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                TrustformersError::invalid_input(format!(
                    "optimizer state entry '{name}' has no '{TENSOR_SHAPE_KEY}' array"
                ))
            })?
            .iter()
            .map(|v| {
                v.as_u64().map(|n| n as usize).ok_or_else(|| {
                    TrustformersError::invalid_input(format!(
                        "optimizer state entry '{name}' has a non-integer dimension"
                    ))
                })
            })
            .collect::<Result<Vec<usize>>>()?;

        let data: Vec<f32> = entry
            .get(TENSOR_DATA_KEY)
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                TrustformersError::invalid_input(format!(
                    "optimizer state entry '{name}' has no '{TENSOR_DATA_KEY}' array"
                ))
            })?
            .iter()
            .map(|v| {
                v.as_f64().map(|n| n as f32).ok_or_else(|| {
                    TrustformersError::invalid_input(format!(
                        "optimizer state entry '{name}' has a non-numeric element"
                    ))
                })
            })
            .collect::<Result<Vec<f32>>>()?;

        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(TrustformersError::invalid_input(format!(
                "optimizer state entry '{name}' has {} elements but shape {shape:?} needs {expected}",
                data.len()
            )));
        }

        decoded.insert(name.clone(), Tensor::from_vec(data, &shape)?);
    }

    Ok(decoded)
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
        let mut params = self
            .parameters
            .lock()
            .map_err(|_| TrustformersError::runtime_error("Mutex lock poisoned".into()))?;
        params.insert(name, param);
        Ok(())
    }

    /// Set gradient for parameter
    pub fn set_grad(&mut self, name: String, grad: Tensor) -> Result<()> {
        let mut grads = self
            .gradients
            .lock()
            .map_err(|_| TrustformersError::runtime_error("Mutex lock poisoned".into()))?;
        grads.insert(name, grad);
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

    fn state_dict(&self) -> Result<PyTorchOptimizerState> {
        // Route through the inner optimizer's own checkpoint format so the moment
        // buffers really are written out.
        let inner_state = StatefulOptimizer::state_dict(&self.inner)?;
        Ok(PyTorchOptimizerState {
            state: [(
                String::from("adam_state"),
                encode_tensor_state(&inner_state)?,
            )]
            .into(),
            param_groups: self.param_groups.clone(),
        })
    }

    fn load_state_dict(&mut self, state: PyTorchOptimizerState) -> Result<()> {
        self.param_groups = state.param_groups;

        let raw = state.state.get("adam_state").ok_or_else(|| {
            TrustformersError::invalid_input("checkpoint has no 'adam_state' entry".to_string())
        })?;
        let decoded = decode_tensor_state(raw)?;
        StatefulOptimizer::load_state_dict(&mut self.inner, decoded)?;
        Ok(())
    }

    fn step(&mut self, closure: Option<Box<dyn Fn() -> f64>>) -> Result<Option<f64>> {
        let loss = closure.map(|closure_fn| closure_fn());

        // Apply gradients to parameters using the inner optimizer
        for group in &self.param_groups {
            for param_name in &group.params {
                // Get copies of parameter and gradient to avoid borrow conflicts
                let param_copy = {
                    let params = self.parameters.lock().map_err(|_| {
                        TrustformersError::runtime_error("Mutex lock poisoned".into())
                    })?;
                    params.get(param_name).cloned()
                };
                let grad_copy = {
                    let grads = self.gradients.lock().map_err(|_| {
                        TrustformersError::runtime_error("Mutex lock poisoned".into())
                    })?;
                    grads.get(param_name).cloned()
                };

                if let (Some(mut param), Some(grad)) = (param_copy, grad_copy) {
                    // Use the *named* identity: this API already carries parameter
                    // names, and the tensor is cloned out of the registry on every
                    // step, so an address-derived key would allocate a fresh state
                    // slot each time.
                    self.inner.update_named(param_name, &mut param, &grad)?;

                    // Store updated parameter back
                    let mut params = self.parameters.lock().map_err(|_| {
                        TrustformersError::runtime_error("Mutex lock poisoned".into())
                    })?;
                    params.insert(param_name.clone(), param);
                }
            }
        }

        // Advance the inner optimizer's global step so bias correction progresses.
        Optimizer::step(&mut self.inner);

        Ok(loss)
    }

    fn zero_grad(&mut self, _set_to_none: bool) -> Result<()> {
        let mut grads = self
            .gradients
            .lock()
            .map_err(|_| TrustformersError::runtime_error("Mutex lock poisoned".into()))?;
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
        let mut params = self
            .parameters
            .lock()
            .map_err(|_| TrustformersError::runtime_error("Mutex lock poisoned".into()))?;
        params.insert(name, param);
        Ok(())
    }

    /// Set gradient for parameter
    pub fn set_grad(&mut self, name: String, grad: Tensor) -> Result<()> {
        let mut grads = self
            .gradients
            .lock()
            .map_err(|_| TrustformersError::runtime_error("Mutex lock poisoned".into()))?;
        grads.insert(name, grad);
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

    fn state_dict(&self) -> Result<PyTorchOptimizerState> {
        // Route through the inner optimizer's own checkpoint format so the moment
        // buffers really are written out.
        let inner_state = StatefulOptimizer::state_dict(&self.inner)?;
        Ok(PyTorchOptimizerState {
            state: [(
                String::from("adamw_state"),
                encode_tensor_state(&inner_state)?,
            )]
            .into(),
            param_groups: self.param_groups.clone(),
        })
    }

    fn load_state_dict(&mut self, state: PyTorchOptimizerState) -> Result<()> {
        self.param_groups = state.param_groups;

        let raw = state.state.get("adamw_state").ok_or_else(|| {
            TrustformersError::invalid_input("checkpoint has no 'adamw_state' entry".to_string())
        })?;
        let decoded = decode_tensor_state(raw)?;
        StatefulOptimizer::load_state_dict(&mut self.inner, decoded)?;
        Ok(())
    }

    fn step(&mut self, closure: Option<Box<dyn Fn() -> f64>>) -> Result<Option<f64>> {
        let loss = closure.map(|closure_fn| closure_fn());

        for group in &self.param_groups {
            for param_name in &group.params {
                // Get copies of parameter and gradient to avoid borrow conflicts
                let param_copy = {
                    let params = self.parameters.lock().map_err(|_| {
                        TrustformersError::runtime_error("Mutex lock poisoned".into())
                    })?;
                    params.get(param_name).cloned()
                };
                let grad_copy = {
                    let grads = self.gradients.lock().map_err(|_| {
                        TrustformersError::runtime_error("Mutex lock poisoned".into())
                    })?;
                    grads.get(param_name).cloned()
                };

                if let (Some(mut param), Some(grad)) = (param_copy, grad_copy) {
                    // Use the *named* identity: this API already carries parameter
                    // names, and the tensor is cloned out of the registry on every
                    // step, so an address-derived key would allocate a fresh state
                    // slot each time.
                    self.inner.update_named(param_name, &mut param, &grad)?;

                    // Store updated parameter back
                    let mut params = self.parameters.lock().map_err(|_| {
                        TrustformersError::runtime_error("Mutex lock poisoned".into())
                    })?;
                    params.insert(param_name.clone(), param);
                }
            }
        }

        // Advance the inner optimizer's global step so bias correction progresses.
        Optimizer::step(&mut self.inner);

        Ok(loss)
    }

    fn zero_grad(&mut self, _set_to_none: bool) -> Result<()> {
        let mut grads = self
            .gradients
            .lock()
            .map_err(|_| TrustformersError::runtime_error("Mutex lock poisoned".into()))?;
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
        let mut params = self
            .parameters
            .lock()
            .map_err(|_| TrustformersError::runtime_error("Mutex lock poisoned".into()))?;
        params.insert(name, param);
        Ok(())
    }

    /// Set gradient for parameter
    pub fn set_grad(&mut self, name: String, grad: Tensor) -> Result<()> {
        let mut grads = self
            .gradients
            .lock()
            .map_err(|_| TrustformersError::runtime_error("Mutex lock poisoned".into()))?;
        grads.insert(name, grad);
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

    fn state_dict(&self) -> Result<PyTorchOptimizerState> {
        // Route through the inner optimizer's own checkpoint format so the moment
        // buffers really are written out.
        let inner_state = StatefulOptimizer::state_dict(&self.inner)?;
        Ok(PyTorchOptimizerState {
            state: [(
                String::from("sgd_state"),
                encode_tensor_state(&inner_state)?,
            )]
            .into(),
            param_groups: self.param_groups.clone(),
        })
    }

    fn load_state_dict(&mut self, state: PyTorchOptimizerState) -> Result<()> {
        self.param_groups = state.param_groups;

        let raw = state.state.get("sgd_state").ok_or_else(|| {
            TrustformersError::invalid_input("checkpoint has no 'sgd_state' entry".to_string())
        })?;
        let decoded = decode_tensor_state(raw)?;
        StatefulOptimizer::load_state_dict(&mut self.inner, decoded)?;
        Ok(())
    }

    fn step(&mut self, closure: Option<Box<dyn Fn() -> f64>>) -> Result<Option<f64>> {
        let loss = closure.map(|closure_fn| closure_fn());

        for group in &self.param_groups {
            for param_name in &group.params {
                // Get copies of parameter and gradient to avoid borrow conflicts
                let param_copy = {
                    let params = self.parameters.lock().map_err(|_| {
                        TrustformersError::runtime_error("Mutex lock poisoned".into())
                    })?;
                    params.get(param_name).cloned()
                };
                let grad_copy = {
                    let grads = self.gradients.lock().map_err(|_| {
                        TrustformersError::runtime_error("Mutex lock poisoned".into())
                    })?;
                    grads.get(param_name).cloned()
                };

                if let (Some(mut param), Some(grad)) = (param_copy, grad_copy) {
                    // Use the *named* identity: this API already carries parameter
                    // names, and the tensor is cloned out of the registry on every
                    // step, so an address-derived key would allocate a fresh state
                    // slot each time.
                    self.inner.update_named(param_name, &mut param, &grad)?;

                    // Store updated parameter back
                    let mut params = self.parameters.lock().map_err(|_| {
                        TrustformersError::runtime_error("Mutex lock poisoned".into())
                    })?;
                    params.insert(param_name.clone(), param);
                }
            }
        }

        // Advance the inner optimizer's global step so bias correction progresses.
        Optimizer::step(&mut self.inner);

        Ok(loss)
    }

    fn zero_grad(&mut self, _set_to_none: bool) -> Result<()> {
        let mut grads = self
            .gradients
            .lock()
            .map_err(|_| TrustformersError::runtime_error("Mutex lock poisoned".into()))?;
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
            (
                "param1".to_string(),
                Tensor::zeros(&[10, 10]).expect("Failed to create tensor"),
            ),
            (
                "param2".to_string(),
                Tensor::zeros(&[5, 5]).expect("Failed to create tensor"),
            ),
        ];

        let optimizer =
            PyTorchAdam::from_params(params).expect("Failed to create optimizer from params");
        assert_eq!(optimizer.param_groups().len(), 1);
        assert_eq!(optimizer.param_groups()[0].params.len(), 2);
    }

    #[test]
    fn test_pytorch_adamw_creation() {
        let params = vec![(
            "param1".to_string(),
            Tensor::zeros(&[10, 10]).expect("Failed to create tensor"),
        )];

        let optimizer =
            PyTorchAdamW::from_params(params).expect("Failed to create optimizer from params");
        assert_eq!(optimizer.param_groups().len(), 1);
        assert_eq!(optimizer.defaults().weight_decay, 0.01);
    }

    #[test]
    fn test_pytorch_sgd_creation() {
        let params = vec![(
            "param1".to_string(),
            Tensor::zeros(&[10, 10]).expect("Failed to create tensor"),
        )];

        let optimizer =
            PyTorchSGD::from_params(params).expect("Failed to create optimizer from params");
        assert_eq!(optimizer.param_groups().len(), 1);
        assert_eq!(optimizer.defaults().lr, 0.01);
    }

    #[test]
    fn test_pytorch_optimizer_factory() {
        let params = vec![(
            "param1".to_string(),
            Tensor::zeros(&[10, 10]).expect("Failed to create tensor"),
        )];

        let adam =
            PyTorchOptimizerFactory::adam(params.clone(), 0.001, (0.9, 0.999), 1e-8, 0.0, false)
                .expect("Operation failed in test");
        assert_eq!(adam.param_groups()[0].lr, 0.001);

        let adamw =
            PyTorchOptimizerFactory::adamw(params.clone(), 0.001, (0.9, 0.999), 1e-8, 0.01, false)
                .expect("Operation failed in test");
        assert_eq!(adamw.param_groups()[0].weight_decay, 0.01);

        let sgd = PyTorchOptimizerFactory::sgd(params, 0.01, 0.9, 0.0, 0.0, false)
            .expect("Operation failed in test");
        assert_eq!(sgd.param_groups()[0].momentum, Some(0.9));
    }

    #[test]
    fn test_param_group_operations() {
        let params = vec![(
            "param1".to_string(),
            Tensor::zeros(&[10, 10]).expect("Failed to create tensor"),
        )];

        let mut optimizer =
            PyTorchAdam::from_params(params).expect("Failed to create optimizer from params");

        let new_group = PyTorchParamGroup {
            params: vec!["param2".to_string()],
            lr: 0.002,
            ..Default::default()
        };

        optimizer.add_param_group(new_group).expect("Failed to add param group");
        assert_eq!(optimizer.param_groups().len(), 2);
        assert_eq!(optimizer.param_groups()[1].lr, 0.002);
    }

    #[test]
    fn test_state_dict_operations() {
        let params = vec![(
            "param1".to_string(),
            Tensor::zeros(&[10, 10]).expect("Failed to create tensor"),
        )];

        let optimizer =
            PyTorchAdam::from_params(params).expect("Failed to create optimizer from params");
        let state_dict = optimizer.state_dict().expect("state_dict");

        assert_eq!(state_dict.param_groups.len(), 1);
        assert!(state_dict.state.contains_key("adam_state"));
    }

    #[test]
    fn test_zero_grad() {
        let params = vec![(
            "param1".to_string(),
            Tensor::zeros(&[10, 10]).expect("Failed to create tensor"),
        )];

        let mut optimizer =
            PyTorchAdam::from_params(params).expect("Failed to create optimizer from params");
        optimizer
            .set_grad(
                "param1".to_string(),
                Tensor::ones(&[10, 10]).expect("Failed to create tensor"),
            )
            .expect("Operation failed in test");

        // Check that gradient is set
        assert_eq!(
            optimizer.gradients.lock().expect("Mutex lock poisoned").len(),
            1
        );

        // Zero gradients
        optimizer.zero_grad(false).expect("Zero grad failed");
        assert_eq!(
            optimizer.gradients.lock().expect("Mutex lock poisoned").len(),
            0
        );
    }

    /// Regression: `load_state_dict` used to restore nothing, insert momentum buffers
    /// into the *parameter* registry, and report success on a malformed checkpoint.
    ///
    /// The check is a real resume: after save → new optimizer → load, the next step
    /// must land exactly where the uninterrupted run's next step lands.
    #[test]
    fn state_dict_round_trip_reproduces_the_trajectory() {
        fn build() -> PyTorchAdam {
            let params = vec![(
                "w".to_string(),
                Tensor::from_vec(vec![1.0_f32, 2.0], &[2]).expect("tensor"),
            )];
            PyTorchAdam::from_params(params).expect("optimizer")
        }

        fn drive(optimizer: &mut PyTorchAdam, steps: usize) {
            let grad = Tensor::from_vec(vec![0.5_f32, -0.5], &[2]).expect("grad");
            for _ in 0..steps {
                optimizer.set_grad("w".to_string(), grad.clone()).expect("grad");
                optimizer.step(None).expect("step");
            }
        }

        fn value_of(optimizer: &PyTorchAdam) -> Vec<f32> {
            optimizer
                .parameters
                .lock()
                .expect("registry")
                .get("w")
                .expect("parameter")
                .data_f32()
                .expect("data")
        }

        let mut original = build();
        original
            .register_param(
                "w".to_string(),
                Tensor::from_vec(vec![1.0_f32, 2.0], &[2]).expect("tensor"),
            )
            .expect("register");
        drive(&mut original, 3);

        let checkpoint = original.state_dict().expect("state_dict");
        let entries = checkpoint
            .state
            .get("adam_state")
            .and_then(|v| v.as_object())
            .expect("adam_state object");
        assert!(
            entries.keys().any(|k| k.starts_with("exp_avg_")),
            "the moment buffers must be checkpointed, found {:?}",
            entries.keys().collect::<Vec<_>>()
        );

        let resume_point = value_of(&original);

        let mut resumed = build();
        resumed.load_state_dict(checkpoint).expect("load_state_dict");
        assert_eq!(
            resumed.parameters.lock().expect("registry").len(),
            0,
            "loading optimizer state must not inject buffers into the parameter map"
        );
        resumed
            .register_param(
                "w".to_string(),
                Tensor::from_vec(resume_point.clone(), &[2]).expect("tensor"),
            )
            .expect("register");

        drive(&mut original, 1);
        drive(&mut resumed, 1);

        let expected = value_of(&original);
        let actual = value_of(&resumed);
        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "resume diverged from the uninterrupted run: {a} vs {b}"
            );
        }
        assert!(
            actual.iter().zip(resume_point.iter()).any(|(a, b)| (a - b).abs() > 1e-9),
            "the post-resume step must actually move the parameter"
        );
    }

    /// A malformed checkpoint must fail loudly instead of "loading" nothing.
    #[test]
    fn malformed_checkpoint_is_rejected() {
        let params = vec![("w".to_string(), Tensor::zeros(&[2]).expect("tensor"))];
        let mut optimizer = PyTorchAdam::from_params(params).expect("optimizer");

        let bogus = PyTorchOptimizerState {
            state: [(
                "adam_state".to_string(),
                serde_json::json!({"exp_avg_p:0": {"shape": [4], "data": [1.0, 2.0]}}),
            )]
            .into(),
            param_groups: Vec::new(),
        };
        assert!(
            optimizer.load_state_dict(bogus).is_err(),
            "a shape/payload mismatch must be an error"
        );

        let missing = PyTorchOptimizerState {
            state: HashMap::new(),
            param_groups: Vec::new(),
        };
        assert!(
            optimizer.load_state_dict(missing).is_err(),
            "a checkpoint with no optimizer state must be an error"
        );
    }
}
