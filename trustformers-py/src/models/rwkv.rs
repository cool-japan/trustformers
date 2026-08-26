//! `RwkvModel`: the RWKV linear-attention / RNN-style causal language model
//! wrapper, plus its config (de)serialization helpers. Split out of
//! `models/mod.rs` (2026-08-24, `py-followups`, keeping this crate's files
//! under the 2000-line policy); behavior is unchanged, this is a pure
//! code-motion -- `parse_rwkv_config`/`rwkv_config_to_dict` moved here
//! alongside their only caller instead of into `helpers`, since (unlike
//! `parse_bert_config`/`config_to_dict`) nothing outside this file uses them.

use super::base::PyPreTrainedModel;
use super::helpers::{argmax_last_token, build_token_tensor, extract_token_ids};
use super::weights::{
    load_config_from_hub, load_pretrained_weights, report_weight_loading,
    save_pretrained_for_model,
};
use crate::tensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;
use trustformers_core::traits::Model;
use trustformers_models::rwkv::{RwkvConfig, RwkvModel};

/// Owned Python reference alias (pyo3 0.28 removed the `PyObject` type alias from
/// the crate root; it is equivalent to `Py<PyAny>`).
type PyObject = Py<PyAny>;

/// RWKV Model wrapper (linear-attention / RNN-style causal language model)
#[pyclass(name = "RwkvModel", module = "trustformers", extends = PyPreTrainedModel)]
pub struct PyRwkvModel {
    inner: RwkvModel,
}

#[pymethods]
impl PyRwkvModel {
    /// Create a new RWKV model
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(
        py: Python<'_>,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, PyPreTrainedModel)> {
        let rwkv_config =
            if let Some(cfg) = config { parse_rwkv_config(cfg)? } else { RwkvConfig::default() };

        let model = RwkvModel::new(rwkv_config.clone())
            .map_err(|e| PyValueError::new_err(format!("Failed to create RWKV model: {}", e)))?;

        let config_dict = rwkv_config_to_dict(py, &rwkv_config)?;

        Ok((
            PyRwkvModel { inner: model },
            PyPreTrainedModel {
                config: config_dict.into(),
            },
        ))
    }

    /// Load a pretrained RWKV model
    #[staticmethod]
    #[pyo3(signature = (model_name_or_path, **_kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        model_name_or_path: &str,
        _kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyRwkvModel>> {
        // Load config from a local config.json (see `load_config_from_hub` docs)
        let config_json = load_config_from_hub(model_name_or_path, None)
            .map_err(|e| PyValueError::new_err(format!("Failed to load config: {}", e)))?;

        // Parse config into RwkvConfig, accepting both RWKV-native and HF key names.
        let config_value: Value = serde_json::from_str(&config_json)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse config JSON: {}", e)))?;
        let mut config = RwkvConfig::default();
        if let Some(v) = config_value
            .get("n_embd")
            .or_else(|| config_value.get("hidden_size"))
            .and_then(|v| v.as_u64())
        {
            config.n_embd = v as usize;
        }
        if let Some(v) = config_value
            .get("n_layer")
            .or_else(|| config_value.get("num_hidden_layers"))
            .and_then(|v| v.as_u64())
        {
            config.n_layer = v as usize;
        }
        if let Some(v) = config_value.get("vocab_size").and_then(|v| v.as_u64()) {
            config.vocab_size = v as usize;
        }
        if let Some(v) = config_value
            .get("n_head")
            .or_else(|| config_value.get("num_attention_heads"))
            .and_then(|v| v.as_u64())
        {
            config.n_head = v as usize;
        }
        if let Some(v) = config_value
            .get("ctx_len")
            .or_else(|| config_value.get("context_length"))
            .and_then(|v| v.as_u64())
        {
            config.ctx_len = v as usize;
        }
        // Preserve the RWKV invariant `n_embd == n_head * head_size`.
        if config.n_head > 0 && config.n_embd % config.n_head == 0 {
            config.head_size = config.n_embd / config.n_head;
        }

        let mut model = RwkvModel::new(config.clone())
            .map_err(|e| PyValueError::new_err(format!("Failed to create model: {}", e)))?;

        report_weight_loading(
            load_pretrained_weights(&mut model, model_name_or_path),
            model_name_or_path,
        )?;

        let config_dict = rwkv_config_to_dict(py, &config)?;

        Py::new(
            py,
            (
                PyRwkvModel { inner: model },
                PyPreTrainedModel {
                    config: config_dict.into(),
                },
            ),
        )
    }

    /// Forward pass — returns the last hidden state under `last_hidden_state`.
    #[pyo3(signature = (input_ids, attention_mask=None, position_ids=None))]
    pub fn forward(
        &self,
        input_ids: &PyTensor,
        attention_mask: Option<&PyTensor>,
        position_ids: Option<&PyTensor>,
    ) -> PyResult<PyObject> {
        // RWKV is a recurrent architecture: it uses neither an attention mask nor
        // position ids. They are accepted for API parity with the other models.
        let _ = (attention_mask, position_ids);
        Python::attach(|py| {
            let outputs = self
                .inner
                .forward(input_ids.inner.clone())
                .map_err(|e| PyValueError::new_err(format!("Forward pass failed: {}", e)))?;

            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("last_hidden_state", PyTensor::from_tensor(outputs))?;
            Ok(dict.into())
        })
    }

    /// Python's `__call__` method
    #[pyo3(signature = (input_ids, attention_mask=None, position_ids=None))]
    pub fn __call__(
        &self,
        input_ids: &PyTensor,
        attention_mask: Option<&PyTensor>,
        position_ids: Option<&PyTensor>,
    ) -> PyResult<PyObject> {
        self.forward(input_ids, attention_mask, position_ids)
    }

    /// Greedy autoregressive generation using the RWKV language-model head.
    #[pyo3(signature = (input_ids, max_length=50))]
    pub fn generate(&self, input_ids: &PyTensor, max_length: usize) -> PyResult<PyTensor> {
        let mut tokens = extract_token_ids(&input_ids.inner)?;
        while tokens.len() < max_length {
            let input_tensor = build_token_tensor(&tokens)?;
            let logits = self
                .inner
                .forward_lm(&input_tensor)
                .map_err(|e| PyValueError::new_err(format!("Generation forward pass failed: {}", e)))?;
            tokens.push(argmax_last_token(&logits)? as i64);
        }
        Ok(PyTensor {
            inner: build_token_tensor(&tokens)?,
            variable: None,
        })
    }

    /// Save this model's config and parameters to `save_directory`.
    pub fn save_pretrained(&self, save_directory: &str) -> PyResult<()> {
        save_pretrained_for_model(&self.inner, save_directory, "RwkvModel")
    }
}

/// Build an `RwkvConfig` from a Python config dict.
fn parse_rwkv_config(config_dict: &Bound<'_, PyAny>) -> PyResult<RwkvConfig> {
    let dict = config_dict.cast::<pyo3::types::PyDict>()?;
    let mut config = RwkvConfig::default();

    if let Ok(Some(v)) = dict.get_item("n_embd") {
        config.n_embd = v.extract()?;
    } else if let Ok(Some(v)) = dict.get_item("hidden_size") {
        config.n_embd = v.extract()?;
    }
    if let Ok(Some(v)) = dict.get_item("n_layer") {
        config.n_layer = v.extract()?;
    } else if let Ok(Some(v)) = dict.get_item("num_hidden_layers") {
        config.n_layer = v.extract()?;
    }
    if let Ok(Some(v)) = dict.get_item("vocab_size") {
        config.vocab_size = v.extract()?;
    }
    if let Ok(Some(v)) = dict.get_item("n_head") {
        config.n_head = v.extract()?;
    } else if let Ok(Some(v)) = dict.get_item("num_attention_heads") {
        config.n_head = v.extract()?;
    }
    if let Ok(Some(v)) = dict.get_item("ctx_len") {
        config.ctx_len = v.extract()?;
    }
    if config.n_head > 0 && config.n_embd % config.n_head == 0 {
        config.head_size = config.n_embd / config.n_head;
    }
    Ok(config)
}

/// Serialize an `RwkvConfig` to a Python dict (with HF-style aliases).
fn rwkv_config_to_dict<'py>(
    py: Python<'py>,
    config: &RwkvConfig,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("model_type", "rwkv")?;
    dict.set_item("n_embd", config.n_embd)?;
    dict.set_item("hidden_size", config.n_embd)?;
    dict.set_item("n_layer", config.n_layer)?;
    dict.set_item("num_hidden_layers", config.n_layer)?;
    dict.set_item("vocab_size", config.vocab_size)?;
    dict.set_item("ctx_len", config.ctx_len)?;
    dict.set_item("n_head", config.n_head)?;
    dict.set_item("head_size", config.head_size)?;
    dict.set_item("layer_norm_epsilon", config.layer_norm_epsilon)?;
    Ok(dict)
}
