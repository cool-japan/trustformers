//! `MambaModel`: the Mamba selective state-space causal language model
//! wrapper, plus its config (de)serialization helpers. Split out of
//! `models/mod.rs` (2026-08-24, `py-followups`, keeping this crate's files
//! under the 2000-line policy); behavior is unchanged, this is a pure
//! code-motion -- `parse_mamba_config`/`mamba_config_to_dict` moved here
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
use trustformers_models::mamba::{MambaConfig, MambaModel};

/// Owned Python reference alias (pyo3 0.28 removed the `PyObject` type alias from
/// the crate root; it is equivalent to `Py<PyAny>`).
type PyObject = Py<PyAny>;

/// Mamba Model wrapper (selective state-space causal language model)
#[pyclass(name = "MambaModel", module = "trustformers", extends = PyPreTrainedModel)]
pub struct PyMambaModel {
    inner: MambaModel,
}

#[pymethods]
impl PyMambaModel {
    /// Create a new Mamba model
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(
        py: Python<'_>,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, PyPreTrainedModel)> {
        let mut mamba_config =
            if let Some(cfg) = config { parse_mamba_config(cfg)? } else { MambaConfig::default() };
        // Materialise an explicit LM head so `generate` yields vocabulary-sized logits.
        mamba_config.tie_word_embeddings = false;

        let model = MambaModel::new(mamba_config.clone())
            .map_err(|e| PyValueError::new_err(format!("Failed to create Mamba model: {}", e)))?;

        let config_dict = mamba_config_to_dict(py, &mamba_config)?;

        Ok((
            PyMambaModel { inner: model },
            PyPreTrainedModel {
                config: config_dict.into(),
            },
        ))
    }

    /// Load a pretrained Mamba model
    #[staticmethod]
    #[pyo3(signature = (model_name_or_path, **_kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        model_name_or_path: &str,
        _kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyMambaModel>> {
        // Load config from a local config.json (see `load_config_from_hub` docs)
        let config_json = load_config_from_hub(model_name_or_path, None)
            .map_err(|e| PyValueError::new_err(format!("Failed to load config: {}", e)))?;

        // Parse config into MambaConfig, accepting both Mamba-native and HF key names.
        let config_value: Value = serde_json::from_str(&config_json)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse config JSON: {}", e)))?;
        let mut config = MambaConfig::default();
        if let Some(v) = config_value
            .get("d_model")
            .or_else(|| config_value.get("hidden_size"))
            .and_then(|v| v.as_u64())
        {
            config.d_model = v as usize;
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
        if let Some(v) = config_value.get("d_state").and_then(|v| v.as_u64()) {
            config.d_state = v as usize;
        }
        if let Some(v) = config_value.get("d_conv").and_then(|v| v.as_u64()) {
            config.d_conv = v as usize;
        }
        if let Some(v) = config_value.get("expand").and_then(|v| v.as_u64()) {
            config.expand = v as usize;
        }
        // Materialise an explicit LM head so `generate` yields vocabulary-sized logits.
        config.tie_word_embeddings = false;

        let mut model = MambaModel::new(config.clone())
            .map_err(|e| PyValueError::new_err(format!("Failed to create model: {}", e)))?;

        report_weight_loading(
            load_pretrained_weights(&mut model, model_name_or_path),
            model_name_or_path,
        )?;

        let config_dict = mamba_config_to_dict(py, &config)?;

        Py::new(
            py,
            (
                PyMambaModel { inner: model },
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
        // Mamba is a state-space architecture: it uses neither an attention mask nor
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

    /// Greedy autoregressive generation using the Mamba language-model head.
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
        save_pretrained_for_model(&self.inner, save_directory, "MambaModel")
    }
}

/// Build a `MambaConfig` from a Python config dict.
fn parse_mamba_config(config_dict: &Bound<'_, PyAny>) -> PyResult<MambaConfig> {
    let dict = config_dict.cast::<pyo3::types::PyDict>()?;
    let mut config = MambaConfig::default();

    if let Ok(Some(v)) = dict.get_item("d_model") {
        config.d_model = v.extract()?;
    } else if let Ok(Some(v)) = dict.get_item("hidden_size") {
        config.d_model = v.extract()?;
    }
    if let Ok(Some(v)) = dict.get_item("n_layer") {
        config.n_layer = v.extract()?;
    } else if let Ok(Some(v)) = dict.get_item("num_hidden_layers") {
        config.n_layer = v.extract()?;
    }
    if let Ok(Some(v)) = dict.get_item("vocab_size") {
        config.vocab_size = v.extract()?;
    }
    if let Ok(Some(v)) = dict.get_item("d_state") {
        config.d_state = v.extract()?;
    }
    if let Ok(Some(v)) = dict.get_item("d_conv") {
        config.d_conv = v.extract()?;
    }
    if let Ok(Some(v)) = dict.get_item("expand") {
        config.expand = v.extract()?;
    }
    Ok(config)
}

/// Serialize a `MambaConfig` to a Python dict (with HF-style aliases).
fn mamba_config_to_dict<'py>(
    py: Python<'py>,
    config: &MambaConfig,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("model_type", "mamba")?;
    dict.set_item("d_model", config.d_model)?;
    dict.set_item("hidden_size", config.d_model)?;
    dict.set_item("n_layer", config.n_layer)?;
    dict.set_item("num_hidden_layers", config.n_layer)?;
    dict.set_item("vocab_size", config.vocab_size)?;
    dict.set_item("d_state", config.d_state)?;
    dict.set_item("d_conv", config.d_conv)?;
    dict.set_item("expand", config.expand)?;
    dict.set_item("rms_norm_eps", config.rms_norm_eps)?;
    Ok(dict)
}
