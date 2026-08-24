//! `LlamaModel`: the LLaMA-family causal decoder wrapper. Split out of
//! `models/mod.rs` (2026-08-24, `py-followups`, keeping this crate's files
//! under the 2000-line policy); behavior is unchanged, this is a pure
//! code-motion.

use super::base::PyPreTrainedModel;
use super::weights::{
    load_config_from_hub, load_pretrained_weights, report_weight_loading,
    save_pretrained_for_model, trustformers_error_to_py_err,
};
use crate::config_utils::{llama_config_to_dict, parse_llama_config};
use crate::tensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;
use trustformers_core::traits::Model;
use trustformers_models::llama::{LlamaConfig, LlamaModel};

/// Owned Python reference alias (pyo3 0.28 removed the `PyObject` type alias from
/// the crate root; it is equivalent to `Py<PyAny>`).
type PyObject = Py<PyAny>;

/// LLaMA Model wrapper
#[pyclass(name = "LlamaModel", module = "trustformers", extends = PyPreTrainedModel)]
pub struct PyLlamaModel {
    inner: LlamaModel,
}

#[pymethods]
impl PyLlamaModel {
    /// Create a new LLaMA model
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(
        py: Python<'_>,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, PyPreTrainedModel)> {
        let llama_config = if let Some(cfg) = config {
            parse_llama_config(cfg)?
        } else {
            LlamaConfig::default()
        };

        let model = LlamaModel::new(llama_config.clone())
            .map_err(|e| PyValueError::new_err(format!("Failed to create LLaMA model: {}", e)))?;

        let config_dict = llama_config_to_dict(py, &llama_config)?;

        Ok((
            PyLlamaModel { inner: model },
            PyPreTrainedModel {
                config: config_dict.into(),
            },
        ))
    }

    /// Load a pretrained LLaMA model from HuggingFace Hub
    #[staticmethod]
    #[pyo3(signature = (model_name_or_path, **_kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        model_name_or_path: &str,
        _kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyLlamaModel>> {
        // Load config from a local config.json (see `load_config_from_hub` docs)
        let config_json = load_config_from_hub(model_name_or_path, None)
            .map_err(|e| PyValueError::new_err(format!("Failed to load config: {}", e)))?;

        // Parse config into LlamaConfig
        let config_value: Value = serde_json::from_str(&config_json)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse config JSON: {}", e)))?;
        let mut config = LlamaConfig::default();

        // Update config with loaded values
        if let Some(vocab_size) = config_value.get("vocab_size").and_then(|v| v.as_u64()) {
            config.vocab_size = vocab_size as usize;
        }
        if let Some(hidden_size) = config_value.get("hidden_size").and_then(|v| v.as_u64()) {
            config.hidden_size = hidden_size as usize;
        }
        if let Some(intermediate_size) =
            config_value.get("intermediate_size").and_then(|v| v.as_u64())
        {
            config.intermediate_size = intermediate_size as usize;
        }
        if let Some(num_layers) = config_value.get("num_hidden_layers").and_then(|v| v.as_u64()) {
            config.num_hidden_layers = num_layers as usize;
        }
        if let Some(num_heads) = config_value.get("num_attention_heads").and_then(|v| v.as_u64()) {
            config.num_attention_heads = num_heads as usize;
        }
        if let Some(max_pos) = config_value.get("max_position_embeddings").and_then(|v| v.as_u64())
        {
            config.max_position_embeddings = max_pos as usize;
        }

        let mut model = LlamaModel::new(config.clone())
            .map_err(|e| PyValueError::new_err(format!("Failed to create model: {}", e)))?;

        report_weight_loading(
            load_pretrained_weights(&mut model, model_name_or_path),
            model_name_or_path,
        )?;

        let config_dict = llama_config_to_dict(py, &config)?;

        Py::new(
            py,
            (
                PyLlamaModel { inner: model },
                PyPreTrainedModel {
                    config: config_dict.into(),
                },
            ),
        )
    }

    /// Forward pass
    #[pyo3(signature = (input_ids, attention_mask=None, position_ids=None))]
    pub fn forward(
        &self,
        input_ids: &PyTensor,
        attention_mask: Option<&PyTensor>,
        position_ids: Option<&PyTensor>,
    ) -> PyResult<PyObject> {
        // LLaMA core forward derives causal masking/positions internally.
        let _ = (attention_mask, position_ids);
        Python::attach(|py| {
            // Convert input tensor to token IDs for LLaMA
            let input_token_ids = input_ids
                .inner
                .to_vec_f32()
                .map_err(trustformers_error_to_py_err)?
                .iter()
                .map(|&x| x as u32)
                .collect::<Vec<u32>>();

            let outputs = self
                .inner
                .forward(input_token_ids)
                .map_err(|e| PyValueError::new_err(format!("Forward pass failed: {}", e)))?;

            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("last_hidden_state", PyTensor::from_tensor(outputs))?;

            Ok(dict.into())
        })
    }

    /// Save this model's config and parameters to `save_directory`.
    pub fn save_pretrained(&self, save_directory: &str) -> PyResult<()> {
        save_pretrained_for_model(&self.inner, save_directory, "LlamaModel")
    }
}
