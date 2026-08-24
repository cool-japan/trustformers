//! `BertModel`: the plain BERT encoder wrapper (no task head). Split out of
//! `models/mod.rs` (2026-08-24, `py-followups`, keeping this crate's files
//! under the 2000-line policy); behavior is unchanged, this is a pure
//! code-motion.

use super::base::PyPreTrainedModel;
use super::helpers::{config_to_dict, parse_bert_config};
use super::weights::{
    load_config_from_hub, load_pretrained_weights, report_weight_loading,
    save_pretrained_for_model, trustformers_error_to_py_err,
};
use crate::tensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;
use trustformers_core::traits::Model;
use trustformers_models::bert::{BertConfig, BertModel};

/// Owned Python reference alias (pyo3 0.28 removed the `PyObject` type alias from
/// the crate root; it is equivalent to `Py<PyAny>`).
type PyObject = Py<PyAny>;

/// BERT Model wrapper
#[pyclass(name = "BertModel", module = "trustformers", extends = PyPreTrainedModel)]
pub struct PyBertModel {
    inner: BertModel,
}

#[pymethods]
impl PyBertModel {
    /// Create a new BERT model
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(
        py: Python<'_>,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, PyPreTrainedModel)> {
        let bert_config = if let Some(cfg) = config {
            // Parse config from dict
            parse_bert_config(cfg)?
        } else {
            BertConfig::default()
        };

        let model = BertModel::new(bert_config.clone())
            .map_err(|e| PyValueError::new_err(format!("Failed to create BERT model: {}", e)))?;

        let config_dict = config_to_dict(py, &bert_config)?;

        Ok((
            PyBertModel { inner: model },
            PyPreTrainedModel {
                config: config_dict.into(),
            },
        ))
    }

    /// Load from pretrained model
    #[staticmethod]
    #[pyo3(signature = (model_name_or_path, **_kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        model_name_or_path: &str,
        _kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyBertModel>> {
        // Load config from a local config.json (see `load_config_from_hub` docs)
        let config_json = load_config_from_hub(model_name_or_path, None)
            .map_err(|e| PyValueError::new_err(format!("Failed to load config: {}", e)))?;

        // Parse config into BertConfig
        let config_value: Value = serde_json::from_str(&config_json)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse config JSON: {}", e)))?;
        let mut config = BertConfig::default();
        if let Some(vocab_size) = config_value.get("vocab_size").and_then(|v| v.as_u64()) {
            config.vocab_size = vocab_size as usize;
        }
        if let Some(hidden_size) = config_value.get("hidden_size").and_then(|v| v.as_u64()) {
            config.hidden_size = hidden_size as usize;
        }
        if let Some(num_layers) = config_value.get("num_hidden_layers").and_then(|v| v.as_u64()) {
            config.num_hidden_layers = num_layers as usize;
        }
        if let Some(num_heads) = config_value.get("num_attention_heads").and_then(|v| v.as_u64()) {
            config.num_attention_heads = num_heads as usize;
        }
        if let Some(intermediate_size) =
            config_value.get("intermediate_size").and_then(|v| v.as_u64())
        {
            config.intermediate_size = intermediate_size as usize;
        }
        if let Some(max_pos) = config_value.get("max_position_embeddings").and_then(|v| v.as_u64())
        {
            config.max_position_embeddings = max_pos as usize;
        }

        let mut model = BertModel::new(config.clone())
            .map_err(|e| PyValueError::new_err(format!("Failed to create model: {}", e)))?;

        // Load weights from a local checkpoint if available. A checkpoint that
        // exists but fails to bind is a hard error -- see `load_pretrained_weights`.
        report_weight_loading(
            load_pretrained_weights(&mut model, model_name_or_path),
            model_name_or_path,
        )?;

        let config_dict = config_to_dict(py, &config)?;

        Py::new(
            py,
            (
                PyBertModel { inner: model },
                PyPreTrainedModel {
                    config: config_dict.into(),
                },
            ),
        )
    }

    /// Forward pass
    #[pyo3(signature = (input_ids, attention_mask=None, token_type_ids=None))]
    pub fn forward(
        &self,
        input_ids: &PyTensor,
        attention_mask: Option<&PyTensor>,
        token_type_ids: Option<&PyTensor>,
    ) -> PyResult<PyObject> {
        Python::attach(|py| {
            // Create TokenizedInput from the tensor arguments
            use trustformers_core::traits::TokenizedInput;

            let tokenized_input = TokenizedInput {
                input_ids: input_ids
                    .inner
                    .to_vec_f32()
                    .map_err(trustformers_error_to_py_err)?
                    .iter()
                    .map(|&x| x as u32)
                    .collect(),
                attention_mask: attention_mask.map_or_else(
                    || vec![1u8; input_ids.inner.shape()[0]],
                    |mask| {
                        mask.inner
                            .to_vec_f32()
                            .unwrap_or_default()
                            .iter()
                            .map(|&x| x as u8)
                            .collect()
                    },
                ),
                token_type_ids: token_type_ids.map(|t| {
                    t.inner.to_vec_f32().unwrap_or_default().iter().map(|&x| x as u32).collect()
                }),
                special_tokens_mask: None,
                offset_mapping: None,
                overflowing_tokens: None,
            };

            let outputs = self
                .inner
                .forward(tokenized_input)
                .map_err(|e| PyValueError::new_err(format!("Forward pass failed: {}", e)))?;

            // Create output dictionary
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item(
                "last_hidden_state",
                PyTensor::from_tensor(outputs.last_hidden_state),
            )?;
            if let Some(pooler) = outputs.pooler_output {
                dict.set_item("pooler_output", PyTensor::from_tensor(pooler))?;
            }

            Ok(dict.into())
        })
    }

    /// Python's __call__ method
    pub fn __call__(
        &self,
        input_ids: &PyTensor,
        attention_mask: Option<&PyTensor>,
        token_type_ids: Option<&PyTensor>,
    ) -> PyResult<PyObject> {
        self.forward(input_ids, attention_mask, token_type_ids)
    }

    /// Save this model's config and parameters to `save_directory`.
    pub fn save_pretrained(&self, save_directory: &str) -> PyResult<()> {
        save_pretrained_for_model(&self.inner, save_directory, "BertModel")
    }
}
