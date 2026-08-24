//! `T5Model`: the encoder-decoder wrapper. Split out of `models/mod.rs`
//! (2026-08-24, `py-followups`, keeping this crate's files under the
//! 2000-line policy); behavior is unchanged, this is a pure code-motion.

use super::base::PyPreTrainedModel;
use super::weights::{
    load_config_from_hub, load_pretrained_weights, report_weight_loading,
    save_pretrained_for_model,
};
use crate::config_utils::{parse_t5_config, t5_config_to_dict};
use crate::tensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Model;
use trustformers_models::t5::{T5Config, T5Model};

/// Owned Python reference alias (pyo3 0.28 removed the `PyObject` type alias from
/// the crate root; it is equivalent to `Py<PyAny>`).
type PyObject = Py<PyAny>;

/// T5 Model wrapper
#[pyclass(name = "T5Model", module = "trustformers", extends = PyPreTrainedModel)]
pub struct PyT5Model {
    inner: T5Model,
}

#[pymethods]
impl PyT5Model {
    /// Create a new T5 model
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(
        py: Python<'_>,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, PyPreTrainedModel)> {
        let t5_config =
            if let Some(cfg) = config { parse_t5_config(cfg)? } else { T5Config::default() };

        let model = T5Model::new(t5_config.clone())
            .map_err(|e| PyValueError::new_err(format!("Failed to create T5 model: {}", e)))?;

        let config_dict = t5_config_to_dict(py, &t5_config)?;

        Ok((
            PyT5Model { inner: model },
            PyPreTrainedModel {
                config: config_dict.into(),
            },
        ))
    }

    /// Load a pretrained T5 model from HuggingFace Hub
    #[staticmethod]
    #[pyo3(signature = (model_name_or_path, **_kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        model_name_or_path: &str,
        _kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyT5Model>> {
        // Load config from a local config.json (see `load_config_from_hub` docs)
        let config_json = load_config_from_hub(model_name_or_path, None)
            .map_err(|e| PyValueError::new_err(format!("Failed to load config: {}", e)))?;

        // Parse config into T5Config
        let config_value: Value = serde_json::from_str(&config_json)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse config JSON: {}", e)))?;
        let mut config = T5Config::default();

        // Update config with loaded values
        if let Some(vocab_size) = config_value.get("vocab_size").and_then(|v| v.as_u64()) {
            config.vocab_size = vocab_size as usize;
        }
        if let Some(d_model) = config_value.get("d_model").and_then(|v| v.as_u64()) {
            config.d_model = d_model as usize;
        }
        if let Some(d_ff) = config_value.get("d_ff").and_then(|v| v.as_u64()) {
            config.d_ff = d_ff as usize;
        }
        if let Some(num_layers) = config_value.get("num_layers").and_then(|v| v.as_u64()) {
            config.num_layers = num_layers as usize;
        }
        if let Some(num_heads) = config_value.get("num_heads").and_then(|v| v.as_u64()) {
            config.num_heads = num_heads as usize;
        }

        let mut model = T5Model::new(config.clone())
            .map_err(|e| PyValueError::new_err(format!("Failed to create model: {}", e)))?;

        report_weight_loading(
            load_pretrained_weights(&mut model, model_name_or_path),
            model_name_or_path,
        )?;

        let config_dict = t5_config_to_dict(py, &config)?;

        Py::new(
            py,
            (
                PyT5Model { inner: model },
                PyPreTrainedModel {
                    config: config_dict.into(),
                },
            ),
        )
    }

    /// Forward pass
    #[pyo3(signature = (input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None))]
    pub fn forward(
        &self,
        input_ids: Option<&PyTensor>,
        attention_mask: Option<&PyTensor>,
        decoder_input_ids: Option<&PyTensor>,
        decoder_attention_mask: Option<&PyTensor>,
    ) -> PyResult<PyObject> {
        use trustformers_core::traits::TokenizedInput;

        // Convert input_ids (required for T5)
        let input_token_ids = if let Some(input_tensor) = input_ids {
            match &input_tensor.inner {
                Tensor::I64(arr) => arr.iter().map(|&x| x as u32).collect::<Vec<u32>>(),
                Tensor::F32(arr) => arr.iter().map(|&x| x as u32).collect::<Vec<u32>>(),
                _ => {
                    return Err(PyValueError::new_err(
                        "Input tensor must contain integer token IDs",
                    ))
                },
            }
        } else {
            return Err(PyValueError::new_err(
                "input_ids is required for T5 forward pass",
            ));
        };

        // Convert attention_mask (use default if not provided)
        let input_attention_mask = if let Some(mask_tensor) = attention_mask {
            match &mask_tensor.inner {
                Tensor::I64(arr) => arr.iter().map(|&x| x as u8).collect::<Vec<u8>>(),
                Tensor::F32(arr) => arr.iter().map(|&x| x as u8).collect::<Vec<u8>>(),
                _ => {
                    return Err(PyValueError::new_err(
                        "Attention mask must contain integer values",
                    ))
                },
            }
        } else {
            vec![1u8; input_token_ids.len()] // Default to all ones
        };

        // Convert decoder_input_ids (optional for T5)
        let decoder_tokenized_input = if let Some(decoder_tensor) = decoder_input_ids {
            let decoder_token_ids = match &decoder_tensor.inner {
                Tensor::I64(arr) => arr.iter().map(|&x| x as u32).collect::<Vec<u32>>(),
                Tensor::F32(arr) => arr.iter().map(|&x| x as u32).collect::<Vec<u32>>(),
                _ => {
                    return Err(PyValueError::new_err(
                        "Decoder input tensor must contain integer token IDs",
                    ))
                },
            };

            // Convert decoder attention mask if provided
            let decoder_att_mask = if let Some(dec_mask_tensor) = decoder_attention_mask {
                match &dec_mask_tensor.inner {
                    Tensor::I64(arr) => arr.iter().map(|&x| x as u8).collect::<Vec<u8>>(),
                    Tensor::F32(arr) => arr.iter().map(|&x| x as u8).collect::<Vec<u8>>(),
                    _ => {
                        return Err(PyValueError::new_err(
                            "Decoder attention mask must contain integer values",
                        ))
                    },
                }
            } else {
                vec![1u8; decoder_token_ids.len()] // Default to all ones
            };

            Some(TokenizedInput {
                input_ids: decoder_token_ids,
                attention_mask: decoder_att_mask,
                token_type_ids: None,
                special_tokens_mask: None,
                offset_mapping: None,
                overflowing_tokens: None,
            })
        } else {
            None
        };

        // Create T5Input
        let input = trustformers_models::t5::T5Input {
            input_ids: TokenizedInput {
                input_ids: input_token_ids,
                attention_mask: input_attention_mask,
                token_type_ids: None,
                special_tokens_mask: None,
                offset_mapping: None,
                overflowing_tokens: None,
            },
            decoder_input_ids: decoder_tokenized_input,
            encoder_outputs: None,
        };

        // Forward pass
        let output = self
            .inner
            .forward(input)
            .map_err(|e| PyValueError::new_err(format!("T5 forward pass failed: {}", e)))?;

        // Convert output to Python dictionary
        Python::attach(|py| {
            let dict = pyo3::types::PyDict::new(py);

            // Convert last_hidden_state to PyTensor
            let last_hidden_py = PyTensor {
                inner: output.last_hidden_state,
                variable: None,
            };
            dict.set_item("last_hidden_state", last_hidden_py)?;

            // Add encoder_last_hidden_state if available
            if let Some(encoder_hidden) = output.encoder_last_hidden_state {
                let encoder_hidden_py = PyTensor {
                    inner: encoder_hidden,
                    variable: None,
                };
                dict.set_item("encoder_last_hidden_state", encoder_hidden_py)?;
            }

            Ok(dict.into())
        })
    }

    /// Save this model's config and parameters to `save_directory`.
    pub fn save_pretrained(&self, save_directory: &str) -> PyResult<()> {
        save_pretrained_for_model(&self.inner, save_directory, "T5Model")
    }
}
