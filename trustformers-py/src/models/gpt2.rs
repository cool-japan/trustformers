//! `GPT2Model` (plain encoder/decoder stack) and `GPT2LMHeadModel` (with a
//! language-modeling head, real sampling-based generation). Split out of
//! `models/mod.rs` (2026-08-24, `py-followups`, keeping this crate's files
//! under the 2000-line policy); behavior is unchanged, this is a pure
//! code-motion.

use super::base::PyPreTrainedModel;
use super::generation;
use super::helpers::{build_usize_token_tensor, extract_token_ids_usize, tokenized_input_from_tensors};
use super::losses;
use super::weights::{
    load_config_from_hub, load_pretrained_weights, report_weight_loading,
    save_pretrained_for_model, trustformers_error_to_py_err,
};
use crate::config_utils::{gpt2_config_to_dict, parse_gpt2_config};
use crate::tensor::PyTensor;
use generation::SamplingOptions;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Model;
use trustformers_models::gpt2::{Gpt2Config, Gpt2LMHeadModel, Gpt2Model};

/// Owned Python reference alias (pyo3 0.28 removed the `PyObject` type alias from
/// the crate root; it is equivalent to `Py<PyAny>`).
type PyObject = Py<PyAny>;

/// GPT-2 Model wrapper
#[pyclass(name = "GPT2Model", module = "trustformers", extends = PyPreTrainedModel)]
pub struct PyGPT2Model {
    inner: Gpt2Model,
}

#[pymethods]
impl PyGPT2Model {
    /// Create a new GPT-2 model
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(
        py: Python<'_>,
        config: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, PyPreTrainedModel)> {
        let gpt2_config = if let Some(cfg) = config {
            parse_gpt2_config(cfg)?
        } else {
            Gpt2Config::default()
        };

        let model = Gpt2Model::new(gpt2_config.clone())
            .map_err(|e| PyValueError::new_err(format!("Failed to create GPT-2 model: {}", e)))?;

        let config_dict = gpt2_config_to_dict(py, &gpt2_config)?;

        Ok((
            PyGPT2Model { inner: model },
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
    ) -> PyResult<Py<PyGPT2Model>> {
        // Load config from a local config.json (see `load_config_from_hub` docs)
        let config_json = load_config_from_hub(model_name_or_path, None)
            .map_err(|e| PyValueError::new_err(format!("Failed to load config: {}", e)))?;

        // Parse config into Gpt2Config
        let config_value: Value = serde_json::from_str(&config_json)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse config JSON: {}", e)))?;
        let mut config = Gpt2Config::default();
        if let Some(vocab_size) = config_value.get("vocab_size").and_then(|v| v.as_u64()) {
            config.vocab_size = vocab_size as usize;
        }
        if let Some(n_embd) = config_value.get("n_embd").and_then(|v| v.as_u64()) {
            config.n_embd = n_embd as usize;
        }
        if let Some(n_layer) = config_value.get("n_layer").and_then(|v| v.as_u64()) {
            config.n_layer = n_layer as usize;
        }
        if let Some(n_head) = config_value.get("n_head").and_then(|v| v.as_u64()) {
            config.n_head = n_head as usize;
        }
        if let Some(n_positions) = config_value.get("n_positions").and_then(|v| v.as_u64()) {
            config.n_positions = n_positions as usize;
        }

        let mut model = Gpt2Model::new(config.clone())
            .map_err(|e| PyValueError::new_err(format!("Failed to create model: {}", e)))?;

        report_weight_loading(
            load_pretrained_weights(&mut model, model_name_or_path),
            model_name_or_path,
        )?;

        let config_dict = gpt2_config_to_dict(py, &config)?;

        Py::new(
            py,
            (
                PyGPT2Model { inner: model },
                PyPreTrainedModel {
                    config: config_dict.into(),
                },
            ),
        )
    }

    /// Forward pass
    #[pyo3(signature = (input_ids, attention_mask=None, past_key_values=None))]
    pub fn forward(
        &self,
        input_ids: &PyTensor,
        attention_mask: Option<&PyTensor>,
        past_key_values: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        // GPT-2 KV-cache is not yet threaded through the Rust core forward pass.
        let _ = past_key_values;
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
                token_type_ids: None, // GPT-2/LLaMA don't use token type IDs
                special_tokens_mask: None,
                offset_mapping: None,
                overflowing_tokens: None,
            };

            let outputs = self
                .inner
                .forward(tokenized_input)
                .map_err(|e| PyValueError::new_err(format!("Forward pass failed: {}", e)))?;

            let dict = pyo3::types::PyDict::new(py);
            dict.set_item(
                "last_hidden_state",
                PyTensor::from_tensor(outputs.last_hidden_state),
            )?;

            Ok(dict.into())
        })
    }

    /// GPT-2 without a language-modeling head has no vocabulary projection,
    /// so there is nothing for autoregressive generation to sample from --
    /// exactly like HuggingFace's own `GPT2Model`, which likewise has no
    /// `.generate()` (only `GPT2LMHeadModel` / `GPT2DoubleHeadsModel` do).
    ///
    /// This used to append repeated GPT-2 EOS tokens (`50256`) up to
    /// `max_length` and call that "generation". Fabricating output for a
    /// headless model is exactly the kind of invented result this crate must
    /// not produce; a structured error pointing at the class that actually
    /// can generate is the honest replacement.
    #[pyo3(signature = (input_ids, max_length=50, temperature=1.0, top_k=50, top_p=0.95))]
    pub fn generate(
        &self,
        input_ids: &PyTensor,
        max_length: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> PyResult<PyTensor> {
        let _ = (input_ids, max_length, temperature, top_k, top_p);
        Err(PyValueError::new_err(
            "GPT2Model has no language-modeling head and cannot generate text (this matches \
             HuggingFace's own GPT2Model). Use GPT2LMHeadModel.from_pretrained(...) instead.",
        ))
    }

    /// Save this model's config and parameters to `save_directory`.
    pub fn save_pretrained(&self, save_directory: &str) -> PyResult<()> {
        save_pretrained_for_model(&self.inner, save_directory, "GPT2Model")
    }
}

/// GPT2 for Language Modeling Head
#[pyclass(name = "GPT2LMHeadModel", module = "trustformers")]
pub struct PyGPT2LMHeadModel {
    inner: Gpt2LMHeadModel,
}

impl PyGPT2LMHeadModel {
    /// The wrapped Rust model, for the `text-generation` pipeline.
    pub(crate) fn model(&self) -> &Gpt2LMHeadModel {
        &self.inner
    }
}

#[pymethods]
impl PyGPT2LMHeadModel {
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(config: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let gpt2_config = if let Some(cfg) = config {
            parse_gpt2_config(cfg)?
        } else {
            Gpt2Config::default()
        };

        let inner = Gpt2LMHeadModel::new(gpt2_config).map_err(|e| {
            PyValueError::new_err(format!("Failed to create GPT-2 LM head model: {}", e))
        })?;

        Ok(PyGPT2LMHeadModel { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (model_name_or_path, **_kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        model_name_or_path: &str,
        _kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyGPT2LMHeadModel>> {
        // Load config from a local config.json (see `load_config_from_hub` docs)
        let config_json = load_config_from_hub(model_name_or_path, None)
            .map_err(|e| PyValueError::new_err(format!("Failed to load config: {}", e)))?;

        // Parse config into Gpt2Config
        let config_value: Value = serde_json::from_str(&config_json)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse config JSON: {}", e)))?;
        let mut config = Gpt2Config::default();
        if let Some(vocab_size) = config_value.get("vocab_size").and_then(|v| v.as_u64()) {
            config.vocab_size = vocab_size as usize;
        }
        if let Some(n_embd) = config_value.get("n_embd").and_then(|v| v.as_u64()) {
            config.n_embd = n_embd as usize;
        }
        if let Some(n_layer) = config_value.get("n_layer").and_then(|v| v.as_u64()) {
            config.n_layer = n_layer as usize;
        }
        if let Some(n_head) = config_value.get("n_head").and_then(|v| v.as_u64()) {
            config.n_head = n_head as usize;
        }
        if let Some(n_positions) = config_value.get("n_positions").and_then(|v| v.as_u64()) {
            config.n_positions = n_positions as usize;
        }

        // `Gpt2LMHeadModel::load_pretrained` (the real `Model::load_pretrained`
        // implementation) binds the transformer backbone AND the LM head --
        // tying it to the token embedding table when the checkpoint carries no
        // separate `lm_head.weight`, exactly as GPT-2 itself does.
        let mut model = Gpt2LMHeadModel::new(config)
            .map_err(|e| PyValueError::new_err(format!("Failed to create model: {}", e)))?;

        report_weight_loading(
            load_pretrained_weights(&mut model, model_name_or_path),
            model_name_or_path,
        )?;

        Py::new(py, PyGPT2LMHeadModel { inner: model })
    }

    #[pyo3(signature = (input_ids, attention_mask=None, labels=None))]
    pub fn forward(
        &self,
        input_ids: &PyTensor,
        attention_mask: Option<&PyTensor>,
        labels: Option<&PyTensor>,
    ) -> PyResult<PyObject> {
        Python::attach(|py| {
            // GPT-2 has no segment embeddings, so `token_type_ids` is always
            // `None` here.
            let tokenized_input = tokenized_input_from_tensors(input_ids, attention_mask, None)?;

            let outputs = self.inner.forward(tokenized_input).map_err(|e| {
                PyValueError::new_err(format!("Transformer forward pass failed: {}", e))
            })?;

            // Real vocabulary-sized logits from the LM head -- the previous
            // implementation returned `transformer_outputs.last_hidden_state`
            // relabelled as "logits" (comment: "Placeholder"), which is
            // `hidden_size`-wide, not `vocab_size`-wide, and was never passed
            // through any language-modeling head at all.
            let logits = outputs.logits;

            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("logits", PyTensor::from_tensor(logits.clone()))?;

            // Calculate loss if labels provided
            if let Some(labels_tensor) = labels {
                // Compute cross-entropy loss for language modeling (next token prediction)
                let loss_value =
                    losses::language_modeling_cross_entropy(&logits, &labels_tensor.inner)
                    .map_err(|e| {
                        PyValueError::new_err(format!(
                            "Language modeling loss calculation failed: {}",
                            e
                        ))
                    })?;

                let loss = PyTensor::from_tensor(Tensor::scalar(loss_value).map_err(|e| {
                    PyValueError::new_err(format!("Failed to create loss tensor: {}", e))
                })?);
                dict.set_item("loss", loss)?;
            }

            Ok(dict.into())
        })
    }

    /// Generate text with the language model.
    ///
    /// Drives [`trustformers_core::generation::TextGenerator`] -- the same
    /// strategy-aware decoder (greedy / temperature / top-k / top-p) wired up
    /// elsewhere in this workspace's real sampling path -- by recomputing the
    /// full forward pass on the growing token sequence at each step (GPT-2's
    /// KV-cache is not yet threaded through the Rust core forward pass, so
    /// `use_cache` is left off rather than claimed).
    ///
    /// This used to append repeated GPT-2 EOS tokens (`50256`) up to
    /// `max_length`, completely ignoring `temperature`/`do_sample`, and
    /// calling that "generation".
    #[pyo3(signature = (input_ids, max_length=50, temperature=1.0, do_sample=true, top_k=None, top_p=None))]
    pub fn generate(
        &self,
        input_ids: &PyTensor,
        max_length: usize,
        temperature: f32,
        do_sample: bool,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> PyResult<PyTensor> {
        let prompt = extract_token_ids_usize(&input_ids.inner)?;
        let options = SamplingOptions {
            max_length,
            do_sample,
            temperature,
            top_k,
            top_p,
            ..SamplingOptions::default()
        };
        let mut sequences = generation::generate_with_gpt2(&self.inner, &prompt, &options)
            .map_err(|e| PyValueError::new_err(format!("Generation failed: {}", e)))?;

        build_usize_token_tensor(&sequences.swap_remove(0))
    }

    /// Get model configuration.
    #[getter]
    pub fn config(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(gpt2_config_to_dict(py, self.inner.get_config())?.into())
    }

    /// Save this model's config and parameters to `save_directory`.
    pub fn save_pretrained(&self, save_directory: &str) -> PyResult<()> {
        save_pretrained_for_model(&self.inner, save_directory, "GPT2LMHeadModel")
    }
}
