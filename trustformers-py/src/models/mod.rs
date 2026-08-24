pub(crate) mod generation;
mod inputs;
mod losses;
mod weights;

use crate::config_utils::{
    gpt2_config_to_dict, llama_config_to_dict, parse_gpt2_config, parse_llama_config,
    parse_t5_config, t5_config_to_dict,
};
use crate::tensor::PyTensor;
use scirs2_core::ndarray::{ArrayD, IxDyn}; // SciRS2 Integration Policy
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;
use generation::SamplingOptions;
use weights::{
    load_config_from_hub, load_pretrained_weights, report_weight_loading,
    save_pretrained_for_model, trustformers_error_to_py_err,
};

/// Owned Python reference alias (pyo3 0.28 removed the `PyObject` type alias from
/// the crate root; it is equivalent to `Py<PyAny>`).
type PyObject = Py<PyAny>;


use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Model, TokenizedInput};
use trustformers_models::{
    bert::{BertConfig, BertForSequenceClassification, BertModel},
    gpt2::{Gpt2Config, Gpt2LMHeadModel, Gpt2Model},
    llama::{LlamaConfig, LlamaModel},
    mamba::{MambaConfig, MambaModel},
    rwkv::{RwkvConfig, RwkvModel},
    t5::{T5Config, T5Model},
};

/// Base class for all models
#[pyclass(name = "PreTrainedModel", module = "trustformers", subclass)]
pub struct PyPreTrainedModel {
    pub config: PyObject,
}

#[pymethods]
impl PyPreTrainedModel {
    /// Save model to directory.
    ///
    /// `PreTrainedModel` itself holds only a config, never weights -- every
    /// concrete model class (`BertModel`, `GPT2Model`, `GPT2LMHeadModel`, ...)
    /// overrides `save_pretrained` with a real implementation that also
    /// exports `model.safetensors` from its own tensors. Reaching this base
    /// implementation directly (instantiating `PreTrainedModel` itself, or a
    /// future subclass that forgets to override) means there is no weight
    /// data to save, so this refuses outright instead of writing a
    /// `config.json` that looks like a complete, loadable export -- which is
    /// what the previous implementation did, additionally writing a
    /// `pytorch_model.bin.info` text file containing the literal string
    /// "Model weights would be saved here...".
    pub fn save_pretrained(&self, save_directory: &str) -> PyResult<()> {
        let _ = save_directory;
        Err(PyValueError::new_err(
            "PreTrainedModel.save_pretrained() has no model weights to save (this is the base \
             class): call save_pretrained on a concrete model subclass such as BertModel or \
             GPT2LMHeadModel instead.",
        ))
    }

    /// Get model configuration
    #[getter]
    pub fn config(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.config.clone_ref(py))
    }
}

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

// ---- Shared helpers for the state-space (RWKV / Mamba) wrappers ----

/// Extract token IDs from a tensor of integer or float token values.
fn extract_token_ids(tensor: &Tensor) -> PyResult<Vec<i64>> {
    match tensor {
        Tensor::I64(arr) => Ok(arr.iter().copied().collect()),
        Tensor::F32(arr) => Ok(arr.iter().map(|&x| x as i64).collect()),
        _ => Err(PyValueError::new_err(
            "Input tensor must contain integer token IDs",
        )),
    }
}

/// Build a [`TokenizedInput`] from the tensors a `forward` binding receives.
///
/// PyO3 boundary over [`inputs::tokenized_input_from_parts`], which holds the
/// pure (and unit-tested) conversion and validation logic.
fn tokenized_input_from_tensors(
    input_ids: &PyTensor,
    attention_mask: Option<&PyTensor>,
    token_type_ids: Option<&PyTensor>,
) -> PyResult<TokenizedInput> {
    inputs::tokenized_input_from_parts(
        &input_ids.inner,
        attention_mask.map(|tensor| &tensor.inner),
        token_type_ids.map(|tensor| &tensor.inner),
    )
    .map_err(trustformers_error_to_py_err)
}

/// Extract non-negative token IDs, for use with
/// [`trustformers_core::generation::TextGenerator`], which indexes with `usize`.
fn extract_token_ids_usize(tensor: &Tensor) -> PyResult<Vec<usize>> {
    extract_token_ids(tensor)?
        .into_iter()
        .map(|id| {
            usize::try_from(id).map_err(|_| {
                PyValueError::new_err(format!(
                    "token id {id} is negative and cannot index a vocabulary"
                ))
            })
        })
        .collect()
}

/// Build a 1-D `I64` token-id tensor from a slice of ids.
fn build_token_tensor(tokens: &[i64]) -> PyResult<Tensor> {
    Ok(Tensor::I64(
        ArrayD::from_shape_vec(IxDyn(&[tokens.len()]), tokens.to_vec())
            .map_err(|e| PyValueError::new_err(format!("Failed to create tensor: {}", e)))?,
    ))
}

/// Build a 1-D `I64` token-id [`PyTensor`] from a slice of `usize` ids, the
/// output shape [`TextGenerator::generate`] produces.
fn build_usize_token_tensor(tokens: &[usize]) -> PyResult<PyTensor> {
    let ids: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
    Ok(PyTensor {
        inner: build_token_tensor(&ids)?,
        variable: None,
    })
}

/// Argmax over the vocabulary axis of the final timestep of a `[seq, vocab]` logit tensor.
fn argmax_last_token(logits: &Tensor) -> PyResult<usize> {
    let shape = logits.shape();
    let vocab = *shape
        .last()
        .ok_or_else(|| PyValueError::new_err("Logits tensor has no dimensions"))?;
    if vocab == 0 {
        return Err(PyValueError::new_err(
            "Logits tensor has an empty vocabulary axis",
        ));
    }
    let data = logits.to_vec_f32().map_err(trustformers_error_to_py_err)?;
    let last_offset = data
        .len()
        .checked_sub(vocab)
        .ok_or_else(|| PyValueError::new_err("Logits tensor is smaller than the vocabulary size"))?;
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (j, &v) in data[last_offset..].iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = j;
        }
    }
    Ok(best_idx)
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

// Helper functions for config parsing
fn parse_bert_config(config_dict: &Bound<'_, PyAny>) -> PyResult<BertConfig> {
    let dict = config_dict.cast::<pyo3::types::PyDict>()?;

    let mut config = BertConfig::default();

    if let Ok(Some(val)) = dict.get_item("vocab_size") {
        config.vocab_size = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("hidden_size") {
        config.hidden_size = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("num_hidden_layers") {
        config.num_hidden_layers = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("num_attention_heads") {
        config.num_attention_heads = val.extract()?;
    }

    Ok(config)
}

fn config_to_dict<'py>(
    py: Python<'py>,
    config: &BertConfig,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("vocab_size", config.vocab_size)?;
    dict.set_item("hidden_size", config.hidden_size)?;
    dict.set_item("num_hidden_layers", config.num_hidden_layers)?;
    dict.set_item("num_attention_heads", config.num_attention_heads)?;
    dict.set_item("intermediate_size", config.intermediate_size)?;
    dict.set_item("hidden_act", &config.hidden_act)?;
    dict.set_item("hidden_dropout_prob", config.hidden_dropout_prob)?;
    dict.set_item(
        "attention_probs_dropout_prob",
        config.attention_probs_dropout_prob,
    )?;
    dict.set_item("max_position_embeddings", config.max_position_embeddings)?;
    dict.set_item("type_vocab_size", config.type_vocab_size)?;
    dict.set_item("initializer_range", config.initializer_range)?;
    dict.set_item("layer_norm_eps", config.layer_norm_eps)?;
    Ok(dict)
}

// Task-specific models using composition pattern

/// BERT for Sequence Classification
///
/// Wraps the real [`trustformers_models::bert::BertForSequenceClassification`]:
/// a BERT encoder, its pooler, and a `hidden_size -> num_labels` linear head,
/// all of which are bound from a checkpoint by `Model::load_pretrained` and all
/// of which run in `forward`.
///
/// The previous implementation held a bare `BertModel` plus a `classifier`
/// field that was literally `py.None()` ("Create classifier as placeholder"),
/// and its `forward` returned the pooled/`[CLS]` hidden state relabelled as
/// `logits` -- a `hidden_size`-wide vector that had never passed through any
/// classification head, so the "logits" had neither `num_labels` entries nor
/// any relation to the labels.
#[pyclass(name = "BertForSequenceClassification", module = "trustformers")]
pub struct PyBertForSequenceClassification {
    inner: BertForSequenceClassification,
    num_labels: usize,
    /// Label names by class index, from the checkpoint's `id2label` when it has
    /// one and `LABEL_0..LABEL_n` otherwise (HuggingFace's own fallback).
    labels: Vec<String>,
}

impl PyBertForSequenceClassification {
    /// The wrapped Rust model, for the classification pipeline.
    pub(crate) fn model(&self) -> &BertForSequenceClassification {
        &self.inner
    }

    /// The class labels, indexed by class id.
    pub(crate) fn labels(&self) -> &[String] {
        &self.labels
    }
}

/// HuggingFace's fallback label names for a head with no `id2label`.
pub(crate) fn default_label_names(num_labels: usize) -> Vec<String> {
    (0..num_labels).map(|index| format!("LABEL_{index}")).collect()
}

/// Read `id2label` out of a parsed `config.json`, falling back to
/// `LABEL_0..LABEL_n` for every class the mapping does not name.
fn label_names_from_config(config_value: &Value, num_labels: usize) -> Vec<String> {
    let mut labels = default_label_names(num_labels);
    if let Some(map) = config_value.get("id2label").and_then(|v| v.as_object()) {
        for (key, value) in map {
            if let (Ok(index), Some(name)) = (key.parse::<usize>(), value.as_str()) {
                if let Some(slot) = labels.get_mut(index) {
                    *slot = name.to_string();
                }
            }
        }
    }
    labels
}

#[pymethods]
impl PyBertForSequenceClassification {
    #[new]
    #[pyo3(signature = (config=None, num_labels=2))]
    pub fn new(config: Option<&Bound<'_, PyAny>>, num_labels: usize) -> PyResult<Self> {
        let bert_config = if let Some(cfg) = config {
            parse_bert_config(cfg)?
        } else {
            BertConfig::default()
        };

        let inner = BertForSequenceClassification::new(bert_config, num_labels).map_err(|e| {
            PyValueError::new_err(format!(
                "Failed to create BERT sequence-classification model: {}",
                e
            ))
        })?;

        Ok(PyBertForSequenceClassification {
            inner,
            num_labels,
            labels: default_label_names(num_labels),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (model_name_or_path, **_kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        model_name_or_path: &str,
        _kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyBertForSequenceClassification>> {
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

        // Extract number of labels from config: `num_labels` when the config
        // states one, otherwise the size of the `id2label` map that a
        // fine-tuned classifier checkpoint always carries.
        let num_labels = config_value
            .get("num_labels")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or_else(|| {
                config_value.get("id2label").and_then(|v| v.as_object()).map(|map| map.len())
            })
            .unwrap_or(2);
        let labels = label_names_from_config(&config_value, num_labels);

        // The whole task model is created and loaded, not just the encoder:
        // `BertForSequenceClassification::load_pretrained` binds the encoder
        // under `bert.` *and* the `classifier.{weight,bias}` head, so a
        // fine-tuned checkpoint's head reaches the model instead of being
        // dropped on the floor.
        let mut model = BertForSequenceClassification::new(config, num_labels)
            .map_err(|e| PyValueError::new_err(format!("Failed to create model: {}", e)))?;

        report_weight_loading(
            load_pretrained_weights(&mut model, model_name_or_path),
            model_name_or_path,
        )?;

        Py::new(
            py,
            PyBertForSequenceClassification {
                inner: model,
                num_labels,
                labels,
            },
        )
    }

    #[pyo3(signature = (input_ids, attention_mask=None, token_type_ids=None, labels=None))]
    pub fn forward(
        &self,
        input_ids: &PyTensor,
        attention_mask: Option<&PyTensor>,
        token_type_ids: Option<&PyTensor>,
        labels: Option<&PyTensor>,
    ) -> PyResult<PyObject> {
        Python::attach(|py| {
            let tokenized_input =
                tokenized_input_from_tensors(input_ids, attention_mask, token_type_ids)?;

            // Real `[1, num_labels]` logits: pooled `[CLS]` representation ->
            // the loaded linear classification head.
            let outputs = self.inner.forward(tokenized_input).map_err(|e| {
                PyValueError::new_err(format!(
                    "BERT sequence-classification forward pass failed: {}",
                    e
                ))
            })?;
            let logits = outputs.logits;

            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("logits", PyTensor::from_tensor(logits.clone()))?;

            // Calculate loss if labels provided
            if let Some(labels_tensor) = labels {
                let loss_value =
                    losses::classification_cross_entropy(&logits, &labels_tensor.inner).map_err(
                        |e| PyValueError::new_err(format!("Loss calculation failed: {}", e)),
                    )?;

                let loss = PyTensor::from_tensor(Tensor::scalar(loss_value).map_err(|e| {
                    PyValueError::new_err(format!("Failed to create loss tensor: {}", e))
                })?);
                dict.set_item("loss", loss)?;
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
        labels: Option<&PyTensor>,
    ) -> PyResult<PyObject> {
        self.forward(input_ids, attention_mask, token_type_ids, labels)
    }

    /// Save this model's config and parameters to `save_directory`.
    ///
    /// Exports the *whole* task model -- the encoder under `bert.…` plus the
    /// `classifier.{weight,bias}` head -- because the head is now a real
    /// `Linear` published by
    /// [`trustformers_models::bert::BertForSequenceClassification`]'s
    /// `named_tensors()`. While the head was a `py.None()` placeholder this
    /// could only write the encoder.
    pub fn save_pretrained(&self, save_directory: &str) -> PyResult<()> {
        save_pretrained_for_model(&self.inner, save_directory, "BertForSequenceClassification")
    }

    /// Number of classification labels this head predicts.
    #[getter]
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    /// Class labels by index (`id2label` from the checkpoint config, or
    /// `LABEL_0..LABEL_n`).
    #[getter]
    pub fn id2label(&self) -> Vec<String> {
        self.labels.clone()
    }

    /// Get model configuration.
    #[getter]
    pub fn config(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(config_to_dict(py, self.inner.get_config())?.into())
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
