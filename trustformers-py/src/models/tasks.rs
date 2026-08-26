//! BERT task-head wrappers: `BertForSequenceClassification`,
//! `BertForTokenClassification`, `BertForQuestionAnswering` -- each a real
//! encoder-plus-head forward pass with a real loss when labels/positions are
//! given. Split out of `models/mod.rs` (2026-08-24, `py-followups`, keeping
//! this crate's files under the 2000-line policy); behavior is unchanged,
//! this is a pure code-motion.

use super::helpers::{config_to_dict, parse_bert_config, tokenized_input_from_tensors};
use super::losses;
use super::weights::{
    load_config_from_hub, load_pretrained_weights, report_weight_loading,
    save_pretrained_for_model,
};
use crate::tensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Model;
use trustformers_models::bert::{
    BertConfig, BertForQuestionAnswering, BertForSequenceClassification,
    BertForTokenClassification,
};

/// Owned Python reference alias (pyo3 0.28 removed the `PyObject` type alias from
/// the crate root; it is equivalent to `Py<PyAny>`).
type PyObject = Py<PyAny>;

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


/// BERT for Token Classification (e.g. named-entity recognition)
///
/// Wraps the real [`trustformers_models::bert::BertForTokenClassification`]: a
/// BERT encoder plus a `hidden_size -> num_labels` linear head applied at
/// *every* sequence position, unlike [`PyBertForSequenceClassification`]'s
/// head, which runs once on the pooled `[CLS]` representation.
///
/// The `token-classification`/`ner` pipeline
/// (`crate::pipelines::PyTokenClassificationPipeline`) wraps this class
/// directly: real per-token logits, aggregated with
/// `aggregation_strategy='simple'` and reported at real (converted)
/// character offsets. Call this class directly instead when per-token logits
/// -- without aggregation or offset conversion -- are what's needed.
#[pyclass(name = "BertForTokenClassification", module = "trustformers")]
pub struct PyBertForTokenClassification {
    inner: BertForTokenClassification,
    num_labels: usize,
    /// Label names by class index, from the checkpoint's `id2label` when it has
    /// one and `LABEL_0..LABEL_n` otherwise (HuggingFace's own fallback).
    labels: Vec<String>,
}

impl PyBertForTokenClassification {
    /// The wrapped Rust model, for the `token-classification` pipeline.
    pub(crate) fn model(&self) -> &BertForTokenClassification {
        &self.inner
    }

    /// The class labels, indexed by class id.
    pub(crate) fn labels(&self) -> &[String] {
        &self.labels
    }
}

#[pymethods]
impl PyBertForTokenClassification {
    #[new]
    #[pyo3(signature = (config=None, num_labels=2))]
    pub fn new(config: Option<&Bound<'_, PyAny>>, num_labels: usize) -> PyResult<Self> {
        let bert_config = if let Some(cfg) = config {
            parse_bert_config(cfg)?
        } else {
            BertConfig::default()
        };

        let inner = BertForTokenClassification::new(bert_config, num_labels).map_err(|e| {
            PyValueError::new_err(format!(
                "Failed to create BERT token-classification model: {}",
                e
            ))
        })?;

        Ok(PyBertForTokenClassification {
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
    ) -> PyResult<Py<PyBertForTokenClassification>> {
        // Load config from a local config.json (see `load_config_from_hub` docs)
        let config_json = load_config_from_hub(model_name_or_path, None)
            .map_err(|e| PyValueError::new_err(format!("Failed to load config: {}", e)))?;

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

        // Number of labels: `num_labels` when the config states one, otherwise
        // the size of the `id2label` map every fine-tuned NER checkpoint carries.
        let num_labels = config_value
            .get("num_labels")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or_else(|| {
                config_value.get("id2label").and_then(|v| v.as_object()).map(|map| map.len())
            })
            .unwrap_or(2);
        let labels = label_names_from_config(&config_value, num_labels);

        // `BertForTokenClassification::load_pretrained_report` binds the
        // encoder under `bert.` *and* the `classifier.{weight,bias}` head, so
        // a fine-tuned checkpoint's head reaches the model.
        let mut model = BertForTokenClassification::new(config, num_labels)
            .map_err(|e| PyValueError::new_err(format!("Failed to create model: {}", e)))?;

        report_weight_loading(
            load_pretrained_weights(&mut model, model_name_or_path),
            model_name_or_path,
        )?;

        Py::new(
            py,
            PyBertForTokenClassification {
                inner: model,
                num_labels,
                labels,
            },
        )
    }

    /// Forward pass: real per-token logits from the loaded encoder and
    /// classification head.
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

            // Real `[1, seq_len, num_labels]` logits: every sequence position's
            // hidden state passed through the loaded linear head.
            let outputs = self.inner.forward(tokenized_input).map_err(|e| {
                PyValueError::new_err(format!(
                    "BERT token-classification forward pass failed: {}",
                    e
                ))
            })?;
            let logits = outputs.logits;

            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("logits", PyTensor::from_tensor(logits.clone()))?;

            // One label per sequence position; `classification_cross_entropy`
            // treats every row of the trailing `num_labels` axis as its own
            // classification example, which is exactly HuggingFace's
            // `CrossEntropyLoss()(logits.view(-1, num_labels), labels.view(-1))`.
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
    /// Exports the encoder under `bert.…` plus the `classifier.{weight,bias}`
    /// head, which is a real `Linear` published by
    /// [`trustformers_models::bert::BertForTokenClassification`]'s
    /// `named_tensors()`.
    pub fn save_pretrained(&self, save_directory: &str) -> PyResult<()> {
        save_pretrained_for_model(&self.inner, save_directory, "BertForTokenClassification")
    }

    /// Number of classification labels this head predicts, per token.
    #[getter]
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    /// Label names by index (`id2label` from the checkpoint config, or
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


/// BERT for Question Answering (extractive span prediction)
///
/// Wraps the real [`trustformers_models::bert::BertForQuestionAnswering`]: a
/// BERT encoder plus a `hidden_size -> 2` linear head, split into
/// `start_logits`/`end_logits` -- one score per sequence position for "does
/// the answer start here" and "does the answer end here".
///
/// The `question-answering` pipeline
/// (`crate::pipelines::PyQuestionAnsweringPipeline`) wraps this class
/// directly: real `start_logits`/`end_logits`, extracted via real
/// joint-argmax search and reported at real (converted) character offsets.
/// That pipeline requires a `WordPieceTokenizer` specifically -- see
/// `pipelines::qa_requires_wordpiece` for why a `BPETokenizer` is rejected.
#[pyclass(name = "BertForQuestionAnswering", module = "trustformers")]
pub struct PyBertForQuestionAnswering {
    inner: BertForQuestionAnswering,
}

impl PyBertForQuestionAnswering {
    /// The wrapped Rust model, for the `question-answering` pipeline.
    pub(crate) fn model(&self) -> &BertForQuestionAnswering {
        &self.inner
    }
}

#[pymethods]
impl PyBertForQuestionAnswering {
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(config: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let bert_config = if let Some(cfg) = config {
            parse_bert_config(cfg)?
        } else {
            BertConfig::default()
        };

        let inner = BertForQuestionAnswering::new(bert_config).map_err(|e| {
            PyValueError::new_err(format!(
                "Failed to create BERT question-answering model: {}",
                e
            ))
        })?;

        Ok(PyBertForQuestionAnswering { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (model_name_or_path, **_kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        model_name_or_path: &str,
        _kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyBertForQuestionAnswering>> {
        let config_json = load_config_from_hub(model_name_or_path, None)
            .map_err(|e| PyValueError::new_err(format!("Failed to load config: {}", e)))?;

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

        // `BertForQuestionAnswering::load_pretrained_report` binds the encoder
        // under `bert.` *and* the `qa_outputs.{weight,bias}` head, so a
        // fine-tuned SQuAD checkpoint's head reaches the model.
        let mut model = BertForQuestionAnswering::new(config)
            .map_err(|e| PyValueError::new_err(format!("Failed to create model: {}", e)))?;

        report_weight_loading(
            load_pretrained_weights(&mut model, model_name_or_path),
            model_name_or_path,
        )?;

        Py::new(py, PyBertForQuestionAnswering { inner: model })
    }

    /// Forward pass: real `start_logits`/`end_logits` from the loaded encoder
    /// and span-prediction head.
    ///
    /// `start_positions`/`end_positions` must be given together (HuggingFace's
    /// own contract: a loss needs both ends of the span) or not at all.
    #[pyo3(signature = (input_ids, attention_mask=None, token_type_ids=None, start_positions=None, end_positions=None))]
    pub fn forward(
        &self,
        input_ids: &PyTensor,
        attention_mask: Option<&PyTensor>,
        token_type_ids: Option<&PyTensor>,
        start_positions: Option<&PyTensor>,
        end_positions: Option<&PyTensor>,
    ) -> PyResult<PyObject> {
        Python::attach(|py| {
            let tokenized_input =
                tokenized_input_from_tensors(input_ids, attention_mask, token_type_ids)?;

            let outputs = self.inner.forward(tokenized_input).map_err(|e| {
                PyValueError::new_err(format!(
                    "BERT question-answering forward pass failed: {}",
                    e
                ))
            })?;

            let dict = pyo3::types::PyDict::new(py);
            dict.set_item(
                "start_logits",
                PyTensor::from_tensor(outputs.start_logits.clone()),
            )?;
            dict.set_item(
                "end_logits",
                PyTensor::from_tensor(outputs.end_logits.clone()),
            )?;

            match (start_positions, end_positions) {
                (Some(start), Some(end)) => {
                    let loss_value = losses::qa_span_cross_entropy(
                        &outputs.start_logits,
                        &outputs.end_logits,
                        &start.inner,
                        &end.inner,
                    )
                    .map_err(|e| PyValueError::new_err(format!("Loss calculation failed: {}", e)))?;

                    let loss = PyTensor::from_tensor(Tensor::scalar(loss_value).map_err(|e| {
                        PyValueError::new_err(format!("Failed to create loss tensor: {}", e))
                    })?);
                    dict.set_item("loss", loss)?;
                },
                (None, None) => {},
                _ => {
                    return Err(PyValueError::new_err(
                        "start_positions and end_positions must be given together: a span loss \
                         needs both ends of the answer",
                    ))
                },
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
        start_positions: Option<&PyTensor>,
        end_positions: Option<&PyTensor>,
    ) -> PyResult<PyObject> {
        self.forward(input_ids, attention_mask, token_type_ids, start_positions, end_positions)
    }

    /// Save this model's config and parameters to `save_directory`.
    ///
    /// Exports the encoder under `bert.…` plus the `qa_outputs.{weight,bias}`
    /// head, a real `Linear` published by
    /// [`trustformers_models::bert::BertForQuestionAnswering`]'s
    /// `named_tensors()`.
    pub fn save_pretrained(&self, save_directory: &str) -> PyResult<()> {
        save_pretrained_for_model(&self.inner, save_directory, "BertForQuestionAnswering")
    }

    /// Get model configuration.
    #[getter]
    pub fn config(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(config_to_dict(py, self.inner.get_config())?.into())
    }
}
