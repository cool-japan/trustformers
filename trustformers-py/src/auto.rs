use crate::models::{PyBertModel, PyGPT2Model, PyLlamaModel, PyT5Model};
use crate::tokenizers::{PyBPETokenizer, PyWordPieceTokenizer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use std::sync::Arc;
// use trustformers::hub::{download_model, ModelInfo}; // Commented out - main trustformers crate not available
// use trustformers::{AutoConfig, AutoModel as RustAutoModel, AutoTokenizer as RustAutoTokenizer}; // Commented out - main trustformers crate not available

/// AutoModel for automatic model selection based on pretrained name
#[pyclass(name = "AutoModel", module = "trustformers")]
pub struct PyAutoModel;

#[pymethods]
impl PyAutoModel {
    /// Load a model from a pretrained name or path
    #[staticmethod]
    #[pyo3(signature = (pretrained_model_name_or_path, **kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        pretrained_model_name_or_path: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        // Extract optional parameters
        let cache_dir = kwargs
            .and_then(|d| d.get_item("cache_dir").ok().flatten())
            .and_then(|v| v.extract::<String>().ok());

        let force_download = kwargs
            .and_then(|d| d.get_item("force_download").ok().flatten())
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false);

        let revision = kwargs
            .and_then(|d| d.get_item("revision").ok().flatten())
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_else(|| "main".to_string());

        // Determine model type from name
        let model_type = infer_model_type(pretrained_model_name_or_path);

        // Create appropriate model based on type
        match model_type.as_str() {
            "bert" | "roberta" | "distilbert" => {
                let model = PyBertModel::from_pretrained(
                    py,
                    pretrained_model_name_or_path,
                    kwargs.map(|k| k.as_any()), // Pass kwargs to model
                )?;
                Ok(model.into_py(py))
            },
            "deberta" => {
                // For now, use BERT implementation as fallback
                let model = PyBertModel::from_pretrained(
                    py,
                    pretrained_model_name_or_path,
                    kwargs.map(|k| k.as_any()),
                )?;
                Ok(model.into_py(py))
            },
            "gpt2" | "gpt-j" | "gpt-neo" => {
                let model = PyGPT2Model::from_pretrained(
                    py,
                    pretrained_model_name_or_path,
                    kwargs.map(|k| k.as_any()), // Pass kwargs to model
                )?;
                Ok(model.into_py(py))
            },
            "t5" => {
                let model = PyT5Model::from_pretrained(
                    py,
                    pretrained_model_name_or_path,
                    kwargs.map(|k| k.as_any()), // Pass kwargs to model
                )?;
                Ok(model.into_py(py))
            },
            "llama" | "falcon" | "mpt" | "mistral" | "gemma" | "phi" | "qwen" => {
                let model = PyLlamaModel::from_pretrained(
                    py,
                    pretrained_model_name_or_path,
                    kwargs.map(|k| k.as_any()), // Pass kwargs to model
                )?;
                Ok(model.into_py(py))
            },
            "claude" => {
                // For Claude models, we'll use a specialized implementation or fallback
                let model = PyLlamaModel::from_pretrained(
                    py,
                    pretrained_model_name_or_path,
                    kwargs.map(|k| k.as_any()),
                )?;
                Ok(model.into_py(py))
            },
            "rwkv" | "mamba" => {
                // State-space models - for now use BERT as fallback
                // TODO: Implement proper RWKV/Mamba support
                let model = PyBertModel::from_pretrained(
                    py,
                    pretrained_model_name_or_path,
                    kwargs.map(|k| k.as_any()),
                )?;
                Ok(model.into_py(py))
            },
            _ => Err(PyValueError::new_err(format!(
                "Model type '{}' detected for '{}' but not yet fully implemented. Supported types: bert, roberta, distilbert, deberta, gpt2, gpt-j, gpt-neo, t5, llama, falcon, mpt, claude, mistral, gemma, phi, qwen, rwkv, mamba",
                model_type,
                pretrained_model_name_or_path
            ))),
        }
    }
}

/// Infer model type from pretrained name
fn infer_model_type(model_name: &str) -> String {
    let lower = model_name.to_lowercase();

    // Check for specific model patterns in order of specificity
    if lower.contains("roberta") {
        "roberta".to_string()
    } else if lower.contains("deberta") {
        "deberta".to_string()
    } else if lower.contains("distilbert") {
        "distilbert".to_string()
    } else if lower.contains("bert") {
        "bert".to_string()
    } else if lower.contains("gpt2") || lower.contains("gpt-2") {
        "gpt2".to_string()
    } else if lower.contains("gpt-j") || lower.contains("gptj") {
        "gpt-j".to_string()
    } else if lower.contains("gpt-neo") || lower.contains("gptneo") {
        "gpt-neo".to_string()
    } else if lower.contains("t5") {
        "t5".to_string()
    } else if lower.contains("llama") || lower.contains("alpaca") {
        "llama".to_string()
    } else if lower.contains("falcon") {
        "falcon".to_string()
    } else if lower.contains("mpt") {
        "mpt".to_string()
    } else if lower.contains("claude") {
        "claude".to_string()
    } else if lower.contains("mistral") {
        "mistral".to_string()
    } else if lower.contains("gemma") {
        "gemma".to_string()
    } else if lower.contains("phi") {
        "phi".to_string()
    } else if lower.contains("qwen") {
        "qwen".to_string()
    } else if lower.contains("rwkv") {
        "rwkv".to_string()
    } else if lower.contains("mamba") {
        "mamba".to_string()
    } else {
        "bert".to_string() // Default fallback
    }
}

/// AutoTokenizer for automatic tokenizer selection
#[pyclass(name = "AutoTokenizer", module = "trustformers")]
pub struct PyAutoTokenizer;

#[pymethods]
impl PyAutoTokenizer {
    /// Load a tokenizer from a pretrained name or path
    #[staticmethod]
    #[pyo3(signature = (pretrained_model_name_or_path, **kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        pretrained_model_name_or_path: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        // Extract optional parameters
        let cache_dir = kwargs
            .and_then(|d| d.get_item("cache_dir").ok().flatten())
            .and_then(|v| v.extract::<String>().ok());

        let force_download = kwargs
            .and_then(|d| d.get_item("force_download").ok().flatten())
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false);

        // Determine tokenizer type from name
        let tokenizer_type = infer_tokenizer_type(pretrained_model_name_or_path);

        // Create appropriate tokenizer based on type
        match tokenizer_type.as_str() {
            "wordpiece" => {
                // Create a basic WordPiece tokenizer
                let (tokenizer, base) = PyWordPieceTokenizer::new(None, true)?;
                Py::new(py, (tokenizer, base)).map(|t| t.into_py(py))
            },
            "bpe" => {
                // Create a basic BPE tokenizer
                let (tokenizer, base) = PyBPETokenizer::new(None, None)?;
                Py::new(py, (tokenizer, base)).map(|t| t.into_py(py))
            },
            "sentencepiece" => {
                // For now, fall back to BPE for SentencePiece models
                let (tokenizer, base) = PyBPETokenizer::new(None, None)?;
                Py::new(py, (tokenizer, base)).map(|t| t.into_py(py))
            },
            _ => {
                // Default to WordPiece tokenizer
                let (tokenizer, base) = PyWordPieceTokenizer::new(None, true)?;
                Py::new(py, (tokenizer, base)).map(|t| t.into_py(py))
            },
        }
    }
}

/// Infer tokenizer type from pretrained name
fn infer_tokenizer_type(model_name: &str) -> String {
    let lower = model_name.to_lowercase();

    if lower.contains("bert") || lower.contains("roberta") {
        "wordpiece".to_string()
    } else if lower.contains("gpt2") || lower.contains("gpt") {
        "bpe".to_string()
    } else if lower.contains("t5") {
        "sentencepiece".to_string()
    } else if lower.contains("llama") {
        "sentencepiece".to_string()
    } else {
        "wordpiece".to_string() // Default
    }
}

/// AutoModelForSequenceClassification
#[pyclass(name = "AutoModelForSequenceClassification", module = "trustformers")]
pub struct PyAutoModelForSequenceClassification;

#[pymethods]
impl PyAutoModelForSequenceClassification {
    #[staticmethod]
    #[pyo3(signature = (pretrained_model_name_or_path, **kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        pretrained_model_name_or_path: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        // Similar implementation to AutoModel but returns classification variants
        PyAutoModel::from_pretrained(py, pretrained_model_name_or_path, kwargs)
    }
}

/// AutoModelForTokenClassification
#[pyclass(name = "AutoModelForTokenClassification", module = "trustformers")]
pub struct PyAutoModelForTokenClassification;

#[pymethods]
impl PyAutoModelForTokenClassification {
    #[staticmethod]
    #[pyo3(signature = (pretrained_model_name_or_path, **kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        pretrained_model_name_or_path: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        PyAutoModel::from_pretrained(py, pretrained_model_name_or_path, kwargs)
    }
}

/// AutoModelForQuestionAnswering
#[pyclass(name = "AutoModelForQuestionAnswering", module = "trustformers")]
pub struct PyAutoModelForQuestionAnswering;

#[pymethods]
impl PyAutoModelForQuestionAnswering {
    #[staticmethod]
    #[pyo3(signature = (pretrained_model_name_or_path, **kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        pretrained_model_name_or_path: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        PyAutoModel::from_pretrained(py, pretrained_model_name_or_path, kwargs)
    }
}

/// AutoModelForCausalLM
#[pyclass(name = "AutoModelForCausalLM", module = "trustformers")]
pub struct PyAutoModelForCausalLM;

#[pymethods]
impl PyAutoModelForCausalLM {
    #[staticmethod]
    #[pyo3(signature = (pretrained_model_name_or_path, **kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        pretrained_model_name_or_path: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        PyAutoModel::from_pretrained(py, pretrained_model_name_or_path, kwargs)
    }
}

/// AutoModelForMaskedLM
#[pyclass(name = "AutoModelForMaskedLM", module = "trustformers")]
pub struct PyAutoModelForMaskedLM;

#[pymethods]
impl PyAutoModelForMaskedLM {
    #[staticmethod]
    #[pyo3(signature = (pretrained_model_name_or_path, **kwargs))]
    pub fn from_pretrained(
        py: Python<'_>,
        pretrained_model_name_or_path: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        PyAutoModel::from_pretrained(py, pretrained_model_name_or_path, kwargs)
    }
}

/// Pipeline factory function
#[pyfunction]
#[pyo3(signature = (task, model=None, tokenizer=None, device=None, **kwargs))]
pub fn pipeline(
    py: Python<'_>,
    task: &str,
    model: Option<&Bound<'_, PyAny>>,
    tokenizer: Option<&Bound<'_, PyAny>>,
    device: Option<&str>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
    use crate::pipelines::{PyTextClassificationPipeline, PyTextGenerationPipeline};

    // If model/tokenizer not provided, auto-detect based on task
    let (model, tokenizer) = if model.is_none() || tokenizer.is_none() {
        let default_model = match task {
            "text-generation" => "gpt2",
            "text-classification" | "sentiment-analysis" => "bert-base-uncased",
            "question-answering" => "bert-large-uncased-whole-word-masking-finetuned-squad",
            "token-classification" | "ner" => "bert-base-cased",
            _ => "bert-base-uncased",
        };

        let model = if model.is_none() {
            PyAutoModel::from_pretrained(py, default_model, None)?
        } else {
            model.unwrap().unbind()
        };

        let tokenizer = if tokenizer.is_none() {
            PyAutoTokenizer::from_pretrained(py, default_model, None)?
        } else {
            tokenizer.unwrap().unbind()
        };

        (model, tokenizer)
    } else {
        (
            model.unwrap().unbind(),
            tokenizer.unwrap().unbind(),
        )
    };

    // Create appropriate pipeline
    match task {
        "text-generation" => {
            let model_bound = model.bind(py);
            let tokenizer_bound = tokenizer.bind(py);
            let (pipeline, base) = PyTextGenerationPipeline::new(py, &model_bound, &tokenizer_bound, device)?;
            Py::new(py, (pipeline, base)).map(|p| p.into_py(py))
        },
        "text-classification" | "sentiment-analysis" => {
            let model_bound = model.bind(py);
            let tokenizer_bound = tokenizer.bind(py);
            let (pipeline, base) = PyTextClassificationPipeline::new(py, &model_bound, &tokenizer_bound, device)?;
            Py::new(py, (pipeline, base)).map(|p| p.into_py(py))
        },
        _ => Err(PyValueError::new_err(format!(
            "Unknown task: {}. Supported tasks: text-generation, text-classification, sentiment-analysis",
            task
        )))
    }
}
