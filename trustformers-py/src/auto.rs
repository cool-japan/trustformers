use crate::models::{
    PyBertForSequenceClassification, PyBertModel, PyGPT2LMHeadModel, PyLlamaModel, PyMambaModel,
    PyRwkvModel, PyT5Model,
};
use crate::tokenizers::{PyBPETokenizer, PyWordPieceTokenizer};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::IntoPyObjectExt;

/// Owned Python reference alias (pyo3 0.28 removed the `PyObject` type alias from
/// the crate root; it is equivalent to `Py<PyAny>`).
type PyObject = Py<PyAny>;
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
        let _cache_dir = kwargs
            .and_then(|d| d.get_item("cache_dir").ok().flatten())
            .and_then(|v| v.extract::<String>().ok());

        let _force_download = kwargs
            .and_then(|d| d.get_item("force_download").ok().flatten())
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false);

        let _revision = kwargs
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
                model.into_py_any(py)
            },
            "deberta" => {
                // For now, use BERT implementation as fallback
                let model = PyBertModel::from_pretrained(
                    py,
                    pretrained_model_name_or_path,
                    kwargs.map(|k| k.as_any()),
                )?;
                model.into_py_any(py)
            },
            "gpt2" | "gpt-j" | "gpt-neo" => {
                // `AutoModel`/`AutoModelForCausalLM` callers expect `.generate()` to
                // work, which requires the language-modeling head; the headless
                // `PyGPT2Model` deliberately has no `.generate()` (see its doc
                // comment -- HuggingFace's own `GPT2Model` has none either).
                let model = PyGPT2LMHeadModel::from_pretrained(
                    py,
                    pretrained_model_name_or_path,
                    kwargs.map(|k| k.as_any()), // Pass kwargs to model
                )?;
                model.into_py_any(py)
            },
            "t5" => {
                let model = PyT5Model::from_pretrained(
                    py,
                    pretrained_model_name_or_path,
                    kwargs.map(|k| k.as_any()), // Pass kwargs to model
                )?;
                model.into_py_any(py)
            },
            "llama" | "falcon" | "mpt" | "mistral" | "gemma" | "phi" | "qwen" => {
                let model = PyLlamaModel::from_pretrained(
                    py,
                    pretrained_model_name_or_path,
                    kwargs.map(|k| k.as_any()), // Pass kwargs to model
                )?;
                model.into_py_any(py)
            },
            "claude" => {
                // For Claude models, we'll use a specialized implementation or fallback
                let model = PyLlamaModel::from_pretrained(
                    py,
                    pretrained_model_name_or_path,
                    kwargs.map(|k| k.as_any()),
                )?;
                model.into_py_any(py)
            },
            "rwkv" => {
                let model = PyRwkvModel::from_pretrained(
                    py,
                    pretrained_model_name_or_path,
                    kwargs.map(|k| k.as_any()),
                )?;
                model.into_py_any(py)
            },
            "mamba" => {
                let model = PyMambaModel::from_pretrained(
                    py,
                    pretrained_model_name_or_path,
                    kwargs.map(|k| k.as_any()),
                )?;
                model.into_py_any(py)
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
        let _cache_dir = kwargs
            .and_then(|d| d.get_item("cache_dir").ok().flatten())
            .and_then(|v| v.extract::<String>().ok());

        let _force_download = kwargs
            .and_then(|d| d.get_item("force_download").ok().flatten())
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false);

        // Determine tokenizer type from name
        let tokenizer_type = infer_tokenizer_type(pretrained_model_name_or_path);

        // Every arm loads the tokenizer's real files from `pretrained_model_name_or_path`.
        // This used to ignore the path entirely and return a freshly constructed,
        // *empty* tokenizer -- a five-token `[PAD]/[UNK]/[CLS]/[SEP]/[MASK]`
        // vocabulary for WordPiece, and no vocabulary and no merges at all for
        // BPE -- while reporting that the requested checkpoint had been loaded.
        // Under those tokenizers every real word encodes to `[UNK]`, so anything
        // downstream (generation, classification) was operating on noise.
        match tokenizer_type.as_str() {
            "wordpiece" => PyWordPieceTokenizer::from_pretrained(
                py,
                pretrained_model_name_or_path,
                None,
            )?
            .into_py_any(py),
            "bpe" => {
                PyBPETokenizer::from_pretrained(py, pretrained_model_name_or_path, None)?
                    .into_py_any(py)
            },
            // T5/LLaMA checkpoints ship a SentencePiece model, and this crate
            // implements WordPiece and BPE only. Loading one of those with the
            // BPE reader (what this used to do) produces a tokenizer that
            // silently disagrees with the checkpoint it claims to serve.
            other => Err(PyNotImplementedError::new_err(format!(
                "no {other} tokenizer is implemented in this crate, so \
                 '{pretrained_model_name_or_path}' cannot be loaded. Available: WordPieceTokenizer \
                 (vocab.txt / vocab.json) and BPETokenizer (vocab.json + merges.txt)."
            ))),
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
    } else if lower.contains("t5") || lower.contains("llama") {
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

/// Pipeline factory function.
///
/// Routes every task name [`crate::pipelines::canonical_task`] knows to its
/// pipeline class, and lets that class decide whether it can be built: the
/// two span-level pipelines refuse construction with a structured
/// `NotImplementedError` rather than returning invented spans, and the two
/// real pipelines refuse a model that does not carry the head their task
/// needs.
///
/// This used to route only `text-generation` and `text-classification`, so
/// `pipeline("ner", ...)` reported "Unknown task" even though a (fake)
/// `TokenClassificationPipeline` existed. A second, unreachable copy of this
/// factory also lived in `pipelines.rs`, never registered with the module and
/// so never callable from Python; it has been deleted rather than left to
/// drift out of sync with this one.
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
    let _ = kwargs;
    use crate::pipelines::{
        canonical_task, PipelineTask, PyQuestionAnsweringPipeline, PyTextClassificationPipeline,
        PyTextGenerationPipeline, PyTokenClassificationPipeline, KNOWN_TASKS,
    };

    let resolved_task = canonical_task(task).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Unknown task: {task}. Supported tasks: {}",
            KNOWN_TASKS.join(", ")
        ))
    })?;

    // A default checkpoint name is only useful if it can actually be loaded.
    // `from_pretrained` resolves a *local* path (this crate has no Hub
    // downloader), so a bare "gpt2" cannot be found and the error says so --
    // which is better than the previous behaviour of quietly building a
    // pipeline around a randomly initialised model.
    let default_model = match resolved_task {
        PipelineTask::TextGeneration => "gpt2",
        PipelineTask::TextClassification => "bert-base-uncased",
        PipelineTask::TokenClassification => "bert-base-cased",
        PipelineTask::QuestionAnswering => {
            "bert-large-uncased-whole-word-masking-finetuned-squad"
        },
    };

    let model = match model {
        Some(model) => model.clone().unbind(),
        None => match resolved_task {
            PipelineTask::TextClassification => {
                PyBertForSequenceClassification::from_pretrained(py, default_model, None)?
                    .into_py_any(py)?
            },
            _ => PyAutoModel::from_pretrained(py, default_model, None)?,
        },
    };
    let tokenizer = match tokenizer {
        Some(tokenizer) => tokenizer.clone().unbind(),
        None => PyAutoTokenizer::from_pretrained(py, default_model, None)?,
    };

    let model_bound = model.bind(py);
    let tokenizer_bound = tokenizer.bind(py);

    match resolved_task {
        PipelineTask::TextGeneration => {
            let parts = PyTextGenerationPipeline::new(model_bound, tokenizer_bound, device)?;
            Py::new(py, parts).and_then(|pipeline| pipeline.into_py_any(py))
        },
        PipelineTask::TextClassification => {
            let parts = PyTextClassificationPipeline::new(model_bound, tokenizer_bound, device)?;
            Py::new(py, parts).and_then(|pipeline| pipeline.into_py_any(py))
        },
        PipelineTask::TokenClassification => {
            let parts = PyTokenClassificationPipeline::new(&PyTuple::empty(py), None)?;
            Py::new(py, parts).and_then(|pipeline| pipeline.into_py_any(py))
        },
        PipelineTask::QuestionAnswering => {
            let parts = PyQuestionAnsweringPipeline::new(&PyTuple::empty(py), None)?;
            Py::new(py, parts).and_then(|pipeline| pipeline.into_py_any(py))
        },
    }
}
