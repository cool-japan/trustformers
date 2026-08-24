use crate::models::{
    PyBertForQuestionAnswering, PyBertForSequenceClassification, PyBertForTokenClassification,
    PyBertModel, PyGPT2LMHeadModel, PyLlamaModel, PyMambaModel, PyRwkvModel, PyT5Model,
};
use crate::tokenizers::{PyBPETokenizer, PyWordPieceTokenizer};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
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

/// Whether `model_type` (an [`infer_model_type`] result) is one of the
/// architectures this crate's `Bert*` task wrappers cover.
///
/// `roberta`/`distilbert`/`deberta` are included because [`infer_model_type`]
/// already collapses them to those names for `AutoModel`'s own routing, on
/// the basis that this crate loads all four through `BertModel`/`BertConfig`
/// -- there is no separate `RobertaModel`/`DistilBertModel` implementation to
/// disagree with.
fn is_bert_family(model_type: &str) -> bool {
    matches!(model_type, "bert" | "roberta" | "distilbert" | "deberta")
}

#[cfg(test)]
mod auto_model_for_task_tests {
    use super::*;

    #[test]
    fn bert_family_checkpoints_are_accepted() {
        for name in ["bert-base-uncased", "roberta-large", "distilbert-base", "deberta-v3-base"] {
            assert!(
                bert_family_gap_message(
                    name,
                    "AutoModelForTokenClassification",
                    "BertForTokenClassification"
                )
                .is_none(),
                "'{name}' should resolve to a BERT-family checkpoint"
            );
        }
    }

    /// The regression this whole helper exists for: before this fix,
    /// `AutoModelForTokenClassification.from_pretrained("gpt2")` (or any
    /// non-BERT checkpoint) silently returned a bare model with no
    /// token-classification head at all, via `AutoModel::from_pretrained`.
    /// It must now be a structured refusal instead.
    #[test]
    fn non_bert_checkpoints_are_refused_not_silently_mislabeled() {
        for name in ["gpt2-medium", "t5-base", "meta-llama/Llama-2-7b", "mamba-130m", "RWKV-4-169m"] {
            let message = bert_family_gap_message(
                name,
                "AutoModelForQuestionAnswering",
                "BertForQuestionAnswering",
            );
            assert!(message.is_some(), "'{name}' must be refused, not silently mislabeled");
        }
    }

    /// The refusal message must name both the checkpoint's inferred type and
    /// the wrapper that cannot serve it, so a caller can see immediately why.
    #[test]
    fn the_refusal_names_the_inferred_type_and_the_missing_wrapper() {
        let message = bert_family_gap_message(
            "gpt2",
            "AutoModelForTokenClassification",
            "BertForTokenClassification",
        )
        .expect("gpt2 is not BERT-family");
        assert!(message.contains("gpt2"), "message must name the inferred type: {message}");
        assert!(
            message.contains("BertForTokenClassification"),
            "message must name the missing wrapper: {message}"
        );
    }

    /// The `PyResult`-returning wrapper's control flow must match the pure
    /// function's, at least structurally (`Ok`/`Err`, not the message text --
    /// see `bert_family_gap_message`'s doc comment for why the text itself is
    /// tested there instead).
    #[test]
    fn require_bert_family_checkpoint_ok_err_matches_the_pure_function() {
        assert!(
            require_bert_family_checkpoint(
                "bert-base-uncased",
                "AutoModelForSequenceClassification",
                "BertForSequenceClassification"
            )
            .is_ok()
        );
        assert!(
            require_bert_family_checkpoint(
                "gpt2",
                "AutoModelForSequenceClassification",
                "BertForSequenceClassification"
            )
            .is_err()
        );
    }

    #[test]
    fn is_bert_family_covers_exactly_the_four_shared_architectures() {
        for name in ["bert", "roberta", "distilbert", "deberta"] {
            assert!(is_bert_family(name), "'{name}' must be BERT-family");
        }
        for name in ["gpt2", "t5", "llama", "rwkv", "mamba", "gpt-j", "mistral", ""] {
            assert!(!is_bert_family(name), "'{name}' must not be BERT-family");
        }
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

/// Refuse `pretrained_model_name_or_path` unless [`infer_model_type`] resolves
/// it to one of the BERT-family architectures the task-specific wrapper this
/// factory constructs actually wraps.
///
/// `AutoModelForSequenceClassification`/`ForTokenClassification`/
/// `ForQuestionAnswering` used to all delegate straight to
/// `AutoModel::from_pretrained`, which returns a *bare* `BertModel` (or
/// `GPT2Model`, `T5Model`, ...) with no task head at all. That meant
/// `AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")`
/// handed back an object that looked like a token-classification model --
/// same `PreTrainedModel`-shaped Python surface, loaded from the very same
/// checkpoint -- but carried no classifier weights and could never produce
/// per-token logits, however its `forward()` was called: the checkpoint's
/// `classifier.{weight,bias}` tensors were silently dropped on the floor by
/// the loader `AutoModel` actually uses (`BertModel`, encoder-only).
///
/// # Errors
///
/// Returns a structured `NotImplementedError` naming the inferred
/// (unsupported) architecture, instead of silently handing back a checkpoint
/// bound onto the wrong model.
/// The pure message-construction half of [`require_bert_family_checkpoint`],
/// returning `None` when `pretrained_model_name_or_path` is BERT-family (no
/// gap to report) and `Some(message)` otherwise.
///
/// Split out from the `PyResult`-returning wrapper so it is unit-testable
/// with a plain `cargo test`: constructing a `PyErr`'s `Display` output
/// requires an initialized Python interpreter (this crate's `cargo test`
/// binary does not embed one -- see `pipelines::scoring`'s module doc for the
/// same reason its logic is kept free of the Python C API), but building the
/// `String` this function returns does not.
fn bert_family_gap_message(
    pretrained_model_name_or_path: &str,
    factory_name: &str,
    wrapper_name: &str,
) -> Option<String> {
    let model_type = infer_model_type(pretrained_model_name_or_path);
    if is_bert_family(&model_type) {
        return None;
    }
    Some(format!(
        "{factory_name}.from_pretrained('{pretrained_model_name_or_path}') resolves to model \
         type '{model_type}', but this crate's only {factory_name} wrapper is {wrapper_name} \
         (BERT-family: bert/roberta/distilbert/deberta share BertModel's architecture here). \
         There is no {wrapper_name}-equivalent head implemented for '{model_type}' in this \
         crate; returning a bare encoder for it, relabelled as a task model, would silently \
         drop the checkpoint's task head."
    ))
}

/// Refuse `pretrained_model_name_or_path` unless it is BERT-family; see
/// [`bert_family_gap_message`] for the logic and why it is split out this way.
fn require_bert_family_checkpoint(
    pretrained_model_name_or_path: &str,
    factory_name: &str,
    wrapper_name: &str,
) -> PyResult<()> {
    match bert_family_gap_message(pretrained_model_name_or_path, factory_name, wrapper_name) {
        None => Ok(()),
        Some(message) => Err(PyNotImplementedError::new_err(message)),
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
        require_bert_family_checkpoint(
            pretrained_model_name_or_path,
            "AutoModelForSequenceClassification",
            "BertForSequenceClassification",
        )?;
        PyBertForSequenceClassification::from_pretrained(
            py,
            pretrained_model_name_or_path,
            kwargs.map(|k| k.as_any()),
        )?
        .into_py_any(py)
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
        require_bert_family_checkpoint(
            pretrained_model_name_or_path,
            "AutoModelForTokenClassification",
            "BertForTokenClassification",
        )?;
        PyBertForTokenClassification::from_pretrained(
            py,
            pretrained_model_name_or_path,
            kwargs.map(|k| k.as_any()),
        )?
        .into_py_any(py)
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
        require_bert_family_checkpoint(
            pretrained_model_name_or_path,
            "AutoModelForQuestionAnswering",
            "BertForQuestionAnswering",
        )?;
        PyBertForQuestionAnswering::from_pretrained(
            py,
            pretrained_model_name_or_path,
            kwargs.map(|k| k.as_any()),
        )?
        .into_py_any(py)
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
/// pipeline class, and lets that class decide whether the given `model`
/// carries the head its task needs -- a mismatch is a `TypeError` from the
/// pipeline class's own `__init__`, not a surprise at call time. All four
/// tasks (`text-generation`, `text-classification`, `token-classification`,
/// `question-answering`) run real inference today; `question-answering`
/// additionally requires its tokenizer to be a `WordPieceTokenizer` (see
/// `pipelines::qa_requires_wordpiece`).
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
        PipelineTask::QuestionAnswering => "bert-large-uncased-whole-word-masking-finetuned-squad",
    };

    let model = match model {
        Some(model) => model.clone().unbind(),
        None => match resolved_task {
            PipelineTask::TextClassification => {
                PyBertForSequenceClassification::from_pretrained(py, default_model, None)?
                    .into_py_any(py)?
            },
            // Both span pipelines need their task head, not a bare encoder:
            // `PyAutoModel::from_pretrained` resolves to a headless
            // `BertModel`, which the `.cast::<PyBertFor...>()` inside
            // `PyTokenClassificationPipeline::new`/
            // `PyQuestionAnsweringPipeline::new` would then always reject.
            PipelineTask::TokenClassification => {
                PyBertForTokenClassification::from_pretrained(py, default_model, None)?
                    .into_py_any(py)?
            },
            PipelineTask::QuestionAnswering => {
                PyBertForQuestionAnswering::from_pretrained(py, default_model, None)?
                    .into_py_any(py)?
            },
            PipelineTask::TextGeneration => PyAutoModel::from_pretrained(py, default_model, None)?,
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
            let parts = PyTokenClassificationPipeline::new(model_bound, tokenizer_bound, device)?;
            Py::new(py, parts).and_then(|pipeline| pipeline.into_py_any(py))
        },
        PipelineTask::QuestionAnswering => {
            let parts = PyQuestionAnsweringPipeline::new(model_bound, tokenizer_bound, device)?;
            Py::new(py, parts).and_then(|pipeline| pipeline.into_py_any(py))
        },
    }
}
