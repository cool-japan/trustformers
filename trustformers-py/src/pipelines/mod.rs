//! HuggingFace-shaped task pipelines, backed by real models.
//!
//! Every pipeline here used to return canned data. `text-generation` answered
//! `format!("{text} [Generated continuation]")` with a hardcoded
//! `score: 0.95`; `text-classification` answered `POSITIVE 0.7 / NEGATIVE 0.3`
//! for every input; `token-classification` answered a `B-PER` entity named
//! `"John"` at characters 0..4 whatever the text was; `question-answering`
//! answered the literal string `"Example answer"`. None of them touched the
//! `model` or `tokenizer` they were constructed with, and every generation
//! argument (`max_length`, `temperature`, `top_k`, ...) was discarded by a
//! `let _ = (...)`.
//!
//! What replaces them:
//!
//! * `text-generation` tokenizes with the pipeline's real tokenizer, decodes
//!   with [`trustformers_core::generation::TextGenerator`] over a real GPT-2
//!   language-model head, and detokenizes the result.
//! * `text-classification` runs a real
//!   [`trustformers_models::bert::BertForSequenceClassification`] forward pass
//!   and reports a real softmax over its logits, labelled from the
//!   checkpoint's `id2label`.
//! * `token-classification` and `question-answering` refuse construction with
//!   a structured `NotImplementedError` that says exactly what is missing --
//!   see [`PyTokenClassificationPipeline`].

mod scoring;

use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::models::generation::{continuation_tokens, generate_with_gpt2, SamplingOptions};
use crate::models::{PyBertForSequenceClassification, PyGPT2LMHeadModel};
use crate::tokenizers::{PyBPETokenizer, PyWordPieceTokenizer};
use scoring::{classify_with_bert, ScoredLabel};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Owned Python reference alias (pyo3 0.28 removed the `PyObject` type alias from
/// the crate root; it is equivalent to `Py<PyAny>`).
type PyObject = Py<PyAny>;

// ---------------------------------------------------------------------------
// Task routing
// ---------------------------------------------------------------------------

/// The pipeline tasks this crate recognises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PipelineTask {
    /// Autoregressive continuation with a language-model head.
    TextGeneration,
    /// Whole-sequence classification.
    TextClassification,
    /// Per-token classification (named-entity recognition).
    TokenClassification,
    /// Extractive question answering.
    QuestionAnswering,
}

/// Resolve a task name (including HuggingFace's aliases) to a
/// [`PipelineTask`].
pub(crate) fn canonical_task(task: &str) -> Option<PipelineTask> {
    match task {
        "text-generation" => Some(PipelineTask::TextGeneration),
        "text-classification" | "sentiment-analysis" => Some(PipelineTask::TextClassification),
        "token-classification" | "ner" => Some(PipelineTask::TokenClassification),
        "question-answering" => Some(PipelineTask::QuestionAnswering),
        _ => None,
    }
}

/// Every task name [`canonical_task`] accepts, for error messages.
pub(crate) const KNOWN_TASKS: &[&str] = &[
    "text-generation",
    "text-classification",
    "sentiment-analysis",
    "token-classification",
    "ner",
    "question-answering",
];

// ---------------------------------------------------------------------------
// Tokenizer handle
// ---------------------------------------------------------------------------

/// A tokenizer resolved to one of this crate's concrete implementations.
///
/// Resolved once, at pipeline construction, so `__call__` cannot be handed an
/// object that merely looks like a tokenizer -- and so a mismatch is reported
/// where the caller can still act on it, as HuggingFace does.
enum PipelineTokenizer {
    /// WordPiece, as BERT-family checkpoints use.
    WordPiece(Py<PyWordPieceTokenizer>),
    /// Byte-pair encoding, as GPT-2 uses.
    Bpe(Py<PyBPETokenizer>),
}

impl PipelineTokenizer {
    /// Resolve a Python object to a supported tokenizer.
    fn resolve(tokenizer: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(wordpiece) = tokenizer.cast::<PyWordPieceTokenizer>() {
            return Ok(Self::WordPiece(wordpiece.clone().unbind()));
        }
        if let Ok(bpe) = tokenizer.cast::<PyBPETokenizer>() {
            return Ok(Self::Bpe(bpe.clone().unbind()));
        }
        Err(PyTypeError::new_err(format!(
            "pipeline tokenizer must be a trustformers WordPieceTokenizer or BPETokenizer, got \
             {}; this crate's pipelines call the Rust tokenizer directly and cannot drive an \
             arbitrary Python object",
            type_name(tokenizer)
        )))
    }

    /// Tokenize `text` with the real tokenizer.
    fn encode(&self, py: Python<'_>, text: &str) -> PyResult<TokenizedInput> {
        let encoded = match self {
            Self::WordPiece(tokenizer) => tokenizer.try_borrow(py)?.tokenizer().encode(text),
            Self::Bpe(tokenizer) => tokenizer.try_borrow(py)?.tokenizer().encode(text),
        };
        encoded.map_err(|e| PyValueError::new_err(format!("Tokenization failed: {e}")))
    }

    /// Detokenize `ids` with the real tokenizer.
    fn decode(&self, py: Python<'_>, ids: &[u32]) -> PyResult<String> {
        let decoded = match self {
            Self::WordPiece(tokenizer) => tokenizer.try_borrow(py)?.tokenizer().decode(ids),
            Self::Bpe(tokenizer) => tokenizer.try_borrow(py)?.tokenizer().decode(ids),
        };
        decoded.map_err(|e| PyValueError::new_err(format!("Detokenization failed: {e}")))
    }
}

/// The Python type name of `object`, for error messages.
fn type_name(object: &Bound<'_, PyAny>) -> String {
    object
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "<unknown type>".to_string())
}

/// Reject keyword arguments the pipeline cannot honour, instead of accepting
/// them and quietly doing something else.
fn reject_unsupported_kwargs(kwargs: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(kwargs) = kwargs else {
        return Ok(());
    };
    let mapping = kwargs.cast::<PyDict>().map_err(|_| {
        PyTypeError::new_err("pipeline keyword arguments must be a mapping".to_string())
    })?;
    if mapping.is_empty() {
        return Ok(());
    }
    let mut names: Vec<String> = mapping
        .keys()
        .iter()
        .map(|key| key.str().map(|value| value.to_string()))
        .collect::<PyResult<Vec<String>>>()?;
    names.sort();
    Err(PyValueError::new_err(format!(
        "unsupported pipeline argument(s): {}. They are rejected rather than silently ignored, \
         which is what this pipeline used to do with every one of its arguments.",
        names.join(", ")
    )))
}

/// Base pipeline class
#[pyclass(name = "Pipeline", module = "trustformers", subclass)]
pub struct PyPipeline {
    /// The model object this pipeline was constructed with.
    pub model: PyObject,
    /// The tokenizer object this pipeline was constructed with.
    pub tokenizer: PyObject,
    /// The device label the pipeline reports.
    pub device: String,
}

impl PyPipeline {
    /// Build the base state shared by every pipeline subclass.
    fn base(model: &Bound<'_, PyAny>, tokenizer: &Bound<'_, PyAny>, device: Option<&str>) -> Self {
        PyPipeline {
            model: model.clone().unbind(),
            tokenizer: tokenizer.clone().unbind(),
            device: device.unwrap_or("cpu").to_string(),
        }
    }
}

#[pymethods]
impl PyPipeline {
    /// Move pipeline to device.
    ///
    /// Only `cpu` is accepted: this crate's Python pipelines run the CPU
    /// forward path, so recording another device string would misreport where
    /// the computation happens.
    pub fn to(&mut self, device: &str) -> PyResult<()> {
        if device != "cpu" {
            return Err(PyValueError::new_err(format!(
                "pipeline device '{device}' is not available: the Python pipelines run the CPU \
                 forward path"
            )));
        }
        self.device = device.to_string();
        Ok(())
    }

    /// Get device
    #[getter]
    pub fn device(&self) -> &str {
        &self.device
    }
}

// ---------------------------------------------------------------------------
// text-generation
// ---------------------------------------------------------------------------

/// Text generation pipeline, driving a real GPT-2 language-model head.
#[pyclass(name = "TextGenerationPipeline", module = "trustformers", extends = PyPipeline)]
pub struct PyTextGenerationPipeline {
    /// The language-model head, resolved at construction.
    model: Py<PyGPT2LMHeadModel>,
    /// The tokenizer, resolved at construction.
    tokenizer: PipelineTokenizer,
}

impl PyTextGenerationPipeline {
    /// Generate the continuations of one prompt.
    fn generate(
        &self,
        py: Python<'_>,
        text: &str,
        options: &SamplingOptions,
    ) -> PyResult<Vec<GenerationResult>> {
        let encoded = self.tokenizer.encode(py, text)?;
        let prompt: Vec<usize> = encoded.input_ids.iter().map(|&id| id as usize).collect();
        if prompt.is_empty() {
            return Err(PyValueError::new_err(
                "the tokenizer produced no tokens for this prompt, so there is nothing to \
                 continue from",
            ));
        }

        let model = self.model.try_borrow(py)?;
        let sequences = generate_with_gpt2(model.model(), &prompt, options)
            .map_err(|e| PyValueError::new_err(format!("Generation failed: {e}")))?;
        drop(model);

        sequences
            .into_iter()
            .map(|sequence| {
                // Surfaced as an error rather than silently truncated: a
                // sequence that does not extend the prompt means the decoder
                // and the pipeline disagree about what was generated.
                continuation_tokens(&prompt, &sequence)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let ids: Vec<u32> = sequence.iter().map(|&token| token as u32).collect();
                Ok(GenerationResult {
                    generated_text: self.tokenizer.decode(py, &ids)?,
                })
            })
            .collect()
    }
}

#[pymethods]
impl PyTextGenerationPipeline {
    /// Create a new text generation pipeline.
    ///
    /// `model` must be a `GPT2LMHeadModel`: it is the only model in this crate
    /// with a real language-model head, and a pipeline that cannot generate is
    /// better refused here than at call time.
    #[new]
    #[pyo3(signature = (model, tokenizer, device=None))]
    pub fn new(
        model: &Bound<'_, PyAny>,
        tokenizer: &Bound<'_, PyAny>,
        device: Option<&str>,
    ) -> PyResult<(Self, PyPipeline)> {
        let lm_head = model.cast::<PyGPT2LMHeadModel>().map_err(|_| {
            PyTypeError::new_err(format!(
                "text-generation requires a GPT2LMHeadModel (the only model in this crate with a \
                 real language-model head), got {}",
                type_name(model)
            ))
        })?;
        let resolved_tokenizer = PipelineTokenizer::resolve(tokenizer)?;

        Ok((
            PyTextGenerationPipeline {
                model: lm_head.clone().unbind(),
                tokenizer: resolved_tokenizer,
            },
            PyPipeline::base(model, tokenizer, device),
        ))
    }

    /// Generate text.
    ///
    /// `top_k` and `top_p` default to `None` rather than to HuggingFace's
    /// `50` / `1.0`: this crate's decoder applies exactly one truncation
    /// strategy, so silently adopting both defaults would mean silently
    /// dropping one of them. Setting both is an error for the same reason.
    #[pyo3(signature = (text_inputs, max_length=50, min_length=0, do_sample=true, temperature=1.0, top_k=None, top_p=None, num_return_sequences=1, **kwargs))]
    pub fn __call__(
        &self,
        py: Python<'_>,
        text_inputs: TextInputs,
        max_length: usize,
        min_length: usize,
        do_sample: bool,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        num_return_sequences: usize,
        kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        reject_unsupported_kwargs(kwargs)?;
        if top_k.is_some() && top_p.is_some() {
            return Err(PyValueError::new_err(
                "set top_k or top_p, not both: this decoder applies a single truncation strategy \
                 per step, so honouring one would mean discarding the other",
            ));
        }
        if num_return_sequences == 0 {
            return Err(PyValueError::new_err(
                "num_return_sequences must be at least 1",
            ));
        }

        let options = SamplingOptions {
            max_length,
            min_length,
            do_sample,
            temperature,
            top_k,
            top_p,
            num_return_sequences,
        };

        match text_inputs {
            TextInputs::Single(text) => self.generate(py, &text, &options)?.into_py_any(py),
            TextInputs::Batch(texts) => texts
                .iter()
                .map(|text| self.generate(py, text, &options))
                .collect::<PyResult<Vec<Vec<GenerationResult>>>>()?
                .into_py_any(py),
        }
    }
}

// ---------------------------------------------------------------------------
// text-classification
// ---------------------------------------------------------------------------

/// Text classification pipeline, driving a real BERT sequence-classification
/// head.
#[pyclass(name = "TextClassificationPipeline", module = "trustformers", extends = PyPipeline)]
pub struct PyTextClassificationPipeline {
    /// The sequence-classification model, resolved at construction.
    model: Py<PyBertForSequenceClassification>,
    /// The tokenizer, resolved at construction.
    tokenizer: PipelineTokenizer,
}

impl PyTextClassificationPipeline {
    /// Classify one text: real tokenization, real forward pass, real softmax.
    fn classify(
        &self,
        py: Python<'_>,
        text: &str,
        top_k: Option<usize>,
    ) -> PyResult<Vec<ScoredLabel>> {
        let encoded = self.tokenizer.encode(py, text)?;
        let model = self.model.try_borrow(py)?;
        classify_with_bert(model.model(), encoded, model.labels(), top_k)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pymethods]
impl PyTextClassificationPipeline {
    /// Create a new text classification pipeline.
    #[new]
    #[pyo3(signature = (model, tokenizer, device=None))]
    pub fn new(
        model: &Bound<'_, PyAny>,
        tokenizer: &Bound<'_, PyAny>,
        device: Option<&str>,
    ) -> PyResult<(Self, PyPipeline)> {
        let classifier = model.cast::<PyBertForSequenceClassification>().map_err(|_| {
            PyTypeError::new_err(format!(
                "text-classification requires a BertForSequenceClassification, got {}",
                type_name(model)
            ))
        })?;
        let resolved_tokenizer = PipelineTokenizer::resolve(tokenizer)?;

        Ok((
            PyTextClassificationPipeline {
                model: classifier.clone().unbind(),
                tokenizer: resolved_tokenizer,
            },
            PyPipeline::base(model, tokenizer, device),
        ))
    }

    /// Classify text.
    ///
    /// Returns every class ranked best-first by default; pass `top_k` to keep
    /// only the leading classes.
    #[pyo3(signature = (text_inputs, top_k=None, **kwargs))]
    pub fn __call__(
        &self,
        py: Python<'_>,
        text_inputs: TextInputs,
        top_k: Option<usize>,
        kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        reject_unsupported_kwargs(kwargs)?;
        match text_inputs {
            TextInputs::Single(text) => self.classify(py, &text, top_k)?.into_py_any(py),
            TextInputs::Batch(texts) => texts
                .iter()
                .map(|text| self.classify(py, text, top_k))
                .collect::<PyResult<Vec<Vec<ScoredLabel>>>>()?
                .into_py_any(py),
        }
    }
}

// ---------------------------------------------------------------------------
// token-classification / question-answering
// ---------------------------------------------------------------------------

/// Why the two span-level pipelines cannot be built yet.
///
/// Both need character offsets to report where in the input a prediction
/// falls, and this workspace's tokenizers set `TokenizedInput::offset_mapping`
/// to `None` unconditionally (`trustformers-tokenizers`' WordPiece and BPE
/// encoders both do). Reporting token indices under HuggingFace's `start` /
/// `end` keys -- which are *character* offsets -- would be wrong in a way
/// callers cannot detect.
fn span_pipeline_unavailable(task: &str, model_name: &str, head_name: &str) -> PyErr {
    PyNotImplementedError::new_err(format!(
        "the '{task}' pipeline is not available. `trustformers_models::bert::{model_name}` is a \
         real model with a real {head_name} head, but this crate exposes no Python wrapper for \
         it, and the pipeline could not report HuggingFace's character-level `start`/`end` keys \
         in any case: trustformers-tokenizers does not produce an offset mapping yet. This is a \
         refusal rather than the placeholder result the pipeline used to return."
    ))
}

/// Token classification (NER) pipeline.
///
/// Construction always fails; see [`span_pipeline_unavailable`]. The class
/// stays registered so `trustformers.TokenClassificationPipeline` keeps
/// resolving, but it can no longer hand back the fixed `B-PER` / `"John"` /
/// `0..4` entity it used to invent for every input.
#[pyclass(name = "TokenClassificationPipeline", module = "trustformers", extends = PyPipeline)]
pub struct PyTokenClassificationPipeline;

#[pymethods]
impl PyTokenClassificationPipeline {
    /// Refuse construction with a structured `NotImplementedError`.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn new(
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<(Self, PyPipeline)> {
        let _ = (args, kwargs);
        Err(span_pipeline_unavailable(
            "token-classification",
            "BertForTokenClassification",
            "per-token classification",
        ))
    }
}

/// Question answering pipeline.
///
/// Construction always fails; see [`span_pipeline_unavailable`]. It used to
/// answer the literal string `"Example answer"` with `score: 0.85` for every
/// question.
#[pyclass(name = "QuestionAnsweringPipeline", module = "trustformers", extends = PyPipeline)]
pub struct PyQuestionAnsweringPipeline;

#[pymethods]
impl PyQuestionAnsweringPipeline {
    /// Refuse construction with a structured `NotImplementedError`.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn new(
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<(Self, PyPipeline)> {
        let _ = (args, kwargs);
        Err(span_pipeline_unavailable(
            "question-answering",
            "BertForQuestionAnswering",
            "span-prediction",
        ))
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Helper enum for one text or a batch of texts.
#[derive(FromPyObject)]
pub enum TextInputs {
    /// A single text.
    Single(String),
    /// A batch of texts.
    Batch(Vec<String>),
}

/// One generated continuation.
///
/// There is no `score` field: HuggingFace's text-generation pipeline has none
/// either, and the `0.95` this used to report was a constant, not a
/// likelihood.
#[derive(Clone)]
struct GenerationResult {
    /// Prompt plus continuation, detokenized.
    generated_text: String,
}

impl<'py> IntoPyObject<'py> for GenerationResult {
    type Target = PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        dict.set_item("generated_text", self.generated_text)?;
        Ok(dict)
    }
}

impl<'py> IntoPyObject<'py> for ScoredLabel {
    type Target = PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        dict.set_item("label", self.label)?;
        dict.set_item("score", self.score)?;
        Ok(dict)
    }
}

#[cfg(test)]
mod task_tests {
    use super::*;

    #[test]
    fn resolves_huggingface_task_aliases() {
        assert_eq!(
            canonical_task("sentiment-analysis"),
            Some(PipelineTask::TextClassification)
        );
        assert_eq!(
            canonical_task("text-classification"),
            Some(PipelineTask::TextClassification)
        );
        assert_eq!(canonical_task("ner"), Some(PipelineTask::TokenClassification));
        assert_eq!(
            canonical_task("token-classification"),
            Some(PipelineTask::TokenClassification)
        );
        assert_eq!(
            canonical_task("text-generation"),
            Some(PipelineTask::TextGeneration)
        );
        assert_eq!(
            canonical_task("question-answering"),
            Some(PipelineTask::QuestionAnswering)
        );
    }

    #[test]
    fn rejects_an_unknown_task() {
        assert_eq!(canonical_task("summarization"), None);
        assert_eq!(canonical_task(""), None);
        assert_eq!(canonical_task("Text-Generation"), None);
    }

    /// The error message lists the task names, so the list must stay in sync
    /// with what `canonical_task` actually accepts.
    #[test]
    fn every_advertised_task_resolves() {
        for task in KNOWN_TASKS {
            assert!(
                canonical_task(task).is_some(),
                "advertised task '{task}' does not resolve"
            );
        }
    }
}
