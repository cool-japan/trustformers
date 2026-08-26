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
//! * `token-classification` runs a real
//!   [`trustformers_models::bert::BertForTokenClassification`] forward pass,
//!   aggregates it with `aggregation_strategy='simple'`
//!   (`span::classify_tokens_with_bert`/`span::aggregate_entities_simple`),
//!   and reports real `entity_group`/`score`/`word`/`start`/`end` entries.
//! * `question-answering` runs a real
//!   [`trustformers_models::bert::BertForQuestionAnswering`] forward pass and
//!   extracts the real joint-argmax answer span
//!   (`span::answer_with_bert`/`span::extract_answer`).
//!
//! Both span pipelines became real once `trustformers-tokenizers`'s
//! `WordPieceTokenizer`/`BPETokenizer` started populating a real
//! `offset_mapping` (verified 2026-08-24 by reading that crate directly, not
//! by trusting a handoff note -- `Tokenizer::encode`/`encode_pair` both
//! return `Some(offsets)` unconditionally today). `span`'s extraction math
//! reports **byte** offsets, matching this crate's in-tree convention; the
//! HuggingFace-facing `start`/`end` keys these pipelines report are
//! **character** offsets, converted once at this module's Python boundary
//! (see [`HfEntity`]/[`HfAnswer`]) with
//! `trustformers_tokenizers::byte_offsets_to_char_offsets` -- not silently,
//! and not inside `span` itself, which stays byte-offset throughout.
//!
//! `question-answering` requires a `WordPieceTokenizer`: see
//! [`qa_requires_wordpiece`] for why a `BPETokenizer` is rejected instead of
//! (mis)supported.

mod scoring;
mod span;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;

use crate::models::generation::{continuation_tokens, generate_with_gpt2, SamplingOptions};
use crate::models::{
    PyBertForQuestionAnswering, PyBertForSequenceClassification, PyBertForTokenClassification,
    PyGPT2LMHeadModel,
};
use crate::tokenizers::{PyBPETokenizer, PyWordPieceTokenizer};
use scoring::{classify_with_bert, ScoredLabel};
use span::{Entity, QaAnswer};
use trustformers_core::errors::TrustformersError;
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

/// Convert `entities`' real **byte** offsets (this crate's in-tree
/// convention; see `span`'s module doc) to HuggingFace's real **character**
/// offsets, once, at this Python boundary -- explicitly, not silently.
/// `span::Entity` itself is never mutated in place: it stays byte-offset,
/// matching its own documented contract.
///
/// # Errors
///
/// Fails when an entity's byte offsets do not land on `text`'s character
/// boundaries (they always do for offsets this crate's own tokenizers
/// produced from `text` itself; see
/// `trustformers_tokenizers::byte_offsets_to_char_offsets`).
fn entities_with_char_offsets(
    text: &str,
    entities: Vec<Entity>,
) -> Result<Vec<HfEntity>, TrustformersError> {
    if entities.is_empty() {
        return Ok(Vec::new());
    }
    let byte_spans: Vec<(usize, usize)> =
        entities.iter().map(|entity| (entity.start, entity.end)).collect();
    let char_spans = trustformers_tokenizers::byte_offsets_to_char_offsets(text, &byte_spans)?;
    Ok(entities
        .into_iter()
        .zip(char_spans)
        .map(|(entity, (start, end))| HfEntity {
            word: entity.word,
            entity_group: entity.entity_group,
            score: entity.score,
            start,
            end,
        })
        .collect())
}

/// The same byte->character conversion as [`entities_with_char_offsets`], for
/// one [`QaAnswer`].
fn answer_with_char_offsets(
    context: &str,
    answer: QaAnswer,
) -> Result<HfAnswer, TrustformersError> {
    let char_spans = trustformers_tokenizers::byte_offsets_to_char_offsets(
        context,
        &[(answer.start, answer.end)],
    )?;
    let (start, end) = char_spans[0];
    Ok(HfAnswer {
        answer: answer.answer,
        score: answer.score,
        start,
        end,
    })
}

/// Why a `BPETokenizer` cannot back the `question-answering` pipeline.
///
/// `BPETokenizer::encode_pair` joins the question and context into one
/// string with a single space and re-encodes it as a single sequence: no
/// separator token, no `token_type_ids`, and its `offset_mapping` indexes
/// that *joined* string rather than either original sequence on its own (see
/// the `trustformers-tokenizers` handoff this pass built on). Without
/// `token_type_ids` there is no reliable way to know where the context
/// begins among the encoded positions -- and `BertForQuestionAnswering`, the
/// only question-answering head this crate wraps, is a WordPiece-family
/// model in any case, so there is no real checkpoint this would ever need to
/// serve.
fn qa_requires_wordpiece(tokenizer: &Bound<'_, PyAny>) -> PyErr {
    PyTypeError::new_err(format!(
        "question-answering requires a WordPieceTokenizer, got {}. BPETokenizer::encode_pair \
         joins the question and context into one string with a single space and reports offsets \
         into that joined string, not per-sequence offsets with a real separator token, so there \
         is no reliable way to find where the context begins in the encoded sequence -- and \
         BertForQuestionAnswering (the only question-answering head in this crate) is a \
         WordPiece-family model in any case.",
        type_name(tokenizer)
    ))
}

/// Token classification (NER) pipeline: real per-token forward pass through a
/// [`crate::models::PyBertForTokenClassification`] head, aggregated with
/// `aggregation_strategy='simple'` and reported at real character offsets.
///
/// Used to always refuse construction (`trustformers-tokenizers` had no
/// offset mapping); before that, it answered a fixed `B-PER` entity named
/// `"John"` at characters `0..4` for every input. Neither survives: a model
/// without a per-token head is a `TypeError` from `__init__`, and every
/// `entity_group`/`score`/`word`/`start`/`end` below comes from a real
/// forward pass over the actual input text.
#[pyclass(name = "TokenClassificationPipeline", module = "trustformers", extends = PyPipeline)]
pub struct PyTokenClassificationPipeline {
    /// The token-classification head, resolved at construction.
    model: Py<PyBertForTokenClassification>,
    /// The tokenizer, resolved at construction. Either `WordPieceTokenizer`
    /// or `BPETokenizer` works here: NER needs only `Tokenizer::encode`'s
    /// single-sequence offsets, which both now produce for real.
    tokenizer: PipelineTokenizer,
}

impl PyTokenClassificationPipeline {
    /// Classify one text: real tokenization, real forward pass, real
    /// `simple`-strategy aggregation, real byte->character conversion.
    fn classify(&self, py: Python<'_>, text: &str) -> PyResult<Vec<HfEntity>> {
        let encoded = self.tokenizer.encode(py, text)?;
        let model = self.model.try_borrow(py)?;
        let entities =
            span::classify_tokens_with_bert(model.model(), encoded, text, model.labels())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        entities_with_char_offsets(text, entities).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pymethods]
impl PyTokenClassificationPipeline {
    /// Create a new token-classification (NER) pipeline.
    #[new]
    #[pyo3(signature = (model, tokenizer, device=None))]
    pub fn new(
        model: &Bound<'_, PyAny>,
        tokenizer: &Bound<'_, PyAny>,
        device: Option<&str>,
    ) -> PyResult<(Self, PyPipeline)> {
        let head = model.cast::<PyBertForTokenClassification>().map_err(|_| {
            PyTypeError::new_err(format!(
                "token-classification requires a BertForTokenClassification, got {}",
                type_name(model)
            ))
        })?;
        let resolved_tokenizer = PipelineTokenizer::resolve(tokenizer)?;

        Ok((
            PyTokenClassificationPipeline {
                model: head.clone().unbind(),
                tokenizer: resolved_tokenizer,
            },
            PyPipeline::base(model, tokenizer, device),
        ))
    }

    /// Tag text with named entities.
    ///
    /// Only `aggregation_strategy='simple'` is implemented (the default): an
    /// unrecognised strategy is rejected outright rather than silently
    /// treated as `'simple'`.
    #[pyo3(signature = (text_inputs, aggregation_strategy=None, **kwargs))]
    pub fn __call__(
        &self,
        py: Python<'_>,
        text_inputs: TextInputs,
        aggregation_strategy: Option<String>,
        kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        reject_unsupported_kwargs(kwargs)?;
        if let Some(strategy) = aggregation_strategy.as_deref() {
            if strategy != "simple" {
                return Err(PyValueError::new_err(format!(
                    "aggregation_strategy '{strategy}' is not supported; only 'simple' \
                     (contiguous same-type tokens merged into one entity) is implemented"
                )));
            }
        }

        match text_inputs {
            TextInputs::Single(text) => self.classify(py, &text)?.into_py_any(py),
            TextInputs::Batch(texts) => texts
                .iter()
                .map(|text| self.classify(py, text))
                .collect::<PyResult<Vec<Vec<HfEntity>>>>()?
                .into_py_any(py),
        }
    }
}

/// Question answering pipeline: real forward pass through a
/// [`crate::models::PyBertForQuestionAnswering`] head, extracting the real
/// joint-argmax answer span and reporting it at real character offsets.
///
/// Used to always refuse construction; before that, it answered the literal
/// string `"Example answer"` with `score: 0.85` for every question. Neither
/// survives: `answer`/`score`/`start`/`end` below come from a real forward
/// pass restricted to the real context span.
///
/// The tokenizer must be a `WordPieceTokenizer`; see
/// [`qa_requires_wordpiece`] for why `BPETokenizer` is rejected rather than
/// (mis)supported.
#[pyclass(name = "QuestionAnsweringPipeline", module = "trustformers", extends = PyPipeline)]
pub struct PyQuestionAnsweringPipeline {
    /// The question-answering head, resolved at construction.
    model: Py<PyBertForQuestionAnswering>,
    /// The tokenizer, resolved at construction. `WordPieceTokenizer` only --
    /// see [`qa_requires_wordpiece`].
    tokenizer: Py<PyWordPieceTokenizer>,
}

impl PyQuestionAnsweringPipeline {
    /// Answer one question against one context: real pair tokenization, real
    /// forward pass, real extraction, real byte->character conversion.
    fn answer(
        &self,
        py: Python<'_>,
        question: &str,
        context: &str,
        max_answer_len: usize,
    ) -> PyResult<HfAnswer> {
        let encoded = self
            .tokenizer
            .try_borrow(py)?
            .tokenizer()
            .encode_pair(question, context)
            .map_err(|e| PyValueError::new_err(format!("Tokenization failed: {e}")))?;
        let model = self.model.try_borrow(py)?;
        let answer = span::answer_with_bert(model.model(), encoded, context, max_answer_len)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        answer_with_char_offsets(context, answer).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pymethods]
impl PyQuestionAnsweringPipeline {
    /// Create a new question-answering pipeline.
    #[new]
    #[pyo3(signature = (model, tokenizer, device=None))]
    pub fn new(
        model: &Bound<'_, PyAny>,
        tokenizer: &Bound<'_, PyAny>,
        device: Option<&str>,
    ) -> PyResult<(Self, PyPipeline)> {
        let head = model.cast::<PyBertForQuestionAnswering>().map_err(|_| {
            PyTypeError::new_err(format!(
                "question-answering requires a BertForQuestionAnswering, got {}",
                type_name(model)
            ))
        })?;
        let wordpiece = tokenizer
            .cast::<PyWordPieceTokenizer>()
            .map_err(|_| qa_requires_wordpiece(tokenizer))?;

        Ok((
            PyQuestionAnsweringPipeline {
                model: head.clone().unbind(),
                tokenizer: wordpiece.clone().unbind(),
            },
            PyPipeline::base(model, tokenizer, device),
        ))
    }

    /// Extract an answer to `question` from `context`.
    ///
    /// `max_answer_len` bounds the answer span's *token* width (HuggingFace's
    /// own default is 15).
    #[pyo3(signature = (question, context, max_answer_len=15, **kwargs))]
    pub fn __call__(
        &self,
        py: Python<'_>,
        question: String,
        context: String,
        max_answer_len: usize,
        kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        reject_unsupported_kwargs(kwargs)?;
        self.answer(py, &question, &context, max_answer_len)?.into_py_any(py)
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

/// One named entity as `TokenClassificationPipeline.__call__` reports it to
/// Python: HuggingFace's own `aggregation_strategy='simple'` keys
/// (`entity_group`/`score`/`word`/`start`/`end`), with `start`/`end` in
/// Unicode **characters** -- converted from `span::Entity`'s native byte
/// offsets by [`entities_with_char_offsets`], once, at this module's Python
/// boundary. `word` is unaffected by the conversion (already the correct
/// surface text either way).
#[derive(Debug, Clone, PartialEq)]
struct HfEntity {
    /// The entity's surface text, unchanged by the byte->character
    /// conversion (which only touches `start`/`end`).
    word: String,
    /// The entity type, `B-`/`I-` prefix stripped (see
    /// `span::entity_type`).
    entity_group: String,
    /// Mean softmax probability of the winning label across the entity's
    /// tokens.
    score: f32,
    /// Start **character** offset in the original text (inclusive) --
    /// HuggingFace's convention, so `text[start:end]` indexes correctly in
    /// Python.
    start: usize,
    /// End **character** offset in the original text (exclusive).
    end: usize,
}

impl<'py> IntoPyObject<'py> for HfEntity {
    type Target = PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        dict.set_item("entity_group", self.entity_group)?;
        dict.set_item("score", self.score)?;
        dict.set_item("word", self.word)?;
        dict.set_item("start", self.start)?;
        dict.set_item("end", self.end)?;
        Ok(dict)
    }
}

/// One extracted answer as `QuestionAnsweringPipeline.__call__` reports it to
/// Python -- HuggingFace's own `score`/`start`/`end`/`answer` keys, with
/// `start`/`end` in Unicode **characters** (see [`HfEntity`]'s doc comment
/// for the same conversion, applied here by [`answer_with_char_offsets`]).
#[derive(Debug, Clone, PartialEq)]
struct HfAnswer {
    /// The answer's surface text, sliced from the original context.
    answer: String,
    /// `softmax(start_logits)[start] * softmax(end_logits)[end]` at the
    /// chosen span.
    score: f32,
    /// Start **character** offset in the original context (inclusive).
    start: usize,
    /// End **character** offset in the original context (exclusive).
    end: usize,
}

impl<'py> IntoPyObject<'py> for HfAnswer {
    type Target = PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        dict.set_item("score", self.score)?;
        dict.set_item("start", self.start)?;
        dict.set_item("end", self.end)?;
        dict.set_item("answer", self.answer)?;
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

    // ---- entities_with_char_offsets / answer_with_char_offsets: the
    // byte->character conversion this module's Python boundary applies --
    // the exact functions `PyTokenClassificationPipeline::classify`/
    // `PyQuestionAnsweringPipeline::answer` call live. `span::Entity`'s own
    // pure tests already lock down that its *byte* offsets are correct (see
    // `slices_multibyte_accented_text_by_byte_offsets_not_char_counts`); what
    // is new here is the conversion this module adds on top, using the same
    // multi-byte fixtures so the two are directly comparable.

    /// The discriminating assertion for this pass's byte/character fix:
    /// `span::Entity`'s pure test asserts "café" is a **5-byte-wide** span;
    /// this asserts the *converted* `HfEntity` -- what Python actually
    /// receives -- reports the **same** span **4 characters wide**, and that
    /// slicing `text` by *characters* at that width recovers "café" exactly
    /// (which byte-slicing at a 4-wide range would not: it would cut the
    /// last byte off "é").
    #[test]
    fn entities_with_char_offsets_converts_byte_widths_to_character_widths() {
        let text = "El café está en el centro";
        let byte_start = text.find("café").expect("fixture contains café");
        let byte_end = byte_start + "café".len();
        assert_eq!(byte_end - byte_start, 5, "café is 5 UTF-8 bytes wide");

        let entity = Entity {
            word: "café".to_string(),
            entity_group: "MISC".to_string(),
            score: 0.99,
            start: byte_start,
            end: byte_end,
        };

        let converted =
            entities_with_char_offsets(text, vec![entity]).expect("valid byte offsets convert");
        assert_eq!(converted.len(), 1);
        let entity = &converted[0];

        assert_eq!(
            entity.end - entity.start,
            4,
            "café must convert to a 4-CHARACTER-wide span"
        );
        assert_ne!(
            entity.end - entity.start,
            byte_end - byte_start,
            "the converted width must differ from the original byte width, or this test proves \
             nothing"
        );
        // The property that actually matters to a Python caller: character
        // slicing at the converted offsets recovers the real word.
        let sliced: String =
            text.chars().skip(entity.start).take(entity.end - entity.start).collect();
        assert_eq!(sliced, entity.word);
        assert_eq!(
            entity.word, "café",
            "the conversion must not alter the surface text itself"
        );
    }

    /// The same discriminator, for a CJK fixture mixed with ASCII: "東京" is
    /// 6 UTF-8 bytes but 2 characters. Also covers more than one entity in a
    /// single call, and an entity that is not the first thing in the text
    /// (so the character-index arithmetic must count from the true start of
    /// `text`, not from the entity's own span).
    #[test]
    fn entities_with_char_offsets_handles_a_cjk_fixture_and_multiple_entities() {
        let text = "Tim visited 東京 last year";
        let tim_start = 0;
        let tim_end = "Tim".len();
        let tokyo_start = text.find("東京").expect("fixture contains 東京");
        let tokyo_end = tokyo_start + "東京".len();
        assert_eq!(tokyo_end - tokyo_start, 6, "東京 is 6 UTF-8 bytes wide");

        let entities = vec![
            Entity {
                word: "Tim".to_string(),
                entity_group: "PER".to_string(),
                score: 0.9,
                start: tim_start,
                end: tim_end,
            },
            Entity {
                word: "東京".to_string(),
                entity_group: "LOC".to_string(),
                score: 0.95,
                start: tokyo_start,
                end: tokyo_end,
            },
        ];

        let converted =
            entities_with_char_offsets(text, entities).expect("valid byte offsets convert");
        assert_eq!(converted.len(), 2);

        // ASCII prefix: byte and character offsets agree.
        assert_eq!(converted[0].start, tim_start);
        assert_eq!(converted[0].end, tim_end);

        // The CJK entity: 2 characters wide, not 6.
        assert_eq!(
            converted[1].end - converted[1].start,
            2,
            "東京 must convert to a 2-CHARACTER span"
        );
        let sliced: String = text
            .chars()
            .skip(converted[1].start)
            .take(converted[1].end - converted[1].start)
            .collect();
        assert_eq!(sliced, "東京");
    }

    /// An empty entity list must convert to an empty list, not an error --
    /// `byte_offsets_to_char_offsets` is never called with an empty slice in
    /// the first place.
    #[test]
    fn entities_with_char_offsets_of_an_empty_list_is_empty() {
        assert_eq!(
            entities_with_char_offsets("anything", vec![]).expect("empty list converts"),
            vec![]
        );
    }

    /// The same conversion, for `QaAnswer` -> `HfAnswer`: byte-wide "café"
    /// must convert to a character-wide span whose text still slices
    /// correctly.
    #[test]
    fn answer_with_char_offsets_converts_byte_widths_to_character_widths() {
        let context = "the café is closed";
        let byte_start = context.find("café").expect("fixture contains café");
        let byte_end = byte_start + "café".len();

        let answer = QaAnswer {
            answer: "café".to_string(),
            score: 0.8,
            start: byte_start,
            end: byte_end,
        };
        let converted =
            answer_with_char_offsets(context, answer).expect("valid byte offsets convert");

        assert_eq!(
            converted.end - converted.start,
            4,
            "café must convert to a 4-CHARACTER-wide span"
        );
        let sliced: String = context
            .chars()
            .skip(converted.start)
            .take(converted.end - converted.start)
            .collect();
        assert_eq!(sliced, "café");
        assert_eq!(converted.answer, "café");
    }
}
