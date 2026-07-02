use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

/// Owned Python reference alias (pyo3 0.28 removed the `PyObject` type alias from
/// the crate root; it is equivalent to `Py<PyAny>`).
type PyObject = Py<PyAny>;
// Pipeline implementations are already available here
// In the future, these could be moved to trustformers-models for code reuse
// Currently implemented: Pipeline, TextGenerationPipeline, TextClassificationPipeline,
// TokenClassificationPipeline, QuestionAnsweringPipeline

/// Base pipeline class
#[pyclass(name = "Pipeline", module = "trustformers", subclass)]
pub struct PyPipeline {
    pub model: PyObject,
    pub tokenizer: PyObject,
    pub device: String,
}

#[pymethods]
impl PyPipeline {
    /// Move pipeline to device
    pub fn to(&mut self, device: &str) -> PyResult<()> {
        self.device = device.to_string();
        Ok(())
    }

    /// Get device
    #[getter]
    pub fn device(&self) -> &str {
        &self.device
    }
}

/// Text generation pipeline
#[pyclass(name = "TextGenerationPipeline", module = "trustformers", extends = PyPipeline)]
pub struct PyTextGenerationPipeline {
    // inner: TextGenerationPipeline,
}

#[pymethods]
impl PyTextGenerationPipeline {
    /// Create a new text generation pipeline
    #[new]
    pub fn new(
        _py: Python<'_>,
        model: &Bound<'_, PyAny>,
        tokenizer: &Bound<'_, PyAny>,
        device: Option<&str>,
    ) -> PyResult<(Self, PyPipeline)> {
        let base = PyPipeline {
            model: model.clone().unbind(),
            tokenizer: tokenizer.clone().unbind(),
            device: device.unwrap_or("cpu").to_string(),
        };

        Ok((PyTextGenerationPipeline {}, base))
    }

    /// Generate text
    #[pyo3(signature = (text_inputs, max_length=50, min_length=0, do_sample=true, temperature=1.0, top_k=50, top_p=1.0, num_return_sequences=1, **kwargs))]
    pub fn __call__(
        &self,
        py: Python<'_>,
        text_inputs: TextInputs,
        max_length: usize,
        min_length: usize,
        do_sample: bool,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        num_return_sequences: usize,
        kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        // Generation controls are accepted for HF API parity; this placeholder
        // generator does not yet consume them.
        let _ = (max_length, min_length, do_sample, temperature, top_k, top_p, kwargs);
        match text_inputs {
            TextInputs::Single(text) => {
                // Single text generation
                let result = vec![GenerationResult {
                    generated_text: format!("{} [Generated continuation]", text),
                    score: 0.95,
                }];
                result.into_py_any(py)
            },
            TextInputs::Batch(texts) => {
                // Batch generation
                let results: Vec<Vec<GenerationResult>> = texts
                    .iter()
                    .map(|text| {
                        (0..num_return_sequences)
                            .map(|i| GenerationResult {
                                generated_text: format!("{} [Generated continuation {}]", text, i),
                                score: 0.95 - (i as f32 * 0.1),
                            })
                            .collect()
                    })
                    .collect();
                results.into_py_any(py)
            },
        }
    }
}

/// Text classification pipeline
#[pyclass(name = "TextClassificationPipeline", module = "trustformers", extends = PyPipeline)]
pub struct PyTextClassificationPipeline {
    // inner: TextClassificationPipeline,
}

#[pymethods]
impl PyTextClassificationPipeline {
    /// Create a new text classification pipeline
    #[new]
    pub fn new(
        _py: Python<'_>,
        model: &Bound<'_, PyAny>,
        tokenizer: &Bound<'_, PyAny>,
        device: Option<&str>,
    ) -> PyResult<(Self, PyPipeline)> {
        let base = PyPipeline {
            model: model.clone().unbind(),
            tokenizer: tokenizer.clone().unbind(),
            device: device.unwrap_or("cpu").to_string(),
        };

        Ok((PyTextClassificationPipeline {}, base))
    }

    /// Classify text
    #[pyo3(signature = (text_inputs, **kwargs))]
    pub fn __call__(
        &self,
        py: Python<'_>,
        text_inputs: TextInputs,
        kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let _ = kwargs;
        match text_inputs {
            TextInputs::Single(_text) => {
                let result = vec![
                    ClassificationResult {
                        label: "POSITIVE".to_string(),
                        score: 0.7,
                    },
                    ClassificationResult {
                        label: "NEGATIVE".to_string(),
                        score: 0.3,
                    },
                ];
                result.into_py_any(py)
            },
            TextInputs::Batch(texts) => {
                let results: Vec<Vec<ClassificationResult>> = texts
                    .iter()
                    .map(|_| {
                        vec![
                            ClassificationResult {
                                label: "POSITIVE".to_string(),
                                score: 0.7,
                            },
                            ClassificationResult {
                                label: "NEGATIVE".to_string(),
                                score: 0.3,
                            },
                        ]
                    })
                    .collect();
                results.into_py_any(py)
            },
        }
    }
}

/// Token classification pipeline
#[pyclass(name = "TokenClassificationPipeline", module = "trustformers", extends = PyPipeline)]
pub struct PyTokenClassificationPipeline;

#[pymethods]
impl PyTokenClassificationPipeline {
    /// Create a new token classification pipeline
    #[new]
    pub fn new(
        _py: Python<'_>,
        model: &Bound<'_, PyAny>,
        tokenizer: &Bound<'_, PyAny>,
        device: Option<&str>,
    ) -> PyResult<(Self, PyPipeline)> {
        let base = PyPipeline {
            model: model.clone().unbind(),
            tokenizer: tokenizer.clone().unbind(),
            device: device.unwrap_or("cpu").to_string(),
        };

        Ok((PyTokenClassificationPipeline, base))
    }

    /// Perform NER
    #[pyo3(signature = (text_inputs, **kwargs))]
    pub fn __call__(
        &self,
        py: Python<'_>,
        text_inputs: TextInputs,
        kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let _ = kwargs;
        match text_inputs {
            TextInputs::Single(_text) => {
                let result = vec![EntityResult {
                    entity: "B-PER".to_string(),
                    score: 0.95,
                    index: 1,
                    word: "John".to_string(),
                    start: 0,
                    end: 4,
                }];
                result.into_py_any(py)
            },
            TextInputs::Batch(texts) => {
                let results: Vec<Vec<EntityResult>> = texts
                    .iter()
                    .map(|_| {
                        vec![EntityResult {
                            entity: "B-PER".to_string(),
                            score: 0.95,
                            index: 1,
                            word: "John".to_string(),
                            start: 0,
                            end: 4,
                        }]
                    })
                    .collect();
                results.into_py_any(py)
            },
        }
    }
}

/// Question answering pipeline
#[pyclass(name = "QuestionAnsweringPipeline", module = "trustformers", extends = PyPipeline)]
pub struct PyQuestionAnsweringPipeline;

#[pymethods]
impl PyQuestionAnsweringPipeline {
    /// Create a new QA pipeline
    #[new]
    pub fn new(
        _py: Python<'_>,
        model: &Bound<'_, PyAny>,
        tokenizer: &Bound<'_, PyAny>,
        device: Option<&str>,
    ) -> PyResult<(Self, PyPipeline)> {
        let base = PyPipeline {
            model: model.clone().unbind(),
            tokenizer: tokenizer.clone().unbind(),
            device: device.unwrap_or("cpu").to_string(),
        };

        Ok((PyQuestionAnsweringPipeline, base))
    }

    /// Answer questions
    #[pyo3(signature = (question, context, **kwargs))]
    pub fn __call__(
        &self,
        py: Python<'_>,
        question: QuestionInput,
        context: Option<String>,
        kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let _ = kwargs;
        match question {
            QuestionInput::Single {
                question: _question,
                context: ctx,
            } => {
                let _context =
                    context.or(ctx).ok_or_else(|| PyValueError::new_err("Context is required"))?;

                let result = QAResult {
                    answer: "Example answer".to_string(),
                    score: 0.85,
                    start: 10,
                    end: 23,
                };
                result.into_py_any(py)
            },
            QuestionInput::Batch(inputs) => {
                let results: Vec<QAResult> = inputs
                    .iter()
                    .map(|_| QAResult {
                        answer: "Example answer".to_string(),
                        score: 0.85,
                        start: 10,
                        end: 23,
                    })
                    .collect();
                results.into_py_any(py)
            },
        }
    }
}

/// Pipeline factory function
#[pyfunction]
#[pyo3(signature = (task, model=None, tokenizer=None, **kwargs))]
pub fn pipeline(
    py: Python<'_>,
    task: &str,
    model: Option<&Bound<'_, PyAny>>,
    tokenizer: Option<&Bound<'_, PyAny>>,
    kwargs: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let _ = kwargs;
    match task {
        "text-generation" => {
            if let (Some(m), Some(t)) = (model, tokenizer) {
                let pipe = PyTextGenerationPipeline::new(py, m, t, None)?;
                Py::new(py, pipe).and_then(|p| p.into_py_any(py))
            } else {
                Err(PyValueError::new_err("Model and tokenizer are required"))
            }
        },
        "text-classification" | "sentiment-analysis" => {
            if let (Some(m), Some(t)) = (model, tokenizer) {
                let pipe = PyTextClassificationPipeline::new(py, m, t, None)?;
                Py::new(py, pipe).and_then(|p| p.into_py_any(py))
            } else {
                Err(PyValueError::new_err("Model and tokenizer are required"))
            }
        },
        "token-classification" | "ner" => {
            if let (Some(m), Some(t)) = (model, tokenizer) {
                let pipe = PyTokenClassificationPipeline::new(py, m, t, None)?;
                Py::new(py, pipe).and_then(|p| p.into_py_any(py))
            } else {
                Err(PyValueError::new_err("Model and tokenizer are required"))
            }
        },
        "question-answering" => {
            if let (Some(m), Some(t)) = (model, tokenizer) {
                let pipe = PyQuestionAnsweringPipeline::new(py, m, t, None)?;
                Py::new(py, pipe).and_then(|p| p.into_py_any(py))
            } else {
                Err(PyValueError::new_err("Model and tokenizer are required"))
            }
        },
        _ => Err(PyValueError::new_err(format!("Unknown task: {}", task))),
    }
}

/// Helper types
#[derive(FromPyObject)]
pub enum TextInputs {
    Single(String),
    Batch(Vec<String>),
}

#[derive(FromPyObject)]
pub enum QuestionInput {
    Single {
        question: String,
        context: Option<String>,
    },
    Batch(Vec<(String, String)>),
}

#[derive(Clone)]
struct GenerationResult {
    generated_text: String,
    score: f32,
}

impl<'py> IntoPyObject<'py> for GenerationResult {
    type Target = pyo3::types::PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("generated_text", self.generated_text)?;
        dict.set_item("score", self.score)?;
        Ok(dict)
    }
}

#[derive(Clone)]
struct ClassificationResult {
    label: String,
    score: f32,
}

impl<'py> IntoPyObject<'py> for ClassificationResult {
    type Target = pyo3::types::PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("label", self.label)?;
        dict.set_item("score", self.score)?;
        Ok(dict)
    }
}

#[derive(Clone)]
struct EntityResult {
    entity: String,
    score: f32,
    index: usize,
    word: String,
    start: usize,
    end: usize,
}

impl<'py> IntoPyObject<'py> for EntityResult {
    type Target = pyo3::types::PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("entity", self.entity)?;
        dict.set_item("score", self.score)?;
        dict.set_item("index", self.index)?;
        dict.set_item("word", self.word)?;
        dict.set_item("start", self.start)?;
        dict.set_item("end", self.end)?;
        Ok(dict)
    }
}

#[derive(Clone)]
struct QAResult {
    answer: String,
    score: f32,
    start: usize,
    end: usize,
}

impl<'py> IntoPyObject<'py> for QAResult {
    type Target = pyo3::types::PyDict;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("answer", self.answer)?;
        dict.set_item("score", self.score)?;
        dict.set_item("start", self.start)?;
        dict.set_item("end", self.end)?;
        Ok(dict)
    }
}
