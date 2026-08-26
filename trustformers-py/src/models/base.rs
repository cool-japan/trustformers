//! `PreTrainedModel`: the base pyclass every concrete model wrapper below
//! `#[pyclass(extends = PyPreTrainedModel)]`-extends, holding just the
//! checkpoint's parsed config dict. Split out of `models/mod.rs` (2026-08-24,
//! `py-followups`, keeping this crate's files under the 2000-line policy);
//! behavior is unchanged, this is a pure code-motion.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Owned Python reference alias (pyo3 0.28 removed the `PyObject` type alias from
/// the crate root; it is equivalent to `Py<PyAny>`).
type PyObject = Py<PyAny>;

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
