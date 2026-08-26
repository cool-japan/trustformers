//! Python model wrappers, one architecture (or task-head family) per
//! submodule.
//!
//! This used to be a single 2096-line file; split 2026-08-24
//! (`py-followups`) to keep every file under this crate's 2000-line policy.
//! The split is a pure code-motion -- every `pub` pyclass is re-exported
//! below at exactly its old `crate::models::PyXxx` path, so no other file in
//! this crate (or downstream Python code, which never saw Rust module paths
//! to begin with) needed to change.

pub(crate) mod generation;
mod inputs;
mod losses;
mod weights;

mod base;
mod bert;
mod gpt2;
mod helpers;
mod llama;
mod mamba;
mod rwkv;
mod t5;
mod tasks;

pub use base::PyPreTrainedModel;
pub use bert::PyBertModel;
pub use gpt2::{PyGPT2LMHeadModel, PyGPT2Model};
pub use llama::PyLlamaModel;
pub use mamba::PyMambaModel;
pub use rwkv::PyRwkvModel;
pub use t5::PyT5Model;
pub use tasks::{PyBertForQuestionAnswering, PyBertForSequenceClassification, PyBertForTokenClassification};
