use pyo3::prelude::*;

// pub mod auto;  // Temporarily disabled due to dependency issues
pub mod complex_tensor;
// pub mod config_utils;  // Temporarily disabled due to dependency issues
pub mod errors;
pub mod memory_manager;
// pub mod models;  // Temporarily disabled due to dependency issues
pub mod performance;
// pub mod pipelines;  // Temporarily disabled due to dependency issues
pub mod tensor;
pub mod tensor_optimized;
// pub mod tokenizers;  // Temporarily disabled due to dependency issues
// pub mod training;  // Temporarily disabled due to dependency issues
pub mod utils;

// Temporarily disabled imports due to dependency issues
// use crate::auto::{
//     pipeline, PyAutoModel, PyAutoModelForCausalLM, PyAutoModelForMaskedLM,
//     PyAutoModelForQuestionAnswering, PyAutoModelForSequenceClassification,
//     PyAutoModelForTokenClassification, PyAutoTokenizer,
// };
// use crate::models::{
//     PyBertForSequenceClassification, PyBertModel, PyGPT2LMHeadModel, PyGPT2Model, PyLlamaModel,
//     PyPreTrainedModel, PyT5Model,
// };
// use crate::pipelines::{PyTextClassificationPipeline, PyTextGenerationPipeline};
// use crate::complex_tensor::PyComplexTensor;
// use crate::performance::{MemoryTracker, PerformanceProfiler, PerformanceUtils, ProfilerContext};
use crate::tensor::PyTensor;
use crate::tensor_optimized::{PyAdvancedActivations, PyTensorOptimized};
// use crate::tokenizers::{PyBPETokenizer, PyWordPieceTokenizer};
// use crate::training::{PyTrainer, PyTrainingArguments};

/// TrustformeRS Python Module
///
/// A high-performance transformer library for Python, written in Rust.
#[pymodule]
fn _trustformers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Core tensor operations
    m.add_class::<PyTensor>()?;
    m.add_class::<PyTensorOptimized>()?;
    m.add_class::<PyAdvancedActivations>()?;
    // m.add_class::<PyComplexTensor>()?;

    // Performance monitoring
    // m.add_class::<PerformanceProfiler>()?;
    // m.add_class::<MemoryTracker>()?;
    // m.add_class::<PerformanceUtils>()?;
    // m.add_class::<ProfilerContext>()?;

    // Temporarily disabled due to dependency issues
    // // Models
    // m.add_class::<PyPreTrainedModel>()?;
    // m.add_class::<PyBertModel>()?;
    // m.add_class::<PyGPT2Model>()?;
    // m.add_class::<PyT5Model>()?;
    // m.add_class::<PyLlamaModel>()?;

    // // Task-specific models
    // m.add_class::<PyBertForSequenceClassification>()?;
    // m.add_class::<PyGPT2LMHeadModel>()?;

    // // Tokenizers
    // m.add_class::<PyWordPieceTokenizer>()?;
    // m.add_class::<PyBPETokenizer>()?;

    // // Pipelines
    // m.add_class::<PyTextGenerationPipeline>()?;
    // m.add_class::<PyTextClassificationPipeline>()?;

    // // Training
    // m.add_class::<PyTrainer>()?;
    // m.add_class::<PyTrainingArguments>()?;

    // Utility functions
    m.add_function(wrap_pyfunction!(utils::get_device, m)?)?;
    m.add_function(wrap_pyfunction!(utils::set_seed, m)?)?;
    m.add_function(wrap_pyfunction!(utils::enable_grad, m)?)?;
    m.add_function(wrap_pyfunction!(utils::no_grad, m)?)?;

    // Temporarily disabled due to dependency issues
    // // Auto classes
    // m.add_class::<PyAutoModel>()?;
    // m.add_class::<PyAutoTokenizer>()?;
    // m.add_class::<PyAutoModelForSequenceClassification>()?;
    // m.add_class::<PyAutoModelForTokenClassification>()?;
    // m.add_class::<PyAutoModelForQuestionAnswering>()?;
    // m.add_class::<PyAutoModelForCausalLM>()?;
    // m.add_class::<PyAutoModelForMaskedLM>()?;

    // // Pipeline factory
    // m.add_function(wrap_pyfunction!(pipeline, m)?)?;

    // Add custom exception classes
    errors::add_exceptions(m)?;

    Ok(())
}
