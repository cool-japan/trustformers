//! Small pure/PyO3-boundary helper functions shared by two or more of the
//! model wrapper submodules (`bert`, `gpt2`, `rwkv`, `mamba`, `tasks`). Split
//! out of `models/mod.rs` (2026-08-24, `py-followups`, keeping this crate's
//! files under the 2000-line policy); behavior is unchanged, this is a pure
//! code-motion. Everything here is `pub(super)`: visible to sibling
//! submodules of `models`, not part of this crate's public API.
//!
//! `parse_rwkv_config`/`rwkv_config_to_dict` and
//! `parse_mamba_config`/`mamba_config_to_dict` -- the other config helpers
//! that used to sit in this same block of the pre-split file -- moved into
//! `rwkv.rs`/`mamba.rs` instead, next to their one and only caller each; only
//! the genuinely cross-architecture helpers stayed here.

use super::inputs;
use super::weights::trustformers_error_to_py_err;
use crate::tensor::PyTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scirs2_core::ndarray::{ArrayD, IxDyn}; // SciRS2 Integration Policy
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::TokenizedInput;
use trustformers_models::bert::BertConfig;

/// Extract token IDs from a tensor of integer or float token values.
pub(super) fn extract_token_ids(tensor: &Tensor) -> PyResult<Vec<i64>> {
    match tensor {
        Tensor::I64(arr) => Ok(arr.iter().copied().collect()),
        Tensor::F32(arr) => Ok(arr.iter().map(|&x| x as i64).collect()),
        _ => Err(PyValueError::new_err(
            "Input tensor must contain integer token IDs",
        )),
    }
}

/// Build a [`TokenizedInput`] from the tensors a `forward` binding receives.
///
/// PyO3 boundary over [`inputs::tokenized_input_from_parts`], which holds the
/// pure (and unit-tested) conversion and validation logic.
pub(super) fn tokenized_input_from_tensors(
    input_ids: &PyTensor,
    attention_mask: Option<&PyTensor>,
    token_type_ids: Option<&PyTensor>,
) -> PyResult<TokenizedInput> {
    inputs::tokenized_input_from_parts(
        &input_ids.inner,
        attention_mask.map(|tensor| &tensor.inner),
        token_type_ids.map(|tensor| &tensor.inner),
    )
    .map_err(trustformers_error_to_py_err)
}

/// Extract non-negative token IDs, for use with
/// [`trustformers_core::generation::TextGenerator`], which indexes with `usize`.
pub(super) fn extract_token_ids_usize(tensor: &Tensor) -> PyResult<Vec<usize>> {
    extract_token_ids(tensor)?
        .into_iter()
        .map(|id| {
            usize::try_from(id).map_err(|_| {
                PyValueError::new_err(format!(
                    "token id {id} is negative and cannot index a vocabulary"
                ))
            })
        })
        .collect()
}

/// Build a 1-D `I64` token-id tensor from a slice of ids.
pub(super) fn build_token_tensor(tokens: &[i64]) -> PyResult<Tensor> {
    Ok(Tensor::I64(
        ArrayD::from_shape_vec(IxDyn(&[tokens.len()]), tokens.to_vec())
            .map_err(|e| PyValueError::new_err(format!("Failed to create tensor: {}", e)))?,
    ))
}

/// Build a 1-D `I64` token-id [`PyTensor`] from a slice of `usize` ids, the
/// output shape [`TextGenerator::generate`] produces.
pub(super) fn build_usize_token_tensor(tokens: &[usize]) -> PyResult<PyTensor> {
    let ids: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
    Ok(PyTensor {
        inner: build_token_tensor(&ids)?,
        variable: None,
    })
}

/// Argmax over the vocabulary axis of the final timestep of a `[seq, vocab]` logit tensor.
pub(super) fn argmax_last_token(logits: &Tensor) -> PyResult<usize> {
    let shape = logits.shape();
    let vocab = *shape
        .last()
        .ok_or_else(|| PyValueError::new_err("Logits tensor has no dimensions"))?;
    if vocab == 0 {
        return Err(PyValueError::new_err(
            "Logits tensor has an empty vocabulary axis",
        ));
    }
    let data = logits.to_vec_f32().map_err(trustformers_error_to_py_err)?;
    let last_offset = data
        .len()
        .checked_sub(vocab)
        .ok_or_else(|| PyValueError::new_err("Logits tensor is smaller than the vocabulary size"))?;
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (j, &v) in data[last_offset..].iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = j;
        }
    }
    Ok(best_idx)
}

// Helper functions for config parsing

pub(super) fn parse_bert_config(config_dict: &Bound<'_, PyAny>) -> PyResult<BertConfig> {
    let dict = config_dict.cast::<pyo3::types::PyDict>()?;

    let mut config = BertConfig::default();

    if let Ok(Some(val)) = dict.get_item("vocab_size") {
        config.vocab_size = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("hidden_size") {
        config.hidden_size = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("num_hidden_layers") {
        config.num_hidden_layers = val.extract()?;
    }
    if let Ok(Some(val)) = dict.get_item("num_attention_heads") {
        config.num_attention_heads = val.extract()?;
    }

    Ok(config)
}

pub(super) fn config_to_dict<'py>(
    py: Python<'py>,
    config: &BertConfig,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("vocab_size", config.vocab_size)?;
    dict.set_item("hidden_size", config.hidden_size)?;
    dict.set_item("num_hidden_layers", config.num_hidden_layers)?;
    dict.set_item("num_attention_heads", config.num_attention_heads)?;
    dict.set_item("intermediate_size", config.intermediate_size)?;
    dict.set_item("hidden_act", &config.hidden_act)?;
    dict.set_item("hidden_dropout_prob", config.hidden_dropout_prob)?;
    dict.set_item(
        "attention_probs_dropout_prob",
        config.attention_probs_dropout_prob,
    )?;
    dict.set_item("max_position_embeddings", config.max_position_embeddings)?;
    dict.set_item("type_vocab_size", config.type_vocab_size)?;
    dict.set_item("initializer_range", config.initializer_range)?;
    dict.set_item("layer_norm_eps", config.layer_norm_eps)?;
    Ok(dict)
}
