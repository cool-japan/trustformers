//! Named weight storage for [`super::wasm_model::WasmModel`].
//!
//! Real checkpoint formats (SafeTensors in particular) associate every
//! tensor with a name. Earlier revisions of this crate stored weights as a
//! bare positional `Vec<WasmTensor>`, which made it impossible to tell which
//! tensor was which — the forward pass simply generated random output
//! instead of indexing into it correctly. `NamedWeights` replaces that with
//! an explicit name -> tensor map plus typed lookup helpers that return a
//! structured error (naming exactly what is missing) rather than silently
//! substituting a default.

use crate::core::tensor::WasmTensor;
use std::collections::HashMap;
use std::format;
use std::string::String;
use std::vec::Vec;

/// A name-keyed collection of model weight tensors.
#[derive(Debug, Clone, Default)]
pub struct NamedWeights {
    tensors: HashMap<String, WasmTensor>,
}

impl NamedWeights {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }

    pub fn from_pairs(pairs: Vec<(String, WasmTensor)>) -> Self {
        Self {
            tensors: pairs.into_iter().collect(),
        }
    }

    pub fn insert(&mut self, name: impl Into<String>, tensor: WasmTensor) {
        self.tensors.insert(name.into(), tensor);
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn total_parameters(&self) -> usize {
        self.tensors.values().map(|t| t.len()).sum()
    }

    /// Look up a required tensor's flat data, returning a structured error
    /// listing the missing name (never a silent zero/random fallback).
    pub fn required(&self, name: &str) -> Result<&[f32], String> {
        self.tensors
            .get(name)
            .map(|t| t.data_ref())
            .ok_or_else(|| format!("missing required weight tensor '{name}'"))
    }

    /// Look up an optional tensor (e.g. a bias that some architectures
    /// omit). Returns `Ok(None)` when absent, `Err` only on a genuine
    /// programming error (never applicable here, kept for symmetry).
    pub fn optional(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(|t| t.data_ref())
    }

    pub fn get(&self, name: &str) -> Option<&WasmTensor> {
        self.tensors.get(name)
    }
}

/// Build the per-layer weight name prefix used throughout this crate's
/// checkpoint convention: `layers.<index>.`.
pub fn layer_prefix(index: usize) -> String {
    format!("layers.{index}.")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tensor(v: f32) -> WasmTensor {
        WasmTensor::new(vec![v], vec![1]).expect("valid tensor")
    }

    #[test]
    fn test_required_present() {
        let mut w = NamedWeights::new();
        w.insert("a", tiny_tensor(1.0));
        assert_eq!(w.required("a").unwrap(), &[1.0]);
    }

    #[test]
    fn test_required_missing_reports_name() {
        let w = NamedWeights::new();
        let err = w.required("token_embeddings.weight").unwrap_err();
        assert!(err.contains("token_embeddings.weight"));
    }

    #[test]
    fn test_optional_missing_is_none_not_error() {
        let w = NamedWeights::new();
        assert!(w.optional("bias").is_none());
    }

    #[test]
    fn test_layer_prefix() {
        assert_eq!(layer_prefix(3), "layers.3.");
    }

    #[test]
    fn test_total_parameters() {
        let mut w = NamedWeights::new();
        w.insert("a", WasmTensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap());
        w.insert("b", WasmTensor::new(vec![1.0, 2.0], vec![2]).unwrap());
        assert_eq!(w.total_parameters(), 5);
    }
}
