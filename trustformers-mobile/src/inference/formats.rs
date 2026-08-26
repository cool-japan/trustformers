//! Model format detection and execution planning.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::Tensor;

use super::tensor_conversion::natural_cmp;

/// Execution plan for mobile inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub strategy: ExecutionStrategy,
    pub num_layers: usize,
    pub batch_size: usize,
    pub checkpoint_interval: usize,
    /// Deterministic application order for the loaded weight tensors.
    ///
    /// A `HashMap<String, Tensor>` has no defined iteration order, so an
    /// engine that walked `model_weights` directly would apply layers in a
    /// different (arbitrary, hash-seed-dependent) order on every run. This
    /// list is computed once, in [`MobileInferenceEngine::load_model`](crate::inference::engine::MobileInferenceEngine::load_model), with
    /// a "natural" sort (numeric runs inside a name compare numerically, so
    /// `"h.2"` sorts before `"h.10"`) so the same checkpoint always executes
    /// the same way.
    pub ordered_weight_names: Vec<String>,
}
impl ExecutionPlan {
    pub fn new(strategy: ExecutionStrategy, num_layers: usize) -> Self {
        Self {
            strategy,
            num_layers,
            batch_size: 1,
            checkpoint_interval: 0,
            ordered_weight_names: Vec::new(),
        }
    }
    /// Recompute the deterministic execution order from a freshly loaded
    /// weight map.
    pub(super) fn set_layer_order(&mut self, weights: &HashMap<String, Tensor>) {
        let mut names: Vec<String> = weights.keys().cloned().collect();
        names.sort_by(|a, b| natural_cmp(a, b));
        self.num_layers = names.len();
        self.ordered_weight_names = names;
    }
}
/// Execution strategy for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    /// Execute layers sequentially
    Sequential,
    /// Parallelize within layers
    LayerParallel,
    /// Full parallel execution
    FullParallel,
}
/// Supported model formats for loading
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// SafeTensors format (.safetensors)
    SafeTensors,
    /// PyTorch format (.pt, .pth, .bin)
    PyTorch,
    /// ONNX format (.onnx)
    ONNX,
    /// TensorFlow format (.pb)
    TensorFlow,
    /// Unknown or unsupported format
    Unknown,
}
