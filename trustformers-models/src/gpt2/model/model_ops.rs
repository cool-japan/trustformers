//! GPT-2 activation types and decoding helpers.
//!
//! Split out of `model_blocks.rs` to keep every file under the 2000-line limit.
//! Holds the activation enum used by the MLP block and the free functions the
//! generation paths rely on: causal masking, top-k / top-p filtering,
//! multinomial sampling, softmax / log-softmax and batch stacking.

use scirs2_core::ndarray::{ArrayD, IxDyn};
#[cfg(all(target_os = "macos", feature = "metal"))]
use trustformers_core::device::Device;
use trustformers_core::{
    errors::{invalid_config, tensor_op_error, Result, TrustformersError},
    ops::activations::{gelu as gelu_core, relu, silu},
    tensor::Tensor,
};

/// Activation function types
#[derive(Clone)]
pub(crate) enum ActivationType {
    Gelu,
    Relu,
    Swish,
}

impl ActivationType {
    pub(crate) fn from_str(s: &str) -> Result<Self> {
        match s {
            "gelu" | "gelu_new" | "gelu_fast" => Ok(Self::Gelu),
            "relu" => Ok(Self::Relu),
            "swish" | "silu" => Ok(Self::Swish),
            _ => Err(invalid_config(
                "activation",
                format!("Unknown activation: {}", s),
            )),
        }
    }

    pub(crate) fn apply(&self, x: Tensor) -> Result<Tensor> {
        match self {
            Self::Gelu => gelu_core(&x), // Use NaN-safe version from trustformers_core
            Self::Relu => relu(&x),
            Self::Swish => silu(&x), // SiLU = Swish
        }
    }
}

/// Create a causal mask for attention
pub(crate) fn create_causal_mask(seq_len: usize) -> Result<Tensor> {
    let mut mask = ArrayD::<f32>::zeros(IxDyn(&[1, 1, seq_len, seq_len]));

    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[[0, 0, i, j]] = f32::NEG_INFINITY;
        }
    }

    Ok(Tensor::F32(mask))
}

/// Apply top-k filtering to logits
pub(crate) fn apply_top_k_filtering(logits: ArrayD<f32>, k: usize) -> Result<ArrayD<f32>> {
    let mut result = logits.clone();
    let mut indices_and_values: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(idx, &val)| (idx, val)).collect();

    // Sort by value in descending order
    indices_and_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Set all values outside top-k to -inf
    for (idx, _) in indices_and_values.iter().skip(k) {
        result[*idx] = f32::NEG_INFINITY;
    }

    Ok(result)
}

/// Apply top-p (nucleus) filtering to logits
pub(crate) fn apply_top_p_filtering(logits: ArrayD<f32>, p: f32) -> Result<ArrayD<f32>> {
    // Convert to probabilities
    let probs = softmax(logits.clone())?;

    let mut indices_and_probs: Vec<(usize, f32)> =
        probs.iter().enumerate().map(|(idx, &prob)| (idx, prob)).collect();

    // Sort by probability in descending order
    indices_and_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find the smallest set of tokens with cumulative probability > p
    let mut cumsum = 0.0;
    let mut cutoff_idx = indices_and_probs.len();

    for (i, (_, prob)) in indices_and_probs.iter().enumerate() {
        cumsum += prob;
        if cumsum > p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Create result with -inf for tokens outside the nucleus
    let mut result = logits;
    let selected_indices: std::collections::HashSet<_> =
        indices_and_probs.iter().take(cutoff_idx).map(|(idx, _)| *idx).collect();

    for (idx, val) in result.iter_mut().enumerate() {
        if !selected_indices.contains(&idx) {
            *val = f32::NEG_INFINITY;
        }
    }

    Ok(result)
}

/// Sample from logits using multinomial sampling
pub(crate) fn sample_from_logits(logits: ArrayD<f32>) -> Result<u32> {
    use scirs2_core::random::*; // SciRS2 Integration Policy (includes WeightedIndex)

    // Convert to probabilities
    let probs = softmax(logits)?;

    // Create weighted distribution
    let weights: Vec<f32> = probs.iter().copied().collect();
    let dist = WeightedIndex::new(weights).map_err(|e| {
        TrustformersError::model_error(format!("Failed to create distribution: {}", e))
    })?;

    // Sample
    let mut rng = thread_rng(); // From scirs2_core::random
    Ok(rng.sample(&dist) as u32)
}

/// Compute softmax of logits
pub(crate) fn softmax(logits: ArrayD<f32>) -> Result<ArrayD<f32>> {
    // Find max for numerical stability
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Compute exp(x - max)
    let exp_vals = logits.mapv(|x| (x - max_val).exp());

    // Sum of exp values
    let sum: f32 = exp_vals.iter().sum();

    // Normalize
    Ok(exp_vals / sum)
}

/// Compute log softmax of logits
pub(crate) fn log_softmax(logits: ArrayD<f32>) -> Result<ArrayD<f32>> {
    // Find max for numerical stability
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Compute log(sum(exp(x - max))) + max
    let shifted = logits.mapv(|x| x - max_val);
    let exp_sum = shifted.mapv(|x| x.exp()).sum();
    let log_sum_exp = exp_sum.ln() + max_val;

    // Return log probabilities
    Ok(logits.mapv(|x| x - log_sum_exp))
}

/// Stack a vector of tensors into a batch tensor
pub(crate) fn stack_tensors(tensors: &[Tensor]) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(tensor_op_error(
            "tensor_operation",
            "Cannot stack empty tensor list".to_string(),
        ));
    }

    match &tensors[0] {
        Tensor::F32(first_arr) => {
            let first_shape = first_arr.shape();
            let batch_size = tensors.len();

            // Create new shape with batch dimension
            let mut new_shape = vec![batch_size];
            new_shape.extend_from_slice(first_shape);

            // Collect all tensor data
            let mut data = Vec::new();
            for tensor in tensors {
                match tensor {
                    Tensor::F32(arr) => {
                        if arr.shape() != first_shape {
                            return Err(TrustformersError::shape_error(
                                "All tensors must have the same shape for stacking".to_string(),
                            ));
                        }
                        data.extend(arr.iter().cloned());
                    },
                    _ => {
                        return Err(tensor_op_error(
                            "tensor_operation",
                            "All tensors must be F32 for stacking".to_string(),
                        ))
                    },
                }
            }

            // Create stacked array
            let stacked = ArrayD::from_shape_vec(IxDyn(&new_shape), data).map_err(|_| {
                TrustformersError::shape_error("Failed to create stacked tensor".into())
            })?;

            Ok(Tensor::F32(stacked))
        },
        #[cfg(all(target_os = "macos", feature = "metal"))]
        Tensor::Metal(first_data) => {
            use trustformers_core::gpu_ops::metal::get_metal_backend;
            use trustformers_core::tensor::MetalTensorData;

            // Try to use GPU stacking kernel
            if let Ok(backend) = get_metal_backend() {
                // All tensors must have the same shape
                let first_shape = &first_data.shape;
                if first_shape.len() == 2 {
                    let seq_len = first_shape[0];
                    let hidden_size = first_shape[1];

                    // Collect all buffer IDs
                    let buffer_ids: Vec<_> = tensors
                        .iter()
                        .map(|t| match t {
                            Tensor::Metal(data) => Ok(data.buffer_id),
                            _ => Err(TrustformersError::tensor_op_error(
                                "All tensors must be Metal for GPU stacking",
                                "stack_tensors",
                            )),
                        })
                        .collect::<Result<Vec<_>>>()?;

                    // Stack on GPU
                    let stacked_buffer_id =
                        backend.stack_gpu_buffers(&buffer_ids, seq_len, hidden_size)?;

                    // Create output shape: [batch_size, seq_len, hidden_size]
                    let output_shape = vec![tensors.len(), seq_len, hidden_size];

                    return Ok(Tensor::Metal(MetalTensorData {
                        buffer_id: stacked_buffer_id,
                        shape: output_shape,
                        dtype: first_data.dtype,
                    }));
                }
            }

            // Fallback: convert to CPU, stack, then convert back to Metal
            let cpu_tensors: Vec<Tensor> = tensors
                .iter()
                .map(|t| t.to_device_enum(&Device::CPU))
                .collect::<Result<Vec<_>>>()?;

            let cpu_stacked = stack_tensors(&cpu_tensors)?;

            let metal_device = Device::Metal(0);
            let metal_stacked = cpu_stacked.to_device_enum(&metal_device)?;

            Ok(metal_stacked)
        },
        _ => Err(tensor_op_error(
            "tensor_operation",
            "Only F32 tensors supported for stacking".to_string(),
        )),
    }
}
