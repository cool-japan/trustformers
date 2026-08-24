//! Elementwise activation functions used by the model layers.
//!
//! Every function here is a thin wrapper over the corresponding SIMD-dispatching
//! `Tensor` method, so the models get the vectorised kernels instead of the
//! scalar `mapv` loops this module used to carry.

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use std::f32::consts::PI;

/// Minimum element count before a Metal GPU dispatch is worth its two host
/// copies plus command-buffer latency.
#[cfg(all(target_os = "macos", feature = "metal"))]
const MIN_SIZE_FOR_METAL_DISPATCH: usize = 16_384;

pub fn gelu(x: &Tensor) -> Result<Tensor> {
    match x {
        // GPU-resident Metal tensor - process directly on GPU (ZERO TRANSFERS!)
        #[cfg(all(target_os = "macos", feature = "metal"))]
        Tensor::Metal(metal_data) => {
            use crate::gpu_ops::metal::get_metal_backend;
            use crate::tensor::MetalTensorData;

            let backend = get_metal_backend()?;
            let size: usize = metal_data.shape.iter().product();

            // eprintln!(
            //     "✅ GELU: GPU-to-GPU path (Metal→Metal, shape: {:?}, size: {})",
            //     metal_data.shape, size
            // );

            // Execute GELU GPU-to-GPU (NO CPU transfers!)
            let output_buffer_id = backend.gelu_gpu_to_gpu(&metal_data.buffer_id(), size)?;

            Ok(Tensor::Metal(MetalTensorData::new(
                &backend,
                output_buffer_id,
                metal_data.shape.clone(),
                metal_data.dtype,
            )?))
        },

        // GPU-resident CUDA tensor - process directly on GPU (ZERO TRANSFERS!)
        #[cfg(feature = "cuda")]
        Tensor::CUDA(cuda_data) => {
            #[allow(unused_imports)]
            use crate::tensor::CudaTensorData;

            #[cfg(any(target_os = "linux", target_os = "windows"))]
            {
                use crate::gpu_ops::cuda::get_cuda_backend;

                // Address the device the resident buffer actually lives on.
                let device_id = cuda_data.device_id();
                let backend = get_cuda_backend(device_id)?;
                let size: usize = cuda_data.shape.iter().product();

                // Execute GELU GPU-to-GPU (NO CPU transfers!)
                let output_buffer_id = backend.gelu_gpu_to_gpu(&cuda_data.buffer_id(), size)?;

                Ok(Tensor::CUDA(CudaTensorData::new(
                    output_buffer_id,
                    device_id,
                    cuda_data.shape.clone(),
                    cuda_data.dtype,
                )))
            }

            // Fallback for non-Linux/Windows platforms
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            {
                use crate::device::Device;
                let cpu_tensor = Tensor::CUDA(cuda_data.clone()).to_device_enum(&Device::CPU)?;
                gelu(&cpu_tensor)
            }
        },

        Tensor::F32(arr) => {
            // Try Metal GPU acceleration if available.
            //
            // The GPU round trip costs two full host copies plus dispatch
            // latency, so it only pays off once the tensor is large enough for
            // the kernel to dominate; below the threshold the CPU/SIMD path is
            // strictly faster.
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                use crate::gpu_ops::metal::get_metal_backend;
                if arr.len() >= MIN_SIZE_FOR_METAL_DISPATCH {
                    if let Ok(backend) = get_metal_backend() {
                        // Borrow the backing slice when the layout allows it;
                        // copy only for non-contiguous inputs.
                        let contiguous = arr.as_standard_layout();
                        if let Some(input_slice) = contiguous.as_slice() {
                            if let Ok(output_vec) = backend.gelu_f32(input_slice) {
                                use scirs2_core::ndarray::ArrayD;
                                let output_arr = ArrayD::from_shape_vec(arr.raw_dim(), output_vec)
                                    .map_err(|e| {
                                        TrustformersError::tensor_op_error(
                                            &format!("Failed to reshape GELU result: {}", e),
                                            "gelu",
                                        )
                                    })?;
                                return Ok(Tensor::F32(output_arr));
                            }
                        }
                    }
                }
            }

            // Saturation guard for extreme inputs (keeps +/-inf from turning into
            // NaN through `0.5 * v * (1 + tanh(inf))`), then the SIMD-dispatching
            // `Tensor::gelu` for everything in the well-behaved range.
            let needs_guard = arr.iter().any(|v| !(-10.0..=10.0).contains(v));
            if needs_guard {
                let result = arr.mapv(|v| {
                    if v > 10.0 {
                        return v; // GELU(x) ~= x for large positive x
                    } else if v < -10.0 {
                        return 0.0; // GELU(x) ~= 0 for large negative x
                    }
                    let inner = (2.0 / PI).sqrt() * (v + 0.044715 * v.powi(3));
                    let inner_clamped = inner.clamp(-20.0, 20.0);
                    0.5 * v * (1.0 + inner_clamped.tanh())
                });
                return Ok(Tensor::F32(result));
            }

            // SIMD path (>= 256 elements) via `Tensor::gelu`; identical tanh
            // approximation, 4-16 lanes at a time instead of one.
            x.gelu()
        },
        _ => Err(TrustformersError::tensor_op_error(
            "Unsupported tensor type for GELU",
            "gelu",
        )),
    }
}

pub fn gelu_new(x: &Tensor) -> Result<Tensor> {
    match x {
        Tensor::F32(arr) => {
            let result =
                arr.mapv(|v| 0.5 * v * (1.0 + (0.7978845608 * (v + 0.044715 * v.powi(3))).tanh()));
            Ok(Tensor::F32(result))
        },
        _ => Err(TrustformersError::tensor_op_error(
            "Unsupported tensor type for GELU new",
            "gelu_new",
        )),
    }
}

/// ReLU activation.
///
/// Delegates to [`Tensor::relu`], which dispatches to the SIMD kernels for
/// tensors of 256 elements or more. This module used to re-implement every
/// activation as a scalar `mapv`, and it is the module the models actually call
/// (`layers::feedforward`, and the BERT/GPT-2/GPT-NeoX/Gemma/Phi-2/DeBERTa/
/// RoBERTa/DistilBERT/ALBERT/S4 model heads), so the scalar versions were the
/// ones on the hot path.
pub fn relu(x: &Tensor) -> Result<Tensor> {
    x.relu()
}

/// Sigmoid activation (SIMD-accelerated via [`Tensor::sigmoid`]).
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    x.sigmoid()
}

/// Hyperbolic tangent activation (SIMD-accelerated via [`Tensor::tanh`]).
pub fn tanh(x: &Tensor) -> Result<Tensor> {
    x.tanh()
}

/// SiLU (Swish) activation: `SiLU(x) = x * sigmoid(x)`.
///
/// SIMD-accelerated via [`Tensor::silu`].
pub fn silu(x: &Tensor) -> Result<Tensor> {
    x.silu()
}

/// SwiGLU activation function
/// SwiGLU(x, gate) = SiLU(gate) * x
pub fn swiglu(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let activated_gate = silu(gate)?;
    x.mul(&activated_gate)
}
