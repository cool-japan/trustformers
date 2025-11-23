use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use std::f32::consts::PI;

pub fn gelu(x: &Tensor) -> Result<Tensor> {
    match x {
        // GPU-resident Metal tensor - process directly on GPU (ZERO TRANSFERS!)
        #[cfg(feature = "metal")]
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
            let output_buffer_id = backend.gelu_gpu_to_gpu(&metal_data.buffer_id, size)?;

            Ok(Tensor::Metal(MetalTensorData {
                buffer_id: output_buffer_id,
                shape: metal_data.shape.clone(),
                dtype: metal_data.dtype,
            }))
        },

        // GPU-resident CUDA tensor - process directly on GPU (ZERO TRANSFERS!)
        #[cfg(feature = "cuda")]
        Tensor::CUDA(cuda_data) => {
            use crate::device::Device;
            #[allow(unused_imports)]
            use crate::tensor::CudaTensorData;

            #[cfg(any(target_os = "linux", target_os = "windows"))]
            {
                use crate::gpu_ops::cuda::get_cuda_backend;

                // Get device ID (default to 0)
                let device_id = 0; // TODO: Get from tensor metadata
                let backend = get_cuda_backend(device_id)?;
                let size: usize = cuda_data.shape.iter().product();

                // Execute GELU GPU-to-GPU (NO CPU transfers!)
                let output_buffer_id = backend.gelu_gpu_to_gpu(&cuda_data.buffer_id, size)?;

                return Ok(Tensor::CUDA(CudaTensorData {
                    buffer_id: output_buffer_id,
                    shape: cuda_data.shape.clone(),
                    dtype: cuda_data.dtype,
                }));
            }

            // Fallback for non-Linux/Windows platforms
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            {
                let cpu_tensor = Tensor::CUDA(cuda_data.clone()).to_device_enum(&Device::CPU)?;
                return gelu(&cpu_tensor);
            }
        },

        Tensor::F32(arr) => {
            // Try Metal GPU acceleration if available
            #[cfg(feature = "metal")]
            {
                use crate::gpu_ops::metal::get_metal_backend;
                if let Ok(backend) = get_metal_backend() {
                    // Convert tensor to slice
                    let input_vec: Vec<f32> = arr.iter().copied().collect();

                    // Execute on GPU
                    if let Ok(output_vec) = backend.gelu_f32(&input_vec) {
                        // Convert back to tensor
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

            // Fallback to CPU implementation with NaN guarding
            // eprintln!(
            //     "⚠️  GELU: CPU fallback (F32 tensor, shape: {:?})",
            //     arr.shape()
            // );
            let result = arr.mapv(|v| {
                // Clamp extreme values to prevent NaN
                if v > 10.0 {
                    return v; // GELU(x) ≈ x for large positive x
                } else if v < -10.0 {
                    return 0.0; // GELU(x) ≈ 0 for large negative x
                }

                let inner = (2.0 / PI).sqrt() * (v + 0.044715 * v.powi(3));
                let inner_clamped = inner.clamp(-20.0, 20.0);
                0.5 * v * (1.0 + inner_clamped.tanh())
            });
            Ok(Tensor::F32(result))
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

pub fn relu(x: &Tensor) -> Result<Tensor> {
    match x {
        Tensor::F32(arr) => {
            let result = arr.mapv(|v| v.max(0.0));
            Ok(Tensor::F32(result))
        },
        _ => Err(TrustformersError::tensor_op_error(
            "Unsupported tensor type for ReLU",
            "relu",
        )),
    }
}

pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    match x {
        Tensor::F32(arr) => {
            let result = arr.mapv(|v| 1.0 / (1.0 + (-v).exp()));
            Ok(Tensor::F32(result))
        },
        _ => Err(TrustformersError::tensor_op_error(
            "Unsupported tensor type for sigmoid",
            "sigmoid",
        )),
    }
}

pub fn tanh(x: &Tensor) -> Result<Tensor> {
    match x {
        Tensor::F32(arr) => {
            let result = arr.mapv(|v| v.tanh());
            Ok(Tensor::F32(result))
        },
        _ => Err(TrustformersError::tensor_op_error(
            "Unsupported tensor type for tanh",
            "tanh",
        )),
    }
}

/// SiLU (Swish) activation function
/// SiLU(x) = x * sigmoid(x)
pub fn silu(x: &Tensor) -> Result<Tensor> {
    match x {
        Tensor::F32(arr) => {
            let result = arr.mapv(|v| v / (1.0 + (-v).exp()));
            Ok(Tensor::F32(result))
        },
        _ => Err(TrustformersError::tensor_op_error(
            "Unsupported tensor type for SiLU",
            "silu",
        )),
    }
}

/// SwiGLU activation function
/// SwiGLU(x, gate) = SiLU(gate) * x
pub fn swiglu(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let activated_gate = silu(gate)?;
    x.mul(&activated_gate)
}
