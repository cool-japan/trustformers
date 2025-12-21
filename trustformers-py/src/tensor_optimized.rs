use crate::errors::{TrustformersPyError, TrustformersPyResult};
use numpy::{PyArray, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use scirs2_core::ndarray::{self as ndarray, ArrayD, Axis, Dimension, IxDyn};
use scirs2_core::parallel_ops::*; // SciRS2 Integration Policy - replaces rayon
use trustformers_core::quantization::{
    QuantizationConfig, QuantizationScheme, QuantizedTensor, Quantizer,
};
use trustformers_core::tensor::Tensor;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Optimized tensor operations with SIMD and parallelization
pub struct TensorOptimizer;

impl TensorOptimizer {
    /// Optimized element-wise addition using SIMD
    #[cfg(target_arch = "x86_64")]
    pub fn simd_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) -> TrustformersPyResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TrustformersPyError::InvalidInputError {
                message: "Arrays must have the same length".to_string(),
            });
        }

        let len = a.len();
        let simd_len = len - (len % 8); // Use 8-element chunks for better compatibility

        // Process 8 elements at a time using AVX/AVX2
        for i in (0..simd_len).step_by(8) {
            unsafe {
                if is_x86_feature_detected!("avx") {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                    let result_vec = _mm256_add_ps(a_vec, b_vec);
                    _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
                } else if is_x86_feature_detected!("sse") {
                    // Process 4 elements at a time with SSE
                    let a_vec1 = _mm_loadu_ps(a.as_ptr().add(i));
                    let b_vec1 = _mm_loadu_ps(b.as_ptr().add(i));
                    let result_vec1 = _mm_add_ps(a_vec1, b_vec1);
                    _mm_storeu_ps(result.as_mut_ptr().add(i), result_vec1);

                    let a_vec2 = _mm_loadu_ps(a.as_ptr().add(i + 4));
                    let b_vec2 = _mm_loadu_ps(b.as_ptr().add(i + 4));
                    let result_vec2 = _mm_add_ps(a_vec2, b_vec2);
                    _mm_storeu_ps(result.as_mut_ptr().add(i + 4), result_vec2);
                } else {
                    // Scalar fallback
                    for j in i..i + 8 {
                        result[j] = a[j] + b[j];
                    }
                }
            }
        }

        // Handle remaining elements
        for i in simd_len..len {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    /// Fallback implementation for non-x86_64 architectures
    #[cfg(not(target_arch = "x86_64"))]
    pub fn simd_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) -> TrustformersPyResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TrustformersPyError::InvalidInputError {
                message: "Arrays must have the same length".to_string(),
            });
        }

        // Use parallel iteration for better performance on non-x86 platforms
        result.par_iter_mut().zip(a.par_iter().zip(b.par_iter())).for_each(
            |(r, (&a_val, &b_val))| {
                *r = a_val + b_val;
            },
        );

        Ok(())
    }

    /// Optimized matrix multiplication using blocked algorithm and SIMD
    pub fn optimized_matmul(a: &ArrayD<f32>, b: &ArrayD<f32>) -> TrustformersPyResult<ArrayD<f32>> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(TrustformersPyError::InvalidInputError {
                message: "Matrix multiplication requires 2D arrays".to_string(),
            });
        }

        if a_shape[1] != b_shape[0] {
            return Err(TrustformersPyError::ShapeMismatchError {
                expected: vec![a_shape[0], b_shape[1]],
                actual: vec![a_shape[1], b_shape[0]],
            });
        }

        let m = a_shape[0];
        let n = b_shape[1];
        let k = a_shape[1];

        let mut result = ArrayD::zeros(IxDyn(&[m, n]));

        // Use blocked matrix multiplication for better cache performance
        const BLOCK_SIZE: usize = 64;

        for i_block in (0..m).step_by(BLOCK_SIZE) {
            for j_block in (0..n).step_by(BLOCK_SIZE) {
                for k_block in (0..k).step_by(BLOCK_SIZE) {
                    let i_end = (i_block + BLOCK_SIZE).min(m);
                    let j_end = (j_block + BLOCK_SIZE).min(n);
                    let k_end = (k_block + BLOCK_SIZE).min(k);

                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = 0.0f32;
                            for k_idx in k_block..k_end {
                                sum += a[[i, k_idx]] * b[[k_idx, j]];
                            }
                            result[[i, j]] += sum;
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Parallel reduction operations
    pub fn parallel_sum(tensor: &ArrayD<f32>) -> f32 {
        if let Some(slice) = tensor.as_slice() {
            slice.par_iter().sum()
        } else {
            // Fallback for non-contiguous arrays
            tensor.par_iter().sum()
        }
    }

    pub fn parallel_max(tensor: &ArrayD<f32>) -> Option<f32> {
        if let Some(slice) = tensor.as_slice() {
            slice.par_iter().cloned().reduce(|| f32::NEG_INFINITY, f32::max).into()
        } else {
            tensor.par_iter().cloned().reduce(|| f32::NEG_INFINITY, f32::max).into()
        }
    }

    pub fn parallel_min(tensor: &ArrayD<f32>) -> Option<f32> {
        if let Some(slice) = tensor.as_slice() {
            slice.par_iter().cloned().reduce(|| f32::INFINITY, f32::min).into()
        } else {
            tensor.par_iter().cloned().reduce(|| f32::INFINITY, f32::min).into()
        }
    }

    /// Memory-efficient tensor reshape with copy avoidance
    pub fn efficient_reshape(
        tensor: &ArrayD<f32>,
        new_shape: &[usize],
    ) -> TrustformersPyResult<ArrayD<f32>> {
        let total_elements: usize = tensor.len();
        let new_total: usize = new_shape.iter().product();

        if total_elements != new_total {
            return Err(TrustformersPyError::ShapeMismatchError {
                expected: vec![total_elements],
                actual: vec![new_total],
            });
        }

        // Try to reshape without copying if memory layout allows
        if let Ok(reshaped) = tensor.view().into_shape_with_order(IxDyn(new_shape)) {
            Ok(reshaped.to_owned())
        } else {
            // Need to create a contiguous copy
            let contiguous = tensor.as_standard_layout();
            Ok(contiguous
                .into_shape_with_order(IxDyn(new_shape))
                .map_err(|e| TrustformersPyError::TensorError {
                    message: format!("Reshape failed: {}", e),
                })?
                .to_owned())
        }
    }

    /// Optimized softmax with numerical stability
    pub fn stable_softmax(
        tensor: &ArrayD<f32>,
        axis: Option<usize>,
    ) -> TrustformersPyResult<ArrayD<f32>> {
        let axis = axis.unwrap_or(tensor.ndim() - 1);

        if axis >= tensor.ndim() {
            return Err(TrustformersPyError::InvalidInputError {
                message: format!(
                    "Axis {} out of bounds for tensor with {} dimensions",
                    axis,
                    tensor.ndim()
                ),
            });
        }

        let mut result = tensor.clone();

        // For each slice perpendicular to the specified axis
        let other_axes: Vec<usize> = (0..tensor.ndim()).filter(|&i| i != axis).collect();

        if other_axes.is_empty() {
            // 1D case
            let max_val = tensor.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            result.mapv_inplace(|x| (x - max_val).exp());
            let sum_val: f32 = result.sum();
            if sum_val > 0.0 {
                result.mapv_inplace(|x| x / sum_val);
            }
        } else {
            // Multi-dimensional case: iterate over all combinations of other axes
            for mut lane in result.lanes_mut(Axis(axis)) {
                // Find max for numerical stability
                let max_val = lane.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                // Subtract max and compute exp
                lane.mapv_inplace(|x| (x - max_val).exp());

                // Normalize
                let sum_val: f32 = lane.sum();
                if sum_val > 0.0 {
                    lane.mapv_inplace(|x| x / sum_val);
                }
            }
        }

        Ok(result)
    }

    /// Optimized layer normalization implementation
    pub fn layer_norm(
        input: &ArrayD<f32>,
        weight: Option<&ArrayD<f32>>,
        bias: Option<&ArrayD<f32>>,
        eps: f32,
        normalized_shape: &[usize],
    ) -> TrustformersPyResult<ArrayD<f32>> {
        let input_shape = input.shape();
        let mut result = input.clone();

        // Determine the axes to normalize over (last `normalized_shape.len()` dimensions)
        let norm_axes: Vec<usize> =
            (input_shape.len() - normalized_shape.len()..input_shape.len()).collect();

        // Compute mean and variance for each normalized slice
        let total_elements: usize = norm_axes.iter().map(|&i| input_shape[i]).product();

        // Calculate statistics and normalize
        for mut slice in result.lanes_mut(Axis(0)) {
            // Compute mean
            let mean = slice.sum() / total_elements as f32;

            // Compute variance
            let variance = slice.mapv(|x| (x - mean).powi(2)).sum() / total_elements as f32;

            // Normalize
            let std_dev = (variance + eps).sqrt();
            slice.mapv_inplace(|x| (x - mean) / std_dev);
        }

        // Apply weight and bias if provided
        if let Some(w) = weight {
            result = &result * w;
        }
        if let Some(b) = bias {
            result = &result + b;
        }

        Ok(result)
    }

    /// Optimized scaled dot-product attention computation
    pub fn scaled_dot_product_attention(
        query: &ArrayD<f32>,
        key: &ArrayD<f32>,
        value: &ArrayD<f32>,
        mask: Option<&ArrayD<f32>>,
        _dropout_p: f32,
        scale: Option<f32>,
    ) -> TrustformersPyResult<ArrayD<f32>> {
        let q_shape = query.shape();
        let k_shape = key.shape();
        let v_shape = value.shape();

        if q_shape.len() < 2 || k_shape.len() < 2 || v_shape.len() < 2 {
            return Err(TrustformersPyError::InvalidInputError {
                message: "Query, key, and value must have at least 2 dimensions".to_string(),
            });
        }

        let seq_len_q = q_shape[q_shape.len() - 2];
        let seq_len_k = k_shape[k_shape.len() - 2];
        let d_k = q_shape[q_shape.len() - 1];
        let d_v = v_shape[v_shape.len() - 1];

        if k_shape[k_shape.len() - 1] != d_k {
            return Err(TrustformersPyError::ShapeMismatchError {
                expected: vec![d_k],
                actual: vec![k_shape[k_shape.len() - 1]],
            });
        }

        // Compute attention scores: Q @ K^T
        let _key_t = key.clone().reversed_axes(); // Transpose last two dimensions

        // For simplicity, compute attention for 2D case (can be extended for batch dims)
        let mut scores = ArrayD::<f32>::zeros(IxDyn(&[seq_len_q, seq_len_k]));

        // Manual matrix multiplication for attention scores
        for i in 0..seq_len_q {
            for j in 0..seq_len_k {
                let mut sum = 0.0;
                for k in 0..d_k {
                    sum += query[[i, k]] * key[[j, k]];
                }
                scores[[i, j]] = sum;
            }
        }

        // Scale by sqrt(d_k) or provided scale
        let scale_factor = scale.unwrap_or(1.0 / (d_k as f32).sqrt());
        scores.mapv_inplace(|x| x * scale_factor);

        // Apply mask if provided
        if let Some(mask_tensor) = mask {
            scores = &scores + mask_tensor;
        }

        // Apply softmax
        let attention_weights = Self::stable_softmax(&scores, Some(1))?;

        // Compute attention output: weights @ V
        let mut output = ArrayD::<f32>::zeros(IxDyn(&[seq_len_q, d_v]));
        for i in 0..seq_len_q {
            for j in 0..d_v {
                let mut sum = 0.0;
                for k in 0..seq_len_k {
                    sum += attention_weights[[i, k]] * value[[k, j]];
                }
                output[[i, j]] = sum;
            }
        }

        Ok(output)
    }

    /// GELU activation function (Gaussian Error Linear Unit)
    pub fn gelu(input: &ArrayD<f32>) -> ArrayD<f32> {
        input.mapv(|x| {
            0.5 * x
                * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
        })
    }

    /// SiLU/Swish activation function
    pub fn silu(input: &ArrayD<f32>) -> ArrayD<f32> {
        input.mapv(|x| x / (1.0 + (-x).exp()))
    }

    /// Mish activation function
    pub fn mish(input: &ArrayD<f32>) -> ArrayD<f32> {
        input.mapv(|x| x * (1.0 + x.exp()).ln().tanh())
    }

    /// ReLU activation function
    pub fn relu(input: &ArrayD<f32>) -> ArrayD<f32> {
        input.mapv(|x| x.max(0.0))
    }

    /// Leaky ReLU activation function
    pub fn leaky_relu(input: &ArrayD<f32>, negative_slope: f32) -> ArrayD<f32> {
        input.mapv(|x| if x > 0.0 { x } else { negative_slope * x })
    }

    /// Fast approximate GELU implementation
    pub fn gelu_fast(input: &ArrayD<f32>) -> ArrayD<f32> {
        input.mapv(|x| 0.5 * x * (1.0 + (1.702 * x).tanh()))
    }

    /// RoPE (Rotary Position Embedding) implementation
    pub fn apply_rotary_pos_emb(
        tensor: &ArrayD<f32>,
        cos: &ArrayD<f32>,
        sin: &ArrayD<f32>,
    ) -> TrustformersPyResult<ArrayD<f32>> {
        let shape = tensor.shape();
        if shape.len() < 2 {
            return Err(TrustformersPyError::InvalidInputError {
                message: "Input tensor must have at least 2 dimensions".to_string(),
            });
        }

        let seq_len = shape[shape.len() - 2];
        let d_model = shape[shape.len() - 1];

        if d_model % 2 != 0 {
            return Err(TrustformersPyError::InvalidInputError {
                message: "Model dimension must be even for RoPE".to_string(),
            });
        }

        let mut result = tensor.clone();

        // Apply rotation to pairs of dimensions
        for i in 0..seq_len {
            for j in (0..d_model).step_by(2) {
                let x1 = tensor[[i, j]];
                let x2 = tensor[[i, j + 1]];
                let cos_val = cos[[i, j / 2]];
                let sin_val = sin[[i, j / 2]];

                result[[i, j]] = x1 * cos_val - x2 * sin_val;
                result[[i, j + 1]] = x1 * sin_val + x2 * cos_val;
            }
        }

        Ok(result)
    }
}

/// Memory pool for reusing tensor allocations
pub struct TensorMemoryPool {
    pools: std::collections::HashMap<Vec<usize>, Vec<ArrayD<f32>>>,
}

/// Python wrapper for optimized tensor operations
#[pyclass(name = "TensorOptimized", module = "trustformers")]
pub struct PyTensorOptimized;

#[pymethods]
impl PyTensorOptimized {
    #[new]
    pub fn new() -> Self {
        PyTensorOptimized
    }

    /// Apply GELU activation function
    #[staticmethod]
    pub fn gelu(py: Python<'_>, input: &Bound<'_, PyArray<f32, IxDyn>>) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = TensorOptimizer::gelu(&input_array);
        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();

        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Apply SiLU/Swish activation function
    #[staticmethod]
    pub fn silu(py: Python<'_>, input: &Bound<'_, PyArray<f32, IxDyn>>) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = TensorOptimizer::silu(&input_array);
        let result_shape = result.shape().to_vec();
        let result_data = result.as_slice().unwrap();

        let py_array = PyArray::from_slice(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Apply Mish activation function
    #[staticmethod]
    pub fn mish(py: Python<'_>, input: &Bound<'_, PyArray<f32, IxDyn>>) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = TensorOptimizer::mish(&input_array);
        let result_shape = result.shape().to_vec();
        let result_data = result.as_slice().unwrap();

        let py_array = PyArray::from_slice(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Apply fast GELU activation function
    #[staticmethod]
    pub fn gelu_fast(
        py: Python<'_>,
        input: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = TensorOptimizer::gelu_fast(&input_array);
        let result_shape = result.shape().to_vec();
        let result_data = result.as_slice().unwrap();

        let py_array = PyArray::from_slice(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Apply stable softmax
    #[staticmethod]
    pub fn softmax(
        py: Python<'_>,
        input: &Bound<'_, PyArray<f32, IxDyn>>,
        axis: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = TensorOptimizer::stable_softmax(&input_array, axis)
            .map_err(|e| PyValueError::new_err(format!("Softmax error: {:?}", e)))?;
        let result_shape = result.shape().to_vec();
        let result_data = result.as_slice().unwrap();

        let py_array = PyArray::from_slice(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Optimized matrix multiplication
    #[staticmethod]
    pub fn matmul(
        py: Python<'_>,
        a: &Bound<'_, PyArray<f32, IxDyn>>,
        b: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<Py<PyAny>> {
        let a_array = a.try_readonly()?.as_array().to_owned();
        let b_array = b.try_readonly()?.as_array().to_owned();
        let result = TensorOptimizer::optimized_matmul(&a_array, &b_array)
            .map_err(|e| PyValueError::new_err(format!("Matrix multiplication error: {:?}", e)))?;
        let result_shape = result.shape().to_vec();
        let result_data = result.as_slice().unwrap();

        let py_array = PyArray::from_slice(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Layer normalization
    #[staticmethod]
    pub fn layer_norm(
        py: Python<'_>,
        input: &Bound<'_, PyArray<f32, IxDyn>>,
        weight: Option<&Bound<'_, PyArray<f32, IxDyn>>>,
        bias: Option<&Bound<'_, PyArray<f32, IxDyn>>>,
        eps: f32,
        normalized_shape: Vec<usize>,
    ) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let weight_array = weight.map(|w| w.try_readonly().unwrap().as_array().to_owned());
        let bias_array = bias.map(|b| b.try_readonly().unwrap().as_array().to_owned());

        let result = TensorOptimizer::layer_norm(
            &input_array,
            weight_array.as_ref(),
            bias_array.as_ref(),
            eps,
            &normalized_shape,
        )
        .map_err(|e| PyValueError::new_err(format!("Layer norm error: {:?}", e)))?;

        let result_shape = result.shape().to_vec();
        let result_data = result.as_slice().unwrap();

        let py_array = PyArray::from_slice(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Scaled dot-product attention
    #[staticmethod]
    pub fn scaled_dot_product_attention(
        py: Python<'_>,
        query: &Bound<'_, PyArray<f32, IxDyn>>,
        key: &Bound<'_, PyArray<f32, IxDyn>>,
        value: &Bound<'_, PyArray<f32, IxDyn>>,
        mask: Option<&Bound<'_, PyArray<f32, IxDyn>>>,
        dropout_p: f32,
        scale: Option<f32>,
    ) -> PyResult<Py<PyAny>> {
        let query_array = query.try_readonly()?.as_array().to_owned();
        let key_array = key.try_readonly()?.as_array().to_owned();
        let value_array = value.try_readonly()?.as_array().to_owned();
        let mask_array = mask.map(|m| m.try_readonly().unwrap().as_array().to_owned());

        let result = TensorOptimizer::scaled_dot_product_attention(
            &query_array,
            &key_array,
            &value_array,
            mask_array.as_ref(),
            dropout_p,
            scale,
        )
        .map_err(|e| PyValueError::new_err(format!("Attention error: {:?}", e)))?;

        let result_shape = result.shape().to_vec();
        let result_data = result.as_slice().unwrap();

        let py_array = PyArray::from_slice(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Apply RoPE (Rotary Position Embedding)
    #[staticmethod]
    pub fn apply_rotary_pos_emb(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        cos: &Bound<'_, PyArray<f32, IxDyn>>,
        sin: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<Py<PyAny>> {
        let tensor_array = tensor.try_readonly()?.as_array().to_owned();
        let cos_array = cos.try_readonly()?.as_array().to_owned();
        let sin_array = sin.try_readonly()?.as_array().to_owned();

        let result = TensorOptimizer::apply_rotary_pos_emb(&tensor_array, &cos_array, &sin_array)
            .map_err(|e| PyValueError::new_err(format!("RoPE error: {:?}", e)))?;

        let result_shape = result.shape().to_vec();
        let result_data = result.as_slice().unwrap();

        let py_array = PyArray::from_slice(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Tensor concatenation along specified axis
    #[staticmethod]
    pub fn concatenate(
        py: Python<'_>,
        tensors: &Bound<'_, PyList>,
        axis: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        if tensors.is_empty() {
            return Err(PyValueError::new_err(
                "Cannot concatenate empty list of tensors",
            ));
        }

        let axis = axis.unwrap_or(0);
        let arrays: Result<Vec<ArrayD<f32>>, PyErr> = tensors
            .iter()
            .map(|item| {
                let tensor = item.cast::<PyArray<f32, IxDyn>>()?;
                Ok(tensor.try_readonly()?.as_array().to_owned())
            })
            .collect();
        let arrays = arrays?;

        // Simple concatenation implementation
        if arrays.len() == 1 {
            let result_shape = arrays[0].shape();
            let result_data: Vec<f32> = arrays[0].iter().cloned().collect();
            let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
            return Ok(py_array.into_any().unbind());
        }

        // For simplicity, concatenate along axis 0 (first dimension)
        let mut all_data = Vec::new();
        let mut total_size_axis = 0;
        let base_shape = arrays[0].shape();

        for array in &arrays {
            if array.ndim() != base_shape.len() {
                return Err(PyValueError::new_err(
                    "All arrays must have same number of dimensions",
                ));
            }
            total_size_axis += array.len_of(Axis(axis));
            all_data.extend(array.iter().cloned());
        }

        let mut new_shape = base_shape.to_vec();
        new_shape[axis] = total_size_axis;

        let py_array = PyArray::from_vec(py, all_data).reshape(new_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Element-wise logarithm
    #[staticmethod]
    pub fn log(py: Python<'_>, input: &Bound<'_, PyArray<f32, IxDyn>>) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = input_array.mapv(|x| if x > 0.0 { x.ln() } else { f32::NEG_INFINITY });

        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();

        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Element-wise exponential
    #[staticmethod]
    pub fn exp(py: Python<'_>, input: &Bound<'_, PyArray<f32, IxDyn>>) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = input_array.mapv(|x| x.exp());

        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();

        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Element-wise square root
    #[staticmethod]
    pub fn sqrt(py: Python<'_>, input: &Bound<'_, PyArray<f32, IxDyn>>) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = input_array.mapv(|x| if x >= 0.0 { x.sqrt() } else { f32::NAN });

        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();

        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Element-wise power
    #[staticmethod]
    pub fn pow(
        py: Python<'_>,
        input: &Bound<'_, PyArray<f32, IxDyn>>,
        exponent: f32,
    ) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = input_array.mapv(|x| x.powf(exponent));

        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();

        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Tensor reshape
    #[staticmethod]
    pub fn reshape(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        new_shape: Vec<usize>,
    ) -> PyResult<Py<PyAny>> {
        let tensor_array = tensor.try_readonly()?.as_array().to_owned();

        // Validate that total elements match
        let current_size: usize = tensor_array.len();
        let new_size: usize = new_shape.iter().product();

        if current_size != new_size {
            return Err(PyValueError::new_err(format!(
                "Cannot reshape tensor of size {} to shape {:?} (size {})",
                current_size, new_shape, new_size
            )));
        }

        let result_data: Vec<f32> = tensor_array.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(new_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Tensor transpose (reverse all axes)
    #[staticmethod]
    pub fn transpose(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<Py<PyAny>> {
        let tensor_array = tensor.try_readonly()?.as_array().to_owned();
        let result = tensor_array.clone().reversed_axes();

        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();

        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Reduce sum along specified axes
    #[staticmethod]
    pub fn reduce_sum(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        axis: Option<usize>,
        keepdims: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let tensor_array = tensor.try_readonly()?.as_array().to_owned();
        let keepdims = keepdims.unwrap_or(false);

        let result = if let Some(ax) = axis {
            if ax >= tensor_array.ndim() {
                return Err(PyValueError::new_err(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    tensor_array.ndim()
                )));
            }
            tensor_array.sum_axis(Axis(ax))
        } else {
            // Sum over all elements
            let total_sum = tensor_array.sum();
            ArrayD::from_elem(vec![], total_sum)
        };

        let result_shape = if keepdims && axis.is_some() {
            let mut shape = tensor_array.shape().to_vec();
            shape[axis.unwrap()] = 1;
            shape
        } else {
            result.shape().to_vec()
        };

        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Reduce mean along specified axes
    #[staticmethod]
    pub fn reduce_mean(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        axis: Option<usize>,
        keepdims: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let tensor_array = tensor.try_readonly()?.as_array().to_owned();
        let keepdims = keepdims.unwrap_or(false);

        let result = if let Some(ax) = axis {
            if ax >= tensor_array.ndim() {
                return Err(PyValueError::new_err(format!(
                    "Axis {} out of bounds for array of dimension {}",
                    ax,
                    tensor_array.ndim()
                )));
            }
            tensor_array
                .mean_axis(Axis(ax))
                .ok_or_else(|| PyValueError::new_err("Mean calculation failed"))?
        } else {
            // Mean over all elements
            let total_mean = tensor_array
                .mean()
                .ok_or_else(|| PyValueError::new_err("Mean calculation failed"))?;
            ArrayD::from_elem(vec![], total_mean)
        };

        let result_shape = if keepdims && axis.is_some() {
            let mut shape = tensor_array.shape().to_vec();
            shape[axis.unwrap()] = 1;
            shape
        } else {
            result.shape().to_vec()
        };

        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Element-wise clamp (clip values between min and max)
    #[staticmethod]
    pub fn clamp(
        py: Python<'_>,
        input: &Bound<'_, PyArray<f32, IxDyn>>,
        min_value: Option<f32>,
        max_value: Option<f32>,
    ) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();

        let result = input_array.mapv(|x| {
            let mut val = x;
            if let Some(min_val) = min_value {
                val = val.max(min_val);
            }
            if let Some(max_val) = max_value {
                val = val.min(max_val);
            }
            val
        });

        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();

        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Element-wise absolute value
    #[staticmethod]
    pub fn abs(py: Python<'_>, input: &Bound<'_, PyArray<f32, IxDyn>>) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = input_array.mapv(|x| x.abs());

        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();

        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Element-wise sign function
    #[staticmethod]
    pub fn sign(py: Python<'_>, input: &Bound<'_, PyArray<f32, IxDyn>>) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = input_array.mapv(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        });

        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();

        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Element-wise sine function
    #[staticmethod]
    pub fn sin(py: Python<'_>, input: &Bound<'_, PyArray<f32, IxDyn>>) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = input_array.mapv(|x| x.sin());
        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Element-wise cosine function
    #[staticmethod]
    pub fn cos(py: Python<'_>, input: &Bound<'_, PyArray<f32, IxDyn>>) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = input_array.mapv(|x| x.cos());
        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Element-wise tangent function
    #[staticmethod]
    pub fn tan(py: Python<'_>, input: &Bound<'_, PyArray<f32, IxDyn>>) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = input_array.mapv(|x| x.tan());
        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Element-wise maximum between two tensors
    #[staticmethod]
    pub fn maximum(
        py: Python<'_>,
        input1: &Bound<'_, PyArray<f32, IxDyn>>,
        input2: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<Py<PyAny>> {
        let array1 = input1.try_readonly()?.as_array().to_owned();
        let array2 = input2.try_readonly()?.as_array().to_owned();
        if array1.shape() != array2.shape() {
            return Err(PyValueError::new_err(
                "Input tensors must have the same shape",
            ));
        }
        let result = ndarray::Zip::from(&array1).and(&array2).map_collect(|&a, &b| a.max(b));
        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Element-wise minimum between two tensors
    #[staticmethod]
    pub fn minimum(
        py: Python<'_>,
        input1: &Bound<'_, PyArray<f32, IxDyn>>,
        input2: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<Py<PyAny>> {
        let array1 = input1.try_readonly()?.as_array().to_owned();
        let array2 = input2.try_readonly()?.as_array().to_owned();
        if array1.shape() != array2.shape() {
            return Err(PyValueError::new_err(
                "Input tensors must have the same shape",
            ));
        }
        let result = ndarray::Zip::from(&array1).and(&array2).map_collect(|&a, &b| a.min(b));
        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Remove dimensions of size 1
    #[staticmethod]
    pub fn squeeze(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        axis: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let array = tensor.try_readonly()?.as_array().to_owned();
        let new_shape: Vec<usize> = if let Some(axis) = axis {
            if axis >= array.ndim() {
                return Err(PyValueError::new_err("Axis out of bounds"));
            }
            if array.shape()[axis] != 1 {
                return Err(PyValueError::new_err("Cannot squeeze axis with size != 1"));
            }
            array
                .shape()
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != axis)
                .map(|(_, &s)| s)
                .collect()
        } else {
            array.shape().iter().filter(|&&s| s != 1).cloned().collect()
        };
        let result_data: Vec<f32> = array.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(new_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Add dimensions of size 1
    #[staticmethod]
    pub fn unsqueeze(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        axis: usize,
    ) -> PyResult<Py<PyAny>> {
        let array = tensor.try_readonly()?.as_array().to_owned();
        if axis > array.ndim() {
            return Err(PyValueError::new_err("Axis out of bounds"));
        }
        let mut new_shape = Vec::from(array.shape());
        new_shape.insert(axis, 1);
        let result_data: Vec<f32> = array.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(new_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Quantize tensor to INT8 format
    #[staticmethod]
    pub fn quantize_int8(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        symmetric: Option<bool>,
        per_channel: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let array = tensor.try_readonly()?.as_array().to_owned();
        let tensor_core = Tensor::from(array.into_dyn());

        let config = QuantizationConfig {
            scheme: QuantizationScheme::Int8,
            symmetric: symmetric.unwrap_or(true),
            per_channel: per_channel.unwrap_or(false),
            calibration_samples: Some(128),
            group_size: Some(128),
            bnb_config: None,
        };

        let quantized = Quantizer::quantize(&tensor_core, &config)
            .map_err(|e| PyValueError::new_err(format!("Quantization failed: {}", e)))?;

        // Return as Python dictionary with quantized data
        let result_dict = PyDict::new(py);
        let data_list = PyList::new(py, &quantized.data)?;
        let scale_list = PyList::new(py, &quantized.scale)?;
        let zero_point_list = PyList::new(py, &quantized.zero_point)?;
        let shape_list = PyList::new(py, &quantized.shape)?;
        result_dict.set_item("data", data_list)?;
        result_dict.set_item("scale", scale_list)?;
        result_dict.set_item("zero_point", zero_point_list)?;
        result_dict.set_item("shape", shape_list)?;
        result_dict.set_item("scheme", "int8")?;
        result_dict.set_item("per_channel", quantized.per_channel)?;

        Ok(result_dict.into_any().unbind())
    }

    /// Quantize tensor to INT4 format
    #[staticmethod]
    pub fn quantize_int4(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        symmetric: Option<bool>,
        per_channel: Option<bool>,
        group_size: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let array = tensor.try_readonly()?.as_array().to_owned();
        let tensor_core = Tensor::from(array.into_dyn());

        let config = QuantizationConfig {
            scheme: QuantizationScheme::Int4,
            symmetric: symmetric.unwrap_or(true),
            per_channel: per_channel.unwrap_or(false),
            calibration_samples: Some(128),
            group_size: group_size.or(Some(128)),
            bnb_config: None,
        };

        let quantized = Quantizer::quantize(&tensor_core, &config)
            .map_err(|e| PyValueError::new_err(format!("Quantization failed: {}", e)))?;

        // Return as Python dictionary with quantized data
        let result_dict = PyDict::new(py);
        let data_list = PyList::new(py, &quantized.data)?;
        let scale_list = PyList::new(py, &quantized.scale)?;
        let zero_point_list = PyList::new(py, &quantized.zero_point)?;
        let shape_list = PyList::new(py, &quantized.shape)?;
        result_dict.set_item("data", data_list)?;
        result_dict.set_item("scale", scale_list)?;
        result_dict.set_item("zero_point", zero_point_list)?;
        result_dict.set_item("shape", shape_list)?;
        result_dict.set_item("scheme", "int4")?;
        result_dict.set_item("per_channel", quantized.per_channel)?;

        Ok(result_dict.into_any().unbind())
    }

    /// Dynamic quantization (runtime quantization)
    #[staticmethod]
    pub fn quantize_dynamic(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        per_channel: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let array = tensor.try_readonly()?.as_array().to_owned();
        let tensor_core = Tensor::from(array.into_dyn());

        let config = QuantizationConfig {
            scheme: QuantizationScheme::Dynamic,
            symmetric: true,
            per_channel: per_channel.unwrap_or(false),
            calibration_samples: None, // Dynamic quantization doesn't need calibration
            group_size: Some(128),
            bnb_config: None,
        };

        let quantized = Quantizer::quantize(&tensor_core, &config)
            .map_err(|e| PyValueError::new_err(format!("Quantization failed: {}", e)))?;

        // Return as Python dictionary with quantized data
        let result_dict = PyDict::new(py);
        let data_list = PyList::new(py, &quantized.data)?;
        let scale_list = PyList::new(py, &quantized.scale)?;
        let zero_point_list = PyList::new(py, &quantized.zero_point)?;
        let shape_list = PyList::new(py, &quantized.shape)?;
        result_dict.set_item("data", data_list)?;
        result_dict.set_item("scale", scale_list)?;
        result_dict.set_item("zero_point", zero_point_list)?;
        result_dict.set_item("shape", shape_list)?;
        result_dict.set_item("scheme", "dynamic")?;
        result_dict.set_item("per_channel", quantized.per_channel)?;

        Ok(result_dict.into_any().unbind())
    }

    /// Dequantize tensor back to FP32
    #[staticmethod]
    pub fn dequantize(py: Python<'_>, quantized_data: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        // Extract quantized tensor data from Python dictionary
        let data: Vec<u8> = quantized_data
            .get_item("data")?
            .ok_or_else(|| PyValueError::new_err("Missing 'data' field"))?
            .extract()?;
        let scale: Vec<f32> = quantized_data
            .get_item("scale")?
            .ok_or_else(|| PyValueError::new_err("Missing 'scale' field"))?
            .extract()?;
        let zero_point: Vec<i32> = quantized_data
            .get_item("zero_point")?
            .ok_or_else(|| PyValueError::new_err("Missing 'zero_point' field"))?
            .extract()?;
        let shape: Vec<usize> = quantized_data
            .get_item("shape")?
            .ok_or_else(|| PyValueError::new_err("Missing 'shape' field"))?
            .extract()?;
        let scheme_str: String = quantized_data
            .get_item("scheme")?
            .ok_or_else(|| PyValueError::new_err("Missing 'scheme' field"))?
            .extract()?;
        let per_channel: bool = quantized_data
            .get_item("per_channel")?
            .ok_or_else(|| PyValueError::new_err("Missing 'per_channel' field"))?
            .extract()?;

        let scheme = match scheme_str.as_str() {
            "int8" => QuantizationScheme::Int8,
            "int4" => QuantizationScheme::Int4,
            "dynamic" => QuantizationScheme::Dynamic,
            "bnb8bit" => QuantizationScheme::BnB8bit,
            "bnb4bit" => QuantizationScheme::BnB4bit,
            "gptq" => QuantizationScheme::GPTQ,
            "awq" => QuantizationScheme::AWQ,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown quantization scheme: {}",
                    scheme_str
                )))
            },
        };

        // Create QuantizedTensor
        let quantized_tensor =
            QuantizedTensor::new(data, scale, zero_point, shape.clone(), scheme, per_channel);

        // Dequantize back to FP32
        let dequantized = quantized_tensor
            .dequantize()
            .map_err(|e| PyValueError::new_err(format!("Dequantization failed: {}", e)))?;

        // Convert back to numpy array
        let result_data = dequantized
            .to_vec_f32()
            .map_err(|e| PyValueError::new_err(format!("Failed to convert to vector: {}", e)))?;
        let py_array = PyArray::from_vec(py, result_data).reshape(shape)?;

        Ok(py_array.into_any().unbind())
    }

    /// Get quantization statistics and recommendations
    #[staticmethod]
    pub fn analyze_quantization_impact(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        schemes: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        let array = tensor.try_readonly()?.as_array().to_owned();
        let tensor_core = Tensor::from(array.into_dyn());

        let schemes_to_test = schemes.unwrap_or_else(|| {
            vec![
                "int8".to_string(),
                "int4".to_string(),
                "dynamic".to_string(),
            ]
        });
        let result_dict = PyDict::new(py);

        for scheme_str in schemes_to_test {
            let scheme = match scheme_str.as_str() {
                "int8" => QuantizationScheme::Int8,
                "int4" => QuantizationScheme::Int4,
                "dynamic" => QuantizationScheme::Dynamic,
                _ => continue,
            };

            let config = QuantizationConfig {
                scheme,
                symmetric: true,
                per_channel: false,
                calibration_samples: Some(128),
                group_size: Some(128),
                bnb_config: None,
            };

            if let Ok(quantized) = Quantizer::quantize(&tensor_core, &config) {
                if let Ok(dequantized) = quantized.dequantize() {
                    // Calculate compression ratio and quality metrics
                    let original_size = tensor_core.len() * 4; // f32 = 4 bytes
                    let quantized_size = quantized.data.len()
                        + quantized.scale.len() * 4
                        + quantized.zero_point.len() * 4;
                    let compression_ratio = original_size as f32 / quantized_size as f32;

                    // Calculate MSE between original and dequantized
                    let original_data = tensor_core.to_vec_f32().unwrap();
                    let dequantized_data = dequantized.to_vec_f32().unwrap();

                    let mse = original_data
                        .iter()
                        .zip(dequantized_data.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        / original_data.len() as f32;

                    let scheme_result = PyDict::new(py);
                    scheme_result.set_item("compression_ratio", compression_ratio)?;
                    scheme_result.set_item("mse", mse)?;
                    scheme_result.set_item("original_size_bytes", original_size)?;
                    scheme_result.set_item("quantized_size_bytes", quantized_size)?;

                    result_dict.set_item(&scheme_str, scheme_result)?;
                }
            }
        }

        Ok(result_dict.into_any().unbind())
    }

    /// Element-wise addition with broadcasting support
    #[staticmethod]
    pub fn add(
        py: Python<'_>,
        input1: &Bound<'_, PyArray<f32, IxDyn>>,
        input2: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<Py<PyAny>> {
        let array1 = input1.try_readonly()?.as_array().to_owned();
        let array2 = input2.try_readonly()?.as_array().to_owned();

        // Simple broadcasting: check if shapes are compatible
        if array1.shape() == array2.shape() {
            let result = &array1 + &array2;
            let result_shape = result.shape();
            let result_data: Vec<f32> = result.iter().cloned().collect();
            let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
            Ok(py_array.into_any().unbind())
        } else if array2.len() == 1 {
            // Scalar broadcasting
            let scalar = array2.first().unwrap_or(&0.0);
            let result = array1.mapv(|x| x + scalar);
            let result_shape = result.shape();
            let result_data: Vec<f32> = result.iter().cloned().collect();
            let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
            Ok(py_array.into_any().unbind())
        } else if array1.len() == 1 {
            // Scalar broadcasting (reversed)
            let scalar = array1.first().unwrap_or(&0.0);
            let result = array2.mapv(|x| scalar + x);
            let result_shape = result.shape();
            let result_data: Vec<f32> = result.iter().cloned().collect();
            let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
            Ok(py_array.into_any().unbind())
        } else {
            Err(PyValueError::new_err(
                "Cannot broadcast tensors: incompatible shapes",
            ))
        }
    }

    /// Element-wise multiplication with broadcasting support
    #[staticmethod]
    pub fn multiply(
        py: Python<'_>,
        input1: &Bound<'_, PyArray<f32, IxDyn>>,
        input2: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<Py<PyAny>> {
        let array1 = input1.try_readonly()?.as_array().to_owned();
        let array2 = input2.try_readonly()?.as_array().to_owned();

        // Simple broadcasting: check if shapes are compatible
        if array1.shape() == array2.shape() {
            let result = &array1 * &array2;
            let result_shape = result.shape();
            let result_data: Vec<f32> = result.iter().cloned().collect();
            let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
            Ok(py_array.into_any().unbind())
        } else if array2.len() == 1 {
            // Scalar broadcasting
            let scalar = array2.first().unwrap_or(&1.0);
            let result = array1.mapv(|x| x * scalar);
            let result_shape = result.shape();
            let result_data: Vec<f32> = result.iter().cloned().collect();
            let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
            Ok(py_array.into_any().unbind())
        } else if array1.len() == 1 {
            // Scalar broadcasting (reversed)
            let scalar = array1.first().unwrap_or(&1.0);
            let result = array2.mapv(|x| scalar * x);
            let result_shape = result.shape();
            let result_data: Vec<f32> = result.iter().cloned().collect();
            let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
            Ok(py_array.into_any().unbind())
        } else {
            Err(PyValueError::new_err(
                "Cannot broadcast tensors: incompatible shapes",
            ))
        }
    }

    /// Gather elements along an axis
    #[staticmethod]
    pub fn gather(
        py: Python<'_>,
        input: &Bound<'_, PyArray<f32, IxDyn>>,
        indices: &Bound<'_, PyArray<i64, IxDyn>>,
        axis: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let array = input.try_readonly()?.as_array().to_owned();
        let indices_array = indices.try_readonly()?.as_array().to_owned();
        let axis = axis.unwrap_or(0);

        if axis >= array.ndim() {
            return Err(PyValueError::new_err("Axis out of bounds"));
        }

        let mut result_data = Vec::new();
        let axis_size = array.len_of(ndarray::Axis(axis));

        for &idx in indices_array.iter() {
            if idx < 0 || idx as usize >= axis_size {
                return Err(PyValueError::new_err("Index out of bounds"));
            }

            let slice = array.index_axis(ndarray::Axis(axis), idx as usize);
            result_data.extend(slice.iter().cloned());
        }

        let mut result_shape = array.shape().to_vec();
        result_shape[axis] = indices_array.len();

        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Create a tensor filled with ones
    #[staticmethod]
    pub fn ones(py: Python<'_>, shape: Vec<usize>) -> PyResult<Py<PyAny>> {
        let total_elements: usize = shape.iter().product();
        let data = vec![1.0f32; total_elements];
        let py_array = PyArray::from_vec(py, data).reshape(shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Create a tensor filled with zeros
    #[staticmethod]
    pub fn zeros(py: Python<'_>, shape: Vec<usize>) -> PyResult<Py<PyAny>> {
        let total_elements: usize = shape.iter().product();
        let data = vec![0.0f32; total_elements];
        let py_array = PyArray::from_vec(py, data).reshape(shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Create a tensor filled with a specific value
    #[staticmethod]
    pub fn full(py: Python<'_>, shape: Vec<usize>, fill_value: f32) -> PyResult<Py<PyAny>> {
        let total_elements: usize = shape.iter().product();
        let data = vec![fill_value; total_elements];
        let py_array = PyArray::from_vec(py, data).reshape(shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Create a random tensor with normal distribution
    #[staticmethod]
    pub fn randn(
        py: Python<'_>,
        shape: Vec<usize>,
        mean: Option<f32>,
        std: Option<f32>,
    ) -> PyResult<Py<PyAny>> {
        use scirs2_core::random::*;

        let total_elements: usize = shape.iter().product();
        let mean_val = mean.unwrap_or(0.0);
        let std_val = std.unwrap_or(1.0);

        let normal = Normal::new(mean_val, std_val).map_err(|e| {
            PyValueError::new_err(format!("Invalid normal distribution parameters: {}", e))
        })?;

        let mut rng = thread_rng();
        let data: Vec<f32> = (0..total_elements).map(|_| normal.sample(&mut rng)).collect();

        let py_array = PyArray::from_vec(py, data).reshape(shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Repeat tensor along specified axes
    #[staticmethod]
    pub fn repeat(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        repeats: Vec<usize>,
    ) -> PyResult<Py<PyAny>> {
        let array = tensor.try_readonly()?.as_array().to_owned();

        if repeats.len() != array.ndim() {
            return Err(PyValueError::new_err(
                "Repeats length must match tensor dimensions",
            ));
        }

        // Calculate new shape
        let new_shape: Vec<usize> =
            array.shape().iter().zip(repeats.iter()).map(|(dim, rep)| dim * rep).collect();

        let total_elements: usize = new_shape.iter().product();
        let mut result_data: Vec<f32> = Vec::with_capacity(total_elements);

        // Simple implementation for demonstration - can be optimized
        let original_data: Vec<f32> = array.iter().cloned().collect();
        let _elements_per_repeat = array.len();

        for _ in 0..repeats.iter().product::<usize>() / array.len() {
            result_data.extend(original_data.iter());
        }

        let py_array = PyArray::from_vec(py, result_data).reshape(new_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Slice tensor along a specific axis
    #[staticmethod]
    pub fn slice(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        axis: usize,
        start: usize,
        end: usize,
    ) -> PyResult<Py<PyAny>> {
        let array = tensor.try_readonly()?.as_array().to_owned();

        if axis >= array.ndim() {
            return Err(PyValueError::new_err("Axis out of bounds"));
        }

        if start >= end || end > array.len_of(ndarray::Axis(axis)) {
            return Err(PyValueError::new_err("Invalid slice indices"));
        }

        let slice = array.slice_axis(ndarray::Axis(axis), ndarray::Slice::from(start..end));
        let result_data: Vec<f32> = slice.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(slice.shape())?;
        Ok(py_array.into_any().unbind())
    }

    /// Multi-dimensional slice of the tensor
    #[staticmethod]
    pub fn slice_multi(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        ranges: Vec<(usize, usize)>,
    ) -> PyResult<Py<PyAny>> {
        let array = tensor.try_readonly()?.as_array().to_owned();

        if ranges.len() != array.ndim() {
            return Err(PyValueError::new_err(
                "Slice dimensions must match tensor dimensions",
            ));
        }

        let mut result = array;
        for (i, &(start, end)) in ranges.iter().enumerate() {
            if end > result.len_of(ndarray::Axis(i)) {
                return Err(PyValueError::new_err(format!(
                    "Slice end {} exceeds dimension size for axis {}",
                    end, i
                )));
            }
            result =
                result.slice_axis(ndarray::Axis(i), ndarray::Slice::from(start..end)).to_owned();
        }

        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result.shape())?;
        Ok(py_array.into_any().unbind())
    }

    /// Select elements along a dimension at specific index
    #[staticmethod]
    pub fn select(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        dim: usize,
        index: i64,
    ) -> PyResult<Py<PyAny>> {
        let array = tensor.try_readonly()?.as_array().to_owned();

        if dim >= array.ndim() {
            return Err(PyValueError::new_err("Dimension out of bounds"));
        }

        let dim_size = array.len_of(ndarray::Axis(dim)) as i64;
        let actual_index = if index < 0 { dim_size + index } else { index };

        if actual_index < 0 || actual_index >= dim_size {
            return Err(PyValueError::new_err("Index out of bounds"));
        }

        let selected = array.index_axis(ndarray::Axis(dim), actual_index as usize);
        let result_data: Vec<f32> = selected.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(selected.shape())?;
        Ok(py_array.into_any().unbind())
    }

    /// Conditional element selection (where condition)
    #[staticmethod]
    pub fn where_cond(
        py: Python<'_>,
        condition: &Bound<'_, PyArray<f32, IxDyn>>,
        input_true: &Bound<'_, PyArray<f32, IxDyn>>,
        input_false: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<Py<PyAny>> {
        let cond_array = condition.try_readonly()?.as_array().to_owned();
        let true_array = input_true.try_readonly()?.as_array().to_owned();
        let false_array = input_false.try_readonly()?.as_array().to_owned();

        if cond_array.shape() != true_array.shape() || cond_array.shape() != false_array.shape() {
            return Err(PyValueError::new_err(
                "All input tensors must have the same shape",
            ));
        }

        let result = ndarray::Zip::from(&cond_array)
            .and(&true_array)
            .and(&false_array)
            .map_collect(|&c, &t, &f| if c != 0.0 { t } else { f });

        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Masked fill operation
    #[staticmethod]
    pub fn masked_fill(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        mask: &Bound<'_, PyArray<f32, IxDyn>>,
        value: f32,
    ) -> PyResult<Py<PyAny>> {
        let array = tensor.try_readonly()?.as_array().to_owned();
        let mask_array = mask.try_readonly()?.as_array().to_owned();

        if array.shape() != mask_array.shape() {
            return Err(PyValueError::new_err(
                "Tensor and mask must have the same shape",
            ));
        }

        let result = ndarray::Zip::from(&array).and(&mask_array).map_collect(|&x, &m| {
            if m != 0.0 {
                value
            } else {
                x
            }
        });

        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Masked select operation
    #[staticmethod]
    pub fn masked_select(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        mask: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<Py<PyAny>> {
        let array = tensor.try_readonly()?.as_array().to_owned();
        let mask_array = mask.try_readonly()?.as_array().to_owned();

        if array.shape() != mask_array.shape() {
            return Err(PyValueError::new_err(
                "Tensor and mask must have the same shape",
            ));
        }

        let result_data: Vec<f32> = array
            .iter()
            .zip(mask_array.iter())
            .filter_map(|(&x, &m)| if m != 0.0 { Some(x) } else { None })
            .collect();

        let result_shape = vec![result_data.len()];
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Advanced indexing with list of indices
    #[staticmethod]
    pub fn index_select(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        dim: usize,
        indices: &Bound<'_, PyArray<i64, IxDyn>>,
    ) -> PyResult<Py<PyAny>> {
        let array = tensor.try_readonly()?.as_array().to_owned();
        let indices_array = indices.try_readonly()?.as_array().to_owned();

        if dim >= array.ndim() {
            return Err(PyValueError::new_err("Dimension out of bounds"));
        }

        let dim_size = array.len_of(ndarray::Axis(dim)) as i64;
        let mut result_data = Vec::new();

        for &idx in indices_array.iter() {
            let actual_index = if idx < 0 { dim_size + idx } else { idx };

            if actual_index < 0 || actual_index >= dim_size {
                return Err(PyValueError::new_err("Index out of bounds"));
            }

            let slice = array.index_axis(ndarray::Axis(dim), actual_index as usize);
            result_data.extend(slice.iter().cloned());
        }

        let mut result_shape = array.shape().to_vec();
        result_shape[dim] = indices_array.len();

        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Non-zero elements (returns indices)
    #[staticmethod]
    pub fn nonzero(py: Python<'_>, tensor: &Bound<'_, PyArray<f32, IxDyn>>) -> PyResult<Py<PyAny>> {
        let array = tensor.try_readonly()?.as_array().to_owned();

        let mut indices: Vec<Vec<usize>> = vec![Vec::new(); array.ndim()];

        for (idx, &value) in array.indexed_iter() {
            if value != 0.0 {
                for (dim, &coord) in idx.as_array_view().iter().enumerate() {
                    indices[dim].push(coord);
                }
            }
        }

        // Return as list of arrays (one for each dimension)
        let result_list = PyList::empty(py);
        for dim_indices in indices {
            let dim_indices_i64: Vec<i64> = dim_indices.iter().map(|&x| x as i64).collect();
            let py_array = numpy::PyArray::from_vec(py, dim_indices_i64);
            result_list.append(py_array)?;
        }

        Ok(result_list.into_any().unbind())
    }

    /// Flip tensor along specified axes
    #[staticmethod]
    pub fn flip(
        py: Python<'_>,
        tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        dims: Option<Vec<usize>>,
    ) -> PyResult<Py<PyAny>> {
        let array = tensor.try_readonly()?.as_array().to_owned();
        let dims_to_flip = dims.unwrap_or_else(|| (0..array.ndim()).collect());

        let mut result = array;
        for &dim in &dims_to_flip {
            if dim >= result.ndim() {
                return Err(PyValueError::new_err("Dimension out of bounds"));
            }
            result.invert_axis(ndarray::Axis(dim));
        }

        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result.shape())?;
        Ok(py_array.into_any().unbind())
    }
}

/// Advanced activation functions for neural networks
#[pyclass(name = "AdvancedActivations")]
pub struct PyAdvancedActivations;

#[pymethods]
impl PyAdvancedActivations {
    #[new]
    pub fn new() -> Self {
        PyAdvancedActivations
    }

    /// Swish activation function (x * sigmoid(x))
    #[staticmethod]
    pub fn swish(py: Python<'_>, input: &Bound<'_, PyArray<f32, IxDyn>>) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = input_array.mapv(|x| x * (1.0 / (1.0 + (-x).exp())));
        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// ELU (Exponential Linear Unit) activation function
    #[staticmethod]
    pub fn elu(
        py: Python<'_>,
        input: &Bound<'_, PyArray<f32, IxDyn>>,
        alpha: Option<f32>,
    ) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let alpha_val = alpha.unwrap_or(1.0);
        let result = input_array.mapv(|x| if x >= 0.0 { x } else { alpha_val * (x.exp() - 1.0) });
        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Hardtanh activation function (clamped linear)
    #[staticmethod]
    pub fn hardtanh(
        py: Python<'_>,
        input: &Bound<'_, PyArray<f32, IxDyn>>,
        min_val: Option<f32>,
        max_val: Option<f32>,
    ) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let min_value = min_val.unwrap_or(-1.0);
        let max_value = max_val.unwrap_or(1.0);
        let result = input_array.mapv(|x| x.max(min_value).min(max_value));
        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// SELU (Scaled Exponential Linear Unit) activation function
    #[staticmethod]
    pub fn selu(py: Python<'_>, input: &Bound<'_, PyArray<f32, IxDyn>>) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let alpha = 1.6732632423543772848170429916717;
        let scale = 1.0507009873554804934193349852946;
        let result =
            input_array.mapv(
                |x| {
                    if x >= 0.0 {
                        scale * x
                    } else {
                        scale * alpha * (x.exp() - 1.0)
                    }
                },
            );
        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Hardswish activation function
    #[staticmethod]
    pub fn hardswish(
        py: Python<'_>,
        input: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = input_array.mapv(|x| {
            let relu6 = (x + 3.0).max(0.0).min(6.0);
            x * relu6 / 6.0
        });
        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Softplus activation function (smooth approximation of ReLU)
    #[staticmethod]
    pub fn softplus(
        py: Python<'_>,
        input: &Bound<'_, PyArray<f32, IxDyn>>,
        beta: Option<f32>,
    ) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let beta_val = beta.unwrap_or(1.0);
        let result = input_array.mapv(|x| (1.0 + (beta_val * x).exp()).ln() / beta_val);
        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Softsign activation function (smooth alternative to tanh)
    #[staticmethod]
    pub fn softsign(py: Python<'_>, input: &Bound<'_, PyArray<f32, IxDyn>>) -> PyResult<Py<PyAny>> {
        let input_array = input.try_readonly()?.as_array().to_owned();
        let result = input_array.mapv(|x| x / (1.0 + x.abs()));
        let result_shape = result.shape();
        let result_data: Vec<f32> = result.iter().cloned().collect();
        let py_array = PyArray::from_vec(py, result_data).reshape(result_shape)?;
        Ok(py_array.into_any().unbind())
    }
}

impl TensorMemoryPool {
    pub fn new() -> Self {
        Self {
            pools: std::collections::HashMap::new(),
        }
    }

    pub fn get_tensor(&mut self, shape: &[usize]) -> ArrayD<f32> {
        let shape_vec = shape.to_vec();
        if let Some(pool) = self.pools.get_mut(&shape_vec) {
            if let Some(mut tensor) = pool.pop() {
                tensor.fill(0.0);
                return tensor;
            }
        }
        ArrayD::zeros(IxDyn(shape))
    }

    pub fn return_tensor(&mut self, tensor: ArrayD<f32>) {
        let shape = tensor.shape().to_vec();
        self.pools.entry(shape).or_insert_with(Vec::new).push(tensor);
    }
}

// Thread-local memory pool for efficient tensor allocation
thread_local! {
    static MEMORY_POOL: std::cell::RefCell<TensorMemoryPool> =
        std::cell::RefCell::new(TensorMemoryPool::new());
}

/// Get a tensor from the thread-local memory pool
pub fn get_pooled_tensor(shape: &[usize]) -> ArrayD<f32> {
    MEMORY_POOL.with(|pool| pool.borrow_mut().get_tensor(shape))
}

/// Return a tensor to the thread-local memory pool
pub fn return_pooled_tensor(tensor: ArrayD<f32>) {
    MEMORY_POOL.with(|pool| pool.borrow_mut().return_tensor(tensor));
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_simd_add() {
        let a = vec![1.0f32; 1000];
        let b = vec![2.0f32; 1000];
        let mut result = vec![0.0f32; 1000];

        TensorOptimizer::simd_add_f32(&a, &b, &mut result).unwrap();

        for &val in &result {
            assert_abs_diff_eq!(val, 3.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_optimized_matmul() {
        let a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = TensorOptimizer::optimized_matmul(&a, &b).unwrap();
        let expected =
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![22.0, 28.0, 49.0, 64.0]).unwrap();

        assert_eq!(result.shape(), expected.shape());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(r, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_stable_softmax() {
        let tensor =
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = TensorOptimizer::stable_softmax(&tensor, Some(1)).unwrap();

        // Check that each row sums to approximately 1.0
        for row in result.axis_iter(Axis(0)) {
            let sum: f32 = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }
}
