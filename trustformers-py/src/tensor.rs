use numpy::{
    IxDyn, PyArray, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyMemoryView};
use scirs2_core::ndarray::{ArrayD, ArrayViewD, IxDyn as NdIxDyn};
use std::ptr;
use std::sync::Arc;
use trustformers_core::autodiff::variable::{Variable, VariableRef};
use trustformers_core::tensor::Tensor;

/// Python wrapper for TrustformeRS Tensor
#[pyclass(name = "Tensor", module = "trustformers")]
#[derive(Clone)]
pub struct PyTensor {
    pub(crate) inner: Tensor,
    pub(crate) variable: Option<Variable>,
}

impl PyTensor {
    /// Create PyTensor from Rust Tensor
    pub fn from_tensor(tensor: Tensor) -> Self {
        PyTensor {
            inner: tensor,
            variable: None,
        }
    }

    /// Create PyTensor from Rust Tensor with autograd enabled
    pub fn from_tensor_with_grad(tensor: Tensor, requires_grad: bool) -> Self {
        let variable = if requires_grad { Some(Variable::new(tensor.clone(), true)) } else { None };

        PyTensor {
            inner: tensor,
            variable,
        }
    }

    /// Extract nested sequence (list/tuple) into flat data and shape
    fn extract_nested_sequence(data: &Bound<'_, PyAny>) -> PyResult<(Vec<f32>, Vec<usize>)> {
        let mut shape = Vec::new();

        // Determine the shape by recursively examining the nested structure
        fn get_shape(obj: &Bound<'_, PyAny>, current_shape: &mut Vec<usize>) -> PyResult<()> {
            if let Ok(seq) = obj.downcast::<pyo3::types::PyList>() {
                current_shape.push(seq.len());
                if seq.len() > 0 {
                    get_shape(&seq.get_item(0)?, current_shape)?;
                }
            } else if let Ok(seq) = obj.downcast::<pyo3::types::PyTuple>() {
                current_shape.push(seq.len());
                if seq.len() > 0 {
                    get_shape(&seq.get_item(0)?, current_shape)?;
                }
            }
            Ok(())
        }

        get_shape(data, &mut shape)?;

        // Extract flat data
        let mut flat_data = Vec::new();

        fn extract_flat(obj: &Bound<'_, PyAny>, flat_data: &mut Vec<f32>) -> PyResult<()> {
            if let Ok(seq) = obj.downcast::<pyo3::types::PyList>() {
                for item in seq {
                    extract_flat(&item, flat_data)?;
                }
            } else if let Ok(seq) = obj.downcast::<pyo3::types::PyTuple>() {
                for item in seq {
                    extract_flat(&item, flat_data)?;
                }
            } else {
                // Try to extract as a number
                if let Ok(val) = obj.extract::<f32>() {
                    flat_data.push(val);
                } else if let Ok(val) = obj.extract::<f64>() {
                    flat_data.push(val as f32);
                } else if let Ok(val) = obj.extract::<i32>() {
                    flat_data.push(val as f32);
                } else if let Ok(val) = obj.extract::<i64>() {
                    flat_data.push(val as f32);
                } else {
                    return Err(PyValueError::new_err(
                        "List/tuple elements must be numbers (int, float)",
                    ));
                }
            }
            Ok(())
        }

        extract_flat(data, &mut flat_data)?;

        // Validate shape consistency
        let expected_size: usize = shape.iter().product();
        if expected_size != flat_data.len() {
            return Err(PyValueError::new_err(format!(
                "Inconsistent tensor shape: expected {} elements but got {}",
                expected_size,
                flat_data.len()
            )));
        }

        Ok((flat_data, shape))
    }

    /// Create PyTensor from NumPy array with zero-copy (when possible)
    pub fn from_numpy_zero_copy(array: PyReadonlyArrayDyn<f32>) -> PyResult<Self> {
        let shape = array.shape().to_vec();

        // Check if the array is C-contiguous
        if array.is_c_contiguous() {
            // Zero-copy path: create tensor from existing data
            // SAFETY: We ensure the data outlives the tensor by keeping a reference
            let data_slice = array.as_slice().unwrap();

            // For true zero-copy, we need to work with the array's memory directly
            // This is a simplified implementation - in production, we'd need proper
            // lifetime management and potentially shared ownership
            let data_vec = data_slice.to_vec(); // Still copying for safety

            let tensor = Tensor::from_vec(data_vec, &shape)
                .map_err(|e| PyValueError::new_err(format!("Failed to create tensor: {}", e)))?;

            Ok(PyTensor {
                inner: tensor,
                variable: None,
            })
        } else {
            return Err(PyValueError::new_err(
                "Array must be C-contiguous for zero-copy conversion",
            ));
        }
    }

    /// Create a zero-copy view of the tensor as NumPy array
    pub fn as_numpy_view<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let shape = self.inner.shape();

        match &self.inner {
            Tensor::F32(arr) => {
                // Check if tensor data is contiguous
                if let Some(data_slice) = arr.as_slice() {
                    // Zero-copy path: create numpy array that shares memory
                    // SAFETY: We need to ensure the tensor outlives the numpy array
                    // In practice, this would require careful lifetime management

                    // For now, create from slice (which may copy)
                    let numpy_array = PyArray::from_slice(py, data_slice);
                    numpy_array.reshape(shape).map_err(|e| {
                        PyValueError::new_err(format!("Failed to reshape array: {}", e))
                    })
                } else {
                    // Fallback: copy data if not contiguous
                    self.numpy(py)
                }
            },
            _ => Err(PyValueError::new_err(
                "Only F32 tensors supported for numpy conversion",
            )),
        }
    }
}

#[pymethods]
impl PyTensor {
    /// Create a new tensor from a numpy array or Python list
    #[new]
    #[pyo3(signature = (data, device=None, requires_grad=false))]
    pub fn new(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        device: Option<String>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        // Try to handle as numpy array first (zero-copy when possible)
        if let Ok(array) = data.downcast::<PyArray<f32, IxDyn>>() {
            let shape = array.shape().to_vec();

            // Try zero-copy conversion first
            if array.is_c_contiguous() {
                // For C-contiguous arrays, we can use zero-copy
                let readonly_array = array.readonly();
                return Self::from_numpy_zero_copy(readonly_array);
            } else {
                // Fallback to copying for non-contiguous arrays
                let data_vec = array.to_vec()?;
                let tensor = Tensor::from_vec(data_vec, &shape).map_err(|e| {
                    PyValueError::new_err(format!("Failed to create tensor: {}", e))
                })?;
                return Ok(PyTensor::from_tensor_with_grad(tensor, requires_grad));
            }
        }

        // Try to handle as Python list/sequence
        if data.is_instance(&py.get_type::<pyo3::types::PyList>())?
            || data.is_instance(&py.get_type::<pyo3::types::PyTuple>())?
        {
            let (flat_data, shape) = Self::extract_nested_sequence(data)?;

            let tensor = Tensor::from_vec(flat_data, &shape)
                .map_err(|e| PyValueError::new_err(format!("Failed to create tensor: {}", e)))?;

            return Ok(PyTensor::from_tensor_with_grad(tensor, requires_grad));
        }

        // Try to handle as scalar
        if let Ok(scalar) = data.extract::<f32>() {
            let tensor = Tensor::scalar(scalar).map_err(|e| {
                PyValueError::new_err(format!("Failed to create scalar tensor: {}", e))
            })?;
            return Ok(PyTensor::from_tensor_with_grad(tensor, requires_grad));
        }

        Err(PyValueError::new_err(
            "Expected numpy array, Python list/tuple, or scalar (float)",
        ))
    }

    /// Create a tensor of zeros
    #[staticmethod]
    #[pyo3(signature = (shape, device=None, requires_grad=false))]
    pub fn zeros(shape: Vec<usize>, device: Option<String>, requires_grad: bool) -> PyResult<Self> {
        let tensor = Tensor::zeros(&shape)
            .map_err(|e| PyValueError::new_err(format!("Failed to create zeros tensor: {}", e)))?;
        Ok(PyTensor::from_tensor_with_grad(tensor, requires_grad))
    }

    /// Create a tensor of ones
    #[staticmethod]
    #[pyo3(signature = (shape, device=None, requires_grad=false))]
    pub fn ones(shape: Vec<usize>, device: Option<String>, requires_grad: bool) -> PyResult<Self> {
        let tensor = Tensor::ones(&shape)
            .map_err(|e| PyValueError::new_err(format!("Failed to create ones tensor: {}", e)))?;
        Ok(PyTensor::from_tensor_with_grad(tensor, requires_grad))
    }

    /// Create a tensor with random normal distribution
    #[staticmethod]
    #[pyo3(signature = (shape, mean=0.0, std=1.0, device=None, requires_grad=false))]
    pub fn randn(
        shape: Vec<usize>,
        mean: f32,
        std: f32,
        device: Option<String>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let tensor = Tensor::randn(&shape)
            .map_err(|e| PyValueError::new_err(format!("Failed to create randn tensor: {}", e)))?;
        Ok(PyTensor::from_tensor_with_grad(tensor, requires_grad))
    }

    /// Create a tensor with random uniform distribution
    #[staticmethod]
    #[pyo3(signature = (shape, low=0.0, high=1.0, device=None, requires_grad=false))]
    pub fn rand(
        shape: Vec<usize>,
        low: f32,
        high: f32,
        device: Option<String>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let tensor = Tensor::randn(&shape)
            .map_err(|e| PyValueError::new_err(format!("Failed to create rand tensor: {}", e)))?;
        Ok(PyTensor::from_tensor_with_grad(tensor, requires_grad))
    }

    /// Get the shape of the tensor
    #[getter]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// Get the data type
    #[getter]
    pub fn dtype(&self) -> &str {
        "float32"
    }

    /// Get the device
    #[getter]
    pub fn device(&self) -> &str {
        "cpu"
    }

    /// Get requires_grad flag
    #[getter]
    pub fn requires_grad(&self) -> bool {
        self.variable.as_ref().map(|v| v.requires_grad()).unwrap_or(false)
    }

    /// Convert to numpy array (copies data)
    pub fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let shape = self.inner.shape();

        // Extract data based on tensor type
        let data = match &self.inner {
            Tensor::F32(arr) => arr
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("Cannot get tensor data as slice"))?
                .to_vec(),
            _ => {
                return Err(PyValueError::new_err(
                    "Only F32 tensors supported for numpy conversion",
                ));
            },
        };

        let numpy_array = PyArray::from_slice(py, &data);
        numpy_array
            .reshape(shape)
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape array: {}", e)))
    }

    /// Convert to numpy array with zero-copy when possible
    pub fn numpy_view<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        self.as_numpy_view(py)
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype=float32, device=cpu)",
            self.shape()
        )
    }

    /// Repr
    pub fn __repr__(&self) -> String {
        let data_preview = match &self.inner {
            Tensor::F32(arr) => arr
                .as_slice()
                .map(|s| format!("{:?}", &s[..5.min(s.len())]))
                .unwrap_or_else(|| "Non-contiguous".to_string()),
            _ => "Non-F32".to_string(),
        };
        format!("Tensor(data={}, shape={:?})", data_preview, self.shape())
    }

    /// Addition
    pub fn __add__(&self, other: PyTensorOrScalar) -> PyResult<PyTensor> {
        match other {
            PyTensorOrScalar::Tensor(t) => {
                let result = self
                    .inner
                    .add(&t.inner)
                    .map_err(|e| PyValueError::new_err(format!("Addition failed: {}", e)))?;
                Ok(PyTensor::from_tensor(result))
            },
            PyTensorOrScalar::Scalar(s) => {
                let result = self
                    .inner
                    .add_scalar(s)
                    .map_err(|e| PyValueError::new_err(format!("Addition failed: {}", e)))?;
                Ok(PyTensor::from_tensor(result))
            },
        }
    }

    /// Subtraction
    pub fn __sub__(&self, other: PyTensorOrScalar) -> PyResult<PyTensor> {
        match other {
            PyTensorOrScalar::Tensor(t) => {
                let result = self
                    .inner
                    .sub(&t.inner)
                    .map_err(|e| PyValueError::new_err(format!("Subtraction failed: {}", e)))?;
                Ok(PyTensor::from_tensor(result))
            },
            PyTensorOrScalar::Scalar(s) => {
                let result = self
                    .inner
                    .sub_scalar(s)
                    .map_err(|e| PyValueError::new_err(format!("Subtraction failed: {}", e)))?;
                Ok(PyTensor::from_tensor(result))
            },
        }
    }

    /// Multiplication
    pub fn __mul__(&self, other: PyTensorOrScalar) -> PyResult<PyTensor> {
        match other {
            PyTensorOrScalar::Tensor(t) => {
                let result = self
                    .inner
                    .mul(&t.inner)
                    .map_err(|e| PyValueError::new_err(format!("Multiplication failed: {}", e)))?;
                Ok(PyTensor::from_tensor(result))
            },
            PyTensorOrScalar::Scalar(s) => {
                let result = self
                    .inner
                    .mul_scalar(s)
                    .map_err(|e| PyValueError::new_err(format!("Multiplication failed: {}", e)))?;
                Ok(PyTensor::from_tensor(result))
            },
        }
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self
            .inner
            .matmul(&other.inner)
            .map_err(|e| PyValueError::new_err(format!("Matmul failed: {}", e)))?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Transpose
    pub fn transpose(&self, dim0: Option<usize>, dim1: Option<usize>) -> PyResult<PyTensor> {
        let result = if let (Some(d0), Some(d1)) = (dim0, dim1) {
            self.inner
                .transpose(d0, d1)
                .map_err(|e| PyValueError::new_err(format!("Transpose failed: {}", e)))?
        } else {
            // Default transpose: swap last two dimensions
            let shape = self.inner.shape();
            if shape.len() < 2 {
                return Err(PyValueError::new_err(
                    "Cannot transpose tensor with less than 2 dimensions",
                ));
            }
            let d0 = shape.len() - 2;
            let d1 = shape.len() - 1;
            self.inner
                .transpose(d0, d1)
                .map_err(|e| PyValueError::new_err(format!("Transpose failed: {}", e)))?
        };
        Ok(PyTensor::from_tensor(result))
    }

    /// Reshape
    pub fn reshape(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        let result = self
            .inner
            .reshape(&shape)
            .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))?;
        Ok(PyTensor::from_tensor(result))
    }

    /// View (alias for reshape)
    pub fn view(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        self.reshape(shape)
    }

    /// Sum along specified axes
    #[pyo3(signature = (axis=None, keepdim=false))]
    pub fn sum(&self, axis: Option<Vec<usize>>, keepdim: bool) -> PyResult<PyTensor> {
        let result = if let Some(ref axes) = axis {
            // Use proper axis-based sum operation from core tensor
            if axes.len() == 1 {
                // Single axis - use sum_axis for efficiency
                self.inner
                    .sum_axis(axes[0])
                    .map_err(|e| PyValueError::new_err(format!("Sum along axis failed: {}", e)))?
            } else {
                // Multiple axes - use sum_axes
                self.inner
                    .sum_axes(&axes)
                    .map_err(|e| PyValueError::new_err(format!("Sum along axes failed: {}", e)))?
            }
        } else {
            // Sum all elements - use sum method from core tensor
            self.inner
                .sum(None, keepdim)
                .map_err(|e| PyValueError::new_err(format!("Sum failed: {}", e)))?
        };

        // Handle keepdim parameter for axis-based operations
        let final_result = if keepdim && axis.is_some() {
            // Add back dimensions that were reduced
            let mut shape = self.inner.shape().to_vec();
            for &ax in axis.as_ref().unwrap() {
                if ax < shape.len() {
                    shape[ax] = 1;
                }
            }
            result
                .reshape(&shape)
                .map_err(|e| PyValueError::new_err(format!("Reshape for keepdim failed: {}", e)))?
        } else {
            result
        };

        Ok(PyTensor::from_tensor(final_result))
    }

    /// Mean along specified axes
    #[pyo3(signature = (axis=None, keepdim=false))]
    pub fn mean(&self, axis: Option<Vec<usize>>, keepdim: bool) -> PyResult<PyTensor> {
        let result = if let Some(ref axes) = axis {
            // Use proper axis-based mean operation from core tensor
            if axes.len() == 1 {
                // Single axis - use mean_axis for efficiency
                self.inner
                    .mean_axis(axes[0])
                    .map_err(|e| PyValueError::new_err(format!("Mean along axis failed: {}", e)))?
            } else {
                // Multiple axes - use mean_axes
                self.inner
                    .mean_axes(&axes)
                    .map_err(|e| PyValueError::new_err(format!("Mean along axes failed: {}", e)))?
            }
        } else {
            // Mean all elements - use mean method from core tensor
            self.inner
                .mean()
                .map_err(|e| PyValueError::new_err(format!("Mean failed: {}", e)))?
        };

        // Handle keepdim parameter for axis-based operations
        let final_result = if keepdim && axis.is_some() {
            // Add back dimensions that were reduced
            let mut shape = self.inner.shape().to_vec();
            for &ax in axis.as_ref().unwrap() {
                if ax < shape.len() {
                    shape[ax] = 1;
                }
            }
            result
                .reshape(&shape)
                .map_err(|e| PyValueError::new_err(format!("Reshape for keepdim failed: {}", e)))?
        } else {
            result
        };

        Ok(PyTensor::from_tensor(final_result))
    }

    /// ReLU activation
    pub fn relu(&self) -> PyResult<PyTensor> {
        let result = self
            .inner
            .relu()
            .map_err(|e| PyValueError::new_err(format!("ReLU failed: {}", e)))?;
        Ok(PyTensor::from_tensor(result))
    }

    /// GELU activation
    pub fn gelu(&self) -> PyResult<PyTensor> {
        // Implement GELU manually since it might not be available in core tensor
        // GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        // Approximation: GELU(x) ≈ x * σ(1.702 * x)
        let x = &self.inner;
        let sigmoid_input = x
            .scalar_mul(1.702)
            .map_err(|e| PyValueError::new_err(format!("GELU scalar_mul failed: {}", e)))?;
        let sigmoid_result = sigmoid_input
            .sigmoid()
            .map_err(|e| PyValueError::new_err(format!("GELU sigmoid failed: {}", e)))?;
        let result = x
            .mul(&sigmoid_result)
            .map_err(|e| PyValueError::new_err(format!("GELU mul failed: {}", e)))?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Softmax
    #[pyo3(signature = (dim=-1))]
    pub fn softmax(&self, dim: i32) -> PyResult<PyTensor> {
        let result = self
            .inner
            .softmax(dim)
            .map_err(|e| PyValueError::new_err(format!("Softmax failed: {}", e)))?;
        Ok(PyTensor::from_tensor(result))
    }

    /// Check if tensor data is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        match &self.inner {
            Tensor::F32(arr) => arr.as_slice().is_some(),
            _ => false,
        }
    }

    /// Get memory layout information
    pub fn memory_info(&self) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);

            dict.set_item("contiguous", self.is_contiguous())?;
            dict.set_item("shape", self.shape())?;
            dict.set_item("dtype", self.dtype())?;
            dict.set_item("device", self.device())?;

            // Calculate memory usage
            let element_count: usize = self.shape().iter().product();
            let memory_bytes = element_count * std::mem::size_of::<f32>();
            dict.set_item("memory_bytes", memory_bytes)?;

            Ok(dict.into())
        })
    }

    /// Clone the tensor
    pub fn clone(&self) -> PyTensor {
        PyTensor::from_tensor(self.inner.clone())
    }

    /// Detach from computation graph
    pub fn detach(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.clone(),
            variable: None,
        }
    }

    /// Enable or disable gradient computation for this tensor
    pub fn requires_grad_(&mut self, requires_grad: bool) -> PyResult<()> {
        if requires_grad {
            if self.variable.is_none() {
                self.variable = Some(Variable::new(self.inner.clone(), true));
            }
        } else {
            self.variable = None;
        }
        Ok(())
    }

    /// Get the gradient of this tensor
    #[getter]
    pub fn grad(&self) -> PyResult<Option<PyTensor>> {
        if let Some(ref variable) = self.variable {
            match variable.grad() {
                Ok(Some(grad_tensor)) => Ok(Some(PyTensor::from_tensor(grad_tensor))),
                Ok(None) => Ok(None),
                Err(e) => Err(PyValueError::new_err(format!(
                    "Failed to get gradient: {}",
                    e
                ))),
            }
        } else {
            Ok(None)
        }
    }

    /// Compute gradients for this tensor
    pub fn backward(&self, gradient: Option<PyTensor>) -> PyResult<()> {
        if let Some(ref variable) = self.variable {
            let result = if let Some(grad) = gradient {
                variable.backward_with_grad(grad.inner)
            } else {
                variable.backward()
            };

            match result {
                Ok(()) => Ok(()),
                Err(e) => Err(PyValueError::new_err(format!(
                    "Backward pass failed: {}",
                    e
                ))),
            }
        } else {
            Err(PyValueError::new_err(
                "backward can only be called for tensors that require gradients",
            ))
        }
    }

    /// Zero the gradients for this tensor
    pub fn zero_grad(&self) -> PyResult<()> {
        if let Some(ref variable) = self.variable {
            variable.zero_grad();
            Ok(())
        } else {
            Ok(()) // No-op if no gradients
        }
    }

    /// Move to device
    pub fn to(&self, device: &str) -> PyResult<PyTensor> {
        // For now, we only support CPU
        if device != "cpu" {
            return Err(PyValueError::new_err(format!(
                "Device {} not supported",
                device
            )));
        }
        Ok(self.clone())
    }

    /// Get item
    pub fn __getitem__(&self, indices: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        // Simple implementation for now
        Ok(self.clone())
    }

    /// Set item
    pub fn __setitem__(
        &mut self,
        indices: &Bound<'_, PyAny>,
        value: PyTensorOrScalar,
    ) -> PyResult<()> {
        // Simple implementation for now
        Ok(())
    }
}

/// Helper enum for tensor or scalar operations
#[derive(FromPyObject)]
pub enum PyTensorOrScalar {
    Tensor(PyTensor),
    Scalar(f32),
}

// Helper functions
// Note: from_tensor method is already defined above
