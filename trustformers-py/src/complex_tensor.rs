use crate::errors::{TrustformersPyError, TrustformersPyResult};
use numpy::{PyArray, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scirs2_core::ndarray::{ArrayD, IxDyn};
use scirs2_core::Complex32; // SciRS2 Integration Policy

/// Complex tensor operations for trustformers-py
#[pyclass(name = "ComplexTensor")]
pub struct PyComplexTensor;

#[pymethods]
impl PyComplexTensor {
    #[new]
    pub fn new() -> Self {
        Self
    }

    /// Create complex tensor from real and imaginary parts
    #[staticmethod]
    pub fn from_real_imag(
        py: Python<'_>,
        real: &Bound<'_, PyArray<f32, IxDyn>>,
        imag: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<PyObject> {
        let real_array = real.try_readonly()?.as_array().to_owned();
        let imag_array = imag.try_readonly()?.as_array().to_owned();

        if real_array.shape() != imag_array.shape() {
            return Err(PyValueError::new_err(
                "Real and imaginary parts must have the same shape",
            ));
        }

        let complex_data: Vec<f32> = real_array
            .iter()
            .zip(imag_array.iter())
            .flat_map(|(&r, &i)| vec![r, i])
            .collect();

        let mut complex_shape = real_array.shape().to_vec();
        complex_shape.push(2); // Add dimension for [real, imag]

        let py_array = PyArray::from_vec(py, complex_data).reshape(complex_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Extract real part from complex tensor
    #[staticmethod]
    pub fn real(
        py: Python<'_>,
        complex_tensor: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<PyObject> {
        let array = complex_tensor.try_readonly()?.as_array().to_owned();

        if array.shape().is_empty() || array.shape()[array.ndim() - 1] != 2 {
            return Err(PyValueError::new_err(
                "Expected complex tensor with last dimension = 2",
            ));
        }

        let flat_data: Vec<f32> = array.iter().cloned().collect();
        let real_data: Vec<f32> = flat_data.chunks(2).map(|chunk| chunk[0]).collect();

        let real_shape = &array.shape()[..array.ndim() - 1];
        let py_array = PyArray::from_vec(py, real_data).reshape(real_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Extract imaginary part from complex tensor
    #[staticmethod]
    pub fn imag(
        py: Python<'_>,
        complex_tensor: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<PyObject> {
        let array = complex_tensor.try_readonly()?.as_array().to_owned();

        if array.shape().is_empty() || array.shape()[array.ndim() - 1] != 2 {
            return Err(PyValueError::new_err(
                "Expected complex tensor with last dimension = 2",
            ));
        }

        let flat_data: Vec<f32> = array.iter().cloned().collect();
        let imag_data: Vec<f32> = flat_data.chunks(2).map(|chunk| chunk[1]).collect();

        let imag_shape = &array.shape()[..array.ndim() - 1];
        let py_array = PyArray::from_vec(py, imag_data).reshape(imag_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Calculate magnitude (absolute value) of complex tensor
    #[staticmethod]
    pub fn magnitude(
        py: Python<'_>,
        complex_tensor: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<PyObject> {
        let array = complex_tensor.try_readonly()?.as_array().to_owned();

        if array.shape().is_empty() || array.shape()[array.ndim() - 1] != 2 {
            return Err(PyValueError::new_err(
                "Expected complex tensor with last dimension = 2",
            ));
        }

        let flat_data: Vec<f32> = array.iter().cloned().collect();
        let magnitude_data: Vec<f32> = flat_data
            .chunks(2)
            .map(|chunk| {
                let real = chunk[0];
                let imag = chunk[1];
                (real * real + imag * imag).sqrt()
            })
            .collect();

        let magnitude_shape = &array.shape()[..array.ndim() - 1];
        let py_array = PyArray::from_vec(py, magnitude_data).reshape(magnitude_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Calculate phase (angle) of complex tensor
    #[staticmethod]
    pub fn phase(
        py: Python<'_>,
        complex_tensor: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<PyObject> {
        let array = complex_tensor.try_readonly()?.as_array().to_owned();

        if array.shape().is_empty() || array.shape()[array.ndim() - 1] != 2 {
            return Err(PyValueError::new_err(
                "Expected complex tensor with last dimension = 2",
            ));
        }

        let flat_data: Vec<f32> = array.iter().cloned().collect();
        let phase_data: Vec<f32> = flat_data
            .chunks(2)
            .map(|chunk| {
                let real = chunk[0];
                let imag = chunk[1];
                imag.atan2(real)
            })
            .collect();

        let phase_shape = &array.shape()[..array.ndim() - 1];
        let py_array = PyArray::from_vec(py, phase_data).reshape(phase_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Complex conjugate
    #[staticmethod]
    pub fn conj(
        py: Python<'_>,
        complex_tensor: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<PyObject> {
        let array = complex_tensor.try_readonly()?.as_array().to_owned();

        if array.shape().is_empty() || array.shape()[array.ndim() - 1] != 2 {
            return Err(PyValueError::new_err(
                "Expected complex tensor with last dimension = 2",
            ));
        }

        let flat_data: Vec<f32> = array.iter().cloned().collect();
        let conj_data: Vec<f32> = flat_data
            .chunks(2)
            .flat_map(|chunk| vec![chunk[0], -chunk[1]]) // real, -imag
            .collect();

        let py_array = PyArray::from_vec(py, conj_data).reshape(array.shape())?;
        Ok(py_array.into_any().unbind())
    }

    /// Complex addition
    #[staticmethod]
    pub fn add(
        py: Python<'_>,
        tensor1: &Bound<'_, PyArray<f32, IxDyn>>,
        tensor2: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<PyObject> {
        let array1 = tensor1.try_readonly()?.as_array().to_owned();
        let array2 = tensor2.try_readonly()?.as_array().to_owned();

        if array1.shape() != array2.shape() {
            return Err(PyValueError::new_err("Tensors must have the same shape"));
        }

        if array1.shape().is_empty() || array1.shape()[array1.ndim() - 1] != 2 {
            return Err(PyValueError::new_err(
                "Expected complex tensors with last dimension = 2",
            ));
        }

        let flat_data1: Vec<f32> = array1.iter().cloned().collect();
        let flat_data2: Vec<f32> = array2.iter().cloned().collect();
        let result_data: Vec<f32> = flat_data1
            .chunks(2)
            .zip(flat_data2.chunks(2))
            .flat_map(|(chunk1, chunk2)| {
                vec![chunk1[0] + chunk2[0], chunk1[1] + chunk2[1]] // (r1+r2, i1+i2)
            })
            .collect();

        let py_array = PyArray::from_vec(py, result_data).reshape(array1.shape())?;
        Ok(py_array.into_any().unbind())
    }

    /// Complex multiplication
    #[staticmethod]
    pub fn multiply(
        py: Python<'_>,
        tensor1: &Bound<'_, PyArray<f32, IxDyn>>,
        tensor2: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<PyObject> {
        let array1 = tensor1.try_readonly()?.as_array().to_owned();
        let array2 = tensor2.try_readonly()?.as_array().to_owned();

        if array1.shape() != array2.shape() {
            return Err(PyValueError::new_err("Tensors must have the same shape"));
        }

        if array1.shape().is_empty() || array1.shape()[array1.ndim() - 1] != 2 {
            return Err(PyValueError::new_err(
                "Expected complex tensors with last dimension = 2",
            ));
        }

        let flat_data1: Vec<f32> = array1.iter().cloned().collect();
        let flat_data2: Vec<f32> = array2.iter().cloned().collect();
        let result_data: Vec<f32> = flat_data1
            .chunks(2)
            .zip(flat_data2.chunks(2))
            .flat_map(|(chunk1, chunk2)| {
                let r1 = chunk1[0];
                let i1 = chunk1[1];
                let r2 = chunk2[0];
                let i2 = chunk2[1];
                // (r1 + i1*i) * (r2 + i2*i) = (r1*r2 - i1*i2) + (r1*i2 + i1*r2)*i
                vec![r1 * r2 - i1 * i2, r1 * i2 + i1 * r2]
            })
            .collect();

        let py_array = PyArray::from_vec(py, result_data).reshape(array1.shape())?;
        Ok(py_array.into_any().unbind())
    }

    /// Complex division
    #[staticmethod]
    pub fn divide(
        py: Python<'_>,
        tensor1: &Bound<'_, PyArray<f32, IxDyn>>,
        tensor2: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<PyObject> {
        let array1 = tensor1.try_readonly()?.as_array().to_owned();
        let array2 = tensor2.try_readonly()?.as_array().to_owned();

        if array1.shape() != array2.shape() {
            return Err(PyValueError::new_err("Tensors must have the same shape"));
        }

        if array1.shape().is_empty() || array1.shape()[array1.ndim() - 1] != 2 {
            return Err(PyValueError::new_err(
                "Expected complex tensors with last dimension = 2",
            ));
        }

        let flat_data1: Vec<f32> = array1.iter().cloned().collect();
        let flat_data2: Vec<f32> = array2.iter().cloned().collect();
        let result_data: Vec<f32> = flat_data1
            .chunks(2)
            .zip(flat_data2.chunks(2))
            .flat_map(|(chunk1, chunk2)| {
                let r1 = chunk1[0];
                let i1 = chunk1[1];
                let r2 = chunk2[0];
                let i2 = chunk2[1];

                let denominator = r2 * r2 + i2 * i2;
                if denominator == 0.0 {
                    vec![f32::NAN, f32::NAN]
                } else {
                    // (r1 + i1*i) / (r2 + i2*i) = ((r1*r2 + i1*i2) + (i1*r2 - r1*i2)*i) / (r2^2 + i2^2)
                    vec![
                        (r1 * r2 + i1 * i2) / denominator,
                        (i1 * r2 - r1 * i2) / denominator,
                    ]
                }
            })
            .collect();

        let py_array = PyArray::from_vec(py, result_data).reshape(array1.shape())?;
        Ok(py_array.into_any().unbind())
    }

    /// Complex exponential
    #[staticmethod]
    pub fn exp(
        py: Python<'_>,
        complex_tensor: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<PyObject> {
        let array = complex_tensor.try_readonly()?.as_array().to_owned();

        if array.shape().is_empty() || array.shape()[array.ndim() - 1] != 2 {
            return Err(PyValueError::new_err(
                "Expected complex tensor with last dimension = 2",
            ));
        }

        let flat_data: Vec<f32> = array.iter().cloned().collect();
        let result_data: Vec<f32> = flat_data
            .chunks(2)
            .flat_map(|chunk| {
                let real = chunk[0];
                let imag = chunk[1];
                // e^(r + i*i) = e^r * (cos(i) + i*sin(i))
                let exp_real = real.exp();
                vec![exp_real * imag.cos(), exp_real * imag.sin()]
            })
            .collect();

        let py_array = PyArray::from_vec(py, result_data).reshape(array.shape())?;
        Ok(py_array.into_any().unbind())
    }

    /// Complex logarithm
    #[staticmethod]
    pub fn log(
        py: Python<'_>,
        complex_tensor: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<PyObject> {
        let array = complex_tensor.try_readonly()?.as_array().to_owned();

        if array.shape().is_empty() || array.shape()[array.ndim() - 1] != 2 {
            return Err(PyValueError::new_err(
                "Expected complex tensor with last dimension = 2",
            ));
        }

        let flat_data: Vec<f32> = array.iter().cloned().collect();
        let result_data: Vec<f32> = flat_data
            .chunks(2)
            .flat_map(|chunk| {
                let real = chunk[0];
                let imag = chunk[1];
                // log(r + i*i) = log(|z|) + i*arg(z)
                let magnitude = (real * real + imag * imag).sqrt();
                let phase = imag.atan2(real);
                vec![magnitude.ln(), phase]
            })
            .collect();

        let py_array = PyArray::from_vec(py, result_data).reshape(array.shape())?;
        Ok(py_array.into_any().unbind())
    }

    /// Complex power
    #[staticmethod]
    pub fn pow(
        py: Python<'_>,
        complex_tensor: &Bound<'_, PyArray<f32, IxDyn>>,
        exponent: f32,
    ) -> PyResult<PyObject> {
        let array = complex_tensor.try_readonly()?.as_array().to_owned();

        if array.shape().is_empty() || array.shape()[array.ndim() - 1] != 2 {
            return Err(PyValueError::new_err(
                "Expected complex tensor with last dimension = 2",
            ));
        }

        let flat_data: Vec<f32> = array.iter().cloned().collect();
        let result_data: Vec<f32> = flat_data
            .chunks(2)
            .flat_map(|chunk| {
                let real = chunk[0];
                let imag = chunk[1];

                if real == 0.0 && imag == 0.0 {
                    vec![0.0, 0.0]
                } else {
                    // z^n = |z|^n * e^(i*n*arg(z)) = |z|^n * (cos(n*arg(z)) + i*sin(n*arg(z)))
                    let magnitude = (real * real + imag * imag).sqrt();
                    let phase = imag.atan2(real);

                    let new_magnitude = magnitude.powf(exponent);
                    let new_phase = phase * exponent;

                    vec![
                        new_magnitude * new_phase.cos(),
                        new_magnitude * new_phase.sin(),
                    ]
                }
            })
            .collect();

        let py_array = PyArray::from_vec(py, result_data).reshape(array.shape())?;
        Ok(py_array.into_any().unbind())
    }

    /// Create complex tensor from polar representation (magnitude and phase)
    #[staticmethod]
    pub fn from_polar(
        py: Python<'_>,
        magnitude: &Bound<'_, PyArray<f32, IxDyn>>,
        phase: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<PyObject> {
        let mag_array = magnitude.try_readonly()?.as_array().to_owned();
        let phase_array = phase.try_readonly()?.as_array().to_owned();

        if mag_array.shape() != phase_array.shape() {
            return Err(PyValueError::new_err(
                "Magnitude and phase arrays must have the same shape",
            ));
        }

        let complex_data: Vec<f32> = mag_array
            .iter()
            .zip(phase_array.iter())
            .flat_map(|(&mag, &ph)| vec![mag * ph.cos(), mag * ph.sin()])
            .collect();

        let mut complex_shape = mag_array.shape().to_vec();
        complex_shape.push(2); // Add dimension for [real, imag]

        let py_array = PyArray::from_vec(py, complex_data).reshape(complex_shape)?;
        Ok(py_array.into_any().unbind())
    }

    /// Convert complex tensor to polar representation
    #[staticmethod]
    pub fn to_polar(
        py: Python<'_>,
        complex_tensor: &Bound<'_, PyArray<f32, IxDyn>>,
    ) -> PyResult<PyObject> {
        let magnitude = Self::magnitude(py, complex_tensor)?;
        let phase = Self::phase(py, complex_tensor)?;

        let result = PyDict::new(py);
        result.set_item("magnitude", magnitude)?;
        result.set_item("phase", phase)?;

        Ok(result.into_any().unbind())
    }
}
