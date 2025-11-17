//! Tensor constructor functions.
//!
//! This module contains functions for creating new tensors with various
//! initialization patterns.

use super::{DType, Tensor};
use crate::errors::{Result, TrustformersError};
use scirs2_core::ndarray::{ArrayD, IxDyn}; // SciRS2 Policy compliant
use scirs2_core::random::{thread_rng, Distribution, Normal};
use scirs2_core::{Complex, Complex32, Complex64}; // SciRS2 Policy compliant (root level)

impl Tensor {
    /// Creates a new 1D tensor from a vector of data.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of f32 values
    ///
    /// # Returns
    ///
    /// A 1D tensor containing the provided data.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0])?;
    /// assert_eq!(tensor.shape(), vec![4]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(data: Vec<f32>) -> Result<Self> {
        Ok(Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&[data.len()]), data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Creates a tensor from data with a specific shape.
    ///
    /// This is an alias for `from_vec` for backward compatibility with tests.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of f32 values
    /// * `shape` - The desired shape of the tensor
    ///
    /// # Returns
    ///
    /// A tensor with the specified shape containing the provided data.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tensor = Tensor::with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    /// assert_eq!(tensor.shape(), vec![2, 2]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_shape(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        Self::from_vec(data, &shape)
    }

    /// Creates a tensor from i64 data with a specific shape.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of i64 values
    /// * `shape` - The desired shape of the tensor
    ///
    /// # Returns
    ///
    /// A tensor with the specified shape containing the provided i64 data.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tensor = Tensor::from_vec_i64(vec![1, 2, 3, 4], &[2, 2])?;
    /// assert_eq!(tensor.shape(), vec![2, 2]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_vec_i64(data: Vec<i64>, shape: &[usize]) -> Result<Self> {
        if data.len() != shape.iter().product::<usize>() {
            return Err(TrustformersError::shape_error(
                "Data length doesn't match shape".into(),
            ));
        }
        Ok(Tensor::I64(
            ArrayD::from_shape_vec(IxDyn(shape), data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Creates a tensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The desired shape of the tensor
    ///
    /// # Returns
    ///
    /// A tensor of the specified shape filled with zeros.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tensor = Tensor::zeros(&[2, 3])?;
    /// assert_eq!(tensor.shape(), vec![2, 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn zeros(shape: &[usize]) -> Result<Self> {
        Ok(Tensor::F32(ArrayD::zeros(IxDyn(shape))))
    }

    /// Creates a tensor filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The desired shape of the tensor
    ///
    /// # Returns
    ///
    /// A tensor of the specified shape filled with ones.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tensor = Tensor::ones(&[2, 3])?;
    /// assert_eq!(tensor.shape(), vec![2, 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn ones(shape: &[usize]) -> Result<Self> {
        Ok(Tensor::F32(ArrayD::ones(IxDyn(shape))))
    }

    /// Creates a tensor filled with random values from a normal distribution.
    ///
    /// # Arguments
    ///
    /// * `shape` - The desired shape of the tensor
    ///
    /// # Returns
    ///
    /// A tensor filled with random values from N(0, 1).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tensor = Tensor::randn(&[2, 3])?;
    /// assert_eq!(tensor.shape(), vec![2, 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn randn(shape: &[usize]) -> Result<Self> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = thread_rng();
        let size = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng)).collect();
        Ok(Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(shape), data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Creates a tensor filled with zeros with the same shape as the input tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to match shape from
    ///
    /// # Returns
    ///
    /// A tensor of the same shape filled with zeros.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let input = Tensor::randn(&[2, 3])?;
    /// let zeros = Tensor::zeros_like(&input)?;
    /// assert_eq!(zeros.shape(), vec![2, 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn zeros_like(tensor: &Tensor) -> Result<Self> {
        Self::zeros(&tensor.shape())
    }

    /// Creates a tensor filled with ones with the same shape as the input tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to match shape from
    ///
    /// # Returns
    ///
    /// A tensor of the same shape filled with ones.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let input = Tensor::randn(&[2, 3])?;
    /// let ones = Tensor::ones_like(&input)?;
    /// assert_eq!(ones.shape(), vec![2, 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn ones_like(tensor: &Tensor) -> Result<Self> {
        Self::ones(&tensor.shape())
    }

    /// Creates a tensor from data with specified shape.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of f32 values
    /// * `shape` - The desired shape of the tensor
    ///
    /// # Returns
    ///
    /// A tensor containing the provided data reshaped to the specified shape.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor = Tensor::from_data(data, &[2, 3])?;
    /// assert_eq!(tensor.shape(), vec![2, 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_data(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(TrustformersError::shape_error(format!(
                "Data length {} does not match expected size {} for shape {:?}",
                data.len(),
                expected_size,
                shape
            )));
        }
        Ok(Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(shape), data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Creates a tensor from a slice with specified shape.
    ///
    /// # Arguments
    ///
    /// * `data` - A slice of f32 values
    /// * `shape` - The desired shape of the tensor
    ///
    /// # Returns
    ///
    /// A tensor containing the provided data reshaped to the specified shape.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let tensor = Tensor::from_slice(&data, &[2, 3])?;
    /// assert_eq!(tensor.shape(), vec![2, 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(TrustformersError::shape_error(format!(
                "Data length {} does not match expected size {} for shape {:?}",
                data.len(),
                expected_size,
                shape
            )));
        }
        Ok(Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(shape), data.to_vec())
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Creates a tensor filled with random values with the same shape as the input tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to match shape from
    ///
    /// # Returns
    ///
    /// A tensor of the same shape filled with random values from N(0, 1).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let input = Tensor::zeros(&[2, 3])?;
    /// let random = Tensor::randn_like(&input)?;
    /// assert_eq!(random.shape(), vec![2, 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn randn_like(tensor: &Tensor) -> Result<Self> {
        Self::randn(&tensor.shape())
    }

    /// Creates a tensor filled with zeros (f64 precision).
    pub fn zeros_f64(shape: &[usize]) -> Result<Self> {
        Ok(Tensor::F64(ArrayD::zeros(IxDyn(shape))))
    }

    /// Creates a tensor filled with zeros (i64 integers).
    pub fn zeros_i64(shape: &[usize]) -> Result<Self> {
        Ok(Tensor::I64(ArrayD::zeros(IxDyn(shape))))
    }

    /// Creates a tensor filled with zeros (complex f32).
    pub fn zeros_c32(shape: &[usize]) -> Result<Self> {
        Ok(Tensor::C32(ArrayD::zeros(IxDyn(shape))))
    }

    /// Creates a tensor filled with zeros (complex f64).
    pub fn zeros_c64(shape: &[usize]) -> Result<Self> {
        Ok(Tensor::C64(ArrayD::zeros(IxDyn(shape))))
    }

    /// Creates a tensor filled with zeros (f16 precision).
    pub fn zeros_f16(shape: &[usize]) -> Result<Self> {
        let total_size: usize = shape.iter().product();
        let data = vec![half::f16::ZERO; total_size];
        Ok(Tensor::F16(ArrayD::from_shape_vec(IxDyn(shape), data)?))
    }

    /// Creates a tensor filled with zeros (bf16 precision).
    pub fn zeros_bf16(shape: &[usize]) -> Result<Self> {
        let total_size: usize = shape.iter().product();
        let data = vec![half::bf16::ZERO; total_size];
        Ok(Tensor::BF16(ArrayD::from_shape_vec(IxDyn(shape), data)?))
    }

    /// Creates a tensor filled with zeros (complex f16).
    pub fn zeros_cf16(shape: &[usize]) -> Result<Self> {
        let total_size: usize = shape.iter().product();
        let data = vec![Complex::new(half::f16::ZERO, half::f16::ZERO); total_size];
        Ok(Tensor::CF16(ArrayD::from_shape_vec(IxDyn(shape), data)?))
    }

    /// Creates a tensor filled with zeros (complex bf16).
    pub fn zeros_cbf16(shape: &[usize]) -> Result<Self> {
        let total_size: usize = shape.iter().product();
        let data = vec![Complex::new(half::bf16::ZERO, half::bf16::ZERO); total_size];
        Ok(Tensor::CBF16(ArrayD::from_shape_vec(IxDyn(shape), data)?))
    }

    /// Creates a complex tensor from real and imaginary parts.
    ///
    /// # Arguments
    ///
    /// * `real` - Real part values
    /// * `imag` - Imaginary part values
    /// * `shape` - The desired shape
    ///
    /// # Returns
    ///
    /// A complex tensor with the specified real and imaginary parts.
    pub fn complex(real: Vec<f32>, imag: Vec<f32>, shape: &[usize]) -> Result<Self> {
        if real.len() != imag.len() {
            return Err(TrustformersError::shape_error(
                "Real and imaginary parts must have the same length".into(),
            ));
        }

        let complex_data: Vec<Complex32> =
            real.into_iter().zip(imag).map(|(r, i)| Complex32::new(r, i)).collect();

        Ok(Tensor::C32(
            ArrayD::from_shape_vec(IxDyn(shape), complex_data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Creates a complex tensor from real and imaginary parts (f64 precision).
    pub fn complex_f64(real: Vec<f64>, imag: Vec<f64>, shape: &[usize]) -> Result<Self> {
        if real.len() != imag.len() {
            return Err(TrustformersError::shape_error(
                "Real and imaginary parts must have the same length".into(),
            ));
        }

        let complex_data: Vec<Complex64> =
            real.into_iter().zip(imag).map(|(r, i)| Complex64::new(r, i)).collect();

        Ok(Tensor::C64(
            ArrayD::from_shape_vec(IxDyn(shape), complex_data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Creates a tensor from a vector with explicit shape.
    ///
    /// # Arguments
    ///
    /// * `data` - The data vector
    /// * `shape` - The desired shape
    ///
    /// # Returns
    ///
    /// A tensor with the specified data and shape.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        if data.len() != shape.iter().product::<usize>() {
            return Err(TrustformersError::shape_error(
                "Data length doesn't match shape".into(),
            ));
        }
        Ok(Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(shape), data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Creates a tensor from a Vec with explicit dtype.
    /// TEMPORARY: Uses ndarray. Will be replaced with SciRS2-Core.
    ///
    /// # Arguments
    ///
    /// * `data` - The data as a Vec<f64>
    /// * `shape` - The desired shape
    /// * `dtype` - The desired data type
    pub fn from_vec_with_dtype(data: Vec<f64>, shape: &[usize], dtype: DType) -> Result<Self> {
        if data.len() != shape.iter().product::<usize>() {
            return Err(TrustformersError::shape_error(
                "Data length doesn't match shape".into(),
            ));
        }

        match dtype {
            DType::F32 => {
                let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                Ok(Tensor::F32(
                    ArrayD::from_shape_vec(IxDyn(shape), data_f32)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
                ))
            },
            DType::F64 => Ok(Tensor::F64(
                ArrayD::from_shape_vec(IxDyn(shape), data)
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            )),
            DType::I64 => {
                let data_i64: Vec<i64> = data.iter().map(|&x| x as i64).collect();
                Ok(Tensor::I64(
                    ArrayD::from_shape_vec(IxDyn(shape), data_i64)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
                ))
            },
            _ => Err(TrustformersError::tensor_op_error(
                &format!("Unsupported dtype {:?} for from_vec_with_dtype", dtype),
                "from_vec_with_dtype",
            )),
        }
    }

    /// Creates a tensor filled with a constant value.
    ///
    /// # Arguments
    ///
    /// * `value` - The constant value to fill with
    /// * `shape` - The desired shape
    ///
    /// # Returns
    ///
    /// A tensor filled with the constant value.
    pub fn full(value: f32, shape: Vec<usize>) -> Result<Self> {
        let arr = ArrayD::from_elem(IxDyn(&shape), value);
        Ok(Tensor::F32(arr))
    }

    /// Creates a tensor filled with a constant value with specified dtype.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the tensor as a slice
    /// * `value` - The fill value (will be cast to target dtype)
    /// * `dtype` - Target data type
    ///
    /// # Returns
    ///
    /// A tensor filled with the constant value.
    ///
    /// # Note
    ///
    /// TEMPORARY: Uses ndarray - will be replaced with SciRS2-Core in future migration
    pub fn full_with_dtype(shape: &[usize], value: f64, dtype: DType) -> Result<Self> {
        match dtype {
            DType::F32 => {
                let arr = ArrayD::from_elem(IxDyn(shape), value as f32);
                Ok(Tensor::F32(arr))
            },
            DType::F64 => {
                let arr = ArrayD::from_elem(IxDyn(shape), value);
                Ok(Tensor::F64(arr))
            },
            DType::I64 => {
                let arr = ArrayD::from_elem(IxDyn(shape), value as i64);
                Ok(Tensor::I64(arr))
            },
            _ => Err(TrustformersError::tensor_op_error(
                &format!("Unsupported dtype {:?} for full_with_dtype", dtype),
                "full_with_dtype",
            )),
        }
    }

    /// Creates a scalar tensor.
    ///
    /// # Arguments
    ///
    /// * `value` - The scalar value
    ///
    /// # Returns
    ///
    /// A 0-dimensional tensor containing the scalar value.
    pub fn scalar(value: f32) -> Result<Self> {
        Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), value)))
    }

    /// Creates an identity matrix tensor.
    ///
    /// # Arguments
    ///
    /// * `n` - The size of the identity matrix (n x n)
    ///
    /// # Returns
    ///
    /// An n x n identity matrix tensor.
    pub fn eye_f32(n: usize) -> Result<Self> {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Ok(Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&[n, n]), data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Creates a tensor filled with ones (f16 precision).
    pub fn ones_f16(shape: &[usize]) -> Result<Self> {
        let total_size: usize = shape.iter().product();
        let data = vec![half::f16::ONE; total_size];
        Ok(Tensor::F16(ArrayD::from_shape_vec(IxDyn(shape), data)?))
    }

    /// Creates a tensor filled with ones (bf16 precision).
    pub fn ones_bf16(shape: &[usize]) -> Result<Self> {
        let total_size: usize = shape.iter().product();
        let data = vec![half::bf16::ONE; total_size];
        Ok(Tensor::BF16(ArrayD::from_shape_vec(IxDyn(shape), data)?))
    }

    /// Creates a tensor filled with random values from a normal distribution (f16 precision).
    pub fn randn_f16(shape: &[usize]) -> Result<Self> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = thread_rng();
        let size = shape.iter().product();
        let data: Vec<half::f16> =
            (0..size).map(|_| half::f16::from_f32(normal.sample(&mut rng))).collect();
        Ok(Tensor::F16(
            ArrayD::from_shape_vec(IxDyn(shape), data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Creates a tensor filled with random values from a normal distribution (bf16 precision).
    pub fn randn_bf16(shape: &[usize]) -> Result<Self> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = thread_rng();
        let size = shape.iter().product();
        let data: Vec<half::bf16> =
            (0..size).map(|_| half::bf16::from_f32(normal.sample(&mut rng))).collect();
        Ok(Tensor::BF16(
            ArrayD::from_shape_vec(IxDyn(shape), data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Creates a complex tensor from real and imaginary parts (f16 precision).
    pub fn complex_f16(real: Vec<f32>, imag: Vec<f32>, shape: &[usize]) -> Result<Self> {
        if real.len() != imag.len() {
            return Err(TrustformersError::shape_error(
                "Real and imaginary parts must have the same length".into(),
            ));
        }

        let complex_data: Vec<Complex<half::f16>> = real
            .into_iter()
            .zip(imag)
            .map(|(r, i)| Complex::new(half::f16::from_f32(r), half::f16::from_f32(i)))
            .collect();

        Ok(Tensor::CF16(
            ArrayD::from_shape_vec(IxDyn(shape), complex_data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Creates a complex tensor from real and imaginary parts (bf16 precision).
    pub fn complex_bf16(real: Vec<f32>, imag: Vec<f32>, shape: &[usize]) -> Result<Self> {
        if real.len() != imag.len() {
            return Err(TrustformersError::shape_error(
                "Real and imaginary parts must have the same length".into(),
            ));
        }

        let complex_data: Vec<Complex<half::bf16>> = real
            .into_iter()
            .zip(imag)
            .map(|(r, i)| {
                Complex::new(half::bf16::from_f32(r), half::bf16::from_f32(i))
            })
            .collect();

        Ok(Tensor::CBF16(
            ArrayD::from_shape_vec(IxDyn(shape), complex_data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Creates a tensor with specified data type.
    ///
    /// # Arguments
    ///
    /// * `dtype` - The desired data type
    /// * `shape` - The desired shape
    ///
    /// # Returns
    ///
    /// A zero tensor with the specified data type and shape.
    pub fn zeros_dtype(dtype: DType, shape: &[usize]) -> Result<Self> {
        match dtype {
            DType::F32 => Self::zeros(shape),
            DType::F64 => Self::zeros_f64(shape),
            DType::F16 => Self::zeros_f16(shape),
            DType::BF16 => Self::zeros_bf16(shape),
            DType::I64 => Self::zeros_i64(shape),
            DType::C32 => Self::zeros_c32(shape),
            DType::C64 => Self::zeros_c64(shape),
            DType::CF16 => Self::zeros_cf16(shape),
            DType::CBF16 => Self::zeros_cbf16(shape),
            _ => Err(TrustformersError::tensor_op_error(
                &format!("Unsupported data type for zeros: {:?}", dtype),
                "Tensor::zeros_dtype",
            )),
        }
    }

    /// Creates a tensor filled with ones with explicit dtype.
    ///
    /// # Arguments
    ///
    /// * `dtype` - The data type for the tensor
    /// * `shape` - The shape of the tensor
    ///
    /// # Returns
    ///
    /// A tensor filled with ones of the specified data type.
    pub fn ones_dtype(dtype: DType, shape: &[usize]) -> Result<Self> {
        match dtype {
            DType::F32 => Self::ones(shape),
            DType::F64 => Ok(Tensor::F64(ArrayD::ones(IxDyn(shape)))),
            DType::F16 => Self::ones_f16(shape),
            DType::BF16 => Self::ones_bf16(shape),
            DType::I64 => Ok(Tensor::I64(ArrayD::ones(IxDyn(shape)))),
            _ => Err(TrustformersError::tensor_op_error(
                &format!("Unsupported data type for ones: {:?}", dtype),
                "Tensor::ones_dtype",
            )),
        }
    }

    /// Creates a tensor filled with a scalar value (alternative signature).
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor
    /// * `value` - The value to fill the tensor with
    ///
    /// # Returns
    ///
    /// A tensor filled with the specified value.
    pub fn full_with_shape(shape: &[usize], value: f32) -> Result<Self> {
        let array = ArrayD::from_elem(IxDyn(shape), value);
        Ok(Tensor::F32(array))
    }

    /// Creates a tensor from a slice of f64 values with specified shape.
    ///
    /// # Arguments
    ///
    /// * `data` - A slice of f64 values
    /// * `shape` - The desired shape of the tensor
    ///
    /// # Returns
    ///
    /// A tensor containing the provided data reshaped to the specified shape.
    pub fn from_slice_f64(data: &[f64], shape: &[usize]) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(TrustformersError::shape_error(format!(
                "Data length {} does not match expected size {} for shape {:?}",
                data.len(),
                expected_size,
                shape
            )));
        }
        Ok(Tensor::F64(
            ArrayD::from_shape_vec(IxDyn(shape), data.to_vec())
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Creates a tensor from a slice of i64 values with specified shape.
    ///
    /// # Arguments
    ///
    /// * `data` - A slice of i64 values
    /// * `shape` - The desired shape of the tensor
    ///
    /// # Returns
    ///
    /// A tensor containing the provided data reshaped to the specified shape.
    pub fn from_slice_i64(data: &[i64], shape: &[usize]) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(TrustformersError::shape_error(format!(
                "Data length {} does not match expected size {} for shape {:?}",
                data.len(),
                expected_size,
                shape
            )));
        }
        Ok(Tensor::I64(
            ArrayD::from_shape_vec(IxDyn(shape), data.to_vec())
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Creates a tensor from a slice of i32 values with specified shape.
    ///
    /// # Arguments
    ///
    /// * `data` - A slice of i32 values
    /// * `shape` - The desired shape of the tensor
    ///
    /// # Returns
    ///
    /// A tensor containing the provided data reshaped to the specified shape.
    pub fn from_slice_i32(data: &[i32], shape: &[usize]) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(TrustformersError::shape_error(format!(
                "Data length {} does not match expected size {} for shape {:?}",
                data.len(),
                expected_size,
                shape
            )));
        }
        // Create i64 tensor since there's no I32 variant in the Tensor enum
        let i64_data: Vec<i64> = data.iter().map(|&x| x as i64).collect();
        Ok(Tensor::I64(
            ArrayD::from_shape_vec(IxDyn(shape), i64_data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ))
    }

    /// Creates a tensor from a scalar value.
    ///
    /// # Arguments
    ///
    /// * `value` - The scalar value
    /// * `dtype` - The data type for the tensor
    ///
    /// # Returns
    ///
    /// A 0-dimensional (scalar) tensor.
    pub fn from_scalar(value: f32, dtype: DType) -> Result<Self> {
        match dtype {
            DType::F32 => Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), value))),
            DType::F64 => Ok(Tensor::F64(ArrayD::from_elem(IxDyn(&[]), value as f64))),
            DType::I64 => Ok(Tensor::I64(ArrayD::from_elem(IxDyn(&[]), value as i64))),
            _ => Err(TrustformersError::tensor_op_error(
                &format!("Unsupported dtype {:?} for from_scalar", dtype),
                "from_scalar",
            )),
        }
    }

    /// Creates a tensor with a range of values.
    ///
    /// # Arguments
    ///
    /// * `start` - Start value (inclusive)
    /// * `end` - End value (exclusive)
    /// * `dtype` - The data type for the tensor
    ///
    /// # Returns
    ///
    /// A 1D tensor with values from start to end-1.
    pub fn range(start: i64, end: i64, dtype: DType) -> Result<Self> {
        if start >= end {
            return Err(TrustformersError::tensor_op_error(
                "Start must be less than end for range",
                "range",
            ));
        }
        match dtype {
            DType::I64 => {
                let data: Vec<i64> = (start..end).collect();
                Ok(Tensor::I64(
                    ArrayD::from_shape_vec(IxDyn(&[data.len()]), data)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
                ))
            },
            DType::F32 => {
                let data: Vec<f32> = (start..end).map(|x| x as f32).collect();
                Ok(Tensor::F32(
                    ArrayD::from_shape_vec(IxDyn(&[data.len()]), data)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
                ))
            },
            DType::F64 => {
                let data: Vec<f64> = (start..end).map(|x| x as f64).collect();
                Ok(Tensor::F64(
                    ArrayD::from_shape_vec(IxDyn(&[data.len()]), data)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
                ))
            },
            _ => Err(TrustformersError::tensor_op_error(
                &format!("Unsupported dtype {:?} for range", dtype),
                "range",
            )),
        }
    }

    /// Creates a tensor filled with random integers in the range [low, high).
    ///
    /// # Arguments
    ///
    /// * `low` - Lower bound (inclusive)
    /// * `high` - Upper bound (exclusive)
    /// * `shape` - Shape of the tensor
    /// * `dtype` - Data type of the tensor
    ///
    /// # Returns
    ///
    /// A tensor filled with random integers.
    pub fn randint(low: i64, high: i64, shape: &[usize], dtype: DType) -> Result<Self> {
        use scirs2_core::random::{rng, Rng}; // SciRS2 Policy compliant (updated API)

        if low >= high {
            return Err(TrustformersError::tensor_op_error(
                "low must be less than high for randint",
                "randint",
            ));
        }

        let mut rng = rng(); // Use updated scirs2-core API
        let size: usize = shape.iter().product();

        match dtype {
            DType::I64 => {
                let data: Vec<i64> = (0..size).map(|_| rng.random_range(low..high)).collect(); // Updated API
                Ok(Tensor::I64(
                    ArrayD::from_shape_vec(IxDyn(shape), data)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
                ))
            },
            DType::F32 => {
                let data: Vec<f32> =
                    (0..size).map(|_| rng.random_range(low..high) as f32).collect(); // Updated API
                Ok(Tensor::F32(
                    ArrayD::from_shape_vec(IxDyn(shape), data)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
                ))
            },
            DType::F64 => {
                let data: Vec<f64> =
                    (0..size).map(|_| rng.random_range(low..high) as f64).collect(); // Updated API
                Ok(Tensor::F64(
                    ArrayD::from_shape_vec(IxDyn(shape), data)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
                ))
            },
            _ => Err(TrustformersError::tensor_op_error(
                &format!("Unsupported dtype {:?} for randint", dtype),
                "randint",
            )),
        }
    }
}
