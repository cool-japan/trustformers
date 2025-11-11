//! Tensor conversion functions.
//!
//! This module contains functions for converting between different tensor types
//! and data formats.

use super::{DType, Tensor};
use crate::errors::{Result, TrustformersError};
use scirs2_core::{Complex32, Complex64};

impl Tensor {
    /// Convert tensor to a different data type.
    ///
    /// # Arguments
    ///
    /// * `dtype` - Target data type
    ///
    /// # Returns
    ///
    /// A tensor with the new data type.
    pub fn to_dtype(&self, dtype: DType) -> Result<Tensor> {
        match (self, dtype) {
            (Tensor::F32(a), DType::F64) => {
                let result = a.mapv(|x| x as f64);
                Ok(Tensor::F64(result))
            },
            (Tensor::F32(a), DType::I64) => {
                let result = a.mapv(|x| x as i64);
                Ok(Tensor::I64(result))
            },
            (Tensor::F32(a), DType::C32) => {
                let result = a.mapv(|x| Complex32::new(x, 0.0));
                Ok(Tensor::C32(result))
            },
            (Tensor::F32(a), DType::C64) => {
                let result = a.mapv(|x| Complex64::new(x as f64, 0.0));
                Ok(Tensor::C64(result))
            },
            (Tensor::F64(a), DType::F32) => {
                let result = a.mapv(|x| x as f32);
                Ok(Tensor::F32(result))
            },
            (Tensor::F64(a), DType::I64) => {
                let result = a.mapv(|x| x as i64);
                Ok(Tensor::I64(result))
            },
            (Tensor::F64(a), DType::C32) => {
                let result = a.mapv(|x| Complex32::new(x as f32, 0.0));
                Ok(Tensor::C32(result))
            },
            (Tensor::F64(a), DType::C64) => {
                let result = a.mapv(|x| Complex64::new(x, 0.0));
                Ok(Tensor::C64(result))
            },
            (Tensor::I64(a), DType::F32) => {
                let result = a.mapv(|x| x as f32);
                Ok(Tensor::F32(result))
            },
            (Tensor::I64(a), DType::F64) => {
                let result = a.mapv(|x| x as f64);
                Ok(Tensor::F64(result))
            },
            (Tensor::C32(a), DType::F32) => {
                let result = a.mapv(|x| x.re);
                Ok(Tensor::F32(result))
            },
            (Tensor::C32(a), DType::F64) => {
                let result = a.mapv(|x| x.re as f64);
                Ok(Tensor::F64(result))
            },
            (Tensor::C64(a), DType::F32) => {
                let result = a.mapv(|x| x.re as f32);
                Ok(Tensor::F32(result))
            },
            (Tensor::C64(a), DType::F64) => {
                let result = a.mapv(|x| x.re);
                Ok(Tensor::F64(result))
            },
            (tensor, target_dtype) if tensor.dtype() == target_dtype => Ok(tensor.clone()),
            #[cfg(feature = "metal")]
            (Tensor::Metal(_), _) => {
                // Convert Metal tensor to CPU first, then apply dtype conversion
                let cpu_tensor = self.to_device_enum(&crate::device::Device::CPU)?;
                cpu_tensor.to_dtype(dtype)
            },
            _ => Err(TrustformersError::tensor_op_error(
                &format!(
                    "Conversion from {:?} to {:?} not supported",
                    self.dtype(),
                    dtype
                ),
                "Tensor::to_dtype",
            )),
        }
    }

    /// Convert tensor to vector of f32 values.
    ///
    /// # Returns
    ///
    /// A vector of f32 values.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        match self {
            Tensor::F32(a) => Ok(a.iter().cloned().collect()),
            Tensor::F64(a) => Ok(a.iter().map(|&x| x as f32).collect()),
            Tensor::I64(a) => Ok(a.iter().map(|&x| x as f32).collect()),
            Tensor::C32(a) => Ok(a.iter().map(|x| x.re).collect()),
            Tensor::C64(a) => Ok(a.iter().map(|x| x.re as f32).collect()),
            #[cfg(feature = "metal")]
            Tensor::Metal(_) => {
                // Convert Metal tensor to CPU first, then get vec
                let cpu_tensor = self.to_device_enum(&crate::device::Device::CPU)?;
                cpu_tensor.to_vec_f32()
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Cannot convert this tensor type to Vec<f32>",
                "Tensor::to_vec_f32",
            )),
        }
    }

    /// Convert tensor to vector of u8 values.
    ///
    /// # Returns
    ///
    /// A vector of u8 values.
    pub fn to_vec_u8(&self) -> Result<Vec<u8>> {
        match self {
            Tensor::F32(a) => Ok(a.iter().map(|&x| x as u8).collect()),
            Tensor::F64(a) => Ok(a.iter().map(|&x| x as u8).collect()),
            Tensor::I64(a) => Ok(a.iter().map(|&x| x as u8).collect()),
            #[cfg(feature = "metal")]
            Tensor::Metal(_) => {
                // Convert Metal tensor to CPU first, then get vec
                let cpu_tensor = self.to_device_enum(&crate::device::Device::CPU)?;
                cpu_tensor.to_vec_u8()
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Cannot convert this tensor type to Vec<u8>",
                "Tensor::to_vec_u8",
            )),
        }
    }

    /// Convert tensor to F32 dtype (convenience method).
    ///
    /// # Returns
    ///
    /// A tensor with F32 dtype.
    pub fn to_f32(&self) -> Result<Tensor> {
        self.to_dtype(DType::F32)
    }

    /// Convert tensor to I64 dtype (convenience method).
    ///
    /// # Returns
    ///
    /// A tensor with I64 dtype.
    pub fn to_i64(&self) -> Result<Tensor> {
        self.to_dtype(DType::I64)
    }
}
