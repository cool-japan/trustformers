//! Arithmetic operations for tensors.
//!
//! This module contains basic arithmetic operations including addition, subtraction,
//! multiplication, division, and scalar operations with numerical stability features.
//! All operations include broadcasting support and numerical stability enhancements.

use super::super::Tensor;
use crate::errors::{Result, TrustformersError};
use ndarray::ArrayD;

// Import stability functions from the stability module
use super::stability::{is_stable_f32, is_stable_f64, stabilize_f32, stabilize_f64};

// Import broadcasting function from the broadcasting module
use super::broadcasting::shapes_are_broadcastable;

impl Tensor {
    /// Element-wise addition with numerical stability enhancements.
    ///
    /// Includes overflow/underflow protection and NaN/infinity detection.
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        match (self, other) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                // Check if shapes are broadcastable before attempting addition
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }

                // Always use ndarray's broadcasting addition (handles all shapes correctly)
                // Then stabilize the result if needed
                let result = a + b;

                // Check if stabilization is needed
                let has_unstable = result.iter().any(|&x| !is_stable_f32(x));

                if has_unstable {
                    // Stabilize the result
                    let stabilized: Vec<f32> = result.iter().map(|&x| stabilize_f32(x)).collect();

                    let result_array = ArrayD::from_shape_vec(result.raw_dim(), stabilized)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    Ok(Tensor::F32(result_array))
                } else {
                    Ok(Tensor::F32(result))
                }
            },
            (Tensor::F64(a), Tensor::F64(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }

                // Always use ndarray's broadcasting addition (handles all shapes correctly)
                // Then stabilize the result if needed
                let result = a + b;

                // Check if stabilization is needed
                let has_unstable = result.iter().any(|&x| !is_stable_f64(x));

                if has_unstable {
                    // Stabilize the result
                    let stabilized: Vec<f64> = result.iter().map(|&x| stabilize_f64(x)).collect();

                    let result_array = ArrayD::from_shape_vec(result.raw_dim(), stabilized)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    Ok(Tensor::F64(result_array))
                } else {
                    Ok(Tensor::F64(result))
                }
            },
            (Tensor::I64(a), Tensor::I64(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                let result = a + b;
                Ok(Tensor::I64(result))
            },
            (Tensor::C32(a), Tensor::C32(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                let result = a + b;
                Ok(Tensor::C32(result))
            },
            (Tensor::C64(a), Tensor::C64(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                let result = a + b;
                Ok(Tensor::C64(result))
            },
            #[cfg(feature = "metal")]
            (Tensor::Metal(_), _) => {
                // Convert Metal tensor to CPU, then perform operation
                let cpu_self = self.to_device_enum(&crate::device::Device::CPU)?;
                cpu_self.add(other)
            },
            #[cfg(feature = "metal")]
            (_, Tensor::Metal(_)) => {
                // Convert Metal tensor to CPU, then perform operation
                let cpu_other = other.to_device_enum(&crate::device::Device::CPU)?;
                self.add(&cpu_other)
            },
            #[cfg(feature = "cuda")]
            (Tensor::CUDA(_), _) => {
                // Convert CUDA tensor to CPU, then perform operation
                let cpu_self = self.to_device_enum(&crate::device::Device::CPU)?;
                cpu_self.add(other)
            },
            #[cfg(feature = "cuda")]
            (_, Tensor::CUDA(_)) => {
                // Convert CUDA tensor to CPU, then perform operation
                let cpu_other = other.to_device_enum(&crate::device::Device::CPU)?;
                self.add(&cpu_other)
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Addition not supported for these tensor types",
                "add",
            )),
        }
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        match (self, other) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                let result = a - b;
                Ok(Tensor::F32(result))
            },
            (Tensor::F64(a), Tensor::F64(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                let result = a - b;
                Ok(Tensor::F64(result))
            },
            (Tensor::I64(a), Tensor::I64(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                let result = a - b;
                Ok(Tensor::I64(result))
            },
            (Tensor::C32(a), Tensor::C32(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                let result = a - b;
                Ok(Tensor::C32(result))
            },
            (Tensor::C64(a), Tensor::C64(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                let result = a - b;
                Ok(Tensor::C64(result))
            },
            #[cfg(feature = "metal")]
            (Tensor::Metal(_), _) => {
                let cpu_self = self.to_device_enum(&crate::device::Device::CPU)?;
                cpu_self.sub(other)
            },
            #[cfg(feature = "metal")]
            (_, Tensor::Metal(_)) => {
                let cpu_other = other.to_device_enum(&crate::device::Device::CPU)?;
                self.sub(&cpu_other)
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Subtraction not supported for these tensor types",
                "sub",
            )),
        }
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        match (self, other) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                let result = a * b;
                Ok(Tensor::F32(result))
            },
            (Tensor::F64(a), Tensor::F64(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                let result = a * b;
                Ok(Tensor::F64(result))
            },
            (Tensor::I64(a), Tensor::I64(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                let result = a * b;
                Ok(Tensor::I64(result))
            },
            (Tensor::C32(a), Tensor::C32(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                let result = a * b;
                Ok(Tensor::C32(result))
            },
            (Tensor::C64(a), Tensor::C64(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                let result = a * b;
                Ok(Tensor::C64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Multiplication not supported for these tensor types",
                "mul",
            )),
        }
    }

    /// Element-wise division.
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        match (self, other) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                // Use element-wise division with numerical stability checks
                let mut result = a.clone();
                result.zip_mut_with(b, |a_val, &b_val| {
                    *a_val = if b_val.abs() < f32::MIN_POSITIVE {
                        // Handle division by zero or very small numbers
                        if *a_val == 0.0 {
                            f32::NAN // 0/0 case
                        } else if *a_val > 0.0 {
                            f32::INFINITY
                        } else {
                            f32::NEG_INFINITY
                        }
                    } else {
                        *a_val / b_val
                    };
                });
                Ok(Tensor::F32(result))
            },
            (Tensor::F64(a), Tensor::F64(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                // Use element-wise division with numerical stability checks
                let mut result = a.clone();
                result.zip_mut_with(b, |a_val, &b_val| {
                    *a_val = if b_val.abs() < f64::MIN_POSITIVE {
                        // Handle division by zero or very small numbers
                        if *a_val == 0.0 {
                            f64::NAN // 0/0 case
                        } else if *a_val > 0.0 {
                            f64::INFINITY
                        } else {
                            f64::NEG_INFINITY
                        }
                    } else {
                        *a_val / b_val
                    };
                });
                Ok(Tensor::F64(result))
            },
            (Tensor::C32(a), Tensor::C32(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                let result = a / b;
                Ok(Tensor::C32(result))
            },
            (Tensor::C64(a), Tensor::C64(b)) => {
                if !shapes_are_broadcastable(a.shape(), b.shape()) {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot broadcast shapes {:?} and {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }
                let result = a / b;
                Ok(Tensor::C64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Division not supported for these tensor types",
                "div",
            )),
        }
    }

    /// Broadcasting addition.
    pub fn broadcast_add(&self, other: &Tensor) -> Result<Tensor> {
        match (self, other) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                let result = a + b;
                Ok(Tensor::F32(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Broadcast add not supported for these tensor types",
                "broadcast_add",
            )),
        }
    }

    /// Scalar multiplication.
    pub fn scalar_mul(&self, scalar: f32) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a * scalar;
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a * scalar as f64;
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                let result = a * scalar as i64;
                Ok(Tensor::I64(result))
            },
            Tensor::C32(a) => {
                let result = a * scalar;
                Ok(Tensor::C32(result))
            },
            Tensor::C64(a) => {
                let result = a * scalar as f64;
                Ok(Tensor::C64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Scalar multiplication not supported for this tensor type",
                "scalar_mul",
            )),
        }
    }

    /// Scalar division.
    pub fn scalar_div(&self, scalar: f32) -> Result<Tensor> {
        self.scalar_mul(1.0 / scalar)
    }

    /// Scalar addition.
    pub fn add_scalar(&self, scalar: f32) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a + scalar;
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a + scalar as f64;
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                let result = a + scalar as i64;
                Ok(Tensor::I64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Scalar addition not supported for this tensor type",
                "add_scalar",
            )),
        }
    }

    /// Scalar subtraction.
    pub fn sub_scalar(&self, scalar: f32) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a - scalar;
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a - scalar as f64;
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                let result = a - scalar as i64;
                Ok(Tensor::I64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Scalar subtraction not supported for this tensor type",
                "sub_scalar",
            )),
        }
    }

    /// Division by scalar.
    pub fn div_scalar(&self, scalar: f32) -> Result<Tensor> {
        self.scalar_div(scalar)
    }

    /// Multiplication by scalar (alias for scalar_mul).
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        self.scalar_mul(scalar)
    }

    /// Scaled subtraction: self - other * factor.
    pub fn sub_scaled(&self, other: &Tensor, factor: f32) -> Result<Tensor> {
        let scaled = other.scalar_mul(factor)?;
        self.sub(&scaled)
    }

    /// Scaled addition: self + other * factor.
    pub fn add_scaled(&self, other: &Tensor, factor: f32) -> Result<Tensor> {
        let scaled = other.scalar_mul(factor)?;
        self.add(&scaled)
    }
}
