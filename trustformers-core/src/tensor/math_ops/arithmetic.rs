//! Arithmetic operations for tensors.
//!
//! This module contains basic arithmetic operations including addition, subtraction,
//! multiplication, division, and scalar operations with numerical stability features.
//! All operations include broadcasting support and numerical stability enhancements.

use super::super::Tensor;
use crate::errors::{Result, TrustformersError};
use scirs2_core::ndarray::ArrayD;

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
            #[cfg(all(target_os = "macos", feature = "metal"))]
            (Tensor::Metal(a_data), Tensor::Metal(b_data)) => {
                use crate::gpu_ops::metal::get_metal_backend;
                use crate::tensor::MetalTensorData;

                // GPU-to-GPU addition - stays on Metal!
                // eprintln!("✅ Tensor::add - GPU-to-GPU path (Metal + Metal → Metal)");

                if a_data.shape != b_data.shape {
                    return Err(TrustformersError::shape_error(format!(
                        "Cannot add Metal tensors with different shapes: {:?} and {:?}",
                        a_data.shape, b_data.shape
                    )));
                }

                let backend = get_metal_backend()?;
                let size = a_data.shape.iter().product();

                let output_buffer_id =
                    backend.add_gpu_to_gpu(&a_data.buffer_id, &b_data.buffer_id, size)?;

                Ok(Tensor::Metal(MetalTensorData {
                    buffer_id: output_buffer_id,
                    shape: a_data.shape.clone(),
                    dtype: a_data.dtype,
                }))
            },
            #[cfg(all(target_os = "macos", feature = "metal"))]
            (Tensor::Metal(_), _) | (_, Tensor::Metal(_)) => {
                // Mixed Metal/CPU - convert to CPU for now
                // TODO: Could upload CPU tensor to GPU instead
                let cpu_self = self.to_device_enum(&crate::device::Device::CPU)?;
                let cpu_other = other.to_device_enum(&crate::device::Device::CPU)?;
                cpu_self.add(&cpu_other)
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
            #[cfg(all(target_os = "macos", feature = "metal"))]
            (Tensor::Metal(_), _) => {
                let cpu_self = self.to_device_enum(&crate::device::Device::CPU)?;
                cpu_self.sub(other)
            },
            #[cfg(all(target_os = "macos", feature = "metal"))]
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

#[cfg(test)]
mod tests {
    use crate::errors::Result;
    use crate::tensor::Tensor;

    #[test]
    fn test_add_f32() -> Result<()> {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3])?;
        let b = Tensor::from_data(vec![4.0, 5.0, 6.0], &[3])?;
        let c = a.add(&b)?;
        let data = c.data()?;
        assert!((data[0] - 5.0).abs() < 1e-6);
        assert!((data[1] - 7.0).abs() < 1e-6);
        assert!((data[2] - 9.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_sub_f32() -> Result<()> {
        let a = Tensor::from_data(vec![5.0, 10.0], &[2])?;
        let b = Tensor::from_data(vec![3.0, 4.0], &[2])?;
        let c = a.sub(&b)?;
        let data = c.data()?;
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 6.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_mul_f32() -> Result<()> {
        let a = Tensor::from_data(vec![2.0, 3.0, 4.0], &[3])?;
        let b = Tensor::from_data(vec![5.0, 6.0, 7.0], &[3])?;
        let c = a.mul(&b)?;
        let data = c.data()?;
        assert!((data[0] - 10.0).abs() < 1e-5);
        assert!((data[1] - 18.0).abs() < 1e-5);
        assert!((data[2] - 28.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_div_f32() -> Result<()> {
        let a = Tensor::from_data(vec![10.0, 20.0], &[2])?;
        let b = Tensor::from_data(vec![2.0, 5.0], &[2])?;
        let c = a.div(&b)?;
        let data = c.data()?;
        assert!((data[0] - 5.0).abs() < 1e-5);
        assert!((data[1] - 4.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_add_shape_mismatch() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3]).expect("create failed");
        let b = Tensor::from_data(vec![1.0, 2.0], &[2]).expect("create failed");
        let result = a.add(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_scalar_mul() -> Result<()> {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3])?;
        let c = a.scalar_mul(3.0)?;
        let data = c.data()?;
        assert!((data[0] - 3.0).abs() < 1e-6);
        assert!((data[1] - 6.0).abs() < 1e-6);
        assert!((data[2] - 9.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_scalar_div() -> Result<()> {
        let a = Tensor::from_data(vec![6.0, 9.0], &[2])?;
        let c = a.scalar_div(3.0)?;
        let data = c.data()?;
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 3.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_add_scalar() -> Result<()> {
        let a = Tensor::from_data(vec![1.0, 2.0], &[2])?;
        let c = a.add_scalar(10.0)?;
        let data = c.data()?;
        assert!((data[0] - 11.0).abs() < 1e-6);
        assert!((data[1] - 12.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_sub_scalar() -> Result<()> {
        let a = Tensor::from_data(vec![10.0, 20.0], &[2])?;
        let c = a.sub_scalar(5.0)?;
        let data = c.data()?;
        assert!((data[0] - 5.0).abs() < 1e-6);
        assert!((data[1] - 15.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_mul_scalar() -> Result<()> {
        let a = Tensor::from_data(vec![2.0, 3.0], &[2])?;
        let c = a.mul_scalar(4.0)?;
        let data = c.data()?;
        assert!((data[0] - 8.0).abs() < 1e-6);
        assert!((data[1] - 12.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_div_scalar() -> Result<()> {
        let a = Tensor::from_data(vec![8.0, 12.0], &[2])?;
        let c = a.div_scalar(4.0)?;
        let data = c.data()?;
        assert!((data[0] - 2.0).abs() < 1e-5);
        assert!((data[1] - 3.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_add_2d() -> Result<()> {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = Tensor::from_data(vec![10.0, 20.0, 30.0, 40.0], &[2, 2])?;
        let c = a.add(&b)?;
        assert_eq!(c.shape(), vec![2, 2]);
        let data = c.data()?;
        assert!((data[0] - 11.0).abs() < 1e-5);
        assert!((data[3] - 44.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_sub_scaled() -> Result<()> {
        let a = Tensor::from_data(vec![10.0, 20.0], &[2])?;
        let b = Tensor::from_data(vec![1.0, 2.0], &[2])?;
        let c = a.sub_scaled(&b, 3.0)?;
        let data = c.data()?;
        assert!((data[0] - 7.0).abs() < 1e-5);
        assert!((data[1] - 14.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_add_scaled() -> Result<()> {
        let a = Tensor::from_data(vec![10.0, 20.0], &[2])?;
        let b = Tensor::from_data(vec![1.0, 2.0], &[2])?;
        let c = a.add_scaled(&b, 5.0)?;
        let data = c.data()?;
        assert!((data[0] - 15.0).abs() < 1e-5);
        assert!((data[1] - 30.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_add_i64() -> Result<()> {
        let a = Tensor::from_vec_i64(vec![1, 2, 3], &[3])?;
        let b = Tensor::from_vec_i64(vec![4, 5, 6], &[3])?;
        let c = a.add(&b)?;
        assert_eq!(c.shape(), vec![3]);
        Ok(())
    }

    #[test]
    fn test_broadcast_add() -> Result<()> {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = Tensor::from_data(vec![10.0, 20.0], &[1, 2])?;
        let c = a.broadcast_add(&b)?;
        assert_eq!(c.shape(), vec![2, 2]);
        Ok(())
    }

    #[test]
    fn test_add_zeros_identity() -> Result<()> {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3])?;
        let z = Tensor::zeros(&[3])?;
        let c = a.add(&z)?;
        let data = c.data()?;
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_mul_ones_identity() -> Result<()> {
        let a = Tensor::from_data(vec![5.0, 10.0], &[2])?;
        let o = Tensor::ones(&[2])?;
        let c = a.mul(&o)?;
        let data = c.data()?;
        assert!((data[0] - 5.0).abs() < 1e-6);
        assert!((data[1] - 10.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_scalar_mul_zero() -> Result<()> {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3])?;
        let c = a.scalar_mul(0.0)?;
        let data = c.data()?;
        for val in &data {
            assert!(val.abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_sub_self_is_zero() -> Result<()> {
        let a = Tensor::from_data(vec![5.0, 10.0, 15.0], &[3])?;
        let c = a.sub(&a)?;
        let data = c.data()?;
        for val in &data {
            assert!(val.abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_add_negative() -> Result<()> {
        let a = Tensor::from_data(vec![-1.0, -2.0], &[2])?;
        let b = Tensor::from_data(vec![-3.0, -4.0], &[2])?;
        let c = a.add(&b)?;
        let data = c.data()?;
        assert!((data[0] - (-4.0)).abs() < 1e-6);
        assert!((data[1] - (-6.0)).abs() < 1e-6);
        Ok(())
    }
}
