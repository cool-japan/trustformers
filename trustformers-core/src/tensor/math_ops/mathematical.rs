//! Mathematical functions for tensors.
//!
//! This module contains comprehensive mathematical functions including:
//! - Power and Root Functions: pow, sqrt, rsqrt, square, reciprocal
//! - Exponential and Logarithmic: exp, log
//! - Trigonometric Functions: sin, cos, tan, asin, acos, atan
//! - Basic Math Functions: abs, neg
//! - Utility Math Functions: sign, round, floor, ceil, trunc
//! - Special Functions: isnan, isinf, isfinite

use super::super::Tensor;
use crate::errors::{Result, TrustformersError};

impl Tensor {
    /// Element-wise power operation.
    pub fn pow(&self, exponent: f32) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.powf(exponent));
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.powf(exponent as f64));
                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Power operation not supported for this tensor type",
                "pow",
            )),
        }
    }

    /// Raise tensor to a scalar power (alias for pow).
    ///
    /// # Arguments
    ///
    /// * `exponent` - The exponent to raise each element to
    ///
    /// # Returns
    ///
    /// A new tensor with each element raised to the given power.
    pub fn pow_scalar(&self, exponent: f64) -> Result<Tensor> {
        self.pow(exponent as f32)
    }

    /// Absolute value.
    pub fn abs(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.abs());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.abs());
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                let result = a.mapv(|x| x.abs());
                Ok(Tensor::I64(result))
            },
            Tensor::C32(a) => {
                let result = a.mapv(|x| x.norm());
                Ok(Tensor::F32(result))
            },
            Tensor::C64(a) => {
                let result = a.mapv(|x| x.norm());
                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Absolute value not supported for this tensor type",
                "abs",
            )),
        }
    }

    /// Negation.
    pub fn neg(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| -x);
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| -x);
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                let result = a.mapv(|x| -x);
                Ok(Tensor::I64(result))
            },
            Tensor::C32(a) => {
                let result = a.mapv(|x| -x);
                Ok(Tensor::C32(result))
            },
            Tensor::C64(a) => {
                let result = a.mapv(|x| -x);
                Ok(Tensor::C64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Negation not supported for this tensor type",
                "neg",
            )),
        }
    }

    /// Element-wise square root.
    ///
    /// # Returns
    ///
    /// A new tensor with square root applied to each element.
    pub fn sqrt(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.sqrt());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.sqrt());
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                // Convert to F64, compute sqrt, and convert back to I64
                let result = a.mapv(|x| (x as f64).sqrt().round() as i64);
                Ok(Tensor::I64(result))
            },
            Tensor::C32(a) => {
                // Complex square root
                let result = a.mapv(|x| x.sqrt());
                Ok(Tensor::C32(result))
            },
            Tensor::C64(a) => {
                // Complex square root
                let result = a.mapv(|x| x.sqrt());
                Ok(Tensor::C64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Square root not supported for this tensor type",
                "sqrt",
            )),
        }
    }

    /// Natural logarithm.
    pub fn log(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.ln());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.ln());
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                // Convert I64 to F64, compute log, then convert back
                let result = a.mapv(|x| (x as f64).ln() as i64);
                Ok(Tensor::I64(result))
            },
            Tensor::C32(a) => {
                let result = a.mapv(|x| x.ln());
                Ok(Tensor::C32(result))
            },
            Tensor::C64(a) => {
                let result = a.mapv(|x| x.ln());
                Ok(Tensor::C64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Log operation not implemented for this tensor type",
                "log",
            )),
        }
    }

    /// Element-wise exponential function.
    ///
    /// # Returns
    ///
    /// A new tensor with exponential function applied to each element.
    pub fn exp(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.exp());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.exp());
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                // Convert I64 to F64, compute exp, then convert back
                let result = a.mapv(|x| (x as f64).exp() as i64);
                Ok(Tensor::I64(result))
            },
            Tensor::C32(a) => {
                let result = a.mapv(|x| x.exp());
                Ok(Tensor::C32(result))
            },
            Tensor::C64(a) => {
                let result = a.mapv(|x| x.exp());
                Ok(Tensor::C64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Exp operation not implemented for this tensor type",
                "exp",
            )),
        }
    }

    /// Sine function.
    pub fn sin(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.sin());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.sin());
                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Sine operation not supported for this tensor type",
                "sin",
            )),
        }
    }

    /// Cosine function.
    pub fn cos(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.cos());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.cos());
                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Cosine operation not supported for this tensor type",
                "cos",
            )),
        }
    }

    /// Tangent function.
    pub fn tan(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.tan());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.tan());
                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Tangent operation not supported for this tensor type",
                "tan",
            )),
        }
    }

    /// Arc sine function.
    pub fn asin(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.asin());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.asin());
                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Arc sine operation not supported for this tensor type",
                "asin",
            )),
        }
    }

    /// Arc cosine function.
    pub fn acos(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.acos());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.acos());
                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Arc cosine operation not supported for this tensor type",
                "acos",
            )),
        }
    }

    /// Arc tangent function.
    pub fn atan(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.atan());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.atan());
                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Arc tangent operation not supported for this tensor type",
                "atan",
            )),
        }
    }

    /// Square operation - x².
    pub fn square(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x * x);
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x * x);
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                let result = a.mapv(|x| x * x);
                Ok(Tensor::I64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Square operation not supported for this tensor type",
                "square",
            )),
        }
    }

    /// Reciprocal operation - 1/x.
    pub fn reciprocal(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| 1.0 / x);
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| 1.0 / x);
                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Reciprocal operation not supported for this tensor type",
                "reciprocal",
            )),
        }
    }

    /// Reciprocal square root - 1/√x.
    pub fn rsqrt(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| 1.0 / x.sqrt());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| 1.0 / x.sqrt());
                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Reciprocal square root not supported for this tensor type",
                "rsqrt",
            )),
        }
    }

    /// Check for NaN values.
    pub fn isnan(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| if x.is_nan() { 1.0f32 } else { 0.0f32 });
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| if x.is_nan() { 1.0f64 } else { 0.0f64 });
                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "IsNaN check not supported for this tensor type",
                "isnan",
            )),
        }
    }

    /// Check for infinite values.
    pub fn isinf(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| if x.is_infinite() { 1.0f32 } else { 0.0f32 });
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| if x.is_infinite() { 1.0f64 } else { 0.0f64 });
                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "IsInf check not supported for this tensor type",
                "isinf",
            )),
        }
    }

    /// Check for finite values.
    pub fn isfinite(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| if x.is_finite() { 1.0f32 } else { 0.0f32 });
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| if x.is_finite() { 1.0f64 } else { 0.0f64 });
                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "IsFinite check not supported for this tensor type",
                "isfinite",
            )),
        }
    }

    /// Element-wise sign function.
    ///
    /// Returns 1 for positive values, -1 for negative values, and 0 for zero.
    pub fn sign(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                });
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                });
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                let result = a.mapv(|x| {
                    if x > 0 {
                        1
                    } else if x < 0 {
                        -1
                    } else {
                        0
                    }
                });
                Ok(Tensor::I64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Sign operation not supported for this tensor type",
                "sign",
            )),
        }
    }

    /// Round values to nearest integer.
    ///
    /// Rounds halfway cases away from zero.
    pub fn round(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.round());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.round());
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                // For integers, round is identity
                Ok(Tensor::I64(a.clone()))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Round operation not supported for this tensor type",
                "round",
            )),
        }
    }

    /// Floor operation - round down to nearest integer.
    ///
    /// Returns the largest integer less than or equal to the input.
    pub fn floor(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.floor());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.floor());
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                // For integers, floor is identity
                Ok(Tensor::I64(a.clone()))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Floor operation not supported for this tensor type",
                "floor",
            )),
        }
    }

    /// Ceiling operation - round up to nearest integer.
    ///
    /// Returns the smallest integer greater than or equal to the input.
    pub fn ceil(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.ceil());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.ceil());
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                // For integers, ceil is identity
                Ok(Tensor::I64(a.clone()))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Ceiling operation not supported for this tensor type",
                "ceil",
            )),
        }
    }

    /// Truncate operation - round towards zero.
    ///
    /// Removes the fractional part, effectively rounding towards zero.
    pub fn trunc(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.trunc());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.trunc());
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                // For integers, trunc is identity
                Ok(Tensor::I64(a.clone()))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Truncate operation not supported for this tensor type",
                "trunc",
            )),
        }
    }
}
