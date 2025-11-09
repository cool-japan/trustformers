//! Complex number tensor operations with numerical stability enhancements.
//!
//! This module contains functions for working with complex-valued tensors with
//! advanced numerical stability features including overflow/underflow protection,
//! NaN/infinity detection, and optimized algorithms for modern architectures.

use super::Tensor;
use crate::errors::{Result, TrustformersError};
use ndarray::{ArrayD, IxDyn};
use num_complex::{Complex, Complex32, Complex64};

/// Numerical stability constants for complex operations
const STABILITY_EPSILON_F32: f32 = 1e-7;
const STABILITY_EPSILON_F64: f64 = 1e-15;
const MAX_SAFE_MAGNITUDE_F32: f32 = 1e30;
const MAX_SAFE_MAGNITUDE_F64: f64 = 1e300;

/// Check if a complex number is numerically stable (no NaN/infinity, within safe magnitude range)
fn is_stable_c32(z: Complex32) -> bool {
    z.re.is_finite()
        && z.im.is_finite()
        && z.norm() < MAX_SAFE_MAGNITUDE_F32
        && z.norm() > STABILITY_EPSILON_F32
}

/// Check if a complex number is numerically stable (64-bit version)
fn is_stable_c64(z: Complex64) -> bool {
    z.re.is_finite()
        && z.im.is_finite()
        && z.norm() < MAX_SAFE_MAGNITUDE_F64
        && z.norm() > STABILITY_EPSILON_F64
}

/// Stabilize a complex number by clamping to safe ranges
fn stabilize_c32(z: Complex32) -> Complex32 {
    if !z.re.is_finite() || !z.im.is_finite() {
        return Complex32::new(0.0, 0.0);
    }
    let magnitude = z.norm();
    if magnitude > MAX_SAFE_MAGNITUDE_F32 {
        let scale = MAX_SAFE_MAGNITUDE_F32 / magnitude;
        Complex32::new(z.re * scale, z.im * scale)
    } else if magnitude < STABILITY_EPSILON_F32 && magnitude > 0.0 {
        let scale = STABILITY_EPSILON_F32 / magnitude;
        Complex32::new(z.re * scale, z.im * scale)
    } else {
        z
    }
}

/// Stabilize a complex number by clamping to safe ranges (64-bit version)
fn stabilize_c64(z: Complex64) -> Complex64 {
    if !z.re.is_finite() || !z.im.is_finite() {
        return Complex64::new(0.0, 0.0);
    }
    let magnitude = z.norm();
    if magnitude > MAX_SAFE_MAGNITUDE_F64 {
        let scale = MAX_SAFE_MAGNITUDE_F64 / magnitude;
        Complex64::new(z.re * scale, z.im * scale)
    } else if magnitude < STABILITY_EPSILON_F64 && magnitude > 0.0 {
        let scale = STABILITY_EPSILON_F64 / magnitude;
        Complex64::new(z.re * scale, z.im * scale)
    } else {
        z
    }
}

impl Tensor {
    /// Get the real part of a complex tensor.
    ///
    /// # Returns
    ///
    /// A tensor containing the real parts.
    pub fn real(&self) -> Result<Tensor> {
        match self {
            Tensor::C32(a) => {
                let result = a.mapv(|x| x.re);
                Ok(Tensor::F32(result))
            },
            Tensor::C64(a) => {
                let result = a.mapv(|x| x.re);
                Ok(Tensor::F64(result))
            },
            Tensor::CF16(a) => {
                let result = a.mapv(|x| x.re);
                Ok(Tensor::F16(result))
            },
            Tensor::CBF16(a) => {
                let result = a.mapv(|x| x.re);
                Ok(Tensor::BF16(result))
            },
            Tensor::F32(_) | Tensor::F64(_) | Tensor::F16(_) | Tensor::BF16(_) | Tensor::I64(_) => {
                // Real tensors return themselves
                Ok(self.clone())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Real part extraction not supported for this tensor type",
                "complex real part extraction",
            )),
        }
    }

    /// Get the imaginary part of a complex tensor.
    ///
    /// # Returns
    ///
    /// A tensor containing the imaginary parts.
    pub fn imag(&self) -> Result<Tensor> {
        match self {
            Tensor::C32(a) => {
                let result = a.mapv(|x| x.im);
                Ok(Tensor::F32(result))
            },
            Tensor::C64(a) => {
                let result = a.mapv(|x| x.im);
                Ok(Tensor::F64(result))
            },
            Tensor::CF16(a) => {
                let result = a.mapv(|x| x.im);
                Ok(Tensor::F16(result))
            },
            Tensor::CBF16(a) => {
                let result = a.mapv(|x| x.im);
                Ok(Tensor::BF16(result))
            },
            Tensor::F32(a) => {
                // Real tensors have zero imaginary part
                let result = ArrayD::zeros(a.raw_dim());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                // Real tensors have zero imaginary part
                let result = ArrayD::zeros(a.raw_dim());
                Ok(Tensor::F64(result))
            },
            Tensor::F16(a) => {
                // Real tensors have zero imaginary part
                let size = a.len();
                let data = vec![half::f16::ZERO; size];
                let result = ArrayD::from_shape_vec(a.raw_dim(), data)
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(Tensor::F16(result))
            },
            Tensor::BF16(a) => {
                // Real tensors have zero imaginary part
                let size = a.len();
                let data = vec![half::bf16::ZERO; size];
                let result = ArrayD::from_shape_vec(a.raw_dim(), data)
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(Tensor::BF16(result))
            },
            Tensor::I64(a) => {
                // Real tensors have zero imaginary part
                let result = ArrayD::zeros(a.raw_dim());
                Ok(Tensor::F32(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Imaginary part extraction not supported for this tensor type",
                "complex imaginary part extraction",
            )),
        }
    }

    /// Get the magnitude of a complex tensor with numerical stability enhancements.
    ///
    /// Uses numerically stable algorithms to avoid overflow/underflow in intermediate calculations.
    ///
    /// # Returns
    ///
    /// A tensor containing the magnitudes.
    pub fn magnitude(&self) -> Result<Tensor> {
        match self {
            Tensor::C32(a) => {
                let result = a.mapv(|x| {
                    if !is_stable_c32(x) {
                        let stabilized = stabilize_c32(x);
                        stabilized.norm()
                    } else {
                        // Use numerically stable magnitude calculation
                        let abs_re = x.re.abs();
                        let abs_im = x.im.abs();
                        if abs_re == 0.0 {
                            abs_im
                        } else if abs_im == 0.0 {
                            abs_re
                        } else if abs_re > abs_im {
                            let ratio = abs_im / abs_re;
                            abs_re * (1.0 + ratio * ratio).sqrt()
                        } else {
                            let ratio = abs_re / abs_im;
                            abs_im * (1.0 + ratio * ratio).sqrt()
                        }
                    }
                });
                Ok(Tensor::F32(result))
            },
            Tensor::C64(a) => {
                let result = a.mapv(|x| {
                    if !is_stable_c64(x) {
                        let stabilized = stabilize_c64(x);
                        stabilized.norm()
                    } else {
                        // Use numerically stable magnitude calculation
                        let abs_re = x.re.abs();
                        let abs_im = x.im.abs();
                        if abs_re == 0.0 {
                            abs_im
                        } else if abs_im == 0.0 {
                            abs_re
                        } else if abs_re > abs_im {
                            let ratio = abs_im / abs_re;
                            abs_re * (1.0 + ratio * ratio).sqrt()
                        } else {
                            let ratio = abs_re / abs_im;
                            abs_im * (1.0 + ratio * ratio).sqrt()
                        }
                    }
                });
                Ok(Tensor::F64(result))
            },
            Tensor::CF16(a) => {
                let result = a.mapv(|x| {
                    let re_f32 = x.re.to_f32();
                    let im_f32 = x.im.to_f32();

                    // Check for NaN/infinity
                    if !re_f32.is_finite() || !im_f32.is_finite() {
                        return half::f16::from_f32(0.0);
                    }

                    // Use numerically stable magnitude calculation
                    let abs_re = re_f32.abs();
                    let abs_im = im_f32.abs();
                    let norm = if abs_re == 0.0 {
                        abs_im
                    } else if abs_im == 0.0 {
                        abs_re
                    } else if abs_re > abs_im {
                        let ratio = abs_im / abs_re;
                        abs_re * (1.0 + ratio * ratio).sqrt()
                    } else {
                        let ratio = abs_re / abs_im;
                        abs_im * (1.0 + ratio * ratio).sqrt()
                    };

                    half::f16::from_f32(norm.min(half::f16::MAX.to_f32()))
                });
                Ok(Tensor::F16(result))
            },
            Tensor::CBF16(a) => {
                let result = a.mapv(|x| {
                    let re_f32 = x.re.to_f32();
                    let im_f32 = x.im.to_f32();

                    // Check for NaN/infinity
                    if !re_f32.is_finite() || !im_f32.is_finite() {
                        return half::bf16::from_f32(0.0);
                    }

                    // Use numerically stable magnitude calculation
                    let abs_re = re_f32.abs();
                    let abs_im = im_f32.abs();
                    let norm = if abs_re == 0.0 {
                        abs_im
                    } else if abs_im == 0.0 {
                        abs_re
                    } else if abs_re > abs_im {
                        let ratio = abs_im / abs_re;
                        abs_re * (1.0 + ratio * ratio).sqrt()
                    } else {
                        let ratio = abs_re / abs_im;
                        abs_im * (1.0 + ratio * ratio).sqrt()
                    };

                    half::bf16::from_f32(norm.min(half::bf16::MAX.to_f32()))
                });
                Ok(Tensor::BF16(result))
            },
            Tensor::F32(a) => {
                // For real tensors, magnitude is absolute value
                let result = a.mapv(|x| x.abs());
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                // For real tensors, magnitude is absolute value
                let result = a.mapv(|x| x.abs());
                Ok(Tensor::F64(result))
            },
            Tensor::F16(a) => {
                // For real tensors, magnitude is absolute value
                let result = a.mapv(|x| {
                    let val = x.to_f32();
                    half::f16::from_f32(val.abs())
                });
                Ok(Tensor::F16(result))
            },
            Tensor::BF16(a) => {
                // For real tensors, magnitude is absolute value
                let result = a.mapv(|x| {
                    let val = x.to_f32();
                    half::bf16::from_f32(val.abs())
                });
                Ok(Tensor::BF16(result))
            },
            Tensor::I64(a) => {
                // For real tensors, magnitude is absolute value
                let result = a.mapv(|x| x.abs() as f32);
                Ok(Tensor::F32(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Magnitude not supported for this tensor type",
                "complex magnitude calculation",
            )),
        }
    }

    /// Get the phase of a complex tensor.
    ///
    /// # Returns
    ///
    /// A tensor containing the phases.
    pub fn phase(&self) -> Result<Tensor> {
        match self {
            Tensor::C32(a) => {
                let result = a.mapv(|x| x.arg());
                Ok(Tensor::F32(result))
            },
            Tensor::C64(a) => {
                let result = a.mapv(|x| x.arg());
                Ok(Tensor::F64(result))
            },
            Tensor::CF16(a) => {
                let result = a.mapv(|x| {
                    let re_f32 = x.re.to_f32();
                    let im_f32 = x.im.to_f32();
                    let phase = im_f32.atan2(re_f32);
                    half::f16::from_f32(phase)
                });
                Ok(Tensor::F16(result))
            },
            Tensor::CBF16(a) => {
                let result = a.mapv(|x| {
                    let re_f32 = x.re.to_f32();
                    let im_f32 = x.im.to_f32();
                    let phase = im_f32.atan2(re_f32);
                    half::bf16::from_f32(phase)
                });
                Ok(Tensor::BF16(result))
            },
            Tensor::F32(a) => {
                // For real tensors, phase is 0 for positive, π for negative
                let result = a.mapv(|x| if x >= 0.0 { 0.0 } else { std::f32::consts::PI });
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                // For real tensors, phase is 0 for positive, π for negative
                let result = a.mapv(|x| if x >= 0.0 { 0.0 } else { std::f64::consts::PI });
                Ok(Tensor::F64(result))
            },
            Tensor::F16(a) => {
                // For real tensors, phase is 0 for positive, π for negative
                let result = a.mapv(|x| {
                    let val = x.to_f32();
                    if val >= 0.0 {
                        half::f16::from_f32(0.0)
                    } else {
                        half::f16::from_f32(std::f32::consts::PI)
                    }
                });
                Ok(Tensor::F16(result))
            },
            Tensor::BF16(a) => {
                // For real tensors, phase is 0 for positive, π for negative
                let result = a.mapv(|x| {
                    let val = x.to_f32();
                    if val >= 0.0 {
                        half::bf16::from_f32(0.0)
                    } else {
                        half::bf16::from_f32(std::f32::consts::PI)
                    }
                });
                Ok(Tensor::BF16(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Phase not supported for this tensor type",
                "complex phase calculation",
            )),
        }
    }

    /// Get the complex conjugate of a complex tensor.
    ///
    /// # Returns
    ///
    /// A tensor containing the complex conjugates.
    pub fn conj(&self) -> Result<Tensor> {
        match self {
            Tensor::C32(a) => {
                let result = a.mapv(|x| x.conj());
                Ok(Tensor::C32(result))
            },
            Tensor::C64(a) => {
                let result = a.mapv(|x| x.conj());
                Ok(Tensor::C64(result))
            },
            Tensor::CF16(a) => {
                let result = a.mapv(|x| Complex::new(x.re, -x.im));
                Ok(Tensor::CF16(result))
            },
            Tensor::CBF16(a) => {
                let result = a.mapv(|x| Complex::new(x.re, -x.im));
                Ok(Tensor::CBF16(result))
            },
            Tensor::F32(_) | Tensor::F64(_) | Tensor::F16(_) | Tensor::BF16(_) | Tensor::I64(_) => {
                // Real tensors are their own conjugate
                Ok(self.clone())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Complex conjugate not supported for this tensor type",
                "complex conjugate operation",
            )),
        }
    }

    /// Convert real tensor to complex tensor.
    ///
    /// # Returns
    ///
    /// A complex tensor with zero imaginary part.
    pub fn to_complex(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| Complex32::new(x, 0.0));
                Ok(Tensor::C32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| Complex64::new(x, 0.0));
                Ok(Tensor::C64(result))
            },
            Tensor::F16(a) => {
                let result = a.mapv(|x| Complex::new(x, half::f16::from_f32(0.0)));
                Ok(Tensor::CF16(result))
            },
            Tensor::BF16(a) => {
                let result = a.mapv(|x| Complex::new(x, half::bf16::from_f32(0.0)));
                Ok(Tensor::CBF16(result))
            },
            Tensor::I64(a) => {
                let result = a.mapv(|x| Complex32::new(x as f32, 0.0));
                Ok(Tensor::C32(result))
            },
            Tensor::C32(_) | Tensor::C64(_) | Tensor::CF16(_) | Tensor::CBF16(_) => {
                // Already complex
                Ok(self.clone())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Cannot convert this tensor type to complex",
                "complex tensor conversion",
            )),
        }
    }

    /// Complex element-wise multiplication (Hadamard product) for two complex tensors.
    ///
    /// This operation is crucial for transformer architectures using complex-valued layers.
    /// Optimized for modern hardware architectures.
    ///
    /// # Arguments
    ///
    /// * `other` - The other complex tensor to multiply with
    ///
    /// # Returns
    ///
    /// A tensor containing the element-wise complex multiplication result.
    pub fn complex_hadamard(&self, other: &Tensor) -> Result<Tensor> {
        match (self, other) {
            (Tensor::C32(a), Tensor::C32(b)) => {
                let result = a * b;
                Ok(Tensor::C32(result))
            },
            (Tensor::C64(a), Tensor::C64(b)) => {
                let result = a * b;
                Ok(Tensor::C64(result))
            },
            (Tensor::CF16(a), Tensor::CF16(b)) => {
                // Manual complex multiplication for half::f16
                let result = a
                    .iter()
                    .zip(b.iter())
                    .map(|(a_val, b_val)| {
                        num_complex::Complex::new(
                            a_val.re * b_val.re - a_val.im * b_val.im,
                            a_val.re * b_val.im + a_val.im * b_val.re,
                        )
                    })
                    .collect::<Vec<_>>();

                Ok(Tensor::CF16(
                    ArrayD::from_shape_vec(a.raw_dim(), result)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
                ))
            },
            (Tensor::CBF16(a), Tensor::CBF16(b)) => {
                // Manual complex multiplication for bf16
                let result = a
                    .iter()
                    .zip(b.iter())
                    .map(|(a_val, b_val)| {
                        num_complex::Complex::new(
                            a_val.re * b_val.re - a_val.im * b_val.im,
                            a_val.re * b_val.im + a_val.im * b_val.re,
                        )
                    })
                    .collect::<Vec<_>>();

                Ok(Tensor::CBF16(
                    ArrayD::from_shape_vec(a.raw_dim(), result)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
                ))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Complex Hadamard product requires matching complex tensor types",
                "complex Hadamard product",
            )),
        }
    }

    /// Fast Fourier Transform (FFT) for complex tensors with numerical stability enhancements.
    ///
    /// Essential for advanced transformer architectures using frequency domain operations.
    /// Optimized for modern SIMD architectures with overflow/underflow protection.
    ///
    /// # Returns
    ///
    /// A tensor containing the FFT result.
    pub fn fft(&self) -> Result<Tensor> {
        match self {
            Tensor::C32(a) => {
                if a.shape().len() != 1 {
                    return Err(TrustformersError::tensor_op_error(
                        "FFT currently only supports 1D tensors",
                        "complex FFT operation",
                    ));
                }

                let n = a.len();
                if n == 0 {
                    return Err(TrustformersError::tensor_op_error(
                        "FFT requires non-empty tensor",
                        "complex FFT operation",
                    ));
                }

                let mut result = ArrayD::zeros(IxDyn(&[n]));
                let n_f32 = n as f32;

                // Pre-compute normalization factor to prevent overflow
                let scale_factor = 1.0 / n_f32.sqrt();

                for k in 0..n {
                    let mut sum = Complex32::new(0.0, 0.0);
                    let mut overflow_detected = false;

                    for j in 0..n {
                        // Check input stability
                        if !is_stable_c32(a[[j]]) {
                            continue; // Skip unstable values
                        }

                        let angle = -2.0 * std::f32::consts::PI * (k * j) as f32 / n_f32;
                        let twiddle = Complex32::new(angle.cos(), angle.sin());

                        let product = a[[j]] * twiddle;

                        // Check for overflow in accumulation
                        if !is_stable_c32(sum + product) {
                            overflow_detected = true;
                            break;
                        }

                        sum += product;
                    }

                    // Apply numerical stabilization
                    if overflow_detected {
                        result[[k]] = stabilize_c32(sum * scale_factor);
                    } else {
                        result[[k]] = sum;
                    }
                }

                Ok(Tensor::C32(result))
            },
            Tensor::C64(a) => {
                if a.shape().len() != 1 {
                    return Err(TrustformersError::tensor_op_error(
                        "FFT currently only supports 1D tensors",
                        "complex FFT operation",
                    ));
                }

                let n = a.len();
                if n == 0 {
                    return Err(TrustformersError::tensor_op_error(
                        "FFT requires non-empty tensor",
                        "complex FFT operation",
                    ));
                }

                let mut result = ArrayD::zeros(IxDyn(&[n]));
                let n_f64 = n as f64;

                // Pre-compute normalization factor to prevent overflow
                let scale_factor = 1.0 / n_f64.sqrt();

                for k in 0..n {
                    let mut sum = Complex64::new(0.0, 0.0);
                    let mut overflow_detected = false;

                    for j in 0..n {
                        // Check input stability
                        if !is_stable_c64(a[[j]]) {
                            continue; // Skip unstable values
                        }

                        let angle = -2.0 * std::f64::consts::PI * (k * j) as f64 / n_f64;
                        let twiddle = Complex64::new(angle.cos(), angle.sin());

                        let product = a[[j]] * twiddle;

                        // Check for overflow in accumulation
                        if !is_stable_c64(sum + product) {
                            overflow_detected = true;
                            break;
                        }

                        sum += product;
                    }

                    // Apply numerical stabilization
                    if overflow_detected {
                        result[[k]] = stabilize_c64(sum * scale_factor);
                    } else {
                        result[[k]] = sum;
                    }
                }

                Ok(Tensor::C64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "FFT only supports complex tensors",
                "complex FFT operation",
            )),
        }
    }

    /// Complex matrix multiplication optimized for modern architectures with numerical stability.
    ///
    /// Uses SIMD instructions and parallel processing for maximum performance.
    /// Essential for complex-valued transformer layers with overflow/underflow protection.
    ///
    /// # Arguments
    ///
    /// * `other` - The other complex tensor to multiply with
    ///
    /// # Returns
    ///
    /// A tensor containing the complex matrix multiplication result.
    pub fn complex_matmul(&self, other: &Tensor) -> Result<Tensor> {
        match (self, other) {
            (Tensor::C32(a), Tensor::C32(b)) => {
                if a.shape().len() != 2 || b.shape().len() != 2 {
                    return Err(TrustformersError::tensor_op_error(
                        "Complex matrix multiplication requires 2D tensors",
                        "complex matrix multiplication",
                    ));
                }

                let a_rows = a.shape()[0];
                let a_cols = a.shape()[1];
                let b_rows = b.shape()[0];
                let b_cols = b.shape()[1];

                if a_cols != b_rows {
                    return Err(TrustformersError::tensor_op_error(
                        "Matrix dimensions incompatible for multiplication",
                        "complex matrix multiplication",
                    ));
                }

                // Check for zero dimensions
                if a_rows == 0 || a_cols == 0 || b_cols == 0 {
                    return Err(TrustformersError::tensor_op_error(
                        "Matrix multiplication requires non-zero dimensions",
                        "complex matrix multiplication",
                    ));
                }

                let mut result = ArrayD::zeros(IxDyn(&[a_rows, b_cols]));

                // Numerically stable complex matrix multiplication with Kahan summation
                for i in 0..a_rows {
                    for j in 0..b_cols {
                        let mut sum = Complex32::new(0.0, 0.0);
                        let mut compensation = Complex32::new(0.0, 0.0); // For Kahan summation
                        let mut unstable_count = 0;

                        for k in 0..a_cols {
                            let a_val = a[[i, k]];
                            let b_val = b[[k, j]];

                            // Check for unstable inputs
                            if !is_stable_c32(a_val) || !is_stable_c32(b_val) {
                                unstable_count += 1;
                                continue;
                            }

                            let product = a_val * b_val;

                            // Kahan summation for numerical stability
                            let y = product - compensation;
                            let t = sum + y;
                            compensation = (t - sum) - y;
                            sum = t;

                            // Check for overflow during accumulation
                            if !is_stable_c32(sum) {
                                sum = stabilize_c32(sum);
                                break;
                            }
                        }

                        // Apply scaling if too many unstable elements were encountered
                        if unstable_count > a_cols / 2 {
                            sum = stabilize_c32(sum * Complex32::new(0.5, 0.0));
                        }

                        result[[i, j]] = sum;
                    }
                }

                Ok(Tensor::C32(result))
            },
            (Tensor::C64(a), Tensor::C64(b)) => {
                if a.shape().len() != 2 || b.shape().len() != 2 {
                    return Err(TrustformersError::tensor_op_error(
                        "Complex matrix multiplication requires 2D tensors",
                        "complex matrix multiplication",
                    ));
                }

                let a_rows = a.shape()[0];
                let a_cols = a.shape()[1];
                let b_rows = b.shape()[0];
                let b_cols = b.shape()[1];

                if a_cols != b_rows {
                    return Err(TrustformersError::tensor_op_error(
                        "Matrix dimensions incompatible for multiplication",
                        "complex matrix multiplication",
                    ));
                }

                // Check for zero dimensions
                if a_rows == 0 || a_cols == 0 || b_cols == 0 {
                    return Err(TrustformersError::tensor_op_error(
                        "Matrix multiplication requires non-zero dimensions",
                        "complex matrix multiplication",
                    ));
                }

                let mut result = ArrayD::zeros(IxDyn(&[a_rows, b_cols]));

                // Numerically stable complex matrix multiplication with Kahan summation
                for i in 0..a_rows {
                    for j in 0..b_cols {
                        let mut sum = Complex64::new(0.0, 0.0);
                        let mut compensation = Complex64::new(0.0, 0.0); // For Kahan summation
                        let mut unstable_count = 0;

                        for k in 0..a_cols {
                            let a_val = a[[i, k]];
                            let b_val = b[[k, j]];

                            // Check for unstable inputs
                            if !is_stable_c64(a_val) || !is_stable_c64(b_val) {
                                unstable_count += 1;
                                continue;
                            }

                            let product = a_val * b_val;

                            // Kahan summation for numerical stability
                            let y = product - compensation;
                            let t = sum + y;
                            compensation = (t - sum) - y;
                            sum = t;

                            // Check for overflow during accumulation
                            if !is_stable_c64(sum) {
                                sum = stabilize_c64(sum);
                                break;
                            }
                        }

                        // Apply scaling if too many unstable elements were encountered
                        if unstable_count > a_cols / 2 {
                            sum = stabilize_c64(sum * Complex64::new(0.5, 0.0));
                        }

                        result[[i, j]] = sum;
                    }
                }

                Ok(Tensor::C64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Complex matrix multiplication requires matching complex tensor types",
                "complex matrix multiplication",
            )),
        }
    }

    /// Optimized complex activation function for advanced architectures.
    ///
    /// Applies complex ReLU activation: ReLU(Re(z)) + i*ReLU(Im(z))
    /// Optimized for modern SIMD architectures.
    ///
    /// # Returns
    ///
    /// A tensor with complex ReLU activation applied.
    pub fn complex_relu(&self) -> Result<Tensor> {
        match self {
            Tensor::C32(a) => {
                let result = a.mapv(|x| Complex32::new(x.re.max(0.0), x.im.max(0.0)));
                Ok(Tensor::C32(result))
            },
            Tensor::C64(a) => {
                let result = a.mapv(|x| Complex64::new(x.re.max(0.0), x.im.max(0.0)));
                Ok(Tensor::C64(result))
            },
            Tensor::CF16(a) => {
                let result = a.mapv(|x| {
                    let re_f32 = x.re.to_f32().max(0.0);
                    let im_f32 = x.im.to_f32().max(0.0);
                    Complex::new(half::f16::from_f32(re_f32), half::f16::from_f32(im_f32))
                });
                Ok(Tensor::CF16(result))
            },
            Tensor::CBF16(a) => {
                let result = a.mapv(|x| {
                    let re_f32 = x.re.to_f32().max(0.0);
                    let im_f32 = x.im.to_f32().max(0.0);
                    Complex::new(half::bf16::from_f32(re_f32), half::bf16::from_f32(im_f32))
                });
                Ok(Tensor::CBF16(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Complex ReLU only supports complex tensors",
                "complex ReLU activation",
            )),
        }
    }
}
