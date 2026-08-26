//! Complex number tensor operations with numerical stability enhancements.
//!
//! This module contains functions for working with complex-valued tensors with
//! advanced numerical stability features including overflow/underflow protection,
//! NaN/infinity detection, and optimized algorithms for modern architectures.

use super::Tensor;
use crate::errors::{Result, TrustformersError};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use scirs2_core::{Complex, Complex32, Complex64};

/// Numerical stability constants for complex operations.
///
/// Only overflow guards remain: the former `STABILITY_EPSILON_*` underflow
/// thresholds were removed together with the predicates that used them, because
/// gradual underflow towards zero is well-defined IEEE-754 behaviour and not a
/// hazard for the operations in this module.
const MAX_SAFE_MAGNITUDE_F32: f32 = 1e30;
const MAX_SAFE_MAGNITUDE_F64: f64 = 1e300;

/// Check if a complex number is numerically stable.
///
/// A value is stable when both components are finite and its magnitude is below
/// [`MAX_SAFE_MAGNITUDE_F32`]. Small magnitudes -- including exact zero -- are
/// **not** instabilities: the previous predicate required
/// `norm() > STABILITY_EPSILON_F32`, so `0 + 0i` was reported as unstable and
/// every consumer (notably the FFT, which skipped "unstable" inputs) silently
/// dropped zeros from its sums.
fn is_stable_c32(z: Complex32) -> bool {
    z.re.is_finite() && z.im.is_finite() && z.norm() < MAX_SAFE_MAGNITUDE_F32
}

/// Check if a complex number is numerically stable (64-bit version).
///
/// See [`is_stable_c32`]; underflow towards zero is not treated as unstable.
fn is_stable_c64(z: Complex64) -> bool {
    z.re.is_finite() && z.im.is_finite() && z.norm() < MAX_SAFE_MAGNITUDE_F64
}

/// Stabilize a complex number by clamping unsafely large magnitudes.
///
/// Non-finite components become `0 + 0i`; magnitudes above
/// [`MAX_SAFE_MAGNITUDE_F32`] are scaled down to it. Small magnitudes are left
/// untouched (they used to be inflated to `STABILITY_EPSILON_F32`, a silent
/// change of the value).
fn stabilize_c32(z: Complex32) -> Complex32 {
    if !z.re.is_finite() || !z.im.is_finite() {
        return Complex32::new(0.0, 0.0);
    }
    let magnitude = z.norm();
    if magnitude > MAX_SAFE_MAGNITUDE_F32 {
        let scale = MAX_SAFE_MAGNITUDE_F32 / magnitude;
        Complex32::new(z.re * scale, z.im * scale)
    } else {
        z
    }
}

/// Stabilize a complex number by clamping unsafely large magnitudes (64-bit).
///
/// See [`stabilize_c32`] for the exact semantics.
fn stabilize_c64(z: Complex64) -> Complex64 {
    if !z.re.is_finite() || !z.im.is_finite() {
        return Complex64::new(0.0, 0.0);
    }
    let magnitude = z.norm();
    if magnitude > MAX_SAFE_MAGNITUDE_F64 {
        let scale = MAX_SAFE_MAGNITUDE_F64 / magnitude;
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
                        Complex::new(
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
                        Complex::new(
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

    /// Fast Fourier Transform (FFT) of a 1-D complex tensor.
    ///
    /// This is a genuine O(n log n) transform, not a renamed DFT:
    ///
    /// * power-of-two lengths use an iterative radix-2 Cooley-Tukey
    ///   decimation-in-time algorithm;
    /// * every other length uses **Bluestein's chirp-z algorithm**, which
    ///   expresses the DFT as a linear convolution evaluated with two
    ///   power-of-two FFTs, so no length falls back to the O(n^2) double loop.
    ///
    /// The transform is unnormalized (the forward convention
    /// `X[k] = sum_j x[j] e^(-2*pi*i*j*k/n)`), and all intermediate arithmetic is
    /// performed in `f64` regardless of the input precision.
    ///
    /// # Errors
    ///
    /// Returns an error for non-1-D tensors, empty tensors, and inputs that
    /// contain non-finite or unsafely large values. (The previous implementation
    /// silently *skipped* such entries, quietly returning the transform of a
    /// different signal, and applied its `1/sqrt(n)` scale factor only on the
    /// overflow branch, so the normalization depended on the data.)
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

                let mut buffer = Vec::with_capacity(n);
                for index in 0..n {
                    let value = a[[index]];
                    if !is_stable_c32(value) {
                        return Err(TrustformersError::tensor_op_error(
                            "FFT input contains non-finite or unsafely large values",
                            "complex FFT operation",
                        ));
                    }
                    buffer.push(Complex64::new(value.re as f64, value.im as f64));
                }

                fft_1d(&mut buffer)?;

                let mut result = ArrayD::zeros(IxDyn(&[n]));
                for (index, value) in buffer.iter().enumerate() {
                    result[[index]] = Complex32::new(value.re as f32, value.im as f32);
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

                let mut buffer = Vec::with_capacity(n);
                for index in 0..n {
                    let value = a[[index]];
                    if !is_stable_c64(value) {
                        return Err(TrustformersError::tensor_op_error(
                            "FFT input contains non-finite or unsafely large values",
                            "complex FFT operation",
                        ));
                    }
                    buffer.push(value);
                }

                fft_1d(&mut buffer)?;

                let mut result = ArrayD::zeros(IxDyn(&[n]));
                for (index, value) in buffer.iter().enumerate() {
                    result[[index]] = *value;
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

// ---------------------------------------------------------------------------
// FFT kernels
// ---------------------------------------------------------------------------

/// In-place forward DFT of `data`, dispatching on the length.
///
/// Power-of-two lengths use radix-2 Cooley-Tukey; all other lengths use
/// Bluestein's chirp-z algorithm. Both are O(n log n).
fn fft_1d(data: &mut Vec<Complex64>) -> Result<()> {
    let n = data.len();
    if n <= 1 {
        return Ok(());
    }
    if n.is_power_of_two() {
        fft_radix2_in_place(data);
        Ok(())
    } else {
        let transformed = fft_bluestein(data)?;
        data.clear();
        data.extend_from_slice(&transformed);
        Ok(())
    }
}

/// Iterative radix-2 decimation-in-time FFT (in place, unnormalized).
///
/// `data.len()` must be a power of two.
fn fft_radix2_in_place(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation.
    let mut target = 0usize;
    for source in 1..n {
        let mut bit = n >> 1;
        while target & bit != 0 {
            target ^= bit;
            bit >>= 1;
        }
        target |= bit;
        if source < target {
            data.swap(source, target);
        }
    }

    // Butterfly stages. Twiddles are computed directly from the angle rather
    // than by repeated multiplication, so error does not accumulate along a
    // stage.
    let mut span = 2usize;
    while span <= n {
        let half = span / 2;
        let base_angle = -2.0 * std::f64::consts::PI / span as f64;
        let mut offset = 0usize;
        while offset < n {
            for k in 0..half {
                let angle = base_angle * k as f64;
                let twiddle = Complex64::new(angle.cos(), angle.sin());
                let even = data[offset + k];
                let odd = data[offset + k + half] * twiddle;
                data[offset + k] = even + odd;
                data[offset + k + half] = even - odd;
            }
            offset += span;
        }
        span <<= 1;
    }
}

/// Inverse of [`fft_radix2_in_place`] (in place, normalized by `1/n`).
fn ifft_radix2_in_place(data: &mut [Complex64]) {
    for value in data.iter_mut() {
        *value = value.conj();
    }
    fft_radix2_in_place(data);
    let inverse_len = 1.0 / data.len() as f64;
    for value in data.iter_mut() {
        *value = value.conj() * inverse_len;
    }
}

/// Bluestein's chirp-z algorithm: DFT of an arbitrary length via convolution.
///
/// `X[k] = w^(k^2/2) * sum_j (x[j] * w^(j^2/2)) * w^(-(k-j)^2/2)` with
/// `w = e^(-2*pi*i/n)`; the sum is a linear convolution, evaluated with two
/// power-of-two FFTs of length `m >= 2n - 1`.
fn fft_bluestein(data: &[Complex64]) -> Result<Vec<Complex64>> {
    let n = data.len();
    let target = 2 * n - 1;
    let m = target.checked_next_power_of_two().ok_or_else(|| {
        TrustformersError::tensor_op_error(
            "FFT length is too large for the Bluestein convolution buffer",
            "complex FFT operation",
        )
    })?;

    // chirp[j] = e^(-i*pi*j^2/n); the exponent is reduced modulo 2n first so it
    // stays exact for large j.
    let modulus = 2u128 * n as u128;
    let chirp = |index: usize| -> Complex64 {
        let squared = (index as u128 * index as u128) % modulus;
        let angle = -std::f64::consts::PI * squared as f64 / n as f64;
        Complex64::new(angle.cos(), angle.sin())
    };

    let mut a = vec![Complex64::new(0.0, 0.0); m];
    let mut b = vec![Complex64::new(0.0, 0.0); m];
    for index in 0..n {
        let c = chirp(index);
        a[index] = data[index] * c;
        // b is the conjugate chirp, extended symmetrically so the cyclic
        // convolution of length m reproduces the linear one.
        let conjugate = c.conj();
        b[index] = conjugate;
        if index > 0 {
            b[m - index] = conjugate;
        }
    }

    fft_radix2_in_place(&mut a);
    fft_radix2_in_place(&mut b);
    for (a_value, b_value) in a.iter_mut().zip(b.iter()) {
        *a_value *= *b_value;
    }
    ifft_radix2_in_place(&mut a);

    Ok((0..n).map(|k| a[k] * chirp(k)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::Result;
    use crate::tensor::DType;

    #[test]
    fn test_is_stable_c32_normal() {
        let z = Complex32::new(1.0, 2.0);
        assert!(is_stable_c32(z));
    }

    #[test]
    fn test_is_stable_c32_nan() {
        let z = Complex32::new(f32::NAN, 0.0);
        assert!(!is_stable_c32(z));
    }

    #[test]
    fn test_is_stable_c32_inf() {
        let z = Complex32::new(f32::INFINITY, 0.0);
        assert!(!is_stable_c32(z));
    }

    #[test]
    fn test_is_stable_c64_normal() {
        let z = Complex64::new(3.0, 4.0);
        assert!(is_stable_c64(z));
    }

    #[test]
    fn test_is_stable_c64_nan() {
        let z = Complex64::new(0.0, f64::NAN);
        assert!(!is_stable_c64(z));
    }

    #[test]
    fn test_stabilize_c32_nan_to_zero() {
        let z = Complex32::new(f32::NAN, f32::NAN);
        let s = stabilize_c32(z);
        assert_eq!(s.re, 0.0);
        assert_eq!(s.im, 0.0);
    }

    #[test]
    fn test_stabilize_c32_normal_unchanged() {
        let z = Complex32::new(1.0, 2.0);
        let s = stabilize_c32(z);
        assert!((s.re - 1.0).abs() < 1e-6);
        assert!((s.im - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_stabilize_c64_nan_to_zero() {
        let z = Complex64::new(f64::INFINITY, 0.0);
        let s = stabilize_c64(z);
        assert_eq!(s.re, 0.0);
        assert_eq!(s.im, 0.0);
    }

    #[test]
    fn test_stabilize_c64_normal_unchanged() {
        let z = Complex64::new(5.0, 3.0);
        let s = stabilize_c64(z);
        assert!((s.re - 5.0).abs() < 1e-10);
        assert!((s.im - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_real_part_c32() -> Result<()> {
        let t = Tensor::complex(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], &[3])?;
        let real = t.real()?;
        assert_eq!(real.dtype(), DType::F32);
        let data = real.data()?;
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_imag_part_c32() -> Result<()> {
        let t = Tensor::complex(vec![1.0, 2.0], vec![3.0, 4.0], &[2])?;
        let imag = t.imag()?;
        assert_eq!(imag.dtype(), DType::F32);
        let data = imag.data()?;
        assert!((data[0] - 3.0).abs() < 1e-6);
        assert!((data[1] - 4.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_real_part_of_real_tensor() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0], &[2])?;
        let real = t.real()?;
        assert_eq!(real.dtype(), DType::F32);
        Ok(())
    }

    #[test]
    fn test_magnitude_c32() -> Result<()> {
        // 3+4i has magnitude 5
        let t = Tensor::complex(vec![3.0], vec![4.0], &[1])?;
        let mag = t.magnitude()?;
        let data = mag.data()?;
        assert!((data[0] - 5.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_magnitude_zero() -> Result<()> {
        let t = Tensor::complex(vec![0.0], vec![0.0], &[1])?;
        let mag = t.magnitude()?;
        let data = mag.data()?;
        assert!(data[0].abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_phase_c32() -> Result<()> {
        // 1+0i has phase 0
        let t = Tensor::complex(vec![1.0], vec![0.0], &[1])?;
        let phase = t.phase()?;
        let data = phase.data()?;
        assert!(data[0].abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_conj_c32() -> Result<()> {
        let t = Tensor::complex(vec![1.0, 2.0], vec![3.0, 4.0], &[2])?;
        let conj = t.conj()?;
        let imag = conj.imag()?;
        let data = imag.data()?;
        assert!((data[0] - (-3.0)).abs() < 1e-6);
        assert!((data[1] - (-4.0)).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_to_complex_from_f32() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3])?;
        let c = t.to_complex()?;
        assert_eq!(c.dtype(), DType::C32);
        assert_eq!(c.shape(), vec![3]);
        // Imaginary parts should be zero
        let imag = c.imag()?;
        let data = imag.data()?;
        for val in &data {
            assert!(val.abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_complex_hadamard() -> Result<()> {
        let a = Tensor::complex(vec![1.0, 2.0], vec![0.0, 0.0], &[2])?;
        let b = Tensor::complex(vec![3.0, 4.0], vec![0.0, 0.0], &[2])?;
        let result = a.complex_hadamard(&b)?;
        let real = result.real()?;
        let data = real.data()?;
        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 8.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_complex_relu_positive_real() -> Result<()> {
        let t = Tensor::complex(vec![1.0, -1.0], vec![2.0, 3.0], &[2])?;
        let result = t.complex_relu()?;
        let real = result.real()?;
        let data = real.data()?;
        // Positive real stays
        assert!((data[0] - 1.0).abs() < 1e-5);
        // Negative real -> 0
        assert!(data[1].abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_complex_c64_real_imag() -> Result<()> {
        let t = Tensor::complex_f64(vec![1.0, 2.0], vec![3.0, 4.0], &[2])?;
        let real = t.real()?;
        assert_eq!(real.dtype(), DType::F64);
        let imag = t.imag()?;
        assert_eq!(imag.dtype(), DType::F64);
        Ok(())
    }

    #[test]
    fn test_conj_of_real_is_itself() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0], &[2])?;
        // conjugate of real number is itself
        let conj = t.conj()?;
        let data = conj.data()?;
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_magnitude_c64() -> Result<()> {
        // 3+4i -> magnitude 5
        let t = Tensor::complex_f64(vec![3.0], vec![4.0], &[1])?;
        let mag = t.magnitude()?;
        assert_eq!(mag.dtype(), DType::F64);
        Ok(())
    }

    #[test]
    fn test_complex_2d() -> Result<()> {
        let t = Tensor::complex(vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
        assert_eq!(t.shape(), vec![2, 2]);
        let real = t.real()?;
        assert_eq!(real.shape(), vec![2, 2]);
        Ok(())
    }

    /// Naive O(n^2) DFT, used only as the reference the fast path is checked
    /// against.
    fn reference_dft(values: &[Complex64]) -> Vec<Complex64> {
        let n = values.len();
        (0..n)
            .map(|k| {
                let mut sum = Complex64::new(0.0, 0.0);
                for (j, value) in values.iter().enumerate() {
                    let angle = -2.0 * std::f64::consts::PI * (k as f64) * (j as f64) / n as f64;
                    sum += value * Complex64::new(angle.cos(), angle.sin());
                }
                sum
            })
            .collect()
    }

    fn deterministic_signal(n: usize) -> Vec<Complex64> {
        (0..n)
            .map(|i| {
                Complex64::new(
                    (i as f64 * 0.37).sin() + 0.25 * i as f64 / n as f64,
                    (i as f64 * 0.11).cos() - 0.1,
                )
            })
            .collect()
    }

    /// The FFT must agree with the naive DFT for power-of-two **and**
    /// non-power-of-two lengths (the latter go through Bluestein).
    #[test]
    fn test_fft_matches_naive_dft() {
        for &n in &[1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16, 17, 31, 32, 60, 64] {
            let signal = deterministic_signal(n);
            let expected = reference_dft(&signal);

            let mut array = ArrayD::zeros(IxDyn(&[n]));
            for (index, value) in signal.iter().enumerate() {
                array[[index]] = *value;
            }
            let transformed = Tensor::C64(array).fft().expect("fft succeeds");
            let Tensor::C64(output) = transformed else {
                panic!("FFT of a C64 tensor must stay C64");
            };

            for k in 0..n {
                let got = output[[k]];
                let want = expected[k];
                let tolerance = 1e-9 * (1.0 + want.norm()) * (n as f64);
                assert!(
                    (got.re - want.re).abs() < tolerance && (got.im - want.im).abs() < tolerance,
                    "n = {n}, k = {k}: got {got:?}, expected {want:?}"
                );
            }
        }
    }

    /// The DFT of a pure unit impulse is a constant 1 across all frequencies;
    /// the DFT of a constant signal is an impulse of magnitude n at k = 0.
    #[test]
    fn test_fft_known_closed_forms() {
        let n = 12usize;

        // Impulse at index 0.
        let mut impulse = ArrayD::zeros(IxDyn(&[n]));
        impulse[[0]] = Complex64::new(1.0, 0.0);
        let Tensor::C64(spectrum) = Tensor::C64(impulse).fft().expect("fft succeeds") else {
            panic!("unexpected dtype");
        };
        for k in 0..n {
            assert!((spectrum[[k]].re - 1.0).abs() < 1e-9);
            assert!(spectrum[[k]].im.abs() < 1e-9);
        }

        // Constant signal.
        let mut constant = ArrayD::zeros(IxDyn(&[n]));
        for k in 0..n {
            constant[[k]] = Complex64::new(2.0, 0.0);
        }
        let Tensor::C64(spectrum) = Tensor::C64(constant).fft().expect("fft succeeds") else {
            panic!("unexpected dtype");
        };
        assert!((spectrum[[0]].re - 2.0 * n as f64).abs() < 1e-8);
        for k in 1..n {
            assert!(spectrum[[k]].norm() < 1e-8, "bin {k} = {:?}", spectrum[[k]]);
        }
    }

    /// Non-finite input must be reported, not silently dropped from the sum.
    #[test]
    fn test_fft_rejects_non_finite_input() {
        let mut array = ArrayD::zeros(IxDyn(&[4]));
        array[[0]] = Complex64::new(1.0, 0.0);
        array[[1]] = Complex64::new(f64::NAN, 0.0);
        assert!(Tensor::C64(array).fft().is_err());
    }

    /// The C32 path must agree with the C64 path within f32 precision.
    #[test]
    fn test_fft_c32_matches_c64() {
        let n = 16usize;
        let signal = deterministic_signal(n);

        let mut array64 = ArrayD::zeros(IxDyn(&[n]));
        let mut array32 = ArrayD::zeros(IxDyn(&[n]));
        for (index, value) in signal.iter().enumerate() {
            array64[[index]] = *value;
            array32[[index]] = Complex32::new(value.re as f32, value.im as f32);
        }

        let Tensor::C64(expected) = Tensor::C64(array64).fft().expect("fft succeeds") else {
            panic!("unexpected dtype");
        };
        let Tensor::C32(got) = Tensor::C32(array32).fft().expect("fft succeeds") else {
            panic!("unexpected dtype");
        };

        for k in 0..n {
            assert!((got[[k]].re as f64 - expected[[k]].re).abs() < 1e-4);
            assert!((got[[k]].im as f64 - expected[[k]].im).abs() < 1e-4);
        }
    }
}
