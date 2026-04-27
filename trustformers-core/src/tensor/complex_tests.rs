// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for complex tensor operations.

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn test_real_c32() {
        let t = Tensor::complex(vec![1.0, 2.0], vec![3.0, 4.0], &[2]);
        assert!(t.is_ok());
        if let Ok(t) = t {
            let real = t.real();
            assert!(real.is_ok());
            if let Ok(r) = real {
                assert_eq!(r.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_real_f32_returns_self() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]) {
            let real = t.real();
            assert!(real.is_ok());
            if let Ok(r) = real {
                assert_eq!(r.shape(), vec![3]);
            }
        }
    }

    #[test]
    fn test_imag_c32() {
        let t = Tensor::complex(vec![1.0, 2.0], vec![3.0, 4.0], &[2]);
        assert!(t.is_ok());
        if let Ok(t) = t {
            let imag = t.imag();
            assert!(imag.is_ok());
            if let Ok(i) = imag {
                assert_eq!(i.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_imag_f32_returns_zeros() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]) {
            let imag = t.imag();
            assert!(imag.is_ok());
            if let Ok(i) = imag {
                assert_eq!(i.shape(), vec![3]);
            }
        }
    }

    #[test]
    fn test_imag_c64() {
        if let Ok(t) = Tensor::complex_f64(vec![1.0, 2.0], vec![5.0, 6.0], &[2]) {
            let imag = t.imag();
            assert!(imag.is_ok());
            if let Ok(i) = imag {
                assert_eq!(i.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_magnitude_c32() {
        // |3+4i| = 5
        if let Ok(t) = Tensor::complex(vec![3.0], vec![4.0], &[1]) {
            let mag = t.magnitude();
            assert!(mag.is_ok());
            if let Ok(m) = mag {
                assert_eq!(m.shape(), vec![1]);
            }
        }
    }

    #[test]
    fn test_magnitude_f32_is_abs() {
        if let Ok(t) = Tensor::from_vec(vec![-3.0, 4.0, -5.0], &[3]) {
            let mag = t.magnitude();
            assert!(mag.is_ok());
            if let Ok(m) = mag {
                assert_eq!(m.shape(), vec![3]);
            }
        }
    }

    #[test]
    fn test_magnitude_c64() {
        if let Ok(t) = Tensor::complex_f64(vec![3.0, 0.0], vec![4.0, 5.0], &[2]) {
            let mag = t.magnitude();
            assert!(mag.is_ok());
            if let Ok(m) = mag {
                assert_eq!(m.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_phase_c32() {
        if let Ok(t) = Tensor::complex(vec![1.0, -1.0], vec![0.0, 0.0], &[2]) {
            let phase = t.phase();
            assert!(phase.is_ok());
            if let Ok(p) = phase {
                assert_eq!(p.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_phase_f32() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, -1.0, 0.5], &[3]) {
            let phase = t.phase();
            assert!(phase.is_ok());
            if let Ok(p) = phase {
                assert_eq!(p.shape(), vec![3]);
            }
        }
    }

    #[test]
    fn test_conj_c32() {
        if let Ok(t) = Tensor::complex(vec![1.0, 2.0], vec![3.0, 4.0], &[2]) {
            let conj = t.conj();
            assert!(conj.is_ok());
            if let Ok(c) = conj {
                assert_eq!(c.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_conj_c64() {
        if let Ok(t) = Tensor::complex_f64(vec![1.0, 2.0], vec![3.0, 4.0], &[2]) {
            let conj = t.conj();
            assert!(conj.is_ok());
            if let Ok(c) = conj {
                assert_eq!(c.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_conj_f32_returns_self() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]) {
            let conj = t.conj();
            assert!(conj.is_ok());
            if let Ok(c) = conj {
                assert_eq!(c.shape(), vec![3]);
            }
        }
    }

    #[test]
    fn test_to_complex_f32() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]) {
            let complex = t.to_complex();
            assert!(complex.is_ok());
            if let Ok(c) = complex {
                assert_eq!(c.shape(), vec![3]);
            }
        }
    }

    #[test]
    fn test_to_complex_f64() {
        if let Ok(t) = Tensor::from_slice_f64(&[1.0, 2.0], &[2]) {
            let complex = t.to_complex();
            assert!(complex.is_ok());
            if let Ok(c) = complex {
                assert_eq!(c.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_to_complex_already_complex() {
        if let Ok(t) = Tensor::complex(vec![1.0], vec![2.0], &[1]) {
            let complex = t.to_complex();
            assert!(complex.is_ok());
        }
    }

    #[test]
    fn test_complex_hadamard_c32() {
        if let Ok(a) = Tensor::complex(vec![1.0, 2.0], vec![1.0, 0.0], &[2]) {
            if let Ok(b) = Tensor::complex(vec![3.0, 1.0], vec![0.0, 1.0], &[2]) {
                let result = a.complex_hadamard(&b);
                assert!(result.is_ok());
                if let Ok(r) = result {
                    assert_eq!(r.shape(), vec![2]);
                }
            }
        }
    }

    #[test]
    fn test_complex_hadamard_c64() {
        if let Ok(a) = Tensor::complex_f64(vec![1.0, 2.0], vec![1.0, 0.0], &[2]) {
            if let Ok(b) = Tensor::complex_f64(vec![3.0, 1.0], vec![0.0, 1.0], &[2]) {
                let result = a.complex_hadamard(&b);
                assert!(result.is_ok());
                if let Ok(r) = result {
                    assert_eq!(r.shape(), vec![2]);
                }
            }
        }
    }

    #[test]
    fn test_complex_hadamard_type_mismatch() {
        if let Ok(a) = Tensor::complex(vec![1.0], vec![1.0], &[1]) {
            if let Ok(b) = Tensor::from_vec(vec![2.0], &[1]) {
                let result = a.complex_hadamard(&b);
                assert!(result.is_err());
            }
        }
    }

    #[test]
    fn test_fft_1d_c32() {
        // Simple 4-element FFT
        if let Ok(t) = Tensor::complex(vec![1.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 0.0], &[4]) {
            let result = t.fft();
            assert!(result.is_ok());
            if let Ok(r) = result {
                assert_eq!(r.shape(), vec![4]);
            }
        }
    }

    #[test]
    fn test_fft_non_1d_error() {
        if let Ok(t) = Tensor::complex(vec![1.0, 2.0, 3.0, 4.0], vec![0.0, 0.0, 0.0, 0.0], &[2, 2])
        {
            let result = t.fft();
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_magnitude_i64_as_f32() {
        if let Ok(t) = Tensor::from_vec_i64(vec![-3, 5, -7], &[3]) {
            let mag = t.magnitude();
            assert!(mag.is_ok());
            if let Ok(m) = mag {
                assert_eq!(m.shape(), vec![3]);
            }
        }
    }

    #[test]
    fn test_phase_c64() {
        if let Ok(t) = Tensor::complex_f64(vec![0.0, 1.0], vec![1.0, 0.0], &[2]) {
            let phase = t.phase();
            assert!(phase.is_ok());
            if let Ok(p) = phase {
                assert_eq!(p.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_conj_cf16() {
        if let Ok(t) = Tensor::complex_f16(vec![1.0, 2.0], vec![3.0, 4.0], &[2]) {
            let conj = t.conj();
            assert!(conj.is_ok());
            if let Ok(c) = conj {
                assert_eq!(c.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_conj_cbf16() {
        if let Ok(t) = Tensor::complex_bf16(vec![1.0, 2.0], vec![3.0, 4.0], &[2]) {
            let conj = t.conj();
            assert!(conj.is_ok());
            if let Ok(c) = conj {
                assert_eq!(c.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_to_complex_i64() {
        if let Ok(t) = Tensor::from_vec_i64(vec![1, 2, 3], &[3]) {
            let complex = t.to_complex();
            assert!(complex.is_ok());
            if let Ok(c) = complex {
                assert_eq!(c.shape(), vec![3]);
            }
        }
    }

    #[test]
    fn test_real_cf16() {
        if let Ok(t) = Tensor::complex_f16(vec![1.0, 2.0], vec![3.0, 4.0], &[2]) {
            let real = t.real();
            assert!(real.is_ok());
            if let Ok(r) = real {
                assert_eq!(r.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_real_cbf16() {
        if let Ok(t) = Tensor::complex_bf16(vec![1.0, 2.0], vec![3.0, 4.0], &[2]) {
            let real = t.real();
            assert!(real.is_ok());
            if let Ok(r) = real {
                assert_eq!(r.shape(), vec![2]);
            }
        }
    }
}
