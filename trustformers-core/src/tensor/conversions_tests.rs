/// Tests for tensor type-conversion operations (to_dtype, to_vec_f32, to_vec_u8,
/// to_f32, to_i64).
#[cfg(test)]
mod conversions_tests {
    use crate::tensor::{DType, Tensor};
    use scirs2_core::{Complex32, Complex64};

    // ── helpers ───────────────────────────────────────────────────────────────

    fn f32_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        match Tensor::with_shape(data, shape) {
            Ok(t) => t,
            Err(e) => panic!("failed to build F32 tensor: {}", e),
        }
    }

    fn i64_tensor(data: Vec<i64>, shape: Vec<usize>) -> Tensor {
        match Tensor::from_vec_i64(data, &shape) {
            Ok(t) => t,
            Err(e) => panic!("failed to build I64 tensor: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // to_dtype: F32 → various targets
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_f32_to_f64_dtype() {
        let t = f32_tensor(vec![1.5, 2.5, 3.5, 4.5], vec![4]);
        match t.to_dtype(DType::F64) {
            Ok(result) => {
                assert_eq!(result.dtype(), DType::F64);
                assert_eq!(result.shape(), vec![4]);
                // Convert back and verify values
                match result.to_dtype(DType::F32) {
                    Ok(back) => {
                        match back.to_vec_f32() {
                            Ok(v) => {
                                for (&a, &b) in v.iter().zip([1.5f32, 2.5, 3.5, 4.5].iter()) {
                                    assert!((a - b).abs() < 1e-5, "{} ≠ {}", a, b);
                                }
                            },
                            Err(e) => panic!("to_vec_f32 failed: {}", e),
                        }
                    },
                    Err(e) => panic!("F64→F32 back failed: {}", e),
                }
            },
            Err(e) => panic!("F32→F64 failed: {}", e),
        }
    }

    #[test]
    fn test_f32_to_i64_dtype() {
        let t = f32_tensor(vec![1.9, 2.1, 3.7, 4.0], vec![4]);
        match t.to_dtype(DType::I64) {
            Ok(result) => {
                assert_eq!(result.dtype(), DType::I64);
                // Cast truncates: [1, 2, 3, 4]
                match result.to_vec_f32() {
                    Ok(v) => {
                        for (&got, &expected) in v.iter().zip([1.0f32, 2.0, 3.0, 4.0].iter()) {
                            assert!((got - expected).abs() < 0.5, "{} ≠ {}", got, expected);
                        }
                    },
                    Err(e) => panic!("I64 to_vec_f32 failed: {}", e),
                }
            },
            Err(e) => panic!("F32→I64 failed: {}", e),
        }
    }

    #[test]
    fn test_f32_to_c32_dtype() {
        let t = f32_tensor(vec![3.0, 4.0], vec![2]);
        match t.to_dtype(DType::C32) {
            Ok(result) => {
                assert_eq!(result.dtype(), DType::C32);
                assert_eq!(result.shape(), vec![2]);
            },
            Err(e) => panic!("F32→C32 failed: {}", e),
        }
    }

    #[test]
    fn test_f32_to_c64_dtype() {
        let t = f32_tensor(vec![3.0, 4.0], vec![2]);
        match t.to_dtype(DType::C64) {
            Ok(result) => {
                assert_eq!(result.dtype(), DType::C64);
                assert_eq!(result.shape(), vec![2]);
            },
            Err(e) => panic!("F32→C64 failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // to_dtype: F64 → various targets
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_f64_to_f32_dtype() {
        // Build via F64 path: F32→F64, then F64→F32
        let t = f32_tensor(vec![1.0, 2.0, 3.0], vec![3]);
        match t.to_dtype(DType::F64) {
            Ok(f64_t) => match f64_t.to_dtype(DType::F32) {
                Ok(back) => {
                    assert_eq!(back.dtype(), DType::F32);
                    match back.to_vec_f32() {
                        Ok(v) => assert_eq!(v, vec![1.0f32, 2.0, 3.0]),
                        Err(e) => panic!("to_vec_f32 failed: {}", e),
                    }
                },
                Err(e) => panic!("F64→F32 failed: {}", e),
            },
            Err(e) => panic!("F32→F64 setup failed: {}", e),
        }
    }

    #[test]
    fn test_f64_to_i64_dtype() {
        let t = f32_tensor(vec![10.0, 20.0, 30.0], vec![3]);
        match t.to_dtype(DType::F64) {
            Ok(f64_t) => match f64_t.to_dtype(DType::I64) {
                Ok(result) => {
                    assert_eq!(result.dtype(), DType::I64);
                },
                Err(e) => panic!("F64→I64 failed: {}", e),
            },
            Err(e) => panic!("F32→F64 setup failed: {}", e),
        }
    }

    #[test]
    fn test_f64_to_c32_dtype() {
        let t = f32_tensor(vec![1.0, 2.0], vec![2]);
        match t.to_dtype(DType::F64) {
            Ok(f64_t) => match f64_t.to_dtype(DType::C32) {
                Ok(result) => assert_eq!(result.dtype(), DType::C32),
                Err(e) => panic!("F64→C32 failed: {}", e),
            },
            Err(e) => panic!("F32→F64 setup failed: {}", e),
        }
    }

    #[test]
    fn test_f64_to_c64_dtype() {
        let t = f32_tensor(vec![1.0, 2.0], vec![2]);
        match t.to_dtype(DType::F64) {
            Ok(f64_t) => match f64_t.to_dtype(DType::C64) {
                Ok(result) => assert_eq!(result.dtype(), DType::C64),
                Err(e) => panic!("F64→C64 failed: {}", e),
            },
            Err(e) => panic!("F32→F64 setup failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // to_dtype: I64 → various targets
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_i64_to_f32_dtype() {
        let t = i64_tensor(vec![5, 10, 15], vec![3]);
        match t.to_dtype(DType::F32) {
            Ok(result) => {
                assert_eq!(result.dtype(), DType::F32);
                match result.to_vec_f32() {
                    Ok(v) => assert_eq!(v, vec![5.0f32, 10.0, 15.0]),
                    Err(e) => panic!("to_vec_f32 after I64→F32 failed: {}", e),
                }
            },
            Err(e) => panic!("I64→F32 failed: {}", e),
        }
    }

    #[test]
    fn test_i64_to_f64_dtype() {
        let t = i64_tensor(vec![100, 200], vec![2]);
        match t.to_dtype(DType::F64) {
            Ok(result) => assert_eq!(result.dtype(), DType::F64),
            Err(e) => panic!("I64→F64 failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // to_dtype: C32 → real extraction
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_c32_to_f32_extracts_real_part() {
        // Build a C32 tensor by converting F32→C32 (imaginary = 0).
        let t = f32_tensor(vec![3.0, 4.0, 5.0], vec![3]);
        match t.to_dtype(DType::C32) {
            Ok(c32_t) => match c32_t.to_dtype(DType::F32) {
                Ok(result) => {
                    assert_eq!(result.dtype(), DType::F32);
                    match result.to_vec_f32() {
                        Ok(v) => {
                            for (&got, &expected) in v.iter().zip([3.0f32, 4.0, 5.0].iter()) {
                                assert!((got - expected).abs() < 1e-5, "{} ≠ {}", got, expected);
                            }
                        },
                        Err(e) => panic!("to_vec_f32 after C32→F32 failed: {}", e),
                    }
                },
                Err(e) => panic!("C32→F32 failed: {}", e),
            },
            Err(e) => panic!("F32→C32 setup failed: {}", e),
        }
    }

    #[test]
    fn test_c32_to_f64_extracts_real_part() {
        let t = f32_tensor(vec![2.0, 3.0], vec![2]);
        match t.to_dtype(DType::C32) {
            Ok(c32_t) => match c32_t.to_dtype(DType::F64) {
                Ok(result) => assert_eq!(result.dtype(), DType::F64),
                Err(e) => panic!("C32→F64 failed: {}", e),
            },
            Err(e) => panic!("F32→C32 failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // to_dtype: C64 → real extraction
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_c64_to_f32_extracts_real_part() {
        let t = f32_tensor(vec![7.0, 8.0], vec![2]);
        match t.to_dtype(DType::C64) {
            Ok(c64_t) => match c64_t.to_dtype(DType::F32) {
                Ok(result) => {
                    assert_eq!(result.dtype(), DType::F32);
                    match result.to_vec_f32() {
                        Ok(v) => {
                            assert!((v[0] - 7.0f32).abs() < 1e-5);
                            assert!((v[1] - 8.0f32).abs() < 1e-5);
                        },
                        Err(e) => panic!("to_vec_f32 failed: {}", e),
                    }
                },
                Err(e) => panic!("C64→F32 failed: {}", e),
            },
            Err(e) => panic!("F32→C64 setup failed: {}", e),
        }
    }

    #[test]
    fn test_c64_to_f64_extracts_real_part() {
        let t = f32_tensor(vec![1.0, 2.0], vec![2]);
        match t.to_dtype(DType::C64) {
            Ok(c64_t) => match c64_t.to_dtype(DType::F64) {
                Ok(result) => assert_eq!(result.dtype(), DType::F64),
                Err(e) => panic!("C64→F64 failed: {}", e),
            },
            Err(e) => panic!("F32→C64 failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // to_dtype: same-dtype is identity
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_same_dtype_is_identity() {
        let data = vec![1.0f32, 2.0, 3.0];
        let t = f32_tensor(data.clone(), vec![3]);
        match t.to_dtype(DType::F32) {
            Ok(result) => {
                assert_eq!(result.dtype(), DType::F32);
                match result.to_vec_f32() {
                    Ok(v) => assert_eq!(v, data),
                    Err(e) => panic!("to_vec_f32 on same-dtype failed: {}", e),
                }
            },
            Err(e) => panic!("F32→F32 identity failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // to_vec_f32
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_to_vec_f32_from_f32_tensor() {
        let data = vec![10.0f32, 20.0, 30.0];
        let t = f32_tensor(data.clone(), vec![3]);
        match t.to_vec_f32() {
            Ok(v) => assert_eq!(v, data),
            Err(e) => panic!("to_vec_f32 from F32 failed: {}", e),
        }
    }

    #[test]
    fn test_to_vec_f32_from_f64_tensor() {
        let t = f32_tensor(vec![1.0, 2.0, 3.0], vec![3]);
        match t.to_dtype(DType::F64) {
            Ok(f64_t) => match f64_t.to_vec_f32() {
                Ok(v) => {
                    for (&got, &exp) in v.iter().zip([1.0f32, 2.0, 3.0].iter()) {
                        assert!((got - exp).abs() < 1e-5);
                    }
                },
                Err(e) => panic!("to_vec_f32 from F64 failed: {}", e),
            },
            Err(e) => panic!("F32→F64 failed: {}", e),
        }
    }

    #[test]
    fn test_to_vec_f32_from_i64_tensor() {
        let t = i64_tensor(vec![1, 2, 3], vec![3]);
        match t.to_vec_f32() {
            Ok(v) => assert_eq!(v, vec![1.0f32, 2.0, 3.0]),
            Err(e) => panic!("to_vec_f32 from I64 failed: {}", e),
        }
    }

    #[test]
    fn test_to_vec_f32_from_c32_tensor() {
        let t = f32_tensor(vec![5.0, 6.0], vec![2]);
        match t.to_dtype(DType::C32) {
            Ok(c32_t) => match c32_t.to_vec_f32() {
                Ok(v) => {
                    assert!((v[0] - 5.0f32).abs() < 1e-5);
                    assert!((v[1] - 6.0f32).abs() < 1e-5);
                },
                Err(e) => panic!("to_vec_f32 from C32 failed: {}", e),
            },
            Err(e) => panic!("F32→C32 failed: {}", e),
        }
    }

    #[test]
    fn test_to_vec_f32_preserves_2d_order() {
        // Row-major order must be preserved.
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = f32_tensor(data.clone(), vec![2, 3]);
        match t.to_vec_f32() {
            Ok(v) => assert_eq!(v, data, "2D element order must be row-major"),
            Err(e) => panic!("to_vec_f32 2D failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // to_vec_u8
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_to_vec_u8_from_f32() {
        let t = f32_tensor(vec![0.0, 1.0, 200.0, 255.0], vec![4]);
        match t.to_vec_u8() {
            Ok(v) => {
                assert_eq!(v[0], 0u8);
                assert_eq!(v[1], 1u8);
                assert_eq!(v[2], 200u8);
                assert_eq!(v[3], 255u8);
            },
            Err(e) => panic!("to_vec_u8 from F32 failed: {}", e),
        }
    }

    #[test]
    fn test_to_vec_u8_from_i64() {
        let t = i64_tensor(vec![0, 100, 255], vec![3]);
        match t.to_vec_u8() {
            Ok(v) => {
                assert_eq!(v[0], 0u8);
                assert_eq!(v[1], 100u8);
                assert_eq!(v[2], 255u8);
            },
            Err(e) => panic!("to_vec_u8 from I64 failed: {}", e),
        }
    }

    #[test]
    fn test_to_vec_u8_from_f64() {
        let t = f32_tensor(vec![10.0, 20.0], vec![2]);
        match t.to_dtype(DType::F64) {
            Ok(f64_t) => match f64_t.to_vec_u8() {
                Ok(v) => {
                    assert_eq!(v[0], 10u8);
                    assert_eq!(v[1], 20u8);
                },
                Err(e) => panic!("to_vec_u8 from F64 failed: {}", e),
            },
            Err(e) => panic!("F32→F64 for u8 test failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // to_f32 convenience method
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_to_f32_from_i64() {
        let t = i64_tensor(vec![3, 6, 9], vec![3]);
        match t.to_f32() {
            Ok(result) => {
                assert_eq!(result.dtype(), DType::F32);
                match result.to_vec_f32() {
                    Ok(v) => assert_eq!(v, vec![3.0f32, 6.0, 9.0]),
                    Err(e) => panic!("to_vec_f32 after to_f32 failed: {}", e),
                }
            },
            Err(e) => panic!("to_f32 from I64 failed: {}", e),
        }
    }

    #[test]
    fn test_to_f32_from_f32_is_identity() {
        let data = vec![1.1f32, 2.2, 3.3];
        let t = f32_tensor(data.clone(), vec![3]);
        match t.to_f32() {
            Ok(result) => match result.to_vec_f32() {
                Ok(v) => {
                    for (&a, &b) in v.iter().zip(data.iter()) {
                        assert!((a - b).abs() < 1e-6);
                    }
                },
                Err(e) => panic!("to_vec_f32 after to_f32 (identity) failed: {}", e),
            },
            Err(e) => panic!("to_f32 on F32 failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // to_i64 convenience method
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_to_i64_from_f32() {
        let t = f32_tensor(vec![1.9, 2.1, 3.0], vec![3]);
        match t.to_i64() {
            Ok(result) => {
                assert_eq!(result.dtype(), DType::I64);
            },
            Err(e) => panic!("to_i64 from F32 failed: {}", e),
        }
    }

    #[test]
    fn test_to_i64_from_i64_is_identity() {
        let data = vec![10i64, 20, 30];
        let t = i64_tensor(data.clone(), vec![3]);
        match t.to_i64() {
            Ok(result) => {
                assert_eq!(result.dtype(), DType::I64);
                // Values should be unchanged
                match result.to_vec_f32() {
                    Ok(v) => {
                        for (&got, &exp) in v.iter().zip([10.0f32, 20.0, 30.0].iter()) {
                            assert!((got - exp).abs() < 0.5);
                        }
                    },
                    Err(e) => panic!("to_vec_f32 after to_i64 (identity) failed: {}", e),
                }
            },
            Err(e) => panic!("to_i64 on I64 failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // shape preservation across conversions
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dtype_conversion_preserves_shape() {
        let t = f32_tensor(vec![1.0; 12], vec![3, 4]);
        match t.to_dtype(DType::F64) {
            Ok(result) => {
                assert_eq!(result.shape(), vec![3, 4], "F32→F64 must preserve shape");
            },
            Err(e) => panic!("F32→F64 shape test failed: {}", e),
        }
    }

    #[test]
    fn test_dtype_conversion_3d_shape_preserved() {
        let t = f32_tensor(vec![1.0; 24], vec![2, 3, 4]);
        match t.to_dtype(DType::I64) {
            Ok(result) => {
                assert_eq!(result.shape(), vec![2, 3, 4], "3D shape must be preserved");
            },
            Err(e) => panic!("F32→I64 3D shape test failed: {}", e),
        }
    }
}
