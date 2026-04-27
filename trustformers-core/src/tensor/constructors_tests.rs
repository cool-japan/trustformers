// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for tensor constructor functions.

#[cfg(test)]
mod tests {
    use crate::tensor::{DType, Tensor};

    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }
        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1u64 << 53) as f32
        }
    }

    #[test]
    fn test_new_1d_tensor() {
        let result = Tensor::new(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![4]);
            assert_eq!(t.len(), 4);
        }
    }

    #[test]
    fn test_new_empty_tensor() {
        let result = Tensor::new(vec![]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![0]);
        }
    }

    #[test]
    fn test_with_shape_2d() {
        let result = Tensor::with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![2, 2]);
        }
    }

    #[test]
    fn test_with_shape_mismatch() {
        let result = Tensor::with_shape(vec![1.0, 2.0, 3.0], vec![2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_vec_i64() {
        let result = Tensor::from_vec_i64(vec![10, 20, 30, 40], &[2, 2]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![2, 2]);
        }
    }

    #[test]
    fn test_from_vec_i64_shape_mismatch() {
        let result = Tensor::from_vec_i64(vec![1, 2, 3], &[2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_zeros_various_shapes() {
        for shape in &[vec![3], vec![2, 3], vec![2, 3, 4]] {
            let result = Tensor::zeros(shape);
            assert!(result.is_ok());
            if let Ok(t) = result {
                assert_eq!(t.shape(), *shape);
            }
        }
    }

    #[test]
    fn test_ones_various_shapes() {
        for shape in &[vec![5], vec![3, 4], vec![2, 3, 5]] {
            let result = Tensor::ones(shape);
            assert!(result.is_ok());
            if let Ok(t) = result {
                assert_eq!(t.shape(), *shape);
            }
        }
    }

    #[test]
    fn test_randn_shape() {
        let result = Tensor::randn(&[4, 5]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![4, 5]);
        }
    }

    #[test]
    fn test_zeros_like() {
        if let Ok(input) = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]) {
            let result = Tensor::zeros_like(&input);
            assert!(result.is_ok());
            if let Ok(t) = result {
                assert_eq!(t.shape(), vec![2, 3]);
            }
        }
    }

    #[test]
    fn test_ones_like() {
        if let Ok(input) = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]) {
            let result = Tensor::ones_like(&input);
            assert!(result.is_ok());
            if let Ok(t) = result {
                assert_eq!(t.shape(), vec![2, 3]);
            }
        }
    }

    #[test]
    fn test_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = Tensor::from_data(data, &[2, 3]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![2, 3]);
        }
    }

    #[test]
    fn test_from_data_size_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        let result = Tensor::from_data(data, &[2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_slice() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let result = Tensor::from_slice(&data, &[2, 2]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![2, 2]);
        }
    }

    #[test]
    fn test_from_slice_size_mismatch() {
        let data = [1.0f32, 2.0, 3.0];
        let result = Tensor::from_slice(&data, &[2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_scalar() {
        let result = Tensor::scalar(42.0);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), Vec::<usize>::new());
            assert_eq!(t.len(), 1);
        }
    }

    #[test]
    fn test_eye_f32() {
        let result = Tensor::eye_f32(3);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![3, 3]);
        }
    }

    #[test]
    fn test_full() {
        let result = Tensor::full(5.0, vec![3, 4]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![3, 4]);
            assert_eq!(t.len(), 12);
        }
    }

    #[test]
    fn test_full_with_dtype_f32() {
        let result = Tensor::full_with_dtype(&[2, 3], 7.0, DType::F32);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![2, 3]);
        }
    }

    #[test]
    fn test_full_with_dtype_f64() {
        let result = Tensor::full_with_dtype(&[2, 3], 7.0, DType::F64);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![2, 3]);
        }
    }

    #[test]
    fn test_full_with_dtype_i64() {
        let result = Tensor::full_with_dtype(&[4], 3.0, DType::I64);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![4]);
        }
    }

    #[test]
    fn test_complex_constructor() {
        let real = vec![1.0, 2.0, 3.0, 4.0];
        let imag = vec![5.0, 6.0, 7.0, 8.0];
        let result = Tensor::complex(real, imag, &[2, 2]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![2, 2]);
        }
    }

    #[test]
    fn test_complex_mismatched_lengths() {
        let real = vec![1.0, 2.0];
        let imag = vec![3.0, 4.0, 5.0];
        let result = Tensor::complex(real, imag, &[2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_complex_f64_constructor() {
        let real = vec![1.0f64, 2.0];
        let imag = vec![3.0f64, 4.0];
        let result = Tensor::complex_f64(real, imag, &[2]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![2]);
        }
    }

    #[test]
    fn test_zeros_f64() {
        let result = Tensor::zeros_f64(&[3, 4]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![3, 4]);
        }
    }

    #[test]
    fn test_zeros_i64() {
        let result = Tensor::zeros_i64(&[5]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![5]);
        }
    }

    #[test]
    fn test_zeros_c32() {
        let result = Tensor::zeros_c32(&[2, 3]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![2, 3]);
        }
    }

    #[test]
    fn test_zeros_c64() {
        let result = Tensor::zeros_c64(&[4]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![4]);
        }
    }

    #[test]
    fn test_zeros_f16() {
        let result = Tensor::zeros_f16(&[2, 2]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![2, 2]);
        }
    }

    #[test]
    fn test_zeros_bf16() {
        let result = Tensor::zeros_bf16(&[3]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![3]);
        }
    }

    #[test]
    fn test_from_vec() {
        let result = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![4]);
        }
    }

    #[test]
    fn test_from_vec_shape_mismatch() {
        let result = Tensor::from_vec(vec![1.0, 2.0], &[3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_vec_with_dtype_f32() {
        let data = vec![1.0f64, 2.0, 3.0];
        let result = Tensor::from_vec_with_dtype(data, &[3], DType::F32);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![3]);
        }
    }

    #[test]
    fn test_from_vec_with_dtype_f64() {
        let data = vec![1.0f64, 2.0, 3.0];
        let result = Tensor::from_vec_with_dtype(data, &[3], DType::F64);
        assert!(result.is_ok());
    }

    #[test]
    fn test_zeros_dtype_f32() {
        let result = Tensor::zeros_dtype(DType::F32, &[2, 3]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_zeros_dtype_c32() {
        let result = Tensor::zeros_dtype(DType::C32, &[4]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ones_dtype_f32() {
        let result = Tensor::ones_dtype(DType::F32, &[3, 2]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ones_dtype_f64() {
        let result = Tensor::ones_dtype(DType::F64, &[2]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_full_with_shape() {
        let result = Tensor::full_with_shape(&[3, 3], 9.0);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![3, 3]);
        }
    }

    #[test]
    fn test_from_slice_f64() {
        let data = [1.0f64, 2.0, 3.0, 4.0];
        let result = Tensor::from_slice_f64(&data, &[2, 2]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![2, 2]);
        }
    }

    #[test]
    fn test_from_slice_i64() {
        let data = [10i64, 20, 30];
        let result = Tensor::from_slice_i64(&data, &[3]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![3]);
        }
    }

    #[test]
    fn test_from_slice_i32() {
        let data = [1i32, 2, 3, 4, 5, 6];
        let result = Tensor::from_slice_i32(&data, &[2, 3]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![2, 3]);
        }
    }

    #[test]
    fn test_from_scalar_f32() {
        let result = Tensor::from_scalar(std::f32::consts::PI, DType::F32);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.len(), 1);
        }
    }

    #[test]
    fn test_from_scalar_f64() {
        let result = Tensor::from_scalar(std::f32::consts::E, DType::F64);
        assert!(result.is_ok());
    }

    #[test]
    fn test_range_i64() {
        let result = Tensor::range(0, 5, DType::I64);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![5]);
        }
    }

    #[test]
    fn test_range_f32() {
        let result = Tensor::range(1, 4, DType::F32);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![3]);
        }
    }

    #[test]
    fn test_range_invalid() {
        let result = Tensor::range(5, 3, DType::I64);
        assert!(result.is_err());
    }

    #[test]
    fn test_randint() {
        let result = Tensor::randint(0, 10, &[3, 3], DType::I64);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![3, 3]);
        }
    }

    #[test]
    fn test_randint_invalid_range() {
        let result = Tensor::randint(10, 5, &[2], DType::I64);
        assert!(result.is_err());
    }

    #[test]
    fn test_ones_f16() {
        let result = Tensor::ones_f16(&[2, 3]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![2, 3]);
        }
    }

    #[test]
    fn test_ones_bf16() {
        let result = Tensor::ones_bf16(&[4]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![4]);
        }
    }

    #[test]
    fn test_randn_like() {
        if let Ok(input) = Tensor::zeros(&[3, 4]) {
            let result = Tensor::randn_like(&input);
            assert!(result.is_ok());
            if let Ok(t) = result {
                assert_eq!(t.shape(), vec![3, 4]);
            }
        }
    }

    #[test]
    fn test_complex_f16_constructor() {
        let real = vec![1.0, 2.0];
        let imag = vec![3.0, 4.0];
        let result = Tensor::complex_f16(real, imag, &[2]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![2]);
        }
    }

    #[test]
    fn test_complex_bf16_constructor() {
        let real = vec![1.0, 2.0];
        let imag = vec![3.0, 4.0];
        let result = Tensor::complex_bf16(real, imag, &[2]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![2]);
        }
    }

    #[test]
    fn test_lcg_deterministic() {
        let mut rng1 = Lcg::new(42);
        let mut rng2 = Lcg::new(42);
        for _ in 0..10 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_lcg_f32_range() {
        let mut rng = Lcg::new(123);
        for _ in 0..100 {
            let val = rng.next_f32();
            assert!(val >= 0.0);
            assert!(val <= 1.0);
        }
    }

    #[test]
    fn test_3d_tensor_construction() {
        let mut lcg = Lcg::new(99);
        let data: Vec<f32> = (0..24).map(|_| lcg.next_f32()).collect();
        let result = Tensor::from_vec(data, &[2, 3, 4]);
        assert!(result.is_ok());
        if let Ok(t) = result {
            assert_eq!(t.shape(), vec![2, 3, 4]);
            assert_eq!(t.len(), 24);
        }
    }
}
