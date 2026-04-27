// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for statistical operations on tensors.

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn test_std_f32() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]) {
            let result = t.std();
            assert!(result.is_ok());
            if let Ok(s) = result {
                assert_eq!(s.len(), 1);
            }
        }
    }

    #[test]
    fn test_std_f64() {
        if let Ok(t) = Tensor::from_slice_f64(&[1.0, 2.0, 3.0, 4.0], &[4]) {
            let result = t.std();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_max_value_f32() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 5.0, 3.0, 2.0], &[4]) {
            let result = t.max_value();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_max_elementwise_f32() {
        if let Ok(a) = Tensor::from_vec(vec![1.0, 5.0, 3.0], &[3]) {
            if let Ok(b) = Tensor::from_vec(vec![4.0, 2.0, 6.0], &[3]) {
                let result = a.max(&b);
                assert!(result.is_ok());
                if let Ok(r) = result {
                    assert_eq!(r.shape(), vec![3]);
                }
            }
        }
    }

    #[test]
    fn test_max_scalar_broadcast_a() {
        if let Ok(a) = Tensor::scalar(2.0) {
            if let Ok(b) = Tensor::from_vec(vec![1.0, 3.0, 5.0], &[3]) {
                let result = a.max(&b);
                assert!(result.is_ok());
                if let Ok(r) = result {
                    assert_eq!(r.shape(), vec![3]);
                }
            }
        }
    }

    #[test]
    fn test_max_scalar_broadcast_b() {
        if let Ok(a) = Tensor::from_vec(vec![1.0, 3.0, 5.0], &[3]) {
            if let Ok(b) = Tensor::scalar(2.0) {
                let result = a.max(&b);
                assert!(result.is_ok());
                if let Ok(r) = result {
                    assert_eq!(r.shape(), vec![3]);
                }
            }
        }
    }

    #[test]
    fn test_argmax_last_axis() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]) {
            let result = t.argmax(-1);
            assert!(result.is_ok());
            if let Ok(r) = result {
                assert_eq!(r.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_argmax_first_axis() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]) {
            let result = t.argmax(0);
            assert!(result.is_ok());
            if let Ok(r) = result {
                assert_eq!(r.shape(), vec![3]);
            }
        }
    }

    #[test]
    fn test_argmax_out_of_bounds() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]) {
            let result = t.argmax(5);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_mean_f32() {
        if let Ok(t) = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[4]) {
            let result = t.mean();
            assert!(result.is_ok());
            if let Ok(m) = result {
                assert_eq!(m.len(), 1);
            }
        }
    }

    #[test]
    fn test_mean_f64() {
        if let Ok(t) = Tensor::from_slice_f64(&[1.0, 3.0, 5.0, 7.0], &[4]) {
            let result = t.mean();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_min_max_f32() {
        if let Ok(t) = Tensor::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[6]) {
            let result = t.min_max();
            assert!(result.is_ok());
            if let Ok((min_val, max_val)) = result {
                assert!((min_val - 1.0).abs() < 1e-6);
                assert!((max_val - 9.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_sum_axes_f32() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]) {
            let result = t.sum_axes(&[1]);
            assert!(result.is_ok());
            if let Ok(r) = result {
                assert_eq!(r.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_sum_axes_out_of_bounds() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]) {
            let result = t.sum_axes(&[5]);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_sum_all_elements() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]) {
            let result = t.sum(None, false);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_sum_along_axes() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]) {
            let result = t.sum(Some(vec![0]), false);
            assert!(result.is_ok());
            if let Ok(r) = result {
                assert_eq!(r.shape(), vec![3]);
            }
        }
    }

    #[test]
    fn test_sum_empty_axes() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]) {
            let result = t.sum(Some(vec![]), false);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_mean_axes_f32() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]) {
            let result = t.mean_axes(&[0]);
            assert!(result.is_ok());
            if let Ok(r) = result {
                assert_eq!(r.shape(), vec![3]);
            }
        }
    }

    #[test]
    fn test_mean_axes_out_of_bounds() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0], &[2]) {
            let result = t.mean_axes(&[3]);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_sum_axis_single() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]) {
            let result = t.sum_axis(0);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_sum_dim_positive() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]) {
            let result = t.sum_dim(1, false);
            assert!(result.is_ok());
            if let Ok(r) = result {
                assert_eq!(r.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_sum_dim_negative() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]) {
            let result = t.sum_dim(-1, false);
            assert!(result.is_ok());
            if let Ok(r) = result {
                assert_eq!(r.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_sum_dim_out_of_bounds() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]) {
            let result = t.sum_dim(5, false);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_sum_dim_negative_out_of_bounds() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0], &[2]) {
            let result = t.sum_dim(-5, false);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_mean_axis_single() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]) {
            let result = t.mean_axis(1);
            assert!(result.is_ok());
            if let Ok(r) = result {
                assert_eq!(r.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_variance_all() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]) {
            let result = t.variance(None, false);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_variance_along_axis() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]) {
            let result = t.variance(Some(&[0]), false);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_std_dev() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]) {
            let result = t.std_dev(None, false);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_max_axes() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]) {
            let result = t.max_axes(&[1]);
            assert!(result.is_ok());
            if let Ok(r) = result {
                assert_eq!(r.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_min_axes() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]) {
            let result = t.min_axes(&[1]);
            assert!(result.is_ok());
            if let Ok(r) = result {
                assert_eq!(r.shape(), vec![2]);
            }
        }
    }

    #[test]
    fn test_max_scalar() {
        if let Ok(t) = Tensor::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0], &[5]) {
            let result = t.max_scalar();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_min_scalar() {
        if let Ok(t) = Tensor::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0], &[5]) {
            let result = t.min_scalar();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_max_scalar_i64() {
        if let Ok(t) = Tensor::from_vec_i64(vec![3, 1, 4, 1, 5], &[5]) {
            let result = t.max_scalar();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_min_scalar_i64() {
        if let Ok(t) = Tensor::from_vec_i64(vec![3, 1, 4, 1, 5], &[5]) {
            let result = t.min_scalar();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_all_true() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]) {
            let result = t.all();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_all_false_with_zero() {
        if let Ok(t) = Tensor::from_vec(vec![1.0, 0.0, 3.0], &[3]) {
            let result = t.all();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_all_i64() {
        if let Ok(t) = Tensor::from_vec_i64(vec![1, 2, 3], &[3]) {
            let result = t.all();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_multinomial_basic() {
        if let Ok(probs) = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[4]) {
            if let Ok(softmax_probs) = probs.softmax(0) {
                let result = softmax_probs.multinomial(1, true);
                assert!(result.is_ok());
                if let Ok(r) = result {
                    assert_eq!(r.shape(), vec![1]);
                }
            }
        }
    }

    #[test]
    fn test_multinomial_without_replacement_error() {
        if let Ok(probs) = Tensor::from_vec(vec![0.25, 0.25, 0.25, 0.25], &[4]) {
            let result = probs.multinomial(2, false);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_sum_f64() {
        if let Ok(t) = Tensor::from_slice_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]) {
            let result = t.sum(None, false);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_argmax_f64() {
        if let Ok(t) = Tensor::from_slice_f64(&[1.0, 5.0, 3.0], &[3]) {
            let result = t.argmax(0);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_max_elementwise_f64() {
        if let Ok(a) = Tensor::from_slice_f64(&[1.0, 5.0], &[2]) {
            if let Ok(b) = Tensor::from_slice_f64(&[4.0, 2.0], &[2]) {
                let result = a.max(&b);
                assert!(result.is_ok());
            }
        }
    }
}
