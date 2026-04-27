// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for Linear (fully connected) layer.

#[cfg(test)]
mod tests {
    use crate::device::Device;
    use crate::layers::Linear;
    use crate::tensor::Tensor;
    use crate::traits::Layer;

    #[test]
    fn test_linear_new_with_bias() {
        let linear = Linear::new(10, 20, true);
        assert!(linear.bias().is_some());
        assert_eq!(linear.device(), Device::CPU);
    }

    #[test]
    fn test_linear_new_without_bias() {
        let linear = Linear::new(10, 20, false);
        assert!(linear.bias().is_none());
    }

    #[test]
    fn test_linear_parameter_count_with_bias() {
        let linear = Linear::new(4, 8, true);
        // weight: 8*4=32, bias: 8 => 40
        assert_eq!(linear.parameter_count(), 40);
    }

    #[test]
    fn test_linear_parameter_count_without_bias() {
        let linear = Linear::new(4, 8, false);
        // weight: 8*4=32, no bias => 32
        assert_eq!(linear.parameter_count(), 32);
    }

    #[test]
    fn test_linear_weight_shape() {
        let linear = Linear::new(16, 32, true);
        let weight = linear.weight();
        assert_eq!(weight.shape(), vec![32, 16]);
    }

    #[test]
    fn test_linear_set_weight() {
        let mut linear = Linear::new(4, 8, false);
        if let Ok(new_weight) = Tensor::ones(&[8, 4]) {
            let result = linear.set_weight(new_weight);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_linear_set_bias() {
        let mut linear = Linear::new(4, 8, false);
        assert!(linear.bias().is_none());
        if let Ok(new_bias) = Tensor::zeros(&[8]) {
            let result = linear.set_bias(new_bias);
            assert!(result.is_ok());
            assert!(linear.bias().is_some());
        }
    }

    #[test]
    fn test_linear_forward_2d() {
        let linear = Linear::new(4, 8, true);
        if let Ok(input) = Tensor::randn(&[3, 4]) {
            let result = linear.forward(input);
            assert!(result.is_ok());
            if let Ok(output) = result {
                assert_eq!(output.shape(), vec![3, 8]);
            }
        }
    }

    #[test]
    fn test_linear_forward_3d() {
        let linear = Linear::new(4, 8, true);
        if let Ok(input) = Tensor::randn(&[2, 3, 4]) {
            let result = linear.forward(input);
            assert!(result.is_ok());
            if let Ok(output) = result {
                assert_eq!(output.shape(), vec![2, 3, 8]);
            }
        }
    }

    #[test]
    fn test_linear_forward_without_bias() {
        let linear = Linear::new(4, 8, false);
        if let Ok(input) = Tensor::randn(&[3, 4]) {
            let result = linear.forward(input);
            assert!(result.is_ok());
            if let Ok(output) = result {
                assert_eq!(output.shape(), vec![3, 8]);
            }
        }
    }

    #[test]
    fn test_linear_to_device() {
        let linear = Linear::new(4, 8, true);
        assert_eq!(linear.device(), Device::CPU);
        let linear = linear.to_device(Device::CPU);
        assert_eq!(linear.device(), Device::CPU);
    }

    #[test]
    fn test_linear_new_with_device() {
        let linear = Linear::new_with_device(16, 32, true, Device::CPU);
        assert_eq!(linear.device(), Device::CPU);
        assert_eq!(linear.parameter_count(), 16 * 32 + 32);
    }

    #[test]
    fn test_linear_clone() {
        let linear = Linear::new(4, 8, true);
        let cloned = linear.clone();
        assert_eq!(cloned.parameter_count(), linear.parameter_count());
        assert_eq!(cloned.device(), linear.device());
    }

    #[test]
    fn test_linear_small_forward() {
        // Create a simple 2x2 linear layer with known weights
        let mut linear = Linear::new(2, 2, false);
        if let Ok(w) = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]) {
            if linear.set_weight(w).is_ok() {
                if let Ok(input) = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]) {
                    let result = linear.forward(input);
                    assert!(result.is_ok());
                    if let Ok(output) = result {
                        assert_eq!(output.shape(), vec![2, 2]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_linear_with_zeros_weight() {
        let mut linear = Linear::new(3, 2, false);
        if let Ok(w) = Tensor::zeros(&[2, 3]) {
            if linear.set_weight(w).is_ok() {
                if let Ok(input) = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]) {
                    let result = linear.forward(input);
                    assert!(result.is_ok());
                }
            }
        }
    }

    #[test]
    fn test_linear_large_dimensions() {
        let linear = Linear::new(768, 3072, true);
        assert_eq!(linear.parameter_count(), 768 * 3072 + 3072);
    }

    #[test]
    fn test_linear_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Linear>();
    }

    #[test]
    fn test_linear_forward_single_sample() {
        let linear = Linear::new(4, 2, true);
        if let Ok(input) = Tensor::randn(&[1, 4]) {
            let result = linear.forward(input);
            assert!(result.is_ok());
            if let Ok(output) = result {
                assert_eq!(output.shape(), vec![1, 2]);
            }
        }
    }

    #[test]
    fn test_linear_bias_shape() {
        let linear = Linear::new(10, 5, true);
        if let Some(bias) = linear.bias() {
            assert_eq!(bias.shape(), vec![5]);
        }
    }

    #[test]
    fn test_linear_forward_preserves_batch_dim() {
        let linear = Linear::new(8, 4, true);
        for batch_size in &[1, 2, 4, 8] {
            if let Ok(input) = Tensor::randn(&[*batch_size, 8]) {
                let result = linear.forward(input);
                assert!(result.is_ok());
                if let Ok(output) = result {
                    assert_eq!(output.shape()[0], *batch_size);
                    assert_eq!(output.shape()[1], 4);
                }
            }
        }
    }

    #[test]
    fn test_linear_debug_format() {
        let linear = Linear::new(4, 8, true);
        let debug_str = format!("{:?}", linear);
        assert!(!debug_str.is_empty());
    }
}
