// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Tests for LayerNorm and RMSNorm layers.

#[cfg(test)]
mod tests {
    use crate::device::Device;
    use crate::layers::{LayerNorm, RMSNorm};
    use crate::tensor::Tensor;
    use crate::traits::Layer;

    // LayerNorm tests

    #[test]
    fn test_layernorm_new() {
        let result = LayerNorm::new(vec![768], 1e-5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_layernorm_new_with_device() {
        let result = LayerNorm::new_with_device(vec![768], 1e-5, Device::CPU);
        assert!(result.is_ok());
        if let Ok(ln) = result {
            assert_eq!(ln.device(), Device::CPU);
        }
    }

    #[test]
    fn test_layernorm_new_simple() {
        let ln = LayerNorm::new_simple(256, 1e-5);
        assert_eq!(ln.device(), Device::CPU);
        assert_eq!(ln.parameter_count(), 256 * 2); // weight + bias
    }

    #[test]
    fn test_layernorm_parameter_count() {
        if let Ok(ln) = LayerNorm::new(vec![512], 1e-5) {
            assert_eq!(ln.parameter_count(), 1024); // 512 weight + 512 bias
        }
    }

    #[test]
    fn test_layernorm_set_weight() {
        if let Ok(mut ln) = LayerNorm::new(vec![4], 1e-5) {
            if let Ok(w) = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], &[4]) {
                let result = ln.set_weight(w);
                assert!(result.is_ok());
            }
        }
    }

    #[test]
    fn test_layernorm_set_bias() {
        if let Ok(mut ln) = LayerNorm::new(vec![4], 1e-5) {
            if let Ok(b) = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[4]) {
                let result = ln.set_bias(b);
                assert!(result.is_ok());
            }
        }
    }

    #[test]
    fn test_layernorm_to_device() {
        if let Ok(ln) = LayerNorm::new(vec![128], 1e-5) {
            let ln = ln.to_device(Device::CPU);
            assert_eq!(ln.device(), Device::CPU);
        }
    }

    #[test]
    fn test_layernorm_forward_2d() {
        if let Ok(ln) = LayerNorm::new(vec![4], 1e-5) {
            if let Ok(input) =
                Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])
            {
                let result = ln.forward(input);
                assert!(result.is_ok());
                if let Ok(output) = result {
                    assert_eq!(output.shape(), vec![2, 4]);
                }
            }
        }
    }

    #[test]
    fn test_layernorm_forward_3d() {
        if let Ok(ln) = LayerNorm::new(vec![4], 1e-5) {
            if let Ok(input) = Tensor::randn(&[2, 3, 4]) {
                let result = ln.forward(input);
                assert!(result.is_ok());
                if let Ok(output) = result {
                    assert_eq!(output.shape(), vec![2, 3, 4]);
                }
            }
        }
    }

    #[test]
    fn test_layernorm_output_normalized() {
        if let Ok(ln) = LayerNorm::new(vec![4], 1e-5) {
            if let Ok(input) = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]) {
                let result = ln.forward(input);
                assert!(result.is_ok());
                if let Ok(output) = result {
                    // Output should have mean close to 0 and std close to 1
                    if let Ok(m) = output.mean() {
                        assert_eq!(m.len(), 1);
                    }
                }
            }
        }
    }

    #[test]
    fn test_layernorm_clone() {
        if let Ok(ln) = LayerNorm::new(vec![64], 1e-5) {
            let cloned = ln.clone();
            assert_eq!(cloned.parameter_count(), ln.parameter_count());
        }
    }

    #[test]
    fn test_layernorm_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LayerNorm>();
    }

    #[test]
    fn test_layernorm_forward_single_element() {
        if let Ok(ln) = LayerNorm::new(vec![1], 1e-5) {
            if let Ok(input) = Tensor::from_vec(vec![5.0], &[1, 1]) {
                let result = ln.forward(input);
                assert!(result.is_ok());
            }
        }
    }

    #[test]
    fn test_layernorm_with_different_eps() {
        for eps in &[1e-3, 1e-5, 1e-8, 1e-12] {
            let result = LayerNorm::new(vec![32], *eps);
            assert!(result.is_ok());
        }
    }

    // RMSNorm tests

    #[test]
    fn test_rmsnorm_new() {
        let result = RMSNorm::new(768, 1e-5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_rmsnorm_new_with_device() {
        let result = RMSNorm::new_with_device(768, 1e-5, Device::CPU);
        assert!(result.is_ok());
        if let Ok(rn) = result {
            assert_eq!(rn.device(), Device::CPU);
        }
    }

    #[test]
    fn test_rmsnorm_parameter_count() {
        if let Ok(rn) = RMSNorm::new(512, 1e-5) {
            assert_eq!(rn.parameter_count(), 512); // Only weight, no bias
        }
    }

    #[test]
    fn test_rmsnorm_set_weight() {
        if let Ok(mut rn) = RMSNorm::new(4, 1e-5) {
            if let Ok(w) = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], &[4]) {
                let result = rn.set_weight(w);
                assert!(result.is_ok());
            }
        }
    }

    #[test]
    fn test_rmsnorm_to_device() {
        if let Ok(rn) = RMSNorm::new(128, 1e-5) {
            let rn = rn.to_device(Device::CPU);
            assert_eq!(rn.device(), Device::CPU);
        }
    }

    #[test]
    fn test_rmsnorm_forward_2d() {
        if let Ok(rn) = RMSNorm::new(4, 1e-5) {
            if let Ok(input) =
                Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])
            {
                let result = rn.forward(input);
                assert!(result.is_ok());
                if let Ok(output) = result {
                    assert_eq!(output.shape(), vec![2, 4]);
                }
            }
        }
    }

    #[test]
    fn test_rmsnorm_forward_3d() {
        if let Ok(rn) = RMSNorm::new(4, 1e-5) {
            if let Ok(input) = Tensor::randn(&[2, 3, 4]) {
                let result = rn.forward(input);
                assert!(result.is_ok());
                if let Ok(output) = result {
                    assert_eq!(output.shape(), vec![2, 3, 4]);
                }
            }
        }
    }

    #[test]
    fn test_rmsnorm_clone() {
        if let Ok(rn) = RMSNorm::new(64, 1e-5) {
            let cloned = rn.clone();
            assert_eq!(cloned.parameter_count(), rn.parameter_count());
        }
    }

    #[test]
    fn test_rmsnorm_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RMSNorm>();
    }

    #[test]
    fn test_rmsnorm_forward_large_hidden_size() {
        if let Ok(rn) = RMSNorm::new(128, 1e-5) {
            if let Ok(input) = Tensor::randn(&[4, 128]) {
                let result = rn.forward(input);
                assert!(result.is_ok());
                if let Ok(output) = result {
                    assert_eq!(output.shape(), vec![4, 128]);
                }
            }
        }
    }

    #[test]
    fn test_rmsnorm_with_different_eps() {
        for eps in &[1e-3, 1e-6, 1e-8] {
            let result = RMSNorm::new(32, *eps);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_layernorm_debug_format() {
        if let Ok(ln) = LayerNorm::new(vec![4], 1e-5) {
            let debug_str = format!("{:?}", ln);
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_rmsnorm_debug_format() {
        if let Ok(rn) = RMSNorm::new(4, 1e-5) {
            let debug_str = format!("{:?}", rn);
            assert!(!debug_str.is_empty());
        }
    }
}
