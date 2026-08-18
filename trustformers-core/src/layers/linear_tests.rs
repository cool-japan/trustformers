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

    /// Build a `Linear` with fully deterministic parameters.
    fn deterministic_linear(in_features: usize, out_features: usize, bias: bool) -> Linear {
        let mut linear = Linear::new(in_features, out_features, bias);
        let weight_values: Vec<f32> =
            (0..out_features * in_features).map(|i| (i as f32) * 0.125 - 1.0).collect();
        linear
            .set_weight(
                Tensor::from_vec(weight_values, &[out_features, in_features])
                    .expect("weight shape must be valid"),
            )
            .expect("set_weight");
        if bias {
            let bias_values: Vec<f32> =
                (0..out_features).map(|i| 0.5 - (i as f32) * 0.25).collect();
            linear
                .set_bias(
                    Tensor::from_vec(bias_values, &[out_features])
                        .expect("bias shape must be valid"),
                )
                .expect("set_bias");
        }
        linear
    }

    fn ramp(shape: &[usize]) -> Tensor {
        let count: usize = shape.iter().product();
        Tensor::from_vec(
            (0..count).map(|i| (i as f32) * 0.0625 - 0.5).collect::<Vec<f32>>(),
            shape,
        )
        .expect("input shape must be valid")
    }

    /// `forward_ref` is the borrowing twin of `forward`; the two must agree
    /// bit-for-bit, otherwise replacing `forward(x.clone())` with
    /// `forward_ref(&x)` in the attention layers would change model outputs.
    #[test]
    fn forward_ref_matches_forward_exactly() {
        for bias in [false, true] {
            let linear = deterministic_linear(6, 4, bias);
            for shape in [vec![3usize, 6], vec![2, 3, 6]] {
                let input = ramp(&shape);
                let owned = linear.forward(input.clone()).expect("forward");
                let borrowed = linear.forward_ref(&input).expect("forward_ref");
                assert_eq!(owned.shape(), borrowed.shape());
                assert_eq!(
                    owned.to_vec_f32().expect("f32 data"),
                    borrowed.to_vec_f32().expect("f32 data"),
                    "forward/forward_ref disagree for bias={bias} shape={shape:?}"
                );
            }
        }
    }

    /// `forward_ref` must leave its input untouched — the caller still owns it.
    #[test]
    fn forward_ref_does_not_consume_its_input() {
        let linear = deterministic_linear(4, 3, true);
        let input = ramp(&[2, 4]);
        let before = input.to_vec_f32().expect("f32 data");
        let _ = linear.forward_ref(&input).expect("forward_ref");
        let _ = linear.forward_ref(&input).expect("forward_ref again");
        assert_eq!(before, input.to_vec_f32().expect("f32 data"));
    }

    /// Regression test for the transpose cache.
    ///
    /// `Linear` caches `W^T`. Handing out `&mut Tensor` through `weight_mut`
    /// (which `Model::named_tensors_mut` does) lets a caller replace the weight
    /// without going through `set_weight`. If `weight_mut` did not invalidate the
    /// cache, every later forward pass would silently keep multiplying by the
    /// *old* transpose. This test fails against a naive `&mut self.weight`.
    #[test]
    fn weight_mut_invalidates_the_cached_transpose() {
        let mut layer = deterministic_linear(3, 2, false);
        let input = ramp(&[2, 3]);

        // Populate the transpose cache with the original weight.
        let _ = layer.forward_ref(&input).expect("prime the cache");

        let replacement = Tensor::from_vec(vec![9.0, -3.0, 0.5, 2.0, 7.0, -1.5], &[2, 3])
            .expect("replacement shape");
        *layer.weight_mut() = replacement.clone();

        let mut reference = Linear::new(3, 2, false);
        reference.set_weight(replacement).expect("set_weight");

        assert_eq!(
            layer.forward_ref(&input).expect("mutated forward").to_vec_f32().expect("f32"),
            reference
                .forward_ref(&input)
                .expect("reference forward")
                .to_vec_f32()
                .expect("f32"),
            "weight_mut must invalidate the cached transpose"
        );
    }

    /// Regression test: invalidating the transpose cache must be *recoverable*.
    ///
    /// The cache is cleared through `&mut self` but can only be rebuilt inside
    /// `forward_ref`, which has `&self`. An earlier revision stored it in a plain
    /// `Option` that was only ever refilled by `set_weight`, so any path that
    /// cleared it without going through `set_weight` — notably
    /// `Model::named_tensors_mut`, which hands out `&mut Tensor` for every
    /// parameter — left it empty *forever*, making every later forward pass pay a
    /// full `[out, in]` transpose. That is invisible to a correctness test: the
    /// numbers stay right while the layer silently gets slower.
    ///
    /// So this test asserts the cache state directly. It fails against the
    /// `Option`-based implementation, where the second assertion finds the cache
    /// still empty after a forward pass.
    #[test]
    fn the_transpose_cache_refills_itself_after_invalidation() {
        let mut layer = deterministic_linear(4, 3, false);
        let input = ramp(&[2, 4]);

        // A fresh layer starts cold and warms on first use.
        assert!(
            layer.transposed_weight_cache_is_empty(),
            "a freshly built layer should not have paid for a transpose yet"
        );
        let first = layer.forward_ref(&input).expect("first forward");
        assert!(
            !layer.transposed_weight_cache_is_empty(),
            "the first forward pass must populate the transpose cache"
        );

        // Mutating through the parameter iterator invalidates it...
        let (weight, _) = layer.parameters_mut();
        let replacement = weight.clone();
        *weight = replacement;
        assert!(
            layer.transposed_weight_cache_is_empty(),
            "a write through parameters_mut must drop the stale transpose"
        );

        // ...and the very next forward pass must restore it, not run uncached
        // from here to eternity.
        let second = layer.forward_ref(&input).expect("second forward");
        assert!(
            !layer.transposed_weight_cache_is_empty(),
            "the transpose cache must refill after invalidation, not stay empty forever"
        );

        // The weight was rewritten with an identical value, so results agree.
        assert_eq!(
            first.to_vec_f32().expect("f32"),
            second.to_vec_f32().expect("f32")
        );
    }

    /// `set_weight` must also leave the cache in a self-healing state, and the
    /// rebuilt transpose must correspond to the *new* weight.
    #[test]
    fn set_weight_invalidates_then_the_cache_rebuilds_from_the_new_weight() {
        let mut layer = deterministic_linear(3, 2, false);
        let input = ramp(&[2, 3]);
        let _ = layer.forward_ref(&input).expect("warm the cache");

        let replacement = Tensor::from_vec(vec![2.0, -1.0, 0.5, 4.0, 0.25, -3.0], &[2, 3])
            .expect("replacement shape");
        layer.set_weight(replacement.clone()).expect("set_weight");
        assert!(
            layer.transposed_weight_cache_is_empty(),
            "set_weight must drop the transpose of the previous weight"
        );

        let produced = layer.forward_ref(&input).expect("forward after set_weight");
        assert!(!layer.transposed_weight_cache_is_empty());

        let mut reference = Linear::new(3, 2, false);
        reference.set_weight(replacement).expect("set_weight");
        assert_eq!(
            produced.to_vec_f32().expect("f32"),
            reference.forward_ref(&input).expect("reference").to_vec_f32().expect("f32"),
            "the rebuilt transpose must come from the new weight"
        );
    }

    /// `bias_mut` writes through to the live parameter, and stays `None` for a
    /// layer that was built without bias (a parameter iterator must not be able
    /// to conjure one into existence).
    #[test]
    fn bias_mut_writes_through_and_respects_absence() {
        let mut with_bias = deterministic_linear(2, 2, true);
        let input = ramp(&[1, 2]);
        let before = with_bias.forward_ref(&input).expect("forward").to_vec_f32().expect("f32");

        let new_bias = Tensor::from_vec(vec![10.0, -10.0], &[2]).expect("bias shape");
        *with_bias.bias_mut().expect("bias present") = new_bias;

        let after = with_bias.forward_ref(&input).expect("forward").to_vec_f32().expect("f32");
        assert_ne!(
            before, after,
            "bias_mut must write through to the live bias"
        );

        let mut without_bias = Linear::new(2, 2, false);
        assert!(without_bias.bias_mut().is_none());
    }
}
