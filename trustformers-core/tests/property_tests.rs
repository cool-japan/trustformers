//! Property-based tests for trustformers-core
//!
//! These tests use proptest to verify properties that should hold
//! for all valid inputs, helping catch edge cases and ensure robustness.

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use trustformers_core::tensor::Tensor;

// Strategy for generating valid tensor shapes
fn tensor_shape_strategy() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1usize..20, 1..3)
}

// Strategy for generating tensor data
fn tensor_data_strategy(size: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(any::<f32>().prop_filter("not nan", |x| !x.is_nan()), size)
}

// Property: Creating a tensor and accessing its properties should be consistent
proptest! {
    #[test]
    fn test_tensor_creation_properties(shape in tensor_shape_strategy()) {
        let size: usize = shape.iter().product();
        let data = vec![0.0; size];

        let tensor = Tensor::from_vec(data.clone(), &shape).unwrap();

        // Properties that should hold
        let shape_clone = shape.clone();
        prop_assert_eq!(tensor.shape(), shape);
        prop_assert_eq!(tensor.data().unwrap().len(), size);
        prop_assert_eq!(tensor.shape().len(), shape_clone.len());
    }
}

// Property: Tensor creation with random data should preserve basic properties
proptest! {
    #[test]
    fn test_tensor_data_consistency(shape in tensor_shape_strategy()) {
        let size: usize = shape.iter().product();
        let data = tensor_data_strategy(size).new_tree(&mut proptest::test_runner::TestRunner::default()).unwrap().current();

        let tensor = Tensor::from_vec(data.clone(), &shape).unwrap();
        let retrieved_data = tensor.data().unwrap();

        // Data should be preserved
        prop_assert_eq!(retrieved_data.len(), data.len());
        prop_assert_eq!(retrieved_data, data);
    }
}

// Property: Creating tensors with different shapes but same total size
proptest! {
    #[test]
    fn test_different_shapes_same_size(
        rows in 1usize..10,
        cols in 1usize..10,
        value in -10.0f32..10.0
    ) {
        let total_size = rows * cols;
        let data = vec![value; total_size];

        // Create as 1D
        let tensor_1d = Tensor::from_vec(data.clone(), &vec![total_size]).unwrap();

        // Create as 2D
        let tensor_2d = Tensor::from_vec(data.clone(), &vec![rows, cols]).unwrap();

        // Both should have the same data
        prop_assert_eq!(tensor_1d.data().unwrap(), tensor_2d.data().unwrap());
        prop_assert_eq!(tensor_1d.data().unwrap().len(), total_size);
        prop_assert_eq!(tensor_2d.data().unwrap().len(), total_size);
    }
}

// Property: Tensor shapes should be consistent
proptest! {
    #[test]
    fn test_shape_consistency(
        dims in prop::collection::vec(1usize..5, 1..4)
    ) {
        let total_size: usize = dims.iter().product();
        let data = vec![1.0; total_size];

        let tensor = Tensor::from_vec(data, &dims).unwrap();

        // Shape should match what we provided
        prop_assert_eq!(tensor.shape(), dims);
        prop_assert_eq!(tensor.shape().iter().product::<usize>(), total_size);
    }
}

// Property: Empty tensors should fail gracefully
proptest! {
    #[test]
    fn test_invalid_tensor_creation(
        _dummy in 0..1u8 // Just a dummy parameter to make proptest happy
    ) {
        // Empty data should fail
        let empty_data: Vec<f32> = vec![];
        let shape = vec![1, 1];
        prop_assert!(Tensor::from_vec(empty_data, &shape).is_err());

        // Mismatched size and shape should fail
        let data = vec![1.0, 2.0, 3.0];
        let wrong_shape = vec![2, 2]; // Should be size 4, not 3
        prop_assert!(Tensor::from_vec(data, &wrong_shape).is_err());
    }
}

// Property: Tensor cloning and copying should work correctly
proptest! {
    #[test]
    fn test_tensor_cloning(
        shape in tensor_shape_strategy(),
        scale in -100.0f32..100.0
    ) {
        let size: usize = shape.iter().product();
        let data = vec![scale; size];

        let tensor = Tensor::from_vec(data.clone(), &shape).unwrap();
        let cloned_tensor = tensor.clone();

        // Cloned tensor should be identical
        prop_assert_eq!(tensor.shape(), cloned_tensor.shape());
        prop_assert_eq!(tensor.data().unwrap(), cloned_tensor.data().unwrap());
    }
}
